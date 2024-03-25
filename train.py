from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import time
import os
import yaml
from matplotlib import pyplot as plt
import glob
from collections import OrderedDict
from tqdm import tqdm
import torch.distributed as dist
from lightPred.timeDetrLoss import cxcy_to_cxcywh

# import NamedTuple
# import List

# class FitResult(NamedTuple):
#     """
#     Represents the result of fitting a model for multiple epochs given a
#     training and test (or validation) set.
#     The losses and the accuracies are per epoch.
#     """

#     num_epochs: int
#     train_loss: List[float]
#     test_loss: List[float]


def count_occurence(x,y):
  coord_counts = {}
  for i in range(len(x)):
      coord = (x[i], y[i])
      if coord in coord_counts:
          coord_counts[coord] += 1
      else:
          coord_counts[coord] = 1


class Trainer(object):
    """
    A class that encapsulates the training loop for a PyTorch model.
    """
    def __init__(self, model, optimizer, criterion, train_dataloader, device, num_classes=2, scheduler=None, val_dataloader=None,
                 optim_params=None, max_iter=np.inf, net_params=None, scaler=None, grad_clip=False,
                   exp_num=None, log_path=None, exp_name=None, plot_every=None, cos_inc=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.grad_clip = grad_clip
        self.cos_inc = cos_inc
        self.num_classes = num_classes
        self.scheduler = scheduler
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.max_iter = max_iter
        self.device = device
        self.optim_params = optim_params
        self.net_params = net_params
        self.exp_num = exp_num
        self.exp_name = exp_name
        self.log_path = log_path
        self.best_state_dict = None
        self.plot_every = plot_every
        self.logger = None
        if log_path is not None:
            self.logger =SummaryWriter(f'{self.log_path}/exp{self.exp_num}')
            # print(f"logger path: {self.log_path}/exp{self.exp_num}")
            if not os.path.exists(f'{self.log_path}/exp{self.exp_num}'):
                os.makedirs(f'{self.log_path}/exp{self.exp_num}')
            with open(f'{self.log_path}/exp{exp_num}/net_params.yml', 'w') as outfile:
                yaml.dump(self.net_params, outfile, default_flow_style=False)
            with open(f'{self.log_path}/exp{exp_num}/optim_params.yml', 'w') as outfile:
                    yaml.dump(self.optim_params, outfile, default_flow_style=False)

        print("logger is: ", self.logger)

    def plot_pred_vs_true(self, output, target, epoch, data='train'):
        plt.scatter(target[:, 1], output[:, 1])
        plt.plot(target, 0.9*target, color='red')
        plt.plot(target, 1.1*target, color='red')
        plt.xlim(0, max(target[:,1]) + 5)
        plt.ylim(0, max(target[:,1]) + 5)
        plt.title(f"{self.exp_name}")
        plt.xlabel("true Period")
        plt.ylabel("predicted Period")
        plt.savefig(f"{self.log_path}/exp{self.exp_num}/Period_{data}_epoch{epoch}.png")
        plt.clf()

        plt.scatter(target[:, 0], output[:, 0])
        plt.title(f"{self.exp_name} Inclination")
        plt.xlabel("true Inclination")
        plt.ylabel("predicted Inclination")
        plt.savefig(f"{self.log_path}/exp{self.exp_num}/Inc_{data}_epoch{epoch}.png")

    def plot_pred_vs_true_cls(self, output, target, epoch, data='train'):
        coords, counts = count_occurence(target, output)
        plt.scatter(coords[:, 0], coords[:, 1], c=counts, cmap='viridis')
        plt.title(f"{self.exp_name} Inclination")
        plt.xlabel("true")
        plt.ylabel("predicted")
        plt.colorbar(label='points frequency')
        plt.savefig(f"{self.log_path}/exp{self.exp_num}/Inc_{data}_epoch{epoch}.png")
        plt.close()

    def fit(self, num_epochs, device,  early_stopping=None, only_p=False, best='loss', conf=False):
        """
        Fits the model for the given number of epochs.
        """
        min_loss = np.inf
        best_acc = 0
        train_loss, val_loss,  = [], []
        train_acc, val_acc = [], []
        self.optim_params['lr_history'] = []
        epochs_without_improvement = 0

        print(f"Starting training for {num_epochs} epochs with parameters: {self.optim_params}, {self.net_params}")
        for epoch in range(num_epochs):
            start_time = time.time()
            plot = (self.plot_every is not None) and (epoch % self.plot_every == 0)
            t_loss, t_acc = self.train_epoch(device, epoch=epoch, only_p=only_p, plot=plot, conf=conf)
            t_loss_mean = np.mean(t_loss)
            train_loss.extend(t_loss)
            global_train_accuracy = torch.tensor(t_acc).cuda()  # Convert accuracy to a tensor on the GPU
            torch.distributed.reduce(global_train_accuracy, dst=0, op=torch.distributed.ReduceOp.SUM)
            global_train_loss = torch.tensor(t_loss_mean).cuda()  # Convert loss to a tensor on the GPU
            torch.distributed.reduce(global_train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)  
            global_train_loss /= torch.distributed.get_world_size()
            if torch.distributed.get_rank() == 0:  # Only perform this on the master GPU
                # global_accuracy /= torch.distributed.get_world_size()
                train_acc.append(global_train_accuracy.mean().item())
                self.logger.add_scalar('train_acc', global_train_accuracy.mean().item(), epoch)


            v_loss, v_acc = self.eval_epoch(device, epoch=epoch, only_p=only_p, plot=plot, conf=conf)
            v_loss_mean = np.mean(v_loss)
            val_loss.extend(v_loss)
            global_val_accuracy = torch.tensor(v_acc).cuda() 
            torch.distributed.reduce(global_val_accuracy, dst=0, op=torch.distributed.ReduceOp.SUM) 
            global_val_loss = torch.tensor(v_loss_mean).cuda()  # Convert loss to a tensor on the GPU
            torch.distributed.reduce(global_val_loss, dst=0, op=torch.distributed.ReduceOp.SUM) 
            global_val_loss /= torch.distributed.get_world_size()
            if torch.distributed.get_rank() == 0:  # Only perform this on the master GPU
                # global_accuracy /= torch.distributed.get_world_size()
                # print("global_val_accuracy: ", global_val_accuracy)
                val_acc.append(global_val_accuracy.mean().item())
                self.logger.add_scalar('validation_acc', global_val_accuracy.mean().item(), epoch)

                if self.scheduler is not None:
                    self.scheduler.step(global_val_loss)
                criterion = min_loss if best == 'loss' else best_acc
                mult = 1 if best == 'loss' else -1
                objective = global_val_loss if best == 'loss' else global_val_accuracy.mean()
                if mult*objective < mult*criterion:
                    print("saving model...")
                    if best == 'loss':
                        min_loss = global_val_loss
                    else:
                        best_acc = global_val_accuracy.mean()
                    torch.save(self.model.state_dict(), f'{self.log_path}/exp{self.exp_num}/{self.exp_name}.pth')
                    self.best_state_dict = self.model.state_dict()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement == early_stopping:
                        print('early stopping!', flush=True)
                        break

                print(f'Epoch {epoch}: Train Loss: {global_train_loss:.6f}, Val Loss: {global_val_loss:.6f}, Train Acc: {global_train_accuracy.round(decimals=4).tolist()}, Val Acc: {global_val_accuracy.round(decimals=4).tolist()}, Time: {time.time() - start_time:.2f}s')

                if epoch % 10 == 0:
                    print(os.system('nvidia-smi'))

        self.optim_params['lr'] = self.optimizer.param_groups[0]['lr']
        self.optim_params['lr_history'].append(self.optim_params['lr'])
        with open(f'{self.log_path}/exp{self.exp_num}/optim_params.yml', 'w') as outfile:
            yaml.dump(self.optim_params, outfile, default_flow_style=False)

        if self.logger is not None:
            self.logger.add_scalar('time', time.time() - start_time, epoch)
            self.logger.close()
        return {"num_epochs":num_epochs, "train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc}
    
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.train_dl)
        for i, (x, y,_,info) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x.float())
            # print("y_pred: ", y_pred.shape, "y: ", y.shape)

            if len(y.shape) == 1:
                y = y.unsqueeze(1)
            if conf:
                y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:].abs()
                conf_y = torch.abs(y - y_pred) 
            # print(f"inclination range: {y_pred[:,1].min()} - {y_pred[:,1].max()}")
            # print(f"pred y: {y_pred[:10,:]}")

            # print("y_pred: ", y_pred.shape, "y: ", y.shape)
            loss = self.criterion(y_pred, y) if not only_p else self.criterion(y_pred, y[:, 0])
            if conf:
                loss += self.criterion(conf_pred, conf_y)
            if self.logger is not None:
                self.logger.add_scalar('train_loss', loss.item(), i + epoch*len(self.train_dl))
                self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i + epoch*len(self.train_dl))
            # print("loss: ", loss, "y_pred: ", y_pred, "y: ", y)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            train_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
            all_acc = (diff < (y/10)).sum(0)
            pbar.set_description(f"train_acc: {all_acc}, train_loss:  {loss.item()}") 
            all_accs = all_accs + all_acc
            # self.train_dl.dataset.step += 1 # for noise addition
            # mean_acc = (diff[:,0]/(y[:,0])).sum().item()
            # train_acc2 += (diff[:,self.num_classes-1] < y[:,self.num_classes-1]/10).sum().item()
            if i > self.max_iter:
                break
        print("number of train_accs: ", train_acc)
        return train_loss, all_accs/len(self.train_dl.dataset)

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.val_dl)
        for i,(x, y,_,_) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            if len(y.shape) == 1:
                y = y.unsqueeze(1)
            with torch.no_grad():
                y_pred = self.model(x)
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:].abs()
                    conf_y = torch.abs(y - y_pred)
                else:
                    conf_pred = torch.ones_like(y_pred)
            loss = self.criterion(y_pred, y) if not only_p else self.criterion(y_pred, y[:, 0])
            if conf:
                loss += self.criterion(conf_pred, conf_y)
            if self.logger is not None:
                self.logger.add_scalar('validation_loss', loss.item(), i + epoch*len(self.train_dl))
            val_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            # print("diff: ", diff.shape, "y: ", y.shape)
            val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()  
            all_acc = (diff < (y/10)).sum(0)
            pbar.set_description(f"val_acc: {all_acc}, val_loss:  {loss.item()}")
            all_accs = all_accs + all_acc
        return val_loss, all_accs/len(self.val_dl.dataset)

    def load_best_model(self, to_ddp=True, from_ddp=True):
        data_dir = f'{self.log_path}/exp{self.exp_num}'
        # data_dir = f'{self.log_path}/exp29' # for debugging

        state_dict_files = glob.glob(data_dir + '/*.pth')
        print("loading model from ", state_dict_files[-1])
        
        state_dict = torch.load(state_dict_files[-1]) if to_ddp else torch.load(state_dict_files[0],map_location=device)
    
        if from_ddp:
            print("loading distributed model")
            # Remove "module." from keys
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    while key.startswith('module.'):
                        key = key[7:]
                new_state_dict[key] = value
            state_dict = new_state_dict
        # print("state_dict: ", state_dict.keys())
        # print("model: ", self.model.state_dict().keys())

        self.model.load_state_dict(state_dict, strict=False)

    def predict(self, test_dataloader, device, conf=True, only_p=False, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model(from_ddp=False)
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        confs = np.zeros((0, self.num_classes))
        tot_kic = []
        tot_teff = []
        for i,(x, y,_,info) in enumerate(test_dataloader):
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
                # print(i, " y_pred: ", y_pred.shape, "y: ", y.shape)
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            if y.shape[1] == self.num_classes:
                targets = np.concatenate((targets, y.cpu().numpy()))
            if conf:
                confs = np.concatenate((confs, conf_pred.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, confs
    
class DoubleInputTrainer(Trainer):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.train_dl)
        for i, (x,y, _,_) in enumerate(pbar):
            # print("x: ", x.shape, "y: ", y.shape)
            x1, x2 = x[:, 0, :], x[:, 1, :]
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x1.float(), x2.float())
            if conf:
                y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
                conf_y = torch.abs(y - y_pred) 
                
            # y_std = torch.std(y.squeeze(), dim=1)
            # y_pred_std = torch.std(y_pred.squeeze(), dim=1)
            if self.cos_inc:
                inc_idx = 0
                y_pred[:, inc_idx] = torch.cos(y_pred[:, inc_idx]*np.pi/2)
                y[:, inc_idx] = torch.cos(y[:, inc_idx]*np.pi/2)
            loss = self.criterion(y_pred, y)
            if conf:
                loss += self.criterion(conf_pred, conf_y)
                
            # loss_std = self.criterion(y_pred_std, y_std)
            # loss = loss + 0.5 * loss_std
            
            # print("loss: ", loss, "y_pred: ", y_pred, "y: ", y)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            # train_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
            all_acc = (diff < (y/10)).sum(0)
            all_accs = all_accs + all_acc
            pbar.set_description(f"train_acc: {all_acc}, train_loss:  {loss.item()}")
            if i > self.max_iter:
                break
        return train_loss, all_accs/len(self.train_dl.dataset)
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.val_dl)
        for i, (x,y, _,_) in enumerate(pbar):
            x1, x2 = x[:, 0, :], x[:, 1, :]
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            with torch.no_grad():
                y_pred = self.model(x1.float().squeeze(), x2.float().squeeze())
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
                    conf_y = torch.abs(y - y_pred)
            # y_std = torch.std(y.squeeze(), dim=1)
            # y_pred_std = torch.std(y_pred.squeeze(), dim=1)
            if self.cos_inc:
                inc_idx = 0
                y_pred[:, inc_idx] = torch.cos(y_pred[:, inc_idx]*np.pi/2)
                y[:, inc_idx] = torch.cos(y[:, inc_idx]*np.pi/2)
            loss = self.criterion(y_pred, y) 
            if conf:
                loss += self.criterion(conf_pred, conf_y)
                
            # loss_std = self.criterion(y_pred_std, y_std)
            # loss = loss + 0.5 * loss_std
            # if self.logger is not None:
            #     self.logger.add_scalar('validation_loss', loss.item(), i + epoch*len(self.train_dl))
            val_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            # print("diff: ", diff.shape, "y: ", y.shape)
            # val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
            all_acc = (diff < (y/10)).sum(0)
            pbar.set_description(f"val_acc: {all_acc}, val_loss:  {loss.item()}")
            all_accs = all_accs + all_acc  
        return val_loss, all_accs/len(self.val_dl.dataset)

    def predict(self, test_dataloader, device, conf=True, only_p=False, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model(from_ddp=False)
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        confs = np.zeros((0, self.num_classes))
        tot_kic = []
        tot_teff = []
        for i,(x,y,_,_) in enumerate(test_dataloader):
            x1, x2 = x[:, 0, :], x[:, 1, :]
            x1 = x1.to(device)
            x2 = x2.to(device)
            with torch.no_grad():
                y_pred = self.model(x1.float(), x2.float())
                # print(i, " y_pred: ", y_pred.shape, "y: ", y.shape)
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            if y.shape[1] == self.num_classes:
                targets = np.concatenate((targets, y.cpu().numpy()))
            if conf:
                confs = np.concatenate((confs, conf_pred.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, confs
    

class KeplerTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        pbar = tqdm(self.train_dl)
        for i, (x, y, _,info) in enumerate(pbar):
            y = y.to(device)
            self.optimizer.zero_grad()
            if x.shape[1] == 2:
                x1, x2 = x[:, 0, :], x[:, 1, :]
                x1 = x1.to(device)
                x2 = x2.to(device)
                y_pred = self.model(x1.float(), x2.float())
            else:
                x = x.to(device)
                y_pred = self.model(x.float())
            if conf:
                y_pred, conf_pred = y_pred[:, :self.num_classes].squeeze(), y_pred[:, self.num_classes:].squeeze()
                conf_y = torch.abs(y - y_pred) 
            loss = self.criterion(y_pred, y) 
            if conf:
                loss += self.criterion(conf_pred, conf_y)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            train_acc += (diff < (y/10)).sum().item() 
            pbar.set_description(f"train_acc: {train_acc}, train_loss:  {loss.item()}")
        print("number of train_accs: ", train_acc)
        return train_loss, train_acc/len(self.train_dl.dataset)

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        for i, (x, y, _,info) in enumerate(self.val_dl):
            x = x.to(device)
            y = y.to(device)
            self.optimizer.zero_grad()
            if x.shape[1] == 2:
                x1, x2 = x[:, 0, :], x[:, 1, :]
                x1 = x1.to(device)
                x2 = x2.to(device)
                y_pred = self.model(x1.float(), x2.float())
            else:
                x = x.to(device)
                y_pred = self.model(x.float())
            # y_val = y['Period'] if only_p else y['i']
            # pred_idx = 1 if only_p else 0
            if conf:
                y_pred, conf_pred = y_pred[:, :self.num_classes].squeeze(), y_pred[:, self.num_classes:].squeeze()
                conf_y = torch.abs(y - y_pred) 
            loss = self.criterion(y_pred, y) 
            if conf:
                loss += self.criterion(conf_pred, conf_y)
            val_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            val_acc += (diff < (y/10)).sum().item() 
        print("number of val_acc: ", val_acc)
        return val_loss, val_acc/len(self.val_dl.dataset)

    def predict(self, test_dataloader, device, conf=True, only_p=False, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model()
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0))
        confs = np.zeros((0, self.num_classes))
        tot_kic = np.zeros((0))
        tot_teff = np.zeros((0))
        tot_r = np.zeros((0))
        tot_g = np.zeros((0))
        tot_qs = []

        print("len test dataset: ", len(test_dataloader.dataset))
        for i,(x, y,_,info) in enumerate(test_dataloader):
            if i > self.max_iter:
                break
            # print("batch: ", i, "x: ", x.shape, "y: ", y.shape, flush=True)
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                if x.shape[1] == 2:
                    x1, x2 = x[:, 0, :], x[:, 1, :]
                    x1 = x1.to(device)
                    x2 = x2.to(device)
                    y_pred = self.model(x1.float(), x2.float())
                else:
                    x = x.to(device)
                    y_pred = self.model(x.float())
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]

            if dist.is_available() and dist.is_initialized():
                # Gather predictions from all GPUs
                # Ensure y_pred is contiguous
                y_pred_contiguous = y_pred.contiguous()

                # Gather predictions from all GPUs
                all_preds = [torch.zeros_like(y_pred_contiguous) for _ in range(dist.get_world_size())]
                dist.all_gather(all_preds, y_pred_contiguous)
                y_pred = torch.cat(all_preds, dim=0)
            preds = np.concatenate((preds, y_pred.squeeze().cpu().numpy()))
            print("y_pred: ", y_pred.shape, "tot pred: ", preds.shape)
            if isinstance(y, dict):
                y_val = y['Period'] if only_p else y['i']
            else:
                y_val = y
            # targets = np.concatenate((targets, y_val.squeeze().cpu().numpy()))
            
            if conf:
                confs = np.concatenate((confs, conf_pred.cpu().numpy()))
            kic = np.array([d['KID'] for d in info])
            teff = np.array([d['Teff'] for d in info])
            r = np.array([d['R'] for d in info])
            g = np.array([d['logg'] for d in info])
            qs = [d['qs'] for d in info]
            tot_kic = np.concatenate((tot_kic, kic))
            tot_teff = np.concatenate((tot_teff, teff))
            tot_r = np.concatenate((tot_r, r))
            tot_g = np.concatenate((tot_g, g))
            tot_qs.extend(qs)
        return preds, targets, confs, tot_kic, tot_teff, tot_r, tot_g, tot_qs


class Trainer2(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.train_dl)
        for i, (x, y,_,_) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            y_pred = self.model(x.float())
            if len(y.shape) == 1:
                y = y.unsqueeze(1)
            if conf:
                y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:].abs()
                conf_y = torch.abs(y - y_pred) 
            # print(f"inclination range: {y_pred[:,1].min()} - {y_pred[:,1].max()}")
            # print(f"true y: {y[:10,:]}")
            # print(f"pred y: {y_pred[:10,:]}")

            # print("y_pred: ", y_pred.shape, "y: ", y.shape, conf)
            loss = self.criterion(y_pred, y) if not only_p else self.criterion(y_pred, y[:, 0])
            if conf:
                loss += self.criterion(conf_pred, conf_y)
            if self.logger is not None:
                self.logger.add_scalar('train_loss', loss.item(), i + epoch*len(self.train_dl))
                self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i + epoch*len(self.train_dl))
            # print("loss: ", loss, "y_pred: ", y_pred, "y: ", y)
            
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()

            train_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            train_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
            all_acc = (diff < (y/10)).sum(0)
            pbar.set_description(f"train_acc: {all_acc}, train_loss:  {loss.item()}") 
            all_accs = all_accs + all_acc
            
            # mean_acc = (diff[:,0]/(y[:,0])).sum().item()
            # train_acc2 += (diff[:,self.num_classes-1] < y[:,self.num_classes-1]/10).sum().item()
        print("number of train_accs: ", train_acc)
        return train_loss, all_accs/len(self.train_dl.dataset)

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.val_dl)
        for i,(x, y,_,_) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            if len(y.shape) == 1:
                y = y.unsqueeze(1)
            with torch.no_grad():
                y_pred = self.model(x)
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:].abs()
                    conf_y = torch.abs(y - y_pred)
                else:
                    conf_pred = torch.ones_like(y_pred)
            loss = self.criterion(y_pred, y) if not only_p else self.criterion(y_pred, y[:, 0])
            if conf:
                loss += self.criterion(conf_pred, conf_y)
            if self.logger is not None:
                self.logger.add_scalar('validation_loss', loss.item(), i + epoch*len(self.train_dl))
            val_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            # print("diff: ", diff.shape, "y: ", y.shape)
            val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()  
            all_acc = (diff < (y/10)).sum(0)
            pbar.set_description(f"val_acc: {all_acc}, val_loss:  {loss.item()}")
            all_accs = all_accs + all_acc
        return val_loss, all_accs/len(self.val_dl.dataset)

    def predict(self, test_dataloader, device, conf=True, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model()
        self.model.eval()
        preds = np.zeros((0, self.num_classes, len(self.quantiles)))
        targets = np.zeros((0, self.num_classes))
        tot_kic = []
        tot_teff = []
        for i,(x, y,_,_) in enumerate(test_dataloader):
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        return preds, targets


class SpotsTrainer(Trainer):
    def __init__(self, spots_loss, num_classes=2, eta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.spots_loss = spots_loss
        self.eta = eta
    
    def get_spot_dict(self, spot_arr):
        spot_arr = spot_arr*180/np.pi
        bs, _,_ = spot_arr.shape
        idx = [spot_arr[b,0,:] != 0 for b in range(bs)]
        res = []
        for i in range(bs):
            spot_dict = {'boxes': cxcy_to_cxcywh(spot_arr[i, :, idx[i]], 1, 1).transpose(0,1).to(spot_arr.device),
                        'labels': torch.ones((spot_arr[i, :, idx[i]].shape[-1]), device=spot_arr.device).long()}
            res.append(spot_dict)
        return res

    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.train_dl)
        for i, (x,y, _,_) in enumerate(pbar):
            tic = time.time()
            x, spots_arr = x[:, :-2, :], x[:, -2:, :]
            print("min spots lat: ", spots_arr[:, 0,:].min(), "max spots lat: ", spots_arr[:,0,:].max())
            x = x.to(device)
            y = y.to(device)
            spots_arr = spots_arr.to(device)
            tgt_spots = self.get_spot_dict(spots_arr)
            shapes = [(tgt_spots[i]['boxes'].shape, tgt_spots[i]['labels'].shape) for i in range(len(tgt_spots))]
            if x.shape[1] == 2:
                x1, x2 = x[:, 0, :], x[:, 1, :]
                out_spots, y_pred = self.model(x1, x2)
            else:
                out_spots, y_pred = self.model(x)
            t1 = time.time()
            src_shapes = (out_spots['pred_boxes'].shape, out_spots['pred_logits'].shape)
            if conf:
                y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
                conf_y = torch.abs(y - y_pred)
            att_loss_val = self.criterion(y_pred, y)
            if conf:
                att_loss_val += self.criterion(conf_pred, conf_y)
            t2 = time.time()
            spots_loss_dict = self.spots_loss(out_spots, tgt_spots)
            weight_dict = self.spots_loss.weight_dict
            spot_loss_val = sum(spots_loss_dict[k] * weight_dict[k] for k in spots_loss_dict.keys() if k in weight_dict)
            loss = att_loss_val
            t3 = time.time()
            loss = self.eta*spot_loss_val + (1-self.eta)*att_loss_val
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            # print(spots_loss_dict['class_error'])
            spots_acc = (100 - spots_loss_dict['class_error'])
            diff = torch.abs(y_pred - y)
            # train_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
            att_acc = (diff < (y/10)).sum(0)
            all_accs = all_accs + (att_acc + spots_acc)/2
            pbar.set_description(f"train_acc: {att_acc, spots_acc}, train_loss:  {loss.item():.2f}, "\
             f"spot_loss: {self.eta*spot_loss_val:.2f}, att_loss: {(1-self.eta)*att_loss_val:.2f}")
            if i > self.max_iter:
                break
            toc = time.time()
            print(f"train time: {toc-tic}, forward time: {t1-tic}, loss time: {t2-t1}, spot loss time: {t3-t2}, step time: {toc-t3}")
        return train_loss, all_accs/len(self.train_dl.dataset)

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        all_accs = torch.zeros(self.num_classes, device=device)
        pbar = tqdm(self.val_dl)
        with torch.no_grad():
            for i, (x,y,_,_) in enumerate(pbar):
                x, spots_arr = x[:, :-2, :], x[:, -2:, :]
                x = x.to(device)
                y = y.to(device)
                spots_arr = spots_arr.to(device)
                tgt_spots = self.get_spot_dict(spots_arr)
                if x.shape[1] == 2:
                    x1, x2 = x[:, 0, :], x[:, 1, :]
                    out_spots, y_pred = self.model(x1, x2)
                else:
                    out_spots, y_pred = self.model(x)
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
                    conf_y = torch.abs(y - y_pred)
                att_loss_val = self.criterion(y_pred, y)
                if conf:
                    att_loss_val += self.criterion(conf_pred, conf_y)
                shapes_tgt = [(tgt_spots[i]['boxes'].shape, tgt_spots[i]['labels'].shape) for i in range(len(tgt_spots))]
                shapes_src = (out_spots['pred_boxes'].shape, out_spots['pred_logits'].shape)
                print(f"tgt shapes: {shapes_tgt}, src shapes: {shapes_src}")
                spots_loss_dict = self.spots_loss(out_spots, tgt_spots)
                weight_dict = self.spots_loss.weight_dict
                spot_loss_val = sum(spots_loss_dict[k] * weight_dict[k] for k in spots_loss_dict.keys() if k in weight_dict)
                loss = att_loss_val

                loss = self.eta*spot_loss_val + (1-self.eta)*att_loss_val
                val_loss.append(loss.item())
                spots_acc = (100 - spots_loss_dict['class_error'])
                diff = torch.abs(y_pred - y)
                # val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
                att_acc = (diff < (y/10)).sum(0)
                all_accs = all_accs + (att_acc + spots_acc)/2
                pbar.set_description(f"val_acc: {att_acc, spots_acc}, val_loss:  {loss.item()}")
        return val_loss, all_accs/len(self.val_dl.dataset)
    
    def predict(self, test_dataloader, device, conf=True, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model()
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        confs = np.zeros((0, self.num_classes))
        spots_bbox = np.zeros((0, 4))
        spots_labels = np.zeros((0))
        spots_target_bbox = np.zeros((0, 4))
        for i,(x, y,_,info) in enumerate(test_dataloader):
            x, spots_arr = x[:, :-2, :], x[:, -2:, :]
            x = x.to(device)
            tgt_spots = self.get_spot_dict(spots_arr)
            with torch.no_grad():
                if x.shape[1] == 2:
                    x1, x2 = x[:, 0, :], x[:, 1, :]
                    out_spots, y_pred = self.model(x1, x2)
                else:
                    out_spots, y_pred = self.model(x.unsqueeze(-1))
                if conf:
                    y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
                    conf_y = torch.abs(y - y_pred)
                spots_loss_dict = self.spots_loss(out_spots, tgt_spots)
                weight_dict = self.spots_loss.weight_dict
                spot_loss_val = sum(spots_loss_dict[k] * weight_dict[k] for k in spots_loss_dict.keys() if k in weight_dict)
                # print(out_att.shape, y[:,0].shape)
                att_loss_val = self.criterion(y_pred, y)
                if conf:
                    att_loss_val += self.criterion(conf_pred, conf_y)
                loss = self.eta*spot_loss_val + (1-self.eta)*att_loss_val
            src_boxes, target_boxes = self.spots_loss.post_process(out_spots, tgt_spots)
            spots_bbox = np.concatenate((spots_bbox, src_boxes.cpu().numpy()))
            spots_target_bbox = np.concatenate((spots_target_bbox, target_boxes.cpu().numpy()))
            labels = out_spots['pred_labels'].argmax().cpu().numpy()
            spots_labels = np.concatenate((spots_labels,labels))
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            if y.shape[1] == self.num_classes:
                targets = np.concatenate((targets, y.cpu().numpy()))
            if conf:
                confs = np.concatenate((confs, conf_pred.cpu().numpy()))
            
        return preds, targets, confs, spots_bbox, spots_target_bbox, spots_labels

class ClassifierTrainer(Trainer):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        for i,(x, y, _,_)  in enumerate(self.train_dl):
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            # print("y_pred: ", y_pred.shape, "y: ", y.shape)
            
            loss = self.criterion(y_pred[:, :self.num_classes], y[:, :self.num_classes].argmax(dim=1))
            if not only_p:
                loss2 = self.criterion(y_pred[:, self.num_classes:], y[:, self.num_classes:].argmax(dim=1))
                loss = loss + loss2
            # print("y_pred: ", y_pred.argmax(1), "y: ", y.argmax(1))
            self.logger.add_scalar('train_loss', loss.item(), i + epoch*len(self.train_dl))
            self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i + epoch*len(self.train_dl))

            # if not only_p:            
            #     # loss1 = self.criterion(y_pred_i, y[:, self.num_classes:])
            #     # loss += loss1
            #     loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            acc = (y_pred[:, :self.num_classes].argmax(dim=1) == y[:, :self.num_classes].argmax(dim=1)).sum().item()
            # if plot:
            #     print(y_pred.argmax(dim=1), y.argmax(dim=1))
                # self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch)
            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            # print(f"train - acc_p: {acc_p}, acc_i: {acc_i}")
            # acc = (acc_p + acc_i) / 2
            train_loss.append(loss.item())
            train_acc += acc
        print("number of train_acc: ", train_acc)
        return train_loss, train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        for i, (x, y, _, _) in enumerate(self.val_dl):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
            
            loss = self.criterion(y_pred[:, :self.num_classes], y[:, :self.num_classes].argmax(dim=1))
            if not only_p:
                loss2 = self.criterion(y_pred[:, self.num_classes:], y[:, self.num_classes:].argmax(dim=1))
                loss = loss + loss2
            self.logger.add_scalar('validation_loss', loss.item(), i + epoch*len(self.train_dl))

            # if not only_p:            
            #     # loss1 = self.criterion(y_pred_i, y[:, self.num_classes:])
            #     # loss += loss1
            #     loss = self.criterion(y_pred, y)
            acc = (y_pred[:, :self.num_classes].argmax(dim=1) == y[:, :self.num_classes].argmax(dim=1)).sum().item()
            # if plot:
            #     print(y_pred.argmax(dim=1), y.argmax(dim=1))
                # self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch)
            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            val_loss.append(loss.item())
            val_acc += acc
        print("number of val_acc: ", val_acc)
        return val_loss, val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dataloader, device, conf=False, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model()
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        for x, y, _, _ in test_dataloader:
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
                # y_pred = torch.cat((y_pred_p, y_pred_i), dim=1)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, np.zeros((0, self.num_classes))


class latTrainer(Trainer):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        for i,(x, y, _,_)  in enumerate(self.train_dl):
            x, y = x.to(device), y.to(device)
            if torch.any(torch.isnan(x)):
                print("nan x")
            if torch.any(torch.isnan(y)):
                print("nan y")
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            # print("train: ", y_pred[:10], y[:10])           

            # print(self.model.named_parameters()[0][1].grad()[:10])
            # print("y_pred: ", y_pred[:10], "y: ", y[:10])
            loss = self.criterion(y_pred, y)
            
            self.logger.add_scalar('train_loss', loss.item(), i + epoch*len(self.train_dl))
            self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i + epoch*len(self.train_dl))
            loss.backward()
            # for name, param in self.model.named_parameters():
            #     print(f'Parameter: {name}, Gradient flag: {param.requires_grad}, Value: {param.grad}')
            self.optimizer.step()
            acc = (y_pred.argmax(-1) == y).sum().item()
            train_loss.append(loss.item())
            train_acc += acc
        print("number of train_acc: ", train_acc)
        return train_loss, train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        for i, (x, y, _, _) in enumerate(self.val_dl):
            x, y = x.to(device), y.to(device)
            # with torch.no_grad():
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            self.logger.add_scalar('validation_loss', loss.item(), i + epoch*len(self.val_dl))
            acc = (y_pred.argmax(-1) == y).sum().item()
            val_loss.append(loss.item())
            val_acc += acc
        print("number of val_acc: ", val_acc)
        return val_loss, val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dataloader, device, conf=False, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model()
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        for x, y, _, _ in test_dataloader:
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
                # y_pred = torch.cat((y_pred_p, y_pred_i), dim=1)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, np.zeros((0, self.num_classes))
    

class IncTrainer(Trainer):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        pbar = tqdm(self.train_dl)
        for i,(x, y, params,_)  in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            if torch.any(torch.isnan(x)):
                print("nan x")
            if torch.any(torch.isnan(y)):
                print("nan y")
            self.optimizer.zero_grad()
            # print('nans: ', torch.any(torch.isnan(x)).item(), torch.any(torch.isnan(params)).item())
            y_pred = self.model(x, params)

            # print("train: ", y_pred[:10], y[:10])           

            # print(self.model.named_parameters()[0][1].grad()[:10])
            # print("y_pred: ", y_pred[:10], "y: ", y[:10])
            loss = self.criterion(y_pred, y)
            
            self.logger.add_scalar('train_loss', loss.item(), i + epoch*len(self.train_dl))
            self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], i + epoch*len(self.train_dl))
            loss.backward()
            # for name, param in self.model.named_parameters():
            #     print(f'Parameter: {name}, Gradient flag: {param.requires_grad}, Value: {param.grad}')
            self.optimizer.step()
            diff = torch.abs(y_pred - y)
            acc = (diff < (y/10)).sum().item() 
            train_loss.append(loss.item())
            train_acc += acc
            pbar.set_description(f"train_acc: {acc}, train_loss:  {loss.item()}")
        return train_loss, train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        for i, (x, y, params, _) in enumerate(self.val_dl):
            x, y = x.to(device), y.to(device)
            # with torch.no_grad():
            y_pred = self.model(x, params)
            loss = self.criterion(y_pred, y)
            self.logger.add_scalar('validation_loss', loss.item(), i + epoch*len(self.val_dl))
            diff = torch.abs(y_pred - y)
            acc = (diff < (y/10)).sum().item()
            val_loss.append(loss.item())
            val_acc += acc
        print("number of val_acc: ", val_acc)
        return val_loss, val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dataloader, device, conf=False, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model()
        self.model.eval()
        preds = np.zeros((0, self.num_classes))
        targets = np.zeros((0, self.num_classes))
        for x, y, params, _ in test_dataloader:
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x, params)
                # y_pred = torch.cat((y_pred_p, y_pred_i), dim=1)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, np.zeros((0, self.num_classes))

class SiameseTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=None):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        for i, (x1, x2) in enumerate(self.train_dl):
            # print(i)
            # print("x1: ", x1.shape, "x2: ", x2.shape)
            x1, x2 = x1.to(device), x2.to(device)
            out = self.model(x1, x2)
            loss = out['loss']
            loss.backward()
            self.optim.step()
            train_loss.append(loss.item())  
            if i > self.max_iter:
                break        
        return train_loss, 0

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        for i, (x1, x2) in enumerate(self.val_dl):
            x1, x2 = x1.to(device), x2.to(device)
            with torch.no_grad():
                out = self.model(x1, x2)
            loss = out['loss']
            val_loss.append(loss.item())  
            if i > self.max_iter:
        
                break       
        return val_loss, 0


class EncoderTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_epoch(self, device, epoch=None, only_p=False ,plot=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = 0
        train_acc = 0
        for x, y  in self.train_dl:
            # print("grads :", len(g), "last grad: ", g[-1])
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y) 
            loss.backward()
            self.optimizer.step()
            if plot:
                self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch)

            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            # print(f"train - acc_p: {acc_p}, acc_i: {acc_i}")
            # acc = (acc_p + acc_i) / 2
            train_loss += loss.item()
            diff = torch.abs(y_pred - y)
            train_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
        print("number of train_acc: ", train_acc)
        return train_loss/len(self.train_dl), train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = 0
        val_acc = 0
        for x, y in self.val_dl:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            if plot:
                self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch, data='val')
            
            acc = (y_pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            val_loss += loss.item()
            diff = torch.abs(y_pred - y)
            val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
        print("number of val_acc: ", val_acc)
        return val_loss/len(self.val_dl), val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dl, device):
        """
        Returns the predictions of the model on the given dataset.
        """
        self.model.to(device)
        self.model.load_state_dict(self.best_state_dict)
        self.model.eval()
        preds = np.zeros((0, 2))
        targets = np.zeros((0, 2))
        for x, y in test_dl:
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x, return_logits=True)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dl.dataset))
        return preds, targets

class EncoderDecoderTrainer(Trainer):
    def __init__(self, recon_loss, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.recon_loss = recon_loss
        self.alpha = alpha


    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        pbar = tqdm(self.train_dl)
        for x, y,_,_  in pbar:
            x, y= x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            y_std = torch.std(y.squeeze(), dim=1)
            y_pred_std = torch.std(y_pred.squeeze(), dim=1)
            # y_pred_std = torch.masked_select(y_pred_std, ~torch.isnan(y_pred_std))
            # y_std = torch.masked_select(y_std, ~torch.isnan(y_pred_std))
            # print("is nan: ", torch.any(torch.isnan(y_pred_std)).item(), torch.any(torch.isnan(y_std)).item())
            # print("y_pred_std: ", y_pred_std.shape, "y_std: ", y_std.shape, "y: ", y.shape, "y_pred: ", y_pred.shape)
            # print(y_pred.squeeze().shape, y.squeeze().shape)
            loss_pred = self.criterion(y_pred.squeeze(), y.squeeze())
            loss_std = self.criterion(y_pred_std, y_std)
            loss = loss_pred + self.alpha * loss_std
            loss.backward()
            self.optimizer.step()
            if plot:
                self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch)
            pbar.set_description(f"train_loss:  {loss.item()}")

            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            # print(f"train - acc_p: {acc_p}, acc_i: {acc_i}")
            # acc = (acc_p + acc_i) / 2
            train_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            train_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
        print("number of train_acc: ", train_acc)
        return train_loss, train_acc/len(self.train_dl.dataset) 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        for x, y,_,_ in self.val_dl:
            x, y= x.to(device), y.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
            loss_pred = self.criterion(y_pred.squeeze(), y.squeeze())
            y_std = torch.std(y.squeeze(), dim=1)
            y_pred_std = torch.std(y_pred.squeeze(), dim=1)
            loss_std = self.criterion(y_pred_std, y_std)
            # loss_recon = self.recon_loss(logits.argmax(dim=-1), x[:,0,:])
            # print(f"val - loss_pred: {loss_pred}, loss_recon: {loss_recon}")
            # loss = loss_pred + self.alpha * loss_recon
            loss = loss_pred + self.alpha * loss_std
            if plot:
                self.plot_pred_vs_true(y_pred.cpu().detach().numpy(), y.cpu().detach().numpy(), epoch=epoch, data='val')
            
            # acc = (y_pred.argmax(dim=1) == y.argmax(dim=1)).sum().item()
            # acc_i = (y_pred[:,self.num_classes:].argmax(dim=1) == y[:,self.num_classes:].argmax(dim=1)).sum().item()
            val_loss.append(loss.item())
            diff = torch.abs(y_pred - y)
            val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
        print("number of val_acc: ", val_acc)
        return val_loss, val_acc/len(self.val_dl.dataset)
    
    def predict(self, test_dl, device):
        """
        Returns the predictions of the model on the given dataset.
        """
        self.model.to(device)
        self.model.load_state_dict(self.best_state_dict)
        self.model.eval()
        preds = np.zeros((0, 2))
        targets = np.zeros((0, 2))
        for x, y,_,_ in test_dl:
            x = x.to(device)
            with torch.no_grad():
                y_pred, _ = self.model(x, return_logits=True)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dl.dataset))
        return preds, targets
    

class MaskedSSLTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        train_acc = 0
        for i, (x, x_masked, mask, _) in enumerate(self.train_dl):
            x_masked, mask, x = x_masked.to(device), mask.to(device), x.to(device)
            out = self.model(x_masked, x).squeeze()
            # print(mask.shape, out.shape, x.shape)
            # tm = mask.expand_as(out)  
            out = out.masked_fill(mask, 0)
            x_target = x.squeeze().masked_fill(mask, 0)
            # print(x_target.shape, out.shape ) 
            loss = self.criterion(out, x_target)
            if self.logger is not None:
                self.logger.add_scalar('train_loss', loss.item(), i + epoch*len(self.train_dl))
            loss.backward()
            self.optim.step()
            train_loss.append(loss.item())
            train_acc += self.mask_accuracy(out, x_target, mask).item()
        return train_loss, train_acc/len(self.train_dl.dataset)

    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        for i, (x, x_masked, mask, _) in enumerate(self.val_dl):
            x_masked, mask, x = x_masked.to(device), mask.to(device), x.to(device)
            with torch.no_grad():
                out = self.model(x_masked, x).squeeze()
            # print(mask.shape, out.shape, x.shape)
            # tm = mask.expand_as(out)  
            out = out.masked_fill(mask, 0)
            x_target = x.squeeze().masked_fill(mask, 0)
            loss = self.criterion(out, x_target)
            if self.logger is not None:
                self.logger.add_scalar('validation_loss', loss.item(), i + epoch*len(self.train_dl))
            if torch.isnan(loss).any():
                print("loss is nan")
                print("out: ", out, "x_target: ", x_target, "mask: ", mask)
            val_loss.append(loss.item())
            val_acc += self.mask_accuracy(out, x_target, mask).item()
            # print("val_loss: ",loss.item(), "val_acc: ", self.mask_accuracy(out, x_target, mask))
        return val_loss, val_acc/len(self.val_dl.dataset)
    
    def mask_accuracy(self, result, target, inverse_token_mask):
        # print(inverse_token_mask.shape, result.shape, target.shape)
        r = result.masked_select(~inverse_token_mask)  
        t = target.masked_select(~inverse_token_mask)  
        s = (r == t).sum()  
        return s
    



class DenoisingTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def train_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        self.model.train()
        train_loss = []
        for x, y  in self.train_dl:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            if y_pred.isnan().any():
                print("y_pred is nan")
            # print(y_pred - y)
            loss = self.criterion(y_pred.squeeze(1), y.squeeze(1)) 
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
        return train_loss, 0 
    
    def eval_epoch(self, device, epoch=None, only_p=False ,plot=False, conf=False):
        self.model.eval()
        val_loss = []
        for x, y  in self.val_dl:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred.squeeze(1), y.squeeze(1)) 
            val_loss.append(loss.item())
        return val_loss, 0
    

