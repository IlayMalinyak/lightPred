from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import time
import os
import yaml
from lightPred.utils import FitResult


class Trainer(object):
    """
    A class that encapsulates the training loop for a PyTorch model.
    """
    def __init__(self, model, optimizer, criterion, scheduler, train_dataloader, val_dataloader, device,
                 optim_params, net_params, exp_num, log_path, exp_name):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.optim_params = optim_params
        self.net_params = net_params
        self.exp_num = exp_num
        self.exp_name = exp_name
        self.log_path = log_path
        if log_path is not None:
            self.logger =SummaryWriter(f'{self.log_path}/exp{self.exp_num}')
            # print(f"logger path: {self.log_path}/exp{self.exp_num}")
            if not os.path.exists(f'{self.log_path}/exp{self.exp_num}'):
                os.makedirs(f'{self.log_path}/exp{self.exp_num}')
            with open(f'{self.log_path}/exp{exp_num}/net_params.yml', 'w') as outfile:
                yaml.dump(self.net_params, outfile, default_flow_style=False)
            with open(f'{self.log_path}/exp{exp_num}/optim_params.yml', 'w') as outfile:
                    yaml.dump(self.optim_params, outfile, default_flow_style=False)

    def fit(self, num_epochs, device, only_p=False, early_stopping=None):
        """
        Fits the model for the given number of epochs.
        """
        min_loss = np.inf
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        self.optim_params['lr_history'] = []
        epochs_without_improvement = 0

        print(f"Starting training for {num_epochs} epochs with parameters: {self.optim_params}, {self.net_params}")
        for epoch in range(num_epochs):
            start_time = time.time()
            t_loss, t_acc = self.train_epoch(device, only_p=only_p)
            self.logger.add_scalar('train_loss', t_loss, epoch)
            self.logger.add_scalar('train_acc', t_acc, epoch)
            train_loss.append(t_loss)
            train_acc.append(t_acc)

            v_loss, v_acc = self.eval_epoch(device, only_p=only_p)
            self.logger.add_scalar('validation_loss', v_loss, epoch)
            self.logger.add_scalar('validation_acc', v_acc, epoch)
            val_loss.append(v_loss)
            val_acc.append(v_acc)

            self.scheduler.step(v_loss)
            if v_loss < min_loss:
                print("saving model...")
                min_loss = v_loss
                torch.save(self.model.state_dict(), f'{self.log_path}/exp{self.exp_num}/{self.exp_name}.pth')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == early_stopping:
                    print('early stopping!', flush=True)
                    break
                
            self.logger.add_scalar('time', time.time() - start_time, epoch)
            print(f'Epoch {epoch}: Train Loss: {t_loss:.6f}, Val Loss: {v_loss:.6f}, Val Acc: {v_acc:.6f}, Train Acc: {t_acc:.6f}')
            self.optim_params['lr'] = self.optimizer.param_groups[0]['lr']
            self.optim_params['lr_history'].append(self.optim_params['lr'])
            with open(f'{self.log_path}/exp{self.exp_num}/optim_params.yml', 'w') as outfile:
                yaml.dump(self.optim_params, outfile, default_flow_style=False)

            if epoch % 10 == 0:
                print(os.system('nvidia-smi'))
        self.logger.close()
        return FitResult(num_epochs, train_loss, train_acc, val_loss, val_acc)

    def predict(self, test_dataloader, device):
        """
        Returns the predictions of the model on the given dataset.
        """
        self.model.eval()
        preds = []
        for x, y in test_dataloader:
            x = x.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
            preds.append(y_pred.cpu().numpy())
        return np.concatenate(preds) 

    def train_epoch(self, device, only_p=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = 0
        train_acc = 0
        for x, y in self.train_dataloader:
            x = x.to(device)
            y = y.to(device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            dummy  = sum([p.sum() for p in self.model.parameters()])*0
            # print("y_pred: ", y_pred.shape, "y: ", y.shape)
            loss = self.criterion(y_pred, y) if not only_p else self.criterion(y_pred, y[:, 0]) + dummy
            # print("loss: ", loss, "y_pred: ", y_pred, "y: ", y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            diff = torch.abs(y_pred - y)
            train_acc += (diff[:,0] < (y[:,0]/10)).sum().item()
        return train_loss/len(self.train_dataloader), train_acc/len(self.train_dataloader.dataset)

    def eval_epoch(self, device, only_p=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = 0
        val_acc = 0
        for x, y in self.val_dataloader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                y_pred = self.model(x)
            loss = self.criterion(y_pred, y) if not only_p else self.criterion(y_pred, y[:, 0])
            val_loss += loss.item()
            diff = torch.abs(y_pred - y)
            val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()        
        return val_loss/len(self.val_dataloader), val_acc/len(self.val_dataloader.dataset)
    
    
class ClassifierTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def train_epoch(self, device, only_p=False):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = 0
        train_acc = 0
        for x, y in self.train_dataloader:
            x = x.to(device)
            y = y.to(device)
            self.optimizer.zero_grad()
            y_hat_p, y_hat_i = self.model(x)
            # print(f"shapes-  y_hat_p:  {y_hat_p.shape}, y_hat_i: {y_hat_i.shape}, y_p: {y_p.shape}, y_i {y_i.shape}")
            # loss_p = self.criterion(y_hat_p, torch.nn.functional.one_hot(y[:,0], num_classes=y_hat_p.shape[1]).float())
            # loss_i = self.criterion(y_hat_i, torch.nn.functional.one_hot(y[:,1], num_classes=y_hat_i.shape[1]).float())
            loss_p = self.criterion(y_hat_p, y[:,:y_hat_p.shape[1]])
            loss_i = self.criterion(y_hat_i, y[:,y_hat_p.shape[1]:])
            loss = loss_p + loss_i
            # print("loss: ", loss, "y_pred: ", y_pred, "y: ", y)
            loss.backward()
            self.optimizer.step()
            acc_p = (y_hat_p.argmax(dim=1) == y[:,:y_hat_p.shape[1]].argmax(dim=1)).sum().item()
            acc_i = (y_hat_i.argmax(dim=1) == y[:,y_hat_p.shape[1]:].argmax(dim=1)).sum().item()
            # print(f"train - acc_p: {acc_p}, acc_i: {acc_i}")
            acc = (acc_p + acc_i) / 2
            train_loss += loss.item()
            train_acc += acc
        return train_loss/len(self.train_dataloader), train_acc/len(self.train_dataloader.dataset) 
    
    def eval_epoch(self, device, only_p=False):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = 0
        val_acc = 0
        for x, y in self.train_dataloader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                y_hat_p, y_hat_i = self.model(x)
            # loss_p = self.criterion(y_hat_p, torch.nn.functional.one_hot(y[:,0], num_classes=y_hat_p.shape[1]).float())
            # loss_i = self.criterion(y_hat_i, torch.nn.functional.one_hot(y[:,1], num_classes=y_hat_i.shape[1]).float())
            loss_p = self.criterion(y_hat_p, y[:,:y_hat_p.shape[1]])
            loss_i = self.criterion(y_hat_i, y[:,y_hat_p.shape[1]:])
            loss = loss_p + loss_i
            # print("loss: ", loss, "y_pred: ", y_pred, "y: ", y)
            acc_p = (y_hat_p.argmax(dim=1) == y[:,:y_hat_p.shape[1]].argmax(dim=1)).sum().item()
            acc_i = (y_hat_i.argmax(dim=1) == y[:,y_hat_p.shape[1]:].argmax(dim=1)).sum().item()
            # print(f"val - acc_p: {acc_p}, acc_i: {acc_i}")
            acc = (acc_p + acc_i) / 2
            val_loss += loss.item()
            val_acc += acc
        return val_loss/len(self.val_dataloader), val_acc/len(self.val_dataloader.dataset)
    
