import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import weight_norm
from lightPred.utils import residual_by_period
from lightPred.period_analysis import analyze_lc_torch, analyze_lc
from torchvision.models.swin_transformer import SwinTransformer
import time

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=64):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        # print("projection_MLP: ", x.shape)
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=32, out_dim=64): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiam(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()
    
    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L}

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Adjust the dimensions using a 1x1 convolutional layer if needed
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(identity)
        out = self.relu(out)

        return out
class ResNet(nn.Module):
    def __init__(self, block, blocks,seq_len=1024, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.num_classes = num_classes
        self.seq_len = seq_len

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        layers = [self.make_layer(block, 64*2**i, blocks[i], stride=min(2,i+1) ) for i in range(len(blocks))]

        self.layers = torch.nn.Sequential(*layers)
        self.out_shape = self._out_shape()
        self.output_dim = self.out_shape[-1]
        # self.avg_pool = nn.MaxPool1d(2)
        # self.fc = nn.Linear(64*2**(len(blocks)-1), num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        print(stride)
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _out_shape(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            dummy_input = torch.randn(1,1, self.seq_len)
            output = self.forward(dummy_input)
            # n_features = output.numel() // output.shape[0]  
            return output.shape 
        finally:
            torch.set_rng_state(rng_state)

    def forward(self, x):
        # print("stating backbone with: ", x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layers(out)
    
        # out = self.avg_pool(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        # print("after backbone: ", out.shape)

        return out.view(out.shape[0], -1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.2):
        super(ConvBlock, self).__init__()
        print("ConvBlock: ", in_channels, out_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.pool = nn.MaxPool1d(kernel_size=stride)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        skip = self.skip(x)
        x = self.activation(self.bn(self.conv(x)))
        x = self.dropout(self.pool(x))
        x = x + skip
        return x

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, seq_len=1024, hidden_size=256, num_layers=5, num_classes=4,
                 in_channels=1, channels=256, dropout=0.2, kernel_size=4 ,stride=4, image=False):
        super(LSTMFeatureExtractor, self).__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.image = image
        self.stride = stride
        self.t_features = self.seq_len//self.stride
        print("image: ", image)
        # self.conv = Conv2dSubampling(in_channels=in_channels, out_channels=channels)
        if not image:
            self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size, padding='same', stride=1)
            self.pool = nn.MaxPool1d(kernel_size=stride)
            self.skip = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=1, padding=0, stride=stride)
            self.drop = nn.Dropout1d(p=dropout)
            self.batchnorm1 = nn.BatchNorm1d(channels)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size, padding='same', stride=1)
            self.pool = nn.MaxPool2d(kernel_size=stride)
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, padding=0, stride=stride)
            self.drop = nn.Dropout2d(p=dropout)
            self.batchnorm1 = nn.BatchNorm2d(channels)        
        # self.conv_pre = nn.Sequential(
        #                         ConvBlock(in_channels,channels//8, kernel_size=9, stride=1, padding='same', dropout=dropout),
        #                         ConvBlock(channels//8,channels//4, kernel_size=15, stride=1, padding='same', dropout=dropout),
        #                         ConvBlock(channels//4,channels//2, kernel_size=25, stride=1, padding='same', dropout=dropout))
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=1, stride=2)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=channels, kernel_size=kernel_size, padding=1, stride=4)
        
        self.lstm = nn.LSTM(channels, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
       
        self.activation = nn.GELU()
        self.num_features = self._out_shape()
        self.output_dim = self.num_features

    def _out_shape(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        # try:
        # print("calculating out shape")
        if not self.image:
            dummy_input = torch.randn(2,self.in_channels, self.seq_len)
        else:
            dummy_input = torch.randn(2,self.in_channels, self.seq_len, self.seq_len)
        # dummy_input = torch.randn(2,self.seq_len, self.in_channels)
        input_length = torch.ones(2, dtype=torch.int64)*self.seq_len
        # print("dummy_input: ", dummy_input.shape)
        # x = self.conv_pre(dummy_input)
        x = self.drop(self.pool(self.activation(self.batchnorm1(self.conv1(dummy_input)))))
        # x = self.conv(dummy_input, input_length)
        x = x.view(x.shape[0], x.shape[1], -1).swapaxes(1,2)
        x_f,(h_f,_) = self.lstm(x)
        h_f = h_f.transpose(0,1).transpose(1,2)
        h_f = h_f.reshape(h_f.shape[0], -1)
        # print("finished")
        return h_f.shape[1] 
        # finally:
        #     torch.set_rng_state(rng_state)

    def forward(self, x, return_cell=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.transpose(-1,-2)
        # if len(x.shape) == 2:
        #     x = x.unsqueeze(-1)
        # elif len(x.shape) == 3 and x.shape[1] == 1:
        #     x = x.transpose(-1,-2)
        # input_length = torch.ones(x.shape[0], dtype=torch.int64)*self.seq_len
        # x = self.conv(x, input_length)
        skip = self.skip(x)
        x = self.drop(self.pool(self.activation(self.batchnorm1(self.conv1(x)))))
        x = x + skip[:, :, :x.shape[-1]] # [B, C, L//stride]
        x = x.view(x.shape[0], x.shape[1], -1).swapaxes(1,2) # [B, L//stride, C]
        x_f,(h_f,c_f) = self.lstm(x) # [B, L//stride, 2*hidden_size], [B, 2*num_layers, hidden_size], [B, 2*num_layers, hidden_size
        if return_cell:
            return x_f, h_f, c_f
        h_f = h_f.transpose(0,1).transpose(1,2)
        h_f = h_f.reshape(h_f.shape[0], -1)
        return h_f
    
class LSTM(nn.Module):
    def __init__(self, seq_len=1024, hidden_size=64, num_layers=5, num_classes=4,
                 in_channels=1, predict_size=128, channels=256, dropout=0.35, kernel_size=4,stride=4, image=False):
        super(LSTM, self).__init__()
        self.activation = nn.GELU()
        self.feature_extractor = LSTMFeatureExtractor(seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes,in_channels=in_channels,
                                                      channels=channels, dropout=dropout, kernel_size=kernel_size,
                                                      image=image)
        self.num_classes = num_classes
        self.out_shape = self.feature_extractor.num_features
        self.predict_size = predict_size
        self.fc1 = nn.Linear(self.out_shape, predict_size)
        self.fc2 = nn.Linear(predict_size, num_classes)
        # self.fc3 = nn.Linear(predict_size, num_classes)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        h_f = self.feature_extractor(x)
        out = self.fc2(self.activation(self.fc1(h_f)))
        # out2 = self.fc3(self.fc1(x_f))
        return out

class LSTM_ATTN(LSTM):
    def __init__(self, **kwargs):
        super(LSTM_ATTN, self).__init__(**kwargs)
        self.fc1 = nn.Linear(self.feature_extractor.hidden_size*2, self.predict_size)
        self.fc2 = nn.Linear(self.predict_size, self.num_classes)
        # self.fc3 = nn.Linear(self.feature_extractor.hidden_size*4, self.predict_size)
        # self.fc4 = nn.Linear(self.predict_size, self.num_classes)


    def attention(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [BxTxK]
        # Values = [BxTxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)
        scale = 1/(keys.size(-1) ** -0.5)
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(1,2) # [BxTxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(scale), dim=2) # scale, normalize

        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return linear_combination

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  
        x_f, h_f, c_f = self.feature_extractor(x, return_cell=True)
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1)

        values = self.attention(c_f, x_f, x_f) 
        out = self.fc2(self.activation(self.fc1(values)))
        # values = (values[...,None] + out[:,None,:]).view(out.shape[0], -1)
        # conf = self.fc4(self.activation(self.fc3(values)))
        return out

class LSTM_ATTN2(LSTM):
    def __init__(self, attn='prob',n_heads=4, factor=5, dropout=0.35, output_attention=False, num_att_layers=4, **kwargs):
        super(LSTM_ATTN2, self).__init__(**kwargs)
        self.fc1 = nn.Linear(self.feature_extractor.hidden_size*self.feature_extractor.hidden_size*2, self.predict_size)
        self.fc2 = nn.Linear(self.predict_size, self.num_classes)
        self.pool = nn.AdaptiveMaxPool1d(self.feature_extractor.hidden_size)
        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=self.feature_extractor.hidden_size*2,
            num_attention_heads=n_heads,
            feed_forward_dropout_p=dropout,
            attention_dropout_p=dropout,
            conv_dropout_p=dropout,
        ) for _ in range(num_att_layers)])
        # Attn = ProbAttention if attn=='prob' else FullAttention
        # self.attention = AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model=self.feature_extractor.hidden_size*2, n_heads=n_heads, mix=False)
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  
        outputs, h_f, c_f = self.feature_extractor(x, return_cell=True)
        # print("x_f: ", x_f.shape)
        for layer in self.layers:
            outputs = layer(outputs)
        outputs = self.pool(outputs.permute(0,2,1)).reshape(outputs.shape[0], -1)
        # c_f = torch.cat([c_f[-1], c_f[-2]], dim=1)

        # values, _ = self.attention(x_f, x_f, x_f, attn_mask=None) 
        outputs = self.fc2(self.activation(self.fc1(outputs)))
        # values = (values[...,None] + out[:,None,:]).view(out.shape[0], -1)
        # conf = self.fc4(self.activation(self.fc3(values)))
        return outputs

class LSTM_ATTN_QUANT(LSTM):
    def __init__(self, attn='prob',n_heads=8, factor=5, dropout=0.1, output_attention=False, n_q=3, **kwargs):
        super(LSTM_ATTN_QUANT, self).__init__(**kwargs)
        self.fc1 = nn.Linear(self.feature_extractor.hidden_size*2, self.predict_size)
        self.fc2 = nn.Linear(self.predict_size, self.num_classes)
        Attn = ProbAttention if attn=='prob' else FullAttention
        self.attention = AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model=self.feature_extractor.hidden_size*2, n_heads=n_heads, mix=False)
        self.quantiles = n_q
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  
        x_f, h_f, c_f = self.feature_extractor(x, return_cell=True)
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1)

        values, _ = self.attention(x_f, x_f, x_f, attn_mask=None) 
        values = values.unsqueeze(1).expand( values.shape[0],self.quantiles, values.shape[1], values.shape[2])
        out = self.fc2(self.activation(self.fc1(values.max(dim=2).values)))
        # values = (values[...,None] + out[:,None,:]).view(out.shape[0], -1)
        # conf = self.fc4(self.activation(self.fc3(values)))
        return out.transpose(1,2)

class LSTM_DUAL(nn.Module):
    def __init__(self, dual_model, encoder_dims, lstm_args, predict_size, num_classes=4, freeze=False, **kwargs):
        super(LSTM_DUAL, self).__init__(**kwargs)
        # print("intializing dual model")
        self.feature_extractor = LSTMFeatureExtractor(**lstm_args)
        if freeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                # for param in self.attention.parameters():
                #     param.requires_grad = False
        num_lstm_features = self.feature_extractor.hidden_size*2
        self.num_features = num_lstm_features + encoder_dims
        self.output_dim = self.num_features
        self.dual_model = dual_model
        self.pred_layer = nn.Sequential(
        nn.Linear(self.num_features, predict_size),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(predict_size,num_classes//2),
    )   
        self.conf_layer = nn.Sequential(
        nn.Linear(16, 16),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(16,num_classes//2),)

    def lstm_attention(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [BxTxK]
        # Values = [BxTxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)
        scale = 1/(keys.size(-1) ** -0.5)
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(1,2) # [BxTxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(scale), dim=2) # scale, normalize

        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return linear_combination
    
    def forward(self, x, x_dual=None, acf_phr=None):
        if x_dual is None:
            x, x_dual = x[:,0,:], x[:,1,:]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x, h_f, c_f = self.feature_extractor(x, return_cell=True) # [B, L//stride, 2*hidden_size], [B, 2*nlayers, hidden_szie], [B, 2*nlayers, hidden_Size]
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1) # [B, 2*hidden_szie]
        t_features = self.lstm_attention(c_f, x, x) # [B, 2*hidden_size]
        d_features, _ = self.dual_model(x_dual) # [B, encoder_dims]
        features = torch.cat([t_features, d_features], dim=1) # [B, 2*hidden_size + encoder_dims]
        out = self.pred_layer(features)
        if acf_phr is not None:
            phr = acf_phr.reshape(-1,1).float().to(features.device)
        else:
            phr = torch.zeros(features.shape[0],1, device=features.device)
        mean_features = torch.nn.functional.adaptive_avg_pool1d(features.unsqueeze(1), 16).squeeze(1)
        mean_features += phr
        conf = self.conf_layer(mean_features)
        return torch.cat([out, conf], dim=1)

class EncoderDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        enc_out, memory = self.encoder(x)
        out = self.decoder(memory.transpose(1,2))
        return out

class SpotNet(torch.nn.Module):
    def __init__(self, encoder, encoder_dims, decoder, num_queries, num_classes=2, lstm_model=None,
     freeze=False, dropout=0.3, **kwargs):
        super(SpotNet, self).__init__()
        # print("intializing dual model")
        assert lstm_model is not None or encoder is not None
        if lstm_model is not None:
            self.feature_extractor = lstm_model.feature_extractor
            self.attention = lstm_model.attention
            self.predict_size = lstm_model.predict_size
            if freeze:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                # for param in self.attention.parameters():
                #     param.requires_grad = False
        self.num_classes = num_classes
        print("num clases in model: ", self.num_classes)
        num_lstm_features = self.feature_extractor.hidden_size*2
        self.num_features = num_lstm_features + encoder_dims
        self.encoder = encoder
        self.decoder = decoder
        self.object_queries = nn.Embedding(num_queries, self.num_features)
        self.pred_layer = nn.Sequential(
        nn.Linear(self.num_features, self.predict_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(self.predict_size,self.num_classes),
    )
        self.spot_class_layer = nn.Linear(self.num_features, 2)
        self.spot_box_layer = nn.Sequential(
        nn.Linear(self.num_features, self.predict_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(self.predict_size, self.predict_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(self.predict_size,4),
    )

    def forward(self, x_acf, x):
        bs, T = x.shape
        if len(x.shape) == 2:
            x_acf = x_acf.unsqueeze(1)
        tic = time.time()
        x_f, h_f, c_f = self.feature_extractor(x_acf, return_cell=True) # [B, L//stride, 2*hidden_size], [B, 2*nlayers, hidden_szie], [B, 2*nlayers, hidden_Size]
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1) # [B, 2*hidden_szie]
        t_features = self.attention(c_f, x_f, x_f) # [B, 2*hidden_size]
        t1 = time.time()
        d_features, memory = self.encoder(x) # [B, encoder_dims], [B, L//stride, encoder_dims]
        t2 = time.time()
        features = torch.cat([t_features, d_features], dim=1) # [B, 2*hidden_size + encoder_dims]
        x_f = nn.functional.adaptive_avg_pool1d(x_f.transpose(1,2), memory.shape[1]).transpose(1,2)
        memory = torch.cat([memory, x_f], dim= -1) # [B, L//stride, 2*hidden_size + encoder_dims]
        query_embed = self.object_queries.weight.unsqueeze(0).repeat(bs,1, 1)
        tgt = torch.zeros_like(query_embed)
        decoder_output = self.decoder(memory, tgt)
        t3 = time.time()
        predictions = self.pred_layer(features)
        class_logits = self.spot_class_layer(decoder_output)
        bbox_logits = self.spot_box_layer(decoder_output).sigmoid()
        t4 = time.time()
        out_dict = {'pred_boxes': bbox_logits, 'pred_logits': class_logits}
        # print("time: ", "lstm: ", t1-tic, "encoder: ",  t2-t1, "deocder: ", t3-t2,"linear layers: ", t4-t3)
        return out_dict, predictions

class NaiveDecoder(nn.Module):
    def __init__(self, encoder_dim, stride, hidden_dim=256):
        super(NaiveDecoder, self).__init__()
        self.conv = torch.nn.Sequential(nn.ConvTranspose1d(in_channels=encoder_dim,
                                                              kernel_size=stride, out_channels=encoder_dim,
                                                              stride=stride, padding=0, bias=True),
                                           nn.BatchNorm1d(encoder_dim),
                                           nn.SiLU())
        self.linear = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x.transpose(1,2)).transpose(1,2)
        return x
    
class LSTM_SWIN(LSTM_ATTN):
    def __init__(self, patch_size=[16,16], im_embed_dim=64, depths=[4,6,6,4], num_heads=[4,8,8,4],
     window_size=[8,8], im_dropout=0.3, swin_weight=1, **kwargs):
        super(LSTM_SWIN, self).__init__(**kwargs)
        self.swin = SwinTransformer(patch_size=patch_size, embed_dim=im_embed_dim, depths=depths, num_heads=num_heads,
                                     window_size=window_size, dropout=im_dropout)
        self.swin.head = nn.Identity()
        num_swin_features =  im_embed_dim * 2 ** (len(depths) - 1)
        num_lstm_features = self.feature_extractor.hidden_size*2
        self.swin_weight = swin_weight
        self.fc1 = nn.Linear(num_swin_features + num_lstm_features, self.predict_size)

        
    def forward(self, x_im, x_t):
        if len(x_t.shape) == 2:
            x_t = x_t.unsqueeze(1)
        if x_im.shape[1] == 1:
            x_im = x_im.repeat(1,3,1,1) # RGB like
        x_f, h_f, c_f = self.feature_extractor(x_t, return_cell=True)
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1)
        t_features = self.attention(c_f, x_f, x_f) 
        im_features = self.swin_weight*self.swin(x_im)
        values = torch.cat([t_features, im_features], dim=1)
        out = self.fc2(self.activation(self.fc1(values)))
        return out

