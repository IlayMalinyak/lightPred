import torch.nn as nn
import torch
import torch.nn.functional as F



class CNN(nn.Module):
    """
    1D CNN model architecture.
    
    Attributes
    ----------
    num_in : int
        Exposure in seconds.
        
    log : _io.TextIOWrapper
        Log file.
    
    kernel1, kernel2 : int
        Kernel width of first and second convolution, respectively.
    stride1, stride2 : int
        Stride of first and second convolution, respectively.
    
    padding1, padding2 : int
        Zero-padding of first and second convolution, respectively.
    dropout : float
        Dropout probability applied to fully-connected part of network.
    
    hidden1, hidden2, hidden3 : int
        Number of hidden units in the first, second, and third fully-connected
        layers, respectively.
    
    Methods
    -------
    forward(x)
        Forward pass through the model architecture.
    """
    def __init__(self, t_samples, kernel1=3, kernel2=3, stride1=1, stride2=1, \
                 padding1=1, padding2=1, dropout=0.2, hidden1=2048, hidden2=1024, \
                 hidden3=256, out_channels1=64, out_channels2=16):
    
        super(CNN, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.num_in = t_samples

        self.out_channels_1 = out_channels1
        dilation1 = 1
        poolsize1 = 4
        
        self.out_channels_2 = out_channels2
        dilation2 = 1
        poolsize2 = 2

        # first convolution
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=self.out_channels_1,
                               kernel_size=kernel1,
                               dilation=dilation1,
                               stride=stride1,
                               padding=padding1)
        self.num_out = ((self.num_in+2*padding1-dilation1* \
                         (kernel1-1)-1)/stride1)+1
        assert str(self.num_out)[-1] == '0'

        self.bn1 = nn.BatchNorm1d(num_features=self.out_channels_1)
        self.pool1 = nn.AvgPool1d(kernel_size=poolsize1)
        self.num_out = (self.num_out/poolsize1)
        assert str(self.num_out)[-1] == '0'

        
        # hidden convolution
        self.conv_hidden = nn.Conv1d(in_channels=self.out_channels_1,
                               out_channels=self.out_channels_1,
                               kernel_size=kernel2,
                               stride=stride2,
                               padding=padding2)
        self.bn_hidden = nn.BatchNorm1d(num_features=self.out_channels_1)
        self.pool_hidden = nn.AvgPool1d(kernel_size=poolsize2)

        self.conv2 = nn.Conv1d(in_channels=self.out_channels_1,
                               out_channels=self.out_channels_2,
                               kernel_size=kernel2,
                               stride=stride2,
                               padding=padding2)
        self.num_out = ((self.num_out+2*padding2-dilation2* \
                         (kernel2-1)-1)/stride2)+1
        assert str(self.num_out)[-1] == '0'
        self.bn2 = nn.BatchNorm1d(num_features=self.out_channels_2)
        self.pool2 = nn.AvgPool1d(kernel_size=poolsize2)
        self.num_out = (self.num_out/(poolsize2**5))
        assert str(self.num_out)[-1] == '0'
        
        # fully-connected network
        self.num_out = self.out_channels_2*self.num_out
        assert str(self.num_out)[-1] == '0'
        self.num_out = int(self.num_out)
        self.linear1 = nn.Linear(2*self.num_out, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, hidden3)
        
        # output prediction
        self.predict = nn.Linear(hidden3, 2)
    

    def forward(self, x):
        """
        Forward pass through the model architecture.
            
        Parameters
        ----------
        x : array_like
            Input time series data.
            
        s : array_like
            Standard deviation array.
            
        Returns
        ----------
        x : array_like
            Output prediction.
        """
        s = torch.ones((x.shape[0], self.num_out),device=x.device)*torch.std(x)
        x = self.pool1(F.relu(self.bn1((self.dropout((self.conv1(x)))))))
        x = self.pool_hidden(F.relu(self.bn_hidden((self.dropout((self.conv1(x)))))))
        x = self.pool_hidden(F.relu(self.bn_hidden((self.dropout((self.conv1(x)))))))
        x = self.pool_hidden(F.relu(self.bn_hidden((self.dropout((self.conv1(x)))))))
        x = self.pool_hidden(F.relu(self.bn_hidden((self.dropout((self.conv1(x)))))))
        x = self.pool2(F.relu(self.bn2((self.dropoout((self.conv2(x)))))))
       
        x = x.view(-1, self.num_out)
        x = torch.cat((x, s), 1)

        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(F.relu(self.linear2(x)))
        x = F.relu(self.linear3(x))
        x = F.softplus(self.predict(x))
        
        return x.float()

class LSTM(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, num_class1=100, num_class2=90, channels=256, dropout=0.2, stride=2):
        super(LSTM, self).__init__()
        self. conv2d = nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=3, padding=1, stride=stride)
        self.lstm = nn.LSTM(channels, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout1d(p=dropout)
        self.batchnorm = nn.BatchNorm1d(channels)
        self.fc1 = nn.Linear(hidden_size*2*seq_len//stride, num_class1) # multiply hidden_size by 2 for concatenation of forward and backward hidden states
        self.fc2 = nn.Linear(hidden_size*2*seq_len//stride, num_class2)

    def forward(self, x):
        x_f = self.batchnorm(self.drop(self.conv2d(x)))
        x_f = torch.swapaxes(x_f, 1,2)
        x_f,_ = self.lstm(x_f)
        x_f = x_f.reshape(x_f.shape[0], -1)
        out1 = self.fc1(x_f)
        out2 = self.fc2(x_f)
        return out1, out2
    

class CNN_B(nn.Module):
    """
    1D CNN model architecture according to Blanckato et al..
    
    Attributes
    ----------
    num_in : int
        Exposure in seconds.
        
    log : _io.TextIOWrapper
        Log file.
    
    kernel1, kernel2 : int
        Kernel width of first and second convolution, respectively.
    stride1, stride2 : int
        Stride of first and second convolution, respectively.
    
    padding1, padding2 : int
        Zero-padding of first and second convolution, respectively.
    dropout : float
        Dropout probability applied to fully-connected part of network.
    
    hidden1, hidden2, hidden3 : int
        Number of hidden units in the first, second, and third fully-connected
        layers, respectively.
    
    Methods
    -------
    forward(x)
        Forward pass through the model architecture.
    """
    def __init__(self, t_samples):
    
        super().__init__()
        
        self.dropout = nn.Dropout(p=0.3)
        self.num_in = t_samples

        self.out_channels_1 = 64
        poolsize1 = 4
        
        self.out_channels_2 = 16
        poolsize2 = 2
        self.t_samples = t_samples 
        # first convolution
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=64,
                               kernel_size=3,
                               stride=3,
                               padding=4)
        

        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.pool1 = nn.AvgPool1d(kernel_size=poolsize1)

    

        self.conv2 = nn.Conv1d(in_channels=64,
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=self.out_channels_2)
        self.pool2 = nn.AvgPool1d(kernel_size=poolsize2)
        
        # fully-connected network
        self.out_shape = self._out_shape()
        self.num_out = self.out_shape[1]*self.out_shape[2]
        # assert str(self.num_out)[-1] == '0'
        self.num_out = int(self.num_out)
        self.linear1 = nn.Linear(self.num_out, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 256)
        
        # output prediction
        self.predict = nn.Linear(256, 1)

    def _out_shape(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            
            dummy_input = torch.randn(1,1, self.t_samples)
            output = self.pool1(F.relu(self.bn1(self.dropout(self.conv1(dummy_input)))))
            output = self.pool2(F.relu(self.bn2(self.dropout(self.conv2(output)))))
            # n_features = output.numel() // output.shape[0]  
            return output.shape 
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    

    def forward(self, x):
        """
        Forward pass through the model architecture.
            
        Parameters
        ----------
        x : array_like
            Input time series data.
            
        s : array_like
            Standard deviation array.
            
        Returns
        ----------
        x : array_like
            Output prediction.
        """
        s = torch.ones((x.shape[0], self.out_shape[1]*self.out_shape[2]),device=x.device)*torch.std(x)
        print("shape of s ", s.shape, "shape of x ", x.shape)
        x = self.pool1(F.relu(self.bn1(self.dropout(self.conv1(x)))))
        x = self.pool2(F.relu(self.bn2(self.dropout(self.conv2(x)))))
        print("x_shape after conv", x.shape)
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, s), 1)

        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(F.relu(self.linear2(x)))
        x = F.relu(self.linear3(x))
        x = F.softplus(self.predict(x))
        
        return x.float()
