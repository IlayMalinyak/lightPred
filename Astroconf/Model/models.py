import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softmax
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .Modules.conformer import ConformerEncoder, ConformerDecoder
from .Modules.mhsa_pro import RotaryEmbedding, ContinuousRotaryEmbedding
from .Modules.cnn import ResNetBlock
# from .Modules.NCE import Net
from .Modules.ResNet18 import ResNet18

class Astroconformer(nn.Module):
  def __init__(self, args) -> None:
    super(Astroconformer, self).__init__()
    self.head_size = args.encoder_dim // args.num_heads
    self.rotary_ndims = int(self.head_size * 0.5)
    print("extractor stride: ", args.stride)
    self.extractor = nn.Sequential(nn.Conv1d(in_channels = args.in_channels,
            kernel_size = args.stride, out_channels = args.encoder_dim, stride = args.stride, padding = 0, bias = True),
                    nn.BatchNorm1d(args.encoder_dim),
                    nn.SiLU(),
    )
    
    self.pe = RotaryEmbedding(self.rotary_ndims)
    
    self.encoder = ConformerEncoder(args)
    
    self.pred_layer = nn.Sequential(
        nn.Linear(args.encoder_dim, args.encoder_dim),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(args.encoder_dim,args.output_dim),
    )
    if getattr(args, 'mean_label', False):
      self.pred_layer[3].bias.data.fill_(args.mean_label)
  
    self.init_weights()

  def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
  def forward(self, inputs: Tensor) -> Tensor:
    x = inputs #initial input_size: [B, L]

    if len(x.shape) == 2:
        x = x.unsqueeze(1) # x: [B, 1, L]
    x = self.extractor(x) # x: [B, encoder_dim, L]
    x = x.permute(0,2,1) # x: [B, L, encoder_dim]
    RoPE = self.pe(x, x.shape[1]) # RoPE: [2, B, L, encoder_dim], 2: sin, cos
    x = self.encoder(x, RoPE) # x: [B, L, encoder_dim]
    memory = x.clone()
    # print("x shape: ", x.shape)
    x = x.mean(dim=1) # x: [B, encoder_dim]
    # print("x shape: ", x.shape)
    # attn = softmax(self.attnpool(x), dim=1) # attn: [B, L, 1]
    # x = torch.matmul(x.permute(0,2,1), attn).squeeze(-1) # x: [B, encoder_dim]
    x = self.pred_layer(x) # x: [B, 1]
    return x, memory

class AstroDecoder(nn.Module):
  def __init__(self, args) -> None:
    super(AstroDecoder, self).__init__()
    self.extractor = nn.Sequential(nn.ConvTranspose1d(in_channels = args.encoder_dim,
            kernel_size = args.stride, out_channels = args.decoder_dim, stride = args.stride, padding = 0, bias = True),
                    nn.BatchNorm1d(args.encoder_dim),
                    nn.SiLU(),
    )
    self.head_size = args.decoder_dim // args.num_heads
    self.rotary_ndims = int(self.head_size * 0.5)    
    self.pe = RotaryEmbedding(self.rotary_ndims)
    self.decoder = ConformerDecoder(args)
    
  def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
    x = tgt #initial input_size: [B, L, encoder_dim]

    RoPE = self.pe(x, x.shape[1]) # RoPE: [2, B, L, encoder_dim], 2: sin, cos
    x = self.decoder(x, memory, RoPE) # x: [B, L, encoder_dim]
    return x

class ResNetBaseline(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    
    self.embedding = nn.Sequential(nn.Conv1d(in_channels = args.in_channels,
            kernel_size=3, out_channels = args.encoder_dim, stride=1, padding = 'same', bias = False),
                    nn.BatchNorm1d(args.encoder_dim),
                    nn.SiLU(),
    )
    
    self.layers = nn.ModuleList([ResNetBlock(args)
      for _ in range(args.num_layers)])
    
    self.pred_layer = nn.Sequential(
        nn.Linear(args.encoder_dim, args.encoder_dim),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(args.encoder_dim,1),
    )
    if getattr(args, 'mean_label', False):
      self.pred_layer[3].bias.fill_(args.mean_label)
    
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.unsqueeze(1)
    x = self.embedding(x)
    for m in self.layers:
      x = m(x)
    x = x.mean(dim=-1)
    x = self.pred_layer(x)
    return x
    
model_dict = {
          'Astroconformer': Astroconformer,
          'ResNetBaseline': ResNetBaseline,
          'ResNet18': ResNet18,
      }