import torch
from torch import nn
from torch.nn import functional as F

def squeeze2d( x, factor=2):
  '''
  Changes the shape of x from (Batch_size, Channels, Height, Width )
  to (Batch_size, 4*Channels, Height/2, Width/2).

  x: (Tensor) input
  factor (int): how many slices to split the data
  '''

  B, C, H, W = x.shape
  assert H % factor == 0 and W % factor == 0, 'Height and Width must be divisible by factor.'

  x = x.view(B, C, H // factor, factor, W // factor, factor)
  x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
  x = x.view(B, C * (factor**2), H // factor, W // factor)
  return x

def unsqueeze2d( x, factor=2):
  '''
  Reverses the Squeeze operation above.

  x: (Tensor) input
  factor (int): how many slices to split the data
  '''
  B, C, H, W = x.shape

  x = x.view(B, C // 4, factor, factor, H, W)
  x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
  x = x.view(B, C // (factor ** 2), H *factor, W * factor)
  return x

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=2, activation='relu'):
        super().__init__()
        self.fc_1 = nn.Linear(in_dim, hid_dim)
        self.fc_2 = nn.Linear(hid_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, out_dim)
        if activation=='sigmoid':
            self.activation = F.sigmoid
        elif activation=='cos':
            self.activation = torch.cos
        elif activation=='tanh':
            self.activation = F.tanh
        elif activation=='relu':
            self.activation = F.relu
        elif activation=='leakyrelu':
            self.activation = lambda x: F.leaky_relu(x, negative_slope=0.2)
        else:
            raise NotImplementedError(activation + " not implemented")

    def forward(self, x):
        
        out = self.activation(self.fc_1(x))
        out = self.activation(self.fc_2(out))
        return self.fc_out(out)

class WeightNormConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
  def __init__(self, in_channels, hidden_features=512):
    super().__init__()
    layers =  [
      WeightNormConv2d(in_channels, hidden_features, kernel_size=3, padding=1),
      nn.ReLU(),
      WeightNormConv2d(hidden_features, hidden_features, kernel_size=1),
      nn.ReLU(),
      WeightNormConv2d(hidden_features, 2*in_channels, kernel_size=3, padding=1),
    ]
    self.net = nn.Sequential(*layers)

    self.net[0].conv.weight.data.normal_(0, 0.05)
    self.net[0].conv.bias.data.zero_()

    self.net[-1].conv.weight.data.normal_(0, 0.05)
    self.net[-1].conv.bias.data.zero_()

  def forward(self, x):
    return self.net(x)

class Preprocess(nn.Module):
    def __init__(self, bits=8, dequantize=True, input_type='image'):
        super().__init__()
        self.n_bits = bits
        self.n_bins = 2**bits
        self.dequantize = dequantize
        self.input_type = input_type
        
    def forward(self, x, train=True):

        if self.input_type !='image':
            return x
        y = x * 255
        if self.n_bits < 8:
            y =  (y / 2 **(8 - self.n_bits)).floor()
        
        if (train and self.dequantize):
             y = (y + torch.rand_like(y))
        y = y / self.n_bins - 0.5
        return y

    def inverse(self, y):
        if self.input_type !='image':
            return y
        x = (y + 0.5) * self.n_bins

        if self.n_bits < 8:
            x =  (x * 2 **(8 - self.n_bits)).floor()
        
        return  x/255
        
    
class Actnorm(nn.Module):
    def __init__(self, input_type='image'):
        super().__init__()
        self.log_scale = None
        self.bias = None
        self.input_size = None
        self.input_type = input_type
        self.register_buffer("initialised", torch.tensor(0, dtype=bool))
        
    def forward(self, x):
        if self.initialised == False:
            if self.input_type is 'image':
                self.bias = nn.Parameter(-x.mean(dim=(0,2,3), keepdim=True), requires_grad=True)
                self.log_scale = nn.Parameter(-torch.log(x.std(dim=(0,2,3), keepdim=True)), requires_grad=True)
                self.input_size = x.shape[2] * x.shape[3] # h * w
            
            elif self.input_type is 'factored':
                self.bias = nn.Parameter(-x.mean(dim=(0), keepdim=True), requires_grad=True)
                self.log_scale = nn.Parameter(-torch.log(x.std(dim=(0), keepdim=True)), requires_grad=True)
                self.input_size = 1
            
            self.initialised = ~self.initialised
            
        z = (x  + self.bias) * self.log_scale.exp()
        log_det_jacobian =  self.log_scale.exp().abs().sum().unsqueeze(0)  *self.input_size
        return z, log_det_jacobian
    
    def inverse(self, z):
        x = (z  * torch.exp(-self.log_scale)) - self.bias
        return x

class Inv1x1Conv(nn.Module):
    def __init__(self, in_channel, input_type='image'):
        super().__init__()
    
        self.input_type = input_type
        self.input_size = None
        
        self.conv = F.conv1d if input_type != 'image' else F.conv2d
    
        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(weight)
        w_p, w_l, w_u = torch.linalg.lu(q)
        w_s = torch.diag(w_u)
        w_u = torch.triu(w_u, 1)
        u_mask = torch.triu(torch.ones_like(w_u), 1)
        l_mask = u_mask.T
        


        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", u_mask)
        self.register_buffer("l_mask", l_mask)
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(abs(w_s)))
        self.w_u = nn.Parameter(w_u)

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight
    
    def forward(self, x):
        if self.input_size is None:
            self.input_size = x.shape[2] * x.shape[3] if self.input_type == 'image' else 1
        logdet = self.input_size * torch.sum(self.w_s)
        
        weight = self.calc_weight()
        
        if self.input_type != 'image':
            return x @ weight, logdet # Jacobian might be wrong because of the matmul instead of mul
        
        weight = weight.unsqueeze(2).unsqueeze(3)
        
        

        z = self.conv(x, weight)
        

        return z, logdet.unsqueeze(0)

    def inverse(self, z):
        weight_inv = self.calc_weight().inverse()
        
        if self.input_type != 'image':
            return z @ weight_inv
            
        
        weight_inv = weight_inv.unsqueeze(2).unsqueeze(3)
        
        return self.conv(z, weight_inv)
        
    
class AffineCoupling(nn.Module):
    def __init__(self, n_channels, input_type='image'):
      super().__init__()
      self.scale_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
      self.shift_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
      # self.net = ResNet(in_channels=n_channels//2)
      self.net = ConvBlock(in_channels= n_channels//2) if input_type== 'image' else MLP(in_dim=n_channels//2, out_dim=n_channels, hid_dim=n_channels)

    def forward(self, x):
      # Split the input in 2 channelwise
      x_a, x_b = x.chunk(2, dim=1)

      log_scale, shift = self.net(x_b).chunk(2, dim=1)
      log_scale = log_scale.tanh() * self.scale_scale + self.shift_scale

      y_a = x_a * log_scale.exp() + shift
      y_b = x_b

      y = torch.cat([y_a, y_b], dim=1)

      return y, log_scale.view(x.shape[0], -1).sum(-1)

    def inverse(self, y):
      y_a, y_b = y.chunk(2, dim=1)

      log_scale, shift = self.net(y_b).chunk(2, dim=1)
      log_scale = log_scale.tanh() * self.scale_scale + self.shift_scale

      x_a = (y_a - shift) * torch.exp(-log_scale)
      x_b = y_b

      x = torch.cat([x_a, x_b], dim=1)

      return x
    
class FlowStep(nn.Module):
    def __init__(self, n_channels , input_type):
        super().__init__()
        self.layers = nn.ModuleList([
            Actnorm(input_type=input_type),
            Inv1x1Conv(n_channels, input_type=input_type),
            AffineCoupling(n_channels, input_type=input_type),
        ])
        
    def forward(self, x):
        log_det_jacobian_total = torch.zeros(x.shape[0], device=x.device)
        z = x

        for layer in self.layers:
            z, log_det_jacobian = layer(z)
            log_det_jacobian_total += log_det_jacobian
        return z, log_det_jacobian_total.to(device=x.device)
    
    def inverse(self, z):
        x = z
        for layer in self.layers[::-1]:
            x = layer.inverse(x)
        return x
        
    
class FlowBlock(nn.Module):
    def __init__(self, n_channels, n_steps, split=True, input_type='image'):
        super().__init__()
        self.flow_steps = nn.ModuleList()
        self.input_type = input_type
        
        self.split = split
        if input_type=='image':
            for _ in range(n_steps):
                self.flow_steps.append(FlowStep(4*n_channels, input_type=input_type))
        else:
            for _ in range(n_steps):
                self.flow_steps.append(FlowStep(n_channels, input_type=input_type))
    def forward(self, x):
        z = x
        # Squeeze
        if self.input_type=='image':
            z = squeeze2d(z)
        
        log_det_jacobian_total = torch.zeros(x.shape[0], device=x.device)
        
        # Flow
        for flow in self.flow_steps:
            z, log_det_jacobian = flow(z)
            log_det_jacobian_total += log_det_jacobian
        
        # Split
        if self.split is True:
            z_i, h_i = z.chunk(2, dim=1)
            return z_i, h_i, log_det_jacobian_total
        
        return z, 0 ,  log_det_jacobian_total
    
    def inverse(self, z, z_prev=None):
        
        # Invert split
        if z_prev is not None:
            z = torch.cat([z_prev, z], dim=1)

        
        # Invert flows
        for flow in self.flow_steps[::-1]:
            z = flow.inverse(z)
        x = z
        # invert squeeze
        if self.input_type=='image':
            x = unsqueeze2d(x)
        return x
        
       
        
        
    
class Glow(nn.Module):
    def __init__(self, n_channels=3, n_steps=3, n_flow_blocks=3, dequantize=True, input_type='image', n_bits=5):
        super(Glow, self).__init__()

        self.preprocess = Preprocess(dequantize=dequantize, bits=n_bits, input_type=input_type)
        self.num_blocks = n_flow_blocks
        self.flow_layers = nn.ModuleList()
        self.input_type = input_type
        
        for layer in range(0, n_flow_blocks-1):
            self.flow_layers.append(FlowBlock(n_channels=n_channels*2**(layer),
                                              n_steps=n_steps,
                                              split=True,
                                              input_type=input_type))
        self.flow_layers.append(FlowBlock(n_channels=n_channels*2**(n_flow_blocks-1),
                                              n_steps=n_steps,
                                              split=False,
                                              input_type=input_type))
        
    def forward(self, x, train=True):
        h = x
        z_list = []
        
        
        h = self.preprocess(h, train=train)
            
        log_det_jacobian_total = torch.zeros(x.shape[0], device=x.device)
        for block in self.flow_layers:
            z, h, log_det_jacobian = block(h)
            z_list.append(z)
            log_det_jacobian_total += log_det_jacobian
        
        
        return z_list, log_det_jacobian_total
    
    def inverse(self, z_list):
        x = z_list.pop()
        
        for block in self.flow_layers[::-1]:
            z = z_list.pop() if block.split is True else None
            x = block.inverse(x,z)

        return self.preprocess.inverse(x)
    # Only do the Squeeze operation if we have an image
    def list_to_z(self, z_list):
        z_0 = z_list.pop()
        if self.input_type == 'image':
            z_0 = unsqueeze2d(z_0)
        

        while z_list:
            z_0 = torch.cat((z_list.pop(), z_0), dim=1) # 6
            if self.input_type == 'image':
                z_0 =  unsqueeze2d(z_0) # 3

        return z_0
    
    def z_to_list(self, input):
        z_list = []
        h = input

        for _ in range(len(self.flow_layers)-1):
            if self.input_type == 'image':
                h =  squeeze2d(h) # 3
            z, h = torch.chunk(h,2, dim=1) # 6
            z_list.append(z)
        
        if self.input_type == 'image':
            h = squeeze2d(h)
        z_list.append(h)
        return z_list
