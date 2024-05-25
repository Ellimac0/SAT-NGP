import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from encoding import get_encoder
from .renderer import NeRFRenderer


class Mish(torch.nn.Module):
    """
    Mish layer
    https://arxiv.org/vc/arxiv/papers/1908/1908.08681v1.pdf
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Siren(torch.nn.Module):
    """
    Siren layer
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        return torch.sin(self.w0 * input)

class Mapping(torch.nn.Module):
    def __init__(self, mapping_size, in_size, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = mapping_size
        self.in_channels = in_size
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = self.in_channels*(len(self.funcs)*self.N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, self.N_freqs-1, self.N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(self.N_freqs-1), self.N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
     
        out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

def custom_weights(m, name="X"):

    if name == "U":
        torch.nn.init.uniform_(m.weight)
    elif name == "X":
        torch.nn.init.xavier_uniform_(m.weight)
    elif name == "K":
        torch.nn.init.kaiming_uniform_(m.weight,
                               a=0, mode="fan_in",
                               nonlinearity="relu")
    elif name == "N":
        torch.nn.init.normal_(m.weight, mean=0, std=1)


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                    encoding="hashgrid",
                    encoding_dir="sphere_harmonics",
                    encoding_bg="hashgrid",
                    num_levels=16,
                    desired_resolution=-1,
                    level_dim=2,
                    base_resolution=16,
                    log2_hashmap_size=19,
                    num_layers=4, # 8
                    hidden_dim=128, # 256
                    degree=4,
                    bound=1,
                    skips=2, # 4
                    init_weight="U",  # U Xu Xn Ku Kn N O
                    nonlinearity="relu", # leaky_relu relu
                    gain_nonlinearity="relu",  #  relu leaky_relu tanh
                    mode="fan_in", # fan_in fan_out
                    siren=True,
                    t_embedding_dims=4,
                    **kwargs
                 ):

        super().__init__(bound,**kwargs)

        self.skips = [skips]
        self.t_embedding_dims = t_embedding_dims
        self.rgb_padding = 0.001
        # activation function
        if siren:
            nl = Siren()
        else:
            nl = Mish()

        self.init_weight = init_weight
        self.nonlinearity = nonlinearity
        self.gain_nonlinearity = gain_nonlinearity
        self.mode = mode
        self.encoding = encoding
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        if encoding == "frequency":
            self.encoder, self.in_dim = get_encoder("frequency")
        elif encoding == "hashgrid":
            desired_resolution = None if desired_resolution == 0 else desired_resolution  * bound
            self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=desired_resolution, level_dim=level_dim,
                                       num_levels=num_levels, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, align_corners=False)
        elif encoding == "mapping" and encoding_dir != "mapping":
            mapping_sizes=[10, 4]
            self.encoder , _ = [Mapping(map_sz, in_sz) for map_sz, in_sz in zip(mapping_sizes, [3,3])]
            self.in_dim  ,  _ = [2 * map_sz * in_sz for map_sz, in_sz in zip(mapping_sizes, [3,3])]


        # encoding
        if encoding_dir == "sphere_harmonics":

            self.encoder_dir, self.in_dim_dir = get_encoder(encoding="sphere_harmonics", degree=degree)

        elif encoding_dir == "frequency":
            self.encoder_dir, self.in_dim_dir = get_encoder(encoding="frequency")

        elif encoding != "mapping" and encoding_dir == "mapping":
            mapping_sizes=[10, 4]
            _ , self.encoder_dir = [Mapping(map_sz, in_sz) for map_sz, in_sz in zip(mapping_sizes, [3,3])]
            _  ,  self.in_dim_dir = [2 * map_sz * in_sz for map_sz, in_sz in zip(mapping_sizes, [3,3])]

        if  encoding == "mapping" and encoding_dir == "mapping" :
            mapping_sizes=[10, 4]
            self.encoder , self.encoder_dir = [Mapping(map_sz, in_sz) for map_sz, in_sz in zip(mapping_sizes, [3,3])]
            self.in_dim  ,  self.in_dim_dir = [2 * map_sz * in_sz for map_sz, in_sz in zip(mapping_sizes, [3,3])]



        # define the main network of fully connected layers, i.e. FC_NET
        fc_layers = []
        fc_layers.append(torch.nn.Linear(self.in_dim, hidden_dim))
        fc_layers.append(Siren(w0=30.0) if siren else nl)
        for i in range(1, num_layers):
            if i in self.skips:
                fc_layers.append(torch.nn.Linear(hidden_dim + self.in_dim, hidden_dim))
            else:
                fc_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            fc_layers.append(nl)
        self.fc_net = torch.nn.Sequential(*fc_layers)  # shared 8-layer structure that takes the encoded xyz vector

        # FC_NET output 1: volume density
        self.sigma_net = torch.nn.Sequential(torch.nn.Linear(hidden_dim, 1), torch.nn.Softplus())
        # FC_NET output 2: vector of features from the spatial coordinates
        self.feats_from_xyz = torch.nn.Linear(hidden_dim, hidden_dim) # No non-linearity here in the original paper
        self.color_net = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim // 2), nl,
                                                   torch.nn.Linear(hidden_dim // 2, 3), torch.nn.Sigmoid())

     
        sun_dir_in_size = self.in_dim_dir 
        sun_v_layers = []
        sun_v_layers.append(torch.nn.Linear(hidden_dim + sun_dir_in_size + 3, hidden_dim // 2))
        sun_v_layers.append(Siren() if siren else nl)
        for i in range(1, 3): # 3 de base
            sun_v_layers.append(torch.nn.Linear(hidden_dim // 2, hidden_dim // 2))
            sun_v_layers.append(nl)
        sun_v_layers.append(torch.nn.Linear(hidden_dim // 2, 1))
        sun_v_layers.append(torch.nn.Sigmoid())
        self.shade_net = torch.nn.Sequential(*sun_v_layers)

        self.sky_net = torch.nn.Sequential(
            torch.nn.Linear(3, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 3),
            torch.nn.Sigmoid(),
        )

        self.fc_net.apply(self.sine_init) 
        self.fc_net[0].apply(self.first_layer_sine_init) 
        self.shade_net.apply(self.sine_init)
        self.shade_net[0].apply(self.first_layer_sine_init)


        self.beta_from_xyz = torch.nn.Sequential(
            torch.nn.Linear(self.t_embedding_dims + hidden_dim, hidden_dim // 2),
            nl,
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.Softplus())



    def forward(self, input_xyz, input_dir=None, input_sun_dir=None, input_t=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        pos_xyz = input_xyz
        # compute shared features
        if self.encoding == "hashgrid":
            input_xyz = self.encoder(input_xyz, bound=self.bound) # on passe de dim 3 à 32
        else :
            input_xyz = self.encoder(input_xyz)

        xyz_ = input_xyz
        for i in range(self.num_layers):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = self.fc_net[2*i](xyz_)
            xyz_ = self.fc_net[2*i + 1](xyz_)
        shared_features = xyz_

        # compute volume density
        sigma = self.sigma_net(shared_features)
   
        # compute geofeat
        xyz_features = self.feats_from_xyz(shared_features)

        # compute color
        rgb = self.color_net(xyz_features)
        # improvement suggested by Jon Barron to help stability (same paper as soft+ suggestion)
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        input_sun_v_net = torch.cat([xyz_features, self.encoder_dir(input_sun_dir), pos_xyz], -1)
        sun_v = self.shade_net(input_sun_v_net)

        sky_color = self.sky_net(input_sun_dir)

        input_for_beta = torch.cat([xyz_features, input_t], -1)
        beta = self.beta_from_xyz(input_for_beta)

        return sigma, rgb , sun_v, sky_color, beta



    def density(self, input_xyz):
        # compute shared features
        if self.encoding == "hashgrid":
            input_xyz = self.encoder(input_xyz, bound=self.bound) # on passe de dim 3 à 32
        else :
            input_xyz = self.encoder(input_xyz)

        xyz_ = input_xyz
        for i in range(self.num_layers):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = self.fc_net[2*i](xyz_)
            xyz_ = self.fc_net[2*i + 1](xyz_)
        shared_features = xyz_

        # compute volume density
        sigma = self.sigma_net(shared_features)
   
        return {
            'sigma': sigma,
        }

    # optimizer utils
    def get_params(self, lr):

        params = [

            {'params': self.fc_net.parameters(), 'lr': lr},

            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},

            {'params': self.beta_from_xyz.parameters(), 'lr': lr},
            {'params': self.shade_net.parameters(), 'lr': lr},
            {'params': self.sky_net.parameters(), 'lr': lr},

        ]

        return params


    def sine_init(self, m):
        name = self.init_weight
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)

                if name=="U":
                    print("U")
                    # See supplement Sec. 1.5 for discussion of factor 30
                    m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))
                elif name=="Xu":
                    gain=nn.init.calculate_gain(self.gain_nonlinearity) # relu leaky_relu tanh
                    torch.nn.init.xavier_uniform_(m.weight, gain=gain)
                elif name=="Xn":
                    # print("Xn")
                    gain=nn.init.calculate_gain(self.gain_nonlinearity) # relu leaky_relu tanh
                    print(self.gain_nonlinearity)
                    torch.nn.init.xavier_normal_(m.weight, gain=gain)

                elif name=="Ku":
                    a = -1 / num_input if self.nonlinearity=="leaky_relu" else 0
                    print(a, self.mode, self.nonlinearity)
                    torch.nn.init.kaiming_uniform_(m.weight, a=a, mode=self.mode, nonlinearity=self.nonlinearity)
                elif name=="Kn":
                    a = -1 / num_input if self.nonlinearity=="leaky_relu" else 0
                    print(a, self.mode, self.nonlinearity)
                    torch.nn.init.kaiming_normal_(m.weight, a=a, mode=self.mode, nonlinearity=self.nonlinearity)

                elif name=="N":
                    torch.nn.init.normal_(m.weight, mean=0, std=1)
                elif name=="O":
                    torch.nn.init.orthogonal_(m.weight)

    def first_layer_sine_init(self, m):
        name = self.init_weight
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)

                if name=="U":
                    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
                    m.weight.uniform_(-1 / num_input, 1 / num_input)
                elif name=="Xu":
                    gain=nn.init.calculate_gain(self.gain_nonlinearity) # relu leaky_relu tanh
                    torch.nn.init.xavier_uniform_(m.weight, gain=gain)
                elif name=="Xn":
                    gain=nn.init.calculate_gain(self.gain_nonlinearity) # relu leaky_relu tanh
                    print(self.nonlinearity)
                    torch.nn.init.xavier_normal_(m.weight, gain=gain)

                elif name=="Ku":
                    a = -1 / num_input if self.nonlinearity=="leaky_relu" else 0
                    print(a, self.mode, self.nonlinearity)
                    torch.nn.init.kaiming_uniform_(m.weight, a=a, mode=self.mode, nonlinearity=self.nonlinearity)
                elif name=="Kn":
                    a = -1 / num_input if self.nonlinearity=="leaky_relu" else 0
                    print(a, self.mode, self.nonlinearity)
                    torch.nn.init.kaiming_normal_(m.weight, a=a, mode=self.mode, nonlinearity=self.nonlinearity)

                elif name=="N":
                    torch.nn.init.normal_(m.weight, mean=0, std=1)
                elif name=="O":
                    torch.nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain(self.gain_nonlinearity)) # relu leaky_relu tanh)
