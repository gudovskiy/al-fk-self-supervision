import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3, s=1):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128//s, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128//s),
            nn.ReLU(True),
            nn.Conv2d(128//s, 256//s, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256//s),
            nn.ReLU(True),
            nn.Conv2d(256//s, 512//s, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512//s),
            nn.ReLU(True),
            nn.Conv2d(512//s, 1024//s, 4, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024//s),
            nn.ReLU(True),
            View((-1, 1024*2*2//s)),                                 # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024*2*2//s, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(1024*2*2//s, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024*4*4//s),                           # B, 1024*8*8
            View((-1, 1024//s, 4, 4)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024//s, 512//s, 4, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512//s),
            nn.ReLU(True),
            nn.ConvTranspose2d(512//s, 256//s, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256//s),
            nn.ReLU(True),
            nn.ConvTranspose2d(256//s, 128//s, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128//s),
            nn.ReLU(True),
            nn.ConvTranspose2d(128//s, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        if (x.size(2) != 32) or (x.size(3) != 32): # resize to VAE input due to missing ImageNet architecture
            x = F.interpolate(x, size=32) # , mode='bilinear', align_corners=False
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x, x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=32, s=1):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512//s),
            nn.ReLU(True),
            nn.Linear(512//s, 512//s),
            nn.ReLU(True),
            nn.Linear(512//s, 512//s),
            nn.ReLU(True),
            nn.Linear(512//s, 512//s),
            nn.ReLU(True),
            nn.Linear(512//s, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1,  10, 5, 1, 0)
        self.conv2 = nn.Conv2d(10, 20, 5, 1, 0)
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.mpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.mpool(x))
        x = self.conv2(x)
        x = self.relu(self.mpool(x))
        x = x.view(-1, 500)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        #
        return x

