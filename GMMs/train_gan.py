from GAN import Generator, Discriminator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----- Toy Gaussian data -----
D = 2  # data dimension
data = torch.load("samples/data_gaussian_train.pt")
dataset = TensorDataset(data)
loader = DataLoader(dataset, batch_size=256, shuffle=True)


z_dim = 8
G = Generator(z_dim=z_dim, x_dim=D).to(device)
D_net = Discriminator(x_dim=D).to(device)


def rke_kernel_frobenius_squared(x, sigma=1.0):
    """
    x: (B, D) generated samples
    Return: ||K||_F^2 where K_ij = exp(-||xi - xj||^2 / (2 sigma^2))
    """
    # pairwise distances
    diff = x.unsqueeze(1) - x.unsqueeze(0)   # (B, B, D)
    dist_sq = (diff ** 2).sum(dim=2)        # (B, B)

    K = torch.exp(-dist_sq / (2 * sigma * sigma))  # Gaussian kernel

    frob_sq = torch.sum(K ** 2)  # ||K||_F^2
    return frob_sq


lr = 1e-4
beta1 = 0.5

opt_D = torch.optim.Adam(D_net.parameters(), lr=lr, betas=(beta1, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

bce = nn.BCELoss()

lambda_rke = 0.000000  # tune this
sigma_rke = 10.0   # kernel bandwidth (also tunable)


num_epochs = 200

for epoch in tqdm(range(num_epochs), desc="Training epochs"):
    # epoch_d_loss = 0.0
    # epoch_g_loss = 0.0
    # epoch_rke = 0.0

    for (x_real,) in loader:
        x_real = x_real.to(device)
        B = x_real.size(0)

        # -------------------------
        # 1. Train Discriminator
        # -------------------------
        z = torch.randn(B, z_dim, device=device)
        x_fake = G(z).detach()  # detach so G not updated here

        D_real = D_net(x_real)
        D_fake = D_net(x_fake)

        labels_real = torch.ones(B, 1, device=device)
        labels_fake = torch.zeros(B, 1, device=device)

        d_loss_real = bce(D_real, labels_real)
        d_loss_fake = bce(D_fake, labels_fake)
        d_loss = d_loss_real + d_loss_fake

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # -------------------------
        # 2. Train Generator (with RKE)
        # -------------------------
        z = torch.randn(B, z_dim, device=device)
        x_fake = G(z)

        D_fake = D_net(x_fake)
        g_adv_loss = bce(D_fake, labels_real)  # want D(fake) -> 1

        # RKE term on generated samples
        rke_term = rke_kernel_frobenius_squared(x_fake, sigma=sigma_rke)

        g_loss = g_adv_loss + lambda_rke * rke_term

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # epoch_d_loss += d_loss.item()
        # epoch_g_loss += g_loss.item()
        # epoch_rke += rke_term.item()

with torch.no_grad():
    z = torch.randn(5000, z_dim, device=device)
    gen_samples = G(z).cpu()  # for analysis / saving


torch.save(gen_samples, "samples/gan_gaussian_rke.pt")

