import torch
import torch.nn as nn
import torch.optim as optim
from model import VAE
import torch.nn.functional as F
from model import VAE #, ConvVAE  


def vae_loss_kl(recon_x, x, mu, logvar):
    # We can use MSE for reconstruction
    # BCE: Treats each pixel as a binary outcome,
    #   which works well since MNIST digits are black and white: [0, 1]
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def vae_loss_mmd(recon_x, x, z_samples, prior_samples, sigma=1.0):

    def rbf_kernel(x, y, sigma=1.0):
        xx = x.unsqueeze(1)
        yy = y.unsqueeze(0)
        pairwise_sq_dists = torch.sum((xx - yy) ** 2, dim=2)
        return torch.exp(-pairwise_sq_dists / (2 * sigma ** 2))

    def mmd_loss(z_samples, prior_samples, sigma=1.0):
        prior_kernel = rbf_kernel(prior_samples, prior_samples, sigma=sigma).mean()
        z_kernel = rbf_kernel(z_samples, z_samples, sigma=sigma).mean()
        cross_kernel = rbf_kernel(z_samples, prior_samples, sigma=sigma).mean()
        return prior_kernel + z_kernel - 2 * cross_kernel

    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    mmd = mmd_loss(z_samples, prior_samples, sigma=sigma)
    return BCE + mmd




def train_vae(vae_model, train_loader, latent_dim, epochs, lr, beta, optimizer_type, debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 

    vae = vae_model
    # print(vae_model)

    # Choose optimizer
    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(vae.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(vae.parameters(), lr=lr)

    # --- Learning rate scheduler ---
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    vae.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(torch.float32).to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
                
            # --- DYNAMIC INPUT PREPARATION ---
            if vae_model.input_type == "flat":
                model_input = data.view(-1, 784)
            else: # "image"
                model_input = data

            BCE = nn.functional.binary_cross_entropy(recon_batch, model_input, reduction='sum') #data.view(-1, 784)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = BCE + beta * KLD   # ← β-VAE adjustment here
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        # Step the scheduler at the end of each epoch
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    return vae

# def train_vae(train_loader, latent_dim, epochs=10, debug=True):
#     vae = VAE(latent_dim=latent_dim)
    
#     optimizer = optim.AdamW(vae.parameters(), lr=1e-3)

#     vae.train()
#     for epoch in range(epochs):
#         train_loss = 0
#         for batch_idx, (data, _) in enumerate(train_loader):
#             data = data.to(torch.float32)
#             optimizer.zero_grad()
#             recon_batch, mu, logvar = vae(data)
#             loss = vae_loss_kl(recon_batch, data, mu, logvar)
#             loss.backward()
#             train_loss += loss.item()
#             optimizer.step()
#             if debug:
#                 return vae
#         print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}")

#     return vae

# if __name__ == '__main__':
#     vae_loss = vae_loss_kl