
from train import train_vae
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def inference_latent_samples(data_loader, vae_model):
    # Given dataset, extract latent representations
    device = next(vae_model.parameters()).device
    vae_model.eval()
    with torch.no_grad():
        latent_samples = []
        all_labels = []
        for batch_idx, (data, labels) in enumerate(data_loader):
            data = data.to(torch.float32).to(device)

            # --- DYNAMIC INPUT PREPARATION ---
            if vae_model.input_type == "flat":
                model_input = data.view(-1, 784)
            else: # "image"
                model_input = data

            mu, logvar = vae_model.encode(model_input)#data.view(-1, 784)
            z = vae_model.reparameterize(mu, logvar)
            latent_samples.append(z)

            batch_labels = labels.cpu().numpy().tolist()
            all_labels.extend(batch_labels)
        latent_samples = torch.cat(latent_samples).cpu().numpy()

    return latent_samples, all_labels

def plot_pdf_2dlatent(latent_samples):
    # Calculate and plot the PDF of the latent space
    x = latent_samples[:, 0]
    y = latent_samples[:, 1]
    kde = gaussian_kde([x, y])  # scipy.stats

    # from sklearn.neighbors import KernelDensity
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(latent_samples.cpu().numpy())

    xmin, xmax = x.min() - 1, x.max() + 1
    ymin, ymax = y.min() - 1, y.max() + 1
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    pdf = np.reshape(kde(positions).T, xx.shape)

    # Plot the PDF
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, pdf, levels=20, cmap="viridis")
    plt.colorbar(label="Density")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Estimated PDF of the Latent Space")
    plt.show()

    return kde


def visualize_latent_space_mnist(z, labels, save_path=None):
    # Convert z to two separate lists for x and y coordinates
    x_vals = [point[0] for point in z]
    y_vals = [point[1] for point in z]

    # Set up the plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(x_vals, y_vals, c=labels, cmap='tab10', alpha=0.6)

    # Adding a color bar and legend to interpret colors
    plt.colorbar(scatter, ticks=range(10), label="Digit Label")
    plt.title("2D Latent Space Visualization with Digit Labels")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved latent space plot to {save_path}")
    plt.show()


def sample_images(z_samples, vae_model, title = None, save_path=None):
    device = next(vae_model.parameters()).device
    num_samples = z_samples.shape[0]
    sample_images = []

    with torch.no_grad():
        for point in z_samples:
            point_tensor = torch.tensor(point, dtype=torch.float32).unsqueeze(0).to(device)
            generated_img = vae_model.decode(point_tensor).view(28, 28).cpu().numpy()
            sample_images.append(generated_img)

    # Plot the generated images
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, img in enumerate(sample_images):
        axes[i].imshow(img, cmap="gray")
        axes[i].axis('off')

    if title is None:
        title = "Generated Images from Sampled Latent Points"
    plt.suptitle(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved sampled images to {save_path}")

    plt.show()
    return sample_images


def interpolate_pts(digit_dict, vae_model, model_name, num_steps=4, save_dir=None):
    device = next(vae_model.parameters()).device
    vae_model.eval()

    with torch.no_grad():
        for key, pair_lst in digit_dict.items():
            txt = f'interpolated from label {key}'
            print(txt)
            for batch_data in pair_lst:
                batch_data = batch_data.to(torch.float32).to(device) 

                # --- DYNAMIC INPUT PREPARATION ---
                if vae_model.input_type == "flat":
                    model_input = batch_data.view(-1, 784)
                else: # "image"
                    model_input = batch_data

                mu, logvar = vae_model.encode(model_input)#batch_data.view(-1, 784)
                # we often ignore the variance during interpolation in the latent space
                # to maintain a smooth, deterministic transition between two specific points
                # without introducing additional randomness

                z1 = mu[0].cpu().numpy()    # one mean per dimension
                z2 = mu[1].cpu().numpy()
                latent_dim = z1.shape[0]

                # Generate interpolated latent points
                interpolated_latents = []
                for alpha in np.linspace(0, 1, num_steps):
                    z_interp = (1 - alpha) * z1 + alpha * z2  # Linear interpolation
                    tensor = torch.tensor(z_interp, dtype=torch.float32).reshape(1, latent_dim).to(device)
                    interpolated_latents.append(tensor)

                # interpolated_latents = np.concatenate(interpolated_latents)
                interpolated_latents = torch.cat(interpolated_latents, dim=0)

                # if save_dir:
                #     save_path = f"{save_dir}/interpolation_label{label}_pair{pair_idx}.png"

                sample_images(interpolated_latents.cpu().numpy(), vae_model, title=txt)


def build_digit_dict(data_loader, pairs=1):
    # Create dictionary from digit, to list of pair of items
    digit_dict = {i: [] for i in range(10)}


    for images, labels in data_loader:
        for i in range(len(labels)):
            label = labels[i].item()
            image = images[i]
            if len(digit_dict[label]) == pairs * 2:
                continue    # let's keep a few pairs
            digit_dict[label].append(image)

    # Stack each pair of consecutive elements in each digit's list
    for digit in digit_dict:
        stacked_pairs = []
        images_list = digit_dict[digit]

        # Stack each pair of consecutive images
        for i in range(0, len(images_list) - 1, 2):  # Skip by 2 to get consecutive pairs
            stacked_image = torch.stack([images_list[i], images_list[i + 1]], dim=0)  # Stack along a new dimension
            stacked_pairs.append(stacked_image)

        # Replace the original list with the stacked pairs list
        digit_dict[digit] = stacked_pairs

    return digit_dict




def compute_p_x(vae_model, x, num_samples=1000):
    """
    Approximates the marginal probability p(x) for a given input x
    using Importance Sampling on a trained VAE.

    Parameters:
    - model: Trained VAE model with encoder and decoder methods.
    - x: Input data sample (tensor) for which we want to compute p(x).
    - num_samples: Number of samples to use in Importance Sampling.

    Returns:
    - Approximate value of p(x).
    """
    # Ensure x is batched and moved to the correct device
    x = x.unsqueeze(0) if x.dim() == 3 else x  # Add batch dimension if not present

    x = x.to(torch.float32)

    with torch.no_grad():
        x = x.view(-1, 784)
        mu, log_var = vae_model.encode(x)
        std = torch.exp(0.5 * log_var)

        log_p_x_estimates = []
        for _ in range(num_samples):    # Sample z from q(z|x) multiple times
            z = mu + std * torch.randn_like(std)    # Reparameterization trick
            x_recon_mu = vae_model.decode(z)
            # Compute log p(x|z) assuming Gaussian likelihood with fixed variance
            log_p_x_given_z = -F.mse_loss(x_recon_mu, x, reduction='none').\
                view(x.shape[0], -1).sum(dim=1)

            # Compute log q(z|x) and log p(z) for the Importance Sampling weight
            log_q_z_given_x = -0.5 * ((z - mu) ** 2 / std ** 2 + log_var +
                                      torch.log(torch.tensor(2 * torch.pi))).sum(dim=1)
            log_p_z = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * torch.pi))).sum(dim=1)

            # Calculate log p(x|z) + log p(z) - log q(z|x)
            log_p_x_estimate = log_p_x_given_z + log_p_z - log_q_z_given_x
            log_p_x_estimates.append(log_p_x_estimate)

        # Average over the sampled log p(x) estimates for final log p(x)
        log_p_x = torch.logsumexp(torch.stack(log_p_x_estimates), dim=0) \
                  - torch.log(torch.tensor(num_samples, dtype=torch.float))

    return log_p_x.exp().item()  # Return the estimated p(x)

# if __name__ == '__main__':

#     vae_model = train_vae(train_loader, latent_dim, epochs)
