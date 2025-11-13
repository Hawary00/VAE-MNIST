import yaml
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import train_vae
from generate_visualise import inference_latent_samples, visualize_latent_space_mnist
from model import VAE, ConvVAE
import numpy as np
from generate_visualise import inference_latent_samples, visualize_latent_space_mnist
from generate_visualise import sample_images, interpolate_pts, build_digit_dict

# --- Step 1: Load configuration from config.yaml ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Step 2: Prepare dataset ---
dataset_cfg = config["dataset"]
transform = transforms.Compose([transforms.ToTensor()])

if dataset_cfg["name"].lower() == "mnist":
    train_dataset = datasets.MNIST(
        root=dataset_cfg["data_dir"],
        train=True,
        download=True,
        transform=transform,
    )
else:
    raise ValueError(f"Unsupported dataset: {dataset_cfg['name']}")

train_loader = DataLoader(
    train_dataset,
    batch_size=dataset_cfg["batch_size"],
    shuffle=dataset_cfg["shuffle"],
    num_workers=dataset_cfg["num_workers"],
)

# --- Step 3: Choose model type ---
model_cfg = config["model"]
if model_cfg["type"].lower() == "convvae":
    model_class = ConvVAE
else:
    model_class = VAE

vae_model = model_class(latent_dim=model_cfg["latent_dim"])

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_model = vae_model.to(device)

# --- Step 4: Training configuration ---
train_cfg = config["training"]
vae_model = train_vae(
    vae_model,
    train_loader=train_loader,
    latent_dim=model_cfg["latent_dim"],
    epochs=train_cfg["epochs"],
    lr=train_cfg["learning_rate"],
    beta=train_cfg["beta"],
    optimizer_type=train_cfg["optimizer"],
)

model_type = config["model"]["type"]  # "VAE" or "ConvVAE"
results_dir = f"results/{model_type}"
import os
os.makedirs(results_dir, exist_ok=True)

# --- Step 5: Visualization ---
if config["visualization"]["plot_latent_space"]:
    latent_samples, labels = inference_latent_samples(train_loader, vae_model)
    visualize_latent_space_mnist(latent_samples, labels, 
                                 save_path=f"{results_dir}/{model_type}_latent_space.png")


# Visualize decoded samples or interpolations
z_points = np.random.normal(0, 1, size=(10, model_cfg["latent_dim"]))
sample_images(z_points, vae_model, 
              save_path=f"{results_dir}/{model_type}_sample_images.png")

# Interpolation between two digits
digit_dict = build_digit_dict(train_loader)
interpolate_pts(digit_dict, vae_model, model_name=model_type)




# import yaml
# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from train import train_vae
# from gurante_visualize import inference_latent_samples, visualize_latent_space_mnist
# from generate_visualise import sample_images, interpolate_pts, build_digit_dict

# # Sample from the latent space
# import numpy as np

# # Load configuration from config.yaml ---
# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)


# # --- Step 1: Load MNIST dataset ---
# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# # --- Step 2: Set model parameters ---
# latent_dim = 2      # 2D latent space for visualization
# epochs = 100       # you can increase later

# # --- Step 3: Train the VAE ---
# vae_model = train_vae(train_loader, latent_dim, epochs)

# # --- Step 4: Extract latent samples for visualization ---
# latent_samples, labels = inference_latent_samples(train_loader)

# # --- Step 5: Plot latent space (2D) ---
# visualize_latent_space_mnist(latent_samples, labels)




# # Visualize decoded samples or interpolations
# z_points = np.random.normal(0, 1, size=(10, latent_dim))
# sample_images(z_points)

# # Interpolation between two digits
# digit_dict = build_digit_dict(train_loader)
# interpolate_pts(digit_dict, vae_model)
