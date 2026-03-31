import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- DATA GENERATION --------------------

def generate_unit_circle_gaussians(n_components=10, n_samples=1000, std_dev=0.15, labels=True):
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    if labels:
        label_vectors = np.eye(n_components)

    samples = []
    all_labels = []
    for i, center in enumerate(centers):
        cluster = np.random.normal(loc=center, scale=std_dev, size=(n_samples, 2))
        samples.append(cluster)
        if labels:
            all_labels.extend([label_vectors[i]] * n_samples)

    samples = np.vstack(samples)
    if labels:
        return (
            torch.tensor(samples, dtype=torch.float32),
            torch.tensor(np.vstack(all_labels), dtype=torch.float32),
            torch.tensor(centers, dtype=torch.float32)
        )
    else:
        return (
            torch.tensor(samples, dtype=torch.float32),
            torch.tensor(centers, dtype=torch.float32)
        )

def generate_grid_gaussians(grid_size=5, spacing=1.0, std_dev=0.15, samples_per_gaussian=500, labels=False):
    centers = []
    coords = np.linspace(-(grid_size // 2), grid_size // 2, grid_size) * spacing

    for x in coords:
        for y in coords:
            centers.append([x, y])
    centers = np.array(centers)

    if labels:
        label_vectors = np.eye(centers.shape[0])

    all_samples = []
    all_labels = []
    for i, center in enumerate(centers):
        samples = np.random.normal(loc=center, scale=std_dev, size=(samples_per_gaussian, 2))
        all_samples.append(samples)
        if labels:
            all_labels.extend([label_vectors[i]] * samples_per_gaussian)
    
    samples = np.vstack(all_samples)
    if labels:
        return (
            torch.tensor(samples, dtype=torch.float32),
            torch.tensor(np.vstack(all_labels), dtype=torch.float32),
            torch.tensor(centers, dtype=torch.float32)
        )
    else:
        return (
            torch.tensor(samples, dtype=torch.float32),
            torch.tensor(centers, dtype=torch.float32)
        )



# ----- Toy Gaussian data (circle modes) -----
D = 10                 # or any variable dim you want
N_CENTERS = 1024
SAMPLES_PER_CENTER = 20

samples, labels, centers = generate_hypercube_gaussians(
    dim=D, n_centers=N_CENTERS, samples_per_center=SAMPLES_PER_CENTER,
    std_dev=0.04, low=-1.0, high=1.0,
    scramble=True, seed=0, device='cpu', return_labels=True
)
samples = samples.to(device)
labels = torch.from_numpy(labels).to(device)
num_classes = labels.shape[1]

# -------------------- CONDITIONAL MODEL --------------------

class ConditionalDenoisingNetwork(nn.Module):
    def __init__(self, num_classes):
        super(ConditionalDenoisingNetwork, self).__init__()
        self.label_embed = nn.Linear(num_classes, 64)
        self.network = nn.Sequential(
            nn.Linear(2 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, label):
        label_embed = self.label_embed(label)
        x = torch.cat([x, label_embed], dim=-1)
        return self.network(x)

conditional_model = ConditionalDenoisingNetwork(num_classes).to(device)

# -------------------- DIFFUSION FUNCTIONS --------------------

timesteps = 100
noise_schedule = torch.linspace(1e-4, 0.02, timesteps).to(device)

def ddim_forward(x_0, t, noise_schedule, eta=1.0):
    alpha_t = torch.sqrt(1 - noise_schedule[t]).unsqueeze(1)
    sigma_t = eta * torch.sqrt(noise_schedule[t]).unsqueeze(1)
    noise = torch.randn_like(x_0)
    x_t = alpha_t * x_0 + sigma_t * noise
    return x_t, noise

@torch.no_grad()
def conditional_ddim_sampling(label, n_samples=1000, eta=1.0):
    x = torch.randn(n_samples, 2).to(device)
    if label.ndim == 1 or label.shape[0] == 1:
        label = label.repeat(n_samples, 1)
    for t in reversed(range(0, timesteps, 10)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        alpha_t = torch.sqrt(1 - noise_schedule[t_tensor]).unsqueeze(1)
        sigma_t = eta * torch.sqrt(noise_schedule[t_tensor]).unsqueeze(1)
        predicted_noise = conditional_model(x, label)
        x = (x - sigma_t * predicted_noise) / alpha_t
    return x.cpu()



# -------------------- TRAINING LOOP --------------------

optimizer = optim.Adam(conditional_model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
batch_size = 256
epochs = 2000

for epoch in range(epochs):
    indices = torch.randint(0, samples.size(0), (batch_size,))
    x_0 = samples[indices]
    label = labels[indices]
    t = torch.randint(0, timesteps, (batch_size,), device=device)

    alpha_t = torch.sqrt(1 - noise_schedule[t]).unsqueeze(1)
    sigma_t = torch.sqrt(noise_schedule[t]).unsqueeze(1)
    noise = torch.randn_like(x_0)
    x_t = alpha_t * x_0 + sigma_t * noise

    pred_noise = conditional_model(x_t, label)
    loss = criterion(pred_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(conditional_model.parameters(), max_norm=1.0)
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

print("Training Complete")

# -------------------- VISUALIZATION --------------------

import matplotlib.pyplot as plt

with torch.no_grad():
    for label_idx in range(num_classes):
        label = torch.eye(num_classes)[label_idx:label_idx+1].to(device)
        samples = conditional_ddim_sampling(label, n_samples=1000)

        plt.figure(figsize=(6, 6))
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
        plt.title(f'Conditional Samples for Class {label_idx}')
        plt.grid(True)
        plt.axis('equal')
        plt.show()


# @torch.no_grad()
def conditional_ddim_sampling(label, n_samples=1000, eta=1.0):
    x = torch.randn(n_samples, 2).to(device)
    if label.ndim == 1 or label.shape[0] == 1:
        label = label.repeat(n_samples, 1)
    for t in reversed(range(0, timesteps, 10)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        alpha_t = torch.sqrt(1 - noise_schedule[t_tensor]).unsqueeze(1)
        sigma_t = eta * torch.sqrt(noise_schedule[t_tensor]).unsqueeze(1)
        predicted_noise = conditional_model(x, label)
        x = (x - sigma_t * predicted_noise) / alpha_t
    return x.cpu()

@torch.enable_grad()
def sequential_conditional_ddim_rke_sampling(n_samples=1000, eta=1.0, guidance_scale=5.0, rke_guide= None, num_classes=10):
    # Initialize RKE guidance

    # Set up the "text features" bank: one-hot labels
    label_bank = torch.eye(num_classes).to(device)
    # rke_guide.F_T = (label_bank.clone(), label_bank.clone())  # F_T = (features, memory)

    samples = []

    for i in range(n_samples):
        x = torch.randn(1, 2).to(device)

        # Randomly choose a class label for this sample
        class_idx = torch.randint(0, num_classes, (1,))
        features_text = label_bank[class_idx]

        for t in reversed(range(0, timesteps, 10)):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            alpha_t = torch.sqrt(1 - noise_schedule[t_tensor]).unsqueeze(1)
            sigma_t = eta * torch.sqrt(noise_schedule[t_tensor]).unsqueeze(1)

            predicted_noise = conditional_model(x, features_text)

            # Apply RKE guidance
            if guidance_scale > 0 and t > 20:
                features_ = x.detach().requires_grad_()
                if rke_guide.F_M is not None:
                    # print(f' fm: {rke_guide.F_M[0].shape}, ft: {rke_guide.F_T[0] if rke_guide.F_T is not None else 0}')
                    F_, M_ = rke_guide.get_F_M(
                        M=rke_guide.F_M[1], F=rke_guide.F_M[0],
                        f=features_, kernel=rke_guide.kernel, sigma=rke_guide.sigma[0]
                    )
                    F_T_, T_ = None, None
                    if 'cond' in rke_guide.algorithm:
                        F_T_ , T_ = rke_guide.get_F_M(
                            M=rke_guide.F_T[1], F=rke_guide.F_T[0],
                            f=features_text, kernel=rke_guide.kernel, sigma=rke_guide.sigma[1]
                        )

                    rank_fake = rke_guide.get_rank(
                        M=M_ / M_.shape[0], M_text=T_ / T_.shape[0] if 'cond' in rke_guide.algorithm else None,
                        feature_m=F_, feature_t=F_T_,
                        kernel=rke_guide.kernel,
                        sigma_image=rke_guide.sigma[0],
                        sigma_text=rke_guide.sigma[1]
                    )
                    rank_fake = torch.clamp(rank_fake, min=-20, max=20)
                    grads = torch.autograd.grad(rank_fake, features_)[0]
                    if torch.isnan(grads).any():
                        guided_noise = predicted_noise
                        print('skipping because grad is nan')
                    else:
                        guided_noise = predicted_noise + guidance_scale * grads
                    print(grads)
                    print(rank_fake)
                    
                else:
                    guided_noise = predicted_noise
            else:
                guided_noise = predicted_noise

            x = (x - sigma_t * guided_noise) / alpha_t


            if t == 0 and guidance_scale > 0:
                if rke_guide.F_M is None:
                    rke_guide.F_M = [features_.detach(), torch.mm(features_, features_.T)]
                    if rke_guide.F_T is None and 'cond' in rke_guide.algorithm:
                        rke_guide.F_T = [features_text.detach(), torch.mm(features_text, features_text.T)]
                else:
                    # update F if reached clean sample
                    rke_guide.F_M = rke_guide.get_F_M(M=rke_guide.F_M[1], F=rke_guide.F_M[0], f=features_.detach(), kernel=rke_guide.kernel, sigma=rke_guide.sigma[0])
                    if 'cond' in rke_guide.algorithm:
                        rke_guide.F_T = rke_guide.get_F_M(M=rke_guide.F_T[1], F=rke_guide.F_T[0], f=features_text.detach(), kernel=rke_guide.kernel,
                                        sigma=rke_guide.sigma[1])
                print(f' fm: {rke_guide.F_M[0].shape}, ft: {rke_guide.F_T[0].shape if rke_guide.F_T is not None else 0}, {rke_guide.algorithm}, {"cond" in rke_guide.algorithm}')
                # print(features_text.shape)
                # rke_guide.update_feature_matrices(features_, features_text)

        samples.append(x.cpu())

    return torch.cat(samples, dim=0)


@torch.enable_grad()
def sequential_conditional_ddim_particle_guide_sampling(
    n_samples=1000,
    eta=1.0,
    kernel_bandwidth=0.25,
    diversity_coeff=1.0,
    power=2.0
):
    num_classes = 10
    label_bank = torch.eye(num_classes).to(device)
    samples = []

    for i in range(n_samples):
        x = torch.randn(1, 2).to(device)
        class_idx = torch.randint(0, num_classes, (1,))
        label = label_bank[class_idx]

        previous_latents = []

        for t in reversed(range(0, timesteps, 10)):
            t_tensor = torch.full((1,), t, dtype=torch.long, device=device)
            alpha_t = torch.sqrt(1 - noise_schedule[t_tensor]).unsqueeze(1)
            sigma_t = eta * torch.sqrt(noise_schedule[t_tensor]).unsqueeze(1)

            # Predict noise
            predicted_noise = conditional_model(x, label)

            # Apply particle-based repulsion guidance
            if t > 20 and len(previous_latents) > 0:
                x_ = x.detach().requires_grad_(True)
                all_latents = torch.cat(previous_latents + [x_], dim=0)  # [N+1, 2]

                # Compute RBF kernel-based repulsion
                diff = x_.unsqueeze(1) - all_latents.unsqueeze(0)         # [1, N+1, 2]
                distance = torch.norm(diff, dim=-1, keepdim=True)         # [1, N+1, 1]

                h = (distance.median(dim=1, keepdim=True)[0]) ** power / torch.log(
                    torch.tensor(all_latents.shape[0], dtype=torch.float32, device=device)
                )

                weights = torch.exp(-distance ** power / kernel_bandwidth)
                grad_phi = 2 * weights * diff / h                         # [1, N+1, 2]
                repulsion = grad_phi.sum(dim=1)                          # [1, 2]

                # Add repulsion directly
                predicted_noise = predicted_noise + diversity_coeff * repulsion

            # DDIM update
            x = (x - sigma_t * predicted_noise) / alpha_t

            previous_latents.append(x.detach())

        samples.append(x.cpu())

    return torch.cat(samples, dim=0)


with torch.no_grad():
    plt.figure(figsize=(6, 6))
    n_per_class = 300
    guidance_scale = 0
    # Generate original samples for reference (for plotting)
    original_samples, original_labels, gaussian_centers = generate_unit_circle_gaussians(
        n_components=num_classes, n_samples=n_per_class, std_dev=0.07, labels=True
    )
    # original_samples = original_samples.cpu()
    
    rke_guide = RKEGuidedSampling(
        algorithm='cond-rke',
        max_bank_size = 20,
        kernel='gaussian',
        sigma_image=0.25,
        sigma_text=0.5,  # Add this if using text guidance
        use_latents_for_guidance=True
    )

    # Plot original data

    # Plot samples per class with conditional RKE-DDIM
    all_samples = []

    for class_idx in range(num_classes):
        label = torch.eye(num_classes)[class_idx:class_idx+1].to(device)
        generated = sequential_conditional_ddim_rke_sampling(
            n_samples=n_per_class,
            eta=1,
            guidance_scale=guidance_scale,
            rke_guide=rke_guide,
            num_classes=16
            # label=label  # Ensure your sampling function accepts this
        )
        # generated = conditional_ddim_sampling(label, n_samples=100)
        all_samples.append(generated.cpu().numpy())

    # Optionally add Gaussian centers
    # plt.scatter(gaussian_centers[:, 0], gaussian_centers[:, 1], marker='x', s=100, c='red', label='Gaussian Centers')
    all_samples = np.concatenate(all_samples)
    plt.figure(figsize=(8, 8))
    plt.scatter(original_samples[:, 0], original_samples[:, 1], s=10, alpha=0.8, label='Original Data', c='blue')

    plt.axis('equal')
    # plt.xlim([-1.3, 1.3])
    # plt.ylim([-1.3, 1.3])
    # plt.title("DDIM", fontsize=25)


    # Plot all samples in the same color (e.g., blue)

    plt.scatter(
        all_samples[:, 0], 
        all_samples[:, 1], 
        s=20, 
        alpha=0.8, 
        color='red',  # Single color for all points
        label='Generated Data'
    )
    # plt.legend(loc='upper left', fontsize=15)
    # print(calculate_scores(torch.tensor(original_samples), all_samples))

    plt.show()
