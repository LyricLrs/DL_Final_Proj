
from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import random
from PIL import Image, ImageFilter, ImageEnhance

class VICRegLoss(nn.Module):
    def __init__(self, lambda_invariance=25, mu_variance=25, nu_covariance=1, gamma_inverse_log=1):
        super().__init__()
        self.lambda_invariance = lambda_invariance
        self.mu_variance = mu_variance
        self.nu_covariance = nu_covariance
        self.gamma_inverse_log = gamma_inverse_log

    def forward(self, pred_encs, target_encs):
        # Invariance term (MSE)
        invariance_loss = F.mse_loss(pred_encs, target_encs)

        # Variance term
        batch_var_pred = torch.var(pred_encs, dim=0) + 1e-4
        batch_var_target = torch.var(target_encs, dim=0) + 1e-4
        variance_loss = torch.mean(F.relu(1 - batch_var_pred)) + torch.mean(F.relu(1 - batch_var_target))

        # Covariance term
        pred_centered = pred_encs - pred_encs.mean(dim=0)
        pred_cov = (pred_centered.mT @ pred_centered) / (pred_encs.size(0) - 1)

        target_centered = target_encs - target_encs.mean(dim=0)
        target_cov = (target_centered.mT @ target_centered) / (target_encs.size(0) - 1)

        pred_cov_loss = torch.sum(torch.triu(pred_cov ** 2, diagonal=1)) / pred_encs.size(1)
        target_cov_loss = torch.sum(torch.triu(target_cov ** 2, diagonal=1)) / target_encs.size(1)
        covariance_loss = pred_cov_loss + target_cov_loss

        epsilon = 1e-6
        pred_std = torch.std(pred_encs, dim=0) + epsilon
        target_std = torch.std(target_encs, dim=0) + epsilon
        inverse_log_loss = -torch.mean(torch.log(pred_std)) - torch.mean(torch.log(target_std))

        # Combine losses
        total_loss = (
            self.lambda_invariance * invariance_loss +
            self.mu_variance * variance_loss +
            self.nu_covariance * covariance_loss
            + self.gamma_inverse_log * inverse_log_loss
        )

        print(f"Invariance loss: {invariance_loss.item():.4f}, Variance loss: {variance_loss.item():.4f}, Covariance loss: {covariance_loss.item():.4f}, Inverse log loss: {inverse_log_loss.item():.4f}")
        return total_loss
      

class GaussianBlur:
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))


class RandomBrightnessContrast:
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def __call__(self, img):
        brightness_factor = 1 + random.uniform(-self.brightness_limit, self.brightness_limit)
        contrast_factor = 1 + random.uniform(-self.contrast_limit, self.contrast_limit)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        return img


class RandomCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        width, height = img.size
        crop_width, crop_height = self.crop_size
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        return img.crop((left, top, left + crop_width, top + crop_height))


class ToTensor:
    def __call__(self, img):
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1)) / 255.0
        return torch.tensor(img)


def custom_transforms():
    return [
        RandomCrop((64, 64)), 
        GaussianBlur(radius=1.5), 
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2), 
        ToTensor(),  
    ]

def apply_transforms(img, transforms):
    for transform in transforms:
        img = transform(img)
    return img

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data 
        self.labels = labels  
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx] 
        label = self.labels[idx] 

        if self.transform:
            img = apply_transforms(img, self.transform)

        return img, label

def adjust_learning_rate(optimizer, epoch, warmup_steps, base_lr):
    if epoch < warmup_steps:
        lr = base_lr * (epoch + 1) / warmup_steps
    else:
        lr = scheduler.get_lr()[0]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): Number of epochs to wait for improvement.
            delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None or current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  

def visualize_latent_states(latent_states, title):
            """
            Visualize latent states using PCA.

            Args:
                latent_states (torch.Tensor): The latent states tensor of shape [B, T, D].
                title (str): Title for the PCA plot.
            """
            latent_states = latent_states.reshape(-1, latent_states.shape[-1]).cpu().detach().numpy() 

            print(f"[Visualize] Final shape for PCA: {latent_states.shape}") 

            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_states)

            plt.figure(figsize=(8, 6))
            plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5, label="Latent States")
            plt.title(title)
            plt.xlabel("PCA Dimension 1")
            plt.ylabel("PCA Dimension 2")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"latent_states_epoch_{title}.png") 
            plt.close()

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)


