
from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib


class VICRegLoss(nn.Module):
    def __init__(self, lambda_invariance=25, mu_variance=25, nu_covariance=1):
        super().__init__()
        self.lambda_invariance = lambda_invariance
        self.mu_variance = mu_variance
        self.nu_covariance = nu_covariance

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

        # Combine losses
        total_loss = (
            self.lambda_invariance * invariance_loss +
            self.mu_variance * variance_loss +
            self.nu_covariance * covariance_loss
        )
        return total_loss


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
        # If it's the first epoch or training loss improves
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
            latent_states = latent_states.reshape(-1, latent_states.shape[-1]).cpu().detach().numpy()  # [B * T, D]

            print(f"[Visualize] Final shape for PCA: {latent_states.shape}")  # Debugging

            # Apply PCA to reduce to 2 dimensions
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_states)

            # Plot the PCA projections
            plt.figure(figsize=(8, 6))
            plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5, label="Latent States")
            plt.title(title)
            plt.xlabel("PCA Dimension 1")
            plt.ylabel("PCA Dimension 2")
            plt.legend()
            plt.grid(True)
            # Save plot as a file
            plt.savefig(f"latent_states_epoch_{title}.png")  # Adjust filename as needed
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


