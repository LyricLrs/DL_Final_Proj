import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # print(f"pred_encs shape: {pred_encs.shape}")  # Expected: [B, D]
        # print(f"pred_cov shape: {pred_cov.shape}")   # Expected: [D, D]

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