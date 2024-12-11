from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from tqdm.auto import tqdm
from resnet import resnet34, resnet50
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from helper import *

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)
  
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.input_proj(x) 
        pos_embed = self.pos_embedding[:, :T, :] 
        x = x + pos_embed  
        x = self.transformer_encoder(x)  
        x = self.output_proj(x) 
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        pass

    def forward(self, x):
        x = self.patch_embed(x)
        B, N, D = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)  
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  
        x = x.flatten(2).transpose(1, 2) 
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        x_attn, _ = self.attn(x, x, x)
        x = x_res + self.drop_path(x_attn)

        x_res = x
        x = self.norm2(x)
        x = x_res + self.drop_path(self.mlp(x))
        return x

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, T, D] - Batch, Temporal Steps, Embedding Dim
        Returns:
            output: [B, T, D] - Attention-enhanced latent states
            weights: [B, T, T] - Attention weights for each time step
        """
        query = self.query_proj(x)  
        key = self.key_proj(x)     
        value = self.value_proj(x) 

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)  
        attn_weights = self.softmax(attn_scores) 

        weighted_values = torch.matmul(attn_weights, value) 
        output = self.dropout(weighted_values)

        return output, attn_weights

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # List of images
        self.labels = labels  # Corresponding labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]  # Retrieve the image
        label = self.labels[idx]  # Retrieve the label

        # Apply the transform pipeline if specified
        if self.transform:
            img = apply_transforms(img, self.transform)

        return img, label

class MockModel(nn.Module):
    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=None):
        super().__init__()
        self.device = device
        self.n_steps = n_steps

        self.vit = VisionTransformer(
            img_size=64,  
            patch_size=8,  
            in_chans=2,  
            embed_dim=512,  
            depth=8, 
            num_heads=8,  
            mlp_ratio=4.0, 
            drop_rate=0.1,  
            attn_drop_rate=0.1,  
        )

        self.repr_dim = 512

        self.mlp = build_mlp([self.repr_dim, 512, 512])

        final_repr_dim = 512
        self.bn = nn.BatchNorm1d(final_repr_dim)

        self.predictor = TransformerPredictor(
            input_dim=final_repr_dim + 2,
            hidden_dim=512,
            num_layers=4,
            n_heads=8,
        )

        self.input_projection_layer = nn.Linear(258, 514) 

        self.temporal_attention = TemporalAttention(input_dim=512, hidden_dim=256) 

    def forward(self, states, actions=None, train=True):
        """
        Args:
            states: [B, T, C, H, W]
            actions: [B, T-1, 2]
        """
        B, T, C, H, W = states.shape

        # Use raw states (skip masking)
        # masked_states = self.apply_mask(states)  # Masking temporarily disabled
        masked_states = states  # Directly use the raw states

        # Flatten states for ViT
        states_flat = masked_states.view(B * T, C, H, W)  # [B*T, C, H, W]
        encoded_states = self.vit(states_flat)  # [B*T, repr_dim]
        encoded_states = self.mlp(encoded_states)  # [B*T, final_repr_dim]
        latent_states = encoded_states.view(B, T, -1)  # [B, T, final_repr_dim]

        latent_states = latent_states.transpose(1, 2)  # [B, final_repr_dim, T]
        latent_states = self.bn(latent_states).transpose(1, 2)  # [B, T, final_repr_dim]

        # Apply temporal attention
        attn_output, attn_weights = self.temporal_attention(latent_states)  # [B, T, final_repr_dim]

        if train:
            # Use the attention-enhanced output for prediction
            input_seq = torch.cat([attn_output[:, :-1, :], actions], dim=-1)  # [B, T-1, final_repr_dim + 2]
            input_seq = self.input_projection_layer(input_seq)
            pred_seq = self.predictor(input_seq)  # [B, T-1, final_repr_dim + 2]
            pred_seq = pred_seq[..., :latent_states.size(-1)]  # [B, T-1, final_repr_dim]
            pred_encs = torch.cat([latent_states[:, 0:1, :], pred_seq], dim=1)  # [B, T, final_repr_dim]
            target_encs = latent_states
            return pred_encs, target_encs
        else:
            if actions is None:
                return latent_states[:, 0, :]  # Return the latent state of the first time step

            preds = [latent_states[:, 0, :]]
            cur_state = latent_states[:, 0, :].unsqueeze(1)
            for t in range(actions.shape[1]):
                step_input = torch.cat([cur_state, actions[:, t:t+1, :]], dim=-1)
                pred_step = self.predictor(step_input)
                pred_step = pred_step[..., :latent_states.size(-1)]
                preds.append(pred_step.squeeze(1))
                cur_state = pred_step
            preds = torch.stack(preds, dim=1)
            return preds

    def apply_mask(self, states, epoch=None, max_epochs=10):
        """
        Dynamically apply masking to focus on trajectories with increasing emphasis over epochs.
        Args:
            states: [B, T, C, H, W]
            epoch: Current epoch (for dynamic emphasis)
            max_epochs: Total epochs (for scaling emphasis)
        """
        mask = self.generate_trajectory_mask(states)  # [B, T, 1, H, W]
        trajectory_weight = min(1.0, (epoch or 0) / max_epochs)  # Scale emphasis over epochs
        masked_states = states * (trajectory_weight * mask + (1 - trajectory_weight) * 0.5)
        return states

    def generate_trajectory_mask(self, states):
        """
        Generate a mask that highlights trajectories (first channel)
        and de-emphasizes other regions (second channel).
        
        Args:
            states: [B, T, C, H, W]
        
        Returns:
            A mask with the same temporal and spatial dimensions as `states`.
        """
        B, T, C, H, W = states.shape

        # Separate channels
        object_channel = states[:, :, 0:1, :, :]  # First channel [B, T, 1, H, W]
        env_channel = states[:, :, 1:2, :, :]  # Second channel [B, T, 1, H, W]

        # Compute differences for the object channel
        diff_object = torch.zeros_like(object_channel)  # Placeholder to match original shape
        diff_object[:, 1:] = object_channel[:, 1:] - object_channel[:, :-1]  # [B, T, 1, H, W]
        mask_object = (diff_object.abs() > 0.1).float()  # Threshold to detect significant changes

        # Compute differences for the environment channel
        diff_env = torch.zeros_like(env_channel)  # Placeholder to match original shape
        diff_env[:, 1:] = env_channel[:, 1:] - env_channel[:, :-1]  # [B, T, 1, H, W]
        mask_env = (diff_env.abs() > 0.1).float()  # Similar threshold for changes

        # Combine the masks with emphasis on the object channel
        combined_mask = 0.5 * mask_object + 0.5 * mask_env  # Weighted emphasis on the object channel
        assert combined_mask.shape == (B, T, 1, H, W), f"Mask shape {combined_mask.shape} does not match input shape {states.shape}"

        return combined_mask


    def train_model(self, dataset):
        """
        Train the model to align predicted latent representations with target representations.

        Args:
            dataset: A PyTorch DataLoader containing the training data.
        """
        learning_rate = 0.0005
        num_epochs = 10     
        device = self.device   
        self.train()

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        # warmup_steps = 5  
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        warmup_steps = 1000  # Number of warmup steps (experiment with values)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, total_steps=num_epochs * len(dataset), anneal_strategy='linear')

        print(f"Starting model training for {num_epochs} epochs with lr={learning_rate}...")

        early_stopping = EarlyStopping(patience=10, delta=1e-4)
        epoch_losses = []
        vicreg_loss_fn = VICRegLoss(
            lambda_invariance=25.0,
            mu_variance=15.0,
            nu_covariance=0.01,
            gamma_inverse_log=10.0  # Adjust weight for inverse log term
        )

        for epoch in tqdm(range(num_epochs), desc="Model training epochs"):
            batch_losses = []  # Track losses per batch
            for batch_idx, batch in enumerate(tqdm(dataset, desc="Model training step")):
                states = batch.states.to(device)
                actions = batch.actions.to(device)

                pred_encs, target_encs = self.forward(states=states, actions=actions, train=True)

                # # Visualize latent states every few epochs
                if batch_idx == 0:  
                    print(f"[Train Debug] Visualizing latent states at epoch {epoch}")
                    print(f"Type of pred_encs: {type(pred_encs)}")  # Ensure it's a tensor
                    visualize_latent_states(target_encs,f'target: {epoch}')
                    visualize_latent_states(pred_encs,f'pred: {epoch}')

                #loss = torch.nn.functional.mse_loss(pred_encs, target_encs)
                loss = vicreg_loss_fn(pred_encs, target_encs)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()

                batch_losses.append(loss.item())

                # for name, param in self.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: Grad mean {param.grad.mean():.4f}, Grad max {param.grad.abs().max():.4f}")

            avg_epoch_loss = sum(batch_losses) / len(batch_losses)  # Average epoch loss
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_epoch_loss:.8f}")
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.8f}")
            latent_var = torch.var(pred_encs.view(-1, pred_encs.size(-1)), dim=0)
            print(f"Epoch {epoch + 1}: Latent variance mean: {latent_var.mean().item():.4f}")

            scheduler.step()

            early_stopping(avg_epoch_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

        print("Model training complete.")

class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

