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
#from vit import VisionTransformer

torch.cuda.empty_cache()

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)
  
class TransformerPredictor(nn.Module):
    """
    A simple Transformer-based sequence predictor that takes in a sequence of embeddings + actions.
    It uses Transformer encoders to process the entire sequence at once.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project back to the latent dimension if needed
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: [B, T, input_dim]
        x = self.input_proj(x)             # [B, T, hidden_dim]
        x = self.transformer_encoder(x)    # [B, T, hidden_dim]
        x = self.output_proj(x)            # [B, T, input_dim]
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        # 1. Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # 2. Class token & Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. Transformer encoder layers
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights (like timm does)
        self._init_weights()

    def _init_weights(self):
        # Implement appropriate weight initialization
        pass

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_embed(x) # [B, num_patches, embed_dim]
        B, N, D = x.shape

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+N, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        # Return the CLS token as representation
        return x[:, 0]

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
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
        # x: [B, N, D]
        x_res = x
        x = self.norm1(x)
        # MultiheadAttention expects [B, N, D], batch_first=True supports this directly
        x_attn, _ = self.attn(x, x, x)
        x = x_res + self.drop_path(x_attn)

        x_res = x
        x = self.norm2(x)
        x = x_res + self.drop_path(self.mlp(x))
        return x



class MockModel(nn.Module):
    def __init__(self, device="cuda", bs=32, n_steps=17, output_dim=None):
        super().__init__()
        self.device = device
        self.n_steps = n_steps

        # Custom ViT structure
        self.vit = VisionTransformer(
            img_size=64,  # Adjust to your input size
            patch_size=8,  # Size of each patch
            in_chans=2,  # Number of channels in your input
            embed_dim=1024,  # Embedding dimension
            depth=8,  # Number of transformer layers
            num_heads=8,  # Number of attention heads
            mlp_ratio=4.0,  # MLP expansion ratio
            drop_rate=0.1,  # Dropout rate
            attn_drop_rate=0.1,  # Attention dropout rate
        )

        # ViT embedding dimension
        self.repr_dim = 1024

        # MLP after ViT if needed
        self.mlp = build_mlp([self.repr_dim, 1024, 512])

        # BatchNorm and predictor
        final_repr_dim = 512
        self.bn = nn.BatchNorm1d(final_repr_dim)

        # Predictor: input dimension is final_repr_dim + 2 (for actions)
        self.predictor = TransformerPredictor(
            input_dim=final_repr_dim + 2,
            hidden_dim=1024,
            num_layers=4,
            n_heads=8,
        )

    def forward(self, states, actions=None, train=True):
        """
        Args:
            states: [B, T, C, H, W]
            actions: [B, T-1, 2]
        """
        B, T, C, H, W = states.shape

        # Flatten states for ViT
        states_flat = states.view(B * T, C, H, W)  # [B*T, C, H, W]
        encoded_states = self.vit(states_flat)  # [B*T, repr_dim]
        encoded_states = self.mlp(encoded_states)  # [B*T, final_repr_dim]
        latent_states = encoded_states.view(B, T, -1)  # [B, T, final_repr_dim]

        # Normalize latent states
        latent_states = latent_states.transpose(1, 2)  # [B, final_repr_dim, T]
        latent_states = self.bn(latent_states).transpose(1, 2)  # [B, T, final_repr_dim]

        # Continue with your existing predictor logic
        # The rest of the `forward` method remains unchanged
        if train:
            # Prepare sequences for the predictor
            input_seq = torch.cat([latent_states[:, :-1, :], actions], dim=-1)  # [B, T-1, final_repr_dim + 2]
            pred_seq = self.predictor(input_seq)  # [B, T-1, final_repr_dim + 2]
            pred_seq = pred_seq[..., :latent_states.size(-1)]  # [B, T-1, final_repr_dim]
            pred_encs = torch.cat([latent_states[:, 0:1, :], pred_seq], dim=1)  # [B, T, final_repr_dim]
            target_encs = latent_states
            return pred_encs, target_encs
        else:
            # Inference logic (unchanged)
            if actions is None:
                # If no actions are provided, return the latent state of the first time step
                return latent_states[:, 0, :]

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

    def train_model(self, dataset):
        """
        Train the model to align predicted latent representations with target representations.

        Args:
            dataset: A PyTorch DataLoader containing the training data.
        """
        learning_rate = 0.001
        num_epochs = 10     
        device = self.device   
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-3)

        # Warmup and cosine annealing with restarts
        warmup_steps = 5  # Number of warmup epochs
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        print(f"Starting model training for {num_epochs} epochs with lr={learning_rate}...")

        early_stopping = EarlyStopping(patience=10, delta=1e-4)
        epoch_losses = []
        vicreg_loss_fn = VICRegLoss(lambda_invariance=10, mu_variance=5, nu_covariance=0.01)

        for epoch in tqdm(range(num_epochs), desc="Model training epochs"):
            batch_losses = []  # Track losses per batch
            for batch_idx, batch in enumerate(tqdm(dataset, desc="Model training step")):
                states = batch.states.to(device)
                actions = batch.actions.to(device)

                pred_encs, target_encs = self.forward(states=states, actions=actions, train=True)

                # # Visualize latent states every few epochs
                if epoch % 2 == 0 and batch_idx == 0:  # For first batch every 2 epochs
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

                batch_losses.append(loss.item())  # Append batch loss

                for name, param in self.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: Grad mean {param.grad.mean():.4f}, Grad max {param.grad.abs().max():.4f}")

            avg_epoch_loss = sum(batch_losses) / len(batch_losses)  # Average epoch loss
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_epoch_loss:.8f}")
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.8f}")
            latent_var = torch.var(pred_encs.view(-1, pred_encs.size(-1)), dim=0)
            print(f"Epoch {epoch + 1}: Latent variance mean: {latent_var.mean().item():.4f}")

             # Step the scheduler
            scheduler.step()

            # Check early stopping
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

