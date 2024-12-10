import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from dataset import create_wall_dataloader
import numpy as np
from models import MockModel
import time

from normalizer import Normalizer

normalizer = Normalizer()

def load_data(device):
    data_path = "/scratch/DL24FA"

    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
    )

    return train_ds


def contrastive_loss(predicted_states, target_states, temperature=0.1):
    """
    Computes contrastive loss between predicted states and target states.

    Args:
        predicted_states: Tensor of shape [B * T, repr_dim].
        target_states: Tensor of shape [B * T, repr_dim].
        temperature: Temperature parameter for contrastive scaling.
    
    Returns:
        Contrastive loss value.
    """
    # Normalize embeddings
    predicted_states = F.normalize(predicted_states, dim=-1)  # [B * T, repr_dim]
    target_states = F.normalize(target_states, dim=-1)  # [B * T, repr_dim]
    
    # Compute similarity logits
    logits = torch.mm(predicted_states, target_states.T) / temperature  # [B*T, B*T]
    
    # Ground-truth labels for positive pairs (diagonal alignment)
    labels = torch.arange(logits.size(0)).to(logits.device)
    
    # Cross-entropy loss
    return F.cross_entropy(logits, labels)

def variance_regularization(representations, eps=1e-4):
    std = torch.std(representations, dim=0)
    return torch.mean(F.relu(eps - std))

def covariance_regularization(representations):
    n, d = representations.size()
    cov = (representations.T @ representations) / n  # Covariance matrix
    off_diag = cov - torch.diag(torch.diag(cov))     # Remove diagonal elements
    return torch.sum(off_diag ** 2)    

@torch.no_grad()
def update_target_encoder(encoder, target_encoder, momentum=0.99):
    for param, target_param in zip(encoder.parameters(), target_encoder.parameters()):
        target_param.data = momentum * target_param.data + (1 - momentum) * param.data

def train_model(
    model,
    train_loader,
    target_encoder,
    epochs,
    optimizer,
    device,
    distance_fn=torch.nn.MSELoss(),
    save_path="best_model.pth"
):
    model.train()
    target_encoder.eval()  # Ensure target encoder is frozen

    best_energy = float("inf")  # Initialize the best energy

    for epoch in range(epochs):
        start_time = time.time()  # Start epoch timer
        total_energy = 0

        for batch_idx, batch in enumerate(train_loader):
            states, actions = batch.states.to(device), batch.actions.to(device)

            # Forward pass through the model
            predicted_states = model(states, actions)  # [B, T, repr_dim]

            # Encode target states
            with torch.no_grad():  # Target encoder is frozen
                target_states = target_encoder(
                    states.view(-1, states.shape[2], states.shape[3], states.shape[4])
                )
                target_states = target_states.view(states.shape[0], states.shape[1], -1)  # [B, T, repr_dim]

            # Compute contrastive loss
            loss = distance_fn(
                F.normalize(predicted_states.reshape(-1, predicted_states.shape[-1]), dim=-1),
                F.normalize(target_states.reshape(-1, target_states.shape[-1]), dim=-1)
            )

            # Add regularization
            variance_loss = variance_regularization(predicted_states.reshape(-1, predicted_states.shape[-1]))
            covariance_loss = covariance_regularization(predicted_states.reshape(-1, predicted_states.shape[-1]))
            loss += 0.1 * variance_loss + 0.1 * covariance_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Momentum update for target encoder
            update_target_encoder(model.encoder, target_encoder)

            total_energy += loss.item()

        # Average energy for the epoch
        avg_energy = total_energy / len(train_loader)

        # Save the model if energy improves
        if avg_energy < best_energy:
            best_energy = avg_energy
            torch.save(model.state_dict(), save_path)  # Save the best model
            print(f"Epoch {epoch + 1}: New best model saved with energy {best_energy:.4f}")

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{epochs}, Average Energy: {avg_energy:.4f}, Time: {epoch_time:.2f}s"
        )
 
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate the model
    model = MockModel().to(device)

    # Define the target encoder (Enc_ψ)
    # For simplicity, you can reuse the encoder of the main model
    target_encoder = model.encoder
    target_encoder.eval()  # Ensure the target encoder is frozen during training

    # Load data
    train_loader = load_data(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        target_encoder=target_encoder,
        epochs=5,
        optimizer=optimizer,
        device=device,
        distance_fn=torch.nn.MSELoss(),
        save_path="best_model.pth"
    )

#   '''
# Epoch 1: New best model saved with energy 0.0001
# Epoch 1/5, Average Energy: 0.0001, Time: 149.65s
# Epoch 2/5, Average Energy: 0.0001, Time: 144.61s
# Epoch 3: New best model saved with energy 0.0000
# Epoch 3/5, Average Energy: 0.0000, Time: 144.12s
# Epoch 4/5, Average Energy: 0.0000, Time: 143.83s
# Epoch 5/5, Average Energy: 0.0000, Time: 143.76s
#   ''' 