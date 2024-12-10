import numpy as np
import torch
import torch.nn as nn
from typing import List
from tqdm import tqdm  # Correct the import
import torch.optim as optim
from dataset import create_wall_dataloader

from transformers import ViTConfig, ViTModel
from loss import VICRegLoss


# Define a simple MLP architecture for testing purposes
def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

# Mock model definition for testing
class MockModel(nn.Module):
    """
    A simple model for testing purposes.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super(MockModel, self).__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = output_dim  # 256-dimensional latent space
        
        # Define CNN layers
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Dropout(0.3),  # Dropout with 30%
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Flatten(),
        # )
        config = ViTConfig(
            image_size=65,
            patch_size=4,
            num_channels=2
        )

        self.vit = ViTModel(config=config)

        self.fc = nn.Linear(257 * 768, self.repr_dim)  # Output is 256-dimensional

        # GRU layer for sequential prediction
        self.predictor = nn.GRU(input_size=self.repr_dim + 2, hidden_size=self.repr_dim, batch_first=True)


    def forward(self, states, actions, train=False):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]        (147008, 17, 2, 65, 65)
            During inference:
                states: [B, 1, Ch, H, W]        (147008, 1, 2, 65, 65)
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, C, H, W = states.shape
        # print("states shape before reshaping:", states.shape)

        if train:
            states_flat = states.view(B * T, C, H, W)
            # print("states shape before reshaping:", states_flat.shape)
            encoded_states = self.vit(states_flat).last_hidden_state

            # print("states shape before reshaping:", encoded_states.shape)
            latent_states = self.fc(encoded_states.view(B, -1)).view(B, T, -1)
        else:
            init_state = states[:, 0, :, :, :]
            encoded_init = self.vit(init_state.view(B, C, H, W)).last_hidden_state
            latent_states = self.fc(encoded_init.view(B, -1)).unsqueeze(1)
            

        predictions = [latent_states[:, 0, :]]  # Start predictions with the first latent state

        for t in range(actions.shape[1]):
            action = actions[:, t, :].unsqueeze(1)
            input_to_predictor = torch.cat([predictions[-1].unsqueeze(1), action], dim=-1)
            output, _ = self.predictor(input_to_predictor)
            predictions.append(output.squeeze(1))

        predictions = torch.stack(predictions, dim=1)  # [B, T, repr_dim]
        return predictions if not train else (predictions, latent_states)


    def train_model(self, dataset):
        """
        Train the model.
        """
        learning_rate = 0.001
        num_epochs = 5
        device = self.device
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        vicreg_loss_fn = VICRegLoss()

        print(f"Training for {num_epochs} epochs with lr={learning_rate}...")

        for epoch in tqdm(range(num_epochs), desc="Epochs"):  # Use tqdm correctly
            epoch_loss = 0

            for batch in tqdm(dataset, desc="Training step"):
                states, locations, actions = batch
                init_states = states[:, 0:1].to(device)  # Extract initial states
                actions = actions.to(device)

                pred_encs, target_encs = self.forward(states=init_states, actions=actions, train=True)

                # loss = torch.nn.functional.mse_loss(pred_encs, target_encs)
                loss = vicreg_loss_fn(pred_encs, target_encs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataset)

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Save model for the epoch
            model_save_path = f"models/vit_epoch{epoch+1}_loss{avg_loss:.4f}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

        print("Training complete.")


def load_data(device):
    data_path = "/scratch/DL24FA"
    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
    )
    return train_ds


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



# if __name__ == "__main__":

#     model = MockModel(device="cuda", bs=64, n_steps=17, output_dim=256)
#     model = model.to("cuda")
#     train_ds = load_data(device="cuda")

#     model.train_model(dataset=train_ds)
