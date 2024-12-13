import numpy as np
import torch
import torch.nn as nn
from typing import List
from tqdm import tqdm
import torch.optim as optim
from dataset import create_wall_dataloader
import torchvision.models as models

from transformers import ViTConfig, ViTModel
from loss import VICRegLoss, BarlowTwinsLoss


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
        
        # Resnet 18
        self.encoder = models.resnet18()

        # increase feature map
        self.encoder.conv1 = nn.Conv2d(
            in_channels=2, 
            out_channels=self.encoder.conv1.out_channels, 
            kernel_size=self.encoder.conv1.kernel_size, 
            # stride=self.encoder.conv1.stride,
            stride=(1, 1),
            padding=self.encoder.conv1.padding, 
            bias=True
        )
        self.encoder.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.encoder.avgpool = nn.AdaptiveAvgPool2d((16, 16))
        self.encoder.final_conv = nn.Conv2d(128, 1, kernel_size=1)

        # spatial predictor
        # [bz, 3, 16, 16] -> [bz, 1, 16, 16]
        self.predictor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )



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
            # print("states_flat shape:", states_flat.shape)
            # encoded_states = self.encoder(states_flat)

            x = self.encoder.conv1(states_flat)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.encoder.avgpool(x)
            encoded_states = self.encoder.final_conv(x)

            # print("states shape after encoder:", encoded_states.shape)
            latent_states = encoded_states.view(B, T, 16, 16)
            # print("latent states shape:", latent_states.shape)

            # latent_states -- (64, 17, 16, 16)
            # actions -- (64, 16, 2)
            # predictions -- (64, 16, D)

            # Predict at each time step t:
            # input_to_predictor -- (64, 3, 16, 16)
            # predicted_latent_state -- (64, 1, 16, 16)

            predicted_latents = []

            for t in range(T - 1):
                # Concatenate latent state and action at time step t
                # actions_t (64, 2) --> (64, 2, 16, 16)
                actions_t = actions[:, t, :]
                actions_broadcasted = actions_t.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)

                input_to_predictor = torch.cat([latent_states[:, t:t+1, :, :], actions_broadcasted], dim=1)
                
                # Predict the next latent state using the predictor
                predicted_latent_state = self.predictor(input_to_predictor)
                predicted_latents.append(predicted_latent_state.view(B, -1))

            # Stack the predicted latents
            predictions = torch.stack(predicted_latents, dim=1)

            return predictions, latent_states.view(B, T, -1)    # [B, T, 256]


        else:
            # Inference logic
            init_state = states[:, 0, :, :, :]
            # encoded_init = self.encoder(init_state.view(B, C, H, W))
            x = self.encoder.conv1(init_state)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.encoder.avgpool(x)
            encoded_init = self.encoder.final_conv(x)       # [B, 1, 16, 16]

            # latent_state = encoded_init.view(B, -1)
            predictions = []
            predictions.append(encoded_init.view(B, -1))

            for t in range(T - 1):
                # Predict the next state using the previous state and action
                actions_t = actions[:, t, :]
                actions_broadcasted = actions_t.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)

                input_to_predictor = torch.cat([encoded_init, actions_broadcasted], dim=1)

                predicted_state = self.predictor(input_to_predictor)
                predictions.append(predicted_state.view(B, -1))

            predictions = torch.stack(predictions, dim=1)

        return predictions      # [B, T, 256]



    def train_model(self, dataset):
        """
        Train the model.
        """
        learning_rate = 0.001       # could try smaller lr like 0.0002 & consine lr scheduler
        num_epochs = 5
        device = self.device
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        vicreg_loss_fn = VICRegLoss()
        barlow_twins_loss_fn = BarlowTwinsLoss(device=device, lambda_param=5e-3)

        print(f"Training for {num_epochs} epochs with lr={learning_rate}...")

        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            epoch_loss = 0

            for batch in tqdm(dataset, desc="Training step"):
                states, locations, actions = batch
                states = states.to(device)
                actions = actions.to(device)

                pred_encs, target_encs = self.forward(states=states, actions=actions, train=True)

                # print("shape of pred_encs:", pred_encs.shape)
                # print("shape of target_encs:", target_encs[:, 1:, :].shape)

                # loss = torch.nn.functional.mse_loss(pred_encs, target_encs[:, 1:, :])
                loss = barlow_twins_loss_fn(pred_encs, target_encs[:, 1:, :])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataset)

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Save model for the epoch
            model_save_path = f"models/resnet18_spatial_epoch{epoch+1}_loss{avg_loss:.4f}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

        print("Training complete.")


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


def load_data(device):
    data_path = "/scratch/DL24FA"
    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
        batch_size=56       # reduce bs to avoid OOM
    )
    return train_ds


if __name__ == "__main__":

    model = MockModel(device="cuda", bs=56, n_steps=17, output_dim=256)
    model = model.to("cuda")

    # for name, module in model.encoder.named_modules():
    #     print(f"{name}: {module}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    train_ds = load_data(device="cuda")

    model.train_model(dataset=train_ds)
