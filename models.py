from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),  # Adjusted for 2-channel input
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsampling
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),  # Outputs [B, C, 1, 1], regardless of input size
            nn.Flatten(),
            build_mlp([128, 512, self.repr_dim])  # 128 -> 512 -> output_dim
        )

        # Predictor (GRU) for sequential state prediction
        self.predictor = nn.GRU(input_size=self.repr_dim + 2, hidden_size=self.repr_dim, batch_first=True)


    def forward(self, states, actions):
        """
        Args:
            states: [B, T, Ch, H, W]  (147008, 17, 2, 65, 65)
            actions: [B, T-1, 2]  (147008, 16, 2)

        Output:
            predictions: [B, T, D]
        """
        B, T, C, H, W = states.shape
        
        # Encode states
        encoded_states = self.encoder(states.view(-1, C, H, W)).view(B, T, -1)

        # Pad actions to match T and concatenate with encoded states
        actions = F.pad(actions, (0, 0, 0, 1))  # Pad along time dimension
        inputs = torch.cat([encoded_states, actions], dim=-1)

        # Predict latent states
        predictions, _ = self.predictor(inputs)
        return predictions


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

# def test_mock_model():
#     # Define test parameters
#     batch_size = 64 
#     time_steps = 17  # Number of time steps for states
#     action_steps = time_steps - 1  # Number of time steps for actions
#     channels = 2  # Number of channels in states
#     height, width = 65, 65  # Spatial dimensions of states
#     action_dim = 2  # Dimensionality of actions
#     repr_dim = 256  # Dimensionality of model representation output

#     # Generate synthetic test data
#     states = torch.randn(batch_size, time_steps, channels, height, width, dtype=torch.float32)  # [B, T, C, H, W]
#     actions = torch.randn(batch_size, action_steps, action_dim, dtype=torch.float32)  # [B, T-1, 2]

#     # Initialize the model
#     model = MockModel(device="cpu", bs=batch_size, n_steps=time_steps, output_dim=repr_dim)

#     # Pass data through the model
#     predictions = model(states, actions)

#     # Print input and output shapes to validate
#     print(f"States shape: {states.shape}")  # Expected: [B, T, C, H, W]
#     print(f"Actions shape: {actions.shape}")  # Expected: [B, T-1, 2]
#     print(f"Predictions shape: {predictions.shape}")  # Expected: [B, T, repr_dim]

#     # Validate output shape
#     assert predictions.shape == (batch_size, time_steps, repr_dim), \
#         f"Output shape mismatch: expected {(batch_size, time_steps, repr_dim)}, got {predictions.shape}"
#     print("Model handled data correctly!")

# if __name__ == "__main__":
#     test_mock_model()
