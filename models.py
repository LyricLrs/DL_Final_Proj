from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from tqdm.auto import tqdm
from resnet import resnet34

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, input_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Adjust input_channels for the first convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet18(num_classes=1000, input_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channels=input_channels)
  
class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=512):
        super(MockModel, self).__init__()
        self.device = device

        self.resnet, self.repr_dim = resnet34(num_channels=2, last_activation="relu")

        self.bn = nn.BatchNorm1d(self.repr_dim)

        self.predictor = nn.GRU(input_size=self.repr_dim + 2, hidden_size=self.repr_dim, batch_first=True)


    def forward(self, states, actions=None, train=True):
        """
        Args:
            states: [B, T, C, H, W] during training or [B, 1, C, H, W] during inference.
            actions: [B, T-1, 2] during training or [B, T-1, 2] during inference (can be None).
            train: Boolean flag indicating training or inference mode.

        Output:
            predictions: [B, T, repr_dim] if train=False
                        [B, T, repr_dim] and latent_states [B, T, repr_dim] if train=True
        """
        B, T, C, H, W = states.shape

        if train:
            # Encode all observations during training
            states_flat = states.view(B * T, C, H, W)
            encoded_states = self.resnet(states_flat)
            latent_states = encoded_states.view(B, T, -1)

            # Normalize latent states
            latent_states = latent_states.transpose(1, 2)  # [B, T, repr_dim] -> [B, repr_dim, T]
            latent_states = self.bn(latent_states).transpose(1, 2)  # [B, repr_dim, T] -> [B, T, repr_dim]

            # During training, the model predicts latent states based on the input
            predictions = [latent_states[:, 0, :]]  # Start with the first latent state

            for t in range(actions.shape[1]):  # Iterate through T-1 actions
                action = actions[:, t, :].unsqueeze(1)  # [B, 1, 2]
                input_to_predictor = torch.cat([predictions[-1].unsqueeze(1), action], dim=-1)  # [B, 1, repr_dim + 2]
                output, _ = self.predictor(input_to_predictor)  # Output: [B, 1, repr_dim]
                predictions.append(output.squeeze(1))  # Append [B, repr_dim]

            predictions = torch.stack(predictions, dim=1)  # Combine predictions: [B, T, repr_dim]
            return predictions, latent_states

        else:
            init_state = states[:, 0, :, :, :].unsqueeze(1)
            encoded_init = self.resnet(init_state.view(B, C, H, W))
            latent_states = encoded_init.unsqueeze(1)  # [B, 1, repr_dim]

            if actions is None:
                return latent_states.squeeze(1)  # [B, repr_dim]

            # Normalize latent states
            latent_states = latent_states.transpose(1, 2)  # [B, T, repr_dim] -> [B, repr_dim, T]
            latent_states = self.bn(latent_states).transpose(1, 2)  # [B, repr_dim, T] -> [B, T, repr_dim]

            # During inference, generate predictions autoregressively
            predictions = [latent_states[:, 0, :]]  # Start with the first latent state

            for t in range(actions.shape[1]):
                action = actions[:, t, :].unsqueeze(1)  # [B, 1, 2]
                input_to_predictor = torch.cat([predictions[-1].unsqueeze(1), action], dim=-1)  # [B, 1, repr_dim + 2]
                output, _ = self.predictor(input_to_predictor)  # Output: [B, 1, repr_dim]
                predictions.append(output.squeeze(1))  # Append [B, repr_dim]

            predictions = torch.stack(predictions, dim=1)  # Combine predictions: [B, T, repr_dim]

            return predictions


    def train_model(self, dataset):
        """
        Train the model to align predicted latent representations with target representations.

        Args:
            dataset: A PyTorch DataLoader containing the training data.
        """
        # Training parameters (hardcoded)
        learning_rate = 0.0001  # Example learning rate
        num_epochs = 100        # Number of epochs
        device = self.device   # Use the device specified in the model
        self.train()

        # Functions
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-3)

        # Warmup and cosine annealing with restarts
        warmup_steps = 10  # Number of warmup epochs
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        # Modify the learning rate at the beginning of each step
        def adjust_learning_rate(optimizer, epoch, warmup_steps, base_lr):
            if epoch < warmup_steps:
                lr = base_lr * (epoch + 1) / warmup_steps
            else:
                lr = scheduler.get_lr()[0]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        def visualize_latent_states(latent_states, title):
            """
            Visualize latent states using PCA.

            Args:
                latent_states (torch.Tensor): The latent states tensor of shape [B, T, D].
                title (str): Title for the PCA plot.
            """
            latent_states = latent_states.view(-1, latent_states.shape[-1]).cpu().detach().numpy()  # [B * T, D]

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

        print(f"Starting model training for {num_epochs} epochs with lr={learning_rate}...")

        early_stopping = EarlyStopping(patience=10, delta=1e-4)
        epoch_losses = []  # To track losses per epoch

        for epoch in tqdm(range(num_epochs), desc="Model training epochs"):
            batch_losses = []  # Track losses per batch
            for batch in tqdm(dataset, desc="Model training step"):
                states = batch.states.to(device)
                actions = batch.actions.to(device)

                pred_encs, target_encs = self.forward(states=states, actions=actions, train=True)

                # # Visualize latent states every few epochs
                # if epoch % 2 == 0 and batch_idx == 0:  # For first batch every 2 epochs
                #     print(f"[Train Debug] Visualizing latent states at epoch {epoch}")
                #     print(f"Type of pred_encs: {type(pred_encs)}")  # Ensure it's a tensor
                #     visualize_latent_states(target_encs,f'target: {epoch}')
                #     visualize_latent_states(pred_encs,f'pred: {epoch}')

                loss = torch.nn.functional.mse_loss(pred_encs, target_encs)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                batch_losses.append(loss.item())  # Append batch loss

            avg_epoch_loss = sum(batch_losses) / len(batch_losses)  # Average epoch loss
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_epoch_loss:.8f}")

             # Step the scheduler
            scheduler.step()

            # Check early stopping
            early_stopping(avg_epoch_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

        print("Model training complete.")

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
