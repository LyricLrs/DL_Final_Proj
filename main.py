from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
import glob

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
        batch_size=64,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
        batch_size=64,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
        batch_size=64,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds

def load_train_data(device):
    data_path = "/scratch/DL24FA"

    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
    )

    return train_ds

# def load_model():
#     """Load or initialize the model."""
#     # TODO: Replace MockModel with your trained model
#     model = MockModel(device="cuda").to("cuda")

#     model = MockModel(device="cuda", bs=56, n_steps=17, output_dim=256)
#     model = model.to("cuda")
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total Parameters: {total_params}")
    
#     train = load_train_data(device="cuda")
#     model.train_model(dataset=train)

#     # Save model for the epoch
#     model_save_path = f"models/10epochs_1213/model_weights.pth"
#     torch.save(model.state_dict(), model_save_path)
#     print(f"Model saved to {model_save_path}")

    
#     return model


def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    model = MockModel()

    model_path = "models/5epochs_1212/resnet18_spatial_epoch5_loss1.0180.pth"
    checkpoint = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
