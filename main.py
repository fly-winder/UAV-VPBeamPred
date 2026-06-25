"""
Multimodal seq2seq model for DeepSense6G Scenario 23
Task: Using past 8 frames [images + positions] to predict next 5 beam indices.

Inputs
  images:  [B, T_in=8, C, H, W]
  positions: [B, T_in=8, 2]   (e.g., gps_x, gps_y; extend as needed)
Labels
  labels:  [B, T_out=5]  (each in [0, 63] for 64-beam codebook)

Author: Jiali Nie
"""


import argparse
import datetime
import os

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torchvision import transforms
from data_loader import BeamDataset
from model import MultiModalTransformer, LocationOnlyTransformer, ImageOnlyTransformer, MultiModalGRU
from torch.optim.lr_scheduler import ReduceLROnPlateau


MODEL_REGISTRY = {
    "multimodal_gru": MultiModalGRU,
    "multimodal": MultiModalTransformer,
    "position": LocationOnlyTransformer,
    "image": ImageOnlyTransformer,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train beam prediction models.")
    parser.add_argument("--model", default="multimodal_gru", choices=MODEL_REGISTRY.keys())
    parser.add_argument("--data-csv", default="data_windows.csv")
    parser.add_argument("--root-dir", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-beams", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(model_name, num_beams, device):
    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(num_beams=num_beams).to(device)


def main():
    args = parse_args()
    # -----------------------
    # Configuration parameters
    # -----------------------
    img_size = args.img_size
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr
    num_beams = 64  
    num_beams = args.num_beams
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data paths
    root_dir = args.root_dir
    data_csv = args.data_csv

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
    ])

     # -----------------------
    # Data loading
    # -----------------------
    full_dataset = BeamDataset(data_csv, root_dir=root_dir, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # -----------------------
    # Initialize model, optimizer, and loss function
    # -----------------------
    model_name = args.model
    model = build_model(model_name, num_beams, device)
    os.makedirs("Model_log", exist_ok=True)
    print(f"Training model: {model_name}")
    print(f"Using device: {device}")
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # Dynamic learning-rate scheduler: if the validation loss does not decrease for 2 epochs,
    # reduce the learning rate to 0.5 of the original value
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-6)

    # -----------------------
    # Training function, validation function, and accuracy calculation
    # -----------------------
    def topk_accuracy(output, target, topk=(1,3,5)):
        """
        output: (N, num_frames, num_beams)
        target: (N, num_frames)
        Return top-1/top-3/top-5 accuracy
        """
        N, num_frames, num_beams = output.shape
        output = output.view(-1, num_beams)  # (N*num_frames, num_beams)
        target = target.view(-1)             # (N*num_frames,)

        maxk = max(topk)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # (N*num_frames, maxk)
        pred = pred.t()  # (maxk, N*num_frames)
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # (maxk, N*num_frames)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item() / target.size(0))
        return res  # Return top1, top3, top5


    def train_one_epoch(model, loader, optimizer, criterion, device):
        model.train()
        all_outputs = []
        all_labels = []
        total_loss = 0.0

        pbar = tqdm(loader, desc="Training", leave=False)
        for batch in pbar:
            images = batch['images'].to(device)
            loc = batch['loc'].to(device)
            distance = batch['distance'].to(device)
            height = batch['height'].to(device)
            beam = batch['beam'].to(device) - 1

            optimizer.zero_grad()
            out = model(images, loc, distance, height)  # (B,5,num_beams)

            loss = sum(criterion(out[:, i, :], beam[:, i]) for i in range(beam.size(1))) / beam.size(1)

 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            all_outputs.append(out.detach().cpu())
            all_labels.append(beam.cpu())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = total_loss / len(loader)
        return torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0), epoch_loss


    def validate(model, loader, criterion, device):
        model.eval()
        all_outputs = []
        all_labels = []
        total_loss = 0.0

        pbar = tqdm(loader, desc="Validating", leave=False)
        with torch.no_grad():
            for batch in pbar:
                images = batch['images'].to(device)
                loc = batch['loc'].to(device)
                distance = batch['distance'].to(device)
                height = batch['height'].to(device)
                beam = batch['beam'].to(device) - 1

                out = model(images, loc, distance, height)

                loss = sum(criterion(out[:, i, :], beam[:, i]) for i in range(beam.size(1))) / beam.size(1)
                total_loss += loss.item()

                all_outputs.append(out.cpu())
                all_labels.append(beam.cpu())

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = total_loss / len(loader)
        return torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0), epoch_loss


    # -----------------------
    # Main training loop
    # -----------------------
    best_val_loss = float('inf')
    best_train_loss = float("inf")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(1, num_epochs+1):
        train_outputs, train_labels, train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_outputs, val_labels, val_loss = validate(model, val_loader, criterion, device)


        # Calculate Top-1/3/5 accuracy
        train_top1, train_top3, train_top5 = topk_accuracy(train_outputs, train_labels, topk=(1, 3, 5))
        val_top1, val_top3, val_top5 = topk_accuracy(val_outputs, val_labels, topk=(1, 3, 5))

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Top1: {train_top1:.4f} Top3: {train_top3:.4f} Top5: {train_top5:.4f} || "
              f"Val Loss: {val_loss:.4f} | Top1: {val_top1:.4f} Top3: {val_top3:.4f} Top5: {val_top5:.4f}")

        # # ---- Learning-rate scheduling ----
        scheduler.step(val_loss)  

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            save_path = f"Model_log/{model_name}_best_model_train_{timestamp}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with train loss {best_train_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"Model_log/{model_name}_best_model_val_{timestamp}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with val loss {best_val_loss:.4f}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  
    main()

