import sys
import argparse
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import MammaMiaCompetitionDataset
from model import FusionPCRNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train FusionPCRNet on DCE-MRI data.")
    parser.add_argument("--data-root", default="data",
                        help="Root directory containing images/ and segmentations/expert/")
    parser.add_argument("--clinical-xlsx", default="data/clinical_and_imaging_info.xlsx")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def collate_fn(batch):
    images, labels, tic = zip(*batch)
    return torch.stack(images), torch.stack(labels), torch.stack(tic)


def train():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    img_dir = os.path.join(args.data_root, "images")
    seg_dir = os.path.join(args.data_root, "segmentations", "expert")
    patient_ids = [
        p for p in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, p))
    ]

    dataset = MammaMiaCompetitionDataset(
        patient_ids=patient_ids,
        images_root=img_dir,
        clinical_xlsx=args.clinical_xlsx,
        segmentation_root=seg_dir,
    )
    # Filter to patients with known labels
    dataset.patient_ids = [pid for pid in dataset.patient_ids if pid in dataset.labels]

    val_size = int(args.val_split * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionPCRNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    best_val_loss = float("inf")

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, tic in train_loader:
            x, y, tic = x.to(device), y.to(device), tic.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, tic), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, y_true, y_pred = 0.0, [], []
        with torch.no_grad():
            for x, y, tic in val_loader:
                x, y, tic = x.to(device), y.to(device), tic.to(device)
                logits = model(x, tic)
                val_loss += criterion(logits, y).item() * x.size(0)
                y_pred.extend(logits.argmax(dim=1).cpu().numpy())
                y_true.extend(y.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        accuracy = accuracy_score(y_true, y_pred)

        print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  acc={accuracy:.4f}")
        print(classification_report(y_true, y_pred, target_names=["no pCR", "pCR"]))

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, accuracy])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print("  → model saved")


if __name__ == "__main__":
    train()
