import sys
import argparse
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import pad
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_mamma_mia import MammaMiaDataset


class Simple3DCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Linear(256, 1024)
        self.fc_out = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return torch.sigmoid(self.fc_out(x))


def pad_collate(batch):
    images, labels = zip(*batch)
    desired_c = 11
    max_shape = [max(s) for s in zip(*[img.shape[1:] for img in images])]
    padded = []
    for img in images:
        c, d, h, w = img.shape
        p = pad(img, (0, max_shape[2] - w, 0, max_shape[1] - h, 0, max_shape[0] - d, 0, max(0, desired_c - c)))
        if desired_c - c < 0:
            p = p[:desired_c]
        padded.append(p)
    return torch.stack(padded), torch.tensor(labels).float()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Simple3DCNN baseline on DCE-MRI data.")
    parser.add_argument("--data-root", default="data",
                        help="Root directory containing images/, segmentations/ and clinical_and_imaging_info.xlsx")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def train():
    args = parse_args()

    IMG_DIR = os.path.join(args.data_root, "images")
    SEG_DIR = os.path.join(args.data_root, "segmentations", "expert")
    CLINICAL_XLSX = os.path.join(args.data_root, "clinical_and_imaging_info.xlsx")
    LOG_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_METRICS = os.path.join(LOG_DIR, "metrics.csv")

    dataset = MammaMiaDataset(
        image_root=IMG_DIR,
        seg_root=SEG_DIR,
        clinical_xlsx=CLINICAL_XLSX,
        margin=10,
        crop=True,
        use_diffs=True,
        skip_missing_labels=True,
    )

    val_size = int(args.val_split * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=pad_collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Simple3DCNN(in_channels=11).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"training_log_{timestamp}.txt")
    best_val_loss = float("inf")

    with open(log_path, "w") as log_f:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss, y_true, y_pred = 0.0, [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device).unsqueeze(1)
                    out = model(x)
                    val_loss += criterion(out, y).item() * x.size(0)
                    y_pred.extend((out > 0.5).int().cpu().numpy())
                    y_true.extend(y.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            accuracy = accuracy_score(y_true, y_pred)

            line = f"Epoch {epoch:3d}: train={train_loss:.4f}  val={val_loss:.4f}  acc={accuracy:.4f}\n"
            print(line, end="")
            log_f.write(line)

            report = classification_report(y_true, y_pred, output_dict=True)
            row = {"epoch": epoch, "accuracy": accuracy}
            for label in report:
                if isinstance(report[label], dict):
                    for metric, val in report[label].items():
                        row[f"{label}_{metric}"] = val

            file_exists = os.path.isfile(CSV_METRICS)
            with open(CSV_METRICS, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(LOG_DIR, "best_model.pt"))
                log_f.write("  → model saved\n")


if __name__ == "__main__":
    train()
