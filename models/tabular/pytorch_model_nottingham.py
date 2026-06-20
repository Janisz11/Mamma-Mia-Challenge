import argparse
import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tabular MLP on Nottingham TIC + clinical features.")
    parser.add_argument("--clinical-xlsx", default="data/gtp_NCCN_based_filled_data_clinical_info.xlsx")
    parser.add_argument("--nottingham-csv", default="data/nottingham_summary.csv")
    parser.add_argument("--output-csv", default="training_losses.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def load_data(clinical_xlsx, nottingham_csv):
    df_nottingham = pd.read_csv(nottingham_csv)
    df_clinical = pd.read_excel(clinical_xlsx)
    df = df_nottingham.merge(df_clinical[["patient_id", "pcr"]], on="patient_id", how="left")
    return df[df["pcr"].notna()]


def preprocess(df):
    numeric_df = df.select_dtypes(include=[np.number])
    X = numeric_df.drop(columns=["pcr"])
    y = numeric_df["pcr"].astype(int)

    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32),
    )


class PCRNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train(args):
    df = load_data(args.clinical_xlsx, args.nottingham_csv)
    X_train, y_train, X_val, y_val = preprocess(df)

    model = PCRNet(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    file_exists = os.path.isfile(args.output_csv)
    with open(args.output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])

        for epoch in range(args.epochs):
            model.train()
            permutation = torch.randperm(X_train.size(0))
            train_loss = 0.0

            for i in range(0, X_train.size(0), args.batch_size):
                idx = permutation[i:i + args.batch_size]
                optimizer.zero_grad()
                out = model(X_train[idx])
                loss = criterion(out, y_train[idx].unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= (X_train.size(0) / args.batch_size)

            model.eval()
            with torch.no_grad():
                val_out = model(X_val)
                val_loss = criterion(val_out, y_val.unsqueeze(1)).item()
                preds = (val_out >= 0.5).float()
                accuracy = (preds.squeeze() == y_val).float().mean().item()

            writer.writerow([epoch + 1, train_loss, val_loss, accuracy])
            print(f"Epoch {epoch + 1:3d}  train={train_loss:.4f}  val={val_loss:.4f}  acc={accuracy:.4f}")


if __name__ == "__main__":
    train(parse_args())
