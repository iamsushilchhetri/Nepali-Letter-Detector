"""Train the Devanagari character CNN on the UCI Devanagari Handwritten Character Dataset.

Usage:
    uv run scripts/train.py --epochs 10
    uv run scripts/train.py --epochs 1 --limit 200   # quick smoke test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.model import DevanagariCNN

ROOT = Path(__file__).resolve().parent.parent


def build_loaders(data_dir: Path, batch_size: int, limit: int | None):
    train_tf = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_ds = datasets.ImageFolder(data_dir / "Train", transform=train_tf)
    test_ds = datasets.ImageFolder(data_dir / "Test", transform=eval_tf)
    assert train_ds.classes == test_ds.classes

    if limit:
        train_ds = Subset(train_ds, range(min(limit, len(train_ds))))
        test_ds = Subset(test_ds, range(min(limit // 5 or 1, len(test_ds))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    classes = test_ds.dataset.classes if isinstance(test_ds, Subset) else test_ds.classes
    return train_loader, test_loader, classes


def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--limit", type=int, default=None, help="Cap dataset size for a quick smoke test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, classes = build_loaders(args.data_dir, args.batch_size, args.limit)
    print(f"Classes: {len(classes)} | Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    model = DevanagariCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running_loss, seen = 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)
            if (i + 1) % 100 == 0:
                print(f"  epoch {epoch} batch {i + 1}/{len(train_loader)} loss {running_loss / seen:.4f}")

        scheduler.step()
        val_loss, val_acc = evaluate(model, test_loader, device, criterion)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss {running_loss / seen:.4f} "
            f"| val_loss {val_loss:.4f} | val_acc {val_acc:.4%} | {elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output_dir / "model.pt")
            (args.output_dir / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2))
            print(f"  -> saved new best model (val_acc={val_acc:.4%})")

    print(f"Best validation accuracy: {best_acc:.4%}")


if __name__ == "__main__":
    main()
