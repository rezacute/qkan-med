"""Train QKAN-Med and evaluate the fixed final epoch on CRC-VAL-HE-7K."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from qkan_med import QKANMed

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_dataset(path, expected):
    transform = transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    data = datasets.ImageFolder(path, transform=transform)
    if data.classes != CLASSES or len(data) != expected:
        raise ValueError(f"Expected {expected} images with classes {CLASSES}; "
                         f"found {len(data)} images with classes {data.classes}")
    return data


def data_manifest(data):
    entries = [f"{Path(path).relative_to(data.root).as_posix()}\t{label}"
               for path, label in data.samples]
    return {"images": len(data), "class_to_idx": data.class_to_idx,
            "path_label_sha256": hashlib.sha256("\n".join(entries).encode()).hexdigest()}


def make_loader(data, batch_size, workers, seed, cuda):
    return DataLoader(data, batch_size=batch_size, num_workers=workers,
                      pin_memory=cuda, generator=torch.Generator().manual_seed(seed),
                      **({"prefetch_factor": 2, "multiprocessing_context": "spawn"} if workers else {}))


def run_epoch(model, data, device, optimizer=None):
    model.train(optimizer is not None)
    confusion = torch.zeros(9, 9, dtype=torch.long)
    total_loss, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch, (images, labels) in enumerate(data, 1):
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            if optimizer is not None:
                loss.backward()
                norms = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
                if not torch.isfinite(torch.stack(norms)).all():
                    raise RuntimeError("Non-finite gradients")
                optimizer.step()
            total_loss += loss.item() * len(labels)
            count += len(labels)
            confusion += torch.bincount((labels * 9 + logits.argmax(1)).cpu(), minlength=81).reshape(9, 9)
            if batch % 100 == 0:
                print(f"  batch {batch}/{len(data)} loss={total_loss/count:.5f}", flush=True)
    support = confusion.sum(1)
    return {"loss": total_loss/count, "accuracy": confusion.diag().sum().item()/count,
            "samples": count, "confusion_matrix": confusion.tolist(),
            "class_support": dict(zip(CLASSES, support.tolist())),
            "per_class_accuracy": {name: confusion[i, i].item()/support[i].item() if support[i] else None
                                   for i, name in enumerate(CLASSES)}}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", default="runs/qkan-med-seed42")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Install a CUDA-enabled PyTorch build and check the NVIDIA driver.")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}. Choose a new --output.")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device("cuda")
    train = load_dataset(args.train_dir, 100000)
    test = load_dataset(args.test_dir, 7180)
    train_paths = {Path(path).resolve() for path, _ in train.samples}
    if any(Path(path).resolve() in train_paths for path, _ in test.samples):
        raise ValueError("Training and test image paths overlap")
    model = QKANMed().to(device)
    model.encoder.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("encoder.")], "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    source = Path(__file__).resolve().parent
    config = {"arguments": vars(args), "torch": str(torch.__version__),
              "torchvision": __import__("torchvision").__version__, "numpy": np.__version__,
              "pillow": __import__("PIL").__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(device), "train": data_manifest(train), "test": data_manifest(test),
              "model": {"qubits": 16, "quantum_parameters": 96, "head_parameters": 8457,
                        "encoding": "RY(pi * sigmoid(adapter(features)))", "dropout": 0.5,
                        "dropout_location": "after quantum output", "cnot_gates": 30},
              "optimizer": {"name": "AdamW", "encoder_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 1e-4},
              "scheduler": {"name": "CosineAnnealingLR", "eta_min": 1e-6},
              "selection": "fixed final epoch",
              "code_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [source/"train.py", *sorted((source/"qkan_med").glob("*.py"))]}}
    write_json(out/"config.json", config)
    history = []
    started = time.monotonic()
    for epoch in range(args.epochs):
        indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        data = make_loader(Subset(train, indices), args.batch_size, args.workers, args.seed + epoch, True)
        print(f"Epoch {epoch+1}/{args.epochs}", flush=True)
        metrics = run_epoch(model, data, device, optimizer)
        history.append({"epoch": epoch+1, "train": metrics, "learning_rates": [g["lr"] for g in optimizer.param_groups]})
        scheduler.step()
        write_json(out/"history.json", history)
        print(f"  training accuracy={metrics['accuracy']:.4%}", flush=True)
    # Evaluate once at the final epoch; test results never select model weights.
    metrics = run_epoch(model, make_loader(test, args.batch_size, args.workers, args.seed, True), device)
    temporary = out/"model.pt.tmp"
    torch.save({"state_dict": model.state_dict(), "config": config, "classes": CLASSES}, temporary)
    temporary.replace(out/"model.pt")
    write_json(out/"results.json", {"epochs": args.epochs, "seed": args.seed,
                                   "elapsed_seconds": time.monotonic()-started, "test": metrics})
    print(f"Test accuracy: {metrics['accuracy']:.4%}. Results saved to {out}", flush=True)


if __name__ == "__main__":
    main()
