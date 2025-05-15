#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal CIFAR-10 / ResNet-50 training script that works on a Slurm + Pyxis
cluster (multi-node, multi-GPU) **without requiring a shared dataset mount**.

Key fixes
──────────
1. **Always set `download=True`** – torchvision’s lock-file makes concurrent
   downloads safe, so every rank can fetch the archive if it is not cached.
2. Removed the “rank-0-only download” logic → no “file not found” on workers.
3. LR scaling and batch-size logic retained.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# ── helpers ──────────────────────────────────────────────────────────────────
def setup_distributed():
    rank = int(os.environ.get("RANK", -1))
    world = int(os.environ.get("WORLD_SIZE", -1))
    local = int(os.environ.get("LOCAL_RANK", -1))
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda", local) if use_cuda and local >= 0 else "cpu"

    if world > 1 and use_cuda:
        dist.init_process_group("nccl", init_method="env://")
        torch.cuda.set_device(local)
        print(f"[Rank {rank}/{world} | Local {local}] running on {device}")
    else:
        rank = world = local = -1
        print(f"Single-process run on {device}")

    return rank, world, local, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batches_per_epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data_path", type=str, default="/data/dataset")
    parser.add_argument("--dry_run_cpu", action="store_true")
    args = parser.parse_args()

    rank, world, local, device = setup_distributed()
    distributed = world > 1
    master = rank in (-1, 0)

    if args.dry_run_cpu:
        device, distributed, world = "cpu", False, 1
        args.epochs, args.batches_per_epoch = 1, 1
        if dist.is_initialized():
            cleanup_distributed()

    # ­­­ Dataset (all ranks download if cache miss) ­­­
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=os.path.expandvars(args.data_path),
        train=True,
        download=True,  # <- FIX: safe concurrent download
        transform=transform,
    )

    sampler = (
        DistributedSampler(trainset, num_replicas=world, rank=rank, shuffle=True)
        if distributed
        else None
    )
    imgs_per_gpu = 1024 if world < 4 else 512
    loader = DataLoader(
        trainset,
        batch_size=imgs_per_gpu,
        shuffle=sampler is None,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True,
        sampler=sampler,
    )

    # ­­­ Model and optimiser ­­­
    model = torchvision.models.resnet50(weights=None, num_classes=10).to(device)
    if distributed:
        model = DDP(model, device_ids=[local])

    crit = nn.CrossEntropyLoss()
    global_batch = imgs_per_gpu * max(world, 1)
    lr = args.lr * global_batch / 128
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    if master:
        print(
            f"Start training: epochs={args.epochs}, global_batch={global_batch}, lr={lr}"
        )

    # ­­­ Training loop ­­­
    for epoch in range(args.epochs):
        if distributed and sampler:
            sampler.set_epoch(epoch)

        running, seen = 0.0, 0
        for i, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            optim.step()

            running += loss.item()
            seen += 1
            if master and i % 100 == 0:
                print(f"[epoch {epoch + 1} | batch {i}] loss={running / 100:.3f}")
                running = 0.0
            if 0 < args.batches_per_epoch <= seen:
                break

        if master:
            print(f"Epoch {epoch + 1} done.")

    if master:
        print("Training finished.")

    cleanup_distributed()


if __name__ == "__main__":
    main()
