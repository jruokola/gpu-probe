import argparse
import os

import mlflow  # For logging metrics
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup_distributed():
    """Initializes torch.distributed if in a distributed environment."""
    rank = -1
    world_size = -1
    local_rank = -1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (
        "WORLD_SIZE" in os.environ
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 0
    ):
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])

            dist.init_process_group(backend="nccl", init_method="env://")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            print(
                f"[Rank {rank}/{world_size}, LocalRank {local_rank}] Distributed training initialized on {device}."
            )
        except Exception as e:
            print(
                f"Error initializing distributed training: {e}. Falling back to single device."
            )
            rank = -1  # Fallback to non-distributed
            world_size = -1
            local_rank = -1
            # device is already set to cuda/cpu above
    else:
        print(
            f"Not a distributed GPU run or environment variables not set. Using single device: {device}"
        )

    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Simple DDP training test for GPU Probe."
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs."
    )
    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        default=0,
        help="Limit batches per epoch for quick test (0 for all).",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data_cifar10",
        help="Path to download/store CIFAR10.",
    )
    parser.add_argument(
        "--dry_run_cpu",
        action="store_true",
        help="Run a minimal test on CPU for 1 batch (for CI).",
    )
    parser.add_argument(
        "--no_mlflow",
        action="store_true",
        help="Disable MLflow logging for this script.",
    )
    cli_args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()
    is_distributed = world_size > 1
    is_master = rank == 0 or rank == -1  # rank -1 for non-distributed

    if cli_args.dry_run_cpu:
        print("Performing dry run on CPU for 1 batch.")
        device = torch.device("cpu")
        cli_args.epochs = 1
        cli_args.batches_per_epoch = 1
        is_distributed = False  # Force non-distributed for CPU dry run
        if dist.is_initialized():
            cleanup_distributed()  # Clean up if it was initialized before override
        rank = 0
        local_rank = 0
        world_size = 1
        is_master = True  # Simulate master for dry run

    if is_master and not cli_args.no_mlflow and mlflow.active_run():
        mlflow.log_params(
            {
                "train_script_epochs": cli_args.epochs,
                "train_script_lr": cli_args.lr,
                "train_script_batches_per_epoch": cli_args.batches_per_epoch,
                "train_script_is_distributed": is_distributed,
                "train_script_world_size": world_size if is_distributed else 1,
                "train_script_dry_run_cpu": cli_args.dry_run_cpu,
            }
        )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    try:
        # Ensure data_path exists
        if not os.path.exists(cli_args.data_path) and is_master:
            os.makedirs(cli_args.data_path, exist_ok=True)
        if is_distributed:
            # Ensure all processes wait for master to create dir if it didn't exist
            dist.barrier()

        trainset = torchvision.datasets.CIFAR10(
            root=cli_args.data_path, train=True, download=is_master, transform=transform
        )
        if is_distributed:
            # Ensure download is complete before other ranks try to load
            dist.barrier()
            train_sampler = DistributedSampler(
                trainset, num_replicas=world_size, rank=rank, shuffle=True
            )
        else:
            train_sampler = None

        trainloader = DataLoader(
            trainset,
            batch_size=16 * (world_size if is_distributed else 1),
            shuffle=(train_sampler is None),
            num_workers=2,
            sampler=train_sampler,
        )

        model = torchvision.models.resnet50(
            weights=None, num_classes=10
        )  # Using weights=None for faster init
        model.to(device)

        if is_distributed:
            model = DDP(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=cli_args.lr, momentum=0.9)

        if is_master:
            print(f"Starting training on {device} for {cli_args.epochs} epochs...")
            if is_distributed:
                print(f"World size: {world_size}")

        for epoch in range(cli_args.epochs):
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            running_loss = 0.0
            processed_batches = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                processed_batches += 1
                if (
                    (i + 1) % 100 == 0 and is_master
                ):  # Print every 100 mini-batches only on master
                    print(
                        f"[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}"
                    )
                    if not cli_args.no_mlflow and mlflow.active_run():
                        mlflow.log_metric(
                            "train_batch_loss",
                            running_loss / 100,
                            step=(epoch * len(trainloader)) + i,
                        )
                    running_loss = 0.0
                if (
                    cli_args.batches_per_epoch > 0
                    and processed_batches >= cli_args.batches_per_epoch
                ):
                    break
            if is_master:
                print(f"Epoch {epoch + 1} finished.")

        if is_master:
            print("Finished Training Test")
            if not cli_args.no_mlflow and mlflow.active_run():
                mlflow.log_metric("train_script_completed", 1)

    except Exception as e:
        print(f"[Rank {rank if rank != -1 else 'N/A'}] Error during training: {e}")
        if is_master and not cli_args.no_mlflow and mlflow.active_run():
            mlflow.log_metric("train_script_completed", 0)
            mlflow.log_param("train_script_error", str(e))
        raise  # Re-raise exception to mark job as failed
    finally:
        if is_distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
