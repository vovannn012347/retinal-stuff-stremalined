# train_retina_nerve_glaucoma_classifier.py
import argparse
import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch_pruning as tp
import torchvision.transforms as tvt

import misc  # your config loader + helpers
from datasets import ImageResultsDataset
from models import (HandmadeGlaucomaClassifier, WeightedBCELoss,
                    save_compressed_checkpoint, load_compressed_checkpoint)

# ────────────────────────────────────────────────────────────────────────────────
#  Argument & Config
# ────────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Glaucoma classifier with compression')
parser.add_argument('--config', type=str, required=True,
                    help="Path to compression training config .yaml")

args = parser.parse_args()
config = misc.get_config(args.config)
device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda_if_available else "cpu")


# ────────────────────────────────────────────────────────────────────────────────
#  Model & Optimizer factory (original base only - pruning modifies in-place)
# ────────────────────────────────────────────────────────────────────────────────
def create_model_and_optimizer(base_channels=64):
    model = HandmadeGlaucomaClassifier(
        input_size=config.img_shapes[0] // 2,  # 64 (after internal downscale)
        num_classes=len(config.data_labels_ordered),
        base=base_channels
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.opt_lr,
        betas=(config.opt_beta1, config.opt_beta2),
        weight_decay=config.weight_decay
    )
    return model, optimizer


# ────────────────────────────────────────────────────────────────────────────────
#  Structured pruning using torch-pruning (in-place modification)
# ────────────────────────────────────────────────────────────────────────────────
def structured_channel_prune(model, amount=0.20, example_inputs=None):
    if example_inputs is None:
        raise ValueError("example_inputs required for torch-pruning dependency graph")

    print(f"→ Structured pruning with torch-pruning (~{amount:.1%} channel reduction)")

    imp = tp.importance.MagnitudeImportance(p=1)  # L1 magnitude

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=imp,
        ch_sparsity=amount,
        global_pruning=True,
        ignored_layers=[],  # add classifier/head layers if needed
    )

    pruner.step()
    print("Pruning completed - model modified in-place")


# ────────────────────────────────────────────────────────────────────────────────
#  Main training
# ────────────────────────────────────────────────────────────────────────────────
def main_training():
    # Start with full base model
    current_base = 64
    model, optimizer = create_model_and_optimizer(base_channels=current_base)

    start_epoch = 0
    pruning_stage = 0

    # ── Resume from checkpoint if provided ───────────────────────────────────
    if config.model_restore and os.path.isfile(config.model_restore):
        model = load_compressed_checkpoint(config.model_restore, device=device)
        # Optimizer will be recreated fresh or loaded if you add it to metadata
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.opt_lr,
            betas=(config.opt_beta1, config.opt_beta2),
            weight_decay=config.weight_decay
        )
        # Extract epoch from filename or metadata if needed
        start_epoch = 0  # you can parse from filename or add to metadata
        print(f"Resumed pruned model from {config.model_restore}")
    else:
        print("Starting fresh training with full model")

    # Loss
    criterion = WeightedBCELoss(weight_positive=20, weight_negative=10, reduction='mean')

    # Transforms
    train_transforms = tvt.Compose([
        tvt.RandomHorizontalFlip(p=0.5),
        tvt.RandomVerticalFlip(p=0.5),
        tvt.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        tvt.RandAugment(
            num_ops=2,
            magnitude=getattr(config.augmentation, 'randaug_magnitude', 9),
            num_magnitude_bins=31,
            interpolation=tvt.InterpolationMode.BILINEAR,
        ) if hasattr(config, 'augmentation') and getattr(config.augmentation, 'randaug_magnitude', 0) > 0 else tvt.Lambda(lambda x: x),
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Dataset & loader
    dataset = ImageResultsDataset(
        folder_path=config.dataset_path,
        data_folder_path=config.dataset_labels_path,
        data_label_ordering=config.data_labels_ordered,
        img_shape=config.img_shapes,
        label_correct=config.data_label_correct,
        scan_subdirs=config.scan_subdirs,
        transforms=train_transforms,
        device=device
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

    current_dropout = getattr(config.dropout, 'start_value', 0.0)

    # Example inputs for pruning (match model forward input: 64×64)
    example_inputs = torch.randn(1, 3, config.img_shapes[0] // 2, config.img_shapes[1] // 2).to(device)

    for epoch in range(start_epoch, config.num_epoch):
        # ── Dropout ramp ─────────────────────────────────────────────────────
        if epoch >= config.dropout.ramp_up_start_epoch:
            progress = min(1.0, (epoch - config.dropout.ramp_up_start_epoch) / config.dropout.ramp_up_over_epochs)
            current_dropout = config.dropout.start_value + progress * (config.dropout.max_value - config.dropout.start_value)
        model.set_dropout(current_dropout)

        # ── Training epoch ───────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_batches += 1

        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"[{epoch+1:4d}] loss: {avg_loss:.5f} | dropout: {current_dropout:.3f}")

        scheduler.step()

        # ── Pruning ──────────────────────────────────────────────────────────
        if (config.pruning.enabled and
            pruning_stage < len(config.pruning.target_channel_multipliers) and
            epoch >= config.pruning.start_after_epoch and
            (epoch % config.pruning.prune_every_epochs == 0)):

            print(f"\n→ Pruning stage {pruning_stage + 1} at epoch {epoch + 1}")

            structured_channel_prune(
                model,
                amount=config.pruning.prune_amount_per_step,
                example_inputs=example_inputs
            )

            pruning_stage += 1

            # LR adjustment after pruning
            for param_group in optimizer.param_groups:
                param_group['lr'] *= config.pruning.lr_scale_after_prune

            if config.pruning.use_lr_restart:
                scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

        # ── Checkpointing ────────────────────────────────────────────────────
        if (epoch + 1) % config.save_checkpoint_iter == 0:
            save_path = os.path.join(config.checkpoint_dir, f"states_ep{epoch + 1}_stage{pruning_stage}.pth")
            save_compressed_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                path=save_path,
                extra_meta={'pruning_stage': pruning_stage}
            )

    print("Training finished.")


if __name__ == "__main__":
    main_training()