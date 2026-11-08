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
                    save_compressed_checkpoint, load_compressed_checkpoint, create_pretrained_model)

# ────────────────────────────────────────────────────────────────────────────────
#  Argument & Config
# ────────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Glaucoma classifier with compression')
parser.add_argument(
    '--config',
    type=str,
    default="config/retina_glaucoma_classifier_versioned/16_01_2026/train_retina_glaucoma_classifier.yaml",
    help="Path to compression training config .yaml")

args = parser.parse_args()
config = misc.get_config(args.config)
device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda_if_available else "cpu")

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
        ignored_layers=[],
    )

    pruner.step()
    print("Pruning completed - model modified in-place")


def print_model_summary(model):
    """Calculates and prints parameter count and estimated file size."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()  #
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()  #

    size_all_mb = (param_size + buffer_size) / 1024 ** 2  #
    total_params = sum(p.numel() for p in model.parameters())

    print(f"--- Model Summary ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"Estimated Size:   {size_all_mb:.3f} MB")
    print(f"----------------------")


# ────────────────────────────────────────────────────────────────────────────────
#  Main training
# ────────────────────────────────────────────────────────────────────────────────
def main_training():
    torch.set_num_threads(4)
    torch.backends.mkldnn.enabled = True

    start_epoch = 0
    pruning_stage = config.pruning.forced_pruning_stage
    current_eta_min = 1e-7  # Initial base eta_min


    # ── Model Initialization & Resume Logic ──────────────────────────────────
    if config.model_restore and os.path.isfile(config.model_restore):
        # Load checkpoint and extract stage/epoch metadata
        ckpt = torch.load(config.model_restore, map_location=device)
        meta = ckpt.get('metadata', {})

        model = load_compressed_checkpoint(config.model_restore, device=device)  #
        pruning_stage = meta.get('pruning_stage', getattr(config.pruning, 'forced_pruning_stage', 0))
        start_epoch = meta.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}, stage {pruning_stage}")
    else:
        print("No restore point found. Initializing with VGG16 pre-trained weights.")
        model = create_pretrained_model(
            num_classes=len(config.data_labels_ordered),
            input_size=config.img_shapes[0]
        ).to(device)

    next_prune_epoch = config.pruning.start_after_epoch
    last_prune_epoch = -1
    for i in range(pruning_stage):
        recovery = config.pruning.recovery_epochs_per_step[i]
        last_prune_epoch = next_prune_epoch
        next_prune_epoch += recovery
        current_eta_min *= config.pruning.lr_scale_after_prune

    print_model_summary(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.opt_lr,
        betas=(config.opt_beta1, config.opt_beta2),
        weight_decay=config.weight_decay
    )

    # ── Calculate next_prune_epoch based on resuming state ───────────────────
    next_prune_epoch = config.pruning.start_after_epoch
    for i in range(pruning_stage):
        next_prune_epoch += config.pruning.recovery_epochs_per_step[i]

    # Initialize scheduler for the current recovery period
    if pruning_stage < len(config.pruning.target_channel_multipliers):
        current_T_max = next_prune_epoch - start_epoch
    else:
        current_T_max = config.num_epoch - start_epoch

    recovery_period = max(1, current_T_max)
    scheduler = CosineAnnealingLR(optimizer, T_max=recovery_period, eta_min=current_eta_min)

    # ── Data & Loss ──────────────────────────────────────────────────────────
    train_transforms = tvt.Compose([
        tvt.RandomHorizontalFlip(p=0.5),
        tvt.RandomVerticalFlip(p=0.5),
        tvt.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        tvt.RandAugment(
            num_ops=2,
            magnitude=getattr(config.augmentation, 'randaug_magnitude', 9),
            num_magnitude_bins=31,
            interpolation=tvt.InterpolationMode.BILINEAR,
        ) if hasattr(config, 'augmentation') and getattr(config.augmentation, 'randaug_magnitude',
                                                         0) > 0 else tvt.Lambda(lambda x: x),
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageResultsDataset(
        folder_path=config.dataset_path,
        label_folder_path=config.dataset_labels_path,
        label_ordering=config.data_labels_ordered,
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

    LOSS_THRESHOLD_debug = 0.8
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    # criterion = WeightedBCELoss(weight_positive=20, weight_negative=10, reduction='mean')
    example_inputs = torch.randn(1, 3, config.img_shapes[0], config.img_shapes[0]).to(device)

    # ── Training Loop ────────────────────────────────────────────────────────
    cur_epoch = 0
    try:
        for epoch in range(start_epoch, config.num_epoch):
            cur_epoch = epoch
            # 1. Dropout Ramp-up
            if epoch >= config.dropout.ramp_up_start_epoch:
                progress = min(1.0, (epoch - config.dropout.ramp_up_start_epoch) / config.dropout.ramp_up_over_epochs)
                base_dropout = config.dropout.start_value + progress * (
                            config.dropout.max_value - config.dropout.start_value)

                # Step back during recovery, then ramp back up
                if last_prune_epoch != -1 and epoch < last_prune_epoch + recovery_period:
                    rec_prog = (epoch - last_prune_epoch) / recovery_period
                    reduction = getattr(config.dropout, 'reduction_after_prune', 0.85)
                    drop_factor = reduction + rec_prog * (1.0 - reduction)
                    current_dropout = base_dropout * drop_factor
                else:
                    current_dropout = base_dropout
                model.set_dropout(current_dropout)

            model.train()
            epoch_loss = 0.0
            iter = 0
            for images, labels, _ in dataloader:
                iter += 1
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                per_sample_loss = criterion(outputs, labels)

                loss = per_sample_loss.mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"[{epoch + 1:4d}] iter {iter} loss: {loss.item():.5f}")

                for i in range(len(labels)):
                    if per_sample_loss[i].item() > LOSS_THRESHOLD_debug:
                        print(
                            f"[{epoch + 1:4d}] iter {iter} "
                            f"sample {i} "
                            f"label={[x.item() for x in labels[i]]} "
                            f"predicted={[x.item() for x in outputs[i]]} "
                            "loss=" + f"{per_sample_loss[i].item():.5f}"
                        )

            scheduler.step()  #

            avg_loss = epoch_loss / len(dataloader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{epoch + 1:4d}] loss: {avg_loss:.5f} | LR: {current_lr:.8f} | Drop: {model.encoder1[4].p:.3f}")

            # 3. Pruning Wave
            if (config.pruning.enabled and
                    pruning_stage < len(config.pruning.target_channel_multipliers) and
                    (epoch + 1) == next_prune_epoch):

                print(f"\n→ Pruning stage {pruning_stage + 1}")
                structured_channel_prune(model, amount=config.pruning.prune_amount_per_step, example_inputs=example_inputs)
                print_model_summary(model)

                # Scale both current LR and eta_min
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= config.pruning.lr_scale_after_prune
                current_eta_min *= config.pruning.lr_scale_after_prune

                # Re-initialize scheduler for the new recovery period
                recovery_period = config.pruning.recovery_epochs_per_step[pruning_stage]
                if config.pruning.use_lr_restart:
                    scheduler = CosineAnnealingLR(optimizer, T_max=recovery_period, eta_min=current_eta_min)

                next_prune_epoch += recovery_period
                pruning_stage += 1

            # 4. Checkpointing
            if (epoch + 1) % config.save_checkpoint_iter == 0:
                save_path = os.path.join(config.checkpoint_dir, f"ep{epoch + 1}_stg{pruning_stage}.pth")
                save_compressed_checkpoint(model, optimizer, epoch + 1, save_path, {'pruning_stage': pruning_stage})

    finally:
        # Emergency save if the process is interrupted
        save_path = os.path.join(config.checkpoint_dir, "emergency_exit_backup.pth")
        save_compressed_checkpoint(model, optimizer, cur_epoch, save_path, {'pruning_stage': pruning_stage})

    print("Training finished.")


if __name__ == "__main__":
    main_training()