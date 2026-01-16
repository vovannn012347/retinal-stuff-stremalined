import os
import time
import argparse
import torch.nn as nn
import torch.nn.functional as tochfunc
import torch
import torch.utils.data
import torchvision.transforms as tvtransf

import utils.misc_retina as misc
from model.retina_classifier_networks import WeightedBCELoss, FcnskipNerveDefinitor2, IoUWeightedMetrics, \
    load_state_dict_custom
from utils.mics import check_model_state_corruption, check_optimizer_state_corruption
from utils.retinaldata import ImageMaskDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Retina nerve definitor training')
parser.add_argument('--config', type=str,
                    default="configs/test_retina_glaucoma_definitor_for_display.yaml", help="Path to yaml config file")

lr_change = 0.95

def testing_loop(nerve_definitor_pass,  # convolution network
                 device,
                 train_loss,  # network loss function
                 eval_metrics,  # network loss function
                 dataset,  # training dataloader
                 last_n_epoch,  # last iteration
                 config,  # Config object
                 ):

    if config.prunning_percent:
        nerve_definitor_pass.prune_layers(config.prunning_percent)

    if config.reinit_optimizer:
        optimizer = torch.optim.Adam(nerve_definitor_pass.parameters(),
                                     lr=config.opt_lr,
                                     betas=(config.opt_beta1, config.opt_beta2),
                                     weight_decay=config.weight_decay)

    test_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False,
                             num_workers=1,
                             pin_memory=True)

    images_difficulty_sorted = []

    with torch.no_grad():
        for batch_real, batch_mask, batch_keys in test_loader:

            image_pass = torch.clone(batch_real).to(device)
            mask_pass = torch.clone(batch_mask).to(device)

            # to 288x288 from 576x576
            image_pass = tochfunc.interpolate(image_pass, scale_factor=0.5, mode='bilinear', align_corners=False)
            mask_pass = tochfunc.interpolate(mask_pass, scale_factor=0.5, mode='bilinear', align_corners=False)

            mask_pass = torch.clamp(mask_pass, min=0.0, max=(1.0 - 1e-8))

            if config.log_debug:
                before_params = {name: param.clone() for name, param in nerve_definitor_pass.named_parameters()}

            output = nerve_definitor_pass(image_pass)
            loss = eval_metrics(output, mask_pass)

            images_difficulty_sorted.append({"loss": loss["IoU"], "file": batch_keys[0]})

    images_difficulty_sorted.sort(key=lambda x: x["loss"], reverse=True)
    for elt in images_difficulty_sorted:
        print(f"{elt['loss']}, {elt['file']}")

    with open("values_hardness.txt", "w", encoding="utf-8") as file:
        for elt in images_difficulty_sorted:
            print(f"{elt['loss']}, {elt['file']}")
            file.write(f"{elt['loss']}, {elt['file']}\n")


def main():
    args = parser.parse_args()
    config = misc.get_config(args.config)

    # set random seed
    if config.random_seed:
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        import numpy as np
        np.random.seed(config.random_seed)

    # make checkpoint folder if nonexistent
    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(os.path.abspath(config.checkpoint_dir))
        print(f"Created checkpoint_dir folder: {config.checkpoint_dir}")

    if not os.path.isdir(config.log_dir):
        os.makedirs(os.path.abspath(config.log_dir))
        print(f"Created checkpoint_dir folder: {config.log_dir}")

    device_str = 'cuda' if torch.cuda.is_available() and config.use_cuda_if_available else 'cpu'

    device = torch.device(device_str)

    if config.use_dropout:
        if config.loading_dropout_from_norm:
            definitor = FcnskipNerveDefinitor2(base=config.learning_base, num_classes=1, use_dropout=False)
        else:
            definitor = FcnskipNerveDefinitor2(base=config.learning_base,
                                               num_classes=1,
                                               use_dropout=True,
                                               dropout_probability=config.dropout_probability)
    else:
        definitor = FcnskipNerveDefinitor2(base=config.learning_base, num_classes=1, use_dropout=False)

    transforms = [misc.RandomGreyscale(1),
                  tvtransf.RandomHorizontalFlip(0.5),
                  tvtransf.RandomVerticalFlip(0.5)]

    train_dataset = ImageMaskDataset(config.dataset_path,
                                     config.mask_folder_path,
                                     img_shape=config.img_shapes,
                                     scan_subdirs=config.scan_subdirs,
                                     transforms=transforms,
                                     device=device)

    if config.gan_loss == 'wbce':
        train_loss = WeightedBCELoss(3, 10, reduction='mean')
    elif config.gan_loss == 'bce':
        train_loss = nn.BCEWithLogitsLoss(weight=torch.tensor([30.0]), reduction='mean')
    elif config.gan_loss == 'avg':
        train_loss = nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError(f"Unsupported loss: {config.gan_loss}")

    eval_loss = IoUWeightedMetrics(30, 100, reduction='mean')

    last_n_iter = -1

    if config.model_restore != '' and os.path.exists(config.model_restore):
        state_dicts = torch.load(config.model_restore, map_location=device)
        definitor = definitor.to(device)
        if config.prunning_restore:
            with torch.no_grad():  # Ensure no gradient computation for this operation
                for param in definitor.parameters():
                    param.zero_()  # Set all parameters to zero

        if config.loading_dropout_from_norm:
            load_state_dict_custom(definitor, state_dicts['nerve_definitor'])
            definitor.add_dropout(dropout_prob=config.dropout_probability)
        else:
            definitor.load_state_dict(state_dicts['nerve_definitor'], strict=False, assign=True)

        optimizer = torch.optim.Adam(definitor.parameters(),
                                     lr=config.opt_lr,
                                     betas=(config.opt_beta1, config.opt_beta2),
                                     weight_decay=config.weight_decay)

        if 'adam_opt_nerve_definitor' in state_dicts.keys():
            optimizer.load_state_dict(state_dicts['adam_opt_nerve_definitor'])
        else:
            optimizer = torch.optim.Adam(definitor.parameters(),
                                         lr=config.opt_lr,
                                         betas=(config.opt_beta1, config.opt_beta2),
                                         weight_decay=config.weight_decay)

        check_model_state_corruption(definitor, state_dicts['nerve_definitor'])
        check_optimizer_state_corruption(optimizer, state_dicts['adam_opt_nerve_definitor'])

        last_n_iter = int(state_dicts['n_iter'])
        print(f"Loaded models from: {config.model_restore}!")
    else:
        if config.learning_base == 64:
            # load vgg-16 normally
            definitor = FcnskipNerveDefinitor2.create_model(
                use_dropout=config.use_dropout,
                dropout_probability=config.dropout_probability
            ).to(device)
        else:
            definitor = definitor.to(device)

        optimizer = torch.optim.Adam(definitor.parameters(),
                                     lr=config.opt_lr,
                                     betas=(config.opt_beta1, config.opt_beta2),
                                     weight_decay=config.weight_decay)
        # scheduler = ExponentialLR(optimizer, gamma=lr_change)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    testing_loop(definitor,
                 device,
                 train_loss,
                 eval_loss,
                 train_dataset,
                 last_n_iter,
                 config)


if __name__ == '__main__':
    main()
