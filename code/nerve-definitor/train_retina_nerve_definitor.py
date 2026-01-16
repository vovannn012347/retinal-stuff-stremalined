import os
import time
import argparse
import torch.nn as nn
import torch.nn.functional as tochfunc
import torch
import torch.utils.data
import torchvision.transforms as tvtransf
import torchvision.transforms.functional as tvtransffunc

from sklearn.model_selection import KFold

import utils.misc_retina as misc
from model.retina_classifier_networks import WeightedBCELoss, FcnskipNerveDefinitor2, IoUWeightedMetrics, \
    load_state_dict_custom
from utils.retinaldata import ImageMaskDataset
from torch.utils.data import DataLoader, Subset
import pandas as pd

parser = argparse.ArgumentParser(description='Retina nerve definitor training')
parser.add_argument('--config', type=str,
                    default="configs/train_retina_glaucoma_definitor.yaml", help="Path to yaml config file")

lr_change = 0.95

def check_optimizer_model_match(optimizer, model):
    model_params = set(p for p in model.parameters())
    optimizer_params = set(p for group in optimizer.param_groups for p in group['params'])

    if model_params == optimizer_params:
        print("Optimizer parameters match the model parameters.")
        return True
    else:
        print("Mismatch detected!")
        extra_in_model = model_params - optimizer_params
        extra_in_optimizer = optimizer_params - model_params

        if extra_in_model:
            print(f"Parameters in model but not in optimizer: {len(extra_in_model)}")
        if extra_in_optimizer:
            print(f"Parameters in optimizer but not in model: {len(extra_in_optimizer)}")
        return False


def check_model_state_corruption(model, state_dict):
    model_params = model.state_dict()
    for name, param in model_params.items():
        if name not in state_dict:
            print(f"Missing parameter in state dict: {name}")
        elif state_dict[name].shape != param.shape:
            print(f"Shape mismatch for {name}: model {param.shape}, state dict {state_dict[name].shape}")
    for name in state_dict.keys():
        if name not in model_params:
            print(f"Extra parameter in state dict: {name}")


def check_optimizer_state_corruption(optimizer, state_dict):
    # Check param_groups length
    if len(optimizer.param_groups) != len(state_dict['param_groups']):
        print("Mismatch in param_groups length.")
        return False

    # Check individual params in param_groups
    for group_idx, (group, saved_group) in enumerate(zip(optimizer.param_groups, state_dict['param_groups'])):
        if len(group['params']) != len(saved_group['params']):
            print(f"Mismatch in param count for group {group_idx}.")
            return False

    # Validate state values (this does not work as it should)
    '''for param_id, state in state_dict['state'].items():
        if param_id not in optimizer.state:
            print(f"Missing parameter state for ID {param_id}.")
        else:
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    if value.shape != optimizer.state[param_id][key].shape:
                        print(f"Shape mismatch for state {key} in param ID {param_id}.")'''
    print("Optimizer state appears valid.")
    return True


def training_loop(nerve_definitor_pass,  # convolution network
                  optimizer,  # network optimizer
                  device,
                  train_loss,  # network loss function
                  eval_metrics,  # network loss function
                  dataset,  # training dataloader
                  last_n_epoch,  # last iteration
                  config,  # Config object
                  ):
    check_optimizer_model_match(optimizer, nerve_definitor_pass)

    if config.prunning_percent:
        nerve_definitor_pass.prune_layers(config.prunning_percent)

    if config.reinit_optimizer:
        optimizer = torch.optim.Adam(nerve_definitor_pass.parameters(),
                                     lr=config.opt_lr,
                                     betas=(config.opt_beta1, config.opt_beta2),
                                     weight_decay=config.weight_decay)

    kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=42)
    init_n_epoch = last_n_epoch + 1

    time0 = time.time()

    highest_difficulty_images = []

    enums = enumerate(kf.split(dataset))

    global_epoch = init_n_epoch

    while global_epoch < config.num_epoch:
        for fold, (train_idx, val_idx) in enums:
            if global_epoch >= config.num_epoch:
                print("Epoch limit")
                break

            print(f"Epoch {global_epoch + 1}/{config.num_epoch}")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.num_workers,
                                      pin_memory=True)

            val_loader = DataLoader(val_subset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=config.num_workers,
                                    pin_memory=True)

            print("eval")
            eval_batches = 0
            total_eval_loss = {
                "IoU": 0,
                "Accuracy": 0,
                "Precision": 0,
                "Recall": 0,
                "F1-Score": 0
            }
            nerve_definitor_pass.eval()
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets, input_files = batch

                    inputs = tochfunc.interpolate(inputs, scale_factor=0.5, mode='bilinear', align_corners=False)
                    targets = tochfunc.interpolate(targets, scale_factor=0.5, mode='bilinear', align_corners=False)

                    outputs = nerve_definitor_pass(inputs)
                    # eval_loss = torch.mean(loss_func(outputs, targets)).item()
                    loss = eval_metrics(outputs, targets).items()
                    for loss_name, loss_value in loss:
                        total_eval_loss[loss_name] += loss_value

                    eval_batches += 1
                    print(f"{eval_batches} {loss}")
                    highest_difficulty_images = inputs
                    # break

            train_batches = 0
            total_train_loss = 0.0  # Accumulate loss
            nerve_definitor_pass.train()
            print("train")
            for batch_real, batch_mask, batch_keys in train_loader:

                image_pass = torch.clone(batch_real).to(device)
                mask_pass = torch.clone(batch_mask).to(device)

                # to 288x288 from 576x576
                image_pass = tochfunc.interpolate(image_pass, scale_factor=0.5, mode='bilinear', align_corners=False)
                mask_pass = tochfunc.interpolate(mask_pass, scale_factor=0.5, mode='bilinear', align_corners=False)

                mask_pass = torch.clamp(mask_pass, min=0.0, max=(1.0 - 1e-8))

                if config.log_debug:
                    before_params = {name: param.clone() for name, param in nerve_definitor_pass.named_parameters()}

                optimizer.zero_grad()
                output = nerve_definitor_pass(image_pass)
                loss = train_loss(output, mask_pass)
                loss.backward()

                if config.log_debug:
                    for name, param in nerve_definitor_pass.named_parameters():
                        if param.grad is not None and torch.all(param.grad == 0):
                            print(f"Gradients for {name} are zeroed!")

                optimizer.step()

                if config.log_debug:
                    after_params = {name: param.clone() for name, param in nerve_definitor_pass.named_parameters()}
                    for name in before_params:
                        diff = (after_params[name] - before_params[name]).abs().sum().item()
                        print(f"Param: {name}, Change: {diff}")

                total_train_loss += loss.item()
                train_batches += 1
                print(f"{train_batches} {loss.item()}")

                if config.log_loss:
                    data_labels_ordered = ['train_loss']
                    log_data = [loss.item()]
                    data = pd.DataFrame([log_data], columns=data_labels_ordered)
                    data.to_csv(f"{config.log_dir}/iter_{global_epoch + 1}_{train_batches}.csv", index=False)
                    print(f"saved loss csv, epoch:{global_epoch + 1}, batch: {train_batches}")

                '''if config.print_iter:
                    with torch.no_grad():
                        image_pass = torch.clone(image_pass).to(device)
                        output = nerve_definitor_pass(image_pass)

                        result_image1 = torch.cat([((image_pass + 1) / 2)[i]
                                                   for i in range(image_pass.size(0))], dim=-1)
                        result_image2 = torch.cat([output.expand(-1, 3, -1, -1)[i] for i in range(output.size(0))], dim=-1)
                        result_image = torch.cat([result_image1, result_image2], dim=-2)
                        img_out = tvtransffunc.to_pil_image(result_image.squeeze().cpu(), mode="RGB")
                        img_out.save(f"{config.log_dir}/iter_{global_epoch + 1}_{train_batches}_mask.jpg")'''

            average_train_loss = total_train_loss / train_batches
            average_eval_loss = {k: v / eval_batches for k, v in total_eval_loss.items()}

            print(f" train(custom loss)/eval iou loss {average_train_loss} / {average_eval_loss}")

            # save state dict snapshot
            if global_epoch % config.save_checkpoint_iter == 0 and global_epoch > last_n_epoch:
                misc.save_nerve_definitor("states.pth", nerve_definitor_pass, optimizer, global_epoch, config)

            if config.log_loss:
                data_labels_ordered = ['train_loss']
                log_data = [average_train_loss]
                for loss_name, loss_value in average_eval_loss.items():
                    data_labels_ordered.append(loss_name)
                    log_data.append(loss_value)

                data = pd.DataFrame([log_data], columns=data_labels_ordered)
                data.to_csv(f"{config.log_dir}/iter_{global_epoch + 1}_avg.csv", index=False)
                print(f"saved average loss csv, epoch:{global_epoch + 1}")

            if config.print_iter and (global_epoch % config.print_iter == 0):
                image_pass = torch.clone(highest_difficulty_images).to(device)
                output = nerve_definitor_pass(image_pass)

                result_image1 = torch.cat([((image_pass + 1) / 2)[i]
                                           for i in range(image_pass.size(0))], dim=-1)
                result_image2 = torch.cat([output.expand(-1, 3, -1, -1)[i] for i in range(output.size(0))], dim=-1)
                result_image = torch.cat([result_image1, result_image2], dim=-2)
                img_out = tvtransffunc.to_pil_image(result_image.squeeze().cpu(), mode="RGB")
                img_out.save(f"{config.log_dir}/iter_{global_epoch + 1}_mask.jpg")
                print(f"saved image ite example, epoch:{global_epoch + 1}")

            # save state dict snapshot backup
            if config.save_cp_backup_iter \
                    and global_epoch % config.save_cp_backup_iter == 0 \
                    and global_epoch > init_n_epoch:
                misc.save_nerve_definitor(f"states_{global_epoch + 1}.pth", nerve_definitor_pass, optimizer,
                                          global_epoch,
                                          config)

            global_epoch += 1


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
        os.makedirs(os.path.abspath(f"{config.checkpoint_dir}/images"))
        print(f"Created checkpoint_dir folder: {config.checkpoint_dir}")

    device_str = 'cuda' if torch.cuda.is_available() and config.use_cuda_if_available else 'cpu'

    device = torch.device(device_str)

    if config.use_dropout:
        if config.loading_dropout_from_norm:
            definitor = FcnskipNerveDefinitor2(num_classes=1, use_dropout=False)
        else:
            definitor = FcnskipNerveDefinitor2(num_classes=1, use_dropout=True)
    else:
        definitor = FcnskipNerveDefinitor2(num_classes=1, use_dropout=False)

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

    if config.model_restore != '':
        state_dicts = torch.load(config.model_restore, map_location=device)
        definitor = definitor.to(device)
        if config.prunning_restore:
            with torch.no_grad():  # Ensure no gradient computation for this operation
                for param in definitor.parameters():
                    param.zero_()  # Set all parameters to zero

        #definitor.load_state_dict(state_dicts['nerve_definitor'], strict=False, assign=True)
        load_state_dict_custom(definitor, state_dicts['nerve_definitor'])
        if config.loading_dropout_from_norm:
            definitor.add_dropout()

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
        definitor = FcnskipNerveDefinitor2.create_model().to(device)
        optimizer = torch.optim.Adam(definitor.parameters(),
                                     lr=config.opt_lr,
                                     betas=(config.opt_beta1, config.opt_beta2),
                                     weight_decay=config.weight_decay)
        # scheduler = ExponentialLR(optimizer, gamma=lr_change)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    training_loop(definitor,
                  optimizer,
                  device,
                  train_loss,
                  eval_loss,
                  train_dataset,
                  last_n_iter,
                  config)


if __name__ == '__main__':
    main()
