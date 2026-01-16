import argparse
import os
import random
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as tochfunc
import torch.utils.data
import torchvision.transforms as tvtransf
import torchvision.transforms.functional as tvtransffunc
from sklearn.model_selection import KFold
from torch.ao.quantization import FakeQuantize, MinMaxObserver, PerChannelMinMaxObserver, MovingAverageMinMaxObserver
from torch.quantization import QConfig
from torch.utils.data import DataLoader, Subset

import utils.misc_retina as misc
from model.retina_classifier_networks import WeightedBCELoss, FcnskipNerveDefinitor2, IoUWeightedMetrics, \
    load_state_dict_custom
from utils.retinaldata import ImageMaskDataset

parser = argparse.ArgumentParser(description='Retina nerve definitor training')
parser.add_argument('--config', type=str,
                    default="configs/train_retina_glaucoma_definitor_for_display.yaml", help="Path to yaml config file")

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

        last_n_epoch = -1

    use_qat = config.quantization['qat']['use_qat']
    if use_qat and not nerve_definitor_pass.quantized:
        activation_dtype = config.quantization['activation_dtype']
        weight_dtype = config.quantization['weight_dtype']
        activation_qscheme = config.quantization['activation_qscheme']
        weight_qscheme = config.quantization['weight_qscheme']
        activation_observer = config.quantization['observer']['activation_observer']
        weight_observer = config.quantization['observer']['weight_observer']
        qat_algorithm = config.quantization['qat']['qat_algorithm']

        observer_map = {
            'MinMaxObserver': MinMaxObserver,
            'PerChannelMinMaxObserver': PerChannelMinMaxObserver,
            'MovingAverageMinMaxObserver': MovingAverageMinMaxObserver
        }

        activation_map = {
            'per_tensor_affine': torch.per_tensor_affine,
            'per_channel_affine': torch.per_channel_affine,
            'per_tensor_symmetric': torch.per_tensor_symmetric
        }


        # Create FakeQuantize layers for activation and weights
        activation_fake_quant = FakeQuantize.with_args(
            observer=observer_map[activation_observer],
            dtype=torch.__dict__[activation_dtype],
            qscheme=activation_map[activation_qscheme]
        )

        weight_fake_quant = FakeQuantize.with_args(
            observer=observer_map[weight_observer],
            dtype=torch.__dict__[weight_dtype],
            qscheme=activation_map[weight_qscheme]
        )

        # Set up the QConfig with the created FakeQuantize layers
        qconfig = QConfig(activation=activation_fake_quant, weight=weight_fake_quant)

        # Apply the QConfig to your model (assuming the model is defined as 'model')
        nerve_definitor_pass.qconfig = qconfig

        nerve_definitor_pass.add_quantize()
        nerve_definitor_pass = torch.quantization.prepare_qat(nerve_definitor_pass)

    if config.reinit_optimizer or config.prunning_percent:
        optimizer = torch.optim.Adam(nerve_definitor_pass.parameters(),
                                     lr=config.opt_lr,
                                     betas=(config.opt_beta1, config.opt_beta2),
                                     weight_decay=config.weight_decay)

    def sorted_insert(array_list, new_item):
        """
        Inserts an item into highest_difficulty_images while keeping it sorted.
        Ensures uniqueness by 'file': If a file exists, updates its loss.
        """
        file_to_loss = {item["file"]: item["loss"] for item in array_list}

        # If file already exists, update its loss
        if new_item["file"] in file_to_loss:
            old_loss = file_to_loss[new_item["file"]]
            if new_item["loss"] <= old_loss:
                return  # Ignore if new loss is lower (to keep the hardest cases)

            # Remove the old entry
            array_list[:] = [item for item in array_list if
                             item["file"] != new_item["file"]]

        # Insert in sorted order (descending by loss)
        index = 0
        while index < len(array_list) and array_list[index]["loss"] > new_item["loss"]:
            index += 1

        array_list.insert(index, new_item)  # Insert at the correct position

    def trim_list(array_list, max_size):
        """
        Trim the highest_difficulty_images list to max_size.
        """
        if len(array_list) > max_size:
            array_list[:] = array_list[:max_size]  # Keep only the highest loss items

    def weighted_random_select(array_list, sample_size):
        """
        Perform a weighted random selection where higher loss values have a greater chance of being chosen.
        """
        if not array_list:
            return []  # Avoid selection if list is empty

        weights = [item["loss"] for item in array_list]  # Use loss values as weights
        return random.choices(array_list, weights=weights, k=sample_size)

    highest_difficulty_images_max = 20
    highest_difficulty_images_trim_above = 35
    highest_difficulty_images = []

    kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=42)
    init_n_epoch = last_n_epoch + 1

    global_epoch = init_n_epoch

    while global_epoch < config.num_epoch:

        if use_qat and global_epoch >= (init_n_epoch + config.quantize_epochs):
            # model was quantized, break
            break

        time0 = time.time()
        enums = enumerate(kf.split(dataset))
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
                loss_item = torch.mean(loss)
                loss_item.backward()

                for name, param in nerve_definitor_pass.named_parameters():
                    if param.grad is not None and torch.all(param.grad == 0):
                        print(f"Gradients for {name} are zeroed!")

                optimizer.step()

                if config.log_debug:
                    after_params = {name: param.clone() for name, param in nerve_definitor_pass.named_parameters()}
                    for name in before_params:
                        diff = (after_params[name] - before_params[name]).abs().sum().item()
                        print(f"Param: {name}, Change: {diff}")

                total_train_loss += loss_item.item()
                train_batches += 1
                print(f"{train_batches} {loss_item.item()}")

                if config.log_loss:
                    data_labels_ordered = ['train_loss']
                    log_data = [loss_item.item()]
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

                # highest difficulty retrainment
                # remember highest difficulty
                for i in range(0, image_pass.size(0)):
                    new_data = {"loss": torch.mean(loss[i]).item(),
                                "file": batch_keys[i],
                                "image": image_pass[i],
                                "mask": mask_pass[i]}
                    sorted_insert(highest_difficulty_images, new_data)

                # trim array
                if len(highest_difficulty_images) > highest_difficulty_images_trim_above:
                    trim_list(highest_difficulty_images, highest_difficulty_images_max)

                selected_images = weighted_random_select(highest_difficulty_images, config.batch_size)

                image_pass = torch.stack([item["image"] for item in selected_images], dim=0)
                mask_pass = torch.stack([item["mask"] for item in selected_images], dim=0)

                image_pass = torch.clone(image_pass).to(device)
                mask_pass = torch.clone(mask_pass).to(device)

                optimizer.zero_grad()
                output = nerve_definitor_pass(image_pass)
                loss = train_loss(output, mask_pass)
                loss_item = torch.mean(loss)
                loss_item.backward()
                optimizer.step()

            average_train_loss = total_train_loss / train_batches
            average_eval_loss = {k: v / eval_batches for k, v in total_eval_loss.items()}

            print(f" train(custom loss)/eval iou loss {average_train_loss} / {average_eval_loss}")

            if use_qat and (not nerve_definitor_pass.quantized) and global_epoch >= (init_n_epoch + config.quantize_epochs - 1):
                nerve_definitor_pass.quantized = True
                # optimization is being busted right NOW, break after saving
                nerve_definitor_pass = torch.quantization.convert(nerve_definitor_pass, inplace=True)

            # save state dict snapshot
            if global_epoch % config.save_checkpoint_iter == 0 and global_epoch > last_n_epoch:
                misc.save_nerve_definitor("states.pth", nerve_definitor_pass, optimizer, global_epoch, config)

            if config.log_loss:
                time_diff = time.time() - time0
                data_labels_ordered = ['time_used', 'train_loss']
                log_data = [time_diff, average_train_loss]
                for loss_name, loss_value in average_eval_loss.items():
                    data_labels_ordered.append(loss_name)
                    log_data.append(loss_value)

                data = pd.DataFrame([log_data], columns=data_labels_ordered)
                data.to_csv(f"{config.log_dir}/iter_{global_epoch + 1}_avg.csv", index=False)
                print(f"saved average loss csv, epoch:{global_epoch + 1}")

            if config.print_iter and (global_epoch % config.print_iter == 0):
                #selected_images = weighted_random_select(highest_difficulty_images, config.batch_size)
                selected_images = highest_difficulty_images[:config.batch_size]
                selected_images = torch.stack([item["image"] for item in selected_images], dim=0)

                image_pass = torch.clone(selected_images).to(device)
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

            if use_qat and global_epoch >= (init_n_epoch + config.quantize_epochs):
                # model was quantized, break
                break

            trim_list(highest_difficulty_images, highest_difficulty_images_max)


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

    definitor.quantized = False
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
        train_loss = WeightedBCELoss(3, 10, reduction='none')
    elif config.gan_loss == 'bce':
        train_loss = nn.BCEWithLogitsLoss(weight=torch.tensor([30.0]), reduction='none')
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

        if 'quantized' in state_dicts.keys() and state_dicts['quantized']:
            definitor.quantized = True

        if config.loading_dropout_from_norm:
            definitor.add_dropout(dropout_prob=config.dropout_probability)

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
