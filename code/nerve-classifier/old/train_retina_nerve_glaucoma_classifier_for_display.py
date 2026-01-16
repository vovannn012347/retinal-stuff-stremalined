import os
import time

import utils.misc_retina as misc
from model.retina_classifier_networks import WeightedBCELoss, HandmadeGlaucomaClassifier, FourMetrics
from utils.retinaldata import ImageMaskDataset, ImageResultsDataset
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold

import argparse
import torch.nn as nn
import torch.nn.functional as tochfunc
import torch
import torch.utils.data
import torchvision.transforms as tvtf
import torchvision.transforms.functional as tvtransffunc
import pandas as pd


parser = argparse.ArgumentParser(description='Glaucoma classify training')
parser.add_argument('--config', type=str,
                    default="configs/train_retina_glaucoma_classifier_for_display.yaml", help="Path to yaml config file")


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


def split_by_position(string, position):
    return [string[:position], string[position:]]


def training_loop(nerve_classifier_pass,  # convolution network
                  optimizer,  # network optimizer
                  device,
                  train_loss,  # network loss function
                  eval_metrics,
                  dataset,  # training dataloader
                  last_n_epoch,  # last iteration
                  config,  # Config object
                  ):

    kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=42)

    nerve_classifier_pass.train()
    init_n_epoch = last_n_epoch + 1

    time0 = time.time()

    highest_difficulty_images = []
    global_epoch = init_n_epoch

    while global_epoch < config.num_epoch:
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
            '''total_eval_loss = {
                    "Accuracy": 0,
                    "Precision": 0,
                    "Recall": 0,
                    "F1-Score": 0
                }'''

            true_pos = 0
            false_pos = 0
            false_neg = 0
            true_neg = 0

            nerve_classifier_pass.eval()
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets, input_files = batch

                    outputs = nerve_classifier_pass(inputs)
                    # eval_loss = torch.mean(loss_func(outputs, targets)).item()
                    tp, fp, fn, tn = eval_metrics(outputs, targets)

                    true_pos += tp
                    false_pos += fp
                    false_neg += fn
                    true_neg += tn

                    eval_batches += 1
                    print(f"{eval_batches} true pos:{tp}, false pos:{fp}, true neg:{tn}, false neg:{fn}")
                    highest_difficulty_images = inputs
                    # break

            train_batches = 0
            total_train_loss = 0.0  # Accumulate loss
            nerve_classifier_pass.train()
            print("train")
            for batch_real, batch_mask, batch_keys in train_loader:
                image_pass = torch.clone(batch_real).to(device)
                mask_pass = torch.clone(batch_mask).to(device)

                if config.log_debug:
                    before_params = {name: param.clone() for name, param in nerve_classifier_pass.named_parameters()}

                optimizer.zero_grad()
                output = nerve_classifier_pass(image_pass)
                loss = train_loss(output, mask_pass)
                loss.backward()

                for name, param in nerve_classifier_pass.named_parameters():
                    if param.grad is not None and torch.all(param.grad == 0):
                        print(f"Gradients for {name} are zeroed!")

                optimizer.step()

                if config.log_debug:
                    after_params = {name: param.clone() for name, param in nerve_classifier_pass.named_parameters()}
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

            accuracy = (true_pos + true_neg) / (eval_batches * 3 * config.batch_size)
            precision = true_pos / (true_pos + false_pos + 1e-8)
            recall = true_pos / (true_pos + false_neg + 1e-8)
            f1_score = 2 * precision * recall / (precision + recall + 1e-8)

            average_train_loss = total_train_loss / train_batches
            average_eval_loss = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1_score
                }

            print(f" train(custom loss)/eval metrics {average_train_loss} / {average_eval_loss}")

            # save state dict snapshot
            if global_epoch % config.save_checkpoint_iter == 0 and global_epoch > last_n_epoch:
                misc.save_nerve_classifier("states.pth", nerve_classifier_pass, optimizer, global_epoch, config)

            if config.log_loss:
                data_labels_ordered = ['train_loss']
                log_data = [average_train_loss]
                for loss_name, loss_value in average_eval_loss.items():
                    data_labels_ordered.append(loss_name)
                    log_data.append(loss_value)

                data = pd.DataFrame([log_data], columns=data_labels_ordered)
                data.to_csv(f"{config.log_dir}/iter_{global_epoch + 1}_avg.csv", index=False)
                print(f"saved average loss csv, epoch:{global_epoch + 1}")

            with torch.no_grad():
                if config.print_iter and (global_epoch % config.print_iter == 0):
                    image_pass = torch.clone(highest_difficulty_images).to(device)

                    result_image1 = torch.cat([((image_pass + 1) / 2)[i]
                                               for i in range(image_pass.size(0))], dim=-1)
                    result_image = torch.cat([result_image1], dim=-2)
                    img_out = tvtransffunc.to_pil_image(result_image.squeeze().cpu(), mode="RGB")
                    img_out.save(f"{config.log_dir}/iter_{global_epoch + 1}_sample.jpg")

                    output = nerve_classifier_pass(image_pass).detach()
                    data = pd.DataFrame(output, columns=config.data_labels_ordered)
                    data.to_csv(f"{config.log_dir}/iter_{global_epoch + 1}_sample.csv", index=False)

                    print(f"saved image ite example, epoch:{global_epoch + 1}")

            # save state dict snapshot backup
            if config.save_cp_backup_iter \
                    and global_epoch % config.save_cp_backup_iter == 0 \
                    and global_epoch > init_n_epoch:
                misc.save_nerve_classifier(f"states_{global_epoch + 1}.pth",
                                           nerve_classifier_pass, optimizer, global_epoch,
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
        os.makedirs(os.path.abspath(config.checkpoint_dir), exist_ok=True)
        os.makedirs(os.path.abspath(f"{config.checkpoint_dir}/images"), exist_ok=True)
        print(f"Created checkpoint_dir folder: {config.checkpoint_dir}")

    device_str = 'cuda' if torch.cuda.is_available() and config.use_cuda_if_available else 'cpu'

    device = torch.device(device_str)

    classifier = HandmadeGlaucomaClassifier(
        input_size=config.img_shapes[0],
        num_classes=config.data_labels_ordered.__len__())

    '''transforms = [misc.RandomGreyscale(0.5),
                  T.RandomHorizontalFlip(0.5),
                  T.RandomVerticalFlip(0.5)]'''
    transforms = [misc.HistogramEqualizationHSV(),
                  misc.CLAHETransformLAB(),
                  tvtf.RandomHorizontalFlip(0.5),
                  tvtf.RandomVerticalFlip(0.5)]

    train_dataset = ImageResultsDataset(config.dataset_path_correct,
                                        config.dataset_path_iccorrect,
                                        config.label_folder_path,
                                        config.data_labels_ordered,
                                        img_shape=config.img_shapes,
                                        scan_subdirs=config.scan_subdirs,
                                        label_correct=config.data_label_correct,
                                        transforms=transforms,
                                        device=device)

    if config.gan_loss == 'wbce':
        train_loss = WeightedBCELoss(2, 1, reduction='sum')
    elif config.gan_loss == 'avg':
        train_loss = nn.MSELoss(reduction='sum')
    else:
        raise NotImplementedError(f"Unsupported loss: {config.gan_loss}")

    eval_loss = FourMetrics()

    last_n_iter = -1

    if config.model_restore != '':
        state_dicts = torch.load(config.model_restore, map_location=device)
        classifier = classifier.to(device)
        classifier.load_state_dict(state_dicts['nerve_classifier'])

        optimizer = torch.optim.Adam(classifier.parameters(),
                                     lr=config.opt_lr,
                                     betas=(config.opt_beta1, config.opt_beta2),
                                     weight_decay=config.weight_decay)

        if 'adam_opt_nerve_classifier' in state_dicts.keys():
            optimizer.load_state_dict(state_dicts['adam_opt_nerve_classifier'])
        else:
            optimizer = torch.optim.Adam(classifier.parameters(),
                                         lr=config.opt_lr,
                                         betas=(config.opt_beta1, config.opt_beta2),
                                         weight_decay=config.weight_decay)

        check_model_state_corruption(classifier, state_dicts['nerve_classifier'])
        check_optimizer_state_corruption(optimizer, state_dicts['adam_opt_nerve_classifier'])

        last_n_iter = int(state_dicts['n_iter'])
        print(f"Loaded models from: {config.model_restore}!")
    else:
        classifier = (HandmadeGlaucomaClassifier.create_model(
            input_size=config.img_shapes[0],
            num_classes=config.data_labels_ordered.__len__())
                      .to(device))
        optimizer = torch.optim.Adam(classifier.parameters(),
                                     lr=config.opt_lr,
                                     betas=(config.opt_beta1, config.opt_beta2),
                                     weight_decay=config.weight_decay)
        # scheduler = ExponentialLR(optimizer, gamma=lr_change)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    training_loop(classifier,
                  optimizer,
                  device,
                  train_loss,
                  eval_loss,
                  train_dataset,
                  last_n_iter,
                  config)


if __name__ == '__main__':
    main()
