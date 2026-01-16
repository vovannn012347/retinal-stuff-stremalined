import os
import time
# from torchvision.models import VGG16_Weights # do not touch - for later implementation
# from utils.misc_retina import magic_wand_mask_selection_faster, apply_clahe_rgb, apply_clahe_lab  # do not touch - for later implementation
from model.retina_classifier_networks import WeightedBCELoss, HandmadeGlaucomaClassifier
from utils.retinaldata import ImageResultsDataset
from utils.misc_retina import save_nerve_classifier
import utils.misc_retina as misc

import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pandas as pd

parser = argparse.ArgumentParser(description='Glaucoma classify training')
parser.add_argument(
    '--config',
    type=str,
    default="config/retina_glaucoma_classifier_versioned/16_01_2026/train_retina_glaucoma_classifier.yaml",
    help="Path to yaml config file")


def main():

    args = parser.parse_args()
    config = misc.get_config(args.config)

    # set random seed
    if config.random_seed:
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        # import numpy as np
        # np.random.seed(config.random_seed)

    # make checkpoint folder if nonexistent
    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(os.path.abspath(config.checkpoint_dir))
        os.makedirs(os.path.abspath(f"{config.checkpoint_dir}/images"))
        print(f"Created checkpoint_dir folder: {config.checkpoint_dir}")

    device_str = 'cuda' if torch.cuda.is_available() and config.use_cuda_if_available else 'cpu'

    device = torch.device(device_str)

    classifier = HandmadeGlaucomaClassifier(
        input_size=config.img_shapes[0]/2,
        num_classes=config.data_labels_ordered.__len__())

    optimizer = torch.optim.Adam(classifier.parameters(),
                                 lr=config.opt_lr,
                                 betas=(config.opt_beta1, config.opt_beta2),
                                 weight_decay=config.weight_decay)

    transforms = [misc.RandomGreyscale(0.5),
                  T.RandomHorizontalFlip(0.5),
                  T.RandomVerticalFlip(0.5)]

    train_dataset = ImageResultsDataset(config.dataset_path_correct,
                                        config.dataset_path_iccorrect,
                                        config.label_folder_path,
                                        config.data_labels_ordered,
                                        img_shape=config.img_shapes,
                                        scan_subdirs=config.scan_subdirs,
                                        label_correct=config.data_label_correct,
                                        transforms=transforms,
                                        device=device)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=config.num_workers,
                                                   pin_memory=True)

    if config.gan_loss == 'wbce':
        train_loss = WeightedBCELoss(20, 100, reduction='none')
    elif config.gan_loss == 'avg':
        train_loss = nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError(f"Unsupported loss: {config.gan_loss}")

    last_n_iter = -1

    if config.model_restore != '':
        state_dicts = torch.load(config.model_restore)
        classifier.load_state_dict(state_dicts['nerve_classifier'])
        if 'adam_opt_nerve_classifier' in state_dicts.keys():
            optimizer.load_state_dict(state_dicts['adam_opt_nerve_classifier'])
        last_n_iter = int(state_dicts['n_iter'])
        print(f"Loaded models from: {config.model_restore}!")
    else:
        classifier = HandmadeGlaucomaClassifier.create_model(
            input_size=config.img_shapes[0]/2,
            num_classes=config.data_labels_ordered.__len__())

    training_loop(classifier,
                  device,
                  optimizer,
                  train_loss,
                  train_dataloader,
                  last_n_iter,
                  config)


def training_loop(
        nerve_classifier_loaded,
        device,
        optimizer,
        loss_func,
        train_dataloader,
        last_n_iter,
        config):

    time0 = time.time()
    nerve_classifier_loaded.train()
    init_n_iter = last_n_iter + 1
    train_iter = iter(train_dataloader)

    highest_difficulty_initialized = False
    highest_difficulty_images = []
    highest_difficulty_labels = []
    highest_difficulty_files = []
    highest_difficulty_losses = []
    lowest_loss = 0

    #kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=42)

    for n_iter in range(init_n_iter, config.max_iters):

        try:
            batch_real, batch_mask, batch_keys1 = next(train_iter)
        except Exception as e:
            train_iter = iter(train_dataloader)
            batch_real, batch_mask, batch_keys1 = next(train_iter)

        image_pass = torch.clone(batch_real).to(device)
        image_data = torch.clone(batch_mask).to(device)
        batch_keys = batch_keys1

        image_pass = F.interpolate(image_pass, scale_factor=0.5, mode='bilinear', align_corners=False)

        output = nerve_classifier_loaded(image_pass)

        loss_raw = loss_func(output, image_data)

        loss = torch.mean(loss_raw)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #cv2.waitKey(1)

        if config.print_loss and (n_iter % config.print_loss == 0):
            # measure iterations/second
            dt = time.time() - time0
            print(f"@iter: {n_iter}: {(config.print_iter/dt):.4f} it/s")
            time0 = time.time()
            print(str(n_iter) + " iter loss: " + str(loss.item()))

        if not highest_difficulty_initialized:
            highest_difficulty_images = torch.zeros_like(image_pass).to(device).copy_(image_pass)
            highest_difficulty_labels = [image_data[i] for i in range(image_data.__len__())]
            highest_difficulty_files = [batch_keys[i] for i in range(batch_keys.__len__())]
            highest_difficulty_losses = [torch.mean(loss_raw[i]).item() for i in range(loss_raw.size(0))]
            highest_difficulty_initialized = True
            lowest_loss = min(highest_difficulty_losses)
        else:
            # select highest losses and re-run on them every time
            current_losses = [torch.mean(loss_raw[i]).item() for i in range(loss_raw.size(0))]
            files_repeated_indexes = []

            # update losses with higher ones if possible
            for file_i in range(batch_keys.__len__()):
                try:
                    # for per-file tracking
                    found_index = highest_difficulty_files.index(batch_keys[file_i])
                    files_repeated_indexes.append(found_index)
                except ValueError:
                    # Return -1 if the value is not found
                    found_index = -1

                if found_index > 0 and current_losses[file_i] > highest_difficulty_losses[found_index]:
                    # highest_difficulty_images[found_index] = image_pass[file_i]
                    # highest_difficulty_masks[found_index] = mask_pass[file_i]
                    highest_difficulty_losses[found_index] = current_losses[file_i]

            lowest_loss = min(highest_difficulty_losses)

            for file_i in range(batch_keys.__len__()):
                if files_repeated_indexes.__contains__(file_i):
                    continue

                # don't need to optimize this much for low batches
                if current_losses[file_i] > lowest_loss:
                    minloss_index = highest_difficulty_losses.index(min(highest_difficulty_losses))
                    highest_difficulty_images[minloss_index] = image_pass[file_i]
                    highest_difficulty_labels[minloss_index] = image_data[file_i]
                    highest_difficulty_losses[minloss_index] = current_losses[file_i]
                    highest_difficulty_files[minloss_index] = batch_keys[file_i]
                    lowest_loss = min(highest_difficulty_losses)

            # re-run optimizer for batch with largest losses
            image_pass = torch.clone(highest_difficulty_images).to(device)
            image_data = torch.stack([highest_difficulty_labels[i] for i in range(highest_difficulty_labels.__len__())])

            output = nerve_classifier_loaded(image_pass)

            loss_raw = loss_func(output, image_data)

            loss = torch.mean(loss_raw)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update losses
            highest_difficulty_losses = [torch.mean(loss_raw[i]).item() for i in range(loss_raw.size(0))]
            lowest_loss = min(highest_difficulty_losses)

        # logging
        if config.print_iter and (n_iter % config.print_iter == 0):

            os.makedirs(f"{config.log_dir}/training_images", exist_ok=True)

            result_image1 = torch.cat([((image_pass + 1) / 2)[i] for i in range(image_pass.size(0))], dim=-1)
            # result_image2 = torch.cat([output.expand(-1, 3, -1, -1)[i] for i in range(output.size(0))], dim=-1)
            result_image = torch.cat([result_image1], dim=-2)
            img_out = TF.to_pil_image(result_image.squeeze().cpu(), mode="RGB")
            img_out.save(f"{config.log_dir}/training_images/iter_{n_iter}.jpg")

            data = pd.DataFrame(output.detach().numpy(), columns=config.data_labels_ordered)
            data.to_csv(f"{config.log_dir}/training_images/iter_{n_iter}.csv", index=False)

            data = pd.DataFrame(image_data.detach().numpy(), columns=config.data_labels_ordered)
            data.to_csv(f"{config.log_dir}/training_images/iter_t_{n_iter}.csv", index=False)

        # save state dict snapshot
        if n_iter % config.save_checkpoint_iter == 0 \
                and n_iter > init_n_iter:
            save_nerve_classifier("states.pth", nerve_classifier_loaded, optimizer, n_iter, config)

        # save state dict snapshot backup
        if config.save_cp_backup_iter \
                and n_iter % config.save_cp_backup_iter == 0 \
                and n_iter > init_n_iter:
            save_nerve_classifier(f"states_{n_iter}.pth", nerve_classifier_loaded, optimizer, n_iter, config)


if __name__ == '__main__':
    main()
