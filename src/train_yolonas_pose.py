import os
import time
import sys

from typing import Any, List, Tuple, Union
import yaml
import json

import matplotlib.pyplot as plt

from super_gradients.training import Trainer
from super_gradients.training.datasets.pose_estimation_datasets import YoloNASPoseCollateFN
from super_gradients.training import models
# from super_gradients.training.utils.distributed_training_utils import wait_for_the_master, get_local_rank

import torch
import torchvision
from torch.utils.data import DataLoader

from super_gradients.training.datasets.pose_estimation_datasets.coco_pose_estimation_dataset import COCOPoseEstimationDataset
from train_params import define_train_params, EDGE_LINKS, EDGE_COLORS, KEYPOINT_COLORS

from keypoint_transforms import define_set_transformations
from pose_estimation_dataset import PoseEstimationDataset
from coco_anns import create_coco_anns


def open_file(file_path: str) -> Union[dict, list, None]:
    """
    Opens and reads the content of a JSON or YAML file.

    Parameters:
    file_path (str): The path to the file.

    Returns:
    Union[dict, list, None]: The content of the file parsed to a dictionary or a list,
                             or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            if file_path.endswith('.json'):
                return json.load(file)
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                return yaml.safe_load(file)
            else:
                raise ValueError(f'Unsupported file format: {file_path}')
    except Exception as e:
        print(f'An error occurred in open_file func: {e}, trying to open file {file_path}')
        return None


def training(model, config):
    """
    Trains YOLO-NAS Pose model and seves model weights

    Arguments:
        model: YOLO-NAS Pose downloaded model
        config (dict): Configuration for training
    """
    try: 

        train_transforms, val_transforms = define_set_transformations()
        NUM_EPOCHS = 10

        # coco train2017 val2017 annotations/person_keypoints_train2017.json annotations/person_keypoints_val2017.json
        #TODO: Take the following args from JSON input file
        dataset_dir = config["dataset_dir"]
        train_imgs_dir = config["train_imgs_dir"]
        val_imgs_dir = config["val_imgs_dir"]
        path_to_train_anns = config["path_to_train_anns"]
        path_to_val_anns = config["path_to_val_anns"]

        if config["NUM_EPOCHS"] != 0:
            NUM_EPOCHS = config["NUM_EPOCHS"]

        # Create instances of the dataset
        print("INFO - Create instance of the train dataset")
        train_dataset = PoseEstimationDataset(
            data_dir=dataset_dir,
            images_dir=train_imgs_dir,
            json_file=path_to_train_anns,
            transforms=train_transforms,
            edge_links = EDGE_LINKS,
            edge_colors = EDGE_COLORS,
            keypoint_colors = KEYPOINT_COLORS
            )

        print("INFO - Create instance of the val dataset")
        val_dataset = PoseEstimationDataset(
            data_dir=dataset_dir,
            images_dir=val_imgs_dir,
            json_file=path_to_val_anns,
            transforms=val_transforms,
            edge_links = EDGE_LINKS,
            edge_colors = EDGE_COLORS,
            keypoint_colors = KEYPOINT_COLORS
            )

        CHECKPOINT_DIR = 'checkpoints'
        if(not os.path.isdir(CHECKPOINT_DIR)):
            print('INFO - Creating', CHECKPOINT_DIR)
            os.makedirs(CHECKPOINT_DIR)
        else:
            print('INFO - Already exists', CHECKPOINT_DIR)

        trainer = Trainer(experiment_name='first_yn_pose_run', ckpt_root_dir=CHECKPOINT_DIR)

        train_params = define_train_params(NUM_EPOCHS)

        # Create dataloaders
        train_dataloader_params = {
            'shuffle': True,
            'batch_size': 16,
            'drop_last': True,
            'pin_memory': False,
            'collate_fn': YoloNASPoseCollateFN()
            }

        val_dataloader_params = {
            'shuffle': True,
            'batch_size': 16,
            'drop_last': True,
            'pin_memory': False,
            'collate_fn': YoloNASPoseCollateFN()
            }

        train_dataloader = DataLoader(train_dataset, **train_dataloader_params)
        val_dataloader = DataLoader(val_dataset, **val_dataloader_params)

        # Обучение модели
        print("INFO - Start training model:")
        start_time = time.time()
        trainer.train(model,
                    training_params=train_params,
                    train_loader=train_dataloader,
                    valid_loader=val_dataloader
                    )
        end_time = time.time()

        training_time = end_time - start_time
        print(f"INFO - Model trained successfully in {training_time} s")

    except Exception as e:
        print(f'ERROR - An error occurred in training_mode(): {e}')

def train_mode(model, input_json):
    try:
        """
        Training program mode.

        Arguments:
            model: YOLO-NAS Pose downloaded model
            input_json (dict): Input JSON data
        """
        frames_folder = "output_frames"
        coco_ann_file = create_coco_anns(input_json, frames_folder)

        train_config = {}

        #TODO: Change validation dataset of remove at all
        train_config["dataset_dir"] = "."
        train_config["train_imgs_dir"] = "output_frames"
        train_config["val_imgs_dir"] = "output_frames"
        train_config["path_to_train_anns"] = "coco_annotations.json"
        train_config["path_to_val_anns"] = "coco_annotations.json"
        train_config["NUM_EPOCHS"] = 1

        training(model, train_config)
    except Exception as err:
        print(f"ERROR - Exception occured in train_mode() {err=}, {type(err)=}")
        raise




