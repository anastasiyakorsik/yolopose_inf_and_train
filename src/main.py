import sys
import argparse
import json
import os
from typing import Dict, Any, List
import uuid

import torch

from super_gradients.training import models
from inference import inference_mode
from train_yolonas_pose import train_mode
from json_processing import load_json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process input data and work format.')
    parser.add_argument('--input_data', type=str, required = True, help='Input data in JSON format')
    parser.add_argument('--work_format_training', action='store_true', default=False, help='Режим работы программы')
    return parser.parse_args()

def main():
    try:
        args = parse_arguments()

        json_data = args.input_data
        training_mode = args.work_format_training

        # Скачивание модели
        print("INFO - Getting model:")
        model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        # Input JSON processing
        data = load_json(json_data)

        if training_mode:
            train_mode(model, data)
        else:
            #TODO: изменить имя выходного файла
            output_json_name = f"yolo-nas-pose-inference.json"
            inference_mode(output_json_name, model, data)
            
    except Exception as err:
        print(f"ERROR - Exception occured in main() {err=}, {type(err)=}")
        raise


if __name__ == "__main__":
    main()