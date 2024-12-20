import time
import uuid
import os

from tqdm import tqdm
from typing import Type

from json_processing import load_json, save_json
from prediction_processor import get_prediction_per_frame, compare_bboxes

from logger import py_logger

from workdirs import INPUT_PATH, OUTPUT_PATH, INPUT_DATA_PATH

def check_video_extension(video_path):
    valid_extensions = {"avi", "mp4", "m4v", "mov", "mpg", "mpeg", "wmv"}
    ext = os.path.splitext(video_path)[1][1:].lower()
    return ext in valid_extensions

def create_json_with_predictions(data_file: dict, model, conf: float = 0.6):
    """
    Adds to input JSON model predictions and saves result in output JOSN for each videofile
    
    Arguments:
        data: Input JSON data for one video
        model: Prediction model
        conf: Model confidence
    """
    try:
        file_name = data_file['file_name']
        file_id = data_file['file_id']
        full_file_name = os.path.join(INPUT_PATH, data_file['file_name'])
            
        # Output JSON file name
        out_json = file_name + '.json'
        out_json_path = os.path.join(OUTPUT_PATH, out_json)

        # Output JSON file name for recording new bboxes
        out_json_additional = file_name + '_add.json'
        output_add_file_data = {}
        output_add_file_data['file_name'] = file_name
        output_add_file_data['file_id'] = file_id
        output_add_file_data['file_data'] = []
        out_add_chains = []
        out_json_additional_path = os.path.join(OUTPUT_PATH, out_json)

        # Inference for each frame of video
        preds, inference_time = get_prediction_per_frame(model, full_file_name, conf=conf)

        # Processing inference result
        for pred in preds:
            frame = pred['markup_frame']
            predicted_path = pred['markup_path']
            predicted_vector = pred['markup_vector']

            matched = False

            # Compare input bboxes (markup_path) with predicted by models

            # if MATCHED - put skeleton data in markup_vector and chain_vector, 
            # also calculate markup_frame and markup_time; 
            # chain_name, markup_parent_id, markup_path - from input

            for chain in data_file.get('file_chains', []):

                for markup in chain.get('chain_markups', []):
                    markup_frame = markup['markup_frame']
                    markup_path = markup['markup_path']

                    # YOLO NAS POSE creates detections per frame
                    # hence, predicted_path is an array of all bboxes per frame
                    for detected_bbox in predicted_path:
                        if markup_frame == frame:
                            if compare_bboxes(markup_path, detected_bbox):
                                markup['markup_vector'] = predicted_vector
                                chain['chain_vector'] = predicted_vector
                                matched = True
                                break
                if matched:
                    break

            # if NOT MATCHED - create new chain with all new inner params (except markup_parent_id)
            # record in additional output JSON
            if not matched:
                new_chain_name = int(uuid.uuid4().int >> 64) % 1000000
                new_chain = {
                    'chain_name': new_chain_id,
                    'chain_vector': predicted_vector,
                    'chain_markups': [
                        {
                            'markup_parent_id': '',
                            'markup_frame': frame,
                            'markup_time': frame, #TODO: count time for frame
                            'markup_path': predicted_path,
                            'markup_vector': predicted_vector
                        }
                    ]
                }
                out_add_chains.append(new_chain)

        # Save new JSONs
        output_add_file_data['file_data'] = out_add_chains

        save_json(data_file, out_json_path)
        save_json(output_add_file_data, out_json_additional_path)
        py_logger.info(f"Inference results for {file_name} saved and recorded into {out_json_path} and {out_json_additional_path}")

    except Exception as e:
        py_logger.exception(f'Exception occurred in inference.create_json_with_predictions(): {e}', exc_info=True)
       
def inference_mode(model, json_files):
    """
    Inference program mode. Processing video and
    adding new data about skeleton to each input JSON and save

    Arguments:
        model: YOLO-NAS Pose downloaded model
        data: Input json data of video
    """
    try:
        # for each video go inference and add its json with new data
        for json_file in json_files:
            full_json_path = os.path.join(INPUT_DATA_PATH, json_file)
            print(full_json_path)
            data = load_json(full_json_path)

            #TODO: Add on_progress
            data_file = data['files'][0]
            file_name = data_file['file_name']
            full_file_name = os.path.join(INPUT_PATH, data_file['file_name'])

            if not os.path.exists(full_file_name) or not check_video_extension(file_name):
                continue  # Skip if file does not exist or extension doens't match appropriate ones

            # Starting inference process for videos
            create_json_with_predictions(data_file, model)

    except Exception as e:
        py_logger.exception(f'Exception occurred in inference.inference_mode(): {e}', exc_info=True)
        #TODO: Add on_error
