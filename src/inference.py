import time
import uuid

from tqdm import tqdm
from typing import Type

from json_processing import save_json
from prediction_processor import get_prediction_per_frame, compare_bboxes

def update_json_with_predictions(data: dict, model, conf: float = 0.6):
    """
    Обновляет JSON данными предсказаний модели.
    """
    for file_entry in data.get('files', []):
        file_name = file_entry.get('file_name')
        if not file_name:
            continue  # Пропускаем, если имя файла не указано

        # Выполняем инференс для каждого кадра видеофайла
        preds, inference_time = get_prediction_per_frame(model, file_entry, conf=conf)

        for pred in preds:
            frame = pred['markup_frame']
            predicted_path = pred['markup_path']
            predicted_vector = pred['markup_vector']

            matched = False
            for chain in file_entry.get('file_chains', []):
                for markup in chain.get('chain_markups', []):
                    if markup.get('markup_frame') == frame:
                        for existing_bbox in markup.get('markup_path', []):
                            if compare_bboxes(existing_bbox, predicted_path[0]):  # Сравниваем первый ббокс из списка
                                # Совпадает, обновляем markup_vector
                                markup['markup_vector'] = predicted_vector
                                matched = True
                                break
                if matched:
                    break

            if not matched:
                # Создаем новый chain_id
                new_chain_id = int(uuid.uuid4().int >> 64) % 1000000
                new_chain = {
                    'chain_id': new_chain_id,
                    'chain_name': f'chain_{new_chain_id}',
                    'chain_vector': [],  # Заполните при необходимости
                    'chain_dataset_id': file_entry.get('dataset_id'),
                    'chain_markups': [
                        {
                            'markup_id': int(uuid.uuid4().int >> 64) % 1000000,
                            'markup_time': frame,  # Или рассчитайте время из кадра
                            'markup_frame': frame,
                            'markup_path': predicted_path,
                            'markup_vector': predicted_vector
                        }
                    ]
                }
                file_entry.setdefault('file_chains', []).append(new_chain)
       
def inference_mode(output_json_name, model, data):
    """
    Inference program mode. Starts processing videos from dataset 
    and creates output JSON configuration file

    Arguments:
        output_json_name: Output JSON file name
        model: YOLO-NAS Pose downloaded model
        data: Input data
    """
    try:
        # Starting inference process for videos
        update_json_with_predictions(data, model)

        # Сохранение результата
        save_json(data, output_json_name)

        print(f"INFO - Inference results saved and recorded into {output_json_name}")
    except Exception as err:
        print(f"ERROR - Exception occured in inference.inference_mode() {err=}, {type(err)=}")
        raise
