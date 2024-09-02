import json
import os

def load_json(input_arg: str) -> dict:
    """
    Загружает JSON из строки или файла.
    """
    if os.path.isfile(input_arg):
        with open(input_arg, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = json.loads(input_arg)
    return data

def save_json(data: dict, output_path: str):
    """
    Сохраняет JSON в файл.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



