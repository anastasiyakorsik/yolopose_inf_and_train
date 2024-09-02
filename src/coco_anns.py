import json
import cv2
import os
import sys
from datetime import datetime
import uuid
from PIL import Image

from extract_frames import extract_frames_from_videos

def info_coco():
    """
    Creates "info" part of annotation in COCO format
    """
    # {"info": {"description": "COCO 2017 Dataset","url": "http://cocodataset.org","version": "1.0","year": 2017,"contributor": "COCO Consortium","date_created": "2017/09/01"}}
    info = {}
    info["description"] = ""
    info["url"] = ""
    info["version"] = ""
    info["year"] = datetime.now().year
    info["contributor"] = ""
    info["date_created"] = datetime.now().strftime("%Y/%m/%d")
    return info

def categories_coco():
    """
    Creates "categories" part of annotation in COCO format
    """
    categories = []

    category = {}
    category["supercategory"] = "person"
    category["id"] = 1
    category["name"] = "person"
    category["keypoints"] = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
    category["skeleton"] = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    
    categories.append(category)
    return categories


def bbox_to_coco(annotation, markup_path):
    """
    Converts information about bboxes from input JSON file into COCO annotations format

    Arguments:
        annotation (dict): Annotation element
        markup_path (list): BBoxes from JSON
    """
    bbox = []

    for path in markup_path:
        bbox.append(path["x"])
        bbox.append(path["y"])
        bbox.append(path["width"])
        bbox.append(path["height"])

    annotation["bbox"] = bbox
    return annotation

def keypoints_to_coco(annotation, markup_vector):
    """
    Converts information about keypoints of skeleton from input JSON file into COCO annotations format

    Arguments:
        annotation (dict): Annotation element
        markup_path (list): Keypoints from JSON
    """
    node_len = len(markup_vector["nodes"])
    annotation["num_keypoints"] = node_len
    keypoints = []

    for node in markup_vector["nodes"]:
        key = list(node.keys())[0]
        v = 1
        keypoints.append(node[key]["x"])
        keypoints.append(node[key]["y"])
        keypoints.append(v)
    
    annotation["keypoints"] = keypoints
    return annotation

def get_image_dimensions(image_path):
    """
    Returns image width and height from given path to the image
    """
    with Image.open(image_path) as img:
        return img.size

def generate_unique_id():
    """
    Generates unique int ID
    """
    return int(uuid.uuid4().int >> 64) % 1000000

def images_and_anns_coco(input_json, extracted_frames):
    """
    Converts input JSON data in COCO annotations format for training model

    Argumants:
        input_json (dict): Input JSON data
        extracted_frames (list): Paths to created images of video frames

    Returns:
        images (list): "images" part of annotation in COCO format
        annotations (list): "annotations" part of annotation in COCO format

    """
    #with open(input_json, 'r') as f:
    #   data = json.load(f)
    data = input_json

    images = []
    annotations = []

    for frame in extracted_frames:
        # Получаем имя видео и номер кадра
        frame_file = os.path.basename(frame)
        video_name, frame_id_with_ext = frame_file.rsplit('_', 1)
        frame_id = int(frame_id_with_ext.split('.')[0])

        # Заполняем информацию об изображении
        image_id = generate_unique_id()
        width, height = get_image_dimensions(frame)
        images.append({
            "file_name": frame_file,
            "id": image_id,
            "width": width,
            "height": height
        })

        # Поиск соответствующего файла в данных
        for file in data["files"]:

            file_name_with_ext = os.path.basename(file["file_name"])
            file_name = os.path.splitext(file_name_with_ext)[0]

            print(file_name, video_name)
            if file_name == video_name:
                # Поиск аннотаций для соответствующего кадра
                for chain in file["file_chains"]:
                    for markup in chain["chain_markups"]:
                        print(markup["markup_frame"], frame_id)
                        if markup["markup_frame"] == frame_id:
                            annotation_id = generate_unique_id()
                            annotation = {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": 1,
                                "iscrowd": 1,
                                # Преобразование keypoints и bbox
                            }
                            annotation = keypoints_to_coco(annotation, markup["markup_vector"])
                            annotation = bbox_to_coco(annotation, markup["markup_path"])
                            annotations.append(annotation)
    
    return images, annotations

def get_filenames(input_json):
    """
    Gets list of video file names from input JSON data
    """
    names = []
    for file in input_json["files"]:
        names.append(file["file_name"])
    return names


def create_coco_anns(input_json, output_folder = "output_frames"):
    """
    Extracts frames from video into images. Creates and writes annotations for model training in COCO format

    Arguments:
        input_json (dict): Input JSON data
        output_folder (str)[OPTIONAL]: Name of output folder for extracted frames from videos

    Returns: 
        coco_ann_file (str): Name of annotetions file
    """
    try:
        video_paths = get_filenames(input_json)
        coco_ann_file = "coco_annotations.json"

        frames = extract_frames_from_videos(video_paths, output_folder)

        coco_annotations = {}
        coco_annotations["info"] = info_coco()
        coco_annotations["images"], coco_annotations["annotations"] = images_and_anns_coco(input_json, frames)
        coco_annotations["categories"] = categories_coco()

        with open(coco_ann_file, 'w') as outfile:
            json.dump(coco_annotations, outfile, indent=4)

        print(f"Annotations created and recorded in {coco_ann_file}")
        return coco_ann_file
    except Exception as err:
        print(f"ERROR - Exception occured in create_coco_anns() {err=}, {type(err)=}")
        raise


def main():

    input_json = sys.argv[1]
    video_paths = ["video_pose1.mp4"]
    output_folder = "output_frames"
    frames = extract_frames_from_videos(video_paths, output_folder)

    coco_annotations = {}
    coco_annotations["info"] = info_coco()
    coco_annotations["images"], coco_annotations["annotations"] = images_and_anns_coco(input_json, frames)
    coco_annotations["categories"] = categories_coco()

    with open('coco_annotations.json', 'w') as outfile:
        json.dump(coco_annotations, outfile, indent=4)

if __name__ == "__main__":
    main()
