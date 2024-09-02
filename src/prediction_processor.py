import json
import time
from tqdm import tqdm



def prediction_processing(raw_frame_pred):
    """
    Processing raw model prediction for one frame in video into dict
    Arguments:
        raw_frame_pred: Model prediction for one frame in video
    Returns:
        Processed frame prediction
    """
    try:
        poses = raw_frame_pred.prediction.poses
        scores = raw_frame_pred.prediction.scores
        bboxes = raw_frame_pred.prediction.bboxes_xyxy
        edge_links = raw_frame_pred.prediction.edge_links

        boxes = []
        for i in range(len(bboxes)):
            box = {}
            box_coords = {}

            bbox = bboxes[i]

            box_coords["score"] = float(scores[i])
            box_coords["x"] = float(bbox[0])
            box_coords["y"] = float(bbox[1])
            box_coords["width"] = float(bbox[2] - bbox[0])
            box_coords["height"] = float(bbox[3] - bbox[1])

            # box[f"bbox_{i}"] = box_coords
            boxes.append(box_coords)

        nodes = []
        for i in range(len(poses)):
                
            person = poses[i]

            for j in range(len(person)):

                pose = person[j]
                node = {}
                node_coords = {}
                node_coords["x"] = float(pose[0])
                node_coords["y"] = float(pose[1])
                node_coords["score"] = float(pose[2])

                node[f"node_{j}"] = node_coords
                nodes.append(node)

        edges = []
        for i in range(len(edge_links)):
            edge = {}
            edge_nodes = {}

            link = edge_links[i]

            edge_nodes["from"] = int(link[0])
            edge_nodes["to"] = int(link[1])

            edge[f"edge_{i}"] = edge_nodes
            edges.append(edge)  

        markup_vector = {}
        markup_vector["nodes"] = nodes
        markup_vector["edges"] = edges        

        markup_path = boxes

        return markup_path, markup_vector
    except Exception as err:
        print(f"ERROR - Exception occured in prediction_processing() {err=}, {type(err)=}")
        raise


def compare_bboxes(existing_bbox: dict, predicted_bbox: dict, threshold: float = 0.5) -> bool:
    """
    Сравнивает два ббокса и определяет, совпадают ли они.
    Использует Intersection over Union (IoU) для сравнения.
    """
    x1, y1, w1, h1 = existing_bbox["x"], existing_bbox["y"], existing_bbox["width"], existing_bbox["height"]
    x2, y2, w2, h2 = predicted_bbox["x"], predicted_bbox["y"], predicted_bbox["width"], predicted_bbox["height"]

    # Координаты пересечения
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_width = max(xi2 - xi1, 0)
    inter_height = max(yi2 - yi1, 0)
    intersection = inter_width * inter_height

    # Площадь объединения
    union = w1 * h1 + w2 * h2 - intersection

    iou = intersection / union if union != 0 else 0

    return iou >= threshold
    
def get_prediction_per_frame(model, file, conf = 0.6):
    """
    Bla bla bla
    """
    try:
        preds = []
        start_time = time.time()

        model_prediction = model.predict(file["file_name"], conf=conf)
        raw_prediction = [res for res in tqdm(model_prediction._images_prediction_gen, total=model_prediction.n_frames, desc="Processing Video")]

        for i in range(len(raw_prediction)):
            prediction = {}
            frame_id = i
            prediction["markup_frame"] = frame_id
            prediction["markup_path"], prediction["markup_vector"] = prediction_processing(raw_prediction[i])
            preds.append(prediction)

        end_time = time.time()

        inference_time = end_time - start_time

        return preds, inference_time
    except Exception as err:
        print(f"ERROR - Exception occured in inference_video() {err=}, {type(err)=}")
        raise