import torch
import os
import pathlib
from super_gradients.training import models
from super_gradients.common.object_names import Models


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process input data and work format.')
    parser.add_argument('--input_video', type=str, required = True, help='Input video file')
    parser.add_argument('--weights', type=str, required = True, help='Path to model weights')
    parser.add_argument('--model_size', type=str, required = True, help='Size of YOLO-NAS Pose Model')
    return parser.parse_args()

def get_model(model_size, weights):
    """
    Uploading YOLO-NAS Pose Model

    Arguments:
        model_size (str): Defines YOLO-NAS Pose model size: small (s), meduim (m), large (l)
        weights (str): Path to downloaded model weights
    """
    model_name = "yolo_nas_pose_" + {model_size}
    model = models.get(model_name, num_classes = 17, checkpoint_path=weights)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    return model

def main():
    try:
        args = parse_arguments()
        video_path = args.input_data
        weights = args.weights
        size = args.model_size

        # Скачивание модели
        print("INFO - Getting model:")
        model = get_model(size, weights)


        output_video_path = pathlib.Path(video_path).stem + "-detections" + pathlib.Path(video_path).suffix
        print("INFO - Processing video:")
        model.predict(video_path, conf=0.4).save(output_video_path)

        print(f"Processed video saved in {output_video_path}")

            
    except Exception as err:
        print(f"ERROR - Exception occured in main() {err=}, {type(err)=}")
        raise

if __name__ == "__main__":
    main()


