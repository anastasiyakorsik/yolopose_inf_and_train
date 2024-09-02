import cv2
import os

def extract_frames_from_videos(video_paths, output_folder):
    """
    Extracts frames as images from videos to create train set

    Arguments:
        video_paths (list): List of paths to input videos
        output_folder (str): Name of output folder to save images

    Returns:
        frame_paths (list): Paths to created images
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        frame_paths = []

        for video_path in video_paths:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            cap = cv2.VideoCapture(video_path)
            frame_id = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_file = f"{video_name}_{frame_id:06d}.jpg"
                frame_path = os.path.join(output_folder, frame_file)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                frame_id += 1

            cap.release()

        return frame_paths

        print(f"Extraction complete. Frames saved in {output_folder}.")
    except Exception as err:
        print(f"ERROR - Exception occured in extract_frames_from_videos() {err=}, {type(err)=}")
        raise
