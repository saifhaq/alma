import os
import time
from typing import Dict, List, Tuple


def gather_image_paths(
    root_dir: str, times: Dict[str, float]
) -> Tuple[List[str], Dict[str, float]]:
    """
    Gather image paths from a directory.

    Inputs:
    - root_dir (str): The root directory to search for images
    - times (Dict[str, float]): A dictionary to store timing information

    Outputs:
    - image_paths (List[str]): A list of image paths
    - times (Dict[str, float]): An updated dictionary, with stored timing information
    """
    times["image_paths_gather_start_time"] = time.time()
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))

    times["image_paths_gather_end_time"] = time.time()
    return image_paths, times
