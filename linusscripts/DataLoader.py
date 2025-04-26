"""
Take a json file and extract keypoints and labels from it.
"""

import json
import os
import numpy as np

class_to_id = {
    "Adult": 1,
    "DJ": 2,
    "J": 3,
}

def load_json(json_path):
    """
    Load a json file and extract keypoints and labels from it.
    :param json_path: path to the json file
    :return: keypoints and labels
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    keypoints = []
    labels = []

    for item in data["keypoints"]:
        keypoints.append(item['coordinates'])
        # Convert class name to id
        item['class'] = class_to_id.get(item['class'], 0)
        labels.append(item['class'])

    return np.array(keypoints, dtype=np.float32), np.array(labels)

kp = load_json(r'C:\Users\linus\NematodeAI/0_Data/Annotations/C0105_cropped_keypoints.json')

print(kp[:2])