import cv2 as cv
import glob

import os
import shutil
import numpy as np


def make_test_dataset(from_path, to_path):
    from_files = glob.glob(os.path.join(from_path, "*.png"))[:800]

    for from_file in from_files:
        # print(type(from_file))
        to_file = os.path.join(to_path, from_file.split('/')[-1])
        shutil.move(from_file, to_file)
        


if __name__ == "__main__":
    from_path = "./datasets/face_lwx/"
    to_path = "./datasets/train/lwx"

    make_test_dataset(from_path, to_path)