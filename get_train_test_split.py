import os
import numpy as np
import shutil

from tqdm import tqdm

def get_train_test_split(train_dir, train_label_dir, test_dir, test_label_dir, split_ratio=0.9, num_test_sample=50):
    # move files from train to test based on split_ratio
    # label ends with .png, where the image ends with .jpg
    files = os.listdir(train_dir)
    # set seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(files)
    if num_test_sample is None:
        split_idx = int(len(files) * split_ratio)
    else:
        split_idx = len(files) - num_test_sample
    for file in tqdm(files[split_idx:]):
        label_ext = file.replace(".jpg", ".png")
        shutil.move(train_dir + file, test_dir + file)
        shutil.move(train_label_dir + label_ext, test_label_dir + label_ext)
    print("moving done!")


if __name__ == "__main__":
    train_dir = "/home/william/extdisk/data/ACE20k/ACE20k_sky/images/training/"
    train_label_dir = "/home/william/extdisk/data/ACE20k/ACE20k_sky/annotations/training/"
    test_dir = "/home/william/extdisk/data/ACE20k/ACE20k_sky/images/testing/"
    test_label_dir = "/home/william/extdisk/data/ACE20k/ACE20k_sky/annotations/testing/"

    get_train_test_split(train_dir, train_label_dir, test_dir, test_label_dir)