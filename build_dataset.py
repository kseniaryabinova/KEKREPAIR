import os
import shutil
import random


PATH = '/home/neiro/FlatDataset/1'
DST_TRAIN_PATH = '/home/neiro/FlatDataset/train/cosmetic'
DST_VAL_PATH = '/home/neiro/FlatDataset/val/cosmetic'

if __name__ == '__main__':
    subdirs = os.listdir(PATH)
    total_volume = len(subdirs)
    random.shuffle(subdirs)

    train_volume = int(0.9 * total_volume)
    train_dirs = subdirs[:train_volume]
    val_dirs = subdirs[train_volume:]

    counter = 0

    for dst_path, dirs in ((DST_TRAIN_PATH, train_dirs), (DST_VAL_PATH, val_dirs)):
        for dir_name in dirs:
            dir_path = os.path.join(PATH, dir_name)
            for dir_entry in filter(
                    lambda x: not x.name.endswith('txt'),
                    os.scandir(dir_path)
            ):
                src_image_name, src_image_ext = os.path.splitext(dir_entry.name)
                src_image_path = dir_entry.path
                dst_image_name = src_image_name + '_' + str(counter) + src_image_ext
                dst_image_path = os.path.join(dst_path, dst_image_name)
                shutil.copy(src_image_path, dst_image_path)
                counter += 1
