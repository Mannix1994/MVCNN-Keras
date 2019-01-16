# coding=utf8

from __future__ import division
from __future__ import print_function

import os
import random
import sys
import zipfile
import cv2
import shutil
import config as g


NUM_VIEWS = g.NUM_VIEWS

DATA_DIR = os.path.join('data', 'modelnet40v1')

UN_TRAIN_DIR = 'untrain'


# 需要把数据集解压到参数指定的目录里面
def main(argv):
    tmp = os.path.join(DATA_DIR, 'tmp')

    # delete pre-made dir
    shutil.rmtree(os.path.join(DATA_DIR, 'train'), ignore_errors=True)
    shutil.rmtree(os.path.join(DATA_DIR, 'test'), ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(os.path.join(DATA_DIR, UN_TRAIN_DIR), ignore_errors=True)

    # make tmp path
    os.makedirs(tmp)

    # check if it is a zip file
    if not zipfile.is_zipfile(argv[1]):
        raise ValueError(argv[1] + 'is not a zip file')

    # unzip data
    a = zipfile.ZipFile(argv[1])
    a.extractall(tmp)

    # get train and test list
    train_list = []
    test_list = []
    face_dataset = []
    objects_dir = os.path.join(tmp, 'modelnet40v1')
    object_dirs = sorted(os.listdir(objects_dir))
    for o_id, d in enumerate(object_dirs):
        object_dir = os.path.join(objects_dir, d)

        def func(parent_dir, sub_dir, object_list):
            object_sub_dir = os.path.join(parent_dir, sub_dir)
            if os.path.isdir(object_sub_dir):
                image_files = sorted(os.listdir(object_sub_dir))
                image_files = [i_f for i_f in image_files if i_f.endswith('.jpg')]
                resize_and_save(DATA_DIR, o_id, object_sub_dir, sub_dir, image_files, object_list, face_dataset)
        func(object_dir, 'train', train_list)
        func(object_dir, 'test', test_list)

    # generate a total.txt
    generate_total_txt(DATA_DIR, train_list, test_list)

    # delete all origin images
    shutil.rmtree(tmp)


def split(image_files, num_views):
    image_files = sorted(image_files)
    objects_list = []
    object_count = len(image_files) // num_views
    for i in range(0, object_count):
        objects_list.append(image_files[i*NUM_VIEWS:(i+1)*NUM_VIEWS])
    return objects_list


def resize_and_save(path, object_id, a_object_dir, to_dir, image_files, image_list, face_dataset):
    # make dir
    base_dir = os.path.join(path, to_dir)
    base_dir = os.path.join(base_dir, a_object_dir.split('/')[-2])

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    assert (len(image_files) % NUM_VIEWS) == 0
    objects_list = split(image_files, NUM_VIEWS)
    for idx, obj in enumerate(objects_list):
        store_dir = os.path.join(base_dir, str(idx))
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        list_file_name = os.path.join(store_dir, '%d.txt' % idx)
        list_file = open(list_file_name, mode='w')
        list_file.write('%d\n' % object_id)
        list_file.write('%d\n' % NUM_VIEWS)

        for i_name in obj:
            img = cv2.imread(os.path.join(a_object_dir, i_name), cv2.IMREAD_UNCHANGED)
            if img.shape[-1] != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, g.IMAGE_SHAPE[0:2])
            new_name = i_name[:-4]+'.png'
            cv2.imwrite(os.path.join(store_dir, new_name), img)
            list_file.write(os.path.join(store_dir, new_name)+'\n')

        list_file.close()
        image_list.append([list_file_name, object_id])


def generate_total_txt(path, train_list, test_list):
    if len(train_list) == 0 or len(test_list) == 0:
        return
    # write a file call total.txt in path
    with open(os.path.join(path, 'train.txt'), mode='w') as total:
        for a_list in train_list:
            total.write(a_list[0] + ' %d\n' % (int(a_list[1])))

    with open(os.path.join(path, 'test.txt'), mode='w') as total:
        for a_list in test_list:
            total.write(a_list[0] + ' %d\n' % (int(a_list[1])))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("python preprocess_modelnet40v1.py data/modelnet40v1.zip")
        exit(-1)
    else:
        main(sys.argv)
