# coding=utf-8

from __future__ import division
from __future__ import print_function

import os
import random
import sys
import zipfile
import cv2
import shutil
import config as g

g.set_seed()


NUM_VIEWS = g.NUM_VIEWS

DATA_DIR = os.path.join('data', 'M-PIE')

FACE_DATASET_DIR = os.path.join(DATA_DIR, 'face-dataset')


# 需要把数据集解压到参数指定的目录里面
def main(argv):
    tmp = os.path.join(DATA_DIR, 'tmp')

    # delete pre-made dir
    shutil.rmtree(os.path.join(DATA_DIR, 'train'), ignore_errors=True)
    shutil.rmtree(os.path.join(DATA_DIR, 'test'), ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(FACE_DATASET_DIR, ignore_errors=True)

    # make tmp path
    os.makedirs(tmp)

    # check if it is a zip file
    if not zipfile.is_zipfile(argv[1]):
        raise ValueError(argv[1] + 'is not a zip file')
    if not zipfile.is_zipfile(argv[2]):
        raise ValueError(argv[2] + 'is not a zip file')

    # unzip data
    a = zipfile.ZipFile(argv[1])
    a.extractall(tmp)
    b = zipfile.ZipFile(argv[2])
    b.extractall(tmp)

    # move every image to tmp
    for d in os.listdir(tmp):
        sub = os.path.join(tmp, d)
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                shutil.move(os.path.join(sub, f), os.path.join(tmp, f))

    # list all image
    all_images = sorted(os.listdir(tmp))
    all_images = [x for x in all_images if x.endswith('.png')]  # delete file name that are not *.png

    # get the 7-th face image as face dataset
    face_dataset = [x for x in all_images if x.endswith('07.png')]
    with open(os.path.join(DATA_DIR, 'face_dataset.txt'), 'w') as f:
        os.makedirs(FACE_DATASET_DIR)
        for face in face_dataset:
            f.write(os.path.join(FACE_DATASET_DIR, face) +
                    ' '+str(int(face.split('_')[0])-1) + '\n')
            shutil.copy(os.path.join(tmp, face), os.path.join(FACE_DATASET_DIR, face))

    # get all people's id
    ids = get_people_ids(all_images)
    print('train-count: ' + str(len(ids)))

    # get train and test list
    train_list = []
    test_list = []
    for a_people_id in ids:
        # find all all_images belong to this id
        images = [image for image in all_images if image.startswith(a_people_id)]
        random.shuffle(images)
        train_images = images[:NUM_VIEWS]
        test_images = images[-NUM_VIEWS:]
        resize_and_save(DATA_DIR, tmp, 'train', a_people_id, train_images, train_list)
        resize_and_save(DATA_DIR, tmp, 'test', a_people_id, test_images, test_list)

    # generate a total.txt
    generate_total_txt(DATA_DIR, train_list, test_list)

    # delete all origin images
    shutil.rmtree(tmp)


def resize_and_save(path, src_dir, to_dir, people_id, images, image_list):
    new_dir = os.path.join(path, to_dir)
    new_dir = os.path.join(new_dir, people_id)
    print(new_dir)
    # make dir
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    list_path = os.path.join(new_dir, 'list.txt')
    f = open(list_path, mode='w')
    image_list.append([list_path, people_id])
    f.write('%d\n' % int(people_id))
    f.write('%d\n' % NUM_VIEWS)
    for image in images:
        img = cv2.imread(os.path.join(src_dir, image))
        img = cv2.resize(img, g.IMAGE_SHAPE[0:2])
        cv2.imwrite(os.path.join(new_dir, image), img)
        f.write(os.path.join(new_dir, image)+'\n')
    f.close()


def get_people_ids(images):
    all_ids = [image.split('_')[0] for image in images]
    single_ids = sorted(list(set(all_ids)))
    return single_ids


def generate_total_txt(path, train_list, test_list):
    if len(train_list) == 0 or len(test_list) == 0:
        return
    # write a file call total.txt in path
    with open(os.path.join(path, 'train.txt'), mode='w') as total:
        for a_list in train_list:
            total.write(a_list[0] + ' %d\n' % (int(a_list[1]) - 1))

    with open(os.path.join(path, 'test.txt'), mode='w') as total:
        for a_list in test_list:
            total.write(a_list[0] + ' %d\n' % (int(a_list[1]) - 1))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("python preprocess_multi_pie.py data/MultiPIE_Lighting_128.zip data/MultiPIE_test_128.zip")
        exit(-1)
    else:
        main(sys.argv)
