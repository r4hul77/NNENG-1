import torchvision.io
from torch.utils.data import Dataset
import os
import re
import cv2
import random
import torch
import PIL as pil
import torchvision.transforms as transforms


class Image_Dataset(Dataset):

    def __init__(self, image_base_dir=None, transform = None, target_transform = None):

        """

        classes:The classes in the dataset

        image_base_dir:The directory of the folders containing the images

        transform:The trasformations for the Images

        Target_transform:The trasformations for the target

        """

        self.image_base_dir = image_base_dir

        self.transform = transform

        self.target_transform = target_transform

        self.string_label_idx = {}

        self.imgs_names = os.listdir(image_base_dir)

        self.img_by_category = {}

        self.total_per_catorgory = {}

    def set_img_base_dir(self, image_base_dir):
        self.image_base_dir = image_base_dir

    def set_imgs_names(self, imgs_names):
        self.imgs_names = imgs_names

    def init(self):
        list(map(self.map_labels, self.imgs_names))

    def map_labels(self, string):
        label = self.get_label(string)
        if label not in self.string_label_idx:
            self.string_label_idx[label] = len(self.string_label_idx)
        if label not in self.img_by_category:
            self.img_by_category[label] = []
            self.total_per_catorgory[label] = 0

        self.img_by_category[label].append(string)
        self.total_per_catorgory[label] += 1

    def __len__(self):

        return len(self.imgs_names)

    def __getitem__(self,idx):

        image_path = os.path.join(self.image_base_dir, self.imgs_names[idx])

        image = torchvision.io.read_image(image_path).float()

        label = torch.Tensor([self.label_idx(self.imgs_names[idx])]).long()

        if self.transform:

            image = self.transform(image)

        if self.target_transform:

            label = self.target_transform(self.label_idx(self.imgs_names[idx]))

        return image, label

    def label_idx(self, img_name):
        return self.string_label_idx[self.get_label(img_name)]

    @staticmethod
    def get_label(string):
        return re.findall(r'(\w+?)(\d+)', string)[0][0]

    def get_classes(self):
        return list(self.string_label_idx.keys())

    def get_number_datapoints(self, label):
        return self.total_per_catorgory[label]

    def get_all_datapoints(self, label):
        return self.img_by_category[label]

    def split(self, percentage_test):
        test_totals = dict( [(key, int(self.get_number_datapoints(key)* percentage_test)) for key in self.get_classes()])
        train_dataset = Image_Dataset(transform=self.transform, target_transform=self.target_transform)
        test_dataset = Image_Dataset(transform=self.transform, target_transform=self.target_transform)
        test_labels = []
        train_labels = []

        for cls in self.get_classes():
            labels = self.get_all_datapoints(cls)
            random.shuffle(labels)
            test_labels.extend(labels[:test_totals[cls]])
            train_labels.extend(labels[test_totals[cls]:])

        train_dataset.set_imgs_names(train_labels)
        test_dataset.set_imgs_names(test_labels)
        train_dataset.set_img_base_dir(self.image_base_dir)
        test_dataset.set_img_base_dir(self.image_base_dir)
        train_dataset.init()
        test_dataset.init()
        return train_dataset, test_dataset

    def split_train_val_test(self, percentage_test, percentage_val):
        train, test = self.split(percentage_test)
        train, val = train.split(percentage_val)
        return train, val, test

# Unittest to check the functionality of the ImageDataset class
import unittest

class TestImageDataset(unittest.TestCase):

        def test_get_label(self):
            self.assertEqual(Image_Dataset.get_label("Label12345.jpg"), "Label")

        def test_map_labels(self):
            image_dataset = Image_Dataset(classes=None, image_base_dir="./weatherDataset")
            self.assertEqual(image_dataset.get_classes(), ['cloudy', 'rain', 'shine'])

if __name__ == '__main__':
    unittest.main()
