import io
import scipy.io as matio
import numpy as np
from PIL import Image

import torch.utils.data


def default_loader(image_path):
    return Image.open(image_path).convert('RGB')


class dataset_stage1(torch.utils.data.Dataset):
    def __init__(self, dataset_indicator, image_path = None, data_path = None, transform=None, loader=default_loader):

        # load image paths
        if dataset_indicator is 'vireo':
            img_path_file = data_path + 'TR.txt'
        else:
            img_path_file = data_path + 'train_images.txt'

        with io.open(img_path_file, encoding='utf-8') as file:
            path_to_images = file.read().split('\n')[:-1]

        self.dataset_indicator = dataset_indicator
        self.image_path = image_path
        self.path_to_images = path_to_images
        self.transform = transform
        self.loader = loader
        #import ipdb; ipdb.set_trace()
    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]
        if self.dataset_indicator is 'vireo':
            img = self.loader(self.image_path + path)
        else:
            img = self.loader(self.image_path + path + '.jpg')

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.path_to_images)


class dataset_stage2(torch.utils.data.Dataset):
    def __init__(self, dataset_indicator, data_path = None):
        # load image paths / label file
        if dataset_indicator is 'vireo':
            ingredients = matio.loadmat(data_path + 'ingredient_train_feature.mat')['ingredient_train_feature']
            indexVectors = matio.loadmat(data_path + 'indexVector_train.mat')['indexVector_train']
        else:
            ingredients = matio.loadmat(data_path + 'ingredient_all_feature.mat')['ingredient_all_feature']
            indexVectors = matio.loadmat(data_path + 'indexVector.mat')['indexVector']

        ingredients = ingredients.astype(np.float32)
        indexVectors = indexVectors.astype(np.long)
        self.ingredients = ingredients
        self.indexVectors = indexVectors

    def __getitem__(self, index):

        # get ingredient vector
        ingredient = self.ingredients[index, :]

        # get index vector for gru input
        indexVector = self.indexVectors[index, :]

        return [indexVector, ingredient]

    def __len__(self):
        return len(self.indexVectors)



class dataset_stage3(torch.utils.data.Dataset):
    def __init__(self, dataset_indicator, image_path = None, data_path = None, transform=None, loader=default_loader, mode = None):

        # load image paths / label file
        if mode is 'train':
            if dataset_indicator is 'vireo':
                with io.open(data_path + 'TR.txt', encoding='utf-8') as file:
                    path_to_images = file.read().split('\n')[:-1]
                labels = matio.loadmat(data_path + 'train_label.mat')['train_label'][0]

                ingredients = matio.loadmat(data_path + 'ingredient_train_feature.mat')['ingredient_train_feature']
                indexVectors = matio.loadmat(data_path + 'indexVector_train.mat')['indexVector_train']
            else:
                with io.open(data_path + 'train_images.txt', encoding='utf-8') as file:
                    path_to_images = file.read().split('\n')[:-1]
                with io.open(data_path + 'train_labels.txt', encoding='utf-8') as file:
                    labels = file.read().split('\n')[:-1]

                ingredients = matio.loadmat(data_path + 'ingredient_all_feature.mat')['ingredient_all_feature']
                indexVectors = matio.loadmat(data_path + 'indexVector.mat')['indexVector']

        elif mode is 'test':
            if dataset_indicator is 'vireo':
                with io.open(data_path + 'TE.txt', encoding='utf-8') as file:
                    path_to_images = file.read().split('\n')[:-1]
                labels = matio.loadmat(data_path + 'test_label.mat')['test_label'][0]

                ingredients = matio.loadmat(data_path + 'ingredient_test_feature.mat')['ingredient_test_feature']
                indexVectors = matio.loadmat(data_path + 'indexVector_test.mat')['indexVector_test']
            else:
                with io.open(data_path + 'test_images.txt', encoding='utf-8') as file:
                    path_to_images = file.read().split('\n')[:-1]
                with io.open(data_path + 'test_labels.txt', encoding='utf-8') as file:
                    labels = file.read().split('\n')[:-1]

                ingredients = matio.loadmat(data_path + 'ingredient_all_feature.mat')['ingredient_all_feature']
                indexVectors = matio.loadmat(data_path + 'indexVector.mat')['indexVector']

        elif mode is 'val':
            if dataset_indicator is 'vireo':
                with io.open(data_path + 'VAL.txt', encoding='utf-8') as file:
                    path_to_images = file.read().split('\n')[:-1]
                labels = matio.loadmat(data_path + 'val_label.mat')['validation_label'][0]

                ingredients = matio.loadmat(data_path + 'ingredient_val_feature.mat')['ingredient_val_feature']
                indexVectors = matio.loadmat(data_path + 'indexVector_val.mat')['indexVector_val']
            else:
                with io.open(data_path + 'val_images.txt', encoding='utf-8') as file:
                    path_to_images = file.read().split('\n')[:-1]
                with io.open(data_path + 'val_labels.txt', encoding='utf-8') as file:
                    labels = file.read().split('\n')[:-1]

                ingredients = matio.loadmat(data_path + 'ingredient_all_feature.mat')['ingredient_all_feature']
                indexVectors = matio.loadmat(data_path + 'indexVector.mat')['indexVector']

        else:
            assert 1<0, 'Please fill mode with any of train/val/test to facilitate dataset creation'

        self.dataset_indicator = dataset_indicator
        self.image_path = image_path
        self.path_to_images = path_to_images
        self.labels = np.array(labels, dtype=int)

        ingredients = ingredients.astype(np.float32)
        indexVectors = indexVectors.astype(np.long)
        self.ingredients = ingredients
        self.indexVectors = indexVectors

        self.transform = transform
        self.loader = loader
        #import ipdb; ipdb.set_trace()

    def __getitem__(self, index):
        # get image matrix and transform to tensor
        path = self.path_to_images[index]
        if self.dataset_indicator is 'vireo':
            img = self.loader(self.image_path + path)
        else:
            img = self.loader(self.image_path + path + '.jpg')

        if self.transform is not None:
            img = self.transform(img)

        # get label
        label = self.labels[index]
        if self.dataset_indicator is 'food101':
            label += 1 #make labels 1-indexed to be consistent with vireo data settings
            # get ingredient vector
            ingredients = self.ingredients[label-1]
            # get index vector for gru input
            indexVectors = self.indexVectors[label-1]
        else:
            # get ingredient vector
            ingredients = self.ingredients[index]
            # get index vector for gru input
            indexVectors = self.indexVectors[index]            
            
        return [img, indexVectors, ingredients, label]
    def __len__(self):
        return len(self.path_to_images)



def build_dataset(train_stage, image_path, data_path, transform, mode, dataset_indicator):

    if  train_stage == 1: #to pretrain image channel
        dataset = dataset_stage1(dataset_indicator, image_path = image_path, data_path = data_path, transform=transform)
    elif train_stage == 2: #to pretrain ingredient channel
        dataset = dataset_stage2(dataset_indicator, data_path = data_path)
    elif train_stage == 3:  #to train the whole network
        dataset = dataset_stage3(dataset_indicator, image_path = image_path, data_path = data_path, transform=transform, mode = mode)
    else:
        assert 1 < 0, 'Please fill the correct train stage!'

    return dataset