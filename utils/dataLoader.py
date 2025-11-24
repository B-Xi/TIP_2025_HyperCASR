from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from utils.my_dataset import AllDataset, LabelDataset
from utils.split_data import initDataset
from torch.utils.data import ConcatDataset
import copy
import numpy as np


def custom_augmentation(image):
    image = np.transpose(image, (1, 0, 2))  # 转置操作
    image = np.flipud(image)  # 垂直翻转操作
    return image


def getDataLoader(args, info):
    dataset: dict = initDataset(args, info)

    all_dataset = AllDataset(dataset['data'], args, info)
    label_dataset = LabelDataset(dataset['data'], dataset['gt'], args, info)

    known_train_dataset = Subset(label_dataset, dataset['known_train_index'])
    known_test_dataset = Subset(label_dataset, dataset['known_test_index'])
    unknown_test_dataset = Subset(label_dataset, dataset['unknown_test_index'])
    unknown_unknown_dataset = Subset(label_dataset, dataset['unknown_unknown_index'])

    data_transform = transforms.Compose([
        transforms.Lambda(lambda img: custom_augmentation(img)),  # 自定义扩增方法
        transforms.ToTensor()  # 转换为张量
    ])
    augmented_train_dataset = copy.deepcopy(known_train_dataset)
    augmented_known_train_dataset = copy.deepcopy(known_train_dataset)
    for i in range(4):
        augmented_known_train_dataset.transform = data_transform
        augmented_train_dataset = ConcatDataset([augmented_train_dataset, augmented_known_train_dataset])

    return {
        'all': DataLoader(all_dataset, batch_size=args.batch, shuffle=False),
        'known': {
            'train': DataLoader(known_train_dataset, batch_size=args.batch, shuffle=True),
            'test': DataLoader(known_test_dataset, batch_size=args.batch, shuffle=True),
            'augtrain': DataLoader(augmented_train_dataset, batch_size=args.batch, shuffle=True)
        },
        'unknown': {
            'test': DataLoader(unknown_test_dataset, batch_size=args.batch, shuffle=False),
            'unknown': DataLoader(unknown_unknown_dataset, batch_size=args.batch, shuffle=False),
            'test_index': dataset['unknown_test_index']
        },
        'gt': dataset['gt'] 
    }
