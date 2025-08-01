import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE


import numpy as np
import os
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True



class CUB():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        
        # 获取所有图片文件名
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])

        # 获取所有标签
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)  # 标签从 1 开始，减去 1 转换为 0-based index

        # 获取训练和测试的标记
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        # 按照 train_test_list 分配训练和测试数据
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        # 根据is_train标志来决定加载训练集还是测试集
        if self.is_train:
            self.train_file_list = train_file_list[:data_len]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_file_list = test_file_list[:data_len]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img_name, target = self.train_file_list[index], self.train_label[index]
            img_path = os.path.join(self.root, 'images', img_name)
        else:
            img_name, target = self.test_file_list[index], self.test_label[index]
            img_path = os.path.join(self.root, 'images', img_name)
        
        # 使用Image.open()延迟加载图像
        img = Image.open(img_path)

        # 如果是灰度图像，将其转换为 RGB 图像
        if len(img.mode) == 1:  # 判断是否是灰度图像
            img = img.convert('RGB')

        # 数据增强和预处理
        if self.is_train:
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
        else:
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_file_list)
        else:
            return len(self.test_file_list)


if __name__ == '__main__':
    dataset = CUB(root='/data/users/jw/data/CUB_200_2011')
    print(len(dataset.train_img))
    print(len(dataset.train_label))
    for data in dataset:
        print(data[0].size(), data[1])
    dataset = CUB(root='/data/users/jw/data/CUB_200_2011', is_train=False)
    print(len(dataset.test_img))
    print(len(dataset.test_label))
    for data in dataset:
        print(data[0].size(), data[1])
