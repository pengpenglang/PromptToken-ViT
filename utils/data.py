import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


def get_dataset(dataset_name, data_transform):
    if dataset_name == 'imagenet':
        train_dataset = ImageNetDataset('/data_1/langwenpeng/imagenet/train_index.txt', '/data_1/langwenpeng/imagenet/task2_train', transform=data_transform["train"])
        val_dataset = ImageNetDataset('/data_1/langwenpeng/imagenet/val_index.txt', '/data_1/langwenpeng/imagenet/val', transform=data_transform["val"])
    elif dataset_name == 'caltech256':
        dataset = Caltech256(root='/data_1/langwenpeng/caltech256', transform=data_transform['train'])
        torch.manual_seed(0)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    elif dataset_name == 'cifar100':
        train_dataset = CIFAR10WithPlaceHolder(root='/data_1/langwenpeng/cifar100', train=True, transform=data_transform['train'], download=True)
        val_dataset = CIFAR10WithPlaceHolder(root='/data_1/langwenpeng/cifar100', train=False, transform=data_transform['val'], download=True)
    elif dataset_name == 'dtd':
        train_dataset = DTDWithPlaceHolder(root='/data_1/langwenpeng/dtd', split='train', transform=data_transform['train'], download=True)
        val_dataset = DTDWithPlaceHolder(root='/data_1/langwenpeng/dtd', split='val', transform=data_transform['val'], download=True)
    return train_dataset, val_dataset


class ImageNetDataset(Dataset):

    def __init__(self, txt_path, data_path, transform=None):
        self.transform = transform
        self.images_path = []
        self.bndbox = []
        # 记录图片所在文件夹的序号
        self.label = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                words = line.split(' ')
                self.images_path.append(data_path + '/' + words[0])
                self.label.append(int(words[1]))
                positions = []
                for i in range(2, len(words)):
                    # 将每个box的四个坐标分开
                    positions.extend(words[i].split(','))
                box = []
                for i in range(0, len(positions), 4):
                    box.extend([int(positions[i]), int(positions[i + 1]), int(positions[i + 2]), int(positions[i + 3])])
                self.bndbox.append(box)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')
        label = self.label[item]
        box = self.bndbox[item]
        if self.transform is not None:
            before_w, before_h = img.size
            img = self.transform(img)
            after_w, after_h = 224, 224
            for i in range(0, len(box), 4):
                box[i] = box[i] * after_w / before_w
                box[i + 1] = box[i + 1] * after_h / before_h
                box[i + 2] = box[i + 2] * after_w / before_w
                box[i + 3] = box[i + 3] * after_h / before_h
        sequence = torch.zeros(196)
        for i in range(0, len(box), 4):
            xmin = min(13, int(box[i] // 16 + (box[i] % 16 > 8)))
            ymin = min(13, int(box[i + 1] // 16 + (box[i + 1] % 16 > 8)))
            xmax = max(min(13, int(box[i + 2] // 16 - (box[i + 2] % 16 < 8))), xmin)
            ymax = max(min(13, int(box[i + 3] // 16 - (box[i + 3] % 16 < 8))), ymin)
            for i in range(ymin, ymax + 1):
                for j in range(xmin, xmax + 1):
                    sequence[i * 14 + j] = 1
        # 生成sequence后将box对齐
        box.extend([0, 0, 0, 0] * (25 - len(box) // 4))
        # box传入用于对比gt需要
        return img, label, sequence, torch.tensor(box).type(torch.float)

    def __len__(self):
        return len(self.images_path)


class Caltech256(datasets.VisionDataset):

    def __init__(self, root, transform=None, target_transform=None, loader=datasets.folder.default_loader):
        super(Caltech256, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader
        self.samples = self.make_dataset(self.root)

    def make_dataset(self, dir):
        images = []
        labels = []
        for target in os.listdir(dir):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for path, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.endswith('.jpg') or fname.endswith('.jpeg') or fname.endswith('.png'):
                        path = os.path.join(d, fname)
                        item = (path, int(target[:3]) - 1)
                        images.append(item[0])
                        labels.append(item[1])
        return list(zip(images, labels))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, 0, 0

    def __len__(self):
        return len(self.samples)


class CIFAR10WithPlaceHolder(datasets.cifar.CIFAR100):

    def __getitem__(self, index):
        img, target = super(CIFAR10WithPlaceHolder, self).__getitem__(index)
        return img, target, 0, 0


class DTDWithPlaceHolder(datasets.dtd.DTD):

    def __getitem__(self, index):
        img, target = super(DTDWithPlaceHolder, self).__getitem__(index)
        return img, target, 0, 0
