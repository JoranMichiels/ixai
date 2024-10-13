import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Imagenette

from helpers import normalize_tensor

root = '/esat/biomeddata/jmichiel/'

SAVES = '/esat/biomeddata/jmichiel/' + 'imagenet_pytorch/'


class ImageNetteDataset(Dataset):
    @classmethod
    def transform(self, X, set='val'):
        # Pytorch defined transforms https://github.com/pytorch/examples/blob/main/imagenet/main.py
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if set == 'train':
            input_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif set == 'val' or set == 'test':
            input_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            raise NotImplementedError

        return input_transform(X)

    def __init__(self, set='val'):
        self.dataset = Imagenette(root=root, split=set,
                                  transform=ImageNetteDataset.transform, download=False)
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return *self.dataset[idx], idx


class ClickMetteDataset(Dataset):
    def __init__(self, remove_dup=True, set='train'):
        self.save_url = SAVES
        # The ten classes that are used in the ImageNette dataset
        self.filter_labels = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
        self.labels = []
        self.images = []
        self.exps = []

        if set == 'train':
            ending = ''
        elif set == 'val':
            ending = 'val'

        self.images = torch.load(self.save_url + f'10images{ending}.pt')
        self.labels = torch.load(self.save_url + f'10labels{ending}.pt')
        self.exps = torch.load(self.save_url + f'10exps{ending}.pt')
        self.length = len(self.labels)
        label_to_encoded = {label: idx for idx, label in enumerate(self.filter_labels)}
        self.labels = torch.tensor([label_to_encoded[label.item()] for label in self.labels])

        # Removing duplicate explanations (some images have more than one explanation)
        if remove_dup:
            self.images, inverse = torch.unique(self.images, sorted=True, return_inverse=True, dim=0)
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(self.images.size(0)).scatter_(0, inverse, perm)
            self.labels = self.labels[perm]
            self.exps = self.exps[perm]
            self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #  Also return the index, to be able to get the explanation afterwards
        return self.images[idx], self.labels[idx], idx

    # This function is used to get the explanation of the image
    def get_exp(self, idx):
        blur_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.GaussianBlur(9, 10),
                                             transforms.Resize((224, 224))])
        exps = blur_transform(self.exps[idx])

        # Normalize the explanation
        assert exps.dim() == 4
        exps = normalize_tensor(exps, method='minmax', samplewise=True)

        return exps
