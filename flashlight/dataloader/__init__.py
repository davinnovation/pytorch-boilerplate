import albumentations as AB
from albumentations.pytorch import ToTensor, ToTensorV2
import torchvision

DATA_DICT = {"MNIST": torchvision.datasets.MNIST}

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Grayscale(num_output_channels=3), torchvision.transforms.ToTensor()]
)


def get_datalaoder(data, root="../datasets/", split="train"):
    if data in ["MNIST"]:  # if torchvision
        if split == "val":
            print(f"{data} dataset dosen't support validation set. val replaced by train")
        if split in ["train", "val"]:
            return DATA_DICT[data](root=root, train=True, download=True, transform=transform)
        else:
            return DATA_DICT[data](root=root, train=False, download=True, transform=transform)
