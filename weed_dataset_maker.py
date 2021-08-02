from torchvision import datasets
from torchvision import transforms
from preprocessor import preprocess

def create_weed_dataset(is_train=True):
    root = "data/val"
    if is_train:
        root = "data/train"
    preprocess()
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return datasets.ImageFolder(root, data_transforms)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataloader = DataLoader(create_weed_dataset(is_train=True), batch_size=4, shuffle=True)
    for image, label in dataloader:
        print(image)
        print(label)
        exit(0)
