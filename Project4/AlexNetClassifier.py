import random
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


def main():
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = ImageFolder('complete_mednode_dataset', transform=transformation)
    print(train_data)

    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    # Select 50 melanoma and 50 naevus images
    melanoma_db = [i for i in range(len(train_data)) if train_data[i][1] == 0]
    naevus_db = [i for i in range(len(train_data)) if train_data[i][1] == 1]
    melanoma_train_photos = random.sample(melanoma_db, 50)
    naevus_train_photos = random.sample(naevus_db, 50)

    print('Melanoma photos size: ',len(melanoma_train_photos))
    print('Naevus photos size: ',len(naevus_train_photos))


if __name__ == "__main__":
    main()
