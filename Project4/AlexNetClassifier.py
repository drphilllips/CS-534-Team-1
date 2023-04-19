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

    train_size = int(0.77 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    # Select 50 melanoma and 50 naevus images
    melanoma_db = [i for i in range(len(train_data)) if train_data[i][1] == 0]
    naevus_db = [i for i in range(len(train_data)) if train_data[i][1] == 1]
    melanoma_train_photos = random.sample(melanoma_db, 50)
    naevus_train_photos = random.sample(naevus_db, 50)

    melanoma_test_photos = random.sample([i for i in train_data if i not in melanoma_train_photos], 20)
    naevus_test_photos = random.sample([i for i in train_data if i not in naevus_train_photos], 20)

    print('Melanoma photos size: ', len(melanoma_train_photos))
    print('Naevus photos size: ', len(naevus_train_photos))

    print('Melanoma test photos size: ', len(melanoma_test_photos))
    print('Naevus test photos size: ', len(naevus_test_photos))

    # Compile train photos into 100x3x256x256 tensor, no labels
    train_photos = []
    for m in melanoma_train_photos:
        train_photos.append(train_data[m][0])
    for n in naevus_train_photos:
        train_photos.append(train_data[n][0])
    train_tensor = torch.stack(train_photos)
    print(train_tensor)
    print(len(train_tensor))

    # run alexnet on train tensor with a range of dropouts
    # ! need to do 5-Fold to find the best dropout number
    dropouts = [i/100 for i in range(0, 100, 1)]
    for d in dropouts:
        alexnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', num_classes=2, dropout=d)
        with torch.no_grad():
            output = alexnet_model(train_tensor)  # non-normalized scores
            probabilities = torch.nn.functional.softmax(output, dim=0)
            print(len(output))
            print(probabilities[0])
        break  # remove once 5-fold is implemented


if __name__ == "__main__":
    main()
