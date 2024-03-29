import random
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold
import numpy as np


def main():
    # Create transformation for images to tensor
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data = ImageFolder('complete_mednode_dataset', transform=transformation)

    # Select 50 melanoma and 50 naevus images for training
    # Need to build in indices in order to indicate which training photos are included
    # Gives Tensor boolean reference errors if you try to compare Tensors instead of indices
    melanoma_db = []
    naevus_db = []
    for i in range(len(data)):
        if data[i][1] == 0:
            melanoma_db.append([data[i][0], i])
        if data[i][1] == 1:
            naevus_db.append([data[i][0], i])
    melanoma_train_photos = random.sample(melanoma_db, 50)
    naevus_train_photos = random.sample(naevus_db, 50)

    # Select 20 melanoma and 20 naevus images for testing
    # Reference training indices to avoid repeats in testing set
    test_melanoma_db = []
    for i in range(len(melanoma_db)):
        found = False
        for j in range(len(melanoma_train_photos)):
            if melanoma_db[i][1] == melanoma_train_photos[j][1]:
                found = True
        if not found:
            test_melanoma_db.append(melanoma_db[i][0])
    test_naevus_db = []
    for i in range(len(naevus_db)):
        found = False
        for j in range(len(naevus_train_photos)):
            if naevus_db[i][1] == naevus_train_photos[j][1]:
                found = True
        if not found:
            test_naevus_db.append(naevus_db[i][0])
    melanoma_test_photos = random.sample(test_melanoma_db, 20)
    naevus_test_photos = random.sample(test_naevus_db, 20)

    # Remove indices from training data
    for i in range(len(melanoma_train_photos)):
        melanoma_train_photos[i] = melanoma_train_photos[i][0]
        naevus_train_photos[i] = naevus_train_photos[i][0]

    print('Melanoma photos size: ', len(melanoma_train_photos))
    print('Naevus photos size: ', len(naevus_train_photos))

    print('Melanoma test photos size: ', len(melanoma_test_photos))
    print('Naevus test photos size: ', len(naevus_test_photos))

    # Compile train photos into 100x3x256x256 tensor, no labels
    train_photos = []
    train_labels = []
    for m in melanoma_train_photos:
        train_photos.append(m)
        train_labels.append(1)
    for n in naevus_train_photos:
        train_photos.append(n)
        train_labels.append(0)
    train_tensor = np.array(torch.stack(train_photos))
    train_labels = np.array(train_labels)

    # run alexnet on train tensor with a range of dropouts
    # 5-Fold to find the best dropout number

    # Perform 5-fold cross-validation to find the best dropout rate
    kfold = KFold(n_splits=5, shuffle=True)

    dropout_rates = np.arange(0, 1.1, 0.1)
    results = []
    best_dropout_rate = None
    best_accuracy = 0.0

    for d in dropout_rates:
        fold_accuracy = []
        # For each fold split of train and validation data...
        for train_idx, val_idx in kfold.split(train_tensor, train_labels):
            train_idx = train_idx.astype(int)
            # Store train and validation tensors and labels
            train_x, train_y = train_tensor[train_idx], train_labels[train_idx]

            train_x = torch.tensor(train_x)
            train_y = torch.tensor(train_y)

            # build the AlexNet model
            model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', num_classes=2, dropout=d)
            model.eval()

            with torch.no_grad():
                output = model(train_x)

            # train the model
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            accuracy = torch.sum(predictions == train_y).item() / len(train_y)
            fold_accuracy.append(accuracy)

            # update fold results
            average_accuracy = np.mean(fold_accuracy)
            if average_accuracy > best_accuracy:
                best_dropout_rate = d
                best_accuracy = average_accuracy

    print("Best dropout rate:", best_dropout_rate)
    print("Best accuracy:", best_accuracy, '\n')

    # Compile test photos into 100x3x256x256 tensor
    test_photos = []
    test_labels = []
    for m in melanoma_test_photos:
        test_photos.append(m)
        test_labels.append(1)
    for n in naevus_test_photos:
        test_photos.append(n)
        test_labels.append(0)
    test_tensor = torch.stack(test_photos)
    test_labels = torch.Tensor(np.array(test_labels))

    # Train and test model using the best dropout rate
    test_model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', num_classes=2, dropout=best_dropout_rate)
    test_model.eval()
    with torch.no_grad():
        test_output = test_model(test_tensor)
    test_probabilities = torch.nn.functional.softmax(test_output, dim=1)
    test_predictions = torch.argmax(test_probabilities, dim=1)
    test_accuracy = torch.sum(test_predictions == test_labels).item() / len(test_labels)
    print("Test Accuracy: "+str(test_accuracy))


if __name__ == "__main__":
    main()
