import random
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold
import numpy as np


def main():
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = ImageFolder('complete_mednode_dataset', transform=transformation)
    # print(train_data)

    train_size = int(0.77 * len(train_data))
    test_size = len(train_data) - train_size
    train_data, test_data = torch.utils.data.random_split(train_data, [train_size, test_size])

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
    train_labels = []
    for m in melanoma_train_photos:
        train_photos.append(train_data[m][0])
        train_labels.append(1)
    for n in naevus_train_photos:
        train_photos.append(train_data[n][0])
        train_labels.append(0)
    train_tensor = np.array(torch.stack(train_photos))
    train_labels = np.array(train_labels)
    print(train_tensor)
    print(len(train_tensor))

    # run alexnet on train tensor with a range of dropouts
    # TODO need to do 5-Fold to find the best dropout number

    # Perform 5-fold cross-validation to find the best dropout rate
    kfold = KFold(n_splits=5, shuffle=True)

    dropout_rates = np.arange(0, 1.1, 0.1)
    results = []
    best_dropout_rate = None
    best_accuracy = 0.0

    for d in dropout_rates:
        print('Dropout rate: ', d)
        fold_accuracy = []
        # For each fold split of train and validation data...
        for train_idx, val_idx in kfold.split(train_tensor, train_labels):
            train_idx = train_idx.astype(int)
            val_idx = val_idx.astype(int)
            # Store train and validation tensors and labels
            train_x, train_y = train_tensor[train_idx], train_labels[train_idx]
            val_x, val_y = train_tensor[val_idx], train_labels[val_idx]

            train_x = torch.tensor(train_x)
            train_y = torch.tensor(train_y)
            val_x = torch.tensor(val_x)
            val_y = torch.tensor(val_y)

            # TODO: build the AlexNet model
            model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', num_classes=2, dropout=d)
            model.eval()

            with torch.no_grad():
                output = model(train_x)
            print(output[0])

            # model = AlexNet(weights=None, include_top=True, input_shape=(224, 224, 3), classes=2)
            #
            # model.compile(optimizer=tf.keras.optimizers.Adam(),
            #               loss='binary_crossentropy',
            #               metrics=['accuracy'])
            #
            # model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=0)

            # score = model.evaluate(val_x, val_y, verbose=0)
            # fold_accuracy.append(score[1])

            probabilities = torch.nn.functional.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            accuracy = torch.sum(predictions == train_y).item() / len(train_y)
            fold_accuracy.append(accuracy)

            average_accuracy = np.mean(fold_accuracy)
            if average_accuracy > best_accuracy:
                best_dropout_rate = d
                best_accuracy = average_accuracy

            print("Best dropout rate:", best_dropout_rate)
            print("Best accuracy:", best_accuracy, '\n')

# with torch.no_grad():
#     output = model(train_tensor)  # non-normalized scores
#     probabilities = torch.nn.functional.softmax(output, dim=0)
#     print(len(output))
#     print(probabilities[0])
# break  # remove once 5-fold is implemented


if __name__ == "__main__":
    main()
