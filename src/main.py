# Alessio Cocco 2087635, Andrea Valentinuzzi 2090451, Giovanni Brejc 2096046
# Plankton Pre-Processing and Classification with AlexNet
# UniPD 2022/23 - Deep Learning Project

# based
import os                       # paths
from scipy import io            # MATLAB files
import numpy as np              # arrays
from datetime import datetime   # time
from tqdm import tqdm           # progress bar

# torch
import torch
from torch.utils.data import Dataset                     # custom dataset
from torch.utils.data import DataLoader                  # dataloader for batches
import torchvision.models as models                      # AlexNet
from torchvision.models.alexnet import AlexNet_Weights   # AlexNet weights
from torchvision.transforms import v2                    # transforms
from PIL import Image                                    # images for transforms
from torch.utils.tensorboard import SummaryWriter        # logging

DEBUG = False   # debug mode (show only image transforms)
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(DEVICE)
print(f"Using {DEVICE} device")

def debug():
    import torchvision.utils
    x = np.array(io.loadmat('dataset/Datas_44.mat')['DATA'])[0][0][0]
    base = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Resize((227, 227), antialias = True),
    ])
    transform = v2.Compose([
        v2.GaussianBlur(5, sigma = (0.1, 2.0)),
        v2.RandomAdjustSharpness(1.5, 1),
        v2.RandomAutocontrast(1),
        v2.ToPILImage()
    ])

    for i in range(1):
        img = transform(base(Image.fromarray(x[i])))
        img.show()

    return

def main():
    # CUSTOM DATASET
    class PlanktonDataset(Dataset):
        def __init__(self, labels, patterns, transform):
            self.labels = torch.tensor(labels, dtype = torch.long)   # labels -> tensor (long)
            self.patterns = patterns                                 # images
            self.transform = transform                               # transforms

        def __len__(self):
            return len(self.patterns)

        def __getitem__(self, idx):
            return self.transform(self.patterns[idx]), self.labels[idx] - 1
        
    # DATASET
    datapath = 'dataset/Datas_44.mat'
    if not os.path.exists(datapath):
        raise Exception('ERROR: Dataset not found. Check the path.')
    
    mat = np.array(io.loadmat(datapath)['DATA'])[0]
    x = mat[0][0]             # patterns
    t = mat[1][0]             # labels
    folds = mat[2].shape[0]   # number of folds
    shuff = mat[2]            # pre-shuffled indexes ('folds' different permutations)
    div = mat[3][0][0]        # training patterns number
    tot = mat[4][0][0]        # UNUSED: total patterns number
    
    # ALEXNET
    print('Loading AlexNet...')
    model = models.alexnet(weights = AlexNet_Weights.DEFAULT)
    layers = list(model.classifier.children())[:-1]
    model.classifier = torch.nn.Sequential(*layers)
    print('LOADING DONE')

    # PARAMETERS
    input_size = (227, 227)          # image size
    batch_size = 32                  # batch size
    lr = 1e-4                        # learning rate
    factor = 20                      # learning rate factor for tuning
    epochs = 50                      # fixed number of epochs
    momentum = 0.9                   # momentum
    iterations = div // batch_size   # iterations per epoch

    # MAIN LOOP
    for fold in range(folds):   # for each fold
        print(f'\n### Fold {fold + 1}/{folds} ###')

        # dataset
        train_pattern , train_label, test_pattern, test_label = [], [], [], []
        for i in shuff[fold][:div] - 1:
            train_pattern.append(Image.fromarray(x[i]))   # 0:div pre-shuffled pattern from fold
            train_label.append(t[i])                      # 0:div pre-shuffled label from fold
        for i in shuff[fold][div:] - 1:
            test_pattern.append(Image.fromarray(x[i]))    # div:tot pre-shuffled pattern from fold
            test_label.append(t[i])                       # div:tot pre-shuffled label from fold
        num_classes = max(train_label)                    # number of classes
        print(f'Training patterns: {len(train_pattern)}')
        print(f'Testing patterns: {len(test_pattern)}')
        print(f'Number of classes: {num_classes}')

        # transforms
        data_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale = True),
            v2.Resize(input_size, antialias = True),
            #v2.GaussianBlur(5, sigma = (0.1, 2.0)),
            #v2.RandomAdjustSharpness(1.5, 1),
            #v2.RandomAutocontrast(1)
        ])

        # custom dataset
        train_data = PlanktonDataset(labels = train_label, patterns = train_pattern, transform = data_transform)
        test_data = PlanktonDataset(labels = test_label, patterns = test_pattern, transform = data_transform)

        # dataloader
        train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = False)
        test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

        # tuning
        model.tuning = torch.nn.Sequential(
            torch.nn.Linear(4096, num_classes, bias = True),
            torch.nn.Softmax(dim = 1)
        )
        model = model.to(DEVICE)   # GPU computing, if available

        # loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # optimizer
        optimizer = torch.optim.SGD([
            {'params': model.features.parameters()},
            {'params': model.avgpool.parameters()},
            {'params': model.classifier.parameters()},
            {'params': model.tuning.parameters(), 'lr': lr * factor}
        ], lr = lr, momentum = momentum)

        # training
        print('\nTraining...')
        model.train()
        writer = SummaryWriter('runs/plankton_AGE-{}_FOLD-{}'.format(datetime.now().strftime("%Y%m%d.%H%M%S"), fold + 1))

        with tqdm(total = epochs, unit = 'epoch') as pbar:   # progress bar
            for epoch in range(epochs):                      # for each epoch
                for i, data in enumerate(train_dataloader):
                    # data (GPU computing, if available)
                    inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                    # training
                    optimizer.zero_grad()             # zero the gradients
                    outputs = model(inputs)           # forward pass
                    loss = loss_fn(outputs, labels)   # loss
                    loss.backward()                   # backward pass
                    optimizer.step()                  # update weights

                    # logging
                    writer.add_scalar("Loss/train", loss.item(), epoch * iterations + i)
                pbar.update(1)   # update progress bar
        writer.flush()
        print('TRAINING DONE')

        # testing
        print('\nTesting...')
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_dataloader:
                # data
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                # testing
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # logging
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Accuracy: {100 * correct / total}")
        print('TESTING DONE')

    # save model
    print('\nSaving model...')
    torch.save(model.state_dict(), 'output/model.pth')
    print('MODEL SAVED')

    # TODO: statistics
    print('\nComputing statistics...')
    print('STATISTICS DONE')

    return

if __name__ == '__main__':
    main() if not DEBUG else debug()