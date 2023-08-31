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

# scikit-image
import skimage, skimage.io, skimage.transform, skimage.restoration, skimage.filters, skimage.exposure, skimage.color, skimage.feature

DEBUG = 0
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(DEVICE)
print(f"Using {DEVICE} device")

def main():
    # CLASSES AND FUNCTIONS
    class PlanktonDataset(Dataset):
        """Plankton dataset.

        Args:
            labels (list): list of labels
            patterns (list): list of patterns
            transform (callable): transform to apply to the patterns
        """

        def __init__(self, labels, patterns, transform):
            self.labels = torch.tensor(labels, dtype = torch.long)   # labels -> tensor (long)
            self.patterns = patterns                                 # images
            self.transform = transform                               # transforms

        def __len__(self):
            return len(self.patterns)

        def __getitem__(self, idx):
            return self.transform(self.patterns[idx]), self.labels[idx] - 1
        
    def plankton_original_features(patterns, size):
        """Extract original features from patterns.

        Args:
            patterns (list): list of patterns

        Returns:
            list: list of original features
        """
        features = []
        for pattern in patterns:
            img = skimage.transform.resize(pattern, size, anti_aliasing = True)
            features.append(img)
        return features
    
    def plankton_global_features(patterns, size):
        """Extract global features from patterns.

        Args:
            patterns (list): list of patterns

        Returns:
            list: list of global features
        """
        features = []
        for pattern in patterns:
            img = skimage.transform.resize(pattern, size, anti_aliasing = True)
            img = skimage.restoration.denoise_bilateral(img, sigma_color = 0.05, sigma_spatial = 2, channel_axis = -1)
            img = skimage.filters.sobel(img)
            img = skimage.exposure.equalize_hist(img)
            p2, p98 = np.percentile(img, (2, 98))
            img = skimage.exposure.rescale_intensity(img, in_range = (p2, p98))
            features.append(img)
        return features
    
    def plankton_local_features(patterns, size):
        """Extract local features from patterns.

        Args:
            patterns (list): list of patterns

        Returns:
            list: list of local features
        """
        features = []
        for pattern in patterns:
            img = skimage.transform.resize(pattern, size, anti_aliasing = True)
            img = skimage.color.rgb2gray(img)
            threshold = skimage.filters.threshold_otsu(img)
            img = img * (img < threshold)
            img = skimage.features.canny(img, low_threshold = 0, high_threshold = 0)
            img = img.astype(np.float64)
            img = np.repeat(img[:, :, np.newaxis], 3, axis = 2)
            features.append(img)
        return features
    
    # INPUT
    datapath = 'dataset/Datas_44.mat'
    if not os.path.exists(datapath):
        raise Exception('ERROR: Dataset not found. Check the path.')
    
    mat = io.loadmat(datapath)['DATA'][0]
    x = mat[0][0]             # patterns
    t = mat[1][0]             # labels
    folds = mat[2].shape[0]   # number of folds
    shuff = mat[2]            # pre-shuffled indexes ('folds' different permutations)
    div = mat[3][0][0]        # training patterns number
    tot = mat[4][0][0]        # UNUSED: total patterns number

    # ALEXNET
    print('\nLoading AlexNet...')
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

    # DATASETS
    print('\nCreating datasets...')
    nets, patterns = [], []
    for i in range(3):
        nets.append(model)
    patterns.append(plankton_original_features(x, input_size))
    patterns.append(plankton_global_features(x, input_size))
    patterns.append(plankton_local_features(x, input_size))
    print('DATASETS CREATED')

    ### RECAP ###
    #
    # nets = [alexnet, alexnet, alexnet]
    # patterns = [original, global, local]
    # t = labels (same for all 3 datasets)
    # 
    # -> ensemble of 3 alexnet models, each trained on a different dataset
    #
    #############

    # MAIN LOOP (train and partial evaluation of each model for each fold)
    for e in range(3):   # for each net
        print(f'\n### Model {e + 1} ###')

        for fold in range(folds):   # for each fold
            print(f'\n>>> Fold {fold + 1} <<<')
            
            # dataset
            train_pattern, train_label, test_pattern, test_label = [], [], [], []
            for i in shuff[fold][:div] - 1:
                train_pattern.append(patterns[e][i])
                train_label.append(t[i])
            for i in shuff[fold][div:] - 1:
                test_pattern.append(patterns[e][i])
                test_label.append(t[i])
            num_classes = max(train_label)
            print(f'Training patterns: {len(train_pattern)}')
            print(f'Test patterns: {len(test_pattern)}')
            print(f'Number of classes: {num_classes}')

            # transforms
            data_transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale = True)
            ])

            # custom datasets
            train_dataset = PlanktonDataset(train_label, train_pattern, data_transform)
            test_dataset = PlanktonDataset(test_label, test_pattern, data_transform)

            # dataloaders
            train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
            test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

            # tuning
            nets[e].tuning = torch.nn.Sequential(
                torch.nn.Linear(4096, num_classes, bias = True),
                torch.nn.Softmax(dim = 1)
            )
            nets[e] = nets[e].to(DEVICE)

            # loss function
            loss_fn = torch.nn.CrossEntropyLoss()

            # optimizer
            optimizer = torch.optim.SGD([
                {'params': nets[e].features.parameters()},
                {'params': nets[e].avgpool.parameters()},
                {'params': nets[e].classifier.parameters()},
                {'params': nets[e].tuning.parameters(), 'lr': lr * factor}
            ], lr = lr, momentum = momentum)

            # training
            print('\nTraining...')
            nets[e].train()
            writer = SummaryWriter('runs/plankton_AGE-{}_NET-{}_FOLD-{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S"), e + 1, fold + 1))

            with tqdm(total = epochs, unit = 'epoch') as pbar:   # progress bar
                for epoch in range(epochs):                      # for each epoch
                    for i, data in enumerate(train_dataloader):  # for each batch
                        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                        optimizer.zero_grad()             # zero the gradients
                        outputs = nets[e](inputs)         # forward pass
                        loss = loss_fn(outputs, labels)   # loss
                        loss.backward()                   # backward pass
                        optimizer.step()                  # update weights

                        writer.add_scalar('Loss/train', loss.item(), epoch * iterations + i)   # logging
                    pbar.update(1)   # update progress bar
            writer.flush()
            print('TRAINING DONE')

            # partial evaluation
            print('\nEvaluating...')
            nets[e].eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for data in test_dataloader:
                    inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                    outputs = nets[e](inputs)
                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print(f'Accuracy: {100 * correct / total:.2f}%')
            print('EVALUATION DONE')

        # save model
        print('\nSaving model...')
        if not os.path.exists('output'):
            os.makedirs('output')
        torch.save(nets[e].state_dict(), 'output/model{}.pth'.format(e + 1))
        print('MODEL SAVED')
    
    # ENSEMBLE (test with sum rule)
    print('\n### Ensemble ###')
    for i in range(3):
        continue







if __name__ == '__main__':
    main()