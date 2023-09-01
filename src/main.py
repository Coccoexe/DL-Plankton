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

DEBUG = 3
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(DEVICE)
print(f"Using {DEVICE} device")

def test_global_features():
    """Image pre-processing, to extract plankton's shape and setae information."""

    import skimage
    print("### TEST GLOBAL FEATURES ###")
    x = io.loadmat('dataset/Datas_44.mat')['DATA'][0][0][0]

    for i in range(5):
        # RESIZE: 227x227
        import skimage.transform
        x[i] = skimage.transform.resize(x[i], (227, 227), anti_aliasing = True)

        # PRE-PROCESSING: denoising, sobel, histogram equalization, intensity rescaling
        import skimage.restoration, skimage.filters, skimage.exposure
        x[i] = skimage.restoration.denoise_bilateral(x[i], sigma_color = 0.05, sigma_spatial = 2, channel_axis = -1)
        x[i] = skimage.filters.sobel(x[i])
        x[i] = skimage.exposure.equalize_hist(x[i])
        p2, p98 = np.percentile(x[i], (2, 98))
        x[i] = skimage.exposure.rescale_intensity(x[i], in_range = (p2, p98))

        import skimage.io
        skimage.io.imshow(x[i])
        skimage.io.show()

    return

def test_local_features():
    """Image pre-processing, to extract plankton's texture."""

    import skimage
    print("### TEST LOCAL FEATURES ###")
    x = io.loadmat('dataset/Datas_44.mat')['DATA'][0][0][0]

    for i in range(5):
        # resize to 227x227
        import skimage.transform
        x[i] = skimage.transform.resize(x[i], (227, 227), anti_aliasing = True)

        # grayscale -> canny edge detection -> rgb
        import skimage.color, skimage.filters, skimage.feature
        x[i] = skimage.color.rgb2gray(x[i])
        thresh = skimage.filters.threshold_otsu(x[i])
        x[i] = x[i] * (x[i] < thresh)
        x[i] = skimage.feature.canny(x[i], low_threshold = 0, high_threshold = 0)
        x[i] = x[i].astype(np.float64)
        x[i] = np.repeat(x[i][:, :, np.newaxis], 3, axis = 2)

        import skimage.io
        skimage.io.imshow(x[i])
        skimage.io.show()
        continue

    return

def test_gabor_features():

    import skimage
    print("### TEST LOCAL FEATURES ###")
    x = io.loadmat('dataset/Datas_44.mat')['DATA'][0][0][0]
    for i in range(5):
        import skimage.transform
        x[i] = skimage.transform.resize(x[i], (227, 227), anti_aliasing = True)

        import skimage.color, skimage.filters
        x[i] = skimage.color.rgb2gray(x[i])
        real, img = skimage.filters.gabor(x[i], frequency = 0.6)

        import skimage.io
        skimage.io.imshow(real)
        skimage.io.show()
        skimage.io.imshow(img)
        skimage.io.show()
        continue
    
    return


def main():
    # CLASSES
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
        
    class PlanktonGlobalFeatures(object):
        """Custom plankton image pre-processing transform, using scikit-image, to extract plankton's shape and setae information.
        
        Args:
        """

        def __init__(self, size):
            self.size = size

        def __call__(self, image):
            image = skimage.transform.resize(image, self.size, anti_aliasing = True)   # resize to size (227x227)

            image = skimage.restoration.denoise_bilateral(image, sigma_color = 0.05, sigma_spatial = 2, channel_axis = -1)   # denoising
            image = skimage.filters.sobel(image)                                                                             # sobel
            image = skimage.exposure.equalize_hist(image)                                                                    # histogram equalization
            p2, p98 = np.percentile(image, (2, 98))                                                                          # intensity rescaling
            image = skimage.exposure.rescale_intensity(image, in_range = (p2, p98))                                          #
            
            return image
        
    class PlanktonLocalFeatures(object):
        """Custom plankton image pre-processing transform, using scikit-image, to extract plankton's texture.
        
        Args:
        """

        def __init__(self, size):
            self.size = size

        def __call__(self, image):
            image = skimage.transform.resize(image, self.size, anti_aliasing = True)   # resize to size (227x227)

            image = skimage.color.rgb2gray(image)                                         # grayscale
            thresh = skimage.filters.threshold_otsu(image)                                # thresholding
            image = image * (image < thresh)                                              #
            image = skimage.feature.canny(image, low_threshold = 0, high_threshold = 0)   # canny edge detection
            image = image.astype(np.float64)                                              # convert to float64
            image = np.repeat(image[:, :, np.newaxis], 3, axis = 2)                       # convert to rgb

            return image
        
    # DATASET
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
    if not os.path.exists('output'):
        os.makedirs('output')
    torch.save(model.state_dict(), 'output/model.pth')
    print('MODEL SAVED')

    # TODO: statistics
    print('\nComputing statistics...')
    print('STATISTICS DONE')

    return

if __name__ == '__main__':
    match DEBUG:
        case 0:
            main()
        case 1:
            test_global_features()
        case 2:
            test_local_features()
        case 3:
            test_gabor_features()
        