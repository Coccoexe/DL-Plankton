# Alessio Cocco 2087635, Andrea Valentinuzzi 2090451, Giovanni Brejc 2096046
# Plankton Pre-Processing and Classification with AlexNet
# UniPD 2022/23 - Deep Learning Project

# based
import os
from datetime import datetime
from scipy import io
import numpy as np

# scikit-image
import skimage
from skimage import data

# torch
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

def main():
    # DEVICE
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.set_default_device(DEVICE)
    print(f"Using {DEVICE} device")

    # CUSTOM DATASET
    class PlanktonDataset(Dataset):
        def __init__(self, labels, patterns, ytransform = None, xtransform = None):
            self.labels = labels
            self.patterns = patterns

        def __len__(self):
            return len(self.patterns)

        def __getitem__(self, idx):
            return self.patterns[idx], self.labels[idx]

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
    model = models.alexnet(pretrained = True)
    print('LOADING DONE')
    input_size = np.array([227, 227])

    # PARAMETERS
    batch_size = 32
    lr = 1e-4
    epochs = 30
    momentum = 0.9
    iterations = div // batch_size

    # MAIN LOOP
    for fold in range(folds):
        print(f'\n### Fold {fold+1}/{folds} ###')
        # dataset
        train_pattern , train_label, test_pattern, test_label = [], [], [], []
        for i in shuff[fold][:div]-1:
            train_pattern.append(x[i])   # 0:div pre-shuffled pattern from fold
            train_label.append(t[i])     # 0:div pre-shuffled label from fold
        for i in shuff[fold][div:]-1:
            test_pattern.append(x[i])    # div:tot pre-shuffled pattern from fold
            test_label.append(t[i])      # div:tot pre-shuffled label from fold
        num_classes = max(train_label)   # number of classes
        print(f'Training patterns: {len(train_pattern)}')
        print(f'Testing patterns: {len(test_pattern)}')
        print(f'Number of classes: {num_classes}')

        # TODO: preprocessing
        print('\nPreprocessing...')
        for i in range(div):
            # labels - 1 to start from 0
            train_label[i] -= 1
            # resize to input_size
            train_pattern[i] = skimage.transform.resize(train_pattern[i], input_size)
        for i in range(tot-div):
            # labels - 1 to start from 0
            test_label[i] -= 1
            # resize to input_size
            test_pattern[i] = skimage.transform.resize(test_pattern[i], input_size)
        print('PREPROCESSING DONE')

        # (height, width, channels) -> (channels, height, width)
        for i in range(div):
            train_pattern[i] = np.transpose(train_pattern[i], (2, 0, 1))
        for i in range(tot-div):
            test_pattern[i] = np.transpose(test_pattern[i], (2, 0, 1))

        # tensor
        train_pattern = torch.tensor(np.array(train_pattern), dtype = torch.float32)
        train_label = torch.tensor(np.array(train_label), dtype = torch.long)
        test_pattern = torch.tensor(np.array(test_pattern), dtype = torch.float32)
        test_label = torch.tensor(np.array(test_label), dtype = torch.long)

        # custom dataset
        train_data = PlanktonDataset(labels = train_label, patterns = train_pattern)
        test_data = PlanktonDataset(labels = test_label, patterns = test_pattern)

        # dataloader
        train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = False)
        test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

        # fine-tuning
        layers = list(model.classifier.children())[:-3]
        layers.extend([torch.nn.Linear(4096, 4096), torch.nn.Linear(4096, num_classes), torch.nn.Softmax(dim = 1)])
        model.classifier = torch.nn.Sequential(*layers)
        model = model.to(DEVICE)

        # loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)

        # TODO: training
        print('\nTraining...')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        model.train()
        for epoch in range(epochs):
            avg_loss = 0.
            for i, data in enumerate(train_dataloader):
                # data
                inputs, labels = data

                # training
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                # logging
                avg_loss += loss.item()
                if i % 100 == 99:
                    print(f"Batch {i} - Loss: {avg_loss / 100}")
                    writer.add_scalar("Loss/train", avg_loss / 100, epoch * len(train_dataloader) + i + 1)
                    avg_loss = 0.
        print('TRAINING DONE')

        # TODO: testing
        print('\nTesting...')
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_dataloader:
                # data
                inputs, labels = data

                # testing
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # logging
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Accuracy: {100 * correct / total}")
            writer.add_scalar("Accuracy/test", 100 * correct / total, epoch * len(train_dataloader) + i + 1)
        print('TESTING DONE')

    # save model
    print('\nSaving model...')
    torch.save(model.state_dict(), 'output/model.pth')
    print('MODEL SAVED')

    # TODO: statistics


    return

#                      __gggrgM**M#mggg__
#                __wgNN@"B*P""mp""@d#"@N#Nw__
#              _g#@0F_a*F#  _*F9m_ ,F9*__9NG#g_
#           _mN#F  aM"    #p"    !q@    9NL "9#Qu_
#          g#MF _pP"L  _g@"9L_  _g""#__  g"9w_ 0N#p
#        _0F jL*"   7_wF     #_gF     9gjF   "bJ  9h_
#       j#  gAF    _@NL     _g@#_      J@u_    2#_  #_
#      ,FF_#" 9_ _#"  "b_  g@   "hg  _#"  !q_ jF "*_09_
#      F N"    #p"      Ng@       `#g"      "w@    "# t
#     j p#    g"9_     g@"9_      gP"#_     gF"q    Pb L
#     0J  k _@   9g_ j#"   "b_  j#"   "b_ _d"   q_ g  ##
#     #F  `NF     "#g"       "Md"       5N#      9W"  j#
#     #k  jFb_    g@"q_     _*"9m_     _*"R_    _#Np  J#
#     tApjF  9g  J"   9M_ _m"    9%_ _*"   "#  gF  9_jNF
#      k`N    "q#       9g@        #gF       ##"    #"j
#      `_0q_   #"q_    _&"9p_    _g"`L_    _*"#   jAF,'
#       9# "b_j   "b_ g"    *g _gF    9_ g#"  "L_*"qNF
#        "b_ "#_    "NL      _B#      _I@     j#" _#"
#          NM_0"*g_ j""9u_  gP  q_  _w@ ]_ _g*"F_g@
#           "NNh_ !w#_   9#g"    "m*"   _#*" _dN@"
#              9##g_0@q__ #"4_  j*"k __*NF_g#@P"
#                "9NN#gIPNL_ "b@" _2M"Lg#N@F"
#                    ""P@*NN#gEZgNN@#@P""

if __name__ == '__main__':
    main()