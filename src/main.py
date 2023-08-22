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

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")

# TODO: training function
def train_one_epoch(epoch_index, tb_writer, model, training_loader, loss_fn, optimizer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100
            running_loss = 0.
            print(f"Batch {i} - Loss: {last_loss}")
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.
        
    return last_loss

def main():
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

    # CLASSES
    class PlanktonDataset(Dataset):
        def __init__(self, labels, patterns, transform = None, target_transform = None):
            self.labels = labels
            self.patterns = patterns

        def __len__(self):
            return len(self.patterns)

        def __getitem__(self, idx):
            return self.patterns[idx], self.labels[idx]
    
    # ALEXNET
    model = models.alexnet(pretrained = True).to(DEVICE)
    input_size = np.array([227, 227])

    # PARAMETERS
    batch_size = 32
    lr = 1e-4
    epochs = 30
    momentum = 0.9
    iterations = div // batch_size

    # MAIN LOOP
    for fold in range(folds):
        # dataset
        train_pattern , train_label, test_pattern, test_label = [], [], [], []
        for i in shuff[fold][:div]-1:
            train_pattern.append(x[i])   # 0:div pre-shuffled pattern from fold
            train_label.append(t[i])     # 0:div pre-shuffled label from fold
        for i in shuff[fold][div:]-1:
            test_pattern.append(x[i])    # div:tot pre-shuffled pattern from fold
            test_label.append(t[i])      # div:tot pre-shuffled label from fold
        num_classes = max(train_label)   # number of classes

        # TODO: preprocessing only on trainig set
        for i in range(div):
            train_pattern[i] = skimage.transform.resize(train_pattern[i], input_size)   # resize to input_size

        # custom dataset
        train_data = PlanktonDataset(labels=train_label, patterns=train_pattern)
        test_data = PlanktonDataset(labels=test_label, patterns=test_pattern)

        # dataloader
        train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = False)
        test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

        # fine-tuning
        layers = list(model.classifier.children())[:-3]
        layers.extend([torch.nn.Linear(4096, 4096), torch.nn.Linear(4096, num_classes), torch.nn.Softmax(dim = 1)])
        model.classifier = torch.nn.Sequential(*layers)

        # loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)

        # TODO: training
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0
        for epoch in range(epochs):
            model.train(True)
            avg_loss = train_one_epoch(epoch, writer, model, train_dataloader, loss_fn, optimizer)
            writer.add_scalar("Loss/train", avg_loss, epoch_number + 1)
            writer.flush()
            epoch_number += 1



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