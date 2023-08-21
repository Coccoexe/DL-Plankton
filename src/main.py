# Alessio Cocco 2087635, Andrea Valentinuzzi 2090451, Giovanni Brejc 2096046
# Plankton Pre-Processing and Classification with AlexNet
# UniPD 2022/23 - Deep Learning Project

# based
import os
from scipy import io
import numpy as np

# scikit-image
import skimage
from skimage import data

# torch
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# matplotlib
import matplotlib.pyplot as plt

def image_preprocessing():
    return

def training():
    return

def main():
    # DATASET
    datapath = 'dataset/Datas_44.mat'
    if not os.path.exists(datapath):
        raise Exception('ERROR: Dataset not found. Check the path.')
    mat = np.array(io.loadmat(datapath)['DATA'])[0]

    x = mat[0]                # patterns
    t = mat[1]                # labels
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
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained = True)
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
        train = [(x[i], t[i]) for i in shuff[fold][:div]]   # 0:div pre-shuffled (pattern, label) from fold
        test = [(x[i], t[i]) for i in shuff[fold][div:]]    # div:tot pre-shuffled (pattern, label) from fold
        num_classes = max(train[1])                         # number of classes

        # TODO: preprocessing only on trainig set
        for i in range(len(train[0])):
            train[0][i] = skimage.transform.resize(train[0][i], input_size)   # resize to input_size
            train[0][i] = ToTensor()(train[0][i])                             # ???: to tensor

        # custom dataset
        train_data = PlanktonDataset(train[1], train[0])
        test_data = PlanktonDataset(test[1], test[0])

        # dataloader
        train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = False)
        test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

        # tuning: remove last 3 layers then add FC layer, softmax, classificator
        model = torch.nn.Sequential(*list(model.children())[:-3])
        model.add_module('fc', torch.nn.Linear(256, num_classes))
        model.add_module('softmax', torch.nn.Softmax(dim = 1))
        model.add_module('classificator', torch.nn.Linear(num_classes, num_classes))

        # loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)

        # training
        for epoch in range(epochs):
            running_loss = 0.0
            last_loss = 0.0

            for i, (inputs, labels) in enumerate(train_dataloader):
                optimizer.zero_grad()             # zero the parameter gradients
                outputs = model(inputs)           # forward pass
                loss = loss_fn(outputs, labels)   # compute loss
                loss.backward()                   # backward pass
                optimizer.step()                  # update weights

                # statistics
                running_loss += loss.item()
                if i % iterations == iterations - 1:
                    last_loss = running_loss / iterations
                    print(f'Batch {i + 1} - Loss: {last_loss}')
                    running_loss = 0.0

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

#                                            ████████████████                                        
#                                ████████░░░░░░░░░░░░░░░░████████                                
#                            ▓▓██░░░░░░        ░░░░░░      ░░▒▒▒▒▓▓██                            
#                        ████      ░░░░░░░░░░░░░░  ░░░░░░░░  ░░    ░░▓▓▓▓                        
#                    ████░░    ░░░░░░░░    ░░░░░░░░░░░░░░░░░░░░░░░░░░  ░░████                    
#                  ▒▒▒▒░░    ░░░░░░    ░░░░░░░░░░░░░░░░░░░░░░░░░░      ░░░░▒▒▓▓                  
#                ██      ░░░░░░      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ░░██                
#              ██░░    ░░░░░░    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██              
#            ██░░    ░░░░░░    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ░░██            
#          ██░░    ░░░░░░      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ░░██          
#        ██░░    ░░░░░░    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ▒▒██        
#        ██░░  ░░░░░░    ░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒██        
#      ▓▓░░  ░░░░░░    ░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░▒▒  ▒▒▓▓      
#      ██    ░░░░░░    ░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░▒▒▒▒  ▒▒██      
#    ██▒▒  ░░░░░░    ░░░░  ░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒░░▒▒▓▓    
#    ██    ░░░░░░  ░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒  ▒▒██    
#  ██░░  ░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒██  
#  ██░░  ░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ▒▒██  
#  ██    ░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ▒▒██  
#  ██  ░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ▒▒██  
#██░░  ░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ░░██
#██░░  ░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ░░██
#██    ░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░██
#██    ░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░██
#██    ░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░██
#██    ░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░██
#██    ░░░░░░▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▓▓▒▒░░░░██
#  ██  ░░░░▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▓▓▓▓▒▒░░██  
#  ██  ░░░░▒▒▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▓▓▓▓▓▓▒▒░░██  
#  ██  ░░░░▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▓▓▒▒▒▒▓▓▓▓▓▓▒▒▒▒██  
#  ██    ░░▒▒▒▒▒▒▓▓▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒██  
#    ██  ░░▒▒░░▒▒▒▒▓▓▓▓▓▓▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▓▓▒▒▓▓▓▓▓▓▒▒░░██    
#    ██  ░░▒▒░░▒▒▒▒▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▒▒▒▒██    
#      ██  ▒▒░░▓▓▒▒▒▒▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▒▒░░██      
#      ██  ░░░░▓▓▒▒▒▒▓▓▓▓▓▓▓▓▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓░░▒▒██      
#        ██  ░░▒▒▓▓▓▓▓▓▓▓▓▓▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▒▒▒▒██        
#        ██  ░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▓▓▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▒▒▒▒██        
#          ██  ░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓░░▓▓▓▓▒▒▒▒██          
#            ██░░░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░▓▓▓▓▓▓▒▒██            
#              ▓▓░░░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒░░░░▓▓▓▓▓▓▓▓██              
#                ██░░░░▒▒▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░▓▓▓▓▓▓▓▓▒▒██                
#                  ██░░▒▒▒▒▓▓▓▓▓▓▓▓░░░░░░▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░  ▓▓▓▓▓▓▓▓▒▒██                  
#                  ░░████▒▒▒▒▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░      ▒▒▓▓▓▓▒▒████░░                  
#                        ████▓▓▒▒▓▓▓▓░░░░                  ▓▓▒▒▒▒▓▓▒▒████                        
#                            ████▒▒▒▒▓▓▓▓▓▓            ▓▓▓▓▓▓▓▓▒▒████                            
#                                ████████▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓████████                                
#                                    ░░  ████████████████                                        