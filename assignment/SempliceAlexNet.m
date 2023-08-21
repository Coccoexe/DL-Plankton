clear all
warning off

%load dataset
datas=44;
load(strcat('Datas_',int2str(datas)),'DATA');
NF=size(DATA{3},1); %number of folds
DIV=DATA{3};%for the division between training and test set
DIM1=DATA{4};%training patterns number
DIM2=DATA{5};%total patterns number
yE=DATA{2};%label of all the patterns
NX=DATA{1};%images

%load pre-trained AlexNet
net = alexnet;  %load AlexNet
siz=[227 227];

%parameters
miniBatchSize = 30;
learningRate = 1e-4;
metodoOptim='sgdm';
options = trainingOptions(metodoOptim,...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',30,...
    'InitialLearnRate',learningRate,...
    'Verbose',false,...
    'Plots','training-progress');
numIterationsPerEpoch = floor(DIM1/miniBatchSize);


for fold=1:NF%for each fold
    close all force
    
    trainPattern=(DIV(fold,1:DIM1));
    testPattern=(DIV(fold,DIM1+1:DIM2));
    y=yE(DIV(fold,1:DIM1));%training label
    yy=yE(DIV(fold,DIM1+1:DIM2));%test label
    numClasses = max(y);%number of classes
    
    %create training set
    clear nome trainingImages
    for pattern=1:DIM1
        IM=NX{DIV(fold,pattern)};%image
        
        %insert here any pre-processing on the IM image
        
        IM=imresize(IM,[siz(1) siz(2)]);%you have to do image resize to make it compatible with CNN
        if size(IM,3)==1
            IM(:,:,2)=IM;
            IM(:,:,3)=IM(:,:,1);
        end
        trainingImages(:,:,:,pattern)=IM;
    end
    imageSize=size(IM);
    
    %data augmentation
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXScale',[1 2]);
    trainingImages = augmentedImageDatastore(imageSize,trainingImages,categorical(y'),'DataAugmentation',imageAugmenter);
    
    %tuning della rete
    % The last three layers of the pretrained network net are configured for 1000 classes.
    %These three layers must be fine-tuned for the new classification problem. Extract all layers, except the last three, from the pretrained network.
    layersTransfer = net.Layers(1:end-3);
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        softmaxLayer
        classificationLayer];
    netTransfer = trainNetwork(trainingImages,layers,options);
    
    %test set
    clear nome test testImages
    for pattern=ceil(DIM1)+1:ceil(DIM2)
        IM=NX{DIV(fold,pattern)};%image
        
        %insert here any pre-processing on the IM image
        
        IM=imresize(IM,[siz(1) siz(2)]);
        if size(IM,3)==1
            IM(:,:,2)=IM;
            IM(:,:,3)=IM(:,:,1);
        end
        testImages(:,:,:,pattern-ceil(DIM1))=uint8(IM);
    end
    
    %classifying test patterns
    [outclass, score{fold}] =  classify(netTransfer,testImages);
    
    %accuracy
    [a,b]=max(score{fold}');
    ACC(fold)=sum(b==yy)./length(yy);

    %save whatever you need
    %%%%%
    
end


