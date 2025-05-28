%% - Training and Save Section
% Clear the workspace, close all figures, and suppress warnings
clc                     
clear all               
close all               
warning off             

% Display message indicating the expected training duration
disp("This training can take up to 8 minutes...")

% Define the learning rate for training the network
learning_rate = 0.000008;

% Load the pre-trained AlexNet model
g = alexnet;            
layers = g.Layers;      

% Modify the AlexNet architecture to suit the gesture classification task
layers(23) = fullyConnectedLayer(5);     % Replace the fully connected layer for 5 gesture classes
layers(25) = classificationLayer;       % Replace the final classification layer

% Load and preprocess the gesture dataset
allImages = imageDatastore('gesturing\', ...  % Load images from the 'gesturing' folder
    'IncludeSubfolders', true, ...           % Include images from subfolders
    'LabelSource', 'foldernames');           % Use folder names as labels
allImages.ReadFcn = @(loc) imresize(imread(loc), [227, 227]); % Resize images to 227x227 (AlexNet input size)

% Define training options for the network
opts = trainingOptions("rmsprop", ...        % Use RMSprop optimizer
    "InitialLearnRate", learning_rate, ...   % Set the initial learning rate
    'MaxEpochs', 200, ...                    % Train for 200 epochs
    'MiniBatchSize', 64);                    % Use a mini-batch size of 64

% Train the network using the specified layers and options
FingerGestureModel = trainNetwork(allImages, layers, opts);

% Save the trained model to a file
save FingerGestureModel;

% Display message indicating that training is complete
disp("Training Completed")
