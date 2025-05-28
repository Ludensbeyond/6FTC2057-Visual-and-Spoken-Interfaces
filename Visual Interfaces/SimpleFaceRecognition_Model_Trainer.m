% Create a datastore for the original training images
im_original = imageDatastore('Faces\train\', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Ensure the 'Faces\cropped\' folder exists
croppedFolder = 'Faces\cropped\';
if ~exist(croppedFolder, 'dir')
    mkdir(croppedFolder);
end

% Dynamically determine the unique labels in the dataset
uniqueLabels = unique(im_original.Labels);
people = cellstr(uniqueLabels); % Ensure it matches the dataset labels
n = numel(people); % Number of unique labels

% Loop through each label, crop faces, and save them in respective folders
for i = 1:n
    str = people{i}; % Use the correct label dynamically
    ds1 = imageDatastore(fullfile('Faces\train\', str), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    dirName = fullfile(croppedFolder, str); % Specify the path for each label's folder
    if ~exist(dirName, 'dir')
        mkdir(dirName);
    end
    cropandsave(ds1, char(str)); % Pass char type for str
end

% Create a datastore for the cropped faces
im = imageDatastore(croppedFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Resize the images to the input size of AlexNet
im.ReadFcn = @(loc) imresize(imread(loc), [227, 227]);

% Split the dataset into training and testing sets
[Train, Test] = splitEachLabel(im, 0.8, 'randomized');

% Dynamically determine the number of unique labels in the training set
uniqueLabels = unique(Train.Labels);
n = numel(uniqueLabels); % Number of unique labels

% Load the AlexNet model
net = alexnet;
ly = net.Layers;

% Replace the fully connected and classification layers
fc = fullyConnectedLayer(n, 'Name', 'fc'); % Output size matches 'n'
ly(23) = fc;

cl = classificationLayer('Name', 'classification');
ly(25) = cl;

% Set training options
learning_rate = 0.000008; % Original value was 0.00001
opts = trainingOptions('rmsprop', ...
    'InitialLearnRate', learning_rate, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 64, ...
    'Plots', 'training-progress');

% Train the network
[faceRecognitionModel, info] = trainNetwork(Train, ly, opts);

% Evaluate the model
[predict, scores] = classify(faceRecognitionModel, Test);
names = Test.Labels;
pred = (predict == names);
s = size(pred);
acc = sum(pred) / s(1);
fprintf('The accuracy of the test set is %f %% \n', acc * 100);

% Save the trained model
save faceRecognitionModel;

%% Prediction Section - Test a New Image
% Use the code below with a path to new image
img = imread('Faces\cropped\Jawad_Expressionless\1.jpg');
[img, face] = cropface(img);
figure;
imshow(img);

% Check if a face is detected
if face == 1
    img = imresize(img, [227, 227]);
    predict = classify(faceRecognitionModel, img);
    
    % Dynamically match prediction to the people array
    for i = 1:n
        if predict == uniqueLabels(i)
            fprintf('The face detected is %s\n', char(uniqueLabels(i)));
            break;
        end
    end
else
    disp("No Faces detected");
end
