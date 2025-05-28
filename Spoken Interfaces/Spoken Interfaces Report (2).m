% SVM_Classifier_Model
clear all; close all; clc;

% Define the data folder
dataFolder = "C:\Education matters\PSB related\PSB Subjects\PSB Term 4\6FTC2057 Visual and Spoken Interfaces\Jawad_530BIYKP_Spoken_Interfaces\timit\";

% Define the specific folders you want to include
specificFolders = { ...
    'dr1-mcpm0', ... % Folder for Speaker 1
    'dr4-maeb0', ... % Folder for Speaker 2
    'dr5-mbgt0', ... % Folder for Speaker 3
    'dr7-madd0', ... % Folder for Speaker 4
    'dr8-mbcg0' ...  % Folder for Speaker 5
};

% Create an empty cell array to hold the file paths
fileList = {};

% Loop through each specific folder and get the .wav files
for i = 1:length(specificFolders)
    folderPath = fullfile(dataFolder, specificFolders{i});
    wavFiles = dir(fullfile(folderPath, '*.wav')); % Get all .wav files in the folder
    fileList = [fileList; fullfile(folderPath, {wavFiles.name})']; % Append to the fileList
end

% Create the audio datastore using the custom file list
ads = audioDatastore(fileList, 'LabelSource', 'foldernames');

% Split the datastore into training and testing sets
[adsTrain, adsTest] = splitEachLabel(ads, 0.6);  % split files - 60% train 40% test

% Display the datastore and the number of speakers in the test datastore
trainDatastoreCount = countEachLabel(adsTrain);
testDatastoreCount = countEachLabel(adsTest);

[sampleTrain, dsInfo] = read(adsTrain); % provide a sample 
sound(sampleTrain, dsInfo.SampleRate);

reset(adsTrain);

%% Feature Extraction
fs = dsInfo.SampleRate;
windowLength = round(0.03 * fs);
overlapLength = round(0.025 * fs);
afe = audioFeatureExtractor(SampleRate=fs, ...
    Window=hamming(windowLength, "periodic"), OverlapLength=overlapLength, ...
    zerocrossrate=true, shortTimeEnergy=true, pitch=true, mfcc=true);

featureMap = info(afe);

features = [];
labels = [];
energyThreshold = 0.005;
zcrThreshold = 0.2;

allFeatures = extract(afe, adsTrain);
allLabels = adsTrain.Labels;

for ii = 1:numel(allFeatures)
    thisFeature = allFeatures{ii};

    isSpeech = thisFeature(:, featureMap.shortTimeEnergy) > energyThreshold;
    isVoiced = thisFeature(:, featureMap.zerocrossrate) < zcrThreshold;

    voicedSpeech = isSpeech & isVoiced;

    thisFeature(~voicedSpeech, :) = [];
    thisFeature(:, [featureMap.zerocrossrate, featureMap.shortTimeEnergy]) = [];
    
    label = repelem(allLabels(ii), size(thisFeature, 1));
    label = label(:);

    features = [features; thisFeature];
    labels = [labels; label];
end

%% Normalize features
M = mean(features, 1);
S = std(features, [], 1);
features = (features - M) ./ S;

%% Feature Selection
[idx, scores] = fscmrmr(features, labels);

% Create a bar graph of feature importance
figure;
bar(scores(idx));
xlabel('Features');
ylabel('Importance Score');
title('Feature Importance (SVM)');

% Optional: Add feature names on the x-axis if they exist
xticks(1:length(idx));
xticklabels(idx); % replace featureNames with your actual feature names
xtickangle(45);

%% Train SVM Model with RBF Kernel
% Using a non-linear kernel for potentially better separation
template = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto');
SVMModel = fitcecoc(features, labels, 'Learners', template);

%% Extract features from the test set
features = [];
labels = [];
allFeatures = extract(afe, adsTest);
allLabels = adsTest.Labels;

for ii = 1:numel(allFeatures)
    thisFeature = allFeatures{ii};

    isSpeech = thisFeature(:, featureMap.shortTimeEnergy) > energyThreshold;
    isVoiced = thisFeature(:, featureMap.zerocrossrate) < zcrThreshold;

    voicedSpeech = isSpeech & isVoiced;

    thisFeature(~voicedSpeech, :) = [];
    thisFeature(:, [featureMap.zerocrossrate, featureMap.shortTimeEnergy]) = [];
    
    label = repelem(allLabels(ii), size(thisFeature, 1));
    label = label(:);

    features = [features; thisFeature];
    labels = [labels; label];
end

% Normalize test features using the mean and std from training set
features = (features - M) ./ S;

%% Make Predictions
[Y_pred, scores] = predict(SVMModel, features);

%% Confusion Matrices
% Frame-level confusion chart
figure;
confusionchart(labels, Y_pred, 'Title', 'Frame-Level Accuracy (SVM)', ...
     'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');

disp('Calculating frame-level accuracy...');
frame_accuracy = sum(labels == Y_pred) / length(labels);
disp(['Frame-Level Accuracy: ', num2str(frame_accuracy * 100), '%']);

%% Per File Accuracy
numVectorsPerFile = countEachLabel(adsTest).Count';
file_start = 1;
Y_test_file = [];
Y_pred_file = [];

for i = 1:length(numVectorsPerFile)
    numFrames = numVectorsPerFile(i);
    if file_start + numFrames - 1 <= length(Y_pred) % Ensure we don't go out of bounds
        file_frames = Y_pred(file_start:file_start + numFrames - 1);
        true_label = labels(file_start);
        
        % Use the mode or majority vote across frames for each file
        if ~isempty(file_frames) % Check if there are any predictions
            Y_pred_file = [Y_pred_file; mode(file_frames)];
        else
            Y_pred_file = [Y_pred_file; NaN]; % Or some default label
        end
        
        Y_test_file = [Y_test_file; true_label];
    end

    file_start = file_start + numFrames;
end

% File-level confusion chart
figure;
confusionchart(Y_test_file, Y_pred_file, 'Title', 'File-Level Accuracy (SVM)', ...
    'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');

file_accuracy = sum(Y_test_file == Y_pred_file) / length(Y_test_file);
disp(['File-Level Accuracy: ', num2str(file_accuracy * 100), '%']);

%% Histogram of Predicted Labels
figure;
histogram(Y_pred);
xlabel('Predicted Labels');
ylabel('Frequency');
title('Histogram of Predicted Labels (SVM)');

%% Validation Accuracy
Y_train_pred = predict(SVMModel, features);
figure;
confusionchart(labels, Y_train_pred, 'Title', 'Validation Accuracy (SVM)', ...
    'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
validation_accuracy = sum(labels == Y_train_pred) / length(labels);
disp(['Validation Accuracy: ', num2str(validation_accuracy * 100), '%']);