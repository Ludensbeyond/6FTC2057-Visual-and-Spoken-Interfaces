% GMM_Classifier_Model
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
%% 

featureMap = info(afe); % with afe, one can seek mfcc
%% 
% Extract features from the training set.

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

    features = [features; thisFeature];  % adding up the features
    labels = [labels, label];  % adding up the labels
end

%% Normalize features
M = mean(features, 1);
S = std(features, [], 1);
features = (features - M) ./ S;

%% - GMM
uniqueLabels = unique(labels);  % Get the unique labels
numLabels = length(uniqueLabels);  % Number of unique classes
gmModel = cell(numLabels, 1);  % Initialize a cell array for GMMs

% Fit GMM for each label
for i = 1:numLabels
    label = uniqueLabels(i);  % Current label
    X_label = features(labels == label', :);  % Features for the current label
    numComponents = 2;  % Number of Gaussian components
    gmModel{i} = fitgmdist(X_label, numComponents);  % Fit GMM
end


%% Extract features from the test set
% Extract features from the data set.

features = [];
labels = [];
energyThreshold = 0.005;
zcrThreshold = 0.2;

allFeatures = extract(afe,adsTrain);
allLabels = adsTrain.Labels;

for ii = 1:numel(allFeatures)

    thisFeature = allFeatures{ii};

    isSpeech = thisFeature(:,featureMap.shortTimeEnergy) > energyThreshold;
    isVoiced = thisFeature(:,featureMap.zerocrossrate) < zcrThreshold;

    voicedSpeech = isSpeech & isVoiced;

    thisFeature(~voicedSpeech,:) = [];
    thisFeature(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];
    label = repelem(allLabels(ii),size(thisFeature,1));
    
    features = [features;thisFeature];  % adding up the features
    labels = [labels,label];  % adding up the labels
end
%% 
% Pitch and MFCC are not on the same scale. This will bias the classifier. Normalize 
% the features by subtracting the mean and dividing the standard deviation.

M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;
%%
[idx,scores] = fscmrmr(features,labels);

% Create a bar graph
figure;
bar(scores(idx));
xlabel('Features');
ylabel('Importance Score');
title('Feature Importance (GMM)');

% Optional: Add feature names on the x-axis if they exist
xticks(1:length(idx));
xticklabels(idx); % replace featureNames with your actual feature names
xtickangle(45); % Angle the x-axis labels if they are overlapping
%% - GMM
% X is your data matrix of size [n, d], where n is the number of data points and d is the number of features
% Y is your label vector of size [n, 1], where each element corresponds to the class label of the feature vector in X

% Get the unique labels (classes) in your data
uniqueLabels = unique(labels);  % e.g., if Y has 16 unique labels, this will contain those labels
numLabels = length(uniqueLabels);  % Number of unique classes (e.g., 16)
% Initialize a cell array to store GMMs for each label
gmModel = cell(numLabels, 1);
% Loop over each unique label (class)
for i = 1:numLabels
    label = uniqueLabels(i);  % Get the current label (class)
    % Extract the feature vectors corresponding to the current label
    X_label = features(labels == label', :);  % Data points that belong to the current label
    % Choose the number of Gaussian components for the GMM (e.g., 2 for each class)
    numComponents = 2;
    % Fit the GMM to the data points for this label
    gmModel{i} = fitgmdist(X_label, numComponents);
end

%%
features = [];
labels = [];
numVectorsPerFile = [];
allFeatures = extract(afe,adsTest);
allLabels = adsTest.Labels;
for ii = 1:numel(allFeatures)
    thisFeature = allFeatures{ii};
    isSpeech = thisFeature(:,featureMap.shortTimeEnergy) > energyThreshold;
    isVoiced = thisFeature(:,featureMap.zerocrossrate) < zcrThreshold;
    voicedSpeech = isSpeech & isVoiced;
    thisFeature(~voicedSpeech,:) = [];
    numVec = size(thisFeature,1);
    thisFeature(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];
    label = repelem(allLabels(ii),numVec);
    numVectorsPerFile = [numVectorsPerFile,numVec];
    features = [features;thisFeature];
    labels = [labels,label];
end
features = (features-M)./S;
%%
% Initialize an array to store predicted labels
Y_pred = zeros(size(features, 1), 1);  % Preallocate array for predicted labels
% Loop over the test data points
for j = 1:size(features, 1)
    x_test = features(j, :);  % Extract one test data point (feature vector)
    % Initialize an array to store likelihoods for each label
    likelihood = zeros(numLabels, 1);
    % Compute likelihood under each GMM
    for i = 1:numLabels
        likelihood(i) = pdf(gmModel{i}, x_test);  % Compute likelihood under GMM for label i
    end   
    % Assign the label with the highest likelihood
    [~, predictedLabelIdx] = max(likelihood);
    Y_pred(j) = uniqueLabels(predictedLabelIdx);  % Predicted label for the current test data point
end
%%
% Calculate the confusion matrix
% Y_test = uint8(labels);
% Y_pred = uint8(Y_pred');
% 
% confusionMatrix = confusionmat(Y_test, Y_pred);
% confusionchart(Y_test, Y_pred);  % Visualize the confusion matrix
% % Calculate accuracy
% accuracy = sum(Y_test == Y_pred) / length(Y_test);
% disp(['Accuracy: ', num2str(accuracy * 100), '%']);

%% Per File Accuracy
% Calculate the per-file prediction by taking the mode of predictions for each file.
file_start = 1;
Y_test_file = [];
Y_pred_file = [];

for i = 1:length(numVectorsPerFile)
    % Get the frame indices for the current file
    numFrames = numVectorsPerFile(i);
    file_frames = Y_pred(file_start:file_start + numFrames - 1);
    true_label = labels(file_start);  % True label for the entire file
    
    % Store the mode of the predictions as the file-level prediction
    Y_pred_file = [Y_pred_file; mode(file_frames)];
    Y_test_file = [Y_test_file; true_label];
    
    % Update the starting frame index for the next file
    file_start = file_start + numFrames;
end


%% - Manual Testing with specific wav file
testfeatures = [];
predictedLabel = [];

% Define only 5 speakers
labelArray = [1, 2, 3, 4, 5];  % Adjust this if necessary for your specific speakers
stringArray = [...
    "dr1-mcpm0/sa1", ... % Speaker 1
    "dr4-maeb0/sa1", ... % Speaker 2
    "dr5-mbgt0/sa1", ... % Speaker 3
    "dr7-madd0/sa1", ... % Speaker 4
    "dr8-mbcg0/sa1" ...  % Speaker 5
];

% Initialize arrays to store results for all speakers
Y_true = [];
Y_predicted = [];

% Loop through all speakers for playback and prediction
for select_Wav = 1:length(stringArray)
    str1 = stringArray(select_Wav);
    str2 = '.wav';
    str123 = strcat(dataFolder, str1, str2);

    % Check if the file exists before attempting to read
    if exist(str123, 'file')
        [audioIn, fs] = audioread(str123);
        sound(audioIn, fs);

        pause(length(audioIn) / fs + 1);

        % Extract features from the test audio
        myfeatures = extract(afe, audioIn);
        testfeatures = [];

        % Process features for speech detection and normalization
        for ii = 1:size(myfeatures, 1)
            thisFeature = myfeatures(ii, :);
            isSpeech = thisFeature(:, featureMap.shortTimeEnergy) > energyThreshold;
            isVoiced = thisFeature(:, featureMap.zerocrossrate) < zcrThreshold;
            voicedSpeech = isSpeech & isVoiced;
            thisFeature(~voicedSpeech, :) = [];
            thisFeature(:, [featureMap.zerocrossrate, featureMap.shortTimeEnergy]) = [];
            testfeatures = [testfeatures; thisFeature];
        end

        % Normalize the features
        testfeatures = (testfeatures - M) ./ S;

        % Predict label for each feature frame
        predictedLabel = zeros(size(testfeatures, 1), 1);
        for j = 1:size(testfeatures, 1)
            likelihood = zeros(numLabels, 1);
            for i = 1:numLabels
                likelihood(i) = pdf(gmModel{i}, testfeatures(j, :));
            end
            [~, predictedLabelIdx] = max(likelihood);
            predictedLabel(j) = uniqueLabels(predictedLabelIdx);
        end

        % Append true and predicted labels for this speaker
        Y_true = [Y_true; uint8(labelArray(select_Wav) * ones(size(predictedLabel)))];
        Y_predicted = [Y_predicted; uint8(predictedLabel)];
    else
        error('Audio file does not exist: %s', str123);
    end
end



myfeatures = extract(afe, audioIn);

% Start processing features
for ii = 1:size(myfeatures, 1)
    thisFeature = myfeatures(ii, :);
    isSpeech = thisFeature(:, featureMap.shortTimeEnergy) > energyThreshold;
    isVoiced = thisFeature(:, featureMap.zerocrossrate) < zcrThreshold;
    voicedSpeech = isSpeech & isVoiced;
    thisFeature(~voicedSpeech, :) = []; % adding empty
    thisFeature(:, [featureMap.zerocrossrate, featureMap.shortTimeEnergy]) = [];
    testfeatures = [testfeatures; thisFeature]; % injecting 3
end

testfeatures = (testfeatures - M) ./ S; % Normalize features

% Now gmModel contains a GMM for each label/class
% You can use these models for classification of new data points

% Initialize an array to store the likelihoods for each label
likelihood = zeros(numLabels, 1);

% Loop over each test feature
for j = 1:size(testfeatures, 1)
    for i = 1:numLabels
        likelihood(i) = pdf(gmModel{i}, testfeatures(j, :));  % Compute the likelihood under the GMM for label i
    end
    % Assign the label with the highest likelihood
    [~, predictedLabelIdx] = max(likelihood);
    predictedLabel(j) = uniqueLabels(predictedLabelIdx);  % Predicted label for the new feature vector
end

% Display the predicted label for the selected wav
fprintf("By Mode, for Selected Wav Index, %d, the predicted label is %d\n", select_Wav, mode(predictedLabel));

% Create a histogram of the predicted labels
figure;
histogram(predictedLabel);
xlabel('Predicted Labels');
ylabel('Frequency');
title('Histogram of Predicted Labels (GMM)');

% Y_true = uint8(labelArray(select_Wav) * ones(1, length(testfeatures)));
% Y_predicted = uint8(predictedLabel);

figure;
confusionchart(Y_true, Y_predicted, title="Validation Accuracy (GMM)", ...
    ColumnSummary="column-normalized", RowSummary="row-normalized");
validation_accuracy = sum(Y_true == Y_predicted) / length(Y_true);
disp(['Validation Accuracy: ', num2str(validation_accuracy * 100), '%']);



% Frame-level confusion chart
Y_test = uint8(labels);
Y_pred = uint8(Y_pred');

figure;
confusionchart(Y_test, Y_pred, 'Title', 'Per Frame Accuracy (GMM)', ...
     'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
frame_accuracy = sum(Y_test == Y_pred) / length(Y_test);
disp(['Frame-Level Accuracy: ', num2str(frame_accuracy * 100), '%']);
  
Y_test_file = uint8(Y_test_file);
Y_pred_file = uint8(Y_pred_file);

% File-level confusion chart
figure;
confusionchart(Y_test_file, Y_pred_file, 'Title', 'Per File Accuracy (GMM)', ...
    'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
file_accuracy = sum(Y_test_file == Y_pred_file) / length(Y_test_file);
disp(['File-Level Accuracy: ', num2str(file_accuracy * 100), '%'])
return; 
