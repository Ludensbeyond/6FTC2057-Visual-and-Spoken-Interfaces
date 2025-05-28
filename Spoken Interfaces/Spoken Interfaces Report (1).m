%KNN_Classifier_Model
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

% Empty cell array to hold the file paths
fileList = {};

% Loop through each specific folder and get the .wav files
for i = 1:length(specificFolders)
    folderPath = fullfile(dataFolder, specificFolders{i});
    wavFiles = dir(fullfile(folderPath, '*.wav')); % Get all .wav files in the folder
    fileList = [fileList; fullfile(folderPath, {wavFiles.name})']; % Append to the fileList
end

% Audio datastore created using the custom file list
ads = audioDatastore(fileList, 'LabelSource', 'foldernames');

% Split the datastore into training and testing sets
[adsTrain, adsTest] = splitEachLabel(ads, 0.6); % split files - 60% train 40% test

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

featureMap = info(afe) 
%% 


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
title('Feature Importance (KNN)');

% Optional: Add feature names on the x-axis if they exist
xticks(1:length(idx));
xticklabels(idx); % replace featureNames with your actual feature names
xtickangle(45); % Angle the x-axis labels if they are overlapping

%% Train Classifier

trainedClassifier = fitcknn(features,labels, ...
    Distance="euclidean", ...
    NumNeighbors=5, ...
    DistanceWeight="squaredinverse", ...
    Standardize=false, ...
    ClassNames=unique(labels));

k = 5;
group = labels;
c = cvpartition(group,KFold=k); % 5-fold stratified cross validation
partitionedModel = crossval(trainedClassifier,CVPartition=c);
%% 
% Compute the validation accuracy.

validationAccuracy = 1 - kfoldLoss(partitionedModel,LossFun="ClassifError");
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);
%% 
% Visualize the confusion chart.

validationPredictions = kfoldPredict(partitionedModel);

    figure(Units="normalized", Position=[0.4 0.4 0.4 0.4])
    confusionchart(categorical(labels), categorical(validationPredictions), ...
        'Title', "Validation Accuracy (KNN)", ...
        'ColumnSummary', "column-normalized", ...
        'RowSummary', "row-normalized");
    
    % Predict the label (speaker) for each frame by calling |predict| on |trainedClassifier|.
    prediction = predict(trainedClassifier, features);
    prediction = categorical(string(prediction));
    
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
% Predict the label (speaker) for each frame by calling |predict| on |trainedClassifier|.

prediction = predict(trainedClassifier,features);
prediction = categorical(string(prediction));

    % Calculate accuracy per frame
    frameAccuracy = sum(prediction == categorical(labels)) / numel(labels);
    
    % Calculate average accuracy across all frames (if you have multiple frames to average)
    averageFrameAccuracy = mean(frameAccuracy); % assuming frameAccuracy holds multiple values in the loop
    fprintf('Average Frame Accuracy: %.2f%%\n', averageFrameAccuracy * 100);
 
    % Visualize the confusion chart for test predictions (per frame)
    figure(Units="normalized", Position=[0.4 0.4 0.4 0.4])
    confusionchart(categorical(labels(:)), prediction, ...
        'Title', "Test Accuracy Per Frame (KNN)", ...
        'ColumnSummary', "column-normalized", ...
        'RowSummary', "row-normalized");
    
    
    
    % Generate predictions for each file
    r2 = prediction(1:numel(adsTest.Files));
    idx = 1;
    for ii = 1:numel(adsTest.Files)
        r2(ii) = mode(prediction(idx:idx + numVectorsPerFile(ii) - 1));
        idx = idx + numVectorsPerFile(ii);
    end

    % Calculate accuracy per file
    fileAccuracy = sum(r2 == categorical(adsTest.Labels)) / numel(adsTest.Files);
    fprintf('Test Accuracy (Per File): %.2f%%\n', fileAccuracy * 100);

    % Visualize the confusion chart for test predictions (per file)
    figure(Units="normalized", Position=[0.4 0.4 0.4 0.4])
    confusionchart(categorical(adsTest.Labels), categorical(r2), ...
    'Title', "Test Accuracy Per File (KNN)", ...
    'ColumnSummary', "column-normalized", ...
    'RowSummary', "row-normalized");
    
    %% - Manual Testing with specific wav file
    testfeatures = [];


% Define the speakers used
labelArray = [1, 2, 3, 4, 5]; % Updated to reflect only 5 speakers
stringArray = [ ...

    "dr1-mcpm0/sa1", ... % Speaker 1
    "dr4-maeb0/sa1", ... % Speaker 2
    "dr5-mbgt0/sa1", ... % Speaker 3
    "dr7-madd0/sa1", ... % Speaker 4
    "dr8-mbcg0/sa1" ...  % Speaker 5
];

% Loop through all speakers
for select_Wav = 1:length(stringArray)
    str1 = stringArray(select_Wav);
    str2 = '.wav';
    str123 = strcat(dataFolder, str1, str2);
    [audioIn, fs] = audioread([str123]);  % Read the audio file
    sound(audioIn, fs);  % Play the audio

    pause(length(audioIn) / fs + 1);  % Pause to allow the audio to finish playing
end

str1 = stringArray(select_Wav);
str2 = '.wav';
% ------------ End of Change  ------------------
str123 = strcat(dataFolder, str1, str2);
[audioIn,fs] = audioread([str123]);  % this is like the readwav function of voicebox

sound(audioIn,fs);
myfeatures = extract(afe, audioIn);
% start
for ii = 1:size(myfeatures,1)
    thisFeature = myfeatures(ii,:);
    isSpeech = thisFeature(:,featureMap.shortTimeEnergy) > energyThreshold;
    isVoiced = thisFeature(:,featureMap.zerocrossrate) < zcrThreshold;
    voicedSpeech = isSpeech & isVoiced;
    thisFeature(~voicedSpeech,:) = []; % adding empty
    thisFeature(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];
    testfeatures = [testfeatures;thisFeature]; % injecting 3
end
testfeatures = (testfeatures-M)./S; % has to be added

% ------------  Change  ------------------
disp('Testing with my values:')
predictedLabel = predict(trainedClassifier,testfeatures);
fprintf("By Mode, for Selected Wav Index, %d, the predicted label is %d\n",select_Wav, mode(uint8(predictedLabel)));


% Create a histogram of the predicted labels
figure;
histogram(predictedLabel);
xlabel('Predicted Labels');
ylabel('Frequency');
title('Histogram of Predicted Labels (KNN)');
return; 
%% 
% Add the following:
%%
disp("Clear KNN model");
clear trainedClassifier;
disp("Completed and Exit");
return;
%%
%  This is specifically for manual prediction
disp("Saving KNN model and other variables");
save('KNN_model.mat', 'trainedClassifier');
save('myVariables.mat', 'dataFolder', 'afe', 'fs', 'featureMap','energyThreshold', 'zcrThreshold','M','S');
disp("Saving completed");
return;
%%
%  This is specifically for manual prediction
disp("Loading KNN model and other variables")
load('KNN_model.mat');  % This will load the 'gmdist' variable back into the workspace
load('myVariables.mat');  % This will load the 'gmdist' variable back into the workspace
disp("Loading completed");
return;
%%