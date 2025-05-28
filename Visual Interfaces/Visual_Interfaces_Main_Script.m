%% Initialization
clc;
clear all;
close all;
warning off;

% Load models
load faceRecognitionModel; % Pre-trained face recognition model
load FingerGestureModel;   % Pre-trained gesture classification model

% Create the face detector object
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART', 'MinSize', [150, 150]);

% Create the point tracker object for face tracking
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize webcam and video player
cam = webcam();
videoFrame = snapshot(cam);
frameSize = size(videoFrame);
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)] + 30]);

% Define the ROI for hand gestures (top-left corner)
x = 10; y = 10; width = 300; height = 300; % Adjust ROI size as needed
gestureROI = [x, y, width, height];

% Variables for face detection/tracking
runLoop = true;
numPts = 0;
bboxPoints = [];
oldPoints = [];
frameCount = 0;

%% Main Loop
while runLoop
    % Capture the current frame
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);

    % Face Detection/Tracking
    if numPts < 10
        % Detection mode
        bbox = faceDetector.step(videoFrameGray);
        if ~isempty(bbox)
            % Detect points within the face region
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));
            xyPoints = points.Location;
            numPts = size(xyPoints, 1);

            % Reinitialize the tracker
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);

            % Save points and bounding box
            oldPoints = xyPoints;
            bboxPoints = bbox2points(bbox(1, :));
        end
    else
        % Tracking mode
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);
        numPts = size(visiblePoints, 1);

        if numPts >= 10
            % Geometric transformation for bounding box
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
            bboxPoints = transformPointsForward(xform, bboxPoints);
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);

            % Face classification
            [img, faceDetected] = cropface(videoFrame);
            if faceDetected == 1
                img = imresize(img, [227, 227]);
                facePrediction = classify(faceRecognitionModel, img);
                feedbackFace = char(facePrediction);
            else
                feedbackFace = 'No Face Detected';
            end

            % Draw face bounding box
            bboxPolygon = reshape(bboxPoints', 1, []);
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            videoFrame = insertText(videoFrame, bboxPoints(1, :), feedbackFace, ...
                'FontSize', 20, 'BoxColor', 'green', 'BoxOpacity', 0.6, 'TextColor', 'white');
        end
    end

    % Hand Gesture Classification
    % Crop the ROI for gestures
    handFrame = imcrop(videoFrame, gestureROI);
    handFrame = imresize(handFrame, [227, 227]);
    gesturePrediction = classify(FingerGestureModel, handFrame);
    feedbackGesture = char(gesturePrediction);

    % Annotate gesture ROI and prediction
    videoFrame = insertShape(videoFrame, 'Rectangle', gestureROI, 'LineWidth', 3, 'Color', 'blue');
    videoFrame = insertText(videoFrame, [gestureROI(1), gestureROI(2) + gestureROI(4) + 10], feedbackGesture, ...
        'FontSize', 20, 'BoxColor', 'blue', 'BoxOpacity', 0.6, 'TextColor', 'white');

    % Display the annotated video frame
    step(videoPlayer, videoFrame);

    % Check if the video player window is still open
    runLoop = isOpen(videoPlayer);
end

%% Clean Up
clear cam;
release(videoPlayer);
release(faceDetector);
release(pointTracker);
