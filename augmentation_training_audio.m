%% scripts that performs audio file augmentation and train netwrok AudioNet, which trains from particular sample sounds
clear all
close all
clc %clear all windows and variables in the working space
global current_folder
current_folder = pwd;
addpath(genpath(current_folder)); %add every files/folders under current working directory to working space
rng default

%% Declaring some paths and our labels

dataset = fullfile(pwd,'datasets');

%% Audio file augmentation and splitting training data

Config_dataset = 0;  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%if you already completed this process, please change it to 0
%else, set it to 1 so that the script will automatically augment and split
%data for you
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if Config_dataset == 1
    column_name = ["brushing_teeth","coughing","clapping","breathing","crying_baby","drinking_sipping","footsteps","laughing","sneezing","snoring","can_opening","sound","background"];
    for i = 1:length(column_name)
        sound_type = column_name(i)
        path = fullfile('data/human_sound/'+sound_type+'/'+sound_type +'*.wav');

        adsBkg{i} = dir(path);
        fpTrain = fullfile("data","human_sound","train",sound_type);
        fpValidation = fullfile("data","human_sound","validation",sound_type);
    
    
        % Create directories
        mkdir(fpTrain)
        mkdir(fpValidation)

        fs = 44100; % Known sample rate of the data set
        segmentDuration = 1;
        segmentSamples = round(segmentDuration*fs);
    
        for backgroundFileIndex = 1:numel(adsBkg{i})
            fn = adsBkg{i}(backgroundFileIndex).name;
            [bkgSegment, fileInfo] = detectVoiced(fullfile('data','human_sound',sound_type,adsBkg{i}(backgroundFileIndex).name), 1);
    
            for segmentIdx = 1:numel(bkgSegment)
                
                if size(bkgSegment{segmentIdx},1) >= segmentSamples
                    % Determine gain of each clip
                    segmentStart = randi([1,size(bkgSegment{segmentIdx},1)-segmentSamples+1],500,1);
                    
                    for seg = 1:numel(segmentStart)
                        
                        % Isolate the randomly chosen segment of data.
                        Segment{seg} = bkgSegment{segmentIdx}(segmentStart(seg):segmentStart(seg)+segmentSamples-1);
                
                        % Clip the audio between -1 and 1.
                        Segment{seg} = max(min(Segment{seg},1),-1);
    
                        % Create a file name.
                        afn = fn + "_segment" + seg + ".wav";
    
                        % Randomly assign background segment to either the train or validation set.
                        if rand > 0.85 % Assign 15% to validation
                            dirToWriteTo = fpValidation;
                        else % Assign 85% to train set.
                            dirToWriteTo = fpTrain;
                        end
    
                        % Write the audio to the file location.
                        ffn = fullfile(dirToWriteTo,afn);
                        audiowrite(ffn,Segment{seg},fs)
            
                    end
                end
            end
        end
    end
end
%% Declare ads where it includes train data
ads = audioDatastore(fullfile(dataset,"train"), ...
    IncludeSubfolders=true, ...
    FileExtensions=".wav", ...
    LabelSource="foldernames");

%%
commands = categorical(categorical(["brushing_teeth","coughing","clapping","sneezing","breathing","crying_baby","laughing","snoring","sound"]));
background = categorical("background");

isCommand = ismember(ads.Labels,commands);
isBackground = ismember(ads.Labels,background);
isUnknown = ~(isCommand|isBackground);

includeFraction = 0.5; % Fraction of unknowns to include.
idx = find(isUnknown);
idx = idx(randperm(numel(idx),round((1-includeFraction)*sum(isUnknown))));
isUnknown(idx) = false;

ads.Labels(isUnknown) = categorical("unknown");

adsTrain = subset(ads,isCommand|isUnknown|isBackground);
adsTrain.Labels = removecats(adsTrain.Labels);

%% Overwrite ads and now work for validation sets
ads = audioDatastore(fullfile(dataset,"validation"), ...
    IncludeSubfolders=true, ...
    FileExtensions=".wav", ...
    LabelSource="foldernames");

isCommand = ismember(ads.Labels,commands);
isBackground = ismember(ads.Labels,background);
isUnknown = ~(isCommand|isBackground);

includeFraction = 0.5; % Fraction of unknowns to include.
idx = find(isUnknown);
idx = idx(randperm(numel(idx),round((1-includeFraction)*sum(isUnknown))));
isUnknown(idx) = false;

ads.Labels(isUnknown) = categorical("unknown");

adsValidation = subset(ads,isCommand|isUnknown|isBackground);
adsValidation.Labels = removecats(adsValidation.Labels);

%% Data distruibution
figure(Units="normalized",Position=[0.2,0.2,0.5,0.5])

tiledlayout(2,1)

nexttile
histogram(adsTrain.Labels)
title("Training Label Distribution")
ylabel("Number of Observations")
grid on

nexttile
histogram(adsValidation.Labels)
title("Validation Label Distribution")
ylabel("Number of Observations")
grid on

%% Check if parallelpool enable
if canUseParallelPool
    useParallel = true;
    gcp;
else
    useParallel = false;
end
%% Data info for audioFeature extraction
fs = 44100; % Known sample rate of the data set.

segmentDuration = 1;
frameDuration = 0.05;
hopDuration = 0.010;
FFTLength = 2^nextpow2(round(frameDuration*fs)); % Use a power-of-two value for FFTLength
numBands = 100;

segmentSamples = round(segmentDuration*fs);
frameSamples = round(frameDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = frameSamples - hopSamples;
 
%% audioFeatureExtractor configuration
afe = audioFeatureExtractor( ...
    'SampleRate', fs, ...
    'Window', hann(frameSamples, 'periodic'), ...
    'FFTLength', FFTLength, ...
    'OverlapLength', overlapSamples, ...
    'barkSpectrum', true);
setExtractorParameters(afe, 'barkSpectrum', 'NumBands', numBands, 'WindowNormalization', false);


%% Train data preprocessing (Padding, Extraction Feature, Logarithm)
transform1 = transform(adsTrain,@(x)[zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)]);
transform2 = transform(transform1,@(x)extract(afe,x));
transform3 = transform(transform2,@(x){log10(x+1e-6)});
XTrain = readall(transform3,UseParallel=useParallel);
XTrain = cat(4,XTrain{:});
[numHops,numBands,numChannels,numFiles] = size(XTrain)

%% Validation data preprocessing (Padding, Extraction Feature, Logarithm)
transform1 = transform(adsValidation,@(x)[zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)]);
transform2 = transform(transform1,@(x)extract(afe,x));
transform3 = transform(transform2,@(x){log10(x+1e-6)});
XValidation = readall(transform3,UseParallel=useParallel);
XValidation = cat(4,XValidation{:});

%% Train & Validation data 
TTrain = adsTrain.Labels;
TValidation = adsValidation.Labels;

%% Dispay Bark spectrogram of training sets
specMin = min(XTrain,[],"all");
specMax = max(XTrain,[],"all");
idx = randperm(numFiles,3);
figure(Units="normalized",Position=[0.2,0.2,0.6,0.6]);

tiledlayout(2,3)
for ii = 1:3
    [x,fs] = audioread(adsTrain.Files{idx(ii)});

    nexttile(ii)
    plot(x)
    axis tight
    title(string(adsTrain.Labels(idx(ii))))
    
    nexttile(ii+3)
    spect = XTrain(:,:,1,idx(ii))';
    pcolor(spect)
    clim([specMin specMax])
    shading flat
    
    sound(x,fs)
    pause(2)
end

%% Model structure, layers and dropout 
classes = categories(TTrain);
classWeights = 1./countcats(TTrain);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(classes);

timePoolSize = ceil(numHops/8);

dropoutProb = 0.2;
numF = 12;
layers = [
    imageInputLayer([numHops,afe.FeatureVectorLength])
    
    convolution2dLayer(3,numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,2*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([timePoolSize,1])
    dropoutLayer(dropoutProb)

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer(Classes=classes,ClassWeights=classWeights)];

%% Training options - you can change parameters for validation
miniBatchSize = 10;
validationFrequency = floor(numel(TTrain)/miniBatchSize);
options = trainingOptions("adam", ...
    InitialLearnRate=3e-4, ...
    MaxEpochs=50, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=false, ...
    ValidationData={XValidation,TValidation}, ...
    ValidationFrequency=validationFrequency);

%% Consturt and train Model, here is where training starts
trainedNet = trainNetwork(XTrain,TTrain,layers,options);

%% Training & Validaton error, displayed via console
YValidation = classify(trainedNet,XValidation);
validationError = mean(YValidation ~= TValidation);
YTrain = classify(trainedNet,XTrain);
trainError = mean(YTrain ~= TTrain);

disp(["Training error: " + trainError*100 + "%";"Validation error: " + validationError*100 + "%"])

%% Confusion Matrix, figure popouts
figure(Units="normalized",Position=[0.2,0.2,0.5,0.5]);
cm = confusionchart(TValidation,YValidation, ...
    Title="Confusion Matrix for Validation Data", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");
sortClasses(cm,[commands,"unknown","background"])

%% Save model
save('audioNet_unknown.mat',"trainedNet");

%% Dispaly Network size and Single-image prediction time
for ii = 1:100
    x = randn([numHops,numBands]);
    predictionTimer = tic;
    [y,probs] = classify(trainedNet,x,ExecutionEnvironment="cpu");
    time(ii) = toc(predictionTimer);
end

disp(["Network size: " + whos("trainedNet").bytes/1024 + " kB"; ...
"Single-image prediction time on CPU: " + mean(time(11:end))*1000 + " ms"])

%% Functions eliminating silent part in Audio
function [segments, fs] = detectVoiced(wavFileName,t)

    fp = fopen(wavFileName, 'rb');
    if (fp<0)
	    fprintf('The file %s has not been found!\n', wavFileName);
	    return;
    end 
    fclose(fp);
    [x,fs] = audioread(wavFileName);

    % Convert mono to stereo
    if (size(x, 2)==2)
	    x = mean(x')';
    end
    % Window length and step (in seconds):
    win = 0.050;
    step = 0.050;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  THRESHOLD ESTIMATION
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    Weight = 5; % used in the threshold estimation method
    % Compute short-time energy and spectral centroid of the signal:
    Eor = ShortTimeEnergy(x, win*fs, step*fs);
    Cor = SpectralCentroid(x, win*fs, step*fs, fs);
    % Apply median filtering in the feature sequences (twice), using 5 windows:
    % (i.e., 250 mseconds)
    E = medfilt1(Eor, 5); E = medfilt1(E, 5);
    C = medfilt1(Cor, 5); C = medfilt1(C, 5);
    % Get the average values of the smoothed feature sequences:
    E_mean = mean(E);
    Z_mean = mean(C);
    % Find energy threshold:
    [HistE, X_E] = hist(E, round(length(E) / 10));  % histogram computation
    [MaximaE, countMaximaE] = findMaxima(HistE, 3); % find the local maxima of the histogram
    if (size(MaximaE,2)>=2) % if at least two local maxima have been found in the histogram:
        T_E = (Weight*X_E(MaximaE(1,1))+X_E(MaximaE(1,2))) / (Weight+1); % ... then compute the threshold as the weighted average between the two first histogram's local maxima.
    else
        T_E = E_mean / 2;
    end
    % Find spectral centroid threshold:
    [HistC, X_C] = hist(C, round(length(C) / 10));
    [MaximaC, countMaximaC] = findMaxima(HistC, 3);
    if (size(MaximaC,2)>=2)
        T_C = (Weight*X_C(MaximaC(1,1))+X_C(MaximaC(1,2))) / (Weight+1);
    else
        T_C = Z_mean / 2;
    end
    % Thresholding:
    Flags1 = (E>=T_E);
    Flags2 = (C>=T_C);
    flags = Flags1 & Flags2;
    % if (nargin==2) % plot results:
	%     clf;
	%     subplot(3,1,1); plot(Eor, 'g'); hold on; plot(E, 'c'); legend({'Short time energy (original)', 'Short time energy (filtered)'});
    %     L = line([0 length(E)],[T_E T_E]); set(L,'Color',[0 0 0]); set(L, 'LineWidth', 2);
    %     axis([0 length(Eor) min(Eor) max(Eor)]);
    % 
    %     subplot(3,1,2); plot(Cor, 'g'); hold on; plot(C, 'c'); legend({'Spectral Centroid (original)', 'Spectral Centroid (filtered)'});    
	%     L = line([0 length(C)],[T_C T_C]); set(L,'Color',[0 0 0]); set(L, 'LineWidth', 2);   
    %     axis([0 length(Cor) min(Cor) max(Cor)]);
    % end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  SPEECH SEGMENTS DETECTION
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    count = 1;
    WIN = 5;
    Limits = [];
    while (count < length(flags)) % while there are windows to be processed:
	    % initilize:
	    curX = [];	
	    countTemp = 1;
	    % while flags=1:
	    while ((flags(count)==1) && (count < length(flags)))
		    if (countTemp==1) % if this is the first of the current speech segment:
			    Limit1 = round((count-WIN)*step*fs)+1; % set start limit:
			    if (Limit1<1)	Limit1 = 1; end        
		    end	
		    count = count + 1; 		% increase overall counter
		    countTemp = countTemp + 1;	% increase counter of the CURRENT speech segment
	    end
	    if (countTemp>1) % if at least one segment has been found in the current loop:
		    Limit2 = round((count+WIN)*step*fs);			% set end counter
		    if (Limit2>length(x))
                Limit2 = length(x);
            end
            
            Limits(end+1, 1) = Limit1;
            Limits(end,   2) = Limit2;
        end
	    count = count + 1; % increase overall counter
    end
    %%%%%%%%%%%%%%%%%%%%%%%
    % POST - PROCESS      %
    %%%%%%%%%%%%%%%%%%%%%%%
    % A. MERGE OVERLAPPING SEGMENTS:
    RUN = 1;
    while (RUN==1)
        RUN = 0;
        for (i=1:size(Limits,1)-1) % for each segment
            if (Limits(i,2)>=Limits(i+1,1))
                RUN = 1;
                Limits(i,2) = Limits(i+1,2);
                Limits(i+1,:) = [];
                break;
            end
        end
    end
    % B. Get final segments:
    segments = {};
    for (i=1:size(Limits,1))
        segments{end+1} = x(Limits(i,1):Limits(i,2)); 
    end
    
    % output = [];
    % for i = length(segments)
    %     output = [segments{i}; output];
    % end
end

function [Maxima, countMaxima] = findMaxima(f, step)

    % STEP 1: find maxima:
    % 
    countMaxima = 0;
    for (i=1:length(f)-step-1) % for each element of the sequence:
        if (i>step)
            if (( mean(f(i-step:i-1))< f(i)) && ( mean(f(i+1:i+step))< f(i)))  
                % IF the current element is larger than its neighbors (2*step window)
                % --> keep maximum:
                countMaxima = countMaxima + 1;
                Maxima(1,countMaxima) = i;
                Maxima(2,countMaxima) = f(i);
            end
        else
            if (( mean(f(1:i))<= f(i)) && ( mean(f(i+1:i+step))< f(i)))  
                % IF the current element is larger than its neighbors (2*step window)
                % --> keep maximum:
                countMaxima = countMaxima + 1;
                Maxima(1,countMaxima) = i;
                Maxima(2,countMaxima) = f(i);
            end
            
        end
    end
    
   
    % STEP 2: post process maxima:
    MaximaNew = [];
    countNewMaxima = 0;
    i = 0;
    while (i<countMaxima)
        % get current maxima:
        i = i + 1;
        curMaxima = Maxima(1,i);
        curMavVal = Maxima(2,i);
        
        tempMax = Maxima(1,i);
        tempVals = Maxima(2,i);
        
        % search for "neighbourh maxima":
        while ((i<countMaxima) && ( Maxima(1,i+1) - tempMax(end) < step / 2))
            i = i + 1;
            tempMax(end+1) = Maxima(1,i);
            tempVals(end+1) = Maxima(2,i);
        end
        
       
        % find the maximum value and index from the tempVals array:
        %MI = findCentroid(tempMax, tempVals); MM = tempVals(MI);
        
        [MM, MI] = max(tempVals);
            
        if (MM>0.02*mean(f)) % if the current maximum is "large" enough:
            countNewMaxima = countNewMaxima + 1;   % add maxima
            % keep the maximum of all maxima in the region:
            MaximaNew(1,countNewMaxima) = tempMax(MI); 
            MaximaNew(2,countNewMaxima) = f(MaximaNew(1,countNewMaxima));
        end        
        tempMax = [];
        tempVals = [];
    end
    Maxima = MaximaNew;
    countMaxima = countNewMaxima;
end

function E = ShortTimeEnergy(signal, windowLength,step);
    signal = signal / max(max(signal));
    curPos = 1;
    L = length(signal);
    numOfFrames = floor((L-windowLength)/step) + 1;
    %H = hamming(windowLength);
    E = zeros(numOfFrames,1);
    for (i=1:numOfFrames)
        window = (signal(curPos:curPos+windowLength-1));
        E(i) = (1/(windowLength)) * sum(abs(window.^2));
        curPos = curPos + step;
    end
end

function C = SpectralCentroid(signal,windowLength, step, fs)

    signal = signal / max(abs(signal));
    curPos = 1;
    L = length(signal);
    numOfFrames = floor((L-windowLength)/step) + 1;
    H = hamming(windowLength);
    m = ((fs/(2*windowLength))*[1:windowLength])';
    C = zeros(numOfFrames,1);
    for (i=1:numOfFrames)
        window = H.*(signal(curPos:curPos+windowLength-1));    
        FFT = (abs(fft(window,2*windowLength)));
        FFT = FFT(1:windowLength);  
        FFT = FFT / max(FFT);
        C(i) = sum(m.*FFT)/sum(FFT);
        if (sum(window.^2)<0.010)
            C(i) = 0.0;
        end
        curPos = curPos + step;
    end
    C = C / (fs/2);
end