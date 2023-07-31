%% load
% load('testAudio.mat');  % audio file (y, fs)
addpath(genpath(pwd));
inputPath='./test';  % test file
outputPath = './results';  % log, cough record
load('./model/audioNet.mat');  % trainedNet
load('./model/videoNet.mat','videoNet');

%% prepare audio
fileName = 'Ha_re.m4a'; % Update with your test file name
[~,name,~] = fileparts(fileName);
[audioData, fs] = audioread(fullfile(inputPath,fileName));
if size(audioData,2) ~= 1
    audioMono = mean(audioData, 2); % Convert to mono if stereo
else
    audioMono = audioData;
end
%% split audio by 1s
n_tick = floor(size(audioMono,1)/fs);
audio_tick = zeros(fs,n_tick);
for i=1:n_tick
    audio_tick(:,i) = audioMono(1+fs*(i-1):fs*i);
end

%%
fs = 48000; % Known sample rate of the data set.

segmentDuration = 1;
frameDuration = 0.05;
hopDuration = 0.010;
FFTLength = 2^nextpow2(round(frameDuration*fs)); % Use a power-of-two value for FFTLength
numBands = 100;

segmentSamples = round(segmentDuration*fs);
frameSamples = round(frameDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = frameSamples - hopSamples;
 
%%
afe = audioFeatureExtractor( ...
    'SampleRate', fs, ...
    'Window', hann(frameSamples, 'periodic'), ...
    'FFTLength', FFTLength, ...
    'OverlapLength', overlapSamples, ...
    'barkSpectrum', true);
setExtractorParameters(afe, 'barkSpectrum', 'NumBands', numBands, 'WindowNormalization', false);

%%
transform1 = [zeros(floor((segmentSamples-size(audio_tick,1))/2),1);audio_tick;zeros(ceil((segmentSamples-size(audio_tick,1))/2),1)];
transform2 = extract(afe,transform1);
transform3 = log10(transform2+1e-6);
XTest = reshape(transform3, size(transform3,1), size(transform3,2), 1, size(transform3,3));

%% Classify
outputLabels = trainedNet.Layers(end).Classes;
[YTest,prob] = classify(trainedNet,XTest);
Yindex_cough = find(YTest=='coughing');
n_total = size(YTest,1);
n_cough = sum(YTest == 'coughing')
coughRate = n_cough/size(YTest,1)*100
[mostFrequentValues, chunkCoughMaxProb, chunkCoughResult, Cough] = coughDetermine(YTest, find(outputLabels=='coughing'), prob);  % Cough=1이면 coughing

%% plot
DateTime = datetime('now');  % record diagnose time
if n_cough >= 4  % 최대 4개만 plot
    n_plot = 4;
else
    n_plot = n_cough;
end
Yindex_plot = sort(Yindex_cough(randperm(length(Yindex_cough), n_plot)));
t = (1:size(audioMono,1))/fs;
bands = 1:numBands;

coughTimeH = floor((Yindex_plot-1)/3600);
coughTimeM = floor(((Yindex_plot-1)-coughTimeH*3600)/60);
coughTimeS = (Yindex_plot-1)-coughTimeM*60-coughTimeH*3600;

figure(Name='Result', Units="normalized", Position=[0,0.3,0.25*n_plot,0.4], NumberTitle='off');
tiledlayout(1,n_plot+2)
figureTitle=sprintf('%s %s',name,DateTime);
sgtitle(figureTitle,'FontSize',25,'FontWeight','bold');
for i=1:n_plot
    nexttile
    imagesc(t((Yindex_plot(i)-1)*fs+1:(Yindex_plot(i))*fs),bands,transform3(:,:,Yindex_plot(i)));
    shading flat
    axis tight;
    axis xy;
    xlabel('Time (s)');
    ylabel('Bark Bands');

    Title=sprintf('%02d:%02d:%02d',coughTimeH(i),coughTimeM(i),coughTimeS(i));
    title(Title);
    colorbar;

    sound(audio_tick(:,Yindex_plot(i)),fs);
    pause(1);
end

imgName = sprintf('./test/%s.jpg',name);
img = imread(imgName);
img = imresize(img, [227, 227])
[label, scores] = classify(videoNet, img);
[~, idx] = sort(scores, 'descend');
idx = idx(3:-1:1);
classNamesTop = videoNet.Layers(end).Classes(idx);
scoresTop = scores(idx);

nexttile;
imshow(img)
title(label)
nexttile;
barh(scoresTop)
xlim([0 1]);
title('Top 3 Predictions')
xlabel('Probability')
yticklabels(classNamesTop)

%% record cough sound
audio_cough = [];
for i=1:n_cough
    audio_cough = cat(1,audio_cough, audio_tick(:,Yindex_cough(i)));
end
audiowrite(fullfile(outputPath,'cough_sound.wav'),audio_cough,fs);

%% print log
% create log file if not exists
if exist(fullfile(outputPath,'patientLog.csv'), 'file') == 0
    log = fopen(fullfile(outputPath,'patientLog.csv'),'w');
    fprintf(log,'Name,DateTime,Kit,Cough,SuspendedCough\n');
    fclose(log);
end
% write log line
log = fopen(fullfile(outputPath,'patientLog.csv'),'a');
% time stamp
coughTimeH = floor((Yindex_cough-1)/3600);
coughTimeM = floor(((Yindex_cough-1)-coughTimeH*3600)/60);
coughTimeS = (Yindex_cough-1)-coughTimeM*60-coughTimeH*3600;
timeMatrix = [coughTimeH, coughTimeM, coughTimeS];
formatTLog = '%02d:%02d:%02d, ';
% cough
if Cough==1
    coughPrint = 'Yes';
else
    coughPrint = 'No';
end
logLine1 = sprintf('%s,%s,%s,%s,[',name,DateTime,label,coughPrint);
logLine2 = sprintf('%02d:%02d:%02d;', timeMatrix');
logLine = strcat(logLine1,logLine2(1:end-1),']')

fprintf(log,logLine);
fprintf(log,'\n');

fclose(log);