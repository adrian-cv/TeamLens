# TeamLens
repository for MathWorksKorea's 2023 AI competition

File explanation
augmentation_training_audio.m - script for data augmentation and creating/training of AudioNet.mat which classifies coughing sounds to detect covid-19 patients.
training_kit_image.m - script for creating and training VideoNet.mat which classifies different types of RDTs.

For the data folder, only few of the samples are included. If you want more data, refer to below:
RDTs image from https://github.com/dmendels-collab/xRcovid
Human relevant sounds - ESC50: https://github.com/karolpiczak/ESC-50
Korean Speech Datasets: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=116
The test data, which is some voice recordings, are our own creation.
