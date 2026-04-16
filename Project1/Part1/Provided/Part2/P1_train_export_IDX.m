%% P1_train_export_IDX.m 
% Trains MLP on IDX files and exports weights/biases
% Folder structure expected:
%   ./training_data/train-images.idx3-ubyte
%   ./training_data/train-labels.idx1-ubyte
%   ./test_data/t10k-images.idx3-ubyte
%   ./test_data/t10k-labels.idx1-ubyte
%
% You may tune: hiddenSizes, hiddenAct, epochs, resizeTo.

clear; clc; close all;

%% ===================== CONFIG (STUDENTS EDIT HERE) ===================== %%
cfg.trainDir = fullfile(pwd, "training_data");
cfg.testDir  = fullfile(pwd, "test_data");

cfg.trainImagesFile = "train-images.idx3-ubyte";
cfg.trainLabelsFile = "train-labels.idx1-ubyte";
cfg.testImagesFile  = "t10k-images.idx3-ubyte";
cfg.testLabelsFile  = "t10k-labels.idx1-ubyte";

% If IDX images are 28x28 but project wants 16x16, set:
cfg.resizeTo = [16 16];   % [] keeps original size (e.g., 28x28 -> 784 inputs)

cfg.normalize01 = true;   % scale pixels to [0,1]
cfg.seed = 1;

% Network hyperparameters
cfg.hiddenSizes = [64 32]; % change vector length to change #hidden layers
cfg.hiddenAct   = "tansig"; % "tansig" | "logsig" | "poslin"
cfg.epochs      = 50;

% Only train/val split inside training set (external test is separate)
cfg.trainRatio = 0.85;
cfg.valRatio   = 0.15;

cfg.saveFile = "trained_mlp_export.mat";
%% ======================================================================= %%

rng(cfg.seed);

trainImagesPath = fullfile(cfg.trainDir, cfg.trainImagesFile);
trainLabelsPath = fullfile(cfg.trainDir, cfg.trainLabelsFile);
testImagesPath  = fullfile(cfg.testDir,  cfg.testImagesFile);
testLabelsPath  = fullfile(cfg.testDir,  cfg.testLabelsFile);

[Xtr, Ytr, infoTr] = load_idx_dataset(trainImagesPath, trainLabelsPath, cfg);
[Xte, Yte, infoTe] = load_idx_dataset(testImagesPath,  testLabelsPath,  cfg);

fprintf("TRAIN: %d samples | %dx%d | inputDim=%d\n", size(Xtr,2), infoTr.rows, infoTr.cols, size(Xtr,1));
fprintf("TEST : %d samples | %dx%d | inputDim=%d\n", size(Xte,2), infoTe.rows, infoTe.cols, size(Xte,1));

% Build MLP (patternnet)
net = patternnet(cfg.hiddenSizes);
net.divideParam.trainRatio = cfg.trainRatio;
net.divideParam.valRatio   = cfg.valRatio;
net.divideParam.testRatio  = 0.00;   % DO NOT use internal test split

net.performFcn = 'crossentropy';
net.trainParam.epochs = cfg.epochs;
net.trainParam.showWindow = true;

for i = 1:numel(cfg.hiddenSizes)
    net.layers{i}.transferFcn = char(cfg.hiddenAct);
end
net.layers{end}.transferFcn = 'softmax';

% Train
[net, tr] = train(net, Xtr, Ytr);

% Toolbox evaluation (OK here)
Ypred = net(Xte);
[~, predLab] = max(Ypred, [], 1);
[~, trueLab] = max(Yte,   [], 1);
acc = mean(predLab == trueLab);

fprintf("\nToolbox external TEST accuracy: %.2f%%\n", 100*acc);

figure; plotconfusion(Yte, Ypred);
title(sprintf("Toolbox Confusion Matrix | Acc=%.2f%%", 100*acc));

% Export W,b
numLayers = net.numLayers;
W = cell(numLayers, 1);
b = cell(numLayers, 1);

W{1} = net.IW{1,1};
b{1} = net.b{1};
for L = 2:numLayers
    W{L} = net.LW{L, L-1};
    b{L} = net.b{L};
end

export = struct();
export.cfg = cfg;
export.W = W;
export.b = b;
export.tr = tr;
export.toolboxTestAcc = acc;
export.netInfo = struct( ...
    "inputDim", size(Xtr,1), ...
    "hiddenSizes", cfg.hiddenSizes, ...
    "outputDim", 10, ...
    "hiddenAct", cfg.hiddenAct, ...
    "outputAct", "softmax" ...
);
export.imgInfoTrain = infoTr;
export.imgInfoTest  = infoTe;

save(cfg.saveFile, "export");
fprintf("Saved weights/biases to: %s\n", cfg.saveFile);

%% ------------------------- Helper functions ------------------------- %%
function [X, Y, info] = load_idx_dataset(imagesPath, labelsPath, cfg)
images = read_idx3_ubyte(imagesPath);  % uint8 [rows x cols x N]
labels = read_idx1_ubyte(labelsPath);  % uint8 [N x 1]

rows = size(images,1);
cols = size(images,2);
N    = size(images,3);

if numel(labels) ~= N
    error("Label count (%d) != image count (%d).", numel(labels), N);
end

if ~isempty(cfg.resizeTo)
    r2 = cfg.resizeTo(1); c2 = cfg.resizeTo(2);
    resized = zeros(r2, c2, N, 'double');
    for i = 1:N
        resized(:,:,i) = imresize(double(images(:,:,i)), [r2 c2]);
    end
    images = resized;
    rows = r2; cols = c2;
else
    images = double(images);
end

if cfg.normalize01 && max(images(:)) > 1
    images = images ./ 255;
end

X = reshape(images, rows*cols, N);

Y = zeros(10, N);
for i = 1:N
    Y(labels(i)+1, i) = 1;
end

info = struct("rows", rows, "cols", cols, "N", N);
end

function images = read_idx3_ubyte(filename)
fid = fopen(filename, 'rb');
if fid < 0, error("Cannot open file: %s", filename); end

magic = fread(fid, 1, 'int32', 0, 'ieee-be');
if magic ~= 2051
    fclose(fid);
    error("Invalid IDX3 magic number in %s (got %d).", filename, magic);
end

numImages = fread(fid, 1, 'int32', 0, 'ieee-be');
numRows   = fread(fid, 1, 'int32', 0, 'ieee-be');
numCols   = fread(fid, 1, 'int32', 0, 'ieee-be');

raw = fread(fid, numImages*numRows*numCols, 'uint8');
fclose(fid);

images = reshape(raw, [numCols, numRows, numImages]);
images = permute(images, [2 1 3]);
end

function labels = read_idx1_ubyte(filename)
fid = fopen(filename, 'rb');
if fid < 0, error("Cannot open file: %s", filename); end

magic = fread(fid, 1, 'int32', 0, 'ieee-be');
if magic ~= 2049
    fclose(fid);
    error("Invalid IDX1 magic number in %s (got %d).", filename, magic);
end

numLabels = fread(fid, 1, 'int32', 0, 'ieee-be');
labels = fread(fid, numLabels, 'uint8');
fclose(fid);
end
