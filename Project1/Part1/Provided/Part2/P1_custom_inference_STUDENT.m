%% P1_custom_inference_STUDENT.m  (STUDENT MUST COMPLETE)
% Goal: Load exported W,b and implement custom MLP inference from scratch.
% RULE: Do NOT use net(X), sim(), classify(), predict(), etc.
%
% Run AFTER you run P1_train_export_IDX.m (it creates trained_mlp_export.mat)

clear; clc; close all;

exportFile = "trained_mlp_export.mat";
S = load(exportFile, "export");
export = S.export;
cfg = export.cfg;

% Load test set (IDX)
testImagesPath = fullfile(cfg.testDir, cfg.testImagesFile);
testLabelsPath = fullfile(cfg.testDir, cfg.testLabelsFile);
[Xte, Yte] = load_idx_dataset(testImagesPath, testLabelsPath, cfg);

% ====================== TODO 1: CUSTOM FORWARD PASS ======================
% Implement forward_mlp(X, W, b, hiddenAct) and use it here:
Ypred = forward_mlp(Xte, export.W, export.b, export.netInfo.hiddenAct);
% =========================================================================

[~, predLab] = max(Ypred, [], 1);
[~, trueLab] = max(Yte,   [], 1);
acc = mean(predLab == trueLab);

fprintf("CUSTOM inference TEST accuracy: %.2f%%\n", 100*acc);

% Confusion matrix
figure;
cm = confusionmat(trueLab, predLab);
confusionchart(cm);
title(sprintf("Custom Inference Confusion Matrix | Acc=%.2f%%", 100*acc));

%% ========================= STUDENT FUNCTIONS ========================= %%
function Y = forward_mlp(X, W, b, hiddenAct)
% TODO 2:
% - X is inputDim x N
% - W{l}, b{l} define each layer
% - For hidden layers: apply hiddenAct activation
% - For output layer: apply softmax
%
% Return Y = 10 x N probabilities (softmax outputs)
%
% Hint:
% A = X;
% L = numel(W);
% for l = 1:L
%    Z = W{l}*A + b{l};
%    if l < L
%        A = activation(Z, hiddenAct);
%    else
%        Y = softmax(Z);
%    end
% end

error("TODO: Implement forward_mlp()");
end

function A = activation(Z, hiddenAct)
% TODO 3: implement one or more activations:
% - logsig: 1/(1+exp(-Z))
% - tansig: 2/(1+exp(-2Z)) - 1
% - poslin: max(0, Z)

error("TODO: Implement activation()");
end

function S = softmax(Z)
% TODO 4: implement a numerically stable softmax:
% Z = Z - max(Z,[],1);
% expZ = exp(Z);
% S = expZ ./ sum(expZ,1);

error("TODO: Implement softmax()");
end

%% ===================== PROVIDED: IDX Loader (DO NOT EDIT) =====================
function [X, Y] = load_idx_dataset(imagesPath, labelsPath, cfg)
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
else
    images = double(images);
end

if cfg.normalize01 && max(images(:)) > 1
    images = images ./ 255;
end

X = reshape(images, [], N);

Y = zeros(10, N);
for i = 1:N
    Y(labels(i)+1, i) = 1;
end
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
