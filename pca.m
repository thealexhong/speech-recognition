% Alexander Hong
% c4hongal 997584706
% March 2015
clc; clear; close all; fclose all;
load('knn_subset.mat');
fprintf('=== KNN Classification using PCA ===\n\n')

% 0-mean
train_data_ = bsxfun(@minus, train_data, mean(train_data, 2));
test_data_ =  bsxfun(@minus, test_data, mean(test_data, 2));
cov_train_data_ = cov(train_data_');
for PC = [5 10 20]
    [V, D] = eigs(cov_train_data_, PC); % PCA or SVD
    pcatrain = train_data_' * V;
    pcatest = test_data_' * V;
    dist = pdist2(pcatest, pcatrain, 'euclidean'); % knn algorithm
    [dist, idx] = sort(dist, 2, 'ascend');
    for K = [1]
       idx_ = idx(:, 1:K);
       prediction = mode(train_targets(idx_), 2);
       err = (1 - sum(test_targets == prediction)/length(test_targets)); 
       fprintf('Test Set Error (k = %d, PC = %d): %.2f%%\n', K, PC, 100 * err);
    end
end