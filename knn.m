% Alexander Hong
% c4hongal 997584706
% March 2015
clc; clear; close all; fclose all;
load('knn_subset.mat');
D = pdist2(test_data', train_data', 'euclidean'); % all combination of dist
[D, idx] = sort(D, 2, 'ascend');
fprintf('=== KNN Classification ===\n\n')

for corrupt = [0, 1]
    if corrupt == 1 % corrupting the data
        inds = randperm(4400);
        train_targets_(inds(1:440)) = randi(44,440,1);
        fprintf('\n10%% label corruption\n');
    else
        train_targets_ = train_targets; % non-corrupt data
    end
    
    for K = [1, 3, 5] % K-values
       idx_ = idx(:, 1:K);
       prediction = mode(train_targets_(idx_), 2); % train knn
       err = (1 - sum(test_targets == prediction)/length(test_targets)); % error calculation

       fprintf('Test Set Error (k = %d): %.2f%%\n', K, 100 * err);
    end
end