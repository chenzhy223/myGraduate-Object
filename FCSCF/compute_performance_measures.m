function [distance_precision, overlap_precision, average_center_location_error,S] = ...
    compute_performance_measures(positions, ground_truth, distance_precision_threshold, overlap_threshold)
% 输入坐标采用MATLAB坐标系，positions, ground_truth前面两个是目标的中心位置
% 计算平均中心误差CLE、重叠率S、距离精度DP、重叠率精度OP
% distance_precision：距离精度DP
% overlap_precision:重叠率精度OP
% average_center_location_error:平均中心误差CLE
% S:重叠率S
% For the given tracker output positions and ground truth it computes the:
% * Distance Precision at the specified threshold (20 pixels as default if
% omitted)
% * overlap Precision at the specified threshold (0.5 as default if omitted)
% * Average Center Location error (CLE).

if nargin < 3 || isempty(distance_precision_threshold)
    distance_precision_threshold = 20;
end
if nargin < 4 || isempty(overlap_threshold)
    overlap_threshold = 0.5;
end

if size(positions,1) ~= size(ground_truth,1)
    disp('Could not calculate precisions, because the number of ground')
    disp('truth frames does not match the number of tracked frames.')
    return
end


%calculate distances to ground truth over all frames
distances = sqrt((positions(:,1) - ground_truth(:,1)).^2 + ...
    (positions(:,2) - ground_truth(:,2)).^2);
distances(isnan(distances)) = [];

%% 计算DP
distance_precision = nnz(distances < distance_precision_threshold) / numel(distances);

%% calculate average center location error (CLE)
%计算CLE
average_center_location_error = mean(distances);

%% 计算重叠区域
% calculate the overlap in each dimension
%重叠部分的高
overlap_height = min(positions(:,1) + positions(:,3)/2, ground_truth(:,1) + ground_truth(:,3)/2) ...
    - max(positions(:,1) - positions(:,3)/2, ground_truth(:,1) - ground_truth(:,3)/2);
%重叠部分的宽
overlap_width = min(positions(:,2) + positions(:,4)/2, ground_truth(:,2) + ground_truth(:,4)/2) ...
    - max(positions(:,2) - positions(:,4)/2, ground_truth(:,2) - ground_truth(:,4)/2);

% if no overlap, set to zero
% 处理超出边界的
overlap_height(overlap_height < 0) = 0;
overlap_width(overlap_width < 0) = 0;

% 得到有效值的索引
% remove NaN values (should not exist any)
valid_ind = ~isnan(overlap_height) & ~isnan(overlap_width);

% calculate area
overlap_area = overlap_height(valid_ind) .* overlap_width(valid_ind);%重叠部分的面积
tracked_area = positions(valid_ind,3) .* positions(valid_ind,4);%跟踪器预测的目标面积
ground_truth_area = ground_truth(valid_ind,3) .* ground_truth(valid_ind,4);%数据集标注的目标面积

%% 计算重叠率S
% calculate PASCAL overlaps
overlaps = overlap_area ./ (tracked_area + ground_truth_area - overlap_area);
S = mean(overlaps);
%% 计算重叠率精度
% calculate PASCAL precision
overlap_precision = nnz(overlaps >= overlap_threshold) / numel(overlaps);
end