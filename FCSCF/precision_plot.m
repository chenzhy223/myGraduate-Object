% 画出在不同精度要求下的准确率，thresholds是精度阈值，误差小于thresholds被认为是正确的，大于thresholds被认为是错误的。结果会在一个新的窗口上画出来。
% 输入参数：
% >>positions           根据算法预测出的目标中心坐标，是一个二维矩阵
% [目标中心坐标y,x ，目标大小height,width]
% >>ground_truth     是数据集里边的真实数据，转为了MATLAB坐标系[y0,x0,height,width]，中心坐标
% >>title                    测试所选的数据集名称
% >>show                 是否把结果画出来，是一个布尔值
% 输出参数：
% >>precisions         是精度，在不同阈值下的精度，默认分为50个精度

function precisions = precision_plot(positions, ground_truth, title, show)
% 传入的参数都是MATLAB坐标系下的，计算中心误差CLE、尺度中心误差
%误差允许的最大阈值
max_threshold = 50;  %used for graphs in the paper
precisions = zeros(max_threshold, 2);

%如果预测出的位置positions和数据集给出的数据ground_truth尺度不一致
%处理方法：直接舍弃多出来的
if size(positions,1) ~= size(ground_truth,1)
    % 		fprintf('%12s - Number of ground truth frames does not match number of tracked frames.\n', title)
    
    %just ignore any extra frames, in either results or ground truth
    n = min(size(positions,1), size(ground_truth,1));
    positions(n+1:end,:) = [];
    ground_truth(n+1:end,:) = [];
end

%calculate distances to ground truth over all frames
distances = sqrt((positions(:,1) - ground_truth(:,1)).^2 + ...
    (positions(:,2) - ground_truth(:,2)).^2);
distances(isnan(distances)) = [];%删除不是数据的数据

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

%compute precisions
for p = 1:max_threshold
    %nnz返回矩阵 X 中的非零元素数
    %这里非零意味着小于等于阈值，是认为正确的情况
    precisions(p,1) = nnz(distances <= p) / numel(distances);
    precisions(p,2) = nnz(overlaps >= 1/max_threshold*p) / numel(overlaps);
end

%show是是否画图的标志
%plot the precisions
    if show == 1
        figure('UserData','off', 'Name',['Precisions - ' title])%创建一个新的窗口
        subplot(1,2,1);
        plot([1:max_threshold],precisions(:,1), 'k-', 'LineWidth',2)
        xlabel('Threshold'), ylabel('Distance-Precision')
        xlim([1 max_threshold])
        
        subplot(1,2,2);
        plot(1/max_threshold*[1:max_threshold],precisions(:,2), 'g-', 'LineWidth',2)
        xlabel('Threshold'), ylabel('Overlap-Precision')
        xlim([1/max_threshold 1])
    end
end