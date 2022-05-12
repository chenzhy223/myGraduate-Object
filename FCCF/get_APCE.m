% 计算平均峰值相关能量APCE。
% 输入是一个响应图，二维矩阵，输出是一个值。
function [max_Gaumap,APCE] = get_APCE(Gaumap)
    max_Gaumap = max(Gaumap(:));
    min_Gaumap = min(Gaumap(:));
    APCE = (max_Gaumap-min_Gaumap).^2 / (mean2((Gaumap-min_Gaumap).^2));
end