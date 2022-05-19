% 生成高斯分布的标签.
% 为维度sz样本创建标签，输出的标签和样本sz的大小一样，高斯分布的，
% 最大值在左上角（0位移），随着距离的增加二衰减，在边界缠绕。
% LABELS = GAUSSIAN_SHAPED_LABELS(SIGMA, SZ) Creates an array of 
% labels (regression targets) for all shifts of a sample of dimensions SZ. 
% The output will have size SZ, representing one label for each possible shift. 
% The labels will be Gaussian-shaped, with the peak at 0-shift 
% (top-left element of the array), decaying as the distance increases, 
% and wrapping around at the borders. The Gaussian function has spatial bandwidth SIGMA.
% Joao F. Henriques, 2014 http://www.isr.uc.pt/~henriques/
function labels = gaussian_shaped_labels(sigma, sz)
%与fDSST中的方法有区别
% 	%as a simple example, the limit sigma = 0 would be a Dirac delta,
% 	%instead of a Gaussian:
% 	labels = zeros(sz(1:2));  %labels for all shifted samples
% 	labels(1,1) = magnitude;  %label for 0-shift (original sample)
	
    %划分高斯分布的坐标，中心为坐标原点
	%evaluate a Gaussian with the peak at the center element
	[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
	labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
    %循环移动，把中心移动到四个角落
	%move the peak to the top-left, with wrap-around
	labels = circshift(labels, -floor(sz(1:2) / 2) + 1);

	%sanity check: make sure it's really at top-left
	assert(labels(1,1) == 1)%断言，看看有无错误
end
