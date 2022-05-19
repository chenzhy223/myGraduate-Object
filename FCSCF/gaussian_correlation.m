% 计算高斯核相关矩阵K.
% 输入的两个相关量的维度必须一致，并且是周期的（经过窗函数处理），输入和输出的维度都一样
% 输入和输出都是在频域上的.
% Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts 
% between input images X and Y, which must both be MxN. They must also be periodic
% (ie., pre-processed with a cosine window). The result is an MxN map of responses.
% Joao F. Henriques, 2014 http://www.isr.uc.pt/~henriques/
function kf = gaussian_correlation(xf, yf, sigma)
	
	N = size(xf,1) * size(xf,2);
    %归一化
    %计算二范数，xf(:)是把xf拉伸成一维列向量
	xx = xf(:)' * xf(:) / N;  %squared norm of x
	yy = yf(:)' * yf(:) / N;  %squared norm of y
	
	%cross-correlation term in Fourier domain
% 	xyf = xf .* conj(yf);%计算两者的点乘而非矩阵相乘
    xyf = bsxfun(@times,xf,conj(yf));
    %转换到时域，并且把多个捅得的值相加，得到的是二维矩阵
    %这样的计算方法是根据公式来的
	xy = sum(real(ifft2(xyf)), 3);  %to spatial domain
	
    %计算核矩阵K，并转换到频域上，这里有个限制，不允许出现负数
	%calculate gaussian response for all positions, then go back to the
	%Fourier domain
    %为啥要除以xf的总个数n*m*c
	kf = fft2(exp(-1 / sigma^2 * max(0, (xx + yy - 2 * xy) / numel(xf))));

end
