function kf = gaussian_correlation(xf, yf, sigma)
%GAUSSIAN_CORRELATION Gaussian Kernel at all shifts, i.e. kernel correlation.
%   Evaluates a Gaussian kernel with bandwidth SIGMA for all relative
%   shifts between input images X and Y, which must both be MxN. They must 
%   also be periodic (ie., pre-processed with a cosine window). The result
%   is an MxN map of responses.
%
%   Inputs and output are all in the Fourier domain.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
	
	N = size(xf,1) * size(xf,2);
    %% 运用帕斯维尔定理求时域上的能量和
	xx = xf(:)' * xf(:) / N;  %squared norm of x
	yy = yf(:)' * yf(:) / N;  %squared norm of y
	
	%cross-correlation term in Fourier domain
	xyf = xf .* conj(yf);%计算两者的点乘
	xy = sum(real(ifft2(xyf)), 3);  %to spatial domain转换到时域，并且把多个捅得的值相加，得到的是二维矩阵
	
    %计算核矩阵K，并转换到频域上，这里有个限制，不允许出现负数
	%calculate gaussian response for all positions, then go back to the
	%Fourier domain
    %为啥要除以xf的总个数n*m*c
    
    kf = fft2(exp(-1 / sigma^2 * max(0, (xx + yy - 2 * xy) / numel(xf))));
    

	

end

