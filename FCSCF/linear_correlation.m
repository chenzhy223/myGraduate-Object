% 计算线性核函数的核相关矩阵K，输入、输出都是频域的，大小都一样.
% Computes the dot-product for all relative shifts between input images X and Y,
% which must both be MxN. They must also be periodic (ie., pre-processed with a cosine window).
% The result is an MxN map of responses.
% Inputs and output are all in the Fourier domain.
% Joao F. Henriques, 2014 http://www.isr.uc.pt/~henriques/
function kf = linear_correlation(xf, yf)
	% sum(A,3)表示在第三个维度上相加，形成二维矩阵
	%cross-correlation term in Fourier domain
    
% 	kf = sum(xf .* conj(yf), 3) / numel(xf);
    kf = sum(bsxfun(@times,xf,conj(yf)), 3) / numel(xf);

end
