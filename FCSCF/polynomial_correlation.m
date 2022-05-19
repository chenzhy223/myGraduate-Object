function kf = polynomial_correlation(xf, yf, a, b)
	
	%cross-correlation term in Fourier domain
% 	xyf = xf .* conj(yf);
    xyf = bsxfun(@times,xf,conj(yf));
	xy = sum(real(ifft2(xyf)), 3);  %to spatial domain
	
	%calculate polynomial response for all positions, then go back to the
	%Fourier domain
    %xy / numel(xf)进行归一化
	kf = fft2((xy / numel(xf) + a) .^ b);

end
