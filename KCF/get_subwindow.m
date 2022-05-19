function out = get_subwindow(im, pos, sz)
%GET_SUBWINDOW Obtain sub-window from image, with replication-padding.
%   Returns sub-window of image IM centered at POS ([y, x] coordinates（坐标）),
%   with size SZ ([height, width]). If any pixels are outside of the image,
%   they will replicate（复制） the values at the borders.
%这个函数保证取出的图像块和目标大小一致，统一处理方法
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
    %判断sz是否是标量（单数）
	if isscalar(sz)  %square sub-window
		sz = [sz, sz];
	end
	%起始的位置为pos-sz/2，终点为pos+sz/2+sz，用一个序列来表示
	xs = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);
	ys = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);
	
	%check for out-of-bounds coordinates, and set them to the values at
	%the borders
    %修正后的数值就是x、y坐标的范围
    %当超出下界，复制边界的值
	xs(xs < 1) = 1;
	ys(ys < 1) = 1;
    %当超出上届，复制边界的值
	xs(xs > size(im,2)) = size(im,2);
	ys(ys > size(im,1)) = size(im,1);
	
	%extract image
	out = im(ys, xs, :);%取出这些位置像素值

end

