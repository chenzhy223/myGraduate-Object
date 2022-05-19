function [positions, time] = tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)
%TRACKER Kernelized/Dual Correlation Filter (KCF/DCF) tracking.
%   This function implements the pipeline（通道） for tracking with the KCF (by
%   choosing a non-linear kernel) and DCF (by choosing a linear kernel).
%
%   It is meant to be called by the interface function RUN_TRACKER, which
%   sets up the parameters and loads the video information.
%
%   Parameters:
%     VIDEO_PATH is the location of the image files (must end with a slash
%      '/' or '\').后面修改了了一下，没有了这个条件的限制
%     IMG_FILES is a cell array of image file names.格式是一个元组
%     POS and TARGET_SZ are the initial position and size of the target
%      (both in format [rows, columns]).
%     PADDING is the additional tracked region, for context, relative to 
%      the target size.
%     KERNEL is a struct describing the kernel. The field TYPE must be one
%      of 'gaussian', 'polynomial' or 'linear'. The optional fields SIGMA,
%      POLY_A and POLY_B are the parameters for the Gaussian and Polynomial
%      kernels.仅支持三种核函数
%     OUTPUT_SIGMA_FACTOR is the spatial bandwidth of the regression
%      target, relative to the target size.
%     INTERP_FACTOR is the adaptation rate of the tracker.
%     CELL_SIZE is the number of pixels per cell (must be 1 if using raw
%      pixels).
%     FEATURES is a struct describing the used features (see GET_FEATURES).
%     SHOW_VISUALIZATION will show an interactive video if set to true.
%
%   Outputs:
%    POSITIONS is an Nx2 matrix of target positions over time (in the
%     format [rows, columns]).是一个二维矩阵
%    TIME is the tracker execution time, without video loading/rendering.
%
%   Joao F. Henriques, 2014
    %如果video_path结尾没有斜杠，则加上去
    if video_path(end) ~= '/' && video_path(end) ~= '\'
		video_path(end+1) = '/';
	end
    
	%if the target is large, lower the resolution, we don't need that much
	%detail   prod计算乘积
	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image%如果对角大小超出了阈值，减半
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
	end


	%window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));%固定大小
	
% 	%we could choose a size that is a power of two, for better FFT
% 	%performance. in practice it is slower, due to the larger window size.
% 	window_sz = 2 .^ nextpow2(window_sz);

	
	%create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
    %加窗处理
	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	%最高点在中心
	
	%调用了这函数，返回的是函数句柄
	if show_visualization  %create video interface
		update_visualization = show_video(img_files, video_path, resize_image);
	end
	
	
	%note: variables ending with 'f' are in the Fourier domain.

	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 4);  %to calculate precision

	for frame = 1:numel(img_files)
		%load image
		im = imread([video_path img_files{frame}]);
        %不转灰度图准确率还会上升一点点
		if size(im,3) > 1
			im = rgb2gray(im);%针对RGB图像，直接二值化，还没有转为double类型
		end
		if resize_image%如果目标区域太大了，降采样减少图片大小
			im = imresize(im, 0.5);
		end

		tic()%开始计时
        %如果不是第一帧，就要开始预测了
        if frame > 1
			%obtain a subwindow for detection at the position from last
			%frame, and convert to Fourier domain (its size is unchanged)
			patch = get_subwindow(im, pos, window_sz);%这个部分选取候选区域作为输入z
			zf = fft2(get_features(patch, features, cell_size, cos_window));
			
			%calculate response of the classifier at all shifts
            switch kernel.type
                case 'gaussian'
                    kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                case 'polynomial'
                    kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
                case 'linear'
                    kzf = linear_correlation(zf, model_xf);
            end
            %这个响应最大值是分布在四个角落的，没有用shift经行移位
            response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
            
            %target location is at the maximum response. we must take into
            %account the fact that, if the target doesn't move, the peak
            %will appear at the top-left corner, not at the center (this is
            %discussed in the paper). the responses wrap around cyclically.
            [vert_delta, horiz_delta] = find(response == max(response(:)), 1);%找到响应最大的位置
            %如果环绕到了负半空间，调整一下
            if vert_delta > size(zf,1) / 2 %wrap around to negative half-space of vertical axis
                vert_delta = vert_delta - size(zf,1);
            end
            if horiz_delta > size(zf,2) / 2  %same for horizontal axis
                horiz_delta = horiz_delta - size(zf,2);
            end
            pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];%计算得到的位置，因为每一个cell都是不重复的
        end

		%obtain a subwindow for training at newly estimated target position
		patch = get_subwindow(im, pos, window_sz);%如果不是第一帧，就用预测得到的结果作为坐标中心，取出感兴趣的图像块
		xf = fft2(get_features(patch, features, cell_size, cos_window));

		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
        %计算核矩阵Kxx，并FFT（即计算核函数，采用三种核）
		switch kernel.type
		case 'gaussian'
			kf = gaussian_correlation(xf, xf, kernel.sigma);
		case 'polynomial'
			kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
		case 'linear'
			kf = linear_correlation(xf, xf);
		end
		alphaf = yf ./ (kf + lambda);   %equation for fast training

		if frame == 1  %first frame, train with a single image
			model_alphaf = alphaf;
			model_xf = xf;
        else
            %调整模型参数，interp_factor是学习率
			%subsequent frames, interpolate model
			model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
			model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
		end

		%save position and timing
		positions(frame,:) = [pos,target_sz];
		time = time + toc();

		%visualization
		if show_visualization
			box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
			stop = update_visualization(frame, box);%前边调用了show_video，返回了函数句柄
			if stop, break, end  %user pressed Esc, stop early
			%更新图窗并处理任何挂起的回调。
            %如果修改图形对象并且需要在屏幕上立即查看这次更新，请使用该命令。
			drawnow
% 			pause(0.05)  %uncomment to run slower
		end
		
	end

	if resize_image
		positions = positions * 2;
	end
end

