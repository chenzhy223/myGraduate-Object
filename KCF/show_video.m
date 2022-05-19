function update_visualization_func = show_video(img_files, video_path, resize_image)
%SHOW_VIDEO
%   Visualizes a tracker in an interactive（交互） figure, given a cell array of
%   image file names, their path, and whether to resize the images to
%   half size or not.
% 
%   This function returns an UPDATE_VISUALIZATION function handle, that
%   can be called with a frame number and a bounding box [x, y, width,
%   height], as soon as the results for a new frame have been calculated.
%   This way, your results are shown in real-time, but they are also
%   remembered so you can navigate and inspect the video afterwards.
%   Press 'Esc' to send a stop signal (returned by UPDATE_VISUALIZATION).
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
% INPUT：
%       img_files：图片的文件名字，是一个元组
%       video_path：图片文件夹路径
%       resize_image：是否降采样的标识
% OUTPUT：
%       update_visualization_func：函数update_visualization的句柄
%       

	%store one instance per frame
	num_frames = numel(img_files);
	boxes = cell(num_frames,1);

	%create window
	[fig_h, axes_h, unused, scroll] = videofig(num_frames, @redraw, [], [], @on_key_press);  %#ok, unused outputs
	%UserData，指定 UserData 对在 App 内共享数据很有用。
    set(fig_h, 'UserData','off', 'Name', ['Tracker - ' video_path])
	axis off;
	
	%image and rectangle handles start empty, they are initialized later
	im_h = [];%图像窗口句柄，全局变量
	rect_h = [];%全局变量
	
	update_visualization_func = @update_visualization;%返回的变量，函数句柄
	stop_tracker = false;
	

	function stop = update_visualization(frame, box)
        %box是边框的参数，起始坐标+目标大小，来确定框框的大小
		%store the tracker instance for one frame, and show it. returns
		%true if processing should stop (user pressed 'Esc').
		boxes{frame} = box;
		scroll(frame);
		stop = stop_tracker;
	end

	function redraw(frame)
		%render main image
		im = imread([video_path img_files{frame}]);%重新读取图片
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %把RGB格式的图片转换为灰度图，如果希望显示的不是灰度的，可以注释掉
% 		if size(im,3) > 1
% 			im = rgb2gray(im);
%         end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		if resize_image
			im = imresize(im, 0.5);
		end
		
		if isempty(im_h)  %create image
			im_h = imshow(im, 'Border','tight', 'InitialMag',200, 'Parent',axes_h);
		else  %just update it
			set(im_h, 'CData', im)%将图像添加到当前坐标区中而不替换现有绘图
		end
		
		%render target bounding box for this frame
		if isempty(rect_h)  %create it for the first time
            %创建带有尖角或圆角的矩形，返回句柄
            %设置框框的颜色EdgeColor
			rect_h = rectangle('Position',[0,0,1,1], 'EdgeColor','g', 'Parent',axes_h);
        end
        %画出边框
		if ~isempty(boxes{frame})
            %Position的格式： [left bottom width height] 
            %Visible，可见性，设置是否显示对象
			set(rect_h, 'Visible', 'on', 'Position', boxes{frame});
		else
			set(rect_h, 'Visible', 'off');
		end
	end

	function on_key_press(key)
		if strcmp(key, 'escape')  %stop on 'Esc'
			stop_tracker = true;
		end
	end

end

