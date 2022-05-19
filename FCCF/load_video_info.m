function [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video)
%% 读取数据，MATLAB坐标系下的结果
% 输入参数：
%                 base_path：视频跟地址
%                 video：所选的视频分类
% 输出参数：
%                 img_files：所选视频中所含的图片，是一个元组
%                 pos：起始目标中心[y0,x0]
%                 target_sz：起始目标大小
%                 ground_truth：数据集中的ground_truth,MATLAB坐标系前面两个是中心坐标
%                 video_path：视频图片存放的路径
	%see if there's a suffix, specifying one of multiple targets, for
	%example the dot and number in 'Jogging.1' or 'Jogging.2'.
	if numel(video) >= 2 && video(end-1) == '.' && ~isnan(str2double(video(end)))
		suffix = video(end-1:end);  %remember the suffix
		video = video(1:end-2);  %remove it from the video name
	else
		suffix = '';
	end

	%full path to the video's files
	if base_path(end) ~= '/' && base_path(end) ~= '\'
		base_path(end+1) = '/';
	end
	video_path = [base_path video '/'];

	%try to load ground truth from text file (Benchmark's format)
	filename = [video_path 'groundtruth_rect' suffix '.txt'];
	f = fopen(filename);
	assert(f ~= -1, ['No initial position or ground truth to load ("' filename '").'])
	
	%the format is [x, y, width, height]，分别是左上角的坐标、目标框的大小
	try
		ground_truth = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false);  
	catch  % try different format (no commas)
		frewind(f);
		ground_truth = textscan(f, '%f %f %f %f');  
	end
	ground_truth = cat(2, ground_truth{:});%把元组变为数组
	fclose(f);
	
	%set initial position and size
	target_sz = [ground_truth(1,4), ground_truth(1,3)];%转MATLAB坐标系
    %起始帧的目标中心坐标
	pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);%转MATLAB坐标系
	
	if size(ground_truth,1) == 1
		%we have ground truth for the first frame only (initial position)
		ground_truth = [];
	else
		%store positions instead of boxes
        %将格式转化了，把[x, y, width, height]转为了目标中心坐标[y0,x0]
        %如果想要输出目标大小的话，还需要输出[x, y, width, height]中后面的两个维度信息
        %转MATLAB坐标系
        ground_truth = [ground_truth(:,[2,1]) + (ground_truth(:,[4,3]) - 1) / 2 , ground_truth(:,[4,3])];
% 		ground_truth = [ground_truth(:,[2,1])+ground_truth(:,[4,3])/2,ground_truth(:,[4,3])];%转为MATLAB坐标系
	end
	
	
	%from now on, work in the subfolder where all the images are
	video_path = [video_path 'img/'];%图片还要再进一个目标
	%这几个数据集需要特殊处理
	%for these sequences, we must limit ourselves to a range of frames.
	%for all others, we just load all png/jpg files in the folder.
	frames = {'David', 300, 770;
			  'Football1', 1, 74;
			  'Freeman3', 1, 460;
			  'Freeman4', 1, 283;
              'BlurCar1',247,988;
              'BlurCar3',3,359;
              'BlurCar4',18,397};
          
	%找到有无在限制名单里的，返回索引
	idx = find(strcmpi(video, frames(:,1)));
	
	if isempty(idx)
		%general case, just list all images
        %先尝试读取png图片
		img_files = dir([video_path '*.png']);
        %如果不是png图片，尝试jpg图片
		if isempty(img_files)
			img_files = dir([video_path '*.jpg']);
			assert(~isempty(img_files), 'No image files to load.')
		end
		img_files = sort({img_files.name});%只取文件名字，并排序
	else
        %如果是特殊的数据集，需要特殊处理才可以
		%list specified frames. try png first, then jpg.
        if exist(sprintf('%s%04i.png', video_path, frames{idx,2}), 'file')
            %组合名字，num2str(A,formatSpec) 将 formatSpec 指定的格式应用到 A 所有元素
            %'%04i.png'是用来控制格式的，不够位数的时候前面用0类填充
            %选出需要读取的数据，设定好了开头核结尾
            img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.png');
            
        elseif exist(sprintf('%s%04i.jpg', video_path, frames{idx,2}), 'file')
            img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.jpg');
            
        else
            error('No image files to load.')
        end
		img_files = cellstr(img_files);%转换为字符向量元胞数组
	end
	
end