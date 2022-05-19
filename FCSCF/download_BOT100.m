%下载OTB-100数据集
base_path = 'C:/Users/86192/Desktop/Graduate-Object/data/OTB/Benchmark2/';
CoreNum = 4;%并行核心数
%要下载的视频列表
videos = {'Bird2', 'BlurCar1', 'BlurCar3', 'BlurCar4', 'Board', 'Bolt2', ...
	'Boy', 'Car2', 'Car24', 'Coke', 'Coupon', 'Crossing', 'Dancer', ...
	'Dancer2', 'David2', 'David3', 'Dog', 'Dog1', 'Doll', 'FaceOcc1', ...
	'FaceOcc2', 'Fish', 'FleetFace', 'Football1', 'Freeman1', 'Freeman3', ...
	'Girl2', 'Gym', 'Human2', 'Human5', 'Human7', 'Human8', ...
	'Jogging', 'KiteSurf', 'Lemming', 'Man', 'Mhyang', ...
	'MountainBike', 'Rubik', 'Singer1', 'Skater', 'Skater2', 'Subway', 'Suv', ...
	'Tiger1', 'Toy', 'Trans', 'Twinnings', 'Vase', 'Dancer2'};

%判断文件是否存在
if ~exist(base_path, 'dir')  %如果不存在这个文件夹，就创建
	mkdir(base_path);
end
%如果不存在并行计算工具包，用普通for循环
if ~exist('parpool', 'file')
	%no parallel toolbox, use a simple 'for' to iterate
	disp('Downloading videos one by one, this may take a while.')
	disp(' ')
	for k = 1:numel(videos)
        download_url = ['http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/' videos{k} '.zip'];
        if ~exist([base_path videos{k}],"dir")
            disp(['Downloading and extracting ' videos{k} '...']);
		    unzip(download_url, base_path);
        else
            disp([videos{k} "is exist!"])
        end
	end
else
	%download all videos in parallel
	disp('Downloading videos in parallel, this may take a while.')
	disp(' ')
	if isempty(gcp('nocreate'))
		parpool(CoreNum);%开启并行
	end
	parfor k = 1:numel(videos)
		download_url = ['http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/' videos{k} '.zip'];
        %如果不存在，就创建下载
        if ~exist([base_path videos{k}],"dir")
            disp(['Downloading and extracting ' videos{k} '...']);
		    unzip(download_url, base_path);
        else
            disp([videos{k} "is exist!"])
        end
	end
    delete(gcp('nocreate'));%释放资源
end



