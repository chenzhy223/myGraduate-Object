% path = 'C:/Users/86192/Desktop/Graduate-Object/data/OTB/Benchmark/Basketball/img/';
%% 设置目标位置
clear;
base_path = 'C:/Users/86192/Desktop/Graduate-Object/Code-for/test-OTB/tracker_benchmark_v1.0/';
base_path2='tracker_benchmark_v1.0/tmp/imgs/CVPR13_OPE_FCSCF_4/';
select_name ='skating1';
path = [base_path base_path2 select_name '_1/'];
start_frame = 10;
end_frame = 400;

%% 读取文件名
%先尝试读取png图片
img_files = dir([path '*.png']);
%如果不是png图片，尝试jpg图片
if isempty(img_files)
    img_files = dir([path '*.jpg']);
    assert(~isempty(img_files), 'No image files to load.')
end

%% 对图片名字进行排序，转为数组进行
img_files = {img_files.name};%只取文件名字
file_end = img_files{1}(end-3:end);%文件名后缀
for i=1:numel(img_files)
    img_files{i} = str2double(img_files{i}(1:end-4));
end
img_files = sort(cell2mat(img_files));%排序

% img_files = sort({img_files.name});%只取文件名字，并排序，字符排，错误

%% 创建视频文件
% aviobj = VideoWriter([select_name '.mp4'],'MPEG-4');%创建mp4文件
% aviobj = VideoWriter([select_name],'Archival');%创建mp4文件
aviobj = VideoWriter([select_name]);%创建mp4文件
%设置帧率
aviobj.FrameRate = 30;%视频播放的速率
open(aviobj)
%我制作了由180张图片构成的视频
for i = start_frame:end_frame
%     frame_name = img_files{i};
%     frame_name = [num2str(img_files(i)) file_end];
    
    frame_name = [num2str(i) '.png'];%文件名
    
    frame = imread([path frame_name]);
    writeVideo(aviobj,frame);
end
close(aviobj)
