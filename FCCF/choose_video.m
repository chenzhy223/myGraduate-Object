function video_name = choose_video(base_path)
	%process path to make sure it's uniform
    %针对PC平台，处理路劲格式
	if ispc(), base_path = strrep(base_path, '\', '/'); end%替换字符，保证路劲一致strrep(str,old,new)
	if base_path(end) ~= '/', base_path(end+1) = '/'; end%保证后面有 '/'
    
% 1.Jogging和Skating2有两个序列，在序列Human4中的groundtruth_rect.1.txt是空的（原版也是空的）
% 2.发现在Jogging序列下有个ground_rect.txt,删去即可
% 3.注意序列BlurCar1，BlurCar3，BlurCar4里面图片序号不是从1开始的
% ————————————————
% 版权声明：本文为CSDN博主「张小波」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
% 原文链接：https://blog.csdn.net/weixin_42495721/article/details/110425611

    double_path_names = {'Jogging','Skating2'};%一个文件夹里边有两个序列
%     tebie = {'BlurCar1','BlurCar3','BlurCar4','Human4'};
    tebie = {};%不用删除这些，在load_video_info文件中定义好起终帧就好了
	%list all sub-folders
	contents = dir(base_path);%返回文件信息的结构体
	names = {};%元组，存放所有视频的名字
	for k = 1:numel(contents)
		name = contents(k).name;
        %是文件夹并且不是两个特殊的文件夹并且不是那几个序列特殊的文件
        %用any函数把一个向量转为一个标量，只要一个为真，结果就为真。
		if isfolder([base_path name]) && ~any(strcmp(name, {'.', '..'})) && ~any(ismember(name,tebie))
			
            if ismember(name,double_path_names)%如果是那两个特殊的成员
                names{end+1} = [name '.1'];
                names{end+1} = [name '.2'];
            else
                names{end+1} = name;  %#ok
            end
		end
	end
	
	%no sub-folders found
	if isempty(names), video_name = []; return; end
	
	%choice GUI
    %创建选择对话框，设置SelectionMode，返回索引
	choice = listdlg('ListString',names, 'Name','Choose video', 'SelectionMode','single');
	
	if isempty(choice)  %user cancelled
		video_name = [];
	else
		video_name = names{choice};
	end
	
end