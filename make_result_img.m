%% 加载数据
data_path = 'C:/Users/86192/Desktop/Graduate-Object/data/OTB/OTB100/test_8/img/';
result = load('FCSCF');
result = result.positions;%得到结果，MATLAB坐标系，前面是中心坐标
% 转为笛卡尔坐标系
result = result(:,[2,1,4,3]);%交互列

%% 读取文件名
%先尝试读取png图片
img_files = dir([data_path '*.png']);%仅仅遍历png文件
%如果不是png图片，尝试jpg图片
if isempty(img_files)
    img_files = dir([data_path '*.jpg']);
    assert(~isempty(img_files), 'No image files to load.')
end
% 图片命名为0001，可以用sort进行排序
img_files = sort({img_files.name});%只取文件名字，并排序

%% 创建视频文件
select_name = 'mytest-paper';
aviobj = VideoWriter([select_name '.mp4'],'MPEG-4');%创建mp4文件
% aviobj = VideoWriter([select_name],'Archival');%创建avi文件
% aviobj = VideoWriter(select_name);%创建avi文件
%设置帧率
aviobj.FrameRate = 30;%视频播放的速率
open(aviobj)

%% 遍历所有图片，生成视频
for i=1:numel(img_files)
    id = num2str(sprintf('%04d',i));%根据约好的格式生成名字
    temp_name = img_files{i};
    img_path = [data_path temp_name];%组合成完整的文件路劲
    img = imread(img_path);%读取图片
    imshow(img, 'Border','tight');%紧凑显示
    %显示第几帧
    text(30, 45, ['#' id], 'Color','y', 'FontWeight','bold', 'FontSize',48);
    %标注目标中心
    text(result(i,1), result(i,2), ...
        '+', 'Color','r', 'FontWeight','bold', 'FontSize',16,...
        'HorizontalAlignment','center','VerticalAlignment','middle');
    
    %画出目标框，要先把中心坐标转为左上角坐标
    rectangle('Position', [result(i,1)-result(i,3)/2, result(i,2)-result(i,4)/2,result(i,[3,4])],...%位置
                        'EdgeColor', [1,0,0], 'LineWidth', 4,...%框的颜色、大小
                        'LineStyle','-');%框的样式
    % 捕获图窗转为图片，写入视频文件中
    writeVideo(aviobj,frame2im(getframe(gcf)));
%     imwrite(frame2im(getframe(gcf)), ['./test_8/'  temp_name]);%保存图片
end
close(aviobj)