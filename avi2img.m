%% 把视频拆成一帧帧的图片
fileName = 'test_8.mp4';  
obj = VideoReader(fileName); %读取视频
numFrames = obj.NumFrames;  % 读取视频的帧数  
for i = 1 : numFrames      
    frame = read(obj,i);                            % 读取每一帧      
    %imshow(frame);                                  %显示每一帧
    temp_name = num2str(sprintf('%04d',i));
    imwrite(frame,strcat('./test_8/',temp_name,'.png'),'png'); % 保存每一帧 
end
