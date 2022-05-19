%% 返回的参数约定：目标左上角坐标，目标大小，采用笛卡尔坐标系，也即与文件中的排序一致
function res = run_FCCF(seq, res_path, bSaveImage)
%% 读取传入的数据集参数
    %跟踪器采用的是MATLAB坐标系
    %把'base'工作区的变量subS赋值给seq，不写也行
    seq = evalin('base', 'subS');%在基本空间中插入subS

    %转为MATLAB坐标系
    target_sz = seq.init_rect(1,[4,3]);
    pos = seq.init_rect(1,[2,1]) + floor(target_sz/2);
    img_files = seq.s_frames;
    video_path = [];
    
%% 算法参数初始化 

	kernel_x1_type = 'gaussian'; 
    kernel_x2_type = 'gaussian';
	feature_x1_type = 'fhog'; 
    feature_x2_type = 'cn'; 
    
    show_visualization = 1; 

	%parameters according to the paper. at this point we can override
	%parameters based on the chosen kernel or feature type
    %定义结构体，存储数据
    %采用的核
	kernel_x1.type = kernel_x1_type;
	kernel_x2.type = kernel_x2_type;
    
	features_x1.gray = false;
	features_x1.fhog = false;
    features_x1.gfhog= false;
    features_x1.cn   = false;
	features_x1.dsst = false;
    
    features_x2.gray = false;
	features_x2.fhog = false;
    features_x2.gfhog = false;
    features_x2.cn   = false;
	features_x2.dsst = false;
   
%     output_sigma_factor = 1/16;%太小了，目标丢失，太大了也会不行
    %定义一个结构体，传输参数
    params.padding = 1.5;%三个滤波器都共用这个参数
    
    params.rho = 0.5;
    params.varepsilon=1e-5;
    params.translation_model_max_area = 1024;%位置滤波器的限制，设为512效果不错
    %位置滤波器
    params.lambda = 1e-4;
    params.output_sigma_factor = 0.1;%位置滤波器采用的标准差，创建高斯标签时所用
 
    %%%%%%%%%%%%%========设置核相关的参数，不同核具有不同的学习率======%%%%%%%%%%%%%%%%%%%
    [kernel_x1,features_x1] = set_kernel_and_features(kernel_x1,features_x1,feature_x1_type);
    [kernel_x2,features_x2] = set_kernel_and_features(kernel_x2,features_x2,feature_x2_type);
    
 
    %% 调用跟踪算法进行跟踪
    %输出MATLAB坐标系下的结果
    %call tracker function with all the relevant parameters
    [positions, time] = tracker(video_path, img_files, pos, target_sz,params,...
        kernel_x1,  kernel_x2,...
        features_x1,features_x2, show_visualization);
		
%%    %%%%%%%%%%%%% 增加 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if bSaveImage
    imwrite(frame2im(getframe(gcf)),[res_path num2str(frame) '.jpg']); 
end

%%  MATLAB坐标系转为笛卡尔坐标系
    rects = [positions(:,2) - target_sz(2)./2, positions(:,1) - target_sz(1)./2];%转到左上角坐标
    rects(:,3) = target_sz(2);
    rects(:,4) = target_sz(1);
    fps = numel(img_files)/time;%加上的，计算帧率

    res.type = 'rect';
    res.res = rects;
    res.fps = fps;

    assignin('base', 'res', res);
end
