% KCF跟踪算法
% 可以实现KCF/DCF，这个函数用于参数设置、加载视频数据、估算位置，具体的实现代码在TRACKER文件中
% 如果不是如任何参数的话，将使用默认的参数，按下 'Esc' 可以退出程序。
% 多项式核中，如果选择高斯核'gaussian'或者多项式核'polynomial'，就是KCF算法，如果选择线性核'linear' ，就是DCF算法
% 第三行需要修改成数据集保存的地址。
% 
% 统一采用MATLAB坐标系====先y后x
% 2022-04-12
% 把CN改成了single后结果正确了，比较诡异
% MATLAB的“诡异之处”之single（单精度）的机器依赖_feynstain的博客-CSDN博客_matlab single
% single对机器存在依赖，不同机器可能结果不一样，这里采用single类型就够了，要求不高
% 
% Useful combinations:
% run_tracker  linear gaussian  fhog cn 
% run_tracker  linear linear  fhog cn 
% 
% 函数用于设置基本参数
function [precision, fps] = run_tracker(kernel_x1_type, kernel_x2_type, ...
    feature_x1_type,feature_x2_type, show_visualization, show_plots)
%% 数据集的位置
%     base_path = 'C:/Users/86192/Desktop/Graduate-Object/data/OTB/Benchmark/';
    base_path = 'C:/Users/86192/Desktop/Graduate-Object/data/OTB/OTB100/';
% base_path = 'C:/Users/86132/Desktop/GraduateObject/OTB100/OTB100/';
    
    %kernel_x1_type、feature_x1_type=====位置滤波器1的特征参数
    %kernel_x2_type、feature_x2_type=====位置滤波器2的特征参数
%% 选择特征，相关的文件有：
% set_translation_kernel_and_feature====设置对应的参数
% get_translation_param=================得到CF的参数
% get_translation_responsef=============得到CF的响应
%% 支持的特征类型有（共9种）:
% === fhog   ：31维的FHOG特征，如果是RGB图，选取梯度模值最大的那个
% === gfhog  ：31维的FHOG特征，如果是RGB图，先转为灰度图再计算
% === gray   ：1维灰度特征
% === dsst   ：1维灰度+fhog特征的前28维
% === cn     ：11维度的CN特征，由概率组成
% === fhogcn ：FHOG前27维+11维CN特征
% === fhogpca：31维的FHOG特征经过PCA降维处理的18维特征
% === cnpca  ：11维的CN特征经过PCA降维处理后的5维特征
% === fhogcnpca：FHOG前27维特征+11维CN特征构成的38维特征，经过PCA降维后的20维特征

	%default settings
	if nargin < 1, kernel_x1_type = 'gaussian'; end
    if nargin < 2, kernel_x2_type = 'gaussian'; end
	if nargin < 3, feature_x1_type = 'dsst'; end
    if nargin < 4, feature_x2_type = 'cn'; end
    
%% 设置是否可视化  
	if nargin < 5, show_visualization = 1; end%支持两种显示，1或者2
	if nargin < 6, show_plots = 1; end

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
    features_x1.fhogcn = false;
    features_x1.fhogpca = false;
    features_x1.cnpca = false;
    features_x1.fhogcnpca = false;
    
    features_x2.gray = false;
	features_x2.fhog = false;
    features_x2.gfhog = false;
    features_x2.cn   = false;
	features_x2.dsst = false;
    features_x2.fhogcn = false;
    features_x2.fhogpca = false;
    features_x2.cnpca = false;
    features_x2.fhogcnpca = false;
%     output_sigma_factor = 1/16;%太小了，目标丢失，太大了也会不行
    %定义一个结构体，传输参数
    params.padding = 1.5;%三个滤波器都共用这个参数
    
    params.rho = 0.8;
    params.varepsilon=1e-5;
    params.translation_model_max_area = 1296;%位置滤波器的限制，设为512效果不错
    %位置滤波器
    params.lambda = 1e-3;
    params.output_sigma_factor = 0.1;%位置滤波器采用的标准差，创建高斯标签时所用
  
    
    %%%%%%%%%%%%%========设置核相关的参数，不同核具有不同的学习率======%%%%%%%%%%%%%%%%%%%
    [kernel_x1,features_x1] = set_kernel_and_features(kernel_x1,features_x1,feature_x1_type);
    [kernel_x2,features_x2] = set_kernel_and_features(kernel_x2,features_x2,feature_x2_type);
    
    
	video = choose_video(base_path);%返回选择的图片名字，一个元组
    assert(~isempty(video),"没有选择有效的视频，错误！！");

    %get image file names, initial state, and ground truth for evaluation
    [img_files, pos, target_sz, ground_truth, video_path] = ...
        load_video_info(base_path, video);

%% 打开日志
close all
diary('runFCCFallresult.txt');%日志记录
    
%% 正常跟踪  ================================================================
    %% 调用跟踪算法进行跟踪
    %输出MATLAB坐标系下的结果
    %call tracker function with all the relevant parameters
    [positions, time] = tracker(video_path, img_files, pos, target_sz,params,...
        kernel_x1,  kernel_x2,...
        features_x1,features_x2, show_visualization);
		
%% 绘图，计算准确率
    %calculate and show precision plot, as well as frames-per-second
    %precisions是一个数组，在不同阈值下的准确率
    %其实已经得到了预测的位置坐标positions，真实的坐标为ground_truth,两者均为MATLAB坐标系，前两者为中心坐标
    %对比不同的算法效果的时候，可以用不同的positions，画出不同颜色的框
    %约定precisions==[位置准确率，大小准确率]
    precisions = precision_plot(positions, ground_truth, video, show_plots);
    
    fps = numel(img_files) / time;
    [distance_precision, overlap_precision, average_center_location_error,S] = ...
        compute_performance_measures(positions, ground_truth);
    fprintf('\n=================================\n');
    fprintf(['参数设置：'...
        '\n 位置滤波器1===： 特征类型： %s 学习率：%f'...
        '\n 位置滤波器2===： 特征类型： %s 学习率：%f'...
        '\n translation_model_max_area：%f  \n rho： %f \n output_sigma_factor：%f\n\n'],...
        feature_x1_type,kernel_x1.interp_factor,...
        feature_x2_type,kernel_x2.interp_factor,...
        params.translation_model_max_area, params.rho, params.output_sigma_factor);
    fprintf(['采用的核函数：===>>>  位置滤波器1： %s,  位置滤波器2： %s\n'],kernel_x1_type,kernel_x2_type);
%     fprintf(['%s --- FCCF with %s  %s: '...
%         '\n## Distance-Precision (20px):% 1.1f%% '...
%         '\n## Overlap_precision  (0.5): % 1.1f%% '...
%         '\n## CLE: %.2fpx'...
%         '\n## S:   %.2f%%'...
%         '\n## FPS: %4.2f'],...
%         video, feature_x1_type,feature_x2_type,distance_precision*100, overlap_precision*100, ...
%         average_center_location_error,S*100,fps)
%     fprintf('\n=================================\n');
    
    fprintf([' FCCF_%s with %s  %s: '...
        '\n## Distance-Precision (20px):% 1.1f '...
        '\n## Overlap_precision  (0.5): % 1.1f '...
        '\n## CLE: %.2f'...
        '\n## S:   %.2f'...
        '\n## FPS: %4.2f'],...
        video, feature_x1_type,feature_x2_type,distance_precision*100, overlap_precision*100, ...
        average_center_location_error,S*100,fps)
    fprintf('\n=================================\n');

    if nargout > 0
        %return precisions at a 20 pixels threshold
        precision = precisions(20);
    end
    
    
%% 扫描调参 ==============================================
%     scan_param = 'rho';
%     scan_step = 0.1;
%     start_pa = 0.1;
%     end_pa   = 1;
%     
      
%     
%     
%     %开启和关闭日志记录。当开启日志记录时，MATLAB® 
%     %从命令行窗口捕获输入的命令、键盘输入和文本输出。它将生成的日志以名为 
%     diary(['./scanparams/' scan_param '.txt']);%日志记录
%     fprintf("开始扫描参数%s :",scan_param);
%     for ii = start_pa:scan_step:end_pa
%         params.rho = ii;
%         [positions, time] = tracker(video_path, img_files, pos, target_sz,params,...
%             kernel_x1,  kernel_x2,...
%             features_x1,features_x2, show_visualization);
%         precisions = precision_plot(positions, ground_truth, video, show_plots);
%         result_temp(end+1,:) = [precisions(20,1),precisions(25,2)];
%         fps = numel(img_files) / time;
%         [distance_precision, overlap_precision, average_center_location_error,S] = ...
%             compute_performance_measures(positions, ground_truth);
%         fprintf(['参数设置：'...
%         '\n 位置滤波器1===： 特征类型： %s 学习率：%f'...
%         '\n 位置滤波器2===： 特征类型： %s 学习率：%f'...
%         '\n translation_model_max_area：%f  \n rho： %f \n output_sigma_factor：%f'],...
%         feature_x1_type,kernel_x1.interp_factor,...
%         feature_x2_type,kernel_x2.interp_factor,...
%         params.translation_model_max_area, params.rho, params.output_sigma_factor);
%         figure
%         plot([start_pa:scan_step:end_pa],result_temp(:,1), 'k-', 'LineWidth',2)
%         xlabel('rho'), ylabel('Distance-Precision')
%         hold on
%         plot([start_pa:scan_step:end_pa],result_temp(:,2), 'g-', 'LineWidth',2)
%         xlabel('rho'), ylabel('Overlap-Precision')
%         fprintf('=================================');
%         hold off

%         fprintf(['%s --- FCCF with %s  %s: '...
%             '\n## Distance-Precision (20px):% 1.1f%% '...
%             '\n## Overlap_precision  (0.5): % 1.1f%% '...
%             '\n## CLE: %.2fpx'...
%             '\n## S:   %.2f%%'...
%             '\n## FPS: %4.2f\n'],...
%             feature_x1_type,feature_x2_type,video, distance_precision*100, overlap_precision*100, ...
%             average_center_location_error,S*100,fps)
%         fprintf('\n\n');
%     end



end
