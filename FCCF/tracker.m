% KCF/DCF的跟踪算法的核心模块，调用其他函数来实现功能。
% 使用FHOG特征+线性核，就是DCF，速度会快一点，但是效果会差一点点
% 使用FHOG特征+高斯核，就是KCF，速度会慢一点点，但是效果会提高一些。
% 这个函数通过RUN_TRACKER来调用，在RUN_TRACKER中加载、设置了一些基本参数。
% 运行流程：
%         先进行位置滤波器，得到目标的中心位置，然后进行尺度滤波，得到目标的尺度变化
% 
% 输入参数：
%                 video_path：视频图片存放地址
%                 img_files：视频图片的文件名，是一个元组
%                 pos、 target_sz：初始目标的中心坐标、目标大小
%                 params：一些参数
% kernel_x1,  kernel_x2,  kernel_s：两个位置滤波器、一个尺度滤波器的参数
% features_x1,features_x2,features_s,：对应的特征参数
% 
%                 kernel：使用的核函数类型，支持三种核函数，是一个结构体，设置了核函数相关的参数
%                 lambda：正则化惩罚系数
%                 output_sigma_factor：高斯标签的标准差
%                 interp_factor：学习率
%                 cell_size：FHOG特征的Cell的大小，就是论文中的binSize
%                 features：是一个结构体，设置了特征图的一些基础参数
%                 show_visualization：是否实时观察结果
% 输出参数：
%                 positions：预测得到的目标中心位置
%                 time：真实计算所用的时间，不包括显示等操作的时间
% 
% 修改后的positions是：[目标中心坐标，目标大小]共四个维度，注意，这是在MATLAB坐标中，也即先y后x，先height后width
% 
% 采用cell，可以加快计算速度
% 采样下采样技术

function [positions, time] = tracker(video_path, img_files, pos, target_sz,params,...
    kernel_x1,  kernel_x2,...
    features_x1,features_x2, show_visualization)
%% 解析出设置的参数
translation_model_max_area = params.translation_model_max_area;%位置滤波器的限制，设为512效果不错

%共用的参数
padding = params.padding;
rho = params.rho;
varepsilon = params.varepsilon ;
%位置滤波器参数
output_sigma_factor = params.output_sigma_factor;%标签生成所用的标准差
lambda = params.lambda;
learning_rate_x1 = kernel_x1.interp_factor;
learning_rate_x2 = kernel_x2.interp_factor;


%加载映射矩阵
data = load("w2c.mat");
w2c = data.w2c;
clear data;%释放变量

% %加载10维的变换矩阵
% data = load("CNnorm.mat");
% w2c = data.CNnorm;
% clear data;%释放变量

%如果对角大小超出了阈值，减半
if prod(target_sz) > translation_model_max_area
    %如果超过了位置滤波器最大值的限制，计算缩放因子
    currentScaleFactor = sqrt(prod(target_sz) / translation_model_max_area);
else
    currentScaleFactor = 1.0;
end

%如果是init_target_sz超过了大小限制，这里就设置为translation_model_max_area
%设置初始目标的大小
base_target_sz = target_sz / currentScaleFactor;

%%%%%%%%%%%%%%    位置滤波器，设大小一样     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 默认两个位置滤波器的部分参数一致
%设置位置滤波器的大小
%window size, taking padding into account
window_sz = floor( base_target_sz * (1 + padding ));%取的图像框窗口大小
cell_size1 = features_x1.cell_size;
cell_size2 = features_x2.cell_size;
assert(cell_size1==cell_size2,'两个滤波器大小不一样，请查看Cell参数！');%当条件错误时，错发错误

%这个值不能乱改，需要和FHOG特征的cell_size保持一致
featureRatio = cell_size1; %特征比例，用于减少参数

%对目标区域与Cell_size为大小提取特征，缩小后的特征图大小，即实际滤波器大小
use_sz = floor(window_sz/featureRatio);

%%==位置滤波器1：
%由目标大小下采样featureRatio后的大小计算σ %创建高斯分布的标签，经过featureRatio降采样
output_sigma1 = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;

%%创建标签，与fDSST一样的
y1f = fft2(gaussian_shaped_labels(output_sigma1, use_sz));%调用函数，后期优化可去除加快速度
y1f = single(y1f);
cos_window1 = hann(size(y1f,1)) * hann(size(y1f,2))';	%最高点在中心
cos_window1 = single(cos_window1);

%%==位置滤波器2：这两个参数与位置滤波器1一致
y2f = y1f;
cos_window2 = cos_window1;

%%%对滤波器响应进行Cell_size插值，插回原图大小，提高精度
interp_sz = use_sz * featureRatio;%插值后的大小

%记录位置中心，目标尺度，共四个信息
positions = zeros(numel(img_files), 4);  %to calculate precision
lambda_fit_ratio = zeros(numel(img_files)-1,2);%存储位置滤波器的学习权重参数

time = 0;  %to calculate FPS
%%%%%%%%%%%%%%%%%%%% 开始检测  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for frame = 1:numel(img_files)
    %load image
    im = imread([video_path img_files{frame}]);

    tic()%开始计时
%% 如果不是第一帧，就要开始预测了
    if frame > 1
        %%*******位置滤波器：
        %cell形式返回，这是不降维的调用方法
        responsef_all = cellfun(@get_translation_responsef,...
            {im,im},{pos,pos},{window_sz,window_sz},{currentScaleFactor,currentScaleFactor},...
            {cos_window1,cos_window2},{features_x1,features_x2},{w2c,w2c},...
            {model_x1f,model_x2f},{model_alpha1f,model_alpha2f},{kernel_x1,kernel_x2},...
            "UniformOutput",false);
        
%% %%%%*******************两个滤波器响应融合模块*******************************%%%%
        %%融合可以采取时域、频域两种方式融合
        
        %对响应进行插值：
        %并行计算大小压缩
        responsef_all = cellfun(@resizeDFT2,...
            responsef_all,{interp_sz,interp_sz},...
            'UniformOutput',false);
        %进行傅里叶反变换
        response_all = cellfun(@(x) real(ifft2(x,"symmetric")),...
            responsef_all,'UniformOutput',false);

        %在时域上计算APCE、PSR
        PSR = cellfun(@get_PSR, response_all,"UniformOutput",false);
        [~, APCE] = cellfun(@get_APCE, response_all,"UniformOutput",false);
        %计算置信度
        C = cellfun(@(APCE,PSR) rho*APCE+(1-rho)*PSR,APCE,PSR,"UniformOutput",false);
        C1=C{1};C2=C{2};
        %计算融合系数
        lambda_fit_1 = C1/(C1+C2+varepsilon);
        lambda_fit_2 = C2/(C1+C2+varepsilon);
        
        response1f = responsef_all{1};
        response2f = responsef_all{2};

        %在频域域上融合
        responsef = lambda_fit_1*response1f + lambda_fit_2*response2f;
        %转到时域
        response = real(ifft2(responsef,"symmetric"));


        %%*******************根据融合结果，求出目标的中心位置**********%%%%%%%%%%
        % 目标位置处于最大响应。 我们必须考虑到，如果目标不移动，峰值将出现在左上角，
        %而不是在中心（这在本文中讨论）。这些响应周期性地绕过。
        [vert_delta, horiz_delta] = find(response == max(response(:)), 1);%找到响应最大的位置
        %根据极大值在矩阵中的位置，求出当前帧的目标中心的预测值即找到响应最大的位置
        %find(X,k)，返回X中第k个非零元素的行列位置，[vert_delta, horiz_delta]行列位置
        %在核相关滤波的目标跟踪中，最大响应值越大，其周围的扩散度越小，该最大响应
        %值对应的位置为目标中心位置的置信度越高。
        
        %如果环绕到了负半空间，调整一下
        if vert_delta > interp_sz(1) / 2 %wrap around to negative half-space of vertical axis
            vert_delta = vert_delta - interp_sz(1);
        end
        if horiz_delta > interp_sz(2) / 2  %same for horizontal axis
            horiz_delta = horiz_delta - interp_sz(2);
        end
        
        %计算得到的位置，因为每一个cell都是不重复的，后面跟着有给缩放因子
        pos = pos + [vert_delta - 1, horiz_delta - 1] * currentScaleFactor;

    end
    

 %% %%%%%%%%%%%%%%%%%%%%%%%%%%% 第一帧开始运行 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [xf_all,alphaf_all] = cellfun(@get_translation_param,...
        {im,im},{pos,pos},{window_sz,window_sz},{currentScaleFactor,currentScaleFactor},...
        {cos_window1,cos_window2},{features_x1,features_x2},{w2c,w2c},...
        {kernel_x1,kernel_x2},{y1f,y2f},{lambda,lambda},"UniformOutput",false);

    x1f=xf_all{1};
    x2f=xf_all{2};
    alpha1f=alphaf_all{1};
    alpha2f=alphaf_all{2};



    %%****************************************************************
    %%**                 更新参数                    ******************
    %%%%%%%%%%       降维需要进行代码修改：      %%%%%%%%%%%%%%%%%%%%%%%
    %%    如果采用了数据降维，则不用更新model_alpha1f、model_alpha2f   %%
    %%****************************************************************
    if frame == 1  %first frame, train with a single image
        %位置滤波器1：
        model_alpha1f = alpha1f;%降维的话也需要这个
        model_x1f = x1f;
       
        %位置滤波器2：
        model_alpha2f = alpha2f;%降维的话也需要这个
        model_x2f = x2f;
    else
        %调整模型参数，interp_factor是学习率
        %subsequent frames, interpolate model
        %根据置信度调C整学习率
        learnRatio_x1 = learning_rate_x1*(1-1/C1);
        learnRatio_x2 = learning_rate_x2*(1-1/C2);
        
%         learnRatio_x1 = learning_rate_x1;
%         learnRatio_x2 = learning_rate_x2;
        
        %用更新后的学习率更新模型
        model_alpha1f = (1 - learnRatio_x1) * model_alpha1f + learnRatio_x1 * alpha1f;%降维不需要这个
        model_x1f     = (1 - learnRatio_x1) * model_x1f     + learnRatio_x1 * x1f;

        model_alpha2f = (1 - learnRatio_x2) * model_alpha2f + learnRatio_x2 * alpha2f;%降维不需要这个
        model_x2f     = (1 - learnRatio_x2) * model_x2f     + learnRatio_x2 * x2f;

    end
    
    % calculate the new target size
    target_sz = floor(base_target_sz * currentScaleFactor);%更新尺度大小
    %save position and timing
%% %保存结果，MATLAB坐标系
    positions(frame,:) = [pos target_sz];
    time = time + toc();
    
    if frame ~= 1
        lambda_fit_ratio(frame-1,:) = [lambda_fit_1,lambda_fit_2];%融合系数
    end
%% %visualization
    if show_visualization == 1
        %在figure中画方框，参数是笛卡尔坐标系的形式
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        %这样写能够快很多，不用每次都创建窗口
        if frame == 1
%             figure;
            figure('Name', 'FCCF-Tracking');%改
            %'Border'设置为'tight'，不留空隙
            % 'InitialMagnification'设置图像显示的初始放大倍率
            im_handle = imshow(im, 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            
            %rectangle('Position',pos) 在二维坐标中创建一个矩形。将 pos 指定为 [x y w h] 形式的四元素向量（以数据单位表示）
            rect_handle = rectangle('Position',rect_position_vis, 'EdgeColor','g','LineWidth',2);%画图像框
            text_handle = text(10, 10, int2str(frame));%显示帧数
            set(text_handle, 'color', [0 1 1]);

        else
            try
               
                set(im_handle, 'CData', im)%更新图片
                set(rect_handle, 'Position', rect_position_vis)%更新图像框
                set(text_handle, 'string', int2str(frame));%更新显示的帧数

            catch
                return
            end
        end
        drawnow
%         pause(0.05);%限制在20帧
    end
    

%% 显示响应的图，修改成功，不过会有点慢
    if show_visualization == 2
        %在figure中画方框，参数是笛卡尔坐标系的形式
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        %这样写能够快很多，不用每次都创建窗口
        if frame == 1
%             figure;
            fig_handle = figure('Name', 'FCCF-Tracking');%改
            imagesc(im);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);%画图像框
            text_handle = text(10, 10, int2str(frame));%显示帧数
            set(text_handle, 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])

        else
            try
               
                xs = [pos(2)-target_sz(2)/2, pos(2)+target_sz(2)/2];
                ys = [pos(1)-target_sz(1)/2, pos(1)+target_sz(1)/2];
                sampled_scores_display = ifftshift(response);
                
                figure(fig_handle);
%                 set(fig_handle, 'Position', [100, 100, 100+size(im,2), 100+size(im,1)]);
                imagesc(im);
                hold on;
                %画出响应图，这个可以参考一下
                resp_handle = imagesc(xs, ys, sampled_scores_display); colormap hsv;
                alpha(resp_handle, 0.5);
                rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
%                 set(rect_handle, 'Position', rect_position_vis)%更新图像框
                text(10, 10, int2str(frame), 'color', [0 1 1]);
%                 set(text_handle, 'string', int2str(frame));%更新显示的帧数
                hold off;
                
            catch
                return
            end
        end
        drawnow
%         pause(0.05);%限制在20帧
    end
 
end

%% 可视化加权系数
figure("Name","位置滤波器加权系数")
plot([1:size(lambda_fit_ratio(:,1),1)]+1,lambda_fit_ratio(:,1), 'k-', 'LineWidth',2)

hold on;%%在该图基础上继续画图
plot([1:size(lambda_fit_ratio(:,1),1)]+1,lambda_fit_ratio(:,2), 'g-', 'LineWidth',2)
legend('位置滤波器1','位置滤波器2');%%用图例标识曲线
xlabel('帧数'), ylabel('权重')
hold off
title('位置滤波器加权系数')
xticks('auto');
xlim([2,size(lambda_fit_ratio(:,1),1)+1])

end
