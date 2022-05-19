% 输入参数：
%                 video_path：视频图片存放地址
%                 img_files：视频图片的文件名，是一个元组
%                 pos、 target_sz：初始目标的中心坐标、目标大小
%                 params：算法的主要参数
% kernel_x1,  kernel_x2：两个位置滤波器的参数
% features_x1,features_x2,features_s,：对应的特征参数
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
    features_x1,features_x2,features_s, show_visualization)
%% 解析参数
% 设725不错
translation_model_max_area = params.translation_model_max_area;%位置滤波器的限制，设为512效果不错
im = imread([video_path img_files{1}]);%读取一帧图像，用于判断大小、通道数
MAX_IMG_SIZE = params.MAX_IMG_SIZE;
RATIO = params.RATIO;
[im_heigt,im_width,~]=size(im);
if numel(im)>MAX_IMG_SIZE
    resize_image = true;
    pos = round(pos*RATIO);
    target_sz = round(target_sz*RATIO);
else
    resize_image = false;
end

%响应最小值阈值
G_RATIO_MIN_x = params.G_RATIO_MIN_x;
G_RATIO_MIN_s = params.G_RATIO_MIN_s;
%置信度最小值阈值
C_RATIO_MIN_x = params.C_RATIO_MIN_x;
C_RATIO_MIN_s = params.C_RATIO_MIN_s;

%解析出设置的参数
%共用的参数
padding = params.padding;

rho = params.rho;
varepsilon = params.varepsilon ;
%位置滤波器参数
output_sigma_factor = params.output_sigma_factor;%标签生成所用的标准差
lambda = params.lambda;
learning_rate_x1 = kernel_x1.interp_factor;
learning_rate_x2 = kernel_x2.interp_factor;

%尺度滤波器参数

% nScales                 = 15;%真实计算采用的尺度数量，17
% nScalesInterp           = 33;%插值后的尺度数量，33
% scale_step              = 1.05;%尺度变换的底数，尺度呈指数变换

nScales                 = params.nScales;%真实计算采用的尺度数量，17
nScalesInterp           = params.nScalesInterp;%插值后的尺度数量，33
scale_step              = params.scale_step;%尺度变换的底数，尺度呈指数变换
scale_sigma_factor      = params.scale_sigma_factor;
scale_model_factor      = params.scale_model_factor;%尺度滤波器的模型大小因子，调整滤波器的大小
scale_model_max_area    = params.scale_model_max_area;%尺度滤波器的最大值，512
scale_lambda            = params.scale_lambda;
scale_cell_size = features_s.cell_size;%尺度滤波器的cell_size大小
learning_rate_s = params.learning_rate_s;%尺度滤波器的学习率

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

%% 尺度滤波器：
if nScales > 0
    scale_sigma = nScalesInterp * scale_sigma_factor;
    %真正创建的尺度，共17个
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2)) * nScalesInterp/nScales;
    scale_exp_shift = circshift(scale_exp, [0 -floor((nScales-1)/2)]);%circshift(A,K) 循环将 A 中的元素平移 K 个位置
    %需要插值到的尺度，共33个
    interp_scale_exp = -floor((nScalesInterp-1)/2):ceil((nScalesInterp-1)/2);
    interp_scale_exp_shift = circshift(interp_scale_exp, [0 -floor((nScalesInterp-1)/2)]);
    
    scaleSizeFactors = scale_step .^ scale_exp;%17个尺度
    interpScaleFactors = scale_step .^ interp_scale_exp_shift;%要插值的尺度
    
    ys = exp(-0.5 * (scale_exp_shift.^2) /scale_sigma^2);%尺度标签，17个
    ysf = single(fft(ys));%行向量
    scale_window = single(hann(size(ysf,2)))';%行向量
    
    %make sure the scale model is not to large, to save computation time
    if scale_model_factor^2 * prod(target_sz) > scale_model_max_area
        %如果尺度滤波器大小太大了，缩放一下，计算缩放因子
        scale_model_factor = sqrt(scale_model_max_area/prod(target_sz));%尺度滤波器变换因子
    end
    
    %set the scale model size
    %得到尺度滤波器的大小，如果超过了限制，就设置为scale_model_max_area
    scale_model_sz = floor(target_sz * scale_model_factor);%设置尺度滤波器的大小
    

    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ window_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
   
end


%记录位置中心，目标尺度，共四个信息
positions = zeros(numel(img_files), 4);  %to calculate precision

%第一帧没有权重
lambda_fit_ratio = zeros(numel(img_files)-1,2);%存储位置滤波器的学习权重参数
G_RATIO_ALL = zeros(numel(img_files)-1,2);%
C_RATIO_ALL = zeros(numel(img_files)-1,2);%

num_frame = 0;%记录计算置信度的帧数
sum_Cx = 0;%计算位置滤波器的置信度和
sum_Cs = 0;%计算尺度滤波器的置信度和
sum_Gx = 0;%计算位置滤波器的最大响应和
sum_Gs = 0;%计算尺度滤波器的最大响应和

time = 0;  %to calculate FPS
%%%%%%%%%%%%%%%%%%%% 开始检测  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for frame = 1:numel(img_files)
    %load image
    im = imread([video_path img_files{frame}]);
    if resize_image%如果目标区域太大了，降采样减少图片大小
%         im = imresize(im, RATIO);
        im = mexResize(im, round([im_heigt,im_width]*RATIO), 'auto');
    end
    
    tic()%开始计时
    
    %% ********************************************************************** %%
    %%                 如果不是第一帧，就要开始预测了                            %%
    %% ********************************************************************** %%
    if frame > 1
        %%*******位置滤波器，并行计算响应
        %cell形式返回，这是不降维的调用方法
        responsef_all = cellfun(@get_translation_responsef,...
            {im,im},{pos,pos},{window_sz,window_sz},{currentScaleFactor,currentScaleFactor},...
            {cos_window1,cos_window2},{features_x1,features_x2},{w2c,w2c},...
            {model_x1f,model_x2f},{model_alpha1f,model_alpha2f},{kernel_x1,kernel_x2},...
            "UniformOutput",false);
        

        %%%%*******************两个滤波器响应融合模块***************************%%%%
        %%%%                采用频域融合                                       %%%%
        %%% ******************************************************************* %%
        %对响应进行插值：并行计算大小压缩
%         response_all = cellfun(@(x,sz) real(ifft2(resizeDFT2(x,sz),"symmetric")),...
%             responsef_all,{interp_sz,interp_sz},'UniformOutput',false);
        %% 奇怪了，居然这两个的效果是不一样的，可能是数据类型不同，下次试试全部换成double
        
        responsef_all = cellfun(@resizeDFT2,...
            responsef_all,{interp_sz,interp_sz},...
            'UniformOutput',false);
        %进行傅里叶反变换
        response_all = cellfun(@(x) real(ifft2(x,"symmetric")),...
            responsef_all,'UniformOutput',false);
        
        
        %在时域上计算APCE、PSR
        PSR_x = cellfun(@get_PSR, response_all,"UniformOutput",false);
        [~, APCE_x] = cellfun(@get_APCE, response_all,"UniformOutput",false);
        %计算置信度
        C_x = cellfun(@(APCE,PSR) rho*APCE+(1-rho)*PSR,APCE_x,PSR_x,"UniformOutput",false);
        C1=C_x{1};
        C2=C_x{2};
        %计算融合系数
        lambda_fit_1 = C1/(C1+C2+varepsilon);
        lambda_fit_2 = C2/(C1+C2+varepsilon);
        %在频域域上融合
        responsef = lambda_fit_1*responsef_all{1} + lambda_fit_2*responsef_all{2};
        %转到时域
        response = real(ifft2(responsef,"symmetric"));
        
        %% 计算融合后的响应置信度，用于更新
        %在时域上计算APCE、PSR
        PSR_x = get_PSR(response);
        [Gmax_x, APCE_x] = get_APCE(response);
        %计算置信度
        C_x = rho * APCE_x + (1-rho) * PSR_x;
        

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
        
        
        %% ********************************************************************** %%
        %%                      尺度滤波器代码                                     %%
        %% ********************************************************************** %%
        %scale search
        if nScales > 0
            %%%%%%%%%%%%%%%%%%%%%%%%%% 使用降维 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            xs_pca = get_scale_subwindow(im, pos, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz, scale_cell_size);
            xs = bsxfun(@times, scale_window, scale_basis * xs_pca);%17*17
            xsf = fft(xs,[],2);%计算每一行的傅里叶变换
            %%%%================采用MOSSE算法===================
            scale_responsef = sum(sf_num .* xsf, 1) ./ (sf_den + scale_lambda);%一维行向量
            %把17个响应值插值到33个尺度上
            interp_scale_response = ifft( resizeDFT(scale_responsef, nScalesInterp), 'symmetric');
            recovered_scale_index = find(interp_scale_response == max(interp_scale_response(:)), 1);%找到最佳尺度响应
        
            %set the scale
            currentScaleFactor = currentScaleFactor * interpScaleFactors(recovered_scale_index);%计算好缩放因子，以初始大小作为基准
            %adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            %在时域上计算APCE、PSR
            PSR_s = get_PSR(interp_scale_response);
            [Gmax_s, APCE_s] = get_APCE(interp_scale_response);
            %计算置信度
            C_s = rho*APCE_s + (1-rho)*PSR_s;
        end
    end

    %% ********************************************************************** %%
    %%                           第一帧开始的代码                              %%
    %% ********************************************************************** %%
    %% 位置滤波器
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
        % 计算历史总和
        sum_Gx = sum_Gx + Gmax_x;
        sum_Cx = sum_Cx + C_x;
        %计算历史平均值
        num_frame = num_frame + 1;
        mean_Gx = sum_Gx / num_frame;
        mean_Cx = sum_Cx / num_frame;
        %% 如果满足更新条件，则更新模型
        if (Gmax_x >= mean_Gx * G_RATIO_MIN_x) && (C_x >= mean_Cx*C_RATIO_MIN_x)
            %根据置信度调C整学习率
            learnRatio_x1 = learning_rate_x1*(1-1/C1);
            %用更新后的学习率更新模型
            model_alpha1f = (1 - learnRatio_x1) * model_alpha1f + learnRatio_x1 * alpha1f;
            model_x1f     = (1 - learnRatio_x1) * model_x1f     + learnRatio_x1 * x1f;
            
            %根据置信度调C整学习率
            learnRatio_x2 = learning_rate_x2*(1-1/C2);
            %用更新后的学习率更新模型
            model_alpha2f = (1 - learnRatio_x2) * model_alpha2f + learnRatio_x2 * alpha2f;
            model_x2f     = (1 - learnRatio_x2) * model_x2f     + learnRatio_x2 * x2f;
        end
    end

    
    %%%%%%%%%%%%%%%%%%%%%%%% 尺度滤波器相关  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Compute coefficents for the scale filter
    if nScales > 0
        %create a new feature projection matrix
        %%%=====================  采用降维方法  =====================
        %计算N*17的特征图
        xs_pca = get_scale_subwindow(im, pos, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz, scale_cell_size);
        %更新尺度滤波器参数
        %这个是尺度滤波器的特征参数，类似KCF中的α参数，目标的外观模型
        if frame == 1
            s_num = xs_pca;%N*17
        else
            %更新特征图，作为一个学习参数
            %        learnRatio_s = learning_rate_s;
            sum_Gs = sum_Gs + Gmax_s;
            sum_Cs = sum_Cs + C_s;
            
            mean_Gs = sum_Gs / num_frame;
            mean_Cs = sum_Cs / num_frame;
            if (Gmax_s >= mean_Gs * G_RATIO_MIN_s) && (C_s >= mean_Cs*C_RATIO_MIN_s)
                learnRatio_s = learning_rate_s*(1-1/C_s);
                s_num = (1 - learnRatio_s) * s_num + learnRatio_s * xs_pca;
            end
        end
        
        %这个是降维参数
        bigY = s_num;%N*17，一直学习的特征参数
        bigY_den = xs_pca;%N*17，当前帧的特征
        
        %%%%%%%%===========采用QR分解
        %bigY的维度大小为：N*17，所以scale_basis的大小为：N*17
        [scale_basis, ~] = qr(bigY, 0);
        [scale_basis_den, ~] = qr(bigY_den, 0);
        %重新计算得到的尺度滤波器降维矩阵，用学习到的特征参数计算，和位置滤波器的降维矩阵类似
        scale_basis = scale_basis';%转置，大小变为17*N
        %create the filter update coefficients
        %对特征进行降维，然后求解响应
        sf_proj = fft(bsxfun(@times, scale_window, scale_basis * s_num),[],2);%17*17
        
        %尺度滤波器的分子参数，学习参数，从s_num间接学习
        sf_num = bsxfun(@times,ysf,conj(sf_proj));%根据一直学习的特征，降维求解出的，17*17
        
        %得到的是提取的尺度特征，加窗后，得到的是当前帧的
        xs = bsxfun(@times, scale_window, scale_basis_den' * xs_pca);%17*17
        xsf = fft(xs,[],2);
        new_sf_den = sum(xsf .* conj(xsf),1);%对每一列求和，返回一个行向量
        
        %更新尺度滤波器的分母
        if frame == 1
            sf_den = new_sf_den;
        else
%             learnRatio_s = learning_rate_s;
            %如果响应的最值和置信度的和前一帧的比值大于阈值，则进行更新
            if (Gmax_s >= mean_Gs * G_RATIO_MIN_s) && (C_s >= mean_Cs*C_RATIO_MIN_s)
                sf_den = (1 - learnRatio_s) * sf_den + learnRatio_s * new_sf_den;
            end
        end
    end
    
    
    % calculate the new target size
    target_sz = floor(base_target_sz * currentScaleFactor);%更新尺度大小
    time = time + toc();
    
    
    %save position and timing
    %保存结果，MATLAB坐标系
    positions(frame,:) = [pos target_sz];%MATLAB坐标系
    if frame ~= 1
        lambda_fit_ratio(frame-1,:) = [lambda_fit_1,lambda_fit_2];%融合系数
        G_RATIO_ALL(frame-1,:) = [Gmax_x/mean_Gx,Gmax_s/mean_Gs];
        C_RATIO_ALL(frame-1,:) = [C_x/mean_Cx,C_s/mean_Cs];
    end
%     
    
    
    %visualization
    if show_visualization ==1
        %在figure中画方框，参数是笛卡尔坐标系的形式
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        %这样写能够快很多，不用每次都创建窗口
        if frame == 1
            figure("Name","FCSCF-Tracking");
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
        %pause
    end
    
    

%% 显示响应的图，修改成功，不过会有点慢
    if show_visualization == 2
        %在figure中画方框，参数是笛卡尔坐标系的形式
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        %这样写能够快很多，不用每次都创建窗口
        if frame == 1
%             figure;
            fig_handle = figure('Name', 'FCSCF-Tracking');%改
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

% figure("Name","位置滤波器加权系数")
% plot(lambda_fit_ratio(:,1), 'k-', 'LineWidth',2)
% hold on;%%在该图基础上继续画图
% plot(lambda_fit_ratio(:,2), 'g-', 'LineWidth',2)
% legend('位置滤波器1','位置滤波器2');%%用图例标识曲线
% xlabel('帧数'), ylabel('权重')
% xticks('auto');
% hold off

figure("Name","位置滤波器加权系数")
plot([1:size(lambda_fit_ratio(:,1),1)]+1,lambda_fit_ratio(:,1), 'k-', 'LineWidth',2)
hold on;%%在该图基础上继续画图
plot([1:size(lambda_fit_ratio(:,1),1)]+1,lambda_fit_ratio(:,2), 'g-', 'LineWidth',2)
legend('位置滤波器1','位置滤波器2');%%用图例标识曲线
xlabel('帧数'), ylabel('权重')
title('位置滤波器加权系数')
hold off
xticks('auto');
xlim([2,size(lambda_fit_ratio(:,1),1)+1])


figure("Name","滤波器响应最值比例")
plot([1:size(G_RATIO_ALL(:,1),1)]+1,G_RATIO_ALL(:,1), 'k-', 'LineWidth',2)
hold on;%%在该图基础上继续画图
plot([1:size(G_RATIO_ALL(:,1),1)]+1,G_RATIO_ALL(:,2), 'g-', 'LineWidth',2)
% hold on;%%在该图基础上继续画图
% plot(G_RATIO_ALL(:,3), 'b-', 'LineWidth',2)
legend('位置滤波器','尺度滤波器');%%用图例标识曲线
xlabel('帧数'), ylabel('响应最值比例')
xlim([2,size(G_RATIO_ALL(:,1),1)+1])
title('滤波器响应最值比例')
xticks('auto');
hold off


figure("Name","滤波器置信度比例")
plot([1:size(G_RATIO_ALL(:,1),1)]+1,C_RATIO_ALL(:,1), 'k-', 'LineWidth',2)
hold on;%%在该图基础上继续画图
plot([1:size(G_RATIO_ALL(:,1),1)]+1,C_RATIO_ALL(:,2), 'g-', 'LineWidth',2)
% hold on;%%在该图基础上继续画图
% plot(C_RATIO_ALL(:,3), 'b-', 'LineWidth',2)
legend('位置滤波器','尺度滤波器');%%用图例标识曲线
xlabel('帧数'), ylabel('置信度比例')
xlim([2,size(G_RATIO_ALL(:,1),1)+1])
hold off
xticks('auto');
title('滤波器置信度比例')




figure("Name","位置滤波器")
plot([1:size(G_RATIO_ALL(:,1),1)]+1,G_RATIO_ALL(:,1), 'k-', 'LineWidth',2)
hold on;%%在该图基础上继续画图
plot([1:size(C_RATIO_ALL(:,1),1)]+1,C_RATIO_ALL(:,1), 'g-', 'LineWidth',2)
% hold on;%%在该图基础上继续画图
% plot(G_RATIO_ALL(:,3), 'b-', 'LineWidth',2)
legend('响应最值比例','响应置信度比例');%%用图例标识曲线
xlabel('帧数'), ylabel('比例')
xlim([2,size(G_RATIO_ALL(:,1),1)+1])
hold off
xticks('auto');
title('位置滤波器')

figure("Name","尺度滤波器")
plot([1:size(G_RATIO_ALL(:,2),1)]+1,G_RATIO_ALL(:,2), 'k-', 'LineWidth',2)
hold on;%%在该图基础上继续画图
plot([1:size(C_RATIO_ALL(:,2),1)]+1,C_RATIO_ALL(:,2), 'g-', 'LineWidth',2)
% hold on;%%在该图基础上继续画图
% plot(G_RATIO_ALL(:,3), 'b-', 'LineWidth',2)
legend('响应最值比例','响应置信度比例');%%用图例标识曲线
xlabel('帧数'), ylabel('比例')
xlim([2,size(G_RATIO_ALL(:,2),1)+1])
hold off
xticks('auto');
title('尺度滤波器')


if resize_image
    positions = positions/RATIO;
end

end
