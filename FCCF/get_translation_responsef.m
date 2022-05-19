function responsef = get_translation_responsef(im, pos, window_sz, currentScaleFactor, cos_window1, features_x1, w2c,model_x1f,model_alpha1f,kernel_x1)
%% 本模块针对不使用降维的位置滤波器编写
%% 计算特征图，是get_translation_sample的实现内容
if isscalar(window_sz)  %square sub-window
    window_sz = [window_sz, window_sz];
end

%% 根据尺度滤波器计算得到的缩放因子currentScaleFactor，得到当前目标的大小
patch_sz = floor(window_sz * currentScaleFactor);

%make sure the size is not to small
if patch_sz(1) < 1
    patch_sz(1) = 2;
end
if patch_sz(2) < 1
    patch_sz(2) = 2;
end

xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);

%如果超出了边界，复制边界值
% check for z1-of-bounds coordinates, and set them to the values at
% the borders
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

% extract image
im_patch = im(ys, xs, :);

%% 调整图片的大小，虽然在图片中采样的大小是根据缩放因子来的，
% 但是因为位置滤波器的大小不变，所以要把采样得到的图片修改成于初始目标大小一致
%
% resize image to model size
im_patch = mexResize(im_patch, window_sz, 'auto');
% im_patch = imresize(im_patch, window_sz);%调用MATLAB自带的速度会慢很多，准确率高一点点

%% 计算得到特征图
% compute feature map
% z1 = get_gray_fhog(im_patch, 1);%采用灰度图的FHOG特征，DSST中Cell_size采用1
% z1 = fhog(single(im_patch),1);%采用FHOG特征，第二个参数设置Cell_size

cell_size = features_x1.cell_size;


%% 31维FHOG特征
if features_x1.fhog
    %调用Piotr's Toolbox里边的函数，求解FHOG特征
    z1 = single(fhog(single(im_patch), cell_size, features_x1.fhog_orientations));
    z1(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
end


%% 使用灰度图计算的31维FHOG特征
if features_x1.gfhog
    %调用Piotr's Toolbox里边的函数，求解FHOG特征
    if size(im_patch,3)>1
        im_patch = rgb2gray(im_patch);
    end
    z1 = fhog(single(im_patch), cell_size, features_x1.fhog_orientations);
    z1(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
end


%% gray特征
if features_x1.gray
    %gray-level (scalar feature)
    if size(im_patch,3)>1
        im_patch = rgb2gray(im_patch);
    end
    z1 = single(im_patch) / 255;
    z1 = single(z1 - mean(z1(:)));
end


%% 如果采用DSST中的特征，灰度+27维FHOG特征
if features_x1.dsst
    temp = fhog(single(im_patch), cell_size);%设置binsize=1，比较小
    z1 = zeros(size(temp, 1), size(temp, 2), 28, 'single');
    z1(:,:,2:28) = temp(:,:,1:27);
    im_gray = mexResize(im_patch, [size(temp,1),size(temp,2)], 'auto');
    % if grayscale image如果是灰度图
    if size(im_patch, 3) == 1
        z1(:,:,1) = single(im_gray)/255 - 0.5;
    else
        z1(:,:,1) = single(rgb2gray(im_gray))/255 - 0.5;%处理成灰度
    end
end


%% 如果采用CN特征
if features_x1.cn
    %返回11维度的颜色概率，也即cn特征
    %先把图像根据cell_size进行压缩，保证大小和fhog特征的大小一致，方便特征融合
    %     im_patch = imresize(im_patch,floor(window_sz./cell_size));
    %% 方法1：对图像进行缩放：
    %     im_patch = mexResize(im_patch, floor(window_sz./cell_size), 'auto');
    %%
    % 如果是灰度图，直接使用灰度信息也没差多少
    if size(im_patch,3)==1
        z1 = single(im_patch) / 255;
        z1 = single(z1 - mean(z1(:)));
    else
        %如果没有加载进映射矩阵，重新加载
        if isempty(w2c)
            data = load("w2c.mat");
            w2c = data.w2c;
            clear data;%释放变量
        end
        %         z1 = zeros(size(im_patch, 1), size(im_patch, 2), size(w2c,2), 'single');
        z1 = im2cn(single(im_patch), w2c, -2);%一定要转一下图片的数据类型
    end
    %% 方法2：利用积分图计算Cell平均：
    %%     需要注释掉93行的im_patch = mexResize(im_patch, floor(window_sz./cell_size), 'auto');
    w = cell_size;
    %compute the integral image计算积分图像
    iImage = integralVecImage(z1);%在左边和上边有一圈0补充，即大小都+1
    %要+1，是因为积分图会用0补充多一个像素点
    i1 = (w:w:size(z1,1)) + 1;
    i2 = (w:w:size(z1,2)) + 1;
    %利用积分图进行计算平均
    z1_sum = iImage(i1,i2,:) - iImage(i1,i2-w,:) - iImage(i1-w,i2,:) + iImage(i1-w,i2-w,:);
    z1 = z1_sum / (w*w) ;
end


%% 如果是fhogcn，即fhog+cn组合而成的特征，FHOG前27维+11维CN特征
if features_x1.fhogcn
    %% 方法1：对图像进行缩放：
    %     im_patch = mexResize(im_patch, floor(window_sz./cell_size), 'auto');
    %%
    temp_fhog = single(fhog(single(im_patch), cell_size, features_x1.fhog_orientations));
    % 如果是灰度图
    if size(im_patch,3)==1
        temp_cn = single(im_patch) / 255;
        temp_cn = single(temp_cn - mean(temp_cn(:)));
    else
        %如果没有加载进映射矩阵，重新加载
        if isempty(w2c)
            data = load("w2c.mat");
            w2c = data.w2c;
            clear data;%释放变量
        end
        %         z1 = zeros(size(im_patch, 1), size(im_patch, 2), size(w2c,2), 'single');
        temp_cn = im2cn(single(im_patch), w2c, -2);%一定要转一下图片的数据类型
    end
    %% 方法2：利用积分图计算Cell平均：
    %%     需要注释掉93行的im_patch = mexResize(im_patch, floor(window_sz./cell_size), 'auto');
    w = cell_size;
    %compute the integral image计算积分图像
    iImage = integralVecImage(temp_cn);%在左边和上边有一圈0补充，即大小都+1
    %要+1，是因为积分图会用0补充多一个像素点
    i1 = (w:w:size(temp_cn,1)) + 1;
    i2 = (w:w:size(temp_cn,2)) + 1;
    %利用积分图进行计算平均
    temp_cn_sum = iImage(i1,i2,:) - iImage(i1,i2-w,:) - iImage(i1-w,i2,:) + iImage(i1-w,i2-w,:);
    temp_cn = temp_cn_sum / (w*w) ;
    
    z1 = cat(3,temp_fhog(:,:,1:27),temp_cn);
end


%% 计算FHOG特征降维后的特征fhogpca
if features_x1.fhogpca
    %调用Piotr's Toolbox里边的函数，求解FHOG特征
    temp = fhog(single(im_patch), cell_size, features_x1.fhog_orientations);
    temp(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
    % 拉伸成2维矩阵
    reshape_temp = reshape(temp, size(temp,1)*size(temp,2), size(temp,3));
    % 降维:
    num_dimension = features_x1.num_dimension;
    [~, score] = pca(reshape_temp);
    % 重新拉回三维矩阵
    z1 = reshape(score(:,1:num_dimension),...
        size(temp,1),size(temp,2),num_dimension);
end



%% 计算CN特征降维后的特征cnpca
if features_x1.cnpca
    % 如果是灰度图
    if size(im_patch,3)==1
        temp_cn = single(im_patch) / 255;
        temp_cn = single(temp_cn - mean(temp_cn(:)));
        w = cell_size;
        %compute the integral image计算积分图像
        iImage = integralVecImage(temp_cn);%在左边和上边有一圈0补充，即大小都+1
        %要+1，是因为积分图会用0补充多一个像素点
        i1 = (w:w:size(temp_cn,1)) + 1;
        i2 = (w:w:size(temp_cn,2)) + 1;
        %利用积分图进行计算平均
        temp_cn_sum = iImage(i1,i2,:) - iImage(i1,i2-w,:) - iImage(i1-w,i2,:) + iImage(i1-w,i2-w,:);
        z1 = temp_cn_sum / (w*w) ;
    else
        %如果没有加载进映射矩阵，重新加载
        if isempty(w2c)
            data = load("w2c.mat");
            w2c = data.w2c;
            clear data;%释放变量
        end
        %         z1 = zeros(size(im_patch, 1), size(im_patch, 2), size(w2c,2), 'single');
        temp_cn = im2cn(single(im_patch), w2c, -2);%一定要转一下图片的数据类型
        w = cell_size;
        %compute the integral image计算积分图像
        iImage = integralVecImage(temp_cn);%是MATLAB自带的函数,在左边和上边有一圈0补充，即大小都+1
        %要+1，是因为积分图会用0补充多一个像素点
        i1 = (w:w:size(temp_cn,1)) + 1;
        i2 = (w:w:size(temp_cn,2)) + 1;
        %利用积分图进行计算平均
        temp_cn_sum = iImage(i1,i2,:) - iImage(i1,i2-w,:) - iImage(i1-w,i2,:) + iImage(i1-w,i2-w,:);
        temp_cn = temp_cn_sum / (w*w) ;
        
        % 拉伸成2维矩阵
        reshape_temp = reshape(temp_cn, size(temp_cn,1)*size(temp_cn,2), size(temp_cn,3));
        % 降维:
        num_dimension = features_x1.num_dimension;
        [~, score] = pca(reshape_temp);
        % 重新拉回三维矩阵
        z1 = reshape(score(:,1:num_dimension),...
            size(temp_cn,1),size(temp_cn,2),num_dimension);
    end
    
    
end



%% 计算fhogcn38维特征降维后的20维特征
if features_x1.fhogcnpca
    % 如果是灰度图
    if size(im_patch,3)==1
        temp_cn = single(im_patch) / 255;
        temp_cn = single(temp_cn - mean(temp_cn(:)));
        w = cell_size;
        %compute the integral image计算积分图像
        iImage = integralVecImage(temp_cn);%在左边和上边有一圈0补充，即大小都+1
        %要+1，是因为积分图会用0补充多一个像素点
        i1 = (w:w:size(temp_cn,1)) + 1;
        i2 = (w:w:size(temp_cn,2)) + 1;
        %利用积分图进行计算平均
        temp_cn_sum = iImage(i1,i2,:) - iImage(i1,i2-w,:) - iImage(i1-w,i2,:) + iImage(i1-w,i2-w,:);
        cnpca = temp_cn_sum / (w*w) ;
    else
        %如果没有加载进映射矩阵，重新加载
        if isempty(w2c)
            data = load("w2c.mat");
            w2c = data.w2c;
            clear data;%释放变量
        end
        %         z1 = zeros(size(im_patch, 1), size(im_patch, 2), size(w2c,2), 'single');
        temp_cn = im2cn(single(im_patch), w2c, -2);%一定要转一下图片的数据类型
        w = cell_size;
        %compute the integral image计算积分图像
        iImage = integralVecImage(temp_cn);%是MATLAB自带的函数,在左边和上边有一圈0补充，即大小都+1
        %要+1，是因为积分图会用0补充多一个像素点
        i1 = (w:w:size(temp_cn,1)) + 1;
        i2 = (w:w:size(temp_cn,2)) + 1;
        %利用积分图进行计算平均
        temp_cn_sum = iImage(i1,i2,:) - iImage(i1,i2-w,:) - iImage(i1-w,i2,:) + iImage(i1-w,i2-w,:);
        temp_cn = temp_cn_sum / (w*w) ;
        
        % 拉伸成2维矩阵
        reshape_temp = reshape(temp_cn, size(temp_cn,1)*size(temp_cn,2), size(temp_cn,3));
        % 降维:
        [~, score] = pca(reshape_temp);
        % 重新拉回三维矩阵
        cnpca = reshape(score(:,1:2),...
            size(temp_cn,1),size(temp_cn,2),2);
    end
    
    temp = fhog(single(im_patch), cell_size, features_x1.fhog_orientations);
    temp(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
    % 拉伸成2维矩阵
    reshape_temp = reshape(temp, size(temp,1)*size(temp,2), size(temp,3));
    % 降维:
    [~, score] = pca(reshape_temp);
    % 重新拉回三维矩阵
    fhogpca = reshape(score(:,1:18),...
        size(temp,1),size(temp,2),18);
    
    z1 = cat(3,fhogpca,cnpca);
    
    
    %     temp_fhog = single(fhog(single(im_patch), cell_size, features_x1.fhog_orientations));
    %
    %     % 如果是灰度图
    %     if size(im_patch,3)==1
    %         temp_cn = single(im_patch) / 255;
    %         temp_cn = single(temp_cn - mean(temp_cn(:)));
    %     else
    %         %如果没有加载进映射矩阵，重新加载
    %         if isempty(w2c)
    %             data = load("w2c.mat");
    %             w2c = data.w2c;
    %             clear data;%释放变量
    %         end
    % %         z1 = zeros(size(im_patch, 1), size(im_patch, 2), size(w2c,2), 'single');
    %         temp_cn = im2cn(single(im_patch), w2c, -2);%一定要转一下图片的数据类型
    %     end
    %     %% 方法2：利用积分图计算Cell平均：
    %     w = cell_size;
    %     %compute the integral image计算积分图像
    %     iImage = integralVecImage(temp_cn);%在左边和上边有一圈0补充，即大小都+1
    %     %要+1，是因为积分图会用0补充多一个像素点
    %     i1 = (w:w:size(temp_cn,1)) + 1;
    %     i2 = (w:w:size(temp_cn,2)) + 1;
    %     %利用积分图进行计算平均
    %     temp_cn_sum = iImage(i1,i2,:) - iImage(i1,i2-w,:) - iImage(i1-w,i2,:) + iImage(i1-w,i2-w,:);
    %     temp_cn = temp_cn_sum / (w*w) ;
    %
    %     %组合成FHOGCN特征，38维
    %     temp = cat(3,temp_fhog(:,:,1:27),temp_cn);
    %
    %     % 拉伸成2维矩阵
    %     reshape_temp = reshape(temp, size(temp,1)*size(temp,2), size(temp,3));
    %     % 降维:
    %     num_dimension = features_x1.num_dimension;
    %     [~, score] = pca(reshape_temp);
    %     % 重新拉回三维矩阵
    %     z1 = reshape(score(:,1:num_dimension),...
    %                 size(temp,1),size(temp,2),num_dimension);
    
end




%%
%统一数据格式为single
z1 = single(z1);

%process with cosine window if needed
if ~isempty(cos_window1)
    %用bsxfun函数进行指定的操作,可以防止内存超出
    z1 = bsxfun(@times, z1, cos_window1);
end


%% 计算剩下的
z1f = fft2(z1);
z1f = single(z1f);%转为单精度
%共用的代码
%calculate response of the classifier at all shifts
switch kernel_x1.type
    case 'gaussian'
        kz1f = gaussian_correlation(z1f, model_x1f, kernel_x1.sigma);
    case 'polynomial'
        kz1f = polynomial_correlation(z1f, model_x1f, kernel_x1.poly_a, kernel_x1.poly_b);
    case 'linear'
        kz1f = linear_correlation(z1f, model_x1f);
end
%这个响应最大值是分布在四个角落的，没有用shift经行移位
%             response1 = real(ifft2(model_alpha1f .* kz1f));  %equation for fast detection
%         response1f = model_alpha1f .* kz1f;
responsef = bsxfun(@times,model_alpha1f,kz1f);
end