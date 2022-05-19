% 尺度采样，然后缩放到base_target_sz大小，然后提取特征，把特征拉伸成1维，最后输出大小：xx*17。
% 此函数的详细说明。
function out_feature = get_scale_subwindow(im, pos, base_target_sz, scaleFactors, scale_model_sz, cell_size)

nScales = length(scaleFactors);
for s = 1:nScales
    
    %%%%%%%%%%%%%%%%  第一步：取出缩放后感兴趣的区域图片  %%%%%%%%%%%%%
    patch_sz = floor(base_target_sz * scaleFactors(s));
    
    xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
    ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
    
    %check for out-of-bounds coordinates, and set them to the values at
    %the borders
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    
    %extract image
    im_patch = im(ys, xs, :);
    
    
    %%%%%%%%%%%%%%%%  第二步：对取出的图片进行缩放，缩放回scale_model_sz  %%%%%%%%%%%%%
    % resize image to model size
    %使用MATLAB自带的会慢很多
%     im_patch_resized = imresize(im_patch, scale_model_sz, 'bilinear');
    im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
%     im_patch_resized = imresize(im_patch, scale_model_sz);
    

    %%%%%%%%%%%%%%%%  第三步：提取特征图  %%%%%%%%%%%%%
    % extract scale features
    temp_hog = fhog(single(im_patch_resized), cell_size);
    
    if s == 1
        dim_scale = size(temp_hog,1)*size(temp_hog,2)*31;
        %提前分配空间
        out_feature = zeros(dim_scale, nScales, 'single');
    end
    
    out_feature(:,s) = reshape(temp_hog(:,:,1:31), dim_scale, 1);
end

end

%% 实测并没有加速，反而满了一点点
% function temp_feature = cell_feature(im, pos, base_target_sz, scaleFactor, scale_model_sz, cell_size)
% 
% %%%%%%%%%%%%%%%%  第一步：取出缩放后感兴趣的区域图片  %%%%%%%%%%%%%
%     patch_sz = floor(base_target_sz * scaleFactor);
%     
%     xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
%     ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
%     
%     %check for out-of-bounds coordinates, and set them to the values at
%     %the borders
%     xs(xs < 1) = 1;
%     ys(ys < 1) = 1;
%     xs(xs > size(im,2)) = size(im,2);
%     ys(ys > size(im,1)) = size(im,1);
%     
%     %extract image
%     im_patch = im(ys, xs, :);
%     
%     
%     %%%%%%%%%%%%%%%%  第二步：对取出的图片进行缩放，缩放回scale_model_sz  %%%%%%%%%%%%%
% 
%     im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
% 
%     
% 
%     %%%%%%%%%%%%%%%%  第三步：提取特征图  %%%%%%%%%%%%%
%     % extract scale features
%     temp_hog = fhog(single(im_patch_resized), cell_size);
%     temp_feature = reshape(temp_hog(:,:,1:31),...
%         size(temp_hog,1)*size(temp_hog,2)*31, 1);
% end
