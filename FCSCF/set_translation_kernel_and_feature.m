function [kernel,features] = set_translation_kernel_and_feature(kernel,features,feature_type)
% 根据选取的特征，设置kernel、feature参数。
% 根据输入的feature_type来确定所选的特征，然后对应设置好参数
switch feature_type
    case 'gray'
        %自适应线性插值因子，就是模型学习率
        kernel.interp_factor = 0.075;  %linear interpolation factor for adaptation
        %计算高斯核相关矩阵时的标准差
        kernel.sigma = 0.2;  %gaussian kernel bandwidth
        %设置多项式核参数，加法项核乘法项
        kernel.poly_a = 1;  %polynomial kernel additive term
        kernel.poly_b = 7;  %polynomial kernel exponent
        
        features.gray = true;
        features.cell_size = 1;
        
    case 'fhog'
        kernel.interp_factor = 0.04;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.fhog = true;
        features.fhog_orientations = 9;%方向个数参数
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;%一个Cell的大小，设大了速度会提高，但是精确度会下降，默认是4
        
    case 'gfhog'
        kernel.interp_factor = 0.04;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.gfhog = true;
        features.fhog_orientations = 9;%方向个数参数
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;%一个Cell的大小，设大了速度会提高，但是精确度会下降，默认是4
        
    case 'cn'
        kernel.interp_factor = 0.02;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.cn = true;
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;%一个Cell的大小，设大了速度会提高，但是精确度会下降，默认是4
    case 'dsst'
        kernel.interp_factor = 0.04;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.dsst = true;
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;%dsst中默认采用1
        
    case 'fhogcn'
        kernel.interp_factor = 0.02*2;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.fhogcn = true;
        features.fhog_orientations = 9;%方向个数参数
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;
        
    case 'fhogpca'
        kernel.interp_factor = 0.02*2;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.fhogpca = true;
        features.fhog_orientations = 9;%方向个数参数
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;
        features.num_dimension = 18;%降维后采用18维
        
    case 'cnpca'
        kernel.interp_factor = 0.02*2;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.cnpca = true;
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;
        features.num_dimension = 5;%降维后采用5维
        
    case 'fhogcnpca'
        kernel.interp_factor = 0.02*2;%学习率
        kernel.sigma = 0.5;
        
        kernel.poly_a = 1;
        kernel.poly_b = 9;
        
        features.fhogcnpca = true;
        features.fhog_orientations = 9;%方向个数参数
        %这个参数不能设的太大，8不行，6一般般
        features.cell_size = 4;
        features.num_dimension = 20;%降维后采用20维
        
    otherwise
        error('Unknown feature.')
end
%异常处理
assert(any(strcmp(kernel.type, {'linear', 'polynomial', 'gaussian'})), 'Unknown kernel.')
end
