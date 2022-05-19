function features = set_scale_feature(features,feature_type)
% 根据尺度滤波器选取的特征，设置feature参数。
% 根据输入的feature_type来确定所选的特征，然后对应设置好参数
    switch feature_type
	case 'gray'
		features.gray = true;
		features.cell_size = 1;
		
	case 'fhog'
		features.fhog = true;
		features.fhog_orientations = 9;%方向个数参数
        %这个参数不能设的太大，8不行，6一般般
		features.cell_size = 4;%一个Cell的大小，设大了速度会提高，但是精确度会下降，默认是4

    case 'gfhog'
		features.gfhog = true;
		features.fhog_orientations = 9;%方向个数参数
        %这个参数不能设的太大，8不行，6一般般
		features.cell_size = 4;%一个Cell的大小，设大了速度会提高，但是精确度会下降，默认是4
			
    case 'cn'
		features.cn = true;
        %这个参数不能设的太大，8不行，6一般般
		features.cell_size = 4;%一个Cell的大小，设大了速度会提高，但是精确度会下降，默认是4    
    case 'dsst'
		features.dsst = true;
        %这个参数不能设的太大，8不行，6一般般
		features.cell_size = 4;%dsst中默认采用1
        
    otherwise
		error('Unknown feature.')
    end
end
