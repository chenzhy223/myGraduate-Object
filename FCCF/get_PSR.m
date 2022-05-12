% 计算峰值旁瓣比PSR。
% 此函数的详细说明。
function PSR = get_PSR(Gaumap)
    max_Gaumap = max(Gaumap(:));%响应最大值
    
    %去除峰值，留下旁瓣
    Gaumap(Gaumap==max_Gaumap) = 0;
    mean_Gaumap = mean(Gaumap(:));%旁瓣均值
    %计算旁瓣标准差
    sigma = std2(Gaumap);
    
    PSR = (max_Gaumap-mean_Gaumap)/sigma;
end
