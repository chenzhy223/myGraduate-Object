function out=im2cn(im,w2c,color)
% input im should be DOUBLE !
% color=0 is color names out（默认）
% color=-1 is colored image with color names out
% color=1-11 is prob（概率） of colorname=color out;
% color=-2 return probabilities
% order of color names: black ,   blue   , brown       , grey       , green   , orange   , pink     , purple  , red     , white    , yellow
color_values = {  [0 0 0] , [0 0 1] , [.5 .4 .25] , [.5 .5 .5] , [0 1 0] ,...
    [1 .8 0] , [1 .5 1] , [1 0 1] , [1 0 0] , [1 1 1 ] , [ 1 1 0 ] };

if(nargin<3)
   color=0;
end

%如果不是单精度或者双精度，转为单精度
if ~isa(im,"single") && ~isa(im,"double")
    im = single(im);
end
RR=im(:,:,1);GG=im(:,:,2);BB=im(:,:,3);
%计算索引，不太清楚怎么来的
index_im = 1+floor(RR(:)/8)+32*floor(GG(:)/8)+32*32*floor(BB(:)/8);

if(color==0)
    %返回每行中最大的数,[最大元素,最大元素的索引]，都是一个列向量
   [~,w2cM]=max(w2c,[],2);  
   %取出每行元素最大值的索引（颜色分类），大小调整到与图像一致
   out=reshape(w2cM(index_im(:)),size(im,1),size(im,2));
end

if(color>0 && color < 12)
   w2cM=w2c(:,color);%取出对应颜色的矩阵
   out=reshape(w2cM(index_im(:)),size(im,1),size(im,2));
end

if(color==-1)
   out=im;
   [~,w2cM]=max(w2c,[],2);  
   out2=reshape(w2cM(index_im(:)),size(im,1),size(im,2));
         
   for jj=1:size(im,1)
        for ii=1:size(im,2) 
          out(jj,ii,:)=color_values{out2(jj,ii)}'*255;
        end
   end
end
%输出11维度的CN特征
if(color==-2)
   out=reshape(w2c(index_im,:),size(im,1),size(im,2),size(w2c,2));
end
