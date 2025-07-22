path='./PUnet_output/';

files = dir(fullfile(path, 'PUnet*'));
fileNames = cell2mat({files.name}');
%Process first 10 files
for i=1:10
    fn=fileNames(i,:);
    om = loadbfn_lgz([path fn],[1200, 3600], 'int16');
    pred(:,:,i)=om/100;
    i
end

pred(pred<0)=NaN;

%Visualize first image
figure()
img=pred(:,:,1);
imagesc(img)
colormap(turbo)
colorbar()
imagesc(img,[0,8]);

%Visualize daily total
figure()
img=nansum(pred,3);
imagesc(img)
colormap(turbo)
colorbar()
imagesc(img,[0,40]);