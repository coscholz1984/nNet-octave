function hFigure = imageData (X, nImages, vSize)
%IMAGEDATA Plot a set of nImages from a dataset X, where each image has size vSize.
%
%   hFigure = IMAGEDATA(X, nImages, vSize) plots a set of nImages from a dataset X.
%   The images are drawn in a square assembly, with empty spaces if nImages is not 
%   a square number. All images are separated by a border of color [1 1 1].
%
%   Input:
%       X: Matrix of data to image with dimension M x (vSize(1) * vSize(2)).
%       nImages: Number of images to display. If nImages is smaller than M, 
%                all M images will be displayed.
%       vSize: 2x1 vector containing the dimensions of the images to display.
%
%   Output:
%       hFigure: Handle to the drawn figure.
%
% Created: 2020-03-25

if nImages > size(X,1)
  disp('Sample not large enough, display all images.')
  nImages = size(X,1);
end

if nImages > 500
  disp('Many images to display. Consider smaller number to display to avoid slowdown.');
end

hFigure = figure;

% Calculate dimensions of 'square' array
longEdgelength  = ceil(sqrt(nImages));
shortEdgelength = ceil(nImages/longEdgelength);

dummyImg = zeros(vSize + [2 2]); % Count a one-pixel border around each image
imgToPlot = repmat(dummyImg, [longEdgelength shortEdgelength]);

% Reshape images into output image to be displayed
iCount = 0;
for iImages = 1:longEdgelength
  for jImages = 1:shortEdgelength
    iCount = iCount + 1;
    if iCount <= nImages
      imgToPlot( (iImages-1)*(vSize(1)+2) + (1:(vSize(1)+2)) , (jImages-1)*(vSize(2)+2) + ...
                            (1:(vSize(2)+2))) = padarray(reshape(X(iCount,:), vSize),[1 1],1);
    end
  end
end

% Display image
imagesc(imgToPlot);
axis image;
title(['Display ',num2str(nImages), ' example images']);
set(gca, 'XTick', [], ...        % Remove axes tick marks
         'XTickLabel', {}, ...   % and tick labels
         'YTick', [], ...
         'YTickLabel', {}, ...
         'TickLength', [0 0]);
end
