function val = gfi()
    % Return current global fi and increment it
    global fi
    if isempty(fi)
        fi = 0;
    end
    val = fi;
    fi = fi + 1;
end

function val = gfi_clear()
% Return current global fi and increment it
global fi
if isempty(fi)
    fi = 0;
end
val = fi;
fi = 0;
end

N = imfinfo('peppers.png');
Ni= imread('peppers.png');
%N.FileModData
%imshow(Ni);


% Convert to grayscale if needed
if size(Ni,3) == 3
    I = rgb2gray(Ni);
else
    I = Ni;
end

I = im2double(I);

% Roberts operator kernels
R1 = ...
[1 0;...
 0 -1];

R2 = ...
[0 1;...
-1 0];
Gx = conv2(I, R1, 'same');
Gy = conv2(I, R2, 'same');
G = sqrt(Gx.^2 + Gy.^2);
G = mat2gray(G);
th = graythresh(G);
edges = imbinarize(G, th);

figure(gfi);%ubplot(1,3,1), 
imshow(I), title('Grayscale');
figure(gfi);%subplot(1,3,2),
imshow(G), title('Roberts Gradient Magnitude');
figure(gfi);%subplot(1,3,3),
imshow(edges), title('Roberts Edges (binary)');

% --- Begin manual Canny edge detector (no toolbox Canny) ---

% Parameters
sigma = 1.0;          % Gaussian smoothing sigma
lowRatio = 0.05;      % low hysteresis threshold ratio (of max)
highRatio = 0.15;     % high hysteresis threshold ratio (of max)

% 1) Smooth with Gaussian
sz = max(3,fix(6*sigma)); if mod(sz,2)==0, sz = sz+1; end
half = (sz-1)/2;
x = -half:half;
G = exp(-(x.^2)/(2*sigma^2));
G = G / sum(G);
% separable convolution
Is = conv2(conv2(I, G(:), 'same'), G(:)', 'same');

% 2) Gradient (use Sobel-like derivative filters)
dx = [-1 0 1; -2 0 2; -1 0 1]/8; % normalized
dy = dx';
Ix = conv2(Is, dx, 'same');
Iy = conv2(Is, dy, 'same');
gradMag = hypot(Ix, Iy);
gradDir = atan2(Iy, Ix); % radians

% 3) Non-maximum suppression
[M, Npix] = size(I);
nms = zeros(M,Npix);
% quantize direction to 4 sectors: 0,45,90,135 degrees
angle = rad2deg(gradDir);
angle(angle < 0) = angle(angle < 0) + 180;
for r = 2:M-1
    for c = 2:Npix-1
        a = angle(r,c);
        q = 255; p = 255;
        if ( (a >= 0 && a < 22.5) || (a >= 157.5 && a <= 180) )
            p = gradMag(r, c+1);
            q = gradMag(r, c-1);
        elseif (a >= 22.5 && a < 67.5)
            p = gradMag(r-1, c+1);
            q = gradMag(r+1, c-1);
        elseif (a >= 67.5 && a < 112.5)
            p = gradMag(r-1, c);
            q = gradMag(r+1, c);
        else % 112.5 - 157.5
            p = gradMag(r-1, c-1);
            q = gradMag(r+1, c+1);
        end
        if gradMag(r,c) >= p && gradMag(r,c) >= q
            nms(r,c) = gradMag(r,c);
        else
            nms(r,c) = 0;
        end
    end
end

% 4) Hysteresis thresholding
maxVal = max(nms(:));
highThresh = highRatio * maxVal;
lowThresh = lowRatio * maxVal;

strong = nms >= highThresh;
weak = nms >= lowThresh & nms < highThresh;

edges_canny = false(M,Npix);
% start from strong pixels and track connected weak pixels (8-connectivity)
[rs, cs] = find(strong);
stack = [rs, cs];
visited = false(M,Npix);
for k = 1:size(stack,1)
    r = stack(k,1); c = stack(k,2);
    if visited(r,c), continue; end
    visited(r,c) = true;
    edges_canny(r,c) = true;
    % examine 8-neighbors
    for dr = -1:1
        for dc = -1:1
            rr = r+dr; cc = c+dc;
            if rr<1 || rr>M || cc<1 || cc>Npix, continue; end
            if ~visited(rr,cc) && weak(rr,cc)
                stack(end+1,:) = [rr,cc]; 
            end
        end
    end
end

% optional clean small isolated edges using morphological opening (small)
edges_canny = bwareaopen(edges_canny, 10);

% Display results
figure(gfi); imshow(gradMag,[]), title('Canny: Gradient Magnitude (smoothed)');
%figure(gfi); imshow(nms,[]), title('Canny: Non-maximum Suppressed');
%figure(gfi); imshow(edges_canny), title('Canny Edges (manual)');
% --- End manual Canny edge detector ---


%Dilated A  B
se = [1 1 1; 1 1 1; 1 1 1];
gradMag_dilated = imdilate(gradMag, se);
figure(gfi); imshow(gradMag_dilated, []), title('Canny: Gradient Magnitude (dilated)');


