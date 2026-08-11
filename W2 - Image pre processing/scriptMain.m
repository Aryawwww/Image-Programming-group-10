img = imread('ngc6543a.jpg');
imshow(img);
whos img

figure(1);imshow(img);
img_R = img(:,:,1);
figure(2);imshow(img_R);

img_G = img(:,:,2);
figure(3);imshow(img_G);

img_B = img(:,:,3);
figure(4);imshow(img_B);

grey = rgb2gray(img);figure(5);imshow(grey);

[ind,map] = rgb2ind(img,256); imshow(ind,map);
figure(5);imshow(ind);
%figure(6);
%imshow(map);

labimg = rgb2lab(img);
limg = labimg(:,:,1);
aimg = labimg(:,:,2);
bimg = labimg(:,:,3);
figure(7); imshow(mat2gray(limg)); title('l channel');
figure(8); imshow(mat2gray(aimg)); title('a channel');
figure(9); imshow(mat2gray(bimg)); title('b channel');

%gaussian vague

h = fspecial('gaussian',[25,25],50);
filtered = imfilter(img,h);
figure(10);imshow(filtered);
%figure(11);imshow();
figure(12);
surf(h);
surf(hs);


