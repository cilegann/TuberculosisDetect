img=imread('m8.jpg');
gray=img(:,:,1)-img(:,:,3);
bw=uint8(im2bw(gray,graythresh(gray)*1.8));
masking=(img.*repmat(bw,[1,1,3]));
imshowpair(img,masking,'montage')