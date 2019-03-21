img=imread('./data/pos/p_105.jpg');

gray=(img(:,:,1)-img(:,:,3));
imshow(gray)
grayadj=imadjust(gray);
bw=uint8(im2bw(grayadj));
masking=(img.*repmat(bw,[1,1,3]));
imshowpair(img,masking,'montage')