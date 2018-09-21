minx=1610;
maxx=1678;
miny=490;
maxy=524;

rs=uint32(0);
gs=uint32(0);
bs=uint32(0);

for k=1:20
    stdd=imread(strcat(strcat('./data/n/n_',int2str(k)),'.jpg'));
    for i = minx:maxx
        for j=miny:maxy
            rs=rs+uint32(stdd(j,i,1));
            gs=gs+uint32(stdd(j,i,2));
            bs=bs+uint32(stdd(j,i,3));
        end
    end
end

for k=1:111
    stdd=imread(strcat(strcat('./data/p/p_',int2str(k)),'.jpg'));
    for i = minx:maxx
        for j=miny:maxy
            rs=rs+uint32(stdd(j,i,1));
            gs=gs+uint32(stdd(j,i,2));
            bs=bs+uint32(stdd(j,i,3));
        end
    end
end

for k=1:20
    stdd=imread(strcat(strcat('./data/x/x_',int2str(k)),'.jpg'));
    for i = minx:maxx
        for j=miny:maxy
            rs=rs+uint32(stdd(j,i,1));
            gs=gs+uint32(stdd(j,i,2));
            bs=bs+uint32(stdd(j,i,3));
        end
    end
end

rm=rs/((maxx-minx+1)*(maxy-miny+1)*151)
gm=gs/((maxx-minx+1)*(maxy-miny+1)*151)
bm=bs/((maxx-minx+1)*(maxy-miny+1)*151)

todd=imread('./data/p/p_93.jpg');
todd_org=todd;
rt=uint32(0);
gt=uint32(0);
bt=uint32(0);
for i = minx:maxx
    for j=miny:maxy
        rt=rt+uint32(todd(j,i,1));
        gt=gt+uint32(todd(j,i,2));
        bt=bt+uint32(todd(j,i,3));
    end
end
rt=rt/((maxx-minx+1)*(maxy-miny+1));
gt=gt/((maxx-minx+1)*(maxy-miny+1));
bt=bt/((maxx-minx+1)*(maxy-miny+1));

for i=1:maxx
    for j=1:maxy
        todd(j,i,1)=uint8(uint32(todd(j,i,1))*rm/rt);
        todd(j,i,2)=uint8(uint32(todd(j,i,2))*gm/gt);
        todd(j,i,3)=uint8(uint32(todd(j,i,3))*bm/bt);
    end
end
subplot(3,1,1);
imshow(todd_org)
subplot(3,1,2);
imshow(todd)