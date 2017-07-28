#!/bin/bash
mkdir images
cd images
wget http://static2.businessinsider.com/image/57b36e46ce38f238008b7187-2048/7579069138_329207ae88_k.jpg
wget http://static6.businessinsider.com/image/577e8e0588e4a7fd018b67ac-2048/franklin-tennessee.jpg
wget https://s-media-cache-ak0.pinimg.com/originals/11/89/9b/11899be730c05cea10ed724ba8e1f762.jpg
wget http://i.imgur.com/TGQeLhU.jpg
cd ..

rm -f image_list_video.txt
#find images/leftImg8bit/demoVideo/ -name "*.png" -print > image_list_video_temp.txt
find images/ -name "*.jpg" -print > image_list_video_temp.txt

sort image_list_video_temp.txt > image_list_video.txt
# use the same images to create a longer list of video frames 
# (so we don't have to deal with GBs of hi-res pngs..)
cat image_list_video.txt > image_list_video_temp.txt
cat image_list_video_temp.txt >> image_list_video.txt
cat image_list_video_temp.txt >> image_list_video.txt
cat image_list_video_temp.txt >> image_list_video.txt
cat image_list_video_temp.txt >> image_list_video.txt
cat image_list_video_temp.txt >> image_list_video.txt
cat image_list_video_temp.txt >> image_list_video.txt
cat image_list_video_temp.txt >> image_list_video.txt
cat image_list_video_temp.txt >> image_list_video.txt
cat image_list_video_temp.txt >> image_list_video.txt
rm -f image_list_video_temp.txt

