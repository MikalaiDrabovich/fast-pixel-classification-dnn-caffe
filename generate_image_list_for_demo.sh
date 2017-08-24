#!/bin/bash

if [ ! -d "images" ]; then
    mkdir images
    cd images
    wget https://github.com/MikalaiDrabovich/sample_images/archive/v1.0.tar.gz
    tar -xzvf v1.0.tar.gz
    mkdir -p 'leftImg8bit/demoVideo/paloalto_00/'
    mv ./sample_images-1.0/*.jpg ./leftImg8bit/demoVideo/paloalto_00/
    rm -f v1.0.tar.gz
    rm -rf sample_images-1.0
    cd ..
else
  echo 'Directory "images" exists, skipping download of a quick test dataset"'
fi
 
rm -f image_list_video.txt

# uncomment if you want to use images from cityscapes demo dataset
#find images/leftImg8bit/demoVideo/ -name "*.png" -print > image_list_video_temp.txt
find images/ -name "*.jpg" -print > image_list_video_temp.txt
sort image_list_video_temp.txt > image_list_video.txt

# use the same images to create a longer list of video frames 
#cat image_list_video.txt > image_list_video_temp.txt
#cat image_list_video_temp.txt >> image_list_video.txt

rm -f image_list_video_temp.txt

