# -*- coding: utf-8 -*-
# Author: wxb
# date: 2019.10.29
# functional description:
"""
"""

import os
import sys
import shutil
from PIL import  Image as pil_image
import threading,signal
import time

def decode_image():
    video_root="/media/zhanglijing/新加卷/wxb/SRGame/dataset/SRGame_traindataset"
    # folders=["SDR_4K_Part1","SDR_4K_Part2","SDR_4K_Part3","SDR_4K_Part4"]
    folders=["SDR_4K_Part3"]
    for folder in folders:
        video_list=[x for x in os.listdir(os.path.join(video_root,folder))]
        save_video_root=os.path.join("/media/zhanglijing/新加卷/wxb/SRGame/dataset/SRGame_traindataset/",folder+"_images")
        if os.path.exists(save_video_root):
            shutil.rmtree(save_video_root)
        os.makedirs(save_video_root)
        for i in video_list:
            save_image_path=os.path.join(save_video_root,i.split(".")[0])
            if os.path.exists(save_image_path):
                shutil.rmtree(save_image_path)
            os.makedirs(save_image_path)
            command="ffmpeg -i {} {}/%04d.png".format(os.path.join(video_root,folder,i),save_image_path)
            os.system(command)

def merge_video(image_root,save_root):
    # image_root="./test_result/SR/"
    # save_root="./test_result/SR_video"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.makedirs(save_root)
    image_folders=[x for x in os.listdir(image_root)]
    for folder in image_folders:
        command="ffmpeg -r 24000/1001 -i {}/%04d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 {}/{}.mp4 -y".format(os.path.join(image_root,folder),save_root,folder)
        os.system(command)

def merge_video_thread(start,end):
    for i in range(end-start):
        image_path=os.path.join(image_root,image_folder_list[i+start])
        save_name=image_folder_list[i+start]
        command = "ffmpeg -r 24000/1001 -i {}/%04d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 {}/{}.mp4 -y".format(image_path, save_root, save_name)
        os.system(command)

def get_file_size():
    file_root="/home/zhanglijing/Desktop/SRGame/test_result/11_12_result_thread"
    for x in os.listdir(file_root):
        if (os.path.getsize(os.path.join(file_root,x))/1024/1024) > 50:
            print(x)



if __name__=='__main__':
    merge_video("/home/zhanglijing/Desktop/SRGame/test_result/SR_image_RDN_patch_128","/home/zhanglijing/Desktop/SRGame/test_result/SR_video_RDN_patch_128")
    # img=pil_image.open("/home/zhanglijing/Desktop/SRGame/test_result/LR/16536366/0001.png").convert("YCbCr")
    # print(img)

    """
    num_threads=10
    image_root="/home/zhanglijing/Desktop/SRGame/test_result/SR_image_RDN_ours"
    save_root="/home/zhanglijing/Desktop/SRGame/test_result/11_12_result_thread"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    image_folder_list=os.listdir(image_root)
    count=len(image_folder_list)
    part=int(count/num_threads)
    thread_list=[]
    for i in range(num_threads):
        if (i == num_threads - 1):
            t = threading.Thread(target=merge_video_thread,
                                 kwargs={'start': i * part, 'end': count})
        else:
            t = threading.Thread(target=merge_video_thread,
                                 kwargs={'start': i * part, 'end': (i + 1) * part})
        t.setDaemon(True)
        thread_list.append(t)
        t.start()

    for i in range(num_threads):
        try:
            while thread_list[i].isAlive():
                pass
        except KeyboardInterrupt:
            break
    """
    # get_file_size()