# -*- coding: utf-8 -*-
# Author:
# date:
# functional description
"""

 Evaluation Index

"""

import numpy as np
import math
import numpy
import cv2
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
import os
import json
from skimage.measure import compare_ssim


def psnr(hr_image, sr_image, hr_edge):

    # [h,w,c] = hr_image.shape
    hr_image_data = np.array(hr_image)
    if hr_edge > 0:
        hr_image_data = hr_image_data[hr_edge:-hr_edge, hr_edge:-hr_edge].astype('float32')

    sr_image_data = np.array(sr_image).astype('float32')


    diff = sr_image_data - hr_image_data

    diff = numpy.abs(diff)
    # mse = (numpy.square(diff)).sum()/(h*w*c)
    # psnr = 20 * math.log10(255 / math.sqrt(mse))
    # print "test:"+str(psnr)

    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))
    psnr = 20 * math.log10(255.0 / rmse)


    return psnr

def psnr_g(hr_image, sr_image):

    grayH = cv2.cvtColor(hr_image, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(sr_image, cv2.COLOR_BGR2GRAY)

    diff = grayR - grayH
    diff = diff.flatten('C')

    diff = numpy.abs(diff)
    rmse = ((numpy.square(diff)).sum())/65025
    # rmse = math.sqrt(np.mean(diff ** 2.))
    print (rmse)
    psnr = 10 * math.log10(255.0*255.0 / rmse)
    print ("gray:"+str(psnr))

def psnr_y(hr_image, sr_image, hr_edge):


    # hr_image_data = np.array(hr_image)
    # if hr_edge > 0:
    #     hr_image = hr_image_data[hr_edge:-hr_edge, hr_edge:-hr_edge].astype('float32')
    #
    # sr_image = np.array(sr_image).astype('float32')

    YCrCbH = cv2.cvtColor(hr_image, cv2.COLOR_BGR2YCrCb).astype('float32')
    YCrCbR = cv2.cvtColor(sr_image, cv2.COLOR_BGR2YCrCb).astype('float32')
    YH, _, _ = cv2.split(YCrCbH)
    YR, _, _ = cv2.split(YCrCbR)

    # print YH.shape
    if hr_edge > 0:
        YH = YH[hr_edge:-hr_edge, hr_edge:-hr_edge]
        YR = YR[hr_edge:-hr_edge, hr_edge:-hr_edge]

    # hr_image_data = np.array(YCrCbH)
    # hr_y = hr_image_data[:,:,1]

    # sr_image_data = np.array(YCrCbR)
    # sr_y = sr_image_data[:,:,1]

    diff_y = YH - YR
    # print diff_y.shape
    diff_y.flatten()

    diff_y = numpy.abs(diff_y)

    rmse = math.sqrt(np.mean(diff_y ** 2.))
    # rmse = ((numpy.square(diff_y)).sum()) / 65025
    psnr = 20 * math.log10(255.0 / rmse)
    # print psnr
    # mse = (numpy.square(diff_y)).sum() / diff_y.size
    # psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    # print "YCrCb"+str(psnr)
    return psnr





def block_view(A, block=(3, 3)):
    shape = (A.shape[0]/block[0], A.shape[1]/block[1])+ block
    strides = (block[0]*A.strides[0], block[1]*A.strides[1])+A.strides
    return ast(A, shape=shape, strides=strides)


def ssim(img1, img2, C1=0.01**2, C2=0.03**2):

    bimg1 = block_view(img1, (4,4))
    bimg2 = block_view(img2, (4,4))
    s1  = numpy.sum(bimg1, (-1, -2))
    s2  = numpy.sum(bimg2, (-1, -2))
    ss  = numpy.sum(bimg1*bimg1, (-1, -2)) + numpy.sum(bimg2*bimg2, (-1, -2))
    s12 = numpy.sum(bimg1*bimg2, (-1, -2))

    vari = ss - s1*s1 - s2*s2
    covar = s12 - s1*s2

    ssim_map =(2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
    return numpy.mean(ssim_map)



def test():
    def psnr(im1, im2):
        diff = numpy.abs(im1 - im2)
        rmse = numpy.sqrt(diff).sum()
        psnr = 20 * numpy.log10(255 / rmse)
        return psnr

    x = numpy.array([[1, 2], [3, 4]])
    print (x)
    y = numpy.array([[5, 6], [7, 8]])
    print (y)

    psnr = psnr(x, y)
    print (psnr)

def calculate_vmaf(format,width,height,reference_path,distorted_path,output_format="json"):
    assert (output_format=="text" or output_format=="xml" or output_format=="json")
    command="/home/zhanglijing/Desktop/SRGame/vmaf/run_vmaf {} {} {} {} {} --out-fmt {} ".format(format,width,height,reference_path,distorted_path,output_format)
    result=os.popen(command)
    return (json.loads(result.read())["aggregate"]["VMAF_score"])
    # return json.loads(result.read())["aggregate"]["VMAF_score"]

def read_log(log_path):
    assert os.path.exists(log_path)
    with open(log_path, 'r') as load_f:
        load_dict = json.load(load_f)
    load_f.close()
    return (load_dict["PSNR score"],load_dict["SSIM score"],load_dict["VMAF score"])

if __name__ == "__main__":
    # result=calculate_vmaf("yuv420p",3840,2160,"/media/zhanglijing/新加卷/wxb/SRGame/dataset/SRGame_traindataset/SDR_4K_Part1/10091373.mp4","/home/zhanglijing/Desktop/test.mp4")
    # print(result)
    # SR_image_root="/media/zhanglijing/新加卷/wxb/SRGame/dataset/SRGame_traindataset/Val/SR_image_ours"
    # label_image_root="/media/zhanglijing/新加卷/wxb/SRGame/dataset/SRGame_traindataset/SDR_4K_Part4_images"

    SR_video_path = "/media/zhanglijing/新加卷/wxb/SRGame/dataset/SRGame_traindataset/Val/SR_video_RDN_patch_32"
    label_video_path = "/media/zhanglijing/新加卷/wxb/SRGame/dataset/SRGame_traindataset/SDR_4K_Part4"
    log_root="/media/zhanglijing/新加卷/wxb/SRGame/dataset/SRGame_traindataset/Val/log"
    date="11_15_18_32_"
    prefix="ours_17epochs_"
    description = "rdn model base game dataset,patch:{},batch:{},epoch:{}"

    video_list=os.listdir(SR_video_path)

    PSNR_total = 0.0
    SSIM_total = 0.0
    VMAF_total = 0.0
    for video in video_list:
        log_path = os.path.join(log_root, prefix + date + video.split(".")[0] + ".log")

        distorted_video=os.path.join(SR_video_path,video)
        reference_video=os.path.join(label_video_path,video)

        command="ffmpeg -i {} -i {} -lavfi libvmaf=\"psnr=1:ssim=1:log_fmt=json:log_path={}\" -f null -".format(distorted_video,reference_video,log_path)
        os.system(command)

        (PSNR, SSIM, VMAF) = read_log(log_path)
        PSNR_total += PSNR
        SSIM_total += SSIM
        VMAF_total += VMAF
        print("video:{}, PSNR:{} ,SSIM:{} ,VMAF:{} ,".format(video,PSNR,SSIM,VMAF))
    # 评测方法： 最终得分 = 25 * PSNR 项 + 25 * SSIM 项 + 50 * VMAF 项



    PSNR_avg = PSNR_total/ len(video_list)
    SSIM_avg=SSIM_total/len(video_list)
    VMAF_avg=VMAF_total/len(video_list)
    score = 25 * PSNR_avg / 50 + 25 * (SSIM_avg - 0.4) / 0.6 + 50 * VMAF_avg / 80
    print("PSNR_avg:{} SSIM_avg:{} VMAF_avg:{} final_score:{} ".format(PSNR_avg,SSIM_avg,VMAF_avg,score))
    final_log="/media/zhanglijing/新加卷/wxb/SRGame/dataset/SRGame_traindataset/Val/final_score.log"
    with open(final_log,"a") as f:
        f.write("Description: {}\nDate:{}\nScore: PSNR_avg:{} SSIM_avg:{} VMAF_avg:{} final_score:{} \n".format(description,date,PSNR_avg,SSIM_avg,VMAF_avg,score))
    f.close()

    # save_file="./result/RDN_ours_result.txt"
    # f=open(save_file,"r")
    # contents = f.readlines()
    # # print(contents)
    # for i in contents:
    #     content=i.split(" ")
    #     # print(content)
    #     PSNR_total+=float(content[1].split(",")[0])
    #     SSIM_total+=float(content[2])
    # f.close()
    #
    # f=open(save_file,"a")
    # # for folder in os.listdir(SR_image_root):
    # #     for img_path in os.listdir(os.path.join(SR_image_root,folder)):
    # #         imageH = cv2.imread(os.path.join(SR_image_root,folder,img_path))
    # #         imageR = cv2.imread(os.path.join(label_image_root,folder,img_path))
    # #         # print(os.path.join(SR_image_root,folder,img_path),os.path.join(label_image_root,folder,img_path))
    # #         # print imageH
    # #         # psnr_val = psnr_ssim.psnr(imageH, imageR, 0)
    # #         # psnr_val = psnr_ssim.psnr_y(imageH, imageR, 0)
    # #         img_psnr= psnr(imageH, imageR, 0)
    # #         PSNR_total+=img_psnr
    # #         grayH = cv2.cvtColor(imageH, cv2.COLOR_BGR2GRAY)
    # #         grayR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)
    # #         (score, diff) = compare_ssim(grayH, grayR, full=True)
    # #         SSIM_total+=score
    # #         f.write(os.path.join(SR_image_root,folder,img_path)+"PSNR: {},SSIM: {}\n".format(img_psnr,score))
    # #         print(os.path.join(SR_image_root,folder,img_path),"PSNR: {},SSIM: {}".format(img_psnr,score))
    # PSNR_avg=PSNR_total/100/len(os.listdir(SR_image_root))
    # SSIM_avg=SSIM_total/100/len(os.listdir(SR_image_root))
    #
    #
    # SR_video_path="/media/zhanglijing/新加卷/wxb/SRGame/dataset/SRGame_traindataset/Val/SR_video_ours"
    # label_video_path="/media/zhanglijing/新加卷/wxb/SRGame/dataset/SRGame_traindataset/SDR_4K_Part4"
    #
    # # VMAF_total=0
    # # for video in os.listdir(SR_video_path):
    # #     if not video.split(".")[-1]=="mp4":
    # #         continue
    # #     print(os.path.join(SR_video_path,video),os.path.join(label_video_path,video))
    # #     video_vmaf=calculate_vmaf("yuv422p", 3840, 2160,os.path.join(label_video_path,video),os.path.join(SR_video_path,video))
    # #     VMAF_total+=video_vmaf
    # #     f.write(os.path.join(SR_video_path,video)+" vmaf:{} \n".format(video_vmaf))
    # #     print(os.path.join(SR_video_path,video),"vmaf:{}".format(video_vmaf))
    # # VMAF_avg=VMAF_total/len(os.listdir(SR_video_path))
    # VMAF_avg=10.643515687406664
    #
    # score=25*PSNR_avg/50+25*(SSIM_avg-0.4)/0.6+50*VMAF_avg/80
    # f.write("PSNR_avg: {},SSIM_avg: {},VMAF_avg:{} \n".format(PSNR_avg, SSIM_avg,VMAF_avg))
    # print("PSNR_avg: {},SSIM_avg: {},VMAF_avg:{} ".format(PSNR_avg, SSIM_avg,VMAF_avg))
    # f.write("score:{} \n".format(score))
    # print("score:{}".format(score))
    # f.close()