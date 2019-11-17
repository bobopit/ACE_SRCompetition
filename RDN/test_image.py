# -*- coding: utf-8 -*-
# Author: wxb
# date: 2019.10.29
# functional description
"""

"""

import argparse
import os
from os import listdir
import torch
# import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from tqdm import tqdm
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
print(current_dir)
sys.path.append(current_dir)
sys.path.append("..")
from model import RDN
from utils import convert_rgb_to_y, denormalize, calc_psnr
from rcan import make_model

def merge_video(image_path,save_path,video_name):

    # video_name=os.path.basename(image_path)
    command = "ffmpeg -r 24000/1001 -i {}/%04d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 {}/{}.mp4 -y".format(image_path, save_path, video_name)
    os.system(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default="./RCAN_checkpoint/x4_64_16_10/epoch_3_0.0.pth")
    parser.add_argument('--image_file_root', type=str, default="/home/zhanglijing/Desktop/SRGame/test_result/LR")
    parser.add_argument('--image_save_root', type=str, default="/home/zhanglijing/Desktop/SRGame/test_result/SR_image_RCAN_patch_32")
    parser.add_argument('--save_video_path', type=str,
                        default="/home/zhanglijing/Desktop/SRGame/test_result/SR_video_RCAN_patch_32")
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)

    # (RCAN parameters)
    parser.add_argument('--n_resgroups', type=int, default=10)
    parser.add_argument('--n_resblocks', type=int, default=16)
    parser.add_argument('--n_feats', type=int, default=64)
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')

    args = parser.parse_args()

    if not os.path.exists(args.image_save_root):
        os.makedirs(args.image_save_root)
    if not os.path.exists(args.save_video_path):
        os.makedirs(args.save_video_path)
    # cudnn.benchmark = True
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')

    # model = RDN(scale_factor=args.scale,
    #             num_channels=3,
    #             num_features=args.num_features,
    #             growth_rate=args.growth_rate,
    #             num_blocks=args.num_blocks,
    #             num_layers=args.num_layers).to(device)
    model = make_model(args).to(device)
    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    image_folder = [x for x in os.listdir(args.image_file_root)]
    # image_folder.sort()
    save_video_path = args.save_video_path
    flag=True
    for k,folder in enumerate(image_folder):
        # if(k<25):
        #     continue
        # if folder=="61883118":
        #     flag=False
        # if flag:
        #     continue
        # print(folder)
        # continue
        image_save_folder=os.path.join(args.image_save_root,folder)
        # if folder=="99849849":
        #     continue
        if os.path.exists(image_save_folder):
            import shutil
            shutil.rmtree(image_save_folder)
        os.makedirs(image_save_folder)
        images_name=[x for x in os.listdir(os.path.join(args.image_file_root,folder))]
        for image_name in tqdm(images_name, desc='convert LR images to HR images'):
            #try:
                lr = pil_image.open(os.path.join(os.path.join(args.image_file_root,folder,image_name))).convert('RGB')

                # image_width = (image.width // args.scale) * args.scale
                # image_height = (image.height // args.scale) * args.scale
                #
                # hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
                # lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
                # bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
                # bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

                lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
                # hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
                lr = torch.from_numpy(lr).to(device)
                # hr = torch.from_numpy(hr).to(device)

                with torch.no_grad():
                    preds = model(lr).squeeze(0)

                # preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
                # hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')

                # preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
                # hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]
                #
                # psnr = calc_psnr(hr_y, preds_y)
                # print('PSNR: {:.2f}'.format(psnr))

                output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())
                output.save(os.path.join(image_save_folder,image_name))
        merge_video(os.path.join(args.image_save_root, folder), save_video_path, folder)
            # except:
            #     continue