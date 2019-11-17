# -*- coding: utf-8 -*-
# Author:
# date:
# functional description
"""

"""

import argparse
import os
import copy
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import RDN
# from datasets import TrainDataset, EvalDataset
from dataset import DatasetFromFolder,DatasetFromFolderEval
from utils import AverageMeter, calc_psnr, convert_rgb_to_y, denormalize
import datetime
from rcan import make_model
# from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default="RCAN")
    parser.add_argument('--img_dir_root', type=str, default="/media/zhanglijing/新加卷/wxb/SRGame/dataset/SRGame_traindataset")
    parser.add_argument('--lr_dir', type=str, default="/media/zhanglijing/新加卷/wxb/SRGame/dataset/SRGame_traindataset/SDR_540p_images")
    parser.add_argument('--outputs-dir', type=str, default="./RCAN_checkpoint/")
    parser.add_argument('--log_dir', type=str, default="./train_log/")
    # parser.add_argument('--weights-file', type=str, default="./checkpoint/x4/epoch_16_0.0.pth")
    parser.add_argument('--weights-file', type=str, default=None)


    #(RDN parameters)
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)



    #(RCAN parameters)
    parser.add_argument('--n_resgroups', type=int, default=10)
    parser.add_argument('--n_resblocks', type=int, default=16)
    parser.add_argument('--n_feats', type=int, default=64)
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')



    #train parameters
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--multi-gpu', type=bool,default=False)
    parser.add_argument('--start-epoch', type=int,default=0)
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    now_time = datetime.datetime.now().strftime("%F_%T")
    log_file = open(os.path.join(args.log_dir, args.model_name+"_"+now_time + ".log"), "w")

    # # RDN parameters
    # log_file.write("args:\nmodel_name:{}\nimg_dir_root:{}\nlr_dir:{}\noutputs-dir:{}\nweights-file:{}\nnum-features:{}\n"
    #                "growth-rate:{}\nnum-blocks:{}\nnum-layers:{}\npatch-size:{}\nlr:{}\nbatch-size:{}\nstart-epoch:{}\n\n\n\ntrain_log:\n"
    #                "".format(args.model_name,args.img_dir_root, args.lr_dir, args.outputs_dir, args.weights_file, args.num_features
    #                          , args.growth_rate, args.num_blocks, args.num_layers, args.patch_size, args.lr,
    #                          args.batch_size, args.start_epoch))


    # args.outputs_dir = os.path.join(args.outputs_dir,
    #                                 'x{}_{}_{}_{}_{}'.format(args.scale, args.num_features, args.growth_rate,
    #                                                          args.num_blocks, args.num_layers))


    #RCAN_parameters
    log_file.write(
        "args:\nmodel_name:{}\nimg_dir_root:{}\nlr_dir:{}\noutputs-dir:{}\nweights-file:{}\nn_feats:{}\n"
        "n_resblocks:{}\nn_resgroups:{}\npatch-size:{}\nlr:{}\nbatch-size:{}\nstart-epoch:{}\n\n\n\ntrain_log:\n"
        "".format(args.model_name, args.img_dir_root, args.lr_dir, args.outputs_dir, args.weights_file,
                  args.n_feats
                  , args.n_resblocks, args.n_resgroups,args.patch_size, args.lr,
                  args.batch_size, args.start_epoch))
    args.outputs_dir = os.path.join(args.outputs_dir,
                                    'x{}_{}_{}_{}'.format(args.scale, args.n_feats, args.n_resblocks,
                                                             args.n_resgroups))

    # writer_train = SummaryWriter(log_dir='tensorboard')
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    #RDN
    # model = RDN(scale_factor=args.scale,
    #             num_channels=3,
    #             num_features=args.num_features,
    #             growth_rate=args.growth_rate,
    #             num_blocks=args.num_blocks,
    #             num_layers=args.num_layers).to(device)


    #RCAN
    model=make_model(args).to(device)
    # if args.multi_gpu == False:
    #     model = RDN(scale_factor=args.scale,
    #                 num_channels=3,
    #                 num_features=args.num_features,
    #                 growth_rate=args.growth_rate,
    #                 num_blocks=args.num_blocks,
    #                 num_layers=args.num_layers).to(device)
    # else:
    # 	model = RDN(scale_factor=args.scale,
    #                 num_channels=3,
    #                 num_features=args.num_features,
    #                 growth_rate=args.growth_rate,
    #                 num_blocks=args.num_blocks,
    #                 num_layers=args.num_layers)
    # 	model = nn.DataParallel(model).cuda()

    if args.weights_file is not None:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # if args.multi_gpu:
    #     optimizer = optim.SGD(model.module.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    # else:
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

    folders=["SDR_4K_Part1_images","SDR_4K_Part2_images","SDR_4K_Part3_images","SDR_4K_Part4_images"]
    train_dataset = DatasetFromFolder(args.img_dir_root,folders,args.lr_dir,patch_size=args.patch_size, upscale_factor=args.scale)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    # eval_dataset = DatasetFromFolderEval()
    # eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_epoch = 0
    best_psnr = 0.0
    F = 2
    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

        model.train()
        epoch_losses = AverageMeter()
        loss_weight = 0.6
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))
            count=0
            for data in train_dataloader:
                count+=1
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                preds = model(inputs)

                loss = criterion(preds, labels)
                # y1_f = model.module.sfe2(model.module.sfe1(labels))
                # y2_f = model.module.sfe2(model.module.sfe1(preds))
                # loss2 = criterion(y1_f, y2_f)
                # loss = loss_weight*loss1 + (1-loss_weight)*loss2
                # epoch_losses.update(loss.item(), len(inputs))
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                #
                # t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                # t.update(len(inputs))
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
                log_file.write("epoch:{}/{},iteration:{}/{},loss={:.6f}\n".format(epoch, args.num_epochs - 1, count,
                                                                                  len(train_dataset), epoch_losses.avg))
        torch.save(model.state_dict(),os.path.join(args.outputs_dir, 'epoch_{}_{}.pth'.format(args.start_epoch + epoch, best_psnr)))
        # if epoch%F == 0:
        #     model.eval()
        #     writer_train.add_scalar("train_loss",epoch_losses.avg)
        #     epoch_psnr = AverageMeter()
        #     epoch_loss_v = AverageMeter()
        #     end_iter = 0
        #     for data in eval_dataloader:
        #         if end_iter >=60:
        #             break
        #         inputs, labels = data
        #
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #
        #         with torch.no_grad():
        #             preds = model(inputs)
        #         loss = criterion(preds, labels)
        #         epoch_loss_v.update(loss.item(), len(inputs))
        #
        #         preds = convert_rgb_to_y(denormalize(preds.squeeze(0)), dim_order='chw')
        #         labels = convert_rgb_to_y(denormalize(labels.squeeze(0)), dim_order='chw')
        #
        #         preds = preds[args.scale:-args.scale, args.scale:-args.scale]
        #         labels = labels[args.scale:-args.scale, args.scale:-args.scale]
        #
        #         epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
        #         end_iter += 1
        #     writer_train.add_scalar("val_loss",epoch_loss_v.avg)
        #     writer_train.add_scalar("val_psnr",epoch_psnr.avg)
        #     print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        #
        #     if epoch_psnr.avg > best_psnr:
        #         best_epoch = epoch
        #         best_psnr = epoch_psnr.avg
        #         print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
        #         torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}_{}.pth'.format(args.start_epoch+epoch,best_psnr)))
    # writer_train.close()
    log_file.close()