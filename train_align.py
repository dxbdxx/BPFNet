import os
import torch
import torch.nn as nn
import numpy as np
from data.dataset import CUHKSZ_align
from models.model import PalmAlignNet
from models.losses import FocalLoss, BinRotLoss
from config import TrainConfig
import utils


def main(config, logger):
    logger.info("Logger is set - training start")

    # split data to train/validation
    data_dir = './data/CUHKSZ/'
    train_dataset = CUHKSZ_align(data_dir, 'train')
    test_dataset = CUHKSZ_align(data_dir, 'test')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=True,
                                               batch_size=config.batch_size,
                                               num_workers=8,
                                               pin_memory=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              shuffle=False,
                                              batch_size=config.batch_size,
                                              num_workers=4,
                                              pin_memory=True)

    # model setting
    criterion = [FocalLoss(),
                 nn.L1Loss(),
                 BinRotLoss(),
                 nn.CrossEntropyLoss()]
    model = PalmAlignNet(config, train_dataset.num_classes).cuda()

    # weights optimizer
    optimizer_det = torch.optim.SGD(model.model_det.parameters(), config.lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    optimizer_cls = torch.optim.SGD(model.model_cls.parameters(), config.lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    lr_scheduler_det = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_det, config.epochs, eta_min=config.lr_min)
    lr_scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cls, config.epochs, eta_min=config.lr_min)

    # training loop
    best_eer = 1
    for epoch in range(config.epochs):
        # training
        train(train_loader, model, optimizer_det, optimizer_cls, criterion, epoch, config)
        lr_scheduler_det.step()
        lr_scheduler_cls.step()

        if (epoch + 1) % 5 == 0:
            # validation
            eer = validate(test_loader, model, epoch, config)

            # save
            if best_eer > eer:
                best_eer = eer
                is_best = True
            else:
                is_best = False

            if (epoch + 1) % config.save_freq == 0:
                utils.save_checkpoint(model.state_dict(), config.path, is_best, (epoch + 1))
            print("")
        torch.cuda.empty_cache()

    logger.info("Final best eer {:.4%}".format(eer))


def train(train_loader, model, optimizer1, optimizer2, criterion, epoch, config):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses1 = utils.AverageMeter()
    losses2 = utils.AverageMeter()
    losses3 = utils.AverageMeter()
    losses4 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_lr = optimizer1.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch + 1, cur_lr))

    model.train()
    for step, (img, y, hm, disp_gt, wh_gt, bbox_gt, theta_gt) in enumerate(train_loader):
        img, y, hm, disp_gt, theta_gt, wh_gt = img.cuda(), y.cuda(), hm.cuda(), disp_gt.cuda(), theta_gt.cuda(), wh_gt.cuda()
        N = img.size(0)

        hd1, wh_pred, disp_pred, bin_pred, logits, ft = model(img, y)

        loss1 = criterion[0](hd1, hm)
        loss2_1 = criterion[1](wh_pred, wh_gt)
        loss2_2 = criterion[1](disp_pred, disp_gt)
        loss2 = loss2_1 + loss2_2
        loss3 = criterion[2](bin_pred, theta_gt)
        loss4 = criterion[3](logits, y)

        if epoch < 15:
            loss = config.lw1 * loss1 + config.lw2 * loss2 + config.lw3 * loss3
            optimizer1.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer1.step()
        else:
            loss = config.lw1 * loss1 + config.lw2 * loss2 + config.lw3 * loss3 + config.lw4 * loss4
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer1.step()
            optimizer2.step()

        losses1.update(loss1.item(), N)
        losses2.update(loss2.item(), N)
        losses3.update(loss3.item(), N)
        losses4.update(loss4.item(), N)
        losses.update(loss.item(), N)

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if (step + 1) % config.print_freq == 0 or step == len(train_loader) - 1:
            logger.info("Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f}, Loss1 {losses1.avg:.3f}, "
                        "Loss2 {losses2.avg:.3f}, Loss3 {losses3.avg:.3f}, Loss4 {losses4.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(epoch + 1, config.epochs, step + 1,
                                len(train_loader), losses=losses, losses1=losses1, losses2=losses2, losses3=losses3,
                                losses4=losses4, top1=top1, top5=top5))


def validate(valid_loader, model, epoch, config):
    IoU = utils.AverageMeter()
    dim_feature = 512
    num_valid_sample = len(valid_loader.dataset)
    feature_map = torch.zeros(num_valid_sample, dim_feature).cuda()
    ground_truth = torch.zeros(num_valid_sample).cuda()
    index_begin = 0
    model.eval()
    with torch.no_grad():
        for step, (img, y, disp_gt, bbox_gt, theta_gt) in enumerate(valid_loader):
            img, y, disp_gt, bbox_gt, theta_gt = img.cuda(), y.cuda(), disp_gt.cuda(), bbox_gt.cuda(), theta_gt.cuda()
            N = img.size(0)

            bbox_pred, theta_pred, logits, ft = model(img, y)

            index_end = index_begin + N
            feature_map[index_begin:index_end, :] = ft.view(N, dim_feature)
            ground_truth[index_begin:index_end] = y
            index_begin = index_end

            iou1 = utils.iou(bbox_gt, bbox_pred, theta_gt, theta_pred)
            IoU.update(iou1, N)

        fm_n = feature_map.norm(p=2, dim=1)
        dist = 1 - torch.matmul(feature_map / fm_n.view(num_valid_sample, 1),
                                (feature_map / fm_n.view(num_valid_sample, 1)).t())

        # metrics
        eer_a = utils.eer_all2all(ground_truth, dist)
        eer_t = utils.eer_test2register(ground_truth, dist, valid_loader.dataset.register_list,
                                        valid_loader.dataset.test_list)
        rank1_a = utils.rankn_all2all(ground_truth, dist, 1)
        rank1_t = utils.rankn_test2register(ground_truth, dist, valid_loader.dataset.register_list,
                                            valid_loader.dataset.test_list, 1)

        logger.info("Valid: [{:2d}/{}] Step {:03d}/{:03d}, EER_a {eer_a:.3%}, EER_t {eer_t:.3%}, Rank-1_a {rank1_a:.3%},"
                    " Rank-1_t {rank1_t:.3%} IoU {iou:.3%}".format(epoch + 1, config.epochs, step + 1, len(valid_loader),
                                             eer_a=eer_a, eer_t=eer_t, rank1_a=rank1_a, rank1_t=rank1_t, iou=IoU.avg))

    return eer_a


def setup_environment():
    config = TrainConfig()

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])
    torch.backends.cudnn.benchmark = True

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # logging
    if not os.path.isdir(config.path):
        os.mkdir(config.path)
    logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
    config.print_params(logger.info)

    return config, logger


if __name__ == "__main__":
    config, logger = setup_environment()
    main(config, logger)
