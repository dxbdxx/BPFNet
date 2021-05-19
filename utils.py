""" Utilities """
import os
import logging
import shutil
import torch
import numpy as np
import multiprocessing
import torch.nn.functional as F
import shapely.geometry
import shapely.affinity
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc


def get_logger(file_path, distributed_rank=0):
    """ Make python logger """
    logger = logging.getLogger('palm_cnn')
    if distributed_rank > 0:
        return logger
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(np.prod(v.size()) for k, v in model.named_parameters())
    return n_params / 1024. / 1024.


def save_checkpoint(state, ckpt_dir, is_best=False, epoch=0):
    filename = os.path.join(ckpt_dir, 'checkpoint_{}.pth.tar'.format(epoch))
    torch.save(state, filename)
    last_filename = os.path.join(ckpt_dir, 'last.pth.tar')
    shutil.copyfile(filename, last_filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def rankn_all2all(label, dist, n=1):
    idx = dist.argsort(dim=1)
    gt = label.view(label.shape[0], 1) == label.view(1, label.shape[0])
    count = 0
    for i in range(gt.shape[0]):
        if True in gt[i,idx[i]][1:n+1]:
            count += 1

    rank_n = count/gt.shape[0]
    return rank_n


def rankn_test2register(label, dist, register_list, test_list, n=1):
    num_register = len(register_list)
    num_test = len(test_list)
    num_classes = len(label.unique())

    dist_test2register = dist[register_list]
    dist_test2register = dist_test2register[:, test_list]
    label_test2register = label[register_list]

    dist_test2class = -F.max_pool2d(-dist_test2register.view(1, num_register, -1), kernel_size=(4, 1), stride=(4, 1))
    label_test2class = -F.max_pool2d(-label_test2register.view(1, num_register, -1), kernel_size=(4, 1), stride=(4, 1))
    gt_all = label_test2class.view(num_classes, 1) == label[test_list].view(1, num_test)

    idx = dist_test2class.squeeze().argsort(dim=0)

    count = 0
    for i in range(num_test):
        if True in gt_all[idx[:,i], i][:n]:
            count += 1

    rank_n = count/num_test
    return rank_n


def eer_all2all(label, dist):
    n = label.shape[0]
    gt = label.view(n, 1) == label.view(1,n)
    score = 1 - dist
    mat_select = (1 - torch.eye(n)).bool()

    gt_all = gt[mat_select]
    score_all = score[mat_select]

    fpr, tpr, thresholds = roc_curve(gt_all.cpu().numpy(), score_all.cpu().numpy())
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    return eer


def eer_test2register(label, dist, register_list, test_list):
    num_register = len(register_list)
    num_test = len(test_list)
    num_classes = len(label.unique())

    score = 1 - dist

    score_test2register = score[register_list]
    score_test2register = score_test2register[:, test_list]
    label_test2register = label[register_list]

    score_test2class = F.max_pool2d(score_test2register.view(1, num_register, -1), kernel_size=(4,1), stride=(4,1))
    label_test2class = F.max_pool2d(label_test2register.view(1, num_register, -1), kernel_size=(4,1), stride=(4,1))
    gt_all = label_test2class.view(num_classes, 1) == label[test_list].view(1, num_test)

    fpr, tpr, thresholds = roc_curve(gt_all.view(-1).cpu().numpy(), score_test2class.view(-1).cpu().numpy())
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)

    return eer


def process_map(r):
    return torch.Tensor([torch.mean(r[:i + 1].float()) for i in range(r.shape[0]) if r[i]])


def map(label, dist):
    if len(label) > 1000:
        idx = dist.argsort(dim=1)
        gt = label.view(label.shape[0], 1) == label.view(1, label.shape[0])
        rs = [gt[i][idx[i]] for i in range(gt.shape[0])]

        pool = multiprocessing.Pool(4)
        result = []
        for r in rs:
            p = pool.apply_async(process_map, args=(r.cpu(),))
            result.append(p)
        pool.close()
        pool.join()

        trec_precisions = []
        for r in result:
            trec_precisions.append(r.get())

        trec_precisions = torch.cat(trec_precisions)
        mAP = torch.mean(trec_precisions)

    else:
        idx = dist.argsort(dim=1)
        gt = label.view(label.shape[0], 1) == label.view(1, label.shape[0])
        rs = [gt[i][idx[i]] for i in range(gt.shape[0])]
        trec_precisions = []
        for r in rs:
            trec_precision = torch.Tensor([torch.mean(r[:i+1].float()) for i in range(r.shape[0]) if r[i]])
            trec_precisions.append(trec_precision)

        trec_precisions = torch.cat(trec_precisions)
        mAP = torch.mean(trec_precisions)

    return mAP.item()


def iou(box1, box2, theta1, theta2):
    box1, box2, theta1, theta2 = box1.cpu().numpy(), box2.cpu().numpy(), theta1.cpu().numpy(), theta2.cpu().numpy()

    iou = []
    for b1, b2, t1, t2 in zip(box1, box2, theta1, theta2):
        r1 = shapely.geometry.box(*b1)
        r1 = shapely.affinity.rotate(r1, t1)
        try:
            r2 = shapely.geometry.box(*b2)
            r2 = shapely.affinity.rotate(r2, t2)
            I = r1.intersection(r2).area
            O = r1.union(r2).area
        except:
            I = 0
            O = 1

        iou.append(I/(O+1))

    iou = np.array(iou)
    return iou.mean()


def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep