import os
import torch
from PIL import Image
import cv2
import numpy as np
from data.dataset import CUHKSZ_align
from models.model import PalmAlignNet
from config import TrainConfig
import utils


def main():
    config = TrainConfig()
    data_dir = './data/CUHKSZ/'
    train_dataset = CUHKSZ_align(data_dir, 'train')
    test_dataset = CUHKSZ_align(data_dir, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              shuffle=False,
                                              batch_size=config.batch_size,
                                              num_workers=4,
                                              pin_memory=True)

    ## ----------------------------
    model = PalmAlignNet(config, train_dataset.num_classes)
    model.load_model('MODEL_PATH')
    model = model.cuda()

    model.eval()
    dim_feature = 512
    num_valid_sample = len(test_dataset)
    feature_map = torch.zeros(num_valid_sample, dim_feature).cuda()
    ground_truth = torch.zeros(num_valid_sample).cuda()
    index_begin = 0
    with torch.no_grad():
        for step, (img, y, disp_gt, bbox_gt, theta_gt) in enumerate(test_loader):
            img, y, disp_gt, bbox_gt, theta_gt = img.cuda(), y.cuda(), disp_gt.cuda(), bbox_gt.cuda(), theta_gt.cuda()

            N = img.size(0)
            index_end = index_begin + N
            bbox_pred, theta_pred, logits, ft = model(img, y, bbox_gt, theta_gt, disp_gt)

            feature_map[index_begin:index_end, :] = ft.view(N, dim_feature)
            ground_truth[index_begin:index_end] = y
            index_begin = index_end

    fm_n = feature_map.norm(p=2, dim=1)
    dist = 1 - torch.matmul(feature_map / fm_n.view(num_valid_sample, 1),
                            (feature_map / fm_n.view(num_valid_sample, 1)).t())

    eer_a = utils.eer_all2all(ground_truth, dist)
    eer_t = utils.eer_test2register(ground_truth, dist, test_loader.dataset.register_list,
                                    test_loader.dataset.test_list)
    rank1_a = utils.rankn_all2all(ground_truth, dist, 1)
    rank1_t = utils.rankn_test2register(ground_truth, dist, test_loader.dataset.register_list,
                                        test_loader.dataset.test_list, 1)

    print(eer_t, rank1_t)


if __name__ == "__main__":
    main()




