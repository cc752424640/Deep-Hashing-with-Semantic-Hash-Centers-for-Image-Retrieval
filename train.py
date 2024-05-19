import copy

from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import random
from data.data_loader import load_data

torch.multiprocessing.set_sharing_strategy('file_system')


class CSQLoss(torch.nn.Module):
    def __init__(self, args, bit, hash_center):
        super(CSQLoss, self).__init__()
        self.is_single_label = True
        self.hash_targets = hash_center.to(args.device)
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(args.device)
        self.criterion = torch.nn.BCELoss().to(args.device)

    def forward(self, u, y, ind, args):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))
        Q_loss = (u.abs() - 1).pow(2).mean()

        return center_loss + args.lambd * Q_loss

    def label2center(self, y):
        hash_center = self.hash_targets[y.argmax(axis=1)]
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit) # 生成一个全1向量
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets


def train_val(args, hash_center, train_loader, test_loader, database_loader, num_database):
    print('==========start to train SHC NetWork==========')
    bit = args.code_length
    net = args.net(bit).to(args.device)

    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-7)

    hash_center = hash_center.to(torch.float32)
    if hash_center.shape != (args.num_classes, args.code_length):
        hash_center = hash_center.t()  # n_class * bit


    criterion = CSQLoss(args, bit, hash_center)

    result_dic = {}
    Best_mAP_ALL = 0
    Best_mAP_100 = 0
    Best_mAP_1000 = 0
    mAP_ALL_list = []
    mAP_100_list = []
    mAP_1000_list = []
    lr_values = []
    loss_list = []

    for epoch in range(0, args.epoch):

        this_lr = optimizer.param_groups[0]['lr']
        lr_values.append(this_lr)
        this_lr_str = "{:.5e}".format(this_lr)
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, Lr:%s, training...." % (
            args.info, epoch + 1, args.epoch, current_time, bit, args.dataset, this_lr_str), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(args.device)
            label = label.to(args.device)

            optimizer.zero_grad()

            u = net(image)

            # loss = criterion(u, label.float(), ind, config, D)
            loss = criterion(u, label.float(), ind, args)
            train_loss += loss.item()


            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()

        scheduler.step()

        train_loss = train_loss / len(train_loader)

        loss_list.append(train_loss)

        print("\b\b\b\b\b\b\b loss:%.5f" % (train_loss))

        if (epoch + 1) % args.test_map == 0:
            net.eval()
            with torch.no_grad():
                # Best_mAP = validate(args, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
                tst_binary, tst_label = compute_result(test_loader, net, device=args.device)
                trn_binary, trn_label = compute_result(database_loader, net, device=args.device)
                # mAP_list = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), args.topK)
                mAP_list, PR_data = CalcTopMapWithPR(tst_binary.numpy(), tst_label.numpy(),
                                                     trn_binary.numpy(), trn_label.numpy(),
                                                     args.topK, num_database)
                mAP_ALL = mAP_list[0]
                mAP_100 = mAP_list[1]
                mAP_1000 = mAP_list[2]
                mAP_ALL_list.append(mAP_ALL)
                mAP_100_list.append(mAP_100)
                mAP_1000_list.append(mAP_1000)

            if mAP_ALL > Best_mAP_ALL:
                Best_mAP_ALL = mAP_ALL
                best_net = copy.deepcopy(net)
            if mAP_100 > Best_mAP_100:
                Best_mAP_100 = mAP_100
            if mAP_1000 > Best_mAP_1000:
                Best_mAP_1000 = mAP_1000

            print(f"{args.info} epoch:{epoch + 1} bit:{bit} dataset:{args.dataset}")
            print(f"MAP ALL:{mAP_ALL} Best MAP ALL: {Best_mAP_ALL}")
            print(f"MAP 100:{mAP_100} Best MAP 100: {Best_mAP_100}")
            print(f"MAP 1000:{mAP_1000} Best MAP 1000: {Best_mAP_1000}")
        if (epoch + 1) % args.epoch == 0:
            print('[SHC] final_mAP_ALL:%.5f' % (Best_mAP_ALL))
            print('[SHC] final_mAP_100:%.5f' % (Best_mAP_100))
            print('[SHC] final_mAP_1000:%.5f' % (Best_mAP_1000))

    tst_binary, tst_label = compute_result(test_loader, best_net, device=args.device)
    trn_binary, trn_label = compute_result(database_loader, best_net, device=args.device)
    _, best_PR_data = CalcTopMapWithPR(tst_binary.numpy(), tst_label.numpy(),
                                         trn_binary.numpy(), trn_label.numpy(),
                                         args.topK, num_database)
    result_dic['loss'] = loss_list
    result_dic['mAP@all'] = mAP_ALL_list
    result_dic['mAP@100'] = mAP_100_list
    result_dic['mAP@1000'] = mAP_1000_list
    result_dic['PR_data'] = best_PR_data
    result_dic['net'] = best_net
    os.makedirs(f'./save/result_log/{args.dataset}', exist_ok=True)
    torch.save(result_dic, f'./save/result_log/{args.dataset}/SHC_bit_{bit}_result_dic.pt')


    plt.figure()
    plt.plot(list(range(epoch + 1)), lr_values)
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title('lr changes over epoch')

    plt.figure()
    # 绘制图形
    plt.plot(list(range(epoch + 1)), loss_list, label='loss')

    # 添加图例、标签等
    plt.legend()  # 显示图例
    plt.xlabel('epoch')  # x轴标签
    plt.ylabel('loss')  # y轴标签
    plt.title('loss changes over epoch')  # 图表标题
    plt.show()

    print('==========finish to train SHC NetWork==========')