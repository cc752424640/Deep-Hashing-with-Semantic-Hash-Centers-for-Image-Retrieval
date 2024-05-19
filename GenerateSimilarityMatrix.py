import os.path
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
from utils.tools import *
from network import *
from loguru import logger

def TrainClassificationNetwork(args, train_loader, test_loader):
    print('==========start to generate ClassificationNetwork==========')
    net = ClassifyNet(args.num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(
        net.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.classify_epoch)

    train_total = 0
    train_correct = 0
    test_total = 0
    test_correct = 0
    running_loss = 0.
    best_pre = 0.
    for epoch in range(args.classify_epoch):
        tic = time.time()
        this_lr = optimizer.param_groups[0]['lr']
        this_lr_str = "{:.5e}".format(this_lr)
        net.train()
        for data, targets, index in train_loader:
            # print(data.shape)
            targets = targets.to(torch.float32)
            data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
            optimizer.zero_grad()

            _, pre_label = net(data)
            _, pre_true_label = torch.max(pre_label, 1)
            train_total += targets.size(0)
            _, true_targets = torch.max(targets, 1)
            train_correct += (pre_true_label == true_targets).sum().item()

            loss = criterion(pre_label, targets)
            running_loss = running_loss + loss.item()
            loss.backward()
            optimizer.step()
        train_pre = train_correct / train_total
        scheduler.step()

        if epoch % args.test_map == args.test_map - 1:
            training_time = time.time() - tic
            tic = time.time()
            net.eval()
            with torch.no_grad():
                for data, targets, index in test_loader:
                    data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
                    _, pre_true_label = torch.max(net(data)[1], 1)
                    _, true_targets = torch.max(targets, 1)
                    test_total += targets.size(0)
                    test_correct += (pre_true_label == true_targets).sum().item()
                test_pre = test_correct / test_total
            if test_pre > best_pre:
                best_pre = test_pre
            testing_time = time.time() - tic
            logger.info('[iter:{}/{}][dataset:{}][lr:{}][loss:{:.2f}][train_pre:{:.4f}%][test_pre:{:.4f}%][best_pre:{:.4f}%][training_time:{:.2f}][testing_time:{:.2f}]'.format(
                epoch + 1,
                args.classify_epoch,
                args.dataset,
                this_lr_str,
                running_loss / args.test_map,
                100 * train_pre,
                100 * test_pre,
                100 * best_pre,
                training_time,
                testing_time,
            ))
            running_loss = 0.
    os.makedirs(f'./save/ClassificationNet/', exist_ok=True)
    torch.save(net, f'./save/ClassificationNet/{args.dataset}_ClassificationNet.pt')
    print('==========success generate ClassificationNetwork==========')
    return net
def GenerateSimilarityMatrix(args, train_loader, test_loader):
    if os.path.exists(f'./save/ClassificationNet/{args.dataset}_ClassificationNet.pt'):
        print('==========ClassificationNet has already generated==========')
        net = torch.load(f'./save/ClassificationNet/{args.dataset}_ClassificationNet.pt').to(args.device)
    else:
        net = TrainClassificationNetwork(args, train_loader, test_loader)
    print('==========start to generate SimilarityMatrix==========')
    S = torch.zeros(args.num_classes, args.num_classes).to(args.device)
    net.eval()
    with torch.no_grad():
        for data, targets, index in train_loader:
            data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
            batch_size = targets.shape[0]
            p_dis, _ = net(data) # p_dis就是p_0
            _, true_targets = torch.max(targets, 1) # 获得要mask的j，矩阵形式
            for i in range(batch_size):
                tmp = p_dis[i].clone()
                tmp[true_targets[i]] = float('-inf') # 对最大值做mask（变成-INF）
                S[true_targets[i]] += F.softmax(tmp)

    mask = torch.eye(args.num_classes).bool()
    S = (S + S.T) / 2
    for i in range(args.num_classes):
        S_max = S[i].max()
        S_min = S[i].min()
        S_mean = S[i].mean()
        S[i] = (S[i] - S_mean) / max(abs(S_max - S_mean), abs(S_min - S_mean))
    S[mask] = 1
    os.makedirs(f'./save/SimilarityMatrix/', exist_ok=True)
    torch.save(S, f'./save/SimilarityMatrix/{args.dataset}_Similarity_Matrix.pt')
    print('==========success generate SimilarityMatrix==========')
    return S
