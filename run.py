import argparse
import os.path

import torch
from network import *
from data.data_loader import load_data
from GenerateSimilarityMatrix import GenerateSimilarityMatrix
from GenerateSemanticHashCenters import GenerateSemanticHashCenters
from train import train_val


def load_config():
    parser = argparse.ArgumentParser(description='SHC_PyTorch')
    parser.add_argument('--seed', default=60, type=int,
                        help='seed')
    parser.add_argument('--info', default='[SHC]', type=str,
                        help='information')
    parser.add_argument('--dataset', default='cifar-100-new-seg', type=str,
                        help='Dataset name.(default: cifar-100-new-seg, stanford_cars-new-seg, stanford_cars-official-seg, NAbirds-new-seg, NAbirds-official-seg,')
    parser.add_argument('--num-classes', default=100, type=int,
                        help='num classes of dataset.(default: 100)')
    parser.add_argument('--root', default='../data/cifar/cifar-100/cifar-100-new-seg/', type=str,
                        help='Path of dataset')
    parser.add_argument('--code-length', default=32, type=int,
                        help='Binary hash code length.(default: 32)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--epoch', default=100, type=int,
                        help='max epoch.(default: 100)')
    parser.add_argument('--classify_epoch', default=100, type=int,
                        help='max epoch for classification.(default: 100)')
    parser.add_argument('--test-map', default=5, type=int,
                        help='test frequency.(default: 10)')
    parser.add_argument('--beta', default=1.00, type=float,
                        help='para')
    parser.add_argument('--lambd', default=0.0001, type=float,
                        help='para')
    parser.add_argument('--topK', default=[-1, 100, 1000], type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size.(default: 128)')
    parser.add_argument('--num-workers', default=6, type=int,
                        help='Number of loading data threads.(default: 6)')
    parser.add_argument('--resize-size', default=256, type=int,
                        help='picture resize size.(default: 256)')
    parser.add_argument('--crop-size', default=224, type=int,
                        help='picture crop size.(default: 224)')
    parser.add_argument('--gpu', default=4, type=int,
                        help='Using gpu.(default: False)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # net
    args.net = ResNet

    return args


if __name__ == '__main__':
    args = load_config()
    # load data
    train_loader, test_loader, database_loader, num_train, num_test, num_database = load_data(args)

    # Stage1：Construct the Data-dependent Pairwise Similarity Matrix
    if os.path.exists(f'./save/SimilarityMatrix/{args.dataset}_Similarity_Matrix.pt'):
        print('==========SimilarityMatrix has already generated==========')
        S = torch.load(f'./save/SimilarityMatrix/{args.dataset}_Similarity_Matrix.pt')
    else:
        S = GenerateSimilarityMatrix(args, train_loader, test_loader)
    S = S.to(args.device)

    # Stage 2: Generate the Semantic Hash Centers
    if os.path.exists(f'./save/HashCenters/{args.dataset}_SHC_HashCenters_bit_{args.code_length}.pt'):
        print('==========SHC HashCenters has already generated==========')
        H = torch.load(f'./save/HashCenters/{args.dataset}_SHC_HashCenters_bit_{args.code_length}.pt')
    H = GenerateSemanticHashCenters(args, S)
    H = H.to(args.device)

    # Stage 3: Train the Deep Hashing Network
    train_val(args, H, train_loader, test_loader, database_loader, num_database)