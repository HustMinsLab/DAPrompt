# -*- coding: utf-8 -*-

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ECI')

    # dataset
    parser.add_argument('--sample_rate', default=0.2, type=float, help='Negetive sample rate')  # Negative sampling ratio
    parser.add_argument('--cluster_num', default=6, type=int, help='cluster number')
    parser.add_argument('--rate_sort', default=0.35, type=float, help='Judge rate')
    parser.add_argument('--fold',         default=5,        type=int, help='Fold number')

    # # model arguments
    parser.add_argument('--model_name',   default='your deberta model path', type=str, help='Log model name')
    parser.add_argument('--vocab_size',   default=50265,    type=int, help='Size of deBERTa vocab')
    parser.add_argument('--len_arg',      default=200,      type=int, help='Sentence length')
    parser.add_argument('--len_temp',     default=0,      type=int, help='template length') 
    parser.add_argument('--mlp_size',     default=200,      type=int, help='mlp layer_size')
    parser.add_argument('--mlp_drop',     default=0.4,      type=int, help='mlp dropout layer')

    # # training arguments
    parser.add_argument('--seed',         default=209,      type=int, help='seed for reproducibility')
    parser.add_argument('--batch_size',   default=16,       type=int, help='batchsize for optimizer updates')
    parser.add_argument('--wd',           default=1e-2,     type=float, help='weight decay')  # 1e-3

    parser.add_argument('--num_epoch',    default=15,      type=int, help='number of total epochs to run')
    parser.add_argument('--t_lr',         default=1e-5,     type=float, help='initial transformer learning rate')

    parser.add_argument('--log',       default='./out/',  type=str, help='Log result file name')
    parser.add_argument('--model',       default='./outmodel/',  type=str, help='Log result file name')
    
    args = parser.parse_args()
    return args
