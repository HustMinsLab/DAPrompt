# -*- coding: utf-8 -*-

# This project is for Deberta model.

import time
import os
import torch
import logging
from datetime import datetime
from parameter import parse_args
from sklearn.metrics import f1_score, precision_score, recall_score


torch.cuda.empty_cache()
args = parse_args()  # load parameters

if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.model):
    os.mkdir(args.model)




t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + 'fold-' + str(args.fold) + '__' + t + '.txt'
'''
 replace your perdiction path
'''

file='prediction_out/test_file'+str(args.fold)+'_new_prompt.txt'
file_out=open(file, "r")

# refine
for name in logging.root.manager.loggerDict:
    if 'transformers' in name:
        logging.getLogger(name).setLevel(logging.CRITICAL)

logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=args.log,
                    filemode='w')

logger = logging.getLogger(__name__)


def printlog(message: object, printout: object = True) -> object:
    message = '{}: {}'.format(datetime.now(), message)
    if printout:
        print(message)
    logger.info(message)


#
# file_out.close()
data=file_out.read()
data=data.split('\n')[:-1]
for i in range(len(data)):
    data[i]=data[i].split('\t')
for i in range(len(data)):
    for j in range(20):
        data[i][j]=float(data[i][j])
    data[i][j + 1] = int(data[i][j + 1]) # 20:event_place1
    data[i][j + 2] = int(data[i][j + 2]) # 21:nothing_place1
    data[i][j + 3] = int(data[i][j + 3]) # 22:event_place2
    data[i][j + 4] = int(data[i][j + 4]) # 23:nothing_place2
    data[i][j + 5] = int(data[i][j + 5]) # 24:label
    data[i][j + 6] = int(data[i][j + 6]) # 25:clabel

# event_place nothing_place label
label=[]
clabel=[]
NA_predt1=[]
NA_predt2=[]
prediction1=[]
prediction2=[]
for i in range(len(data)):
    if int(data[i][20]) == -1:
        prediction1.append(0)
    else:
        prediction1.append(data[i][int(data[i][20])])
    if int(data[i][21]) == -1:
        NA_predt1.append(0)
    else:
        NA_predt1.append(data[i][int(data[i][21])])
    if int(data[i][22]) == -1:
        prediction2.append(0)
    else:
        prediction2.append(data[i][int(data[i][22]) + 10])
    if int(data[i][23]) == -1:
        NA_predt2.append(0)
    else:
        NA_predt2.append(data[i][int(data[i][23]) + 10])
    label.append(data[i][24])
    clabel.append(data[i][25])







# calculate p, r, f1
def calculate(all_label_t, all_predt_t, all_clabel_t, epoch):
        exact_t = [0 for j in range(len(all_label_t))]
        for k in range(len(all_label_t)):
            if all_label_t[k] == 1 and all_label_t[k] == all_predt_t[k]:
                exact_t[k] = 1

        tpi = 0
        li = 0
        pi = 0
        tpc = 0
        lc = 0
        pc = 0

        for i in range(len(exact_t)):

            if exact_t[i] == 1:
                if all_clabel_t[i] == 0:
                    tpi += 1
                else:
                    tpc += 1

            if all_label_t[i] == 1:
                if all_clabel_t[i] == 0:
                    li += 1
                else:
                    lc += 1

            if all_predt_t[i] == 1:
                if all_clabel_t[i] == 0:
                    pi += 1
                else:
                    pc += 1

        printlog('\tINTRA-SENTENCE:')
        recli = tpi / li
        preci = tpi / (pi + 1e-9)
        f1cri = 2 * preci * recli / (preci + recli + 1e-9)

        intra = {
            'epoch': epoch,
            'p': preci,
            'r': recli,
            'f1': f1cri
        }
        printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(tpi, pi, li))
        printlog("\t\tprecision score: {}".format(intra['p']))
        printlog("\t\trecall score: {}".format(intra['r']))
        printlog("\t\tf1 score: {}".format(intra['f1']))

        # CROSS SENTENCE
        reclc = tpc / lc
        precc = tpc / (pc + 1e-9)
        f1crc = 2 * precc * reclc / (precc + reclc + 1e-9)
        cross = {
            'epoch': epoch,
            'p': precc,
            'r': reclc,
            'f1': f1crc
        }

        printlog('\tCROSS-SENTENCE:')
        printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(tpc, pc, lc))
        printlog("\t\tprecision score: {}".format(cross['p']))
        printlog("\t\trecall score: {}".format(cross['r']))
        printlog("\t\tf1 score: {}".format(cross['f1']))
        return tpi + tpc, pi + pc, li + lc, intra, cross

def judge5(prediction1,prediction2,sort_rate):
    predt = []
    for i in range(len(prediction1)):
        if prediction1[i] + prediction2[i] > sort_rate:
            predt.append(1)
        else:
            predt.append(0)
    return predt

sort_rate=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.0]


printlog('file:{}'.format(file))
for epoch in range(len(sort_rate)):
    printlog('plan：{}'.format(5))
    printlog('plan5: {}'.format(sort_rate[epoch]))

    predt = judge5(prediction1, prediction2, sort_rate[epoch])


    t_1, t_2, t_3, dev_intra, dev_cross = calculate(label, predt, clabel, 0)

    test_intra_cross = {
            'p': precision_score(label, predt, average=None)[1],
            'r': recall_score(label, predt, average=None)[1],
            'f1': f1_score(label, predt, average=None)[1]
        }

    printlog('\tINTRA + CROSS:')
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(t_1, t_2, t_3))
    printlog("\t\tprecision score: {}".format(test_intra_cross['p']))
    printlog("\t\trecall score: {}".format(test_intra_cross['r']))
    printlog("\t\tf1 score: {}".format(test_intra_cross['f1']))

    tp=0
    fp=0
    fn=0
    tn=0
    for i in range(len(predt)):
        if label[i]==1:
            if predt[i]==1:
                tp+=1
            if predt[i]==0:
                fn+=1
        if label[i]==0:
            if predt[i] == 1:
                fp += 1
            if predt[i] == 0:
                tn += 1


# print(data)