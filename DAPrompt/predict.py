# -*- coding: utf-8 -*-

# This project is for Deberta model.

import time
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import tokenizers
import torch
import torch.nn as nn
import logging
import tqdm
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from load_data import load_data
from transformers import DebertaTokenizer, AdamW
from parameter import parse_args
from util import convert

from model import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
torch.cuda.empty_cache()
args = parse_args()  # load parameters

if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.model):
    os.mkdir(args.model)


t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + 'fold-planB1.40_' + str(args.fold) + '__' + t + '.txt'
'''
 replace your model path and you will get 2 prediction files
'''
args.model = '  '

# args.batch_size=4

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

# set seed for random number
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

setup_seed(args.seed)

# load Roberta model
printlog('Passed args:')
printlog('log path: {}'.format(args.log))
printlog('transformer model: {}'.format(args.model_name))

tokenizer = DebertaTokenizer.from_pretrained(args.model_name)

# load data tsv file
printlog('Loading data')

train_data, dev_data, test_data = load_data(args)
train_size = len(train_data)
dev_size = len(dev_data)
test_size = len(test_data)
print('Data loaded')


# In ESC data sets, some event-mentions are not continuous.
# put ...  on ---> put Tompsion on
def isContinue(id_list):
    for i in range(len(id_list)-1):
        if int(id_list[i])!=int(id_list[i+1])-1:
            return False
    return True

def correct_data(data):
    for i in range(len(data)):
        e1_id = data[i][14].split('_')[1:]
        e2_id = data[i][15].split('_')[1:]
        if not isContinue(e1_id):
            s_1 = data[i][10].split()
            event1=s_1[int(e1_id[0]):int(e1_id[-1])+1]
            event1=' '.join(event1)
            event1+=' '
            new_e1_id=[str(i) for i in range(int(e1_id[0]),int(e1_id[-1])+1)]
            event_place1='_'+'_'.join(new_e1_id)
            sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], event1, data[i][8],data[i][9], data[i][10],
                        data[i][11], data[i][12], data[i][13], event_place1, data[i][15])
            data.pop(i)
            data.insert(i, sentence)
        if not isContinue(e2_id):
            s_2 = data[i][12].split()
            event2=s_2[int(e2_id[0]):int(e2_id[-1])+1]
            event2=' '.join(event2)
            event2+=' '
            new_e2_id=[str(i) for i in range(int(e2_id[0]),int(e2_id[-1])+1)]
            event_place2='_'+'_'.join(new_e2_id)
            sentence=(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7],event2,data[i][9],data[i][10],
                      data[i][11],data[i][12],data[i][13],data[i][14],event_place2)
            data.pop(i)
            data.insert(i, sentence)
    return data

train_data=correct_data(train_data)
dev_data=correct_data(dev_data)
test_data=correct_data(test_data)


# collect multi-token event-mentions
def collect_mult_event(train_data,tokenizer):
    multi_event=[]
    to_add={}
    special_multi_event_token=[]
    event_dict={}
    reverse_event_dict={}
    for sentence in train_data:
        if len(tokenizer(' '+sentence[7].strip())['input_ids'][1:-1])>1 and sentence[7] not in multi_event:
            multi_event.append(sentence[7])
            special_multi_event_token.append("<a_"+str(len(special_multi_event_token))+">")
            event_dict[special_multi_event_token[-1]]=multi_event[-1]
            reverse_event_dict[multi_event[-1]]=special_multi_event_token[-1]
            to_add[special_multi_event_token[-1]]=tokenizer(multi_event[-1].strip())['input_ids'][1: -1]
        if len(tokenizer(' '+sentence[8].strip())['input_ids'][1:-1])>1 and sentence[8] not in multi_event:
            multi_event.append(sentence[8])
            special_multi_event_token.append("<a_"+str(len(special_multi_event_token))+">")
            event_dict[special_multi_event_token[-1]]=multi_event[-1]
            reverse_event_dict[multi_event[-1]]=special_multi_event_token[-1]
            to_add[special_multi_event_token[-1]] = tokenizer(multi_event[-1].strip())['input_ids'][1: -1]

    return multi_event,special_multi_event_token,event_dict,reverse_event_dict,to_add

multi_event,special_multi_event_token,event_dict,reverse_event_dict,to_add=collect_mult_event(train_data+dev_data+test_data,tokenizer)

additional_mask=['[mask2]','<c>','<c2>','</c>','</c2>','<na>']     #50265、50266、50267、50268、50269、50270
to_add[additional_mask[0]]=[50264]
to_add[additional_mask[5]]=[i for i in range(50265)]   # initialization
tokenizer.add_tokens(additional_mask)           #5
tokenizer.add_tokens(special_multi_event_token) #516
args.vocab_size = len(tokenizer)                #50265+5+516


# Replace multi-token events with special characters <A-i>
# For example：He has went to the school.--->He <A_3> the school.
def replace_mult_event(data,reverse_event_dict):
    for i in range(len(data)):
        if (data[i][7] in reverse_event_dict) and (data[i][8] not in reverse_event_dict):
            s_1 = data[i][10].split()
            e1_id = data[i][14].split('_')[1:]
            e1_id.reverse()
            for id in e1_id:
                s_1.pop(int(id))
            s_1.insert(int(e1_id[-1]), reverse_event_dict[data[i][7]])
            sentence=(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],reverse_event_dict[data[i][7]],data[i][8],data[i][9]," ".join(s_1),
                          data[i][11],data[i][12],data[i][13],'_'+e1_id[-1],data[i][15])
            data.pop(i)
            data.insert(i,sentence)
        if (data[i][7] not in reverse_event_dict) and (data[i][8] in reverse_event_dict):
            s_2 = data[i][12].split()
            e2_id = data[i][15].split('_')[1:]
            e2_id.reverse()
            for id in e2_id:
                s_2.pop(int(id))
            s_2.insert(int(e2_id[-1]), reverse_event_dict[data[i][8]])
            sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7], reverse_event_dict[data[i][8]],data[i][9], data[i][10],
                            data[i][11], " ".join(s_2), data[i][13], data[i][14], '_'+e2_id[-1])
            data.pop(i)
            data.insert(i, sentence)
        if (data[i][7] in reverse_event_dict) and (data[i][8] in reverse_event_dict):
            e1_id = data[i][14].split('_')[1:]
            e2_id = data[i][15].split('_')[1:]
            e1_id.reverse()
            e2_id.reverse()
            if data[i][11] == data[i][13]:
                s = data[i][10].split()
                if int(e1_id[0])<int(e2_id[0]):
                    for id in e2_id:
                        s.pop(int(id))
                    s.insert(int(e2_id[-1]), reverse_event_dict[data[i][8]])
                    for id in e1_id:
                        s.pop(int(id))
                    s.insert(int(e1_id[-1]), reverse_event_dict[data[i][7]])
                else:
                    for id in e1_id:
                        s.pop(int(id))
                    s.insert(int(e1_id[-1]), reverse_event_dict[data[i][7]])
                    for id in e2_id:
                        s.pop(int(id))
                    s.insert(int(e2_id[-1]), reverse_event_dict[data[i][8]])
                sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], reverse_event_dict[data[i][7]],reverse_event_dict[data[i][8]],data[i][9], " ".join(s),
                            data[i][11], " ".join(s), data[i][13], '_'+str(s.index(reverse_event_dict[data[i][7]])), '_'+str(s.index(reverse_event_dict[data[i][8]])))
                data.pop(i)
                data.insert(i, sentence)
            if data[i][11] != data[i][13]:
                s_1 = data[i][10].split()
                for id in e1_id:
                    s_1.pop(int(id))
                s_1.insert(int(e1_id[-1]), reverse_event_dict[data[i][7]])

                s_2 = data[i][12].split()
                for id in e2_id:
                    s_2.pop(int(id))
                s_2.insert(int(e2_id[-1]), reverse_event_dict[data[i][8]])
                sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], reverse_event_dict[data[i][7]],reverse_event_dict[data[i][8]], data[i][9], " ".join(s_1),
                            data[i][11], " ".join(s_2), data[i][13], '_'+str(s_1.index(reverse_event_dict[data[i][7]])), '_'+str(s_2.index(reverse_event_dict[data[i][8]])))
                data.pop(i)
                data.insert(i, sentence)
    return data


train_data = replace_mult_event(train_data,reverse_event_dict)
dev_data = replace_mult_event(dev_data,reverse_event_dict)
test_data = replace_mult_event(test_data,reverse_event_dict)



# tokenize sentence and get event idx
def get_batch(arg, indices):
    batch_idx = []
    batch_mask = []
    mask_indices = []
    casual_label = []
    label_b = []
    label_c = []
    clabel_b = []
    for idx in indices:
        label = 1
        label_1 = 1
        label_2 = 1
        clabel_1 = 1
        e1_id = arg[idx][14]
        e2_id = arg[idx][15]
        sentence1_id = arg[idx][11]
        sentence2_id = arg[idx][13]
        s_1 = arg[idx][10]
        s_2 = arg[idx][12]
        s_1 = s_1.split()[0:int((args.len_arg-args.len_temp)/2)]
        s_2 = s_2.split()[0:int((args.len_arg-args.len_temp)/2)]
        e1_id = e1_id.split("_")
        e2_id = e2_id.split("_")
        if sentence1_id == sentence2_id:
            clabel_1 = 0
            temp = 'There is a causal relation between [MASK] and [mask2] .'
            if int(e1_id[1]) > int(e2_id[1]):
                s_1.insert(int(e1_id[1]), '<c2>')
                s_1.insert(int(e1_id[1]) + len(e1_id), '</c2>')
                s_1.insert(int(e2_id[1]), '<c>')
                s_1.insert(int(e2_id[1]) + len(e2_id), '</c>')
            else:
                s_1.insert(int(e2_id[1]) , '<c>')
                s_1.insert(int(e2_id[1]) + len(e2_id), '</c>')
                s_1.insert(int(e1_id[1]), '<c2>')
                s_1.insert(int(e1_id[1]) + len(e1_id), '</c2>')
            s_1 = " ".join(s_1)
            encode_dict = tokenizer.encode_plus(
                s_1,
                text_pair=temp,
                add_special_tokens=True,
                padding='max_length',
                max_length=args.len_arg,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        else:
            s_1.insert(int(e1_id[1]), '<c2>')
            s_1.insert(int(e1_id[1]) + len(e1_id), '</c2>')
            s_2.insert(int(e2_id[1]), '<c>')
            s_2.insert(int(e2_id[1]) + len(e2_id), '</c>')
            s_1 = " ".join(s_1)
            s_2 = " ".join(s_2)
            temp = 'There is a causal relation between [MASK] and [mask2] .'
            if sentence1_id < sentence2_id:
                encode_dict = tokenizer.encode_plus(
                    s_1 + '[SEP]' + s_2,
                    text_pair=temp,
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=args.len_arg,
                    truncation=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
            else:
                encode_dict = tokenizer.encode_plus(
                    s_2 + '[SEP]' + s_1,
                    text_pair=temp,
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=args.len_arg,
                    truncation=True,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
        arg_1_idx = encode_dict['input_ids']
        arg_1_mask = encode_dict['attention_mask']
        if arg[idx][9] == 'NONE':
            label = 0
            label_1 = tokenizer('<na>')['input_ids'][1:-1] # 50270
            label_2 = tokenizer('<na>')['input_ids'][1:-1] # 50270
        if arg[idx][9] != 'NONE':
            label_1 = tokenizer('<c2>')['input_ids'][1:-1] # 50267
            label_2 = tokenizer('<c>')['input_ids'][1:-1]  # 50266
        label_b+=label_1
        label_c+=label_2
        casual_label.append(label)
        clabel_b.append(clabel_1)
        if len(batch_idx) == 0:
            batch_idx = arg_1_idx
            batch_mask = arg_1_mask
            mask_list=[]
            mask_list.append(torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1].item()) # [MASK1]
            mask_list.append(torch.nonzero(arg_1_idx == 50265, as_tuple=False)[0][1].item()) # [MASK2]
            mask_indices = torch.unsqueeze(torch.tensor(mask_list),dim=0)
        else:
            batch_idx = torch.cat((batch_idx, arg_1_idx), dim=0)
            batch_mask = torch.cat((batch_mask, arg_1_mask), dim=0)
            mask_list = []
            mask_list.append(torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1].item())
            mask_list.append(torch.nonzero(arg_1_idx == 50265, as_tuple=False)[0][1].item())
            mask_indices = torch.cat((mask_indices, torch.unsqueeze(torch.tensor(mask_list), dim=0)), dim=0)
    return batch_idx, batch_mask, casual_label, label_b, label_c, clabel_b, mask_indices


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

# ---------- network ----------

net = MLP(args).to(device)
net.handler(to_add, tokenizer)

dict_trained = torch.load(args.model)
net.deberta_model.load_state_dict(dict_trained['deberta_model'])
net.deberta_model2.load_state_dict(dict_trained['deberta_model2'])

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
    {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.t_lr)


cross_entropy = nn.CrossEntropyLoss().to(device)

# save model and result
best_intra = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
best_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
best_intra_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
dev_best_intra_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
state = {}

best_epoch = 0

printlog('fold: {}'.format(args.fold))
printlog('batch_size:{}'.format(args.batch_size))
printlog('epoch_num: {}'.format(args.num_epoch))
printlog('initial_t_lr: {}'.format(args.t_lr))
printlog('sample_rate: {}'.format(args.sample_rate))
printlog('sort rate: {}'.format(args.rate_sort))
printlog('seed: {}'.format(args.seed))
printlog('mlp_size: {}'.format(args.mlp_size))
printlog('mlp_drop: {}'.format(args.mlp_drop))
printlog('wd: {}'.format(args.wd))
printlog('len_arg: {}'.format(args.len_arg))
printlog('len_temp: {}'.format(args.len_temp))

printlog('Start training ...')
breakout = 0

args.num_epoch=1
##################################  epoch  #################################
for epoch in range(args.num_epoch):
    print('=' * 20)
    printlog('Epoch: {}'.format(epoch))
    torch.cuda.empty_cache()

    loss_epoch = 0.0
    acc = 0.0
    all_label_ = []
    all_label_1=[]
    all_label_2= []
    all_predt_ = []
    all_clabel_ = []
    f1_pred = torch.IntTensor([]).to(device)
    f1_truth = torch.IntTensor([]).to(device)

    start = time.time()

    printlog('lr:{}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
    printlog('t_lr:{}'.format(optimizer.state_dict()['param_groups'][1]['lr']))

    ############################################################################
    ##################################  train  #################################
    ############################################################################
    test_file = open('./' + 'test_file' + str(args.fold) + '_new_prompt.txt', "w")
    dev_file = open('./' + 'dev_file' + str(args.fold) + '_new_prompt.txt', "w")

    ############################################################################
    ##################################  dev  ###################################
    ############################################################################
    all_indices = torch.tensor(list(range(dev_size))).split(args.batch_size)
    all_label = []
    all_predt = []
    all_clabel = []

    progress = tqdm.tqdm(total=len(dev_data) // args.batch_size + 1, ncols=75,
                         desc='Eval {}'.format(epoch))

    net.eval()
    for batch_indices in all_indices:
        progress.update(1)

        # get a batch of dev_data
        batch_arg, mask_arg, label, label1, label2, clabel, mask_indices = get_batch(dev_data, batch_indices)

        batch_arg = batch_arg.to(device)
        mask_arg = mask_arg.to(device)
        mask_indices = mask_indices.to(device)
        length = len(batch_indices)
        # fed data into network
        prediction1,prediction2 = net(batch_arg, mask_arg, mask_indices, length)
        temp_predict1 = torch.softmax(prediction1, dim=1)
        temp_predict2 = torch.softmax(prediction2, dim=1)

        predt1 = torch.argmax(prediction1, dim=1).detach()
        predt2 = torch.argmax(prediction2, dim=1).detach()
        predt = []
        for iii in range(len(predt1)):
            # if temp_predict1[iii][50267]+temp_predict2[iii][50266]>0.5:
            if predt1[iii] == 50267 and predt2[iii] == 50266:
                predt.append(1)
            else:
                predt.append(0)

        all_label += label
        all_predt += predt
        all_clabel += clabel

        to_save_prediction1, event_place1, nothing_place1 = convert(prediction1, 50267)
        to_save_prediction2, event_place2, nothing_place2 = convert(prediction2, 50266)
        topic_id=[]
        document_id=[]
        for temp_i in range(len(predt)):
            topic_id.append(dev_data[batch_indices[temp_i]][1])
            document_id.append(dev_data[batch_indices[temp_i]][2])

        # Save the prediction
        for predt_i in range(len(prediction1)):
            for predt_j in range(len(to_save_prediction1[predt_i])):
                dev_file.write(str(to_save_prediction1[predt_i][predt_j]) + '\t')
            for predt_j in range(len(to_save_prediction2[predt_i])):
                dev_file.write(str(to_save_prediction2[predt_i][predt_j]) + '\t')
            dev_file.write(str(event_place1[predt_i]) + '\t')
            dev_file.write(str(nothing_place1[predt_i]) + '\t')
            dev_file.write(str(event_place2[predt_i]) + '\t')
            dev_file.write(str(nothing_place2[predt_i]) + '\t')
            dev_file.write(str(label[predt_i]) + '\t')
            dev_file.write(str(clabel[predt_i]) + '\t')
            dev_file.write(str(topic_id[predt_i]) + '\t')
            dev_file.write(str(document_id[predt_i] + '\n'))

    progress.close()

    ############################################################################
    ##################################  test  ##################################
    ############################################################################
    all_indices = torch.tensor(list(range(test_size))).split(args.batch_size)
    all_label_t = []
    all_predt_t = []
    all_clabel_t = []
    acc = 0.0

    progress = tqdm.tqdm(total=len(test_data) // args.batch_size + 1, ncols=75,
                         desc='Eval {}'.format(epoch))

    net.eval()
    for batch_indices in all_indices:
        progress.update(1)

        # get a batch of dev_data
        batch_arg, mask_arg, label, label1, label2, clabel, mask_indices = get_batch(test_data, batch_indices)

        batch_arg = batch_arg.to(device)
        mask_arg = mask_arg.to(device)
        mask_indices = mask_indices.to(device)
        length = len(batch_indices)

        all_clabel_t += clabel
        # fed data into network
        prediction1,prediction2 = net(batch_arg, mask_arg, mask_indices, length)
        temp_predict1 = torch.softmax(prediction1, dim=1)
        temp_predict2 = torch.softmax(prediction2, dim=1)

        predt1 = torch.argmax(prediction1, dim=1).detach()
        predt2 = torch.argmax(prediction2, dim=1).detach()
        predt = []
        for iii in range(len(predt1)):
            if temp_predict1[iii][50267] + temp_predict2[iii][50266] > 0.5:
            # if predt1[iii] == 50267 and predt2[iii] == 50266:
                predt.append(1)
            else:
                predt.append(0)

        all_label_t += label
        all_predt_t += predt

        # Save the prediction
        to_save_prediction1, event_place1, nothing_place1 = convert(prediction1, 50267)
        to_save_prediction2, event_place2, nothing_place2 = convert(prediction2, 50266)
        topic_id = []
        document_id = []
        for temp_i in range(len(predt)):
            topic_id.append(test_data[batch_indices[temp_i]][1])
            document_id.append(test_data[batch_indices[temp_i]][2])

        for predt_i in range(len(prediction1)):
            for predt_j in range(len(to_save_prediction1[predt_i])):
                test_file.write(str(to_save_prediction1[predt_i][predt_j]) + '\t')
            for predt_j in range(len(to_save_prediction2[predt_i])):
                test_file.write(str(to_save_prediction2[predt_i][predt_j]) + '\t')
            test_file.write(str(event_place1[predt_i]) + '\t')
            test_file.write(str(nothing_place1[predt_i]) + '\t')
            test_file.write(str(event_place2[predt_i]) + '\t')
            test_file.write(str(nothing_place2[predt_i]) + '\t')
            test_file.write(str(label[predt_i]) + '\t')
            test_file.write(str(clabel[predt_i]) + '\t')
            test_file.write(str(topic_id[predt_i]) + '\t')
            test_file.write(str(document_id[predt_i] + '\n'))


    progress.close()

    ############################################################################
    ##################################  result  ##################################
    ############################################################################
    ######### Train Results Print #########

    ######### Dev Results Print #########
    printlog("DEV:")
    d_1, d_2, d_3, dev_intra, dev_cross = calculate(all_label, all_predt, all_clabel, epoch)
    # INTRA + CROSS SENTENCE
    dev_intra_cross = {
        'epoch': epoch,
        'p': precision_score(all_label, all_predt, average=None)[1],
        'r': recall_score(all_label, all_predt, average=None)[1],
        'f1': f1_score(all_label, all_predt, average=None)[1]
    }

    printlog('\tINTRA + CROSS:')
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(d_1, d_2, d_3))
    printlog("\t\tprecision score: {}".format(dev_intra_cross['p']))
    printlog("\t\trecall score: {}".format(dev_intra_cross['r']))
    printlog("\t\tf1 score: {}".format(dev_intra_cross['f1']))

    ######### Dev Results Print #########
    printlog("TEST:")
    t_1, t_2, t_3, test_intra, test_cross = calculate(all_label_t, all_predt_t, all_clabel_t, epoch)

    # INTRA + CROSS SENTENCE
    test_intra_cross = {
        'epoch': epoch,
        'p': precision_score(all_label_t, all_predt_t, average=None)[1],
        'r': recall_score(all_label_t, all_predt_t, average=None)[1],
        'f1': f1_score(all_label_t, all_predt_t, average=None)[1]
    }
    printlog('\tINTRA + CROSS:')
    printlog("\t\tTest Acc={:.4f}".format(acc / test_size))
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(t_1, t_2, t_3))
    printlog("\t\tprecision score: {}".format(test_intra_cross['p']))
    printlog("\t\trecall score: {}".format(test_intra_cross['r']))
    printlog("\t\tf1 score: {}".format(test_intra_cross['f1']))

    breakout += 1

    # record the best result
    if dev_intra_cross['f1'] > dev_best_intra_cross['f1']:
        printlog('New best epoch...')
        dev_best_intra_cross = dev_intra_cross
        best_intra_cross = test_intra_cross
        best_intra = test_intra
        best_cross = test_cross
        best_epoch = epoch
        breakout = 0
        state = {'deberta_model': net.deberta_model.state_dict(),
                 'deberta_model2': net.deberta_model2.state_dict()}
        # torch.save(state, args.model)
        # time.sleep(15)

    printlog('=' * 20)
    printlog('Best result at epoch: {}'.format(best_epoch))
    printlog('Eval intra: {}'.format(best_intra))
    printlog('Eval cross: {}'.format(best_cross))
    printlog('Eval intra cross: {}'.format(best_intra_cross))
    printlog('Breakout: {}'.format(breakout))

    if breakout == 3:
        break
    test_file.close()
    dev_file.close()

# torch.save(state, args.model)
