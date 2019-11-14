import argparse
import logging
import math
import os
import time

import editdistance
import kaldi_io
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from model import Transducer, RNNModel
from DataLoader import SequentialLoader, TokenAcc, rephone

parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Acoustic Model on TIMIT.')
parser.add_argument('model', help='trained model filename')
parser.add_argument('--beam', type=int, default=0, help='apply beam search, beam width')
parser.add_argument('--ctc', default=False, action='store_true', help='decode CTC acoustic model')
parser.add_argument('--bi', default=False, action='store_true', help='bidirectional LSTM')
parser.add_argument('--dataset', default='test', help='decoding data set')
parser.add_argument('--out', type=str, default='', help='decoded result output dir')
args = parser.parse_args()

logdir = args.out if args.out else os.path.dirname(args.model) + '/decode_13.log'
# if args.out: os.makedirs(args.out, exist_ok=True)
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt="%H:%M:%S", filename=logdir, level=logging.INFO)
# Load model
Model = RNNModel if args.ctc else Transducer
model = Model(40, 7531, 250, 2, bidirectional=args.bi)
model.load_state_dict(torch.load(args.model, map_location='cpu'))

use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()

# data set
feat = 'ark:copy-feats scp:data/{}/feats.scp ark:- | apply-cmvn --utt2spk=ark:data/{}/utt2spk scp:data/{}/cmvn.scp ark:- ark:- |\
 add-deltas --delta-order=0 ark:- ark:- | nnet-forward data/final.feature_transform ark:- ark:- |'.format(args.dataset, args.dataset, args.dataset)
with open('data/'+args.dataset+'/text', 'r', encoding='utf-8' ) as f:
    label = {}
    for line in f:
        line = line.split()
        label[line[0]] = line[1:]

with open('data/words.txt', 'r', encoding='utf-8') as f:
    wmap={}
    for line in f:
        line = line.split()
        wmap[int(line[1])] = line[0]

def distance(y, t, blank='<s>'):
    def remap(y, blank):
        prev = blank
        seq = []
        for i in y:
            if i != blank and i != prev: seq.append(i)
            prev = i
        return seq
    y = remap(y, blank)
    t = remap(t, blank)
    return y, t, editdistance.eval(y, t)

model.eval()
def decode():
    logging.info('Decoding transduction model:')
    err = cnt = 0
    for k, v in kaldi_io.read_mat_ark(feat):
        with torch.no_grad():
            xs = Variable(torch.FloatTensor(v[None, ...]), volatile=True)
            if use_gpu:
                xs = xs.cuda()
            if args.beam > 0:
                y, nll = model.beam_search(xs, args.beam)
            else:
                y, nll = model.greedy_decode(xs)
            y = [wmap[i] for i in y]
            t = label[k]
            y, t, e = distance(y, t)
            err += e; cnt += len(t)
            logging.info('utt-id: %s' % k)
            logging.info('Ref: {}'.format(' '.join(t)))
            logging.info('Hyp: {}'.format(' '.join(y)))
            logging.info('Log-Likelihood: {:.2f}'.format(nll))
            logging.info('-' * 80)
    logging.info('{} set {} CER {:.2f}%'.format(args.dataset.capitalize(), 'CTC' if args.ctc else 'Transducer', 100*err/cnt))
    
decode()
