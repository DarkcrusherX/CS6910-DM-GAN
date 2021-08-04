from __future__ import print_function
from datasets import TextDataset
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz

import numpy as np

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


CONFIG_NAME = 'DMGAN'
DATASET_NAME = 'birds'
DATA_DIR = '../data/birds'
GPU_ID = 0
WORKERS = 4
BRANCH_NUM = 3
FLAG = True
NET_G= ''
B_NET_D= True
B_VALIDATION = False
BATCH_SIZE= 2
MAX_EPOCH = 800
SNAPSHOT_INTERVAL = 25
DISCRIMINATOR_LR = 0.0002
GENERATOR_LR = 0.0002
NET_E = '../DAMSMencoders/bird/text_encoder200.pth'
GAMMA1 = 4.0
GAMMA2 = 5.0
GAMMA3 = 10.0
LAMBDA = 5.0
DF_DIM = 32
GF_DIM = 64
Z_DIM = 100
R_NUM = 2
EMBEDDING_DIM = 256
CAPTIONS_PER_IMAGE = 10
CUDA = True
BASE_SIZE = 64
ENCODER_LR: 0.0002
RNN_GRAD_CLIP: 0.25
WORDS_NUM = 2 
RNN_TYPE = 'LSTM'
CONDITION_DIM = 100
B_ATTENTION = True
B_DCGAN = False


def gen_example(wordtoix, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '%s/example_filenames.txt' % (DATA_DIR)
    data_dic = {}
    with open(filepath, "r") as f:
        filenames = f.read().decode('utf8').split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (DATA_DIR, name)
            with open(filepath, "r") as f:
                print('Load from:', name)
                sentences = f.read().decode('utf8').split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)

if __name__ == "__main__":


    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if CUDA:
        torch.cuda.manual_seed_all(manualSeed)
    torch.cuda.set_device(GPU_ID)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print("Seed: %d" % (manualSeed))

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (DATASET_NAME,CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = BASE_SIZE * (2 ** (BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(DATA_DIR, split_dir,
                          base_size=BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword, dataset)

    start_t = time.time()
    if FLAG:
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if B_VALIDATION:
            algo.sampling(split_dir)  # generate images for the whole valid dataset
        else:
            gen_example(dataset.wordtoix, algo)  # generate images for customized captions
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
