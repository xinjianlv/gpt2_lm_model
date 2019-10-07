import os , pdb
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader, TensorDataset


PADDED_INPUTS = ['input_ids' , 'label_ids']


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset[PADDED_INPUTS[0]])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def get_data_loaders(data_file , tokenizer , batch_size ,train_precent = 0.7 , n_ctx = 1024):
    fr = open(data_file)
    data_set = {PADDED_INPUTS[0]:[] , PADDED_INPUTS[1]:[]}
    paragraph = []
    lines = fr.read().splitlines()[0:100]
    for line in tqdm(lines):
        if line != '':
            paragraph.append(line)
        else:
            line = '[CLS]'
            for l in paragraph:
                if len(line) + len(l) < n_ctx:
                    line +=l
                else:
                    break
            line += '[SEP]'

            indexed_tokens = tokenizer.encode(line)
            data_set[PADDED_INPUTS[0]].append(indexed_tokens)
            data_set[PADDED_INPUTS[1]].append(indexed_tokens)

    data_set = pad_dataset(data_set, padding=tokenizer.convert_tokens_to_ids('[PAD]'))

    tensor_datasets = {"train": [], "valid": []}
    train_max_num = int(len(data_set[PADDED_INPUTS[0]]) * train_precent)

    for name  in PADDED_INPUTS: ##['input_ids' , 'label_ids']
        tensor_datasets['train'].append(torch.Tensor(data_set[name][0:train_max_num]).long())
        tensor_datasets['valid'].append(torch.Tensor(data_set[name][train_max_num:]).long())

    train_data_set , valid_data_set =TensorDataset(*tensor_datasets['train']) , TensorDataset(*tensor_datasets['valid'])
    train_data_loader = DataLoader(train_data_set , batch_size = batch_size)
    valid_data_loader = DataLoader(valid_data_set , batch_size = batch_size)
    return train_data_loader , valid_data_loader

'''=========================================================================================================='''

def get_data_set_from_file(tokenized_file , stride ,  n_ctx ):
    with open(tokenized_file , 'r') as f:
        line = f.read().strip()
    tokens = line.split()
    tokens = [int(token) for token in tokens]
    start_point = 0
    samples = []
    while start_point < len(tokens) - n_ctx:
        samples.append(tokens[start_point: start_point + n_ctx])
        start_point += stride
    if start_point < len(tokens):
        samples.append(tokens[len(tokens) - n_ctx:])
    return samples


def get_data_loaders_from_tokenized_files(tokenized_file_path , stride , batch_size ,train_precent = 0.7 , n_ctx = 1024):
    samples = []
    total_length = 0
    for root, dirs, files in os.walk(tokenized_file_path):
        for fname in tqdm(files):
            samp = get_data_set_from_file(root + fname , stride , n_ctx)
            total_length += len(samp)
            samples.extend(samp)
    tensor_datasets = {"train": [], "valid": []}
    train_max_num = int(len(samples) * train_precent)
    for name in PADDED_INPUTS:  ##['input_ids' , 'label_ids']
        tensor_datasets['train'].append(torch.Tensor(samples[0:train_max_num]).long())
        tensor_datasets['valid'].append(torch.Tensor(samples[train_max_num:]).long())

    train_data_set , valid_data_set = TensorDataset(*tensor_datasets['train']) , TensorDataset(*tensor_datasets['valid'])
    train_data_loader = DataLoader(train_data_set , batch_size = batch_size)
    valid_data_loader = DataLoader(valid_data_set , batch_size = batch_size)
    return train_data_loader , valid_data_loader , total_length

'''=========================================================================================================='''


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset[PADDED_INPUTS[0]])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def process_paragraph(paragraph , n_ctx):
    sentence_in_paragraph = ''
    for line in paragraph:
        if len(sentence_in_paragraph) == 0:
            sentence_in_paragraph += ('[MASK]' + line)
        else:
            if len(sentence_in_paragraph) + len(line) > n_ctx:
                break
            sentence_in_paragraph += ('[SEP]' + line)
    sentence_in_paragraph += '[CLS]'
    return sentence_in_paragraph


def get_data_loaders_for_paragraph(data_file , tokenizer , stride ,batch_size ,train_precent = 0.7 , n_ctx = 1024):
    fr = open(data_file)
    data_set = {PADDED_INPUTS[0]:[] , PADDED_INPUTS[1]:[]}
    paragraph = []
    lines = fr.read().splitlines()
    for line in tqdm(lines):
        if line != '':
            paragraph.append(line)
        else:
            sentence_in_paragraph = process_paragraph(paragraph,n_ctx)
            paragraph = []
            if sentence_in_paragraph != None:
                indexed_tokens = tokenizer.encode(sentence_in_paragraph)
                data_set[PADDED_INPUTS[0]].append(indexed_tokens)
                data_set[PADDED_INPUTS[1]].append(indexed_tokens)

    data_set = pad_dataset(data_set, padding=tokenizer.convert_tokens_to_ids('[PAD]'))

    tensor_datasets = {"train": [], "valid": []}
    train_max_num = int(len(data_set[PADDED_INPUTS[0]]) * train_precent)

    for name  in PADDED_INPUTS: ##['input_ids' , 'label_ids']
        tensor_datasets['train'].append(torch.Tensor(data_set[name][0:train_max_num]).long())
        tensor_datasets['valid'].append(torch.Tensor(data_set[name][train_max_num:]).long())

    train_data_set , valid_data_set =TensorDataset(*tensor_datasets['train']) , TensorDataset(*tensor_datasets['valid'])
    train_data_loader = DataLoader(train_data_set , batch_size = batch_size)
    valid_data_loader = DataLoader(valid_data_set , batch_size = batch_size)

    return train_data_loader , valid_data_loader , len(data_set[PADDED_INPUTS[0]])

from transformers import tokenization_bert
if __name__ == '__main__':
    # get_data_loaders_from_tokenized_files('../data/dpcq/tokenized/' , 768 , 100 ,train_precent = 0.7 , n_ctx = 1024)
    tokenizer = tokenization_bert.BertTokenizer('../data/vocab_small.txt')
    train_data_loader, valid_data_loader, data_length = get_data_loaders_for_paragraph('../data/text.data/data/muti_all.data' , tokenizer , 768 ,10 ,train_precent = 0.7 , n_ctx = 1024)
    pdb.set_trace()