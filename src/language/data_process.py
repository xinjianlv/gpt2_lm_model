from itertools import chain
import os
import pdb
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch

from transformers import tokenization_bert



def all_path(dirname):
    print('in all_path')
    result = []#所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)#合并成一个完整路径
            result.append(apath)
    return result

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset[PADDED_INPUTS[0]])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def process_data_by_file(in_file , out_file , tokenizer):
    dataset = []
    f = open(in_file , 'r')
    lines = f.read().splitlines()
    paragraph = []
    for line in tqdm(lines[0:1000]):
        if line != '':
            paragraph.append(line)
        else:
            instances_list = process_paragraph(paragraph , tokenizer)
            paragraph = []
            if instances_list != None:
                dataset.extend(instances_list)

    f.close()
    max_l = max(len(x) for x in dataset)
    padding = tokenizer.convert_tokens_to_ids('[PAD]')
    dataset = [x + [padding] * (max_l - len(x)) for x in dataset]
    return dataset


def process_paragraph(paragraph , tokenizer):
    sentence_in_paragraph = ''
    for line in paragraph:
        if len(sentence_in_paragraph) == 0:
            sentence_in_paragraph += ('[MASK]' + line)
        else:
            sentence_in_paragraph += ('[SEP]' + line)
    sentence_in_paragraph += '[CLS]这是个测试'

    # words = tokenizer.tokenize(sentence_in_paragraph)
    # paragraph_aft_tokenizer = tokenizer.convert_tokens_to_ids(words)

    # print(paragraph_aft_tokenizer == paragraph_aft_tokenizer_2)

    # transfer = tokenizer.convert_ids_to_tokens(paragraph_aft_tokenizer)

    paragraph_aft_tokenizer = tokenizer.encode(sentence_in_paragraph)
    return paragraph_aft_tokenizer


def get_data_loaders(data_file , tokenizer):
    fr = open(data_file)
    data_set = []
    for line in fr.read().splitlines():
        indexed_tokens = tokenizer.encode(line)
        data_set.append(torch.Tensor(indexed_tokens))

    tensor_data_set =TensorDataset(*data_set)
    data_loader = DataLoader(tensor_data_set)
    return data_loader

if __name__ == '__main__':
    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file='../data/vocab_small.txt')
    full_tokenizer.max_len = 1024
    data = process_data_by_file('../data/text.data/data/multi_1_4.4_100w.data' , '',full_tokenizer)



    # f = open('./data/text.data/cnToks.txt')
    # line = f.readline()
    # terms_list = [a for a in line]
    # fw = open('./data/text.data/vocab.cntoks.txt' , 'w')
    # for c in terms_list:
    #     fw.write(c + '\n')
    # fw.flush()
    # fw.close()

    # words, segments, position, sequence = build_inputs(persona, history, reply)
    # print(words)
    # print(segments)
    # print(position)
    # print(sequence)
    # f = open('../data/personachat_self_original.json')
    # data = json.loads(f.read())
    # pdb.set_trace()
    # print(len(data))