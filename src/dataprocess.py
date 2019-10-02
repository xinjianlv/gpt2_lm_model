from itertools import chain
import os
import pdb
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
# Let's define our contexts and special tokens
#persona = [["i", "like", "playing", "football", "."],["i", "am", "from", "NYC", "."]]
#history = [["hello", "how", "are", "you", "?"],["i", "am", "fine", "thanks", "."],["hello", "how", "are", "you", "?"],["i", "am", "fine", "thanks", "."]]
persona = [["i", "like" ],["i", "am", "from"]]
history = [["hello", "how"],["i", "am", "fine"],["what", "is"],["it", "is", "a"]]
reply = ["great", "to", "hear"]
bos, eos, speaker1, speaker2 = "<bos>", "<eos>", "<speaker1>", "<speaker2>"

def build_inputs(persona, history, reply):
    # Build our sequence by adding delimiters and concatenating
    sequence = [[bos] + list(chain(*persona))] + history + [reply + [eos]]
    sequence = [sequence[0]] + [ [speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    # Build our word, segments and position inputs from the sequence
    words = list(chain(*sequence))
    segments = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    position = list(range(len(words)))                     #
    return words, segments, position, sequence




def all_path(dirname):
    print('in all_path')
    result = []#所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)#合并成一个完整路径
            result.append(apath)
    return result

def process_data(in_file_path , out_file):
    filenames = all_path(in_file_path)


def process_data_by_file(in_file , out_file):
    data = []
    f = open(in_file , 'r')
    lines = f.read().splitlines()
    paragraph = []
    for line in tqdm(lines[0:1000]):
        if line != '':
            paragraph.append(line)
        else:
            instances_list = process_paragraph(paragraph)
            paragraph = []
            if instances_list != None:
                data.extend(instances_list)
    f.close()
    return data
from src.data_strcuct import Instance
from tokenizations import tokenization_bert
full_tokenizer = tokenization_bert.BertTokenizer(vocab_file='./data/text.data/vocab_processed.txt')
full_tokenizer.max_len = 100
def process_paragraph(paragraph):
        if len(paragraph) < 3:
            return None
        history = []
        instance_list = []
        for i in range(0 , len(paragraph) - 2, 1):
            history_q = paragraph[i]
            history_a = paragraph[i + 1]
            reply = paragraph[i + 2]
            history.append(history_q)
            history.append(history_a)
            instance_list.append(Instance(history , reply))
        return instance_list

def get_data_loaders(data_file , tokenizer):
    fr = open(data_file)
    data_set = []
    for line in fr.read().splitlines():
        indexed_tokens = tokenizer.encode(line)
        data_set.append(torch.Tensor(indexed_tokens))

    tensor_data_set =TensorDataset(*data_set)
    data_loader = DataLoader(tensor_data_set)
    return data_loader

import json
if __name__ == '__main__':
    data = process_data_by_file('./data/text.data/multi_1_4.4_100w.data' , '')
    for ins in data:
        history = ins.get_history()
        reply = ins.get_reply()
        print('history :%s'%history)
        print('reply : %s'%reply)
        words, segments, position, sequence ,  = build_inputs(persona=[''] , history=history , reply=reply)
        print(words)
        print(segments)
        print(position)
        print(sequence)

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