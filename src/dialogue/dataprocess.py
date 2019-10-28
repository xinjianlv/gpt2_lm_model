from itertools import chain
import os , math
import pdb
import random
import logging
from tqdm import tqdm
from collections import defaultdict
from data_strcuct import Instance


from torch.utils.data import DataLoader, TensorDataset
import torch
from transformers import tokenization_bert
# from train import logger
logger = logging.getLogger(__file__)
# Let's define our contexts and special tokens
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
bos, eos, speaker1, speaker2 = "<bos>", "<eos>", "<speaker1>", "<speaker2>"
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]



def process_data_by_file(in_file , cache_path, tokenizer):
    data = []
    f = open(in_file , 'r')
    lines = f.read().splitlines()
    paragraph = []
    for line in tqdm(lines):
        if line != '':
            paragraph.append(line)
        else:
            instances_list = process_paragraph(paragraph)
            paragraph = []
            if instances_list:
                #为每个样本添加负例
                for ndx in range(len(instances_list)):
                    set_success = False
                    while not set_success:
                        set_success = instances_list[ndx].set_distractors(lines[random.randint(0 , len(lines) - 1)])
                    instances_list[ndx].transform(tokenizer , SPECIAL_TOKENS[:-1])
                data.extend(instances_list)
    f.close()
    # torch.save(data , open(cache_path + 'process_data_cached' , 'w+'))
    return data


def process_paragraph(paragraph, max_history=2, stride=2, num_candidates = 2):
        if len(paragraph) < max_history + 1:
            return None
        history=[]
        instance_list = []
        for i in range(0 , len(paragraph) - max_history, stride):
            for ndx in range(max_history):
                history.append(paragraph[i + ndx])
            reply = paragraph[i + max_history]
            instance_list.append(Instance(history.copy(), reply))
            history.clear()
        return instance_list


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def get_data_loaders(data_file, tokenizer, cache_path, batch_size ,train_r= 0.7):
    instances = process_data_by_file(data_file, '', tokenizer)
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    dataset_name = 'train'
    for i , instance in enumerate(instances):
        if i > int(len(instances) * train_r):
            dataset_name = 'valid'
        for ins in instance.get_elemnets_list():
            for input_name , input_array in ins.items():
                datasets[dataset_name][input_name].append(input_array)
        # Next-sentence prediction labels
        # optional multiple choice labels: torch.LongTensor of shape[batch_size] with indices selected in[0, ..., num_choices].
        # 最后一个样本是正例
        datasets[dataset_name]["mc_labels"].append(len(instance.get_elemnets_list()) - 1)
        datasets[dataset_name]["n_candidates"] = len(instance.get_elemnets_list())

    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                # 转换成gpt2 输入格式  input_ids = (bsz, number of choice, seq length)
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    train_data_set, valid_data_set = TensorDataset(*tensor_datasets['train']), TensorDataset(*tensor_datasets['valid'])
    train_data_loader , valid_data_loader = DataLoader(train_data_set,batch_size=batch_size), DataLoader(valid_data_set,batch_size=batch_size)
    return train_data_loader, valid_data_loader


'''==============================================加载小黄鸡语料========================================================'''
def process_data_by_file_2(in_file , cache_path, tokenizer):
    data = []
    f = open(in_file , 'r')
    lines = f.read().splitlines()
    ndx = 0
    while ndx < len(lines):
        if ndx % 10000 == 0 :
            # logger.info('load [%d] line.'%ndx)
            logger.info('load [%d] line.'%ndx)
        line = lines[ndx]
        if line.startswith('E'):
            ndx += 1
        else:
            # if len(lines[ndx].split(' ')) < 2:
            question , ndx = get_sentences(lines , ndx)
            answer , ndx = get_sentences(lines , ndx)

            ins = Instance(question, answer)
            suc = False
            while not suc :
                rint = random.randint( 0 , len(lines) - 1)
                distractors = lines[rint]
                if distractors.startswith('E') or len(distractors.split(' ')) < 2 :
                    continue
                suc = ins.set_distractors(distractors.split(' ')[1])
            ins.transform(tokenizer , SPECIAL_TOKENS[:-1])
            data.append(ins)
    f.close()
    # torch.save(data , open(cache_path + 'process_data_cached' , 'w+'))
    return data

def get_sentences(lines , ndx):
    line = ' '.join(lines[ndx].split(' ')[1:])
    while ndx + 1 < len(lines) and ( not lines[ndx+1].startswith('M') and not lines[ndx+1].startswith('E')):
            line = line + ' ' + lines[ndx + 1]
            ndx += 1
    ndx += 1
    return line , ndx


def get_data_loaders_2(data_file, tokenizer, cache_path, batch_size ,train_r= 0.7):
    instances = process_data_by_file_2(data_file, '', tokenizer)
    # pdb.set_trace()
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    dataset_name = 'train'
    for i , instance in enumerate(instances):
        if i > int(len(instances) * train_r):
            dataset_name = 'valid'
        for ins in instance.get_elemnets_list():
            for input_name , input_array in ins.items():
                datasets[dataset_name][input_name].append(input_array)
        # Next-sentence prediction labels
        # optional multiple choice labels: torch.LongTensor of shape[batch_size] with indices selected in[0, ..., num_choices].
        # 最后一个样本是正例
        datasets[dataset_name]["mc_labels"].append(len(instance.get_elemnets_list()) - 1)
        datasets[dataset_name]["n_candidates"] = len(instance.get_elemnets_list())
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                # 转换成gpt2 输入格式  input_ids = (bsz, number of choice, seq length)
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    train_data_set, valid_data_set = TensorDataset(*tensor_datasets['train']), TensorDataset(*tensor_datasets['valid'])
    train_data_loader , valid_data_loader = DataLoader(train_data_set,batch_size=batch_size), DataLoader(valid_data_set,batch_size=batch_size)
    return train_data_loader, valid_data_loader



from transformers import tokenization_gpt2
if __name__ == '__main__':
    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file='../../config/vocab_small.txt')
    full_tokenizer.add_tokens(SPECIAL_TOKENS)
    full_tokenizer.save_pretrained('../../config/pretrained/')
    full_tokenizer.max_len = 512
    train_data_loader, valid_data_loader = get_data_loaders_2('../../data/xiaohuangji/small.txt', full_tokenizer , '../../cache/',2)
        # print('token_type_ids' , ins.token_type_ids)
        # print('mc_token_ids' , ins.mc_token_ids)
        # print('lm_labels' , ins.lm_labels)
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