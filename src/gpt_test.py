import torch
import pdb
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel , GPT2Config
from transformers import tokenization_bert
from collections import defaultdict# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

model_config = GPT2Config.from_json_file('./config/model_config_small.json')


# Load pre-trained model tokenizer (vocabulary)
tokenizer = tokenization_bert.BertTokenizer('./data/vocab_small.txt')
# tokenizer = tokenization_bert.BertTokenizer('./data/vocab_origin.txt')
# tokenizer = GPT2Tokenizer('./model/vocab.json' , './model/merges.txt')
# Encode a text inputs
text1 = "thisisaenglishtest!"#[5661, 9160, 39126, 9288, 0]
text = 'this is a english test!'#[5661, 318, 257, 46932, 1332, 0]
text2 = '山里有座庙，庙里有两个'
indexed_tokens = tokenizer.encode(text2)
print(indexed_tokens)
# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

# model = GPT2LMHeadModel(config=model_config)

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
# model.eval()

# If you have a GPU, put everything on cuda
# tokens_tensor = tokens_tensor.to('cuda')
# model.to('cuda')

# # Predict all tokens
# with torch.no_grad():
#     outputs = model(tokens_tensor)
#     predictions = outputs[0]
#
# # get the predicted next sub-word (in our case, the word 'man')
# predicted_index = torch.argmax(predictions[0, -1, :]).item()
# predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
# print(predicted_text)

from torch.utils.data import DataLoader, TensorDataset
PADDED_INPUTS = ['input_ids' , 'label_ids']
def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset[PADDED_INPUTS[0]])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding] * (max_l - len(x)) for x in dataset[name]]
    return dataset
def get_data_loaders(data_file , tokenizer , train_precent = 0.7):
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
                line +=l
            line += '[SEP]'
            indexed_tokens = tokenizer.encode(line)
            data_set[PADDED_INPUTS[0]].append(indexed_tokens)
            data_set[PADDED_INPUTS[1]].append(indexed_tokens)

    data_set = pad_dataset(data_set, padding=tokenizer.convert_tokens_to_ids('[PAD]'))

    tensor_datasets = {"train": [], "valid": []}
    train_max_num = int(len(data_set[PADDED_INPUTS[0]]) * train_precent)

    for name  in PADDED_INPUTS: ##['input_ids' , 'label_ids']
        tensor_datasets['train'].append(torch.Tensor(data_set[name][0:train_max_num]))
        tensor_datasets['valid'].append(torch.Tensor(data_set[name][train_max_num:]))

    train_data_set , valid_data_set =TensorDataset(*tensor_datasets['train']) , TensorDataset(*tensor_datasets['valid'])
    train_data_loader = DataLoader(train_data_set)
    valid_data_loader = DataLoader(valid_data_set)
    return train_data_loader , valid_data_loader

train_loader = get_data_loaders('./data/text.data/multi_1_4.4_100w.data' , tokenizer )
print('end .. ')