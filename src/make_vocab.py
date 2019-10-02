import argparse
import thulac
import json

from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
import pdb
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', default='./data/text.data/muti_all.data', type=str, required=False, help='原始训练语料')
    parser.add_argument('--vocab_file', default='./data/text.data/vocab_processed.txt', type=str, required=False, help='生成vocab链接')
    parser.add_argument('--vocab_size', default=50000, type=int, required=False, help='词表大小')
    args = parser.parse_args()

    lac = thulac.thulac(seg_only=True)
    tokenizer = Tokenizer(num_words=args.vocab_size)
    print('args:\n' + args.__repr__())
    print('This script is extremely slow especially for large corpus. Take a break.')

    f = open(args.raw_data_path, 'r')
    # lines = json.load(f)
    lines = f.readlines(1000)
    for i, line in enumerate(tqdm(lines)):
        lines[i] = lac.cut(line, text=True)

    tokenizer.fit_on_texts(lines)
    vocab = list(tokenizer.word_index.keys())
    pre = ['[SEP]', '[CLS]', '[MASK]', '[PAD]', '[UNK]']
    vocab = pre + vocab
    with open(args.vocab_file, 'w') as f:
        for word in vocab[:args.vocab_size + 5]:
            f.write(word + '\n')

def tokenizer_test():
    segment = False
    if segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file='./data/text.data/vocab_processed.txt')
    full_tokenizer.max_len = 100
    line = '你还不了解我,蛛哥就知道,我很快就毛事了,只是被好朋友误会有点不好受'
    line1 = full_tokenizer.tokenize(line)
    print(line1)
    ids = full_tokenizer.convert_tokens_to_ids(line1)
    print(ids)
if __name__ == "__main__":
    # main()

    tokenizer_test()