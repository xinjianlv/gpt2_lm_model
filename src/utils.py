import os , pdb
from tqdm import tqdm
from transformers import tokenization_bert
def build_files(raw_data_path, tokenized_data_path, full_tokenizer, num_pieces):
    with open(raw_data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = f.read().splitlines()
        lines = [line.replace('\n', ' [SEP] ') for line in lines if len(line) > 0]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
        print('num of lines :%d' % len(lines))
    single = ''.join(lines)
    len_single = len(single)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        single_ids = full_tokenizer.convert_tokens_to_ids(
            full_tokenizer.tokenize(single[len_single // num_pieces * i: len_single // num_pieces * (i + 1)]))
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in single_ids[:-1]:
                f.write(str(id) + ' ')
            f.write(str(single_ids[-1]))
            f.write('\n')
    print('finish')


if __name__ == '__main__':
    input_data_file = '../data/dpcq/dpcq.txt'
    tokenized_data_path = '../data/tokenized/'
    full_tokenizer = tokenization_bert.BertTokenizer('../data/vocab_small.txt')
    num_pieces = 1
    build_files(input_data_file , tokenized_data_path , full_tokenizer , num_pieces)