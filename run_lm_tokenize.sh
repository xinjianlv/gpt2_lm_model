nohup python3 src/language/utils.py \
--input_data_file ./data/dpcq/dpcq.txt \
--tokenized_data_path ./data/dpcq/tokenized/   \
--vocab_file  ./config/vocab_small.txt \
--num_pieces 10 \
> log.tokenize.out &