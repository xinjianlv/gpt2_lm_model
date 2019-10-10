nohup python3 language/train.py \
 --train_batch_size 4 \
 --dataset_path ../data/text.data/multi_1_4.4_100w.data  \
 --tokenized_data_path ../data/dpcq/tokenized/ \
 --model_config_file ../config/model_config_small.json \
 --vocab_file ../config/vocab_small.txt \
 > log.train.out &
