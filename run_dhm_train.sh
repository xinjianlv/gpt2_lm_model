nohup python3 ./src/dialogue/train.py \
 --train_batch_size 4 \
 --dataset_path ./data/text.data/multi_1_4.4_100w.data  \
 --model_config_file ./config/model_config_small.json \
 --vocab_file ./config/vocab_small.txt
 > log.train.out &
