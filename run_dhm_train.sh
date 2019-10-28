nohup python3 ./src/dialogue/train.py \
 --train_batch_size 1 \
 --data_name small.txt \
 --dataset_cache  ./cache/ \
 --dataset_path ./data/xiaohuangji/small.txt \
 --model_config_file ./config/model_config_small.json \
 --vocab_file ./config/vocab_small.txt \
 > log.train.out &
