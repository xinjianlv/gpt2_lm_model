nohup python3 src/train.py  --dialogue 1 --train_batch_size 4 --dataset_path ./data/text.data/data/multi_all.data  --model_checkpoint ./model/dh/all/ --model_config_file ./config/model_config_small.json --vocab_file ./data/vocab_small.txt > log.train.out &
