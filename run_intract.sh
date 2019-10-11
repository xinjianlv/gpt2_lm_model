checkpoint_dir = ''
 python3 src/language/run_intract.py \
 --model_file ${checkpoint_dir}  \
 --device cpu \
 --vocab_file ${checkpoint_dir}/vocab.txt
