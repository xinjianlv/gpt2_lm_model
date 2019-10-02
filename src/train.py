import os
import pdb
import math
import logging
import torch
from argparse import ArgumentParser
from pprint import pformat
from itertools import chain
from tqdm import tqdm

from transformers import GPT2LMHeadModel , GPT2Config
from transformers import tokenization_bert , CONFIG_NAME , WEIGHTS_NAME
from transformers import AdamW , WarmupLinearSchedule

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
logger = logging.getLogger()

PADDED_INPUTS = ['input_ids' , 'label_ids']


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset[PADDED_INPUTS[0]])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def get_data_loaders(data_file , tokenizer , batch_size ,train_precent = 0.7 , n_ctx = 1024):
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
                if len(line) + len(l) < n_ctx:
                    line +=l
                else:
                    break
            line += '[SEP]'

            indexed_tokens = tokenizer.encode(line)
            data_set[PADDED_INPUTS[0]].append(indexed_tokens)
            data_set[PADDED_INPUTS[1]].append(indexed_tokens)

    data_set = pad_dataset(data_set, padding=tokenizer.convert_tokens_to_ids('[PAD]'))

    tensor_datasets = {"train": [], "valid": []}
    train_max_num = int(len(data_set[PADDED_INPUTS[0]]) * train_precent)

    for name  in PADDED_INPUTS: ##['input_ids' , 'label_ids']
        tensor_datasets['train'].append(torch.Tensor(data_set[name][0:train_max_num]).long())
        tensor_datasets['valid'].append(torch.Tensor(data_set[name][train_max_num:]).long())

    train_data_set , valid_data_set =TensorDataset(*tensor_datasets['train']) , TensorDataset(*tensor_datasets['valid'])
    train_data_loader = DataLoader(train_data_set , batch_size = batch_size)
    valid_data_loader = DataLoader(valid_data_set , batch_size = batch_size)
    return train_data_loader , valid_data_loader

fw = open('./log_run_time.out' , 'r+')

def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data/muti_all_head_100w.txt", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache/', help="Path or url of the dataset cache")
    #parser.add_argument("--model_checkpoint", type=str, default="openai-gpt", help="Path, url or short name of the model")
    parser.add_argument("--model_checkpoint", type=str, default="../model/", help="Path, url or short name of the model")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()


    ##准备数据

    ##准备模型
    tokenizer = tokenization_bert.BertTokenizer('./data/vocab_small.txt')

    model_config = GPT2Config.from_json_file('./config/model_config_small.json')
    model = GPT2LMHeadModel(config=model_config)
    model.to(args.device)
    optimizer = AdamW(model.parameters(),lr=args.lr, correct_bias=True)
    warmup_steps = 1
    total_steps = 2
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
    ##准备训练参数



    train_data_loader , valid_data_loader = get_data_loaders(args.dataset_path, tokenizer , args.train_batch_size)

    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids , label_ids = batch
        loss, logits , _ = model.forward(input_ids = input_ids , labels = label_ids)

        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return loss.item()

    trainer = Engine(update)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        info = "Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output)
        fw.write(info + '\n')
        fw.flush()
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, label_ids = batch
            loss, logits, _ = model.forward(input_ids=input_ids, labels=label_ids)
            logger.info('loss in evaluator is : %f'%loss)
    evaluator = Engine(inference)

    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=None)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        logger.info('log dir is :%s'%tb_logger.writer.logdir)
        checkpoint_handler = ModelCheckpoint(tb_logger.writer.logdir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, tb_logger.writer.logdir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.logdir, CONFIG_NAME))
        tokenizer.save_vocabulary(tb_logger.writer.logdir)

        trainer.run(train_data_loader, max_epochs=args.n_epochs)
if __name__ == '__main__':
    train()