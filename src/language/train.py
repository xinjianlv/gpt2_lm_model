import os
import logging
from argparse import ArgumentParser

from transformers import GPT2LMHeadModel , GPT2Config
from transformers import tokenization_bert , CONFIG_NAME , WEIGHTS_NAME
from transformers import AdamW , WarmupLinearSchedule

import torch
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import  RunningAverage
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from get_loader import get_data_loaders , get_data_loaders_from_tokenized_files , get_data_loaders_for_paragraph
logger = logging.getLogger()




def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='../data/', help="Path or url of the dataset cache")
    parser.add_argument("--dataset_cache", type=str, default='../cache/', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--vocab_file", type=str , default="../data/vocab_small.txt" , help="vocab to init tokenizer.")
    parser.add_argument("--model_config_file" , type=str , default="../config/model_config_small.json" , help="vocab to init model config.")
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--dialogue', type=int , default=0, help='1是对话语料，0是其它语料')
    parser.add_argument('--log_step', type=int , default=10, help='1是对话语料，0是其它语料')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument("--fp16_opt_level", type=str, default="O1",help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")



    args = parser.parse_args()


    ##准备模型
    tokenizer = tokenization_bert.BertTokenizer(args.vocab_file)
    model_config = GPT2Config.from_json_file(args.model_config_file)
    model = GPT2LMHeadModel(config=model_config)
    if len(args.model_checkpoint) > 1:
        logger.info('load pretrain model form : %s'%args.model_checkpoint)
        model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    optimizer = AdamW(model.parameters(),lr=args.lr, correct_bias=True)

    if args.fp16:
        logger.info('use fp16')
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    ##准备训练参数
    if args.dialogue == 1 :
        train_data_loader, valid_data_loader, total_length = get_data_loaders_for_paragraph(data_file = args.dataset_path, tokenizer = tokenizer, stride = args.stride, batch_size = args.train_batch_size,n_ctx=model_config.n_ctx)
    else:
        train_data_loader , valid_data_loader , total_length = get_data_loaders_from_tokenized_files(args.tokenized_data_path , args.stride ,args.train_batch_size)

    total_steps = int(total_length / args.stride * args.n_epochs / args.train_batch_size / args.gradient_accumulation_steps)

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_steps)

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
            scheduler.step()
        return loss.item()

    trainer = Engine(update)
    steps = len(train_data_loader.dataset) // train_data_loader.batch_size
    logger.info('data length:%d'%len(train_data_loader.dataset))
    if len(train_data_loader.dataset) % train_data_loader.batch_size != 0:
        steps += 1

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        if trainer.state.iteration % args.log_step == 0:
           logger.info("Epoch[{}/{}] Step[{}/{}] Loss: {:.6f}".format(trainer.state.epoch,
                                                                   trainer.state.max_epochs ,
                                                                   trainer.state.iteration % steps,
                                                                   steps,
                                                                   trainer.state.output * args.gradient_accumulation_steps)
                    )


    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    tb_logger = TensorboardLogger(log_dir=None)
    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
    logger.info('log dir is :%s'%tb_logger.writer.logdir)
    checkpoint_handler = ModelCheckpoint(tb_logger.writer.logdir, 'checkpoint', save_interval=1, n_saved=10)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
        'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

    torch.save(args, tb_logger.writer.logdir + '/model_training_args.bin')
    getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.logdir, CONFIG_NAME))
    tokenizer.save_vocabulary(tb_logger.writer.logdir)

    @trainer.on(Events.ITERATION_COMPLETED)
    def save_model(trainer):
        if trainer.state.iteration % 50000 == 0:
            output_dir = tb_logger.writer.logdir + '/' + str(trainer.state.iteration)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)

    trainer.run(train_data_loader, max_epochs=args.n_epochs)

if __name__ == '__main__':
    train()