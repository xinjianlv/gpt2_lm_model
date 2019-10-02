from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
import torch
#
# print('load model...')
# model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt' , cache_dir='../cache/cache')
# print('load tokeizer...')
# tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt', cache_dir='../cache/cache')
# SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
# tokenizer.set_special_tokens(SPECIAL_TOKENS)


tokenizer = OpenAIGPTTokenizer.from_pretrained('../data/')
model = OpenAIGPTDoubleHeadsModel.from_pretrained('../data/')
# tokenizer.add_special_tokens({'cls_token': '[CLS]'})  # Add a [CLS] to the vocabulary (we should train it also!)
choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
mc_token_ids = torch.tensor([input_ids.size(-1), input_ids.size(-1)]).unsqueeze(0)  # Batch size 1
print(input_ids)
print(mc_token_ids)

outputs = model(input_ids, mc_token_ids=mc_token_ids)
lm_prediction_scores, mc_prediction_scores = outputs[:2]

print('end...')