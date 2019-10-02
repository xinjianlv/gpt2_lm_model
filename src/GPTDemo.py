from pytorch_pretrained_bert import (OpenAIAdam, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)
from pytorch_pretrained_bert import OpenAIGPTModel , OpenAIGPTConfig , OpenAIGPTLMHeadModel
import torch



# tokenizer = OpenAIGPTTokenizer('./data/text.data/vocab_processed.txt' , )
# model = OpenAIGPTModel(OpenAIGPTConfig())
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=input_ids)
loss, logits = outputs[:2]


# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
#
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')
#
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids, labels=input_ids)
# loss, logits = outputs[:2]