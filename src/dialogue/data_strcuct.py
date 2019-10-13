from itertools import chain
import pdb

class Instance(object):

    def __init__(self , history , reply):
        self.history = history
        self.reply = reply
        self.distractors = None
        self.pos_instance = {}
        self.neg_instance = {}

    def set_history(self,history):
        self.history = history

    def get_history(self):
        return self.history

    def set_reply(self , question):
        self.reply= question

    def get_reply(self):
        return self.reply

    def set_distractors(self, distractors):
        if self.reply == distractors:
            return False
        self.distractors = distractors
        return True

    def get_distractors(self):
        return self.distractors

    def transform(self , tokenizer , special_lokens):
        self.neg_instance = self.__transform(tokenizer=tokenizer, history=self.history, reply=self.distractors, lm_labels_req=False, special_lokens=special_lokens)
        self.pos_instance = self.__transform(tokenizer=tokenizer, history=self.history, reply=self.reply, lm_labels_req=True, special_lokens=special_lokens)
        return self.neg_instance , self.pos_instance

    def __transform(self,tokenizer , history , reply , lm_labels_req, special_lokens , with_eos=True):
        persona = []
        bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(special_lokens)
        # Build our sequence by adding delimiters and concatenating
        #sequence = [[bos] + list(chain(*persona))] + [tokenizer.encode(self.history)] + [tokenizer.encode(self.reply) + [eos]]
        # sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
        sequence = [[bos]] + [tokenizer.encode(tokenized_list) for tokenized_list in history ] + [(tokenizer.encode(reply) if with_eos else reply)+ ([eos] if with_eos else [])]

        sequence = [sequence[0]] + [ [speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
        # Build our word, segments and position inputs from the sequence
        # words tokens
        input_ids = list(chain(*sequence))
        # segment tokens
        token_type_ids = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        # Last tokens location
        mc_token_ids = len(input_ids) - 1
        # Language modeling labels : All labels set to -1 are ignored (masked),
        # the loss is only computed for the labels set in [0, ..., config.vocab_size]
        lm_labels = [-1] * len(input_ids)
        if lm_labels_req:
            # 如果是预测label ，则对应的-1改成要预测句子的编码
            lm_labels = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return {'input_ids':input_ids,
                'token_type_ids':token_type_ids,
                'mc_token_ids':mc_token_ids,
                'lm_labels':lm_labels}

    def get_elemnets_list(self):
        return [self.neg_instance, self.pos_instance]

    def get_repy_inputs(self , tokenizer , special_lokens):
        return self.__transform(tokenizer=tokenizer, history=self.history, reply=self.reply, lm_labels_req=False, special_lokens=special_lokens,with_eos=False)
