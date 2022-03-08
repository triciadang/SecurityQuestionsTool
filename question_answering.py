import torch
import nlp
from transformers import LongformerTokenizerFast,LongformerTokenizer, LongformerForQuestionAnswering, EvalPrediction

import json
import os
import re
import string
import numpy as np
import re

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#https://huggingface.co/allenai/longformer-base-4096
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')


# Not reqd

class Sample:
    def __init__(self, question, context, start_char_idx=None, answer_text=None, all_answers=None):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False
        self.start_token_idx = -1
        self.end_token_idx = -1

    def preprocess(self):
        context = " ".join(str(self.context).split())
        input_pairs = [self.question, self.context]
        encodings = tokenizer.encode_plus(input_pairs, pad_to_max_length=True, max_length=1024)
        context_encodings = tokenizer.encode_plus(self.context)
        # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
        # this will give us the position of answer span in the context text
        if self.answer_text is not None:
            answer = " ".join(str(self.answer_text).split())
            end_idx = self.start_char_idx + len(answer)
            if end_idx >= len(context):
                self.skip = True
                return
        start_positions_context = context_encodings.char_to_token(self.start_char_idx)
        end_positions_context = context_encodings.char_to_token(end_idx - 1)
        # here we will compute the start and end position of the answer in the whole example
        # as the example is encoded like this <s> question</s></s> context</s>
        # and we know the postion of the answer in the context
        # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
        # this will give us the position of the answer span in whole example
        sep_idx = encodings['input_ids'].index(tokenizer.sep_token_id)
        start_positions = start_positions_context + sep_idx + 1
        end_positions = end_positions_context + sep_idx + 1

        if end_positions > 750:
            start_positions, end_positions = 0, 0

        encodings.update({'start_positions': start_positions,
                          'end_positions': end_positions,
                          'attention_mask': encodings['attention_mask']})
        return encodings