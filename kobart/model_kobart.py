#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:22:22 2021


@author: Luca Medeiros, lucamedeiros@outlook.com
"""

import os
import torch
import torch.nn as nn
from typing import Union, List
from transformers import BartModel

from .pytorch_kobart import get_pytorch_kobart_model
from .utilities import get_kobart_tokenizer


os.environ["TOKENIZERS_PARALLELISM"] = 'False'


class Kobart(nn.Module):
    def __init__(self, args, classes):
        super().__init__()
        self.tokenizer = get_kobart_tokenizer()
        self.encoder = BartModel.from_pretrained(get_pytorch_kobart_model()).get_encoder()
        self.all_tokens = self.tokenize(classes)

    def tokenize(self, texts: Union[str, List[str]]) -> torch.LongTensor:
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        Returns
        -------
        Dict of tokens and attention mask
        """
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(texts, tuple):
            texts = list(texts)
        result = self.tokenizer(texts, return_tensors='pt', padding=True)
        
        if 'token_type_ids' in result:
            del result['token_type_ids']

        return result.to(self.encoder.device)

    def encode_text(self, texts: Union[str, List[str]]):
        tokens = self.tokenize(texts)
        emb = self.encoder(**tokens)['last_hidden_state']
        embeddings = torch.sum(emb * tokens["attention_mask"].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(tokens["attention_mask"], dim=1, keepdims=True), min=1e-9)

        return embeddings

    def forward(self, text):
        text_features = self.encode_text(text)

        return text_features
