import torch.nn as nn
import data
import fl_ml_tools.nlp.sequence_sampler as sequence_sampler
from fl_ml.xps.xp17_transformers.xp17_sentence_model_training import NetworkBasedEncoderDecoder
from fl_ml_tools import device



class CustomEmbedder(nn.Module):
    def __init__(self, dict : data.Dictionary, embedding_size):
        super().__init__()
        self.dict = dict

        self.embedding_size = embedding_size

    def forward(self, word_ids):
        pass
