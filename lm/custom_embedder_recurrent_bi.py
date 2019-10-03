import torch.nn as nn
import data
import fl_ml_tools.nlp.sequence_sampler as sequence_sampler
from fl_ml.xps.xp17_transformers.xp17_sentence_model_training import NetworkBasedEncoderDecoder
from fl_ml_tools import device
import pkm_layer

class DummyDataSet():
    def __init__(self, sequences, sequences_length):
        self.sequences = device.prepare_object(sequences)
        self.sequences_length = device.prepare_object(sequences_length)

    def get_letters_and_length(self, word_ids):
        shape = word_ids.shape
        word_ids_flatten = word_ids.view(-1)

        sequences = self.sequences[word_ids_flatten]
        sequences_length = self.sequences_length[word_ids_flatten]

        # Truncate the sequences to the maximum length of all sequence members
        max_len = sequences_length.max()
        sequences = sequences[:,0:max_len]

        sequences = sequences.view(shape + (-1,))
        sequences_length = sequences_length.view(shape)

        return sequences, sequences_length

class CustomEmbedder(nn.Module):
    def __init__(self,
                 dict : data.Dictionary,
                 embedding_size,
                 disembedding_size = None,
                 optimized = False,
                 bidirectional = True,
                 pkm_embedding_size = 0,
                 pkm_add = False,
                 ):
        super().__init__()
        self.dict = dict

        self.embedding_size = embedding_size
        self.disembedding_size = disembedding_size
        self.optimized = optimized
        self.encoder_count = 2 if bidirectional else 1

        self.build()
        self.pkm_embedding_size = pkm_embedding_size
        self.pkm_add = pkm_add
        if self.pkm_embedding_size > 0:
            params = pkm_layer.AttrDict ({
                "sparse": False,
                "k_dim": 32,
                "heads": 4,
                "knn": 32,
                "n_keys": 128,  # the memory will have (n_keys ** 2) values
                "query_batchnorm": False,
                "input_dropout": 0,
                "query_dropout": 0,
                "value_dropout": 0,
            })

            self.pkm =  pkm_layer.HashingMemory(embedding_size, self.pkm_embedding_size, params)


    def build(self):
        enumerator = [(w, 1) for w in self.dict.word2idx]
        self.token_dict = sequence_sampler.build_word_sampler(enumerator, include_count = False)

        self.encoders = nn.ModuleList()
        for i in range(self.encoder_count):
            sequences, sequences_length = self.token_dict.get_all_sequences_tensor(reverse = i == 1)

            data_set = DummyDataSet(sequences, sequences_length)
            encoder = NetworkBasedEncoderDecoder(self.token_dict,
                                                 self.embedding_size,
                                                 data_set,
                                                 word_disembedding_size=self.disembedding_size,
                                                 optimized = self.optimized
                                                 )
                                                 #layer_sizes=[100, 200])
                                                 #letter_embedding_size = 12,
                                                 #use_letter_as_input = False)
            self.encoders += [device.prepare_object(encoder)]

    def forward(self, word_ids):
        ret = [encoder.encode(word_ids) for encoder in self.encoders]
        ret = sum(ret)

        if self.pkm_embedding_size > 0:
            ret2 = self.pkm(ret)

        if self.pkm_add:
            return ret2 + ret
        else:
            return ret2

    def last_batch_loss(self):
        rets = [encoder.last_batch_report() for encoder in self.encoders]

        return sum([ret["loss"] for ret in rets])


