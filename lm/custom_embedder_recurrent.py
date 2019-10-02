import torch.nn as nn
import data
import fl_ml_tools.nlp.sequence_sampler as sequence_sampler
from fl_ml.xps.xp17_transformers.xp17_sentence_model_training import NetworkBasedEncoderDecoder
from fl_ml_tools import device

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
    def __init__(self, dict : data.Dictionary, embedding_size):
        super().__init__()
        self.dict = dict

        self.embedding_size = embedding_size

        self.build()

    def build(self):
        enumerator = [(w, 1) for w in self.dict.word2idx]
        self.token_dict = sequence_sampler.build_word_sampler(enumerator, include_count = False)

        sequences, sequences_length = self.token_dict.get_all_sequences_tensor()

        self.data_set = DummyDataSet(sequences, sequences_length)
        encoder = NetworkBasedEncoderDecoder(self.token_dict,
                                             self.embedding_size,
                                             self.data_set)
                                             #layer_sizes=[100, 200])
                                             #letter_embedding_size = 12,
                                             #use_letter_as_input = False)
        self.encoder = device.prepare_object(encoder)

    def forward(self, word_ids):
        return self.encoder.encode(word_ids)

    def last_batch_loss(self):
        ret = self.encoder.last_batch_report()

        return ret.get("loss", 0.0)


