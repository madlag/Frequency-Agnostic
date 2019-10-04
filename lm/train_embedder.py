import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from data import Dictionary
from custom_embedder_recurrent import CustomEmbedder
from optimizer import RAdam
import tqdm
import transformers


tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
gpt2 = transformers.GPT2Model.from_pretrained("gpt2")


embedding = gpt2.wte
vocab = tokenizer.decoder

dictionary = Dictionary()
dictionary.word2idx = {v: int(k) for k, v in vocab.items()}
dictionary.idx2word = {int(k): v for k, v in vocab.items()}

model = CustomEmbedder(dictionary, 768)
embedding.weight.requires_grad = False
model = model.cuda()
optimizer = RAdam(model.parameters(), lr=0.001)
writer = SummaryWriter()


class EDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        x = torch.Tensor([i]).long()
        x.requires_grad = False
        return x, self.data[i]


dataset = EDataset(embedding.weight)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)


epochs = 1000
step = 0
for i in tqdm.tqdm(range(epochs)):
    for x, y in tqdm.tqdm(dataloader):
        y = y.cuda()
        x = x.cuda()

        optimizer.zero_grad()

        encoded = model(x).squeeze(0)

        mse = nn.MSELoss()(encoded, y)
        decode_loss = model.last_batch_loss()
        l = 1e-3
        loss = mse + l * decode_loss
        loss.backward()
        optimizer.step()

        writer.add_scalar("loss", loss, global_step=step)
        writer.add_scalar("mse", mse, global_step=step)
        writer.add_scalar("decode_loss", decode_loss, global_step=step)
        step += 1
    torch.save(model, "model.pt")
