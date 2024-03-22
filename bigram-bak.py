import torch
import torch.nn as nn
from simple_parsing import ArgumentParser
from torch.nn import functional as F

from config.args import Args
from config.logger import logger

# Set seed
torch.manual_seed(1337)

# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Reading data file
with open("input.txt", encoding="utf-8") as f:
    text = f.read()


### Utility Functions ###

# Set training device
# def set_device():
#     if torch.cuda.is_available():
#         device = "gpu"
#     else:
#         device = "cpu"
#     return device


# for m1 mac
def set_device():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


# Create mapping function
def gen_mapping(chars):
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


# Function to split data into train and val
def split_data(data, split_ratio):
    # TODO create split function
    n = int(split_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


# Funcition to generate batches, dataloader functiom
def get_batch(split):
    # generate a small batch of data of input x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(
        len(data) - Args.block_size, (Args.batch_size,)
    )  # because last will start from -8 and go until the end of text
    x = torch.stack([data[i : i + Args.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + Args.block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


### Model class ###


# Our simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each idx will go to the table and pluck out the corresponding row
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)

        # if no targets, then we ignore loss
        if targets is None:
            loss = None
        else:
            # reshape for loss, loss needs BxCxT instead
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss  # logits are the score for next character in the sequence

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices i nthe current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time steop
            logits = logits[:, -1, :]  # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # appends sampled index to the running index
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


### Initialize ###

device = set_device()
logger.info(f"Currently using {device} device...")

# Read args
logger.info("Reading arguments...")
parser = ArgumentParser()
parser.add_arguments(Args, dest="options")
args_namespace = parser.parse_args()
args = args_namespace.options

# get all the unique characters in corpus
chars = sorted(list(set(text)))
vocab_size = len(chars)
logger.info(f"Total vocabulary size is {vocab_size}.")

# generate the string to int mapping and vice versa
stoi, itos = gen_mapping(chars)


# encoder: take a string, output a list of ints
def encode(s):
    return [stoi[c] for c in s]


# decode: take a list of ints, output a string
def decode(lst):
    return "".join(itos[i] for i in lst)


# Encode the dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Split into train and val set
train_data, val_data = split_data(data, args.split_ratio)
logger.info(f"Total training data: {len(train_data)}")
logger.info(f"Total validation data: {len(val_data)}")

# Instantiate model and put on device
model = BigramLanguageModel(vocab_size)
m = model.to(device)
logger.info(m)

# Create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=args.learning_rate)

### Training loop ###

for iters in range(args.max_iters):
    # every once in a while, evaluate the loss on train and val sets
    losses = estimate_loss()
    logger.info(f"step {iter} train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
