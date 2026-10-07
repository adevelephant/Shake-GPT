import torch

#Reading it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#Every single character
chars = sorted(list(set(text)))
vocab_size = len(chars)

#Tokeniser
#string to int
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for ch,i in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #encoder,: take a string, output list of integers
decode = lambda l: ''.join([itos[i]for i in l])

#tokenise the dataset
data = torch.tensor(encode(text), dtype=torch.long)

#Train and test set
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    # print(f"When input is {context} the target is {target}")

#Batches
torch.manual_seed(1337)
batch_size = 4 # How many independent squences will we proccess?
block_size = 8 #What is the maximum context length for predictions?

def get_batch(split):
    #Generate a small batch data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')


