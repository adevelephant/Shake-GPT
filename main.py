import torch
import torch.nn as nn
from torch.nn import functional as F

#Reading it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#Every single character
chars = sorted(list(set(text)))
vocab_size = len(chars)

#Tokeniser
#string to int
device = "cpu"
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
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
n_embd = 32

def get_batch(split):
    #Generate a small batch data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.postition_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.postition_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            #Get prdictions
            #focus only on the last time step
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  
            #Use softmax to go from raw logits to prediction prbs
            probs = F.softmax(logits, dim=1)
            #Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            #Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model_0 = BigramLanguageModel()
model_0.to(device)
logits, loss = model_0(xb, yb)
print(logits.shape)
print(loss)


print(list(itos.items())[:5])
print(decode(model_0.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

#Create optimizer
optimizer= torch.optim.Adam(params=model_0.parameters(),
                            lr=1e-3)

batch_size=32
epochs = 10000
for epoch in range(epochs):
    xb, yb = get_batch('train')

    #Evaluate the loss
    logits, loss = model_0(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(loss.item())

#generate
context = torch.zeros((1 ,1), dtype=torch.long, device=device)
print(decode(model_0.generate(context, max_new_tokens=500)[0].tolist()))

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print(x.shape)

xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, 0) 

#version 1
# torch.manual_seed(42)
# a = torch.tril(torch.ones(3, 3))
# a = a / torch.sum(a, 1, keepdim=True)
# b = torch.randint(0, 10, (3, 2)).float()
# c = a @ b
# print("a=")
# print(a)
# print("b=")
# print(b)
# print("c=")
# print(c)

# Version 2
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x

#Version 3
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)

#version 4
torch.manual_seed(42)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
k = key(x)
q = query(x)
wei = q @ k.transpose(-2, -1)

tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=1)
out = wei @ x
print(out.shape)

