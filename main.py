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
  