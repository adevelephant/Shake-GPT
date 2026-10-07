import torch

#Dataset
with open('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()