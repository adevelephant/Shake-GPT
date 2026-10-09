import torch
import torch.nn as nn
from torch.nn import functional as F

#Reading it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()