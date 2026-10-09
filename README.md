# Shake-GPT

## What it is
Shake-GPT is a small 10 million parameter, character-level GPT that generates Shakespeare-stle text.
This model uses a Transformer with self-attention, which is basically the same architecture behind the massive models like Chat and Claude, but a *bit* smaller.

## What it does
* Reads a dataset of Shakespeare text (input.txt)
* Learns the patterns of the text using self-attention
* Generates new, Shakespeare-like text one character at a time (so it doesn't have the best context)

## What I learnt
* How the Transformer architecture works (self-attention, multi-head attention, and blocks)
* What Query, Key, and Value do, and how tokens "pay attention" to each other
* How positional embeddings and token embeddings feed into the model
* How a GPT is trained and samples text, and how this scales up to real models like ChatGPT

## Preview


https://github.com/user-attachments/assets/b9efb130-e21b-455b-8c52-e0901702b76e


Credits:
Andrej Karpathy's 'Nano-GPT'
