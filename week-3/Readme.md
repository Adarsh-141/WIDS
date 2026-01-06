# WIDS Assignment Week 3

This project demonstrates the basics of character-level language modeling using PyTorch. The notebook walks through data preparation, encoding, batching, and building a simple neural network to generate text, focusing on the bigram language model.

## Contents

- **Data Preparation:**
  - Downloads the Tiny Shakespeare dataset and reads it as a string.
  - Builds a vocabulary of unique characters and encodes the text into integer sequences.

- **Batching:**
  - Splits the data into training and validation sets.
  - Implements batching functions to efficiently feed data into the model.

- **Bigram Language Model:**
  - Implements a simple neural network (Bigram Language Model) that predicts the next character based only on the current character.
  - Uses an embedding layer to map input characters to logits for the next character.

- **Training:**
  - Trains the model using cross-entropy loss and the AdamW optimizer.
  - After training, the model can generate new text character by character.

## Key Concepts

### Bigram Language Model
A bigram model is a statistical language model that predicts the next item in a sequence based only on the previous item. In this notebook, the model learns the probability of each character following another character. While simple, this approach forms the foundation for more complex models.

### Transformers (General Concept)
Transformers are advanced neural network architectures that have revolutionized natural language processing. Unlike bigram models, transformers can consider the entire context of a sequence using mechanisms called self-attention. This allows them to capture long-range dependencies and generate much more coherent and meaningful text.

While this notebook implements a simple bigram model, it sets the stage for understanding how more powerful models like transformers work.

### Building a Small ChatGPT-like Model
This project is a stepping stone toward building your own small version of ChatGPT. ChatGPT is based on the transformer architecture and is trained on large amounts of text data to generate human-like responses in a conversational setting. The key ideas include:
- **Tokenization:** Breaking text into smaller units (characters, words, or subwords) for processing.
- **Contextual Understanding:** Using models like transformers to understand and generate text based on the context of the conversation.
- **Training on Dialogue:** Exposing the model to conversational data so it learns to respond appropriately.

By starting with a bigram model, you learn the basics of sequence modeling. Extending this to transformers and training on dialogue data are the next steps toward building a chatbot like ChatGPT.

## How to Run
1. Run all cells in the notebook sequentially.
2. The notebook will download the dataset, process it, train the model, and generate sample text.

## References
- [Tiny Shakespeare Dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [PyTorch Documentation](https://pytorch.org/)
- [Attention Is All You Need (Transformer Paper)](https://arxiv.org/abs/1706.03762)
- [OpenAI ChatGPT](https://openai.com/blog/chatgpt/)