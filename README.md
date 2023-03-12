# Overview
In this project, I build a character-level transformer-based language model something like GPT(Generative Pretrained Transformer). Once the model is trained, we can generate infitnite text. 

# Requirements
```
pytorch
simple-parsing
rich
```

# Dataset
Since training on chunk of internet requries a lot of *compute* and *time*, here I work with a [tiny shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset.