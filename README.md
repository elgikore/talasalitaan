# A Filipino Tokenizer Trained on KapitBisig
> [!NOTE]
> The write-up: [Talasalitaan, or How Did a Tokenizer Performed Suprisingly Well on Corpora Roughly the Size of Tiny Shakespere](https://elgikore.github.io/talasalitaan/).

This is a minimal SentencePiece wrapper with BPE, trained on the [KapitBisig](https://www.kapitbisig.com/philippines) site. The corpus is designed to minimize English as much as possible, thereby the only limitation it has are Taglish and pure English words. It uses a 32,768 vocabulary size.

This tokenizer serves as a demonstration on how it became suprisingly performant on a tiny corpus.

# Usage
Since the `Talasalitaan.py` itself is just a wrapper, downloading the model weights and `.vocab` file and importing it in SentencePiece is already sufficient, only taking note that it uses `bpe` with `vocab_size=32768`. You can also use `Talasalitaan.py` and import it in your Python file if you want to train on a custom dataset.
