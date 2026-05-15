# Talasalitaan, or How Did a Tokenizer Performed So Well on Corpora Roughly the Size of Tiny Shakespere

Talasalitaan (lit. vocabulary, but usage is more like a glossary/dictionary) is a vanilla BPE model from SentencePiece trained on the ![KapitBisig](https://www.kapitbisig.com/philippines) site, using only Filipino as much as possible. The entire corpus is just 1.3 MB, but as this page will show later on, it is surprisingly performant.

The whole code for Talasalitaan is a simple SentencePiece wrapper, but the highlight of this article is the corpora, not the architecture.
