---
title: Talasalitaan, or How Did a Tokenizer Performed So Well on Corpora Roughly the Size of Tiny Shakespere
layout: page
---

Talasalitaan (lit. vocabulary, but usage is more like a glossary/dictionary) is a vanilla BPE model from SentencePiece trained on the ![KapitBisig](https://www.kapitbisig.com/philippines) site, using only Filipino as much as possible. The entire corpus is just 1.3 MB, but as this page will show later on, it is surprisingly performant.

The whole code for Talasalitaan is a simple SentencePiece wrapper, but the highlight of this article is the corpora, not the architecture.

## Some Backstory
I was doing an assingment for COMP423 Deep Learning subject, and one of the assignment is to build a GPT-like arctecture with only PyTorch, tokenized using GPT-2 `tiktoken` and train it on the Tiny Shakespere corpus. As I made that assignment, doing Shakespere texts is pretty analogous to doing your Rizal, Noli Me Tangere, and El Filibusterismo in Filipino textbooks. And that's where I thought, 

> "Why not do this for Rizal texts as well? It fits the bill though."

but I quickly realized, even the state of the art (SOTA) LLMs are still inefficient on other languages, especially Filipino since they are trained on English-heavy domains. Thus the rest is history.

Later on, while doing the assignment, and after painful debugging, I commonly run into `OutOfMemoryError`s in PyTorch (on a 4070 GPU nonetheless) while experimenting with hyperparameters. One of the things that bloat memory is the context size. Sure increasing the size attends to more tokens but Attention has a quadratic memory complexity, which is catastrophic as the number of tokens grows.

Another conundrum is that having a small context size will make monsters of a word like *nakakapagpabagabag* (worrisome) harder to capture in an Attention layer just because of the sheer number of tokens needed (10 in GPT-2) -- and that's only one word. What if it is part of a long sentence, since there are pretty common long Filipino words like *kinaroroonan* (whereabouts/current location)? What if it is *pinakanakakapagpabagabag* (most worrisome) that will make PyTorch very worried that you allocated 9 GB on a 8 GB GPU? Imagine all of this when parsing Filipino text to an LLM instead of training Tiny Shakespere, for which my GPU struggled with the latter when scaling hyperparameters.

I thought, why not deal with it at the source, and compress tokens? Look at common English words in OpenAI tokenizers, there are treated as one token. If it is decomposable, maybe a few tokens at most. This is my mindset when creating a "Filipino-aware" tokenizer. The good thing is that, unlike English, prefixes/infixes/suffixes are **very** predicable and rarely has exceptions.

## Choice of Corpus
