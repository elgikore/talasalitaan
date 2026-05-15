---
title: Talasalitaan, or How Did a Tokenizer Performed Suprisingly Well on Corpora Roughly the Size of Tiny Shakespere
layout: page
---

Talasalitaan (lit. vocabulary, but usage is more like a glossary/dictionary) is a vanilla BPE model from SentencePiece trained on the [KapitBisig](https://www.kapitbisig.com/philippines) site, using only Filipino as much as possible. The entire corpus is just 1.3 MB, but as this page will show later on, it is surprisingly performant.

The whole code for Talasalitaan is a simple SentencePiece wrapper, but the highlight of this article is the corpora, not the architecture.

## Some Backstory
I was doing an assingment for COMP423 Deep Learning subject, and one of the assignment is to build a GPT-like arctecture with only PyTorch, tokenized using GPT-2 `tiktoken` and train it on the Tiny Shakespere corpus. As I made that assignment, doing Shakespere texts is pretty analogous to doing your Rizal, Noli Me Tangere, and El Filibusterismo in Filipino textbooks. And that's where I thought, 

> "Why not do this for Rizal texts as well? It fits the bill though."

but I quickly realized, even the state of the art (SOTA) LLMs are still inefficient on other languages, especially Filipino since they are trained on English-heavy domains. Thus the rest is history.

Later on, while doing the assignment, and after painful debugging, I commonly run into `OutOfMemoryError`s in PyTorch (on a 4070 GPU nonetheless) while experimenting with hyperparameters. One of the things that bloat memory is the context size. Sure increasing the size attends to more tokens but Attention has a quadratic memory complexity, which is catastrophic as the number of tokens grows.

Another conundrum is that having a small context size will make monsters of a word like *nakakapagpabagabag* (worrisome) harder to capture in an Attention layer just because of the sheer number of tokens needed (10 in GPT-2) -- and that's only one word. What if it is part of a long sentence, since there are pretty common long Filipino words like *kinaroroonan* (whereabouts/current location)? What if it is *pinakanakakapagpabagabag* (most worrisome) that will make PyTorch very worried that you allocated 9 GB on a 8 GB GPU? Imagine all of this when parsing Filipino text to an LLM instead of training Tiny Shakespere, for which my GPU struggled with the latter when scaling hyperparameters.

I thought, why not deal with it at the source, and compress tokens? Look at common English words in OpenAI tokenizers, there are treated as one token. If it is decomposable, maybe a few tokens at most. This is my mindset when creating a "Filipino-aware" tokenizer. The good thing is that, unlike English, prefixes/infixes/suffixes are **very** predicable and rarely has exceptions.

## Choice of Corpus
Everybody says "you just scrape more from Common Crawl" or "curate your data", which is both true. The former understands the reality of AI as data-hungry monsters, and the latter to having representativeness and quality in the model. But what if we model the corpus similar to how a human acquires and masters a language, which is through textbooks, some history, local references, oral traditions, and culture? This is where KapitBisig comes in.

The website has:
- The four main required readings every Filipino goes through:
  - Ibong Adarna
  - Florante at Laura
  - Noli Me Tangere
  - El Filibusterismo
- Works by Dr. Jose Rizal
- Poetical Debate (Balagtasan)
- Awit (Songs in a classical sense)
- Poems (Tula)
- Plays (Dula)
- Parables (Parabula)
- Legends (Alamat)
- Epics (Epiko)
- Myths (Mitolohiya/Mito)
- Riddles (Bugtong)
- Filipino Nursery Rhymes (Tugmang Tagalog)
- Sabayang Pagbigkas (Choral Recitation is a close equivalent in English)
- Fables (Pabula)
- Filipino Proverbs (Salawikaing Filipino)
- Filipino Idioms (Kawikaang Tagalog)
- Basic Level Learn Filipino page
- The full 1987 Constitution in Filipino
- Short summary of each Philippine Presidents
- Short summary of Filipino Heroes (Mga Bayani)

All of these are basically what you would expect in a Filipino subject.

Why did I model the corpus that way? This is because of a phenomenon in Deep Learning that even if AI cannot "understand" or "learn" in a strict sense like humans do, they do arrive at similar conclusions on average. If this logic is true, then it will hold up here. 

## Data Prep
The data is manually copy-pasted and cleaned.

All of the text uses Filipino versions whenever possible. Most of the English is stripped away or (rarely) translated, such as \[Chorus\] → \[Koro\] or \[Repeat from 1\] → \[Uulitin ang 1\]. The only exceptions for this are proper nouns, place names, or there is no natural alternative. This choice was made to ensure that the model learns Filipino-style word constructions first as English-style word constructions are very different.

For the handling of Spanish diacritics, all of them are removed with the exception of ñ as it is part of the Filipino alphabet. Most diacritics you see in Filipino are to guide pronounciation and is used as standard practice on some Filipino textbooks, but it can be safely ignored. The only time diacritics are preserved are on Spanish proper nouns like in the four required readings.

For the required readings, all of them are book summaries because the original books use the old Filipino orthography system. Some examples are n͠g for "ng" and frequent use of Spanish-influnenced orthography like *cuarto* for *kwarto* (room). Even if the Spanish-influnenced orthography of old Filipino spellings are pronounced exactly the same as their modern spelling, it is laborious to replace it with modern spellng styles and risking corrupting the text. The only exception to this rule is Florante at Laura, which conveniently has the original book in modern Filipino orthography. I used both the original book and the book summary for this as a result.

Weird spacing like "  " is corrected (" "), tab spaces are stripped, unicode elipsis ("…") is expanded to "...", and the "/"'s in Sabayang Pagbigkas are removed because it is frequently occuring, which might bias the BPE. All upper and lowercase are preserved.

