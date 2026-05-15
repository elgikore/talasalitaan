---
title: Talasalitaan, or How Did a Tokenizer Performed Suprisingly Well on Corpora Roughly the Size of Tiny Shakespere
layout: page
---

Talasalitaan (lit. vocabulary, but usage is more like a glossary/dictionary) is a vanilla BPE model from SentencePiece trained on the [KapitBisig](https://www.kapitbisig.com/philippines) site, using only Filipino as much as possible. The entire corpus is just 1.3 MB, but as this page will show later on, it is surprisingly performant. Token counts are also reduced.

The whole code for Talasalitaan is a simple SentencePiece wrapper, but the highlight of this article is the corpora, not the architecture.

# Some Backstory
I was doing an assingment for COMP423 Deep Learning subject, and one of the assignment is to build a GPT-like arctecture with only PyTorch, tokenized using GPT-2 `tiktoken` and train it on the Tiny Shakespere corpus. As I made that assignment, doing Shakespere texts is pretty analogous to doing your Rizal, Noli Me Tangere, and El Filibusterismo in Filipino textbooks. And that's where I thought, 

> "Why not do this for Rizal texts as well? It fits the bill though."

but I quickly realized, even the state of the art (SOTA) LLMs are still inefficient on other languages, especially Filipino since they are trained on English-heavy domains. Thus the rest is history.

Later on, while doing the assignment, and after painful debugging, I commonly run into `OutOfMemoryError`s in PyTorch (on a 4070 GPU nonetheless) while experimenting with hyperparameters. One of the things that bloat memory is the context size. Sure increasing the size attends to more tokens but Attention has a quadratic memory complexity, which is catastrophic as the number of tokens grows.

Another conundrum is that having a small context size will make monsters of a word like *nakakapagpabagabag* (worrisome) harder to capture in an Attention layer just because of the sheer number of tokens needed (10 in GPT-2) -- and that's only one word. What if it is part of a long sentence, since there are pretty common long Filipino words like *kinaroroonan* (whereabouts/current location)? What if it is *pinakanakakapagpabagabag* (most worrisome) that will make PyTorch very worried that you allocated 9 GB on a 8 GB GPU? Imagine all of this when parsing Filipino text to an LLM instead of training Tiny Shakespere, for which my GPU struggled with the latter when scaling hyperparameters.

I thought, why not deal with it at the source, and compress tokens? Look at common English words in OpenAI tokenizers, there are treated as one token. If it is decomposable, maybe a few tokens at most. This is my mindset when creating a "Filipino-aware" tokenizer. The good thing is that, unlike English, prefixes/infixes/suffixes are **very** predicable and rarely has exceptions.

# Data
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

# Some RRL
Initially, I was going to compare mine with GPT-2 and GPT-4o tokenizers to test performance, as I don't believe that there is a specialized Filipino tokenizer. But for completeness sake, I researched "Filipino Tokenizers" on GitHub, and it actually has results, but only one fits the idea which is JpCurada's [`filipino-tokenizer`](https://github.com/JpCurada/filipino-tokenizer). It describes itself as the "first open-source, morphologically-aware subword tokenize for Philippine languages". It is a BPE with handwritten rules for prefixes/infixes/suffixes made in Python and part Rust. The presence of this repo alone makes it possible to compare apples to apples with aside from apples to green apples (not oranges as OpenAI GPT models use BPE).

# Limitations
It doesn't perform well in English words, which is expected for a tokenizer that is trained on mainly Filipino words ~99.9% of the time. It also doesn't aim to be "morphologically accurate" like in JpCurada's case as I let the data speak to itself during training -- which merges are valuable is for BPE to decide statistically. This is because I am confident that even BPE can pick up very common prefix/infix/suffix styles in Filipino since they are ubiquitous in everyday speech and writing, whether it is simple or stacked affixing.

This proof of concept is more on reducing token cost rather than achieving full linguistic coverage across all Philippine languages. 

# Test Data
All sentences are not seen during training except indicated otherwise as a quick smoke check.

Sample sentences are as follows:
1. Greeting
   
   > Kamusta, mga kababayan!

2. Long affixes (This word is seen during training, but it acts as a sanity check on whether this model mastered it at all)

   > pagpapanibagong-tatag

3. The famous tongue-twister

   > nakakapagpabagabag

4. Number 3's sidekick

   > pinakanakapagpapabagabag

5. Good morning

   > Magandang umaga, kapatid!

6. Simple sentence

   > Kumain siya ng pagkain.

7. Simple question

    > kumain ka na ba?

8. Full sentence from [KapitBisig](https://www.kapitbisig.com/philippines/information/arts-and-literature-mga-kuwentong-bayan-folktales_190.html) (This word is seen during training, but it acts as a sanity check on whether this model mastered it at all)

   > Ito ay pagsasalaysay ng mga katutubo sa kanilang paniniwalang lakas ng pisikal na kapaligiran at lakas ng pananampalataya ng lumilimbag sa kanilang buhay at kapalaran.

9. Historical Wikipedia [article](https://tl.wikipedia.org/wiki/Kasaysayan_ng_Pilipinas_(1565%E2%80%931898)#Pagdating_ni_Ruy_L%C3%B3pez_de_Villalobos) sentence

   > Ang unang paglalayag na pambuong mundo sa ngalan ng Espanya ay nasundan ng apat pang mga ekspedisyon mula 1525 hanggang 1542. Sa ikaapat na panggagalugad, narating ni Ruy Lopez de Villalobos ang Kapuluan ng Pilipinas at pinangalanan niya ang mga pulo mula kay Philip II na noon ay may katayuan bilang tagapagmana ng trono ng Kaharian ng Espanya, bagaman hindi pa pormal na naitatag ang Pilipinas bilang opisyal na teritoryo ng Espanya.

10. Declaration of Human Rights Preamble in Filipino

    > Sapagkat ang pagkilala sa katutubong karangalan at sa pantay at di-maikakait na mga karapatan ng lahat ng nabibilang sa angkan ng tao ay siyang saligan ng kalayaan, katarungan at kapayapaan sa daigdig.

11. [Patungkol](https://tl.wikipedia.org/wiki/Unang_Pahina#Patungkol) ng Wikipedia

    > Ang Wikipedia ay isang proyektong online na ensiklopedya na panlahat, nakasulat sa maraming wika, at pinagtutulungan ang paggawa ng mga artikulo sa prinsipyong wiki. Naglalayon ang proyektong ito na mag-alok ng mga nilalaman na malayang muling magagamit, walang pinapanigan, at napapatunayan, na maaring baguhin at mapabuti ninuman. Nakikilala ang Wikipedia sa pamamagitan ng mga naitatag na prinsipyo. Nakalisensiya ang nilalaman nito sa ilalim ng Creative Commons BY-SA. Maari itong kopyahin at muling gamitin sa ilalim ng parehong lisensiya, na sumasailalim sa paggalang sa mga kondisyon. Ibinbigay ng Wikipedia ang mga nilalaman nito ng walang bayad, walang patalastas, at hindi nagsasamantala sa paggamit ng personal na datos ng mga gumagamit nito.

12. One sentence of [Nelson Mandela's speech](https://www.tagaloglang.com/talumpati-ni-nelson-mandela/) in Filipino

    > Ang ating mga nagawa bilang ordinaryong mamamayan ng Timog Africa ay kailangang magbunga ng tunay na mamamayan nito na magpapalawak sa paniniwala ng sangkatauhan sa katarungan, magpapalakas sa tiwala sa kadakilaan ng kaluluwa, at magtutustos sa lahat ng ating pag-asa sa kapakinabangan ng buhay ng lahat.

13. [KMJS Article](https://www.gmanetwork.com/news/balitambayan/umg/987386/drawer-ng-cabinet-minulto-nga-ba-matapos-na-mahuli-cam-na-nagbukas-sara/story/ )

    > Nabalot ng kababalaghan ang masaya sanang bonding ng magkakaibigan nang bigla na lang magbukas-sara na mag-isa sa kanilang harapan ang drawer ng isang cabinet. Ang kinaroroonan ng cabinet, isang bahay-bakasyunan na pinaparentahan at kamamatay lang umano ng may-ari.

14. [BINI Article](https://bandera.inquirer.net/444456/bini-jhoanna-kinabog-weather-report-sa-good-day-la-achieve-sa-bucket-list)

    > NATUPAD ang isa sa bucket list ng BINI leader na si Jhoanna Robles, habang nasa Amerika. Biglaan kasi siyang naging weather presenter nang mag-guest ang nation's girl group sa morning show na Good Day LA, kung saan una nilang ibinahagi ang kanilang makasaysayang performance sa Coachella, pati na rin ang kanilang bagong EP na Signals at nalalapit na world tour. Pero imbes na matapos lang sa chikahan, biglang nagkaroon ng nakakatuwang twist!

# Results
## Tokens Used
|  Sentence № | GPT-2 | GPT-4o | filipino-tokenizer | Talasalitaan 
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 1  | 12  | 8  | 9  | ***6***  |
| 2  | 11  | 8  | 7  | ***4***  |
| 3  | 10  | 6  | 7  | ***3***  |
| 4  | 12  | 7  | ***3***  | 5  |
| 5  | 11  | 8  | 11  | ***5***  |
| 6  | 10  | 7  | 14  | ***6***  |
| 7  | 7  | 6  | 12  | ***5***  |
| 8  | 69  | 46  | 81  | ***27***  |
| 9  | 150  | 120  | 192  | ***94***  |
| 10  | 78  | 59  | 99  | ***39***  |
| 11  | 269  | 192  | 324  | ***174***  |
| 12  | 115  | 85  | 144  | ***58***  |
| 13  | 90  | 72  | 114  | ***63***  |
| 14  | 143  | ***106***  | 199  | 139  |
