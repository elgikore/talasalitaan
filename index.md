---
title: Talasalitaan, or How Did a Tokenizer Performed Suprisingly Well on Corpora Roughly the Size of Tiny Shakespere
layout: page
---

Talasalitaan (lit. vocabulary, but usage is more like a glossary/dictionary) is a vanilla BPE model from SentencePiece trained on the [KapitBisig](https://www.kapitbisig.com/philippines) site, using only Filipino as much as possible. The entire corpus is just 1.3 MB, but as this page will show later on, it is surprisingly performant. Token counts are also reduced.

The whole code for Talasalitaan is a simple SentencePiece wrapper, but the highlight of this article is the corpora, not the architecture.

* TOC
{:toc}

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

<details>
<summary>1. Greeting</summary>

<br>
  
> Kamusta, mga kababayan!

<br>
</details>
<details>
<summary>2. Long affixes (This word is seen during training, but it acts as a sanity check on whether this model mastered it at all)</summary>

<br>
  
> pagpapanibagong-tatag

<br>
</details>
<details>
<summary>3. The famous tongue-twister</summary>

<br>
  
> nakakapagpabagabag

<br>
</details>
<details>
<summary>4. Number 3's sidekick</summary>

<br>
  
> pinakanakapagpapabagabag


<br>
</details>
<details>
<summary>5. Good morning</summary>

<br>
  
> Magandang umaga, kapatid!


<br>
</details>
<details>
<summary>6. Simple sentence</summary>

<br>
  
> Kumain siya ng pagkain.


<br>
</details>
<details>
<summary>7. Simple question</summary>

<br>
  
> kumain ka na ba?


<br>
</details>
<details>
<summary>8. Full sentence from <a href="https://www.kapitbisig.com/philippines/information/arts-and-literature-mga-kuwentong-bayan-folktales_190.html">KapitBisig</a> (This word is seen during training, but it acts as a sanity check on whether this model mastered it at all)</summary>

<br>
  
> Ito ay pagsasalaysay ng mga katutubo sa kanilang paniniwalang lakas ng pisikal na kapaligiran at lakas ng pananampalataya ng lumilimbag sa kanilang buhay at kapalaran.


<br>
</details>
<details>
<summary>9. Historical Wikipedia <a href="https://tl.wikipedia.org/wiki/Kasaysayan_ng_Pilipinas_(1565%E2%80%931898)#Pagdating_ni_Ruy_L%C3%B3pez_de_Villalobos">article</a> sentence</summary>

<br>
  
> Ang unang paglalayag na pambuong mundo sa ngalan ng Espanya ay nasundan ng apat pang mga ekspedisyon mula 1525 hanggang 1542. Sa ikaapat na panggagalugad, narating ni Ruy Lopez de Villalobos ang Kapuluan ng Pilipinas at pinangalanan niya ang mga pulo mula kay Philip II na noon ay may katayuan bilang tagapagmana ng trono ng Kaharian ng Espanya, bagaman hindi pa pormal na naitatag ang Pilipinas bilang opisyal na teritoryo ng Espanya.


<br>
</details>
<details>
<summary>10. Declaration of Human Rights Preamble in Filipino</summary>

<br>
  
> Sapagkat ang pagkilala sa katutubong karangalan at sa pantay at di-maikakait na mga karapatan ng lahat ng nabibilang sa angkan ng tao ay siyang saligan ng kalayaan, katarungan at kapayapaan sa daigdig.


<br>
</details>
<details>
<summary>11. <a href="https://tl.wikipedia.org/wiki/Unang_Pahina#Patungkol">Patungkol</a> ng Wikipedia</summary>

<br>
  
> Ang Wikipedia ay isang proyektong online na ensiklopedya na panlahat, nakasulat sa maraming wika, at pinagtutulungan ang paggawa ng mga artikulo sa prinsipyong wiki. Naglalayon ang proyektong ito na mag-alok ng mga nilalaman na malayang muling magagamit, walang pinapanigan, at napapatunayan, na maaring baguhin at mapabuti ninuman. Nakikilala ang Wikipedia sa pamamagitan ng mga naitatag na prinsipyo. Nakalisensiya ang nilalaman nito sa ilalim ng Creative Commons BY-SA. Maari itong kopyahin at muling gamitin sa ilalim ng parehong lisensiya, na sumasailalim sa paggalang sa mga kondisyon. Ibinbigay ng Wikipedia ang mga nilalaman nito ng walang bayad, walang patalastas, at hindi nagsasamantala sa paggamit ng personal na datos ng mga gumagamit nito.


<br>
</details>
<details>
<summary>12. One sentence of <a href="https://www.tagaloglang.com/talumpati-ni-nelson-mandela/">Nelson Mandela's speech</a> in Filipino</summary>

<br>
  
> Ang ating mga nagawa bilang ordinaryong mamamayan ng Timog Africa ay kailangang magbunga ng tunay na mamamayan nito na magpapalawak sa paniniwala ng sangkatauhan sa katarungan, magpapalakas sa tiwala sa kadakilaan ng kaluluwa, at magtutustos sa lahat ng ating pag-asa sa kapakinabangan ng buhay ng lahat.


<br>
</details>
<details>
<summary>
13. <a href="https://www.gmanetwork.com/news/balitambayan/umg/987386/drawer-ng-cabinet-minulto-nga-ba-matapos-na-mahuli-cam-na-nagbukas-sara/story/">KMJS Article</a></summary>

<br>
  
> Nabalot ng kababalaghan ang masaya sanang bonding ng magkakaibigan nang bigla na lang magbukas-sara na mag-isa sa kanilang harapan ang drawer ng isang cabinet. Ang kinaroroonan ng cabinet, isang bahay-bakasyunan na pinaparentahan at kamamatay lang umano ng may-ari.


<br>
</details>
<details>
<summary>
14. <a href="https://bandera.inquirer.net/444456/bini-jhoanna-kinabog-weather-report-sa-good-day-la-achieve-sa-bucket-list">BINI Article</a></summary>

<br>
  
> NATUPAD ang isa sa bucket list ng BINI leader na si Jhoanna Robles, habang nasa Amerika. Biglaan kasi siyang naging weather presenter nang mag-guest ang nation's girl group sa morning show na Good Day LA, kung saan una nilang ibinahagi ang kanilang makasaysayang performance sa Coachella, pati na rin ang kanilang bagong EP na Signals at nalalapit na world tour. Pero imbes na matapos lang sa chikahan, biglang nagkaroon ng nakakatuwang twist!


<br>
</details>

# Results
## Tokens Used

|  Sentence № | GPT-2 | GPT-4o | filipino-tokenizer | Talasalitaan 
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 1 | 12 | 8 | 9 | ***6*** |
| 2 | 11 | 8 | 7 | ***4*** |
| 3 | 10 | 6 | 7 | ***3*** |
| 4 | 12 | 7 | ***3*** | 5 |
| 5 | 11 | 8 | 11 | ***5*** |
| 6 | 10 | 7 | 14 | ***6*** |
| 7 | 7 | 6 | 12 | ***5*** |
| 8 | 69 | 46 | 81 | ***27*** |
| 9 | 150 | 120 | 192 | ***94*** |
| 10 | 78 | 59 | 99 | ***39*** |
| 11 | 269 | 192 | 324 | ***174*** |
| 12 | 115 | 85 | 144 | ***58*** |
| 13 | 90 | 72 | 114 | ***63*** |
| 14 | 143 | ***106*** | 199 | 139 |

> **NOTE:**
> ***Bold and italic*** is the lowest recorded token count

## Word-to-Token Ratio for Long Sentences

|  Sentence № | № of Words (Theoretical Floor) | GPT-2 | GPT-4o | filipino-tokenizer | Talasalitaan 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 8  | 25  | 2.76  | 1.84  | 3.24  | ***1.08***  |
| 9  | 72  | 2.08  | 1.67  | 2.67  | ***1.31***  |
| 10  | 32  | 2.44  | 1.84  | 3.09  | ***1.22***  |
| 11  | 110  | 2.45  | 1.75  | 2.95  | ***1.58***  |
| 12  | 46  | 2.50  | 1.85  | 3.13  | ***1.26***  |
| 13  | 38  | 2.37  | 1.89  | 3.00  | ***1.66***  |
| 14  | 72  | 1.99  | ***1.47***  | 2.76  | 1.93  |

> **NOTE:**
> Lower means better compressed.
> Theoretical ratio is 1.0, meaning one word = one token.

## Tokens Generated By Talasalitaan
<details>
<summary>Sentence №1</summary>

    ['▁Kam', 'usta', ',', '▁mga', '▁kababayan', '!']

</details>
<details>
<summary>Sentence №2</summary>

    ['▁pagpap', 'anibagong', '-', 'tatag']

</details>
<details>
<summary>Sentence №3</summary>

    ['▁nakak', 'apagp', 'abagabag']

</details>
<details>
<summary>Sentence №4</summary>

    ['▁pinakan', 'ak', 'apag', 'pap', 'abagabag']

</details>
<details>
<summary>Sentence №5</summary>

    ['▁Magandang', '▁umaga', ',', '▁kapatid', '!']

</details>
<details>
<summary>Sentence №6</summary>

    ['▁Kum', 'ain', '▁siya', '▁ng', '▁pagkain', '.']

</details>
<details>
<summary>Sentence №7</summary>

    ['▁kumain', '▁ka', '▁na', '▁ba', '?']

</details>
<details>
<summary>Sentence №8</summary>

    ['▁Ito', '▁ay', '▁pagsasalaysay', '▁ng', '▁mga', '▁katutubo', '▁sa', '▁kanilang', '▁paniniwalang',
    '▁lakas', '▁ng', '▁pisikal', '▁na', '▁kapaligiran', '▁at', '▁lakas', '▁ng', '▁pananampalataya', '▁ng',
    '▁lum', 'ilimbag', '▁sa', '▁kanilang', '▁buhay', '▁at', '▁kapalaran', '.']

</details>
<details>
<summary>Sentence №9</summary>

    ['▁Ang', '▁unang', '▁paglalayag', '▁na', '▁pamb', 'uong', '▁mundo', '▁sa', '▁ngalan', '▁ng',
    '▁Espanya', '▁ay', '▁nasundan', '▁ng', '▁apat', '▁pang', '▁mga', '▁eks', 'p', 'ed', 'isyon', '▁mula',
    '▁15', '2', '5', '▁hanggang', '▁15', '4', '2.', '▁Sa', '▁ikaapat', '▁na', '▁pangg', 'agal', 'ugad',
    ',', '▁narating', '▁ni', '▁R', 'uy', '▁Lopez', '▁de', '▁Vill', 'al', 'ob', 'os', '▁ang', '▁Kapuluan',
    '▁ng', '▁Pilipinas', '▁at', '▁pinangalan', 'an', '▁niya', '▁ang', '▁mga', '▁pulo', '▁mula', '▁kay',
    '▁Ph', 'ilip', '▁II', '▁na', '▁noon', '▁ay', '▁may', '▁katayuan', '▁bilang', '▁tagapagmana', '▁ng',
    '▁trono', '▁ng', '▁Kah', 'arian', '▁ng', '▁Espanya', ',', '▁bagaman', '▁hindi', '▁pa', '▁p', 'ormal',
    '▁na', '▁nait', 'atag', '▁ang', '▁Pilipinas', '▁bilang', '▁opisyal', '▁na', '▁teritoryo', '▁ng',
    '▁Espanya', '.']

</details>
<details>
<summary>Sentence №10</summary>

    ['▁Sapagkat', '▁ang', '▁pagkilala', '▁sa', '▁katutubong', '▁karangalan', '▁at', '▁sa', '▁pantay',
    '▁at', '▁di', '-', 'ma', 'ik', 'aka', 'it', '▁na', '▁mga', '▁karapatan', '▁ng', '▁lahat', '▁ng',
    '▁nabibilang', '▁sa', '▁angkan', '▁ng', '▁tao', '▁ay', '▁siyang', '▁saligan', '▁ng', '▁kalayaan',
    ',', '▁katarungan', '▁at', '▁kapayapaan', '▁sa', '▁daigdig', '.']

</details>
<details>
<summary>Sentence №11</summary>

    ['▁Ang', '▁W', 'ik', 'ipe', 'dia', '▁ay', '▁isang', '▁proy', 'ekt', 'ong', '▁on', 'l', 'ine', 
    '▁na', '▁en', 'sik', 'l', 'op', 'edya', '▁na', '▁panlahat', ',', '▁nakasulat', '▁sa', '▁maraming'
    , '▁wika', ',', '▁at', '▁pinagt', 'utulungan', '▁ang', '▁paggawa', '▁ng', '▁mga', '▁ar', 'tik',
    'ulo', '▁sa', '▁prins', 'ipyong', '▁w', 'iki', '.', '▁Nagl', 'alayon', '▁ang', '▁proy', 'ekt',
    'ong', '▁ito', '▁na', '▁mag', '-', 'alok', '▁ng', '▁mga', '▁nilalaman', '▁na', '▁malayang',
    '▁muling', '▁magagamit', ',', '▁walang', '▁pinapan', 'igan', ',', '▁at', '▁napapat', 'unayan',
    ',', '▁na', '▁maaring', '▁baguhin', '▁at', '▁map', 'abuti', '▁ninuman', '.', '▁Nakikilala', '▁ang',
    '▁W', 'ik', 'ipe', 'dia', '▁sa', '▁pamamagitan', '▁ng', '▁mga', '▁nait', 'atag', '▁na', '▁prins',
    'ip', 'yo', '.', '▁Nak', 'alis', 'ensiya', '▁ang', '▁nilalaman', '▁nito', '▁sa', '▁ilalim', '▁ng',
    '▁C', 're', 'ative', '▁Comm', 'ons', '▁B', 'Y', '-', 'SA', '.', '▁Maari', '▁itong', '▁ko', 'p',
    'y', 'ahin', '▁at', '▁muling', '▁gamitin', '▁sa', '▁ilalim', '▁ng', '▁parehong', '▁l', 'is', 
    'ensiya', ',', '▁na', '▁sumasa', 'ilalim', '▁sa', '▁paggalang', '▁sa', '▁mga', '▁kondisyon', '.',
    '▁Ibin', 'bigay', '▁ng', '▁W', 'ik', 'ipe', 'dia', '▁ang', '▁mga', '▁nilalaman', '▁nito', '▁ng',
    '▁walang', '▁bayad', ',', '▁walang', '▁pat', 'alastas', ',', '▁at', '▁hindi', '▁nagsas',
    'amantala', '▁sa', '▁paggamit', '▁ng', '▁personal', '▁na', '▁datos', '▁ng', '▁mga', '▁gumagamit',
    '▁nito', '.']

</details>
<details>
<summary>Sentence №12</summary>

    ['▁Ang', '▁ating', '▁mga', '▁nagawa', '▁bilang', '▁ordin', 'aryong', '▁mamamayan', '▁ng', '▁Timog',
    '▁A', 'f', 'r', 'ica', '▁ay', '▁kailangang', '▁magbunga', '▁ng', '▁tunay', '▁na', '▁mamamayan',
    '▁nito', '▁na', '▁magpap', 'alawak', '▁sa', '▁paniniwala', '▁ng', '▁sangkatauhan', '▁sa',
    '▁katarungan', ',', '▁magpap', 'alakas', '▁sa', '▁tiwala', '▁sa', '▁kadakilaan', '▁ng', '▁kaluluwa',
    ',', '▁at', '▁magtut', 'ustos', '▁sa', '▁lahat', '▁ng', '▁ating', '▁pag', '-', 'asa', '▁sa',
    '▁kapakinabangan', '▁ng', '▁buhay', '▁ng', '▁lahat', '.']

</details>
<details>
<summary>Sentence №13</summary>

    ['▁Nab', 'alot', '▁ng', '▁kababalaghan', '▁ang', '▁masaya', '▁sanang', '▁b', 'on', 'ding', '▁ng',
    '▁magkakaibigan', '▁nang', '▁bigla', '▁na', '▁lang', '▁magbukas', '-', 's', 'ara', '▁na', '▁mag',
    '-', 'isa', '▁sa', '▁kanilang', '▁harapan', '▁ang', '▁dr', 'aw', 'er', '▁ng', '▁isang', '▁c',
    'abin', 'et', '.', '▁Ang', '▁kinaroroonan', '▁ng', '▁c', 'abin', 'et', ',', '▁isang', '▁bahay',
    '-', 'bakasyunan', '▁na', '▁pinap', 'ar', 'ent', 'ahan', '▁at', '▁kam', 'amatay', '▁lang',
    '▁umano', '▁ng', '▁may', '-', 'ari', '.']

</details>
<details>
<summary>Sentence №14</summary>

    ['▁N', 'AT', 'UP', 'AD', '▁ang', '▁isa', '▁sa', '▁bu', 'c', 'ket', '▁l', 'ist', '▁ng', '▁B', 'INI',
    '▁le', 'ad', 'er', '▁na', '▁si', '▁J', 'ho', 'an', 'na', '▁R', 'ob', 'les', ',', '▁habang', '▁nasa',
    '▁Amerika', '.', '▁B', 'igl', 'aan', '▁kasi', '▁siyang', '▁naging', '▁w', 'eat', 'h', 'er', '▁pres',
    'enter', '▁nang', '▁mag', '-', 'g', 'uest', '▁ang', '▁nat', 'ion', "'", 's', '▁g', 'ir', 'l', '▁g',
    'ro', 'up', '▁sa', '▁m', 'orn', 'ing', '▁s', 'h', 'ow', '▁na', '▁G', 'ood', '▁Day', '▁L', 'A', ',',
    '▁kung', '▁saan', '▁una', '▁nilang', '▁ibin', 'ahagi', '▁ang', '▁kanilang', '▁makasaysayang', '▁per',
    'f', 'or', 'man', 'ce', '▁sa', '▁Co', 'ach', 'el', 'la', ',', '▁pati', '▁na', '▁rin', '▁ang',
    '▁kanilang', '▁bagong', '▁E', 'P', '▁na', '▁"', 'Sig', 'nal', 's', '"', '▁at', '▁nal', 'alapit',
    '▁na', '▁w', 'or', 'l', 'd', '▁to', 'ur', '.', '▁Pero', '▁imb', 'es', '▁na', '▁matapos', '▁lang',
    '▁sa', '▁c', 'hik', 'ahan', ',', '▁biglang', '▁nagkaroon', '▁ng', '▁nakakat', 'uwang', '▁t', 'w',
    'ist', '!']

</details>

## First and last 100 tokens

<details>
<summary>Token vocabulary</summary>

|  First 100 | Last 100 |
| ------------- | ------------- |
| ng	0	| ▁napasabihan	-32594
| ang	-1 |	▁napipinsala	-32595
| an	-2 |	▁napipintong	-32596
| ▁n	-3 |	▁napuluputan	-32597
| ▁s	-4 |	▁napupuhunan	-32598
| at	-5 |	▁nararanasan	-32599
| ag	-6 |	Pepe	-32600
| ▁m	-7 |	Sari	-32601
| al	-8 |	Totoo	-32602
| in	-9 |	ipala	-32603
| ay	-10 |	lagda	-32604
| ▁k	-11 |	nahan	-32605
| ▁p	-12 |	▁Laws	-32606
| ▁ng	-13 |	▁tuba	-32607
| ▁sa	-14 |	▁utal	-32608
| ▁na	-15 |		abasan	-32609
| ▁ang	-16 |		ihimay	-32610
| ak	-17 |		ilipos	-32611
| it	-18 |		lahing	-32612
| as	-19	 |	sakong	-32613
| il	-20	 |	uester	-32614
| am	-21 |		unsini	-32615
| ar	-22 |		▁Unawa	-32616
| ap	-23 |		▁iigsi	-32617
| ing	-24 |		▁lilik	-32618
| ▁b	-25 |		▁tabug	-32619
| ong	-26 |		▁tubis	-32620
| ah	-27 |		▁Niluto	-32621
| aw	-28 |		▁Sybyla	-32622
| iy	-29 |		▁bagsak	-32623
| ab	-30 |		▁inihim	-32624
| ▁at	-31 |		▁langka	-32625
| ▁d	-32 |		▁legwas	-32626
| ▁t	-33 |		▁saksak	-32627
| ▁l	-34 |		▁sugong	-32628
| ga	-35 |		▁Kaipala	-32629
| ul	-36 |		▁Matulin	-32630
| is	-37 |		▁Nakakap	-32631
| ▁mga	-38 |		▁Panitik	-32632
| on	-39 |		▁hihimay	-32633
| ▁ay	-40 |		▁ikahiya	-32634
| ▁A	-41 |		▁inuunan	-32635
| un	-42 |		▁madilig	-32636
| um	-43 |		▁masilip	-32637
| ▁h	-44 |		▁nahilig	-32638
| ig	-45 |		▁naitala	-32639
| ▁S	-46 |		▁palatok	-32640
| ▁pag	-47 |		▁tinapon	-32641
| ▁P	-48 |		▁tubigan	-32642
| ad	-49 |		▁umampon	-32643
| ik	-50 |		amamalagi	-32644
| ▁N	-51 |		▁Alunsini	-32645
| ▁K	-52 |		▁Dimatiga	-32646
| ▁M	-53 |		▁Pagsipot	-32647
| ib	-54 |		▁bahalang	-32648
| iya	-55 |		▁dadalhan	-32649
| ung	-56 |		▁kainipan	-32650
| ip	-57 |		▁magbalag	-32651
| ▁kan	-58 |		▁maputing	-32652
| ▁I	-59 |		▁maunahan	-32653
| ▁mag	-60 |		▁nakasiya	-32654
| ▁nag	-61 |		▁paglilip	-32655
| ▁si	-62 |		▁pasyente	-32656
| ▁ni	-63 |		▁Hinagupit	-32657
| ▁g	-64 |		▁kaikalawa	-32658
| us	-65 |		▁lilikumin	-32659
| ito	-66 |		▁nakatawag	-32660
| ▁D	-67 |		▁nakatungo	-32661
| im	-68 |		▁nakilaban	-32662
| ut	-69 |		▁nalilipos	-32663
| ▁Ang	-70 |		▁pagtaghoy	-32664
| ▁is	-71 |		▁Naglabasan	-32665
| ala	-72 |		▁Nakakapaso	-32666
| di	-73 |		▁Napakabuti	-32667
| ▁B	-74 |		▁kabulukang	-32668
| uh	-75 |		▁maitimbang	-32669
| ilang	-76 |		▁nagparungg	-32670
| uk	-77 |		▁nagtatahan	-32671
| ▁T	-78 |		▁nakatataas	-32672
| os	-79 |		▁napaloloko	-32673
| er	-80 |		▁pamilihang	-32674
| ▁kany	-81 |		▁saliksikin	-32675
| apat	-82 |		▁sumusulong	-32676
| and	-83 |		▁(1971-1972)	-32677
| ▁H	-84 |		▁inihimatong	-32678
| up	-85 |		▁naririmarim	-32679
| ari	-86 |		▁narurumihan	-32680
| or	-87 |		▁nasaksihang	-32681
| indi	-88 |		▁nasasabugan	-32682
| ila	-89 |		▁nasasangkap	-32683
| ▁L	-90 |		▁nasusubukan	-32684
| ▁isang	-91 |		▁nasusuungan	-32685
| ub	-92 |		▁nataguriang	-32686
| ▁"	-93 |		▁natambangan	-32687
| ▁kanyang	-94 |		▁natatagalan	-32688
| ▁nang	-95 |		▁natitigatig	-32689
| uw	-96 |		▁natititigan	-32690
| en	-97 |		▁natuklasang	-32691
| ur	-98 |		▁natutularan	-32692
| ot	-99	 |	▁natutulayan	-32693
| ▁siya	-100 |		▁natututuhan	-32694

</details>


> **NOTE:**
> It is interesting that at the tail end of the `.vocab` file, it still produced meaningful tokens for most of them.
