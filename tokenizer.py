import sentencepiece as spm
import subprocess, json


with open("settings.json", 'r', encoding="utf-8") as f:
    FOLDER_PATH = json.load(f)

print(FOLDER_PATH["path/to/corpus"])



# result = subprocess.run(
#     f"cat \"{folder_path}\"/*.txt",
#     shell=True,
#     check=True,
#     stdout=subprocess.PIPE,
#     text=True
# )

# with open("corpus.txt", "w", encoding="utf-8") as f:
#     f.write(result.stdout)

# spm.SentencePieceTrainer.train(
#     input="corpus.txt",      
#     model_prefix="munting_tokenayser",
#     vocab_size=16384,      
#     model_type="bpe")

# sp = spm.SentencePieceProcessor(model_file="munting_tokenayser.model")

# texts = [
#     "Kamusta, mga kababayan!", 
#     "pagpapanibagong-tatag", 
#     "nakakapagpabagabag", 
#     "Magandang umaga, kapatid!",
#     "Kumain siya ng pagkain.",
#     "kumain ka na ba?",
#     "Ito ay pagsasalaysay ng mga katutubo sa kanilang paniniwalang lakas ng pisikal na kapaligiran at lakas ng pananampalataya ng lumilimbag sa kanilang buhay at kapalaran.",
#     "Ang unang paglalayag na pambuong mundo sa ngalan ng Espanya ay nasundan ng apat pang mga ekspedisyon mula 1525 hanggang 1542. Sa ikaapat na panggagalugad, narating ni Ruy Lopez de Villalobos ang Kapuluan ng Pilipinas at pinangalanan niya ang mga pulo mula kay Philip II na noon ay may katayuan bilang tagapagmana ng trono ng Kaharian ng Espanya, bagaman hindi pa pormal na naitatag ang Pilipinas bilang opisyal na teritoryo ng Espanya.",
#     "Sapagkat ang pagkilala sa katutubong karangalan at sa pantay at di-maikakait na mga karapatan ng lahat ng nabibilang sa angkan ng tao ay siyang saligan ng kalayaan, katarungan at kapayapaan sa daigdig."
# ]

# for text in texts:
#     tokens = sp.encode(text, out_type=str)
#     ids = sp.encode(text, out_type=int)

#     print(tokens)
#     print(f"Length: {len(ids)}")

#     # Decode back
#     decoded = sp.decode(ids)
#     print(decoded)
#     print()