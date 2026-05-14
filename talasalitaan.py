import sentencepiece as spm
import subprocess

NAME = "talasalitaan"

class Tasalitaan:
    """Minimal adapter for SentencePiece, built for Filipino tokenization."""
    
    def __init__(self):
        self.__MODEL_PREFIX = NAME
        self.__MODEL_FILE = NAME + ".model"
    
    def train(self, corpus_folder: str) -> None:
        result = subprocess.run(
                    f"cat \"{corpus_folder}\"/*.txt",
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True
                )
