import sentencepiece as spm
import subprocess, tempfile

NAME = "talasalitaan"

class Tasalitaan:
    """Minimal adapter for SentencePiece, built for Filipino tokenization."""
    
    def __init__(self):
        self.__MODEL_PREFIX = NAME
        self.__MODEL_FILE = NAME + ".model"
        self.spm_instance = spm.SentencePieceProcessor(model_file=self.__MODEL_FILE)
    
    def train(self, corpus_folder: str) -> None:
        result = subprocess.run(
            f"cat \"{corpus_folder}\"/*.txt",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            text=True
        )
    
        with tempfile.NamedTemporaryFile(mode='w', delete=True, suffix=".txt") as temp_file:
            temp_file.write(result.stdout)
            
            spm.SentencePieceTrainer.train(
                input=temp_file.name,      
                model_prefix=self.__MODEL_PREFIX,
                vocab_size=16384,      
                model_type="bpe"
            )
