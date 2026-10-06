class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"


    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        self.word_to_id, self.id_to_word = {}, {}
        texts = [text.lower() for text in texts]
        all_texts = []
        for text in texts:
            all_texts = all_texts+text.split()
        
        
        all_texts = list(set(all_texts))
        all_texts.sort()

        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in range(len(special_tokens)):
            self.word_to_id[special_tokens[token]] = token
        
        self.id_to_word = {value: key for key, value in self.word_to_id.items()}

        
        for text in range(len(all_texts)):
            self.word_to_id[all_texts[text].lower()] = 4 + text

        self.id_to_word = {value: key for key, value in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id.keys())
    
    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        texts = text.split()
        texts = [text.lower() for text in texts]
        
        tokens = []
        for i in texts:
          tokens.append(
                self.word_to_id.get(
                    i, self.word_to_id[self.unk_token]
                )
            )
        return tokens
        
    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        tokens = []
        for id in ids:
            tokens.append(
                self.id_to_word.get(id, self.id_to_word[1])
            )
        return ' '.join(tokens)