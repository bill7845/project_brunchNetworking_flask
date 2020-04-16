from soynlp.tokenizer import LTokenizer


ltk = LTokenizer()

def ltk_tokenizer(text):
    tokens_ltk = ltk.tokenize(text,flatten=True)
    return tokens_ltk
