import torch


''' SPECIAL TOKENS FOR MARKING THE SMILE STREAMS '''


class SpecialTokens:
    pad = '<pad>'
    unk = '<unk>'
    bos = '<bos>'
    eos = '<eos>'


# Confirmed: Takes list of strings ("one word")
def chars_to_id_vocab(data, tokens):
    char2idx = {}
    for t in tokens:
        if t not in char2idx:
            char2idx[t] = len(char2idx)

    for string in data:
        for char in string:
            if char not in char2idx:
                char2idx[char] = len(char2idx)
    # print(char2idx)
    return char2idx


''' END OF SPECIAL TOKENS CLASS '''


# This should work regards of data type. Takes the char2id dict and reverses the indexing
def id_to_char_vocab(c2i):
    id2char = {idx: char for (char, idx) in c2i.items()}
    return id2char


''' CHAR VOCAB CLASS '''


class CharVocab:

    def __init__(self, data, chars=None, ss=SpecialTokens):
        # Update tokens list as Special Tokens are updated
        tokens = [ss.pad, ss.unk, ss.bos, ss.eos]
        if chars:
            # update as you update tokens list
            if (ss.pad in chars) or (ss.unk in chars) or (ss.eos in chars) or (ss.bos in chars):
                raise ValueError('Special Token in chars')

        self.ss = ss
        self.c2i = chars_to_id_vocab(data, tokens)
        self.i2c = id_to_char_vocab(self.c2i)

    def __len__(self):
        return len(self.c2i)

    @property
    def pad(self):
        return self.c2i[self.ss.pad]

    @property
    def unk(self):
        return self.c2i[self.ss.unk]

    @property
    def bos(self):
        return self.c2i[self.ss.bos]

    @property
    def eos(self):
        return self.c2i[self.ss.eos]

    # Per char conversion
    def char2id(self, char):
        if char not in self.c2i:
            return self.unk
        return self.c2i[char]

    # Per id conversion
    def id2char(self, id):
        if id not in self.i2c:
            return self.ss.unk
        if id == self.c2i[self.ss.pad]:
            return ''
        return self.i2c[id]

    # Per string Conversion
    def string2ids(self, string, add_bos=False, add_eos=False):
        ids = [self.char2id(c) for c in string]

        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]

        return ids

    # Per id list conversion
    def ids2string(self, ids, rem_bos=True, rem_eos=True):
        # print(len(ids))
        if len(ids) <= 1:  # CORRECTION: if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]

        string = ''.join([self.id2char(id) for id in ids])
        return string


''' END OF CHAR VOCAB CLASS '''

''' EXTRA FUNCTIONALITY... '''


def string2tensor(string, vocab):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ids = vocab.string2ids(string, add_bos=True, add_eos=True)
    tensor = torch.tensor(
        ids, dtype=torch.long, device=device
    )
    return tensor


def property2tensor(property):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.tensor(
        property, dtype=torch.long, device=device
    )
    return tensor
