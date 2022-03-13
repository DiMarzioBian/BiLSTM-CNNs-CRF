import codecs
import torch
from torch.utils.data import Dataset, DataLoader

# import numpy as np
# import matplotlib.pyplot as plt


# Dictionary to store all words and their indices
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def load_sentences(path, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


class CoNLLData(Dataset):
    """ CoNLLData Dataset """
    def __init__(self, args, tks_file):
        self.initial_preprocess = args.initial_preprocess
        self.n_gram = args.n_gram
        self.tokens_file = tks_file
        if self.initial_preprocess:
            self.length = len(tks_file) // (args.n_gram+1)
        else:
            self.length = len(tks_file) - args.n_gram

        # EDA: plot frequency
        # tks = np.array(tks_file)
        # unique, counts = np.unique(tks, return_counts=True)
        # counts = counts[::-1]
        #
        # fig, axs = plt.subplots(2, 1)
        # axs[0].plot(counts)
        # axs[0].set_ylabel('Frequency')
        # axs[0].grid(True)
        #
        # axs[1].plot(np.log(counts))
        # axs[1].set_ylabel('Frequency in log')
        # axs[1].set_xlabel('Words / tokens')
        # axs[1].grid(True)
        #
        # fig.tight_layout()
        # plt.show()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.initial_preprocess:
            return self.tokens_file[index * self.n_gram: (index+1) * self.n_gram], \
                   self.tokens_file[(index+1) * self.n_gram]
        else:
            return self.tokens_file[index: index+self.n_gram], self.tokens_file[index+self.n_gram]


def collate_fn(insts):
    """ Batch preprocess """
    seq_tokens_batch, tgt_tokens_batch = list(zip(*insts))

    seq_tokens_batch = torch.LongTensor(seq_tokens_batch)
    tgt_tokens_batch = torch.LongTensor(tgt_tokens_batch)
    return seq_tokens_batch, tgt_tokens_batch


def get_dataloader(args, no_dataloader=False):
    """ Get dataloader and dictionary """
    train_data = load_sentences(args.path_data+'eng.train', args.zero_digit)
    valid_data = load_sentences(args.path_data+'eng.testa', args.zero_digit)
    test_sentences = load_sentences(args.path_data+'eng.testb', args.zero_digit)

    train_loader = DataLoader(CoNLLData(args, train_data), batch_size=args.batch_size,
                              num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(CoNLLData(args, valid_data), batch_size=args.batch_size,
                              num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(CoNLLData(args, test_sentences), batch_size=args.batch_size, num_workers=args.num_worker,
                             collate_fn=collate_fn, shuffle=True)
    return train_loader, valid_loader, test_loader

