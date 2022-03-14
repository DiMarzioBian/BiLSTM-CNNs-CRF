import codecs
import torch
from torch.utils.data import Dataset, DataLoader

from utils import load_sentences, update_tag_scheme, prepare_dataset


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


class CoNLLData(Dataset):
    """ CoNLLData Dataset """
    def __init__(self, args, f_sentence):
        self.initial_preprocess = args.initial_preprocess
        self.f_sentence = f_sentence

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


def get_dataloader(args, word2idx, tag2idx, char2idx):
    """ Get dataloader and dictionary """
    train_data = load_sentences(args.path_data+'eng.train', args.digi_zero)
    valid_data = load_sentences(args.path_data+'eng.testa', args.digi_zero)
    test_data = load_sentences(args.path_data+'eng.testb', args.digi_zero)

    update_tag_scheme(train_data, args.tag_scheme)
    update_tag_scheme(valid_data, args.tag_scheme)
    update_tag_scheme(test_data, args.tag_scheme)

    train_data = prepare_dataset(train_data, word2idx, char2idx, tag2idx, args.is_lowercase)
    valid_data = prepare_dataset(valid_data, word2idx, char2idx, tag2idx, args.is_lowercase)
    test_data = prepare_dataset(test_data, word2idx, char2idx, tag2idx, args.is_lowercase)

    train_loader = DataLoader(CoNLLData(args, train_data), batch_size=args.batch_size,
                              num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(CoNLLData(args, valid_data), batch_size=args.batch_size,
                              num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(CoNLLData(args, test_data), batch_size=args.batch_size, num_workers=args.num_worker,
                             collate_fn=collate_fn, shuffle=True)
    return train_loader, valid_loader, test_loader

