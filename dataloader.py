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
    def __init__(self, args, data):
        self.initial_preprocess = args.initial_preprocess
        self.data = data
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index]['words'], self.data[index]['chars'], self.data[index]['tags']


def collate_fn(insts, mode_char, mode_word):
    """ Batch preprocess """

    words_batch, chars_batch, tags_batch = list(zip(*insts))

    words_batch = torch.LongTensor(words_batch)
    chars_batch = torch.LongTensor(chars_batch)
    tags_batch = torch.LongTensor(tags_batch)
    return words_batch, chars_batch, tags_batch


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

    train_loader = DataLoader(CoNLLData(args, train_data), batch_size=args.batch_size, num_workers=args.num_worker,
                              shuffle=True, collate_fn=lambda x: collate_fn(x, args.mode_char, args.mode_word))
    valid_loader = DataLoader(CoNLLData(args, valid_data), batch_size=args.batch_size, num_workers=args.num_worker,
                              shuffle=True, collate_fn=lambda x: collate_fn(x, args.mode_char, args.mode_word))
    test_loader = DataLoader(CoNLLData(args, test_data), batch_size=args.batch_size, num_workers=args.num_worker,
                             shuffle=True, collate_fn=lambda x: collate_fn(x, args.mode_char, args.mode_word))
    return train_loader, valid_loader, test_loader

