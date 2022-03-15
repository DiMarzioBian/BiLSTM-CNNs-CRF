import codecs
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence

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
        self.max_len_word = args.max_len_word
        self.idx_pad_char = args.idx_pad_char

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        char_masked = []
        lens_char = []
        for word in self.data[index]['chars']:
            char_masked.append(word + [self.idx_pad_char] * (self.max_len_word - len(word)))
            lens_char.append(len(word))

        return \
            torch.LongTensor(self.data[index]['words']), \
            torch.LongTensor(char_masked), \
            torch.LongTensor(lens_char), \
            torch.LongTensor(self.data[index]['tags'])


def collate_fn(insts, args):
    """ Batch preprocess """
    words_batch, chars_batch, lens_char_batch, tags_batch = list(zip(*insts))

    if args.mode_word == 'lstm':
        words_pack = pack_sequence(words_batch, enforce_sorted=False)
    elif args.mode_word == 'cnn':
        words_pack = torch.stack(words_batch, dim=1)

    if args.mode_char == 'lstm':
        chars_pack = pack_sequence(chars_batch, enforce_sorted=False)
    elif args.mode_char == 'cnn':
        chars_pack = torch.stack(chars_batch, dim=1)

    tags_batch = torch.LongTensor(tags_batch)
    return words_pack, chars_pack, tags_batch


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
                              shuffle=True, collate_fn=lambda x: collate_fn(x, args))
    valid_loader = DataLoader(CoNLLData(args, valid_data), batch_size=args.batch_size, num_workers=args.num_worker,
                              shuffle=True, collate_fn=lambda x: collate_fn(x, args))
    test_loader = DataLoader(CoNLLData(args, test_data), batch_size=args.batch_size, num_workers=args.num_worker,
                             shuffle=True, collate_fn=lambda x: collate_fn(x, args))
    return train_loader, valid_loader, test_loader

