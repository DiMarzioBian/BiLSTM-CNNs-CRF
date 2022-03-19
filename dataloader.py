import codecs
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import load_sentences, update_tag_scheme, prepare_dataset


class CoNLLData(Dataset):
    """ CoNLLData Dataset """
    def __init__(self, args, data):
        self.data = data
        self.length = len(data)
        self.max_len_word = args.max_len_word
        self.idx_pad_char = args.idx_pad_char

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        chars_padded = [x + [self.idx_pad_char]*(self.max_len_word - len(x)) for x in self.data[index]['chars']]
        return \
            torch.LongTensor(self.data[index]['words']), torch.LongTensor(chars_padded), \
            torch.LongTensor(self.data[index]['tags'])


def collate_fn(insts, args):
    """ Batch preprocess """
    words_batch, chars_batch, tags_batch = list(zip(*insts))

    # get sorted indices
    lens = torch.as_tensor([v.size(0) for v in words_batch])
    sorted_lens, sorted_indices = torch.sort(lens, descending=True)

    # sort data
    words_batch = pad_sequence(words_batch, batch_first=True, padding_value=args.idx_pad_word)
    words_batch = words_batch.index_select(0, sorted_indices)

    chars_batch = pad_sequence(chars_batch, batch_first=True, padding_value=args.idx_pad_char)
    chars_batch = chars_batch.index_select(0, sorted_indices)

    tags_batch = pad_sequence(tags_batch, batch_first=True, padding_value=args.idx_pad_tag)
    tags_batch = tags_batch.index_select(0, sorted_indices)

    return words_batch, chars_batch, tags_batch, sorted_lens


def get_dataloader(args, word2idx, tag2idx, char2idx):
    """ Get dataloader """
    train_data = load_sentences(args.path_data+'eng.train', args.digi_zero, filter_word=args.filter_word)
    valid_data = load_sentences(args.path_data+'eng.testa', args.digi_zero, filter_word=args.filter_word)
    test_data = load_sentences(args.path_data+'eng.testb', args.digi_zero, filter_word=args.filter_word)

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

