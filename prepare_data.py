import argparse
import codecs
import pickle
import numpy as np

from utils import load_sentences, update_tag_scheme, prepare_dataset

START_TAG = '<START>'
STOP_TAG = '<STOP>'


def main():
    parser = argparse.ArgumentParser(description='BiLSTM-CNNs-CRF project')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--word_dim', type=int, default=100,
                        help='token embedding dimension')
    parser.add_argument('--path_data', type=str, default='./data/raw/',
                        help='location of the data corpus')
    parser.add_argument('--path_embedding', type=str, default='./data/raw/glove.6B.100d.txt',
                        help='path to save embedding file')
    parser.add_argument('--path_processed', type=str, default='./data/data_bundle.pkl',
                        help='path to save the processed data')
    parser.add_argument('--path_filtered', type=str, default='./data/data_filtered_bundle.pkl',
                        help='path to save the filtered processed data')

    # settings
    parser.add_argument('--filter_word', type=bool, default=True,
                        help='filter meaningless words')
    parser.add_argument('--tag_scheme', type=str, default='BIOES',
                        help='BIO or BIOES')
    parser.add_argument('--is_lowercase', type=bool, default=True,
                        help='control lowercasing of words')
    parser.add_argument('--digi_zero', type=bool, default=True,
                        help='control replacement of  all digits by 0')
    args = parser.parse_args()

    # preprocess
    print('\n[info] Data preprocess starts...')
    train_sentences = load_sentences(args.path_data + 'eng.train', args.digi_zero, filter_word=args.filter_word)
    valid_sentences = load_sentences(args.path_data + 'eng.testa', args.digi_zero, filter_word=args.filter_word)
    test_sentences = load_sentences(args.path_data + 'eng.testb', args.digi_zero, filter_word=args.filter_word)

    update_tag_scheme(train_sentences, args.tag_scheme)
    update_tag_scheme(valid_sentences, args.tag_scheme)
    update_tag_scheme(test_sentences, args.tag_scheme)

    dico_words, word2idx, idx2word = word_mapping(train_sentences, args.is_lowercase)
    dico_chars, char2idx, idx2char = char_mapping(train_sentences)
    dico_tags, tag2idx, idx2tag = tag_mapping(train_sentences)

    train_data = prepare_dataset(train_sentences, word2idx, char2idx, tag2idx, args.is_lowercase)
    dev_data = prepare_dataset(valid_sentences, word2idx, char2idx, tag2idx, args.is_lowercase)
    test_data = prepare_dataset(test_sentences, word2idx, char2idx, tag2idx, args.is_lowercase)
    print(
        'Loaded {} / {} / {} sentences in train / valid / test.'.format(len(train_data), len(dev_data), len(test_data)))

    # load embeddings
    print('\n[info] Load embeddings...')
    all_embeds_word = {}
    for i, line in enumerate(codecs.open(args.path_embedding, 'r', 'utf-8')):
        s = line.strip().split()
        if len(s) == (args.word_dim + 1):
            all_embeds_word[s[0]] = np.array([float(i) for i in s[1:]])

    embeds_word = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word2idx), args.word_dim))  # initializing
    for w in word2idx:
        if w in all_embeds_word:
            embeds_word[word2idx[w]] = all_embeds_word[w]
        elif w.lower() in all_embeds_word:
            embeds_word[word2idx[w]] = all_embeds_word[w.lower()]
    print('Loaded %i pretrained embeddings.' % len(all_embeds_word))

    # save pickle file
    print('\n[info] Saving files...')
    if not args.filter_word:
        with open(args.path_processed, 'wb') as f:
            mappings = {
                'word2idx': word2idx,
                'char2idx': char2idx,
                'tag2idx': tag2idx,
                'embeds_word': embeds_word
            }
            pickle.dump(mappings, f)
        print('\n[info] Preprocess finished.')
    else:
        with open(args.path_filtered, 'wb') as f:
            mappings = {
                'word2idx': word2idx,
                'char2idx': char2idx,
                'tag2idx': tag2idx,
                'embeds_word': embeds_word
            }
            pickle.dump(mappings, f)
        print('\n[info] Preprocess & filtering finished.')


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = []
    for s in sentences:
        words.append([x[0].lower() if lower else x[0] for x in s])
    dico = create_dico(words)
    dico['<UNK>'] = 10000000  # UNK tag for unknown words
    word2idx, idx2word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word2idx, idx2word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char2idx, idx2char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char2idx, idx2char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    dico[START_TAG] = -1
    dico[STOP_TAG] = -2
    tag2idx, idx2tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag2idx, idx2tag


if __name__ == '__main__':
    main()
