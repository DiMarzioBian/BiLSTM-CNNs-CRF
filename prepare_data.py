import argparse
import codecs
import re
import pickle
import numpy as np
from tqdm import tqdm

START_TAG = '<START>'
STOP_TAG = '<STOP>'


def main():
    parser = argparse.ArgumentParser(description='BLSTM-CNNs-CRF project')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--path_data', type=str, default='./data/raw/',
                        help='location of the data corpus')
    parser.add_argument('--path_embedding', type=str, default='./data/raw/glove.6B.100d.txt',
                        help='path to save embedding file')
    parser.add_argument('--path_processed', type=str, default='./data/data_bundle.pkl',
                        help='path to save the processed data')
    parser.add_argument('--word_dim', type=int, default=100,
                        help='token embedding dimension')

    # settings
    parser.add_argument('--tag_scheme', type=str, default='BIOES',
                        help='BIO or BIOES')
    parser.add_argument('--lowercase', type=bool, default=True,
                        help='control lowercasing of words')
    parser.add_argument('--zero_digits', type=bool, default=True,
                        help='control replacement of  all digits by 0 ')
    args = parser.parse_args()

    # preprocess
    print('\n[info] Data preprocess starts...')
    train_sentences = load_sentences(args.path_data + 'eng.train', args.zero_digits)
    valid_sentences = load_sentences(args.path_data + 'eng.testa', args.zero_digits)
    test_sentences = load_sentences(args.path_data + 'eng.testb', args.zero_digits)

    update_tag_scheme(train_sentences, args.tag_scheme)
    update_tag_scheme(valid_sentences, args.tag_scheme)
    update_tag_scheme(test_sentences, args.tag_scheme)

    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, args.lowercase)
    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    train_data = prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id, args.lowercase)
    dev_data = prepare_dataset(valid_sentences, word_to_id, char_to_id, tag_to_id, args.lowercase)
    test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, args.lowercase)
    print('Loaded {} / {} / {} sentences in train / valid / test.'.format(len(train_data), len(dev_data), len(test_data)))

    # load embeddings
    print('\n[info] Load embeddings...')
    all_word_embeds = {}
    for i, line in enumerate(codecs.open(args.path_embedding, 'r', 'utf-8')):
        s = line.strip().split()
        if len(s) == (args.word_dim + 1):
            all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), args.word_dim)) # initializing
    for w in word_to_id:
        if w in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w]
        elif w.lower() in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]
    print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

    # save pickle file
    print('\n[info] Saving files...')
    with open(args.path_processed, 'wb') as f:
        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'char_to_id': char_to_id,
            'parameters': args,
            'word_embeds': word_embeds
        }
        pickle.dump(mappings, f)
    print('\n[info] Preprocess finished.')


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


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


def iob2(tags):
    """
    Check that tags have a valid BIO format.
    Tags in BIO1 format are converted to BIO2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    the function is used to convert
    BIO -> BIOES tagging
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to BIO2
    Only BIO1 and BIO2 schemes are accepted for input data.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the BIO format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in BIO format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'BIOES':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Wrong tagging scheme!')


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
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000  # UNK tag for unknown words
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    dico[START_TAG] = -1
    dico[STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def lower_case(x, lower=False):
    if lower:
        return x.lower()
    else:
        return x


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[lower_case(w,lower) if lower_case(w,lower) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'tags': tags,
        })
    return data


if __name__ == '__main__':
    main()
