import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from utils import init_embedding, init_lstm, init_linear


def log_sum_exp(vec):
    """
    This function calculates the score explained above for the forward algorithm
    vec 2D: 1 * size_tag
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax(vec):
    """
    This function returns the max index in a vector
    """
    _, idx = torch.max(vec, 1)
    return idx.view(-1).data.tolist()[0]


class BiLSTM_CRF(nn.Module):
    def __init__(self, args, word2idx, char2idx, tag2idx, glove_word=None):
        """
        Input parameters from args:

                args = Dictionary that maps NER tags to indices
                word2idx = Dimension of word embeddings (int)
                tag2idx = hidden state dimension
                char2idx = Dictionary that maps characters to indices
                glove_word = Numpy array which provides mapping from word embeddings to word indices
        """
        super(BiLSTM_CRF, self).__init__()
        self.START_TAG = args.START_TAG
        self.STOP_TAG = args.STOP_TAG
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.tag2idx = tag2idx
        self.n_word = len(word2idx)
        self.n_char = len(char2idx)
        self.n_tag = len(tag2idx)

        self.idx_pad_char = args.idx_pad_char
        self.idx_pad_word = args.idx_pad_word

        self.dim_emb_word = args.dim_emb_word
        self.dim_emb_char = args.dim_emb_char

        self.dim_lstm_char = args.dim_lstm_hidden
        self.dim_lstm_hidden = args.dim_lstm_hidden  # The hidden dimension of the LSTM layer (int)
        self.dim_out_char = args.dim_out_char  # dimension of the character embeddings

        self.mode_char = args.mode_char
        self.mode_word = args.mode_word

        # embedding layer
        self.embedding_char = nn.Embedding(self.n_char+1, self.dim_emb_char, padding_idx=self.idx_pad_char)
        init_embedding(self.embedding_char)

        if args.enable_pretrained:
            self.embedding_word = nn.Embedding.from_pretrained(torch.FloatTensor(glove_word), freeze=args.freeze_glove)
            self.embedding_word.padding_idx = self.idx_pad_word  # set pad word index
        else:
            self.embedding_word = nn.Embedding(self.n_word+1, self.dim_emb_word)
        init_embedding(self.embedding_word)

        # character encoder
        if self.mode_char == 'lstm':
            self.lstm_char = nn.LSTM(self.dim_emb_char, self.dim_lstm_char, num_layers=1, bidirectional=True)
            init_lstm(self.lstm_char)
        elif self.mode_char == 'cnn':
            self.cnn_char = nn.Conv2d(in_channels=1, out_channels=self.dim_out_char,
                                      kernel_size=(3, self.dim_emb_char), padding=(2, 0))
            init_linear(self.cnn_char)
        else:
            raise Exception('Character encoder mode unknown...')
        self.dropout = nn.Dropout(args.dropout)

        # word encoder
        if self.mode_word == 'lstm':
            self.lstm = nn.LSTM(self.dim_emb_word + self.dim_lstm_char * 2, self.dim_lstm_hidden, bidirectional=True)
        elif self.mode_word == 'cnn':
            self.lstm = nn.LSTM(self.dim_emb_word + self.dim_out_char, self.dim_lstm_hidden, bidirectional=True)
        else:
            raise Exception('Word encoder mode '+self.mode_char+'unknown...')
        init_lstm(self.lstm)

        # predictor
        self.hidden2tag = nn.Linear(self.dim_lstm_hidden * 2, self.n_tag)
        init_linear(self.hidden2tag)
        if args.enable_crf:
            self.transitions = nn.Parameter(torch.zeros(self.n_tag, self.n_tag))
            self.transitions.data[self.tag2idx[self.START_TAG], :] = -10000
            self.transitions.data[:, self.tag2idx[self.STOP_TAG]] = -10000

    def forward(self, words_batch, chars_batch, tags_batch, lens_batch):  # (self, sentence, tags, chars2, chars2_length, d):

        # character-level modelling
        emb_chars = self.embedding_char(chars_batch)
        if self.mode_char == 'lstm':
            packed = pack_padded_sequence(words_batch, lens_batch.cpu(), batch_first=True)
            lstm_out, _ = self.lstm_char(packed)

            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)

            outputs = outputs.transpose(0, 1)

            chars_embeds_temp = torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2))), requires_grad=True)

            chars_embeds_temp.to(sentence.device)

            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat(
                    (outputs[i, index - 1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))

            chars_embeds = chars_embeds_temp.clone()

            for i in range(chars_embeds.size(0)):
                chars_embeds[d[i]] = chars_embeds_temp[i]

        elif self.mode_char == 'cnn':
            chars_embeds = self.char_embeds(chars2)

            # Creating Character level representation using Convolutional Neural Netowrk
            # followed by a max pooling Layer
            chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3, kernel_size=(chars_cnn_out3.size(2), 1)).view(
                chars_cnn_out3.size(0), self.dim_out_char)
        else:
            raise Exception('Unknown character model...')

        # load word embeddings
        embeds = torch.cat((self.word_embeds(sentence), chars_embeds), 1).unsqueeze(1)
        embeds = self.dropout(embeds)

        # word lstm
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.dim_lstm_hidden * 2)
        lstm_out = self.dropout(lstm_out)
        feats = self.hidden2tag(lstm_out)

        if self.use_crf:
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            return forward_score - gold_score
        else:
            scores = nn.functional.cross_entropy(feats, tags)
            return scores

    def forward_alg(self, feats):
        """
        This function performs the forward algorithm explained above
        """
        # calculate in log domain
        # feats is len(sentence) * size_tag
        # initialize alpha with a Tensor with values all equal to -10000.

        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.size_tag).fill_(-10000.)

        # START_TAG has all score.
        init_alphas[0][self.tag2idx[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas.clone().to(feats.device)
        forward_var.require_grad = True

        # Iterate through the sentence
        for feat in feats:
            # broadcast the emission score: it is the same regardless of
            # the previous tag
            emit_score = feat.view(-1, 1)

            # the ith entry of trans_score is the score of transitioning to
            # next_tag from i
            tag_var = forward_var + self.transitions + emit_score

            # The ith entry of next_tag_var is the value for the
            # edge (i -> next_tag) before we do log-sum-exp
            max_tag_var, _ = torch.max(tag_var, dim=1)

            # The forward variable for this tag is log-sum-exp of all the
            # scores.
            tag_var = tag_var - max_tag_var.view(-1, 1)

            # Compute log sum exp in a numerically stable way for the forward algorithm
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)  # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag2idx[self.STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha

    def score_sentences(self, feats, tags):
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tag_size
        r = torch.LongTensor(range(feats.size()[0])).to(feats.device)
        pad_start_tags = torch.cat([torch.LongTensor([self.tag2idx[self.START_TAG]]), tags])
        pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag2idx[self.STOP_TAG]])])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

        return score

    def viterbi_decode(self, feats):
        """
        In this function, we implement the viterbi algorithm explained above.
        A Dynamic programming based approach to find the best tag sequence
        """
        backpointers = []
        # analogous to forward

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars.re
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()  # holds the backpointers for this step
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]  # holds the viterbi variables for this step
            viterbivars_t = torch.FloatTensor(viterbivars_t, require_grad=True, device=feats.device)

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2idx[self.STOP_TAG]]
        terminal_var.data[self.tag2idx[self.STOP_TAG]] = -10000.
        terminal_var.data[self.tag2idx[self.START_TAG]] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag2idx[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def cal_nll(self, sentence, tags, chars2, chars2_length, d):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tag_size
        feats = self._get_lstm_features(sentence, chars2, chars2_length, d)

        if self.use_crf:
            forward_score = self.forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            return forward_score - gold_score
        else:
            scores = nn.functional.cross_entropy(feats, tags)
            return scores
