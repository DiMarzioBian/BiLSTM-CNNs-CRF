import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, args, glove_word, word2idx, tag2idx, char2idx):
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
        self.glove_word = glove_word
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.char2idx = char2idx
        self.n_word = len(word2idx)
        self.n_tag = len(tag2idx)
        self.n_char = len(char2idx)

        self.dim_word_glove = args.dim_word_glove
        self.dim_word_emb = args.dim_word_emb  # dimension of word embeddings (int)
        self.dim_char_emb = args.dim_char_emb

        self.dim_lstm_char = args.dim_lstm_hidden
        self.dim_lstm_hidden = args.dim_lstm_hidden  # The hidden dimension of the LSTM layer (int)
        self.out_channels = args.char_out_dim  # dimension of the character embeddings

        self.mode_char = args.mode_char
        self.mode_word = args.mode_word
        self.use_crf = args.use_crf  # parameter which decides if you want to use the CRF layer for output decoding
        self.mode_char = args.mode_char

        # model architecture
        self.embedding_word = nn.Embedding(self.n_word, self.dim_word_emb)

        if self.dim_char_emb is not None:
            self.embedding_char = nn.Embedding(len(char2idx), self.dim_char_emb)

            # Performing LSTM encoding on the character embeddings
            if self.mode_char == 'LSTM':
                self.char_lstm = nn.LSTM(self.dim_char_emb, self.dim_lstm_char, num_layers=1, bidirectional=True)

            # Performing CNN encoding on the character embeddings
            if self.mode_char == 'CNN':
                self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels,
                                           kernel_size=(3, self.char_emb_dim), padding=(2, 0))

        if self.dim_word_glove is not None:
            # Initializes the word embeddings with pretrained word embeddings
            self.dim_word_glove = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(self.dim_word_glove))
        else:
            self.dim_word_glove = False

        self.dropout = nn.Dropout(self.dropout)

        # Lstm Layer
        if self.char_mode == 'LSTM':
            self.lstm = nn.LSTM(self.emb_dim + self.dim_lstm_char * 2, self.dim_lstm_hidden, bidirectional=True)
        if self.char_mode == 'CNN':
            self.lstm = nn.LSTM(self.emb_dim + self.out_channels, self.dim_lstm_hidden, bidirectional=True)

        # Linear layer which maps the output of the bidirectional LSTM into tag space.
        self.hidden2tag = nn.Linear(self.dim_lstm_hidden * 2, self.size_tag)

        if self.use_crf:
            # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
            # Matrix has a dimension of (total number of tags * total number of tags)
            self.transitions = nn.Parameter(
                torch.zeros(self.size_tag, self.size_tag))

            # These two statements enforce the constraint that we never transfer
            # to the start tag and we never transfer from the stop tag
            self.transitions.data[self.tag2idx[self.START_TAG], :] = -10000
            self.transitions.data[:, self.tag2idx[self.STOP_TAG]] = -10000

    def get_lstm_features(self, sentence, chars2, chars2_length, d):
        if self.mode_char == 'lstm':

            chars_embeds = self.char_embeds(chars2).transpose(0, 1)

            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)

            lstm_out, _ = self.char_lstm(packed)

            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)

            outputs = outputs.transpose(0, 1)

            chars_embeds_temp = torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2))), requires_grad=True)

            if self.use_gpu:
                chars_embeds_temp = chars_embeds_temp.cuda()

            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat(
                    (outputs[i, index - 1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))

            chars_embeds = chars_embeds_temp.clone()

            for i in range(chars_embeds.size(0)):
                chars_embeds[d[i]] = chars_embeds_temp[i]

        elif self.mode_char == 'cnn':
            chars_embeds = self.char_embeds(chars2).unsqueeze(1)

            # Creating Character level representation using Convolutional Neural Netowrk
            # followed by a max pooling Layer
            chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                                    kernel_size=(chars_cnn_out3.size(2), 1)).view(
                chars_cnn_out3.size(0),
                self.out_channels)

        else:
            raise Exception('Unknown character model...')

        # load word embeddings
        embeds = torch.cat((self.word_embeds(sentence), chars_embeds), 1).unsqueeze(1)
        embeds = self.dropout(embeds)

        # word lstm
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.dim_lstm_hidden * 2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def forward(self, sentence, chars, chars2_length, d):
        # Get the emission scores from the BiLSTM
        feats = self.get_lstm_features(sentence, chars, chars2_length, d)
        # viterbi to get tag_seq

        # Find the best path, given the features.
        if self.use_crf:
            score, tag_seq = self.viterbi_decode(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq

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
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            return forward_score - gold_score
        else:
            scores = nn.functional.cross_entropy(feats, tags)
            return scores
