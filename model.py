import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def log_sum_exp(vec):
    '''
    This function calculates the score explained above for the forward algorithm
    vec 2D: 1 * size_tag
    '''
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax(vec):
    '''
    This function returns the max index in a vector
    '''
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def to_scalar(var):
    '''
    Function to convert pytorch tensor to a scalar
    '''
    return var.view(-1).data.tolist()[0]


class BiLSTM_CRF(nn.Module):
    def __init__(self, args, tag2idx, emb_word):
        """
        Input parameters from args:

                tag2idx = Dictionary that maps NER tags to indices
                embedding_dim = Dimension of word embeddings (int)
                hidden_dim =
                char_to_ix = Dictionary that maps characters to indices
                pre_word_embeds = Numpy array which provides mapping from word embeddings to word indices
                char_out_dimension = Output dimension from the CNN encoder for character
                char_embedding_dim = Dimension of the character embeddings
        """
        super(BiLSTM_CRF, self).__init__()
        self.START_TAG = args.START_TAG
        self.STOP_TAG = args.STOP_TAG
        self.emb_dim = args.word_dim  # dimension of word embeddings (int)
        self.hidden_dim = args.hidden_dim  # The hidden dimension of the LSTM layer (int)
        self.size_vocab = args.size_vocab  # Size of vocabulary (int)
        self.tag2idx = tag2idx
        self.size_tag = len(tag2idx)
        self.use_crf = args.use_crf  # parameter which decides if you want to use the CRF layer for output decoding
        self.char_emb_dim = args.char_emb_dim  # dimension of the character embeddings
        self.out_channels = args.char_out_dim  # dimension of the character embeddings
        self.mode_char = args.mode_char

        if char_embedding_dim is not None:
            self.char_embedding_dim = char_embedding_dim

            # Initializing the character embedding layer
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            init_embedding(self.char_embeds.weight)

            # Performing LSTM encoding on the character embeddings
            if self.char_mode == 'LSTM':
                self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_dim, num_layers=1, bidirectional=True)
                init_lstm(self.char_lstm)

            # Performing CNN encoding on the character embeddings
            if self.char_mode == 'CNN':
                self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels,
                                           kernel_size=(3, char_embedding_dim), padding=(2, 0))

        # Creating Embedding layer with dimension of ( number of words * dimension of each word)
        self.word_embeds = nn.Embedding(self.size_vocab, embedding_dim)
        if pre_word_embeds is not None:
            # Initializes the word embeddings with pretrained word embeddings
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False

        # Initializing the dropout layer, with dropout specificed in parameters
        self.dropout = nn.Dropout(parameters['dropout'])

        # Lstm Layer:
        # input dimension: word embedding dimension + character level representation
        # bidirectional=True, specifies that we are using the bidirectional LSTM
        if self.char_mode == 'LSTM':
            self.lstm = nn.LSTM(self.embedding_dim + char_lstm_dim * 2, self.hidden_dim, bidirectional=True)
        if self.char_mode == 'CNN':
            self.lstm = nn.LSTM(self.embedding_dim + self.out_channels, self.hidden_dim, bidirectional=True)

        # Initializing the lstm layer using predefined function for initialization
        init_lstm(self.lstm)

        # Linear layer which maps the output of the bidirectional LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.size_tag)

        # Initializing the linear layer using predefined function for initialization
        init_linear(self.hidden2tag)

        if self.use_crf:
            # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
            # Matrix has a dimension of (total number of tags * total number of tags)
            self.transitions = nn.Parameter(
                torch.zeros(self.size_tag, self.size_tag))

            # These two statements enforce the constraint that we never transfer
            # to the start tag and we never transfer from the stop tag
            self.transitions.data[self.tag2idx[self.START_TAG], :] = -10000
            self.transitions.data[:, self.tag2idx[self.STOP_TAG]] = -10000

    # assigning the functions, which we have defined earlier
    _score_sentence = score_sentences
    _forward_alg = forward_alg
    viterbi_decode = viterbi_algo
    neg_log_likelihood = get_neg_log_likelihood

    def get_lstm_features(self, sentence, chars2, chars2_length, d):
        if self.mode_char == 'lstm':

            chars_embeds = self.char_embeds(chars2).transpose(0, 1)

            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)

            lstm_out, _ = self.char_lstm(packed)

            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)

            outputs = outputs.transpose(0, 1)

            chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))

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
            # followed by a Maxpooling Layer
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
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim * 2)
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

        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)
        if self.use_gpu:
            forward_var = forward_var.cuda()

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
        terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha

