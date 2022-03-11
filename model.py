import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_CRF(nn.Module):

    def __init__(self, args, tag2idx, ):
        """
        Input parameters from args:

                vocab_size= Size of vocabulary (int)
                tag2idx = Dictionary that maps NER tags to indices
                embedding_dim = Dimension of word embeddings (int)
                hidden_dim = The hidden dimension of the LSTM layer (int)
                char_to_ix = Dictionary that maps characters to indices
                pre_word_embeds = Numpy array which provides mapping from word embeddings to word indices
                char_out_dimension = Output dimension from the CNN encoder for character
                char_embedding_dim = Dimension of the character embeddings
                use_crf = parameter which decides if you want to use the CRF layer for output decoding
        """
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = args.word_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2idx = tag2idx
        self.use_crf = use_crf
        self.tagset_size = len(tag2idx)
        self.out_channels = char_out_dimension
        self.char_mode = args.char_mode
        self.START_TAG = args.START_TAG
        self.STOP_TAG = args.STOP_TAG

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
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
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
        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)

        # Initializing the linear layer using predefined function for initialization
        init_linear(self.hidden2tag)

        if self.use_crf:
            # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
            # Matrix has a dimension of (total number of tags * total number of tags)
            self.transitions = nn.Parameter(
                torch.zeros(self.tagset_size, self.tagset_size))

            # These two statements enforce the constraint that we never transfer
            # to the start tag and we never transfer from the stop tag
            self.transitions.data[self.tag2idx[self.START_TAG], :] = -10000
            self.transitions.data[:, self.tag2idx[self.STOP_TAG]] = -10000

    # assigning the functions, which we have defined earlier
    _score_sentence = score_sentences
    _get_lstm_features = get_lstm_features
    _forward_alg = forward_alg
    viterbi_decode = viterbi_algo
    neg_log_likelihood = get_neg_log_likelihood
    forward = forward_calc