import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
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
    """ This function returns the max index in a vector """
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
        self.device = args.device
        self.enable_crf = args.enable_crf
        self.idx_pad_tag = args.idx_pad_tag

        self.START_TAG = args.START_TAG
        self.STOP_TAG = args.STOP_TAG
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.tag2idx = tag2idx
        self.n_word = len(word2idx)
        self.n_char = len(char2idx)
        self.n_tag = len(tag2idx)

        self.max_len_word = args.max_len_word

        self.idx_pad_char = args.idx_pad_char
        self.idx_pad_word = args.idx_pad_word

        self.dim_emb_char = args.dim_emb_char
        self.dim_emb_word = args.dim_emb_word
        self.dim_out_char = args.dim_out_char  # dimension of the character embeddings
        self.dim_out_word = args.dim_out_word  # The hidden dimension of the LSTM layer (int)
        self.window_kernel = args.window_kernel

        self.mode_char = args.mode_char
        self.mode_word = args.mode_word

        # embedding layer
        self.embedding_char = nn.Embedding(self.n_char+1, self.dim_emb_char, padding_idx=self.idx_pad_char)
        init_embedding(self.embedding_char)

        if args.enable_pretrained:
            self.embedding_word = nn.Embedding.from_pretrained(torch.FloatTensor(glove_word), freeze=args.freeze_glove,
                                                               padding_idx=self.idx_pad_word)
        else:
            self.embedding_word = nn.Embedding(self.n_word+1, self.dim_emb_word)
            init_embedding(self.embedding_word)

        # character encoder
        if self.mode_char == 'lstm':
            self.lstm_char = nn.LSTM(self.dim_emb_char, self.dim_out_char, num_layers=1, batch_first=True,
                                     bidirectional=True)
            init_lstm(self.lstm_char)
        elif self.mode_char == 'cnn':
            self.conv_char = nn.Conv2d(in_channels=1, out_channels=self.dim_out_char * 2,
                                       kernel_size=(3, self.dim_emb_char), padding=(2, 0))
            init_linear(self.conv_char)
            self.mp_char = nn.MaxPool2d((self.max_len_word + 2, 1))  # padding x 2 - kernel_size + 1
        else:
            raise Exception('Character encoder mode unknown...')
        self.dropout1 = nn.Dropout(args.dropout)

        # word encoder
        self.dim_in_word = self.dim_emb_word + self.dim_out_char * 2
        if self.mode_word == 'lstm':
            self.lstm_word = nn.LSTM(self.dim_in_word, self.dim_out_word, batch_first=True,
                                     bidirectional=True)
            init_lstm(self.lstm_word)

        elif self.mode_word == 'cnn1':
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.dim_out_word * 2,
                                   kernel_size=(self.window_kernel, self.dim_in_word),
                                   padding=(self.window_kernel // 2, 0))
            init_linear(self.conv1)

        elif self.mode_word == 'cnn2':
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.dim_out_word * 2,
                                   kernel_size=(self.window_kernel, self.dim_in_word),
                                   padding=(self.window_kernel // 2, 0))
            self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.dim_out_word * 2,
                                   kernel_size=(self.window_kernel, self.dim_out_word * 2),
                                   padding=(self.window_kernel // 2, 0))
            init_linear(self.conv1)
            init_linear(self.conv2)

        elif self.mode_word == 'cnn3':
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.dim_out_word * 2,
                                   kernel_size=(self.window_kernel, self.dim_in_word),
                                   padding=(self.window_kernel // 2, 0))
            self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.dim_out_word * 2,
                                   kernel_size=(self.window_kernel, self.dim_out_word * 2),
                                   padding=(self.window_kernel // 2, 0))
            self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.dim_out_word * 2,
                                   kernel_size=(self.window_kernel, self.dim_out_word * 2),
                                   padding=(self.window_kernel // 2, 0))
            init_linear(self.conv1)
            init_linear(self.conv2)
            init_linear(self.conv3)

        elif self.mode_word == 'cnn_d':
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.dim_out_word * 2,
                                   kernel_size=(self.window_kernel, self.dim_in_word),
                                   padding=(self.window_kernel // 2, 0), dilation=(1, 1))
            self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.dim_out_word * 2,
                                   kernel_size=(self.window_kernel, self.dim_out_word * 2),
                                   padding=((self.window_kernel // 2) * 2, 0), dilation=(2, 1))
            self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.dim_out_word * 2,
                                   kernel_size=(self.window_kernel, self.dim_out_word * 2),
                                   padding=((self.window_kernel // 2) * 3, 0), dilation=(3, 1))
            init_linear(self.conv1)
            init_linear(self.conv2)
            init_linear(self.conv3)

        else:
            raise Exception('Word encoder mode '+self.mode_char+' unknown...')

        self.dropout2 = nn.Dropout(args.dropout)

        # predictor
        self.hidden2tag = nn.Linear(self.dim_out_word * 2, self.n_tag)
        init_linear(self.hidden2tag)
        if args.enable_crf:
            self.transitions = nn.Parameter(torch.zeros(self.n_tag, self.n_tag))
            self.transitions.data[self.tag2idx[self.START_TAG], :] = -10000
            self.transitions.data[:, self.tag2idx[self.STOP_TAG]] = -10000

    @staticmethod
    def reformat_conv2d(t):
        return t.squeeze(-1).transpose(-1, -2).unsqueeze(1)

    def forward(self, words_batch, chars_batch, lens_word):
        len_batch, len_sent = words_batch.shape

        # character-level modelling
        emb_chars = self.embedding_char(chars_batch)
        if self.mode_char == 'lstm':
            # covered padded characters that have 0 length to 1
            lens_char = (chars_batch != self.idx_pad_char).sum(dim=2)
            lens_char_covered = torch.where(lens_char == 0, 1, lens_char)
            packed_char = pack_padded_sequence(emb_chars.view(-1, self.max_len_word, self.dim_emb_char),
                                               lens_char_covered.view(-1).cpu(), batch_first=True, enforce_sorted=False)
            out_lstm_char, _ = self.lstm_char(packed_char)

            # return to (len_batch x len_sent x len_char x dim_emb_char)
            output_char, _ = pad_packed_sequence(out_lstm_char, batch_first=True, total_length=self.max_len_word)
            output_char = output_char * lens_char.view(-1, 1, 1).bool()
            output_char = output_char.reshape(len_batch, len_sent, self.max_len_word, self.dim_emb_char*2)

            output_char = torch.cat(
                (torch.stack(
                    [sample[torch.arange(len_sent).long(), lens-1, :self.dim_out_char]
                     for sample, lens in zip(output_char, lens_char)]),
                 torch.stack(
                     [sample[torch.arange(len_sent).long(), lens*0, self.dim_out_char:]
                      for sample, lens in zip(output_char, lens_char)]))
                , dim=-1)

        elif self.mode_char == 'cnn':
            enc_char = self.conv_char(emb_chars.unsqueeze(2).view(-1, 1, self.max_len_word, self.dim_emb_char))
            output_char = self.mp_char(enc_char).view(len_batch, len_sent, self.dim_out_char * 2)
        else:
            raise Exception('Unknown character encoder: '+self.mode_char+'...')

        # load word embeddings
        emb_words = self.embedding_word(words_batch)
        emb_words_chars = torch.cat((emb_words, output_char), dim=-1)
        emb_words_chars = self.dropout1(emb_words_chars)

        # word lstm
        if self.mode_word == 'lstm':
            packed_word = pack_padded_sequence(emb_words_chars, lens_word.cpu(), batch_first=True)
            out_lstm_word, _ = self.lstm_word(packed_word)
            enc_word, _ = pad_packed_sequence(out_lstm_word, batch_first=True)

        elif self.mode_word == 'cnn1':
            out_cnn_word = self.conv1(emb_words_chars.unsqueeze(1))
            enc_word = self.reformat_conv2d(out_cnn_word).squeeze(1)

        elif self.mode_word == 'cnn2':
            out_cnn_word = self.conv1(emb_words_chars.unsqueeze(1))
            out_cnn_word = self.reformat_conv2d(out_cnn_word)
            out_cnn_word = self.conv2(out_cnn_word)
            enc_word = self.reformat_conv2d(out_cnn_word).squeeze(1)

        elif self.mode_word in ['cnn3', 'cnn_d']:
            out_cnn_word = self.conv1(emb_words_chars.unsqueeze(1))
            out_cnn_word = self.reformat_conv2d(out_cnn_word)
            out_cnn_word = self.conv2(out_cnn_word)
            out_cnn_word = self.reformat_conv2d(out_cnn_word)
            out_cnn_word = self.conv3(out_cnn_word)
            enc_word = self.reformat_conv2d(out_cnn_word).squeeze(1)

        else:
            raise Exception('Unknown word encoder: '+self.mode_word+'...')

        outputs = self.hidden2tag(enc_word)
        return outputs

    def get_loss(self, words_batch, chars_batch, tags_batch, lens_batch):
        """ calculate both predicted scores and losses"""
        feats_batch = self.forward(words_batch, chars_batch, lens_batch)

        if self.enable_crf:
            loss_batch, pred_batch = [], []
            # iterate each sentence
            for i, (feats, tags, len_sent) in enumerate(zip(feats_batch, tags_batch, lens_batch)):
                feats, tags = feats[:len_sent, :], tags[:len_sent]
                forward_score = self.forward_alg(feats)
                gold_score = self.score_sentence(feats, tags)
                score, pred = self.viterbi_decode(feats)
                loss_batch.append(forward_score - gold_score)
                pred_batch.append(pred)

            return torch.stack(loss_batch).sum() / lens_batch.sum(), pred_batch
        else:
            loss = F.cross_entropy(feats_batch.view(-1, self.n_tag), tags_batch.view(-1), ignore_index=self.idx_pad_tag)

            _, pred_batch = torch.max(feats_batch.view(-1, self.n_tag)[:, :-1], 1)
            pred_batch = pred_batch.reshape(feats_batch.shape[0], feats_batch.shape[1])

            return loss, pred_batch.cpu().tolist()

    def forward_alg(self, feats):
        """
        This function performs the forward algorithm explained above
        """
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.zeros((1, self.n_tag)).fill_(-10000.)

        # START_TAG has all score.
        init_alphas[0][self.tag2idx[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas.clone().to(feats.device)
        forward_var.requires_grad = True

        # Iterate through the sentence
        for feat in feats:
            # broadcast the emission score: it is the same regardless of the previous tag
            emit_score = feat.view(-1, 1)

            # the ith entry of trans_score is the score of transitioning to next_tag from i
            tag_var = forward_var + self.transitions + emit_score

            # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
            max_tag_var, _ = torch.max(tag_var, dim=1)

            # The forward variable for this tag is log-sum-exp of all the scores.
            tag_var = tag_var - max_tag_var.view(-1, 1)

            # Compute log sum exp in a numerically stable way for the forward algorithm
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)  # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag2idx[self.STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha

    def score_sentence(self, feats, tags):
        # remove padding word

        r = torch.tensor(range(feats.size()[0]), dtype=torch.long, device=feats.device)
        pad_start_tags = torch.cat([torch.tensor([self.tag2idx[self.START_TAG]], dtype=torch.long, device=tags.device),
                                    tags])
        pad_stop_tags = torch.cat([tags,
                                   torch.tensor([self.tag2idx[self.STOP_TAG]], dtype=torch.long, device=tags.device)])
        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

        return score

    def viterbi_decode(self, feats):
        """ viterbi algorithm """
        backpointers = []
        # analogous to forward

        # Initialize the viterbi variables in log space
        init_vars = torch.zeros((1, self.n_tag), device=feats.device).fill_(-10000.)
        init_vars[0][self.tag2idx[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vars.clone()
        forward_var.requires_grad = True
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.n_tag, self.n_tag) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()  # holds the backpointers for this step
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]  # holds the viterbi variables for this step
            viterbivars_t = torch.tensor(viterbivars_t, device=feats.device, requires_grad=True)

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


