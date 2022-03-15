import argparse
import os
import pickle
import math
import torch
import torch.nn as nn
import torch.onnx

from dataloader import get_dataloader
from model import BiLSTM_CRF
from epoch import train, evaluate


def main():
    parser = argparse.ArgumentParser(description='BiLSTM-CNNs-CRF project')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for modeling')
    parser.add_argument('--path_data', type=str, default='./data/raw/',
                        help='path of the data corpus')
    parser.add_argument('--path_processed', type=str, default='./data/data_bundle.pkl',
                        help='path of the processed data information')
    parser.add_argument('--path_pretrained', type=str, default='./data/trained-model-cpu',
                        help='path of the data corpus')
    parser.add_argument('--num_worker', type=int, default=0,
                        help='number of dataloader worker')
    parser.add_argument('--initial_preprocess', type=bool, default=False,
                        help='use initial data preprocess strategy')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--es_patience_max', type=int, default=10,
                        help='Max early stopped patience')

    # dimension setting
    parser.add_argument('--dim_char_emb', type=int, default=25,
                        help='char embedding dimension')
    parser.add_argument('--dim_char_out', type=int, default=25,
                        help='output dimension from the CNN encoder for character')
    parser.add_argument('--dim_word_emb', type=int, default=25,
                        help='word embedding dimension')
    parser.add_argument('--dim_lstm_hidden', type=int, default=50,
                        help='lstm hidden state dimension')
    parser.add_argument('--dim_lstm_char', type=int, default=25,
                        help='character encoder lstm dimension')

    # general settings
    parser.add_argument('--enable_pretrained', type=bool, default=True,
                        help='use pretrained glove dimension')
    parser.add_argument('--freeze_glove', type=bool, default=True,
                        help='free pretrained glove embedding')
    parser.add_argument('--clip_gradient', type=float, default=5.0,
                        help='gradient clipping threshold')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate applied to layers (0 = no dropout)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Adam, AdamW, RMSprop, Adagrad, SGD & Initial')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--lr_step', type=int, default=10,
                        help='number of epoch for each lr downgrade')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='strength of lr downgrade')
    parser.add_argument('--eps_f1', type=float, default=1e-5,
                        help='minimum f1 score difference threshold')

    # NLP related settings
    parser.add_argument('--is_lowercase', type=bool, default=True,
                        help='use lowercase of words')
    parser.add_argument('--digi_zero', type=bool, default=True,
                        help='control replacement of  all digits by 0')
    parser.add_argument('--tag_scheme', type=str, default='BIOES',
                        help='BIO or BIOES')
    parser.add_argument('--mode_char', type=str, default='lstm',
                        help='character encoder: lstm or cnn')
    parser.add_argument('--mode_word', type=str, default='lstm',
                        help='word encoder: lstm or cnn')
    parser.add_argument('--enable_crf', type=bool, default=True,
                        help='employ CRF')
    parser.add_argument('--n_cnn', type=int, default=3,
                        help='number of layer of CNN')
    parser.add_argument('--dilated', type=bool, default=False,
                        help='employ dilated CNN')

    args = parser.parse_args()
    args.device = torch.device(args.device)
    args.START_TAG = '<START>'
    args.STOP_TAG = '<STOP>'
    print('\n[info] Project starts...')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # get data and prepare model, optimizer and scheduler
    print('\n[info] Load dataset and other resources...')
    with open(args.path_processed, 'rb') as f:
        mappings = pickle.load(f)
    word2idx = mappings['word2idx']
    char2idx = mappings['char2idx']
    tag2idx = mappings['tag2idx']
    args.max_len_word = max([len(s) for s in word2idx.keys()])
    args.idx_pad_char = max(char2idx.values()) + 1

    if args.enable_pretrained:
        glove_word = mappings['embeds_word']
    else:
        glove_word = None

    train_loader, valid_loader, test_loader = get_dataloader(args, word2idx, tag2idx, char2idx)

    model = BiLSTM_CRF(args, word2idx, char2idx, tag2idx, glove_word).to(args.device)
    args.criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # Start modeling
    print('[info] | char: {mode_char} | word: {mode_word} | CRF: {crf} | N_CNN: {n_cnn} | Dilated: {dilated} |'
          .format(mode_char=args.mode_char, mode_word=args.mode_word, crf=args.enable_crf, n_cnn=args.n_cnn,
                  dilated=args.dilated))
    best_val_loss = 1e5
    best_f1 = 0
    best_epoch = 0
    es_patience = 0

    for epoch in range(1, args.epochs+1):
        print('\n[Epoch {epoch}]'.format(epoch=epoch))

        train(args, model, train_loader, optimizer)
        if args.optimizer != 'Initial':
            scheduler.step()
        val_loss, val_f1 = evaluate(args, model, valid_loader, mode='valid', es_patience=es_patience)

        # Save the model if the validation loss is the best we've seen so far.
        if val_f1 > best_f1:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            best_f1 = val_f1
            best_epoch = epoch
            if val_f1 - best_f1 > args.eps_val:
                es_patience = 0
        else:
            # Early stopping condition
            es_patience += 1
            if es_patience >= args.es_patience_max:
                print('\n[Warning] Early stopping model')
                print('  | Best | Epoch {:d} | Loss {:5.4f} | F1 {:5.4f} |'
                      .format(best_epoch, best_val_loss, best_f1))
                break

    # Load the best saved model and test
    print('\n[Testing]')
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    test_loss, test_acc = evaluate(args, model, test_loader, es_patience, mode='test')
    print('[info] | char: {mode_char} | word: {mode_word} | CRF: {crf} | N_CNN: {n_cnn} | Dilated: {dilated} |'
          .format(mode_char=args.mode_char, mode_word=args.mode_word, crf=args.enable_crf, n_cnn=args.n_cnn,
                  dilated=args.dilated))


if __name__ == '__main__':
    main()
