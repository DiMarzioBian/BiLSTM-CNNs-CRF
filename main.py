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
    parser.add_argument('--num_worker', type=int, default=25,
                        help='number of dataloader worker')
    parser.add_argument('--initial_preprocess', type=bool, default=False,
                        help='use initial data preprocess strategy')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--es_patience_max', type=int, default=10,
                        help='Max early stopped patience')

    # dimension setting
    parser.add_argument('--dim_char_emb', type=int, default=25,
                        help='char embedding dimension')
    parser.add_argument('--char_out_dim', type=int, default=25,
                        help='output dimension from the CNN encoder for character')
    parser.add_argument('--dim_word_glove', type=int, default=100,
                        help='glove dimension')
    parser.add_argument('--dim_word_emb', type=int, default=25,
                        help='word embedding dimension')
    parser.add_argument('--dim_lstm_hidden', type=int, default=50,
                        help='lstm hidden state dimension')
    parser.add_argument('--dim_lstm_char', type=int, default=25,
                        help='character encoder lstm dimension')

    # general settings
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
    parser.add_argument('--eps_loss', type=float, default=1e-5,
                        help='minimum loss difference threshold')

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
    glove_word = mappings['embeds_word']
    word2idx = mappings['word2idx']
    char2idx = mappings['char2idx']
    tag2idx = mappings['tag2idx']

    train_loader, valid_loader, test_loader = get_dataloader(args, word2idx, tag2idx, char2idx)

    model = BiLSTM_CRF(args, glove_word, word2idx, char2idx, tag2idx).to(args.device)
    args.criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # Start modeling
    print('[info] | char: {mode_char} | word: {mode_word} | CRF: {crf} | N_CNN: {n_cnn} | Dilated: {dilated} |'
          .format(mode_char=args.mode_char, mode_word=args.mode_word, crf=args.enable_crf, n_cnn=args.n_cnn,
                  dilated=args.dilated))
    best_val_loss = 1e5
    best_acc = 0
    best_epoch = 0
    es_patience = 0

    for epoch in range(1, args.epochs+1):
        print('\n[Epoch {epoch}]'.format(epoch=epoch))

        train(args, model, train_loader, optimizer)
        if args.optimizer != 'Initial':
            scheduler.step()
        val_loss, val_acc = evaluate(args, model, valid_loader, mode='Valid', es_patience=es_patience)

        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            best_acc = val_acc
            best_epoch = epoch
            if best_val_loss - val_loss > args.eps_loss:
                es_patience = 0
        else:
            # Early stopping condition
            es_patience += 1
            if es_patience >= args.es_patience_max:
                print('\n[Warning] Early stopping model')
                print('  | Best  |  Loss {:5.4f} | Accuracy {:5.4f} | F1 {:5.4f} |'
                      .format(best_val_loss, math.exp(best_val_loss), best_acc))
                break

    # Load the best saved model and test
    print('\n[Testing]')
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    test_loss, test_acc = evaluate(args, model, test_loader, es_patience, mode='Test')
    print('\n[info] | {optimizer} | N_gram {ngram:d} | Shared: {shared} | Direct: {direct} | Dropout {dropout:3.2f} '
          '| H_dim {h_dim:d} | Epoch {epoch:.0f} |\n'
          .format(optimizer=args.optimizer, ngram=args.n_gram, shared=str(args.share_embedding),
                  direct=str(args.skip_connect), dropout=args.dropout, h_dim=args.h_dim, epoch=best_epoch))


if __name__ == '__main__':
    main()
