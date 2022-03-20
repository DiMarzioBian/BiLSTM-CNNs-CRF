import argparse
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.onnx

from dataloader import get_dataloader
from model import BiLSTM_CRF
from epoch import train, evaluate


def main():
    parser = argparse.ArgumentParser(description='BiLSTM-CNNs-CRF project')
    parser.add_argument('--num_worker', type=int, default=5,
                        help='number of dataloader worker')
    parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--es_patience_max', type=int, default=10,
                        help='max early stopped patience')

    # dimension setting
    parser.add_argument('--dim_emb_char', type=int, default=25,
                        help='character embedding dimension')
    parser.add_argument('--dim_emb_word', type=int, default=100,
                        help='word embedding dimension')
    parser.add_argument('--dim_out_char', type=int, default=25,
                        help='character encoder output dimension')
    parser.add_argument('--dim_out_word', type=int, default=25,
                        help='word encoder output dimension')
    parser.add_argument('--window_kernel', type=int, default=5,
                        help='window width of CNN kernel')

    # general settings
    parser.add_argument('--enable_pretrained', type=bool, default=True,
                        help='use pretrained glove dimension')
    parser.add_argument('--freeze_glove', type=bool, default=False,
                        help='free pretrained glove embedding')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate applied to layers (0 = no dropout)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='initial learning rate')
    parser.add_argument('--lr_step', type=int, default=10,
                        help='number of epoch for each lr downgrade')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='strength of lr downgrade')
    parser.add_argument('--eps_f1', type=float, default=1e-4,
                        help='minimum f1 score difference threshold')

    # NLP related settings
    parser.add_argument('--mode_char', type=str, default='lstm',
                        help='character encoder: lstm or cnn')
    parser.add_argument('--mode_word', type=str, default='cnn2',
                        help='word encoder: lstm or cnn1, cnn2, cnn3, cnn_d')
    parser.add_argument('--enable_crf', type=bool, default=False,
                        help='employ CRF')
    parser.add_argument('--filter_word', type=bool, default=False,
                        help='filter meaningless words')

    # Default settings
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for computing')
    parser.add_argument('--path_data', type=str, default='./data/raw/',
                        help='path of the data corpus')
    parser.add_argument('--path_processed', type=str, default='./data/data_bundle.pkl',
                        help='path of the processed data information')
    parser.add_argument('--path_filtered', type=str, default='./data/data_filtered_bundle.pkl',
                        help='path to save the filtered processed data')
    parser.add_argument('--path_pretrained', type=str, default='./data/trained-model-cpu',
                        help='path of the data corpus')
    parser.add_argument('--path_model', type=str, default='./result/models/model.pt',
                        help='path of the trained model')

    args = parser.parse_args()
    args.device = torch.device(args.device)
    args.START_TAG = '<START>'
    args.STOP_TAG = '<STOP>'
    args.is_lowercase = True  # use lowercase of words
    args.digi_zero = True  # control replacement of  all digits by 0
    args.tag_scheme = 'BIOES'  # BIO or BIOES

    print('\n[info] Project starts...')
    print('\n[info] Load dataset and other resources...')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # get data and prepare model, optimizer and scheduler
    if not args.filter_word:
        with open(args.path_processed, 'rb') as f:
            mappings = pickle.load(f)
    else:
        with open(args.path_filtered, 'rb') as f:
            mappings = pickle.load(f)

    word2idx = mappings['word2idx']
    char2idx = mappings['char2idx']
    tag2idx = mappings['tag2idx']
    args.max_len_word = max([max([len(s) for s in word2idx.keys()]), 37])  # 37 is the longest length in testing set
    args.idx_pad_char = max(char2idx.values()) + 1
    args.idx_pad_word = max(word2idx.values()) + 1
    args.idx_pad_tag = max(tag2idx.values()) + 1

    if args.enable_pretrained:
        glove_word = mappings['embeds_word']
        glove_word = np.append(glove_word, [[0]*args.dim_emb_word], axis=0)
    else:
        glove_word = None

    train_loader, valid_loader, test_loader = get_dataloader(args, word2idx, tag2idx, char2idx)

    model = BiLSTM_CRF(args, word2idx, char2idx, tag2idx, glove_word).to(args.device)
    args.criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Start modeling
    print('\n[info] | lr: {lr} | dropout: {dropout} |char: {mode_char} | word: {mode_word} | CRF: {crf} '
          '| Param: {n_param} | '
          .format(lr=args.lr, dropout=args.dropout, mode_char=args.mode_char, mode_word=args.mode_word,
                  crf=args.enable_crf, n_param=n_param))
    best_val_loss = 1e5
    best_f1 = 0
    best_epoch = 0
    es_patience = 0

    for epoch in range(1, args.epochs+1):
        print('\n[Epoch {epoch}]'.format(epoch=epoch))

        t_start = time.time()
        loss_train, f1_train = train(args, model, train_loader, optimizer)
        scheduler.step()
        print('  | Train | loss {:5.4f} | F1 {:5.4f} | {:5.2f} s |'.format(loss_train, f1_train, time.time() - t_start))
        val_loss, val_f1 = evaluate(args, model, valid_loader)

        # Save the model if the validation loss is the best we've seen so far.
        if val_f1 > best_f1:
            if val_f1 - best_f1 > args.eps_f1:
                es_patience = 0  # reset if beyond threshold
            with open(args.path_model, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            best_epoch = epoch
            best_f1 = val_f1
        else:
            # Early stopping condition
            es_patience += 1
            if es_patience >= args.es_patience_max:
                print('\n[Warning] Early stopping model')
                print('  | Best | Epoch {:d} | Loss {:5.4f} | F1 {:5.4f} |'
                      .format(best_epoch, best_val_loss, best_f1))
                break
        # logging
        print('  | Valid | loss {:5.4f} | F1 {:5.4f} | es_patience {:.0f} |'.format(val_loss, val_f1, es_patience))

    # Load the best saved model and test
    print('\n[Testing]')
    with open(args.path_model, 'rb') as f:
        model = torch.load(f)
    loss_test, f1_test = evaluate(args, model, test_loader)

    print('  | Test | loss {:5.4f} | F1 {:5.4f} |'.format(loss_test, f1_test))
    print('\n[info] | lr: {lr} | dropout: {dropout} |char: {mode_char} | word: {mode_word} | CRF: {crf} '
          '| Param: {n_param} | '
          .format(lr=args.lr, dropout=args.dropout, mode_char=args.mode_char, mode_word=args.mode_word,
                  crf=args.enable_crf, n_param=n_param))


if __name__ == '__main__':
    main()
