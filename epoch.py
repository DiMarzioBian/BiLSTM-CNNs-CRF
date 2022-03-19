from tqdm import tqdm
import math
import time
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def train(args, model, data, optimizer):
    model.train()
    n_sample = 0
    loss_total = 0.
    start_time = time.time()
    pred_all, gt_all = [], []  # to calculate f1 score

    for batch in tqdm(data, desc='  - training', leave=False):
        len_batch = batch[-1].shape[0]
        n_sample += len_batch
        words_batch, chars_batch, tags_batch, lens_batch = map(lambda x: x.to(args.device), batch)
        model.zero_grad()

        loss_batch, pred_batch = model.get_loss(words_batch, chars_batch, tags_batch, lens_batch)
        loss_batch.backward()
        optimizer.step()

        # calculate loss and f1
        for pred, tags, _len in zip(pred_batch, tags_batch, lens_batch):
            gt_all += tags[:_len].tolist()
            pred_all += pred[:_len]

    f1 = f1_score(gt_all, pred_all, average='macro')
    time_elapse = (time.time() - start_time)
    loss_mean = loss_total / n_sample
    print('  | Train | loss {:5.4f} | F1 {:5.4f} | {:5.2f} s |'
          .format(loss_mean, f1, time_elapse))

    return loss_mean, f1


def evaluate(args, model, data, es_patience=0, mode='valid'):
    model.eval()
    n_sample = 0
    loss_total = 0.
    pred_all, gt_all = [], []

    with torch.no_grad():
        for batch in tqdm(data, desc='  - evaluating', leave=False):
            len_batch = batch[0].shape[0]
            n_sample += len_batch
            words_batch, chars_batch, tags_batch, lens_batch = map(lambda x: x.to(args.device), batch)
            model.zero_grad()

            loss_batch, pred_batch = model.get_loss(words_batch, chars_batch, tags_batch, lens_batch)

            # calculate loss and f1
            for pred, tags, _len in zip(pred_batch, tags_batch, lens_batch):
                gt_all += tags[:_len].tolist()
                pred_all += pred[:_len]

        f1 = f1_score(gt_all, pred_all, average='macro')
        loss_mean = loss_total / n_sample
        if mode == 'valid':
            print('  | Valid | loss {:5.4f} | F1 {:5.4f} | es_patience {:.0f} |'
                  .format(loss_mean, f1, es_patience))
        else:
            print('  | Test | loss {:5.4f} | F1 {:5.4f} |'
                  .format(loss_mean, f1))

    return loss_mean, f1
