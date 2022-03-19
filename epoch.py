from tqdm import tqdm
import math
import time
import torch


def train(args, model, data, optimizer):
    model.train()
    n_sample = 0
    loss_total = 0.
    total_f1 = 0
    start_time = time.time()

    for batch in tqdm(data, desc='  - training', leave=False):
        len_batch = batch[-1].shape[0]
        n_sample += len_batch
        words_batch, chars_batch, tags_batch, lens_batch = map(lambda x: x.to(args.device), batch)
        model.zero_grad()

        loss_batch, pred_batch = model.get_loss(words_batch, chars_batch, tags_batch, lens_batch)
        loss_batch.backward()
        optimizer.step()

        # calculate loss and f1
        mask_batch = (tags_batch != 19)
        loss_total += loss_batch.item() * mask_batch.sum()

        f1 = torch.eq(gt_batch, output_batch.max(-1).indices).sum().item() / len_batch
        total_f1 += f1

    time_elapse = (time.time() - start_time)
    loss_mean = loss_total / n_sample
    f1 = total_f1 / n_sample
    print('  | Train | loss {:5.4f} | F1 {:5.4f} | {:5.2f} s |'
          .format(loss_mean, f1, time_elapse))

    return loss_mean, f1


def evaluate(args, model, data, es_patience=0, mode='valid'):
    model.eval()
    n_sample = 0
    loss_total = 0.
    total_f1 = 0

    with torch.no_grad():
        for batch in tqdm(data, desc='  - evaluating', leave=False):
            len_batch = batch[0].shape[0]
            n_sample += len_batch
            words_batch, chars_batch, tags_batch, lens_batch = map(lambda x: x.to(args.device), batch)
            model.zero_grad()

            loss_batch, pred_batch = model.get_loss(words_batch, chars_batch, tags_batch, lens_batch)

            total_loss += loss_batch.item() * len_batch
            precision = 1
            recall = 1
            f1 = 2 * precision * recall / (precision + recall)
            total_f1 += f1

        loss_mean = loss_total / n_sample
        f1 = total_f1 / n_sample
        if mode == 'valid':
            print('  | Valid | loss {:5.4f} | F1 {:5.4f} | es_patience {:.0f} |'
                  .format(loss_mean, f1, es_patience))
        else:
            print('  | Test | loss {:5.4f} | F1 {:5.4f} |'
                  .format(loss_mean, f1))

    return loss_mean, f1
