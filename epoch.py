from tqdm import tqdm
import math
import time
import torch


def train(args, model, data, optimizer):
    model.train()
    total_loss = 0.
    total_sample = 0
    total_f1 = 0
    start_time = time.time()

    for batch in tqdm(data, desc='  - training', leave=False):
        len_batch = batch[-1].shape[0]
        total_sample += len_batch
        words_batch, chars_batch, tags_batch, lens_batch = map(lambda x: x.to(args.device), batch)
        model.zero_grad()

        output_batch = model(words_batch, chars_batch, tags_batch, lens_batch)
        loss = args.criterion(words_batch, chars_batch, tags_batch, lens_batch)
        loss.backward()
        if args.optimizer != 'Initial':
            optimizer.step()
        else:
            for p in filter(lambda x: x.requires_grad, model.parameters()):
                p.data.add_(p.grad, alpha=-args.lr)

        total_loss += loss.item() * len_batch
        hit = torch.eq(gt_batch, output_batch.max(-1).indices).sum().item() / len_batch
        total_f1 += total_f1

    time_elapse = (time.time() - start_time)
    mean_loss = total_loss / total_sample
    f1 = total_f1 / total_sample
    print('  | Train | loss {:5.4f} | F1 {:5.4f} | {:5.2f} s |'
          .format(mean_loss, f1, time_elapse))

    return mean_loss, f1


def evaluate(args, model, data, es_patience=0, mode='valid'):
    model.eval()
    total_loss = 0.
    total_sample = 0
    total_f1 = 0

    with torch.no_grad():
        for batch in tqdm(data, desc='  - evaluating', leave=False):
            len_batch = batch[0].shape[0]
            total_sample += len_batch
            seq_batch, gt_batch = map(lambda x: x.to(args.device), batch)
            model.zero_grad()

            output_batch = model(seq_batch)
            loss = args.criterion(output_batch, gt_batch)

            total_loss += loss.item() * len_batch
            hit = torch.eq(gt_batch, output_batch.max(-1).indices).sum().item() / len_batch
            total_f1 += hit

        mean_loss = total_loss / total_sample
        f1 = total_f1 / total_sample
        if mode == 'valid':
            print('  | Valid | loss {:5.4f} | F1 {:5.4f} | es_patience {:.0f} |'
                  .format(mean_loss, f1, es_patience))
        else:
            print('  | Test | loss {:5.4f} | F1 {:5.4f} |'
                  .format(mean_loss, f1))

    return mean_loss, f1
