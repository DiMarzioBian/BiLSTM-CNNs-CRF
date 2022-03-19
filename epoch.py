from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def train(args, model, data, optimizer):
    model.train()
    n_sample = 0
    loss_total = 0.
    pred_all, gt_all = [], []  # to calculate f1 score
    ss = []

    for batch in tqdm(data, desc='  - training', leave=False):
        words_batch, chars_batch, tags_batch, lens_batch = map(lambda x: x.to(args.device), batch)

        optimizer.zero_grad()
        loss_batch, pred_batch = model.get_loss(words_batch, chars_batch, tags_batch, lens_batch)
        loss_batch.backward()
        optimizer.step()
        ss.append(loss_batch.cpu().tolist())

        # calculate loss and f1
        n_sample += lens_batch.sum()
        loss_total += loss_batch * lens_batch.sum()
        for pred, tags, _len in zip(pred_batch, tags_batch, lens_batch):
            gt_all += tags[:_len].cpu().tolist()
            pred_all += pred[:_len]

    f1 = f1_score(gt_all, pred_all, average='macro')
    loss_mean = loss_total / n_sample

    return loss_mean, f1


def evaluate(args, model, data):
    model.eval()
    n_sample = 0
    loss_total = 0.
    pred_all, gt_all = [], []

    with torch.no_grad():
        for batch in tqdm(data, desc='  - evaluating', leave=False):
            words_batch, chars_batch, tags_batch, lens_batch = map(lambda x: x.to(args.device), batch)
            model.zero_grad()

            loss_batch, pred_batch = model.get_loss(words_batch, chars_batch, tags_batch, lens_batch)

            # calculate loss and f1
            n_sample += lens_batch.sum()
            loss_total += loss_batch * lens_batch.sum()
            for pred, tags, _len in zip(pred_batch, tags_batch, lens_batch):
                gt_all += tags[:_len].cpu().tolist()
                pred_all += pred[:_len]

        f1 = f1_score(gt_all, pred_all, average='macro')
        loss_mean = loss_total / n_sample

    return loss_mean, f1
