import torch
import numpy as np
import seaborn as sns
from torch.optim import Adam
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torchmetrics import Accuracy, AUROC
from .cssr import CSSRModel, CSSRCriterion
from utils.scheduler import WarmupCosineLR
from .proposed import proposed

class OSC_SCL(pl.LightningModule):
    def __init__(self, args, info, num_classes, data):
        super(OSC_SCL, self).__init__()
        self.args = args
        self.info = info
        self.num_classes = num_classes
        self.encoder = proposed(args.dataset, args.patch, args.group)
        self.crt = CSSRCriterion(info['arch_type'], False)
        self.model = CSSRModel(self.num_classes, info, self.crt)
        self.data = data
        self.thresh = args.thresh
        self.mean = 0
        self.std = 0
        self.epoch = 0
        self.prepared = -999
        self.train_acc_meter = Accuracy(task="multiclass", num_classes=num_classes)
        self.scores = []
        self.prediction = []
        self.y = []

    def forward(self, x):
        encoder_out = self.encoder(x)
        return encoder_out

    def training_step(self, batch, idx_batch):
        x, y = batch
        out = self(x)
        loss, pred, scores = self.model(out, y, reqpredauc=True)
        precision = torch.tensor(pred).cuda()
        self.train_acc_meter.update(precision, y)
        self.log('loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc_meter, prog_bar=True)
        self.epoch += 1

    def on_test_epoch_start(self):
        if self.prepared != self.epoch:
            self.tpred, self.tscores, _, _ = self.scoring(self.data['known']['train'], True)
            self.prepared = self.epoch

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self(x)
        y = y.cpu().numpy()
        pred, scr, dev = self.model(x, reqpredauc=True, prepareTest=False)
        self.prediction.append(pred)
        self.scores.append(scr)
        self.y.append(y)

    def on_test_epoch_end(self):
        self.prediction = np.concatenate(self.prediction)
        self.scores = np.concatenate(self.scores)
        y = torch.tensor(np.concatenate(self.y))
        if self.info['integrate_score'] != "S[0]":
            self.tpred, self.tscores, _, _ = self.scoring(self.data['known']['train'], False)
            mean, std = self.tscores.mean(axis=0), self.tscores.std(axis=0)
            self.scores = (self.scores - mean) / (std + 1e-8)
        S = self.scores.T
        scores = -(eval(self.info['integrate_score']))
        scores1 = scores[np.concatenate(self.y) == self.num_classes]
        # sns.distplot(scores, kde=True, label="scores", color='red')
        # sns.distplot(scores1, kde=True, label="unknown", color='blue')
        # plt.show()
        self.prediction[scores > self.thresh] = self.num_classes
        self.prediction = torch.tensor(self.prediction)
        self.y=y

    def scoring(self, loader, prepare=False):
        gts = []
        deviations = []
        scores = []
        prediction = []
        with torch.no_grad():
            for d in loader:
                x1 = d[0].cuda(non_blocking=True)
                gt = d[1].numpy()
                x1 = self(x1)
                pred, scr, dev = self.model(x1, reqpredauc=True, prepareTest=prepare)
                prediction.append(pred)
                scores.append(scr)
                gts.append(gt)

        prediction = np.concatenate(prediction)
        scores = np.concatenate(scores)
        gts = np.concatenate(gts)

        return prediction, scores, deviations, gts

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = WarmupCosineLR(optimizer, lr_min=1e-6, lr_max=1e-3, warm_up=100, T_max=self.trainer.max_epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    

def get_model(args, data_info, data):
    model_args = {
        'args': args,
        'info': data_info,
        'num_classes': len(args.known_classes),
        'data': data,
    }
    model = OSC_SCL(**model_args)
    return model

# train_num 50;  thresh 8
# train_num 200; thresh 5
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', type=str, default='mymodel')
    parser.add_argument('--train_num', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--patch', type=int, default=13,help='15,19')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--dataset', default='Houston')
    parser.add_argument('--group', type=int, default=2, help='group number 2,8')
    parser.add_argument('--thresh', type=float, default=4, help='thresh 9,8')
    args = parser.parse_args()

    if args.dataset == 'PaviaU':
        args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        args.unknown_classes = [10]
    elif args.dataset == 'Indian_pines':
        args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8]
        args.unknown_classes = [9]
    elif args.dataset == 'Houston':
        args.known_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        args.unknown_classes = [12]

    return args
