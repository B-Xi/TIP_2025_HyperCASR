import pytorch_lightning as pl
import numpy as np
from utils.dataLoader import getDataLoader
from utils.utils import getDatasetInfo
from model.OSC_SCL_HU import get_model, parse_args
from sklearn.metrics import precision_score, classification_report,cohen_kappa_score,recall_score
from utils.get_cls_map import get_cls_map_HU

def F_measure(labels, preds, unknown=-1): # F1
    true_pos = 0.
    false_pos = 0.
    false_neg = 0.
    for i in range(len(labels)):
        true_pos += 1 if preds[i] == labels[i] and labels[i] != unknown else 0
        false_pos += 1 if preds[i] != labels[i] and preds[i] != unknown else 0
        false_neg += 1 if preds[i] != labels[i] and labels[i] != unknown else 0

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    return 2 * ((precision * recall) / (precision + recall))

def main(seed):
    pl.seed_everything(seed, workers=True)
    args = parse_args()
    data_info = getDatasetInfo(args.dataset)
    data_loader: dict = getDataLoader(args, data_info)
    model = get_model(args, data_info, data_loader)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=[1],  # gpu index
        enable_checkpointing=False,
        deterministic=True,
    )

    trainer.fit(
        model,
        train_dataloaders=data_loader['known']['train'],
    )
    
    trainer.test(model, data_loader['unknown']['test'])
    label, prediction=model.y, model.prediction
    kappa=cohen_kappa_score(label,prediction)
    f1 = F_measure(label, prediction, unknown=len(args.known_classes))
    oa = precision_score(label, prediction, average="micro")
    aa = recall_score(label, prediction, average="macro")
    all_result = classification_report(label, prediction, digits=4)
    print('test_oa: {:.4f}'.format(oa))
    print('test_aa: {:.4f}'.format(aa))
    print('f1_micro: {:.4f}'.format(f1))
    print('kappa: {:.4f}'.format(kappa))
    print(all_result)
    get_cls_map_HU(prediction, data_loader['gt'], data_loader['unknown']['test_index'])

if __name__ == '__main__':
    seed = 900
    main(seed)