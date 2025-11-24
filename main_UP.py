import pytorch_lightning as pl
import numpy as np
from utils.dataLoader import getDataLoader
from utils.utils import getDatasetInfo
from model.OSC_SCL_UP import get_model, parse_args
from sklearn.metrics import precision_score, classification_report,cohen_kappa_score,recall_score

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


def every_epoch(seed, results, path):
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
    label,prediction=model.y, model.prediction
    kappa=cohen_kappa_score(label,prediction)
    f1 = F_measure(label, prediction, unknown=len(args.known_classes))
    oa = precision_score(label, prediction, average="micro")
    aa = recall_score(label, prediction, average="macro")
    all_result = classification_report(label, prediction, digits=4)
    all_result_dict = classification_report(label, prediction, digits=6, output_dict = True)
    print('test_oa: {:.4f}'.format(oa))
    print('test_aa: {:.4f}'.format(aa))
    print('f1_micro: {:.4f}'.format(f1))
    print('kappa: {:.4f}'.format(kappa))
    print(all_result)
    results['kappa'].append(kappa)
    results['f1_micro'].append(f1)
    results['test_oa'].append(oa)
    results['test_aa'].append(aa)
    recall_values = {key: value['recall'] for key, value in all_result_dict.items() if key not in ['accuracy', 'macro avg', 'weighted avg']}
    results['recall_values'].append(recall_values)
    with open(path, 'a') as f:
        f.write('seed: {}\n'.format(seed))
        f.write('oa: {:.4f}\n'.format(oa))
        f.write('aa: {:.4f}\n'.format(aa))
        f.write('kappa: {:.4f}\n'.format(kappa))
        f.write('f1: {:.4f}\n'.format(f1))
        f.write(all_result)
        f.write('\n')    
    return results

def main(seedx):
    results = {
        'recall_values': [],
        'test_oa': [],
        'test_aa': [],
        'kappa': [],
        'f1_micro': []
    }
    for seed in seedx:
        pl.seed_everything(seed, workers=True)
        train_num = parse_args().patch
        path='/mnt/HDD/data/zwj/model_1/my_model/'+str(train_num)+'_result_UP.txt'
        results = every_epoch(seed, results, path)
        
    avg_results = {
        'recall_values': {key: np.mean([recall[key] for recall in results['recall_values']]) for key in results['recall_values'][0]},
        'test_oa': np.mean(results['test_oa']),
        'test_aa': np.mean(results['test_aa']),
        'kappa': np.mean(results['kappa']),
        'f1_micro': np.mean(results['f1_micro'])
    }
    
    std_results = {
        'recall_values': {key: np.std([recall[key] for recall in results['recall_values']]) for key in results['recall_values'][0]},
        'test_oa': np.std(results['test_oa']),
        'test_aa': np.std(results['test_aa']),
        'kappa': np.std(results['kappa']),
        'f1_micro': np.std(results['f1_micro'])
    }
    
    print('Results:')
    for key, value in avg_results.items():
        if key == 'recall_values':
            print('recall_values:')
            for class_key in value:
                avg_val = avg_results['recall_values'][class_key]
                std_val = std_results['recall_values'][class_key]
                print(f'{class_key}: {avg_val:.4f} ± {std_val:.4f}')
        else:
            avg_val = avg_results[key]
            std_val = std_results[key]
            print(f'{key}: {avg_val:.4f} ± {std_val:.4f}')
    
    with open(path, 'a') as f:
        f.write('Results:\n')
        for key, value in avg_results.items():
            if key == 'recall_values':
                f.write('recall_values:\n')
                for class_key in value:
                    avg_val = avg_results['recall_values'][class_key]
                    std_val = std_results['recall_values'][class_key]
                    f.write(f'{class_key}: {avg_val:.4f} ± {std_val:.4f}\n')
            else:
                avg_val = avg_results[key]
                std_val = std_results[key]
                f.write(f'{key}: {avg_val:.4f} ± {std_val:.4f}\n')

if __name__ == '__main__':
    seedx = [100,200,300,400,500,600,700,800,900,1000]
    main(seedx)