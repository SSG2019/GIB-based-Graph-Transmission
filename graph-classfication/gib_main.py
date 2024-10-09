import os
from itertools import product

import argparse
from datasets import get_dataset
from train_eval import cross_validation_with_val_set

from gib_gin import GIBGIN, Discriminator
from gib_gcn import GIBGCN
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--dataset', type=str, default='PROTEINS')
parser.add_argument('--net', type=int, default=0)
parser.add_argument('--inner_loop', type=int, default=50)
parser.add_argument('--mi_weight', type=float, default=0)
parser.add_argument('--pp_weight', type=float, default=0.1)
parser.add_argument('--model_n', default='test_'+str(uuid.uuid4())[:8])
parser.add_argument('--result_n', type=str, default="test")
args = parser.parse_args()

layers = [2]
hiddens = [16, 32]
# datasets = ['PROTEINS']
datasets = ['COLLAB']
nets = [GIBGCN]
model_n = args.model_n
result_n = args.result_n

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))

model_folder = 'model_saved'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model_path = os.path.join(model_folder, model_n)

result_folder = "RESULT"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
result_path = os.path.join(result_folder, result_n)

results = []
for dataset_name, Net in product(datasets, nets):

    if result_n:
        with open(result_path, 'a') as f:
            f.write(f"-------dataset_name: {dataset_name} ------- Net{Net}-------\n")

    for num_layers, hidden in product(layers, hiddens):
        dataset = get_dataset(dataset_name, sparse=True)
        model = Net(dataset, num_layers, hidden)
        discriminator = Discriminator(hidden)
        val_acc, test_acc, ACCs = cross_validation_with_val_set(
            dataset,
            model,
            discriminator,
            model_path,
            snr=5,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            inner_loop = args.inner_loop,
            mi_weight = args.mi_weight,
            pp_weight=args.pp_weight,
            logger= None
        )

        with open(result_path, 'a') as f:
            f.write(f"------num_layers{num_layers}, -------hidden{hidden}\n")
            fold = 0
            for row in ACCs:
                f.write(f"fold: {fold + 1} ")
                fold += 1
                s = -15
                for item in row:
                    f.write("snr:{:}, ACC: {:.4f}  ".format(s, item))
                    s += 10
                f.write("\n")
            f.write("---MEAN--- \n")
            s = -15
            for acc in test_acc:
                f.write("snr:{:}, ACC: {:.3f}  ".format(s, acc))
                s += 10
            f.write("\n\n")
