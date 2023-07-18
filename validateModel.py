import torch
import numpy as np
import GasDataset
from torch.utils.data import DataLoader
import random
from torch.autograd import Variable
from sklearn import metrics
# from main import test
from RecoveryMain import test
import argparse
from utils.logger import setup_logger

filepath = 'Results/Recover/ATT_Adam_6.962_checkpoint.t7'
model = torch.load(filepath)['model']
model.cuda()

model.eval()
correct = 0.
total = 0

predicted_list = []
target_list = []

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic=True

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['SVM', 'DNN', 'ATTDNN', 'ATT'], default='ATT')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batchsize", type=int, default=10240)  # 486400
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else 'cpu', choices=["cpu", "cuda"])
parser.add_argument('--optimizer', default='Adam', type=str, choices=['SGD','Adam','Adamax'])
parser.add_argument('--save_dir', default='./Results/Recover', type=str)
parser.add_argument('--nonlin', default="elu", type=str, choices=["relu", "elu", "softplus"])
parser.add_argument('--weight_decay', default=0., type=float, help='coefficient for weight decay')
parser.add_argument('-deterministic', '--deterministic', dest='deterministic', action='store_true',
                    help='fix random seeds and set cuda deterministic')
parser.add_argument('--warmup_steps', default=10, type=int)
parser.add_argument('--warmup_start_lr', default=1e-5, type=float)
parser.add_argument('--power', default=0.5, type=float)
parser.add_argument('-batchnorm', '--batchnorm', action='store_true')
parser.add_argument('--dropout_rate', default=0., type=float)
parser.add_argument('--nClasses', default=79, type=int)
parser.add_argument('--input_droprate', default=0.1, type=float, help='the max rate of the detectors may fail')

args = parser.parse_args()

testset = GasDataset.GasDataset(args, False)
testloader = DataLoader(testset, batch_size=10240, shuffle=False, num_workers=2, worker_init_fn=np.random.seed(1234))
best_result = 0
args.device = torch.device('cuda')
logger = setup_logger('test', './', 0, 'test_log.txt', mode='w+')
new_best = test(best_result, args, model, 2, testloader, logger)