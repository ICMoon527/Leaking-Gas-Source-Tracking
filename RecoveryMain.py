import os
import random
import argparse
import torch
from sklearn.model_selection import train_test_split
from sklearn import metrics
import Model
import GasDataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
from utils.logger import setup_logger
import time
from utils import SomeUtils
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

criterion = nn.MSELoss()
lr_list = []

def train(args, model, optimizer, epoch, trainloader, trainset, logger):
    model.train()

    # update lr for this epoch
    lr = SomeUtils.learning_rate_2(epoch, args.warmup_steps, args.warmup_start_lr, args.epochs, args.lr, args.power)
    SomeUtils.update_lr(optimizer, lr)
    lr_list.append(lr)

    params = sum([np.prod(p.size()) for p in model.parameters()])
    logger.info('|  Number of Trainable Parameters: ' + str(params))
    logger.info('\n=> Training Epoch #%d, LR=%.8f' % (epoch, lr))

    for batch_idx, (inputs, targets, inputs_origin) in enumerate(trainloader):

        if args.device == torch.device('cuda'):
            inputs, targets, inputs_origin = inputs.cuda(), targets.cuda(), inputs_origin.cuda()  # GPU settings
        optimizer.zero_grad()  # 梯度清零
        
        inputs, targets, inputs_origin = Variable(inputs, requires_grad=False), Variable(targets), Variable(inputs_origin)
        
        out = model(inputs)
        loss = criterion(out, inputs_origin) # Loss
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            logger.info('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f'
                                % (epoch, args.epochs, batch_idx+1,
                                (len(trainset)//args.batchsize)+1, loss.detach().item()))

    return loss.detach().item(), 0

def test(best_result, args, model, epoch, testloader, logger):
    model.eval()
    correct = 0.
    total = 0

    loss_list = []

    for batch_idx, (inputs, targets, inputs_origin) in enumerate(testloader):
        if args.device == torch.device('cuda'):
            inputs, targets, inputs_origin = inputs.cuda(), targets.cuda(), inputs_origin.cuda()
        inputs, targets, inputs_origin = Variable(inputs, requires_grad=False), Variable(targets), Variable(inputs_origin)

        out = model(inputs)  # (batch, nClasses)
        loss_list.append(criterion(out, inputs_origin).detach().item())

        if batch_idx % 10 == 0:
            logger.info('output: \n{}\ninput_origin: \n{}\ninput: \n{}\ndelta: \n{}'.format(out.detach().cpu().numpy()[0],
                                                                                            inputs_origin.detach().cpu().numpy()[0],
                                                                                            inputs.detach().cpu().numpy()[0],
                                                                                            (out-inputs_origin).detach().cpu().numpy()[0]))
    
    loss = np.mean(np.array(loss_list), axis=0)

    # accuracy = 100. * float(correct) / float(total)
    logger.info("\n| Validation Epoch #%d\t\t\tloss =  %.4f" % (epoch, loss))
    # logger.info('classification report: \n{}'.format(classification_report))
    if loss < best_result:
        logger.info('\n| Saving Best model...\t\t\tloss = %.4f < %.4f (Best Before)' % (loss, best_result))
        state = {
            'model': model,
            'accuracy': loss,
            'epoch': epoch,
        }

        prefix = args.model + '_' + args.optimizer + '_' + str(loss).split('.')[0]+'.'+str(loss).split('.')[1][0:2] + '_'
        torch.save(state, os.path.join(args.save_dir, prefix+'checkpoint.t7'))
        best_result = loss
    else:
        logger.info('\n| Not best... {:.4f} > {:.4f}'.format(loss, best_result))

    return best_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['SVM', 'DNN', 'ATTDNN', 'ATT', 'DNNATT'], default='ATT')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batchsize", type=int, default=10240)  # 486400
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else 'cpu', choices=["cpu", "cuda"])
    parser.add_argument('--optimizer', default='Adam', type=str, choices=['SGD','Adam','Adamax'])
    parser.add_argument('--save_dir', default='./Results/Recover', type=str)
    parser.add_argument('--nonlin', default="elu", type=str, choices=["relu", "elu", "softplus", 'sigmoid'])
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

    if args.deterministic:  # 就是方便复现
        print('\033[31mModel Deterministic\033[0m')
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        torch.backends.cudnn.deterministic=True
    SomeUtils.try_make_dir(args.save_dir)
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 检索当前条件下结果最佳的模型
    best_result = np.inf
    files = os.listdir(args.save_dir)
    for file in files:
        if args.model+'_' in file and args.optimizer+'_' in file and 't7' in file:
            best_result = float(file.split('_')[-2]) if float(file.split('_')[-2]) < best_result else best_result
    print('\033[31mBest Result Before: {}\033[0m'.format(best_result))

    """
    Read Data
    """
    trainset = GasDataset.GasDataset(args, True)
    testset = GasDataset.GasDataset(args, False)
    if args.deterministic:
        trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2, worker_init_fn=np.random.seed(1234))
        testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2, worker_init_fn=np.random.seed(1234))
    else:
        trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    """
    Choose model
    """
    input_charac_num = 61
    nClasses = 79
    nSensors = 61
    if args.model == 'SVM':
        model = Model.SVM(args, input_charac_num, nClasses)
    elif args.model == 'DNN':
        model = Model.DNN(args, input_charac_num, nSensors)
    elif args.model == 'ATTDNN':
        model = Model.ATTDNN(args, input_charac_num, nClasses)
    elif args.model == 'ATT':
        model = Model.AttentionLayer(args, input_charac_num, input_charac_num*nClasses, nClasses)
    elif args.model == 'DNNATT':
        model = Model.DNNATT(args, input_charac_num, nClasses)

    model.to(args.device)
    if args.device == torch.device('cuda'):
        model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))  # 并行
        cudnn.benchmark = True  # 统一输入大小的情况下能加快训练速度

    """
    Choose Optimizer
    """
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.999, weight_decay=args.weight_decay)  # lr 0.000005
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

###############################################################################################
    """
    TRAIN
    """
    # set up a logger
    logger = setup_logger(args.model+'_'+args.optimizer, args.save_dir, 0, args.model+'_'+args.optimizer+'_log.txt', mode='w+')

    logger.info(model)
    logger.info('='*20+'Training Model'+'='*20)
    elapsed_time = 0
    loss_list, accuracy_list = [], []
    for epoch in range(1, 1+args.epochs):

        start_time = time.time()
        loss, _ = train(args, model, optimizer, epoch, trainloader, trainset, logger)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        logger.info('| Elapsed time : %d:%02d:%02d' % (SomeUtils.get_hms(elapsed_time)))

        loss_list.append(loss)
        # accuracy_list.append(accuracy)

    """
    TEST
    """
    logger.info('='*20+'Testing Model'+'='*20)
    new_best = test(best_result, args, model, epoch, testloader, logger)
###############################################################################################

    """
    画loss, accuracy图
    """
    prefix = args.model + '_' + args.optimizer + '_'#  + str(args.lr) + '_'
    if new_best < best_result:  # MSE
        SomeUtils.draw_fig(args, loss_list, prefix+'Train_Loss')
        # SomeUtils.draw_fig(args, accuracy_list, prefix+'Train_Accuracy')
    SomeUtils.draw_fig(args, lr_list, prefix+'Learning Rate')


    """
    添加softmax层后做mseloss会让输出更加倾向于输出负数
    用sigmoid,不收敛
    试试在recovery中用dropout,

    可逆全连接恢复网络
    如unet一般的桥接,使网络专注于拟合缺失部分 or 反unet
    对输入做position encoding
    用皮尔森相关系数衡量data recovery的好坏
    添加风向预测模型以辅助data recovery
    """