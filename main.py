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
from utils.FocalLoss import *

bad_classes = [25, 27, 38, 39, 40, 41, 42, 44, 46, 47, 50, 51, 52, 53]

# class_weights = torch.Tensor([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  
#                                1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  
#                                1.,  1.,  1.,  1.,  6.,  1.,  6.,  1.,  1.,  1.,  
#                                1.,  1.,  1.,  1.,  1.,  1.,  1.,  7.,  4., 40., 
#                                40.,  4., 1.,  7.,  1.,  7.,  7.,  1.,  1.,  4.,  
#                                4., 100., 100.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  
#                                1.,  1.,  1.,  1.,  2.,  2.,  1.,  1.,  1.,  1.,
#                                1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  2.])

# class_weights = torch.Tensor([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  
#                                1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  
#                                1.,  1.,  1.,  1.,  .3,  1.,  .3,  1.,  1.,  1.,  
#                                1.,  1.,  1.,  1.,  1.,  1.,  1.,  .3,  .3,  .05, 
#                                .05,  .3, 1.,  .3,  1.,  .3,  .3,  1.,  1.,  .2,  
#                                .3,  .01, .01,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  
#                                1.,  1.,  1.,  1.,  .5,  .5,  1.,  1.,  1.,  1.,
#                                1.,  1.,  1.,  1.,  1.,  1.,  1.,  .5,  .5])
# if torch.cuda.is_available():
#     class_weights = class_weights.cuda()

criterion = nn.CrossEntropyLoss()
# criterion = FocalLossV1()
lr_list = []

def train(args, model, optimizer, epoch, trainloader, trainset, logger, model_att=None):
    model.train()
    total = 0
    correct = 0
    accuracy = 0

    # update lr for this epoch
    # lr = SomeUtils.learning_rate(args.lr, epoch)  # 0.000005
    lr = SomeUtils.learning_rate_2(epoch, args.warmup_steps, args.warmup_start_lr, args.epochs, args.lr, args.power)
    # 专门针对超长epochs
    if args.epochs>3000:
        if epoch>3000:  # 收尾
            lr = SomeUtils.learning_rate_2(3000, args.warmup_steps, args.warmup_start_lr, 3000, args.lr, args.power)
        else:
            lr = SomeUtils.learning_rate_2(epoch, args.warmup_steps, args.warmup_start_lr, 3000, args.lr, args.power)
    SomeUtils.update_lr(optimizer, lr)
    lr_list.append(lr)

    params = sum([np.prod(p.size()) for p in model.parameters()])
    logger.info('|  Number of Trainable Parameters: ' + str(params))
    logger.info('\n=> Training Epoch #%d, LR=%.8f' % (epoch, lr))

    for batch_idx, (inputs, targets, _) in enumerate(trainloader):

        if args.device == torch.device('cuda'):
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()  # 梯度清零
        
        inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)
        
        if model_att is not None:
            inputs = model_att(inputs)  # 前面预训练网络的输出
        out = model(inputs)
        # loss = criterion(out, F.one_hot(targets, num_classes=args.nClasses)) # Loss for focal loss
        loss = criterion(out, targets) # Loss for CE loss
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            _, predicted = torch.max(out.detach(), 1)
            total += targets.size(0)
            correct += predicted.eq(targets.detach()).cpu().sum()
            accuracy = 100.*correct.type(torch.FloatTensor)/float(total)
            logger.info('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc: %.3f = correct: %3d / total: %3d'
                                % (epoch, args.epochs, batch_idx+1,
                                (len(trainset)//args.batchsize)+1, loss.detach().item(),
                                accuracy,
                                correct, total))
            
    if epoch % 100 == 1:
        logger.info('\n| Saving model...\t\t\taccuracy = %.4f' % (accuracy))
        state = {
            'model': model.module if isinstance(model, torch.nn.DataParallel) else model,
            'optimizer':optimizer.state_dict(),
            'accuracy': accuracy,
            'epoch': epoch,
        }
        
        prefix = args.model + '_' + args.optimizer + '_epoch_' + str(epoch) + '_'
        # torch.save(state, os.path.join(args.save_dir, prefix+'checkpoint.t7'))

    return loss.detach().item(), accuracy

def test(best_result, args, model, epoch, testloader, logger, model_att):
    model.eval()
    correct = 0.
    total = 0

    predicted_list = []
    target_list = []

    for batch_idx, (inputs, targets, _) in enumerate(testloader):
        if args.device == torch.device('cuda'):
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)

        if model_att is not None:
            inputs = model_att(inputs)
        out = model(inputs)  # (batch, nClasses)
        _, predicted = torch.max(out.detach(), 1)
        correct += predicted.eq(targets.detach()).sum().item()
        total += targets.size(0)

        predicted_list.append(predicted.cpu().numpy())
        target_list.append(targets.cpu().numpy())
    
    predicted_list = np.hstack(predicted_list)
    target_list = np.hstack(target_list)
    classification_report = metrics.classification_report(target_list, predicted_list, target_names=['Source '+str(i+1) for i in range(79)])

    accuracy = 100. * float(correct) / float(total)
    logger.info("\n| Validation Epoch #%d\t\t\taccuracy =  %.4f" % (epoch, accuracy))
    logger.info('classification report: \n{}'.format(classification_report))
    if accuracy > best_result:
        logger.info('\n| Saving Best model...\t\t\taccuracy = %.4f > %.4f (Best Before)' % (accuracy, best_result))
        state = {
            'model': model.module if isinstance(model, torch.nn.DataParallel) else model,
            'accuracy': accuracy,
            'epoch': epoch,
        }
        
        prefix = args.model + '_' + args.optimizer + '_' + str(accuracy)[0:5] + '_'
        torch.save(state, os.path.join(args.save_dir, prefix+'checkpoint.t7'))
        best_result = accuracy
    else:
        logger.info('\n| Not best... {:.4f} < {:.4f}'.format(accuracy, best_result))

    return best_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['SVM', 'DNN', 'ATTDNN', 'preDN', 'DNNATT', 'UDNN', 'Resume'], default='DNN')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batchsize", type=int, default=10240)  # 486400
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cpu" if torch.cuda.is_available() else 'cpu', choices=["cpu", "cuda"])
    parser.add_argument('--optimizer', default='Adam', type=str, choices=['SGD','Adam','Adamax'])
    parser.add_argument('--save_dir', default='./Results/79sources', type=str)
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
    parser.add_argument('--input_droprate', default=0., type=float, help='the max rate of the detectors may fail')
    parser.add_argument('--initial_dim', default=256, type=int)
    parser.add_argument('--continueFile', default='./Results/79sources/DNN-Adam-0-3000-largerRange-focalLoss/bk.t7', type=str)

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
    best_result = -np.inf
    files = os.listdir(args.save_dir)
    for file in files:
        if args.model+'_' in file and args.optimizer+'_' in file and 't7' in file:
            best_result = float(file.split('_')[-2]) if float(file.split('_')[-2]) > best_result else best_result
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
    model_att = None
    if args.model == 'SVM':
        model = Model.SVM(args, input_charac_num, nClasses)
    elif args.model == 'DNN':
        model = Model.DNN(args, input_charac_num, nClasses)
    elif args.model == 'ATTDNN':
        model = Model.ATTDNN(args, input_charac_num, nClasses)
    elif args.model == 'UDNN':
        model = Model.UDNN(args, input_charac_num, nClasses)
    elif args.model == 'preDN':
        # 预训练的恢复模型，接上分类模型DNN
        # 检索最佳的恢复模型
        recover_result = np.inf
        files = os.listdir('./Results/Recover')
        for file in files:
            if 'DNN_' in file and 'Adam_' in file and 't7' in file:
                recover_result = file.split('_')[-2] if float(file.split('_')[-2]) < float(recover_result) else recover_result
        model_att_path = os.path.join('./Results/Recover', 'DNN_Adam_'+str(recover_result)+'_checkpoint.t7')
        model_att = torch.load(model_att_path)['model']  # for recover original data
        # print('\033[31mpretrained ATT model {}_checkpoint.t7 is loaded\033[0m'.format(recover_result))
        model = Model.DNN(args, input_charac_num, nClasses)  # for classification
    elif args.model == 'Resume':
        # 断点恢复训练
        file = torch.load(args.continueFile)
        model, epoch, accuracy, optimizer = file['model'], file['epoch'], file['accuracy'], file['optimizer']

    model.to(args.device)
    if args.model == 'preDN':
        model_att.to(args.device)

    if args.device == torch.device('cuda') and torch.cuda.device_count()>1:
        if args.model == 'preATTDNN':
            model_att = torch.nn.DataParallel(model_att, range(torch.cuda.device_count()))  # 并行

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
    
    if args.model == 'Resume':
        optimizer.load_state_dict(file['optimizer'])

###############################################################################################
    """
    TRAIN
    """
    # set up a logger
    logger = setup_logger(args.model+'_'+args.optimizer, args.save_dir, 0, args.model+'_'+args.optimizer+'_log.txt', mode='w+')

    if args.model == 'preDN':
        logger.info('pretrained model {}_checkpoint.t7 is loaded'.format(recover_result))
    logger.info(model)
    logger.info('='*20+'Training Model'+'='*20)
    elapsed_time = 0
    loss_list, accuracy_list = [], []
    start_epoch = 0
    if args.model == 'Resume':
            start_epoch = file['epoch']

    for epoch in range(start_epoch+1, 1+args.epochs):

        start_time = time.time()
        loss, accuracy = train(args, model, optimizer, epoch, trainloader, trainset, logger, model_att)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        logger.info('| Elapsed time : %d:%02d:%02d' % (SomeUtils.get_hms(elapsed_time)))

        loss_list.append(loss)
        accuracy_list.append(accuracy)

    """
    TEST
    """
    logger.info('='*20+'Testing Model'+'='*20)
    new_best = test(best_result, args, model, epoch, testloader, logger, model_att)
###############################################################################################

    """
    画loss, accuracy图
    """
    prefix = args.model + '_' + args.optimizer + '_'#  + str(args.lr) + '_'
    if new_best > best_result:  # accuracy
        SomeUtils.draw_fig(args, loss_list, prefix+'Train_Loss')
        SomeUtils.draw_fig(args, accuracy_list, prefix+'Train_Accuracy')
    SomeUtils.draw_fig(args, lr_list, prefix+'Learning Rate')

    """
    Test Again after Loading Model
    """
    # prefix = args.model + '_' + args.optimizer + '_' + str(new_best)[0:5] + '_'
    # model = torch.load(os.path.join(args.save_dir, prefix+'checkpoint.t7'))['model'].cuda()
    # logger.info('='*20+'Testing Model Again'+'='*20)
    # new_best = test(best_result, args, model, epoch, testloader, logger)



    """
    如果不把风速加进分类的原因是风速变化很大，特别是城市中由于建筑物影响也比较难以确定风速
    数据模拟的scale很大,包含了四个工业园区(不止
    数据模拟的AERMOD模型国际认可度高(https://www.kdainfo.com/show/1597.html)模型简介，并且数据有随机性(评价标准为最值牌序)
    归一化做了,效果不如不做,还需要另外加batchnorm层,收敛还慢,精度也不那么高,batchnorm层还会让batchsize减小一半(再验证一下) 应该是归一化后对低频高频信息有影响的原因吧
    learning rate warmup strategy power=0.5效果挺不错
    数据增强，输入增加扰动
    加入受体随机失活检查稳定性(Dropout Layer),仅作用于第一层之前
    CNN+DNN的效果, 范围太大，数据稀疏，不方便

    不好分类的source名单: 25, 27, 38, 39, 40, 41, 42, 44, 46, 47, 49, 50, 51, 52, 53 可以尝试使用center loss

    TODO
    堆叠边的新结构（与核函数结合
    Positional Encoding(self attention), 以及各种encoding对比对data recovery效果的提升情况
    有没有可能弄一个扩大数据之间差异的激活函数
    先看看把风速加进去对recovery有没有帮助
    
    加一个小网络专门对精度不高的烟囱进行分类作为补充信息
    
    如果以上效果不太行的话,加入attention机制看看效果,测试加入self-attention的效果,以及普通结果加入attention的效果

    最后的分类结果把one-hot变成类似二进制的表示缩小输出的维度?
    最后再用低lr跑一段
    把精度较低的source单独拎出来做数据增强
    """