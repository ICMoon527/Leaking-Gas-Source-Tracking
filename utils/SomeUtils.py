import math
import os
from traceback import print_last

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

def learning_rate(init, epoch):
    optim_factor = 0
    if epoch > 160:
        optim_factor = 3
    elif epoch > 120:
        optim_factor = 2
    elif epoch > 60:
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)

def learning_rate_2(epoch, warmup_steps, warmup_start_lr, max_iter, lr0, power=0.5):
    warmup_factor = (lr0/warmup_start_lr)**(1/warmup_steps)
    if epoch <= warmup_steps:
        lr = warmup_start_lr*(warmup_factor**epoch)
    else: # epoch <= max_iter/4*3:
        factor = (1-(epoch-warmup_steps)/(max_iter-warmup_steps))**power
        lr = lr0*factor
    # else:
    #     lr = learning_rate(5e-4, epoch-max_iter/4*3)
    return lr

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def draw_fig(args, data_list, name):
    import matplotlib
    matplotlib.use('Agg')  # FIX: tkinter.TclError: couldn't connect to display "localhost:11.0" 
    import matplotlib.pyplot as plt
    x1 = range(1, args.epochs+1)
    y1 = data_list

    plt.cla()
    plt.title(name.split('_')[-1]+' vs. epoch', fontsize=15)
    # plt.plot(x1, y1, '.-')
    plt.plot(x1, y1)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel(name.split('_')[-1], fontsize=15)
    plt.grid()
    plt.savefig(args.save_dir+'/'+name+".png")

    # plt.show()

def try_make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)

if __name__ == '__main__':
    # import numpy as np
    # a = np.ones((2, 4))*10.
    # b = np.random.random((2,4))
    # print(a*b)
    # print(b)
    
    
    # # 画learning rate曲线图
    # max_iter = 200
    # epochs = [i for i in range(max_iter)]
    # lr_list  = []
    # for epoch in epochs:
    #     lr_list.append(learning_rate_2(epoch, 10, 1e-5, max_iter, 1e-3, 2))
    #     # lr_list.append(learning_rate(1e-3, epoch))
    
    # import matplotlib
    # matplotlib.use('Agg')  # FIX: tkinter.TclError: couldn't connect to display "localhost:11.0" 
    # import matplotlib.pyplot as plt
    # x1 = range(1, max_iter+1)
    # y1 = lr_list

    # plt.cla()
    # plt.title('lr test'+' vs. epoch', fontsize=15)
    # # plt.plot(x1, y1, '.-')
    # plt.plot(x1, y1)
    # plt.xlabel('epoch', fontsize=15)
    # plt.ylabel('lr_test', fontsize=15)
    # plt.grid()
    # plt.savefig('test.png')
    
    import torch.nn.functional as F
    import torch
    print(torch.arange(0, 5) % 3)
    print(F.one_hot(torch.arange(0, 5) % 3, num_classes=5))
    print(F.one_hot(torch.tensor([1,2,0]), num_classes=5))
    