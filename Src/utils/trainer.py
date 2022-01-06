import torch
from torch.autograd import Variable
from datetime import datetime
import os
import numpy as np
# from apex import amp
import torch.nn.functional as F


def eval_mae(y_pred, y):
    """
    evaluate MAE (for test or validation phase)
    :param y_pred:
    :param y:
    :return: Mean Absolute Error
    """
    return torch.abs(y_pred - y).mean()


def numpy2tensor(numpy):
    """
    convert numpy_array in cpu to tensor in gpu
    :param numpy:
    :return: torch.from_numpy(numpy).cuda()
    """
    return torch.from_numpy(numpy).cuda()


def clip_gradient(optimizer, grad_clip):
    """
    recalibrate the misdirection in the training
    重新校准训练中的错误方向
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=80):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def trainer(train_loader, model, optimizer, epoch, opt, loss_func, total_step):
    """
    Training iteration
    :param train_loader:训练数据集
    :param model:自定义训练模型
    :param optimizer:优化器
    :param epoch:第i次迭代次数
    :param opt:解析对象的实例化，见MyTest.py，存放总迭代次数
    :param loss_func:损失函数
    :param total_step:当前训练数据的size
    :return:
    """
    model.train()

    Loss_i = []
    Loss_s = []

    for step, data_pack in enumerate(train_loader):
        optimizer.zero_grad()
        # print(data_pack)
        images, gts = data_pack
        images = Variable(images).cuda()  # 2*3*352*352
        # print(images.shape)
        gts = Variable(gts).cuda()  # 2*1*352*352

        cam_sm, cam_im = model(images)  # 预测结果 y_predict

        # 计算当前batchsize的平均损失,损失函数为mae,item()将tensor数据转换成标量
        loss_sm = loss_func(cam_sm, gts)
        runing_loss_s = loss_sm.item()
        loss_im = loss_func(cam_im, gts)
        runing_loss_i = loss_im.item()
        loss_total = loss_sm + loss_im

        # 保存当前batchsize的平均损失
        Loss_i.append(runing_loss_i)
        Loss_s.append(runing_loss_s)

        # with amp.scale_loss(loss_total, optimizer) as scale_loss:
        #     scale_loss.backward()
        # 去掉amp后，是否直接对loss_total进行backward？
        loss_total.backward()

        # clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if step % 10 == 0 or step == total_step:
            # 每隔10个batchsize 打印当前batchsize的训练loss的平均值。
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => '
                  '[Loss_s: {:.4f} Loss_i: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, total_step, step, loss_sm.data, loss_im.data))

    save_path = opt.save_model
    loss_save_path = opt.save_loss_mae
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(loss_save_path, exist_ok=True)

    if (epoch+1) % opt.save_epoch == 0:
        # 运行一轮数据，保存当前轮次中的所有的
        torch.save(model.state_dict(), save_path + 'SINet_%d.pth' % (epoch + 1))
        return np.mean(Loss_s), np.mean(Loss_i)
