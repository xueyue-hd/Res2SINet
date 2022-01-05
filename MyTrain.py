import torch
import argparse
from Src.utils import Dataloader
from Src.utils.trainer import eval_mae,  trainer
from Src.SINet import SINet_ResNet50

# 创建一个解析对象
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30)
# 如果在colab上跑的话，尤其是代码在drive里面时，会直接建在/
parser.add_argument('--save_model', type=str, default='./Result/2020-CVPR-SINet-New/Model/')
parser.add_argument('--save_epoch', type=int, default=0)
parser.add_argument('--save_loss_mae', type=str, default='./Result/2020-CVPR-SINet-New/Loss_Mae/')
# 需要根据已有的模型改变路由,记住同时在训练模型部分修改已经训练的次数
parser.add_argument('--model_path', type=str,
                    default='./Result/2020-CVPR-SINet-New/Model/SINet_30.pth')
# 进行解析
opt = parser.parse_args()

# 获取训练数据
train_size = 352
for dataset in ['COD10K_CAMO_CombinedTrainingDataset']:
    train_loader = Dataloader.get_loader(image_root='./Dataset/TrainDataset/{}/Image/'.format(dataset),
                                         gt_root='./Dataset/TrainDataset/{}/GT/'.format(dataset),
                                         batchsize=2,
                                         trainsize=train_size)
# 构建模型
# 实例化模型
# model = SINet_ResNet50()
model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))

# 训练模型
# epoch = 30, lr = 0.0001, Adam optimizer, batch = 36, total_step = len(train_loader)
# 4.2 实例化优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for i in range(opt.epoch):
    # 此处修改已经训练的次数
    opt.save_epoch = i+1
    trainer(train_loader=train_loader, model=model, optimizer=optimizer, epoch=i, opt=opt, loss_func=eval_mae,
            total_step=len(train_loader))

print("训练完毕")
