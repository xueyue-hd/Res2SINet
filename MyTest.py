import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse  # argparse 是 Python 内置的一个用于命令项选项与参数解析的模块
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor


# 创建一个解析对象
parser = argparse.ArgumentParser()
# 向该对象中添加你要关注的命令行参数和选项
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
# --testsize 为训练图片的尺寸
parser.add_argument('--model_path', type=str,
                    default='./Result/2020-CVPR-SINet-New/Model/SINet_30.pth')
# --model_path 为训练模型目录
# default='./Snapshot/2020-CVPR-SINet/SINet_40.pth
parser.add_argument('--test_save', type=str,
                    default='./Result/2020-CVPR-SINet-New/')
# --test_save 为推断掩码的保存目录
# 进行解析
opt = parser.parse_args()

model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['COD10K_TestingDataset']:
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    # NOTES:
    #  if you plan to inference on your customized dataset without grouth-truth,
    #  you just modify the params (i.e., `image_root=your_test_img_path` and `gt_root=your_test_img_path`)
    #  with the same filepath. We recover the original size according to the shape of grouth-truth, and thus,
    #  the grouth-truth map is unnecessary actually.
    test_loader = test_dataset(image_root='./Dataset/TestDataset1/{}/Image/'.format(dataset),
                               gt_root='./Dataset/TestDataset1/{}/GT/'.format(dataset),
                               testsize=opt.testsize)
    img_count = 1
    for iteration in range(test_loader.size):
        # load data
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # inference 将图片作为输入，传入model，得到model的预测的camouflage_map_im
        _, cam = model(image)
        # reshape and squeeze
        cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        misc.imsave(save_path+name, cam)
        # evaluate  平均绝对误差
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        # coarse score
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae))
        img_count += 1

print("\n[Congratulations! Testing Done]")
