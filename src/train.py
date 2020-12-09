import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from dataset_builder import TrainDataset, TestDataset
from trainer import Trainer

# import model
from model.three_layer_conv_net import three_layer_conv_net
from model.alexnet import alexnet
from model.googlenet import GoogLeNet

parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', dest='train_dir', default="../data/public/train/")
parser.add_argument('--train_csv_dir', dest='train_csv_dir', default="../data/public/train.csv")
parser.add_argument('--train_csv_exist_dir', dest='train_csv_exist_dir', default="../data/public/train_exist.csv")

parser.add_argument('--test_dir', dest='test_dir', default="../data/public/test/")
parser.add_argument('--test_csv_dir', dest='test_csv_dir', default="../data/public/sample_submission.csv")
parser.add_argument('--test_csv_exist_dir', dest='test_csv_exist_dir', default="../data/public/sample_submission_exist.csv")

parser.add_argument('--test_csv_submission_dir', dest='test_csv_submission_dir', default="../data/public/my_submission.csv")
parser.add_argument('--model_dir', dest='model_dir', default="../data/ckpt/temp/")

parser.add_argument('--image_size', dest='image_size', type=int, default=256)
parser.add_argument('--epochs', dest='epochs', type=int, default=100)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--wd', dest='wd', type=float, default=1e-5)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)

parser.add_argument('--train', dest='train', action='store_false', default=True)
parser.add_argument('--continue_train', dest='continue_train', action='store_true', default=False)
parser.add_argument('--load_epoch', dest='load_epoch', type=int, default=29)

args = parser.parse_args()

# 경로 생성
if not os.path.isdir(args.model_dir) :
    os.makedirs(args.model_dir)

# DataLoader 생성을 위한 collate_fn
def collate_fn(batch) :
	image = [x['image'] for x in batch]
	label = [x['label'] for x in batch]

	return torch.tensor(image).float().cuda(), torch.tensor(label).long().cuda()

def collate_fn_test(batch) :
    image = [x['image'] for x in batch]
    label = [x['label'] for x in batch]

    return torch.tensor(image).float().cuda(), label

# writer = SummaryWriter('runs/alexnet')

# Dataset, Dataloader 정의
train_dataset = TrainDataset(args)
test_dataset = TestDataset(args)
train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_test)

# model = three_layer_conv_net()
# model = alexnet()
model = GoogLeNet(num_classes=1049, aux_logits=False, init_weights=True)
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.wd)

# set trainer
trainer = Trainer(model, criterion, optimizer, args)

#train
if args.train:
	if args.continue_train :
		last_epoch = int(os.listdir(args.model_dir)[-1].split('epoch_')[1][:3])
		model.load_state_dict(torch.load(args.model_dir + "epoch_{0:03}.pth".format(last_epoch)))
		# 그 다음 epoch부터 학습 시작
		trainer.fit(train_data, last_epoch+1)
	else :
		trainer.fit(train_data)
else:
	model.load_state_dict(torch.load(args.model_dir + "epoch_{0:03}.pth".format(args.load_epoch)))

trainer.test(test_data)