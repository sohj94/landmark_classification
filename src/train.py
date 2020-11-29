import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from torch.utils.data import DataLoader

from dataset_builder import TrainDataset, TestDataset

parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', dest='train_dir', default="../data/public/train/")
parser.add_argument('--train_csv_dir', dest='train_csv_dir', default="../data/public/train.csv")
parser.add_argument('--train_csv_exist_dir', dest='train_csv_exist_dir', default="../data/public/train_exist.csv")

parser.add_argument('--test_dir', dest='test_dir', default="../data/public/test/")
parser.add_argument('--test_csv_dir', dest='test_csv_dir', default="../data/public/sample_submission.csv")
parser.add_argument('--test_csv_exist_dir', dest='test_csv_exist_dir', default="../data/public/sample_submission_exist.csv")

parser.add_argument('--test_csv_submission_dir', dest='test_csv_submission_dir', default="../data/public/my_submission.csv")
parser.add_argument('--model_dir', dest='model_dir', default="../data/ckpt/")

parser.add_argument('--image_size', dest='image_size', type=int, default=256)
parser.add_argument('--epochs', dest='epochs', type=int, default=100)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--wd', dest='wd', type=float, default=1e-5)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)

parser.add_argument('--train', dest='train', type=bool, default=True)
parser.add_argument('--load_epoch', dest='load_epoch', type=int, default=29)

args = parser.parse_args()

# 경로 생성
if not os.path.isdir(args.model_dir) :
    os.makedirs(args.model_dir)

# Dataset, Dataloader 정의
train_dataset = TrainDataset(args)
test_dataset = TestDataset(args)
