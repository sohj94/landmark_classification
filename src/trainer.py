import pandas as pd
import numpy as np
import torch
from torch import nn
import time

class Trainer:
	def __init__(self, model, criterion, optimizer, args):
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.args = args

	def fit(self, train_data, last_epoch=0):
		self.model.train()
		for epoch in range(self.args.epochs):
			epoch_loss = 0.
			for iter, (image, label) in enumerate(train_data) :
				# tmp = time.time()
				pred = self.model(image)
				loss = self.criterion(input=pred, target=label)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				epoch_loss += loss.detach().item()
				# print("Ran in {} seconds".format(time.time() - tmp))
				print('epoch : {0} step : [{1}/{2}] loss : {3}'.format(epoch+last_epoch, iter, len(train_data), loss.detach().item()))
			epoch_loss /= len(train_data)
			print('\nepoch : {0} epoch loss : {1}\n'.format(epoch, epoch_loss))

			torch.save(self.model.state_dict(), self.args.model_dir + "epoch_{0:03}.pth".format(epoch))

	def test(self, test_data):
		self.model.eval()
		submission = pd.read_csv(self.args.test_csv_dir)
		for iter, (image, label) in enumerate(test_data):
			pred = self.model(image)
			pred = nn.Softmax(dim=0)(pred)
			pred = pred.detach().cpu().numpy()
			landmark_id = np.argmax(pred)
			confidence = pred[landmark_id]
			print(confidence, landmark_id)
			submission.loc[iter, 'landmark_id'] = landmark_id
			submission.loc[iter, 'conf'] = confidence
		submission.to_csv(self.args.test_csv_submission_dir, index=False)