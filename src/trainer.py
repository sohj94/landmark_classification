import pandas as pd
import numpy as np
from torch import nn

class Trainer:
	def __init__(self, model, criterion, optimizer, args):
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.args = args

	def fit(self, train_data):
		self.model.train()
		for epoch in range(self.args.epochs):
			epoch_loss = 0.
			for iter, (image, label) in enumerate(train_data) :
				pred = self.model(image)
				loss = self.criterion(input=pred, target=label)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				epoch_loss += loss.detach().item()
				print('epoch : {0} step : [{1}/{2}] loss : {3}'.format(epoch, iter, len(train_data), loss.detach().item()))
			epoch_loss /= len(train_data)
			print('\nepoch : {0} epoch loss : {1}\n'.format(epoch, epoch_loss))

			torch.save(self.model.state_dict(), args.model_dir + "epoch_{0:03}.pth".format(epoch))

	def test(self, test_data):
		model.eval()
		submission = pd.read_csv(args.test_csv_dir)
		for iter, (image, label) in enumerate(test_data):
			pred = model(image)
			pred = nn.Softmax(dim=1)(pred)
			pred = pred.detach().cpu().numpy()
			landmark_id = np.argmax(pred, axis=1)
			confidence = pred[0,landmark_id]
			submission.loc[iter, 'landmark_id'] = landmark_id
			submission.loc[iter, 'conf'] = confidence
		submission.to_csv(args.test_csv_submission_dir, index=False)