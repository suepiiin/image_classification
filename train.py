'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import sys
import io
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from common.MyDataSet import MyDataset
from models.samplenet import SampleNet, SimpleNet

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../config")

DISCRETIZATION = 3


def main():
	# Parse arguments.
	args = parse_args()
	
	# Set device.
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Prepare dataset.
	np.random.seed(seed=0)
	#"""
	image_dataframe = pd.read_csv(args.data_csv + "train_test.csv", engine='python', header=None)
	image_dataframe = image_dataframe.reindex(np.random.permutation(image_dataframe.index))
	test_num = int(len(image_dataframe) * 0.2)
	train_dataframe = image_dataframe[test_num:]
	test_dataframe = image_dataframe[:test_num]
	"""
	image_dataframe = pd.read_csv(args.data_csv + "train.csv", engine='python', header=None)
	train_dataframe = image_dataframe
	image_dataframe = pd.read_csv(args.data_csv + "test.csv", engine='python', header=None)
	test_dataframe = image_dataframe
	"""
	transform = transforms.Compose([
		#transforms.Resize(100),
		transforms.ToTensor()
	])

	train_data = MyDataset(train_dataframe, transform=transform)
	test_data = MyDataset(test_dataframe, transform=transform)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
	
	print('data set')
	# Set a model.
	if args.model == 'resnet18':
		model = models.resnet18()
		model.fc = torch.nn.Linear(512, DISCRETIZATION)
	elif args.model == 'simplenet':
		model = SimpleNet(DISCRETIZATION)
	else:
		raise NotImplementedError()
	model.train()
	model = model.to(device)

	# Set loss function and optimization function.
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	
	# Train and test.
	print('Train starts')
	for epoch in range(args.n_epoch):
		# Train and test a model.
		train_acc, train_loss = train(model, device, train_loader, criterion, optimizer)
		
		# Output score.
                # Save a model checkpoint.
		if(epoch%args.test_interval == 0 or epoch+1 == args.n_epoch):
			test_acc, test_loss = test(model, device, test_loader, criterion)

			stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
			print(stdout_temp.format(epoch+1, train_acc, train_loss, test_acc, test_loss))

			model_ckpt_path = args.model_ckpt_path_temp.format(epoch+1, test_acc)
			torch.save(model.state_dict(), model_ckpt_path)
			print('Saved a model checkpoint at {}'.format(model_ckpt_path))
			print('')

		else:	
			stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}' #, test acc: {:<8}, test loss: {:<8}'
			print(stdout_temp.format(epoch+1, train_acc, train_loss)) #, test_acc, test_loss))



def train(model, device, train_loader, criterion, optimizer):
	model.train()

	output_list = []
	target_list = []
	running_loss = 0.0
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		# Forward processing.
		inputs, targets = inputs.to(device), targets.to(device)
		outputs = model(inputs)
		loss = criterion(outputs, targets)

		# Backward processing.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Set data to calculate score.
		output_list += [int(o.argmax()) for o in outputs]
		target_list += [int(t) for t in targets]
		running_loss += loss.item()

		# Calculate score at present.
		train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)
		if batch_idx % 10 == 0 and batch_idx != 0:
			stdout_temp = 'batch: {:>3}/{:<3}, train acc: {:<8}, train loss: {:<8}'
			print(stdout_temp.format(batch_idx, len(train_loader), train_acc, train_loss))

	# Calculate score.
	train_acc, train_loss = calc_score(output_list, target_list, running_loss, train_loader)

	return train_acc, train_loss


def test(model, device, test_loader, criterion):
	model.eval()

	output_list = []
	target_list = []
	running_loss = 0.0
	for batch_idx, (inputs, targets) in enumerate(test_loader):
		# Forward processing.
		inputs, targets = inputs.to(device), targets.to(device)
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		
		# Set data to calculate score.
		output_list += [int(o.argmax()) for o in outputs]
		target_list += [int(t) for t in targets]
		running_loss += loss.item()
		
	test_acc, test_loss = calc_score(output_list, target_list, running_loss, test_loader)

	#print('confusion_matrix')
	#print(confusion_matrix(output_list, target_list))
	#print('classification_report')
	#print(classification_report(output_list, target_list))

	return test_acc, test_loss



def calc_score(output_list, target_list, running_loss, data_loader):
	acc = round(f1_score(output_list, target_list, average='micro'), 6)
	loss = round(running_loss / len(data_loader.dataset), 6)

	return acc, loss


def parse_args():
	# Set arguments.
	arg_parser = argparse.ArgumentParser(description="Image Classification")
	
	arg_parser.add_argument("--data_csv", type=str, default='/datas/path/')
	arg_parser.add_argument("--model", type=str, default='simplenet')
	arg_parser.add_argument("--model_ckpt_dir", type=str, default='output/checkpoints/')
	arg_parser.add_argument("--model_ckpt_path_temp", type=str, default='output/checkpoints/epoch{}_acc{:.3f}.pth')
	arg_parser.add_argument('--n_epoch', default=10, type=int, help='The number of epoch')
	arg_parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
	arg_parser.add_argument('--test_interval', default=2, type=int, help='test interval')

	args = arg_parser.parse_args()

	# Make directory.
	os.makedirs(args.model_ckpt_dir, exist_ok=True)

	print(args.data_csv)
	# Validate paths.
	assert os.path.exists(args.data_csv)
	assert os.path.exists(args.model_ckpt_dir)

	return args


if __name__ == "__main__":
	main()
	print("finished successfully.")
	os._exit(0)
