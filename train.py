#!/usr/bin/env Python
# coding=utf-8
# Core libraries
import os
import numpy as np
import sys
import argparse
from tqdm import tqdm

# PyTorch stuff
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

# Local libraries
from utilities.loss import *
from utilities.mining_utils import *
from utilities.utils import Utilities, Augment


"""
File is for training the network via cross fold validation
"""

# Let's cross validate
def crossValidate(args):
	# Sort out which fold to end with
	end_fold = args.end_fold+1 if args.end_fold >= 0 else args.num_folds

	# Store fold accuracies
	fold_accuracies = {}

	# Loop through each fold for cross validation
	for k in range(args.start_fold, end_fold):
		print(f"Beginning training for fold {k+1} of {end_fold}")

		# Directory for storing data to do with this fold
		args.fold_out_path = os.path.join(args.out_path, f"fold_{k}")

		# Create a folder in the results folder for this fold as well as to store embeddings
		os.makedirs(args.fold_out_path, exist_ok=True)

		# Store the current fold
		args.current_fold = k

		# Let's train!
		best_accuracy = trainFold(args)

		# Store this accuracy
		fold_accuracies[k] = best_accuracy

		# Report the accuracies for each fold
		for fold, accuracy in fold_accuracies.items():
			print(f"Fold[{fold}] accuracy = {accuracy}")

# Train for a single fold
def trainFold(args):
	# Create a new instance of the utilities class for this fold
	utils = Utilities(args)
	print('Learning_rate:',args.learning_rate)
	print('Augmentation: ', end='')
	if args.augment:
		pass
	else:
		print(f'NO')
		transform = None
	print(f'\n*****Save to : {args.out_path}*****')
	# Let's prepare the objects we need for training based on command line arguments
	data_loader, model, loss_fn, optimiser = utils.setupForTraining(args, split='train', transform=transform)
	if args.valLoss:
		t_data_loader, _, _, _ = utils.setupForTraining(args=args, split='valid')  ###valid
	# Training tracking variables
	global_step = 0 
	accuracy_best = 0
	valLoss_All_mean = []

	print('Load trained weight :', end='')
	if args.trainedWeight !='defultp':
		print(f'from {args.trainedWeight}')
		weights_init = torch.load(args.trainedWeight, map_location='cpu')['model_state']
		model.load_state_dict(weights_init)
	else:
		print(f'No')

	# Main training loop
	for epoch in tqdm(range(args.num_epochs), desc="Training epochs"):
		# Mini-batch training loop over the training set
		for images, images_pos, images_neg, labels, labels_neg, sub_p, sub_n, random_mark, soft_label in data_loader:
		###for images, images_pos, images_neg, labels, labels_neg in data_loader:
			# Put the images on the GPU and express them as PyTorch variables
			if not os.path.exists("/home/io18230/Desktop"):
				images = Variable(images.cuda())
				images_pos = Variable(images_pos.cuda())
				images_neg = Variable(images_neg.cuda())

			# Zero the optimiser
			model.train()
			optimiser.zero_grad()

			if "Softmax" in args.loss_function:
				# Get the embeddings/predictions for each
				embed_anch, embed_pos, embed_neg, preds = model(images, images_pos, images_neg)
				# Calculate the loss on this minibatch
				loss, triplet_loss, loss_softmax = loss_fn(embed_anch, embed_pos, embed_neg, preds, labels, sub_p, labels_neg, sub_n, soft_label)
			else:
				embed_anch, embed_pos, embed_neg = model(images, images_pos, images_neg)
				# loss = loss_fn(embed_anch, embed_pos, embed_neg, labels) ###
				loss = loss_fn(embed_anch, embed_pos, embed_neg, labels, sub_p, labels_neg, sub_n)

			# Backprop and optimise
			loss.backward()
			optimiser.step()
			global_step += 1

			# Log the loss if its time to do so
			if global_step % args.logs_freq == 0:
				if "Softmax" in args.loss_function:
					utils.logTrainInfo(epoch, global_step, loss.item(),
										loss_triplet=triplet_loss.item(),
										loss_softmax=loss_softmax.item())
				else:
					utils.logTrainInfo(epoch, global_step, loss.item())


		# Every x epochs, let's evaluate on the validation set
		if epoch % args.eval_freq == 0:
			# log validation loss
			if args.valLoss:
				with torch.no_grad():
					val_losses = []
					for images, images_pos, images_neg, labels, labels_neg, sub_p, sub_n, random_mark in t_data_loader:
						if not os.path.exists("/home/io18230/Desktop"):
							# Put the images on the GPU and express them as PyTorch variables
							images = Variable(images.cuda())
							images_pos = Variable(images_pos.cuda())
							images_neg = Variable(images_neg.cuda())
						model.eval()
						embed_anch, embed_pos, embed_neg = model(images, images_pos, images_neg)
						val_loss = loss_fn(embed_anch, embed_pos, embed_neg, labels, sub_p, labels_neg, sub_n)
						val_losses.append(val_loss.item())
					valLoss_All_mean.append(round(np.mean(val_losses), 4))
					np.savez(os.path.join(args.fold_out_path, "valLoss_All_mean.npz"),
							 valLoss_All_mean=valLoss_All_mean)

			# acc
			# Temporarily save model weights for the evaluation to use
			if args.save_best_or_each:
				#utils.saveCheckpoint(epoch, model, optimiser, "current{}".format(epoch))
				utils.saveCheckpoint(epoch, model, optimiser, "c{}".format("%03d" % epoch))
			else:
				utils.saveCheckpoint(epoch, model, optimiser, "c")

			# Test on the validation set
			if args.run_acc:
				accuracy_curr = utils.test(global_step)

				# Save the model weights as the best if it surpasses the previous best results
				if accuracy_curr > accuracy_best:
					utils.saveCheckpoint(epoch, model, optimiser, "best")
					accuracy_best = accuracy_curr

	return accuracy_best

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Parameters for network training')

	# File configuration (the only required arguments)
	parser.add_argument('--out_path', type=str, default="/home/io18230/Desktop/train_tmp",
						help="Path to folder to store results in")

	# Fold settings
	parser.add_argument('--num_folds', type=int, default=10,
						help="Total of folds to cross validate across")
	parser.add_argument('--start_fold', type=int, default=2, # 0
						help="The fold number to START at")
	parser.add_argument('--end_fold', type=int, default=2, # -1
						help="The fold number to END at")

	# Dataset settings
	parser.add_argument('--dataset', type=str, default='RGBDCows2020',
						help='Which dataset to use: [RGBDCows2020, OpenSetCows2020]')
	parser.add_argument('--img_type', type=str, default='RGB',
						help='Which image type to retrieve from RGBDCows2020: [RGB, D, RGBD]')
	parser.add_argument('--exclude_difficult', type=int, default=0,
						help='Whether to exclude difficult categories from the dataset')

	# Model settings (e.g. loss, mining strategy)
	parser.add_argument('--model', type=str, default='TripletResnetSoftmax',
						help='Which model to use: [TripletResnetSoftmax, TripletResnet]')
	parser.add_argument('--triplet_selection', type=str, default='HardestNegative',
						help='Which triplet selection method to use: [PositiveNegative, HardestNegative, RandomNegative,\
						SemihardNegative, AllTriplets]')
	parser.add_argument('--loss_function', type=str, default='OnlineReciprocalSoftmaxLoss',
						help='Which loss function to use: [TripletLoss, TripletSoftmaxLoss, \
						OnlineTripletLoss, OnlineTripletSoftmaxLoss, OnlineReciprocalTripletLoss, \
						OnlineReciprocalSoftmaxLoss]')

	# Model Hyperparameters
	parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
						help='Height of the input image')
	parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
						help='Height of the input image')
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128, # 128
						help='dense layer size for inference')
	parser.add_argument('--batch_size', nargs='?', type=int, default=16,
						help='Batch Size')
	parser.add_argument('--learning_rate', type=float, default=0.001, #0.001
						help="Optimiser learning rate")
	parser.add_argument('--weight_decay', type=float, default=1e-4,
						help="Weight decay")
	parser.add_argument('--triplet_margin', type=float, default=2,
						help="Margin parameter for triplet loss")
	parser.add_argument('--Positive_margin', type=float, default=0, ###
						help="Margin parameter for triplet loss")
	parser.add_argument('--lambda_factor', type=float, default=0.01)

	# Training settings
	parser.add_argument('--num_epochs', nargs='?', type=int, default=100, 
						help='# of the epochs to train for')
	parser.add_argument('--eval_freq', nargs='?', type=int, default=1,
						help='Frequency for evaluating model [epochs num]')
	parser.add_argument('--save_best_or_each', nargs='?', type=int, default=1,
						help='1: each epoch')
	parser.add_argument('--logs_freq', nargs='?', type=int, default=20,
						help='Frequency for saving logs [steps num]')
	parser.add_argument('--run_acc', type=int, default=0)
	parser.add_argument('--valLoss', type=int, default=0)
	parser.add_argument('--trainedWeight', type=str, default='defultp') #   /home/io18230/Desktop/May/current050_model_state.pkl  defultp
	parser.add_argument('--augment', type=int, default=0)
	args = parser.parse_args()


	crossValidate(args)