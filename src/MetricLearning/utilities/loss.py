# Core libraries
import numpy as np

# PyTorch stuff
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
File contains loss functions selectable during training
"""

# Reciprocal triplet loss from
# "Who Goes There? Exploiting Silhouettes and Wearable Signals for Subject Identification
# in Multi-Person Environments"
class OnlineReciprocalTripletLoss(nn.Module):
	def __init__(self, triplet_selector):
		super(OnlineReciprocalTripletLoss, self).__init__()
		self.triplet_selector = triplet_selector

	def forward(self, anchor_embed, pos_embed, neg_embed, labels, sub_p, labels_neg, sub_n):
		# Combine the embeddings from each network
		# embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0) Jing
		embeddings = torch.cat((anchor_embed, neg_embed), dim=0)

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, labels, sub_p, labels_neg, sub_n)

		# There might be no triplets selected, if so, just compute the loss over the entire minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute distances over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)

		# Actually compute reciprocal triplet loss
		#losses = ap_distances + (1 / (an_distances))
		losses = ap_distances + (1/(an_distances+0.001))

		# tmp
		# print(ap_distances)
		# for item in an_distances:
		# 	if item < 0.03:
		# 		print(item, 1/item)
		# 		print(losses.mean())

		return losses.mean()


class OnlineReciprocalSoftmaxLoss(nn.Module):
	def __init__(self, triplet_selector, margin=0.0, lambda_factor=0.01):
		super(OnlineReciprocalSoftmaxLoss, self).__init__()
		self.margin = margin
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor
		self.triplet_selector = triplet_selector

	def forward(self, anchor_embed, pos_embed, neg_embed, preds, labels, sub_p, labels_neg, sub_n, soft_label):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, neg_embed), dim=0)

		# match cross entropy

		# Define the labels as variables and put on the GPU
		#gpu_labels = labels.view(len(labels))
		gpu_labels =     soft_label.view(len(soft_label))
		gpu_labels_neg = labels_neg.view(len(labels_neg))
		preds_TMP =      preds[:len(soft_label)*2,:] # sometimes it is not the batchsieze
		if not os.path.exists("/home/io18230/Desktop"):
			gpu_labels = Variable(gpu_labels.cuda())
			gpu_labels_neg = Variable(gpu_labels_neg.cuda())

		# Concatenate labels for softmax/crossentropy targets
		#target = torch.cat((gpu_labels, gpu_labels, gpu_labels_neg), dim=0)
		target = torch.cat((gpu_labels, gpu_labels), dim=0)

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, labels, sub_p, labels_neg, sub_n)

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)

		# Compute the triplet losses
		triplet_losses = ap_distances + (1 / an_distances)
		#triplet_losses = ap_distances + (1 / (an_distances + 0.001))

		# Compute softmax loss
		loss_softmax = self.loss_fn(input=preds_TMP, target=target)

		# Compute the total loss
		loss_total = self.lambda_factor * triplet_losses.mean() + loss_softmax

		# Return them all!
		return loss_total, triplet_losses.mean(), loss_softmax