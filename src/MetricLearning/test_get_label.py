# Core libraries
import os
import sys
import pandas as pd
import argparse
import numpy as np

# Local libraries
from models.embeddings import resnet50
from utilities.utils_test import inferEmbedding_from_folder, inferEmbedding_from_folder_3, makedirs

"""
File for inferring the embeddings of the test portion of a selected database and
evaluating its classification performance using KNN
"""

def evaluateModel(args, num_classes=186, softmax_enabled=1):
    """
    For a trained model, let's evaluate its performance
    """
    # Define our embeddings model
    model = resnet50(	pretrained=True,
                        num_classes=num_classes,
                        ckpt_path=model_path,
                        embedding_size=args.embedding_size,
                        img_type=args.img_type,
                        softmax_enabled=softmax_enabled	)

    # Put the model on the GPU and in evaluation mode
    if not os.path.exists("/home/io18230/Desktop"):
        model.cuda()
    model.eval()


    print('Inferring embeddings and labels of the test folder')
    if args.test_folder_class == 2:
        f_labels, name = inferEmbedding_from_folder(args, model)

    dateinfo = str(args.date[0]) + '-' + str(args.date[1]) + '_' + str(args.date[2]) + '-' + str(args.date[3])
    makedirs(os.path.join(args.save_path, dateinfo, '1Will'))
    save_path = os.path.join(args.save_path, dateinfo,'1Will', f"folder_embeddings.npz")
    np.savez(save_path, labels_folder= f_labels, path = name) #numpy array

    print(f'Saving to {save_path}.\n********************************************\n')
    print('Total images:', len(name))

    # Path to statistics file
    sys.stdout.flush()
    sys.exit(0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--img_type', type=str, default="RGB",
                        help="Which image type should we retrieve: [RGB, D, RGBD]")
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--embedding_size', nargs='?', type=int, default=128)
    parser.add_argument('--save_path', type=str, default='/home/io18230/Desktop/output',
                        help="Where to store the embeddings and statistics")
    parser.add_argument('--test_dir', nargs='?', default="/home/io18230/Desktop/sub_will") #RGB2  sub_will
    parser.add_argument('--test_folder_class', type=int, default=2, help ='2 or 3')
    parser.add_argument('--date', default=[1, 1, 200, 200], nargs = '+', type = int, help='start month, day, end ')
    args = parser.parse_args()

    if os.path.exists("/home/io18230/Desktop"):  # Home machine
        pt = '/home/io18230/Desktop/temp/code/ATI-Pilot-Project-masterfriday/ATI-Pilot-Project-master/src/Identifier/MetricLearning/models'
        model_path = pt + '/best_model_state.pkl'  # what will trained.
    else:
        pt = '/user/work'                  # BP
        model_path = pt + '/io18230/Projects/ATI-workspace/Embeddings/will_trained_186/best_model_state.pkl'
    evaluateModel(args)

    # defult test_folder_class = 2