# Core libraries
import argparse, sys, os
import numpy as np

# Local libraries
from utilities.utils import Utilities
from models.embeddings import resnet50
from utilities.utils_test import inferEmbedding_from_folder, makedirs


def evaluateModel_allpkl(args, num_classes=172, softmax_enabled=1, ckpt_path ='', save_name=''):
    # Define our embeddings model
    model = resnet50(	pretrained=True,
                        num_classes=num_classes,
                        ckpt_path=ckpt_path,
                        embedding_size=args.embedding_size,
                        img_type=args.img_type,
                        softmax_enabled=softmax_enabled	)
    # Put the model on the GPU and in evaluation mode
    if not os.path.exists("/home/io18230/Desktop"):
        model.cuda()
    model.eval()


    save_path = os.path.join(args.save_path, save_name)
    makedirs(save_path)
    print('\nSave embeddings to {}'.format(save_path))
    if args.test_folder_class == 3: # we only need embeddings
        f_embeddings, _, name = inferEmbedding_from_folder(args, model)

    np.savez(save_path+"/folder_embeddings.npz", embeddings=f_embeddings, path=name, labels_knn=C_KNNGT, labels_folder = RealGT)
    # Path to statistics file
    sys.stdout.flush()


# Main/entry method
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--img_type', type=str, default="RGB",
                        help="Which image type should we retrieve: [RGB, D, RGBD]")
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--embedding_size', nargs='?', type=int, default=128) ##2 128
    parser.add_argument('--class_num', type=int,
                        default=156)#)40)  # need to fill in one that is the same as the number of trainng folders if using softmax.

    # flexible
    parser.add_argument('--save_path', type=str, default='/home/io18230/Desktop/output/',
                        help="Where to store the embeddings and statistics")
    parser.add_argument('--test_dir', nargs='?', default="/home/io18230/Desktop/sub_will") #RGBDCows2020_val23small/Identification/RGB
    parser.add_argument('--test_folder_class', type=int, default=2, help ='2 or 3')                            # 171            167
    parser.add_argument('--date', default=[1,1,200,200], nargs = '+', type = int, help='start month, day, end ')  #2,5,2,13   2, 14, 2, 24  1, 1, 12, 30    2, 20, 2, 24
                                                                                    # I_train  I_train_today current_model_state.pkl
    parser.add_argument('--model_path', nargs='?', default='/home/io18230/Desktop/output/trained/') # current_model_state best
    args = parser.parse_args()

    # save path
    date_folder = str(args.date[0]) + '-' + str(args.date[1]) + '_' + str(args.date[2]) + '-' + str(args.date[3])
    tmp = args.save_path
    args.save_path =  os.path.join(tmp, date_folder)

    # Load will's label
    Replace_label_path = os.path.join(args.save_path,'1Will','folder_embeddings.npz') #label.csv
    W_embeddings = np.load(Replace_label_path)
    RealGT = W_embeddings['labels_folder']
    C_KNNGT = W_embeddings['labels_correct_knn']   #KNNGT  = W_embeddings['labels_knn']
    # evaluate a specific pkl model or all models in a folder
    print(f'Inferring embeddings and labels from folder: {args.test_dir}, note the pkl file must be in a folder')
    file1 = os.listdir(args.model_path)
    for item in file1:
        file_name, file_extension = os.path.splitext(item)
        if file_extension == '.pkl':
            if args.embedding_size == 128:
                args.class_num = args.class_num #128d =1000
            elif args.embedding_size ==2:
                args.class_num = 500  #2d == 500
            evaluateModel_allpkl(args,
                                 num_classes=args.class_num,
                                 ckpt_path=os.path.join(args.model_path, item),
                                 save_name= '2combined/'+ file_name)
    print('Done')
    sys.exit(0)
