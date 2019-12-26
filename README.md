# Food


Note: Our implementation provided in this repository is patent-banned: ILO Ref: 2019-243-01, SG Non-Provisional Application No. 10201907991T, Singapore, 29 Aug., 2019. Please contact the Author for commercialization usage.  

This repository provides our Pytorch implementation of the cross-modal alignment and transfer network (ATNet) presented in "Learning Using Privileged Information for food recognition". Please cite our paper using the following information if our codes are used in your research:

@inproceedings{meng2019learning,
  title={Learning Using Privileged Information for Food Recognition},\\
  author={Meng, Lei and Chen, Long and Yang, Xun and Tao, Dacheng and Zhang, Hanwang and Miao, Chunyan and Chua, Tat-Seng},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={557--565},
  year={2019},
  organization={ACM}
}


Please follow the steps below to reproduce ATNet for food recognition:

Step 1. Download datasets and related files

1.1. Get vireo172 dataset: obtain vireo dataset from the authors via procedures in http://vireo.cs.cityu.edu.hk/VireoFood172/, including the food images in folder 'ready_chinese_food' and the data info files in 'SplitAndIngreLabel'. Save them to <User_root_path> + '/vireo/'.

1.2. Get glove data: obtain the glove data 'glove.6B.300d.txt' from 'glove.6B.zip' available in https://nlp.stanford.edu/projects/glove/. Save it to the folder 'SplitAndIngreLabel'.

1.3. Get ingredient101 dataset: obtain food101 images from https://www.vision.ee.ethz.ch/datasets_extra/food-101/. Save the 101 image folders to <User_root_path> + '/food-101/images/'. Obtain data splits and ingredient info from http://www.ub.edu/cvub/ingredients101/. Save the txt files to the folder <User_root_path> + '/food-101/data/'.

1.3. Download the Pytorch version of WRN50-2 model pretrained on ImageNet 'wide-resnet-50-2-export-5ae25d50.pth' from https://github.com/szagoruyko/functional-zoo/blob/master/wide-resnet-50-2-export.ipynb. Save it to <User_root_path>.

Step 2. Install necessary environment: Python 3.6, Pytorch 0.41, Cuda 9.1.

Step 3. Process datasets

3.1. Git clone all .py files to <User_root_path>.

3.2. Process vireo172 data: run python ingre_process-vireo.py --data_path <User_root_path> + '/vireo/SplitAndIngreLabel/'. All algorithm inputs and related files will be saved to '/vireo/SplitAndIngreLabel'

3.3. Process ingredient101 data: run python ingre_process-food101.py --data_path <User_root_path> + '/food-101/data/' --glove_root_path <User_root_path> + '/vireo/SplitAndIngreLabel/'. All algorithm inputs and related files will be saved to '/food-101/data/'

Step 4. Training

4.1. Open opt.py for one-time configuration for new environments: Set correct paths for root_path and img_path as defined in opt_algorithm() for both datasets.

4.2. Perform image-channel pretrain: run python train.py --dataset 'vireo' --stage 1 --mode 'train' --img_net 'resnet50' --ingre_net 'gru' to train a variant of the image channel of our model using the vireo172 dataset. More options can be found in opt.py. Hyperparameters can be set in lines 62-92 in train.py. Results are in <User_root_path> + '/vireo/algorithm_results/' - 'train_loss.txt' and 'model_batch_train_loss.txt'. Name the best model as 'model_'+ --img_net + '.pt', and save it to the folder <User_root_path> + '/vireo/stage1_model/' if using vireo dataset.

4.3. Perform ingredient-channel pretrain: run python train.py --dataset 'vireo' --stage 2 --mode 'train' --img_net 'resnet50' --ingre_net 'gru' to train a variant of the ingredient channel of our model using the vireo172 dataset. More options can be found in opt.py. Hyperparameters can be set in lines 62-92 in train.py (especially the loss weights in lines 77-79). Results are in <User_root_path> + '/vireo/algorithm_results/' - 'train_loss.txt' and 'model_batch_train_loss.txt'. Name the best model as 'model_'+ --ingre_net + '.pt', and save it to the folder <User_root_path> + '/vireo/stage2_model/' if using vireo dataset.

4.4. Training the whole model: run python train.py --dataset 'vireo' --stage 3 --mode 'train' --img_net 'resnet50' --ingre_net 'gru' to train a variant of our model using the vireo172 dataset. More options can be found in opt.py. Hyperparameters can be set in lines 62-92 in train.py (especially the loss weights in lines 81-87). Results are in <User_root_path> + '/vireo/algorithm_results/' - 'train_loss.txt' and 'model_batch_train_loss.txt'.

Step 5. Testing

5.1. Testing the whole model: run python test.py --dataset 'vireo' --stage 3 --mode 'test' --img_net 'resnet50' --ingre_net 'gru' to test the varient of our model trained using the vireo172 dataset in Step 4. Hyperparameters can be set in lines 85-97 in test.py. Results are in <User_root_path> + '/vireo/algorithm_results/' - 'test_performance.txt', 'model_batch_test_performance.txt', and 'img2tag.txt'. 
