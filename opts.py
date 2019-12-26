import argparse

def opt_ingre_process_vireo():
    #receive inputs for ingredient processing

    parser = argparse.ArgumentParser()

    root_path = '/mnt/FoodRecog/work_lupi/vireo/'

    parser.add_argument('--data_path', type=str, default= root_path + 'SplitAndIngreLabel/',
                    help='path to data files')


    args = parser.parse_args()

    # # Check if args are valid
    # assert args.rnn_size > 0, "rnn_size should be greater than 0"
    #

    return args

def opt_ingre_process_food101():
    #receive inputs for ingredient processing

    parser = argparse.ArgumentParser()

    root_path = '/mnt/FoodRecog/work_lupi/food101/' #'path to root folder'

    parser.add_argument('--data_path', type=str, default= root_path + 'data/',
                    help='path to data files')

    parser.add_argument('--glove_root_path', type=str, default= root_path + '/vireo/SplitAndIngreLabel/',
                    help='path to glove files')

    args = parser.parse_args()

    # # Check if args are valid
    # assert args.rnn_size > 0, "rnn_size should be greater than 0"
    #

    return args


def opt_algorithm():
    #receive algorithm settings

    parser = argparse.ArgumentParser()

    #dataset choose
    dataset = 'food101' #selection from 'vireo' and 'food101'

    if dataset is 'vireo':
        root_path = '/mnt/FoodRecog/work_lupi/vireo/'
        img_path = '/mnt/FoodRecog/vireo172/ready_chinese_food'
        data_path =  root_path + 'SplitAndIngreLabel/'
        food_class_name_path = data_path + 'FoodList.txt'
        dataset_num_class = 172
        dataset_max_seq = 15
        dataset_max_seq_test = 11
        dataset_max_seq_val = 10
        dataset_num_ingre = 353
        dataset_avg_ingre = 3
    else:
        root_path = '/mnt/FoodRecog/work_lupi/food101/'
        img_path = '/mnt/FoodRecog/food101/food-101/images/'
        data_path =  root_path + 'data/'
        food_class_name_path = data_path + 'classes.txt'
        dataset_num_class = 101
        dataset_max_seq = 25
        dataset_max_seq_test = None
        dataset_max_seq_val = None
        dataset_num_ingre = 446
        dataset_avg_ingre = 9

    #environments
    parser.add_argument('--dataset', type=str, default= dataset,
                    help='indicator to dataset')

    parser.add_argument('--root_path', type=str, default= root_path,
                    help='path to root folder')

    parser.add_argument('--img_path', type=str, default= img_path,
                    help='path to image folder')

    parser.add_argument('--data_path', type=str, default= data_path,
                    help='path to data folder')

    parser.add_argument('--result_path', type=str, default= root_path + 'algorithm_results/',
                    help='path to the folder to save results')

    parser.add_argument('--stage1_model_path', type=str, default= root_path + 'stage1_model/',
                    help='path to pretrained model for image channel')

    parser.add_argument('--stage2_model_path', type=str, default= root_path + 'stage2_model/',
                    help='path to pretrained model for ingre channel')

    #data details
    parser.add_argument('--food_class_name_path', type=str, default= food_class_name_path,
                    help='path to the list of names for classes')

    parser.add_argument('--ingre_term_list_path', type=str, default= data_path + 'ingreList.mat',
                    help='path to the list of original ingredient terms')

    parser.add_argument('--ingre_word_list_path', type=str, default= data_path + 'wordList.mat',
                    help='path to the list of our extracted ingredient words')

    parser.add_argument('--dataset_num_class', type=str, default= dataset_num_class,
                    help='number of classes in the dataset')

    parser.add_argument('--dataset_max_seq', type=str, default= dataset_max_seq,
                    help='max number of ingredients for a sample in train data')

    parser.add_argument('--dataset_max_seq_test', type=str, default= dataset_max_seq_test,
                    help='max number of ingredients for a sample in test data')

    parser.add_argument('--dataset_max_seq_val', type=str, default= dataset_max_seq_val,
                    help='max number of ingredients for a sample in val data')

    parser.add_argument('--dataset_num_ingre', type=str, default= dataset_num_ingre,
                    help='number of ingredients in the dataset')

    parser.add_argument('--dataset_avg_ingre', type=str, default=dataset_avg_ingre,
                        help='average number of ingredients to be considered per sample in the dataset. Used to compute ingredient prediction performance')

    #experiment controls
    parser.add_argument('--stage', type=int, default= 3,
                    help='1: pretrain image channel independently; 2: pretrain ingredient channel independently; 3: train the whole network')

    parser.add_argument('--mode', type=str, default= 'test',
                    help='select from train, val, test. Used in dataset creation')

    parser.add_argument('--img_net', type=str, default= 'resnet50',
                    help='choose network backbone for image channel: vgg19bn, resnet50, wrn, wiser')

    parser.add_argument('--ingre_net', type=str, default= 'gru',
                    help='choose network backbone for ingredient channel: gru, nn')

    parser.add_argument('--pretrained_img_net_path', type=str, default= '/mnt/FoodRecog/work_lupi/vireo/codes/' + 'wide-resnet-50-2-export-5ae25d50.pth',
                        help='used in stage 1 to initialize WRN and WISER with imgnet-pretrained model')

    args = parser.parse_args()


    return args