import socket
from datetime import datetime
import os
import GPUtil
import setting_2 as fst
import shutil
import utils as ut
import numpy as np
import time

""" experiment description """
exp_date = str(datetime.today().year) + '%02d'%datetime.today().month + '%02d'% datetime.today().day
# exp_name = '/Exp_1'
exp_title = '/tmp'

# n_list = ['/exp', '/baseline', '/L_bagNet57_attn']
# exp_name = ''.join(n_list)


"""GPU connection"""
# devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
# os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
devices = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = devices

f = open('./gpu_num.txt', 'w')
tmp = '{}_{}'.format(socket.gethostname(), devices)
print(tmp)
f.write(tmp)
f.close()

""" data """
data_type_num = 2 ## 0, 5
list_data_type = ['Density',
                  'ADNI_JSY',
                  'ADNI_Jacob_64',          # 2
                  'ADNI_Jacob_256',         # 3
                  'ADNI_AAL_256',           # 4
                  'ADNI_AAL_256_2',         # 5
                  'ADNI_AAL_256_3',         # 6
                  'ADNI_AAL_256_4'          # 7
                  ]

""" data normalization """
data_norm_type_num = 0
list_data_norm_type = ['woNorm', 'minmax']


""" task selection """
list_class_type = ['NC', 'MCI', 'AD', 'sMCI', 'pMCI']

# list_class_for_train = [1, 0, 0, 0, 0] # NC
# list_class_for_test = [1, 0, 0, 0, 0] #

# list_class_for_train = [0, 1, 1, 0, 0] #
# list_class_for_test = [0, 1, 1, 0, 0] #

list_class_for_train = [1, 0, 1, 0, 0] # NC AD
list_class_for_test = [1, 0, 1, 0, 0] #

# list_class_for_train = [0, 0, 0, 1, 1] # sMCI pMCI
# list_class_for_test = [0, 0, 0, 1, 1] #


list_class_for_total = [1, 1, 1, 1, 1] # for plotting
list_eval_metric = ['MAE', 'RMSE', 'R_squared',  'Acc', 'Sen', 'Spe', 'AUC']
# list_class_for_train = [1, 0, 1] # for train and eval
# list_class_for_total = [1, 1, 1] # for plotting
# list_class_type = ['NC', 'MCI', 'AD']

list_age_estimation_factor_depending_on_disease = [1, 1, 1, 0, 0]
list_train_loss_factor_depending_on_disease = [1, 0, 0, 0, 0]

# list_lambdas_for_each_lbl_at_loss = [1, 0, 0, 0, 0]
# list_lambdas_for_each_lbl_at_loss = [1, 1, 1, 0, 0]
# list_lambdas_for_each_lbl_at_loss_for_MAE = [1, 0, 0, 0, 0]

""" selected task """
list_selected_for_train = [] # ['NC', 'MCI', 'AD']
list_selected_for_test = [] # ['NC', 'MCI', 'AD']
list_selected_for_total = [] # ['NC', 'MCI', 'AD']
list_selected_lambdas_at_age = [] # [1, 0.5]
list_selected_lambdas_at_loss = [] # [1. 0.5]
# list_selected_lambdas_at_loss_for_MAE = [] # [1. 0.5]

for i in range(len(list_class_for_total)):
    if list_class_for_total[i] == 1:
        list_selected_for_total.append(list_class_type[i])
        list_selected_lambdas_at_age.append(list_age_estimation_factor_depending_on_disease[i])
        list_selected_lambdas_at_loss.append(list_train_loss_factor_depending_on_disease[i])
        # list_selected_lambdas_at_loss_for_MAE.append(list_lambdas_for_each_lbl_at_loss_for_MAE[i])
    if list_class_for_train[i] == 1:
        list_selected_for_train.append(list_class_type[i])
    if list_class_for_test[i] == 1:
        list_selected_for_test.append(list_class_type[i])

""" eval metric """
if fst.flag_regression == True:
    list_standard_eval_dir = ['/val_loss']
else:
    # list_standard_eval_dir = ['/val_loss', '/val_acc', '/val_auc']
    # list_standard_eval_dir = ['/val_loss']
    list_standard_eval_dir = ['/val_auc']

# list_standard_eval_dir = ['/val_loss', '/val_mean_loss', '/val_acc', '/val_auc']

list_standard_eval = ['{}'.format(list_standard_eval_dir[i][1:]) for i in range(len(list_standard_eval_dir))]


""" parmas """
epoch = 200
lr = 5e-4
LR_decay_rate = 0.98
step_size = 1

focal_gamma = 0
focal_alpha = None

weight_decay = 0.0005
batch_size = 6

early_stopping_start_epoch = 1
early_stopping_patience = epoch
# early_stopping_mean_loss_delta = 1.0

kfold = 10
start_fold = 1
# end_fold = 5
# start_fold = 6
end_fold = 10

#TODO: should be '1' , because of calculation of loss and the total # of the dataset
v_batch_size = 1
t_batch_size = 1

""" root dir """
if(socket.gethostname() == "DESKTOP-C48ASA0"): # windows (local)
    dir_root = 'D:/data/ADNI'
else: #ubuntu (server)
    # dir_root = '/Data/chpark'
    dir_root = '/Datafast/chpark'

""" model setting """
model_arch_dir = "/model_arch"


model_num_0 = 85
# model_num_0 = 64
# model_num_1 = 17
# model_num_2 = 80
# dir_preTrain_1 = './resnet50_age_revGrad/val_loss'
# dir_preTrain_2 = './resnet50_revGrad/val_loss'
# dir_preTrain_1 = dir_root + '/pretrained/200421/L_bagNet33_attn/L_bagNet33_with_attention_1/val_loss'
# dir_mask_img = dir_root + '/mask_info/1_NC_AD/AD_logit_mean_std/mean_std/fold_1/featuremap/std_NC_AD.nii.gz'
# dir_mask_img = dir_root + '/mask_info/2_sMCI_pMCI/mean_std/fold_1/featuremap/std_sMCI_pMCI.nii.gz'

model_name = [None] * 100

model_name[12] = "L_bagNet9"
model_name[13] = "L_bagNet17"
model_name[14] = "L_bagNet33"
model_name[15] = "L_bagNet9_with_attention_1"
model_name[16] = "L_bagNet17_with_attention_1"
model_name[17] = "L_bagNet33_with_attention_1"
model_name[85] = "L_bagNet33_multi_scale"

dir_to_save_1 = './' + model_name[model_num_0]
# dir_to_save_1 = './' + socket.gethostname() +'_'+devices + '/' + model_name[model_num_0]

""" age estimation function selection """
selected_function = 5
if fst.flag_estimate_age == True:
    list_age_estimating_function = [None] * 12
    list_age_estimating_function[0] = 'none'
    list_age_estimating_function[1] = 'linear_1'
    list_age_estimating_function[2] = 'linear_1_with_age'
    list_age_estimating_function[3] = 'sigmoid_1'
    list_age_estimating_function[4] = 'sigmoid_1_with_age'
    list_age_estimating_function[5] = 'sqrt_1'
    list_age_estimating_function[6] = 'quadratic_1'
    list_age_estimating_function[7] = 'constant'
    print("estimation function to use : {}".format(list_age_estimating_function[selected_function]))


""" path setting """
if(socket.gethostname() == "DESKTOP-C48ASA0"): # windows (local)
    tmp_data_path = '/' + list_data_type[data_type_num]
    if tmp_data_path == "/Density":
        orig_data_dir = dir_root + '/ADNI/Whole_data'
        data_size = None
        num_modality = 1
    elif tmp_data_path == "/ADNI_AAL_256":
        orig_data_dir = dir_root + '/ADNI_RAVENS_AAL/200312'
        data_size = None
        num_modality = 1
    """ data type path """
    exp_data_dir = dir_root + '/ADNI_exp' + tmp_data_path
    tadpole_dir = dir_root + '/TADPOLE-challenge/TADPOLE_D1_D2.csv'

else: #ubuntu (server)
    """ original data path """
    tmp_data_path = '/' + list_data_type[data_type_num]
    if tmp_data_path == "/Density":
        orig_data_dir = '/Datafast/cwkim/Whole_data'
        data_size = None
        num_modality = 1
    elif tmp_data_path == "/ADNI_JSY":
        orig_data_dir = '/Data/chpark/ADNI_JSY'
        data_size = None
        num_modality = 1
    elif tmp_data_path == "/ADNI_Jacob_64":
        orig_data_dir = '/Data/chpark/ADNI_RAVENS/Resized_Ravens_Map_64'
        data_size = None
        num_modality = 1
    elif tmp_data_path == "/ADNI_Jacob_256":
        orig_data_dir = '/Data/chpark/ADNI_RAVENS/ADNI4Dregistration'
        data_size = None
        num_modality = 3
    elif "/ADNI_AAL_256" in tmp_data_path :
        orig_data_dir = '/Data/chpark/ADNI_AAL'
        template_dir = '/Data/chpark/ADNI_AAL/AAL_Templates/atlas90-seg-256_256_256-removeCere.img'
        RoI_template_dir = '/Data/chpark/ADNI_AAL/AAL_Templates/atlas90-label-256_256_256.img'
        data_size = None
        num_modality = 1

    """ data type path """
    exp_data_dir = dir_root + '/ADNI_exp' + tmp_data_path
    tadpole_dir = dir_root + '/TADPOLE-challenge/TADPOLE_D1_D2.csv'

if list_data_type[data_type_num] == 'Density':
    dir_list = ['/NORMAL', '/MCI', '/AD', '/sMCI', '/pMCI']
    x_range = [0, 121]
    y_range = [0, 145]
    z_range = [0, 121]
    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]
    z_size = z_range[1] - z_range[0]
elif list_data_type[data_type_num] == 'ADNI_JSY':
    x_range = [0, 193]
    y_range = [0, 229]
    z_range = [0, 193]
    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]
    z_size = z_range[1] - z_range[0]
elif list_data_type[data_type_num] == 'ADNI_Jacob_64':
    dir_list = ['/NORMAL', '/MCI', '/AD', '/sMCI', '/pMCI']
    x_range = [0, 64]
    y_range = [0, 64]
    z_range = [0, 64]
    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]
    z_size = z_range[1] - z_range[0]
elif list_data_type[data_type_num] == 'ADNI_Jacob_256':
    dir_list = ['/NORMAL', '/MCI', '/AD', '/sMCI', '/pMCI']
    x_range = [0, 256]
    y_range = [0, 256]
    z_range = [0, 256]
    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]
    z_size = z_range[1] - z_range[0]
elif list_data_type[data_type_num] == 'ADNI_AAL_256':
    dir_list = ['/NORMAL', '/MCI', '/AD', '/sMCI', '/pMCI']
    x_range = [0, 256]
    y_range = [0, 256]
    z_range = [0, 256]
    x_size = x_range[1] - x_range[0] # 154
    y_size = y_range[1] - y_range[0] # 193
    z_size = z_range[1] - z_range[0] # 144

elif list_data_type[data_type_num] == 'ADNI_AAL_256_2':
    # template_minimum_size = [144, 183, 134]
    template_x_range = [58, 201+1]
    template_y_range = [38, 220+1]
    template_z_range = [28, 161+1]
    spare_value = 5

    dir_list = ['/NORMAL', '/MCI', '/AD', '/sMCI', '/pMCI']
    # x_range = [0, 256]
    # y_range = [0, 256]
    # z_range = [0, 256]
    x_range = template_x_range
    y_range = template_y_range
    z_range = template_z_range

    x_range[0] -= spare_value
    x_range[1] += spare_value

    y_range[0] -= spare_value
    y_range[1] += spare_value

    z_range[0] -= spare_value
    z_range[1] += spare_value

    x_size = x_range[1] - x_range[0] # 154
    y_size = y_range[1] - y_range[0] # 193
    z_size = z_range[1] - z_range[0] # 144

elif list_data_type[data_type_num] == 'ADNI_AAL_256_3':
    # (52, 204), (37, 221), (25, 175)
    # (174, 213, 164)
    template_x_range = [58, 201+1]
    template_y_range = [38, 220+1]
    template_z_range = [28, 161+1]
    spare_value = 15

    dir_list = ['/NORMAL', '/MCI', '/AD', '/sMCI', '/pMCI']
    # x_range = [0, 256]
    # y_range = [0, 256]
    # z_range = [0, 256]
    x_range = template_x_range
    y_range = template_y_range
    z_range = template_z_range

    x_range[0] -= spare_value
    x_range[1] += spare_value

    y_range[0] -= spare_value
    y_range[1] += spare_value

    z_range[0] -= spare_value
    z_range[1] += spare_value

    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]
    z_size = z_range[1] - z_range[0]

elif list_data_type[data_type_num] == 'ADNI_AAL_256_4':
    # (52, 204), (37, 221), (25, 175)
    # (174, 213, 164)
    template_x_range = [58, 201+1]
    template_y_range = [38, 220+1]
    template_z_range = [28, 161+1]
    spare_value = 25

    dir_list = ['/NORMAL', '/MCI', '/AD', '/sMCI', '/pMCI']
    # x_range = [0, 256]
    # y_range = [0, 256]
    # z_range = [0, 256]
    x_range = template_x_range
    y_range = template_y_range
    z_range = template_z_range

    x_range[0] -= spare_value
    x_range[1] += spare_value

    y_range[0] -= spare_value
    y_range[1] += spare_value

    z_range[0] -= spare_value
    z_range[1] += spare_value

    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]
    z_size = z_range[1] - z_range[0]

""" 1. raw npy dir """
orig_npy_dir = exp_data_dir + '/orig_npy'
ADNI_fold_image_path = []
ADNI_fold_age_path = []
ADNI_fold_MMSE_path = []
for i in range(len(list_class_type)):
    ADNI_fold_image_path.append(orig_npy_dir + "/ADNI_" + str(list_class_type[i]) + "_image.npy")
    ADNI_fold_age_path.append(orig_npy_dir + "/ADNI_" + str(list_class_type[i]) + "_age.npy")
    ADNI_fold_MMSE_path.append(orig_npy_dir + "/ADNI_" + str(list_class_type[i]) + "_MMSE.npy")

""" 2. fold index """
fold_index_dir = exp_data_dir + '/fold_index'
train_index_dir = []
val_index_dir  = []
test_index_dir = []
for i in range(len(list_class_type)):
    train_index_dir.append(fold_index_dir + '/train_index_' + list_class_type[i])
    val_index_dir.append(fold_index_dir + '/val_index_' + list_class_type[i])
    test_index_dir.append(fold_index_dir + '/test_index_' + list_class_type[i])

""" 3. fold npy dir """
tmp_norm_dir = '/' + list_data_norm_type[data_norm_type_num]
fold_npy_dir = exp_data_dir + '/fold_npy' + tmp_norm_dir
train_fold_dir = []
val_fold_dir = []
test_fold_dir = []
for j in range(1, kfold+1):
    tmp_train_fold_dir = []
    tmp_val_fold_dir = []
    tmp_test_fold_dir = []
    for i in range(len(list_class_type)):
        tmp_train_fold_dir.append('/train_fold_{}_c_{}'.format(j, list_class_type[i]))
        tmp_val_fold_dir.append('/val_fold_{}_c_{}'.format(j, list_class_type[i]))
        tmp_test_fold_dir.append('/test_fold_{}_c_{}'.format(j, list_class_type[i]))
    train_fold_dir.append(tmp_train_fold_dir)
    val_fold_dir.append(tmp_val_fold_dir)
    test_fold_dir.append(tmp_test_fold_dir)
list_data_name = ['image', 'lbl', 'age', 'MMSE']


""" gaussian """
if len(list_selected_for_train) ==2 :
    tmp_dir = '/gaussian'
    gaussian_dir = exp_data_dir + tmp_dir + '/' + '{}_{}'.format(list_selected_for_train[0], list_selected_for_train[1])

""" t_test """
if len(list_selected_for_train) ==2 :
    tmp_dir = '/t_test'
    ttest_dir = exp_data_dir + tmp_dir + '/' + '{}_{}'.format(list_selected_for_train[0], list_selected_for_train[1])

"""openpyxl setting """
push_start_row = 2

""" task """
if fst.flag_classification == True:
    exp_description = "classification"
elif fst.flag_classification_using_pretrained == True:
    exp_description = "classification using model pretrained"
elif fst.flag_classification_using_pretrained_2 == True:
    exp_description = "classification using model pretrained_2"
elif fst.flag_classification_fine_tune == True:
    exp_description = "classification fine tune"
elif fst.flag_regression == True:
    exp_description = "regression"
elif fst.flag_multi_task== True:
    exp_description = "multi-task"

""" print out setting """
print(socket.gethostname())
print("data : {}".format(exp_date))
print("exp name : {}".format(exp_title))
print("discription : {}".format(exp_description))

print(' ')
print("epoch : {}".format(epoch))
print("lr : {}".format(lr))
print("lr_decay : {}".format(LR_decay_rate))
print("batch_size : {}".format(batch_size))

print(' ')
print("Dataset for train : {}".format(list_selected_for_train))
print("model arch : {}".format(model_name[model_num_0]))

print(' ')
print("data type : {}".format(list_data_type[data_type_num]))



if list_data_type[data_type_num] == 'Density'or list_data_type[data_type_num] == 'ADNI_Jacob_128':
    min_crop_size = 105
    max_crop_size = 121
    if 'bagNet33' in model_name[model_num_0]:
        patch_size = 33
        patch_stride = 8
        size_translation = 8
    else:
        size_translation = 8
if list_data_type[data_type_num] == 'ADNI_JSY' or\
        'ADNI_Jacob' in list_data_type[data_type_num] or \
        'ADNI_AAL_256' in list_data_type[data_type_num]:

    max_crop_size = [x_size, y_size, z_size]
    tmp_space_ratio = 0.5
    min_crop_size = [int(x_size * tmp_space_ratio),
                     int(y_size * tmp_space_ratio),
                     int(z_size * tmp_space_ratio)
                     ]


    if 'bagNet9' in model_name[model_num_0]:
        if fst.flag_downSample == True:
            patch_size = 19
            patch_stride = 16
            size_translation = 16
        else:
            patch_size = 9
            patch_stride = 8
            size_translation = 8
    elif 'bagNet17' in model_name[model_num_0] or model_name[model_num_0] == 'low_rank_local_connectivity':
        if fst.flag_downSample == True:
            patch_size = 35
            patch_stride = 16
            size_translation = 16
        else:
            patch_size = 17
            patch_stride = 8
            size_translation = 8
    elif 'bagNet33' in model_name[model_num_0]:
        if fst.flag_downSample == True:
            patch_size = 67
            patch_stride = 16
            size_translation = 16
        if 'spatial_wise' in model_name[model_num_0]:
            patch_size = 33
            patch_stride = 16
            size_translation = 16
        else:
            patch_size = 33
            patch_stride = 8
            size_translation = 8

    elif 'bagNet49' in model_name[model_num_0]:
        patch_size = 49
        patch_stride = 8
        size_translation = 8
    elif 'bagNet57' in model_name[model_num_0]:
        patch_size = 57
        patch_stride = 8
        size_translation = 8
        if 'bagNet57_2' in model_name[model_num_0]:
            patch_size = 57
            patch_stride = 16
            size_translation = 16


    elif model_name[model_num_0] == 'bagNet_baseline':
        patch_size = 33
        patch_stride = 8
        size_translation = 8

    elif model_name[model_num_0] == 'dilation_conv':
        if fst.flag_downSample == True:
            size_translation = 16
        else:
            size_translation = 8

    else :
        size_translation = 8
# start = time.time()
# list_gpu_index = [1] * 8
# list_gpuMemmory_more_than_thresh = [0] * 8
# while(1):
#     if time.time() - start > 10:
#         break
#     threshold = 5000
#     for i in range(8):
#         list_gpuMemmory_more_than_thresh[i] += GPUtil.getGPUs()[i].memoryFree
#         if GPUtil.getGPUs()[i].memoryFree < threshold:
#             list_gpu_index[i] = 0
# np_index = np.array(list_gpu_index)
# np_freeMem = np.array(list_gpuMemmory_more_than_thresh)
# devices = np.argmin(np.where(np_index ==0 , np.max(np_freeMem) + 1000, np_freeMem))
# os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)

