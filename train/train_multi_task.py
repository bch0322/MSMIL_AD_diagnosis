import os
import sys
import utils as ut
from torch.autograd import Variable
import torch
import torch.nn as nn
import setting as st
import setting_2 as fst
from data_load import data_load as DL
from scipy import stats
import numpy as np
import nibabel as nib

def train(config, fold, model, optimizer, scheduler, list_dir_save_model, dir_pyplot, Validation=True, Test_flag = True):

    """ loss """
    criterion_cls = ut.FocalLoss(gamma=st.focal_gamma, alpha=st.focal_alpha)
    criterion_MAE = nn.L1Loss()
    # criterion = nn.MSELoss()
    EMS = ut.eval_metric_storage()
    list_selected_EMS = []
    list_ES = []
    for i_tmp in range(len(st.list_standard_eval_dir)):
        list_selected_EMS.append(ut.eval_selected_metirc_storage())
        list_ES.append(ut.EarlyStopping(delta=0, patience=st.early_stopping_patience, verbose=True))

    """ load data """
    list_train_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_train, flag_tr_val_te='train') # image, label, age, MMSE
    list_val_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_test, flag_tr_val_te='val')
    list_test_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_test, flag_tr_val_te='test')

    if fst.flag_gaussian_norm == True:
        if fst.flag_gaussian_norm_init == True:
            """ save dir """
            tmp_save_dir = st.gaussian_dir + '/fold_{}'.format(fold)
            ut.make_dir(dir=tmp_save_dir, flag_rm=False)
            """ normalize """
            list_train_data[0], mu, sigma = ut.Gauss_Norm_subjectWise(list_train_data[0], train=True)
            np.savez(tmp_save_dir + '/mu_sigma', mu=mu, sigma=sigma)
            list_val_data[0] = ut.Gauss_Norm_subjectWise(list_val_data[0], mu, sigma, train=False)
            list_test_data[0] = ut.Gauss_Norm_subjectWise(list_test_data[0], mu, sigma, train=False)
        else:
            tmp_load_dir = st.gaussian_dir + '/fold_{}'.format(fold)
            npzfile = np.load(tmp_load_dir + '/mu_sigma.npz')
            mu = npzfile['mu']
            sigma = npzfile['sigma']
            list_train_data[0] = ut.Gauss_Norm_subjectWise(list_train_data[0], mu, sigma, train=False)
            list_val_data[0] = ut.Gauss_Norm_subjectWise(list_val_data[0], mu, sigma, train=False)
            list_test_data[0] = ut.Gauss_Norm_subjectWise(list_test_data[0], mu, sigma, train=False)

    if fst.flag_ttest_save ==True:
        tmp_save_dir = st.ttest_dir + '/fold_{}'.format(fold)
        ut.make_dir(dir=tmp_save_dir, flag_rm=False)
        statistic, pvalue = stats.ttest_ind(list_train_data[0][list_train_data[1] == 0][:, 0], list_train_data[0][list_train_data[1] == 1][:, 0], axis=0, equal_var=False)
        list_thresh= [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]
        thresh = 0.5
        percentile = 0.1
        for i, value in enumerate(list_thresh):
            pvalue_1 = 1 * (pvalue < value)
            pvalue_2 = np.where(np.isnan(pvalue_1), 0, pvalue_1)
            pvalue_2 = pvalue_2.astype(np.float64)
            statistic_1 = statistic * (pvalue < value)
            statistic_2 = np.where(np.isnan(statistic_1), 0, statistic_1)
            ut.save_featureMap_numpy(pvalue_2, tmp_save_dir, 'binary')
            ut.save_featureMap_numpy(statistic_2, tmp_save_dir, 'p_value')
            ut.plot_heatmap_with_overlay(orig_img=list_train_data[0][0][0].squeeze(), heatmap_img=statistic_2,
                                         save_dir=tmp_save_dir + '/value_{}_{}_per_{}_th_{}_n_{}.png'.format(
                                             st.list_selected_for_train[0], st.list_selected_for_train[1],percentile, thresh, i),
                                         fig_title=str(value),
                                         thresh=thresh,
                                         percentile=percentile)

            ut.plot_heatmap_with_overlay(orig_img=list_train_data[0][0][0].squeeze(), heatmap_img=pvalue_2,
                                         save_dir=tmp_save_dir + '/binary_{}_{}_per_{}_th_{}_n_{}.png'.format(
                                             st.list_selected_for_train[0], st.list_selected_for_train[1], percentile, thresh, i),
                                         fig_title=str(value),
                                         thresh=thresh,
                                         percentile=percentile)
    if fst.flag_ttest_load == True:
        tmp_load_dir = st.ttest_dir + '/fold_{}/featuremap'.format(fold)
        t_test_binary = nib.load(tmp_load_dir + '/binary.nii.gz').get_data() # 256, 256, 256
        t_test_value = nib.load(tmp_load_dir + '/p_value.nii.gz').get_data() # 256, 256, 256

    """ make the age label different depending on the class label (NC vs MCI)"""
    # class, lbl, age, MMSE
    if fst.flag_estimate_age == True:
        for i in range(len(st.list_selected_for_train)):
            lambda_disease_factor = st.list_selected_lambdas_at_age[i]
            list_train_data[2][(list_train_data[1] == i)] = ut.estimate_biological_age(
                age=list_train_data[2][(list_train_data[1] == i)],
                MMSE=list_train_data[3][(list_train_data[1] == i)],
                lambda_disease_factor=lambda_disease_factor)

    """ train loader """
    num_data = list_train_data[0].shape[0]
    train_loader = DL.convert_Dloader_3(config.batch_size, list_train_data[0], list_train_data[1], list_train_data[2], list_train_data[3],
                                        is_training=True, num_workers=0, shuffle=True)
    val_loader = DL.convert_Dloader_3(config.v_batch_size, list_val_data[0], list_val_data[1], list_val_data[2], list_val_data[3],
                                      is_training=False, num_workers=0, shuffle=False)
    test_loader = DL.convert_Dloader_3(config.v_batch_size, list_test_data[0], list_test_data[1], list_test_data[2], list_test_data[3],
                                      is_training=False, num_workers=0, shuffle=False)

    del list_train_data, list_val_data, list_test_data


    print('training')
    """ epoch """
    for epoch in range(config.num_epochs):
        epoch = epoch + 1 # increase the # of the epoch
        print(" ")
        print("---------------  epoch {} ----------------".format(epoch))
        torch.cuda.empty_cache()

        """ print learning rate """
        for param_group in optimizer.param_groups:
            print('current LR : {}'.format(param_group['lr']))

        """ batch """
        for i, (datas, labels, alabels, mlabel) in enumerate(train_loader):
            # start = time.time()
            model.train()
            EMS.total_step += 1

            """ input"""
            datas = Variable(datas).cuda()
            labels = Variable(labels.long()).cuda()
            alabels = Variable(alabels.float()).cuda()

            """ forward propagation """
            dict_result = model(datas, alabels)
            output_logit_1 = dict_result['logit_1']
            # output_logit_2 = dict_result['logit_2']
            output_pred_1 = dict_result['pred_1']
            # output_pred_2 = dict_result['pred_2']

            """ loss 1 """
            loss_list_total = []

            """ classification loss """
            loss_2 = criterion_cls(output_logit_1, labels)
            loss_list_total.append(loss_2)
            EMS.train_aux_loss_1.append(loss_2.data.cpu().numpy())

            """ rev_regression loss """
            loss_list_2 = []
            total = 0
            for i_disease in range(len(st.list_selected_for_train)):
                if output_pred_1[(labels == i_disease)].nelement() != 0:
                    loss_list_2.append(criterion_MAE(output_pred_1[(labels == i_disease)].squeeze(),
                                                     alabels[(labels == i_disease)].squeeze()) * st.list_selected_lambdas_at_loss[i_disease])
                    total +=(labels == i_disease).sum().cpu().item() * st.list_selected_lambdas_at_loss[i_disease]

            loss_2 = sum(loss_list_2)
            if total != 0:
                loss_2 = loss_2 / total

            EMS.train_aux_loss_2.append(loss_2.data.cpu().numpy())
            loss_list_total.append(loss_2)

            """ regression loss """
            # loss_list_2 = []
            # total = 0
            # for i_disease in range(len(st.list_selected_for_train)):
            #     if output_pred_2[(labels == i_disease)].nelement() != 0:
            #         loss_list_2.append(criterion_MAE(output_pred_2[(labels == i_disease)].squeeze(),
            #                                          alabels[(labels == i_disease)].squeeze()) * st.list_selected_lambdas_at_loss[i_disease])
            #         total +=(labels == i_disease).sum().cpu().item() * st.list_selected_lambdas_at_loss[i_disease]
            # loss_2 = sum(loss_list_2)
            # if total != 0:
            #     loss_2 = loss_2 / total
            # EMS.train_aux_loss_3.append(loss_2.data.cpu().numpy())
            # loss_list_total.append(loss_2)


            """ classification loss """
            # loss_2 = criterion_cls(output_logit_2, labels)
            # loss_list_total.append(loss_2)
            # EMS.train_aux_loss_4.append(loss_2.data.cpu().numpy())

            loss = sum(loss_list_total)

            """ calculation of the loss through the output"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """ print the train loss and tensorboard"""
            if (EMS.total_step) % 10 == 0 :
                # print('time : ', time.time() - start)
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      %(epoch, config.num_epochs, i + 1, (round(num_data / config.batch_size)), loss.data.cpu().numpy()))

            """ pyplot """
            EMS.train_loss.append(loss.data.cpu().numpy())
            EMS.train_step.append(EMS.total_step)


        """ val """
        if Validation == True:
            print("------------------  val  --------------------------")
            dict_result = ut.eval_multi_task_model(config, fold, val_loader, model, criterion_cls, criterion_MAE)
            val_loss = dict_result['Loss']
            acc = dict_result['Acc']
            auc = dict_result['AUC']
            loss_cls_1 = dict_result['loss_cls_1']
            # loss_cls_2 = dict_result['loss_cls_2']
            loss_age_1 = dict_result['loss_age_1']
            # loss_age_2 = dict_result['loss_age_2']
            EMS.val_loss_1.append(loss_cls_1)
            EMS.val_loss_2.append(loss_age_1)
            # EMS.val_loss_3.append(loss_age_2)
            # EMS.val_loss_4.append(loss_cls_2)
            print('Fold : %d, Epoch [%d/%d] val Loss = %f, val Acc = %f' % (fold, epoch, config.num_epochs, val_loss, acc))

            """ save the metric """
            EMS.dict_val_metric['val_loss'].append(loss_cls_1)
            EMS.dict_val_metric['val_acc'].append(acc)
            EMS.dict_val_metric['val_auc'].append(auc)
            EMS.val_step.append(EMS.total_step)

            n_stacking_loss_for_selection = 5
            if len(EMS.dict_val_metric['val_loss_queue']) > n_stacking_loss_for_selection:
                EMS.dict_val_metric['val_loss_queue'].popleft()
            EMS.dict_val_metric['val_loss_queue'].append(loss_cls_1)
            EMS.dict_val_metric['val_mean_loss'].append(np.mean(EMS.dict_val_metric['val_loss_queue']))

            """ save model """
            for i_tmp in range(len(list_selected_EMS)):
                save_flag = ut.model_save_through_validation(fold, epoch, EMS=EMS,
                                                             selected_EMS=list_selected_EMS[i_tmp],
                                                             ES=list_ES[i_tmp],
                                                             model=model,
                                                             dir_save_model=list_dir_save_model[i_tmp],
                                                             metric_1=st.list_standard_eval[i_tmp], metric_2='',
                                                             save_flag=False)

        if Test_flag== True:
            print("------------------  test _ test dataset  --------------------------")
            """ load data """
            dict_result = ut.eval_multi_task_model(config, fold, test_loader, model, criterion_cls, criterion_MAE)
            test_loss = dict_result['Loss']
            acc = dict_result['Acc']
            loss_cls_1 = dict_result['loss_cls_1']
            # loss_cls_2 = dict_result['loss_cls_2']
            loss_age_1 = dict_result['loss_age_1']
            # loss_age_2 = dict_result['loss_age_2']
            EMS.test_loss_1.append(loss_cls_1)
            EMS.test_loss_2.append(loss_age_1)
            # EMS.test_loss_3.append(loss_age_2)
            # EMS.test_loss_4.append(loss_cls_2)

            """ pyplot """
            EMS.test_acc.append(acc)
            EMS.test_loss.append(loss_cls_1)
            EMS.test_step.append(EMS.total_step)

            print('number of test samples : {}'.format(len(test_loader.dataset)))
            print('Fold : %d, Epoch [%d/%d] test Loss = %f test Acc = %f' % (fold, epoch, config.num_epochs, test_loss, acc))

        """ learning rate decay"""
        EMS.LR.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
        # scheduler.step(val_loss)

        """ plot the chat """
        ut.plot_training_info_1(fold, dir_pyplot, EMS, flag='percentile', flag_match=False)

        tmp_count = 0
        for i in range(len(list_ES)):
            if list_ES[i].early_stop == True:
                tmp_count += 1
        if tmp_count == len(list_ES):
            break

    """ release the model """
    del model
    torch.cuda.empty_cache()

