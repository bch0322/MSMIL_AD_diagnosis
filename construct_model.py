import setting as st

""" model """
from model_arch import L_bagNet
from model_arch import L_bagNet_with_attention_1
from model_arch import L_bagNet_multi_scale

def construct_model(config, flag_model_num = 0):
    """ construct model """
    if flag_model_num == 0:
        model_num = st.model_num_0
    elif flag_model_num == 1:
        model_num = st.model_num_1
    elif flag_model_num == 2:
        model_num = st.model_num_2

    if model_num == 0:
        pass
    elif model_num == 12:
        model = L_bagNet.bagNet9(config).cuda()
    elif model_num == 13:
        model = L_bagNet.bagNet17(config).cuda()
    elif model_num == 14:
        model = L_bagNet.bagNet33(config).cuda()

    elif model_num == 15:
        model = L_bagNet_with_attention_1.bagNet9(config).cuda()
    elif model_num == 16:
        model = L_bagNet_with_attention_1.bagNet17(config).cuda()
    elif model_num == 17:
        model = L_bagNet_with_attention_1.bagNet33(config).cuda()


    elif model_num == 85:
        model = L_bagNet_multi_scale.bagNet33(config).cuda()


    return model

