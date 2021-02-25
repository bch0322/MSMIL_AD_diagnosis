# from __future__ import division
from modules import *

class Residual_Conv(nn.Module):
    def __init__(self, config, strides=[2,2,2,1], kernel3 = [1,1,1,1], in_p = 1, f_out = [16, 32, 64, 128]):

        """ init """
        self.cur_shape = np.array([st.x_size, st.y_size, st.z_size])
        self.num_classes = config.num_classes
        self.widening_factor = 1
        self.inplanes = in_p * self.widening_factor
        f_out = [f_out[i] * self.widening_factor for i in range(len(f_out))]
        self.kernel = kernel3
        self.stride = strides
        super(Residual_Conv, self).__init__()

        """ filter bank """
        self.layer0 = BasicConv_Block(in_planes=1, out_planes=self.inplanes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, act_func='relu', norm_layer='bn', bias=False)
        self.layer1 = BasicConv_Block(in_planes=self.inplanes, out_planes=f_out[0], kernel_size=kernel3[0], stride=strides[0], padding=0, dilation=1, groups=1, act_func='relu', norm_layer='bn', bias=False)
        self.layer2 = BasicConv_Block(in_planes=f_out[0], out_planes=f_out[1], kernel_size=kernel3[1], stride=strides[1], padding=0, dilation=1, groups=1, act_func='relu', norm_layer='bn', bias=False)


        self.layer3_a = BasicConv_Block(in_planes=f_out[1], out_planes=f_out[2], kernel_size=1, stride=strides[2], padding=0, dilation=1, groups=1, act_func='relu', norm_layer='bn', bias=False)
        self.layer3_bc = BasicConv_Block(in_planes=f_out[1], out_planes=f_out[2], kernel_size=kernel3[2], stride=strides[2], padding=0, dilation=1, groups=1, act_func='relu', norm_layer='bn', bias=False)

        self.layer4_a = BasicConv_Block(in_planes=f_out[2], out_planes=f_out[3], kernel_size=1, stride=strides[3], padding=0, dilation=1, groups=1, act_func='relu', norm_layer='bn', bias=False)
        self.layer4_b = BasicConv_Block(in_planes=f_out[2], out_planes=f_out[3], kernel_size=1, stride=strides[3], padding=0, dilation=1, groups=1, act_func='relu', norm_layer='bn', bias=False)
        self.layer4_c = BasicConv_Block(in_planes=f_out[2], out_planes=f_out[3], kernel_size=kernel3[3], stride=strides[3], padding=0, dilation=1, groups=1, act_func='relu', norm_layer='bn', bias=False)

        """ classifier """
        f_out_encoder = f_out[-1]
        self.classifier_a = nn.Sequential(
            nn.Conv3d(f_out_encoder , self.num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        """ attn  """
        f_out_encoder = f_out[-1]

        self.attn_1 = nn.Sequential(
            BasicConv_Block(in_planes=f_out_encoder, out_planes=f_out_encoder // 2, kernel_size=1, stride=1,
                            padding=0, dilation=1, groups=1, act_func='tanh', norm_layer=None, bias=False),
            BasicConv_Block(in_planes=f_out_encoder // 2, out_planes=1, kernel_size=1, stride=1, padding=0, dilation=1,
                            groups=1, act_func='sigmoid', norm_layer=None, bias=False)
        )

        # self.attn_1 = BasicConv_Block(in_planes=f_out_encoder, out_planes=f_out_encoder//2, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, act_func='tanh', norm_layer=None, bias=False)
        # self.gate_1 = BasicConv_Block(in_planes=f_out_encoder, out_planes=f_out_encoder//2, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, act_func='sigmoid', norm_layer=None, bias=False)
        # self.attn_2 = BasicConv_Block(in_planes=f_out_encoder//2, out_planes=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, act_func='sigmoid', norm_layer=None, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, datas, *args):
        """ feature extraction grid patches """
        if len(datas.shape) != 5:
            datas = datas[:, :, 0, :, :, :]
        else:
            datas = datas

        if datas.shape[1] != 1: # GM only
            datas = datas[:, 0:1]

        x_0 = datas

        """ cropoping """
        if self.training == True:
            dict_result = ut.data_augmentation(x_0)
            x_0 = dict_result['datas']

        """ down sampling """
        if fst.flag_downSample == True:
            x_0 = nn.AvgPool3d(kernel_size=3, stride=2)(x_0) # batch, 1, 127, 127, 127

        """ encoder """
        x_0 = self.layer0(x_0)  #
        x_0 = self.layer1(x_0)  #
        x_0 = self.layer2(x_0)  #

        x_a = x_0
        x_bc = x_0
        x_a = self.layer3_a(x_a)  #
        x_bc = self.layer3_bc(x_bc)  #

        x_a = self.layer4_a(x_a)  #
        x_b = self.layer4_b(x_bc)  #
        x_c = self.layer4_c(x_bc)  #

        """ classifier """
        logitMap_a = self.classifier_a(x_a)
        logitMap_b = self.classifier_a(x_b)
        logitMap_c = self.classifier_a(x_c)

        f_attn_a = self.attn_1(x_a) # 343
        f_attn_b = self.attn_1(x_b) # 216
        f_attn_c = self.attn_1(x_c) # 64

        # f_gate_a = self.gate_1(x_a) # 343
        # f_gate_b = self.gate_1(x_b) # 216
        # f_gate_c = self.gate_1(x_c) # 64
        #
        # f_attn_a = self.attn_2(f_attn_a * f_gate_a)
        # f_attn_b = self.attn_2(f_attn_b * f_gate_b)
        # f_attn_c = self.attn_2(f_attn_c * f_gate_c)


        logitMap_a = logitMap_a * f_attn_a
        logitMap_b = logitMap_b * f_attn_b
        logitMap_c = logitMap_c * f_attn_c


        tmp_a = logitMap_a.view(logitMap_a.size(0), logitMap_a.size(1), -1)
        tmp_b = logitMap_b.view(logitMap_b.size(0), logitMap_b.size(1), -1)
        tmp_c = logitMap_c.view(logitMap_c.size(0), logitMap_c.size(1), -1)

        logitMaps = torch.cat([tmp_a, tmp_b, tmp_c], dim=-1)

        """ flatten """
        image_level_logit = torch.mean(logitMaps, dim=-1)

        dict_result = {
            "logits": image_level_logit, # batch, 2
            "Aux_logits": None,  # batch, 2
            "attn_1" : f_attn_a, # batch, 1, w, h, d
            "attn_2": f_attn_b,  # batch, 1, w, h, d
            "attn_3": f_attn_c,  # batch, 1, w, h, d
            "logitMap" : None, # batch, 2, w, h ,d
            "final_evidence" : None, # batch, 2, w, h, d
            "final_evidence_a": logitMap_a,
            "final_evidence_b": logitMap_b,
            "final_evidence_c": logitMap_c,
            "featureMaps" : [],
        }
        return dict_result

def bagNet9(config):
    """BagNet 9 """
    model = Residual_Conv(config, strides=[2, 2, 2, 1], kernel3=[3, 3, 1, 1], in_p=8, f_out=[16, 32, 64, 128])
    return model

def bagNet17(config):
    """BagNet 17 """
    model = Residual_Conv(config, strides=[2, 2, 2, 1], kernel3=[3, 3, 3, 1], in_p=8, f_out=[16, 32, 64, 128])
    return model

def bagNet33(config):
    """BagNet 33 """
    model = Residual_Conv(config, strides=[2, 2, 2, 1], kernel3=[3, 3, 3, 3], in_p=8, f_out=[16, 32, 64, 128])
    return model
