import os
import torch
from torch import nn
from model_irse import Backbone

class IDLoss(nn.Module):
    def __init__(self, base_dir='./', device='cuda', ckpt_dict=None):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace', flush=True)
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').to(device)
        if ckpt_dict is None:
            self.facenet.load_state_dict(torch.load(os.path.join(base_dir, 'weights', 'model_ir_se50.pth'), map_location=torch.device('cpu')))
        else:
            self.facenet.load_state_dict(ckpt_dict)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        _, _, h, w = x.shape
        assert h==w
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ss = h//256
        # ss = h // 64
        # print("ss ->", ss)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # x = x[:, :, 19 * ss:-17 * ss, 16 * ss:-20 * ss]
        x = x[:, :, 35*ss:-33*ss, 32*ss:-36*ss]  # Crop interesting region
        # print("ss return", x.shape)
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    # id_loss(fake_img, real_img, input_img)
    def forward(self, y_hat, y, x):
        # print("--------------------------------------------IDLoss forward--------------------------------------------")
        # y_hat -> torch.Size([2, 3, 64, 64])
        # y -> torch.Size([2, 3, 64, 64])
        # x -> torch.Size([2, 3, 64, 64])
        # print("y_hat ->", y_hat.shape)
        # print("y ->", y.shape)
        # print("x ->", x.shape)
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there # В противном случае используйте функцию оттуда
        y_hat_feats = self.extract_feats(y_hat)

        # y_hat_feats -> torch.Size([2, 512])
        # y_feats -> torch.Size([2, 512])
        # x_feats -> torch.Size([2, 512])
        # print("y_hat_feats ->", y_hat_feats.shape)
        # print("y_feats ->", y_feats.shape)
        # print("x_feats ->", x_feats.shape)

        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({'diff_target': float(diff_target),
                            'diff_input': float(diff_input),
                            'diff_views': float(diff_views)})
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs

