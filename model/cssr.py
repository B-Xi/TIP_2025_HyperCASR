import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class GramRecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.gram_feats = []
        self.collecting = False

    def begin_collect(self, ):
        self.gram_feats.clear()
        self.collecting = True
        # print("begin collect")

    def record(self, ft):
        if self.collecting:
            self.gram_feats.append(ft)
            # print("record")

    def obtain_gram_feats(self, ):
        tmp = self.gram_feats
        self.collecting = False
        self.gram_feats = []
        # print("record")
        return tmp


class LinearClassifier(nn.Module):

    def __init__(self, inchannels, num_class, config):
        super().__init__()
        self.gamma = config['gamma']
        self.cls = nn.Conv2d(inchannels, num_class, 1, padding=0, bias=False)

    def forward(self, x):
        x = self.cls(x)
        return x * self.gamma


def sim_conv_layer(input_channel, output_channel, kernel_size=1, padding=0, use_activation=True):
    if use_activation:
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False),
            nn.Tanh())
    else:
        res = nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False)
    return res


class AutoEncoder(nn.Module):

    def __init__(self, inchannel, hidden_layers, latent_chan):
        super().__init__()
        layer_block = sim_conv_layer
        self.latent_size = latent_chan
        if latent_chan > 0:
            self.encode_convs = []
            self.decode_convs = []
            for i in range(len(hidden_layers)):
                h = hidden_layers[i]
                ecv = layer_block(inchannel, h, )
                dcv = layer_block(h, inchannel, use_activation=i != 0)
                inchannel = h
                self.encode_convs.append(ecv)
                self.decode_convs.append(dcv)
            self.encode_convs = nn.ModuleList(self.encode_convs)
            self.decode_convs.reverse()
            self.decode_convs = nn.ModuleList(self.decode_convs)
            self.latent_conv = layer_block(inchannel, latent_chan)
            self.latent_deconv = layer_block(latent_chan, inchannel, use_activation=(len(hidden_layers) > 0))
        else:
            self.center = nn.Parameter(torch.rand([inchannel, 1, 1]), True)

    def forward(self, x):
        if self.latent_size > 0:
            output = x
            for cv in self.encode_convs:
                output = cv(output)
            latent = self.latent_conv(output)
            output = self.latent_deconv(latent)
            for cv in self.decode_convs:
                output = cv(output)
            return output, latent
        else:
            return self.center, self.center


class CSSRClassifier(nn.Module):

    def __init__(self, inchannels, num_class, config):
        super().__init__()
        ae_hidden = config['ae_hidden']
        ae_latent = config['ae_latent']
        self.class_aes = []
        for i in range(num_class):
            ae = AutoEncoder(inchannels, ae_hidden, ae_latent)
            self.class_aes.append(ae)
        self.class_aes = nn.ModuleList(self.class_aes)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.useL1 = config['error_measure'] == 'L1'

        self.reduction = -1 if config['model'] == 'pcssr' else 1
        self.reduction *= config['gamma']

    def ae_error(self, rc, x):
        if self.useL1:
            # return torch.sum(torch.abs(rc-x) * self.reduction,dim=1,keepdim=True)
            return torch.norm(rc - x, p=1, dim=1, keepdim=True) * self.reduction
        else:
            return torch.norm(rc - x, p=2, dim=1, keepdim=True) ** 2 * self.reduction

    clip_len = 100

    def forward(self, x):
        cls_ers = []
        for i in range(len(self.class_aes)):
            rc, lt = self.class_aes[i](x)
            cls_er = self.ae_error(rc, x)
            if CSSRClassifier.clip_len > 0:
                cls_er = torch.clamp(cls_er, -CSSRClassifier.clip_len, CSSRClassifier.clip_len)
            cls_ers.append(cls_er)
        logits = torch.cat(cls_ers, dim=1)
        return logits


def G_p(ob, p):
    temp = ob.detach()
    temp = temp ** p
    temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
    temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1))))  #
    temp = temp.reshape([temp.shape[0], -1])  # .sum(dim=2)
    temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(temp.shape[0], -1)

    return temp


def G_p_pro(ob, p=8):
    temp = ob.detach()
    temp = temp ** p
    temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
    temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1))))  #
    # temp = temp.reshape([temp.shape[0],-1])#.sum(dim=2)
    temp = (temp.sign() * torch.abs(temp) ** (1 / p))  # .reshape(temp.shape[0],ob.shape[1],ob.shape[1])

    return temp


def G_p_inf(ob, p=1):
    temp = ob.detach()
    temp = temp ** p
    # print(temp.shape)
    temp = temp.reshape([temp.shape[0], temp.shape[1], -1]).transpose(dim0=2, dim1=1).reshape([-1, temp.shape[1], 1])
    # print(temp.shape)
    temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1))))  #
    temp = (temp.sign() * torch.abs(temp) ** (1 / p))
    # print(temp.shape)
    return temp.reshape(ob.shape[0], ob.shape[2], ob.shape[3], ob.shape[1], ob.shape[1])


# import methods.pooling.MPNConv as MPN

class BackboneAndClassifier(nn.Module):

    def __init__(self, num_classes, config):
        super().__init__()
        clsblock = {'linear': LinearClassifier, 'pcssr': CSSRClassifier, 'rcssr': CSSRClassifier}
        cat_config = config['category_model']
        self.cat_cls = clsblock[cat_config['model']](64, num_classes, cat_config)

    def forward(self, x, feature_only=False):
        if feature_only:
            return x
        return x, self.cat_cls(x)


class CSSRModel(nn.Module):

    def __init__(self, num_classes, config, crt):
        super().__init__()
        self.crt = crt

        # ------ New Arch
        self.backbone_cs = BackboneAndClassifier(num_classes, config)

        self.config = config
        self.mins = {i: [] for i in range(num_classes)}
        self.maxs = {i: [] for i in range(num_classes)}
        self.num_classes = num_classes

        self.avg_feature = [[0, 0] for i in range(num_classes)]
        self.avg_gram = [[[0, 0] for i in range(num_classes)] for i in self.powers]
        self.enable_gram = config['enable_gram']

    def update_minmax(self, feat_list, power=[], ypred=None):
        # feat_list = self.gram_feature_list(batch)
        for pr in range(self.num_classes):
            cond = ypred == pr
            if not cond.any():
                continue
            for L, feat_L in enumerate(feat_list):
                if L == len(self.mins[pr]):
                    self.mins[pr].append([None] * len(power))
                    self.maxs[pr].append([None] * len(power))

                for p, P in enumerate(power):
                    g_p = G_p(feat_L[cond], P)

                    current_min = g_p.min(dim=0, keepdim=True)[0]
                    current_max = g_p.max(dim=0, keepdim=True)[0]

                    if self.mins[pr][L][p] is None:
                        self.mins[pr][L][p] = current_min
                        self.maxs[pr][L][p] = current_max
                    else:
                        self.mins[pr][L][p] = torch.min(current_min, self.mins[pr][L][p])
                        self.maxs[pr][L][p] = torch.max(current_max, self.maxs[pr][L][p])

    def get_deviations(self, feat_list, power, ypred):
        batch_deviations = None
        for pr in range(self.num_classes):
            mins, maxs = self.mins[pr], self.maxs[pr]
            cls_batch_deviations = []
            cond = ypred == pr
            if not cond.any():
                continue
            for L, feat_L in enumerate(feat_list):
                dev = 0
                for p, P in enumerate(power):
                    g_p = G_p(feat_L[cond], P)
                    # print(L,len(mins))
                    # print(p,len(mins[L]))
                    dev += (F.relu(mins[L][p] - g_p) / torch.abs(mins[L][p] + 10 ** -6)).sum(dim=1, keepdim=True)
                    dev += (F.relu(g_p - maxs[L][p]) / torch.abs(maxs[L][p] + 10 ** -6)).sum(dim=1, keepdim=True)
                cls_batch_deviations.append(dev.cpu().detach().numpy())
            cls_batch_deviations = np.concatenate(cls_batch_deviations, axis=1)
            if batch_deviations is None:
                batch_deviations = np.zeros([ypred.shape[0], cls_batch_deviations.shape[1]])
            batch_deviations[cond] = cls_batch_deviations
        return batch_deviations

    powers = [8]

    def cal_feature_prototype(self, feat, ypred):
        feat = torch.abs(feat)
        for pr in range(self.num_classes):
            cond = ypred == pr
            if not cond.any():
                continue
            csfeat = feat[cond]#pr类的特征
            cf = csfeat.mean(dim=[0, 2, 3])  # .cpu().numpy() #pr类的平均特征
            # print(cf.shape)
            ct = cond.sum()
            ft = self.avg_feature[pr]
            self.avg_feature[pr] = [ft[0] + ct, (ft[1] * ft[0] + cf * ct) / (ft[0] + ct)]#pr类的平均特征
            if self.enable_gram:
                for p in range(len(self.powers)):
                    gram = G_p_pro(csfeat, self.powers[p]).mean(dim=0)
                    gm = self.avg_gram[p][pr]
                    self.avg_gram[p][pr] = [gm[0] + ct, (gm[1] * gm[0] + gram * ct) / (gm[0] + ct)]

    def obtain_usable_feature_prototype(self):
        if isinstance(self.avg_feature, list):
            clsft_lost = []
            exm = None
            for x in self.avg_feature:
                if x[0] > 0:
                    clsft_lost.append(x[1])
                    exm = x[1]
                else:
                    clsft_lost.append(None)
            clsft = torch.stack([torch.zeros_like(exm) if x is None else x for x in clsft_lost])
            # print(clsft.shape)
            clsft /= clsft.sum(dim=0)  # **2
            # clsft /= clsft.sum(dim = 1,keepdim = True)
            # print(clsft)
            self.avg_feature = clsft.reshape([clsft.shape[0], 1, clsft.shape[1], 1, 1])
            if self.enable_gram:
                for i in range(len(self.powers)):
                    self.avg_gram[i] = torch.stack(
                        [x[1] if x[0] > 0 else torch.zeros([exm.shape[0], exm.shape[0]]).cuda() for x in
                         self.avg_gram[i]])
            # self.avg_gram /= self.avg_gram.sum(dim = 0)
            # print(self.avg_gram.shape)
        return self.avg_feature, self.avg_gram

    def get_feature_prototype_deviation(self, feat, ypred):
        # feat = torch.abs(feat)
        avg_feature, _ = self.obtain_usable_feature_prototype()
        scores = np.zeros([feat.shape[0], feat.shape[2], feat.shape[3]])
        for pr in range(self.num_classes):
            cond = ypred == pr
            if not cond.any():
                continue
            scores[cond] = (avg_feature[pr] * feat[cond]).mean(axis=1).cpu().numpy()
        return scores

    def get_feature_gram_deviation(self, feat, ypred):
        _, avg_gram = self.obtain_usable_feature_prototype()
        scores = np.zeros([feat.shape[0], feat.shape[2], feat.shape[3]])
        for pr in range(self.num_classes):
            cond = ypred == pr
            if not cond.any():
                continue
            res = 0
            for i in range(len(self.powers)):
                gm = G_p_pro(feat[cond], p=self.powers[i])
                # scores[cond] = (gm / gm.mean(dim = [3,4],keepdim = True) * avg_gram[pr]).sum(dim = [3,4]).cpu().numpy()
                res += (gm * avg_gram[i][pr]).sum(dim=[1, 2], keepdim=True).cpu().numpy()
            scores[cond] = res
        return scores

    def pred_by_feature_gram(self, feat):
        _, avg_gram = self.obtain_usable_feature_prototype()
        scores = np.zeros([self.num_classes, feat.shape[0]])
        gm = G_p_pro(feat)
        for pr in range(self.num_classes):
            # scores[cond] = (gm / gm.mean(dim = [3,4],keepdim = True) * avg_gram[pr]).sum(dim = [3,4]).cpu().numpy()
            scores[pr] = (gm * avg_gram[pr]).sum(dim=[1, 2]).cpu().numpy()
        return scores.argmax(axis=0)

    def forward(self, x, ycls=None, reqpredauc=False, prepareTest=False, reqfeature=False):

        # ----- New Arch
        x = self.backbone_cs(x, feature_only=reqfeature)
        if reqfeature:
            return x
        x, xcls_raw = x

        def pred_score(xcls):
            score_reduce = lambda x: x.reshape([x.shape[0], -1]).mean(axis=1)
            x_detach = x.detach()
            probs = self.crt(xcls, prob=True).cpu().numpy()
            pred = probs.argmax(axis=1)#预测类
            max_prob = probs.max(axis=1)#预测类的概率

            cls_scores = xcls.cpu().numpy()[[i for i in range(pred.shape[0])], pred]#预测类的损失
            rep_scores = torch.abs(x_detach).mean(dim=1).cpu().numpy()#平均特征
            if not self.training and not prepareTest and (
                    not isinstance(self.avg_feature, list) or self.avg_feature[0][0] != 0):
                rep_cspt = self.get_feature_prototype_deviation(x_detach, pred)
                if self.enable_gram:
                    rep_gram = self.get_feature_gram_deviation(x_detach, pred)
                else:
                    rep_gram = np.zeros_like(cls_scores)
            else:
                rep_cspt = np.zeros_like(cls_scores)
                rep_gram = np.zeros_like(cls_scores)
            R = [cls_scores, rep_scores, rep_cspt, rep_gram, max_prob]

            scores = np.stack(
                [score_reduce(eval(self.config['score'])), score_reduce(rep_cspt), score_reduce(rep_gram)], axis=1)
            return pred, scores

        if self.training:
            xcls = self.crt(xcls_raw, ycls)
            if reqpredauc:
                pred, score = pred_score(xcls_raw.detach())
                return xcls, pred, score
        else:
            xcls = xcls_raw
            # xrot = self.rot_cls(x)
            if reqpredauc:
                pred, score = pred_score(xcls)
                deviations = None
                # powers = range(1,10)
                if prepareTest:
                    if not isinstance(self.avg_feature, list):
                        self.avg_feature = [[0, 0] for i in range(self.num_classes)]
                        self.avg_gram = [[[0, 0] for i in range(self.num_classes)] for i in self.powers]
                    # hdfts = self.backbone.backbone.obtain_gram_feats()
                    # self.update_minmax(hdfts + [x] + clslatents,powers,pred)
                    self.cal_feature_prototype(x, pred)
                # else:
                #     deviations = self.get_deviations(self.backbone.backbone.obtain_gram_feats() + [x]+ clslatents,powers,pred)
                return pred, score, deviations

        return xcls


class CSSRCriterion(nn.Module):

    def get_onehot_label(self, y, clsnum):
        y = torch.reshape(y, [-1, 1]).long()
        return torch.zeros(y.shape[0], clsnum).cuda().scatter_(1, y, 1)

    def __init__(self, avg_order, enable_sigma=True):
        super().__init__()
        self.avg_order = {"avg_softmax": 1, "softmax_avg": 2}[avg_order]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.enable_sigma = enable_sigma

    def forward(self, x, y=None, prob=False, pred=False):
        if self.avg_order == 1:
            g = self.avg_pool(x).view(x.shape[0], -1)
            g = torch.softmax(g, dim=1)
        elif self.avg_order == 2:
            g = torch.softmax(x, dim=1)
            g = self.avg_pool(g).view(x.size(0), -1)
        if prob: return g
        if pred: return torch.argmax(g, dim=1)
        loss = -torch.sum(self.get_onehot_label(y, g.shape[1]) * torch.log(g), dim=1).mean()
        return loss
