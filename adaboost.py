import numpy as np
import time


class AdaBoost:
    def __init__(self, pos_features, neg_features, weak_cls):
        self.pos_f = pos_features
        self.neg_f = neg_features
        self.p_count = self.pos_f.shape[0]
        self.n_count = self.neg_f.shape[0]
        self.weak_cls = weak_cls

    def train(self, iteration, indices=None, pos=None, neg=None):
        strong_cls = []
        if indices is None:
            indices = np.array([i for i in range(self.weak_cls.shape[0])])
        if pos is None:
            pos = self.pos_f
        if neg is None:
            neg = self.neg_f
        p_count = pos.shape[0]
        n_count = neg.shape[0]
        w_p = np.ones(p_count) * 1 / (2 * p_count)
        w_n = np.ones(n_count) * 1 / (2 * n_count)
        cur = time.time()
        for i in range(iteration):
            norm_sum = np.sum(w_p) + np.sum(w_n)
            w_p = w_p / norm_sum
            w_n = w_n / norm_sum
            selected_index = -1
            selected_threshold = 0
            selected_parity = 1
            selected_p_res = None
            selected_n_res = None
            error = 1
            for j, index in enumerate(indices):
                threshold = self.weak_cls[index][0]
                parity = self.weak_cls[index][1]
                err, p_res, n_res = self.get_cls_err(pos[:, index], neg[:, index], w_p, w_n, threshold,
                                                     parity)
                if err < error:
                    # print('index: {}, errpr: {}'.format(index, err))
                    error = err
                    selected_index = index
                    selected_threshold = threshold
                    selected_parity = parity
                    selected_p_res = p_res
                    selected_n_res = n_res
            beta = error / (1 - error)
            w_p = w_p * np.float_power(beta, selected_p_res)
            w_n = w_n * np.float_power(beta, 1 - selected_n_res)
            alpha = np.log(1 / beta)
            strong_cls.append([selected_index, selected_threshold, selected_parity, alpha])
            print('Classifier {}--index: {}, threshold: {}, parity: {}, error: {}, alpha: {}'.format(i + 1,
                                                                                                     selected_index,
                                                                                                     selected_threshold,
                                                                                                     selected_parity,
                                                                                                     error, alpha))
        save_file = './cascade_v3/strong_cls-{}k-{}.npy'.format(int(len(indices) / 1000), iteration)
        np.save(save_file, strong_cls)
        print('*' * 10, 'Saved [{}], training using: {}s'.format(save_file, time.time() - cur), '*' * 10)
        return np.asarray(strong_cls)

    def predict(self, strong_cls, features, label=None):
        if len(features.shape) > 2:
            n_sample = 1
            features = features.reshape(1, -1)
        else:
            n_sample = features.shape[0]
        pred = np.zeros(n_sample)
        tmp = np.zeros_like(pred)
        for i in range(strong_cls.shape[0]):
            alpha = strong_cls[i, 3]
            index = strong_cls[i, 0]
            threshold = strong_cls[i, 1]
            parity = strong_cls[i, 2]
            weak_pred = (((features[:, int(index)] - threshold) * parity) > 0)
            weak_pred = weak_pred.astype(np.int)
            tmp += alpha * weak_pred
        strong_threshold = 0.5 * np.sum(strong_cls[:, 3])
        pred[tmp >= strong_threshold] = 1
        if not label:
            return pred
        else:
            return pred

    def get_cls_err(self, p_feature, n_feature, w_p, w_n, threshold, parity):
        p_res = np.zeros(p_features.shape[0])
        n_res = np.zeros(n_features.shape[0])
        p_res[(p_feature - threshold) * parity > 0] = 1
        n_res[(n_feature - threshold) * parity > 0] = 1
        err = np.sum(w_p * np.abs(p_res - 1)) + np.sum(w_n * n_res)
        return err, p_res, n_res


def generate_cls(p_features, n_features, kernels):
    weak_classifiers = []
    m = kernels.shape[0]
    for f_num in range(m):
        pf = p_features[:3000, f_num]
        nf = n_features[:3000, f_num]
        elems = np.unique(np.concatenate((pf, nf), axis=0))
        print('min: {}, max: {}'.format(elems.min(), elems.max()))
        err = 0
        threshold = 0
        parity = 1
        cur = time.time()
        for e in elems:
            parity = 1
            p_res = np.zeros(pf.shape[0])
            n_res = np.zeros(nf.shape[0])
            p_res[pf > e] = 1
            n_res[nf > e] = 1
            num_pf1 = len(p_res[p_res == 1])
            num_nf0 = len(n_res[n_res == 0])
            num_pf = len(pf)
            num_nf = len(nf)
            err_rate = (num_pf1 + num_nf0) / (num_pf + num_nf)
            if 1. - err_rate > err_rate:
                parity = -1
                err_rate = 1. - err_rate
            if err_rate > err:
                err = err_rate
                threshold = e
        weak_classifiers.append([threshold, parity])
        if (f_num + 1) % 10000 == 0:
            print('*' * 10, 'saved weak_classifiers-{}.npy'.format(f_num + 1), '*' * 10)
            np.save('./weak_classifiers-{}.npy'.format(f_num + 1), np.asarray(weak_classifiers))
            weak_classifiers = []
        print(
            'feature: {}/{}, threshold: {}, error rate: {}, parity: {}, using: {}s'.format(f_num + 1, m, threshold, err,
                                                                                           parity, time.time() - cur))
    weak_classifiers = np.asarray(weak_classifiers)
    np.save('./weak_classifiers-last.npy', weak_classifiers)
    cls1 = np.load('./weak_classifiers-10000.npy')
    cls2 = np.load('./weak_classifiers-20000.npy')
    cls3 = np.load('./weak_classifiers-30000.npy')
    cls4 = np.load('./weak_classifiers-40000.npy')
    cls5 = np.load('./weak_classifiers-50000.npy')
    cls6 = np.load('./weak_classifiers-60000.npy')
    cls7 = np.load('./weak_classifiers-70000.npy')
    cls8 = np.load('./weak_classifiers-80000.npy')
    cls9 = np.load('./weak_classifiers-90000.npy')
    cls10 = np.load('./weak_classifiers-100000.npy')
    cls11 = np.load('./weak_classifiers-110000.npy')
    cls12 = np.load('./weak_classifiers-120000.npy')
    cls13 = np.load('./weak_classifiers-130000.npy')
    cls14 = np.load('./weak_classifiers-140000.npy')
    cls15 = np.load('./weak_classifiers-150000.npy')
    cls16 = np.load('./weak_classifiers-160000.npy')
    cls17 = np.load('./weak_classifiers-last.npy')
    cls = np.concatenate(
        (cls1, cls2, cls3, cls4, cls5, cls6, cls7, cls8, cls9, cls10, cls11, cls12, cls13, cls14, cls15, cls16, cls17),
        axis=0)
    np.save('./weak_classifiers.npy', cls)
    return cls


if __name__ == '__main__':
    # p = np.load('./dataset/positive.npy')
    # n = np.load('./dataset/negative-pre.npy')
    n_features = np.load('./n_features-pre.npy')
    # np.random.shuffle(n_features)
    # n_features = n_features[:3000]
    print('load negative features')
    p_features = np.load('./p_features.npy')
    np.random.shuffle(p_features)
    p_features = p_features[:3000]
    print('load positive features')

    # kernels = np.load('./haar-(24,24).npy')
    cls = np.load('./weak_classifiers.npy')
    print('load weak classifiers')
    adaboost = AdaBoost(p_features, n_features, cls)
    # rand_idx = np.random.permutation(cls.shape[0])
    layers = [1, 5, 10, 15, 20]
    for i in layers:
        # np.random.shuffle(rand_idx)
        # indices = rand_idx
        strong_cls = adaboost.train(i, indices=None, neg=n_features)
        # strong_cls = np.load('./strong_cls-160k-10.npy')
        cur = time.time()
        pred_p = adaboost.predict(strong_cls, p_features[:1000])
        pred_n = adaboost.predict(strong_cls, n_features)
        n_count = len(pred_n)
        total = n_count + 1000
        tp = len(pred_p[pred_p == 1])
        fp = 1000 - tp
        tn = len(pred_n[pred_n == 0])
        fn = n_count - tn
        print(
            'test using: {}s, tp: {}, fp: {}, fn: {}, tn: {}, total: {}'.format(time.time() - cur,
                                                                                tp / total,
                                                                                fp / total,
                                                                                fn / total,
                                                                                tn / total,
                                                                                (tp + tn) / total))
        # n_features_tn = n_features[pred_n == 0]
        n_features = n_features[pred_n == 1]
        # n_features = np.concatenate((n_features, n_features_tn[:int(0.5 * n_features_tn.shape[0])]), axis=0)
        print('negative: {}'.format(n_features.shape[0]))
