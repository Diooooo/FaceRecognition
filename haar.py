from utils import *
import time


class HaarType:
    TWO_V = 0
    TWO_H = 1
    THREE_V = 2
    THREE_H = 3
    FOUR_DIA = 4
    TYPE_COUNT = 5
    kernel_type = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]


class HaarFeature:
    def __init__(self, wnd_x, wnd_y):
        self.wnd_x = wnd_x
        self.wnd_y = wnd_y

    def generate_feature_kernels(self, save=True, save_path='./'):
        feature_kernel = []

        for kernel_type in range(HaarType.TYPE_COUNT):
            kernel_x, kernel_y = HaarType.kernel_type[kernel_type]
            for kx in range(kernel_x, self.wnd_x + 1, kernel_x):
                for ky in range(kernel_y, self.wnd_y + 1, kernel_y):
                    for pos_x in range(0, self.wnd_x - kx + 1):
                        for pos_y in range(0, self.wnd_y - ky + 1):
                            feature_kernel.append([kernel_type, kx, ky, pos_x, pos_y])
        feature_kernel = np.asarray(feature_kernel)
        if len(feature_kernel.shape) > 2:
            feature_kernel = np.expand_dims(feature_kernel, axis=0)
        if save:
            if os.path.exists(save_path):
                np.save(os.path.join(save_path, 'haar-({},{})'.format(self.wnd_x, self.wnd_y)), feature_kernel)
        return feature_kernel

    def get_features(self, wnd_img, path, k=None):
        if path:
            kernels = np.load(path)
        else:
            kernels = k
        if kernels is None:
            raise ValueError('wrong feature descriptors')
        integral_img = integral(wnd_img)
        features = np.zeros(kernels.shape[0])
        for i in range(kernels.shape[0]):
            features[i] = self._get_single_feature(integral_img, kernels[i])
        return features

    def _get_single_feature(self, integral_img, kernel):
        kernel_type, kx, ky, pos_x, pos_y = kernel
        if pos_x + kx > self.wnd_x or pos_y + ky > self.wnd_y:
            raise ValueError('position out of range')
        black = 0
        white = get_sum_pixel(integral_img, kx, ky, pos_x, pos_y)
        if kernel_type == HaarType.TWO_V:
            black = get_sum_pixel(integral_img, kx, ky // 2, pos_x, pos_y)
        elif kernel_type == HaarType.TWO_H:
            black = get_sum_pixel(integral_img, kx // 2, ky, pos_x, pos_y)
        elif kernel_type == HaarType.THREE_V:
            black = get_sum_pixel(integral_img, kx, ky // 3, pos_x, pos_y + ky // 3)
        elif kernel_type == HaarType.THREE_H:
            black = get_sum_pixel(integral_img, kx // 3, ky, pos_x + kx // 3, pos_y)
        elif kernel_type == HaarType.FOUR_DIA:
            black = get_sum_pixel(integral_img, kx // 2, ky // 2, pos_x + kx // 2, pos_y) \
                    + get_sum_pixel(integral_img, kx // 2, ky // 2, pos_x, pos_y + ky // 2)
        else:
            raise ValueError('unknown kernel type')
        return 2 * black - white


if __name__ == '__main__':
    haar = HaarFeature(24, 24)
    # features = haar.generate_feature_kernels()
    # p = np.load('./dataset/positive.npy')
    n = np.load('./dataset/negative.npy')
    path = './haar-(24,24).npy'
    kernel = np.load(path)
    cur = time.time()
    p_features = []
    n_features = []
    cur = time.time()
    for i in range(min(10000, n.shape[0])):
        f = haar.get_features(n[i], None, kernel)
        n_features.append(f)
        if i % 10 == 0:
            print('computed: {}, using: {}s'.format(i + 1, time.time() - cur))
            cur = time.time()
    print('using: {}s'.format(time.time() - cur))
    np.save('n_features.npy', n_features)
    # img = p[0]
    # inte = integral(img)
    # kernel = np.load('./haar-(24,24).npy')
    # kernel_type, kx, ky, pos_x, pos_y = kernel[0]
    # white = get_sum_pixel(inte, kx, ky, pos_x, pos_y)
