import albumentations as A
import numpy as np

class Cutout(A.ImageOnlyTransform):
    def __init__(self, num_holes=1, max_h_size=256, max_w_size=256, always_apply=False, p=0.5):
        super(Cutout, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def apply(self, image, **params):
        img_height, img_width = image.shape[0], image.shape[1]
        image_copy = image.copy()

        for _ in range(self.num_holes):
            h_size = self.max_h_size
            w_size = self.max_w_size

            h_loc = np.random.randint(0, img_height)
            w_loc = np.random.randint(0, img_width)

            h1 = max(0, h_loc - h_size // 2)
            h2 = min(img_height, h_loc + h_size // 2)
            w1 = max(0, w_loc - w_size // 2)
            w2 = min(img_width, w_loc + w_size // 2)

            image_copy[h1:h2, w1:w2, :] = 0

        return image_copy