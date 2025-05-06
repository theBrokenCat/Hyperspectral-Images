import numpy as np
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from tqdm.auto import trange

class AbundanceCalculator:
    def __init__(self, hb_sign, hbo2_sign, masked_cube):
        self.hb_sign = hb_sign
        self.hbo2_sign = hbo2_sign
        self.masked_cube = masked_cube
        print(masked_cube.shape)
        print(hb_sign.shape)
        print(hbo2_sign.shape)
        
        self.height, self.width, self.bands = masked_cube.shape
        self.ab_hb = np.zeros((self.height, self.width))
        self.ab_hbo2 = np.zeros((self.height, self.width))

    def residuals(self, x, pixel, hb_sign, hbo2_sign):
        ab_hb, ab_hbo2 = x
        model = ab_hb * hb_sign + ab_hbo2 * hbo2_sign
        return pixel - model

    def calculate_abundances(self, n_jobs=40):
        def process_pixel(i, j):
            pixel = self.masked_cube[i, j, :]
            if pixel.sum() == 0:
                return i, j, [0, 0]
            initial_guess = [0.01, 0.01]
            result = least_squares(
                self.residuals,
                initial_guess,
                args=(pixel, self.hb_sign, self.hbo2_sign),
                bounds=([0, 0], [1, 1]),
                x_scale='jac',
                loss='soft_l1'
            )
            return i, j, result.x

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_pixel)(i, j) for i in trange(self.height) for j in range(self.width)
        )

        for i, j, abundances in results:
            self.ab_hb[i, j], self.ab_hbo2[i, j] = abundances

        return self.ab_hb, self.ab_hbo2
