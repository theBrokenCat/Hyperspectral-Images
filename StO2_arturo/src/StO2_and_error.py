from scipy import ndimage
import numpy as np

class StO2:
    def __init__(self, ab_hb, ab_hbo2, mask):
        self.ab_hb = ab_hb
        self.ab_hbo2 = ab_hbo2
        self.mask = mask
    
    def calculate_sto2(self):

        ab_hbo2 = self.ab_hbo2/self.ab_hbo2.max()
        ab_hb = self.ab_hb/self.ab_hb.max()
    
        epsilon = 1e-10
        sto2 = ((ab_hbo2) / (ab_hb + ab_hbo2+epsilon))

        sto2 = ndimage.median_filter(sto2, size=3)

        sto2 = np.nan_to_num(sto2, nan=0, posinf=0, neginf=0)
        print("tipo de dato de sto2: ", sto2.dtype)
        return sto2
        
class Error:
    def __init__(self, sto2, sto2_official, full_mask):
        self.sto2 = sto2
        self.sto2_official = sto2_official
        self.full_mask = full_mask
        
    def _apply_mask_2d(self, img, mask):
        masked_data = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                masked_data[i, j] = np.where(mask[i, j], img[i, j], np.nan)
        return masked_data

    def _mse(self, img1, img2):
        # Convertir a tipo float
        arr1 = img1.astype(float)
        arr2 = img2.astype(float)
        
        # Crear máscara para ignorar NaN
        mask = ~np.isnan(arr1) & ~np.isnan(arr2)
        
        # Si no hay píxeles válidos, devolver NaN
        if not np.any(mask):
            return np.nan
        
        # Calcular el MSE sólo en los píxeles válidos
        err = np.sum((arr1[mask] - arr2[mask]) ** 2)
        err /= float(np.count_nonzero(mask))  # Normalizar por el número de píxeles válidos
        
        return err

    def _RMSE(self, img1, img2):
        # Cálculo del RMSE usando la función mse modificada
        err = self._mse(img1, img2)
        if np.isnan(err):
            return np.nan
        
        return np.sqrt(err)

    def _PSNR(self, img1, img2, max_value=1):
        # Convertir las imágenes a float32
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)
        
        # Crear una máscara que ignore los NaN
        # Esta máscara será True sólo donde no hay NaN en ninguna de las dos imágenes
        mask = ~np.isnan(arr1) & ~np.isnan(arr2)
        
        # Si no hay elementos válidos, no se puede calcular el PSNR
        if not np.any(mask):
            return float('nan')
        
        # Calcular el MSE sólo en los píxeles válidos
        mse = np.mean((arr1[mask] - arr2[mask]) ** 2)
        
        # Si el MSE es cero, significa que las imágenes son idénticas en la región válida
        if mse == 0:
            return 100.0
        
        # Calcular el PSNR
        psnr = 20 * np.log10(max_value / np.sqrt(mse))
        return psnr
    
    def calculate_error(self):
        sto2ar = self._apply_mask_2d(self.sto2, self.full_mask)
        s_to2ar = self._apply_mask_2d(self.sto2_official, self.full_mask)
        errorRMSE = self._RMSE(sto2ar, s_to2ar)
        errorPSNR = self._PSNR(sto2ar, s_to2ar)
        return errorPSNR

    