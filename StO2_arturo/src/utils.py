import numpy as np

def normalize_signatures(hb_sign, hbo2_sign):
    #hb_sign es la segunda columna del archivo hb_sign_reduced.txt
    #hbo2_sign es la tercera columna del archivo hbo2_sign_reduced.txt
    # los cargo
    #hb_sign = np.loadtxt(hb_sign)
    #hbo2_sign = np.loadtxt(hbo2_sign)
    
    if np.max(hb_sign) > np.max(hbo2_sign):
        hb_sign = hb_sign / np.max(hb_sign)
        hbo2_sign = hbo2_sign / np.max(hb_sign)
    else:
        hb_sign = hb_sign / np.max(hbo2_sign)
        hbo2_sign = hbo2_sign / np.max(hbo2_sign)
        
    # selecciono las segundas columnas
    return hb_sign, hbo2_sign

def from_reflectance_to_absorbance(data_reflectance):
    epsilon = 1e-10  # Un valor pequeño para evitar el logaritmo de cero
    reflectance = np.clip(data_reflectance, epsilon, np.max(data_reflectance))  # Recorta los valores entre epsilon y 1
    absorbance = -np.log10(reflectance)
    return absorbance

def apply_mask(data, mask):
    masked_data = np.zeros_like(data)
    for i in range(data.shape[2]):
        masked_data[:, :, i] = np.where(mask, data[:, :, i], 0)
    return masked_data

def always_positive(data):
    return data + np.abs(np.min(data))




