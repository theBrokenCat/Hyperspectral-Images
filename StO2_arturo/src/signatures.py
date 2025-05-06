import numpy as np
from scipy.interpolate import interp1d
from hypyflow import constants as cte
"""
Este archivo contiene una clase: SpectralSigns, que sirve para manejar las firmas espectrales de Hb y HbO2, y los datos de absorción espectral, respectivamente.

SpectralSigns:
- Se encarga de manejar los datos de absorción espectral y realizar interpolaciones.
- Métodos:
    - __init__: Inicializa la clase con el archivo CSV de absorción.
    - interpolate_to_selected_wavelengths: Interpola los valores de absorción a longitudes de onda seleccionadas.
    - find_line: Encuentra la línea en un archivo que contiene un valor específico en la primera columna.
    - spectral_signs_bands: Calcula las bandas de signos espectrales en un rango de longitudes de onda.
"""



class SpectralSigns:
    def __init__(self, absorcion_csv):
        self.absorcion_csv = absorcion_csv
        self.hb_sign = "/home/arturo.samayor/Practicas/StO2_arturo/data/spectral_signs/hb_sign_reduced.txt"
        self.hbo2_sign = "/home/arturo.samayor/Practicas/StO2_arturo/data/spectral_signs/hbo2_sign_reduced.txt"

    def interpolate_to_selected_wavelengths(self, wavelengths, absorbances, selected_wavelengths):
        interpolator = interp1d(wavelengths, absorbances, kind='linear', bounds_error=False, fill_value=0)
        selected_absorbances = interpolator(selected_wavelengths)
        #print(f"\n\n[+]Interpolated absorbances at selected wavelengths: {selected_absorbances[:10]}")# esto muestra
        return selected_absorbances
    
    def find_line(self, file_path, value):
        with open(file_path) as f:
            for i, line in enumerate(f):
                columns = line.split()
                if not columns:
                    continue
                first_column = columns[0]
                if first_column == str(value) or first_column == str(int(value) - 1):
                    #print(f"Contenido: {line.strip()}")
                    #print(f"Línea: {i}")
                    return i
        return -1
    
    def spectral_signs_bands(self, lambda_min, lambda_max, pdf, headwall):
        skip_rows_abs = self.find_line(self.absorcion_csv, lambda_min)
        #print(f"skip_rows_abs: {skip_rows_abs}")
        absorption_data = np.loadtxt(self.absorcion_csv, delimiter="\t", skiprows=skip_rows_abs, usecols=(0, 1, 2), dtype=float)
        '''with open(self.absorcion_csv) as f:
            print(f"Primera fila de {self.absorcion_csv}: {f.readline()}")
            print(f"Segunda fila de {self.absorcion_csv}: {f.readline()}")
        print(f"primera fila de absorption_data: {absorption_data[0]}")'''
        
        steps = (int(lambda_max) - int(lambda_min)) / 5
        #print(f"steps: {steps}")

        if pdf:
            CAMERA_WAVELENGTHS = np.linspace(lambda_min, lambda_max, int(steps)).tolist()
        elif headwall:
            CAMERA_WAVELENGTHS = cte.HEADWALL_VNIR_WAVELENGHTS
        else:
            CAMERA_WAVELENGTHS = cte.XIMEA_SNAPSHOT_V1_WAVELENGHTS
        
        
        wavelengths = absorption_data[:, 0]
        hb_absorbances = absorption_data[:, 2]
        hbo2_absorbances = absorption_data[:, 1]
        #np.savetxt("/home/arturo.samayor/Practicas/StO2_arturo/data/spectral_signs/absorption_data.txt", absorption_data)
        
        selected_absorbances_hb = self.interpolate_to_selected_wavelengths(wavelengths, hb_absorbances, CAMERA_WAVELENGTHS)
        selected_absorbances_hbo2 = self.interpolate_to_selected_wavelengths(wavelengths, hbo2_absorbances, CAMERA_WAVELENGTHS)
        
        # hasta aquí es todo lo mismo que en el archivo longitudesOndaManual.py
        #print(f"primeras 10 absorbancias de Hb: {selected_absorbances_hb[:10]}")
        #print(f"primeras 10 absorbancias de HbO2: {selected_absorbances_hbo2[:10]}")
        
        
        np.savetxt(self.hb_sign, selected_absorbances_hb, fmt='%.8f')
        np.savetxt(self.hb_sign, selected_absorbances_hbo2, fmt='%.8f')
        return selected_absorbances_hb, selected_absorbances_hbo2
        
        