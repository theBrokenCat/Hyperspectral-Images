import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.data_loader import DataLoader
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.signatures import SpectralSigns
from src.abundance_calculator import AbundanceCalculator
from src.visualization import Visualizer
from src.AI_comments import AI_commenter
from src.StO2_and_error import StO2, Error
from src.utils import from_reflectance_to_absorbance, normalize_signatures
from hypyflow import blocks
from htc import DataPath
import numpy as np
from src.utils import apply_mask
from scipy import ndimage
import matplotlib.pyplot as plt
from src.extras import ProcessChains
from datetime import datetime

def main(image_name, number, output_dir):
    # Inicializar DataLoader
    data_loader = DataLoader(base_path="/home/arturo.samayor/Practicas/StO2_arturo/data/Hiperspectral_images/HeiPorSPECTRAL_example")
    
    # Cargar cubo hiperespectral
    cube = data_loader.load_cube(image_name)
    
    # Definir la cadena de preprocesamiento
    chains = [ProcessChains.pc1, ProcessChains.pc2, ProcessChains.pc3, ProcessChains.pc4, ProcessChains.pc5, ProcessChains.pc6]
    
    # Cargar máscara existente si existe
    mask_path = f"/home/arturo.samayor/Practicas/StO2_arturo/data/masks/mask_1.npy"
    #mask_path_snapshot = f"/home/arturo.samayor/Practicas/StO2_arturo/data/masks/mask{number}.npy"
    
    #try:
    #    if os.path.exists(mask_path):
    #       full_mask = data_loader.load_mask(mask_path)
    #    elif os.path.exists(mask_path_snapshot):
    #        full_mask = data_loader.load_mask(mask_path_snapshot)
    #    else:
    #        full_mask = None
    #except:
    #    full_mask = None
    
    '''# Preprocesar el cubo
    filtered_data, full_mask, masks = preprocessor.preprocess()
    
    # Guardar la máscara
    np.save(mask_path, full_mask)'''
    
    # Calcular absorbancia y aplicar máscara
    '''filtered_data_absorbance = from_reflectance_to_absorbance(filtered_data)
    masked_cube = apply_mask(filtered_data_absorbance, full_mask)'''
    
    valores_minimos = [500, 600, 650, 700]
    valores_maximos = [600, 1000, 1000, 1000]
    
    spectral_signs = SpectralSigns("/home/arturo.samayor/Practicas/StO2_arturo/data/spectral_signs/extintionCoefficient/coeficiente_absorcion.csv")
    
    
    
    
    
    # BUCLE PARA DISTINTAS LONGITUDES DE ONDA
    for regulaciones in chains:
        cube_ = cube
        
        # ============================================================== #
        # [+] Preprocesamiento
        # ============================================================== #
        preprocesschain = regulaciones # Seleccionar una de las cadenas de preprocesamiento
        # Inicializar Preprocessor
        preprocessor = Preprocessor(cube_, preprocesschain, number, "/home/arturo.samayor/Practicas/StO2_arturo/data/masks/mask_1.npy")
        # Preprocesar el cubo
        filtered_data, full_mask, masks = preprocessor.preprocess()
        # Guardar la máscara
        np.save(mask_path, full_mask)
        
        # ============================================================== #
        # [+] Procesado de cubo hiperespectral
        # ============================================================== #
        filtered_data_absorbance = from_reflectance_to_absorbance(filtered_data)
        masked_cube = apply_mask(filtered_data_absorbance, full_mask)
        
        for aux in range(valores_minimos.__len__()):
            masked_cube_new = masked_cube # Creo una copia del cubo para no modificar el original
            # Defino intervalos (valores mínimos y máximos)
            intervals = [(valores_minimos[aux], valores_maximos[aux])]
            # Defino rango de longitudes de onda
            a = int((valores_minimos[aux]-500)/5)
            b = int((valores_maximos[aux]-500)/5)
            print(f"Rango de longitudes de onda: {a} - {b}\n")
            
            
            masked_cube_new = masked_cube_new[:, :, a:b]
            # This block of code is printing out information about the cube data before and after some
            # processing steps. Here's what each line does:
            print(f"[+] Valores del Cubo")
            print(f"\t[-] El valor máximo del cubo es: {np.max(masked_cube_new)}")
            print(f"\t[-] La media del cubo es: {np.mean(masked_cube_new)}")
            masked_cube = masked_cube/np.max(masked_cube)
            print(f"[+] Valores del Cubo normalizados")
            print(f"\t[-] El valor máximo del cubo es: {np.max(masked_cube)}")
            print(f"\t[-] La media del cubo es: {np.mean(masked_cube)}")
            
            #print(f"masked_cube_new shape: {masked_cube_new.shape}")
            
            # Calcular abundancias
            hb_sign, hbo2_sign = spectral_signs.spectral_signs_bands(valores_minimos[aux], valores_maximos[aux], pdf=True, headwall=False)
            '''print(f"[+] hb_sign shape: {hb_sign.shape}")
            print(f"[+] hbo2_sign shape: {hbo2_sign.shape}")
            print(f"[+] media de hb_sign: {np.mean(hb_sign)}")
            print(f"[+] media de hbo2_sign: {np.mean(hbo2_sign)}")'''
            
            hb_sign_norm, hbo2_sign_norm = normalize_signatures(hb_sign, hbo2_sign)
            #print(f"[+] media de hb_sign_norm: {np.mean(hb_sign_norm)}")
            #print(f"[+] media de hbo2_sign_norm: {np.mean(hbo2_sign_norm)}")
            
            # ============================================================== #
            # [+] Calculo de abundancias
            # ============================================================== #
            abundance_calculator = AbundanceCalculator(
                hb_sign=hb_sign_norm,
                hbo2_sign=hbo2_sign_norm,
                masked_cube=masked_cube_new
            )
            ab_hb, ab_hbo2 = abundance_calculator.calculate_abundances(n_jobs=40)
            print(f"media de ab_hb: {np.mean(ab_hb)}")
            print(f"media de ab_hbo2: {np.mean(ab_hbo2)}")
            # Calcular sto2
            '''epsilon = 1e-10 ANTIGUO*****************************
            sto2 = ((ab_hbo2) / (ab_hb + ab_hbo2+epsilon))
            sto2 = ndimage.median_filter(sto2, size=3)
            # Comparar con sto2 oficial
            s_to2 = DataPath.from_image_name(image_name)
            s_to2 = s_to2.compute_sto2()
            s_to2 = np.where(s_to2 == "--", 0, s_to2)
            error = RMSE(sto2, s_to2)'''
            s_to2 = DataPath.from_image_name(image_name)
            s_to2 = s_to2.compute_sto2()
            s_to2 = np.where(s_to2 == "--", 0, s_to2)
            
            sto2_ = StO2(ab_hb, ab_hbo2, full_mask)
            sto2 = sto2_.calculate_sto2()
            error = Error(sto2, s_to2, full_mask).calculate_error()
            
            # Visualizar resultados
            visualizer = Visualizer()
            #visualizer.plot_abundances(ab_hb, ab_hbo2)
            #visualizer.plot_sto2_comparison(sto2, s_to2, error)
            
            # Guardar abundancias e imágenes en la carpeta output_dir, comprobar si dentro existe el directorio llamado
            # Directorio_regulacion_x, si no existe, crearlo y si existe guardar las imágenes en ese directorio
            output_dir_ = os.path.join(output_dir, f"Directorio_regulacion_{chains.index(regulaciones)}")
            if not os.path.exists(output_dir_):
                try:
                    os.mkdir(output_dir_)  # os.mkdir() solo crea el último directorio, no los padres.
                    print(f"Directorio creado: {output_dir_}")
                except OSError as e:
                    print(f"Error al crear el directorio {output_dir_}: {e}")
                else:
                    print(f"El directorio ya existe: {output_dir_}")
            print (f"output_dir: {output_dir_}")
            
            
            visualizer.plot_sto2_comparison(sto2, s_to2, error)
            
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            visualizer.save_sto2_comparison(
            sto2=sto2,
            s_to2=s_to2,
            error=error,
            # guardo la imagen con el nombre de valores mínimos y máximos, seguidos de la hora y fecha creados
            
            path = os.path.join(output_dir_, f"{valores_minimos[aux]}-{valores_maximos[aux]}_{current_time}.png"),
            #path=os.path.join(output_dir_, f"{valores_minimos[aux]}-{valores_maximos[aux]}.png"),
            cmap="jet",
            vmin=0,
            vmax=1
            )
            
            comment_path = os.path.join(output_dir_, f"{valores_minimos[aux]}-{valores_maximos[aux]}.txt")
            print(f"Imágenes guardadas en la carpeta: {comment_path}")
            # Comentarios
            
            AI_commenter(comment_path).main()
            
    
    

if __name__ == "__main__":
    import argparse

    '''parser = argparse.ArgumentParser(description="Procesamiento de imágenes STO2")
    parser.add_argument('--image_name', type=str, required=True, help='Nombre único de la imagen')
    parser.add_argument('--number', type=int, required=True, help='Número de identificación de la imagen')
    parser.add_argument('--output_dir', type=str, default="/home/arturo.samayor/Practicas/StO2_arturo/data/StO2_comparaciones/Directorio_regulacion_1", help='Directorio para guardar los resultados')
    
    args = parser.parse_args()'''
    
    main('P086#2021_04_15_09_22_02', 1111, "/home/arturo.samayor/Practicas/StO2_arturo/data/StO2_comparaciones")
