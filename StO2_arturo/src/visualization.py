# src/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import logging
"""
Este archivo contiene la clase Visualizer, que proporciona métodos para visualizar y guardar imágenes relacionadas con
abundancias de Hb y HbO2, así como comparaciones de StO2 calculado y oficial.

Clases:
    Visualizer: Clase que contiene métodos para visualizar y guardar imágenes de abundancias y comparaciones de StO2.

Métodos:
    __init__: Inicializa la clase y configura el logging.
    plot_abundances: Genera y muestra gráficos de abundancias normalizadas de Hb y HbO2.
    plot_sto2_comparison: Genera y muestra gráficos comparativos de StO2 calculado y oficial, incluyendo el error RMSE.
    save_image: Guarda una imagen de los datos proporcionados con un título y ruta especificados.
    save_sto2_comparison: Guarda una gráfica comparativa de StO2 calculado y oficial, incluyendo el error RMSE, en la ruta especificada.
"""
class Visualizer:

    def __init__(self):
        # Configurar logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def plot_abundances(self, ab_hb, ab_hbo2, figsize=(15, 4)):
        logging.info("Iniciando visualización de abundancias.")

        plt.figure(figsize=figsize)
        epsilon = 1e-10  # Para evitar división por cero

        # Normalizar ab_hb
        '''max_ab_hb = np.max(ab_hb)
        if max_ab_hb > 0:
            ab_hb_normalized = ab_hb / (max_ab_hb + epsilon)
            logging.info(f"Abundancia Hb normalizada con max {max_ab_hb}.")
        else:
            ab_hb_normalized = np.zeros_like(ab_hb)
            logging.warning("np.max(ab_hb) es 0. Abundancia Hb normalizada establecida a cero.")'''

        plt.subplot(121)
        plt.imshow(ab_hb, cmap="jet", vmin=0)
        plt.title("Abundancia Hb")
        plt.axis('off')

        '''# Normalizar ab_hbo2
        max_ab_hbo2 = np.max(ab_hbo2)
        if max_ab_hbo2 > 0:
            ab_hbo2_normalized = ab_hbo2 / (max_ab_hbo2 + epsilon)
            logging.info(f"Abundancia HbO2 normalizada con max {max_ab_hbo2}.")
        else:
            ab_hbo2_normalized = np.zeros_like(ab_hbo2)
            logging.warning("np.max(ab_hbo2) es 0. Abundancia HbO2 normalizada establecida a cero.")'''

        plt.subplot(122)
        plt.imshow(ab_hbo2, cmap="jet", vmin=0)
        plt.colorbar(label="Levels")
        plt.title("Abundancia HbO2")
        plt.axis('off')

        plt.show()
        logging.info("Visualización de abundancias completada.")

    def plot_sto2_comparison(self, sto2, s_to2, error, figsize=(15, 4)):
        logging.info("Iniciando visualización de comparación de StO2.")

        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.imshow(sto2, cmap="jet", vmin=0)
        plt.title("StO2 Calculado")
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(s_to2, cmap="jet", vmin=0)
        plt.colorbar(label="Levels")
        plt.title("StO2 Oficial")
        plt.figtext(0.5, 0.02, f'Error (RMSE): {str(error)}', ha='center', va='center', fontsize=12, color='black')
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show()
        logging.info("Visualización de comparación de StO2 completada.")

    def save_image(self, data, title, path, cmap="jet", vmin=0, vmax=None):
        plt.figure()
        plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.colorbar()
        plt.axis('off')
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()
        logging.info(f"Imagen '{title}' guardada en: {path}")

    def save_sto2_comparison(self, sto2, s_to2, error, path, cmap="jet", vmin=0, vmax=1, figsize=(15, 4)):
        logging.info("Guardando gráfica de comparación de StO2.")

        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.imshow(sto2, cmap=cmap, vmin=vmin)
        plt.colorbar(label="Levels")
        plt.title("StO2 Calculado")
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(s_to2, cmap=cmap, vmin=vmin)
        plt.colorbar(label="Levels")
        plt.title("StO2 Oficial")
        plt.figtext(0.5, 0.02, f'Error (RMSE): {str(error)}', ha='center', va='center', fontsize=12, color='black')
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()
        logging.info(f"Gráfica de comparación de StO2 guardada en: {path}")
