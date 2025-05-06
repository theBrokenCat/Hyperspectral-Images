# src/preprocessing.py

from hypyflow import PreprocessingPipeline
import numpy as np
import os

class Preprocessor:
    # defino el constructor
    def __init__(self, cube, preprocesschain, number, mask_path):
        self.cube = cube
        self.preprocesschain = preprocesschain
        self.number = number # para futuros usos de muchas imágenes
        self.mask_path = mask_path
        self.pre = PreprocessingPipeline(cube)

    def preprocess(self):
        try:
            if os.path.exists(self.mask_path):# and "headwall" in cube_path.lower():
                full_mask = np.load(self.mask_path)
            #elif os.path.exists(f"/home/arturo.samayor/Practicas/data/mask_{self.number}.npy") and "snapshot" in self.cube.lower():
            #    full_mask = np.load(f"/home/arturo.samayor/Practicas/data/mask_{self.number}.npy")
            else:
                full_mask = None
            masks = None
        except Exception as e:
            print(f"Error al cargar las máscaras: {e}")
            full_mask = None
            masks = None

        # Llamada corregida: eliminar self.cube como argumento posicional
        filtered_data, full_mask, masks = self.pre(
            self.preprocesschain,
            verbose=False,
            plot_masks=True,
            mask=full_mask
        )
        return filtered_data, full_mask, masks
