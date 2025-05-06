import os
import subprocess

def batch_process(image_names, numbers, output_dir, main_script='main_processing.py'):
    for image_name, number in zip(image_names, numbers):
        print(f"Procesando imagen: {image_name} con número: {number}")
        subprocess.run([
            'python', main_script,
            '--image_name', image_name,
            '--number', str(number),
            '--output_dir', output_dir
        ])
        print(f"Imagen {image_name} procesada y guardada.")

if __name__ == "__main__":
    # Ejemplo de uso:
    # Definir listas de nombres de imágenes y sus respectivos números
    image_names = [
        'P086#2021_04_15_09_22_02',
        'P087#2021_04_16_10_30_15',
        # Añade más nombres según sea necesario
    ]
    numbers = [1111, 1112]  # Lista de números correspondientes
    
    output_dir = "/home/arturo.samayor/Practicas/data/abundancias"
    
    batch_process(image_names, numbers, output_dir)
