#!/usr/bin/env python3
import os
import pty
import re
import select
import sys
import time
import requests

class AI_commenter:
    
    def __init__(self, directory):
        self.directory = directory
        
    
    def _message(self):
        # Leer ambos archivos con el código a comparar
        with open("/home/arturo.samayor/Practicas/StO2_arturo/src/abundance_calculator.py", "r") as file:
            first_code = file.read()
        with open("/home/arturo.samayor/Practicas/StO2_arturo/extras/abundance_calculator-BASE.py", "r") as file:
            second_code = file.read()
        
        # Construir el mensaje con las instrucciones y el contenido de ambos archivos
        message = (
            "Forget everything you know. You are a bullet-list generator. "
            "Your output must contain ONLY a bullet list (each item starting with '-') of the differences between the following two pieces of code, method by method. "
            "Do not include any introductory text, explanations, commentary, meta information, or any markers (such as <think>). "
            "Output NOTHING except the bullet list of differences. If a method has no differences, output nothing for that method.\n\n"
            "First code:\n"
            f"\"{first_code}\"\n\n"
            "Second code:\n"
            f"\"{second_code}\"\n\n"
        )



        #print("Constructed message:")
        #print(message)
        return message

    def _send_message(self, message):
        # Endpoint y headers para la petición POST
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        
        # Payload con el mensaje, el modelo y sin streaming
        payload = {
            "model": "deepseek-r1:14b",
            "prompt": message,
            "stream": False
        }
        
        response = requests.post(url, json=payload, headers=headers)
        return response
    
    def _create_save_output(self, message):
            # guardamos el mensaje en un archivo ubicado en el directorio directory
            print(message)
            print("Saving output to the file..."+self.directory)
            with open(self.directory, "w") as file:
                file.write(message)

    def main(self):
        message = self._message()
        #print("Sending POST request to Ollama API...")
        response = self._send_message(message)
        
        if response.status_code == 200:
            # Como stream es false, la respuesta es un único objeto JSON
            data = response.json()
            final_message = data.get("response", "")
            # Elimina todo lo que esté entre <think> y </think> (incluyendo las etiquetas)
            final_message = re.sub(r"<think>.*?</think>", "", final_message, flags=re.DOTALL)
            final_message = final_message.strip()
            # al principioi de final_message añado "Modified things:"
            final_message = "Modified things:\n"+final_message
            self._create_save_output(final_message)
            print("\nModified things:")
            print(final_message)
        else:
            print("Error:")
            print(response.status_code, response.text)
