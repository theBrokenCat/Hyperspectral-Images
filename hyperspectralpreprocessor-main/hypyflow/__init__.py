from hypyflow.src import blocks
from hypyflow.src.toolchain import PreprocessingPipeline
from multiprocessing import current_process
from hypyflow.src.clusterers import SpatialSpectralCluster
from hypyflow.src.visualization import HyperspectralViewer
import hypyflow.src.constants as constants
import hypyflow.src.spectral_similarity_measures as distance_metrics

__version__ = "0.0.5"


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Singleton(metaclass=SingletonMeta):
    def __call__(cls, *args, **kwargs):
        # Print information
        if current_process().name == "MainProcess":
            print(
                """                     _____                                                                    _____ 
                    ( ___ )------------------------------------------------------------------( ___ )
                     |   |                                                                    |   | 
                     |   |\033[1;31m  ██░ ██▓██   ██▓ ██▓███ ▓██   ██▓  █████▒██▓     ▒█████   █     █░ \033[0m|   | 
                     |   |\033[1;31m ▓██░ ██▒▒██  ██▒▓██░  ██▒▒██  ██▒▓██   ▒▓██▒    ▒██▒  ██▒▓█░ █ ░█░ \033[0m|   | 
                     |   |\033[1;31m ▒██▀▀██░ ▒██ ██░▓██░ ██▓▒ ▒██ ██░▒████ ░▒██░    ▒██░  ██▒▒█░ █ ░█  \033[0m|   | 
                     |   |\033[1;31m ░▓█ ░██  ░ ▐██▓░▒██▄█▓▒ ▒ ░ ▐██▓░░▓█▒  ░▒██░    ▒██   ██░░█░ █ ░█  \033[0m|   | 
                     |   |\033[1;31m ░▓█▒░██▓ ░ ██▒▓░▒██▒ ░  ░ ░ ██▒▓░░▒█░   ░██████▒░ ████▓▒░░░██▒██▓  \033[0m|   | 
                     |   |\033[1;31m  ▒ ░░▒░▒  ██▒▒▒ ▒▓▒░ ░  ░  ██▒▒▒  ▒ ░   ░ ▒░▓  ░░ ▒░▒░▒░ ░ ▓░▒ ▒   \033[0m|   | 
                     |   |\033[1;31m  ▒ ░▒░ ░▓██ ░▒░ ░▒ ░     ▓██ ░▒░  ░     ░ ░ ▒  ░  ░ ▒ ▒░   ▒ ░ ░   \033[0m|   | 
                     |   |\033[1;31m  ░  ░░ ░▒ ▒ ░░  ░░       ▒ ▒ ░░   ░ ░     ░ ░   ░ ░ ░ ▒    ░   ░   \033[0m|   | 
                     |   |\033[1;31m     ░  ░░                ░                  ░         ░      ░     \033[0m|   | 
                     |   |                                                                    |   | 
                     |   |       -\033[0;34mPython\033[0m library for \033[0;34mPreprocessing Hyperspectral Images\033[0m       |   | 
                     |   |       -Version: """
                                     + str(__version__)
                                     + """                                              |   | 
                     |   |       -Contact: a.mruiz@upm.es                                     |   | 
                     |___|              \033[1;32m    May the Clean Pixels be with You!     \033[0m            |___| 
                    (_____)------------------------------------------------------------------(_____)
                    """
            )


# This Singleton strategy is just to print the logo and version of the package when it is imported
s = Singleton()
s()
