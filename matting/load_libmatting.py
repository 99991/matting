import pkg_resources
from ctypes import CDLL

def load_libmatting():
    # Content of python *.egg package files are extracted into
    # a temporary directory.
    # Get path to libmatting.so in temporary directory.
    library_path = pkg_resources.resource_filename('matting', 'c/libmatting.so')

    # If you want to use a library you compiled yourself,
    # here would be the place to set its path:
    
    #library_path = "path/of/your/libmatting.so

    library = CDLL(library_path)

    return library
