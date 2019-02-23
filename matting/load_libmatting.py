import pkg_resources
from ctypes import CDLL

def load_libmatting():
    # Content of python *.egg package files are extracted into
    # a temporary directory.
    # Get path to libmatting in temporary directory.
    
    # If you want to use a library you compiled yourself,
    # here would be the place to set add path.
    
    paths = [
        pkg_resources.resource_filename('matting', 'c/libmatting.dll'),
        pkg_resources.resource_filename('matting', 'c/libmatting.so'),
    ]
    
    for library_path in paths:
        try:
            return CDLL(library_path)
        except OSError:
            pass

    raise Exception("Failed to find libmatting library. Checked paths:\n%s", "\n".join(paths))
