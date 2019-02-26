import os
from setuptools import setup, find_packages
from setuptools.command.install import install
import shutil
import sys

compiler = "gcc"

# If you want to compile with Visual Studio, find the location of
# vcvars64.bat and run it before running "python setup.py install"
# and set use_vcc to True.
use_vcc = False

src = """

ichol.c
knn.c
kdtree.c
boxfilter.c
labelexpand.c

""".replace("\n", " ")

flags = "-O3 -Wall -Wextra -pedantic -shared"

compile_commands = {
    "win32": "%s %s %s -o libmatting.dll"%(compiler, flags, src),
    "linux": "%s %s %s -fPIC -lm -o libmatting.so"%(compiler, flags, src),
}

# Additionally, uncomment the next line:
if use_vcc:
    compile_commands["win32"] = "cl /LD /O2 /Felibmatting.dll %s"%src

def load_text(path):
    with open(path) as f:
        return f.read()

# load information about package
path = os.path.join(os.path.dirname(__file__), "matting", "__about__.py")
about = {}
exec(load_text(path), about)

def cleanup():
    print("cleanup")
    os.chdir("matting/c")
    try:
        os.remove("libmatting.so")
    except OSError:
        pass
    os.chdir("../..")
    for directory in [
        "build",
        "matting.egg-info",
    ]:
        shutil.rmtree(directory, ignore_errors=True)

class InstallLibmatting(install):
    def run(self):
        if sys.platform not in compile_commands:
            raise Exception("%s platform not supported"%sys.platform)
        
        compile_command = compile_commands[sys.platform]
        
        print("building libmatting library")
        os.chdir("matting/c")
        
        err = os.system(compile_command)
        
        os.chdir("../..")
        
        if err:
            cleanup()
            raise Exception("Failed to compile libmatting")
        
        install.run(self)
        
        cleanup()
setup(
    name=about["__title__"],
    version=about["__version__"],
    url=about["__uri__"],
    author=about["__author__"],
    author_email=about["__email__"],
    description=about["__summary__"],
    long_description=load_text("README.md"),
    long_description_content_type="text/markdown",
    license=about["__license__"],
    packages=find_packages(),
    package_data={"matting": ["c/libmatting.so", "c/libmatting.dll"]},
    install_requires=load_text("requirements.txt").strip().split("\n"),
    keywords='alpha matting',
    python_requires='>=3',
    cmdclass={'install': InstallLibmatting},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: Linux",
    ],
    project_urls={
        "Source": about["__uri__"],
    }
)
