import os
from setuptools import setup, find_packages
from setuptools.command.install import install
import shutil

src = """

ichol.c
knn.c
kdtree.c
boxfilter.c
labelexpand.c

""".replace("\n", " ")

flags = "-O3 -Wall -Wextra -pedantic -shared"

commands = [
    # Windows
    "gcc %s %s -o libmatting.dll"%(flags, src),
    # Linux
    "gcc %s %s -fPIC -lm -o libmatting.so"%(flags, src),
]

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

def try_to_compile():
    for command in commands:
        err = os.system(command)
        
        if err == 0:
            print('Command succeeded:\n"%s"'%command)
            return err
        
        print('Command failed (that is ok if a later comand succeeds):\n%s'%command)
    
    return err

class InstallLibmatting(install):
    def run(self):
        print("building libmatting library")
        os.chdir("matting/c")
        
        err = try_to_compile()
        
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
