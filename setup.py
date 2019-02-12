import os
from setuptools import setup, find_packages
from setuptools.command.install import install
import shutil

def load_text(path):
    with open(path) as f:
        return f.read()

# load information about package
path = os.path.join(os.path.dirname(__file__), "matting", "__about__.py")
about = {}
exec(load_text(path), about)

class InstallLibmatting(install):
    def run(self):
        print("building libmatting.so")
        os.chdir("matting/c")
        os.system("make")
        os.chdir("../..")
        
        install.run(self)
        
        print("cleanup")
        os.chdir("matting/c")
        os.remove("libmatting.so")
        os.chdir("../..")
        for directory in [
            "build",
            "matting.egg-info",
        ]:
            shutil.rmtree(directory, ignore_errors=True)

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
    package_data={"matting": ["c/libmatting.so"]},
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
