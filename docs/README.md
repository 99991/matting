# Installation instructions for Windows

### 1. Install Anaconda Python 3.7 64 bit (not Python 2 and not 32 bit):

https://www.anaconda.com/distribution/#download-section

![Anaconda download](https://raw.githubusercontent.com/99991/matting/master/docs/anaconda_64.png)

### 2. Install a C compiler

For example gcc from the mingw-w64 distribution.

Make sure you select the x86_64 version (not i686) during installation.

https://mingw-w64.org/doku.php/download/mingw-builds

![select 64-bit](https://raw.githubusercontent.com/99991/matting/master/docs/mingw64.png)

Find out where gcc was installed. In my case, it was installed at:

`C:\Program Files\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin`

![gcc install location](https://raw.githubusercontent.com/99991/matting/master/docs/mingw64_install_path.png)

Add the directory to your system environment path.

![add to path](https://raw.githubusercontent.com/99991/matting/master/docs/gcc_PATH.png)

Check if gcc works by opening a command prompt and running

```
gcc --version
```

![gcc --version](https://raw.githubusercontent.com/99991/matting/master/docs/gcc_version.png)

Alternatively, you can install the 64-bit version of the Tiny C Compiler
(not recommended, much slower):

http://download.savannah.gnu.org/releases/tinycc/tcc-0.9.27-win64-bin.zip

In this case, you will have to change the compiler in "setup.py" from
`compiler = "gcc"` to `compiler = "tcc"` and add the directory which contains tcc.exe to your PATH.

If you want to use Visual Studio (not recommended, rarely tested),
you will have to

- set `use_vcc = True` in setup.py
- find and run `vcvars64.bat` before running `python.setup.py install` in step 4
- install the Visual Studio compiler somehow, for example from:

https://visualstudio.microsoft.com/visual-cpp-build-tools/

### 3. Download and extract the matting repository somewhere

![Extracted directory](https://raw.githubusercontent.com/99991/matting/master/docs/unzipped.png)

### 4. Open an Anaconda Prompt from the start menu and navigate to the directory

### 5. Run

```
python setup.py install
```

### 6. Run an example

```
cd examples
python plant_example.py
```
