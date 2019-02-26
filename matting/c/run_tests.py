import os
import sys

binary_path = "./tests"

if sys.platform == "win32":
    binary_path = "tests.exe"

err = os.system("gcc -Wall -Wextra -pedantic -lm tests.c -o " + binary_path)

if err == 0:
    print("Tests compiled")
else:
    print("Error: Failed to compile tests")
    sys.exit(-1)

err = os.system(binary_path)

if err == 0:
    print("All tests passed")
else:
    print("Test failed")

os.remove(binary_path)
