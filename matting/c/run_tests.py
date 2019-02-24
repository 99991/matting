import os

err = os.system("gcc -Wall -Wextra -pedantic -lm tests.c -o tests")

if err != 0:
    print("Error: Failed to compile tests")

err = os.system("./tests")

if err != 0:
    print("Test failed")

os.remove("./tests")

print("tests passed")
