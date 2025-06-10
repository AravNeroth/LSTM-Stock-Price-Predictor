import sys
print("Python version: ", sys.version)
print("Directory where python files are installed: ", sys.base_prefix)
print("Directory of virtual environment(if any): ", sys.prefix)
print("Location of python executable: ", sys.executable)
print("Path to libraries: ", *sys.path, sep="\n\t")