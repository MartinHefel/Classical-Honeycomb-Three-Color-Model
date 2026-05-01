import sys
import importlib

packages = [
    {"name": "python",        "code_version": "3.12.3"},
    {"name": "matplotlib",    "code_version": "3.10.6"},
    {"name": "numpy",         "code_version": "2.3.3"},
    {"name": "numba",         "code_version": "0.62.1"},
    {"name": "scipy",         "code_version": "1.17.0"},
    {"name": "tensornetwork", "code_version": "0.4.6"},
    {"name": "jax",           "code_version": "0.9.1"},
]

print("\nPackage".ljust(16), "|", "Tested with".ljust(15), "|", "Installed".ljust(15))
print("-" * 49)

for pkg in packages:
    name = pkg["name"]
    code_version = pkg["code_version"]

    if name == "python":
        installed = sys.version.split()[0]
    else:
        try:
            module = importlib.import_module(name)
            installed = module.__version__
        except ImportError:
            installed = "missing"
        if name == "jax":
            name += "*"

    print(name.ljust(15), "|", code_version.ljust(15), "|", str(installed).ljust(15))

print("\n*optional\n")
