from distutils.core import setup, Extension
import os
from subprocess import PIPE
import numpy as np
import subprocess
import sys
import re


def _find_val_by_key_from_list(str_list, key):
    """Parse a list of string to extract key value pairs."""

    # match pattern "key : (value)", where both key and value are strings.
    pattern = re.compile(r"""{}\s:\s([\w|\/]+)""".format(key))
    for s in str_list:
        result = pattern.search(s)
        if result is not None:
            return result.groups()[0]

    return None


def _parse_conda_info(conda_info_byte):
    """Parse string returned by running \"conda info\""""
    conda_info_str = conda_info_byte.decode('utf-8')
    conda_info_list = conda_info_str.split('\n')
    print(conda_info_list)
    active_env = _find_val_by_key_from_list(conda_info_list,
                                            "active environment")
    active_env_location = _find_val_by_key_from_list(conda_info_list,
                                                     "active env location")
    if active_env == 'None' or active_env is None:
        raise RuntimeError("Execute the script in a valid conda environment.")

    return active_env, active_env_location


def _get_conda_env_info():
    """conda environment name and location.f"""
    try:
        conda_info_byte = subprocess.run(["conda", "info"],
                                         check=True,
                                         stdout=subprocess.PIPE).stdout
        env_name, env_location = _parse_conda_info(conda_info_byte)
    except Exception as e:
        print("Failed to get conda information due to {}".format(e))
        raise e

    return env_name, env_location


def _get_python_version_str():
    """Get Python version."""
    info = sys.version_info
    return '{}.{}'.format(info.major, info.minor)


def _get_include_dirs(env_location, py_version):
    include_dirs = []

    # opencv include files
    include_dirs.append("{}/include/opencv4".format(env_location))

    # Python.h
    include_dirs.append("{}/include/python{}m".format(env_location,
                                                      py_version))

    include_dirs.append(np.get_include())

    # header files in the current directory
    include_dirs.append(os.getcwd())

    return include_dirs


env_name, env_location = _get_conda_env_info()
py_version = _get_python_version_str()

example_util_module = Extension(
    "example_utils", ['example.cc', 'interface.cc'],
    include_dirs=_get_include_dirs(env_location, py_version),
    extra_compile_args=["-std=c++17", '-Wno-undef', '-fopenmp'],
    extra_link_args=[
        "-lpython{}m".format(py_version), "-lopencv_core", "-lopencv_imgproc",
        "-lopencv_imgcodecs", '-lgomp'
    ])

setup(name='example_utils', version="0.1", ext_modules=[example_util_module])
