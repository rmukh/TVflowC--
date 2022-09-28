Total Variation (TV) flow implementation in C/C++
## Download
```
git clone --recursive https://github.com/rmukh/TVflowC--.git
```
## Pre-requisites
Eigen3 library. It must be 'installed' before building or running python installation (it invokes the building process).
[The simplies instruction](https://robots.uc3m.es/installation-guides/install-eigen.html)

## Installation
You can install the python package as 
```
cd TVflowC--
python -m pip install .
```
It will automatically invoke the building process.

## Building
The simples procedure to build the project as a shared library with python build is:

```
cd TVflowC--
mkdir build
cd build
cmake .. *optionally might require* -DEigen3_DIR=path/to/installed/eigen3
cmake --build . --config Release -j #number_of_threads
cmake install --build
```

## Examples
You can learn the most basic usage example in examples/main_gray_test.py

It requires: matplotlib and Pillow libraries to be installed

More details you can find at [https://rinatm.com/total-variation-flow-implementation-in-c-c/](https://rinatm.com/total-variation-flow-implementation/)
