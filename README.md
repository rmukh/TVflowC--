# TVflowC--
Total Variation (TV) flow implementation in C/C++
## Download
git clone --recursive https://github.com/rmukh/TVflowC--.git
## Pre-requestied
Eigen3 library. It must be 'installed' first before buidling or running python installation (it invokes building process).
[The simplies instruction](https://robots.uc3m.es/installation-guides/install-eigen.html)

## Building
The simples procedure to build the project after clonning is:

```cd TVflowC--
mkdir build
cd build
cmake .. *optionally might require* -DEigen3_DIR=path/to/installed/eigen3
cmake --build . --config Release -j #number_of_threads
cmake install --build
```

## Installation
You can install python package as 
```
cd TVflowC--
python -m pip install .
```
It will automatically invoke building process

## Examples
You can learn the most basic usage example in examples/main_gray_test.py

It requres: matplotlib and Pillow libraries to be installed