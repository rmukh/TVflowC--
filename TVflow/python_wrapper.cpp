#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "tvflow.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(tv_flow_python, m) {
    m.doc() = "TV flow computation for gray-scale images";
    m.def("derivative_index_2D", &derivative_index_2D, "Compute the indices of the derivatives of an image");
    m.def("run_TV_flow", &run_TV_flow, "Function to run the whole process of spectre estimation using TV flow");
    m.def("run_TV_flow_RGB", &run_TV_flow_RGB, "Function to run the whole process of spectre estimation using TV flow on RGB images");
    m.def("grad", &grad, "Function to compute the gradient of an image");
    m.def("tvdff", &tvdff, "single step via TV denoising");
    m.def("tvdff_fast", &tvdff_fast, "single step via TV denoising, fast version");
    m.def("tvdff_color", &tvdff_color, "single step via TV denoising on RGB images");
}
