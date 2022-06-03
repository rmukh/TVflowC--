#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "tvflow.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(tv_flow_python, m)
{
    m.doc() = R"pbdoc(
        TV flow computation for gray-scale and color images
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           run_TV_flow
           run_TV_flow_RGB
           grad
           tvdff
           tvdff_color
    )pbdoc";
    m.def("run_TV_flow", &run_TV_flow, R"pbdoc(Function to run the whole process of spectre estimation using TV flow)pbdoc");
    m.def("run_TV_flow_RGB", &run_TV_flow_RGB, R"pbdoc(Function to run the whole process of spectre estimation using TV flow on RGB images)pbdoc");
    m.def("grad", &grad, R"pbdoc(Function to compute the gradient of an image)pbdoc");
    m.def("tvdff", &tvdff, R"pbdoc(single step via TV denoising)pbdoc");
    m.def("tvdff_color", &tvdff_color, R"pbdoc(single step via TV denoising on RGB images)pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
