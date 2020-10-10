#include "example.h"
#include "numpy/arrayobject.h"
#include "numpy/ndarrayobject.h"
#include "opencv2/imgcodecs.hpp"
#include <Python.h>
#include <iostream>
#include <opencv2/core/ocl.hpp>

namespace {
using ::example::overlay_prediction_to_image;
}

static PyObject *run_example(PyObject *self, PyObject *args) {
  PyObject *input_image = nullptr;
  PyObject *prediction = nullptr;

  // The input contains two objects.
  if (not PyArg_ParseTuple(args, "OO", &input_image, &prediction)) {
    PyErr_SetString(PyExc_TypeError, "input arguments are not pyobject");
  }
  PyObject *res = overlay_prediction_to_image(input_image, prediction);
  return Py_BuildValue("O", res);
}

static PyMethodDef example_methods[] = {
    {"run_example" /*method name*/, run_example /*function pointer*/,
     METH_VARARGS /*variable argument functions*/},
    {NULL, NULL, 0, NULL} /* ending indicator */

};

static PyModuleDef example_utils_modules = {
    PyModuleDef_HEAD_INIT,
    "example_utils",
    "Example of interfacing Python and C++, with numpy and opencv",
    -1,
    example_methods,
};

PyMODINIT_FUNC PyInit_example_utils(void) {
  import_array();
  // From:
  // https://answers.opencv.org/question/123990/why-is-the-first-opencv-api-call-so-slow/
  // disable opencl so that the first opencv call does not compile opencl code.
  cv::ocl::setUseOpenCL(false);

  return PyModule_Create(&example_utils_modules);
}
