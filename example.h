#ifndef EXAMPLE_H_
#define EXAMPLE_H_

#include <Python.h>

namespace example {

PyObject *overlay_prediction_to_image(PyObject *input_image,
                                      PyObject *prediction);

} // namespace example

#endif // EXAMPLE_H_