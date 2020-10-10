#include "example.h"
#include "numpy/arrayobject.h"
#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <Python.h>
#include <iostream>
#include <iterator>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/ocl.hpp>
#include <optional>
#include <vector>

namespace example {
namespace {

const static int kNumpyInitialized = _import_array();
const static cv::Scalar kContourColor = cv::Scalar(178, 0, 0);

// Check ndarray has certain dimensions.
#define RETURN_IF_DIM_NOT_MATCH(nd_array, n_dim)                               \
  if (PyArray_NDIM(nd_array) != n_dim) {                                       \
    PyErr_SetString(                                                           \
        PyExc_TypeError,                                                       \
        ("Number of dimensions should be " + std::to_string(n_dim)).c_str());  \
    return nullptr;                                                            \
  }

// Check if the array is uint8.
#define RETURN_IF_ARRAY_NOT_UINT8(nd_array)                                    \
  if (PyArray_TYPE(nd_array) != NPY_UBYTE) {                                   \
    PyErr_SetString(PyExc_TypeError, "Array must be in np.uint8");             \
    return nullptr;                                                            \
  }

// Convert numpy array to a batch of cv::Mat, where the first dimension is the
// batch size.
//
// The conversion has zero copy.
std::vector<cv::Mat> convert_array_to_mat_vec(PyObject *input_tensor) {
  int num_dim = PyArray_NDIM(input_tensor);
  std::vector<cv::Mat> mat_vec;

  npy_intp *dims = PyArray_DIMS(input_tensor);
  int num_mat = dims[0];

  for (int i = 0; i < num_mat; ++i) {
    if (num_dim == 4) {
      mat_vec.push_back(
          cv::Mat(dims[1], dims[2], CV_8UC3, PyArray_GETPTR1(input_tensor, i)));
    } else {
      mat_vec.push_back(
          cv::Mat(dims[1], dims[2], CV_8UC1, PyArray_GETPTR1(input_tensor, i)));
    }
  }
  return mat_vec;
}

// Draw contours directly onto the original image.
double draw_contours(const cv::Mat &predictions, int class_type,
                     cv::Mat &original_image) {

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::Mat mask = (predictions == class_type);

  cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);

  if (contours.size() == 0) {
    return 0;
  }

  double area = 0;
  for (auto it = contours.begin(); it != contours.end(); ++it) {
    area += cv::contourArea(*it);
    cv::drawContours(original_image, contours, it - contours.begin(),
                     kContourColor, 2 /*thickness*/);
  }

  return area;
}

} // namespace

PyObject *overlay_prediction_to_image(PyObject *input_image,
                                      PyObject *prediction) {

  if (!PyArray_Check(input_image) && !PyArray_Check(prediction)) {
    PyErr_SetString(PyExc_TypeError,
                    "Both input image and prediction should be np.array");
    return nullptr;
  } else {
    // input_image dim: (Batch, 3, H, W)
    RETURN_IF_DIM_NOT_MATCH(input_image, 4);
    RETURN_IF_ARRAY_NOT_UINT8(input_image);

    // input_image dim: (Batch, H, W)
    RETURN_IF_DIM_NOT_MATCH(prediction, 3);
    RETURN_IF_ARRAY_NOT_UINT8(prediction);

    std::vector<cv::Mat> input_mats = convert_array_to_mat_vec(input_image);
    const std::vector<cv::Mat> prediction_mats =
        convert_array_to_mat_vec(prediction);

    // Return a list of dict containing area of each class.
    PyObject *result = PyList_New(input_mats.size());

    for (int i = 0; i < input_mats.size(); ++i) {
      double min_class, max_class;
      cv::minMaxLoc(prediction_mats[i], &min_class, &max_class);

      PyObject *tmp_dict = PyDict_New();
      for (int j = 0; j <= max_class; ++j) {
        double area =
            draw_contours(prediction_mats[i], j /*class type*/, input_mats[i]);
        PyDict_SetItemString(tmp_dict, ("class_" + std::to_string(j)).c_str(),
                             PyFloat_FromDouble(area));
      }
      PyList_SetItem(result, i, tmp_dict);
    }

    return result;
  }
}

} // namespace example