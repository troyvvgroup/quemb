#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>

namespace py = pybind11;

// Function that sums all elements of a 2D NumPy array
auto sum_array(py::array_t<double> array) {
    // Request buffer info from NumPy array
    py::buffer_info buf = array.request();

    // Ensure it's 2D
    if (buf.ndim != 2)
        throw std::runtime_error("Input must be a 2D array");

    // Map the data to an Eigen matrix
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(
        static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1]);

    // Use Eigen to compute the sum
    return mat.sum();
}



// Binding code
PYBIND11_MODULE(mymodule, m) {
    m.doc() = "Minimal pybind11 + Eigen example";
    m.def("sum_array", &sum_array, "Sum all elements of a 2D NumPy array");
}
