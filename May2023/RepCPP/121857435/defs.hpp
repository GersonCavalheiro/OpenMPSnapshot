#pragma once

#include <Eigen/Dense>
#include <gsl/gsl_randist.h>
#include <unsupported/Eigen/CXX11/Tensor>


#ifdef BUILD_TESTING
#include <stdexcept>
#define BNMF_ASSERT(condition, msg)                                            \
if (not(condition))                                                        \
throw std::invalid_argument(msg)
#else
#include <cassert>
#define BNMF_ASSERT(condition, msg) (assert((condition) && msg))
#endif

namespace bnmf_algs {

template <typename Scalar>
using vector_t = Eigen::Matrix<Scalar, 1, Eigen::Dynamic, Eigen::RowMajor>;


using vectord = vector_t<double>;


template <typename Scalar>
using matrix_t =
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


using matrixd = matrix_t<double>;


template <typename Scalar, size_t N>
using tensor_t = Eigen::Tensor<Scalar, N, Eigen::RowMajor>;


template <size_t N> using tensord = tensor_t<double, N>;


template <size_t N> using shape = Eigen::array<size_t, N>;

namespace util {


using gsl_rng_wrapper = std::unique_ptr<gsl_rng, decltype(&gsl_rng_free)>;

} 
} 
