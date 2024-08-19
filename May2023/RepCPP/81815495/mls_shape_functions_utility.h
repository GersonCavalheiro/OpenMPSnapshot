
#pragma once



#include "containers/array_1d.h"
#include "includes/define.h"
#include "includes/ublas_interface.h"

namespace Kratos
{


class MLSShapeFunctionsUtility
{

public:


static double CalculateKernel(
const array_1d<double,3>& rRadVect,
const double h);


template<std::size_t TDim>
static void CalculateKernelDerivative(
const array_1d<double,3>& rRadVect,
const double h,
array_1d<double,TDim>& rKernelDerivative);


static void EvaluatePolynomialBasis(
const array_1d<double,3>& rX,
array_1d<double, 3>& rBasis);


static void EvaluatePolynomialBasis(
const array_1d<double,3>& rX,
array_1d<double, 4>& rBasis);


static void EvaluatePolynomialBasis(
const array_1d<double,3>& rX,
array_1d<double, 6>& rBasis);


static void EvaluatePolynomialBasis(
const array_1d<double,3>& rX,
array_1d<double, 10>& rBasis);


static void EvaluatePolynomialBasisDerivatives(
const array_1d<double,3>& rX,
BoundedMatrix<double, 2, 3>& rBasisDerivatives);


static void EvaluatePolynomialBasisDerivatives(
const array_1d<double,3>& rX,
BoundedMatrix<double, 3, 4>& rBasisDerivatives);


static void EvaluatePolynomialBasisDerivatives(
const array_1d<double,3>& rX,
BoundedMatrix<double, 2, 6>& rBasisDerivatives);


static void EvaluatePolynomialBasisDerivatives(
const array_1d<double,3>& rX,
BoundedMatrix<double, 3, 10>& rBasisDerivatives);


template<std::size_t TDim, std::size_t TOrder>
static void CalculateShapeFunctions(
const Matrix& rPoints,
const array_1d<double,3>& rX,
const double h,
Vector& rN);


template<std::size_t TDim, std::size_t TOrder>
static void CalculateShapeFunctionsAndGradients(
const Matrix& rPoints,
const array_1d<double,3>& rX,
const double h,
Vector& rN,
Matrix& rDNDX);
};

}  
