
#pragma once

#include <array>


#include "containers/variable.h"
#include "fluid_dynamics_application_variables.h"
#include "geometries/geometry.h"
#include "includes/constitutive_law.h"
#include "includes/node.h"
#include "includes/process_info.h"
#include "includes/ublas_interface.h"
#include "utilities/time_discretization.h"

#include "custom_utilities/fluid_adjoint_utilities.h"

namespace Kratos
{


template <unsigned int TDim>
class QSVMSDerivativeUtilities
{
public:

using NodeType = Node;

using GeometryType = Geometry<NodeType>;

using IndexType = std::size_t;

using DependentVariablesListType = std::vector<
std::tuple<
const Variable<double>&,
std::vector<const Variable<double>*>
>
>;

using DerivativeGradientsArray = std::array<const Variable<double>*, 9>;

constexpr static IndexType TStrainSize = (TDim - 1) * 3; 


static void CalculateStrainRate(
Vector& rOutput,
const Matrix& rNodalVelocity,
const Matrix& rdNdX);

static const std::array<const Variable<double>*, TStrainSize> GetStrainRateVariables();



template<unsigned int TComponentIndex = 0>
class Derivative
{
public:

static constexpr IndexType ComponentIndex = TComponentIndex;


Derivative(
const IndexType NodeIndex,
const GeometryType& rGeometry,
const double W,
const Vector& rN,
const Matrix& rdNdX,
const double WDerivative,
const double DetJDerivative,
const Matrix& rdNdXDerivative);


DependentVariablesListType GetEffectiveViscosityDependentVariables() const { return DependentVariablesListType({}); }


protected:

const IndexType mNodeIndex;
const GeometryType& mrGeometry;
const double mW;
const Vector& mrN;
const Matrix& mrdNdX;
const double mWDerivative;
const double mDetJDerivative;
const Matrix& mrdNdXDerivative;

};


template<unsigned int TNumNodes, unsigned int TComponentIndex>
class VelocityDerivative : public Derivative<TComponentIndex>
{
public:

using BaseType = Derivative<TComponentIndex>;

static constexpr IndexType ComponentIndex = BaseType::ComponentIndex;

static constexpr double VelocityDerivativeFactor = 1.0;

static constexpr double PressureDerivativeFactor = 0.0;

static constexpr unsigned int TDerivativeDimension = TDim;


VelocityDerivative(
const IndexType NodeIndex,
const GeometryType& rGeometry,
const double W,
const Vector& rN,
const Matrix& rdNdX,
const double WDerivative,
const double DetJDerivative,
const Matrix& rdNdXDerivative)
: BaseType(NodeIndex, rGeometry, W, rN, rdNdX, WDerivative, DetJDerivative, rdNdXDerivative)
{
}


const Variable<double>& GetDerivativeVariable() const;

array_1d<double, TDim> CalculateEffectiveVelocityDerivative(const array_1d<double, TDim>& rVelocity) const;

double CalculateElementLengthDerivative(const double ElementLength) const;

void CalculateStrainRateDerivative(
Vector& rOutput,
const Matrix& rNodalVelocity) const;

};


template<unsigned int TNumNodes>
class PressureDerivative : public Derivative<0>
{
public:

using BaseType = Derivative<0>;

static constexpr IndexType ComponentIndex = BaseType::ComponentIndex;

static constexpr double VelocityDerivativeFactor = 0.0;

static constexpr double PressureDerivativeFactor = 1.0;

static constexpr unsigned int TDerivativeDimension = 1;


PressureDerivative(
const IndexType NodeIndex,
const GeometryType& rGeometry,
const double W,
const Vector& rN,
const Matrix& rdNdX,
const double WDerivative,
const double DetJDerivative,
const Matrix& rdNdXDerivative)
: BaseType(NodeIndex, rGeometry, W, rN, rdNdX, WDerivative, DetJDerivative, rdNdXDerivative)
{
}


const Variable<double>& GetDerivativeVariable() const { return PRESSURE; }

array_1d<double, TDim> CalculateEffectiveVelocityDerivative(const array_1d<double, TDim>& rVelocity) const;

double CalculateElementLengthDerivative(const double ElementLength) const;

void CalculateStrainRateDerivative(
Vector& rOutput,
const Matrix& rNodalVelocity) const;

};


template<unsigned int TNumNodes, unsigned int TComponentIndex>
class ShapeDerivative : public Derivative<TComponentIndex>
{
public:

using BaseType = Derivative<TComponentIndex>;

static constexpr IndexType ComponentIndex = BaseType::ComponentIndex;

static constexpr double VelocityDerivativeFactor = 0.0;

static constexpr double PressureDerivativeFactor = 0.0;

static constexpr unsigned int TDerivativeDimension = TDim;


ShapeDerivative(
const IndexType NodeIndex,
const GeometryType& rGeometry,
const double W,
const Vector& rN,
const Matrix& rdNdX,
const double WDerivative,
const double DetJDerivative,
const Matrix& rdNdXDerivative)
: BaseType(NodeIndex, rGeometry, W, rN, rdNdX, WDerivative, DetJDerivative, rdNdXDerivative)
{
}


const Variable<double>& GetDerivativeVariable() const;

array_1d<double, TDim> CalculateEffectiveVelocityDerivative(const array_1d<double, TDim>& rVelocity) const;

double CalculateElementLengthDerivative(const double ElementLength) const;

void CalculateStrainRateDerivative(
Vector& rOutput,
const Matrix& rNodalVelocity) const;

};

private:

static void CalculateStrainRateVelocityDerivative(
Vector& rOutput,
const IndexType DerivativeNodeIndex,
const IndexType DerivativeDirectionIndex,
const Matrix& rdNdX);

};


} 