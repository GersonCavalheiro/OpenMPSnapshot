
#pragma once




#include "custom_elements/data_containers/fluid_adjoint_derivatives.h"
#include "custom_elements/data_containers/qs_vms/qs_vms_derivative_utilities.h"
#include "custom_elements/data_containers/qs_vms/qs_vms_residual_derivatives.h"

namespace Kratos
{


template <unsigned int TDim, unsigned int TNumNodes>
class QSVMSAdjointElementData
{
private:

using IndexType = std::size_t;

using TResidualsDerivatives = QSVMSResidualDerivatives<TDim, TNumNodes>;

using Data = typename TResidualsDerivatives::QSVMSResidualData;

using ResidualsContributions = typename TResidualsDerivatives::ResidualsContributions;

using PressureDerivativeContributions = typename TResidualsDerivatives::template VariableDerivatives<typename QSVMSDerivativeUtilities<TDim>::template PressureDerivative<TNumNodes>>;

template<unsigned int TDirectionIndex>
using VelocityDerivativeContributions = typename TResidualsDerivatives::template VariableDerivatives<typename QSVMSDerivativeUtilities<TDim>::template VelocityDerivative<TNumNodes, TDirectionIndex>>;

template<unsigned int TDirectionIndex>
using ShapeDerivatives = typename TResidualsDerivatives::template VariableDerivatives<typename QSVMSDerivativeUtilities<TDim>::template ShapeDerivative<TNumNodes, TDirectionIndex>>;

template<unsigned int TDirectionIndex>
using AccelerationDerivativeContributions = typename TResidualsDerivatives::template SecondDerivatives<TDirectionIndex>;

static constexpr IndexType ElementDataContainerIndex = 0;

static constexpr IndexType ResidualColumnOffset = 0;


public:


using EquationAuxiliaries = TResidualsDerivatives;


using Residual = CalculationContainerTraits<
std::tuple<
Data>,
std::tuple<
SubAssembly<ResidualsContributions, ElementDataContainerIndex, 0, ResidualColumnOffset>>
>;

using ResidualStateVariableFirstDerivatives = std::conditional_t<
TDim == 2,
CalculationContainerTraits<
std::tuple<
Data>,
std::tuple<
SubAssembly<VelocityDerivativeContributions<0>, ElementDataContainerIndex, 0, ResidualColumnOffset>,
SubAssembly<VelocityDerivativeContributions<1>, ElementDataContainerIndex, 1, ResidualColumnOffset>,
SubAssembly<PressureDerivativeContributions,    ElementDataContainerIndex, 2, ResidualColumnOffset>>
>,
CalculationContainerTraits<
std::tuple<
Data>,
std::tuple<
SubAssembly<VelocityDerivativeContributions<0>, ElementDataContainerIndex, 0, ResidualColumnOffset>,
SubAssembly<VelocityDerivativeContributions<1>, ElementDataContainerIndex, 1, ResidualColumnOffset>,
SubAssembly<VelocityDerivativeContributions<2>, ElementDataContainerIndex, 2, ResidualColumnOffset>,
SubAssembly<PressureDerivativeContributions,    ElementDataContainerIndex, 3, ResidualColumnOffset>>
>
>;

using ResidualStateVariableSecondDerivatives = std::conditional_t<
TDim == 2,
CalculationContainerTraits<
std::tuple<
Data>,
std::tuple<
SubAssembly<AccelerationDerivativeContributions<0>, ElementDataContainerIndex, 0, ResidualColumnOffset>,
SubAssembly<AccelerationDerivativeContributions<1>, ElementDataContainerIndex, 1, ResidualColumnOffset>,
SubAssembly<ZeroDerivatives<TNumNodes, 3>,          ElementDataContainerIndex, 2, ResidualColumnOffset>>
>,
CalculationContainerTraits<
std::tuple<
Data>,
std::tuple<
SubAssembly<AccelerationDerivativeContributions<0>, ElementDataContainerIndex, 0, ResidualColumnOffset>,
SubAssembly<AccelerationDerivativeContributions<1>, ElementDataContainerIndex, 1, ResidualColumnOffset>,
SubAssembly<AccelerationDerivativeContributions<2>, ElementDataContainerIndex, 2, ResidualColumnOffset>,
SubAssembly<ZeroDerivatives<TNumNodes, 4>,          ElementDataContainerIndex, 3, ResidualColumnOffset>>
>
>;


using ResidualShapeDerivatives = std::conditional_t<
TDim == 2,
CalculationContainerTraits<
std::tuple<
Data>,
std::tuple<
SubAssembly<ShapeDerivatives<0>, ElementDataContainerIndex, 0, ResidualColumnOffset>,
SubAssembly<ShapeDerivatives<1>, ElementDataContainerIndex, 1, ResidualColumnOffset>>
>,
CalculationContainerTraits<
std::tuple<
Data>,
std::tuple<
SubAssembly<ShapeDerivatives<0>, ElementDataContainerIndex, 0, ResidualColumnOffset>,
SubAssembly<ShapeDerivatives<1>, ElementDataContainerIndex, 1, ResidualColumnOffset>,
SubAssembly<ShapeDerivatives<2>, ElementDataContainerIndex, 2, ResidualColumnOffset>>
>
>;

};
} 