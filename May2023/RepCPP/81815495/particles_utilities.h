
#pragma once

#include <string>
#include <iostream>


#include "includes/define.h"
#include "utilities/binbased_fast_point_locator.h"
#include "utilities/atomic_utilities.h"

namespace Kratos
{







class ParticlesUtilities
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ParticlesUtilities);


ParticlesUtilities() = delete;


template<unsigned int TDim, bool CounterHasHistory=false >
static void CountParticlesInNodes(
BinBasedFastPointLocator<TDim>& rLocator,
ModelPart& rVolumeModelPart,
const ModelPart& rParticlesModelPart,
const Variable<double>& rCounterVariable,
const double SearchTolerance=1e-5
)
{
block_for_each(rVolumeModelPart.Nodes(), [&rCounterVariable](auto& rNode)
{
if constexpr (CounterHasHistory)
rNode.FastGetSolutionStepValue(rCounterVariable) = 0.0;
else
rNode.SetValue(rCounterVariable,0.0);
});


unsigned int max_results = 10000;
typename BinBasedFastPointLocator<TDim>::ResultContainerType TLS(max_results);

block_for_each(rParticlesModelPart.Nodes(), TLS, [&rLocator, &rCounterVariable, SearchTolerance](const auto& rNode, auto& rTLS)
{

Vector shape_functions;
Element::Pointer p_element;
const bool is_found = rLocator.FindPointOnMesh(rNode.Coordinates(), shape_functions, p_element, rTLS.begin(), rTLS.size(), SearchTolerance);

if(is_found)
{

auto& r_geom = p_element->GetGeometry();
for(unsigned int i=0; i<r_geom.size(); ++i)
{
if constexpr (CounterHasHistory)
{
auto& rcounter = r_geom[i].FastGetSolutionStepValue(rCounterVariable);
AtomicAdd(rcounter, 1.0);
}
else
{
auto& rcounter = r_geom[i].GetValue(rCounterVariable);
AtomicAdd(rcounter, 1.0);
}

}
}

});

}


template<unsigned int TDim, class TScalarType, bool ParticleTypeVariableHasHistory=false>
static void ClassifyParticlesInElements(
BinBasedFastPointLocator<TDim>& rLocator,
ModelPart& rVolumeModelPart,
const ModelPart& rParticlesModelPart,
const int NumberOfTypes,
const Variable<TScalarType>& rParticleTypeVariable=AUX_INDEX,
const Variable<Vector>& rClassificationVectorVariable=MARKER_LABELS,
const double SearchTolerance=1e-5
)
{
Vector zero = ZeroVector(NumberOfTypes);
block_for_each(rVolumeModelPart.Elements(), [&rClassificationVectorVariable, &zero](auto& rElement){
rElement.SetValue(rClassificationVectorVariable,zero);
});


unsigned int max_results = 10000;
auto TLS = std::make_pair(typename BinBasedFastPointLocator<TDim>::ResultContainerType(max_results), Vector());

block_for_each(rParticlesModelPart.Nodes(),
TLS,
[&rLocator, &rParticleTypeVariable, &rClassificationVectorVariable, &NumberOfTypes, SearchTolerance]
(const auto& rNode, auto& rTLS)
{
auto& results = rTLS.first;
Vector& shape_functions = rTLS.second;
Element::Pointer p_element;
const bool is_found = rLocator.FindPointOnMesh(rNode.Coordinates(), shape_functions, p_element, results.begin(), results.size(), SearchTolerance);

if(is_found)
{
int particle_type;
if constexpr (ParticleTypeVariableHasHistory)
particle_type = static_cast<int>(rNode.FastGetSolutionStepValue(rParticleTypeVariable));
else
particle_type = static_cast<int>(rNode.GetValue(rParticleTypeVariable));

if(particle_type>=0 && particle_type<NumberOfTypes) 
{
auto& rclassification = p_element->GetValue(rClassificationVectorVariable);
AtomicAdd(rclassification[particle_type], 1.0);
}
}
});
}



template<unsigned int TDim, class TDataType, bool VariableHasHistory >
static void MarkOutsiderParticles(
BinBasedFastPointLocator<TDim>& rLocator,
ModelPart& rParticlesModelPart,
const Variable<TDataType>& rVariable,
const TDataType& OutsiderValue,
const double SearchTolerance=1e-5
)
{
unsigned int max_results = 10000;
typename BinBasedFastPointLocator<TDim>::ResultContainerType TLS(max_results);

block_for_each(rParticlesModelPart.Nodes(), TLS, [&rLocator, &rVariable, &OutsiderValue, SearchTolerance](auto& rNode, auto& rTLS)
{

Vector shape_functions;
Element::Pointer p_element;
const bool is_found = rLocator.FindPointOnMesh(rNode.Coordinates(), shape_functions, p_element, rTLS.begin(), rTLS.size(), SearchTolerance);

if(!is_found)
{
if constexpr (VariableHasHistory)
rNode.FastGetSolutionStepValue(rVariable) = OutsiderValue;
else
rNode.SetValue(rVariable, OutsiderValue);
}
});
}


template<unsigned int TDim, class TDataType, bool InterpolationVariableHasHistory>
static std::pair< DenseVector<bool>, std::vector<TDataType> > InterpolateValuesAtCoordinates(
BinBasedFastPointLocator<TDim>& rLocator,
const Matrix& rCoordinates,
const Variable<TDataType>& rInterpolationVariable,
const double SearchTolerance
)
{
unsigned int max_results = 10000;
typename BinBasedFastPointLocator<TDim>::ResultContainerType TLS(max_results);

auto interpolations = std::make_pair(DenseVector<bool>(rCoordinates.size1()), std::vector<TDataType>(rCoordinates.size1()));

const auto zero = rInterpolationVariable.Zero();
IndexPartition(rCoordinates.size1()).for_each(TLS, [&rLocator, &rCoordinates, &interpolations, &rInterpolationVariable, &zero, SearchTolerance](const auto& i, auto& rTLS)
{
Vector shape_functions;
Element::Pointer p_element;
const bool is_found = rLocator.FindPointOnMesh(row(rCoordinates,i), shape_functions, p_element, rTLS.begin(), rTLS.size(), SearchTolerance);

(interpolations.first)[i] = is_found;
if(is_found)
{
auto& r_geom = p_element->GetGeometry();
(interpolations.second)[i] = zero;
for(unsigned int k=0; k<r_geom.size(); ++k)
{
if constexpr (InterpolationVariableHasHistory)
(interpolations.second)[i] += shape_functions[k]*r_geom[k].FastGetSolutionStepValue(rInterpolationVariable);
else
(interpolations.second)[i] += shape_functions[k]*r_geom[k].GetValue(rInterpolationVariable);

}
}
});

return interpolations;
}






std::string Info() const
{
return std::string("ParticlesUtilities");
};

void PrintInfo(std::ostream& rOStream) const {};

void PrintData(std::ostream& rOStream) const {};



private:







ParticlesUtilities& operator=(ParticlesUtilities const& rOther) = delete;

ParticlesUtilities(ParticlesUtilities const& rOther) = delete;

}; 




inline std::istream& operator >> (std::istream& rIStream,
ParticlesUtilities& rThis)
{
return rIStream;
}

inline std::ostream& operator << (std::ostream& rOStream,
const ParticlesUtilities& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

