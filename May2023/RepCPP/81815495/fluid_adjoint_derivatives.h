
#pragma once

#include <tuple>
#include <utility>


#include "includes/ublas_interface.h"


namespace Kratos
{


template<unsigned int TNumNodes, unsigned int TBlockSize>
class ZeroDerivatives
{
public:

using IndexType = std::size_t;

constexpr static IndexType NumNodes = TNumNodes;

constexpr static IndexType BlockSize = TBlockSize;


template<IndexType TSize, class... TArgs>
void inline CalculateGaussPointResidualsDerivativeContributions(
BoundedVector<double, TSize>& rResidualDerivative,
TArgs&... rArgs) const
{
rResidualDerivative.clear();
}

};


template<class TSubAssemblyType, unsigned int TElementDataHolderIndex, unsigned int TRowStartingIndex, unsigned int TColumnStartingIndex = 0>
class SubAssembly : public TSubAssemblyType
{
public:

using IndexType = std::size_t;

using SubAssemblyType = TSubAssemblyType;

static constexpr unsigned int ElementDataIndex = TElementDataHolderIndex;

static constexpr unsigned int RowStartingIndex = TRowStartingIndex;

static constexpr unsigned int ColumnStartingIndex = TColumnStartingIndex;

static constexpr unsigned int NumNodes = SubAssemblyType::NumNodes;

static constexpr unsigned int BlockSize = SubAssemblyType::BlockSize;

static constexpr unsigned int ResidualSize = NumNodes * BlockSize;



inline BoundedVector<double, ResidualSize>& GetSubVector()
{
return mSubVector;
}


template<class TCombinedElementDataContainer>
inline typename std::tuple_element<ElementDataIndex, TCombinedElementDataContainer>::type& GetElementDataContainer(TCombinedElementDataContainer& rCombinedDataContainer) const
{
static_assert(
ElementDataIndex < std::tuple_size_v<TCombinedElementDataContainer>,
"Required Element data container index is more than the available element data containers.");
return std::get<ElementDataIndex>(rCombinedDataContainer);
}


template<IndexType TAssemblyRowBlockSize, IndexType TAssemblyColumnBlockSize = TAssemblyRowBlockSize>
inline void AssembleSubVectorToMatrix(
Matrix& rOutput,
const IndexType NodeIndex) const
{
KRATOS_TRY

KRATOS_DEBUG_ERROR_IF(rOutput.size1() != TAssemblyRowBlockSize * NumNodes)
<< "rOuput.size1() does not have the required size. [ rOutput.size1() = "
<< rOutput.size1() << ", required_size = TAssemblyRowBlockSize * NumNodes = "
<< TAssemblyRowBlockSize * NumNodes << " ].\n";

KRATOS_DEBUG_ERROR_IF(rOutput.size2() != TAssemblyColumnBlockSize * NumNodes)
<< "rOuput.size2() does not have the required size. [ rOutput.size2() = "
<< rOutput.size2() << ", required_size = TAssemblyColumnBlockSize * NumNodes = "
<< TAssemblyColumnBlockSize * NumNodes << " ].\n";

for (IndexType i = 0; i < NumNodes; ++i) {
for (IndexType j = 0; j < BlockSize; ++j) {
rOutput(TAssemblyRowBlockSize * NodeIndex + RowStartingIndex, i * TAssemblyColumnBlockSize + j + ColumnStartingIndex) += mSubVector[i * BlockSize + j];
}
}

KRATOS_CATCH("");
}


template<IndexType TAssemblyRowBlockSize>
inline void AssembleSubVectorToVector(Vector& rOutput) const
{
KRATOS_TRY

static_assert(ColumnStartingIndex == 0);

KRATOS_DEBUG_ERROR_IF(rOutput.size() != TAssemblyRowBlockSize * NumNodes)
<< "rOuput.size() does not have the required size. [ rOutput.size() = "
<< rOutput.size() << ", required_size = TAssemblyRowBlockSize * NumNodes = "
<< TAssemblyRowBlockSize * NumNodes << " ].\n";

for (IndexType i = 0; i < NumNodes; ++i) {
for (IndexType j = 0; j < BlockSize; ++j) {
rOutput[i * TAssemblyRowBlockSize + j] += mSubVector[i * BlockSize + j];
}
}

KRATOS_CATCH("");
}

private:

BoundedVector<double, ResidualSize> mSubVector;

};


template <
class TCombinedElementDataContainer,
class TCombinedCalculationContainers
>
class CalculationContainerTraits
{
public:

using CombinedElementDataContainerType = TCombinedElementDataContainer;

using CombinedCalculationContainersType = TCombinedCalculationContainers;

};
}