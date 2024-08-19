
#pragma once



#include "custom_frictional_laws/frictional_law.h"

namespace Kratos
{






template< std::size_t TDim, std::size_t TNumNodes, bool TNormalVariation, std::size_t TNumNodesMaster = TNumNodes>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) FrictionalLawWithDerivative
: public FrictionalLaw
{
public:


typedef FrictionalLaw BaseType;

typedef Node NodeType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

typedef DerivativeDataFrictional<TDim, TNumNodes, TNumNodesMaster> DerivativeDataType;

typedef MortarOperatorWithDerivatives<TDim, TNumNodes, true, TNumNodesMaster> MortarConditionMatrices;

static constexpr double ZeroTolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION( FrictionalLawWithDerivative );



FrictionalLawWithDerivative()
{
}

FrictionalLawWithDerivative(const FrictionalLawWithDerivative& rhs)
{
}

~FrictionalLawWithDerivative()
{
}




virtual double GetDerivativeThresholdValue(
const NodeType& rNode,
const PairedCondition& rCondition,
const ProcessInfo& rCurrentProcessInfo,
const DerivativeDataType& rDerivativeData,
const MortarConditionMatrices& rMortarConditionMatrices,
const IndexType IndexDerivative,
const IndexType IndexNode
);




std::string Info() const override
{
return "FrictionalLawWithDerivative";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info() << std::endl;
}

void PrintData(std::ostream& rOStream) const override
{
rOStream << Info() << std::endl;
}


protected:








private:





friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, BaseType );
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, BaseType );
}


}; 






}  