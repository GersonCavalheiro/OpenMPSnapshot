
#pragma once



#include "custom_frictional_laws/frictional_law_with_derivative.h"

namespace Kratos
{






template< std::size_t TDim, std::size_t TNumNodes, bool TNormalVariation, std::size_t TNumNodesMaster = TNumNodes>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) TrescaFrictionalLaw
: public FrictionalLawWithDerivative<TDim,TNumNodes,TNormalVariation, TNumNodesMaster>
{
public:


typedef FrictionalLawWithDerivative<TDim,TNumNodes,TNormalVariation, TNumNodesMaster> BaseType;

typedef typename BaseType::DerivativeDataType DerivativeDataType;

typedef typename BaseType::MortarConditionMatrices MortarConditionMatrices;

typedef Node NodeType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

static constexpr double ZeroTolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION( TrescaFrictionalLaw );



TrescaFrictionalLaw()
{
}

TrescaFrictionalLaw(const TrescaFrictionalLaw& rhs)
{
}

~TrescaFrictionalLaw()
{
}



double GetThresholdValue(
const NodeType& rNode,
const PairedCondition& rCondition,
const ProcessInfo& rCurrentProcessInfo
) override;


double GetDerivativeThresholdValue(
const NodeType& rNode,
const PairedCondition& rCondition,
const ProcessInfo& rCurrentProcessInfo,
const DerivativeDataType& rDerivativeData,
const MortarConditionMatrices& rMortarConditionMatrices,
const IndexType IndexDerivative,
const IndexType IndexNode
) override;




std::string Info() const override
{
return "TrescaFrictionalLaw";
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
