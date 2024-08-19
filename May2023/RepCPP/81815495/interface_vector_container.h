
#pragma once



#include "includes/define.h"
#include "includes/model_part.h"
#include "mapper_utilities.h"

namespace Kratos
{



template<class TSparseSpace, class TDenseSpace>
class KRATOS_API(MAPPING_APPLICATION) InterfaceVectorContainer
{
public:

KRATOS_CLASS_POINTER_DEFINITION(InterfaceVectorContainer);

typedef typename TSparseSpace::VectorType TSystemVectorType;

typedef Kratos::unique_ptr<TSystemVectorType> TSystemVectorUniquePointerType;


explicit InterfaceVectorContainer(ModelPart& rModelPart) : mrModelPart(rModelPart) {}

virtual ~InterfaceVectorContainer() = default;



void UpdateSystemVectorFromModelPart(const Variable<double>& rVariable,
const Kratos::Flags& rMappingOptions);

void UpdateModelPartFromSystemVector(const Variable<double>& rVariable,
const Kratos::Flags& rMappingOptions);


TSystemVectorType& GetVector()
{
KRATOS_DEBUG_ERROR_IF_NOT(mpInterfaceVector)
<< "The Interface-Vector was not initialized" << std::endl;
return *mpInterfaceVector;
}

const TSystemVectorType& GetVector() const
{
KRATOS_DEBUG_ERROR_IF_NOT(mpInterfaceVector)
<< "The Interface-Vector was not initialized" << std::endl;
return *mpInterfaceVector;
}

TSystemVectorUniquePointerType& pGetVector() { return mpInterfaceVector; }
const TSystemVectorUniquePointerType& pGetVector() const { return mpInterfaceVector; }

ModelPart& GetModelPart() { return mrModelPart; }
const ModelPart& GetModelPart() const { return mrModelPart; }


private:

ModelPart& mrModelPart;
TSystemVectorUniquePointerType mpInterfaceVector = nullptr;


InterfaceVectorContainer& operator=(InterfaceVectorContainer const& rOther);

InterfaceVectorContainer(InterfaceVectorContainer const& rOther);


}; 



}  
