
#pragma once

#include <iostream>


#include "includes/model_part.h"
#include "includes/define.h"
#include "includes/kratos_parameters.h"

namespace Kratos
{






class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) InterfacePreprocessCondition
{
public:

typedef Point                                              PointType;
typedef Node                                             NodeType;
typedef Geometry<NodeType>                              GeometryType;
typedef Geometry<PointType>                        GeometryPointType;

typedef std::size_t                                        IndexType;

typedef std::size_t                                         SizeType;

typedef ModelPart::NodesContainerType                 NodesArrayType;
typedef ModelPart::ElementsContainerType           ElementsArrayType;
typedef ModelPart::ConditionsContainerType       ConditionsArrayType;

KRATOS_CLASS_POINTER_DEFINITION(InterfacePreprocessCondition);




InterfacePreprocessCondition(ModelPart& rMainModelPrt)
:mrMainModelPart(rMainModelPrt)
{
}

virtual ~InterfacePreprocessCondition() = default;




void GenerateInterfacePart(
ModelPart& rInterfacePart,
Parameters ThisParameters =  Parameters(R"({})")
);

protected:






private:


ModelPart& mrMainModelPart; 




void CheckAndCreateProperties(ModelPart& rInterfacePart);


bool CheckOnTheFace(
const std::vector<std::size_t>& rIndexVector,
GeometryType& rElementGeometry
);


std::unordered_map<IndexType, Properties::Pointer> CreateNewProperties();


template<class TClass>
void CopyProperties(
Properties::Pointer pOriginalProperty,
Properties::Pointer pNewProperty,
const Variable<TClass>& rVariable,
const bool AssignZero = true
)
{
if(pOriginalProperty->Has(rVariable)) {
const TClass& value = pOriginalProperty->GetValue(rVariable);
pNewProperty->SetValue(rVariable, value);
} else if (AssignZero) {
KRATOS_INFO("InterfacePreprocessCondition") << "Property " << rVariable.Name() << " not available. Assigning zero value" << std::endl;
pNewProperty->SetValue(rVariable, rVariable.Zero());
}
}


void CreateNewCondition(
Properties::Pointer prThisProperties,
const GeometryType& rGeometry,
const IndexType ConditionId,
Condition const& rCondition
);


void AssignMasterSlaveCondition(Condition::Pointer pCond);


void PrintNodesAndConditions(
const IndexType NodesCounter,
const IndexType rCondCounter
);


IndexType ReorderConditions();


inline void GenerateEdgeCondition(
ModelPart& rInterfacePart,
Properties::Pointer prThisProperties,
const GeometryType& rEdgeGeometry,
const bool SimplestGeometry,
IndexType& rCondCounter,
IndexType& rConditionId
);


inline void GenerateFaceCondition(
ModelPart& rInterfacePart,
Properties::Pointer prThisProperties,
const GeometryType& rFaceGeometry,
const bool SimplestGeometry,
IndexType& rCondCounter,
IndexType& rConditionId
);




}; 
}
