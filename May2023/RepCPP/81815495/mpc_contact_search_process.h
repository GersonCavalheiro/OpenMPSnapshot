
#pragma once



#include "custom_processes/base_contact_search_process.h"

namespace Kratos
{






template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster = TNumNodes>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) MPCContactSearchProcess
: public BaseContactSearchProcess<TDim, TNumNodes, TNumNodesMaster>
{
public:

typedef BaseContactSearchProcess<TDim, TNumNodes, TNumNodesMaster> BaseType;

typedef typename BaseType::NodesArrayType           NodesArrayType;
typedef typename BaseType::ConditionsArrayType ConditionsArrayType;
typedef typename BaseType::NodeType                       NodeType;
typedef typename BaseType::GeometryType               GeometryType;

typedef std::size_t IndexType;

KRATOS_CLASS_POINTER_DEFINITION( MPCContactSearchProcess );




MPCContactSearchProcess(
ModelPart& rMainModelPart,
Parameters ThisParameters =  Parameters(R"({})"),
Properties::Pointer pPairedProperties = nullptr
);

~MPCContactSearchProcess()= default;;




void CheckContactModelParts() override;


void ResetContactOperators() override;







std::string Info() const override
{
return "MPCContactSearchProcess";
}




void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}



protected:






void CleanModelPart(ModelPart& rModelPart) override;





private:





inline IndexType GetMaximumConstraintsIds();


Condition::Pointer AddPairing(
ModelPart& rComputingModelPart,
IndexType& rConditionId,
GeometricalObject::Pointer pCondSlave,
const array_1d<double, 3>& rSlaveNormal,
GeometricalObject::Pointer pCondMaster,
const array_1d<double, 3>& rMasterNormal,
IndexMap::Pointer pIndexesPairs,
Properties::Pointer pProperties
) override;





}; 








template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster>
inline std::istream& operator >> (std::istream& rIStream,
MPCContactSearchProcess<TDim, TNumNodes, TNumNodesMaster>& rThis);




template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster>
inline std::ostream& operator << (std::ostream& rOStream,
const MPCContactSearchProcess<TDim, TNumNodes, TNumNodesMaster>& rThis)
{
return rOStream;
}


}  
