
#pragma once



#include "custom_processes/base_contact_search_process.h"

namespace Kratos
{


typedef std::size_t SizeType;





template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster = TNumNodes>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) AdvancedContactSearchProcess
: public BaseContactSearchProcess<TDim, TNumNodes, TNumNodesMaster>
{
public:

typedef BaseContactSearchProcess<TDim, TNumNodes, TNumNodesMaster> BaseType;

typedef typename BaseType::NodesArrayType           NodesArrayType;
typedef typename BaseType::ConditionsArrayType ConditionsArrayType;
typedef typename BaseType::NodeType                       NodeType;
typedef typename BaseType::GeometryType               GeometryType;

typedef std::size_t IndexType;

static constexpr double ZeroTolerance = std::numeric_limits<double>::epsilon();

static constexpr double GapThreshold = 2.0e-4;

KRATOS_CLASS_POINTER_DEFINITION( AdvancedContactSearchProcess );




AdvancedContactSearchProcess(
ModelPart& rMainModelPart,
Parameters ThisParameters =  Parameters(R"({})"),
Properties::Pointer pPairedProperties = nullptr
);

virtual ~AdvancedContactSearchProcess()= default;;









std::string Info() const override
{
return "AdvancedContactSearchProcess";
}




void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}



protected:






void ComputeActiveInactiveNodes() override;


void CheckPairing(
ModelPart& rComputingModelPart,
IndexType& rConditionId
) override;





private:





void ComputeLinearRegressionGapPressure(
double& a,
double& b
);


void SetActiveNodeWithRegression(
NodeType& rNode,
const double a,
const double b
);


void CorrectScalarMortarLM(
NodeType& rNode,
const double a,
const double b
);


void CorrectComponentsMortarLM(
NodeType& rNode,
const double a,
const double b
);


void CorrectALMFrictionlessMortarLM(
NodeType& rNode,
const double a,
const double b
);


void CorrectALMFrictionlessComponentsMortarLM(
NodeType& rNode,
const double a,
const double b
);


void CorrectALMFrictionalMortarLM(
NodeType& rNode,
const double a,
const double b
);


void PredictScalarMortarLM(
NodeType& rNode,
const double a,
const double b
);


void PredictComponentsMortarLM(
NodeType& rNode,
const double a,
const double b
);


void PredictALMFrictionlessMortarLM(
NodeType& rNode,
const double a,
const double b
);


void PredictALMFrictionlessComponentsMortarLM(
NodeType& rNode,
const double a,
const double b
);


void PredictALMFrictionalMortarLM(
NodeType& rNode,
const double a,
const double b
);





}; 








template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster = TNumNodes>
inline std::istream& operator >> (std::istream& rIStream,
AdvancedContactSearchProcess<TDim, TNumNodes, TNumNodesMaster>& rThis);




template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster = TNumNodes>
inline std::ostream& operator << (std::ostream& rOStream,
const AdvancedContactSearchProcess<TDim, TNumNodes, TNumNodesMaster>& rThis)
{
return rOStream;
}


}  
