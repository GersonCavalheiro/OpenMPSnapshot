
#pragma once



#include "includes/define.h"
#include "processes/process.h"
#include "includes/model_part.h"
#include "includes/kratos_parameters.h"

namespace Kratos
{







struct ComputeNodalGradientProcessSettings
{
constexpr static bool SaveAsHistoricalVariable = true;
constexpr static bool SaveAsNonHistoricalVariable = false;
constexpr static bool GetAsHistoricalVariable = true;
constexpr static bool GetAsNonHistoricalVariable = false;
};


struct AuxiliarVariableVectorRetriever
{
virtual ~AuxiliarVariableVectorRetriever()
{
}


virtual void GetVariableVector(
const Geometry<Node>& rGeometry,
const Variable<double>& rVariable,
Vector& rVector
)
{
KRATOS_ERROR << "Calling base class implementation" << std::endl;
}
};


template<bool THistorical>
struct VariableVectorRetriever
: public AuxiliarVariableVectorRetriever
{
~VariableVectorRetriever() override
{
}


void GetVariableVector(
const Geometry<Node>& rGeometry,
const Variable<double>& rVariable,
Vector& rVector
) override;
};


template<bool THistorical>
class KRATOS_API(KRATOS_CORE) ComputeNodalGradientProcess
: public Process
{
public:

typedef Node NodeType;

KRATOS_CLASS_POINTER_DEFINITION(ComputeNodalGradientProcess);


ComputeNodalGradientProcess(
ModelPart& rModelPart,
Parameters ThisParameters = Parameters(R"({})")
);

ComputeNodalGradientProcess(
ModelPart& rModelPart,
const Variable<double>& rOriginVariable,
const Variable<array_1d<double,3> >& rGradientVariable,
const Variable<double>& rAreaVariable = NODAL_AREA,
const bool NonHistoricalVariable = false
);

~ComputeNodalGradientProcess() override
{
}


void operator()()
{
Execute();
}



void Execute() override;


const Parameters GetDefaultParameters() const override;




std::string Info() const override
{
return "ComputeNodalGradientProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ComputeNodalGradientProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:







private:


ModelPart& mrModelPart;                                           
const Variable<double>* mpOriginVariable = nullptr;               
const Variable<array_1d<double,3>>* mpGradientVariable;           
const Variable<double>* mpAreaVariable = nullptr;                 
bool mNonHistoricalVariable = false;                              





void CheckOriginAndAreaVariables();


void ClearGradient();


array_1d<double, 3>& GetGradient(
Element::GeometryType& rThisGeometry,
unsigned int i
);


void ComputeElementalContributionsAndVolume();


void PonderateGradient();


void SynchronizeGradientAndVolume();




ComputeNodalGradientProcess& operator=(ComputeNodalGradientProcess const& rOther);


}; 






}  


