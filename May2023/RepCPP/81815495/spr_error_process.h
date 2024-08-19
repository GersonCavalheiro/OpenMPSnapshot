
#pragma once



#include "processes/process.h"
#include "includes/kratos_parameters.h"
#include "includes/model_part.h"

namespace Kratos
{


typedef std::size_t SizeType;





template<SizeType TDim>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SPRErrorProcess
: public Process
{
public:


typedef ModelPart::NodesContainerType                                     NodesArrayType;
typedef ModelPart::ElementsContainerType                               ElementsArrayType;
typedef ModelPart::ConditionsContainerType                           ConditionsArrayType;

typedef Node                                                                NodeType;

typedef GlobalPointersVector< Element >::iterator                         WeakElementItType;
typedef NodesArrayType::iterator                                              NodeItType;
typedef ElementsArrayType::iterator                                        ElementItType;

typedef std::size_t                                                            IndexType;

KRATOS_CLASS_POINTER_DEFINITION(SPRErrorProcess);

static constexpr SizeType SigmaSize = (TDim == 2) ? 3 : 6;




SPRErrorProcess(
ModelPart& rThisModelPart,
Parameters ThisParameters = Parameters(R"({})")
);

virtual ~SPRErrorProcess() {}


void operator()()
{
Execute();
}



void Execute() override;


const Parameters GetDefaultParameters() const override;






virtual std::string Info() const override
{
return "SPRErrorProcess";
}

virtual void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SPRErrorProcess";
}

virtual void PrintData(std::ostream& rOStream) const override
{
}

protected:



ModelPart& mThisModelPart;                                  
Variable<Vector>* mpStressVariable = &CAUCHY_STRESS_VECTOR; 
SizeType mEchoLevel;                                        





void CalculateSuperconvergentStresses();


void CalculateErrorEstimation(
double& rEnergyNormOverall,
double& rErrorOverall
);


virtual void CalculatePatch(
NodeItType itNode,
NodeItType itPatchNode,
const SizeType NeighbourSize,
Vector& rSigmaRecovered
);








private:





static inline void FindNodalNeighbours(ModelPart& rModelPart);





SPRErrorProcess& operator=(SPRErrorProcess const& rOther)
{
return *this;
};


};

};
