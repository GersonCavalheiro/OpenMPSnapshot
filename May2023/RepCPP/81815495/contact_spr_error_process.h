
#pragma once



#include "custom_processes/spr_error_process.h"

namespace Kratos
{


typedef std::size_t SizeType;





template<SizeType TDim>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) ContactSPRErrorProcess
: public SPRErrorProcess<TDim>
{
public:


typedef SPRErrorProcess<TDim>                                                   BaseType;

typedef ModelPart::NodesContainerType                                     NodesArrayType;
typedef ModelPart::ElementsContainerType                               ElementsArrayType;
typedef ModelPart::ConditionsContainerType                           ConditionsArrayType;

typedef Node                                                                NodeType;

typedef GlobalPointersVector< Element >::iterator                      WeakElementItType;
typedef NodesArrayType::iterator                                              NodeItType;
typedef ElementsArrayType::iterator                                        ElementItType;

typedef std::size_t                                                            IndexType;

KRATOS_CLASS_POINTER_DEFINITION(ContactSPRErrorProcess);

static constexpr SizeType SigmaSize = (TDim == 2) ? 3 : 6;



ContactSPRErrorProcess(
ModelPart& rThisModelPart,
Parameters ThisParameters = Parameters(R"({})")
);

~ContactSPRErrorProcess() override {}


void operator()()
{
this->Execute();
}


const Parameters GetDefaultParameters() const override;







std::string Info() const override
{
return "ContactSPRErrorProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ContactSPRErrorProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}

protected:








void CalculatePatch(
NodeItType itNode,
NodeItType itPatchNode,
const SizeType NeighbourSize,
Vector& rSigmaRecovered
) override;








private:


double mPenaltyNormal;   
double mPenaltyTangent;  




void ComputeNormalTangentMatrices(
BoundedMatrix<double, 1, SigmaSize>& rNk,
BoundedMatrix<double, 1, SigmaSize>& rTk1,
BoundedMatrix<double, 1, SigmaSize>& rTk2,
const array_1d<double, 3>& rNormal
);





ContactSPRErrorProcess& operator=(ContactSPRErrorProcess const& rOther)
{
return *this;
};


};

};
