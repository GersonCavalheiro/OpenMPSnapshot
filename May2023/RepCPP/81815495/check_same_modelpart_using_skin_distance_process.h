
#pragma once



#include "includes/define.h"
#include "processes/process.h"
#include "containers/model.h"
#include "includes/kratos_parameters.h"

namespace Kratos
{







template<std::size_t TDim>
class KRATOS_API(KRATOS_CORE) CheckSameModelPartUsingSkinDistanceProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(CheckSameModelPartUsingSkinDistanceProcess);


explicit CheckSameModelPartUsingSkinDistanceProcess(
Model& rModel,
Parameters ThisParameters = Parameters(R"({})")
)
: mrModel(rModel),
mThisParameters(ThisParameters)
{
KRATOS_TRY

Parameters default_parameters = GetDefaultParameters();
mThisParameters.RecursivelyValidateAndAssignDefaults(default_parameters);

KRATOS_CATCH("");
}

~CheckSameModelPartUsingSkinDistanceProcess() override = default;




void Execute() override;


Process::Pointer Create(
Model& rModel,
Parameters ThisParameters
) override
{
return Kratos::make_shared<CheckSameModelPartUsingSkinDistanceProcess<TDim>>(rModel, ThisParameters);
}


const Parameters GetDefaultParameters() const override;




std::string Info() const override
{
return "CheckSameModelPartUsingSkinDistanceProcess" + std::to_string(TDim) + "D";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "CheckSameModelPartUsingSkinDistanceProcess" << TDim << "D";
}

void PrintData(std::ostream& rOStream) const override
{
}


private:


Model& mrModel;             
Parameters mThisParameters; 






CheckSameModelPartUsingSkinDistanceProcess& operator=(CheckSameModelPartUsingSkinDistanceProcess const& rOther);


}; 




template<std::size_t TDim>
inline std::istream& operator >> (std::istream& rIStream,
CheckSameModelPartUsingSkinDistanceProcess<TDim>& rThis);

template<std::size_t TDim>
inline std::ostream& operator << (std::ostream& rOStream,
const CheckSameModelPartUsingSkinDistanceProcess<TDim>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  
