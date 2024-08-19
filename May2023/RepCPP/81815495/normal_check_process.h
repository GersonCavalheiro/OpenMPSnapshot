
#pragma once



#include "processes/process.h"
#include "includes/model_part.h"

namespace Kratos
{






class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) NormalCheckProcess
: public Process
{
public:

typedef Node                                              NodeType;
typedef Point                                               PointType;
typedef PointType::CoordinatesArrayType          CoordinatesArrayType;

typedef Geometry<NodeType>                               GeometryType;
typedef Geometry<PointType>                         GeometryPointType;

static constexpr double ZeroTolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION( NormalCheckProcess );




NormalCheckProcess(
ModelPart& rModelPart,
Parameters ThisParameters =  Parameters(R"({})")
) : mrModelPart(rModelPart),
mParameters(ThisParameters)
{
const Parameters default_parameters = GetDefaultParameters();
mParameters.ValidateAndAssignDefaults(default_parameters);
}

virtual ~NormalCheckProcess()= default;



void operator()()
{
Execute();
}



void Execute() override;


const Parameters GetDefaultParameters() const override;







std::string Info() const override
{
return "NormalCheckProcess";
}




void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}



protected:



ModelPart& mrModelPart;  
Parameters mParameters;  







private:








}; 








inline std::istream& operator >> (std::istream& rIStream,
NormalCheckProcess& rThis);




inline std::ostream& operator << (std::ostream& rOStream,
const NormalCheckProcess& rThis)
{
return rOStream;
}


}  
