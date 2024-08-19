
#if !defined(SET_DUMMY_PROPERTY_FOR_RIGID_ELEMENTS_PROCESS)
#define  SET_DUMMY_PROPERTY_FOR_RIGID_ELEMENTS_PROCESS

#include "includes/define.h"
#include "includes/model_part.h"
#include "processes/process.h"

namespace Kratos
{

class SetDummyPropertyForRigidElementsProcess : public Process {

public:

KRATOS_CLASS_POINTER_DEFINITION(SetDummyPropertyForRigidElementsProcess);

SetDummyPropertyForRigidElementsProcess(ModelPart &model_part,
unsigned int dummy_property_id)
: rModelPart(model_part) {
rDummyPropertyId = dummy_property_id;
}

~SetDummyPropertyForRigidElementsProcess() override {}

void operator()() {
Execute();
}

void Execute() override {

KRATOS_TRY;

Properties::Pointer rDummyProperty = rModelPart.pGetProperties(rDummyPropertyId);
const auto& it_elem_begin = rModelPart.ElementsBegin();

#pragma omp parallel for
for (int i = 0; i < static_cast<int>(rModelPart.Elements().size()); i++) {
auto it_elem = it_elem_begin + i;
it_elem->SetProperties(rDummyProperty);
}

KRATOS_CATCH("");

}

void ExecuteInitialize() override {}

void ExecuteInitializeSolutionStep() override {}

protected:

ModelPart& rModelPart;
unsigned int rDummyPropertyId;

private:

}; 

inline std::istream &operator>>(std::istream &rIStream,
SetDummyPropertyForRigidElementsProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const SetDummyPropertyForRigidElementsProcess &rThis) {
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 