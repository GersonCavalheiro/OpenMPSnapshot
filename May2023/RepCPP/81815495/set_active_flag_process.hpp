
#if !defined(KRATOS_SET_ACTIVE_FLAG_PROCESS_H_INCLUDED)
#define KRATOS_SET_ACTIVE_FLAG_PROCESS_H_INCLUDED




#include "spatial_containers/spatial_containers.h"

#include "custom_processes/set_active_flag_process.hpp"
#include "custom_utilities/mesher_utilities.hpp"
#include "includes/model_part.h"
#include "utilities/openmp_utils.h"
#include "utilities/math_utils.h"
#include "custom_processes/mesher_process.hpp"


namespace Kratos
{


typedef ModelPart::NodesContainerType NodesContainerType;
typedef ModelPart::ElementsContainerType ElementsContainerType;
typedef ModelPart::MeshType::GeometryType::PointsArrayType PointsArrayType;

typedef GlobalPointersVector<Node> NodeWeakPtrVectorType;
typedef GlobalPointersVector<Element> ElementWeakPtrVectorType;





class SetActiveFlagProcess
: public MesherProcess
{
public:

KRATOS_CLASS_POINTER_DEFINITION(SetActiveFlagProcess);


SetActiveFlagProcess(ModelPart &rModelPart,
bool unactivePeakElements,
bool unactiveSliverElements,
int EchoLevel)
: mrModelPart(rModelPart)
{
mUnactivePeakElements = unactivePeakElements;
mUnactiveSliverElements = unactiveSliverElements;
mEchoLevel = EchoLevel;
}

virtual ~SetActiveFlagProcess()
{
}

void operator()()
{
Execute();
}


void Execute() override{
KRATOS_TRY
#pragma omp parallel
{
ModelPart::ElementIterator ElemBegin;
ModelPart::ElementIterator ElemEnd;
OpenMPUtils::PartitionedIterators(mrModelPart.Elements(), ElemBegin, ElemEnd);
for (ModelPart::ElementIterator itElem = ElemBegin; itElem != ElemEnd; ++itElem)
{
if ((itElem)->IsNot(ACTIVE))
{
unsigned int numNodes = itElem->GetGeometry().size();
for (unsigned int i = 0; i < numNodes; i++)
{
if (itElem->GetGeometry()[i].Is(RIGID) && itElem->GetGeometry()[i].IsNot(SOLID) && itElem->GetGeometry()[i].Is(FREE_SURFACE))
{
ElementWeakPtrVectorType &neighb_elems = itElem->GetGeometry()[i].GetValue(NEIGHBOUR_ELEMENTS);
bool doNotSetNullPressure = false;
for (ElementWeakPtrVectorType::iterator ne = neighb_elems.begin(); ne != neighb_elems.end(); ne++)
{
if ((ne)->Is(ACTIVE))
{
doNotSetNullPressure = true;
break;
}
}
if (doNotSetNullPressure == false)
itElem->GetGeometry()[i].FastGetSolutionStepValue(PRESSURE) = 0;
}
}
unsigned int elementRigidNodes = 0;
for (unsigned int i = 0; i < numNodes; i++)
{
if (itElem->GetGeometry()[i].Is(RIGID) && itElem->GetGeometry()[i].IsNot(SOLID))
{
elementRigidNodes++;
}
}

if (elementRigidNodes == numNodes)
{
Geometry<Node> wallElementNodes = itElem->GetGeometry();
this->SetPressureToIsolatedWallNodes(wallElementNodes);
}
}
(itElem)->Set(ACTIVE, true);
}

} KRATOS_CATCH(" ")
}; 





std::string Info() const override
{
return "SetActiveFlagProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "SetActiveFlagProcess";
}

protected:


ModelPart &mrModelPart;

int mEchoLevel;
bool mUnactivePeakElements;
bool mUnactiveSliverElements;


void SetPressureToIsolatedWallNodes(Geometry<Node> &wallElementNodes)
{
KRATOS_TRY
unsigned int numNodes = wallElementNodes.size();
double currentPressureForIsolatedWall = 0;
double previousPressureForIsolatedWall = 0;
unsigned int isolatedWallID = 0;
bool foundedIsolatedWall = false;
for (unsigned int i = 0; i < numNodes; i++)
{
NodeWeakPtrVectorType &rN = wallElementNodes[i].GetValue(NEIGHBOUR_NODES);
bool localIsolatedWallNode = true;
for (unsigned int j = 0; j < rN.size(); j++)
{
if (rN[j].IsNot(RIGID))
{
localIsolatedWallNode = false;
break;
}
}
if (localIsolatedWallNode == true)
{
isolatedWallID = i;
foundedIsolatedWall = true;
}
else
{
if (wallElementNodes[i].FastGetSolutionStepValue(PRESSURE, 0) < currentPressureForIsolatedWall)
{
currentPressureForIsolatedWall = wallElementNodes[i].FastGetSolutionStepValue(PRESSURE, 0);
}
if (wallElementNodes[i].FastGetSolutionStepValue(PRESSURE, 1) < previousPressureForIsolatedWall)
{
previousPressureForIsolatedWall = wallElementNodes[i].FastGetSolutionStepValue(PRESSURE, 1);
}
}
}
if (foundedIsolatedWall == true)
{
wallElementNodes[isolatedWallID].FastGetSolutionStepValue(PRESSURE, 0) = currentPressureForIsolatedWall;
wallElementNodes[isolatedWallID].FastGetSolutionStepValue(PRESSURE, 1) = previousPressureForIsolatedWall;
}

KRATOS_CATCH(" ")
};





private:







SetActiveFlagProcess &operator=(SetActiveFlagProcess const &rOther);


}
; 




inline std::istream &operator>>(std::istream &rIStream,
SetActiveFlagProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const SetActiveFlagProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
