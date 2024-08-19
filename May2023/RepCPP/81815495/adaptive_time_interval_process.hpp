
#if !defined(KRATOS_ADAPTIVE_TIME_INTERVAL_PROCESS_H_INCLUDED)
#define KRATOS_ADAPTIVE_TIME_INTERVAL_PROCESS_H_INCLUDED




#include "spatial_containers/spatial_containers.h"

#include "custom_processes/adaptive_time_interval_process.hpp"
#include "includes/model_part.h"
#include "utilities/openmp_utils.h"
#include "geometries/triangle_2d_3.h"
#include "geometries/triangle_2d_6.h"
#include "geometries/tetrahedra_3d_4.h"
#include "geometries/tetrahedra_3d_10.h"


namespace Kratos
{


typedef ModelPart::NodesContainerType NodesContainerType;
typedef ModelPart::ElementsContainerType ElementsContainerType;
typedef ModelPart::MeshType::GeometryType::PointsArrayType PointsArrayType;





class AdaptiveTimeIntervalProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(AdaptiveTimeIntervalProcess);


AdaptiveTimeIntervalProcess(ModelPart &rModelPart,
int EchoLevel)
: mrModelPart(rModelPart)
{
mEchoLevel = EchoLevel;
}

virtual ~AdaptiveTimeIntervalProcess()
{
}

void operator()()
{
Execute();
}


void Execute() override
{

KRATOS_TRY
KRATOS_INFO("AdaptiveTimeIntervalProcess") << " Execute() " << std::endl;

ProcessInfo &rCurrentProcessInfo = mrModelPart.GetProcessInfo();

const double initialTimeInterval = rCurrentProcessInfo[INITIAL_DELTA_TIME];
const double currentTimeInterval = rCurrentProcessInfo[CURRENT_DELTA_TIME];
double updatedTime = rCurrentProcessInfo[TIME];
double updatedTimeInterval = rCurrentProcessInfo[DELTA_TIME];
double deltaTimeToNewMilestone = initialTimeInterval;
double minimumTimeInterval = initialTimeInterval * 0.0001;
rCurrentProcessInfo.SetValue(PREVIOUS_DELTA_TIME, currentTimeInterval);

bool milestoneTimeReached = true;
bool increaseTimeInterval = true;
bool timeIntervalReduced = rCurrentProcessInfo[TIME_INTERVAL_CHANGED];

unsigned int &stepsWithChangedDt = rCurrentProcessInfo[STEPS_WITH_CHANGED_DT];

if (stepsWithChangedDt == 2)
{
if (timeIntervalReduced == false)
{
stepsWithChangedDt = 0;
}
else
{
stepsWithChangedDt = 1;
}
}
timeIntervalReduced = false;
rCurrentProcessInfo.SetValue(TIME_INTERVAL_CHANGED, false);

double tolerance = 0.0001;
updatedTime -= initialTimeInterval;
unsigned int previousMilestoneStep = updatedTime / initialTimeInterval;
deltaTimeToNewMilestone = initialTimeInterval * (previousMilestoneStep + 1) - updatedTime;

updatedTimeInterval = currentTimeInterval;

bool badVelocityConvergence = rCurrentProcessInfo[BAD_VELOCITY_CONVERGENCE];
bool badPressureConvergence = rCurrentProcessInfo[BAD_PRESSURE_CONVERGENCE];

if (updatedTimeInterval < 2.0 * minimumTimeInterval && mEchoLevel > 0 && mrModelPart.GetCommunicator().MyPID() == 0)
{
std::cout << "ATTENTION! time step much smaller than initial time step, I'll not reduce it" << std::endl;
}
if ((badPressureConvergence == true || badVelocityConvergence == true) && updatedTimeInterval > (2.0 * minimumTimeInterval))
{
updatedTimeInterval *= 0.5;
rCurrentProcessInfo.SetValue(TIME_INTERVAL_CHANGED, true);
timeIntervalReduced = true;
}

if (deltaTimeToNewMilestone < (1.0 + tolerance) * updatedTimeInterval && deltaTimeToNewMilestone > initialTimeInterval * tolerance)
{
rCurrentProcessInfo.SetValue(DELTA_TIME, deltaTimeToNewMilestone);
if (deltaTimeToNewMilestone < 0.75 * updatedTimeInterval)
{
timeIntervalReduced = true;
rCurrentProcessInfo.SetValue(TIME_INTERVAL_CHANGED, true);
}
updatedTimeInterval = deltaTimeToNewMilestone;
milestoneTimeReached = true;
}
else
{
milestoneTimeReached = false;
double ratioInitialTime=fabs(updatedTimeInterval-initialTimeInterval*0.5)/(initialTimeInterval*0.5);
if(ratioInitialTime<0.05){
updatedTimeInterval=initialTimeInterval*0.5;;
};
rCurrentProcessInfo.SetValue(DELTA_TIME, updatedTimeInterval);
}

if (timeIntervalReduced == false)
{
if (updatedTimeInterval > (2.0 * minimumTimeInterval))
{

const unsigned int dimension = mrModelPart.ElementsBegin()->GetGeometry().WorkingSpaceDimension();
if (dimension == 2)
{
CheckNodalCriterionForTimeStepReduction(updatedTimeInterval, increaseTimeInterval, timeIntervalReduced);
}
}

if (stepsWithChangedDt == 0 && increaseTimeInterval == true && initialTimeInterval > (1.0 + tolerance) * updatedTimeInterval && badVelocityConvergence == false)
{
if(increaseTimeInterval == true && (stepsWithChangedDt == 0  || stepsWithChangedDt == 2)){
IncreaseTimeInterval(updatedTimeInterval, deltaTimeToNewMilestone, tolerance, increaseTimeInterval);
}
}
else
{
increaseTimeInterval = false;
}
}

double newTimeInterval = rCurrentProcessInfo[DELTA_TIME];
double milestoneGap = fabs(newTimeInterval - deltaTimeToNewMilestone);
if (milestoneGap < 0.49 * newTimeInterval && milestoneTimeReached == false)
{
newTimeInterval += milestoneGap;
rCurrentProcessInfo.SetValue(DELTA_TIME, newTimeInterval);
milestoneTimeReached = true;
}

updatedTime += newTimeInterval;
rCurrentProcessInfo.SetValue(TIME, updatedTime);
rCurrentProcessInfo.SetValue(CURRENT_DELTA_TIME, newTimeInterval);





if (increaseTimeInterval == false && milestoneTimeReached == true && fabs(newTimeInterval - initialTimeInterval) > tolerance && !(deltaTimeToNewMilestone > newTimeInterval * (1.0 + tolerance)))
{
rCurrentProcessInfo.SetValue(CURRENT_DELTA_TIME, currentTimeInterval);
}

if (newTimeInterval < initialTimeInterval)
{
KRATOS_INFO("AdaptiveTimeIntervalProcess") << "current time " << updatedTime << " time step: new  " << newTimeInterval << " previous " << currentTimeInterval << " initial  " << initialTimeInterval << std::endl;
}
if (stepsWithChangedDt == 0 && timeIntervalReduced == true)
{
stepsWithChangedDt += 1;
}
if (stepsWithChangedDt == 1 && timeIntervalReduced == false)
{
stepsWithChangedDt += 1;
}
if ((stepsWithChangedDt == 0  || stepsWithChangedDt == 1) && increaseTimeInterval == true && timeIntervalReduced == false)
{
stepsWithChangedDt += 1;
}
KRATOS_CATCH("");
};

void CheckNodalCriterionForTimeStepReduction(double updatedTimeInterval,
bool &increaseTimeInterval,
bool &timeIntervalReduced)
{

ProcessInfo &rCurrentProcessInfo = mrModelPart.GetProcessInfo();

#pragma omp parallel
{
ModelPart::NodeIterator NodeBegin;
ModelPart::NodeIterator NodeEnd;
OpenMPUtils::PartitionedIterators(mrModelPart.Nodes(), NodeBegin, NodeEnd);
for (ModelPart::NodeIterator itNode = NodeBegin; itNode != NodeEnd; ++itNode)
{
if (itNode->IsNot(TO_ERASE) && itNode->IsNot(ISOLATED) && itNode->IsNot(SOLID))
{
const array_1d<double, 3> &Vel = itNode->FastGetSolutionStepValue(VELOCITY);
double NormVelNode = 0;
for (unsigned int d = 0; d < 3; ++d)
{
NormVelNode += Vel[d] * Vel[d];
}
double motionInStep = sqrt(NormVelNode) * updatedTimeInterval;
double unsafetyFactor = 0;
NodeWeakPtrVectorType &neighb_nodes = itNode->GetValue(NEIGHBOUR_NODES);
for (NodeWeakPtrVectorType::iterator nn = neighb_nodes.begin(); nn != neighb_nodes.end(); nn++)
{
array_1d<double, 3> CoorNeighDifference = itNode->Coordinates() - (nn)->Coordinates();
double squaredDistance = 0;
for (unsigned int d = 0; d < 3; ++d)
{
squaredDistance += CoorNeighDifference[d] * CoorNeighDifference[d];
}
double nodeDistance = sqrt(squaredDistance);
double tempUnsafetyFactor = motionInStep / nodeDistance;
if (tempUnsafetyFactor > unsafetyFactor)
{
unsafetyFactor = tempUnsafetyFactor;
}
}

if (unsafetyFactor > 0.35)
{
increaseTimeInterval = false;
if (unsafetyFactor > 1.0)
{
double temporaryTimeInterval = rCurrentProcessInfo[DELTA_TIME];
double reducedTimeInterval = 0.5 * updatedTimeInterval;
if (reducedTimeInterval < temporaryTimeInterval)
{
rCurrentProcessInfo.SetValue(DELTA_TIME, reducedTimeInterval);
rCurrentProcessInfo.SetValue(TIME_INTERVAL_CHANGED, true);
timeIntervalReduced = true;
break;
}
}
}
}
}
}
}

void CheckElementalCriterionForTimeStepReduction(bool &increaseTimeInterval)
{

ProcessInfo &rCurrentProcessInfo = mrModelPart.GetProcessInfo();

#pragma omp parallel
{
ModelPart::ElementIterator ElemBegin;
ModelPart::ElementIterator ElemEnd;
OpenMPUtils::PartitionedIterators(mrModelPart.Elements(), ElemBegin, ElemEnd);
for (ModelPart::ElementIterator itElem = ElemBegin; itElem != ElemEnd; ++itElem)
{
double temporaryTimeInterval = rCurrentProcessInfo[DELTA_TIME];
double currentElementalArea = 0;
const unsigned int dimension = (itElem)->GetGeometry().WorkingSpaceDimension();

if (dimension == 2)
{
currentElementalArea = (itElem)->GetGeometry().Area();
Geometry<Node> updatedElementCoordinates;
bool solidElement = false;
for (unsigned int i = 0; i < itElem->GetGeometry().size(); i++)
{
if (itElem->GetGeometry()[i].Is(SOLID) || itElem->GetGeometry()[i].Is(TO_ERASE) || itElem->IsNot(ACTIVE))
{
solidElement = true;
}

const array_1d<double, 3> &Vel = itElem->GetGeometry()[i].FastGetSolutionStepValue(VELOCITY);
Point updatedNodalCoordinates = Point{itElem->GetGeometry()[i].Coordinates() + Vel * temporaryTimeInterval};
updatedElementCoordinates.push_back(Node::Pointer(new Node(i, updatedNodalCoordinates.X(), updatedNodalCoordinates.Y(), updatedNodalCoordinates.Z())));
}

double newArea = 0;
if (itElem->GetGeometry().size() == 3)
{
Triangle2D3<Node> myGeometry(updatedElementCoordinates);
newArea = myGeometry.Area();
}
else if (itElem->GetGeometry().size() == 6)
{
Triangle2D6<Node> myGeometry(updatedElementCoordinates);
newArea = myGeometry.Area();
}
else
{
std::cout << "GEOMETRY NOT DEFINED" << std::endl;
}

if (solidElement == true)
{
newArea = currentElementalArea;
}

if (newArea < 0.001 * currentElementalArea && currentElementalArea > 0)
{
double reducedTimeInterval = 0.5 * temporaryTimeInterval;

if (reducedTimeInterval < temporaryTimeInterval)
{
rCurrentProcessInfo.SetValue(DELTA_TIME, reducedTimeInterval);
rCurrentProcessInfo.SetValue(TIME_INTERVAL_CHANGED, true);
increaseTimeInterval = false;
break;
}
}
else
{
Geometry<Node> updatedEnlargedElementCoordinates;

for (unsigned int i = 0; i < itElem->GetGeometry().size(); i++)
{
const array_1d<double, 3> &Vel = itElem->GetGeometry()[i].FastGetSolutionStepValue(VELOCITY);
Point updatedNodalCoordinates = Point{itElem->GetGeometry()[i].Coordinates() + Vel * temporaryTimeInterval * 2.5};
updatedEnlargedElementCoordinates.push_back(Node::Pointer(new Node(i, updatedNodalCoordinates.X(), updatedNodalCoordinates.Y(), updatedNodalCoordinates.Z())));
}

if (itElem->GetGeometry().size() == 3)
{
Triangle2D3<Node> myGeometry(updatedEnlargedElementCoordinates);
newArea = myGeometry.Area();
}
else if (itElem->GetGeometry().size() == 6)
{
Triangle2D6<Node> myGeometry(updatedEnlargedElementCoordinates);
newArea = myGeometry.Area();
}
else
{
std::cout << "GEOMETRY NOT DEFINED" << std::endl;
}

if (newArea < 0.001 * currentElementalArea && currentElementalArea > 0)
{
increaseTimeInterval = false;

}
}
}
else if (dimension == 3)
{
double currentElementalVolume = (itElem)->GetGeometry().Volume();
Geometry<Node> updatedElementCoordinates;
bool solidElement = false;
for (unsigned int i = 0; i < itElem->GetGeometry().size(); i++)
{
if (itElem->GetGeometry()[i].Is(SOLID) || itElem->IsNot(ACTIVE))
{
solidElement = true;
}
const array_1d<double, 3> &Vel = itElem->GetGeometry()[i].FastGetSolutionStepValue(VELOCITY);
Point updatedNodalCoordinates = Point{itElem->GetGeometry()[i].Coordinates() + Vel * temporaryTimeInterval};
updatedElementCoordinates.push_back(Node::Pointer(new Node(i, updatedNodalCoordinates.X(), updatedNodalCoordinates.Y(), updatedNodalCoordinates.Z())));
}

double newVolume = 0;
if (itElem->GetGeometry().size() == 4)
{
Tetrahedra3D4<Node> myGeometry(updatedElementCoordinates);
newVolume = myGeometry.Volume();
}
else if (itElem->GetGeometry().size() == 10)
{
Tetrahedra3D10<Node> myGeometry(updatedElementCoordinates);
newVolume = myGeometry.Volume();
}
else
{
std::cout << "GEOMETRY NOT DEFINED" << std::endl;
}

if (solidElement == true)
{
newVolume = currentElementalVolume;
}

if (newVolume < 0.001 * currentElementalVolume && currentElementalVolume > 0)
{
double reducedTimeInterval = 0.5 * temporaryTimeInterval;

if (reducedTimeInterval < temporaryTimeInterval)
{
rCurrentProcessInfo.SetValue(DELTA_TIME, reducedTimeInterval);

rCurrentProcessInfo.SetValue(TIME_INTERVAL_CHANGED, true);
increaseTimeInterval = false;
break;
}
}
else
{
Geometry<Node> updatedEnlargedElementCoordinates;

for (unsigned int i = 0; i < itElem->GetGeometry().size(); i++)
{
const array_1d<double, 3> &Vel = itElem->GetGeometry()[i].FastGetSolutionStepValue(VELOCITY);
Point updatedNodalCoordinates = Point{itElem->GetGeometry()[i].Coordinates() + Vel * temporaryTimeInterval * 2.5};
updatedEnlargedElementCoordinates.push_back(Node::Pointer(new Node(i, updatedNodalCoordinates.X(), updatedNodalCoordinates.Y(), updatedNodalCoordinates.Z())));
}

if (itElem->GetGeometry().size() == 4)
{
Tetrahedra3D4<Node> myGeometry(updatedEnlargedElementCoordinates);
newVolume = myGeometry.Volume();
}
else if (itElem->GetGeometry().size() == 10)
{
Tetrahedra3D10<Node> myGeometry(updatedEnlargedElementCoordinates);
newVolume = myGeometry.Volume();
}
else
{
std::cout << "GEOMETRY NOT DEFINED" << std::endl;
}

if (newVolume < 0.001 * currentElementalVolume && currentElementalVolume > 0)
{
increaseTimeInterval = false;

}
}
}
}
}
}

void IncreaseTimeInterval(double updatedTimeInterval,
double deltaTimeToNewMilestone,
double tolerance,
bool &increaseTimeInterval)
{
ProcessInfo &rCurrentProcessInfo = mrModelPart.GetProcessInfo();
double increasedTimeInterval = 2.0 * updatedTimeInterval;
if (increasedTimeInterval < deltaTimeToNewMilestone * (1.0 + tolerance))
{
rCurrentProcessInfo.SetValue(DELTA_TIME, increasedTimeInterval);

rCurrentProcessInfo.SetValue(TIME_INTERVAL_CHANGED, true);
}
else
{
increaseTimeInterval = false;
}
}





std::string Info() const override
{
return "AdaptiveTimeIntervalProcess";
}

void PrintInfo(std::ostream &rOStream) const override
{
rOStream << "AdaptiveTimeIntervalProcess";
}

void ExecuteInitialize() override
{
}

void ExecuteFinalize() override
{
}

protected:







private:
ModelPart &mrModelPart;

int mEchoLevel;







AdaptiveTimeIntervalProcess &operator=(AdaptiveTimeIntervalProcess const &rOther);



}; 




inline std::istream &operator>>(std::istream &rIStream,
AdaptiveTimeIntervalProcess &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const AdaptiveTimeIntervalProcess &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

#endif 
