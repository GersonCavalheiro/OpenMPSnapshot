
#if !defined(KRATOS_PARAMETRIC_WALL_CONTACT_SEARCH_PROCESS_H_INCLUDED )
#define  KRATOS_PARAMETRIC_WALL_CONTACT_SEARCH_PROCESS_H_INCLUDED



#ifdef _OPENMP
#include <omp.h>
#endif

#include "includes/model_part.h"
#include "utilities/openmp_utils.h"
#include "geometries/point_2d.h"
#include "geometries/point_3d.h"

#include "custom_conditions/rigid_contact/point_rigid_contact_penalty_2D_condition.hpp"
#include "custom_conditions/rigid_contact/axisym_point_rigid_contact_penalty_2D_condition.hpp"

#include "custom_conditions/rigid_contact/EP_point_rigid_contact_penalty_3D_condition.hpp"
#include "custom_conditions/rigid_contact/EP_point_rigid_contact_penalty_2D_condition.hpp"
#include "custom_conditions/rigid_contact/EP_point_rigid_contact_penalty_wP_3D_condition.hpp"
#include "custom_conditions/rigid_contact/EP_axisym_point_rigid_contact_penalty_2D_condition.hpp"




#include "contact_mechanics_application_variables.h"


namespace Kratos
{







class ParametricWallContactSearchProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION( ParametricWallContactSearchProcess );

typedef ModelPart::NodeType                   NodeType;
typedef ModelPart::ConditionType         ConditionType;
typedef ModelPart::PropertiesType       PropertiesType;
typedef ConditionType::GeometryType       GeometryType;
typedef Point2D<ModelPart::NodeType>       Point2DType;
typedef Point3D<ModelPart::NodeType>       Point3DType;
typedef FrictionLaw::Pointer           FrictionLawType;


ParametricWallContactSearchProcess(ModelPart& rMainModelPart): mrMainModelPart(rMainModelPart) {}


ParametricWallContactSearchProcess( ModelPart& rMainModelPart,
std::string rSubModelPartName,
SpatialBoundingBox::Pointer pParametricWall,
Parameters CustomParameters)
: mrMainModelPart(rMainModelPart)
{
KRATOS_TRY

mEchoLevel = 1;

mpParametricWall = pParametricWall;

Parameters DefaultParameters( R"(
{
"contact_condition_type": "PointContactCondition2D1N",
"hydraulic_condition_type": "HydraulicPointContactCondition2D1N",
"kratos_module": "KratosMultiphysics.ContactMechanicsApplication",
"friction_law_type": "FrictionLaw",
"variables_of_properties":{
"FRICTION_ACTIVE": false,
"MU_STATIC": 0.3,
"MU_DYNAMIC": 0.2,
"PENALTY_PARAMETER": 1000,
"TANGENTIAL_PENALTY_RATIO": 0.1,
"TAU_STAB": 1
}

}  )" );


CustomParameters.ValidateAndAssignDefaults(DefaultParameters);

mpConditionType = CreateConditionPrototype( CustomParameters );


if(mpConditionType.get() == nullptr)
std::cout<<" ERROR:: PROTOTYPE CONTACT WALL CONDITION NOT DEFINED PROPERLY "<<std::endl;

mContactModelPartName = rSubModelPartName;

KRATOS_CATCH(" ")

}

virtual ~ParametricWallContactSearchProcess() {}



void operator()()
{
Execute();
}




void Execute() override
{
KRATOS_TRY

if( mEchoLevel > 1 )
std::cout<<"  [PARAMETRIC_CONTACT_SEARCH]:: -START- "<<std::endl;

const ProcessInfo& rCurrentProcessInfo= mrMainModelPart.GetProcessInfo();
const double Time = rCurrentProcessInfo[TIME];

mpParametricWall->UpdateBoxPosition( Time );

ClearContactFlags();

SearchContactConditions();

this->CreateContactConditions();

if( mEchoLevel > 1 )
std::cout<<"  [PARAMETRIC_CONTACT_SEARCH]:: -END- "<<std::endl;

KRATOS_CATCH( "" )
}

void ExecuteInitialize() override
{
KRATOS_TRY

for(ModelPart::ElementsContainerType::iterator ie = mrMainModelPart.ElementsBegin(); ie!=mrMainModelPart.ElementsEnd(); ie++)
{

if( ie->GetGeometry().size() == 2 ){
for( unsigned int i=0; i<ie->GetGeometry().size(); i++ )
{
ie->GetGeometry()[i].Set(BOUNDARY,true);
}
}

}






KRATOS_CATCH( "" )
}

void ExecuteBeforeSolutionLoop() override
{
}


void ExecuteInitializeSolutionStep() override
{
}


void ExecuteFinalizeSolutionStep() override
{
}


void ExecuteBeforeOutputStep() override
{
}


void ExecuteAfterOutputStep() override
{
}


void ExecuteFinalize() override
{
}







std::string Info() const override
{
return "ParametricWallContactSearchProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ParametricWallContactSearchProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}





protected:



ModelPart&  mrMainModelPart;

SpatialBoundingBox::Pointer  mpParametricWall;

ConditionType::Pointer  mpConditionType;

PropertiesType::Pointer mpProperties;

std::string  mContactModelPartName;

int  mEchoLevel;


virtual void CreateContactConditions()
{
KRATOS_TRY

ProcessInfo& rCurrentProcessInfo= mrMainModelPart.GetProcessInfo();
double Dimension = rCurrentProcessInfo[SPACE_DIMENSION];

ModelPart::ConditionsContainerType ContactConditions;

ModelPart& rContactModelPart = mrMainModelPart.GetSubModelPart(mContactModelPartName);

if( mEchoLevel > 1 ){
std::cout<<"    ["<<rContactModelPart.Name()<<" :: CONDITIONS [OLD:"<<rContactModelPart.NumberOfConditions();
}

unsigned int id = mrMainModelPart.Conditions().back().Id() + 1;

ModelPart::NodesContainerType& rNodes = mrMainModelPart.Nodes();

for(ModelPart::NodesContainerType::ptr_iterator nd = rNodes.ptr_begin(); nd != rNodes.ptr_end(); ++nd)
{
if( (*nd)->Is(BOUNDARY) && (*nd)->Is(CONTACT) ){

ConditionType::Pointer pCondition;


if( (*nd)->Is(RIGID) ){  

GeometryType::Pointer pGeometry;
if( Dimension == 2 )
pGeometry = Kratos::make_shared<Point2DType>(*nd);
else if( Dimension == 3 )
pGeometry = Kratos::make_shared<Point3DType>(*nd);


ContactConditions.push_back(pCondition);

}
else{ 

Condition::NodesArrayType  pConditionNode;
pConditionNode.push_back( (*nd) );

ConditionType::Pointer pConditionType = FindPointCondition(rContactModelPart, (*nd) );

pCondition = pConditionType->Clone(id, pConditionNode);

pCondition->Set(CONTACT);

ContactConditions.push_back(pCondition);

}

id +=1;
}

}


rContactModelPart.Conditions().swap(ContactConditions);


if( mEchoLevel > 1 ){
std::cout<<" / NEW:"<<rContactModelPart.NumberOfConditions()<<"] "<<std::endl;
}

std::string ModelPartName;

for(ModelPart::SubModelPartIterator i_mp= mrMainModelPart.SubModelPartsBegin(); i_mp!=mrMainModelPart.SubModelPartsEnd(); i_mp++)
{
if(i_mp->Is(SOLID) && i_mp->Is(ACTIVE))
ModelPartName = i_mp->Name();
}

AddContactConditions(rContactModelPart, mrMainModelPart.GetSubModelPart(ModelPartName));


if( mEchoLevel >= 1 )
std::cout<<"  [CONTACT CANDIDATES : "<<rContactModelPart.NumberOfConditions()<<"] ("<<mContactModelPartName<<") "<<std::endl;

KRATOS_CATCH( "" )

}



void AddContactConditions(ModelPart& rOriginModelPart, ModelPart& rDestinationModelPart)
{

KRATOS_TRY


if( mEchoLevel > 1 ){
std::cout<<"    ["<<rDestinationModelPart.Name()<<" :: CONDITIONS [OLD:"<<rDestinationModelPart.NumberOfConditions();
}

for(ModelPart::ConditionsContainerType::iterator ic = rOriginModelPart.ConditionsBegin(); ic!= rOriginModelPart.ConditionsEnd(); ic++)
{

if(ic->Is(CONTACT))
rDestinationModelPart.AddCondition(*(ic.base()));

}

if( mEchoLevel > 1 ){
std::cout<<" / NEW:"<<rDestinationModelPart.NumberOfConditions()<<"] "<<std::endl;
}

KRATOS_CATCH( "" )

}










private:




ConditionType::Pointer CreateConditionPrototype( Parameters& CustomParameters )
{
KRATOS_TRY

ProcessInfo& rCurrentProcessInfo= mrMainModelPart.GetProcessInfo();
double Dimension = rCurrentProcessInfo[SPACE_DIMENSION];

unsigned int NumberOfProperties = mrMainModelPart.NumberOfProperties();

mpProperties = Kratos::make_shared<PropertiesType>(NumberOfProperties);


Parameters CustomProperties = CustomParameters["variables_of_properties"];

mpProperties->SetValue(FRICTION_ACTIVE, CustomProperties["FRICTION_ACTIVE"].GetBool());
mpProperties->SetValue(MU_STATIC, CustomProperties["MU_STATIC"].GetDouble());
mpProperties->SetValue(MU_DYNAMIC, CustomProperties["MU_DYNAMIC"].GetDouble());
mpProperties->SetValue(PENALTY_PARAMETER, CustomProperties["PENALTY_PARAMETER"].GetDouble());
mpProperties->SetValue(TANGENTIAL_PENALTY_RATIO, CustomProperties["TANGENTIAL_PENALTY_RATIO"].GetDouble());
mpProperties->SetValue(TAU_STAB, CustomProperties["TAU_STAB"].GetDouble());
mpProperties->SetValue(THICKNESS, 1.0);
mpProperties->SetValue(CONTACT_FRICTION_ANGLE, 0.0);

mrMainModelPart.AddProperties(mpProperties);

GeometryType::Pointer pGeometry;
if( Dimension == 2 )
pGeometry = Kratos::make_shared<Point2DType>(*((mrMainModelPart.Nodes().begin()).base()));
else if( Dimension == 3 )
pGeometry = Kratos::make_shared<Point3DType>(*((mrMainModelPart.Nodes().begin()).base()));


std::string ConditionName = CustomParameters["contact_condition_type"].GetString();

unsigned int LastConditionId = 1;
if( mrMainModelPart.NumberOfConditions() != 0 )
LastConditionId = mrMainModelPart.Conditions().back().Id() + 1;


if(  ConditionName == "PointContactPenaltyCondition2D1N" ){
return Kratos::make_intrusive<PointRigidContactPenalty2DCondition>(LastConditionId, pGeometry, mpProperties, mpParametricWall);
}
else if(  ConditionName == "PointContactPenaltyCondition3D1N" ){
return Kratos::make_intrusive<PointRigidContactPenalty3DCondition>(LastConditionId, pGeometry, mpProperties, mpParametricWall);
}
else if(  ConditionName == "AxisymPointContactPenaltyCondition2D1N" ){
return Kratos::make_intrusive<AxisymPointRigidContactPenalty2DCondition>(LastConditionId, pGeometry, mpProperties, mpParametricWall);
}
else if(  ConditionName == "EPPointContactPenaltyCondition3D1N" ) {
return Kratos::make_intrusive<EPPointRigidContactPenalty3DCondition>(LastConditionId, pGeometry, mpProperties, mpParametricWall);
}
else if(  ConditionName == "EPPointContactPenaltyCondition2D1N" ) {
return Kratos::make_intrusive<EPPointRigidContactPenalty2DCondition>(LastConditionId, pGeometry, mpProperties, mpParametricWall);
}
else if(  ConditionName == "EPPointContactPenaltywPCondition3D1N" ) {
return Kratos::make_intrusive<EPPointRigidContactPenaltywP3DCondition>(LastConditionId, pGeometry, mpProperties, mpParametricWall);
}
else if(  ConditionName == "EPAxisymPointContactPenaltyCondition2D1N" ) {
return Kratos::make_intrusive<EPAxisymPointRigidContactPenalty2DCondition>(LastConditionId, pGeometry, mpProperties, mpParametricWall);
} else {
std::cout << ConditionName << std::endl;
KRATOS_ERROR << "the specified contact condition does not exist " << std::endl;
}




return NULL;

KRATOS_CATCH( "" )
}


void SearchContactConditions()
{
KRATOS_TRY


ProcessInfo& rCurrentProcessInfo= mrMainModelPart.GetProcessInfo();
double Time = rCurrentProcessInfo[TIME];


#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif

ModelPart::NodesContainerType& rNodes = mrMainModelPart.Nodes();

vector<unsigned int> nodes_partition;
OpenMPUtils::CreatePartition(number_of_threads, rNodes.size(), nodes_partition);


#pragma omp parallel
{

int k = OpenMPUtils::ThisThread();
ModelPart::NodesContainerType::iterator NodesBegin = rNodes.begin() + nodes_partition[k];
ModelPart::NodesContainerType::iterator NodesEnd = rNodes.begin() + nodes_partition[k + 1];

for(ModelPart::NodesContainerType::const_iterator nd = NodesBegin; nd != NodesEnd; nd++)
{
if( nd->Is(BOUNDARY) ){

if( nd->IsNot(RIGID) )
{
double Radius = 0;

if( nd->IsNot(SLAVE) ){
}
else{
Radius = 0;
}

Vector Point(3);
Point[0] = nd->X();
Point[1] = nd->Y();
Point[2] = nd->Z();

if( mpParametricWall->IsInside(Point,Time,Radius) ){
nd->Set(CONTACT);
}
}

}

}



}













KRATOS_CATCH( "" )

}



Condition::Pointer FindPointCondition(ModelPart& rModelPart, Node::Pointer pPoint)
{

KRATOS_TRY
const ProcessInfo& rCurrentProcessInfo= mrMainModelPart.GetProcessInfo();
if ( rCurrentProcessInfo.Has(IS_RESTARTED) && rCurrentProcessInfo.Has(LOAD_RESTART) ) {
if ( rCurrentProcessInfo[IS_RESTARTED] == true) {
if ( rCurrentProcessInfo[STEP] == rCurrentProcessInfo[LOAD_RESTART] ) {
std::cout << " doing my.... ";
return mpConditionType;

}
}
}

for(ModelPart::ConditionsContainerType::iterator i_cond =rModelPart.ConditionsBegin(); i_cond!= rModelPart.ConditionsEnd(); i_cond++)
{
if( i_cond->Is(CONTACT) && i_cond->GetGeometry().size() == 1 ){
if( i_cond->GetGeometry()[0].Id() == pPoint->Id() ){
return ( *(i_cond.base()) );
}
}
}

return  mpConditionType;

KRATOS_CATCH( "" )

}


void ClearContactFlags ( )
{
KRATOS_TRY

for(ModelPart::NodesContainerType::iterator i_node = mrMainModelPart.NodesBegin(); i_node!= mrMainModelPart.NodesEnd(); i_node++)
{
if( i_node->Is(CONTACT) ){
i_node->Set(CONTACT,false);
}
}

KRATOS_CATCH( "" )
}


void RestoreContactFlags ( )
{
KRATOS_TRY

for(ModelPart::ConditionsContainerType::iterator i_cond = mrMainModelPart.ConditionsBegin(); i_cond!= mrMainModelPart.ConditionsEnd(); i_cond++)
{
if( i_cond->Is(CONTACT) ){
for(unsigned int i=0; i<i_cond->GetGeometry().size(); i++)
{
i_cond->GetGeometry()[i].Set(CONTACT,true);
}
}
}

KRATOS_CATCH( "" )
}








ParametricWallContactSearchProcess& operator=(ParametricWallContactSearchProcess const& rOther);




}; 






inline std::istream& operator >> (std::istream& rIStream,
ParametricWallContactSearchProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ParametricWallContactSearchProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
