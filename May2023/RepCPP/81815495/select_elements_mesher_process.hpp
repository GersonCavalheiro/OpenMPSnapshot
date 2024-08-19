
#if !defined( KRATOS_SELECT_ELEMENTS_MESHER_PROCESS_H_INCLUDED )
#define KRATOS_SELECT_ELEMENTS_MESHER_PROCESS_H_INCLUDED




#include "containers/variables_list_data_value_container.h"
#include "spatial_containers/spatial_containers.h"

#include "custom_utilities/mesher_utilities.hpp"
#include "custom_processes/mesher_process.hpp"


namespace Kratos
{



class SelectElementsMesherProcess
: public MesherProcess
{
public:

KRATOS_CLASS_POINTER_DEFINITION( SelectElementsMesherProcess );

typedef ModelPart::ConditionType         ConditionType;
typedef ModelPart::PropertiesType       PropertiesType;
typedef ConditionType::GeometryType       GeometryType;


SelectElementsMesherProcess(ModelPart& rModelPart,
MesherUtilities::MeshingParameters& rRemeshingParameters,
int EchoLevel)
: mrModelPart(rModelPart),
mrRemesh(rRemeshingParameters)
{
mEchoLevel = EchoLevel;
}


virtual ~SelectElementsMesherProcess() {}



void operator()()
{
Execute();
}




void Execute() override
{
KRATOS_TRY

if( mEchoLevel > 0 )
std::cout<<" [ SELECT MESH ELEMENTS: ("<<mrRemesh.OutMesh.GetNumberOfElements()<<") "<<std::endl;

const int& OutNumberOfElements = mrRemesh.OutMesh.GetNumberOfElements();
mrRemesh.PreservedElements.clear();
mrRemesh.PreservedElements.resize(OutNumberOfElements);
std::fill( mrRemesh.PreservedElements.begin(), mrRemesh.PreservedElements.end(), 0 );
mrRemesh.MeshElementsSelectedFlag = true;

mrRemesh.Info->NumberOfElements=0;

bool box_side_element = false;
bool wrong_added_node = false;

unsigned int number_of_slivers = 0;

unsigned int passed_alpha_shape = 0;
unsigned int passed_inner_outer = 0;


if(mrRemesh.ExecutionOptions.IsNot(MesherUtilities::SELECT_TESSELLATION_ELEMENTS))
{
for(int el=0; el<OutNumberOfElements; ++el)
{
mrRemesh.PreservedElements[el]=1;
mrRemesh.Info->NumberOfElements+=1;
}
}
else
{
if( mEchoLevel > 0 )
std::cout<<"   Start Element Selection "<<OutNumberOfElements<<std::endl;

this->LabelEdgeNodes(mrModelPart);

unsigned int dimension = 0;
unsigned int number_of_vertices = 0;

this->GetElementDimension(dimension,number_of_vertices);

const int* OutElementList = mrRemesh.OutMesh.GetElementList();

ModelPart::NodesContainerType& rNodes = mrModelPart.Nodes();

int el = 0;
int number = 0;


for(el=0; el<OutNumberOfElements; ++el)
{
GeometryType Vertices;


wrong_added_node = false;
box_side_element = false;

NodalFlags VerticesFlags;
for(unsigned int pn=0; pn<number_of_vertices; ++pn)
{
unsigned int id = el*number_of_vertices+pn;

if(OutElementList[id]<=0)
std::cout<<" ERROR: something is wrong: nodal id < 0 "<<el<<std::endl;

if( (unsigned int)OutElementList[id] >= mrRemesh.NodalPreIds.size() ){
if(mrRemesh.Options.Is(MesherUtilities::CONTACT_SEARCH))
wrong_added_node = true;
std::cout<<" ERROR: something is wrong: node out of bounds "<<std::endl;
break;
}

if(mrRemesh.NodalPreIds[OutElementList[id]]<0){
if(mrRemesh.Options.IsNot(MesherUtilities::CONTACT_SEARCH))
std::cout<<" ERROR: something is wrong: nodal id < 0 "<<std::endl;
box_side_element = true;
break;
}

Vertices.push_back(rNodes(OutElementList[id]));

VerticesFlags.CountFlags(Vertices.back());
}


if(box_side_element || wrong_added_node){
continue;
}

bool accepted=false;


accepted = this->CheckElementBoundaries(Vertices,VerticesFlags);

double Alpha = mrRemesh.AlphaParameter;

this->GetAlphaParameter(Alpha,Vertices,VerticesFlags,dimension);


MesherUtilities MesherUtils;

if( accepted )
{
if(mrRemesh.Options.Is(MesherUtilities::CONTACT_SEARCH))
{
accepted=MesherUtils.ShrankAlphaShape(Alpha,Vertices,mrRemesh.OffsetFactor,dimension);
}
else
{
if(mrModelPart.Is(FLUID)){
accepted=MesherUtils.AlphaShape(Alpha,Vertices,dimension,4.0*mrRemesh.Refine->CriticalRadius);
}
else{ 
accepted=MesherUtils.AlphaShape(Alpha,Vertices,dimension);
}
}
}

bool self_contact = false;
if(accepted)
{
++passed_alpha_shape;

if(mrRemesh.Options.Is(MesherUtilities::CONTACT_SEARCH))
self_contact = MesherUtils.CheckSubdomain(Vertices);
}


if(accepted)
{
if(mrRemesh.Options.Is(MesherUtilities::CONTACT_SEARCH))
{
if(self_contact)
accepted = MesherUtils.CheckOuterCentre(Vertices,mrRemesh.OffsetFactor, self_contact);
}
else
{
}
}

if(accepted)
{
++passed_inner_outer;
accepted = this->CheckElementShape(Vertices,VerticesFlags,dimension,number_of_slivers);
}


if(accepted)
{
number+=1;
mrRemesh.PreservedElements[el] = number;
}


}

mrRemesh.Info->NumberOfElements=number;

}

if( mEchoLevel > 0 ){
std::cout<<"  [Preserved Elements "<<mrRemesh.Info->NumberOfElements<<"] ("<<mrModelPart.NumberOfElements() <<") :: (slivers detected: "<<number_of_slivers<<") "<<std::endl;
std::cout<<"  (passed_alpha_shape: "<<passed_alpha_shape<<", passed_inner_outer: "<<passed_inner_outer<<") "<<std::endl;
}

if( mrModelPart.IsNot(CONTACT) )
this->SelectNodesToErase();



if( mEchoLevel > 0 ){
std::cout<<"   SELECT MESH ELEMENTS ("<<mrRemesh.Info->NumberOfElements<<") ]; "<<std::endl;

}

KRATOS_CATCH( "" )

}







std::string Info() const override
{
return "SelectElementsMesherProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SelectElementsMesherProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}




protected:


ModelPart& mrModelPart;

MesherUtilities::MeshingParameters& mrRemesh;

MesherUtilities mMesherUtilities;

int mEchoLevel;

struct NodalFlags{

unsigned int  Solid;
unsigned int  Fluid;
unsigned int  Rigid;
unsigned int  Boundary;
unsigned int  FreeSurface;
unsigned int  NoWallFreeSurface;
unsigned int  Contact;
unsigned int  Inlet;
unsigned int  Isolated;
unsigned int  Sliver;
unsigned int  NewEntity;
unsigned int  OldEntity;
unsigned int  Slave;

double Radius;

NodalFlags()
{
Solid = 0;
Fluid = 0;
Rigid = 0;
Boundary = 0;
FreeSurface = 0;
NoWallFreeSurface = 0;
Contact = 0;
Inlet = 0;
Isolated = 0;
Sliver = 0;
NewEntity = 0;
OldEntity = 0;
Radius = 0;
Slave = 0;
}

void CountFlags(const Node& rNode)
{
if(rNode.Is(SOLID))
++Solid;
if(rNode.Is(FLUID))
++Fluid;
if(rNode.Is(RIGID))
++Rigid;
if(rNode.Is(BOUNDARY))
++Boundary;
if(rNode.Is(CONTACT))
++Contact;
if(rNode.Is(INLET))
++Inlet;
if(rNode.Is(ISOLATED))
++Isolated;
if(rNode.Is(FREE_SURFACE)){
++FreeSurface;
if(rNode.IsNot(SOLID) && rNode.IsNot(RIGID))
++NoWallFreeSurface;
}
if(rNode.Is(SELECTED))
++Sliver;
if(rNode.Is(NEW_ENTITY))
++NewEntity;
if(rNode.Is(OLD_ENTITY))
++OldEntity;
if(rNode.Is(SLAVE))
++Slave;

Radius+=rNode.FastGetSolutionStepValue(NODAL_H);
}

};



bool IsFirstTimeStep()
{
const ProcessInfo& rCurrentProcessInfo = mrModelPart.GetProcessInfo();
const double& CurrentTime = rCurrentProcessInfo[TIME];
const double& TimeStep = rCurrentProcessInfo[DELTA_TIME];
if(CurrentTime<=TimeStep)
return true;
else
return false;
}


void LabelEdgeNodes(ModelPart& rModelPart)
{

if( rModelPart.Is(FLUID) ){
unsigned int count_rigid;
for(ModelPart::ElementsContainerType::iterator i_elem = rModelPart.ElementsBegin() ; i_elem != rModelPart.ElementsEnd() ; ++i_elem)
{
GeometryType::PointsArrayType& vertices=i_elem->GetGeometry().Points();

count_rigid = 0;
for(unsigned int i=0; i<vertices.size(); ++i)
{
if( vertices[i].Is(RIGID) )
++count_rigid;
}

if( count_rigid == vertices.size() ){

for(unsigned int i=0; i<vertices.size(); ++i)
{
vertices[i].Set(OLD_ENTITY,true);
}
}
}


for(ModelPart::ElementsContainerType::iterator i_elem = rModelPart.ElementsBegin() ; i_elem != rModelPart.ElementsEnd() ; ++i_elem)
{
GeometryType::PointsArrayType& vertices=i_elem->GetGeometry().Points();
count_rigid = 0;
for(unsigned int i=0; i<vertices.size(); ++i)
{
if( vertices[i].Is(RIGID) )
++count_rigid;
}

if( count_rigid != vertices.size() ){

for(unsigned int i=0; i<vertices.size(); ++i)
{
vertices[i].Set(OLD_ENTITY,false);
}

}
}
}


}


void SelectNodesToErase()
{

unsigned int dimension = 0;
unsigned int number_of_vertices = 0;
unsigned int isolated_nodes = 0;

this->GetElementDimension(dimension,number_of_vertices);

int* OutElementList = mrRemesh.OutMesh.GetElementList();

ModelPart::NodesContainerType& rNodes = mrModelPart.Nodes();

const int& OutNumberOfElements = mrRemesh.OutMesh.GetNumberOfElements();

for(int el=0; el<OutNumberOfElements; ++el)
{
if( mrRemesh.PreservedElements[el] ){
for(unsigned int pn=0; pn<number_of_vertices; ++pn)
{
rNodes[OutElementList[el*number_of_vertices+pn]].Set(BLOCKED);
}
}

}

int count_release = 0;
for(ModelPart::NodesContainerType::iterator i_node = rNodes.begin() ; i_node != rNodes.end() ; ++i_node)
{
if( i_node->IsNot(BLOCKED) ){

if(mrModelPart.Is(FLUID)){

if( i_node->Is(RIGID) || i_node->Is(SOLID) || i_node->Is(INLET) ){
if( i_node->Is(TO_ERASE) ){
i_node->Set(TO_ERASE,false);
std::cout<<" WARNING TRYING TO DELETE A WALL NODE (fluid): "<<i_node->Id()<<std::endl;
}
}
else{

if(mrRemesh.ExecutionOptions.Is(MesherUtilities::KEEP_ISOLATED_NODES)){

if( i_node->IsNot(TO_ERASE) ){
i_node->Set(ISOLATED,true);
++isolated_nodes;
}

}
else{

i_node->Set(TO_ERASE,true);
if( mEchoLevel > 0 )
std::cout<<" NODE "<<i_node->Id()<<" IS INSIDE RELEASE "<<std::endl;
if( i_node->Is(BOUNDARY) )
std::cout<<" NODE "<<i_node->Id()<<" IS BOUNDARY RELEASE "<<std::endl;
++count_release;

}

}

}
else{

if( i_node->Is(RIGID) || i_node->Is(INLET) ){

if( i_node->Is(TO_ERASE) ){
i_node->Set(TO_ERASE,false);
std::cout<<" WARNING TRYING TO DELETE A WALL NODE (solid): "<<i_node->Id()<<std::endl;
}

}
else{

if(mrRemesh.ExecutionOptions.Is(MesherUtilities::KEEP_ISOLATED_NODES)){

if( i_node->IsNot(TO_ERASE) ){
i_node->Set(ISOLATED,true);
++isolated_nodes;
}

}
else{

i_node->Set(TO_ERASE);
if( mEchoLevel > 0 )
std::cout<<" NODE "<<i_node->Id()<<" IS INSIDE RELEASE "<<std::endl;
if( i_node->Is(BOUNDARY) )
std::cout<<" NODE "<<i_node->Id()<<" IS BOUNDARY RELEASE "<<std::endl;
++count_release;
}
}
}

}
else{

i_node->Set(TO_ERASE,false);
i_node->Set(ISOLATED,false);

}

i_node->Set(BLOCKED,false);
i_node->Set(OLD_ENTITY,false);
if(i_node->Is(VISITED))
i_node->Set(SELECTED,true);
else
i_node->Set(SELECTED,false);
i_node->Set(VISITED,false);

}

if( mEchoLevel > 0 ){
std::cout<<"   NUMBER OF RELEASED NODES "<<count_release<<std::endl;
std::cout<<"   NUMBER OF ISOLATED NODES "<<isolated_nodes<<std::endl;
}
}


void CheckIds(const int* OutElementList, const int& OutNumberOfElements, const unsigned int number_of_vertices)
{

unsigned int max_out_id = 0;
for(int el=0; el<OutNumberOfElements; ++el)
{
for(unsigned int pn=0; pn<number_of_vertices; ++pn)
{
unsigned int id = el*number_of_vertices+pn;
if( int(max_out_id) < OutElementList[id] )
max_out_id = OutElementList[id];
}
}

if( max_out_id >= mrRemesh.NodalPreIds.size() )
std::cout<<" ERROR ID PRE IDS "<<max_out_id<<" > "<<mrRemesh.NodalPreIds.size()<<" (nodes size:"<<mrModelPart.Nodes().size()<<")"<<std::endl;

}


void GetElementDimension(unsigned int& dimension, unsigned int& number_of_vertices)
{
if( mrModelPart.NumberOfElements() ){
ModelPart::ElementsContainerType::iterator element_begin = mrModelPart.ElementsBegin();
dimension = element_begin->GetGeometry().WorkingSpaceDimension();
number_of_vertices = element_begin->GetGeometry().size();
}
else if ( mrModelPart.NumberOfConditions() ){
ModelPart::ConditionsContainerType::iterator condition_begin = mrModelPart.ConditionsBegin();
dimension = condition_begin->GetGeometry().WorkingSpaceDimension();
if( dimension == 3 ) 
number_of_vertices = 4;
else if( dimension == 2 ) 
number_of_vertices = 3;
}

}


bool CheckElementBoundaries(GeometryType& rVertices,const NodalFlags& rVerticesFlags)
{
bool accepted = true;
unsigned int NumberOfVertices = rVertices.size();

if ( mrModelPart.Is(FLUID) ){

MesherUtilities MesherUtils;
if( rVerticesFlags.Rigid >= NumberOfVertices-1 )
accepted = !MesherUtils.CheckRigidOuterCentre(rVertices);

if( accepted ){

if( rVerticesFlags.Rigid == NumberOfVertices && rVerticesFlags.Fluid>0){
if( rVerticesFlags.Fluid < NumberOfVertices-1 ){
accepted=false;
}
else if( rVerticesFlags.Fluid == NumberOfVertices && rVerticesFlags.OldEntity >= 2  ){
accepted=false;
}
}

if( (rVerticesFlags.Rigid + rVerticesFlags.NewEntity) == NumberOfVertices && rVerticesFlags.Fluid>0){
if( rVerticesFlags.Fluid == NumberOfVertices && rVerticesFlags.OldEntity >= NumberOfVertices-1 ){
std::cout<<" OLD RIGID NEW ENTITY EDGE DISCARDED (old_entity: "<<rVerticesFlags.OldEntity<<" fluid: "<<rVerticesFlags.Fluid<<" rigid: "<<rVerticesFlags.Rigid<<" free_surface: "<<rVerticesFlags.FreeSurface<<")"<<std::endl;
accepted=false;
}
}

if( (rVerticesFlags.Solid + rVerticesFlags.Rigid) >= NumberOfVertices && rVerticesFlags.Fluid == 0)
accepted=false;

if( rVerticesFlags.Solid == NumberOfVertices )
accepted=false;
}
}

return accepted;
}


void GetAlphaParameter(double &rAlpha,GeometryType& rVertices,const NodalFlags& rVerticesFlags,const unsigned int& rDimension)
{
unsigned int NumberOfVertices = rVertices.size();

rAlpha = mrRemesh.AlphaParameter;

if( mrModelPart.Is(SOLID) ){

if( rVerticesFlags.Boundary >= NumberOfVertices )
rAlpha*=1.2;

}
else if( mrModelPart.Is(FLUID) ){

MesherUtilities MesherUtils;

if( (rVerticesFlags.Isolated+rVerticesFlags.FreeSurface) == NumberOfVertices && (rVerticesFlags.Isolated > 0 || rVerticesFlags.NoWallFreeSurface == NumberOfVertices ||  (rVerticesFlags.NoWallFreeSurface+rVerticesFlags.Isolated)== NumberOfVertices) ){
const double MaxRelativeVelocity = 1.5; 
if( MesherUtils.CheckRelativeVelocities(rVertices, MaxRelativeVelocity) ){
rAlpha=0;
for(unsigned int i=0; i<rVertices.size(); ++i)
{
if( rVertices[i].Is(ISOLATED) )
rVertices[i].Set(TO_ERASE);
}
}
}

if(rDimension==2){
this->GetTriangleFluidElementAlpha(rAlpha,rVertices,rVerticesFlags,rDimension);
}
else if(rDimension==3){
this->GetTetrahedronFluidElementAlpha(rAlpha,rVertices,rVerticesFlags,rDimension);
}
}

}


void GetTriangleFluidElementAlpha(double &rAlpha,GeometryType& rVertices,const NodalFlags& rVerticesFlags,const unsigned int& rDimension)
{

MesherUtilities MesherUtils;

double VolumeChange = 0;
double VolumeTolerance = 1.15e-4*pow(4.0*mrRemesh.Refine->CriticalRadius,rDimension);
unsigned int NumberOfVertices = rVertices.size();

if( rVerticesFlags.Fluid != NumberOfVertices ){

if( rVerticesFlags.Fluid == 0 ){
rAlpha = 0;
}
else{

VolumeChange = 0;
if(MesherUtils.CheckVolumeDecrease(rVertices,rDimension,VolumeTolerance,VolumeChange)){

if( rVerticesFlags.Isolated > 0 ){
rAlpha*=0.80;
}
else if( rVerticesFlags.Rigid == 1 || rVerticesFlags.Solid == 1 ){
if( rVerticesFlags.NoWallFreeSurface == 2 )
rAlpha*=0.60;
else
rAlpha*=0.80;
}
else if( rVerticesFlags.Rigid == 2 || rVerticesFlags.Solid == 2 || (rVerticesFlags.Rigid+rVerticesFlags.Solid)==2 ){
if( rVerticesFlags.NoWallFreeSurface == 1 ){
rAlpha*=0.60;
}
else{
rAlpha*=0.80;
}
}
else{
rAlpha*=0.80;
}

}
else{

if( VolumeChange > 0 && VolumeChange < VolumeTolerance ){
rAlpha*=0.95;
}
else{
rAlpha=0;
}
}
}
}
else{ 

if( rVerticesFlags.FreeSurface == 3 ){

if( rVerticesFlags.NoWallFreeSurface == 3 ){

VolumeChange = 0;
if(MesherUtils.CheckVolumeDecrease(rVertices,rDimension,VolumeTolerance,VolumeChange)){
const double MaxRelativeVelocity = 1.5;
if(MesherUtils.CheckRelativeVelocities(rVertices, MaxRelativeVelocity))
rAlpha=0;
else
rAlpha*=0.80;
}
else{
rAlpha*=0.60;
}
}
else if( rVerticesFlags.NoWallFreeSurface == 2 ){
rAlpha*=0.80;
}
else if( rVerticesFlags.NoWallFreeSurface == 1 ){
rAlpha*=0.80;
}
else{
rAlpha*=1.20;
}
}
else if( rVerticesFlags.FreeSurface == 2 ){

if( rVerticesFlags.NoWallFreeSurface == 2 && (rVerticesFlags.Rigid == 1 || rVerticesFlags.Solid == 1) )
rAlpha*=0.80;
else
rAlpha*=1.20;

}
else{
rAlpha*=1.20;
}
}

}



void GetTetrahedronFluidElementAlpha(double &rAlpha,GeometryType& rVertices,const NodalFlags& rVerticesFlags,const unsigned int& rDimension)
{
MesherUtilities MesherUtils;

double VolumeChange = 0;
double VolumeTolerance = 2.5e-3*pow(4.0*mrRemesh.Refine->CriticalRadius,rDimension);
unsigned int NumberOfVertices = rVertices.size();

if( rVerticesFlags.Fluid != NumberOfVertices ){ 

if( rVerticesFlags.Fluid == 0 ){
rAlpha = 0;
}
else{

VolumeChange = 0;
if(MesherUtils.CheckVolumeDecrease(rVertices,rDimension,VolumeTolerance,VolumeChange)){

if( rVerticesFlags.Isolated > 0 ){
rAlpha*=0.8;
}
else if( (rVerticesFlags.Rigid == 1 || rVerticesFlags.Solid == 1) ){
if( rVerticesFlags.NoWallFreeSurface == 3 )
rAlpha *= 0.60;
else
rAlpha *= 0.70;
}
else if( rVerticesFlags.Rigid == 2 || rVerticesFlags.Solid == 2 || (rVerticesFlags.Rigid+rVerticesFlags.Solid)==2 ){
if( rVerticesFlags.NoWallFreeSurface == 2 )
rAlpha *= 0.50;
else
rAlpha *= 0.80;
}
else if( rVerticesFlags.Rigid == 3 || rVerticesFlags.Solid == 3 || (rVerticesFlags.Rigid+rVerticesFlags.Solid)==3 ){
if( rVerticesFlags.NoWallFreeSurface == 1 ){
if( rVerticesFlags.Fluid == 1 )
rAlpha *= 0.40;
else
rAlpha *= 0.60;
}
else{
rAlpha *= 0.80;
}
}
else{
rAlpha*=0.80;
}

}
else{

if( VolumeChange > 0 && VolumeChange < VolumeTolerance && rVerticesFlags.Rigid == NumberOfVertices ){
rAlpha*=0.80;
}
else{
rAlpha=0;
}

}
}

rAlpha*=1.10;

}
else{ 

if( rVerticesFlags.FreeSurface == 4 && rVerticesFlags.Sliver == 0 ){

if( rVerticesFlags.NoWallFreeSurface == 4){

VolumeChange = 0;
if(MesherUtils.CheckVolumeDecrease(rVertices,rDimension,VolumeTolerance,VolumeChange))
rAlpha*=0.70;
else
rAlpha*=0.40;

}
else if( rVerticesFlags.NoWallFreeSurface == 3 ){
if(MesherUtils.CheckVolumeDecrease(rVertices,rDimension,VolumeTolerance,VolumeChange))
rAlpha*=0.70;
else
rAlpha*=0.50;
}
else if( rVerticesFlags.NoWallFreeSurface == 2 ){
if(MesherUtils.CheckVolumeDecrease(rVertices,rDimension,VolumeTolerance,VolumeChange))
rAlpha*=0.70;
else
rAlpha*=0.50;
}
else{
rAlpha*=0.80;
}
}
else{

if( rVerticesFlags.FreeSurface > 0 && rVerticesFlags.Rigid > 0 ){
if( MesherUtils.CheckVolumeDecrease(rVertices,rDimension,VolumeTolerance,VolumeChange) )
rAlpha*=1.50;
else
rAlpha*=1.10;
}
else if(rVerticesFlags.FreeSurface == 0 &&  rVerticesFlags.Rigid == 3 ){
rAlpha*=2.0; 
}
else{
rAlpha*=1.70;
}
}

rAlpha*=1.10;

}

}


bool CheckElementShape(GeometryType& rVertices, const NodalFlags& rVerticesFlags,const unsigned int& rDimension, unsigned int& number_of_slivers)
{
bool accepted = true;
unsigned int NumberOfVertices = rVertices.size();

if( mrModelPart.Is(SOLID) ){

int sliver = 0;
if(rDimension == 3 && NumberOfVertices==4){

Tetrahedra3D4<Node > Tetrahedron(rVertices);

MesherUtilities MesherUtils;
accepted = MesherUtils.CheckGeometryShape(Tetrahedron,sliver);

if( sliver ){
if(mrRemesh.Options.Is(MesherUtilities::CONTACT_SEARCH))
accepted = true;
else
accepted = false;

++number_of_slivers;
}
else{

if(mrRemesh.Options.Is(MesherUtilities::CONTACT_SEARCH))
accepted = false;
else
accepted = true;
}

}

}
else if ( mrModelPart.Is(FLUID) ){

if(rDimension == 2 && NumberOfVertices==3){

if( rVerticesFlags.NoWallFreeSurface > 1 && (rVerticesFlags.Rigid+rVerticesFlags.Solid)==0 ){

double MaxEdgeLength = std::numeric_limits<double>::min();
double MinEdgeLength = std::numeric_limits<double>::max();

for(unsigned int i=0; i<NumberOfVertices-1; ++i)
{
for(unsigned int j=i+1; j<NumberOfVertices; ++j)
{
double Length = norm_2(rVertices[j].Coordinates() - rVertices[i].Coordinates());
if( Length < MinEdgeLength ){
MinEdgeLength = Length;
}
if( Length > MaxEdgeLength ){
MaxEdgeLength = Length;
}

}
}
MesherUtilities MesherUtils;
const double MaxRelativeVelocity = 1.5; 
if( MinEdgeLength*5 < MaxEdgeLength ){
if( MesherUtils.CheckRelativeVelocities(rVertices, MaxRelativeVelocity) ){
std::cout<<" WARNING 2D sliver "<<std::endl;
if( rVerticesFlags.FreeSurface == NumberOfVertices)
accepted = false;
++number_of_slivers;
for(unsigned int i=0; i<NumberOfVertices; ++i)
{
if( rVertices[i].Is(VISITED) )
std::cout<<" WARNING Second sliver in the same node bis "<<std::endl;
rVertices[i].Set(VISITED);
}
}
}
}

}
else if(rDimension == 3 && NumberOfVertices==4){

if( rVerticesFlags.FreeSurface >=2 || (rVerticesFlags.Boundary == 4 && rVerticesFlags.NoWallFreeSurface !=4) ){

Tetrahedra3D4<Node > Tetrahedron(rVertices);
double Volume = Tetrahedron.Volume();

if( Volume < 0.01*pow(4.0*mrRemesh.Refine->CriticalRadius,rDimension) ){

if( rVerticesFlags.FreeSurface == NumberOfVertices)
accepted = false;

++number_of_slivers;

for(unsigned int i=0; i<NumberOfVertices; ++i)
{
rVertices[i].Set(VISITED);
}

}





}
else if( ( rVerticesFlags.Rigid == 1 || rVerticesFlags.Rigid == 2 )  ){

MesherUtilities MesherUtils;

double VolumeIncrement = MesherUtils.GetDeformationGradientDeterminant(rVertices,rDimension);

double MovedVolume = MesherUtils.GetMovedVolume(rVertices,rDimension,1.0);

if( VolumeIncrement < 0.0 || MovedVolume < 0.0 ){



++number_of_slivers;

for(unsigned int i=0; i<NumberOfVertices; ++i)
{
rVertices[i].Set(VISITED);
}
}

}

}


}
return accepted;

}



private:


SelectElementsMesherProcess& operator=(SelectElementsMesherProcess const& rOther);



}; 






inline std::istream& operator >> (std::istream& rIStream,
SelectElementsMesherProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const SelectElementsMesherProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
