

#if !defined(KRATOS_RECOVER_VOLUME_LOSSES_PROCESS_H_INCLUDED )
#define  KRATOS_RECOVER_VOLUME_LOSSES_PROCESS_H_INCLUDED



#include "includes/model_part.h"
#include "custom_utilities/mesher_utilities.hpp"
#include "processes/process.h"


namespace Kratos
{




class RecoverVolumeLossesProcess : public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION( RecoverVolumeLossesProcess );

typedef ModelPart::NodeType                   NodeType;
typedef ModelPart::ConditionType         ConditionType;
typedef ModelPart::PropertiesType       PropertiesType;
typedef ConditionType::GeometryType       GeometryType;


RecoverVolumeLossesProcess(ModelPart& rModelPart,
int EchoLevel)
: Process(Flags()), mrModelPart(rModelPart), mEchoLevel(EchoLevel)
{
mTotalVolume = 0;
}


virtual ~RecoverVolumeLossesProcess() {}



void operator()()
{
Execute();
}


void Execute()  override
{
}

void ExecuteInitialize() override
{
}

void ExecuteBeforeSolutionLoop() override
{
mTotalVolume = this->ComputeVolume(mrModelPart);
}


void ExecuteInitializeSolutionStep() override
{
KRATOS_TRY

mTotalVolume = this->RecoverVolume(mrModelPart);

KRATOS_CATCH("")
}

void ExecuteFinalizeSolutionStep() override
{

mTotalVolume = this->RecoverVolume(mrModelPart);

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
return "RecoverVolumeLossesProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "RecoverVolumeLossesProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:


private:


ModelPart& mrModelPart;

double mTotalVolume;

int mEchoLevel;



double RecoverVolume(ModelPart& rModelPart)
{

KRATOS_TRY

double Tolerance = 1e-6;
unsigned int NumberOfIterations = 5;
unsigned int iteration = -1;

double CurrentVolume = this->ComputeVolume(rModelPart);

double Error = fabs(mTotalVolume-CurrentVolume);

this->SetFreeSurfaceElements(rModelPart);
double FreeSurfaceVolume = this->ComputeFreeSurfaceVolume(rModelPart);
double FreeSurfaceArea   = this->ComputeFreeSurfaceArea(rModelPart);

double VolumeIncrement = mTotalVolume-CurrentVolume;
double Offset = VolumeIncrement/FreeSurfaceArea;

FreeSurfaceVolume += VolumeIncrement;

double CurrentFreeSurfaceVolume = 0;

while( ++iteration < NumberOfIterations && Error > Tolerance )
{

this->MoveFreeSurface(rModelPart,Offset);

CurrentFreeSurfaceVolume = this->ComputeFreeSurfaceVolume(rModelPart);

VolumeIncrement = (FreeSurfaceVolume - CurrentFreeSurfaceVolume);

Offset = (VolumeIncrement / FreeSurfaceArea );

Error = fabs(VolumeIncrement);

}


this->ResetFreeSurfaceElements(rModelPart);

CurrentVolume = this->ComputeVolume(rModelPart);


return CurrentVolume;

KRATOS_CATCH("")
}



void MoveFreeSurface(ModelPart& rModelPart, const double& rOffset)
{
KRATOS_TRY

ModelPart::NodesContainerType::iterator it_begin = rModelPart.NodesBegin();
int NumberOfNodes = rModelPart.NumberOfNodes();

#pragma omp parallel for
for (int i=0; i<NumberOfNodes; ++i)
{
ModelPart::NodesContainerType::iterator i_node = it_begin + i;
if(i_node->Is(FREE_SURFACE) && (i_node->IsNot(RIGID) && i_node->IsNot(SOLID)) ){
const array_1d<double,3>& rNormal = i_node->FastGetSolutionStepValue(NORMAL);
i_node->Coordinates() += rOffset * rNormal;
i_node->FastGetSolutionStepValue(DISPLACEMENT) += rOffset * rNormal;
i_node->FastGetSolutionStepValue(DISPLACEMENT,1) += rOffset * rNormal;
}
}

KRATOS_CATCH("")
}



void SetFreeSurfaceElements(ModelPart& rModelPart)
{
KRATOS_TRY

for(ModelPart::ElementsContainerType::iterator i_elem = rModelPart.ElementsBegin() ; i_elem != rModelPart.ElementsEnd() ; ++i_elem)
{
for(unsigned int i=0; i<i_elem->GetGeometry().size(); ++i)
{
if(i_elem->GetGeometry()[i].Is(FREE_SURFACE)){
i_elem->Set(FREE_SURFACE,true);
break;
}
}
}

KRATOS_CATCH("")
}



void ResetFreeSurfaceElements(ModelPart& rModelPart)
{
KRATOS_TRY

for(ModelPart::ElementsContainerType::iterator i_elem = rModelPart.ElementsBegin() ; i_elem != rModelPart.ElementsEnd() ; ++i_elem)
{
i_elem->Set(FREE_SURFACE,false);
}

KRATOS_CATCH("")
}


double ComputeVolume(ModelPart& rModelPart)
{
KRATOS_TRY

const unsigned int dimension = rModelPart.GetProcessInfo()[SPACE_DIMENSION];
double ModelPartVolume = 0;

if( dimension == 2 ){

ModelPart::ElementsContainerType::iterator it_begin = rModelPart.ElementsBegin();
int NumberOfElements = rModelPart.NumberOfElements();

#pragma omp parallel for reduction(+:ModelPartVolume)
for (int i=0; i<NumberOfElements; ++i)
{
ModelPart::ElementsContainerType::iterator i_elem = it_begin + i;
if( i_elem->GetGeometry().Dimension() == 2 && i_elem->Is(FLUID) )
ModelPartVolume += i_elem->GetGeometry().Area();
}
}
else{ 

ModelPart::ElementsContainerType::iterator it_begin = rModelPart.ElementsBegin();
int NumberOfElements = rModelPart.NumberOfElements();

#pragma omp parallel for reduction(+:ModelPartVolume)
for (int i=0; i<NumberOfElements; ++i)
{
ModelPart::ElementsContainerType::iterator i_elem = it_begin + i;
if( i_elem->GetGeometry().Dimension() == 3 && i_elem->Is(FLUID) )
ModelPartVolume += i_elem->GetGeometry().Volume();
}
}

return ModelPartVolume;

KRATOS_CATCH("")

}


double ComputeFreeSurfaceVolume(ModelPart& rModelPart)
{
KRATOS_TRY

const unsigned int dimension = rModelPart.GetProcessInfo()[SPACE_DIMENSION];
double ModelPartVolume = 0;

if( dimension == 2 ){

ModelPart::ElementsContainerType::iterator it_begin = rModelPart.ElementsBegin();
int NumberOfElements = rModelPart.NumberOfElements();

#pragma omp parallel for reduction(+:ModelPartVolume)
for (int i=0; i<NumberOfElements; ++i)
{
ModelPart::ElementsContainerType::iterator i_elem = it_begin + i;
if( i_elem->GetGeometry().Dimension() == 2 && i_elem->Is(FREE_SURFACE) && i_elem->Is(FLUID) )
ModelPartVolume += i_elem->GetGeometry().Area();
}
}
else{ 

ModelPart::ElementsContainerType::iterator it_begin = rModelPart.ElementsBegin();
int NumberOfElements = rModelPart.NumberOfElements();

#pragma omp parallel for reduction(+:ModelPartVolume)
for (int i=0; i<NumberOfElements; ++i)
{
ModelPart::ElementsContainerType::iterator i_elem = it_begin + i;
if( i_elem->GetGeometry().Dimension() == 3  && i_elem->Is(FREE_SURFACE) && i_elem->Is(FLUID) )
ModelPartVolume += i_elem->GetGeometry().Volume();
}
}

return ModelPartVolume;

KRATOS_CATCH("")

}



double ComputeFreeSurfaceArea(ModelPart& rModelPart)
{
KRATOS_TRY

const unsigned int dimension = rModelPart.GetProcessInfo()[SPACE_DIMENSION];
double FreeSurfaceArea = 0;

if( dimension == 2 ){

ModelPart::ElementsContainerType::iterator it_begin = rModelPart.ElementsBegin();
int NumberOfElements = rModelPart.NumberOfElements();

#pragma omp parallel for reduction(+:FreeSurfaceArea)
for (int i=0; i<NumberOfElements; ++i)
{
ModelPart::ElementsContainerType::iterator i_elem = it_begin + i;
GeometryType& rGeometry = i_elem->GetGeometry();
if( rGeometry.Dimension() == 2 && i_elem->Is(FREE_SURFACE) && i_elem->Is(FLUID) ){
for(unsigned int j=0; j<rGeometry.size()-1; ++j)
{
if(rGeometry[j].Is(FREE_SURFACE)){
for(unsigned int k=j+1; k<rGeometry.size(); ++k)
{
if(rGeometry[k].Is(FREE_SURFACE)){
FreeSurfaceArea += norm_2( rGeometry[k].Coordinates() - rGeometry[j].Coordinates() );
}
}

}
}
}
}
}
else{ 

DenseMatrix<unsigned int> lpofa; 
DenseVector<unsigned int> lnofa; 

ModelPart::ElementsContainerType::iterator it_begin = rModelPart.ElementsBegin();
int NumberOfElements = rModelPart.NumberOfElements();

#pragma omp parallel for private(lpofa,lnofa) reduction(+:FreeSurfaceArea)
for (int i=0; i<NumberOfElements; ++i)
{
ModelPart::ElementsContainerType::iterator i_elem = it_begin + i;

GeometryType& rGeometry = i_elem->GetGeometry();

if( rGeometry.Dimension() == 3 && i_elem->Is(FREE_SURFACE) && i_elem->Is(FLUID) ){

rGeometry.NodesInFaces(lpofa);
rGeometry.NumberNodesInFaces(lnofa);

for(unsigned int iface=0; iface<rGeometry.FacesNumber(); ++iface)
{
unsigned int free_surface = 0;
for(unsigned int j=1; j<=lnofa[iface]; ++j)
if(rGeometry[j].Is(FREE_SURFACE))
++free_surface;

if(free_surface==lnofa[iface])
FreeSurfaceArea+=Compute3DArea(rGeometry[lpofa(1,iface)].Coordinates(),
rGeometry[lpofa(2,iface)].Coordinates(),
rGeometry[lpofa(3,iface)].Coordinates());
}
}
}
}

return FreeSurfaceArea;

KRATOS_CATCH("")

}


double Compute3DArea(array_1d<double,3> PointA, array_1d<double,3> PointB, array_1d<double,3> PointC){
double a = MathUtils<double>::Norm3(PointA - PointB);
double b = MathUtils<double>::Norm3(PointB - PointC);
double c = MathUtils<double>::Norm3(PointC - PointA);
double s = (a+b+c) / 2.0;
double Area=std::sqrt(s*(s-a)*(s-b)*(s-c));
return Area;
}







RecoverVolumeLossesProcess& operator=(RecoverVolumeLossesProcess const& rOther);







}; 






inline std::istream& operator >> (std::istream& rIStream,
RecoverVolumeLossesProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const RecoverVolumeLossesProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  

#endif 
