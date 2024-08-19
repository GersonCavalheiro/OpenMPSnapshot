
#if !defined(KRATOS_CALCULATE_EMBEDDED_SIGNED_DISTANCE_TO_3D_SKIN_PROCESS_H_INCLUDED )
#define  KRATOS_CALCULATE_EMBEDDED_SIGNED_DISTANCE_TO_3D_SKIN_PROCESS_H_INCLUDED


#include <string>
#include <iostream>
#include <algorithm>

#include "includes/kratos_flags.h"

#include "includes/define.h"
#include "processes/process.h"
#include "includes/kratos_flags.h"
#include "includes/element.h"
#include "includes/model_part.h"
#include "geometries/geometry_data.h"
#include "utilities/openmp_utils.h"

namespace Kratos {






class CalculateEmbeddedSignedDistanceTo3DSkinProcess : public Process
{
public:


KRATOS_CLASS_POINTER_DEFINITION(CalculateEmbeddedSignedDistanceTo3DSkinProcess);


CalculateEmbeddedSignedDistanceTo3DSkinProcess(ModelPart& rThisModelPartStruc, ModelPart& rThisModelPartFluid, bool DiscontinuousDistance = false)
: mrSkinModelPart(rThisModelPartStruc), mrFluidModelPart(rThisModelPartFluid), mDiscontinuousDistance(DiscontinuousDistance)
{
}

~CalculateEmbeddedSignedDistanceTo3DSkinProcess() override
{
}


void operator()()
{
Execute();
}


void Execute() override
{
CalculateDiscontinuousDistanceToSkinProcess<3>::Pointer pdistance_calculator;
if(mDiscontinuousDistance)
{
pdistance_calculator = CalculateDiscontinuousDistanceToSkinProcess<3>::Pointer(
new CalculateDiscontinuousDistanceToSkinProcess<3>(mrFluidModelPart, mrSkinModelPart));
}
else
{
pdistance_calculator = CalculateDiscontinuousDistanceToSkinProcess<3>::Pointer(
new CalculateDistanceToSkinProcess<3>(mrFluidModelPart, mrSkinModelPart));
}

pdistance_calculator->Initialize();
pdistance_calculator->FindIntersections();
pdistance_calculator->CalculateDistances(pdistance_calculator->GetIntersections());


this->PeakValuesCorrection(); 

this->CalculateEmbeddedVelocity(pdistance_calculator->GetIntersections());

pdistance_calculator->Clear();
}

void Clear() override
{
}




std::string Info() const override
{
return "CalculateEmbeddedSignedDistanceTo3DSkinProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "CalculateEmbeddedSignedDistanceTo3DSkinProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}



protected:





void CalculateEmbeddedVelocity(std::vector<PointerVector<GeometricalObject>>& rIntersectedObjects)
{
const array_1d<double, 3> aux_zero = ZeroVector(3);

for (int k = 0; k < static_cast<int>(mrFluidModelPart.NumberOfElements()); ++k)
{
ModelPart::ElementsContainerType::iterator itFluidElement = mrFluidModelPart.ElementsBegin() + k;
const PointerVector<GeometricalObject>& intersected_skin_elems = rIntersectedObjects[k];

itFluidElement->SetValue(EMBEDDED_VELOCITY, aux_zero);

unsigned int intersection_counter = 0;

for(auto itSkinElement : intersected_skin_elems)
{
array_1d<double,3> emb_vel = (itSkinElement.GetGeometry()[0]).GetSolutionStepValue(VELOCITY);
emb_vel += (itSkinElement.GetGeometry()[1]).GetSolutionStepValue(VELOCITY);
emb_vel += (itSkinElement.GetGeometry()[2]).GetSolutionStepValue(VELOCITY);

itFluidElement->GetValue(EMBEDDED_VELOCITY) += emb_vel/3;
intersection_counter++;
}

if (intersection_counter!=0)
{
itFluidElement->GetValue(EMBEDDED_VELOCITY) /= intersection_counter;
}
}
}

void PeakValuesCorrection()
{
double max_distance, min_distance;
this->SetMaximumAndMinimumDistanceValues(max_distance, min_distance);

block_for_each(mrFluidModelPart.Nodes(), [&](Node& rNode){
if(rNode.IsNot(TO_SPLIT))
{
double& rnode_distance = rNode.FastGetSolutionStepValue(DISTANCE);
rnode_distance = (rnode_distance > 0.0) ? max_distance : min_distance;
}
});
}

void SetMaximumAndMinimumDistanceValues(double& max_distance, double& min_distance)
{
for (int k = 0; k < static_cast<int>(mrFluidModelPart.NumberOfElements()); ++k)
{
ModelPart::ElementsContainerType::iterator itFluidElement = mrFluidModelPart.ElementsBegin() + k;

if(itFluidElement->Is(TO_SPLIT))
{
Geometry<Node>& rGeom = itFluidElement->GetGeometry();
for (unsigned int i=0; i<rGeom.size(); ++i)
{
rGeom[i].Set(TO_SPLIT, true);
}
}
}

const unsigned int num_threads = ParallelUtilities::GetNumThreads();
OpenMPUtils::PartitionVector nodes_partition;
OpenMPUtils::DivideInPartitions(mrFluidModelPart.NumberOfNodes(), num_threads, nodes_partition);

std::vector<double> max_distance_vect(num_threads, 1.0);
std::vector<double> min_distance_vect(num_threads, 1.0);

#pragma omp parallel shared(max_distance_vect, min_distance_vect)
{
const int k = OpenMPUtils::ThisThread();
ModelPart::NodeIterator nodes_begin = mrFluidModelPart.NodesBegin() + nodes_partition[k];
ModelPart::NodeIterator nodes_end   = mrFluidModelPart.NodesBegin() + nodes_partition[k+1];

double max_local_distance = 1.0;
double min_local_distance = 1.0;

for( ModelPart::NodeIterator itFluidNode = nodes_begin; itFluidNode != nodes_end; ++itFluidNode)
{
if(itFluidNode->Is(TO_SPLIT))
{
const double node_distance = itFluidNode->FastGetSolutionStepValue(DISTANCE);
max_local_distance = (node_distance>max_local_distance) ? node_distance : max_local_distance;
min_local_distance = (node_distance<min_local_distance) ? node_distance : min_local_distance;
}
}

max_distance_vect[k] = max_local_distance;
min_distance_vect[k] = min_local_distance;
}

max_distance = max_distance_vect[0];
min_distance = min_distance_vect[0];
for (unsigned int k = 1; k < num_threads; k++)
{
max_distance = (max_distance > max_distance_vect[k]) ?  max_distance : max_distance_vect[k];
min_distance = (min_distance < min_distance_vect[k]) ?  min_distance : min_distance_vect[k];
}
}





private:


ModelPart& mrSkinModelPart;
ModelPart& mrFluidModelPart;

bool mDiscontinuousDistance;






CalculateEmbeddedSignedDistanceTo3DSkinProcess& operator=(CalculateEmbeddedSignedDistanceTo3DSkinProcess const& rOther);



}; 




inline std::istream& operator >> (std::istream& rIStream,
CalculateEmbeddedSignedDistanceTo3DSkinProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const CalculateEmbeddedSignedDistanceTo3DSkinProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  

#endif 
