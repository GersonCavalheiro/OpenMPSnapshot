
#if !defined(KRATOS_DEM_FEM_SEARCH_H_INCLUDED )
#define  KRATOS_DEM_FEM_SEARCH_H_INCLUDED

#include <string>
#include <iostream>

#include "includes/define.h"

#include "utilities/openmp_utils.h"

#include "rigid_face_geometrical_object_configure.h"
#include "spatial_containers/bins_dynamic_objects.h"
#include "spatial_containers/bins_dynamic.h"

namespace Kratos
{








class KRATOS_API(DEM_APPLICATION) DEM_FEM_Search : public SpatialSearch
{
public:

KRATOS_CLASS_POINTER_DEFINITION(DEM_FEM_Search);

typedef PointType*                                PtrPointType;
typedef std::vector<PtrPointType>*                PointVector;
typedef std::vector<PtrPointType>::iterator       PointIterator;

typedef double*                                   DistanceVector;
typedef double*                                   DistanceIterator;

typedef RigidFaceGeometricalObjectConfigure<3>        RigidFaceGeometricalConfigureType;
typedef BinsObjectDynamic<RigidFaceGeometricalConfigureType>   GeometricalBinsType;
typedef typename RigidFaceGeometricalConfigureType::ElementsContainerType GeometricalObjectType;




DEM_FEM_Search(){

mBins = NULL;

}

~DEM_FEM_Search(){
}


void SearchRigidFaceForDEMInRadiusExclusiveImplementation (
ElementsContainerType   const& rElements,
ConditionsContainerType const& rConditions,
VectorResultConditionsContainerType& rResults,
VectorDistanceType& rResultsDistance)
{
KRATOS_TRY




int MaxNumberOfElements = rConditions.size();

ElementsContainerType::ContainerType& elements_sear   = const_cast<ElementsContainerType::ContainerType&>  (rElements.GetContainer());
ConditionsContainerType::ContainerType& conditions_bins = const_cast<ConditionsContainerType::ContainerType&>(rConditions.GetContainer());

GeometricalObjectType::ContainerType BinsConditionPointerToGeometricalObjecPointerTemporalVector;
RadiusArrayType Radius_out;

int num_of_threads = ParallelUtilities::GetNumThreads();
std::vector<unsigned int> total_fem_partition_index;
OpenMPUtils::CreatePartition(num_of_threads, conditions_bins.size(), total_fem_partition_index);

std::vector<GeometricalObjectType::ContainerType> Vector_BinsConditionPointerToGeometricalObjecPointerTemporalVector(num_of_threads);

std::vector<array_1d<double, 3> > Vector_DEM_BB_LowPoint(num_of_threads); std::vector <array_1d<double, 3 > > Vector_DEM_BB_HighPoint(num_of_threads);
std::vector<array_1d<double, 3> > Vector_GLOBAL_BB_LowPoint(num_of_threads); std::vector <array_1d<double, 3 > > Vector_GLOBAL_BB_HighPoint(num_of_threads);

std::vector<double> Vector_Ref_Radius(num_of_threads);
std::vector<RadiusArrayType> Vector_Radius_out(num_of_threads);

double Global_Ref_Radius = 0.0;
double inf = std::numeric_limits<double>::infinity();

for (std::size_t i = 0; i < 3; i++) {
DEM_BB_LowPoint[i]      = inf;
DEM_BB_HighPoint[i]     = -inf;

mGlobal_BB_LowPoint[i]  = inf;
mGlobal_BB_HighPoint[i] = -inf;
}

typedef ElementsContainerType::ContainerType::iterator   Elem_iter;
typedef ConditionsContainerType::ContainerType::iterator Cond_iter;


#pragma omp parallel
{
double radius = 0.0;
int k = OpenMPUtils::ThisThread();

for(std::size_t i = 0; i < 3; i++) {
Vector_DEM_BB_LowPoint[k][i]  = inf;
Vector_DEM_BB_HighPoint[k][i] = -inf;
}

#pragma omp for
for (int p = 0; p <(int) elements_sear.size(); p++) {

Elem_iter it = elements_sear.begin() + p;
GeometryType& pGeometry = (*it)->GetGeometry();
const array_1d<double, 3 >& aux_coor = pGeometry[0].Coordinates();

SphericParticle* p_particle = dynamic_cast<SphericParticle*>((*it).get());
radius = p_particle->GetSearchRadius();

Vector_Ref_Radius[k]    = (Vector_Ref_Radius[k]  < radius) ? radius : Vector_Ref_Radius[k] ;

for(std::size_t i = 0; i < 3; i++) {
Vector_DEM_BB_LowPoint[k][i]   = (Vector_DEM_BB_LowPoint[k][i] > aux_coor[i]) ? aux_coor[i] : Vector_DEM_BB_LowPoint[k][i];
Vector_DEM_BB_HighPoint[k][i]  = (Vector_DEM_BB_HighPoint[k][i] < aux_coor[i]) ? aux_coor[i] : Vector_DEM_BB_HighPoint[k][i];
}
} 
}


for(int k = 0; k < num_of_threads; k++) {
for(std::size_t i = 0; i < 3; i++) {
DEM_BB_LowPoint[i]  = (DEM_BB_LowPoint[i] > Vector_DEM_BB_LowPoint[k][i]) ? Vector_DEM_BB_LowPoint[k][i] : DEM_BB_LowPoint[i];
DEM_BB_HighPoint[i] = (DEM_BB_HighPoint[i] < Vector_DEM_BB_HighPoint[k][i]) ? Vector_DEM_BB_HighPoint[k][i] : DEM_BB_HighPoint[i];
}

Global_Ref_Radius = (Global_Ref_Radius < Vector_Ref_Radius[k]) ? Vector_Ref_Radius[k] : Global_Ref_Radius;
}

for(std::size_t i = 0; i < 3; i++) {
DEM_BB_LowPoint[i]  -= 1.00f * Global_Ref_Radius;
DEM_BB_HighPoint[i] += 1.00f * Global_Ref_Radius;
}



#pragma omp parallel
{
int k = OpenMPUtils::ThisThread();
Vector_BinsConditionPointerToGeometricalObjecPointerTemporalVector[k].reserve(total_fem_partition_index[k+1]);

for(std::size_t i = 0; i < 3; i++) {
Vector_GLOBAL_BB_LowPoint[k][i]  = inf;
Vector_GLOBAL_BB_HighPoint[k][i] = -inf;
}

array_1d<double, 3> rHighPoint;
array_1d<double, 3> rLowPoint;

#pragma omp for private(rHighPoint,rLowPoint)
for (int c = 0; c < (int)conditions_bins.size(); c++) {

Cond_iter it = conditions_bins.begin() + c;

const GeometryType& pGeometry = (*it)->GetGeometry();
noalias(rLowPoint)  = pGeometry[0];
noalias(rHighPoint) = pGeometry[0];

for(unsigned int point = 1; point < pGeometry.size(); point++ ) {
for(unsigned int i = 0; i < 3; i++ ) {
rHighPoint[i] = ( rHighPoint[i] < pGeometry[point][i] ) ? pGeometry[point][i] : rHighPoint[i];
rLowPoint[i]  = ( rLowPoint[i]  > pGeometry[point][i] ) ? pGeometry[point][i] : rLowPoint[i];
}
}

bool add = true;

for(unsigned int i = 0; i < 3; i++) {
if(( rHighPoint[i]  < DEM_BB_LowPoint[i] ) || ( rLowPoint[i]  > DEM_BB_HighPoint[i] )) {
add = false;
break;
}
}

if(add) {
for(unsigned int i = 0; i < 3; i++ ) {
Vector_GLOBAL_BB_LowPoint[k][i]   = (Vector_GLOBAL_BB_LowPoint[k][i] > rLowPoint[i]) ? rLowPoint[i] : Vector_GLOBAL_BB_LowPoint[k][i];
Vector_GLOBAL_BB_HighPoint[k][i]  = (Vector_GLOBAL_BB_HighPoint[k][i] < rHighPoint[i]) ? rHighPoint[i] : Vector_GLOBAL_BB_HighPoint[k][i];
}
Vector_BinsConditionPointerToGeometricalObjecPointerTemporalVector[k].push_back(*it);
}
}

}

int fem_total_size = 0;
for(int k = 0; k < num_of_threads; k++) {
fem_total_size += Vector_BinsConditionPointerToGeometricalObjecPointerTemporalVector[k].size();
}

BinsConditionPointerToGeometricalObjecPointerTemporalVector.reserve(fem_total_size);

for(int k = 0; k < num_of_threads; k++) {
BinsConditionPointerToGeometricalObjecPointerTemporalVector.insert(
BinsConditionPointerToGeometricalObjecPointerTemporalVector.end(),
Vector_BinsConditionPointerToGeometricalObjecPointerTemporalVector[k].begin(),
Vector_BinsConditionPointerToGeometricalObjecPointerTemporalVector[k].end()
);

for(std::size_t i = 0; i < 3; i++) {
mGlobal_BB_LowPoint[i]  = (mGlobal_BB_LowPoint[i] > Vector_GLOBAL_BB_LowPoint[k][i]) ? Vector_GLOBAL_BB_LowPoint[k][i] : mGlobal_BB_LowPoint[i];
mGlobal_BB_HighPoint[i] = (mGlobal_BB_HighPoint[i] < Vector_GLOBAL_BB_HighPoint[k][i]) ? Vector_GLOBAL_BB_HighPoint[k][i] : mGlobal_BB_HighPoint[i];
}
}

if(BinsConditionPointerToGeometricalObjecPointerTemporalVector.size() >0 ) {


delete mBins;
mBins = new GeometricalBinsType(BinsConditionPointerToGeometricalObjecPointerTemporalVector.begin(), BinsConditionPointerToGeometricalObjecPointerTemporalVector.end());

#pragma omp parallel
{
GeometricalObjectType::ContainerType  localResults(MaxNumberOfElements);

DistanceType                          localResultsDistances(MaxNumberOfElements);
std::size_t                           NumberOfResults = 0;

#pragma omp for schedule(dynamic, 100)
for (int p = 0; p < (int)elements_sear.size(); p++) {

Elem_iter it = elements_sear.begin() + p;

GeometricalObject::Pointer go_it(*it);
bool search_particle = true;

array_1d<double, 3 > & aux_coor = go_it->GetGeometry()[0].Coordinates();

SphericParticle* p_particle = dynamic_cast<SphericParticle*>((*it).get());
double Rad = p_particle->GetSearchRadius();

for(unsigned int i = 0; i < 3; i++ ) {
search_particle &= !(aux_coor[i]  < (mGlobal_BB_LowPoint[i] - Rad) ) || (aux_coor[i]  > (mGlobal_BB_HighPoint[i] + Rad) ); 
}

if(search_particle) {

auto  ResultsPointer          = localResults.begin();
DistanceType::iterator                           ResultsDistancesPointer = localResultsDistances.begin();

NumberOfResults = (*mBins).SearchObjectsInRadiusExclusive(go_it,Rad,ResultsPointer,ResultsDistancesPointer,MaxNumberOfElements);

rResults[p].reserve(NumberOfResults);

for(auto c_it = localResults.begin(); c_it != localResults.begin() + NumberOfResults; c_it++) {
auto presult = *c_it;
Condition::Pointer condition = dynamic_pointer_cast<Condition>(presult);
rResults[p].push_back(condition);
}

rResultsDistance[p].insert(rResultsDistance[p].begin(),localResultsDistances.begin(),localResultsDistances.begin()+NumberOfResults);

}

} 
} 

}

KRATOS_CATCH("")
}


array_1d<double, 3 > GetBBHighPoint() {
return (mGlobal_BB_HighPoint);
}
array_1d<double, 3 > GetBBLowPoint() {
return (mGlobal_BB_LowPoint);
}

virtual std::string Info() const override
{
std::stringstream buffer;
buffer << "DEM_FEM_Search" ;

return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const override {rOStream << "DEM_FEM_Search";}

virtual void PrintData(std::ostream& rOStream) const override {}





protected:
















private:
array_1d<double, 3 > mGlobal_BB_HighPoint;
array_1d<double, 3 > mGlobal_BB_LowPoint;

array_1d<double, 3 > DEM_BB_HighPoint;
array_1d<double, 3 > DEM_BB_LowPoint;
GeometricalBinsType* mBins;

















DEM_FEM_Search& operator=(DEM_FEM_Search const& rOther)
{
return *this;
}

DEM_FEM_Search(DEM_FEM_Search const& rOther)
{
*this = rOther;
}




}; 


}  

#endif 
