



#if !defined(KRATOS_MOVE_PART_UTILITY_FLUID_ONLY_DIFF2_INCLUDED )
#define  KRATOS_MOVE_PART_UTILITY_FLUID_ONLY_DIFF2_INCLUDED



#include <string>
#include <iostream>
#include <algorithm>



#include "includes/define.h"
#include "includes/node.h"

#include "includes/dof.h"
#include "includes/variables.h"
#include "containers/array_1d.h"
#include "containers/data_value_container.h"
#include "includes/mesh.h"
#include "utilities/math_utils.h"

#include "utilities/geometry_utilities.h"

#include "includes/model_part.h"


#include "spatial_containers/spatial_containers.h"
#include "spatial_containers/cell.h"
#include "spatial_containers/bins_dynamic_objects.h"

#include "utilities/spatial_containers_configure.h"

#include "geometries/line_2d_2.h"
#include "geometries/triangle_2d_3.h"
#include "geometries/triangle_3d_3.h"
#include "geometries/point.h"

#include "pfem_2_application_variables.h"
#include "utilities/openmp_utils.h"

#include "time.h"


namespace Kratos
{
template< unsigned int TDim>
class MoveParticleUtilityDiffFluidOnly
{
public:

typedef SpatialContainersConfigure<TDim>     Configure;
typedef typename Configure::PointType                      PointType;
typedef typename Configure::ContainerType                  ContainerType;
typedef typename Configure::IteratorType                   IteratorType;
typedef typename Configure::ResultContainerType            ResultContainerType;
typedef typename Configure::ResultIteratorType             ResultIteratorType;

KRATOS_CLASS_POINTER_DEFINITION(TransferUtility);

TransferUtility(ModelPart& calculation_model_part, ModelPart& topographic_model_part)
: mcalculation_model_part(calculation_model_part) , mtopographic_model_part(topographic_model_part)
{
KRATOS_WATCH("initializing transfer utility")

ProcessInfo& CurrentProcessInfo = mcalculation_model_part.GetProcessInfo();






ContainerType& rElements           =  mtopographic_model_part.ElementsArray();
IteratorType it_begin              =  rElements.begin();
IteratorType it_end                =  rElements.end();

typename BinsObjectDynamic<Configure>::Pointer paux = typename BinsObjectDynamic<Configure>::Pointer(new BinsObjectDynamic<Configure>(it_begin, it_end  ) );
paux.swap(mpBinsObjectDynamic);

}


~TransferUtility()
{}


void GatherInformationFromTopographicDomain()
{
KRATOS_TRY
KRATOS_WATCH("Gathering Information From Topographic Domain ")
ProcessInfo& CurrentProcessInfo = mcalculation_model_part.GetProcessInfo();
double delta_t = CurrentProcessInfo[DELTA_TIME];
array_1d<double,3> & gravity= CurrentProcessInfo[GRAVITY];

const unsigned int max_results = 1000;


ModelPart::NodesContainerType::iterator inodebegin = mcalculation_model_part.NodesBegin();


vector<unsigned int> node_partition;
#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif
OpenMPUtils::CreatePartition(number_of_threads, mcalculation_model_part.Nodes().size(), node_partition);

#pragma omp parallel for
for(int kkk=0; kkk<number_of_threads; kkk++)
{
array_1d<double,TDim+1> N;
ResultContainerType results(max_results);
ResultIteratorType result_begin = results.begin();

for(unsigned int ii=node_partition[kkk]; ii<node_partition[kkk+1]; ii++)
{
if ( (results.size()) !=max_results)
results.resize(max_results);

ModelPart::NodesContainerType::iterator inode = inodebegin+ii;
Element::Pointer pelement(*ielem.base());
Geometry<Node >& geom = ielem->GetGeometry();

ParticlePointerVector&  element_particle_pointers =  (ielem->GetValue(FLUID_PARTICLE_POINTERS));
int & number_of_particles_in_elem=ielem->GetValue(NUMBER_OF_PARTICLES);


is_found = FindNodeOnMesh(position, N ,pelement,result_begin,MaxNumberOfResults); 


}




}
}
KRATOS_CATCH("")
}




protected:


private:






bool FindNodeOnMesh( 
array_1d<double,3>& position,
array_1d<double,TDim+1>& N,
Element::Pointer & pelement,
ResultIteratorType result_begin,
const unsigned int MaxNumberOfResults)
{
typedef std::size_t SizeType;

const array_1d<double,3>& coords = position;
array_1d<double,TDim+1> aux_N;

Geometry<Node >& geom_default = pelement->GetGeometry(); 
bool is_found_1 = CalculatePosition(geom_default,coords[0],coords[1],coords[2],N);
if(is_found_1 == true)
{
return true;
}


GlobalPointersVector< Element >& neighb_elems = pelement->GetValue(NEIGHBOUR_ELEMENTS);


for (unsigned int i=0;i!=(neighb_elems.size());i++)
{
if(neighb_elems(i).get()!=nullptr)
{
Geometry<Node >& geom = neighb_elems[i].GetGeometry();
bool is_found_2 = CalculatePosition(geom,coords[0],coords[1],coords[2],N);
if (is_found_2)
{
pelement=Element::Pointer(((neighb_elems(i))));
return true;
}
}
}


SizeType results_found = mpBinsObjectDynamic->SearchObjectsInCell(coords, result_begin, MaxNumberOfResults );

if(results_found>0){
for(SizeType i = 0; i< results_found; i++)
{
Geometry<Node >& geom = (*(result_begin+i))->GetGeometry();


bool is_found = CalculatePosition(geom,coords[0],coords[1],coords[2],N);


if(is_found == true)
{
pelement=Element::Pointer((*(result_begin+i).base()));
return true;
}
}
}

return false;
}




inline bool CalculatePosition(Geometry<Node >&geom,
const double xc, const double yc, const double zc,
array_1d<double, 3 > & N
)
{
double x0 = geom[0].X();
double y0 = geom[0].Y();
double x1 = geom[1].X();
double y1 = geom[1].Y();
double x2 = geom[2].X();
double y2 = geom[2].Y();

double area = CalculateVol(x0, y0, x1, y1, x2, y2);
double inv_area = 0.0;
if (area == 0.0)
{
KRATOS_THROW_ERROR(std::logic_error, "element with zero area found", "");
} else
{
inv_area = 1.0 / area;
}


N[0] = CalculateVol(x1, y1, x2, y2, xc, yc) * inv_area;
N[1] = CalculateVol(x2, y2, x0, y0, xc, yc) * inv_area;
N[2] = CalculateVol(x0, y0, x1, y1, xc, yc) * inv_area;

if (N[0] >= 0.0 && N[1] >= 0.0 && N[2] >= 0.0 && N[0] <= 1.0 && N[1] <= 1.0 && N[2] <= 1.0) 
return true;

return false;
}


inline bool CalculatePosition(Geometry<Node >&geom,
const double xc, const double yc, const double zc,
array_1d<double, 4 > & N
)
{

double x0 = geom[0].X();
double y0 = geom[0].Y();
double z0 = geom[0].Z();
double x1 = geom[1].X();
double y1 = geom[1].Y();
double z1 = geom[1].Z();
double x2 = geom[2].X();
double y2 = geom[2].Y();
double z2 = geom[2].Z();
double x3 = geom[3].X();
double y3 = geom[3].Y();
double z3 = geom[3].Z();

double vol = CalculateVol(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3);

double inv_vol = 0.0;
if (vol < 0.0000000000001)
{
KRATOS_THROW_ERROR(std::logic_error, "element with zero vol found", "");
} else
{
inv_vol = 1.0 / vol;
}

N[0] = CalculateVol(x1, y1, z1, x3, y3, z3, x2, y2, z2, xc, yc, zc) * inv_vol;
N[1] = CalculateVol(x0, y0, z0, x1, y1, z1, x2, y2, z2, xc, yc, zc) * inv_vol;
N[2] = CalculateVol(x3, y3, z3, x1, y1, z1, x0, y0, z0, xc, yc, zc) * inv_vol;
N[3] = CalculateVol(x3, y3, z3, x0, y0, z0, x2, y2, z2, xc, yc, zc) * inv_vol;


if (N[0] >= 0.0 && N[1] >= 0.0 && N[2] >= 0.0 && N[3] >= 0.0 &&
N[0] <= 1.0 && N[1] <= 1.0 && N[2] <= 1.0 && N[3] <= 1.0)
return true;

return false;
}

inline double CalculateVol(const double x0, const double y0,
const double x1, const double y1,
const double x2, const double y2
)
{
return 0.5 * ((x1 - x0)*(y2 - y0)- (y1 - y0)*(x2 - x0));
}

inline double CalculateVol(const double x0, const double y0, const double z0,
const double x1, const double y1, const double z1,
const double x2, const double y2, const double z2,
const double x3, const double y3, const double z3
)
{
double x10 = x1 - x0;
double y10 = y1 - y0;
double z10 = z1 - z0;

double x20 = x2 - x0;
double y20 = y2 - y0;
double z20 = z2 - z0;

double x30 = x3 - x0;
double y30 = y3 - y0;
double z30 = z3 - z0;

double detJ = x10 * y20 * z30 - x10 * y30 * z20 + y10 * z20 * x30 - y10 * x20 * z30 + z10 * x20 * y30 - z10 * y20 * x30;
return detJ * 0.1666666666666666666667;
}



ModelPart& mcalculation_model_part;
ModelPart& mtopographic_model_part;

typename BinsObjectDynamic<Configure>::Pointer  mpBinsObjectDynamic;

};

}  

#endif 
