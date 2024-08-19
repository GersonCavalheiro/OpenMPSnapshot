

#include "compute_embedded_lift_process.h"
#include "compressible_potential_flow_application_variables.h"
#include "includes/cfd_variables.h"
#include "custom_utilities/potential_flow_utilities.h"

namespace Kratos
{
template <unsigned int Dim, unsigned int NumNodes>
ComputeEmbeddedLiftProcess<Dim, NumNodes>::ComputeEmbeddedLiftProcess(ModelPart& rModelPart,
Vector& rResultantForce
):
Process(),
mrModelPart(rModelPart),
mrResultantForce(rResultantForce)
{
}

template <unsigned int Dim, unsigned int NumNodes>
void ComputeEmbeddedLiftProcess<Dim, NumNodes>::Execute()
{
KRATOS_TRY;

mrResultantForce = ZeroVector(3);

double fx = 0.0;
double fy = 0.0;
double fz = 0.0;

#pragma omp parallel for reduction(+:fx,fy,fz)
for(int i = 0; i <  static_cast<int>(mrModelPart.NumberOfElements()); ++i) {
auto it_elem=mrModelPart.ElementsBegin()+i;
auto r_geometry = it_elem->GetGeometry();

BoundedVector<double, NumNodes> geometry_distances;
for(unsigned int i_node = 0; i_node<NumNodes; i_node++){
geometry_distances[i_node] = r_geometry[i_node].GetSolutionStepValue(GEOMETRY_DISTANCE);
}
const bool is_embedded = PotentialFlowUtilities::CheckIfElementIsCutByDistance<Dim,NumNodes>(geometry_distances);

if (is_embedded && it_elem->Is(ACTIVE)){

ModifiedShapeFunctions::Pointer pModifiedShFunc = this->pGetModifiedShapeFunctions(it_elem->pGetGeometry(), Vector(geometry_distances));

std::vector<array_1d<double,3>> cut_normal;
pModifiedShFunc -> ComputePositiveSideInterfaceAreaNormals(cut_normal,GeometryData::IntegrationMethod::GI_GAUSS_1);

std::vector<double> pressure_coefficient;
it_elem->CalculateOnIntegrationPoints(PRESSURE_COEFFICIENT,pressure_coefficient,mrModelPart.GetProcessInfo());

it_elem->SetValue(PRESSURE_COEFFICIENT,pressure_coefficient[0]);
it_elem->SetValue(NORMAL,cut_normal[0]);

fx += pressure_coefficient[0]*cut_normal[0][0];
fy += pressure_coefficient[0]*cut_normal[0][1];
fz += pressure_coefficient[0]*cut_normal[0][2];
}
}

mrResultantForce[0] = fx;
mrResultantForce[1] = fy;
mrResultantForce[2] = fz;

KRATOS_CATCH("");
}

template<>
ModifiedShapeFunctions::Pointer ComputeEmbeddedLiftProcess<2, 3>::pGetModifiedShapeFunctions(const GeomPointerType pGeometry, const Vector& rDistances) const {
return Kratos::make_unique<Triangle2D3ModifiedShapeFunctions>(pGeometry, rDistances);
}

template<>
ModifiedShapeFunctions::Pointer ComputeEmbeddedLiftProcess<3, 4>::pGetModifiedShapeFunctions(const GeomPointerType pGeometry, const Vector& rDistances) const {
return Kratos::make_unique<Tetrahedra3D4ModifiedShapeFunctions>(pGeometry, rDistances);
}

template class ComputeEmbeddedLiftProcess<2, 3>;
template class ComputeEmbeddedLiftProcess<3, 4>;
}
