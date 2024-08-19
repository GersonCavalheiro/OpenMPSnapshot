


#include "geometries/geometry.h"
#include "geometries/geometry_data.h"
#include "utilities/openmp_utils.h"
#include "utilities/variable_utils.h"

#include "drag_utilities.h"
#include "fluid_dynamics_application_variables.h"

namespace Kratos
{


array_1d<double, 3> DragUtilities::CalculateBodyFittedDrag(ModelPart& rModelPart) {
VariableUtils variable_utils;
auto drag_force = variable_utils.SumHistoricalVariable<array_1d<double,3>>(REACTION, rModelPart, 0);
drag_force *= -1.0;

return drag_force;
}

array_1d<double, 3> DragUtilities::CalculateEmbeddedDrag(ModelPart& rModelPart) {

array_1d<double, 3> drag_force = ZeroVector(3);
double& drag_x = drag_force[0];
double& drag_y = drag_force[1];
double& drag_z = drag_force[2];

array_1d<double, 3> elem_drag;

double drag_x_red = 0.0;
double drag_y_red = 0.0;
double drag_z_red = 0.0;

#pragma omp parallel for reduction(+:drag_x_red) reduction(+:drag_y_red) reduction(+:drag_z_red) private(elem_drag) schedule(dynamic)
for(int i = 0; i < static_cast<int>(rModelPart.Elements().size()); ++i){
auto it_elem = rModelPart.ElementsBegin() + i;
it_elem->Calculate(DRAG_FORCE, elem_drag, rModelPart.GetProcessInfo());
drag_x_red += elem_drag[0];
drag_y_red += elem_drag[1];
drag_z_red += elem_drag[2];
}

drag_x += drag_x_red;
drag_y += drag_y_red;
drag_z += drag_z_red;

drag_force = rModelPart.GetCommunicator().GetDataCommunicator().SumAll(drag_force);

return drag_force;
}

array_1d<double, 3> DragUtilities::CalculateEmbeddedDragCenter(const ModelPart& rModelPart)
{
double tot_cut_area = 0.0;
array_1d<double, 3> drag_force_center = ZeroVector(3);
double& r_drag_center_x = drag_force_center[0];
double& r_drag_center_y = drag_force_center[1];
double& r_drag_center_z = drag_force_center[2];

double elem_cut_area;
array_1d<double, 3> elem_drag_center;

double drag_x_center_red = 0.0;
double drag_y_center_red = 0.0;
double drag_z_center_red = 0.0;

#pragma omp parallel for reduction(+:drag_x_center_red) reduction(+:drag_y_center_red) reduction(+:drag_z_center_red) reduction(+:tot_cut_area) private(elem_drag_center, elem_cut_area) schedule(dynamic)
for(int i = 0; i < static_cast<int>(rModelPart.Elements().size()); ++i){
auto it_elem = rModelPart.ElementsBegin() + i;
it_elem->Calculate(CUTTED_AREA, elem_cut_area, rModelPart.GetProcessInfo());
it_elem->Calculate(DRAG_FORCE_CENTER, elem_drag_center, rModelPart.GetProcessInfo());
tot_cut_area += elem_cut_area;
drag_x_center_red += elem_cut_area * elem_drag_center[0];
drag_y_center_red += elem_cut_area * elem_drag_center[1];
drag_z_center_red += elem_cut_area * elem_drag_center[2];
}

r_drag_center_x = drag_x_center_red;
r_drag_center_y = drag_y_center_red;
r_drag_center_z = drag_z_center_red;

const double tol = 1.0e-12;
if (tot_cut_area > tol) {
drag_force_center /= tot_cut_area;
}

drag_force_center = rModelPart.GetCommunicator().GetDataCommunicator().SumAll(drag_force_center);

return drag_force_center;
}



inline std::ostream& operator << (
std::ostream& rOStream,
const DragUtilities& rThis) {

rThis.PrintData(rOStream);
return rOStream;
}

}
