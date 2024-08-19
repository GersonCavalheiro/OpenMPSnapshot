



#if !defined(KRATOS_ULF_UTILITIES_INCLUDED )
#define  KRATOS_ULF_UTILITIES_INCLUDED



#include <string>
#include <iostream>
#include <algorithm>



#include <pybind11/pybind11.h>
#include "includes/define.h"
#include "includes/define_python.h"

#include "includes/model_part.h"
#include "includes/node.h"
#include "utilities/geometry_utilities.h"
#include "geometries/tetrahedra_3d_4.h"
#include "ULF_application.h"
#include "utilities/openmp_utils.h"

namespace Kratos
{
class UlfUtils
{
public:
typedef Node NodeType;
/

for(ModelPart::NodesContainerType::iterator in = ThisModelPart.NodesBegin();
in!=ThisModelPart.NodesEnd(); in++)
{
if((in->GetValue(NEIGHBOUR_ELEMENTS)).size() == 0 && in->FastGetSolutionStepValue(IS_STRUCTURE)==0.0 && in->FastGetSolutionStepValue(IS_LAGRANGIAN_INLET)!=1 && in->FastGetSolutionStepValue(IS_LAGRANGIAN_INLET,1)!=1.0)
{
in->Set(TO_ERASE,true);
}

}

}

bool AlphaShape(double alpha_param, Geometry<Node >& pgeom)
{
KRATOS_TRY
BoundedMatrix<double,2,2> J; 
BoundedMatrix<double,2,2> Jinv; 
static array_1d<double,2> c; 
static array_1d<double,2> rhs; 

double x0 = pgeom[0].X();
double x1 = pgeom[1].X();
double x2 = pgeom[2].X();

double y0 = pgeom[0].Y();
double y1 = pgeom[1].Y();
double y2 = pgeom[2].Y();

J(0,0)=2.0*(x1-x0);
J(0,1)=2.0*(y1-y0);
J(1,0)=2.0*(x2-x0);
J(1,1)=2.0*(y2-y0);


double detJ = J(0,0)*J(1,1)-J(0,1)*J(1,0);

Jinv(0,0) =  J(1,1);
Jinv(0,1) = -J(0,1);
Jinv(1,0) = -J(1,0);
Jinv(1,1) =  J(0,0);

BoundedMatrix<double,2,2> check;


if(detJ < 1e-12)
{
pgeom[0].GetSolutionStepValue(IS_BOUNDARY) = 1;
pgeom[1].GetSolutionStepValue(IS_BOUNDARY) = 1;
pgeom[2].GetSolutionStepValue(IS_BOUNDARY) = 1;
return false;
}

else
{

double x0_2 = x0*x0 + y0*y0;
double x1_2 = x1*x1 + y1*y1;
double x2_2 = x2*x2 + y2*y2;

Jinv /= detJ;
rhs[0] = (x1_2 - x0_2);
rhs[1] = (x2_2 - x0_2);

noalias(c) = prod(Jinv,rhs);

double radius = sqrt(pow(c[0]-x0,2)+pow(c[1]-y0,2));

double h;
h =  pgeom[0].FastGetSolutionStepValue(NODAL_H);
h += pgeom[1].FastGetSolutionStepValue(NODAL_H);
h += pgeom[2].FastGetSolutionStepValue(NODAL_H);
h *= 0.333333333;
if (radius < h*alpha_param)
{
return true;
}
else
{
return false;
}
}


KRATOS_CATCH("")
}
bool AlphaShape3D( double alpha_param, Geometry<Node >& geom	)
{
KRATOS_TRY

BoundedMatrix<double,3,3> J; 
BoundedMatrix<double,3,3> Jinv; 
array_1d<double,3> Rhs; 
array_1d<double,3> xc;
double radius=0.0;

const double x0 = geom[0].X();
const double y0 = geom[0].Y();
const double z0 = geom[0].Z();
const double x1 = geom[1].X();
const double y1 = geom[1].Y();
const double z1 = geom[1].Z();
const double x2 = geom[2].X();
const double y2 = geom[2].Y();
const double z2 = geom[2].Z();
const double x3 = geom[3].X();
const double y3 = geom[3].Y();
const double z3 = geom[3].Z();

J(0,0) = x1-x0;
J(0,1) = y1-y0;
J(0,2) = z1-z0;
J(1,0) = x2-x0;
J(1,1) = y2-y0;
J(1,2) = z2-z0;
J(2,0) = x3-x0;
J(2,1) = y3-y0;
J(2,2) = z3-z0;

Jinv(0,0) = J(1,1)*J(2,2) - J(1,2)*J(2,1);
Jinv(1,0) = -J(1,0)*J(2,2) + J(1,2)*J(2,0);
Jinv(2,0) = J(1,0)*J(2,1) - J(1,1)*J(2,0);
Jinv(0,1) = -J(0,1)*J(2,2) + J(0,2)*J(2,1);
Jinv(1,1) = J(0,0)*J(2,2) - J(0,2)*J(2,0);
Jinv(2,1) = -J(0,0)*J(2,1) + J(0,1)*J(2,0);
Jinv(0,2) = J(0,1)*J(1,2) - J(0,2)*J(1,1);
Jinv(1,2) = -J(0,0)*J(1,2) + J(0,2)*J(1,0);
Jinv(2,2) = J(0,0)*J(1,1) - J(0,1)*J(1,0);

double detJ = J(0,0)*Jinv(0,0)
+ J(0,1)*Jinv(1,0)
+ J(0,2)*Jinv(2,0);



double x0_2 = x0*x0 + y0*y0 + z0*z0;
double x1_2 = x1*x1 + y1*y1 + z1*z1;
double x2_2 = x2*x2 + y2*y2 + z2*z2;
double x3_2 = x3*x3 + y3*y3 + z3*z3;

Jinv /= detJ;

Rhs[0] = 0.5*(x1_2 - x0_2);
Rhs[1] = 0.5*(x2_2 - x0_2);
Rhs[2] = 0.5*(x3_2 - x0_2);

noalias(xc) = prod(Jinv,Rhs);
radius = pow(xc[0] - x0,2);
radius		  += pow(xc[1] - y0,2);
radius		  += pow(xc[2] - z0,2);
radius = sqrt(radius);

double h;
h =  geom[0].FastGetSolutionStepValue(NODAL_H);
h += geom[1].FastGetSolutionStepValue(NODAL_H);
h += geom[2].FastGetSolutionStepValue(NODAL_H);
h += geom[3].FastGetSolutionStepValue(NODAL_H);
h *= 0.250;

if (radius < h*alpha_param)
{
return true;
}
else
{
return false;
}

KRATOS_CATCH("")
}

/


/
}
/
}
/
private:

static BoundedMatrix<double,3,3> msJ; 
static BoundedMatrix<double,3,3> msJinv; 
static array_1d<double,3> msc; 
static array_1d<double,3> ms_rhs; 


};

}  

#endif 


