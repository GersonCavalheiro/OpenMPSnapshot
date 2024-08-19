
#pragma once



#include "includes/variables.h"
#include "utilities/math_utils.h"

namespace Kratos
{

#if !defined(INITIAL_CURRENT)
#define INITIAL_CURRENT
enum Configuration {Initial = 0, Current = 1};
#endif

class StructuralMechanicsMathUtilities
{
public:


typedef long double                                RealType;

typedef Node                                    NodeType;

typedef Geometry<NodeType>                     GeometryType;



static inline void Comp_Orthonor_Vect(
array_1d<double, 3 > & t1g,
array_1d<double, 3 > & t2g,
array_1d<double, 3 > & t3g,
const array_1d<double, 3 > & vxe,
const array_1d<double, 3 > & vye
)
{
double n;

MathUtils<double>::CrossProduct(t3g, vxe, vye);
n = norm_2(t3g);
t3g /= n;

MathUtils<double>::CrossProduct(t2g, t3g, vxe);
n = norm_2(t2g);
t2g /= n;

MathUtils<double>::CrossProduct(t1g, t2g, t3g);
n = norm_2(t1g);
t1g /= n;
}


static inline void Comp_Orthonor_Base(
array_1d<double, 3 > & t1g,
array_1d<double, 3 > & t2g,
array_1d<double, 3 > & t3g,
const array_1d<double, 3 > & vxe,
const array_1d<double, 3 > & Xdxi,
const array_1d<double, 3 > & Xdeta
)
{
double n;

MathUtils<double>::CrossProduct(t3g, Xdxi, Xdeta);
n = norm_2(t3g);
t3g /= n;

MathUtils<double>::CrossProduct(t2g, t3g, vxe);
n = norm_2(t2g);
t2g /= n;

MathUtils<double>::CrossProduct(t1g, t2g, t3g);
n = norm_2(t1g);
t1g /= n;
}

static inline void Comp_Orthonor_Base(
BoundedMatrix<double, 3, 3 > & t,
const array_1d<double, 3 > & vxe,
const array_1d<double, 3 > & Xdxi,
const array_1d<double, 3 > & Xdeta
)
{
double n;

array_1d<double, 3 > t1g, t2g, t3g;

MathUtils<double>::CrossProduct(t3g, Xdxi, Xdeta);

n = norm_2(t3g);
t3g /= n;

MathUtils<double>::CrossProduct(t2g, t3g, vxe);
n = norm_2(t2g);
t2g /= n;

MathUtils<double>::CrossProduct(t1g, t2g, t3g);
n = norm_2(t1g);
t1g /= n;

for (std::size_t i = 0; i < 3; ++i) {
t(0, i) = t1g[i];
t(1, i) = t2g[i];
t(2, i) = t3g[i];
}
}


static inline Matrix InterpolPrismGiD(const int nG)
{
Matrix interpol;
interpol.resize(nG, 6, false);


if (nG == 1)
{
for (unsigned int i = 0; i < 6; i++)
{
interpol(0, i) = 1.0;
}
}
else if (nG == 2)
{
for (unsigned int i = 0; i < 3; i++)
{
interpol(0, i) = 1.0;
interpol(1, i) = 0.0;
}
for (unsigned int i = 3; i < 6; i++)
{
interpol(0, i) = 0.0;
interpol(1, i) = 1.0;
}
}
else if (nG == 3)
{
for (unsigned int i = 0; i < 3; i++)
{
interpol(0, i) = 0.745326;
interpol(1, i) = 0.254644;
interpol(2, i) = 0.0;
}
for (unsigned int i = 3; i < 6; i++)
{
interpol(0, i) = 0.0;
interpol(1, i) = 0.254644;
interpol(2, i) = 0.745326;
}
}
else if (nG == 4)
{
for (unsigned int i = 0; i < 3; i++)
{
interpol(0, i) = 0.455467382132614037538;
interpol(1, i) = 0.544532617867385962462;
interpol(2, i) = 0.0;
interpol(3, i) = 0.0;
}
for (unsigned int i = 3; i < 6; i++)
{
interpol(0, i) = 0.0;
interpol(1, i) = 0.0;
interpol(2, i) = 0.544532617867385962462;
interpol(3, i) = 0.455467382132614037538;
}
}
else if (nG == 5)
{
for (unsigned int i = 0; i < 3; i++)
{
interpol(0, i) = 0.062831503549096234958;
interpol(1, i) = 0.907868;
interpol(2, i) = 0.0293;
interpol(3, i) = 0.0;
interpol(4, i) = 0.0;
}
for (unsigned int i = 3; i < 6; i++)
{
interpol(0, i) = 0.0;
interpol(1, i) = 0.0;
interpol(2, i) = 0.0293;
interpol(3, i) = 0.907868;
interpol(4, i) = 0.062831503549096234958;
}
}
else if (nG == 7)
{
for (unsigned int i = 0; i < 3; i++)
{
interpol(0, i) = 0.0;
interpol(1, i) = 0.51090930312223869755;
interpol(2, i) = 0.48909069687776130245;
interpol(3, i) = 0.0;
interpol(4, i) = 0.0;
interpol(5, i) = 0.0;
interpol(6, i) = 0.0;
}
for (unsigned int i = 3; i < 6; i++)
{
interpol(0, i) = 0.0;
interpol(1, i) = 0.0;
interpol(2, i) = 0.0;
interpol(3, i) = 0.0;
interpol(4, i) = 0.48909069687776130245;
interpol(5, i) = 0.51090930312223869755;
interpol(6, i) = 0.0;
}
}
else if (nG == 11)
{
for (unsigned int i = 0; i < 3; i++)
{
interpol( 0, i) = 0.0;
interpol( 1, i) = 0.0;
interpol( 2, i) = 0.27601287860590845062;
interpol( 3, i) = 0.72398712139409154938;
interpol( 4, i) = 0.0;
interpol( 5, i) = 0.0;
interpol( 6, i) = 0.0;
interpol( 7, i) = 0.0;
interpol( 8, i) = 0.0;
interpol( 9, i) = 0.0;
interpol(10, i) = 0.0;
}
for (unsigned int i = 3; i < 6; i++)
{
interpol( 0, i) = 0.0;
interpol( 1, i) = 0.0;
interpol( 2, i) = 0.0;
interpol( 3, i) = 0.0;
interpol( 4, i) = 0.0;
interpol( 5, i) = 0.0;
interpol( 6, i) = 0.0;
interpol( 7, i) = 0.72398712139409154938;
interpol( 8, i) = 0.27601287860590845062;
interpol( 9, i) = 0.0;
interpol(10, i) = 0.0;
}
}
return interpol;
}



static inline bool SolveSecondOrderEquation(
const RealType& a,
const RealType& b,
const RealType& c,
std::vector<RealType>& solution
)
{
const RealType disc = b*b - 4.00*a*c;
RealType q = 0.0;

solution.resize(2, false);

if (b > 0.00)
{
q = -0.5 * (b + std::sqrt(disc));
}
else
{
q = -0.5 * (b - std::sqrt(disc));
}

solution[0] = q / a;
solution[1] = c / q;

return true;
}



static inline double CalculateRadius(
const Vector& N,
const GeometryType& Geom,
const Configuration ThisConfiguration = Current
)
{
double Radius = 0.0;

for (unsigned int iNode = 0; iNode < Geom.size(); iNode++)
{
if (ThisConfiguration == Current)
{
const array_1d<double, 3 > CurrentPosition = Geom[iNode].Coordinates();
Radius += CurrentPosition[0] * N[iNode];
}
else
{
const array_1d<double, 3 > DeltaDisplacement = Geom[iNode].FastGetSolutionStepValue(DISPLACEMENT) - Geom[iNode].FastGetSolutionStepValue(DISPLACEMENT,1);
const array_1d<double, 3 > CurrentPosition = Geom[iNode].Coordinates();
const array_1d<double, 3 > ReferencePosition = CurrentPosition - DeltaDisplacement;
Radius += ReferencePosition[0] * N[iNode];
}
}

return Radius;
}



static inline double CalculateRadiusPoint(
const GeometryType& Geom,
const Configuration ThisConfiguration = Current
)
{
if (ThisConfiguration == Current)
{
const array_1d<double, 3 > CurrentPosition = Geom[0].Coordinates();
return CurrentPosition[0];
}
else
{
const array_1d<double, 3 > DeltaDisplacement = Geom[0].FastGetSolutionStepValue(DISPLACEMENT) - Geom[0].FastGetSolutionStepValue(DISPLACEMENT,1);
const array_1d<double, 3 > CurrentPosition = Geom[0].Coordinates();
const array_1d<double, 3 > ReferencePosition = CurrentPosition - DeltaDisplacement;
return ReferencePosition[0];
}
}


template<int TDim>
static inline void TensorTransformation(
BoundedMatrix<double,TDim,TDim>& rOriginLeft,
BoundedMatrix<double,TDim,TDim>& rOriginRight,
BoundedMatrix<double,TDim,TDim>& rTargetLeft,
BoundedMatrix<double,TDim,TDim>& rTargetRight,
BoundedMatrix<double,TDim,TDim>& rTensor)
{
BoundedMatrix<double,TDim,TDim> metric_left = ZeroMatrix(TDim,TDim);
BoundedMatrix<double,TDim,TDim> metric_right = ZeroMatrix(TDim,TDim);
for(int i=0;i<TDim;i++){
for(int j=0;j<TDim;j++){
metric_left(i,j) += inner_prod(column(rTargetLeft,i),column(rTargetLeft,j));
metric_right(i,j) += inner_prod(column(rTargetRight,i),column(rTargetRight,j));
}
}

double det;
Matrix inv_metric_left = Matrix(TDim,TDim);
Matrix inv_metric_right = Matrix(TDim,TDim);
MathUtils<double>::InvertMatrix(Matrix(metric_left),inv_metric_left,det);
MathUtils<double>::InvertMatrix(metric_right,inv_metric_right,det);

BoundedMatrix<double,TDim,TDim> target_left_dual = ZeroMatrix(TDim,TDim); 
BoundedMatrix<double,TDim,TDim> target_right_dual = ZeroMatrix(TDim,TDim); 
for(int i=0;i<TDim;i++){
for(int j=0;j<TDim;j++){
column(target_left_dual,i) += inv_metric_left(i,j)*column(rTargetLeft,j);
column(target_right_dual,i) += inv_metric_right(i,j)*column(rTargetRight,j);
}
}

BoundedMatrix<double, TDim, TDim> transformed_tensor = ZeroMatrix(TDim, TDim); 
for(int k=0;k<TDim;k++){
for(int l=0;l<TDim;l++){
for(int i=0;i<TDim;i++){
for(int j=0;j<TDim;j++){
transformed_tensor(k,l) += rTensor(i,j)*inner_prod(column(target_left_dual,k),column(rOriginLeft,i))*inner_prod(column(target_right_dual,l),column(rOriginRight,j));
}
}
}
}
rTensor = transformed_tensor;
}

private:
};
}

