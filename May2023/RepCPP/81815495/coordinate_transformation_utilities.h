

#ifndef KRATOS_COORDINATE_TRANSFORMATION_UTILITIES_H
#define KRATOS_COORDINATE_TRANSFORMATION_UTILITIES_H


#include "boost/numeric/ublas/matrix_proxy.hpp"

#include "includes/define.h"
#include "includes/node.h"
#include "includes/model_part.h"
#include "containers/variable.h"
#include "geometries/geometry.h"

namespace Kratos {







template<class TLocalMatrixType, class TLocalVectorType, class TValueType>
class CoordinateTransformationUtils {
public:

KRATOS_CLASS_POINTER_DEFINITION(CoordinateTransformationUtils);

typedef Node NodeType;

typedef Geometry< Node > GeometryType;




CoordinateTransformationUtils(const unsigned int DomainSize,
const unsigned int NumRowsPerNode,
const Kratos::Flags& rSelectionFlag = SLIP):
mDomainSize(DomainSize),
mBlockSize(NumRowsPerNode),
mrFlag(rSelectionFlag)
{}

virtual ~CoordinateTransformationUtils() {}




virtual void CalculateRotationOperatorPure(
TLocalMatrixType& rRotationMatrix,
const GeometryType::PointType& rThisPoint) const
{
KRATOS_TRY

if (mDomainSize == 2) {
BoundedMatrix<double, 2, 2> local_matrix;
this->LocalRotationOperatorPure(local_matrix, rThisPoint);
if (rRotationMatrix.size1() != 2 || rRotationMatrix.size2() != 2) {
rRotationMatrix.resize(2, 2, false);
}
noalias(rRotationMatrix) = local_matrix;
} else if (mDomainSize == 3) {
BoundedMatrix<double, 3, 3> local_matrix;
this->LocalRotationOperatorPure(local_matrix, rThisPoint);
if (rRotationMatrix.size1() != 3 || rRotationMatrix.size2() != 3) {
rRotationMatrix.resize(3, 3, false);
}
noalias(rRotationMatrix) = local_matrix;
} else {
KRATOS_ERROR << "Unsupported domain size [ mDomainSize = " << mDomainSize
<< " ].\n";
}

KRATOS_CATCH("");
}

void LocalRotationOperatorPure(
BoundedMatrix<double, 3, 3>& rRot,
const GeometryType::PointType& rThisPoint) const
{
const array_1d<double, 3>& rNormal = rThisPoint.FastGetSolutionStepValue(NORMAL);

double aux = rNormal[0] * rNormal[0] + rNormal[1] * rNormal[1] +
rNormal[2] * rNormal[2];
aux = sqrt(aux);
rRot(0, 0) = rNormal[0] / aux;
rRot(0, 1) = rNormal[1] / aux;
rRot(0, 2) = rNormal[2] / aux;

array_1d<double, 3> rT1;
rT1(0) = 1.0;
rT1(1) = 0.0;
rT1(2) = 0.0;
double dot = rRot(0, 0); 

if (fabs(dot) > 0.99) {
rT1(0) = 0.0;
rT1(1) = 1.0;
rT1(2) = 0.0;

dot = rRot(0, 1); 
}

rT1[0] -= dot * rRot(0, 0);
rT1[1] -= dot * rRot(0, 1);
rT1[2] -= dot * rRot(0, 2);
Normalize(rT1);
rRot(1, 0) = rT1[0];
rRot(1, 1) = rT1[1];
rRot(1, 2) = rT1[2];

rRot(2, 0) = rRot(0, 1) * rT1[2] - rRot(0, 2) * rT1[1];
rRot(2, 1) = rRot(0, 2) * rT1[0] - rRot(0, 0) * rT1[2];
rRot(2, 2) = rRot(0, 0) * rT1[1] - rRot(0, 1) * rT1[0];
}

void LocalRotationOperatorPure(
BoundedMatrix<double, 2, 2>& rRot,
const GeometryType::PointType& rThisPoint) const
{
const array_1d<double, 3>& rNormal = rThisPoint.FastGetSolutionStepValue(NORMAL);

double aux = rNormal[0] * rNormal[0] + rNormal[1] * rNormal[1];
aux = sqrt(aux);

rRot(0, 0) = rNormal[0] / aux;
rRot(0, 1) = rNormal[1] / aux;
rRot(1, 0) = -rNormal[1] / aux;
rRot(1, 1) = rNormal[0] / aux;
}


virtual void CalculateRotationOperatorPureShapeSensitivities(
TLocalMatrixType& rRotationMatrixShapeDerivative,
const std::size_t DerivativeNodeIndex,
const std::size_t DerivativeDirectionIndex,
const GeometryType::PointType& rThisPoint) const
{
KRATOS_TRY

if (mDomainSize == 2) {
BoundedMatrix<double, 2, 2> local_matrix;
this->CalculateRotationOperatorPureShapeSensitivities(
local_matrix, DerivativeNodeIndex, DerivativeDirectionIndex, rThisPoint);
if (rRotationMatrixShapeDerivative.size1() != 2 ||
rRotationMatrixShapeDerivative.size2() != 2) {
rRotationMatrixShapeDerivative.resize(2, 2, false);
}
noalias(rRotationMatrixShapeDerivative) = local_matrix;
} else if (mDomainSize == 3) {
BoundedMatrix<double, 3, 3> local_matrix;
this->CalculateRotationOperatorPureShapeSensitivities(
local_matrix, DerivativeNodeIndex, DerivativeDirectionIndex, rThisPoint);
if (rRotationMatrixShapeDerivative.size1() != 3 ||
rRotationMatrixShapeDerivative.size2() != 3) {
rRotationMatrixShapeDerivative.resize(3, 3, false);
}
noalias(rRotationMatrixShapeDerivative) = local_matrix;
} else {
KRATOS_ERROR << "Unsupported domain size [ mDomainSize = " << mDomainSize
<< " ].\n";
}


KRATOS_CATCH("");
}


virtual void CalculateRotationOperatorPureShapeSensitivities(
BoundedMatrix<double, 2, 2>& rOutput,
const std::size_t DerivativeNodeIndex,
const std::size_t DerivativeDirectionIndex,
const GeometryType::PointType& rThisPoint) const
{
KRATOS_TRY

KRATOS_ERROR_IF(!rThisPoint.SolutionStepsDataHas(NORMAL))
<< "NORMAL is not found in node at " << rThisPoint.Coordinates() << ".";
KRATOS_ERROR_IF(!rThisPoint.Has(NORMAL_SHAPE_DERIVATIVE))
<< "NORMAL_SHAPE_DERIVATIVE is not found in node [ Node.Id() = "
<< rThisPoint.Id() << " ] at " << rThisPoint.Coordinates() << ".";

const array_1d<double, 3>& r_nodal_normal =
rThisPoint.FastGetSolutionStepValue(NORMAL);
const double nodal_normal_magnitude = norm_2(r_nodal_normal);

KRATOS_ERROR_IF(nodal_normal_magnitude == 0.0)
<< "NORMAL at node " << rThisPoint.Coordinates()
<< " is not properly initialized.";

const Matrix& r_sensitivity_values = rThisPoint.GetValue(NORMAL_SHAPE_DERIVATIVE);

KRATOS_DEBUG_ERROR_IF(r_sensitivity_values.size2() != 2)
<< "NORMAL_SHAPE_DERIVATIVE is not properly initialized at node [ Node.Id() = "
<< rThisPoint.Id() << " ] "
<< rThisPoint.Coordinates() << " to calculate 2D rotation operator shape sensitivities. [ required number of columns = 2, available number of columns = "
<< r_sensitivity_values.size2() << " ].";

const std::size_t require_rows = (DerivativeNodeIndex + 1) * 2;
KRATOS_DEBUG_ERROR_IF(r_sensitivity_values.size1() < require_rows)
<< "NORMAL_SHAPE_DERIVATIVE is not properly initialized at node [ Node.Id() = "
<< rThisPoint.Id() << " ] "
<< rThisPoint.Coordinates() << " to calculate 2D rotation operator shape sensitivities. [ required number of rows >= "
<< require_rows
<< ", available number of rows = " << r_sensitivity_values.size1() << " ].";

const Vector& r_nodal_normal_derivatives =
row(r_sensitivity_values, DerivativeNodeIndex * 2 + DerivativeDirectionIndex);

rOutput(0, 0) = r_nodal_normal_derivatives[0] / nodal_normal_magnitude;
rOutput(0, 1) = r_nodal_normal_derivatives[1] / nodal_normal_magnitude;
rOutput(1, 0) = -r_nodal_normal_derivatives[1] / nodal_normal_magnitude;
rOutput(1, 1) = r_nodal_normal_derivatives[0] / nodal_normal_magnitude;

const double nodal_normal_magnitude_derivative =
(r_nodal_normal[0] * r_nodal_normal_derivatives[0] +
r_nodal_normal[1] * r_nodal_normal_derivatives[1]) /
nodal_normal_magnitude;

const double coeff = nodal_normal_magnitude_derivative /
(std::pow(nodal_normal_magnitude, 2));

rOutput(0, 0) -= r_nodal_normal[0] * coeff;
rOutput(0, 1) -= r_nodal_normal[1] * coeff;
rOutput(1, 0) -= -r_nodal_normal[1] * coeff;
rOutput(1, 1) -= r_nodal_normal[0] * coeff;

KRATOS_CATCH("");
}


virtual void CalculateRotationOperatorPureShapeSensitivities(
BoundedMatrix<double, 3, 3>& rOutput,
const std::size_t DerivativeNodeIndex,
const std::size_t DerivativeDirectionIndex,
const GeometryType::PointType& rThisPoint) const
{
KRATOS_TRY

KRATOS_ERROR_IF(!rThisPoint.SolutionStepsDataHas(NORMAL))
<< "NORMAL is not found in node at " << rThisPoint.Coordinates() << ".";
KRATOS_ERROR_IF(!rThisPoint.Has(NORMAL_SHAPE_DERIVATIVE))
<< "NORMAL_SHAPE_DERIVATIVE is not found in node at "
<< rThisPoint.Coordinates() << ".";

const array_1d<double, 3>& r_nodal_normal =
rThisPoint.FastGetSolutionStepValue(NORMAL);
const double nodal_normal_magnitude = norm_2(r_nodal_normal);

KRATOS_ERROR_IF(nodal_normal_magnitude == 0.0)
<< "NORMAL at node " << rThisPoint.Coordinates()
<< " is not properly initialized.";

const Matrix& r_sensitivity_values = rThisPoint.GetValue(NORMAL_SHAPE_DERIVATIVE);

KRATOS_DEBUG_ERROR_IF(r_sensitivity_values.size2() != 3)
<< "NORMAL_SHAPE_DERIVATIVE is not properly initialized at node "
<< rThisPoint.Coordinates() << " to calculate 3D rotation operator shape sensitivities. [ required number of columns = 3, available number of columns = "
<< r_sensitivity_values.size2() << " ].";

const std::size_t require_rows = (DerivativeNodeIndex + 1) * 3;
KRATOS_DEBUG_ERROR_IF(r_sensitivity_values.size1() < require_rows)
<< "NORMAL_SHAPE_DERIVATIVE is not properly initialized at node "
<< rThisPoint.Coordinates() << " to calculate 3D rotation operator shape sensitivities. [ required number of rows >= "
<< require_rows
<< ", available number of rows = " << r_sensitivity_values.size1() << " ].";

const Vector& r_nodal_normal_derivative =
row(r_sensitivity_values, DerivativeNodeIndex * 3 + DerivativeDirectionIndex);

const double nodal_normal_magnitude_derivative = VectorNormDerivative(nodal_normal_magnitude, r_nodal_normal, r_nodal_normal_derivative);
const array_1d<double, 3>& unit_normal = r_nodal_normal / nodal_normal_magnitude;
const array_1d<double, 3>& unit_normal_derivative = UnitVectorDerivative(nodal_normal_magnitude, nodal_normal_magnitude_derivative, r_nodal_normal, r_nodal_normal_derivative);

rOutput(0, 0) = unit_normal_derivative[0];
rOutput(0, 1) = unit_normal_derivative[1];
rOutput(0, 2) = unit_normal_derivative[2];

array_1d<double, 3> rT1(3, 0.0);
rT1[0] = 1.0;
double dot = unit_normal[0];
double dot_derivative = unit_normal_derivative[0];

if (std::abs(dot) > 0.99) {
rT1[0] = 0.0;
rT1[1] = 1.0;
dot = unit_normal[1];
dot_derivative = unit_normal_derivative[1];
}

noalias(rT1) -=  unit_normal * dot;
const double rT1_norm = norm_2(rT1);
const array_1d<double, 3>& unit_rT1 = rT1 / rT1_norm;

const array_1d<double, 3>& rT1_derivative = (unit_normal_derivative * dot + unit_normal * dot_derivative) * -1.0;

const double rT1_norm_derivative = VectorNormDerivative(rT1_norm, rT1, rT1_derivative);

const array_1d<double, 3>& unit_rT1_derivative =
UnitVectorDerivative(rT1_norm, rT1_norm_derivative, rT1, rT1_derivative);

rOutput(1, 0) = unit_rT1_derivative[0];
rOutput(1, 1) = unit_rT1_derivative[1];
rOutput(1, 2) = unit_rT1_derivative[2];

rOutput(2, 0) = unit_normal_derivative[1] * unit_rT1[2]
+ unit_normal[1] * unit_rT1_derivative[2]
- unit_normal_derivative[2] * unit_rT1[1]
- unit_normal[2] * unit_rT1_derivative[1];


rOutput(2, 1) = unit_normal_derivative[2] * unit_rT1[0]
+ unit_normal[2] * unit_rT1_derivative[0]
- unit_normal_derivative[0] * unit_rT1[2]
- unit_normal[0] * unit_rT1_derivative[2];

rOutput(2, 2) = unit_normal_derivative[0] * unit_rT1[1]
+ unit_normal[0] * unit_rT1_derivative[1]
- unit_normal_derivative[1] * unit_rT1[0]
- unit_normal[1] * unit_rT1_derivative[0];

KRATOS_CATCH("");
}


virtual void Rotate(TLocalMatrixType& rLocalMatrix,
TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const
{
if(mBlockSize != mDomainSize) 
{
if(mDomainSize == 2) RotateAux<2,3>(rLocalMatrix,rLocalVector,rGeometry);
if(mDomainSize == 3) RotateAux<3,4>(rLocalMatrix,rLocalVector,rGeometry);
}
else 
{
if(mDomainSize == 2) RotateAuxPure<2>(rLocalMatrix,rLocalVector,rGeometry);
if(mDomainSize == 3) RotateAuxPure<3>(rLocalMatrix,rLocalVector,rGeometry);
}

}

virtual void Rotate(TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const
{


if (rLocalVector.size() > 0)
{
if(mBlockSize != mDomainSize) 
{
for(unsigned int j = 0; j < rGeometry.PointsNumber(); ++j)
{
if( this->IsSlip(rGeometry[j]) )
{
if(mDomainSize == 3)
{
array_1d<double,4> aux,aux1;
BoundedMatrix<double,4,4> rRot;
LocalRotationOperator3D<4>(rRot,rGeometry[j]);

for(unsigned int k=0; k<4; k++)
aux[k] = rLocalVector[j*mBlockSize+k];

noalias(aux1) = prod(rRot,aux);

for(unsigned int k=0; k<4; k++)
rLocalVector[j*mBlockSize+k] = aux1[k];
}
else
{
array_1d<double,3> aux,aux1;
BoundedMatrix<double,3,3> rRot;
LocalRotationOperator2D<3>(rRot,rGeometry[j]);

for(unsigned int k=0; k<3; k++)
{
aux[k] = rLocalVector[j*mBlockSize+k];
}

noalias(aux1) = prod(rRot,aux);

for(unsigned int k=0; k<3; k++)
rLocalVector[j*mBlockSize+k] = aux1[k];
}
}
}

}
else 
{
for(unsigned int j = 0; j < rGeometry.PointsNumber(); ++j)
{
if( this->IsSlip(rGeometry[j]) )
{
if(mDomainSize == 3)
{
array_1d<double,3> aux,aux1;
BoundedMatrix<double,3,3> rRot;
LocalRotationOperatorPure(rRot,rGeometry[j]);

for(unsigned int k=0; k<3; k++)
aux[k] = rLocalVector[j*mBlockSize+k];

noalias(aux1) = prod(rRot,aux);

for(unsigned int k=0; k<3; k++)
rLocalVector[j*mBlockSize+k] = aux1[k];
}
else
{
array_1d<double,2> aux,aux1;
BoundedMatrix<double,2,2> rRot;
LocalRotationOperatorPure(rRot,rGeometry[j]);

for(unsigned int k=0; k<2; k++)
aux[k] = rLocalVector[j*mBlockSize+k];

noalias(aux1) = prod(rRot,aux);

for(unsigned int k=0; k<2; k++)
rLocalVector[j*mBlockSize+k] = aux1[k];
}
}
}

}

}

}


virtual void ApplySlipCondition(TLocalMatrixType& rLocalMatrix,
TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const
{
const unsigned int LocalSize = rLocalVector.size(); 

if (LocalSize > 0)
{
for(unsigned int itNode = 0; itNode < rGeometry.PointsNumber(); ++itNode)
{
if( this->IsSlip(rGeometry[itNode]))
{
unsigned int j = itNode * mBlockSize;

array_1d<double,3> VMesh = rGeometry[itNode].FastGetSolutionStepValue(MESH_VELOCITY);
VMesh -= rGeometry[itNode].FastGetSolutionStepValue(VELOCITY);
array_1d<double,3> rN = rGeometry[itNode].FastGetSolutionStepValue(NORMAL);
this->Normalize(rN);

for( unsigned int i = 0; i < j; ++i)
{
rLocalMatrix(i,j) = 0.0;
rLocalMatrix(j,i) = 0.0;
}
for( unsigned int i = j+1; i < LocalSize; ++i)
{
rLocalMatrix(i,j) = 0.0;
rLocalMatrix(j,i) = 0.0;
}

rLocalVector(j) = inner_prod(rN,VMesh);
rLocalMatrix(j,j) = 1.0;
}
}
}
}

virtual void ApplySlipCondition(TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const
{
if (rLocalVector.size() > 0)
{
for(unsigned int itNode = 0; itNode < rGeometry.PointsNumber(); ++itNode)
{
if( this->IsSlip(rGeometry[itNode]) )
{
unsigned int j = itNode * mBlockSize;

array_1d<double,3> VMesh = rGeometry[itNode].FastGetSolutionStepValue(MESH_VELOCITY);
VMesh -= rGeometry[itNode].FastGetSolutionStepValue(VELOCITY);
array_1d<double,3> rN = rGeometry[itNode].FastGetSolutionStepValue(NORMAL);
this->Normalize(rN);

rLocalVector[j] = inner_prod(rN,VMesh);
}
}
}
}

virtual void RotateVelocities(ModelPart& rModelPart) const
{
TLocalVectorType Vel(mDomainSize);
TLocalVectorType Tmp(mDomainSize);

ModelPart::NodeIterator it_begin = rModelPart.NodesBegin();
#pragma omp parallel for firstprivate(Vel,Tmp)
for(int iii=0; iii<static_cast<int>(rModelPart.Nodes().size()); iii++)
{
ModelPart::NodeIterator itNode = it_begin+iii;
if( this->IsSlip(*itNode) )
{
if(mDomainSize == 3)
{
BoundedMatrix<double,3,3> rRot;
LocalRotationOperatorPure(rRot,*itNode);

array_1d<double,3>& rVelocity = itNode->FastGetSolutionStepValue(VELOCITY);
for(unsigned int i = 0; i < 3; i++) Vel[i] = rVelocity[i];
noalias(Tmp) = prod(rRot,Vel);
for(unsigned int i = 0; i < 3; i++) rVelocity[i] = Tmp[i];
}
else
{
BoundedMatrix<double,2,2> rRot;
LocalRotationOperatorPure(rRot,*itNode);

array_1d<double,3>& rVelocity = itNode->FastGetSolutionStepValue(VELOCITY);
for(unsigned int i = 0; i < 2; i++) Vel[i] = rVelocity[i];
noalias(Tmp) = prod(rRot,Vel);
for(unsigned int i = 0; i < 2; i++) rVelocity[i] = Tmp[i];
}
}
}
}

virtual void RecoverVelocities(ModelPart& rModelPart) const
{
TLocalVectorType Vel(mDomainSize);
TLocalVectorType Tmp(mDomainSize);

ModelPart::NodeIterator it_begin = rModelPart.NodesBegin();
#pragma omp parallel for firstprivate(Vel,Tmp)
for(int iii=0; iii<static_cast<int>(rModelPart.Nodes().size()); iii++)
{
ModelPart::NodeIterator itNode = it_begin+iii;
if( this->IsSlip(*itNode) )
{
if(mDomainSize == 3)
{
BoundedMatrix<double,3,3> rRot;
LocalRotationOperatorPure(rRot,*itNode);

array_1d<double,3>& rVelocity = itNode->FastGetSolutionStepValue(VELOCITY);
for(unsigned int i = 0; i < 3; i++) Vel[i] = rVelocity[i];
noalias(Tmp) = prod(trans(rRot),Vel);
for(unsigned int i = 0; i < 3; i++) rVelocity[i] = Tmp[i];
}
else
{
BoundedMatrix<double,2,2> rRot;
LocalRotationOperatorPure(rRot,*itNode);

array_1d<double,3>& rVelocity = itNode->FastGetSolutionStepValue(VELOCITY);
for(unsigned int i = 0; i < 2; i++) Vel[i] = rVelocity[i];
noalias(Tmp) = prod(trans(rRot),Vel);
for(unsigned int i = 0; i < 2; i++) rVelocity[i] = Tmp[i];
}
}
}
}




virtual std::string Info() const
{
std::stringstream buffer;
buffer << "CoordinateTransformationUtils";
return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "CoordinateTransformationUtils";
}

virtual void PrintData(std::ostream& rOStream) const {}



protected:




template<unsigned int TDim, unsigned int TBlockSize, unsigned int TSkip = 0>
void RotateAux(TLocalMatrixType& rLocalMatrix,
TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const
{
const unsigned int LocalSize = rLocalVector.size();

int rotations_needed = 0;
const unsigned int NumBlocks = LocalSize / TBlockSize;
DenseVector<bool> NeedRotation( NumBlocks, false);

std::vector< BoundedMatrix<double,TBlockSize,TBlockSize> > rRot(NumBlocks);
for(unsigned int j = 0; j < NumBlocks; ++j)
{
if( this->IsSlip(rGeometry[j]) )
{
NeedRotation[j] = true;
rotations_needed++;

if constexpr (TDim == 2) LocalRotationOperator2D<TBlockSize,TSkip>(rRot[j],rGeometry[j]);
else LocalRotationOperator3D<TBlockSize,TSkip>(rRot[j],rGeometry[j]);
}

}

if(rotations_needed > 0)
{
BoundedMatrix<double,TBlockSize,TBlockSize> mat_block, tmp;
array_1d<double,TBlockSize> aux, aux1;

for(unsigned int i=0; i<NumBlocks; i++)
{
if(NeedRotation[i] == true)
{
for(unsigned int j=0; j<NumBlocks; j++)
{
if(NeedRotation[j] == true)
{
ReadBlockMatrix<TBlockSize>(mat_block, rLocalMatrix, i*TBlockSize, j*TBlockSize);
noalias(tmp) = prod(mat_block,trans(rRot[j]));
noalias(mat_block) = prod(rRot[i],tmp);
WriteBlockMatrix<TBlockSize>(mat_block, rLocalMatrix, i*TBlockSize, j*TBlockSize);
}
else
{
ReadBlockMatrix<TBlockSize>(mat_block, rLocalMatrix, i*TBlockSize, j*TBlockSize);
noalias(tmp) = prod(rRot[i],mat_block);
WriteBlockMatrix<TBlockSize>(tmp, rLocalMatrix, i*TBlockSize, j*TBlockSize);
}
}

for(unsigned int k=0; k<TBlockSize; k++)
aux[k] = rLocalVector[i*TBlockSize+k];

noalias(aux1) = prod(rRot[i],aux);

for(unsigned int k=0; k<TBlockSize; k++)
rLocalVector[i*TBlockSize+k] = aux1[k];

}
else
{
for(unsigned int j=0; j<NumBlocks; j++)
{
if(NeedRotation[j] == true)
{
ReadBlockMatrix<TBlockSize>(mat_block, rLocalMatrix, i*TBlockSize, j*TBlockSize);
noalias(tmp) = prod(mat_block,trans(rRot[j]));
WriteBlockMatrix<TBlockSize>(tmp, rLocalMatrix, i*TBlockSize, j*TBlockSize);
}
}
}

}
}
}

template<unsigned int TDim>
void RotateAuxPure(TLocalMatrixType& rLocalMatrix,
TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const
{
const unsigned int LocalSize = rLocalVector.size();

int rotations_needed = 0;
const unsigned int NumBlocks = LocalSize / mBlockSize;
DenseVector<bool> NeedRotation( NumBlocks, false);

std::vector< BoundedMatrix<double,TDim,TDim> > rRot(NumBlocks);
for(unsigned int j = 0; j < NumBlocks; ++j)
{
if( this->IsSlip(rGeometry[j]) )
{
NeedRotation[j] = true;
rotations_needed++;

LocalRotationOperatorPure(rRot[j],rGeometry[j]);
}

}

if(rotations_needed > 0)
{
BoundedMatrix<double,TDim,TDim> mat_block, tmp;
array_1d<double,TDim> aux, aux1;

for(unsigned int i=0; i<NumBlocks; i++)
{
if(NeedRotation[i] == true)
{
for(unsigned int j=0; j<NumBlocks; j++)
{
if(NeedRotation[j] == true)
{
ReadBlockMatrix<TDim>(mat_block, rLocalMatrix, i*mBlockSize, j*mBlockSize);
noalias(tmp) = prod(mat_block,trans(rRot[j]));
noalias(mat_block) = prod(rRot[i],tmp);
WriteBlockMatrix<TDim>(mat_block, rLocalMatrix, i*mBlockSize, j*mBlockSize);
}
else
{
ReadBlockMatrix<TDim>(mat_block, rLocalMatrix, i*mBlockSize, j*mBlockSize);
noalias(tmp) = prod(rRot[i],mat_block);
WriteBlockMatrix<TDim>(tmp, rLocalMatrix, i*mBlockSize, j*mBlockSize);
}
}

for(unsigned int k=0; k<TDim; k++)
aux[k] = rLocalVector[i*mBlockSize+k];

noalias(aux1) = prod(rRot[i],aux);

for(unsigned int k=0; k<TDim; k++)
rLocalVector[i*mBlockSize+k] = aux1[k];

}
else
{
for(unsigned int j=0; j<NumBlocks; j++)
{
if(NeedRotation[j] == true)
{
ReadBlockMatrix<TDim>(mat_block, rLocalMatrix, i*mBlockSize, j*mBlockSize);
noalias(tmp) = prod(mat_block,trans(rRot[j]));
WriteBlockMatrix<TDim>(tmp, rLocalMatrix, i*mBlockSize, j*mBlockSize);
}
}
}

}
}
}

template<unsigned int TBlockSize, unsigned int TSkip = 0>
void LocalRotationOperator2D(
BoundedMatrix<double,TBlockSize,TBlockSize>& rRot,
GeometryType::PointType& rThisPoint) const
{
noalias(rRot) = IdentityMatrix(TBlockSize);

const array_1d<double,3>& rNormal = rThisPoint.FastGetSolutionStepValue(NORMAL);

double aux = rNormal[0]*rNormal[0] + rNormal[1]*rNormal[1];
aux = sqrt(aux);

rRot(TSkip  ,TSkip  ) = rNormal[0]/aux;
rRot(TSkip  ,TSkip+1) = rNormal[1]/aux;
rRot(TSkip+1,TSkip  ) = -rNormal[1]/aux;
rRot(TSkip+1,TSkip+1) = rNormal[0]/aux;
}

template<unsigned int TBlockSize, unsigned int TSkip = 0>
void LocalRotationOperator3D(
BoundedMatrix<double,TBlockSize,TBlockSize>& rRot,
GeometryType::PointType& rThisPoint) const
{
noalias(rRot) = IdentityMatrix(TBlockSize);

const array_1d<double,3>& rNormal = rThisPoint.FastGetSolutionStepValue(NORMAL);

double aux = rNormal[0]*rNormal[0] + rNormal[1]*rNormal[1] + rNormal[2]*rNormal[2];
aux = sqrt(aux);
rRot(TSkip,TSkip  ) = rNormal[0]/aux;
rRot(TSkip,TSkip+1) = rNormal[1]/aux;
rRot(TSkip,TSkip+2) = rNormal[2]/aux;

array_1d<double,3> rT1;
rT1(0) = 1.0;
rT1(1) = 0.0;
rT1(2) = 0.0;
double dot = rRot(TSkip,TSkip);

if ( fabs(dot) > 0.99 )
{
rT1(0) = 0.0;
rT1(1) = 1.0;
rT1(2) = 0.0;

dot = rRot(TSkip,TSkip+1); 
}

rT1[0] -= dot*rRot(TSkip,TSkip);
rT1[1] -= dot*rRot(TSkip,TSkip+1);
rT1[2] -= dot*rRot(TSkip,TSkip+2);
this->Normalize(rT1);
rRot(TSkip+1,TSkip  ) = rT1[0];
rRot(TSkip+1,TSkip+1) = rT1[1];
rRot(TSkip+1,TSkip+2) = rT1[2];

rRot(TSkip+2,TSkip  ) = rRot(TSkip,TSkip+1)*rT1[2] - rRot(TSkip,TSkip+2)*rT1[1];
rRot(TSkip+2,TSkip+1) = rRot(TSkip,TSkip+2)*rT1[0] - rRot(TSkip,TSkip  )*rT1[2];
rRot(TSkip+2,TSkip+2) = rRot(TSkip,TSkip  )*rT1[1] - rRot(TSkip,TSkip+1)*rT1[0];
}

bool IsSlip(const Node& rNode) const
{
return rNode.Is(mrFlag);
}


template< class TVectorType >
double Normalize(TVectorType& rThis) const
{
double Norm = 0;
for(typename TVectorType::iterator iComponent = rThis.begin(); iComponent < rThis.end(); ++iComponent)
Norm += (*iComponent)*(*iComponent);
Norm = sqrt(Norm);
for(typename TVectorType::iterator iComponent = rThis.begin(); iComponent < rThis.end(); ++iComponent)
*iComponent /= Norm;
return Norm;
}


unsigned int GetDomainSize() const
{
return mDomainSize;
}

unsigned int GetBlockSize() const
{
return mBlockSize;
}




private:


const unsigned int mDomainSize;


const unsigned int mBlockSize;

const Kratos::Flags& mrFlag;




template< class TVectorType >
double Dot(const TVectorType& rV1,
const TVectorType& rV2) const
{
double dot = 0.0;
for( typename TVectorType::const_iterator iV1 = rV1.begin(),iV2 = rV2.begin(); iV1 != rV1.end(); ++iV1, ++iV2)
{
dot += (*iV1) * (*iV2);
}
return dot;
}

inline double VectorNormDerivative(
const double ValueNorm,
const array_1d<double, 3>& rValue,
const array_1d<double, 3>& rValueDerivative) const
{
return inner_prod(rValue, rValueDerivative) / ValueNorm;
}

inline array_1d<double, 3> UnitVectorDerivative(
const double VectorNorm,
const double VectorNormDerivative,
const array_1d<double, 3>& rVector,
const array_1d<double, 3>& rVectorDerivative) const
{
return (rVectorDerivative * VectorNorm - rVector * VectorNormDerivative) /
std::pow(VectorNorm, 2);
}


template< unsigned int TBlockSize >
void ReadBlockMatrix( BoundedMatrix<double,TBlockSize, TBlockSize>& block, const Matrix& origin, const unsigned int Ibegin, const unsigned int Jbegin) const
{
for(unsigned int i=0; i<TBlockSize; i++)
{
for(unsigned int j=0; j<TBlockSize; j++)
{
block(i,j) = origin(Ibegin+i, Jbegin+j);
}
}
}

template< unsigned int TBlockSize >
void WriteBlockMatrix( const BoundedMatrix<double,TBlockSize, TBlockSize>& block, Matrix& destination, const unsigned int Ibegin, const unsigned int Jbegin) const
{
for(unsigned int i=0; i<TBlockSize; i++)
{
for(unsigned int j=0; j<TBlockSize; j++)
{
destination(Ibegin+i, Jbegin+j) = block(i,j);
}
}
}




CoordinateTransformationUtils& operator=(CoordinateTransformationUtils const& rOther) {}

CoordinateTransformationUtils(CoordinateTransformationUtils const& rOther) {}

};




template<class TLocalMatrixType, class TLocalVectorType, class TValueType>
inline std::istream& operator >>(std::istream& rIStream,
CoordinateTransformationUtils<TLocalMatrixType, TLocalVectorType,
TValueType>& rThis) {
return rIStream;
}

template<class TLocalMatrixType, class TLocalVectorType, class TValueType>
inline std::ostream& operator <<(std::ostream& rOStream,
const CoordinateTransformationUtils<TLocalMatrixType, TLocalVectorType,
TValueType>& rThis) {
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}



}

#endif 
