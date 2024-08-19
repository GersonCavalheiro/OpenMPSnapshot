

#ifndef KRATOS_MPM_BOUNDARY_ROTATION_UTILITY
#define KRATOS_MPM_BOUNDARY_ROTATION_UTILITY



#include "includes/define.h"
#include "includes/node.h"
#include "containers/variable.h"
#include "geometries/geometry.h"
#include "utilities/coordinate_transformation_utilities.h"

namespace Kratos {








template<class TLocalMatrixType, class TLocalVectorType>
class MPMBoundaryRotationUtility: public CoordinateTransformationUtils<TLocalMatrixType,TLocalVectorType,double> {
public:

KRATOS_CLASS_POINTER_DEFINITION(MPMBoundaryRotationUtility);

using CoordinateTransformationUtils<TLocalMatrixType,TLocalVectorType,double>::Rotate;

typedef Node NodeType;

typedef Geometry< Node > GeometryType;



MPMBoundaryRotationUtility(
const unsigned int DomainSize,
const unsigned int BlockSize,
const Variable<double>& rVariable):
CoordinateTransformationUtils<TLocalMatrixType,TLocalVectorType,double>(DomainSize,BlockSize,SLIP), mrFlagVariable(rVariable)
{}

~MPMBoundaryRotationUtility() override {}

MPMBoundaryRotationUtility& operator=(MPMBoundaryRotationUtility const& rOther) {}




void Rotate(
TLocalMatrixType& rLocalMatrix,
TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const override
{
if (this->GetBlockSize() == this->GetDomainSize()) 
{
if (this->GetDomainSize() == 2) this->template RotateAuxPure<2>(rLocalMatrix,rLocalVector,rGeometry);
else if (this->GetDomainSize() == 3) this->template RotateAuxPure<3>(rLocalMatrix,rLocalVector,rGeometry);
}
else 
{
if (this->GetDomainSize() == 2) this->template RotateAux<2,3>(rLocalMatrix,rLocalVector,rGeometry);
else if (this->GetDomainSize() == 3) this->template RotateAux<3,4>(rLocalMatrix,rLocalVector,rGeometry);
}

}

void RotateRHS(
TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const
{
this->Rotate(rLocalVector,rGeometry);
}


void ApplySlipCondition(TLocalMatrixType& rLocalMatrix,
TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const override
{
const unsigned int LocalSize = rLocalVector.size();

if (LocalSize > 0)
{
for(unsigned int itNode = 0; itNode < rGeometry.PointsNumber(); ++itNode)
{
if(this->IsSlip(rGeometry[itNode]) )
{
unsigned int j = itNode * this->GetBlockSize();

const array_1d<double,3> & displacement = rGeometry[itNode].FastGetSolutionStepValue(DISPLACEMENT);

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

rLocalVector[j] = inner_prod(rN,displacement);
rLocalMatrix(j,j) = 1.0;
}
}
}
}

void ApplySlipCondition(TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const override
{
if (rLocalVector.size() > 0)
{
for(unsigned int itNode = 0; itNode < rGeometry.PointsNumber(); ++itNode)
{
if( this->IsSlip(rGeometry[itNode]) )
{
unsigned int j = itNode * this->GetBlockSize(); 

const array_1d<double,3> & displacement = rGeometry[itNode].FastGetSolutionStepValue(DISPLACEMENT);
array_1d<double,3> rN = rGeometry[itNode].FastGetSolutionStepValue(NORMAL);
this->Normalize(rN);

rLocalVector[j] = inner_prod(rN,displacement);

}
}
}
}

void ElementApplySlipCondition(TLocalMatrixType& rLocalMatrix,
TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const
{
if (!this->IsPenalty(rGeometry))
{
this->ApplySlipCondition(rLocalMatrix, rLocalVector, rGeometry);
}
}

void ElementApplySlipCondition(TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const
{
if (!this->IsPenalty(rGeometry))
{
this->ApplySlipCondition(rLocalVector, rGeometry);
}
}

void ConditionApplySlipCondition(TLocalMatrixType& rLocalMatrix,
TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const
{
if (!this->IsPenalty(rGeometry))
{
this->ApplySlipCondition(rLocalMatrix, rLocalVector, rGeometry);
}
else
{
const unsigned int LocalSize = rLocalVector.size();

if (LocalSize > 0)
{
const unsigned int block_size = this->GetBlockSize();
TLocalMatrixType temp_matrix = ZeroMatrix(rLocalMatrix.size1(),rLocalMatrix.size2());
for(unsigned int itNode = 0; itNode < rGeometry.PointsNumber(); ++itNode)
{
if(this->IsSlip(rGeometry[itNode]) )
{
unsigned int j = itNode * block_size;

for (unsigned int i = j; i < rLocalMatrix.size1(); i+= block_size)
{
temp_matrix(i,j) = rLocalMatrix(i,j);
temp_matrix(j,i) = rLocalMatrix(j,i);
}

for(unsigned int i = j; i < (j + block_size); ++i)
{
if (i!=j) rLocalVector[i] = 0.0;
}
}
}
rLocalMatrix = temp_matrix;
}
}
}

void ConditionApplySlipCondition(TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const
{
if (!this->IsPenalty(rGeometry))
{
this->ApplySlipCondition(rLocalVector, rGeometry);
}
else
{
if (rLocalVector.size() > 0)
{
const unsigned int block_size = this->GetBlockSize();
for(unsigned int itNode = 0; itNode < rGeometry.PointsNumber(); ++itNode)
{
if( this->IsSlip(rGeometry[itNode]) )
{
unsigned int j = itNode * block_size;

for(unsigned int i = j; i < (j + block_size); ++i)
{
if (i!=j) rLocalVector[i] = 0.0;
}
}
}
}
}

}

bool IsPenalty(GeometryType& rGeometry) const
{
bool is_penalty = false;
for(unsigned int itNode = 0; itNode < rGeometry.PointsNumber(); ++itNode)
{
if(this->IsSlip(rGeometry[itNode]) )
{
const double identifier = rGeometry[itNode].FastGetSolutionStepValue(mrFlagVariable);
const double tolerance  = 1.e-6;
if (identifier > 1.00 + tolerance)
{
is_penalty = true;
break;
}
}
}

return is_penalty;
}

virtual	void RotateDisplacements(ModelPart& rModelPart) const
{
this->RotateVelocities(rModelPart);
}

void RotateVelocities(ModelPart& rModelPart) const override
{
TLocalVectorType displacement(this->GetDomainSize());
TLocalVectorType Tmp(this->GetDomainSize());

ModelPart::NodeIterator it_begin = rModelPart.NodesBegin();
#pragma omp parallel for firstprivate(displacement,Tmp)
for(int iii=0; iii<static_cast<int>(rModelPart.Nodes().size()); iii++)
{
ModelPart::NodeIterator itNode = it_begin+iii;
if( this->IsSlip(*itNode) )
{
if(this->GetDomainSize() == 3)
{
BoundedMatrix<double,3,3> rRot;
this->LocalRotationOperatorPure(rRot,*itNode);

array_1d<double,3>& rDisplacement = itNode->FastGetSolutionStepValue(DISPLACEMENT);
for(unsigned int i = 0; i < 3; i++) displacement[i] = rDisplacement[i];
noalias(Tmp) = prod(rRot,displacement);
for(unsigned int i = 0; i < 3; i++) rDisplacement[i] = Tmp[i];
}
else
{
BoundedMatrix<double,2,2> rRot;
this->LocalRotationOperatorPure(rRot,*itNode);

array_1d<double,3>& rDisplacement = itNode->FastGetSolutionStepValue(DISPLACEMENT);
for(unsigned int i = 0; i < 2; i++) displacement[i] = rDisplacement[i];
noalias(Tmp) = prod(rRot,displacement);
for(unsigned int i = 0; i < 2; i++) rDisplacement[i] = Tmp[i];
}
}
}
}

virtual void RecoverDisplacements(ModelPart& rModelPart) const
{
this->RecoverVelocities(rModelPart);
}

void RecoverVelocities(ModelPart& rModelPart) const override
{
TLocalVectorType displacement(this->GetDomainSize());
TLocalVectorType Tmp(this->GetDomainSize());

ModelPart::NodeIterator it_begin = rModelPart.NodesBegin();
#pragma omp parallel for firstprivate(displacement,Tmp)
for(int iii=0; iii<static_cast<int>(rModelPart.Nodes().size()); iii++)
{
ModelPart::NodeIterator itNode = it_begin+iii;
if( this->IsSlip(*itNode) )
{
if(this->GetDomainSize() == 3)
{
BoundedMatrix<double,3,3> rRot;
this->LocalRotationOperatorPure(rRot,*itNode);

array_1d<double,3>& rDisplacement = itNode->FastGetSolutionStepValue(DISPLACEMENT);
for(unsigned int i = 0; i < 3; i++) displacement[i] = rDisplacement[i];
noalias(Tmp) = prod(trans(rRot),displacement);
for(unsigned int i = 0; i < 3; i++) rDisplacement[i] = Tmp[i];
}
else
{
BoundedMatrix<double,2,2> rRot;
this->LocalRotationOperatorPure(rRot,*itNode);

array_1d<double,3>& rDisplacement = itNode->FastGetSolutionStepValue(DISPLACEMENT);
for(unsigned int i = 0; i < 2; i++) displacement[i] = rDisplacement[i];
noalias(Tmp) = prod(trans(rRot),displacement);
for(unsigned int i = 0; i < 2; i++) rDisplacement[i] = Tmp[i];
}
}
}
}




std::string Info() const override
{
std::stringstream buffer;
buffer << "MPMBoundaryRotationUtility";
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "MPMBoundaryRotationUtility";
}

void PrintData(std::ostream& rOStream) const override {}



protected:








private:

const Variable<double>& mrFlagVariable;







};




template<class TLocalMatrixType, class TLocalVectorType>
inline std::istream& operator >>(std::istream& rIStream,
MPMBoundaryRotationUtility<TLocalMatrixType, TLocalVectorType>& rThis) {
return rIStream;
}

template<class TLocalMatrixType, class TLocalVectorType>
inline std::ostream& operator <<(std::ostream& rOStream,
const MPMBoundaryRotationUtility<TLocalMatrixType, TLocalVectorType>& rThis) {
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}



}

#endif 
