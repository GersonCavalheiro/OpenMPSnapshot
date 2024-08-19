

#ifndef KRATOS_COMPRESSIBLE_ELEMENT_ROTATION_UTILITY
#define KRATOS_COMPRESSIBLE_ELEMENT_ROTATION_UTILITY



#include "includes/define.h"
#include "includes/node.h"
#include "containers/variable.h"
#include "geometries/geometry.h"
#include "utilities/coordinate_transformation_utilities.h"

namespace Kratos {







template<class TLocalMatrixType, class TLocalVectorType>
class CompressibleElementRotationUtility: public CoordinateTransformationUtils<TLocalMatrixType,TLocalVectorType,double> {
public:

KRATOS_CLASS_POINTER_DEFINITION(CompressibleElementRotationUtility);

typedef Node NodeType;

typedef Geometry< Node > GeometryType;



CompressibleElementRotationUtility(
const unsigned int DomainSize,
const Kratos::Flags& rFlag = SLIP):
CoordinateTransformationUtils<TLocalMatrixType,TLocalVectorType,double>(DomainSize,DomainSize+2,rFlag)
{}

~CompressibleElementRotationUtility() override {}




void Rotate(
TLocalMatrixType& rLocalMatrix,
TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const override
{
if (this->GetDomainSize() == 2) this->template RotateAux<2,4,1>(rLocalMatrix,rLocalVector,rGeometry);
else if (this->GetDomainSize() == 3) this->template RotateAux<3,5,1>(rLocalMatrix,rLocalVector,rGeometry);
}

void Rotate(
TLocalVectorType& rLocalVector,
GeometryType& rGeometry) const override
{
TLocalMatrixType dummy = ZeroMatrix(rLocalVector.size(),rLocalVector.size());
if (this->GetDomainSize() == 2) this->template RotateAux<2,4,1>(dummy,rLocalVector,rGeometry);
else if (this->GetDomainSize() == 3) this->template RotateAux<3,5,1>(dummy,rLocalVector,rGeometry);
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
unsigned int j = itNode * this->GetBlockSize() + 1; 

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

rLocalVector(j) = 0.0;
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
unsigned int j = itNode * this->GetBlockSize() + 1; 
rLocalVector[j] = 0.0;
}
}
}
}

void RotateVelocities(ModelPart& rModelPart) const override
{
TLocalVectorType momentum(this->GetDomainSize());
TLocalVectorType Tmp(this->GetDomainSize());

ModelPart::NodeIterator it_begin = rModelPart.NodesBegin();
#pragma omp parallel for firstprivate(momentum,Tmp)
for(int iii=0; iii<static_cast<int>(rModelPart.Nodes().size()); iii++)
{
ModelPart::NodeIterator itNode = it_begin+iii;
if( this->IsSlip(*itNode) )
{
if(this->GetDomainSize() == 3)
{
BoundedMatrix<double,3,3> rRot;
this->LocalRotationOperatorPure(rRot,*itNode);

array_1d<double,3>& rMomentum = itNode->FastGetSolutionStepValue(MOMENTUM);
for(unsigned int i = 0; i < 3; i++) momentum[i] = rMomentum[i];
noalias(Tmp) = prod(rRot,momentum);
for(unsigned int i = 0; i < 3; i++) rMomentum[i] = Tmp[i];
}
else
{
BoundedMatrix<double,2,2> rRot;
this->LocalRotationOperatorPure(rRot,*itNode);

array_1d<double,3>& rMomentum = itNode->FastGetSolutionStepValue(MOMENTUM);
for(unsigned int i = 0; i < 2; i++) momentum[i] = rMomentum[i];
noalias(Tmp) = prod(rRot,momentum);
for(unsigned int i = 0; i < 2; i++) rMomentum[i] = Tmp[i];
}
}
}
}

void RecoverVelocities(ModelPart& rModelPart) const override
{
TLocalVectorType momentum(this->GetDomainSize());
TLocalVectorType Tmp(this->GetDomainSize());

ModelPart::NodeIterator it_begin = rModelPart.NodesBegin();
#pragma omp parallel for firstprivate(momentum,Tmp)
for(int iii=0; iii<static_cast<int>(rModelPart.Nodes().size()); iii++)
{
ModelPart::NodeIterator itNode = it_begin+iii;
if( this->IsSlip(*itNode) )
{
if(this->GetDomainSize() == 3)
{
BoundedMatrix<double,3,3> rRot;
this->LocalRotationOperatorPure(rRot,*itNode);

array_1d<double,3>& rMomentum = itNode->FastGetSolutionStepValue(MOMENTUM);
for(unsigned int i = 0; i < 3; i++) momentum[i] = rMomentum[i];
noalias(Tmp) = prod(trans(rRot),momentum);
for(unsigned int i = 0; i < 3; i++) rMomentum[i] = Tmp[i];
}
else
{
BoundedMatrix<double,2,2> rRot;
this->LocalRotationOperatorPure(rRot,*itNode);

array_1d<double,3>& rMomentum = itNode->FastGetSolutionStepValue(MOMENTUM);
for(unsigned int i = 0; i < 2; i++) momentum[i] = rMomentum[i];
noalias(Tmp) = prod(trans(rRot),momentum);
for(unsigned int i = 0; i < 2; i++) rMomentum[i] = Tmp[i];
}
}
}
}




std::string Info() const override
{
std::stringstream buffer;
buffer << "CompressibleElementRotationUtility";
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "CompressibleElementRotationUtility";
}

void PrintData(std::ostream& rOStream) const override {}



protected:








private:







CompressibleElementRotationUtility& operator=(CompressibleElementRotationUtility const& rOther) {}

CompressibleElementRotationUtility(CompressibleElementRotationUtility const& rOther) {}

};




template<class TLocalMatrixType, class TLocalVectorType>
inline std::istream& operator >>(std::istream& rIStream,
CompressibleElementRotationUtility<TLocalMatrixType, TLocalVectorType>& rThis) {
return rIStream;
}

template<class TLocalMatrixType, class TLocalVectorType>
inline std::ostream& operator <<(std::ostream& rOStream,
const CompressibleElementRotationUtility<TLocalMatrixType, TLocalVectorType>& rThis) {
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}



}

#endif 
