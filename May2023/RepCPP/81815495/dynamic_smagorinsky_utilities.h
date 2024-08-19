
#include <vector>
#include <map>


#include "includes/define.h"
#include "includes/model_part.h"
#include "includes/node.h"
#include "includes/element.h"
#include "utilities/openmp_utils.h"
#include "utilities/parallel_utilities.h"
#include "utilities/geometry_utilities.h"

#include "includes/cfd_variables.h"
#include "fluid_dynamics_application_variables.h"
#include "includes/global_pointer_variables.h"

#ifndef KRATOS_DYNAMIC_SMAGORINSKY_UTILITIES_H_INCLUDED
#define	KRATOS_DYNAMIC_SMAGORINSKY_UTILITIES_H_INCLUDED

namespace Kratos
{



class DynamicSmagorinskyUtils
{
public:



DynamicSmagorinskyUtils(ModelPart& rModelPart, unsigned int DomainSize):
mrModelPart(rModelPart),
mDomainSize(DomainSize),
mCoarseMesh(),
mPatchIndices()
{}

~DynamicSmagorinskyUtils() {}



void StoreCoarseMesh()
{
mCoarseMesh.clear();

for( ModelPart::ElementsContainerType::ptr_iterator itpElem = mrModelPart.Elements().ptr_begin();
itpElem != mrModelPart.Elements().ptr_end(); ++itpElem)
{
mCoarseMesh.push_back(*itpElem);
}

const int NumThreads = ParallelUtilities::GetNumThreads();
OpenMPUtils::PartitionVector ElementPartition;
OpenMPUtils::DivideInPartitions(mCoarseMesh.size(),NumThreads,ElementPartition);

std::vector< std::vector<int> > LocalIndices(NumThreads);

#pragma omp parallel
{
int k = OpenMPUtils::ThisThread();
ModelPart::ElementsContainerType::iterator ElemBegin = mCoarseMesh.begin() + ElementPartition[k];
ModelPart::ElementsContainerType::iterator ElemEnd = mCoarseMesh.begin() + ElementPartition[k+1];

for( ModelPart::ElementIterator itElem = ElemBegin; itElem != ElemEnd; ++itElem)
{
this->AddNewIndex(LocalIndices[k],itElem->GetValue(PATCH_INDEX));
}
}

unsigned int Counter = 0;
std::pair<int, unsigned int> NewVal;
std::pair< std::map<int, unsigned int>::iterator, bool > Result;
for( std::vector< std::vector<int> >::iterator itList = LocalIndices.begin(); itList != LocalIndices.end(); ++itList )
{
for( std::vector<int>::iterator itIndex = itList->begin(); itIndex != itList->end(); ++itIndex)
{
NewVal.first = *itIndex;
NewVal.second = Counter;
Result = mPatchIndices.insert(NewVal);
if (Result.second)
++Counter;
}
}
}

void CalculateC()
{
this->SetCoarseVel();

const int NumThreads = ParallelUtilities::GetNumThreads();
OpenMPUtils::PartitionVector CoarseElementPartition,FineElementPartition;
OpenMPUtils::DivideInPartitions(mCoarseMesh.size(),NumThreads,CoarseElementPartition);
OpenMPUtils::DivideInPartitions(mrModelPart.Elements().size(),NumThreads,FineElementPartition);

unsigned int PatchNumber = mPatchIndices.size();

std::vector< std::vector<double> > GlobalPatchNum(NumThreads); 
std::vector< std::vector<double> > GlobalPatchDen(NumThreads); 

const double EnergyTol = 0.005;
double TotalDissipation = 0;

#pragma omp parallel reduction(+:TotalDissipation)
{
int k = OpenMPUtils::ThisThread();

ModelPart::ElementsContainerType::iterator CoarseElemBegin = mCoarseMesh.begin() + CoarseElementPartition[k];
ModelPart::ElementsContainerType::iterator CoarseElemEnd = mCoarseMesh.begin() + CoarseElementPartition[k+1];

ModelPart::ElementsContainerType::iterator FineElemBegin = mrModelPart.ElementsBegin() + FineElementPartition[k];
ModelPart::ElementsContainerType::iterator FineElemEnd = mrModelPart.ElementsBegin() + FineElementPartition[k+1];

Vector LocalValues, LocalCoarseVel;
Matrix LocalMassMatrix;
ProcessInfo& rProcessInfo = mrModelPart.GetProcessInfo();

double Residual,Model;
unsigned int PatchPosition;

std::vector<double>& rPatchNum = GlobalPatchNum[k];
std::vector<double>& rPatchDen = GlobalPatchDen[k];
rPatchNum.resize(PatchNumber,0.0);
rPatchDen.resize(PatchNumber,0.0);

if (mDomainSize == 2)
{
LocalValues.resize(9);
LocalCoarseVel.resize(9);
LocalMassMatrix.resize(9,9,false);
array_1d<double,3> N;
BoundedMatrix<double,3,2> DN_DX;
BoundedMatrix<double,2,2> dv_dx;

for( ModelPart::ElementsContainerType::iterator itElem = CoarseElemBegin; itElem != CoarseElemEnd; ++itElem)
{
PatchPosition = mPatchIndices[ itElem->GetValue(PATCH_INDEX) ];
this->GermanoTerms2D(*itElem,N,DN_DX,dv_dx,LocalValues,LocalCoarseVel,LocalMassMatrix,rProcessInfo,Residual,Model);

rPatchNum[PatchPosition] += Residual;
rPatchDen[PatchPosition] += Model;
TotalDissipation += Residual;
}

for( ModelPart::ElementsContainerType::iterator itElem = FineElemBegin; itElem != FineElemEnd; ++itElem)
{
itElem->GetValue(C_SMAGORINSKY) = 0.0;

PatchPosition = mPatchIndices[ itElem->GetValue(PATCH_INDEX) ];
this->GermanoTerms2D(*itElem,N,DN_DX,dv_dx,LocalValues,LocalCoarseVel,LocalMassMatrix,rProcessInfo,Residual,Model);

rPatchNum[PatchPosition] -= Residual;
rPatchDen[PatchPosition] -= Model;
}
}
else 
{
LocalValues.resize(16);
LocalCoarseVel.resize(16);
LocalMassMatrix.resize(16,16,false);
array_1d<double,4> N;
BoundedMatrix<double,4,3> DN_DX;
BoundedMatrix<double,3,3> dv_dx;

for( ModelPart::ElementsContainerType::iterator itElem = CoarseElemBegin; itElem != CoarseElemEnd; ++itElem)
{
PatchPosition = mPatchIndices[ itElem->GetValue(PATCH_INDEX) ];
this->GermanoTerms3D(*itElem,N,DN_DX,dv_dx,LocalValues,LocalCoarseVel,LocalMassMatrix,rProcessInfo,Residual,Model);

rPatchNum[PatchPosition] += Residual;
rPatchDen[PatchPosition] += Model;
TotalDissipation += Residual;
}

for( ModelPart::ElementsContainerType::iterator itElem = FineElemBegin; itElem != FineElemEnd; ++itElem)
{
itElem->GetValue(C_SMAGORINSKY) = 0.0;

PatchPosition = mPatchIndices[ itElem->GetValue(PATCH_INDEX) ];
this->GermanoTerms3D(*itElem,N,DN_DX,dv_dx,LocalValues,LocalCoarseVel,LocalMassMatrix,rProcessInfo,Residual,Model);

rPatchNum[PatchPosition] -= Residual;
rPatchDen[PatchPosition] -= Model;
}
}
}

for( std::vector< std::vector<double> >::iterator itNum = GlobalPatchNum.begin()+1, itDen = GlobalPatchDen.begin()+1;
itNum != GlobalPatchNum.end(); ++itNum, ++itDen)
{
for( std::vector<double>::iterator TotalNum = GlobalPatchNum[0].begin(), LocalNum = itNum->begin(),
TotalDen = GlobalPatchDen[0].begin(), LocalDen = itDen->begin();
TotalNum != GlobalPatchNum[0].end(); ++TotalNum,++LocalNum,++TotalDen,++LocalDen)
{
*TotalNum += *LocalNum;
*TotalDen += *LocalDen;
}
}

std::vector<double> PatchC(PatchNumber);
double NumTol = EnergyTol * fabs(TotalDissipation);
for( std::vector<double>::iterator itNum = GlobalPatchNum[0].begin(), itDen = GlobalPatchDen[0].begin(), itC = PatchC.begin();
itC != PatchC.end(); ++itNum, ++itDen, ++itC)
{
if ( (fabs(*itNum) < NumTol) )
*itC = 0.0;
else
*itC = sqrt( 0.5 * fabs( *itNum / *itDen ) );
}

#pragma omp parallel
{
int k = OpenMPUtils::ThisThread();
ModelPart::ElementsContainerType::iterator ElemBegin = mrModelPart.ElementsBegin() + FineElementPartition[k];
ModelPart::ElementsContainerType::iterator ElemEnd = mrModelPart.ElementsBegin() + FineElementPartition[k+1];

unsigned int PatchPosition;

for( ModelPart::ElementIterator itElem = ElemBegin; itElem != ElemEnd; ++itElem)
{
PatchPosition = mPatchIndices[ itElem->GetValue(PATCH_INDEX) ];
itElem->GetValue(C_SMAGORINSKY) = PatchC[PatchPosition];
}
}
}


void CorrectFlagValues(Variable<double>& rThisVariable = FLAG_VARIABLE)
{
const int NumThreads = ParallelUtilities::GetNumThreads();
OpenMPUtils::PartitionVector NodePartition;
OpenMPUtils::DivideInPartitions(mrModelPart.NumberOfNodes(),NumThreads,NodePartition);

#pragma omp parallel
{
int k = OpenMPUtils::ThisThread();
ModelPart::NodeIterator NodesBegin = mrModelPart.NodesBegin() + NodePartition[k];
ModelPart::NodeIterator NodesEnd = mrModelPart.NodesBegin() + NodePartition[k+1];

double Value0, Value1;

for( ModelPart::NodeIterator itNode = NodesBegin; itNode != NodesEnd; ++itNode)
{
if( itNode->GetValue(FATHER_NODES).size() == 2 ) 
{
Value0 = itNode->GetValue(FATHER_NODES)[0].FastGetSolutionStepValue(rThisVariable);
Value1 = itNode->GetValue(FATHER_NODES)[1].FastGetSolutionStepValue(rThisVariable);

if( Value0 != Value1 ) 
{
if ( Value0 == 0.0 || Value1 == 0.0 )
{
itNode->FastGetSolutionStepValue(rThisVariable) = 0.0;
}

else if( Value0 == 3.0 )
{
itNode->FastGetSolutionStepValue(rThisVariable) = Value0;
}
else if( Value1 == 3.0 )
{
itNode->FastGetSolutionStepValue(rThisVariable) = Value1;
}
else 
{
itNode->FastGetSolutionStepValue(rThisVariable) = Value0;
}
}
}
}
}
}


private:


ModelPart& mrModelPart;
unsigned int mDomainSize;
ModelPart::ElementsContainerType mCoarseMesh;
std::map<int, unsigned int> mPatchIndices;



void SetCoarseVel()
{

for( ModelPart::NodeIterator itNode = mrModelPart.NodesBegin(); itNode != mrModelPart.NodesEnd(); ++itNode)
{
if( itNode->GetValue(FATHER_NODES).size() == 2 )
{
Node& rParent1 = itNode->GetValue(FATHER_NODES)[0];
Node& rParent2 = itNode->GetValue(FATHER_NODES)[1];

itNode->GetValue(COARSE_VELOCITY) = 0.5 * ( rParent1.FastGetSolutionStepValue(VELOCITY) + rParent2.FastGetSolutionStepValue(VELOCITY) );
}
else
{
itNode->GetValue(COARSE_VELOCITY) = itNode->FastGetSolutionStepValue(VELOCITY);
}
}
}

void GermanoTerms2D(Element& rElem,
array_1d<double,3>& rShapeFunc,
BoundedMatrix<double,3,2>& rShapeDeriv,
BoundedMatrix<double,2,2>& rGradient,
Vector& rNodalResidualContainer,
Vector& rNodalVelocityContainer,
Matrix& rMassMatrix,
ProcessInfo& rProcessInfo,
double& rResidual,
double& rModel)
{
const double Dim = 2;
const double NumNodes = 3;

double Area;
double Density = 0.0;
rGradient = ZeroMatrix(Dim,Dim);

rResidual = 0.0;
rModel = 0.0;

this->CalculateResidual(rElem,rMassMatrix,rNodalVelocityContainer,rNodalResidualContainer,rProcessInfo); 
this->GetCoarseVelocity2D(rElem,rNodalVelocityContainer);

for( Vector::iterator itRHS = rNodalResidualContainer.begin(), itVel = rNodalVelocityContainer.begin(); itRHS != rNodalResidualContainer.end(); ++itRHS, ++itVel)
rResidual += (*itVel) * (*itRHS);

GeometryUtils::CalculateGeometryData( rElem.GetGeometry(), rShapeDeriv, rShapeFunc, Area);

for (unsigned int j = 0; j < NumNodes; ++j) 
{
Density += rShapeFunc[j] * rElem.GetGeometry()[j].FastGetSolutionStepValue(DENSITY);
const array_1d< double,3 >& rNodeVel = rElem.GetGeometry()[j].FastGetSolutionStepValue(VELOCITY); 

for (unsigned int i = 0; i < NumNodes; ++i) 
{
const array_1d< double,3 >& rNodeTest = rElem.GetGeometry()[i].GetValue(COARSE_VELOCITY); 

for (unsigned int k = 0; k < Dim; ++k) 
rModel += rNodeTest[k] * rShapeDeriv(i,k) * rShapeDeriv(j,k) * rNodeVel[k];
}

for (unsigned int m = 0; m < Dim; ++m) 
{
for (unsigned int n = 0; n < m; ++n) 
rGradient(m,n) += 0.5 * (rShapeDeriv(j,n) * rNodeVel[m] + rShapeDeriv(j,m) * rNodeVel[n]); 
rGradient(m,m) += rShapeDeriv(j,m) * rNodeVel[m]; 
}
}

rModel *= Area; 

double SqNorm = 0.0;
for (unsigned int i = 0; i < Dim; ++i)
{
for (unsigned int j = 0; j < i; ++j)
SqNorm += 2.0 * rGradient(i,j) * rGradient(i,j); 
SqNorm += rGradient(i,i) * rGradient(i,i); 
}

const double sqH = 2*Area;
rModel *= Density * sqH * sqrt(SqNorm);
}

void GermanoTerms3D(Element& rElem,
array_1d<double,4>& rShapeFunc,
BoundedMatrix<double,4,3>& rShapeDeriv,
BoundedMatrix<double,3,3>& rGradient,
Vector& rNodalResidualContainer,
Vector& rNodalVelocityContainer,
Matrix& rMassMatrix,
ProcessInfo& rProcessInfo,
double& rResidual,
double& rModel)
{
const double Dim = 3;
const double NumNodes = 4;

double Volume;
double Density = 0.0;
rGradient = ZeroMatrix(Dim,Dim);

rResidual = 0.0;
rModel = 0.0;

this->CalculateResidual(rElem,rMassMatrix,rNodalVelocityContainer,rNodalResidualContainer,rProcessInfo); 
this->GetCoarseVelocity3D(rElem,rNodalVelocityContainer);

for( Vector::iterator itRHS = rNodalResidualContainer.begin(), itVel = rNodalVelocityContainer.begin(); itRHS != rNodalResidualContainer.end(); ++itRHS, ++itVel)
rResidual += (*itVel) * (*itRHS);

GeometryUtils::CalculateGeometryData( rElem.GetGeometry(), rShapeDeriv, rShapeFunc, Volume);

for (unsigned int j = 0; j < NumNodes; ++j) 
{
Density += rShapeFunc[j] * rElem.GetGeometry()[j].FastGetSolutionStepValue(DENSITY);
const array_1d< double,3 >& rNodeVel = rElem.GetGeometry()[j].FastGetSolutionStepValue(VELOCITY); 

for (unsigned int i = 0; i < NumNodes; ++i) 
{
const array_1d< double,3 >& rNodeTest = rElem.GetGeometry()[i].GetValue(COARSE_VELOCITY); 

for (unsigned int k = 0; k < Dim; ++k) 
rModel += rNodeTest[k] * rShapeDeriv(i,k) * rShapeDeriv(j,k) * rNodeVel[k];
}

for (unsigned int m = 0; m < Dim; ++m) 
{
for (unsigned int n = 0; n < m; ++n) 
rGradient(m,n) += 0.5 * (rShapeDeriv(j,n) * rNodeVel[m] + rShapeDeriv(j,m) * rNodeVel[n]); 
rGradient(m,m) += rShapeDeriv(j,m) * rNodeVel[m]; 
}
}

rModel *= Volume; 

double SqNorm = 0.0;
for (unsigned int i = 0; i < Dim; ++i)
{
for (unsigned int j = 0; j < i; ++j)
SqNorm += 2.0 * rGradient(i,j) * rGradient(i,j); 
SqNorm += rGradient(i,i) * rGradient(i,i); 
}

const double cubeH = 6*Volume;
rModel *= Density * pow(cubeH, 2.0/3.0) * sqrt(2.0 * SqNorm);
}

void GetCoarseVelocity2D(Element& rElement,
Vector& rVar)
{
unsigned int LocalIndex = 0;
const Element::GeometryType& rGeom = rElement.GetGeometry();

for (unsigned int itNode = 0; itNode < 3; ++itNode)
{
const array_1d< double,3>& rCoarseVel = rGeom[itNode].GetValue(COARSE_VELOCITY);
rVar[LocalIndex++] = rCoarseVel[0];
rVar[LocalIndex++] = rCoarseVel[1];
rVar[LocalIndex++] = 0.0; 
}
}

void GetCoarseVelocity3D(Element& rElement,
Vector& rVar)
{
unsigned int LocalIndex = 0;
const Element::GeometryType& rGeom = rElement.GetGeometry();

for (unsigned int itNode = 0; itNode < 4; ++itNode)
{
const array_1d< double,3>& rCoarseVel = rGeom[itNode].GetValue(COARSE_VELOCITY);
rVar[LocalIndex++] = rCoarseVel[0];
rVar[LocalIndex++] = rCoarseVel[1];
rVar[LocalIndex++] = rCoarseVel[2];
rVar[LocalIndex++] = 0.0; 
}
}

void CalculateResidual(Element& rElement,
Matrix& rMassMatrix, 
Vector& rAuxVector,
Vector& rResidual,
const ProcessInfo& rCurrentProcessInfo)
{
const auto& r_const_elem_ref = rElement;
rElement.InitializeNonLinearIteration(rCurrentProcessInfo);

rElement.CalculateRightHandSide(rResidual,rCurrentProcessInfo);

rElement.CalculateMassMatrix(rMassMatrix,rCurrentProcessInfo);
r_const_elem_ref.GetSecondDerivativesVector(rAuxVector,0);

noalias(rResidual) -= prod(rMassMatrix,rAuxVector);

rElement.CalculateLocalVelocityContribution(rMassMatrix,rResidual,rCurrentProcessInfo); 
}

void AddNewIndex( std::vector<int>& rIndices,
int ThisIndex )
{
bool IsNew = true;
for( std::vector<int>::iterator itIndex = rIndices.begin(); itIndex != rIndices.end(); ++itIndex)
{
if( ThisIndex == *itIndex)
{
IsNew = false;
break;
}
}

if (IsNew)
rIndices.push_back(ThisIndex);
}


};



} 

#endif	
