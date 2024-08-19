
#pragma once

#include <string>
#include <iostream>




#include "includes/define.h"
#include "includes/condition.h"
#include "includes/model_part.h"
#include "includes/serializer.h"
#include "includes/process_info.h"

#include "fluid_dynamics_application_variables.h"

namespace Kratos
{








template<unsigned int TDim, unsigned int TNumNodes, class... TWallModel>
class KRATOS_API(FLUID_DYNAMICS_APPLICATION) NavierStokesWallCondition : public Condition
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(NavierStokesWallCondition);

struct ConditionDataStruct
{
double wGauss;                                  
array_1d<double, 3> Normal;                     
array_1d<double, TNumNodes> N;                  
Vector ViscousStress;                           
};

static constexpr std::size_t VoigtSize = 3 * (TDim-1);
static constexpr std::size_t BlockSize = TDim + 1;
static constexpr std::size_t LocalSize = TNumNodes*BlockSize;

using Condition::SizeType;

typedef Node NodeType;

typedef Properties PropertiesType;

typedef Geometry<NodeType> GeometryType;

typedef Geometry<NodeType>::PointsArrayType NodesArrayType;

typedef Vector VectorType;

typedef Matrix MatrixType;

typedef std::size_t IndexType;

typedef std::vector<std::size_t> EquationIdVectorType;

typedef std::vector< Dof<double>::Pointer > DofsVectorType;



NavierStokesWallCondition(IndexType NewId = 0):Condition(NewId)
{
}


NavierStokesWallCondition(IndexType NewId, const NodesArrayType& ThisNodes):
Condition(NewId,ThisNodes)
{
}


NavierStokesWallCondition(IndexType NewId, GeometryType::Pointer pGeometry):
Condition(NewId,pGeometry)
{
}


NavierStokesWallCondition(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties):
Condition(NewId,pGeometry,pProperties)
{
}

NavierStokesWallCondition(NavierStokesWallCondition const& rOther):
Condition(rOther)
{
}

~NavierStokesWallCondition() override {}



NavierStokesWallCondition & operator=(NavierStokesWallCondition const& rOther)
{
Condition::operator=(rOther);
return *this;
}



Condition::Pointer Create(IndexType NewId, NodesArrayType const& ThisNodes, PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<NavierStokesWallCondition>(NewId, GetGeometry().Create(ThisNodes), pProperties);
}


Condition::Pointer Create(IndexType NewId, GeometryType::Pointer pGeom, PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive< NavierStokesWallCondition >(NewId, pGeom, pProperties);
}


Condition::Pointer Clone(IndexType NewId, NodesArrayType const& rThisNodes) const override
{
Condition::Pointer pNewCondition = Create(NewId, GetGeometry().Create( rThisNodes ), pGetProperties() );

pNewCondition->SetData(this->GetData());
pNewCondition->SetFlags(this->GetFlags());

return pNewCondition;
}



void CalculateLocalSystem(MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;



void CalculateLeftHandSide(MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;



void CalculateRightHandSide(VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;




int Check(const ProcessInfo& rCurrentProcessInfo) const override;


void EquationIdVector(
EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo) const override;


void GetDofList(
DofsVectorType& rConditionDofList,
const ProcessInfo& rCurrentProcessInfo) const override;

void Calculate(
const Variable< array_1d<double,3> >& rVariable,
array_1d<double,3>& Output,
const ProcessInfo& rCurrentProcessInfo) override;






std::string Info() const override
{
std::stringstream buffer;
buffer << "NavierStokesWallCondition" << TDim << "D";
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "NavierStokesWallCondition";
}

void PrintData(std::ostream& rOStream) const override {}





protected:








void CalculateNormal(array_1d<double,3>& rAreaNormal);


void ComputeGaussPointLHSContribution(
BoundedMatrix<double, LocalSize, LocalSize>& rLHS,
const ConditionDataStruct& rData,
const ProcessInfo& rProcessInfo);


void ComputeGaussPointRHSContribution(
array_1d<double, LocalSize>& rRHS,
const ConditionDataStruct& rData,
const ProcessInfo& rProcessInfo);


void ComputeRHSNeumannContribution(
array_1d<double,LocalSize>& rRHS,
const ConditionDataStruct& data);


void ComputeRHSOutletInflowContribution(
array_1d<double, LocalSize>& rRHS,
const ConditionDataStruct& rData,
const ProcessInfo& rProcessInfo);








private:





friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Condition );
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Condition );
}





void CalculateGaussPointSlipTangentialCorrectionLHSContribution(
BoundedMatrix<double,LocalSize,LocalSize>& rLeftHandSideMatrix,
const ConditionDataStruct& rDataStruct);


void CalculateGaussPointSlipTangentialCorrectionRHSContribution(
array_1d<double,LocalSize>& rRightHandSideVector,
const ConditionDataStruct& rDataStruct);


void ProjectViscousStress(
const Vector& rViscousStress,
const array_1d<double,3> rNormal,
array_1d<double,3>& rProjectedViscousStress);


void SetTangentialProjectionMatrix(
const array_1d<double,3>& rUnitNormal,
BoundedMatrix<double,TDim,TDim>& rTangProjMat)
{
noalias(rTangProjMat) = IdentityMatrix(TDim,TDim);
for (std::size_t d1 = 0; d1 < TDim; ++d1) {
for (std::size_t d2 = 0; d2 < TDim; ++d2) {
rTangProjMat(d1,d2) -= rUnitNormal[d1]*rUnitNormal[d2];
}
}
}

template<typename TWallModelType>
int WallModelCheckCall(const ProcessInfo& rProcessInfo) const
{
return TWallModelType::Check(this, rProcessInfo);
}

template<typename TWallModelType>
void AddWallModelRightHandSideCall(
VectorType& rRHS,
const ProcessInfo& rProcessInfo)
{
TWallModelType::AddWallModelRightHandSide(rRHS, this, rProcessInfo);
}

template<typename TWallModelType>
void AddWallModelLeftHandSideCall(
MatrixType& rLHS,
const ProcessInfo& rProcessInfo)
{
TWallModelType::AddWallModelLeftHandSide(rLHS, this, rProcessInfo);
}

template<typename TWallModelType>
void AddWallModelLocalSystemCall(
MatrixType& rLHS,
VectorType& rRHS,
const ProcessInfo& rProcessInfo)
{
TWallModelType::AddWallModelLocalSystem(rLHS, rRHS, this, rProcessInfo);
}








}; 







template< unsigned int TDim, unsigned int TNumNodes, class TWallModel >
inline std::istream& operator >> (std::istream& rIStream, NavierStokesWallCondition<TDim,TNumNodes,TWallModel>& rThis)
{
return rIStream;
}

template< unsigned int TDim, unsigned int TNumNodes, class TWallModel >
inline std::ostream& operator << (std::ostream& rOStream, const NavierStokesWallCondition<TDim,TNumNodes,TWallModel>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}




}  
