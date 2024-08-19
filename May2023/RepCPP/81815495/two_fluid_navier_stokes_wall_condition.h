
#pragma once

#include <string>
#include <iostream>






#include "custom_conditions/navier_stokes_wall_condition.h"

namespace Kratos
{











template< unsigned int TDim, unsigned int TNumNodes, class... TWallModel>
class KRATOS_API(FLUID_DYNAMICS_APPLICATION) TwoFluidNavierStokesWallCondition : public NavierStokesWallCondition<TDim, TNumNodes, TWallModel...>
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(TwoFluidNavierStokesWallCondition);

typedef NavierStokesWallCondition<TDim, TNumNodes, TWallModel...> BaseType;

using BaseType::LocalSize;

typedef typename BaseType::ConditionDataStruct ConditionDataStruct;

typedef Node NodeType;

typedef Properties PropertiesType;

typedef Geometry<NodeType> GeometryType;

typedef Geometry<NodeType>::PointsArrayType NodesArrayType;

typedef Vector VectorType;

typedef Matrix MatrixType;

typedef std::size_t IndexType;

typedef std::vector<std::size_t> EquationIdVectorType;

typedef std::vector< Dof<double>::Pointer > DofsVectorType;



TwoFluidNavierStokesWallCondition(IndexType NewId = 0)
: NavierStokesWallCondition<TDim, TNumNodes, TWallModel...>(NewId)
{
}


TwoFluidNavierStokesWallCondition(
IndexType NewId,
const NodesArrayType& ThisNodes)
: NavierStokesWallCondition<TDim, TNumNodes, TWallModel...>(NewId, ThisNodes)
{
}


TwoFluidNavierStokesWallCondition(
IndexType NewId,
GeometryType::Pointer pGeometry)
: NavierStokesWallCondition<TDim, TNumNodes, TWallModel...>(NewId, pGeometry)
{
}


TwoFluidNavierStokesWallCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties)
: NavierStokesWallCondition<TDim, TNumNodes, TWallModel...>(NewId, pGeometry, pProperties)
{
}

TwoFluidNavierStokesWallCondition(TwoFluidNavierStokesWallCondition const& rOther):
NavierStokesWallCondition<TDim, TNumNodes, TWallModel...>(rOther)
{
}

~TwoFluidNavierStokesWallCondition() override {}



TwoFluidNavierStokesWallCondition & operator=(TwoFluidNavierStokesWallCondition const& rOther)
{
Condition::operator=(rOther);
return *this;
}



Condition::Pointer Create(
IndexType NewId,
NodesArrayType const& ThisNodes,
PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<TwoFluidNavierStokesWallCondition>(NewId, BaseType::GetGeometry().Create(ThisNodes), pProperties);
}


Condition::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive< TwoFluidNavierStokesWallCondition >(NewId, pGeom, pProperties);
}


Condition::Pointer Clone(
IndexType NewId,
NodesArrayType const& rThisNodes) const override
{
Condition::Pointer pNewCondition = Create(NewId, BaseType::GetGeometry().Create( rThisNodes ), BaseType::pGetProperties() );

pNewCondition->SetData(this->GetData());
pNewCondition->SetFlags(this->GetFlags());

return pNewCondition;
}


int Check(const ProcessInfo& rCurrentProcessInfo) const override;






std::string Info() const override
{
std::stringstream buffer;
buffer << "TwoFluidNavierStokesWallCondition" << TDim << "D";
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "TwoFluidNavierStokesWallCondition";
}

void PrintData(std::ostream& rOStream) const override {}



protected:















private:





friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, BaseType);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, BaseType);
}











}; 




template< unsigned int TDim, unsigned int TNumNodes, class TWallModel >
inline std::istream& operator >> (std::istream& rIStream, TwoFluidNavierStokesWallCondition<TDim,TNumNodes,TWallModel>& rThis)
{
return rIStream;
}

template< unsigned int TDim, unsigned int TNumNodes, class TWallModel >
inline std::ostream& operator << (std::ostream& rOStream, const TwoFluidNavierStokesWallCondition<TDim,TNumNodes,TWallModel>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  
