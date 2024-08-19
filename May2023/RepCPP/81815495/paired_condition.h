
#pragma once



#include "includes/condition.h"
#include "geometries/coupling_geometry.h"

namespace Kratos
{







class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) PairedCondition
: public Condition
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( PairedCondition );

typedef Condition                                                           BaseType;

typedef Point                                                              PointType;

typedef Node                                                             NodeType;

typedef Geometry<NodeType>                                              GeometryType;

typedef CouplingGeometry<NodeType>                              CouplingGeometryType;

typedef BaseType::VectorType                                              VectorType;

typedef BaseType::MatrixType                                              MatrixType;

typedef BaseType::IndexType                                                IndexType;

typedef BaseType::GeometryType::Pointer                          GeometryPointerType;

typedef BaseType::NodesArrayType                                      NodesArrayType;

typedef BaseType::PropertiesType::Pointer                      PropertiesPointerType;


PairedCondition()
: Condition()
{}

PairedCondition(
IndexType NewId,
GeometryType::Pointer pGeometry
) :Condition(NewId, Kratos::make_shared<CouplingGeometryType>(pGeometry, nullptr))
{
KRATOS_WARNING_FIRST_N("PairedCondition", 10) << "This class pairs two geometries, please use the other constructor (the one with two geometries as input)" << std::endl;
}

PairedCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
) :Condition( NewId, Kratos::make_shared<CouplingGeometryType>(pGeometry, nullptr), pProperties )
{
KRATOS_WARNING_FIRST_N("PairedCondition", 10) << "This class pairs two geometries, please use the other constructor (the one with two geometries as input)" << std::endl;
}

PairedCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties,
GeometryType::Pointer pPairedGeometry
)
:Condition( NewId, Kratos::make_shared<CouplingGeometryType>(pGeometry, pPairedGeometry), pProperties )
{}

PairedCondition( PairedCondition const& rOther){}

~PairedCondition() override;





void Initialize(const ProcessInfo& rCurrentProcessInfo) override;


void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;


void InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;


Condition::Pointer Create(
IndexType NewId,
NodesArrayType const& rThisNodes,
PropertiesType::Pointer pProperties
) const override;


Condition::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
) const override;


virtual Condition::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties,
GeometryType::Pointer pPairedGeom
) const;



GeometryType::Pointer pGetParentGeometry()
{
return this->GetGeometry().pGetGeometryPart(CouplingGeometryType::Master);
}


GeometryType::Pointer const pGetParentGeometry() const
{
return this->GetGeometry().pGetGeometryPart(CouplingGeometryType::Master);
}


GeometryType::Pointer pGetPairedGeometry()
{
return this->GetGeometry().pGetGeometryPart(CouplingGeometryType::Slave);
}


GeometryType::Pointer const pGetPairedGeometry() const
{
return this->GetGeometry().pGetGeometryPart(CouplingGeometryType::Slave);
}


GeometryType& GetParentGeometry()
{
return this->GetGeometry().GetGeometryPart(CouplingGeometryType::Master);
}


GeometryType const& GetParentGeometry() const
{
return this->GetGeometry().GetGeometryPart(CouplingGeometryType::Master);
}


GeometryType& GetPairedGeometry()
{
return this->GetGeometry().GetGeometryPart(CouplingGeometryType::Slave);
}


GeometryType const& GetPairedGeometry() const
{
return this->GetGeometry().GetGeometryPart(CouplingGeometryType::Slave);
}


void SetPairedNormal(const array_1d<double, 3>& rPairedNormal)
{
noalias(mPairedNormal) = rPairedNormal;
}


array_1d<double, 3> const& GetPairedNormal() const
{
return mPairedNormal;
}



std::string Info() const override
{
std::stringstream buffer;
buffer << "PairedCondition #" << this->Id();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "PairedCondition #" << this->Id();
}

void PrintData(std::ostream& rOStream) const override
{
PrintInfo(rOStream);
this->GetParentGeometry().PrintData(rOStream);
this->GetPairedGeometry().PrintData(rOStream);
}



protected:







private:


array_1d<double, 3> mPairedNormal = ZeroVector(3);







friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, Condition );
rSerializer.save("PairedNormal", mPairedNormal);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, Condition );
rSerializer.load("PairedNormal", mPairedNormal);
}


}; 





}
