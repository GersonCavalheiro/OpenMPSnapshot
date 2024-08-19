
#pragma once



#include "includes/define.h"
#include "includes/master_slave_constraint.h"

namespace Kratos
{






class KRATOS_API(KRATOS_CORE) LinearMasterSlaveConstraint
:  public MasterSlaveConstraint
{
public:

typedef MasterSlaveConstraint BaseType;

typedef BaseType::IndexType IndexType;

typedef BaseType::DofType DofType;

typedef BaseType::DofPointerVectorType DofPointerVectorType;

typedef BaseType::NodeType NodeType;

typedef BaseType::EquationIdVectorType EquationIdVectorType;

typedef BaseType::MatrixType MatrixType;

typedef BaseType::VectorType VectorType;

typedef BaseType::VariableType VariableType;

KRATOS_CLASS_POINTER_DEFINITION(LinearMasterSlaveConstraint);




explicit LinearMasterSlaveConstraint(IndexType Id = 0)
: BaseType(Id)
{
}


LinearMasterSlaveConstraint(
IndexType Id,
DofPointerVectorType& rMasterDofsVector,
DofPointerVectorType& rSlaveDofsVector,
const MatrixType& rRelationMatrix,
const VectorType& rConstantVector
) : BaseType(Id),
mSlaveDofsVector(rSlaveDofsVector),
mMasterDofsVector(rMasterDofsVector),
mRelationMatrix(rRelationMatrix),
mConstantVector(rConstantVector)
{
}


LinearMasterSlaveConstraint(
IndexType Id,
NodeType& rMasterNode,
const VariableType& rMasterVariable,
NodeType& rSlaveNode,
const VariableType& rSlaveVariable,
const double Weight,
const double Constant
);

~LinearMasterSlaveConstraint() override
{

}

LinearMasterSlaveConstraint(const LinearMasterSlaveConstraint& rOther)
: BaseType(rOther),
mSlaveDofsVector(rOther.mSlaveDofsVector),
mMasterDofsVector(rOther.mMasterDofsVector),
mRelationMatrix(rOther.mRelationMatrix),
mConstantVector(rOther.mConstantVector)
{
}

LinearMasterSlaveConstraint& operator=(const LinearMasterSlaveConstraint& rOther)
{
BaseType::operator=( rOther );
mSlaveDofsVector = rOther.mSlaveDofsVector;
mMasterDofsVector = rOther.mMasterDofsVector;
mRelationMatrix = rOther.mRelationMatrix;
mConstantVector = rOther.mConstantVector;
return *this;
}




MasterSlaveConstraint::Pointer Create(
IndexType Id,
DofPointerVectorType& rMasterDofsVector,
DofPointerVectorType& rSlaveDofsVector,
const MatrixType& rRelationMatrix,
const VectorType& rConstantVector
) const override
{
KRATOS_TRY
return Kratos::make_shared<LinearMasterSlaveConstraint>(Id, rMasterDofsVector, rSlaveDofsVector, rRelationMatrix, rConstantVector);
KRATOS_CATCH("");
}


MasterSlaveConstraint::Pointer Create(
IndexType Id,
NodeType& rMasterNode,
const VariableType& rMasterVariable,
NodeType& rSlaveNode,
const VariableType& rSlaveVariable,
const double Weight,
const double Constant
) const override
{
KRATOS_TRY
return Kratos::make_shared<LinearMasterSlaveConstraint>(Id, rMasterNode, rMasterVariable, rSlaveNode, rSlaveVariable, Weight, Constant);
KRATOS_CATCH("");
}


MasterSlaveConstraint::Pointer Clone (IndexType NewId) const override
{
KRATOS_TRY

MasterSlaveConstraint::Pointer p_new_const = Kratos::make_shared<LinearMasterSlaveConstraint>(*this);
p_new_const->SetId(NewId);
p_new_const->SetData(this->GetData());
p_new_const->Set(Flags(*this));
return p_new_const;

KRATOS_CATCH("");
}


void GetDofList(
DofPointerVectorType& rSlaveDofsVector,
DofPointerVectorType& rMasterDofsVector,
const ProcessInfo& rCurrentProcessInfo
) const override
{
rSlaveDofsVector = mSlaveDofsVector;
rMasterDofsVector = mMasterDofsVector;
}


void SetDofList(
const DofPointerVectorType& rSlaveDofsVector,
const DofPointerVectorType& rMasterDofsVector,
const ProcessInfo& rCurrentProcessInfo
) override
{
mSlaveDofsVector = rSlaveDofsVector;
mMasterDofsVector = rMasterDofsVector;
}


void EquationIdVector(
EquationIdVectorType& rSlaveEquationIds,
EquationIdVectorType& rMasterEquationIds,
const ProcessInfo& rCurrentProcessInfo
) const override;


const DofPointerVectorType& GetSlaveDofsVector() const override
{
return mSlaveDofsVector;
}


void SetSlaveDofsVector(const DofPointerVectorType& rSlaveDofsVector) override
{
mSlaveDofsVector = rSlaveDofsVector;
}


const DofPointerVectorType& GetMasterDofsVector() const override
{
return mMasterDofsVector;
}


void SetMasterDofsVector(const DofPointerVectorType& rMasterDofsVector) override
{
mMasterDofsVector = rMasterDofsVector;
}


void ResetSlaveDofs(const ProcessInfo& rCurrentProcessInfo) override;


void Apply(const ProcessInfo& rCurrentProcessInfo) override;


void SetLocalSystem(
const MatrixType& rRelationMatrix,
const VectorType& rConstantVector,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateLocalSystem(
MatrixType& rRelationMatrix,
VectorType& rConstantVector,
const ProcessInfo& rCurrentProcessInfo
) const override;



std::string GetInfo() const override
{
return "Linear User Provided Master Slave Constraint class !";
}


void PrintInfo(std::ostream &rOStream) const override
{
rOStream << " LinearMasterSlaveConstraint Id  : " << this->Id() << std::endl;
rOStream << " Number of Slaves          : " << mSlaveDofsVector.size() << std::endl;
rOStream << " Number of Masters         : " << mMasterDofsVector.size() << std::endl;
}

protected:


DofPointerVectorType mSlaveDofsVector;  
DofPointerVectorType mMasterDofsVector; 
MatrixType mRelationMatrix;             
VectorType mConstantVector;             


private:


friend class Serializer;

void save(Serializer &rSerializer) const override;

void load(Serializer &rSerializer) override;


};


inline std::istream& operator>>(std::istream& rIStream, LinearMasterSlaveConstraint& rThis);

inline std::ostream& operator<<(std::ostream& rOStream,
const LinearMasterSlaveConstraint& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;

return rOStream;
}


} 
