
#pragma once



#include "constraints/linear_master_slave_constraint.h"

namespace Kratos
{


typedef std::size_t SizeType;





class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) ContactMasterSlaveConstraint
:  public LinearMasterSlaveConstraint
{
public:

typedef MasterSlaveConstraint BaseConstraintType;

typedef LinearMasterSlaveConstraint BaseType;

typedef BaseType::IndexType IndexType;

typedef BaseType::DofType DofType;

typedef BaseType::DofPointerVectorType DofPointerVectorType;

typedef BaseType::NodeType NodeType;

typedef BaseType::EquationIdVectorType EquationIdVectorType;

typedef BaseType::MatrixType MatrixType;

typedef BaseType::VectorType VectorType;

typedef BaseType::VariableType VariableType;

KRATOS_CLASS_POINTER_DEFINITION(ContactMasterSlaveConstraint);




explicit ContactMasterSlaveConstraint(IndexType Id = 0);


ContactMasterSlaveConstraint(
IndexType Id,
DofPointerVectorType& rMasterDofsVector,
DofPointerVectorType& rSlaveDofsVector,
const MatrixType& rRelationMatrix,
const VectorType& rConstantVector
);


ContactMasterSlaveConstraint(
IndexType Id,
NodeType& rMasterNode,
const VariableType& rMasterVariable,
NodeType& rSlaveNode,
const VariableType& rSlaveVariable,
const double Weight,
const double Constant
);

~ContactMasterSlaveConstraint() override;

ContactMasterSlaveConstraint(const ContactMasterSlaveConstraint& rOther);

ContactMasterSlaveConstraint& operator=(const ContactMasterSlaveConstraint& rOther);




MasterSlaveConstraint::Pointer Create(
IndexType Id,
DofPointerVectorType& rMasterDofsVector,
DofPointerVectorType& rSlaveDofsVector,
const MatrixType& rRelationMatrix,
const VectorType& rConstantVector
) const override;


MasterSlaveConstraint::Pointer Create(
IndexType Id,
NodeType& rMasterNode,
const VariableType& rMasterVariable,
NodeType& rSlaveNode,
const VariableType& rSlaveVariable,
const double Weight,
const double Constant
) const override;


void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;



std::string GetInfo() const override;


void PrintInfo(std::ostream &rOStream) const override;

protected:








private:


friend class Serializer;

void save(Serializer &rSerializer) const override;

void load(Serializer &rSerializer) override;
};


inline std::istream& operator>>(std::istream& rIStream, ContactMasterSlaveConstraint& rThis);

inline std::ostream& operator<<(std::ostream& rOStream,
const ContactMasterSlaveConstraint& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;

return rOStream;
}


} 
