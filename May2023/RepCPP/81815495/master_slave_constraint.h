
#pragma once


#include "includes/define.h"
#include "includes/node.h"
#include "containers/flags.h"
#include "containers/variable.h"
#include "includes/process_info.h"
#include "includes/indexed_object.h"

namespace Kratos
{





class KRATOS_API(KRATOS_CORE) MasterSlaveConstraint
:  public IndexedObject, public Flags
{
public:

typedef IndexedObject BaseType;

typedef std::size_t IndexType;

typedef Dof<double> DofType;

typedef std::vector< DofType::Pointer > DofPointerVectorType;

typedef Node NodeType;

typedef std::vector<std::size_t> EquationIdVectorType;

typedef Matrix MatrixType;

typedef Vector VectorType;

typedef Kratos::Variable<double> VariableType;

KRATOS_CLASS_POINTER_DEFINITION(MasterSlaveConstraint);




explicit MasterSlaveConstraint(IndexType Id = 0) : IndexedObject(Id), Flags()
{
}

virtual ~MasterSlaveConstraint() override
{

}

MasterSlaveConstraint(const MasterSlaveConstraint& rOther)
: BaseType(rOther),
mData(rOther.mData)
{
}

MasterSlaveConstraint& operator=(const MasterSlaveConstraint& rOther)
{
BaseType::operator=( rOther );
mData = rOther.mData;
return *this;
}




virtual MasterSlaveConstraint::Pointer Create(
IndexType Id,
DofPointerVectorType& rMasterDofsVector,
DofPointerVectorType& rSlaveDofsVector,
const MatrixType& rRelationMatrix,
const VectorType& rConstantVector
) const
{
KRATOS_TRY

KRATOS_ERROR << "Create not implemented in MasterSlaveConstraintBaseClass" << std::endl;

KRATOS_CATCH("");
}


virtual MasterSlaveConstraint::Pointer Create(
IndexType Id,
NodeType& rMasterNode,
const VariableType& rMasterVariable,
NodeType& rSlaveNode,
const VariableType& rSlaveVariable,
const double Weight,
const double Constant
) const
{
KRATOS_TRY

KRATOS_ERROR << "Create not implemented in MasterSlaveConstraintBaseClass" << std::endl;

KRATOS_CATCH("");
}


virtual Pointer Clone (IndexType NewId) const
{
KRATOS_TRY

KRATOS_WARNING("MasterSlaveConstraint") << " Call base class constraint Clone " << std::endl;
MasterSlaveConstraint::Pointer p_new_const = Kratos::make_shared<MasterSlaveConstraint>(*this);
p_new_const->SetId(NewId);
p_new_const->SetData(this->GetData());
p_new_const->Set(Flags(*this));
return p_new_const;

KRATOS_CATCH("");
}


virtual void Clear()
{
}


virtual void Initialize(const ProcessInfo& rCurrentProcessInfo)
{
}


virtual void Finalize(const ProcessInfo& rCurrentProcessInfo)
{
this->Clear();
}


virtual void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo)
{
}


virtual void InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo)
{
}


virtual void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo)
{
}


virtual void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo)
{
}


virtual void GetDofList(
DofPointerVectorType& rSlaveDofsVector,
DofPointerVectorType& rMasterDofsVector,
const ProcessInfo& rCurrentProcessInfo
) const
{
KRATOS_ERROR << "GetDofList not implemented in MasterSlaveConstraintBaseClass" << std::endl;
}


virtual void SetDofList(
const DofPointerVectorType& rSlaveDofsVector,
const DofPointerVectorType& rMasterDofsVector,
const ProcessInfo& rCurrentProcessInfo
)
{
KRATOS_ERROR << "SetDofList not implemented in MasterSlaveConstraintBaseClass" << std::endl;
}


virtual void EquationIdVector(
EquationIdVectorType& rSlaveEquationIds,
EquationIdVectorType& rMasterEquationIds,
const ProcessInfo& rCurrentProcessInfo
) const
{
if (rSlaveEquationIds.size() != 0)
rSlaveEquationIds.resize(0);

if (rMasterEquationIds.size() != 0)
rMasterEquationIds.resize(0);
}


virtual const DofPointerVectorType& GetSlaveDofsVector() const
{
KRATOS_ERROR << "GetSlaveDofsVector not implemented in MasterSlaveConstraintBaseClass" << std::endl;
}


virtual void SetSlaveDofsVector(const DofPointerVectorType& rSlaveDofsVector)
{
KRATOS_ERROR << "SetSlaveDofsVector not implemented in MasterSlaveConstraintBaseClass" << std::endl;
}


virtual const DofPointerVectorType& GetMasterDofsVector() const
{
KRATOS_ERROR << "GetMasterDofsVector not implemented in MasterSlaveConstraintBaseClass" << std::endl;
}


virtual void SetMasterDofsVector(const DofPointerVectorType& rMasterDofsVector)
{
KRATOS_ERROR << "SetMasterDofsVector not implemented in MasterSlaveConstraintBaseClass" << std::endl;
}


virtual void ResetSlaveDofs(const ProcessInfo& rCurrentProcessInfo)
{
KRATOS_ERROR << "ResetSlaveDofs not implemented in MasterSlaveConstraintBaseClass" << std::endl;
}


virtual void Apply(const ProcessInfo& rCurrentProcessInfo)
{
KRATOS_ERROR << "Apply not implemented in MasterSlaveConstraintBaseClass" << std::endl;
}


virtual void SetLocalSystem(
const MatrixType& rRelationMatrix,
const VectorType& rConstantVector,
const ProcessInfo& rCurrentProcessInfo
)
{
KRATOS_TRY

KRATOS_ERROR << "SetLocalSystem not implemented in MasterSlaveConstraintBaseClass" << std::endl;

KRATOS_CATCH("");
}


virtual void GetLocalSystem(
MatrixType& rRelationMatrix,
VectorType& rConstantVector,
const ProcessInfo& rCurrentProcessInfo
) const
{
KRATOS_TRY

this->CalculateLocalSystem(rRelationMatrix, rConstantVector, rCurrentProcessInfo);

KRATOS_CATCH("");
}


virtual void CalculateLocalSystem(
MatrixType& rRelationMatrix,
VectorType& rConstantVector,
const ProcessInfo& rCurrentProcessInfo
) const
{
if (rRelationMatrix.size1() != 0) {
rRelationMatrix.resize(0, 0, false);
}

if (rConstantVector.size() != 0) {
rConstantVector.resize(0, false);
}
}


virtual int Check(const ProcessInfo& rCurrentProcessInfo) const
{
KRATOS_TRY

KRATOS_ERROR_IF( this->Id() < 1 ) << "MasterSlaveConstraint found with Id " << this->Id() << std::endl;

return 0;

KRATOS_CATCH("")
}



virtual std::string GetInfo() const
{
return " Constraint base class !";
}


virtual void PrintInfo(std::ostream &rOStream) const override
{
rOStream << " MasterSlaveConstraint Id  : " << this->Id() << std::endl;
}



DataValueContainer& Data()
{
return mData;
}


DataValueContainer const& GetData() const
{
return mData;
}


void SetData(DataValueContainer const& rThisData)
{
mData = rThisData;
}


template<class TDataType> 
bool Has(const Variable<TDataType>& rThisVariable) const
{
return mData.Has(rThisVariable);
}


template<class TVariableType>
void SetValue(
const TVariableType& rThisVariable,
typename TVariableType::Type const& rValue
)
{
mData.SetValue(rThisVariable, rValue);
}


template<class TVariableType>
typename TVariableType::Type& GetValue(const TVariableType& rThisVariable)
{
return mData.GetValue(rThisVariable);
}


template<class TVariableType>
typename TVariableType::Type& GetValue(const TVariableType& rThisVariable) const
{
return mData.GetValue(rThisVariable);
}



bool IsActive() const;

protected:








private:



DataValueContainer mData; 


friend class Serializer;

virtual void save(Serializer &rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, IndexedObject);
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Flags);
rSerializer.save("Data", mData);
}

virtual void load(Serializer &rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, IndexedObject);
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Flags);
rSerializer.load("Data", mData);
}
};

KRATOS_API_EXTERN template class KRATOS_API(KRATOS_CORE) KratosComponents<MasterSlaveConstraint>;


inline std::istream& operator >> (std::istream& rIStream,
MasterSlaveConstraint& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const MasterSlaveConstraint& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;

return rOStream;
}


} 
