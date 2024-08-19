
#if !defined(KRATOS_TRILINOS_DOF_UPDATER_H_INCLUDED )
#define  KRATOS_TRILINOS_DOF_UPDATER_H_INCLUDED

#include "includes/define.h"
#include "includes/model_part.h"
#include "utilities/dof_updater.h"

namespace Kratos
{



template< class TSparseSpace >
class TrilinosDofUpdater : public DofUpdater<TSparseSpace>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(TrilinosDofUpdater);

using BaseType = DofUpdater<TSparseSpace>;
using DofsArrayType = typename BaseType::DofsArrayType;
using SystemVectorType = typename BaseType::SystemVectorType;


TrilinosDofUpdater():
DofUpdater<TSparseSpace>()
{}

TrilinosDofUpdater(TrilinosDofUpdater const& rOther) = delete;

~TrilinosDofUpdater() override {}

TrilinosDofUpdater& operator=(TrilinosDofUpdater const& rOther) = delete;



typename BaseType::UniquePointer Create() const override
{
return Kratos::make_unique<TrilinosDofUpdater>();
}


void Initialize(
const DofsArrayType& rDofSet,
const SystemVectorType& rDx) override
{
int system_size = TSparseSpace::Size(rDx);
int number_of_dofs = rDofSet.size();
std::vector< int > index_array(number_of_dofs);

unsigned int counter = 0;
for(typename DofsArrayType::const_iterator i_dof = rDofSet.begin() ; i_dof != rDofSet.end() ; ++i_dof)
{
int id = i_dof->EquationId();
if( id < system_size )
{
index_array[counter++] = id;
}
}

std::sort(index_array.begin(),index_array.end());
std::vector<int>::iterator new_end = std::unique(index_array.begin(),index_array.end());
index_array.resize(new_end - index_array.begin());

int check_size = -1;
int tot_update_dofs = index_array.size();
rDx.Comm().SumAll(&tot_update_dofs,&check_size,1);
if ( (check_size < system_size) && (rDx.Comm().MyPID() == 0) )
{
std::stringstream msg;
msg << "Dof count is not correct. There are less dofs then expected." << std::endl;
msg << "Expected number of active dofs: " << system_size << ", dofs found: " << check_size << std::endl;
KRATOS_ERROR << msg.str();
}

Epetra_Map dof_update_map(-1,index_array.size(), &(*(index_array.begin())),0,rDx.Comm() );

std::unique_ptr<Epetra_Import> p_dof_import(new Epetra_Import(dof_update_map,rDx.Map()));
mpDofImport.swap(p_dof_import);

mImportIsInitialized = true;
}

void Clear() override
{
mpDofImport.reset();
mImportIsInitialized = false;
}


void UpdateDofs(
DofsArrayType& rDofSet,
const SystemVectorType& rDx) override
{
KRATOS_TRY;

if (!mImportIsInitialized)
this->Initialize(rDofSet,rDx);

int system_size = TSparseSpace::Size(rDx);

Epetra_Vector local_dx( mpDofImport->TargetMap() );

int ierr = local_dx.Import(rDx,*mpDofImport,Insert) ;
KRATOS_ERROR_IF(ierr != 0) << "Epetra failure found while trying to import Dx." << std::endl;

int num_dof = rDofSet.size();

#pragma omp parallel for
for(int i = 0;  i < num_dof; ++i) {
auto it_dof = rDofSet.begin() + i;

if (it_dof->IsFree()) {
int global_id = it_dof->EquationId();
if(global_id < system_size) {
double dx_i = local_dx[mpDofImport->TargetMap().LID(global_id)];
it_dof->GetSolutionStepValue() += dx_i;
}
}
}

KRATOS_CATCH("");
}


std::string Info() const override
{
std::stringstream buffer;
buffer << "TrilinosDofUpdater" ;
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << this->Info() << std::endl;
}

void PrintData(std::ostream& rOStream) const override
{
rOStream << this->Info() << std::endl;
}


private:


bool mImportIsInitialized = false;

std::unique_ptr<Epetra_Import> mpDofImport = nullptr;


}; 


template< class TSparseSpace >
inline std::istream& operator >> (
std::istream& rIStream,
TrilinosDofUpdater<TSparseSpace>& rThis)
{
return rIStream;
}

template< class TSparseSpace >
inline std::ostream& operator << (
std::ostream& rOStream,
const TrilinosDofUpdater<TSparseSpace>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}



}  

#endif 
