
#if !defined(KRATOS_EXPLICIT_BUILDER_AND_SOLVER_H_INCLUDED)
#define  KRATOS_EXPLICIT_BUILDER_AND_SOLVER_H_INCLUDED



#include "custom_solvers/solution_builders_and_solvers/solution_builder_and_solver.hpp"

namespace Kratos
{







template<class TSparseSpace,
class TDenseSpace, 
class TLinearSolver 
>
class ExplicitBuilderAndSolver : public SolutionBuilderAndSolver< TSparseSpace, TDenseSpace, TLinearSolver >
{
public:



KRATOS_CLASS_POINTER_DEFINITION( ExplicitBuilderAndSolver );

typedef SolutionBuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver>      BaseType;

typedef typename BaseType::LocalFlagType                                   LocalFlagType;
typedef typename BaseType::DofsArrayType                                   DofsArrayType;

typedef typename BaseType::SystemMatrixType                             SystemMatrixType;
typedef typename BaseType::SystemVectorType                             SystemVectorType;
typedef typename BaseType::SystemMatrixPointerType               SystemMatrixPointerType;
typedef typename BaseType::SystemVectorPointerType               SystemVectorPointerType;
typedef typename BaseType::LocalSystemVectorType                   LocalSystemVectorType;
typedef typename BaseType::LocalSystemMatrixType                   LocalSystemMatrixType;

typedef typename ModelPart::NodesContainerType                        NodesContainerType;
typedef typename ModelPart::ElementsContainerType                  ElementsContainerType;
typedef typename ModelPart::ConditionsContainerType              ConditionsContainerType;

typedef typename BaseType::SchemePointerType                           SchemePointerType;


ExplicitBuilderAndSolver()
: BaseType()
{
}

~ExplicitBuilderAndSolver() override
{
}


void BuildLHS(SchemePointerType pScheme,
ModelPart& rModelPart,
SystemMatrixType& rA) override
{
KRATOS_TRY

NodesContainerType& pNodes         = rModelPart.Nodes();
ElementsContainerType& pElements   = rModelPart.Elements();
ProcessInfo& rCurrentProcessInfo   = rModelPart.GetProcessInfo();

#ifdef _OPENMP
int number_of_threads = omp_get_max_threads();
#else
int number_of_threads = 1;
#endif

vector<unsigned int> node_partition;
OpenMPUtils::CreatePartition(number_of_threads, pNodes.size(), node_partition);

vector<unsigned int> element_partition;
OpenMPUtils::CreatePartition(number_of_threads, pElements.size(), element_partition);

#pragma omp parallel
{
#pragma omp for

for(int k=0; k<number_of_threads; k++)
{
typename NodesContainerType::iterator i_begin=pNodes.ptr_begin()+node_partition[k];
typename NodesContainerType::iterator i_end=pNodes.ptr_begin()+node_partition[k+1];

for(ModelPart::NodeIterator i=i_begin; i!= i_end; ++i)
{
double& nodal_mass    =  i->FastGetSolutionStepValue(NODAL_MASS);
nodal_mass = 0.0;
}
}

}


bool CalculateLumpedMassMatrix = false;
if( rCurrentProcessInfo.Has(COMPUTE_LUMPED_MASS_MATRIX) ){
CalculateLumpedMassMatrix = rCurrentProcessInfo[COMPUTE_LUMPED_MASS_MATRIX];
}

#pragma omp parallel
{
int k = OpenMPUtils::ThisThread();
typename ElementsContainerType::iterator ElemBegin = pElements.begin() + element_partition[k];
typename ElementsContainerType::iterator ElemEnd = pElements.begin() + element_partition[k + 1];

for (typename ElementsContainerType::iterator itElem = ElemBegin; itElem != ElemEnd; ++itElem)  
{
Matrix MassMatrix;

Element::GeometryType& geometry = itElem->GetGeometry();

(itElem)->CalculateMassMatrix(MassMatrix, rCurrentProcessInfo);

const unsigned int dimension   = geometry.WorkingSpaceDimension();

unsigned int index = 0;
for (unsigned int i = 0; i <geometry.size(); i++)
{
index = i*dimension;

double& mass = geometry(i)->FastGetSolutionStepValue(NODAL_MASS);

geometry(i)->SetLock();

if(!CalculateLumpedMassMatrix){
for (unsigned int j = 0; j <MassMatrix.size2(); j++)
{
mass += MassMatrix(index,j);
}
}
else{
mass += MassMatrix(index,index);
}

geometry(i)->UnSetLock();
}
}
}

rCurrentProcessInfo[COMPUTE_LUMPED_MASS_MATRIX] = CalculateLumpedMassMatrix;

KRATOS_CATCH( "" )

}

/
void Clear() override
{
BaseType::Clear();
}


int Check(ModelPart& rModelPart) override
{
KRATOS_TRY

return 0;

KRATOS_CATCH( "" )
}





protected:

private:


}; 






} 

#endif 


