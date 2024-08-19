


#if !defined(KRATOS_RESIDUAL_BASED_ELIMINATION_QUASI_INCOMPRESSIBLE_BUILDER_AND_SOLVER )
#define  KRATOS_RESIDUAL_BASED_ELIMINATION_QUASI_INCOMPRESSIBLE_BUILDER_AND_SOLVER



#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif


#include <pybind11/pybind11.h>
#include "includes/define.h"
#include "includes/define_python.h"


#include "includes/define.h"
#include "ULF_application.h"
#include "solving_strategies/builder_and_solvers/builder_and_solver.h"
#include "utilities/geometry_utilities.h"

#include "boost/smart_ptr.hpp"
#include "utilities/timer.h"

namespace Kratos
{



























template
<
class TSparseSpace,
class TDenseSpace ,
class TLinearSolver,
int TDim
>
class ResidualBasedEliminationQuasiIncompressibleBuilderAndSolver
: public BuilderAndSolver< TSparseSpace,TDenseSpace,TLinearSolver >
{
public:


KRATOS_CLASS_POINTER_DEFINITION( ResidualBasedEliminationQuasiIncompressibleBuilderAndSolver );

typedef BuilderAndSolver<TSparseSpace,TDenseSpace, TLinearSolver> BaseType;

typedef typename BaseType::TSchemeType TSchemeType;
typedef typename BaseType::TDataType TDataType;

typedef typename BaseType::DofsArrayType DofsArrayType;

typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

typedef typename BaseType::TSystemVectorType TSystemVectorType;

typedef typename BaseType::LocalSystemVectorType LocalSystemVectorType;

typedef typename BaseType::LocalSystemMatrixType LocalSystemMatrixType;
typedef typename BaseType::TSystemMatrixPointerType TSystemMatrixPointerType;
typedef typename BaseType::TSystemVectorPointerType TSystemVectorPointerType;

typedef typename BaseType::NodesArrayType NodesContainerType;

typedef typename BaseType::NodesArrayType NodesArrayType;
typedef typename BaseType::ElementsArrayType ElementsArrayType;
typedef typename BaseType::ConditionsArrayType ConditionsArrayType;

typedef typename BaseType::ElementsContainerType ElementsContainerType;






ResidualBasedEliminationQuasiIncompressibleBuilderAndSolver(
typename TLinearSolver::Pointer pNewLinearSystemSolver)
: BuilderAndSolver< TSparseSpace,TDenseSpace,TLinearSolver >(pNewLinearSystemSolver)
{
}



~ResidualBasedEliminationQuasiIncompressibleBuilderAndSolver() override {}








/

KRATOS_CATCH("")
}

/

}
}
vector<unsigned int> condition_partition;
CreatePartition(number_of_threads, ConditionsArray.size(), condition_partition);

#pragma omp parallel for
for (int k = 0; k < number_of_threads; k++)
{
LocalSystemMatrixType LHS_Contribution = LocalSystemMatrixType(0, 0);
LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Condition::EquationIdVectorType EquationId;

ProcessInfo& CurrentProcessInfo = r_model_part.GetProcessInfo();

typename ConditionsArrayType::ptr_iterator it_begin = ConditionsArray.ptr_begin() + condition_partition[k];
typename ConditionsArrayType::ptr_iterator it_end = ConditionsArray.ptr_begin() + condition_partition[k + 1];

for (typename ConditionsArrayType::ptr_iterator it = it_begin; it != it_end; ++it)
{
pScheme->Condition_CalculateSystemContributions(*it, LHS_Contribution, RHS_Contribution, EquationId, CurrentProcessInfo);

Assemble(A, b, LHS_Contribution, RHS_Contribution, EquationId, lock_array);

}
}



double stop_prod = omp_get_wtime();
std::cout << "time: " << stop_prod - start_prod << std::endl;

for (int i = 0; i < A_size; i++)
omp_destroy_lock(&lock_array[i]);

#endif

KRATOS_CATCH("")

}























protected:








































public:








TSystemMatrixType mD;
TSystemMatrixType mMconsistent;
TSystemVectorType mMdiagInv;
TSystemVectorType mpreconditioner;
unsigned int mnumber_of_active_nodes;
GlobalPointersVector<Node > mActiveNodes;













/

typename DofsArrayType::ptr_iterator it2;
for (it2=BaseType::mDofSet.ptr_begin(); it2 != BaseType::mDofSet.ptr_end(); ++it2)
{



if ( (*it2)->IsFixed()  )
{
unsigned int eq_id=(*it2)->EquationId();

(*it2)->GetSolutionStepReactionValue() = b[eq_id];
}
}

}

/

}
}
vector<unsigned int> condition_partition;
CreatePartition(number_of_threads, ConditionsArray.size(), condition_partition);

#pragma omp parallel for
for (int k = 0; k < number_of_threads; k++)
{

LocalSystemVectorType RHS_Contribution = LocalSystemVectorType(0);

Condition::EquationIdVectorType EquationId;

ProcessInfo& CurrentProcessInfo = r_model_part.GetProcessInfo();

typename ConditionsArrayType::ptr_iterator it_begin = ConditionsArray.ptr_begin() + condition_partition[k];
typename ConditionsArrayType::ptr_iterator it_end = ConditionsArray.ptr_begin() + condition_partition[k + 1];

for (typename ConditionsArrayType::ptr_iterator it = it_begin; it != it_end; ++it)
{
pScheme->Condition_Calculate_RHS_Contribution(*it,RHS_Contribution,EquationId,CurrentProcessInfo);

AssembleRHS_parallel(b, RHS_Contribution, EquationId, lock_array);

}
}



double stop_prod = omp_get_wtime();
std::cout << "time: " << stop_prod - start_prod << std::endl;

for (int i = 0; i < b_size; i++)
omp_destroy_lock(&lock_array[i]);

#endif

KRATOS_CATCH("")

}
/

/


}
KRATOS_CATCH("");

}

/
void SavePressureIteration(ModelPart& model_part)
{
KRATOS_TRY
double pres=0.0;
for (typename ModelPart::NodesContainerType::iterator it=model_part.NodesBegin(); it!=model_part.NodesEnd(); ++it)
{
pres=it->FastGetSolutionStepValue(PRESSURE);
it->FastGetSolutionStepValue(PRESSURE_OLD_IT)=pres;
}
KRATOS_CATCH("");
}
void FractionalStepProjection(ModelPart& model_part, double alpha_bossak)
{
KRATOS_TRY
double dt = model_part.GetProcessInfo()[DELTA_TIME];
BoundedMatrix<double,3,2> DN_DX;
array_1d<double,3> N;
array_1d<double,3> aux0, aux1, aux2; 


for (typename ModelPart::NodesContainerType::iterator it=model_part.NodesBegin(); it!=model_part.NodesEnd(); ++it)
{
it->FastGetSolutionStepValue(VAUX)=ZeroVector(3);
}


for (typename ModelPart::ElementsContainerType::iterator im=model_part.ElementsBegin(); im!=model_part.ElementsEnd(); ++im)
{
Geometry< Node >& geom = im->GetGeometry();

double volume;
GeometryUtils::CalculateGeometryData(geom, DN_DX, N, volume);

array_1d<double,3> pres_inc;


pres_inc[0] = geom[0].FastGetSolutionStepValue(PRESSURE_OLD_IT)-geom[0].FastGetSolutionStepValue(PRESSURE);
pres_inc[1] = geom[1].FastGetSolutionStepValue(PRESSURE_OLD_IT)-geom[1].FastGetSolutionStepValue(PRESSURE);
pres_inc[2] = geom[2].FastGetSolutionStepValue(PRESSURE_OLD_IT)-geom[2].FastGetSolutionStepValue(PRESSURE);



BoundedMatrix<double,6,2> shape_func = ZeroMatrix(6, 2);
BoundedMatrix<double,6,3> G = ZeroMatrix(6,3);
for (int ii = 0; ii< 3; ii++)
{
int column = ii*2;
shape_func(column,0) = N[ii];
shape_func(column + 1, 1) = shape_func(column,0);
}
noalias(G)=prod(shape_func, trans(DN_DX));
G*=volume;

array_1d<double,6> aaa;
noalias(aaa) = prod(G,pres_inc);

array_1d<double,3> aux;

aux[0]=aaa[0];
aux[1]=aaa[1];
aux[2]=0.0;
geom[0].FastGetSolutionStepValue(VAUX) += aux;

aux[0]=aaa[2];
aux[1]=aaa[3];

geom[1].FastGetSolutionStepValue(VAUX) += aux;
aux[0]=aaa[4];
aux[1]=aaa[5];

geom[2].FastGetSolutionStepValue(VAUX) += aux;
}

alpha_bossak=-0.3;
double coef=0.25*(1.0-alpha_bossak);



for (typename ModelPart::NodesContainerType::iterator it=model_part.NodesBegin(); it!=model_part.NodesEnd(); ++it)
{
if( (it->GetValue(NEIGHBOUR_NODES)).size() != 0)
{
if (it->FastGetSolutionStepValue(NODAL_MASS)>0.0000000001)
{

double dt_sq_Minv =coef*dt*dt / it->FastGetSolutionStepValue(NODAL_MASS);

array_1d<double,3>& temp = it->FastGetSolutionStepValue(VAUX);

if(!it->IsFixed(DISPLACEMENT_X))
{
it->FastGetSolutionStepValue(DISPLACEMENT_X)+=dt_sq_Minv*temp[0];
}
if(!it->IsFixed(DISPLACEMENT_Y))
{
it->FastGetSolutionStepValue(DISPLACEMENT_Y)+=dt_sq_Minv*temp[1];
}
}
}
}
KRATOS_CATCH("");
}
void UpdateAfterProjection( ModelPart& model_part, double alpha_bossak)
{
KRATOS_TRY
double dt = model_part.GetProcessInfo()[DELTA_TIME];
array_1d<double,3> DeltaDisp;
double beta_newmark = 0.25*pow((1.00-alpha_bossak),2);

double gamma_newmark = 0.5-alpha_bossak;



double ma0=1.0/(beta_newmark*pow(dt,2));
double ma1=gamma_newmark/(beta_newmark*dt);
double ma2=1.0/(beta_newmark*dt);
double ma3=(1.0/(2.0*beta_newmark))-1.0;
double ma4=(gamma_newmark/beta_newmark)-1.0;
double ma5=dt*0.5*((gamma_newmark/beta_newmark)-2.0);

for(ModelPart::NodeIterator i = model_part.NodesBegin() ; i != model_part.NodesEnd() ; ++i)
{
noalias(DeltaDisp) = (i)->FastGetSolutionStepValue(DISPLACEMENT)  - (i)->FastGetSolutionStepValue(DISPLACEMENT,1);
array_1d<double,3>& CurrentVelocity = (i)->FastGetSolutionStepValue(VELOCITY,0);
array_1d<double,3>& OldVelocity = (i)->FastGetSolutionStepValue(VELOCITY,1);

array_1d<double,3>& CurrentAcceleration = (i)->FastGetSolutionStepValue(ACCELERATION,0);
array_1d<double,3>& OldAcceleration = (i)->FastGetSolutionStepValue(ACCELERATION,1);

UpdateVelocity(CurrentVelocity,DeltaDisp,OldVelocity,OldAcceleration, ma1, ma4, ma5);
UpdateAcceleration(CurrentAcceleration,DeltaDisp,OldVelocity,OldAcceleration, ma0, ma2, ma3);
}
KRATOS_CATCH("");
}
inline void UpdateVelocity(array_1d<double, 3>& CurrentVelocity, const array_1d<double, 3>& DeltaDisp,
const array_1d<double, 3>& OldVelocity,
const array_1d<double, 3>& OldAcceleration, double& ma1, double& ma4, double & ma5)
{
noalias(CurrentVelocity) = ma1*DeltaDisp - ma4*OldVelocity - ma5*OldAcceleration;
}
inline void UpdateAcceleration(array_1d<double, 3>& CurrentAcceleration, const array_1d<double, 3>& DeltaDisp,
const array_1d<double, 3>& OldVelocity,
const array_1d<double, 3>& OldAcceleration, double& ma0, double& ma2, double & ma3)
{
noalias(CurrentAcceleration) = ma0*DeltaDisp - ma2*OldVelocity - ma3*OldAcceleration;
}

void UpdatePressuresNew (TSystemMatrixType& mMconsistent, TSystemVectorType& mMdiagInv,ModelPart& r_model_part, double bulk_modulus, double density)
{
KRATOS_TRY
unsigned int dof_position = (r_model_part.NodesBegin())->GetDofPosition(DISPLACEMENT_X);

for (typename NodesArrayType::iterator in=r_model_part.NodesBegin(); in!=r_model_part.NodesEnd(); ++in)
{
in->FastGetSolutionStepValue(PRESSURE)=0.0;
}
const int size = TSparseSpace::Size(mMdiagInv);

TSystemVectorType p_n(size);
TSystemVectorType temp(size);
TSystemVectorType history(size);

TSparseSpace::SetToZero(p_n);
TSparseSpace::SetToZero(history);






int i=0;
for (typename NodesArrayType::iterator in=r_model_part.NodesBegin(); in!=r_model_part.NodesEnd(); ++in)
{
if( (in->GetValue(NEIGHBOUR_NODES)).size() != 0 )
{
i=in->GetDof(DISPLACEMENT_X,dof_position).EquationId()/TDim;
p_n[i]=in->FastGetSolutionStepValue(PRESSURE,1);

}


}

TSparseSpace::Mult(mMconsistent, p_n, history);

int aa=0;

for (typename NodesArrayType::iterator in=r_model_part.NodesBegin(); in!=r_model_part.NodesEnd(); ++in)
{
if( (in->GetValue(NEIGHBOUR_NODES)).size() != 0)
{
aa=in->GetDof(DISPLACEMENT_X,dof_position).EquationId()/TDim;
if (in->FastGetSolutionStepValue(IS_FLUID)==1.0)
{
in->FastGetSolutionStepValue(PRESSURE)=(mMdiagInv[aa]*history[aa])+bulk_modulus*density*(in->FastGetSolutionStepValue(NODAL_AREA) - in->FastGetSolutionStepValue(NODAL_AREA,1))/(in->FastGetSolutionStepValue(NODAL_AREA));

}
}

}


KRATOS_CATCH("");
}
void UpdatePressures (	TSystemMatrixType& mD,
TSystemMatrixType& mMconsistent, TSystemVectorType& mMdiagInv,ModelPart& r_model_part, double bulk_modulus, double density)
{
KRATOS_TRY


const int size = TSparseSpace::Size(mMdiagInv);
const int size_disp = TDim*TSparseSpace::Size(mMdiagInv);

TSystemVectorType p_n(size);
TSystemVectorType dp(size);
TSystemVectorType p_n1(size);
TSystemVectorType history(size);

TSparseSpace::SetToZero(p_n);
TSparseSpace::SetToZero(dp);
TSparseSpace::SetToZero(p_n1);
TSparseSpace::SetToZero(history);

TSystemMatrixType aux(size,size);
TSystemVectorType temp(size);


TSystemVectorType displ(size_disp);

int i=0;
for (typename NodesArrayType::iterator in=r_model_part.NodesBegin(); in!=r_model_part.NodesEnd(); ++in)
{
if( (in->GetValue(NEIGHBOUR_NODES)).size() != 0 )
{
if (i<size)
p_n[i]=in->FastGetSolutionStepValue(PRESSURE,1);
i++;

}
}
TSparseSpace::Mult(mMconsistent, p_n, history);


for(typename DofsArrayType::iterator i_dof = BaseType::mDofSet.begin() ; i_dof != BaseType::mDofSet.end() ; ++i_dof)

{

displ[i_dof->EquationId()]=i_dof->GetSolutionStepValue()-i_dof->GetSolutionStepValue(1);
}


TSparseSpace::Mult(mD, displ, dp);

dp*=(bulk_modulus*density);






for (int ii=0; ii<size; ii++)
{
p_n1[ii]=mMdiagInv[ii]*(history[ii]+dp[ii]);
}



for (typename NodesArrayType::iterator in=r_model_part.NodesBegin(); in!=r_model_part.NodesEnd(); ++in)
{
if( (in->GetValue(NEIGHBOUR_NODES)).size() != 0 )
{
in->FastGetSolutionStepValue(PRESSURE)=0.0;
}
}

int aa=0;
for (typename NodesArrayType::iterator in=r_model_part.NodesBegin(); in!=r_model_part.NodesEnd(); ++in)
{
if( (in->GetValue(NEIGHBOUR_NODES)).size() != 0 )
{
if (aa<size)
in->FastGetSolutionStepValue(PRESSURE)=p_n1[aa];
aa++;
}
}


KRATOS_CATCH("");
}

/


















}; 









}  

#endif 


