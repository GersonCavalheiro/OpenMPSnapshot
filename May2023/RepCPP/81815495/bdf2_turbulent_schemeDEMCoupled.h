
#if !defined(KRATOS_BDF2_TURBULENT_SCHEME_DEM_COUPLED_H_INCLUDED )
#define  KRATOS_BDF2_TURBULENT_SCHEME_DEM_COUPLED_H_INCLUDED

#include <string>
#include <iostream>


#include "solving_strategies/schemes/scheme.h"
#include "includes/define.h"
#include "includes/dof.h"
#include "processes/process.h"
#include "containers/pointer_vector_set.h"
#include "utilities/coordinate_transformation_utilities.h"
#include "utilities/parallel_utilities.h"

#include "fluid_dynamics_application_variables.h"
#include "custom_strategies/schemes/bdf2_turbulent_scheme.h"
#include "swimming_dem_application_variables.h"


namespace Kratos
{







template<class TSparseSpace,class TDenseSpace>
class BDF2TurbulentSchemeDEMCoupled : public BDF2TurbulentScheme<TSparseSpace, TDenseSpace>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(BDF2TurbulentSchemeDEMCoupled);
typedef Scheme<TSparseSpace,TDenseSpace> BaseType;
typedef typename TSparseSpace::DataType TDataType;
typedef typename TSparseSpace::MatrixType TSystemMatrixType;
typedef typename TSparseSpace::VectorType TSystemVectorType;

typedef typename TDenseSpace::MatrixType LocalSystemMatrixType;
typedef typename TDenseSpace::VectorType LocalSystemVectorType;

typedef Dof<TDataType> TDofType;
typedef typename BaseType::DofsArrayType DofsArrayType;

typedef CoordinateTransformationUtils<LocalSystemMatrixType, LocalSystemVectorType, double> RotationToolType;
typedef typename RotationToolType::UniquePointer RotationToolPointerType;


BDF2TurbulentSchemeDEMCoupled()
: BDF2TurbulentScheme<TSparseSpace, TDenseSpace>()
, mrPeriodicIdVar(Kratos::Variable<int>::StaticObject())
{}


BDF2TurbulentSchemeDEMCoupled(Process::Pointer pTurbulenceModel)
: BDF2TurbulentScheme<TSparseSpace, TDenseSpace>()
, mpTurbulenceModel(pTurbulenceModel)
, mrPeriodicIdVar(Kratos::Variable<int>::StaticObject())
{}


BDF2TurbulentSchemeDEMCoupled(const Kratos::Variable<int>& rPeriodicVar)
: BDF2TurbulentScheme<TSparseSpace, TDenseSpace>()
, mrPeriodicIdVar(rPeriodicVar)
{}


~BDF2TurbulentSchemeDEMCoupled() override
{}




void InitializeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
ProcessInfo CurrentProcessInfo = rModelPart.GetProcessInfo();

this->SetTimeCoefficients(rModelPart.GetProcessInfo());

BaseType::InitializeSolutionStep(rModelPart,A,Dx,b);

const double tol = 1.0e-12;
const double Dt = rModelPart.GetProcessInfo()[DELTA_TIME];
const double OldDt = rModelPart.GetProcessInfo().GetPreviousSolutionStepInfo(1)[DELTA_TIME];
if(std::abs(Dt - OldDt) > tol) {
const int n_nodes = rModelPart.NumberOfNodes();
const Vector& BDFcoefs = rModelPart.GetProcessInfo()[BDF_COEFFICIENTS];

#pragma omp parallel for
for(int i_node = 0; i_node < n_nodes; ++i_node) {
auto it_node = rModelPart.NodesBegin() + i_node;
auto& rMeshVel = it_node->FastGetSolutionStepValue(MESH_VELOCITY);
const auto& rDisp0 = it_node->FastGetSolutionStepValue(DISPLACEMENT);
const auto& rDisp1 = it_node->FastGetSolutionStepValue(DISPLACEMENT,1);
const auto& rDisp2 = it_node->FastGetSolutionStepValue(DISPLACEMENT,2);
rMeshVel = BDFcoefs[0] * rDisp0 + BDFcoefs[1] * rDisp1 + BDFcoefs[2] * rDisp2;
}
}
}

void SetTimeCoefficients(ProcessInfo& rCurrentProcessInfo)
{
KRATOS_TRY;

double OldDt;
double Dt = rCurrentProcessInfo[DELTA_TIME];
double step = rCurrentProcessInfo[STEP];
if (rCurrentProcessInfo[MANUFACTURED] && step < 2){
OldDt = rCurrentProcessInfo[DELTA_TIME];
}
else {
OldDt = rCurrentProcessInfo.GetPreviousTimeStepInfo(1)[DELTA_TIME];
}

double Rho = OldDt / Dt;
double TimeCoeff = 1.0 / (Dt * Rho * Rho + Dt * Rho);

Vector& BDFcoeffs = rCurrentProcessInfo[BDF_COEFFICIENTS];
BDFcoeffs.resize(3, false);

BDFcoeffs[0] = TimeCoeff * (Rho * Rho + 2.0 * Rho); 
BDFcoeffs[1] = -TimeCoeff * (Rho * Rho + 2.0 * Rho + 1.0); 
BDFcoeffs[2] = TimeCoeff; 

KRATOS_CATCH("");
}

void FullProjection(ModelPart& rModelPart)
{
const ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();

const int n_nodes = rModelPart.NumberOfNodes();
const int n_elems = rModelPart.NumberOfElements();
const array_1d<double,3> zero_vect = ZeroVector(3);
#pragma omp parallel for firstprivate(zero_vect)
for (int i_node = 0; i_node < n_nodes; ++i_node) {
auto ind = rModelPart.NodesBegin() + i_node;
noalias(ind->FastGetSolutionStepValue(ADVPROJ)) = zero_vect; 
ind->FastGetSolutionStepValue(DIVPROJ) = 0.0; 
ind->FastGetSolutionStepValue(NODAL_AREA) = 0.0; 
}

const double RelTol = rModelPart.GetProcessInfo()[RELAXATION_ALPHA] * 1e-4 * rModelPart.NumberOfNodes();
const double AbsTol = rModelPart.GetProcessInfo()[RELAXATION_ALPHA] * 1e-6 * rModelPart.NumberOfNodes();
const unsigned int MaxIter = 100;

unsigned int iter = 0;
array_1d<double,3> dMomProj = zero_vect;
double dMassProj = 0.0;

double RelMomErr = 1000.0 * RelTol;
double RelMassErr = 1000.0 * RelTol;
double AbsMomErr = 1000.0 * AbsTol;
double AbsMassErr = 1000.0 * AbsTol;

while( ( (AbsMomErr > AbsTol && RelMomErr > RelTol) || (AbsMassErr > AbsTol && RelMassErr > RelTol) ) && iter < MaxIter)
{
#pragma omp parallel for firstprivate(zero_vect)
for (int i_node = 0; i_node < n_nodes; ++i_node)
{
auto ind = rModelPart.NodesBegin() + i_node;
noalias(ind->GetValue(ADVPROJ)) = zero_vect; 
ind->GetValue(DIVPROJ) = 0.0; 
ind->FastGetSolutionStepValue(NODAL_AREA) = 0.0; 
}

RelMomErr = 0.0;
RelMassErr = 0.0;
AbsMomErr = 0.0;
AbsMassErr = 0.0;

array_1d<double, 3 > output;
#pragma omp parallel for private(output)
for (int i_elem = 0; i_elem < n_elems; ++i_elem) {
auto it_elem = rModelPart.ElementsBegin() + i_elem;
it_elem->Calculate(SUBSCALE_VELOCITY, output, rCurrentProcessInfo);
}

rModelPart.GetCommunicator().AssembleCurrentData(NODAL_AREA);
rModelPart.GetCommunicator().AssembleCurrentData(DIVPROJ);
rModelPart.GetCommunicator().AssembleCurrentData(ADVPROJ);
rModelPart.GetCommunicator().AssembleNonHistoricalData(DIVPROJ);
rModelPart.GetCommunicator().AssembleNonHistoricalData(ADVPROJ);

#pragma omp parallel for
for (int i_node = 0; i_node < n_nodes; ++i_node) {
auto ind = rModelPart.NodesBegin() + i_node;
const double Area = ind->FastGetSolutionStepValue(NODAL_AREA); 
dMomProj = ind->GetValue(ADVPROJ) / Area;
dMassProj = ind->GetValue(DIVPROJ) / Area;

RelMomErr += std::sqrt(std::pow(dMomProj[0],2) + std::pow(dMomProj[1],2) + std::pow(dMomProj[2],2));
RelMassErr += std::fabs(dMassProj);

auto& rMomRHS = ind->FastGetSolutionStepValue(ADVPROJ);
double& rMassRHS = ind->FastGetSolutionStepValue(DIVPROJ);
rMomRHS += dMomProj;
rMassRHS += dMassProj;

AbsMomErr += std::sqrt(std::pow(rMomRHS[0],2) + std::pow(rMomRHS[1],2) + std::pow(rMomRHS[2],2));
AbsMassErr += std::fabs(rMassRHS);
}

if(AbsMomErr > 1e-10)
RelMomErr /= AbsMomErr;
else 
RelMomErr = 1000.0;

if(AbsMassErr > 1e-10)
RelMassErr /= AbsMassErr;
else
RelMassErr = 1000.0;

iter++;
}

KRATOS_INFO("BDF2TurbulentSchemeDEMCoupled") << "Performed OSS Projection in " << iter << " iterations" << std::endl;
}

void InitializeNonLinIteration(
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY

if (mpTurbulenceModel != 0) mpTurbulenceModel->Execute();

const ProcessInfo& CurrentProcessInfo = rModelPart.GetProcessInfo();

if (CurrentProcessInfo[OSS_SWITCH] == 1.0)
{
this->FullProjection(rModelPart);
}

KRATOS_CATCH("")
}

void FinalizeNonLinIteration(
ModelPart &rModelPart,
TSystemMatrixType &A,
TSystemVectorType &Dx,
TSystemVectorType &b) override
{

BaseType::FinalizeNonLinIteration(rModelPart, A, Dx, b);

}

void UpdateFluidFraction(
ModelPart& r_model_part,
ProcessInfo& r_current_process_info)
{
BDF2TurbulentScheme<TSparseSpace, TDenseSpace>::SetTimeCoefficients(r_current_process_info);
const Vector& BDFcoefs = r_current_process_info[BDF_COEFFICIENTS];
double step = r_current_process_info[STEP];

block_for_each(r_model_part.Nodes(), [&](Node& rNode)
{
double& fluid_fraction_0 = rNode.FastGetSolutionStepValue(FLUID_FRACTION);
double& fluid_fraction_1 = rNode.FastGetSolutionStepValue(FLUID_FRACTION_OLD);
double& fluid_fraction_2 = rNode.FastGetSolutionStepValue(FLUID_FRACTION_OLD_2);

if (step <= 2){
fluid_fraction_2 = fluid_fraction_0;
fluid_fraction_1 = fluid_fraction_0;
}

rNode.FastGetSolutionStepValue(FLUID_FRACTION_RATE) = BDFcoefs[0] * fluid_fraction_0 + BDFcoefs[1] * fluid_fraction_1 + BDFcoefs[2] * fluid_fraction_2;

rNode.GetSolutionStepValue(FLUID_FRACTION_OLD_2) = rNode.GetSolutionStepValue(FLUID_FRACTION_OLD);
rNode.GetSolutionStepValue(FLUID_FRACTION_OLD) = rNode.GetSolutionStepValue(FLUID_FRACTION);
});
}

void FinalizeSolutionStep(
ModelPart& r_model_part,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY
BDF2TurbulentScheme<TSparseSpace, TDenseSpace>::FinalizeSolutionStep(r_model_part, A, Dx, b);
KRATOS_CATCH("")
}

void Clear() override
{
this->mpDofUpdater->Clear();
}






std::string Info() const override
{
std::stringstream buffer;
buffer << "BDF2TurbulentSchemeDEMCoupled";
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override
{}



protected:













private:



Process::Pointer mpTurbulenceModel = nullptr;

RotationToolPointerType mpRotationTool = nullptr;

typename TSparseSpace::DofUpdaterPointerType mpDofUpdater = TSparseSpace::CreateDofUpdater();

const Kratos::Variable<int>& mrPeriodicIdVar;











BDF2TurbulentSchemeDEMCoupled & operator=(BDF2TurbulentSchemeDEMCoupled const& rOther)
{}

BDF2TurbulentSchemeDEMCoupled(BDF2TurbulentSchemeDEMCoupled const& rOther)
{}


}; 





template<class TSparseSpace,class TDenseSpace>
inline std::istream& operator >>(std::istream& rIStream,BDF2TurbulentSchemeDEMCoupled<TSparseSpace,TDenseSpace>& rThis)
{
return rIStream;
}

template<class TSparseSpace,class TDenseSpace>
inline std::ostream& operator <<(std::ostream& rOStream,const BDF2TurbulentSchemeDEMCoupled<TSparseSpace,TDenseSpace>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}



} 

#endif 
