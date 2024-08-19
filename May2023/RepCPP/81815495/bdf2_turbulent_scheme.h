
#if !defined(KRATOS_BDF2_TURBULENT_SCHEME_H_INCLUDED )
#define  KRATOS_BDF2_TURBULENT_SCHEME_H_INCLUDED

#include <string>
#include <iostream>


#include "solving_strategies/schemes/scheme.h"
#include "includes/define.h"
#include "includes/dof.h"
#include "processes/process.h"
#include "containers/pointer_vector_set.h"
#include "utilities/coordinate_transformation_utilities.h"

#include "fluid_dynamics_application_variables.h"


namespace Kratos
{







template<class TSparseSpace,class TDenseSpace>
class BDF2TurbulentScheme : public Scheme<TSparseSpace, TDenseSpace>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(BDF2TurbulentScheme);
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


BDF2TurbulentScheme()
: Scheme<TSparseSpace, TDenseSpace>()
, mrPeriodicIdVar(Kratos::Variable<int>::StaticObject())
{}


BDF2TurbulentScheme(Process::Pointer pTurbulenceModel)
: Scheme<TSparseSpace, TDenseSpace>()
, mpTurbulenceModel(pTurbulenceModel)
, mrPeriodicIdVar(Kratos::Variable<int>::StaticObject())
{}


BDF2TurbulentScheme(const Kratos::Variable<int>& rPeriodicVar)
: Scheme<TSparseSpace, TDenseSpace>()
, mrPeriodicIdVar(rPeriodicVar)
{}


~BDF2TurbulentScheme() override
{}





int Check(ModelPart& rModelPart) override
{
KRATOS_TRY

int error_code = BaseType::Check(rModelPart);
if (error_code != 0) {
return error_code;
}

KRATOS_ERROR_IF(rModelPart.GetBufferSize() < 3)
<< "Insufficient buffer size for BDF2, should be at least 3, got " << rModelPart.GetBufferSize() << std::endl;

return 0;

KRATOS_CATCH("");
}

void Initialize(ModelPart& rModelPart) override
{
const auto& r_proces_info = rModelPart.GetProcessInfo();
const unsigned int domain_size = r_proces_info[DOMAIN_SIZE];
auto p_aux = Kratos::make_unique<RotationToolType>(domain_size, domain_size + 1, SLIP);
mpRotationTool.swap(p_aux);

BaseType::Initialize(rModelPart);
}

void InitializeSolutionStep(
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
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

void InitializeNonLinIteration(
ModelPart& rModelPart,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY

if (mpTurbulenceModel != 0) mpTurbulenceModel->Execute();

KRATOS_CATCH("")
}

void FinalizeNonLinIteration(
ModelPart &rModelPart,
TSystemMatrixType &A,
TSystemVectorType &Dx,
TSystemVectorType &b) override
{
const ProcessInfo& CurrentProcessInfo = rModelPart.GetProcessInfo();

if (CurrentProcessInfo[OSS_SWITCH] == 1.0)
{
this->LumpedProjection(rModelPart);
}

}

void Predict(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY

const int n_nodes = rModelPart.NumberOfNodes();
const Vector& BDFcoefs = rModelPart.GetProcessInfo()[BDF_COEFFICIENTS];

#pragma omp parallel for
for(int i_node = 0; i_node < n_nodes; ++i_node) {
auto it_node = rModelPart.NodesBegin() + i_node;
auto& rVel0 = it_node->FastGetSolutionStepValue(VELOCITY);
const auto& rVel1 = it_node->FastGetSolutionStepValue(VELOCITY,1);
const auto& rVel2 = it_node->FastGetSolutionStepValue(VELOCITY,2);
auto& rAcceleration = it_node->FastGetSolutionStepValue(ACCELERATION);

if(!it_node->IsFixed(VELOCITY_X))
rVel0[0] = 2.00 * rVel1[0] - rVel2[0];
if(!it_node->IsFixed(VELOCITY_Y))
rVel0[1] = 2.00 * rVel1[1] - rVel2[1];
if(!it_node->IsFixed(VELOCITY_Z))
rVel0[2] = 2.00 * rVel1[2] - rVel2[2];

rAcceleration = BDFcoefs[0] * rVel0 + BDFcoefs[1] * rVel1 + BDFcoefs[2] * rVel2;
}

KRATOS_CATCH("")
}


void Update(
ModelPart& rModelPart,
DofsArrayType& rDofSet,
TSystemMatrixType& A,
TSystemVectorType& Dx,
TSystemVectorType& b) override
{
KRATOS_TRY

mpRotationTool->RotateVelocities(rModelPart);

mpDofUpdater->UpdateDofs(rDofSet,Dx);

mpRotationTool->RecoverVelocities(rModelPart);

const Vector& BDFCoefs = rModelPart.GetProcessInfo()[BDF_COEFFICIENTS];
this->UpdateAcceleration(rModelPart,BDFCoefs);

KRATOS_CATCH("")
}

void CalculateSystemContributions(
Element& rCurrentElement,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& rEquationId,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_TRY

LocalSystemMatrixType Mass;
LocalSystemMatrixType Damp;

rCurrentElement.EquationIdVector(rEquationId,rCurrentProcessInfo);

rCurrentElement.CalculateLocalSystem(LHS_Contribution,RHS_Contribution,rCurrentProcessInfo);
rCurrentElement.CalculateMassMatrix(Mass,rCurrentProcessInfo);
rCurrentElement.CalculateLocalVelocityContribution(Damp,RHS_Contribution,rCurrentProcessInfo);

this->CombineLHSContributions(LHS_Contribution,Mass,Damp,rCurrentProcessInfo);
this->AddDynamicRHSContribution<Kratos::Element>(rCurrentElement,RHS_Contribution,Mass,rCurrentProcessInfo);

mpRotationTool->Rotate(LHS_Contribution, RHS_Contribution, rCurrentElement.GetGeometry());
mpRotationTool->ApplySlipCondition(LHS_Contribution, RHS_Contribution, rCurrentElement.GetGeometry());

KRATOS_CATCH("")
}

void CalculateRHSContribution(
Element& rCurrentElement,
LocalSystemVectorType &RHS_Contribution,
Element::EquationIdVectorType &rEquationId,
const ProcessInfo &rCurrentProcessInfo) override
{
KRATOS_TRY

LocalSystemMatrixType Mass;
LocalSystemMatrixType Damp;

rCurrentElement.EquationIdVector(rEquationId,rCurrentProcessInfo);

rCurrentElement.CalculateRightHandSide(RHS_Contribution,rCurrentProcessInfo);
rCurrentElement.CalculateMassMatrix(Mass,rCurrentProcessInfo);
rCurrentElement.CalculateLocalVelocityContribution(Damp,RHS_Contribution,rCurrentProcessInfo);

this->AddDynamicRHSContribution<Kratos::Element>(rCurrentElement,RHS_Contribution,Mass,rCurrentProcessInfo);

mpRotationTool->Rotate(RHS_Contribution, rCurrentElement.GetGeometry());
mpRotationTool->ApplySlipCondition(RHS_Contribution, rCurrentElement.GetGeometry());

KRATOS_CATCH("")
}

void CalculateSystemContributions(
Condition& rCurrentCondition,
LocalSystemMatrixType& LHS_Contribution,
LocalSystemVectorType& RHS_Contribution,
Element::EquationIdVectorType& rEquationId,
const ProcessInfo& rCurrentProcessInfo) override
{
KRATOS_TRY

LocalSystemMatrixType Mass;
LocalSystemMatrixType Damp;

rCurrentCondition.EquationIdVector(rEquationId,rCurrentProcessInfo);

rCurrentCondition.CalculateLocalSystem(LHS_Contribution,RHS_Contribution,rCurrentProcessInfo);
rCurrentCondition.CalculateMassMatrix(Mass,rCurrentProcessInfo);
rCurrentCondition.CalculateLocalVelocityContribution(Damp,RHS_Contribution,rCurrentProcessInfo);

this->CombineLHSContributions(LHS_Contribution,Mass,Damp,rCurrentProcessInfo);
this->AddDynamicRHSContribution<Kratos::Condition>(rCurrentCondition,RHS_Contribution,Mass,rCurrentProcessInfo);

mpRotationTool->Rotate(LHS_Contribution, RHS_Contribution, rCurrentCondition.GetGeometry());
mpRotationTool->ApplySlipCondition(LHS_Contribution, RHS_Contribution, rCurrentCondition.GetGeometry());

KRATOS_CATCH("")
}

void CalculateRHSContribution(
Condition &rCurrentCondition,
LocalSystemVectorType &RHS_Contribution,
Element::EquationIdVectorType &rEquationId,
const ProcessInfo &rCurrentProcessInfo) override
{
KRATOS_TRY

LocalSystemMatrixType Mass;
LocalSystemMatrixType Damp;

rCurrentCondition.EquationIdVector(rEquationId,rCurrentProcessInfo);

rCurrentCondition.CalculateRightHandSide(RHS_Contribution,rCurrentProcessInfo);
rCurrentCondition.CalculateMassMatrix(Mass,rCurrentProcessInfo);
rCurrentCondition.CalculateLocalVelocityContribution(Damp,RHS_Contribution,rCurrentProcessInfo);

this->AddDynamicRHSContribution<Kratos::Condition>(rCurrentCondition,RHS_Contribution,Mass,rCurrentProcessInfo);

mpRotationTool->Rotate(RHS_Contribution, rCurrentCondition.GetGeometry());
mpRotationTool->ApplySlipCondition(RHS_Contribution, rCurrentCondition.GetGeometry());

KRATOS_CATCH("")
}

void Clear() override
{
this->mpDofUpdater->Clear();
}






std::string Info() const override
{
std::stringstream buffer;
buffer << "BDF2TurbulentScheme";
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override
{}



protected:








void SetTimeCoefficients(ProcessInfo& rCurrentProcessInfo)
{
KRATOS_TRY;

double Dt = rCurrentProcessInfo[DELTA_TIME];
double OldDt = rCurrentProcessInfo.GetPreviousTimeStepInfo(1)[DELTA_TIME];

double Rho = OldDt / Dt;
double TimeCoeff = 1.0 / (Dt * Rho * Rho + Dt * Rho);

Vector& BDFcoeffs = rCurrentProcessInfo[BDF_COEFFICIENTS];
BDFcoeffs.resize(3, false);

BDFcoeffs[0] = TimeCoeff * (Rho * Rho + 2.0 * Rho); 
BDFcoeffs[1] = -TimeCoeff * (Rho * Rho + 2.0 * Rho + 1.0); 
BDFcoeffs[2] = TimeCoeff; 

KRATOS_CATCH("");
}


virtual void UpdateDofs(
DofsArrayType& rDofSet,
TSystemVectorType& Dx)
{
KRATOS_TRY

const int n_dof = rDofSet.size();

#pragma omp parallel for
for (int i_dof = 0; i_dof < n_dof; ++i_dof) {
auto it_dof = rDofSet.begin() + i_dof;
if (it_dof->IsFree()) {
it_dof->GetSolutionStepValue() += TSparseSpace::GetValue(Dx, it_dof->EquationId());
}
}

KRATOS_CATCH("")
}


void UpdateAcceleration(
ModelPart& rModelPart,
const Vector& rBDFcoefs)
{
KRATOS_TRY

const double Coef0 = rBDFcoefs[0];
const double Coef1 = rBDFcoefs[1];
const double Coef2 = rBDFcoefs[2];
const int n_nodes = rModelPart.NumberOfNodes();

#pragma omp parallel for
for (int i_node = 0; i_node < n_nodes; ++i_node) {
auto it_node = rModelPart.NodesBegin() + i_node;
const auto& rVel0 = it_node->FastGetSolutionStepValue(VELOCITY);
const auto& rVel1 = it_node->FastGetSolutionStepValue(VELOCITY,1);
const auto& rVel2 = it_node->FastGetSolutionStepValue(VELOCITY,2);
auto& rAcceleration = it_node->FastGetSolutionStepValue(ACCELERATION);

rAcceleration = Coef0 * rVel0 + Coef1 * rVel1 + Coef2 * rVel2;
}

KRATOS_CATCH("")
}

void CombineLHSContributions(
LocalSystemMatrixType& rLHS,
LocalSystemMatrixType& rMass,
LocalSystemMatrixType& rDamp,
const ProcessInfo& rCurrentProcessInfo)
{
const double Coef0 = rCurrentProcessInfo.GetValue(BDF_COEFFICIENTS)[0];
if (rMass.size1() != 0) noalias(rLHS) += Coef0 * rMass;
if (rDamp.size1() != 0) noalias(rLHS) += rDamp;
}

template<class TObject>
void AddDynamicRHSContribution(
TObject& rObject,
LocalSystemVectorType& rRHS,
LocalSystemMatrixType& rMass,
const ProcessInfo& rCurrentProcessInfo)
{
if (rMass.size1() != 0)
{
const Vector& rCoefs = rCurrentProcessInfo.GetValue(BDF_COEFFICIENTS);
const auto& r_const_obj_ref = rObject;
LocalSystemVectorType Acc;
r_const_obj_ref.GetFirstDerivativesVector(Acc);
Acc *= rCoefs[0];

for(unsigned int n = 1; n < 3; ++n)
{
LocalSystemVectorType rVel;
r_const_obj_ref.GetFirstDerivativesVector(rVel,n);
noalias(Acc) += rCoefs[n] * rVel;
}

noalias(rRHS) -= prod(rMass,Acc);
}
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

const double RelTol = 1e-4 * rModelPart.NumberOfNodes();
const double AbsTol = 1e-6 * rModelPart.NumberOfNodes();
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

RelMomErr += sqrt( dMomProj[0]*dMomProj[0] + dMomProj[1]*dMomProj[1] + dMomProj[2]*dMomProj[2]);
RelMassErr += fabs(dMassProj);

auto& rMomRHS = ind->FastGetSolutionStepValue(ADVPROJ);
double& rMassRHS = ind->FastGetSolutionStepValue(DIVPROJ);
rMomRHS += dMomProj;
rMassRHS += dMassProj;

AbsMomErr += sqrt( rMomRHS[0]*rMomRHS[0] + rMomRHS[1]*rMomRHS[1] + rMomRHS[2]*rMomRHS[2]);
AbsMassErr += fabs(rMassRHS);
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

KRATOS_INFO("BDF2TurbulentScheme") << "Performed OSS Projection in " << iter << " iterations" << std::endl;
}

void LumpedProjection(ModelPart& rModelPart)
{
const int n_nodes = rModelPart.NumberOfNodes();
const int n_elems = rModelPart.NumberOfElements();
const ProcessInfo& rCurrentProcessInfo = rModelPart.GetProcessInfo();

const array_1d<double,3> zero_vect = ZeroVector(3);
#pragma omp parallel for firstprivate(zero_vect)
for (int i_node = 0; i_node < n_nodes; ++i_node) {
auto itNode = rModelPart.NodesBegin() + i_node;
noalias(itNode->FastGetSolutionStepValue(ADVPROJ)) = zero_vect;
itNode->FastGetSolutionStepValue(DIVPROJ) = 0.0;
itNode->FastGetSolutionStepValue(NODAL_AREA) = 0.0;
}

array_1d<double, 3 > Out;
#pragma omp parallel for private(Out)
for (int i_elem = 0; i_elem < n_elems; ++i_elem) {
auto itElem = rModelPart.ElementsBegin() + i_elem;
itElem->Calculate(ADVPROJ, Out, rCurrentProcessInfo);
}

rModelPart.GetCommunicator().AssembleCurrentData(NODAL_AREA);
rModelPart.GetCommunicator().AssembleCurrentData(DIVPROJ);
rModelPart.GetCommunicator().AssembleCurrentData(ADVPROJ);

if (mrPeriodicIdVar.Key() != 0) {
this->PeriodicConditionProjectionCorrection(rModelPart);
}

const double zero_tol = 1.0e-12;
#pragma omp parallel for firstprivate(zero_tol)
for (int i_node = 0; i_node < n_nodes; ++i_node){
auto iNode = rModelPart.NodesBegin() + i_node;
if (iNode->FastGetSolutionStepValue(NODAL_AREA) < zero_tol) {
iNode->FastGetSolutionStepValue(NODAL_AREA) = 1.0;
}
const double Area = iNode->FastGetSolutionStepValue(NODAL_AREA);
iNode->FastGetSolutionStepValue(ADVPROJ) /= Area;
iNode->FastGetSolutionStepValue(DIVPROJ) /= Area;
}

KRATOS_INFO("BDF2TurbulentScheme") << "Computing OSS projections" << std::endl;
}


void PeriodicConditionProjectionCorrection(ModelPart& rModelPart)
{
const int num_nodes = rModelPart.NumberOfNodes();
const int num_conditions = rModelPart.NumberOfConditions();

#pragma omp parallel for
for (int i = 0; i < num_nodes; i++) {
auto it_node = rModelPart.NodesBegin() + i;

it_node->SetValue(NODAL_AREA,0.0);
it_node->SetValue(ADVPROJ,ZeroVector(3));
it_node->SetValue(DIVPROJ,0.0);
}

#pragma omp parallel for
for (int i = 0; i < num_conditions; i++) {
auto it_cond = rModelPart.ConditionsBegin() + i;

if(it_cond->Is(PERIODIC)) {
this->AssemblePeriodicContributionToProjections(it_cond->GetGeometry());
}
}

rModelPart.GetCommunicator().AssembleNonHistoricalData(NODAL_AREA);
rModelPart.GetCommunicator().AssembleNonHistoricalData(ADVPROJ);
rModelPart.GetCommunicator().AssembleNonHistoricalData(DIVPROJ);

#pragma omp parallel for
for (int i = 0; i < num_nodes; i++) {
auto it_node = rModelPart.NodesBegin() + i;
this->CorrectContributionsOnPeriodicNode(*it_node);
}
}

void AssemblePeriodicContributionToProjections(Geometry< Node >& rGeometry)
{
unsigned int nodes_in_cond = rGeometry.PointsNumber();

double nodal_area = 0.0;
array_1d<double,3> momentum_projection = ZeroVector(3);
double mass_projection = 0.0;
for ( unsigned int i = 0; i < nodes_in_cond; i++ )
{
auto& r_node = rGeometry[i];
nodal_area += r_node.FastGetSolutionStepValue(NODAL_AREA);
noalias(momentum_projection) += r_node.FastGetSolutionStepValue(ADVPROJ);
mass_projection += r_node.FastGetSolutionStepValue(DIVPROJ);
}

for ( unsigned int i = 0; i < nodes_in_cond; i++ )
{
auto& r_node = rGeometry[i];

r_node.SetLock();
r_node.GetValue(NODAL_AREA) = nodal_area;
noalias(r_node.GetValue(ADVPROJ)) = momentum_projection;
r_node.GetValue(DIVPROJ) = mass_projection;
r_node.UnSetLock();
}
}

void CorrectContributionsOnPeriodicNode(Node& rNode)
{
if (rNode.GetValue(NODAL_AREA) != 0.0) 
{
rNode.FastGetSolutionStepValue(NODAL_AREA) = rNode.GetValue(NODAL_AREA);
noalias(rNode.FastGetSolutionStepValue(ADVPROJ)) = rNode.GetValue(ADVPROJ);
rNode.FastGetSolutionStepValue(DIVPROJ) = rNode.GetValue(DIVPROJ);
}
}







private:



Process::Pointer mpTurbulenceModel = nullptr;

RotationToolPointerType mpRotationTool = nullptr;

typename TSparseSpace::DofUpdaterPointerType mpDofUpdater = TSparseSpace::CreateDofUpdater();

const Kratos::Variable<int>& mrPeriodicIdVar;











BDF2TurbulentScheme & operator=(BDF2TurbulentScheme const& rOther)
{}

BDF2TurbulentScheme(BDF2TurbulentScheme const& rOther)
{}


}; 





template<class TSparseSpace,class TDenseSpace>
inline std::istream& operator >>(std::istream& rIStream,BDF2TurbulentScheme<TSparseSpace,TDenseSpace>& rThis)
{
return rIStream;
}

template<class TSparseSpace,class TDenseSpace>
inline std::ostream& operator <<(std::ostream& rOStream,const BDF2TurbulentScheme<TSparseSpace,TDenseSpace>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}



} 

#endif 
