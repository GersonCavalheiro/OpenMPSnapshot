
#pragma once

#include <random>


#include "solving_strategies/convergence_accelerators/convergence_accelerator.h"
#include "utilities/dense_qr_decomposition.h"
#include "utilities/dense_svd_decomposition.h"
#include "utilities/parallel_utilities.h"

#include "mvqn_convergence_accelerator.hpp"
#include "ibqn_mvqn_randomized_svd_convergence_accelerator.h"

namespace Kratos
{











template<class TSparseSpace, class TDenseSpace>
class MVQNRandomizedSVDConvergenceAccelerator: public MVQNFullJacobianConvergenceAccelerator<TSparseSpace, TDenseSpace>
{
public:

KRATOS_CLASS_POINTER_DEFINITION( MVQNRandomizedSVDConvergenceAccelerator );

typedef std::size_t SizeType;

typedef MVQNFullJacobianConvergenceAccelerator<TSparseSpace, TDenseSpace> BaseType;
typedef typename BaseType::Pointer BaseTypePointer;

typedef typename BaseType::DenseVectorType VectorType;
typedef typename BaseType::DenseVectorPointerType VectorPointerType;

typedef typename BaseType::DenseMatrixType MatrixType;
typedef typename BaseType::DenseMatrixPointerType MatrixPointerType;

typedef typename DenseQRDecomposition<TDenseSpace>::Pointer DenseQRPointerType;
typedef typename DenseSingularValueDecomposition<TDenseSpace>::Pointer DenseSVDPointerType;



explicit MVQNRandomizedSVDConvergenceAccelerator(
DenseQRPointerType pDenseQR,
DenseSVDPointerType pDenseSVD,
Parameters ConvAcceleratorParameters)
: BaseType()
, mpDenseQR(pDenseQR)
, mpDenseSVD(pDenseSVD)
{
ConvAcceleratorParameters.ValidateAndAssignDefaults(GetDefaultParameters());
mUserNumberOfModes = ConvAcceleratorParameters["jacobian_modes"].GetInt();
mAutomaticJacobianModes = ConvAcceleratorParameters["automatic_jacobian_modes"].GetBool();
mLimitModesToIterations = ConvAcceleratorParameters["limit_modes_to_iterations"].GetBool();
mMinRandSVDExtraModes = ConvAcceleratorParameters["min_rand_svd_extra_modes"].GetInt();
BaseType::SetInitialRelaxationOmega(ConvAcceleratorParameters["w_0"].GetDouble());
BaseType::SetCutOffTolerance(ConvAcceleratorParameters["cut_off_tol"].GetDouble());

KRATOS_WARNING_IF("MVQNRandomizedSVDConvergenceAccelerator", mAutomaticJacobianModes && mUserNumberOfModes != 0)
<< "Custom and automatic number of modes have been selected. Automatic will be used." << std::endl;
}



MVQNRandomizedSVDConvergenceAccelerator(const MVQNRandomizedSVDConvergenceAccelerator& rOther) = default;


virtual ~MVQNRandomizedSVDConvergenceAccelerator(){}




void FinalizeSolutionStep() override
{
KRATOS_TRY;

if (!BaseType::IsUsedInBlockNewtonEquations() && BaseType::IsFirstCorrectionPerformed()) {
CalculateInverseJacobianRandomizedSVD();
}

if (mpJacQU != nullptr && mpJacSigmaV != nullptr) {
mpOldJacQU = mpJacQU;
mpOldJacSigmaV = mpJacSigmaV;
std::size_t n_obs = BaseType::GetNumberOfObservations();
if (n_obs > mOldMaxRank) {
mOldMaxRank = n_obs;
}
}

BaseType::FinalizeSolutionStep();

mpJacQU = nullptr;
mpJacSigmaV = nullptr;

mSeed++;

KRATOS_CATCH( "" );
}

Parameters GetDefaultParameters() const override
{
Parameters mvqn_randomized_svd_default_parameters(R"({
"solver_type"               : "MVQN_randomized_SVD",
"automatic_jacobian_modes"  : true,
"jacobian_modes"            : 0,
"w_0"                       : 0.825,
"cut_off_tol"               : 1e-8,
"interface_block_newton"    : false,
"limit_modes_to_iterations" : true,
"min_rand_svd_extra_modes"  : 1
})");

return mvqn_randomized_svd_default_parameters;
}








friend class IBQNMVQNRandomizedSVDConvergenceAccelerator<TSparseSpace, TDenseSpace>;

protected:


explicit MVQNRandomizedSVDConvergenceAccelerator(
DenseQRPointerType pDenseQR,
DenseSVDPointerType pDenseSVD,
const bool AutomaticJacobianModes,
const unsigned int JacobianModes,
const double CutOffTolerance,
const bool LimitModesToIterations,
const unsigned int MinRandSVDExtraModes)
: BaseType(CutOffTolerance)
, mUserNumberOfModes(JacobianModes)
, mMinRandSVDExtraModes(MinRandSVDExtraModes)
, mAutomaticJacobianModes(AutomaticJacobianModes)
, mLimitModesToIterations(LimitModesToIterations)
, mpDenseQR(pDenseQR)
, mpDenseSVD(pDenseSVD)
{
KRATOS_WARNING_IF("MVQNRandomizedSVDConvergenceAccelerator", mAutomaticJacobianModes && mUserNumberOfModes != 0)
<< "Custom and automatic number of modes have been selected. Automatic will be used." << std::endl;
}


void UpdateInverseJacobianApproximation(
const VectorType& rResidualVector,
const VectorType& rIterationGuess) override
{
BaseType::AppendCurrentIterationInformation(rResidualVector, rIterationGuess);

if (BaseType::IsUsedInBlockNewtonEquations() && BaseType::GetConvergenceAcceleratorIteration() != 0) {
CalculateInverseJacobianRandomizedSVD();
}
}

void CalculateCorrectionWithJacobian(VectorType& rCorrection) override
{
const SizeType n_dofs = BaseType::GetProblemSize();
const auto p_V = BaseType::pGetResidualObservationMatrix();
const auto p_W = BaseType::pGetSolutionObservationMatrix();
auto& r_res_vect = *(BaseType::pGetCurrentIterationResidualVector());

if (p_V != nullptr && p_W != nullptr) {
const auto& r_V = *p_V;
const auto& r_W = *p_W;

MatrixType aux_M;
CalculateAuxiliaryMatrixM(aux_M);
VectorType M_res = prod(aux_M, r_res_vect);
VectorType V_M_res = prod(r_V, M_res);
VectorType W_M_res = prod(r_W, M_res);

noalias(rCorrection) = ZeroVector(n_dofs);
TDenseSpace::UnaliasedAdd(rCorrection, 1.0, V_M_res);
TDenseSpace::UnaliasedAdd(rCorrection, 1.0, W_M_res);
TDenseSpace::UnaliasedAdd(rCorrection, -1.0, r_res_vect);

if (mpOldJacQU != nullptr && mpOldJacSigmaV != nullptr) {
const auto& r_A = *mpOldJacQU;
const auto& r_B = *mpOldJacSigmaV;
VectorType B_res = prod(r_B, r_res_vect);
VectorType A_B_res = prod(r_A, B_res);
VectorType B_V_M_res = prod(r_B, V_M_res);
VectorType A_B_V_M_res = prod(r_A, B_V_M_res);
TDenseSpace::UnaliasedAdd(rCorrection, 1.0, A_B_res);
TDenseSpace::UnaliasedAdd(rCorrection, -1.0, A_B_V_M_res);
}
} else {
if (mpOldJacQU != nullptr && mpOldJacSigmaV != nullptr) {
const auto& r_A = *mpOldJacQU;
const auto& r_B = *mpOldJacSigmaV;
const SizeType n_modes = TDenseSpace::Size2(r_A);
VectorType B_res(n_modes);
TDenseSpace::Mult(r_B, r_res_vect, B_res);
TDenseSpace::Mult(r_A, B_res, rCorrection);
TDenseSpace::UnaliasedAdd(rCorrection, -1.0, r_res_vect);
} else {
KRATOS_ERROR << "There is neither observation nor old Jacobian decomposition. Correction cannot be computed." << std::endl;
}
}
}

void CalculateInverseJacobianRandomizedSVD()
{
const auto p_W = BaseType::pGetSolutionObservationMatrix();
const auto p_V = BaseType::pGetResidualObservationMatrix();

if (p_V != nullptr && p_W != nullptr) {
MatrixType aux_M;
CalculateAuxiliaryMatrixM(aux_M);

if (!mRandomValuesAreInitialized || mpOmega == nullptr) {
InitializeRandomValuesMatrix();
}

MatrixType y;
MultiplyRight(aux_M, *mpOmega, y);

mpDenseQR->Compute(y);
const SizeType n_dofs = BaseType::GetProblemSize();
const SizeType n_modes = TDenseSpace::Size2(*mpOmega);
MatrixType Q(n_dofs, n_modes);
mpDenseQR->MatrixQ(Q);

MatrixType phi;
MultiplyTransposeLeft(aux_M, Q, phi);

VectorType s_svd; 
MatrixType u_svd; 
MatrixType v_svd; 
Parameters svd_settings(R"({
"compute_thin_u" : true,
"compute_thin_v" : true
})");
mpDenseSVD->Compute(phi, s_svd, u_svd, v_svd, svd_settings);

SizeType n_modes_final = n_modes - mNumberOfExtraModes;
auto p_aux_Q_U = Kratos::make_shared<MatrixType>(ZeroMatrix(n_dofs, n_modes_final));
auto p_aux_sigma_V = Kratos::make_shared<MatrixType>(n_modes_final, n_dofs);
auto& r_aux_Q_U = *p_aux_Q_U;
auto& r_aux_sigma_V = *p_aux_sigma_V;
IndexPartition<SizeType>(n_dofs).for_each([&](SizeType I){
for (SizeType j = 0; j < n_modes_final; ++j) {
for (SizeType k = 0; k < n_modes_final; ++k) {
r_aux_Q_U(I,j) += Q(I,k) * u_svd(k,j);
}
r_aux_sigma_V(j,I) = s_svd(j) * v_svd(I,j);
}
});

std::swap(mpJacQU, p_aux_Q_U);
std::swap(mpJacSigmaV, p_aux_sigma_V);

if (mLimitModesToIterations) {
mpOmega = nullptr;
}

} else {
mpJacQU = nullptr;
mpJacSigmaV = nullptr;
}
}


MatrixPointerType pGetJacobianDecompositionMatrixQU() override
{
return mpJacQU;
}

MatrixPointerType pGetJacobianDecompositionMatrixSigmaV() override
{
return mpJacSigmaV;
}

MatrixPointerType pGetOldJacobianDecompositionMatrixQU() override
{
return mpOldJacQU;
}

MatrixPointerType pGetOldJacobianDecompositionMatrixSigmaV() override
{
return mpOldJacSigmaV;
}

private:



SizeType mSeed = 0; 
SizeType mUserNumberOfModes; 
SizeType mNumberOfExtraModes; 
SizeType mMinRandSVDExtraModes; 
SizeType mCurrentNumberOfModes = 0; 
bool mAutomaticJacobianModes = true; 
bool mLimitModesToIterations = true; 
bool mRandomValuesAreInitialized = false; 

DenseQRPointerType mpDenseQR; 
DenseSVDPointerType mpDenseSVD; 

MatrixPointerType mpOmega = nullptr; 
MatrixPointerType mpJacQU = nullptr; 
MatrixPointerType mpJacSigmaV = nullptr; 
MatrixPointerType mpOldJacQU = nullptr; 
MatrixPointerType mpOldJacSigmaV = nullptr; 

SizeType mOldMaxRank = 0; 




void InitializeRandomValuesMatrix()
{
const SizeType n_obs = BaseType::GetNumberOfObservations();
const SizeType full_rank_modes = mOldMaxRank + n_obs;
SizeType n_modes;
if (mAutomaticJacobianModes) {
n_modes = full_rank_modes; 
} else {
n_modes = mLimitModesToIterations ? std::min(full_rank_modes, mUserNumberOfModes) : mUserNumberOfModes;
}

const SizeType aux_extra_modes = std::ceil(0.1 * n_modes);
mNumberOfExtraModes = mMinRandSVDExtraModes > aux_extra_modes ? mMinRandSVDExtraModes : aux_extra_modes;
mCurrentNumberOfModes = n_modes + mNumberOfExtraModes;

const SizeType n_dofs = BaseType::GetProblemSize();
auto p_aux_omega = Kratos::make_shared<MatrixType>(n_dofs, mCurrentNumberOfModes);
std::swap(p_aux_omega, mpOmega);

std::mt19937 generator(mSeed); 
std::uniform_real_distribution<> distribution(0.0, 1.0);

auto& r_omega_matrix = *mpOmega;
for (SizeType i = 0; i < n_dofs; ++i) {
for (SizeType j = 0; j < mCurrentNumberOfModes; ++j) {
r_omega_matrix(i,j) = distribution(generator);
}
}

mRandomValuesAreInitialized = !mLimitModesToIterations;
}

void MultiplyRight(
const Matrix& rAuxM,
const Matrix& rRightMatrix,
Matrix& rSolution)
{
const SizeType n_dofs = BaseType::GetProblemSize();
KRATOS_ERROR_IF(TDenseSpace::Size1(rRightMatrix) != n_dofs) << "Obtained right multiplication matrix size " << TDenseSpace::Size1(rRightMatrix) << " does not match the problem size " << n_dofs << " expected one." << std::endl;
if (TDenseSpace::Size1(rSolution) != n_dofs || TDenseSpace::Size2(rSolution) != mCurrentNumberOfModes) {
rSolution.resize(n_dofs, mCurrentNumberOfModes);
}

const auto& r_W = *(BaseType::pGetSolutionObservationMatrix());
const auto& r_V = *(BaseType::pGetResidualObservationMatrix());

MatrixType M_omega = prod(rAuxM, rRightMatrix);
noalias(rSolution) = prod(r_W, M_omega);

if (mpOldJacQU == nullptr && mpOldJacSigmaV == nullptr) {
if (!BaseType::IsUsedInBlockNewtonEquations()) {
MatrixType V_M_omega = prod(r_V, M_omega);
IndexPartition<SizeType>(n_dofs).for_each([&rSolution,&V_M_omega,this](SizeType I){
for (SizeType j = 0; j < mCurrentNumberOfModes; ++j) {
rSolution(I,j) += V_M_omega(I,j);
}
});
}
} else {
MatrixType V_M_omega = prod(r_V, M_omega);
MatrixType B_omega = prod(*mpOldJacSigmaV, rRightMatrix);
MatrixType A_B_omega = prod(*mpOldJacQU, B_omega);
MatrixType B_V_M_omega = prod(*mpOldJacSigmaV, V_M_omega);
MatrixType A_B_V_M_omega = prod(*mpOldJacQU, B_V_M_omega);
if (!BaseType::IsUsedInBlockNewtonEquations()) {
IndexPartition<SizeType>(n_dofs).for_each([&rSolution,&A_B_omega,&V_M_omega,&A_B_V_M_omega,this](SizeType I){
for (SizeType j = 0; j < mCurrentNumberOfModes; ++j) {
rSolution(I,j) += A_B_omega(I,j) + V_M_omega(I,j) - A_B_V_M_omega(I,j);
}
});
} else {
IndexPartition<SizeType>(n_dofs).for_each([&rSolution,&A_B_omega,&A_B_V_M_omega,this](SizeType I){
for (SizeType j = 0; j < mCurrentNumberOfModes; ++j) {
rSolution(I,j) += A_B_omega(I,j) - A_B_V_M_omega(I,j);
}
});
}
}
}

void MultiplyTransposeLeft(
const Matrix& rAuxM,
const Matrix& rLeftMatrix,
Matrix& rSolution)
{
const SizeType n_dofs = BaseType::GetProblemSize();
KRATOS_ERROR_IF(TDenseSpace::Size1(rLeftMatrix) != n_dofs) << "Obtained left multiplication matrix size " << TDenseSpace::Size1(rLeftMatrix) << " does not match the problem size " << n_dofs << " expected one." << std::endl;
if (TDenseSpace::Size1(rSolution) != mCurrentNumberOfModes|| TDenseSpace::Size2(rSolution) != n_dofs) {
rSolution.resize(mCurrentNumberOfModes, n_dofs);
}

const auto& r_W = *(BaseType::pGetSolutionObservationMatrix());
const auto& r_V = *(BaseType::pGetResidualObservationMatrix());

const auto Qtrans = trans(rLeftMatrix);
MatrixType Qtrans_W = prod(Qtrans, r_W);
noalias(rSolution) = prod(Qtrans_W, rAuxM);

if (mpOldJacQU == nullptr && mpOldJacSigmaV == nullptr) {
if (!BaseType::IsUsedInBlockNewtonEquations()) {
MatrixType Qtrans_V = prod(Qtrans, r_V);
MatrixType Qtrans_V_M = prod(Qtrans_V, rAuxM);
IndexPartition<SizeType>(n_dofs).for_each([&rSolution,&Qtrans_V_M,this](SizeType J){
for (SizeType i = 0; i < mCurrentNumberOfModes; ++i) {
rSolution(i,J) += Qtrans_V_M(i,J);
}
});
}
} else {
MatrixType Qtrans_V = prod(Qtrans, r_V);
MatrixType Qtrans_V_M = prod(Qtrans_V, rAuxM);
MatrixType Qtrans_A = prod(Qtrans, *mpOldJacQU);
MatrixType Qtrans_A_B = prod(Qtrans_A, *mpOldJacSigmaV);
MatrixType Qtrans_A_B_V = prod(Qtrans_A_B, r_V);
MatrixType Qtrans_A_B_V_M = prod(Qtrans_A_B_V, rAuxM);
if (!BaseType::IsUsedInBlockNewtonEquations()) {
IndexPartition<SizeType>(n_dofs).for_each([&rSolution,&Qtrans_A_B,&Qtrans_V_M,&Qtrans_A_B_V_M,this](SizeType J){
for (SizeType i = 0; i < mCurrentNumberOfModes; ++i) {
rSolution(i,J) += Qtrans_A_B(i,J) + Qtrans_V_M(i,J) - Qtrans_A_B_V_M(i,J);
}
});
} else {
IndexPartition<SizeType>(n_dofs).for_each([&rSolution,&Qtrans_A_B,&Qtrans_A_B_V_M,this](SizeType J){
for (SizeType i = 0; i < mCurrentNumberOfModes; ++i) {
rSolution(i,J) += Qtrans_A_B(i,J) - Qtrans_A_B_V_M(i,J);
}
});
}
}
}

void CalculateAuxiliaryMatrixM(MatrixType& rAuxM)
{
const auto& r_V = *(BaseType::pGetResidualObservationMatrix());
MatrixType aux_V(r_V);
mpDenseQR->Compute(aux_V);

const std::size_t n_dofs = TDenseSpace::Size1(r_V);
const std::size_t n_data_cols = TDenseSpace::Size2(r_V);
MatrixType Q(n_dofs, n_data_cols);
MatrixType R(n_data_cols, n_data_cols);
MatrixType P(n_data_cols, n_data_cols);
mpDenseQR->MatrixQ(Q);
mpDenseQR->MatrixR(R);
mpDenseQR->MatrixP(P);

MatrixType R_transP(n_data_cols, n_data_cols);
noalias(R_transP) = prod(R, trans(P));

MatrixType R_transP_inv;
CalculateMoorePenroseInverse(R_transP, R_transP_inv);

const std::size_t m = TDenseSpace::Size1(rAuxM);
const std::size_t n = TDenseSpace::Size2(rAuxM);
if (m != n_data_cols || n != n_dofs) {
TDenseSpace::Resize(rAuxM, n_data_cols, n_dofs);
}
const MatrixType trans_Q = trans(Q);
noalias(rAuxM) = prod(R_transP_inv, trans_Q);
}

void CalculateMoorePenroseInverse(
const MatrixType& rInputMatrix,
MatrixType& rMoorePenroseInverse)
{
IndexType aux_size_1 = TDenseSpace::Size1(rInputMatrix);
IndexType aux_size_2 = TDenseSpace::Size2(rInputMatrix);
KRATOS_ERROR_IF_NOT(aux_size_1 == aux_size_2) << "Input matrix is not squared." << std::endl;

VectorType s_svd; 
MatrixType u_svd; 
MatrixType v_svd; 
Parameters svd_settings(R"({
"compute_thin_u" : true,
"compute_thin_v" : true
})");
mpDenseSVD->Compute(const_cast<MatrixType&>(rInputMatrix), s_svd, u_svd, v_svd, svd_settings);
const std::size_t n_sing_val = s_svd.size();
MatrixType s_inv = ZeroMatrix(n_sing_val, n_sing_val);
for (std::size_t i = 0; i < n_sing_val; ++i) {
s_inv(i,i) = 1.0 / s_svd(i);
}

rMoorePenroseInverse = ZeroMatrix(aux_size_2, aux_size_1);

for (std::size_t i = 0; i < aux_size_2; ++i) {
for (std::size_t j = 0; j < aux_size_1; ++j) {
double& r_value = rMoorePenroseInverse(i,j);
for (std::size_t k = 0; k < n_sing_val; ++k) {
const double v_ik = v_svd(i,k);
for (std::size_t m = 0; m < n_sing_val; ++m) {
const double ut_mj = u_svd(j,m);
const double s_inv_km = s_inv(k,m);
r_value += v_ik * s_inv_km * ut_mj;
}
}
}
}
}









}; 






}  
