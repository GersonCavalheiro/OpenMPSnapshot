#include "solver.h"
#include "omp.h"
#define PRECISION 1e-8
#define MY_TAG 777
vector<double> MPI_OMP_Solver(const size_t &N, 
const function<double(size_t)> &A,
const function<double(size_t)> &B,
const function<double(size_t)> &C,
const function<double(size_t)> &F,
const size_t &num_mpi,
const size_t &max_mpi,
const size_t &max_omp)
{
MPI_Status status;
#ifdef TEST
vector<double> correct = SequentialThomasSolver(N, A, B, C, F);
#endif
size_t mp = max_mpi*max_omp;
vector<double> BUFFER_A(mp), BUFFER_B(mp), BUFFER_C(mp), BUFFER_F(mp);
vector<double> ANSWER(N + 1);
vector<double> B_V(N + 1), C_V(N + 1), F_V(N + 1), L_V(N + 1), R_V(N + 1);
#pragma omp parallel num_threads(max_omp) 
{
const size_t num_omp = omp_get_thread_num(); 
const size_t np = num_mpi*max_omp + num_omp;
const size_t l = np * (N + 1) / mp;
const size_t r = (np == mp - 1) ? N : (np + 1) * (N + 1) / mp - 1;
for (size_t i = l; i <= r; i++) {
B_V[i] = -B(i);
C_V[i] = C(i);
F_V[i] = F(i);
}
L_V[l] = A(l);
for (size_t i = l + 1; i <= r; i++) {
double tmp_ = A(i) / B_V[i - 1];
B_V[i] -= tmp_ * C_V[i - 1];
F_V[i] -= tmp_ * F_V[i - 1];
if (np != 0 && i != l)
L_V[i] -= tmp_ * L_V[i - 1];
}
#ifdef TEST
TopTriangleCheck(l, r, np, L_V, B_V, C_V, F_V, correct);
#pragma omp barrier
#endif
R_V[r - 1] = C_V[r - 1];
for (size_t i = r - 2; i >= l; i--) {
if (i == 0 && np == 0)
break;
double tmp = C_V[i] / B_V[i + 1];
C_V[i] -= tmp * B_V[i + 1];
R_V[i] -= tmp * R_V[i + 1];
L_V[i] -= tmp * L_V[i + 1];
F_V[i] -= tmp * F_V[i + 1];
}
#pragma omp barrier
if(num_mpi != 0 && num_omp == 0) 
{
vector<double> tmp_vector = {L_V[l], B_V[l], R_V[l], F_V[l]}; 
MPI_Send(tmp_vector.data(), 4, MPI_DOUBLE, num_mpi - 1, MY_TAG, MPI_COMM_WORLD);
}
if( num_mpi != max_mpi-1 && num_omp == (max_omp - 1) ) {				
vector<double> tmp_vector(4);
MPI_Recv(tmp_vector.data(), 4, MPI_DOUBLE, num_mpi + 1, MY_TAG, MPI_COMM_WORLD, &status);
L_V[r+1] = tmp_vector[0];
B_V[r+1] = tmp_vector[1];
R_V[r+1] = tmp_vector[2];
F_V[r+1] = tmp_vector[3];
}
if (np != mp-1) {
size_t i = r;
double tmp = C_V[i] / B_V[i + 1];
C_V[i] -= tmp * B_V[i + 1];
R_V[i] -= tmp * R_V[i + 1];
B_V[i] -= tmp * L_V[i + 1];
F_V[i] -= tmp * F_V[i + 1];
}
BUFFER_A[np] = L_V[r];
BUFFER_B[np] = B_V[r];
BUFFER_C[np] = R_V[r];
BUFFER_F[np] = F_V[r];
#pragma omp barrier
if(num_omp == 0)
for(int i=0; i < max_mpi; i++)
{
MPI_Bcast(BUFFER_A.data() + i*max_omp, max_omp, MPI_DOUBLE, i, MPI_COMM_WORLD);
MPI_Bcast(BUFFER_B.data() + i*max_omp, max_omp, MPI_DOUBLE, i, MPI_COMM_WORLD);		
MPI_Bcast(BUFFER_C.data() + i*max_omp, max_omp, MPI_DOUBLE, i, MPI_COMM_WORLD);
MPI_Bcast(BUFFER_F.data() + i*max_omp, max_omp, MPI_DOUBLE, i, MPI_COMM_WORLD);
}
#pragma omp barrier
#ifdef TEST 
PseudoParallelThomasSolver(N, A, B, C, F, mp,  
B_V, C_V, F_V, L_V, R_V, BUFFER_A, BUFFER_B, BUFFER_C, BUFFER_F, np);
MatrixCheck(l, r, np, mp, N, L_V, B_V, C_V, F_V, R_V, correct);
#endif
vector<double> small_solution = SequentialThomasSolver(mp - 1,
[=](size_t i) { return BUFFER_A[i];},
[=](size_t i) { return -BUFFER_B[i];},
[=](size_t i) { return BUFFER_C[i];},
[=](size_t i) { return BUFFER_F[i];});
for(int i=0;  i<mp; i++) {
size_t r_ = (i == mp - 1) ? N : (i + 1) * (N + 1) / mp - 1;
ANSWER[r_] = small_solution[i];	
}
for (ssize_t i = r - 1; i >= (ssize_t) l; i--) {
if (np == 0) {
ANSWER[i] = (F_V[i] - ANSWER[r] * R_V[i]) / B_V[i];
} else {
ANSWER[i] = (F_V[i] - ANSWER[l - 1] * L_V[i] - ANSWER[r] * R_V[i]) / B_V[i];
}
}
#pragma omp barrier
if(num_omp == 0) 
for(size_t i = 0; i<max_mpi; i++) {
size_t l_ = i * (N + 1) / max_mpi;
size_t r_ = (i == max_mpi - 1) ? N : (i + 1) * (N + 1) / max_mpi - 1;
MPI_Bcast(ANSWER.data() + l_, r_-l_, MPI_DOUBLE, i, MPI_COMM_WORLD);
}
}
#ifdef TEST
for (size_t i = 0; i <= N; i++) {
assert(abs(ANSWER[i] - correct[i]) < PRECISION);
}
ThomasSolutionTest(ANSWER, N, A, B, C, F);
#endif
return ANSWER;
}
vector<double> SequentialThomasSolver(const size_t &N,
const function<double(size_t)> &A,
const function<double(size_t)> &B,
const function<double(size_t)> &C,
const function<double(size_t)> &F) {
#ifdef TEST
DiagonalDominance(N, A, B, C, F);
#endif
vector<double> s(N + 1);
vector<double> t(N + 1);
vector<double> result(N + 1);
s[0] = C(0) / B(0);
t[0] = -F(0) / B(0);
for (size_t i = 1; i <= N; i++) {
s[i] = C(i) / (B(i) - A(i) * s[i - 1]);
t[i] = (A(i) * t[i - 1] - F(i)) / (B(i) - A(i) * s[i - 1]);
}
result[N] = t[N];
for (size_t i = N - 1; i > 0; i--) {
result[i] = result[i + 1] * s[i] + t[i];
}
result[0] = result[1] * s[0] + t[0];
#ifdef TEST
ThomasSolutionTest(result, N, A, B, C, F);
#endif
return result;
}
vector<double> PseudoParallelThomasSolver(const size_t &N,
const function<double(size_t)> &A,
const function<double(size_t)> &B,
const function<double(size_t)> &C,
const function<double(size_t)> &F, const size_t &mp,
const vector<double> B_V_,
const vector<double> C_V_,
const vector<double> F_V_,
const vector<double> L_V_,
const vector<double> R_V_,
const vector<double> BUFFER_A_,
const vector<double> BUFFER_B_,
const vector<double> BUFFER_C_,
const vector<double> BUFFER_F_, size_t np_) {
#ifdef TEST
vector<double> correct = SequentialThomasSolver(N, A, B, C, F);
ThomasSolutionTest(correct, N, A, B, C, F);
#endif
vector<double> BUFFER_A(mp), BUFFER_B(mp), BUFFER_C(mp), BUFFER_F(mp);
vector<double> ANSWER(N + 1);
vector<double> B_V(N + 1), C_V(N + 1), F_V(N + 1), L_V(N + 1), R_V(N + 1);
for (size_t i = 0; i < N + 1; i++) {
B_V[i] = -B(i);
C_V[i] = C(i);
F_V[i] = F(i);
}
for (size_t np = 0; np < mp; np++) {
size_t l = np * (N + 1) / mp;
size_t r = (np + 1) * (N + 1) / mp - 1;
if (np == mp-1) {
r = N;
}
L_V[l] = A(l);
for (size_t i = l + 1; i <= r; i++) {
double tmp = A(i) / B_V[i - 1];
B_V[i] -= tmp * C_V[i - 1];
F_V[i] -= tmp * F_V[i - 1];
if (np != 0 && i != l)
L_V[i] -= tmp * L_V[i - 1];
}
#ifdef TEST
TopTriangleCheck(l, r, np, L_V, B_V, C_V, F_V, correct);
#endif
R_V[r - 1] = C_V[r - 1];
for (size_t i = r - 2; i >= l; i--) {
if (i == 0 && np == 0)
break;
double tmp = C_V[i] / B_V[i + 1];
C_V[i] -= tmp * B_V[i + 1];
R_V[i] -= tmp * R_V[i + 1];
L_V[i] -= tmp * L_V[i + 1];
F_V[i] -= tmp * F_V[i + 1];
}
if (np != 0) {
size_t i = l - 1;
double tmp = C_V[i] / B_V[i + 1];
C_V[i] -= tmp * B_V[i + 1];
R_V[i] -= tmp * R_V[i + 1];
B_V[i] -= tmp * L_V[i + 1];
F_V[i] -= tmp * F_V[i + 1];
}
}
for (size_t np = 0; np < mp; np++) {
size_t l = np * (N + 1) / mp;
size_t r = (np + 1) * (N + 1) / mp - 1;
if (np == mp-1)
r = N;
BUFFER_A[np] = L_V[r];
BUFFER_B[np] = B_V[r];
BUFFER_C[np] = R_V[r];
BUFFER_F[np] = F_V[r];
#ifdef TEST
if(np == np_)
{
size_t l_ = np_ * (N + 1) / mp;
size_t r_ = (np_ + 1) * (N + 1) / mp - 1;
if (np_ == mp-1) {
r_ = N;
}
for(size_t i = l_; i<=r_; i++) {
assert(abs(B_V[i] - B_V_[i])<PRECISION);
assert(abs(F_V[i] - F_V_[i])<PRECISION);
assert(abs(L_V[i] - L_V_[i])<PRECISION);
assert(abs(R_V[i] - R_V_[i])<PRECISION);
}
assert( abs(BUFFER_A[np] - BUFFER_A_[np]) < PRECISION);
assert( abs(BUFFER_B[np] - BUFFER_B_[np]) < PRECISION);
assert( abs(BUFFER_C[np] - BUFFER_C_[np]) < PRECISION);
assert( abs(BUFFER_F[np] - BUFFER_F_[np]) < PRECISION);
}
#endif
#ifdef TEST
MatrixCheck(l, r, np, mp, N, L_V, B_V, C_V, F_V, R_V, correct);
#endif
}
vector<double> small_solution = SequentialThomasSolver(mp - 1,
[=](size_t i) { return BUFFER_A[i]; },
[=](size_t i) { return -BUFFER_B[i]; },
[=](size_t i) { return BUFFER_C[i]; },
[=](size_t i) { return BUFFER_F[i]; });
for (size_t np = 0; np < mp; np++) {
size_t l = np * (N + 1) / mp;
size_t r = (np + 1) * (N + 1) / mp - 1;
if (np == mp-1)
r = N;
ANSWER[r] = small_solution[np];
for (ssize_t i = r - 1; i >= (ssize_t) l; i--) {
if (np == 0) {
ANSWER[i] = (F_V[i] - ANSWER[r] * R_V[i]) / B_V[i];
} else {
ANSWER[i] = (F_V[i] - ANSWER[l - 1] * L_V[i] - ANSWER[r] * R_V[i]) / B_V[i];
}
}
}
#ifdef TEST
{
double max_d = -1;
for (size_t i = 0; i <= N; i++) {
max_d = max(max_d, abs(ANSWER[i] - correct[i]));
assert(abs(ANSWER[i] - correct[i]) < PRECISION);
}
ThomasSolutionTest(ANSWER, N, A, B, C, F);
};
#endif
return ANSWER;
}
vector<double> MPISolver(const size_t &N,
const function<double(size_t)> &A,
const function<double(size_t)> &B,
const function<double(size_t)> &C,
const function<double(size_t)> &F,
const size_t &np,
const size_t &mp)
{
MPI_Status status;
const size_t l = np * (N + 1) / mp;
const size_t r = (np == mp - 1) ? N : (np + 1) * (N + 1) / mp - 1;
#ifdef TEST
vector<double> correct = SequentialThomasSolver(N, A, B, C, F);
#endif
vector<double> BUFFER_A(mp), BUFFER_B(mp), BUFFER_C(mp), BUFFER_F(mp);
vector<double> ANSWER(N + 1);
vector<double> B_V(N + 1), C_V(N + 1), F_V(N + 1), L_V(N + 1), R_V(N + 1);
for (size_t i = 0; i <= N; i++) {
B_V[i] = -B(i);
C_V[i] = C(i);
F_V[i] = F(i);
}
L_V[l] = A(l);
for (size_t i = l + 1; i <= r; i++) {
double tmp_ = A(i) / B_V[i - 1];
B_V[i] -= tmp_ * C_V[i - 1];
F_V[i] -= tmp_ * F_V[i - 1];
if (np != 0 && i != l)
L_V[i] -= tmp_ * L_V[i - 1];
}
#ifdef TEST
TopTriangleCheck(l, r, np, L_V, B_V, C_V, F_V, correct);
#endif
R_V[r - 1] = C_V[r - 1];
for (size_t i = r - 2; i >= l; i--) {
if (i == 0 && np == 0)
break;
double tmp = C_V[i] / B_V[i + 1];
C_V[i] -= tmp * B_V[i + 1];
R_V[i] -= tmp * R_V[i + 1];
L_V[i] -= tmp * L_V[i + 1];
F_V[i] -= tmp * F_V[i + 1];
}
if(np != 0) {
MPI_Send(&L_V[l], 1, MPI_DOUBLE, np-1, MY_TAG, MPI_COMM_WORLD);
MPI_Send(&B_V[l], 1, MPI_DOUBLE, np-1, MY_TAG, MPI_COMM_WORLD);		
MPI_Send(&R_V[l], 1, MPI_DOUBLE, np-1, MY_TAG, MPI_COMM_WORLD);
MPI_Send(&F_V[l], 1, MPI_DOUBLE, np-1, MY_TAG, MPI_COMM_WORLD);	
}
if( np != mp-1) {				
MPI_Recv(L_V.data()+r+1, 1, MPI_DOUBLE, np+1, MY_TAG, MPI_COMM_WORLD, &status);
MPI_Recv(B_V.data()+r+1, 1, MPI_DOUBLE, np+1, MY_TAG, MPI_COMM_WORLD, &status);
MPI_Recv(R_V.data()+r+1, 1, MPI_DOUBLE, np+1, MY_TAG, MPI_COMM_WORLD, &status);
MPI_Recv(F_V.data()+r+1, 1, MPI_DOUBLE, np+1, MY_TAG, MPI_COMM_WORLD, &status);
}
if (np != mp-1) {
size_t i = r;
double tmp = C_V[i] / B_V[i + 1];
C_V[i] -= tmp * B_V[i + 1];
R_V[i] -= tmp * R_V[i + 1];
B_V[i] -= tmp * L_V[i + 1];
F_V[i] -= tmp * F_V[i + 1];
}
#ifdef TEST 
PseudoParallelThomasSolver(N, A, B, C, F, mp,  
B_V, C_V, F_V, L_V, R_V, BUFFER_A, BUFFER_B, BUFFER_C, BUFFER_F, np);
MatrixCheck(l, r, np, mp, N, L_V, B_V, C_V, F_V, R_V, correct);
#endif
for(int i=0; i<mp; i++) {
if (i != np) {
MPI_Send(&L_V[r], 1, MPI_DOUBLE, i, MY_TAG, MPI_COMM_WORLD);
MPI_Send(&B_V[r], 1, MPI_DOUBLE, i, MY_TAG, MPI_COMM_WORLD);		
MPI_Send(&R_V[r], 1, MPI_DOUBLE, i, MY_TAG, MPI_COMM_WORLD);
MPI_Send(&F_V[r], 1, MPI_DOUBLE, i, MY_TAG, MPI_COMM_WORLD);	
}
else {
BUFFER_A[np] = L_V[r];
BUFFER_B[np] = B_V[r];
BUFFER_C[np] = R_V[r];
BUFFER_F[np] = F_V[r];
}
}
for(int i=0; i<mp; i++) {
if (i != np) {
MPI_Recv(BUFFER_A.data()+i, 1, MPI_DOUBLE, i, MY_TAG, MPI_COMM_WORLD, &status);
MPI_Recv(BUFFER_B.data()+i, 1, MPI_DOUBLE, i, MY_TAG, MPI_COMM_WORLD, &status);		
MPI_Recv(BUFFER_C.data()+i, 1, MPI_DOUBLE, i, MY_TAG, MPI_COMM_WORLD, &status);
MPI_Recv(BUFFER_F.data()+i, 1, MPI_DOUBLE, i, MY_TAG, MPI_COMM_WORLD, &status);	
}
}
vector<double> small_solution = SequentialThomasSolver(mp - 1,
[=](size_t i) { return BUFFER_A[i];},
[=](size_t i) { return -BUFFER_B[i];},
[=](size_t i) { return BUFFER_C[i];},
[=](size_t i) { return BUFFER_F[i];});
for(int i=0;  i<mp; i++) {
size_t r_ = (i == mp - 1) ? N : (i + 1) * (N + 1) / mp - 1;
ANSWER[r_] = small_solution[i];	
}
for (ssize_t i = r - 1; i >= (ssize_t) l; i--) {
if (np == 0) {
ANSWER[i] = (F_V[i] - ANSWER[r] * R_V[i]) / B_V[i];
} else {
ANSWER[i] = (F_V[i] - ANSWER[l - 1] * L_V[i] - ANSWER[r] * R_V[i]) / B_V[i];
}
}
for(size_t i=0; i<mp; i++)
{
if(i!= np)
{
MPI_Send(ANSWER.data()+l, r-l, MPI_DOUBLE, i, MY_TAG, MPI_COMM_WORLD);
}
}
for(size_t i=0; i<mp; i++) {
if(i != np) {	
size_t l_ = i * (N + 1) / mp;
size_t r_ = (i == mp - 1) ? N : (i + 1) * (N + 1) / mp - 1;
MPI_Recv(ANSWER.data()+l_, r_-l_, MPI_DOUBLE, i, MY_TAG, MPI_COMM_WORLD, &status);
}
}
#ifdef TEST
double max_d = -1;
for (size_t i = 0; i <= N; i++) {
max_d = max(max_d, abs(ANSWER[i] - correct[i]));
assert(abs(ANSWER[i] - correct[i]) < PRECISION);
}
ThomasSolutionTest(ANSWER, N, A, B, C, F);
#endif
return ANSWER;
}