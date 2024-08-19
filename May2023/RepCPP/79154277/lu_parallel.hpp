

#pragma once

#include "../serial/matrix.hpp"
#include "parallelism.hpp"

namespace advscicomp{


using ParC = ParallelismConfig;



template<SizeT N, typename T>
void DistributeMatrixAllAll(Matrix<N,N,T> & A)
{

if (ParC::num_procs>1)
for (unsigned ii{0}; ii<N; ++ii)
MPI_Bcast(A[ii].data(), N, NumTraits<T>::MPI_Type, ParC::head, MPI_COMM_WORLD);
}



std::tuple<std::vector<unsigned>,std::vector<unsigned>> PartitionOf(SizeT N)
{
std::vector<unsigned> lower_indices(ParC::num_procs);
std::vector<unsigned> upper_indices(ParC::num_procs);

if (ParC::num_procs <= N){
auto nrows_uniform = N/ParC::num_procs;

for (unsigned ii{0}; ii<ParC::num_procs; ++ii)
{
lower_indices[ii] = ii * nrows_uniform;
upper_indices[ii] = lower_indices[ii] + nrows_uniform;
}
upper_indices[ParC::num_procs-1] = N; 
}
else
{
for (SizeT ii{0}; ii<N; ++ii)
{
lower_indices[ii] = ii;
upper_indices[ii] = ii+1;
}
for (SizeT ii{N}; ii<ParC::num_procs; ++ii)
{
lower_indices[ii] = N;
upper_indices[ii] = N;
}
}

return std::make_tuple(lower_indices, upper_indices);
}


template <SizeT N>
int Possessor(unsigned k)
{
static bool computed{false};
static std::vector<int> who(N); 

if (!computed)
{
const auto bounds = PartitionOf(N);
const auto& lower = std::get<0>(bounds);
const auto& upper = std::get<1>(bounds);
const int num_procs = ParC::num_procs;

for (SizeT i{0}; i<N; ++i)
for (int j{0}; j<ParC::num_procs; ++j)
if (lower[j]<=i && i<upper[j])
{
who[i] = j;
break;
}
computed = true;
}

return who[k];
}

template<SizeT N>
bool IsPossessor(int k)
{
return Possessor<N>(k)==ParC::my_id;
}


template<SizeT N, typename T>
void LU_InPlace_Col_MPI(Matrix<N,N,T> & A)
{
DistributeMatrixAllAll(A); 


for (unsigned k=0; k<N-1; ++k) 
{
if (IsPossessor<N>(k)){ 
for (unsigned i=k+1; i<N; ++i)
{
A[i][k] /= A[k][k];
}
}

for (unsigned i=k+1; i<N; ++i)
{
MPI_Bcast(&A[i][k], 1, NumTraits<T>::MPI_Type, Possessor<N>(k), MPI_COMM_WORLD);
}

for (int j=k+1; j<N; ++j)
{
for (int i=k+1; i<N; ++i)
{
if (IsPossessor<N>(j)){ 
A[i][j] -= A[i][k]*A[k][j];
}
}
}
}

for (int k=0; k<N; ++k)
{
for (int i=1; i<=k; ++i) 
MPI_Bcast(&A[i][k], 1, NumTraits<T>::MPI_Type, Possessor<N>(k), MPI_COMM_WORLD);
}
}




template<SizeT N, typename T>
void LU_InPlace_Row_MPI(Matrix<N,N,T> & A)
{
DistributeMatrixAllAll(A); 


for (unsigned k=0; k<N-1; ++k) 
{
MPI_Bcast(&A[k][k], N-k, NumTraits<T>::MPI_Type, Possessor<N>(k), MPI_COMM_WORLD);

for (unsigned i=k+1; i<N; ++i)
{
if (IsPossessor<N>(i)) 
{
A[i][k] /= A[k][k];
}
}

for (unsigned i=k+1; i<N; ++i)
{
if (IsPossessor<N>(i))
for (int j=k+1; j<N; ++j)
A[i][j] -= A[i][k]*A[k][j];
}

}

for (int k=1; k<N; ++k)
{
MPI_Bcast(&A[k][0], k+1, NumTraits<T>::MPI_Type, Possessor<N>(k), MPI_COMM_WORLD);
}
}




} 


