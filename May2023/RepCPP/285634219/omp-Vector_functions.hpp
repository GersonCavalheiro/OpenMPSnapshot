#ifndef _Vector_functions_hpp_
#define _Vector_functions_hpp_


#include <vector>
#include <sstream>
#include <fstream>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef MINIFE_HAVE_TBB
#include <LockingVector.hpp>
#endif

#include <TypeTraits.hpp>
#include <Vector.hpp>

#define MINIFE_MIN(X, Y)  ((X) < (Y) ? (X) : (Y))

namespace miniFE {


template<typename VectorType>
void write_vector(const std::string& filename,
const VectorType& vec)
{
int numprocs = 1, myproc = 0;
#ifdef HAVE_MPI
MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

std::ostringstream osstr;
osstr << filename << "." << numprocs << "." << myproc;
std::string full_name = osstr.str();
std::ofstream ofs(full_name.c_str());

typedef typename VectorType::ScalarType ScalarType;

const std::vector<ScalarType>& coefs = vec.coefs;
for(int p=0; p<numprocs; ++p) {
if (p == myproc) {
if (p == 0) {
ofs << vec.local_size << std::endl;
}

typename VectorType::GlobalOrdinalType first = vec.startIndex;
for(size_t i=0; i<vec.local_size; ++i) {
ofs << first+i << " " << coefs[i] << std::endl;
}
}
#ifdef HAVE_MPI
MPI_Barrier(MPI_COMM_WORLD);
#endif
}
}

template<typename VectorType>
void sum_into_vector(size_t num_indices,
const typename VectorType::GlobalOrdinalType* indices,
const typename VectorType::ScalarType* coefs,
VectorType& vec)
{
typedef typename VectorType::GlobalOrdinalType GlobalOrdinal;
typedef typename VectorType::ScalarType Scalar;

GlobalOrdinal first = vec.startIndex;
GlobalOrdinal last = first + vec.local_size - 1;

std::vector<Scalar>& vec_coefs = vec.coefs;

for(size_t i=0; i<num_indices; ++i) {
if (indices[i] < first || indices[i] > last) continue;
size_t idx = indices[i] - first;

#pragma omp atomic
vec_coefs[idx] += coefs[i];
}
}

#ifdef MINIFE_HAVE_TBB
template<typename VectorType>
void sum_into_vector(size_t num_indices,
const typename VectorType::GlobalOrdinalType* indices,
const typename VectorType::ScalarType* coefs,
LockingVector<VectorType>& vec)
{
vec.sum_in(num_indices, indices, coefs);
}
#endif

template<typename VectorType>
void
waxpby(typename VectorType::ScalarType alpha, const VectorType& x,
typename VectorType::ScalarType beta, const VectorType& y,
VectorType& w)
{
typedef typename VectorType::ScalarType ScalarType;

#ifdef MINIFE_DEBUG_OPENMP
std::cout << "Starting WAXPBY..." << std::endl;
#endif

#ifdef MINIFE_DEBUG
if (y.local_size < x.local_size || w.local_size < x.local_size) {
std::cerr << "miniFE::waxpby ERROR, y and w must be at least as long as x." << std::endl;
return;
}
#endif

const int n = x.coefs.size();
const ScalarType*  xcoefs = &x.coefs[0];
const ScalarType*  ycoefs = &y.coefs[0];
ScalarType*  wcoefs = &w.coefs[0];

if(beta == 0.0) {
if(alpha == 1.0) {
#pragma omp target teams distribute parallel for simd
for(int i=0; i<n; ++i) {
wcoefs[i] = xcoefs[i];
}
} else {
#pragma omp target teams distribute parallel for simd
for(int i=0; i<n; ++i) {
wcoefs[i] = alpha * xcoefs[i];
}
}
} else {
if(alpha == 1.0) {
#pragma omp target teams distribute parallel for simd
for(int i=0; i<n; ++i) {
wcoefs[i] = xcoefs[i] + beta * ycoefs[i];
}
} else {
#pragma omp simd
for(int i=0; i<n; ++i) {
wcoefs[i] = alpha * xcoefs[i] + beta * ycoefs[i];
}
}
}

#ifdef MINIFE_DEBUG_OPENMP
std::cout << "Finished WAXPBY." << std::endl;
#endif
}

template<typename VectorType>
void
daxpby(const MINIFE_SCALAR alpha, 
const VectorType& x,
const MINIFE_SCALAR beta, 
VectorType& y)
{

const MINIFE_LOCAL_ORDINAL n = MINIFE_MIN(x.coefs.size(), y.coefs.size());
const MINIFE_SCALAR*  xcoefs = &x.coefs[0];
MINIFE_SCALAR*  ycoefs = &y.coefs[0];

if(alpha == 1.0 && beta == 1.0) {
#pragma omp target teams distribute parallel for simd
for(int i = 0; i < n; ++i) {
ycoefs[i] += xcoefs[i];
}
} else if (beta == 1.0) {
#pragma omp target teams distribute parallel for simd
for(int i = 0; i < n; ++i) {
ycoefs[i] += alpha * xcoefs[i];
}
} else if (alpha == 1.0) {
#pragma omp target teams distribute parallel for simd
for(int i = 0; i < n; ++i) {
ycoefs[i] = xcoefs[i] + beta * ycoefs[i];
}
} else if (beta == 0.0) {
#pragma omp target teams distribute parallel for simd
for(int i = 0; i < n; ++i) {
ycoefs[i] = alpha * xcoefs[i];
}
} else {
#pragma omp target teams distribute parallel for simd
for(int i = 0; i < n; ++i) {
ycoefs[i] = alpha * xcoefs[i] + beta * ycoefs[i];
}
}

}

template<typename Vector>
typename TypeTraits<typename Vector::ScalarType>::magnitude_type
dot(const Vector& x,
const Vector& y)
{
const MINIFE_LOCAL_ORDINAL n = x.coefs.size();

typedef typename Vector::ScalarType Scalar;
typedef typename TypeTraits<typename Vector::ScalarType>::magnitude_type magnitude;

const Scalar*  xcoefs = &x.coefs[0];
const Scalar*  ycoefs = &y.coefs[0];
MINIFE_SCALAR result = 0;

map(tofrom:result) num_teams(512)

#pragma omp target data map(tofrom: result)
{
#pragma omp target teams distribute parallel for reduction(+:result) num_teams(512)
for(int i=0; i<n; ++i) {
result += xcoefs[i] * ycoefs[i];
}
}

#ifdef HAVE_MPI
magnitude local_dot = result, global_dot = 0;
MPI_Datatype mpi_dtype = TypeTraits<magnitude>::mpi_type();  
MPI_Allreduce(&local_dot, &global_dot, 1, mpi_dtype, MPI_SUM, MPI_COMM_WORLD);
return global_dot;
#else
return result;
#endif
}

template<typename Vector>
typename TypeTraits<typename Vector::ScalarType>::magnitude_type
dot_r2(const Vector& x)
{
#ifdef MINIFE_DEBUG_OPENMP
int myrank;
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
std::cout << "[" << myrank << "] Starting dot..." << std::endl;
#endif

const MINIFE_LOCAL_ORDINAL n = x.coefs.size();

#ifdef MINIFE_DEBUG
if (y.local_size < n) {
std::cerr << "miniFE::dot ERROR, y must be at least as long as x."<<std::endl;
n = y.local_size;
}
#endif

typedef typename Vector::ScalarType Scalar;
typedef typename TypeTraits<typename Vector::ScalarType>::magnitude_type magnitude;

const MINIFE_SCALAR*  xcoefs = &x.coefs[0];
MINIFE_SCALAR result = 0;

map(tofrom: result) num_teams(512)
#pragma omp target data map(tofrom: result)
{
#pragma omp target teams distribute parallel for reduction(+:result) num_teams(512)
for(int i=0; i<n; ++i) {
result += xcoefs[i] * xcoefs[i];
}
}

#ifdef HAVE_MPI
magnitude local_dot = result, global_dot = 0;
MPI_Datatype mpi_dtype = TypeTraits<magnitude>::mpi_type();  
MPI_Allreduce(&local_dot, &global_dot, 1, mpi_dtype, MPI_SUM, MPI_COMM_WORLD);

#ifdef MINIFE_DEBUG_OPENMP
std::cout << "[" << myrank << "] Completed dot." << std::endl;
#endif

return global_dot;
#else
#ifdef MINIFE_DEBUG_OPENMP
std::cout << "[" << myrank << "] Completed dot." << std::endl;
#endif
return result;
#endif
}

}

#endif

