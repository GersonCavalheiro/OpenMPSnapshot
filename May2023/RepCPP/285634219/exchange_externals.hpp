#ifndef _exchange_externals_hpp_
#define _exchange_externals_hpp_


#include <cstdlib>
#include <iostream>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <outstream.hpp>

#include <TypeTraits.hpp>

namespace miniFE {

template<typename MatrixType,
typename VectorType>
void
exchange_externals(MatrixType& A,
VectorType& x)
{
#ifdef HAVE_MPI
#ifdef MINIFE_DEBUG
std::ostream& os = outstream();
os << "entering exchange_externals\n";
#endif

int numprocs = 1;
MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

if (numprocs < 2) return;

typedef typename MatrixType::ScalarType Scalar;
typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;


int local_nrow = A.rows.size();
int num_neighbors = A.neighbors.size();
const std::vector<LocalOrdinal>& recv_length = A.recv_length;
const std::vector<LocalOrdinal>& send_length = A.send_length;
const std::vector<int>& neighbors = A.neighbors;
const std::vector<GlobalOrdinal>& elements_to_send = A.elements_to_send;

std::vector<Scalar>& send_buffer = A.send_buffer;


int MPI_MY_TAG = 99;

std::vector<MPI_Request>& request = A.request;


std::vector<Scalar>& x_coefs = x.coefs;
Scalar* x_external = &(x_coefs[local_nrow]);

MPI_Datatype mpi_dtype = TypeTraits<Scalar>::mpi_type();

for(int i=0; i<num_neighbors; ++i) {
int n_recv = recv_length[i];
MPI_Irecv(x_external, n_recv, mpi_dtype, neighbors[i], MPI_MY_TAG,
MPI_COMM_WORLD, &request[i]);
x_external += n_recv;
}

#ifdef MINIFE_DEBUG
os << "launched recvs\n";
#endif


size_t total_to_be_sent = elements_to_send.size();
#ifdef MINIFE_DEBUG
os << "total_to_be_sent: " << total_to_be_sent << std::endl;
#endif

#pragma omp parallel for
for(size_t i=0; i<total_to_be_sent; ++i) {
#ifdef MINIFE_DEBUG
if (elements_to_send[i] < 0 || elements_to_send[i] > x.coefs.size()) {
os << "error, out-of-range. x.coefs.size()=="<<x.coefs.size()<<", elements_to_send[i]=="<<elements_to_send[i]<<std::endl;
}
#endif
send_buffer[i] = x.coefs[elements_to_send[i]];
}


Scalar* s_buffer = &send_buffer[0];

for(int i=0; i<num_neighbors; ++i) {
int n_send = send_length[i];
MPI_Send(s_buffer, n_send, mpi_dtype, neighbors[i], MPI_MY_TAG,
MPI_COMM_WORLD);
s_buffer += n_send;
}

#ifdef MINIFE_DEBUG
os << "send to " << num_neighbors << std::endl;
#endif


MPI_Status status;
for(int i=0; i<num_neighbors; ++i) {
if (MPI_Wait(&request[i], &status) != MPI_SUCCESS) {
std::cerr << "MPI_Wait error\n"<<std::endl;
MPI_Abort(MPI_COMM_WORLD, -1);
}
}

#ifdef MINIFE_DEBUG
os << "leaving exchange_externals"<<std::endl;
#endif

#endif
}

#ifdef HAVE_MPI
static std::vector<MPI_Request> exch_ext_requests;
#endif

template<typename MatrixType,
typename VectorType>
void
begin_exchange_externals(MatrixType& A,
VectorType& x)
{
#ifdef HAVE_MPI

int numprocs = 1, myproc = 0;
MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &myproc);

if (numprocs < 2) return;

typedef typename MatrixType::ScalarType Scalar;
typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;


int local_nrow = A.rows.size();
int num_neighbors = A.neighbors.size();
const std::vector<LocalOrdinal>& recv_length = A.recv_length;
const std::vector<LocalOrdinal>& send_length = A.send_length;
const std::vector<int>& neighbors = A.neighbors;
const std::vector<GlobalOrdinal>& elements_to_send = A.elements_to_send;

std::vector<Scalar> send_buffer(elements_to_send.size(), 0);


int MPI_MY_TAG = 99;

exch_ext_requests.resize(num_neighbors);


std::vector<Scalar>& x_coefs = x.coefs;
Scalar* x_external = &(x_coefs[local_nrow]);

MPI_Datatype mpi_dtype = TypeTraits<Scalar>::mpi_type();

for(int i=0; i<num_neighbors; ++i) {
int n_recv = recv_length[i];
MPI_Irecv(x_external, n_recv, mpi_dtype, neighbors[i], MPI_MY_TAG,
MPI_COMM_WORLD, &exch_ext_requests[i]);
x_external += n_recv;
}


size_t total_to_be_sent = elements_to_send.size();
for(size_t i=0; i<total_to_be_sent; ++i) send_buffer[i] = x.coefs[elements_to_send[i]];


Scalar* s_buffer = &send_buffer[0];

for(int i=0; i<num_neighbors; ++i) {
int n_send = send_length[i];
MPI_Send(s_buffer, n_send, mpi_dtype, neighbors[i], MPI_MY_TAG,
MPI_COMM_WORLD);
s_buffer += n_send;
}
#endif
}

inline
void
finish_exchange_externals(int num_neighbors)
{
#ifdef HAVE_MPI

MPI_Status status;
for(int i=0; i<num_neighbors; ++i) {
if (MPI_Wait(&exch_ext_requests[i], &status) != MPI_SUCCESS) {
std::cerr << "MPI_Wait error\n"<<std::endl;
MPI_Abort(MPI_COMM_WORLD, -1);
}
}

#endif
}

}

#endif

