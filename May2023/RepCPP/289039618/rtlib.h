#ifndef CATO_RTLIB_RTLIB_H
#define CATO_RTLIB_RTLIB_H

#include <iostream>
#include <memory>
#include <mpi.h>

#include "CatoRuntimeLogger.h"
#include "MemoryAbstractionHandler.h"


enum BinOp
{
Xchg,
Add,
Sub,
And,
Nand,
Or,
Xor,
Max,
Min,
UMax,
UMin,

FAdd,

FSub,

FIRST_BINOP = Xchg,
LAST_BINOP = FSub,
BAD_BINOP
};



int MPI_RANK = 0;
int MPI_SIZE = 0;

std::unique_ptr<MemoryAbstractionHandler> _memory_handler;


void print_hello();


void test_func(int num_args, ...);


void cato_initialize(bool logging);


void cato_finalize();


int get_mpi_rank();


int get_mpi_size();


void mpi_barrier();


void *allocate_shared_memory(long size, MPI_Datatype, int dimensions);


void shared_memory_free(void *base_ptr);


void shared_memory_store(void *base_ptr, void *value_ptr, int num_indices, ...);


void shared_memory_load(void *base_ptr, void *dest_ptr, int num_indices, ...);


void shared_memory_sequential_store(void *base_ptr, void *value_ptr, int num_indices, ...);


void shared_memory_sequential_load(void *base_ptr, void *dest_ptr, int num_indices, ...);


void shared_memory_pointer_store(void *dest_ptr, void *source_ptr, long dest_index);


void allocate_shared_value(void *base_ptr, MPI_Datatype type);


void shared_value_store(void *base_ptr, void *value_ptr);


void shared_value_load(void *base_ptr, void *dest_ptr);


void shared_value_synchronize(void *base_ptr);


void modify_parallel_for_bounds(int *lower_bound, int *upper_bound, int increment);
void modify_parallel_for_bounds(long *lower_bound, long *upper_bound, long increment);

template <typename T>
void modify_parallel_for_bounds(T *lower_bound, T *upper_bound, T increment)
{
Debug(std::cout << "Modifing parallel for loop bounds.\n";);
Debug(std::cout << "Lower bound: " << *lower_bound << "\nUpper bound: " << *upper_bound
<< "\nIncrement: " << increment << "\n";);

if (increment != 1)
{
std::cerr << "Warning: Currently only loop increments of value 1 are supported\n";
}

T local_lbound, local_ubound;

T total_iterations = *upper_bound - *lower_bound + 1;
T div = total_iterations / MPI_SIZE;
T rest = total_iterations % MPI_SIZE;

if (MPI_RANK < rest)
{
local_lbound = *lower_bound + MPI_RANK * (div + 1);
local_ubound = local_lbound + div;
}
else
{
local_lbound = *lower_bound + MPI_RANK * div + rest;
local_ubound = local_lbound + div - 1;
}

Debug(std::cout << "Local lower bound: " << local_lbound << "\nLocal upper bound: "
<< local_ubound << "\nIncrement: " << increment << "\n";);

*lower_bound = local_lbound;
*upper_bound = local_ubound;
}


void *critical_section_init();


void critical_section_enter(void *mpi_mutex);


void critical_section_leave(void *mpi_mutex);


void critical_section_finalize(void *mpi_mutex);


void reduce_local_vars(void *local_var, int bin_op, MPI_Datatype type);

#endif
