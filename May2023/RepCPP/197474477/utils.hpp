

#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

#define PI                          (3.14159)
#define MAX_PRINT_NEDGE             (100000)

#define MLCG                        (2147483647)    
#define ALCG                        (16807)         
#define BLCG                        (0)

#ifdef USE_SHARED_MEMORY
#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES       10
#endif
#endif
#ifndef NTIMES
#   define NTIMES       10
#endif
#ifdef ALIGNMENT
#if ALIGNMENT<=8
#   define ALIGNMENT    16
#endif
#endif
#ifndef ALIGNMENT
#   define ALIGNMENT    16
#endif
#ifndef DEFAULT_NV
#define DEFAULT_NV      524288
#endif
#else
#define SR_UP_TAG                   100
#define SR_DOWN_TAG                 101
#define SR_SIZES_UP_TAG             102
#define SR_SIZES_DOWN_TAG           103
#define SR_X_UP_TAG                 104
#define SR_X_DOWN_TAG               105
#define SR_Y_UP_TAG                 106
#define SR_Y_DOWN_TAG               107
#define SR_LCG_TAG                  108
#endif

#include <random>
#include <utility>
#include <cstring>

#ifdef USE_32_BIT_GRAPH
using GraphElem = int32_t;
using GraphWeight = float;
#ifdef USE_SHARED_MEMORY
typedef std::aligned_storage<sizeof(GraphElem),alignof(GraphElem)>::type __GraphElem__;
typedef std::aligned_storage<sizeof(GraphWeight),alignof(GraphWeight)>::type __GraphWeight__;
#else
const MPI_Datatype MPI_GRAPH_TYPE = MPI_INT32_T;
const MPI_Datatype MPI_WEIGHT_TYPE = MPI_FLOAT;
#endif
#else
using GraphElem = int64_t;
using GraphWeight = double;
#ifdef USE_SHARED_MEMORY
typedef std::aligned_storage<sizeof(GraphElem),alignof(GraphElem)>::type __GraphElem__;
typedef std::aligned_storage<sizeof(GraphWeight),alignof(GraphWeight)>::type __GraphWeight__;
#else
const MPI_Datatype MPI_GRAPH_TYPE = MPI_INT64_T;
const MPI_Datatype MPI_WEIGHT_TYPE = MPI_DOUBLE;
#endif
#endif

extern unsigned seed;

int is_pwr2(int pes) 
{ return ((pes != 0) && !(pes & (pes - 1))); }

GraphElem reseeder(unsigned initseed)
{
std::seed_seq seq({initseed});
std::vector<std::uint32_t> seeds(1);
seq.generate(seeds.begin(), seeds.end());

return (GraphElem)seeds[0];
}

template<typename T, typename G = std::default_random_engine>
T genRandom(T lo, T hi)
{
thread_local static G gen(seed);
using Dist = typename std::conditional
<
std::is_integral<T>::value
, std::uniform_int_distribution<T>
, std::uniform_real_distribution<T>
>::type;

thread_local static Dist utd {};
return utd(gen, typename Dist::param_type{lo, hi});
}


#ifdef USE_SHARED_MEMORY
class LCG
{
public:
LCG(unsigned seed, GraphWeight* drand, GraphElem n) : 
seed_(seed), drand_(drand), n_(n)
{
rnums_.resize(n_);

x0_ = reseeder(seed_);

prefix_op();
}

~LCG() { rnums_.clear(); }

void matmat_2x2(GraphElem c[], GraphElem a[], GraphElem b[])
{
for (int i = 0; i < 2; i++) {
for (int j = 0; j < 2; j++) {
GraphElem sum = 0;
for (int k = 0; k < 2; k++) {
sum += a[i*2+k]*b[k*2+j];
}
c[i*2+j] = sum;
}
}
}

void matop_2x2(GraphElem x[], GraphElem y[])
{
GraphElem tmp[4];
matmat_2x2(tmp, x, y);
memcpy(x, tmp, sizeof(GraphElem[4]));
}

void mat_power(GraphElem mat[], GraphElem k)
{
GraphElem tmp[4];
memcpy(tmp, mat, sizeof(GraphElem[4]));

for (GraphElem p = 0; p < k-1; p++)
matop_2x2(mat, tmp);
}

void prefix_op()
{
GraphElem global_op[4]; 
global_op[0] = ALCG;
global_op[1] = 0;
global_op[2] = BLCG;
global_op[3] = 1;

mat_power(global_op, n_);        
GraphElem prefix_op[4] = {1,0,0,1};  

rnums_[0] = x0_;
}

void generate()
{
#if defined(PRINT_LCG_LONG_RANDOM_NUMBERS)
std::cout << rnums_[0] << std::endl;
for (GraphElem i = 1; i < n_; i++) {
rnums_[i] = (rnums_[i-1]*ALCG + BLCG)%MLCG;
std::cout << rnums_[i] << std::endl;
}
#else
for (GraphElem i = 1; i < n_; i++) {
rnums_[i] = (rnums_[i-1]*ALCG + BLCG)%MLCG;
}
#endif
GraphWeight mult = 1.0 / (GraphWeight)(1.0 + (GraphWeight)(MLCG-1));

#if defined(PRINT_LCG_DOUBLE_RANDOM_NUMBERS)
for (GraphElem i = 0; i < n_; i++) {
drand_[i] = (GraphWeight)((GraphWeight)fabs(rnums_[i]) * mult ); 
std::cout << drand_[i] << std::endl;
}
#else
for (GraphElem i = 0; i < n_; i++)
drand_[i] = (GraphWeight)((GraphWeight)fabs(rnums_[i]) * mult); 
#endif
}

void rescale(GraphWeight* new_drand, GraphElem idx_start, GraphWeight const& lo)
{
GraphWeight range = 1.0;

#if defined(PRINT_LCG_DOUBLE_LOHI_RANDOM_NUMBERS)
for (GraphElem i = idx_start, j = 0; i < n_; i++, j++) {
new_drand[j] = lo + (GraphWeight)(range * drand_[i]);
std::cout << new_drand[j] << std::endl;
}
#else
for (GraphElem i = idx_start, j = 0; i < n_; i++, j++)
new_drand[j] = lo + (GraphWeight)(range * drand_[i]); 
#endif
}

private:
unsigned seed_;
GraphElem n_, x0_;
GraphWeight* drand_;
std::vector<GraphElem> rnums_;
};
#else 
class LCG
{
public:
LCG(unsigned seed, GraphWeight* drand, 
GraphElem n, MPI_Comm comm = MPI_COMM_WORLD): 
seed_(seed), drand_(drand), n_(n)
{
comm_ = comm;
MPI_Comm_size(comm_, &nprocs_);
MPI_Comm_rank(comm_, &rank_);

rnums_.resize(n_);

if (rank_ == 0)
x0_ = reseeder(seed_);

MPI_Bcast(&x0_, 1, MPI_GRAPH_TYPE, 0, comm_);

parallel_prefix_op();
}

~LCG() { rnums_.clear(); }

void matmat_2x2(GraphElem c[], GraphElem a[], GraphElem b[])
{
for (int i = 0; i < 2; i++) {
for (int j = 0; j < 2; j++) {
GraphElem sum = 0;
for (int k = 0; k < 2; k++) {
sum += a[i*2+k]*b[k*2+j];
}
c[i*2+j] = sum;
}
}
}

void matop_2x2(GraphElem x[], GraphElem y[])
{
GraphElem tmp[4];
matmat_2x2(tmp, x, y);
memcpy(x, tmp, sizeof(GraphElem[4]));
}

void mat_power(GraphElem mat[], GraphElem k)
{
GraphElem tmp[4];
memcpy(tmp, mat, sizeof(GraphElem[4]));

for (GraphElem p = 0; p < k-1; p++)
matop_2x2(mat, tmp);
}

void parallel_prefix_op()
{
GraphElem global_op[4]; 
global_op[0] = ALCG;
global_op[1] = 0;
global_op[2] = BLCG;
global_op[3] = 1;

mat_power(global_op, n_);        
GraphElem prefix_op[4] = {1,0,0,1};  

GraphElem global_op_recv[4];

int steps = (int)(log2((double)nprocs_));

for (int s = 0; s < steps; s++) {

int mate = rank_^(1 << s); 

MPI_Sendrecv(global_op, 4, MPI_GRAPH_TYPE, mate, SR_LCG_TAG, 
global_op_recv, 4, MPI_GRAPH_TYPE, mate, SR_LCG_TAG, 
comm_, MPI_STATUS_IGNORE);

matop_2x2(global_op, global_op_recv);   

if (mate < rank_) 
matop_2x2(prefix_op, global_op_recv);

MPI_Barrier(comm_);
}

if (rank_ == 0)
rnums_[0] = x0_;
else
rnums_[0] = (x0_*prefix_op[0] + prefix_op[2])%MLCG;
}

void generate()
{
#if defined(PRINT_LCG_LONG_RANDOM_NUMBERS)
for (int k = 0; k < nprocs_; k++) {
if (k == rank_) {
std::cout << "------------" << std::endl;
std::cout << "Process#" << rank_ << " :" << std::endl;
std::cout << "------------" << std::endl;
std::cout << rnums_[0] << std::endl;
for (GraphElem i = 1; i < n_; i++) {
rnums_[i] = (rnums_[i-1]*ALCG + BLCG)%MLCG;
std::cout << rnums_[i] << std::endl;
}
}
MPI_Barrier(comm_);
}
#else
for (GraphElem i = 1; i < n_; i++) {
rnums_[i] = (rnums_[i-1]*ALCG + BLCG)%MLCG;
}
#endif
GraphWeight mult = 1.0 / (GraphWeight)(1.0 + (GraphWeight)(MLCG-1));

#if defined(PRINT_LCG_DOUBLE_RANDOM_NUMBERS)
for (int k = 0; k < nprocs_; k++) {
if (k == rank_) {
std::cout << "------------" << std::endl;
std::cout << "Process#" << rank_ << " :" << std::endl;
std::cout << "------------" << std::endl;

for (GraphElem i = 0; i < n_; i++) {
drand_[i] = (GraphWeight)((GraphWeight)fabs(rnums_[i]) * mult ); 
std::cout << drand_[i] << std::endl;
}
}
MPI_Barrier(comm_);
}
#else
for (GraphElem i = 0; i < n_; i++)
drand_[i] = (GraphWeight)((GraphWeight)fabs(rnums_[i]) * mult); 
#endif
}

void rescale(GraphWeight* new_drand, GraphElem idx_start, GraphWeight const& lo)
{
GraphWeight range = (1.0 / (GraphWeight)nprocs_);

#if defined(PRINT_LCG_DOUBLE_LOHI_RANDOM_NUMBERS)
for (int k = 0; k < nprocs_; k++) {
if (k == rank_) {
std::cout << "------------" << std::endl;
std::cout << "Process#" << rank_ << " :" << std::endl;
std::cout << "------------" << std::endl;

for (GraphElem i = idx_start, j = 0; i < n_; i++, j++) {
new_drand[j] = lo + (GraphWeight)(range * drand_[i]);
std::cout << new_drand[j] << std::endl;
}
}
MPI_Barrier(comm_);
}
#else
for (GraphElem i = idx_start, j = 0; i < n_; i++, j++)
new_drand[j] = lo + (GraphWeight)(range * drand_[i]); 
#endif
}

private:
MPI_Comm comm_;
int nprocs_, rank_;
unsigned seed_;
GraphElem n_, x0_;
GraphWeight* drand_;
std::vector<GraphElem> rnums_;
};
#endif

#ifndef SSTMAC
#ifdef USE_OPENMP_LOCK
#else
#ifdef USE_SPINLOCK 
#include <atomic>
std::atomic_flag lkd_ = ATOMIC_FLAG_INIT;
#else
#include <mutex>
std::mutex mtx_;
#endif
void lock() {
#ifdef USE_SPINLOCK 
while (lkd_.test_and_set(std::memory_order_acquire)) { ; } 
#else
mtx_.lock();
#endif
}
void unlock() { 
#ifdef USE_SPINLOCK 
lkd_.clear(std::memory_order_release); 
#else
mtx_.unlock();
#endif
}
#endif 
#endif

#endif 
