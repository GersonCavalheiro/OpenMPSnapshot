#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

#if defined (TS16)
#define TEAM_SIZE 16
#elif defined (TS32)
#define TEAM_SIZE 32
#elif defined (TS64)
#define TEAM_SIZE 64
#elif defined (TS128)
#define TEAM_SIZE 128
#elif defined (TS256)
#define TEAM_SIZE 256
#elif defined (TS512)
#define TEAM_SIZE 512
#else
#define TEAM_SIZE 32
#endif


#define PI                          (3.14159)
#define MAX_PRINT_NEDGE             (100000)

#define MLCG                        (2147483647)    
#define ALCG                        (16807)         
#define BLCG                        (0)

#ifndef NTIMES
#define NTIMES       (10)
#endif

#ifndef ALIGNMENT
#define ALIGNMENT    (16)
#endif
#ifndef DEFAULT_NV
#define DEFAULT_NV   (524288)
#endif

#ifndef GRAPH_FT_LOAD
#define GRAPH_FT_LOAD (1)
#endif

#include <random>
#include <utility>
#include <cstring>
#include <numeric>

#ifdef USE_32_BIT_GRAPH
using GraphElem = int32_t;
using GraphWeight = float;
#else
using GraphElem = int64_t;
using GraphWeight = double;
#endif

#ifdef EDGE_AS_VERTEX_PAIR
struct Edge
{   
GraphElem head_, tail_;
GraphWeight weight_;

Edge(): head_(-1), tail_(-1), weight_(-1.0) 
{}
};
#else
struct Edge
{   
GraphElem tail_;
GraphWeight weight_;

Edge(): tail_(-1), weight_(-1.0) {}
};
#endif

struct EdgeTuple
{
GraphElem ij_[2];
GraphWeight w_;

EdgeTuple(GraphElem i, GraphElem j, GraphWeight w): 
ij_{i, j}, w_(w)
{}
EdgeTuple(GraphElem i, GraphElem j): 
ij_{i, j}, w_(1.0) 
{}
EdgeTuple(): 
ij_{-1, -1}, w_(0.0)
{}
};

extern unsigned seed;

inline int is_pwr2(int pes) 
{ return ((pes != 0) && !(pes & (pes - 1))); }


inline bool is_same(GraphWeight a, GraphWeight b) 
{ return std::abs(a - b) <= std::numeric_limits<GraphWeight>::epsilon(); }


inline GraphElem reseeder(unsigned initseed)
{
std::seed_seq seq({initseed});
std::vector<std::uint32_t> seeds(1);
seq.generate(seeds.begin(), seeds.end());

return (GraphElem)seeds[0];
}

template<typename T, typename G = std::default_random_engine>
T genRandom(T lo, T hi)
{
thread_local static G gen(std::random_device{}());
using Dist = typename std::conditional
<
std::is_integral<T>::value
, std::uniform_int_distribution<T>
, std::uniform_real_distribution<T>
>::type;

thread_local static Dist utd {};
return utd(gen, typename Dist::param_type{lo, hi});
}


class LCG
{
public:
LCG(unsigned seed, GraphWeight* drand, GraphElem n) : 
seed_(seed), n_(n), drand_(drand)
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
drand_[i] = (GraphWeight)((GraphWeight)std::fabs(rnums_[i]) * mult ); 
std::cout << drand_[i] << std::endl;
}
#else
for (GraphElem i = 0; i < n_; i++)
drand_[i] = (GraphWeight)((GraphWeight)std::fabs(rnums_[i]) * mult); 
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

#ifdef USE_OPENMP_LOCK
#else
#ifdef USE_SPINLOCK 
#include <atomic>
std::atomic_flag lkd_ = ATOMIC_FLAG_INIT;
#else
#include <mutex>
extern std::mutex mtx_;
#endif
inline void lock() {
#ifdef USE_SPINLOCK 
while (lkd_.test_and_set(std::memory_order_acquire)) { ; } 
#else
mtx_.lock();
#endif
}
inline void unlock() { 
#ifdef USE_SPINLOCK 
lkd_.clear(std::memory_order_release); 
#else
mtx_.unlock();
#endif
}
#endif

#ifdef USE_OMP_OFFLOAD
template <typename T>
void ompMemcpy(T *dst, T *src, size_t length, const char* direction) 
{
int num_devices = omp_get_num_devices();
assert(num_devices > 0);

int host_device_num = omp_get_initial_device();

int gpu_device_num = omp_get_default_device();

int dst_device_num = gpu_device_num;
int src_device_num = host_device_num;

if (std::strncmp(direction, "D2H", 3) == 0) 
{
dst_device_num = host_device_num;
src_device_num = gpu_device_num;
}

omp_target_memcpy(dst, src, length, 0, 0, dst_device_num, src_device_num);
}
#endif
#endif 
