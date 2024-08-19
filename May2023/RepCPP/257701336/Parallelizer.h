
#ifndef EIGEN_PARALLELIZER_H
#define EIGEN_PARALLELIZER_H

#if EIGEN_HAS_CXX11_ATOMIC
#include <atomic>
#endif

namespace Eigen {

namespace internal {


inline void manage_multi_threading(Action action, int* v)
{
static EIGEN_UNUSED int m_maxThreads = -1;

if(action==SetAction)
{
eigen_internal_assert(v!=0);
m_maxThreads = *v;
}
else if(action==GetAction)
{
eigen_internal_assert(v!=0);
#ifdef EIGEN_HAS_OPENMP
if(m_maxThreads>0)
*v = m_maxThreads;
else
*v = omp_get_max_threads();
#else
*v = 1;
#endif
}
else
{
eigen_internal_assert(false);
}
}

}


inline void initParallel()
{
int nbt;
internal::manage_multi_threading(GetAction, &nbt);
std::ptrdiff_t l1, l2, l3;
internal::manage_caching_sizes(GetAction, &l1, &l2, &l3);
}


inline int nbThreads()
{
int ret;
internal::manage_multi_threading(GetAction, &ret);
return ret;
}


inline void setNbThreads(int v)
{
internal::manage_multi_threading(SetAction, &v);
}

namespace internal {

template<typename Index> struct GemmParallelInfo
{
GemmParallelInfo() : sync(-1), users(0), lhs_start(0), lhs_length(0) {}

#if EIGEN_HAS_CXX11_ATOMIC
std::atomic<Index> sync;
std::atomic<int> users;
#else
Index volatile sync;
int volatile users;
#endif

Index lhs_start;
Index lhs_length;
};

template<bool Condition, typename Functor, typename Index>
void parallelize_gemm(const Functor& func, Index rows, Index cols, Index depth, bool transpose)
{
#if (! defined(EIGEN_HAS_OPENMP)) || defined(EIGEN_USE_BLAS) || ((!EIGEN_HAS_CXX11_ATOMIC) && !(EIGEN_ARCH_i386_OR_x86_64))
EIGEN_UNUSED_VARIABLE(depth);
EIGEN_UNUSED_VARIABLE(transpose);
func(0,rows, 0,cols);
#else


Index size = transpose ? rows : cols;
Index pb_max_threads = std::max<Index>(1,size / Functor::Traits::nr);

double work = static_cast<double>(rows) * static_cast<double>(cols) *
static_cast<double>(depth);
double kMinTaskSize = 50000;  
pb_max_threads = std::max<Index>(1, std::min<Index>(pb_max_threads, work / kMinTaskSize));

Index threads = std::min<Index>(nbThreads(), pb_max_threads);

if((!Condition) || (threads==1) || (omp_get_num_threads()>1))
return func(0,rows, 0,cols);

Eigen::initParallel();
func.initParallelSession(threads);

if(transpose)
std::swap(rows,cols);

ei_declare_aligned_stack_constructed_variable(GemmParallelInfo<Index>,info,threads,0);

#pragma omp parallel num_threads(threads)
{
Index i = omp_get_thread_num();
Index actual_threads = omp_get_num_threads();

Index blockCols = (cols / actual_threads) & ~Index(0x3);
Index blockRows = (rows / actual_threads);
blockRows = (blockRows/Functor::Traits::mr)*Functor::Traits::mr;

Index r0 = i*blockRows;
Index actualBlockRows = (i+1==actual_threads) ? rows-r0 : blockRows;

Index c0 = i*blockCols;
Index actualBlockCols = (i+1==actual_threads) ? cols-c0 : blockCols;

info[i].lhs_start = r0;
info[i].lhs_length = actualBlockRows;

if(transpose) func(c0, actualBlockCols, 0, rows, info);
else          func(0, rows, c0, actualBlockCols, info);
}
#endif
}

} 

} 

#endif 
