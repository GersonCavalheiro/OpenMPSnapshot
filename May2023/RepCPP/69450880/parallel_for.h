
#pragma once


#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC
#include <hydra/detail/external/hydra_thrust/system/cuda/config.h>

#include <hydra/detail/external/hydra_thrust/system/cuda/detail/util.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/result_of_adaptable_function.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/par_to_seq.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/core/agent_launcher.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/par_to_seq.h>

HYDRA_THRUST_BEGIN_NS

namespace cuda_cub {

namespace __parallel_for {

template <int _BLOCK_THREADS,
int _ITEMS_PER_THREAD = 1,
int _MIN_BLOCKS       = 1>
struct PtxPolicy
{
enum
{
BLOCK_THREADS    = _BLOCK_THREADS,
ITEMS_PER_THREAD = _ITEMS_PER_THREAD,
ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD,
MIN_BLOCKS       = _MIN_BLOCKS
};
};    

template <class Arch, class F>
struct Tuning;

template <class F>
struct Tuning<sm30, F>
{
typedef PtxPolicy<256, 2> type;
};


template <class F,
class Size>
struct ParallelForAgent
{
template <class Arch>
struct PtxPlan : Tuning<Arch, F>::type
{
typedef Tuning<Arch, F> tuning;
};
typedef core::specialize_plan<PtxPlan> ptx_plan;

enum
{
ITEMS_PER_THREAD = ptx_plan::ITEMS_PER_THREAD,
ITEMS_PER_TILE   = ptx_plan::ITEMS_PER_TILE,
BLOCK_THREADS    = ptx_plan::BLOCK_THREADS
};

template <bool IS_FULL_TILE>
static void    HYDRA_THRUST_DEVICE_FUNCTION
consume_tile(F    f,
Size tile_base,
int  items_in_tile)
{
#pragma unroll
for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
{
Size idx = BLOCK_THREADS * ITEM + threadIdx.x;
if (IS_FULL_TILE || idx < items_in_tile)
f(tile_base + idx);
}
}

HYDRA_THRUST_AGENT_ENTRY(F     f,
Size  num_items,
char *  )
{
Size tile_base     = static_cast<Size>(blockIdx.x) * ITEMS_PER_TILE;
Size num_remaining = num_items - tile_base;
Size items_in_tile = static_cast<Size>(
num_remaining < ITEMS_PER_TILE ? num_remaining : ITEMS_PER_TILE);

if (items_in_tile == ITEMS_PER_TILE)
{
consume_tile<true>(f, tile_base, ITEMS_PER_TILE);
}
else
{
consume_tile<false>(f, tile_base, items_in_tile);
}
}
};    

template <class F,
class Size>
HYDRA_THRUST_RUNTIME_FUNCTION cudaError_t
parallel_for(Size         num_items,
F            f,
cudaStream_t stream)
{
if (num_items == 0)
return cudaSuccess;
using core::AgentLauncher;
using core::AgentPlan;

bool debug_sync = HYDRA_THRUST_DEBUG_SYNC_FLAG;

typedef AgentLauncher<ParallelForAgent<F, Size> > parallel_for_agent;
AgentPlan parallel_for_plan = parallel_for_agent::get_plan(stream);

parallel_for_agent pfa(parallel_for_plan, num_items, stream, "transform::agent", debug_sync);
pfa.launch(f, num_items);
CUDA_CUB_RET_IF_FAIL(cudaPeekAtLastError());

return cudaSuccess;
}
}    

__hydra_thrust_exec_check_disable__ 
template <class Derived,
class F,
class Size>
void __host__ __device__
parallel_for(execution_policy<Derived> &policy,
F                          f,
Size                       count)
{
if (count == 0)
return;

if (__HYDRA_THRUST_HAS_CUDART__)
{
cudaStream_t stream = cuda_cub::stream(policy);
cudaError_t  status = __parallel_for::parallel_for(count, f, stream);
cuda_cub::throw_on_error(status, "parallel_for failed");
}
else
{
#if !__HYDRA_THRUST_HAS_CUDART__
for (Size idx = 0; idx != count; ++idx)
f(idx);
#endif
}
}

}    

HYDRA_THRUST_END_NS
#endif
