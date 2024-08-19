



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#include <mutex>

#include <hydra/detail/external/hydra_thrust/mr/pool.h>

namespace hydra_thrust
{
namespace mr
{




template<typename Upstream>
struct synchronized_pool_resource : public memory_resource<typename Upstream::pointer>
{
typedef unsynchronized_pool_resource<Upstream> unsync_pool;
typedef std::lock_guard<std::mutex> lock_t;

typedef typename Upstream::pointer void_ptr;

public:

static pool_options get_default_options()
{
return unsync_pool::get_default_options();
}


synchronized_pool_resource(Upstream * upstream, pool_options options = get_default_options())
: upstream_pool(upstream, options)
{
}


synchronized_pool_resource(pool_options options = get_default_options())
: upstream_pool(get_global_resource<Upstream>(), options)
{
}


void release()
{
lock_t lock(mtx);
upstream_pool.release();
}

HYDRA_THRUST_NODISCARD virtual void_ptr do_allocate(std::size_t bytes, std::size_t alignment = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT) HYDRA_THRUST_OVERRIDE
{
lock_t lock(mtx);
return upstream_pool.do_allocate(bytes, alignment);
}

virtual void do_deallocate(void_ptr p, std::size_t n, std::size_t alignment = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT) HYDRA_THRUST_OVERRIDE
{
lock_t lock(mtx);
upstream_pool.do_deallocate(p, n, alignment);
}

private:
std::mutex mtx;
unsync_pool upstream_pool;
};



} 
} 

#endif 

