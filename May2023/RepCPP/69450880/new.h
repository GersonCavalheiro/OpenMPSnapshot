



#pragma once

#include <hydra/detail/external/hydra_thrust/mr/memory_resource.h>

namespace hydra_thrust
{
namespace mr
{




class new_delete_resource HYDRA_THRUST_FINAL : public memory_resource<>
{
public:
void * do_allocate(std::size_t bytes, std::size_t alignment = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT) HYDRA_THRUST_OVERRIDE
{
#if __cplusplus >= 201703L
return ::operator new(bytes, std::align_val_t(alignment));
#else
void * p = ::operator new(bytes + alignment + sizeof(std::size_t));
std::size_t ptr_int = reinterpret_cast<std::size_t>(p);
std::size_t offset = (ptr_int % alignment) ? (alignment - ptr_int % alignment) : 0;
char * ptr = static_cast<char *>(p) + offset;
std::size_t * offset_store = reinterpret_cast<std::size_t *>(ptr + bytes);
*offset_store = offset;
return static_cast<void *>(ptr);
#endif
}

void do_deallocate(void * p, std::size_t bytes, std::size_t alignment = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT) HYDRA_THRUST_OVERRIDE
{
#if __cplusplus >= 201703L
::operator delete(p, bytes, std::align_val_t(alignment));
#else
(void)alignment;
char * ptr = static_cast<char *>(p);
std::size_t * offset = reinterpret_cast<std::size_t *>(ptr + bytes);
p = static_cast<void *>(ptr - *offset);
::operator delete(p);
#endif
}
};



} 
} 

