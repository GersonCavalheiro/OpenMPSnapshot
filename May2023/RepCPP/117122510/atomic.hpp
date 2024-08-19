

#ifndef GKO_OMP_COMPONENTS_ATOMIC_HPP_
#define GKO_OMP_COMPONENTS_ATOMIC_HPP_


#include <type_traits>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace omp {


template <typename ValueType,
std::enable_if_t<!is_complex<ValueType>()>* = nullptr>
void atomic_add(ValueType& out, ValueType val)
{
#pragma omp atomic
out += val;
}

template <typename ValueType,
std::enable_if_t<is_complex<ValueType>()>* = nullptr>
void atomic_add(ValueType& out, ValueType val)
{
auto values = reinterpret_cast<gko::remove_complex<ValueType>*>(&out);
#pragma omp atomic
values[0] += real(val);
#pragma omp atomic
values[1] += imag(val);
}


}  
}  
}  


#endif  
