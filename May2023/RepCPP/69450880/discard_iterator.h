




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/discard_iterator_base.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_facade.h>

HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

namespace hydra_thrust
{






template<typename System = use_default>
class discard_iterator
: public detail::discard_iterator_base<System>::type
{

friend class hydra_thrust::iterator_core_access;
typedef typename detail::discard_iterator_base<System>::type          super_t;
typedef typename detail::discard_iterator_base<System>::incrementable incrementable;
typedef typename detail::discard_iterator_base<System>::base_iterator base_iterator;

public:
typedef typename super_t::reference  reference;
typedef typename super_t::value_type value_type;




__host__ __device__
discard_iterator(discard_iterator const &rhs)
: super_t(rhs.base()) {}


__host__ __device__
discard_iterator(incrementable const &i = incrementable())
: super_t(base_iterator(i)) {}



private: 
__host__ __device__
reference dereference() const
{
return m_element;
}

mutable value_type m_element;


}; 



inline __host__ __device__
discard_iterator<> make_discard_iterator(discard_iterator<>::difference_type i = discard_iterator<>::difference_type(0))
{
return discard_iterator<>(i);
} 





} 

HYDRA_THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

