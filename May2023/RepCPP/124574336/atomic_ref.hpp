


#ifndef BOOST_ATOMIC_ATOMIC_REF_HPP_INCLUDED_
#define BOOST_ATOMIC_ATOMIC_REF_HPP_INCLUDED_

#include <boost/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/memory_order.hpp>
#include <boost/atomic/capabilities.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/intptr.hpp>
#include <boost/atomic/detail/classify.hpp>
#include <boost/atomic/detail/atomic_ref_impl.hpp>
#include <boost/atomic/detail/type_traits/is_trivially_copyable.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {

template< typename T >
class atomic_ref :
public atomics::detail::base_atomic_ref< T, typename atomics::detail::classify< T >::type, false >
{
private:
typedef atomics::detail::base_atomic_ref< T, typename atomics::detail::classify< T >::type, false > base_type;
typedef typename base_type::value_arg_type value_arg_type;

public:
typedef typename base_type::value_type value_type;

BOOST_STATIC_ASSERT_MSG(sizeof(value_type) > 0u, "boost::atomic_ref<T> requires T to be a complete type");
#if !defined(BOOST_ATOMIC_DETAIL_NO_CXX11_IS_TRIVIALLY_COPYABLE)
BOOST_STATIC_ASSERT_MSG(atomics::detail::is_trivially_copyable< value_type >::value, "boost::atomic_ref<T> requires T to be a trivially copyable type");
#endif

private:
typedef typename base_type::storage_type storage_type;

public:
BOOST_DEFAULTED_FUNCTION(atomic_ref(atomic_ref const& that) BOOST_ATOMIC_DETAIL_DEF_NOEXCEPT_DECL, BOOST_ATOMIC_DETAIL_DEF_NOEXCEPT_IMPL : base_type(static_cast< base_type const& >(that)) {})
BOOST_FORCEINLINE explicit atomic_ref(value_type& v) BOOST_NOEXCEPT : base_type(v)
{
BOOST_ASSERT((((atomics::detail::uintptr_t)this->m_value) & (base_type::required_alignment - 1u)) == 0u);
}

BOOST_FORCEINLINE value_type operator= (value_arg_type v) const BOOST_NOEXCEPT
{
this->store(v);
return v;
}

BOOST_FORCEINLINE operator value_type() const BOOST_NOEXCEPT
{
return this->load();
}

BOOST_DELETED_FUNCTION(atomic_ref& operator= (atomic_ref const&))
};

} 

using atomics::atomic_ref;

} 

#include <boost/atomic/detail/footer.hpp>

#endif 
