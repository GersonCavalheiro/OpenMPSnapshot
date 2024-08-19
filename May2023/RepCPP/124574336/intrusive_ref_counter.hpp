


#ifndef BOOST_SMART_PTR_INTRUSIVE_REF_COUNTER_HPP_INCLUDED_
#define BOOST_SMART_PTR_INTRUSIVE_REF_COUNTER_HPP_INCLUDED_

#include <boost/config.hpp>
#include <boost/smart_ptr/detail/atomic_count.hpp>
#include <boost/smart_ptr/detail/sp_noexcept.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4396)
#endif

namespace boost {

namespace sp_adl_block {


struct thread_unsafe_counter
{
typedef unsigned int type;

static unsigned int load(unsigned int const& counter) BOOST_SP_NOEXCEPT
{
return counter;
}

static void increment(unsigned int& counter) BOOST_SP_NOEXCEPT
{
++counter;
}

static unsigned int decrement(unsigned int& counter) BOOST_SP_NOEXCEPT
{
return --counter;
}
};


struct thread_safe_counter
{
typedef boost::detail::atomic_count type;

static unsigned int load(boost::detail::atomic_count const& counter) BOOST_SP_NOEXCEPT
{
return static_cast< unsigned int >(static_cast< long >(counter));
}

static void increment(boost::detail::atomic_count& counter) BOOST_SP_NOEXCEPT
{
++counter;
}

static unsigned int decrement(boost::detail::atomic_count& counter) BOOST_SP_NOEXCEPT
{
return static_cast< unsigned int >(--counter);
}
};

template< typename DerivedT, typename CounterPolicyT = thread_safe_counter >
class intrusive_ref_counter;

template< typename DerivedT, typename CounterPolicyT >
void intrusive_ptr_add_ref(const intrusive_ref_counter< DerivedT, CounterPolicyT >* p) BOOST_SP_NOEXCEPT;
template< typename DerivedT, typename CounterPolicyT >
void intrusive_ptr_release(const intrusive_ref_counter< DerivedT, CounterPolicyT >* p) BOOST_SP_NOEXCEPT;


template< typename DerivedT, typename CounterPolicyT >
class intrusive_ref_counter
{
private:
typedef typename CounterPolicyT::type counter_type;
mutable counter_type m_ref_counter;

public:

intrusive_ref_counter() BOOST_SP_NOEXCEPT : m_ref_counter(0)
{
}


intrusive_ref_counter(intrusive_ref_counter const&) BOOST_SP_NOEXCEPT : m_ref_counter(0)
{
}


intrusive_ref_counter& operator= (intrusive_ref_counter const&) BOOST_SP_NOEXCEPT { return *this; }


unsigned int use_count() const BOOST_SP_NOEXCEPT
{
return CounterPolicyT::load(m_ref_counter);
}

protected:

BOOST_DEFAULTED_FUNCTION(~intrusive_ref_counter(), {})

friend void intrusive_ptr_add_ref< DerivedT, CounterPolicyT >(const intrusive_ref_counter< DerivedT, CounterPolicyT >* p) BOOST_SP_NOEXCEPT;
friend void intrusive_ptr_release< DerivedT, CounterPolicyT >(const intrusive_ref_counter< DerivedT, CounterPolicyT >* p) BOOST_SP_NOEXCEPT;
};

template< typename DerivedT, typename CounterPolicyT >
inline void intrusive_ptr_add_ref(const intrusive_ref_counter< DerivedT, CounterPolicyT >* p) BOOST_SP_NOEXCEPT
{
CounterPolicyT::increment(p->m_ref_counter);
}

template< typename DerivedT, typename CounterPolicyT >
inline void intrusive_ptr_release(const intrusive_ref_counter< DerivedT, CounterPolicyT >* p) BOOST_SP_NOEXCEPT
{
if (CounterPolicyT::decrement(p->m_ref_counter) == 0)
delete static_cast< const DerivedT* >(p);
}

} 

using sp_adl_block::intrusive_ref_counter;
using sp_adl_block::thread_unsafe_counter;
using sp_adl_block::thread_safe_counter;

} 

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif 
