
#ifndef BOOST_INTERPROCESS_ENABLE_SHARED_FROM_THIS_HPP_INCLUDED
#define BOOST_INTERPROCESS_ENABLE_SHARED_FROM_THIS_HPP_INCLUDED

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/assert.hpp>
#include <boost/interprocess/smart_ptr/weak_ptr.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>


namespace boost{
namespace interprocess{

template<class T, class A, class D>
class enable_shared_from_this
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
protected:
enable_shared_from_this()
{}

enable_shared_from_this(enable_shared_from_this const &)
{}

enable_shared_from_this & operator=(enable_shared_from_this const &)
{  return *this;  }

~enable_shared_from_this()
{}
#endif   

public:
shared_ptr<T, A, D> shared_from_this()
{
shared_ptr<T, A, D> p(_internal_weak_this);
BOOST_ASSERT(ipcdetail::to_raw_pointer(p.get()) == this);
return p;
}

shared_ptr<T const, A, D> shared_from_this() const
{
shared_ptr<T const, A, D> p(_internal_weak_this);
BOOST_ASSERT(ipcdetail::to_raw_pointer(p.get()) == this);
return p;
}

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
typedef T element_type;
mutable weak_ptr<element_type, A, D> _internal_weak_this;
#endif   
};

} 
} 

#include <boost/interprocess/detail/config_end.hpp>

#endif  

