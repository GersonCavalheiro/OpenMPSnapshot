

#ifndef BOOST_MOVE_DETAIL_DESTRUCT_N_HPP
#define BOOST_MOVE_DETAIL_DESTRUCT_N_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <cstddef>

namespace boost {
namespace movelib{

template<class T, class RandItUninit>
class destruct_n
{
public:
explicit destruct_n(RandItUninit raw)
: m_ptr(raw), m_size()
{}

void incr()
{
++m_size;
}

void incr(std::size_t n)
{
m_size += n;
}

void release()
{
m_size = 0u;
}

~destruct_n()
{
while(m_size--){
m_ptr[m_size].~T();
}
}
private:
RandItUninit m_ptr;
std::size_t m_size;
};

}} 

#endif 
