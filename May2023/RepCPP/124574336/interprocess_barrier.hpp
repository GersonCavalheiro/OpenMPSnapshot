
#include<boost/interprocess/exceptions.hpp>
#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace interprocess {

inline barrier::barrier(unsigned int count)
: m_threshold(count), m_count(count), m_generation(0)
{
if (count == 0)
throw std::invalid_argument("count cannot be zero.");
}

inline barrier::~barrier(){}

inline bool barrier::wait()
{
scoped_lock<interprocess_mutex> lock(m_mutex);
unsigned int gen = m_generation;

if (--m_count == 0){
m_generation++;
m_count = m_threshold;
m_cond.notify_all();
return true;
}

while (gen == m_generation){
m_cond.wait(lock);
}
return false;
}

}  
}  
