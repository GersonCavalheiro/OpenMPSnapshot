
#ifndef BOOST_INTERPROCESS_DELETER_HPP
#define BOOST_INTERPROCESS_DELETER_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/intrusive/pointer_traits.hpp>


namespace boost {
namespace interprocess {

template<class T, class SegmentManager>
class deleter
{
public:
typedef typename boost::intrusive::
pointer_traits<typename SegmentManager::void_pointer>::template
rebind_pointer<T>::type                pointer;

private:
typedef typename boost::intrusive::
pointer_traits<pointer>::template
rebind_pointer<SegmentManager>::type                segment_manager_pointer;

segment_manager_pointer mp_mngr;

public:
deleter(segment_manager_pointer pmngr)
:  mp_mngr(pmngr)
{}

void operator()(const pointer &p)
{  mp_mngr->destroy_ptr(ipcdetail::to_raw_pointer(p));   }
};

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
