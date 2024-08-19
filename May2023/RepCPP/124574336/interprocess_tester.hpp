
#ifndef BOOST_INTERPROCESS_DETAIL_INTERPROCESS_TESTER_HPP
#define BOOST_INTERPROCESS_DETAIL_INTERPROCESS_TESTER_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost{
namespace interprocess{
namespace ipcdetail{

class interprocess_tester
{
public:
template<class T>
static void dont_close_on_destruction(T &t)
{  t.dont_close_on_destruction(); }
};

}  
}  
}  

#endif   

