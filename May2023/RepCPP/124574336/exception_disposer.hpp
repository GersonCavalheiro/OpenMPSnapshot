
#ifndef BOOST_INTRUSIVE_DETAIL_EXCEPTION_DISPOSER_HPP
#define BOOST_INTRUSIVE_DETAIL_EXCEPTION_DISPOSER_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/intrusive/detail/workaround.hpp>

namespace boost {
namespace intrusive {
namespace detail {

template<class Container, class Disposer>
class exception_disposer
{
Container *cont_;
Disposer  &disp_;

exception_disposer(const exception_disposer&);
exception_disposer &operator=(const exception_disposer&);

public:
exception_disposer(Container &cont, Disposer &disp)
:  cont_(&cont), disp_(disp)
{}

BOOST_INTRUSIVE_FORCEINLINE void release()
{  cont_ = 0;  }

~exception_disposer()
{
if(cont_){
cont_->clear_and_dispose(disp_);
}
}
};

template<class Container, class Disposer, class SizeType>
class exception_array_disposer
{
Container *cont_;
Disposer  &disp_;
SizeType  &constructed_;

exception_array_disposer(const exception_array_disposer&);
exception_array_disposer &operator=(const exception_array_disposer&);

public:

exception_array_disposer
(Container &cont, Disposer &disp, SizeType &constructed)
:  cont_(&cont), disp_(disp), constructed_(constructed)
{}

BOOST_INTRUSIVE_FORCEINLINE void release()
{  cont_ = 0;  }

~exception_array_disposer()
{
SizeType n = constructed_;
if(cont_){
while(n--){
cont_[n].clear_and_dispose(disp_);
}
}
}
};

}  
}  
}  

#endif 
