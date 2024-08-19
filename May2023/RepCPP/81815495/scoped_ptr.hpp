
#ifndef ASIO_DETAIL_SCOPED_PTR_HPP
#define ASIO_DETAIL_SCOPED_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T>
class scoped_ptr
{
public:
explicit scoped_ptr(T* p = 0)
: p_(p)
{
}

~scoped_ptr()
{
delete p_;
}

T* get()
{
return p_;
}

T* operator->()
{
return p_;
}

T& operator*()
{
return *p_;
}

void reset(T* p = 0)
{
delete p_;
p_ = p;
}

T* release()
{
T* tmp = p_;
p_ = 0;
return tmp;
}

private:
scoped_ptr(const scoped_ptr&);
scoped_ptr& operator=(const scoped_ptr&);

T* p_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
