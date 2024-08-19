
#ifndef BOOST_XPRESSIVE_DETAIL_UTILITY_SAVE_RESTORE_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_UTILITY_SAVE_RESTORE_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/noncopyable.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename T>
struct save_restore
: private noncopyable
{
explicit save_restore(T &t)
: ref(t)
, val(t)
{
}

save_restore(T &t, T const &n)
: ref(t)
, val(t)
{
this->ref = n;
}

~save_restore()
{
this->ref = this->val;
}

void restore()
{
this->ref = this->val;
}

private:
T &ref;
T const val;
};

}}}

#endif
