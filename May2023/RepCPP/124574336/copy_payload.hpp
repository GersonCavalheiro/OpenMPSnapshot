
#ifndef BOOST_LOCKFREE_DETAIL_COPY_PAYLOAD_HPP_INCLUDED
#define BOOST_LOCKFREE_DETAIL_COPY_PAYLOAD_HPP_INCLUDED

#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_convertible.hpp>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4512) 
#endif

namespace boost    {
namespace lockfree {
namespace detail   {

struct copy_convertible
{
template <typename T, typename U>
static void copy(T & t, U & u)
{
u = t;
}
};

struct copy_constructible_and_copyable
{
template <typename T, typename U>
static void copy(T & t, U & u)
{
u = U(t);
}
};

template <typename T, typename U>
void copy_payload(T & t, U & u)
{
typedef typename boost::mpl::if_<typename boost::is_convertible<T, U>::type,
copy_convertible,
copy_constructible_and_copyable
>::type copy_type;
copy_type::copy(t, u);
}

template <typename T>
struct consume_via_copy
{
consume_via_copy(T & out):
out_(out)
{}

template <typename U>
void operator()(U & element)
{
copy_payload(element, out_);
}

T & out_;
};

struct consume_noop
{
template <typename U>
void operator()(const U &)
{
}
};


}}}

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif  
