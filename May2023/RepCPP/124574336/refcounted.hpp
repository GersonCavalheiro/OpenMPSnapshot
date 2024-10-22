

#ifndef BOOST_FLYWEIGHT_REFCOUNTED_HPP
#define BOOST_FLYWEIGHT_REFCOUNTED_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <algorithm>
#include <boost/detail/atomic_count.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/flyweight/refcounted_fwd.hpp>
#include <boost/flyweight/tracking_tag.hpp>
#include <boost/utility/swap.hpp>

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#include <utility>
#endif



namespace boost{

namespace flyweights{

namespace detail{

template<typename Value,typename Key>
class refcounted_value
{
public:
explicit refcounted_value(const Value& x_):
x(x_),ref(0),del_ref(0)
{}

refcounted_value(const refcounted_value& r):
x(r.x),ref(0),del_ref(0)
{}

refcounted_value& operator=(const refcounted_value& r)
{
x=r.x;
return *this;
}

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
explicit refcounted_value(Value&& x_):
x(std::move(x_)),ref(0),del_ref(0)
{}

refcounted_value(refcounted_value&& r):
x(std::move(r.x)),ref(0),del_ref(0)
{}

refcounted_value& operator=(refcounted_value&& r)
{
x=std::move(r.x);
return *this;
}
#endif

operator const Value&()const{return x;}
operator const Key&()const{return x;}

#if !defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS)
private:
template<typename,typename> friend class refcounted_handle;
#endif

long count()const{return ref;}
long add_ref()const{return ++ref;}
bool release()const{return (--ref==0);}

void add_deleter()const{++del_ref;}
bool release_deleter()const{return (--del_ref==0);}

private:
Value                               x;
mutable boost::detail::atomic_count ref;
mutable long                        del_ref;
};

template<typename Handle,typename TrackingHelper>
class refcounted_handle
{
public:
explicit refcounted_handle(const Handle& h_):h(h_)
{
if(TrackingHelper::entry(*this).add_ref()==1){
TrackingHelper::entry(*this).add_deleter();
}
}

refcounted_handle(const refcounted_handle& x):h(x.h)
{
TrackingHelper::entry(*this).add_ref();
}

refcounted_handle& operator=(refcounted_handle x)
{
this->swap(x);
return *this;
}

~refcounted_handle()
{
if(TrackingHelper::entry(*this).release()){
TrackingHelper::erase(*this,check_erase);
}
}

operator const Handle&()const{return h;}

void swap(refcounted_handle& x)
{
std::swap(h,x.h);
}

private:
static bool check_erase(const refcounted_handle& x)
{
return TrackingHelper::entry(x).release_deleter();
}

Handle h;
};

template<typename Handle,typename TrackingHelper>
void swap(
refcounted_handle<Handle,TrackingHelper>& x,
refcounted_handle<Handle,TrackingHelper>& y)
{
x.swap(y);
}

} 

#if BOOST_WORKAROUND(BOOST_MSVC,<=1500)


} 

template<typename Handle,typename TrackingHelper>
void swap(
::boost::flyweights::detail::refcounted_handle<Handle,TrackingHelper>& x,
::boost::flyweights::detail::refcounted_handle<Handle,TrackingHelper>& y)
{
::boost::flyweights::detail::swap(x,y);
}

namespace flyweights{
#endif

struct refcounted:tracking_marker
{
struct entry_type
{
template<typename Value,typename Key>
struct apply
{
typedef detail::refcounted_value<Value,Key> type;
};
};

struct handle_type
{
template<typename Handle,typename TrackingHelper>
struct apply
{
typedef detail::refcounted_handle<Handle,TrackingHelper> type;
};
};
};

} 

} 

#endif
