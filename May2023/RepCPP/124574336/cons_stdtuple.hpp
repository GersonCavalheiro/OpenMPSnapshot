

#ifndef BOOST_MULTI_INDEX_DETAIL_CONS_STDTUPLE_HPP
#define BOOST_MULTI_INDEX_DETAIL_CONS_STDTUPLE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/mpl/if.hpp>
#include <boost/tuple/tuple.hpp>
#include <tuple>

namespace boost{

namespace multi_index{

namespace detail{



template<typename StdTuple,std::size_t N>
struct cons_stdtuple;

struct cons_stdtuple_ctor_terminal
{
typedef boost::tuples::null_type result_type;

template<typename StdTuple>
static result_type create(const StdTuple&)
{
return boost::tuples::null_type();
}
};

template<typename StdTuple,std::size_t N>
struct cons_stdtuple_ctor_normal
{
typedef cons_stdtuple<StdTuple,N> result_type;

static result_type create(const StdTuple& t)
{
return result_type(t);
}
};

template<typename StdTuple,std::size_t N=0>
struct cons_stdtuple_ctor:
boost::mpl::if_c<
N<std::tuple_size<StdTuple>::value,
cons_stdtuple_ctor_normal<StdTuple,N>,
cons_stdtuple_ctor_terminal
>::type
{};

template<typename StdTuple,std::size_t N>
struct cons_stdtuple
{
typedef typename std::tuple_element<N,StdTuple>::type head_type;
typedef cons_stdtuple_ctor<StdTuple,N+1>              tail_ctor;
typedef typename tail_ctor::result_type               tail_type;

cons_stdtuple(const StdTuple& t_):t(t_){}

const head_type& get_head()const{return std::get<N>(t);}
tail_type get_tail()const{return tail_ctor::create(t);}

const StdTuple& t;
};

template<typename StdTuple>
typename cons_stdtuple_ctor<StdTuple>::result_type
make_cons_stdtuple(const StdTuple& t)
{
return cons_stdtuple_ctor<StdTuple>::create(t);
}

} 

} 

} 

#endif
