

#ifndef BOOST_FLYWEIGHT_DETAIL_SERIALIZATION_HELPER_HPP
#define BOOST_FLYWEIGHT_DETAIL_SERIALIZATION_HELPER_HPP

#if defined(_MSC_VER)&&(_MSC_VER>=1200)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/noncopyable.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <vector>

namespace boost{

namespace flyweights{

namespace detail{



template<typename Flyweight>
struct flyweight_value_address
{
typedef const typename Flyweight::value_type* result_type;

result_type operator()(const Flyweight& x)const{return &x.get();}
};

template<typename Flyweight>
class save_helper:private noncopyable
{
typedef multi_index::multi_index_container<
Flyweight,
multi_index::indexed_by<
multi_index::random_access<>,
multi_index::hashed_unique<flyweight_value_address<Flyweight> >
>
> table;

public:

typedef typename table::size_type size_type;

size_type size()const{return t.size();}

size_type find(const Flyweight& x)const
{
return multi_index::project<0>(t,multi_index::get<1>(t).find(&x.get()))
-t.begin();
}

void push_back(const Flyweight& x){t.push_back(x);}

private:
table t;
};

template<typename Flyweight>
class load_helper:private noncopyable
{
typedef std::vector<Flyweight> table;

public:

typedef typename table::size_type size_type;

size_type size()const{return t.size();}

Flyweight operator[](size_type n)const{return t[n];}

void push_back(const Flyweight& x){t.push_back(x);}

private:
table t;
};

} 

} 

} 

#endif
