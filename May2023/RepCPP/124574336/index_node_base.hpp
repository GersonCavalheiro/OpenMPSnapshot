

#ifndef BOOST_MULTI_INDEX_DETAIL_INDEX_NODE_BASE_HPP
#define BOOST_MULTI_INDEX_DETAIL_INDEX_NODE_BASE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/type_traits/aligned_storage.hpp>
#include <boost/type_traits/alignment_of.hpp> 

#if !defined(BOOST_MULTI_INDEX_DISABLE_SERIALIZATION)
#include <boost/archive/archive_exception.hpp>
#include <boost/serialization/access.hpp>
#include <boost/throw_exception.hpp> 
#endif

namespace boost{

namespace multi_index{

namespace detail{



template<typename Value>
struct pod_value_holder
{
typename aligned_storage<
sizeof(Value),
alignment_of<Value>::value
>::type                      space;
};

template<typename Value,typename Allocator>
struct index_node_base:private pod_value_holder<Value>
{
typedef index_node_base base_type; 
typedef Value           value_type;
typedef Allocator       allocator_type;

#include <boost/multi_index/detail/ignore_wstrict_aliasing.hpp>

value_type& value()
{
return *reinterpret_cast<value_type*>(&this->space);
}

const value_type& value()const
{
return *reinterpret_cast<const value_type*>(&this->space);
}

#include <boost/multi_index/detail/restore_wstrict_aliasing.hpp>

static index_node_base* from_value(const value_type* p)
{
return static_cast<index_node_base *>(
reinterpret_cast<pod_value_holder<Value>*>( 
const_cast<value_type*>(p))); 
}

private:
#if !defined(BOOST_MULTI_INDEX_DISABLE_SERIALIZATION)
friend class boost::serialization::access;



template<class Archive>
void serialize(Archive&,const unsigned int)
{
}
#endif
};

template<typename Node,typename Value>
Node* node_from_value(const Value* p)
{
typedef typename Node::allocator_type allocator_type;
return static_cast<Node*>(
index_node_base<Value,allocator_type>::from_value(p));
}

} 

} 

#if !defined(BOOST_MULTI_INDEX_DISABLE_SERIALIZATION)


#if defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
namespace serialization{
#else
namespace multi_index{
namespace detail{
#endif

template<class Archive,typename Value,typename Allocator>
inline void load_construct_data(
Archive&,boost::multi_index::detail::index_node_base<Value,Allocator>*,
const unsigned int)
{
throw_exception(
archive::archive_exception(archive::archive_exception::other_exception));
}

#if defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
} 
#else
} 
} 
#endif

#endif

} 

#endif
