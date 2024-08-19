

#ifndef BOOST_MULTI_INDEX_DETAIL_HASH_INDEX_ITERATOR_HPP
#define BOOST_MULTI_INDEX_DETAIL_HASH_INDEX_ITERATOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/operators.hpp>

#if !defined(BOOST_MULTI_INDEX_DISABLE_SERIALIZATION)
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>
#endif

namespace boost{

namespace multi_index{

namespace detail{



struct hashed_index_global_iterator_tag{};
struct hashed_index_local_iterator_tag{};

template<
typename Node,typename BucketArray,
typename IndexCategory,typename IteratorCategory
>
class hashed_index_iterator:
public forward_iterator_helper<
hashed_index_iterator<Node,BucketArray,IndexCategory,IteratorCategory>,
typename Node::value_type,
typename Node::difference_type,
const typename Node::value_type*,
const typename Node::value_type&>
{
public:

hashed_index_iterator(){}
hashed_index_iterator(Node* node_):node(node_){}

const typename Node::value_type& operator*()const
{
return node->value();
}

hashed_index_iterator& operator++()
{
this->increment(IteratorCategory());
return *this;
}

#if !defined(BOOST_MULTI_INDEX_DISABLE_SERIALIZATION)


BOOST_SERIALIZATION_SPLIT_MEMBER()

typedef typename Node::base_type node_base_type;

template<class Archive>
void save(Archive& ar,const unsigned int)const
{
node_base_type* bnode=node;
ar<<serialization::make_nvp("pointer",bnode);
}

template<class Archive>
void load(Archive& ar,const unsigned int version)
{
load(ar,version,IteratorCategory());
}

template<class Archive>
void load(
Archive& ar,const unsigned int version,hashed_index_global_iterator_tag)
{
node_base_type* bnode;
ar>>serialization::make_nvp("pointer",bnode);
node=static_cast<Node*>(bnode);
if(version<1){
BucketArray* throw_away; 
ar>>serialization::make_nvp("pointer",throw_away);
}
}

template<class Archive>
void load(
Archive& ar,const unsigned int version,hashed_index_local_iterator_tag)
{
node_base_type* bnode;
ar>>serialization::make_nvp("pointer",bnode);
node=static_cast<Node*>(bnode);
if(version<1){
BucketArray* buckets;
ar>>serialization::make_nvp("pointer",buckets);
if(buckets&&node&&node->impl()==buckets->end()->prior()){

node=0;
}
}
}
#endif



typedef Node node_type;

Node* get_node()const{return node;}

private:

void increment(hashed_index_global_iterator_tag)
{
Node::template increment<IndexCategory>(node);
}

void increment(hashed_index_local_iterator_tag)
{
Node::template increment_local<IndexCategory>(node);
}

Node* node;
};

template<
typename Node,typename BucketArray,
typename IndexCategory,typename IteratorCategory
>
bool operator==(
const hashed_index_iterator<
Node,BucketArray,IndexCategory,IteratorCategory>& x,
const hashed_index_iterator<
Node,BucketArray,IndexCategory,IteratorCategory>& y)
{
return x.get_node()==y.get_node();
}

} 

} 

#if !defined(BOOST_MULTI_INDEX_DISABLE_SERIALIZATION)


namespace serialization {
template<
typename Node,typename BucketArray,
typename IndexCategory,typename IteratorCategory
>
struct version<
boost::multi_index::detail::hashed_index_iterator<
Node,BucketArray,IndexCategory,IteratorCategory
>
>
{
BOOST_STATIC_CONSTANT(int,value=1);
};
} 
#endif

} 

#endif
