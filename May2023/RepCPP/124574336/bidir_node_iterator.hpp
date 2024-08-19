

#ifndef BOOST_MULTI_INDEX_DETAIL_BIDIR_NODE_ITERATOR_HPP
#define BOOST_MULTI_INDEX_DETAIL_BIDIR_NODE_ITERATOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/operators.hpp>

#if !defined(BOOST_MULTI_INDEX_DISABLE_SERIALIZATION)
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_member.hpp>
#endif

namespace boost{

namespace multi_index{

namespace detail{



template<typename Node>
class bidir_node_iterator:
public bidirectional_iterator_helper<
bidir_node_iterator<Node>,
typename Node::value_type,
typename Node::difference_type,
const typename Node::value_type*,
const typename Node::value_type&>
{
public:

bidir_node_iterator(){}
explicit bidir_node_iterator(Node* node_):node(node_){}

const typename Node::value_type& operator*()const
{
return node->value();
}

bidir_node_iterator& operator++()
{
Node::increment(node);
return *this;
}

bidir_node_iterator& operator--()
{
Node::decrement(node);
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
void load(Archive& ar,const unsigned int)
{
node_base_type* bnode;
ar>>serialization::make_nvp("pointer",bnode);
node=static_cast<Node*>(bnode);
}
#endif



typedef Node node_type;

Node* get_node()const{return node;}

private:
Node* node;
};

template<typename Node>
bool operator==(
const bidir_node_iterator<Node>& x,
const bidir_node_iterator<Node>& y)
{
return x.get_node()==y.get_node();
}

} 

} 

} 

#endif
