

#ifndef BOOST_MULTI_INDEX_DETAIL_RND_NODE_ITERATOR_HPP
#define BOOST_MULTI_INDEX_DETAIL_RND_NODE_ITERATOR_HPP

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
class rnd_node_iterator:
public random_access_iterator_helper<
rnd_node_iterator<Node>,
typename Node::value_type,
typename Node::difference_type,
const typename Node::value_type*,
const typename Node::value_type&>
{
public:

rnd_node_iterator(){}
explicit rnd_node_iterator(Node* node_):node(node_){}

const typename Node::value_type& operator*()const
{
return node->value();
}

rnd_node_iterator& operator++()
{
Node::increment(node);
return *this;
}

rnd_node_iterator& operator--()
{
Node::decrement(node);
return *this;
}

rnd_node_iterator& operator+=(typename Node::difference_type n)
{
Node::advance(node,n);
return *this;
}

rnd_node_iterator& operator-=(typename Node::difference_type n)
{
Node::advance(node,-n);
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
const rnd_node_iterator<Node>& x,
const rnd_node_iterator<Node>& y)
{
return x.get_node()==y.get_node();
}

template<typename Node>
bool operator<(
const rnd_node_iterator<Node>& x,
const rnd_node_iterator<Node>& y)
{
return Node::distance(x.get_node(),y.get_node())>0;
}

template<typename Node>
typename Node::difference_type operator-(
const rnd_node_iterator<Node>& x,
const rnd_node_iterator<Node>& y)
{
return Node::distance(y.get_node(),x.get_node());
}

} 

} 

} 

#endif
