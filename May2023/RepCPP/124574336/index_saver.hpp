

#ifndef BOOST_MULTI_INDEX_DETAIL_INDEX_SAVER_HPP
#define BOOST_MULTI_INDEX_DETAIL_INDEX_SAVER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/multi_index/detail/index_matcher.hpp>
#include <boost/noncopyable.hpp>
#include <boost/serialization/nvp.hpp>
#include <cstddef>

namespace boost{

namespace multi_index{

namespace detail{



template<typename Node,typename Allocator>
class index_saver:private noncopyable
{
public:
index_saver(const Allocator& al,std::size_t size):alg(al,size){}

template<class Archive>
void add(Node* node,Archive& ar,const unsigned int)
{
ar<<serialization::make_nvp("position",*node);
alg.add(node);
}

template<class Archive>
void add_track(Node* node,Archive& ar,const unsigned int)
{
ar<<serialization::make_nvp("position",*node);
}

template<typename IndexIterator,class Archive>
void save(
IndexIterator first,IndexIterator last,Archive& ar,
const unsigned int)const
{


alg.execute(first,last);



std::size_t last_saved=3; 
for(IndexIterator it=first,prev=first;it!=last;prev=it++,++last_saved){
if(!alg.is_ordered(get_node(it))){
if(last_saved>1)save_node(get_node(prev),ar);
save_node(get_node(it),ar);
last_saved=0;
}
else if(last_saved==2)save_node(null_node(),ar);
}
if(last_saved<=2)save_node(null_node(),ar);



save_node(null_node(),ar);
}

private:
template<typename IndexIterator>
static Node* get_node(IndexIterator it)
{
return it.get_node();
}

static Node* null_node(){return 0;}

template<typename Archive>
static void save_node(Node* node,Archive& ar)
{
ar<<serialization::make_nvp("pointer",node);
}

index_matcher::algorithm<Node,Allocator> alg;
};

} 

} 

} 

#endif
