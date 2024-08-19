

#ifndef BOOST_MULTI_INDEX_DETAIL_INDEX_LOADER_HPP
#define BOOST_MULTI_INDEX_DETAIL_INDEX_LOADER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <algorithm>
#include <boost/archive/archive_exception.hpp>
#include <boost/noncopyable.hpp>
#include <boost/multi_index/detail/auto_space.hpp>
#include <boost/multi_index/detail/raw_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/throw_exception.hpp> 
#include <cstddef>

namespace boost{

namespace multi_index{

namespace detail{



template<typename Node,typename FinalNode,typename Allocator>
class index_loader:private noncopyable
{
public:
index_loader(const Allocator& al,std::size_t size):
spc(al,size),size_(size),n(0),sorted(false)
{
}

template<class Archive>
void add(Node* node,Archive& ar,const unsigned int)
{
ar>>serialization::make_nvp("position",*node);
entries()[n++]=node;
}

template<class Archive>
void add_track(Node* node,Archive& ar,const unsigned int)
{
ar>>serialization::make_nvp("position",*node);
}



template<typename Rearranger,class Archive>
void load(Rearranger r,Archive& ar,const unsigned int)const
{
FinalNode* prev=unchecked_load_node(ar);
if(!prev)return;

if(!sorted){
std::sort(entries(),entries()+size_);
sorted=true;
}

check_node(prev);

for(;;){
for(;;){
FinalNode* node=load_node(ar);
if(!node)break;

if(node==prev)prev=0;
r(prev,node);

prev=node;
}
prev=load_node(ar);
if(!prev)break;
}
}

private:
Node** entries()const{return raw_ptr<Node**>(spc.data());}



template<class Archive>
FinalNode* unchecked_load_node(Archive& ar)const
{
Node* node=0;
ar>>serialization::make_nvp("pointer",node);
return static_cast<FinalNode*>(node);
}

template<class Archive>
FinalNode* load_node(Archive& ar)const
{
Node* node=0;
ar>>serialization::make_nvp("pointer",node);
check_node(node);
return static_cast<FinalNode*>(node);
}

void check_node(Node* node)const
{
if(node!=0&&!std::binary_search(entries(),entries()+size_,node)){
throw_exception(
archive::archive_exception(
archive::archive_exception::other_exception));
}
}

auto_space<Node*,Allocator> spc;
std::size_t                 size_;
std::size_t                 n;
mutable bool                sorted;
};

} 

} 

} 

#endif
