

#ifndef BOOST_MULTI_INDEX_DETAIL_ORD_INDEX_NODE_HPP
#define BOOST_MULTI_INDEX_DETAIL_ORD_INDEX_NODE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <cstddef>
#include <boost/multi_index/detail/allocator_traits.hpp>
#include <boost/multi_index/detail/raw_ptr.hpp>

#if !defined(BOOST_MULTI_INDEX_DISABLE_COMPRESSED_ORDERED_INDEX_NODES)
#include <boost/mpl/and.hpp>
#include <boost/mpl/if.hpp>
#include <boost/multi_index/detail/uintptr_type.hpp>
#include <boost/type_traits/alignment_of.hpp>
#include <boost/type_traits/is_same.hpp>
#endif

namespace boost{

namespace multi_index{

namespace detail{



enum ordered_index_color{red=false,black=true};
enum ordered_index_side{to_left=false,to_right=true};

template<typename AugmentPolicy,typename Allocator>
struct ordered_index_node_impl; 

template<typename AugmentPolicy,typename Allocator>
struct ordered_index_node_traits
{
typedef typename rebind_alloc_for<
Allocator,
ordered_index_node_impl<AugmentPolicy,Allocator>
>::type                                            allocator;
typedef allocator_traits<allocator>                alloc_traits;
typedef typename alloc_traits::pointer             pointer;
typedef typename alloc_traits::const_pointer       const_pointer;
typedef typename alloc_traits::difference_type     difference_type;
typedef typename alloc_traits::size_type           size_type;
};

template<typename AugmentPolicy,typename Allocator>
struct ordered_index_node_std_base
{
typedef ordered_index_node_traits<
AugmentPolicy,Allocator>                    node_traits;
typedef typename node_traits::allocator       node_allocator;
typedef typename node_traits::pointer         pointer;
typedef typename node_traits::const_pointer   const_pointer;
typedef typename node_traits::difference_type difference_type;
typedef typename node_traits::size_type       size_type;
typedef ordered_index_color&                  color_ref;
typedef pointer&                              parent_ref;

ordered_index_color& color(){return color_;}
ordered_index_color  color()const{return color_;}
pointer&             parent(){return parent_;}
pointer              parent()const{return parent_;}
pointer&             left(){return left_;}
pointer              left()const{return left_;}
pointer&             right(){return right_;}
pointer              right()const{return right_;}

private:
ordered_index_color color_; 
pointer             parent_;
pointer             left_;
pointer             right_;
};

#if !defined(BOOST_MULTI_INDEX_DISABLE_COMPRESSED_ORDERED_INDEX_NODES)


#if defined(BOOST_MSVC)


#pragma warning(push)
#pragma warning(disable:4312 4311)
#endif

template<typename AugmentPolicy,typename Allocator>
struct ordered_index_node_compressed_base
{
typedef ordered_index_node_traits<
AugmentPolicy,Allocator>                    node_traits;
typedef ordered_index_node_impl<
AugmentPolicy,Allocator>*                   pointer;
typedef const ordered_index_node_impl<
AugmentPolicy,Allocator>*                   const_pointer;
typedef typename node_traits::difference_type difference_type;
typedef typename node_traits::size_type       size_type;

struct color_ref
{
color_ref(uintptr_type* r_):r(r_){}
color_ref(const color_ref& x):r(x.r){}

operator ordered_index_color()const
{
return ordered_index_color(*r&uintptr_type(1));
}

color_ref& operator=(ordered_index_color c)
{
*r&=~uintptr_type(1);
*r|=uintptr_type(c);
return *this;
}

color_ref& operator=(const color_ref& x)
{
return operator=(x.operator ordered_index_color());
}

private:
uintptr_type* r;
};

struct parent_ref
{
parent_ref(uintptr_type* r_):r(r_){}
parent_ref(const parent_ref& x):r(x.r){}

operator pointer()const
{
return (pointer)(void*)(*r&~uintptr_type(1));
}

parent_ref& operator=(pointer p)
{
*r=((uintptr_type)(void*)p)|(*r&uintptr_type(1));
return *this;
}

parent_ref& operator=(const parent_ref& x)
{
return operator=(x.operator pointer());
}

pointer operator->()const
{
return operator pointer();
}

private:
uintptr_type* r;
};

color_ref           color(){return color_ref(&parentcolor_);}
ordered_index_color color()const
{
return ordered_index_color(parentcolor_&uintptr_type(1));
}

parent_ref parent(){return parent_ref(&parentcolor_);}
pointer    parent()const
{
return (pointer)(void*)(parentcolor_&~uintptr_type(1));
}

pointer& left(){return left_;}
pointer  left()const{return left_;}
pointer& right(){return right_;}
pointer  right()const{return right_;}

private:
uintptr_type parentcolor_;
pointer      left_;
pointer      right_;
};
#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif
#endif

template<typename AugmentPolicy,typename Allocator>
struct ordered_index_node_impl_base:

#if !defined(BOOST_MULTI_INDEX_DISABLE_COMPRESSED_ORDERED_INDEX_NODES)
AugmentPolicy::template augmented_node<
typename mpl::if_c<
!(has_uintptr_type::value)||
(alignment_of<
ordered_index_node_compressed_base<AugmentPolicy,Allocator>
>::value%2)||
!(is_same<
typename ordered_index_node_traits<AugmentPolicy,Allocator>::pointer,
ordered_index_node_impl<AugmentPolicy,Allocator>*>::value),
ordered_index_node_std_base<AugmentPolicy,Allocator>,
ordered_index_node_compressed_base<AugmentPolicy,Allocator>
>::type
>::type
#else
AugmentPolicy::template augmented_node<
ordered_index_node_std_base<AugmentPolicy,Allocator>
>::type
#endif

{};

template<typename AugmentPolicy,typename Allocator>
struct ordered_index_node_impl:
ordered_index_node_impl_base<AugmentPolicy,Allocator>
{
private:
typedef ordered_index_node_impl_base<AugmentPolicy,Allocator> super;

public:
typedef typename super::color_ref                             color_ref;
typedef typename super::parent_ref                            parent_ref;
typedef typename super::pointer                               pointer;
typedef typename super::const_pointer                         const_pointer;



static void increment(pointer& x)
{
if(x->right()!=pointer(0)){
x=x->right();
while(x->left()!=pointer(0))x=x->left();
}
else{
pointer y=x->parent();
while(x==y->right()){
x=y;
y=y->parent();
}
if(x->right()!=y)x=y;
}
}

static void decrement(pointer& x)
{
if(x->color()==red&&x->parent()->parent()==x){
x=x->right();
}
else if(x->left()!=pointer(0)){
pointer y=x->left();
while(y->right()!=pointer(0))y=y->right();
x=y;
}else{
pointer y=x->parent();
while(x==y->left()){
x=y;
y=y->parent();
}
x=y;
}
}



static void rotate_left(pointer x,parent_ref root)
{
pointer y=x->right();
x->right()=y->left();
if(y->left()!=pointer(0))y->left()->parent()=x;
y->parent()=x->parent();

if(x==root)                    root=y;
else if(x==x->parent()->left())x->parent()->left()=y;
else                           x->parent()->right()=y;
y->left()=x;
x->parent()=y;
AugmentPolicy::rotate_left(x,y);
}

static pointer minimum(pointer x)
{
while(x->left()!=pointer(0))x=x->left();
return x;
}

static pointer maximum(pointer x)
{
while(x->right()!=pointer(0))x=x->right();
return x;
}

static void rotate_right(pointer x,parent_ref root)
{
pointer y=x->left();
x->left()=y->right();
if(y->right()!=pointer(0))y->right()->parent()=x;
y->parent()=x->parent();

if(x==root)                     root=y;
else if(x==x->parent()->right())x->parent()->right()=y;
else                            x->parent()->left()=y;
y->right()=x;
x->parent()=y;
AugmentPolicy::rotate_right(x,y);
}

static void rebalance(pointer x,parent_ref root)
{
x->color()=red;
while(x!=root&&x->parent()->color()==red){
if(x->parent()==x->parent()->parent()->left()){
pointer y=x->parent()->parent()->right();
if(y!=pointer(0)&&y->color()==red){
x->parent()->color()=black;
y->color()=black;
x->parent()->parent()->color()=red;
x=x->parent()->parent();
}
else{
if(x==x->parent()->right()){
x=x->parent();
rotate_left(x,root);
}
x->parent()->color()=black;
x->parent()->parent()->color()=red;
rotate_right(x->parent()->parent(),root);
}
}
else{
pointer y=x->parent()->parent()->left();
if(y!=pointer(0)&&y->color()==red){
x->parent()->color()=black;
y->color()=black;
x->parent()->parent()->color()=red;
x=x->parent()->parent();
}
else{
if(x==x->parent()->left()){
x=x->parent();
rotate_right(x,root);
}
x->parent()->color()=black;
x->parent()->parent()->color()=red;
rotate_left(x->parent()->parent(),root);
}
}
}
root->color()=black;
}

static void link(
pointer x,ordered_index_side side,pointer position,pointer header)
{
if(side==to_left){
position->left()=x;  
if(position==header){
header->parent()=x;
header->right()=x;
}
else if(position==header->left()){
header->left()=x;  
}
}
else{
position->right()=x;
if(position==header->right()){
header->right()=x; 
}
}
x->parent()=position;
x->left()=pointer(0);
x->right()=pointer(0);
AugmentPolicy::add(x,pointer(header->parent()));
ordered_index_node_impl::rebalance(x,header->parent());
}

static pointer rebalance_for_extract(
pointer z,parent_ref root,pointer& leftmost,pointer& rightmost)
{
pointer y=z;
pointer x=pointer(0);
pointer x_parent=pointer(0);
if(y->left()==pointer(0)){    
x=y->right();               
}
else{
if(y->right()==pointer(0)){ 
x=y->left();              
}
else{                       
y=y->right();             
while(y->left()!=pointer(0))y=y->left();
x=y->right();
}
}
AugmentPolicy::remove(y,pointer(root));
if(y!=z){
AugmentPolicy::copy(z,y);
z->left()->parent()=y;   
y->left()=z->left();
if(y!=z->right()){
x_parent=y->parent();
if(x!=pointer(0))x->parent()=y->parent();
y->parent()->left()=x; 
y->right()=z->right();
z->right()->parent()=y;
}
else{
x_parent=y;
}

if(root==z)                    root=y;
else if(z->parent()->left()==z)z->parent()->left()=y;
else                           z->parent()->right()=y;
y->parent()=z->parent();
ordered_index_color c=y->color();
y->color()=z->color();
z->color()=c;
y=z;                    
}
else{                     
x_parent=y->parent();
if(x!=pointer(0))x->parent()=y->parent();   
if(root==z){
root=x;
}
else{
if(z->parent()->left()==z)z->parent()->left()=x;
else                      z->parent()->right()=x;
}
if(leftmost==z){
if(z->right()==pointer(0)){ 
leftmost=z->parent();
}
else{              
leftmost=minimum(x);      
}
}
if(rightmost==z){
if(z->left()==pointer(0)){  
rightmost=z->parent();
}
else{                   
rightmost=maximum(x); 
}
}
}
if(y->color()!=red){
while(x!=root&&(x==pointer(0)|| x->color()==black)){
if(x==x_parent->left()){
pointer w=x_parent->right();
if(w->color()==red){
w->color()=black;
x_parent->color()=red;
rotate_left(x_parent,root);
w=x_parent->right();
}
if((w->left()==pointer(0)||w->left()->color()==black) &&
(w->right()==pointer(0)||w->right()->color()==black)){
w->color()=red;
x=x_parent;
x_parent=x_parent->parent();
} 
else{
if(w->right()==pointer(0 )
|| w->right()->color()==black){
if(w->left()!=pointer(0)) w->left()->color()=black;
w->color()=red;
rotate_right(w,root);
w=x_parent->right();
}
w->color()=x_parent->color();
x_parent->color()=black;
if(w->right()!=pointer(0))w->right()->color()=black;
rotate_left(x_parent,root);
break;
}
} 
else{                   
pointer w=x_parent->left();
if(w->color()==red){
w->color()=black;
x_parent->color()=red;
rotate_right(x_parent,root);
w=x_parent->left();
}
if((w->right()==pointer(0)||w->right()->color()==black) &&
(w->left()==pointer(0)||w->left()->color()==black)){
w->color()=red;
x=x_parent;
x_parent=x_parent->parent();
}
else{
if(w->left()==pointer(0)||w->left()->color()==black){
if(w->right()!=pointer(0))w->right()->color()=black;
w->color()=red;
rotate_left(w,root);
w=x_parent->left();
}
w->color()=x_parent->color();
x_parent->color()=black;
if(w->left()!=pointer(0))w->left()->color()=black;
rotate_right(x_parent,root);
break;
}
}
}
if(x!=pointer(0))x->color()=black;
}
return y;
}

static void restore(pointer x,pointer position,pointer header)
{
if(position->left()==pointer(0)||position->left()==header){
link(x,to_left,position,header);
}
else{
decrement(position);
link(x,to_right,position,header);
}
}

#if defined(BOOST_MULTI_INDEX_ENABLE_INVARIANT_CHECKING)


static std::size_t black_count(pointer node,pointer root)
{
if(node==pointer(0))return 0;
std::size_t sum=0;
for(;;){
if(node->color()==black)++sum;
if(node==root)break;
node=node->parent();
} 
return sum;
}
#endif
};

template<typename AugmentPolicy,typename Super>
struct ordered_index_node_trampoline:
ordered_index_node_impl<
AugmentPolicy,
typename rebind_alloc_for<
typename Super::allocator_type,
char
>::type
>
{
typedef ordered_index_node_impl<
AugmentPolicy,
typename rebind_alloc_for<
typename Super::allocator_type,
char
>::type
> impl_type;
};

template<typename AugmentPolicy,typename Super>
struct ordered_index_node:
Super,ordered_index_node_trampoline<AugmentPolicy,Super>
{
private:
typedef ordered_index_node_trampoline<AugmentPolicy,Super> trampoline;

public:
typedef typename trampoline::impl_type       impl_type;
typedef typename trampoline::color_ref       impl_color_ref;
typedef typename trampoline::parent_ref      impl_parent_ref;
typedef typename trampoline::pointer         impl_pointer;
typedef typename trampoline::const_pointer   const_impl_pointer;
typedef typename trampoline::difference_type difference_type;
typedef typename trampoline::size_type       size_type;

impl_color_ref      color(){return trampoline::color();}
ordered_index_color color()const{return trampoline::color();}
impl_parent_ref     parent(){return trampoline::parent();}
impl_pointer        parent()const{return trampoline::parent();}
impl_pointer&       left(){return trampoline::left();}
impl_pointer        left()const{return trampoline::left();}
impl_pointer&       right(){return trampoline::right();}
impl_pointer        right()const{return trampoline::right();}

impl_pointer impl()
{
return static_cast<impl_pointer>(
static_cast<impl_type*>(static_cast<trampoline*>(this)));
}

const_impl_pointer impl()const
{
return static_cast<const_impl_pointer>(
static_cast<const impl_type*>(static_cast<const trampoline*>(this)));
}

static ordered_index_node* from_impl(impl_pointer x)
{
return
static_cast<ordered_index_node*>(
static_cast<trampoline*>(
raw_ptr<impl_type*>(x)));
}

static const ordered_index_node* from_impl(const_impl_pointer x)
{
return
static_cast<const ordered_index_node*>(
static_cast<const trampoline*>(
raw_ptr<const impl_type*>(x)));
}



static void increment(ordered_index_node*& x)
{
impl_pointer xi=x->impl();
trampoline::increment(xi);
x=from_impl(xi);
}

static void decrement(ordered_index_node*& x)
{
impl_pointer xi=x->impl();
trampoline::decrement(xi);
x=from_impl(xi);
}
};

} 

} 

} 

#endif
