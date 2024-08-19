
#ifndef BOOST_INTERPROCESS_NAMED_PROXY_HPP
#define BOOST_INTERPROCESS_NAMED_PROXY_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/detail/in_place_interface.hpp>
#include <boost/interprocess/detail/mpl.hpp>
#include <boost/move/utility_core.hpp>
#ifndef BOOST_INTERPROCESS_PERFECT_FORWARDING
#include <boost/move/detail/fwd_macros.hpp>
#else
#include <boost/move/utility_core.hpp>
#include <boost/interprocess/detail/variadic_templates_tools.hpp>
#endif   
#include <boost/container/detail/placement_new.hpp>

#include <cstddef>


namespace boost {
namespace interprocess {
namespace ipcdetail {

#ifdef BOOST_INTERPROCESS_PERFECT_FORWARDING

template<class T, bool is_iterator, class ...Args>
struct CtorArgN : public placement_destroy<T>
{
typedef bool_<is_iterator> IsIterator;
typedef CtorArgN<T, is_iterator, Args...> self_t;
typedef typename build_number_seq<sizeof...(Args)>::type index_tuple_t;

self_t& operator++()
{
this->do_increment(IsIterator(), index_tuple_t());
return *this;
}

self_t  operator++(int) {  return ++*this;   *this;  }

CtorArgN(Args && ...args)
:  args_(args...)
{}

virtual void construct_n(void *mem
, std::size_t num
, std::size_t &constructed)
{
T* memory      = static_cast<T*>(mem);
for(constructed = 0; constructed < num; ++constructed){
this->construct(memory++, IsIterator(), index_tuple_t());
this->do_increment(IsIterator(), index_tuple_t());
}
}

private:
template<std::size_t ...IdxPack>
void construct(void *mem, true_, const index_tuple<IdxPack...>&)
{  ::new((void*)mem, boost_container_new_t())T(*boost::forward<Args>(get<IdxPack>(args_))...); }

template<std::size_t ...IdxPack>
void construct(void *mem, false_, const index_tuple<IdxPack...>&)
{  ::new((void*)mem, boost_container_new_t())T(boost::forward<Args>(get<IdxPack>(args_))...); }

template<std::size_t ...IdxPack>
void do_increment(true_, const index_tuple<IdxPack...>&)
{
this->expansion_helper(++get<IdxPack>(args_)...);
}

template<class ...ExpansionArgs>
void expansion_helper(ExpansionArgs &&...)
{}

template<std::size_t ...IdxPack>
void do_increment(false_, const index_tuple<IdxPack...>&)
{}

tuple<Args&...> args_;
};

template
< class SegmentManager  
, class T               
, bool is_iterator      
>
class named_proxy
{
typedef typename SegmentManager::char_type char_type;
const char_type *    mp_name;
SegmentManager *     mp_mngr;
mutable std::size_t  m_num;
const bool           m_find;
const bool           m_dothrow;

public:
named_proxy(SegmentManager *mngr, const char_type *name, bool find, bool dothrow)
:  mp_name(name), mp_mngr(mngr), m_num(1)
,  m_find(find),  m_dothrow(dothrow)
{}

template<class ...Args>
T *operator()(Args &&...args) const
{
CtorArgN<T, is_iterator, Args...> &&ctor_obj = CtorArgN<T, is_iterator, Args...>
(boost::forward<Args>(args)...);
return mp_mngr->template
generic_construct<T>(mp_name, m_num, m_find, m_dothrow, ctor_obj);
}

const named_proxy &operator[](std::size_t num) const
{  m_num *= num; return *this;  }
};

#else 


#define BOOST_INTERPROCESS_NAMED_PROXY_CTORARGN(N)\
\
template<class T BOOST_MOVE_I##N BOOST_MOVE_CLASS##N >  \
struct CtorArg##N : placement_destroy<T>\
{\
typedef CtorArg##N self_t;\
\
CtorArg##N ( BOOST_MOVE_UREF##N  )\
BOOST_MOVE_COLON##N BOOST_MOVE_FWD_INIT##N{}\
\
virtual void construct_n(void *mem, std::size_t num, std::size_t &constructed)\
{\
T* memory      = static_cast<T*>(mem);\
for(constructed = 0; constructed < num; ++constructed){\
::new((void*)memory++) T ( BOOST_MOVE_MFWD##N );\
}\
}\
\
private:\
BOOST_MOVE_MREF##N\
};\
BOOST_MOVE_ITERATE_0TO9(BOOST_INTERPROCESS_NAMED_PROXY_CTORARGN)
#undef BOOST_INTERPROCESS_NAMED_PROXY_CTORARGN

#define BOOST_INTERPROCESS_NAMED_PROXY_CTORITN(N)\
\
template<class T BOOST_MOVE_I##N BOOST_MOVE_CLASS##N > \
struct CtorIt##N : public placement_destroy<T>\
{\
typedef CtorIt##N self_t;\
\
self_t& operator++()\
{  BOOST_MOVE_MINC##N;  return *this;  }\
\
self_t  operator++(int) {  return ++*this; *this;  }\
\
CtorIt##N ( BOOST_MOVE_VAL##N  )\
BOOST_MOVE_COLON##N BOOST_MOVE_VAL_INIT##N{}\
\
virtual void construct_n(void *mem, std::size_t num, std::size_t &constructed)\
{\
T* memory      = static_cast<T*>(mem);\
for(constructed = 0; constructed < num; ++constructed){\
::new((void*)memory++) T( BOOST_MOVE_MITFWD##N );\
++(*this);\
}\
}\
\
private:\
BOOST_MOVE_MEMB##N\
};\
BOOST_MOVE_ITERATE_0TO9(BOOST_INTERPROCESS_NAMED_PROXY_CTORITN)
#undef BOOST_INTERPROCESS_NAMED_PROXY_CTORITN

template
< class SegmentManager  
, class T               
, bool is_iterator      
>
class named_proxy
{
typedef typename SegmentManager::char_type char_type;
const char_type *    mp_name;
SegmentManager *     mp_mngr;
mutable std::size_t  m_num;
const bool           m_find;
const bool           m_dothrow;

public:
named_proxy(SegmentManager *mngr, const char_type *name, bool find, bool dothrow)
:  mp_name(name), mp_mngr(mngr), m_num(1)
,  m_find(find),  m_dothrow(dothrow)
{}

#define BOOST_INTERPROCESS_NAMED_PROXY_CALL_OPERATOR(N)\
\
BOOST_MOVE_TMPL_LT##N BOOST_MOVE_CLASS##N BOOST_MOVE_GT##N \
T *operator()( BOOST_MOVE_UREF##N ) const\
{\
typedef typename if_c<is_iterator \
, CtorIt##N <T BOOST_MOVE_I##N BOOST_MOVE_TARG##N> \
, CtorArg##N<T BOOST_MOVE_I##N BOOST_MOVE_TARG##N> \
>::type ctor_obj_t;\
ctor_obj_t ctor_obj = ctor_obj_t( BOOST_MOVE_FWD##N );\
return mp_mngr->template generic_construct<T>(mp_name, m_num, m_find, m_dothrow, ctor_obj);\
}\
BOOST_MOVE_ITERATE_0TO9(BOOST_INTERPROCESS_NAMED_PROXY_CALL_OPERATOR)
#undef BOOST_INTERPROCESS_NAMED_PROXY_CALL_OPERATOR


const named_proxy &operator[](std::size_t num) const
{  m_num *= num; return *this;  }
};

#endif   

}}}   

#include <boost/interprocess/detail/config_end.hpp>

#endif 
