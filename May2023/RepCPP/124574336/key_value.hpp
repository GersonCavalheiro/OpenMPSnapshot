

#ifndef BOOST_FLYWEIGHT_KEY_VALUE_HPP
#define BOOST_FLYWEIGHT_KEY_VALUE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>
#include <boost/flyweight/detail/perfect_fwd.hpp>
#include <boost/flyweight/detail/value_tag.hpp>
#include <boost/flyweight/key_value_fwd.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/type_traits/aligned_storage.hpp>
#include <boost/type_traits/alignment_of.hpp> 
#include <boost/type_traits/is_same.hpp>
#include <new>



namespace boost{

namespace flyweights{

namespace detail{

template<typename Key,typename Value,typename KeyFromValue>
struct optimized_key_value:value_marker
{
typedef Key   key_type;
typedef Value value_type;

class rep_type
{
public:


#define BOOST_FLYWEIGHT_PERFECT_FWD_CTR_BODY(args)       \
:value_ptr(0)                                          \
{                                                        \
new(spc_ptr())key_type(BOOST_FLYWEIGHT_FORWARD(args)); \
}

BOOST_FLYWEIGHT_PERFECT_FWD(
explicit rep_type,
BOOST_FLYWEIGHT_PERFECT_FWD_CTR_BODY)

#undef BOOST_FLYWEIGHT_PERFECT_FWD_CTR_BODY

rep_type(const rep_type& x):value_ptr(x.value_ptr)
{
if(!x.value_ptr)new(key_ptr())key_type(*x.key_ptr());
}

rep_type(const value_type& x):value_ptr(&x){}

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
rep_type(rep_type&& x):value_ptr(x.value_ptr)
{
if(!x.value_ptr)new(key_ptr())key_type(std::move(*x.key_ptr()));
}

rep_type(value_type&& x):value_ptr(&x){}
#endif

~rep_type()
{
if(!value_ptr)       key_ptr()->~key_type();
else if(value_cted())value_ptr->~value_type();
}

operator const key_type&()const
{
if(value_ptr)return key_from_value(*value_ptr);
else         return *key_ptr();
}

operator const value_type&()const
{


return *static_cast<value_type*>(spc_ptr());
}

private:
friend struct optimized_key_value;

void* spc_ptr()const{return static_cast<void*>(&spc);}
bool  value_cted()const{return value_ptr==spc_ptr();}

key_type* key_ptr()const
{
return static_cast<key_type*>(static_cast<void*>(&spc));
}

static const key_type& key_from_value(const value_type& x)
{
KeyFromValue k;
return k(x);
}

void construct_value()const
{
if(!value_cted()){


#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
key_type k(std::move(*key_ptr()));
#else
key_type k(*key_ptr());
#endif

key_ptr()->~key_type();
value_ptr= 
static_cast<value_type*>(spc_ptr())+1; 
value_ptr=new(spc_ptr())value_type(k);
}
}

void copy_value()const
{
if(!value_cted())value_ptr=new(spc_ptr())value_type(*value_ptr);
}

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
void move_value()const
{
if(!value_cted())value_ptr=
new(spc_ptr())value_type(std::move(const_cast<value_type&>(*value_ptr)));
}
#endif

mutable typename boost::aligned_storage<
(sizeof(key_type)>sizeof(value_type))?
sizeof(key_type):sizeof(value_type),
(boost::alignment_of<key_type>::value >
boost::alignment_of<value_type>::value)?
boost::alignment_of<key_type>::value:
boost::alignment_of<value_type>::value
>::type                                    spc;
mutable const value_type*                  value_ptr;
};

static void construct_value(const rep_type& r)
{
r.construct_value();
}

static void copy_value(const rep_type& r)
{
r.copy_value();
}

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
static void move_value(const rep_type& r)
{
r.move_value();
}
#endif
};

template<typename Key,typename Value>
struct regular_key_value:value_marker
{
typedef Key   key_type;
typedef Value value_type;

class rep_type
{
public:


#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)&&\
!defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)&&\
BOOST_WORKAROUND(__GNUC__,<=4)&&(__GNUC__<4||__GNUC_MINOR__<=4)



rep_type():key(),value_ptr(0){}
#endif

#define BOOST_FLYWEIGHT_PERFECT_FWD_CTR_BODY(args) \
:key(BOOST_FLYWEIGHT_FORWARD(args)),value_ptr(0){}

BOOST_FLYWEIGHT_PERFECT_FWD(
explicit rep_type,
BOOST_FLYWEIGHT_PERFECT_FWD_CTR_BODY)

#undef BOOST_FLYWEIGHT_PERFECT_FWD_CTR_BODY

rep_type(const rep_type& x):key(x.key),value_ptr(0){}
rep_type(const value_type&):key(no_key_from_value_failure()){}

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
rep_type(rep_type&& x):key(std::move(x.key)),value_ptr(0){}
rep_type(value_type&&):key(no_key_from_value_failure()){}
#endif

~rep_type()
{
if(value_ptr)value_ptr->~value_type();
}

operator const key_type&()const{return key;}

operator const value_type&()const
{


return *static_cast<value_type*>(spc_ptr());
}

private:
friend struct regular_key_value;

void* spc_ptr()const{return static_cast<void*>(&spc);}

struct no_key_from_value_failure
{
BOOST_MPL_ASSERT_MSG(
false,
NO_KEY_FROM_VALUE_CONVERSION_PROVIDED,
(key_type,value_type));

operator const key_type&()const;
};

void construct_value()const
{
if(!value_ptr)value_ptr=new(spc_ptr())value_type(key);
}

key_type                                 key;
mutable typename boost::aligned_storage<
sizeof(value_type),
boost::alignment_of<value_type>::value
>::type                                  spc;
mutable const value_type*                value_ptr;
};

static void construct_value(const rep_type& r)
{
r.construct_value();
}



static void copy_value(const rep_type&){}

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
static void move_value(const rep_type&){}
#endif
};

} 

template<typename Key,typename Value,typename KeyFromValue>
struct key_value:
mpl::if_<
is_same<KeyFromValue,no_key_from_value>,
detail::regular_key_value<Key,Value>,
detail::optimized_key_value<Key,Value,KeyFromValue>
>::type
{};

} 

} 

#endif
