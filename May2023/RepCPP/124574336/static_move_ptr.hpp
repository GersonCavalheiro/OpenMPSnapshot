

#ifndef BOOST_STATIC_MOVE_PTR_HPP_INCLUDED
#define BOOST_STATIC_MOVE_PTR_HPP_INCLUDED

#include <boost/config.hpp> 
#include <cstddef>          
#include <boost/compressed_pair.hpp> 
#include <boost/ptr_container/detail/default_deleter.hpp>       
#include <boost/ptr_container/detail/is_convertible.hpp>       
#include <boost/ptr_container/detail/move.hpp>       
#include <boost/static_assert.hpp>
#include <boost/type_traits/add_reference.hpp>
#include <boost/type_traits/is_array.hpp>

#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable:4521)        
#endif

namespace boost { namespace ptr_container_detail {


template< typename T, 
typename Deleter = 
move_ptrs::default_deleter<T> >
class static_move_ptr 
{
public:

typedef typename remove_bounds<T>::type             element_type;
typedef Deleter                                     deleter_type;

private:

struct safe_bool_helper { int x; };
typedef int safe_bool_helper::* safe_bool;
typedef boost::compressed_pair<element_type*, Deleter> impl_type;

public:
typedef typename impl_type::second_reference        deleter_reference;
typedef typename impl_type::second_const_reference  deleter_const_reference;


static_move_ptr() : impl_(0) { }

static_move_ptr(const static_move_ptr& p)
: impl_(p.get(), p.get_deleter())    
{ 
const_cast<static_move_ptr&>(p).release();
}

#if BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x564))    
static_move_ptr( const move_ptrs::move_source<static_move_ptr<T,Deleter> >& src )
#else
static_move_ptr( const move_ptrs::move_source<static_move_ptr>& src )
#endif    
: impl_(src.ptr().get(), src.ptr().get_deleter())
{
src.ptr().release();
}

template<typename TT>
static_move_ptr(TT* tt, Deleter del) 
: impl_(tt, del) 
{ }


~static_move_ptr() { if (ptr()) get_deleter()(ptr()); }


static_move_ptr& operator=(static_move_ptr rhs)
{
rhs.swap(*this);
return *this;
}


element_type* get() const { return ptr(); }

element_type& operator*() 
{ 
return *ptr(); 
}

const element_type& operator*() const 
{ 
return *ptr(); 
}

element_type* operator->()  
{ 
return ptr(); 
}    

const element_type* operator->() const 
{ 
return ptr(); 
}    


element_type* release()
{
element_type* result = ptr();
ptr() = 0;
return result;
}

void reset()
{
if (ptr()) get_deleter()(ptr());
ptr() = 0;
}

template<typename TT>
void reset(TT* tt, Deleter dd) 
{
static_move_ptr(tt, dd).swap(*this);
}

operator safe_bool() const { return ptr() ? &safe_bool_helper::x : 0; }

void swap(static_move_ptr& p) { impl_.swap(p.impl_); }

deleter_reference get_deleter() { return impl_.second(); }

deleter_const_reference get_deleter() const { return impl_.second(); }
private:
template<typename TT, typename DD>
void check(const static_move_ptr<TT, DD>&)
{
typedef move_ptrs::is_smart_ptr_convertible<TT, T> convertible;
BOOST_STATIC_ASSERT(convertible::value);
}   

#if defined(BOOST_NO_FUNCTION_TEMPLATE_ORDERING) || defined(BOOST_NO_SFINAE)
#else 

template<typename Ptr> struct cant_move_from_const;

template<typename TT, typename DD> 
struct cant_move_from_const< const static_move_ptr<TT, DD> > { 
typedef typename static_move_ptr<TT, DD>::error type; 
};

template<typename Ptr> 
static_move_ptr(Ptr&, typename cant_move_from_const<Ptr>::type = 0);


public:
static_move_ptr(static_move_ptr&);


private:
template<typename TT, typename DD>
static_move_ptr( static_move_ptr<TT, DD>&,
typename 
move_ptrs::enable_if_convertible<
TT, T, static_move_ptr&
>::type::type* = 0 );

#endif 

public:
typename impl_type::first_reference 
ptr() { return impl_.first(); } 

typename impl_type::first_const_reference 
ptr() const { return impl_.first(); }

impl_type impl_;
};

} 
} 

#if defined(BOOST_MSVC)
#pragma warning(pop) 
#endif

#endif      
