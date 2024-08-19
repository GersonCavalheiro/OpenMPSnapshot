#ifndef BOOST_DETAIL_SHARED_COUNT_132_HPP_INCLUDED
#define BOOST_DETAIL_SHARED_COUNT_132_HPP_INCLUDED


#if defined(_MSC_VER)
# pragma once
#endif


#include <boost/config.hpp>

#if defined(BOOST_SP_USE_STD_ALLOCATOR) && defined(BOOST_SP_USE_QUICK_ALLOCATOR)
# error BOOST_SP_USE_STD_ALLOCATOR and BOOST_SP_USE_QUICK_ALLOCATOR are incompatible.
#endif

#include <boost/checked_delete.hpp>
#include <boost/serialization/throw_exception.hpp>
#include <boost/detail/lightweight_mutex.hpp>

#if defined(BOOST_SP_USE_QUICK_ALLOCATOR)
#include <boost/detail/quick_allocator.hpp>
#endif

#include <memory>           
#include <functional>       
#include <exception>        
#include <new>              
#include <typeinfo>         
#include <cstddef>          

#include <boost/config.hpp> 
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::size_t;
} 
#endif

namespace boost_132 {


#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)

void sp_scalar_constructor_hook(void * px, std::size_t size, void * pn);
void sp_array_constructor_hook(void * px);
void sp_scalar_destructor_hook(void * px, std::size_t size, void * pn);
void sp_array_destructor_hook(void * px);

#endif



class bad_weak_ptr: public std::exception
{
public:

virtual char const * what() const throw()
{
return "boost::bad_weak_ptr";
}
};

namespace detail{

class sp_counted_base
{

typedef boost::detail::lightweight_mutex mutex_type;

public:

sp_counted_base(): use_count_(1), weak_count_(1)
{
}

virtual ~sp_counted_base() 
{
}


virtual void dispose() = 0; 


virtual void destruct() 
{
delete this;
}

virtual void * get_deleter(std::type_info const & ti) = 0;

void add_ref_copy()
{
#if defined(BOOST_HAS_THREADS)
mutex_type::scoped_lock lock(mtx_);
#endif
++use_count_;
}

void add_ref_lock()
{
#if defined(BOOST_HAS_THREADS)
mutex_type::scoped_lock lock(mtx_);
#endif
if(use_count_ == 0) boost::serialization::throw_exception(bad_weak_ptr());
++use_count_;
}

void release() 
{
{
#if defined(BOOST_HAS_THREADS)
mutex_type::scoped_lock lock(mtx_);
#endif
long new_use_count = --use_count_;

if(new_use_count != 0) return;
}

dispose();
weak_release();
}

void weak_add_ref() 
{
#if defined(BOOST_HAS_THREADS)
mutex_type::scoped_lock lock(mtx_);
#endif
++weak_count_;
}

void weak_release() 
{
long new_weak_count;

{
#if defined(BOOST_HAS_THREADS)
mutex_type::scoped_lock lock(mtx_);
#endif
new_weak_count = --weak_count_;
}

if(new_weak_count == 0)
{
destruct();
}
}

long use_count() const 
{
#if defined(BOOST_HAS_THREADS)
mutex_type::scoped_lock lock(mtx_);
#endif
return use_count_;
}

public:
sp_counted_base(sp_counted_base const &);
sp_counted_base & operator= (sp_counted_base const &);

long use_count_;        
long weak_count_;       

#if defined(BOOST_HAS_THREADS) || defined(BOOST_LWM_WIN32)
mutable mutex_type mtx_;
#endif
};

#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)

template<class T> void cbi_call_constructor_hook(sp_counted_base * pn, T * px, boost::checked_deleter< T > const &)
{
boost::sp_scalar_constructor_hook(px, sizeof(T), pn);
}

template<class T> void cbi_call_constructor_hook(sp_counted_base *, T * px, boost::checked_array_deleter< T > const &)
{
boost::sp_array_constructor_hook(px);
}

template<class P, class D> void cbi_call_constructor_hook(sp_counted_base *, P const &, D const &, long)
{
}

template<class T> void cbi_call_destructor_hook(sp_counted_base * pn, T * px, boost::checked_deleter< T > const &)
{
boost::sp_scalar_destructor_hook(px, sizeof(T), pn);
}

template<class T> void cbi_call_destructor_hook(sp_counted_base *, T * px, boost::checked_array_deleter< T > const &)
{
boost::sp_array_destructor_hook(px);
}

template<class P, class D> void cbi_call_destructor_hook(sp_counted_base *, P const &, D const &, long)
{
}

#endif

#ifdef __CODEGUARD__
# pragma option push -Vx-
#endif

template<class P, class D> class sp_counted_base_impl: public sp_counted_base
{
public:
P ptr; 
D del; 

sp_counted_base_impl(sp_counted_base_impl const &);
sp_counted_base_impl & operator= (sp_counted_base_impl const &);

typedef sp_counted_base_impl<P, D> this_type;

public:


sp_counted_base_impl(P p, D d): ptr(p), del(d)
{
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
detail::cbi_call_constructor_hook(this, p, d, 0);
#endif
}

virtual void dispose() 
{
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
detail::cbi_call_destructor_hook(this, ptr, del, 0);
#endif
del(ptr);
}

virtual void * get_deleter(std::type_info const & ti)
{
return ti == typeid(D)? &del: 0;
}

#if defined(BOOST_SP_USE_STD_ALLOCATOR)

void * operator new(std::size_t)
{
return std::allocator<this_type>().allocate(1, static_cast<this_type *>(0));
}

void operator delete(void * p)
{
std::allocator<this_type>().deallocate(static_cast<this_type *>(p), 1);
}

#endif

#if defined(BOOST_SP_USE_QUICK_ALLOCATOR)

void * operator new(std::size_t)
{
return boost::detail::quick_allocator<this_type>::alloc();
}

void operator delete(void * p)
{
boost::detail::quick_allocator<this_type>::dealloc(p);
}

#endif
};

#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)

int const shared_count_id = 0x2C35F101;
int const   weak_count_id = 0x298C38A4;

#endif

class weak_count;

class shared_count
{
public:
sp_counted_base * pi_;

#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
int id_;
#endif

friend class weak_count;

public:

shared_count(): pi_(0) 
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
, id_(shared_count_id)
#endif
{
}

template<class P, class D> shared_count(P p, D d): pi_(0)
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
, id_(shared_count_id)
#endif
{
#ifndef BOOST_NO_EXCEPTIONS

try
{
pi_ = new sp_counted_base_impl<P, D>(p, d);
}
catch(...)
{
d(p); 
throw;
}

#else

pi_ = new sp_counted_base_impl<P, D>(p, d);

if(pi_ == 0)
{
d(p); 
boost::serialization::throw_exception(std::bad_alloc());
}

#endif
}

#ifndef BOOST_NO_AUTO_PTR


template<class Y>
explicit shared_count(std::auto_ptr<Y> & r): pi_(
new sp_counted_base_impl<
Y *,
boost::checked_deleter<Y>
>(r.get(), boost::checked_deleter<Y>()))
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
, id_(shared_count_id)
#endif
{
r.release();
}

#endif

~shared_count() 
{
if(pi_ != 0) pi_->release();
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
id_ = 0;
#endif
}

shared_count(shared_count const & r): pi_(r.pi_) 
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
, id_(shared_count_id)
#endif
{
if(pi_ != 0) pi_->add_ref_copy();
}

explicit shared_count(weak_count const & r); 

shared_count & operator= (shared_count const & r) 
{
sp_counted_base * tmp = r.pi_;

if(tmp != pi_)
{
if(tmp != 0) tmp->add_ref_copy();
if(pi_ != 0) pi_->release();
pi_ = tmp;
}

return *this;
}

void swap(shared_count & r) 
{
sp_counted_base * tmp = r.pi_;
r.pi_ = pi_;
pi_ = tmp;
}

long use_count() const 
{
return pi_ != 0? pi_->use_count(): 0;
}

bool unique() const 
{
return use_count() == 1;
}

friend inline bool operator==(shared_count const & a, shared_count const & b)
{
return a.pi_ == b.pi_;
}

friend inline bool operator<(shared_count const & a, shared_count const & b)
{
return std::less<sp_counted_base *>()(a.pi_, b.pi_);
}

void * get_deleter(std::type_info const & ti) const
{
return pi_? pi_->get_deleter(ti): 0;
}
};

#ifdef __CODEGUARD__
# pragma option pop
#endif


class weak_count
{
private:

sp_counted_base * pi_;

#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
int id_;
#endif

friend class shared_count;

public:

weak_count(): pi_(0) 
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
, id_(weak_count_id)
#endif
{
}

weak_count(shared_count const & r): pi_(r.pi_) 
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
, id_(shared_count_id)
#endif
{
if(pi_ != 0) pi_->weak_add_ref();
}

weak_count(weak_count const & r): pi_(r.pi_) 
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
, id_(shared_count_id)
#endif
{
if(pi_ != 0) pi_->weak_add_ref();
}

~weak_count() 
{
if(pi_ != 0) pi_->weak_release();
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
id_ = 0;
#endif
}

weak_count & operator= (shared_count const & r) 
{
sp_counted_base * tmp = r.pi_;
if(tmp != 0) tmp->weak_add_ref();
if(pi_ != 0) pi_->weak_release();
pi_ = tmp;

return *this;
}

weak_count & operator= (weak_count const & r) 
{
sp_counted_base * tmp = r.pi_;
if(tmp != 0) tmp->weak_add_ref();
if(pi_ != 0) pi_->weak_release();
pi_ = tmp;

return *this;
}

void swap(weak_count & r) 
{
sp_counted_base * tmp = r.pi_;
r.pi_ = pi_;
pi_ = tmp;
}

long use_count() const 
{
return pi_ != 0? pi_->use_count(): 0;
}

friend inline bool operator==(weak_count const & a, weak_count const & b)
{
return a.pi_ == b.pi_;
}

friend inline bool operator<(weak_count const & a, weak_count const & b)
{
return std::less<sp_counted_base *>()(a.pi_, b.pi_);
}
};

inline shared_count::shared_count(weak_count const & r): pi_(r.pi_)
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
, id_(shared_count_id)
#endif
{
if(pi_ != 0)
{
pi_->add_ref_lock();
}
else
{
boost::serialization::throw_exception(bad_weak_ptr());
}
}

} 

} 

BOOST_SERIALIZATION_ASSUME_ABSTRACT(boost_132::detail::sp_counted_base)

#endif  
