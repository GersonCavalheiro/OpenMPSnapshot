#ifndef BOOST_SMART_PTR_SCOPED_PTR_HPP_INCLUDED
#define BOOST_SMART_PTR_SCOPED_PTR_HPP_INCLUDED


#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/checked_delete.hpp>
#include <boost/smart_ptr/detail/sp_nullptr_t.hpp>
#include <boost/smart_ptr/detail/sp_disable_deprecated.hpp>
#include <boost/smart_ptr/detail/sp_noexcept.hpp>
#include <boost/config/workaround.hpp>

#ifndef BOOST_NO_AUTO_PTR
# include <memory>          
#endif

#if defined( BOOST_SP_DISABLE_DEPRECATED )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace boost
{


#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)

void sp_scalar_constructor_hook(void * p);
void sp_scalar_destructor_hook(void * p);

#endif


template<class T> class scoped_ptr 
{
private:

T * px;

scoped_ptr(scoped_ptr const &);
scoped_ptr & operator=(scoped_ptr const &);

typedef scoped_ptr<T> this_type;

void operator==( scoped_ptr const& ) const;
void operator!=( scoped_ptr const& ) const;

public:

typedef T element_type;

explicit scoped_ptr( T * p = 0 ) BOOST_SP_NOEXCEPT : px( p )
{
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
boost::sp_scalar_constructor_hook( px );
#endif
}

#ifndef BOOST_NO_AUTO_PTR

explicit scoped_ptr( std::auto_ptr<T> p ) BOOST_SP_NOEXCEPT : px( p.release() )
{
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
boost::sp_scalar_constructor_hook( px );
#endif
}

#endif

~scoped_ptr() BOOST_SP_NOEXCEPT
{
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
boost::sp_scalar_destructor_hook( px );
#endif
boost::checked_delete( px );
}

void reset(T * p = 0) BOOST_SP_NOEXCEPT_WITH_ASSERT
{
BOOST_ASSERT( p == 0 || p != px ); 
this_type(p).swap(*this);
}

T & operator*() const BOOST_SP_NOEXCEPT_WITH_ASSERT
{
BOOST_ASSERT( px != 0 );
return *px;
}

T * operator->() const BOOST_SP_NOEXCEPT_WITH_ASSERT
{
BOOST_ASSERT( px != 0 );
return px;
}

T * get() const BOOST_SP_NOEXCEPT
{
return px;
}

#include <boost/smart_ptr/detail/operator_bool.hpp>

void swap(scoped_ptr & b) BOOST_SP_NOEXCEPT
{
T * tmp = b.px;
b.px = px;
px = tmp;
}
};

#if !defined( BOOST_NO_CXX11_NULLPTR )

template<class T> inline bool operator==( scoped_ptr<T> const & p, boost::detail::sp_nullptr_t ) BOOST_SP_NOEXCEPT
{
return p.get() == 0;
}

template<class T> inline bool operator==( boost::detail::sp_nullptr_t, scoped_ptr<T> const & p ) BOOST_SP_NOEXCEPT
{
return p.get() == 0;
}

template<class T> inline bool operator!=( scoped_ptr<T> const & p, boost::detail::sp_nullptr_t ) BOOST_SP_NOEXCEPT
{
return p.get() != 0;
}

template<class T> inline bool operator!=( boost::detail::sp_nullptr_t, scoped_ptr<T> const & p ) BOOST_SP_NOEXCEPT
{
return p.get() != 0;
}

#endif

template<class T> inline void swap(scoped_ptr<T> & a, scoped_ptr<T> & b) BOOST_SP_NOEXCEPT
{
a.swap(b);
}


template<class T> inline T * get_pointer(scoped_ptr<T> const & p) BOOST_SP_NOEXCEPT
{
return p.get();
}

} 

#if defined( BOOST_SP_DISABLE_DEPRECATED )
#pragma GCC diagnostic pop
#endif

#endif 
