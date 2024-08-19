#ifndef BOOST_CORE_REF_HPP
#define BOOST_CORE_REF_HPP


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <boost/config.hpp>
#include <boost/config/workaround.hpp>
#include <boost/core/addressof.hpp>





namespace boost
{

#if defined( BOOST_MSVC ) && BOOST_WORKAROUND( BOOST_MSVC, == 1600 )

struct ref_workaround_tag {};

#endif



template<class T> class reference_wrapper
{
public:

typedef T type;


BOOST_FORCEINLINE explicit reference_wrapper(T& t): t_(boost::addressof(t)) {}

#if defined( BOOST_MSVC ) && BOOST_WORKAROUND( BOOST_MSVC, == 1600 )

BOOST_FORCEINLINE explicit reference_wrapper( T & t, ref_workaround_tag ): t_( boost::addressof( t ) ) {}

#endif

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)

BOOST_DELETED_FUNCTION(reference_wrapper(T&& t))
public:
#endif


BOOST_FORCEINLINE operator T& () const { return *t_; }


BOOST_FORCEINLINE T& get() const { return *t_; }


BOOST_FORCEINLINE T* get_pointer() const { return t_; }

private:

T* t_;
};



#if defined( BOOST_BORLANDC ) && BOOST_WORKAROUND( BOOST_BORLANDC, BOOST_TESTED_AT(0x581) )
#  define BOOST_REF_CONST
#else
#  define BOOST_REF_CONST const
#endif



template<class T> BOOST_FORCEINLINE reference_wrapper<T> BOOST_REF_CONST ref( T & t )
{
#if defined( BOOST_MSVC ) && BOOST_WORKAROUND( BOOST_MSVC, == 1600 )

return reference_wrapper<T>( t, ref_workaround_tag() );

#else

return reference_wrapper<T>( t );

#endif
}



template<class T> BOOST_FORCEINLINE reference_wrapper<T const> BOOST_REF_CONST cref( T const & t )
{
return reference_wrapper<T const>(t);
}

#undef BOOST_REF_CONST

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)


#if defined(BOOST_NO_CXX11_DELETED_FUNCTIONS)
#  define BOOST_REF_DELETE
#else
#  define BOOST_REF_DELETE = delete
#endif



template<class T> void ref(T const&&) BOOST_REF_DELETE;


template<class T> void cref(T const&&) BOOST_REF_DELETE;

#undef BOOST_REF_DELETE

#endif



template<typename T> struct is_reference_wrapper
{
BOOST_STATIC_CONSTANT( bool, value = false );
};


template<typename T> struct is_reference_wrapper< reference_wrapper<T> >
{
BOOST_STATIC_CONSTANT( bool, value = true );
};

#if !defined(BOOST_NO_CV_SPECIALIZATIONS)

template<typename T> struct is_reference_wrapper< reference_wrapper<T> const >
{
BOOST_STATIC_CONSTANT( bool, value = true );
};

template<typename T> struct is_reference_wrapper< reference_wrapper<T> volatile >
{
BOOST_STATIC_CONSTANT( bool, value = true );
};

template<typename T> struct is_reference_wrapper< reference_wrapper<T> const volatile >
{
BOOST_STATIC_CONSTANT( bool, value = true );
};

#endif 






template<typename T> struct unwrap_reference
{
typedef T type;
};


template<typename T> struct unwrap_reference< reference_wrapper<T> >
{
typedef T type;
};

#if !defined(BOOST_NO_CV_SPECIALIZATIONS)

template<typename T> struct unwrap_reference< reference_wrapper<T> const >
{
typedef T type;
};

template<typename T> struct unwrap_reference< reference_wrapper<T> volatile >
{
typedef T type;
};

template<typename T> struct unwrap_reference< reference_wrapper<T> const volatile >
{
typedef T type;
};

#endif 





template<class T> BOOST_FORCEINLINE typename unwrap_reference<T>::type& unwrap_ref( T & t )
{
return t;
}



template<class T> BOOST_FORCEINLINE T* get_pointer( reference_wrapper<T> const & r )
{
return r.get_pointer();
}


} 

#endif 
