#ifndef GET_POINTER_DWA20021219_HPP
#define GET_POINTER_DWA20021219_HPP

#include <boost/config.hpp>

#include <boost/config/no_tr1/memory.hpp>  

namespace boost { 


template<class T> T * get_pointer(T * p)
{
return p;
}


#if !defined( BOOST_NO_AUTO_PTR )

#if defined( __GNUC__ ) && (defined( __GXX_EXPERIMENTAL_CXX0X__ ) || (__cplusplus >= 201103L))
#if defined( BOOST_GCC )
#if BOOST_GCC >= 40600
#define BOOST_CORE_DETAIL_DISABLE_LIBSTDCXX_DEPRECATED_WARNINGS
#endif 
#elif defined( __clang__ ) && defined( __has_warning )
#if __has_warning("-Wdeprecated-declarations")
#define BOOST_CORE_DETAIL_DISABLE_LIBSTDCXX_DEPRECATED_WARNINGS
#endif 
#endif
#endif 

#if defined( BOOST_CORE_DETAIL_DISABLE_LIBSTDCXX_DEPRECATED_WARNINGS )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#define BOOST_CORE_DETAIL_DISABLED_DEPRECATED_WARNINGS
#endif

template<class T> T * get_pointer(std::auto_ptr<T> const& p)
{
return p.get();
}

#if defined( BOOST_CORE_DETAIL_DISABLE_LIBSTDCXX_DEPRECATED_WARNINGS )
#pragma GCC diagnostic pop
#undef BOOST_CORE_DETAIL_DISABLE_LIBSTDCXX_DEPRECATED_WARNINGS
#endif

#endif 

#if !defined( BOOST_NO_CXX11_SMART_PTR )

template<class T> T * get_pointer( std::unique_ptr<T> const& p )
{
return p.get();
}

template<class T> T * get_pointer( std::shared_ptr<T> const& p )
{
return p.get();
}

#endif

} 

#endif 
