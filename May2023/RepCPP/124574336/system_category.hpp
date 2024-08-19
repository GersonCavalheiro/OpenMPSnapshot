#ifndef BOOST_SYSTEM_DETAIL_SYSTEM_CATEGORY_HPP_INCLUDED
#define BOOST_SYSTEM_DETAIL_SYSTEM_CATEGORY_HPP_INCLUDED


#include <boost/system/detail/error_category.hpp>
#include <boost/system/detail/config.hpp>
#include <boost/config.hpp>

namespace boost
{

namespace system
{

namespace detail
{


#if ( defined( BOOST_GCC ) && BOOST_GCC >= 40600 ) || defined( BOOST_CLANG )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

class BOOST_SYMBOL_VISIBLE system_error_category: public error_category
{
public:

BOOST_SYSTEM_CONSTEXPR system_error_category() BOOST_NOEXCEPT:
error_category( detail::system_category_id )
{
}

const char * name() const BOOST_NOEXCEPT BOOST_OVERRIDE
{
return "system";
}

error_condition default_error_condition( int ev ) const BOOST_NOEXCEPT BOOST_OVERRIDE;

std::string message( int ev ) const BOOST_OVERRIDE;
char const * message( int ev, char * buffer, std::size_t len ) const BOOST_NOEXCEPT BOOST_OVERRIDE;
};

#if ( defined( BOOST_GCC ) && BOOST_GCC >= 40600 ) || defined( BOOST_CLANG )
#pragma GCC diagnostic pop
#endif

} 


#if defined(BOOST_SYSTEM_HAS_CONSTEXPR)

namespace detail
{

template<class T> struct BOOST_SYMBOL_VISIBLE system_cat_holder
{
static constexpr system_error_category instance{};
};

#if defined(BOOST_NO_CXX17_INLINE_VARIABLES)
template<class T> constexpr system_error_category system_cat_holder<T>::instance;
#endif

} 

constexpr error_category const & system_category() BOOST_NOEXCEPT
{
return detail::system_cat_holder<void>::instance;
}

#else 

#if !defined(__SUNPRO_CC) 
inline error_category const & system_category() BOOST_NOEXCEPT BOOST_SYMBOL_VISIBLE;
#endif

inline error_category const & system_category() BOOST_NOEXCEPT
{
static const detail::system_error_category instance;
return instance;
}

#endif 


#ifdef BOOST_SYSTEM_ENABLE_DEPRECATED

BOOST_SYSTEM_DEPRECATED("please use system_category()") inline const error_category & get_system_category() { return system_category(); }
BOOST_SYSTEM_DEPRECATED("please use system_category()") static const error_category & native_ecat BOOST_ATTRIBUTE_UNUSED = system_category();

#endif

} 

} 

#endif 
