
#ifndef BOOST_LEXICAL_CAST_BAD_LEXICAL_CAST_HPP
#define BOOST_LEXICAL_CAST_BAD_LEXICAL_CAST_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

#include <exception>
#include <typeinfo>
#include <boost/throw_exception.hpp>

namespace boost
{
class BOOST_SYMBOL_VISIBLE bad_lexical_cast :
#if defined(BOOST_MSVC) && defined(_HAS_EXCEPTIONS) && !_HAS_EXCEPTIONS 
public std::exception 
#else 
public std::bad_cast 
#endif 

#if defined(BOOST_BORLANDC) && BOOST_WORKAROUND( BOOST_BORLANDC, < 0x560 )
, public std::exception
#endif

{
public:
bad_lexical_cast() BOOST_NOEXCEPT
#ifndef BOOST_NO_TYPEID
: source(&typeid(void)), target(&typeid(void))
#endif
{}

const char *what() const BOOST_NOEXCEPT_OR_NOTHROW BOOST_OVERRIDE {
return "bad lexical cast: "
"source type value could not be interpreted as target";
}

~bad_lexical_cast() BOOST_NOEXCEPT_OR_NOTHROW BOOST_OVERRIDE
{}

#ifndef BOOST_NO_TYPEID
private:
#ifdef BOOST_NO_STD_TYPEINFO
typedef ::type_info type_info_t;
#else
typedef ::std::type_info type_info_t;
#endif
public:
bad_lexical_cast(
const type_info_t &source_type_arg,
const type_info_t &target_type_arg) BOOST_NOEXCEPT
: source(&source_type_arg), target(&target_type_arg)
{}

const type_info_t &source_type() const BOOST_NOEXCEPT {
return *source;
}

const type_info_t &target_type() const BOOST_NOEXCEPT {
return *target;
}

private:
const type_info_t *source;
const type_info_t *target;
#endif
};

namespace conversion { namespace detail {
#ifdef BOOST_NO_TYPEID
template <class S, class T>
inline void throw_bad_cast() {
boost::throw_exception(bad_lexical_cast());
}
#else
template <class S, class T>
inline void throw_bad_cast() {
boost::throw_exception(bad_lexical_cast(typeid(S), typeid(T)));
}
#endif
}} 

} 

#endif 
