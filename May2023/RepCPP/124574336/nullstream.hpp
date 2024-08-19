

#ifndef BOOST_TEST_UTILS_NULLSTREAM_HPP
#define BOOST_TEST_UTILS_NULLSTREAM_HPP

#include <ostream>    
#include <streambuf>  
#include <string>     

#include <boost/utility/base_from_member.hpp>

#include <boost/test/detail/suppress_warnings.hpp>


namespace boost {


template<typename CharType, class CharTraits = ::std::char_traits<CharType> >
class basic_nullbuf : public ::std::basic_streambuf<CharType, CharTraits> {
typedef ::std::basic_streambuf<CharType, CharTraits>  base_type;
public:
typedef typename base_type::char_type    char_type;
typedef typename base_type::traits_type  traits_type;
typedef typename base_type::int_type     int_type;
typedef typename base_type::pos_type     pos_type;
typedef typename base_type::off_type     off_type;


protected:


virtual  ::std::streamsize  xsputn( char_type const* , ::std::streamsize n )   { return n; } 
virtual  int_type           overflow( int_type c = traits_type::eof() )         { return traits_type::not_eof( c ); }
};

typedef basic_nullbuf<char>      nullbuf;
typedef basic_nullbuf<wchar_t>  wnullbuf;


#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable: 4355) 
#endif

template< typename CharType, class CharTraits = ::std::char_traits<CharType> >
class basic_onullstream : private boost::base_from_member<basic_nullbuf<CharType, CharTraits> >
, public ::std::basic_ostream<CharType, CharTraits> {
typedef boost::base_from_member<basic_nullbuf<CharType, CharTraits> >   pbase_type;
typedef ::std::basic_ostream<CharType, CharTraits>                      base_type;
public:
basic_onullstream() : pbase_type(), base_type( &this->pbase_type::member ) {}
};

#ifdef BOOST_MSVC
# pragma warning(default: 4355)
# pragma warning(pop)
#endif

typedef basic_onullstream<char>      onullstream;
typedef basic_onullstream<wchar_t>  wonullstream;

}  


#include <boost/test/detail/enable_warnings.hpp>

#endif  
