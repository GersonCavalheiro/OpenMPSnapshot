

#include <algorithm>
#include <list>

#if (defined _MSC_VER) && (_MSC_VER == 1200)
#  pragma warning (disable : 4786) 
#endif

#include <cstdlib>
#include <cstddef> 

#include <boost/config.hpp>
#ifdef BOOST_NO_STDC_NAMESPACE
namespace std{ 
using ::rand; 
using ::size_t;
}
#endif

#include <boost/serialization/pfto.hpp>

#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/insert_linebreaks.hpp>
#include <boost/archive/iterators/remove_whitespace.hpp>
#include <boost/archive/iterators/transform_width.hpp>

#include "../test/test_tools.hpp"

#include <iostream>

template<typename CharType>
void test_base64(){
CharType rawdata[150];
std::size_t size = sizeof(rawdata) / sizeof(CharType);
CharType * rptr;
for(rptr = rawdata + size; rptr-- > rawdata;)
*rptr = std::rand();

typedef std::list<CharType> text_base64_type;
text_base64_type text_base64;

typedef 
boost::archive::iterators::insert_linebreaks<
boost::archive::iterators::base64_from_binary<
boost::archive::iterators::transform_width<
CharType *
,6
,sizeof(CharType) * 8
>
> 
,72
> 
translate_out;

std::copy(
translate_out(BOOST_MAKE_PFTO_WRAPPER(static_cast<CharType *>(rawdata))),
translate_out(BOOST_MAKE_PFTO_WRAPPER(rawdata + size)),
std::back_inserter(text_base64)
);

typedef 
boost::archive::iterators::transform_width<
boost::archive::iterators::binary_from_base64<
boost::archive::iterators::remove_whitespace<
BOOST_DEDUCED_TYPENAME text_base64_type::iterator
>
>,
sizeof(CharType) * 8,
6
> translate_in;

BOOST_CHECK(
std::equal(
rawdata,
rawdata + size,
translate_in(BOOST_MAKE_PFTO_WRAPPER(text_base64.begin()))
)
);

}

int
test_main( int argc, char* argv[] )
{
test_base64<char>();
#ifndef BOOST_NO_CWCHAR
test_base64<wchar_t>();
#endif
return EXIT_SUCCESS;
}
