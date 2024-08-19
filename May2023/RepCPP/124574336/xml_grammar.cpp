


#if (defined _MSC_VER) && (_MSC_VER == 1200)
#  pragma warning (disable : 4786) 
#endif

#include <boost/config.hpp>

#define BOOST_ARCHIVE_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/archive/impl/basic_xml_grammar.hpp>

using namespace boost::spirit::classic;

#if ! defined(__SGI_STL_PORT) \
&& defined(BOOST_RWSTD_VER) && BOOST_RWSTD_VER<=0x020101
#include <string>
namespace std {
template<>
inline string & 
string::replace (
char * first1, 
char * last1,
const char * first2,
const char * last2
){
replace(first1-begin(),last1-first1,first2,last2-first2,0,last2-first2);
return *this;
}
} 
#endif

namespace boost {
namespace archive {

typedef basic_xml_grammar<char> xml_grammar;


template<>
void xml_grammar::init_chset(){
Char = chset_t("\x9\xA\xD\x20-\x7f\x80\x81-\xFF"); 
Letter = chset_t("\x41-\x5A\x61-\x7A\xC0-\xD6\xD8-\xF6\xF8-\xFF");
Digit = chset_t("0-9");
Extender = chset_t('\xB7');
Sch = chset_t("\x20\x9\xD\xA");
NameChar = Letter | Digit | chset_p("._:-") | Extender ;
}

} 
} 

#include "basic_xml_grammar.ipp"

namespace boost {
namespace archive {

template class basic_xml_grammar<char>;

} 
} 

