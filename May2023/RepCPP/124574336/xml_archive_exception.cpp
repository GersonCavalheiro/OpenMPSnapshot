


#if (defined _MSC_VER) && (_MSC_VER == 1200)
#  pragma warning (disable : 4786) 
#endif


#include <exception>
#include <string>

#include <boost/assert.hpp>

#define BOOST_ARCHIVE_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/archive/xml_archive_exception.hpp>

namespace boost {
namespace archive {

BOOST_ARCHIVE_DECL
xml_archive_exception::xml_archive_exception(
exception_code c, 
const char * e1,
const char * e2
) : 
archive_exception(other_exception, e1, e2)
{
switch(c){
case xml_archive_parsing_error:
archive_exception::append(0, "unrecognized XML syntax");
break;
case xml_archive_tag_mismatch:{
unsigned int l;
l = archive_exception::append(0, "XML start/end tag mismatch");
if(NULL != e1){
l = archive_exception::append(l, " - ");
archive_exception::append(l, e1);
}    
break;
}
case xml_archive_tag_name_error:
archive_exception::append(0, "Invalid XML tag name");
break;
default:
BOOST_ASSERT(false);
archive_exception::append(0, "programming error");
break;
}
}

BOOST_ARCHIVE_DECL
xml_archive_exception::xml_archive_exception(xml_archive_exception const & oth) :
archive_exception(oth)
{
}

BOOST_ARCHIVE_DECL xml_archive_exception::~xml_archive_exception() BOOST_NOEXCEPT_OR_NOTHROW {}

} 
} 
