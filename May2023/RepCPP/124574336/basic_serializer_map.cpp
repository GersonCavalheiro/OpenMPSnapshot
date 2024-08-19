


#if (defined _MSC_VER) && (_MSC_VER == 1200)
#  pragma warning (disable : 4786) 
#endif

#include <set>
#include <utility>

#define BOOST_ARCHIVE_SOURCE
#define BOOST_SERIALIZATION_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/serialization/throw_exception.hpp>

#include <boost/archive/archive_exception.hpp>
#include <boost/archive/detail/basic_serializer.hpp>
#include <boost/archive/detail/basic_serializer_map.hpp>

namespace boost {
namespace serialization {
class extended_type_info;
}
namespace archive {
namespace detail {

bool  
basic_serializer_map::type_info_pointer_compare::operator()(
const basic_serializer * lhs, const basic_serializer * rhs
) const {
return *lhs < *rhs;
}

BOOST_ARCHIVE_DECL bool
basic_serializer_map::insert(const basic_serializer * bs){

m_map.insert(bs);



return true;
}

BOOST_ARCHIVE_DECL void 
basic_serializer_map::erase(const basic_serializer * bs){
map_type::iterator it = m_map.begin();
map_type::iterator it_end = m_map.end();

while(it != it_end){
if(*it == bs)
m_map.erase(it++);
else
it++;
}
}
BOOST_ARCHIVE_DECL const basic_serializer *
basic_serializer_map::find(
const boost::serialization::extended_type_info & eti
) const {
const basic_serializer_arg bs(eti);
map_type::const_iterator it;
it = m_map.find(& bs);
if(it == m_map.end()){
BOOST_ASSERT(false);
return 0;
}
return *it;
}

} 
} 
} 

