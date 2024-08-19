


#include <algorithm>
#include <cstddef> 
#include <set>
#include <typeinfo>

#include <boost/assert.hpp>
#include <boost/core/no_exceptions_support.hpp>

#define BOOST_SERIALIZATION_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/serialization/singleton.hpp>
#include <boost/serialization/extended_type_info_typeid.hpp>

namespace boost { 
namespace serialization { 
namespace typeid_system {

#define EXTENDED_TYPE_INFO_TYPE_KEY 1

struct type_compare
{
bool
operator()(
const extended_type_info_typeid_0 * lhs,
const extended_type_info_typeid_0 * rhs
) const {
return lhs->is_less_than(*rhs);
}
};

typedef std::multiset<
const extended_type_info_typeid_0 *,
type_compare
> tkmap;

BOOST_SERIALIZATION_DECL bool
extended_type_info_typeid_0::is_less_than(
const boost::serialization::extended_type_info & rhs
) const {
if(this == & rhs)
return false;
return 0 != m_ti->before(
*(static_cast<const extended_type_info_typeid_0 &>(rhs).m_ti)
);
}

BOOST_SERIALIZATION_DECL bool
extended_type_info_typeid_0::is_equal(
const boost::serialization::extended_type_info & rhs
) const {
return 
! (
* m_ti 
!= *(static_cast<const extended_type_info_typeid_0 &>(rhs).m_ti)
)
;
}

BOOST_SERIALIZATION_DECL
extended_type_info_typeid_0::extended_type_info_typeid_0(
const char * key
) :
extended_type_info(EXTENDED_TYPE_INFO_TYPE_KEY, key),
m_ti(NULL)
{}

BOOST_SERIALIZATION_DECL
extended_type_info_typeid_0::~extended_type_info_typeid_0()
{}

BOOST_SERIALIZATION_DECL void 
extended_type_info_typeid_0::type_register(const std::type_info & ti){
m_ti = & ti;
singleton<tkmap>::get_mutable_instance().insert(this);
}

BOOST_SERIALIZATION_DECL void 
extended_type_info_typeid_0::type_unregister()
{
if(NULL != m_ti){
if(! singleton<tkmap>::is_destroyed()){
tkmap & x = singleton<tkmap>::get_mutable_instance();

while(true){
const tkmap::iterator & it = x.find(this);
if(it == x.end())
break;
x.erase(it);
}
}
}
m_ti = NULL;
}

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

class extended_type_info_typeid_arg : 
public extended_type_info_typeid_0
{
void * construct(unsigned int , ...) const BOOST_OVERRIDE {
BOOST_ASSERT(false);
return NULL;
}
void destroy(void const * const ) const BOOST_OVERRIDE {
BOOST_ASSERT(false);
}
public:
extended_type_info_typeid_arg(const std::type_info & ti) :
extended_type_info_typeid_0(NULL)
{ 
m_ti = & ti;
}
~extended_type_info_typeid_arg() BOOST_OVERRIDE {
m_ti = NULL;
}
};

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

BOOST_SERIALIZATION_DECL const extended_type_info *
extended_type_info_typeid_0::get_extended_type_info(
const std::type_info & ti
) const {
typeid_system::extended_type_info_typeid_arg etia(ti);
const tkmap & t = singleton<tkmap>::get_const_instance();
const tkmap::const_iterator it = t.find(& etia);
if(t.end() == it)
return NULL;
return *(it);
}

} 
} 
} 
