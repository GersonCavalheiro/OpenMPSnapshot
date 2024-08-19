


#if (defined _MSC_VER) && (_MSC_VER == 1200)
#  pragma warning (disable : 4786) 
#endif

#include <algorithm>
#include <set>
#include <utility>
#include <boost/assert.hpp>
#include <cstddef> 

#include <cstring>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{ using ::strcmp; }
#endif

#include <boost/config.hpp> 

#include <boost/core/no_exceptions_support.hpp>

#define BOOST_SERIALIZATION_SOURCE
#include <boost/serialization/config.hpp>
#include <boost/serialization/singleton.hpp>
#include <boost/serialization/force_include.hpp>
#include <boost/serialization/extended_type_info.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost { 
namespace serialization {
namespace detail {

struct key_compare
{
bool
operator()(
const extended_type_info * lhs, 
const extended_type_info * rhs
) const {
if(lhs == rhs)
return false;
const char * l = lhs->get_key();
BOOST_ASSERT(NULL != l);
const char * r = rhs->get_key();
BOOST_ASSERT(NULL != r);
if(l == r)
return false;
return std::strcmp(l, r) < 0;
}
};

typedef std::multiset<const extended_type_info *, key_compare> ktmap;

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

class extended_type_info_arg : public extended_type_info
{
bool
is_less_than(const extended_type_info & ) const BOOST_OVERRIDE {
BOOST_ASSERT(false);
return false;
}
bool
is_equal(const extended_type_info & ) const BOOST_OVERRIDE {
BOOST_ASSERT(false);
return false;
}
const char * get_debug_info() const BOOST_OVERRIDE {
return get_key();
}
void * construct(unsigned int , ...) const BOOST_OVERRIDE {
BOOST_ASSERT(false);
return NULL;
}
void destroy(void const * const ) const BOOST_OVERRIDE {
BOOST_ASSERT(false);
}
public:
extended_type_info_arg(const char * key) :
extended_type_info(0, key)
{}

~extended_type_info_arg() BOOST_OVERRIDE {}
};

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

} 

BOOST_SERIALIZATION_DECL void  
extended_type_info::key_register() const{
if(NULL == get_key())
return;
singleton<detail::ktmap>::get_mutable_instance().insert(this);
}

BOOST_SERIALIZATION_DECL void  
extended_type_info::key_unregister() const{
if(NULL == get_key())
return;
if(! singleton<detail::ktmap>::is_destroyed()){
detail::ktmap & x = singleton<detail::ktmap>::get_mutable_instance();
detail::ktmap::iterator start = x.lower_bound(this);
detail::ktmap::iterator end = x.upper_bound(this);
for(;start != end; ++start){
if(this == *start){
x.erase(start);
break;
}
}
}
}

BOOST_SERIALIZATION_DECL const extended_type_info *
extended_type_info::find(const char *key) {
BOOST_ASSERT(NULL != key);
const detail::ktmap & k = singleton<detail::ktmap>::get_const_instance();
const detail::extended_type_info_arg eti_key(key);
const detail::ktmap::const_iterator it = k.find(& eti_key);
if(k.end() == it)
return NULL;
return *(it);
}

BOOST_SERIALIZATION_DECL
extended_type_info::extended_type_info(
const unsigned int type_info_key,
const char * key
) :
m_type_info_key(type_info_key),
m_key(key)
{
}

BOOST_SERIALIZATION_DECL 
extended_type_info::~extended_type_info(){
}

BOOST_SERIALIZATION_DECL bool
extended_type_info::operator<(const extended_type_info &rhs) const {
if(this == & rhs)
return false;
if(m_type_info_key == rhs.m_type_info_key){
return is_less_than(rhs);
}
if(m_type_info_key < rhs.m_type_info_key)
return true;
return false;
}

BOOST_SERIALIZATION_DECL bool
extended_type_info::operator==(const extended_type_info &rhs) const {
if(this == & rhs)
return true;
if(m_type_info_key != rhs.m_type_info_key){
return false;
}
return is_equal(rhs);
}

} 
} 
