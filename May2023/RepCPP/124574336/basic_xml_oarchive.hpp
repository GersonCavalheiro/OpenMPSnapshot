#ifndef BOOST_ARCHIVE_BASIC_XML_OARCHIVE_HPP
#define BOOST_ARCHIVE_BASIC_XML_OARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#include <boost/mpl/assert.hpp>

#include <boost/archive/detail/common_oarchive.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/string.hpp>

#include <boost/archive/detail/abi_prefix.hpp> 

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace archive {

namespace detail {
template<class Archive> class interface_oarchive;
} 

template<class Archive>
class BOOST_SYMBOL_VISIBLE basic_xml_oarchive :
public detail::common_oarchive<Archive>
{
unsigned int depth;
bool pending_preamble;
#ifdef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
protected:
friend class detail::interface_oarchive<Archive>;
#endif
bool indent_next;
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
indent();
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
init();
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
windup();
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
write_attribute(
const char *attribute_name,
int t,
const char *conjunction = "=\""
);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
write_attribute(
const char *attribute_name,
const char *key
);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save_start(const char *name);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save_end(const char *name);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
end_preamble();

template<class T>
void save_override(T & t)
{
BOOST_MPL_ASSERT((serialization::is_wrapper< T >));
this->detail_common_oarchive::save_override(t);
}

typedef detail::common_oarchive<Archive> detail_common_oarchive;
template<class T>
void save_override(
const ::boost::serialization::nvp< T > & t
){
this->This()->save_start(t.name());
this->detail_common_oarchive::save_override(t.const_value());
this->This()->save_end(t.name());
}

BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save_override(const class_id_type & t);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save_override(const class_id_optional_type & t);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save_override(const class_id_reference_type & t);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save_override(const object_id_type & t);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save_override(const object_reference_type & t);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save_override(const version_type & t);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save_override(const class_name_type & t);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
save_override(const tracking_type & t);

BOOST_ARCHIVE_OR_WARCHIVE_DECL
basic_xml_oarchive(unsigned int flags);
BOOST_ARCHIVE_OR_WARCHIVE_DECL
~basic_xml_oarchive() BOOST_OVERRIDE;
};

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
