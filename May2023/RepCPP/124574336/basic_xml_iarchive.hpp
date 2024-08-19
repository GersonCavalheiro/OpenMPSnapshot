#ifndef BOOST_ARCHIVE_BASIC_XML_IARCHIVE_HPP
#define BOOST_ARCHIVE_BASIC_XML_IARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#include <boost/mpl/assert.hpp>

#include <boost/archive/detail/common_iarchive.hpp>
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
template<class Archive> class interface_iarchive;
} 

template<class Archive>
class BOOST_SYMBOL_VISIBLE basic_xml_iarchive :
public detail::common_iarchive<Archive>
{
unsigned int depth;
#ifdef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
protected:
friend class detail::interface_iarchive<Archive>;
#endif
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
load_start(const char *name);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
load_end(const char *name);

template<class T>
void load_override(T & t)
{
BOOST_MPL_ASSERT((serialization::is_wrapper< T >));
this->detail_common_iarchive::load_override(t);
}

typedef detail::common_iarchive<Archive> detail_common_iarchive;
template<class T>
void load_override(
const boost::serialization::nvp< T > & t
){
this->This()->load_start(t.name());
this->detail_common_iarchive::load_override(t.value());
this->This()->load_end(t.name());
}

BOOST_ARCHIVE_OR_WARCHIVE_DECL void
load_override(class_id_type & t);
void load_override(class_id_optional_type & ){}
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
load_override(object_id_type & t);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
load_override(version_type & t);
BOOST_ARCHIVE_OR_WARCHIVE_DECL void
load_override(tracking_type & t);

BOOST_ARCHIVE_OR_WARCHIVE_DECL
basic_xml_iarchive(unsigned int flags);
BOOST_ARCHIVE_OR_WARCHIVE_DECL
~basic_xml_iarchive() BOOST_OVERRIDE;
};

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
