#ifndef BOOST_LOCALE_BOUNDARY_FACETS_HPP_INCLUDED
#define BOOST_LOCALE_BOUNDARY_FACETS_HPP_INCLUDED

#include <boost/locale/config.hpp>
#include <boost/locale/boundary/types.hpp>
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4275 4251 4231 4660)
#endif
#include <locale>
#include <vector>




namespace boost {

namespace locale {

namespace boundary {


struct break_info {

break_info() : 
offset(0),
rule(0)
{
}
break_info(size_t v) :
offset(v),
rule(0)
{
}

size_t offset;
rule_type rule;

bool operator<(break_info const &other) const
{
return offset < other.offset;
}
};

typedef std::vector<break_info> index_type;


template<typename CharType>
class boundary_indexing;

#ifdef BOOST_LOCALE_DOXYGEN
template<typename Char>
class BOOST_LOCALE_DECL boundary_indexing : public std::locale::facet {
public:
boundary_indexing(size_t refs=0) : std::locale::facet(refs)
{
}
virtual index_type map(boundary_type t,Char const *begin,Char const *end) const = 0;
static std::locale::id id;

#if defined (__SUNPRO_CC) && defined (_RWSTD_VER)
std::locale::id& __get_id (void) const { return id; }
#endif
};

#else

template<>
class BOOST_LOCALE_DECL boundary_indexing<char> : public std::locale::facet {
public:
boundary_indexing(size_t refs=0) : std::locale::facet(refs)
{
}
virtual index_type map(boundary_type t,char const *begin,char const *end) const = 0;
static std::locale::id id;
#if defined (__SUNPRO_CC) && defined (_RWSTD_VER)
std::locale::id& __get_id (void) const { return id; }
#endif
};

template<>
class BOOST_LOCALE_DECL boundary_indexing<wchar_t> : public std::locale::facet {
public:
boundary_indexing(size_t refs=0) : std::locale::facet(refs)
{
}
virtual index_type map(boundary_type t,wchar_t const *begin,wchar_t const *end) const = 0;

static std::locale::id id;
#if defined (__SUNPRO_CC) && defined (_RWSTD_VER)
std::locale::id& __get_id (void) const { return id; }
#endif
};

#ifdef BOOST_LOCALE_ENABLE_CHAR16_T
template<>
class BOOST_LOCALE_DECL boundary_indexing<char16_t> : public std::locale::facet {
public:
boundary_indexing(size_t refs=0) : std::locale::facet(refs)
{
}
virtual index_type map(boundary_type t,char16_t const *begin,char16_t const *end) const = 0;
static std::locale::id id;
#if defined (__SUNPRO_CC) && defined (_RWSTD_VER)
std::locale::id& __get_id (void) const { return id; }
#endif
};
#endif

#ifdef BOOST_LOCALE_ENABLE_CHAR32_T
template<>
class BOOST_LOCALE_DECL boundary_indexing<char32_t> : public std::locale::facet {
public:
boundary_indexing(size_t refs=0) : std::locale::facet(refs)
{
}
virtual index_type map(boundary_type t,char32_t const *begin,char32_t const *end) const = 0;
static std::locale::id id;
#if defined (__SUNPRO_CC) && defined (_RWSTD_VER)
std::locale::id& __get_id (void) const { return id; }
#endif
};
#endif

#endif



} 

} 
} 


#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif
