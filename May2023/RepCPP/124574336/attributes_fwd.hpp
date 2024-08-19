
#if !defined(BOOST_SPIRIT_ATTRIBUTES_FWD_OCT_01_2009_0715AM)
#define BOOST_SPIRIT_ATTRIBUTES_FWD_OCT_01_2009_0715AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>
#if (defined(__GNUC__) && (__GNUC__ < 4)) || \
(defined(__APPLE__) && defined(__INTEL_COMPILER))
#include <boost/utility/enable_if.hpp>
#endif
#include <boost/spirit/home/support/unused.hpp>

namespace boost { namespace spirit { namespace result_of
{
template <typename Exposed, typename Attribute>
struct extract_from;

template <typename T, typename Attribute>
struct attribute_as;

template <typename T>
struct optional_value;

template <typename Container>
struct begin;

template <typename Container>
struct end;

template <typename Iterator>
struct deref;
}}}

namespace boost { namespace spirit { namespace traits
{
template <typename T, typename Expected, typename Enable = void>
struct is_substitute;

template <typename T, typename Expected, typename Enable = void>
struct is_weak_substitute;

template <typename T, typename Enable = void>
struct is_proxy;

template <typename Attribute, typename Enable = void>
struct attribute_type;

template <typename T>
struct sequence_size;

template <typename Attribute, typename Enable = void>
struct attribute_size;

template <typename Attribute>
typename attribute_size<Attribute>::type
size(Attribute const& attr);

template <typename Component, typename Attribute, typename Enable = void>
struct pass_attribute;

template <typename T, typename Enable = void>
struct optional_attribute;

template <typename Exposed, typename Transformed, typename Domain
, typename Enable = void>
struct transform_attribute;

template <typename Attribute, typename Iterator, typename Enable = void>
struct assign_to_attribute_from_iterators;

template <typename Iterator, typename Attribute>
void assign_to(Iterator const& first, Iterator const& last, Attribute& attr);

template <typename Iterator>
void assign_to(Iterator const&, Iterator const&, unused_type);

template <typename Attribute, typename T, typename Enable = void>
struct assign_to_attribute_from_value;

template <typename Attribute, typename T, typename Enable = void>
struct assign_to_container_from_value;

template <typename T, typename Attribute>
void assign_to(T const& val, Attribute& attr);

template <typename T>
void assign_to(T const&, unused_type);

template <typename Attribute, typename Exposed, typename Enable = void>
struct extract_from_attribute;

template <typename Attribute, typename Exposed, typename Enable = void>
struct extract_from_container;

template <typename Exposed, typename Attribute, typename Context>
typename spirit::result_of::extract_from<Exposed, Attribute>::type
extract_from(Attribute const& attr, Context& ctx
#if (defined(__GNUC__) && (__GNUC__ < 4)) || \
(defined(__APPLE__) && defined(__INTEL_COMPILER))
, typename enable_if<traits::not_is_unused<Attribute> >::type* = NULL
#endif
);

template <typename T, typename Attribute, typename Enable = void>
struct attribute_as;

template <typename T, typename Attribute>
typename spirit::result_of::attribute_as<T, Attribute>::type
as(Attribute const& attr);

template <typename T, typename Attribute>
bool valid_as(Attribute const& attr);

template <typename T, typename Enable = void>
struct variant_which;

template <typename T>
int which(T const& v);

template <typename T, typename Domain = unused_type, typename Enable = void>
struct not_is_variant;

template <typename T, typename Domain = unused_type, typename Enable = void>
struct not_is_optional;

template <typename T, typename Enable = void>
struct clear_value;

template <typename Container, typename Enable = void>
struct container_value;

template <typename Container, typename Enable = void>
struct container_iterator;

template <typename T, typename Enable = void>
struct is_container;

template <typename T, typename Enable = void>
struct is_iterator_range;

template <typename T, typename Attribute, typename Context = unused_type
, typename Iterator = unused_type, typename Enable = void>
struct handles_container;

template <typename Container, typename ValueType, typename Attribute
, typename Sequence, typename Domain, typename Enable = void>
struct pass_through_container;

template <typename Container, typename T, typename Enable = void>
struct push_back_container;

template <typename Container, typename Enable = void>
struct is_empty_container;

template <typename Container, typename Enable = void>
struct make_container_attribute;

template <typename Container, typename Enable = void>
struct begin_container;

template <typename Container, typename Enable = void>
struct end_container;

template <typename Iterator, typename Enable = void>
struct deref_iterator;

template <typename Iterator, typename Enable = void>
struct next_iterator;

template <typename Iterator, typename Enable = void>
struct compare_iterators;

template <typename Out, typename T, typename Enable = void>
struct print_attribute_debug;

template <typename Out, typename T>
void print_attribute(Out&, T const&);

template <typename Out>
void print_attribute(Out&, unused_type);

template <typename Char, typename Enable = void>
struct token_printer_debug;

template<typename Out, typename T>
void print_token(Out&, T const&);

template <typename T, typename Attribute, typename Enable = void>
struct symbols_lookup;

template <typename Attribute, typename T, typename Enable = void>
struct symbols_value;

template <typename Attribute, typename Domain>
struct alternative_attribute_transform;

template <typename Attribute, typename Domain>
struct sequence_attribute_transform;

template <typename Attribute, typename Domain>
struct permutation_attribute_transform;

template <typename Attribute, typename Domain>
struct sequential_or_attribute_transform;
}}}

#endif

