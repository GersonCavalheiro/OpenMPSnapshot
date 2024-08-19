
#if !defined(BOOST_SPIRIT_ATTRIBUTES_JANUARY_29_2007_0954AM)
#define BOOST_SPIRIT_ATTRIBUTES_JANUARY_29_2007_0954AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/support/has_semantic_action.hpp>
#include <boost/spirit/home/support/attributes_fwd.hpp>
#include <boost/spirit/home/support/container.hpp>
#include <boost/spirit/home/support/detail/hold_any.hpp>
#include <boost/spirit/home/support/detail/as_variant.hpp>
#include <boost/optional/optional.hpp>
#include <boost/fusion/include/transform.hpp>
#include <boost/fusion/include/filter_if.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/push_front.hpp>
#include <boost/fusion/include/pop_front.hpp>
#include <boost/fusion/include/is_sequence.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/is_view.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/is_reference.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/end.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/distance.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/has_xxx.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/proto/traits.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/variant.hpp>
#include <boost/range/range_fwd.hpp>
#include <boost/config.hpp>
#include <iterator> 
#include <vector>
#include <utility>
#include <ios>

namespace boost { namespace spirit { namespace traits
{

namespace detail
{
template <typename T, typename Expected>
struct value_type_is_substitute
: is_substitute<
typename container_value<T>::type
, typename container_value<Expected>::type>
{};

template <typename T, typename Expected, typename Enable = void>
struct is_substitute_impl : is_same<T, Expected> {};

template <typename T, typename Expected>
struct is_substitute_impl<T, Expected,
typename enable_if<
mpl::and_<
fusion::traits::is_sequence<T>,
fusion::traits::is_sequence<Expected>,
mpl::equal<T, Expected, is_substitute<mpl::_1, mpl::_2> >
>
>::type>
: mpl::true_ {};

template <typename T, typename Expected>
struct is_substitute_impl<T, Expected,
typename enable_if<
mpl::and_<
is_container<T>,
is_container<Expected>,
detail::value_type_is_substitute<T, Expected>
>
>::type>
: mpl::true_ {};
}

template <typename T, typename Expected, typename Enable >
struct is_substitute
: detail::is_substitute_impl<T, Expected> {};

template <typename T, typename Expected>
struct is_substitute<optional<T>, optional<Expected> >
: is_substitute<T, Expected> {};

template <typename T>
struct is_substitute<T, T
, typename enable_if<not_is_optional<T> >::type>
: mpl::true_ {};

namespace detail
{
template <typename T, typename Expected, typename Enable = void>
struct is_weak_substitute_impl : is_convertible<T, Expected> {};


template <typename T, typename Expected>
struct value_type_is_weak_substitute
: is_weak_substitute<
typename container_value<T>::type
, typename container_value<Expected>::type>
{};

template <typename T, typename Expected>
struct is_weak_substitute_impl<T, Expected,
typename enable_if<
mpl::and_<
is_container<T>
, is_container<Expected>
, value_type_is_weak_substitute<T, Expected> >
>::type>
: mpl::true_ {};

template <typename T, typename Expected>
struct is_weak_substitute_impl<T, Expected,
typename enable_if<
mpl::and_<
fusion::traits::is_sequence<T>
, fusion::traits::is_sequence<Expected>
, mpl::equal<T, Expected, is_weak_substitute<mpl::_1, mpl::_2> > >
>::type>
: mpl::true_ {};

template <typename T, typename Expected>
struct is_weak_substitute_impl<T, Expected,
typename enable_if<
mpl::and_<
mpl::not_<fusion::traits::is_sequence<T> >
, fusion::traits::is_sequence<Expected> >
>::type>
: mpl::false_ {};
}

template <typename T, typename Expected, typename Enable >
struct is_weak_substitute
: detail::is_weak_substitute_impl<T, Expected> {};

template <typename T, typename Expected>
struct is_weak_substitute<optional<T>, optional<Expected> >
: is_weak_substitute<T, Expected> {};

template <typename T, typename Expected>
struct is_weak_substitute<optional<T>, Expected>
: is_weak_substitute<T, Expected> {};

template <typename T, typename Expected>
struct is_weak_substitute<T, optional<Expected> >
: is_weak_substitute<T, Expected> {};

#if !defined(BOOST_VARIANT_DO_NOT_USE_VARIADIC_TEMPLATES)
template <typename T, typename Expected>
struct is_weak_substitute<boost::variant<T>, Expected>
: is_weak_substitute<T, Expected>
{};

template <typename T0, typename T1, typename ...TN, typename Expected>
struct is_weak_substitute<boost::variant<T0, T1, TN...>,
Expected>
: mpl::bool_<is_weak_substitute<T0, Expected>::type::value &&
is_weak_substitute<boost::variant<T1, TN...>, Expected>::type::value>
{};
#else
#define BOOST_SPIRIT_IS_WEAK_SUBSTITUTE(z, N, _)                              \
is_weak_substitute<BOOST_PP_CAT(T, N), Expected>::type::value &&          \


template <typename Expected>
struct is_weak_substitute<boost::detail::variant::void_, Expected>
: mpl::true_
{};

template <BOOST_VARIANT_ENUM_PARAMS(typename T), typename Expected>
struct is_weak_substitute<
boost::variant<BOOST_VARIANT_ENUM_PARAMS(T)>, Expected>
: mpl::bool_<BOOST_PP_REPEAT(BOOST_VARIANT_LIMIT_TYPES
, BOOST_SPIRIT_IS_WEAK_SUBSTITUTE, _) true>
{};

#undef BOOST_SPIRIT_IS_WEAK_SUBSTITUTE
#endif

template <typename T>
struct is_weak_substitute<T, T
, typename enable_if<
mpl::and_<not_is_optional<T>, not_is_variant<T> >
>::type>
: mpl::true_ {};

template <typename T, typename Enable>
struct is_proxy : mpl::false_ {};

template <typename T>
struct is_proxy<T,
typename enable_if<
mpl::and_<
fusion::traits::is_sequence<T>,
fusion::traits::is_view<T>
>
>::type>
: mpl::true_ {};

namespace detail
{
BOOST_MPL_HAS_XXX_TRAIT_DEF(adapted_variant_tag)
}

template <typename T, typename Domain, typename Enable>
struct not_is_variant
: mpl::not_<detail::has_adapted_variant_tag<T> >
{};

template <BOOST_VARIANT_ENUM_PARAMS(typename T), typename Domain>
struct not_is_variant<boost::variant<BOOST_VARIANT_ENUM_PARAMS(T)>, Domain>
: mpl::false_
{};

template <typename T>
struct variant_type
: mpl::identity<T>
{};

template <typename T>
struct variant_type<boost::optional<T> >
: variant_type<T>
{};

template <typename T, typename Domain>
struct not_is_variant_or_variant_in_optional
: not_is_variant<typename variant_type<T>::type, Domain>
{};

namespace detail
{
template <typename Expected, typename Attribute>
struct attribute_is_compatible
: is_convertible<Attribute, Expected>
{};

template <typename Expected, typename Attribute>
struct attribute_is_compatible<Expected, boost::optional<Attribute> >
: is_convertible<Attribute, Expected>
{};

template <typename Container>
struct is_hold_any_container
: traits::is_hold_any<typename traits::container_value<Container>::type>
{};
}

template <typename Attribute, typename Expected
, typename IsNotVariant = mpl::false_, typename Enable = void>
struct compute_compatible_component_variant
: mpl::or_<
traits::detail::attribute_is_compatible<Expected, Attribute>
, traits::is_hold_any<Expected>
, mpl::eval_if<
is_container<Expected>
, traits::detail::is_hold_any_container<Expected>
, mpl::false_> >
{};

namespace detail
{
BOOST_MPL_HAS_XXX_TRAIT_DEF(types)
}

template <typename Variant, typename Expected>
struct compute_compatible_component_variant<Variant, Expected, mpl::false_
, typename enable_if<detail::has_types<typename variant_type<Variant>::type> >::type>
{
typedef typename traits::variant_type<Variant>::type variant_type;
typedef typename variant_type::types types;
typedef typename mpl::end<types>::type end;

typedef typename
mpl::find_if<types, is_same<Expected, mpl::_1> >::type
iter;

typedef typename mpl::distance<
typename mpl::begin<types>::type, iter
>::type distance;

typedef typename mpl::not_<is_same<iter, end> >::type type;
enum { value = type::value };

typedef typename
mpl::eval_if<type, mpl::deref<iter>, mpl::identity<unused_type> >::type
compatible_type;

static bool is_compatible(int which)
{
return which == distance::value;
}
};

template <typename Expected, typename Attribute, typename Domain>
struct compute_compatible_component
: compute_compatible_component_variant<Attribute, Expected
, typename not_is_variant_or_variant_in_optional<Attribute, Domain>::type> {};

template <typename Expected, typename Domain>
struct compute_compatible_component<Expected, unused_type, Domain>
: mpl::false_ {};

template <typename Attribute, typename Domain>
struct compute_compatible_component<unused_type, Attribute, Domain>
: mpl::false_ {};

template <typename Domain>
struct compute_compatible_component<unused_type, unused_type, Domain>
: mpl::false_ {};

template <BOOST_VARIANT_ENUM_PARAMS(typename T)>
struct variant_which<boost::variant<BOOST_VARIANT_ENUM_PARAMS(T)> >
{
static int call(boost::variant<BOOST_VARIANT_ENUM_PARAMS(T)> const& v)
{
return v.which();
}
};

template <typename T>
int which(T const& v)
{
return variant_which<T>::call(v);
}

template <typename T, typename Domain, typename Enable>
struct not_is_optional
: mpl::true_
{};

template <typename T, typename Domain>
struct not_is_optional<boost::optional<T>, Domain>
: mpl::false_
{};

template <typename Component
, typename Context = unused_type, typename Iterator = unused_type>
struct attribute_of
{
typedef typename Component::template
attribute<Context, Iterator>::type type;
};

template <typename Context, typename Iterator = unused_type>
struct attribute_not_unused
{
template <typename Component>
struct apply
: not_is_unused<typename
attribute_of<Component, Context, Iterator>::type>
{};
};

template <typename Attribute, typename Enable>
struct attribute_type : mpl::identity<Attribute> {};

template <typename T>
struct sequence_size
: fusion::result_of::size<T>
{};

template <>
struct sequence_size<unused_type>
: mpl::int_<0>
{};

namespace detail
{
template <typename Attribute, typename Enable = void>
struct attribute_size_impl
{
typedef std::size_t type;

static type call(Attribute const&)
{
return 1;
}
};

template <typename Attribute>
struct attribute_size_impl<Attribute
, typename enable_if<
mpl::and_<
fusion::traits::is_sequence<Attribute>
, mpl::not_<traits::is_container<Attribute> >
>
>::type>
{
typedef typename fusion::result_of::size<Attribute>::value_type type;

static type call(Attribute const& attr)
{
return fusion::size(attr);
}
};

template <typename Attribute>
struct attribute_size_impl<Attribute
, typename enable_if<
mpl::and_<
traits::is_container<Attribute>
, mpl::not_<traits::is_iterator_range<Attribute> >
>
>::type>
{
typedef typename Attribute::size_type type;

static type call(Attribute const& attr)
{
return attr.size();
}
};
}

template <typename Attribute, typename Enable>
struct attribute_size
: detail::attribute_size_impl<Attribute>
{};

template <typename Attribute>
struct attribute_size<optional<Attribute> >
{
typedef typename attribute_size<Attribute>::type type;

static type call(optional<Attribute> const& val)
{
if (!val)
return 0;
return traits::size(val.get());
}
};

namespace detail
{
struct attribute_size_visitor : static_visitor<std::size_t>
{
template <typename T>
std::size_t operator()(T const& val) const
{
return spirit::traits::size(val);
}
};
}

template <BOOST_VARIANT_ENUM_PARAMS(typename T)>
struct attribute_size<variant<BOOST_VARIANT_ENUM_PARAMS(T)> >
{
typedef std::size_t type;

static type call(variant<BOOST_VARIANT_ENUM_PARAMS(T)> const& val)
{
return apply_visitor(detail::attribute_size_visitor(), val);
}
};

template <typename Iterator>
struct attribute_size<iterator_range<Iterator> >
{
typedef typename std::iterator_traits<Iterator>::
difference_type type;

static type call(iterator_range<Iterator> const& r)
{
return std::distance(r.begin(), r.end());
}
};

template <>
struct attribute_size<unused_type>
{
typedef std::size_t type;

static type call(unused_type)
{
return 0;
}
};

template <typename Attribute>
typename attribute_size<Attribute>::type
size (Attribute const& attr)
{
return attribute_size<Attribute>::call(attr);
}

template <typename Component, typename Attribute, typename Enable>
struct pass_attribute
{
typedef fusion::vector1<Attribute&> type;
};

template <typename Attribute, typename Force = mpl::false_>
struct wrap_if_not_tuple
: mpl::if_<
fusion::traits::is_sequence<Attribute>
, Attribute&, fusion::vector1<Attribute&> >
{};

template <typename Attribute>
struct wrap_if_not_tuple<Attribute, mpl::true_>
{
typedef fusion::vector1<Attribute&> type;
};

template <>
struct wrap_if_not_tuple<unused_type, mpl::false_>
{
typedef unused_type type;
};

template <>
struct wrap_if_not_tuple<unused_type const, mpl::false_>
{
typedef unused_type type;
};

template <typename T>
struct build_optional
{
typedef boost::optional<T> type;
};

template <typename T>
struct build_optional<boost::optional<T> >
{
typedef boost::optional<T> type;
};

template <>
struct build_optional<unused_type>
{
typedef unused_type type;
};

template <typename T>
struct build_std_vector
{
typedef std::vector<T> type;
};

template <>
struct build_std_vector<unused_type>
{
typedef unused_type type;
};


template <typename Sequence>
struct filter_unused_attributes
: fusion::result_of::filter_if<Sequence, not_is_unused<mpl::_> >
{};

template <typename Attribute, typename Domain>
struct sequence_attribute_transform
: mpl::identity<Attribute>
{};

template <typename Attribute, typename Domain>
struct permutation_attribute_transform
: traits::build_optional<Attribute>
{};

template <typename Attribute, typename Domain>
struct sequential_or_attribute_transform
: traits::build_optional<Attribute>
{};

template <typename Sequence>
struct build_fusion_vector
{
typedef typename
filter_unused_attributes<Sequence>::type
filtered_attributes;


typedef typename
mpl::eval_if<
fusion::result_of::empty<filtered_attributes>
, mpl::identity<unused_type>
, fusion::result_of::as_vector<filtered_attributes>
>::type
type;
};

template <>
struct build_fusion_vector<unused_type>
{
typedef unused_type type;
};

template <typename Sequence, typename Context
, template <typename T, typename D> class Transform
, typename Iterator = unused_type, typename Domain = unused_type>
struct build_attribute_sequence
{
struct element_attribute
{
template <typename T>
struct result;

template <typename F, typename Element>
struct result<F(Element)>
{
typedef typename
Transform<
typename attribute_of<Element, Context, Iterator>::type
, Domain
>::type
type;
};

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
template <typename Element>
typename result<element_attribute(Element)>::type
operator()(Element&&) const;
#endif
};

typedef typename
fusion::result_of::transform<Sequence, element_attribute>::type
type;
};

template <typename Sequence>
struct has_no_unused
: is_same<
typename mpl::find_if<Sequence, is_same<mpl::_, unused_type> >::type
, typename mpl::end<Sequence>::type>
{};

namespace detail
{
template <typename Sequence, bool no_unused
, int size = mpl::size<Sequence>::value>
struct build_collapsed_variant;

template <typename Sequence, int size>
struct build_collapsed_variant<Sequence, true, size>
: spirit::detail::as_variant<Sequence> {};

template <typename Sequence, int size>
struct build_collapsed_variant<Sequence, false, size>
{
typedef boost::optional<
typename spirit::detail::as_variant<
typename fusion::result_of::pop_front<Sequence>::type
>::type
> type;
};

template <typename Sequence>
struct build_collapsed_variant<Sequence, true, 1>
: mpl::front<Sequence> {};

template <typename Sequence>
struct build_collapsed_variant<Sequence, false, 1>
: mpl::front<Sequence> {};

template <typename Sequence>
struct build_collapsed_variant<Sequence, true, 2>
: spirit::detail::as_variant<Sequence> {};

template <typename Sequence>
struct build_collapsed_variant<Sequence, false, 2>
{
typedef boost::optional<
typename mpl::deref<
typename mpl::next<
typename mpl::begin<Sequence>::type
>::type
>::type
>
type;
};
}

template <typename Attribute, typename Domain>
struct alternative_attribute_transform
: mpl::identity<Attribute>
{};

template <typename Sequence>
struct build_variant
{
typedef typename
filter_unused_attributes<Sequence>::type
filtered_attributes;

typedef has_no_unused<Sequence> no_unused;

typedef typename
mpl::eval_if<
no_unused,
mpl::identity<Sequence>,
fusion::result_of::push_front<filtered_attributes, unused_type>
>::type
attribute_sequence;

typedef typename
mpl::fold<
attribute_sequence, mpl::vector<>,
mpl::if_<
mpl::contains<mpl::_1, mpl::_2>,
mpl::_1, mpl::push_back<mpl::_1, mpl::_2>
>
>::type
no_duplicates;

typedef typename
traits::detail::build_collapsed_variant<
no_duplicates, no_unused::value>::type
type;
};

namespace detail {
template <typename Exposed, typename Transformed, typename Domain>
struct transform_attribute_base;

template <typename Attribute>
struct synthesize_attribute
{
typedef Attribute type;
static Attribute pre(unused_type) { return Attribute(); }
static void post(unused_type, Attribute const&) {}
static void fail(unused_type) {}
};
}
template <typename Exposed, typename Transformed, typename Domain
, typename Enable>
struct transform_attribute
: detail::transform_attribute_base<Exposed, Transformed, Domain>
{
BOOST_STATIC_ASSERT_MSG(!is_reference<Exposed>::value,
"Exposed cannot be a reference type");
BOOST_STATIC_ASSERT_MSG(!is_reference<Transformed>::value,
"Transformed cannot be a reference type");
};

template <typename Transformed, typename Domain>
struct transform_attribute<unused_type, Transformed, Domain>
: detail::synthesize_attribute<Transformed>
{};

template <typename Transformed, typename Domain>
struct transform_attribute<unused_type const, Transformed, Domain>
: detail::synthesize_attribute<Transformed>
{};

template <typename A, typename B>
void swap_impl(A& a, B& b)
{
A temp = a;
a = b;
b = temp;
}

template <typename T>
void swap_impl(T& a, T& b)
{
boost::swap(a, b);
}

template <typename A>
void swap_impl(A&, unused_type)
{
}

template <typename A>
void swap_impl(unused_type, A&)
{
}

inline void swap_impl(unused_type, unused_type)
{
}

template <typename T>
struct strip_single_element_vector
{
typedef T type;
};

#if !defined(BOOST_FUSION_HAS_VARIADIC_VECTOR)
template <typename T>
struct strip_single_element_vector<fusion::vector1<T> >
{
typedef T type;
};
#endif
template <typename T>
struct strip_single_element_vector<fusion::vector<T> >
{
typedef T type;
};

template <typename T
, bool IsFusionSeq = fusion::traits::is_sequence<T>::value
, bool IsProtoExpr = proto::is_expr<T>::value>
struct one_element_sequence
: mpl::false_
{};

template <typename T>
struct one_element_sequence<T, true, false>
: mpl::bool_<mpl::size<T>::value == 1>
{};

template <typename T>
void clear(T& val);

namespace detail
{
struct clear_visitor : static_visitor<>
{
template <typename T>
void operator()(T& val) const
{
spirit::traits::clear(val);
}
};

template <typename T>
void clear_impl2(T& val, mpl::false_)
{
val = T();
}

template <typename T>
void clear_impl2(T& val, mpl::true_)
{
fusion::for_each(val, clear_visitor());
}

template <typename T>
void clear_impl(T& val, mpl::false_)
{
clear_impl2(val, fusion::traits::is_sequence<T>());
}

template <typename T>
void clear_impl(T& val, mpl::true_)
{
val.clear();
}
}

template <typename T, typename Enable>
struct clear_value
{
static void call(T& val)
{
detail::clear_impl(val, typename is_container<T>::type());
}
};

template <typename T>
struct clear_value<boost::optional<T> >
{
static void call(boost::optional<T>& val)
{
if (val)
val = none;   
}
};

template <BOOST_VARIANT_ENUM_PARAMS(typename T)>
struct clear_value<variant<BOOST_VARIANT_ENUM_PARAMS(T)> >
{
static void call(variant<BOOST_VARIANT_ENUM_PARAMS(T)>& val)
{
apply_visitor(detail::clear_visitor(), val);
}
};

template <typename T>
struct clear_value<iterator_range<T> >
{
static void call(iterator_range<T>& val)
{
val = iterator_range<T>(val.end(), val.end());
}
};

template <typename T>
void clear(T& val)
{
clear_value<T>::call(val);
}

inline void clear(unused_type)
{
}

namespace detail
{
template <typename Out>
struct print_fusion_sequence
{
print_fusion_sequence(Out& out_)
: out(out_), is_first(true) {}

typedef void result_type;

template <typename T>
void operator()(T const& val) const
{
if (is_first)
is_first = false;
else
out << ", ";
spirit::traits::print_attribute(out, val);
}

Out& out;
mutable bool is_first;
};

template <typename Out>
struct print_visitor : static_visitor<>
{
print_visitor(Out& out_) : out(out_) {}

template <typename T>
void operator()(T const& val) const
{
spirit::traits::print_attribute(out, val);
}

Out& out;
};
}

template <typename Out, typename T, typename Enable>
struct print_attribute_debug
{
template <typename T_>
static void call_impl3(Out& out, T_ const& val, mpl::false_)
{
out << val;
}

template <typename T_>
static void call_impl3(Out& out, T_ const& val, mpl::true_)
{
out << '[';
fusion::for_each(val, detail::print_fusion_sequence<Out>(out));
out << ']';
}

template <typename T_>
static void call_impl2(Out& out, T_ const& val, mpl::false_)
{
call_impl3(out, val, fusion::traits::is_sequence<T_>());
}

template <typename T_>
static void call_impl2(Out& out, T_ const& val, mpl::true_)
{
out << '[';
if (!traits::is_empty(val))
{
bool first = true;
typename container_iterator<T_ const>::type iend = traits::end(val);
for (typename container_iterator<T_ const>::type i = traits::begin(val);
!traits::compare(i, iend); traits::next(i))
{
if (!first)
out << ", ";
first = false;
spirit::traits::print_attribute(out, traits::deref(i));
}
}
out << ']';
}

template <typename T_>
static void call_impl(Out& out, T_ const& val, mpl::false_)
{
apply_visitor(detail::print_visitor<Out>(out), val);
}

template <typename T_>
static void call_impl(Out& out, T_ const& val, mpl::true_)
{
call_impl2(out, val, is_container<T_>());
}

static void call(Out& out, T const& val)
{
call_impl(out, val, not_is_variant<T>());
}
};

template <typename Out, typename T>
struct print_attribute_debug<Out, boost::optional<T> >
{
static void call(Out& out, boost::optional<T> const& val)
{
if (val)
spirit::traits::print_attribute(out, *val);
else
out << "[empty]";
}
};

template <typename Out, typename T>
inline void print_attribute(Out& out, T const& val)
{
print_attribute_debug<Out, T>::call(out, val);
}

template <typename Out>
inline void print_attribute(Out&, unused_type)
{
}

namespace detail
{
struct token_printer_debug_for_chars
{
template<typename Out, typename Char>
static void print(Out& o, Char c)
{
using namespace std;    

if (c == static_cast<Char>('\a'))
o << "\\a";
else if (c == static_cast<Char>('\b'))
o << "\\b";
else if (c == static_cast<Char>('\f'))
o << "\\f";
else if (c == static_cast<Char>('\n'))
o << "\\n";
else if (c == static_cast<Char>('\r'))
o << "\\r";
else if (c == static_cast<Char>('\t'))
o << "\\t";
else if (c == static_cast<Char>('\v'))
o << "\\v";
else if (c >= 0 && c < 127 && iscntrl(c))
o << "\\" << std::oct << static_cast<int>(c);
else
o << static_cast<char>(c);
}
};

struct token_printer_debug
{
template<typename Out, typename T>
static void print(Out& o, T const& val)
{
o << val;
}
};
}

template <typename T, typename Enable>
struct token_printer_debug
: mpl::if_<
mpl::and_<
is_convertible<T, char>, is_convertible<char, T> >
, detail::token_printer_debug_for_chars
, detail::token_printer_debug>::type
{};

template <typename Out, typename T>
inline void print_token(Out& out, T const& val)
{
token_printer_debug<T>::print(out, val);
}
}}}

#endif
