
#if !defined(BOOST_SPIRIT_LEX_TOKEN_FEB_10_2008_0751PM)
#define BOOST_SPIRIT_LEX_TOKEN_FEB_10_2008_0751PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/spirit/home/qi/detail/assign_to.hpp>
#include <boost/spirit/home/support/attributes.hpp>
#include <boost/spirit/home/support/argument.hpp>
#include <boost/spirit/home/support/detail/lexer/generator.hpp>
#include <boost/spirit/home/support/detail/lexer/rules.hpp>
#include <boost/spirit/home/support/detail/lexer/consts.hpp>
#include <boost/spirit/home/support/utree/utree_traits_fwd.hpp>
#include <boost/spirit/home/lex/lexer/terminals.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/variant.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/begin.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/or.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <boost/type_traits/integral_promotion.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/static_assert.hpp>

#if defined(BOOST_SPIRIT_DEBUG)
#include <iosfwd>
#endif

namespace boost { namespace spirit { namespace lex { namespace lexertl
{ 
template <typename Iterator = char const*
, typename AttributeTypes = mpl::vector0<>
, typename HasState = mpl::true_
, typename Idtype = std::size_t> 
struct token;

template <typename Iterator, typename Idtype>
struct token<Iterator, lex::omit, mpl::false_, Idtype>
{
typedef Iterator iterator_type;
typedef mpl::false_ has_state;
typedef Idtype id_type;
typedef unused_type token_value_type;

token() : id_(id_type(boost::lexer::npos)) {}

explicit token(int) : id_(id_type(0)) {}

token(id_type id, std::size_t) : id_(id) {}

token(id_type id, std::size_t, token_value_type)
: id_(id) {}

token_value_type& value() { static token_value_type u; return u; }
token_value_type const& value() const { return unused; }

#if defined(BOOST_SPIRIT_DEBUG)
token(id_type id, std::size_t, Iterator const& first
, Iterator const& last)
: matched_(first, last)
, id_(id) {}
#else
token(id_type id, std::size_t, Iterator const&, Iterator const&)
: id_(id) {}
#endif

operator id_type() const { return static_cast<id_type>(id_); }

id_type id() const { return static_cast<id_type>(id_); }
void id(id_type newid) { id_ = newid; }

std::size_t state() const { return 0; }   

bool is_valid() const 
{ 
return 0 != id_ && id_type(boost::lexer::npos) != id_; 
}

#if defined(BOOST_SPIRIT_DEBUG)
#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
token& operator= (token const& rhs)
{
if (this != &rhs) 
{
id_ = rhs.id_;
if (is_valid()) 
matched_ = rhs.matched_;
}
return *this;
}
#endif
std::pair<Iterator, Iterator> matched_;
#endif

protected:
typename boost::integral_promotion<id_type>::type id_;            
};

#if defined(BOOST_SPIRIT_DEBUG)
template <typename Char, typename Traits, typename Iterator
, typename AttributeTypes, typename HasState, typename Idtype> 
inline std::basic_ostream<Char, Traits>& 
operator<< (std::basic_ostream<Char, Traits>& os
, token<Iterator, AttributeTypes, HasState, Idtype> const& t)
{
if (t.is_valid()) {
Iterator end = t.matched_.second;
for (Iterator it = t.matched_.first; it != end; ++it)
os << *it;
}
else {
os << "<invalid token>";
}
return os;
}
#endif

template <typename Iterator, typename Idtype>
struct token<Iterator, lex::omit, mpl::true_, Idtype>
: token<Iterator, lex::omit, mpl::false_, Idtype>
{
private:
typedef token<Iterator, lex::omit, mpl::false_, Idtype> base_type;

public:
typedef typename base_type::id_type id_type;
typedef Iterator iterator_type;
typedef mpl::true_ has_state;
typedef unused_type token_value_type;

token() : state_(boost::lexer::npos) {}

explicit token(int) : base_type(0), state_(boost::lexer::npos) {}

token(id_type id, std::size_t state)
: base_type(id, boost::lexer::npos), state_(state) {}

token(id_type id, std::size_t state, token_value_type)
: base_type(id, boost::lexer::npos, unused)
, state_(state) {}

token(id_type id, std::size_t state
, Iterator const& first, Iterator const& last)
: base_type(id, boost::lexer::npos, first, last)
, state_(state) {}

std::size_t state() const { return state_; }

#if defined(BOOST_SPIRIT_DEBUG) && BOOST_WORKAROUND(BOOST_MSVC, == 1600)
token& operator= (token const& rhs)
{
if (this != &rhs) 
{
this->base_type::operator=(static_cast<base_type const&>(rhs));
state_ = rhs.state_;
}
return *this;
}
#endif

protected:
std::size_t state_;      
};

namespace detail
{
template <typename IteratorPair, typename AttributeTypes>
struct token_value_typesequence
{
typedef typename mpl::insert<
AttributeTypes
, typename mpl::begin<AttributeTypes>::type
, IteratorPair
>::type sequence_type;
typedef typename make_variant_over<sequence_type>::type type;
};

template <typename IteratorPair, typename AttributeTypes>
struct token_value_type
: mpl::eval_if<
mpl::or_<
is_same<AttributeTypes, mpl::vector0<> >
, is_same<AttributeTypes, mpl::vector<> > >
, mpl::identity<IteratorPair>
, token_value_typesequence<IteratorPair, AttributeTypes> >
{};
}

template <typename Iterator, typename AttributeTypes, typename HasState
, typename Idtype>
struct token : token<Iterator, lex::omit, HasState, Idtype>
{
private: 
BOOST_STATIC_ASSERT((mpl::is_sequence<AttributeTypes>::value || 
is_same<AttributeTypes, lex::omit>::value));
typedef token<Iterator, lex::omit, HasState, Idtype> base_type;

protected: 
typedef iterator_range<Iterator> iterpair_type;

public:
typedef typename base_type::id_type id_type;
typedef typename detail::token_value_type<
iterpair_type, AttributeTypes
>::type token_value_type;

typedef Iterator iterator_type;

token() : value_(iterpair_type(iterator_type(), iterator_type())) {}

explicit token(int)
: base_type(0)
, value_(iterpair_type(iterator_type(), iterator_type())) {}

token(id_type id, std::size_t state, token_value_type const& value)
: base_type(id, state, value)
, value_(value) {}

token(id_type id, std::size_t state, Iterator const& first
, Iterator const& last)
: base_type(id, state, first, last)
, value_(iterpair_type(first, last)) {}

token_value_type& value() { return value_; }
token_value_type const& value() const { return value_; }

#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
token& operator= (token const& rhs)
{
if (this != &rhs) 
{
this->base_type::operator=(static_cast<base_type const&>(rhs));
if (this->is_valid()) 
value_ = rhs.value_;
}
return *this;
}
#endif

protected:
token_value_type value_; 
};

template <typename Iterator, typename AttributeTypes, typename HasState
, typename Idtype>
inline bool 
operator== (token<Iterator, AttributeTypes, HasState, Idtype> const& lhs, 
token<Iterator, AttributeTypes, HasState, Idtype> const& rhs)
{
return lhs.id() == rhs.id();
}

template <typename Iterator, typename AttributeTypes, typename HasState
, typename Idtype>
inline bool 
token_is_valid(token<Iterator, AttributeTypes, HasState, Idtype> const& t)
{
return t.is_valid();
}
}}}}

namespace boost { namespace spirit { namespace traits
{

template <typename Attribute, typename Iterator, typename AttributeTypes
, typename HasState, typename Idtype>
struct assign_to_attribute_from_value<Attribute
, lex::lexertl::token<Iterator, AttributeTypes, HasState, Idtype> >
{
static void 
call(lex::lexertl::token<Iterator, AttributeTypes, HasState, Idtype> const& t
, Attribute& attr)
{

if (0 == t.value().which()) {
typedef iterator_range<Iterator> iterpair_type;
iterpair_type const& ip = boost::get<iterpair_type>(t.value());

spirit::traits::assign_to(ip.begin(), ip.end(), attr);


typedef lex::lexertl::token<
Iterator, AttributeTypes, HasState, Idtype> token_type;
spirit::traits::assign_to(
attr, const_cast<token_type&>(t).value());   
}
else {
spirit::traits::assign_to(boost::get<Attribute>(t.value()), attr);
}
}
};

template <typename Attribute, typename Iterator, typename AttributeTypes
, typename HasState, typename Idtype>
struct assign_to_container_from_value<Attribute
, lex::lexertl::token<Iterator, AttributeTypes, HasState, Idtype> >
: assign_to_attribute_from_value<Attribute
, lex::lexertl::token<Iterator, AttributeTypes, HasState, Idtype> >
{};

template <typename Iterator, typename AttributeTypes
, typename HasState, typename Idtype>
struct assign_to_container_from_value<utree
, lex::lexertl::token<Iterator, AttributeTypes, HasState, Idtype> >
: assign_to_attribute_from_value<utree
, lex::lexertl::token<Iterator, AttributeTypes, HasState, Idtype> >
{};

template <typename Iterator>
struct assign_to_container_from_value<
iterator_range<Iterator>, iterator_range<Iterator> >
{
static void 
call(iterator_range<Iterator> const& val, iterator_range<Iterator>& attr)
{
attr = val;
}
};

template <typename Attribute, typename Iterator, typename HasState
, typename Idtype>
struct assign_to_attribute_from_value<Attribute
, lex::lexertl::token<Iterator, mpl::vector0<>, HasState, Idtype> >
{
static void 
call(lex::lexertl::token<Iterator, mpl::vector0<>, HasState, Idtype> const& t
, Attribute& attr)
{
spirit::traits::assign_to(t.value().begin(), t.value().end(), attr);
}
};


template <typename Attribute, typename Iterator, typename HasState
, typename Idtype>
struct assign_to_attribute_from_value<Attribute
, lex::lexertl::token<Iterator, mpl::vector<>, HasState, Idtype> >
{
static void 
call(lex::lexertl::token<Iterator, mpl::vector<>, HasState, Idtype> const& t
, Attribute& attr)
{
spirit::traits::assign_to(t.value().begin(), t.value().end(), attr);
}
};


template <typename Attribute, typename Iterator, typename HasState
, typename Idtype>
struct assign_to_attribute_from_value<Attribute
, lex::lexertl::token<Iterator, lex::omit, HasState, Idtype> >
{
static void 
call(lex::lexertl::token<Iterator, lex::omit, HasState, Idtype> const&
, Attribute&)
{
}
};

template <typename Attribute, typename Iterator, typename HasState
, typename Idtype>
struct assign_to_container_from_value<Attribute
, lex::lexertl::token<Iterator, lex::omit, HasState, Idtype> >
: assign_to_attribute_from_value<Attribute
, lex::lexertl::token<Iterator, lex::omit, HasState, Idtype> >
{};

template <typename Iterator, typename AttributeTypes, typename HasState
, typename Idtype_, typename Idtype>
struct assign_to_attribute_from_value<
fusion::vector2<Idtype_, iterator_range<Iterator> >
, lex::lexertl::token<Iterator, AttributeTypes, HasState, Idtype> >
{
static void 
call(lex::lexertl::token<Iterator, AttributeTypes, HasState, Idtype> const& t
, fusion::vector2<Idtype_, iterator_range<Iterator> >& attr)
{
typedef iterator_range<Iterator> iterpair_type;
typedef fusion::vector2<Idtype_, iterator_range<Iterator> > 
attribute_type;

iterpair_type const& ip = boost::get<iterpair_type>(t.value());
attr = attribute_type(t.id(), ip);
}
};

template <typename Iterator, typename AttributeTypes, typename HasState
, typename Idtype_, typename Idtype>
struct assign_to_container_from_value<
fusion::vector2<Idtype_, iterator_range<Iterator> >
, lex::lexertl::token<Iterator, AttributeTypes, HasState, Idtype> >
: assign_to_attribute_from_value<
fusion::vector2<Idtype_, iterator_range<Iterator> >
, lex::lexertl::token<Iterator, AttributeTypes, HasState, Idtype> >
{};

template <typename Iterator, typename Attribute, typename HasState
, typename Idtype>
struct token_printer_debug<
lex::lexertl::token<Iterator, Attribute, HasState, Idtype> >
{
typedef lex::lexertl::token<Iterator, Attribute, HasState, Idtype> token_type;

template <typename Out>
static void print(Out& out, token_type const& val) 
{
out << '[';
spirit::traits::print_token(out, val.value());
out << ']';
}
};
}}}

#endif
