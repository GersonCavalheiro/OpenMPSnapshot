
#if !defined(BOOST_SPIRIT_LEX_POSITION_TOKEN_MAY_13_2011_0846PM)
#define BOOST_SPIRIT_LEX_POSITION_TOKEN_MAY_13_2011_0846PM

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
#include <boost/mpl/vector.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/is_sequence.hpp>
#include <boost/mpl/begin.hpp>
#include <boost/mpl/insert.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/or.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/range/iterator_range_core.hpp>
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
struct position_token;

template <typename Iterator, typename Idtype>
struct position_token<Iterator, lex::omit, mpl::false_, Idtype>
{
typedef Iterator iterator_type;
typedef iterator_range<iterator_type> iterpair_type;
typedef mpl::false_ has_state;
typedef Idtype id_type;
typedef unused_type token_value_type;

position_token() 
: id_(id_type(boost::lexer::npos)) {}

explicit position_token(int) 
: id_(id_type(0)) {}

position_token(id_type id, std::size_t) 
: id_(id) {}

position_token(id_type id, std::size_t, token_value_type)
: id_(id) {}

position_token(id_type id, std::size_t, Iterator const& first
, Iterator const& last)
: id_(id), matched_(first, last) {}

operator id_type() const { return id_; }

id_type id() const { return id_; }
void id(id_type newid) { id_ = newid; }

std::size_t state() const { return 0; }   

bool is_valid() const 
{ 
return 0 != id_ && id_type(boost::lexer::npos) != id_; 
}

iterator_type begin() const { return matched_.begin(); }
iterator_type end() const { return matched_.end(); }

iterpair_type& matched() { return matched_; }
iterpair_type const& matched() const { return matched_; }

token_value_type& value() { static token_value_type u; return u; }
token_value_type const& value() const { return unused; }

#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
position_token& operator= (position_token const& rhs)
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

protected:
id_type id_;              
iterpair_type matched_;   
};

#if defined(BOOST_SPIRIT_DEBUG)
template <typename Char, typename Traits, typename Iterator
, typename AttributeTypes, typename HasState, typename Idtype> 
inline std::basic_ostream<Char, Traits>& 
operator<< (std::basic_ostream<Char, Traits>& os
, position_token<Iterator, AttributeTypes, HasState, Idtype> const& t)
{
if (t.is_valid()) {
Iterator end = t.end();
for (Iterator it = t.begin(); it != end; ++it)
os << *it;
}
else {
os << "<invalid token>";
}
return os;
}
#endif

template <typename Iterator, typename Idtype>
struct position_token<Iterator, lex::omit, mpl::true_, Idtype>
: position_token<Iterator, lex::omit, mpl::false_, Idtype>
{
private:
typedef position_token<Iterator, lex::omit, mpl::false_, Idtype> 
base_type;

public:
typedef typename base_type::id_type id_type;
typedef Iterator iterator_type;
typedef mpl::true_ has_state;
typedef unused_type token_value_type;

position_token() : state_(boost::lexer::npos) {}

explicit position_token(int) 
: base_type(0), state_(boost::lexer::npos) {}

position_token(id_type id, std::size_t state)
: base_type(id, boost::lexer::npos), state_(state) {}

position_token(id_type id, std::size_t state, token_value_type)
: base_type(id, boost::lexer::npos, unused)
, state_(state) {}

position_token(id_type id, std::size_t state
, Iterator const& first, Iterator const& last)
: base_type(id, boost::lexer::npos, first, last)
, state_(state) {}

std::size_t state() const { return state_; }

#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
position_token& operator= (position_token const& rhs)
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

template <typename Iterator, typename HasState, typename Idtype>
struct position_token<Iterator, mpl::vector<>, HasState, Idtype>
: position_token<Iterator, lex::omit, HasState, Idtype>
{
private:
typedef position_token<Iterator, lex::omit, HasState, Idtype> base_type;

public:
typedef typename base_type::id_type id_type;
typedef typename base_type::iterator_type iterator_type;
typedef typename base_type::iterpair_type iterpair_type;
typedef HasState has_state;
typedef iterpair_type token_value_type;

position_token() {}

explicit position_token(int) 
: base_type(0) {}

position_token(id_type id, std::size_t state)
: base_type(id, state) {}

position_token(id_type id, std::size_t state, token_value_type)
: base_type(id, state, unused) {}

position_token(id_type id, std::size_t state
, Iterator const& first, Iterator const& last)
: base_type(id, state, first, last) {}

token_value_type& value() { return this->base_type::matched(); }
token_value_type const& value() const { return this->base_type::matched(); }
};

template <typename Iterator, typename HasState, typename Idtype>
struct position_token<Iterator, mpl::vector0<>, HasState, Idtype>
: position_token<Iterator, lex::omit, HasState, Idtype>
{
private:
typedef position_token<Iterator, lex::omit, HasState, Idtype> base_type;

public:
typedef typename base_type::id_type id_type;
typedef typename base_type::iterator_type iterator_type;
typedef typename base_type::iterpair_type iterpair_type;
typedef HasState has_state;
typedef iterpair_type token_value_type;

position_token() {}

explicit position_token(int) 
: base_type(0) {}

position_token(id_type id, std::size_t state)
: base_type(id, state) {}

position_token(id_type id, std::size_t state, token_value_type)
: base_type(id, state, unused) {}

position_token(id_type id, std::size_t state
, Iterator const& first, Iterator const& last)
: base_type(id, state, first, last) {}

token_value_type& value() { return this->base_type::matched(); }
token_value_type const& value() const { return this->base_type::matched(); }
};

template <typename Iterator, typename Attribute, typename HasState
, typename Idtype>
struct position_token<Iterator, mpl::vector<Attribute>, HasState, Idtype>
: position_token<Iterator, lex::omit, HasState, Idtype>
{
private:
typedef position_token<Iterator, lex::omit, HasState, Idtype> base_type;

public:
typedef typename base_type::id_type id_type;
typedef typename base_type::iterator_type iterator_type;
typedef typename base_type::iterpair_type iterpair_type;
typedef HasState has_state;
typedef boost::optional<Attribute> token_value_type;

position_token() {}

explicit position_token(int) 
: base_type(0) {}

position_token(id_type id, std::size_t state)
: base_type(id, state) {}

position_token(id_type id, std::size_t state, token_value_type const& v)
: base_type(id, state, unused), value_(v) {}

position_token(id_type id, std::size_t state
, Iterator const& first, Iterator const& last)
: base_type(id, state, first, last) {}

token_value_type& value() { return value_; }
token_value_type const& value() const { return value_; }

bool has_value() const { return !!value_; }

#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
position_token& operator= (position_token const& rhs)
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

template <typename Iterator, typename Attribute, typename HasState
, typename Idtype>
struct position_token<Iterator, mpl::vector1<Attribute>, HasState, Idtype>
: position_token<Iterator, lex::omit, HasState, Idtype>
{
private:
typedef position_token<Iterator, lex::omit, HasState, Idtype> base_type;

public:
typedef typename base_type::id_type id_type;
typedef typename base_type::iterator_type iterator_type;
typedef typename base_type::iterpair_type iterpair_type;
typedef HasState has_state;
typedef boost::optional<Attribute> token_value_type;

position_token() {}

explicit position_token(int) 
: base_type(0) {}

position_token(id_type id, std::size_t state)
: base_type(id, state) {}

position_token(id_type id, std::size_t state, token_value_type const& v)
: base_type(id, state, unused), value_(v) {}

position_token(id_type id, std::size_t state
, Iterator const& first, Iterator const& last)
: base_type(id, state, first, last) {}

token_value_type& value() { return value_; }
token_value_type const& value() const { return value_; }

bool has_value() const { return value_; }

#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
position_token& operator= (position_token const& rhs)
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

namespace detail
{
template <typename IteratorPair, typename AttributeTypes>
struct position_token_value_typesequence
{
typedef typename mpl::insert<
AttributeTypes
, typename mpl::begin<AttributeTypes>::type
, IteratorPair
>::type sequence_type;
typedef typename make_variant_over<sequence_type>::type type;
};

template <typename IteratorPair, typename AttributeTypes>
struct position_token_value
: mpl::eval_if<
mpl::or_<
is_same<AttributeTypes, mpl::vector0<> >
, is_same<AttributeTypes, mpl::vector<> > >
, mpl::identity<IteratorPair>
, position_token_value_typesequence<IteratorPair, AttributeTypes> >
{};
}

template <typename Iterator, typename AttributeTypes, typename HasState
, typename Idtype>
struct position_token 
: position_token<Iterator, lex::omit, HasState, Idtype>
{
private: 
BOOST_STATIC_ASSERT((mpl::is_sequence<AttributeTypes>::value || 
is_same<AttributeTypes, lex::omit>::value));
typedef position_token<Iterator, lex::omit, HasState, Idtype> 
base_type;

protected: 
typedef iterator_range<Iterator> iterpair_type;

public:
typedef typename base_type::id_type id_type;
typedef typename detail::position_token_value<
iterpair_type, AttributeTypes>::type token_value_type;

typedef Iterator iterator_type;

position_token() {}

explicit position_token(int)
: base_type(0) {}

position_token(id_type id, std::size_t state, token_value_type const& value)
: base_type(id, state, value), value_(value) {}

position_token(id_type id, std::size_t state, Iterator const& first
, Iterator const& last)
: base_type(id, state, first, last)
, value_(iterpair_type(first, last)) 
{}

token_value_type& value() { return value_; }
token_value_type const& value() const { return value_; }

#if BOOST_WORKAROUND(BOOST_MSVC, == 1600)
position_token& operator= (position_token const& rhs)
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
operator== (position_token<Iterator, AttributeTypes, HasState, Idtype> const& lhs, 
position_token<Iterator, AttributeTypes, HasState, Idtype> const& rhs)
{
return lhs.id() == rhs.id();
}

template <typename Iterator, typename AttributeTypes, typename HasState
, typename Idtype>
inline bool 
token_is_valid(position_token<Iterator, AttributeTypes, HasState, Idtype> const& t)
{
return t.is_valid();
}
}}}}

namespace boost { namespace spirit { namespace traits
{

template <typename Attribute, typename Iterator, typename AttributeTypes
, typename HasState, typename Idtype>
struct assign_to_attribute_from_value<Attribute
, lex::lexertl::position_token<Iterator, AttributeTypes, HasState, Idtype> >
{
static void 
call(lex::lexertl::position_token<
Iterator, AttributeTypes, HasState, Idtype> const& t
, Attribute& attr)
{

if (0 == t.value().which()) {
typedef iterator_range<Iterator> iterpair_type;
iterpair_type const& ip = t.matched();

spirit::traits::assign_to(ip.begin(), ip.end(), attr);


typedef lex::lexertl::position_token<
Iterator, AttributeTypes, HasState, Idtype> token_type;
spirit::traits::assign_to(
attr, const_cast<token_type&>(t).value());   
}
else {
spirit::traits::assign_to(get<Attribute>(t.value()), attr);
}
}
};

template <typename Attribute, typename Iterator, typename AttributeTypes
, typename HasState, typename Idtype>
struct assign_to_container_from_value<Attribute
, lex::lexertl::position_token<Iterator, AttributeTypes, HasState, Idtype> >
: assign_to_attribute_from_value<Attribute
, lex::lexertl::position_token<Iterator, AttributeTypes, HasState, Idtype> >
{};

template <typename Attribute, typename Iterator, typename HasState
, typename Idtype>
struct assign_to_attribute_from_value<Attribute
, lex::lexertl::position_token<Iterator, mpl::vector0<>, HasState, Idtype> >
{
static void 
call(lex::lexertl::position_token<
Iterator, mpl::vector0<>, HasState, Idtype> const& t
, Attribute& attr)
{
spirit::traits::assign_to(t.begin(), t.end(), attr);
}
};


template <typename Attribute, typename Iterator, typename HasState
, typename Idtype>
struct assign_to_attribute_from_value<Attribute
, lex::lexertl::position_token<Iterator, mpl::vector<>, HasState, Idtype> >
{
static void 
call(lex::lexertl::position_token<
Iterator, mpl::vector<>, HasState, Idtype> const& t
, Attribute& attr)
{
spirit::traits::assign_to(t.begin(), t.end(), attr);
}
};


template <typename Attribute, typename Iterator, typename Attr
, typename HasState, typename Idtype>
struct assign_to_attribute_from_value<Attribute
, lex::lexertl::position_token<Iterator, mpl::vector1<Attr>, HasState, Idtype> >
{
static void 
call(lex::lexertl::position_token<
Iterator, mpl::vector1<Attr>, HasState, Idtype> const& t
, Attribute& attr)
{

if (!t.has_value()) {
typedef iterator_range<Iterator> iterpair_type;
iterpair_type const& ip = t.matched();

spirit::traits::assign_to(ip.begin(), ip.end(), attr);

typedef lex::lexertl::position_token<
Iterator, mpl::vector1<Attr>, HasState, Idtype> token_type;
spirit::traits::assign_to(
attr, const_cast<token_type&>(t).value());
}
else {
spirit::traits::assign_to(t.value(), attr);
}
}
};


template <typename Attribute, typename Iterator, typename Attr
, typename HasState, typename Idtype>
struct assign_to_attribute_from_value<Attribute
, lex::lexertl::position_token<Iterator, mpl::vector<Attr>, HasState, Idtype> >
{
static void 
call(lex::lexertl::position_token<
Iterator, mpl::vector<Attr>, HasState, Idtype> const& t
, Attribute& attr)
{

if (!t.has_value()) {
typedef iterator_range<Iterator> iterpair_type;
iterpair_type const& ip = t.matched();

spirit::traits::assign_to(ip.begin(), ip.end(), attr);

typedef lex::lexertl::position_token<
Iterator, mpl::vector<Attr>, HasState, Idtype> token_type;
spirit::traits::assign_to(
attr, const_cast<token_type&>(t).value());
}
else {
spirit::traits::assign_to(t.value(), attr);
}
}
};


template <typename Attribute, typename Iterator, typename HasState
, typename Idtype>
struct assign_to_attribute_from_value<Attribute
, lex::lexertl::position_token<Iterator, lex::omit, HasState, Idtype> >
{
static void 
call(lex::lexertl::position_token<Iterator, lex::omit, HasState, Idtype> const&
, Attribute&)
{
}
};

template <typename Attribute, typename Iterator, typename HasState
, typename Idtype>
struct assign_to_container_from_value<Attribute
, lex::lexertl::position_token<Iterator, lex::omit, HasState, Idtype> >
: assign_to_attribute_from_value<Attribute
, lex::lexertl::position_token<Iterator, lex::omit, HasState, Idtype> >
{};

template <typename Iterator, typename AttributeTypes, typename HasState
, typename Idtype_, typename Idtype>
struct assign_to_attribute_from_value<
fusion::vector2<Idtype_, iterator_range<Iterator> >
, lex::lexertl::position_token<Iterator, AttributeTypes, HasState, Idtype> >
{
static void 
call(lex::lexertl::position_token<Iterator, AttributeTypes, HasState, Idtype> const& t
, fusion::vector2<Idtype_, iterator_range<Iterator> >& attr)
{
typedef iterator_range<Iterator> iterpair_type;
typedef fusion::vector2<Idtype_, iterator_range<Iterator> > 
attribute_type;

iterpair_type const& ip = t.matched();
attr = attribute_type(t.id(), ip);
}
};

template <typename Iterator, typename AttributeTypes, typename HasState
, typename Idtype_, typename Idtype>
struct assign_to_container_from_value<
fusion::vector2<Idtype_, iterator_range<Iterator> >
, lex::lexertl::position_token<Iterator, AttributeTypes, HasState, Idtype> >
: assign_to_attribute_from_value<
fusion::vector2<Idtype_, iterator_range<Iterator> >
, lex::lexertl::position_token<Iterator, AttributeTypes, HasState, Idtype> >
{};

template <typename Iterator, typename Attribute, typename HasState
, typename Idtype>
struct token_printer_debug<
lex::lexertl::position_token<Iterator, Attribute, HasState, Idtype> >
{
typedef lex::lexertl::position_token<Iterator, Attribute, HasState, Idtype> token_type;

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
