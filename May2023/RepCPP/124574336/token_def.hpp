
#if !defined(BOOST_SPIRIT_LEX_TOKEN_DEF_MAR_13_2007_0145PM)
#define BOOST_SPIRIT_LEX_TOKEN_DEF_MAR_13_2007_0145PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/support/argument.hpp>
#include <boost/spirit/home/support/info.hpp>
#include <boost/spirit/home/support/handles_container.hpp>
#include <boost/spirit/home/qi/parser.hpp>
#include <boost/spirit/home/qi/skip_over.hpp>
#include <boost/spirit/home/qi/detail/construct.hpp>
#include <boost/spirit/home/qi/detail/assign_to.hpp>
#include <boost/spirit/home/lex/reference.hpp>
#include <boost/spirit/home/lex/lexer_type.hpp>
#include <boost/spirit/home/lex/lexer/terminals.hpp>

#include <boost/fusion/include/vector.hpp>
#include <boost/mpl/if.hpp>
#include <boost/proto/extends.hpp>
#include <boost/proto/traits.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/variant.hpp>

#include <iterator> 
#include <string>
#include <cstdlib>

#if defined(BOOST_MSVC)
# pragma warning(push)
# pragma warning(disable: 4355) 
#endif

namespace boost { namespace spirit { namespace lex
{
template<typename Attribute = unused_type
, typename Char = char
, typename Idtype = std::size_t>
struct token_def
: proto::extends<
typename proto::terminal<
lex::reference<token_def<Attribute, Char, Idtype> const, Idtype> 
>::type
, token_def<Attribute, Char, Idtype> >
, qi::parser<token_def<Attribute, Char, Idtype> >
, lex::lexer_type<token_def<Attribute, Char, Idtype> >
{
private:
typedef lex::reference<token_def const, Idtype> reference_;
typedef typename proto::terminal<reference_>::type terminal_type;
typedef proto::extends<terminal_type, token_def> proto_base_type;

static std::size_t const all_states_id = static_cast<std::size_t>(-2);

public:
template <typename Context, typename Iterator>
struct attribute
{
typedef typename Iterator::base_iterator_type iterator_type;
typedef typename mpl::if_<
traits::not_is_unused<Attribute>
, typename mpl::if_<
is_same<Attribute, lex::omit>, unused_type, Attribute
>::type
, iterator_range<iterator_type>
>::type type;
};

public:
template <typename Iterator, typename Context
, typename Skipper, typename Attribute_>
bool parse(Iterator& first, Iterator const& last
, Context& , Skipper const& skipper
, Attribute_& attr) const
{
qi::skip_over(first, last, skipper);   

if (first != last) {
typedef typename 
std::iterator_traits<Iterator>::value_type 
token_type;

BOOST_ASSERT(std::size_t(~0) != token_state_);

token_type const& t = *first;
if (token_id_ == t.id() && 
(all_states_id == token_state_ || token_state_ == t.state())) 
{
spirit::traits::assign_to(t, attr);
++first;
return true;
}
}
return false;
}

template <typename Context>
info what(Context& ) const
{
if (0 == def_.which()) 
return info("token_def", boost::get<string_type>(def_));

return info("token_def", boost::get<char_type>(def_));
}

template <typename LexerDef, typename String>
void collect(LexerDef& lexdef, String const& state
, String const& targetstate) const
{
std::size_t state_id = lexdef.add_state(state.c_str());

BOOST_ASSERT(
(std::size_t(~0) == token_state_ || state_id == token_state_) &&
"Can't use single token_def with more than one lexer state");

char_type const* target = targetstate.empty() ? 0 : targetstate.c_str();
if (target)
lexdef.add_state(target);

token_state_ = state_id;
if (0 == token_id_)
token_id_ = lexdef.get_next_id();

if (0 == def_.which()) {
unique_id_ = lexdef.add_token(state.c_str()
, boost::get<string_type>(def_), token_id_, target);
}
else {
unique_id_ = lexdef.add_token(state.c_str()
, boost::get<char_type>(def_), token_id_, target);
}
}

template <typename LexerDef>
void add_actions(LexerDef&) const {}

public:
typedef Char char_type;
typedef Idtype id_type;
typedef std::basic_string<char_type> string_type;

token_def() 
: proto_base_type(terminal_type::make(reference_(*this)))
, def_('\0'), token_id_()
, unique_id_(std::size_t(~0)), token_state_(std::size_t(~0)) {}

token_def(token_def const& rhs) 
: proto_base_type(terminal_type::make(reference_(*this)))
, def_(rhs.def_), token_id_(rhs.token_id_)
, unique_id_(rhs.unique_id_), token_state_(rhs.token_state_) {}

explicit token_def(char_type def_, Idtype id_ = Idtype())
: proto_base_type(terminal_type::make(reference_(*this)))
, def_(def_)
, token_id_(Idtype() == id_ ? Idtype(def_) : id_)
, unique_id_(std::size_t(~0)), token_state_(std::size_t(~0)) {}

explicit token_def(string_type const& def_, Idtype id_ = Idtype())
: proto_base_type(terminal_type::make(reference_(*this)))
, def_(def_), token_id_(id_)
, unique_id_(std::size_t(~0)), token_state_(std::size_t(~0)) {}

template <typename String>
token_def& operator= (String const& definition)
{
def_ = definition;
token_id_ = Idtype();
unique_id_ = std::size_t(~0);
token_state_ = std::size_t(~0);
return *this;
}
token_def& operator= (token_def const& rhs)
{
def_ = rhs.def_;
token_id_ = rhs.token_id_;
unique_id_ = rhs.unique_id_;
token_state_ = rhs.token_state_;
return *this;
}

Idtype const& id() const { return token_id_; }
void id(Idtype const& id) { token_id_ = id; }
std::size_t unique_id() const { return unique_id_; }

string_type definition() const 
{ 
return (0 == def_.which()) ? 
boost::get<string_type>(def_) : 
string_type(1, boost::get<char_type>(def_));
}
std::size_t state() const { return token_state_; }

private:
variant<string_type, char_type> def_;
mutable Idtype token_id_;
mutable std::size_t unique_id_;
mutable std::size_t token_state_;
};
}}}

namespace boost { namespace spirit { namespace traits
{
template<typename Attribute, typename Char, typename Idtype
, typename Attr, typename Context, typename Iterator>
struct handles_container<
lex::token_def<Attribute, Char, Idtype>, Attr, Context, Iterator>
: traits::is_container<
typename attribute_of<
lex::token_def<Attribute, Char, Idtype>, Context, Iterator
>::type>
{};
}}}

#if defined(BOOST_MSVC)
# pragma warning(pop)
#endif

#endif
