
#if !defined(BOOST_SPIRIT_LEX_LEXER_MAR_17_2007_0139PM)
#define BOOST_SPIRIT_LEX_LEXER_MAR_17_2007_0139PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <iosfwd>

#include <boost/spirit/home/support/detail/lexer/generator.hpp>
#include <boost/spirit/home/support/detail/lexer/rules.hpp>
#include <boost/spirit/home/support/detail/lexer/consts.hpp>
#include <boost/spirit/home/support/unused.hpp>

#include <boost/spirit/home/lex/lexer/lexertl/token.hpp>
#include <boost/spirit/home/lex/lexer/lexertl/functor.hpp>
#include <boost/spirit/home/lex/lexer/lexertl/functor_data.hpp>
#include <boost/spirit/home/lex/lexer/lexertl/iterator.hpp>
#if defined(BOOST_SPIRIT_LEXERTL_DEBUG)
#include <boost/spirit/home/support/detail/lexer/debug.hpp>
#endif

#include <boost/foreach.hpp>

#include <iterator> 

namespace boost { namespace spirit { namespace lex { namespace lexertl
{
namespace detail
{
template <typename Char>
inline bool must_escape(Char c)
{
switch (c) {
case '+': case '/': case '*': case '?':
case '|':
case '(': case ')':
case '[': case ']':
case '{': case '}':
case '.':
case '^': case '$':
case '\\':
case '"':
return true;

default:
break;
}
return false;
}

template <typename Char>
inline std::basic_string<Char> escape(Char ch)
{
std::basic_string<Char> result(1, ch);
if (detail::must_escape(ch))
{
typedef typename std::basic_string<Char>::size_type size_type;
result.insert((size_type)0, 1, '\\');
}
return result;
}

inline boost::lexer::regex_flags map_flags(unsigned int flags)
{
unsigned int retval = boost::lexer::none;
if (flags & match_flags::match_not_dot_newline)
retval |= boost::lexer::dot_not_newline;
if (flags & match_flags::match_icase)
retval |= boost::lexer::icase;

return boost::lexer::regex_flags(retval);
}
}

template <typename Lexer, typename F>
bool generate_static(Lexer const&
, std::basic_ostream<typename Lexer::char_type>&
, typename Lexer::char_type const*, F);


template <typename Token = token<>
, typename Iterator = typename Token::iterator_type
, typename Functor = functor<Token, lexertl::detail::data, Iterator> >
class lexer
{
private:
struct dummy { void true_() {} };
typedef void (dummy::*safe_bool)();

static std::size_t const all_states_id = static_cast<std::size_t>(-2);

public:
operator safe_bool() const
{ return initialized_dfa_ ? &dummy::true_ : 0; }

typedef typename std::iterator_traits<Iterator>::value_type char_type;
typedef std::basic_string<char_type> string_type;

typedef boost::lexer::basic_rules<char_type> basic_rules_type;

typedef Token token_type;
typedef typename Token::id_type id_type;
typedef iterator<Functor> iterator_type;

private:
struct iterator_data_type
{
typedef typename Functor::semantic_actions_type semantic_actions_type;

iterator_data_type(
boost::lexer::basic_state_machine<char_type> const& sm
, boost::lexer::basic_rules<char_type> const& rules
, semantic_actions_type const& actions)
: state_machine_(sm), rules_(rules), actions_(actions)
{}

boost::lexer::basic_state_machine<char_type> const& state_machine_;
boost::lexer::basic_rules<char_type> const& rules_;
semantic_actions_type const& actions_;

BOOST_DELETED_FUNCTION(iterator_data_type& operator= (iterator_data_type const&))
};

public:
iterator_type begin(Iterator& first, Iterator const& last
, char_type const* initial_state = 0) const
{
if (!init_dfa())    
return iterator_type();

iterator_data_type iterator_data(state_machine_, rules_, actions_);
return iterator_type(iterator_data, first, last, initial_state);
}

iterator_type end() const
{
return iterator_type();
}

protected:
lexer(unsigned int flags)
: flags_(detail::map_flags(flags))
, rules_(flags_)
, initialized_dfa_(false)
{}

public:
std::size_t add_token(char_type const* state, char_type tokendef,
std::size_t token_id, char_type const* targetstate)
{
add_state(state);
initialized_dfa_ = false;
if (state == all_states())
return rules_.add(state, detail::escape(tokendef), token_id, rules_.dot());

if (0 == targetstate)
targetstate = state;
else
add_state(targetstate);
return rules_.add(state, detail::escape(tokendef), token_id, targetstate);
}
std::size_t add_token(char_type const* state, string_type const& tokendef,
std::size_t token_id, char_type const* targetstate)
{
add_state(state);
initialized_dfa_ = false;
if (state == all_states())
return rules_.add(state, tokendef, token_id, rules_.dot());

if (0 == targetstate)
targetstate = state;
else
add_state(targetstate);
return rules_.add(state, tokendef, token_id, targetstate);
}

void add_pattern (char_type const* state, string_type const& name,
string_type const& patterndef)
{
add_state(state);
rules_.add_macro(name.c_str(), patterndef);
initialized_dfa_ = false;
}

boost::lexer::rules const& get_rules() const { return rules_; }

void clear(char_type const* state)
{
std::size_t s = rules_.state(state);
if (boost::lexer::npos != s)
rules_.clear(state);
initialized_dfa_ = false;
}
std::size_t add_state(char_type const* state)
{
if (state == all_states())
return all_states_id;

std::size_t stateid = rules_.state(state);
if (boost::lexer::npos == stateid) {
stateid = rules_.add_state(state);
initialized_dfa_ = false;
}
return stateid;
}
string_type initial_state() const
{
return string_type(rules_.initial());
}
string_type all_states() const
{
return string_type(rules_.all_states());
}

template <typename F>
void add_action(std::size_t unique_id, std::size_t state, F act)
{
typedef typename Functor::wrap_action_type wrapper_type;
if (state == all_states_id) {
typedef typename
basic_rules_type::string_size_t_map::value_type
state_type;

std::size_t states = rules_.statemap().size();
BOOST_FOREACH(state_type const& s, rules_.statemap()) {
for (std::size_t j = 0; j < states; ++j)
actions_.add_action(unique_id + j, s.second, wrapper_type::call(act));
}
}
else {
actions_.add_action(unique_id, state, wrapper_type::call(act));
}
}

bool init_dfa(bool minimize = false) const
{
if (!initialized_dfa_) {
state_machine_.clear();
typedef boost::lexer::basic_generator<char_type> generator;
generator::build (rules_, state_machine_);
if (minimize)
generator::minimise (state_machine_);

#if defined(BOOST_SPIRIT_LEXERTL_DEBUG)
boost::lexer::debug::dump(state_machine_, std::cerr);
#endif
initialized_dfa_ = true;

}
return true;
}

private:
mutable boost::lexer::basic_state_machine<char_type> state_machine_;
boost::lexer::regex_flags flags_;
basic_rules_type rules_;

typename Functor::semantic_actions_type actions_;
mutable bool initialized_dfa_;

template <typename Lexer, typename F>
friend bool generate_static(Lexer const&
, std::basic_ostream<typename Lexer::char_type>&
, typename Lexer::char_type const*, F);
};

template <typename Token = token<>
, typename Iterator = typename Token::iterator_type
, typename Functor = functor<Token, lexertl::detail::data, Iterator, mpl::true_> >
class actor_lexer : public lexer<Token, Iterator, Functor>
{
protected:
actor_lexer(unsigned int flags)
: lexer<Token, Iterator, Functor>(flags) {}
};

}}}}

#endif
