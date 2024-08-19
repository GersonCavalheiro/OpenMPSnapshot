
#if !defined(BOOST_SPIRIT_LEX_STATIC_LEXER_FEB_10_2008_0753PM)
#define BOOST_SPIRIT_LEX_STATIC_LEXER_FEB_10_2008_0753PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/lex/lexer/lexertl/token.hpp>
#include <boost/spirit/home/lex/lexer/lexertl/functor.hpp>
#include <boost/spirit/home/lex/lexer/lexertl/static_functor_data.hpp>
#include <boost/spirit/home/lex/lexer/lexertl/iterator.hpp>
#include <boost/spirit/home/lex/lexer/lexertl/static_version.hpp>
#if defined(BOOST_SPIRIT_DEBUG)
#include <boost/spirit/home/support/detail/lexer/debug.hpp>
#endif
#include <iterator> 

namespace boost { namespace spirit { namespace lex { namespace lexertl
{ 
namespace static_
{
struct lexer;
}


template <typename Token = token<>
, typename LexerTables = static_::lexer
, typename Iterator = typename Token::iterator_type
, typename Functor = functor<Token, detail::static_data, Iterator> >
class static_lexer 
{
private:
struct dummy { void true_() {} };
typedef void (dummy::*safe_bool)();

public:
operator safe_bool() const { return &dummy::true_; }

typedef typename std::iterator_traits<Iterator>::value_type char_type;
typedef std::basic_string<char_type> string_type;

typedef Token token_type;
typedef typename Token::id_type id_type;
typedef iterator<Functor> iterator_type;

private:
struct iterator_data_type 
{
typedef typename Functor::next_token_functor next_token_functor;
typedef typename Functor::semantic_actions_type semantic_actions_type;
typedef typename Functor::get_state_name_type get_state_name_type;

iterator_data_type(next_token_functor next
, semantic_actions_type const& actions
, get_state_name_type get_state_name, std::size_t num_states
, bool bol)
: next_(next), actions_(actions), get_state_name_(get_state_name)
, num_states_(num_states), bol_(bol)
{}

next_token_functor next_;
semantic_actions_type const& actions_;
get_state_name_type get_state_name_;
std::size_t num_states_;
bool bol_;

BOOST_DELETED_FUNCTION(iterator_data_type& operator= (iterator_data_type const&))
};

typedef LexerTables tables_type;

BOOST_SPIRIT_ASSERT_MSG(
tables_type::static_version == SPIRIT_STATIC_LEXER_VERSION
, incompatible_static_lexer_version, (LexerTables));

public:
template <typename Iterator_>
iterator_type begin(Iterator_& first, Iterator_ const& last
, char_type const* initial_state = 0) const
{ 
iterator_data_type iterator_data( 
&tables_type::template next<Iterator_>, actions_
, &tables_type::state_name, tables_type::state_count()
, tables_type::supports_bol
);
return iterator_type(iterator_data, first, last, initial_state);
}

iterator_type end() const
{ 
return iterator_type(); 
}

protected:
static_lexer(unsigned int) : unique_id_(0) {}

public:
std::size_t add_token (char_type const*, char_type, std::size_t
, char_type const*) 
{
return unique_id_++;
}
std::size_t add_token (char_type const*, string_type const&
, std::size_t, char_type const*) 
{
return unique_id_++;
}

void add_pattern (char_type const*, string_type const&
, string_type const&) {}

void clear(char_type const*) {}

std::size_t add_state(char_type const* state)
{
return detail::get_state_id(state, &tables_type::state_name
, tables_type::state_count());
}
string_type initial_state() const 
{ 
return tables_type::state_name(0);
}

template <typename F>
void add_action(id_type unique_id, std::size_t state, F act) 
{
typedef typename Functor::wrap_action_type wrapper_type;
actions_.add_action(unique_id, state, wrapper_type::call(act));
}

bool init_dfa(bool  = false) const { return true; }

private:
typename Functor::semantic_actions_type actions_;
std::size_t unique_id_;
};

template <typename Token = token<>
, typename LexerTables = static_::lexer
, typename Iterator = typename Token::iterator_type
, typename Functor 
= functor<Token, detail::static_data, Iterator, mpl::true_> >
class static_actor_lexer 
: public static_lexer<Token, LexerTables, Iterator, Functor>
{
protected:
static_actor_lexer(unsigned int flags) 
: static_lexer<Token, LexerTables, Iterator, Functor>(flags) 
{}
};

}}}}

#endif
