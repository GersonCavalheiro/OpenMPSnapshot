
#ifndef BOOST_SPIRIT_LEX_LEXER_SUPPORT_FUNCTIONS_HPP
#define BOOST_SPIRIT_LEX_LEXER_SUPPORT_FUNCTIONS_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/detail/scoped_enum_emulation.hpp>
#include <boost/spirit/home/lex/lexer/pass_flags.hpp>
#include <boost/spirit/home/lex/lexer/support_functions_expression.hpp>
#include <boost/phoenix/core/actor.hpp>
#include <boost/phoenix/core/as_actor.hpp>
#include <boost/phoenix/core/value.hpp> 

namespace boost { namespace spirit { namespace lex
{
template <typename Actor>
struct less_type
{
typedef mpl::true_ no_nullary;

template <typename Env>
struct result
{
typedef typename remove_reference< 
typename remove_const<
typename mpl::at_c<typename Env::args_type, 4>::type
>::type
>::type context_type;
typedef typename context_type::base_iterator_type type;
};

template <typename Env>
typename result<Env>::type 
eval(Env const& env) const
{
typename result<Env>::type it;
return fusion::at_c<4>(env.args()).less(it, actor_());
}

less_type(Actor const& actor)
: actor_(actor) {}

Actor actor_;
};

template <typename T>
inline typename expression::less<
typename phoenix::as_actor<T>::type
>::type const
less(T const& v)
{
return expression::less<T>::make(phoenix::as_actor<T>::convert(v));
}

struct more_type
{
typedef mpl::true_ no_nullary;

template <typename Env>
struct result
{
typedef void type;
};

template <typename Env>
void eval(Env const& env) const
{
fusion::at_c<4>(env.args()).more();
}
};

inline phoenix::actor<more_type> more()
{
return phoenix::actor<more_type>();
}

template <typename IdActor, typename StateActor>
struct lookahead_type
{
typedef mpl::true_ no_nullary;

template <typename Env>
struct result
{
typedef bool type;
};

template <typename Env>
bool eval(Env const& env) const
{
return fusion::at_c<4>(env.args()).
lookahead(id_actor_(), state_actor_());
}

lookahead_type(IdActor const& id_actor, StateActor const& state_actor)
: id_actor_(id_actor), state_actor_(state_actor) {}

IdActor id_actor_;
StateActor state_actor_;
};

template <typename T>
inline typename expression::lookahead<
typename phoenix::as_actor<T>::type
, typename phoenix::as_actor<std::size_t>::type
>::type const
lookahead(T const& id)
{
typedef typename phoenix::as_actor<T>::type id_actor_type;
typedef typename phoenix::as_actor<std::size_t>::type state_actor_type;

return expression::lookahead<id_actor_type, state_actor_type>::make(
phoenix::as_actor<T>::convert(id),
phoenix::as_actor<std::size_t>::convert(std::size_t(~0)));
}

template <typename Attribute, typename Char, typename Idtype>
inline typename expression::lookahead<
typename phoenix::as_actor<Idtype>::type
, typename phoenix::as_actor<std::size_t>::type
>::type const
lookahead(token_def<Attribute, Char, Idtype> const& tok)
{
typedef typename phoenix::as_actor<Idtype>::type id_actor_type;
typedef typename phoenix::as_actor<std::size_t>::type state_actor_type;

std::size_t state = tok.state();

BOOST_ASSERT(std::size_t(~0) != state && 
"token_def instance not associated with lexer yet");

return expression::lookahead<id_actor_type, state_actor_type>::make(
phoenix::as_actor<Idtype>::convert(tok.id()),
phoenix::as_actor<std::size_t>::convert(state));
}

inline BOOST_SCOPED_ENUM(pass_flags) ignore()
{
return pass_flags::pass_ignore;
}

}}}

#endif
