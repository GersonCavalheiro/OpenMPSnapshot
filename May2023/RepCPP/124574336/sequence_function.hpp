
#if !defined(BOOST_SPIRIT_LEX_SEQUENCE_FUNCTION_FEB_28_2007_0249PM)
#define BOOST_SPIRIT_LEX_SEQUENCE_FUNCTION_FEB_28_2007_0249PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/lex/domain.hpp>
#include <boost/spirit/home/support/unused.hpp>

namespace boost { namespace spirit { namespace lex { namespace detail
{
template <typename LexerDef, typename String>
struct sequence_collect_function
{
sequence_collect_function(LexerDef& def_, String const& state_
, String const& targetstate_)
: def(def_), state(state_), targetstate(targetstate_) {}

template <typename Component>
bool operator()(Component const& component) const
{
component.collect(def, state, targetstate);
return false;     
}

LexerDef& def;
String const& state;
String const& targetstate;

BOOST_DELETED_FUNCTION(sequence_collect_function& operator= (sequence_collect_function const&))
};

template <typename LexerDef>
struct sequence_add_actions_function
{
sequence_add_actions_function(LexerDef& def_)
: def(def_) {}

template <typename Component>
bool operator()(Component const& component) const
{
component.add_actions(def);
return false;     
}

LexerDef& def;

BOOST_DELETED_FUNCTION(sequence_add_actions_function& operator= (sequence_add_actions_function const&))
};

}}}}

#endif
