
#if !defined(BOOST_SPIRIT_LEX_LEXER_SEMANTIC_ACTION_DATA_JUN_10_2009_0417PM)
#define BOOST_SPIRIT_LEX_LEXER_SEMANTIC_ACTION_DATA_JUN_10_2009_0417PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/lex/lexer/pass_flags.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/function.hpp>
#include <vector>

namespace boost { namespace spirit { namespace lex { namespace lexertl
{ 
namespace detail
{
template <typename Iterator, typename SupportsState, typename Data>
struct semantic_actions;

template <typename Iterator, typename Data>
struct semantic_actions<Iterator, mpl::false_, Data>
{
typedef void functor_type(Iterator&, Iterator&
, BOOST_SCOPED_ENUM(pass_flags)&, std::size_t&, Data&);
typedef boost::function<functor_type> functor_wrapper_type;

template <typename F>
void add_action(std::size_t unique_id, std::size_t, F act) 
{
if (actions_.size() <= unique_id)
actions_.resize(unique_id + 1); 

actions_[unique_id] = act;
}

BOOST_SCOPED_ENUM(pass_flags) invoke_actions(std::size_t 
, std::size_t& id, std::size_t unique_id, Iterator& end
, Data& data) const
{
if (unique_id >= actions_.size() || !actions_[unique_id]) 
return pass_flags::pass_normal;

BOOST_SCOPED_ENUM(pass_flags) match = pass_flags::pass_normal;
actions_[unique_id](data.get_first(), end, match, id, data);
return match;
}

std::vector<functor_wrapper_type> actions_;
}; 

template <typename Iterator, typename Data>
struct semantic_actions<Iterator, mpl::true_, Data>
{
typedef void functor_type(Iterator&, Iterator&
, BOOST_SCOPED_ENUM(pass_flags)&, std::size_t&, Data&);
typedef boost::function<functor_type> functor_wrapper_type;

template <typename F>
void add_action(std::size_t unique_id, std::size_t state, F act) 
{
if (actions_.size() <= state)
actions_.resize(state + 1); 

std::vector<functor_wrapper_type>& actions (actions_[state]);
if (actions.size() <= unique_id)
actions.resize(unique_id + 1); 

actions[unique_id] = act;
}

BOOST_SCOPED_ENUM(pass_flags) invoke_actions(std::size_t state
, std::size_t& id, std::size_t unique_id, Iterator& end
, Data& data) const
{
if (state >= actions_.size())
return pass_flags::pass_normal;

std::vector<functor_wrapper_type> const& actions = actions_[state];
if (unique_id >= actions.size() || !actions[unique_id]) 
return pass_flags::pass_normal;

data.set_end(end);

BOOST_SCOPED_ENUM(pass_flags) match = pass_flags::pass_normal;
actions[unique_id](data.get_first(), end, match, id, data);
return match;
}

std::vector<std::vector<functor_wrapper_type> > actions_;
}; 
}

}}}}

#endif
