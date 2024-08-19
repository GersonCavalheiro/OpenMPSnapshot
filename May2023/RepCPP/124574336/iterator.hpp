
#if !defined(BOOST_SPIRIT_LEX_LEXER_ITERATOR_MAR_16_2007_0353PM)
#define BOOST_SPIRIT_LEX_LEXER_ITERATOR_MAR_16_2007_0353PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/multi_pass_wrapper.hpp>
#if defined(BOOST_SPIRIT_DEBUG)
#include <boost/spirit/home/support/iterators/detail/buf_id_check_policy.hpp>
#else
#include <boost/spirit/home/support/iterators/detail/no_check_policy.hpp>
#endif
#include <boost/spirit/home/support/iterators/detail/split_functor_input_policy.hpp>
#include <boost/spirit/home/support/iterators/detail/ref_counted_policy.hpp>
#include <boost/spirit/home/support/iterators/detail/split_std_deque_policy.hpp>
#include <boost/spirit/home/support/iterators/multi_pass.hpp>

namespace boost { namespace spirit { namespace lex { namespace lexertl
{ 
template <typename FunctorData>
struct make_multi_pass
{
typedef std::pair<typename FunctorData::unique
, typename FunctorData::shared> functor_data_type;

typedef typename FunctorData::result_type result_type;

typedef iterator_policies::split_functor_input input_policy;
typedef iterator_policies::ref_counted ownership_policy;
#if defined(BOOST_SPIRIT_DEBUG)
typedef iterator_policies::buf_id_check check_policy;
#else
typedef iterator_policies::no_check check_policy;
#endif
typedef iterator_policies::split_std_deque storage_policy;

typedef iterator_policies::default_policy<
ownership_policy, check_policy, input_policy, storage_policy>
policy_type;

typedef spirit::multi_pass<functor_data_type, policy_type> type;
};

template <typename Functor>
class iterator : public make_multi_pass<Functor>::type
{
public:
typedef typename Functor::unique unique_functor_type;
typedef typename Functor::shared shared_functor_type;

typedef typename Functor::iterator_type base_iterator_type;
typedef typename Functor::result_type token_type;

private:
typedef typename make_multi_pass<Functor>::functor_data_type 
functor_type;
typedef typename make_multi_pass<Functor>::type base_type;
typedef typename Functor::char_type char_type;

public:
template <typename IteratorData>
iterator(IteratorData const& iterdata_, base_iterator_type& first
, base_iterator_type const& last, char_type const* state = 0)
: base_type(functor_type(unique_functor_type()
, shared_functor_type(iterdata_, first, last))) 
{
set_state(map_state(state));
}

iterator() {}

iterator(const base_type& base)
: base_type(base) { }

std::size_t set_state(std::size_t state)
{
return unique_functor_type::set_state(*this, state);
}

std::size_t get_state()
{
return unique_functor_type::get_state(*this);
}

std::size_t map_state(char_type const* statename)
{
return (0 != statename) 
? unique_functor_type::map_state(*this, statename)
: 0;
}
};
}}

namespace traits 
{ 
template <typename Functor>
struct is_multi_pass<spirit::lex::lexertl::iterator<Functor> >
: mpl::true_ {};

template <typename Functor>
void clear_queue(spirit::lex::lexertl::iterator<Functor> & mp
, BOOST_SCOPED_ENUM(traits::clear_mode) mode)
{
mp.clear_queue(mode);
}

template <typename Functor>
void inhibit_clear_queue(spirit::lex::lexertl::iterator<Functor>& mp, bool flag)
{
mp.inhibit_clear_queue(flag);
}

template <typename Functor> 
bool inhibit_clear_queue(spirit::lex::lexertl::iterator<Functor>& mp)
{
return mp.inhibit_clear_queue();
}
}

}}

#endif
