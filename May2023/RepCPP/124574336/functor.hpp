
#if !defined(BOOST_SPIRIT_LEX_LEXER_FUNCTOR_NOV_18_2007_1112PM)
#define BOOST_SPIRIT_LEX_LEXER_FUNCTOR_NOV_18_2007_1112PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/mpl/bool.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/spirit/home/lex/lexer/pass_flags.hpp>
#include <boost/assert.hpp>
#include <iterator> 

#if 0 != __COMO_VERSION__ || !BOOST_WORKAROUND(BOOST_MSVC, <= 1310)
#define BOOST_SPIRIT_STATIC_EOF 1
#define BOOST_SPIRIT_EOF_PREFIX static
#else
#define BOOST_SPIRIT_EOF_PREFIX 
#endif

namespace boost { namespace spirit { namespace lex { namespace lexertl
{ 
template <typename Token
, template <typename, typename, typename, typename> class FunctorData
, typename Iterator = typename Token::iterator_type
, typename SupportsActors = mpl::false_
, typename SupportsState = typename Token::has_state>
class functor
{
public:
typedef typename 
std::iterator_traits<Iterator>::value_type 
char_type;

private:
typedef typename Token::token_value_type token_value_type;
friend class FunctorData<Iterator, SupportsActors, SupportsState
, token_value_type>;

template <typename T>
struct assign_on_exit
{
assign_on_exit(T& dst, T const& src)
: dst_(dst), src_(src) {}

~assign_on_exit()
{
dst_ = src_;
}

T& dst_;
T const& src_;

BOOST_DELETED_FUNCTION(assign_on_exit& operator= (assign_on_exit const&))
};

public:
functor() {}

#if BOOST_WORKAROUND(BOOST_MSVC, <= 1310)
functor& operator=(functor const& rhs)
{
return *this;
}
#endif

typedef Token result_type;
typedef functor unique;
typedef FunctorData<Iterator, SupportsActors, SupportsState
, token_value_type> shared;

BOOST_SPIRIT_EOF_PREFIX result_type const eof;

typedef Iterator iterator_type;
typedef typename shared::semantic_actions_type semantic_actions_type;
typedef typename shared::next_token_functor next_token_functor;
typedef typename shared::get_state_name_type get_state_name_type;

typedef typename shared::wrap_action_type wrap_action_type;

template <typename MultiPass>
static result_type& get_next(MultiPass& mp, result_type& result)
{
typedef typename result_type::id_type id_type;

shared& data = mp.shared()->ftor;
for(;;) 
{
if (data.get_first() == data.get_last()) 
#if defined(BOOST_SPIRIT_STATIC_EOF)
return result = eof;
#else
return result = mp.ftor.eof;
#endif

data.reset_value();
Iterator end = data.get_first();
std::size_t unique_id = boost::lexer::npos;
bool prev_bol = false;

std::size_t state = data.get_state();
std::size_t id = data.next(end, unique_id, prev_bol);

if (boost::lexer::npos == id) {   
#if defined(BOOST_SPIRIT_LEXERTL_DEBUG)
std::string next;
Iterator it = data.get_first();
for (std::size_t i = 0; i < 10 && it != data.get_last(); ++it, ++i)
next += *it;

std::cerr << "Not matched, in state: " << state 
<< ", lookahead: >" << next << "<" << std::endl;
#endif
return result = result_type(0);
}
else if (0 == id) {         
#if defined(BOOST_SPIRIT_STATIC_EOF)
return result = eof;
#else
return result = mp.ftor.eof;
#endif
}

#if defined(BOOST_SPIRIT_LEXERTL_DEBUG)
{
std::string next;
Iterator it = end;
for (std::size_t i = 0; i < 10 && it != data.get_last(); ++it, ++i)
next += *it;

std::cerr << "Matched: " << id << ", in state: " 
<< state << ", string: >" 
<< std::basic_string<char_type>(data.get_first(), end) << "<"
<< ", lookahead: >" << next << "<" << std::endl;
if (data.get_state() != state) {
std::cerr << "Switched to state: " 
<< data.get_state() << std::endl;
}
}
#endif
bool adjusted = data.adjust_start();

data.set_end(end);

BOOST_SCOPED_ENUM(pass_flags) pass = 
data.invoke_actions(state, id, unique_id, end);

if (data.has_value()) {
assign_on_exit<Iterator> on_exit(data.get_first(), end);
return result = result_type(id_type(id), state, data.get_value());
}
else if (pass_flags::pass_normal == pass) {
assign_on_exit<Iterator> on_exit(data.get_first(), end);
return result = result_type(id_type(id), state, data.get_first(), end);
}
else if (pass_flags::pass_fail == pass) {
#if defined(BOOST_SPIRIT_LEXERTL_DEBUG)
std::cerr << "Matching forced to fail" << std::endl; 
#endif
if (adjusted)
data.revert_adjust_start();

data.reset_bol(prev_bol);
if (state != data.get_state())
continue;       

return result = result_type(0);
}

#if defined(BOOST_SPIRIT_LEXERTL_DEBUG)
std::cerr << "Token ignored, continuing matching" << std::endl; 
#endif
data.get_first() = end;
}
}

template <typename MultiPass>
static std::size_t set_state(MultiPass& mp, std::size_t state) 
{ 
std::size_t oldstate = mp.shared()->ftor.get_state();
mp.shared()->ftor.set_state(state);

#if defined(BOOST_SPIRIT_LEXERTL_DEBUG)
std::cerr << "Switching state from: " << oldstate 
<< " to: " << state
<< std::endl;
#endif
return oldstate; 
}

template <typename MultiPass>
static std::size_t get_state(MultiPass& mp) 
{ 
return mp.shared()->ftor.get_state();
}

template <typename MultiPass>
static std::size_t 
map_state(MultiPass const& mp, char_type const* statename)  
{ 
return mp.shared()->ftor.get_state_id(statename);
}

template <typename MultiPass>
static void destroy(MultiPass const&) {}
};

#if defined(BOOST_SPIRIT_STATIC_EOF)
template <typename Token
, template <typename, typename, typename, typename> class FunctorData
, typename Iterator, typename SupportsActors, typename SupportsState>
typename functor<Token, FunctorData, Iterator, SupportsActors, SupportsState>::result_type const
functor<Token, FunctorData, Iterator, SupportsActors, SupportsState>::eof = 
typename functor<Token, FunctorData, Iterator, SupportsActors
, SupportsState>::result_type();
#endif

}}}}

#undef BOOST_SPIRIT_EOF_PREFIX
#undef BOOST_SPIRIT_STATIC_EOF

#endif
