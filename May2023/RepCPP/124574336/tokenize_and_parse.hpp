
#if !defined(BOOST_SPIRIT_LEXER_PARSE_NOV_17_2007_0246PM)
#define BOOST_SPIRIT_LEXER_PARSE_NOV_17_2007_0246PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/skip_over.hpp>
#include <boost/spirit/home/qi/parse.hpp>
#include <boost/spirit/home/qi/nonterminal/grammar.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/lex/lexer.hpp>
#include <boost/mpl/assert.hpp>

namespace boost { namespace phoenix
{
template <typename Expr>
struct actor;
}}

namespace boost { namespace spirit { namespace lex
{
using qi::skip_flag;

template <typename Iterator, typename Lexer, typename ParserExpr>
inline bool
tokenize_and_parse(Iterator& first, Iterator last, Lexer const& lex,
ParserExpr const& xpr)
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, ParserExpr);

typename Lexer::iterator_type iter = lex.begin(first, last);
return compile<qi::domain>(xpr).parse(
iter, lex.end(), unused, unused, unused);
}

template <typename Iterator, typename Lexer, typename ParserExpr
, typename Attribute>
inline bool
tokenize_and_parse(Iterator& first, Iterator last, Lexer const& lex
, ParserExpr const& xpr, Attribute& attr)
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, ParserExpr);

typename Lexer::iterator_type iter = lex.begin(first, last);
return compile<qi::domain>(xpr).parse(
iter, lex.end(), unused, unused, attr);
}

template <typename Iterator, typename Lexer, typename ParserExpr
, typename Skipper>
inline bool
tokenize_and_phrase_parse(Iterator& first, Iterator last
, Lexer const& lex, ParserExpr const& xpr, Skipper const& skipper
, BOOST_SCOPED_ENUM(skip_flag) post_skip = skip_flag::postskip)
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, ParserExpr);
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, Skipper);

typedef
typename spirit::result_of::compile<qi::domain, Skipper>::type
skipper_type;
skipper_type const skipper_ = compile<qi::domain>(skipper);

typename Lexer::iterator_type iter = lex.begin(first, last);
typename Lexer::iterator_type end = lex.end();
if (!compile<qi::domain>(xpr).parse(
iter, end, unused, skipper_, unused))
return false;

if (post_skip == skip_flag::postskip)
qi::skip_over(iter, end, skipper_);
return true;
}

template <typename Iterator, typename Lexer, typename ParserExpr
, typename Skipper, typename Attribute>
inline bool
tokenize_and_phrase_parse(Iterator& first, Iterator last
, Lexer const& lex, ParserExpr const& xpr, Skipper const& skipper
, BOOST_SCOPED_ENUM(skip_flag) post_skip, Attribute& attr)
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, ParserExpr);
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, Skipper);

typedef
typename spirit::result_of::compile<qi::domain, Skipper>::type
skipper_type;
skipper_type const skipper_ = compile<qi::domain>(skipper);

typename Lexer::iterator_type iter = lex.begin(first, last);
typename Lexer::iterator_type end = lex.end();
if (!compile<qi::domain>(xpr).parse(
iter, end, unused, skipper_, attr))
return false;

if (post_skip == skip_flag::postskip)
qi::skip_over(iter, end, skipper_);
return true;
}

template <typename Iterator, typename Lexer, typename ParserExpr
, typename Skipper, typename Attribute>
inline bool
tokenize_and_phrase_parse(Iterator& first, Iterator last
, Lexer const& lex, ParserExpr const& xpr, Skipper const& skipper
, Attribute& attr)
{
return tokenize_and_phrase_parse(first, last, lex, xpr, skipper
, skip_flag::postskip, attr);
}

namespace detail
{
template <typename Token, typename F>
bool tokenize_callback(Token const& t, F f)
{
return f(t);
}

template <typename Token, typename Eval>
bool tokenize_callback(Token const& t, phoenix::actor<Eval> const& f)
{
f(t);
return true;
}

template <typename Token>
bool tokenize_callback(Token const& t, void (*f)(Token const&))
{
f(t);
return true;
}

template <typename Token>
bool tokenize_callback(Token const& t, bool (*f)(Token const&))
{
return f(t);
}
}

template <typename Iterator, typename Lexer, typename F>
inline bool
tokenize(Iterator& first, Iterator last, Lexer const& lex, F f
, typename Lexer::char_type const* initial_state = 0)
{
typedef typename Lexer::iterator_type iterator_type;

iterator_type iter = lex.begin(first, last, initial_state);
iterator_type end = lex.end();
for (; iter != end && token_is_valid(*iter); ++iter) 
{
if (!detail::tokenize_callback(*iter, f))
return false;
}
return (iter == end) ? true : false;
}

template <typename Iterator, typename Lexer>
inline bool
tokenize(Iterator& first, Iterator last, Lexer const& lex
, typename Lexer::char_type const* initial_state = 0)
{
typedef typename Lexer::iterator_type iterator_type;

iterator_type iter = lex.begin(first, last, initial_state);
iterator_type end = lex.end();

while (iter != end && token_is_valid(*iter))
++iter;

return (iter == end) ? true : false;
}

}}}

#endif
