

#if !defined(BOOST_WAVE_LEX_INTERFACE_GENERATOR_HPP_INCLUDED)
#define BOOST_WAVE_LEX_INTERFACE_GENERATOR_HPP_INCLUDED

#include <boost/wave/wave_config.hpp>
#include <boost/wave/util/file_position.hpp>
#include <boost/wave/language_support.hpp>
#include <boost/wave/cpplexer/cpp_lex_interface.hpp>
#include <boost/wave/cpplexer/cpp_lex_token.hpp>      

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_PREFIX
#endif

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable : 4251 4231 4660)
#endif

namespace boost {
namespace wave {
namespace cpplexer {

#if BOOST_WAVE_SEPARATE_LEXER_INSTANTIATION != 0
#define BOOST_WAVE_NEW_LEXER_DECL BOOST_WAVE_DECL
#else
#define BOOST_WAVE_NEW_LEXER_DECL
#endif

template <
typename IteratorT,
typename PositionT = boost::wave::util::file_position_type,
typename TokenT = lex_token<PositionT>
>
struct BOOST_WAVE_NEW_LEXER_DECL new_lexer_gen
{
static lex_input_interface<TokenT> *
new_lexer(IteratorT const &first, IteratorT const &last,
PositionT const &pos, boost::wave::language_support language);
};

#undef BOOST_WAVE_NEW_LEXER_DECL


template <typename TokenT>
struct lex_input_interface_generator
:   lex_input_interface<TokenT>
{
typedef typename lex_input_interface<TokenT>::position_type position_type;

lex_input_interface_generator() {}
~lex_input_interface_generator() {}

template <typename IteratorT>
static lex_input_interface<TokenT> *
new_lexer(IteratorT const &first, IteratorT const &last,
position_type const &pos, boost::wave::language_support language)
{
return new_lexer_gen<IteratorT, position_type, TokenT>::new_lexer (
first, last, pos, language);
}
};

}   
}   
}   

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_SUFFIX
#endif

#endif 
