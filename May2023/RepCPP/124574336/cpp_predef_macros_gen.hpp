

#if !defined(BOOST_CPP_PREDEF_MACROS_GEN_HPP_CADB6D2C_76A4_4988_83E1_EFFC6902B9A2_INCLUDED)
#define BOOST_CPP_PREDEF_MACROS_GEN_HPP_CADB6D2C_76A4_4988_83E1_EFFC6902B9A2_INCLUDED

#include <boost/spirit/include/classic_parse_tree.hpp>

#include <boost/wave/wave_config.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_PREFIX
#endif

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable : 4251 4231 4660)
#endif

namespace boost {
namespace wave {
namespace grammars {

#define BOOST_WAVE_PLAIN_DEFINE_ID      5
#define BOOST_WAVE_MACRO_PARAMETERS_ID  6
#define BOOST_WAVE_MACRO_DEFINITION_ID  7


template <typename LexIteratorT>
struct BOOST_WAVE_DECL predefined_macros_grammar_gen
{
typedef LexIteratorT iterator_type;

static boost::spirit::classic::tree_parse_info<iterator_type>
parse_predefined_macro (iterator_type const &first, iterator_type const &last);
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
