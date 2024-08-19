

#if !defined(BOOST_CPP_GRAMMAR_GEN_HPP_80CB8A59_5411_4E45_B406_62531A12FB99_INCLUDED)
#define BOOST_CPP_GRAMMAR_GEN_HPP_80CB8A59_5411_4E45_B406_62531A12FB99_INCLUDED

#include <boost/wave/wave_config.hpp>
#include <boost/wave/language_support.hpp>

#include <boost/spirit/include/classic_nil.hpp>
#include <boost/spirit/include/classic_parse_tree.hpp>

#include <boost/pool/pool_alloc.hpp>

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

#define BOOST_WAVE_PP_STATEMENT_ID        1
#define BOOST_WAVE_INCLUDE_FILE_ID        2
#define BOOST_WAVE_SYSINCLUDE_FILE_ID     3
#define BOOST_WAVE_MACROINCLUDE_FILE_ID   4
#define BOOST_WAVE_PLAIN_DEFINE_ID        5
#define BOOST_WAVE_MACRO_PARAMETERS_ID    6
#define BOOST_WAVE_MACRO_DEFINITION_ID    7
#define BOOST_WAVE_UNDEFINE_ID            8
#define BOOST_WAVE_IFDEF_ID               9
#define BOOST_WAVE_IFNDEF_ID             10
#define BOOST_WAVE_IF_ID                 11
#define BOOST_WAVE_ELIF_ID               12
#define BOOST_WAVE_ELSE_ID               13
#define BOOST_WAVE_ENDIF_ID              14
#define BOOST_WAVE_LINE_ID               15
#define BOOST_WAVE_ERROR_ID              16
#define BOOST_WAVE_WARNING_ID            17
#define BOOST_WAVE_PRAGMA_ID             18
#define BOOST_WAVE_ILLFORMED_ID          19
#define BOOST_WAVE_PPSPACE_ID            20
#define BOOST_WAVE_PPQUALIFIEDNAME_ID    21
#define BOOST_WAVE_REGION_ID             22
#define BOOST_WAVE_ENDREGION_ID          23


template <typename LexIteratorT, typename TokenContainerT>
struct BOOST_WAVE_DECL cpp_grammar_gen
{
typedef LexIteratorT                          iterator_type;
typedef typename LexIteratorT::token_type     token_type;
typedef TokenContainerT                       token_container_type;
typedef typename token_type::position_type    position_type;
typedef boost::spirit::classic::node_val_data_factory<
> node_factory_type;

static boost::spirit::classic::tree_parse_info<iterator_type, node_factory_type>
parse_cpp_grammar (iterator_type const &first, iterator_type const &last,
position_type const &act_pos, bool &found_eof,
token_type &found_directive, token_container_type &found_eoltokens);
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
