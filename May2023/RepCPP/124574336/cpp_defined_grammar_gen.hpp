

#if !defined(BOOST_CPP_DEFINED_GRAMMAR_GEN_HPP_825BE9F5_98A3_400D_A97C_AD76B3B08632_INCLUDED)
#define BOOST_CPP_DEFINED_GRAMMAR_GEN_HPP_825BE9F5_98A3_400D_A97C_AD76B3B08632_INCLUDED

#include <boost/wave/wave_config.hpp>

#include <list>

#include <boost/spirit/include/classic_parser.hpp>
#include <boost/pool/pool_alloc.hpp>

#include <boost/wave/util/unput_queue_iterator.hpp>

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

template <typename LexIteratorT>
struct BOOST_WAVE_DECL defined_grammar_gen
{
typedef typename LexIteratorT::token_type token_type;
typedef std::list<token_type, boost::fast_pool_allocator<token_type> >
token_sequence_type;


typedef boost::wave::util::unput_queue_iterator<
typename token_sequence_type::iterator, token_type, token_sequence_type>
iterator1_type;

typedef boost::wave::util::unput_queue_iterator<
LexIteratorT, token_type, token_sequence_type>
iterator2_type;

static boost::spirit::classic::parse_info<iterator1_type>
parse_operator_defined (iterator1_type const &first,
iterator1_type const &last, token_sequence_type &found_qualified_name);

static boost::spirit::classic::parse_info<iterator2_type>
parse_operator_defined (iterator2_type const &first,
iterator2_type const &last, token_sequence_type &found_qualified_name);
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
