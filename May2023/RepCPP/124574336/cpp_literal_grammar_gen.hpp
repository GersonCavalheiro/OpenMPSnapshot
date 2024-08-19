

#if !defined(BOOST_CPP_LITERAL_GRAMMAR_GEN_HPP_67794A6C_468A_4AAB_A757_DEDDB182F5A0_INCLUDED)
#define BOOST_CPP_LITERAL_GRAMMAR_GEN_HPP_67794A6C_468A_4AAB_A757_DEDDB182F5A0_INCLUDED

#include <boost/wave/wave_config.hpp>
#include <boost/wave/grammars/cpp_value_error.hpp>

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

template <typename TokenT>
struct BOOST_WAVE_DECL intlit_grammar_gen {

static uint_literal_type evaluate(TokenT const &tok, bool &is_unsigned);
};

template <typename IntegralResult, typename TokenT>
struct BOOST_WAVE_DECL chlit_grammar_gen {

static IntegralResult evaluate(TokenT const &tok, value_error& status);
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
