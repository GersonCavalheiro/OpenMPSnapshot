

#if !defined(BOOST_CPP_LEX_INTERFACE_HPP_E83F52A4_90AC_4FBE_A9A7_B65F7F94C497_INCLUDED)
#define BOOST_CPP_LEX_INTERFACE_HPP_E83F52A4_90AC_4FBE_A9A7_B65F7F94C497_INCLUDED

#include <boost/wave/wave_config.hpp>
#include <boost/wave/util/file_position.hpp>
#include <boost/wave/language_support.hpp>

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


template <typename TokenT>
struct lex_input_interface
{
typedef typename TokenT::position_type position_type;

lex_input_interface() {}
virtual ~lex_input_interface() {}

virtual TokenT& get(TokenT&) = 0;
virtual void set_position(position_type const &pos) = 0;
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
virtual bool has_include_guards(std::string& guard_name) const = 0;
#endif
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
