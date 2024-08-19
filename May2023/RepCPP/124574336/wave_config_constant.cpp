

#define BOOST_WAVE_SOURCE 1

#include <boost/config/warning_disable.hpp>

#include <boost/preprocessor/stringize.hpp>

#include <boost/wave/wave_config.hpp>
#include <boost/wave/wave_config_constant.hpp>

namespace boost { namespace wave {

BOOST_WAVE_DECL bool
test_configuration(unsigned int config, char const* pragma_keyword,
char const* string_type_str)
{
if (NULL == pragma_keyword || NULL == string_type_str)
return false;

using namespace std;;   
if (config != BOOST_WAVE_CONFIG ||
strcmp(pragma_keyword, BOOST_WAVE_PRAGMA_KEYWORD) ||
strcmp(string_type_str, BOOST_PP_STRINGIZE((BOOST_WAVE_STRINGTYPE))))
{
return false;
}
return true;
}

}}  

