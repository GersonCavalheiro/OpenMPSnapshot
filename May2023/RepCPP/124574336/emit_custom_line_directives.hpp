

#if !defined(BOOST_WAVE_emit_custom_line_directives_HOOKS_INCLUDED)
#define BOOST_WAVE_emit_custom_line_directives_HOOKS_INCLUDED

#include <cstdio>
#include <ostream>
#include <string>
#include <algorithm>

#include <boost/assert.hpp>
#include <boost/config.hpp>

#include <boost/wave/token_ids.hpp>
#include <boost/wave/util/macro_helpers.hpp>
#include <boost/wave/preprocessing_hooks.hpp>

class emit_custom_line_directives_hooks
:   public boost::wave::context_policies::default_preprocessing_hooks
{
public:
template <typename ContextT, typename ContainerT>
bool 
emit_line_directive(ContextT const& ctx, ContainerT &pending, 
typename ContextT::token_type const& act_token)
{
typename ContextT::position_type pos = act_token.get_position();
unsigned int column = 1;

typedef typename ContextT::token_type result_type;
using namespace boost::wave;

pos.set_column(column);
pending.push_back(result_type(T_POUND, "#", pos));

pos.set_column(++column);      
pending.push_back(result_type(T_SPACE, " ", pos));

char buffer[22];

using namespace std;    
sprintf (buffer, "%d", pos.get_line());

pos.set_column(++column);                 
pending.push_back(result_type(T_INTLIT, buffer, pos));
pos.set_column(column += (unsigned int)strlen(buffer)); 
pending.push_back(result_type(T_SPACE, " ", pos));
pos.set_column(++column);                 

std::string file("\"");
boost::filesystem::path filename(
boost::wave::util::create_path(ctx.get_current_relative_filename().c_str()));

using boost::wave::util::impl::escape_lit;
file += escape_lit(boost::wave::util::native_file_string(filename)) + "\"";

pending.push_back(result_type(T_STRINGLIT, file.c_str(), pos));
pos.set_column(column += (unsigned int)file.size());    
pending.push_back(result_type(T_GENERATEDNEWLINE, "\n", pos));

return true;
}
};

#endif 
