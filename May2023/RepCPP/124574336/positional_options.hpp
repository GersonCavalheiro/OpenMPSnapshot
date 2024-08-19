
#ifndef BOOST_PROGRAM_OPTIONS_POSITIONAL_OPTIONS_VP_2004_03_02
#define BOOST_PROGRAM_OPTIONS_POSITIONAL_OPTIONS_VP_2004_03_02

#include <boost/program_options/config.hpp>

#include <vector>
#include <string>

#if defined(BOOST_MSVC)
#   pragma warning (push)
#   pragma warning (disable:4251) 
#endif

namespace boost { namespace program_options {


class BOOST_PROGRAM_OPTIONS_DECL positional_options_description {
public:
positional_options_description();


positional_options_description&
add(const char* name, int max_count);


unsigned max_total_count() const;


const std::string& name_for_position(unsigned position) const;

private:
std::vector<std::string> m_names;
std::string m_trailing;
};

}}

#if defined(BOOST_MSVC)
#   pragma warning (pop)
#endif

#endif

