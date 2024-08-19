#ifndef BOOST_LOCALE_LOCALIZATION_BACKEND_HPP
#define BOOST_LOCALE_LOCALIZATION_BACKEND_HPP
#include <boost/locale/config.hpp>
#include <boost/locale/generator.hpp>
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4275 4251 4231 4660)
#endif
#include <string>
#include <locale>
#include <vector>
#include <memory>
#include <boost/locale/hold_ptr.hpp>

namespace boost {
namespace locale {


class localization_backend {
localization_backend(localization_backend const &);
void operator=(localization_backend const &);
public:

localization_backend()
{
}

virtual ~localization_backend()
{
}

virtual localization_backend *clone() const = 0;

virtual void set_option(std::string const &name,std::string const &value) = 0;

virtual void clear_options() = 0;

virtual std::locale install(std::locale const &base,locale_category_type category,character_facet_type type = nochar_facet) = 0;

}; 



class BOOST_LOCALE_DECL localization_backend_manager {
public:
localization_backend_manager();
localization_backend_manager(localization_backend_manager const &);
localization_backend_manager const &operator=(localization_backend_manager const &);

~localization_backend_manager();

#if !defined(BOOST_LOCALE_HIDE_AUTO_PTR) && !defined(BOOST_NO_AUTO_PTR)
std::auto_ptr<localization_backend> get() const;

void add_backend(std::string const &name,std::auto_ptr<localization_backend> backend);
#endif

localization_backend *create() const;
void adopt_backend(std::string const &name,localization_backend *backend);
#ifndef BOOST_NO_CXX11_SMART_PTR
std::unique_ptr<localization_backend> get_unique_ptr() const;

void add_backend(std::string const &name,std::unique_ptr<localization_backend> backend);
#endif

void remove_all_backends();

std::vector<std::string> get_all_backends() const;

void select(std::string const &backend_name,locale_category_type category = all_categories);

static localization_backend_manager global(localization_backend_manager const &);
static localization_backend_manager global();
private:
class impl;
hold_ptr<impl> pimpl_;
};

} 
} 


#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif

