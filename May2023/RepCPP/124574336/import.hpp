
#ifndef BOOST_DLL_IMPORT_HPP
#define BOOST_DLL_IMPORT_HPP

#include <boost/dll/config.hpp>
#include <boost/core/addressof.hpp>
#include <boost/core/enable_if.hpp>
#include <boost/type_traits/is_object.hpp>
#include <boost/make_shared.hpp>
#include <boost/dll/shared_library.hpp>
#include <boost/move/move.hpp>

#if defined(BOOST_NO_CXX11_TRAILING_RESULT_TYPES) || defined(BOOST_NO_CXX11_DECLTYPE) || defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) || defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#   include <boost/function.hpp>
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif


namespace boost { namespace dll {


namespace detail {

template <class T>
class library_function {
boost::shared_ptr<T>   f_;

public:
inline library_function(const boost::shared_ptr<shared_library>& lib, T* func_ptr) BOOST_NOEXCEPT
: f_(lib, func_ptr)
{}

#if defined(BOOST_NO_CXX11_TRAILING_RESULT_TYPES) || defined(BOOST_NO_CXX11_DECLTYPE) || defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) || defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
operator T*() const BOOST_NOEXCEPT {
return f_.get();
}
#else

template <class... Args>
inline auto operator()(Args&&... args) const
-> decltype( (*f_)(static_cast<Args&&>(args)...) )
{
return (*f_)(static_cast<Args&&>(args)...);
}
#endif
};

template <class T, class = void>
struct import_type;

template <class T>
struct import_type<T, typename boost::disable_if<boost::is_object<T> >::type> {
typedef boost::dll::detail::library_function<T> base_type;

#if defined(BOOST_NO_CXX11_TRAILING_RESULT_TYPES) || defined(BOOST_NO_CXX11_DECLTYPE) || defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) || defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
typedef boost::function<T>                      type;
#else
typedef boost::dll::detail::library_function<T> type;
#endif
};

template <class T>
struct import_type<T, typename boost::enable_if<boost::is_object<T> >::type> {
typedef boost::shared_ptr<T> base_type;
typedef boost::shared_ptr<T> type;
};
} 


#ifndef BOOST_DLL_DOXYGEN
#   define BOOST_DLL_IMPORT_RESULT_TYPE inline typename boost::dll::detail::import_type<T>::type
#endif



template <class T>
BOOST_DLL_IMPORT_RESULT_TYPE import(const boost::dll::fs::path& lib, const char* name,
load_mode::type mode = load_mode::default_mode)
{
typedef typename boost::dll::detail::import_type<T>::base_type type;

boost::shared_ptr<boost::dll::shared_library> p = boost::make_shared<boost::dll::shared_library>(lib, mode);
return type(p, boost::addressof(p->get<T>(name)));
}

template <class T>
BOOST_DLL_IMPORT_RESULT_TYPE import(const boost::dll::fs::path& lib, const std::string& name,
load_mode::type mode = load_mode::default_mode)
{
return import<T>(lib, name.c_str(), mode);
}

template <class T>
BOOST_DLL_IMPORT_RESULT_TYPE import(const shared_library& lib, const char* name) {
typedef typename boost::dll::detail::import_type<T>::base_type type;

boost::shared_ptr<boost::dll::shared_library> p = boost::make_shared<boost::dll::shared_library>(lib);
return type(p, boost::addressof(p->get<T>(name)));
}

template <class T>
BOOST_DLL_IMPORT_RESULT_TYPE import(const shared_library& lib, const std::string& name) {
return import<T>(lib, name.c_str());
}

template <class T>
BOOST_DLL_IMPORT_RESULT_TYPE import(BOOST_RV_REF(shared_library) lib, const char* name) {
typedef typename boost::dll::detail::import_type<T>::base_type type;

boost::shared_ptr<boost::dll::shared_library> p = boost::make_shared<boost::dll::shared_library>(
boost::move(lib)
);
return type(p, boost::addressof(p->get<T>(name)));
}

template <class T>
BOOST_DLL_IMPORT_RESULT_TYPE import(BOOST_RV_REF(shared_library) lib, const std::string& name) {
return import<T>(boost::move(lib), name.c_str());
}





template <class T>
BOOST_DLL_IMPORT_RESULT_TYPE import_alias(const boost::dll::fs::path& lib, const char* name,
load_mode::type mode = load_mode::default_mode)
{
typedef typename boost::dll::detail::import_type<T>::base_type type;

boost::shared_ptr<boost::dll::shared_library> p = boost::make_shared<boost::dll::shared_library>(lib, mode);
return type(p, p->get<T*>(name));
}

template <class T>
BOOST_DLL_IMPORT_RESULT_TYPE import_alias(const boost::dll::fs::path& lib, const std::string& name,
load_mode::type mode = load_mode::default_mode)
{
return import_alias<T>(lib, name.c_str(), mode);
}

template <class T>
BOOST_DLL_IMPORT_RESULT_TYPE import_alias(const shared_library& lib, const char* name) {
typedef typename boost::dll::detail::import_type<T>::base_type type;

boost::shared_ptr<boost::dll::shared_library> p = boost::make_shared<boost::dll::shared_library>(lib);
return type(p, p->get<T*>(name));
}

template <class T>
BOOST_DLL_IMPORT_RESULT_TYPE import_alias(const shared_library& lib, const std::string& name) {
return import_alias<T>(lib, name.c_str());
}

template <class T>
BOOST_DLL_IMPORT_RESULT_TYPE import_alias(BOOST_RV_REF(shared_library) lib, const char* name) {
typedef typename boost::dll::detail::import_type<T>::base_type type;

boost::shared_ptr<boost::dll::shared_library> p = boost::make_shared<boost::dll::shared_library>(
boost::move(lib)
);
return type(p, p->get<T*>(name));
}

template <class T>
BOOST_DLL_IMPORT_RESULT_TYPE import_alias(BOOST_RV_REF(shared_library) lib, const std::string& name) {
return import_alias<T>(boost::move(lib), name.c_str());
}

#undef BOOST_DLL_IMPORT_RESULT_TYPE


}} 

#endif 

