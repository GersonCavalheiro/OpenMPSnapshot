

#ifndef BOOST_DLL_IMPORT_MANGLED_HPP_
#define BOOST_DLL_IMPORT_MANGLED_HPP_


#include <boost/dll/config.hpp>
#if (__cplusplus < 201103L) && (!defined(_MSVC_LANG) || _MSVC_LANG < 201103L)
#  error This file requires C++11 at least!
#endif

#include <boost/make_shared.hpp>
#include <boost/move/move.hpp>
#include <boost/dll/smart_library.hpp>
#include <boost/dll/detail/import_mangled_helpers.hpp>
#include <boost/core/addressof.hpp>
#include <boost/core/enable_if.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/type_traits/is_object.hpp>


#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

namespace boost { namespace dll { namespace experimental {

namespace detail
{

template <class ... Ts>
class mangled_library_function {
boost::shared_ptr<shared_library> lib_;
function_tuple<Ts...>   f_;
public:
constexpr mangled_library_function(const boost::shared_ptr<shared_library>& lib, Ts*... func_ptr) BOOST_NOEXCEPT
: lib_(lib)
, f_(func_ptr...)
{}


template <class... Args>
auto operator()(Args&&... args) const
-> decltype( f_(static_cast<Args&&>(args)...) )
{
return f_(static_cast<Args&&>(args)...);
}
};


template<class Class, class Sequence>
class mangled_library_mem_fn;

template <class Class, class ... Ts>
class mangled_library_mem_fn<Class, sequence<Ts...>> {
typedef mem_fn_tuple<Ts...> call_tuple_t;
boost::shared_ptr<shared_library>   lib_;
call_tuple_t f_;

public:
constexpr mangled_library_mem_fn(const boost::shared_ptr<shared_library>& lib, typename Ts::mem_fn... func_ptr) BOOST_NOEXCEPT
: lib_(lib)
, f_(func_ptr...)
{}

template <class ClassIn, class... Args>
auto operator()(ClassIn *cl, Args&&... args) const
-> decltype( f_(cl, static_cast<Args&&>(args)...) )
{
return f_(cl, static_cast<Args&&>(args)...);
}
};




template<class Seq>  struct is_variable : boost::false_type {};
template<typename T> struct is_variable<sequence<T>> : boost::is_object<T> {};

template <class Sequence,
bool isFunction = is_function_seq<Sequence>::value,
bool isMemFn    = is_mem_fn_seq  <Sequence>::value,
bool isVariable = is_variable    <Sequence>::value>
struct mangled_import_type;

template <class ...Args>
struct mangled_import_type<sequence<Args...>, true,false,false> 
{
typedef boost::dll::experimental::detail::mangled_library_function<Args...> type;
static type make(
const boost::dll::experimental::smart_library& p,
const std::string& name)
{
return type(
boost::make_shared<shared_library>(p.shared_lib()),
boost::addressof(p.get_function<Args>(name))...);
}
};

template <class Class, class ...Args>
struct mangled_import_type<sequence<Class, Args...>, false, true, false> 
{
typedef typename boost::dll::experimental::detail::make_mem_fn_seq<Class, Args...>::type actual_sequence;
typedef typename boost::dll::experimental::detail::mangled_library_mem_fn<Class, actual_sequence> type;


template<class ... ArgsIn>
static type make_impl(
const boost::dll::experimental::smart_library& p,
const std::string & name,
sequence<ArgsIn...> * )
{
return type(boost::make_shared<shared_library>(p.shared_lib()),
p.get_mem_fn<typename ArgsIn::class_type, typename ArgsIn::func_type>(name)...);
}

static type make(
const boost::dll::experimental::smart_library& p,
const std::string& name)
{
return make_impl(p, name, static_cast<actual_sequence*>(nullptr));
}

};

template <class T>
struct mangled_import_type<sequence<T>, false, false, true> 
{
typedef boost::shared_ptr<T> type;

static type make(
const boost::dll::experimental::smart_library& p,
const std::string& name)
{
return type(
boost::make_shared<shared_library>(p.shared_lib()),
boost::addressof(p.get_variable<T>(name)));
}

};


} 


#ifndef BOOST_DLL_DOXYGEN
#   define BOOST_DLL_MANGLED_IMPORT_RESULT_TYPE inline typename \
boost::dll::experimental::detail::mangled_import_type<boost::dll::experimental::detail::sequence<Args...>>::type
#endif






template <class ...Args>
BOOST_DLL_MANGLED_IMPORT_RESULT_TYPE import_mangled(const boost::dll::fs::path& lib, const char* name,
load_mode::type mode = load_mode::default_mode)
{
typedef typename boost::dll::experimental::detail::mangled_import_type<
boost::dll::experimental::detail::sequence<Args...>> type;

boost::dll::experimental::smart_library p(lib, mode);
return type::make(p, name);
}



template <class ...Args>
BOOST_DLL_MANGLED_IMPORT_RESULT_TYPE import_mangled(const boost::dll::fs::path& lib, const std::string& name,
load_mode::type mode = load_mode::default_mode)
{
return import_mangled<Args...>(lib, name.c_str(), mode);
}

template <class ...Args>
BOOST_DLL_MANGLED_IMPORT_RESULT_TYPE import_mangled(const smart_library& lib, const char* name) {
typedef typename boost::dll::experimental::detail::mangled_import_type<detail::sequence<Args...>> type;

return type::make(lib, name);
}

template <class ...Args>
BOOST_DLL_MANGLED_IMPORT_RESULT_TYPE import_mangled(const smart_library& lib, const std::string& name) {
return import_mangled<Args...>(lib, name.c_str());
}

template <class ...Args>
BOOST_DLL_MANGLED_IMPORT_RESULT_TYPE import_mangled(BOOST_RV_REF(smart_library) lib, const char* name) {
typedef typename boost::dll::experimental::detail::mangled_import_type<detail::sequence<Args...>> type;

return type::make(lib, name);
}

template <class ...Args>
BOOST_DLL_MANGLED_IMPORT_RESULT_TYPE import_mangled(BOOST_RV_REF(smart_library) lib, const std::string& name) {
return import_mangled<Args...>(boost::move(lib), name.c_str());
}

template <class ...Args>
BOOST_DLL_MANGLED_IMPORT_RESULT_TYPE import_mangled(const shared_library& lib, const char* name) {
typedef typename boost::dll::experimental::detail::mangled_import_type<detail::sequence<Args...>> type;

boost::shared_ptr<boost::dll::experimental::smart_library> p = boost::make_shared<boost::dll::experimental::smart_library>(lib);
return type::make(p, name);
}

template <class ...Args>
BOOST_DLL_MANGLED_IMPORT_RESULT_TYPE import_mangled(const shared_library& lib, const std::string& name) {
return import_mangled<Args...>(lib, name.c_str());
}

template <class ...Args>
BOOST_DLL_MANGLED_IMPORT_RESULT_TYPE import_mangled(BOOST_RV_REF(shared_library) lib, const char* name) {
typedef typename boost::dll::experimental::detail::mangled_import_type<detail::sequence<Args...>> type;

boost::dll::experimental::smart_library p(boost::move(lib));

return type::make(p, name);
}

template <class ...Args>
BOOST_DLL_MANGLED_IMPORT_RESULT_TYPE import_mangled(BOOST_RV_REF(shared_library) lib, const std::string& name) {
return import_mangled<Args...>(boost::move(lib), name.c_str());
}

#undef BOOST_DLL_MANGLED_IMPORT_RESULT_TYPE

}}}


#endif 
