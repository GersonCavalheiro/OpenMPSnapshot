
#ifndef BOOST_DLL_SHARED_LIBRARY_HPP
#define BOOST_DLL_SHARED_LIBRARY_HPP


#include <boost/dll/config.hpp>
#include <boost/predef/os.h>
#include <boost/core/enable_if.hpp>
#include <boost/core/explicit_operator_bool.hpp>
#include <boost/type_traits/is_member_pointer.hpp>
#include <boost/dll/detail/system_error.hpp>
#include <boost/dll/detail/aggressive_ptr_cast.hpp>

#if BOOST_OS_WINDOWS
#   include <boost/dll/detail/windows/shared_library_impl.hpp>
#else
#   include <boost/dll/detail/posix/shared_library_impl.hpp>
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

namespace boost { namespace dll {


class shared_library
: private boost::dll::detail::shared_library_impl
{
typedef boost::dll::detail::shared_library_impl base_t;
BOOST_COPYABLE_AND_MOVABLE(shared_library)

public:
#ifdef BOOST_DLL_DOXYGEN
typedef platform_specific native_handle_t;
#else
typedef shared_library_impl::native_handle_t native_handle_t;
#endif


shared_library() BOOST_NOEXCEPT {}


shared_library(const shared_library& lib)
: base_t()
{
assign(lib);
}


shared_library(const shared_library& lib, boost::dll::fs::error_code& ec)
: base_t()
{
assign(lib, ec);
}


shared_library(BOOST_RV_REF(shared_library) lib) BOOST_NOEXCEPT
: base_t(boost::move(static_cast<base_t&>(lib)))
{}


explicit shared_library(const boost::dll::fs::path& lib_path, load_mode::type mode = load_mode::default_mode) {
shared_library::load(lib_path, mode);
}


shared_library(const boost::dll::fs::path& lib_path, boost::dll::fs::error_code& ec, load_mode::type mode = load_mode::default_mode) {
shared_library::load(lib_path, mode, ec);
}

shared_library(const boost::dll::fs::path& lib_path, load_mode::type mode, boost::dll::fs::error_code& ec) {
shared_library::load(lib_path, mode, ec);
}


shared_library& operator=(BOOST_COPY_ASSIGN_REF(shared_library) lib) {
boost::dll::fs::error_code ec;
assign(lib, ec);
if (ec) {
boost::dll::detail::report_error(ec, "boost::dll::shared_library::operator= failed");
}

return *this;
}


shared_library& operator=(BOOST_RV_REF(shared_library) lib) BOOST_NOEXCEPT {
if (lib.native() != native()) {
swap(lib);
}

return *this;
}


~shared_library() BOOST_NOEXCEPT {}


shared_library& assign(const shared_library& lib, boost::dll::fs::error_code& ec) {
ec.clear();

if (native() == lib.native()) {
return *this;
}

if (!lib) {
unload();
return *this;
}

boost::dll::fs::path loc = lib.location(ec);
if (ec) {
return *this;
}

shared_library copy(loc, ec);
if (ec) {
return *this;
}

swap(copy);
return *this;
}


shared_library& assign(const shared_library& lib) {
boost::dll::fs::error_code ec;
assign(lib, ec);
if (ec) {
boost::dll::detail::report_error(ec, "boost::dll::shared_library::assign() failed");
}

return *this;
}


void load(const boost::dll::fs::path& lib_path, load_mode::type mode = load_mode::default_mode) {
boost::dll::fs::error_code ec;

base_t::load(lib_path, mode, ec);

if (ec) {
boost::dll::detail::report_error(ec, "boost::dll::shared_library::load() failed");
}
}


void load(const boost::dll::fs::path& lib_path, boost::dll::fs::error_code& ec, load_mode::type mode = load_mode::default_mode) {
ec.clear();
base_t::load(lib_path, mode, ec);
}

void load(const boost::dll::fs::path& lib_path, load_mode::type mode, boost::dll::fs::error_code& ec) {
ec.clear();
base_t::load(lib_path, mode, ec);
}


void unload() BOOST_NOEXCEPT {
base_t::unload();
}


bool is_loaded() const BOOST_NOEXCEPT {
return base_t::is_loaded();
}


bool operator!() const BOOST_NOEXCEPT {
return !is_loaded();
}


BOOST_EXPLICIT_OPERATOR_BOOL()


bool has(const char* symbol_name) const BOOST_NOEXCEPT {
boost::dll::fs::error_code ec;
return is_loaded() && !!base_t::symbol_addr(symbol_name, ec) && !ec;
}

bool has(const std::string& symbol_name) const BOOST_NOEXCEPT {
return has(symbol_name.c_str());
}


template <typename T>
inline typename boost::enable_if_c<boost::is_member_pointer<T>::value || boost::is_reference<T>::value, T>::type  get(const std::string& symbol_name) const {
return get<T>(symbol_name.c_str());
}

template <typename T>
inline typename boost::disable_if_c<boost::is_member_pointer<T>::value || boost::is_reference<T>::value, T&>::type get(const std::string& symbol_name) const {
return get<T>(symbol_name.c_str());
}

template <typename T>
inline typename boost::enable_if_c<boost::is_member_pointer<T>::value || boost::is_reference<T>::value, T>::type get(const char* symbol_name) const {
return boost::dll::detail::aggressive_ptr_cast<T>(
get_void(symbol_name)
);
}

template <typename T>
inline typename boost::disable_if_c<boost::is_member_pointer<T>::value || boost::is_reference<T>::value, T&>::type get(const char* symbol_name) const {
return *boost::dll::detail::aggressive_ptr_cast<T*>(
get_void(symbol_name)
);
}


template <typename T>
inline T& get_alias(const char* alias_name) const {
return *get<T*>(alias_name);
}

template <typename T>
inline T& get_alias(const std::string& alias_name) const {
return *get<T*>(alias_name.c_str());
}

private:
void* get_void(const char* sb) const {
boost::dll::fs::error_code ec;

if (!is_loaded()) {
ec = boost::dll::fs::make_error_code(
boost::dll::fs::errc::bad_file_descriptor
);

boost::throw_exception(
boost::dll::fs::system_error(
ec, "boost::dll::shared_library::get() failed: no library was loaded"
)
);
}

void* const ret = base_t::symbol_addr(sb, ec);
if (ec || !ret) {
boost::dll::detail::report_error(ec, "boost::dll::shared_library::get() failed");
}

return ret;
}

public:


native_handle_t native() const BOOST_NOEXCEPT {
return base_t::native();
}


boost::dll::fs::path location() const {
boost::dll::fs::error_code ec;
if (!is_loaded()) {
ec = boost::dll::fs::make_error_code(
boost::dll::fs::errc::bad_file_descriptor
);

boost::throw_exception(
boost::dll::fs::system_error(
ec, "boost::dll::shared_library::location() failed (no library was loaded)"
)
);
}

boost::dll::fs::path full_path = base_t::full_module_path(ec);

if (ec) {
boost::dll::detail::report_error(ec, "boost::dll::shared_library::location() failed");
}

return full_path;
}


boost::dll::fs::path location(boost::dll::fs::error_code& ec) const {
if (!is_loaded()) {
ec = boost::dll::fs::make_error_code(
boost::dll::fs::errc::bad_file_descriptor
);

return boost::dll::fs::path();
}

ec.clear();
return base_t::full_module_path(ec);
}


static boost::dll::fs::path suffix() {
return base_t::suffix();
}


static boost::dll::fs::path decorate(const boost::dll::fs::path& sl) {
return base_t::decorate(sl);
}


void swap(shared_library& rhs) BOOST_NOEXCEPT {
base_t::swap(rhs);
}
};



inline bool operator==(const shared_library& lhs, const shared_library& rhs) BOOST_NOEXCEPT {
return lhs.native() == rhs.native();
}

inline bool operator!=(const shared_library& lhs, const shared_library& rhs) BOOST_NOEXCEPT {
return lhs.native() != rhs.native();
}

inline bool operator<(const shared_library& lhs, const shared_library& rhs) BOOST_NOEXCEPT {
return lhs.native() < rhs.native();
}

inline void swap(shared_library& lhs, shared_library& rhs) BOOST_NOEXCEPT {
lhs.swap(rhs);
}

}} 

#endif 
