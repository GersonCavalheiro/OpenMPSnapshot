
#ifndef BOOST_DLL_LIBRARY_INFO_HPP
#define BOOST_DLL_LIBRARY_INFO_HPP

#include <boost/dll/config.hpp>
#include <boost/assert.hpp>
#include <boost/noncopyable.hpp>
#include <boost/predef/os.h>
#include <boost/predef/architecture.h>
#include <boost/throw_exception.hpp>
#include <boost/type_traits/integral_constant.hpp>

#include <fstream>

#include <boost/dll/detail/pe_info.hpp>
#include <boost/dll/detail/elf_info.hpp>
#include <boost/dll/detail/macho_info.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif


namespace boost { namespace dll {


class library_info: private boost::noncopyable {
private:
std::ifstream f_;

enum {
fmt_elf_info32,
fmt_elf_info64,
fmt_pe_info32,
fmt_pe_info64,
fmt_macho_info32,
fmt_macho_info64
} fmt_;

inline static void throw_if_in_32bit_impl(boost::true_type ) {
boost::throw_exception(std::runtime_error("Not native format: 64bit binary"));
}

inline static void throw_if_in_32bit_impl(boost::false_type ) BOOST_NOEXCEPT {}


inline static void throw_if_in_32bit() {
throw_if_in_32bit_impl( boost::integral_constant<bool, (sizeof(void*) == 4)>() );
}

static void throw_if_in_windows() {
#if BOOST_OS_WINDOWS
boost::throw_exception(std::runtime_error("Not native format: not a PE binary"));
#endif
}

static void throw_if_in_linux() {
#if !BOOST_OS_WINDOWS && !BOOST_OS_MACOS && !BOOST_OS_IOS
boost::throw_exception(std::runtime_error("Not native format: not an ELF binary"));
#endif
}

static void throw_if_in_macos() {
#if BOOST_OS_MACOS || BOOST_OS_IOS
boost::throw_exception(std::runtime_error("Not native format: not an Mach-O binary"));
#endif
}

void init(bool throw_if_not_native) {
if (boost::dll::detail::elf_info32::parsing_supported(f_)) {
if (throw_if_not_native) { throw_if_in_windows(); throw_if_in_macos(); }

fmt_ = fmt_elf_info32;
} else if (boost::dll::detail::elf_info64::parsing_supported(f_)) {
if (throw_if_not_native) { throw_if_in_windows(); throw_if_in_macos(); throw_if_in_32bit(); }

fmt_ = fmt_elf_info64;
} else if (boost::dll::detail::pe_info32::parsing_supported(f_)) {
if (throw_if_not_native) { throw_if_in_linux(); throw_if_in_macos(); }

fmt_ = fmt_pe_info32;
} else if (boost::dll::detail::pe_info64::parsing_supported(f_)) {
if (throw_if_not_native) { throw_if_in_linux(); throw_if_in_macos(); throw_if_in_32bit(); }

fmt_ = fmt_pe_info64;
} else if (boost::dll::detail::macho_info32::parsing_supported(f_)) {
if (throw_if_not_native) { throw_if_in_linux(); throw_if_in_windows(); }

fmt_ = fmt_macho_info32;
} else if (boost::dll::detail::macho_info64::parsing_supported(f_)) {
if (throw_if_not_native) { throw_if_in_linux(); throw_if_in_windows(); throw_if_in_32bit(); }

fmt_ = fmt_macho_info64;
} else {
boost::throw_exception(std::runtime_error("Unsupported binary format"));
}
}

public:

explicit library_info(const boost::dll::fs::path& library_path, bool throw_if_not_native_format = true)
: f_(
#ifdef BOOST_DLL_USE_STD_FS
library_path,
#elif defined(BOOST_WINDOWS_API)  && (!defined(_CPPLIB_VER) || _CPPLIB_VER < 405 || defined(_STLPORT_VERSION))
library_path.string().c_str(),  
#else  
library_path.c_str(),
#endif
std::ios_base::in | std::ios_base::binary
)
{
f_.exceptions(
std::ios_base::failbit
| std::ifstream::badbit
| std::ifstream::eofbit
);

init(throw_if_not_native_format);
}


std::vector<std::string> sections() {
switch (fmt_) {
case fmt_elf_info32:   return boost::dll::detail::elf_info32::sections(f_);
case fmt_elf_info64:   return boost::dll::detail::elf_info64::sections(f_);
case fmt_pe_info32:    return boost::dll::detail::pe_info32::sections(f_);
case fmt_pe_info64:    return boost::dll::detail::pe_info64::sections(f_);
case fmt_macho_info32: return boost::dll::detail::macho_info32::sections(f_);
case fmt_macho_info64: return boost::dll::detail::macho_info64::sections(f_);
};
BOOST_ASSERT(false);
BOOST_UNREACHABLE_RETURN(std::vector<std::string>())
}


std::vector<std::string> symbols() {
switch (fmt_) {
case fmt_elf_info32:   return boost::dll::detail::elf_info32::symbols(f_);
case fmt_elf_info64:   return boost::dll::detail::elf_info64::symbols(f_);
case fmt_pe_info32:    return boost::dll::detail::pe_info32::symbols(f_);
case fmt_pe_info64:    return boost::dll::detail::pe_info64::symbols(f_);
case fmt_macho_info32: return boost::dll::detail::macho_info32::symbols(f_);
case fmt_macho_info64: return boost::dll::detail::macho_info64::symbols(f_);
};
BOOST_ASSERT(false);
BOOST_UNREACHABLE_RETURN(std::vector<std::string>())
}


std::vector<std::string> symbols(const char* section_name) {
switch (fmt_) {
case fmt_elf_info32:   return boost::dll::detail::elf_info32::symbols(f_, section_name);
case fmt_elf_info64:   return boost::dll::detail::elf_info64::symbols(f_, section_name);
case fmt_pe_info32:    return boost::dll::detail::pe_info32::symbols(f_, section_name);
case fmt_pe_info64:    return boost::dll::detail::pe_info64::symbols(f_, section_name);
case fmt_macho_info32: return boost::dll::detail::macho_info32::symbols(f_, section_name);
case fmt_macho_info64: return boost::dll::detail::macho_info64::symbols(f_, section_name);
};
BOOST_ASSERT(false);
BOOST_UNREACHABLE_RETURN(std::vector<std::string>())
}


std::vector<std::string> symbols(const std::string& section_name) {
switch (fmt_) {
case fmt_elf_info32:   return boost::dll::detail::elf_info32::symbols(f_, section_name.c_str());
case fmt_elf_info64:   return boost::dll::detail::elf_info64::symbols(f_, section_name.c_str());
case fmt_pe_info32:    return boost::dll::detail::pe_info32::symbols(f_, section_name.c_str());
case fmt_pe_info64:    return boost::dll::detail::pe_info64::symbols(f_, section_name.c_str());
case fmt_macho_info32: return boost::dll::detail::macho_info32::symbols(f_, section_name.c_str());
case fmt_macho_info64: return boost::dll::detail::macho_info64::symbols(f_, section_name.c_str());
};
BOOST_ASSERT(false);
BOOST_UNREACHABLE_RETURN(std::vector<std::string>())
}
};

}} 
#endif 
