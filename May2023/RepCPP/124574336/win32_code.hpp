

#ifndef BOOST_OUTCOME_SYSTEM_ERROR2_WIN32_CODE_HPP
#define BOOST_OUTCOME_SYSTEM_ERROR2_WIN32_CODE_HPP

#if !defined(_WIN32) && !defined(BOOST_OUTCOME_STANDARDESE_IS_IN_THE_HOUSE)
#error This file should only be included on Windows
#endif

#include "quick_status_code_from_enum.hpp"

BOOST_OUTCOME_SYSTEM_ERROR2_NAMESPACE_BEGIN

namespace win32
{
using DWORD = unsigned long;
extern DWORD __stdcall GetLastError();
extern DWORD __stdcall FormatMessageW(DWORD dwFlags, const void *lpSource, DWORD dwMessageId, DWORD dwLanguageId, wchar_t *lpBuffer, DWORD nSize, void  *Arguments);
extern int __stdcall WideCharToMultiByte(unsigned int CodePage, DWORD dwFlags, const wchar_t *lpWideCharStr, int cchWideChar, char *lpMultiByteStr, int cbMultiByte, const char *lpDefaultChar, int *lpUsedDefaultChar);
#pragma comment(lib, "kernel32.lib")
#if defined(_WIN64)
#pragma comment(linker, "/alternatename:?GetLastError@win32@system_error2@@YAKXZ=GetLastError")
#pragma comment(linker, "/alternatename:?FormatMessageW@win32@system_error2@@YAKKPEBXKKPEA_WKPEAX@Z=FormatMessageW")
#pragma comment(linker, "/alternatename:?WideCharToMultiByte@win32@system_error2@@YAHIKPEB_WHPEADHPEBDPEAH@Z=WideCharToMultiByte")
#else
#pragma comment(linker, "/alternatename:?GetLastError@win32@system_error2@@YGKXZ=__imp__GetLastError@0")
#pragma comment(linker, "/alternatename:?FormatMessageW@win32@system_error2@@YGKKPBXKKPA_WKPAX@Z=__imp__FormatMessageW@28")
#pragma comment(linker, "/alternatename:?WideCharToMultiByte@win32@system_error2@@YGHIKPB_WHPADHPBDPAH@Z=__imp__WideCharToMultiByte@32")
#endif
}  

class _win32_code_domain;
class _com_code_domain;
using win32_code = status_code<_win32_code_domain>;
using win32_error = status_error<_win32_code_domain>;

namespace mixins
{
template <class Base> struct mixin<Base, _win32_code_domain> : public Base
{
using Base::Base;

static inline win32_code current() noexcept;
};
}  


class _win32_code_domain : public status_code_domain
{
template <class DomainType> friend class status_code;
template <class StatusCode> friend class detail::indirecting_domain;
friend class _com_code_domain;
using _base = status_code_domain;
static int _win32_code_to_errno(win32::DWORD c)
{
switch(c)
{
case 0:
return 0;
#include "detail/win32_code_to_generic_code.ipp"
}
return -1;
}
static _base::string_ref _make_string_ref(win32::DWORD c) noexcept
{
wchar_t buffer[32768];
win32::DWORD wlen = win32::FormatMessageW(0x00001000  | 0x00000200 , nullptr, c, 0, buffer, 32768, nullptr);
size_t allocation = wlen + (wlen >> 1);
win32::DWORD bytes;
if(wlen == 0)
{
return _base::string_ref("failed to get message from system");
}
for(;;)
{
auto *p = static_cast<char *>(malloc(allocation));  
if(p == nullptr)
{
return _base::string_ref("failed to get message from system");
}
bytes = win32::WideCharToMultiByte(65001 , 0, buffer, (int) (wlen + 1), p, (int) allocation, nullptr, nullptr);
if(bytes != 0)
{
char *end = strchr(p, 0);
while(end[-1] == 10 || end[-1] == 13)
{
--end;
}
*end = 0;  
return _base::atomic_refcounted_string_ref(p, end - p);
}
free(p);  
if(win32::GetLastError() == 0x7a )
{
allocation += allocation >> 2;
continue;
}
return _base::string_ref("failed to get message from system");
}
}

public:
using value_type = win32::DWORD;
using _base::string_ref;

public:
constexpr explicit _win32_code_domain(typename _base::unique_id_type id = 0x8cd18ee72d680f1b) noexcept
: _base(id)
{
}
_win32_code_domain(const _win32_code_domain &) = default;
_win32_code_domain(_win32_code_domain &&) = default;
_win32_code_domain &operator=(const _win32_code_domain &) = default;
_win32_code_domain &operator=(_win32_code_domain &&) = default;
~_win32_code_domain() = default;

static inline constexpr const _win32_code_domain &get();

virtual string_ref name() const noexcept override { return string_ref("win32 domain"); }  
protected:
virtual bool _do_failure(const status_code<void> &code) const noexcept override  
{
assert(code.domain() == *this);
return static_cast<const win32_code &>(code).value() != 0;  
}
virtual bool _do_equivalent(const status_code<void> &code1, const status_code<void> &code2) const noexcept override  
{
assert(code1.domain() == *this);
const auto &c1 = static_cast<const win32_code &>(code1);  
if(code2.domain() == *this)
{
const auto &c2 = static_cast<const win32_code &>(code2);  
return c1.value() == c2.value();
}
if(code2.domain() == generic_code_domain)
{
const auto &c2 = static_cast<const generic_code &>(code2);  
if(static_cast<int>(c2.value()) == _win32_code_to_errno(c1.value()))
{
return true;
}
}
return false;
}
virtual generic_code _generic_code(const status_code<void> &code) const noexcept override  
{
assert(code.domain() == *this);
const auto &c = static_cast<const win32_code &>(code);  
return generic_code(static_cast<errc>(_win32_code_to_errno(c.value())));
}
virtual string_ref _do_message(const status_code<void> &code) const noexcept override  
{
assert(code.domain() == *this);
const auto &c = static_cast<const win32_code &>(code);  
return _make_string_ref(c.value());
}
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS) || defined(BOOST_OUTCOME_STANDARDESE_IS_IN_THE_HOUSE)
BOOST_OUTCOME_SYSTEM_ERROR2_NORETURN virtual void _do_throw_exception(const status_code<void> &code) const override  
{
assert(code.domain() == *this);
const auto &c = static_cast<const win32_code &>(code);  
throw status_error<_win32_code_domain>(c);
}
#endif
};
constexpr _win32_code_domain win32_code_domain;
inline constexpr const _win32_code_domain &_win32_code_domain::get()
{
return win32_code_domain;
}

namespace mixins
{
template <class Base> inline win32_code mixin<Base, _win32_code_domain>::current() noexcept { return win32_code(win32::GetLastError()); }
}  

BOOST_OUTCOME_SYSTEM_ERROR2_NAMESPACE_END

#endif
