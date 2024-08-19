

#ifndef BOOST_OUTCOME_SYSTEM_ERROR2_NT_CODE_HPP
#define BOOST_OUTCOME_SYSTEM_ERROR2_NT_CODE_HPP

#if !defined(_WIN32) && !defined(BOOST_OUTCOME_STANDARDESE_IS_IN_THE_HOUSE)
#error This file should only be included on Windows
#endif

#include "win32_code.hpp"

BOOST_OUTCOME_SYSTEM_ERROR2_NAMESPACE_BEGIN

namespace win32
{
using NTSTATUS = long;
using HMODULE = void *;
extern HMODULE __stdcall GetModuleHandleW(const wchar_t *lpModuleName);
#pragma comment(lib, "kernel32.lib")
#if defined(_WIN64)
#pragma comment(linker, "/alternatename:?GetModuleHandleW@win32@system_error2@@YAPEAXPEB_W@Z=GetModuleHandleW")
#else
#pragma comment(linker, "/alternatename:?GetModuleHandleW@win32@system_error2@@YGPAXPB_W@Z=__imp__GetModuleHandleW@4")
#endif
}  

class _nt_code_domain;
using nt_code = status_code<_nt_code_domain>;
using nt_error = status_error<_nt_code_domain>;


class _nt_code_domain : public status_code_domain
{
template <class DomainType> friend class status_code;
template <class StatusCode> friend class detail::indirecting_domain;
friend class _com_code_domain;
using _base = status_code_domain;
static int _nt_code_to_errno(win32::NTSTATUS c)
{
if(c >= 0)
{
return 0;  
}
switch(static_cast<unsigned>(c))
{
#include "detail/nt_code_to_generic_code.ipp"
}
return -1;
}
static win32::DWORD _nt_code_to_win32_code(win32::NTSTATUS c)  
{
if(c >= 0)
{
return 0;  
}
switch(static_cast<unsigned>(c))
{
#include "detail/nt_code_to_win32_code.ipp"
}
return static_cast<win32::DWORD>(-1);
}
static _base::string_ref _make_string_ref(win32::NTSTATUS c) noexcept
{
wchar_t buffer[32768];
static win32::HMODULE ntdll = win32::GetModuleHandleW(L"NTDLL.DLL");
win32::DWORD wlen = win32::FormatMessageW(0x00000800  | 0x00001000  | 0x00000200 , ntdll, c, (1 << 10) , buffer, 32768, nullptr);
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
using value_type = win32::NTSTATUS;
using _base::string_ref;

public:
constexpr explicit _nt_code_domain(typename _base::unique_id_type id = 0x93f3b4487e4af25b) noexcept
: _base(id)
{
}
_nt_code_domain(const _nt_code_domain &) = default;
_nt_code_domain(_nt_code_domain &&) = default;
_nt_code_domain &operator=(const _nt_code_domain &) = default;
_nt_code_domain &operator=(_nt_code_domain &&) = default;
~_nt_code_domain() = default;

static inline constexpr const _nt_code_domain &get();

virtual string_ref name() const noexcept override { return string_ref("NT domain"); }  
protected:
virtual bool _do_failure(const status_code<void> &code) const noexcept override  
{
assert(code.domain() == *this);
return static_cast<const nt_code &>(code).value() < 0;  
}
virtual bool _do_equivalent(const status_code<void> &code1, const status_code<void> &code2) const noexcept override  
{
assert(code1.domain() == *this);
const auto &c1 = static_cast<const nt_code &>(code1);  
if(code2.domain() == *this)
{
const auto &c2 = static_cast<const nt_code &>(code2);  
return c1.value() == c2.value();
}
if(code2.domain() == generic_code_domain)
{
const auto &c2 = static_cast<const generic_code &>(code2);  
if(static_cast<int>(c2.value()) == _nt_code_to_errno(c1.value()))
{
return true;
}
}
if(code2.domain() == win32_code_domain)
{
const auto &c2 = static_cast<const win32_code &>(code2);  
if(c2.value() == _nt_code_to_win32_code(c1.value()))
{
return true;
}
}
return false;
}
virtual generic_code _generic_code(const status_code<void> &code) const noexcept override  
{
assert(code.domain() == *this);
const auto &c = static_cast<const nt_code &>(code);  
return generic_code(static_cast<errc>(_nt_code_to_errno(c.value())));
}
virtual string_ref _do_message(const status_code<void> &code) const noexcept override  
{
assert(code.domain() == *this);
const auto &c = static_cast<const nt_code &>(code);  
return _make_string_ref(c.value());
}
#if defined(_CPPUNWIND) || defined(__EXCEPTIONS) || defined(BOOST_OUTCOME_STANDARDESE_IS_IN_THE_HOUSE)
BOOST_OUTCOME_SYSTEM_ERROR2_NORETURN virtual void _do_throw_exception(const status_code<void> &code) const override  
{
assert(code.domain() == *this);
const auto &c = static_cast<const nt_code &>(code);  
throw status_error<_nt_code_domain>(c);
}
#endif
};
constexpr _nt_code_domain nt_code_domain;
inline constexpr const _nt_code_domain &_nt_code_domain::get()
{
return nt_code_domain;
}

BOOST_OUTCOME_SYSTEM_ERROR2_NAMESPACE_END

#endif
