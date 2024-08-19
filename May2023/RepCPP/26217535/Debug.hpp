

#pragma once

#include <alpaka/core/BoostPredef.hpp>

#include <iostream>
#include <string>
#include <utility>

#define ALPAKA_DEBUG_DISABLED 0
#define ALPAKA_DEBUG_MINIMAL 1
#define ALPAKA_DEBUG_FULL 2

#ifndef ALPAKA_DEBUG
#    define ALPAKA_DEBUG ALPAKA_DEBUG_DISABLED
#endif

namespace alpaka::core::detail
{
class ScopeLogStdOut final
{
public:
explicit ScopeLogStdOut(std::string sScope) : m_sScope(std::move(sScope))
{
std::cout << "[+] " << m_sScope << std::endl;
}
ScopeLogStdOut(ScopeLogStdOut const&) = delete;
ScopeLogStdOut(ScopeLogStdOut&&) = delete;
auto operator=(ScopeLogStdOut const&) -> ScopeLogStdOut& = delete;
auto operator=(ScopeLogStdOut&&) -> ScopeLogStdOut& = delete;
~ScopeLogStdOut()
{
std::cout << "[-] " << m_sScope << std::endl;
}

private:
std::string const m_sScope;
};
} 

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#    define ALPAKA_DEBUG_MINIMAL_LOG_SCOPE ::alpaka::core::detail::ScopeLogStdOut const scopeLogStdOut(__func__)
#else
#    define ALPAKA_DEBUG_MINIMAL_LOG_SCOPE
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
#    define ALPAKA_DEBUG_FULL_LOG_SCOPE ::alpaka::core::detail::ScopeLogStdOut const scopeLogStdOut(__func__)
#else
#    define ALPAKA_DEBUG_FULL_LOG_SCOPE
#endif

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#    if BOOST_COMP_GNUC || BOOST_COMP_CLANG
#        define ALPAKA_DEBUG_BREAK ::__builtin_trap()
#    elif BOOST_COMP_MSVC
#        define ALPAKA_DEBUG_BREAK ::__debugbreak()
#    else
#        define ALPAKA_DEBUG_BREAK
#    endif
#else
#    define ALPAKA_DEBUG_BREAK
#endif
