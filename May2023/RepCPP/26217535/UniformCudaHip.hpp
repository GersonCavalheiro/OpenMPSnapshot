

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <array>
#    include <stdexcept>
#    include <string>
#    include <tuple>
#    include <type_traits>

namespace alpaka::uniform_cuda_hip::detail
{
template<typename TApi, bool TThrow>
ALPAKA_FN_HOST inline void rtCheck(
typename TApi::Error_t const& error,
char const* desc,
char const* file,
int const& line) noexcept(!TThrow)
{
if(error != TApi::success)
{
auto const sError = std::string{
std::string(file) + "(" + std::to_string(line) + ") " + std::string(desc) + " : '"
+ TApi::getErrorName(error) + "': '" + std::string(TApi::getErrorString(error)) + "'!"};

if constexpr(!TThrow || ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL)
std::cerr << sError << std::endl;

ALPAKA_DEBUG_BREAK;
std::ignore = TApi::getLastError();

if constexpr(TThrow)
throw std::runtime_error(sError);
}
}

template<typename TApi, bool TThrow, typename... TErrors>
ALPAKA_FN_HOST inline void rtCheckIgnore(
typename TApi::Error_t const& error,
char const* cmd,
char const* file,
int const& line,
TErrors&&... ignoredErrorCodes) noexcept(!TThrow)
{
if(error != TApi::success)
{
std::array<typename TApi::Error_t, sizeof...(ignoredErrorCodes)> const aIgnoredErrorCodes{
{ignoredErrorCodes...}};

if(std::find(std::cbegin(aIgnoredErrorCodes), std::cend(aIgnoredErrorCodes), error)
== std::cend(aIgnoredErrorCodes))
{
rtCheck<TApi, TThrow>(error, ("'" + std::string(cmd) + "' returned error ").c_str(), file, line);
}
else
{
std::ignore = TApi::getLastError();
}
}
}

template<typename TApi, bool TThrow>
ALPAKA_FN_HOST inline void rtCheckLastError(char const* desc, char const* file, int const& line) noexcept(!TThrow)
{
typename TApi::Error_t const error(TApi::getLastError());
rtCheck<TApi, TThrow>(error, desc, file, line);
}
} 

#    if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#        define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(cmd, ...)                                                     \
::alpaka::uniform_cuda_hip::detail::rtCheckLastError<TApi, true>(                                         \
"'" #cmd "' A previous API call (not this one) set the error ",                                       \
__FILE__,                                                                                             \
__LINE__);                                                                                            \
::alpaka::uniform_cuda_hip::detail::rtCheckIgnore<TApi, true>(cmd, #cmd, __FILE__, __LINE__, __VA_ARGS__)
#    else
#        if BOOST_COMP_CLANG
#            pragma clang diagnostic push
#            pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#        endif
#        define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(cmd, ...)                                                     \
::alpaka::uniform_cuda_hip::detail::rtCheckLastError<TApi, true>(                                         \
"'" #cmd "' A previous API call (not this one) set the error ",                                       \
__FILE__,                                                                                             \
__LINE__);                                                                                            \
::alpaka::uniform_cuda_hip::detail::rtCheckIgnore<TApi, true>(cmd, #cmd, __FILE__, __LINE__, ##__VA_ARGS__)
#        if BOOST_COMP_CLANG
#            pragma clang diagnostic pop
#        endif
#    endif

#    define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cmd) ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(cmd)

#    if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#        define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd, ...)                                            \
::alpaka::uniform_cuda_hip::detail::rtCheckLastError<TApi, false>(                                        \
"'" #cmd "' A previous API call (not this one) set the error ",                                       \
__FILE__,                                                                                             \
__LINE__);                                                                                            \
::alpaka::uniform_cuda_hip::detail::rtCheckIgnore<TApi, false>(cmd, #cmd, __FILE__, __LINE__, __VA_ARGS__)
#    else
#        if BOOST_COMP_CLANG
#            pragma clang diagnostic push
#            pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#        endif
#        define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd, ...)                                            \
::alpaka::uniform_cuda_hip::detail::rtCheckLastError<TApi, false>(                                        \
"'" #cmd "' A previous API call (not this one) set the error ",                                       \
__FILE__,                                                                                             \
__LINE__);                                                                                            \
::alpaka::uniform_cuda_hip::detail::rtCheckIgnore<TApi, false>(                                           \
cmd,                                                                                                  \
#cmd,                                                                                                 \
__FILE__,                                                                                             \
__LINE__,                                                                                             \
##__VA_ARGS__)
#        if BOOST_COMP_CLANG
#            pragma clang diagnostic pop
#        endif
#    endif

#    define ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(cmd) ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE_NOEXCEPT(cmd)
#endif
