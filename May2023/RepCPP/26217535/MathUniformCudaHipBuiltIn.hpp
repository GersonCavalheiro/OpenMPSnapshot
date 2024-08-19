

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/CudaHipCommon.hpp>
#include <alpaka/core/Decay.hpp>
#include <alpaka/core/UniformCudaHip.hpp>
#include <alpaka/core/Unreachable.hpp>
#include <alpaka/math/Complex.hpp>
#include <alpaka/math/Traits.hpp>

#include <type_traits>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

namespace alpaka::math
{
class AbsUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAbs, AbsUniformCudaHipBuiltIn>
{
};

class AcosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAcos, AcosUniformCudaHipBuiltIn>
{
};

class AcoshUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAcosh, AcoshUniformCudaHipBuiltIn>
{
};

class ArgUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathArg, ArgUniformCudaHipBuiltIn>
{
};

class AsinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAsin, AsinUniformCudaHipBuiltIn>
{
};

class AsinhUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAsinh, AsinhUniformCudaHipBuiltIn>
{
};

class AtanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtan, AtanUniformCudaHipBuiltIn>
{
};

class AtanhUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtanh, AtanhUniformCudaHipBuiltIn>
{
};

class Atan2UniformCudaHipBuiltIn : public concepts::Implements<ConceptMathAtan2, Atan2UniformCudaHipBuiltIn>
{
};

class CbrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCbrt, CbrtUniformCudaHipBuiltIn>
{
};

class CeilUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCeil, CeilUniformCudaHipBuiltIn>
{
};

class ConjUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathConj, ConjUniformCudaHipBuiltIn>
{
};

class CosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCos, CosUniformCudaHipBuiltIn>
{
};

class CoshUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathCosh, CoshUniformCudaHipBuiltIn>
{
};

class ErfUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathErf, ErfUniformCudaHipBuiltIn>
{
};

class ExpUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathExp, ExpUniformCudaHipBuiltIn>
{
};

class FloorUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathFloor, FloorUniformCudaHipBuiltIn>
{
};

class FmodUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathFmod, FmodUniformCudaHipBuiltIn>
{
};

class IsfiniteUniformCudaHipBuiltIn
: public concepts::Implements<ConceptMathIsfinite, IsfiniteUniformCudaHipBuiltIn>
{
};

class IsinfUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathIsinf, IsinfUniformCudaHipBuiltIn>
{
};

class IsnanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathIsnan, IsnanUniformCudaHipBuiltIn>
{
};

class LogUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathLog, LogUniformCudaHipBuiltIn>
{
};

class MaxUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathMax, MaxUniformCudaHipBuiltIn>
{
};

class MinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathMin, MinUniformCudaHipBuiltIn>
{
};

class PowUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathPow, PowUniformCudaHipBuiltIn>
{
};

class RemainderUniformCudaHipBuiltIn
: public concepts::Implements<ConceptMathRemainder, RemainderUniformCudaHipBuiltIn>
{
};

class RoundUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathRound, RoundUniformCudaHipBuiltIn>
{
};

class RsqrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathRsqrt, RsqrtUniformCudaHipBuiltIn>
{
};

class SinUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSin, SinUniformCudaHipBuiltIn>
{
};

class SinhUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSinh, SinhUniformCudaHipBuiltIn>
{
};

class SinCosUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSinCos, SinCosUniformCudaHipBuiltIn>
{
};

class SqrtUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathSqrt, SqrtUniformCudaHipBuiltIn>
{
};

class TanUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathTan, TanUniformCudaHipBuiltIn>
{
};

class TanhUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathTanh, TanhUniformCudaHipBuiltIn>
{
};

class TruncUniformCudaHipBuiltIn : public concepts::Implements<ConceptMathTrunc, TruncUniformCudaHipBuiltIn>
{
};

class MathUniformCudaHipBuiltIn
: public AbsUniformCudaHipBuiltIn
, public AcosUniformCudaHipBuiltIn
, public AcoshUniformCudaHipBuiltIn
, public ArgUniformCudaHipBuiltIn
, public AsinUniformCudaHipBuiltIn
, public AsinhUniformCudaHipBuiltIn
, public AtanUniformCudaHipBuiltIn
, public AtanhUniformCudaHipBuiltIn
, public Atan2UniformCudaHipBuiltIn
, public CbrtUniformCudaHipBuiltIn
, public CeilUniformCudaHipBuiltIn
, public ConjUniformCudaHipBuiltIn
, public CosUniformCudaHipBuiltIn
, public CoshUniformCudaHipBuiltIn
, public ErfUniformCudaHipBuiltIn
, public ExpUniformCudaHipBuiltIn
, public FloorUniformCudaHipBuiltIn
, public FmodUniformCudaHipBuiltIn
, public LogUniformCudaHipBuiltIn
, public MaxUniformCudaHipBuiltIn
, public MinUniformCudaHipBuiltIn
, public PowUniformCudaHipBuiltIn
, public RemainderUniformCudaHipBuiltIn
, public RoundUniformCudaHipBuiltIn
, public RsqrtUniformCudaHipBuiltIn
, public SinUniformCudaHipBuiltIn
, public SinhUniformCudaHipBuiltIn
, public SinCosUniformCudaHipBuiltIn
, public SqrtUniformCudaHipBuiltIn
, public TanUniformCudaHipBuiltIn
, public TanhUniformCudaHipBuiltIn
, public TruncUniformCudaHipBuiltIn
, public IsnanUniformCudaHipBuiltIn
, public IsinfUniformCudaHipBuiltIn
, public IsfiniteUniformCudaHipBuiltIn
{
};

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)
#            include <cuda_runtime.h>
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__)
#            include <hip/math_functions.h>
#        endif

namespace trait
{
template<typename TArg>
struct Abs<AbsUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_signed_v<TArg>>>
{
__host__ __device__ auto operator()(AbsUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::fabsf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::fabs(arg);
else if constexpr(is_decayed_v<TArg, int>)
return ::abs(arg);
else if constexpr(is_decayed_v<TArg, long int>)
return ::labs(arg);
else if constexpr(is_decayed_v<TArg, long long int>)
return ::llabs(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Abs<AbsUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
return sqrt(ctx, arg.real() * arg.real() + arg.imag() * arg.imag());
}
};

template<typename TArg>
struct Acos<AcosUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(AcosUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::acosf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::acos(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Acos<AcosUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
return Complex<T>{0.0, -1.0} * log(ctx, arg + Complex<T>{0.0, 1.0} * sqrt(ctx, T(1.0) - arg * arg));
}
};

template<typename TArg>
struct Acosh<AcoshUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(AcoshUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::acoshf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::acosh(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Acosh<AcoshUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
return log(ctx, arg + sqrt(ctx, arg - static_cast<T>(1.0)) * sqrt(ctx, arg + static_cast<T>(1.0)));
}
};

template<typename TArgument>
struct Arg<ArgUniformCudaHipBuiltIn, TArgument, std::enable_if_t<std::is_floating_point_v<TArgument>>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, TArgument const& argument)
{
return atan2(ctx, TArgument{0.0}, argument);
}
};

template<typename T>
struct Arg<ArgUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& argument)
{
return atan2(ctx, argument.imag(), argument.real());
}
};

template<typename TArg>
struct Asin<AsinUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(AsinUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::asinf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::asin(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Asin<AsinUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
return Complex<T>{0.0, 1.0} * log(ctx, sqrt(ctx, T(1.0) - arg * arg) - Complex<T>{0.0, 1.0} * arg);
}
};

template<typename TArg>
struct Asinh<AsinhUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(AsinhUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::asinhf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::asinh(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Asinh<AsinhUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
return log(ctx, arg + sqrt(ctx, arg * arg + static_cast<T>(1.0)));
}
};

template<typename TArg>
struct Atan<AtanUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(AtanUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::atanf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::atan(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Atan<AtanUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
return Complex<T>{0.0, -0.5} * log(ctx, (Complex<T>{0.0, 1.0} - arg) / (Complex<T>{0.0, 1.0} + arg));
}
};

template<typename TArg>
struct Atanh<AtanhUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(AtanhUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::atanhf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::atanh(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Atanh<AtanhUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
return static_cast<T>(0.5)
* (log(ctx, static_cast<T>(1.0) + arg) - log(ctx, static_cast<T>(1.0) - arg));
}
};

template<typename Ty, typename Tx>
struct Atan2<
Atan2UniformCudaHipBuiltIn,
Ty,
Tx,
std::enable_if_t<std::is_floating_point_v<Ty> && std::is_floating_point_v<Tx>>>
{
__host__ __device__ auto operator()(
Atan2UniformCudaHipBuiltIn const& ,
Ty const& y,
Tx const& x)
{
if constexpr(is_decayed_v<Ty, float> && is_decayed_v<Tx, float>)
return ::atan2f(y, x);
else if constexpr(is_decayed_v<Ty, double> || is_decayed_v<Tx, double>)
return ::atan2(y, x);
else
static_assert(!sizeof(Ty), "Unsupported data type");

ALPAKA_UNREACHABLE(Ty{});
}
};

template<typename TArg>
struct Cbrt<CbrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
{
__host__ __device__ auto operator()(CbrtUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::cbrtf(arg);
else if constexpr(is_decayed_v<TArg, double> || std::is_integral_v<TArg>)
return ::cbrt(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename TArg>
struct Ceil<CeilUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(CeilUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::ceilf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::ceil(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename TArg>
struct Conj<ConjUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(ConjUniformCudaHipBuiltIn const& , TArg const& arg)
{
return Complex<TArg>{arg, TArg{0.0}};
}
};

template<typename T>
struct Conj<ConjUniformCudaHipBuiltIn, Complex<T>>
{
__host__ __device__ auto operator()(ConjUniformCudaHipBuiltIn const& , Complex<T> const& arg)
{
return Complex<T>{arg.real(), -arg.imag()};
}
};

template<typename TArg>
struct Cos<CosUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(CosUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::cosf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::cos(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Cos<CosUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
return T(0.5) * (exp(ctx, Complex<T>{0.0, 1.0} * arg) + exp(ctx, Complex<T>{0.0, -1.0} * arg));
}
};

template<typename TArg>
struct Cosh<CoshUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(CoshUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::coshf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::cosh(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Cosh<CoshUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
return T(0.5) * (exp(ctx, arg) + exp(ctx, static_cast<T>(-1.0) * arg));
}
};

template<typename TArg>
struct Erf<ErfUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(ErfUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::erff(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::erf(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename TArg>
struct Exp<ExpUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(ExpUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::expf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::exp(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Exp<ExpUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
auto re = T{}, im = T{};
sincos(ctx, arg.imag(), im, re);
return exp(ctx, arg.real()) * Complex<T>{re, im};
}
};

template<typename TArg>
struct Floor<FloorUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(FloorUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::floorf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::floor(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename Tx, typename Ty>
struct Fmod<
FmodUniformCudaHipBuiltIn,
Tx,
Ty,
std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
{
__host__ __device__ auto operator()(
FmodUniformCudaHipBuiltIn const& ,
Tx const& x,
Ty const& y)
{
if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
return ::fmodf(x, y);
else if constexpr(is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
return ::fmod(x, y);
else
static_assert(!sizeof(Tx), "Unsupported data type");

using Ret [[maybe_unused]]
= std::conditional_t<is_decayed_v<Tx, float> && is_decayed_v<Ty, float>, float, double>;
ALPAKA_UNREACHABLE(Ret{});
}
};

template<typename TArg>
struct Isfinite<IsfiniteUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(IsfiniteUniformCudaHipBuiltIn const& , TArg const& arg)
{
return ::isfinite(arg);
}
};

template<typename TArg>
struct Isinf<IsinfUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(IsinfUniformCudaHipBuiltIn const& , TArg const& arg)
{
return ::isinf(arg);
}
};

template<typename TArg>
struct Isnan<IsnanUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(IsnanUniformCudaHipBuiltIn const& , TArg const& arg)
{
return ::isnan(arg);
}
};

template<typename TArg>
struct Log<LogUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(LogUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::logf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::log(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Log<LogUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& argument)
{
return log(ctx, abs(ctx, argument)) + Complex<T>{0.0, 1.0} * arg(ctx, argument);
}
};

template<typename Tx, typename Ty>
struct Max<
MaxUniformCudaHipBuiltIn,
Tx,
Ty,
std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
{
__host__ __device__ auto operator()(
MaxUniformCudaHipBuiltIn const& ,
Tx const& x,
Ty const& y)
{
if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
return ::max(x, y);
else if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
return ::fmaxf(x, y);
else if constexpr(
is_decayed_v<
Tx,
double> || is_decayed_v<Ty, double> || (is_decayed_v<Tx, float> && std::is_integral_v<Ty>)
|| (std::is_integral_v<Tx> && is_decayed_v<Ty, float>) )
return ::fmax(x, y);
else
static_assert(!sizeof(Tx), "Unsupported data type");

using Ret [[maybe_unused]] = std::conditional_t<
std::is_integral_v<Tx> && std::is_integral_v<Ty>,
decltype(::max(x, y)),
std::conditional_t<is_decayed_v<Tx, float> && is_decayed_v<Ty, float>, float, double>>;
ALPAKA_UNREACHABLE(Ret{});
}
};

template<typename Tx, typename Ty>
struct Min<
MinUniformCudaHipBuiltIn,
Tx,
Ty,
std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
{
__host__ __device__ auto operator()(
MinUniformCudaHipBuiltIn const& ,
Tx const& x,
Ty const& y)
{
if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
return ::min(x, y);
else if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
return ::fminf(x, y);
else if constexpr(
is_decayed_v<
Tx,
double> || is_decayed_v<Ty, double> || (is_decayed_v<Tx, float> && std::is_integral_v<Ty>)
|| (std::is_integral_v<Tx> && is_decayed_v<Ty, float>) )
return ::fmin(x, y);
else
static_assert(!sizeof(Tx), "Unsupported data type");

using Ret [[maybe_unused]] = std::conditional_t<
std::is_integral_v<Tx> && std::is_integral_v<Ty>,
decltype(::min(x, y)),
std::conditional_t<is_decayed_v<Tx, float> && is_decayed_v<Ty, float>, float, double>>;
ALPAKA_UNREACHABLE(Ret{});
}
};

template<typename TBase, typename TExp>
struct Pow<
PowUniformCudaHipBuiltIn,
TBase,
TExp,
std::enable_if_t<std::is_floating_point_v<TBase> && std::is_floating_point_v<TExp>>>
{
__host__ __device__ auto operator()(
PowUniformCudaHipBuiltIn const& ,
TBase const& base,
TExp const& exp)
{
if constexpr(is_decayed_v<TBase, float> && is_decayed_v<TExp, float>)
return ::powf(base, exp);
else if constexpr(is_decayed_v<TBase, double> || is_decayed_v<TExp, double>)
return ::pow(static_cast<double>(base), static_cast<double>(exp));
else
static_assert(!sizeof(TBase), "Unsupported data type");

using Ret [[maybe_unused]]
= std::conditional_t<is_decayed_v<TBase, float> && is_decayed_v<TExp, float>, float, double>;
ALPAKA_UNREACHABLE(Ret{});
}
};

template<typename T, typename U>
struct Pow<PowUniformCudaHipBuiltIn, Complex<T>, Complex<U>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& base, Complex<U> const& exponent)
{
using Promoted
= Complex<std::conditional_t<is_decayed_v<T, float> && is_decayed_v<U, float>, float, double>>;
return exp(ctx, Promoted{exponent} * log(ctx, Promoted{base}));
}
};

template<typename T, typename U>
struct Pow<PowUniformCudaHipBuiltIn, Complex<T>, U>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& base, U const& exponent)
{
return pow(ctx, base, Complex<U>{exponent});
}
};

template<typename T, typename U>
struct Pow<PowUniformCudaHipBuiltIn, T, Complex<U>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, T const& base, Complex<U> const& exponent)
{
return pow(ctx, Complex<T>{base}, exponent);
}
};

template<typename Tx, typename Ty>
struct Remainder<
RemainderUniformCudaHipBuiltIn,
Tx,
Ty,
std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
{
__host__ __device__ auto operator()(
RemainderUniformCudaHipBuiltIn const& ,
Tx const& x,
Ty const& y)
{
if constexpr(is_decayed_v<Tx, float> && is_decayed_v<Ty, float>)
return ::remainderf(x, y);
else if constexpr(is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
return ::remainder(x, y);
else
static_assert(!sizeof(Tx), "Unsupported data type");

using Ret [[maybe_unused]]
= std::conditional_t<is_decayed_v<Tx, float> && is_decayed_v<Ty, float>, float, double>;
ALPAKA_UNREACHABLE(Ret{});
}
};


template<typename TArg>
struct Round<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(RoundUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::roundf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::round(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename TArg>
struct Lround<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(RoundUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::lroundf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::lround(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(long{});
}
};

template<typename TArg>
struct Llround<RoundUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(RoundUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::llroundf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::llround(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

using Ret [[maybe_unused]] = long long;
ALPAKA_UNREACHABLE(Ret{});
}
};

template<typename TArg>
struct Rsqrt<RsqrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
{
__host__ __device__ auto operator()(RsqrtUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::rsqrtf(arg);
else if constexpr(is_decayed_v<TArg, double> || std::is_integral_v<TArg>)
return ::rsqrt(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Rsqrt<RsqrtUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
return T{1.0} / sqrt(ctx, arg);
}
};

template<typename TArg>
struct Sin<SinUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(SinUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::sinf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::sin(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Sin<SinUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
return (exp(ctx, Complex<T>{0.0, 1.0} * arg) - exp(ctx, Complex<T>{0.0, -1.0} * arg))
/ Complex<T>{0.0, 2.0};
}
};

template<typename TArg>
struct Sinh<SinhUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(SinhUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::sinhf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::sinh(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Sinh<SinhUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
return (exp(ctx, arg) - exp(ctx, static_cast<T>(-1.0) * arg)) / static_cast<T>(2.0);
}
};

template<typename TArg>
struct SinCos<SinCosUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(
SinCosUniformCudaHipBuiltIn const& ,
TArg const& arg,
TArg& result_sin,
TArg& result_cos) -> void
{
if constexpr(is_decayed_v<TArg, float>)
::sincosf(arg, &result_sin, &result_cos);
else if constexpr(is_decayed_v<TArg, double>)
::sincos(arg, &result_sin, &result_cos);
else
static_assert(!sizeof(TArg), "Unsupported data type");
}
};

template<typename T>
struct SinCos<SinCosUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(
TCtx const& ctx,
Complex<T> const& arg,
Complex<T>& result_sin,
Complex<T>& result_cos) -> void
{
result_sin = sin(ctx, arg);
result_cos = cos(ctx, arg);
}
};

template<typename TArg>
struct Sqrt<SqrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
{
__host__ __device__ auto operator()(SqrtUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::sqrtf(arg);
else if constexpr(is_decayed_v<TArg, double> || std::is_integral_v<TArg>)
return ::sqrt(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Sqrt<SqrtUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& argument)
{
auto const halfArg = T(0.5) * arg(ctx, argument);
auto re = T{}, im = T{};
sincos(ctx, halfArg, im, re);
return sqrt(ctx, abs(ctx, argument)) * Complex<T>(re, im);
}
};

template<typename TArg>
struct Tan<TanUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(TanUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::tanf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::tan(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Tan<TanUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
auto const expValue = exp(ctx, Complex<T>{0.0, 2.0} * arg);
return Complex<T>{0.0, 1.0} * (T{1.0} - expValue) / (T{1.0} + expValue);
}
};

template<typename TArg>
struct Tanh<TanhUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(TanhUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::tanhf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::tanh(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};

template<typename T>
struct Tanh<TanhUniformCudaHipBuiltIn, Complex<T>>
{
template<typename TCtx>
__host__ __device__ auto operator()(TCtx const& ctx, Complex<T> const& arg)
{
return (exp(ctx, arg) - exp(ctx, static_cast<T>(-1.0) * arg))
/ (exp(ctx, arg) + exp(ctx, static_cast<T>(-1.0) * arg));
}
};

template<typename TArg>
struct Trunc<TruncUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
__host__ __device__ auto operator()(TruncUniformCudaHipBuiltIn const& , TArg const& arg)
{
if constexpr(is_decayed_v<TArg, float>)
return ::truncf(arg);
else if constexpr(is_decayed_v<TArg, double>)
return ::trunc(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");

ALPAKA_UNREACHABLE(TArg{});
}
};
} 
#    endif
} 

#endif
