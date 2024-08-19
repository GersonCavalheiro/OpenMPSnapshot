

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/math/Complex.hpp>
#    include <alpaka/math/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <type_traits>

namespace alpaka::math
{
class AbsGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAbs, AbsGenericSycl>
{
};

class AcosGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAcos, AcosGenericSycl>
{
};

class AcoshGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAcosh, AcoshGenericSycl>
{
};

class ArgGenericSycl : public concepts::Implements<alpaka::math::ConceptMathArg, ArgGenericSycl>
{
};

class AsinGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAsin, AsinGenericSycl>
{
};

class AsinhGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAsinh, AsinhGenericSycl>
{
};

class AtanGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAtan, AtanGenericSycl>
{
};

class AtanhGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAtanh, AtanhGenericSycl>
{
};

class Atan2GenericSycl : public concepts::Implements<alpaka::math::ConceptMathAtan2, Atan2GenericSycl>
{
};

class CbrtGenericSycl : public concepts::Implements<alpaka::math::ConceptMathCbrt, CbrtGenericSycl>
{
};

class CeilGenericSycl : public concepts::Implements<alpaka::math::ConceptMathCeil, CeilGenericSycl>
{
};

class ConjGenericSycl : public concepts::Implements<alpaka::math::ConceptMathConj, ConjGenericSycl>
{
};

class CosGenericSycl : public concepts::Implements<alpaka::math::ConceptMathCos, CosGenericSycl>
{
};

class CoshGenericSycl : public concepts::Implements<alpaka::math::ConceptMathCosh, CoshGenericSycl>
{
};

class ErfGenericSycl : public concepts::Implements<alpaka::math::ConceptMathErf, ErfGenericSycl>
{
};

class ExpGenericSycl : public concepts::Implements<alpaka::math::ConceptMathExp, ExpGenericSycl>
{
};

class FloorGenericSycl : public concepts::Implements<alpaka::math::ConceptMathFloor, FloorGenericSycl>
{
};

class FmodGenericSycl : public concepts::Implements<alpaka::math::ConceptMathFmod, FmodGenericSycl>
{
};

class IsfiniteGenericSycl : public concepts::Implements<alpaka::math::ConceptMathIsfinite, IsfiniteGenericSycl>
{
};

class IsinfGenericSycl : public concepts::Implements<alpaka::math::ConceptMathIsinf, IsinfGenericSycl>
{
};

class IsnanGenericSycl : public concepts::Implements<alpaka::math::ConceptMathIsnan, IsnanGenericSycl>
{
};

class LogGenericSycl : public concepts::Implements<alpaka::math::ConceptMathLog, LogGenericSycl>
{
};

class MaxGenericSycl : public concepts::Implements<alpaka::math::ConceptMathMax, MaxGenericSycl>
{
};

class MinGenericSycl : public concepts::Implements<alpaka::math::ConceptMathMin, MinGenericSycl>
{
};

class PowGenericSycl : public concepts::Implements<alpaka::math::ConceptMathPow, PowGenericSycl>
{
};

class RemainderGenericSycl : public concepts::Implements<alpaka::math::ConceptMathRemainder, RemainderGenericSycl>
{
};

class RoundGenericSycl : public concepts::Implements<alpaka::math::ConceptMathRound, RoundGenericSycl>
{
};

class RsqrtGenericSycl : public concepts::Implements<alpaka::math::ConceptMathRsqrt, RsqrtGenericSycl>
{
};

class SinGenericSycl : public concepts::Implements<alpaka::math::ConceptMathSin, SinGenericSycl>
{
};

class SinhGenericSycl : public concepts::Implements<alpaka::math::ConceptMathSinh, SinhGenericSycl>
{
};

class SinCosGenericSycl : public concepts::Implements<alpaka::math::ConceptMathSinCos, SinCosGenericSycl>
{
};

class SqrtGenericSycl : public concepts::Implements<alpaka::math::ConceptMathSqrt, SqrtGenericSycl>
{
};

class TanGenericSycl : public concepts::Implements<alpaka::math::ConceptMathTan, TanGenericSycl>
{
};

class TanhGenericSycl : public concepts::Implements<alpaka::math::ConceptMathTanh, TanhGenericSycl>
{
};

class TruncGenericSycl : public concepts::Implements<alpaka::math::ConceptMathTrunc, TruncGenericSycl>
{
};

class MathGenericSycl
: public AbsGenericSycl
, public AcosGenericSycl
, public AcoshGenericSycl
, public ArgGenericSycl
, public AsinGenericSycl
, public AsinhGenericSycl
, public AtanGenericSycl
, public AtanhGenericSycl
, public Atan2GenericSycl
, public CbrtGenericSycl
, public CeilGenericSycl
, public ConjGenericSycl
, public CosGenericSycl
, public CoshGenericSycl
, public ErfGenericSycl
, public ExpGenericSycl
, public FloorGenericSycl
, public FmodGenericSycl
, public IsfiniteGenericSycl
, public IsinfGenericSycl
, public IsnanGenericSycl
, public LogGenericSycl
, public MaxGenericSycl
, public MinGenericSycl
, public PowGenericSycl
, public RemainderGenericSycl
, public RoundGenericSycl
, public RsqrtGenericSycl
, public SinGenericSycl
, public SinhGenericSycl
, public SinCosGenericSycl
, public SqrtGenericSycl
, public TanGenericSycl
, public TanhGenericSycl
, public TruncGenericSycl
{
};
} 

namespace alpaka::math::trait
{
template<typename TArg>
struct Abs<math::AbsGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
{
auto operator()(math::AbsGenericSycl const&, TArg const& arg)
{
if constexpr(std::is_integral_v<TArg>)
return sycl::abs(arg);
else if constexpr(std::is_floating_point_v<TArg>)
return sycl::fabs(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");
}
};

template<typename TArg>
struct Acos<math::AcosGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::AcosGenericSycl const&, TArg const& arg)
{
return sycl::acos(arg);
}
};

template<typename TArg>
struct Acosh<math::AcoshGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::AcoshGenericSycl const&, TArg const& arg)
{
return sycl::acosh(arg);
}
};

template<typename TArgument>
struct Arg<math::ArgGenericSycl, TArgument, std::enable_if_t<std::is_arithmetic_v<TArgument>>>
{
auto operator()(math::ArgGenericSycl const&, TArgument const& argument)
{
if constexpr(std::is_integral_v<TArgument>)
return sycl::atan2(0.0, static_cast<double>(argument));
else if constexpr(std::is_floating_point_v<TArgument>)
return sycl::atan2(TArgument{0.0}, argument);
else
static_assert(!sizeof(TArgument), "Unsupported data type");
}
};

template<typename TArg>
struct Asin<math::AsinGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::AsinGenericSycl const&, TArg const& arg)
{
return sycl::asin(arg);
}
};

template<typename TArg>
struct Asinh<math::AsinhGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::AsinhGenericSycl const&, TArg const& arg)
{
return sycl::asinh(arg);
}
};

template<typename TArg>
struct Atan<math::AtanGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::AtanGenericSycl const&, TArg const& arg)
{
return sycl::atan(arg);
}
};

template<typename TArg>
struct Atanh<math::AtanhGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::AtanhGenericSycl const&, TArg const& arg)
{
return sycl::atanh(arg);
}
};

template<typename Ty, typename Tx>
struct Atan2<
math::Atan2GenericSycl,
Ty,
Tx,
std::enable_if_t<std::is_floating_point_v<Ty> && std::is_floating_point_v<Tx>>>
{
auto operator()(math::Atan2GenericSycl const&, Ty const& y, Tx const& x)
{
return sycl::atan2(y, x);
}
};

template<typename TArg>
struct Cbrt<math::CbrtGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
{
auto operator()(math::CbrtGenericSycl const&, TArg const& arg)
{
if constexpr(std::is_integral_v<TArg>)
return sycl::cbrt(static_cast<double>(arg)); 
else if constexpr(std::is_floating_point_v<TArg>)
return sycl::cbrt(arg);
else
static_assert(!sizeof(TArg), "Unsupported data type");
}
};

template<typename TArg>
struct Ceil<math::CeilGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::CeilGenericSycl const&, TArg const& arg)
{
return sycl::ceil(arg);
}
};

template<typename TArg>
struct Conj<math::ConjGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::ConjGenericSycl const&, TArg const& arg)
{
return Complex<TArg>{arg, TArg{0.0}};
}
};

template<typename TArg>
struct Cos<math::CosGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::CosGenericSycl const&, TArg const& arg)
{
return sycl::cos(arg);
}
};

template<typename TArg>
struct Cosh<math::CoshGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::CoshGenericSycl const&, TArg const& arg)
{
return sycl::cosh(arg);
}
};

template<typename TArg>
struct Erf<math::ErfGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::ErfGenericSycl const&, TArg const& arg)
{
return sycl::erf(arg);
}
};

template<typename TArg>
struct Exp<math::ExpGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::ExpGenericSycl const&, TArg const& arg)
{
return sycl::exp(arg);
}
};

template<typename TArg>
struct Floor<math::FloorGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::FloorGenericSycl const&, TArg const& arg)
{
return sycl::floor(arg);
}
};

template<typename Tx, typename Ty>
struct Fmod<
math::FmodGenericSycl,
Tx,
Ty,
std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
{
auto operator()(math::FmodGenericSycl const&, Tx const& x, Ty const& y)
{
return sycl::fmod(x, y);
}
};

template<typename TArg>
struct Isfinite<math::IsfiniteGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::IsfiniteGenericSycl const&, TArg const& arg)
{
return sycl::isfinite(arg);
}
};

template<typename TArg>
struct Isinf<math::IsinfGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::IsinfGenericSycl const&, TArg const& arg)
{
return sycl::isinf(arg);
}
};

template<typename TArg>
struct Isnan<math::IsnanGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::IsnanGenericSycl const&, TArg const& arg)
{
return sycl::isnan(arg);
}
};

template<typename TArg>
struct Log<math::LogGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::LogGenericSycl const&, TArg const& arg)
{
return sycl::log(arg);
}
};

template<typename Tx, typename Ty>
struct Max<math::MaxGenericSycl, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
{
auto operator()(math::MaxGenericSycl const&, Tx const& x, Ty const& y)
{
if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
return sycl::max(x, y);
else if constexpr(std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>)
return sycl::fmax(x, y);
else if constexpr(
(std::is_floating_point_v<Tx> && std::is_integral_v<Ty>)
|| (std::is_integral_v<Tx> && std::is_floating_point_v<Ty>) )
return sycl::fmax(static_cast<double>(x), static_cast<double>(y)); 
else
static_assert(!sizeof(Tx), "Unsupported data type");
}
};

template<typename Tx, typename Ty>
struct Min<math::MinGenericSycl, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
{
auto operator()(math::MinGenericSycl const&, Tx const& x, Ty const& y)
{
if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
return sycl::min(x, y);
else if constexpr(std::is_floating_point_v<Tx> || std::is_floating_point_v<Ty>)
return sycl::fmin(x, y);
else if constexpr(
(std::is_floating_point_v<Tx> && std::is_integral_v<Ty>)
|| (std::is_integral_v<Tx> && std::is_floating_point_v<Ty>) )
return sycl::fmin(static_cast<double>(x), static_cast<double>(y)); 
else
static_assert(!sizeof(Tx), "Unsupported data type");
}
};

template<typename TBase, typename TExp>
struct Pow<
math::PowGenericSycl,
TBase,
TExp,
std::enable_if_t<std::is_floating_point_v<TBase> && std::is_floating_point_v<TExp>>>
{
auto operator()(math::PowGenericSycl const&, TBase const& base, TExp const& exp)
{
return sycl::pow(base, exp);
}
};

template<typename Tx, typename Ty>
struct Remainder<
math::RemainderGenericSycl,
Tx,
Ty,
std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
{
auto operator()(math::RemainderGenericSycl const&, Tx const& x, Ty const& y)
{
return sycl::remainder(x, y);
}
};

template<typename TArg>
struct Round<math::RoundGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::RoundGenericSycl const&, TArg const& arg)
{
return sycl::round(arg);
}
};

template<typename TArg>
struct Lround<math::RoundGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::RoundGenericSycl const&, TArg const& arg)
{
return static_cast<long>(sycl::round(arg));
}
};

template<typename TArg>
struct Llround<math::RoundGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::RoundGenericSycl const&, TArg const& arg)
{
return static_cast<long long>(sycl::round(arg));
}
};

template<typename TArg>
struct Rsqrt<math::RsqrtGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
{
auto operator()(math::RsqrtGenericSycl const&, TArg const& arg)
{
if(std::is_floating_point_v<TArg>)
return sycl::rsqrt(arg);
else if(std::is_integral_v<TArg>)
return sycl::rsqrt(static_cast<double>(arg)); 
else
static_assert(!sizeof(TArg), "Unsupported data type");
}
};

template<typename TArg>
struct Sin<math::SinGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::SinGenericSycl const&, TArg const& arg)
{
return sycl::sin(arg);
}
};

template<typename TArg>
struct Sinh<math::SinhGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::SinhGenericSycl const&, TArg const& arg)
{
return sycl::sinh(arg);
}
};

template<typename TArg>
struct SinCos<math::SinCosGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::SinCosGenericSycl const&, TArg const& arg, TArg& result_sin, TArg& result_cos) -> void
{
result_sin = sycl::sincos(arg, &result_cos);
}
};

template<typename TArg>
struct Sqrt<math::SqrtGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
{
auto operator()(math::SqrtGenericSycl const&, TArg const& arg)
{
if constexpr(std::is_floating_point_v<TArg>)
return sycl::sqrt(arg);
else if constexpr(std::is_integral_v<TArg>)
return sycl::sqrt(static_cast<double>(arg)); 
}
};

template<typename TArg>
struct Tan<math::TanGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::TanGenericSycl const&, TArg const& arg)
{
return sycl::tan(arg);
}
};

template<typename TArg>
struct Tanh<math::TanhGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::TanhGenericSycl const&, TArg const& arg)
{
return sycl::tanh(arg);
}
};

template<typename TArg>
struct Trunc<math::TruncGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
{
auto operator()(math::TruncGenericSycl const&, TArg const& arg)
{
return sycl::trunc(arg);
}
};
} 

#endif
