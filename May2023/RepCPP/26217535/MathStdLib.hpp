

#pragma once

#include <alpaka/core/Decay.hpp>
#include <alpaka/math/Traits.hpp>

namespace alpaka::math
{
class AbsStdLib : public concepts::Implements<ConceptMathAbs, AbsStdLib>
{
};

class AcosStdLib : public concepts::Implements<ConceptMathAcos, AcosStdLib>
{
};

class AcoshStdLib : public concepts::Implements<ConceptMathAcosh, AcoshStdLib>
{
};

class ArgStdLib : public concepts::Implements<ConceptMathArg, ArgStdLib>
{
};

class AsinStdLib : public concepts::Implements<ConceptMathAsin, AsinStdLib>
{
};

class AsinhStdLib : public concepts::Implements<ConceptMathAsinh, AsinhStdLib>
{
};

class AtanStdLib : public concepts::Implements<ConceptMathAtan, AtanStdLib>
{
};

class AtanhStdLib : public concepts::Implements<ConceptMathAtanh, AtanhStdLib>
{
};

class Atan2StdLib : public concepts::Implements<ConceptMathAtan2, Atan2StdLib>
{
};

class CbrtStdLib : public concepts::Implements<ConceptMathCbrt, CbrtStdLib>
{
};

class CeilStdLib : public concepts::Implements<ConceptMathCeil, CeilStdLib>
{
};

class ConjStdLib : public concepts::Implements<ConceptMathConj, ConjStdLib>
{
};

class CosStdLib : public concepts::Implements<ConceptMathCos, CosStdLib>
{
};

class CoshStdLib : public concepts::Implements<ConceptMathCosh, CoshStdLib>
{
};

class ErfStdLib : public concepts::Implements<ConceptMathErf, ErfStdLib>
{
};

class ExpStdLib : public concepts::Implements<ConceptMathExp, ExpStdLib>
{
};

class FloorStdLib : public concepts::Implements<ConceptMathFloor, FloorStdLib>
{
};

class FmodStdLib : public concepts::Implements<ConceptMathFmod, FmodStdLib>
{
};

class IsfiniteStdLib : public concepts::Implements<ConceptMathIsfinite, IsfiniteStdLib>
{
};

class IsinfStdLib : public concepts::Implements<ConceptMathIsinf, IsinfStdLib>
{
};

class IsnanStdLib : public concepts::Implements<ConceptMathIsnan, IsnanStdLib>
{
};

class LogStdLib : public concepts::Implements<ConceptMathLog, LogStdLib>
{
};

class MaxStdLib : public concepts::Implements<ConceptMathMax, MaxStdLib>
{
};

class MinStdLib : public concepts::Implements<ConceptMathMin, MinStdLib>
{
};

class PowStdLib : public concepts::Implements<ConceptMathPow, PowStdLib>
{
};

class RemainderStdLib : public concepts::Implements<ConceptMathRemainder, RemainderStdLib>
{
};

class RoundStdLib : public concepts::Implements<ConceptMathRound, RoundStdLib>
{
};

class RsqrtStdLib : public concepts::Implements<ConceptMathRsqrt, RsqrtStdLib>
{
};

class SinStdLib : public concepts::Implements<ConceptMathSin, SinStdLib>
{
};

class SinhStdLib : public concepts::Implements<ConceptMathSinh, SinhStdLib>
{
};

class SinCosStdLib : public concepts::Implements<ConceptMathSinCos, SinCosStdLib>
{
};

class SqrtStdLib : public concepts::Implements<ConceptMathSqrt, SqrtStdLib>
{
};

class TanStdLib : public concepts::Implements<ConceptMathTan, TanStdLib>
{
};

class TanhStdLib : public concepts::Implements<ConceptMathTanh, TanhStdLib>
{
};

class TruncStdLib : public concepts::Implements<ConceptMathTrunc, TruncStdLib>
{
};

class MathStdLib
: public AbsStdLib
, public AcosStdLib
, public AcoshStdLib
, public ArgStdLib
, public AsinStdLib
, public AsinhStdLib
, public AtanStdLib
, public AtanhStdLib
, public Atan2StdLib
, public CbrtStdLib
, public CeilStdLib
, public ConjStdLib
, public CosStdLib
, public CoshStdLib
, public ErfStdLib
, public ExpStdLib
, public FloorStdLib
, public FmodStdLib
, public LogStdLib
, public MaxStdLib
, public MinStdLib
, public PowStdLib
, public RemainderStdLib
, public RoundStdLib
, public RsqrtStdLib
, public SinStdLib
, public SinhStdLib
, public SinCosStdLib
, public SqrtStdLib
, public TanStdLib
, public TanhStdLib
, public TruncStdLib
, public IsnanStdLib
, public IsinfStdLib
, public IsfiniteStdLib
{
};

namespace trait
{
template<typename Tx, typename Ty>
struct Max<MaxStdLib, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
{
ALPAKA_FN_HOST auto operator()(MaxStdLib const& , Tx const& x, Ty const& y)
{
using std::fmax;
using std::max;

if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
return max(x, y);
else if constexpr(
is_decayed_v<
Tx,
float> || is_decayed_v<Ty, float> || is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
return fmax(x, y);
else
static_assert(!sizeof(Tx), "Unsupported data type");

ALPAKA_UNREACHABLE(std::common_type_t<Tx, Ty>{});
}
};

template<typename Tx, typename Ty>
struct Min<MinStdLib, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
{
ALPAKA_FN_HOST auto operator()(MinStdLib const& , Tx const& x, Ty const& y)
{
using std::fmin;
using std::min;

if constexpr(std::is_integral_v<Tx> && std::is_integral_v<Ty>)
return min(x, y);
else if constexpr(
is_decayed_v<
Tx,
float> || is_decayed_v<Ty, float> || is_decayed_v<Tx, double> || is_decayed_v<Ty, double>)
return fmin(x, y);
else
static_assert(!sizeof(Tx), "Unsupported data type");

ALPAKA_UNREACHABLE(std::common_type_t<Tx, Ty>{});
}
};
} 

} 
