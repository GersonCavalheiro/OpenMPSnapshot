

#pragma once

#include <cstdint> 

namespace rawspeed {

namespace ieee_754_2008 {



template <int StorageWidth_, int FractionWidth_, int ExponentWidth_>
struct BinaryN {
static constexpr uint32_t StorageWidth = StorageWidth_;


static constexpr uint32_t FractionWidth = FractionWidth_;
static constexpr uint32_t ExponentWidth = ExponentWidth_;
static_assert(FractionWidth + ExponentWidth + 1 == StorageWidth);

static constexpr uint32_t Precision = FractionWidth + 1;

static constexpr uint32_t ExponentMax = (1 << (ExponentWidth - 1)) - 1;

static constexpr int32_t Bias = ExponentMax;

static constexpr uint32_t ExponentPos = FractionWidth;
static constexpr uint32_t SignBitPos = StorageWidth - 1;
};

struct Binary16 : public BinaryN<16, 10,
5> {
static_assert(Precision == 11);
static_assert(ExponentMax == 15);
static_assert(ExponentPos == 10);
static_assert(SignBitPos == 15);
};

struct Binary24 : public BinaryN<24, 16,
7> {
static_assert(Precision == 17);
static_assert(ExponentMax == 63);
static_assert(ExponentPos == 16);
static_assert(SignBitPos == 23);
};

struct Binary32 : public BinaryN<32, 23,
8> {
static_assert(Precision == 24);
static_assert(ExponentMax == 127);
static_assert(ExponentPos == 23);
static_assert(SignBitPos == 31);
};


} 

template <typename NarrowType, typename WideType>
inline uint32_t extendBinaryFloatingPoint(uint32_t narrow) {
uint32_t sign = (narrow >> NarrowType::SignBitPos) & 1;
uint32_t narrow_exponent = (narrow >> NarrowType::ExponentPos) &
((1 << NarrowType::ExponentWidth) - 1);
uint32_t narrow_fraction = narrow & ((1 << NarrowType::FractionWidth) - 1);

uint32_t wide_exponent =
static_cast<int32_t>(narrow_exponent) - NarrowType::Bias + WideType::Bias;
uint32_t wide_fraction =
narrow_fraction << (WideType::FractionWidth - NarrowType::FractionWidth);

if (narrow_exponent == ((1 << NarrowType::ExponentWidth) - 1)) {
wide_exponent = ((1 << WideType::ExponentWidth) - 1);
} else if (narrow_exponent == 0) {
if (narrow_fraction == 0) {
wide_exponent = 0;
wide_fraction = 0;
} else {
wide_exponent = 1 - NarrowType::Bias + WideType::Bias;
while (!(wide_fraction & (1 << WideType::FractionWidth))) {
wide_exponent -= 1;
wide_fraction <<= 1;
}
wide_fraction &= ((1 << WideType::FractionWidth) - 1);
}
}
return (sign << WideType::SignBitPos) |
(wide_exponent << WideType::ExponentPos) | wide_fraction;
}

inline uint32_t fp16ToFloat(uint16_t fp16) {
return extendBinaryFloatingPoint<ieee_754_2008::Binary16,
ieee_754_2008::Binary32>(fp16);
}

inline uint32_t fp24ToFloat(uint32_t fp24) {
return extendBinaryFloatingPoint<ieee_754_2008::Binary24,
ieee_754_2008::Binary32>(fp24);
}

} 
