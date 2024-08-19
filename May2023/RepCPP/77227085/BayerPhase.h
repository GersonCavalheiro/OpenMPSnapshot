

#pragma once

#include "adt/Array2DRef.h"            
#include "adt/Point.h"                 
#include "metadata/ColorFilterArray.h" 
#include <algorithm>                   
#include <array>                       
#include <cassert>                     
#include <cmath>                       
#include <cstdlib>                     
#include <iterator>                    
#include <optional>                    
#include <utility>                     

namespace rawspeed {

enum class BayerPhase : int {
RGGB = 0b00, 
GRBG = 0b01, 
GBRG = 0b10, 
BGGR = 0b11, 

};

inline iPoint2D getTranslationalOffset(BayerPhase src, BayerPhase tgt) {
auto getCanonicalPosition = [](BayerPhase p) -> iPoint2D {
auto i = static_cast<unsigned>(p);
return {(i & 0b01) != 0, (i & 0b10) != 0};
};

iPoint2D off = getCanonicalPosition(tgt) - getCanonicalPosition(src);
return {std::abs(off.x), std::abs(off.y)};
}

template <typename T>
inline std::array<T, 4> applyPhaseShift(std::array<T, 4> srcData,
BayerPhase srcPhase,
BayerPhase tgtPhase) {
const iPoint2D coordOffset = getTranslationalOffset(srcPhase, tgtPhase);
assert(coordOffset >= iPoint2D(0, 0) && "Offset is non-negative.");
const Array2DRef<const T> src(srcData.data(), 2, 2);

std::array<T, 4> tgtData;
const Array2DRef<T> tgt(tgtData.data(), 2, 2);
for (int row = 0; row < tgt.height; ++row) {
for (int col = 0; col < tgt.width; ++col) {
tgt(row, col) = src((coordOffset.y + row) % 2, (coordOffset.x + col) % 2);
}
}

return tgtData;
}

inline std::array<CFAColor, 4> getAsCFAColors(BayerPhase p) {
const BayerPhase basePhase = BayerPhase::RGGB;
const std::array<CFAColor, 4> basePat = {CFAColor::RED, CFAColor::GREEN,
CFAColor::GREEN, CFAColor::BLUE};
return applyPhaseShift(basePat, basePhase, p);
}

template <typename T>
inline std::array<T, 4> applyStablePhaseShift(std::array<T, 4> srcData,
BayerPhase srcPhase,
BayerPhase tgtPhase) {
std::array<T, 4> tgtData = applyPhaseShift(srcData, srcPhase, tgtPhase);

if (!getTranslationalOffset(srcPhase, tgtPhase).y)
return tgtData;

auto is_green = [](const CFAColor& c) { return c == CFAColor::GREEN; };

const std::array<CFAColor, 4> tgtColors = getAsCFAColors(tgtPhase);
int green0Idx =
std::distance(tgtColors.begin(),
std::find_if(tgtColors.begin(), tgtColors.end(), is_green));
int green1Idx = std::distance(std::find_if(tgtColors.rbegin(),
tgtColors.rend(), is_green),
tgtColors.rend()) -
1;

std::swap(tgtData[green0Idx], tgtData[green1Idx]);

return tgtData;
}

inline std::optional<BayerPhase> getAsBayerPhase(const ColorFilterArray& CFA) {
if (CFA.getSize() != iPoint2D(2, 2))
return {};

std::array<CFAColor, 4> patData;
const Array2DRef<CFAColor> pat(patData.data(), 2, 2);
for (int row = 0; row < pat.height; ++row) {
for (int col = 0; col < pat.width; ++col) {
pat(row, col) = CFA.getColorAt(col, row);
}
}

for (auto i = (int)BayerPhase::RGGB; i <= (int)BayerPhase::BGGR; ++i) {
if (auto p = static_cast<BayerPhase>(i); getAsCFAColors(p) == patData)
return p;
}

return {};
}

} 
