

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

using XTransPhase = iPoint2D;

inline iPoint2D getTranslationalOffset(XTransPhase src, XTransPhase tgt) {
iPoint2D off = tgt - src;
return {std::abs(off.x), std::abs(off.y)};
}

template <typename T>
inline std::array<T, 6 * 6> applyPhaseShift(std::array<T, 6 * 6> srcData,
XTransPhase srcPhase,
XTransPhase tgtPhase) {
const iPoint2D coordOffset = getTranslationalOffset(srcPhase, tgtPhase);
assert(coordOffset >= iPoint2D(0, 0) && "Offset is non-negative.");
const Array2DRef<const T> src(srcData.data(), 6, 6);

std::array<T, 6 * 6> tgtData;
const Array2DRef<T> tgt(tgtData.data(), 6, 6);
for (int row = 0; row < tgt.height; ++row) {
for (int col = 0; col < tgt.width; ++col) {
tgt(row, col) = src((coordOffset.y + row) % 6, (coordOffset.x + col) % 6);
}
}

return tgtData;
}

inline std::array<CFAColor, 6 * 6> getAsCFAColors(XTransPhase p) {
const XTransPhase basePhase(0, 0);
const std::array<CFAColor, 6 * 6> basePat = {
CFAColor::GREEN, CFAColor::GREEN, CFAColor::RED,   CFAColor::GREEN,
CFAColor::GREEN, CFAColor::BLUE,  CFAColor::GREEN, CFAColor::GREEN,
CFAColor::BLUE,  CFAColor::GREEN, CFAColor::GREEN, CFAColor::RED,
CFAColor::BLUE,  CFAColor::RED,   CFAColor::GREEN, CFAColor::RED,
CFAColor::BLUE,  CFAColor::GREEN, CFAColor::GREEN, CFAColor::GREEN,
CFAColor::BLUE,  CFAColor::GREEN, CFAColor::GREEN, CFAColor::RED,
CFAColor::GREEN, CFAColor::GREEN, CFAColor::RED,   CFAColor::GREEN,
CFAColor::GREEN, CFAColor::BLUE,  CFAColor::RED,   CFAColor::BLUE,
CFAColor::GREEN, CFAColor::BLUE,  CFAColor::RED,   CFAColor::GREEN};
return applyPhaseShift(basePat, basePhase, p);
}

inline std::optional<XTransPhase>
getAsXTransPhase(const ColorFilterArray& CFA) {
if (CFA.getSize() != iPoint2D(6, 6))
return {};

std::array<CFAColor, 6 * 6> patData;
const Array2DRef<CFAColor> pat(patData.data(), 6, 6);
for (int row = 0; row < pat.height; ++row) {
for (int col = 0; col < pat.width; ++col) {
pat(row, col) = CFA.getColorAt(col, row);
}
}

iPoint2D off;
for (off.y = 0; off.y < pat.height; ++off.y) {
for (off.x = 0; off.x < pat.width; ++off.x) {
if (getAsCFAColors(off) == patData)
return off;
}
}

return {};
}

} 
