

#pragma once

#include "rawspeedconfig.h" 
#include <algorithm>        
#include <vector>           

namespace rawspeed {

class CameraSensorInfo final {
public:
CameraSensorInfo(int black_level, int white_level, int min_iso, int max_iso,
std::vector<int> black_separate);
[[nodiscard]] bool RAWSPEED_READONLY isIsoWithin(int iso) const;
[[nodiscard]] bool RAWSPEED_READONLY isDefault() const;
int mBlackLevel;
int mWhiteLevel;
int mMinIso;
int mMaxIso;
std::vector<int> mBlackLevelSeparate;
};

} 
