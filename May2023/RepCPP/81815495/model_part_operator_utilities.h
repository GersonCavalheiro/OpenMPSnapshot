
#pragma once

#include <vector>
#include <set>



namespace Kratos {


struct ModelPartUnionOperator {
template <class TCheckType>
static bool IsValid(
const TCheckType& rEntity,
const std::vector<std::set<TCheckType>>& mUnionSets)
{
bool is_valid = false;

for (const auto& r_set : mUnionSets) {
if (r_set.find(rEntity) != r_set.end()) {
is_valid = true;
break;
}
}

return is_valid;
}
};

struct ModelPartSubstractionOperator {
template <class TCheckType>
static bool IsValid(
const TCheckType& rEntity,
const std::vector<std::set<TCheckType>>& mSubstractionSets)
{
bool is_valid = true;

for (const auto& r_set : mSubstractionSets) {
if (r_set.find(rEntity) != r_set.end()) {
is_valid = false;
break;
}
}

return is_valid;
}
};

struct ModelPartIntersectionOperator {
template <class TCheckType>
static bool IsValid(
const TCheckType& rEntity,
const std::vector<std::set<TCheckType>>& mIntersectionSets)
{
bool is_valid = true;

for (const auto& r_set : mIntersectionSets) {
if (r_set.find(rEntity) == r_set.end()) {
is_valid = false;
break;
}
}

return is_valid;
}
};


} 
