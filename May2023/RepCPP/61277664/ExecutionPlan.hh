#pragma once

#include <utility>
#include <vector>
#include <set>
#include "VertexSet.h"
#include "cmap.h"

struct uidType {
static const int64_t MIN = -1;
static const int64_t MAX = -2;

int64_t id;
uidType(size_t uid) : id(uid) {}
uidType(int64_t uid) : id(uid) {}
bool is_bounded() const {
return id >= 0;
}
bool is_min() const {
return id == MIN;
}
bool is_max() const {
return id == MAX;
}
};

enum UPDATE_RANGE { ALL, FILTERED };        
enum UPDATE_OP { COMP_SET, AND_OR, NO_OP }; 

template <typename VT>
struct UpdatePolicy {
bool update_cmap;
UPDATE_RANGE range;
UPDATE_OP opcode;
VT cond_value;
VT upd_value;

inline bool update_cond(VT bucket_value) const {
switch (opcode) {
case COMP_SET:
return bucket_value == cond_value;
case AND_OR:
return static_cast<bool>(bucket_value & cond_value);
default:
return false;
}
}

inline VT update_op(VT bucket_value) const {
switch (opcode) {
case COMP_SET:
return upd_value;
case AND_OR:
return bucket_value | upd_value;
default:
return bucket_value;
}
}

inline bool restore_cond(VT bucket_value) const {
switch (opcode) {
case COMP_SET:
return bucket_value == upd_value;
case AND_OR:
return true;
default:
return false;
}
}

inline VT restore_op(VT bucket_value) const {
switch (opcode) {
case COMP_SET:
return cond_value;
case AND_OR:
return bucket_value & (~upd_value);
default:
return bucket_value;
}
}

};

struct Rule {
std::pair<uidType, uidType> bound;
std::set<uidType> connected;
std::set<uidType> disconnected;
};

template <typename idT>
struct EncodedRule {
idT adj_id;     
idT upd_id;     
idT lower_id;
idT upper_id;
UpdatePolicy<uint8_t> policy;
};

template <>
struct EncodedRule<uidType> {
uidType src_id;
uidType lower_id;
uidType upper_id;
UpdatePolicy<uint8_t> policy;

EncodedRule<vidType> inst_with(const std::vector<vidType> &history) {
return { history.at(src_id),
lower_id.is_bounded() ? MIN_VID : history.at(lower_id),
upper_id.is_bounded() ? MAX_VID : history.at(upper_id),
policy
};
}
};

using EncodedRuleAbst = EncodedRule<uidType>;
using EncodedRuleInst = EncodedRule<vidType>;

using embidType = size_t;
using EmbRelation = std::pair<embidType, embidType>;
using EmbExtension = std::pair<embidType, Rule>;

class ExecutionPlan {
private:
std::vector<EmbRelation> relations;
std::vector<EncodedRuleAbst> extend_rules;

public:

ExecutionPlan(const std::vector<EmbExtension> &extensions) {
}

size_t pattern_size() {
return extend_rules.size() + 1;
}

const EncodedRuleAbst &rule_at(size_t level) {
return extend_rules.at(level-1);
}
}

