

#if _MSC_VER==1500 && !__INTEL_COMPILER
#pragma warning( push )
#pragma warning( disable: 4985 )
#endif
#include <cmath>
#if _MSC_VER==1500 && !__INTEL_COMPILER
#pragma warning( pop )
#endif
#include "tbb/tbb_stddef.h"
#include "harness.h"
#include <vector>

namespace test_partitioner_utils {

struct RangeStatisticData {
size_t m_rangeNum;

size_t m_minRangeSize;
size_t m_maxRangeSize;

bool m_wasMinRangeSizeWritten; 
};

using tbb::internal::uint64_t;
using tbb::split;
using tbb::proportional_split;
using tbb::blocked_range;

class RangeStatisticCollector {
public:
RangeStatisticCollector(RangeStatisticData *statisticData) :
m_statData(statisticData)
{
m_called = false;
if (m_statData)
m_statData->m_rangeNum = 1;
}

RangeStatisticCollector(RangeStatisticCollector& sc, size_t rangeSize) {
if (!sc.m_called) {
sc.m_called = true;

if (sc.m_statData) {
size_t *minRangeSize = &sc.m_statData->m_minRangeSize;
if (*minRangeSize > rangeSize || !sc.m_statData->m_wasMinRangeSizeWritten) { 
*minRangeSize = rangeSize;
sc.m_statData->m_wasMinRangeSizeWritten = true;
}
size_t *maxRangeSize = &sc.m_statData->m_maxRangeSize;
if (*maxRangeSize < rangeSize) { 
*maxRangeSize = rangeSize;
}
}
}
*this = sc;
}

RangeStatisticCollector(RangeStatisticCollector& sc, proportional_split&) {
if (sc.m_statData)
sc.m_statData->m_rangeNum++;
*this = sc;
}

private:
RangeStatisticData *m_statData;

bool m_called;
};

template <typename DerivedRange, typename T>
class RangeBase: public RangeStatisticCollector {
protected:
size_t my_begin, my_end;
bool m_provide_feedback;
bool m_ensure_non_empty_size;
public:
RangeBase(size_t _begin, size_t _end, RangeStatisticData *statData,
bool provide_feedback, bool ensure_non_empty_size)
: RangeStatisticCollector(statData)
, my_begin(_begin), my_end(_end)
, m_provide_feedback(provide_feedback)
, m_ensure_non_empty_size(ensure_non_empty_size)
{ }
RangeBase(RangeBase& r, tbb::split) : RangeStatisticCollector(r, r.size()) {
*this = r;
size_t middle = r.my_begin + (r.my_end - r.my_begin) / 2u;
r.my_end = my_begin = middle;
}

RangeBase(RangeBase& r, proportional_split& p) : RangeStatisticCollector(r, p) {
*this = r;
size_t original_size = r.size();
T right = self().compute_right_part(r, p);
size_t right_part = self().round(right);
if( m_ensure_non_empty_size ) {
right_part = (original_size == right_part) ? (original_size - 1) : right_part;
right_part = (right_part != 0) ? right_part : 1;
}
r.my_end = my_begin = r.my_end - right_part;
#if __TBB_ENABLE_RANGE_FEEDBACK
if( m_provide_feedback )
p.set_proportion(original_size - right_part, right_part);
#endif
if( m_ensure_non_empty_size )
ASSERT(r.my_end != r.my_begin && my_end != my_begin, "Incorrect range split");
}

size_t begin() const { return my_begin; }
size_t end() const { return my_end; }
bool is_divisible() const { return (my_end - my_begin) > 1; }
bool empty() const { return my_end == my_begin; }
size_t size() const { return my_end - my_begin; }

DerivedRange& self() { return static_cast<DerivedRange&>(*this); }
size_t round(T part) { return size_t(part); }
T compute_right_part(RangeBase& r, proportional_split& p) {
return T(r.size() * T(p.right())) / T(p.left() + p.right());
}
bool is_ensure_non_emptiness() { return m_ensure_non_empty_size; }
};

namespace TestRanges {


class RoundedDownRange: public RangeBase<RoundedDownRange, float> {
public:
RoundedDownRange(size_t _begin, size_t _end, RangeStatisticData *statData,
bool provide_feedback, bool ensure_non_empty_size)
: RangeBase<RoundedDownRange, float>(_begin, _end, statData, provide_feedback,
ensure_non_empty_size) { }
RoundedDownRange(RoundedDownRange& r, tbb::split)
: RangeBase<RoundedDownRange, float>(r, tbb::split()) { }
RoundedDownRange(RoundedDownRange& r, proportional_split& p)
: RangeBase<RoundedDownRange, float>(r, p) { }
static const bool is_splittable_in_proportion = true;
};

class RoundedUpRange: public RangeBase<RoundedUpRange, float> {
public:
RoundedUpRange(size_t _begin, size_t _end, RangeStatisticData *statData,
bool provide_feedback, bool ensure_non_empty_size)
: RangeBase<RoundedUpRange, float>(_begin, _end, statData, provide_feedback,
ensure_non_empty_size) { }
RoundedUpRange(RoundedUpRange& r, tbb::split)
: RangeBase<RoundedUpRange, float>(r, tbb::split()) { }
RoundedUpRange(RoundedUpRange& r, proportional_split& p)
: RangeBase<RoundedUpRange, float>(r, p) { }
size_t round(float part) { return size_t(std::ceil(part)); }
static const bool is_splittable_in_proportion = true;
};

class Range1_2: public RangeBase<Range1_2, float> {
public:
Range1_2(size_t _begin, size_t _end, RangeStatisticData *statData,
bool provide_feedback, bool ensure_non_empty_size)
: RangeBase<Range1_2, float>(_begin, _end, statData, provide_feedback,
ensure_non_empty_size) { }
Range1_2(Range1_2& r, tbb::split) : RangeBase<Range1_2, float>(r, tbb::split()) { }
Range1_2(Range1_2& r, proportional_split& p) : RangeBase<Range1_2, float>(r, p) { }
static const bool is_splittable_in_proportion = true;
float compute_right_part(RangeBase<Range1_2, float>& r, proportional_split&) {
return float(r.size() * 2) / 3.0f;
}
};

class Range1_999: public RangeBase<Range1_999, float> {
public:
Range1_999(size_t _begin, size_t _end, RangeStatisticData *statData,
bool provide_feedback, bool ensure_non_empty_size)
: RangeBase<Range1_999, float>(_begin, _end, statData, provide_feedback,
ensure_non_empty_size) { }
Range1_999(Range1_999& r, tbb::split) : RangeBase<Range1_999, float>(r, tbb::split()) { }
Range1_999(Range1_999& r, proportional_split& p) : RangeBase<Range1_999, float>(r, p) { }
static const bool is_splittable_in_proportion = true;
float compute_right_part(RangeBase<Range1_999, float>& r, proportional_split&) {
return float(r.size() * 999) / 1000.0f;
}
};

class Range999_1: public RangeBase<Range999_1, float> {
public:
Range999_1(size_t _begin, size_t _end, RangeStatisticData *statData,
bool provide_feedback, bool ensure_non_empty_size)
: RangeBase<Range999_1, float>(_begin, _end, statData, provide_feedback,
ensure_non_empty_size) { }
Range999_1(Range999_1& r, tbb::split) : RangeBase<Range999_1, float>(r, tbb::split()) { }
Range999_1(Range999_1& r, proportional_split& p) : RangeBase<Range999_1, float>(r, p) { }
static const bool is_splittable_in_proportion = true;
float compute_right_part(RangeBase<Range999_1, float>& r, proportional_split&) {
return float(r.size()) / 1000.0f;
}
};

class BlockedRange: public RangeStatisticCollector, public blocked_range<size_t>  {
public:
BlockedRange(size_t _begin, size_t _end, RangeStatisticData *statData, bool, bool)
: RangeStatisticCollector(statData), blocked_range<size_t>(_begin, _end) { }
BlockedRange(BlockedRange& r, split)
: RangeStatisticCollector(r, r.size()), blocked_range<size_t>(r, split()) { }
BlockedRange(BlockedRange& r, proportional_split& p)
: RangeStatisticCollector(r, p), blocked_range<size_t>(r, p) { }
static const bool is_splittable_in_proportion = true;
bool is_ensure_non_emptiness() { return false; }
};

class InvertedProportionRange: public RangeBase<InvertedProportionRange, float> {
public:
InvertedProportionRange(size_t _begin, size_t _end, RangeStatisticData *statData,
bool provide_feedback, bool ensure_non_empty_size)
: RangeBase<InvertedProportionRange, float>(_begin, _end, statData, provide_feedback,
ensure_non_empty_size) { }
InvertedProportionRange(InvertedProportionRange& r, split)
: RangeBase<InvertedProportionRange, float>(r, split()) { }
InvertedProportionRange(InvertedProportionRange& r, proportional_split& p)
: RangeBase<InvertedProportionRange, float>(r, p) { }
float compute_right_part(RangeBase<InvertedProportionRange, float>& r,
proportional_split& p) {
return float(r.size() * float(p.left())) / float(p.left() + p.right());
}
static const bool is_splittable_in_proportion = true;
};

class ExactSplitRange: public RangeBase<ExactSplitRange, size_t> {
public:
ExactSplitRange(size_t _begin, size_t _end, RangeStatisticData *statData,
bool provide_feedback, bool ensure_non_empty_size)
: RangeBase<ExactSplitRange, size_t>(_begin, _end, statData, provide_feedback,
ensure_non_empty_size) { }
ExactSplitRange(ExactSplitRange& r, split)
: RangeBase<ExactSplitRange, size_t>(r, split()) { }
ExactSplitRange(ExactSplitRange& r, proportional_split& p)
: RangeBase<ExactSplitRange, size_t>(r, p) { }
size_t compute_right_part(RangeBase<ExactSplitRange, size_t>& r, proportional_split& p) {
size_t parts = size_t(p.left() + p.right());
size_t currSize = r.size();
size_t int_part = currSize / parts * p.right();
size_t remainder = currSize % parts * p.right();
int_part += remainder / parts;
remainder %= parts;
size_t right_part = int_part + (remainder > parts/2 ? 1 : 0);
return right_part;
}
static const bool is_splittable_in_proportion = true;
};

} 

struct TreeNode {
size_t m_affinity;
size_t m_range_begin, m_range_end;
TreeNode *m_left, *m_right;
private:
TreeNode(size_t range_begin, size_t range_end, size_t affinity,
TreeNode* left, TreeNode* right)
: m_affinity(affinity), m_range_begin(range_begin), m_range_end(range_end),
m_left(left), m_right(right) { }

friend TreeNode* make_node(size_t range_begin, size_t range_end, size_t affinity,
TreeNode *left, TreeNode *right);
};

TreeNode* make_node(size_t range_begin, size_t range_end, size_t affinity,
TreeNode* left = NULL, TreeNode* right = NULL) {
ASSERT(range_begin <= range_end, "Incorrect range interval");
return new TreeNode(range_begin, range_end, affinity, left, right);
}

class BinaryTree {
public:
BinaryTree() : m_root(NULL) { }
~BinaryTree() {
if (m_root)
remove_node_recursively(m_root);
}

void push_node(TreeNode* node) {
if (!node)
return;

if (m_root) {
ASSERT(node->m_range_begin >= m_root->m_range_begin &&
node->m_range_end <= m_root->m_range_end,
"Cannot push node not from subrange");
}

push_subnode(m_root, node);
}

void visualize() {
if (!m_root) { 
REPORT("Tree is empty\n");
return;
}
visualize_node(m_root);
}

bool operator ==(const BinaryTree& other_tree) const { return compare_nodes(m_root, other_tree.m_root); }
void fill_leafs(std::vector<TreeNode*>& leafs) const { fill_leafs_impl(m_root, leafs); }

private:
TreeNode *m_root;

void push_subnode(TreeNode *&root_node, TreeNode *node) {
if (!root_node) {
root_node = node;
return;
} else if (are_nodes_equal(root_node, node)) {
return;
}

if (!has_children(root_node)) {
if (is_look_like_left_sibling(root_node, node))
push_subnode(root_node->m_left, node);
else
push_subnode(root_node->m_right, node);
return;
}

if (has_left_child(root_node)) {
if (is_subnode(root_node->m_left, node)) {
push_subnode(root_node->m_left, node);
return;
}
push_subnode(root_node->m_right, node);
return;
}

ASSERT(root_node->m_right != NULL, "Right child is NULL but must be present");
if (is_subnode(root_node->m_right, node)) {
push_subnode(root_node->m_right, node);
return;
}
push_subnode(root_node->m_left, node);
return;
}

bool has_children(TreeNode *node) { return node->m_left || node->m_right; }

bool is_look_like_left_sibling(TreeNode *root_node, TreeNode *node) {
if (root_node->m_range_begin == node->m_range_begin)
return true;
ASSERT(root_node->m_range_end == node->m_range_end, NULL);
return false;
}

bool has_left_child(TreeNode *node) { return node->m_left != NULL; }

bool is_subnode(TreeNode *root_node, TreeNode *node) {
return root_node->m_range_begin <= node->m_range_begin &&
node->m_range_end <= root_node->m_range_end;
}

bool are_nodes_equal(TreeNode *node1, TreeNode *node2) const {
return node1->m_range_begin == node2->m_range_begin &&
node1->m_range_end == node2->m_range_end;
}

void remove_node_recursively(TreeNode *node) {
if (node->m_left)
remove_node_recursively(node->m_left);
if (node->m_right)
remove_node_recursively(node->m_right);
delete node;
}

static void visualize_node(const TreeNode* node, unsigned indent = 0) {
const char *indentStep = "    ";
for (unsigned i = 0; i < indent; ++i)
REPORT("%s", indentStep);

size_t rangeSize = node->m_range_end - node->m_range_begin;
REPORT("[%llu, %llu)%%%llu@%llu\n", uint64_t(node->m_range_begin), uint64_t(node->m_range_end),
uint64_t(rangeSize), uint64_t(node->m_affinity));

if (node->m_left)
visualize_node(node->m_left, indent + 1);
if (node->m_right)
visualize_node(node->m_right, indent + 1);
}

bool compare_nodes(TreeNode* node1, TreeNode* node2) const {
if (node1 == NULL && node2 == NULL) return true;
if (node1 == NULL || node2 == NULL) return false;
return are_nodes_equal(node1, node2) && compare_nodes(node1->m_left, node2->m_left)
&& compare_nodes(node1->m_right, node2->m_right);
}

void fill_leafs_impl(TreeNode* node, std::vector<TreeNode*>& leafs) const {
if (node->m_left == NULL && node->m_right == NULL)
leafs.push_back(node);
if (node->m_left != NULL) fill_leafs_impl(node->m_left, leafs);
if (node->m_right != NULL) fill_leafs_impl(node->m_right, leafs);
}
};

class SimpleBody {
public:
SimpleBody() { }
template <typename Range>
void operator()(Range&) const { }
};

class SimpleReduceBody {
public:
SimpleReduceBody() { }
SimpleReduceBody(SimpleReduceBody&, tbb::split) { }
template <typename Range>
void operator()(Range&) { }
void join(SimpleReduceBody&) { }
};

namespace interaction_with_range_and_partitioner {

class SplitConstructorAssertedRange {
mutable bool is_divisible_called;
mutable bool is_empty_called;
bool my_assert_in_nonproportional, my_assert_in_proportional;
public:
SplitConstructorAssertedRange(bool assert_in_nonproportional, bool assert_in_proportional)
: is_divisible_called(false),
is_empty_called(false),
my_assert_in_nonproportional(assert_in_nonproportional),
my_assert_in_proportional(assert_in_proportional) { }
SplitConstructorAssertedRange(SplitConstructorAssertedRange& r, tbb::split) {
*this = r;
ASSERT( !my_assert_in_nonproportional, "Disproportional splitting constructor was called but should not been" );
}
SplitConstructorAssertedRange(SplitConstructorAssertedRange& r, proportional_split&) {
*this = r;
ASSERT( !my_assert_in_proportional, "Proportional splitting constructor was called but should not been" );
}
bool is_divisible() const {
if (!is_divisible_called) {
is_divisible_called = true;
return true;
}
return false;
}
bool empty() const {
if (!is_empty_called) {
is_empty_called = true;
return false;
}
return true;
}
};




class Range1: public SplitConstructorAssertedRange {
public:
Range1(bool assert_in_nonproportional, bool assert_in_proportional)
: SplitConstructorAssertedRange(assert_in_nonproportional, assert_in_proportional) { }
Range1( Range1& r, tbb::split ) : SplitConstructorAssertedRange(r, tbb::split()) { }
Range1( Range1& r, proportional_split& proportion ) : SplitConstructorAssertedRange(r, proportion) { }
static const bool is_splittable_in_proportion = true;
};

class Range2: public SplitConstructorAssertedRange {
public:
Range2(bool assert_in_nonproportional, bool assert_in_proportional)
: SplitConstructorAssertedRange(assert_in_nonproportional, assert_in_proportional) { }
Range2(Range2& r, tbb::split) : SplitConstructorAssertedRange(r, tbb::split()) { }
Range2(Range2& r, proportional_split& p) : SplitConstructorAssertedRange(r, p) {
}
static const bool is_splittable_in_proportion = false;
};

class Range3: public SplitConstructorAssertedRange {
public:
Range3(bool assert_in_nonproportional, bool assert_in_proportional)
: SplitConstructorAssertedRange(assert_in_nonproportional, assert_in_proportional) { }
Range3(Range3& r, tbb::split) : SplitConstructorAssertedRange(r, tbb::split()) { }
Range3(Range3& r, proportional_split& p) : SplitConstructorAssertedRange(r, p) {
}
};

class Range4: public SplitConstructorAssertedRange {
public:
Range4(bool assert_in_nonproportional, bool assert_in_proportional)
: SplitConstructorAssertedRange(assert_in_nonproportional, assert_in_proportional) { }
Range4(Range4& r, tbb::split) : SplitConstructorAssertedRange(r, tbb::split()) { }
static const bool is_splittable_in_proportion = true;
};

class Range5: public SplitConstructorAssertedRange {
public:
Range5(bool assert_in_nonproportional, bool assert_in_proportional)
: SplitConstructorAssertedRange(assert_in_nonproportional, assert_in_proportional) { }
Range5(Range5& r, tbb::split) : SplitConstructorAssertedRange(r, tbb::split()) { }
static const bool is_splittable_in_proportion = false;
};

class Range6: public SplitConstructorAssertedRange {
public:
Range6(bool assert_in_nonproportional, bool assert_in_proportional)
: SplitConstructorAssertedRange(assert_in_nonproportional, assert_in_proportional) { }
Range6(Range6& r, tbb::split) : SplitConstructorAssertedRange(r, tbb::split()) { }
};

} 

} 
