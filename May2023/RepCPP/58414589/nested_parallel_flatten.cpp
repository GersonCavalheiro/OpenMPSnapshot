
#include "nested_parallel_flatten.hpp"
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/visitor.hpp>
#include <runtime/barrier.hpp>
#include <runtime/config.hpp>
#include <unordered_map>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(nested_parallel_flattener,
SC_PASS_DEPENDS_ON(validator, buffer_rescheduling_tensor_hoisting),
SC_PASS_REQUIRE_STATE(CONST_FOLDED, IR_SIMPLIFIED),
SC_PASS_REQUIRE_NOT_STATE(), SC_PASS_SET_STATE(),
SC_PASS_UNSET_STATE(CONST_FOLDED, IR_SIMPLIFIED));

class nested_parallel_flatten_impl_t : public ir_visitor_t {
struct parallel_info_t {
int num_groups_;
int threads_per_group_;
expr thread_id_;
expr group_id_;
expr barriers_;
parallel_info_t(int num_groups, int threads_per_group)
: num_groups_(num_groups), threads_per_group_(threads_per_group) {}
};

std::vector<parallel_info_t> info_;
std::vector<stmt> *top_level_parallel_seq_ = nullptr;
expr global_tid_;
int runtime_threads_ = runtime_config_t::get().get_num_threads();
int count_ = 0;
int var_count_ = 0;
int for_count_ = 0;
bool cannot_parallel_ = false;
bool need_pre_barrier_ = false;
bool need_post_barrier_ = false;

public:
using ir_visitor_t::dispatch;

std::string make_name(const char *n) {
std::string name = n;
name += std::to_string(count_);
name += '_';
name += std::to_string(info_.size());
name += '_';
name += std::to_string(++var_count_);
return name;
}

expr get_barrier_for_current_for() {
COMPILE_ASSERT(top_level_parallel_seq_ && info_.size() > 1UL,
"Invalid for-loop");
int num_barrier = 1;
constexpr uint64_t barrier_size = sizeof(runtime::barrier_t);
expr idx = info_[info_.size() - 2].group_id_ * barrier_size;
for (int64_t i = info_.size() - 2; i >= 0; i--) {
num_barrier *= info_[i].num_groups_;
if (i != 0) {
idx = info_[i - 1].group_id_ * (num_barrier * barrier_size)
+ idx;
}
}
if (info_.back().barriers_.defined()) {
return builder::tensor_ptr(info_.back().barriers_, {idx});
}

info_.back().barriers_ = builder::make_tensor(make_name("_barrier"),
{num_barrier * barrier_size}, datatypes::u8);
top_level_parallel_seq_->emplace_back(
builder::make_var_tensor_def_unattached(
info_.back().barriers_));
top_level_parallel_seq_->emplace_back(builder::make_evaluate_unattached(
builtin::get_init_barrier_func()(info_.back().barriers_,
num_barrier,
uint64_t(info_[info_.size() - 2].threads_per_group_))));
return builder::tensor_ptr(info_.back().barriers_, {idx});
}

void gen_call_to_barrier(
std::vector<stmt> *cur_insert_point, int post_barrier_id) {
auto b = get_barrier_for_current_for();
auto the_call = builtin::get_barrier_arrive_func()(
b, get_ir_null(), get_ir_null());
cur_insert_point->emplace_back(
builder::make_evaluate_unattached(the_call));
if (post_barrier_id >= 0) {
the_call->attr()["post_barrier_id"] = post_barrier_id;
}
}

bool is_trace_call(const stmt &v) {
return v.cast<evaluate>()
.map([](const evaluate &v) { return v->value_.as<call>(); })
.map([](const call &v) {
return dynamic_cast<func_base *>(v->func_.get());
})
.filter([](func_base *f) {
return f->attr_
&& f->attr_->get_or_else(
function_attrs::is_trace_func, false);
})
.has_value();
}


void transform_loop(const for_loop_c &v, int num_threads_parent_group,
std::vector<stmt> &seq, expr tid0, bool need_pre_barrier, 
bool need_post_barrier) {
int cur_post_barrier_id = for_count_;
for_count_++;
COMPILE_ASSERT(info_.empty() || info_.front().threads_per_group_ != 0,
"Cannot handle nested parallel-for without num threads in most "
"outer parallel-for");
int num_threads = v->num_threads_;
if (num_threads == 0) { num_threads = num_threads_parent_group; }
bool divisible = num_threads_parent_group % num_threads == 0;
uint64_t threads_per_group = num_threads_parent_group / num_threads;
COMPILE_ASSERT(threads_per_group > 0,
"Too many threads in this parallel: " << v);

info_.emplace_back(num_threads, threads_per_group);
auto gid1 = builder::make_var(datatypes::index, make_name("_gid"));
info_.back().group_id_ = gid1;
auto tid1 = builder::make_var(datatypes::index, make_name("_tid"));
info_.back().thread_id_ = tid1;
seq.emplace_back(builder::make_var_tensor_def_unattached(
gid1, linkage::local, tid0 / threads_per_group));
seq.emplace_back(builder::make_var_tensor_def_unattached(
tid1, linkage::local, tid0 % threads_per_group));
std::vector<stmt> *cur_insert_point = &seq;
if (!divisible) {
auto gid_ok_body = make_stmt<stmts_node_t>(std::vector<stmt> {});
stmts gid_skip_body;
if (need_pre_barrier) {
gid_skip_body = make_stmt<stmts_node_t>(std::vector<stmt> {});
gen_call_to_barrier(&gid_skip_body->seq_, -1);
}
seq.emplace_back(builder::make_if_else_unattached(
gid1 < make_expr<constant_node>(
uint64_t(num_threads), datatypes::index),
gid_ok_body, gid_skip_body));
cur_insert_point = &gid_ok_body->seq_;
}
expr begin, end;
builtin::generate_balance211(
num_threads, v->iter_begin_, v->iter_end_, v->step_, gid1,
[&](const char *v) { return make_name(v); }, &begin, nullptr,
&end, cur_insert_point);
if (need_pre_barrier) { gen_call_to_barrier(cur_insert_point, -1); }

auto new_body = make_stmt<stmts_node_t>(std::vector<stmt> {});
auto step_expr = v->step_->dtype_ == datatypes::index
? v->step_
: constant_folder_t()(
builder::make_cast(datatypes::index, v->step_));
cur_insert_point->emplace_back(builder::make_for_loop_unattached(
v->var_, begin, end, step_expr, new_body, v->incremental_,
for_type::NORMAL));

auto &old_body = v->body_.checked_as<stmts>()->seq_;
stmts single_thread_body;
bool local_need_pre_barrier = false;
for (size_t i = 0; i < old_body.size(); i++) {
if (old_body[i].isa<for_loop>()
&& old_body[i].static_as<for_loop>()->kind_
== for_type::PARALLEL) {
cannot_parallel_ = false;
need_pre_barrier_ = local_need_pre_barrier;
need_post_barrier_ = false;
for (size_t n = i + 1; n < old_body.size(); n++) {
if (old_body[n].isa<define_c>()) {
auto &initv = old_body[n].static_as<define_c>()->init_;
if (!initv.defined() || initv.isa<constant>()) {
continue;
}
} else if (is_trace_call(old_body[n])) {
continue;
}

need_post_barrier_ = true;
break;
}
auto body = dispatch(old_body[i])
.remove_const()
.checked_as<stmts>();
new_body->seq_.insert(new_body->seq_.end(), body->seq_.begin(),
body->seq_.end());
single_thread_body = stmts();
local_need_pre_barrier = false;
} else if (old_body[i]
.cast<define>()
.filter([](const define &v) {
return !v->init_.defined()
|| !v->init_.isa<indexing>();
})
.has_value()) {
if (old_body[i].static_as<define>()->init_.defined()) {
single_thread_body = stmts();
new_body->seq_.emplace_back(
dispatch(old_body[i]).remove_const());
} else {
new_body->seq_.insert(new_body->seq_.begin(),
dispatch(old_body[i]).remove_const());
}
} else if (is_trace_call(old_body[i])) {
new_body->seq_.emplace_back(
dispatch(old_body[i]).remove_const());
} else {
cannot_parallel_ = true;
auto dispatched = dispatch(old_body[i]).remove_const();
bool is_set_idle_func_call = dispatched.isa<evaluate>()
&& dispatched.static_as<evaluate>()
->value_.isa<intrin_call>()
&& dispatched.static_as<evaluate>()
->value_.static_as<intrin_call>()
->type_
== intrin_type::set_thread_idle_func;
if (is_set_idle_func_call) {
dispatched.static_as<evaluate_c>()
->value_.static_as<intrin_call>()
->attr()["post_barrier_id"]
= for_count_;
}
if (threads_per_group > 1 && !is_set_idle_func_call) {
if (dispatched.isa<stmts>()
&& dispatched.static_as<stmts>()->seq_.empty()) {
} else {
local_need_pre_barrier = true;
if (!single_thread_body.defined()) {
single_thread_body = make_stmt<stmts_node_t>(
std::vector<stmt> {});
new_body->seq_.emplace_back(
builder::make_if_else_unattached(
tid1 == UINT64_C(0),
single_thread_body, stmt()));
}
single_thread_body->seq_.emplace_back(dispatched);
}

} else {
new_body->seq_.emplace_back(dispatched);
}
}
}
if (need_post_barrier) {
gen_call_to_barrier(&seq, cur_post_barrier_id);
}

info_.pop_back();
}

stmt_c visit(for_loop_c v) override {
COMPILE_ASSERT(
v->num_threads_ >= 0 && v->num_threads_ <= runtime_threads_,
"Bad thread count: " << v);
if (v->kind_ == for_type::PARALLEL) {
COMPILE_ASSERT(!cannot_parallel_,
"Cannot parallel here. The inner parallel for must be "
"directly nested in parent parallel-for. "
<< v);
if (info_.empty()) {
if (v->num_threads_ == 0
|| v->num_threads_ == runtime_threads_) {
info_.emplace_back(0, 0);
top_level_parallel_seq_ = nullptr;
auto ret = ir_visitor_t::visit(v);
info_.pop_back();
return ret;
}

auto body_lv0 = make_stmt<stmts_node_t>(std::vector<stmt> {});
top_level_parallel_seq_ = &body_lv0->seq_;
auto body_lv1 = make_stmt<stmts_node_t>(std::vector<stmt> {});
auto tid0
= builder::make_var(datatypes::index, make_name("tid"));
count_++;
int num_threads = v->num_threads_;
COMPILE_ASSERT(runtime_threads_ >= num_threads,
"num_threads of the loop excesses the total number of "
"threads: "
<< v);
num_threads = runtime_threads_ / num_threads * num_threads;
global_tid_ = tid0;

auto for_lv1 = builder::make_for_loop_unattached(tid0,
UINT64_C(0), uint64_t(num_threads), UINT64_C(1),
body_lv1, true, for_type::PARALLEL);
transform_loop(
v, num_threads, body_lv1->seq_, tid0, false, false);
body_lv0->seq_.emplace_back(for_lv1);
if (v->attr_
&& v->attr_->get_or_else(
stmt_attr_key::no_post_barrier, false)) {
for_lv1->attr()[stmt_attr_key::no_post_barrier] = true;
}
global_tid_ = expr();
top_level_parallel_seq_ = nullptr;
cannot_parallel_ = false;
return body_lv0;
} else {
assert(!info_.empty());
auto &parent_info = info_.back();
auto body_lv1 = make_stmt<stmts_node_t>(std::vector<stmt> {});
transform_loop(v, parent_info.threads_per_group_,
body_lv1->seq_, parent_info.thread_id_,
need_pre_barrier_,
need_post_barrier_
&& !(v->attr_
&& v->attr_->get_or_else(
stmt_attr_key::no_post_barrier,
false)));
return body_lv1;
}
} else {
return ir_visitor_t::visit(v);
}
}

stmt_c visit(stmts_c v) override {
std::vector<stmt_c> newseq;
newseq.reserve(v->seq_.size());
bool changed = false;
for (auto &s : v->seq_) {
auto news = dispatch(s);
if (!news.ptr_same(s)) { changed = true; }
if (s.isa<for_loop>() && news.isa<stmts>()) {
auto &inner = news.static_as<stmts>()->seq_;
newseq.insert(newseq.end(), inner.begin(), inner.end());
} else {
newseq.emplace_back(news);
}
}
if (!changed) {
return v;
} else {
return copy_attr(*v, builder::make_stmts_unattached(newseq));
}
}

expr_c visit(intrin_call_c v) override {
if (v->type_ == intrin_type::get_group_id) {
uint64_t level_id
= get_const_as_int(v->args_[0].checked_as<constant_c>());
COMPILE_ASSERT(
level_id < info_.size(), "Level of group out of range");
return info_[level_id].group_id_;
} else if (v->type_ == intrin_type::get_group_thread_id) {
int64_t level_id
= get_const_as_int(v->args_[0].checked_as<constant_c>());
COMPILE_ASSERT(level_id < (int64_t)info_.size(),
"Level of group out of range");
if (level_id < 0) {
if (global_tid_.defined()) {
return builder::make_cast(datatypes::s32, global_tid_);
} else {
return v;
}
} else {
return info_[level_id].thread_id_;
}
} else {
return ir_visitor_t::visit(v);
}
}
};

func_c nested_parallel_flattener_t::operator()(func_c f) {
nested_parallel_flatten_impl_t impl;
f = impl.dispatch(std::move(f));
return f;
}

} 
} 
} 
} 
