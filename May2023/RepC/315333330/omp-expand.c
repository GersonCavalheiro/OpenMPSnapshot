#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "memmodel.h"
#include "backend.h"
#include "target.h"
#include "rtl.h"
#include "tree.h"
#include "gimple.h"
#include "cfghooks.h"
#include "tree-pass.h"
#include "ssa.h"
#include "optabs.h"
#include "cgraph.h"
#include "pretty-print.h"
#include "diagnostic-core.h"
#include "fold-const.h"
#include "stor-layout.h"
#include "cfganal.h"
#include "internal-fn.h"
#include "gimplify.h"
#include "gimple-iterator.h"
#include "gimplify-me.h"
#include "gimple-walk.h"
#include "tree-cfg.h"
#include "tree-into-ssa.h"
#include "tree-ssa.h"
#include "splay-tree.h"
#include "cfgloop.h"
#include "omp-general.h"
#include "omp-offload.h"
#include "tree-cfgcleanup.h"
#include "symbol-summary.h"
#include "gomp-constants.h"
#include "gimple-pretty-print.h"
#include "hsa-common.h"
#include "stringpool.h"
#include "attribs.h"
struct omp_region
{
struct omp_region *outer;
struct omp_region *inner;
struct omp_region *next;
basic_block entry;
basic_block exit;
basic_block cont;
vec<tree, va_gc> *ws_args;
enum gimple_code type;
enum omp_clause_schedule_kind sched_kind;
unsigned char sched_modifiers;
bool is_combined_parallel;
gomp_ordered *ord_stmt;
};
static struct omp_region *root_omp_region;
static bool omp_any_child_fn_dumped;
static void expand_omp_build_assign (gimple_stmt_iterator *, tree, tree,
bool = false);
static gphi *find_phi_with_arg_on_edge (tree, edge);
static void expand_omp (struct omp_region *region);
static inline bool
is_combined_parallel (struct omp_region *region)
{
return region->is_combined_parallel;
}
static bool
workshare_safe_to_combine_p (basic_block ws_entry_bb)
{
struct omp_for_data fd;
gimple *ws_stmt = last_stmt (ws_entry_bb);
if (gimple_code (ws_stmt) == GIMPLE_OMP_SECTIONS)
return true;
gcc_assert (gimple_code (ws_stmt) == GIMPLE_OMP_FOR);
omp_extract_for_data (as_a <gomp_for *> (ws_stmt), &fd, NULL);
if (fd.collapse > 1 && TREE_CODE (fd.loop.n2) != INTEGER_CST)
return false;
if (fd.iter_type != long_integer_type_node)
return false;
if (!is_gimple_min_invariant (fd.loop.n1)
|| !is_gimple_min_invariant (fd.loop.n2)
|| !is_gimple_min_invariant (fd.loop.step)
|| (fd.chunk_size && !is_gimple_min_invariant (fd.chunk_size)))
return false;
return true;
}
static tree
omp_adjust_chunk_size (tree chunk_size, bool simd_schedule)
{
if (!simd_schedule)
return chunk_size;
poly_uint64 vf = omp_max_vf ();
if (known_eq (vf, 1U))
return chunk_size;
tree type = TREE_TYPE (chunk_size);
chunk_size = fold_build2 (PLUS_EXPR, type, chunk_size,
build_int_cst (type, vf - 1));
return fold_build2 (BIT_AND_EXPR, type, chunk_size,
build_int_cst (type, -vf));
}
static vec<tree, va_gc> *
get_ws_args_for (gimple *par_stmt, gimple *ws_stmt)
{
tree t;
location_t loc = gimple_location (ws_stmt);
vec<tree, va_gc> *ws_args;
if (gomp_for *for_stmt = dyn_cast <gomp_for *> (ws_stmt))
{
struct omp_for_data fd;
tree n1, n2;
omp_extract_for_data (for_stmt, &fd, NULL);
n1 = fd.loop.n1;
n2 = fd.loop.n2;
if (gimple_omp_for_combined_into_p (for_stmt))
{
tree innerc
= omp_find_clause (gimple_omp_parallel_clauses (par_stmt),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
n1 = OMP_CLAUSE_DECL (innerc);
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
n2 = OMP_CLAUSE_DECL (innerc);
}
vec_alloc (ws_args, 3 + (fd.chunk_size != 0));
t = fold_convert_loc (loc, long_integer_type_node, n1);
ws_args->quick_push (t);
t = fold_convert_loc (loc, long_integer_type_node, n2);
ws_args->quick_push (t);
t = fold_convert_loc (loc, long_integer_type_node, fd.loop.step);
ws_args->quick_push (t);
if (fd.chunk_size)
{
t = fold_convert_loc (loc, long_integer_type_node, fd.chunk_size);
t = omp_adjust_chunk_size (t, fd.simd_schedule);
ws_args->quick_push (t);
}
return ws_args;
}
else if (gimple_code (ws_stmt) == GIMPLE_OMP_SECTIONS)
{
basic_block bb = single_succ (gimple_bb (ws_stmt));
t = build_int_cst (unsigned_type_node, EDGE_COUNT (bb->succs) - 1);
vec_alloc (ws_args, 1);
ws_args->quick_push (t);
return ws_args;
}
gcc_unreachable ();
}
static void
determine_parallel_type (struct omp_region *region)
{
basic_block par_entry_bb, par_exit_bb;
basic_block ws_entry_bb, ws_exit_bb;
if (region == NULL || region->inner == NULL
|| region->exit == NULL || region->inner->exit == NULL
|| region->inner->cont == NULL)
return;
if (region->type != GIMPLE_OMP_PARALLEL
|| (region->inner->type != GIMPLE_OMP_FOR
&& region->inner->type != GIMPLE_OMP_SECTIONS))
return;
par_entry_bb = region->entry;
par_exit_bb = region->exit;
ws_entry_bb = region->inner->entry;
ws_exit_bb = region->inner->exit;
if (single_succ (par_entry_bb) == ws_entry_bb
&& single_succ (ws_exit_bb) == par_exit_bb
&& workshare_safe_to_combine_p (ws_entry_bb)
&& (gimple_omp_parallel_combined_p (last_stmt (par_entry_bb))
|| (last_and_only_stmt (ws_entry_bb)
&& last_and_only_stmt (par_exit_bb))))
{
gimple *par_stmt = last_stmt (par_entry_bb);
gimple *ws_stmt = last_stmt (ws_entry_bb);
if (region->inner->type == GIMPLE_OMP_FOR)
{
tree clauses = gimple_omp_for_clauses (ws_stmt);
tree c = omp_find_clause (clauses, OMP_CLAUSE_SCHEDULE);
if (c == NULL
|| ((OMP_CLAUSE_SCHEDULE_KIND (c) & OMP_CLAUSE_SCHEDULE_MASK)
== OMP_CLAUSE_SCHEDULE_STATIC)
|| omp_find_clause (clauses, OMP_CLAUSE_ORDERED))
{
region->is_combined_parallel = false;
region->inner->is_combined_parallel = false;
return;
}
}
region->is_combined_parallel = true;
region->inner->is_combined_parallel = true;
region->ws_args = get_ws_args_for (par_stmt, ws_stmt);
}
}
void dump_omp_region (FILE *, struct omp_region *, int);
void debug_omp_region (struct omp_region *);
void debug_all_omp_regions (void);
void
dump_omp_region (FILE *file, struct omp_region *region, int indent)
{
fprintf (file, "%*sbb %d: %s\n", indent, "", region->entry->index,
gimple_code_name[region->type]);
if (region->inner)
dump_omp_region (file, region->inner, indent + 4);
if (region->cont)
{
fprintf (file, "%*sbb %d: GIMPLE_OMP_CONTINUE\n", indent, "",
region->cont->index);
}
if (region->exit)
fprintf (file, "%*sbb %d: GIMPLE_OMP_RETURN\n", indent, "",
region->exit->index);
else
fprintf (file, "%*s[no exit marker]\n", indent, "");
if (region->next)
dump_omp_region (file, region->next, indent);
}
DEBUG_FUNCTION void
debug_omp_region (struct omp_region *region)
{
dump_omp_region (stderr, region, 0);
}
DEBUG_FUNCTION void
debug_all_omp_regions (void)
{
dump_omp_region (stderr, root_omp_region, 0);
}
static struct omp_region *
new_omp_region (basic_block bb, enum gimple_code type,
struct omp_region *parent)
{
struct omp_region *region = XCNEW (struct omp_region);
region->outer = parent;
region->entry = bb;
region->type = type;
if (parent)
{
region->next = parent->inner;
parent->inner = region;
}
else
{
region->next = root_omp_region;
root_omp_region = region;
}
return region;
}
static void
free_omp_region_1 (struct omp_region *region)
{
struct omp_region *i, *n;
for (i = region->inner; i ; i = n)
{
n = i->next;
free_omp_region_1 (i);
}
free (region);
}
void
omp_free_regions (void)
{
struct omp_region *r, *n;
for (r = root_omp_region; r ; r = n)
{
n = r->next;
free_omp_region_1 (r);
}
root_omp_region = NULL;
}
static gcond *
gimple_build_cond_empty (tree cond)
{
enum tree_code pred_code;
tree lhs, rhs;
gimple_cond_get_ops_from_tree (cond, &pred_code, &lhs, &rhs);
return gimple_build_cond (pred_code, lhs, rhs, NULL_TREE, NULL_TREE);
}
static bool
parallel_needs_hsa_kernel_p (struct omp_region *region)
{
bool indirect = false;
for (region = region->outer; region; region = region->outer)
{
if (region->type == GIMPLE_OMP_PARALLEL)
indirect = true;
else if (region->type == GIMPLE_OMP_TARGET)
{
gomp_target *tgt_stmt
= as_a <gomp_target *> (last_stmt (region->entry));
if (omp_find_clause (gimple_omp_target_clauses (tgt_stmt),
OMP_CLAUSE__GRIDDIM_))
return indirect;
else
return true;
}
}
if (lookup_attribute ("omp declare target",
DECL_ATTRIBUTES (current_function_decl)))
return true;
return false;
}
static void
adjust_context_and_scope (struct omp_region *region, tree entry_block,
tree child_fndecl)
{
tree parent_fndecl = NULL_TREE;
gimple *entry_stmt;
for (region = region->outer;
region && parent_fndecl == NULL_TREE; region = region->outer)
switch (region->type)
{
case GIMPLE_OMP_PARALLEL:
case GIMPLE_OMP_TASK:
entry_stmt = last_stmt (region->entry);
parent_fndecl = gimple_omp_taskreg_child_fn (entry_stmt);
break;
case GIMPLE_OMP_TARGET:
entry_stmt = last_stmt (region->entry);
parent_fndecl
= gimple_omp_target_child_fn (as_a <gomp_target *> (entry_stmt));
break;
default:
break;
}
if (parent_fndecl == NULL_TREE)
parent_fndecl = current_function_decl;
DECL_CONTEXT (child_fndecl) = parent_fndecl;
if (entry_block != NULL_TREE && TREE_CODE (entry_block) == BLOCK)
{
tree b = BLOCK_SUPERCONTEXT (entry_block);
if (TREE_CODE (b) == BLOCK)
{
DECL_CHAIN (child_fndecl) = BLOCK_VARS (b);
BLOCK_VARS (b) = child_fndecl;
}
}
}
static void
expand_parallel_call (struct omp_region *region, basic_block bb,
gomp_parallel *entry_stmt,
vec<tree, va_gc> *ws_args)
{
tree t, t1, t2, val, cond, c, clauses, flags;
gimple_stmt_iterator gsi;
gimple *stmt;
enum built_in_function start_ix;
int start_ix2;
location_t clause_loc;
vec<tree, va_gc> *args;
clauses = gimple_omp_parallel_clauses (entry_stmt);
start_ix = BUILT_IN_GOMP_PARALLEL;
if (is_combined_parallel (region))
{
switch (region->inner->type)
{
case GIMPLE_OMP_FOR:
gcc_assert (region->inner->sched_kind != OMP_CLAUSE_SCHEDULE_AUTO);
switch (region->inner->sched_kind)
{
case OMP_CLAUSE_SCHEDULE_RUNTIME:
start_ix2 = 3;
break;
case OMP_CLAUSE_SCHEDULE_DYNAMIC:
case OMP_CLAUSE_SCHEDULE_GUIDED:
if (region->inner->sched_modifiers
& OMP_CLAUSE_SCHEDULE_NONMONOTONIC)
{
start_ix2 = 3 + region->inner->sched_kind;
break;
}
default:
start_ix2 = region->inner->sched_kind;
break;
}
start_ix2 += (int) BUILT_IN_GOMP_PARALLEL_LOOP_STATIC;
start_ix = (enum built_in_function) start_ix2;
break;
case GIMPLE_OMP_SECTIONS:
start_ix = BUILT_IN_GOMP_PARALLEL_SECTIONS;
break;
default:
gcc_unreachable ();
}
}
cond = NULL_TREE;
val = build_int_cst (unsigned_type_node, 0);
flags = build_int_cst (unsigned_type_node, 0);
c = omp_find_clause (clauses, OMP_CLAUSE_IF);
if (c)
cond = OMP_CLAUSE_IF_EXPR (c);
c = omp_find_clause (clauses, OMP_CLAUSE_NUM_THREADS);
if (c)
{
val = OMP_CLAUSE_NUM_THREADS_EXPR (c);
clause_loc = OMP_CLAUSE_LOCATION (c);
}
else
clause_loc = gimple_location (entry_stmt);
c = omp_find_clause (clauses, OMP_CLAUSE_PROC_BIND);
if (c)
flags = build_int_cst (unsigned_type_node, OMP_CLAUSE_PROC_BIND_KIND (c));
val = fold_convert_loc (clause_loc, unsigned_type_node, val);
if (cond)
{
cond = gimple_boolify (cond);
if (integer_zerop (val))
val = fold_build2_loc (clause_loc,
EQ_EXPR, unsigned_type_node, cond,
build_int_cst (TREE_TYPE (cond), 0));
else
{
basic_block cond_bb, then_bb, else_bb;
edge e, e_then, e_else;
tree tmp_then, tmp_else, tmp_join, tmp_var;
tmp_var = create_tmp_var (TREE_TYPE (val));
if (gimple_in_ssa_p (cfun))
{
tmp_then = make_ssa_name (tmp_var);
tmp_else = make_ssa_name (tmp_var);
tmp_join = make_ssa_name (tmp_var);
}
else
{
tmp_then = tmp_var;
tmp_else = tmp_var;
tmp_join = tmp_var;
}
e = split_block_after_labels (bb);
cond_bb = e->src;
bb = e->dest;
remove_edge (e);
then_bb = create_empty_bb (cond_bb);
else_bb = create_empty_bb (then_bb);
set_immediate_dominator (CDI_DOMINATORS, then_bb, cond_bb);
set_immediate_dominator (CDI_DOMINATORS, else_bb, cond_bb);
stmt = gimple_build_cond_empty (cond);
gsi = gsi_start_bb (cond_bb);
gsi_insert_after (&gsi, stmt, GSI_CONTINUE_LINKING);
gsi = gsi_start_bb (then_bb);
expand_omp_build_assign (&gsi, tmp_then, val, true);
gsi = gsi_start_bb (else_bb);
expand_omp_build_assign (&gsi, tmp_else,
build_int_cst (unsigned_type_node, 1),
true);
make_edge (cond_bb, then_bb, EDGE_TRUE_VALUE);
make_edge (cond_bb, else_bb, EDGE_FALSE_VALUE);
add_bb_to_loop (then_bb, cond_bb->loop_father);
add_bb_to_loop (else_bb, cond_bb->loop_father);
e_then = make_edge (then_bb, bb, EDGE_FALLTHRU);
e_else = make_edge (else_bb, bb, EDGE_FALLTHRU);
if (gimple_in_ssa_p (cfun))
{
gphi *phi = create_phi_node (tmp_join, bb);
add_phi_arg (phi, tmp_then, e_then, UNKNOWN_LOCATION);
add_phi_arg (phi, tmp_else, e_else, UNKNOWN_LOCATION);
}
val = tmp_join;
}
gsi = gsi_start_bb (bb);
val = force_gimple_operand_gsi (&gsi, val, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
}
gsi = gsi_last_nondebug_bb (bb);
t = gimple_omp_parallel_data_arg (entry_stmt);
if (t == NULL)
t1 = null_pointer_node;
else
t1 = build_fold_addr_expr (t);
tree child_fndecl = gimple_omp_parallel_child_fn (entry_stmt);
t2 = build_fold_addr_expr (child_fndecl);
vec_alloc (args, 4 + vec_safe_length (ws_args));
args->quick_push (t2);
args->quick_push (t1);
args->quick_push (val);
if (ws_args)
args->splice (*ws_args);
args->quick_push (flags);
t = build_call_expr_loc_vec (UNKNOWN_LOCATION,
builtin_decl_explicit (start_ix), args);
force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
if (hsa_gen_requested_p ()
&& parallel_needs_hsa_kernel_p (region))
{
cgraph_node *child_cnode = cgraph_node::get (child_fndecl);
hsa_register_kernel (child_cnode);
}
}
static void
expand_task_call (struct omp_region *region, basic_block bb,
gomp_task *entry_stmt)
{
tree t1, t2, t3;
gimple_stmt_iterator gsi;
location_t loc = gimple_location (entry_stmt);
tree clauses = gimple_omp_task_clauses (entry_stmt);
tree ifc = omp_find_clause (clauses, OMP_CLAUSE_IF);
tree untied = omp_find_clause (clauses, OMP_CLAUSE_UNTIED);
tree mergeable = omp_find_clause (clauses, OMP_CLAUSE_MERGEABLE);
tree depend = omp_find_clause (clauses, OMP_CLAUSE_DEPEND);
tree finalc = omp_find_clause (clauses, OMP_CLAUSE_FINAL);
tree priority = omp_find_clause (clauses, OMP_CLAUSE_PRIORITY);
unsigned int iflags
= (untied ? GOMP_TASK_FLAG_UNTIED : 0)
| (mergeable ? GOMP_TASK_FLAG_MERGEABLE : 0)
| (depend ? GOMP_TASK_FLAG_DEPEND : 0);
bool taskloop_p = gimple_omp_task_taskloop_p (entry_stmt);
tree startvar = NULL_TREE, endvar = NULL_TREE, step = NULL_TREE;
tree num_tasks = NULL_TREE;
bool ull = false;
if (taskloop_p)
{
gimple *g = last_stmt (region->outer->entry);
gcc_assert (gimple_code (g) == GIMPLE_OMP_FOR
&& gimple_omp_for_kind (g) == GF_OMP_FOR_KIND_TASKLOOP);
struct omp_for_data fd;
omp_extract_for_data (as_a <gomp_for *> (g), &fd, NULL);
startvar = omp_find_clause (clauses, OMP_CLAUSE__LOOPTEMP_);
endvar = omp_find_clause (OMP_CLAUSE_CHAIN (startvar),
OMP_CLAUSE__LOOPTEMP_);
startvar = OMP_CLAUSE_DECL (startvar);
endvar = OMP_CLAUSE_DECL (endvar);
step = fold_convert_loc (loc, fd.iter_type, fd.loop.step);
if (fd.loop.cond_code == LT_EXPR)
iflags |= GOMP_TASK_FLAG_UP;
tree tclauses = gimple_omp_for_clauses (g);
num_tasks = omp_find_clause (tclauses, OMP_CLAUSE_NUM_TASKS);
if (num_tasks)
num_tasks = OMP_CLAUSE_NUM_TASKS_EXPR (num_tasks);
else
{
num_tasks = omp_find_clause (tclauses, OMP_CLAUSE_GRAINSIZE);
if (num_tasks)
{
iflags |= GOMP_TASK_FLAG_GRAINSIZE;
num_tasks = OMP_CLAUSE_GRAINSIZE_EXPR (num_tasks);
}
else
num_tasks = integer_zero_node;
}
num_tasks = fold_convert_loc (loc, long_integer_type_node, num_tasks);
if (ifc == NULL_TREE)
iflags |= GOMP_TASK_FLAG_IF;
if (omp_find_clause (tclauses, OMP_CLAUSE_NOGROUP))
iflags |= GOMP_TASK_FLAG_NOGROUP;
ull = fd.iter_type == long_long_unsigned_type_node;
}
else if (priority)
iflags |= GOMP_TASK_FLAG_PRIORITY;
tree flags = build_int_cst (unsigned_type_node, iflags);
tree cond = boolean_true_node;
if (ifc)
{
if (taskloop_p)
{
tree t = gimple_boolify (OMP_CLAUSE_IF_EXPR (ifc));
t = fold_build3_loc (loc, COND_EXPR, unsigned_type_node, t,
build_int_cst (unsigned_type_node,
GOMP_TASK_FLAG_IF),
build_int_cst (unsigned_type_node, 0));
flags = fold_build2_loc (loc, PLUS_EXPR, unsigned_type_node,
flags, t);
}
else
cond = gimple_boolify (OMP_CLAUSE_IF_EXPR (ifc));
}
if (finalc)
{
tree t = gimple_boolify (OMP_CLAUSE_FINAL_EXPR (finalc));
t = fold_build3_loc (loc, COND_EXPR, unsigned_type_node, t,
build_int_cst (unsigned_type_node,
GOMP_TASK_FLAG_FINAL),
build_int_cst (unsigned_type_node, 0));
flags = fold_build2_loc (loc, PLUS_EXPR, unsigned_type_node, flags, t);
}
if (depend)
depend = OMP_CLAUSE_DECL (depend);
else
depend = build_int_cst (ptr_type_node, 0);
if (priority)
priority = fold_convert (integer_type_node,
OMP_CLAUSE_PRIORITY_EXPR (priority));
else
priority = integer_zero_node;
gsi = gsi_last_nondebug_bb (bb);
tree t = gimple_omp_task_data_arg (entry_stmt);
if (t == NULL)
t2 = null_pointer_node;
else
t2 = build_fold_addr_expr_loc (loc, t);
t1 = build_fold_addr_expr_loc (loc, gimple_omp_task_child_fn (entry_stmt));
t = gimple_omp_task_copy_fn (entry_stmt);
if (t == NULL)
t3 = null_pointer_node;
else
t3 = build_fold_addr_expr_loc (loc, t);
if (taskloop_p)
t = build_call_expr (ull
? builtin_decl_explicit (BUILT_IN_GOMP_TASKLOOP_ULL)
: builtin_decl_explicit (BUILT_IN_GOMP_TASKLOOP),
11, t1, t2, t3,
gimple_omp_task_arg_size (entry_stmt),
gimple_omp_task_arg_align (entry_stmt), flags,
num_tasks, priority, startvar, endvar, step);
else
t = build_call_expr (builtin_decl_explicit (BUILT_IN_GOMP_TASK),
9, t1, t2, t3,
gimple_omp_task_arg_size (entry_stmt),
gimple_omp_task_arg_align (entry_stmt), cond, flags,
depend, priority);
force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
}
static tree
vec2chain (vec<tree, va_gc> *v)
{
tree chain = NULL_TREE, t;
unsigned ix;
FOR_EACH_VEC_SAFE_ELT_REVERSE (v, ix, t)
{
DECL_CHAIN (t) = chain;
chain = t;
}
return chain;
}
static void
remove_exit_barrier (struct omp_region *region)
{
gimple_stmt_iterator gsi;
basic_block exit_bb;
edge_iterator ei;
edge e;
gimple *stmt;
int any_addressable_vars = -1;
exit_bb = region->exit;
if (! exit_bb)
return;
gsi = gsi_last_nondebug_bb (exit_bb);
gcc_assert (gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_RETURN);
gsi_prev_nondebug (&gsi);
if (!gsi_end_p (gsi) && gimple_code (gsi_stmt (gsi)) != GIMPLE_LABEL)
return;
FOR_EACH_EDGE (e, ei, exit_bb->preds)
{
gsi = gsi_last_nondebug_bb (e->src);
if (gsi_end_p (gsi))
continue;
stmt = gsi_stmt (gsi);
if (gimple_code (stmt) == GIMPLE_OMP_RETURN
&& !gimple_omp_return_nowait_p (stmt))
{
if (any_addressable_vars < 0)
{
gomp_parallel *parallel_stmt
= as_a <gomp_parallel *> (last_stmt (region->entry));
tree child_fun = gimple_omp_parallel_child_fn (parallel_stmt);
tree local_decls, block, decl;
unsigned ix;
any_addressable_vars = 0;
FOR_EACH_LOCAL_DECL (DECL_STRUCT_FUNCTION (child_fun), ix, decl)
if (TREE_ADDRESSABLE (decl))
{
any_addressable_vars = 1;
break;
}
for (block = gimple_block (stmt);
!any_addressable_vars
&& block
&& TREE_CODE (block) == BLOCK;
block = BLOCK_SUPERCONTEXT (block))
{
for (local_decls = BLOCK_VARS (block);
local_decls;
local_decls = DECL_CHAIN (local_decls))
if (TREE_ADDRESSABLE (local_decls))
{
any_addressable_vars = 1;
break;
}
if (block == gimple_block (parallel_stmt))
break;
}
}
if (!any_addressable_vars)
gimple_omp_return_set_nowait (stmt);
}
}
}
static void
remove_exit_barriers (struct omp_region *region)
{
if (region->type == GIMPLE_OMP_PARALLEL)
remove_exit_barrier (region);
if (region->inner)
{
region = region->inner;
remove_exit_barriers (region);
while (region->next)
{
region = region->next;
remove_exit_barriers (region);
}
}
}
static void
optimize_omp_library_calls (gimple *entry_stmt)
{
basic_block bb;
gimple_stmt_iterator gsi;
tree thr_num_tree = builtin_decl_explicit (BUILT_IN_OMP_GET_THREAD_NUM);
tree thr_num_id = DECL_ASSEMBLER_NAME (thr_num_tree);
tree num_thr_tree = builtin_decl_explicit (BUILT_IN_OMP_GET_NUM_THREADS);
tree num_thr_id = DECL_ASSEMBLER_NAME (num_thr_tree);
bool untied_task = (gimple_code (entry_stmt) == GIMPLE_OMP_TASK
&& omp_find_clause (gimple_omp_task_clauses (entry_stmt),
OMP_CLAUSE_UNTIED) != NULL);
FOR_EACH_BB_FN (bb, cfun)
for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi); gsi_next (&gsi))
{
gimple *call = gsi_stmt (gsi);
tree decl;
if (is_gimple_call (call)
&& (decl = gimple_call_fndecl (call))
&& DECL_EXTERNAL (decl)
&& TREE_PUBLIC (decl)
&& DECL_INITIAL (decl) == NULL)
{
tree built_in;
if (DECL_NAME (decl) == thr_num_id)
{
if (untied_task)
continue;
built_in = builtin_decl_explicit (BUILT_IN_OMP_GET_THREAD_NUM);
}
else if (DECL_NAME (decl) == num_thr_id)
built_in = builtin_decl_explicit (BUILT_IN_OMP_GET_NUM_THREADS);
else
continue;
if (DECL_ASSEMBLER_NAME (decl) != DECL_ASSEMBLER_NAME (built_in)
|| gimple_call_num_args (call) != 0)
continue;
if (flag_exceptions && !TREE_NOTHROW (decl))
continue;
if (TREE_CODE (TREE_TYPE (decl)) != FUNCTION_TYPE
|| !types_compatible_p (TREE_TYPE (TREE_TYPE (decl)),
TREE_TYPE (TREE_TYPE (built_in))))
continue;
gimple_call_set_fndecl (call, built_in);
}
}
}
static tree
expand_omp_regimplify_p (tree *tp, int *walk_subtrees, void *)
{
tree t = *tp;
if (VAR_P (t) && DECL_HAS_VALUE_EXPR_P (t))
return t;
if (TREE_CODE (t) == ADDR_EXPR)
recompute_tree_invariant_for_addr_expr (t);
*walk_subtrees = !TYPE_P (t) && !DECL_P (t);
return NULL_TREE;
}
static void
expand_omp_build_assign (gimple_stmt_iterator *gsi_p, tree to, tree from,
bool after)
{
bool simple_p = DECL_P (to) && TREE_ADDRESSABLE (to);
from = force_gimple_operand_gsi (gsi_p, from, simple_p, NULL_TREE,
!after, after ? GSI_CONTINUE_LINKING
: GSI_SAME_STMT);
gimple *stmt = gimple_build_assign (to, from);
if (after)
gsi_insert_after (gsi_p, stmt, GSI_CONTINUE_LINKING);
else
gsi_insert_before (gsi_p, stmt, GSI_SAME_STMT);
if (walk_tree (&from, expand_omp_regimplify_p, NULL, NULL)
|| walk_tree (&to, expand_omp_regimplify_p, NULL, NULL))
{
gimple_stmt_iterator gsi = gsi_for_stmt (stmt);
gimple_regimplify_operands (stmt, &gsi);
}
}
static void
expand_omp_taskreg (struct omp_region *region)
{
basic_block entry_bb, exit_bb, new_bb;
struct function *child_cfun;
tree child_fn, block, t;
gimple_stmt_iterator gsi;
gimple *entry_stmt, *stmt;
edge e;
vec<tree, va_gc> *ws_args;
entry_stmt = last_stmt (region->entry);
child_fn = gimple_omp_taskreg_child_fn (entry_stmt);
child_cfun = DECL_STRUCT_FUNCTION (child_fn);
entry_bb = region->entry;
if (gimple_code (entry_stmt) == GIMPLE_OMP_TASK)
exit_bb = region->cont;
else
exit_bb = region->exit;
if (is_combined_parallel (region))
ws_args = region->ws_args;
else
ws_args = NULL;
if (child_cfun->cfg)
{
edge entry_succ_e, exit_succ_e;
entry_succ_e = single_succ_edge (entry_bb);
gsi = gsi_last_nondebug_bb (entry_bb);
gcc_assert (gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_PARALLEL
|| gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_TASK);
gsi_remove (&gsi, true);
new_bb = entry_bb;
if (exit_bb)
{
exit_succ_e = single_succ_edge (exit_bb);
make_edge (new_bb, exit_succ_e->dest, EDGE_FALLTHRU);
}
remove_edge_and_dominated_blocks (entry_succ_e);
}
else
{
unsigned srcidx, dstidx, num;
if (gimple_omp_taskreg_data_arg (entry_stmt))
{
basic_block entry_succ_bb
= single_succ_p (entry_bb) ? single_succ (entry_bb)
: FALLTHRU_EDGE (entry_bb)->dest;
tree arg;
gimple *parcopy_stmt = NULL;
for (gsi = gsi_start_bb (entry_succ_bb); ; gsi_next (&gsi))
{
gimple *stmt;
gcc_assert (!gsi_end_p (gsi));
stmt = gsi_stmt (gsi);
if (gimple_code (stmt) != GIMPLE_ASSIGN)
continue;
if (gimple_num_ops (stmt) == 2)
{
tree arg = gimple_assign_rhs1 (stmt);
if (TREE_CODE (arg) == ADDR_EXPR
&& TREE_OPERAND (arg, 0)
== gimple_omp_taskreg_data_arg (entry_stmt))
{
parcopy_stmt = stmt;
break;
}
}
}
gcc_assert (parcopy_stmt != NULL);
arg = DECL_ARGUMENTS (child_fn);
if (!gimple_in_ssa_p (cfun))
{
if (gimple_assign_lhs (parcopy_stmt) == arg)
gsi_remove (&gsi, true);
else
{
gimple_omp_set_subcode (parcopy_stmt, TREE_CODE (arg));
gimple_assign_set_rhs1 (parcopy_stmt, arg);
}
}
else
{
tree lhs = gimple_assign_lhs (parcopy_stmt);
gcc_assert (SSA_NAME_VAR (lhs) == arg);
gimple_assign_set_rhs1 (parcopy_stmt, arg);
update_stmt (parcopy_stmt);
}
}
block = DECL_INITIAL (child_fn);
BLOCK_VARS (block) = vec2chain (child_cfun->local_decls);
for (t = BLOCK_VARS (block); t; t = DECL_CHAIN (t))
if (VAR_P (t) && TREE_STATIC (t) && !DECL_EXTERNAL (t))
varpool_node::finalize_decl (t);
DECL_SAVED_TREE (child_fn) = NULL;
gimple_set_body (child_fn, NULL);
TREE_USED (block) = 1;
for (t = DECL_ARGUMENTS (child_fn); t; t = DECL_CHAIN (t))
DECL_CONTEXT (t) = child_fn;
gsi = gsi_last_nondebug_bb (entry_bb);
stmt = gsi_stmt (gsi);
gcc_assert (stmt && (gimple_code (stmt) == GIMPLE_OMP_PARALLEL
|| gimple_code (stmt) == GIMPLE_OMP_TASK));
e = split_block (entry_bb, stmt);
gsi_remove (&gsi, true);
entry_bb = e->dest;
edge e2 = NULL;
if (gimple_code (entry_stmt) == GIMPLE_OMP_PARALLEL)
single_succ_edge (entry_bb)->flags = EDGE_FALLTHRU;
else
{
e2 = make_edge (e->src, BRANCH_EDGE (entry_bb)->dest, EDGE_ABNORMAL);
gcc_assert (e2->dest == region->exit);
remove_edge (BRANCH_EDGE (entry_bb));
set_immediate_dominator (CDI_DOMINATORS, e2->dest, e->src);
gsi = gsi_last_nondebug_bb (region->exit);
gcc_assert (!gsi_end_p (gsi)
&& gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_RETURN);
gsi_remove (&gsi, true);
}
if (exit_bb)
{
gsi = gsi_last_nondebug_bb (exit_bb);
gcc_assert (!gsi_end_p (gsi)
&& (gimple_code (gsi_stmt (gsi))
== (e2 ? GIMPLE_OMP_CONTINUE : GIMPLE_OMP_RETURN)));
stmt = gimple_build_return (NULL);
gsi_insert_after (&gsi, stmt, GSI_SAME_STMT);
gsi_remove (&gsi, true);
}
if (gimple_in_ssa_p (cfun))
{
init_tree_ssa (child_cfun);
init_ssa_operands (child_cfun);
child_cfun->gimple_df->in_ssa_p = true;
block = NULL_TREE;
}
else
block = gimple_block (entry_stmt);
new_bb = move_sese_region_to_fn (child_cfun, entry_bb, exit_bb, block);
if (exit_bb)
single_succ_edge (new_bb)->flags = EDGE_FALLTHRU;
if (e2)
{
basic_block dest_bb = e2->dest;
if (!exit_bb)
make_edge (new_bb, dest_bb, EDGE_FALLTHRU);
remove_edge (e2);
set_immediate_dominator (CDI_DOMINATORS, dest_bb, new_bb);
}
if (loops_state_satisfies_p (LOOPS_NEED_FIXUP))
child_cfun->x_current_loops->state |= LOOPS_NEED_FIXUP;
num = vec_safe_length (child_cfun->local_decls);
for (srcidx = 0, dstidx = 0; srcidx < num; srcidx++)
{
t = (*child_cfun->local_decls)[srcidx];
if (DECL_CONTEXT (t) == cfun->decl)
continue;
if (srcidx != dstidx)
(*child_cfun->local_decls)[dstidx] = t;
dstidx++;
}
if (dstidx != num)
vec_safe_truncate (child_cfun->local_decls, dstidx);
child_cfun->curr_properties = cfun->curr_properties;
child_cfun->has_simduid_loops |= cfun->has_simduid_loops;
child_cfun->has_force_vectorize_loops |= cfun->has_force_vectorize_loops;
cgraph_node *node = cgraph_node::get_create (child_fn);
node->parallelized_function = 1;
cgraph_node::add_new_function (child_fn, true);
bool need_asm = DECL_ASSEMBLER_NAME_SET_P (current_function_decl)
&& !DECL_ASSEMBLER_NAME_SET_P (child_fn);
push_cfun (child_cfun);
if (need_asm)
assign_assembler_name_if_needed (child_fn);
if (optimize)
optimize_omp_library_calls (entry_stmt);
update_max_bb_count ();
cgraph_edge::rebuild_edges ();
if (flag_exceptions)
{
basic_block bb;
bool changed = false;
FOR_EACH_BB_FN (bb, cfun)
changed |= gimple_purge_dead_eh_edges (bb);
if (changed)
cleanup_tree_cfg ();
}
if (gimple_in_ssa_p (cfun))
update_ssa (TODO_update_ssa);
if (flag_checking && !loops_state_satisfies_p (LOOPS_NEED_FIXUP))
verify_loop_structure ();
pop_cfun ();
if (dump_file && !gimple_in_ssa_p (cfun))
{
omp_any_child_fn_dumped = true;
dump_function_header (dump_file, child_fn, dump_flags);
dump_function_to_file (child_fn, dump_file, dump_flags);
}
}
adjust_context_and_scope (region, gimple_block (entry_stmt), child_fn);
if (gimple_code (entry_stmt) == GIMPLE_OMP_PARALLEL)
expand_parallel_call (region, new_bb,
as_a <gomp_parallel *> (entry_stmt), ws_args);
else
expand_task_call (region, new_bb, as_a <gomp_task *> (entry_stmt));
if (gimple_in_ssa_p (cfun))
update_ssa (TODO_update_ssa_only_virtuals);
}
struct oacc_collapse
{
tree base;  
tree iters; 
tree step;  
tree tile;  
tree outer; 
};
static tree
expand_oacc_collapse_init (const struct omp_for_data *fd,
gimple_stmt_iterator *gsi,
oacc_collapse *counts, tree bound_type,
location_t loc)
{
tree tiling = fd->tiling;
tree total = build_int_cst (bound_type, 1);
int ix;
gcc_assert (integer_onep (fd->loop.step));
gcc_assert (integer_zerop (fd->loop.n1));
for (ix = fd->collapse; ix--;)
{
const omp_for_data_loop *loop = &fd->loops[ix];
tree iter_type = TREE_TYPE (loop->v);
tree diff_type = iter_type;
tree plus_type = iter_type;
gcc_assert (loop->cond_code == fd->loop.cond_code);
if (POINTER_TYPE_P (iter_type))
plus_type = sizetype;
if (POINTER_TYPE_P (diff_type) || TYPE_UNSIGNED (diff_type))
diff_type = signed_type_for (diff_type);
if (TYPE_PRECISION (diff_type) < TYPE_PRECISION (integer_type_node))
diff_type = integer_type_node;
if (tiling)
{
tree num = build_int_cst (integer_type_node, fd->collapse);
tree loop_no = build_int_cst (integer_type_node, ix);
tree tile = TREE_VALUE (tiling);
gcall *call
= gimple_build_call_internal (IFN_GOACC_TILE, 5, num, loop_no, tile,
integer_zero_node,
integer_zero_node);
counts[ix].outer = create_tmp_var (iter_type, ".outer");
counts[ix].tile = create_tmp_var (diff_type, ".tile");
gimple_call_set_lhs (call, counts[ix].tile);
gimple_set_location (call, loc);
gsi_insert_before (gsi, call, GSI_SAME_STMT);
tiling = TREE_CHAIN (tiling);
}
else
{
counts[ix].tile = NULL;
counts[ix].outer = loop->v;
}
tree b = loop->n1;
tree e = loop->n2;
tree s = loop->step;
bool up = loop->cond_code == LT_EXPR;
tree dir = build_int_cst (diff_type, up ? +1 : -1);
bool negating;
tree expr;
b = force_gimple_operand_gsi (gsi, b, true, NULL_TREE,
true, GSI_SAME_STMT);
e = force_gimple_operand_gsi (gsi, e, true, NULL_TREE,
true, GSI_SAME_STMT);
negating = !up && TYPE_UNSIGNED (TREE_TYPE (s));
if (negating)
s = fold_build1 (NEGATE_EXPR, TREE_TYPE (s), s);
s = fold_convert (diff_type, s);
if (negating)
s = fold_build1 (NEGATE_EXPR, diff_type, s);
s = force_gimple_operand_gsi (gsi, s, true, NULL_TREE,
true, GSI_SAME_STMT);
negating = !up && TYPE_UNSIGNED (iter_type);
expr = fold_build2 (MINUS_EXPR, plus_type,
fold_convert (plus_type, negating ? b : e),
fold_convert (plus_type, negating ? e : b));
expr = fold_convert (diff_type, expr);
if (negating)
expr = fold_build1 (NEGATE_EXPR, diff_type, expr);
tree range = force_gimple_operand_gsi
(gsi, expr, true, NULL_TREE, true, GSI_SAME_STMT);
expr = fold_build2 (MINUS_EXPR, diff_type, range, dir);
expr = fold_build2 (PLUS_EXPR, diff_type, expr, s);
expr = fold_build2 (TRUNC_DIV_EXPR, diff_type, expr, s);
tree iters = force_gimple_operand_gsi (gsi, expr, true, NULL_TREE,
true, GSI_SAME_STMT);
counts[ix].base = b;
counts[ix].iters = iters;
counts[ix].step = s;
total = fold_build2 (MULT_EXPR, bound_type, total,
fold_convert (bound_type, iters));
}
return total;
}
static void
expand_oacc_collapse_vars (const struct omp_for_data *fd, bool inner,
gimple_stmt_iterator *gsi,
const oacc_collapse *counts, tree ivar)
{
tree ivar_type = TREE_TYPE (ivar);
for (int ix = fd->collapse; ix--;)
{
const omp_for_data_loop *loop = &fd->loops[ix];
const oacc_collapse *collapse = &counts[ix];
tree v = inner ? loop->v : collapse->outer;
tree iter_type = TREE_TYPE (v);
tree diff_type = TREE_TYPE (collapse->step);
tree plus_type = iter_type;
enum tree_code plus_code = PLUS_EXPR;
tree expr;
if (POINTER_TYPE_P (iter_type))
{
plus_code = POINTER_PLUS_EXPR;
plus_type = sizetype;
}
expr = ivar;
if (ix)
{
tree mod = fold_convert (ivar_type, collapse->iters);
ivar = fold_build2 (TRUNC_DIV_EXPR, ivar_type, expr, mod);
expr = fold_build2 (TRUNC_MOD_EXPR, ivar_type, expr, mod);
ivar = force_gimple_operand_gsi (gsi, ivar, true, NULL_TREE,
true, GSI_SAME_STMT);
}
expr = fold_build2 (MULT_EXPR, diff_type, fold_convert (diff_type, expr),
collapse->step);
expr = fold_build2 (plus_code, iter_type,
inner ? collapse->outer : collapse->base,
fold_convert (plus_type, expr));
expr = force_gimple_operand_gsi (gsi, expr, false, NULL_TREE,
true, GSI_SAME_STMT);
gassign *ass = gimple_build_assign (v, expr);
gsi_insert_before (gsi, ass, GSI_SAME_STMT);
}
}
static void
expand_omp_for_init_counts (struct omp_for_data *fd, gimple_stmt_iterator *gsi,
basic_block &entry_bb, tree *counts,
basic_block &zero_iter1_bb, int &first_zero_iter1,
basic_block &zero_iter2_bb, int &first_zero_iter2,
basic_block &l2_dom_bb)
{
tree t, type = TREE_TYPE (fd->loop.v);
edge e, ne;
int i;
gcc_assert (!gimple_in_ssa_p (cfun));
if (gimple_omp_for_combined_into_p (fd->for_stmt)
&& TREE_CODE (fd->loop.n2) != INTEGER_CST)
{
gcc_assert (fd->ordered == 0);
tree innerc = omp_find_clause (gimple_omp_for_clauses (fd->for_stmt),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
for (i = 0; i < fd->collapse; i++)
{
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
if (i)
counts[i] = OMP_CLAUSE_DECL (innerc);
else
counts[0] = NULL_TREE;
}
return;
}
for (i = fd->collapse; i < fd->ordered; i++)
{
tree itype = TREE_TYPE (fd->loops[i].v);
counts[i] = NULL_TREE;
t = fold_binary (fd->loops[i].cond_code, boolean_type_node,
fold_convert (itype, fd->loops[i].n1),
fold_convert (itype, fd->loops[i].n2));
if (t && integer_zerop (t))
{
for (i = fd->collapse; i < fd->ordered; i++)
counts[i] = build_int_cst (type, 0);
break;
}
}
for (i = 0; i < (fd->ordered ? fd->ordered : fd->collapse); i++)
{
tree itype = TREE_TYPE (fd->loops[i].v);
if (i >= fd->collapse && counts[i])
continue;
if ((SSA_VAR_P (fd->loop.n2) || i >= fd->collapse)
&& ((t = fold_binary (fd->loops[i].cond_code, boolean_type_node,
fold_convert (itype, fd->loops[i].n1),
fold_convert (itype, fd->loops[i].n2)))
== NULL_TREE || !integer_onep (t)))
{
gcond *cond_stmt;
tree n1, n2;
n1 = fold_convert (itype, unshare_expr (fd->loops[i].n1));
n1 = force_gimple_operand_gsi (gsi, n1, true, NULL_TREE,
true, GSI_SAME_STMT);
n2 = fold_convert (itype, unshare_expr (fd->loops[i].n2));
n2 = force_gimple_operand_gsi (gsi, n2, true, NULL_TREE,
true, GSI_SAME_STMT);
cond_stmt = gimple_build_cond (fd->loops[i].cond_code, n1, n2,
NULL_TREE, NULL_TREE);
gsi_insert_before (gsi, cond_stmt, GSI_SAME_STMT);
if (walk_tree (gimple_cond_lhs_ptr (cond_stmt),
expand_omp_regimplify_p, NULL, NULL)
|| walk_tree (gimple_cond_rhs_ptr (cond_stmt),
expand_omp_regimplify_p, NULL, NULL))
{
*gsi = gsi_for_stmt (cond_stmt);
gimple_regimplify_operands (cond_stmt, gsi);
}
e = split_block (entry_bb, cond_stmt);
basic_block &zero_iter_bb
= i < fd->collapse ? zero_iter1_bb : zero_iter2_bb;
int &first_zero_iter
= i < fd->collapse ? first_zero_iter1 : first_zero_iter2;
if (zero_iter_bb == NULL)
{
gassign *assign_stmt;
first_zero_iter = i;
zero_iter_bb = create_empty_bb (entry_bb);
add_bb_to_loop (zero_iter_bb, entry_bb->loop_father);
*gsi = gsi_after_labels (zero_iter_bb);
if (i < fd->collapse)
assign_stmt = gimple_build_assign (fd->loop.n2,
build_zero_cst (type));
else
{
counts[i] = create_tmp_reg (type, ".count");
assign_stmt
= gimple_build_assign (counts[i], build_zero_cst (type));
}
gsi_insert_before (gsi, assign_stmt, GSI_SAME_STMT);
set_immediate_dominator (CDI_DOMINATORS, zero_iter_bb,
entry_bb);
}
ne = make_edge (entry_bb, zero_iter_bb, EDGE_FALSE_VALUE);
ne->probability = profile_probability::very_unlikely ();
e->flags = EDGE_TRUE_VALUE;
e->probability = ne->probability.invert ();
if (l2_dom_bb == NULL)
l2_dom_bb = entry_bb;
entry_bb = e->dest;
*gsi = gsi_last_nondebug_bb (entry_bb);
}
if (POINTER_TYPE_P (itype))
itype = signed_type_for (itype);
t = build_int_cst (itype, (fd->loops[i].cond_code == LT_EXPR
? -1 : 1));
t = fold_build2 (PLUS_EXPR, itype,
fold_convert (itype, fd->loops[i].step), t);
t = fold_build2 (PLUS_EXPR, itype, t,
fold_convert (itype, fd->loops[i].n2));
t = fold_build2 (MINUS_EXPR, itype, t,
fold_convert (itype, fd->loops[i].n1));
if (TYPE_UNSIGNED (itype) && fd->loops[i].cond_code == GT_EXPR)
t = fold_build2 (TRUNC_DIV_EXPR, itype,
fold_build1 (NEGATE_EXPR, itype, t),
fold_build1 (NEGATE_EXPR, itype,
fold_convert (itype,
fd->loops[i].step)));
else
t = fold_build2 (TRUNC_DIV_EXPR, itype, t,
fold_convert (itype, fd->loops[i].step));
t = fold_convert (type, t);
if (TREE_CODE (t) == INTEGER_CST)
counts[i] = t;
else
{
if (i < fd->collapse || i != first_zero_iter2)
counts[i] = create_tmp_reg (type, ".count");
expand_omp_build_assign (gsi, counts[i], t);
}
if (SSA_VAR_P (fd->loop.n2) && i < fd->collapse)
{
if (i == 0)
t = counts[0];
else
t = fold_build2 (MULT_EXPR, type, fd->loop.n2, counts[i]);
expand_omp_build_assign (gsi, fd->loop.n2, t);
}
}
}
static void
expand_omp_for_init_vars (struct omp_for_data *fd, gimple_stmt_iterator *gsi,
tree *counts, gimple *inner_stmt, tree startvar)
{
int i;
if (gimple_omp_for_combined_p (fd->for_stmt))
{
if (TREE_CODE (fd->loop.n2) == INTEGER_CST)
return;
tree clauses = gimple_code (inner_stmt) != GIMPLE_OMP_FOR
? gimple_omp_taskreg_clauses (inner_stmt)
: gimple_omp_for_clauses (inner_stmt);
tree innerc = omp_find_clause (clauses, OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
for (i = 0; i < fd->collapse; i++)
{
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
if (i)
{
tree tem = OMP_CLAUSE_DECL (innerc);
tree t = fold_convert (TREE_TYPE (tem), counts[i]);
t = force_gimple_operand_gsi (gsi, t, false, NULL_TREE,
false, GSI_CONTINUE_LINKING);
gassign *stmt = gimple_build_assign (tem, t);
gsi_insert_after (gsi, stmt, GSI_CONTINUE_LINKING);
}
}
return;
}
tree type = TREE_TYPE (fd->loop.v);
tree tem = create_tmp_reg (type, ".tem");
gassign *stmt = gimple_build_assign (tem, startvar);
gsi_insert_after (gsi, stmt, GSI_CONTINUE_LINKING);
for (i = fd->collapse - 1; i >= 0; i--)
{
tree vtype = TREE_TYPE (fd->loops[i].v), itype, t;
itype = vtype;
if (POINTER_TYPE_P (vtype))
itype = signed_type_for (vtype);
if (i != 0)
t = fold_build2 (TRUNC_MOD_EXPR, type, tem, counts[i]);
else
t = tem;
t = fold_convert (itype, t);
t = fold_build2 (MULT_EXPR, itype, t,
fold_convert (itype, fd->loops[i].step));
if (POINTER_TYPE_P (vtype))
t = fold_build_pointer_plus (fd->loops[i].n1, t);
else
t = fold_build2 (PLUS_EXPR, itype, fd->loops[i].n1, t);
t = force_gimple_operand_gsi (gsi, t,
DECL_P (fd->loops[i].v)
&& TREE_ADDRESSABLE (fd->loops[i].v),
NULL_TREE, false,
GSI_CONTINUE_LINKING);
stmt = gimple_build_assign (fd->loops[i].v, t);
gsi_insert_after (gsi, stmt, GSI_CONTINUE_LINKING);
if (i != 0)
{
t = fold_build2 (TRUNC_DIV_EXPR, type, tem, counts[i]);
t = force_gimple_operand_gsi (gsi, t, false, NULL_TREE,
false, GSI_CONTINUE_LINKING);
stmt = gimple_build_assign (tem, t);
gsi_insert_after (gsi, stmt, GSI_CONTINUE_LINKING);
}
}
}
static basic_block
extract_omp_for_update_vars (struct omp_for_data *fd, basic_block cont_bb,
basic_block body_bb)
{
basic_block last_bb, bb, collapse_bb = NULL;
int i;
gimple_stmt_iterator gsi;
edge e;
tree t;
gimple *stmt;
last_bb = cont_bb;
for (i = fd->collapse - 1; i >= 0; i--)
{
tree vtype = TREE_TYPE (fd->loops[i].v);
bb = create_empty_bb (last_bb);
add_bb_to_loop (bb, last_bb->loop_father);
gsi = gsi_start_bb (bb);
if (i < fd->collapse - 1)
{
e = make_edge (last_bb, bb, EDGE_FALSE_VALUE);
e->probability = profile_probability::guessed_always ().apply_scale (1, 8);
t = fd->loops[i + 1].n1;
t = force_gimple_operand_gsi (&gsi, t,
DECL_P (fd->loops[i + 1].v)
&& TREE_ADDRESSABLE (fd->loops[i
+ 1].v),
NULL_TREE, false,
GSI_CONTINUE_LINKING);
stmt = gimple_build_assign (fd->loops[i + 1].v, t);
gsi_insert_after (&gsi, stmt, GSI_CONTINUE_LINKING);
}
else
collapse_bb = bb;
set_immediate_dominator (CDI_DOMINATORS, bb, last_bb);
if (POINTER_TYPE_P (vtype))
t = fold_build_pointer_plus (fd->loops[i].v, fd->loops[i].step);
else
t = fold_build2 (PLUS_EXPR, vtype, fd->loops[i].v, fd->loops[i].step);
t = force_gimple_operand_gsi (&gsi, t,
DECL_P (fd->loops[i].v)
&& TREE_ADDRESSABLE (fd->loops[i].v),
NULL_TREE, false, GSI_CONTINUE_LINKING);
stmt = gimple_build_assign (fd->loops[i].v, t);
gsi_insert_after (&gsi, stmt, GSI_CONTINUE_LINKING);
if (i > 0)
{
t = fd->loops[i].n2;
t = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
tree v = fd->loops[i].v;
if (DECL_P (v) && TREE_ADDRESSABLE (v))
v = force_gimple_operand_gsi (&gsi, v, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
t = fold_build2 (fd->loops[i].cond_code, boolean_type_node, v, t);
stmt = gimple_build_cond_empty (t);
gsi_insert_after (&gsi, stmt, GSI_CONTINUE_LINKING);
if (walk_tree (gimple_cond_lhs_ptr (as_a <gcond *> (stmt)),
expand_omp_regimplify_p, NULL, NULL)
|| walk_tree (gimple_cond_rhs_ptr (as_a <gcond *> (stmt)),
expand_omp_regimplify_p, NULL, NULL))
gimple_regimplify_operands (stmt, &gsi);
e = make_edge (bb, body_bb, EDGE_TRUE_VALUE);
e->probability = profile_probability::guessed_always ().apply_scale (7, 8);
}
else
make_edge (bb, body_bb, EDGE_FALLTHRU);
last_bb = bb;
}
return collapse_bb;
}
static void
expand_omp_ordered_source (gimple_stmt_iterator *gsi, struct omp_for_data *fd,
tree *counts, location_t loc)
{
enum built_in_function source_ix
= fd->iter_type == long_integer_type_node
? BUILT_IN_GOMP_DOACROSS_POST : BUILT_IN_GOMP_DOACROSS_ULL_POST;
gimple *g
= gimple_build_call (builtin_decl_explicit (source_ix), 1,
build_fold_addr_expr (counts[fd->ordered]));
gimple_set_location (g, loc);
gsi_insert_before (gsi, g, GSI_SAME_STMT);
}
static void
expand_omp_ordered_sink (gimple_stmt_iterator *gsi, struct omp_for_data *fd,
tree *counts, tree c, location_t loc)
{
auto_vec<tree, 10> args;
enum built_in_function sink_ix
= fd->iter_type == long_integer_type_node
? BUILT_IN_GOMP_DOACROSS_WAIT : BUILT_IN_GOMP_DOACROSS_ULL_WAIT;
tree t, off, coff = NULL_TREE, deps = OMP_CLAUSE_DECL (c), cond = NULL_TREE;
int i;
gimple_stmt_iterator gsi2 = *gsi;
bool warned_step = false;
for (i = 0; i < fd->ordered; i++)
{
tree step = NULL_TREE;
off = TREE_PURPOSE (deps);
if (TREE_CODE (off) == TRUNC_DIV_EXPR)
{
step = TREE_OPERAND (off, 1);
off = TREE_OPERAND (off, 0);
}
if (!integer_zerop (off))
{
gcc_assert (fd->loops[i].cond_code == LT_EXPR
|| fd->loops[i].cond_code == GT_EXPR);
bool forward = fd->loops[i].cond_code == LT_EXPR;
if (step)
{
if (TREE_CODE (step) != INTEGER_CST)
break;
forward = tree_int_cst_sgn (step) != -1;
}
if (forward ^ OMP_CLAUSE_DEPEND_SINK_NEGATIVE (deps))
warning_at (loc, 0, "%<depend(sink)%> clause waiting for "
"lexically later iteration");
break;
}
deps = TREE_CHAIN (deps);
}
deps = OMP_CLAUSE_DECL (c);
gsi_prev (&gsi2);
edge e1 = split_block (gsi_bb (gsi2), gsi_stmt (gsi2));
edge e2 = split_block_after_labels (e1->dest);
gsi2 = gsi_after_labels (e1->dest);
*gsi = gsi_last_bb (e1->src);
for (i = 0; i < fd->ordered; i++)
{
tree itype = TREE_TYPE (fd->loops[i].v);
tree step = NULL_TREE;
tree orig_off = NULL_TREE;
if (POINTER_TYPE_P (itype))
itype = sizetype;
if (i)
deps = TREE_CHAIN (deps);
off = TREE_PURPOSE (deps);
if (TREE_CODE (off) == TRUNC_DIV_EXPR)
{
step = TREE_OPERAND (off, 1);
off = TREE_OPERAND (off, 0);
gcc_assert (fd->loops[i].cond_code == LT_EXPR
&& integer_onep (fd->loops[i].step)
&& !POINTER_TYPE_P (TREE_TYPE (fd->loops[i].v)));
}
tree s = fold_convert_loc (loc, itype, step ? step : fd->loops[i].step);
if (step)
{
off = fold_convert_loc (loc, itype, off);
orig_off = off;
off = fold_build2_loc (loc, TRUNC_DIV_EXPR, itype, off, s);
}
if (integer_zerop (off))
t = boolean_true_node;
else
{
tree a;
tree co = fold_convert_loc (loc, itype, off);
if (POINTER_TYPE_P (TREE_TYPE (fd->loops[i].v)))
{
if (OMP_CLAUSE_DEPEND_SINK_NEGATIVE (deps))
co = fold_build1_loc (loc, NEGATE_EXPR, itype, co);
a = fold_build2_loc (loc, POINTER_PLUS_EXPR,
TREE_TYPE (fd->loops[i].v), fd->loops[i].v,
co);
}
else if (OMP_CLAUSE_DEPEND_SINK_NEGATIVE (deps))
a = fold_build2_loc (loc, MINUS_EXPR, TREE_TYPE (fd->loops[i].v),
fd->loops[i].v, co);
else
a = fold_build2_loc (loc, PLUS_EXPR, TREE_TYPE (fd->loops[i].v),
fd->loops[i].v, co);
if (step)
{
tree t1, t2;
if (OMP_CLAUSE_DEPEND_SINK_NEGATIVE (deps))
t1 = fold_build2_loc (loc, GE_EXPR, boolean_type_node, a,
fd->loops[i].n1);
else
t1 = fold_build2_loc (loc, LT_EXPR, boolean_type_node, a,
fd->loops[i].n2);
if (OMP_CLAUSE_DEPEND_SINK_NEGATIVE (deps))
t2 = fold_build2_loc (loc, LT_EXPR, boolean_type_node, a,
fd->loops[i].n2);
else
t2 = fold_build2_loc (loc, GE_EXPR, boolean_type_node, a,
fd->loops[i].n1);
t = fold_build2_loc (loc, LT_EXPR, boolean_type_node,
step, build_int_cst (TREE_TYPE (step), 0));
if (TREE_CODE (step) != INTEGER_CST)
{
t1 = unshare_expr (t1);
t1 = force_gimple_operand_gsi (gsi, t1, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
t2 = unshare_expr (t2);
t2 = force_gimple_operand_gsi (gsi, t2, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
}
t = fold_build3_loc (loc, COND_EXPR, boolean_type_node,
t, t2, t1);
}
else if (fd->loops[i].cond_code == LT_EXPR)
{
if (OMP_CLAUSE_DEPEND_SINK_NEGATIVE (deps))
t = fold_build2_loc (loc, GE_EXPR, boolean_type_node, a,
fd->loops[i].n1);
else
t = fold_build2_loc (loc, LT_EXPR, boolean_type_node, a,
fd->loops[i].n2);
}
else if (OMP_CLAUSE_DEPEND_SINK_NEGATIVE (deps))
t = fold_build2_loc (loc, GT_EXPR, boolean_type_node, a,
fd->loops[i].n2);
else
t = fold_build2_loc (loc, LE_EXPR, boolean_type_node, a,
fd->loops[i].n1);
}
if (cond)
cond = fold_build2_loc (loc, BIT_AND_EXPR, boolean_type_node, cond, t);
else
cond = t;
off = fold_convert_loc (loc, itype, off);
if (step
|| (fd->loops[i].cond_code == LT_EXPR
? !integer_onep (fd->loops[i].step)
: !integer_minus_onep (fd->loops[i].step)))
{
if (step == NULL_TREE
&& TYPE_UNSIGNED (itype)
&& fd->loops[i].cond_code == GT_EXPR)
t = fold_build2_loc (loc, TRUNC_MOD_EXPR, itype, off,
fold_build1_loc (loc, NEGATE_EXPR, itype,
s));
else
t = fold_build2_loc (loc, TRUNC_MOD_EXPR, itype,
orig_off ? orig_off : off, s);
t = fold_build2_loc (loc, EQ_EXPR, boolean_type_node, t,
build_int_cst (itype, 0));
if (integer_zerop (t) && !warned_step)
{
warning_at (loc, 0, "%<depend(sink)%> refers to iteration never "
"in the iteration space");
warned_step = true;
}
cond = fold_build2_loc (loc, BIT_AND_EXPR, boolean_type_node,
cond, t);
}
if (i <= fd->collapse - 1 && fd->collapse > 1)
t = fd->loop.v;
else if (counts[i])
t = counts[i];
else
{
t = fold_build2_loc (loc, MINUS_EXPR, TREE_TYPE (fd->loops[i].v),
fd->loops[i].v, fd->loops[i].n1);
t = fold_convert_loc (loc, fd->iter_type, t);
}
if (step)
;
else if (TYPE_UNSIGNED (itype) && fd->loops[i].cond_code == GT_EXPR)
off = fold_build2_loc (loc, TRUNC_DIV_EXPR, itype, off,
fold_build1_loc (loc, NEGATE_EXPR, itype,
s));
else
off = fold_build2_loc (loc, TRUNC_DIV_EXPR, itype, off, s);
if (OMP_CLAUSE_DEPEND_SINK_NEGATIVE (deps))
off = fold_build1_loc (loc, NEGATE_EXPR, itype, off);
off = fold_convert_loc (loc, fd->iter_type, off);
if (i <= fd->collapse - 1 && fd->collapse > 1)
{
if (i)
off = fold_build2_loc (loc, PLUS_EXPR, fd->iter_type, coff,
off);
if (i < fd->collapse - 1)
{
coff = fold_build2_loc (loc, MULT_EXPR, fd->iter_type, off,
counts[i]);
continue;
}
}
off = unshare_expr (off);
t = fold_build2_loc (loc, PLUS_EXPR, fd->iter_type, t, off);
t = force_gimple_operand_gsi (&gsi2, t, true, NULL_TREE,
true, GSI_SAME_STMT);
args.safe_push (t);
}
gimple *g = gimple_build_call_vec (builtin_decl_explicit (sink_ix), args);
gimple_set_location (g, loc);
gsi_insert_before (&gsi2, g, GSI_SAME_STMT);
cond = unshare_expr (cond);
cond = force_gimple_operand_gsi (gsi, cond, true, NULL_TREE, false,
GSI_CONTINUE_LINKING);
gsi_insert_after (gsi, gimple_build_cond_empty (cond), GSI_NEW_STMT);
edge e3 = make_edge (e1->src, e2->dest, EDGE_FALSE_VALUE);
e3->probability = profile_probability::guessed_always ().apply_scale (1, 8);
e1->probability = e3->probability.invert ();
e1->flags = EDGE_TRUE_VALUE;
set_immediate_dominator (CDI_DOMINATORS, e2->dest, e1->src);
*gsi = gsi_after_labels (e2->dest);
}
static void
expand_omp_ordered_source_sink (struct omp_region *region,
struct omp_for_data *fd, tree *counts,
basic_block cont_bb)
{
struct omp_region *inner;
int i;
for (i = fd->collapse - 1; i < fd->ordered; i++)
if (i == fd->collapse - 1 && fd->collapse > 1)
counts[i] = NULL_TREE;
else if (i >= fd->collapse && !cont_bb)
counts[i] = build_zero_cst (fd->iter_type);
else if (!POINTER_TYPE_P (TREE_TYPE (fd->loops[i].v))
&& integer_onep (fd->loops[i].step))
counts[i] = NULL_TREE;
else
counts[i] = create_tmp_var (fd->iter_type, ".orditer");
tree atype
= build_array_type_nelts (fd->iter_type, fd->ordered - fd->collapse + 1);
counts[fd->ordered] = create_tmp_var (atype, ".orditera");
TREE_ADDRESSABLE (counts[fd->ordered]) = 1;
for (inner = region->inner; inner; inner = inner->next)
if (inner->type == GIMPLE_OMP_ORDERED)
{
gomp_ordered *ord_stmt = inner->ord_stmt;
gimple_stmt_iterator gsi = gsi_for_stmt (ord_stmt);
location_t loc = gimple_location (ord_stmt);
tree c;
for (c = gimple_omp_ordered_clauses (ord_stmt);
c; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_DEPEND_KIND (c) == OMP_CLAUSE_DEPEND_SOURCE)
break;
if (c)
expand_omp_ordered_source (&gsi, fd, counts, loc);
for (c = gimple_omp_ordered_clauses (ord_stmt);
c; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_DEPEND_KIND (c) == OMP_CLAUSE_DEPEND_SINK)
expand_omp_ordered_sink (&gsi, fd, counts, c, loc);
gsi_remove (&gsi, true);
}
}
static basic_block
expand_omp_for_ordered_loops (struct omp_for_data *fd, tree *counts,
basic_block cont_bb, basic_block body_bb,
bool ordered_lastprivate)
{
if (fd->ordered == fd->collapse)
return cont_bb;
if (!cont_bb)
{
gimple_stmt_iterator gsi = gsi_after_labels (body_bb);
for (int i = fd->collapse; i < fd->ordered; i++)
{
tree type = TREE_TYPE (fd->loops[i].v);
tree n1 = fold_convert (type, fd->loops[i].n1);
expand_omp_build_assign (&gsi, fd->loops[i].v, n1);
tree aref = build4 (ARRAY_REF, fd->iter_type, counts[fd->ordered],
size_int (i - fd->collapse + 1),
NULL_TREE, NULL_TREE);
expand_omp_build_assign (&gsi, aref, build_zero_cst (fd->iter_type));
}
return NULL;
}
for (int i = fd->ordered - 1; i >= fd->collapse; i--)
{
tree t, type = TREE_TYPE (fd->loops[i].v);
gimple_stmt_iterator gsi = gsi_after_labels (body_bb);
expand_omp_build_assign (&gsi, fd->loops[i].v,
fold_convert (type, fd->loops[i].n1));
if (counts[i])
expand_omp_build_assign (&gsi, counts[i],
build_zero_cst (fd->iter_type));
tree aref = build4 (ARRAY_REF, fd->iter_type, counts[fd->ordered],
size_int (i - fd->collapse + 1),
NULL_TREE, NULL_TREE);
expand_omp_build_assign (&gsi, aref, build_zero_cst (fd->iter_type));
if (!gsi_end_p (gsi))
gsi_prev (&gsi);
else
gsi = gsi_last_bb (body_bb);
edge e1 = split_block (body_bb, gsi_stmt (gsi));
basic_block new_body = e1->dest;
if (body_bb == cont_bb)
cont_bb = new_body;
edge e2 = NULL;
basic_block new_header;
if (EDGE_COUNT (cont_bb->preds) > 0)
{
gsi = gsi_last_bb (cont_bb);
if (POINTER_TYPE_P (type))
t = fold_build_pointer_plus (fd->loops[i].v,
fold_convert (sizetype,
fd->loops[i].step));
else
t = fold_build2 (PLUS_EXPR, type, fd->loops[i].v,
fold_convert (type, fd->loops[i].step));
expand_omp_build_assign (&gsi, fd->loops[i].v, t);
if (counts[i])
{
t = fold_build2 (PLUS_EXPR, fd->iter_type, counts[i],
build_int_cst (fd->iter_type, 1));
expand_omp_build_assign (&gsi, counts[i], t);
t = counts[i];
}
else
{
t = fold_build2 (MINUS_EXPR, TREE_TYPE (fd->loops[i].v),
fd->loops[i].v, fd->loops[i].n1);
t = fold_convert (fd->iter_type, t);
t = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
true, GSI_SAME_STMT);
}
aref = build4 (ARRAY_REF, fd->iter_type, counts[fd->ordered],
size_int (i - fd->collapse + 1),
NULL_TREE, NULL_TREE);
expand_omp_build_assign (&gsi, aref, t);
gsi_prev (&gsi);
e2 = split_block (cont_bb, gsi_stmt (gsi));
new_header = e2->dest;
}
else
new_header = cont_bb;
gsi = gsi_after_labels (new_header);
tree v = force_gimple_operand_gsi (&gsi, fd->loops[i].v, true, NULL_TREE,
true, GSI_SAME_STMT);
tree n2
= force_gimple_operand_gsi (&gsi, fold_convert (type, fd->loops[i].n2),
true, NULL_TREE, true, GSI_SAME_STMT);
t = build2 (fd->loops[i].cond_code, boolean_type_node, v, n2);
gsi_insert_before (&gsi, gimple_build_cond_empty (t), GSI_NEW_STMT);
edge e3 = split_block (new_header, gsi_stmt (gsi));
cont_bb = e3->dest;
remove_edge (e1);
make_edge (body_bb, new_header, EDGE_FALLTHRU);
e3->flags = EDGE_FALSE_VALUE;
e3->probability = profile_probability::guessed_always ().apply_scale (1, 8);
e1 = make_edge (new_header, new_body, EDGE_TRUE_VALUE);
e1->probability = e3->probability.invert ();
set_immediate_dominator (CDI_DOMINATORS, new_header, body_bb);
set_immediate_dominator (CDI_DOMINATORS, new_body, new_header);
if (e2)
{
struct loop *loop = alloc_loop ();
loop->header = new_header;
loop->latch = e2->src;
add_loop (loop, body_bb->loop_father);
}
}
bool need_inits = false;
for (int i = fd->collapse; ordered_lastprivate && i < fd->ordered; i++)
if (need_inits)
{
tree type = TREE_TYPE (fd->loops[i].v);
gimple_stmt_iterator gsi = gsi_after_labels (body_bb);
expand_omp_build_assign (&gsi, fd->loops[i].v,
fold_convert (type, fd->loops[i].n1));
}
else
{
tree type = TREE_TYPE (fd->loops[i].v);
tree this_cond = fold_build2 (fd->loops[i].cond_code,
boolean_type_node,
fold_convert (type, fd->loops[i].n1),
fold_convert (type, fd->loops[i].n2));
if (!integer_onep (this_cond))
need_inits = true;
}
return cont_bb;
}
static void
expand_omp_for_generic (struct omp_region *region,
struct omp_for_data *fd,
enum built_in_function start_fn,
enum built_in_function next_fn,
gimple *inner_stmt)
{
tree type, istart0, iend0, iend;
tree t, vmain, vback, bias = NULL_TREE;
basic_block entry_bb, cont_bb, exit_bb, l0_bb, l1_bb, collapse_bb;
basic_block l2_bb = NULL, l3_bb = NULL;
gimple_stmt_iterator gsi;
gassign *assign_stmt;
bool in_combined_parallel = is_combined_parallel (region);
bool broken_loop = region->cont == NULL;
edge e, ne;
tree *counts = NULL;
int i;
bool ordered_lastprivate = false;
gcc_assert (!broken_loop || !in_combined_parallel);
gcc_assert (fd->iter_type == long_integer_type_node
|| !in_combined_parallel);
entry_bb = region->entry;
cont_bb = region->cont;
collapse_bb = NULL;
gcc_assert (EDGE_COUNT (entry_bb->succs) == 2);
gcc_assert (broken_loop
|| BRANCH_EDGE (entry_bb)->dest == FALLTHRU_EDGE (cont_bb)->dest);
l0_bb = split_edge (FALLTHRU_EDGE (entry_bb));
l1_bb = single_succ (l0_bb);
if (!broken_loop)
{
l2_bb = create_empty_bb (cont_bb);
gcc_assert (BRANCH_EDGE (cont_bb)->dest == l1_bb
|| (single_succ_edge (BRANCH_EDGE (cont_bb)->dest)->dest
== l1_bb));
gcc_assert (EDGE_COUNT (cont_bb->succs) == 2);
}
else
l2_bb = NULL;
l3_bb = BRANCH_EDGE (entry_bb)->dest;
exit_bb = region->exit;
gsi = gsi_last_nondebug_bb (entry_bb);
gcc_assert (gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_FOR);
if (fd->ordered
&& omp_find_clause (gimple_omp_for_clauses (gsi_stmt (gsi)),
OMP_CLAUSE_LASTPRIVATE))
ordered_lastprivate = false;
if (fd->collapse > 1 || fd->ordered)
{
int first_zero_iter1 = -1, first_zero_iter2 = -1;
basic_block zero_iter1_bb = NULL, zero_iter2_bb = NULL, l2_dom_bb = NULL;
counts = XALLOCAVEC (tree, fd->ordered ? fd->ordered + 1 : fd->collapse);
expand_omp_for_init_counts (fd, &gsi, entry_bb, counts,
zero_iter1_bb, first_zero_iter1,
zero_iter2_bb, first_zero_iter2, l2_dom_bb);
if (zero_iter1_bb)
{
for (i = first_zero_iter1;
i < (fd->ordered ? fd->ordered : fd->collapse); i++)
if (SSA_VAR_P (counts[i]))
TREE_NO_WARNING (counts[i]) = 1;
gsi_prev (&gsi);
e = split_block (entry_bb, gsi_stmt (gsi));
entry_bb = e->dest;
make_edge (zero_iter1_bb, entry_bb, EDGE_FALLTHRU);
gsi = gsi_last_nondebug_bb (entry_bb);
set_immediate_dominator (CDI_DOMINATORS, entry_bb,
get_immediate_dominator (CDI_DOMINATORS,
zero_iter1_bb));
}
if (zero_iter2_bb)
{
for (i = first_zero_iter2; i < fd->ordered; i++)
if (SSA_VAR_P (counts[i]))
TREE_NO_WARNING (counts[i]) = 1;
if (zero_iter1_bb)
make_edge (zero_iter2_bb, entry_bb, EDGE_FALLTHRU);
else
{
gsi_prev (&gsi);
e = split_block (entry_bb, gsi_stmt (gsi));
entry_bb = e->dest;
make_edge (zero_iter2_bb, entry_bb, EDGE_FALLTHRU);
gsi = gsi_last_nondebug_bb (entry_bb);
set_immediate_dominator (CDI_DOMINATORS, entry_bb,
get_immediate_dominator
(CDI_DOMINATORS, zero_iter2_bb));
}
}
if (fd->collapse == 1)
{
counts[0] = fd->loop.n2;
fd->loop = fd->loops[0];
}
}
type = TREE_TYPE (fd->loop.v);
istart0 = create_tmp_var (fd->iter_type, ".istart0");
iend0 = create_tmp_var (fd->iter_type, ".iend0");
TREE_ADDRESSABLE (istart0) = 1;
TREE_ADDRESSABLE (iend0) = 1;
if (fd->iter_type == long_long_unsigned_type_node
&& TREE_CODE (type) == INTEGER_TYPE
&& !TYPE_UNSIGNED (type)
&& fd->ordered == 0)
{
tree n1, n2;
if (fd->loop.cond_code == LT_EXPR)
{
n1 = fd->loop.n1;
n2 = fold_build2 (PLUS_EXPR, type, fd->loop.n2, fd->loop.step);
}
else
{
n1 = fold_build2 (MINUS_EXPR, type, fd->loop.n2, fd->loop.step);
n2 = fd->loop.n1;
}
if (TREE_CODE (n1) != INTEGER_CST
|| TREE_CODE (n2) != INTEGER_CST
|| ((tree_int_cst_sgn (n1) < 0) ^ (tree_int_cst_sgn (n2) < 0)))
bias = fold_convert (fd->iter_type, TYPE_MIN_VALUE (type));
}
gimple_stmt_iterator gsif = gsi;
gsi_prev (&gsif);
tree arr = NULL_TREE;
if (in_combined_parallel)
{
gcc_assert (fd->ordered == 0);
t = build_call_expr (builtin_decl_explicit (next_fn), 2,
build_fold_addr_expr (istart0),
build_fold_addr_expr (iend0));
}
else
{
tree t0, t1, t2, t3, t4;
t4 = build_fold_addr_expr (iend0);
t3 = build_fold_addr_expr (istart0);
if (fd->ordered)
{
t0 = build_int_cst (unsigned_type_node,
fd->ordered - fd->collapse + 1);
arr = create_tmp_var (build_array_type_nelts (fd->iter_type,
fd->ordered
- fd->collapse + 1),
".omp_counts");
DECL_NAMELESS (arr) = 1;
TREE_ADDRESSABLE (arr) = 1;
TREE_STATIC (arr) = 1;
vec<constructor_elt, va_gc> *v;
vec_alloc (v, fd->ordered - fd->collapse + 1);
int idx;
for (idx = 0; idx < fd->ordered - fd->collapse + 1; idx++)
{
tree c;
if (idx == 0 && fd->collapse > 1)
c = fd->loop.n2;
else
c = counts[idx + fd->collapse - 1];
tree purpose = size_int (idx);
CONSTRUCTOR_APPEND_ELT (v, purpose, c);
if (TREE_CODE (c) != INTEGER_CST)
TREE_STATIC (arr) = 0;
}
DECL_INITIAL (arr) = build_constructor (TREE_TYPE (arr), v);
if (!TREE_STATIC (arr))
force_gimple_operand_gsi (&gsi, build1 (DECL_EXPR,
void_type_node, arr),
true, NULL_TREE, true, GSI_SAME_STMT);
t1 = build_fold_addr_expr (arr);
t2 = NULL_TREE;
}
else
{
t2 = fold_convert (fd->iter_type, fd->loop.step);
t1 = fd->loop.n2;
t0 = fd->loop.n1;
if (gimple_omp_for_combined_into_p (fd->for_stmt))
{
tree innerc
= omp_find_clause (gimple_omp_for_clauses (fd->for_stmt),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
t0 = OMP_CLAUSE_DECL (innerc);
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
t1 = OMP_CLAUSE_DECL (innerc);
}
if (POINTER_TYPE_P (TREE_TYPE (t0))
&& TYPE_PRECISION (TREE_TYPE (t0))
!= TYPE_PRECISION (fd->iter_type))
{
tree itype = signed_type_for (type);
t1 = fold_convert (fd->iter_type, fold_convert (itype, t1));
t0 = fold_convert (fd->iter_type, fold_convert (itype, t0));
}
else
{
t1 = fold_convert (fd->iter_type, t1);
t0 = fold_convert (fd->iter_type, t0);
}
if (bias)
{
t1 = fold_build2 (PLUS_EXPR, fd->iter_type, t1, bias);
t0 = fold_build2 (PLUS_EXPR, fd->iter_type, t0, bias);
}
}
if (fd->iter_type == long_integer_type_node || fd->ordered)
{
if (fd->chunk_size)
{
t = fold_convert (fd->iter_type, fd->chunk_size);
t = omp_adjust_chunk_size (t, fd->simd_schedule);
if (fd->ordered)
t = build_call_expr (builtin_decl_explicit (start_fn),
5, t0, t1, t, t3, t4);
else
t = build_call_expr (builtin_decl_explicit (start_fn),
6, t0, t1, t2, t, t3, t4);
}
else if (fd->ordered)
t = build_call_expr (builtin_decl_explicit (start_fn),
4, t0, t1, t3, t4);
else
t = build_call_expr (builtin_decl_explicit (start_fn),
5, t0, t1, t2, t3, t4);
}
else
{
tree t5;
tree c_bool_type;
tree bfn_decl;
bfn_decl = builtin_decl_explicit (start_fn);
c_bool_type = TREE_TYPE (TREE_TYPE (bfn_decl));
t5 = build_int_cst (c_bool_type,
fd->loop.cond_code == LT_EXPR ? 1 : 0);
if (fd->chunk_size)
{
tree bfn_decl = builtin_decl_explicit (start_fn);
t = fold_convert (fd->iter_type, fd->chunk_size);
t = omp_adjust_chunk_size (t, fd->simd_schedule);
t = build_call_expr (bfn_decl, 7, t5, t0, t1, t2, t, t3, t4);
}
else
t = build_call_expr (builtin_decl_explicit (start_fn),
6, t5, t0, t1, t2, t3, t4);
}
}
if (TREE_TYPE (t) != boolean_type_node)
t = fold_build2 (NE_EXPR, boolean_type_node,
t, build_int_cst (TREE_TYPE (t), 0));
t = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
true, GSI_SAME_STMT);
if (arr && !TREE_STATIC (arr))
{
tree clobber = build_constructor (TREE_TYPE (arr), NULL);
TREE_THIS_VOLATILE (clobber) = 1;
gsi_insert_before (&gsi, gimple_build_assign (arr, clobber),
GSI_SAME_STMT);
}
gsi_insert_after (&gsi, gimple_build_cond_empty (t), GSI_SAME_STMT);
gsi_remove (&gsi, true);
if (gsi_end_p (gsif))
gsif = gsi_after_labels (gsi_bb (gsif));
gsi_next (&gsif);
tree startvar = fd->loop.v;
tree endvar = NULL_TREE;
if (gimple_omp_for_combined_p (fd->for_stmt))
{
gcc_assert (gimple_code (inner_stmt) == GIMPLE_OMP_FOR
&& gimple_omp_for_kind (inner_stmt)
== GF_OMP_FOR_KIND_SIMD);
tree innerc = omp_find_clause (gimple_omp_for_clauses (inner_stmt),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
startvar = OMP_CLAUSE_DECL (innerc);
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
endvar = OMP_CLAUSE_DECL (innerc);
}
gsi = gsi_start_bb (l0_bb);
t = istart0;
if (fd->ordered && fd->collapse == 1)
t = fold_build2 (MULT_EXPR, fd->iter_type, t,
fold_convert (fd->iter_type, fd->loop.step));
else if (bias)
t = fold_build2 (MINUS_EXPR, fd->iter_type, t, bias);
if (fd->ordered && fd->collapse == 1)
{
if (POINTER_TYPE_P (TREE_TYPE (startvar)))
t = fold_build2 (POINTER_PLUS_EXPR, TREE_TYPE (startvar),
fd->loop.n1, fold_convert (sizetype, t));
else
{
t = fold_convert (TREE_TYPE (startvar), t);
t = fold_build2 (PLUS_EXPR, TREE_TYPE (startvar),
fd->loop.n1, t);
}
}
else
{
if (POINTER_TYPE_P (TREE_TYPE (startvar)))
t = fold_convert (signed_type_for (TREE_TYPE (startvar)), t);
t = fold_convert (TREE_TYPE (startvar), t);
}
t = force_gimple_operand_gsi (&gsi, t,
DECL_P (startvar)
&& TREE_ADDRESSABLE (startvar),
NULL_TREE, false, GSI_CONTINUE_LINKING);
assign_stmt = gimple_build_assign (startvar, t);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
t = iend0;
if (fd->ordered && fd->collapse == 1)
t = fold_build2 (MULT_EXPR, fd->iter_type, t,
fold_convert (fd->iter_type, fd->loop.step));
else if (bias)
t = fold_build2 (MINUS_EXPR, fd->iter_type, t, bias);
if (fd->ordered && fd->collapse == 1)
{
if (POINTER_TYPE_P (TREE_TYPE (startvar)))
t = fold_build2 (POINTER_PLUS_EXPR, TREE_TYPE (startvar),
fd->loop.n1, fold_convert (sizetype, t));
else
{
t = fold_convert (TREE_TYPE (startvar), t);
t = fold_build2 (PLUS_EXPR, TREE_TYPE (startvar),
fd->loop.n1, t);
}
}
else
{
if (POINTER_TYPE_P (TREE_TYPE (startvar)))
t = fold_convert (signed_type_for (TREE_TYPE (startvar)), t);
t = fold_convert (TREE_TYPE (startvar), t);
}
iend = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
if (endvar)
{
assign_stmt = gimple_build_assign (endvar, iend);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
if (useless_type_conversion_p (TREE_TYPE (fd->loop.v), TREE_TYPE (iend)))
assign_stmt = gimple_build_assign (fd->loop.v, iend);
else
assign_stmt = gimple_build_assign (fd->loop.v, NOP_EXPR, iend);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
}
tree itercnt = NULL_TREE;
if (gimple_omp_for_kind (fd->for_stmt) == GF_OMP_FOR_KIND_FOR)
for (tree c = gimple_omp_for_clauses (fd->for_stmt);
c; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LINEAR
&& !OMP_CLAUSE_LINEAR_NO_COPYIN (c))
{
tree d = OMP_CLAUSE_DECL (c);
bool is_ref = omp_is_reference (d);
tree t = d, a, dest;
if (is_ref)
t = build_simple_mem_ref_loc (OMP_CLAUSE_LOCATION (c), t);
tree type = TREE_TYPE (t);
if (POINTER_TYPE_P (type))
type = sizetype;
dest = unshare_expr (t);
tree v = create_tmp_var (TREE_TYPE (t), NULL);
expand_omp_build_assign (&gsif, v, t);
if (itercnt == NULL_TREE)
{
itercnt = startvar;
tree n1 = fd->loop.n1;
if (POINTER_TYPE_P (TREE_TYPE (itercnt)))
{
itercnt
= fold_convert (signed_type_for (TREE_TYPE (itercnt)),
itercnt);
n1 = fold_convert (TREE_TYPE (itercnt), n1);
}
itercnt = fold_build2 (MINUS_EXPR, TREE_TYPE (itercnt),
itercnt, n1);
itercnt = fold_build2 (EXACT_DIV_EXPR, TREE_TYPE (itercnt),
itercnt, fd->loop.step);
itercnt = force_gimple_operand_gsi (&gsi, itercnt, true,
NULL_TREE, false,
GSI_CONTINUE_LINKING);
}
a = fold_build2 (MULT_EXPR, type,
fold_convert (type, itercnt),
fold_convert (type, OMP_CLAUSE_LINEAR_STEP (c)));
t = fold_build2 (type == TREE_TYPE (t) ? PLUS_EXPR
: POINTER_PLUS_EXPR, TREE_TYPE (t), v, a);
t = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
assign_stmt = gimple_build_assign (dest, t);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
}
if (fd->collapse > 1)
expand_omp_for_init_vars (fd, &gsi, counts, inner_stmt, startvar);
if (fd->ordered)
{
if (fd->collapse > 1)
memmove (counts, counts + 1, (fd->collapse - 1) * sizeof (counts[0]));
if (broken_loop)
{
int i;
for (i = fd->collapse; i < fd->ordered; i++)
{
tree type = TREE_TYPE (fd->loops[i].v);
tree this_cond
= fold_build2 (fd->loops[i].cond_code, boolean_type_node,
fold_convert (type, fd->loops[i].n1),
fold_convert (type, fd->loops[i].n2));
if (!integer_onep (this_cond))
break;
}
if (i < fd->ordered)
{
cont_bb
= create_empty_bb (EXIT_BLOCK_PTR_FOR_FN (cfun)->prev_bb);
add_bb_to_loop (cont_bb, l1_bb->loop_father);
gimple_stmt_iterator gsi = gsi_after_labels (cont_bb);
gimple *g = gimple_build_omp_continue (fd->loop.v, fd->loop.v);
gsi_insert_before (&gsi, g, GSI_SAME_STMT);
make_edge (cont_bb, l3_bb, EDGE_FALLTHRU);
make_edge (cont_bb, l1_bb, 0);
l2_bb = create_empty_bb (cont_bb);
broken_loop = false;
}
}
expand_omp_ordered_source_sink (region, fd, counts, cont_bb);
cont_bb = expand_omp_for_ordered_loops (fd, counts, cont_bb, l1_bb,
ordered_lastprivate);
if (counts[fd->collapse - 1])
{
gcc_assert (fd->collapse == 1);
gsi = gsi_last_bb (l0_bb);
expand_omp_build_assign (&gsi, counts[fd->collapse - 1],
istart0, true);
gsi = gsi_last_bb (cont_bb);
t = fold_build2 (PLUS_EXPR, fd->iter_type, counts[fd->collapse - 1],
build_int_cst (fd->iter_type, 1));
expand_omp_build_assign (&gsi, counts[fd->collapse - 1], t);
tree aref = build4 (ARRAY_REF, fd->iter_type, counts[fd->ordered],
size_zero_node, NULL_TREE, NULL_TREE);
expand_omp_build_assign (&gsi, aref, counts[fd->collapse - 1]);
t = counts[fd->collapse - 1];
}
else if (fd->collapse > 1)
t = fd->loop.v;
else
{
t = fold_build2 (MINUS_EXPR, TREE_TYPE (fd->loops[0].v),
fd->loops[0].v, fd->loops[0].n1);
t = fold_convert (fd->iter_type, t);
}
gsi = gsi_last_bb (l0_bb);
tree aref = build4 (ARRAY_REF, fd->iter_type, counts[fd->ordered],
size_zero_node, NULL_TREE, NULL_TREE);
t = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
expand_omp_build_assign (&gsi, aref, t, true);
}
if (!broken_loop)
{
gsi = gsi_last_nondebug_bb (cont_bb);
gomp_continue *cont_stmt = as_a <gomp_continue *> (gsi_stmt (gsi));
gcc_assert (gimple_code (cont_stmt) == GIMPLE_OMP_CONTINUE);
vmain = gimple_omp_continue_control_use (cont_stmt);
vback = gimple_omp_continue_control_def (cont_stmt);
if (!gimple_omp_for_combined_p (fd->for_stmt))
{
if (POINTER_TYPE_P (type))
t = fold_build_pointer_plus (vmain, fd->loop.step);
else
t = fold_build2 (PLUS_EXPR, type, vmain, fd->loop.step);
t = force_gimple_operand_gsi (&gsi, t,
DECL_P (vback)
&& TREE_ADDRESSABLE (vback),
NULL_TREE, true, GSI_SAME_STMT);
assign_stmt = gimple_build_assign (vback, t);
gsi_insert_before (&gsi, assign_stmt, GSI_SAME_STMT);
if (fd->ordered && counts[fd->collapse - 1] == NULL_TREE)
{
tree tem;
if (fd->collapse > 1)
tem = fd->loop.v;
else
{
tem = fold_build2 (MINUS_EXPR, TREE_TYPE (fd->loops[0].v),
fd->loops[0].v, fd->loops[0].n1);
tem = fold_convert (fd->iter_type, tem);
}
tree aref = build4 (ARRAY_REF, fd->iter_type,
counts[fd->ordered], size_zero_node,
NULL_TREE, NULL_TREE);
tem = force_gimple_operand_gsi (&gsi, tem, true, NULL_TREE,
true, GSI_SAME_STMT);
expand_omp_build_assign (&gsi, aref, tem);
}
t = build2 (fd->loop.cond_code, boolean_type_node,
DECL_P (vback) && TREE_ADDRESSABLE (vback) ? t : vback,
iend);
gcond *cond_stmt = gimple_build_cond_empty (t);
gsi_insert_before (&gsi, cond_stmt, GSI_SAME_STMT);
}
gsi_remove (&gsi, true);
if (fd->collapse > 1 && !gimple_omp_for_combined_p (fd->for_stmt))
collapse_bb = extract_omp_for_update_vars (fd, cont_bb, l1_bb);
gsi = gsi_start_bb (l2_bb);
t = build_call_expr (builtin_decl_explicit (next_fn), 2,
build_fold_addr_expr (istart0),
build_fold_addr_expr (iend0));
t = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
if (TREE_TYPE (t) != boolean_type_node)
t = fold_build2 (NE_EXPR, boolean_type_node,
t, build_int_cst (TREE_TYPE (t), 0));
gcond *cond_stmt = gimple_build_cond_empty (t);
gsi_insert_after (&gsi, cond_stmt, GSI_CONTINUE_LINKING);
}
gsi = gsi_last_nondebug_bb (exit_bb);
if (gimple_omp_return_nowait_p (gsi_stmt (gsi)))
t = builtin_decl_explicit (BUILT_IN_GOMP_LOOP_END_NOWAIT);
else if (gimple_omp_return_lhs (gsi_stmt (gsi)))
t = builtin_decl_explicit (BUILT_IN_GOMP_LOOP_END_CANCEL);
else
t = builtin_decl_explicit (BUILT_IN_GOMP_LOOP_END);
gcall *call_stmt = gimple_build_call (t, 0);
if (gimple_omp_return_lhs (gsi_stmt (gsi)))
gimple_call_set_lhs (call_stmt, gimple_omp_return_lhs (gsi_stmt (gsi)));
gsi_insert_after (&gsi, call_stmt, GSI_SAME_STMT);
if (fd->ordered)
{
tree arr = counts[fd->ordered];
tree clobber = build_constructor (TREE_TYPE (arr), NULL);
TREE_THIS_VOLATILE (clobber) = 1;
gsi_insert_after (&gsi, gimple_build_assign (arr, clobber),
GSI_SAME_STMT);
}
gsi_remove (&gsi, true);
find_edge (entry_bb, l0_bb)->flags = EDGE_TRUE_VALUE;
find_edge (entry_bb, l3_bb)->flags = EDGE_FALSE_VALUE;
if (!broken_loop)
{
gimple_seq phis;
e = find_edge (cont_bb, l3_bb);
ne = make_edge (l2_bb, l3_bb, EDGE_FALSE_VALUE);
phis = phi_nodes (l3_bb);
for (gsi = gsi_start (phis); !gsi_end_p (gsi); gsi_next (&gsi))
{
gimple *phi = gsi_stmt (gsi);
SET_USE (PHI_ARG_DEF_PTR_FROM_EDGE (phi, ne),
PHI_ARG_DEF_FROM_EDGE (phi, e));
}
remove_edge (e);
make_edge (cont_bb, l2_bb, EDGE_FALSE_VALUE);
e = find_edge (cont_bb, l1_bb);
if (e == NULL)
{
e = BRANCH_EDGE (cont_bb);
gcc_assert (single_succ (e->dest) == l1_bb);
}
if (gimple_omp_for_combined_p (fd->for_stmt))
{
remove_edge (e);
e = NULL;
}
else if (fd->collapse > 1)
{
remove_edge (e);
e = make_edge (cont_bb, collapse_bb, EDGE_TRUE_VALUE);
}
else
e->flags = EDGE_TRUE_VALUE;
if (e)
{
e->probability = profile_probability::guessed_always ().apply_scale (7, 8);
find_edge (cont_bb, l2_bb)->probability = e->probability.invert ();
}
else
{
e = find_edge (cont_bb, l2_bb);
e->flags = EDGE_FALLTHRU;
}
make_edge (l2_bb, l0_bb, EDGE_TRUE_VALUE);
if (gimple_in_ssa_p (cfun))
{
gphi_iterator psi;
for (psi = gsi_start_phis (l3_bb); !gsi_end_p (psi); gsi_next (&psi))
{
source_location locus;
gphi *nphi;
gphi *exit_phi = psi.phi ();
if (virtual_operand_p (gimple_phi_result (exit_phi)))
continue;
edge l2_to_l3 = find_edge (l2_bb, l3_bb);
tree exit_res = PHI_ARG_DEF_FROM_EDGE (exit_phi, l2_to_l3);
basic_block latch = BRANCH_EDGE (cont_bb)->dest;
edge latch_to_l1 = find_edge (latch, l1_bb);
gphi *inner_phi
= find_phi_with_arg_on_edge (exit_res, latch_to_l1);
tree t = gimple_phi_result (exit_phi);
tree new_res = copy_ssa_name (t, NULL);
nphi = create_phi_node (new_res, l0_bb);
edge l0_to_l1 = find_edge (l0_bb, l1_bb);
t = PHI_ARG_DEF_FROM_EDGE (inner_phi, l0_to_l1);
locus = gimple_phi_arg_location_from_edge (inner_phi, l0_to_l1);
edge entry_to_l0 = find_edge (entry_bb, l0_bb);
add_phi_arg (nphi, t, entry_to_l0, locus);
edge l2_to_l0 = find_edge (l2_bb, l0_bb);
add_phi_arg (nphi, exit_res, l2_to_l0, UNKNOWN_LOCATION);
add_phi_arg (inner_phi, new_res, l0_to_l1, UNKNOWN_LOCATION);
}
}
set_immediate_dominator (CDI_DOMINATORS, l2_bb,
recompute_dominator (CDI_DOMINATORS, l2_bb));
set_immediate_dominator (CDI_DOMINATORS, l3_bb,
recompute_dominator (CDI_DOMINATORS, l3_bb));
set_immediate_dominator (CDI_DOMINATORS, l0_bb,
recompute_dominator (CDI_DOMINATORS, l0_bb));
set_immediate_dominator (CDI_DOMINATORS, l1_bb,
recompute_dominator (CDI_DOMINATORS, l1_bb));
struct loop *outer_loop = entry_bb->loop_father;
bool orig_loop_has_loop_struct = l1_bb->loop_father != outer_loop;
add_bb_to_loop (l2_bb, outer_loop);
struct loop *new_loop = alloc_loop ();
new_loop->header = l0_bb;
new_loop->latch = l2_bb;
add_loop (new_loop, outer_loop);
if (!orig_loop_has_loop_struct
&& !gimple_omp_for_combined_p (fd->for_stmt))
{
struct loop *orig_loop = alloc_loop ();
orig_loop->header = l1_bb;
add_loop (orig_loop, new_loop);
}
}
}
static void
expand_omp_for_static_nochunk (struct omp_region *region,
struct omp_for_data *fd,
gimple *inner_stmt)
{
tree n, q, s0, e0, e, t, tt, nthreads, threadid;
tree type, itype, vmain, vback;
basic_block entry_bb, second_bb, third_bb, exit_bb, seq_start_bb;
basic_block body_bb, cont_bb, collapse_bb = NULL;
basic_block fin_bb;
gimple_stmt_iterator gsi;
edge ep;
bool broken_loop = region->cont == NULL;
tree *counts = NULL;
tree n1, n2, step;
itype = type = TREE_TYPE (fd->loop.v);
if (POINTER_TYPE_P (type))
itype = signed_type_for (type);
entry_bb = region->entry;
cont_bb = region->cont;
gcc_assert (EDGE_COUNT (entry_bb->succs) == 2);
fin_bb = BRANCH_EDGE (entry_bb)->dest;
gcc_assert (broken_loop
|| (fin_bb == FALLTHRU_EDGE (cont_bb)->dest));
seq_start_bb = split_edge (FALLTHRU_EDGE (entry_bb));
body_bb = single_succ (seq_start_bb);
if (!broken_loop)
{
gcc_assert (BRANCH_EDGE (cont_bb)->dest == body_bb
|| single_succ (BRANCH_EDGE (cont_bb)->dest) == body_bb);
gcc_assert (EDGE_COUNT (cont_bb->succs) == 2);
}
exit_bb = region->exit;
gsi = gsi_last_nondebug_bb (entry_bb);
gcc_assert (gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_FOR);
if (fd->collapse > 1)
{
int first_zero_iter = -1, dummy = -1;
basic_block l2_dom_bb = NULL, dummy_bb = NULL;
counts = XALLOCAVEC (tree, fd->collapse);
expand_omp_for_init_counts (fd, &gsi, entry_bb, counts,
fin_bb, first_zero_iter,
dummy_bb, dummy, l2_dom_bb);
t = NULL_TREE;
}
else if (gimple_omp_for_combined_into_p (fd->for_stmt))
t = integer_one_node;
else
t = fold_binary (fd->loop.cond_code, boolean_type_node,
fold_convert (type, fd->loop.n1),
fold_convert (type, fd->loop.n2));
if (fd->collapse == 1
&& TYPE_UNSIGNED (type)
&& (t == NULL_TREE || !integer_onep (t)))
{
n1 = fold_convert (type, unshare_expr (fd->loop.n1));
n1 = force_gimple_operand_gsi (&gsi, n1, true, NULL_TREE,
true, GSI_SAME_STMT);
n2 = fold_convert (type, unshare_expr (fd->loop.n2));
n2 = force_gimple_operand_gsi (&gsi, n2, true, NULL_TREE,
true, GSI_SAME_STMT);
gcond *cond_stmt = gimple_build_cond (fd->loop.cond_code, n1, n2,
NULL_TREE, NULL_TREE);
gsi_insert_before (&gsi, cond_stmt, GSI_SAME_STMT);
if (walk_tree (gimple_cond_lhs_ptr (cond_stmt),
expand_omp_regimplify_p, NULL, NULL)
|| walk_tree (gimple_cond_rhs_ptr (cond_stmt),
expand_omp_regimplify_p, NULL, NULL))
{
gsi = gsi_for_stmt (cond_stmt);
gimple_regimplify_operands (cond_stmt, &gsi);
}
ep = split_block (entry_bb, cond_stmt);
ep->flags = EDGE_TRUE_VALUE;
entry_bb = ep->dest;
ep->probability = profile_probability::very_likely ();
ep = make_edge (ep->src, fin_bb, EDGE_FALSE_VALUE);
ep->probability = profile_probability::very_unlikely ();
if (gimple_in_ssa_p (cfun))
{
int dest_idx = find_edge (entry_bb, fin_bb)->dest_idx;
for (gphi_iterator gpi = gsi_start_phis (fin_bb);
!gsi_end_p (gpi); gsi_next (&gpi))
{
gphi *phi = gpi.phi ();
add_phi_arg (phi, gimple_phi_arg_def (phi, dest_idx),
ep, UNKNOWN_LOCATION);
}
}
gsi = gsi_last_bb (entry_bb);
}
switch (gimple_omp_for_kind (fd->for_stmt))
{
case GF_OMP_FOR_KIND_FOR:
nthreads = builtin_decl_explicit (BUILT_IN_OMP_GET_NUM_THREADS);
threadid = builtin_decl_explicit (BUILT_IN_OMP_GET_THREAD_NUM);
break;
case GF_OMP_FOR_KIND_DISTRIBUTE:
nthreads = builtin_decl_explicit (BUILT_IN_OMP_GET_NUM_TEAMS);
threadid = builtin_decl_explicit (BUILT_IN_OMP_GET_TEAM_NUM);
break;
default:
gcc_unreachable ();
}
nthreads = build_call_expr (nthreads, 0);
nthreads = fold_convert (itype, nthreads);
nthreads = force_gimple_operand_gsi (&gsi, nthreads, true, NULL_TREE,
true, GSI_SAME_STMT);
threadid = build_call_expr (threadid, 0);
threadid = fold_convert (itype, threadid);
threadid = force_gimple_operand_gsi (&gsi, threadid, true, NULL_TREE,
true, GSI_SAME_STMT);
n1 = fd->loop.n1;
n2 = fd->loop.n2;
step = fd->loop.step;
if (gimple_omp_for_combined_into_p (fd->for_stmt))
{
tree innerc = omp_find_clause (gimple_omp_for_clauses (fd->for_stmt),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
n1 = OMP_CLAUSE_DECL (innerc);
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
n2 = OMP_CLAUSE_DECL (innerc);
}
n1 = force_gimple_operand_gsi (&gsi, fold_convert (type, n1),
true, NULL_TREE, true, GSI_SAME_STMT);
n2 = force_gimple_operand_gsi (&gsi, fold_convert (itype, n2),
true, NULL_TREE, true, GSI_SAME_STMT);
step = force_gimple_operand_gsi (&gsi, fold_convert (itype, step),
true, NULL_TREE, true, GSI_SAME_STMT);
t = build_int_cst (itype, (fd->loop.cond_code == LT_EXPR ? -1 : 1));
t = fold_build2 (PLUS_EXPR, itype, step, t);
t = fold_build2 (PLUS_EXPR, itype, t, n2);
t = fold_build2 (MINUS_EXPR, itype, t, fold_convert (itype, n1));
if (TYPE_UNSIGNED (itype) && fd->loop.cond_code == GT_EXPR)
t = fold_build2 (TRUNC_DIV_EXPR, itype,
fold_build1 (NEGATE_EXPR, itype, t),
fold_build1 (NEGATE_EXPR, itype, step));
else
t = fold_build2 (TRUNC_DIV_EXPR, itype, t, step);
t = fold_convert (itype, t);
n = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE, true, GSI_SAME_STMT);
q = create_tmp_reg (itype, "q");
t = fold_build2 (TRUNC_DIV_EXPR, itype, n, nthreads);
t = force_gimple_operand_gsi (&gsi, t, false, NULL_TREE, true, GSI_SAME_STMT);
gsi_insert_before (&gsi, gimple_build_assign (q, t), GSI_SAME_STMT);
tt = create_tmp_reg (itype, "tt");
t = fold_build2 (TRUNC_MOD_EXPR, itype, n, nthreads);
t = force_gimple_operand_gsi (&gsi, t, false, NULL_TREE, true, GSI_SAME_STMT);
gsi_insert_before (&gsi, gimple_build_assign (tt, t), GSI_SAME_STMT);
t = build2 (LT_EXPR, boolean_type_node, threadid, tt);
gcond *cond_stmt = gimple_build_cond_empty (t);
gsi_insert_before (&gsi, cond_stmt, GSI_SAME_STMT);
second_bb = split_block (entry_bb, cond_stmt)->dest;
gsi = gsi_last_nondebug_bb (second_bb);
gcc_assert (gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_FOR);
gsi_insert_before (&gsi, gimple_build_assign (tt, build_int_cst (itype, 0)),
GSI_SAME_STMT);
gassign *assign_stmt
= gimple_build_assign (q, PLUS_EXPR, q, build_int_cst (itype, 1));
gsi_insert_before (&gsi, assign_stmt, GSI_SAME_STMT);
third_bb = split_block (second_bb, assign_stmt)->dest;
gsi = gsi_last_nondebug_bb (third_bb);
gcc_assert (gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_FOR);
t = build2 (MULT_EXPR, itype, q, threadid);
t = build2 (PLUS_EXPR, itype, t, tt);
s0 = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE, true, GSI_SAME_STMT);
t = fold_build2 (PLUS_EXPR, itype, s0, q);
e0 = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE, true, GSI_SAME_STMT);
t = build2 (GE_EXPR, boolean_type_node, s0, e0);
gsi_insert_before (&gsi, gimple_build_cond_empty (t), GSI_SAME_STMT);
gsi_remove (&gsi, true);
gsi = gsi_start_bb (seq_start_bb);
tree startvar = fd->loop.v;
tree endvar = NULL_TREE;
if (gimple_omp_for_combined_p (fd->for_stmt))
{
tree clauses = gimple_code (inner_stmt) == GIMPLE_OMP_PARALLEL
? gimple_omp_parallel_clauses (inner_stmt)
: gimple_omp_for_clauses (inner_stmt);
tree innerc = omp_find_clause (clauses, OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
startvar = OMP_CLAUSE_DECL (innerc);
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
endvar = OMP_CLAUSE_DECL (innerc);
if (fd->collapse > 1 && TREE_CODE (fd->loop.n2) != INTEGER_CST
&& gimple_omp_for_kind (fd->for_stmt) == GF_OMP_FOR_KIND_DISTRIBUTE)
{
int i;
for (i = 1; i < fd->collapse; i++)
{
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
}
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
if (innerc)
{
tree t = fold_convert (TREE_TYPE (OMP_CLAUSE_DECL (innerc)),
fd->loop.n2);
t = force_gimple_operand_gsi (&gsi, t, false, NULL_TREE, false,
GSI_CONTINUE_LINKING);
assign_stmt = gimple_build_assign (OMP_CLAUSE_DECL (innerc), t);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
}
}
}
t = fold_convert (itype, s0);
t = fold_build2 (MULT_EXPR, itype, t, step);
if (POINTER_TYPE_P (type))
t = fold_build_pointer_plus (n1, t);
else
t = fold_build2 (PLUS_EXPR, type, t, n1);
t = fold_convert (TREE_TYPE (startvar), t);
t = force_gimple_operand_gsi (&gsi, t,
DECL_P (startvar)
&& TREE_ADDRESSABLE (startvar),
NULL_TREE, false, GSI_CONTINUE_LINKING);
assign_stmt = gimple_build_assign (startvar, t);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
t = fold_convert (itype, e0);
t = fold_build2 (MULT_EXPR, itype, t, step);
if (POINTER_TYPE_P (type))
t = fold_build_pointer_plus (n1, t);
else
t = fold_build2 (PLUS_EXPR, type, t, n1);
t = fold_convert (TREE_TYPE (startvar), t);
e = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
if (endvar)
{
assign_stmt = gimple_build_assign (endvar, e);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
if (useless_type_conversion_p (TREE_TYPE (fd->loop.v), TREE_TYPE (e)))
assign_stmt = gimple_build_assign (fd->loop.v, e);
else
assign_stmt = gimple_build_assign (fd->loop.v, NOP_EXPR, e);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
}
tree itercnt = NULL_TREE;
if (gimple_omp_for_kind (fd->for_stmt) == GF_OMP_FOR_KIND_FOR)
for (tree c = gimple_omp_for_clauses (fd->for_stmt);
c; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LINEAR
&& !OMP_CLAUSE_LINEAR_NO_COPYIN (c))
{
tree d = OMP_CLAUSE_DECL (c);
bool is_ref = omp_is_reference (d);
tree t = d, a, dest;
if (is_ref)
t = build_simple_mem_ref_loc (OMP_CLAUSE_LOCATION (c), t);
if (itercnt == NULL_TREE)
{
if (gimple_omp_for_combined_into_p (fd->for_stmt))
{
itercnt = fold_build2 (MINUS_EXPR, itype,
fold_convert (itype, n1),
fold_convert (itype, fd->loop.n1));
itercnt = fold_build2 (EXACT_DIV_EXPR, itype, itercnt, step);
itercnt = fold_build2 (PLUS_EXPR, itype, itercnt, s0);
itercnt = force_gimple_operand_gsi (&gsi, itercnt, true,
NULL_TREE, false,
GSI_CONTINUE_LINKING);
}
else
itercnt = s0;
}
tree type = TREE_TYPE (t);
if (POINTER_TYPE_P (type))
type = sizetype;
a = fold_build2 (MULT_EXPR, type,
fold_convert (type, itercnt),
fold_convert (type, OMP_CLAUSE_LINEAR_STEP (c)));
dest = unshare_expr (t);
t = fold_build2 (type == TREE_TYPE (t) ? PLUS_EXPR
: POINTER_PLUS_EXPR, TREE_TYPE (t), t, a);
t = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
assign_stmt = gimple_build_assign (dest, t);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
}
if (fd->collapse > 1)
expand_omp_for_init_vars (fd, &gsi, counts, inner_stmt, startvar);
if (!broken_loop)
{
gsi = gsi_last_nondebug_bb (cont_bb);
gomp_continue *cont_stmt = as_a <gomp_continue *> (gsi_stmt (gsi));
gcc_assert (gimple_code (cont_stmt) == GIMPLE_OMP_CONTINUE);
vmain = gimple_omp_continue_control_use (cont_stmt);
vback = gimple_omp_continue_control_def (cont_stmt);
if (!gimple_omp_for_combined_p (fd->for_stmt))
{
if (POINTER_TYPE_P (type))
t = fold_build_pointer_plus (vmain, step);
else
t = fold_build2 (PLUS_EXPR, type, vmain, step);
t = force_gimple_operand_gsi (&gsi, t,
DECL_P (vback)
&& TREE_ADDRESSABLE (vback),
NULL_TREE, true, GSI_SAME_STMT);
assign_stmt = gimple_build_assign (vback, t);
gsi_insert_before (&gsi, assign_stmt, GSI_SAME_STMT);
t = build2 (fd->loop.cond_code, boolean_type_node,
DECL_P (vback) && TREE_ADDRESSABLE (vback)
? t : vback, e);
gsi_insert_before (&gsi, gimple_build_cond_empty (t), GSI_SAME_STMT);
}
gsi_remove (&gsi, true);
if (fd->collapse > 1 && !gimple_omp_for_combined_p (fd->for_stmt))
collapse_bb = extract_omp_for_update_vars (fd, cont_bb, body_bb);
}
gsi = gsi_last_nondebug_bb (exit_bb);
if (!gimple_omp_return_nowait_p (gsi_stmt (gsi)))
{
t = gimple_omp_return_lhs (gsi_stmt (gsi));
gsi_insert_after (&gsi, omp_build_barrier (t), GSI_SAME_STMT);
}
gsi_remove (&gsi, true);
ep = make_edge (entry_bb, third_bb, EDGE_FALSE_VALUE);
ep->probability = profile_probability::guessed_always ().apply_scale (3, 4);
ep = find_edge (entry_bb, second_bb);
ep->flags = EDGE_TRUE_VALUE;
ep->probability = profile_probability::guessed_always ().apply_scale (1, 4);
find_edge (third_bb, seq_start_bb)->flags = EDGE_FALSE_VALUE;
find_edge (third_bb, fin_bb)->flags = EDGE_TRUE_VALUE;
if (!broken_loop)
{
ep = find_edge (cont_bb, body_bb);
if (ep == NULL)
{
ep = BRANCH_EDGE (cont_bb);
gcc_assert (single_succ (ep->dest) == body_bb);
}
if (gimple_omp_for_combined_p (fd->for_stmt))
{
remove_edge (ep);
ep = NULL;
}
else if (fd->collapse > 1)
{
remove_edge (ep);
ep = make_edge (cont_bb, collapse_bb, EDGE_TRUE_VALUE);
}
else
ep->flags = EDGE_TRUE_VALUE;
find_edge (cont_bb, fin_bb)->flags
= ep ? EDGE_FALSE_VALUE : EDGE_FALLTHRU;
}
set_immediate_dominator (CDI_DOMINATORS, second_bb, entry_bb);
set_immediate_dominator (CDI_DOMINATORS, third_bb, entry_bb);
set_immediate_dominator (CDI_DOMINATORS, seq_start_bb, third_bb);
set_immediate_dominator (CDI_DOMINATORS, body_bb,
recompute_dominator (CDI_DOMINATORS, body_bb));
set_immediate_dominator (CDI_DOMINATORS, fin_bb,
recompute_dominator (CDI_DOMINATORS, fin_bb));
struct loop *loop = body_bb->loop_father;
if (loop != entry_bb->loop_father)
{
gcc_assert (broken_loop || loop->header == body_bb);
gcc_assert (broken_loop
|| loop->latch == region->cont
|| single_pred (loop->latch) == region->cont);
return;
}
if (!broken_loop && !gimple_omp_for_combined_p (fd->for_stmt))
{
loop = alloc_loop ();
loop->header = body_bb;
if (collapse_bb == NULL)
loop->latch = cont_bb;
add_loop (loop, body_bb->loop_father);
}
}
static gphi *
find_phi_with_arg_on_edge (tree arg, edge e)
{
basic_block bb = e->dest;
for (gphi_iterator gpi = gsi_start_phis (bb);
!gsi_end_p (gpi);
gsi_next (&gpi))
{
gphi *phi = gpi.phi ();
if (PHI_ARG_DEF_FROM_EDGE (phi, e) == arg)
return phi;
}
return NULL;
}
static void
expand_omp_for_static_chunk (struct omp_region *region,
struct omp_for_data *fd, gimple *inner_stmt)
{
tree n, s0, e0, e, t;
tree trip_var, trip_init, trip_main, trip_back, nthreads, threadid;
tree type, itype, vmain, vback, vextra;
basic_block entry_bb, exit_bb, body_bb, seq_start_bb, iter_part_bb;
basic_block trip_update_bb = NULL, cont_bb, collapse_bb = NULL, fin_bb;
gimple_stmt_iterator gsi;
edge se;
bool broken_loop = region->cont == NULL;
tree *counts = NULL;
tree n1, n2, step;
itype = type = TREE_TYPE (fd->loop.v);
if (POINTER_TYPE_P (type))
itype = signed_type_for (type);
entry_bb = region->entry;
se = split_block (entry_bb, last_stmt (entry_bb));
entry_bb = se->src;
iter_part_bb = se->dest;
cont_bb = region->cont;
gcc_assert (EDGE_COUNT (iter_part_bb->succs) == 2);
fin_bb = BRANCH_EDGE (iter_part_bb)->dest;
gcc_assert (broken_loop
|| fin_bb == FALLTHRU_EDGE (cont_bb)->dest);
seq_start_bb = split_edge (FALLTHRU_EDGE (iter_part_bb));
body_bb = single_succ (seq_start_bb);
if (!broken_loop)
{
gcc_assert (BRANCH_EDGE (cont_bb)->dest == body_bb
|| single_succ (BRANCH_EDGE (cont_bb)->dest) == body_bb);
gcc_assert (EDGE_COUNT (cont_bb->succs) == 2);
trip_update_bb = split_edge (FALLTHRU_EDGE (cont_bb));
}
exit_bb = region->exit;
gsi = gsi_last_nondebug_bb (entry_bb);
gcc_assert (gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_FOR);
if (fd->collapse > 1)
{
int first_zero_iter = -1, dummy = -1;
basic_block l2_dom_bb = NULL, dummy_bb = NULL;
counts = XALLOCAVEC (tree, fd->collapse);
expand_omp_for_init_counts (fd, &gsi, entry_bb, counts,
fin_bb, first_zero_iter,
dummy_bb, dummy, l2_dom_bb);
t = NULL_TREE;
}
else if (gimple_omp_for_combined_into_p (fd->for_stmt))
t = integer_one_node;
else
t = fold_binary (fd->loop.cond_code, boolean_type_node,
fold_convert (type, fd->loop.n1),
fold_convert (type, fd->loop.n2));
if (fd->collapse == 1
&& TYPE_UNSIGNED (type)
&& (t == NULL_TREE || !integer_onep (t)))
{
n1 = fold_convert (type, unshare_expr (fd->loop.n1));
n1 = force_gimple_operand_gsi (&gsi, n1, true, NULL_TREE,
true, GSI_SAME_STMT);
n2 = fold_convert (type, unshare_expr (fd->loop.n2));
n2 = force_gimple_operand_gsi (&gsi, n2, true, NULL_TREE,
true, GSI_SAME_STMT);
gcond *cond_stmt = gimple_build_cond (fd->loop.cond_code, n1, n2,
NULL_TREE, NULL_TREE);
gsi_insert_before (&gsi, cond_stmt, GSI_SAME_STMT);
if (walk_tree (gimple_cond_lhs_ptr (cond_stmt),
expand_omp_regimplify_p, NULL, NULL)
|| walk_tree (gimple_cond_rhs_ptr (cond_stmt),
expand_omp_regimplify_p, NULL, NULL))
{
gsi = gsi_for_stmt (cond_stmt);
gimple_regimplify_operands (cond_stmt, &gsi);
}
se = split_block (entry_bb, cond_stmt);
se->flags = EDGE_TRUE_VALUE;
entry_bb = se->dest;
se->probability = profile_probability::very_likely ();
se = make_edge (se->src, fin_bb, EDGE_FALSE_VALUE);
se->probability = profile_probability::very_unlikely ();
if (gimple_in_ssa_p (cfun))
{
int dest_idx = find_edge (iter_part_bb, fin_bb)->dest_idx;
for (gphi_iterator gpi = gsi_start_phis (fin_bb);
!gsi_end_p (gpi); gsi_next (&gpi))
{
gphi *phi = gpi.phi ();
add_phi_arg (phi, gimple_phi_arg_def (phi, dest_idx),
se, UNKNOWN_LOCATION);
}
}
gsi = gsi_last_bb (entry_bb);
}
switch (gimple_omp_for_kind (fd->for_stmt))
{
case GF_OMP_FOR_KIND_FOR:
nthreads = builtin_decl_explicit (BUILT_IN_OMP_GET_NUM_THREADS);
threadid = builtin_decl_explicit (BUILT_IN_OMP_GET_THREAD_NUM);
break;
case GF_OMP_FOR_KIND_DISTRIBUTE:
nthreads = builtin_decl_explicit (BUILT_IN_OMP_GET_NUM_TEAMS);
threadid = builtin_decl_explicit (BUILT_IN_OMP_GET_TEAM_NUM);
break;
default:
gcc_unreachable ();
}
nthreads = build_call_expr (nthreads, 0);
nthreads = fold_convert (itype, nthreads);
nthreads = force_gimple_operand_gsi (&gsi, nthreads, true, NULL_TREE,
true, GSI_SAME_STMT);
threadid = build_call_expr (threadid, 0);
threadid = fold_convert (itype, threadid);
threadid = force_gimple_operand_gsi (&gsi, threadid, true, NULL_TREE,
true, GSI_SAME_STMT);
n1 = fd->loop.n1;
n2 = fd->loop.n2;
step = fd->loop.step;
if (gimple_omp_for_combined_into_p (fd->for_stmt))
{
tree innerc = omp_find_clause (gimple_omp_for_clauses (fd->for_stmt),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
n1 = OMP_CLAUSE_DECL (innerc);
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
n2 = OMP_CLAUSE_DECL (innerc);
}
n1 = force_gimple_operand_gsi (&gsi, fold_convert (type, n1),
true, NULL_TREE, true, GSI_SAME_STMT);
n2 = force_gimple_operand_gsi (&gsi, fold_convert (itype, n2),
true, NULL_TREE, true, GSI_SAME_STMT);
step = force_gimple_operand_gsi (&gsi, fold_convert (itype, step),
true, NULL_TREE, true, GSI_SAME_STMT);
tree chunk_size = fold_convert (itype, fd->chunk_size);
chunk_size = omp_adjust_chunk_size (chunk_size, fd->simd_schedule);
chunk_size
= force_gimple_operand_gsi (&gsi, chunk_size, true, NULL_TREE, true,
GSI_SAME_STMT);
t = build_int_cst (itype, (fd->loop.cond_code == LT_EXPR ? -1 : 1));
t = fold_build2 (PLUS_EXPR, itype, step, t);
t = fold_build2 (PLUS_EXPR, itype, t, n2);
t = fold_build2 (MINUS_EXPR, itype, t, fold_convert (itype, n1));
if (TYPE_UNSIGNED (itype) && fd->loop.cond_code == GT_EXPR)
t = fold_build2 (TRUNC_DIV_EXPR, itype,
fold_build1 (NEGATE_EXPR, itype, t),
fold_build1 (NEGATE_EXPR, itype, step));
else
t = fold_build2 (TRUNC_DIV_EXPR, itype, t, step);
t = fold_convert (itype, t);
n = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
true, GSI_SAME_STMT);
trip_var = create_tmp_reg (itype, ".trip");
if (gimple_in_ssa_p (cfun))
{
trip_init = make_ssa_name (trip_var);
trip_main = make_ssa_name (trip_var);
trip_back = make_ssa_name (trip_var);
}
else
{
trip_init = trip_var;
trip_main = trip_var;
trip_back = trip_var;
}
gassign *assign_stmt
= gimple_build_assign (trip_init, build_int_cst (itype, 0));
gsi_insert_before (&gsi, assign_stmt, GSI_SAME_STMT);
t = fold_build2 (MULT_EXPR, itype, threadid, chunk_size);
t = fold_build2 (MULT_EXPR, itype, t, step);
if (POINTER_TYPE_P (type))
t = fold_build_pointer_plus (n1, t);
else
t = fold_build2 (PLUS_EXPR, type, t, n1);
vextra = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
true, GSI_SAME_STMT);
gsi_remove (&gsi, true);
gimple_stmt_iterator gsif = gsi;
gsi = gsi_last_bb (iter_part_bb);
t = fold_build2 (MULT_EXPR, itype, trip_main, nthreads);
t = fold_build2 (PLUS_EXPR, itype, t, threadid);
t = fold_build2 (MULT_EXPR, itype, t, chunk_size);
s0 = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
t = fold_build2 (PLUS_EXPR, itype, s0, chunk_size);
t = fold_build2 (MIN_EXPR, itype, t, n);
e0 = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
t = build2 (LT_EXPR, boolean_type_node, s0, n);
gsi_insert_after (&gsi, gimple_build_cond_empty (t), GSI_CONTINUE_LINKING);
gsi = gsi_start_bb (seq_start_bb);
tree startvar = fd->loop.v;
tree endvar = NULL_TREE;
if (gimple_omp_for_combined_p (fd->for_stmt))
{
tree clauses = gimple_code (inner_stmt) == GIMPLE_OMP_PARALLEL
? gimple_omp_parallel_clauses (inner_stmt)
: gimple_omp_for_clauses (inner_stmt);
tree innerc = omp_find_clause (clauses, OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
startvar = OMP_CLAUSE_DECL (innerc);
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
endvar = OMP_CLAUSE_DECL (innerc);
if (fd->collapse > 1 && TREE_CODE (fd->loop.n2) != INTEGER_CST
&& gimple_omp_for_kind (fd->for_stmt) == GF_OMP_FOR_KIND_DISTRIBUTE)
{
int i;
for (i = 1; i < fd->collapse; i++)
{
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
}
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
if (innerc)
{
tree t = fold_convert (TREE_TYPE (OMP_CLAUSE_DECL (innerc)),
fd->loop.n2);
t = force_gimple_operand_gsi (&gsi, t, false, NULL_TREE, false,
GSI_CONTINUE_LINKING);
assign_stmt = gimple_build_assign (OMP_CLAUSE_DECL (innerc), t);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
}
}
}
t = fold_convert (itype, s0);
t = fold_build2 (MULT_EXPR, itype, t, step);
if (POINTER_TYPE_P (type))
t = fold_build_pointer_plus (n1, t);
else
t = fold_build2 (PLUS_EXPR, type, t, n1);
t = fold_convert (TREE_TYPE (startvar), t);
t = force_gimple_operand_gsi (&gsi, t,
DECL_P (startvar)
&& TREE_ADDRESSABLE (startvar),
NULL_TREE, false, GSI_CONTINUE_LINKING);
assign_stmt = gimple_build_assign (startvar, t);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
t = fold_convert (itype, e0);
t = fold_build2 (MULT_EXPR, itype, t, step);
if (POINTER_TYPE_P (type))
t = fold_build_pointer_plus (n1, t);
else
t = fold_build2 (PLUS_EXPR, type, t, n1);
t = fold_convert (TREE_TYPE (startvar), t);
e = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
if (endvar)
{
assign_stmt = gimple_build_assign (endvar, e);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
if (useless_type_conversion_p (TREE_TYPE (fd->loop.v), TREE_TYPE (e)))
assign_stmt = gimple_build_assign (fd->loop.v, e);
else
assign_stmt = gimple_build_assign (fd->loop.v, NOP_EXPR, e);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
}
tree itercnt = NULL_TREE, itercntbias = NULL_TREE;
if (gimple_omp_for_kind (fd->for_stmt) == GF_OMP_FOR_KIND_FOR)
for (tree c = gimple_omp_for_clauses (fd->for_stmt);
c; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LINEAR
&& !OMP_CLAUSE_LINEAR_NO_COPYIN (c))
{
tree d = OMP_CLAUSE_DECL (c);
bool is_ref = omp_is_reference (d);
tree t = d, a, dest;
if (is_ref)
t = build_simple_mem_ref_loc (OMP_CLAUSE_LOCATION (c), t);
tree type = TREE_TYPE (t);
if (POINTER_TYPE_P (type))
type = sizetype;
dest = unshare_expr (t);
tree v = create_tmp_var (TREE_TYPE (t), NULL);
expand_omp_build_assign (&gsif, v, t);
if (itercnt == NULL_TREE)
{
if (gimple_omp_for_combined_into_p (fd->for_stmt))
{
itercntbias
= fold_build2 (MINUS_EXPR, itype, fold_convert (itype, n1),
fold_convert (itype, fd->loop.n1));
itercntbias = fold_build2 (EXACT_DIV_EXPR, itype,
itercntbias, step);
itercntbias
= force_gimple_operand_gsi (&gsif, itercntbias, true,
NULL_TREE, true,
GSI_SAME_STMT);
itercnt = fold_build2 (PLUS_EXPR, itype, itercntbias, s0);
itercnt = force_gimple_operand_gsi (&gsi, itercnt, true,
NULL_TREE, false,
GSI_CONTINUE_LINKING);
}
else
itercnt = s0;
}
a = fold_build2 (MULT_EXPR, type,
fold_convert (type, itercnt),
fold_convert (type, OMP_CLAUSE_LINEAR_STEP (c)));
t = fold_build2 (type == TREE_TYPE (t) ? PLUS_EXPR
: POINTER_PLUS_EXPR, TREE_TYPE (t), v, a);
t = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
assign_stmt = gimple_build_assign (dest, t);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
}
if (fd->collapse > 1)
expand_omp_for_init_vars (fd, &gsi, counts, inner_stmt, startvar);
if (!broken_loop)
{
gsi = gsi_last_nondebug_bb (cont_bb);
gomp_continue *cont_stmt = as_a <gomp_continue *> (gsi_stmt (gsi));
vmain = gimple_omp_continue_control_use (cont_stmt);
vback = gimple_omp_continue_control_def (cont_stmt);
if (!gimple_omp_for_combined_p (fd->for_stmt))
{
if (POINTER_TYPE_P (type))
t = fold_build_pointer_plus (vmain, step);
else
t = fold_build2 (PLUS_EXPR, type, vmain, step);
if (DECL_P (vback) && TREE_ADDRESSABLE (vback))
t = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
true, GSI_SAME_STMT);
assign_stmt = gimple_build_assign (vback, t);
gsi_insert_before (&gsi, assign_stmt, GSI_SAME_STMT);
if (tree_int_cst_equal (fd->chunk_size, integer_one_node))
t = build2 (EQ_EXPR, boolean_type_node,
build_int_cst (itype, 0),
build_int_cst (itype, 1));
else
t = build2 (fd->loop.cond_code, boolean_type_node,
DECL_P (vback) && TREE_ADDRESSABLE (vback)
? t : vback, e);
gsi_insert_before (&gsi, gimple_build_cond_empty (t), GSI_SAME_STMT);
}
gsi_remove (&gsi, true);
if (fd->collapse > 1 && !gimple_omp_for_combined_p (fd->for_stmt))
collapse_bb = extract_omp_for_update_vars (fd, cont_bb, body_bb);
gsi = gsi_start_bb (trip_update_bb);
t = build_int_cst (itype, 1);
t = build2 (PLUS_EXPR, itype, trip_main, t);
assign_stmt = gimple_build_assign (trip_back, t);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
}
gsi = gsi_last_nondebug_bb (exit_bb);
if (!gimple_omp_return_nowait_p (gsi_stmt (gsi)))
{
t = gimple_omp_return_lhs (gsi_stmt (gsi));
gsi_insert_after (&gsi, omp_build_barrier (t), GSI_SAME_STMT);
}
gsi_remove (&gsi, true);
find_edge (iter_part_bb, seq_start_bb)->flags = EDGE_TRUE_VALUE;
find_edge (iter_part_bb, fin_bb)->flags = EDGE_FALSE_VALUE;
if (!broken_loop)
{
se = find_edge (cont_bb, body_bb);
if (se == NULL)
{
se = BRANCH_EDGE (cont_bb);
gcc_assert (single_succ (se->dest) == body_bb);
}
if (gimple_omp_for_combined_p (fd->for_stmt))
{
remove_edge (se);
se = NULL;
}
else if (fd->collapse > 1)
{
remove_edge (se);
se = make_edge (cont_bb, collapse_bb, EDGE_TRUE_VALUE);
}
else
se->flags = EDGE_TRUE_VALUE;
find_edge (cont_bb, trip_update_bb)->flags
= se ? EDGE_FALSE_VALUE : EDGE_FALLTHRU;
redirect_edge_and_branch (single_succ_edge (trip_update_bb),
iter_part_bb);
}
if (gimple_in_ssa_p (cfun))
{
gphi_iterator psi;
gphi *phi;
edge re, ene;
edge_var_map *vm;
size_t i;
gcc_assert (fd->collapse == 1 && !broken_loop);
se = find_edge (iter_part_bb, fin_bb);
re = single_succ_edge (trip_update_bb);
vec<edge_var_map> *head = redirect_edge_var_map_vector (re);
ene = single_succ_edge (entry_bb);
psi = gsi_start_phis (fin_bb);
for (i = 0; !gsi_end_p (psi) && head->iterate (i, &vm);
gsi_next (&psi), ++i)
{
gphi *nphi;
source_location locus;
phi = psi.phi ();
if (operand_equal_p (gimple_phi_arg_def (phi, 0),
redirect_edge_var_map_def (vm), 0))
continue;
t = gimple_phi_result (phi);
gcc_assert (t == redirect_edge_var_map_result (vm));
if (!single_pred_p (fin_bb))
t = copy_ssa_name (t, phi);
nphi = create_phi_node (t, iter_part_bb);
t = PHI_ARG_DEF_FROM_EDGE (phi, se);
locus = gimple_phi_arg_location_from_edge (phi, se);
if (t == fd->loop.v)
t = vextra;
add_phi_arg (nphi, t, ene, locus);
locus = redirect_edge_var_map_location (vm);
tree back_arg = redirect_edge_var_map_def (vm);
add_phi_arg (nphi, back_arg, re, locus);
edge ce = find_edge (cont_bb, body_bb);
if (ce == NULL)
{
ce = BRANCH_EDGE (cont_bb);
gcc_assert (single_succ (ce->dest) == body_bb);
ce = single_succ_edge (ce->dest);
}
gphi *inner_loop_phi = find_phi_with_arg_on_edge (back_arg, ce);
gcc_assert (inner_loop_phi != NULL);
add_phi_arg (inner_loop_phi, gimple_phi_result (nphi),
find_edge (seq_start_bb, body_bb), locus);
if (!single_pred_p (fin_bb))
add_phi_arg (phi, gimple_phi_result (nphi), se, locus);
}
gcc_assert (gsi_end_p (psi) && (head == NULL || i == head->length ()));
redirect_edge_var_map_clear (re);
if (single_pred_p (fin_bb))
while (1)
{
psi = gsi_start_phis (fin_bb);
if (gsi_end_p (psi))
break;
remove_phi_node (&psi, false);
}
phi = create_phi_node (trip_main, iter_part_bb);
add_phi_arg (phi, trip_back, single_succ_edge (trip_update_bb),
UNKNOWN_LOCATION);
add_phi_arg (phi, trip_init, single_succ_edge (entry_bb),
UNKNOWN_LOCATION);
}
if (!broken_loop)
set_immediate_dominator (CDI_DOMINATORS, trip_update_bb, cont_bb);
set_immediate_dominator (CDI_DOMINATORS, iter_part_bb,
recompute_dominator (CDI_DOMINATORS, iter_part_bb));
set_immediate_dominator (CDI_DOMINATORS, fin_bb,
recompute_dominator (CDI_DOMINATORS, fin_bb));
set_immediate_dominator (CDI_DOMINATORS, seq_start_bb,
recompute_dominator (CDI_DOMINATORS, seq_start_bb));
set_immediate_dominator (CDI_DOMINATORS, body_bb,
recompute_dominator (CDI_DOMINATORS, body_bb));
if (!broken_loop)
{
struct loop *loop = body_bb->loop_father;
struct loop *trip_loop = alloc_loop ();
trip_loop->header = iter_part_bb;
trip_loop->latch = trip_update_bb;
add_loop (trip_loop, iter_part_bb->loop_father);
if (loop != entry_bb->loop_father)
{
gcc_assert (loop->header == body_bb);
gcc_assert (loop->latch == region->cont
|| single_pred (loop->latch) == region->cont);
trip_loop->inner = loop;
return;
}
if (!gimple_omp_for_combined_p (fd->for_stmt))
{
loop = alloc_loop ();
loop->header = body_bb;
if (collapse_bb == NULL)
loop->latch = cont_bb;
add_loop (loop, trip_loop);
}
}
}
static void
expand_omp_simd (struct omp_region *region, struct omp_for_data *fd)
{
tree type, t;
basic_block entry_bb, cont_bb, exit_bb, l0_bb, l1_bb, l2_bb, l2_dom_bb;
gimple_stmt_iterator gsi;
gimple *stmt;
gcond *cond_stmt;
bool broken_loop = region->cont == NULL;
edge e, ne;
tree *counts = NULL;
int i;
int safelen_int = INT_MAX;
tree safelen = omp_find_clause (gimple_omp_for_clauses (fd->for_stmt),
OMP_CLAUSE_SAFELEN);
tree simduid = omp_find_clause (gimple_omp_for_clauses (fd->for_stmt),
OMP_CLAUSE__SIMDUID_);
tree n1, n2;
if (safelen)
{
poly_uint64 val;
safelen = OMP_CLAUSE_SAFELEN_EXPR (safelen);
if (!poly_int_tree_p (safelen, &val))
safelen_int = 0;
else
safelen_int = MIN (constant_lower_bound (val), INT_MAX);
if (safelen_int == 1)
safelen_int = 0;
}
type = TREE_TYPE (fd->loop.v);
entry_bb = region->entry;
cont_bb = region->cont;
gcc_assert (EDGE_COUNT (entry_bb->succs) == 2);
gcc_assert (broken_loop
|| BRANCH_EDGE (entry_bb)->dest == FALLTHRU_EDGE (cont_bb)->dest);
l0_bb = FALLTHRU_EDGE (entry_bb)->dest;
if (!broken_loop)
{
gcc_assert (BRANCH_EDGE (cont_bb)->dest == l0_bb);
gcc_assert (EDGE_COUNT (cont_bb->succs) == 2);
l1_bb = split_block (cont_bb, last_stmt (cont_bb))->dest;
l2_bb = BRANCH_EDGE (entry_bb)->dest;
}
else
{
BRANCH_EDGE (entry_bb)->flags &= ~EDGE_ABNORMAL;
l1_bb = split_edge (BRANCH_EDGE (entry_bb));
l2_bb = single_succ (l1_bb);
}
exit_bb = region->exit;
l2_dom_bb = NULL;
gsi = gsi_last_nondebug_bb (entry_bb);
gcc_assert (gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_FOR);
gcc_assert (!gimple_in_ssa_p (cfun));
if (fd->collapse > 1)
{
int first_zero_iter = -1, dummy = -1;
basic_block zero_iter_bb = l2_bb, dummy_bb = NULL;
counts = XALLOCAVEC (tree, fd->collapse);
expand_omp_for_init_counts (fd, &gsi, entry_bb, counts,
zero_iter_bb, first_zero_iter,
dummy_bb, dummy, l2_dom_bb);
}
if (l2_dom_bb == NULL)
l2_dom_bb = l1_bb;
n1 = fd->loop.n1;
n2 = fd->loop.n2;
if (gimple_omp_for_combined_into_p (fd->for_stmt))
{
tree innerc = omp_find_clause (gimple_omp_for_clauses (fd->for_stmt),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
n1 = OMP_CLAUSE_DECL (innerc);
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
n2 = OMP_CLAUSE_DECL (innerc);
}
tree step = fd->loop.step;
bool is_simt = omp_find_clause (gimple_omp_for_clauses (fd->for_stmt),
OMP_CLAUSE__SIMT_);
if (is_simt)
{
cfun->curr_properties &= ~PROP_gimple_lomp_dev;
is_simt = safelen_int > 1;
}
tree simt_lane = NULL_TREE, simt_maxlane = NULL_TREE;
if (is_simt)
{
simt_lane = create_tmp_var (unsigned_type_node);
gimple *g = gimple_build_call_internal (IFN_GOMP_SIMT_LANE, 0);
gimple_call_set_lhs (g, simt_lane);
gsi_insert_before (&gsi, g, GSI_SAME_STMT);
tree offset = fold_build2 (MULT_EXPR, TREE_TYPE (step), step,
fold_convert (TREE_TYPE (step), simt_lane));
n1 = fold_convert (type, n1);
if (POINTER_TYPE_P (type))
n1 = fold_build_pointer_plus (n1, offset);
else
n1 = fold_build2 (PLUS_EXPR, type, n1, fold_convert (type, offset));
if (fd->collapse > 1)
simt_maxlane = build_one_cst (unsigned_type_node);
else if (safelen_int < omp_max_simt_vf ())
simt_maxlane = build_int_cst (unsigned_type_node, safelen_int);
tree vf
= build_call_expr_internal_loc (UNKNOWN_LOCATION, IFN_GOMP_SIMT_VF,
unsigned_type_node, 0);
if (simt_maxlane)
vf = fold_build2 (MIN_EXPR, unsigned_type_node, vf, simt_maxlane);
vf = fold_convert (TREE_TYPE (step), vf);
step = fold_build2 (MULT_EXPR, TREE_TYPE (step), step, vf);
}
expand_omp_build_assign (&gsi, fd->loop.v, fold_convert (type, n1));
if (fd->collapse > 1)
{
if (gimple_omp_for_combined_into_p (fd->for_stmt))
{
gsi_prev (&gsi);
expand_omp_for_init_vars (fd, &gsi, counts, NULL, n1);
gsi_next (&gsi);
}
else
for (i = 0; i < fd->collapse; i++)
{
tree itype = TREE_TYPE (fd->loops[i].v);
if (POINTER_TYPE_P (itype))
itype = signed_type_for (itype);
t = fold_convert (TREE_TYPE (fd->loops[i].v), fd->loops[i].n1);
expand_omp_build_assign (&gsi, fd->loops[i].v, t);
}
}
gsi_remove (&gsi, true);
if (!broken_loop)
{
gsi = gsi_last_nondebug_bb (cont_bb);
stmt = gsi_stmt (gsi);
gcc_assert (gimple_code (stmt) == GIMPLE_OMP_CONTINUE);
if (POINTER_TYPE_P (type))
t = fold_build_pointer_plus (fd->loop.v, step);
else
t = fold_build2 (PLUS_EXPR, type, fd->loop.v, step);
expand_omp_build_assign (&gsi, fd->loop.v, t);
if (fd->collapse > 1)
{
i = fd->collapse - 1;
if (POINTER_TYPE_P (TREE_TYPE (fd->loops[i].v)))
{
t = fold_convert (sizetype, fd->loops[i].step);
t = fold_build_pointer_plus (fd->loops[i].v, t);
}
else
{
t = fold_convert (TREE_TYPE (fd->loops[i].v),
fd->loops[i].step);
t = fold_build2 (PLUS_EXPR, TREE_TYPE (fd->loops[i].v),
fd->loops[i].v, t);
}
expand_omp_build_assign (&gsi, fd->loops[i].v, t);
for (i = fd->collapse - 1; i > 0; i--)
{
tree itype = TREE_TYPE (fd->loops[i].v);
tree itype2 = TREE_TYPE (fd->loops[i - 1].v);
if (POINTER_TYPE_P (itype2))
itype2 = signed_type_for (itype2);
t = fold_convert (itype2, fd->loops[i - 1].step);
t = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE, true,
GSI_SAME_STMT);
t = build3 (COND_EXPR, itype2,
build2 (fd->loops[i].cond_code, boolean_type_node,
fd->loops[i].v,
fold_convert (itype, fd->loops[i].n2)),
build_int_cst (itype2, 0), t);
if (POINTER_TYPE_P (TREE_TYPE (fd->loops[i - 1].v)))
t = fold_build_pointer_plus (fd->loops[i - 1].v, t);
else
t = fold_build2 (PLUS_EXPR, itype2, fd->loops[i - 1].v, t);
expand_omp_build_assign (&gsi, fd->loops[i - 1].v, t);
t = fold_convert (itype, fd->loops[i].n1);
t = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE, true,
GSI_SAME_STMT);
t = build3 (COND_EXPR, itype,
build2 (fd->loops[i].cond_code, boolean_type_node,
fd->loops[i].v,
fold_convert (itype, fd->loops[i].n2)),
fd->loops[i].v, t);
expand_omp_build_assign (&gsi, fd->loops[i].v, t);
}
}
gsi_remove (&gsi, true);
}
gsi = gsi_start_bb (l1_bb);
t = fold_convert (type, n2);
t = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
tree v = fd->loop.v;
if (DECL_P (v) && TREE_ADDRESSABLE (v))
v = force_gimple_operand_gsi (&gsi, v, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
t = build2 (fd->loop.cond_code, boolean_type_node, v, t);
cond_stmt = gimple_build_cond_empty (t);
gsi_insert_after (&gsi, cond_stmt, GSI_CONTINUE_LINKING);
if (walk_tree (gimple_cond_lhs_ptr (cond_stmt), expand_omp_regimplify_p,
NULL, NULL)
|| walk_tree (gimple_cond_rhs_ptr (cond_stmt), expand_omp_regimplify_p,
NULL, NULL))
{
gsi = gsi_for_stmt (cond_stmt);
gimple_regimplify_operands (cond_stmt, &gsi);
}
if (is_simt)
{
gsi = gsi_start_bb (l2_bb);
step = fold_build2 (MINUS_EXPR, TREE_TYPE (step), fd->loop.step, step);
if (POINTER_TYPE_P (type))
t = fold_build_pointer_plus (fd->loop.v, step);
else
t = fold_build2 (PLUS_EXPR, type, fd->loop.v, step);
expand_omp_build_assign (&gsi, fd->loop.v, t);
}
gsi = gsi_last_nondebug_bb (exit_bb);
gsi_remove (&gsi, true);
remove_edge (FALLTHRU_EDGE (entry_bb));
if (!broken_loop)
{
remove_edge (BRANCH_EDGE (entry_bb));
make_edge (entry_bb, l1_bb, EDGE_FALLTHRU);
e = BRANCH_EDGE (l1_bb);
ne = FALLTHRU_EDGE (l1_bb);
e->flags = EDGE_TRUE_VALUE;
}
else
{
single_succ_edge (entry_bb)->flags = EDGE_FALLTHRU;
ne = single_succ_edge (l1_bb);
e = make_edge (l1_bb, l0_bb, EDGE_TRUE_VALUE);
}
ne->flags = EDGE_FALSE_VALUE;
e->probability = profile_probability::guessed_always ().apply_scale (7, 8);
ne->probability = e->probability.invert ();
set_immediate_dominator (CDI_DOMINATORS, l1_bb, entry_bb);
set_immediate_dominator (CDI_DOMINATORS, l0_bb, l1_bb);
if (simt_maxlane)
{
cond_stmt = gimple_build_cond (LT_EXPR, simt_lane, simt_maxlane,
NULL_TREE, NULL_TREE);
gsi = gsi_last_bb (entry_bb);
gsi_insert_after (&gsi, cond_stmt, GSI_NEW_STMT);
make_edge (entry_bb, l2_bb, EDGE_FALSE_VALUE);
FALLTHRU_EDGE (entry_bb)->flags = EDGE_TRUE_VALUE;
FALLTHRU_EDGE (entry_bb)->probability
= profile_probability::guessed_always ().apply_scale (7, 8);
BRANCH_EDGE (entry_bb)->probability 
= FALLTHRU_EDGE (entry_bb)->probability.invert ();
l2_dom_bb = entry_bb;
}
set_immediate_dominator (CDI_DOMINATORS, l2_bb, l2_dom_bb);
if (!broken_loop)
{
struct loop *loop = alloc_loop ();
loop->header = l1_bb;
loop->latch = cont_bb;
add_loop (loop, l1_bb->loop_father);
loop->safelen = safelen_int;
if (simduid)
{
loop->simduid = OMP_CLAUSE__SIMDUID__DECL (simduid);
cfun->has_simduid_loops = true;
}
if ((flag_tree_loop_vectorize
|| !global_options_set.x_flag_tree_loop_vectorize)
&& flag_tree_loop_optimize
&& loop->safelen > 1)
{
loop->force_vectorize = true;
cfun->has_force_vectorize_loops = true;
}
}
else if (simduid)
cfun->has_simduid_loops = true;
}
static void
expand_omp_taskloop_for_outer (struct omp_region *region,
struct omp_for_data *fd,
gimple *inner_stmt)
{
tree type, bias = NULL_TREE;
basic_block entry_bb, cont_bb, exit_bb;
gimple_stmt_iterator gsi;
gassign *assign_stmt;
tree *counts = NULL;
int i;
gcc_assert (inner_stmt);
gcc_assert (region->cont);
gcc_assert (gimple_code (inner_stmt) == GIMPLE_OMP_TASK
&& gimple_omp_task_taskloop_p (inner_stmt));
type = TREE_TYPE (fd->loop.v);
if (fd->iter_type == long_long_unsigned_type_node
&& TREE_CODE (type) == INTEGER_TYPE
&& !TYPE_UNSIGNED (type))
{
tree n1, n2;
if (fd->loop.cond_code == LT_EXPR)
{
n1 = fd->loop.n1;
n2 = fold_build2 (PLUS_EXPR, type, fd->loop.n2, fd->loop.step);
}
else
{
n1 = fold_build2 (MINUS_EXPR, type, fd->loop.n2, fd->loop.step);
n2 = fd->loop.n1;
}
if (TREE_CODE (n1) != INTEGER_CST
|| TREE_CODE (n2) != INTEGER_CST
|| ((tree_int_cst_sgn (n1) < 0) ^ (tree_int_cst_sgn (n2) < 0)))
bias = fold_convert (fd->iter_type, TYPE_MIN_VALUE (type));
}
entry_bb = region->entry;
cont_bb = region->cont;
gcc_assert (EDGE_COUNT (entry_bb->succs) == 2);
gcc_assert (BRANCH_EDGE (entry_bb)->dest == FALLTHRU_EDGE (cont_bb)->dest);
exit_bb = region->exit;
gsi = gsi_last_nondebug_bb (entry_bb);
gimple *for_stmt = gsi_stmt (gsi);
gcc_assert (gimple_code (for_stmt) == GIMPLE_OMP_FOR);
if (fd->collapse > 1)
{
int first_zero_iter = -1, dummy = -1;
basic_block zero_iter_bb = NULL, dummy_bb = NULL, l2_dom_bb = NULL;
counts = XALLOCAVEC (tree, fd->collapse);
expand_omp_for_init_counts (fd, &gsi, entry_bb, counts,
zero_iter_bb, first_zero_iter,
dummy_bb, dummy, l2_dom_bb);
if (zero_iter_bb)
{
for (i = first_zero_iter; i < fd->collapse; i++)
if (SSA_VAR_P (counts[i]))
TREE_NO_WARNING (counts[i]) = 1;
gsi_prev (&gsi);
edge e = split_block (entry_bb, gsi_stmt (gsi));
entry_bb = e->dest;
make_edge (zero_iter_bb, entry_bb, EDGE_FALLTHRU);
gsi = gsi_last_bb (entry_bb);
set_immediate_dominator (CDI_DOMINATORS, entry_bb,
get_immediate_dominator (CDI_DOMINATORS,
zero_iter_bb));
}
}
tree t0, t1;
t1 = fd->loop.n2;
t0 = fd->loop.n1;
if (POINTER_TYPE_P (TREE_TYPE (t0))
&& TYPE_PRECISION (TREE_TYPE (t0))
!= TYPE_PRECISION (fd->iter_type))
{
tree itype = signed_type_for (type);
t1 = fold_convert (fd->iter_type, fold_convert (itype, t1));
t0 = fold_convert (fd->iter_type, fold_convert (itype, t0));
}
else
{
t1 = fold_convert (fd->iter_type, t1);
t0 = fold_convert (fd->iter_type, t0);
}
if (bias)
{
t1 = fold_build2 (PLUS_EXPR, fd->iter_type, t1, bias);
t0 = fold_build2 (PLUS_EXPR, fd->iter_type, t0, bias);
}
tree innerc = omp_find_clause (gimple_omp_task_clauses (inner_stmt),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
tree startvar = OMP_CLAUSE_DECL (innerc);
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc), OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
tree endvar = OMP_CLAUSE_DECL (innerc);
if (fd->collapse > 1 && TREE_CODE (fd->loop.n2) != INTEGER_CST)
{
gcc_assert (innerc);
for (i = 1; i < fd->collapse; i++)
{
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
}
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
if (innerc)
{
tree t = force_gimple_operand_gsi (&gsi, fd->loop.n2, false,
NULL_TREE, false,
GSI_CONTINUE_LINKING);
assign_stmt = gimple_build_assign (OMP_CLAUSE_DECL (innerc), t);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
}
}
t0 = force_gimple_operand_gsi (&gsi, t0, false, NULL_TREE, false,
GSI_CONTINUE_LINKING);
assign_stmt = gimple_build_assign (startvar, t0);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
t1 = force_gimple_operand_gsi (&gsi, t1, false, NULL_TREE, false,
GSI_CONTINUE_LINKING);
assign_stmt = gimple_build_assign (endvar, t1);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
if (fd->collapse > 1)
expand_omp_for_init_vars (fd, &gsi, counts, inner_stmt, startvar);
gsi = gsi_for_stmt (for_stmt);
gsi_remove (&gsi, true);
gsi = gsi_last_nondebug_bb (cont_bb);
gsi_remove (&gsi, true);
gsi = gsi_last_nondebug_bb (exit_bb);
gsi_remove (&gsi, true);
FALLTHRU_EDGE (entry_bb)->probability = profile_probability::always ();
remove_edge (BRANCH_EDGE (entry_bb));
FALLTHRU_EDGE (cont_bb)->probability = profile_probability::always ();
remove_edge (BRANCH_EDGE (cont_bb));
set_immediate_dominator (CDI_DOMINATORS, exit_bb, cont_bb);
set_immediate_dominator (CDI_DOMINATORS, region->entry,
recompute_dominator (CDI_DOMINATORS, region->entry));
}
static void
expand_omp_taskloop_for_inner (struct omp_region *region,
struct omp_for_data *fd,
gimple *inner_stmt)
{
tree e, t, type, itype, vmain, vback, bias = NULL_TREE;
basic_block entry_bb, exit_bb, body_bb, cont_bb, collapse_bb = NULL;
basic_block fin_bb;
gimple_stmt_iterator gsi;
edge ep;
bool broken_loop = region->cont == NULL;
tree *counts = NULL;
tree n1, n2, step;
itype = type = TREE_TYPE (fd->loop.v);
if (POINTER_TYPE_P (type))
itype = signed_type_for (type);
if (fd->iter_type == long_long_unsigned_type_node
&& TREE_CODE (type) == INTEGER_TYPE
&& !TYPE_UNSIGNED (type))
{
tree n1, n2;
if (fd->loop.cond_code == LT_EXPR)
{
n1 = fd->loop.n1;
n2 = fold_build2 (PLUS_EXPR, type, fd->loop.n2, fd->loop.step);
}
else
{
n1 = fold_build2 (MINUS_EXPR, type, fd->loop.n2, fd->loop.step);
n2 = fd->loop.n1;
}
if (TREE_CODE (n1) != INTEGER_CST
|| TREE_CODE (n2) != INTEGER_CST
|| ((tree_int_cst_sgn (n1) < 0) ^ (tree_int_cst_sgn (n2) < 0)))
bias = fold_convert (fd->iter_type, TYPE_MIN_VALUE (type));
}
entry_bb = region->entry;
cont_bb = region->cont;
gcc_assert (EDGE_COUNT (entry_bb->succs) == 2);
fin_bb = BRANCH_EDGE (entry_bb)->dest;
gcc_assert (broken_loop
|| (fin_bb == FALLTHRU_EDGE (cont_bb)->dest));
body_bb = FALLTHRU_EDGE (entry_bb)->dest;
if (!broken_loop)
{
gcc_assert (BRANCH_EDGE (cont_bb)->dest == body_bb);
gcc_assert (EDGE_COUNT (cont_bb->succs) == 2);
}
exit_bb = region->exit;
gsi = gsi_last_nondebug_bb (entry_bb);
gcc_assert (gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_FOR);
if (fd->collapse > 1)
{
int first_zero_iter = -1, dummy = -1;
basic_block l2_dom_bb = NULL, dummy_bb = NULL;
counts = XALLOCAVEC (tree, fd->collapse);
expand_omp_for_init_counts (fd, &gsi, entry_bb, counts,
fin_bb, first_zero_iter,
dummy_bb, dummy, l2_dom_bb);
t = NULL_TREE;
}
else
t = integer_one_node;
step = fd->loop.step;
tree innerc = omp_find_clause (gimple_omp_for_clauses (fd->for_stmt),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
n1 = OMP_CLAUSE_DECL (innerc);
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc), OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
n2 = OMP_CLAUSE_DECL (innerc);
if (bias)
{
n1 = fold_build2 (PLUS_EXPR, fd->iter_type, n1, bias);
n2 = fold_build2 (PLUS_EXPR, fd->iter_type, n2, bias);
}
n1 = force_gimple_operand_gsi (&gsi, fold_convert (type, n1),
true, NULL_TREE, true, GSI_SAME_STMT);
n2 = force_gimple_operand_gsi (&gsi, fold_convert (itype, n2),
true, NULL_TREE, true, GSI_SAME_STMT);
step = force_gimple_operand_gsi (&gsi, fold_convert (itype, step),
true, NULL_TREE, true, GSI_SAME_STMT);
tree startvar = fd->loop.v;
tree endvar = NULL_TREE;
if (gimple_omp_for_combined_p (fd->for_stmt))
{
tree clauses = gimple_omp_for_clauses (inner_stmt);
tree innerc = omp_find_clause (clauses, OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
startvar = OMP_CLAUSE_DECL (innerc);
innerc = omp_find_clause (OMP_CLAUSE_CHAIN (innerc),
OMP_CLAUSE__LOOPTEMP_);
gcc_assert (innerc);
endvar = OMP_CLAUSE_DECL (innerc);
}
t = fold_convert (TREE_TYPE (startvar), n1);
t = force_gimple_operand_gsi (&gsi, t,
DECL_P (startvar)
&& TREE_ADDRESSABLE (startvar),
NULL_TREE, false, GSI_CONTINUE_LINKING);
gimple *assign_stmt = gimple_build_assign (startvar, t);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
t = fold_convert (TREE_TYPE (startvar), n2);
e = force_gimple_operand_gsi (&gsi, t, true, NULL_TREE,
false, GSI_CONTINUE_LINKING);
if (endvar)
{
assign_stmt = gimple_build_assign (endvar, e);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
if (useless_type_conversion_p (TREE_TYPE (fd->loop.v), TREE_TYPE (e)))
assign_stmt = gimple_build_assign (fd->loop.v, e);
else
assign_stmt = gimple_build_assign (fd->loop.v, NOP_EXPR, e);
gsi_insert_after (&gsi, assign_stmt, GSI_CONTINUE_LINKING);
}
if (fd->collapse > 1)
expand_omp_for_init_vars (fd, &gsi, counts, inner_stmt, startvar);
if (!broken_loop)
{
gsi = gsi_last_nondebug_bb (cont_bb);
gomp_continue *cont_stmt = as_a <gomp_continue *> (gsi_stmt (gsi));
gcc_assert (gimple_code (cont_stmt) == GIMPLE_OMP_CONTINUE);
vmain = gimple_omp_continue_control_use (cont_stmt);
vback = gimple_omp_continue_control_def (cont_stmt);
if (!gimple_omp_for_combined_p (fd->for_stmt))
{
if (POINTER_TYPE_P (type))
t = fold_build_pointer_plus (vmain, step);
else
t = fold_build2 (PLUS_EXPR, type, vmain, step);
t = force_gimple_operand_gsi (&gsi, t,
DECL_P (vback)
&& TREE_ADDRESSABLE (vback),
NULL_TREE, true, GSI_SAME_STMT);
assign_stmt = gimple_build_assign (vback, t);
gsi_insert_before (&gsi, assign_stmt, GSI_SAME_STMT);
t = build2 (fd->loop.cond_code, boolean_type_node,
DECL_P (vback) && TREE_ADDRESSABLE (vback)
? t : vback, e);
gsi_insert_before (&gsi, gimple_build_cond_empty (t), GSI_SAME_STMT);
}
gsi_remove (&gsi, true);
if (fd->collapse > 1 && !gimple_omp_for_combined_p (fd->for_stmt))
collapse_bb = extract_omp_for_update_vars (fd, cont_bb, body_bb);
}
gsi = gsi_for_stmt (fd->for_stmt);
gsi_remove (&gsi, true);
gsi = gsi_last_nondebug_bb (exit_bb);
gsi_remove (&gsi, true);
FALLTHRU_EDGE (entry_bb)->probability = profile_probability::always ();
if (!broken_loop)
remove_edge (BRANCH_EDGE (entry_bb));
else
{
remove_edge_and_dominated_blocks (BRANCH_EDGE (entry_bb));
region->outer->cont = NULL;
}
if (!broken_loop)
{
ep = find_edge (cont_bb, body_bb);
if (gimple_omp_for_combined_p (fd->for_stmt))
{
remove_edge (ep);
ep = NULL;
}
else if (fd->collapse > 1)
{
remove_edge (ep);
ep = make_edge (cont_bb, collapse_bb, EDGE_TRUE_VALUE);
}
else
ep->flags = EDGE_TRUE_VALUE;
find_edge (cont_bb, fin_bb)->flags
= ep ? EDGE_FALSE_VALUE : EDGE_FALLTHRU;
}
set_immediate_dominator (CDI_DOMINATORS, body_bb,
recompute_dominator (CDI_DOMINATORS, body_bb));
if (!broken_loop)
set_immediate_dominator (CDI_DOMINATORS, fin_bb,
recompute_dominator (CDI_DOMINATORS, fin_bb));
if (!broken_loop && !gimple_omp_for_combined_p (fd->for_stmt))
{
struct loop *loop = alloc_loop ();
loop->header = body_bb;
if (collapse_bb == NULL)
loop->latch = cont_bb;
add_loop (loop, body_bb->loop_father);
}
}
static void
expand_oacc_for (struct omp_region *region, struct omp_for_data *fd)
{
tree v = fd->loop.v;
enum tree_code cond_code = fd->loop.cond_code;
enum tree_code plus_code = PLUS_EXPR;
tree chunk_size = integer_minus_one_node;
tree gwv = integer_zero_node;
tree iter_type = TREE_TYPE (v);
tree diff_type = iter_type;
tree plus_type = iter_type;
struct oacc_collapse *counts = NULL;
gcc_checking_assert (gimple_omp_for_kind (fd->for_stmt)
== GF_OMP_FOR_KIND_OACC_LOOP);
gcc_assert (!gimple_omp_for_combined_into_p (fd->for_stmt));
gcc_assert (cond_code == LT_EXPR || cond_code == GT_EXPR);
if (POINTER_TYPE_P (iter_type))
{
plus_code = POINTER_PLUS_EXPR;
plus_type = sizetype;
}
if (POINTER_TYPE_P (diff_type) || TYPE_UNSIGNED (diff_type))
diff_type = signed_type_for (diff_type);
if (TYPE_PRECISION (diff_type) < TYPE_PRECISION (integer_type_node))
diff_type = integer_type_node;
basic_block entry_bb = region->entry; 
basic_block exit_bb = region->exit; 
basic_block cont_bb = region->cont; 
basic_block bottom_bb = NULL;
gcc_assert (EDGE_COUNT (entry_bb->succs) == 2
&& BRANCH_EDGE (entry_bb)->dest == exit_bb);
if (cont_bb)
{
basic_block body_bb = FALLTHRU_EDGE (entry_bb)->dest;
basic_block bed = BRANCH_EDGE (cont_bb)->dest;
gcc_assert (FALLTHRU_EDGE (cont_bb)->dest == exit_bb);
gcc_assert (bed == body_bb || single_succ_edge (bed)->dest == body_bb);
}
else
gcc_assert (!gimple_in_ssa_p (cfun));
gcc_assert (EDGE_COUNT (exit_bb->preds) == 1 + (cont_bb != NULL));
tree chunk_no;
tree chunk_max = NULL_TREE;
tree bound, offset;
tree step = create_tmp_var (diff_type, ".step");
bool up = cond_code == LT_EXPR;
tree dir = build_int_cst (diff_type, up ? +1 : -1);
bool chunking = !gimple_in_ssa_p (cfun);
bool negating;
tree tile_size = NULL_TREE;
tree element_s = NULL_TREE;
tree e_bound = NULL_TREE, e_offset = NULL_TREE, e_step = NULL_TREE;
basic_block elem_body_bb = NULL;
basic_block elem_cont_bb = NULL;
tree offset_incr = NULL_TREE;
tree offset_init = NULL_TREE;
gimple_stmt_iterator gsi;
gassign *ass;
gcall *call;
gimple *stmt;
tree expr;
location_t loc;
edge split, be, fte;
split = split_block (entry_bb, last_stmt (entry_bb));
basic_block head_bb = split->dest;
entry_bb = split->src;
gsi = gsi_last_nondebug_bb (entry_bb);
gomp_for *for_stmt = as_a <gomp_for *> (gsi_stmt (gsi));
loc = gimple_location (for_stmt);
if (gimple_in_ssa_p (cfun))
{
offset_init = gimple_omp_for_index (for_stmt, 0);
gcc_assert (integer_zerop (fd->loop.n1));
gwv = build_int_cst (integer_type_node, GOMP_DIM_MASK (GOMP_DIM_GANG));
}
if (fd->collapse > 1 || fd->tiling)
{
gcc_assert (!gimple_in_ssa_p (cfun) && up);
counts = XALLOCAVEC (struct oacc_collapse, fd->collapse);
tree total = expand_oacc_collapse_init (fd, &gsi, counts,
TREE_TYPE (fd->loop.n2), loc);
if (SSA_VAR_P (fd->loop.n2))
{
total = force_gimple_operand_gsi (&gsi, total, false, NULL_TREE,
true, GSI_SAME_STMT);
ass = gimple_build_assign (fd->loop.n2, total);
gsi_insert_before (&gsi, ass, GSI_SAME_STMT);
}
}
tree b = fd->loop.n1;
tree e = fd->loop.n2;
tree s = fd->loop.step;
b = force_gimple_operand_gsi (&gsi, b, true, NULL_TREE, true, GSI_SAME_STMT);
e = force_gimple_operand_gsi (&gsi, e, true, NULL_TREE, true, GSI_SAME_STMT);
negating = !up && TYPE_UNSIGNED (TREE_TYPE (s));
if (negating)
s = fold_build1 (NEGATE_EXPR, TREE_TYPE (s), s);
s = fold_convert (diff_type, s);
if (negating)
s = fold_build1 (NEGATE_EXPR, diff_type, s);
s = force_gimple_operand_gsi (&gsi, s, true, NULL_TREE, true, GSI_SAME_STMT);
if (!chunking)
chunk_size = integer_zero_node;
expr = fold_convert (diff_type, chunk_size);
chunk_size = force_gimple_operand_gsi (&gsi, expr, true,
NULL_TREE, true, GSI_SAME_STMT);
if (fd->tiling)
{
tile_size = create_tmp_var (diff_type, ".tile_size");
expr = build_int_cst (diff_type, 1);
for (int ix = 0; ix < fd->collapse; ix++)
expr = fold_build2 (MULT_EXPR, diff_type, counts[ix].tile, expr);
expr = force_gimple_operand_gsi (&gsi, expr, true,
NULL_TREE, true, GSI_SAME_STMT);
ass = gimple_build_assign (tile_size, expr);
gsi_insert_before (&gsi, ass, GSI_SAME_STMT);
element_s = create_tmp_var (diff_type, ".element_s");
ass = gimple_build_assign (element_s, s);
gsi_insert_before (&gsi, ass, GSI_SAME_STMT);
expr = fold_build2 (MULT_EXPR, diff_type, s, tile_size);
s = force_gimple_operand_gsi (&gsi, expr, true,
NULL_TREE, true, GSI_SAME_STMT);
}
negating = !up && TYPE_UNSIGNED (iter_type);
expr = fold_build2 (MINUS_EXPR, plus_type,
fold_convert (plus_type, negating ? b : e),
fold_convert (plus_type, negating ? e : b));
expr = fold_convert (diff_type, expr);
if (negating)
expr = fold_build1 (NEGATE_EXPR, diff_type, expr);
tree range = force_gimple_operand_gsi (&gsi, expr, true,
NULL_TREE, true, GSI_SAME_STMT);
chunk_no = build_int_cst (diff_type, 0);
if (chunking)
{
gcc_assert (!gimple_in_ssa_p (cfun));
expr = chunk_no;
chunk_max = create_tmp_var (diff_type, ".chunk_max");
chunk_no = create_tmp_var (diff_type, ".chunk_no");
ass = gimple_build_assign (chunk_no, expr);
gsi_insert_before (&gsi, ass, GSI_SAME_STMT);
call = gimple_build_call_internal (IFN_GOACC_LOOP, 6,
build_int_cst (integer_type_node,
IFN_GOACC_LOOP_CHUNKS),
dir, range, s, chunk_size, gwv);
gimple_call_set_lhs (call, chunk_max);
gimple_set_location (call, loc);
gsi_insert_before (&gsi, call, GSI_SAME_STMT);
}
else
chunk_size = chunk_no;
call = gimple_build_call_internal (IFN_GOACC_LOOP, 6,
build_int_cst (integer_type_node,
IFN_GOACC_LOOP_STEP),
dir, range, s, chunk_size, gwv);
gimple_call_set_lhs (call, step);
gimple_set_location (call, loc);
gsi_insert_before (&gsi, call, GSI_SAME_STMT);
gsi_remove (&gsi, true);
be = BRANCH_EDGE (head_bb);
fte = FALLTHRU_EDGE (head_bb);
be->flags |= EDGE_FALSE_VALUE;
fte->flags ^= EDGE_FALLTHRU | EDGE_TRUE_VALUE;
basic_block body_bb = fte->dest;
if (gimple_in_ssa_p (cfun))
{
gsi = gsi_last_nondebug_bb (cont_bb);
gomp_continue *cont_stmt = as_a <gomp_continue *> (gsi_stmt (gsi));
offset = gimple_omp_continue_control_use (cont_stmt);
offset_incr = gimple_omp_continue_control_def (cont_stmt);
}
else
{
offset = create_tmp_var (diff_type, ".offset");
offset_init = offset_incr = offset;
}
bound = create_tmp_var (TREE_TYPE (offset), ".bound");
gsi = gsi_start_bb (head_bb);
call = gimple_build_call_internal (IFN_GOACC_LOOP, 7,
build_int_cst (integer_type_node,
IFN_GOACC_LOOP_OFFSET),
dir, range, s,
chunk_size, gwv, chunk_no);
gimple_call_set_lhs (call, offset_init);
gimple_set_location (call, loc);
gsi_insert_after (&gsi, call, GSI_CONTINUE_LINKING);
call = gimple_build_call_internal (IFN_GOACC_LOOP, 7,
build_int_cst (integer_type_node,
IFN_GOACC_LOOP_BOUND),
dir, range, s,
chunk_size, gwv, offset_init);
gimple_call_set_lhs (call, bound);
gimple_set_location (call, loc);
gsi_insert_after (&gsi, call, GSI_CONTINUE_LINKING);
expr = build2 (cond_code, boolean_type_node, offset_init, bound);
gsi_insert_after (&gsi, gimple_build_cond_empty (expr),
GSI_CONTINUE_LINKING);
if (!gimple_in_ssa_p (cfun))
{
gsi = gsi_start_bb (body_bb);
expr = build2 (plus_code, iter_type, b,
fold_convert (plus_type, offset));
expr = force_gimple_operand_gsi (&gsi, expr, false, NULL_TREE,
true, GSI_SAME_STMT);
ass = gimple_build_assign (v, expr);
gsi_insert_before (&gsi, ass, GSI_SAME_STMT);
if (fd->collapse > 1 || fd->tiling)
expand_oacc_collapse_vars (fd, false, &gsi, counts, v);
if (fd->tiling)
{
tree e_range = create_tmp_var (diff_type, ".e_range");
expr = build2 (MIN_EXPR, diff_type,
build2 (MINUS_EXPR, diff_type, bound, offset),
build2 (MULT_EXPR, diff_type, tile_size,
element_s));
expr = force_gimple_operand_gsi (&gsi, expr, false, NULL_TREE,
true, GSI_SAME_STMT);
ass = gimple_build_assign (e_range, expr);
gsi_insert_before (&gsi, ass, GSI_SAME_STMT);
e_bound = create_tmp_var (diff_type, ".e_bound");
e_offset = create_tmp_var (diff_type, ".e_offset");
e_step = create_tmp_var (diff_type, ".e_step");
tree t, e_gwv = integer_minus_one_node;
tree chunk = build_int_cst (diff_type, 0); 
t = build_int_cst (integer_type_node, IFN_GOACC_LOOP_OFFSET);
call = gimple_build_call_internal (IFN_GOACC_LOOP, 7, t, dir, e_range,
element_s, chunk, e_gwv, chunk);
gimple_call_set_lhs (call, e_offset);
gimple_set_location (call, loc);
gsi_insert_before (&gsi, call, GSI_SAME_STMT);
t = build_int_cst (integer_type_node, IFN_GOACC_LOOP_BOUND);
call = gimple_build_call_internal (IFN_GOACC_LOOP, 7, t, dir, e_range,
element_s, chunk, e_gwv, e_offset);
gimple_call_set_lhs (call, e_bound);
gimple_set_location (call, loc);
gsi_insert_before (&gsi, call, GSI_SAME_STMT);
t = build_int_cst (integer_type_node, IFN_GOACC_LOOP_STEP);
call = gimple_build_call_internal (IFN_GOACC_LOOP, 6, t, dir, e_range,
element_s, chunk, e_gwv);
gimple_call_set_lhs (call, e_step);
gimple_set_location (call, loc);
gsi_insert_before (&gsi, call, GSI_SAME_STMT);
expr = build2 (cond_code, boolean_type_node, e_offset, e_bound);
stmt = gimple_build_cond_empty (expr);
gsi_insert_before (&gsi, stmt, GSI_SAME_STMT);
split = split_block (body_bb, stmt);
elem_body_bb = split->dest;
if (cont_bb == body_bb)
cont_bb = elem_body_bb;
body_bb = split->src;
split->flags ^= EDGE_FALLTHRU | EDGE_TRUE_VALUE;
if (cont_bb == NULL)
{
edge e = make_edge (body_bb, exit_bb, EDGE_FALSE_VALUE);
e->probability = profile_probability::even ();
split->probability = profile_probability::even ();
}
gsi = gsi_start_bb (elem_body_bb);
expand_oacc_collapse_vars (fd, true, &gsi, counts, e_offset);
}
}
if (cont_bb)
{
gsi = gsi_last_nondebug_bb (cont_bb);
gomp_continue *cont_stmt = as_a <gomp_continue *> (gsi_stmt (gsi));
loc = gimple_location (cont_stmt);
if (fd->tiling)
{
expr = build2 (PLUS_EXPR, diff_type, e_offset, e_step);
expr = force_gimple_operand_gsi (&gsi, expr, false, NULL_TREE,
true, GSI_SAME_STMT);
ass = gimple_build_assign (e_offset, expr);
gsi_insert_before (&gsi, ass, GSI_SAME_STMT);
expr = build2 (cond_code, boolean_type_node, e_offset, e_bound);
stmt = gimple_build_cond_empty (expr);
gsi_insert_before (&gsi, stmt, GSI_SAME_STMT);
split = split_block (cont_bb, stmt);
elem_cont_bb = split->src;
cont_bb = split->dest;
split->flags ^= EDGE_FALLTHRU | EDGE_FALSE_VALUE;
split->probability = profile_probability::unlikely ().guessed ();
edge latch_edge
= make_edge (elem_cont_bb, elem_body_bb, EDGE_TRUE_VALUE);
latch_edge->probability = profile_probability::likely ().guessed ();
edge skip_edge = make_edge (body_bb, cont_bb, EDGE_FALSE_VALUE);
skip_edge->probability = profile_probability::unlikely ().guessed ();
edge loop_entry_edge = EDGE_SUCC (body_bb, 1 - skip_edge->dest_idx);
loop_entry_edge->probability
= profile_probability::likely ().guessed ();
gsi = gsi_for_stmt (cont_stmt);
}
if (gimple_in_ssa_p (cfun))
expr = build2 (plus_code, iter_type, offset,
fold_convert (plus_type, step));
else
expr = build2 (PLUS_EXPR, diff_type, offset, step);
expr = force_gimple_operand_gsi (&gsi, expr, false, NULL_TREE,
true, GSI_SAME_STMT);
ass = gimple_build_assign (offset_incr, expr);
gsi_insert_before (&gsi, ass, GSI_SAME_STMT);
expr = build2 (cond_code, boolean_type_node, offset_incr, bound);
gsi_insert_before (&gsi, gimple_build_cond_empty (expr), GSI_SAME_STMT);
gsi_remove (&gsi, true);
be = BRANCH_EDGE (cont_bb);
fte = FALLTHRU_EDGE (cont_bb);
be->flags |= EDGE_TRUE_VALUE;
fte->flags ^= EDGE_FALLTHRU | EDGE_FALSE_VALUE;
if (chunking)
{
gsi = gsi_start_bb (exit_bb);
stmt = gimple_build_nop ();
gsi_insert_before (&gsi, stmt, GSI_SAME_STMT);
split = split_block (exit_bb, stmt);
bottom_bb = split->src;
exit_bb = split->dest;
gsi = gsi_last_bb (bottom_bb);
expr = build2 (PLUS_EXPR, diff_type, chunk_no,
build_int_cst (diff_type, 1));
ass = gimple_build_assign (chunk_no, expr);
gsi_insert_after (&gsi, ass, GSI_CONTINUE_LINKING);
expr = build2 (LT_EXPR, boolean_type_node, chunk_no, chunk_max);
gsi_insert_after (&gsi, gimple_build_cond_empty (expr),
GSI_CONTINUE_LINKING);
split->flags ^= EDGE_FALLTHRU | EDGE_FALSE_VALUE;
split->probability = profile_probability::unlikely ().guessed ();
edge latch_edge = make_edge (bottom_bb, head_bb, EDGE_TRUE_VALUE);
latch_edge->probability = profile_probability::likely ().guessed ();
}
}
gsi = gsi_last_nondebug_bb (exit_bb);
gcc_assert (gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_RETURN);
loc = gimple_location (gsi_stmt (gsi));
if (!gimple_in_ssa_p (cfun))
{
expr = fold_build2 (MINUS_EXPR, diff_type, range, dir);
expr = fold_build2 (PLUS_EXPR, diff_type, expr, s);
expr = fold_build2 (TRUNC_DIV_EXPR, diff_type, expr, s);
expr = fold_build2 (MULT_EXPR, diff_type, expr, s);
expr = build2 (plus_code, iter_type, b, fold_convert (plus_type, expr));
expr = force_gimple_operand_gsi (&gsi, expr, false, NULL_TREE,
true, GSI_SAME_STMT);
ass = gimple_build_assign (v, expr);
gsi_insert_before (&gsi, ass, GSI_SAME_STMT);
}
gsi_remove (&gsi, true);
if (cont_bb)
{
struct loop *parent = entry_bb->loop_father;
struct loop *body = body_bb->loop_father;
if (chunking)
{
struct loop *chunk_loop = alloc_loop ();
chunk_loop->header = head_bb;
chunk_loop->latch = bottom_bb;
add_loop (chunk_loop, parent);
parent = chunk_loop;
}
else if (parent != body)
{
gcc_assert (body->header == body_bb);
gcc_assert (body->latch == cont_bb
|| single_pred (body->latch) == cont_bb);
parent = NULL;
}
if (parent)
{
struct loop *body_loop = alloc_loop ();
body_loop->header = body_bb;
body_loop->latch = cont_bb;
add_loop (body_loop, parent);
if (fd->tiling)
{
struct loop *inner_loop = alloc_loop ();
inner_loop->header = elem_body_bb;
inner_loop->latch = elem_cont_bb;
add_loop (inner_loop, body_loop);
}
}
}
}
static void
expand_omp_for (struct omp_region *region, gimple *inner_stmt)
{
struct omp_for_data fd;
struct omp_for_data_loop *loops;
loops
= (struct omp_for_data_loop *)
alloca (gimple_omp_for_collapse (last_stmt (region->entry))
* sizeof (struct omp_for_data_loop));
omp_extract_for_data (as_a <gomp_for *> (last_stmt (region->entry)),
&fd, loops);
region->sched_kind = fd.sched_kind;
region->sched_modifiers = fd.sched_modifiers;
gcc_assert (EDGE_COUNT (region->entry->succs) == 2);
BRANCH_EDGE (region->entry)->flags &= ~EDGE_ABNORMAL;
FALLTHRU_EDGE (region->entry)->flags &= ~EDGE_ABNORMAL;
if (region->cont)
{
gcc_assert (EDGE_COUNT (region->cont->succs) == 2);
BRANCH_EDGE (region->cont)->flags &= ~EDGE_ABNORMAL;
FALLTHRU_EDGE (region->cont)->flags &= ~EDGE_ABNORMAL;
}
else
loops_state_set (LOOPS_NEED_FIXUP);
if (gimple_omp_for_kind (fd.for_stmt) & GF_OMP_FOR_SIMD)
expand_omp_simd (region, &fd);
else if (gimple_omp_for_kind (fd.for_stmt) == GF_OMP_FOR_KIND_OACC_LOOP)
{
gcc_assert (!inner_stmt);
expand_oacc_for (region, &fd);
}
else if (gimple_omp_for_kind (fd.for_stmt) == GF_OMP_FOR_KIND_TASKLOOP)
{
if (gimple_omp_for_combined_into_p (fd.for_stmt))
expand_omp_taskloop_for_inner (region, &fd, inner_stmt);
else
expand_omp_taskloop_for_outer (region, &fd, inner_stmt);
}
else if (fd.sched_kind == OMP_CLAUSE_SCHEDULE_STATIC
&& !fd.have_ordered)
{
if (fd.chunk_size == NULL)
expand_omp_for_static_nochunk (region, &fd, inner_stmt);
else
expand_omp_for_static_chunk (region, &fd, inner_stmt);
}
else
{
int fn_index, start_ix, next_ix;
gcc_assert (gimple_omp_for_kind (fd.for_stmt)
== GF_OMP_FOR_KIND_FOR);
if (fd.chunk_size == NULL
&& fd.sched_kind == OMP_CLAUSE_SCHEDULE_STATIC)
fd.chunk_size = integer_zero_node;
gcc_assert (fd.sched_kind != OMP_CLAUSE_SCHEDULE_AUTO);
switch (fd.sched_kind)
{
case OMP_CLAUSE_SCHEDULE_RUNTIME:
fn_index = 3;
break;
case OMP_CLAUSE_SCHEDULE_DYNAMIC:
case OMP_CLAUSE_SCHEDULE_GUIDED:
if ((fd.sched_modifiers & OMP_CLAUSE_SCHEDULE_NONMONOTONIC)
&& !fd.ordered
&& !fd.have_ordered)
{
fn_index = 3 + fd.sched_kind;
break;
}
default:
fn_index = fd.sched_kind;
break;
}
if (!fd.ordered)
fn_index += fd.have_ordered * 6;
if (fd.ordered)
start_ix = ((int)BUILT_IN_GOMP_LOOP_DOACROSS_STATIC_START) + fn_index;
else
start_ix = ((int)BUILT_IN_GOMP_LOOP_STATIC_START) + fn_index;
next_ix = ((int)BUILT_IN_GOMP_LOOP_STATIC_NEXT) + fn_index;
if (fd.iter_type == long_long_unsigned_type_node)
{
start_ix += ((int)BUILT_IN_GOMP_LOOP_ULL_STATIC_START
- (int)BUILT_IN_GOMP_LOOP_STATIC_START);
next_ix += ((int)BUILT_IN_GOMP_LOOP_ULL_STATIC_NEXT
- (int)BUILT_IN_GOMP_LOOP_STATIC_NEXT);
}
expand_omp_for_generic (region, &fd, (enum built_in_function) start_ix,
(enum built_in_function) next_ix, inner_stmt);
}
if (gimple_in_ssa_p (cfun))
update_ssa (TODO_update_ssa_only_virtuals);
}
static void
expand_omp_sections (struct omp_region *region)
{
tree t, u, vin = NULL, vmain, vnext, l2;
unsigned len;
basic_block entry_bb, l0_bb, l1_bb, l2_bb, default_bb;
gimple_stmt_iterator si, switch_si;
gomp_sections *sections_stmt;
gimple *stmt;
gomp_continue *cont;
edge_iterator ei;
edge e;
struct omp_region *inner;
unsigned i, casei;
bool exit_reachable = region->cont != NULL;
gcc_assert (region->exit != NULL);
entry_bb = region->entry;
l0_bb = single_succ (entry_bb);
l1_bb = region->cont;
l2_bb = region->exit;
if (single_pred_p (l2_bb) && single_pred (l2_bb) == l0_bb)
l2 = gimple_block_label (l2_bb);
else
{
len = EDGE_COUNT (l0_bb->succs);
gcc_assert (len > 0);
e = EDGE_SUCC (l0_bb, len - 1);
si = gsi_last_nondebug_bb (e->dest);
l2 = NULL_TREE;
if (gsi_end_p (si)
|| gimple_code (gsi_stmt (si)) != GIMPLE_OMP_SECTION)
l2 = gimple_block_label (e->dest);
else
FOR_EACH_EDGE (e, ei, l0_bb->succs)
{
si = gsi_last_nondebug_bb (e->dest);
if (gsi_end_p (si)
|| gimple_code (gsi_stmt (si)) != GIMPLE_OMP_SECTION)
{
l2 = gimple_block_label (e->dest);
break;
}
}
}
if (exit_reachable)
default_bb = create_empty_bb (l1_bb->prev_bb);
else
default_bb = create_empty_bb (l0_bb);
len = EDGE_COUNT (l0_bb->succs);
auto_vec<tree> label_vec (len);
si = gsi_last_nondebug_bb (entry_bb);
sections_stmt = as_a <gomp_sections *> (gsi_stmt (si));
gcc_assert (gimple_code (sections_stmt) == GIMPLE_OMP_SECTIONS);
vin = gimple_omp_sections_control (sections_stmt);
if (!is_combined_parallel (region))
{
t = build_int_cst (unsigned_type_node, len - 1);
u = builtin_decl_explicit (BUILT_IN_GOMP_SECTIONS_START);
stmt = gimple_build_call (u, 1, t);
}
else
{
u = builtin_decl_explicit (BUILT_IN_GOMP_SECTIONS_NEXT);
stmt = gimple_build_call (u, 0);
}
gimple_call_set_lhs (stmt, vin);
gsi_insert_after (&si, stmt, GSI_SAME_STMT);
gsi_remove (&si, true);
switch_si = gsi_last_nondebug_bb (l0_bb);
gcc_assert (gimple_code (gsi_stmt (switch_si)) == GIMPLE_OMP_SECTIONS_SWITCH);
if (exit_reachable)
{
cont = as_a <gomp_continue *> (last_stmt (l1_bb));
gcc_assert (gimple_code (cont) == GIMPLE_OMP_CONTINUE);
vmain = gimple_omp_continue_control_use (cont);
vnext = gimple_omp_continue_control_def (cont);
}
else
{
vmain = vin;
vnext = NULL_TREE;
}
t = build_case_label (build_int_cst (unsigned_type_node, 0), NULL, l2);
label_vec.quick_push (t);
i = 1;
for (inner = region->inner, casei = 1;
inner;
inner = inner->next, i++, casei++)
{
basic_block s_entry_bb, s_exit_bb;
if (inner->type == GIMPLE_OMP_ATOMIC_LOAD)
{
--i;
--casei;
continue;
}
s_entry_bb = inner->entry;
s_exit_bb = inner->exit;
t = gimple_block_label (s_entry_bb);
u = build_int_cst (unsigned_type_node, casei);
u = build_case_label (u, NULL, t);
label_vec.quick_push (u);
si = gsi_last_nondebug_bb (s_entry_bb);
gcc_assert (gimple_code (gsi_stmt (si)) == GIMPLE_OMP_SECTION);
gcc_assert (i < len || gimple_omp_section_last_p (gsi_stmt (si)));
gsi_remove (&si, true);
single_succ_edge (s_entry_bb)->flags = EDGE_FALLTHRU;
if (s_exit_bb == NULL)
continue;
si = gsi_last_nondebug_bb (s_exit_bb);
gcc_assert (gimple_code (gsi_stmt (si)) == GIMPLE_OMP_RETURN);
gsi_remove (&si, true);
single_succ_edge (s_exit_bb)->flags = EDGE_FALLTHRU;
}
t = gimple_block_label (default_bb);
u = build_case_label (NULL, NULL, t);
make_edge (l0_bb, default_bb, 0);
add_bb_to_loop (default_bb, current_loops->tree_root);
stmt = gimple_build_switch (vmain, u, label_vec);
gsi_insert_after (&switch_si, stmt, GSI_SAME_STMT);
gsi_remove (&switch_si, true);
si = gsi_start_bb (default_bb);
stmt = gimple_build_call (builtin_decl_explicit (BUILT_IN_TRAP), 0);
gsi_insert_after (&si, stmt, GSI_CONTINUE_LINKING);
if (exit_reachable)
{
tree bfn_decl;
si = gsi_last_nondebug_bb (l1_bb);
gcc_assert (gimple_code (gsi_stmt (si)) == GIMPLE_OMP_CONTINUE);
bfn_decl = builtin_decl_explicit (BUILT_IN_GOMP_SECTIONS_NEXT);
stmt = gimple_build_call (bfn_decl, 0);
gimple_call_set_lhs (stmt, vnext);
gsi_insert_after (&si, stmt, GSI_SAME_STMT);
gsi_remove (&si, true);
single_succ_edge (l1_bb)->flags = EDGE_FALLTHRU;
}
si = gsi_last_nondebug_bb (l2_bb);
if (gimple_omp_return_nowait_p (gsi_stmt (si)))
t = builtin_decl_explicit (BUILT_IN_GOMP_SECTIONS_END_NOWAIT);
else if (gimple_omp_return_lhs (gsi_stmt (si)))
t = builtin_decl_explicit (BUILT_IN_GOMP_SECTIONS_END_CANCEL);
else
t = builtin_decl_explicit (BUILT_IN_GOMP_SECTIONS_END);
stmt = gimple_build_call (t, 0);
if (gimple_omp_return_lhs (gsi_stmt (si)))
gimple_call_set_lhs (stmt, gimple_omp_return_lhs (gsi_stmt (si)));
gsi_insert_after (&si, stmt, GSI_SAME_STMT);
gsi_remove (&si, true);
set_immediate_dominator (CDI_DOMINATORS, default_bb, l0_bb);
}
static void
expand_omp_single (struct omp_region *region)
{
basic_block entry_bb, exit_bb;
gimple_stmt_iterator si;
entry_bb = region->entry;
exit_bb = region->exit;
si = gsi_last_nondebug_bb (entry_bb);
gcc_assert (gimple_code (gsi_stmt (si)) == GIMPLE_OMP_SINGLE);
gsi_remove (&si, true);
single_succ_edge (entry_bb)->flags = EDGE_FALLTHRU;
si = gsi_last_nondebug_bb (exit_bb);
if (!gimple_omp_return_nowait_p (gsi_stmt (si)))
{
tree t = gimple_omp_return_lhs (gsi_stmt (si));
gsi_insert_after (&si, omp_build_barrier (t), GSI_SAME_STMT);
}
gsi_remove (&si, true);
single_succ_edge (exit_bb)->flags = EDGE_FALLTHRU;
}
static void
expand_omp_synch (struct omp_region *region)
{
basic_block entry_bb, exit_bb;
gimple_stmt_iterator si;
entry_bb = region->entry;
exit_bb = region->exit;
si = gsi_last_nondebug_bb (entry_bb);
gcc_assert (gimple_code (gsi_stmt (si)) == GIMPLE_OMP_SINGLE
|| gimple_code (gsi_stmt (si)) == GIMPLE_OMP_MASTER
|| gimple_code (gsi_stmt (si)) == GIMPLE_OMP_TASKGROUP
|| gimple_code (gsi_stmt (si)) == GIMPLE_OMP_ORDERED
|| gimple_code (gsi_stmt (si)) == GIMPLE_OMP_CRITICAL
|| gimple_code (gsi_stmt (si)) == GIMPLE_OMP_TEAMS);
gsi_remove (&si, true);
single_succ_edge (entry_bb)->flags = EDGE_FALLTHRU;
if (exit_bb)
{
si = gsi_last_nondebug_bb (exit_bb);
gcc_assert (gimple_code (gsi_stmt (si)) == GIMPLE_OMP_RETURN);
gsi_remove (&si, true);
single_succ_edge (exit_bb)->flags = EDGE_FALLTHRU;
}
}
static bool
expand_omp_atomic_load (basic_block load_bb, tree addr,
tree loaded_val, int index)
{
enum built_in_function tmpbase;
gimple_stmt_iterator gsi;
basic_block store_bb;
location_t loc;
gimple *stmt;
tree decl, call, type, itype;
gsi = gsi_last_nondebug_bb (load_bb);
stmt = gsi_stmt (gsi);
gcc_assert (gimple_code (stmt) == GIMPLE_OMP_ATOMIC_LOAD);
loc = gimple_location (stmt);
tmpbase = (enum built_in_function) (BUILT_IN_ATOMIC_LOAD_N + index + 1);
decl = builtin_decl_explicit (tmpbase);
if (decl == NULL_TREE)
return false;
type = TREE_TYPE (loaded_val);
itype = TREE_TYPE (TREE_TYPE (decl));
call = build_call_expr_loc (loc, decl, 2, addr,
build_int_cst (NULL,
gimple_omp_atomic_seq_cst_p (stmt)
? MEMMODEL_SEQ_CST
: MEMMODEL_RELAXED));
if (!useless_type_conversion_p (type, itype))
call = fold_build1_loc (loc, VIEW_CONVERT_EXPR, type, call);
call = build2_loc (loc, MODIFY_EXPR, void_type_node, loaded_val, call);
force_gimple_operand_gsi (&gsi, call, true, NULL_TREE, true, GSI_SAME_STMT);
gsi_remove (&gsi, true);
store_bb = single_succ (load_bb);
gsi = gsi_last_nondebug_bb (store_bb);
gcc_assert (gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_ATOMIC_STORE);
gsi_remove (&gsi, true);
if (gimple_in_ssa_p (cfun))
update_ssa (TODO_update_ssa_no_phi);
return true;
}
static bool
expand_omp_atomic_store (basic_block load_bb, tree addr,
tree loaded_val, tree stored_val, int index)
{
enum built_in_function tmpbase;
gimple_stmt_iterator gsi;
basic_block store_bb = single_succ (load_bb);
location_t loc;
gimple *stmt;
tree decl, call, type, itype;
machine_mode imode;
bool exchange;
gsi = gsi_last_nondebug_bb (load_bb);
stmt = gsi_stmt (gsi);
gcc_assert (gimple_code (stmt) == GIMPLE_OMP_ATOMIC_LOAD);
exchange = gimple_omp_atomic_need_value_p (stmt);
gsi = gsi_last_nondebug_bb (store_bb);
stmt = gsi_stmt (gsi);
gcc_assert (gimple_code (stmt) == GIMPLE_OMP_ATOMIC_STORE);
loc = gimple_location (stmt);
tmpbase = (exchange ? BUILT_IN_ATOMIC_EXCHANGE_N : BUILT_IN_ATOMIC_STORE_N);
tmpbase = (enum built_in_function) ((int) tmpbase + index + 1);
decl = builtin_decl_explicit (tmpbase);
if (decl == NULL_TREE)
return false;
type = TREE_TYPE (stored_val);
itype = TREE_TYPE (decl);
itype = TYPE_ARG_TYPES (itype);
itype = TREE_CHAIN (itype);
itype = TREE_VALUE (itype);
imode = TYPE_MODE (itype);
if (exchange && !can_atomic_exchange_p (imode, true))
return false;
if (!useless_type_conversion_p (itype, type))
stored_val = fold_build1_loc (loc, VIEW_CONVERT_EXPR, itype, stored_val);
call = build_call_expr_loc (loc, decl, 3, addr, stored_val,
build_int_cst (NULL,
gimple_omp_atomic_seq_cst_p (stmt)
? MEMMODEL_SEQ_CST
: MEMMODEL_RELAXED));
if (exchange)
{
if (!useless_type_conversion_p (type, itype))
call = build1_loc (loc, VIEW_CONVERT_EXPR, type, call);
call = build2_loc (loc, MODIFY_EXPR, void_type_node, loaded_val, call);
}
force_gimple_operand_gsi (&gsi, call, true, NULL_TREE, true, GSI_SAME_STMT);
gsi_remove (&gsi, true);
gsi = gsi_last_nondebug_bb (load_bb);
gsi_remove (&gsi, true);
if (gimple_in_ssa_p (cfun))
update_ssa (TODO_update_ssa_no_phi);
return true;
}
static bool
expand_omp_atomic_fetch_op (basic_block load_bb,
tree addr, tree loaded_val,
tree stored_val, int index)
{
enum built_in_function oldbase, newbase, tmpbase;
tree decl, itype, call;
tree lhs, rhs;
basic_block store_bb = single_succ (load_bb);
gimple_stmt_iterator gsi;
gimple *stmt;
location_t loc;
enum tree_code code;
bool need_old, need_new;
machine_mode imode;
bool seq_cst;
gsi = gsi_after_labels (store_bb);
stmt = gsi_stmt (gsi);
if (is_gimple_debug (stmt))
{
gsi_next_nondebug (&gsi);
if (gsi_end_p (gsi))
return false;
stmt = gsi_stmt (gsi);
}
loc = gimple_location (stmt);
if (!is_gimple_assign (stmt))
return false;
gsi_next_nondebug (&gsi);
if (gimple_code (gsi_stmt (gsi)) != GIMPLE_OMP_ATOMIC_STORE)
return false;
need_new = gimple_omp_atomic_need_value_p (gsi_stmt (gsi));
need_old = gimple_omp_atomic_need_value_p (last_stmt (load_bb));
seq_cst = gimple_omp_atomic_seq_cst_p (last_stmt (load_bb));
gcc_checking_assert (!need_old || !need_new);
if (!operand_equal_p (gimple_assign_lhs (stmt), stored_val, 0))
return false;
code = gimple_assign_rhs_code (stmt);
switch (code)
{
case PLUS_EXPR:
case POINTER_PLUS_EXPR:
oldbase = BUILT_IN_ATOMIC_FETCH_ADD_N;
newbase = BUILT_IN_ATOMIC_ADD_FETCH_N;
break;
case MINUS_EXPR:
oldbase = BUILT_IN_ATOMIC_FETCH_SUB_N;
newbase = BUILT_IN_ATOMIC_SUB_FETCH_N;
break;
case BIT_AND_EXPR:
oldbase = BUILT_IN_ATOMIC_FETCH_AND_N;
newbase = BUILT_IN_ATOMIC_AND_FETCH_N;
break;
case BIT_IOR_EXPR:
oldbase = BUILT_IN_ATOMIC_FETCH_OR_N;
newbase = BUILT_IN_ATOMIC_OR_FETCH_N;
break;
case BIT_XOR_EXPR:
oldbase = BUILT_IN_ATOMIC_FETCH_XOR_N;
newbase = BUILT_IN_ATOMIC_XOR_FETCH_N;
break;
default:
return false;
}
if (operand_equal_p (gimple_assign_rhs1 (stmt), loaded_val, 0))
rhs = gimple_assign_rhs2 (stmt);
else if (commutative_tree_code (gimple_assign_rhs_code (stmt))
&& operand_equal_p (gimple_assign_rhs2 (stmt), loaded_val, 0))
rhs = gimple_assign_rhs1 (stmt);
else
return false;
tmpbase = ((enum built_in_function)
((need_new ? newbase : oldbase) + index + 1));
decl = builtin_decl_explicit (tmpbase);
if (decl == NULL_TREE)
return false;
itype = TREE_TYPE (TREE_TYPE (decl));
imode = TYPE_MODE (itype);
if (!can_compare_and_swap_p (imode, true) || !can_atomic_load_p (imode))
return false;
gsi = gsi_last_nondebug_bb (load_bb);
gcc_assert (gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_ATOMIC_LOAD);
call = build_call_expr_loc (loc, decl, 3, addr,
fold_convert_loc (loc, itype, rhs),
build_int_cst (NULL,
seq_cst ? MEMMODEL_SEQ_CST
: MEMMODEL_RELAXED));
if (need_old || need_new)
{
lhs = need_old ? loaded_val : stored_val;
call = fold_convert_loc (loc, TREE_TYPE (lhs), call);
call = build2_loc (loc, MODIFY_EXPR, void_type_node, lhs, call);
}
else
call = fold_convert_loc (loc, void_type_node, call);
force_gimple_operand_gsi (&gsi, call, true, NULL_TREE, true, GSI_SAME_STMT);
gsi_remove (&gsi, true);
gsi = gsi_last_nondebug_bb (store_bb);
gcc_assert (gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_ATOMIC_STORE);
gsi_remove (&gsi, true);
gsi = gsi_last_nondebug_bb (store_bb);
stmt = gsi_stmt (gsi);
gsi_remove (&gsi, true);
if (gimple_in_ssa_p (cfun))
{
release_defs (stmt);
update_ssa (TODO_update_ssa_no_phi);
}
return true;
}
static bool
expand_omp_atomic_pipeline (basic_block load_bb, basic_block store_bb,
tree addr, tree loaded_val, tree stored_val,
int index)
{
tree loadedi, storedi, initial, new_storedi, old_vali;
tree type, itype, cmpxchg, iaddr, atype;
gimple_stmt_iterator si;
basic_block loop_header = single_succ (load_bb);
gimple *phi, *stmt;
edge e;
enum built_in_function fncode;
fncode = (enum built_in_function)((int)BUILT_IN_SYNC_VAL_COMPARE_AND_SWAP_N
+ index + 1);
cmpxchg = builtin_decl_explicit (fncode);
if (cmpxchg == NULL_TREE)
return false;
type = TYPE_MAIN_VARIANT (TREE_TYPE (loaded_val));
atype = type;
itype = TREE_TYPE (TREE_TYPE (cmpxchg));
if (!can_compare_and_swap_p (TYPE_MODE (itype), true)
|| !can_atomic_load_p (TYPE_MODE (itype)))
return false;
si = gsi_last_nondebug_bb (load_bb);
gcc_assert (gimple_code (gsi_stmt (si)) == GIMPLE_OMP_ATOMIC_LOAD);
if (!INTEGRAL_TYPE_P (type) && !POINTER_TYPE_P (type))
{
tree iaddr_val;
iaddr = create_tmp_reg (build_pointer_type_for_mode (itype, ptr_mode,
true));
atype = itype;
iaddr_val
= force_gimple_operand_gsi (&si,
fold_convert (TREE_TYPE (iaddr), addr),
false, NULL_TREE, true, GSI_SAME_STMT);
stmt = gimple_build_assign (iaddr, iaddr_val);
gsi_insert_before (&si, stmt, GSI_SAME_STMT);
loadedi = create_tmp_var (itype);
if (gimple_in_ssa_p (cfun))
loadedi = make_ssa_name (loadedi);
}
else
{
iaddr = addr;
loadedi = loaded_val;
}
fncode = (enum built_in_function) (BUILT_IN_ATOMIC_LOAD_N + index + 1);
tree loaddecl = builtin_decl_explicit (fncode);
if (loaddecl)
initial
= fold_convert (atype,
build_call_expr (loaddecl, 2, iaddr,
build_int_cst (NULL_TREE,
MEMMODEL_RELAXED)));
else
{
tree off
= build_int_cst (build_pointer_type_for_mode (atype, ptr_mode,
true), 0);
initial = build2 (MEM_REF, atype, iaddr, off);
}
initial
= force_gimple_operand_gsi (&si, initial, true, NULL_TREE, true,
GSI_SAME_STMT);
if (gimple_in_ssa_p (cfun))
{
gcc_assert (gimple_seq_empty_p (phi_nodes (loop_header)));
phi = create_phi_node (loadedi, loop_header);
SET_USE (PHI_ARG_DEF_PTR_FROM_EDGE (phi, single_succ_edge (load_bb)),
initial);
}
else
gsi_insert_before (&si,
gimple_build_assign (loadedi, initial),
GSI_SAME_STMT);
if (loadedi != loaded_val)
{
gimple_stmt_iterator gsi2;
tree x;
x = build1 (VIEW_CONVERT_EXPR, type, loadedi);
gsi2 = gsi_start_bb (loop_header);
if (gimple_in_ssa_p (cfun))
{
gassign *stmt;
x = force_gimple_operand_gsi (&gsi2, x, true, NULL_TREE,
true, GSI_SAME_STMT);
stmt = gimple_build_assign (loaded_val, x);
gsi_insert_before (&gsi2, stmt, GSI_SAME_STMT);
}
else
{
x = build2 (MODIFY_EXPR, TREE_TYPE (loaded_val), loaded_val, x);
force_gimple_operand_gsi (&gsi2, x, true, NULL_TREE,
true, GSI_SAME_STMT);
}
}
gsi_remove (&si, true);
si = gsi_last_nondebug_bb (store_bb);
gcc_assert (gimple_code (gsi_stmt (si)) == GIMPLE_OMP_ATOMIC_STORE);
if (iaddr == addr)
storedi = stored_val;
else
storedi
= force_gimple_operand_gsi (&si,
build1 (VIEW_CONVERT_EXPR, itype,
stored_val), true, NULL_TREE, true,
GSI_SAME_STMT);
new_storedi = build_call_expr (cmpxchg, 3, iaddr, loadedi, storedi);
new_storedi = force_gimple_operand_gsi (&si,
fold_convert (TREE_TYPE (loadedi),
new_storedi),
true, NULL_TREE,
true, GSI_SAME_STMT);
if (gimple_in_ssa_p (cfun))
old_vali = loadedi;
else
{
old_vali = create_tmp_var (TREE_TYPE (loadedi));
stmt = gimple_build_assign (old_vali, loadedi);
gsi_insert_before (&si, stmt, GSI_SAME_STMT);
stmt = gimple_build_assign (loadedi, new_storedi);
gsi_insert_before (&si, stmt, GSI_SAME_STMT);
}
tree ne = build2 (NE_EXPR, boolean_type_node, new_storedi, old_vali);
stmt = gimple_build_cond_empty (ne);
gsi_insert_before (&si, stmt, GSI_SAME_STMT);
e = single_succ_edge (store_bb);
e->flags &= ~EDGE_FALLTHRU;
e->flags |= EDGE_FALSE_VALUE;
e->probability = profile_probability::guessed_always ();
e = make_edge (store_bb, loop_header, EDGE_TRUE_VALUE);
e->probability = profile_probability::guessed_never ();
if (gimple_in_ssa_p (cfun))
{
phi = gimple_seq_first_stmt (phi_nodes (loop_header));
SET_USE (PHI_ARG_DEF_PTR_FROM_EDGE (phi, e), new_storedi);
}
gsi_remove (&si, true);
struct loop *loop = alloc_loop ();
loop->header = loop_header;
loop->latch = store_bb;
add_loop (loop, loop_header->loop_father);
if (gimple_in_ssa_p (cfun))
update_ssa (TODO_update_ssa_no_phi);
return true;
}
static bool
expand_omp_atomic_mutex (basic_block load_bb, basic_block store_bb,
tree addr, tree loaded_val, tree stored_val)
{
gimple_stmt_iterator si;
gassign *stmt;
tree t;
si = gsi_last_nondebug_bb (load_bb);
gcc_assert (gimple_code (gsi_stmt (si)) == GIMPLE_OMP_ATOMIC_LOAD);
t = builtin_decl_explicit (BUILT_IN_GOMP_ATOMIC_START);
t = build_call_expr (t, 0);
force_gimple_operand_gsi (&si, t, true, NULL_TREE, true, GSI_SAME_STMT);
tree mem = build_simple_mem_ref (addr);
TREE_TYPE (mem) = TREE_TYPE (loaded_val);
TREE_OPERAND (mem, 1)
= fold_convert (build_pointer_type_for_mode (TREE_TYPE (mem), ptr_mode,
true),
TREE_OPERAND (mem, 1));
stmt = gimple_build_assign (loaded_val, mem);
gsi_insert_before (&si, stmt, GSI_SAME_STMT);
gsi_remove (&si, true);
si = gsi_last_nondebug_bb (store_bb);
gcc_assert (gimple_code (gsi_stmt (si)) == GIMPLE_OMP_ATOMIC_STORE);
stmt = gimple_build_assign (unshare_expr (mem), stored_val);
gsi_insert_before (&si, stmt, GSI_SAME_STMT);
t = builtin_decl_explicit (BUILT_IN_GOMP_ATOMIC_END);
t = build_call_expr (t, 0);
force_gimple_operand_gsi (&si, t, true, NULL_TREE, true, GSI_SAME_STMT);
gsi_remove (&si, true);
if (gimple_in_ssa_p (cfun))
update_ssa (TODO_update_ssa_no_phi);
return true;
}
static void
expand_omp_atomic (struct omp_region *region)
{
basic_block load_bb = region->entry, store_bb = region->exit;
gomp_atomic_load *load = as_a <gomp_atomic_load *> (last_stmt (load_bb));
gomp_atomic_store *store = as_a <gomp_atomic_store *> (last_stmt (store_bb));
tree loaded_val = gimple_omp_atomic_load_lhs (load);
tree addr = gimple_omp_atomic_load_rhs (load);
tree stored_val = gimple_omp_atomic_store_val (store);
tree type = TYPE_MAIN_VARIANT (TREE_TYPE (loaded_val));
HOST_WIDE_INT index;
index = tree_to_uhwi (TYPE_SIZE_UNIT (type));
index = exact_log2 (index);
if (index >= 0 && index <= 4)
{
unsigned int align = TYPE_ALIGN_UNIT (type);
if (exact_log2 (align) >= index)
{
scalar_mode smode;
if (loaded_val == stored_val
&& (is_int_mode (TYPE_MODE (type), &smode)
|| is_float_mode (TYPE_MODE (type), &smode))
&& GET_MODE_BITSIZE (smode) <= BITS_PER_WORD
&& expand_omp_atomic_load (load_bb, addr, loaded_val, index))
return;
if ((is_int_mode (TYPE_MODE (type), &smode)
|| is_float_mode (TYPE_MODE (type), &smode))
&& GET_MODE_BITSIZE (smode) <= BITS_PER_WORD
&& store_bb == single_succ (load_bb)
&& first_stmt (store_bb) == store
&& expand_omp_atomic_store (load_bb, addr, loaded_val,
stored_val, index))
return;
if ((INTEGRAL_TYPE_P (type) || POINTER_TYPE_P (type))
&& store_bb == single_succ (load_bb)
&& expand_omp_atomic_fetch_op (load_bb, addr,
loaded_val, stored_val, index))
return;
if (expand_omp_atomic_pipeline (load_bb, store_bb, addr,
loaded_val, stored_val, index))
return;
}
}
expand_omp_atomic_mutex (load_bb, store_bb, addr, loaded_val, stored_val);
}
static void
mark_loops_in_oacc_kernels_region (basic_block region_entry,
basic_block region_exit)
{
struct loop *outer = region_entry->loop_father;
gcc_assert (region_exit == NULL || outer == region_exit->loop_father);
unsigned int nr_outer_loops = 0;
struct loop *single_outer = NULL;
for (struct loop *loop = outer->inner; loop != NULL; loop = loop->next)
{
gcc_assert (loop_outer (loop) == outer);
if (!dominated_by_p (CDI_DOMINATORS, loop->header, region_entry))
continue;
if (region_exit != NULL
&& dominated_by_p (CDI_DOMINATORS, loop->header, region_exit))
continue;
nr_outer_loops++;
single_outer = loop;
}
if (nr_outer_loops != 1)
return;
for (struct loop *loop = single_outer->inner;
loop != NULL;
loop = loop->inner)
if (loop->next)
return;
for (struct loop *loop = single_outer; loop != NULL; loop = loop->inner)
loop->in_oacc_kernels_region = true;
}
struct GTY(()) grid_launch_attributes_trees
{
tree kernel_dim_array_type;
tree kernel_lattrs_dimnum_decl;
tree kernel_lattrs_grid_decl;
tree kernel_lattrs_group_decl;
tree kernel_launch_attributes_type;
};
static GTY(()) struct grid_launch_attributes_trees *grid_attr_trees;
static void
grid_create_kernel_launch_attr_types (void)
{
if (grid_attr_trees)
return;
grid_attr_trees = ggc_alloc <grid_launch_attributes_trees> ();
tree dim_arr_index_type
= build_index_type (build_int_cst (integer_type_node, 2));
grid_attr_trees->kernel_dim_array_type
= build_array_type (uint32_type_node, dim_arr_index_type);
grid_attr_trees->kernel_launch_attributes_type = make_node (RECORD_TYPE);
grid_attr_trees->kernel_lattrs_dimnum_decl
= build_decl (BUILTINS_LOCATION, FIELD_DECL, get_identifier ("ndim"),
uint32_type_node);
DECL_CHAIN (grid_attr_trees->kernel_lattrs_dimnum_decl) = NULL_TREE;
grid_attr_trees->kernel_lattrs_grid_decl
= build_decl (BUILTINS_LOCATION, FIELD_DECL, get_identifier ("grid_size"),
grid_attr_trees->kernel_dim_array_type);
DECL_CHAIN (grid_attr_trees->kernel_lattrs_grid_decl)
= grid_attr_trees->kernel_lattrs_dimnum_decl;
grid_attr_trees->kernel_lattrs_group_decl
= build_decl (BUILTINS_LOCATION, FIELD_DECL, get_identifier ("group_size"),
grid_attr_trees->kernel_dim_array_type);
DECL_CHAIN (grid_attr_trees->kernel_lattrs_group_decl)
= grid_attr_trees->kernel_lattrs_grid_decl;
finish_builtin_struct (grid_attr_trees->kernel_launch_attributes_type,
"__gomp_kernel_launch_attributes",
grid_attr_trees->kernel_lattrs_group_decl, NULL_TREE);
}
static void
grid_insert_store_range_dim (gimple_stmt_iterator *gsi, tree range_var,
tree fld_decl, int index, tree value)
{
tree ref = build4 (ARRAY_REF, uint32_type_node,
build3 (COMPONENT_REF,
grid_attr_trees->kernel_dim_array_type,
range_var, fld_decl, NULL_TREE),
build_int_cst (integer_type_node, index),
NULL_TREE, NULL_TREE);
gsi_insert_before (gsi, gimple_build_assign (ref, value), GSI_SAME_STMT);
}
static tree
grid_get_kernel_launch_attributes (gimple_stmt_iterator *gsi,
gomp_target *tgt_stmt)
{
grid_create_kernel_launch_attr_types ();
tree lattrs = create_tmp_var (grid_attr_trees->kernel_launch_attributes_type,
"__kernel_launch_attrs");
unsigned max_dim = 0;
for (tree clause = gimple_omp_target_clauses (tgt_stmt);
clause;
clause = OMP_CLAUSE_CHAIN (clause))
{
if (OMP_CLAUSE_CODE (clause) != OMP_CLAUSE__GRIDDIM_)
continue;
unsigned dim = OMP_CLAUSE__GRIDDIM__DIMENSION (clause);
max_dim = MAX (dim, max_dim);
grid_insert_store_range_dim (gsi, lattrs,
grid_attr_trees->kernel_lattrs_grid_decl,
dim, OMP_CLAUSE__GRIDDIM__SIZE (clause));
grid_insert_store_range_dim (gsi, lattrs,
grid_attr_trees->kernel_lattrs_group_decl,
dim, OMP_CLAUSE__GRIDDIM__GROUP (clause));
}
tree dimref = build3 (COMPONENT_REF, uint32_type_node, lattrs,
grid_attr_trees->kernel_lattrs_dimnum_decl, NULL_TREE);
gcc_checking_assert (max_dim <= 2);
tree dimensions = build_int_cstu (uint32_type_node, max_dim + 1);
gsi_insert_before (gsi, gimple_build_assign (dimref, dimensions),
GSI_SAME_STMT);
TREE_ADDRESSABLE (lattrs) = 1;
return build_fold_addr_expr (lattrs);
}
static tree
get_target_argument_identifier_1 (int device, bool subseqent_param, int id)
{
tree t = build_int_cst (integer_type_node, device);
if (subseqent_param)
t = fold_build2 (BIT_IOR_EXPR, integer_type_node, t,
build_int_cst (integer_type_node,
GOMP_TARGET_ARG_SUBSEQUENT_PARAM));
t = fold_build2 (BIT_IOR_EXPR, integer_type_node, t,
build_int_cst (integer_type_node, id));
return t;
}
static tree
get_target_argument_identifier (int device, bool subseqent_param, int id)
{
tree t = get_target_argument_identifier_1 (device, subseqent_param, id);
return fold_convert (ptr_type_node, t);
}
static tree
get_target_argument_value (gimple_stmt_iterator *gsi, int device, int id,
tree value)
{
tree t = fold_build2 (LSHIFT_EXPR, integer_type_node,
fold_convert (integer_type_node, value),
build_int_cst (unsigned_type_node,
GOMP_TARGET_ARG_VALUE_SHIFT));
t = fold_build2 (BIT_IOR_EXPR, integer_type_node, t,
get_target_argument_identifier_1 (device, false, id));
t = fold_convert (ptr_type_node, t);
return force_gimple_operand_gsi (gsi, t, true, NULL, true, GSI_SAME_STMT);
}
static void
push_target_argument_according_to_value (gimple_stmt_iterator *gsi, int device,
int id, tree value, vec <tree> *args)
{
if (tree_fits_shwi_p (value)
&& tree_to_shwi (value) > -(1 << 15)
&& tree_to_shwi (value) < (1 << 15))
args->quick_push (get_target_argument_value (gsi, device, id, value));
else
{
args->quick_push (get_target_argument_identifier (device, true, id));
value = fold_convert (ptr_type_node, value);
value = force_gimple_operand_gsi (gsi, value, true, NULL, true,
GSI_SAME_STMT);
args->quick_push (value);
}
}
static tree
get_target_arguments (gimple_stmt_iterator *gsi, gomp_target *tgt_stmt)
{
auto_vec <tree, 6> args;
tree clauses = gimple_omp_target_clauses (tgt_stmt);
tree t, c = omp_find_clause (clauses, OMP_CLAUSE_NUM_TEAMS);
if (c)
t = OMP_CLAUSE_NUM_TEAMS_EXPR (c);
else
t = integer_minus_one_node;
push_target_argument_according_to_value (gsi, GOMP_TARGET_ARG_DEVICE_ALL,
GOMP_TARGET_ARG_NUM_TEAMS, t, &args);
c = omp_find_clause (clauses, OMP_CLAUSE_THREAD_LIMIT);
if (c)
t = OMP_CLAUSE_THREAD_LIMIT_EXPR (c);
else
t = integer_minus_one_node;
push_target_argument_according_to_value (gsi, GOMP_TARGET_ARG_DEVICE_ALL,
GOMP_TARGET_ARG_THREAD_LIMIT, t,
&args);
if (omp_find_clause (gimple_omp_target_clauses (tgt_stmt),
OMP_CLAUSE__GRIDDIM_))
{
int id = GOMP_TARGET_ARG_HSA_KERNEL_ATTRIBUTES;
t = get_target_argument_identifier (GOMP_DEVICE_HSA, true, id);
args.quick_push (t);
args.quick_push (grid_get_kernel_launch_attributes (gsi, tgt_stmt));
}
tree argarray = create_tmp_var (build_array_type_nelts (ptr_type_node,
args.length () + 1),
".omp_target_args");
for (unsigned i = 0; i < args.length (); i++)
{
tree ref = build4 (ARRAY_REF, ptr_type_node, argarray,
build_int_cst (integer_type_node, i),
NULL_TREE, NULL_TREE);
gsi_insert_before (gsi, gimple_build_assign (ref, args[i]),
GSI_SAME_STMT);
}
tree ref = build4 (ARRAY_REF, ptr_type_node, argarray,
build_int_cst (integer_type_node, args.length ()),
NULL_TREE, NULL_TREE);
gsi_insert_before (gsi, gimple_build_assign (ref, null_pointer_node),
GSI_SAME_STMT);
TREE_ADDRESSABLE (argarray) = 1;
return build_fold_addr_expr (argarray);
}
static void
expand_omp_target (struct omp_region *region)
{
basic_block entry_bb, exit_bb, new_bb;
struct function *child_cfun;
tree child_fn, block, t;
gimple_stmt_iterator gsi;
gomp_target *entry_stmt;
gimple *stmt;
edge e;
bool offloaded, data_region;
entry_stmt = as_a <gomp_target *> (last_stmt (region->entry));
new_bb = region->entry;
offloaded = is_gimple_omp_offloaded (entry_stmt);
switch (gimple_omp_target_kind (entry_stmt))
{
case GF_OMP_TARGET_KIND_REGION:
case GF_OMP_TARGET_KIND_UPDATE:
case GF_OMP_TARGET_KIND_ENTER_DATA:
case GF_OMP_TARGET_KIND_EXIT_DATA:
case GF_OMP_TARGET_KIND_OACC_PARALLEL:
case GF_OMP_TARGET_KIND_OACC_KERNELS:
case GF_OMP_TARGET_KIND_OACC_UPDATE:
case GF_OMP_TARGET_KIND_OACC_ENTER_EXIT_DATA:
case GF_OMP_TARGET_KIND_OACC_DECLARE:
data_region = false;
break;
case GF_OMP_TARGET_KIND_DATA:
case GF_OMP_TARGET_KIND_OACC_DATA:
case GF_OMP_TARGET_KIND_OACC_HOST_DATA:
data_region = true;
break;
default:
gcc_unreachable ();
}
child_fn = NULL_TREE;
child_cfun = NULL;
if (offloaded)
{
child_fn = gimple_omp_target_child_fn (entry_stmt);
child_cfun = DECL_STRUCT_FUNCTION (child_fn);
}
if (child_cfun != NULL)
gcc_checking_assert (!child_cfun->cfg);
gcc_checking_assert (!gimple_in_ssa_p (cfun));
entry_bb = region->entry;
exit_bb = region->exit;
if (gimple_omp_target_kind (entry_stmt) == GF_OMP_TARGET_KIND_OACC_KERNELS)
{
mark_loops_in_oacc_kernels_region (region->entry, region->exit);
DECL_ATTRIBUTES (child_fn)
= tree_cons (get_identifier ("oacc kernels"),
NULL_TREE, DECL_ATTRIBUTES (child_fn));
}
if (offloaded)
{
unsigned srcidx, dstidx, num;
tree data_arg = gimple_omp_target_data_arg (entry_stmt);
if (data_arg)
{
basic_block entry_succ_bb = single_succ (entry_bb);
gimple_stmt_iterator gsi;
tree arg;
gimple *tgtcopy_stmt = NULL;
tree sender = TREE_VEC_ELT (data_arg, 0);
for (gsi = gsi_start_bb (entry_succ_bb); ; gsi_next (&gsi))
{
gcc_assert (!gsi_end_p (gsi));
stmt = gsi_stmt (gsi);
if (gimple_code (stmt) != GIMPLE_ASSIGN)
continue;
if (gimple_num_ops (stmt) == 2)
{
tree arg = gimple_assign_rhs1 (stmt);
if (TREE_CODE (arg) == ADDR_EXPR
&& TREE_OPERAND (arg, 0) == sender)
{
tgtcopy_stmt = stmt;
break;
}
}
}
gcc_assert (tgtcopy_stmt != NULL);
arg = DECL_ARGUMENTS (child_fn);
gcc_assert (gimple_assign_lhs (tgtcopy_stmt) == arg);
gsi_remove (&gsi, true);
}
block = DECL_INITIAL (child_fn);
BLOCK_VARS (block) = vec2chain (child_cfun->local_decls);
for (t = BLOCK_VARS (block); t; t = DECL_CHAIN (t))
if (VAR_P (t) && TREE_STATIC (t) && !DECL_EXTERNAL (t))
varpool_node::finalize_decl (t);
DECL_SAVED_TREE (child_fn) = NULL;
gimple_set_body (child_fn, NULL);
TREE_USED (block) = 1;
for (t = DECL_ARGUMENTS (child_fn); t; t = DECL_CHAIN (t))
DECL_CONTEXT (t) = child_fn;
gsi = gsi_last_nondebug_bb (entry_bb);
stmt = gsi_stmt (gsi);
gcc_assert (stmt
&& gimple_code (stmt) == gimple_code (entry_stmt));
e = split_block (entry_bb, stmt);
gsi_remove (&gsi, true);
entry_bb = e->dest;
single_succ_edge (entry_bb)->flags = EDGE_FALLTHRU;
if (exit_bb)
{
gsi = gsi_last_nondebug_bb (exit_bb);
gcc_assert (!gsi_end_p (gsi)
&& gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_RETURN);
stmt = gimple_build_return (NULL);
gsi_insert_after (&gsi, stmt, GSI_SAME_STMT);
gsi_remove (&gsi, true);
}
block = gimple_block (entry_stmt);
new_bb = move_sese_region_to_fn (child_cfun, entry_bb, exit_bb, block);
if (exit_bb)
single_succ_edge (new_bb)->flags = EDGE_FALLTHRU;
if (loops_state_satisfies_p (LOOPS_NEED_FIXUP))
child_cfun->x_current_loops->state |= LOOPS_NEED_FIXUP;
num = vec_safe_length (child_cfun->local_decls);
for (srcidx = 0, dstidx = 0; srcidx < num; srcidx++)
{
t = (*child_cfun->local_decls)[srcidx];
if (DECL_CONTEXT (t) == cfun->decl)
continue;
if (srcidx != dstidx)
(*child_cfun->local_decls)[dstidx] = t;
dstidx++;
}
if (dstidx != num)
vec_safe_truncate (child_cfun->local_decls, dstidx);
child_cfun->curr_properties = cfun->curr_properties;
child_cfun->has_simduid_loops |= cfun->has_simduid_loops;
child_cfun->has_force_vectorize_loops |= cfun->has_force_vectorize_loops;
cgraph_node *node = cgraph_node::get_create (child_fn);
node->parallelized_function = 1;
cgraph_node::add_new_function (child_fn, true);
if (ENABLE_OFFLOADING)
{
if (in_lto_p)
DECL_PRESERVE_P (child_fn) = 1;
vec_safe_push (offload_funcs, child_fn);
}
bool need_asm = DECL_ASSEMBLER_NAME_SET_P (current_function_decl)
&& !DECL_ASSEMBLER_NAME_SET_P (child_fn);
push_cfun (child_cfun);
if (need_asm)
assign_assembler_name_if_needed (child_fn);
cgraph_edge::rebuild_edges ();
if (flag_exceptions)
{
basic_block bb;
bool changed = false;
FOR_EACH_BB_FN (bb, cfun)
changed |= gimple_purge_dead_eh_edges (bb);
if (changed)
cleanup_tree_cfg ();
}
if (flag_checking && !loops_state_satisfies_p (LOOPS_NEED_FIXUP))
verify_loop_structure ();
pop_cfun ();
if (dump_file && !gimple_in_ssa_p (cfun))
{
omp_any_child_fn_dumped = true;
dump_function_header (dump_file, child_fn, dump_flags);
dump_function_to_file (child_fn, dump_file, dump_flags);
}
adjust_context_and_scope (region, gimple_block (entry_stmt), child_fn);
}
tree t1, t2, t3, t4, device, cond, depend, c, clauses;
enum built_in_function start_ix;
location_t clause_loc;
unsigned int flags_i = 0;
switch (gimple_omp_target_kind (entry_stmt))
{
case GF_OMP_TARGET_KIND_REGION:
start_ix = BUILT_IN_GOMP_TARGET;
break;
case GF_OMP_TARGET_KIND_DATA:
start_ix = BUILT_IN_GOMP_TARGET_DATA;
break;
case GF_OMP_TARGET_KIND_UPDATE:
start_ix = BUILT_IN_GOMP_TARGET_UPDATE;
break;
case GF_OMP_TARGET_KIND_ENTER_DATA:
start_ix = BUILT_IN_GOMP_TARGET_ENTER_EXIT_DATA;
break;
case GF_OMP_TARGET_KIND_EXIT_DATA:
start_ix = BUILT_IN_GOMP_TARGET_ENTER_EXIT_DATA;
flags_i |= GOMP_TARGET_FLAG_EXIT_DATA;
break;
case GF_OMP_TARGET_KIND_OACC_KERNELS:
case GF_OMP_TARGET_KIND_OACC_PARALLEL:
start_ix = BUILT_IN_GOACC_PARALLEL;
break;
case GF_OMP_TARGET_KIND_OACC_DATA:
case GF_OMP_TARGET_KIND_OACC_HOST_DATA:
start_ix = BUILT_IN_GOACC_DATA_START;
break;
case GF_OMP_TARGET_KIND_OACC_UPDATE:
start_ix = BUILT_IN_GOACC_UPDATE;
break;
case GF_OMP_TARGET_KIND_OACC_ENTER_EXIT_DATA:
start_ix = BUILT_IN_GOACC_ENTER_EXIT_DATA;
break;
case GF_OMP_TARGET_KIND_OACC_DECLARE:
start_ix = BUILT_IN_GOACC_DECLARE;
break;
default:
gcc_unreachable ();
}
clauses = gimple_omp_target_clauses (entry_stmt);
cond = NULL_TREE;
device = build_int_cst (integer_type_node, GOMP_DEVICE_ICV);
c = omp_find_clause (clauses, OMP_CLAUSE_IF);
if (c)
cond = OMP_CLAUSE_IF_EXPR (c);
c = omp_find_clause (clauses, OMP_CLAUSE_DEVICE);
if (c)
{
gcc_checking_assert (start_ix == BUILT_IN_GOMP_TARGET
|| start_ix == BUILT_IN_GOMP_TARGET_DATA
|| start_ix == BUILT_IN_GOMP_TARGET_UPDATE
|| start_ix == BUILT_IN_GOMP_TARGET_ENTER_EXIT_DATA);
device = OMP_CLAUSE_DEVICE_ID (c);
clause_loc = OMP_CLAUSE_LOCATION (c);
}
else
clause_loc = gimple_location (entry_stmt);
c = omp_find_clause (clauses, OMP_CLAUSE_NOWAIT);
if (c)
flags_i |= GOMP_TARGET_FLAG_NOWAIT;
device = fold_convert_loc (clause_loc, integer_type_node, device);
if (cond)
{
cond = gimple_boolify (cond);
basic_block cond_bb, then_bb, else_bb;
edge e;
tree tmp_var;
tmp_var = create_tmp_var (TREE_TYPE (device));
if (offloaded)
e = split_block_after_labels (new_bb);
else
{
gsi = gsi_last_nondebug_bb (new_bb);
gsi_prev (&gsi);
e = split_block (new_bb, gsi_stmt (gsi));
}
cond_bb = e->src;
new_bb = e->dest;
remove_edge (e);
then_bb = create_empty_bb (cond_bb);
else_bb = create_empty_bb (then_bb);
set_immediate_dominator (CDI_DOMINATORS, then_bb, cond_bb);
set_immediate_dominator (CDI_DOMINATORS, else_bb, cond_bb);
stmt = gimple_build_cond_empty (cond);
gsi = gsi_last_bb (cond_bb);
gsi_insert_after (&gsi, stmt, GSI_CONTINUE_LINKING);
gsi = gsi_start_bb (then_bb);
stmt = gimple_build_assign (tmp_var, device);
gsi_insert_after (&gsi, stmt, GSI_CONTINUE_LINKING);
gsi = gsi_start_bb (else_bb);
stmt = gimple_build_assign (tmp_var,
build_int_cst (integer_type_node,
GOMP_DEVICE_HOST_FALLBACK));
gsi_insert_after (&gsi, stmt, GSI_CONTINUE_LINKING);
make_edge (cond_bb, then_bb, EDGE_TRUE_VALUE);
make_edge (cond_bb, else_bb, EDGE_FALSE_VALUE);
add_bb_to_loop (then_bb, cond_bb->loop_father);
add_bb_to_loop (else_bb, cond_bb->loop_father);
make_edge (then_bb, new_bb, EDGE_FALLTHRU);
make_edge (else_bb, new_bb, EDGE_FALLTHRU);
device = tmp_var;
gsi = gsi_last_nondebug_bb (new_bb);
}
else
{
gsi = gsi_last_nondebug_bb (new_bb);
device = force_gimple_operand_gsi (&gsi, device, true, NULL_TREE,
true, GSI_SAME_STMT);
}
t = gimple_omp_target_data_arg (entry_stmt);
if (t == NULL)
{
t1 = size_zero_node;
t2 = build_zero_cst (ptr_type_node);
t3 = t2;
t4 = t2;
}
else
{
t1 = TYPE_MAX_VALUE (TYPE_DOMAIN (TREE_TYPE (TREE_VEC_ELT (t, 1))));
t1 = size_binop (PLUS_EXPR, t1, size_int (1));
t2 = build_fold_addr_expr (TREE_VEC_ELT (t, 0));
t3 = build_fold_addr_expr (TREE_VEC_ELT (t, 1));
t4 = build_fold_addr_expr (TREE_VEC_ELT (t, 2));
}
gimple *g;
bool tagging = false;
auto_vec<tree, 11> args;
args.quick_push (device);
if (offloaded)
args.quick_push (build_fold_addr_expr (child_fn));
args.quick_push (t1);
args.quick_push (t2);
args.quick_push (t3);
args.quick_push (t4);
switch (start_ix)
{
case BUILT_IN_GOACC_DATA_START:
case BUILT_IN_GOACC_DECLARE:
case BUILT_IN_GOMP_TARGET_DATA:
break;
case BUILT_IN_GOMP_TARGET:
case BUILT_IN_GOMP_TARGET_UPDATE:
case BUILT_IN_GOMP_TARGET_ENTER_EXIT_DATA:
args.quick_push (build_int_cst (unsigned_type_node, flags_i));
c = omp_find_clause (clauses, OMP_CLAUSE_DEPEND);
if (c)
depend = OMP_CLAUSE_DECL (c);
else
depend = build_int_cst (ptr_type_node, 0);
args.quick_push (depend);
if (start_ix == BUILT_IN_GOMP_TARGET)
args.quick_push (get_target_arguments (&gsi, entry_stmt));
break;
case BUILT_IN_GOACC_PARALLEL:
oacc_set_fn_attrib (child_fn, clauses, &args);
tagging = true;
case BUILT_IN_GOACC_ENTER_EXIT_DATA:
case BUILT_IN_GOACC_UPDATE:
{
tree t_async = NULL_TREE;
c = omp_find_clause (clauses, OMP_CLAUSE_ASYNC);
if (c)
t_async = fold_convert_loc (OMP_CLAUSE_LOCATION (c),
integer_type_node,
OMP_CLAUSE_ASYNC_EXPR (c));
else if (!tagging)
t_async = fold_convert_loc (gimple_location (entry_stmt),
integer_type_node,
build_int_cst (integer_type_node,
GOMP_ASYNC_SYNC));
if (tagging && t_async)
{
unsigned HOST_WIDE_INT i_async = GOMP_LAUNCH_OP_MAX;
if (TREE_CODE (t_async) == INTEGER_CST)
{
i_async = TREE_INT_CST_LOW (t_async);
if (i_async < GOMP_LAUNCH_OP_MAX)
t_async = NULL_TREE;
else
i_async = GOMP_LAUNCH_OP_MAX;
}
args.safe_push (oacc_launch_pack (GOMP_LAUNCH_ASYNC, NULL_TREE,
i_async));
}
if (t_async)
args.safe_push (t_async);
unsigned t_wait_idx = args.length ();
unsigned num_waits = 0;
c = omp_find_clause (clauses, OMP_CLAUSE_WAIT);
if (!tagging || c)
args.safe_push (integer_zero_node);
for (; c; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_WAIT)
{
args.safe_push (fold_convert_loc (OMP_CLAUSE_LOCATION (c),
integer_type_node,
OMP_CLAUSE_WAIT_EXPR (c)));
num_waits++;
}
if (!tagging || num_waits)
{
tree len;
if (tagging)
len = oacc_launch_pack (GOMP_LAUNCH_WAIT, NULL_TREE, num_waits);
else
len = build_int_cst (integer_type_node, num_waits);
len = fold_convert_loc (gimple_location (entry_stmt),
unsigned_type_node, len);
args[t_wait_idx] = len;
}
}
break;
default:
gcc_unreachable ();
}
if (tagging)
args.safe_push (oacc_launch_pack (0, NULL_TREE, 0));
g = gimple_build_call_vec (builtin_decl_explicit (start_ix), args);
gimple_set_location (g, gimple_location (entry_stmt));
gsi_insert_before (&gsi, g, GSI_SAME_STMT);
if (!offloaded)
{
g = gsi_stmt (gsi);
gcc_assert (g && gimple_code (g) == GIMPLE_OMP_TARGET);
gsi_remove (&gsi, true);
}
if (data_region && region->exit)
{
gsi = gsi_last_nondebug_bb (region->exit);
g = gsi_stmt (gsi);
gcc_assert (g && gimple_code (g) == GIMPLE_OMP_RETURN);
gsi_remove (&gsi, true);
}
}
static void
grid_expand_omp_for_loop (struct omp_region *kfor, bool intra_group)
{
gimple_stmt_iterator gsi;
gomp_for *for_stmt = as_a <gomp_for *> (last_stmt (kfor->entry));
gcc_checking_assert (gimple_omp_for_kind (for_stmt)
== GF_OMP_FOR_KIND_GRID_LOOP);
size_t collapse = gimple_omp_for_collapse (for_stmt);
struct omp_for_data_loop *loops
= XALLOCAVEC (struct omp_for_data_loop,
gimple_omp_for_collapse (for_stmt));
struct omp_for_data fd;
remove_edge (BRANCH_EDGE (kfor->entry));
basic_block body_bb = FALLTHRU_EDGE (kfor->entry)->dest;
gcc_assert (kfor->cont);
omp_extract_for_data (for_stmt, &fd, loops);
gsi = gsi_start_bb (body_bb);
for (size_t dim = 0; dim < collapse; dim++)
{
tree type, itype;
itype = type = TREE_TYPE (fd.loops[dim].v);
if (POINTER_TYPE_P (type))
itype = signed_type_for (type);
tree n1 = fd.loops[dim].n1;
tree step = fd.loops[dim].step;
n1 = force_gimple_operand_gsi (&gsi, fold_convert (type, n1),
true, NULL_TREE, true, GSI_SAME_STMT);
step = force_gimple_operand_gsi (&gsi, fold_convert (itype, step),
true, NULL_TREE, true, GSI_SAME_STMT);
tree threadid;
if (gimple_omp_for_grid_group_iter (for_stmt))
{
gcc_checking_assert (!intra_group);
threadid = build_call_expr (builtin_decl_explicit
(BUILT_IN_HSA_WORKGROUPID), 1,
build_int_cstu (unsigned_type_node, dim));
}
else if (intra_group)
threadid = build_call_expr (builtin_decl_explicit
(BUILT_IN_HSA_WORKITEMID), 1,
build_int_cstu (unsigned_type_node, dim));
else
threadid = build_call_expr (builtin_decl_explicit
(BUILT_IN_HSA_WORKITEMABSID), 1,
build_int_cstu (unsigned_type_node, dim));
threadid = fold_convert (itype, threadid);
threadid = force_gimple_operand_gsi (&gsi, threadid, true, NULL_TREE,
true, GSI_SAME_STMT);
tree startvar = fd.loops[dim].v;
tree t = fold_build2 (MULT_EXPR, itype, threadid, step);
if (POINTER_TYPE_P (type))
t = fold_build_pointer_plus (n1, t);
else
t = fold_build2 (PLUS_EXPR, type, t, n1);
t = fold_convert (type, t);
t = force_gimple_operand_gsi (&gsi, t,
DECL_P (startvar)
&& TREE_ADDRESSABLE (startvar),
NULL_TREE, true, GSI_SAME_STMT);
gassign *assign_stmt = gimple_build_assign (startvar, t);
gsi_insert_before (&gsi, assign_stmt, GSI_SAME_STMT);
}
gsi = gsi_last_nondebug_bb (kfor->entry);
gsi_remove (&gsi, true);
gsi = gsi_last_nondebug_bb (kfor->cont);
gcc_assert (!gsi_end_p (gsi)
&& gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_CONTINUE);
gsi_remove (&gsi, true);
gsi = gsi_last_nondebug_bb (kfor->exit);
gcc_assert (!gsi_end_p (gsi)
&& gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_RETURN);
if (intra_group)
gsi_insert_before (&gsi, omp_build_barrier (NULL_TREE), GSI_SAME_STMT);
gsi_remove (&gsi, true);
remove_edge (find_edge (kfor->cont, body_bb));
if (kfor->cont != body_bb)
set_immediate_dominator (CDI_DOMINATORS, kfor->cont, body_bb);
set_immediate_dominator (CDI_DOMINATORS, kfor->exit, kfor->cont);
}
struct grid_arg_decl_map
{
tree old_arg;
tree new_arg;
};
static tree
grid_remap_kernel_arg_accesses (tree *tp, int *walk_subtrees, void *data)
{
struct walk_stmt_info *wi = (struct walk_stmt_info *) data;
struct grid_arg_decl_map *adm = (struct grid_arg_decl_map *) wi->info;
tree t = *tp;
if (t == adm->old_arg)
*tp = adm->new_arg;
*walk_subtrees = !TYPE_P (t) && !DECL_P (t);
return NULL_TREE;
}
static void
grid_expand_target_grid_body (struct omp_region *target)
{
if (!hsa_gen_requested_p ())
return;
gomp_target *tgt_stmt = as_a <gomp_target *> (last_stmt (target->entry));
struct omp_region **pp;
for (pp = &target->inner; *pp; pp = &(*pp)->next)
if ((*pp)->type == GIMPLE_OMP_GRID_BODY)
break;
struct omp_region *gpukernel = *pp;
tree orig_child_fndecl = gimple_omp_target_child_fn (tgt_stmt);
if (!gpukernel)
{
if (gimple_omp_target_kind (tgt_stmt) != GF_OMP_TARGET_KIND_REGION)
return;
gcc_checking_assert (orig_child_fndecl);
gcc_assert (!omp_find_clause (gimple_omp_target_clauses (tgt_stmt),
OMP_CLAUSE__GRIDDIM_));
cgraph_node *n = cgraph_node::get (orig_child_fndecl);
hsa_register_kernel (n);
return;
}
gcc_assert (omp_find_clause (gimple_omp_target_clauses (tgt_stmt),
OMP_CLAUSE__GRIDDIM_));
tree inside_block
= gimple_block (first_stmt (single_succ (gpukernel->entry)));
*pp = gpukernel->next;
for (pp = &gpukernel->inner; *pp; pp = &(*pp)->next)
if ((*pp)->type == GIMPLE_OMP_FOR)
break;
struct omp_region *kfor = *pp;
gcc_assert (kfor);
gomp_for *for_stmt = as_a <gomp_for *> (last_stmt (kfor->entry));
gcc_assert (gimple_omp_for_kind (for_stmt) == GF_OMP_FOR_KIND_GRID_LOOP);
*pp = kfor->next;
if (kfor->inner)
{
if (gimple_omp_for_grid_group_iter (for_stmt))
{
struct omp_region **next_pp;
for (pp = &kfor->inner; *pp; pp = next_pp)
{
next_pp = &(*pp)->next;
if ((*pp)->type != GIMPLE_OMP_FOR)
continue;
gomp_for *inner = as_a <gomp_for *> (last_stmt ((*pp)->entry));
gcc_assert (gimple_omp_for_kind (inner)
== GF_OMP_FOR_KIND_GRID_LOOP);
grid_expand_omp_for_loop (*pp, true);
*pp = (*pp)->next;
next_pp = pp;
}
}
expand_omp (kfor->inner);
}
if (gpukernel->inner)
expand_omp (gpukernel->inner);
tree kern_fndecl = copy_node (orig_child_fndecl);
DECL_NAME (kern_fndecl) = clone_function_name (kern_fndecl, "kernel");
SET_DECL_ASSEMBLER_NAME (kern_fndecl, DECL_NAME (kern_fndecl));
tree tgtblock = gimple_block (tgt_stmt);
tree fniniblock = make_node (BLOCK);
BLOCK_ABSTRACT_ORIGIN (fniniblock) = tgtblock;
BLOCK_SOURCE_LOCATION (fniniblock) = BLOCK_SOURCE_LOCATION (tgtblock);
BLOCK_SOURCE_END_LOCATION (fniniblock) = BLOCK_SOURCE_END_LOCATION (tgtblock);
BLOCK_SUPERCONTEXT (fniniblock) = kern_fndecl;
DECL_INITIAL (kern_fndecl) = fniniblock;
push_struct_function (kern_fndecl);
cfun->function_end_locus = gimple_location (tgt_stmt);
init_tree_ssa (cfun);
pop_cfun ();
tree old_parm_decl = DECL_ARGUMENTS (kern_fndecl);
gcc_assert (!DECL_CHAIN (old_parm_decl));
tree new_parm_decl = copy_node (DECL_ARGUMENTS (kern_fndecl));
DECL_CONTEXT (new_parm_decl) = kern_fndecl;
DECL_ARGUMENTS (kern_fndecl) = new_parm_decl;
gcc_assert (VOID_TYPE_P (TREE_TYPE (DECL_RESULT (kern_fndecl))));
DECL_RESULT (kern_fndecl) = copy_node (DECL_RESULT (kern_fndecl));
DECL_CONTEXT (DECL_RESULT (kern_fndecl)) = kern_fndecl;
struct function *kern_cfun = DECL_STRUCT_FUNCTION (kern_fndecl);
kern_cfun->curr_properties = cfun->curr_properties;
grid_expand_omp_for_loop (kfor, false);
gimple_stmt_iterator gsi = gsi_last_nondebug_bb (gpukernel->entry);
gsi_remove (&gsi, true);
gsi = gsi_last_nondebug_bb (gpukernel->exit);
gcc_assert (!gsi_end_p (gsi)
&& gimple_code (gsi_stmt (gsi)) == GIMPLE_OMP_RETURN);
gimple *ret_stmt = gimple_build_return (NULL);
gsi_insert_after (&gsi, ret_stmt, GSI_SAME_STMT);
gsi_remove (&gsi, true);
gsi = gsi_start_bb (single_succ (gpukernel->entry));
tree data_arg = gimple_omp_target_data_arg (tgt_stmt);
tree sender = data_arg ? TREE_VEC_ELT (data_arg, 0) : NULL;
for (gimple_stmt_iterator tsi = gsi_start_bb (single_succ (target->entry));
!gsi_end_p (tsi); gsi_next (&tsi))
{
gimple *stmt = gsi_stmt (tsi);
if (is_gimple_omp (stmt))
break;
if (sender
&& is_gimple_assign (stmt)
&& TREE_CODE (gimple_assign_rhs1 (stmt)) == ADDR_EXPR
&& TREE_OPERAND (gimple_assign_rhs1 (stmt), 0) == sender)
continue;
gimple *copy = gimple_copy (stmt);
gsi_insert_before (&gsi, copy, GSI_SAME_STMT);
gimple_set_block (copy, fniniblock);
}
move_sese_region_to_fn (kern_cfun, single_succ (gpukernel->entry),
gpukernel->exit, inside_block);
cgraph_node *kcn = cgraph_node::get_create (kern_fndecl);
kcn->mark_force_output ();
cgraph_node *orig_child = cgraph_node::get (orig_child_fndecl);
hsa_register_kernel (kcn, orig_child);
cgraph_node::add_new_function (kern_fndecl, true);
push_cfun (kern_cfun);
cgraph_edge::rebuild_edges ();
struct grid_arg_decl_map adm;
adm.old_arg = old_parm_decl;
adm.new_arg = new_parm_decl;
basic_block bb;
FOR_EACH_BB_FN (bb, kern_cfun)
{
for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi); gsi_next (&gsi))
{
gimple *stmt = gsi_stmt (gsi);
struct walk_stmt_info wi;
memset (&wi, 0, sizeof (wi));
wi.info = &adm;
walk_gimple_op (stmt, grid_remap_kernel_arg_accesses, &wi);
}
}
pop_cfun ();
return;
}
static void
expand_omp (struct omp_region *region)
{
omp_any_child_fn_dumped = false;
while (region)
{
location_t saved_location;
gimple *inner_stmt = NULL;
if (region->type == GIMPLE_OMP_PARALLEL)
determine_parallel_type (region);
else if (region->type == GIMPLE_OMP_TARGET)
grid_expand_target_grid_body (region);
if (region->type == GIMPLE_OMP_FOR
&& gimple_omp_for_combined_p (last_stmt (region->entry)))
inner_stmt = last_stmt (region->inner->entry);
if (region->inner)
expand_omp (region->inner);
saved_location = input_location;
if (gimple_has_location (last_stmt (region->entry)))
input_location = gimple_location (last_stmt (region->entry));
switch (region->type)
{
case GIMPLE_OMP_PARALLEL:
case GIMPLE_OMP_TASK:
expand_omp_taskreg (region);
break;
case GIMPLE_OMP_FOR:
expand_omp_for (region, inner_stmt);
break;
case GIMPLE_OMP_SECTIONS:
expand_omp_sections (region);
break;
case GIMPLE_OMP_SECTION:
break;
case GIMPLE_OMP_SINGLE:
expand_omp_single (region);
break;
case GIMPLE_OMP_ORDERED:
{
gomp_ordered *ord_stmt
= as_a <gomp_ordered *> (last_stmt (region->entry));
if (omp_find_clause (gimple_omp_ordered_clauses (ord_stmt),
OMP_CLAUSE_DEPEND))
{
gcc_assert (region->outer
&& region->outer->type == GIMPLE_OMP_FOR);
region->ord_stmt = ord_stmt;
break;
}
}
case GIMPLE_OMP_MASTER:
case GIMPLE_OMP_TASKGROUP:
case GIMPLE_OMP_CRITICAL:
case GIMPLE_OMP_TEAMS:
expand_omp_synch (region);
break;
case GIMPLE_OMP_ATOMIC_LOAD:
expand_omp_atomic (region);
break;
case GIMPLE_OMP_TARGET:
expand_omp_target (region);
break;
default:
gcc_unreachable ();
}
input_location = saved_location;
region = region->next;
}
if (omp_any_child_fn_dumped)
{
if (dump_file)
dump_function_header (dump_file, current_function_decl, dump_flags);
omp_any_child_fn_dumped = false;
}
}
static void
build_omp_regions_1 (basic_block bb, struct omp_region *parent,
bool single_tree)
{
gimple_stmt_iterator gsi;
gimple *stmt;
basic_block son;
gsi = gsi_last_nondebug_bb (bb);
if (!gsi_end_p (gsi) && is_gimple_omp (gsi_stmt (gsi)))
{
struct omp_region *region;
enum gimple_code code;
stmt = gsi_stmt (gsi);
code = gimple_code (stmt);
if (code == GIMPLE_OMP_RETURN)
{
gcc_assert (parent);
region = parent;
region->exit = bb;
parent = parent->outer;
}
else if (code == GIMPLE_OMP_ATOMIC_STORE)
{
gcc_assert (parent);
gcc_assert (parent->type == GIMPLE_OMP_ATOMIC_LOAD);
region = parent;
region->exit = bb;
parent = parent->outer;
}
else if (code == GIMPLE_OMP_CONTINUE)
{
gcc_assert (parent);
parent->cont = bb;
}
else if (code == GIMPLE_OMP_SECTIONS_SWITCH)
{
}
else
{
region = new_omp_region (bb, code, parent);
if (code == GIMPLE_OMP_TARGET)
{
switch (gimple_omp_target_kind (stmt))
{
case GF_OMP_TARGET_KIND_REGION:
case GF_OMP_TARGET_KIND_DATA:
case GF_OMP_TARGET_KIND_OACC_PARALLEL:
case GF_OMP_TARGET_KIND_OACC_KERNELS:
case GF_OMP_TARGET_KIND_OACC_DATA:
case GF_OMP_TARGET_KIND_OACC_HOST_DATA:
break;
case GF_OMP_TARGET_KIND_UPDATE:
case GF_OMP_TARGET_KIND_ENTER_DATA:
case GF_OMP_TARGET_KIND_EXIT_DATA:
case GF_OMP_TARGET_KIND_OACC_UPDATE:
case GF_OMP_TARGET_KIND_OACC_ENTER_EXIT_DATA:
case GF_OMP_TARGET_KIND_OACC_DECLARE:
region = NULL;
break;
default:
gcc_unreachable ();
}
}
else if (code == GIMPLE_OMP_ORDERED
&& omp_find_clause (gimple_omp_ordered_clauses
(as_a <gomp_ordered *> (stmt)),
OMP_CLAUSE_DEPEND))
region = NULL;
if (region)
parent = region;
}
}
if (single_tree && !parent)
return;
for (son = first_dom_son (CDI_DOMINATORS, bb);
son;
son = next_dom_son (CDI_DOMINATORS, son))
build_omp_regions_1 (son, parent, single_tree);
}
static void
build_omp_regions_root (basic_block root)
{
gcc_assert (root_omp_region == NULL);
build_omp_regions_1 (root, NULL, true);
gcc_assert (root_omp_region != NULL);
}
void
omp_expand_local (basic_block head)
{
build_omp_regions_root (head);
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "\nOMP region tree\n\n");
dump_omp_region (dump_file, root_omp_region, 0);
fprintf (dump_file, "\n");
}
remove_exit_barriers (root_omp_region);
expand_omp (root_omp_region);
omp_free_regions ();
}
static void
build_omp_regions (void)
{
gcc_assert (root_omp_region == NULL);
calculate_dominance_info (CDI_DOMINATORS);
build_omp_regions_1 (ENTRY_BLOCK_PTR_FOR_FN (cfun), NULL, false);
}
static unsigned int
execute_expand_omp (void)
{
build_omp_regions ();
if (!root_omp_region)
return 0;
if (dump_file)
{
fprintf (dump_file, "\nOMP region tree\n\n");
dump_omp_region (dump_file, root_omp_region, 0);
fprintf (dump_file, "\n");
}
remove_exit_barriers (root_omp_region);
expand_omp (root_omp_region);
if (flag_checking && !loops_state_satisfies_p (LOOPS_NEED_FIXUP))
verify_loop_structure ();
cleanup_tree_cfg ();
omp_free_regions ();
return 0;
}
namespace {
const pass_data pass_data_expand_omp =
{
GIMPLE_PASS, 
"ompexp", 
OPTGROUP_OMP, 
TV_NONE, 
PROP_gimple_any, 
PROP_gimple_eomp, 
0, 
0, 
0, 
};
class pass_expand_omp : public gimple_opt_pass
{
public:
pass_expand_omp (gcc::context *ctxt)
: gimple_opt_pass (pass_data_expand_omp, ctxt)
{}
virtual unsigned int execute (function *)
{
bool gate = ((flag_openacc != 0 || flag_openmp != 0
|| flag_openmp_simd != 0)
&& !seen_error ());
if (!gate)
return 0;
return execute_expand_omp ();
}
}; 
} 
gimple_opt_pass *
make_pass_expand_omp (gcc::context *ctxt)
{
return new pass_expand_omp (ctxt);
}
namespace {
const pass_data pass_data_expand_omp_ssa =
{
GIMPLE_PASS, 
"ompexpssa", 
OPTGROUP_OMP, 
TV_NONE, 
PROP_cfg | PROP_ssa, 
PROP_gimple_eomp, 
0, 
0, 
TODO_cleanup_cfg | TODO_rebuild_alias, 
};
class pass_expand_omp_ssa : public gimple_opt_pass
{
public:
pass_expand_omp_ssa (gcc::context *ctxt)
: gimple_opt_pass (pass_data_expand_omp_ssa, ctxt)
{}
virtual bool gate (function *fun)
{
return !(fun->curr_properties & PROP_gimple_eomp);
}
virtual unsigned int execute (function *) { return execute_expand_omp (); }
opt_pass * clone () { return new pass_expand_omp_ssa (m_ctxt); }
}; 
} 
gimple_opt_pass *
make_pass_expand_omp_ssa (gcc::context *ctxt)
{
return new pass_expand_omp_ssa (ctxt);
}
bool
omp_make_gimple_edges (basic_block bb, struct omp_region **region,
int *region_idx)
{
gimple *last = last_stmt (bb);
enum gimple_code code = gimple_code (last);
struct omp_region *cur_region = *region;
bool fallthru = false;
switch (code)
{
case GIMPLE_OMP_PARALLEL:
case GIMPLE_OMP_TASK:
case GIMPLE_OMP_FOR:
case GIMPLE_OMP_SINGLE:
case GIMPLE_OMP_TEAMS:
case GIMPLE_OMP_MASTER:
case GIMPLE_OMP_TASKGROUP:
case GIMPLE_OMP_CRITICAL:
case GIMPLE_OMP_SECTION:
case GIMPLE_OMP_GRID_BODY:
cur_region = new_omp_region (bb, code, cur_region);
fallthru = true;
break;
case GIMPLE_OMP_ORDERED:
cur_region = new_omp_region (bb, code, cur_region);
fallthru = true;
if (omp_find_clause (gimple_omp_ordered_clauses
(as_a <gomp_ordered *> (last)),
OMP_CLAUSE_DEPEND))
cur_region = cur_region->outer;
break;
case GIMPLE_OMP_TARGET:
cur_region = new_omp_region (bb, code, cur_region);
fallthru = true;
switch (gimple_omp_target_kind (last))
{
case GF_OMP_TARGET_KIND_REGION:
case GF_OMP_TARGET_KIND_DATA:
case GF_OMP_TARGET_KIND_OACC_PARALLEL:
case GF_OMP_TARGET_KIND_OACC_KERNELS:
case GF_OMP_TARGET_KIND_OACC_DATA:
case GF_OMP_TARGET_KIND_OACC_HOST_DATA:
break;
case GF_OMP_TARGET_KIND_UPDATE:
case GF_OMP_TARGET_KIND_ENTER_DATA:
case GF_OMP_TARGET_KIND_EXIT_DATA:
case GF_OMP_TARGET_KIND_OACC_UPDATE:
case GF_OMP_TARGET_KIND_OACC_ENTER_EXIT_DATA:
case GF_OMP_TARGET_KIND_OACC_DECLARE:
cur_region = cur_region->outer;
break;
default:
gcc_unreachable ();
}
break;
case GIMPLE_OMP_SECTIONS:
cur_region = new_omp_region (bb, code, cur_region);
fallthru = true;
break;
case GIMPLE_OMP_SECTIONS_SWITCH:
fallthru = false;
break;
case GIMPLE_OMP_ATOMIC_LOAD:
case GIMPLE_OMP_ATOMIC_STORE:
fallthru = true;
break;
case GIMPLE_OMP_RETURN:
cur_region->exit = bb;
if (cur_region->type == GIMPLE_OMP_TASK)
make_edge (cur_region->entry, bb, EDGE_ABNORMAL);
fallthru = cur_region->type != GIMPLE_OMP_SECTION;
cur_region = cur_region->outer;
break;
case GIMPLE_OMP_CONTINUE:
cur_region->cont = bb;
switch (cur_region->type)
{
case GIMPLE_OMP_FOR:
single_succ_edge (cur_region->entry)->flags |= EDGE_ABNORMAL;
make_edge (bb, single_succ (cur_region->entry),
EDGE_ABNORMAL);
make_edge (cur_region->entry, bb->next_bb, EDGE_ABNORMAL);
make_edge (bb, bb->next_bb, EDGE_FALLTHRU | EDGE_ABNORMAL);
fallthru = false;
break;
case GIMPLE_OMP_SECTIONS:
{
basic_block switch_bb = single_succ (cur_region->entry);
struct omp_region *i;
for (i = cur_region->inner; i ; i = i->next)
{
gcc_assert (i->type == GIMPLE_OMP_SECTION);
make_edge (switch_bb, i->entry, 0);
make_edge (i->exit, bb, EDGE_FALLTHRU);
}
make_edge (bb, switch_bb, 0);
make_edge (switch_bb, bb->next_bb, 0);
fallthru = false;
}
break;
case GIMPLE_OMP_TASK:
fallthru = true;
break;
default:
gcc_unreachable ();
}
break;
default:
gcc_unreachable ();
}
if (*region != cur_region)
{
*region = cur_region;
if (cur_region)
*region_idx = cur_region->entry->index;
else
*region_idx = 0;
}
return fallthru;
}
#include "gt-omp-expand.h"
