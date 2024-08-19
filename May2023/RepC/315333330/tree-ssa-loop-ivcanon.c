#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "tree.h"
#include "gimple.h"
#include "cfghooks.h"
#include "tree-pass.h"
#include "ssa.h"
#include "cgraph.h"
#include "gimple-pretty-print.h"
#include "fold-const.h"
#include "profile.h"
#include "gimple-fold.h"
#include "tree-eh.h"
#include "gimple-iterator.h"
#include "tree-cfg.h"
#include "tree-ssa-loop-manip.h"
#include "tree-ssa-loop-niter.h"
#include "tree-ssa-loop.h"
#include "tree-into-ssa.h"
#include "cfgloop.h"
#include "tree-chrec.h"
#include "tree-scalar-evolution.h"
#include "params.h"
#include "tree-inline.h"
#include "tree-cfgcleanup.h"
#include "builtins.h"
enum unroll_level
{
UL_SINGLE_ITER,	
UL_NO_GROWTH,		
UL_ALL		
};
void
create_canonical_iv (struct loop *loop, edge exit, tree niter,
tree *var_before = NULL, tree *var_after = NULL)
{
edge in;
tree type, var;
gcond *cond;
gimple_stmt_iterator incr_at;
enum tree_code cmp;
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "Added canonical iv to loop %d, ", loop->num);
print_generic_expr (dump_file, niter, TDF_SLIM);
fprintf (dump_file, " iterations.\n");
}
cond = as_a <gcond *> (last_stmt (exit->src));
in = EDGE_SUCC (exit->src, 0);
if (in == exit)
in = EDGE_SUCC (exit->src, 1);
type = TREE_TYPE (niter);
niter = fold_build2 (PLUS_EXPR, type,
niter,
build_int_cst (type, 1));
incr_at = gsi_last_bb (in->src);
create_iv (niter,
build_int_cst (type, -1),
NULL_TREE, loop,
&incr_at, false, var_before, &var);
if (var_after)
*var_after = var;
cmp = (exit->flags & EDGE_TRUE_VALUE) ? EQ_EXPR : NE_EXPR;
gimple_cond_set_code (cond, cmp);
gimple_cond_set_lhs (cond, var);
gimple_cond_set_rhs (cond, build_int_cst (type, 0));
update_stmt (cond);
}
struct loop_size
{
int overall;
int eliminated_by_peeling;
int last_iteration;
int last_iteration_eliminated_by_peeling;
bool constant_iv;
int num_pure_calls_on_hot_path;
int num_non_pure_calls_on_hot_path;
int non_call_stmts_on_hot_path;
int num_branches_on_hot_path;
};
static bool
constant_after_peeling (tree op, gimple *stmt, struct loop *loop)
{
if (is_gimple_min_invariant (op))
return true;
if (TREE_CODE (op) != SSA_NAME)
{
tree base = op;
while (handled_component_p (base))
base = TREE_OPERAND (base, 0);
if ((DECL_P (base)
&& ctor_for_folding (base) != error_mark_node)
|| CONSTANT_CLASS_P (base))
{
base = op;
while (handled_component_p (base))
{
if (TREE_CODE (base) == ARRAY_REF
&& !constant_after_peeling (TREE_OPERAND (base, 1), stmt, loop))
return false;
base = TREE_OPERAND (base, 0);
}
return true;
}
return false;
}
if (loop_containing_stmt (stmt) != loop)
return false;
tree ev = analyze_scalar_evolution (loop, op);
if (chrec_contains_undetermined (ev)
|| chrec_contains_symbols (ev))
return false;
return true;
}
static bool
tree_estimate_loop_size (struct loop *loop, edge exit, edge edge_to_cancel,
struct loop_size *size, int upper_bound)
{
basic_block *body = get_loop_body (loop);
gimple_stmt_iterator gsi;
unsigned int i;
bool after_exit;
vec<basic_block> path = get_loop_hot_path (loop);
size->overall = 0;
size->eliminated_by_peeling = 0;
size->last_iteration = 0;
size->last_iteration_eliminated_by_peeling = 0;
size->num_pure_calls_on_hot_path = 0;
size->num_non_pure_calls_on_hot_path = 0;
size->non_call_stmts_on_hot_path = 0;
size->num_branches_on_hot_path = 0;
size->constant_iv = 0;
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "Estimating sizes for loop %i\n", loop->num);
for (i = 0; i < loop->num_nodes; i++)
{
if (edge_to_cancel && body[i] != edge_to_cancel->src
&& dominated_by_p (CDI_DOMINATORS, body[i], edge_to_cancel->src))
after_exit = true;
else
after_exit = false;
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, " BB: %i, after_exit: %i\n", body[i]->index,
after_exit);
for (gsi = gsi_start_bb (body[i]); !gsi_end_p (gsi); gsi_next (&gsi))
{
gimple *stmt = gsi_stmt (gsi);
int num = estimate_num_insns (stmt, &eni_size_weights);
bool likely_eliminated = false;
bool likely_eliminated_last = false;
bool likely_eliminated_peeled = false;
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "  size: %3i ", num);
print_gimple_stmt (dump_file, gsi_stmt (gsi), 0);
}
if (!gimple_has_side_effects (stmt))
{
if (exit && body[i] == exit->src
&& stmt == last_stmt (exit->src))
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "   Exit condition will be eliminated "
"in peeled copies.\n");
likely_eliminated_peeled = true;
}
if (edge_to_cancel && body[i] == edge_to_cancel->src
&& stmt == last_stmt (edge_to_cancel->src))
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "   Exit condition will be eliminated "
"in last copy.\n");
likely_eliminated_last = true;
}
if (gimple_code (stmt) == GIMPLE_ASSIGN
&& constant_after_peeling (gimple_assign_lhs (stmt), stmt, loop))
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "   Induction variable computation will"
" be folded away.\n");
likely_eliminated = true;
}
else if (gimple_code (stmt) == GIMPLE_ASSIGN
&& TREE_CODE (gimple_assign_lhs (stmt)) == SSA_NAME
&& constant_after_peeling (gimple_assign_rhs1 (stmt),
stmt, loop)
&& (gimple_assign_rhs_class (stmt) != GIMPLE_BINARY_RHS
|| constant_after_peeling (gimple_assign_rhs2 (stmt),
stmt, loop)))
{
size->constant_iv = true;
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file,
"   Constant expression will be folded away.\n");
likely_eliminated = true;
}
else if ((gimple_code (stmt) == GIMPLE_COND
&& constant_after_peeling (gimple_cond_lhs (stmt), stmt,
loop)
&& constant_after_peeling (gimple_cond_rhs (stmt), stmt,
loop)
&& (! is_gimple_min_invariant (gimple_cond_lhs (stmt))
|| ! is_gimple_min_invariant
(gimple_cond_rhs (stmt))))
|| (gimple_code (stmt) == GIMPLE_SWITCH
&& constant_after_peeling (gimple_switch_index (
as_a <gswitch *>
(stmt)),
stmt, loop)
&& ! is_gimple_min_invariant
(gimple_switch_index
(as_a <gswitch *> (stmt)))))
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "   Constant conditional.\n");
likely_eliminated = true;
}
}
size->overall += num;
if (likely_eliminated || likely_eliminated_peeled)
size->eliminated_by_peeling += num;
if (!after_exit)
{
size->last_iteration += num;
if (likely_eliminated || likely_eliminated_last)
size->last_iteration_eliminated_by_peeling += num;
}
if ((size->overall * 3 / 2 - size->eliminated_by_peeling
- size->last_iteration_eliminated_by_peeling) > upper_bound)
{
free (body);
path.release ();
return true;
}
}
}
while (path.length ())
{
basic_block bb = path.pop ();
for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi); gsi_next (&gsi))
{
gimple *stmt = gsi_stmt (gsi);
if (gimple_code (stmt) == GIMPLE_CALL
&& !gimple_inexpensive_call_p (as_a <gcall *>  (stmt)))
{
int flags = gimple_call_flags (stmt);
if (flags & (ECF_PURE | ECF_CONST))
size->num_pure_calls_on_hot_path++;
else
size->num_non_pure_calls_on_hot_path++;
size->num_branches_on_hot_path ++;
}
else if (gimple_code (stmt) != GIMPLE_DEBUG)
size->non_call_stmts_on_hot_path++;
if (((gimple_code (stmt) == GIMPLE_COND
&& (!constant_after_peeling (gimple_cond_lhs (stmt), stmt, loop)
|| !constant_after_peeling (gimple_cond_rhs (stmt), stmt,
loop)))
|| (gimple_code (stmt) == GIMPLE_SWITCH
&& !constant_after_peeling (gimple_switch_index (
as_a <gswitch *> (stmt)),
stmt, loop)))
&& (!exit || bb != exit->src))
size->num_branches_on_hot_path++;
}
}
path.release ();
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "size: %i-%i, last_iteration: %i-%i\n", size->overall,
size->eliminated_by_peeling, size->last_iteration,
size->last_iteration_eliminated_by_peeling);
free (body);
return false;
}
static unsigned HOST_WIDE_INT
estimated_unrolled_size (struct loop_size *size,
unsigned HOST_WIDE_INT nunroll)
{
HOST_WIDE_INT unr_insns = ((nunroll)
* (HOST_WIDE_INT) (size->overall
- size->eliminated_by_peeling));
if (!nunroll)
unr_insns = 0;
unr_insns += size->last_iteration - size->last_iteration_eliminated_by_peeling;
unr_insns = unr_insns * 2 / 3;
if (unr_insns <= 0)
unr_insns = 1;
return unr_insns;
}
static edge
loop_edge_to_cancel (struct loop *loop)
{
vec<edge> exits;
unsigned i;
edge edge_to_cancel;
gimple_stmt_iterator gsi;
if (EDGE_COUNT (loop->latch->preds) > 1)
return NULL;
exits = get_loop_exit_edges (loop);
FOR_EACH_VEC_ELT (exits, i, edge_to_cancel)
{
if (EDGE_COUNT (edge_to_cancel->src->succs) != 2)
continue;
if (EDGE_SUCC (edge_to_cancel->src, 0) == edge_to_cancel)
edge_to_cancel = EDGE_SUCC (edge_to_cancel->src, 1);
else
edge_to_cancel = EDGE_SUCC (edge_to_cancel->src, 0);
if (!(edge_to_cancel->flags & (EDGE_TRUE_VALUE | EDGE_FALSE_VALUE)))
continue;
gcc_assert (edge_to_cancel->dest != loop->header);
if (edge_to_cancel->dest != loop->latch)
continue;
exits.release ();
for (gsi = gsi_start_bb (loop->latch); !gsi_end_p (gsi); gsi_next (&gsi))
if (gimple_has_side_effects (gsi_stmt (gsi)))
return NULL;
return edge_to_cancel;
}
exits.release ();
return NULL;
}
static bool
remove_exits_and_undefined_stmts (struct loop *loop, unsigned int npeeled)
{
struct nb_iter_bound *elt;
bool changed = false;
for (elt = loop->bounds; elt; elt = elt->next)
{
if (!elt->is_exit
&& wi::ltu_p (elt->bound, npeeled))
{
gimple_stmt_iterator gsi = gsi_for_stmt (elt->stmt);
gcall *stmt = gimple_build_call
(builtin_decl_implicit (BUILT_IN_UNREACHABLE), 0);
gimple_set_location (stmt, gimple_location (elt->stmt));
gsi_insert_before (&gsi, stmt, GSI_NEW_STMT);
split_block (gimple_bb (stmt), stmt);
changed = true;
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "Forced statement unreachable: ");
print_gimple_stmt (dump_file, elt->stmt, 0);
}
}
else if (elt->is_exit
&& wi::leu_p (elt->bound, npeeled))
{
basic_block bb = gimple_bb (elt->stmt);
edge exit_edge = EDGE_SUCC (bb, 0);
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "Forced exit to be taken: ");
print_gimple_stmt (dump_file, elt->stmt, 0);
}
if (!loop_exit_edge_p (loop, exit_edge))
exit_edge = EDGE_SUCC (bb, 1);
exit_edge->probability = profile_probability::always ();
gcc_checking_assert (loop_exit_edge_p (loop, exit_edge));
gcond *cond_stmt = as_a <gcond *> (elt->stmt);
if (exit_edge->flags & EDGE_TRUE_VALUE)
gimple_cond_make_true (cond_stmt);
else
gimple_cond_make_false (cond_stmt);
update_stmt (cond_stmt);
changed = true;
}
}
return changed;
}
static bool
remove_redundant_iv_tests (struct loop *loop)
{
struct nb_iter_bound *elt;
bool changed = false;
if (!loop->any_upper_bound)
return false;
for (elt = loop->bounds; elt; elt = elt->next)
{
if (elt->is_exit && loop->any_upper_bound
&& wi::ltu_p (loop->nb_iterations_upper_bound, elt->bound))
{
basic_block bb = gimple_bb (elt->stmt);
edge exit_edge = EDGE_SUCC (bb, 0);
struct tree_niter_desc niter;
if (!loop_exit_edge_p (loop, exit_edge))
exit_edge = EDGE_SUCC (bb, 1);
if (!number_of_iterations_exit (loop, exit_edge,
&niter, false, false)
|| !integer_onep (niter.assumptions)
|| !integer_zerop (niter.may_be_zero)
|| !niter.niter
|| TREE_CODE (niter.niter) != INTEGER_CST
|| !wi::ltu_p (loop->nb_iterations_upper_bound,
wi::to_widest (niter.niter)))
continue;
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "Removed pointless exit: ");
print_gimple_stmt (dump_file, elt->stmt, 0);
}
gcond *cond_stmt = as_a <gcond *> (elt->stmt);
if (exit_edge->flags & EDGE_TRUE_VALUE)
gimple_cond_make_false (cond_stmt);
else
gimple_cond_make_true (cond_stmt);
update_stmt (cond_stmt);
changed = true;
}
}
return changed;
}
static vec<loop_p> loops_to_unloop;
static vec<int> loops_to_unloop_nunroll;
static vec<edge> edges_to_remove;
static bitmap peeled_loops;
static void
unloop_loops (bitmap loop_closed_ssa_invalidated,
bool *irred_invalidated)
{
while (loops_to_unloop.length ())
{
struct loop *loop = loops_to_unloop.pop ();
int n_unroll = loops_to_unloop_nunroll.pop ();
basic_block latch = loop->latch;
edge latch_edge = loop_latch_edge (loop);
int flags = latch_edge->flags;
location_t locus = latch_edge->goto_locus;
gcall *stmt;
gimple_stmt_iterator gsi;
remove_exits_and_undefined_stmts (loop, n_unroll);
unloop (loop, irred_invalidated, loop_closed_ssa_invalidated);
stmt = gimple_build_call (builtin_decl_implicit (BUILT_IN_UNREACHABLE), 0);
latch_edge = make_edge (latch, create_basic_block (NULL, NULL, latch), flags);
latch_edge->probability = profile_probability::never ();
latch_edge->flags |= flags;
latch_edge->goto_locus = locus;
add_bb_to_loop (latch_edge->dest, current_loops->tree_root);
latch_edge->dest->count = profile_count::zero ();
set_immediate_dominator (CDI_DOMINATORS, latch_edge->dest, latch_edge->src);
gsi = gsi_start_bb (latch_edge->dest);
gsi_insert_after (&gsi, stmt, GSI_NEW_STMT);
}
loops_to_unloop.release ();
loops_to_unloop_nunroll.release ();
unsigned i;
edge e;
auto_vec<int, 20> src_bbs;
src_bbs.reserve_exact (edges_to_remove.length ());
FOR_EACH_VEC_ELT (edges_to_remove, i, e)
src_bbs.quick_push (e->src->index);
FOR_EACH_VEC_ELT (edges_to_remove, i, e)
if (BASIC_BLOCK_FOR_FN (cfun, src_bbs[i]))
{
bool ok = remove_path (e, irred_invalidated,
loop_closed_ssa_invalidated);
gcc_assert (ok);
}
edges_to_remove.release ();
}
static bool
try_unroll_loop_completely (struct loop *loop,
edge exit, tree niter, bool may_be_zero,
enum unroll_level ul,
HOST_WIDE_INT maxiter,
location_t locus, bool allow_peel)
{
unsigned HOST_WIDE_INT n_unroll = 0;
bool n_unroll_found = false;
edge edge_to_cancel = NULL;
if (tree_fits_uhwi_p (niter))
{
n_unroll = tree_to_uhwi (niter);
n_unroll_found = true;
edge_to_cancel = EDGE_SUCC (exit->src, 0);
if (edge_to_cancel == exit)
edge_to_cancel = EDGE_SUCC (exit->src, 1);
}
else
exit = NULL;
if ((allow_peel || maxiter == 0 || ul == UL_NO_GROWTH)
&& maxiter >= 0
&& (!n_unroll_found || (unsigned HOST_WIDE_INT)maxiter < n_unroll))
{
n_unroll = maxiter;
n_unroll_found = true;
edge_to_cancel = NULL;
}
if (!n_unroll_found)
return false;
if (!loop->unroll
&& n_unroll > (unsigned) PARAM_VALUE (PARAM_MAX_COMPLETELY_PEEL_TIMES))
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "Not unrolling loop %d "
"(--param max-completely-peel-times limit reached).\n",
loop->num);
return false;
}
if (!edge_to_cancel)
edge_to_cancel = loop_edge_to_cancel (loop);
if (n_unroll)
{
if (ul == UL_SINGLE_ITER)
return false;
if (loop->unroll)
{
if (n_unroll > (unsigned)loop->unroll)
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file,
"Not unrolling loop %d: "
"user didn't want it unrolled completely.\n",
loop->num);
return false;
}
}
else
{
struct loop_size size;
bool remove_exit = (exit && niter
&& TREE_CODE (niter) == INTEGER_CST
&& wi::leu_p (n_unroll, wi::to_widest (niter)));
bool large
= tree_estimate_loop_size
(loop, remove_exit ? exit : NULL, edge_to_cancel, &size,
PARAM_VALUE (PARAM_MAX_COMPLETELY_PEELED_INSNS));
if (large)
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "Not unrolling loop %d: it is too large.\n",
loop->num);
return false;
}
unsigned HOST_WIDE_INT ninsns = size.overall;
unsigned HOST_WIDE_INT unr_insns
= estimated_unrolled_size (&size, n_unroll);
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "  Loop size: %d\n", (int) ninsns);
fprintf (dump_file, "  Estimated size after unrolling: %d\n",
(int) unr_insns);
}
if (unr_insns
<= ninsns + (size.constant_iv != false))
;
else if (ul == UL_NO_GROWTH)
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "Not unrolling loop %d: size would grow.\n",
loop->num);
return false;
}
else if (loop->inner)
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "Not unrolling loop %d: "
"it is not innermost and code would grow.\n",
loop->num);
return false;
}
else if (size.num_non_pure_calls_on_hot_path)
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "Not unrolling loop %d: "
"contains call and code would grow.\n",
loop->num);
return false;
}
else if (size.num_pure_calls_on_hot_path
&& (size.non_call_stmts_on_hot_path
<= 3 + size.num_pure_calls_on_hot_path))
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "Not unrolling loop %d: "
"contains just pure calls and code would grow.\n",
loop->num);
return false;
}
else if (size.num_branches_on_hot_path * (int)n_unroll
> PARAM_VALUE (PARAM_MAX_PEEL_BRANCHES))
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "Not unrolling loop %d: "
"number of branches on hot path in the unrolled "
"sequence reaches --param max-peel-branches limit.\n",
loop->num);
return false;
}
else if (unr_insns
> (unsigned) PARAM_VALUE (PARAM_MAX_COMPLETELY_PEELED_INSNS))
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "Not unrolling loop %d: "
"number of insns in the unrolled sequence reaches "
"--param max-completely-peeled-insns limit.\n",
loop->num);
return false;
}
}
initialize_original_copy_tables ();
auto_sbitmap wont_exit (n_unroll + 1);
if (exit && niter
&& TREE_CODE (niter) == INTEGER_CST
&& wi::leu_p (n_unroll, wi::to_widest (niter)))
{
bitmap_ones (wont_exit);
if (wi::eq_p (wi::to_widest (niter), n_unroll)
|| edge_to_cancel)
bitmap_clear_bit (wont_exit, 0);
}
else
{
exit = NULL;
bitmap_clear (wont_exit);
}
if (may_be_zero)
bitmap_clear_bit (wont_exit, 1);
if (!gimple_duplicate_loop_to_header_edge (loop, loop_preheader_edge (loop),
n_unroll, wont_exit,
exit, &edges_to_remove,
DLTHE_FLAG_UPDATE_FREQ
| DLTHE_FLAG_COMPLETTE_PEEL))
{
free_original_copy_tables ();
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "Failed to duplicate the loop\n");
return false;
}
free_original_copy_tables ();
}
if (edge_to_cancel)
{
gcond *cond = as_a <gcond *> (last_stmt (edge_to_cancel->src));
force_edge_cold (edge_to_cancel, true);
if (edge_to_cancel->flags & EDGE_TRUE_VALUE)
gimple_cond_make_false (cond);
else
gimple_cond_make_true (cond);
update_stmt (cond);
}
loops_to_unloop.safe_push (loop);
loops_to_unloop_nunroll.safe_push (n_unroll);
if (dump_enabled_p ())
{
if (!n_unroll)
dump_printf_loc (MSG_OPTIMIZED_LOCATIONS | TDF_DETAILS, locus,
"loop turned into non-loop; it never loops\n");
else
{
dump_printf_loc (MSG_OPTIMIZED_LOCATIONS | TDF_DETAILS, locus,
"loop with %d iterations completely unrolled",
(int) n_unroll);
if (loop->header->count.initialized_p ())
dump_printf (MSG_OPTIMIZED_LOCATIONS | TDF_DETAILS,
" (header execution count %d)",
(int)loop->header->count.to_gcov_type ());
dump_printf (MSG_OPTIMIZED_LOCATIONS | TDF_DETAILS, "\n");
}
}
if (dump_file && (dump_flags & TDF_DETAILS))
{
if (exit)
fprintf (dump_file, "Exit condition of peeled iterations was "
"eliminated.\n");
if (edge_to_cancel)
fprintf (dump_file, "Last iteration exit edge was proved true.\n");
else
fprintf (dump_file, "Latch of last iteration was marked by "
"__builtin_unreachable ().\n");
}
return true;
}
static unsigned HOST_WIDE_INT
estimated_peeled_sequence_size (struct loop_size *size,
unsigned HOST_WIDE_INT npeel)
{
return MAX (npeel * (HOST_WIDE_INT) (size->overall
- size->eliminated_by_peeling), 1);
}
static bool
try_peel_loop (struct loop *loop,
edge exit, tree niter, bool may_be_zero,
HOST_WIDE_INT maxiter)
{
HOST_WIDE_INT npeel;
struct loop_size size;
int peeled_size;
if (!flag_peel_loops
|| PARAM_VALUE (PARAM_MAX_PEEL_TIMES) <= 0
|| !peeled_loops)
return false;
if (bitmap_bit_p (peeled_loops, loop->num))
{
if (dump_file)
fprintf (dump_file, "Not peeling: loop is already peeled\n");
return false;
}
if (loop->unroll)
{
if (dump_file)
fprintf (dump_file, "Not peeling: user didn't want it peeled.\n");
return false;
}
if (loop->inner)
{
if (dump_file)
fprintf (dump_file, "Not peeling: outer loop\n");
return false;
}
if (!optimize_loop_for_speed_p (loop))
{
if (dump_file)
fprintf (dump_file, "Not peeling: cold loop\n");
return false;
}
npeel = estimated_loop_iterations_int (loop);
if (npeel < 0)
npeel = likely_max_loop_iterations_int (loop);
if (npeel < 0)
{
if (dump_file)
fprintf (dump_file, "Not peeling: number of iterations is not "
"estimated\n");
return false;
}
if (maxiter >= 0 && maxiter <= npeel)
{
if (dump_file)
fprintf (dump_file, "Not peeling: upper bound is known so can "
"unroll completely\n");
return false;
}
if (npeel > PARAM_VALUE (PARAM_MAX_PEEL_TIMES) - 1)
{
if (dump_file)
fprintf (dump_file, "Not peeling: rolls too much "
"(%i + 1 > --param max-peel-times)\n", (int) npeel);
return false;
}
npeel++;
tree_estimate_loop_size (loop, exit, NULL, &size,
PARAM_VALUE (PARAM_MAX_PEELED_INSNS));
if ((peeled_size = estimated_peeled_sequence_size (&size, (int) npeel))
> PARAM_VALUE (PARAM_MAX_PEELED_INSNS))
{
if (dump_file)
fprintf (dump_file, "Not peeling: peeled sequence size is too large "
"(%i insns > --param max-peel-insns)", peeled_size);
return false;
}
initialize_original_copy_tables ();
auto_sbitmap wont_exit (npeel + 1);
if (exit && niter
&& TREE_CODE (niter) == INTEGER_CST
&& wi::leu_p (npeel, wi::to_widest (niter)))
{
bitmap_ones (wont_exit);
bitmap_clear_bit (wont_exit, 0);
}
else
{
exit = NULL;
bitmap_clear (wont_exit);
}
if (may_be_zero)
bitmap_clear_bit (wont_exit, 1);
if (!gimple_duplicate_loop_to_header_edge (loop, loop_preheader_edge (loop),
npeel, wont_exit,
exit, &edges_to_remove,
DLTHE_FLAG_UPDATE_FREQ))
{
free_original_copy_tables ();
return false;
}
free_original_copy_tables ();
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "Peeled loop %d, %i times.\n",
loop->num, (int) npeel);
}
if (loop->any_estimate)
{
if (wi::ltu_p (npeel, loop->nb_iterations_estimate))
loop->nb_iterations_estimate -= npeel;
else
loop->nb_iterations_estimate = 0;
}
if (loop->any_upper_bound)
{
if (wi::ltu_p (npeel, loop->nb_iterations_upper_bound))
loop->nb_iterations_upper_bound -= npeel;
else
loop->nb_iterations_upper_bound = 0;
}
if (loop->any_likely_upper_bound)
{
if (wi::ltu_p (npeel, loop->nb_iterations_likely_upper_bound))
loop->nb_iterations_likely_upper_bound -= npeel;
else
{
loop->any_estimate = true;
loop->nb_iterations_estimate = 0;
loop->nb_iterations_likely_upper_bound = 0;
}
}
profile_count entry_count = profile_count::zero ();
edge e;
edge_iterator ei;
FOR_EACH_EDGE (e, ei, loop->header->preds)
if (e->src != loop->latch)
{
if (e->src->count.initialized_p ())
entry_count = e->src->count + e->src->count;
gcc_assert (!flow_bb_inside_loop_p (loop, e->src));
}
profile_probability p = profile_probability::very_unlikely ();
p = entry_count.probability_in (loop->header->count);
scale_loop_profile (loop, p, 0);
bitmap_set_bit (peeled_loops, loop->num);
return true;
}
static bool
canonicalize_loop_induction_variables (struct loop *loop,
bool create_iv, enum unroll_level ul,
bool try_eval, bool allow_peel)
{
edge exit = NULL;
tree niter;
HOST_WIDE_INT maxiter;
bool modified = false;
location_t locus = UNKNOWN_LOCATION;
struct tree_niter_desc niter_desc;
bool may_be_zero = false;
exit = single_exit (loop);
niter = chrec_dont_know;
if (exit && number_of_iterations_exit (loop, exit, &niter_desc, false))
{
niter = niter_desc.niter;
may_be_zero
= niter_desc.may_be_zero && !integer_zerop (niter_desc.may_be_zero);
}
if (TREE_CODE (niter) == INTEGER_CST)
locus = gimple_location (last_stmt (exit->src));
else
{
if (may_be_zero)
{
if (COMPARISON_CLASS_P (niter_desc.may_be_zero))
niter = fold_build3 (COND_EXPR, TREE_TYPE (niter),
niter_desc.may_be_zero,
build_int_cst (TREE_TYPE (niter), 0), niter);
else
niter = chrec_dont_know;
may_be_zero = false;
}
if (!exit)
niter = find_loop_niter (loop, &exit);
if (try_eval
&& (chrec_contains_undetermined (niter)
|| TREE_CODE (niter) != INTEGER_CST))
niter = find_loop_niter_by_eval (loop, &exit);
if (exit)
locus = gimple_location (last_stmt (exit->src));
if (TREE_CODE (niter) != INTEGER_CST)
exit = NULL;
}
if (niter && TREE_CODE (niter) == INTEGER_CST)
{
record_niter_bound (loop, wi::to_widest (niter),
exit == single_likely_exit (loop), true);
}
maxiter = max_loop_iterations_int (loop);
if (dump_file && (dump_flags & TDF_DETAILS)
&& TREE_CODE (niter) == INTEGER_CST)
{
fprintf (dump_file, "Loop %d iterates ", loop->num);
print_generic_expr (dump_file, niter, TDF_SLIM);
fprintf (dump_file, " times.\n");
}
if (dump_file && (dump_flags & TDF_DETAILS)
&& maxiter >= 0)
{
fprintf (dump_file, "Loop %d iterates at most %i times.\n", loop->num,
(int)maxiter);
}
if (dump_file && (dump_flags & TDF_DETAILS)
&& likely_max_loop_iterations_int (loop) >= 0)
{
fprintf (dump_file, "Loop %d likely iterates at most %i times.\n",
loop->num, (int)likely_max_loop_iterations_int (loop));
}
modified |= remove_redundant_iv_tests (loop);
if (try_unroll_loop_completely (loop, exit, niter, may_be_zero, ul,
maxiter, locus, allow_peel))
return true;
if (create_iv
&& niter && !chrec_contains_undetermined (niter)
&& exit && just_once_each_iteration_p (loop, exit->src))
{
tree iv_niter = niter;
if (may_be_zero)
{
if (COMPARISON_CLASS_P (niter_desc.may_be_zero))
iv_niter = fold_build3 (COND_EXPR, TREE_TYPE (iv_niter),
niter_desc.may_be_zero,
build_int_cst (TREE_TYPE (iv_niter), 0),
iv_niter);
else
iv_niter = NULL_TREE;
}
if (iv_niter)
create_canonical_iv (loop, exit, iv_niter);
}
if (ul == UL_ALL)
modified |= try_peel_loop (loop, exit, niter, may_be_zero, maxiter);
return modified;
}
unsigned int
canonicalize_induction_variables (void)
{
struct loop *loop;
bool changed = false;
bool irred_invalidated = false;
bitmap loop_closed_ssa_invalidated = BITMAP_ALLOC (NULL);
estimate_numbers_of_iterations (cfun);
FOR_EACH_LOOP (loop, LI_FROM_INNERMOST)
{
changed |= canonicalize_loop_induction_variables (loop,
true, UL_SINGLE_ITER,
true, false);
}
gcc_assert (!need_ssa_update_p (cfun));
unloop_loops (loop_closed_ssa_invalidated, &irred_invalidated);
if (irred_invalidated
&& loops_state_satisfies_p (LOOPS_HAVE_MARKED_IRREDUCIBLE_REGIONS))
mark_irreducible_loops ();
free_numbers_of_iterations_estimates (cfun);
scev_reset ();
if (!bitmap_empty_p (loop_closed_ssa_invalidated))
{
gcc_checking_assert (loops_state_satisfies_p (LOOP_CLOSED_SSA));
rewrite_into_loop_closed_ssa (NULL, TODO_update_ssa);
}
BITMAP_FREE (loop_closed_ssa_invalidated);
if (changed)
return TODO_cleanup_cfg;
return 0;
}
static void
propagate_constants_for_unrolling (basic_block bb)
{
for (gphi_iterator gsi = gsi_start_phis (bb); !gsi_end_p (gsi); )
{
gphi *phi = gsi.phi ();
tree result = gimple_phi_result (phi);
tree arg = gimple_phi_arg_def (phi, 0);
if (! SSA_NAME_OCCURS_IN_ABNORMAL_PHI (result)
&& gimple_phi_num_args (phi) == 1
&& CONSTANT_CLASS_P (arg))
{
replace_uses_by (result, arg);
gsi_remove (&gsi, true);
release_ssa_name (result);
}
else
gsi_next (&gsi);
}
for (gimple_stmt_iterator gsi = gsi_start_bb (bb); !gsi_end_p (gsi); )
{
gimple *stmt = gsi_stmt (gsi);
tree lhs;
if (is_gimple_assign (stmt)
&& TREE_CODE_CLASS (gimple_assign_rhs_code (stmt)) == tcc_constant
&& (lhs = gimple_assign_lhs (stmt), TREE_CODE (lhs) == SSA_NAME)
&& !SSA_NAME_OCCURS_IN_ABNORMAL_PHI (lhs))
{
replace_uses_by (lhs, gimple_assign_rhs1 (stmt));
gsi_remove (&gsi, true);
release_ssa_name (lhs);
}
else
gsi_next (&gsi);
}
}
static bool
tree_unroll_loops_completely_1 (bool may_increase_size, bool unroll_outer,
bitmap father_bbs, struct loop *loop)
{
struct loop *loop_father;
bool changed = false;
struct loop *inner;
enum unroll_level ul;
unsigned num = number_of_loops (cfun);
for (inner = loop->inner; inner != NULL; inner = inner->next)
if ((unsigned) inner->num < num)
changed |= tree_unroll_loops_completely_1 (may_increase_size,
unroll_outer, father_bbs,
inner);
if (changed)
return true;
if (loop->force_vectorize)
return false;
loop_father = loop_outer (loop);
if (!loop_father)
return false;
if (loop->unroll > 1)
ul = UL_ALL;
else if (may_increase_size && optimize_loop_nest_for_speed_p (loop)
&& (unroll_outer || loop_outer (loop_father)))
ul = UL_ALL;
else
ul = UL_NO_GROWTH;
if (canonicalize_loop_induction_variables
(loop, false, ul, !flag_tree_loop_ivcanon, unroll_outer))
{
if (loop_outer (loop_father))
bitmap_set_bit (father_bbs, loop_father->header->index);
return true;
}
return false;
}
static unsigned int
tree_unroll_loops_completely (bool may_increase_size, bool unroll_outer)
{
bitmap father_bbs = BITMAP_ALLOC (NULL);
bool changed;
int iteration = 0;
bool irred_invalidated = false;
estimate_numbers_of_iterations (cfun);
do
{
changed = false;
bitmap loop_closed_ssa_invalidated = NULL;
if (loops_state_satisfies_p (LOOP_CLOSED_SSA))
loop_closed_ssa_invalidated = BITMAP_ALLOC (NULL);
free_numbers_of_iterations_estimates (cfun);
estimate_numbers_of_iterations (cfun);
changed = tree_unroll_loops_completely_1 (may_increase_size,
unroll_outer, father_bbs,
current_loops->tree_root);
if (changed)
{
unsigned i;
unloop_loops (loop_closed_ssa_invalidated, &irred_invalidated);
if (loop_closed_ssa_invalidated
&& !bitmap_empty_p (loop_closed_ssa_invalidated))
rewrite_into_loop_closed_ssa (loop_closed_ssa_invalidated,
TODO_update_ssa);
else
update_ssa (TODO_update_ssa);
bitmap_iterator bi;
bitmap fathers = BITMAP_ALLOC (NULL);
EXECUTE_IF_SET_IN_BITMAP (father_bbs, 0, i, bi)
{
basic_block unrolled_loop_bb = BASIC_BLOCK_FOR_FN (cfun, i);
if (! unrolled_loop_bb)
continue;
if (loop_outer (unrolled_loop_bb->loop_father))
bitmap_set_bit (fathers,
unrolled_loop_bb->loop_father->num);
}
bitmap_clear (father_bbs);
EXECUTE_IF_SET_IN_BITMAP (fathers, 0, i, bi)
{
loop_p father = get_loop (cfun, i);
basic_block *body = get_loop_body_in_dom_order (father);
for (unsigned j = 0; j < father->num_nodes; j++)
propagate_constants_for_unrolling (body[j]);
free (body);
}
BITMAP_FREE (fathers);
if (cleanup_tree_cfg ())
update_ssa (TODO_update_ssa_only_virtuals);
scev_reset ();
if (flag_checking && loops_state_satisfies_p (LOOP_CLOSED_SSA))
verify_loop_closed_ssa (true);
}
if (loop_closed_ssa_invalidated)
BITMAP_FREE (loop_closed_ssa_invalidated);
}
while (changed
&& ++iteration <= PARAM_VALUE (PARAM_MAX_UNROLL_ITERATIONS));
BITMAP_FREE (father_bbs);
if (irred_invalidated
&& loops_state_satisfies_p (LOOPS_HAVE_MARKED_IRREDUCIBLE_REGIONS))
mark_irreducible_loops ();
return 0;
}
namespace {
const pass_data pass_data_iv_canon =
{
GIMPLE_PASS, 
"ivcanon", 
OPTGROUP_LOOP, 
TV_TREE_LOOP_IVCANON, 
( PROP_cfg | PROP_ssa ), 
0, 
0, 
0, 
0, 
};
class pass_iv_canon : public gimple_opt_pass
{
public:
pass_iv_canon (gcc::context *ctxt)
: gimple_opt_pass (pass_data_iv_canon, ctxt)
{}
virtual bool gate (function *) { return flag_tree_loop_ivcanon != 0; }
virtual unsigned int execute (function *fun);
}; 
unsigned int
pass_iv_canon::execute (function *fun)
{
if (number_of_loops (fun) <= 1)
return 0;
return canonicalize_induction_variables ();
}
} 
gimple_opt_pass *
make_pass_iv_canon (gcc::context *ctxt)
{
return new pass_iv_canon (ctxt);
}
namespace {
const pass_data pass_data_complete_unroll =
{
GIMPLE_PASS, 
"cunroll", 
OPTGROUP_LOOP, 
TV_COMPLETE_UNROLL, 
( PROP_cfg | PROP_ssa ), 
0, 
0, 
0, 
0, 
};
class pass_complete_unroll : public gimple_opt_pass
{
public:
pass_complete_unroll (gcc::context *ctxt)
: gimple_opt_pass (pass_data_complete_unroll, ctxt)
{}
virtual unsigned int execute (function *);
}; 
unsigned int
pass_complete_unroll::execute (function *fun)
{
if (number_of_loops (fun) <= 1)
return 0;
if (flag_peel_loops)
peeled_loops = BITMAP_ALLOC (NULL);
unsigned int val = tree_unroll_loops_completely (flag_unroll_loops
|| flag_peel_loops
|| optimize >= 3, true);
if (peeled_loops)
{
BITMAP_FREE (peeled_loops);
peeled_loops = NULL;
}
return val;
}
} 
gimple_opt_pass *
make_pass_complete_unroll (gcc::context *ctxt)
{
return new pass_complete_unroll (ctxt);
}
namespace {
const pass_data pass_data_complete_unrolli =
{
GIMPLE_PASS, 
"cunrolli", 
OPTGROUP_LOOP, 
TV_COMPLETE_UNROLL, 
( PROP_cfg | PROP_ssa ), 
0, 
0, 
0, 
0, 
};
class pass_complete_unrolli : public gimple_opt_pass
{
public:
pass_complete_unrolli (gcc::context *ctxt)
: gimple_opt_pass (pass_data_complete_unrolli, ctxt)
{}
virtual bool gate (function *) { return optimize >= 2; }
virtual unsigned int execute (function *);
}; 
unsigned int
pass_complete_unrolli::execute (function *fun)
{
unsigned ret = 0;
loop_optimizer_init (LOOPS_NORMAL | LOOPS_HAVE_RECORDED_EXITS);
if (number_of_loops (fun) > 1)
{
scev_initialize ();
ret = tree_unroll_loops_completely (optimize >= 3, false);
scev_finalize ();
}
loop_optimizer_finalize ();
return ret;
}
} 
gimple_opt_pass *
make_pass_complete_unrolli (gcc::context *ctxt)
{
return new pass_complete_unrolli (ctxt);
}
