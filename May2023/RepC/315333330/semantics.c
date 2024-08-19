#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "bitmap.h"
#include "cp-tree.h"
#include "stringpool.h"
#include "cgraph.h"
#include "stmt.h"
#include "varasm.h"
#include "stor-layout.h"
#include "c-family/c-objc.h"
#include "tree-inline.h"
#include "intl.h"
#include "tree-iterator.h"
#include "omp-general.h"
#include "convert.h"
#include "stringpool.h"
#include "attribs.h"
#include "gomp-constants.h"
#include "predict.h"
static tree maybe_convert_cond (tree);
static tree finalize_nrv_r (tree *, int *, void *);
static tree capture_decltype (tree);
static hash_map<tree, tree> *omp_private_member_map;
static vec<tree> omp_private_member_vec;
static bool omp_private_member_ignore_next;
struct GTY(()) deferred_access {
vec<deferred_access_check, va_gc> * GTY(()) deferred_access_checks;
enum deferring_kind deferring_access_checks_kind;
};
static GTY(()) vec<deferred_access, va_gc> *deferred_access_stack;
static GTY(()) unsigned deferred_access_no_check;
void
push_deferring_access_checks (deferring_kind deferring)
{
if (deferred_access_no_check || deferring == dk_no_check)
deferred_access_no_check++;
else
{
deferred_access e = {NULL, deferring};
vec_safe_push (deferred_access_stack, e);
}
}
void
reopen_deferring_access_checks (vec<deferred_access_check, va_gc> * checks)
{
push_deferring_access_checks (dk_deferred);
if (!deferred_access_no_check)
deferred_access_stack->last().deferred_access_checks = checks;
}
void
resume_deferring_access_checks (void)
{
if (!deferred_access_no_check)
deferred_access_stack->last().deferring_access_checks_kind = dk_deferred;
}
void
stop_deferring_access_checks (void)
{
if (!deferred_access_no_check)
deferred_access_stack->last().deferring_access_checks_kind = dk_no_deferred;
}
void
pop_deferring_access_checks (void)
{
if (deferred_access_no_check)
deferred_access_no_check--;
else
deferred_access_stack->pop ();
}
vec<deferred_access_check, va_gc> *
get_deferred_access_checks (void)
{
if (deferred_access_no_check)
return NULL;
else
return (deferred_access_stack->last().deferred_access_checks);
}
void
pop_to_parent_deferring_access_checks (void)
{
if (deferred_access_no_check)
deferred_access_no_check--;
else
{
vec<deferred_access_check, va_gc> *checks;
deferred_access *ptr;
checks = (deferred_access_stack->last ().deferred_access_checks);
deferred_access_stack->pop ();
ptr = &deferred_access_stack->last ();
if (ptr->deferring_access_checks_kind == dk_no_deferred)
{
perform_access_checks (checks, tf_warning_or_error);
}
else
{
int i, j;
deferred_access_check *chk, *probe;
FOR_EACH_VEC_SAFE_ELT (checks, i, chk)
{
FOR_EACH_VEC_SAFE_ELT (ptr->deferred_access_checks, j, probe)
{
if (probe->binfo == chk->binfo &&
probe->decl == chk->decl &&
probe->diag_decl == chk->diag_decl)
goto found;
}
vec_safe_push (ptr->deferred_access_checks, *chk);
found:;
}
}
}
}
bool
perform_access_checks (vec<deferred_access_check, va_gc> *checks,
tsubst_flags_t complain)
{
int i;
deferred_access_check *chk;
location_t loc = input_location;
bool ok = true;
if (!checks)
return true;
FOR_EACH_VEC_SAFE_ELT (checks, i, chk)
{
input_location = chk->loc;
ok &= enforce_access (chk->binfo, chk->decl, chk->diag_decl, complain);
}
input_location = loc;
return (complain & tf_error) ? true : ok;
}
bool
perform_deferred_access_checks (tsubst_flags_t complain)
{
return perform_access_checks (get_deferred_access_checks (), complain);
}
bool
perform_or_defer_access_check (tree binfo, tree decl, tree diag_decl,
tsubst_flags_t complain,
access_failure_info *afi)
{
int i;
deferred_access *ptr;
deferred_access_check *chk;
if (deferred_access_no_check)
return true;
gcc_assert (TREE_CODE (binfo) == TREE_BINFO);
ptr = &deferred_access_stack->last ();
if (ptr->deferring_access_checks_kind == dk_no_deferred)
{
bool ok = enforce_access (binfo, decl, diag_decl, complain, afi);
return (complain & tf_error) ? true : ok;
}
FOR_EACH_VEC_SAFE_ELT (ptr->deferred_access_checks, i, chk)
{
if (chk->decl == decl && chk->binfo == binfo &&
chk->diag_decl == diag_decl)
{
return true;
}
}
deferred_access_check new_access = {binfo, decl, diag_decl, input_location};
vec_safe_push (ptr->deferred_access_checks, new_access);
return true;
}
int
stmts_are_full_exprs_p (void)
{
return current_stmt_tree ()->stmts_are_full_exprs_p;
}
tree
add_stmt (tree t)
{
enum tree_code code = TREE_CODE (t);
if (EXPR_P (t) && code != LABEL_EXPR)
{
if (!EXPR_HAS_LOCATION (t))
SET_EXPR_LOCATION (t, input_location);
STMT_IS_FULL_EXPR_P (t) = stmts_are_full_exprs_p ();
}
if (code == LABEL_EXPR || code == CASE_LABEL_EXPR)
STATEMENT_LIST_HAS_LABEL (cur_stmt_list) = 1;
gcc_checking_assert (!stmt_list_stack->is_empty ());
append_to_statement_list_force (t, &cur_stmt_list);
return t;
}
stmt_tree
current_stmt_tree (void)
{
return (cfun
? &cfun->language->base.x_stmt_tree
: &scope_chain->x_stmt_tree);
}
static tree
maybe_cleanup_point_expr (tree expr)
{
if (!processing_template_decl && stmts_are_full_exprs_p ())
expr = fold_build_cleanup_point_expr (TREE_TYPE (expr), expr);
return expr;
}
tree
maybe_cleanup_point_expr_void (tree expr)
{
if (!processing_template_decl && stmts_are_full_exprs_p ())
expr = fold_build_cleanup_point_expr (void_type_node, expr);
return expr;
}
void
add_decl_expr (tree decl)
{
tree r = build_stmt (DECL_SOURCE_LOCATION (decl), DECL_EXPR, decl);
if (DECL_INITIAL (decl)
|| (DECL_SIZE (decl) && TREE_SIDE_EFFECTS (DECL_SIZE (decl))))
r = maybe_cleanup_point_expr_void (r);
add_stmt (r);
}
tree
do_poplevel (tree stmt_list)
{
tree block = NULL;
if (stmts_are_full_exprs_p ())
block = poplevel (kept_level_p (), 1, 0);
stmt_list = pop_stmt_list (stmt_list);
if (!processing_template_decl)
{
stmt_list = c_build_bind_expr (input_location, block, stmt_list);
}
return stmt_list;
}
static tree
do_pushlevel (scope_kind sk)
{
tree ret = push_stmt_list ();
if (stmts_are_full_exprs_p ())
begin_scope (sk, NULL);
return ret;
}
void
push_cleanup (tree decl, tree cleanup, bool eh_only)
{
tree stmt = build_stmt (input_location, CLEANUP_STMT, NULL, cleanup, decl);
CLEANUP_EH_ONLY (stmt) = eh_only;
add_stmt (stmt);
CLEANUP_BODY (stmt) = push_stmt_list ();
}
static void
begin_maybe_infinite_loop (tree cond)
{
if (!cfun || (DECL_TEMPLATE_INSTANTIATION (current_function_decl)
&& !processing_template_decl))
return;
bool maybe_infinite = true;
if (cond)
{
cond = fold_non_dependent_expr (cond);
maybe_infinite = integer_nonzerop (cond);
}
vec_safe_push (cp_function_chain->infinite_loops,
maybe_infinite ? error_mark_node : NULL_TREE);
}
void
break_maybe_infinite_loop (void)
{
if (!cfun)
return;
cp_function_chain->infinite_loops->last() = NULL_TREE;
}
static void
end_maybe_infinite_loop (tree cond)
{
if (!cfun || (DECL_TEMPLATE_INSTANTIATION (current_function_decl)
&& !processing_template_decl))
return;
tree current = cp_function_chain->infinite_loops->pop();
if (current != NULL_TREE)
{
cond = fold_non_dependent_expr (cond);
if (integer_nonzerop (cond))
current_function_infinite_loop = 1;
}
}
static void
begin_cond (tree *cond_p)
{
if (processing_template_decl)
*cond_p = push_stmt_list ();
}
static void
finish_cond (tree *cond_p, tree expr)
{
if (processing_template_decl)
{
tree cond = pop_stmt_list (*cond_p);
if (expr == NULL_TREE)
gcc_assert (empty_expr_stmt_p (cond));
else if (check_for_bare_parameter_packs (expr))
expr = error_mark_node;
else if (!empty_expr_stmt_p (cond))
expr = build2 (COMPOUND_EXPR, TREE_TYPE (expr), cond, expr);
}
*cond_p = expr;
}
static void
simplify_loop_decl_cond (tree *cond_p, tree body)
{
tree cond, if_stmt;
if (!TREE_SIDE_EFFECTS (body))
return;
cond = *cond_p;
*cond_p = boolean_true_node;
if_stmt = begin_if_stmt ();
cond = cp_build_unary_op (TRUTH_NOT_EXPR, cond, false, tf_warning_or_error);
finish_if_stmt_cond (cond, if_stmt);
finish_break_stmt ();
finish_then_clause (if_stmt);
finish_if_stmt (if_stmt);
}
tree
finish_goto_stmt (tree destination)
{
if (identifier_p (destination))
destination = lookup_label (destination);
if (TREE_CODE (destination) == LABEL_DECL)
TREE_USED (destination) = 1;
else
{
destination = mark_rvalue_use (destination);
if (!processing_template_decl)
{
destination = cp_convert (ptr_type_node, destination,
tf_warning_or_error);
if (error_operand_p (destination))
return NULL_TREE;
destination
= fold_build_cleanup_point_expr (TREE_TYPE (destination),
destination);
}
}
check_goto (destination);
add_stmt (build_predict_expr (PRED_GOTO, NOT_TAKEN));
return add_stmt (build_stmt (input_location, GOTO_EXPR, destination));
}
static tree
maybe_convert_cond (tree cond)
{
if (!cond)
return NULL_TREE;
if (processing_template_decl)
return cond;
if (warn_sequence_point)
verify_sequence_points (cond);
cond = convert_from_reference (cond);
if (TREE_CODE (cond) == MODIFY_EXPR
&& !TREE_NO_WARNING (cond)
&& warn_parentheses)
{
warning_at (EXPR_LOC_OR_LOC (cond, input_location), OPT_Wparentheses,
"suggest parentheses around assignment used as truth value");
TREE_NO_WARNING (cond) = 1;
}
return condition_conversion (cond);
}
tree
finish_expr_stmt (tree expr)
{
tree r = NULL_TREE;
location_t loc = EXPR_LOCATION (expr);
if (expr != NULL_TREE)
{
gcc_assert (expr != error_mark_node || seen_error ());
if (!processing_template_decl)
{
if (warn_sequence_point)
verify_sequence_points (expr);
expr = convert_to_void (expr, ICV_STATEMENT, tf_warning_or_error);
}
else if (!type_dependent_expression_p (expr))
convert_to_void (build_non_dependent_expr (expr), ICV_STATEMENT, 
tf_warning_or_error);
if (check_for_bare_parameter_packs (expr))
expr = error_mark_node;
if (TREE_CODE (expr) != CLEANUP_POINT_EXPR)
{
if (TREE_CODE (expr) != EXPR_STMT)
expr = build_stmt (loc, EXPR_STMT, expr);
expr = maybe_cleanup_point_expr_void (expr);
}
r = add_stmt (expr);
}
return r;
}
tree
begin_if_stmt (void)
{
tree r, scope;
scope = do_pushlevel (sk_cond);
r = build_stmt (input_location, IF_STMT, NULL_TREE,
NULL_TREE, NULL_TREE, scope);
current_binding_level->this_entity = r;
begin_cond (&IF_COND (r));
return r;
}
tree
finish_if_stmt_cond (tree cond, tree if_stmt)
{
cond = maybe_convert_cond (cond);
if (IF_STMT_CONSTEXPR_P (if_stmt)
&& !type_dependent_expression_p (cond)
&& require_constant_expression (cond)
&& !instantiation_dependent_expression_p (cond)
&& TYPE_MAIN_VARIANT (TREE_TYPE (cond)) == boolean_type_node)
{
cond = instantiate_non_dependent_expr (cond);
cond = cxx_constant_value (cond, NULL_TREE);
}
finish_cond (&IF_COND (if_stmt), cond);
add_stmt (if_stmt);
THEN_CLAUSE (if_stmt) = push_stmt_list ();
return cond;
}
tree
finish_then_clause (tree if_stmt)
{
THEN_CLAUSE (if_stmt) = pop_stmt_list (THEN_CLAUSE (if_stmt));
return if_stmt;
}
void
begin_else_clause (tree if_stmt)
{
ELSE_CLAUSE (if_stmt) = push_stmt_list ();
}
void
finish_else_clause (tree if_stmt)
{
ELSE_CLAUSE (if_stmt) = pop_stmt_list (ELSE_CLAUSE (if_stmt));
}
void
finish_if_stmt (tree if_stmt)
{
tree scope = IF_SCOPE (if_stmt);
IF_SCOPE (if_stmt) = NULL;
add_stmt (do_poplevel (scope));
}
tree
begin_while_stmt (void)
{
tree r;
r = build_stmt (input_location, WHILE_STMT, NULL_TREE, NULL_TREE);
add_stmt (r);
WHILE_BODY (r) = do_pushlevel (sk_block);
begin_cond (&WHILE_COND (r));
return r;
}
void
finish_while_stmt_cond (tree cond, tree while_stmt, bool ivdep,
unsigned short unroll)
{
cond = maybe_convert_cond (cond);
finish_cond (&WHILE_COND (while_stmt), cond);
begin_maybe_infinite_loop (cond);
if (ivdep && cond != error_mark_node)
WHILE_COND (while_stmt) = build3 (ANNOTATE_EXPR,
TREE_TYPE (WHILE_COND (while_stmt)),
WHILE_COND (while_stmt),
build_int_cst (integer_type_node,
annot_expr_ivdep_kind),
integer_zero_node);
if (unroll && cond != error_mark_node)
WHILE_COND (while_stmt) = build3 (ANNOTATE_EXPR,
TREE_TYPE (WHILE_COND (while_stmt)),
WHILE_COND (while_stmt),
build_int_cst (integer_type_node,
annot_expr_unroll_kind),
build_int_cst (integer_type_node,
unroll));
simplify_loop_decl_cond (&WHILE_COND (while_stmt), WHILE_BODY (while_stmt));
}
void
finish_while_stmt (tree while_stmt)
{
end_maybe_infinite_loop (boolean_true_node);
WHILE_BODY (while_stmt) = do_poplevel (WHILE_BODY (while_stmt));
}
tree
begin_do_stmt (void)
{
tree r = build_stmt (input_location, DO_STMT, NULL_TREE, NULL_TREE);
begin_maybe_infinite_loop (boolean_true_node);
add_stmt (r);
DO_BODY (r) = push_stmt_list ();
return r;
}
void
finish_do_body (tree do_stmt)
{
tree body = DO_BODY (do_stmt) = pop_stmt_list (DO_BODY (do_stmt));
if (TREE_CODE (body) == STATEMENT_LIST && STATEMENT_LIST_TAIL (body))
body = STATEMENT_LIST_TAIL (body)->stmt;
if (IS_EMPTY_STMT (body))
warning (OPT_Wempty_body,
"suggest explicit braces around empty body in %<do%> statement");
}
void
finish_do_stmt (tree cond, tree do_stmt, bool ivdep, unsigned short unroll)
{
cond = maybe_convert_cond (cond);
end_maybe_infinite_loop (cond);
if (ivdep && cond != error_mark_node)
cond = build3 (ANNOTATE_EXPR, TREE_TYPE (cond), cond,
build_int_cst (integer_type_node, annot_expr_ivdep_kind),
integer_zero_node);
if (unroll && cond != error_mark_node)
cond = build3 (ANNOTATE_EXPR, TREE_TYPE (cond), cond,
build_int_cst (integer_type_node, annot_expr_unroll_kind),
build_int_cst (integer_type_node, unroll));
DO_COND (do_stmt) = cond;
}
tree
finish_return_stmt (tree expr)
{
tree r;
bool no_warning;
expr = check_return_expr (expr, &no_warning);
if (error_operand_p (expr)
|| (flag_openmp && !check_omp_return ()))
{
if (warn_return_type)
TREE_NO_WARNING (current_function_decl) = true;
return error_mark_node;
}
if (!processing_template_decl)
{
if (warn_sequence_point)
verify_sequence_points (expr);
if (DECL_DESTRUCTOR_P (current_function_decl)
|| (DECL_CONSTRUCTOR_P (current_function_decl)
&& targetm.cxx.cdtor_returns_this ()))
{
return finish_goto_stmt (cdtor_label);
}
}
r = build_stmt (input_location, RETURN_EXPR, expr);
TREE_NO_WARNING (r) |= no_warning;
r = maybe_cleanup_point_expr_void (r);
r = add_stmt (r);
return r;
}
tree
begin_for_scope (tree *init)
{
tree scope = NULL_TREE;
if (flag_new_for_scope)
scope = do_pushlevel (sk_for);
if (processing_template_decl)
*init = push_stmt_list ();
else
*init = NULL_TREE;
return scope;
}
tree
begin_for_stmt (tree scope, tree init)
{
tree r;
r = build_stmt (input_location, FOR_STMT, NULL_TREE, NULL_TREE,
NULL_TREE, NULL_TREE, NULL_TREE);
if (scope == NULL_TREE)
{
gcc_assert (!init || !flag_new_for_scope);
if (!init)
scope = begin_for_scope (&init);
}
FOR_INIT_STMT (r) = init;
FOR_SCOPE (r) = scope;
return r;
}
void
finish_init_stmt (tree for_stmt)
{
if (processing_template_decl)
FOR_INIT_STMT (for_stmt) = pop_stmt_list (FOR_INIT_STMT (for_stmt));
add_stmt (for_stmt);
FOR_BODY (for_stmt) = do_pushlevel (sk_block);
begin_cond (&FOR_COND (for_stmt));
}
void
finish_for_cond (tree cond, tree for_stmt, bool ivdep, unsigned short unroll)
{
cond = maybe_convert_cond (cond);
finish_cond (&FOR_COND (for_stmt), cond);
begin_maybe_infinite_loop (cond);
if (ivdep && cond != error_mark_node)
FOR_COND (for_stmt) = build3 (ANNOTATE_EXPR,
TREE_TYPE (FOR_COND (for_stmt)),
FOR_COND (for_stmt),
build_int_cst (integer_type_node,
annot_expr_ivdep_kind),
integer_zero_node);
if (unroll && cond != error_mark_node)
FOR_COND (for_stmt) = build3 (ANNOTATE_EXPR,
TREE_TYPE (FOR_COND (for_stmt)),
FOR_COND (for_stmt),
build_int_cst (integer_type_node,
annot_expr_unroll_kind),
build_int_cst (integer_type_node,
unroll));
simplify_loop_decl_cond (&FOR_COND (for_stmt), FOR_BODY (for_stmt));
}
void
finish_for_expr (tree expr, tree for_stmt)
{
if (!expr)
return;
if (type_unknown_p (expr))
{
cxx_incomplete_type_error (expr, TREE_TYPE (expr));
expr = error_mark_node;
}
if (!processing_template_decl)
{
if (warn_sequence_point)
verify_sequence_points (expr);
expr = convert_to_void (expr, ICV_THIRD_IN_FOR,
tf_warning_or_error);
}
else if (!type_dependent_expression_p (expr))
convert_to_void (build_non_dependent_expr (expr), ICV_THIRD_IN_FOR,
tf_warning_or_error);
expr = maybe_cleanup_point_expr_void (expr);
if (check_for_bare_parameter_packs (expr))
expr = error_mark_node;
FOR_EXPR (for_stmt) = expr;
}
void
finish_for_stmt (tree for_stmt)
{
end_maybe_infinite_loop (boolean_true_node);
if (TREE_CODE (for_stmt) == RANGE_FOR_STMT)
RANGE_FOR_BODY (for_stmt) = do_poplevel (RANGE_FOR_BODY (for_stmt));
else
FOR_BODY (for_stmt) = do_poplevel (FOR_BODY (for_stmt));
if (flag_new_for_scope)
{
tree scope;
tree *scope_ptr = (TREE_CODE (for_stmt) == RANGE_FOR_STMT
? &RANGE_FOR_SCOPE (for_stmt)
: &FOR_SCOPE (for_stmt));
scope = *scope_ptr;
*scope_ptr = NULL;
add_stmt (do_poplevel (scope));
}
}
tree
begin_range_for_stmt (tree scope, tree init)
{
tree r;
begin_maybe_infinite_loop (boolean_false_node);
r = build_stmt (input_location, RANGE_FOR_STMT,
NULL_TREE, NULL_TREE, NULL_TREE, NULL_TREE, NULL_TREE);
if (scope == NULL_TREE)
{
gcc_assert (!init || !flag_new_for_scope);
if (!init)
scope = begin_for_scope (&init);
}
if (init)
pop_stmt_list (init);
RANGE_FOR_SCOPE (r) = scope;
return r;
}
void
finish_range_for_decl (tree range_for_stmt, tree decl, tree expr)
{
RANGE_FOR_DECL (range_for_stmt) = decl;
RANGE_FOR_EXPR (range_for_stmt) = expr;
add_stmt (range_for_stmt);
RANGE_FOR_BODY (range_for_stmt) = do_pushlevel (sk_block);
}
tree
finish_break_stmt (void)
{
if (!block_may_fallthru (cur_stmt_list))
return void_node;
note_break_stmt ();
return add_stmt (build_stmt (input_location, BREAK_STMT));
}
tree
finish_continue_stmt (void)
{
return add_stmt (build_stmt (input_location, CONTINUE_STMT));
}
tree
begin_switch_stmt (void)
{
tree r, scope;
scope = do_pushlevel (sk_cond);
r = build_stmt (input_location, SWITCH_STMT, NULL_TREE, NULL_TREE, NULL_TREE, scope);
begin_cond (&SWITCH_STMT_COND (r));
return r;
}
void
finish_switch_cond (tree cond, tree switch_stmt)
{
tree orig_type = NULL;
if (!processing_template_decl)
{
cond = build_expr_type_conversion (WANT_INT | WANT_ENUM, cond, true);
if (cond == NULL_TREE)
{
error ("switch quantity not an integer");
cond = error_mark_node;
}
orig_type = unlowered_expr_type (cond);
if (TREE_CODE (orig_type) != ENUMERAL_TYPE)
orig_type = TREE_TYPE (cond);
if (cond != error_mark_node)
{
cond = perform_integral_promotions (cond);
cond = maybe_cleanup_point_expr (cond);
}
}
if (check_for_bare_parameter_packs (cond))
cond = error_mark_node;
else if (!processing_template_decl && warn_sequence_point)
verify_sequence_points (cond);
finish_cond (&SWITCH_STMT_COND (switch_stmt), cond);
SWITCH_STMT_TYPE (switch_stmt) = orig_type;
add_stmt (switch_stmt);
push_switch (switch_stmt);
SWITCH_STMT_BODY (switch_stmt) = push_stmt_list ();
}
void
finish_switch_stmt (tree switch_stmt)
{
tree scope;
SWITCH_STMT_BODY (switch_stmt) =
pop_stmt_list (SWITCH_STMT_BODY (switch_stmt));
pop_switch ();
scope = SWITCH_STMT_SCOPE (switch_stmt);
SWITCH_STMT_SCOPE (switch_stmt) = NULL;
add_stmt (do_poplevel (scope));
}
tree
begin_try_block (void)
{
tree r = build_stmt (input_location, TRY_BLOCK, NULL_TREE, NULL_TREE);
add_stmt (r);
TRY_STMTS (r) = push_stmt_list ();
return r;
}
tree
begin_function_try_block (tree *compound_stmt)
{
tree r;
*compound_stmt = begin_compound_stmt (0);
r = begin_try_block ();
FN_TRY_BLOCK_P (r) = 1;
return r;
}
void
finish_try_block (tree try_block)
{
TRY_STMTS (try_block) = pop_stmt_list (TRY_STMTS (try_block));
TRY_HANDLERS (try_block) = push_stmt_list ();
}
void
finish_cleanup_try_block (tree try_block)
{
TRY_STMTS (try_block) = pop_stmt_list (TRY_STMTS (try_block));
}
void
finish_cleanup (tree cleanup, tree try_block)
{
TRY_HANDLERS (try_block) = cleanup;
CLEANUP_P (try_block) = 1;
}
void
finish_function_try_block (tree try_block)
{
finish_try_block (try_block);
in_function_try_handler = 1;
}
void
finish_handler_sequence (tree try_block)
{
TRY_HANDLERS (try_block) = pop_stmt_list (TRY_HANDLERS (try_block));
check_handlers (TRY_HANDLERS (try_block));
}
void
finish_function_handler_sequence (tree try_block, tree compound_stmt)
{
in_function_try_handler = 0;
finish_handler_sequence (try_block);
finish_compound_stmt (compound_stmt);
}
tree
begin_handler (void)
{
tree r;
r = build_stmt (input_location, HANDLER, NULL_TREE, NULL_TREE);
add_stmt (r);
HANDLER_BODY (r) = do_pushlevel (sk_catch);
return r;
}
void
finish_handler_parms (tree decl, tree handler)
{
tree type = NULL_TREE;
if (processing_template_decl)
{
if (decl)
{
decl = pushdecl (decl);
decl = push_template_decl (decl);
HANDLER_PARMS (handler) = decl;
type = TREE_TYPE (decl);
}
}
else
{
type = expand_start_catch_block (decl);
if (warn_catch_value
&& type != NULL_TREE
&& type != error_mark_node
&& TREE_CODE (TREE_TYPE (decl)) != REFERENCE_TYPE)
{
tree orig_type = TREE_TYPE (decl);
if (CLASS_TYPE_P (orig_type))
{
if (TYPE_POLYMORPHIC_P (orig_type))
warning (OPT_Wcatch_value_,
"catching polymorphic type %q#T by value", orig_type);
else if (warn_catch_value > 1)
warning (OPT_Wcatch_value_,
"catching type %q#T by value", orig_type);
}
else if (warn_catch_value > 2)
warning (OPT_Wcatch_value_,
"catching non-reference type %q#T", orig_type);
}
}
HANDLER_TYPE (handler) = type;
}
void
finish_handler (tree handler)
{
if (!processing_template_decl)
expand_end_catch_block ();
HANDLER_BODY (handler) = do_poplevel (HANDLER_BODY (handler));
}
tree
begin_compound_stmt (unsigned int flags)
{
tree r;
if (flags & BCS_NO_SCOPE)
{
r = push_stmt_list ();
STATEMENT_LIST_NO_SCOPE (r) = 1;
keep_next_level (false);
}
else
{
scope_kind sk = sk_block;
if (flags & BCS_TRY_BLOCK)
sk = sk_try;
else if (flags & BCS_TRANSACTION)
sk = sk_transaction;
r = do_pushlevel (sk);
}
if (processing_template_decl)
{
r = build3 (BIND_EXPR, NULL, NULL, r, NULL);
BIND_EXPR_TRY_BLOCK (r) = (flags & BCS_TRY_BLOCK) != 0;
BIND_EXPR_BODY_BLOCK (r) = (flags & BCS_FN_BODY) != 0;
TREE_SIDE_EFFECTS (r) = 1;
}
return r;
}
void
finish_compound_stmt (tree stmt)
{
if (TREE_CODE (stmt) == BIND_EXPR)
{
tree body = do_poplevel (BIND_EXPR_BODY (stmt));
if (TREE_CODE (body) == STATEMENT_LIST
&& STATEMENT_LIST_HEAD (body) == NULL
&& !BIND_EXPR_BODY_BLOCK (stmt)
&& !BIND_EXPR_TRY_BLOCK (stmt))
stmt = body;
else
BIND_EXPR_BODY (stmt) = body;
}
else if (STATEMENT_LIST_NO_SCOPE (stmt))
stmt = pop_stmt_list (stmt);
else
{
objc_clear_super_receiver ();
stmt = do_poplevel (stmt);
}
add_stmt (stmt);
}
tree
finish_asm_stmt (int volatile_p, tree string, tree output_operands,
tree input_operands, tree clobbers, tree labels, bool inline_p)
{
tree r;
tree t;
int ninputs = list_length (input_operands);
int noutputs = list_length (output_operands);
if (!processing_template_decl)
{
const char *constraint;
const char **oconstraints;
bool allows_mem, allows_reg, is_inout;
tree operand;
int i;
oconstraints = XALLOCAVEC (const char *, noutputs);
string = resolve_asm_operand_names (string, output_operands,
input_operands, labels);
for (i = 0, t = output_operands; t; t = TREE_CHAIN (t), ++i)
{
operand = TREE_VALUE (t);
STRIP_NOPS (operand);
operand = mark_lvalue_use (operand);
if (!lvalue_or_else (operand, lv_asm, tf_warning_or_error))
operand = error_mark_node;
if (operand != error_mark_node
&& (TREE_READONLY (operand)
|| CP_TYPE_CONST_P (TREE_TYPE (operand))
|| TREE_CODE (TREE_TYPE (operand)) == FUNCTION_TYPE
|| TREE_CODE (TREE_TYPE (operand)) == METHOD_TYPE
|| (CLASS_TYPE_P (TREE_TYPE (operand))
&& C_TYPE_FIELDS_READONLY (TREE_TYPE (operand)))))
cxx_readonly_error (operand, lv_asm);
tree *op = &operand;
while (TREE_CODE (*op) == COMPOUND_EXPR)
op = &TREE_OPERAND (*op, 1);
switch (TREE_CODE (*op))
{
case PREINCREMENT_EXPR:
case PREDECREMENT_EXPR:
case MODIFY_EXPR:
*op = genericize_compound_lvalue (*op);
op = &TREE_OPERAND (*op, 1);
break;
default:
break;
}
constraint = TREE_STRING_POINTER (TREE_VALUE (TREE_PURPOSE (t)));
oconstraints[i] = constraint;
if (parse_output_constraint (&constraint, i, ninputs, noutputs,
&allows_mem, &allows_reg, &is_inout))
{
if (!allows_reg && !cxx_mark_addressable (*op))
operand = error_mark_node;
}
else
operand = error_mark_node;
TREE_VALUE (t) = operand;
}
for (i = 0, t = input_operands; t; ++i, t = TREE_CHAIN (t))
{
constraint = TREE_STRING_POINTER (TREE_VALUE (TREE_PURPOSE (t)));
bool constraint_parsed
= parse_input_constraint (&constraint, i, ninputs, noutputs, 0,   
oconstraints, &allows_mem, &allows_reg);
if (constraint_parsed && !allows_reg && allows_mem)
operand = mark_lvalue_use (TREE_VALUE (t));
else
operand = decay_conversion (TREE_VALUE (t), tf_warning_or_error);
if (TREE_TYPE (operand) == unknown_type_node)
{
error ("type of asm operand %qE could not be determined",
TREE_VALUE (t));
operand = error_mark_node;
}
if (constraint_parsed)
{
if (!allows_reg && allows_mem)
{
STRIP_NOPS (operand);
tree *op = &operand;
while (TREE_CODE (*op) == COMPOUND_EXPR)
op = &TREE_OPERAND (*op, 1);
switch (TREE_CODE (*op))
{
case PREINCREMENT_EXPR:
case PREDECREMENT_EXPR:
case MODIFY_EXPR:
*op = genericize_compound_lvalue (*op);
op = &TREE_OPERAND (*op, 1);
break;
default:
break;
}
if (!cxx_mark_addressable (*op))
operand = error_mark_node;
}
else if (!allows_reg && !allows_mem)
{
tree constop = maybe_constant_value (operand);
if (TREE_CONSTANT (constop))
operand = constop;
}
}
else
operand = error_mark_node;
TREE_VALUE (t) = operand;
}
}
r = build_stmt (input_location, ASM_EXPR, string,
output_operands, input_operands,
clobbers, labels);
ASM_VOLATILE_P (r) = volatile_p || noutputs == 0;
ASM_INLINE_P (r) = inline_p;
r = maybe_cleanup_point_expr_void (r);
return add_stmt (r);
}
tree
finish_label_stmt (tree name)
{
tree decl = define_label (input_location, name);
if (decl == error_mark_node)
return error_mark_node;
add_stmt (build_stmt (input_location, LABEL_EXPR, decl));
return decl;
}
void
finish_label_decl (tree name)
{
if (!at_function_scope_p ())
{
error ("__label__ declarations are only allowed in function scopes");
return;
}
add_decl_expr (declare_local_label (name));
}
void
finish_decl_cleanup (tree decl, tree cleanup)
{
push_cleanup (decl, cleanup, false);
}
void
finish_eh_cleanup (tree cleanup)
{
push_cleanup (NULL, cleanup, true);
}
void
finish_mem_initializers (tree mem_inits)
{
mem_inits = nreverse (mem_inits);
if (processing_template_decl)
{
tree mem;
for (mem = mem_inits; mem; mem = TREE_CHAIN (mem))
{
if (TREE_CODE (TREE_PURPOSE (mem)) != TYPE_PACK_EXPANSION
&& check_for_bare_parameter_packs (TREE_VALUE (mem)))
TREE_VALUE (mem) = error_mark_node;
}
add_stmt (build_min_nt_loc (UNKNOWN_LOCATION,
CTOR_INITIALIZER, mem_inits));
}
else
emit_mem_initializers (mem_inits);
}
tree
force_paren_expr (tree expr)
{
if (cxx_dialect < cxx14)
return expr;
if (cp_unevaluated_operand)
return expr;
if (!DECL_P (expr) && TREE_CODE (expr) != COMPONENT_REF
&& TREE_CODE (expr) != SCOPE_REF)
return expr;
if (TREE_CODE (expr) == COMPONENT_REF
|| TREE_CODE (expr) == SCOPE_REF)
REF_PARENTHESIZED_P (expr) = true;
else if (processing_template_decl)
expr = build1 (PAREN_EXPR, TREE_TYPE (expr), expr);
else if (VAR_P (expr) && DECL_HARD_REGISTER (expr))
;
else
{
cp_lvalue_kind kind = lvalue_kind (expr);
if ((kind & ~clk_class) != clk_none)
{
tree type = unlowered_expr_type (expr);
bool rval = !!(kind & clk_rvalueref);
type = cp_build_reference_type (type, rval);
warning_sentinel s (extra_warnings);
expr = build_static_cast (type, expr, tf_error);
if (expr != error_mark_node)
REF_PARENTHESIZED_P (expr) = true;
}
}
return expr;
}
tree
maybe_undo_parenthesized_ref (tree t)
{
if (cxx_dialect < cxx14)
return t;
if (INDIRECT_REF_P (t) && REF_PARENTHESIZED_P (t))
{
t = TREE_OPERAND (t, 0);
while (TREE_CODE (t) == NON_LVALUE_EXPR
|| TREE_CODE (t) == NOP_EXPR)
t = TREE_OPERAND (t, 0);
gcc_assert (TREE_CODE (t) == ADDR_EXPR
|| TREE_CODE (t) == STATIC_CAST_EXPR);
t = TREE_OPERAND (t, 0);
}
else if (TREE_CODE (t) == PAREN_EXPR)
t = TREE_OPERAND (t, 0);
return t;
}
cp_expr
finish_parenthesized_expr (cp_expr expr)
{
if (EXPR_P (expr))
TREE_NO_WARNING (expr) = 1;
if (TREE_CODE (expr) == OFFSET_REF
|| TREE_CODE (expr) == SCOPE_REF)
PTRMEM_OK_P (expr) = 0;
if (TREE_CODE (expr) == STRING_CST)
PAREN_STRING_LITERAL_P (expr) = 1;
expr = cp_expr (force_paren_expr (expr), expr.get_location ());
return expr;
}
tree
finish_non_static_data_member (tree decl, tree object, tree qualifying_scope)
{
gcc_assert (TREE_CODE (decl) == FIELD_DECL);
bool try_omp_private = !object && omp_private_member_map;
tree ret;
if (!object)
{
tree scope = qualifying_scope;
if (scope == NULL_TREE)
scope = context_for_name_lookup (decl);
object = maybe_dummy_object (scope, NULL);
}
object = maybe_resolve_dummy (object, true);
if (object == error_mark_node)
return error_mark_node;
if (is_dummy_object (object) && cp_unevaluated_operand == 0
&& (!processing_template_decl || !current_class_ref))
{
if (current_function_decl
&& DECL_STATIC_FUNCTION_P (current_function_decl))
error ("invalid use of member %qD in static member function", decl);
else
error ("invalid use of non-static data member %qD", decl);
inform (DECL_SOURCE_LOCATION (decl), "declared here");
return error_mark_node;
}
if (current_class_ptr)
TREE_USED (current_class_ptr) = 1;
if (processing_template_decl && !qualifying_scope)
{
tree type = TREE_TYPE (decl);
if (TREE_CODE (type) == REFERENCE_TYPE)
;
else if (PACK_EXPANSION_P (type))
type = NULL_TREE;
else
{
int quals = cp_type_quals (TREE_TYPE (object));
if (DECL_MUTABLE_P (decl))
quals &= ~TYPE_QUAL_CONST;
quals |= cp_type_quals (TREE_TYPE (decl));
type = cp_build_qualified_type (type, quals);
}
ret = (convert_from_reference
(build_min (COMPONENT_REF, type, object, decl, NULL_TREE)));
}
else if (processing_template_decl)
ret = build_qualified_name (TREE_TYPE (decl),
qualifying_scope,
decl,
false);
else
{
tree access_type = TREE_TYPE (object);
perform_or_defer_access_check (TYPE_BINFO (access_type), decl,
decl, tf_warning_or_error);
if (qualifying_scope)
{
tree binfo = NULL_TREE;
object = build_scoped_ref (object, qualifying_scope,
&binfo);
}
ret = build_class_member_access_expr (object, decl,
NULL_TREE,
false,
tf_warning_or_error);
}
if (try_omp_private)
{
tree *v = omp_private_member_map->get (decl);
if (v)
ret = convert_from_reference (*v);
}
return ret;
}
void
add_typedef_to_current_template_for_access_check (tree typedef_decl,
tree context,
location_t location)
{
tree template_info = NULL;
tree cs = current_scope ();
if (!is_typedef_decl (typedef_decl)
|| !context
|| !CLASS_TYPE_P (context)
|| !cs)
return;
if (CLASS_TYPE_P (cs) || TREE_CODE (cs) == FUNCTION_DECL)
template_info = get_template_info (cs);
if (template_info
&& TI_TEMPLATE (template_info)
&& !currently_open_class (context))
append_type_to_template_for_access_check (cs, typedef_decl,
context, location);
}
void
check_accessibility_of_qualified_id (tree decl,
tree object_type,
tree nested_name_specifier)
{
tree scope;
tree qualifying_type = NULL_TREE;
add_typedef_to_current_template_for_access_check (decl,
nested_name_specifier
? nested_name_specifier
: DECL_CONTEXT (decl),
input_location);
if (deferred_access_no_check)
return;
scope = context_for_name_lookup (decl);
if (!TYPE_P (scope))
return;
if (object_type
&& CLASS_TYPE_P (object_type)
&& DERIVED_FROM_P (scope, object_type))
qualifying_type = object_type;
else if (nested_name_specifier)
{
tree ct;
if (DECL_NONSTATIC_MEMBER_P (decl)
&& current_class_ptr
&& DERIVED_FROM_P (scope, ct = current_nonlambda_class_type ()))
qualifying_type = ct;
else
qualifying_type = nested_name_specifier;
}
else
qualifying_type = currently_open_derived_class (scope);
if (qualifying_type 
&& CLASS_TYPE_P (qualifying_type)
&& !dependent_type_p (qualifying_type))
perform_or_defer_access_check (TYPE_BINFO (qualifying_type), decl,
decl, tf_warning_or_error);
}
tree
finish_qualified_id_expr (tree qualifying_class,
tree expr,
bool done,
bool address_p,
bool template_p,
bool template_arg_p,
tsubst_flags_t complain)
{
gcc_assert (TYPE_P (qualifying_class));
if (error_operand_p (expr))
return error_mark_node;
if ((DECL_P (expr) || BASELINK_P (expr))
&& !mark_used (expr, complain))
return error_mark_node;
if (template_p)
{
if (TREE_CODE (expr) == UNBOUND_CLASS_TEMPLATE)
{
qualifying_class = TYPE_CONTEXT (expr);
expr = TYPE_IDENTIFIER (expr);
}
else
check_template_keyword (expr);
}
if (address_p && done)
{
if (TREE_CODE (expr) == SCOPE_REF)
expr = TREE_OPERAND (expr, 1);
expr = build_offset_ref (qualifying_class, expr,
true, complain);
return expr;
}
if (TREE_CODE (qualifying_class) == ENUMERAL_TYPE
&& TREE_CODE (expr) != IDENTIFIER_NODE)
return expr;
if (template_arg_p)
;
else if (TREE_CODE (expr) == FIELD_DECL)
{
push_deferring_access_checks (dk_no_check);
expr = finish_non_static_data_member (expr, NULL_TREE,
qualifying_class);
pop_deferring_access_checks ();
}
else if (BASELINK_P (expr))
{
if (!shared_member_p (expr)
&& current_class_ptr
&& DERIVED_FROM_P (qualifying_class,
current_nonlambda_class_type ()))
expr = (build_class_member_access_expr
(maybe_dummy_object (qualifying_class, NULL),
expr,
BASELINK_ACCESS_BINFO (expr),
false,
complain));
else if (done)
expr = build_offset_ref (qualifying_class, expr, false,
complain);
}
else
{
if (processing_template_decl
&& (!currently_open_class (qualifying_class)
|| TREE_CODE (expr) == IDENTIFIER_NODE
|| TREE_CODE (expr) == TEMPLATE_ID_EXPR
|| TREE_CODE (expr) == BIT_NOT_EXPR))
expr = build_qualified_name (TREE_TYPE (expr),
qualifying_class, expr,
template_p);
expr = convert_from_reference (expr);
}
return expr;
}
tree
begin_stmt_expr (void)
{
return push_stmt_list ();
}
tree
finish_stmt_expr_expr (tree expr, tree stmt_expr)
{
if (error_operand_p (expr))
{
TREE_TYPE (stmt_expr) = error_mark_node;
return error_mark_node;
}
if (expr)
{
tree type = TREE_TYPE (expr);
if (type && type_unknown_p (type))
{
error ("a statement expression is an insufficient context"
" for overload resolution");
TREE_TYPE (stmt_expr) = error_mark_node;
return error_mark_node;
}
else if (processing_template_decl)
{
expr = build_stmt (input_location, EXPR_STMT, expr);
expr = add_stmt (expr);
EXPR_STMT_STMT_EXPR_RESULT (expr) = 1;
}
else if (VOID_TYPE_P (type))
{
expr = finish_expr_stmt (expr);
}
else
{
expr = force_rvalue (expr, tf_warning_or_error);
if (error_operand_p (expr))
return error_mark_node;
type = TREE_TYPE (expr);
if (TREE_CODE (expr) != CLEANUP_POINT_EXPR)
expr = maybe_cleanup_point_expr (expr);
add_stmt (expr);
}
TREE_TYPE (stmt_expr) = type;
}
return stmt_expr;
}
tree
finish_stmt_expr (tree stmt_expr, bool has_no_scope)
{
tree type;
tree result;
if (error_operand_p (stmt_expr))
{
pop_stmt_list (stmt_expr);
return error_mark_node;
}
gcc_assert (TREE_CODE (stmt_expr) == STATEMENT_LIST);
type = TREE_TYPE (stmt_expr);
result = pop_stmt_list (stmt_expr);
TREE_TYPE (result) = type;
if (processing_template_decl)
{
result = build_min (STMT_EXPR, type, result);
TREE_SIDE_EFFECTS (result) = 1;
STMT_EXPR_NO_SCOPE (result) = has_no_scope;
}
else if (CLASS_TYPE_P (type))
{
result = force_target_expr (type, result, tf_warning_or_error);
}
return result;
}
tree
stmt_expr_value_expr (tree stmt_expr)
{
tree t = STMT_EXPR_STMT (stmt_expr);
if (TREE_CODE (t) == BIND_EXPR)
t = BIND_EXPR_BODY (t);
if (TREE_CODE (t) == STATEMENT_LIST && STATEMENT_LIST_TAIL (t))
t = STATEMENT_LIST_TAIL (t)->stmt;
if (TREE_CODE (t) == EXPR_STMT)
t = EXPR_STMT_EXPR (t);
return t;
}
bool
empty_expr_stmt_p (tree expr_stmt)
{
tree body = NULL_TREE;
if (expr_stmt == void_node)
return true;
if (expr_stmt)
{
if (TREE_CODE (expr_stmt) == EXPR_STMT)
body = EXPR_STMT_EXPR (expr_stmt);
else if (TREE_CODE (expr_stmt) == STATEMENT_LIST)
body = expr_stmt;
}
if (body)
{
if (TREE_CODE (body) == STATEMENT_LIST)
return tsi_end_p (tsi_start (body));
else
return empty_expr_stmt_p (body);
}
return false;
}
cp_expr
perform_koenig_lookup (cp_expr fn, vec<tree, va_gc> *args,
tsubst_flags_t complain)
{
tree identifier = NULL_TREE;
tree functions = NULL_TREE;
tree tmpl_args = NULL_TREE;
bool template_id = false;
location_t loc = fn.get_location ();
if (TREE_CODE (fn) == TEMPLATE_ID_EXPR)
{
template_id = true;
tmpl_args = TREE_OPERAND (fn, 1);
fn = TREE_OPERAND (fn, 0);
}
if (identifier_p (fn))
identifier = fn;
else
{
functions = fn;
identifier = OVL_NAME (functions);
}
if (!any_type_dependent_arguments_p (args)
&& !any_dependent_template_arguments_p (tmpl_args))
{
fn = lookup_arg_dependent (identifier, functions, args);
if (!fn)
{
if (complain & tf_error)
fn = unqualified_fn_lookup_error (cp_expr (identifier, loc));
else
fn = identifier;
}
}
if (fn && template_id && fn != error_mark_node)
fn = build2 (TEMPLATE_ID_EXPR, unknown_type_node, fn, tmpl_args);
return fn;
}
tree
finish_call_expr (tree fn, vec<tree, va_gc> **args, bool disallow_virtual,
bool koenig_p, tsubst_flags_t complain)
{
tree result;
tree orig_fn;
vec<tree, va_gc> *orig_args = NULL;
if (fn == error_mark_node)
return error_mark_node;
gcc_assert (!TYPE_P (fn));
fn = maybe_undo_parenthesized_ref (fn);
orig_fn = fn;
if (processing_template_decl)
{
if (is_overloaded_fn (fn))
{
tree ifn = get_first_fn (fn);
if (TREE_CODE (ifn) == FUNCTION_DECL
&& DECL_LOCAL_FUNCTION_P (ifn))
orig_fn = DECL_NAME (ifn);
}
if (type_dependent_expression_p (fn)
|| any_type_dependent_arguments_p (*args))
{
result = build_min_nt_call_vec (orig_fn, *args);
SET_EXPR_LOCATION (result, EXPR_LOC_OR_LOC (fn, input_location));
KOENIG_LOOKUP_P (result) = koenig_p;
if (is_overloaded_fn (fn))
{
fn = get_fns (fn);
lookup_keep (fn, true);
}
if (cfun)
{
bool abnormal = true;
for (lkp_iterator iter (fn); abnormal && iter; ++iter)
{
tree fndecl = *iter;
if (TREE_CODE (fndecl) != FUNCTION_DECL
|| !TREE_THIS_VOLATILE (fndecl))
abnormal = false;
}
if (abnormal)
current_function_returns_abnormally = 1;
}
return result;
}
orig_args = make_tree_vector_copy (*args);
if (!BASELINK_P (fn)
&& TREE_CODE (fn) != PSEUDO_DTOR_EXPR
&& TREE_TYPE (fn) != unknown_type_node)
fn = build_non_dependent_expr (fn);
make_args_non_dependent (*args);
}
if (TREE_CODE (fn) == COMPONENT_REF)
{
tree member = TREE_OPERAND (fn, 1);
if (BASELINK_P (member))
{
tree object = TREE_OPERAND (fn, 0);
return build_new_method_call (object, member,
args, NULL_TREE,
(disallow_virtual
? LOOKUP_NORMAL | LOOKUP_NONVIRTUAL
: LOOKUP_NORMAL),
NULL,
complain);
}
}
if (TREE_CODE (fn) == ADDR_EXPR
&& TREE_CODE (TREE_OPERAND (fn, 0)) == OVERLOAD)
fn = TREE_OPERAND (fn, 0);
if (is_overloaded_fn (fn))
fn = baselink_for_fns (fn);
result = NULL_TREE;
if (BASELINK_P (fn))
{
tree object;
if (DECL_MAYBE_IN_CHARGE_CONSTRUCTOR_P (get_first_fn (fn)))
{
object = build_dummy_object (BINFO_TYPE (BASELINK_ACCESS_BINFO (fn)));
}
else
object = maybe_dummy_object (BINFO_TYPE (BASELINK_ACCESS_BINFO (fn)),
NULL);
result = build_new_method_call (object, fn, args, NULL_TREE,
(disallow_virtual
? LOOKUP_NORMAL|LOOKUP_NONVIRTUAL
: LOOKUP_NORMAL),
NULL,
complain);
}
else if (is_overloaded_fn (fn))
{
if (TREE_CODE (fn) == FUNCTION_DECL
&& (DECL_BUILT_IN_CLASS (fn) == BUILT_IN_NORMAL
|| DECL_BUILT_IN_CLASS (fn) == BUILT_IN_MD))
result = resolve_overloaded_builtin (input_location, fn, *args);
if (!result)
{
if (warn_sizeof_pointer_memaccess
&& (complain & tf_warning)
&& !vec_safe_is_empty (*args)
&& !processing_template_decl)
{
location_t sizeof_arg_loc[3];
tree sizeof_arg[3];
unsigned int i;
for (i = 0; i < 3; i++)
{
tree t;
sizeof_arg_loc[i] = UNKNOWN_LOCATION;
sizeof_arg[i] = NULL_TREE;
if (i >= (*args)->length ())
continue;
t = (**args)[i];
if (TREE_CODE (t) != SIZEOF_EXPR)
continue;
if (SIZEOF_EXPR_TYPE_P (t))
sizeof_arg[i] = TREE_TYPE (TREE_OPERAND (t, 0));
else
sizeof_arg[i] = TREE_OPERAND (t, 0);
sizeof_arg_loc[i] = EXPR_LOCATION (t);
}
sizeof_pointer_memaccess_warning
(sizeof_arg_loc, fn, *args,
sizeof_arg, same_type_ignoring_top_level_qualifiers_p);
}
result = build_new_function_call (fn, args, complain);
}
}
else if (TREE_CODE (fn) == PSEUDO_DTOR_EXPR)
{
if (!vec_safe_is_empty (*args))
error ("arguments to destructor are not allowed");
result = build1 (NOP_EXPR,
void_type_node,
TREE_OPERAND (fn, 0));
TREE_SIDE_EFFECTS (result) = 1;
}
else if (CLASS_TYPE_P (TREE_TYPE (fn)))
result = build_op_call (fn, args, complain);
if (!result)
result = cp_build_function_call_vec (fn, args, complain);
if (processing_template_decl && result != error_mark_node)
{
if (INDIRECT_REF_P (result))
result = TREE_OPERAND (result, 0);
result = build_call_vec (TREE_TYPE (result), orig_fn, orig_args);
SET_EXPR_LOCATION (result, input_location);
KOENIG_LOOKUP_P (result) = koenig_p;
release_tree_vector (orig_args);
result = convert_from_reference (result);
}
if (is_overloaded_fn (orig_fn))
lookup_keep (get_fns (orig_fn), processing_template_decl);
return result;
}
cp_expr
finish_increment_expr (cp_expr expr, enum tree_code code)
{
location_t combined_loc = make_location (input_location,
expr.get_start (),
get_finish (input_location));
cp_expr result = build_x_unary_op (combined_loc, code, expr,
tf_warning_or_error);
result.set_location (combined_loc);
return result;
}
tree
finish_this_expr (void)
{
tree result = NULL_TREE;
if (current_class_ptr)
{
tree type = TREE_TYPE (current_class_ref);
if (LAMBDA_TYPE_P (type))
result = lambda_expr_this_capture (CLASSTYPE_LAMBDA_EXPR (type), true);
else
result = current_class_ptr;
}
if (result)
return rvalue (result);
tree fn = current_nonlambda_function ();
if (fn && DECL_STATIC_FUNCTION_P (fn))
error ("%<this%> is unavailable for static member functions");
else if (fn)
error ("invalid use of %<this%> in non-member function");
else
error ("invalid use of %<this%> at top level");
return error_mark_node;
}
tree
finish_pseudo_destructor_expr (tree object, tree scope, tree destructor,
location_t loc)
{
if (object == error_mark_node || destructor == error_mark_node)
return error_mark_node;
gcc_assert (TYPE_P (destructor));
if (!processing_template_decl)
{
if (scope == error_mark_node)
{
error_at (loc, "invalid qualifying scope in pseudo-destructor name");
return error_mark_node;
}
if (is_auto (destructor))
destructor = TREE_TYPE (object);
if (scope && TYPE_P (scope) && !check_dtor_name (scope, destructor))
{
error_at (loc,
"qualified type %qT does not match destructor name ~%qT",
scope, destructor);
return error_mark_node;
}
if (!same_type_ignoring_top_level_qualifiers_p (TREE_TYPE (object),
destructor))
{
error_at (loc, "%qE is not of type %qT", object, destructor);
return error_mark_node;
}
}
return build3_loc (loc, PSEUDO_DTOR_EXPR, void_type_node, object,
scope, destructor);
}
cp_expr
finish_unary_op_expr (location_t op_loc, enum tree_code code, cp_expr expr,
tsubst_flags_t complain)
{
location_t combined_loc = make_location (op_loc,
op_loc, expr.get_finish ());
cp_expr result = build_x_unary_op (combined_loc, code, expr, complain);
result.set_location (combined_loc);
if (result == error_mark_node)
return result;
if (!(complain & tf_warning))
return result;
tree result_ovl = result;
tree expr_ovl = expr;
if (!processing_template_decl)
expr_ovl = cp_fully_fold (expr_ovl);
if (!CONSTANT_CLASS_P (expr_ovl)
|| TREE_OVERFLOW_P (expr_ovl))
return result;
if (!processing_template_decl)
result_ovl = cp_fully_fold (result_ovl);
if (CONSTANT_CLASS_P (result_ovl) && TREE_OVERFLOW_P (result_ovl))
overflow_warning (combined_loc, result_ovl);
return result;
}
tree
finish_compound_literal (tree type, tree compound_literal,
tsubst_flags_t complain,
fcl_t fcl_context)
{
if (type == error_mark_node)
return error_mark_node;
if (TREE_CODE (type) == REFERENCE_TYPE)
{
compound_literal
= finish_compound_literal (TREE_TYPE (type), compound_literal,
complain, fcl_context);
return cp_build_c_cast (type, compound_literal, complain);
}
if (!TYPE_OBJ_P (type))
{
if (complain & tf_error)
error ("compound literal of non-object type %qT", type);
return error_mark_node;
}
if (tree anode = type_uses_auto (type))
if (CLASS_PLACEHOLDER_TEMPLATE (anode))
{
type = do_auto_deduction (type, compound_literal, anode, complain,
adc_variable_type);
if (type == error_mark_node)
return error_mark_node;
}
if (processing_template_decl)
{
TREE_TYPE (compound_literal) = type;
TREE_HAS_CONSTRUCTOR (compound_literal) = 1;
if (fcl_context == fcl_c99)
CONSTRUCTOR_C99_COMPOUND_LITERAL (compound_literal) = 1;
return compound_literal;
}
type = complete_type (type);
if (TYPE_NON_AGGREGATE_CLASS (type))
{
CONSTRUCTOR_IS_DIRECT_INIT (compound_literal) = 1;
compound_literal = build_tree_list (NULL_TREE, compound_literal);
return build_functional_cast (type, compound_literal, complain);
}
if (TREE_CODE (type) == ARRAY_TYPE
&& check_array_initializer (NULL_TREE, type, compound_literal))
return error_mark_node;
compound_literal = reshape_init (type, compound_literal, complain);
if (SCALAR_TYPE_P (type)
&& !BRACE_ENCLOSED_INITIALIZER_P (compound_literal)
&& !check_narrowing (type, compound_literal, complain))
return error_mark_node;
if (TREE_CODE (type) == ARRAY_TYPE
&& TYPE_DOMAIN (type) == NULL_TREE)
{
cp_complete_array_type_or_error (&type, compound_literal,
false, complain);
if (type == error_mark_node)
return error_mark_node;
}
compound_literal = digest_init_flags (type, compound_literal, LOOKUP_NORMAL,
complain);
if (TREE_CODE (compound_literal) == CONSTRUCTOR)
{
TREE_HAS_CONSTRUCTOR (compound_literal) = true;
if (fcl_context == fcl_c99)
CONSTRUCTOR_C99_COMPOUND_LITERAL (compound_literal) = 1;
}
if ((!at_function_scope_p () || CP_TYPE_CONST_P (type))
&& fcl_context == fcl_c99
&& TREE_CODE (type) == ARRAY_TYPE
&& !TYPE_HAS_NONTRIVIAL_DESTRUCTOR (type)
&& initializer_constant_valid_p (compound_literal, type))
{
tree decl = create_temporary_var (type);
DECL_INITIAL (decl) = compound_literal;
TREE_STATIC (decl) = 1;
if (literal_type_p (type) && CP_TYPE_CONST_NON_VOLATILE_P (type))
{
DECL_DECLARED_CONSTEXPR_P (decl) = true;
DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P (decl) = true;
TREE_CONSTANT (decl) = true;
}
cp_apply_type_quals_to_decl (cp_type_quals (type), decl);
decl = pushdecl_top_level (decl);
DECL_NAME (decl) = make_anon_name ();
SET_DECL_ASSEMBLER_NAME (decl, DECL_NAME (decl));
tree clean = cxx_maybe_build_cleanup (decl, complain);
if (clean == error_mark_node)
return error_mark_node;
return decl;
}
if (!VECTOR_TYPE_P (type))
compound_literal = get_target_expr_sfinae (compound_literal, complain);
return compound_literal;
}
tree
finish_fname (tree id)
{
tree decl;
decl = fname_decl (input_location, C_RID_CODE (id), id);
if (processing_template_decl && current_function_decl
&& decl != error_mark_node)
decl = DECL_NAME (decl);
return decl;
}
void
finish_translation_unit (void)
{
pop_everything ();
while (current_namespace != global_namespace)
pop_namespace ();
finish_fname_decls ();
}
tree
finish_template_type_parm (tree aggr, tree identifier)
{
if (aggr != class_type_node)
{
permerror (input_location, "template type parameters must use the keyword %<class%> or %<typename%>");
aggr = class_type_node;
}
return build_tree_list (aggr, identifier);
}
tree
finish_template_template_parm (tree aggr, tree identifier)
{
tree decl = build_decl (input_location,
TYPE_DECL, identifier, NULL_TREE);
tree tmpl = build_lang_decl (TEMPLATE_DECL, identifier, NULL_TREE);
DECL_TEMPLATE_PARMS (tmpl) = current_template_parms;
DECL_TEMPLATE_RESULT (tmpl) = decl;
DECL_ARTIFICIAL (decl) = 1;
tree reqs = TEMPLATE_PARMS_CONSTRAINTS (current_template_parms);
tree constr = build_constraints (reqs, NULL_TREE);
set_constraints (decl, constr);
end_template_decl ();
gcc_assert (DECL_TEMPLATE_PARMS (tmpl));
check_default_tmpl_args (decl, DECL_TEMPLATE_PARMS (tmpl), 
true, false,
0);
return finish_template_type_parm (aggr, tmpl);
}
tree
check_template_template_default_arg (tree argument)
{
if (TREE_CODE (argument) != TEMPLATE_DECL
&& TREE_CODE (argument) != TEMPLATE_TEMPLATE_PARM
&& TREE_CODE (argument) != UNBOUND_CLASS_TEMPLATE)
{
if (TREE_CODE (argument) == TYPE_DECL)
error ("invalid use of type %qT as a default value for a template "
"template-parameter", TREE_TYPE (argument));
else
error ("invalid default argument for a template template parameter");
return error_mark_node;
}
return argument;
}
tree
begin_class_definition (tree t)
{
if (error_operand_p (t) || error_operand_p (TYPE_MAIN_DECL (t)))
return error_mark_node;
if (processing_template_parmlist)
{
error ("definition of %q#T inside template parameter list", t);
return error_mark_node;
}
if (TREE_CODE (t) == RECORD_TYPE
&& !processing_template_decl)
{
tree ns = TYPE_CONTEXT (t);
if (ns && TREE_CODE (ns) == NAMESPACE_DECL
&& DECL_CONTEXT (ns) == std_node
&& DECL_NAME (ns)
&& id_equal (DECL_NAME (ns), "decimal"))
{
const char *n = TYPE_NAME_STRING (t);
if ((strcmp (n, "decimal32") == 0)
|| (strcmp (n, "decimal64") == 0)
|| (strcmp (n, "decimal128") == 0))
TYPE_TRANSPARENT_AGGR (t) = 1;
}
}
else if (TREE_CODE (t) == TYPENAME_TYPE)
{
error ("invalid definition of qualified type %qT", t);
t = error_mark_node;
}
if (t == error_mark_node || ! MAYBE_CLASS_TYPE_P (t))
{
t = make_class_type (RECORD_TYPE);
pushtag (make_anon_name (), t, ts_current);
}
if (TYPE_BEING_DEFINED (t))
{
t = make_class_type (TREE_CODE (t));
pushtag (TYPE_IDENTIFIER (t), t, ts_current);
}
maybe_process_partial_specialization (t);
pushclass (t);
TYPE_BEING_DEFINED (t) = 1;
class_binding_level->defining_class_p = 1;
if (flag_pack_struct)
{
tree v;
TYPE_PACKED (t) = 1;
for (v = TYPE_NEXT_VARIANT (t); v; v = TYPE_NEXT_VARIANT (v))
TYPE_PACKED (v) = 1;
}
if (! TYPE_UNNAMED_P (t))
{
struct c_fileinfo *finfo = \
get_fileinfo (LOCATION_FILE (input_location));
CLASSTYPE_INTERFACE_ONLY (t) = finfo->interface_only;
SET_CLASSTYPE_INTERFACE_UNKNOWN_X
(t, finfo->interface_unknown);
}
reset_specialization();
build_self_reference ();
return t;
}
void
finish_member_declaration (tree decl)
{
if (decl == error_mark_node || decl == NULL_TREE)
return;
if (decl == void_type_node)
return;
gcc_assert (DECL_CHAIN (decl) == NULL_TREE);
gcc_assert (TYPE_BEING_DEFINED (current_class_type)
|| LAMBDA_TYPE_P (TREE_TYPE (decl)));
TREE_PRIVATE (decl)
= (current_access_specifier == access_private_node);
TREE_PROTECTED (decl)
= (current_access_specifier == access_protected_node);
if (TREE_CODE (decl) == TEMPLATE_DECL)
{
TREE_PRIVATE (DECL_TEMPLATE_RESULT (decl)) = TREE_PRIVATE (decl);
TREE_PROTECTED (DECL_TEMPLATE_RESULT (decl)) = TREE_PROTECTED (decl);
}
if (TREE_CODE (decl) != CONST_DECL)
DECL_CONTEXT (decl) = current_class_type;
if (TREE_CODE (decl) == USING_DECL)
DECL_IGNORED_P (decl) = 1;
if (TREE_CODE (decl) == FIELD_DECL)
{
if (check_for_bare_parameter_packs (TREE_TYPE (decl)))
TREE_TYPE (decl) = error_mark_node;
if (check_for_bare_parameter_packs (DECL_ATTRIBUTES (decl)))
DECL_ATTRIBUTES (decl) = NULL_TREE;
}
if (DECL_LANG_SPECIFIC (decl))
SET_DECL_LANGUAGE (decl, lang_cplusplus);
bool add = false;
if (DECL_DECLARES_FUNCTION_P (decl))
add = add_method (current_class_type, decl, false);
else if (CLASSTYPE_LAMBDA_EXPR (current_class_type)
|| pushdecl_class_level (decl))
add = true;
if (add)
{
if (TREE_CODE (decl) == TYPE_DECL)
TYPE_FIELDS (current_class_type)
= chainon (TYPE_FIELDS (current_class_type), decl);
else
{
DECL_CHAIN (decl) = TYPE_FIELDS (current_class_type);
TYPE_FIELDS (current_class_type) = decl;
}
maybe_add_class_template_decl_list (current_class_type, decl,
0);
}
}
void
finish_template_decl (tree parms)
{
if (parms)
end_template_decl ();
else
end_specialization ();
}
static tree
fixup_template_type (tree type)
{
tree parms = current_template_parms;
int depth = template_class_depth (type);
for (int n = processing_template_decl; n > depth && parms; --n)
parms = TREE_CHAIN (parms);
if (!parms)
return type;
tree cur_reqs = TEMPLATE_PARMS_CONSTRAINTS (parms);
tree cur_constr = build_constraints (cur_reqs, NULL_TREE);
tree tmpl = CLASSTYPE_TI_TEMPLATE (type);
tree specs = DECL_TEMPLATE_SPECIALIZATIONS (tmpl);
while (specs)
{
tree spec_constr = get_constraints (TREE_VALUE (specs));
if (same_type_p (type, TREE_TYPE (specs))
&& equivalent_constraints (cur_constr, spec_constr))
return TREE_TYPE (specs);
specs = TREE_CHAIN (specs);
}
return type;
}
tree
finish_template_type (tree name, tree args, int entering_scope)
{
tree type;
type = lookup_template_class (name, args,
NULL_TREE, NULL_TREE, entering_scope,
tf_warning_or_error | tf_user);
if (flag_concepts
&& entering_scope
&& CLASS_TYPE_P (type)
&& CLASSTYPE_TEMPLATE_INFO (type)
&& dependent_type_p (type)
&& PRIMARY_TEMPLATE_P (CLASSTYPE_TI_TEMPLATE (type)))
type = fixup_template_type (type);
if (type == error_mark_node)
return type;
else if (CLASS_TYPE_P (type) && !alias_type_or_template_p (type))
return TYPE_STUB_DECL (type);
else
return TYPE_NAME (type);
}
tree
finish_base_specifier (tree base, tree access, bool virtual_p)
{
tree result;
if (base == error_mark_node)
{
error ("invalid base-class specification");
result = NULL_TREE;
}
else if (! MAYBE_CLASS_TYPE_P (base))
{
error ("%qT is not a class type", base);
result = NULL_TREE;
}
else
{
if (cp_type_quals (base) != 0)
{
base = TYPE_MAIN_VARIANT (base);
}
result = build_tree_list (access, base);
if (virtual_p)
TREE_TYPE (result) = integer_type_node;
}
return result;
}
tree
baselink_for_fns (tree fns)
{
tree scope;
tree cl;
if (BASELINK_P (fns) 
|| error_operand_p (fns))
return fns;
scope = ovl_scope (fns);
if (!CLASS_TYPE_P (scope))
return fns;
cl = currently_open_derived_class (scope);
if (!cl)
cl = scope;
cl = TYPE_BINFO (cl);
return build_baselink (cl, cl, fns, NULL_TREE);
}
static bool
outer_var_p (tree decl)
{
return ((VAR_P (decl) || TREE_CODE (decl) == PARM_DECL)
&& DECL_FUNCTION_SCOPE_P (decl)
&& DECL_NAME (decl)
&& (DECL_CONTEXT (decl) != current_function_decl
|| parsing_nsdmi ()));
}
bool
outer_automatic_var_p (tree decl)
{
return (outer_var_p (decl)
&& !TREE_STATIC (decl));
}
tree
process_outer_var_ref (tree decl, tsubst_flags_t complain, bool odr_use)
{
if (cp_unevaluated_operand)
return decl;
if (decl == error_mark_node)
return decl;
tree context = DECL_CONTEXT (decl);
tree containing_function = current_function_decl;
tree lambda_stack = NULL_TREE;
tree lambda_expr = NULL_TREE;
tree initializer = convert_from_reference (decl);
if (!mark_used (decl, complain))
return error_mark_node;
if (parsing_nsdmi ())
containing_function = NULL_TREE;
if (containing_function && LAMBDA_FUNCTION_P (containing_function))
{
tree var = decl;
while (is_normal_capture_proxy (var))
var = DECL_CAPTURED_VARIABLE (var);
tree d = retrieve_local_specialization (var);
if (d && d != decl && is_capture_proxy (d))
{
if (DECL_CONTEXT (d) == containing_function)
return d;
else
return process_outer_var_ref (d, complain, odr_use);
}
}
while (context != containing_function
&& containing_function
&& LAMBDA_FUNCTION_P (containing_function))
{
tree closure = DECL_CONTEXT (containing_function);
lambda_expr = CLASSTYPE_LAMBDA_EXPR (closure);
if (TYPE_CLASS_SCOPE_P (closure))
break;
if (LAMBDA_EXPR_DEFAULT_CAPTURE_MODE (lambda_expr)
== CPLD_NONE)
break;
lambda_stack = tree_cons (NULL_TREE,
lambda_expr,
lambda_stack);
containing_function
= decl_function_context (containing_function);
}
if (context == containing_function
&& dependent_type_p (TREE_TYPE (decl)))
return decl;
if (lambda_expr && VAR_P (decl)
&& DECL_ANON_UNION_VAR_P (decl))
{
if (complain & tf_error)
error ("cannot capture member %qD of anonymous union", decl);
return error_mark_node;
}
if (!odr_use && context == containing_function)
{
decl = add_default_capture (lambda_stack,
DECL_NAME (decl),
initializer);
}
else if (!odr_use && decl_constant_var_p (decl))
return decl;
else if (lambda_expr)
{
if (complain & tf_error)
{
error ("%qD is not captured", decl);
tree closure = LAMBDA_EXPR_CLOSURE (lambda_expr);
if (LAMBDA_EXPR_DEFAULT_CAPTURE_MODE (lambda_expr)
== CPLD_NONE)
inform (location_of (closure),
"the lambda has no capture-default");
else if (TYPE_CLASS_SCOPE_P (closure))
inform (UNKNOWN_LOCATION, "lambda in local class %q+T cannot "
"capture variables from the enclosing context",
TYPE_CONTEXT (closure));
inform (DECL_SOURCE_LOCATION (decl), "%q#D declared here", decl);
}
return error_mark_node;
}
else
{
if (complain & tf_error)
{
error (VAR_P (decl)
? G_("use of local variable with automatic storage from "
"containing function")
: G_("use of parameter from containing function"));
inform (DECL_SOURCE_LOCATION (decl), "%q#D declared here", decl);
}
return error_mark_node;
}
return decl;
}
cp_expr
finish_id_expression (tree id_expression,
tree decl,
tree scope,
cp_id_kind *idk,
bool integral_constant_expression_p,
bool allow_non_integral_constant_expression_p,
bool *non_integral_constant_expression_p,
bool template_p,
bool done,
bool address_p,
bool template_arg_p,
const char **error_msg,
location_t location)
{
decl = strip_using_decl (decl);
*idk = CP_ID_KIND_NONE;
*error_msg = NULL;
if (id_expression == error_mark_node)
return error_mark_node;
else if (TREE_CODE (decl) == TEMPLATE_ID_EXPR
|| TREE_CODE (decl) == TYPE_DECL)
;
else
{
if (decl == error_mark_node)
{
if (scope
&& (!TYPE_P (scope)
|| (!dependent_type_p (scope)
&& !(identifier_p (id_expression)
&& IDENTIFIER_CONV_OP_P (id_expression)
&& dependent_type_p (TREE_TYPE (id_expression))))))
{
qualified_name_lookup_error (scope, id_expression, decl, location);
return error_mark_node;
}
else if (!scope)
{
*idk = CP_ID_KIND_UNQUALIFIED;
return id_expression;
}
else
decl = id_expression;
}
else
decl = check_for_out_of_scope_variable (decl);
if (!scope && decl != error_mark_node && identifier_p (id_expression))
maybe_note_name_used_in_class (id_expression, decl);
if (processing_template_decl && cp_unevaluated_operand)
mark_type_use (decl);
if (outer_automatic_var_p (decl))
{
decl = process_outer_var_ref (decl, tf_warning_or_error);
if (decl == error_mark_node)
return error_mark_node;
}
if (TREE_CODE (decl) == PARM_DECL
&& DECL_CONTEXT (decl) == NULL_TREE
&& !cp_unevaluated_operand)
{
*error_msg = G_("use of parameter outside function body");
return error_mark_node;
}
}
if (TREE_CODE (decl) == TEMPLATE_DECL
&& !DECL_FUNCTION_TEMPLATE_P (decl))
{
*error_msg = G_("missing template arguments");
return error_mark_node;
}
else if (TREE_CODE (decl) == TYPE_DECL
|| TREE_CODE (decl) == NAMESPACE_DECL)
{
*error_msg = G_("expected primary-expression");
return error_mark_node;
}
if ((TREE_CODE (decl) == CONST_DECL && DECL_TEMPLATE_PARM_P (decl))
|| TREE_CODE (decl) == TEMPLATE_PARM_INDEX)
{
tree r;
*idk = CP_ID_KIND_NONE;
if (TREE_CODE (decl) == TEMPLATE_PARM_INDEX)
decl = TEMPLATE_PARM_DECL (decl);
r = convert_from_reference (DECL_INITIAL (decl));
if (integral_constant_expression_p
&& !dependent_type_p (TREE_TYPE (decl))
&& !(INTEGRAL_OR_ENUMERATION_TYPE_P (TREE_TYPE (r))))
{
if (!allow_non_integral_constant_expression_p)
error ("template parameter %qD of type %qT is not allowed in "
"an integral constant expression because it is not of "
"integral or enumeration type", decl, TREE_TYPE (decl));
*non_integral_constant_expression_p = true;
}
return r;
}
else
{
bool dependent_p = type_dependent_expression_p (decl);
*idk = (scope
? CP_ID_KIND_QUALIFIED
: (TREE_CODE (decl) == TEMPLATE_ID_EXPR
? CP_ID_KIND_TEMPLATE_ID
: (dependent_p
? CP_ID_KIND_UNQUALIFIED_DEPENDENT
: CP_ID_KIND_UNQUALIFIED)));
if (dependent_p
&& DECL_P (decl)
&& any_dependent_type_attributes_p (DECL_ATTRIBUTES (decl)))
return id_expression;
if (TREE_CODE (decl) == NAMESPACE_DECL)
{
error ("use of namespace %qD as expression", decl);
return error_mark_node;
}
else if (DECL_CLASS_TEMPLATE_P (decl))
{
error ("use of class template %qT as expression", decl);
return error_mark_node;
}
else if (TREE_CODE (decl) == TREE_LIST)
{
error ("request for member %qD is ambiguous in "
"multiple inheritance lattice", id_expression);
print_candidates (decl);
return error_mark_node;
}
if ((VAR_P (decl)
|| TREE_CODE (decl) == PARM_DECL
|| TREE_CODE (decl) == CONST_DECL
|| TREE_CODE (decl) == RESULT_DECL)
&& !mark_used (decl))
return error_mark_node;
if (! error_operand_p (decl)
&& !dependent_p
&& integral_constant_expression_p
&& ! decl_constant_var_p (decl)
&& TREE_CODE (decl) != CONST_DECL
&& ! builtin_valid_in_constant_expr_p (decl))
{
if (!allow_non_integral_constant_expression_p)
{
error ("%qD cannot appear in a constant-expression", decl);
return error_mark_node;
}
*non_integral_constant_expression_p = true;
}
tree wrap;
if (VAR_P (decl)
&& !cp_unevaluated_operand
&& !processing_template_decl
&& (TREE_STATIC (decl) || DECL_EXTERNAL (decl))
&& CP_DECL_THREAD_LOCAL_P (decl)
&& (wrap = get_tls_wrapper_fn (decl)))
{
decl = build_cxx_call (wrap, 0, NULL, tf_warning_or_error);
}
else if (TREE_CODE (decl) == TEMPLATE_ID_EXPR
&& !dependent_p
&& variable_template_p (TREE_OPERAND (decl, 0)))
{
decl = finish_template_variable (decl);
mark_used (decl);
decl = convert_from_reference (decl);
}
else if (scope)
{
if (TREE_CODE (decl) == SCOPE_REF)
{
gcc_assert (same_type_p (scope, TREE_OPERAND (decl, 0)));
decl = TREE_OPERAND (decl, 1);
}
decl = (adjust_result_of_qualified_name_lookup
(decl, scope, current_nonlambda_class_type()));
if (TREE_CODE (decl) == FUNCTION_DECL)
mark_used (decl);
if (TYPE_P (scope))
decl = finish_qualified_id_expr (scope,
decl,
done,
address_p,
template_p,
template_arg_p,
tf_warning_or_error);
else
decl = convert_from_reference (decl);
}
else if (TREE_CODE (decl) == FIELD_DECL)
{
push_deferring_access_checks (dk_no_check);
decl = finish_non_static_data_member (decl, NULL_TREE,
NULL_TREE);
pop_deferring_access_checks ();
}
else if (is_overloaded_fn (decl))
{
tree first_fn = get_first_fn (decl);
if (TREE_CODE (first_fn) == TEMPLATE_DECL)
first_fn = DECL_TEMPLATE_RESULT (first_fn);
if (done
&& !really_overloaded_fn (decl)
&& !mark_used (first_fn))
return error_mark_node;
if (!template_arg_p
&& (TREE_CODE (first_fn) == USING_DECL
|| (TREE_CODE (first_fn) == FUNCTION_DECL
&& DECL_FUNCTION_MEMBER_P (first_fn)
&& !shared_member_p (decl))))
{
decl = maybe_dummy_object (DECL_CONTEXT (first_fn), 0);
return finish_class_member_access_expr (decl, id_expression,
false,
tf_warning_or_error);
}
decl = baselink_for_fns (decl);
}
else
{
if (DECL_P (decl) && DECL_NONLOCAL (decl)
&& DECL_CLASS_SCOPE_P (decl))
{
tree context = context_for_name_lookup (decl); 
if (context != current_class_type)
{
tree path = currently_open_derived_class (context);
perform_or_defer_access_check (TYPE_BINFO (path),
decl, decl,
tf_warning_or_error);
}
}
decl = convert_from_reference (decl);
}
}
return cp_expr (decl, location);
}
tree
finish_typeof (tree expr)
{
tree type;
if (type_dependent_expression_p (expr))
{
type = cxx_make_type (TYPEOF_TYPE);
TYPEOF_TYPE_EXPR (type) = expr;
SET_TYPE_STRUCTURAL_EQUALITY (type);
return type;
}
expr = mark_type_use (expr);
type = unlowered_expr_type (expr);
if (!type || type == unknown_type_node)
{
error ("type of %qE is unknown", expr);
return error_mark_node;
}
return type;
}
tree
finish_underlying_type (tree type)
{
tree underlying_type;
if (processing_template_decl)
{
underlying_type = cxx_make_type (UNDERLYING_TYPE);
UNDERLYING_TYPE_TYPE (underlying_type) = type;
SET_TYPE_STRUCTURAL_EQUALITY (underlying_type);
return underlying_type;
}
if (!complete_type_or_else (type, NULL_TREE))
return error_mark_node;
if (TREE_CODE (type) != ENUMERAL_TYPE)
{
error ("%qT is not an enumeration type", type);
return error_mark_node;
}
underlying_type = ENUM_UNDERLYING_TYPE (type);
if (!ENUM_FIXED_UNDERLYING_TYPE_P (type))
underlying_type
= c_common_type_for_mode (TYPE_MODE (underlying_type),
TYPE_UNSIGNED (underlying_type));
return underlying_type;
}
tree
calculate_direct_bases (tree type, tsubst_flags_t complain)
{
if (!complete_type_or_maybe_complain (type, NULL_TREE, complain)
|| !NON_UNION_CLASS_TYPE_P (type))
return make_tree_vec (0);
vec<tree, va_gc> *vector = make_tree_vector ();
vec<tree, va_gc> *base_binfos = BINFO_BASE_BINFOS (TYPE_BINFO (type));
tree binfo;
unsigned i;
for (i = 0; base_binfos->iterate (i, &binfo); i++)
if (BINFO_VIRTUAL_P (binfo))
vec_safe_push (vector, binfo);
for (i = 0; base_binfos->iterate (i, &binfo); i++)
if (!BINFO_VIRTUAL_P (binfo))
vec_safe_push (vector, binfo);
tree bases_vec = make_tree_vec (vector->length ());
for (i = 0; i < vector->length (); ++i)
TREE_VEC_ELT (bases_vec, i) = BINFO_TYPE ((*vector)[i]);
release_tree_vector (vector);
return bases_vec;
}
static tree
dfs_calculate_bases_pre (tree binfo, void * )
{
return BINFO_VIRTUAL_P (binfo) ? dfs_skip_bases : NULL_TREE;
}
static tree
dfs_calculate_bases_post (tree binfo, void *data_)
{
vec<tree, va_gc> **data = ((vec<tree, va_gc> **) data_);
if (!BINFO_VIRTUAL_P (binfo))
vec_safe_push (*data, BINFO_TYPE (binfo));
return NULL_TREE;
}
static vec<tree, va_gc> *
calculate_bases_helper (tree type)
{
vec<tree, va_gc> *vector = make_tree_vector ();
if (TYPE_BINFO (type))
dfs_walk_all (TYPE_BINFO (type),
dfs_calculate_bases_pre, dfs_calculate_bases_post, &vector);
return vector;
}
tree
calculate_bases (tree type, tsubst_flags_t complain)
{
if (!complete_type_or_maybe_complain (type, NULL_TREE, complain)
|| !NON_UNION_CLASS_TYPE_P (type))
return make_tree_vec (0);
vec<tree, va_gc> *vector = make_tree_vector ();
tree bases_vec = NULL_TREE;
unsigned i;
vec<tree, va_gc> *vbases;
vec<tree, va_gc> *nonvbases;
tree binfo;
for (vbases = CLASSTYPE_VBASECLASSES (type), i = 0;
vec_safe_iterate (vbases, i, &binfo); i++)
{
vec<tree, va_gc> *vbase_bases
= calculate_bases_helper (BINFO_TYPE (binfo));
vec_safe_splice (vector, vbase_bases);
release_tree_vector (vbase_bases);
}
nonvbases = calculate_bases_helper (type);
vec_safe_splice (vector, nonvbases);
release_tree_vector (nonvbases);
if (vector->length () > 1)
{
bases_vec = make_tree_vec (vector->length () - 1);
for (i = 0; i < vector->length () - 1; ++i)
TREE_VEC_ELT (bases_vec, i) = (*vector)[i];
}
else
bases_vec = make_tree_vec (0);
release_tree_vector (vector);
return bases_vec;
}
tree
finish_bases (tree type, bool direct)
{
tree bases = NULL_TREE;
if (!processing_template_decl)
{
error ("Parameter pack __bases only valid in template declaration");
return error_mark_node;
}
bases = cxx_make_type (BASES);
BASES_TYPE (bases) = type;
BASES_DIRECT (bases) = direct;
SET_TYPE_STRUCTURAL_EQUALITY (bases);
return bases;
}
tree
finish_offsetof (tree object_ptr, tree expr, location_t loc)
{
if (processing_template_decl)
{
expr = build2 (OFFSETOF_EXPR, size_type_node, expr, object_ptr);
SET_EXPR_LOCATION (expr, loc);
return expr;
}
if (TREE_CODE (expr) == PSEUDO_DTOR_EXPR)
{
error ("cannot apply %<offsetof%> to destructor %<~%T%>",
TREE_OPERAND (expr, 2));
return error_mark_node;
}
if (TREE_CODE (TREE_TYPE (expr)) == FUNCTION_TYPE
|| TREE_CODE (TREE_TYPE (expr)) == METHOD_TYPE
|| TREE_TYPE (expr) == unknown_type_node)
{
while (TREE_CODE (expr) == COMPONENT_REF
|| TREE_CODE (expr) == COMPOUND_EXPR)
expr = TREE_OPERAND (expr, 1);
if (DECL_P (expr))
{
error ("cannot apply %<offsetof%> to member function %qD", expr);
inform (DECL_SOURCE_LOCATION (expr), "declared here");
}
else
error ("cannot apply %<offsetof%> to member function");
return error_mark_node;
}
if (TREE_CODE (expr) == CONST_DECL)
{
error ("cannot apply %<offsetof%> to an enumerator %qD", expr);
return error_mark_node;
}
if (REFERENCE_REF_P (expr))
expr = TREE_OPERAND (expr, 0);
if (!complete_type_or_else (TREE_TYPE (TREE_TYPE (object_ptr)), object_ptr))
return error_mark_node;
if (warn_invalid_offsetof
&& CLASS_TYPE_P (TREE_TYPE (TREE_TYPE (object_ptr)))
&& CLASSTYPE_NON_STD_LAYOUT (TREE_TYPE (TREE_TYPE (object_ptr)))
&& cp_unevaluated_operand == 0)
warning_at (loc, OPT_Winvalid_offsetof, "offsetof within "
"non-standard-layout type %qT is conditionally-supported",
TREE_TYPE (TREE_TYPE (object_ptr)));
return fold_offsetof (expr);
}
void
simplify_aggr_init_expr (tree *tp)
{
tree aggr_init_expr = *tp;
tree fn = AGGR_INIT_EXPR_FN (aggr_init_expr);
tree slot = AGGR_INIT_EXPR_SLOT (aggr_init_expr);
tree type = TREE_TYPE (slot);
tree call_expr;
enum style_t { ctor, arg, pcc } style;
if (AGGR_INIT_VIA_CTOR_P (aggr_init_expr))
style = ctor;
#ifdef PCC_STATIC_STRUCT_RETURN
else if (1)
style = pcc;
#endif
else
{
gcc_assert (TREE_ADDRESSABLE (type));
style = arg;
}
call_expr = build_call_array_loc (input_location,
TREE_TYPE (TREE_TYPE (TREE_TYPE (fn))),
fn,
aggr_init_expr_nargs (aggr_init_expr),
AGGR_INIT_EXPR_ARGP (aggr_init_expr));
TREE_NOTHROW (call_expr) = TREE_NOTHROW (aggr_init_expr);
CALL_FROM_THUNK_P (call_expr) = AGGR_INIT_FROM_THUNK_P (aggr_init_expr);
CALL_EXPR_OPERATOR_SYNTAX (call_expr)
= CALL_EXPR_OPERATOR_SYNTAX (aggr_init_expr);
CALL_EXPR_ORDERED_ARGS (call_expr) = CALL_EXPR_ORDERED_ARGS (aggr_init_expr);
CALL_EXPR_REVERSE_ARGS (call_expr) = CALL_EXPR_REVERSE_ARGS (aggr_init_expr);
if (style == ctor)
{
cxx_mark_addressable (slot);
CALL_EXPR_ARG (call_expr, 0) =
build1 (ADDR_EXPR, build_pointer_type (type), slot);
}
else if (style == arg)
{
cxx_mark_addressable (slot);
CALL_EXPR_RETURN_SLOT_OPT (call_expr) = true;
call_expr = build2 (INIT_EXPR, TREE_TYPE (call_expr), slot, call_expr);
}
else if (style == pcc)
{
push_deferring_access_checks (dk_no_check);
call_expr = build_aggr_init (slot, call_expr,
DIRECT_BIND | LOOKUP_ONLYCONVERTING,
tf_warning_or_error);
pop_deferring_access_checks ();
call_expr = build2 (COMPOUND_EXPR, TREE_TYPE (slot), call_expr, slot);
}
if (AGGR_INIT_ZERO_FIRST (aggr_init_expr))
{
tree init = build_zero_init (type, NULL_TREE,
false);
init = build2 (INIT_EXPR, void_type_node, slot, init);
call_expr = build2 (COMPOUND_EXPR, TREE_TYPE (call_expr),
init, call_expr);
}
*tp = call_expr;
}
void
emit_associated_thunks (tree fn)
{
if (DECL_VIRTUAL_P (fn)
&& ! DECL_REALLY_EXTERN (fn))
{
tree thunk;
for (thunk = DECL_THUNKS (fn); thunk; thunk = DECL_CHAIN (thunk))
{
if (!THUNK_ALIAS (thunk))
{
use_thunk (thunk, 1);
if (DECL_RESULT_THUNK_P (thunk))
{
tree probe;
for (probe = DECL_THUNKS (thunk);
probe; probe = DECL_CHAIN (probe))
use_thunk (probe, 1);
}
}
else
gcc_assert (!DECL_THUNKS (thunk));
}
}
}
bool
expand_or_defer_fn_1 (tree fn)
{
if (processing_template_decl)
{
if (!function_depth)
ggc_collect ();
return false;
}
gcc_assert (DECL_SAVED_TREE (fn));
if (DECL_DECLARED_INLINE_P (fn) || DECL_IMPLICIT_INSTANTIATION (fn))
{
if (DECL_INTERFACE_KNOWN (fn))
;
else if (!at_eof)
tentative_decl_linkage (fn);
else
import_export_decl (fn);
if (DECL_DECLARED_INLINE_P (fn)
&& !DECL_REALLY_EXTERN (fn)
&& (flag_keep_inline_functions
|| (flag_keep_inline_dllexport
&& lookup_attribute ("dllexport", DECL_ATTRIBUTES (fn)))))
{
mark_needed (fn);
DECL_EXTERNAL (fn) = 0;
}
}
if (maybe_clone_body (fn))
{
TREE_ASM_WRITTEN (fn) = 1;
if (!DECL_DECLARED_CONSTEXPR_P (fn))
DECL_SAVED_TREE (fn) = NULL_TREE;
return false;
}
if (flag_syntax_only)
return false;
return true;
}
void
expand_or_defer_fn (tree fn)
{
if (expand_or_defer_fn_1 (fn))
{
function_depth++;
cgraph_node::finalize_function (fn, function_depth > 1);
emit_associated_thunks (fn);
function_depth--;
}
}
struct nrv_data
{
nrv_data () : visited (37) {}
tree var;
tree result;
hash_table<nofree_ptr_hash <tree_node> > visited;
};
static tree
finalize_nrv_r (tree* tp, int* walk_subtrees, void* data)
{
struct nrv_data *dp = (struct nrv_data *)data;
tree_node **slot;
if (TYPE_P (*tp))
*walk_subtrees = 0;
else if (TREE_CODE (*tp) == RETURN_EXPR)
TREE_OPERAND (*tp, 0) = dp->result;
else if (TREE_CODE (*tp) == CLEANUP_STMT
&& CLEANUP_DECL (*tp) == dp->var)
CLEANUP_EH_ONLY (*tp) = 1;
else if (TREE_CODE (*tp) == DECL_EXPR
&& DECL_EXPR_DECL (*tp) == dp->var)
{
tree init;
if (DECL_INITIAL (dp->var)
&& DECL_INITIAL (dp->var) != error_mark_node)
init = build2 (INIT_EXPR, void_type_node, dp->result,
DECL_INITIAL (dp->var));
else
init = build_empty_stmt (EXPR_LOCATION (*tp));
DECL_INITIAL (dp->var) = NULL_TREE;
SET_EXPR_LOCATION (init, EXPR_LOCATION (*tp));
*tp = init;
}
else if (*tp == dp->var)
*tp = dp->result;
slot = dp->visited.find_slot (*tp, INSERT);
if (*slot)
*walk_subtrees = 0;
else
*slot = *tp;
return NULL_TREE;
}
void
finalize_nrv (tree *tp, tree var, tree result)
{
struct nrv_data data;
DECL_NAME (result) = DECL_NAME (var);
TREE_ADDRESSABLE (result) = TREE_ADDRESSABLE (var);
SET_DECL_VALUE_EXPR (var, result);
DECL_HAS_VALUE_EXPR_P (var) = 1;
data.var = var;
data.result = result;
cp_walk_tree (tp, finalize_nrv_r, &data, 0);
}

bool
cxx_omp_create_clause_info (tree c, tree type, bool need_default_ctor,
bool need_copy_ctor, bool need_copy_assignment,
bool need_dtor)
{
int save_errorcount = errorcount;
tree info, t;
info = make_tree_vec (3);
CP_OMP_CLAUSE_INFO (c) = info;
if (need_default_ctor || need_copy_ctor)
{
if (need_default_ctor)
t = get_default_ctor (type);
else
t = get_copy_ctor (type, tf_warning_or_error);
if (t && !trivial_fn_p (t))
TREE_VEC_ELT (info, 0) = t;
}
if (need_dtor && TYPE_HAS_NONTRIVIAL_DESTRUCTOR (type))
TREE_VEC_ELT (info, 1) = get_dtor (type, tf_warning_or_error);
if (need_copy_assignment)
{
t = get_copy_assign (type);
if (t && !trivial_fn_p (t))
TREE_VEC_ELT (info, 2) = t;
}
return errorcount != save_errorcount;
}
static tree
omp_clause_decl_field (tree decl)
{
if (VAR_P (decl)
&& DECL_HAS_VALUE_EXPR_P (decl)
&& DECL_ARTIFICIAL (decl)
&& DECL_LANG_SPECIFIC (decl)
&& DECL_OMP_PRIVATIZED_MEMBER (decl))
{
tree f = DECL_VALUE_EXPR (decl);
if (INDIRECT_REF_P (f))
f = TREE_OPERAND (f, 0);
if (TREE_CODE (f) == COMPONENT_REF)
{
f = TREE_OPERAND (f, 1);
gcc_assert (TREE_CODE (f) == FIELD_DECL);
return f;
}
}
return NULL_TREE;
}
static tree
omp_clause_printable_decl (tree decl)
{
tree t = omp_clause_decl_field (decl);
if (t)
return t;
return decl;
}
static void
omp_note_field_privatization (tree f, tree t)
{
if (!omp_private_member_map)
omp_private_member_map = new hash_map<tree, tree>;
tree &v = omp_private_member_map->get_or_insert (f);
if (v == NULL_TREE)
{
v = t;
omp_private_member_vec.safe_push (f);
omp_private_member_vec.safe_push (integer_zero_node);
}
}
tree
omp_privatize_field (tree t, bool shared)
{
tree m = finish_non_static_data_member (t, NULL_TREE, NULL_TREE);
if (m == error_mark_node)
return error_mark_node;
if (!omp_private_member_map && !shared)
omp_private_member_map = new hash_map<tree, tree>;
if (TREE_CODE (TREE_TYPE (t)) == REFERENCE_TYPE)
{
gcc_assert (INDIRECT_REF_P (m));
m = TREE_OPERAND (m, 0);
}
tree vb = NULL_TREE;
tree &v = shared ? vb : omp_private_member_map->get_or_insert (t);
if (v == NULL_TREE)
{
v = create_temporary_var (TREE_TYPE (m));
retrofit_lang_decl (v);
DECL_OMP_PRIVATIZED_MEMBER (v) = 1;
SET_DECL_VALUE_EXPR (v, m);
DECL_HAS_VALUE_EXPR_P (v) = 1;
if (!shared)
omp_private_member_vec.safe_push (t);
}
return v;
}
static tree
handle_omp_array_sections_1 (tree c, tree t, vec<tree> &types,
bool &maybe_zero_len, unsigned int &first_non_one,
enum c_omp_region_type ort)
{
tree ret, low_bound, length, type;
if (TREE_CODE (t) != TREE_LIST)
{
if (error_operand_p (t))
return error_mark_node;
if (REFERENCE_REF_P (t)
&& TREE_CODE (TREE_OPERAND (t, 0)) == COMPONENT_REF)
t = TREE_OPERAND (t, 0);
ret = t;
if (TREE_CODE (t) == COMPONENT_REF
&& ort == C_ORT_OMP
&& (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_MAP
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_TO
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_FROM)
&& !type_dependent_expression_p (t))
{
if (TREE_CODE (TREE_OPERAND (t, 1)) == FIELD_DECL
&& DECL_BIT_FIELD (TREE_OPERAND (t, 1)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"bit-field %qE in %qs clause",
t, omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
while (TREE_CODE (t) == COMPONENT_REF)
{
if (TREE_TYPE (TREE_OPERAND (t, 0))
&& TREE_CODE (TREE_TYPE (TREE_OPERAND (t, 0))) == UNION_TYPE)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is a member of a union", t);
return error_mark_node;
}
t = TREE_OPERAND (t, 0);
}
if (REFERENCE_REF_P (t))
t = TREE_OPERAND (t, 0);
}
if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL)
{
if (processing_template_decl && TREE_CODE (t) != OVERLOAD)
return NULL_TREE;
if (DECL_P (t))
error_at (OMP_CLAUSE_LOCATION (c),
"%qD is not a variable in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
else
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is not a variable in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
else if (TREE_CODE (t) == PARM_DECL
&& DECL_ARTIFICIAL (t)
&& DECL_NAME (t) == this_identifier)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<this%> allowed in OpenMP only in %<declare simd%>"
" clauses");
return error_mark_node;
}
else if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND
&& VAR_P (t) && CP_DECL_THREAD_LOCAL_P (t))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qD is threadprivate variable in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
if (type_dependent_expression_p (ret))
return NULL_TREE;
ret = convert_from_reference (ret);
return ret;
}
if (ort == C_ORT_OMP
&& OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION
&& TREE_CODE (TREE_CHAIN (t)) == FIELD_DECL)
TREE_CHAIN (t) = omp_privatize_field (TREE_CHAIN (t), false);
ret = handle_omp_array_sections_1 (c, TREE_CHAIN (t), types,
maybe_zero_len, first_non_one, ort);
if (ret == error_mark_node || ret == NULL_TREE)
return ret;
type = TREE_TYPE (ret);
low_bound = TREE_PURPOSE (t);
length = TREE_VALUE (t);
if ((low_bound && type_dependent_expression_p (low_bound))
|| (length && type_dependent_expression_p (length)))
return NULL_TREE;
if (low_bound == error_mark_node || length == error_mark_node)
return error_mark_node;
if (low_bound && !INTEGRAL_TYPE_P (TREE_TYPE (low_bound)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"low bound %qE of array section does not have integral type",
low_bound);
return error_mark_node;
}
if (length && !INTEGRAL_TYPE_P (TREE_TYPE (length)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"length %qE of array section does not have integral type",
length);
return error_mark_node;
}
if (low_bound)
low_bound = mark_rvalue_use (low_bound);
if (length)
length = mark_rvalue_use (length);
if (length)
length = fold_simple (length);
if (low_bound)
low_bound = fold_simple (low_bound);
if (low_bound
&& TREE_CODE (low_bound) == INTEGER_CST
&& TYPE_PRECISION (TREE_TYPE (low_bound))
> TYPE_PRECISION (sizetype))
low_bound = fold_convert (sizetype, low_bound);
if (length
&& TREE_CODE (length) == INTEGER_CST
&& TYPE_PRECISION (TREE_TYPE (length))
> TYPE_PRECISION (sizetype))
length = fold_convert (sizetype, length);
if (low_bound == NULL_TREE)
low_bound = integer_zero_node;
if (length != NULL_TREE)
{
if (!integer_nonzerop (length))
{
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_DEPEND
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION)
{
if (integer_zerop (length))
{
error_at (OMP_CLAUSE_LOCATION (c),
"zero length array section in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
}
else
maybe_zero_len = true;
}
if (first_non_one == types.length ()
&& (TREE_CODE (length) != INTEGER_CST || integer_onep (length)))
first_non_one++;
}
if (TREE_CODE (type) == ARRAY_TYPE)
{
if (length == NULL_TREE
&& (TYPE_DOMAIN (type) == NULL_TREE
|| TYPE_MAX_VALUE (TYPE_DOMAIN (type)) == NULL_TREE))
{
error_at (OMP_CLAUSE_LOCATION (c),
"for unknown bound array type length expression must "
"be specified");
return error_mark_node;
}
if (TREE_CODE (low_bound) == INTEGER_CST
&& tree_int_cst_sgn (low_bound) == -1)
{
error_at (OMP_CLAUSE_LOCATION (c),
"negative low bound in array section in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
if (length != NULL_TREE
&& TREE_CODE (length) == INTEGER_CST
&& tree_int_cst_sgn (length) == -1)
{
error_at (OMP_CLAUSE_LOCATION (c),
"negative length in array section in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
if (TYPE_DOMAIN (type)
&& TYPE_MAX_VALUE (TYPE_DOMAIN (type))
&& TREE_CODE (TYPE_MAX_VALUE (TYPE_DOMAIN (type)))
== INTEGER_CST)
{
tree size
= fold_convert (sizetype, TYPE_MAX_VALUE (TYPE_DOMAIN (type)));
size = size_binop (PLUS_EXPR, size, size_one_node);
if (TREE_CODE (low_bound) == INTEGER_CST)
{
if (tree_int_cst_lt (size, low_bound))
{
error_at (OMP_CLAUSE_LOCATION (c),
"low bound %qE above array section size "
"in %qs clause", low_bound,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
if (tree_int_cst_equal (size, low_bound))
{
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_DEPEND
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION)
{
error_at (OMP_CLAUSE_LOCATION (c),
"zero length array section in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
maybe_zero_len = true;
}
else if (length == NULL_TREE
&& first_non_one == types.length ()
&& tree_int_cst_equal
(TYPE_MAX_VALUE (TYPE_DOMAIN (type)),
low_bound))
first_non_one++;
}
else if (length == NULL_TREE)
{
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND
&& OMP_CLAUSE_CODE (c) != OMP_CLAUSE_REDUCTION)
maybe_zero_len = true;
if (first_non_one == types.length ())
first_non_one++;
}
if (length && TREE_CODE (length) == INTEGER_CST)
{
if (tree_int_cst_lt (size, length))
{
error_at (OMP_CLAUSE_LOCATION (c),
"length %qE above array section size "
"in %qs clause", length,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
if (TREE_CODE (low_bound) == INTEGER_CST)
{
tree lbpluslen
= size_binop (PLUS_EXPR,
fold_convert (sizetype, low_bound),
fold_convert (sizetype, length));
if (TREE_CODE (lbpluslen) == INTEGER_CST
&& tree_int_cst_lt (size, lbpluslen))
{
error_at (OMP_CLAUSE_LOCATION (c),
"high bound %qE above array section size "
"in %qs clause", lbpluslen,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
}
}
}
else if (length == NULL_TREE)
{
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND
&& OMP_CLAUSE_CODE (c) != OMP_CLAUSE_REDUCTION)
maybe_zero_len = true;
if (first_non_one == types.length ())
first_non_one++;
}
if (length == NULL_TREE && OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND)
{
tree lb = cp_save_expr (low_bound);
if (lb != low_bound)
{
TREE_PURPOSE (t) = lb;
low_bound = lb;
}
}
}
else if (TREE_CODE (type) == POINTER_TYPE)
{
if (length == NULL_TREE)
{
error_at (OMP_CLAUSE_LOCATION (c),
"for pointer type length expression must be specified");
return error_mark_node;
}
if (length != NULL_TREE
&& TREE_CODE (length) == INTEGER_CST
&& tree_int_cst_sgn (length) == -1)
{
error_at (OMP_CLAUSE_LOCATION (c),
"negative length in array section in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND
&& TREE_CODE (TREE_CHAIN (t)) == TREE_LIST)
{
error_at (OMP_CLAUSE_LOCATION (c),
"array section is not contiguous in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
}
else
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE does not have pointer or array type", ret);
return error_mark_node;
}
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND)
types.safe_push (TREE_TYPE (ret));
tree lb = cp_save_expr (low_bound);
if (lb != low_bound)
{
TREE_PURPOSE (t) = lb;
low_bound = lb;
}
ret = grok_array_decl (OMP_CLAUSE_LOCATION (c), ret, low_bound, false);
return ret;
}
static bool
handle_omp_array_sections (tree c, enum c_omp_region_type ort)
{
bool maybe_zero_len = false;
unsigned int first_non_one = 0;
auto_vec<tree, 10> types;
tree first = handle_omp_array_sections_1 (c, OMP_CLAUSE_DECL (c), types,
maybe_zero_len, first_non_one,
ort);
if (first == error_mark_node)
return true;
if (first == NULL_TREE)
return false;
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_DEPEND)
{
tree t = OMP_CLAUSE_DECL (c);
tree tem = NULL_TREE;
if (processing_template_decl)
return false;
while (TREE_CODE (t) == TREE_LIST)
{
if (TREE_VALUE (t) && TREE_SIDE_EFFECTS (TREE_VALUE (t)))
{
if (tem == NULL_TREE)
tem = TREE_VALUE (t);
else
tem = build2 (COMPOUND_EXPR, TREE_TYPE (tem),
TREE_VALUE (t), tem);
}
t = TREE_CHAIN (t);
}
if (tem)
first = build2 (COMPOUND_EXPR, TREE_TYPE (first), tem, first);
OMP_CLAUSE_DECL (c) = first;
}
else
{
unsigned int num = types.length (), i;
tree t, side_effects = NULL_TREE, size = NULL_TREE;
tree condition = NULL_TREE;
if (int_size_in_bytes (TREE_TYPE (first)) <= 0)
maybe_zero_len = true;
if (processing_template_decl && maybe_zero_len)
return false;
for (i = num, t = OMP_CLAUSE_DECL (c); i > 0;
t = TREE_CHAIN (t))
{
tree low_bound = TREE_PURPOSE (t);
tree length = TREE_VALUE (t);
i--;
if (low_bound
&& TREE_CODE (low_bound) == INTEGER_CST
&& TYPE_PRECISION (TREE_TYPE (low_bound))
> TYPE_PRECISION (sizetype))
low_bound = fold_convert (sizetype, low_bound);
if (length
&& TREE_CODE (length) == INTEGER_CST
&& TYPE_PRECISION (TREE_TYPE (length))
> TYPE_PRECISION (sizetype))
length = fold_convert (sizetype, length);
if (low_bound == NULL_TREE)
low_bound = integer_zero_node;
if (!maybe_zero_len && i > first_non_one)
{
if (integer_nonzerop (low_bound))
goto do_warn_noncontiguous;
if (length != NULL_TREE
&& TREE_CODE (length) == INTEGER_CST
&& TYPE_DOMAIN (types[i])
&& TYPE_MAX_VALUE (TYPE_DOMAIN (types[i]))
&& TREE_CODE (TYPE_MAX_VALUE (TYPE_DOMAIN (types[i])))
== INTEGER_CST)
{
tree size;
size = size_binop (PLUS_EXPR,
TYPE_MAX_VALUE (TYPE_DOMAIN (types[i])),
size_one_node);
if (!tree_int_cst_equal (length, size))
{
do_warn_noncontiguous:
error_at (OMP_CLAUSE_LOCATION (c),
"array section is not contiguous in %qs "
"clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return true;
}
}
if (!processing_template_decl
&& length != NULL_TREE
&& TREE_SIDE_EFFECTS (length))
{
if (side_effects == NULL_TREE)
side_effects = length;
else
side_effects = build2 (COMPOUND_EXPR,
TREE_TYPE (side_effects),
length, side_effects);
}
}
else if (processing_template_decl)
continue;
else
{
tree l;
if (i > first_non_one
&& ((length && integer_nonzerop (length))
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION))
continue;
if (length)
l = fold_convert (sizetype, length);
else
{
l = size_binop (PLUS_EXPR,
TYPE_MAX_VALUE (TYPE_DOMAIN (types[i])),
size_one_node);
l = size_binop (MINUS_EXPR, l,
fold_convert (sizetype, low_bound));
}
if (i > first_non_one)
{
l = fold_build2 (NE_EXPR, boolean_type_node, l,
size_zero_node);
if (condition == NULL_TREE)
condition = l;
else
condition = fold_build2 (BIT_AND_EXPR, boolean_type_node,
l, condition);
}
else if (size == NULL_TREE)
{
size = size_in_bytes (TREE_TYPE (types[i]));
tree eltype = TREE_TYPE (types[num - 1]);
while (TREE_CODE (eltype) == ARRAY_TYPE)
eltype = TREE_TYPE (eltype);
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION)
size = size_binop (EXACT_DIV_EXPR, size,
size_in_bytes (eltype));
size = size_binop (MULT_EXPR, size, l);
if (condition)
size = fold_build3 (COND_EXPR, sizetype, condition,
size, size_zero_node);
}
else
size = size_binop (MULT_EXPR, size, l);
}
}
if (!processing_template_decl)
{
if (side_effects)
size = build2 (COMPOUND_EXPR, sizetype, side_effects, size);
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION)
{
size = size_binop (MINUS_EXPR, size, size_one_node);
tree index_type = build_index_type (size);
tree eltype = TREE_TYPE (first);
while (TREE_CODE (eltype) == ARRAY_TYPE)
eltype = TREE_TYPE (eltype);
tree type = build_array_type (eltype, index_type);
tree ptype = build_pointer_type (eltype);
if (TREE_CODE (TREE_TYPE (t)) == REFERENCE_TYPE
&& POINTER_TYPE_P (TREE_TYPE (TREE_TYPE (t))))
t = convert_from_reference (t);
else if (TREE_CODE (TREE_TYPE (t)) == ARRAY_TYPE)
t = build_fold_addr_expr (t);
tree t2 = build_fold_addr_expr (first);
t2 = fold_convert_loc (OMP_CLAUSE_LOCATION (c),
ptrdiff_type_node, t2);
t2 = fold_build2_loc (OMP_CLAUSE_LOCATION (c), MINUS_EXPR,
ptrdiff_type_node, t2,
fold_convert_loc (OMP_CLAUSE_LOCATION (c),
ptrdiff_type_node, t));
if (tree_fits_shwi_p (t2))
t = build2 (MEM_REF, type, t,
build_int_cst (ptype, tree_to_shwi (t2)));
else
{
t2 = fold_convert_loc (OMP_CLAUSE_LOCATION (c),
sizetype, t2);
t = build2_loc (OMP_CLAUSE_LOCATION (c), POINTER_PLUS_EXPR,
TREE_TYPE (t), t, t2);
t = build2 (MEM_REF, type, t, build_int_cst (ptype, 0));
}
OMP_CLAUSE_DECL (c) = t;
return false;
}
OMP_CLAUSE_DECL (c) = first;
OMP_CLAUSE_SIZE (c) = size;
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_MAP
|| (TREE_CODE (t) == COMPONENT_REF
&& TREE_CODE (TREE_TYPE (t)) == ARRAY_TYPE))
return false;
if (ort == C_ORT_OMP || ort == C_ORT_ACC)
switch (OMP_CLAUSE_MAP_KIND (c))
{
case GOMP_MAP_ALLOC:
case GOMP_MAP_TO:
case GOMP_MAP_FROM:
case GOMP_MAP_TOFROM:
case GOMP_MAP_ALWAYS_TO:
case GOMP_MAP_ALWAYS_FROM:
case GOMP_MAP_ALWAYS_TOFROM:
case GOMP_MAP_RELEASE:
case GOMP_MAP_DELETE:
case GOMP_MAP_FORCE_TO:
case GOMP_MAP_FORCE_FROM:
case GOMP_MAP_FORCE_TOFROM:
case GOMP_MAP_FORCE_PRESENT:
OMP_CLAUSE_MAP_MAYBE_ZERO_LENGTH_ARRAY_SECTION (c) = 1;
break;
default:
break;
}
tree c2 = build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_MAP);
if ((ort & C_ORT_OMP_DECLARE_SIMD) != C_ORT_OMP && ort != C_ORT_ACC)
OMP_CLAUSE_SET_MAP_KIND (c2, GOMP_MAP_POINTER);
else if (TREE_CODE (t) == COMPONENT_REF)
OMP_CLAUSE_SET_MAP_KIND (c2, GOMP_MAP_ALWAYS_POINTER);
else if (REFERENCE_REF_P (t)
&& TREE_CODE (TREE_OPERAND (t, 0)) == COMPONENT_REF)
{
t = TREE_OPERAND (t, 0);
OMP_CLAUSE_SET_MAP_KIND (c2, GOMP_MAP_ALWAYS_POINTER);
}
else
OMP_CLAUSE_SET_MAP_KIND (c2, GOMP_MAP_FIRSTPRIVATE_POINTER);
if (OMP_CLAUSE_MAP_KIND (c2) != GOMP_MAP_FIRSTPRIVATE_POINTER
&& !cxx_mark_addressable (t))
return false;
OMP_CLAUSE_DECL (c2) = t;
t = build_fold_addr_expr (first);
t = fold_convert_loc (OMP_CLAUSE_LOCATION (c),
ptrdiff_type_node, t);
tree ptr = OMP_CLAUSE_DECL (c2);
ptr = convert_from_reference (ptr);
if (!POINTER_TYPE_P (TREE_TYPE (ptr)))
ptr = build_fold_addr_expr (ptr);
t = fold_build2_loc (OMP_CLAUSE_LOCATION (c), MINUS_EXPR,
ptrdiff_type_node, t,
fold_convert_loc (OMP_CLAUSE_LOCATION (c),
ptrdiff_type_node, ptr));
OMP_CLAUSE_SIZE (c2) = t;
OMP_CLAUSE_CHAIN (c2) = OMP_CLAUSE_CHAIN (c);
OMP_CLAUSE_CHAIN (c) = c2;
ptr = OMP_CLAUSE_DECL (c2);
if (OMP_CLAUSE_MAP_KIND (c2) != GOMP_MAP_FIRSTPRIVATE_POINTER
&& TREE_CODE (TREE_TYPE (ptr)) == REFERENCE_TYPE
&& POINTER_TYPE_P (TREE_TYPE (TREE_TYPE (ptr))))
{
tree c3 = build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_MAP);
OMP_CLAUSE_SET_MAP_KIND (c3, OMP_CLAUSE_MAP_KIND (c2));
OMP_CLAUSE_DECL (c3) = ptr;
if (OMP_CLAUSE_MAP_KIND (c2) == GOMP_MAP_ALWAYS_POINTER)
OMP_CLAUSE_DECL (c2) = build_simple_mem_ref (ptr);
else
OMP_CLAUSE_DECL (c2) = convert_from_reference (ptr);
OMP_CLAUSE_SIZE (c3) = size_zero_node;
OMP_CLAUSE_CHAIN (c3) = OMP_CLAUSE_CHAIN (c2);
OMP_CLAUSE_CHAIN (c2) = c3;
}
}
}
return false;
}
tree
omp_reduction_id (enum tree_code reduction_code, tree reduction_id, tree type)
{
const char *p = NULL;
const char *m = NULL;
switch (reduction_code)
{
case PLUS_EXPR:
case MULT_EXPR:
case MINUS_EXPR:
case BIT_AND_EXPR:
case BIT_XOR_EXPR:
case BIT_IOR_EXPR:
case TRUTH_ANDIF_EXPR:
case TRUTH_ORIF_EXPR:
reduction_id = ovl_op_identifier (false, reduction_code);
break;
case MIN_EXPR:
p = "min";
break;
case MAX_EXPR:
p = "max";
break;
default:
break;
}
if (p == NULL)
{
if (TREE_CODE (reduction_id) != IDENTIFIER_NODE)
return error_mark_node;
p = IDENTIFIER_POINTER (reduction_id);
}
if (type != NULL_TREE)
m = mangle_type_string (TYPE_MAIN_VARIANT (type));
const char prefix[] = "omp declare reduction ";
size_t lenp = sizeof (prefix);
if (strncmp (p, prefix, lenp - 1) == 0)
lenp = 1;
size_t len = strlen (p);
size_t lenm = m ? strlen (m) + 1 : 0;
char *name = XALLOCAVEC (char, lenp + len + lenm);
if (lenp > 1)
memcpy (name, prefix, lenp - 1);
memcpy (name + lenp - 1, p, len + 1);
if (m)
{
name[lenp + len - 1] = '~';
memcpy (name + lenp + len, m, lenm);
}
return get_identifier (name);
}
static tree
omp_reduction_lookup (location_t loc, tree id, tree type, tree *baselinkp,
vec<tree> *ambiguousp)
{
tree orig_id = id;
tree baselink = NULL_TREE;
if (identifier_p (id))
{
cp_id_kind idk;
bool nonint_cst_expression_p;
const char *error_msg;
id = omp_reduction_id (ERROR_MARK, id, type);
tree decl = lookup_name (id);
if (decl == NULL_TREE)
decl = error_mark_node;
id = finish_id_expression (id, decl, NULL_TREE, &idk, false, true,
&nonint_cst_expression_p, false, true, false,
false, &error_msg, loc);
if (idk == CP_ID_KIND_UNQUALIFIED
&& identifier_p (id))
{
vec<tree, va_gc> *args = NULL;
vec_safe_push (args, build_reference_type (type));
id = perform_koenig_lookup (id, args, tf_none);
}
}
else if (TREE_CODE (id) == SCOPE_REF)
id = lookup_qualified_name (TREE_OPERAND (id, 0),
omp_reduction_id (ERROR_MARK,
TREE_OPERAND (id, 1),
type),
false, false);
tree fns = id;
id = NULL_TREE;
if (fns && is_overloaded_fn (fns))
{
for (lkp_iterator iter (get_fns (fns)); iter; ++iter)
{
tree fndecl = *iter;
if (TREE_CODE (fndecl) == FUNCTION_DECL)
{
tree argtype = TREE_VALUE (TYPE_ARG_TYPES (TREE_TYPE (fndecl)));
if (same_type_p (TREE_TYPE (argtype), type))
{
id = fndecl;
break;
}
}
}
if (id && BASELINK_P (fns))
{
if (baselinkp)
*baselinkp = fns;
else
baselink = fns;
}
}
if (!id && CLASS_TYPE_P (type) && TYPE_BINFO (type))
{
vec<tree> ambiguous = vNULL;
tree binfo = TYPE_BINFO (type), base_binfo, ret = NULL_TREE;
unsigned int ix;
if (ambiguousp == NULL)
ambiguousp = &ambiguous;
for (ix = 0; BINFO_BASE_ITERATE (binfo, ix, base_binfo); ix++)
{
id = omp_reduction_lookup (loc, orig_id, BINFO_TYPE (base_binfo),
baselinkp ? baselinkp : &baselink,
ambiguousp);
if (id == NULL_TREE)
continue;
if (!ambiguousp->is_empty ())
ambiguousp->safe_push (id);
else if (ret != NULL_TREE)
{
ambiguousp->safe_push (ret);
ambiguousp->safe_push (id);
ret = NULL_TREE;
}
else
ret = id;
}
if (ambiguousp != &ambiguous)
return ret;
if (!ambiguous.is_empty ())
{
const char *str = _("candidates are:");
unsigned int idx;
tree udr;
error_at (loc, "user defined reduction lookup is ambiguous");
FOR_EACH_VEC_ELT (ambiguous, idx, udr)
{
inform (DECL_SOURCE_LOCATION (udr), "%s %#qD", str, udr);
if (idx == 0)
str = get_spaces (str);
}
ambiguous.release ();
ret = error_mark_node;
baselink = NULL_TREE;
}
id = ret;
}
if (id && baselink)
perform_or_defer_access_check (BASELINK_BINFO (baselink),
id, id, tf_warning_or_error);
return id;
}
tree
cp_remove_omp_priv_cleanup_stmt (tree *tp, int *walk_subtrees, void *data)
{
if (TYPE_P (*tp))
*walk_subtrees = 0;
else if (TREE_CODE (*tp) == CLEANUP_STMT && CLEANUP_DECL (*tp) == (tree) data)
*tp = CLEANUP_BODY (*tp);
else if (TREE_CODE (*tp) == DECL_EXPR)
{
tree decl = DECL_EXPR_DECL (*tp);
if (!processing_template_decl
&& decl == (tree) data
&& DECL_INITIAL (decl)
&& DECL_INITIAL (decl) != error_mark_node)
{
tree list = NULL_TREE;
append_to_statement_list_force (*tp, &list);
tree init_expr = build2 (INIT_EXPR, void_type_node,
decl, DECL_INITIAL (decl));
DECL_INITIAL (decl) = NULL_TREE;
append_to_statement_list_force (init_expr, &list);
*tp = list;
}
}
return NULL_TREE;
}
struct cp_check_omp_declare_reduction_data
{
location_t loc;
tree stmts[7];
bool combiner_p;
};
static tree
cp_check_omp_declare_reduction_r (tree *tp, int *, void *data)
{
struct cp_check_omp_declare_reduction_data *udr_data
= (struct cp_check_omp_declare_reduction_data *) data;
if (SSA_VAR_P (*tp)
&& !DECL_ARTIFICIAL (*tp)
&& *tp != DECL_EXPR_DECL (udr_data->stmts[udr_data->combiner_p ? 0 : 3])
&& *tp != DECL_EXPR_DECL (udr_data->stmts[udr_data->combiner_p ? 1 : 4]))
{
location_t loc = udr_data->loc;
if (udr_data->combiner_p)
error_at (loc, "%<#pragma omp declare reduction%> combiner refers to "
"variable %qD which is not %<omp_out%> nor %<omp_in%>",
*tp);
else
error_at (loc, "%<#pragma omp declare reduction%> initializer refers "
"to variable %qD which is not %<omp_priv%> nor "
"%<omp_orig%>",
*tp);
return *tp;
}
return NULL_TREE;
}
void
cp_check_omp_declare_reduction (tree udr)
{
tree type = TREE_VALUE (TYPE_ARG_TYPES (TREE_TYPE (udr)));
gcc_assert (TREE_CODE (type) == REFERENCE_TYPE);
type = TREE_TYPE (type);
int i;
location_t loc = DECL_SOURCE_LOCATION (udr);
if (type == error_mark_node)
return;
if (ARITHMETIC_TYPE_P (type))
{
static enum tree_code predef_codes[]
= { PLUS_EXPR, MULT_EXPR, MINUS_EXPR, BIT_AND_EXPR, BIT_XOR_EXPR,
BIT_IOR_EXPR, TRUTH_ANDIF_EXPR, TRUTH_ORIF_EXPR };
for (i = 0; i < 8; i++)
{
tree id = omp_reduction_id (predef_codes[i], NULL_TREE, NULL_TREE);
const char *n1 = IDENTIFIER_POINTER (DECL_NAME (udr));
const char *n2 = IDENTIFIER_POINTER (id);
if (strncmp (n1, n2, IDENTIFIER_LENGTH (id)) == 0
&& (n1[IDENTIFIER_LENGTH (id)] == '~'
|| n1[IDENTIFIER_LENGTH (id)] == '\0'))
break;
}
if (i == 8
&& TREE_CODE (type) != COMPLEX_EXPR)
{
const char prefix_minmax[] = "omp declare reduction m";
size_t prefix_size = sizeof (prefix_minmax) - 1;
const char *n = IDENTIFIER_POINTER (DECL_NAME (udr));
if (strncmp (IDENTIFIER_POINTER (DECL_NAME (udr)),
prefix_minmax, prefix_size) == 0
&& ((n[prefix_size] == 'i' && n[prefix_size + 1] == 'n')
|| (n[prefix_size] == 'a' && n[prefix_size + 1] == 'x'))
&& (n[prefix_size + 2] == '~' || n[prefix_size + 2] == '\0'))
i = 0;
}
if (i < 8)
{
error_at (loc, "predeclared arithmetic type %qT in "
"%<#pragma omp declare reduction%>", type);
return;
}
}
else if (TREE_CODE (type) == FUNCTION_TYPE
|| TREE_CODE (type) == METHOD_TYPE
|| TREE_CODE (type) == ARRAY_TYPE)
{
error_at (loc, "function or array type %qT in "
"%<#pragma omp declare reduction%>", type);
return;
}
else if (TREE_CODE (type) == REFERENCE_TYPE)
{
error_at (loc, "reference type %qT in %<#pragma omp declare reduction%>",
type);
return;
}
else if (TYPE_QUALS_NO_ADDR_SPACE (type))
{
error_at (loc, "const, volatile or __restrict qualified type %qT in "
"%<#pragma omp declare reduction%>", type);
return;
}
tree body = DECL_SAVED_TREE (udr);
if (body == NULL_TREE || TREE_CODE (body) != STATEMENT_LIST)
return;
tree_stmt_iterator tsi;
struct cp_check_omp_declare_reduction_data data;
memset (data.stmts, 0, sizeof data.stmts);
for (i = 0, tsi = tsi_start (body);
i < 7 && !tsi_end_p (tsi);
i++, tsi_next (&tsi))
data.stmts[i] = tsi_stmt (tsi);
data.loc = loc;
gcc_assert (tsi_end_p (tsi));
if (i >= 3)
{
gcc_assert (TREE_CODE (data.stmts[0]) == DECL_EXPR
&& TREE_CODE (data.stmts[1]) == DECL_EXPR);
if (TREE_NO_WARNING (DECL_EXPR_DECL (data.stmts[0])))
return;
data.combiner_p = true;
if (cp_walk_tree (&data.stmts[2], cp_check_omp_declare_reduction_r,
&data, NULL))
TREE_NO_WARNING (DECL_EXPR_DECL (data.stmts[0])) = 1;
}
if (i >= 6)
{
gcc_assert (TREE_CODE (data.stmts[3]) == DECL_EXPR
&& TREE_CODE (data.stmts[4]) == DECL_EXPR);
data.combiner_p = false;
if (cp_walk_tree (&data.stmts[5], cp_check_omp_declare_reduction_r,
&data, NULL)
|| cp_walk_tree (&DECL_INITIAL (DECL_EXPR_DECL (data.stmts[3])),
cp_check_omp_declare_reduction_r, &data, NULL))
TREE_NO_WARNING (DECL_EXPR_DECL (data.stmts[0])) = 1;
if (i == 7)
gcc_assert (TREE_CODE (data.stmts[6]) == DECL_EXPR);
}
}
static tree
clone_omp_udr (tree stmt, tree omp_decl1, tree omp_decl2,
tree decl, tree placeholder)
{
copy_body_data id;
hash_map<tree, tree> decl_map;
decl_map.put (omp_decl1, placeholder);
decl_map.put (omp_decl2, decl);
memset (&id, 0, sizeof (id));
id.src_fn = DECL_CONTEXT (omp_decl1);
id.dst_fn = current_function_decl;
id.src_cfun = DECL_STRUCT_FUNCTION (id.src_fn);
id.decl_map = &decl_map;
id.copy_decl = copy_decl_no_change;
id.transform_call_graph_edges = CB_CGE_DUPLICATE;
id.transform_new_cfg = true;
id.transform_return_to_modify = false;
id.transform_lang_insert_block = NULL;
id.eh_lp_nr = 0;
walk_tree (&stmt, copy_tree_body_r, &id, NULL);
return stmt;
}
static tree
find_omp_placeholder_r (tree *tp, int *, void *data)
{
if (*tp == (tree) data)
return *tp;
return NULL_TREE;
}
static bool
finish_omp_reduction_clause (tree c, bool *need_default_ctor, bool *need_dtor)
{
tree t = OMP_CLAUSE_DECL (c);
bool predefined = false;
if (TREE_CODE (t) == TREE_LIST)
{
gcc_assert (processing_template_decl);
return false;
}
tree type = TREE_TYPE (t);
if (TREE_CODE (t) == MEM_REF)
type = TREE_TYPE (type);
if (TREE_CODE (type) == REFERENCE_TYPE)
type = TREE_TYPE (type);
if (TREE_CODE (type) == ARRAY_TYPE)
{
tree oatype = type;
gcc_assert (TREE_CODE (t) != MEM_REF);
while (TREE_CODE (type) == ARRAY_TYPE)
type = TREE_TYPE (type);
if (!processing_template_decl)
{
t = require_complete_type (t);
if (t == error_mark_node)
return true;
tree size = size_binop (EXACT_DIV_EXPR, TYPE_SIZE_UNIT (oatype),
TYPE_SIZE_UNIT (type));
if (integer_zerop (size))
{
error ("%qE in %<reduction%> clause is a zero size array",
omp_clause_printable_decl (t));
return true;
}
size = size_binop (MINUS_EXPR, size, size_one_node);
tree index_type = build_index_type (size);
tree atype = build_array_type (type, index_type);
tree ptype = build_pointer_type (type);
if (TREE_CODE (TREE_TYPE (t)) == ARRAY_TYPE)
t = build_fold_addr_expr (t);
t = build2 (MEM_REF, atype, t, build_int_cst (ptype, 0));
OMP_CLAUSE_DECL (c) = t;
}
}
if (type == error_mark_node)
return true;
else if (ARITHMETIC_TYPE_P (type))
switch (OMP_CLAUSE_REDUCTION_CODE (c))
{
case PLUS_EXPR:
case MULT_EXPR:
case MINUS_EXPR:
predefined = true;
break;
case MIN_EXPR:
case MAX_EXPR:
if (TREE_CODE (type) == COMPLEX_TYPE)
break;
predefined = true;
break;
case BIT_AND_EXPR:
case BIT_IOR_EXPR:
case BIT_XOR_EXPR:
if (FLOAT_TYPE_P (type) || TREE_CODE (type) == COMPLEX_TYPE)
break;
predefined = true;
break;
case TRUTH_ANDIF_EXPR:
case TRUTH_ORIF_EXPR:
if (FLOAT_TYPE_P (type))
break;
predefined = true;
break;
default:
break;
}
else if (TYPE_READONLY (type))
{
error ("%qE has const type for %<reduction%>",
omp_clause_printable_decl (t));
return true;
}
else if (!processing_template_decl)
{
t = require_complete_type (t);
if (t == error_mark_node)
return true;
OMP_CLAUSE_DECL (c) = t;
}
if (predefined)
{
OMP_CLAUSE_REDUCTION_PLACEHOLDER (c) = NULL_TREE;
return false;
}
else if (processing_template_decl)
{
if (OMP_CLAUSE_REDUCTION_PLACEHOLDER (c) == error_mark_node)
return true;
return false;
}
tree id = OMP_CLAUSE_REDUCTION_PLACEHOLDER (c);
type = TYPE_MAIN_VARIANT (type);
OMP_CLAUSE_REDUCTION_PLACEHOLDER (c) = NULL_TREE;
if (id == NULL_TREE)
id = omp_reduction_id (OMP_CLAUSE_REDUCTION_CODE (c),
NULL_TREE, NULL_TREE);
id = omp_reduction_lookup (OMP_CLAUSE_LOCATION (c), id, type, NULL, NULL);
if (id)
{
if (id == error_mark_node)
return true;
mark_used (id);
tree body = DECL_SAVED_TREE (id);
if (!body)
return true;
if (TREE_CODE (body) == STATEMENT_LIST)
{
tree_stmt_iterator tsi;
tree placeholder = NULL_TREE, decl_placeholder = NULL_TREE;
int i;
tree stmts[7];
tree atype = TREE_VALUE (TYPE_ARG_TYPES (TREE_TYPE (id)));
atype = TREE_TYPE (atype);
bool need_static_cast = !same_type_p (type, atype);
memset (stmts, 0, sizeof stmts);
for (i = 0, tsi = tsi_start (body);
i < 7 && !tsi_end_p (tsi);
i++, tsi_next (&tsi))
stmts[i] = tsi_stmt (tsi);
gcc_assert (tsi_end_p (tsi));
if (i >= 3)
{
gcc_assert (TREE_CODE (stmts[0]) == DECL_EXPR
&& TREE_CODE (stmts[1]) == DECL_EXPR);
placeholder = build_lang_decl (VAR_DECL, NULL_TREE, type);
DECL_ARTIFICIAL (placeholder) = 1;
DECL_IGNORED_P (placeholder) = 1;
OMP_CLAUSE_REDUCTION_PLACEHOLDER (c) = placeholder;
if (TREE_CODE (t) == MEM_REF)
{
decl_placeholder = build_lang_decl (VAR_DECL, NULL_TREE,
type);
DECL_ARTIFICIAL (decl_placeholder) = 1;
DECL_IGNORED_P (decl_placeholder) = 1;
OMP_CLAUSE_REDUCTION_DECL_PLACEHOLDER (c) = decl_placeholder;
}
if (TREE_ADDRESSABLE (DECL_EXPR_DECL (stmts[0])))
cxx_mark_addressable (placeholder);
if (TREE_ADDRESSABLE (DECL_EXPR_DECL (stmts[1]))
&& TREE_CODE (TREE_TYPE (OMP_CLAUSE_DECL (c)))
!= REFERENCE_TYPE)
cxx_mark_addressable (decl_placeholder ? decl_placeholder
: OMP_CLAUSE_DECL (c));
tree omp_out = placeholder;
tree omp_in = decl_placeholder ? decl_placeholder
: convert_from_reference (OMP_CLAUSE_DECL (c));
if (need_static_cast)
{
tree rtype = build_reference_type (atype);
omp_out = build_static_cast (rtype, omp_out,
tf_warning_or_error);
omp_in = build_static_cast (rtype, omp_in,
tf_warning_or_error);
if (omp_out == error_mark_node || omp_in == error_mark_node)
return true;
omp_out = convert_from_reference (omp_out);
omp_in = convert_from_reference (omp_in);
}
OMP_CLAUSE_REDUCTION_MERGE (c)
= clone_omp_udr (stmts[2], DECL_EXPR_DECL (stmts[0]),
DECL_EXPR_DECL (stmts[1]), omp_in, omp_out);
}
if (i >= 6)
{
gcc_assert (TREE_CODE (stmts[3]) == DECL_EXPR
&& TREE_CODE (stmts[4]) == DECL_EXPR);
if (TREE_ADDRESSABLE (DECL_EXPR_DECL (stmts[3])))
cxx_mark_addressable (decl_placeholder ? decl_placeholder
: OMP_CLAUSE_DECL (c));
if (TREE_ADDRESSABLE (DECL_EXPR_DECL (stmts[4])))
cxx_mark_addressable (placeholder);
tree omp_priv = decl_placeholder ? decl_placeholder
: convert_from_reference (OMP_CLAUSE_DECL (c));
tree omp_orig = placeholder;
if (need_static_cast)
{
if (i == 7)
{
error_at (OMP_CLAUSE_LOCATION (c),
"user defined reduction with constructor "
"initializer for base class %qT", atype);
return true;
}
tree rtype = build_reference_type (atype);
omp_priv = build_static_cast (rtype, omp_priv,
tf_warning_or_error);
omp_orig = build_static_cast (rtype, omp_orig,
tf_warning_or_error);
if (omp_priv == error_mark_node
|| omp_orig == error_mark_node)
return true;
omp_priv = convert_from_reference (omp_priv);
omp_orig = convert_from_reference (omp_orig);
}
if (i == 6)
*need_default_ctor = true;
OMP_CLAUSE_REDUCTION_INIT (c)
= clone_omp_udr (stmts[5], DECL_EXPR_DECL (stmts[4]),
DECL_EXPR_DECL (stmts[3]),
omp_priv, omp_orig);
if (cp_walk_tree (&OMP_CLAUSE_REDUCTION_INIT (c),
find_omp_placeholder_r, placeholder, NULL))
OMP_CLAUSE_REDUCTION_OMP_ORIG_REF (c) = 1;
}
else if (i >= 3)
{
if (CLASS_TYPE_P (type) && !pod_type_p (type))
*need_default_ctor = true;
else
{
tree init;
tree v = decl_placeholder ? decl_placeholder
: convert_from_reference (t);
if (AGGREGATE_TYPE_P (TREE_TYPE (v)))
init = build_constructor (TREE_TYPE (v), NULL);
else
init = fold_convert (TREE_TYPE (v), integer_zero_node);
OMP_CLAUSE_REDUCTION_INIT (c)
= build2 (INIT_EXPR, TREE_TYPE (v), v, init);
}
}
}
}
if (OMP_CLAUSE_REDUCTION_PLACEHOLDER (c))
*need_dtor = true;
else
{
error ("user defined reduction not found for %qE",
omp_clause_printable_decl (t));
return true;
}
if (TREE_CODE (OMP_CLAUSE_DECL (c)) == MEM_REF)
gcc_assert (TYPE_SIZE_UNIT (type)
&& TREE_CODE (TYPE_SIZE_UNIT (type)) == INTEGER_CST);
return false;
}
void
finish_omp_declare_simd_methods (tree t)
{
if (processing_template_decl)
return;
for (tree x = TYPE_FIELDS (t); x; x = DECL_CHAIN (x))
{
if (TREE_CODE (TREE_TYPE (x)) != METHOD_TYPE)
continue;
tree ods = lookup_attribute ("omp declare simd", DECL_ATTRIBUTES (x));
if (!ods || !TREE_VALUE (ods))
continue;
for (tree c = TREE_VALUE (TREE_VALUE (ods)); c; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LINEAR
&& integer_zerop (OMP_CLAUSE_DECL (c))
&& OMP_CLAUSE_LINEAR_STEP (c)
&& TREE_CODE (TREE_TYPE (OMP_CLAUSE_LINEAR_STEP (c)))
== POINTER_TYPE)
{
tree s = OMP_CLAUSE_LINEAR_STEP (c);
s = fold_convert_loc (OMP_CLAUSE_LOCATION (c), sizetype, s);
s = fold_build2_loc (OMP_CLAUSE_LOCATION (c), MULT_EXPR,
sizetype, s, TYPE_SIZE_UNIT (t));
OMP_CLAUSE_LINEAR_STEP (c) = s;
}
}
}
static bool
cp_finish_omp_clause_depend_sink (tree sink_clause)
{
tree t = OMP_CLAUSE_DECL (sink_clause);
gcc_assert (TREE_CODE (t) == TREE_LIST);
if (processing_template_decl)
return false;
for (; t; t = TREE_CHAIN (t))
{
tree decl = TREE_VALUE (t);
if (TREE_CODE (TREE_TYPE (decl)) == POINTER_TYPE)
{
tree offset = TREE_PURPOSE (t);
bool neg = wi::neg_p (wi::to_wide (offset));
offset = fold_unary (ABS_EXPR, TREE_TYPE (offset), offset);
decl = mark_rvalue_use (decl);
decl = convert_from_reference (decl);
tree t2 = pointer_int_sum (OMP_CLAUSE_LOCATION (sink_clause),
neg ? MINUS_EXPR : PLUS_EXPR,
decl, offset);
t2 = fold_build2_loc (OMP_CLAUSE_LOCATION (sink_clause),
MINUS_EXPR, sizetype,
fold_convert (sizetype, t2),
fold_convert (sizetype, decl));
if (t2 == error_mark_node)
return true;
TREE_PURPOSE (t) = t2;
}
}
return false;
}
tree
finish_omp_clauses (tree clauses, enum c_omp_region_type ort)
{
bitmap_head generic_head, firstprivate_head, lastprivate_head;
bitmap_head aligned_head, map_head, map_field_head, oacc_reduction_head;
tree c, t, *pc;
tree safelen = NULL_TREE;
bool branch_seen = false;
bool copyprivate_seen = false;
bool ordered_seen = false;
bool oacc_async = false;
bitmap_obstack_initialize (NULL);
bitmap_initialize (&generic_head, &bitmap_default_obstack);
bitmap_initialize (&firstprivate_head, &bitmap_default_obstack);
bitmap_initialize (&lastprivate_head, &bitmap_default_obstack);
bitmap_initialize (&aligned_head, &bitmap_default_obstack);
bitmap_initialize (&map_head, &bitmap_default_obstack);
bitmap_initialize (&map_field_head, &bitmap_default_obstack);
bitmap_initialize (&oacc_reduction_head, &bitmap_default_obstack);
if (ort & C_ORT_ACC)
for (c = clauses; c; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_ASYNC)
{
oacc_async = true;
break;
}
for (pc = &clauses, c = clauses; c ; c = *pc)
{
bool remove = false;
bool field_ok = false;
switch (OMP_CLAUSE_CODE (c))
{
case OMP_CLAUSE_SHARED:
field_ok = ((ort & C_ORT_OMP_DECLARE_SIMD) == C_ORT_OMP);
goto check_dup_generic;
case OMP_CLAUSE_PRIVATE:
field_ok = ((ort & C_ORT_OMP_DECLARE_SIMD) == C_ORT_OMP);
goto check_dup_generic;
case OMP_CLAUSE_REDUCTION:
field_ok = ((ort & C_ORT_OMP_DECLARE_SIMD) == C_ORT_OMP);
t = OMP_CLAUSE_DECL (c);
if (TREE_CODE (t) == TREE_LIST)
{
if (handle_omp_array_sections (c, ort))
{
remove = true;
break;
}
if (TREE_CODE (t) == TREE_LIST)
{
while (TREE_CODE (t) == TREE_LIST)
t = TREE_CHAIN (t);
}
else
{
gcc_assert (TREE_CODE (t) == MEM_REF);
t = TREE_OPERAND (t, 0);
if (TREE_CODE (t) == POINTER_PLUS_EXPR)
t = TREE_OPERAND (t, 0);
if (TREE_CODE (t) == ADDR_EXPR
|| INDIRECT_REF_P (t))
t = TREE_OPERAND (t, 0);
}
tree n = omp_clause_decl_field (t);
if (n)
t = n;
goto check_dup_generic_t;
}
if (oacc_async)
cxx_mark_addressable (t);
goto check_dup_generic;
case OMP_CLAUSE_COPYPRIVATE:
copyprivate_seen = true;
field_ok = ((ort & C_ORT_OMP_DECLARE_SIMD) == C_ORT_OMP);
goto check_dup_generic;
case OMP_CLAUSE_COPYIN:
goto check_dup_generic;
case OMP_CLAUSE_LINEAR:
field_ok = ((ort & C_ORT_OMP_DECLARE_SIMD) == C_ORT_OMP);
t = OMP_CLAUSE_DECL (c);
if (ort != C_ORT_OMP_DECLARE_SIMD
&& OMP_CLAUSE_LINEAR_KIND (c) != OMP_CLAUSE_LINEAR_DEFAULT)
{
error_at (OMP_CLAUSE_LOCATION (c),
"modifier should not be specified in %<linear%> "
"clause on %<simd%> or %<for%> constructs");
OMP_CLAUSE_LINEAR_KIND (c) = OMP_CLAUSE_LINEAR_DEFAULT;
}
if ((VAR_P (t) || TREE_CODE (t) == PARM_DECL)
&& !type_dependent_expression_p (t))
{
tree type = TREE_TYPE (t);
if ((OMP_CLAUSE_LINEAR_KIND (c) == OMP_CLAUSE_LINEAR_REF
|| OMP_CLAUSE_LINEAR_KIND (c) == OMP_CLAUSE_LINEAR_UVAL)
&& TREE_CODE (type) != REFERENCE_TYPE)
{
error ("linear clause with %qs modifier applied to "
"non-reference variable with %qT type",
OMP_CLAUSE_LINEAR_KIND (c) == OMP_CLAUSE_LINEAR_REF
? "ref" : "uval", TREE_TYPE (t));
remove = true;
break;
}
if (TREE_CODE (type) == REFERENCE_TYPE)
type = TREE_TYPE (type);
if (OMP_CLAUSE_LINEAR_KIND (c) != OMP_CLAUSE_LINEAR_REF)
{
if (!INTEGRAL_TYPE_P (type)
&& TREE_CODE (type) != POINTER_TYPE)
{
error ("linear clause applied to non-integral non-pointer"
" variable with %qT type", TREE_TYPE (t));
remove = true;
break;
}
}
}
t = OMP_CLAUSE_LINEAR_STEP (c);
if (t == NULL_TREE)
t = integer_one_node;
if (t == error_mark_node)
{
remove = true;
break;
}
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t))
&& (ort != C_ORT_OMP_DECLARE_SIMD
|| TREE_CODE (t) != PARM_DECL
|| TREE_CODE (TREE_TYPE (t)) != REFERENCE_TYPE
|| !INTEGRAL_TYPE_P (TREE_TYPE (TREE_TYPE (t)))))
{
error ("linear step expression must be integral");
remove = true;
break;
}
else
{
t = mark_rvalue_use (t);
if (ort == C_ORT_OMP_DECLARE_SIMD && TREE_CODE (t) == PARM_DECL)
{
OMP_CLAUSE_LINEAR_VARIABLE_STRIDE (c) = 1;
goto check_dup_generic;
}
if (!processing_template_decl
&& (VAR_P (OMP_CLAUSE_DECL (c))
|| TREE_CODE (OMP_CLAUSE_DECL (c)) == PARM_DECL))
{
if (ort == C_ORT_OMP_DECLARE_SIMD)
{
t = maybe_constant_value (t);
if (TREE_CODE (t) != INTEGER_CST)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<linear%> clause step %qE is neither "
"constant nor a parameter", t);
remove = true;
break;
}
}
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
tree type = TREE_TYPE (OMP_CLAUSE_DECL (c));
if (TREE_CODE (type) == REFERENCE_TYPE)
type = TREE_TYPE (type);
if (OMP_CLAUSE_LINEAR_KIND (c) == OMP_CLAUSE_LINEAR_REF)
{
type = build_pointer_type (type);
tree d = fold_convert (type, OMP_CLAUSE_DECL (c));
t = pointer_int_sum (OMP_CLAUSE_LOCATION (c), PLUS_EXPR,
d, t);
t = fold_build2_loc (OMP_CLAUSE_LOCATION (c),
MINUS_EXPR, sizetype,
fold_convert (sizetype, t),
fold_convert (sizetype, d));
if (t == error_mark_node)
{
remove = true;
break;
}
}
else if (TREE_CODE (type) == POINTER_TYPE
&& (ort != C_ORT_OMP_DECLARE_SIMD
|| TREE_CODE (OMP_CLAUSE_DECL (c)) != PARM_DECL
|| !DECL_ARTIFICIAL (OMP_CLAUSE_DECL (c))
|| DECL_NAME (OMP_CLAUSE_DECL (c))
!= this_identifier
|| !TYPE_BEING_DEFINED (TREE_TYPE (type))))
{
tree d = convert_from_reference (OMP_CLAUSE_DECL (c));
t = pointer_int_sum (OMP_CLAUSE_LOCATION (c), PLUS_EXPR,
d, t);
t = fold_build2_loc (OMP_CLAUSE_LOCATION (c),
MINUS_EXPR, sizetype,
fold_convert (sizetype, t),
fold_convert (sizetype, d));
if (t == error_mark_node)
{
remove = true;
break;
}
}
else
t = fold_convert (type, t);
}
OMP_CLAUSE_LINEAR_STEP (c) = t;
}
goto check_dup_generic;
check_dup_generic:
t = omp_clause_decl_field (OMP_CLAUSE_DECL (c));
if (t)
{
if (!remove && OMP_CLAUSE_CODE (c) != OMP_CLAUSE_SHARED)
omp_note_field_privatization (t, OMP_CLAUSE_DECL (c));
}
else
t = OMP_CLAUSE_DECL (c);
check_dup_generic_t:
if (t == current_class_ptr
&& (ort != C_ORT_OMP_DECLARE_SIMD
|| (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_LINEAR
&& OMP_CLAUSE_CODE (c) != OMP_CLAUSE_UNIFORM)))
{
error ("%<this%> allowed in OpenMP only in %<declare simd%>"
" clauses");
remove = true;
break;
}
if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL
&& (!field_ok || TREE_CODE (t) != FIELD_DECL))
{
if (processing_template_decl && TREE_CODE (t) != OVERLOAD)
break;
if (DECL_P (t))
error ("%qD is not a variable in clause %qs", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
else
error ("%qE is not a variable in clause %qs", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (ort == C_ORT_ACC
&& OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION)
{
if (bitmap_bit_p (&oacc_reduction_head, DECL_UID (t)))
{
error ("%qD appears more than once in reduction clauses", t);
remove = true;
}
else
bitmap_set_bit (&oacc_reduction_head, DECL_UID (t));
}
else if (bitmap_bit_p (&generic_head, DECL_UID (t))
|| bitmap_bit_p (&firstprivate_head, DECL_UID (t))
|| bitmap_bit_p (&lastprivate_head, DECL_UID (t)))
{
error ("%qD appears more than once in data clauses", t);
remove = true;
}
else if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_PRIVATE
&& bitmap_bit_p (&map_head, DECL_UID (t)))
{
if (ort == C_ORT_ACC)
error ("%qD appears more than once in data clauses", t);
else
error ("%qD appears both in data and map clauses", t);
remove = true;
}
else
bitmap_set_bit (&generic_head, DECL_UID (t));
if (!field_ok)
break;
handle_field_decl:
if (!remove
&& TREE_CODE (t) == FIELD_DECL
&& t == OMP_CLAUSE_DECL (c)
&& ort != C_ORT_ACC)
{
OMP_CLAUSE_DECL (c)
= omp_privatize_field (t, (OMP_CLAUSE_CODE (c)
== OMP_CLAUSE_SHARED));
if (OMP_CLAUSE_DECL (c) == error_mark_node)
remove = true;
}
break;
case OMP_CLAUSE_FIRSTPRIVATE:
t = omp_clause_decl_field (OMP_CLAUSE_DECL (c));
if (t)
omp_note_field_privatization (t, OMP_CLAUSE_DECL (c));
else
t = OMP_CLAUSE_DECL (c);
if (ort != C_ORT_ACC && t == current_class_ptr)
{
error ("%<this%> allowed in OpenMP only in %<declare simd%>"
" clauses");
remove = true;
break;
}
if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL
&& ((ort & C_ORT_OMP_DECLARE_SIMD) != C_ORT_OMP
|| TREE_CODE (t) != FIELD_DECL))
{
if (processing_template_decl && TREE_CODE (t) != OVERLOAD)
break;
if (DECL_P (t))
error ("%qD is not a variable in clause %<firstprivate%>", t);
else
error ("%qE is not a variable in clause %<firstprivate%>", t);
remove = true;
}
else if (bitmap_bit_p (&generic_head, DECL_UID (t))
|| bitmap_bit_p (&firstprivate_head, DECL_UID (t)))
{
error ("%qD appears more than once in data clauses", t);
remove = true;
}
else if (bitmap_bit_p (&map_head, DECL_UID (t)))
{
if (ort == C_ORT_ACC)
error ("%qD appears more than once in data clauses", t);
else
error ("%qD appears both in data and map clauses", t);
remove = true;
}
else
bitmap_set_bit (&firstprivate_head, DECL_UID (t));
goto handle_field_decl;
case OMP_CLAUSE_LASTPRIVATE:
t = omp_clause_decl_field (OMP_CLAUSE_DECL (c));
if (t)
omp_note_field_privatization (t, OMP_CLAUSE_DECL (c));
else
t = OMP_CLAUSE_DECL (c);
if (t == current_class_ptr)
{
error ("%<this%> allowed in OpenMP only in %<declare simd%>"
" clauses");
remove = true;
break;
}
if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL
&& ((ort & C_ORT_OMP_DECLARE_SIMD) != C_ORT_OMP
|| TREE_CODE (t) != FIELD_DECL))
{
if (processing_template_decl && TREE_CODE (t) != OVERLOAD)
break;
if (DECL_P (t))
error ("%qD is not a variable in clause %<lastprivate%>", t);
else
error ("%qE is not a variable in clause %<lastprivate%>", t);
remove = true;
}
else if (bitmap_bit_p (&generic_head, DECL_UID (t))
|| bitmap_bit_p (&lastprivate_head, DECL_UID (t)))
{
error ("%qD appears more than once in data clauses", t);
remove = true;
}
else
bitmap_set_bit (&lastprivate_head, DECL_UID (t));
goto handle_field_decl;
case OMP_CLAUSE_IF:
t = OMP_CLAUSE_IF_EXPR (c);
t = maybe_convert_cond (t);
if (t == error_mark_node)
remove = true;
else if (!processing_template_decl)
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
OMP_CLAUSE_IF_EXPR (c) = t;
break;
case OMP_CLAUSE_FINAL:
t = OMP_CLAUSE_FINAL_EXPR (c);
t = maybe_convert_cond (t);
if (t == error_mark_node)
remove = true;
else if (!processing_template_decl)
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
OMP_CLAUSE_FINAL_EXPR (c) = t;
break;
case OMP_CLAUSE_GANG:
t = OMP_CLAUSE_OPERAND (c, 1);
if (t != NULL_TREE)
{
if (t == error_mark_node)
remove = true;
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t)))
{
error ("%<gang%> static expression must be integral");
remove = true;
}
else
{
t = mark_rvalue_use (t);
if (!processing_template_decl)
{
t = maybe_constant_value (t);
if (TREE_CODE (t) == INTEGER_CST
&& tree_int_cst_sgn (t) != 1
&& t != integer_minus_one_node)
{
warning_at (OMP_CLAUSE_LOCATION (c), 0,
"%<gang%> static value must be "
"positive");
t = integer_one_node;
}
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
}
}
OMP_CLAUSE_OPERAND (c, 1) = t;
}
case OMP_CLAUSE_WORKER:
case OMP_CLAUSE_VECTOR:
if (OMP_CLAUSE_OPERAND (c, 0) == NULL_TREE)
break;
case OMP_CLAUSE_NUM_TASKS:
case OMP_CLAUSE_NUM_TEAMS:
case OMP_CLAUSE_NUM_THREADS:
case OMP_CLAUSE_NUM_GANGS:
case OMP_CLAUSE_NUM_WORKERS:
case OMP_CLAUSE_VECTOR_LENGTH:
t = OMP_CLAUSE_OPERAND (c, 0);
if (t == error_mark_node)
remove = true;
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t)))
{
switch (OMP_CLAUSE_CODE (c))
{
case OMP_CLAUSE_GANG:
error_at (OMP_CLAUSE_LOCATION (c),
"%<gang%> num expression must be integral"); break;
case OMP_CLAUSE_VECTOR:
error_at (OMP_CLAUSE_LOCATION (c),
"%<vector%> length expression must be integral");
break;
case OMP_CLAUSE_WORKER:
error_at (OMP_CLAUSE_LOCATION (c),
"%<worker%> num expression must be integral");
break;
default:
error_at (OMP_CLAUSE_LOCATION (c),
"%qs expression must be integral",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
}
remove = true;
}
else
{
t = mark_rvalue_use (t);
if (!processing_template_decl)
{
t = maybe_constant_value (t);
if (TREE_CODE (t) == INTEGER_CST
&& tree_int_cst_sgn (t) != 1)
{
switch (OMP_CLAUSE_CODE (c))
{
case OMP_CLAUSE_GANG:
warning_at (OMP_CLAUSE_LOCATION (c), 0,
"%<gang%> num value must be positive");
break;
case OMP_CLAUSE_VECTOR:
warning_at (OMP_CLAUSE_LOCATION (c), 0,
"%<vector%> length value must be "
"positive");
break;
case OMP_CLAUSE_WORKER:
warning_at (OMP_CLAUSE_LOCATION (c), 0,
"%<worker%> num value must be "
"positive");
break;
default:
warning_at (OMP_CLAUSE_LOCATION (c), 0,
"%qs value must be positive",
omp_clause_code_name
[OMP_CLAUSE_CODE (c)]);
}
t = integer_one_node;
}
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
}
OMP_CLAUSE_OPERAND (c, 0) = t;
}
break;
case OMP_CLAUSE_SCHEDULE:
if (OMP_CLAUSE_SCHEDULE_KIND (c) & OMP_CLAUSE_SCHEDULE_NONMONOTONIC)
{
const char *p = NULL;
switch (OMP_CLAUSE_SCHEDULE_KIND (c) & OMP_CLAUSE_SCHEDULE_MASK)
{
case OMP_CLAUSE_SCHEDULE_STATIC: p = "static"; break;
case OMP_CLAUSE_SCHEDULE_DYNAMIC: break;
case OMP_CLAUSE_SCHEDULE_GUIDED: break;
case OMP_CLAUSE_SCHEDULE_AUTO: p = "auto"; break;
case OMP_CLAUSE_SCHEDULE_RUNTIME: p = "runtime"; break;
default: gcc_unreachable ();
}
if (p)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<nonmonotonic%> modifier specified for %qs "
"schedule kind", p);
OMP_CLAUSE_SCHEDULE_KIND (c)
= (enum omp_clause_schedule_kind)
(OMP_CLAUSE_SCHEDULE_KIND (c)
& ~OMP_CLAUSE_SCHEDULE_NONMONOTONIC);
}
}
t = OMP_CLAUSE_SCHEDULE_CHUNK_EXPR (c);
if (t == NULL)
;
else if (t == error_mark_node)
remove = true;
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t)))
{
error ("schedule chunk size expression must be integral");
remove = true;
}
else
{
t = mark_rvalue_use (t);
if (!processing_template_decl)
{
t = maybe_constant_value (t);
if (TREE_CODE (t) == INTEGER_CST
&& tree_int_cst_sgn (t) != 1)
{
warning_at (OMP_CLAUSE_LOCATION (c), 0,
"chunk size value must be positive");
t = integer_one_node;
}
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
}
OMP_CLAUSE_SCHEDULE_CHUNK_EXPR (c) = t;
}
break;
case OMP_CLAUSE_SIMDLEN:
case OMP_CLAUSE_SAFELEN:
t = OMP_CLAUSE_OPERAND (c, 0);
if (t == error_mark_node)
remove = true;
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t)))
{
error ("%qs length expression must be integral",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else
{
t = mark_rvalue_use (t);
if (!processing_template_decl)
{
t = maybe_constant_value (t);
if (TREE_CODE (t) != INTEGER_CST
|| tree_int_cst_sgn (t) != 1)
{
error ("%qs length expression must be positive constant"
" integer expression",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
}
OMP_CLAUSE_OPERAND (c, 0) = t;
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_SAFELEN)
safelen = c;
}
break;
case OMP_CLAUSE_ASYNC:
t = OMP_CLAUSE_ASYNC_EXPR (c);
if (t == error_mark_node)
remove = true;
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t)))
{
error ("%<async%> expression must be integral");
remove = true;
}
else
{
t = mark_rvalue_use (t);
if (!processing_template_decl)
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
OMP_CLAUSE_ASYNC_EXPR (c) = t;
}
break;
case OMP_CLAUSE_WAIT:
t = OMP_CLAUSE_WAIT_EXPR (c);
if (t == error_mark_node)
remove = true;
else if (!processing_template_decl)
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
OMP_CLAUSE_WAIT_EXPR (c) = t;
break;
case OMP_CLAUSE_THREAD_LIMIT:
t = OMP_CLAUSE_THREAD_LIMIT_EXPR (c);
if (t == error_mark_node)
remove = true;
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t)))
{
error ("%<thread_limit%> expression must be integral");
remove = true;
}
else
{
t = mark_rvalue_use (t);
if (!processing_template_decl)
{
t = maybe_constant_value (t);
if (TREE_CODE (t) == INTEGER_CST
&& tree_int_cst_sgn (t) != 1)
{
warning_at (OMP_CLAUSE_LOCATION (c), 0,
"%<thread_limit%> value must be positive");
t = integer_one_node;
}
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
}
OMP_CLAUSE_THREAD_LIMIT_EXPR (c) = t;
}
break;
case OMP_CLAUSE_DEVICE:
t = OMP_CLAUSE_DEVICE_ID (c);
if (t == error_mark_node)
remove = true;
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t)))
{
error ("%<device%> id must be integral");
remove = true;
}
else
{
t = mark_rvalue_use (t);
if (!processing_template_decl)
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
OMP_CLAUSE_DEVICE_ID (c) = t;
}
break;
case OMP_CLAUSE_DIST_SCHEDULE:
t = OMP_CLAUSE_DIST_SCHEDULE_CHUNK_EXPR (c);
if (t == NULL)
;
else if (t == error_mark_node)
remove = true;
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t)))
{
error ("%<dist_schedule%> chunk size expression must be "
"integral");
remove = true;
}
else
{
t = mark_rvalue_use (t);
if (!processing_template_decl)
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
OMP_CLAUSE_DIST_SCHEDULE_CHUNK_EXPR (c) = t;
}
break;
case OMP_CLAUSE_ALIGNED:
t = OMP_CLAUSE_DECL (c);
if (t == current_class_ptr && ort != C_ORT_OMP_DECLARE_SIMD)
{
error ("%<this%> allowed in OpenMP only in %<declare simd%>"
" clauses");
remove = true;
break;
}
if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL)
{
if (processing_template_decl && TREE_CODE (t) != OVERLOAD)
break;
if (DECL_P (t))
error ("%qD is not a variable in %<aligned%> clause", t);
else
error ("%qE is not a variable in %<aligned%> clause", t);
remove = true;
}
else if (!type_dependent_expression_p (t)
&& TREE_CODE (TREE_TYPE (t)) != POINTER_TYPE
&& TREE_CODE (TREE_TYPE (t)) != ARRAY_TYPE
&& (TREE_CODE (TREE_TYPE (t)) != REFERENCE_TYPE
|| (!POINTER_TYPE_P (TREE_TYPE (TREE_TYPE (t)))
&& (TREE_CODE (TREE_TYPE (TREE_TYPE (t)))
!= ARRAY_TYPE))))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE in %<aligned%> clause is neither a pointer nor "
"an array nor a reference to pointer or array", t);
remove = true;
}
else if (bitmap_bit_p (&aligned_head, DECL_UID (t)))
{
error ("%qD appears more than once in %<aligned%> clauses", t);
remove = true;
}
else
bitmap_set_bit (&aligned_head, DECL_UID (t));
t = OMP_CLAUSE_ALIGNED_ALIGNMENT (c);
if (t == error_mark_node)
remove = true;
else if (t == NULL_TREE)
break;
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t)))
{
error ("%<aligned%> clause alignment expression must "
"be integral");
remove = true;
}
else
{
t = mark_rvalue_use (t);
if (!processing_template_decl)
{
t = maybe_constant_value (t);
if (TREE_CODE (t) != INTEGER_CST
|| tree_int_cst_sgn (t) != 1)
{
error ("%<aligned%> clause alignment expression must be "
"positive constant integer expression");
remove = true;
}
}
OMP_CLAUSE_ALIGNED_ALIGNMENT (c) = t;
}
break;
case OMP_CLAUSE_DEPEND:
t = OMP_CLAUSE_DECL (c);
if (t == NULL_TREE)
{
gcc_assert (OMP_CLAUSE_DEPEND_KIND (c)
== OMP_CLAUSE_DEPEND_SOURCE);
break;
}
if (OMP_CLAUSE_DEPEND_KIND (c) == OMP_CLAUSE_DEPEND_SINK)
{
if (cp_finish_omp_clause_depend_sink (c))
remove = true;
break;
}
if (TREE_CODE (t) == TREE_LIST)
{
if (handle_omp_array_sections (c, ort))
remove = true;
break;
}
if (t == error_mark_node)
remove = true;
else if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL)
{
if (processing_template_decl && TREE_CODE (t) != OVERLOAD)
break;
if (DECL_P (t))
error ("%qD is not a variable in %<depend%> clause", t);
else
error ("%qE is not a variable in %<depend%> clause", t);
remove = true;
}
else if (t == current_class_ptr)
{
error ("%<this%> allowed in OpenMP only in %<declare simd%>"
" clauses");
remove = true;
}
else if (!processing_template_decl
&& !cxx_mark_addressable (t))
remove = true;
break;
case OMP_CLAUSE_MAP:
case OMP_CLAUSE_TO:
case OMP_CLAUSE_FROM:
case OMP_CLAUSE__CACHE_:
t = OMP_CLAUSE_DECL (c);
if (TREE_CODE (t) == TREE_LIST)
{
if (handle_omp_array_sections (c, ort))
remove = true;
else
{
t = OMP_CLAUSE_DECL (c);
if (TREE_CODE (t) != TREE_LIST
&& !type_dependent_expression_p (t)
&& !cp_omp_mappable_type (TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"array section does not have mappable type "
"in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
while (TREE_CODE (t) == ARRAY_REF)
t = TREE_OPERAND (t, 0);
if (TREE_CODE (t) == COMPONENT_REF
&& TREE_CODE (TREE_TYPE (t)) == ARRAY_TYPE)
{
while (TREE_CODE (t) == COMPONENT_REF)
t = TREE_OPERAND (t, 0);
if (REFERENCE_REF_P (t))
t = TREE_OPERAND (t, 0);
if (bitmap_bit_p (&map_field_head, DECL_UID (t)))
break;
if (bitmap_bit_p (&map_head, DECL_UID (t)))
{
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_MAP)
error ("%qD appears more than once in motion"
" clauses", t);
else if (ort == C_ORT_ACC)
error ("%qD appears more than once in data"
" clauses", t);
else
error ("%qD appears more than once in map"
" clauses", t);
remove = true;
}
else
{
bitmap_set_bit (&map_head, DECL_UID (t));
bitmap_set_bit (&map_field_head, DECL_UID (t));
}
}
}
break;
}
if (t == error_mark_node)
{
remove = true;
break;
}
if (REFERENCE_REF_P (t)
&& TREE_CODE (TREE_OPERAND (t, 0)) == COMPONENT_REF)
{
t = TREE_OPERAND (t, 0);
OMP_CLAUSE_DECL (c) = t;
}
if (TREE_CODE (t) == COMPONENT_REF
&& (ort & C_ORT_OMP_DECLARE_SIMD) == C_ORT_OMP
&& OMP_CLAUSE_CODE (c) != OMP_CLAUSE__CACHE_)
{
if (type_dependent_expression_p (t))
break;
if (TREE_CODE (TREE_OPERAND (t, 1)) == FIELD_DECL
&& DECL_BIT_FIELD (TREE_OPERAND (t, 1)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"bit-field %qE in %qs clause",
t, omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (!cp_omp_mappable_type (TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE does not have a mappable type in %qs clause",
t, omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
while (TREE_CODE (t) == COMPONENT_REF)
{
if (TREE_TYPE (TREE_OPERAND (t, 0))
&& (TREE_CODE (TREE_TYPE (TREE_OPERAND (t, 0)))
== UNION_TYPE))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is a member of a union", t);
remove = true;
break;
}
t = TREE_OPERAND (t, 0);
}
if (remove)
break;
if (REFERENCE_REF_P (t))
t = TREE_OPERAND (t, 0);
if (VAR_P (t) || TREE_CODE (t) == PARM_DECL)
{
if (bitmap_bit_p (&map_field_head, DECL_UID (t)))
goto handle_map_references;
}
}
if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL)
{
if (processing_template_decl && TREE_CODE (t) != OVERLOAD)
break;
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_MAP
&& (OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_POINTER
|| OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_ALWAYS_POINTER))
break;
if (DECL_P (t))
error ("%qD is not a variable in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
else
error ("%qE is not a variable in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (VAR_P (t) && CP_DECL_THREAD_LOCAL_P (t))
{
error ("%qD is threadprivate variable in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (ort != C_ORT_ACC && t == current_class_ptr)
{
error ("%<this%> allowed in OpenMP only in %<declare simd%>"
" clauses");
remove = true;
break;
}
else if (!processing_template_decl
&& TREE_CODE (TREE_TYPE (t)) != REFERENCE_TYPE
&& (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_MAP
|| (OMP_CLAUSE_MAP_KIND (c)
!= GOMP_MAP_FIRSTPRIVATE_POINTER))
&& !cxx_mark_addressable (t))
remove = true;
else if (!(OMP_CLAUSE_CODE (c) == OMP_CLAUSE_MAP
&& (OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_POINTER
|| (OMP_CLAUSE_MAP_KIND (c)
== GOMP_MAP_FIRSTPRIVATE_POINTER)))
&& t == OMP_CLAUSE_DECL (c)
&& !type_dependent_expression_p (t)
&& !cp_omp_mappable_type ((TREE_CODE (TREE_TYPE (t))
== REFERENCE_TYPE)
? TREE_TYPE (TREE_TYPE (t))
: TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qD does not have a mappable type in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_MAP
&& OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_FORCE_DEVICEPTR
&& !type_dependent_expression_p (t)
&& !POINTER_TYPE_P (TREE_TYPE (t)))
{
error ("%qD is not a pointer variable", t);
remove = true;
}
else if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_MAP
&& OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_FIRSTPRIVATE_POINTER)
{
if (bitmap_bit_p (&generic_head, DECL_UID (t))
|| bitmap_bit_p (&firstprivate_head, DECL_UID (t)))
{
error ("%qD appears more than once in data clauses", t);
remove = true;
}
else if (bitmap_bit_p (&map_head, DECL_UID (t)))
{
if (ort == C_ORT_ACC)
error ("%qD appears more than once in data clauses", t);
else
error ("%qD appears both in data and map clauses", t);
remove = true;
}
else
bitmap_set_bit (&generic_head, DECL_UID (t));
}
else if (bitmap_bit_p (&map_head, DECL_UID (t)))
{
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_MAP)
error ("%qD appears more than once in motion clauses", t);
if (ort == C_ORT_ACC)
error ("%qD appears more than once in data clauses", t);
else
error ("%qD appears more than once in map clauses", t);
remove = true;
}
else if (bitmap_bit_p (&generic_head, DECL_UID (t))
|| bitmap_bit_p (&firstprivate_head, DECL_UID (t)))
{
if (ort == C_ORT_ACC)
error ("%qD appears more than once in data clauses", t);
else
error ("%qD appears both in data and map clauses", t);
remove = true;
}
else
{
bitmap_set_bit (&map_head, DECL_UID (t));
if (t != OMP_CLAUSE_DECL (c)
&& TREE_CODE (OMP_CLAUSE_DECL (c)) == COMPONENT_REF)
bitmap_set_bit (&map_field_head, DECL_UID (t));
}
handle_map_references:
if (!remove
&& !processing_template_decl
&& (ort & C_ORT_OMP_DECLARE_SIMD) == C_ORT_OMP
&& TREE_CODE (TREE_TYPE (OMP_CLAUSE_DECL (c))) == REFERENCE_TYPE)
{
t = OMP_CLAUSE_DECL (c);
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_MAP)
{
OMP_CLAUSE_DECL (c) = build_simple_mem_ref (t);
if (OMP_CLAUSE_SIZE (c) == NULL_TREE)
OMP_CLAUSE_SIZE (c)
= TYPE_SIZE_UNIT (TREE_TYPE (TREE_TYPE (t)));
}
else if (OMP_CLAUSE_MAP_KIND (c)
!= GOMP_MAP_FIRSTPRIVATE_POINTER
&& (OMP_CLAUSE_MAP_KIND (c)
!= GOMP_MAP_FIRSTPRIVATE_REFERENCE)
&& (OMP_CLAUSE_MAP_KIND (c)
!= GOMP_MAP_ALWAYS_POINTER))
{
tree c2 = build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_MAP);
if (TREE_CODE (t) == COMPONENT_REF)
OMP_CLAUSE_SET_MAP_KIND (c2, GOMP_MAP_ALWAYS_POINTER);
else
OMP_CLAUSE_SET_MAP_KIND (c2,
GOMP_MAP_FIRSTPRIVATE_REFERENCE);
OMP_CLAUSE_DECL (c2) = t;
OMP_CLAUSE_SIZE (c2) = size_zero_node;
OMP_CLAUSE_CHAIN (c2) = OMP_CLAUSE_CHAIN (c);
OMP_CLAUSE_CHAIN (c) = c2;
OMP_CLAUSE_DECL (c) = build_simple_mem_ref (t);
if (OMP_CLAUSE_SIZE (c) == NULL_TREE)
OMP_CLAUSE_SIZE (c)
= TYPE_SIZE_UNIT (TREE_TYPE (TREE_TYPE (t)));
c = c2;
}
}
break;
case OMP_CLAUSE_TO_DECLARE:
case OMP_CLAUSE_LINK:
t = OMP_CLAUSE_DECL (c);
if (TREE_CODE (t) == FUNCTION_DECL
&& OMP_CLAUSE_CODE (c) == OMP_CLAUSE_TO_DECLARE)
;
else if (!VAR_P (t))
{
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_TO_DECLARE)
{
if (TREE_CODE (t) == TEMPLATE_ID_EXPR)
error_at (OMP_CLAUSE_LOCATION (c),
"template %qE in clause %qs", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
else if (really_overloaded_fn (t))
error_at (OMP_CLAUSE_LOCATION (c),
"overloaded function name %qE in clause %qs", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
else
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is neither a variable nor a function name "
"in clause %qs", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
}
else
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is not a variable in clause %qs", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (DECL_THREAD_LOCAL_P (t))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qD is threadprivate variable in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (!cp_omp_mappable_type (TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qD does not have a mappable type in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
if (remove)
break;
if (bitmap_bit_p (&generic_head, DECL_UID (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE appears more than once on the same "
"%<declare target%> directive", t);
remove = true;
}
else
bitmap_set_bit (&generic_head, DECL_UID (t));
break;
case OMP_CLAUSE_UNIFORM:
t = OMP_CLAUSE_DECL (c);
if (TREE_CODE (t) != PARM_DECL)
{
if (processing_template_decl)
break;
if (DECL_P (t))
error ("%qD is not an argument in %<uniform%> clause", t);
else
error ("%qE is not an argument in %<uniform%> clause", t);
remove = true;
break;
}
bitmap_set_bit (&map_head, DECL_UID (t));
goto check_dup_generic;
case OMP_CLAUSE_GRAINSIZE:
t = OMP_CLAUSE_GRAINSIZE_EXPR (c);
if (t == error_mark_node)
remove = true;
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t)))
{
error ("%<grainsize%> expression must be integral");
remove = true;
}
else
{
t = mark_rvalue_use (t);
if (!processing_template_decl)
{
t = maybe_constant_value (t);
if (TREE_CODE (t) == INTEGER_CST
&& tree_int_cst_sgn (t) != 1)
{
warning_at (OMP_CLAUSE_LOCATION (c), 0,
"%<grainsize%> value must be positive");
t = integer_one_node;
}
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
}
OMP_CLAUSE_GRAINSIZE_EXPR (c) = t;
}
break;
case OMP_CLAUSE_PRIORITY:
t = OMP_CLAUSE_PRIORITY_EXPR (c);
if (t == error_mark_node)
remove = true;
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t)))
{
error ("%<priority%> expression must be integral");
remove = true;
}
else
{
t = mark_rvalue_use (t);
if (!processing_template_decl)
{
t = maybe_constant_value (t);
if (TREE_CODE (t) == INTEGER_CST
&& tree_int_cst_sgn (t) == -1)
{
warning_at (OMP_CLAUSE_LOCATION (c), 0,
"%<priority%> value must be non-negative");
t = integer_one_node;
}
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
}
OMP_CLAUSE_PRIORITY_EXPR (c) = t;
}
break;
case OMP_CLAUSE_HINT:
t = OMP_CLAUSE_HINT_EXPR (c);
if (t == error_mark_node)
remove = true;
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t)))
{
error ("%<num_tasks%> expression must be integral");
remove = true;
}
else
{
t = mark_rvalue_use (t);
if (!processing_template_decl)
{
t = maybe_constant_value (t);
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
}
OMP_CLAUSE_HINT_EXPR (c) = t;
}
break;
case OMP_CLAUSE_IS_DEVICE_PTR:
case OMP_CLAUSE_USE_DEVICE_PTR:
field_ok = (ort & C_ORT_OMP_DECLARE_SIMD) == C_ORT_OMP;
t = OMP_CLAUSE_DECL (c);
if (!type_dependent_expression_p (t))
{
tree type = TREE_TYPE (t);
if (TREE_CODE (type) != POINTER_TYPE
&& TREE_CODE (type) != ARRAY_TYPE
&& (TREE_CODE (type) != REFERENCE_TYPE
|| (TREE_CODE (TREE_TYPE (type)) != POINTER_TYPE
&& TREE_CODE (TREE_TYPE (type)) != ARRAY_TYPE)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qs variable is neither a pointer, nor an array "
"nor reference to pointer or array",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
}
goto check_dup_generic;
case OMP_CLAUSE_NOWAIT:
case OMP_CLAUSE_DEFAULT:
case OMP_CLAUSE_UNTIED:
case OMP_CLAUSE_COLLAPSE:
case OMP_CLAUSE_MERGEABLE:
case OMP_CLAUSE_PARALLEL:
case OMP_CLAUSE_FOR:
case OMP_CLAUSE_SECTIONS:
case OMP_CLAUSE_TASKGROUP:
case OMP_CLAUSE_PROC_BIND:
case OMP_CLAUSE_NOGROUP:
case OMP_CLAUSE_THREADS:
case OMP_CLAUSE_SIMD:
case OMP_CLAUSE_DEFAULTMAP:
case OMP_CLAUSE_AUTO:
case OMP_CLAUSE_INDEPENDENT:
case OMP_CLAUSE_SEQ:
break;
case OMP_CLAUSE_TILE:
for (tree list = OMP_CLAUSE_TILE_LIST (c); !remove && list;
list = TREE_CHAIN (list))
{
t = TREE_VALUE (list);
if (t == error_mark_node)
remove = true;
else if (!type_dependent_expression_p (t)
&& !INTEGRAL_TYPE_P (TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<tile%> argument needs integral type");
remove = true;
}
else
{
t = mark_rvalue_use (t);
if (!processing_template_decl)
{
t = maybe_constant_value (t);
if (!tree_fits_shwi_p (t)
|| tree_to_shwi (t) < 0)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<tile%> argument needs positive "
"integral constant");
remove = true;
}
t = fold_build_cleanup_point_expr (TREE_TYPE (t), t);
}
}
TREE_VALUE (list) = t;
}
break;
case OMP_CLAUSE_ORDERED:
ordered_seen = true;
break;
case OMP_CLAUSE_INBRANCH:
case OMP_CLAUSE_NOTINBRANCH:
if (branch_seen)
{
error ("%<inbranch%> clause is incompatible with "
"%<notinbranch%>");
remove = true;
}
branch_seen = true;
break;
default:
gcc_unreachable ();
}
if (remove)
*pc = OMP_CLAUSE_CHAIN (c);
else
pc = &OMP_CLAUSE_CHAIN (c);
}
for (pc = &clauses, c = clauses; c ; c = *pc)
{
enum omp_clause_code c_kind = OMP_CLAUSE_CODE (c);
bool remove = false;
bool need_complete_type = false;
bool need_default_ctor = false;
bool need_copy_ctor = false;
bool need_copy_assignment = false;
bool need_implicitly_determined = false;
bool need_dtor = false;
tree type, inner_type;
switch (c_kind)
{
case OMP_CLAUSE_SHARED:
need_implicitly_determined = true;
break;
case OMP_CLAUSE_PRIVATE:
need_complete_type = true;
need_default_ctor = true;
need_dtor = true;
need_implicitly_determined = true;
break;
case OMP_CLAUSE_FIRSTPRIVATE:
need_complete_type = true;
need_copy_ctor = true;
need_dtor = true;
need_implicitly_determined = true;
break;
case OMP_CLAUSE_LASTPRIVATE:
need_complete_type = true;
need_copy_assignment = true;
need_implicitly_determined = true;
break;
case OMP_CLAUSE_REDUCTION:
need_implicitly_determined = true;
break;
case OMP_CLAUSE_LINEAR:
if (ort != C_ORT_OMP_DECLARE_SIMD)
need_implicitly_determined = true;
else if (OMP_CLAUSE_LINEAR_VARIABLE_STRIDE (c)
&& !bitmap_bit_p (&map_head,
DECL_UID (OMP_CLAUSE_LINEAR_STEP (c))))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<linear%> clause step is a parameter %qD not "
"specified in %<uniform%> clause",
OMP_CLAUSE_LINEAR_STEP (c));
*pc = OMP_CLAUSE_CHAIN (c);
continue;
}
break;
case OMP_CLAUSE_COPYPRIVATE:
need_copy_assignment = true;
break;
case OMP_CLAUSE_COPYIN:
need_copy_assignment = true;
break;
case OMP_CLAUSE_SIMDLEN:
if (safelen
&& !processing_template_decl
&& tree_int_cst_lt (OMP_CLAUSE_SAFELEN_EXPR (safelen),
OMP_CLAUSE_SIMDLEN_EXPR (c)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<simdlen%> clause value is bigger than "
"%<safelen%> clause value");
OMP_CLAUSE_SIMDLEN_EXPR (c)
= OMP_CLAUSE_SAFELEN_EXPR (safelen);
}
pc = &OMP_CLAUSE_CHAIN (c);
continue;
case OMP_CLAUSE_SCHEDULE:
if (ordered_seen
&& (OMP_CLAUSE_SCHEDULE_KIND (c)
& OMP_CLAUSE_SCHEDULE_NONMONOTONIC))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<nonmonotonic%> schedule modifier specified "
"together with %<ordered%> clause");
OMP_CLAUSE_SCHEDULE_KIND (c)
= (enum omp_clause_schedule_kind)
(OMP_CLAUSE_SCHEDULE_KIND (c)
& ~OMP_CLAUSE_SCHEDULE_NONMONOTONIC);
}
pc = &OMP_CLAUSE_CHAIN (c);
continue;
case OMP_CLAUSE_NOWAIT:
if (copyprivate_seen)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<nowait%> clause must not be used together "
"with %<copyprivate%>");
*pc = OMP_CLAUSE_CHAIN (c);
continue;
}
default:
pc = &OMP_CLAUSE_CHAIN (c);
continue;
}
t = OMP_CLAUSE_DECL (c);
if (processing_template_decl
&& !VAR_P (t) && TREE_CODE (t) != PARM_DECL)
{
pc = &OMP_CLAUSE_CHAIN (c);
continue;
}
switch (c_kind)
{
case OMP_CLAUSE_LASTPRIVATE:
if (!bitmap_bit_p (&firstprivate_head, DECL_UID (t)))
{
need_default_ctor = true;
need_dtor = true;
}
break;
case OMP_CLAUSE_REDUCTION:
if (finish_omp_reduction_clause (c, &need_default_ctor,
&need_dtor))
remove = true;
else
t = OMP_CLAUSE_DECL (c);
break;
case OMP_CLAUSE_COPYIN:
if (!VAR_P (t) || !CP_DECL_THREAD_LOCAL_P (t))
{
error ("%qE must be %<threadprivate%> for %<copyin%>", t);
remove = true;
}
break;
default:
break;
}
if (need_complete_type || need_copy_assignment)
{
t = require_complete_type (t);
if (t == error_mark_node)
remove = true;
else if (TREE_CODE (TREE_TYPE (t)) == REFERENCE_TYPE
&& !complete_type_or_else (TREE_TYPE (TREE_TYPE (t)), t))
remove = true;
}
if (need_implicitly_determined)
{
const char *share_name = NULL;
if (VAR_P (t) && CP_DECL_THREAD_LOCAL_P (t))
share_name = "threadprivate";
else switch (cxx_omp_predetermined_sharing_1 (t))
{
case OMP_CLAUSE_DEFAULT_UNSPECIFIED:
break;
case OMP_CLAUSE_DEFAULT_SHARED:
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_FIRSTPRIVATE
&& cxx_omp_const_qual_no_mutable (t))
break;
share_name = "shared";
break;
case OMP_CLAUSE_DEFAULT_PRIVATE:
share_name = "private";
break;
default:
gcc_unreachable ();
}
if (share_name)
{
error ("%qE is predetermined %qs for %qs",
omp_clause_printable_decl (t), share_name,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
}
inner_type = type = TREE_TYPE (t);
if ((need_complete_type
|| need_copy_assignment
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION)
&& TREE_CODE (inner_type) == REFERENCE_TYPE)
inner_type = TREE_TYPE (inner_type);
while (TREE_CODE (inner_type) == ARRAY_TYPE)
inner_type = TREE_TYPE (inner_type);
if (CLASS_TYPE_P (inner_type)
&& COMPLETE_TYPE_P (inner_type)
&& (need_default_ctor || need_copy_ctor
|| need_copy_assignment || need_dtor)
&& !type_dependent_expression_p (t)
&& cxx_omp_create_clause_info (c, inner_type, need_default_ctor,
need_copy_ctor, need_copy_assignment,
need_dtor))
remove = true;
if (!remove
&& c_kind == OMP_CLAUSE_SHARED
&& processing_template_decl)
{
t = omp_clause_decl_field (OMP_CLAUSE_DECL (c));
if (t)
OMP_CLAUSE_DECL (c) = t;
}
if (remove)
*pc = OMP_CLAUSE_CHAIN (c);
else
pc = &OMP_CLAUSE_CHAIN (c);
}
bitmap_obstack_release (NULL);
return clauses;
}
tree
push_omp_privatization_clauses (bool ignore_next)
{
if (omp_private_member_ignore_next)
{
omp_private_member_ignore_next = ignore_next;
return NULL_TREE;
}
omp_private_member_ignore_next = ignore_next;
if (omp_private_member_map)
omp_private_member_vec.safe_push (error_mark_node);
return push_stmt_list ();
}
void
pop_omp_privatization_clauses (tree stmt)
{
if (stmt == NULL_TREE)
return;
stmt = pop_stmt_list (stmt);
if (omp_private_member_map)
{
while (!omp_private_member_vec.is_empty ())
{
tree t = omp_private_member_vec.pop ();
if (t == error_mark_node)
{
add_stmt (stmt);
return;
}
bool no_decl_expr = t == integer_zero_node;
if (no_decl_expr)
t = omp_private_member_vec.pop ();
tree *v = omp_private_member_map->get (t);
gcc_assert (v);
if (!no_decl_expr)
add_decl_expr (*v);
omp_private_member_map->remove (t);
}
delete omp_private_member_map;
omp_private_member_map = NULL;
}
add_stmt (stmt);
}
void
save_omp_privatization_clauses (vec<tree> &save)
{
save = vNULL;
if (omp_private_member_ignore_next)
save.safe_push (integer_one_node);
omp_private_member_ignore_next = false;
if (!omp_private_member_map)
return;
while (!omp_private_member_vec.is_empty ())
{
tree t = omp_private_member_vec.pop ();
if (t == error_mark_node)
{
save.safe_push (t);
continue;
}
tree n = t;
if (t == integer_zero_node)
t = omp_private_member_vec.pop ();
tree *v = omp_private_member_map->get (t);
gcc_assert (v);
save.safe_push (*v);
save.safe_push (t);
if (n != t)
save.safe_push (n);
}
delete omp_private_member_map;
omp_private_member_map = NULL;
}
void
restore_omp_privatization_clauses (vec<tree> &save)
{
gcc_assert (omp_private_member_vec.is_empty ());
omp_private_member_ignore_next = false;
if (save.is_empty ())
return;
if (save.length () == 1 && save[0] == integer_one_node)
{
omp_private_member_ignore_next = true;
save.release ();
return;
}
omp_private_member_map = new hash_map <tree, tree>;
while (!save.is_empty ())
{
tree t = save.pop ();
tree n = t;
if (t != error_mark_node)
{
if (t == integer_one_node)
{
omp_private_member_ignore_next = true;
gcc_assert (save.is_empty ());
break;
}
if (t == integer_zero_node)
t = save.pop ();
tree &v = omp_private_member_map->get_or_insert (t);
v = save.pop ();
}
omp_private_member_vec.safe_push (t);
if (n != t)
omp_private_member_vec.safe_push (n);
}
save.release ();
}
void
finish_omp_threadprivate (tree vars)
{
tree t;
for (t = vars; t; t = TREE_CHAIN (t))
{
tree v = TREE_PURPOSE (t);
if (error_operand_p (v))
;
else if (!VAR_P (v))
error ("%<threadprivate%> %qD is not file, namespace "
"or block scope variable", v);
else if (TREE_USED (v)
&& (DECL_LANG_SPECIFIC (v) == NULL
|| !CP_DECL_THREADPRIVATE_P (v)))
error ("%qE declared %<threadprivate%> after first use", v);
else if (! TREE_STATIC (v) && ! DECL_EXTERNAL (v))
error ("automatic variable %qE cannot be %<threadprivate%>", v);
else if (! COMPLETE_TYPE_P (complete_type (TREE_TYPE (v))))
error ("%<threadprivate%> %qE has incomplete type", v);
else if (TREE_STATIC (v) && TYPE_P (CP_DECL_CONTEXT (v))
&& CP_DECL_CONTEXT (v) != current_class_type)
error ("%<threadprivate%> %qE directive not "
"in %qT definition", v, CP_DECL_CONTEXT (v));
else
{
if (DECL_LANG_SPECIFIC (v) == NULL)
{
retrofit_lang_decl (v);
if (DECL_DISCRIMINATOR_P (v))
DECL_LANG_SPECIFIC (v)->u.base.u2sel = 1;
}
if (! CP_DECL_THREAD_LOCAL_P (v))
{
CP_DECL_THREAD_LOCAL_P (v) = true;
set_decl_tls_model (v, decl_default_tls_model (v));
if (DECL_RTL_SET_P (v))
make_decl_rtl (v);
}
CP_DECL_THREADPRIVATE_P (v) = 1;
}
}
}
tree
begin_omp_structured_block (void)
{
return do_pushlevel (sk_omp);
}
tree
finish_omp_structured_block (tree block)
{
return do_poplevel (block);
}
tree
begin_omp_parallel (void)
{
keep_next_level (true);
return begin_omp_structured_block ();
}
tree
finish_oacc_data (tree clauses, tree block)
{
tree stmt;
block = finish_omp_structured_block (block);
stmt = make_node (OACC_DATA);
TREE_TYPE (stmt) = void_type_node;
OACC_DATA_CLAUSES (stmt) = clauses;
OACC_DATA_BODY (stmt) = block;
return add_stmt (stmt);
}
tree
finish_oacc_host_data (tree clauses, tree block)
{
tree stmt;
block = finish_omp_structured_block (block);
stmt = make_node (OACC_HOST_DATA);
TREE_TYPE (stmt) = void_type_node;
OACC_HOST_DATA_CLAUSES (stmt) = clauses;
OACC_HOST_DATA_BODY (stmt) = block;
return add_stmt (stmt);
}
tree
finish_omp_construct (enum tree_code code, tree body, tree clauses)
{
body = finish_omp_structured_block (body);
tree stmt = make_node (code);
TREE_TYPE (stmt) = void_type_node;
OMP_BODY (stmt) = body;
OMP_CLAUSES (stmt) = clauses;
return add_stmt (stmt);
}
tree
finish_omp_parallel (tree clauses, tree body)
{
tree stmt;
body = finish_omp_structured_block (body);
stmt = make_node (OMP_PARALLEL);
TREE_TYPE (stmt) = void_type_node;
OMP_PARALLEL_CLAUSES (stmt) = clauses;
OMP_PARALLEL_BODY (stmt) = body;
return add_stmt (stmt);
}
tree
begin_omp_task (void)
{
keep_next_level (true);
return begin_omp_structured_block ();
}
tree
finish_omp_task (tree clauses, tree body)
{
tree stmt;
body = finish_omp_structured_block (body);
stmt = make_node (OMP_TASK);
TREE_TYPE (stmt) = void_type_node;
OMP_TASK_CLAUSES (stmt) = clauses;
OMP_TASK_BODY (stmt) = body;
return add_stmt (stmt);
}
static bool
handle_omp_for_class_iterator (int i, location_t locus, enum tree_code code,
tree declv, tree orig_declv, tree initv,
tree condv, tree incrv, tree *body,
tree *pre_body, tree &clauses, tree *lastp,
int collapse, int ordered)
{
tree diff, iter_init, iter_incr = NULL, last;
tree incr_var = NULL, orig_pre_body, orig_body, c;
tree decl = TREE_VEC_ELT (declv, i);
tree init = TREE_VEC_ELT (initv, i);
tree cond = TREE_VEC_ELT (condv, i);
tree incr = TREE_VEC_ELT (incrv, i);
tree iter = decl;
location_t elocus = locus;
if (init && EXPR_HAS_LOCATION (init))
elocus = EXPR_LOCATION (init);
cond = cp_fully_fold (cond);
switch (TREE_CODE (cond))
{
case GT_EXPR:
case GE_EXPR:
case LT_EXPR:
case LE_EXPR:
case NE_EXPR:
if (TREE_OPERAND (cond, 1) == iter)
cond = build2 (swap_tree_comparison (TREE_CODE (cond)),
TREE_TYPE (cond), iter, TREE_OPERAND (cond, 0));
if (TREE_OPERAND (cond, 0) != iter)
cond = error_mark_node;
else
{
tree tem = build_x_binary_op (EXPR_LOCATION (cond),
TREE_CODE (cond),
iter, ERROR_MARK,
TREE_OPERAND (cond, 1), ERROR_MARK,
NULL, tf_warning_or_error);
if (error_operand_p (tem))
return true;
}
break;
default:
cond = error_mark_node;
break;
}
if (cond == error_mark_node)
{
error_at (elocus, "invalid controlling predicate");
return true;
}
diff = build_x_binary_op (elocus, MINUS_EXPR, TREE_OPERAND (cond, 1),
ERROR_MARK, iter, ERROR_MARK, NULL,
tf_warning_or_error);
diff = cp_fully_fold (diff);
if (error_operand_p (diff))
return true;
if (TREE_CODE (TREE_TYPE (diff)) != INTEGER_TYPE)
{
error_at (elocus, "difference between %qE and %qD does not have integer type",
TREE_OPERAND (cond, 1), iter);
return true;
}
if (!c_omp_check_loop_iv_exprs (locus, orig_declv,
TREE_VEC_ELT (declv, i), NULL_TREE,
cond, cp_walk_subtrees))
return true;
switch (TREE_CODE (incr))
{
case PREINCREMENT_EXPR:
case PREDECREMENT_EXPR:
case POSTINCREMENT_EXPR:
case POSTDECREMENT_EXPR:
if (TREE_OPERAND (incr, 0) != iter)
{
incr = error_mark_node;
break;
}
iter_incr = build_x_unary_op (EXPR_LOCATION (incr),
TREE_CODE (incr), iter,
tf_warning_or_error);
if (error_operand_p (iter_incr))
return true;
else if (TREE_CODE (incr) == PREINCREMENT_EXPR
|| TREE_CODE (incr) == POSTINCREMENT_EXPR)
incr = integer_one_node;
else
incr = integer_minus_one_node;
break;
case MODIFY_EXPR:
if (TREE_OPERAND (incr, 0) != iter)
incr = error_mark_node;
else if (TREE_CODE (TREE_OPERAND (incr, 1)) == PLUS_EXPR
|| TREE_CODE (TREE_OPERAND (incr, 1)) == MINUS_EXPR)
{
tree rhs = TREE_OPERAND (incr, 1);
if (TREE_OPERAND (rhs, 0) == iter)
{
if (TREE_CODE (TREE_TYPE (TREE_OPERAND (rhs, 1)))
!= INTEGER_TYPE)
incr = error_mark_node;
else
{
iter_incr = build_x_modify_expr (EXPR_LOCATION (rhs),
iter, TREE_CODE (rhs),
TREE_OPERAND (rhs, 1),
tf_warning_or_error);
if (error_operand_p (iter_incr))
return true;
incr = TREE_OPERAND (rhs, 1);
incr = cp_convert (TREE_TYPE (diff), incr,
tf_warning_or_error);
if (TREE_CODE (rhs) == MINUS_EXPR)
{
incr = build1 (NEGATE_EXPR, TREE_TYPE (diff), incr);
incr = fold_simple (incr);
}
if (TREE_CODE (incr) != INTEGER_CST
&& (TREE_CODE (incr) != NOP_EXPR
|| (TREE_CODE (TREE_OPERAND (incr, 0))
!= INTEGER_CST)))
iter_incr = NULL;
}
}
else if (TREE_OPERAND (rhs, 1) == iter)
{
if (TREE_CODE (TREE_TYPE (TREE_OPERAND (rhs, 0))) != INTEGER_TYPE
|| TREE_CODE (rhs) != PLUS_EXPR)
incr = error_mark_node;
else
{
iter_incr = build_x_binary_op (EXPR_LOCATION (rhs),
PLUS_EXPR,
TREE_OPERAND (rhs, 0),
ERROR_MARK, iter,
ERROR_MARK, NULL,
tf_warning_or_error);
if (error_operand_p (iter_incr))
return true;
iter_incr = build_x_modify_expr (EXPR_LOCATION (rhs),
iter, NOP_EXPR,
iter_incr,
tf_warning_or_error);
if (error_operand_p (iter_incr))
return true;
incr = TREE_OPERAND (rhs, 0);
iter_incr = NULL;
}
}
else
incr = error_mark_node;
}
else
incr = error_mark_node;
break;
default:
incr = error_mark_node;
break;
}
if (incr == error_mark_node)
{
error_at (elocus, "invalid increment expression");
return true;
}
incr = cp_convert (TREE_TYPE (diff), incr, tf_warning_or_error);
bool taskloop_iv_seen = false;
for (c = clauses; c ; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LASTPRIVATE
&& OMP_CLAUSE_DECL (c) == iter)
{
if (code == OMP_TASKLOOP)
{
taskloop_iv_seen = true;
OMP_CLAUSE_LASTPRIVATE_TASKLOOP_IV (c) = 1;
}
break;
}
else if (code == OMP_TASKLOOP
&& OMP_CLAUSE_CODE (c) == OMP_CLAUSE_PRIVATE
&& OMP_CLAUSE_DECL (c) == iter)
{
taskloop_iv_seen = true;
OMP_CLAUSE_PRIVATE_TASKLOOP_IV (c) = 1;
}
decl = create_temporary_var (TREE_TYPE (diff));
pushdecl (decl);
add_decl_expr (decl);
last = create_temporary_var (TREE_TYPE (diff));
pushdecl (last);
add_decl_expr (last);
if (c && iter_incr == NULL && TREE_CODE (incr) != INTEGER_CST
&& (!ordered || (i < collapse && collapse > 1)))
{
incr_var = create_temporary_var (TREE_TYPE (diff));
pushdecl (incr_var);
add_decl_expr (incr_var);
}
gcc_assert (stmts_are_full_exprs_p ());
tree diffvar = NULL_TREE;
if (code == OMP_TASKLOOP)
{
if (!taskloop_iv_seen)
{
tree ivc = build_omp_clause (locus, OMP_CLAUSE_FIRSTPRIVATE);
OMP_CLAUSE_DECL (ivc) = iter;
cxx_omp_finish_clause (ivc, NULL);
OMP_CLAUSE_CHAIN (ivc) = clauses;
clauses = ivc;
}
tree lvc = build_omp_clause (locus, OMP_CLAUSE_FIRSTPRIVATE);
OMP_CLAUSE_DECL (lvc) = last;
OMP_CLAUSE_CHAIN (lvc) = clauses;
clauses = lvc;
diffvar = create_temporary_var (TREE_TYPE (diff));
pushdecl (diffvar);
add_decl_expr (diffvar);
}
orig_pre_body = *pre_body;
*pre_body = push_stmt_list ();
if (orig_pre_body)
add_stmt (orig_pre_body);
if (init != NULL)
finish_expr_stmt (build_x_modify_expr (elocus,
iter, NOP_EXPR, init,
tf_warning_or_error));
init = build_int_cst (TREE_TYPE (diff), 0);
if (c && iter_incr == NULL
&& (!ordered || (i < collapse && collapse > 1)))
{
if (incr_var)
{
finish_expr_stmt (build_x_modify_expr (elocus,
incr_var, NOP_EXPR,
incr, tf_warning_or_error));
incr = incr_var;
}
iter_incr = build_x_modify_expr (elocus,
iter, PLUS_EXPR, incr,
tf_warning_or_error);
}
if (c && ordered && i < collapse && collapse > 1)
iter_incr = incr;
finish_expr_stmt (build_x_modify_expr (elocus,
last, NOP_EXPR, init,
tf_warning_or_error));
if (diffvar)
{
finish_expr_stmt (build_x_modify_expr (elocus,
diffvar, NOP_EXPR,
diff, tf_warning_or_error));
diff = diffvar;
}
*pre_body = pop_stmt_list (*pre_body);
cond = cp_build_binary_op (elocus,
TREE_CODE (cond), decl, diff,
tf_warning_or_error);
incr = build_modify_expr (elocus, decl, NULL_TREE, PLUS_EXPR,
elocus, incr, NULL_TREE);
orig_body = *body;
*body = push_stmt_list ();
iter_init = build2 (MINUS_EXPR, TREE_TYPE (diff), decl, last);
iter_init = build_x_modify_expr (elocus,
iter, PLUS_EXPR, iter_init,
tf_warning_or_error);
if (iter_init != error_mark_node)
iter_init = build1 (NOP_EXPR, void_type_node, iter_init);
finish_expr_stmt (iter_init);
finish_expr_stmt (build_x_modify_expr (elocus,
last, NOP_EXPR, decl,
tf_warning_or_error));
add_stmt (orig_body);
*body = pop_stmt_list (*body);
if (c)
{
OMP_CLAUSE_LASTPRIVATE_STMT (c) = push_stmt_list ();
if (!ordered)
finish_expr_stmt (iter_incr);
else
{
iter_init = decl;
if (i < collapse && collapse > 1 && !error_operand_p (iter_incr))
iter_init = build2 (PLUS_EXPR, TREE_TYPE (diff),
iter_init, iter_incr);
iter_init = build2 (MINUS_EXPR, TREE_TYPE (diff), iter_init, last);
iter_init = build_x_modify_expr (elocus,
iter, PLUS_EXPR, iter_init,
tf_warning_or_error);
if (iter_init != error_mark_node)
iter_init = build1 (NOP_EXPR, void_type_node, iter_init);
finish_expr_stmt (iter_init);
}
OMP_CLAUSE_LASTPRIVATE_STMT (c)
= pop_stmt_list (OMP_CLAUSE_LASTPRIVATE_STMT (c));
}
TREE_VEC_ELT (declv, i) = decl;
TREE_VEC_ELT (initv, i) = init;
TREE_VEC_ELT (condv, i) = cond;
TREE_VEC_ELT (incrv, i) = incr;
*lastp = last;
return false;
}
tree
finish_omp_for (location_t locus, enum tree_code code, tree declv,
tree orig_declv, tree initv, tree condv, tree incrv,
tree body, tree pre_body, vec<tree> *orig_inits, tree clauses)
{
tree omp_for = NULL, orig_incr = NULL;
tree decl = NULL, init, cond, incr;
tree last = NULL_TREE;
location_t elocus;
int i;
int collapse = 1;
int ordered = 0;
gcc_assert (TREE_VEC_LENGTH (declv) == TREE_VEC_LENGTH (initv));
gcc_assert (TREE_VEC_LENGTH (declv) == TREE_VEC_LENGTH (condv));
gcc_assert (TREE_VEC_LENGTH (declv) == TREE_VEC_LENGTH (incrv));
if (TREE_VEC_LENGTH (declv) > 1)
{
tree c;
c = omp_find_clause (clauses, OMP_CLAUSE_TILE);
if (c)
collapse = list_length (OMP_CLAUSE_TILE_LIST (c));
else
{
c = omp_find_clause (clauses, OMP_CLAUSE_COLLAPSE);
if (c)
collapse = tree_to_shwi (OMP_CLAUSE_COLLAPSE_EXPR (c));
if (collapse != TREE_VEC_LENGTH (declv))
ordered = TREE_VEC_LENGTH (declv);
}
}
for (i = 0; i < TREE_VEC_LENGTH (declv); i++)
{
decl = TREE_VEC_ELT (declv, i);
init = TREE_VEC_ELT (initv, i);
cond = TREE_VEC_ELT (condv, i);
incr = TREE_VEC_ELT (incrv, i);
elocus = locus;
if (decl == NULL)
{
if (init != NULL)
switch (TREE_CODE (init))
{
case MODIFY_EXPR:
decl = TREE_OPERAND (init, 0);
init = TREE_OPERAND (init, 1);
break;
case MODOP_EXPR:
if (TREE_CODE (TREE_OPERAND (init, 1)) == NOP_EXPR)
{
decl = TREE_OPERAND (init, 0);
init = TREE_OPERAND (init, 2);
}
break;
default:
break;
}
if (decl == NULL)
{
error_at (locus,
"expected iteration declaration or initialization");
return NULL;
}
}
if (init && EXPR_HAS_LOCATION (init))
elocus = EXPR_LOCATION (init);
if (cond == NULL)
{
error_at (elocus, "missing controlling predicate");
return NULL;
}
if (incr == NULL)
{
error_at (elocus, "missing increment expression");
return NULL;
}
TREE_VEC_ELT (declv, i) = decl;
TREE_VEC_ELT (initv, i) = init;
}
if (orig_inits)
{
bool fail = false;
tree orig_init;
FOR_EACH_VEC_ELT (*orig_inits, i, orig_init)
if (orig_init
&& !c_omp_check_loop_iv_exprs (locus, declv,
TREE_VEC_ELT (declv, i), orig_init,
NULL_TREE, cp_walk_subtrees))
fail = true;
if (fail)
return NULL;
}
if (dependent_omp_for_p (declv, initv, condv, incrv))
{
tree stmt;
stmt = make_node (code);
for (i = 0; i < TREE_VEC_LENGTH (declv); i++)
{
TREE_VEC_ELT (initv, i)
= build2 (MODIFY_EXPR, void_type_node, TREE_VEC_ELT (declv, i),
TREE_VEC_ELT (initv, i));
}
TREE_TYPE (stmt) = void_type_node;
OMP_FOR_INIT (stmt) = initv;
OMP_FOR_COND (stmt) = condv;
OMP_FOR_INCR (stmt) = incrv;
OMP_FOR_BODY (stmt) = body;
OMP_FOR_PRE_BODY (stmt) = pre_body;
OMP_FOR_CLAUSES (stmt) = clauses;
SET_EXPR_LOCATION (stmt, locus);
return add_stmt (stmt);
}
if (!orig_declv)
orig_declv = copy_node (declv);
if (processing_template_decl)
orig_incr = make_tree_vec (TREE_VEC_LENGTH (incrv));
for (i = 0; i < TREE_VEC_LENGTH (declv); )
{
decl = TREE_VEC_ELT (declv, i);
init = TREE_VEC_ELT (initv, i);
cond = TREE_VEC_ELT (condv, i);
incr = TREE_VEC_ELT (incrv, i);
if (orig_incr)
TREE_VEC_ELT (orig_incr, i) = incr;
elocus = locus;
if (init && EXPR_HAS_LOCATION (init))
elocus = EXPR_LOCATION (init);
if (!DECL_P (decl))
{
error_at (elocus, "expected iteration declaration or initialization");
return NULL;
}
if (incr && TREE_CODE (incr) == MODOP_EXPR)
{
if (orig_incr)
TREE_VEC_ELT (orig_incr, i) = incr;
incr = cp_build_modify_expr (elocus, TREE_OPERAND (incr, 0),
TREE_CODE (TREE_OPERAND (incr, 1)),
TREE_OPERAND (incr, 2),
tf_warning_or_error);
}
if (CLASS_TYPE_P (TREE_TYPE (decl)))
{
if (code == OMP_SIMD)
{
error_at (elocus, "%<#pragma omp simd%> used with class "
"iteration variable %qE", decl);
return NULL;
}
if (handle_omp_for_class_iterator (i, locus, code, declv, orig_declv,
initv, condv, incrv, &body,
&pre_body, clauses, &last,
collapse, ordered))
return NULL;
continue;
}
if (!INTEGRAL_TYPE_P (TREE_TYPE (decl))
&& !TYPE_PTR_P (TREE_TYPE (decl)))
{
error_at (elocus, "invalid type for iteration variable %qE", decl);
return NULL;
}
if (!processing_template_decl)
{
init = fold_build_cleanup_point_expr (TREE_TYPE (init), init);
init = cp_build_modify_expr (elocus, decl, NOP_EXPR, init,
tf_warning_or_error);
}
else
init = build2 (MODIFY_EXPR, void_type_node, decl, init);
if (cond
&& TREE_SIDE_EFFECTS (cond)
&& COMPARISON_CLASS_P (cond)
&& !processing_template_decl)
{
tree t = TREE_OPERAND (cond, 0);
if (TREE_SIDE_EFFECTS (t)
&& t != decl
&& (TREE_CODE (t) != NOP_EXPR
|| TREE_OPERAND (t, 0) != decl))
TREE_OPERAND (cond, 0)
= fold_build_cleanup_point_expr (TREE_TYPE (t), t);
t = TREE_OPERAND (cond, 1);
if (TREE_SIDE_EFFECTS (t)
&& t != decl
&& (TREE_CODE (t) != NOP_EXPR
|| TREE_OPERAND (t, 0) != decl))
TREE_OPERAND (cond, 1)
= fold_build_cleanup_point_expr (TREE_TYPE (t), t);
}
if (decl == error_mark_node || init == error_mark_node)
return NULL;
TREE_VEC_ELT (declv, i) = decl;
TREE_VEC_ELT (initv, i) = init;
TREE_VEC_ELT (condv, i) = cond;
TREE_VEC_ELT (incrv, i) = incr;
i++;
}
if (IS_EMPTY_STMT (pre_body))
pre_body = NULL;
omp_for = c_finish_omp_for (locus, code, declv, orig_declv, initv, condv,
incrv, body, pre_body);
if (omp_for && !c_omp_check_loop_iv (omp_for, orig_declv, cp_walk_subtrees))
omp_for = NULL_TREE;
if (omp_for == NULL)
{
return NULL;
}
add_stmt (omp_for);
for (i = 0; i < TREE_VEC_LENGTH (OMP_FOR_INCR (omp_for)); i++)
{
decl = TREE_OPERAND (TREE_VEC_ELT (OMP_FOR_INIT (omp_for), i), 0);
incr = TREE_VEC_ELT (OMP_FOR_INCR (omp_for), i);
if (TREE_CODE (incr) != MODIFY_EXPR)
continue;
if (TREE_SIDE_EFFECTS (TREE_OPERAND (incr, 1))
&& BINARY_CLASS_P (TREE_OPERAND (incr, 1))
&& !processing_template_decl)
{
tree t = TREE_OPERAND (TREE_OPERAND (incr, 1), 0);
if (TREE_SIDE_EFFECTS (t)
&& t != decl
&& (TREE_CODE (t) != NOP_EXPR
|| TREE_OPERAND (t, 0) != decl))
TREE_OPERAND (TREE_OPERAND (incr, 1), 0)
= fold_build_cleanup_point_expr (TREE_TYPE (t), t);
t = TREE_OPERAND (TREE_OPERAND (incr, 1), 1);
if (TREE_SIDE_EFFECTS (t)
&& t != decl
&& (TREE_CODE (t) != NOP_EXPR
|| TREE_OPERAND (t, 0) != decl))
TREE_OPERAND (TREE_OPERAND (incr, 1), 1)
= fold_build_cleanup_point_expr (TREE_TYPE (t), t);
}
if (orig_incr)
TREE_VEC_ELT (OMP_FOR_INCR (omp_for), i) = TREE_VEC_ELT (orig_incr, i);
}
OMP_FOR_CLAUSES (omp_for) = clauses;
if (code == OMP_SIMD && !processing_template_decl
&& TREE_VEC_LENGTH (OMP_FOR_INCR (omp_for)) == 1)
for (tree c = omp_find_clause (clauses, OMP_CLAUSE_LINEAR); c;
c = omp_find_clause (OMP_CLAUSE_CHAIN (c), OMP_CLAUSE_LINEAR))
if (OMP_CLAUSE_LINEAR_STEP (c) == NULL_TREE)
{
decl = TREE_OPERAND (TREE_VEC_ELT (OMP_FOR_INIT (omp_for), 0), 0);
gcc_assert (decl == OMP_CLAUSE_DECL (c));
incr = TREE_VEC_ELT (OMP_FOR_INCR (omp_for), 0);
tree step, stept;
switch (TREE_CODE (incr))
{
case PREINCREMENT_EXPR:
case POSTINCREMENT_EXPR:
gcc_assert (!POINTER_TYPE_P (TREE_TYPE (decl)));
OMP_CLAUSE_LINEAR_STEP (c) = build_int_cst (TREE_TYPE (decl), 1);
break;
case PREDECREMENT_EXPR:
case POSTDECREMENT_EXPR:
gcc_assert (!POINTER_TYPE_P (TREE_TYPE (decl)));
OMP_CLAUSE_LINEAR_STEP (c)
= build_int_cst (TREE_TYPE (decl), -1);
break;
case MODIFY_EXPR:
gcc_assert (TREE_OPERAND (incr, 0) == decl);
incr = TREE_OPERAND (incr, 1);
switch (TREE_CODE (incr))
{
case PLUS_EXPR:
if (TREE_OPERAND (incr, 1) == decl)
step = TREE_OPERAND (incr, 0);
else
step = TREE_OPERAND (incr, 1);
break;
case MINUS_EXPR:
case POINTER_PLUS_EXPR:
gcc_assert (TREE_OPERAND (incr, 0) == decl);
step = TREE_OPERAND (incr, 1);
break;
default:
gcc_unreachable ();
}
stept = TREE_TYPE (decl);
if (POINTER_TYPE_P (stept))
stept = sizetype;
step = fold_convert (stept, step);
if (TREE_CODE (incr) == MINUS_EXPR)
step = fold_build1 (NEGATE_EXPR, stept, step);
OMP_CLAUSE_LINEAR_STEP (c) = step;
break;
default:
gcc_unreachable ();
}
}
return omp_for;
}
void
finish_omp_atomic (enum tree_code code, enum tree_code opcode, tree lhs,
tree rhs, tree v, tree lhs1, tree rhs1, bool seq_cst)
{
tree orig_lhs;
tree orig_rhs;
tree orig_v;
tree orig_lhs1;
tree orig_rhs1;
bool dependent_p;
tree stmt;
orig_lhs = lhs;
orig_rhs = rhs;
orig_v = v;
orig_lhs1 = lhs1;
orig_rhs1 = rhs1;
dependent_p = false;
stmt = NULL_TREE;
if (processing_template_decl)
{
dependent_p = (type_dependent_expression_p (lhs)
|| (rhs && type_dependent_expression_p (rhs))
|| (v && type_dependent_expression_p (v))
|| (lhs1 && type_dependent_expression_p (lhs1))
|| (rhs1 && type_dependent_expression_p (rhs1)));
if (!dependent_p)
{
lhs = build_non_dependent_expr (lhs);
if (rhs)
rhs = build_non_dependent_expr (rhs);
if (v)
v = build_non_dependent_expr (v);
if (lhs1)
lhs1 = build_non_dependent_expr (lhs1);
if (rhs1)
rhs1 = build_non_dependent_expr (rhs1);
}
}
if (!dependent_p)
{
bool swapped = false;
if (rhs1 && cp_tree_equal (lhs, rhs))
{
std::swap (rhs, rhs1);
swapped = !commutative_tree_code (opcode);
}
if (rhs1 && !cp_tree_equal (lhs, rhs1))
{
if (code == OMP_ATOMIC)
error ("%<#pragma omp atomic update%> uses two different "
"expressions for memory");
else
error ("%<#pragma omp atomic capture%> uses two different "
"expressions for memory");
return;
}
if (lhs1 && !cp_tree_equal (lhs, lhs1))
{
if (code == OMP_ATOMIC)
error ("%<#pragma omp atomic update%> uses two different "
"expressions for memory");
else
error ("%<#pragma omp atomic capture%> uses two different "
"expressions for memory");
return;
}
stmt = c_finish_omp_atomic (input_location, code, opcode, lhs, rhs,
v, lhs1, rhs1, swapped, seq_cst,
processing_template_decl != 0);
if (stmt == error_mark_node)
return;
}
if (processing_template_decl)
{
if (code == OMP_ATOMIC_READ)
{
stmt = build_min_nt_loc (EXPR_LOCATION (orig_lhs),
OMP_ATOMIC_READ, orig_lhs);
OMP_ATOMIC_SEQ_CST (stmt) = seq_cst;
stmt = build2 (MODIFY_EXPR, void_type_node, orig_v, stmt);
}
else
{
if (opcode == NOP_EXPR)
stmt = build2 (MODIFY_EXPR, void_type_node, orig_lhs, orig_rhs);
else 
stmt = build2 (opcode, void_type_node, orig_lhs, orig_rhs);
if (orig_rhs1)
stmt = build_min_nt_loc (EXPR_LOCATION (orig_rhs1),
COMPOUND_EXPR, orig_rhs1, stmt);
if (code != OMP_ATOMIC)
{
stmt = build_min_nt_loc (EXPR_LOCATION (orig_lhs1),
code, orig_lhs1, stmt);
OMP_ATOMIC_SEQ_CST (stmt) = seq_cst;
stmt = build2 (MODIFY_EXPR, void_type_node, orig_v, stmt);
}
}
stmt = build2 (OMP_ATOMIC, void_type_node, integer_zero_node, stmt);
OMP_ATOMIC_SEQ_CST (stmt) = seq_cst;
}
finish_expr_stmt (stmt);
}
void
finish_omp_barrier (void)
{
tree fn = builtin_decl_explicit (BUILT_IN_GOMP_BARRIER);
vec<tree, va_gc> *vec = make_tree_vector ();
tree stmt = finish_call_expr (fn, &vec, false, false, tf_warning_or_error);
release_tree_vector (vec);
finish_expr_stmt (stmt);
}
void
finish_omp_flush (void)
{
tree fn = builtin_decl_explicit (BUILT_IN_SYNC_SYNCHRONIZE);
vec<tree, va_gc> *vec = make_tree_vector ();
tree stmt = finish_call_expr (fn, &vec, false, false, tf_warning_or_error);
release_tree_vector (vec);
finish_expr_stmt (stmt);
}
void
finish_omp_taskwait (void)
{
tree fn = builtin_decl_explicit (BUILT_IN_GOMP_TASKWAIT);
vec<tree, va_gc> *vec = make_tree_vector ();
tree stmt = finish_call_expr (fn, &vec, false, false, tf_warning_or_error);
release_tree_vector (vec);
finish_expr_stmt (stmt);
}
void
finish_omp_taskyield (void)
{
tree fn = builtin_decl_explicit (BUILT_IN_GOMP_TASKYIELD);
vec<tree, va_gc> *vec = make_tree_vector ();
tree stmt = finish_call_expr (fn, &vec, false, false, tf_warning_or_error);
release_tree_vector (vec);
finish_expr_stmt (stmt);
}
void
finish_omp_cancel (tree clauses)
{
tree fn = builtin_decl_explicit (BUILT_IN_GOMP_CANCEL);
int mask = 0;
if (omp_find_clause (clauses, OMP_CLAUSE_PARALLEL))
mask = 1;
else if (omp_find_clause (clauses, OMP_CLAUSE_FOR))
mask = 2;
else if (omp_find_clause (clauses, OMP_CLAUSE_SECTIONS))
mask = 4;
else if (omp_find_clause (clauses, OMP_CLAUSE_TASKGROUP))
mask = 8;
else
{
error ("%<#pragma omp cancel%> must specify one of "
"%<parallel%>, %<for%>, %<sections%> or %<taskgroup%> clauses");
return;
}
vec<tree, va_gc> *vec = make_tree_vector ();
tree ifc = omp_find_clause (clauses, OMP_CLAUSE_IF);
if (ifc != NULL_TREE)
{
if (!processing_template_decl)
ifc = maybe_convert_cond (OMP_CLAUSE_IF_EXPR (ifc));
else
ifc = build_x_binary_op (OMP_CLAUSE_LOCATION (ifc), NE_EXPR,
OMP_CLAUSE_IF_EXPR (ifc), ERROR_MARK,
integer_zero_node, ERROR_MARK,
NULL, tf_warning_or_error);
}
else
ifc = boolean_true_node;
vec->quick_push (build_int_cst (integer_type_node, mask));
vec->quick_push (ifc);
tree stmt = finish_call_expr (fn, &vec, false, false, tf_warning_or_error);
release_tree_vector (vec);
finish_expr_stmt (stmt);
}
void
finish_omp_cancellation_point (tree clauses)
{
tree fn = builtin_decl_explicit (BUILT_IN_GOMP_CANCELLATION_POINT);
int mask = 0;
if (omp_find_clause (clauses, OMP_CLAUSE_PARALLEL))
mask = 1;
else if (omp_find_clause (clauses, OMP_CLAUSE_FOR))
mask = 2;
else if (omp_find_clause (clauses, OMP_CLAUSE_SECTIONS))
mask = 4;
else if (omp_find_clause (clauses, OMP_CLAUSE_TASKGROUP))
mask = 8;
else
{
error ("%<#pragma omp cancellation point%> must specify one of "
"%<parallel%>, %<for%>, %<sections%> or %<taskgroup%> clauses");
return;
}
vec<tree, va_gc> *vec
= make_tree_vector_single (build_int_cst (integer_type_node, mask));
tree stmt = finish_call_expr (fn, &vec, false, false, tf_warning_or_error);
release_tree_vector (vec);
finish_expr_stmt (stmt);
}

tree
begin_transaction_stmt (location_t loc, tree *pcompound, int flags)
{
tree r;
if (pcompound)
*pcompound = begin_compound_stmt (0);
r = build_stmt (loc, TRANSACTION_EXPR, NULL_TREE);
if (flag_tm)
add_stmt (r);
else
error_at (loc, ((flags & TM_STMT_ATTR_RELAXED) != 0
? G_("%<__transaction_relaxed%> without "
"transactional memory support enabled")
: G_("%<__transaction_atomic%> without "
"transactional memory support enabled")));
TRANSACTION_EXPR_BODY (r) = push_stmt_list ();
TREE_SIDE_EFFECTS (r) = 1;
return r;
}
void
finish_transaction_stmt (tree stmt, tree compound_stmt, int flags, tree noex)
{
TRANSACTION_EXPR_BODY (stmt) = pop_stmt_list (TRANSACTION_EXPR_BODY (stmt));
TRANSACTION_EXPR_OUTER (stmt) = (flags & TM_STMT_ATTR_OUTER) != 0;
TRANSACTION_EXPR_RELAXED (stmt) = (flags & TM_STMT_ATTR_RELAXED) != 0;
TRANSACTION_EXPR_IS_STMT (stmt) = 1;
gcc_assert (!(noex && compound_stmt));
if (noex)
{
tree body = build_must_not_throw_expr (TRANSACTION_EXPR_BODY (stmt),
noex);
protected_set_expr_location
(body, EXPR_LOCATION (TRANSACTION_EXPR_BODY (stmt)));
TREE_SIDE_EFFECTS (body) = 1;
TRANSACTION_EXPR_BODY (stmt) = body;
}
if (compound_stmt)
finish_compound_stmt (compound_stmt);
}
tree
build_transaction_expr (location_t loc, tree expr, int flags, tree noex)
{
tree ret;
if (noex)
{
expr = build_must_not_throw_expr (expr, noex);
protected_set_expr_location (expr, loc);
TREE_SIDE_EFFECTS (expr) = 1;
}
ret = build1 (TRANSACTION_EXPR, TREE_TYPE (expr), expr);
if (flags & TM_STMT_ATTR_RELAXED)
TRANSACTION_EXPR_RELAXED (ret) = 1;
TREE_SIDE_EFFECTS (ret) = 1;
SET_EXPR_LOCATION (ret, loc);
return ret;
}

void
init_cp_semantics (void)
{
}

void 
finish_static_assert (tree condition, tree message, location_t location, 
bool member_p)
{
tsubst_flags_t complain = tf_warning_or_error;
if (message == NULL_TREE
|| message == error_mark_node
|| condition == NULL_TREE
|| condition == error_mark_node)
return;
if (check_for_bare_parameter_packs (condition))
condition = error_mark_node;
if (instantiation_dependent_expression_p (condition))
{
tree assertion;
assertion = make_node (STATIC_ASSERT);
STATIC_ASSERT_CONDITION (assertion) = condition;
STATIC_ASSERT_MESSAGE (assertion) = message;
STATIC_ASSERT_SOURCE_LOCATION (assertion) = location;
if (member_p)
maybe_add_class_template_decl_list (current_class_type, 
assertion,
0);
else
add_stmt (assertion);
return;
}
condition = perform_implicit_conversion_flags (boolean_type_node, condition,
complain, LOOKUP_NORMAL);
condition = fold_non_dependent_expr (condition);
if (TREE_CODE (condition) == INTEGER_CST && !integer_zerop (condition))
;
else 
{
location_t saved_loc = input_location;
input_location = location;
if (TREE_CODE (condition) == INTEGER_CST 
&& integer_zerop (condition))
{
int sz = TREE_INT_CST_LOW (TYPE_SIZE_UNIT
(TREE_TYPE (TREE_TYPE (message))));
int len = TREE_STRING_LENGTH (message) / sz - 1;
if (len == 0)
error ("static assertion failed");
else
error ("static assertion failed: %s",
TREE_STRING_POINTER (message));
}
else if (condition && condition != error_mark_node)
{
error ("non-constant condition for static assertion");
if (require_rvalue_constant_expression (condition))
cxx_constant_value (condition);
}
input_location = saved_loc;
}
}

tree
finish_decltype_type (tree expr, bool id_expression_or_member_access_p,
tsubst_flags_t complain)
{
tree type = NULL_TREE;
if (!expr || error_operand_p (expr))
return error_mark_node;
if (TYPE_P (expr)
|| TREE_CODE (expr) == TYPE_DECL
|| (TREE_CODE (expr) == BIT_NOT_EXPR
&& TYPE_P (TREE_OPERAND (expr, 0))))
{
if (complain & tf_error)
error ("argument to decltype must be an expression");
return error_mark_node;
}
if (instantiation_dependent_uneval_expression_p (expr))
{
type = cxx_make_type (DECLTYPE_TYPE);
DECLTYPE_TYPE_EXPR (type) = expr;
DECLTYPE_TYPE_ID_EXPR_OR_MEMBER_ACCESS_P (type)
= id_expression_or_member_access_p;
SET_TYPE_STRUCTURAL_EQUALITY (type);
return type;
}
expr = resolve_nondeduced_context (expr, complain);
if (invalid_nonstatic_memfn_p (input_location, expr, complain))
return error_mark_node;
if (type_unknown_p (expr))
{
if (complain & tf_error)
error ("decltype cannot resolve address of overloaded function");
return error_mark_node;
}
if (VAR_P (expr)
&& VAR_HAD_UNKNOWN_BOUND (expr)
&& DECL_TEMPLATE_INSTANTIATION (expr))
instantiate_decl (expr, true, false);
if (id_expression_or_member_access_p)
{
if (identifier_p (expr))
expr = lookup_name (expr);
if (INDIRECT_REF_P (expr))
expr = TREE_OPERAND (expr, 0);
if (TREE_CODE (expr) == OFFSET_REF
|| TREE_CODE (expr) == MEMBER_REF
|| TREE_CODE (expr) == SCOPE_REF)
expr = TREE_OPERAND (expr, 1);
if (BASELINK_P (expr))
expr = BASELINK_FUNCTIONS (expr);
if (DECL_DECOMPOSITION_P (expr))
{
if (DECL_HAS_VALUE_EXPR_P (expr))
return unlowered_expr_type (expr);
else
return lookup_decomp_type (expr);
}
switch (TREE_CODE (expr))
{
case FIELD_DECL:
if (DECL_BIT_FIELD_TYPE (expr))
{
type = DECL_BIT_FIELD_TYPE (expr);
break;
}
gcc_fallthrough ();
case FUNCTION_DECL:
case VAR_DECL:
case CONST_DECL:
case PARM_DECL:
case RESULT_DECL:
case TEMPLATE_PARM_INDEX:
expr = mark_type_use (expr);
type = TREE_TYPE (expr);
break;
case ERROR_MARK:
type = error_mark_node;
break;
case COMPONENT_REF:
case COMPOUND_EXPR:
mark_type_use (expr);
type = is_bitfield_expr_with_lowered_type (expr);
if (!type)
type = TREE_TYPE (TREE_OPERAND (expr, 1));
break;
case BIT_FIELD_REF:
gcc_unreachable ();
case INTEGER_CST:
case PTRMEM_CST:
type = TREE_TYPE (expr);
break;
default:
type = TREE_TYPE (expr);
break;
}
}
else
{
if (outer_automatic_var_p (expr)
&& current_function_decl
&& LAMBDA_FUNCTION_P (current_function_decl))
type = capture_decltype (expr);
else if (error_operand_p (expr))
type = error_mark_node;
else if (expr == current_class_ptr)
type = TYPE_MAIN_VARIANT (TREE_TYPE (expr));
else
{
cp_lvalue_kind clk = lvalue_kind (expr);
type = unlowered_expr_type (expr);
gcc_assert (TREE_CODE (type) != REFERENCE_TYPE);
if (VECTOR_TYPE_P (type))
type = strip_typedefs (type);
if (clk != clk_none && !(clk & clk_class))
type = cp_build_reference_type (type, (clk & clk_rvalueref));
}
}
return type;
}
static bool
classtype_has_nothrow_assign_or_copy_p (tree type, bool assign_p)
{
tree fns = NULL_TREE;
if (assign_p || TYPE_HAS_COPY_CTOR (type))
fns = get_class_binding (type, assign_p ? assign_op_identifier
: ctor_identifier);
bool saw_copy = false;
for (ovl_iterator iter (fns); iter; ++iter)
{
tree fn = *iter;
if (copy_fn_p (fn) > 0)
{
saw_copy = true;
maybe_instantiate_noexcept (fn);
if (!TYPE_NOTHROW_P (TREE_TYPE (fn)))
return false;
}
}
return saw_copy;
}
static bool
trait_expr_value (cp_trait_kind kind, tree type1, tree type2)
{
enum tree_code type_code1;
tree t;
type_code1 = TREE_CODE (type1);
switch (kind)
{
case CPTK_HAS_NOTHROW_ASSIGN:
type1 = strip_array_types (type1);
return (!CP_TYPE_CONST_P (type1) && type_code1 != REFERENCE_TYPE
&& (trait_expr_value (CPTK_HAS_TRIVIAL_ASSIGN, type1, type2)
|| (CLASS_TYPE_P (type1)
&& classtype_has_nothrow_assign_or_copy_p (type1,
true))));
case CPTK_HAS_TRIVIAL_ASSIGN:
type1 = strip_array_types (type1);
return (!CP_TYPE_CONST_P (type1) && type_code1 != REFERENCE_TYPE
&& (trivial_type_p (type1)
|| (CLASS_TYPE_P (type1)
&& TYPE_HAS_TRIVIAL_COPY_ASSIGN (type1))));
case CPTK_HAS_NOTHROW_CONSTRUCTOR:
type1 = strip_array_types (type1);
return (trait_expr_value (CPTK_HAS_TRIVIAL_CONSTRUCTOR, type1, type2) 
|| (CLASS_TYPE_P (type1)
&& (t = locate_ctor (type1))
&& (maybe_instantiate_noexcept (t),
TYPE_NOTHROW_P (TREE_TYPE (t)))));
case CPTK_HAS_TRIVIAL_CONSTRUCTOR:
type1 = strip_array_types (type1);
return (trivial_type_p (type1)
|| (CLASS_TYPE_P (type1) && TYPE_HAS_TRIVIAL_DFLT (type1)));
case CPTK_HAS_NOTHROW_COPY:
type1 = strip_array_types (type1);
return (trait_expr_value (CPTK_HAS_TRIVIAL_COPY, type1, type2)
|| (CLASS_TYPE_P (type1)
&& classtype_has_nothrow_assign_or_copy_p (type1, false)));
case CPTK_HAS_TRIVIAL_COPY:
type1 = strip_array_types (type1);
return (trivial_type_p (type1) || type_code1 == REFERENCE_TYPE
|| (CLASS_TYPE_P (type1) && TYPE_HAS_TRIVIAL_COPY_CTOR (type1)));
case CPTK_HAS_TRIVIAL_DESTRUCTOR:
type1 = strip_array_types (type1);
return (trivial_type_p (type1) || type_code1 == REFERENCE_TYPE
|| (CLASS_TYPE_P (type1)
&& TYPE_HAS_TRIVIAL_DESTRUCTOR (type1)));
case CPTK_HAS_VIRTUAL_DESTRUCTOR:
return type_has_virtual_destructor (type1);
case CPTK_HAS_UNIQUE_OBJ_REPRESENTATIONS:
return type_has_unique_obj_representations (type1);
case CPTK_IS_ABSTRACT:
return ABSTRACT_CLASS_TYPE_P (type1);
case CPTK_IS_AGGREGATE:
return CP_AGGREGATE_TYPE_P (type1);
case CPTK_IS_BASE_OF:
return (NON_UNION_CLASS_TYPE_P (type1) && NON_UNION_CLASS_TYPE_P (type2)
&& (same_type_ignoring_top_level_qualifiers_p (type1, type2)
|| DERIVED_FROM_P (type1, type2)));
case CPTK_IS_CLASS:
return NON_UNION_CLASS_TYPE_P (type1);
case CPTK_IS_EMPTY:
return NON_UNION_CLASS_TYPE_P (type1) && CLASSTYPE_EMPTY_P (type1);
case CPTK_IS_ENUM:
return type_code1 == ENUMERAL_TYPE;
case CPTK_IS_FINAL:
return CLASS_TYPE_P (type1) && CLASSTYPE_FINAL (type1);
case CPTK_IS_LITERAL_TYPE:
return literal_type_p (type1);
case CPTK_IS_POD:
return pod_type_p (type1);
case CPTK_IS_POLYMORPHIC:
return CLASS_TYPE_P (type1) && TYPE_POLYMORPHIC_P (type1);
case CPTK_IS_SAME_AS:
return same_type_p (type1, type2);
case CPTK_IS_STD_LAYOUT:
return std_layout_type_p (type1);
case CPTK_IS_TRIVIAL:
return trivial_type_p (type1);
case CPTK_IS_TRIVIALLY_ASSIGNABLE:
return is_trivially_xible (MODIFY_EXPR, type1, type2);
case CPTK_IS_TRIVIALLY_CONSTRUCTIBLE:
return is_trivially_xible (INIT_EXPR, type1, type2);
case CPTK_IS_TRIVIALLY_COPYABLE:
return trivially_copyable_p (type1);
case CPTK_IS_UNION:
return type_code1 == UNION_TYPE;
case CPTK_IS_ASSIGNABLE:
return is_xible (MODIFY_EXPR, type1, type2);
case CPTK_IS_CONSTRUCTIBLE:
return is_xible (INIT_EXPR, type1, type2);
default:
gcc_unreachable ();
return false;
}
}
static bool
check_trait_type (tree type)
{
if (type == NULL_TREE)
return true;
if (TREE_CODE (type) == TREE_LIST)
return (check_trait_type (TREE_VALUE (type))
&& check_trait_type (TREE_CHAIN (type)));
if (TREE_CODE (type) == ARRAY_TYPE && !TYPE_DOMAIN (type)
&& COMPLETE_TYPE_P (TREE_TYPE (type)))
return true;
if (VOID_TYPE_P (type))
return true;
return !!complete_type_or_else (strip_array_types (type), NULL_TREE);
}
tree
finish_trait_expr (cp_trait_kind kind, tree type1, tree type2)
{
if (type1 == error_mark_node
|| type2 == error_mark_node)
return error_mark_node;
if (processing_template_decl)
{
tree trait_expr = make_node (TRAIT_EXPR);
TREE_TYPE (trait_expr) = boolean_type_node;
TRAIT_EXPR_TYPE1 (trait_expr) = type1;
TRAIT_EXPR_TYPE2 (trait_expr) = type2;
TRAIT_EXPR_KIND (trait_expr) = kind;
return trait_expr;
}
switch (kind)
{
case CPTK_HAS_NOTHROW_ASSIGN:
case CPTK_HAS_TRIVIAL_ASSIGN:
case CPTK_HAS_NOTHROW_CONSTRUCTOR:
case CPTK_HAS_TRIVIAL_CONSTRUCTOR:
case CPTK_HAS_NOTHROW_COPY:
case CPTK_HAS_TRIVIAL_COPY:
case CPTK_HAS_TRIVIAL_DESTRUCTOR:
case CPTK_HAS_UNIQUE_OBJ_REPRESENTATIONS:
case CPTK_HAS_VIRTUAL_DESTRUCTOR:
case CPTK_IS_ABSTRACT:
case CPTK_IS_AGGREGATE:
case CPTK_IS_EMPTY:
case CPTK_IS_FINAL:
case CPTK_IS_LITERAL_TYPE:
case CPTK_IS_POD:
case CPTK_IS_POLYMORPHIC:
case CPTK_IS_STD_LAYOUT:
case CPTK_IS_TRIVIAL:
case CPTK_IS_TRIVIALLY_COPYABLE:
if (!check_trait_type (type1))
return error_mark_node;
break;
case CPTK_IS_ASSIGNABLE:
case CPTK_IS_CONSTRUCTIBLE:
break;
case CPTK_IS_TRIVIALLY_ASSIGNABLE:
case CPTK_IS_TRIVIALLY_CONSTRUCTIBLE:
if (!check_trait_type (type1)
|| !check_trait_type (type2))
return error_mark_node;
break;
case CPTK_IS_BASE_OF:
if (NON_UNION_CLASS_TYPE_P (type1) && NON_UNION_CLASS_TYPE_P (type2)
&& !same_type_ignoring_top_level_qualifiers_p (type1, type2)
&& !complete_type_or_else (type2, NULL_TREE))
return error_mark_node;
break;
case CPTK_IS_CLASS:
case CPTK_IS_ENUM:
case CPTK_IS_UNION:
case CPTK_IS_SAME_AS:
break;
default:
gcc_unreachable ();
}
return (trait_expr_value (kind, type1, type2)
? boolean_true_node : boolean_false_node);
}
void
set_float_const_decimal64 (void)
{
}
void
clear_float_const_decimal64 (void)
{
}
bool
float_const_decimal64_p (void)
{
return 0;
}

bool
is_this_parameter (tree t)
{
if (!DECL_P (t) || DECL_NAME (t) != this_identifier)
return false;
gcc_assert (TREE_CODE (t) == PARM_DECL || is_capture_proxy (t)
|| (cp_binding_oracle && TREE_CODE (t) == VAR_DECL));
return true;
}
void
apply_deduced_return_type (tree fco, tree return_type)
{
tree result;
if (return_type == error_mark_node)
return;
if (DECL_CONV_FN_P (fco))
DECL_NAME (fco) = make_conv_op_name (return_type);
TREE_TYPE (fco) = change_return_type (return_type, TREE_TYPE (fco));
result = DECL_RESULT (fco);
if (result == NULL_TREE)
return;
if (TREE_TYPE (result) == return_type)
return;
if (!processing_template_decl && !VOID_TYPE_P (return_type)
&& !complete_type_or_else (return_type, NULL_TREE))
return;
gcc_assert (current_function_decl == fco);
result = build_decl (input_location, RESULT_DECL, NULL_TREE,
TYPE_MAIN_VARIANT (return_type));
DECL_ARTIFICIAL (result) = 1;
DECL_IGNORED_P (result) = 1;
cp_apply_type_quals_to_decl (cp_type_quals (return_type),
result);
DECL_RESULT (fco) = result;
if (!processing_template_decl)
{
bool aggr = aggregate_value_p (result, fco);
#ifdef PCC_STATIC_STRUCT_RETURN
cfun->returns_pcc_struct = aggr;
#endif
cfun->returns_struct = aggr;
}
}
static tree
capture_decltype (tree decl)
{
tree lam = CLASSTYPE_LAMBDA_EXPR (DECL_CONTEXT (current_function_decl));
tree cap = value_member (decl, LAMBDA_EXPR_CAPTURE_LIST (lam));
tree type;
if (cap)
type = TREE_TYPE (TREE_PURPOSE (cap));
else
switch (LAMBDA_EXPR_DEFAULT_CAPTURE_MODE (lam))
{
case CPLD_NONE:
error ("%qD is not captured", decl);
return error_mark_node;
case CPLD_COPY:
type = TREE_TYPE (decl);
if (TREE_CODE (type) == REFERENCE_TYPE
&& TREE_CODE (TREE_TYPE (type)) != FUNCTION_TYPE)
type = TREE_TYPE (type);
break;
case CPLD_REFERENCE:
type = TREE_TYPE (decl);
if (TREE_CODE (type) != REFERENCE_TYPE)
type = build_reference_type (TREE_TYPE (decl));
break;
default:
gcc_unreachable ();
}
if (TREE_CODE (type) != REFERENCE_TYPE)
{
if (!LAMBDA_EXPR_MUTABLE_P (lam))
type = cp_build_qualified_type (type, (cp_type_quals (type)
|TYPE_QUAL_CONST));
type = build_reference_type (type);
}
return type;
}
static tree
finish_unary_fold_expr (tree expr, int op, tree_code dir)
{
if (!uses_parameter_packs (expr))
{
error_at (location_of (expr), "operand of fold expression has no "
"unexpanded parameter packs");
return error_mark_node;
}
tree pack = make_pack_expansion (expr);
tree code = build_int_cstu (integer_type_node, abs (op));
tree fold = build_min_nt_loc (UNKNOWN_LOCATION, dir, code, pack);
FOLD_EXPR_MODIFY_P (fold) = (op < 0);
return fold;
}
tree
finish_left_unary_fold_expr (tree expr, int op)
{
return finish_unary_fold_expr (expr, op, UNARY_LEFT_FOLD_EXPR);
}
tree
finish_right_unary_fold_expr (tree expr, int op)
{
return finish_unary_fold_expr (expr, op, UNARY_RIGHT_FOLD_EXPR);
}
tree
finish_binary_fold_expr (tree pack, tree init, int op, tree_code dir)
{
pack = make_pack_expansion (pack);
tree code = build_int_cstu (integer_type_node, abs (op));
tree fold = build_min_nt_loc (UNKNOWN_LOCATION, dir, code, pack, init);
FOLD_EXPR_MODIFY_P (fold) = (op < 0);
return fold;
}
tree
finish_binary_fold_expr (tree expr1, tree expr2, int op)
{
bool pack1 = uses_parameter_packs (expr1);
bool pack2 = uses_parameter_packs (expr2);
if (pack1 && !pack2)
return finish_binary_fold_expr (expr1, expr2, op, BINARY_RIGHT_FOLD_EXPR);
else if (pack2 && !pack1)
return finish_binary_fold_expr (expr2, expr1, op, BINARY_LEFT_FOLD_EXPR);
else
{
if (pack1)
error ("both arguments in binary fold have unexpanded parameter packs");
else
error ("no unexpanded parameter packs in binary fold");
}
return error_mark_node;
}
tree
finish_builtin_launder (location_t loc, tree arg, tsubst_flags_t complain)
{
tree orig_arg = arg;
if (!type_dependent_expression_p (arg))
arg = decay_conversion (arg, complain);
if (error_operand_p (arg))
return error_mark_node;
if (!type_dependent_expression_p (arg)
&& TREE_CODE (TREE_TYPE (arg)) != POINTER_TYPE)
{
error_at (loc, "non-pointer argument to %<__builtin_launder%>");
return error_mark_node;
}
if (processing_template_decl)
arg = orig_arg;
return build_call_expr_internal_loc (loc, IFN_LAUNDER,
TREE_TYPE (arg), 1, arg);
}
#include "gt-cp-semantics.h"
