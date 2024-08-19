#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "rtl.h"
#include "tree.h"
#include "gimple.h"
#include "tree-pass.h"
#include "ssa.h"
#include "gimple-pretty-print.h"
#include "diagnostic-core.h"
#include "stor-layout.h"
#include "fold-const.h"
#include "calls.h"
#include "intl.h"
#include "gimplify.h"
#include "gimple-iterator.h"
#include "tree-cfg.h"
#include "tree-ssa-loop-ivopts.h"
#include "tree-ssa-loop-niter.h"
#include "tree-ssa-loop.h"
#include "cfgloop.h"
#include "tree-chrec.h"
#include "tree-scalar-evolution.h"
#include "params.h"
#include "tree-dfa.h"
#define MAX_DOMINATORS_TO_WALK 8
struct bounds
{
mpz_t below, up;
};
static void
split_to_var_and_offset (tree expr, tree *var, mpz_t offset)
{
tree type = TREE_TYPE (expr);
tree op0, op1;
bool negate = false;
*var = expr;
mpz_set_ui (offset, 0);
switch (TREE_CODE (expr))
{
case MINUS_EXPR:
negate = true;
case PLUS_EXPR:
case POINTER_PLUS_EXPR:
op0 = TREE_OPERAND (expr, 0);
op1 = TREE_OPERAND (expr, 1);
if (TREE_CODE (op1) != INTEGER_CST)
break;
*var = op0;
wi::to_mpz (wi::to_wide (op1), offset, SIGNED);
if (negate)
mpz_neg (offset, offset);
break;
case INTEGER_CST:
*var = build_int_cst_type (type, 0);
wi::to_mpz (wi::to_wide (expr), offset, TYPE_SIGN (type));
break;
default:
break;
}
}
static void
refine_value_range_using_guard (tree type, tree var,
tree c0, enum tree_code cmp, tree c1,
mpz_t below, mpz_t up)
{
tree varc0, varc1, ctype;
mpz_t offc0, offc1;
mpz_t mint, maxt, minc1, maxc1;
wide_int minv, maxv;
bool no_wrap = nowrap_type_p (type);
bool c0_ok, c1_ok;
signop sgn = TYPE_SIGN (type);
switch (cmp)
{
case LT_EXPR:
case LE_EXPR:
case GT_EXPR:
case GE_EXPR:
STRIP_SIGN_NOPS (c0);
STRIP_SIGN_NOPS (c1);
ctype = TREE_TYPE (c0);
if (!useless_type_conversion_p (ctype, type))
return;
break;
case EQ_EXPR:
return;
case NE_EXPR:
if (TREE_CODE (c1) != INTEGER_CST
|| !INTEGRAL_TYPE_P (type))
return;
ctype = TREE_TYPE (c0);
if (TYPE_PRECISION (ctype) != TYPE_PRECISION (type))
return;
c0 = fold_convert (type, c0);
c1 = fold_convert (type, c1);
if (operand_equal_p (var, c0, 0))
{
mpz_t valc1;
mpz_init (valc1);
wi::to_mpz (wi::to_wide (c1), valc1, TYPE_SIGN (type));
if (mpz_cmp (valc1, below) == 0)
cmp = GT_EXPR;
if (mpz_cmp (valc1, up) == 0)
cmp = LT_EXPR;
mpz_clear (valc1);
}
else
{
wide_int min = wi::min_value (type);
wide_int max = wi::max_value (type);
if (wi::to_wide (c1) == min)
cmp = GT_EXPR;
if (wi::to_wide (c1) == max)
cmp = LT_EXPR;
}
if (cmp == NE_EXPR)
return;
break;
default:
return;
}
mpz_init (offc0);
mpz_init (offc1);
split_to_var_and_offset (expand_simple_operations (c0), &varc0, offc0);
split_to_var_and_offset (expand_simple_operations (c1), &varc1, offc1);
if (operand_equal_p (var, varc1, 0))
{
std::swap (varc0, varc1);
mpz_swap (offc0, offc1);
cmp = swap_tree_comparison (cmp);
}
else if (!operand_equal_p (var, varc0, 0))
{
mpz_clear (offc0);
mpz_clear (offc1);
return;
}
mpz_init (mint);
mpz_init (maxt);
get_type_static_bounds (type, mint, maxt);
mpz_init (minc1);
mpz_init (maxc1);
if (integer_zerop (varc1))
{
wi::to_mpz (0, minc1, TYPE_SIGN (type));
wi::to_mpz (0, maxc1, TYPE_SIGN (type));
}
else if (TREE_CODE (varc1) == SSA_NAME
&& INTEGRAL_TYPE_P (type)
&& get_range_info (varc1, &minv, &maxv) == VR_RANGE)
{
gcc_assert (wi::le_p (minv, maxv, sgn));
wi::to_mpz (minv, minc1, sgn);
wi::to_mpz (maxv, maxc1, sgn);
}
else
{
mpz_set (minc1, mint);
mpz_set (maxc1, maxt);
}
mpz_add (minc1, minc1, offc1);
mpz_add (maxc1, maxc1, offc1);
c1_ok = (no_wrap
|| mpz_sgn (offc1) == 0
|| (mpz_sgn (offc1) < 0 && mpz_cmp (minc1, mint) >= 0)
|| (mpz_sgn (offc1) > 0 && mpz_cmp (maxc1, maxt) <= 0));
if (!c1_ok)
goto end;
if (mpz_cmp (minc1, mint) < 0)
mpz_set (minc1, mint);
if (mpz_cmp (maxc1, maxt) > 0)
mpz_set (maxc1, maxt);
if (cmp == LT_EXPR)
{
cmp = LE_EXPR;
mpz_sub_ui (maxc1, maxc1, 1);
}
if (cmp == GT_EXPR)
{
cmp = GE_EXPR;
mpz_add_ui (minc1, minc1, 1);
}
mpz_sub (minc1, minc1, offc0);
mpz_sub (maxc1, maxc1, offc0);
c0_ok = (no_wrap
|| mpz_sgn (offc0) == 0
|| (cmp == LE_EXPR
&& mpz_sgn (offc0) < 0 && mpz_cmp (maxc1, maxt) <= 0)
|| (cmp == GE_EXPR
&& mpz_sgn (offc0) > 0 && mpz_cmp (minc1, mint) >= 0));
if (!c0_ok)
goto end;
if (cmp == LE_EXPR)
{
if (mpz_cmp (up, maxc1) > 0)
mpz_set (up, maxc1);
}
else
{
if (mpz_cmp (below, minc1) < 0)
mpz_set (below, minc1);
}
end:
mpz_clear (mint);
mpz_clear (maxt);
mpz_clear (minc1);
mpz_clear (maxc1);
mpz_clear (offc0);
mpz_clear (offc1);
}
static void
determine_value_range (struct loop *loop, tree type, tree var, mpz_t off,
mpz_t min, mpz_t max)
{
int cnt = 0;
mpz_t minm, maxm;
basic_block bb;
wide_int minv, maxv;
enum value_range_type rtype = VR_VARYING;
if (integer_zerop (var))
{
mpz_set (min, off);
mpz_set (max, off);
return;
}
get_type_static_bounds (type, min, max);
if (TREE_CODE (var) == SSA_NAME && INTEGRAL_TYPE_P (type))
{
edge e = loop_preheader_edge (loop);
signop sgn = TYPE_SIGN (type);
gphi_iterator gsi;
rtype = get_range_info (var, &minv, &maxv);
for (gsi = gsi_start_phis (loop->header); !gsi_end_p (gsi); gsi_next (&gsi))
{
gphi *phi = gsi.phi ();
wide_int minc, maxc;
if (PHI_ARG_DEF_FROM_EDGE (phi, e) == var
&& (get_range_info (gimple_phi_result (phi), &minc, &maxc)
== VR_RANGE))
{
if (rtype != VR_RANGE)
{
rtype = VR_RANGE;
minv = minc;
maxv = maxc;
}
else
{
minv = wi::max (minv, minc, sgn);
maxv = wi::min (maxv, maxc, sgn);
if (wi::gt_p (minv, maxv, sgn))
{
rtype = get_range_info (var, &minv, &maxv);
break;
}
}
}
}
mpz_init (minm);
mpz_init (maxm);
if (rtype != VR_RANGE)
{
mpz_set (minm, min);
mpz_set (maxm, max);
}
else
{
gcc_assert (wi::le_p (minv, maxv, sgn));
wi::to_mpz (minv, minm, sgn);
wi::to_mpz (maxv, maxm, sgn);
}
for (bb = loop->header;
bb != ENTRY_BLOCK_PTR_FOR_FN (cfun) && cnt < MAX_DOMINATORS_TO_WALK;
bb = get_immediate_dominator (CDI_DOMINATORS, bb))
{
edge e;
tree c0, c1;
gimple *cond;
enum tree_code cmp;
if (!single_pred_p (bb))
continue;
e = single_pred_edge (bb);
if (!(e->flags & (EDGE_TRUE_VALUE | EDGE_FALSE_VALUE)))
continue;
cond = last_stmt (e->src);
c0 = gimple_cond_lhs (cond);
cmp = gimple_cond_code (cond);
c1 = gimple_cond_rhs (cond);
if (e->flags & EDGE_FALSE_VALUE)
cmp = invert_tree_comparison (cmp, false);
refine_value_range_using_guard (type, var, c0, cmp, c1, minm, maxm);
++cnt;
}
mpz_add (minm, minm, off);
mpz_add (maxm, maxm, off);
if (nowrap_type_p (type)
|| mpz_sgn (off) == 0
|| (mpz_sgn (off) < 0 && mpz_cmp (minm, min) >= 0)
|| (mpz_sgn (off) > 0 && mpz_cmp (maxm, max) <= 0))
{
mpz_set (min, minm);
mpz_set (max, maxm);
mpz_clear (minm);
mpz_clear (maxm);
return;
}
mpz_clear (minm);
mpz_clear (maxm);
}
if (!nowrap_type_p (type))
return;
if (mpz_sgn (off) < 0)
mpz_add (max, max, off);
else
mpz_add (min, min, off);
}
static void
bound_difference_of_offsetted_base (tree type, mpz_t x, mpz_t y,
bounds *bnds)
{
int rel = mpz_cmp (x, y);
bool may_wrap = !nowrap_type_p (type);
mpz_t m;
if (rel == 0)
{
mpz_set_ui (bnds->below, 0);
mpz_set_ui (bnds->up, 0);
return;
}
mpz_init (m);
wi::to_mpz (wi::minus_one (TYPE_PRECISION (type)), m, UNSIGNED);
mpz_add_ui (m, m, 1);
mpz_sub (bnds->up, x, y);
mpz_set (bnds->below, bnds->up);
if (may_wrap)
{
if (rel > 0)
mpz_sub (bnds->below, bnds->below, m);
else
mpz_add (bnds->up, bnds->up, m);
}
mpz_clear (m);
}
static void
refine_bounds_using_guard (tree type, tree varx, mpz_t offx,
tree vary, mpz_t offy,
tree c0, enum tree_code cmp, tree c1,
bounds *bnds)
{
tree varc0, varc1, ctype;
mpz_t offc0, offc1, loffx, loffy, bnd;
bool lbound = false;
bool no_wrap = nowrap_type_p (type);
bool x_ok, y_ok;
switch (cmp)
{
case LT_EXPR:
case LE_EXPR:
case GT_EXPR:
case GE_EXPR:
STRIP_SIGN_NOPS (c0);
STRIP_SIGN_NOPS (c1);
ctype = TREE_TYPE (c0);
if (!useless_type_conversion_p (ctype, type))
return;
break;
case EQ_EXPR:
return;
case NE_EXPR:
if (TREE_CODE (c1) != INTEGER_CST
|| !INTEGRAL_TYPE_P (type))
return;
ctype = TREE_TYPE (c0);
if (TYPE_PRECISION (ctype) != TYPE_PRECISION (type))
return;
c0 = fold_convert (type, c0);
c1 = fold_convert (type, c1);
if (TYPE_MIN_VALUE (type)
&& operand_equal_p (c1, TYPE_MIN_VALUE (type), 0))
{
cmp = GT_EXPR;
break;
}
if (TYPE_MAX_VALUE (type)
&& operand_equal_p (c1, TYPE_MAX_VALUE (type), 0))
{
cmp = LT_EXPR;
break;
}
return;
default:
return;
}
mpz_init (offc0);
mpz_init (offc1);
split_to_var_and_offset (expand_simple_operations (c0), &varc0, offc0);
split_to_var_and_offset (expand_simple_operations (c1), &varc1, offc1);
if (operand_equal_p (varx, varc1, 0))
{
std::swap (varc0, varc1);
mpz_swap (offc0, offc1);
cmp = swap_tree_comparison (cmp);
}
if (!operand_equal_p (varx, varc0, 0)
|| !operand_equal_p (vary, varc1, 0))
goto end;
mpz_init_set (loffx, offx);
mpz_init_set (loffy, offy);
if (cmp == GT_EXPR || cmp == GE_EXPR)
{
std::swap (varx, vary);
mpz_swap (offc0, offc1);
mpz_swap (loffx, loffy);
cmp = swap_tree_comparison (cmp);
lbound = true;
}
if (no_wrap)
{
x_ok = true;
y_ok = true;
}
else
{
x_ok = (integer_zerop (varx)
|| mpz_cmp (loffx, offc0) >= 0);
y_ok = (integer_zerop (vary)
|| mpz_cmp (loffy, offc1) <= 0);
}
if (x_ok && y_ok)
{
mpz_init (bnd);
mpz_sub (bnd, loffx, loffy);
mpz_add (bnd, bnd, offc1);
mpz_sub (bnd, bnd, offc0);
if (cmp == LT_EXPR)
mpz_sub_ui (bnd, bnd, 1);
if (lbound)
{
mpz_neg (bnd, bnd);
if (mpz_cmp (bnds->below, bnd) < 0)
mpz_set (bnds->below, bnd);
}
else
{
if (mpz_cmp (bnd, bnds->up) < 0)
mpz_set (bnds->up, bnd);
}
mpz_clear (bnd);
}
mpz_clear (loffx);
mpz_clear (loffy);
end:
mpz_clear (offc0);
mpz_clear (offc1);
}
static void
bound_difference (struct loop *loop, tree x, tree y, bounds *bnds)
{
tree type = TREE_TYPE (x);
tree varx, vary;
mpz_t offx, offy;
mpz_t minx, maxx, miny, maxy;
int cnt = 0;
edge e;
basic_block bb;
tree c0, c1;
gimple *cond;
enum tree_code cmp;
STRIP_SIGN_NOPS (x);
STRIP_SIGN_NOPS (y);
mpz_init (bnds->below);
mpz_init (bnds->up);
mpz_init (offx);
mpz_init (offy);
split_to_var_and_offset (x, &varx, offx);
split_to_var_and_offset (y, &vary, offy);
if (!integer_zerop (varx)
&& operand_equal_p (varx, vary, 0))
{
bound_difference_of_offsetted_base (type, offx, offy, bnds);
}
else
{
mpz_init (minx);
mpz_init (maxx);
mpz_init (miny);
mpz_init (maxy);
determine_value_range (loop, type, varx, offx, minx, maxx);
determine_value_range (loop, type, vary, offy, miny, maxy);
mpz_sub (bnds->below, minx, maxy);
mpz_sub (bnds->up, maxx, miny);
mpz_clear (minx);
mpz_clear (maxx);
mpz_clear (miny);
mpz_clear (maxy);
}
if (integer_zerop (varx) && integer_zerop (vary))
goto end;
for (bb = loop->header;
bb != ENTRY_BLOCK_PTR_FOR_FN (cfun) && cnt < MAX_DOMINATORS_TO_WALK;
bb = get_immediate_dominator (CDI_DOMINATORS, bb))
{
if (!single_pred_p (bb))
continue;
e = single_pred_edge (bb);
if (!(e->flags & (EDGE_TRUE_VALUE | EDGE_FALSE_VALUE)))
continue;
cond = last_stmt (e->src);
c0 = gimple_cond_lhs (cond);
cmp = gimple_cond_code (cond);
c1 = gimple_cond_rhs (cond);
if (e->flags & EDGE_FALSE_VALUE)
cmp = invert_tree_comparison (cmp, false);
refine_bounds_using_guard (type, varx, offx, vary, offy,
c0, cmp, c1, bnds);
++cnt;
}
end:
mpz_clear (offx);
mpz_clear (offy);
}
static void
bounds_add (bounds *bnds, const widest_int &delta, tree type)
{
mpz_t mdelta, max;
mpz_init (mdelta);
wi::to_mpz (delta, mdelta, SIGNED);
mpz_init (max);
wi::to_mpz (wi::minus_one (TYPE_PRECISION (type)), max, UNSIGNED);
mpz_add (bnds->up, bnds->up, mdelta);
mpz_add (bnds->below, bnds->below, mdelta);
if (mpz_cmp (bnds->up, max) > 0)
mpz_set (bnds->up, max);
mpz_neg (max, max);
if (mpz_cmp (bnds->below, max) < 0)
mpz_set (bnds->below, max);
mpz_clear (mdelta);
mpz_clear (max);
}
static void
bounds_negate (bounds *bnds)
{
mpz_t tmp;
mpz_init_set (tmp, bnds->up);
mpz_neg (bnds->up, bnds->below);
mpz_neg (bnds->below, tmp);
mpz_clear (tmp);
}
static tree
inverse (tree x, tree mask)
{
tree type = TREE_TYPE (x);
tree rslt;
unsigned ctr = tree_floor_log2 (mask);
if (TYPE_PRECISION (type) <= HOST_BITS_PER_WIDE_INT)
{
unsigned HOST_WIDE_INT ix;
unsigned HOST_WIDE_INT imask;
unsigned HOST_WIDE_INT irslt = 1;
gcc_assert (cst_and_fits_in_hwi (x));
gcc_assert (cst_and_fits_in_hwi (mask));
ix = int_cst_value (x);
imask = int_cst_value (mask);
for (; ctr; ctr--)
{
irslt *= ix;
ix *= ix;
}
irslt &= imask;
rslt = build_int_cst_type (type, irslt);
}
else
{
rslt = build_int_cst (type, 1);
for (; ctr; ctr--)
{
rslt = int_const_binop (MULT_EXPR, rslt, x);
x = int_const_binop (MULT_EXPR, x, x);
}
rslt = int_const_binop (BIT_AND_EXPR, rslt, mask);
}
return rslt;
}
static void
number_of_iterations_ne_max (mpz_t bnd, bool no_overflow, tree c, tree s,
bounds *bnds, bool exit_must_be_taken)
{
widest_int max;
mpz_t d;
tree type = TREE_TYPE (c);
bool bnds_u_valid = ((no_overflow && exit_must_be_taken)
|| mpz_sgn (bnds->below) >= 0);
if (integer_onep (s)
|| (TREE_CODE (c) == INTEGER_CST
&& TREE_CODE (s) == INTEGER_CST
&& wi::mod_trunc (wi::to_wide (c), wi::to_wide (s),
TYPE_SIGN (type)) == 0)
|| (TYPE_OVERFLOW_UNDEFINED (type)
&& multiple_of_p (type, c, s)))
{
no_overflow = true;
exit_must_be_taken = true;
}
if (!no_overflow)
{
max = wi::mask <widest_int> (TYPE_PRECISION (type)
- wi::ctz (wi::to_wide (s)), false);
wi::to_mpz (max, bnd, UNSIGNED);
return;
}
wi::to_mpz (wi::minus_one (TYPE_PRECISION (type)), bnd, UNSIGNED);
if (exit_must_be_taken)
{
if (TREE_CODE (c) == INTEGER_CST)
wi::to_mpz (wi::to_wide (c), bnd, UNSIGNED);
else if (bnds_u_valid)
mpz_set (bnd, bnds->up);
}
mpz_init (d);
wi::to_mpz (wi::to_wide (s), d, UNSIGNED);
mpz_fdiv_q (bnd, bnd, d);
mpz_clear (d);
}
static bool
number_of_iterations_ne (struct loop *loop, tree type, affine_iv *iv,
tree final, struct tree_niter_desc *niter,
bool exit_must_be_taken, bounds *bnds)
{
tree niter_type = unsigned_type_for (type);
tree s, c, d, bits, assumption, tmp, bound;
mpz_t max;
niter->control = *iv;
niter->bound = final;
niter->cmp = NE_EXPR;
if (tree_int_cst_sign_bit (iv->step))
{
s = fold_convert (niter_type,
fold_build1 (NEGATE_EXPR, type, iv->step));
c = fold_build2 (MINUS_EXPR, niter_type,
fold_convert (niter_type, iv->base),
fold_convert (niter_type, final));
bounds_negate (bnds);
}
else
{
s = fold_convert (niter_type, iv->step);
c = fold_build2 (MINUS_EXPR, niter_type,
fold_convert (niter_type, final),
fold_convert (niter_type, iv->base));
}
mpz_init (max);
number_of_iterations_ne_max (max, iv->no_overflow, c, s, bnds,
exit_must_be_taken);
niter->max = widest_int::from (wi::from_mpz (niter_type, max, false),
TYPE_SIGN (niter_type));
mpz_clear (max);
if (!niter->control.no_overflow
&& (integer_onep (s) || multiple_of_p (type, c, s)))
{
tree t, cond, new_c, relaxed_cond = boolean_false_node;
if (tree_int_cst_sign_bit (iv->step))
{
cond = fold_build2 (GE_EXPR, boolean_type_node, iv->base, final);
if (TREE_CODE (type) == INTEGER_TYPE)
{
t = TYPE_MAX_VALUE (type);
t = fold_build2 (PLUS_EXPR, type, t, iv->step);
t = fold_build2 (GE_EXPR, boolean_type_node, t, iv->base);
if (integer_nonzerop (t))
{
t = fold_build2 (MINUS_EXPR, type, iv->base, iv->step);
new_c = fold_build2 (MINUS_EXPR, niter_type,
fold_convert (niter_type, t),
fold_convert (niter_type, final));
if (multiple_of_p (type, new_c, s))
relaxed_cond = fold_build2 (GT_EXPR, boolean_type_node,
t, final);
}
}
}
else
{
cond = fold_build2 (LE_EXPR, boolean_type_node, iv->base, final);
if (TREE_CODE (type) == INTEGER_TYPE)
{
t = TYPE_MIN_VALUE (type);
t = fold_build2 (PLUS_EXPR, type, t, iv->step);
t = fold_build2 (LE_EXPR, boolean_type_node, t, iv->base);
if (integer_nonzerop (t))
{
t = fold_build2 (MINUS_EXPR, type, iv->base, iv->step);
new_c = fold_build2 (MINUS_EXPR, niter_type,
fold_convert (niter_type, final),
fold_convert (niter_type, t));
if (multiple_of_p (type, new_c, s))
relaxed_cond = fold_build2 (LT_EXPR, boolean_type_node,
t, final);
}
}
}
t = simplify_using_initial_conditions (loop, cond);
if (!t || !integer_onep (t))
t = simplify_using_initial_conditions (loop, relaxed_cond);
if (t && integer_onep (t))
niter->control.no_overflow = true;
}
if (integer_onep (s))
{
niter->niter = c;
return true;
}
if (niter->control.no_overflow && multiple_of_p (type, c, s))
{
niter->niter = fold_build2 (FLOOR_DIV_EXPR, niter_type, c, s);
return true;
}
bits = num_ending_zeros (s);
bound = build_low_bits_mask (niter_type,
(TYPE_PRECISION (niter_type)
- tree_to_uhwi (bits)));
d = fold_binary_to_constant (LSHIFT_EXPR, niter_type,
build_int_cst (niter_type, 1), bits);
s = fold_binary_to_constant (RSHIFT_EXPR, niter_type, s, bits);
if (!exit_must_be_taken)
{
assumption = fold_build2 (FLOOR_MOD_EXPR, niter_type, c, d);
assumption = fold_build2 (EQ_EXPR, boolean_type_node,
assumption, build_int_cst (niter_type, 0));
if (!integer_nonzerop (assumption))
niter->assumptions = fold_build2 (TRUTH_AND_EXPR, boolean_type_node,
niter->assumptions, assumption);
}
c = fold_build2 (EXACT_DIV_EXPR, niter_type, c, d);
tmp = fold_build2 (MULT_EXPR, niter_type, c, inverse (s, bound));
niter->niter = fold_build2 (BIT_AND_EXPR, niter_type, tmp, bound);
return true;
}
static bool
number_of_iterations_lt_to_ne (tree type, affine_iv *iv0, affine_iv *iv1,
struct tree_niter_desc *niter,
tree *delta, tree step,
bool exit_must_be_taken, bounds *bnds)
{
tree niter_type = TREE_TYPE (step);
tree mod = fold_build2 (FLOOR_MOD_EXPR, niter_type, *delta, step);
tree tmod;
mpz_t mmod;
tree assumption = boolean_true_node, bound, noloop;
bool ret = false, fv_comp_no_overflow;
tree type1 = type;
if (POINTER_TYPE_P (type))
type1 = sizetype;
if (TREE_CODE (mod) != INTEGER_CST)
return false;
if (integer_nonzerop (mod))
mod = fold_build2 (MINUS_EXPR, niter_type, step, mod);
tmod = fold_convert (type1, mod);
mpz_init (mmod);
wi::to_mpz (wi::to_wide (mod), mmod, UNSIGNED);
mpz_neg (mmod, mmod);
if (integer_zerop (mod) || POINTER_TYPE_P (type))
fv_comp_no_overflow = true;
else if (!exit_must_be_taken)
fv_comp_no_overflow = false;
else
fv_comp_no_overflow =
(iv0->no_overflow && integer_nonzerop (iv0->step))
|| (iv1->no_overflow && integer_nonzerop (iv1->step));
if (integer_nonzerop (iv0->step))
{
if (!fv_comp_no_overflow)
{
bound = fold_build2 (MINUS_EXPR, type1,
TYPE_MAX_VALUE (type1), tmod);
assumption = fold_build2 (LE_EXPR, boolean_type_node,
iv1->base, bound);
if (integer_zerop (assumption))
goto end;
}
if (mpz_cmp (mmod, bnds->below) < 0)
noloop = boolean_false_node;
else if (POINTER_TYPE_P (type))
noloop = fold_build2 (GT_EXPR, boolean_type_node,
iv0->base,
fold_build_pointer_plus (iv1->base, tmod));
else
noloop = fold_build2 (GT_EXPR, boolean_type_node,
iv0->base,
fold_build2 (PLUS_EXPR, type1,
iv1->base, tmod));
}
else
{
if (!fv_comp_no_overflow)
{
bound = fold_build2 (PLUS_EXPR, type1,
TYPE_MIN_VALUE (type1), tmod);
assumption = fold_build2 (GE_EXPR, boolean_type_node,
iv0->base, bound);
if (integer_zerop (assumption))
goto end;
}
if (mpz_cmp (mmod, bnds->below) < 0)
noloop = boolean_false_node;
else if (POINTER_TYPE_P (type))
noloop = fold_build2 (GT_EXPR, boolean_type_node,
fold_build_pointer_plus (iv0->base,
fold_build1 (NEGATE_EXPR,
type1, tmod)),
iv1->base);
else
noloop = fold_build2 (GT_EXPR, boolean_type_node,
fold_build2 (MINUS_EXPR, type1,
iv0->base, tmod),
iv1->base);
}
if (!integer_nonzerop (assumption))
niter->assumptions = fold_build2 (TRUTH_AND_EXPR, boolean_type_node,
niter->assumptions,
assumption);
if (!integer_zerop (noloop))
niter->may_be_zero = fold_build2 (TRUTH_OR_EXPR, boolean_type_node,
niter->may_be_zero,
noloop);
bounds_add (bnds, wi::to_widest (mod), type);
*delta = fold_build2 (PLUS_EXPR, niter_type, *delta, mod);
ret = true;
end:
mpz_clear (mmod);
return ret;
}
static bool
assert_no_overflow_lt (tree type, affine_iv *iv0, affine_iv *iv1,
struct tree_niter_desc *niter, tree step)
{
tree bound, d, assumption, diff;
tree niter_type = TREE_TYPE (step);
if (integer_nonzerop (iv0->step))
{
if (iv0->no_overflow)
return true;
if (TREE_CODE (iv0->base) == INTEGER_CST)
{
d = fold_build2 (MINUS_EXPR, niter_type,
fold_convert (niter_type, TYPE_MAX_VALUE (type)),
fold_convert (niter_type, iv0->base));
diff = fold_build2 (FLOOR_MOD_EXPR, niter_type, d, step);
}
else
diff = fold_build2 (MINUS_EXPR, niter_type, step,
build_int_cst (niter_type, 1));
bound = fold_build2 (MINUS_EXPR, type,
TYPE_MAX_VALUE (type), fold_convert (type, diff));
assumption = fold_build2 (LE_EXPR, boolean_type_node,
iv1->base, bound);
}
else
{
if (iv1->no_overflow)
return true;
if (TREE_CODE (iv1->base) == INTEGER_CST)
{
d = fold_build2 (MINUS_EXPR, niter_type,
fold_convert (niter_type, iv1->base),
fold_convert (niter_type, TYPE_MIN_VALUE (type)));
diff = fold_build2 (FLOOR_MOD_EXPR, niter_type, d, step);
}
else
diff = fold_build2 (MINUS_EXPR, niter_type, step,
build_int_cst (niter_type, 1));
bound = fold_build2 (PLUS_EXPR, type,
TYPE_MIN_VALUE (type), fold_convert (type, diff));
assumption = fold_build2 (GE_EXPR, boolean_type_node,
iv0->base, bound);
}
if (integer_zerop (assumption))
return false;
if (!integer_nonzerop (assumption))
niter->assumptions = fold_build2 (TRUTH_AND_EXPR, boolean_type_node,
niter->assumptions, assumption);
iv0->no_overflow = true;
iv1->no_overflow = true;
return true;
}
static void
assert_loop_rolls_lt (tree type, affine_iv *iv0, affine_iv *iv1,
struct tree_niter_desc *niter, bounds *bnds)
{
tree assumption = boolean_true_node, bound, diff;
tree mbz, mbzl, mbzr, type1;
bool rolls_p, no_overflow_p;
widest_int dstep;
mpz_t mstep, max;
if (integer_nonzerop (iv0->step))
dstep = wi::to_widest (iv0->step);
else
{
dstep = wi::sext (wi::to_widest (iv1->step), TYPE_PRECISION (type));
dstep = -dstep;
}
mpz_init (mstep);
wi::to_mpz (dstep, mstep, UNSIGNED);
mpz_neg (mstep, mstep);
mpz_add_ui (mstep, mstep, 1);
rolls_p = mpz_cmp (mstep, bnds->below) <= 0;
mpz_init (max);
wi::to_mpz (wi::minus_one (TYPE_PRECISION (type)), max, UNSIGNED);
mpz_add (max, max, mstep);
no_overflow_p = (mpz_cmp (bnds->up, max) <= 0
|| POINTER_TYPE_P (type));
mpz_clear (mstep);
mpz_clear (max);
if (rolls_p && no_overflow_p)
return;
type1 = type;
if (POINTER_TYPE_P (type))
type1 = sizetype;
if (integer_nonzerop (iv0->step))
{
diff = fold_build2 (MINUS_EXPR, type1,
iv0->step, build_int_cst (type1, 1));
if (!POINTER_TYPE_P (type))
{
bound = fold_build2 (PLUS_EXPR, type1,
TYPE_MIN_VALUE (type), diff);
assumption = fold_build2 (GE_EXPR, boolean_type_node,
iv0->base, bound);
}
mbzl = fold_build2 (MINUS_EXPR, type1,
fold_convert (type1, iv0->base), diff);
mbzr = fold_convert (type1, iv1->base);
}
else
{
diff = fold_build2 (PLUS_EXPR, type1,
iv1->step, build_int_cst (type1, 1));
if (!POINTER_TYPE_P (type))
{
bound = fold_build2 (PLUS_EXPR, type1,
TYPE_MAX_VALUE (type), diff);
assumption = fold_build2 (LE_EXPR, boolean_type_node,
iv1->base, bound);
}
mbzl = fold_convert (type1, iv0->base);
mbzr = fold_build2 (MINUS_EXPR, type1,
fold_convert (type1, iv1->base), diff);
}
if (!integer_nonzerop (assumption))
niter->assumptions = fold_build2 (TRUTH_AND_EXPR, boolean_type_node,
niter->assumptions, assumption);
if (!rolls_p)
{
mbz = fold_build2 (GT_EXPR, boolean_type_node, mbzl, mbzr);
niter->may_be_zero = fold_build2 (TRUTH_OR_EXPR, boolean_type_node,
niter->may_be_zero, mbz);
}
}
static bool
number_of_iterations_lt (struct loop *loop, tree type, affine_iv *iv0,
affine_iv *iv1, struct tree_niter_desc *niter,
bool exit_must_be_taken, bounds *bnds)
{
tree niter_type = unsigned_type_for (type);
tree delta, step, s;
mpz_t mstep, tmp;
if (integer_nonzerop (iv0->step))
{
niter->control = *iv0;
niter->cmp = LT_EXPR;
niter->bound = iv1->base;
}
else
{
niter->control = *iv1;
niter->cmp = GT_EXPR;
niter->bound = iv0->base;
}
delta = fold_build2 (MINUS_EXPR, niter_type,
fold_convert (niter_type, iv1->base),
fold_convert (niter_type, iv0->base));
if ((integer_onep (iv0->step) && integer_zerop (iv1->step))
|| (integer_all_onesp (iv1->step) && integer_zerop (iv0->step)))
{
if (mpz_sgn (bnds->below) < 0)
niter->may_be_zero = fold_build2 (LT_EXPR, boolean_type_node,
iv1->base, iv0->base);
niter->niter = delta;
niter->max = widest_int::from (wi::from_mpz (niter_type, bnds->up, false),
TYPE_SIGN (niter_type));
niter->control.no_overflow = true;
return true;
}
if (integer_nonzerop (iv0->step))
step = fold_convert (niter_type, iv0->step);
else
step = fold_convert (niter_type,
fold_build1 (NEGATE_EXPR, type, iv1->step));
if (number_of_iterations_lt_to_ne (type, iv0, iv1, niter, &delta, step,
exit_must_be_taken, bnds))
{
affine_iv zps;
zps.base = build_int_cst (niter_type, 0);
zps.step = step;
zps.no_overflow = true;
return number_of_iterations_ne (loop, type, &zps,
delta, niter, true, bnds);
}
if (!assert_no_overflow_lt (type, iv0, iv1, niter, step))
return false;
assert_loop_rolls_lt (type, iv0, iv1, niter, bnds);
s = fold_build2 (MINUS_EXPR, niter_type,
step, build_int_cst (niter_type, 1));
delta = fold_build2 (PLUS_EXPR, niter_type, delta, s);
niter->niter = fold_build2 (FLOOR_DIV_EXPR, niter_type, delta, step);
mpz_init (mstep);
mpz_init (tmp);
wi::to_mpz (wi::to_wide (step), mstep, UNSIGNED);
mpz_add (tmp, bnds->up, mstep);
mpz_sub_ui (tmp, tmp, 1);
mpz_fdiv_q (tmp, tmp, mstep);
niter->max = widest_int::from (wi::from_mpz (niter_type, tmp, false),
TYPE_SIGN (niter_type));
mpz_clear (mstep);
mpz_clear (tmp);
return true;
}
static bool
number_of_iterations_le (struct loop *loop, tree type, affine_iv *iv0,
affine_iv *iv1, struct tree_niter_desc *niter,
bool exit_must_be_taken, bounds *bnds)
{
tree assumption;
tree type1 = type;
if (POINTER_TYPE_P (type))
type1 = sizetype;
if (!exit_must_be_taken && !POINTER_TYPE_P (type))
{
if (integer_nonzerop (iv0->step))
assumption = fold_build2 (NE_EXPR, boolean_type_node,
iv1->base, TYPE_MAX_VALUE (type));
else
assumption = fold_build2 (NE_EXPR, boolean_type_node,
iv0->base, TYPE_MIN_VALUE (type));
if (integer_zerop (assumption))
return false;
if (!integer_nonzerop (assumption))
niter->assumptions = fold_build2 (TRUTH_AND_EXPR, boolean_type_node,
niter->assumptions, assumption);
}
if (integer_nonzerop (iv0->step))
{
if (POINTER_TYPE_P (type))
iv1->base = fold_build_pointer_plus_hwi (iv1->base, 1);
else
iv1->base = fold_build2 (PLUS_EXPR, type1, iv1->base,
build_int_cst (type1, 1));
}
else if (POINTER_TYPE_P (type))
iv0->base = fold_build_pointer_plus_hwi (iv0->base, -1);
else
iv0->base = fold_build2 (MINUS_EXPR, type1,
iv0->base, build_int_cst (type1, 1));
bounds_add (bnds, 1, type1);
return number_of_iterations_lt (loop, type, iv0, iv1, niter, exit_must_be_taken,
bnds);
}
static void
dump_affine_iv (FILE *file, affine_iv *iv)
{
if (!integer_zerop (iv->step))
fprintf (file, "[");
print_generic_expr (dump_file, iv->base, TDF_SLIM);
if (!integer_zerop (iv->step))
{
fprintf (file, ", + , ");
print_generic_expr (dump_file, iv->step, TDF_SLIM);
fprintf (file, "]%s", iv->no_overflow ? "(no_overflow)" : "");
}
}
static bool
number_of_iterations_cond (struct loop *loop,
tree type, affine_iv *iv0, enum tree_code code,
affine_iv *iv1, struct tree_niter_desc *niter,
bool only_exit, bool every_iteration)
{
bool exit_must_be_taken = false, ret;
bounds bnds;
if (!every_iteration
&& (!iv0->no_overflow || !iv1->no_overflow
|| code == NE_EXPR || code == EQ_EXPR))
return false;
niter->assumptions = boolean_true_node;
niter->may_be_zero = boolean_false_node;
niter->niter = NULL_TREE;
niter->max = 0;
niter->bound = NULL_TREE;
niter->cmp = ERROR_MARK;
if (code == GE_EXPR || code == GT_EXPR
|| (code == NE_EXPR && integer_zerop (iv0->step)))
{
std::swap (iv0, iv1);
code = swap_tree_comparison (code);
}
if (POINTER_TYPE_P (type))
{
iv0->no_overflow = true;
iv1->no_overflow = true;
}
if (only_exit)
{
if (!integer_zerop (iv0->step) && iv0->no_overflow)
exit_must_be_taken = true;
else if (!integer_zerop (iv1->step) && iv1->no_overflow)
exit_must_be_taken = true;
}
if (!integer_zerop (iv0->step) && !integer_zerop (iv1->step))
{
tree step_type = POINTER_TYPE_P (type) ? sizetype : type;
tree step = fold_binary_to_constant (MINUS_EXPR, step_type,
iv0->step, iv1->step);
if (code != NE_EXPR
&& (TREE_CODE (step) != INTEGER_CST
|| !iv0->no_overflow || !iv1->no_overflow))
return false;
iv0->step = step;
if (!POINTER_TYPE_P (type))
iv0->no_overflow = false;
iv1->step = build_int_cst (step_type, 0);
iv1->no_overflow = true;
}
if (integer_zerop (iv0->step) && integer_zerop (iv1->step))
return false;
if (code != NE_EXPR)
{
if (iv0->step && tree_int_cst_sign_bit (iv0->step))
return false;
if (!integer_zerop (iv1->step) && !tree_int_cst_sign_bit (iv1->step))
return false;
}
tree tem = fold_binary (code, boolean_type_node, iv0->base, iv1->base);
if (tem && integer_zerop (tem))
{
niter->niter = build_int_cst (unsigned_type_for (type), 0);
niter->max = 0;
return true;
}
bound_difference (loop, iv1->base, iv0->base, &bnds);
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file,
"Analyzing # of iterations of loop %d\n", loop->num);
fprintf (dump_file, "  exit condition ");
dump_affine_iv (dump_file, iv0);
fprintf (dump_file, " %s ",
code == NE_EXPR ? "!="
: code == LT_EXPR ? "<"
: "<=");
dump_affine_iv (dump_file, iv1);
fprintf (dump_file, "\n");
fprintf (dump_file, "  bounds on difference of bases: ");
mpz_out_str (dump_file, 10, bnds.below);
fprintf (dump_file, " ... ");
mpz_out_str (dump_file, 10, bnds.up);
fprintf (dump_file, "\n");
}
switch (code)
{
case NE_EXPR:
gcc_assert (integer_zerop (iv1->step));
ret = number_of_iterations_ne (loop, type, iv0, iv1->base, niter,
exit_must_be_taken, &bnds);
break;
case LT_EXPR:
ret = number_of_iterations_lt (loop, type, iv0, iv1, niter,
exit_must_be_taken, &bnds);
break;
case LE_EXPR:
ret = number_of_iterations_le (loop, type, iv0, iv1, niter,
exit_must_be_taken, &bnds);
break;
default:
gcc_unreachable ();
}
mpz_clear (bnds.up);
mpz_clear (bnds.below);
if (dump_file && (dump_flags & TDF_DETAILS))
{
if (ret)
{
fprintf (dump_file, "  result:\n");
if (!integer_nonzerop (niter->assumptions))
{
fprintf (dump_file, "    under assumptions ");
print_generic_expr (dump_file, niter->assumptions, TDF_SLIM);
fprintf (dump_file, "\n");
}
if (!integer_zerop (niter->may_be_zero))
{
fprintf (dump_file, "    zero if ");
print_generic_expr (dump_file, niter->may_be_zero, TDF_SLIM);
fprintf (dump_file, "\n");
}
fprintf (dump_file, "    # of iterations ");
print_generic_expr (dump_file, niter->niter, TDF_SLIM);
fprintf (dump_file, ", bounded by ");
print_decu (niter->max, dump_file);
fprintf (dump_file, "\n");
}
else
fprintf (dump_file, "  failed\n\n");
}
return ret;
}
static tree
simplify_replace_tree (tree expr, tree old, tree new_tree)
{
unsigned i, n;
tree ret = NULL_TREE, e, se;
if (!expr)
return NULL_TREE;
if (CONSTANT_CLASS_P (old))
return expr;
if (expr == old
|| operand_equal_p (expr, old, 0))
return unshare_expr (new_tree);
if (!EXPR_P (expr))
return expr;
n = TREE_OPERAND_LENGTH (expr);
for (i = 0; i < n; i++)
{
e = TREE_OPERAND (expr, i);
se = simplify_replace_tree (e, old, new_tree);
if (e == se)
continue;
if (!ret)
ret = copy_node (expr);
TREE_OPERAND (ret, i) = se;
}
return (ret ? fold (ret) : expr);
}
tree
expand_simple_operations (tree expr, tree stop)
{
unsigned i, n;
tree ret = NULL_TREE, e, ee, e1;
enum tree_code code;
gimple *stmt;
if (expr == NULL_TREE)
return expr;
if (is_gimple_min_invariant (expr))
return expr;
code = TREE_CODE (expr);
if (IS_EXPR_CODE_CLASS (TREE_CODE_CLASS (code)))
{
n = TREE_OPERAND_LENGTH (expr);
for (i = 0; i < n; i++)
{
e = TREE_OPERAND (expr, i);
ee = expand_simple_operations (e, stop);
if (e == ee)
continue;
if (!ret)
ret = copy_node (expr);
TREE_OPERAND (ret, i) = ee;
}
if (!ret)
return expr;
fold_defer_overflow_warnings ();
ret = fold (ret);
fold_undefer_and_ignore_overflow_warnings ();
return ret;
}
if (TREE_CODE (expr) != SSA_NAME || expr == stop)
return expr;
stmt = SSA_NAME_DEF_STMT (expr);
if (gimple_code (stmt) == GIMPLE_PHI)
{
basic_block src, dest;
if (gimple_phi_num_args (stmt) != 1)
return expr;
e = PHI_ARG_DEF (stmt, 0);
dest = gimple_bb (stmt);
src = single_pred (dest);
if (TREE_CODE (e) == SSA_NAME
&& src->loop_father != dest->loop_father)
return expr;
return expand_simple_operations (e, stop);
}
if (gimple_code (stmt) != GIMPLE_ASSIGN)
return expr;
ssa_op_iter iter;
FOR_EACH_SSA_TREE_OPERAND (e, stmt, iter, SSA_OP_USE)
if (SSA_NAME_OCCURS_IN_ABNORMAL_PHI (e))
return expr;
e = gimple_assign_rhs1 (stmt);
code = gimple_assign_rhs_code (stmt);
if (get_gimple_rhs_class (code) == GIMPLE_SINGLE_RHS)
{
if (is_gimple_min_invariant (e))
return e;
if (code == SSA_NAME)
return expand_simple_operations (e, stop);
else if (code == ADDR_EXPR)
{
poly_int64 offset;
tree base = get_addr_base_and_unit_offset (TREE_OPERAND (e, 0),
&offset);
if (base
&& TREE_CODE (base) == MEM_REF)
{
ee = expand_simple_operations (TREE_OPERAND (base, 0), stop);
return fold_build2 (POINTER_PLUS_EXPR, TREE_TYPE (expr), ee,
wide_int_to_tree (sizetype,
mem_ref_offset (base)
+ offset));
}
}
return expr;
}
switch (code)
{
CASE_CONVERT:
ee = expand_simple_operations (e, stop);
return fold_build1 (code, TREE_TYPE (expr), ee);
case PLUS_EXPR:
case MINUS_EXPR:
if (ANY_INTEGRAL_TYPE_P (TREE_TYPE (expr))
&& TYPE_OVERFLOW_TRAPS (TREE_TYPE (expr)))
return expr;
case POINTER_PLUS_EXPR:
e1 = gimple_assign_rhs2 (stmt);
if (!is_gimple_min_invariant (e1))
return expr;
ee = expand_simple_operations (e, stop);
return fold_build2 (code, TREE_TYPE (expr), ee, e1);
default:
return expr;
}
}
static tree
tree_simplify_using_condition_1 (tree cond, tree expr)
{
bool changed;
tree e, e0, e1, e2, notcond;
enum tree_code code = TREE_CODE (expr);
if (code == INTEGER_CST)
return expr;
if (code == TRUTH_OR_EXPR
|| code == TRUTH_AND_EXPR
|| code == COND_EXPR)
{
changed = false;
e0 = tree_simplify_using_condition_1 (cond, TREE_OPERAND (expr, 0));
if (TREE_OPERAND (expr, 0) != e0)
changed = true;
e1 = tree_simplify_using_condition_1 (cond, TREE_OPERAND (expr, 1));
if (TREE_OPERAND (expr, 1) != e1)
changed = true;
if (code == COND_EXPR)
{
e2 = tree_simplify_using_condition_1 (cond, TREE_OPERAND (expr, 2));
if (TREE_OPERAND (expr, 2) != e2)
changed = true;
}
else
e2 = NULL_TREE;
if (changed)
{
if (code == COND_EXPR)
expr = fold_build3 (code, boolean_type_node, e0, e1, e2);
else
expr = fold_build2 (code, boolean_type_node, e0, e1);
}
return expr;
}
if (TREE_CODE (cond) == EQ_EXPR)
{
e0 = TREE_OPERAND (cond, 0);
e1 = TREE_OPERAND (cond, 1);
e = simplify_replace_tree (expr, e0, e1);
if (integer_zerop (e) || integer_nonzerop (e))
return e;
e = simplify_replace_tree (expr, e1, e0);
if (integer_zerop (e) || integer_nonzerop (e))
return e;
}
if (TREE_CODE (expr) == EQ_EXPR)
{
e0 = TREE_OPERAND (expr, 0);
e1 = TREE_OPERAND (expr, 1);
e = simplify_replace_tree (cond, e0, e1);
if (integer_zerop (e))
return e;
e = simplify_replace_tree (cond, e1, e0);
if (integer_zerop (e))
return e;
}
if (TREE_CODE (expr) == NE_EXPR)
{
e0 = TREE_OPERAND (expr, 0);
e1 = TREE_OPERAND (expr, 1);
e = simplify_replace_tree (cond, e0, e1);
if (integer_zerop (e))
return boolean_true_node;
e = simplify_replace_tree (cond, e1, e0);
if (integer_zerop (e))
return boolean_true_node;
}
notcond = invert_truthvalue (cond);
e = fold_binary (TRUTH_OR_EXPR, boolean_type_node, notcond, expr);
if (e && integer_nonzerop (e))
return e;
e = fold_binary (TRUTH_AND_EXPR, boolean_type_node, cond, expr);
if (e && integer_zerop (e))
return e;
return expr;
}
static tree
tree_simplify_using_condition (tree cond, tree expr)
{
cond = expand_simple_operations (cond);
return tree_simplify_using_condition_1 (cond, expr);
}
tree
simplify_using_initial_conditions (struct loop *loop, tree expr)
{
edge e;
basic_block bb;
gimple *stmt;
tree cond, expanded, backup;
int cnt = 0;
if (TREE_CODE (expr) == INTEGER_CST)
return expr;
backup = expanded = expand_simple_operations (expr);
for (bb = loop->header;
bb != ENTRY_BLOCK_PTR_FOR_FN (cfun) && cnt < MAX_DOMINATORS_TO_WALK;
bb = get_immediate_dominator (CDI_DOMINATORS, bb))
{
if (!single_pred_p (bb))
continue;
e = single_pred_edge (bb);
if (!(e->flags & (EDGE_TRUE_VALUE | EDGE_FALSE_VALUE)))
continue;
stmt = last_stmt (e->src);
cond = fold_build2 (gimple_cond_code (stmt),
boolean_type_node,
gimple_cond_lhs (stmt),
gimple_cond_rhs (stmt));
if (e->flags & EDGE_FALSE_VALUE)
cond = invert_truthvalue (cond);
expanded = tree_simplify_using_condition (cond, expanded);
if (expanded
&& (integer_zerop (expanded) || integer_nonzerop (expanded)))
return expanded;
++cnt;
}
return operand_equal_p (backup, expanded, 0) ? expr : expanded;
}
static tree
simplify_using_outer_evolutions (struct loop *loop, tree expr)
{
enum tree_code code = TREE_CODE (expr);
bool changed;
tree e, e0, e1, e2;
if (is_gimple_min_invariant (expr))
return expr;
if (code == TRUTH_OR_EXPR
|| code == TRUTH_AND_EXPR
|| code == COND_EXPR)
{
changed = false;
e0 = simplify_using_outer_evolutions (loop, TREE_OPERAND (expr, 0));
if (TREE_OPERAND (expr, 0) != e0)
changed = true;
e1 = simplify_using_outer_evolutions (loop, TREE_OPERAND (expr, 1));
if (TREE_OPERAND (expr, 1) != e1)
changed = true;
if (code == COND_EXPR)
{
e2 = simplify_using_outer_evolutions (loop, TREE_OPERAND (expr, 2));
if (TREE_OPERAND (expr, 2) != e2)
changed = true;
}
else
e2 = NULL_TREE;
if (changed)
{
if (code == COND_EXPR)
expr = fold_build3 (code, boolean_type_node, e0, e1, e2);
else
expr = fold_build2 (code, boolean_type_node, e0, e1);
}
return expr;
}
e = instantiate_parameters (loop, expr);
if (is_gimple_min_invariant (e))
return e;
return expr;
}
bool
loop_only_exit_p (const struct loop *loop, const_edge exit)
{
basic_block *body;
gimple_stmt_iterator bsi;
unsigned i;
if (exit != single_exit (loop))
return false;
body = get_loop_body (loop);
for (i = 0; i < loop->num_nodes; i++)
{
for (bsi = gsi_start_bb (body[i]); !gsi_end_p (bsi); gsi_next (&bsi))
if (stmt_can_terminate_bb_p (gsi_stmt (bsi)))
{
free (body);
return true;
}
}
free (body);
return true;
}
bool
number_of_iterations_exit_assumptions (struct loop *loop, edge exit,
struct tree_niter_desc *niter,
gcond **at_stmt, bool every_iteration)
{
gimple *last;
gcond *stmt;
tree type;
tree op0, op1;
enum tree_code code;
affine_iv iv0, iv1;
bool safe;
if (loop_constraint_set_p (loop, LOOP_C_INFINITE))
return false;
safe = dominated_by_p (CDI_DOMINATORS, loop->latch, exit->src);
if (every_iteration && !safe)
return false;
niter->assumptions = boolean_false_node;
niter->control.base = NULL_TREE;
niter->control.step = NULL_TREE;
niter->control.no_overflow = false;
last = last_stmt (exit->src);
if (!last)
return false;
stmt = dyn_cast <gcond *> (last);
if (!stmt)
return false;
code = gimple_cond_code (stmt);
if (exit->flags & EDGE_TRUE_VALUE)
code = invert_tree_comparison (code, false);
switch (code)
{
case GT_EXPR:
case GE_EXPR:
case LT_EXPR:
case LE_EXPR:
case NE_EXPR:
break;
default:
return false;
}
op0 = gimple_cond_lhs (stmt);
op1 = gimple_cond_rhs (stmt);
type = TREE_TYPE (op0);
if (TREE_CODE (type) != INTEGER_TYPE
&& !POINTER_TYPE_P (type))
return false;
tree iv0_niters = NULL_TREE;
if (!simple_iv_with_niters (loop, loop_containing_stmt (stmt),
op0, &iv0, safe ? &iv0_niters : NULL, false))
return false;
tree iv1_niters = NULL_TREE;
if (!simple_iv_with_niters (loop, loop_containing_stmt (stmt),
op1, &iv1, safe ? &iv1_niters : NULL, false))
return false;
if (iv0_niters && iv1_niters)
return false;
fold_defer_overflow_warnings ();
iv0.base = expand_simple_operations (iv0.base);
iv1.base = expand_simple_operations (iv1.base);
if (!number_of_iterations_cond (loop, type, &iv0, code, &iv1, niter,
loop_only_exit_p (loop, exit), safe))
{
fold_undefer_and_ignore_overflow_warnings ();
return false;
}
tree iv_niters = iv0_niters ? iv0_niters : iv1_niters;
if (iv_niters)
{
tree assumption = fold_build2 (LE_EXPR, boolean_type_node, niter->niter,
fold_convert (TREE_TYPE (niter->niter),
iv_niters));
if (!integer_nonzerop (assumption))
niter->assumptions = fold_build2 (TRUTH_AND_EXPR, boolean_type_node,
niter->assumptions, assumption);
if (TREE_CODE (iv_niters) == INTEGER_CST
&& niter->max > wi::to_widest (iv_niters))
niter->max = wi::to_widest (iv_niters);
}
if (!integer_zerop (niter->assumptions)
&& loop_constraint_set_p (loop, LOOP_C_FINITE))
niter->assumptions = boolean_true_node;
if (optimize >= 3)
{
niter->assumptions = simplify_using_outer_evolutions (loop,
niter->assumptions);
niter->may_be_zero = simplify_using_outer_evolutions (loop,
niter->may_be_zero);
niter->niter = simplify_using_outer_evolutions (loop, niter->niter);
}
niter->assumptions
= simplify_using_initial_conditions (loop,
niter->assumptions);
niter->may_be_zero
= simplify_using_initial_conditions (loop,
niter->may_be_zero);
fold_undefer_and_ignore_overflow_warnings ();
if (TREE_CODE (niter->niter) == INTEGER_CST)
niter->max = wi::to_widest (niter->niter);
if (at_stmt)
*at_stmt = stmt;
return (!integer_zerop (niter->assumptions));
}
bool
number_of_iterations_exit (struct loop *loop, edge exit,
struct tree_niter_desc *niter,
bool warn, bool every_iteration)
{
gcond *stmt;
if (!number_of_iterations_exit_assumptions (loop, exit, niter,
&stmt, every_iteration))
return false;
if (integer_nonzerop (niter->assumptions))
return true;
if (warn)
dump_printf_loc (MSG_MISSED_OPTIMIZATION, gimple_location_safe (stmt),
"missed loop optimization: niters analysis ends up "
"with assumptions.\n");
return false;
}
tree
find_loop_niter (struct loop *loop, edge *exit)
{
unsigned i;
vec<edge> exits = get_loop_exit_edges (loop);
edge ex;
tree niter = NULL_TREE, aniter;
struct tree_niter_desc desc;
*exit = NULL;
FOR_EACH_VEC_ELT (exits, i, ex)
{
if (!number_of_iterations_exit (loop, ex, &desc, false))
continue;
if (integer_nonzerop (desc.may_be_zero))
{
niter = build_int_cst (unsigned_type_node, 0);
*exit = ex;
break;
}
if (!integer_zerop (desc.may_be_zero))
continue;
aniter = desc.niter;
if (!niter)
{
niter = aniter;
*exit = ex;
continue;
}
if (TREE_CODE (aniter) != INTEGER_CST)
continue;
if (TREE_CODE (niter) != INTEGER_CST)
{
niter = aniter;
*exit = ex;
continue;
}
if (tree_int_cst_lt (aniter, niter))
{
niter = aniter;
*exit = ex;
continue;
}
}
exits.release ();
return niter ? niter : chrec_dont_know;
}
bool
finite_loop_p (struct loop *loop)
{
widest_int nit;
int flags;
flags = flags_from_decl_or_type (current_function_decl);
if ((flags & (ECF_CONST|ECF_PURE)) && !(flags & ECF_LOOPING_CONST_OR_PURE))
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "Found loop %i to be finite: it is within pure or const function.\n",
loop->num);
return true;
}
if (loop->any_upper_bound
|| max_loop_iterations (loop, &nit))
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "Found loop %i to be finite: upper bound found.\n",
loop->num);
return true;
}
return false;
}
#define MAX_ITERATIONS_TO_TRACK \
((unsigned) PARAM_VALUE (PARAM_MAX_ITERATIONS_TO_TRACK))
static gphi *
chain_of_csts_start (struct loop *loop, tree x)
{
gimple *stmt = SSA_NAME_DEF_STMT (x);
tree use;
basic_block bb = gimple_bb (stmt);
enum tree_code code;
if (!bb
|| !flow_bb_inside_loop_p (loop, bb))
return NULL;
if (gimple_code (stmt) == GIMPLE_PHI)
{
if (bb == loop->header)
return as_a <gphi *> (stmt);
return NULL;
}
if (gimple_code (stmt) != GIMPLE_ASSIGN
|| gimple_assign_rhs_class (stmt) == GIMPLE_TERNARY_RHS)
return NULL;
code = gimple_assign_rhs_code (stmt);
if (gimple_references_memory_p (stmt)
|| TREE_CODE_CLASS (code) == tcc_reference
|| (code == ADDR_EXPR
&& !is_gimple_min_invariant (gimple_assign_rhs1 (stmt))))
return NULL;
use = SINGLE_SSA_TREE_OPERAND (stmt, SSA_OP_USE);
if (use == NULL_TREE)
return NULL;
return chain_of_csts_start (loop, use);
}
static gphi *
get_base_for (struct loop *loop, tree x)
{
gphi *phi;
tree init, next;
if (is_gimple_min_invariant (x))
return NULL;
phi = chain_of_csts_start (loop, x);
if (!phi)
return NULL;
init = PHI_ARG_DEF_FROM_EDGE (phi, loop_preheader_edge (loop));
next = PHI_ARG_DEF_FROM_EDGE (phi, loop_latch_edge (loop));
if (!is_gimple_min_invariant (init))
return NULL;
if (TREE_CODE (next) == SSA_NAME
&& chain_of_csts_start (loop, next) != phi)
return NULL;
return phi;
}
static tree
get_val_for (tree x, tree base)
{
gimple *stmt;
gcc_checking_assert (is_gimple_min_invariant (base));
if (!x)
return base;
else if (is_gimple_min_invariant (x))
return x;
stmt = SSA_NAME_DEF_STMT (x);
if (gimple_code (stmt) == GIMPLE_PHI)
return base;
gcc_checking_assert (is_gimple_assign (stmt));
if (gimple_assign_ssa_name_copy_p (stmt))
return get_val_for (gimple_assign_rhs1 (stmt), base);
else if (gimple_assign_rhs_class (stmt) == GIMPLE_UNARY_RHS
&& TREE_CODE (gimple_assign_rhs1 (stmt)) == SSA_NAME)
return fold_build1 (gimple_assign_rhs_code (stmt),
gimple_expr_type (stmt),
get_val_for (gimple_assign_rhs1 (stmt), base));
else if (gimple_assign_rhs_class (stmt) == GIMPLE_BINARY_RHS)
{
tree rhs1 = gimple_assign_rhs1 (stmt);
tree rhs2 = gimple_assign_rhs2 (stmt);
if (TREE_CODE (rhs1) == SSA_NAME)
rhs1 = get_val_for (rhs1, base);
else if (TREE_CODE (rhs2) == SSA_NAME)
rhs2 = get_val_for (rhs2, base);
else
gcc_unreachable ();
return fold_build2 (gimple_assign_rhs_code (stmt),
gimple_expr_type (stmt), rhs1, rhs2);
}
else
gcc_unreachable ();
}
tree
loop_niter_by_eval (struct loop *loop, edge exit)
{
tree acnd;
tree op[2], val[2], next[2], aval[2];
gphi *phi;
gimple *cond;
unsigned i, j;
enum tree_code cmp;
cond = last_stmt (exit->src);
if (!cond || gimple_code (cond) != GIMPLE_COND)
return chrec_dont_know;
cmp = gimple_cond_code (cond);
if (exit->flags & EDGE_TRUE_VALUE)
cmp = invert_tree_comparison (cmp, false);
switch (cmp)
{
case EQ_EXPR:
case NE_EXPR:
case GT_EXPR:
case GE_EXPR:
case LT_EXPR:
case LE_EXPR:
op[0] = gimple_cond_lhs (cond);
op[1] = gimple_cond_rhs (cond);
break;
default:
return chrec_dont_know;
}
for (j = 0; j < 2; j++)
{
if (is_gimple_min_invariant (op[j]))
{
val[j] = op[j];
next[j] = NULL_TREE;
op[j] = NULL_TREE;
}
else
{
phi = get_base_for (loop, op[j]);
if (!phi)
return chrec_dont_know;
val[j] = PHI_ARG_DEF_FROM_EDGE (phi, loop_preheader_edge (loop));
next[j] = PHI_ARG_DEF_FROM_EDGE (phi, loop_latch_edge (loop));
}
}
fold_defer_overflow_warnings ();
for (i = 0; i < MAX_ITERATIONS_TO_TRACK; i++)
{
for (j = 0; j < 2; j++)
aval[j] = get_val_for (op[j], val[j]);
acnd = fold_binary (cmp, boolean_type_node, aval[0], aval[1]);
if (acnd && integer_zerop (acnd))
{
fold_undefer_and_ignore_overflow_warnings ();
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file,
"Proved that loop %d iterates %d times using brute force.\n",
loop->num, i);
return build_int_cst (unsigned_type_node, i);
}
for (j = 0; j < 2; j++)
{
aval[j] = val[j];
val[j] = get_val_for (next[j], val[j]);
if (!is_gimple_min_invariant (val[j]))
{
fold_undefer_and_ignore_overflow_warnings ();
return chrec_dont_know;
}
}
if (val[0] == aval[0] && val[1] == aval[1])
break;
}
fold_undefer_and_ignore_overflow_warnings ();
return chrec_dont_know;
}
tree
find_loop_niter_by_eval (struct loop *loop, edge *exit)
{
unsigned i;
vec<edge> exits = get_loop_exit_edges (loop);
edge ex;
tree niter = NULL_TREE, aniter;
*exit = NULL;
if (!flag_expensive_optimizations
&& exits.length () > 1)
{
exits.release ();
return chrec_dont_know;
}
FOR_EACH_VEC_ELT (exits, i, ex)
{
if (!just_once_each_iteration_p (loop, ex->src))
continue;
aniter = loop_niter_by_eval (loop, ex);
if (chrec_contains_undetermined (aniter))
continue;
if (niter
&& !tree_int_cst_lt (aniter, niter))
continue;
niter = aniter;
*exit = ex;
}
exits.release ();
return niter ? niter : chrec_dont_know;
}
static widest_int derive_constant_upper_bound_ops (tree, tree,
enum tree_code, tree);
static widest_int
derive_constant_upper_bound_assign (gimple *stmt)
{
enum tree_code code = gimple_assign_rhs_code (stmt);
tree op0 = gimple_assign_rhs1 (stmt);
tree op1 = gimple_assign_rhs2 (stmt);
return derive_constant_upper_bound_ops (TREE_TYPE (gimple_assign_lhs (stmt)),
op0, code, op1);
}
static widest_int
derive_constant_upper_bound (tree val)
{
enum tree_code code;
tree op0, op1, op2;
extract_ops_from_tree (val, &code, &op0, &op1, &op2);
return derive_constant_upper_bound_ops (TREE_TYPE (val), op0, code, op1);
}
static widest_int
derive_constant_upper_bound_ops (tree type, tree op0,
enum tree_code code, tree op1)
{
tree subtype, maxt;
widest_int bnd, max, cst;
gimple *stmt;
if (INTEGRAL_TYPE_P (type))
maxt = TYPE_MAX_VALUE (type);
else
maxt = upper_bound_in_type (type, type);
max = wi::to_widest (maxt);
switch (code)
{
case INTEGER_CST:
return wi::to_widest (op0);
CASE_CONVERT:
subtype = TREE_TYPE (op0);
if (!TYPE_UNSIGNED (subtype)
&& TYPE_UNSIGNED (type)
&& !tree_expr_nonnegative_p (op0))
{
return max;
}
bnd = derive_constant_upper_bound (op0);
if (wi::ltu_p (max, bnd))
return max;
return bnd;
case PLUS_EXPR:
case POINTER_PLUS_EXPR:
case MINUS_EXPR:
if (TREE_CODE (op1) != INTEGER_CST
|| !tree_expr_nonnegative_p (op0))
return max;
cst = wi::sext (wi::to_widest (op1), TYPE_PRECISION (type));
if (code != MINUS_EXPR)
cst = -cst;
bnd = derive_constant_upper_bound (op0);
if (wi::neg_p (cst))
{
cst = -cst;
if (wi::neg_p (cst))
return max;
widest_int mmax = max - cst;
if (wi::leu_p (bnd, mmax))
return max;
return bnd + cst;
}
else
{
if (wi::ltu_p (bnd, cst))
return max;
if (TYPE_UNSIGNED (type))
{
tree tem = fold_binary (GE_EXPR, boolean_type_node, op0,
wide_int_to_tree (type, cst));
if (!tem || integer_nonzerop (tem))
return max;
}
bnd -= cst;
}
return bnd;
case FLOOR_DIV_EXPR:
case EXACT_DIV_EXPR:
if (TREE_CODE (op1) != INTEGER_CST
|| tree_int_cst_sign_bit (op1))
return max;
bnd = derive_constant_upper_bound (op0);
return wi::udiv_floor (bnd, wi::to_widest (op1));
case BIT_AND_EXPR:
if (TREE_CODE (op1) != INTEGER_CST
|| tree_int_cst_sign_bit (op1))
return max;
return wi::to_widest (op1);
case SSA_NAME:
stmt = SSA_NAME_DEF_STMT (op0);
if (gimple_code (stmt) != GIMPLE_ASSIGN
|| gimple_assign_lhs (stmt) != op0)
return max;
return derive_constant_upper_bound_assign (stmt);
default:
return max;
}
}
static void
do_warn_aggressive_loop_optimizations (struct loop *loop,
widest_int i_bound, gimple *stmt)
{
if (!loop->nb_iterations
|| TREE_CODE (loop->nb_iterations) != INTEGER_CST
|| !warn_aggressive_loop_optimizations
|| (cfun->curr_properties & PROP_loops) == 0
|| loop->warned_aggressive_loop_optimizations
|| wi::cmpu (i_bound, wi::to_widest (loop->nb_iterations)) >= 0
|| !dominated_by_p (CDI_DOMINATORS, loop->latch, gimple_bb (stmt)))
return;
edge e = single_exit (loop);
if (e == NULL)
return;
gimple *estmt = last_stmt (e->src);
char buf[WIDE_INT_PRINT_BUFFER_SIZE];
print_dec (i_bound, buf, TYPE_UNSIGNED (TREE_TYPE (loop->nb_iterations))
? UNSIGNED : SIGNED);
if (warning_at (gimple_location (stmt), OPT_Waggressive_loop_optimizations,
"iteration %s invokes undefined behavior", buf))
inform (gimple_location (estmt), "within this loop");
loop->warned_aggressive_loop_optimizations = true;
}
static void
record_estimate (struct loop *loop, tree bound, const widest_int &i_bound,
gimple *at_stmt, bool is_exit, bool realistic, bool upper)
{
widest_int delta;
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "Statement %s", is_exit ? "(exit)" : "");
print_gimple_stmt (dump_file, at_stmt, 0, TDF_SLIM);
fprintf (dump_file, " is %sexecuted at most ",
upper ? "" : "probably ");
print_generic_expr (dump_file, bound, TDF_SLIM);
fprintf (dump_file, " (bounded by ");
print_decu (i_bound, dump_file);
fprintf (dump_file, ") + 1 times in loop %d.\n", loop->num);
}
if (TREE_CODE (bound) != INTEGER_CST)
realistic = false;
else
gcc_checking_assert (i_bound == wi::to_widest (bound));
if (upper
&& (is_exit
|| loop->nb_iterations == NULL_TREE
|| TREE_CODE (loop->nb_iterations) != INTEGER_CST))
{
struct nb_iter_bound *elt = ggc_alloc<nb_iter_bound> ();
elt->bound = i_bound;
elt->stmt = at_stmt;
elt->is_exit = is_exit;
elt->next = loop->bounds;
loop->bounds = elt;
}
if (!dominated_by_p (CDI_DOMINATORS, loop->latch, gimple_bb (at_stmt)))
upper = false;
if (is_exit)
delta = 0;
else
delta = 1;
widest_int new_i_bound = i_bound + delta;
if (wi::ltu_p (new_i_bound, delta))
return;
if (upper && !is_exit)
do_warn_aggressive_loop_optimizations (loop, new_i_bound, at_stmt);
record_niter_bound (loop, new_i_bound, realistic, upper);
}
static void
record_control_iv (struct loop *loop, struct tree_niter_desc *niter)
{
struct control_iv *iv;
if (!niter->control.base || !niter->control.step)
return;
if (!integer_onep (niter->assumptions) || !niter->control.no_overflow)
return;
iv = ggc_alloc<control_iv> ();
iv->base = niter->control.base;
iv->step = niter->control.step;
iv->next = loop->control_ivs;
loop->control_ivs = iv;
return;
}
static bool
get_cst_init_from_scev (tree var, wide_int *init, bool is_min)
{
if (TREE_CODE (var) != SSA_NAME)
return false;
gimple *def_stmt = SSA_NAME_DEF_STMT (var);
struct loop *loop = loop_containing_stmt (def_stmt);
if (loop == NULL)
return false;
affine_iv iv;
if (!simple_iv (loop, loop, var, &iv, false))
return false;
if (!iv.no_overflow)
return false;
if (TREE_CODE (iv.base) != INTEGER_CST || TREE_CODE (iv.step) != INTEGER_CST)
return false;
if (is_min == tree_int_cst_sign_bit (iv.step))
return false;
*init = wi::to_wide (iv.base);
return true;
}
static void
record_nonwrapping_iv (struct loop *loop, tree base, tree step, gimple *stmt,
tree low, tree high, bool realistic, bool upper)
{
tree niter_bound, extreme, delta;
tree type = TREE_TYPE (base), unsigned_type;
tree orig_base = base;
if (TREE_CODE (step) != INTEGER_CST || integer_zerop (step))
return;
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "Induction variable (");
print_generic_expr (dump_file, TREE_TYPE (base), TDF_SLIM);
fprintf (dump_file, ") ");
print_generic_expr (dump_file, base, TDF_SLIM);
fprintf (dump_file, " + ");
print_generic_expr (dump_file, step, TDF_SLIM);
fprintf (dump_file, " * iteration does not wrap in statement ");
print_gimple_stmt (dump_file, stmt, 0, TDF_SLIM);
fprintf (dump_file, " in loop %d.\n", loop->num);
}
unsigned_type = unsigned_type_for (type);
base = fold_convert (unsigned_type, base);
step = fold_convert (unsigned_type, step);
if (tree_int_cst_sign_bit (step))
{
wide_int min, max;
extreme = fold_convert (unsigned_type, low);
if (TREE_CODE (orig_base) == SSA_NAME
&& TREE_CODE (high) == INTEGER_CST
&& INTEGRAL_TYPE_P (TREE_TYPE (orig_base))
&& (get_range_info (orig_base, &min, &max) == VR_RANGE
|| get_cst_init_from_scev (orig_base, &max, false))
&& wi::gts_p (wi::to_wide (high), max))
base = wide_int_to_tree (unsigned_type, max);
else if (TREE_CODE (base) != INTEGER_CST
&& dominated_by_p (CDI_DOMINATORS,
loop->latch, gimple_bb (stmt)))
base = fold_convert (unsigned_type, high);
delta = fold_build2 (MINUS_EXPR, unsigned_type, base, extreme);
step = fold_build1 (NEGATE_EXPR, unsigned_type, step);
}
else
{
wide_int min, max;
extreme = fold_convert (unsigned_type, high);
if (TREE_CODE (orig_base) == SSA_NAME
&& TREE_CODE (low) == INTEGER_CST
&& INTEGRAL_TYPE_P (TREE_TYPE (orig_base))
&& (get_range_info (orig_base, &min, &max) == VR_RANGE
|| get_cst_init_from_scev (orig_base, &min, true))
&& wi::gts_p (min, wi::to_wide (low)))
base = wide_int_to_tree (unsigned_type, min);
else if (TREE_CODE (base) != INTEGER_CST
&& dominated_by_p (CDI_DOMINATORS,
loop->latch, gimple_bb (stmt)))
base = fold_convert (unsigned_type, low);
delta = fold_build2 (MINUS_EXPR, unsigned_type, extreme, base);
}
niter_bound = fold_build2 (FLOOR_DIV_EXPR, unsigned_type, delta, step);
widest_int max = derive_constant_upper_bound (niter_bound);
record_estimate (loop, niter_bound, max, stmt, false, realistic, upper);
}
struct ilb_data
{
struct loop *loop;
gimple *stmt;
};
static bool
idx_infer_loop_bounds (tree base, tree *idx, void *dta)
{
struct ilb_data *data = (struct ilb_data *) dta;
tree ev, init, step;
tree low, high, type, next;
bool sign, upper = true, at_end = false;
struct loop *loop = data->loop;
if (TREE_CODE (base) != ARRAY_REF)
return true;
if (array_at_struct_end_p (base))
{
at_end = true;
upper = false;
}
struct loop *dloop = loop_containing_stmt (data->stmt);
if (!dloop)
return true;
ev = analyze_scalar_evolution (dloop, *idx);
ev = instantiate_parameters (loop, ev);
init = initial_condition (ev);
step = evolution_part_in_loop_num (ev, loop->num);
if (!init
|| !step
|| TREE_CODE (step) != INTEGER_CST
|| integer_zerop (step)
|| tree_contains_chrecs (init, NULL)
|| chrec_contains_symbols_defined_in_loop (init, loop->num))
return true;
low = array_ref_low_bound (base);
high = array_ref_up_bound (base);
if (TREE_CODE (low) != INTEGER_CST
|| !high
|| TREE_CODE (high) != INTEGER_CST)
return true;
sign = tree_int_cst_sign_bit (step);
type = TREE_TYPE (step);
if (at_end
&& operand_equal_p (low, high, 0))
return true;
if (!int_fits_type_p (high, type)
|| !int_fits_type_p (low, type))
return true;
low = fold_convert (type, low);
high = fold_convert (type, high);
if (sign)
next = fold_binary (PLUS_EXPR, type, low, step);
else
next = fold_binary (PLUS_EXPR, type, high, step);
if (tree_int_cst_compare (low, next) <= 0
&& tree_int_cst_compare (next, high) <= 0)
return true;
if (!dominated_by_p (CDI_DOMINATORS, loop->latch, gimple_bb (data->stmt))
&& scev_probably_wraps_p (NULL_TREE,
initial_condition_in_loop_num (ev, loop->num),
step, data->stmt, loop, true))
upper = false;
record_nonwrapping_iv (loop, init, step, data->stmt, low, high, false, upper);
return true;
}
static void
infer_loop_bounds_from_ref (struct loop *loop, gimple *stmt, tree ref)
{
struct ilb_data data;
data.loop = loop;
data.stmt = stmt;
for_each_index (&ref, idx_infer_loop_bounds, &data);
}
static void
infer_loop_bounds_from_array (struct loop *loop, gimple *stmt)
{
if (is_gimple_assign (stmt))
{
tree op0 = gimple_assign_lhs (stmt);
tree op1 = gimple_assign_rhs1 (stmt);
if (REFERENCE_CLASS_P (op0))
infer_loop_bounds_from_ref (loop, stmt, op0);
if (REFERENCE_CLASS_P (op1))
infer_loop_bounds_from_ref (loop, stmt, op1);
}
else if (is_gimple_call (stmt))
{
tree arg, lhs;
unsigned i, n = gimple_call_num_args (stmt);
lhs = gimple_call_lhs (stmt);
if (lhs && REFERENCE_CLASS_P (lhs))
infer_loop_bounds_from_ref (loop, stmt, lhs);
for (i = 0; i < n; i++)
{
arg = gimple_call_arg (stmt, i);
if (REFERENCE_CLASS_P (arg))
infer_loop_bounds_from_ref (loop, stmt, arg);
}
}
}
static void
infer_loop_bounds_from_pointer_arith (struct loop *loop, gimple *stmt)
{
tree def, base, step, scev, type, low, high;
tree var, ptr;
if (!is_gimple_assign (stmt)
|| gimple_assign_rhs_code (stmt) != POINTER_PLUS_EXPR)
return;
def = gimple_assign_lhs (stmt);
if (TREE_CODE (def) != SSA_NAME)
return;
type = TREE_TYPE (def);
if (!nowrap_type_p (type))
return;
ptr = gimple_assign_rhs1 (stmt);
if (!expr_invariant_in_loop_p (loop, ptr))
return;
var = gimple_assign_rhs2 (stmt);
if (TYPE_PRECISION (type) != TYPE_PRECISION (TREE_TYPE (var)))
return;
struct loop *uloop = loop_containing_stmt (stmt);
scev = instantiate_parameters (loop, analyze_scalar_evolution (uloop, def));
if (chrec_contains_undetermined (scev))
return;
base = initial_condition_in_loop_num (scev, loop->num);
step = evolution_part_in_loop_num (scev, loop->num);
if (!base || !step
|| TREE_CODE (step) != INTEGER_CST
|| tree_contains_chrecs (base, NULL)
|| chrec_contains_symbols_defined_in_loop (base, loop->num))
return;
low = lower_bound_in_type (type, type);
high = upper_bound_in_type (type, type);
if (flag_delete_null_pointer_checks && int_cst_value (low) == 0)
low = build_int_cstu (TREE_TYPE (low), TYPE_ALIGN_UNIT (TREE_TYPE (type)));
record_nonwrapping_iv (loop, base, step, stmt, low, high, false, true);
}
static void
infer_loop_bounds_from_signedness (struct loop *loop, gimple *stmt)
{
tree def, base, step, scev, type, low, high;
if (gimple_code (stmt) != GIMPLE_ASSIGN)
return;
def = gimple_assign_lhs (stmt);
if (TREE_CODE (def) != SSA_NAME)
return;
type = TREE_TYPE (def);
if (!INTEGRAL_TYPE_P (type)
|| !TYPE_OVERFLOW_UNDEFINED (type))
return;
scev = instantiate_parameters (loop, analyze_scalar_evolution (loop, def));
if (chrec_contains_undetermined (scev))
return;
base = initial_condition_in_loop_num (scev, loop->num);
step = evolution_part_in_loop_num (scev, loop->num);
if (!base || !step
|| TREE_CODE (step) != INTEGER_CST
|| tree_contains_chrecs (base, NULL)
|| chrec_contains_symbols_defined_in_loop (base, loop->num))
return;
low = lower_bound_in_type (type, type);
high = upper_bound_in_type (type, type);
wide_int minv, maxv;
if (get_range_info (def, &minv, &maxv) == VR_RANGE)
{
low = wide_int_to_tree (type, minv);
high = wide_int_to_tree (type, maxv);
}
record_nonwrapping_iv (loop, base, step, stmt, low, high, false, true);
}
static void
infer_loop_bounds_from_undefined (struct loop *loop)
{
unsigned i;
basic_block *bbs;
gimple_stmt_iterator bsi;
basic_block bb;
bool reliable;
bbs = get_loop_body (loop);
for (i = 0; i < loop->num_nodes; i++)
{
bb = bbs[i];
reliable = dominated_by_p (CDI_DOMINATORS, loop->latch, bb);
for (bsi = gsi_start_bb (bb); !gsi_end_p (bsi); gsi_next (&bsi))
{
gimple *stmt = gsi_stmt (bsi);
infer_loop_bounds_from_array (loop, stmt);
if (reliable)
{
infer_loop_bounds_from_signedness (loop, stmt);
infer_loop_bounds_from_pointer_arith (loop, stmt);
}
}
}
free (bbs);
}
static int
wide_int_cmp (const void *p1, const void *p2)
{
const widest_int *d1 = (const widest_int *) p1;
const widest_int *d2 = (const widest_int *) p2;
return wi::cmpu (*d1, *d2);
}
static int
bound_index (vec<widest_int> bounds, const widest_int &bound)
{
unsigned int end = bounds.length ();
unsigned int begin = 0;
while (begin != end)
{
unsigned int middle = (begin + end) / 2;
widest_int index = bounds[middle];
if (index == bound)
return middle;
else if (wi::ltu_p (index, bound))
begin = middle + 1;
else
end = middle;
}
gcc_unreachable ();
}
static void
discover_iteration_bound_by_body_walk (struct loop *loop)
{
struct nb_iter_bound *elt;
auto_vec<widest_int> bounds;
vec<vec<basic_block> > queues = vNULL;
vec<basic_block> queue = vNULL;
ptrdiff_t queue_index;
ptrdiff_t latch_index = 0;
for (elt = loop->bounds; elt; elt = elt->next)
{
widest_int bound = elt->bound;
if (!elt->is_exit)
{
bound += 1;
if (bound == 0)
continue;
}
if (!loop->any_upper_bound
|| wi::ltu_p (bound, loop->nb_iterations_upper_bound))
bounds.safe_push (bound);
}
if (!bounds.exists ())
return;
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, " Trying to walk loop body to reduce the bound.\n");
bounds.qsort (wide_int_cmp);
hash_map<basic_block, ptrdiff_t> bb_bounds;
for (elt = loop->bounds; elt; elt = elt->next)
{
widest_int bound = elt->bound;
if (!elt->is_exit)
{
bound += 1;
if (bound == 0)
continue;
}
if (!loop->any_upper_bound
|| wi::ltu_p (bound, loop->nb_iterations_upper_bound))
{
ptrdiff_t index = bound_index (bounds, bound);
ptrdiff_t *entry = bb_bounds.get (gimple_bb (elt->stmt));
if (!entry)
bb_bounds.put (gimple_bb (elt->stmt), index);
else if ((ptrdiff_t)*entry > index)
*entry = index;
}
}
hash_map<basic_block, ptrdiff_t> block_priority;
latch_index = -1;
queue_index = bounds.length ();
queues.safe_grow_cleared (queue_index + 1);
queue.safe_push (loop->header);
queues[queue_index] = queue;
block_priority.put (loop->header, queue_index);
for (; queue_index >= 0; queue_index--)
{
if (latch_index < queue_index)
{
while (queues[queue_index].length ())
{
basic_block bb;
ptrdiff_t bound_index = queue_index;
edge e;
edge_iterator ei;
queue = queues[queue_index];
bb = queue.pop ();
if (*block_priority.get (bb) > queue_index)
continue;
ptrdiff_t *entry = bb_bounds.get (bb);
if (entry && *entry < bound_index)
bound_index = *entry;
FOR_EACH_EDGE (e, ei, bb->succs)
{
bool insert = false;
if (loop_exit_edge_p (loop, e))
continue;
if (e == loop_latch_edge (loop)
&& latch_index < bound_index)
latch_index = bound_index;
else if (!(entry = block_priority.get (e->dest)))
{
insert = true;
block_priority.put (e->dest, bound_index);
}
else if (*entry < bound_index)
{
insert = true;
*entry = bound_index;
}
if (insert)
queues[bound_index].safe_push (e->dest);
}
}
}
queues[queue_index].release ();
}
gcc_assert (latch_index >= 0);
if ((unsigned)latch_index < bounds.length ())
{
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "Found better loop bound ");
print_decu (bounds[latch_index], dump_file);
fprintf (dump_file, "\n");
}
record_niter_bound (loop, bounds[latch_index], false, true);
}
queues.release ();
}
static void
maybe_lower_iteration_bound (struct loop *loop)
{
hash_set<gimple *> *not_executed_last_iteration = NULL;
struct nb_iter_bound *elt;
bool found_exit = false;
auto_vec<basic_block> queue;
bitmap visited;
for (elt = loop->bounds; elt; elt = elt->next)
{
if (!elt->is_exit
&& wi::ltu_p (elt->bound, loop->nb_iterations_upper_bound))
{
if (!not_executed_last_iteration)
not_executed_last_iteration = new hash_set<gimple *>;
not_executed_last_iteration->add (elt->stmt);
}
}
if (!not_executed_last_iteration)
return;
queue.safe_push (loop->header);
visited = BITMAP_ALLOC (NULL);
bitmap_set_bit (visited, loop->header->index);
found_exit = false;
do
{
basic_block bb = queue.pop ();
gimple_stmt_iterator gsi;
bool stmt_found = false;
for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi); gsi_next (&gsi))
{
gimple *stmt = gsi_stmt (gsi);
if (not_executed_last_iteration->contains (stmt))
{
stmt_found = true;
break;
}
if (gimple_has_side_effects (stmt))
{
found_exit = true;
break;
}
}
if (found_exit)
break;
if (!stmt_found)
{
edge e;
edge_iterator ei;
FOR_EACH_EDGE (e, ei, bb->succs)
{
if (loop_exit_edge_p (loop, e)
|| e == loop_latch_edge (loop))
{
found_exit = true;
break;
}
if (bitmap_set_bit (visited, e->dest->index))
queue.safe_push (e->dest);
}
}
}
while (queue.length () && !found_exit);
if (!found_exit)
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "Reducing loop iteration estimate by 1; "
"undefined statement must be executed at the last iteration.\n");
record_niter_bound (loop, loop->nb_iterations_upper_bound - 1,
false, true);
}
BITMAP_FREE (visited);
delete not_executed_last_iteration;
}
void
estimate_numbers_of_iterations (struct loop *loop)
{
vec<edge> exits;
tree niter, type;
unsigned i;
struct tree_niter_desc niter_desc;
edge ex;
widest_int bound;
edge likely_exit;
if (loop->estimate_state != EST_NOT_COMPUTED)
return;
loop->estimate_state = EST_AVAILABLE;
if (!loop->any_estimate
&& loop->header->count.reliable_p ())
{
gcov_type nit = expected_loop_iterations_unbounded (loop);
bound = gcov_type_to_wide_int (nit);
record_niter_bound (loop, bound, true, false);
}
number_of_latch_executions (loop);
exits = get_loop_exit_edges (loop);
likely_exit = single_likely_exit (loop);
FOR_EACH_VEC_ELT (exits, i, ex)
{
if (!number_of_iterations_exit (loop, ex, &niter_desc, false, false))
continue;
niter = niter_desc.niter;
type = TREE_TYPE (niter);
if (TREE_CODE (niter_desc.may_be_zero) != INTEGER_CST)
niter = build3 (COND_EXPR, type, niter_desc.may_be_zero,
build_int_cst (type, 0),
niter);
record_estimate (loop, niter, niter_desc.max,
last_stmt (ex->src),
true, ex == likely_exit, true);
record_control_iv (loop, &niter_desc);
}
exits.release ();
if (flag_aggressive_loop_optimizations)
infer_loop_bounds_from_undefined (loop);
discover_iteration_bound_by_body_walk (loop);
maybe_lower_iteration_bound (loop);
if (loop->nb_iterations
&& TREE_CODE (loop->nb_iterations) == INTEGER_CST)
{
loop->any_upper_bound = true;
loop->nb_iterations_upper_bound = wi::to_widest (loop->nb_iterations);
}
}
bool
estimated_loop_iterations (struct loop *loop, widest_int *nit)
{
if (scev_initialized_p ())
estimate_numbers_of_iterations (loop);
return (get_estimated_loop_iterations (loop, nit));
}
HOST_WIDE_INT
estimated_loop_iterations_int (struct loop *loop)
{
widest_int nit;
HOST_WIDE_INT hwi_nit;
if (!estimated_loop_iterations (loop, &nit))
return -1;
if (!wi::fits_shwi_p (nit))
return -1;
hwi_nit = nit.to_shwi ();
return hwi_nit < 0 ? -1 : hwi_nit;
}
bool
max_loop_iterations (struct loop *loop, widest_int *nit)
{
if (scev_initialized_p ())
estimate_numbers_of_iterations (loop);
return get_max_loop_iterations (loop, nit);
}
HOST_WIDE_INT
max_loop_iterations_int (struct loop *loop)
{
widest_int nit;
HOST_WIDE_INT hwi_nit;
if (!max_loop_iterations (loop, &nit))
return -1;
if (!wi::fits_shwi_p (nit))
return -1;
hwi_nit = nit.to_shwi ();
return hwi_nit < 0 ? -1 : hwi_nit;
}
bool
likely_max_loop_iterations (struct loop *loop, widest_int *nit)
{
if (scev_initialized_p ())
estimate_numbers_of_iterations (loop);
return get_likely_max_loop_iterations (loop, nit);
}
HOST_WIDE_INT
likely_max_loop_iterations_int (struct loop *loop)
{
widest_int nit;
HOST_WIDE_INT hwi_nit;
if (!likely_max_loop_iterations (loop, &nit))
return -1;
if (!wi::fits_shwi_p (nit))
return -1;
hwi_nit = nit.to_shwi ();
return hwi_nit < 0 ? -1 : hwi_nit;
}
HOST_WIDE_INT
estimated_stmt_executions_int (struct loop *loop)
{
HOST_WIDE_INT nit = estimated_loop_iterations_int (loop);
HOST_WIDE_INT snit;
if (nit == -1)
return -1;
snit = (HOST_WIDE_INT) ((unsigned HOST_WIDE_INT) nit + 1);
return snit < 0 ? -1 : snit;
}
bool
max_stmt_executions (struct loop *loop, widest_int *nit)
{
widest_int nit_minus_one;
if (!max_loop_iterations (loop, nit))
return false;
nit_minus_one = *nit;
*nit += 1;
return wi::gtu_p (*nit, nit_minus_one);
}
bool
likely_max_stmt_executions (struct loop *loop, widest_int *nit)
{
widest_int nit_minus_one;
if (!likely_max_loop_iterations (loop, nit))
return false;
nit_minus_one = *nit;
*nit += 1;
return wi::gtu_p (*nit, nit_minus_one);
}
bool
estimated_stmt_executions (struct loop *loop, widest_int *nit)
{
widest_int nit_minus_one;
if (!estimated_loop_iterations (loop, nit))
return false;
nit_minus_one = *nit;
*nit += 1;
return wi::gtu_p (*nit, nit_minus_one);
}
void
estimate_numbers_of_iterations (function *fn)
{
struct loop *loop;
fold_defer_overflow_warnings ();
FOR_EACH_LOOP_FN (fn, loop, 0)
estimate_numbers_of_iterations (loop);
fold_undefer_and_ignore_overflow_warnings ();
}
bool
stmt_dominates_stmt_p (gimple *s1, gimple *s2)
{
basic_block bb1 = gimple_bb (s1), bb2 = gimple_bb (s2);
if (!bb1
|| s1 == s2)
return true;
if (bb1 == bb2)
{
gimple_stmt_iterator bsi;
if (gimple_code (s2) == GIMPLE_PHI)
return false;
if (gimple_code (s1) == GIMPLE_PHI)
return true;
for (bsi = gsi_start_bb (bb1); gsi_stmt (bsi) != s2; gsi_next (&bsi))
if (gsi_stmt (bsi) == s1)
return true;
return false;
}
return dominated_by_p (CDI_DOMINATORS, bb2, bb1);
}
static bool
n_of_executions_at_most (gimple *stmt,
struct nb_iter_bound *niter_bound,
tree niter)
{
widest_int bound = niter_bound->bound;
tree nit_type = TREE_TYPE (niter), e;
enum tree_code cmp;
gcc_assert (TYPE_UNSIGNED (nit_type));
if (!wi::fits_to_tree_p (bound, nit_type))
return false;
if (niter_bound->is_exit)
{
if (stmt == niter_bound->stmt
|| !stmt_dominates_stmt_p (niter_bound->stmt, stmt))
return false;
cmp = GE_EXPR;
}
else
{
if (!stmt_dominates_stmt_p (niter_bound->stmt, stmt))
{
gimple_stmt_iterator bsi;
if (gimple_bb (stmt) != gimple_bb (niter_bound->stmt)
|| gimple_code (stmt) == GIMPLE_PHI
|| gimple_code (niter_bound->stmt) == GIMPLE_PHI)
return false;
for (bsi = gsi_for_stmt (stmt); gsi_stmt (bsi) != niter_bound->stmt;
gsi_next (&bsi))
if (gimple_has_side_effects (gsi_stmt (bsi)))
return false;
bound += 1;
if (bound == 0
|| !wi::fits_to_tree_p (bound, nit_type))
return false;
}
cmp = GT_EXPR;
}
e = fold_binary (cmp, boolean_type_node,
niter, wide_int_to_tree (nit_type, bound));
return e && integer_nonzerop (e);
}
bool
nowrap_type_p (tree type)
{
if (ANY_INTEGRAL_TYPE_P (type)
&& TYPE_OVERFLOW_UNDEFINED (type))
return true;
if (POINTER_TYPE_P (type))
return true;
return false;
}
static bool
loop_exits_before_overflow (tree base, tree step,
gimple *at_stmt, struct loop *loop)
{
widest_int niter;
struct control_iv *civ;
struct nb_iter_bound *bound;
tree e, delta, step_abs, unsigned_base;
tree type = TREE_TYPE (step);
tree unsigned_type, valid_niter;
fold_defer_overflow_warnings ();
unsigned_type = unsigned_type_for (type);
unsigned_base = fold_convert (unsigned_type, base);
if (tree_int_cst_sign_bit (step))
{
tree extreme = fold_convert (unsigned_type,
lower_bound_in_type (type, type));
delta = fold_build2 (MINUS_EXPR, unsigned_type, unsigned_base, extreme);
step_abs = fold_build1 (NEGATE_EXPR, unsigned_type,
fold_convert (unsigned_type, step));
}
else
{
tree extreme = fold_convert (unsigned_type,
upper_bound_in_type (type, type));
delta = fold_build2 (MINUS_EXPR, unsigned_type, extreme, unsigned_base);
step_abs = fold_convert (unsigned_type, step);
}
valid_niter = fold_build2 (FLOOR_DIV_EXPR, unsigned_type, delta, step_abs);
estimate_numbers_of_iterations (loop);
if (max_loop_iterations (loop, &niter)
&& wi::fits_to_tree_p (niter, TREE_TYPE (valid_niter))
&& (e = fold_binary (GT_EXPR, boolean_type_node, valid_niter,
wide_int_to_tree (TREE_TYPE (valid_niter),
niter))) != NULL
&& integer_nonzerop (e))
{
fold_undefer_and_ignore_overflow_warnings ();
return true;
}
if (at_stmt)
for (bound = loop->bounds; bound; bound = bound->next)
{
if (n_of_executions_at_most (at_stmt, bound, valid_niter))
{
fold_undefer_and_ignore_overflow_warnings ();
return true;
}
}
fold_undefer_and_ignore_overflow_warnings ();
if (TREE_CODE (step) == INTEGER_CST)
{
for (civ = loop->control_ivs; civ; civ = civ->next)
{
enum tree_code code;
tree civ_type = TREE_TYPE (civ->step);
if (TYPE_UNSIGNED (type) != TYPE_UNSIGNED (civ_type)
|| element_precision (type) != element_precision (civ_type))
continue;
if (!operand_equal_p (step, civ->step, 0))
continue;
if (operand_equal_p (base, civ->base, 0))
return true;
tree expanded_base = expand_simple_operations (base);
if (operand_equal_p (expanded_base, civ->base, 0))
return true;
if (POINTER_TYPE_P (TREE_TYPE (base)))
code = POINTER_PLUS_EXPR;
else
code = PLUS_EXPR;
tree stepped = fold_build2 (code, TREE_TYPE (base), base, step);
tree expanded_stepped = fold_build2 (code, TREE_TYPE (base),
expanded_base, step);
if (operand_equal_p (stepped, civ->base, 0)
|| operand_equal_p (expanded_stepped, civ->base, 0))
{
tree extreme;
if (tree_int_cst_sign_bit (step))
{
code = LT_EXPR;
extreme = lower_bound_in_type (type, type);
}
else
{
code = GT_EXPR;
extreme = upper_bound_in_type (type, type);
}
extreme = fold_build2 (MINUS_EXPR, type, extreme, step);
e = fold_build2 (code, boolean_type_node, base, extreme);
e = simplify_using_initial_conditions (loop, e);
if (integer_zerop (e))
return true;
}
}
}
return false;
}
static bool
scev_var_range_cant_overflow (tree var, tree step, struct loop *loop)
{
tree type;
wide_int minv, maxv, diff, step_wi;
enum value_range_type rtype;
if (TREE_CODE (step) != INTEGER_CST || !INTEGRAL_TYPE_P (TREE_TYPE (var)))
return false;
basic_block def_bb = gimple_bb (SSA_NAME_DEF_STMT (var));
if (!def_bb || !dominated_by_p (CDI_DOMINATORS, loop->latch, def_bb))
return false;
rtype = get_range_info (var, &minv, &maxv);
if (rtype != VR_RANGE)
return false;
step_wi = wi::to_wide (step);
type = TREE_TYPE (var);
if (tree_int_cst_sign_bit (step))
{
diff = minv - wi::to_wide (lower_bound_in_type (type, type));
step_wi = - step_wi;
}
else
diff = wi::to_wide (upper_bound_in_type (type, type)) - maxv;
return (wi::geu_p (diff, step_wi));
}
bool
scev_probably_wraps_p (tree var, tree base, tree step,
gimple *at_stmt, struct loop *loop,
bool use_overflow_semantics)
{
if (chrec_contains_undetermined (base)
|| chrec_contains_undetermined (step))
return true;
if (integer_zerop (step))
return false;
if (use_overflow_semantics && nowrap_type_p (TREE_TYPE (base)))
return false;
if (TREE_CODE (step) != INTEGER_CST)
return true;
if (var && TREE_CODE (var) == SSA_NAME
&& scev_var_range_cant_overflow (var, step, loop))
return false;
if (loop_exits_before_overflow (base, step, at_stmt, loop))
return false;
return true;
}
void
free_numbers_of_iterations_estimates (struct loop *loop)
{
struct control_iv *civ;
struct nb_iter_bound *bound;
loop->nb_iterations = NULL;
loop->estimate_state = EST_NOT_COMPUTED;
for (bound = loop->bounds; bound;)
{
struct nb_iter_bound *next = bound->next;
ggc_free (bound);
bound = next;
}
loop->bounds = NULL;
for (civ = loop->control_ivs; civ;)
{
struct control_iv *next = civ->next;
ggc_free (civ);
civ = next;
}
loop->control_ivs = NULL;
}
void
free_numbers_of_iterations_estimates (function *fn)
{
struct loop *loop;
FOR_EACH_LOOP_FN (fn, loop, 0)
free_numbers_of_iterations_estimates (loop);
}
void
substitute_in_loop_info (struct loop *loop, tree name, tree val)
{
loop->nb_iterations = simplify_replace_tree (loop->nb_iterations, name, val);
}
