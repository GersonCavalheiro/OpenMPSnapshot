#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "target.h"
#include "rtl.h"
#include "tree.h"
#include "gimple.h"
#include "predict.h"
#include "memmodel.h"
#include "tm_p.h"
#include "tree-ssa-operands.h"
#include "optabs-query.h"
#include "cgraph.h"
#include "diagnostic-core.h"
#include "flags.h"
#include "alias.h"
#include "fold-const.h"
#include "fold-const-call.h"
#include "stor-layout.h"
#include "calls.h"
#include "tree-iterator.h"
#include "expr.h"
#include "intl.h"
#include "langhooks.h"
#include "tree-eh.h"
#include "gimplify.h"
#include "tree-dfa.h"
#include "builtins.h"
#include "generic-match.h"
#include "gimple-fold.h"
#include "params.h"
#include "tree-into-ssa.h"
#include "md5.h"
#include "case-cfn-macros.h"
#include "stringpool.h"
#include "tree-vrp.h"
#include "tree-ssanames.h"
#include "selftest.h"
#include "stringpool.h"
#include "attribs.h"
#include "tree-vector-builder.h"
#include "vec-perm-indices.h"
int folding_initializer = 0;
enum comparison_code {
COMPCODE_FALSE = 0,
COMPCODE_LT = 1,
COMPCODE_EQ = 2,
COMPCODE_LE = 3,
COMPCODE_GT = 4,
COMPCODE_LTGT = 5,
COMPCODE_GE = 6,
COMPCODE_ORD = 7,
COMPCODE_UNORD = 8,
COMPCODE_UNLT = 9,
COMPCODE_UNEQ = 10,
COMPCODE_UNLE = 11,
COMPCODE_UNGT = 12,
COMPCODE_NE = 13,
COMPCODE_UNGE = 14,
COMPCODE_TRUE = 15
};
static bool negate_expr_p (tree);
static tree negate_expr (tree);
static tree associate_trees (location_t, tree, tree, enum tree_code, tree);
static enum comparison_code comparison_to_compcode (enum tree_code);
static enum tree_code compcode_to_comparison (enum comparison_code);
static int twoval_comparison_p (tree, tree *, tree *);
static tree eval_subst (location_t, tree, tree, tree, tree, tree);
static tree optimize_bit_field_compare (location_t, enum tree_code,
tree, tree, tree);
static int simple_operand_p (const_tree);
static bool simple_operand_p_2 (tree);
static tree range_binop (enum tree_code, tree, tree, int, tree, int);
static tree range_predecessor (tree);
static tree range_successor (tree);
static tree fold_range_test (location_t, enum tree_code, tree, tree, tree);
static tree fold_cond_expr_with_comparison (location_t, tree, tree, tree, tree);
static tree unextend (tree, int, int, tree);
static tree extract_muldiv (tree, tree, enum tree_code, tree, bool *);
static tree extract_muldiv_1 (tree, tree, enum tree_code, tree, bool *);
static tree fold_binary_op_with_conditional_arg (location_t,
enum tree_code, tree,
tree, tree,
tree, tree, int);
static tree fold_negate_const (tree, tree);
static tree fold_not_const (const_tree, tree);
static tree fold_relational_const (enum tree_code, tree, tree, tree);
static tree fold_convert_const (enum tree_code, tree, tree);
static tree fold_view_convert_expr (tree, tree);
static tree fold_negate_expr (location_t, tree);
static location_t
expr_location_or (tree t, location_t loc)
{
location_t tloc = EXPR_LOCATION (t);
return tloc == UNKNOWN_LOCATION ? loc : tloc;
}
static inline tree
protected_set_expr_location_unshare (tree x, location_t loc)
{
if (CAN_HAVE_LOCATION_P (x)
&& EXPR_LOCATION (x) != loc
&& !(TREE_CODE (x) == SAVE_EXPR
|| TREE_CODE (x) == TARGET_EXPR
|| TREE_CODE (x) == BIND_EXPR))
{
x = copy_node (x);
SET_EXPR_LOCATION (x, loc);
}
return x;
}

tree
div_if_zero_remainder (const_tree arg1, const_tree arg2)
{
widest_int quo;
if (wi::multiple_of_p (wi::to_widest (arg1), wi::to_widest (arg2),
SIGNED, &quo))
return wide_int_to_tree (TREE_TYPE (arg1), quo);
return NULL_TREE; 
}

static int fold_deferring_overflow_warnings;
static const char* fold_deferred_overflow_warning;
static enum warn_strict_overflow_code fold_deferred_overflow_code;
void
fold_defer_overflow_warnings (void)
{
++fold_deferring_overflow_warnings;
}
void
fold_undefer_overflow_warnings (bool issue, const gimple *stmt, int code)
{
const char *warnmsg;
location_t locus;
gcc_assert (fold_deferring_overflow_warnings > 0);
--fold_deferring_overflow_warnings;
if (fold_deferring_overflow_warnings > 0)
{
if (fold_deferred_overflow_warning != NULL
&& code != 0
&& code < (int) fold_deferred_overflow_code)
fold_deferred_overflow_code = (enum warn_strict_overflow_code) code;
return;
}
warnmsg = fold_deferred_overflow_warning;
fold_deferred_overflow_warning = NULL;
if (!issue || warnmsg == NULL)
return;
if (gimple_no_warning_p (stmt))
return;
if (code == 0 || code > (int) fold_deferred_overflow_code)
code = fold_deferred_overflow_code;
if (!issue_strict_overflow_warning (code))
return;
if (stmt == NULL)
locus = input_location;
else
locus = gimple_location (stmt);
warning_at (locus, OPT_Wstrict_overflow, "%s", warnmsg);
}
void
fold_undefer_and_ignore_overflow_warnings (void)
{
fold_undefer_overflow_warnings (false, NULL, 0);
}
bool
fold_deferring_overflow_warnings_p (void)
{
return fold_deferring_overflow_warnings > 0;
}
void
fold_overflow_warning (const char* gmsgid, enum warn_strict_overflow_code wc)
{
if (fold_deferring_overflow_warnings > 0)
{
if (fold_deferred_overflow_warning == NULL
|| wc < fold_deferred_overflow_code)
{
fold_deferred_overflow_warning = gmsgid;
fold_deferred_overflow_code = wc;
}
}
else if (issue_strict_overflow_warning (wc))
warning (OPT_Wstrict_overflow, gmsgid);
}

bool
negate_mathfn_p (combined_fn fn)
{
switch (fn)
{
CASE_CFN_ASIN:
CASE_CFN_ASINH:
CASE_CFN_ATAN:
CASE_CFN_ATANH:
CASE_CFN_CASIN:
CASE_CFN_CASINH:
CASE_CFN_CATAN:
CASE_CFN_CATANH:
CASE_CFN_CBRT:
CASE_CFN_CPROJ:
CASE_CFN_CSIN:
CASE_CFN_CSINH:
CASE_CFN_CTAN:
CASE_CFN_CTANH:
CASE_CFN_ERF:
CASE_CFN_LLROUND:
CASE_CFN_LROUND:
CASE_CFN_ROUND:
CASE_CFN_SIN:
CASE_CFN_SINH:
CASE_CFN_TAN:
CASE_CFN_TANH:
CASE_CFN_TRUNC:
return true;
CASE_CFN_LLRINT:
CASE_CFN_LRINT:
CASE_CFN_NEARBYINT:
CASE_CFN_RINT:
return !flag_rounding_math;
default:
break;
}
return false;
}
bool
may_negate_without_overflow_p (const_tree t)
{
tree type;
gcc_assert (TREE_CODE (t) == INTEGER_CST);
type = TREE_TYPE (t);
if (TYPE_UNSIGNED (type))
return false;
return !wi::only_sign_bit_p (wi::to_wide (t));
}
static bool
negate_expr_p (tree t)
{
tree type;
if (t == 0)
return false;
type = TREE_TYPE (t);
STRIP_SIGN_NOPS (t);
switch (TREE_CODE (t))
{
case INTEGER_CST:
if (INTEGRAL_TYPE_P (type) && TYPE_UNSIGNED (type))
return true;
return may_negate_without_overflow_p (t);
case BIT_NOT_EXPR:
return (INTEGRAL_TYPE_P (type)
&& TYPE_OVERFLOW_WRAPS (type));
case FIXED_CST:
return true;
case NEGATE_EXPR:
return !TYPE_OVERFLOW_SANITIZED (type);
case REAL_CST:
return REAL_VALUE_NEGATIVE (TREE_REAL_CST (t));
case COMPLEX_CST:
return negate_expr_p (TREE_REALPART (t))
&& negate_expr_p (TREE_IMAGPART (t));
case VECTOR_CST:
{
if (FLOAT_TYPE_P (TREE_TYPE (type)) || TYPE_OVERFLOW_WRAPS (type))
return true;
unsigned int count = vector_cst_encoded_nelts (t);
for (unsigned int i = 0; i < count; ++i)
if (!negate_expr_p (VECTOR_CST_ENCODED_ELT (t, i)))
return false;
return true;
}
case COMPLEX_EXPR:
return negate_expr_p (TREE_OPERAND (t, 0))
&& negate_expr_p (TREE_OPERAND (t, 1));
case CONJ_EXPR:
return negate_expr_p (TREE_OPERAND (t, 0));
case PLUS_EXPR:
if (HONOR_SIGN_DEPENDENT_ROUNDING (element_mode (type))
|| HONOR_SIGNED_ZEROS (element_mode (type))
|| (ANY_INTEGRAL_TYPE_P (type)
&& ! TYPE_OVERFLOW_WRAPS (type)))
return false;
if (negate_expr_p (TREE_OPERAND (t, 1)))
return true;
return negate_expr_p (TREE_OPERAND (t, 0));
case MINUS_EXPR:
return !HONOR_SIGN_DEPENDENT_ROUNDING (element_mode (type))
&& !HONOR_SIGNED_ZEROS (element_mode (type))
&& (! ANY_INTEGRAL_TYPE_P (type)
|| TYPE_OVERFLOW_WRAPS (type));
case MULT_EXPR:
if (TYPE_UNSIGNED (type))
break;
if (INTEGRAL_TYPE_P (TREE_TYPE (t))
&& ! TYPE_OVERFLOW_WRAPS (TREE_TYPE (t))
&& ! ((TREE_CODE (TREE_OPERAND (t, 0)) == INTEGER_CST
&& (wi::popcount
(wi::abs (wi::to_wide (TREE_OPERAND (t, 0))))) != 1)
|| (TREE_CODE (TREE_OPERAND (t, 1)) == INTEGER_CST
&& (wi::popcount
(wi::abs (wi::to_wide (TREE_OPERAND (t, 1))))) != 1)))
break;
case RDIV_EXPR:
if (! HONOR_SIGN_DEPENDENT_ROUNDING (element_mode (TREE_TYPE (t))))
return negate_expr_p (TREE_OPERAND (t, 1))
|| negate_expr_p (TREE_OPERAND (t, 0));
break;
case TRUNC_DIV_EXPR:
case ROUND_DIV_EXPR:
case EXACT_DIV_EXPR:
if (TYPE_UNSIGNED (type))
break;
if (TREE_CODE (TREE_OPERAND (t, 0)) == INTEGER_CST
&& negate_expr_p (TREE_OPERAND (t, 0)))
return true;
if (! ANY_INTEGRAL_TYPE_P (TREE_TYPE (t))
|| TYPE_OVERFLOW_WRAPS (TREE_TYPE (t))
|| (TREE_CODE (TREE_OPERAND (t, 1)) == INTEGER_CST
&& ! integer_onep (TREE_OPERAND (t, 1))))
return negate_expr_p (TREE_OPERAND (t, 1));
break;
case NOP_EXPR:
if (TREE_CODE (type) == REAL_TYPE)
{
tree tem = strip_float_extensions (t);
if (tem != t)
return negate_expr_p (tem);
}
break;
case CALL_EXPR:
if (negate_mathfn_p (get_call_combined_fn (t)))
return negate_expr_p (CALL_EXPR_ARG (t, 0));
break;
case RSHIFT_EXPR:
if (TREE_CODE (TREE_OPERAND (t, 1)) == INTEGER_CST)
{
tree op1 = TREE_OPERAND (t, 1);
if (wi::to_wide (op1) == TYPE_PRECISION (type) - 1)
return true;
}
break;
default:
break;
}
return false;
}
static tree
fold_negate_expr_1 (location_t loc, tree t)
{
tree type = TREE_TYPE (t);
tree tem;
switch (TREE_CODE (t))
{
case BIT_NOT_EXPR:
if (INTEGRAL_TYPE_P (type))
return fold_build2_loc (loc, PLUS_EXPR, type, TREE_OPERAND (t, 0),
build_one_cst (type));
break;
case INTEGER_CST:
tem = fold_negate_const (t, type);
if (TREE_OVERFLOW (tem) == TREE_OVERFLOW (t)
|| (ANY_INTEGRAL_TYPE_P (type)
&& !TYPE_OVERFLOW_TRAPS (type)
&& TYPE_OVERFLOW_WRAPS (type))
|| (flag_sanitize & SANITIZE_SI_OVERFLOW) == 0)
return tem;
break;
case POLY_INT_CST:
case REAL_CST:
case FIXED_CST:
tem = fold_negate_const (t, type);
return tem;
case COMPLEX_CST:
{
tree rpart = fold_negate_expr (loc, TREE_REALPART (t));
tree ipart = fold_negate_expr (loc, TREE_IMAGPART (t));
if (rpart && ipart)
return build_complex (type, rpart, ipart);
}
break;
case VECTOR_CST:
{
tree_vector_builder elts;
elts.new_unary_operation (type, t, true);
unsigned int count = elts.encoded_nelts ();
for (unsigned int i = 0; i < count; ++i)
{
tree elt = fold_negate_expr (loc, VECTOR_CST_ELT (t, i));
if (elt == NULL_TREE)
return NULL_TREE;
elts.quick_push (elt);
}
return elts.build ();
}
case COMPLEX_EXPR:
if (negate_expr_p (t))
return fold_build2_loc (loc, COMPLEX_EXPR, type,
fold_negate_expr (loc, TREE_OPERAND (t, 0)),
fold_negate_expr (loc, TREE_OPERAND (t, 1)));
break;
case CONJ_EXPR:
if (negate_expr_p (t))
return fold_build1_loc (loc, CONJ_EXPR, type,
fold_negate_expr (loc, TREE_OPERAND (t, 0)));
break;
case NEGATE_EXPR:
if (!TYPE_OVERFLOW_SANITIZED (type))
return TREE_OPERAND (t, 0);
break;
case PLUS_EXPR:
if (!HONOR_SIGN_DEPENDENT_ROUNDING (element_mode (type))
&& !HONOR_SIGNED_ZEROS (element_mode (type)))
{
if (negate_expr_p (TREE_OPERAND (t, 1)))
{
tem = negate_expr (TREE_OPERAND (t, 1));
return fold_build2_loc (loc, MINUS_EXPR, type,
tem, TREE_OPERAND (t, 0));
}
if (negate_expr_p (TREE_OPERAND (t, 0)))
{
tem = negate_expr (TREE_OPERAND (t, 0));
return fold_build2_loc (loc, MINUS_EXPR, type,
tem, TREE_OPERAND (t, 1));
}
}
break;
case MINUS_EXPR:
if (!HONOR_SIGN_DEPENDENT_ROUNDING (element_mode (type))
&& !HONOR_SIGNED_ZEROS (element_mode (type)))
return fold_build2_loc (loc, MINUS_EXPR, type,
TREE_OPERAND (t, 1), TREE_OPERAND (t, 0));
break;
case MULT_EXPR:
if (TYPE_UNSIGNED (type))
break;
case RDIV_EXPR:
if (! HONOR_SIGN_DEPENDENT_ROUNDING (element_mode (type)))
{
tem = TREE_OPERAND (t, 1);
if (negate_expr_p (tem))
return fold_build2_loc (loc, TREE_CODE (t), type,
TREE_OPERAND (t, 0), negate_expr (tem));
tem = TREE_OPERAND (t, 0);
if (negate_expr_p (tem))
return fold_build2_loc (loc, TREE_CODE (t), type,
negate_expr (tem), TREE_OPERAND (t, 1));
}
break;
case TRUNC_DIV_EXPR:
case ROUND_DIV_EXPR:
case EXACT_DIV_EXPR:
if (TYPE_UNSIGNED (type))
break;
if (TREE_CODE (TREE_OPERAND (t, 0)) == INTEGER_CST
&& negate_expr_p (TREE_OPERAND (t, 0)))
return fold_build2_loc (loc, TREE_CODE (t), type,
negate_expr (TREE_OPERAND (t, 0)),
TREE_OPERAND (t, 1));
if ((! ANY_INTEGRAL_TYPE_P (TREE_TYPE (t))
|| TYPE_OVERFLOW_WRAPS (TREE_TYPE (t))
|| (TREE_CODE (TREE_OPERAND (t, 1)) == INTEGER_CST
&& ! integer_onep (TREE_OPERAND (t, 1))))
&& negate_expr_p (TREE_OPERAND (t, 1)))
return fold_build2_loc (loc, TREE_CODE (t), type,
TREE_OPERAND (t, 0),
negate_expr (TREE_OPERAND (t, 1)));
break;
case NOP_EXPR:
if (TREE_CODE (type) == REAL_TYPE)
{
tem = strip_float_extensions (t);
if (tem != t && negate_expr_p (tem))
return fold_convert_loc (loc, type, negate_expr (tem));
}
break;
case CALL_EXPR:
if (negate_mathfn_p (get_call_combined_fn (t))
&& negate_expr_p (CALL_EXPR_ARG (t, 0)))
{
tree fndecl, arg;
fndecl = get_callee_fndecl (t);
arg = negate_expr (CALL_EXPR_ARG (t, 0));
return build_call_expr_loc (loc, fndecl, 1, arg);
}
break;
case RSHIFT_EXPR:
if (TREE_CODE (TREE_OPERAND (t, 1)) == INTEGER_CST)
{
tree op1 = TREE_OPERAND (t, 1);
if (wi::to_wide (op1) == TYPE_PRECISION (type) - 1)
{
tree ntype = TYPE_UNSIGNED (type)
? signed_type_for (type)
: unsigned_type_for (type);
tree temp = fold_convert_loc (loc, ntype, TREE_OPERAND (t, 0));
temp = fold_build2_loc (loc, RSHIFT_EXPR, ntype, temp, op1);
return fold_convert_loc (loc, type, temp);
}
}
break;
default:
break;
}
return NULL_TREE;
}
static tree
fold_negate_expr (location_t loc, tree t)
{
tree type = TREE_TYPE (t);
STRIP_SIGN_NOPS (t);
tree tem = fold_negate_expr_1 (loc, t);
if (tem == NULL_TREE)
return NULL_TREE;
return fold_convert_loc (loc, type, tem);
}
static tree
negate_expr (tree t)
{
tree type, tem;
location_t loc;
if (t == NULL_TREE)
return NULL_TREE;
loc = EXPR_LOCATION (t);
type = TREE_TYPE (t);
STRIP_SIGN_NOPS (t);
tem = fold_negate_expr (loc, t);
if (!tem)
tem = build1_loc (loc, NEGATE_EXPR, TREE_TYPE (t), t);
return fold_convert_loc (loc, type, tem);
}

static tree
split_tree (tree in, tree type, enum tree_code code,
tree *minus_varp, tree *conp, tree *minus_conp,
tree *litp, tree *minus_litp, int negate_p)
{
tree var = 0;
*minus_varp = 0;
*conp = 0;
*minus_conp = 0;
*litp = 0;
*minus_litp = 0;
STRIP_SIGN_NOPS (in);
if (TREE_CODE (in) == INTEGER_CST || TREE_CODE (in) == REAL_CST
|| TREE_CODE (in) == FIXED_CST)
*litp = in;
else if (TREE_CODE (in) == code
|| ((! FLOAT_TYPE_P (TREE_TYPE (in)) || flag_associative_math)
&& ! SAT_FIXED_POINT_TYPE_P (TREE_TYPE (in))
&& ((code == PLUS_EXPR && TREE_CODE (in) == POINTER_PLUS_EXPR)
|| (code == PLUS_EXPR && TREE_CODE (in) == MINUS_EXPR)
|| (code == MINUS_EXPR
&& (TREE_CODE (in) == PLUS_EXPR
|| TREE_CODE (in) == POINTER_PLUS_EXPR)))))
{
tree op0 = TREE_OPERAND (in, 0);
tree op1 = TREE_OPERAND (in, 1);
int neg1_p = TREE_CODE (in) == MINUS_EXPR;
int neg_litp_p = 0, neg_conp_p = 0, neg_var_p = 0;
if (TREE_CODE (op0) == INTEGER_CST || TREE_CODE (op0) == REAL_CST
|| TREE_CODE (op0) == FIXED_CST)
*litp = op0, op0 = 0;
else if (TREE_CODE (op1) == INTEGER_CST || TREE_CODE (op1) == REAL_CST
|| TREE_CODE (op1) == FIXED_CST)
*litp = op1, neg_litp_p = neg1_p, op1 = 0;
if (op0 != 0 && TREE_CONSTANT (op0))
*conp = op0, op0 = 0;
else if (op1 != 0 && TREE_CONSTANT (op1))
*conp = op1, neg_conp_p = neg1_p, op1 = 0;
if (op0 != 0 && op1 != 0)
var = in;
else if (op0 != 0)
var = op0;
else
var = op1, neg_var_p = neg1_p;
if (neg_litp_p)
*minus_litp = *litp, *litp = 0;
if (neg_conp_p && *conp)
*minus_conp = *conp, *conp = 0;
if (neg_var_p && var)
*minus_varp = var, var = 0;
}
else if (TREE_CONSTANT (in))
*conp = in;
else if (TREE_CODE (in) == BIT_NOT_EXPR
&& code == PLUS_EXPR)
{
*litp = build_minus_one_cst (type);
*minus_varp = TREE_OPERAND (in, 0);
}
else
var = in;
if (negate_p)
{
if (*litp)
*minus_litp = *litp, *litp = 0;
else if (*minus_litp)
*litp = *minus_litp, *minus_litp = 0;
if (*conp)
*minus_conp = *conp, *conp = 0;
else if (*minus_conp)
*conp = *minus_conp, *minus_conp = 0;
if (var)
*minus_varp = var, var = 0;
else if (*minus_varp)
var = *minus_varp, *minus_varp = 0;
}
if (*litp
&& TREE_OVERFLOW_P (*litp))
*litp = drop_tree_overflow (*litp);
if (*minus_litp
&& TREE_OVERFLOW_P (*minus_litp))
*minus_litp = drop_tree_overflow (*minus_litp);
return var;
}
static tree
associate_trees (location_t loc, tree t1, tree t2, enum tree_code code, tree type)
{
if (t1 == 0)
{
gcc_assert (t2 == 0 || code != MINUS_EXPR);
return t2;
}
else if (t2 == 0)
return t1;
if (TREE_CODE (t1) == code || TREE_CODE (t2) == code
|| TREE_CODE (t1) == PLUS_EXPR || TREE_CODE (t2) == PLUS_EXPR
|| TREE_CODE (t1) == MINUS_EXPR || TREE_CODE (t2) == MINUS_EXPR)
{
if (code == PLUS_EXPR)
{
if (TREE_CODE (t1) == NEGATE_EXPR)
return build2_loc (loc, MINUS_EXPR, type,
fold_convert_loc (loc, type, t2),
fold_convert_loc (loc, type,
TREE_OPERAND (t1, 0)));
else if (TREE_CODE (t2) == NEGATE_EXPR)
return build2_loc (loc, MINUS_EXPR, type,
fold_convert_loc (loc, type, t1),
fold_convert_loc (loc, type,
TREE_OPERAND (t2, 0)));
else if (integer_zerop (t2))
return fold_convert_loc (loc, type, t1);
}
else if (code == MINUS_EXPR)
{
if (integer_zerop (t2))
return fold_convert_loc (loc, type, t1);
}
return build2_loc (loc, code, type, fold_convert_loc (loc, type, t1),
fold_convert_loc (loc, type, t2));
}
return fold_build2_loc (loc, code, type, fold_convert_loc (loc, type, t1),
fold_convert_loc (loc, type, t2));
}

static bool
int_binop_types_match_p (enum tree_code code, const_tree type1, const_tree type2)
{
if (!INTEGRAL_TYPE_P (type1) && !POINTER_TYPE_P (type1))
return false;
if (!INTEGRAL_TYPE_P (type2) && !POINTER_TYPE_P (type2))
return false;
switch (code)
{
case LSHIFT_EXPR:
case RSHIFT_EXPR:
case LROTATE_EXPR:
case RROTATE_EXPR:
return true;
default:
break;
}
return TYPE_UNSIGNED (type1) == TYPE_UNSIGNED (type2)
&& TYPE_PRECISION (type1) == TYPE_PRECISION (type2)
&& TYPE_MODE (type1) == TYPE_MODE (type2);
}
static tree
int_const_binop_2 (enum tree_code code, const_tree parg1, const_tree parg2,
int overflowable)
{
wide_int res;
tree t;
tree type = TREE_TYPE (parg1);
signop sign = TYPE_SIGN (type);
bool overflow = false;
wi::tree_to_wide_ref arg1 = wi::to_wide (parg1);
wide_int arg2 = wi::to_wide (parg2, TYPE_PRECISION (type));
switch (code)
{
case BIT_IOR_EXPR:
res = wi::bit_or (arg1, arg2);
break;
case BIT_XOR_EXPR:
res = wi::bit_xor (arg1, arg2);
break;
case BIT_AND_EXPR:
res = wi::bit_and (arg1, arg2);
break;
case RSHIFT_EXPR:
case LSHIFT_EXPR:
if (wi::neg_p (arg2))
{
arg2 = -arg2;
if (code == RSHIFT_EXPR)
code = LSHIFT_EXPR;
else
code = RSHIFT_EXPR;
}
if (code == RSHIFT_EXPR)
res = wi::rshift (arg1, arg2, sign);
else
res = wi::lshift (arg1, arg2);
break;
case RROTATE_EXPR:
case LROTATE_EXPR:
if (wi::neg_p (arg2))
{
arg2 = -arg2;
if (code == RROTATE_EXPR)
code = LROTATE_EXPR;
else
code = RROTATE_EXPR;
}
if (code == RROTATE_EXPR)
res = wi::rrotate (arg1, arg2);
else
res = wi::lrotate (arg1, arg2);
break;
case PLUS_EXPR:
res = wi::add (arg1, arg2, sign, &overflow);
break;
case MINUS_EXPR:
res = wi::sub (arg1, arg2, sign, &overflow);
break;
case MULT_EXPR:
res = wi::mul (arg1, arg2, sign, &overflow);
break;
case MULT_HIGHPART_EXPR:
res = wi::mul_high (arg1, arg2, sign);
break;
case TRUNC_DIV_EXPR:
case EXACT_DIV_EXPR:
if (arg2 == 0)
return NULL_TREE;
res = wi::div_trunc (arg1, arg2, sign, &overflow);
break;
case FLOOR_DIV_EXPR:
if (arg2 == 0)
return NULL_TREE;
res = wi::div_floor (arg1, arg2, sign, &overflow);
break;
case CEIL_DIV_EXPR:
if (arg2 == 0)
return NULL_TREE;
res = wi::div_ceil (arg1, arg2, sign, &overflow);
break;
case ROUND_DIV_EXPR:
if (arg2 == 0)
return NULL_TREE;
res = wi::div_round (arg1, arg2, sign, &overflow);
break;
case TRUNC_MOD_EXPR:
if (arg2 == 0)
return NULL_TREE;
res = wi::mod_trunc (arg1, arg2, sign, &overflow);
break;
case FLOOR_MOD_EXPR:
if (arg2 == 0)
return NULL_TREE;
res = wi::mod_floor (arg1, arg2, sign, &overflow);
break;
case CEIL_MOD_EXPR:
if (arg2 == 0)
return NULL_TREE;
res = wi::mod_ceil (arg1, arg2, sign, &overflow);
break;
case ROUND_MOD_EXPR:
if (arg2 == 0)
return NULL_TREE;
res = wi::mod_round (arg1, arg2, sign, &overflow);
break;
case MIN_EXPR:
res = wi::min (arg1, arg2, sign);
break;
case MAX_EXPR:
res = wi::max (arg1, arg2, sign);
break;
default:
return NULL_TREE;
}
t = force_fit_type (type, res, overflowable,
(((sign == SIGNED || overflowable == -1)
&& overflow)
| TREE_OVERFLOW (parg1) | TREE_OVERFLOW (parg2)));
return t;
}
static tree
int_const_binop_1 (enum tree_code code, const_tree arg1, const_tree arg2,
int overflowable)
{
if (TREE_CODE (arg1) == INTEGER_CST && TREE_CODE (arg2) == INTEGER_CST)
return int_const_binop_2 (code, arg1, arg2, overflowable);
gcc_assert (NUM_POLY_INT_COEFFS != 1);
if (poly_int_tree_p (arg1) && poly_int_tree_p (arg2))
{
poly_wide_int res;
bool overflow;
tree type = TREE_TYPE (arg1);
signop sign = TYPE_SIGN (type);
switch (code)
{
case PLUS_EXPR:
res = wi::add (wi::to_poly_wide (arg1),
wi::to_poly_wide (arg2), sign, &overflow);
break;
case MINUS_EXPR:
res = wi::sub (wi::to_poly_wide (arg1),
wi::to_poly_wide (arg2), sign, &overflow);
break;
case MULT_EXPR:
if (TREE_CODE (arg2) == INTEGER_CST)
res = wi::mul (wi::to_poly_wide (arg1),
wi::to_wide (arg2), sign, &overflow);
else if (TREE_CODE (arg1) == INTEGER_CST)
res = wi::mul (wi::to_poly_wide (arg2),
wi::to_wide (arg1), sign, &overflow);
else
return NULL_TREE;
break;
case LSHIFT_EXPR:
if (TREE_CODE (arg2) == INTEGER_CST)
res = wi::to_poly_wide (arg1) << wi::to_wide (arg2);
else
return NULL_TREE;
break;
case BIT_IOR_EXPR:
if (TREE_CODE (arg2) != INTEGER_CST
|| !can_ior_p (wi::to_poly_wide (arg1), wi::to_wide (arg2),
&res))
return NULL_TREE;
break;
default:
return NULL_TREE;
}
return force_fit_type (type, res, overflowable,
(((sign == SIGNED || overflowable == -1)
&& overflow)
| TREE_OVERFLOW (arg1) | TREE_OVERFLOW (arg2)));
}
return NULL_TREE;
}
tree
int_const_binop (enum tree_code code, const_tree arg1, const_tree arg2)
{
return int_const_binop_1 (code, arg1, arg2, 1);
}
static bool
distributes_over_addition_p (tree_code op, int opno)
{
switch (op)
{
case PLUS_EXPR:
case MINUS_EXPR:
case MULT_EXPR:
return true;
case LSHIFT_EXPR:
return opno == 1;
default:
return false;
}
}
static tree
const_binop (enum tree_code code, tree arg1, tree arg2)
{
if (!arg1 || !arg2)
return NULL_TREE;
STRIP_NOPS (arg1);
STRIP_NOPS (arg2);
if (poly_int_tree_p (arg1) && poly_int_tree_p (arg2))
{
if (code == POINTER_PLUS_EXPR)
return int_const_binop (PLUS_EXPR,
arg1, fold_convert (TREE_TYPE (arg1), arg2));
return int_const_binop (code, arg1, arg2);
}
if (TREE_CODE (arg1) == REAL_CST && TREE_CODE (arg2) == REAL_CST)
{
machine_mode mode;
REAL_VALUE_TYPE d1;
REAL_VALUE_TYPE d2;
REAL_VALUE_TYPE value;
REAL_VALUE_TYPE result;
bool inexact;
tree t, type;
switch (code)
{
case PLUS_EXPR:
case MINUS_EXPR:
case MULT_EXPR:
case RDIV_EXPR:
case MIN_EXPR:
case MAX_EXPR:
break;
default:
return NULL_TREE;
}
d1 = TREE_REAL_CST (arg1);
d2 = TREE_REAL_CST (arg2);
type = TREE_TYPE (arg1);
mode = TYPE_MODE (type);
if (HONOR_SNANS (mode)
&& (REAL_VALUE_ISSIGNALING_NAN (d1)
|| REAL_VALUE_ISSIGNALING_NAN (d2)))
return NULL_TREE;
if (code == RDIV_EXPR
&& real_equal (&d2, &dconst0)
&& (flag_trapping_math || ! MODE_HAS_INFINITIES (mode)))
return NULL_TREE;
if (REAL_VALUE_ISNAN (d1))
{
d1.signalling = 0;
t = build_real (type, d1);
return t;
}
else if (REAL_VALUE_ISNAN (d2))
{
d2.signalling = 0;
t = build_real (type, d2);
return t;
}
inexact = real_arithmetic (&value, code, &d1, &d2);
real_convert (&result, mode, &value);
if (flag_trapping_math
&& MODE_HAS_INFINITIES (mode)
&& REAL_VALUE_ISINF (result)
&& !REAL_VALUE_ISINF (d1)
&& !REAL_VALUE_ISINF (d2))
return NULL_TREE;
if ((flag_rounding_math
|| (MODE_COMPOSITE_P (mode) && !flag_unsafe_math_optimizations))
&& (inexact || !real_identical (&result, &value)))
return NULL_TREE;
t = build_real (type, result);
TREE_OVERFLOW (t) = TREE_OVERFLOW (arg1) | TREE_OVERFLOW (arg2);
return t;
}
if (TREE_CODE (arg1) == FIXED_CST)
{
FIXED_VALUE_TYPE f1;
FIXED_VALUE_TYPE f2;
FIXED_VALUE_TYPE result;
tree t, type;
int sat_p;
bool overflow_p;
switch (code)
{
case PLUS_EXPR:
case MINUS_EXPR:
case MULT_EXPR:
case TRUNC_DIV_EXPR:
if (TREE_CODE (arg2) != FIXED_CST)
return NULL_TREE;
f2 = TREE_FIXED_CST (arg2);
break;
case LSHIFT_EXPR:
case RSHIFT_EXPR:
{
if (TREE_CODE (arg2) != INTEGER_CST)
return NULL_TREE;
wi::tree_to_wide_ref w2 = wi::to_wide (arg2);
f2.data.high = w2.elt (1);
f2.data.low = w2.ulow ();
f2.mode = SImode;
}
break;
default:
return NULL_TREE;
}
f1 = TREE_FIXED_CST (arg1);
type = TREE_TYPE (arg1);
sat_p = TYPE_SATURATING (type);
overflow_p = fixed_arithmetic (&result, code, &f1, &f2, sat_p);
t = build_fixed (type, result);
if (overflow_p | TREE_OVERFLOW (arg1) | TREE_OVERFLOW (arg2))
TREE_OVERFLOW (t) = 1;
return t;
}
if (TREE_CODE (arg1) == COMPLEX_CST && TREE_CODE (arg2) == COMPLEX_CST)
{
tree type = TREE_TYPE (arg1);
tree r1 = TREE_REALPART (arg1);
tree i1 = TREE_IMAGPART (arg1);
tree r2 = TREE_REALPART (arg2);
tree i2 = TREE_IMAGPART (arg2);
tree real, imag;
switch (code)
{
case PLUS_EXPR:
case MINUS_EXPR:
real = const_binop (code, r1, r2);
imag = const_binop (code, i1, i2);
break;
case MULT_EXPR:
if (COMPLEX_FLOAT_TYPE_P (type))
return do_mpc_arg2 (arg1, arg2, type,
folding_initializer,
mpc_mul);
real = const_binop (MINUS_EXPR,
const_binop (MULT_EXPR, r1, r2),
const_binop (MULT_EXPR, i1, i2));
imag = const_binop (PLUS_EXPR,
const_binop (MULT_EXPR, r1, i2),
const_binop (MULT_EXPR, i1, r2));
break;
case RDIV_EXPR:
if (COMPLEX_FLOAT_TYPE_P (type))
return do_mpc_arg2 (arg1, arg2, type,
folding_initializer,
mpc_div);
case TRUNC_DIV_EXPR:
case CEIL_DIV_EXPR:
case FLOOR_DIV_EXPR:
case ROUND_DIV_EXPR:
if (flag_complex_method == 0)
{
tree magsquared
= const_binop (PLUS_EXPR,
const_binop (MULT_EXPR, r2, r2),
const_binop (MULT_EXPR, i2, i2));
tree t1
= const_binop (PLUS_EXPR,
const_binop (MULT_EXPR, r1, r2),
const_binop (MULT_EXPR, i1, i2));
tree t2
= const_binop (MINUS_EXPR,
const_binop (MULT_EXPR, i1, r2),
const_binop (MULT_EXPR, r1, i2));
real = const_binop (code, t1, magsquared);
imag = const_binop (code, t2, magsquared);
}
else
{
tree compare = fold_build2 (LT_EXPR, boolean_type_node,
fold_abs_const (r2, TREE_TYPE (type)),
fold_abs_const (i2, TREE_TYPE (type)));
if (integer_nonzerop (compare))
{
tree ratio = const_binop (code, r2, i2);
tree div = const_binop (PLUS_EXPR, i2,
const_binop (MULT_EXPR, r2, ratio));
real = const_binop (MULT_EXPR, r1, ratio);
real = const_binop (PLUS_EXPR, real, i1);
real = const_binop (code, real, div);
imag = const_binop (MULT_EXPR, i1, ratio);
imag = const_binop (MINUS_EXPR, imag, r1);
imag = const_binop (code, imag, div);
}
else
{
tree ratio = const_binop (code, i2, r2);
tree div = const_binop (PLUS_EXPR, r2,
const_binop (MULT_EXPR, i2, ratio));
real = const_binop (MULT_EXPR, i1, ratio);
real = const_binop (PLUS_EXPR, real, r1);
real = const_binop (code, real, div);
imag = const_binop (MULT_EXPR, r1, ratio);
imag = const_binop (MINUS_EXPR, i1, imag);
imag = const_binop (code, imag, div);
}
}
break;
default:
return NULL_TREE;
}
if (real && imag)
return build_complex (type, real, imag);
}
if (TREE_CODE (arg1) == VECTOR_CST
&& TREE_CODE (arg2) == VECTOR_CST
&& known_eq (TYPE_VECTOR_SUBPARTS (TREE_TYPE (arg1)),
TYPE_VECTOR_SUBPARTS (TREE_TYPE (arg2))))
{
tree type = TREE_TYPE (arg1);
bool step_ok_p;
if (VECTOR_CST_STEPPED_P (arg1)
&& VECTOR_CST_STEPPED_P (arg2))
step_ok_p = (code == PLUS_EXPR || code == MINUS_EXPR);
else if (VECTOR_CST_STEPPED_P (arg1))
step_ok_p = distributes_over_addition_p (code, 1);
else
step_ok_p = distributes_over_addition_p (code, 2);
tree_vector_builder elts;
if (!elts.new_binary_operation (type, arg1, arg2, step_ok_p))
return NULL_TREE;
unsigned int count = elts.encoded_nelts ();
for (unsigned int i = 0; i < count; ++i)
{
tree elem1 = VECTOR_CST_ELT (arg1, i);
tree elem2 = VECTOR_CST_ELT (arg2, i);
tree elt = const_binop (code, elem1, elem2);
if (elt == NULL_TREE)
return NULL_TREE;
elts.quick_push (elt);
}
return elts.build ();
}
if (TREE_CODE (arg1) == VECTOR_CST
&& TREE_CODE (arg2) == INTEGER_CST)
{
tree type = TREE_TYPE (arg1);
bool step_ok_p = distributes_over_addition_p (code, 1);
tree_vector_builder elts;
if (!elts.new_unary_operation (type, arg1, step_ok_p))
return NULL_TREE;
unsigned int count = elts.encoded_nelts ();
for (unsigned int i = 0; i < count; ++i)
{
tree elem1 = VECTOR_CST_ELT (arg1, i);
tree elt = const_binop (code, elem1, arg2);
if (elt == NULL_TREE)
return NULL_TREE;
elts.quick_push (elt);
}
return elts.build ();
}
return NULL_TREE;
}
tree
const_binop (enum tree_code code, tree type, tree arg1, tree arg2)
{
if (TREE_CODE_CLASS (code) == tcc_comparison)
return fold_relational_const (code, type, arg1, arg2);
switch (code)
{
case VEC_SERIES_EXPR:
if (CONSTANT_CLASS_P (arg1)
&& CONSTANT_CLASS_P (arg2))
return build_vec_series (type, arg1, arg2);
return NULL_TREE;
case COMPLEX_EXPR:
if ((TREE_CODE (arg1) == REAL_CST
&& TREE_CODE (arg2) == REAL_CST)
|| (TREE_CODE (arg1) == INTEGER_CST
&& TREE_CODE (arg2) == INTEGER_CST))
return build_complex (type, arg1, arg2);
return NULL_TREE;
case POINTER_DIFF_EXPR:
if (TREE_CODE (arg1) == INTEGER_CST && TREE_CODE (arg2) == INTEGER_CST)
{
offset_int res = wi::sub (wi::to_offset (arg1),
wi::to_offset (arg2));
return force_fit_type (type, res, 1,
TREE_OVERFLOW (arg1) | TREE_OVERFLOW (arg2));
}
return NULL_TREE;
case VEC_PACK_TRUNC_EXPR:
case VEC_PACK_FIX_TRUNC_EXPR:
{
unsigned int HOST_WIDE_INT out_nelts, in_nelts, i;
if (TREE_CODE (arg1) != VECTOR_CST
|| TREE_CODE (arg2) != VECTOR_CST)
return NULL_TREE;
if (!VECTOR_CST_NELTS (arg1).is_constant (&in_nelts))
return NULL_TREE;
out_nelts = in_nelts * 2;
gcc_assert (known_eq (in_nelts, VECTOR_CST_NELTS (arg2))
&& known_eq (out_nelts, TYPE_VECTOR_SUBPARTS (type)));
tree_vector_builder elts (type, out_nelts, 1);
for (i = 0; i < out_nelts; i++)
{
tree elt = (i < in_nelts
? VECTOR_CST_ELT (arg1, i)
: VECTOR_CST_ELT (arg2, i - in_nelts));
elt = fold_convert_const (code == VEC_PACK_TRUNC_EXPR
? NOP_EXPR : FIX_TRUNC_EXPR,
TREE_TYPE (type), elt);
if (elt == NULL_TREE || !CONSTANT_CLASS_P (elt))
return NULL_TREE;
elts.quick_push (elt);
}
return elts.build ();
}
case VEC_WIDEN_MULT_LO_EXPR:
case VEC_WIDEN_MULT_HI_EXPR:
case VEC_WIDEN_MULT_EVEN_EXPR:
case VEC_WIDEN_MULT_ODD_EXPR:
{
unsigned HOST_WIDE_INT out_nelts, in_nelts, out, ofs, scale;
if (TREE_CODE (arg1) != VECTOR_CST || TREE_CODE (arg2) != VECTOR_CST)
return NULL_TREE;
if (!VECTOR_CST_NELTS (arg1).is_constant (&in_nelts))
return NULL_TREE;
out_nelts = in_nelts / 2;
gcc_assert (known_eq (in_nelts, VECTOR_CST_NELTS (arg2))
&& known_eq (out_nelts, TYPE_VECTOR_SUBPARTS (type)));
if (code == VEC_WIDEN_MULT_LO_EXPR)
scale = 0, ofs = BYTES_BIG_ENDIAN ? out_nelts : 0;
else if (code == VEC_WIDEN_MULT_HI_EXPR)
scale = 0, ofs = BYTES_BIG_ENDIAN ? 0 : out_nelts;
else if (code == VEC_WIDEN_MULT_EVEN_EXPR)
scale = 1, ofs = 0;
else 
scale = 1, ofs = 1;
tree_vector_builder elts (type, out_nelts, 1);
for (out = 0; out < out_nelts; out++)
{
unsigned int in = (out << scale) + ofs;
tree t1 = fold_convert_const (NOP_EXPR, TREE_TYPE (type),
VECTOR_CST_ELT (arg1, in));
tree t2 = fold_convert_const (NOP_EXPR, TREE_TYPE (type),
VECTOR_CST_ELT (arg2, in));
if (t1 == NULL_TREE || t2 == NULL_TREE)
return NULL_TREE;
tree elt = const_binop (MULT_EXPR, t1, t2);
if (elt == NULL_TREE || !CONSTANT_CLASS_P (elt))
return NULL_TREE;
elts.quick_push (elt);
}
return elts.build ();
}
default:;
}
if (TREE_CODE_CLASS (code) != tcc_binary)
return NULL_TREE;
gcc_checking_assert (TYPE_SATURATING (type)
== TYPE_SATURATING (TREE_TYPE (arg1)));
return const_binop (code, arg1, arg2);
}
tree
const_unop (enum tree_code code, tree type, tree arg0)
{
if (TREE_CODE (arg0) == REAL_CST
&& HONOR_SNANS (TYPE_MODE (TREE_TYPE (arg0)))
&& REAL_VALUE_ISSIGNALING_NAN (TREE_REAL_CST (arg0))
&& code != NEGATE_EXPR
&& code != ABS_EXPR)
return NULL_TREE;
switch (code)
{
CASE_CONVERT:
case FLOAT_EXPR:
case FIX_TRUNC_EXPR:
case FIXED_CONVERT_EXPR:
return fold_convert_const (code, type, arg0);
case ADDR_SPACE_CONVERT_EXPR:
if (integer_zerop (arg0)
&& !(targetm.addr_space.zero_address_valid
(TYPE_ADDR_SPACE (TREE_TYPE (TREE_TYPE (arg0))))))
return fold_convert_const (code, type, arg0);
break;
case VIEW_CONVERT_EXPR:
return fold_view_convert_expr (type, arg0);
case NEGATE_EXPR:
{
tree tem = fold_negate_expr (UNKNOWN_LOCATION, arg0);
if (tem && CONSTANT_CLASS_P (tem))
return tem;
break;
}
case ABS_EXPR:
if (TREE_CODE (arg0) == INTEGER_CST || TREE_CODE (arg0) == REAL_CST)
return fold_abs_const (arg0, type);
break;
case CONJ_EXPR:
if (TREE_CODE (arg0) == COMPLEX_CST)
{
tree ipart = fold_negate_const (TREE_IMAGPART (arg0),
TREE_TYPE (type));
return build_complex (type, TREE_REALPART (arg0), ipart);
}
break;
case BIT_NOT_EXPR:
if (TREE_CODE (arg0) == INTEGER_CST)
return fold_not_const (arg0, type);
else if (POLY_INT_CST_P (arg0))
return wide_int_to_tree (type, -poly_int_cst_value (arg0));
else if (TREE_CODE (arg0) == VECTOR_CST)
{
tree elem;
tree_vector_builder elements;
elements.new_unary_operation (type, arg0, true);
unsigned int i, count = elements.encoded_nelts ();
for (i = 0; i < count; ++i)
{
elem = VECTOR_CST_ELT (arg0, i);
elem = const_unop (BIT_NOT_EXPR, TREE_TYPE (type), elem);
if (elem == NULL_TREE)
break;
elements.quick_push (elem);
}
if (i == count)
return elements.build ();
}
break;
case TRUTH_NOT_EXPR:
if (TREE_CODE (arg0) == INTEGER_CST)
return constant_boolean_node (integer_zerop (arg0), type);
break;
case REALPART_EXPR:
if (TREE_CODE (arg0) == COMPLEX_CST)
return fold_convert (type, TREE_REALPART (arg0));
break;
case IMAGPART_EXPR:
if (TREE_CODE (arg0) == COMPLEX_CST)
return fold_convert (type, TREE_IMAGPART (arg0));
break;
case VEC_UNPACK_LO_EXPR:
case VEC_UNPACK_HI_EXPR:
case VEC_UNPACK_FLOAT_LO_EXPR:
case VEC_UNPACK_FLOAT_HI_EXPR:
{
unsigned HOST_WIDE_INT out_nelts, in_nelts, i;
enum tree_code subcode;
if (TREE_CODE (arg0) != VECTOR_CST)
return NULL_TREE;
if (!VECTOR_CST_NELTS (arg0).is_constant (&in_nelts))
return NULL_TREE;
out_nelts = in_nelts / 2;
gcc_assert (known_eq (out_nelts, TYPE_VECTOR_SUBPARTS (type)));
unsigned int offset = 0;
if ((!BYTES_BIG_ENDIAN) ^ (code == VEC_UNPACK_LO_EXPR
|| code == VEC_UNPACK_FLOAT_LO_EXPR))
offset = out_nelts;
if (code == VEC_UNPACK_LO_EXPR || code == VEC_UNPACK_HI_EXPR)
subcode = NOP_EXPR;
else
subcode = FLOAT_EXPR;
tree_vector_builder elts (type, out_nelts, 1);
for (i = 0; i < out_nelts; i++)
{
tree elt = fold_convert_const (subcode, TREE_TYPE (type),
VECTOR_CST_ELT (arg0, i + offset));
if (elt == NULL_TREE || !CONSTANT_CLASS_P (elt))
return NULL_TREE;
elts.quick_push (elt);
}
return elts.build ();
}
case VEC_DUPLICATE_EXPR:
if (CONSTANT_CLASS_P (arg0))
return build_vector_from_val (type, arg0);
return NULL_TREE;
default:
break;
}
return NULL_TREE;
}
tree
size_int_kind (poly_int64 number, enum size_type_kind kind)
{
return build_int_cst (sizetype_tab[(int) kind], number);
}

tree
size_binop_loc (location_t loc, enum tree_code code, tree arg0, tree arg1)
{
tree type = TREE_TYPE (arg0);
if (arg0 == error_mark_node || arg1 == error_mark_node)
return error_mark_node;
gcc_assert (int_binop_types_match_p (code, TREE_TYPE (arg0),
TREE_TYPE (arg1)));
if (poly_int_tree_p (arg0) && poly_int_tree_p (arg1))
{
if (code == PLUS_EXPR)
{
if (integer_zerop (arg0) && !TREE_OVERFLOW (arg0))
return arg1;
if (integer_zerop (arg1) && !TREE_OVERFLOW (arg1))
return arg0;
}
else if (code == MINUS_EXPR)
{
if (integer_zerop (arg1) && !TREE_OVERFLOW (arg1))
return arg0;
}
else if (code == MULT_EXPR)
{
if (integer_onep (arg0) && !TREE_OVERFLOW (arg0))
return arg1;
}
tree res = int_const_binop_1 (code, arg0, arg1, -1);
if (res != NULL_TREE)
return res;
}
return fold_build2_loc (loc, code, type, arg0, arg1);
}
tree
size_diffop_loc (location_t loc, tree arg0, tree arg1)
{
tree type = TREE_TYPE (arg0);
tree ctype;
gcc_assert (int_binop_types_match_p (MINUS_EXPR, TREE_TYPE (arg0),
TREE_TYPE (arg1)));
if (!TYPE_UNSIGNED (type))
return size_binop_loc (loc, MINUS_EXPR, arg0, arg1);
if (type == sizetype)
ctype = ssizetype;
else if (type == bitsizetype)
ctype = sbitsizetype;
else
ctype = signed_type_for (type);
if (TREE_CODE (arg0) != INTEGER_CST || TREE_CODE (arg1) != INTEGER_CST)
return size_binop_loc (loc, MINUS_EXPR,
fold_convert_loc (loc, ctype, arg0),
fold_convert_loc (loc, ctype, arg1));
if (tree_int_cst_equal (arg0, arg1))
return build_int_cst (ctype, 0);
else if (tree_int_cst_lt (arg1, arg0))
return fold_convert_loc (loc, ctype,
size_binop_loc (loc, MINUS_EXPR, arg0, arg1));
else
return size_binop_loc (loc, MINUS_EXPR, build_int_cst (ctype, 0),
fold_convert_loc (loc, ctype,
size_binop_loc (loc,
MINUS_EXPR,
arg1, arg0)));
}

static tree
fold_convert_const_int_from_int (tree type, const_tree arg1)
{
return force_fit_type (type, wi::to_widest (arg1),
!POINTER_TYPE_P (TREE_TYPE (arg1)),
TREE_OVERFLOW (arg1));
}
static tree
fold_convert_const_int_from_real (enum tree_code code, tree type, const_tree arg1)
{
bool overflow = false;
tree t;
wide_int val;
REAL_VALUE_TYPE r;
REAL_VALUE_TYPE x = TREE_REAL_CST (arg1);
switch (code)
{
case FIX_TRUNC_EXPR:
real_trunc (&r, VOIDmode, &x);
break;
default:
gcc_unreachable ();
}
if (REAL_VALUE_ISNAN (r))
{
overflow = true;
val = wi::zero (TYPE_PRECISION (type));
}
if (! overflow)
{
tree lt = TYPE_MIN_VALUE (type);
REAL_VALUE_TYPE l = real_value_from_int_cst (NULL_TREE, lt);
if (real_less (&r, &l))
{
overflow = true;
val = wi::to_wide (lt);
}
}
if (! overflow)
{
tree ut = TYPE_MAX_VALUE (type);
if (ut)
{
REAL_VALUE_TYPE u = real_value_from_int_cst (NULL_TREE, ut);
if (real_less (&u, &r))
{
overflow = true;
val = wi::to_wide (ut);
}
}
}
if (! overflow)
val = real_to_integer (&r, &overflow, TYPE_PRECISION (type));
t = force_fit_type (type, val, -1, overflow | TREE_OVERFLOW (arg1));
return t;
}
static tree
fold_convert_const_int_from_fixed (tree type, const_tree arg1)
{
tree t;
double_int temp, temp_trunc;
scalar_mode mode;
temp = TREE_FIXED_CST (arg1).data;
mode = TREE_FIXED_CST (arg1).mode;
if (GET_MODE_FBIT (mode) < HOST_BITS_PER_DOUBLE_INT)
{
temp = temp.rshift (GET_MODE_FBIT (mode),
HOST_BITS_PER_DOUBLE_INT,
SIGNED_FIXED_POINT_MODE_P (mode));
temp_trunc = temp.lshift (GET_MODE_FBIT (mode),
HOST_BITS_PER_DOUBLE_INT,
SIGNED_FIXED_POINT_MODE_P (mode));
}
else
{
temp = double_int_zero;
temp_trunc = double_int_zero;
}
if (SIGNED_FIXED_POINT_MODE_P (mode)
&& temp_trunc.is_negative ()
&& TREE_FIXED_CST (arg1).data != temp_trunc)
temp += double_int_one;
t = force_fit_type (type, temp, -1,
(temp.is_negative ()
&& (TYPE_UNSIGNED (type)
< TYPE_UNSIGNED (TREE_TYPE (arg1))))
| TREE_OVERFLOW (arg1));
return t;
}
static tree
fold_convert_const_real_from_real (tree type, const_tree arg1)
{
REAL_VALUE_TYPE value;
tree t;
if (HONOR_SNANS (TYPE_MODE (TREE_TYPE (arg1)))
&& REAL_VALUE_ISSIGNALING_NAN (TREE_REAL_CST (arg1)))
return NULL_TREE; 
real_convert (&value, TYPE_MODE (type), &TREE_REAL_CST (arg1));
t = build_real (type, value);
if (REAL_VALUE_ISINF (TREE_REAL_CST (arg1))
&& !MODE_HAS_INFINITIES (TYPE_MODE (type)))
TREE_OVERFLOW (t) = 1;
else if (REAL_VALUE_ISNAN (TREE_REAL_CST (arg1))
&& !MODE_HAS_NANS (TYPE_MODE (type)))
TREE_OVERFLOW (t) = 1;
else if (!MODE_HAS_INFINITIES (TYPE_MODE (type))
&& REAL_VALUE_ISINF (value)
&& !REAL_VALUE_ISINF (TREE_REAL_CST (arg1)))
TREE_OVERFLOW (t) = 1;
else
TREE_OVERFLOW (t) = TREE_OVERFLOW (arg1);
return t;
}
static tree
fold_convert_const_real_from_fixed (tree type, const_tree arg1)
{
REAL_VALUE_TYPE value;
tree t;
real_convert_from_fixed (&value, SCALAR_FLOAT_TYPE_MODE (type),
&TREE_FIXED_CST (arg1));
t = build_real (type, value);
TREE_OVERFLOW (t) = TREE_OVERFLOW (arg1);
return t;
}
static tree
fold_convert_const_fixed_from_fixed (tree type, const_tree arg1)
{
FIXED_VALUE_TYPE value;
tree t;
bool overflow_p;
overflow_p = fixed_convert (&value, SCALAR_TYPE_MODE (type),
&TREE_FIXED_CST (arg1), TYPE_SATURATING (type));
t = build_fixed (type, value);
if (overflow_p | TREE_OVERFLOW (arg1))
TREE_OVERFLOW (t) = 1;
return t;
}
static tree
fold_convert_const_fixed_from_int (tree type, const_tree arg1)
{
FIXED_VALUE_TYPE value;
tree t;
bool overflow_p;
double_int di;
gcc_assert (TREE_INT_CST_NUNITS (arg1) <= 2);
di.low = TREE_INT_CST_ELT (arg1, 0);
if (TREE_INT_CST_NUNITS (arg1) == 1)
di.high = (HOST_WIDE_INT) di.low < 0 ? HOST_WIDE_INT_M1 : 0;
else
di.high = TREE_INT_CST_ELT (arg1, 1);
overflow_p = fixed_convert_from_int (&value, SCALAR_TYPE_MODE (type), di,
TYPE_UNSIGNED (TREE_TYPE (arg1)),
TYPE_SATURATING (type));
t = build_fixed (type, value);
if (overflow_p | TREE_OVERFLOW (arg1))
TREE_OVERFLOW (t) = 1;
return t;
}
static tree
fold_convert_const_fixed_from_real (tree type, const_tree arg1)
{
FIXED_VALUE_TYPE value;
tree t;
bool overflow_p;
overflow_p = fixed_convert_from_real (&value, SCALAR_TYPE_MODE (type),
&TREE_REAL_CST (arg1),
TYPE_SATURATING (type));
t = build_fixed (type, value);
if (overflow_p | TREE_OVERFLOW (arg1))
TREE_OVERFLOW (t) = 1;
return t;
}
static tree
fold_convert_const (enum tree_code code, tree type, tree arg1)
{
tree arg_type = TREE_TYPE (arg1);
if (arg_type == type)
return arg1;
if (POLY_INT_CST_P (arg1)
&& (POINTER_TYPE_P (type) || INTEGRAL_TYPE_P (type))
&& TYPE_PRECISION (type) <= TYPE_PRECISION (arg_type))
return build_poly_int_cst (type,
poly_wide_int::from (poly_int_cst_value (arg1),
TYPE_PRECISION (type),
TYPE_SIGN (arg_type)));
if (POINTER_TYPE_P (type) || INTEGRAL_TYPE_P (type)
|| TREE_CODE (type) == OFFSET_TYPE)
{
if (TREE_CODE (arg1) == INTEGER_CST)
return fold_convert_const_int_from_int (type, arg1);
else if (TREE_CODE (arg1) == REAL_CST)
return fold_convert_const_int_from_real (code, type, arg1);
else if (TREE_CODE (arg1) == FIXED_CST)
return fold_convert_const_int_from_fixed (type, arg1);
}
else if (TREE_CODE (type) == REAL_TYPE)
{
if (TREE_CODE (arg1) == INTEGER_CST)
return build_real_from_int_cst (type, arg1);
else if (TREE_CODE (arg1) == REAL_CST)
return fold_convert_const_real_from_real (type, arg1);
else if (TREE_CODE (arg1) == FIXED_CST)
return fold_convert_const_real_from_fixed (type, arg1);
}
else if (TREE_CODE (type) == FIXED_POINT_TYPE)
{
if (TREE_CODE (arg1) == FIXED_CST)
return fold_convert_const_fixed_from_fixed (type, arg1);
else if (TREE_CODE (arg1) == INTEGER_CST)
return fold_convert_const_fixed_from_int (type, arg1);
else if (TREE_CODE (arg1) == REAL_CST)
return fold_convert_const_fixed_from_real (type, arg1);
}
else if (TREE_CODE (type) == VECTOR_TYPE)
{
if (TREE_CODE (arg1) == VECTOR_CST
&& known_eq (TYPE_VECTOR_SUBPARTS (type), VECTOR_CST_NELTS (arg1)))
{
tree elttype = TREE_TYPE (type);
tree arg1_elttype = TREE_TYPE (TREE_TYPE (arg1));
bool step_ok_p
= (INTEGRAL_TYPE_P (elttype)
&& INTEGRAL_TYPE_P (arg1_elttype)
&& TYPE_PRECISION (elttype) <= TYPE_PRECISION (arg1_elttype));
tree_vector_builder v;
if (!v.new_unary_operation (type, arg1, step_ok_p))
return NULL_TREE;
unsigned int len = v.encoded_nelts ();
for (unsigned int i = 0; i < len; ++i)
{
tree elt = VECTOR_CST_ELT (arg1, i);
tree cvt = fold_convert_const (code, elttype, elt);
if (cvt == NULL_TREE)
return NULL_TREE;
v.quick_push (cvt);
}
return v.build ();
}
}
return NULL_TREE;
}
static tree
build_zero_vector (tree type)
{
tree t;
t = fold_convert_const (NOP_EXPR, TREE_TYPE (type), integer_zero_node);
return build_vector_from_val (type, t);
}
bool
fold_convertible_p (const_tree type, const_tree arg)
{
tree orig = TREE_TYPE (arg);
if (type == orig)
return true;
if (TREE_CODE (arg) == ERROR_MARK
|| TREE_CODE (type) == ERROR_MARK
|| TREE_CODE (orig) == ERROR_MARK)
return false;
if (TYPE_MAIN_VARIANT (type) == TYPE_MAIN_VARIANT (orig))
return true;
switch (TREE_CODE (type))
{
case INTEGER_TYPE: case ENUMERAL_TYPE: case BOOLEAN_TYPE:
case POINTER_TYPE: case REFERENCE_TYPE:
case OFFSET_TYPE:
return (INTEGRAL_TYPE_P (orig) || POINTER_TYPE_P (orig)
|| TREE_CODE (orig) == OFFSET_TYPE);
case REAL_TYPE:
case FIXED_POINT_TYPE:
case VECTOR_TYPE:
case VOID_TYPE:
return TREE_CODE (type) == TREE_CODE (orig);
default:
return false;
}
}
tree
fold_convert_loc (location_t loc, tree type, tree arg)
{
tree orig = TREE_TYPE (arg);
tree tem;
if (type == orig)
return arg;
if (TREE_CODE (arg) == ERROR_MARK
|| TREE_CODE (type) == ERROR_MARK
|| TREE_CODE (orig) == ERROR_MARK)
return error_mark_node;
switch (TREE_CODE (type))
{
case POINTER_TYPE:
case REFERENCE_TYPE:
if (POINTER_TYPE_P (orig)
&& (TYPE_ADDR_SPACE (TREE_TYPE (type))
!= TYPE_ADDR_SPACE (TREE_TYPE (orig))))
return fold_build1_loc (loc, ADDR_SPACE_CONVERT_EXPR, type, arg);
case INTEGER_TYPE: case ENUMERAL_TYPE: case BOOLEAN_TYPE:
case OFFSET_TYPE:
if (TREE_CODE (arg) == INTEGER_CST)
{
tem = fold_convert_const (NOP_EXPR, type, arg);
if (tem != NULL_TREE)
return tem;
}
if (INTEGRAL_TYPE_P (orig) || POINTER_TYPE_P (orig)
|| TREE_CODE (orig) == OFFSET_TYPE)
return fold_build1_loc (loc, NOP_EXPR, type, arg);
if (TREE_CODE (orig) == COMPLEX_TYPE)
return fold_convert_loc (loc, type,
fold_build1_loc (loc, REALPART_EXPR,
TREE_TYPE (orig), arg));
gcc_assert (TREE_CODE (orig) == VECTOR_TYPE
&& tree_int_cst_equal (TYPE_SIZE (type), TYPE_SIZE (orig)));
return fold_build1_loc (loc, VIEW_CONVERT_EXPR, type, arg);
case REAL_TYPE:
if (TREE_CODE (arg) == INTEGER_CST)
{
tem = fold_convert_const (FLOAT_EXPR, type, arg);
if (tem != NULL_TREE)
return tem;
}
else if (TREE_CODE (arg) == REAL_CST)
{
tem = fold_convert_const (NOP_EXPR, type, arg);
if (tem != NULL_TREE)
return tem;
}
else if (TREE_CODE (arg) == FIXED_CST)
{
tem = fold_convert_const (FIXED_CONVERT_EXPR, type, arg);
if (tem != NULL_TREE)
return tem;
}
switch (TREE_CODE (orig))
{
case INTEGER_TYPE:
case BOOLEAN_TYPE: case ENUMERAL_TYPE:
case POINTER_TYPE: case REFERENCE_TYPE:
return fold_build1_loc (loc, FLOAT_EXPR, type, arg);
case REAL_TYPE:
return fold_build1_loc (loc, NOP_EXPR, type, arg);
case FIXED_POINT_TYPE:
return fold_build1_loc (loc, FIXED_CONVERT_EXPR, type, arg);
case COMPLEX_TYPE:
tem = fold_build1_loc (loc, REALPART_EXPR, TREE_TYPE (orig), arg);
return fold_convert_loc (loc, type, tem);
default:
gcc_unreachable ();
}
case FIXED_POINT_TYPE:
if (TREE_CODE (arg) == FIXED_CST || TREE_CODE (arg) == INTEGER_CST
|| TREE_CODE (arg) == REAL_CST)
{
tem = fold_convert_const (FIXED_CONVERT_EXPR, type, arg);
if (tem != NULL_TREE)
goto fold_convert_exit;
}
switch (TREE_CODE (orig))
{
case FIXED_POINT_TYPE:
case INTEGER_TYPE:
case ENUMERAL_TYPE:
case BOOLEAN_TYPE:
case REAL_TYPE:
return fold_build1_loc (loc, FIXED_CONVERT_EXPR, type, arg);
case COMPLEX_TYPE:
tem = fold_build1_loc (loc, REALPART_EXPR, TREE_TYPE (orig), arg);
return fold_convert_loc (loc, type, tem);
default:
gcc_unreachable ();
}
case COMPLEX_TYPE:
switch (TREE_CODE (orig))
{
case INTEGER_TYPE:
case BOOLEAN_TYPE: case ENUMERAL_TYPE:
case POINTER_TYPE: case REFERENCE_TYPE:
case REAL_TYPE:
case FIXED_POINT_TYPE:
return fold_build2_loc (loc, COMPLEX_EXPR, type,
fold_convert_loc (loc, TREE_TYPE (type), arg),
fold_convert_loc (loc, TREE_TYPE (type),
integer_zero_node));
case COMPLEX_TYPE:
{
tree rpart, ipart;
if (TREE_CODE (arg) == COMPLEX_EXPR)
{
rpart = fold_convert_loc (loc, TREE_TYPE (type),
TREE_OPERAND (arg, 0));
ipart = fold_convert_loc (loc, TREE_TYPE (type),
TREE_OPERAND (arg, 1));
return fold_build2_loc (loc, COMPLEX_EXPR, type, rpart, ipart);
}
arg = save_expr (arg);
rpart = fold_build1_loc (loc, REALPART_EXPR, TREE_TYPE (orig), arg);
ipart = fold_build1_loc (loc, IMAGPART_EXPR, TREE_TYPE (orig), arg);
rpart = fold_convert_loc (loc, TREE_TYPE (type), rpart);
ipart = fold_convert_loc (loc, TREE_TYPE (type), ipart);
return fold_build2_loc (loc, COMPLEX_EXPR, type, rpart, ipart);
}
default:
gcc_unreachable ();
}
case VECTOR_TYPE:
if (integer_zerop (arg))
return build_zero_vector (type);
gcc_assert (tree_int_cst_equal (TYPE_SIZE (type), TYPE_SIZE (orig)));
gcc_assert (INTEGRAL_TYPE_P (orig) || POINTER_TYPE_P (orig)
|| TREE_CODE (orig) == VECTOR_TYPE);
return fold_build1_loc (loc, VIEW_CONVERT_EXPR, type, arg);
case VOID_TYPE:
tem = fold_ignored_result (arg);
return fold_build1_loc (loc, NOP_EXPR, type, tem);
default:
if (TYPE_MAIN_VARIANT (type) == TYPE_MAIN_VARIANT (orig))
return fold_build1_loc (loc, NOP_EXPR, type, arg);
gcc_unreachable ();
}
fold_convert_exit:
protected_set_expr_location_unshare (tem, loc);
return tem;
}

static bool
maybe_lvalue_p (const_tree x)
{
switch (TREE_CODE (x))
{
case VAR_DECL:
case PARM_DECL:
case RESULT_DECL:
case LABEL_DECL:
case FUNCTION_DECL:
case SSA_NAME:
case COMPONENT_REF:
case MEM_REF:
case INDIRECT_REF:
case ARRAY_REF:
case ARRAY_RANGE_REF:
case BIT_FIELD_REF:
case OBJ_TYPE_REF:
case REALPART_EXPR:
case IMAGPART_EXPR:
case PREINCREMENT_EXPR:
case PREDECREMENT_EXPR:
case SAVE_EXPR:
case TRY_CATCH_EXPR:
case WITH_CLEANUP_EXPR:
case COMPOUND_EXPR:
case MODIFY_EXPR:
case TARGET_EXPR:
case COND_EXPR:
case BIND_EXPR:
break;
default:
if ((int)TREE_CODE (x) >= NUM_TREE_CODES)
break;
return false;
}
return true;
}
tree
non_lvalue_loc (location_t loc, tree x)
{
if (in_gimple_form)
return x;
if (! maybe_lvalue_p (x))
return x;
return build1_loc (loc, NON_LVALUE_EXPR, TREE_TYPE (x), x);
}
static tree
pedantic_non_lvalue_loc (location_t loc, tree x)
{
return protected_set_expr_location_unshare (x, loc);
}

enum tree_code
invert_tree_comparison (enum tree_code code, bool honor_nans)
{
if (honor_nans && flag_trapping_math && code != EQ_EXPR && code != NE_EXPR
&& code != ORDERED_EXPR && code != UNORDERED_EXPR)
return ERROR_MARK;
switch (code)
{
case EQ_EXPR:
return NE_EXPR;
case NE_EXPR:
return EQ_EXPR;
case GT_EXPR:
return honor_nans ? UNLE_EXPR : LE_EXPR;
case GE_EXPR:
return honor_nans ? UNLT_EXPR : LT_EXPR;
case LT_EXPR:
return honor_nans ? UNGE_EXPR : GE_EXPR;
case LE_EXPR:
return honor_nans ? UNGT_EXPR : GT_EXPR;
case LTGT_EXPR:
return UNEQ_EXPR;
case UNEQ_EXPR:
return LTGT_EXPR;
case UNGT_EXPR:
return LE_EXPR;
case UNGE_EXPR:
return LT_EXPR;
case UNLT_EXPR:
return GE_EXPR;
case UNLE_EXPR:
return GT_EXPR;
case ORDERED_EXPR:
return UNORDERED_EXPR;
case UNORDERED_EXPR:
return ORDERED_EXPR;
default:
gcc_unreachable ();
}
}
enum tree_code
swap_tree_comparison (enum tree_code code)
{
switch (code)
{
case EQ_EXPR:
case NE_EXPR:
case ORDERED_EXPR:
case UNORDERED_EXPR:
case LTGT_EXPR:
case UNEQ_EXPR:
return code;
case GT_EXPR:
return LT_EXPR;
case GE_EXPR:
return LE_EXPR;
case LT_EXPR:
return GT_EXPR;
case LE_EXPR:
return GE_EXPR;
case UNGT_EXPR:
return UNLT_EXPR;
case UNGE_EXPR:
return UNLE_EXPR;
case UNLT_EXPR:
return UNGT_EXPR;
case UNLE_EXPR:
return UNGE_EXPR;
default:
gcc_unreachable ();
}
}
static enum comparison_code
comparison_to_compcode (enum tree_code code)
{
switch (code)
{
case LT_EXPR:
return COMPCODE_LT;
case EQ_EXPR:
return COMPCODE_EQ;
case LE_EXPR:
return COMPCODE_LE;
case GT_EXPR:
return COMPCODE_GT;
case NE_EXPR:
return COMPCODE_NE;
case GE_EXPR:
return COMPCODE_GE;
case ORDERED_EXPR:
return COMPCODE_ORD;
case UNORDERED_EXPR:
return COMPCODE_UNORD;
case UNLT_EXPR:
return COMPCODE_UNLT;
case UNEQ_EXPR:
return COMPCODE_UNEQ;
case UNLE_EXPR:
return COMPCODE_UNLE;
case UNGT_EXPR:
return COMPCODE_UNGT;
case LTGT_EXPR:
return COMPCODE_LTGT;
case UNGE_EXPR:
return COMPCODE_UNGE;
default:
gcc_unreachable ();
}
}
static enum tree_code
compcode_to_comparison (enum comparison_code code)
{
switch (code)
{
case COMPCODE_LT:
return LT_EXPR;
case COMPCODE_EQ:
return EQ_EXPR;
case COMPCODE_LE:
return LE_EXPR;
case COMPCODE_GT:
return GT_EXPR;
case COMPCODE_NE:
return NE_EXPR;
case COMPCODE_GE:
return GE_EXPR;
case COMPCODE_ORD:
return ORDERED_EXPR;
case COMPCODE_UNORD:
return UNORDERED_EXPR;
case COMPCODE_UNLT:
return UNLT_EXPR;
case COMPCODE_UNEQ:
return UNEQ_EXPR;
case COMPCODE_UNLE:
return UNLE_EXPR;
case COMPCODE_UNGT:
return UNGT_EXPR;
case COMPCODE_LTGT:
return LTGT_EXPR;
case COMPCODE_UNGE:
return UNGE_EXPR;
default:
gcc_unreachable ();
}
}
tree
combine_comparisons (location_t loc,
enum tree_code code, enum tree_code lcode,
enum tree_code rcode, tree truth_type,
tree ll_arg, tree lr_arg)
{
bool honor_nans = HONOR_NANS (ll_arg);
enum comparison_code lcompcode = comparison_to_compcode (lcode);
enum comparison_code rcompcode = comparison_to_compcode (rcode);
int compcode;
switch (code)
{
case TRUTH_AND_EXPR: case TRUTH_ANDIF_EXPR:
compcode = lcompcode & rcompcode;
break;
case TRUTH_OR_EXPR: case TRUTH_ORIF_EXPR:
compcode = lcompcode | rcompcode;
break;
default:
return NULL_TREE;
}
if (!honor_nans)
{
compcode &= ~COMPCODE_UNORD;
if (compcode == COMPCODE_LTGT)
compcode = COMPCODE_NE;
else if (compcode == COMPCODE_ORD)
compcode = COMPCODE_TRUE;
}
else if (flag_trapping_math)
{
bool ltrap = (lcompcode & COMPCODE_UNORD) == 0
&& (lcompcode != COMPCODE_EQ)
&& (lcompcode != COMPCODE_ORD);
bool rtrap = (rcompcode & COMPCODE_UNORD) == 0
&& (rcompcode != COMPCODE_EQ)
&& (rcompcode != COMPCODE_ORD);
bool trap = (compcode & COMPCODE_UNORD) == 0
&& (compcode != COMPCODE_EQ)
&& (compcode != COMPCODE_ORD);
if ((code == TRUTH_ORIF_EXPR && (lcompcode & COMPCODE_UNORD))
|| (code == TRUTH_ANDIF_EXPR && !(lcompcode & COMPCODE_UNORD)))
rtrap = false;
if (rtrap && !ltrap
&& (code == TRUTH_ANDIF_EXPR || code == TRUTH_ORIF_EXPR))
return NULL_TREE;
if ((ltrap || rtrap) != trap)
return NULL_TREE;
}
if (compcode == COMPCODE_TRUE)
return constant_boolean_node (true, truth_type);
else if (compcode == COMPCODE_FALSE)
return constant_boolean_node (false, truth_type);
else
{
enum tree_code tcode;
tcode = compcode_to_comparison ((enum comparison_code) compcode);
return fold_build2_loc (loc, tcode, truth_type, ll_arg, lr_arg);
}
}

int
operand_equal_p (const_tree arg0, const_tree arg1, unsigned int flags)
{
if (flag_checking && !(flags & OEP_NO_HASH_CHECK))
{
if (operand_equal_p (arg0, arg1, flags | OEP_NO_HASH_CHECK))
{
if (arg0 != arg1)
{
inchash::hash hstate0 (0), hstate1 (0);
inchash::add_expr (arg0, hstate0, flags | OEP_HASH_CHECK);
inchash::add_expr (arg1, hstate1, flags | OEP_HASH_CHECK);
hashval_t h0 = hstate0.end ();
hashval_t h1 = hstate1.end ();
gcc_assert (h0 == h1);
}
return 1;
}
else
return 0;
}
if (TREE_CODE (arg0) == ERROR_MARK || TREE_CODE (arg1) == ERROR_MARK
|| TREE_TYPE (arg0) == error_mark_node
|| TREE_TYPE (arg1) == error_mark_node)
return 0;
if (!TREE_TYPE (arg0) || !TREE_TYPE (arg1))
return 0;
if (POINTER_TYPE_P (TREE_TYPE (arg0))
&& POINTER_TYPE_P (TREE_TYPE (arg1))
&& (TYPE_ADDR_SPACE (TREE_TYPE (TREE_TYPE (arg0)))
!= TYPE_ADDR_SPACE (TREE_TYPE (TREE_TYPE (arg1)))))
return 0;
if (TREE_CODE (arg0) == INTEGER_CST && TREE_CODE (arg1) == INTEGER_CST)
{
gcc_checking_assert (!(flags & OEP_ADDRESS_OF));
return tree_int_cst_equal (arg0, arg1);
}
if (!(flags & OEP_ADDRESS_OF))
{
if (TYPE_UNSIGNED (TREE_TYPE (arg0)) != TYPE_UNSIGNED (TREE_TYPE (arg1))
|| POINTER_TYPE_P (TREE_TYPE (arg0))
!= POINTER_TYPE_P (TREE_TYPE (arg1)))
return 0;
if (element_precision (TREE_TYPE (arg0))
!= element_precision (TREE_TYPE (arg1)))
return 0;
STRIP_NOPS (arg0);
STRIP_NOPS (arg1);
}
#if 0
else
gcc_checking_assert (!CONVERT_EXPR_P (arg0) && !CONVERT_EXPR_P (arg1)
&& TREE_CODE (arg0) != SSA_NAME);
#endif
if (TREE_CODE (arg0) != TREE_CODE (arg1)
&& COMPARISON_CLASS_P (arg0)
&& COMPARISON_CLASS_P (arg1))
{
enum tree_code swap_code = swap_tree_comparison (TREE_CODE (arg1));
if (TREE_CODE (arg0) == swap_code)
return operand_equal_p (TREE_OPERAND (arg0, 0),
TREE_OPERAND (arg1, 1), flags)
&& operand_equal_p (TREE_OPERAND (arg0, 1),
TREE_OPERAND (arg1, 0), flags);
}
if (TREE_CODE (arg0) != TREE_CODE (arg1))
{
if (CONVERT_EXPR_P (arg0) && CONVERT_EXPR_P (arg1))
;
else if (flags & OEP_ADDRESS_OF)
{
if (TREE_CODE (arg0) == MEM_REF
&& DECL_P (arg1)
&& TREE_CODE (TREE_OPERAND (arg0, 0)) == ADDR_EXPR
&& TREE_OPERAND (TREE_OPERAND (arg0, 0), 0) == arg1
&& integer_zerop (TREE_OPERAND (arg0, 1)))
return 1;
else if (TREE_CODE (arg1) == MEM_REF
&& DECL_P (arg0)
&& TREE_CODE (TREE_OPERAND (arg1, 0)) == ADDR_EXPR
&& TREE_OPERAND (TREE_OPERAND (arg1, 0), 0) == arg0
&& integer_zerop (TREE_OPERAND (arg1, 1)))
return 1;
return 0;
}
else
return 0;
}
if (TREE_CODE (TREE_TYPE (arg0)) == ERROR_MARK
|| TREE_CODE (TREE_TYPE (arg1)) == ERROR_MARK
|| (TYPE_MODE (TREE_TYPE (arg0)) != TYPE_MODE (TREE_TYPE (arg1))
&& !(flags & OEP_ADDRESS_OF)))
return 0;
if (arg0 == arg1 && ! (flags & OEP_ONLY_CONST)
&& (TREE_CODE (arg0) == SAVE_EXPR
|| (flags & OEP_MATCH_SIDE_EFFECTS)
|| (! TREE_SIDE_EFFECTS (arg0) && ! TREE_SIDE_EFFECTS (arg1))))
return 1;
if (TREE_CONSTANT (arg0) && TREE_CONSTANT (arg1))
switch (TREE_CODE (arg0))
{
case INTEGER_CST:
return tree_int_cst_equal (arg0, arg1);
case FIXED_CST:
return FIXED_VALUES_IDENTICAL (TREE_FIXED_CST (arg0),
TREE_FIXED_CST (arg1));
case REAL_CST:
if (real_identical (&TREE_REAL_CST (arg0), &TREE_REAL_CST (arg1)))
return 1;
if (!HONOR_SIGNED_ZEROS (arg0))
{
if (real_zerop (arg0) && real_zerop (arg1))
return 1;
}
return 0;
case VECTOR_CST:
{
if (VECTOR_CST_LOG2_NPATTERNS (arg0)
!= VECTOR_CST_LOG2_NPATTERNS (arg1))
return 0;
if (VECTOR_CST_NELTS_PER_PATTERN (arg0)
!= VECTOR_CST_NELTS_PER_PATTERN (arg1))
return 0;
unsigned int count = vector_cst_encoded_nelts (arg0);
for (unsigned int i = 0; i < count; ++i)
if (!operand_equal_p (VECTOR_CST_ENCODED_ELT (arg0, i),
VECTOR_CST_ENCODED_ELT (arg1, i), flags))
return 0;
return 1;
}
case COMPLEX_CST:
return (operand_equal_p (TREE_REALPART (arg0), TREE_REALPART (arg1),
flags)
&& operand_equal_p (TREE_IMAGPART (arg0), TREE_IMAGPART (arg1),
flags));
case STRING_CST:
return (TREE_STRING_LENGTH (arg0) == TREE_STRING_LENGTH (arg1)
&& ! memcmp (TREE_STRING_POINTER (arg0),
TREE_STRING_POINTER (arg1),
TREE_STRING_LENGTH (arg0)));
case ADDR_EXPR:
gcc_checking_assert (!(flags & OEP_ADDRESS_OF));
return operand_equal_p (TREE_OPERAND (arg0, 0), TREE_OPERAND (arg1, 0),
flags | OEP_ADDRESS_OF
| OEP_MATCH_SIDE_EFFECTS);
case CONSTRUCTOR:
return !CONSTRUCTOR_NELTS (arg0) && !CONSTRUCTOR_NELTS (arg1);
default:
break;
}
if (flags & OEP_ONLY_CONST)
return 0;
#define OP_SAME(N) operand_equal_p (TREE_OPERAND (arg0, N),	\
TREE_OPERAND (arg1, N), flags)
#define OP_SAME_WITH_NULL(N)				\
((!TREE_OPERAND (arg0, N) || !TREE_OPERAND (arg1, N))	\
? TREE_OPERAND (arg0, N) == TREE_OPERAND (arg1, N) : OP_SAME (N))
switch (TREE_CODE_CLASS (TREE_CODE (arg0)))
{
case tcc_unary:
switch (TREE_CODE (arg0))
{
CASE_CONVERT:
case FIX_TRUNC_EXPR:
if (TYPE_UNSIGNED (TREE_TYPE (arg0))
!= TYPE_UNSIGNED (TREE_TYPE (arg1)))
return 0;
break;
default:
break;
}
return OP_SAME (0);
case tcc_comparison:
case tcc_binary:
if (OP_SAME (0) && OP_SAME (1))
return 1;
return (commutative_tree_code (TREE_CODE (arg0))
&& operand_equal_p (TREE_OPERAND (arg0, 0),
TREE_OPERAND (arg1, 1), flags)
&& operand_equal_p (TREE_OPERAND (arg0, 1),
TREE_OPERAND (arg1, 0), flags));
case tcc_reference:
if ((flags & OEP_MATCH_SIDE_EFFECTS) == 0
&& (TREE_SIDE_EFFECTS (arg0)
|| TREE_SIDE_EFFECTS (arg1)))
return 0;
switch (TREE_CODE (arg0))
{
case INDIRECT_REF:
if (!(flags & OEP_ADDRESS_OF)
&& (TYPE_ALIGN (TREE_TYPE (arg0))
!= TYPE_ALIGN (TREE_TYPE (arg1))))
return 0;
flags &= ~OEP_ADDRESS_OF;
return OP_SAME (0);
case IMAGPART_EXPR:
if (!operand_equal_p (TYPE_SIZE (TREE_TYPE (arg0)),
TYPE_SIZE (TREE_TYPE (arg1)),
flags & ~OEP_ADDRESS_OF))
return 0;
case REALPART_EXPR:
case VIEW_CONVERT_EXPR:
return OP_SAME (0);
case TARGET_MEM_REF:
case MEM_REF:
if (!(flags & OEP_ADDRESS_OF))
{
if (TYPE_SIZE (TREE_TYPE (arg0)) != TYPE_SIZE (TREE_TYPE (arg1))
&& (!TYPE_SIZE (TREE_TYPE (arg0))
|| !TYPE_SIZE (TREE_TYPE (arg1))
|| !operand_equal_p (TYPE_SIZE (TREE_TYPE (arg0)),
TYPE_SIZE (TREE_TYPE (arg1)),
flags)))
return 0;
if (!types_compatible_p (TREE_TYPE (arg0), TREE_TYPE (arg1)))
return 0;
if (!alias_ptr_types_compatible_p
(TREE_TYPE (TREE_OPERAND (arg0, 1)),
TREE_TYPE (TREE_OPERAND (arg1, 1)))
|| (MR_DEPENDENCE_CLIQUE (arg0)
!= MR_DEPENDENCE_CLIQUE (arg1))
|| (MR_DEPENDENCE_BASE (arg0)
!= MR_DEPENDENCE_BASE (arg1)))
return 0;
if (TYPE_ALIGN (TREE_TYPE (arg0))
!= TYPE_ALIGN (TREE_TYPE (arg1)))
return 0;
}
flags &= ~OEP_ADDRESS_OF;
return (OP_SAME (0) && OP_SAME (1)
&& (TREE_CODE (arg0) != TARGET_MEM_REF
|| (OP_SAME_WITH_NULL (2)
&& OP_SAME_WITH_NULL (3)
&& OP_SAME_WITH_NULL (4))));
case ARRAY_REF:
case ARRAY_RANGE_REF:
if (!OP_SAME (0))
return 0;
flags &= ~OEP_ADDRESS_OF;
return ((tree_int_cst_equal (TREE_OPERAND (arg0, 1),
TREE_OPERAND (arg1, 1))
|| OP_SAME (1))
&& OP_SAME_WITH_NULL (2)
&& OP_SAME_WITH_NULL (3)
&& (TREE_TYPE (TREE_OPERAND (arg0, 0))
== TREE_TYPE (TREE_OPERAND (arg1, 0))
|| (operand_equal_p (array_ref_low_bound
(CONST_CAST_TREE (arg0)),
array_ref_low_bound
(CONST_CAST_TREE (arg1)), flags)
&& operand_equal_p (array_ref_element_size
(CONST_CAST_TREE (arg0)),
array_ref_element_size
(CONST_CAST_TREE (arg1)),
flags))));
case COMPONENT_REF:
if (!OP_SAME_WITH_NULL (0)
|| !OP_SAME (1))
return 0;
flags &= ~OEP_ADDRESS_OF;
return OP_SAME_WITH_NULL (2);
case BIT_FIELD_REF:
if (!OP_SAME (0))
return 0;
flags &= ~OEP_ADDRESS_OF;
return OP_SAME (1) && OP_SAME (2);
default:
return 0;
}
case tcc_expression:
switch (TREE_CODE (arg0))
{
case ADDR_EXPR:
gcc_checking_assert (!(flags & OEP_ADDRESS_OF));
return operand_equal_p (TREE_OPERAND (arg0, 0),
TREE_OPERAND (arg1, 0),
flags | OEP_ADDRESS_OF);
case TRUTH_NOT_EXPR:
return OP_SAME (0);
case TRUTH_ANDIF_EXPR:
case TRUTH_ORIF_EXPR:
return OP_SAME (0) && OP_SAME (1);
case FMA_EXPR:
case WIDEN_MULT_PLUS_EXPR:
case WIDEN_MULT_MINUS_EXPR:
if (!OP_SAME (2))
return 0;
case TRUTH_AND_EXPR:
case TRUTH_OR_EXPR:
case TRUTH_XOR_EXPR:
if (OP_SAME (0) && OP_SAME (1))
return 1;
return (operand_equal_p (TREE_OPERAND (arg0, 0),
TREE_OPERAND (arg1, 1), flags)
&& operand_equal_p (TREE_OPERAND (arg0, 1),
TREE_OPERAND (arg1, 0), flags));
case COND_EXPR:
if (! OP_SAME (1) || ! OP_SAME_WITH_NULL (2))
return 0;
flags &= ~OEP_ADDRESS_OF;
return OP_SAME (0);
case BIT_INSERT_EXPR:
if (TREE_CODE (TREE_OPERAND (arg0, 1)) == INTEGER_CST
&& TREE_CODE (TREE_OPERAND (arg1, 1)) == INTEGER_CST
&& TYPE_PRECISION (TREE_TYPE (TREE_OPERAND (arg0, 1)))
!= TYPE_PRECISION (TREE_TYPE (TREE_OPERAND (arg1, 1))))
return false;
case VEC_COND_EXPR:
case DOT_PROD_EXPR:
return OP_SAME (0) && OP_SAME (1) && OP_SAME (2);
case MODIFY_EXPR:
case INIT_EXPR:
case COMPOUND_EXPR:
case PREDECREMENT_EXPR:
case PREINCREMENT_EXPR:
case POSTDECREMENT_EXPR:
case POSTINCREMENT_EXPR:
if (flags & OEP_LEXICOGRAPHIC)
return OP_SAME (0) && OP_SAME (1);
return 0;
case CLEANUP_POINT_EXPR:
case EXPR_STMT:
if (flags & OEP_LEXICOGRAPHIC)
return OP_SAME (0);
return 0;
default:
return 0;
}
case tcc_vl_exp:
switch (TREE_CODE (arg0))
{
case CALL_EXPR:
if ((CALL_EXPR_FN (arg0) == NULL_TREE)
!= (CALL_EXPR_FN (arg1) == NULL_TREE))
return 0;
else if (CALL_EXPR_FN (arg0) == NULL_TREE)
{
if (CALL_EXPR_IFN (arg0) != CALL_EXPR_IFN (arg1))
return 0;
}
else
{
if (! operand_equal_p (CALL_EXPR_FN (arg0), CALL_EXPR_FN (arg1),
flags))
return 0;
}
{
unsigned int cef = call_expr_flags (arg0);
if (flags & OEP_PURE_SAME)
cef &= ECF_CONST | ECF_PURE;
else
cef &= ECF_CONST;
if (!cef && !(flags & OEP_LEXICOGRAPHIC))
return 0;
}
{
const_call_expr_arg_iterator iter0, iter1;
const_tree a0, a1;
for (a0 = first_const_call_expr_arg (arg0, &iter0),
a1 = first_const_call_expr_arg (arg1, &iter1);
a0 && a1;
a0 = next_const_call_expr_arg (&iter0),
a1 = next_const_call_expr_arg (&iter1))
if (! operand_equal_p (a0, a1, flags))
return 0;
return ! (a0 || a1);
}
default:
return 0;
}
case tcc_declaration:
return (TREE_CODE (arg0) == FUNCTION_DECL
&& DECL_BUILT_IN (arg0) && DECL_BUILT_IN (arg1)
&& DECL_BUILT_IN_CLASS (arg0) == DECL_BUILT_IN_CLASS (arg1)
&& DECL_FUNCTION_CODE (arg0) == DECL_FUNCTION_CODE (arg1));
case tcc_exceptional:
if (TREE_CODE (arg0) == CONSTRUCTOR)
{
if (!VECTOR_TYPE_P (TREE_TYPE (arg0))
|| !VECTOR_TYPE_P (TREE_TYPE (arg1)))
return 0;
if (maybe_ne (TYPE_VECTOR_SUBPARTS (TREE_TYPE (arg0)),
TYPE_VECTOR_SUBPARTS (TREE_TYPE (arg1))))
return 0;
vec<constructor_elt, va_gc> *v0 = CONSTRUCTOR_ELTS (arg0);
vec<constructor_elt, va_gc> *v1 = CONSTRUCTOR_ELTS (arg1);
unsigned int len = vec_safe_length (v0);
if (len != vec_safe_length (v1))
return 0;
for (unsigned int i = 0; i < len; i++)
{
constructor_elt *c0 = &(*v0)[i];
constructor_elt *c1 = &(*v1)[i];
if (!operand_equal_p (c0->value, c1->value, flags)
|| (c0->index
&& (TREE_CODE (c0->index) != INTEGER_CST 
|| !compare_tree_int (c0->index, i)))
|| (c1->index
&& (TREE_CODE (c1->index) != INTEGER_CST 
|| !compare_tree_int (c1->index, i))))
return 0;
}
return 1;
}
else if (TREE_CODE (arg0) == STATEMENT_LIST
&& (flags & OEP_LEXICOGRAPHIC))
{
tree_stmt_iterator tsi1, tsi2;
tree body1 = CONST_CAST_TREE (arg0);
tree body2 = CONST_CAST_TREE (arg1);
for (tsi1 = tsi_start (body1), tsi2 = tsi_start (body2); ;
tsi_next (&tsi1), tsi_next (&tsi2))
{
if (tsi_end_p (tsi1) ^ tsi_end_p (tsi2))
return 0;
if (tsi_end_p (tsi1) && tsi_end_p (tsi2))
return 1;
if (!operand_equal_p (tsi_stmt (tsi1), tsi_stmt (tsi2),
flags & (OEP_LEXICOGRAPHIC
| OEP_NO_HASH_CHECK)))
return 0;
}
}
return 0;
case tcc_statement:
switch (TREE_CODE (arg0))
{
case RETURN_EXPR:
if (flags & OEP_LEXICOGRAPHIC)
return OP_SAME_WITH_NULL (0);
return 0;
case DEBUG_BEGIN_STMT:
if (flags & OEP_LEXICOGRAPHIC)
return 1;
return 0;
default:
return 0;
}
default:
return 0;
}
#undef OP_SAME
#undef OP_SAME_WITH_NULL
}

static bool
operand_equal_for_comparison_p (tree arg0, tree arg1)
{
if (operand_equal_p (arg0, arg1, 0))
return true;
if (! INTEGRAL_TYPE_P (TREE_TYPE (arg0))
|| ! INTEGRAL_TYPE_P (TREE_TYPE (arg1)))
return false;
tree op0 = arg0;
tree op1 = arg1;
STRIP_NOPS (op0);
STRIP_NOPS (op1);
if (operand_equal_p (op0, op1, 0))
return true;
if (CONVERT_EXPR_P (arg1)
&& INTEGRAL_TYPE_P (TREE_TYPE (TREE_OPERAND (arg1, 0)))
&& TYPE_PRECISION (TREE_TYPE (TREE_OPERAND (arg1, 0)))
< TYPE_PRECISION (TREE_TYPE (arg1))
&& operand_equal_p (arg0, TREE_OPERAND (arg1, 0), 0))
return true;
return false;
}

static int
twoval_comparison_p (tree arg, tree *cval1, tree *cval2)
{
enum tree_code code = TREE_CODE (arg);
enum tree_code_class tclass = TREE_CODE_CLASS (code);
if (tclass == tcc_expression && code == TRUTH_NOT_EXPR)
tclass = tcc_unary;
else if (tclass == tcc_expression
&& (code == TRUTH_ANDIF_EXPR || code == TRUTH_ORIF_EXPR
|| code == COMPOUND_EXPR))
tclass = tcc_binary;
switch (tclass)
{
case tcc_unary:
return twoval_comparison_p (TREE_OPERAND (arg, 0), cval1, cval2);
case tcc_binary:
return (twoval_comparison_p (TREE_OPERAND (arg, 0), cval1, cval2)
&& twoval_comparison_p (TREE_OPERAND (arg, 1), cval1, cval2));
case tcc_constant:
return 1;
case tcc_expression:
if (code == COND_EXPR)
return (twoval_comparison_p (TREE_OPERAND (arg, 0), cval1, cval2)
&& twoval_comparison_p (TREE_OPERAND (arg, 1), cval1, cval2)
&& twoval_comparison_p (TREE_OPERAND (arg, 2), cval1, cval2));
return 0;
case tcc_comparison:
if (operand_equal_p (TREE_OPERAND (arg, 0),
TREE_OPERAND (arg, 1), 0))
return 0;
if (*cval1 == 0)
*cval1 = TREE_OPERAND (arg, 0);
else if (operand_equal_p (*cval1, TREE_OPERAND (arg, 0), 0))
;
else if (*cval2 == 0)
*cval2 = TREE_OPERAND (arg, 0);
else if (operand_equal_p (*cval2, TREE_OPERAND (arg, 0), 0))
;
else
return 0;
if (operand_equal_p (*cval1, TREE_OPERAND (arg, 1), 0))
;
else if (*cval2 == 0)
*cval2 = TREE_OPERAND (arg, 1);
else if (operand_equal_p (*cval2, TREE_OPERAND (arg, 1), 0))
;
else
return 0;
return 1;
default:
return 0;
}
}

static tree
eval_subst (location_t loc, tree arg, tree old0, tree new0,
tree old1, tree new1)
{
tree type = TREE_TYPE (arg);
enum tree_code code = TREE_CODE (arg);
enum tree_code_class tclass = TREE_CODE_CLASS (code);
if (tclass == tcc_expression && code == TRUTH_NOT_EXPR)
tclass = tcc_unary;
else if (tclass == tcc_expression
&& (code == TRUTH_ANDIF_EXPR || code == TRUTH_ORIF_EXPR))
tclass = tcc_binary;
switch (tclass)
{
case tcc_unary:
return fold_build1_loc (loc, code, type,
eval_subst (loc, TREE_OPERAND (arg, 0),
old0, new0, old1, new1));
case tcc_binary:
return fold_build2_loc (loc, code, type,
eval_subst (loc, TREE_OPERAND (arg, 0),
old0, new0, old1, new1),
eval_subst (loc, TREE_OPERAND (arg, 1),
old0, new0, old1, new1));
case tcc_expression:
switch (code)
{
case SAVE_EXPR:
return eval_subst (loc, TREE_OPERAND (arg, 0), old0, new0,
old1, new1);
case COMPOUND_EXPR:
return eval_subst (loc, TREE_OPERAND (arg, 1), old0, new0,
old1, new1);
case COND_EXPR:
return fold_build3_loc (loc, code, type,
eval_subst (loc, TREE_OPERAND (arg, 0),
old0, new0, old1, new1),
eval_subst (loc, TREE_OPERAND (arg, 1),
old0, new0, old1, new1),
eval_subst (loc, TREE_OPERAND (arg, 2),
old0, new0, old1, new1));
default:
break;
}
case tcc_comparison:
{
tree arg0 = TREE_OPERAND (arg, 0);
tree arg1 = TREE_OPERAND (arg, 1);
if (arg0 == old0 || operand_equal_p (arg0, old0, 0))
arg0 = new0;
else if (arg0 == old1 || operand_equal_p (arg0, old1, 0))
arg0 = new1;
if (arg1 == old0 || operand_equal_p (arg1, old0, 0))
arg1 = new0;
else if (arg1 == old1 || operand_equal_p (arg1, old1, 0))
arg1 = new1;
return fold_build2_loc (loc, code, type, arg0, arg1);
}
default:
return arg;
}
}

tree
omit_one_operand_loc (location_t loc, tree type, tree result, tree omitted)
{
tree t = fold_convert_loc (loc, type, result);
if (IS_EMPTY_STMT (t) && TREE_SIDE_EFFECTS (omitted))
return build1_loc (loc, NOP_EXPR, void_type_node,
fold_ignored_result (omitted));
if (TREE_SIDE_EFFECTS (omitted))
return build2_loc (loc, COMPOUND_EXPR, type,
fold_ignored_result (omitted), t);
return non_lvalue_loc (loc, t);
}
tree
omit_two_operands_loc (location_t loc, tree type, tree result,
tree omitted1, tree omitted2)
{
tree t = fold_convert_loc (loc, type, result);
if (TREE_SIDE_EFFECTS (omitted2))
t = build2_loc (loc, COMPOUND_EXPR, type, omitted2, t);
if (TREE_SIDE_EFFECTS (omitted1))
t = build2_loc (loc, COMPOUND_EXPR, type, omitted1, t);
return TREE_CODE (t) != COMPOUND_EXPR ? non_lvalue_loc (loc, t) : t;
}

static tree
fold_truth_not_expr (location_t loc, tree arg)
{
tree type = TREE_TYPE (arg);
enum tree_code code = TREE_CODE (arg);
location_t loc1, loc2;
if (TREE_CODE_CLASS (code) == tcc_comparison)
{
tree op_type = TREE_TYPE (TREE_OPERAND (arg, 0));
if (FLOAT_TYPE_P (op_type)
&& flag_trapping_math
&& code != ORDERED_EXPR && code != UNORDERED_EXPR
&& code != NE_EXPR && code != EQ_EXPR)
return NULL_TREE;
code = invert_tree_comparison (code, HONOR_NANS (op_type));
if (code == ERROR_MARK)
return NULL_TREE;
tree ret = build2_loc (loc, code, type, TREE_OPERAND (arg, 0),
TREE_OPERAND (arg, 1));
if (TREE_NO_WARNING (arg))
TREE_NO_WARNING (ret) = 1;
return ret;
}
switch (code)
{
case INTEGER_CST:
return constant_boolean_node (integer_zerop (arg), type);
case TRUTH_AND_EXPR:
loc1 = expr_location_or (TREE_OPERAND (arg, 0), loc);
loc2 = expr_location_or (TREE_OPERAND (arg, 1), loc);
return build2_loc (loc, TRUTH_OR_EXPR, type,
invert_truthvalue_loc (loc1, TREE_OPERAND (arg, 0)),
invert_truthvalue_loc (loc2, TREE_OPERAND (arg, 1)));
case TRUTH_OR_EXPR:
loc1 = expr_location_or (TREE_OPERAND (arg, 0), loc);
loc2 = expr_location_or (TREE_OPERAND (arg, 1), loc);
return build2_loc (loc, TRUTH_AND_EXPR, type,
invert_truthvalue_loc (loc1, TREE_OPERAND (arg, 0)),
invert_truthvalue_loc (loc2, TREE_OPERAND (arg, 1)));
case TRUTH_XOR_EXPR:
if (TREE_CODE (TREE_OPERAND (arg, 1)) == TRUTH_NOT_EXPR)
return build2_loc (loc, TRUTH_XOR_EXPR, type, TREE_OPERAND (arg, 0),
TREE_OPERAND (TREE_OPERAND (arg, 1), 0));
else
return build2_loc (loc, TRUTH_XOR_EXPR, type,
invert_truthvalue_loc (loc, TREE_OPERAND (arg, 0)),
TREE_OPERAND (arg, 1));
case TRUTH_ANDIF_EXPR:
loc1 = expr_location_or (TREE_OPERAND (arg, 0), loc);
loc2 = expr_location_or (TREE_OPERAND (arg, 1), loc);
return build2_loc (loc, TRUTH_ORIF_EXPR, type,
invert_truthvalue_loc (loc1, TREE_OPERAND (arg, 0)),
invert_truthvalue_loc (loc2, TREE_OPERAND (arg, 1)));
case TRUTH_ORIF_EXPR:
loc1 = expr_location_or (TREE_OPERAND (arg, 0), loc);
loc2 = expr_location_or (TREE_OPERAND (arg, 1), loc);
return build2_loc (loc, TRUTH_ANDIF_EXPR, type,
invert_truthvalue_loc (loc1, TREE_OPERAND (arg, 0)),
invert_truthvalue_loc (loc2, TREE_OPERAND (arg, 1)));
case TRUTH_NOT_EXPR:
return TREE_OPERAND (arg, 0);
case COND_EXPR:
{
tree arg1 = TREE_OPERAND (arg, 1);
tree arg2 = TREE_OPERAND (arg, 2);
loc1 = expr_location_or (TREE_OPERAND (arg, 1), loc);
loc2 = expr_location_or (TREE_OPERAND (arg, 2), loc);
return build3_loc (loc, COND_EXPR, type, TREE_OPERAND (arg, 0),
VOID_TYPE_P (TREE_TYPE (arg1))
? arg1 : invert_truthvalue_loc (loc1, arg1),
VOID_TYPE_P (TREE_TYPE (arg2))
? arg2 : invert_truthvalue_loc (loc2, arg2));
}
case COMPOUND_EXPR:
loc1 = expr_location_or (TREE_OPERAND (arg, 1), loc);
return build2_loc (loc, COMPOUND_EXPR, type,
TREE_OPERAND (arg, 0),
invert_truthvalue_loc (loc1, TREE_OPERAND (arg, 1)));
case NON_LVALUE_EXPR:
loc1 = expr_location_or (TREE_OPERAND (arg, 0), loc);
return invert_truthvalue_loc (loc1, TREE_OPERAND (arg, 0));
CASE_CONVERT:
if (TREE_CODE (TREE_TYPE (arg)) == BOOLEAN_TYPE)
return build1_loc (loc, TRUTH_NOT_EXPR, type, arg);
case FLOAT_EXPR:
loc1 = expr_location_or (TREE_OPERAND (arg, 0), loc);
return build1_loc (loc, TREE_CODE (arg), type,
invert_truthvalue_loc (loc1, TREE_OPERAND (arg, 0)));
case BIT_AND_EXPR:
if (!integer_onep (TREE_OPERAND (arg, 1)))
return NULL_TREE;
return build2_loc (loc, EQ_EXPR, type, arg, build_int_cst (type, 0));
case SAVE_EXPR:
return build1_loc (loc, TRUTH_NOT_EXPR, type, arg);
case CLEANUP_POINT_EXPR:
loc1 = expr_location_or (TREE_OPERAND (arg, 0), loc);
return build1_loc (loc, CLEANUP_POINT_EXPR, type,
invert_truthvalue_loc (loc1, TREE_OPERAND (arg, 0)));
default:
return NULL_TREE;
}
}
static tree
fold_invert_truthvalue (location_t loc, tree arg)
{
tree type = TREE_TYPE (arg);
return fold_unary_loc (loc, VECTOR_TYPE_P (type)
? BIT_NOT_EXPR
: TRUTH_NOT_EXPR,
type, arg);
}
tree
invert_truthvalue_loc (location_t loc, tree arg)
{
if (TREE_CODE (arg) == ERROR_MARK)
return arg;
tree type = TREE_TYPE (arg);
return fold_build1_loc (loc, VECTOR_TYPE_P (type)
? BIT_NOT_EXPR
: TRUTH_NOT_EXPR,
type, arg);
}

static tree
make_bit_field_ref (location_t loc, tree inner, tree orig_inner, tree type,
HOST_WIDE_INT bitsize, poly_int64 bitpos,
int unsignedp, int reversep)
{
tree result, bftype;
if (TREE_CODE (orig_inner) == COMPONENT_REF)
{
tree ninner = TREE_OPERAND (orig_inner, 0);
machine_mode nmode;
poly_int64 nbitsize, nbitpos;
tree noffset;
int nunsignedp, nreversep, nvolatilep = 0;
tree base = get_inner_reference (ninner, &nbitsize, &nbitpos,
&noffset, &nmode, &nunsignedp,
&nreversep, &nvolatilep);
if (base == inner
&& noffset == NULL_TREE
&& known_subrange_p (bitpos, bitsize, nbitpos, nbitsize)
&& !reversep
&& !nreversep
&& !nvolatilep)
{
inner = ninner;
bitpos -= nbitpos;
}
}
alias_set_type iset = get_alias_set (orig_inner);
if (iset == 0 && get_alias_set (inner) != iset)
inner = fold_build2 (MEM_REF, TREE_TYPE (inner),
build_fold_addr_expr (inner),
build_int_cst (ptr_type_node, 0));
if (known_eq (bitpos, 0) && !reversep)
{
tree size = TYPE_SIZE (TREE_TYPE (inner));
if ((INTEGRAL_TYPE_P (TREE_TYPE (inner))
|| POINTER_TYPE_P (TREE_TYPE (inner)))
&& tree_fits_shwi_p (size)
&& tree_to_shwi (size) == bitsize)
return fold_convert_loc (loc, type, inner);
}
bftype = type;
if (TYPE_PRECISION (bftype) != bitsize
|| TYPE_UNSIGNED (bftype) == !unsignedp)
bftype = build_nonstandard_integer_type (bitsize, 0);
result = build3_loc (loc, BIT_FIELD_REF, bftype, inner,
bitsize_int (bitsize), bitsize_int (bitpos));
REF_REVERSE_STORAGE_ORDER (result) = reversep;
if (bftype != type)
result = fold_convert_loc (loc, type, result);
return result;
}
static tree
optimize_bit_field_compare (location_t loc, enum tree_code code,
tree compare_type, tree lhs, tree rhs)
{
poly_int64 plbitpos, plbitsize, rbitpos, rbitsize;
HOST_WIDE_INT lbitpos, lbitsize, nbitpos, nbitsize;
tree type = TREE_TYPE (lhs);
tree unsigned_type;
int const_p = TREE_CODE (rhs) == INTEGER_CST;
machine_mode lmode, rmode;
scalar_int_mode nmode;
int lunsignedp, runsignedp;
int lreversep, rreversep;
int lvolatilep = 0, rvolatilep = 0;
tree linner, rinner = NULL_TREE;
tree mask;
tree offset;
linner = get_inner_reference (lhs, &plbitsize, &plbitpos, &offset, &lmode,
&lunsignedp, &lreversep, &lvolatilep);
if (linner == lhs
|| !known_size_p (plbitsize)
|| !plbitsize.is_constant (&lbitsize)
|| !plbitpos.is_constant (&lbitpos)
|| known_eq (lbitsize, GET_MODE_BITSIZE (lmode))
|| offset != 0
|| TREE_CODE (linner) == PLACEHOLDER_EXPR
|| lvolatilep)
return 0;
if (const_p)
rreversep = lreversep;
else
{
rinner
= get_inner_reference (rhs, &rbitsize, &rbitpos, &offset, &rmode,
&runsignedp, &rreversep, &rvolatilep);
if (rinner == rhs
|| maybe_ne (lbitpos, rbitpos)
|| maybe_ne (lbitsize, rbitsize)
|| lunsignedp != runsignedp
|| lreversep != rreversep
|| offset != 0
|| TREE_CODE (rinner) == PLACEHOLDER_EXPR
|| rvolatilep)
return 0;
}
poly_uint64 bitstart = 0;
poly_uint64 bitend = 0;
if (TREE_CODE (lhs) == COMPONENT_REF)
{
get_bit_range (&bitstart, &bitend, lhs, &plbitpos, &offset);
if (!plbitpos.is_constant (&lbitpos) || offset != NULL_TREE)
return 0;
}
if (!get_best_mode (lbitsize, lbitpos, bitstart, bitend,
const_p ? TYPE_ALIGN (TREE_TYPE (linner))
: MIN (TYPE_ALIGN (TREE_TYPE (linner)),
TYPE_ALIGN (TREE_TYPE (rinner))),
BITS_PER_WORD, false, &nmode))
return 0;
unsigned_type = lang_hooks.types.type_for_mode (nmode, 1);
nbitsize = GET_MODE_BITSIZE (nmode);
nbitpos = lbitpos & ~ (nbitsize - 1);
lbitpos -= nbitpos;
if (nbitsize == lbitsize)
return 0;
if (lreversep ? !BYTES_BIG_ENDIAN : BYTES_BIG_ENDIAN)
lbitpos = nbitsize - lbitsize - lbitpos;
mask = build_int_cst_type (unsigned_type, -1);
mask = const_binop (LSHIFT_EXPR, mask, size_int (nbitsize - lbitsize));
mask = const_binop (RSHIFT_EXPR, mask,
size_int (nbitsize - lbitsize - lbitpos));
if (! const_p)
{
if (nbitpos < 0)
return 0;
tree t1 = make_bit_field_ref (loc, linner, lhs, unsigned_type,
nbitsize, nbitpos, 1, lreversep);
t1 = fold_build2_loc (loc, BIT_AND_EXPR, unsigned_type, t1, mask);
tree t2 = make_bit_field_ref (loc, rinner, rhs, unsigned_type,
nbitsize, nbitpos, 1, rreversep);
t2 = fold_build2_loc (loc, BIT_AND_EXPR, unsigned_type, t2, mask);
return fold_build2_loc (loc, code, compare_type, t1, t2);
}
if (lunsignedp)
{
if (wi::lrshift (wi::to_wide (rhs), lbitsize) != 0)
{
warning (0, "comparison is always %d due to width of bit-field",
code == NE_EXPR);
return constant_boolean_node (code == NE_EXPR, compare_type);
}
}
else
{
wide_int tem = wi::arshift (wi::to_wide (rhs), lbitsize - 1);
if (tem != 0 && tem != -1)
{
warning (0, "comparison is always %d due to width of bit-field",
code == NE_EXPR);
return constant_boolean_node (code == NE_EXPR, compare_type);
}
}
if (nbitpos < 0)
return 0;
if (lbitsize == 1 && ! integer_zerop (rhs))
{
code = code == EQ_EXPR ? NE_EXPR : EQ_EXPR;
rhs = build_int_cst (type, 0);
}
lhs = make_bit_field_ref (loc, linner, lhs, unsigned_type,
nbitsize, nbitpos, 1, lreversep);
rhs = const_binop (BIT_AND_EXPR,
const_binop (LSHIFT_EXPR,
fold_convert_loc (loc, unsigned_type, rhs),
size_int (lbitpos)),
mask);
lhs = build2_loc (loc, code, compare_type,
build2 (BIT_AND_EXPR, unsigned_type, lhs, mask), rhs);
return lhs;
}

static tree
decode_field_reference (location_t loc, tree *exp_, HOST_WIDE_INT *pbitsize,
HOST_WIDE_INT *pbitpos, machine_mode *pmode,
int *punsignedp, int *preversep, int *pvolatilep,
tree *pmask, tree *pand_mask)
{
tree exp = *exp_;
tree outer_type = 0;
tree and_mask = 0;
tree mask, inner, offset;
tree unsigned_type;
unsigned int precision;
if (! INTEGRAL_TYPE_P (TREE_TYPE (exp)))
return 0;
if (CONVERT_EXPR_P (exp)
|| TREE_CODE (exp) == NON_LVALUE_EXPR)
outer_type = TREE_TYPE (exp);
STRIP_NOPS (exp);
if (TREE_CODE (exp) == BIT_AND_EXPR)
{
and_mask = TREE_OPERAND (exp, 1);
exp = TREE_OPERAND (exp, 0);
STRIP_NOPS (exp); STRIP_NOPS (and_mask);
if (TREE_CODE (and_mask) != INTEGER_CST)
return 0;
}
poly_int64 poly_bitsize, poly_bitpos;
inner = get_inner_reference (exp, &poly_bitsize, &poly_bitpos, &offset,
pmode, punsignedp, preversep, pvolatilep);
if ((inner == exp && and_mask == 0)
|| !poly_bitsize.is_constant (pbitsize)
|| !poly_bitpos.is_constant (pbitpos)
|| *pbitsize < 0
|| offset != 0
|| TREE_CODE (inner) == PLACEHOLDER_EXPR
|| (! AGGREGATE_TYPE_P (TREE_TYPE (inner))
&& compare_tree_int (TYPE_SIZE (TREE_TYPE (inner)),
*pbitpos + *pbitsize) < 0))
return 0;
*exp_ = exp;
if (outer_type && *pbitsize == TYPE_PRECISION (outer_type))
*punsignedp = TYPE_UNSIGNED (outer_type);
unsigned_type = lang_hooks.types.type_for_size (*pbitsize, 1);
precision = TYPE_PRECISION (unsigned_type);
mask = build_int_cst_type (unsigned_type, -1);
mask = const_binop (LSHIFT_EXPR, mask, size_int (precision - *pbitsize));
mask = const_binop (RSHIFT_EXPR, mask, size_int (precision - *pbitsize));
if (and_mask != 0)
mask = fold_build2_loc (loc, BIT_AND_EXPR, unsigned_type,
fold_convert_loc (loc, unsigned_type, and_mask), mask);
*pmask = mask;
*pand_mask = and_mask;
return inner;
}
static int
all_ones_mask_p (const_tree mask, unsigned int size)
{
tree type = TREE_TYPE (mask);
unsigned int precision = TYPE_PRECISION (type);
if (size > precision || TYPE_SIGN (type) == UNSIGNED)
return false;
return wi::mask (size, false, precision) == wi::to_wide (mask);
}
tree
sign_bit_p (tree exp, const_tree val)
{
int width;
tree t;
t = TREE_TYPE (exp);
if (! INTEGRAL_TYPE_P (t))
return NULL_TREE;
if (TREE_CODE (val) != INTEGER_CST
|| TREE_OVERFLOW (val))
return NULL_TREE;
width = TYPE_PRECISION (t);
if (wi::only_sign_bit_p (wi::to_wide (val), width))
return exp;
if (TREE_CODE (exp) == NOP_EXPR
&& TYPE_PRECISION (TREE_TYPE (TREE_OPERAND (exp, 0))) < width)
return sign_bit_p (TREE_OPERAND (exp, 0), val);
return NULL_TREE;
}
static int
simple_operand_p (const_tree exp)
{
STRIP_NOPS (exp);
return (CONSTANT_CLASS_P (exp)
|| TREE_CODE (exp) == SSA_NAME
|| (DECL_P (exp)
&& ! TREE_ADDRESSABLE (exp)
&& ! TREE_THIS_VOLATILE (exp)
&& ! DECL_NONLOCAL (exp)
&& ! TREE_PUBLIC (exp)
&& ! DECL_EXTERNAL (exp)
&& (! VAR_OR_FUNCTION_DECL_P (exp) || ! DECL_WEAK (exp))
&& (! TREE_STATIC (exp) || DECL_REGISTER (exp))));
}
static bool
simple_operand_p_2 (tree exp)
{
enum tree_code code;
if (TREE_SIDE_EFFECTS (exp)
|| tree_could_trap_p (exp))
return false;
while (CONVERT_EXPR_P (exp))
exp = TREE_OPERAND (exp, 0);
code = TREE_CODE (exp);
if (TREE_CODE_CLASS (code) == tcc_comparison)
return (simple_operand_p (TREE_OPERAND (exp, 0))
&& simple_operand_p (TREE_OPERAND (exp, 1)));
if (code == TRUTH_NOT_EXPR)
return simple_operand_p_2 (TREE_OPERAND (exp, 0));
return simple_operand_p (exp);
}

static tree
range_binop (enum tree_code code, tree type, tree arg0, int upper0_p,
tree arg1, int upper1_p)
{
tree tem;
int result;
int sgn0, sgn1;
if (arg0 != 0 && arg1 != 0)
{
tem = fold_build2 (code, type != 0 ? type : TREE_TYPE (arg0),
arg0, fold_convert (TREE_TYPE (arg0), arg1));
STRIP_NOPS (tem);
return TREE_CODE (tem) == INTEGER_CST ? tem : 0;
}
if (TREE_CODE_CLASS (code) != tcc_comparison)
return 0;
sgn0 = arg0 != 0 ? 0 : (upper0_p ? 1 : -1);
sgn1 = arg1 != 0 ? 0 : (upper1_p ? 1 : -1);
switch (code)
{
case EQ_EXPR:
result = sgn0 == sgn1;
break;
case NE_EXPR:
result = sgn0 != sgn1;
break;
case LT_EXPR:
result = sgn0 < sgn1;
break;
case LE_EXPR:
result = sgn0 <= sgn1;
break;
case GT_EXPR:
result = sgn0 > sgn1;
break;
case GE_EXPR:
result = sgn0 >= sgn1;
break;
default:
gcc_unreachable ();
}
return constant_boolean_node (result, type);
}

tree
make_range_step (location_t loc, enum tree_code code, tree arg0, tree arg1,
tree exp_type, tree *p_low, tree *p_high, int *p_in_p,
bool *strict_overflow_p)
{
tree arg0_type = TREE_TYPE (arg0);
tree n_low, n_high, low = *p_low, high = *p_high;
int in_p = *p_in_p, n_in_p;
switch (code)
{
case TRUTH_NOT_EXPR:
if (low == NULL_TREE || high == NULL_TREE
|| ! integer_zerop (low) || ! integer_zerop (high))
return NULL_TREE;
*p_in_p = ! in_p;
return arg0;
case EQ_EXPR: case NE_EXPR:
case LT_EXPR: case LE_EXPR: case GE_EXPR: case GT_EXPR:
if (low == NULL_TREE || high == NULL_TREE
|| ! integer_zerop (low) || ! integer_zerop (high)
|| TREE_CODE (arg1) != INTEGER_CST)
return NULL_TREE;
switch (code)
{
case NE_EXPR:  
low = high = arg1;
break;
case EQ_EXPR:  
in_p = ! in_p, low = high = arg1;
break;
case GT_EXPR:  
low = 0, high = arg1;
break;
case GE_EXPR:  
in_p = ! in_p, low = arg1, high = 0;
break;
case LT_EXPR:  
low = arg1, high = 0;
break;
case LE_EXPR:  
in_p = ! in_p, low = 0, high = arg1;
break;
default:
gcc_unreachable ();
}
if (TYPE_UNSIGNED (arg0_type) && (low == 0 || high == 0))
{
if (! merge_ranges (&n_in_p, &n_low, &n_high,
in_p, low, high, 1,
build_int_cst (arg0_type, 0),
NULL_TREE))
return NULL_TREE;
in_p = n_in_p, low = n_low, high = n_high;
if (high == 0 && low && ! integer_zerop (low))
{
in_p = ! in_p;
high = range_binop (MINUS_EXPR, NULL_TREE, low, 0,
build_int_cst (TREE_TYPE (low), 1), 0);
low = build_int_cst (arg0_type, 0);
}
}
*p_low = low;
*p_high = high;
*p_in_p = in_p;
return arg0;
case NEGATE_EXPR:
if (!TYPE_UNSIGNED (arg0_type)
&& !TYPE_OVERFLOW_UNDEFINED (arg0_type))
{
if (low == NULL_TREE)
low = TYPE_MIN_VALUE (arg0_type);
if (high == NULL_TREE)
high = TYPE_MAX_VALUE (arg0_type);
}
n_low = range_binop (MINUS_EXPR, exp_type,
build_int_cst (exp_type, 0),
0, high, 1);
n_high = range_binop (MINUS_EXPR, exp_type,
build_int_cst (exp_type, 0),
0, low, 0);
if (n_high != 0 && TREE_OVERFLOW (n_high))
return NULL_TREE;
goto normalize;
case BIT_NOT_EXPR:
return build2_loc (loc, MINUS_EXPR, exp_type, negate_expr (arg0),
build_int_cst (exp_type, 1));
case PLUS_EXPR:
case MINUS_EXPR:
if (TREE_CODE (arg1) != INTEGER_CST)
return NULL_TREE;
if (!TYPE_UNSIGNED (arg0_type)
&& !TYPE_OVERFLOW_UNDEFINED (arg0_type))
return NULL_TREE;
n_low = range_binop (code == MINUS_EXPR ? PLUS_EXPR : MINUS_EXPR,
arg0_type, low, 0, arg1, 0);
n_high = range_binop (code == MINUS_EXPR ? PLUS_EXPR : MINUS_EXPR,
arg0_type, high, 1, arg1, 0);
if ((n_low != 0 && TREE_OVERFLOW (n_low))
|| (n_high != 0 && TREE_OVERFLOW (n_high)))
return NULL_TREE;
if (TYPE_OVERFLOW_UNDEFINED (arg0_type))
*strict_overflow_p = true;
normalize:
if (n_low && n_high && tree_int_cst_lt (n_high, n_low))
{
low = range_binop (PLUS_EXPR, arg0_type, n_high, 0,
build_int_cst (TREE_TYPE (n_high), 1), 0);
high = range_binop (MINUS_EXPR, arg0_type, n_low, 0,
build_int_cst (TREE_TYPE (n_low), 1), 0);
if (tree_int_cst_equal (n_low, low)
&& tree_int_cst_equal (n_high, high))
low = high = 0;
else
in_p = ! in_p;
}
else
low = n_low, high = n_high;
*p_low = low;
*p_high = high;
*p_in_p = in_p;
return arg0;
CASE_CONVERT:
case NON_LVALUE_EXPR:
if (TYPE_PRECISION (arg0_type) > TYPE_PRECISION (exp_type))
return NULL_TREE;
if (! INTEGRAL_TYPE_P (arg0_type)
|| (low != 0 && ! int_fits_type_p (low, arg0_type))
|| (high != 0 && ! int_fits_type_p (high, arg0_type)))
return NULL_TREE;
n_low = low, n_high = high;
if (n_low != 0)
n_low = fold_convert_loc (loc, arg0_type, n_low);
if (n_high != 0)
n_high = fold_convert_loc (loc, arg0_type, n_high);
if (!TYPE_UNSIGNED (exp_type) && TYPE_UNSIGNED (arg0_type))
{
tree high_positive;
tree equiv_type;
if (ALL_FIXED_POINT_MODE_P (TYPE_MODE (arg0_type)))
equiv_type
= lang_hooks.types.type_for_mode (TYPE_MODE (arg0_type),
TYPE_SATURATING (arg0_type));
else
equiv_type
= lang_hooks.types.type_for_mode (TYPE_MODE (arg0_type), 1);
high_positive
= TYPE_MAX_VALUE (equiv_type) ? TYPE_MAX_VALUE (equiv_type)
: TYPE_MAX_VALUE (arg0_type);
if (TYPE_PRECISION (exp_type) == TYPE_PRECISION (arg0_type))
high_positive = fold_build2_loc (loc, RSHIFT_EXPR, arg0_type,
fold_convert_loc (loc, arg0_type,
high_positive),
build_int_cst (arg0_type, 1));
if (low != 0)
{
if (! merge_ranges (&n_in_p, &n_low, &n_high, 1, n_low, n_high,
1, fold_convert_loc (loc, arg0_type,
integer_zero_node),
high_positive))
return NULL_TREE;
in_p = (n_in_p == in_p);
}
else
{
if (! merge_ranges (&n_in_p, &n_low, &n_high, 0, n_low, n_high,
1, fold_convert_loc (loc, arg0_type,
integer_zero_node),
high_positive))
return NULL_TREE;
in_p = (in_p != n_in_p);
}
}
*p_low = n_low;
*p_high = n_high;
*p_in_p = in_p;
return arg0;
default:
return NULL_TREE;
}
}
tree
make_range (tree exp, int *pin_p, tree *plow, tree *phigh,
bool *strict_overflow_p)
{
enum tree_code code;
tree arg0, arg1 = NULL_TREE;
tree exp_type, nexp;
int in_p;
tree low, high;
location_t loc = EXPR_LOCATION (exp);
in_p = 0;
low = high = build_int_cst (TREE_TYPE (exp), 0);
while (1)
{
code = TREE_CODE (exp);
exp_type = TREE_TYPE (exp);
arg0 = NULL_TREE;
if (IS_EXPR_CODE_CLASS (TREE_CODE_CLASS (code)))
{
if (TREE_OPERAND_LENGTH (exp) > 0)
arg0 = TREE_OPERAND (exp, 0);
if (TREE_CODE_CLASS (code) == tcc_binary
|| TREE_CODE_CLASS (code) == tcc_comparison
|| (TREE_CODE_CLASS (code) == tcc_expression
&& TREE_OPERAND_LENGTH (exp) > 1))
arg1 = TREE_OPERAND (exp, 1);
}
if (arg0 == NULL_TREE)
break;
nexp = make_range_step (loc, code, arg0, arg1, exp_type, &low,
&high, &in_p, strict_overflow_p);
if (nexp == NULL_TREE)
break;
exp = nexp;
}
if (TREE_CODE (exp) == INTEGER_CST)
{
in_p = in_p == (integer_onep (range_binop (GE_EXPR, integer_type_node,
exp, 0, low, 0))
&& integer_onep (range_binop (LE_EXPR, integer_type_node,
exp, 1, high, 1)));
low = high = 0;
exp = 0;
}
*pin_p = in_p, *plow = low, *phigh = high;
return exp;
}
static bool
maskable_range_p (const_tree low, const_tree high, tree type, tree *mask,
tree *value)
{
if (TREE_CODE (low) != INTEGER_CST
|| TREE_CODE (high) != INTEGER_CST)
return false;
unsigned prec = TYPE_PRECISION (type);
wide_int lo = wi::to_wide (low, prec);
wide_int hi = wi::to_wide (high, prec);
wide_int end_mask = lo ^ hi;
if ((end_mask & (end_mask + 1)) != 0
|| (lo & end_mask) != 0)
return false;
wide_int stem_mask = ~end_mask;
wide_int stem = lo & stem_mask;
if (stem != (hi & stem_mask))
return false;
*mask = wide_int_to_tree (type, stem_mask);
*value = wide_int_to_tree (type, stem);
return true;
}

tree
range_check_type (tree etype)
{
if (TREE_CODE (etype) == ENUMERAL_TYPE || TREE_CODE (etype) == BOOLEAN_TYPE)
etype = lang_hooks.types.type_for_size (TYPE_PRECISION (etype),
TYPE_UNSIGNED (etype));
if (TREE_CODE (etype) == INTEGER_TYPE && !TYPE_OVERFLOW_WRAPS (etype))
{
tree utype, minv, maxv;
utype = unsigned_type_for (etype);
maxv = fold_convert (utype, TYPE_MAX_VALUE (etype));
maxv = range_binop (PLUS_EXPR, NULL_TREE, maxv, 1,
build_int_cst (TREE_TYPE (maxv), 1), 1);
minv = fold_convert (utype, TYPE_MIN_VALUE (etype));
if (integer_zerop (range_binop (NE_EXPR, integer_type_node,
minv, 1, maxv, 1)))
etype = utype;
else
return NULL_TREE;
}
return etype;
}
tree
build_range_check (location_t loc, tree type, tree exp, int in_p,
tree low, tree high)
{
tree etype = TREE_TYPE (exp), mask, value;
if (targetm.have_canonicalize_funcptr_for_compare ()
&& POINTER_TYPE_P (etype)
&& FUNC_OR_METHOD_TYPE_P (TREE_TYPE (etype)))
return NULL_TREE;
if (! in_p)
{
value = build_range_check (loc, type, exp, 1, low, high);
if (value != 0)
return invert_truthvalue_loc (loc, value);
return 0;
}
if (low == 0 && high == 0)
return omit_one_operand_loc (loc, type, build_int_cst (type, 1), exp);
if (low == 0)
return fold_build2_loc (loc, LE_EXPR, type, exp,
fold_convert_loc (loc, etype, high));
if (high == 0)
return fold_build2_loc (loc, GE_EXPR, type, exp,
fold_convert_loc (loc, etype, low));
if (operand_equal_p (low, high, 0))
return fold_build2_loc (loc, EQ_EXPR, type, exp,
fold_convert_loc (loc, etype, low));
if (TREE_CODE (exp) == BIT_AND_EXPR
&& maskable_range_p (low, high, etype, &mask, &value))
return fold_build2_loc (loc, EQ_EXPR, type,
fold_build2_loc (loc, BIT_AND_EXPR, etype,
exp, mask),
value);
if (integer_zerop (low))
{
if (! TYPE_UNSIGNED (etype))
{
etype = unsigned_type_for (etype);
high = fold_convert_loc (loc, etype, high);
exp = fold_convert_loc (loc, etype, exp);
}
return build_range_check (loc, type, exp, 1, 0, high);
}
if (integer_onep (low) && TREE_CODE (high) == INTEGER_CST)
{
int prec = TYPE_PRECISION (etype);
if (wi::mask <widest_int> (prec - 1, false) == wi::to_widest (high))
{
if (TYPE_UNSIGNED (etype))
{
tree signed_etype = signed_type_for (etype);
if (TYPE_PRECISION (signed_etype) != TYPE_PRECISION (etype))
etype
= build_nonstandard_integer_type (TYPE_PRECISION (etype), 0);
else
etype = signed_etype;
exp = fold_convert_loc (loc, etype, exp);
}
return fold_build2_loc (loc, GT_EXPR, type, exp,
build_int_cst (etype, 0));
}
}
etype = range_check_type (etype);
if (etype == NULL_TREE)
return NULL_TREE;
if (POINTER_TYPE_P (etype))
etype = unsigned_type_for (etype);
high = fold_convert_loc (loc, etype, high);
low = fold_convert_loc (loc, etype, low);
exp = fold_convert_loc (loc, etype, exp);
value = const_binop (MINUS_EXPR, high, low);
if (value != 0 && !TREE_OVERFLOW (value))
return build_range_check (loc, type,
fold_build2_loc (loc, MINUS_EXPR, etype, exp, low),
1, build_int_cst (etype, 0), value);
return 0;
}

static tree
range_predecessor (tree val)
{
tree type = TREE_TYPE (val);
if (INTEGRAL_TYPE_P (type)
&& operand_equal_p (val, TYPE_MIN_VALUE (type), 0))
return 0;
else
return range_binop (MINUS_EXPR, NULL_TREE, val, 0,
build_int_cst (TREE_TYPE (val), 1), 0);
}
static tree
range_successor (tree val)
{
tree type = TREE_TYPE (val);
if (INTEGRAL_TYPE_P (type)
&& operand_equal_p (val, TYPE_MAX_VALUE (type), 0))
return 0;
else
return range_binop (PLUS_EXPR, NULL_TREE, val, 0,
build_int_cst (TREE_TYPE (val), 1), 0);
}
bool
merge_ranges (int *pin_p, tree *plow, tree *phigh, int in0_p, tree low0,
tree high0, int in1_p, tree low1, tree high1)
{
int no_overlap;
int subset;
int temp;
tree tem;
int in_p;
tree low, high;
int lowequal = ((low0 == 0 && low1 == 0)
|| integer_onep (range_binop (EQ_EXPR, integer_type_node,
low0, 0, low1, 0)));
int highequal = ((high0 == 0 && high1 == 0)
|| integer_onep (range_binop (EQ_EXPR, integer_type_node,
high0, 1, high1, 1)));
if (integer_onep (range_binop (GT_EXPR, integer_type_node,
low0, 0, low1, 0))
|| (lowequal
&& integer_onep (range_binop (GT_EXPR, integer_type_node,
high1, 1, high0, 1))))
{
temp = in0_p, in0_p = in1_p, in1_p = temp;
tem = low0, low0 = low1, low1 = tem;
tem = high0, high0 = high1, high1 = tem;
}
no_overlap = integer_onep (range_binop (LT_EXPR, integer_type_node,
high0, 1, low1, 0));
subset = integer_onep (range_binop (LE_EXPR, integer_type_node,
high1, 1, high0, 1));
if (in0_p && in1_p)
{
if (no_overlap)
in_p = 0, low = high = 0;
else if (subset)
in_p = 1, low = low1, high = high1;
else
in_p = 1, low = low1, high = high0;
}
else if (in0_p && ! in1_p)
{
if (no_overlap)
in_p = 1, low = low0, high = high0;
else if (lowequal && highequal)
in_p = 0, low = high = 0;
else if (subset && lowequal)
{
low = range_successor (high1);
high = high0;
in_p = 1;
if (low == 0)
{
return 0;
}
}
else if (! subset || highequal)
{
low = low0;
high = range_predecessor (low1);
in_p = 1;
if (high == 0)
{
return 0;
}
}
else
return 0;
}
else if (! in0_p && in1_p)
{
if (no_overlap)
in_p = 1, low = low1, high = high1;
else if (subset || highequal)
in_p = 0, low = high = 0;
else
{
low = range_successor (high0);
high = high1;
in_p = 1;
if (low == 0)
{
return 0;
}
}
}
else
{
if (no_overlap)
{
if (integer_onep (range_binop (EQ_EXPR, integer_type_node,
range_successor (high0),
1, low1, 0)))
in_p = 0, low = low0, high = high1;
else
{
if (low0 && TREE_CODE (low0) == INTEGER_CST)
switch (TREE_CODE (TREE_TYPE (low0)))
{
case ENUMERAL_TYPE:
if (maybe_ne (TYPE_PRECISION (TREE_TYPE (low0)),
GET_MODE_BITSIZE
(TYPE_MODE (TREE_TYPE (low0)))))
break;
case INTEGER_TYPE:
if (tree_int_cst_equal (low0,
TYPE_MIN_VALUE (TREE_TYPE (low0))))
low0 = 0;
break;
case POINTER_TYPE:
if (TYPE_UNSIGNED (TREE_TYPE (low0))
&& integer_zerop (low0))
low0 = 0;
break;
default:
break;
}
if (high1 && TREE_CODE (high1) == INTEGER_CST)
switch (TREE_CODE (TREE_TYPE (high1)))
{
case ENUMERAL_TYPE:
if (maybe_ne (TYPE_PRECISION (TREE_TYPE (high1)),
GET_MODE_BITSIZE
(TYPE_MODE (TREE_TYPE (high1)))))
break;
case INTEGER_TYPE:
if (tree_int_cst_equal (high1,
TYPE_MAX_VALUE (TREE_TYPE (high1))))
high1 = 0;
break;
case POINTER_TYPE:
if (TYPE_UNSIGNED (TREE_TYPE (high1))
&& integer_zerop (range_binop (PLUS_EXPR, NULL_TREE,
high1, 1,
build_int_cst (TREE_TYPE (high1), 1),
1)))
high1 = 0;
break;
default:
break;
}
if (low0 == 0 && high1 == 0)
{
low = range_successor (high0);
high = range_predecessor (low1);
if (low == 0 || high == 0)
return 0;
in_p = 1;
}
else
return 0;
}
}
else if (subset)
in_p = 0, low = low0, high = high0;
else
in_p = 0, low = low0, high = high1;
}
*pin_p = in_p, *plow = low, *phigh = high;
return 1;
}

static tree
fold_cond_expr_with_comparison (location_t loc, tree type,
tree arg0, tree arg1, tree arg2)
{
enum tree_code comp_code = TREE_CODE (arg0);
tree arg00 = TREE_OPERAND (arg0, 0);
tree arg01 = TREE_OPERAND (arg0, 1);
tree arg1_type = TREE_TYPE (arg1);
tree tem;
STRIP_NOPS (arg1);
STRIP_NOPS (arg2);
if (!HONOR_SIGNED_ZEROS (element_mode (type))
&& (FLOAT_TYPE_P (TREE_TYPE (arg01))
? real_zerop (arg01)
: integer_zerop (arg01))
&& ((TREE_CODE (arg2) == NEGATE_EXPR
&& operand_equal_p (TREE_OPERAND (arg2, 0), arg1, 0))
|| (TREE_CODE (arg1) == MINUS_EXPR
&& TREE_CODE (arg2) == MINUS_EXPR
&& operand_equal_p (TREE_OPERAND (arg1, 0),
TREE_OPERAND (arg2, 1), 0)
&& operand_equal_p (TREE_OPERAND (arg1, 1),
TREE_OPERAND (arg2, 0), 0))))
switch (comp_code)
{
case EQ_EXPR:
case UNEQ_EXPR:
tem = fold_convert_loc (loc, arg1_type, arg1);
return fold_convert_loc (loc, type, negate_expr (tem));
case NE_EXPR:
case LTGT_EXPR:
return fold_convert_loc (loc, type, arg1);
case UNGE_EXPR:
case UNGT_EXPR:
if (flag_trapping_math)
break;
case GE_EXPR:
case GT_EXPR:
if (TYPE_UNSIGNED (TREE_TYPE (arg1)))
break;
tem = fold_build1_loc (loc, ABS_EXPR, TREE_TYPE (arg1), arg1);
return fold_convert_loc (loc, type, tem);
case UNLE_EXPR:
case UNLT_EXPR:
if (flag_trapping_math)
break;
case LE_EXPR:
case LT_EXPR:
if (TYPE_UNSIGNED (TREE_TYPE (arg1)))
break;
tem = fold_build1_loc (loc, ABS_EXPR, TREE_TYPE (arg1), arg1);
return negate_expr (fold_convert_loc (loc, type, tem));
default:
gcc_assert (TREE_CODE_CLASS (comp_code) == tcc_comparison);
break;
}
if (!HONOR_SIGNED_ZEROS (element_mode (type))
&& integer_zerop (arg01) && integer_zerop (arg2))
{
if (comp_code == NE_EXPR)
return fold_convert_loc (loc, type, arg1);
else if (comp_code == EQ_EXPR)
return build_zero_cst (type);
}
if (!HONOR_SIGNED_ZEROS (element_mode (type))
&& operand_equal_for_comparison_p (arg01, arg2)
&& (in_gimple_form
|| VECTOR_TYPE_P (type)
|| (! lang_GNU_CXX ()
&& strcmp (lang_hooks.name, "GNU Objective-C++") != 0)
|| ! maybe_lvalue_p (arg1)
|| ! maybe_lvalue_p (arg2)))
{
tree comp_op0 = arg00;
tree comp_op1 = arg01;
tree comp_type = TREE_TYPE (comp_op0);
switch (comp_code)
{
case EQ_EXPR:
return fold_convert_loc (loc, type, arg2);
case NE_EXPR:
return fold_convert_loc (loc, type, arg1);
case LE_EXPR:
case LT_EXPR:
case UNLE_EXPR:
case UNLT_EXPR:
if (!HONOR_NANS (arg1))
{
comp_op0 = fold_convert_loc (loc, comp_type, comp_op0);
comp_op1 = fold_convert_loc (loc, comp_type, comp_op1);
tem = (comp_code == LE_EXPR || comp_code == UNLE_EXPR)
? fold_build2_loc (loc, MIN_EXPR, comp_type, comp_op0, comp_op1)
: fold_build2_loc (loc, MIN_EXPR, comp_type,
comp_op1, comp_op0);
return fold_convert_loc (loc, type, tem);
}
break;
case GE_EXPR:
case GT_EXPR:
case UNGE_EXPR:
case UNGT_EXPR:
if (!HONOR_NANS (arg1))
{
comp_op0 = fold_convert_loc (loc, comp_type, comp_op0);
comp_op1 = fold_convert_loc (loc, comp_type, comp_op1);
tem = (comp_code == GE_EXPR || comp_code == UNGE_EXPR)
? fold_build2_loc (loc, MAX_EXPR, comp_type, comp_op0, comp_op1)
: fold_build2_loc (loc, MAX_EXPR, comp_type,
comp_op1, comp_op0);
return fold_convert_loc (loc, type, tem);
}
break;
case UNEQ_EXPR:
if (!HONOR_NANS (arg1))
return fold_convert_loc (loc, type, arg2);
break;
case LTGT_EXPR:
if (!HONOR_NANS (arg1))
return fold_convert_loc (loc, type, arg1);
break;
default:
gcc_assert (TREE_CODE_CLASS (comp_code) == tcc_comparison);
break;
}
}
return NULL_TREE;
}

#ifndef LOGICAL_OP_NON_SHORT_CIRCUIT
#define LOGICAL_OP_NON_SHORT_CIRCUIT \
(BRANCH_COST (optimize_function_for_speed_p (cfun), \
false) >= 2)
#endif
static tree
fold_range_test (location_t loc, enum tree_code code, tree type,
tree op0, tree op1)
{
int or_op = (code == TRUTH_ORIF_EXPR
|| code == TRUTH_OR_EXPR);
int in0_p, in1_p, in_p;
tree low0, low1, low, high0, high1, high;
bool strict_overflow_p = false;
tree tem, lhs, rhs;
const char * const warnmsg = G_("assuming signed overflow does not occur "
"when simplifying range test");
if (!INTEGRAL_TYPE_P (type))
return 0;
lhs = make_range (op0, &in0_p, &low0, &high0, &strict_overflow_p);
rhs = make_range (op1, &in1_p, &low1, &high1, &strict_overflow_p);
if (or_op)
in0_p = ! in0_p, in1_p = ! in1_p;
if ((lhs == 0 || rhs == 0 || operand_equal_p (lhs, rhs, 0))
&& merge_ranges (&in_p, &low, &high, in0_p, low0, high0,
in1_p, low1, high1)
&& (tem = (build_range_check (loc, type,
lhs != 0 ? lhs
: rhs != 0 ? rhs : integer_zero_node,
in_p, low, high))) != 0)
{
if (strict_overflow_p)
fold_overflow_warning (warnmsg, WARN_STRICT_OVERFLOW_COMPARISON);
return or_op ? invert_truthvalue_loc (loc, tem) : tem;
}
else if (LOGICAL_OP_NON_SHORT_CIRCUIT
&& !flag_sanitize_coverage
&& lhs != 0 && rhs != 0
&& (code == TRUTH_ANDIF_EXPR
|| code == TRUTH_ORIF_EXPR)
&& operand_equal_p (lhs, rhs, 0))
{
if (simple_operand_p (lhs))
return build2_loc (loc, code == TRUTH_ANDIF_EXPR
? TRUTH_AND_EXPR : TRUTH_OR_EXPR,
type, op0, op1);
else if (!lang_hooks.decls.global_bindings_p ()
&& !CONTAINS_PLACEHOLDER_P (lhs))
{
tree common = save_expr (lhs);
if ((lhs = build_range_check (loc, type, common,
or_op ? ! in0_p : in0_p,
low0, high0)) != 0
&& (rhs = build_range_check (loc, type, common,
or_op ? ! in1_p : in1_p,
low1, high1)) != 0)
{
if (strict_overflow_p)
fold_overflow_warning (warnmsg,
WARN_STRICT_OVERFLOW_COMPARISON);
return build2_loc (loc, code == TRUTH_ANDIF_EXPR
? TRUTH_AND_EXPR : TRUTH_OR_EXPR,
type, lhs, rhs);
}
}
}
return 0;
}

static tree
unextend (tree c, int p, int unsignedp, tree mask)
{
tree type = TREE_TYPE (c);
int modesize = GET_MODE_BITSIZE (SCALAR_INT_TYPE_MODE (type));
tree temp;
if (p == modesize || unsignedp)
return c;
temp = build_int_cst (TREE_TYPE (c),
wi::extract_uhwi (wi::to_wide (c), p - 1, 1));
if (TYPE_UNSIGNED (type))
temp = fold_convert (signed_type_for (type), temp);
temp = const_binop (LSHIFT_EXPR, temp, size_int (modesize - 1));
temp = const_binop (RSHIFT_EXPR, temp, size_int (modesize - p - 1));
if (mask != 0)
temp = const_binop (BIT_AND_EXPR, temp,
fold_convert (TREE_TYPE (c), mask));
if (TYPE_UNSIGNED (type))
temp = fold_convert (type, temp);
return fold_convert (type, const_binop (BIT_XOR_EXPR, c, temp));
}

static tree
merge_truthop_with_opposite_arm (location_t loc, tree op, tree cmpop,
bool rhs_only)
{
tree type = TREE_TYPE (cmpop);
enum tree_code code = TREE_CODE (cmpop);
enum tree_code truthop_code = TREE_CODE (op);
tree lhs = TREE_OPERAND (op, 0);
tree rhs = TREE_OPERAND (op, 1);
tree orig_lhs = lhs, orig_rhs = rhs;
enum tree_code rhs_code = TREE_CODE (rhs);
enum tree_code lhs_code = TREE_CODE (lhs);
enum tree_code inv_code;
if (TREE_SIDE_EFFECTS (op) || TREE_SIDE_EFFECTS (cmpop))
return NULL_TREE;
if (TREE_CODE_CLASS (code) != tcc_comparison)
return NULL_TREE;
if (rhs_code == truthop_code)
{
tree newrhs = merge_truthop_with_opposite_arm (loc, rhs, cmpop, rhs_only);
if (newrhs != NULL_TREE)
{
rhs = newrhs;
rhs_code = TREE_CODE (rhs);
}
}
if (lhs_code == truthop_code && !rhs_only)
{
tree newlhs = merge_truthop_with_opposite_arm (loc, lhs, cmpop, false);
if (newlhs != NULL_TREE)
{
lhs = newlhs;
lhs_code = TREE_CODE (lhs);
}
}
inv_code = invert_tree_comparison (code, HONOR_NANS (type));
if (inv_code == rhs_code
&& operand_equal_p (TREE_OPERAND (rhs, 0), TREE_OPERAND (cmpop, 0), 0)
&& operand_equal_p (TREE_OPERAND (rhs, 1), TREE_OPERAND (cmpop, 1), 0))
return lhs;
if (!rhs_only && inv_code == lhs_code
&& operand_equal_p (TREE_OPERAND (lhs, 0), TREE_OPERAND (cmpop, 0), 0)
&& operand_equal_p (TREE_OPERAND (lhs, 1), TREE_OPERAND (cmpop, 1), 0))
return rhs;
if (rhs != orig_rhs || lhs != orig_lhs)
return fold_build2_loc (loc, truthop_code, TREE_TYPE (cmpop),
lhs, rhs);
return NULL_TREE;
}
static tree
fold_truth_andor_1 (location_t loc, enum tree_code code, tree truth_type,
tree lhs, tree rhs)
{
enum tree_code wanted_code;
enum tree_code lcode, rcode;
tree ll_arg, lr_arg, rl_arg, rr_arg;
tree ll_inner, lr_inner, rl_inner, rr_inner;
HOST_WIDE_INT ll_bitsize, ll_bitpos, lr_bitsize, lr_bitpos;
HOST_WIDE_INT rl_bitsize, rl_bitpos, rr_bitsize, rr_bitpos;
HOST_WIDE_INT xll_bitpos, xlr_bitpos, xrl_bitpos, xrr_bitpos;
HOST_WIDE_INT lnbitsize, lnbitpos, rnbitsize, rnbitpos;
int ll_unsignedp, lr_unsignedp, rl_unsignedp, rr_unsignedp;
int ll_reversep, lr_reversep, rl_reversep, rr_reversep;
machine_mode ll_mode, lr_mode, rl_mode, rr_mode;
scalar_int_mode lnmode, rnmode;
tree ll_mask, lr_mask, rl_mask, rr_mask;
tree ll_and_mask, lr_and_mask, rl_and_mask, rr_and_mask;
tree l_const, r_const;
tree lntype, rntype, result;
HOST_WIDE_INT first_bit, end_bit;
int volatilep;
if (TREE_SIDE_EFFECTS (lhs) || TREE_SIDE_EFFECTS (rhs))
return 0;
lcode = TREE_CODE (lhs);
rcode = TREE_CODE (rhs);
if (lcode == BIT_AND_EXPR && integer_onep (TREE_OPERAND (lhs, 1)))
{
lhs = build2 (NE_EXPR, truth_type, lhs,
build_int_cst (TREE_TYPE (lhs), 0));
lcode = NE_EXPR;
}
if (rcode == BIT_AND_EXPR && integer_onep (TREE_OPERAND (rhs, 1)))
{
rhs = build2 (NE_EXPR, truth_type, rhs,
build_int_cst (TREE_TYPE (rhs), 0));
rcode = NE_EXPR;
}
if (TREE_CODE_CLASS (lcode) != tcc_comparison
|| TREE_CODE_CLASS (rcode) != tcc_comparison)
return 0;
ll_arg = TREE_OPERAND (lhs, 0);
lr_arg = TREE_OPERAND (lhs, 1);
rl_arg = TREE_OPERAND (rhs, 0);
rr_arg = TREE_OPERAND (rhs, 1);
if (simple_operand_p (ll_arg)
&& simple_operand_p (lr_arg))
{
if (operand_equal_p (ll_arg, rl_arg, 0)
&& operand_equal_p (lr_arg, rr_arg, 0))
{
result = combine_comparisons (loc, code, lcode, rcode,
truth_type, ll_arg, lr_arg);
if (result)
return result;
}
else if (operand_equal_p (ll_arg, rr_arg, 0)
&& operand_equal_p (lr_arg, rl_arg, 0))
{
result = combine_comparisons (loc, code, lcode,
swap_tree_comparison (rcode),
truth_type, ll_arg, lr_arg);
if (result)
return result;
}
}
code = ((code == TRUTH_AND_EXPR || code == TRUTH_ANDIF_EXPR)
? TRUTH_AND_EXPR : TRUTH_OR_EXPR);
if (BRANCH_COST (optimize_function_for_speed_p (cfun),
false) >= 2
&& ! FLOAT_TYPE_P (TREE_TYPE (rl_arg))
&& simple_operand_p (rl_arg)
&& simple_operand_p (rr_arg))
{
if (code == TRUTH_OR_EXPR
&& lcode == NE_EXPR && integer_zerop (lr_arg)
&& rcode == NE_EXPR && integer_zerop (rr_arg)
&& TREE_TYPE (ll_arg) == TREE_TYPE (rl_arg)
&& INTEGRAL_TYPE_P (TREE_TYPE (ll_arg)))
return build2_loc (loc, NE_EXPR, truth_type,
build2 (BIT_IOR_EXPR, TREE_TYPE (ll_arg),
ll_arg, rl_arg),
build_int_cst (TREE_TYPE (ll_arg), 0));
if (code == TRUTH_AND_EXPR
&& lcode == EQ_EXPR && integer_zerop (lr_arg)
&& rcode == EQ_EXPR && integer_zerop (rr_arg)
&& TREE_TYPE (ll_arg) == TREE_TYPE (rl_arg)
&& INTEGRAL_TYPE_P (TREE_TYPE (ll_arg)))
return build2_loc (loc, EQ_EXPR, truth_type,
build2 (BIT_IOR_EXPR, TREE_TYPE (ll_arg),
ll_arg, rl_arg),
build_int_cst (TREE_TYPE (ll_arg), 0));
}
if ((lcode != EQ_EXPR && lcode != NE_EXPR)
|| (rcode != EQ_EXPR && rcode != NE_EXPR))
return 0;
ll_reversep = lr_reversep = rl_reversep = rr_reversep = 0;
volatilep = 0;
ll_inner = decode_field_reference (loc, &ll_arg,
&ll_bitsize, &ll_bitpos, &ll_mode,
&ll_unsignedp, &ll_reversep, &volatilep,
&ll_mask, &ll_and_mask);
lr_inner = decode_field_reference (loc, &lr_arg,
&lr_bitsize, &lr_bitpos, &lr_mode,
&lr_unsignedp, &lr_reversep, &volatilep,
&lr_mask, &lr_and_mask);
rl_inner = decode_field_reference (loc, &rl_arg,
&rl_bitsize, &rl_bitpos, &rl_mode,
&rl_unsignedp, &rl_reversep, &volatilep,
&rl_mask, &rl_and_mask);
rr_inner = decode_field_reference (loc, &rr_arg,
&rr_bitsize, &rr_bitpos, &rr_mode,
&rr_unsignedp, &rr_reversep, &volatilep,
&rr_mask, &rr_and_mask);
if (volatilep
|| ll_reversep != rl_reversep
|| ll_inner == 0 || rl_inner == 0
|| ! operand_equal_p (ll_inner, rl_inner, 0))
return 0;
if (TREE_CODE (lr_arg) == INTEGER_CST
&& TREE_CODE (rr_arg) == INTEGER_CST)
{
l_const = lr_arg, r_const = rr_arg;
lr_reversep = ll_reversep;
}
else if (lr_reversep != rr_reversep
|| lr_inner == 0 || rr_inner == 0
|| ! operand_equal_p (lr_inner, rr_inner, 0))
return 0;
else
l_const = r_const = 0;
wanted_code = (code == TRUTH_AND_EXPR ? EQ_EXPR : NE_EXPR);
if (lcode != wanted_code)
{
if (l_const && integer_zerop (l_const) && integer_pow2p (ll_mask))
{
ll_unsignedp = 1;
l_const = ll_mask;
}
else
return 0;
}
if (rcode != wanted_code)
{
if (r_const && integer_zerop (r_const) && integer_pow2p (rl_mask))
{
rl_unsignedp = 1;
r_const = rl_mask;
}
else
return 0;
}
first_bit = MIN (ll_bitpos, rl_bitpos);
end_bit = MAX (ll_bitpos + ll_bitsize, rl_bitpos + rl_bitsize);
if (!get_best_mode (end_bit - first_bit, first_bit, 0, 0,
TYPE_ALIGN (TREE_TYPE (ll_inner)), BITS_PER_WORD,
volatilep, &lnmode))
return 0;
lnbitsize = GET_MODE_BITSIZE (lnmode);
lnbitpos = first_bit & ~ (lnbitsize - 1);
lntype = lang_hooks.types.type_for_size (lnbitsize, 1);
xll_bitpos = ll_bitpos - lnbitpos, xrl_bitpos = rl_bitpos - lnbitpos;
if (ll_reversep ? !BYTES_BIG_ENDIAN : BYTES_BIG_ENDIAN)
{
xll_bitpos = lnbitsize - xll_bitpos - ll_bitsize;
xrl_bitpos = lnbitsize - xrl_bitpos - rl_bitsize;
}
ll_mask = const_binop (LSHIFT_EXPR, fold_convert_loc (loc, lntype, ll_mask),
size_int (xll_bitpos));
rl_mask = const_binop (LSHIFT_EXPR, fold_convert_loc (loc, lntype, rl_mask),
size_int (xrl_bitpos));
if (l_const)
{
l_const = fold_convert_loc (loc, lntype, l_const);
l_const = unextend (l_const, ll_bitsize, ll_unsignedp, ll_and_mask);
l_const = const_binop (LSHIFT_EXPR, l_const, size_int (xll_bitpos));
if (! integer_zerop (const_binop (BIT_AND_EXPR, l_const,
fold_build1_loc (loc, BIT_NOT_EXPR,
lntype, ll_mask))))
{
warning (0, "comparison is always %d", wanted_code == NE_EXPR);
return constant_boolean_node (wanted_code == NE_EXPR, truth_type);
}
}
if (r_const)
{
r_const = fold_convert_loc (loc, lntype, r_const);
r_const = unextend (r_const, rl_bitsize, rl_unsignedp, rl_and_mask);
r_const = const_binop (LSHIFT_EXPR, r_const, size_int (xrl_bitpos));
if (! integer_zerop (const_binop (BIT_AND_EXPR, r_const,
fold_build1_loc (loc, BIT_NOT_EXPR,
lntype, rl_mask))))
{
warning (0, "comparison is always %d", wanted_code == NE_EXPR);
return constant_boolean_node (wanted_code == NE_EXPR, truth_type);
}
}
if (l_const == 0)
{
if (ll_bitsize != lr_bitsize || rl_bitsize != rr_bitsize
|| ll_unsignedp != lr_unsignedp || rl_unsignedp != rr_unsignedp
|| ll_reversep != lr_reversep
|| ll_bitpos - rl_bitpos != lr_bitpos - rr_bitpos)
return 0;
first_bit = MIN (lr_bitpos, rr_bitpos);
end_bit = MAX (lr_bitpos + lr_bitsize, rr_bitpos + rr_bitsize);
if (!get_best_mode (end_bit - first_bit, first_bit, 0, 0,
TYPE_ALIGN (TREE_TYPE (lr_inner)), BITS_PER_WORD,
volatilep, &rnmode))
return 0;
rnbitsize = GET_MODE_BITSIZE (rnmode);
rnbitpos = first_bit & ~ (rnbitsize - 1);
rntype = lang_hooks.types.type_for_size (rnbitsize, 1);
xlr_bitpos = lr_bitpos - rnbitpos, xrr_bitpos = rr_bitpos - rnbitpos;
if (lr_reversep ? !BYTES_BIG_ENDIAN : BYTES_BIG_ENDIAN)
{
xlr_bitpos = rnbitsize - xlr_bitpos - lr_bitsize;
xrr_bitpos = rnbitsize - xrr_bitpos - rr_bitsize;
}
lr_mask = const_binop (LSHIFT_EXPR, fold_convert_loc (loc,
rntype, lr_mask),
size_int (xlr_bitpos));
rr_mask = const_binop (LSHIFT_EXPR, fold_convert_loc (loc,
rntype, rr_mask),
size_int (xrr_bitpos));
ll_mask = const_binop (BIT_IOR_EXPR, ll_mask, rl_mask);
lr_mask = const_binop (BIT_IOR_EXPR, lr_mask, rr_mask);
if (lnbitsize == rnbitsize
&& xll_bitpos == xlr_bitpos
&& lnbitpos >= 0
&& rnbitpos >= 0)
{
lhs = make_bit_field_ref (loc, ll_inner, ll_arg,
lntype, lnbitsize, lnbitpos,
ll_unsignedp || rl_unsignedp, ll_reversep);
if (! all_ones_mask_p (ll_mask, lnbitsize))
lhs = build2 (BIT_AND_EXPR, lntype, lhs, ll_mask);
rhs = make_bit_field_ref (loc, lr_inner, lr_arg,
rntype, rnbitsize, rnbitpos,
lr_unsignedp || rr_unsignedp, lr_reversep);
if (! all_ones_mask_p (lr_mask, rnbitsize))
rhs = build2 (BIT_AND_EXPR, rntype, rhs, lr_mask);
return build2_loc (loc, wanted_code, truth_type, lhs, rhs);
}
if (((ll_bitsize + ll_bitpos == rl_bitpos
&& lr_bitsize + lr_bitpos == rr_bitpos)
|| (ll_bitpos == rl_bitpos + rl_bitsize
&& lr_bitpos == rr_bitpos + rr_bitsize))
&& ll_bitpos >= 0
&& rl_bitpos >= 0
&& lr_bitpos >= 0
&& rr_bitpos >= 0)
{
tree type;
lhs = make_bit_field_ref (loc, ll_inner, ll_arg, lntype,
ll_bitsize + rl_bitsize,
MIN (ll_bitpos, rl_bitpos),
ll_unsignedp, ll_reversep);
rhs = make_bit_field_ref (loc, lr_inner, lr_arg, rntype,
lr_bitsize + rr_bitsize,
MIN (lr_bitpos, rr_bitpos),
lr_unsignedp, lr_reversep);
ll_mask = const_binop (RSHIFT_EXPR, ll_mask,
size_int (MIN (xll_bitpos, xrl_bitpos)));
lr_mask = const_binop (RSHIFT_EXPR, lr_mask,
size_int (MIN (xlr_bitpos, xrr_bitpos)));
type = lntype;
if (lntype != rntype)
{
if (lnbitsize > rnbitsize)
{
lhs = fold_convert_loc (loc, rntype, lhs);
ll_mask = fold_convert_loc (loc, rntype, ll_mask);
type = rntype;
}
else if (lnbitsize < rnbitsize)
{
rhs = fold_convert_loc (loc, lntype, rhs);
lr_mask = fold_convert_loc (loc, lntype, lr_mask);
type = lntype;
}
}
if (! all_ones_mask_p (ll_mask, ll_bitsize + rl_bitsize))
lhs = build2 (BIT_AND_EXPR, type, lhs, ll_mask);
if (! all_ones_mask_p (lr_mask, lr_bitsize + rr_bitsize))
rhs = build2 (BIT_AND_EXPR, type, rhs, lr_mask);
return build2_loc (loc, wanted_code, truth_type, lhs, rhs);
}
return 0;
}
result = const_binop (BIT_AND_EXPR, ll_mask, rl_mask);
if (! integer_zerop (result)
&& simple_cst_equal (const_binop (BIT_AND_EXPR, result, l_const),
const_binop (BIT_AND_EXPR, result, r_const)) != 1)
{
if (wanted_code == NE_EXPR)
{
warning (0, "%<or%> of unmatched not-equal tests is always 1");
return constant_boolean_node (true, truth_type);
}
else
{
warning (0, "%<and%> of mutually exclusive equal-tests is always 0");
return constant_boolean_node (false, truth_type);
}
}
if (lnbitpos < 0)
return 0;
result = make_bit_field_ref (loc, ll_inner, ll_arg,
lntype, lnbitsize, lnbitpos,
ll_unsignedp || rl_unsignedp, ll_reversep);
ll_mask = const_binop (BIT_IOR_EXPR, ll_mask, rl_mask);
if (! all_ones_mask_p (ll_mask, lnbitsize))
result = build2_loc (loc, BIT_AND_EXPR, lntype, result, ll_mask);
return build2_loc (loc, wanted_code, truth_type, result,
const_binop (BIT_IOR_EXPR, l_const, r_const));
}

static tree
extract_muldiv (tree t, tree c, enum tree_code code, tree wide_type,
bool *strict_overflow_p)
{
static int depth;
tree ret;
if (depth > 3)
return NULL;
depth++;
ret = extract_muldiv_1 (t, c, code, wide_type, strict_overflow_p);
depth--;
return ret;
}
static tree
extract_muldiv_1 (tree t, tree c, enum tree_code code, tree wide_type,
bool *strict_overflow_p)
{
tree type = TREE_TYPE (t);
enum tree_code tcode = TREE_CODE (t);
tree ctype = (wide_type != 0
&& (GET_MODE_SIZE (SCALAR_INT_TYPE_MODE (wide_type))
> GET_MODE_SIZE (SCALAR_INT_TYPE_MODE (type)))
? wide_type : type);
tree t1, t2;
int same_p = tcode == code;
tree op0 = NULL_TREE, op1 = NULL_TREE;
bool sub_strict_overflow_p;
if (integer_zerop (c))
return NULL_TREE;
if (TREE_CODE_CLASS (tcode) == tcc_unary)
op0 = TREE_OPERAND (t, 0);
if (TREE_CODE_CLASS (tcode) == tcc_binary)
op0 = TREE_OPERAND (t, 0), op1 = TREE_OPERAND (t, 1);
switch (tcode)
{
case INTEGER_CST:
if (code == MULT_EXPR
|| wi::multiple_of_p (wi::to_wide (t), wi::to_wide (c),
TYPE_SIGN (type)))
{
tree tem = const_binop (code, fold_convert (ctype, t),
fold_convert (ctype, c));
if (TREE_OVERFLOW (tem))
return NULL_TREE;
return tem;
}
break;
CASE_CONVERT: case NON_LVALUE_EXPR:
if ((COMPARISON_CLASS_P (op0)
|| UNARY_CLASS_P (op0)
|| BINARY_CLASS_P (op0)
|| VL_EXP_CLASS_P (op0)
|| EXPRESSION_CLASS_P (op0))
&& (((ANY_INTEGRAL_TYPE_P (TREE_TYPE (op0))
&& TYPE_OVERFLOW_WRAPS (TREE_TYPE (op0)))
&& (TYPE_PRECISION (ctype)
> TYPE_PRECISION (TREE_TYPE (op0))))
|| (TYPE_PRECISION (type)
< TYPE_PRECISION (TREE_TYPE (op0)))
|| (code != MULT_EXPR
&& (TYPE_UNSIGNED (ctype)
!= TYPE_UNSIGNED (TREE_TYPE (op0))))
|| ((ANY_INTEGRAL_TYPE_P (TREE_TYPE (op0))
&& TYPE_OVERFLOW_UNDEFINED (TREE_TYPE (op0)))
&& !TYPE_OVERFLOW_UNDEFINED (type))))
break;
if ((t2 = fold_convert (TREE_TYPE (op0), c)) != 0
&& TREE_CODE (t2) == INTEGER_CST
&& !TREE_OVERFLOW (t2)
&& (t1 = extract_muldiv (op0, t2, code,
code == MULT_EXPR ? ctype : NULL_TREE,
strict_overflow_p)) != 0)
return t1;
break;
case ABS_EXPR:
if (TYPE_UNSIGNED (ctype) && !TYPE_UNSIGNED (type))
{
tree cstype = (*signed_type_for) (ctype);
if ((t1 = extract_muldiv (op0, c, code, cstype, strict_overflow_p))
!= 0)
{
t1 = fold_build1 (tcode, cstype, fold_convert (cstype, t1));
return fold_convert (ctype, t1);
}
break;
}
if (tree_int_cst_sgn (c) == -1)
break;
case NEGATE_EXPR:
if (code != MULT_EXPR && TYPE_UNSIGNED (type))
break;
if ((t1 = extract_muldiv (op0, c, code, wide_type, strict_overflow_p))
!= 0)
return fold_build1 (tcode, ctype, fold_convert (ctype, t1));
break;
case MIN_EXPR:  case MAX_EXPR:
if (TYPE_UNSIGNED (ctype) != TYPE_UNSIGNED (type))
break;
sub_strict_overflow_p = false;
if ((t1 = extract_muldiv (op0, c, code, wide_type,
&sub_strict_overflow_p)) != 0
&& (t2 = extract_muldiv (op1, c, code, wide_type,
&sub_strict_overflow_p)) != 0)
{
if (tree_int_cst_sgn (c) < 0)
tcode = (tcode == MIN_EXPR ? MAX_EXPR : MIN_EXPR);
if (sub_strict_overflow_p)
*strict_overflow_p = true;
return fold_build2 (tcode, ctype, fold_convert (ctype, t1),
fold_convert (ctype, t2));
}
break;
case LSHIFT_EXPR:  case RSHIFT_EXPR:
if (TREE_CODE (op1) == INTEGER_CST
&& (tcode == RSHIFT_EXPR || TYPE_UNSIGNED (TREE_TYPE (op0)))
&& wi::gtu_p (TYPE_PRECISION (TREE_TYPE (size_one_node)),
wi::to_wide (op1))
&& (t1 = fold_convert (ctype,
const_binop (LSHIFT_EXPR, size_one_node,
op1))) != 0
&& !TREE_OVERFLOW (t1))
return extract_muldiv (build2 (tcode == LSHIFT_EXPR
? MULT_EXPR : FLOOR_DIV_EXPR,
ctype,
fold_convert (ctype, op0),
t1),
c, code, wide_type, strict_overflow_p);
break;
case PLUS_EXPR:  case MINUS_EXPR:
sub_strict_overflow_p = false;
t1 = extract_muldiv (op0, c, code, wide_type, &sub_strict_overflow_p);
t2 = extract_muldiv (op1, c, code, wide_type, &sub_strict_overflow_p);
if (t1 != 0 && t2 != 0
&& TYPE_OVERFLOW_WRAPS (ctype)
&& (code == MULT_EXPR
|| (multiple_of_p (ctype, op0, c)
&& multiple_of_p (ctype, op1, c))))
{
if (sub_strict_overflow_p)
*strict_overflow_p = true;
return fold_build2 (tcode, ctype, fold_convert (ctype, t1),
fold_convert (ctype, t2));
}
if (tcode == MINUS_EXPR)
{
tcode = PLUS_EXPR, op1 = negate_expr (op1);
if (TREE_CODE (op0) == INTEGER_CST)
{
std::swap (op0, op1);
std::swap (t1, t2);
}
}
if (TREE_CODE (op1) != INTEGER_CST)
break;
if (tree_int_cst_sgn (op1) < 0 || tree_int_cst_sgn (c) < 0)
{
if (code == CEIL_DIV_EXPR)
code = FLOOR_DIV_EXPR;
else if (code == FLOOR_DIV_EXPR)
code = CEIL_DIV_EXPR;
else if (code != MULT_EXPR
&& code != CEIL_MOD_EXPR && code != FLOOR_MOD_EXPR)
break;
}
if (code == MULT_EXPR
|| wi::multiple_of_p (wi::to_wide (op1), wi::to_wide (c),
TYPE_SIGN (type)))
{
op1 = const_binop (code, fold_convert (ctype, op1),
fold_convert (ctype, c));
if (op1 == 0
|| (TREE_OVERFLOW (op1) && !TYPE_OVERFLOW_WRAPS (ctype)))
break;
}
else
break;
if (TYPE_UNSIGNED (ctype) && ctype != type)
break;
if (code == MULT_EXPR && TYPE_OVERFLOW_WRAPS (ctype))
return fold_build2 (tcode, ctype,
fold_build2 (code, ctype,
fold_convert (ctype, op0),
fold_convert (ctype, c)),
op1);
break;
case MULT_EXPR:
if ((code == TRUNC_MOD_EXPR || code == CEIL_MOD_EXPR
|| code == FLOOR_MOD_EXPR || code == ROUND_MOD_EXPR)
&& TYPE_OVERFLOW_UNDEFINED (TREE_TYPE (t))
&& TREE_CODE (TREE_OPERAND (t, 1)) == INTEGER_CST
&& wi::multiple_of_p (wi::to_wide (op1), wi::to_wide (c),
TYPE_SIGN (type)))
{
*strict_overflow_p = true;
return omit_one_operand (type, integer_zero_node, op0);
}
case TRUNC_DIV_EXPR:  case CEIL_DIV_EXPR:  case FLOOR_DIV_EXPR:
case ROUND_DIV_EXPR:  case EXACT_DIV_EXPR:
if (same_p
&& TYPE_OVERFLOW_WRAPS (ctype)
&& (t1 = extract_muldiv (op0, c, code, wide_type,
strict_overflow_p)) != 0)
return fold_build2 (tcode, ctype, fold_convert (ctype, t1),
fold_convert (ctype, op1));
else if (tcode == MULT_EXPR && code == MULT_EXPR
&& TYPE_OVERFLOW_WRAPS (ctype)
&& (t1 = extract_muldiv (op1, c, code, wide_type,
strict_overflow_p)) != 0)
return fold_build2 (tcode, ctype, fold_convert (ctype, op0),
fold_convert (ctype, t1));
else if (TREE_CODE (op1) != INTEGER_CST)
return 0;
if (tcode == code)
{
bool overflow_p = false;
bool overflow_mul_p;
signop sign = TYPE_SIGN (ctype);
unsigned prec = TYPE_PRECISION (ctype);
wide_int mul = wi::mul (wi::to_wide (op1, prec),
wi::to_wide (c, prec),
sign, &overflow_mul_p);
overflow_p = TREE_OVERFLOW (c) | TREE_OVERFLOW (op1);
if (overflow_mul_p
&& ((sign == UNSIGNED && tcode != MULT_EXPR) || sign == SIGNED))
overflow_p = true;
if (!overflow_p)
return fold_build2 (tcode, ctype, fold_convert (ctype, op0),
wide_int_to_tree (ctype, mul));
}
if (TYPE_OVERFLOW_UNDEFINED (ctype)
&& ((code == MULT_EXPR && tcode == EXACT_DIV_EXPR)
|| (tcode == MULT_EXPR
&& code != TRUNC_MOD_EXPR && code != CEIL_MOD_EXPR
&& code != FLOOR_MOD_EXPR && code != ROUND_MOD_EXPR
&& code != MULT_EXPR)))
{
if (wi::multiple_of_p (wi::to_wide (op1), wi::to_wide (c),
TYPE_SIGN (type)))
{
if (TYPE_OVERFLOW_UNDEFINED (ctype))
*strict_overflow_p = true;
return fold_build2 (tcode, ctype, fold_convert (ctype, op0),
fold_convert (ctype,
const_binop (TRUNC_DIV_EXPR,
op1, c)));
}
else if (wi::multiple_of_p (wi::to_wide (c), wi::to_wide (op1),
TYPE_SIGN (type)))
{
if (TYPE_OVERFLOW_UNDEFINED (ctype))
*strict_overflow_p = true;
return fold_build2 (code, ctype, fold_convert (ctype, op0),
fold_convert (ctype,
const_binop (TRUNC_DIV_EXPR,
c, op1)));
}
}
break;
default:
break;
}
return 0;
}

tree
constant_boolean_node (bool value, tree type)
{
if (type == integer_type_node)
return value ? integer_one_node : integer_zero_node;
else if (type == boolean_type_node)
return value ? boolean_true_node : boolean_false_node;
else if (TREE_CODE (type) == VECTOR_TYPE)
return build_vector_from_val (type,
build_int_cst (TREE_TYPE (type),
value ? -1 : 0));
else
return fold_convert (type, value ? integer_one_node : integer_zero_node);
}
static tree
fold_binary_op_with_conditional_arg (location_t loc,
enum tree_code code,
tree type, tree op0, tree op1,
tree cond, tree arg, int cond_first_p)
{
tree cond_type = cond_first_p ? TREE_TYPE (op0) : TREE_TYPE (op1);
tree arg_type = cond_first_p ? TREE_TYPE (op1) : TREE_TYPE (op0);
tree test, true_value, false_value;
tree lhs = NULL_TREE;
tree rhs = NULL_TREE;
enum tree_code cond_code = COND_EXPR;
if (TREE_CODE (cond) == COND_EXPR
|| TREE_CODE (cond) == VEC_COND_EXPR)
{
test = TREE_OPERAND (cond, 0);
true_value = TREE_OPERAND (cond, 1);
false_value = TREE_OPERAND (cond, 2);
if (VOID_TYPE_P (TREE_TYPE (true_value)))
lhs = true_value;
if (VOID_TYPE_P (TREE_TYPE (false_value)))
rhs = false_value;
}
else if (!(TREE_CODE (type) != VECTOR_TYPE
&& TREE_CODE (TREE_TYPE (cond)) == VECTOR_TYPE))
{
tree testtype = TREE_TYPE (cond);
test = cond;
true_value = constant_boolean_node (true, testtype);
false_value = constant_boolean_node (false, testtype);
}
else
return NULL_TREE;
if (TREE_CODE (TREE_TYPE (test)) == VECTOR_TYPE)
cond_code = VEC_COND_EXPR;
if (!TREE_CONSTANT (arg)
&& (TREE_SIDE_EFFECTS (arg)
|| TREE_CODE (arg) == COND_EXPR || TREE_CODE (arg) == VEC_COND_EXPR
|| TREE_CONSTANT (true_value) || TREE_CONSTANT (false_value)))
return NULL_TREE;
arg = fold_convert_loc (loc, arg_type, arg);
if (lhs == 0)
{
true_value = fold_convert_loc (loc, cond_type, true_value);
if (cond_first_p)
lhs = fold_build2_loc (loc, code, type, true_value, arg);
else
lhs = fold_build2_loc (loc, code, type, arg, true_value);
}
if (rhs == 0)
{
false_value = fold_convert_loc (loc, cond_type, false_value);
if (cond_first_p)
rhs = fold_build2_loc (loc, code, type, false_value, arg);
else
rhs = fold_build2_loc (loc, code, type, arg, false_value);
}
if (!TREE_CONSTANT (arg) && !TREE_CONSTANT (lhs) && !TREE_CONSTANT (rhs))
return NULL_TREE;
return fold_build3_loc (loc, cond_code, type, test, lhs, rhs);
}

bool
fold_real_zero_addition_p (const_tree type, const_tree addend, int negate)
{
if (!real_zerop (addend))
return false;
if (HONOR_SNANS (element_mode (type)))
return false;
if (!HONOR_SIGNED_ZEROS (element_mode (type)))
return true;
if (TREE_CODE (addend) != REAL_CST)
return false;
if (REAL_VALUE_MINUS_ZERO (TREE_REAL_CST (addend)))
negate = !negate;
return negate && !HONOR_SIGN_DEPENDENT_ROUNDING (element_mode (type));
}
enum tree_code
fold_div_compare (enum tree_code code, tree c1, tree c2, tree *lo,
tree *hi, bool *neg_overflow)
{
tree prod, tmp, type = TREE_TYPE (c1);
signop sign = TYPE_SIGN (type);
bool overflow;
wide_int val = wi::mul (wi::to_wide (c1), wi::to_wide (c2), sign, &overflow);
prod = force_fit_type (type, val, -1, overflow);
*neg_overflow = false;
if (sign == UNSIGNED)
{
tmp = int_const_binop (MINUS_EXPR, c1, build_int_cst (type, 1));
*lo = prod;
val = wi::add (wi::to_wide (prod), wi::to_wide (tmp), sign, &overflow);
*hi = force_fit_type (type, val, -1, overflow | TREE_OVERFLOW (prod));
}
else if (tree_int_cst_sgn (c1) >= 0)
{
tmp = int_const_binop (MINUS_EXPR, c1, build_int_cst (type, 1));
switch (tree_int_cst_sgn (c2))
{
case -1:
*neg_overflow = true;
*lo = int_const_binop (MINUS_EXPR, prod, tmp);
*hi = prod;
break;
case 0:
*lo = fold_negate_const (tmp, type);
*hi = tmp;
break;
case 1:
*hi = int_const_binop (PLUS_EXPR, prod, tmp);
*lo = prod;
break;
default:
gcc_unreachable ();
}
}
else
{
code = swap_tree_comparison (code);
tmp = int_const_binop (PLUS_EXPR, c1, build_int_cst (type, 1));
switch (tree_int_cst_sgn (c2))
{
case -1:
*hi = int_const_binop (MINUS_EXPR, prod, tmp);
*lo = prod;
break;
case 0:
*hi = fold_negate_const (tmp, type);
*lo = tmp;
break;
case 1:
*neg_overflow = true;
*lo = int_const_binop (PLUS_EXPR, prod, tmp);
*hi = prod;
break;
default:
gcc_unreachable ();
}
}
if (code != EQ_EXPR && code != NE_EXPR)
return code;
if (TREE_OVERFLOW (*lo)
|| operand_equal_p (*lo, TYPE_MIN_VALUE (type), 0))
*lo = NULL_TREE;
if (TREE_OVERFLOW (*hi)
|| operand_equal_p (*hi, TYPE_MAX_VALUE (type), 0))
*hi = NULL_TREE;
return code;
}
static tree
fold_single_bit_test_into_sign_test (location_t loc,
enum tree_code code, tree arg0, tree arg1,
tree result_type)
{
if ((code == NE_EXPR || code == EQ_EXPR)
&& TREE_CODE (arg0) == BIT_AND_EXPR && integer_zerop (arg1)
&& integer_pow2p (TREE_OPERAND (arg0, 1)))
{
tree arg00 = sign_bit_p (TREE_OPERAND (arg0, 0), TREE_OPERAND (arg0, 1));
if (arg00 != NULL_TREE
&& type_has_mode_precision_p (TREE_TYPE (arg00)))
{
tree stype = signed_type_for (TREE_TYPE (arg00));
return fold_build2_loc (loc, code == EQ_EXPR ? GE_EXPR : LT_EXPR,
result_type,
fold_convert_loc (loc, stype, arg00),
build_int_cst (stype, 0));
}
}
return NULL_TREE;
}
tree
fold_single_bit_test (location_t loc, enum tree_code code,
tree arg0, tree arg1, tree result_type)
{
if ((code == NE_EXPR || code == EQ_EXPR)
&& TREE_CODE (arg0) == BIT_AND_EXPR && integer_zerop (arg1)
&& integer_pow2p (TREE_OPERAND (arg0, 1)))
{
tree inner = TREE_OPERAND (arg0, 0);
tree type = TREE_TYPE (arg0);
int bitnum = tree_log2 (TREE_OPERAND (arg0, 1));
scalar_int_mode operand_mode = SCALAR_INT_TYPE_MODE (type);
int ops_unsigned;
tree signed_type, unsigned_type, intermediate_type;
tree tem, one;
tem = fold_single_bit_test_into_sign_test (loc, code, arg0, arg1,
result_type);
if (tem)
return tem;
if (TREE_CODE (inner) == RSHIFT_EXPR
&& TREE_CODE (TREE_OPERAND (inner, 1)) == INTEGER_CST
&& bitnum < TYPE_PRECISION (type)
&& wi::ltu_p (wi::to_wide (TREE_OPERAND (inner, 1)),
TYPE_PRECISION (type) - bitnum))
{
bitnum += tree_to_uhwi (TREE_OPERAND (inner, 1));
inner = TREE_OPERAND (inner, 0);
}
ops_unsigned = (load_extend_op (operand_mode) == SIGN_EXTEND
&& !flag_syntax_only) ? 0 : 1;
signed_type = lang_hooks.types.type_for_mode (operand_mode, 0);
unsigned_type = lang_hooks.types.type_for_mode (operand_mode, 1);
intermediate_type = ops_unsigned ? unsigned_type : signed_type;
inner = fold_convert_loc (loc, intermediate_type, inner);
if (bitnum != 0)
inner = build2 (RSHIFT_EXPR, intermediate_type,
inner, size_int (bitnum));
one = build_int_cst (intermediate_type, 1);
if (code == EQ_EXPR)
inner = fold_build2_loc (loc, BIT_XOR_EXPR, intermediate_type, inner, one);
inner = build2 (BIT_AND_EXPR, intermediate_type, inner, one);
inner = fold_convert_loc (loc, result_type, inner);
return inner;
}
return NULL_TREE;
}
bool
tree_swap_operands_p (const_tree arg0, const_tree arg1)
{
if (CONSTANT_CLASS_P (arg1))
return 0;
if (CONSTANT_CLASS_P (arg0))
return 1;
STRIP_NOPS (arg0);
STRIP_NOPS (arg1);
if (TREE_CONSTANT (arg1))
return 0;
if (TREE_CONSTANT (arg0))
return 1;
if (TREE_CODE (arg0) == SSA_NAME
&& TREE_CODE (arg1) == SSA_NAME
&& SSA_NAME_VERSION (arg0) > SSA_NAME_VERSION (arg1))
return 1;
if (TREE_CODE (arg1) == SSA_NAME)
return 0;
if (TREE_CODE (arg0) == SSA_NAME)
return 1;
if (DECL_P (arg1))
return 0;
if (DECL_P (arg0))
return 1;
return 0;
}
static tree
fold_to_nonsharp_ineq_using_bound (location_t loc, tree ineq, tree bound)
{
tree a, typea, type = TREE_TYPE (ineq), a1, diff, y;
if (TREE_CODE (bound) == LT_EXPR)
a = TREE_OPERAND (bound, 0);
else if (TREE_CODE (bound) == GT_EXPR)
a = TREE_OPERAND (bound, 1);
else
return NULL_TREE;
typea = TREE_TYPE (a);
if (!INTEGRAL_TYPE_P (typea)
&& !POINTER_TYPE_P (typea))
return NULL_TREE;
if (TREE_CODE (ineq) == LT_EXPR)
{
a1 = TREE_OPERAND (ineq, 1);
y = TREE_OPERAND (ineq, 0);
}
else if (TREE_CODE (ineq) == GT_EXPR)
{
a1 = TREE_OPERAND (ineq, 0);
y = TREE_OPERAND (ineq, 1);
}
else
return NULL_TREE;
if (TREE_TYPE (a1) != typea)
return NULL_TREE;
if (POINTER_TYPE_P (typea))
{
tree ta = fold_convert_loc (loc, ssizetype, a);
tree ta1 = fold_convert_loc (loc, ssizetype, a1);
diff = fold_binary_loc (loc, MINUS_EXPR, ssizetype, ta1, ta);
}
else
diff = fold_binary_loc (loc, MINUS_EXPR, typea, a1, a);
if (!diff || !integer_onep (diff))
return NULL_TREE;
return fold_build2_loc (loc, GE_EXPR, type, a, y);
}
static tree
fold_plusminus_mult_expr (location_t loc, enum tree_code code, tree type,
tree arg0, tree arg1)
{
tree arg00, arg01, arg10, arg11;
tree alt0 = NULL_TREE, alt1 = NULL_TREE, same;
if (TREE_CODE (arg0) == MULT_EXPR)
{
arg00 = TREE_OPERAND (arg0, 0);
arg01 = TREE_OPERAND (arg0, 1);
}
else if (TREE_CODE (arg0) == INTEGER_CST)
{
arg00 = build_one_cst (type);
arg01 = arg0;
}
else
{
if (ALL_FRACT_MODE_P (TYPE_MODE (type)))
return NULL_TREE;
arg00 = arg0;
arg01 = build_one_cst (type);
}
if (TREE_CODE (arg1) == MULT_EXPR)
{
arg10 = TREE_OPERAND (arg1, 0);
arg11 = TREE_OPERAND (arg1, 1);
}
else if (TREE_CODE (arg1) == INTEGER_CST)
{
arg10 = build_one_cst (type);
if (wi::neg_p (wi::to_wide (arg1), TYPE_SIGN (TREE_TYPE (arg1)))
&& negate_expr_p (arg1)
&& code == PLUS_EXPR)
{
arg11 = negate_expr (arg1);
code = MINUS_EXPR;
}
else
arg11 = arg1;
}
else
{
if (ALL_FRACT_MODE_P (TYPE_MODE (type)))
return NULL_TREE;
arg10 = arg1;
arg11 = build_one_cst (type);
}
same = NULL_TREE;
if (operand_equal_p (arg00, arg10, 0))
same = arg00, alt0 = arg01, alt1 = arg11;
else if (operand_equal_p (arg01, arg11, 0))
same = arg01, alt0 = arg00, alt1 = arg10;
else if (operand_equal_p (arg00, arg11, 0))
same = arg00, alt0 = arg01, alt1 = arg10;
else if (operand_equal_p (arg01, arg10, 0))
same = arg01, alt0 = arg00, alt1 = arg11;
else if (tree_fits_shwi_p (arg01)
&& tree_fits_shwi_p (arg11))
{
HOST_WIDE_INT int01, int11, tmp;
bool swap = false;
tree maybe_same;
int01 = tree_to_shwi (arg01);
int11 = tree_to_shwi (arg11);
if (absu_hwi (int01) < absu_hwi (int11))
{
tmp = int01, int01 = int11, int11 = tmp;
alt0 = arg00, arg00 = arg10, arg10 = alt0;
maybe_same = arg01;
swap = true;
}
else
maybe_same = arg11;
if (exact_log2 (absu_hwi (int11)) > 0 && int01 % int11 == 0
&& TREE_CODE (arg10) != INTEGER_CST)
{
alt0 = fold_build2_loc (loc, MULT_EXPR, TREE_TYPE (arg00), arg00,
build_int_cst (TREE_TYPE (arg00),
int01 / int11));
alt1 = arg10;
same = maybe_same;
if (swap)
maybe_same = alt0, alt0 = alt1, alt1 = maybe_same;
}
}
if (!same)
return NULL_TREE;
if (! INTEGRAL_TYPE_P (type)
|| TYPE_OVERFLOW_WRAPS (type)
|| TREE_CODE (same) == INTEGER_CST)
return fold_build2_loc (loc, MULT_EXPR, type,
fold_build2_loc (loc, code, type,
fold_convert_loc (loc, type, alt0),
fold_convert_loc (loc, type, alt1)),
fold_convert_loc (loc, type, same));
tree utype = unsigned_type_for (type);
tree tem = fold_build2_loc (loc, code, utype,
fold_convert_loc (loc, utype, alt0),
fold_convert_loc (loc, utype, alt1));
if (TREE_CODE (tem) == INTEGER_CST
&& (wi::to_wide (tem)
!= wi::min_value (TYPE_PRECISION (utype), SIGNED)))
return fold_build2_loc (loc, MULT_EXPR, type,
fold_convert (type, tem), same);
return NULL_TREE;
}
static int
native_encode_int (const_tree expr, unsigned char *ptr, int len, int off)
{
tree type = TREE_TYPE (expr);
int total_bytes = GET_MODE_SIZE (SCALAR_INT_TYPE_MODE (type));
int byte, offset, word, words;
unsigned char value;
if ((off == -1 && total_bytes > len) || off >= total_bytes)
return 0;
if (off == -1)
off = 0;
if (ptr == NULL)
return MIN (len, total_bytes - off);
words = total_bytes / UNITS_PER_WORD;
for (byte = 0; byte < total_bytes; byte++)
{
int bitpos = byte * BITS_PER_UNIT;
value = wi::extract_uhwi (wi::to_widest (expr), bitpos, BITS_PER_UNIT);
if (total_bytes > UNITS_PER_WORD)
{
word = byte / UNITS_PER_WORD;
if (WORDS_BIG_ENDIAN)
word = (words - 1) - word;
offset = word * UNITS_PER_WORD;
if (BYTES_BIG_ENDIAN)
offset += (UNITS_PER_WORD - 1) - (byte % UNITS_PER_WORD);
else
offset += byte % UNITS_PER_WORD;
}
else
offset = BYTES_BIG_ENDIAN ? (total_bytes - 1) - byte : byte;
if (offset >= off && offset - off < len)
ptr[offset - off] = value;
}
return MIN (len, total_bytes - off);
}
static int
native_encode_fixed (const_tree expr, unsigned char *ptr, int len, int off)
{
tree type = TREE_TYPE (expr);
scalar_mode mode = SCALAR_TYPE_MODE (type);
int total_bytes = GET_MODE_SIZE (mode);
FIXED_VALUE_TYPE value;
tree i_value, i_type;
if (total_bytes * BITS_PER_UNIT > HOST_BITS_PER_DOUBLE_INT)
return 0;
i_type = lang_hooks.types.type_for_size (GET_MODE_BITSIZE (mode), 1);
if (NULL_TREE == i_type || TYPE_PRECISION (i_type) != total_bytes)
return 0;
value = TREE_FIXED_CST (expr);
i_value = double_int_to_tree (i_type, value.data);
return native_encode_int (i_value, ptr, len, off);
}
static int
native_encode_real (const_tree expr, unsigned char *ptr, int len, int off)
{
tree type = TREE_TYPE (expr);
int total_bytes = GET_MODE_SIZE (SCALAR_FLOAT_TYPE_MODE (type));
int byte, offset, word, words, bitpos;
unsigned char value;
long tmp[6];
if ((off == -1 && total_bytes > len) || off >= total_bytes)
return 0;
if (off == -1)
off = 0;
if (ptr == NULL)
return MIN (len, total_bytes - off);
words = (32 / BITS_PER_UNIT) / UNITS_PER_WORD;
real_to_target (tmp, TREE_REAL_CST_PTR (expr), TYPE_MODE (type));
for (bitpos = 0; bitpos < total_bytes * BITS_PER_UNIT;
bitpos += BITS_PER_UNIT)
{
byte = (bitpos / BITS_PER_UNIT) & 3;
value = (unsigned char) (tmp[bitpos / 32] >> (bitpos & 31));
if (UNITS_PER_WORD < 4)
{
word = byte / UNITS_PER_WORD;
if (WORDS_BIG_ENDIAN)
word = (words - 1) - word;
offset = word * UNITS_PER_WORD;
if (BYTES_BIG_ENDIAN)
offset += (UNITS_PER_WORD - 1) - (byte % UNITS_PER_WORD);
else
offset += byte % UNITS_PER_WORD;
}
else
{
offset = byte;
if (BYTES_BIG_ENDIAN)
{
offset = MIN (3, total_bytes - 1) - offset;
gcc_assert (offset >= 0);
}
}
offset = offset + ((bitpos / BITS_PER_UNIT) & ~3);
if (offset >= off
&& offset - off < len)
ptr[offset - off] = value;
}
return MIN (len, total_bytes - off);
}
static int
native_encode_complex (const_tree expr, unsigned char *ptr, int len, int off)
{
int rsize, isize;
tree part;
part = TREE_REALPART (expr);
rsize = native_encode_expr (part, ptr, len, off);
if (off == -1 && rsize == 0)
return 0;
part = TREE_IMAGPART (expr);
if (off != -1)
off = MAX (0, off - GET_MODE_SIZE (SCALAR_TYPE_MODE (TREE_TYPE (part))));
isize = native_encode_expr (part, ptr ? ptr + rsize : NULL,
len - rsize, off);
if (off == -1 && isize != rsize)
return 0;
return rsize + isize;
}
static int
native_encode_vector (const_tree expr, unsigned char *ptr, int len, int off)
{
unsigned HOST_WIDE_INT i, count;
int size, offset;
tree itype, elem;
offset = 0;
if (!VECTOR_CST_NELTS (expr).is_constant (&count))
return 0;
itype = TREE_TYPE (TREE_TYPE (expr));
size = GET_MODE_SIZE (SCALAR_TYPE_MODE (itype));
for (i = 0; i < count; i++)
{
if (off >= size)
{
off -= size;
continue;
}
elem = VECTOR_CST_ELT (expr, i);
int res = native_encode_expr (elem, ptr ? ptr + offset : NULL,
len - offset, off);
if ((off == -1 && res != size) || res == 0)
return 0;
offset += res;
if (offset >= len)
return (off == -1 && i < count - 1) ? 0 : offset;
if (off != -1)
off = 0;
}
return offset;
}
static int
native_encode_string (const_tree expr, unsigned char *ptr, int len, int off)
{
tree type = TREE_TYPE (expr);
if (BITS_PER_UNIT != CHAR_BIT
|| TREE_CODE (type) != ARRAY_TYPE
|| TREE_CODE (TREE_TYPE (type)) != INTEGER_TYPE
|| !tree_fits_shwi_p (TYPE_SIZE_UNIT (type)))
return 0;
HOST_WIDE_INT total_bytes = tree_to_shwi (TYPE_SIZE_UNIT (TREE_TYPE (expr)));
if ((off == -1 && total_bytes > len) || off >= total_bytes)
return 0;
if (off == -1)
off = 0;
if (ptr == NULL)
;
else if (TREE_STRING_LENGTH (expr) - off < MIN (total_bytes, len))
{
int written = 0;
if (off < TREE_STRING_LENGTH (expr))
{
written = MIN (len, TREE_STRING_LENGTH (expr) - off);
memcpy (ptr, TREE_STRING_POINTER (expr) + off, written);
}
memset (ptr + written, 0,
MIN (total_bytes - written, len - written));
}
else
memcpy (ptr, TREE_STRING_POINTER (expr) + off, MIN (total_bytes, len));
return MIN (total_bytes - off, len);
}
int
native_encode_expr (const_tree expr, unsigned char *ptr, int len, int off)
{
if (off < -1)
return 0;
switch (TREE_CODE (expr))
{
case INTEGER_CST:
return native_encode_int (expr, ptr, len, off);
case REAL_CST:
return native_encode_real (expr, ptr, len, off);
case FIXED_CST:
return native_encode_fixed (expr, ptr, len, off);
case COMPLEX_CST:
return native_encode_complex (expr, ptr, len, off);
case VECTOR_CST:
return native_encode_vector (expr, ptr, len, off);
case STRING_CST:
return native_encode_string (expr, ptr, len, off);
default:
return 0;
}
}
static tree
native_interpret_int (tree type, const unsigned char *ptr, int len)
{
int total_bytes = GET_MODE_SIZE (SCALAR_INT_TYPE_MODE (type));
if (total_bytes > len
|| total_bytes * BITS_PER_UNIT > HOST_BITS_PER_DOUBLE_INT)
return NULL_TREE;
wide_int result = wi::from_buffer (ptr, total_bytes);
return wide_int_to_tree (type, result);
}
static tree
native_interpret_fixed (tree type, const unsigned char *ptr, int len)
{
scalar_mode mode = SCALAR_TYPE_MODE (type);
int total_bytes = GET_MODE_SIZE (mode);
double_int result;
FIXED_VALUE_TYPE fixed_value;
if (total_bytes > len
|| total_bytes * BITS_PER_UNIT > HOST_BITS_PER_DOUBLE_INT)
return NULL_TREE;
result = double_int::from_buffer (ptr, total_bytes);
fixed_value = fixed_from_double_int (result, mode);
return build_fixed (type, fixed_value);
}
static tree
native_interpret_real (tree type, const unsigned char *ptr, int len)
{
scalar_float_mode mode = SCALAR_FLOAT_TYPE_MODE (type);
int total_bytes = GET_MODE_SIZE (mode);
unsigned char value;
REAL_VALUE_TYPE r;
long tmp[6];
if (total_bytes > len || total_bytes > 24)
return NULL_TREE;
int words = (32 / BITS_PER_UNIT) / UNITS_PER_WORD;
memset (tmp, 0, sizeof (tmp));
for (int bitpos = 0; bitpos < total_bytes * BITS_PER_UNIT;
bitpos += BITS_PER_UNIT)
{
int offset, byte = (bitpos / BITS_PER_UNIT) & 3;
if (UNITS_PER_WORD < 4)
{
int word = byte / UNITS_PER_WORD;
if (WORDS_BIG_ENDIAN)
word = (words - 1) - word;
offset = word * UNITS_PER_WORD;
if (BYTES_BIG_ENDIAN)
offset += (UNITS_PER_WORD - 1) - (byte % UNITS_PER_WORD);
else
offset += byte % UNITS_PER_WORD;
}
else
{
offset = byte;
if (BYTES_BIG_ENDIAN)
{
offset = MIN (3, total_bytes - 1) - offset;
gcc_assert (offset >= 0);
}
}
value = ptr[offset + ((bitpos / BITS_PER_UNIT) & ~3)];
tmp[bitpos / 32] |= (unsigned long)value << (bitpos & 31);
}
real_from_target (&r, tmp, mode);
return build_real (type, r);
}
static tree
native_interpret_complex (tree type, const unsigned char *ptr, int len)
{
tree etype, rpart, ipart;
int size;
etype = TREE_TYPE (type);
size = GET_MODE_SIZE (SCALAR_TYPE_MODE (etype));
if (size * 2 > len)
return NULL_TREE;
rpart = native_interpret_expr (etype, ptr, size);
if (!rpart)
return NULL_TREE;
ipart = native_interpret_expr (etype, ptr+size, size);
if (!ipart)
return NULL_TREE;
return build_complex (type, rpart, ipart);
}
static tree
native_interpret_vector (tree type, const unsigned char *ptr, unsigned int len)
{
tree etype, elem;
unsigned int i, size;
unsigned HOST_WIDE_INT count;
etype = TREE_TYPE (type);
size = GET_MODE_SIZE (SCALAR_TYPE_MODE (etype));
if (!TYPE_VECTOR_SUBPARTS (type).is_constant (&count)
|| size * count > len)
return NULL_TREE;
tree_vector_builder elements (type, count, 1);
for (i = 0; i < count; ++i)
{
elem = native_interpret_expr (etype, ptr+(i*size), size);
if (!elem)
return NULL_TREE;
elements.quick_push (elem);
}
return elements.build ();
}
tree
native_interpret_expr (tree type, const unsigned char *ptr, int len)
{
switch (TREE_CODE (type))
{
case INTEGER_TYPE:
case ENUMERAL_TYPE:
case BOOLEAN_TYPE:
case POINTER_TYPE:
case REFERENCE_TYPE:
return native_interpret_int (type, ptr, len);
case REAL_TYPE:
return native_interpret_real (type, ptr, len);
case FIXED_POINT_TYPE:
return native_interpret_fixed (type, ptr, len);
case COMPLEX_TYPE:
return native_interpret_complex (type, ptr, len);
case VECTOR_TYPE:
return native_interpret_vector (type, ptr, len);
default:
return NULL_TREE;
}
}
static bool
can_native_interpret_type_p (tree type)
{
switch (TREE_CODE (type))
{
case INTEGER_TYPE:
case ENUMERAL_TYPE:
case BOOLEAN_TYPE:
case POINTER_TYPE:
case REFERENCE_TYPE:
case FIXED_POINT_TYPE:
case REAL_TYPE:
case COMPLEX_TYPE:
case VECTOR_TYPE:
return true;
default:
return false;
}
}
static tree
fold_view_convert_expr (tree type, tree expr)
{
unsigned char buffer[64];
int len;
if (CHAR_BIT != 8 || BITS_PER_UNIT != 8)
return NULL_TREE;
len = native_encode_expr (expr, buffer, sizeof (buffer));
if (len == 0)
return NULL_TREE;
return native_interpret_expr (type, buffer, len);
}
tree
build_fold_addr_expr_with_type_loc (location_t loc, tree t, tree ptrtype)
{
if (TREE_CODE (t) == WITH_SIZE_EXPR)
t = TREE_OPERAND (t, 0);
if (TREE_CODE (t) == INDIRECT_REF)
{
t = TREE_OPERAND (t, 0);
if (TREE_TYPE (t) != ptrtype)
t = build1_loc (loc, NOP_EXPR, ptrtype, t);
}
else if (TREE_CODE (t) == MEM_REF
&& integer_zerop (TREE_OPERAND (t, 1)))
return TREE_OPERAND (t, 0);
else if (TREE_CODE (t) == MEM_REF
&& TREE_CODE (TREE_OPERAND (t, 0)) == INTEGER_CST)
return fold_binary (POINTER_PLUS_EXPR, ptrtype,
TREE_OPERAND (t, 0),
convert_to_ptrofftype (TREE_OPERAND (t, 1)));
else if (TREE_CODE (t) == VIEW_CONVERT_EXPR)
{
t = build_fold_addr_expr_loc (loc, TREE_OPERAND (t, 0));
if (TREE_TYPE (t) != ptrtype)
t = fold_convert_loc (loc, ptrtype, t);
}
else
t = build1_loc (loc, ADDR_EXPR, ptrtype, t);
return t;
}
tree
build_fold_addr_expr_loc (location_t loc, tree t)
{
tree ptrtype = build_pointer_type (TREE_TYPE (t));
return build_fold_addr_expr_with_type_loc (loc, t, ptrtype);
}
tree
fold_unary_loc (location_t loc, enum tree_code code, tree type, tree op0)
{
tree tem;
tree arg0;
enum tree_code_class kind = TREE_CODE_CLASS (code);
gcc_assert (IS_EXPR_CODE_CLASS (kind)
&& TREE_CODE_LENGTH (code) == 1);
arg0 = op0;
if (arg0)
{
if (CONVERT_EXPR_CODE_P (code)
|| code == FLOAT_EXPR || code == ABS_EXPR || code == NEGATE_EXPR)
{
STRIP_SIGN_NOPS (arg0);
}
else
{
STRIP_NOPS (arg0);
}
if (CONSTANT_CLASS_P (arg0))
{
tree tem = const_unop (code, type, arg0);
if (tem)
{
if (TREE_TYPE (tem) != type)
tem = fold_convert_loc (loc, type, tem);
return tem;
}
}
}
tem = generic_simplify (loc, code, type, op0);
if (tem)
return tem;
if (TREE_CODE_CLASS (code) == tcc_unary)
{
if (TREE_CODE (arg0) == COMPOUND_EXPR)
return build2 (COMPOUND_EXPR, type, TREE_OPERAND (arg0, 0),
fold_build1_loc (loc, code, type,
fold_convert_loc (loc, TREE_TYPE (op0),
TREE_OPERAND (arg0, 1))));
else if (TREE_CODE (arg0) == COND_EXPR)
{
tree arg01 = TREE_OPERAND (arg0, 1);
tree arg02 = TREE_OPERAND (arg0, 2);
if (! VOID_TYPE_P (TREE_TYPE (arg01)))
arg01 = fold_build1_loc (loc, code, type,
fold_convert_loc (loc,
TREE_TYPE (op0), arg01));
if (! VOID_TYPE_P (TREE_TYPE (arg02)))
arg02 = fold_build1_loc (loc, code, type,
fold_convert_loc (loc,
TREE_TYPE (op0), arg02));
tem = fold_build3_loc (loc, COND_EXPR, type, TREE_OPERAND (arg0, 0),
arg01, arg02);
if ((CONVERT_EXPR_CODE_P (code)
|| code == NON_LVALUE_EXPR)
&& TREE_CODE (tem) == COND_EXPR
&& TREE_CODE (TREE_OPERAND (tem, 1)) == code
&& TREE_CODE (TREE_OPERAND (tem, 2)) == code
&& ! VOID_TYPE_P (TREE_OPERAND (tem, 1))
&& ! VOID_TYPE_P (TREE_OPERAND (tem, 2))
&& (TREE_TYPE (TREE_OPERAND (TREE_OPERAND (tem, 1), 0))
== TREE_TYPE (TREE_OPERAND (TREE_OPERAND (tem, 2), 0)))
&& (! (INTEGRAL_TYPE_P (TREE_TYPE (tem))
&& (INTEGRAL_TYPE_P
(TREE_TYPE (TREE_OPERAND (TREE_OPERAND (tem, 1), 0))))
&& TYPE_PRECISION (TREE_TYPE (tem)) <= BITS_PER_WORD)
|| flag_syntax_only))
tem = build1_loc (loc, code, type,
build3 (COND_EXPR,
TREE_TYPE (TREE_OPERAND
(TREE_OPERAND (tem, 1), 0)),
TREE_OPERAND (tem, 0),
TREE_OPERAND (TREE_OPERAND (tem, 1), 0),
TREE_OPERAND (TREE_OPERAND (tem, 2),
0)));
return tem;
}
}
switch (code)
{
case NON_LVALUE_EXPR:
if (!maybe_lvalue_p (op0))
return fold_convert_loc (loc, type, op0);
return NULL_TREE;
CASE_CONVERT:
case FLOAT_EXPR:
case FIX_TRUNC_EXPR:
if (COMPARISON_CLASS_P (op0))
{
if (TREE_CODE (type) == BOOLEAN_TYPE)
return build2_loc (loc, TREE_CODE (op0), type,
TREE_OPERAND (op0, 0),
TREE_OPERAND (op0, 1));
else if (!INTEGRAL_TYPE_P (type) && !VOID_TYPE_P (type)
&& TREE_CODE (type) != VECTOR_TYPE)
return build3_loc (loc, COND_EXPR, type, op0,
constant_boolean_node (true, type),
constant_boolean_node (false, type));
}
if (TREE_CODE (op0) == ADDR_EXPR
&& POINTER_TYPE_P (type)
&& handled_component_p (TREE_OPERAND (op0, 0)))
{
poly_int64 bitsize, bitpos;
tree offset;
machine_mode mode;
int unsignedp, reversep, volatilep;
tree base
= get_inner_reference (TREE_OPERAND (op0, 0), &bitsize, &bitpos,
&offset, &mode, &unsignedp, &reversep,
&volatilep);
if (!offset
&& known_eq (bitpos, 0)
&& (TYPE_MAIN_VARIANT (TREE_TYPE (type))
== TYPE_MAIN_VARIANT (TREE_TYPE (base)))
&& TYPE_QUALS (type) == TYPE_UNQUALIFIED)
return fold_convert_loc (loc, type,
build_fold_addr_expr_loc (loc, base));
}
if (TREE_CODE (op0) == MODIFY_EXPR
&& TREE_CONSTANT (TREE_OPERAND (op0, 1))
&& !(TREE_CODE (TREE_OPERAND (op0, 0)) == COMPONENT_REF
&& DECL_BIT_FIELD
(TREE_OPERAND (TREE_OPERAND (op0, 0), 1))))
{
tem = fold_build1_loc (loc, code, type, TREE_OPERAND (op0, 1));
tem = build2_loc (loc, COMPOUND_EXPR, TREE_TYPE (tem), op0, tem);
TREE_NO_WARNING (tem) = 1;
TREE_USED (tem) = 1;
return tem;
}
if (TREE_CODE (type) == INTEGER_TYPE
&& TREE_CODE (op0) == BIT_AND_EXPR
&& TREE_CODE (TREE_OPERAND (op0, 1)) == INTEGER_CST)
{
tree and_expr = op0;
tree and0 = TREE_OPERAND (and_expr, 0);
tree and1 = TREE_OPERAND (and_expr, 1);
int change = 0;
if (TYPE_UNSIGNED (TREE_TYPE (and_expr))
|| (TYPE_PRECISION (type)
<= TYPE_PRECISION (TREE_TYPE (and_expr))))
change = 1;
else if (TYPE_PRECISION (TREE_TYPE (and1))
<= HOST_BITS_PER_WIDE_INT
&& tree_fits_uhwi_p (and1))
{
unsigned HOST_WIDE_INT cst;
cst = tree_to_uhwi (and1);
cst &= HOST_WIDE_INT_M1U
<< (TYPE_PRECISION (TREE_TYPE (and1)) - 1);
change = (cst == 0);
if (change
&& !flag_syntax_only
&& (load_extend_op (TYPE_MODE (TREE_TYPE (and0)))
== ZERO_EXTEND))
{
tree uns = unsigned_type_for (TREE_TYPE (and0));
and0 = fold_convert_loc (loc, uns, and0);
and1 = fold_convert_loc (loc, uns, and1);
}
}
if (change)
{
tem = force_fit_type (type, wi::to_widest (and1), 0,
TREE_OVERFLOW (and1));
return fold_build2_loc (loc, BIT_AND_EXPR, type,
fold_convert_loc (loc, type, and0), tem);
}
}
if (POINTER_TYPE_P (type)
&& TREE_CODE (arg0) == POINTER_PLUS_EXPR
&& CONVERT_EXPR_P (TREE_OPERAND (arg0, 0)))
{
tree arg00 = TREE_OPERAND (arg0, 0);
tree arg01 = TREE_OPERAND (arg0, 1);
return fold_build_pointer_plus_loc
(loc, fold_convert_loc (loc, type, arg00), arg01);
}
if (INTEGRAL_TYPE_P (type)
&& TREE_CODE (op0) == BIT_NOT_EXPR
&& INTEGRAL_TYPE_P (TREE_TYPE (op0))
&& CONVERT_EXPR_P (TREE_OPERAND (op0, 0))
&& TYPE_PRECISION (type) == TYPE_PRECISION (TREE_TYPE (op0)))
{
tem = TREE_OPERAND (TREE_OPERAND (op0, 0), 0);
if (INTEGRAL_TYPE_P (TREE_TYPE (tem))
&& TYPE_PRECISION (type) <= TYPE_PRECISION (TREE_TYPE (tem)))
return fold_build1_loc (loc, BIT_NOT_EXPR, type,
fold_convert_loc (loc, type, tem));
}
if (INTEGRAL_TYPE_P (type)
&& TREE_CODE (op0) == MULT_EXPR
&& INTEGRAL_TYPE_P (TREE_TYPE (op0))
&& TYPE_PRECISION (type) < TYPE_PRECISION (TREE_TYPE (op0)))
{
tree mult_type;
if (TYPE_OVERFLOW_WRAPS (type))
mult_type = type;
else
mult_type = unsigned_type_for (type);
if (TYPE_PRECISION (mult_type) < TYPE_PRECISION (TREE_TYPE (op0)))
{
tem = fold_build2_loc (loc, MULT_EXPR, mult_type,
fold_convert_loc (loc, mult_type,
TREE_OPERAND (op0, 0)),
fold_convert_loc (loc, mult_type,
TREE_OPERAND (op0, 1)));
return fold_convert_loc (loc, type, tem);
}
}
return NULL_TREE;
case VIEW_CONVERT_EXPR:
if (TREE_CODE (op0) == MEM_REF)
{
if (TYPE_ALIGN (TREE_TYPE (op0)) != TYPE_ALIGN (type))
type = build_aligned_type (type, TYPE_ALIGN (TREE_TYPE (op0)));
tem = fold_build2_loc (loc, MEM_REF, type,
TREE_OPERAND (op0, 0), TREE_OPERAND (op0, 1));
REF_REVERSE_STORAGE_ORDER (tem) = REF_REVERSE_STORAGE_ORDER (op0);
return tem;
}
return NULL_TREE;
case NEGATE_EXPR:
tem = fold_negate_expr (loc, arg0);
if (tem)
return fold_convert_loc (loc, type, tem);
return NULL_TREE;
case ABS_EXPR:
if (TREE_CODE (arg0) == NOP_EXPR
&& TREE_CODE (type) == REAL_TYPE)
{
tree targ0 = strip_float_extensions (arg0);
if (targ0 != arg0)
return fold_convert_loc (loc, type,
fold_build1_loc (loc, ABS_EXPR,
TREE_TYPE (targ0),
targ0));
}
return NULL_TREE;
case BIT_NOT_EXPR:
if (TREE_CODE (arg0) == BIT_XOR_EXPR
&& (tem = fold_unary_loc (loc, BIT_NOT_EXPR, type,
fold_convert_loc (loc, type,
TREE_OPERAND (arg0, 0)))))
return fold_build2_loc (loc, BIT_XOR_EXPR, type, tem,
fold_convert_loc (loc, type,
TREE_OPERAND (arg0, 1)));
else if (TREE_CODE (arg0) == BIT_XOR_EXPR
&& (tem = fold_unary_loc (loc, BIT_NOT_EXPR, type,
fold_convert_loc (loc, type,
TREE_OPERAND (arg0, 1)))))
return fold_build2_loc (loc, BIT_XOR_EXPR, type,
fold_convert_loc (loc, type,
TREE_OPERAND (arg0, 0)), tem);
return NULL_TREE;
case TRUTH_NOT_EXPR:
tem = fold_truth_not_expr (loc, arg0);
if (!tem)
return NULL_TREE;
return fold_convert_loc (loc, type, tem);
case INDIRECT_REF:
if (TREE_CODE (op0) == ADDR_EXPR)
{
tree op00 = TREE_OPERAND (op0, 0);
if ((VAR_P (op00)
|| TREE_CODE (op00) == PARM_DECL
|| TREE_CODE (op00) == RESULT_DECL)
&& !TREE_READONLY (op00))
return op00;
}
return NULL_TREE;
default:
return NULL_TREE;
} 
}
tree
fold_unary_ignore_overflow_loc (location_t loc, enum tree_code code,
tree type, tree op0)
{
tree res = fold_unary_loc (loc, code, type, op0);
if (res
&& TREE_CODE (res) == INTEGER_CST
&& TREE_CODE (op0) == INTEGER_CST
&& CONVERT_EXPR_CODE_P (code))
TREE_OVERFLOW (res) = TREE_OVERFLOW (op0);
return res;
}
static tree
fold_truth_andor (location_t loc, enum tree_code code, tree type,
tree arg0, tree arg1, tree op0, tree op1)
{
tree tem;
if (!optimize)
return NULL_TREE;
if (TREE_CODE (arg0) == TREE_CODE (arg1)
&& (TREE_CODE (arg0) == TRUTH_ANDIF_EXPR
|| TREE_CODE (arg0) == TRUTH_ORIF_EXPR
|| TREE_CODE (arg0) == TRUTH_AND_EXPR
|| TREE_CODE (arg0) == TRUTH_OR_EXPR)
&& ! TREE_SIDE_EFFECTS (TREE_OPERAND (arg0, 1)))
{
tree a00 = TREE_OPERAND (arg0, 0);
tree a01 = TREE_OPERAND (arg0, 1);
tree a10 = TREE_OPERAND (arg1, 0);
tree a11 = TREE_OPERAND (arg1, 1);
int commutative = ((TREE_CODE (arg0) == TRUTH_OR_EXPR
|| TREE_CODE (arg0) == TRUTH_AND_EXPR)
&& (code == TRUTH_AND_EXPR
|| code == TRUTH_OR_EXPR));
if (operand_equal_p (a00, a10, 0))
return fold_build2_loc (loc, TREE_CODE (arg0), type, a00,
fold_build2_loc (loc, code, type, a01, a11));
else if (commutative && operand_equal_p (a00, a11, 0))
return fold_build2_loc (loc, TREE_CODE (arg0), type, a00,
fold_build2_loc (loc, code, type, a01, a10));
else if (commutative && operand_equal_p (a01, a10, 0))
return fold_build2_loc (loc, TREE_CODE (arg0), type, a01,
fold_build2_loc (loc, code, type, a00, a11));
else if ((commutative || ! TREE_SIDE_EFFECTS (a10))
&& operand_equal_p (a01, a11, 0))
return fold_build2_loc (loc, TREE_CODE (arg0), type,
fold_build2_loc (loc, code, type, a00, a10),
a01);
}
if ((tem = fold_range_test (loc, code, type, op0, op1)) != 0)
return tem;
if ((code == TRUTH_ANDIF_EXPR && TREE_CODE (arg0) == TRUTH_ORIF_EXPR)
|| (code == TRUTH_ORIF_EXPR && TREE_CODE (arg0) == TRUTH_ANDIF_EXPR))
{
tem = merge_truthop_with_opposite_arm (loc, arg0, arg1, true);
if (tem)
return fold_build2_loc (loc, code, type, tem, arg1);
}
if ((code == TRUTH_ANDIF_EXPR && TREE_CODE (arg1) == TRUTH_ORIF_EXPR)
|| (code == TRUTH_ORIF_EXPR && TREE_CODE (arg1) == TRUTH_ANDIF_EXPR))
{
tem = merge_truthop_with_opposite_arm (loc, arg1, arg0, false);
if (tem)
return fold_build2_loc (loc, code, type, arg0, tem);
}
if (TREE_CODE (arg0) == code
&& (tem = fold_truth_andor_1 (loc, code, type,
TREE_OPERAND (arg0, 1), arg1)) != 0)
return fold_build2_loc (loc, code, type, TREE_OPERAND (arg0, 0), tem);
if ((tem = fold_truth_andor_1 (loc, code, type, arg0, arg1)) != 0)
return tem;
if (LOGICAL_OP_NON_SHORT_CIRCUIT
&& !flag_sanitize_coverage
&& (code == TRUTH_AND_EXPR
|| code == TRUTH_ANDIF_EXPR
|| code == TRUTH_OR_EXPR
|| code == TRUTH_ORIF_EXPR))
{
enum tree_code ncode, icode;
ncode = (code == TRUTH_ANDIF_EXPR || code == TRUTH_AND_EXPR)
? TRUTH_AND_EXPR : TRUTH_OR_EXPR;
icode = ncode == TRUTH_AND_EXPR ? TRUTH_ANDIF_EXPR : TRUTH_ORIF_EXPR;
if (TREE_CODE (arg0) == icode
&& simple_operand_p_2 (arg1)
&& simple_operand_p_2 (TREE_OPERAND (arg0, 1)))
{
tem = fold_build2_loc (loc, ncode, type, TREE_OPERAND (arg0, 1),
arg1);
return fold_build2_loc (loc, icode, type, TREE_OPERAND (arg0, 0),
tem);
}
else if (TREE_CODE (arg1) == icode
&& simple_operand_p_2 (arg0)
&& simple_operand_p_2 (TREE_OPERAND (arg1, 0)))
{
tem = fold_build2_loc (loc, ncode, type, 
arg0, TREE_OPERAND (arg1, 0));
return fold_build2_loc (loc, icode, type, tem,
TREE_OPERAND (arg1, 1));
}
else if (code == icode && simple_operand_p_2 (arg0)
&& simple_operand_p_2 (arg1))
return fold_build2_loc (loc, ncode, type, arg0, arg1);
}
return NULL_TREE;
}
static tree
maybe_canonicalize_comparison_1 (location_t loc, enum tree_code code, tree type,
tree arg0, tree arg1,
bool *strict_overflow_p)
{
enum tree_code code0 = TREE_CODE (arg0);
tree t, cst0 = NULL_TREE;
int sgn0;
if (!((ANY_INTEGRAL_TYPE_P (TREE_TYPE (arg0))
&& TYPE_OVERFLOW_UNDEFINED (TREE_TYPE (arg0)))
&& !POINTER_TYPE_P (TREE_TYPE (arg0))
&& (code0 == MINUS_EXPR
|| code0 == PLUS_EXPR)
&& TREE_CODE (TREE_OPERAND (arg0, 1)) == INTEGER_CST))
return NULL_TREE;
cst0 = TREE_OPERAND (arg0, 1);
sgn0 = tree_int_cst_sgn (cst0);
if (integer_zerop (cst0)
|| TREE_OVERFLOW (cst0))
return NULL_TREE;
if (code == LT_EXPR
&& code0 == ((sgn0 == -1) ? PLUS_EXPR : MINUS_EXPR))
code = LE_EXPR;
else if (code == GT_EXPR
&& code0 == ((sgn0 == -1) ? MINUS_EXPR : PLUS_EXPR))
code = GE_EXPR;
else if (code == LE_EXPR
&& code0 == ((sgn0 == -1) ? MINUS_EXPR : PLUS_EXPR))
code = LT_EXPR;
else if (code == GE_EXPR
&& code0 == ((sgn0 == -1) ? PLUS_EXPR : MINUS_EXPR))
code = GT_EXPR;
else
return NULL_TREE;
*strict_overflow_p = true;
if (INTEGRAL_TYPE_P (TREE_TYPE (cst0))
&& ((sgn0 == 1
&& TYPE_MIN_VALUE (TREE_TYPE (cst0))
&& tree_int_cst_equal (cst0, TYPE_MIN_VALUE (TREE_TYPE (cst0))))
|| (sgn0 == -1
&& TYPE_MAX_VALUE (TREE_TYPE (cst0))
&& tree_int_cst_equal (cst0, TYPE_MAX_VALUE (TREE_TYPE (cst0))))))
return NULL_TREE;
t = int_const_binop (sgn0 == -1 ? PLUS_EXPR : MINUS_EXPR,
cst0, build_int_cst (TREE_TYPE (cst0), 1));
t = fold_build2_loc (loc, code0, TREE_TYPE (arg0), TREE_OPERAND (arg0, 0), t);
t = fold_convert (TREE_TYPE (arg1), t);
return fold_build2_loc (loc, code, type, t, arg1);
}
static tree
maybe_canonicalize_comparison (location_t loc, enum tree_code code, tree type,
tree arg0, tree arg1)
{
tree t;
bool strict_overflow_p;
const char * const warnmsg = G_("assuming signed overflow does not occur "
"when reducing constant in comparison");
strict_overflow_p = false;
t = maybe_canonicalize_comparison_1 (loc, code, type, arg0, arg1,
&strict_overflow_p);
if (t)
{
if (strict_overflow_p)
fold_overflow_warning (warnmsg, WARN_STRICT_OVERFLOW_MAGNITUDE);
return t;
}
code = swap_tree_comparison (code);
strict_overflow_p = false;
t = maybe_canonicalize_comparison_1 (loc, code, type, arg1, arg0,
&strict_overflow_p);
if (t && strict_overflow_p)
fold_overflow_warning (warnmsg, WARN_STRICT_OVERFLOW_MAGNITUDE);
return t;
}
static bool
pointer_may_wrap_p (tree base, tree offset, poly_int64 bitpos)
{
if (!POINTER_TYPE_P (TREE_TYPE (base)))
return true;
if (maybe_lt (bitpos, 0))
return true;
poly_wide_int wi_offset;
int precision = TYPE_PRECISION (TREE_TYPE (base));
if (offset == NULL_TREE)
wi_offset = wi::zero (precision);
else if (!poly_int_tree_p (offset) || TREE_OVERFLOW (offset))
return true;
else
wi_offset = wi::to_poly_wide (offset);
bool overflow;
poly_wide_int units = wi::shwi (bits_to_bytes_round_down (bitpos),
precision);
poly_wide_int total = wi::add (wi_offset, units, UNSIGNED, &overflow);
if (overflow)
return true;
poly_uint64 total_hwi, size;
if (!total.to_uhwi (&total_hwi)
|| !poly_int_tree_p (TYPE_SIZE_UNIT (TREE_TYPE (TREE_TYPE (base))),
&size)
|| known_eq (size, 0U))
return true;
if (known_le (total_hwi, size))
return false;
if (TREE_CODE (base) == ADDR_EXPR
&& poly_int_tree_p (TYPE_SIZE_UNIT (TREE_TYPE (TREE_OPERAND (base, 0))),
&size)
&& maybe_ne (size, 0U)
&& known_le (total_hwi, size))
return false;
return true;
}
static int
maybe_nonzero_address (tree decl)
{
if (DECL_P (decl) && decl_in_symtab_p (decl))
if (struct symtab_node *symbol = symtab_node::get_create (decl))
return symbol->nonzero_address ();
if (DECL_P (decl)
&& (DECL_CONTEXT (decl)
&& TREE_CODE (DECL_CONTEXT (decl)) == FUNCTION_DECL
&& auto_var_in_fn_p (decl, DECL_CONTEXT (decl))))
return 1;
return -1;
}
static tree
fold_comparison (location_t loc, enum tree_code code, tree type,
tree op0, tree op1)
{
const bool equality_code = (code == EQ_EXPR || code == NE_EXPR);
tree arg0, arg1, tem;
arg0 = op0;
arg1 = op1;
STRIP_SIGN_NOPS (arg0);
STRIP_SIGN_NOPS (arg1);
if (POINTER_TYPE_P (TREE_TYPE (arg0))
&& (TREE_CODE (arg0) == ADDR_EXPR
|| TREE_CODE (arg1) == ADDR_EXPR
|| TREE_CODE (arg0) == POINTER_PLUS_EXPR
|| TREE_CODE (arg1) == POINTER_PLUS_EXPR))
{
tree base0, base1, offset0 = NULL_TREE, offset1 = NULL_TREE;
poly_int64 bitsize, bitpos0 = 0, bitpos1 = 0;
machine_mode mode;
int volatilep, reversep, unsignedp;
bool indirect_base0 = false, indirect_base1 = false;
base0 = arg0;
if (TREE_CODE (arg0) == ADDR_EXPR)
{
base0
= get_inner_reference (TREE_OPERAND (arg0, 0),
&bitsize, &bitpos0, &offset0, &mode,
&unsignedp, &reversep, &volatilep);
if (TREE_CODE (base0) == INDIRECT_REF)
base0 = TREE_OPERAND (base0, 0);
else
indirect_base0 = true;
}
else if (TREE_CODE (arg0) == POINTER_PLUS_EXPR)
{
base0 = TREE_OPERAND (arg0, 0);
STRIP_SIGN_NOPS (base0);
if (TREE_CODE (base0) == ADDR_EXPR)
{
base0
= get_inner_reference (TREE_OPERAND (base0, 0),
&bitsize, &bitpos0, &offset0, &mode,
&unsignedp, &reversep, &volatilep);
if (TREE_CODE (base0) == INDIRECT_REF)
base0 = TREE_OPERAND (base0, 0);
else
indirect_base0 = true;
}
if (offset0 == NULL_TREE || integer_zerop (offset0))
offset0 = TREE_OPERAND (arg0, 1);
else
offset0 = size_binop (PLUS_EXPR, offset0,
TREE_OPERAND (arg0, 1));
if (poly_int_tree_p (offset0))
{
poly_offset_int tem = wi::sext (wi::to_poly_offset (offset0),
TYPE_PRECISION (sizetype));
tem <<= LOG2_BITS_PER_UNIT;
tem += bitpos0;
if (tem.to_shwi (&bitpos0))
offset0 = NULL_TREE;
}
}
base1 = arg1;
if (TREE_CODE (arg1) == ADDR_EXPR)
{
base1
= get_inner_reference (TREE_OPERAND (arg1, 0),
&bitsize, &bitpos1, &offset1, &mode,
&unsignedp, &reversep, &volatilep);
if (TREE_CODE (base1) == INDIRECT_REF)
base1 = TREE_OPERAND (base1, 0);
else
indirect_base1 = true;
}
else if (TREE_CODE (arg1) == POINTER_PLUS_EXPR)
{
base1 = TREE_OPERAND (arg1, 0);
STRIP_SIGN_NOPS (base1);
if (TREE_CODE (base1) == ADDR_EXPR)
{
base1
= get_inner_reference (TREE_OPERAND (base1, 0),
&bitsize, &bitpos1, &offset1, &mode,
&unsignedp, &reversep, &volatilep);
if (TREE_CODE (base1) == INDIRECT_REF)
base1 = TREE_OPERAND (base1, 0);
else
indirect_base1 = true;
}
if (offset1 == NULL_TREE || integer_zerop (offset1))
offset1 = TREE_OPERAND (arg1, 1);
else
offset1 = size_binop (PLUS_EXPR, offset1,
TREE_OPERAND (arg1, 1));
if (poly_int_tree_p (offset1))
{
poly_offset_int tem = wi::sext (wi::to_poly_offset (offset1),
TYPE_PRECISION (sizetype));
tem <<= LOG2_BITS_PER_UNIT;
tem += bitpos1;
if (tem.to_shwi (&bitpos1))
offset1 = NULL_TREE;
}
}
if (indirect_base0 == indirect_base1
&& operand_equal_p (base0, base1,
indirect_base0 ? OEP_ADDRESS_OF : 0))
{
if ((offset0 == offset1
|| (offset0 && offset1
&& operand_equal_p (offset0, offset1, 0)))
&& (equality_code
|| (indirect_base0
&& (DECL_P (base0) || CONSTANT_CLASS_P (base0)))
|| TYPE_OVERFLOW_UNDEFINED (TREE_TYPE (arg0))))
{
if (!equality_code
&& maybe_ne (bitpos0, bitpos1)
&& (pointer_may_wrap_p (base0, offset0, bitpos0)
|| pointer_may_wrap_p (base1, offset1, bitpos1)))
fold_overflow_warning (("assuming pointer wraparound does not "
"occur when comparing P +- C1 with "
"P +- C2"),
WARN_STRICT_OVERFLOW_CONDITIONAL);
switch (code)
{
case EQ_EXPR:
if (known_eq (bitpos0, bitpos1))
return constant_boolean_node (true, type);
if (known_ne (bitpos0, bitpos1))
return constant_boolean_node (false, type);
break;
case NE_EXPR:
if (known_ne (bitpos0, bitpos1))
return constant_boolean_node (true, type);
if (known_eq (bitpos0, bitpos1))
return constant_boolean_node (false, type);
break;
case LT_EXPR:
if (known_lt (bitpos0, bitpos1))
return constant_boolean_node (true, type);
if (known_ge (bitpos0, bitpos1))
return constant_boolean_node (false, type);
break;
case LE_EXPR:
if (known_le (bitpos0, bitpos1))
return constant_boolean_node (true, type);
if (known_gt (bitpos0, bitpos1))
return constant_boolean_node (false, type);
break;
case GE_EXPR:
if (known_ge (bitpos0, bitpos1))
return constant_boolean_node (true, type);
if (known_lt (bitpos0, bitpos1))
return constant_boolean_node (false, type);
break;
case GT_EXPR:
if (known_gt (bitpos0, bitpos1))
return constant_boolean_node (true, type);
if (known_le (bitpos0, bitpos1))
return constant_boolean_node (false, type);
break;
default:;
}
}
else if (known_eq (bitpos0, bitpos1)
&& (equality_code
|| (indirect_base0
&& (DECL_P (base0) || CONSTANT_CLASS_P (base0)))
|| TYPE_OVERFLOW_UNDEFINED (TREE_TYPE (arg0))))
{
if (offset0 == NULL_TREE)
offset0 = build_int_cst (ssizetype, 0);
else
offset0 = fold_convert_loc (loc, ssizetype, offset0);
if (offset1 == NULL_TREE)
offset1 = build_int_cst (ssizetype, 0);
else
offset1 = fold_convert_loc (loc, ssizetype, offset1);
if (!equality_code
&& (pointer_may_wrap_p (base0, offset0, bitpos0)
|| pointer_may_wrap_p (base1, offset1, bitpos1)))
fold_overflow_warning (("assuming pointer wraparound does not "
"occur when comparing P +- C1 with "
"P +- C2"),
WARN_STRICT_OVERFLOW_COMPARISON);
return fold_build2_loc (loc, code, type, offset0, offset1);
}
}
else if (known_eq (bitpos0, bitpos1)
&& (indirect_base0
? base0 != TREE_OPERAND (arg0, 0) : base0 != arg0)
&& (indirect_base1
? base1 != TREE_OPERAND (arg1, 0) : base1 != arg1)
&& ((offset0 == offset1)
|| (offset0 && offset1
&& operand_equal_p (offset0, offset1, 0))))
{
if (indirect_base0)
base0 = build_fold_addr_expr_loc (loc, base0);
if (indirect_base1)
base1 = build_fold_addr_expr_loc (loc, base1);
return fold_build2_loc (loc, code, type, base0, base1);
}
else if (((DECL_P (base0)
&& maybe_nonzero_address (base0) > 0
&& (offset0 == NULL_TREE && known_ne (bitpos0, 0)))
|| CONSTANT_CLASS_P (base0))
&& indirect_base0
&& integer_zerop (arg1))
{
switch (code)
{
case EQ_EXPR:
case LE_EXPR:
case LT_EXPR:
return constant_boolean_node (false, type);
case GE_EXPR:
case GT_EXPR:
case NE_EXPR:
return constant_boolean_node (true, type);
default:
gcc_unreachable ();
}
}
}
if (ANY_INTEGRAL_TYPE_P (TREE_TYPE (arg0))
&& TYPE_OVERFLOW_UNDEFINED (TREE_TYPE (arg0))
&& (TREE_CODE (arg0) == PLUS_EXPR || TREE_CODE (arg0) == MINUS_EXPR)
&& (TREE_CODE (TREE_OPERAND (arg0, 1)) == INTEGER_CST
&& !TREE_OVERFLOW (TREE_OPERAND (arg0, 1)))
&& (TREE_CODE (arg1) == PLUS_EXPR || TREE_CODE (arg1) == MINUS_EXPR)
&& (TREE_CODE (TREE_OPERAND (arg1, 1)) == INTEGER_CST
&& !TREE_OVERFLOW (TREE_OPERAND (arg1, 1))))
{
tree const1 = TREE_OPERAND (arg0, 1);
tree const2 = TREE_OPERAND (arg1, 1);
tree variable1 = TREE_OPERAND (arg0, 0);
tree variable2 = TREE_OPERAND (arg1, 0);
tree cst;
const char * const warnmsg = G_("assuming signed overflow does not "
"occur when combining constants around "
"a comparison");
cst = int_const_binop (TREE_CODE (arg0) == TREE_CODE (arg1)
? MINUS_EXPR : PLUS_EXPR,
const2, const1);
if (!TREE_OVERFLOW (cst)
&& tree_int_cst_compare (const2, cst) == tree_int_cst_sgn (const2)
&& tree_int_cst_sgn (cst) == tree_int_cst_sgn (const2))
{
fold_overflow_warning (warnmsg, WARN_STRICT_OVERFLOW_COMPARISON);
return fold_build2_loc (loc, code, type,
variable1,
fold_build2_loc (loc, TREE_CODE (arg1),
TREE_TYPE (arg1),
variable2, cst));
}
cst = int_const_binop (TREE_CODE (arg0) == TREE_CODE (arg1)
? MINUS_EXPR : PLUS_EXPR,
const1, const2);
if (!TREE_OVERFLOW (cst)
&& tree_int_cst_compare (const1, cst) == tree_int_cst_sgn (const1)
&& tree_int_cst_sgn (cst) == tree_int_cst_sgn (const1))
{
fold_overflow_warning (warnmsg, WARN_STRICT_OVERFLOW_COMPARISON);
return fold_build2_loc (loc, code, type,
fold_build2_loc (loc, TREE_CODE (arg0),
TREE_TYPE (arg0),
variable1, cst),
variable2);
}
}
tem = maybe_canonicalize_comparison (loc, code, type, arg0, arg1);
if (tem)
return tem;
if (TREE_CODE (arg1) == INTEGER_CST && TREE_CODE (arg0) != INTEGER_CST)
{
tree cval1 = 0, cval2 = 0;
if (twoval_comparison_p (arg0, &cval1, &cval2)
&& cval1 != 0 && cval2 != 0
&& ! (TREE_CONSTANT (cval1) && TREE_CONSTANT (cval2))
&& TREE_TYPE (cval1) == TREE_TYPE (cval2)
&& INTEGRAL_TYPE_P (TREE_TYPE (cval1))
&& TYPE_MAX_VALUE (TREE_TYPE (cval1))
&& TYPE_MAX_VALUE (TREE_TYPE (cval2))
&& ! operand_equal_p (TYPE_MIN_VALUE (TREE_TYPE (cval1)),
TYPE_MAX_VALUE (TREE_TYPE (cval2)), 0))
{
tree maxval = TYPE_MAX_VALUE (TREE_TYPE (cval1));
tree minval = TYPE_MIN_VALUE (TREE_TYPE (cval1));
tree high_result
= fold_build2_loc (loc, code, type,
eval_subst (loc, arg0, cval1, maxval,
cval2, minval),
arg1);
tree equal_result
= fold_build2_loc (loc, code, type,
eval_subst (loc, arg0, cval1, maxval,
cval2, maxval),
arg1);
tree low_result
= fold_build2_loc (loc, code, type,
eval_subst (loc, arg0, cval1, minval,
cval2, maxval),
arg1);
if (TREE_CODE (high_result) == INTEGER_CST
&& TREE_CODE (equal_result) == INTEGER_CST
&& TREE_CODE (low_result) == INTEGER_CST)
{
switch ((integer_onep (high_result) * 4)
+ (integer_onep (equal_result) * 2)
+ integer_onep (low_result))
{
case 0:
return omit_one_operand_loc (loc, type, integer_zero_node, arg0);
case 1:
code = LT_EXPR;
break;
case 2:
code = EQ_EXPR;
break;
case 3:
code = LE_EXPR;
break;
case 4:
code = GT_EXPR;
break;
case 5:
code = NE_EXPR;
break;
case 6:
code = GE_EXPR;
break;
case 7:
return omit_one_operand_loc (loc, type, integer_one_node, arg0);
}
return fold_build2_loc (loc, code, type, cval1, cval2);
}
}
}
return NULL_TREE;
}
static tree
fold_mult_zconjz (location_t loc, tree type, tree expr)
{
tree itype = TREE_TYPE (type);
tree rpart, ipart, tem;
if (TREE_CODE (expr) == COMPLEX_EXPR)
{
rpart = TREE_OPERAND (expr, 0);
ipart = TREE_OPERAND (expr, 1);
}
else if (TREE_CODE (expr) == COMPLEX_CST)
{
rpart = TREE_REALPART (expr);
ipart = TREE_IMAGPART (expr);
}
else
{
expr = save_expr (expr);
rpart = fold_build1_loc (loc, REALPART_EXPR, itype, expr);
ipart = fold_build1_loc (loc, IMAGPART_EXPR, itype, expr);
}
rpart = save_expr (rpart);
ipart = save_expr (ipart);
tem = fold_build2_loc (loc, PLUS_EXPR, itype,
fold_build2_loc (loc, MULT_EXPR, itype, rpart, rpart),
fold_build2_loc (loc, MULT_EXPR, itype, ipart, ipart));
return fold_build2_loc (loc, COMPLEX_EXPR, type, tem,
build_zero_cst (itype));
}
static bool
vec_cst_ctor_to_array (tree arg, unsigned int nelts, tree *elts)
{
unsigned HOST_WIDE_INT i, nunits;
if (TREE_CODE (arg) == VECTOR_CST
&& VECTOR_CST_NELTS (arg).is_constant (&nunits))
{
for (i = 0; i < nunits; ++i)
elts[i] = VECTOR_CST_ELT (arg, i);
}
else if (TREE_CODE (arg) == CONSTRUCTOR)
{
constructor_elt *elt;
FOR_EACH_VEC_SAFE_ELT (CONSTRUCTOR_ELTS (arg), i, elt)
if (i >= nelts || TREE_CODE (TREE_TYPE (elt->value)) == VECTOR_TYPE)
return false;
else
elts[i] = elt->value;
}
else
return false;
for (; i < nelts; i++)
elts[i]
= fold_convert (TREE_TYPE (TREE_TYPE (arg)), integer_zero_node);
return true;
}
static tree
fold_vec_perm (tree type, tree arg0, tree arg1, const vec_perm_indices &sel)
{
unsigned int i;
unsigned HOST_WIDE_INT nelts;
bool need_ctor = false;
if (!sel.length ().is_constant (&nelts))
return NULL_TREE;
gcc_assert (known_eq (TYPE_VECTOR_SUBPARTS (type), nelts)
&& known_eq (TYPE_VECTOR_SUBPARTS (TREE_TYPE (arg0)), nelts)
&& known_eq (TYPE_VECTOR_SUBPARTS (TREE_TYPE (arg1)), nelts));
if (TREE_TYPE (TREE_TYPE (arg0)) != TREE_TYPE (type)
|| TREE_TYPE (TREE_TYPE (arg1)) != TREE_TYPE (type))
return NULL_TREE;
tree *in_elts = XALLOCAVEC (tree, nelts * 2);
if (!vec_cst_ctor_to_array (arg0, nelts, in_elts)
|| !vec_cst_ctor_to_array (arg1, nelts, in_elts + nelts))
return NULL_TREE;
tree_vector_builder out_elts (type, nelts, 1);
for (i = 0; i < nelts; i++)
{
HOST_WIDE_INT index;
if (!sel[i].is_constant (&index))
return NULL_TREE;
if (!CONSTANT_CLASS_P (in_elts[index]))
need_ctor = true;
out_elts.quick_push (unshare_expr (in_elts[index]));
}
if (need_ctor)
{
vec<constructor_elt, va_gc> *v;
vec_alloc (v, nelts);
for (i = 0; i < nelts; i++)
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE, out_elts[i]);
return build_constructor (type, v);
}
else
return out_elts.build ();
}
static tree
fold_addr_of_array_ref_difference (location_t loc, tree type,
tree aref0, tree aref1,
bool use_pointer_diff)
{
tree base0 = TREE_OPERAND (aref0, 0);
tree base1 = TREE_OPERAND (aref1, 0);
tree base_offset = build_int_cst (type, 0);
if ((TREE_CODE (base0) == ARRAY_REF
&& TREE_CODE (base1) == ARRAY_REF
&& (base_offset
= fold_addr_of_array_ref_difference (loc, type, base0, base1,
use_pointer_diff)))
|| (INDIRECT_REF_P (base0)
&& INDIRECT_REF_P (base1)
&& (base_offset
= use_pointer_diff
? fold_binary_loc (loc, POINTER_DIFF_EXPR, type,
TREE_OPERAND (base0, 0),
TREE_OPERAND (base1, 0))
: fold_binary_loc (loc, MINUS_EXPR, type,
fold_convert (type,
TREE_OPERAND (base0, 0)),
fold_convert (type,
TREE_OPERAND (base1, 0)))))
|| operand_equal_p (base0, base1, OEP_ADDRESS_OF))
{
tree op0 = fold_convert_loc (loc, type, TREE_OPERAND (aref0, 1));
tree op1 = fold_convert_loc (loc, type, TREE_OPERAND (aref1, 1));
tree esz = fold_convert_loc (loc, type, array_ref_element_size (aref0));
tree diff = fold_build2_loc (loc, MINUS_EXPR, type, op0, op1);
return fold_build2_loc (loc, PLUS_EXPR, type,
base_offset,
fold_build2_loc (loc, MULT_EXPR, type,
diff, esz));
}
return NULL_TREE;
}
tree
exact_inverse (tree type, tree cst)
{
REAL_VALUE_TYPE r;
tree unit_type;
machine_mode mode;
switch (TREE_CODE (cst))
{
case REAL_CST:
r = TREE_REAL_CST (cst);
if (exact_real_inverse (TYPE_MODE (type), &r))
return build_real (type, r);
return NULL_TREE;
case VECTOR_CST:
{
unit_type = TREE_TYPE (type);
mode = TYPE_MODE (unit_type);
tree_vector_builder elts;
if (!elts.new_unary_operation (type, cst, false))
return NULL_TREE;
unsigned int count = elts.encoded_nelts ();
for (unsigned int i = 0; i < count; ++i)
{
r = TREE_REAL_CST (VECTOR_CST_ELT (cst, i));
if (!exact_real_inverse (mode, &r))
return NULL_TREE;
elts.quick_push (build_real (unit_type, r));
}
return elts.build ();
}
default:
return NULL_TREE;
}
}
static wide_int
mask_with_tz (tree type, const wide_int &x, const wide_int &y)
{
int tz = wi::ctz (y);
if (tz > 0)
return wi::mask (tz, true, TYPE_PRECISION (type)) & x;
return x;
}
static bool
tree_expr_nonzero_warnv_p (tree t, bool *strict_overflow_p)
{
tree type = TREE_TYPE (t);
enum tree_code code;
if (!INTEGRAL_TYPE_P (type) && !POINTER_TYPE_P (type))
return false;
code = TREE_CODE (t);
switch (TREE_CODE_CLASS (code))
{
case tcc_unary:
return tree_unary_nonzero_warnv_p (code, type, TREE_OPERAND (t, 0),
strict_overflow_p);
case tcc_binary:
case tcc_comparison:
return tree_binary_nonzero_warnv_p (code, type,
TREE_OPERAND (t, 0),
TREE_OPERAND (t, 1),
strict_overflow_p);
case tcc_constant:
case tcc_declaration:
case tcc_reference:
return tree_single_nonzero_warnv_p (t, strict_overflow_p);
default:
break;
}
switch (code)
{
case TRUTH_NOT_EXPR:
return tree_unary_nonzero_warnv_p (code, type, TREE_OPERAND (t, 0),
strict_overflow_p);
case TRUTH_AND_EXPR:
case TRUTH_OR_EXPR:
case TRUTH_XOR_EXPR:
return tree_binary_nonzero_warnv_p (code, type,
TREE_OPERAND (t, 0),
TREE_OPERAND (t, 1),
strict_overflow_p);
case COND_EXPR:
case CONSTRUCTOR:
case OBJ_TYPE_REF:
case ASSERT_EXPR:
case ADDR_EXPR:
case WITH_SIZE_EXPR:
case SSA_NAME:
return tree_single_nonzero_warnv_p (t, strict_overflow_p);
case COMPOUND_EXPR:
case MODIFY_EXPR:
case BIND_EXPR:
return tree_expr_nonzero_warnv_p (TREE_OPERAND (t, 1),
strict_overflow_p);
case SAVE_EXPR:
return tree_expr_nonzero_warnv_p (TREE_OPERAND (t, 0),
strict_overflow_p);
case CALL_EXPR:
{
tree fndecl = get_callee_fndecl (t);
if (!fndecl) return false;
if (flag_delete_null_pointer_checks && !flag_check_new
&& DECL_IS_OPERATOR_NEW (fndecl)
&& !TREE_NOTHROW (fndecl))
return true;
if (flag_delete_null_pointer_checks
&& lookup_attribute ("returns_nonnull",
TYPE_ATTRIBUTES (TREE_TYPE (fndecl))))
return true;
return alloca_call_p (t);
}
default:
break;
}
return false;
}
bool
tree_expr_nonzero_p (tree t)
{
bool ret, strict_overflow_p;
strict_overflow_p = false;
ret = tree_expr_nonzero_warnv_p (t, &strict_overflow_p);
if (strict_overflow_p)
fold_overflow_warning (("assuming signed overflow does not occur when "
"determining that expression is always "
"non-zero"),
WARN_STRICT_OVERFLOW_MISC);
return ret;
}
bool
expr_not_equal_to (tree t, const wide_int &w)
{
wide_int min, max, nz;
value_range_type rtype;
switch (TREE_CODE (t))
{
case INTEGER_CST:
return wi::to_wide (t) != w;
case SSA_NAME:
if (!INTEGRAL_TYPE_P (TREE_TYPE (t)))
return false;
rtype = get_range_info (t, &min, &max);
if (rtype == VR_RANGE)
{
if (wi::lt_p (max, w, TYPE_SIGN (TREE_TYPE (t))))
return true;
if (wi::lt_p (w, min, TYPE_SIGN (TREE_TYPE (t))))
return true;
}
else if (rtype == VR_ANTI_RANGE
&& wi::le_p (min, w, TYPE_SIGN (TREE_TYPE (t)))
&& wi::le_p (w, max, TYPE_SIGN (TREE_TYPE (t))))
return true;
if (wi::ne_p (wi::zext (wi::bit_and_not (w, get_nonzero_bits (t)),
TYPE_PRECISION (TREE_TYPE (t))), 0))
return true;
return false;
default:
return false;
}
}
tree
fold_binary_loc (location_t loc, enum tree_code code, tree type,
tree op0, tree op1)
{
enum tree_code_class kind = TREE_CODE_CLASS (code);
tree arg0, arg1, tem;
tree t1 = NULL_TREE;
bool strict_overflow_p;
unsigned int prec;
gcc_assert (IS_EXPR_CODE_CLASS (kind)
&& TREE_CODE_LENGTH (code) == 2
&& op0 != NULL_TREE
&& op1 != NULL_TREE);
arg0 = op0;
arg1 = op1;
if (kind == tcc_comparison || code == MIN_EXPR || code == MAX_EXPR)
{
STRIP_SIGN_NOPS (arg0);
STRIP_SIGN_NOPS (arg1);
}
else
{
STRIP_NOPS (arg0);
STRIP_NOPS (arg1);
}
if (CONSTANT_CLASS_P (arg0) && CONSTANT_CLASS_P (arg1))
{
tem = const_binop (code, type, arg0, arg1);
if (tem != NULL_TREE)
{
if (TREE_TYPE (tem) != type)
tem = fold_convert_loc (loc, type, tem);
return tem;
}
}
if (commutative_tree_code (code)
&& tree_swap_operands_p (arg0, arg1))
return fold_build2_loc (loc, code, type, op1, op0);
if (kind == tcc_comparison
&& tree_swap_operands_p (arg0, arg1))
return fold_build2_loc (loc, swap_tree_comparison (code), type, op1, op0);
tem = generic_simplify (loc, code, type, op0, op1);
if (tem)
return tem;
if ((code == BIT_AND_EXPR || code == BIT_IOR_EXPR
|| code == EQ_EXPR || code == NE_EXPR)
&& !VECTOR_TYPE_P (TREE_TYPE (arg0))
&& ((truth_value_p (TREE_CODE (arg0))
&& (truth_value_p (TREE_CODE (arg1))
|| (TREE_CODE (arg1) == BIT_AND_EXPR
&& integer_onep (TREE_OPERAND (arg1, 1)))))
|| (truth_value_p (TREE_CODE (arg1))
&& (truth_value_p (TREE_CODE (arg0))
|| (TREE_CODE (arg0) == BIT_AND_EXPR
&& integer_onep (TREE_OPERAND (arg0, 1)))))))
{
tem = fold_build2_loc (loc, code == BIT_AND_EXPR ? TRUTH_AND_EXPR
: code == BIT_IOR_EXPR ? TRUTH_OR_EXPR
: TRUTH_XOR_EXPR,
boolean_type_node,
fold_convert_loc (loc, boolean_type_node, arg0),
fold_convert_loc (loc, boolean_type_node, arg1));
if (code == EQ_EXPR)
tem = invert_truthvalue_loc (loc, tem);
return fold_convert_loc (loc, type, tem);
}
if (TREE_CODE_CLASS (code) == tcc_binary
|| TREE_CODE_CLASS (code) == tcc_comparison)
{
if (TREE_CODE (arg0) == COMPOUND_EXPR)
{
tem = fold_build2_loc (loc, code, type,
fold_convert_loc (loc, TREE_TYPE (op0),
TREE_OPERAND (arg0, 1)), op1);
return build2_loc (loc, COMPOUND_EXPR, type, TREE_OPERAND (arg0, 0),
tem);
}
if (TREE_CODE (arg1) == COMPOUND_EXPR)
{
tem = fold_build2_loc (loc, code, type, op0,
fold_convert_loc (loc, TREE_TYPE (op1),
TREE_OPERAND (arg1, 1)));
return build2_loc (loc, COMPOUND_EXPR, type, TREE_OPERAND (arg1, 0),
tem);
}
if (TREE_CODE (arg0) == COND_EXPR
|| TREE_CODE (arg0) == VEC_COND_EXPR
|| COMPARISON_CLASS_P (arg0))
{
tem = fold_binary_op_with_conditional_arg (loc, code, type, op0, op1,
arg0, arg1,
1);
if (tem != NULL_TREE)
return tem;
}
if (TREE_CODE (arg1) == COND_EXPR
|| TREE_CODE (arg1) == VEC_COND_EXPR
|| COMPARISON_CLASS_P (arg1))
{
tem = fold_binary_op_with_conditional_arg (loc, code, type, op0, op1,
arg1, arg0,
0);
if (tem != NULL_TREE)
return tem;
}
}
switch (code)
{
case MEM_REF:
if (TREE_CODE (arg0) == ADDR_EXPR
&& TREE_CODE (TREE_OPERAND (arg0, 0)) == MEM_REF)
{
tree iref = TREE_OPERAND (arg0, 0);
return fold_build2 (MEM_REF, type,
TREE_OPERAND (iref, 0),
int_const_binop (PLUS_EXPR, arg1,
TREE_OPERAND (iref, 1)));
}
if (TREE_CODE (arg0) == ADDR_EXPR
&& handled_component_p (TREE_OPERAND (arg0, 0)))
{
tree base;
poly_int64 coffset;
base = get_addr_base_and_unit_offset (TREE_OPERAND (arg0, 0),
&coffset);
if (!base)
return NULL_TREE;
return fold_build2 (MEM_REF, type,
build_fold_addr_expr (base),
int_const_binop (PLUS_EXPR, arg1,
size_int (coffset)));
}
return NULL_TREE;
case POINTER_PLUS_EXPR:
if (INTEGRAL_TYPE_P (TREE_TYPE (arg1))
&& INTEGRAL_TYPE_P (TREE_TYPE (arg0)))
return fold_convert_loc (loc, type,
fold_build2_loc (loc, PLUS_EXPR, sizetype,
fold_convert_loc (loc, sizetype,
arg1),
fold_convert_loc (loc, sizetype,
arg0)));
return NULL_TREE;
case PLUS_EXPR:
if (INTEGRAL_TYPE_P (type) || VECTOR_INTEGER_TYPE_P (type))
{
if (TREE_CODE (arg1) == MULT_EXPR
&& TREE_CODE (TREE_OPERAND (arg1, 0)) == TRUNC_DIV_EXPR
&& operand_equal_p (arg0,
TREE_OPERAND (TREE_OPERAND (arg1, 0), 0), 0))
{
tree cst0 = TREE_OPERAND (TREE_OPERAND (arg1, 0), 1);
tree cst1 = TREE_OPERAND (arg1, 1);
tree sum = fold_binary_loc (loc, PLUS_EXPR, TREE_TYPE (cst1),
cst1, cst0);
if (sum && integer_zerop (sum))
return fold_convert_loc (loc, type,
fold_build2_loc (loc, TRUNC_MOD_EXPR,
TREE_TYPE (arg0), arg0,
cst0));
}
}
if ((TREE_CODE (arg0) == MULT_EXPR
|| TREE_CODE (arg1) == MULT_EXPR)
&& !TYPE_SATURATING (type)
&& TYPE_UNSIGNED (type) == TYPE_UNSIGNED (TREE_TYPE (arg0))
&& TYPE_UNSIGNED (type) == TYPE_UNSIGNED (TREE_TYPE (arg1))
&& (!FLOAT_TYPE_P (type) || flag_associative_math))
{
tree tem = fold_plusminus_mult_expr (loc, code, type, arg0, arg1);
if (tem)
return tem;
}
if (! FLOAT_TYPE_P (type))
{
if (ANY_INTEGRAL_TYPE_P (type)
&& TYPE_OVERFLOW_WRAPS (type)
&& (((TREE_CODE (arg0) == PLUS_EXPR
|| TREE_CODE (arg0) == MINUS_EXPR)
&& TREE_CODE (arg1) == MULT_EXPR)
|| ((TREE_CODE (arg1) == PLUS_EXPR
|| TREE_CODE (arg1) == MINUS_EXPR)
&& TREE_CODE (arg0) == MULT_EXPR)))
{
tree parg0, parg1, parg, marg;
enum tree_code pcode;
if (TREE_CODE (arg1) == MULT_EXPR)
parg = arg0, marg = arg1;
else
parg = arg1, marg = arg0;
pcode = TREE_CODE (parg);
parg0 = TREE_OPERAND (parg, 0);
parg1 = TREE_OPERAND (parg, 1);
STRIP_NOPS (parg0);
STRIP_NOPS (parg1);
if (TREE_CODE (parg0) == MULT_EXPR
&& TREE_CODE (parg1) != MULT_EXPR)
return fold_build2_loc (loc, pcode, type,
fold_build2_loc (loc, PLUS_EXPR, type,
fold_convert_loc (loc, type,
parg0),
fold_convert_loc (loc, type,
marg)),
fold_convert_loc (loc, type, parg1));
if (TREE_CODE (parg0) != MULT_EXPR
&& TREE_CODE (parg1) == MULT_EXPR)
return
fold_build2_loc (loc, PLUS_EXPR, type,
fold_convert_loc (loc, type, parg0),
fold_build2_loc (loc, pcode, type,
fold_convert_loc (loc, type, marg),
fold_convert_loc (loc, type,
parg1)));
}
}
else
{
if (!HONOR_SNANS (element_mode (arg0))
&& !HONOR_SIGNED_ZEROS (element_mode (arg0))
&& COMPLEX_FLOAT_TYPE_P (TREE_TYPE (arg0)))
{
tree rtype = TREE_TYPE (TREE_TYPE (arg0));
tree arg0r = fold_unary_loc (loc, REALPART_EXPR, rtype, arg0);
tree arg0i = fold_unary_loc (loc, IMAGPART_EXPR, rtype, arg0);
bool arg0rz = false, arg0iz = false;
if ((arg0r && (arg0rz = real_zerop (arg0r)))
|| (arg0i && (arg0iz = real_zerop (arg0i))))
{
tree arg1r = fold_unary_loc (loc, REALPART_EXPR, rtype, arg1);
tree arg1i = fold_unary_loc (loc, IMAGPART_EXPR, rtype, arg1);
if (arg0rz && arg1i && real_zerop (arg1i))
{
tree rp = arg1r ? arg1r
: build1 (REALPART_EXPR, rtype, arg1);
tree ip = arg0i ? arg0i
: build1 (IMAGPART_EXPR, rtype, arg0);
return fold_build2_loc (loc, COMPLEX_EXPR, type, rp, ip);
}
else if (arg0iz && arg1r && real_zerop (arg1r))
{
tree rp = arg0r ? arg0r
: build1 (REALPART_EXPR, rtype, arg0);
tree ip = arg1i ? arg1i
: build1 (IMAGPART_EXPR, rtype, arg1);
return fold_build2_loc (loc, COMPLEX_EXPR, type, rp, ip);
}
}
}
if (flag_associative_math
&& TREE_CODE (arg1) == PLUS_EXPR
&& TREE_CODE (arg0) != MULT_EXPR)
{
tree tree10 = TREE_OPERAND (arg1, 0);
tree tree11 = TREE_OPERAND (arg1, 1);
if (TREE_CODE (tree11) == MULT_EXPR
&& TREE_CODE (tree10) == MULT_EXPR)
{
tree tree0;
tree0 = fold_build2_loc (loc, PLUS_EXPR, type, arg0, tree10);
return fold_build2_loc (loc, PLUS_EXPR, type, tree0, tree11);
}
}
if (flag_associative_math
&& TREE_CODE (arg0) == PLUS_EXPR
&& TREE_CODE (arg1) != MULT_EXPR)
{
tree tree00 = TREE_OPERAND (arg0, 0);
tree tree01 = TREE_OPERAND (arg0, 1);
if (TREE_CODE (tree01) == MULT_EXPR
&& TREE_CODE (tree00) == MULT_EXPR)
{
tree tree0;
tree0 = fold_build2_loc (loc, PLUS_EXPR, type, tree01, arg1);
return fold_build2_loc (loc, PLUS_EXPR, type, tree00, tree0);
}
}
}
bit_rotate:
{
enum tree_code code0, code1;
tree rtype;
code0 = TREE_CODE (arg0);
code1 = TREE_CODE (arg1);
if (((code0 == RSHIFT_EXPR && code1 == LSHIFT_EXPR)
|| (code1 == RSHIFT_EXPR && code0 == LSHIFT_EXPR))
&& operand_equal_p (TREE_OPERAND (arg0, 0),
TREE_OPERAND (arg1, 0), 0)
&& (rtype = TREE_TYPE (TREE_OPERAND (arg0, 0)),
TYPE_UNSIGNED (rtype))
&& (element_precision (rtype)
== GET_MODE_UNIT_PRECISION (TYPE_MODE (rtype))))
{
tree tree01, tree11;
tree orig_tree01, orig_tree11;
enum tree_code code01, code11;
tree01 = orig_tree01 = TREE_OPERAND (arg0, 1);
tree11 = orig_tree11 = TREE_OPERAND (arg1, 1);
STRIP_NOPS (tree01);
STRIP_NOPS (tree11);
code01 = TREE_CODE (tree01);
code11 = TREE_CODE (tree11);
if (code11 != MINUS_EXPR
&& (code01 == MINUS_EXPR || code01 == BIT_AND_EXPR))
{
std::swap (code0, code1);
std::swap (code01, code11);
std::swap (tree01, tree11);
std::swap (orig_tree01, orig_tree11);
}
if (code01 == INTEGER_CST
&& code11 == INTEGER_CST
&& (wi::to_widest (tree01) + wi::to_widest (tree11)
== element_precision (rtype)))
{
tem = build2_loc (loc, LROTATE_EXPR,
rtype, TREE_OPERAND (arg0, 0),
code0 == LSHIFT_EXPR
? orig_tree01 : orig_tree11);
return fold_convert_loc (loc, type, tem);
}
else if (code11 == MINUS_EXPR)
{
tree tree110, tree111;
tree110 = TREE_OPERAND (tree11, 0);
tree111 = TREE_OPERAND (tree11, 1);
STRIP_NOPS (tree110);
STRIP_NOPS (tree111);
if (TREE_CODE (tree110) == INTEGER_CST
&& compare_tree_int (tree110,
element_precision (rtype)) == 0
&& operand_equal_p (tree01, tree111, 0))
{
tem = build2_loc (loc, (code0 == LSHIFT_EXPR
? LROTATE_EXPR : RROTATE_EXPR),
rtype, TREE_OPERAND (arg0, 0),
orig_tree01);
return fold_convert_loc (loc, type, tem);
}
}
else if (code == BIT_IOR_EXPR
&& code11 == BIT_AND_EXPR
&& pow2p_hwi (element_precision (rtype)))
{
tree tree110, tree111;
tree110 = TREE_OPERAND (tree11, 0);
tree111 = TREE_OPERAND (tree11, 1);
STRIP_NOPS (tree110);
STRIP_NOPS (tree111);
if (TREE_CODE (tree110) == NEGATE_EXPR
&& TREE_CODE (tree111) == INTEGER_CST
&& compare_tree_int (tree111,
element_precision (rtype) - 1) == 0
&& operand_equal_p (tree01, TREE_OPERAND (tree110, 0), 0))
{
tem = build2_loc (loc, (code0 == LSHIFT_EXPR
? LROTATE_EXPR : RROTATE_EXPR),
rtype, TREE_OPERAND (arg0, 0),
orig_tree01);
return fold_convert_loc (loc, type, tem);
}
}
}
}
associate:
if ((! FLOAT_TYPE_P (type) || flag_associative_math)
&& !TYPE_SATURATING (type))
{
tree var0, minus_var0, con0, minus_con0, lit0, minus_lit0;
tree var1, minus_var1, con1, minus_con1, lit1, minus_lit1;
tree atype = type;
bool ok = true;
var0 = split_tree (arg0, type, code,
&minus_var0, &con0, &minus_con0,
&lit0, &minus_lit0, 0);
var1 = split_tree (arg1, type, code,
&minus_var1, &con1, &minus_con1,
&lit1, &minus_lit1, code == MINUS_EXPR);
if (code == MINUS_EXPR)
code = PLUS_EXPR;
if ((POINTER_TYPE_P (type) || INTEGRAL_TYPE_P (type))
&& !TYPE_OVERFLOW_WRAPS (type))
{
if (INTEGRAL_TYPE_P (TREE_TYPE (arg0))
&& TYPE_OVERFLOW_WRAPS (TREE_TYPE (arg0)))
atype = TREE_TYPE (arg0);
else if (INTEGRAL_TYPE_P (TREE_TYPE (arg1))
&& TYPE_OVERFLOW_WRAPS (TREE_TYPE (arg1)))
atype = TREE_TYPE (arg1);
gcc_assert (TYPE_PRECISION (atype) == TYPE_PRECISION (type));
}
if ((POINTER_TYPE_P (atype) || INTEGRAL_TYPE_P (atype))
&& !TYPE_OVERFLOW_WRAPS (atype))
{
if ((var0 && var1) || (minus_var0 && minus_var1))
{
tree tmp0 = var0 ? var0 : minus_var0;
tree tmp1 = var1 ? var1 : minus_var1;
bool one_neg = false;
if (TREE_CODE (tmp0) == NEGATE_EXPR)
{
tmp0 = TREE_OPERAND (tmp0, 0);
one_neg = !one_neg;
}
if (CONVERT_EXPR_P (tmp0)
&& INTEGRAL_TYPE_P (TREE_TYPE (TREE_OPERAND (tmp0, 0)))
&& (TYPE_PRECISION (TREE_TYPE (TREE_OPERAND (tmp0, 0)))
<= TYPE_PRECISION (atype)))
tmp0 = TREE_OPERAND (tmp0, 0);
if (TREE_CODE (tmp1) == NEGATE_EXPR)
{
tmp1 = TREE_OPERAND (tmp1, 0);
one_neg = !one_neg;
}
if (CONVERT_EXPR_P (tmp1)
&& INTEGRAL_TYPE_P (TREE_TYPE (TREE_OPERAND (tmp1, 0)))
&& (TYPE_PRECISION (TREE_TYPE (TREE_OPERAND (tmp1, 0)))
<= TYPE_PRECISION (atype)))
tmp1 = TREE_OPERAND (tmp1, 0);
if (!one_neg
|| !operand_equal_p (tmp0, tmp1, 0))
ok = false;
}
else if ((var0 && minus_var1
&& ! operand_equal_p (var0, minus_var1, 0))
|| (minus_var0 && var1
&& ! operand_equal_p (minus_var0, var1, 0)))
ok = false;
}
if (ok
&& ((var0 != 0) + (var1 != 0)
+ (minus_var0 != 0) + (minus_var1 != 0)
+ (con0 != 0) + (con1 != 0)
+ (minus_con0 != 0) + (minus_con1 != 0)
+ (lit0 != 0) + (lit1 != 0)
+ (minus_lit0 != 0) + (minus_lit1 != 0)) > 2)
{
var0 = associate_trees (loc, var0, var1, code, atype);
minus_var0 = associate_trees (loc, minus_var0, minus_var1,
code, atype);
con0 = associate_trees (loc, con0, con1, code, atype);
minus_con0 = associate_trees (loc, minus_con0, minus_con1,
code, atype);
lit0 = associate_trees (loc, lit0, lit1, code, atype);
minus_lit0 = associate_trees (loc, minus_lit0, minus_lit1,
code, atype);
if (minus_var0 && var0)
{
var0 = associate_trees (loc, var0, minus_var0,
MINUS_EXPR, atype);
minus_var0 = 0;
}
if (minus_con0 && con0)
{
con0 = associate_trees (loc, con0, minus_con0,
MINUS_EXPR, atype);
minus_con0 = 0;
}
if (minus_lit0 && lit0)
{
if (TREE_CODE (lit0) == INTEGER_CST
&& TREE_CODE (minus_lit0) == INTEGER_CST
&& tree_int_cst_lt (lit0, minus_lit0)
&& (var0 || con0))
{
minus_lit0 = associate_trees (loc, minus_lit0, lit0,
MINUS_EXPR, atype);
lit0 = 0;
}
else
{
lit0 = associate_trees (loc, lit0, minus_lit0,
MINUS_EXPR, atype);
minus_lit0 = 0;
}
}
if ((lit0 && TREE_OVERFLOW_P (lit0))
|| (minus_lit0 && TREE_OVERFLOW_P (minus_lit0)))
return NULL_TREE;
con0 = associate_trees (loc, con0, lit0, code, atype);
lit0 = 0;
minus_con0 = associate_trees (loc, minus_con0, minus_lit0,
code, atype);
minus_lit0 = 0;
if (minus_con0)
{
if (con0)
con0 = associate_trees (loc, con0, minus_con0,
MINUS_EXPR, atype);
else if (var0)
var0 = associate_trees (loc, var0, minus_con0,
MINUS_EXPR, atype);
else
gcc_unreachable ();
minus_con0 = 0;
}
if (minus_var0)
{
if (con0)
con0 = associate_trees (loc, con0, minus_var0,
MINUS_EXPR, atype);
else
gcc_unreachable ();
minus_var0 = 0;
}
return
fold_convert_loc (loc, type, associate_trees (loc, var0, con0,
code, atype));
}
}
return NULL_TREE;
case POINTER_DIFF_EXPR:
case MINUS_EXPR:
if (TREE_CODE (arg0) == ADDR_EXPR
&& TREE_CODE (TREE_OPERAND (arg0, 0)) == ARRAY_REF
&& TREE_CODE (arg1) == ADDR_EXPR
&& TREE_CODE (TREE_OPERAND (arg1, 0)) == ARRAY_REF)
{
tree tem = fold_addr_of_array_ref_difference (loc, type,
TREE_OPERAND (arg0, 0),
TREE_OPERAND (arg1, 0),
code
== POINTER_DIFF_EXPR);
if (tem)
return tem;
}
if (code == POINTER_DIFF_EXPR)
return NULL_TREE;
if (TREE_CODE (arg0) == NEGATE_EXPR
&& negate_expr_p (op1)
&& !(ANY_INTEGRAL_TYPE_P (type)
&& TYPE_OVERFLOW_UNDEFINED (type)
&& ANY_INTEGRAL_TYPE_P (TREE_TYPE (arg0))
&& !TYPE_OVERFLOW_UNDEFINED (TREE_TYPE (arg0))))
return fold_build2_loc (loc, MINUS_EXPR, type, negate_expr (op1),
fold_convert_loc (loc, type,
TREE_OPERAND (arg0, 0)));
if (!HONOR_SNANS (element_mode (arg0))
&& !HONOR_SIGNED_ZEROS (element_mode (arg0))
&& COMPLEX_FLOAT_TYPE_P (TREE_TYPE (arg0)))
{
tree rtype = TREE_TYPE (TREE_TYPE (arg0));
tree arg0r = fold_unary_loc (loc, REALPART_EXPR, rtype, arg0);
tree arg0i = fold_unary_loc (loc, IMAGPART_EXPR, rtype, arg0);
bool arg0rz = false, arg0iz = false;
if ((arg0r && (arg0rz = real_zerop (arg0r)))
|| (arg0i && (arg0iz = real_zerop (arg0i))))
{
tree arg1r = fold_unary_loc (loc, REALPART_EXPR, rtype, arg1);
tree arg1i = fold_unary_loc (loc, IMAGPART_EXPR, rtype, arg1);
if (arg0rz && arg1i && real_zerop (arg1i))
{
tree rp = fold_build1_loc (loc, NEGATE_EXPR, rtype,
arg1r ? arg1r
: build1 (REALPART_EXPR, rtype, arg1));
tree ip = arg0i ? arg0i
: build1 (IMAGPART_EXPR, rtype, arg0);
return fold_build2_loc (loc, COMPLEX_EXPR, type, rp, ip);
}
else if (arg0iz && arg1r && real_zerop (arg1r))
{
tree rp = arg0r ? arg0r
: build1 (REALPART_EXPR, rtype, arg0);
tree ip = fold_build1_loc (loc, NEGATE_EXPR, rtype,
arg1i ? arg1i
: build1 (IMAGPART_EXPR, rtype, arg1));
return fold_build2_loc (loc, COMPLEX_EXPR, type, rp, ip);
}
}
}
if (negate_expr_p (op1)
&& ! TYPE_OVERFLOW_SANITIZED (type)
&& ((FLOAT_TYPE_P (type)
&& (TREE_CODE (op1) != REAL_CST
|| REAL_VALUE_NEGATIVE (TREE_REAL_CST (op1))))
|| INTEGRAL_TYPE_P (type)))
return fold_build2_loc (loc, PLUS_EXPR, type,
fold_convert_loc (loc, type, arg0),
negate_expr (op1));
if ((TREE_CODE (arg0) == MULT_EXPR
|| TREE_CODE (arg1) == MULT_EXPR)
&& !TYPE_SATURATING (type)
&& TYPE_UNSIGNED (type) == TYPE_UNSIGNED (TREE_TYPE (arg0))
&& TYPE_UNSIGNED (type) == TYPE_UNSIGNED (TREE_TYPE (arg1))
&& (!FLOAT_TYPE_P (type) || flag_associative_math))
{
tree tem = fold_plusminus_mult_expr (loc, code, type, arg0, arg1);
if (tem)
return tem;
}
goto associate;
case MULT_EXPR:
if (! FLOAT_TYPE_P (type))
{
if (TREE_CODE (op1) == INTEGER_CST
&& tree_int_cst_sgn (op1) == -1
&& negate_expr_p (op0)
&& negate_expr_p (op1)
&& (tem = negate_expr (op1)) != op1
&& ! TREE_OVERFLOW (tem))
return fold_build2_loc (loc, MULT_EXPR, type,
fold_convert_loc (loc, type,
negate_expr (op0)), tem);
strict_overflow_p = false;
if (TREE_CODE (arg1) == INTEGER_CST
&& (tem = extract_muldiv (op0, arg1, code, NULL_TREE,
&strict_overflow_p)) != 0)
{
if (strict_overflow_p)
fold_overflow_warning (("assuming signed overflow does not "
"occur when simplifying "
"multiplication"),
WARN_STRICT_OVERFLOW_MISC);
return fold_convert_loc (loc, type, tem);
}
if (TREE_CODE (arg0) == CONJ_EXPR
&& operand_equal_p (TREE_OPERAND (arg0, 0), arg1, 0))
return fold_mult_zconjz (loc, type, arg1);
if (TREE_CODE (arg1) == CONJ_EXPR
&& operand_equal_p (arg0, TREE_OPERAND (arg1, 0), 0))
return fold_mult_zconjz (loc, type, arg0);
}
else
{
if (!HONOR_NANS (arg0)
&& !HONOR_SIGNED_ZEROS (element_mode (arg0))
&& COMPLEX_FLOAT_TYPE_P (TREE_TYPE (arg0))
&& TREE_CODE (arg1) == COMPLEX_CST
&& real_zerop (TREE_REALPART (arg1)))
{
tree rtype = TREE_TYPE (TREE_TYPE (arg0));
if (real_onep (TREE_IMAGPART (arg1)))
return
fold_build2_loc (loc, COMPLEX_EXPR, type,
negate_expr (fold_build1_loc (loc, IMAGPART_EXPR,
rtype, arg0)),
fold_build1_loc (loc, REALPART_EXPR, rtype, arg0));
else if (real_minus_onep (TREE_IMAGPART (arg1)))
return
fold_build2_loc (loc, COMPLEX_EXPR, type,
fold_build1_loc (loc, IMAGPART_EXPR, rtype, arg0),
negate_expr (fold_build1_loc (loc, REALPART_EXPR,
rtype, arg0)));
}
if (flag_unsafe_math_optimizations
&& TREE_CODE (arg0) == CONJ_EXPR
&& operand_equal_p (TREE_OPERAND (arg0, 0), arg1, 0))
return fold_mult_zconjz (loc, type, arg1);
if (flag_unsafe_math_optimizations
&& TREE_CODE (arg1) == CONJ_EXPR
&& operand_equal_p (arg0, TREE_OPERAND (arg1, 0), 0))
return fold_mult_zconjz (loc, type, arg0);
}
goto associate;
case BIT_IOR_EXPR:
if (TREE_CODE (arg0) == BIT_AND_EXPR
&& TREE_CODE (arg1) == INTEGER_CST
&& TREE_CODE (TREE_OPERAND (arg0, 1)) == INTEGER_CST)
{
int width = TYPE_PRECISION (type), w;
wide_int c1 = wi::to_wide (TREE_OPERAND (arg0, 1));
wide_int c2 = wi::to_wide (arg1);
if ((c1 & c2) == c1)
return omit_one_operand_loc (loc, type, arg1,
TREE_OPERAND (arg0, 0));
wide_int msk = wi::mask (width, false,
TYPE_PRECISION (TREE_TYPE (arg1)));
if (wi::bit_and_not (msk, c1 | c2) == 0)
{
tem = fold_convert_loc (loc, type, TREE_OPERAND (arg0, 0));
return fold_build2_loc (loc, BIT_IOR_EXPR, type, tem, arg1);
}
c1 &= msk;
c2 &= msk;
wide_int c3 = wi::bit_and_not (c1, c2);
for (w = BITS_PER_UNIT; w <= width; w <<= 1)
{
wide_int mask = wi::mask (w, false,
TYPE_PRECISION (type));
if (((c1 | c2) & mask) == mask
&& wi::bit_and_not (c1, mask) == 0)
{
c3 = mask;
break;
}
}
if (c3 != c1)
{
tem = fold_convert_loc (loc, type, TREE_OPERAND (arg0, 0));
tem = fold_build2_loc (loc, BIT_AND_EXPR, type, tem,
wide_int_to_tree (type, c3));
return fold_build2_loc (loc, BIT_IOR_EXPR, type, tem, arg1);
}
}
goto bit_rotate;
case BIT_XOR_EXPR:
if (TREE_CODE (arg0) == BIT_AND_EXPR
&& INTEGRAL_TYPE_P (type)
&& integer_onep (TREE_OPERAND (arg0, 1))
&& integer_onep (arg1))
return fold_build2_loc (loc, EQ_EXPR, type, arg0,
build_zero_cst (TREE_TYPE (arg0)));
goto bit_rotate;
case BIT_AND_EXPR:
if (TREE_CODE (arg0) == BIT_XOR_EXPR
&& INTEGRAL_TYPE_P (type)
&& integer_onep (TREE_OPERAND (arg0, 1))
&& integer_onep (arg1))
{
tree tem2;
tem = TREE_OPERAND (arg0, 0);
tem2 = fold_convert_loc (loc, TREE_TYPE (tem), arg1);
tem2 = fold_build2_loc (loc, BIT_AND_EXPR, TREE_TYPE (tem),
tem, tem2);
return fold_build2_loc (loc, EQ_EXPR, type, tem2,
build_zero_cst (TREE_TYPE (tem)));
}
if (TREE_CODE (arg0) == BIT_NOT_EXPR
&& INTEGRAL_TYPE_P (type)
&& integer_onep (arg1))
{
tree tem2;
tem = TREE_OPERAND (arg0, 0);
tem2 = fold_convert_loc (loc, TREE_TYPE (tem), arg1);
tem2 = fold_build2_loc (loc, BIT_AND_EXPR, TREE_TYPE (tem),
tem, tem2);
return fold_build2_loc (loc, EQ_EXPR, type, tem2,
build_zero_cst (TREE_TYPE (tem)));
}
if (TREE_CODE (arg0) == TRUTH_NOT_EXPR
&& integer_onep (arg1))
{
tem = TREE_OPERAND (arg0, 0);
return fold_build2_loc (loc, EQ_EXPR, type, tem,
build_zero_cst (TREE_TYPE (tem)));
}
if (TREE_CODE (arg1) == INTEGER_CST)
{
wi::tree_to_wide_ref cst1 = wi::to_wide (arg1);
wide_int ncst1 = -cst1;
if ((cst1 & ncst1) == ncst1
&& multiple_of_p (type, arg0,
wide_int_to_tree (TREE_TYPE (arg1), ncst1)))
return fold_convert_loc (loc, type, arg0);
}
if (TREE_CODE (arg1) == INTEGER_CST
&& TREE_CODE (arg0) == MULT_EXPR
&& TREE_CODE (TREE_OPERAND (arg0, 1)) == INTEGER_CST)
{
wi::tree_to_wide_ref warg1 = wi::to_wide (arg1);
wide_int masked
= mask_with_tz (type, warg1, wi::to_wide (TREE_OPERAND (arg0, 1)));
if (masked == 0)
return omit_two_operands_loc (loc, type, build_zero_cst (type),
arg0, arg1);
else if (masked != warg1)
{
int pop = wi::popcount (warg1);
if (!(pop >= BITS_PER_UNIT
&& pow2p_hwi (pop)
&& wi::mask (pop, false, warg1.get_precision ()) == warg1))
return fold_build2_loc (loc, code, type, op0,
wide_int_to_tree (type, masked));
}
}
if (TREE_CODE (arg1) == INTEGER_CST)
{
wi::tree_to_wide_ref cst1 = wi::to_wide (arg1);
if ((~cst1 != 0) && (cst1 & (cst1 + 1)) == 0
&& INTEGRAL_TYPE_P (TREE_TYPE (arg0))
&& (TREE_CODE (arg0) == PLUS_EXPR
|| TREE_CODE (arg0) == MINUS_EXPR
|| TREE_CODE (arg0) == NEGATE_EXPR)
&& (TYPE_OVERFLOW_WRAPS (TREE_TYPE (arg0))
|| TREE_CODE (TREE_TYPE (arg0)) == INTEGER_TYPE))
{
tree pmop[2];
int which = 0;
wide_int cst0;
pmop[0] = TREE_OPERAND (arg0, 0);
pmop[1] = NULL;
if (TREE_CODE (arg0) != NEGATE_EXPR)
{
pmop[1] = TREE_OPERAND (arg0, 1);
which = 1;
}
if ((wi::max_value (TREE_TYPE (arg0)) & cst1) != cst1)
which = -1;
for (; which >= 0; which--)
switch (TREE_CODE (pmop[which]))
{
case BIT_AND_EXPR:
case BIT_IOR_EXPR:
case BIT_XOR_EXPR:
if (TREE_CODE (TREE_OPERAND (pmop[which], 1))
!= INTEGER_CST)
break;
cst0 = wi::to_wide (TREE_OPERAND (pmop[which], 1)) & cst1;
if (TREE_CODE (pmop[which]) == BIT_AND_EXPR)
{
if (cst0 != cst1)
break;
}
else if (cst0 != 0)
break;
pmop[which] = TREE_OPERAND (pmop[which], 0);
break;
case INTEGER_CST:
if ((TREE_CODE (arg0) == PLUS_EXPR
|| (TREE_CODE (arg0) == MINUS_EXPR && which == 0))
&& (cst1 & wi::to_wide (pmop[which])) == 0)
pmop[which] = NULL;
break;
default:
break;
}
if (pmop[0] != TREE_OPERAND (arg0, 0)
|| (TREE_CODE (arg0) != NEGATE_EXPR
&& pmop[1] != TREE_OPERAND (arg0, 1)))
{
tree utype = TREE_TYPE (arg0);
if (! TYPE_OVERFLOW_WRAPS (TREE_TYPE (arg0)))
{
utype = unsigned_type_for (TREE_TYPE (arg0));
if (pmop[0] != NULL)
pmop[0] = fold_convert_loc (loc, utype, pmop[0]);
if (pmop[1] != NULL)
pmop[1] = fold_convert_loc (loc, utype, pmop[1]);
}
if (TREE_CODE (arg0) == NEGATE_EXPR)
tem = fold_build1_loc (loc, NEGATE_EXPR, utype, pmop[0]);
else if (TREE_CODE (arg0) == PLUS_EXPR)
{
if (pmop[0] != NULL && pmop[1] != NULL)
tem = fold_build2_loc (loc, PLUS_EXPR, utype,
pmop[0], pmop[1]);
else if (pmop[0] != NULL)
tem = pmop[0];
else if (pmop[1] != NULL)
tem = pmop[1];
else
return build_int_cst (type, 0);
}
else if (pmop[0] == NULL)
tem = fold_build1_loc (loc, NEGATE_EXPR, utype, pmop[1]);
else
tem = fold_build2_loc (loc, MINUS_EXPR, utype,
pmop[0], pmop[1]);
tem = fold_build2_loc (loc, BIT_AND_EXPR, utype, tem,
fold_convert_loc (loc, utype, arg1));
return fold_convert_loc (loc, type, tem);
}
}
}
if (TREE_CODE (arg1) == INTEGER_CST && TREE_CODE (arg0) == NOP_EXPR
&& TYPE_UNSIGNED (TREE_TYPE (TREE_OPERAND (arg0, 0))))
{
prec = element_precision (TREE_TYPE (TREE_OPERAND (arg0, 0)));
wide_int mask = wide_int::from (wi::to_wide (arg1), prec, UNSIGNED);
if (mask == -1)
return
fold_convert_loc (loc, type, TREE_OPERAND (arg0, 0));
}
goto associate;
case RDIV_EXPR:
if (TREE_CODE (arg1) == REAL_CST
&& !MODE_HAS_INFINITIES (TYPE_MODE (TREE_TYPE (arg1)))
&& real_zerop (arg1))
return NULL_TREE;
if (TREE_CODE (arg0) == NEGATE_EXPR && negate_expr_p (arg1))
return fold_build2_loc (loc, RDIV_EXPR, type,
TREE_OPERAND (arg0, 0),
negate_expr (arg1));
if (TREE_CODE (arg1) == NEGATE_EXPR && negate_expr_p (arg0))
return fold_build2_loc (loc, RDIV_EXPR, type,
negate_expr (arg0),
TREE_OPERAND (arg1, 0));
return NULL_TREE;
case TRUNC_DIV_EXPR:
case FLOOR_DIV_EXPR:
strict_overflow_p = false;
if (TREE_CODE (arg1) == LSHIFT_EXPR
&& (TYPE_UNSIGNED (type)
|| tree_expr_nonnegative_warnv_p (op0, &strict_overflow_p)))
{
tree sval = TREE_OPERAND (arg1, 0);
if (integer_pow2p (sval) && tree_int_cst_sgn (sval) > 0)
{
tree sh_cnt = TREE_OPERAND (arg1, 1);
tree pow2 = build_int_cst (TREE_TYPE (sh_cnt),
wi::exact_log2 (wi::to_wide (sval)));
if (strict_overflow_p)
fold_overflow_warning (("assuming signed overflow does not "
"occur when simplifying A / (B << N)"),
WARN_STRICT_OVERFLOW_MISC);
sh_cnt = fold_build2_loc (loc, PLUS_EXPR, TREE_TYPE (sh_cnt),
sh_cnt, pow2);
return fold_build2_loc (loc, RSHIFT_EXPR, type,
fold_convert_loc (loc, type, arg0), sh_cnt);
}
}
case ROUND_DIV_EXPR:
case CEIL_DIV_EXPR:
case EXACT_DIV_EXPR:
if (integer_zerop (arg1))
return NULL_TREE;
if ((!INTEGRAL_TYPE_P (type) || TYPE_OVERFLOW_UNDEFINED (type))
&& TREE_CODE (op0) == NEGATE_EXPR
&& negate_expr_p (op1))
{
if (INTEGRAL_TYPE_P (type))
fold_overflow_warning (("assuming signed overflow does not occur "
"when distributing negation across "
"division"),
WARN_STRICT_OVERFLOW_MISC);
return fold_build2_loc (loc, code, type,
fold_convert_loc (loc, type,
TREE_OPERAND (arg0, 0)),
negate_expr (op1));
}
if ((!INTEGRAL_TYPE_P (type) || TYPE_OVERFLOW_UNDEFINED (type))
&& TREE_CODE (arg1) == NEGATE_EXPR
&& negate_expr_p (op0))
{
if (INTEGRAL_TYPE_P (type))
fold_overflow_warning (("assuming signed overflow does not occur "
"when distributing negation across "
"division"),
WARN_STRICT_OVERFLOW_MISC);
return fold_build2_loc (loc, code, type,
negate_expr (op0),
fold_convert_loc (loc, type,
TREE_OPERAND (arg1, 0)));
}
if ((code == CEIL_DIV_EXPR || code == FLOOR_DIV_EXPR)
&& multiple_of_p (type, arg0, arg1))
return fold_build2_loc (loc, EXACT_DIV_EXPR, type,
fold_convert (type, arg0),
fold_convert (type, arg1));
strict_overflow_p = false;
if (TREE_CODE (arg1) == INTEGER_CST
&& (tem = extract_muldiv (op0, arg1, code, NULL_TREE,
&strict_overflow_p)) != 0)
{
if (strict_overflow_p)
fold_overflow_warning (("assuming signed overflow does not occur "
"when simplifying division"),
WARN_STRICT_OVERFLOW_MISC);
return fold_convert_loc (loc, type, tem);
}
return NULL_TREE;
case CEIL_MOD_EXPR:
case FLOOR_MOD_EXPR:
case ROUND_MOD_EXPR:
case TRUNC_MOD_EXPR:
strict_overflow_p = false;
if (TREE_CODE (arg1) == INTEGER_CST
&& (tem = extract_muldiv (op0, arg1, code, NULL_TREE,
&strict_overflow_p)) != 0)
{
if (strict_overflow_p)
fold_overflow_warning (("assuming signed overflow does not occur "
"when simplifying modulus"),
WARN_STRICT_OVERFLOW_MISC);
return fold_convert_loc (loc, type, tem);
}
return NULL_TREE;
case LROTATE_EXPR:
case RROTATE_EXPR:
case RSHIFT_EXPR:
case LSHIFT_EXPR:
if (TREE_CODE (arg1) == INTEGER_CST && tree_int_cst_sgn (arg1) < 0)
return NULL_TREE;
prec = element_precision (type);
if (code == RROTATE_EXPR && TREE_CODE (arg1) == INTEGER_CST
&& (TREE_CODE (arg0) == BIT_AND_EXPR
|| TREE_CODE (arg0) == BIT_IOR_EXPR
|| TREE_CODE (arg0) == BIT_XOR_EXPR)
&& TREE_CODE (TREE_OPERAND (arg0, 1)) == INTEGER_CST)
{
tree arg00 = fold_convert_loc (loc, type, TREE_OPERAND (arg0, 0));
tree arg01 = fold_convert_loc (loc, type, TREE_OPERAND (arg0, 1));
return fold_build2_loc (loc, TREE_CODE (arg0), type,
fold_build2_loc (loc, code, type,
arg00, arg1),
fold_build2_loc (loc, code, type,
arg01, arg1));
}
if (code == RROTATE_EXPR && TREE_CODE (arg1) == INTEGER_CST
&& TREE_CODE (arg0) == RROTATE_EXPR
&& TREE_CODE (TREE_OPERAND (arg0, 1)) == INTEGER_CST
&& wi::umod_trunc (wi::to_wide (arg1)
+ wi::to_wide (TREE_OPERAND (arg0, 1)),
prec) == 0)
return fold_convert_loc (loc, type, TREE_OPERAND (arg0, 0));
return NULL_TREE;
case MIN_EXPR:
case MAX_EXPR:
goto associate;
case TRUTH_ANDIF_EXPR:
if (integer_zerop (arg0))
return fold_convert_loc (loc, type, arg0);
case TRUTH_AND_EXPR:
if (TREE_CODE (arg0) == INTEGER_CST && ! integer_zerop (arg0))
return non_lvalue_loc (loc, fold_convert_loc (loc, type, arg1));
if (TREE_CODE (arg1) == INTEGER_CST && ! integer_zerop (arg1)
&& (code != TRUTH_ANDIF_EXPR || ! TREE_SIDE_EFFECTS (arg0)))
return non_lvalue_loc (loc, fold_convert_loc (loc, type, arg0));
if (integer_zerop (arg1))
return omit_one_operand_loc (loc, type, arg1, arg0);
if (integer_zerop (arg0))
return omit_one_operand_loc (loc, type, arg0, arg1);
if (TREE_CODE (arg0) == TRUTH_NOT_EXPR
&& operand_equal_p (TREE_OPERAND (arg0, 0), arg1, 0))
return omit_one_operand_loc (loc, type, integer_zero_node, arg1);
if (TREE_CODE (arg1) == TRUTH_NOT_EXPR
&& operand_equal_p (arg0, TREE_OPERAND (arg1, 0), 0))
return omit_one_operand_loc (loc, type, integer_zero_node, arg0);
if (!TREE_SIDE_EFFECTS (arg0)
&& !TREE_SIDE_EFFECTS (arg1))
{
tem = fold_to_nonsharp_ineq_using_bound (loc, arg0, arg1);
if (tem && !operand_equal_p (tem, arg0, 0))
return fold_build2_loc (loc, code, type, tem, arg1);
tem = fold_to_nonsharp_ineq_using_bound (loc, arg1, arg0);
if (tem && !operand_equal_p (tem, arg1, 0))
return fold_build2_loc (loc, code, type, arg0, tem);
}
if ((tem = fold_truth_andor (loc, code, type, arg0, arg1, op0, op1))
!= NULL_TREE)
return tem;
return NULL_TREE;
case TRUTH_ORIF_EXPR:
if (TREE_CODE (arg0) == INTEGER_CST && ! integer_zerop (arg0))
return fold_convert_loc (loc, type, arg0);
case TRUTH_OR_EXPR:
if (TREE_CODE (arg0) == INTEGER_CST && integer_zerop (arg0))
return non_lvalue_loc (loc, fold_convert_loc (loc, type, arg1));
if (TREE_CODE (arg1) == INTEGER_CST && integer_zerop (arg1)
&& (code != TRUTH_ORIF_EXPR || ! TREE_SIDE_EFFECTS (arg0)))
return non_lvalue_loc (loc, fold_convert_loc (loc, type, arg0));
if (TREE_CODE (arg1) == INTEGER_CST && ! integer_zerop (arg1))
return omit_one_operand_loc (loc, type, arg1, arg0);
if (TREE_CODE (arg0) == INTEGER_CST && ! integer_zerop (arg0))
return omit_one_operand_loc (loc, type, arg0, arg1);
if (TREE_CODE (arg0) == TRUTH_NOT_EXPR
&& operand_equal_p (TREE_OPERAND (arg0, 0), arg1, 0))
return omit_one_operand_loc (loc, type, integer_one_node, arg1);
if (TREE_CODE (arg1) == TRUTH_NOT_EXPR
&& operand_equal_p (arg0, TREE_OPERAND (arg1, 0), 0))
return omit_one_operand_loc (loc, type, integer_one_node, arg0);
if (TREE_CODE (arg0) == TRUTH_AND_EXPR
&& TREE_CODE (arg1) == TRUTH_AND_EXPR)
{
tree a0, a1, l0, l1, n0, n1;
a0 = fold_convert_loc (loc, type, TREE_OPERAND (arg1, 0));
a1 = fold_convert_loc (loc, type, TREE_OPERAND (arg1, 1));
l0 = fold_convert_loc (loc, type, TREE_OPERAND (arg0, 0));
l1 = fold_convert_loc (loc, type, TREE_OPERAND (arg0, 1));
n0 = fold_build1_loc (loc, TRUTH_NOT_EXPR, type, l0);
n1 = fold_build1_loc (loc, TRUTH_NOT_EXPR, type, l1);
if ((operand_equal_p (n0, a0, 0)
&& operand_equal_p (n1, a1, 0))
|| (operand_equal_p (n0, a1, 0)
&& operand_equal_p (n1, a0, 0)))
return fold_build2_loc (loc, TRUTH_XOR_EXPR, type, l0, n1);
}
if ((tem = fold_truth_andor (loc, code, type, arg0, arg1, op0, op1))
!= NULL_TREE)
return tem;
return NULL_TREE;
case TRUTH_XOR_EXPR:
if (integer_zerop (arg1))
return non_lvalue_loc (loc, fold_convert_loc (loc, type, arg0));
if (integer_onep (arg1))
{
tem = invert_truthvalue_loc (loc, arg0);
return non_lvalue_loc (loc, fold_convert_loc (loc, type, tem));
}
if (operand_equal_p (arg0, arg1, 0))
return omit_one_operand_loc (loc, type, integer_zero_node, arg0);
if (TREE_CODE (arg0) == TRUTH_NOT_EXPR
&& operand_equal_p (TREE_OPERAND (arg0, 0), arg1, 0))
return omit_one_operand_loc (loc, type, integer_one_node, arg1);
if (TREE_CODE (arg1) == TRUTH_NOT_EXPR
&& operand_equal_p (arg0, TREE_OPERAND (arg1, 0), 0))
return omit_one_operand_loc (loc, type, integer_one_node, arg0);
return NULL_TREE;
case EQ_EXPR:
case NE_EXPR:
STRIP_NOPS (arg0);
STRIP_NOPS (arg1);
tem = fold_comparison (loc, code, type, op0, op1);
if (tem != NULL_TREE)
return tem;
if (TREE_CODE (TREE_TYPE (arg0)) == BOOLEAN_TYPE && integer_onep (arg1)
&& code == NE_EXPR)
return fold_convert_loc (loc, type,
fold_build1_loc (loc, TRUTH_NOT_EXPR,
TREE_TYPE (arg0), arg0));
if (TREE_CODE (TREE_TYPE (arg0)) == BOOLEAN_TYPE && integer_zerop (arg1)
&& code == EQ_EXPR)
return fold_convert_loc (loc, type,
fold_build1_loc (loc, TRUTH_NOT_EXPR,
TREE_TYPE (arg0), arg0));
if (TREE_CODE (arg0) == TRUTH_NOT_EXPR && integer_zerop (arg1)
&& code == NE_EXPR)
return non_lvalue_loc (loc, fold_convert_loc (loc, type, arg0));
if (TREE_CODE (arg0) == BIT_AND_EXPR
&& integer_zerop (arg1))
{
tree arg00 = TREE_OPERAND (arg0, 0);
tree arg01 = TREE_OPERAND (arg0, 1);
if (TREE_CODE (arg00) == LSHIFT_EXPR
&& integer_onep (TREE_OPERAND (arg00, 0)))
{
tree tem = fold_build2_loc (loc, RSHIFT_EXPR, TREE_TYPE (arg00),
arg01, TREE_OPERAND (arg00, 1));
tem = fold_build2_loc (loc, BIT_AND_EXPR, TREE_TYPE (arg0), tem,
build_int_cst (TREE_TYPE (arg0), 1));
return fold_build2_loc (loc, code, type,
fold_convert_loc (loc, TREE_TYPE (arg1), tem),
arg1);
}
else if (TREE_CODE (arg01) == LSHIFT_EXPR
&& integer_onep (TREE_OPERAND (arg01, 0)))
{
tree tem = fold_build2_loc (loc, RSHIFT_EXPR, TREE_TYPE (arg01),
arg00, TREE_OPERAND (arg01, 1));
tem = fold_build2_loc (loc, BIT_AND_EXPR, TREE_TYPE (arg0), tem,
build_int_cst (TREE_TYPE (arg0), 1));
return fold_build2_loc (loc, code, type,
fold_convert_loc (loc, TREE_TYPE (arg1), tem),
arg1);
}
}
if (integer_zerop (arg1)
&& !TYPE_UNSIGNED (TREE_TYPE (arg0))
&& (TREE_CODE (arg0) == TRUNC_MOD_EXPR
|| TREE_CODE (arg0) == CEIL_MOD_EXPR
|| TREE_CODE (arg0) == FLOOR_MOD_EXPR
|| TREE_CODE (arg0) == ROUND_MOD_EXPR)
&& integer_pow2p (TREE_OPERAND (arg0, 1)))
{
tree newtype = unsigned_type_for (TREE_TYPE (arg0));
tree newmod = fold_build2_loc (loc, TREE_CODE (arg0), newtype,
fold_convert_loc (loc, newtype,
TREE_OPERAND (arg0, 0)),
fold_convert_loc (loc, newtype,
TREE_OPERAND (arg0, 1)));
return fold_build2_loc (loc, code, type, newmod,
fold_convert_loc (loc, newtype, arg1));
}
if (TREE_CODE (arg0) == BIT_AND_EXPR
&& TREE_CODE (TREE_OPERAND (arg0, 0)) == RSHIFT_EXPR
&& TREE_CODE (TREE_OPERAND (TREE_OPERAND (arg0, 0), 1))
== INTEGER_CST
&& integer_pow2p (TREE_OPERAND (arg0, 1))
&& integer_zerop (arg1))
{
tree itype = TREE_TYPE (arg0);
tree arg001 = TREE_OPERAND (TREE_OPERAND (arg0, 0), 1);
prec = TYPE_PRECISION (itype);
if (wi::ltu_p (wi::to_wide (arg001), prec))
{
tree arg01 = TREE_OPERAND (arg0, 1);
tree arg000 = TREE_OPERAND (TREE_OPERAND (arg0, 0), 0);
unsigned HOST_WIDE_INT log2 = tree_log2 (arg01);
if ((log2 + TREE_INT_CST_LOW (arg001)) < prec)
{
tem = fold_build2_loc (loc, LSHIFT_EXPR, itype, arg01, arg001);
tem = fold_build2_loc (loc, BIT_AND_EXPR, itype, arg000, tem);
return fold_build2_loc (loc, code, type, tem,
fold_convert_loc (loc, itype, arg1));
}
else if (!TYPE_UNSIGNED (itype))
return fold_build2_loc (loc, code == EQ_EXPR ? GE_EXPR : LT_EXPR, type,
arg000, build_int_cst (itype, 0));
else
return omit_one_operand_loc (loc, type,
code == EQ_EXPR ? integer_one_node
: integer_zero_node,
arg000);
}
}
if ((TREE_CODE (arg0) == COMPONENT_REF
|| TREE_CODE (arg0) == BIT_FIELD_REF)
&& (optimize || TREE_CODE (arg1) == INTEGER_CST))
{
t1 = optimize_bit_field_compare (loc, code, type, arg0, arg1);
if (t1)
return t1;
}
if (TREE_CODE (arg0) == CALL_EXPR && integer_zerop (arg1))
{
tree fndecl = get_callee_fndecl (arg0);
if (fndecl
&& DECL_BUILT_IN_CLASS (fndecl) == BUILT_IN_NORMAL
&& DECL_FUNCTION_CODE (fndecl) == BUILT_IN_STRLEN
&& call_expr_nargs (arg0) == 1
&& (TREE_CODE (TREE_TYPE (CALL_EXPR_ARG (arg0, 0)))
== POINTER_TYPE))
{
tree ptrtype
= build_pointer_type (build_qualified_type (char_type_node,
TYPE_QUAL_CONST));
tree ptr = fold_convert_loc (loc, ptrtype,
CALL_EXPR_ARG (arg0, 0));
tree iref = build_fold_indirect_ref_loc (loc, ptr);
return fold_build2_loc (loc, code, type, iref,
build_int_cst (TREE_TYPE (iref), 0));
}
}
if (TREE_CODE (arg0) == RSHIFT_EXPR
&& integer_zerop (arg1)
&& TREE_CODE (TREE_OPERAND (arg0, 1)) == INTEGER_CST)
{
tree arg00 = TREE_OPERAND (arg0, 0);
tree arg01 = TREE_OPERAND (arg0, 1);
tree itype = TREE_TYPE (arg00);
if (wi::to_wide (arg01) == element_precision (itype) - 1)
{
if (TYPE_UNSIGNED (itype))
{
itype = signed_type_for (itype);
arg00 = fold_convert_loc (loc, itype, arg00);
}
return fold_build2_loc (loc, code == EQ_EXPR ? GE_EXPR : LT_EXPR,
type, arg00, build_zero_cst (itype));
}
}
if (TREE_CODE (arg0) == BIT_AND_EXPR
&& TREE_CODE (TREE_OPERAND (arg0, 0)) == BIT_NOT_EXPR
&& integer_zerop (arg1)
&& integer_pow2p (TREE_OPERAND (arg0, 1)))
{
tem = fold_build2_loc (loc, BIT_AND_EXPR, TREE_TYPE (arg0),
TREE_OPERAND (TREE_OPERAND (arg0, 0), 0),
TREE_OPERAND (arg0, 1));
return fold_build2_loc (loc, code == EQ_EXPR ? NE_EXPR : EQ_EXPR,
type, tem,
fold_convert_loc (loc, TREE_TYPE (arg0),
arg1));
}
if (TREE_CODE (arg0) == BIT_XOR_EXPR
&& TREE_CODE (TREE_OPERAND (arg0, 0)) == BIT_AND_EXPR
&& integer_zerop (arg1)
&& integer_pow2p (TREE_OPERAND (arg0, 1))
&& operand_equal_p (TREE_OPERAND (TREE_OPERAND (arg0, 0), 1),
TREE_OPERAND (arg0, 1), OEP_ONLY_CONST))
{
tree arg00 = TREE_OPERAND (arg0, 0);
return fold_build2_loc (loc, code == EQ_EXPR ? NE_EXPR : EQ_EXPR, type,
arg00, build_int_cst (TREE_TYPE (arg00), 0));
}
if (TREE_CODE (arg0) == BIT_AND_EXPR
&& TREE_CODE (TREE_OPERAND (arg0, 0)) == BIT_XOR_EXPR
&& integer_zerop (arg1)
&& integer_pow2p (TREE_OPERAND (arg0, 1))
&& operand_equal_p (TREE_OPERAND (TREE_OPERAND (arg0, 0), 1),
TREE_OPERAND (arg0, 1), OEP_ONLY_CONST))
{
tree arg000 = TREE_OPERAND (TREE_OPERAND (arg0, 0), 0);
tem = fold_build2_loc (loc, BIT_AND_EXPR, TREE_TYPE (arg000),
arg000, TREE_OPERAND (arg0, 1));
return fold_build2_loc (loc, code == EQ_EXPR ? NE_EXPR : EQ_EXPR, type,
tem, build_int_cst (TREE_TYPE (tem), 0));
}
if (integer_zerop (arg1)
&& tree_expr_nonzero_p (arg0))
{
tree res = constant_boolean_node (code==NE_EXPR, type);
return omit_one_operand_loc (loc, type, res, arg0);
}
if (TREE_CODE (arg0) == BIT_AND_EXPR
&& TREE_CODE (arg1) == BIT_AND_EXPR)
{
tree arg00 = TREE_OPERAND (arg0, 0);
tree arg01 = TREE_OPERAND (arg0, 1);
tree arg10 = TREE_OPERAND (arg1, 0);
tree arg11 = TREE_OPERAND (arg1, 1);
tree itype = TREE_TYPE (arg0);
if (operand_equal_p (arg01, arg11, 0))
{
tem = fold_convert_loc (loc, itype, arg10);
tem = fold_build2_loc (loc, BIT_XOR_EXPR, itype, arg00, tem);
tem = fold_build2_loc (loc, BIT_AND_EXPR, itype, tem, arg01);
return fold_build2_loc (loc, code, type, tem,
build_zero_cst (itype));
}
if (operand_equal_p (arg01, arg10, 0))
{
tem = fold_convert_loc (loc, itype, arg11);
tem = fold_build2_loc (loc, BIT_XOR_EXPR, itype, arg00, tem);
tem = fold_build2_loc (loc, BIT_AND_EXPR, itype, tem, arg01);
return fold_build2_loc (loc, code, type, tem,
build_zero_cst (itype));
}
if (operand_equal_p (arg00, arg11, 0))
{
tem = fold_convert_loc (loc, itype, arg10);
tem = fold_build2_loc (loc, BIT_XOR_EXPR, itype, arg01, tem);
tem = fold_build2_loc (loc, BIT_AND_EXPR, itype, tem, arg00);
return fold_build2_loc (loc, code, type, tem,
build_zero_cst (itype));
}
if (operand_equal_p (arg00, arg10, 0))
{
tem = fold_convert_loc (loc, itype, arg11);
tem = fold_build2_loc (loc, BIT_XOR_EXPR, itype, arg01, tem);
tem = fold_build2_loc (loc, BIT_AND_EXPR, itype, tem, arg00);
return fold_build2_loc (loc, code, type, tem,
build_zero_cst (itype));
}
}
if (TREE_CODE (arg0) == BIT_XOR_EXPR
&& TREE_CODE (arg1) == BIT_XOR_EXPR)
{
tree arg00 = TREE_OPERAND (arg0, 0);
tree arg01 = TREE_OPERAND (arg0, 1);
tree arg10 = TREE_OPERAND (arg1, 0);
tree arg11 = TREE_OPERAND (arg1, 1);
tree itype = TREE_TYPE (arg0);
if (operand_equal_p (arg01, arg11, 0))
return fold_build2_loc (loc, code, type, arg00,
fold_convert_loc (loc, TREE_TYPE (arg00),
arg10));
if (operand_equal_p (arg01, arg10, 0))
return fold_build2_loc (loc, code, type, arg00,
fold_convert_loc (loc, TREE_TYPE (arg00),
arg11));
if (operand_equal_p (arg00, arg11, 0))
return fold_build2_loc (loc, code, type, arg01,
fold_convert_loc (loc, TREE_TYPE (arg01),
arg10));
if (operand_equal_p (arg00, arg10, 0))
return fold_build2_loc (loc, code, type, arg01,
fold_convert_loc (loc, TREE_TYPE (arg01),
arg11));
if (TREE_CODE (arg01) == INTEGER_CST
&& TREE_CODE (arg11) == INTEGER_CST)
{
tem = fold_build2_loc (loc, BIT_XOR_EXPR, itype, arg01,
fold_convert_loc (loc, itype, arg11));
tem = fold_build2_loc (loc, BIT_XOR_EXPR, itype, arg00, tem);
return fold_build2_loc (loc, code, type, tem,
fold_convert_loc (loc, itype, arg10));
}
}
if ((TREE_CODE (arg0) == COMPLEX_EXPR
|| TREE_CODE (arg0) == COMPLEX_CST)
&& (TREE_CODE (arg1) == COMPLEX_EXPR
|| TREE_CODE (arg1) == COMPLEX_CST))
{
tree real0, imag0, real1, imag1;
tree rcond, icond;
if (TREE_CODE (arg0) == COMPLEX_EXPR)
{
real0 = TREE_OPERAND (arg0, 0);
imag0 = TREE_OPERAND (arg0, 1);
}
else
{
real0 = TREE_REALPART (arg0);
imag0 = TREE_IMAGPART (arg0);
}
if (TREE_CODE (arg1) == COMPLEX_EXPR)
{
real1 = TREE_OPERAND (arg1, 0);
imag1 = TREE_OPERAND (arg1, 1);
}
else
{
real1 = TREE_REALPART (arg1);
imag1 = TREE_IMAGPART (arg1);
}
rcond = fold_binary_loc (loc, code, type, real0, real1);
if (rcond && TREE_CODE (rcond) == INTEGER_CST)
{
if (integer_zerop (rcond))
{
if (code == EQ_EXPR)
return omit_two_operands_loc (loc, type, boolean_false_node,
imag0, imag1);
return fold_build2_loc (loc, NE_EXPR, type, imag0, imag1);
}
else
{
if (code == NE_EXPR)
return omit_two_operands_loc (loc, type, boolean_true_node,
imag0, imag1);
return fold_build2_loc (loc, EQ_EXPR, type, imag0, imag1);
}
}
icond = fold_binary_loc (loc, code, type, imag0, imag1);
if (icond && TREE_CODE (icond) == INTEGER_CST)
{
if (integer_zerop (icond))
{
if (code == EQ_EXPR)
return omit_two_operands_loc (loc, type, boolean_false_node,
real0, real1);
return fold_build2_loc (loc, NE_EXPR, type, real0, real1);
}
else
{
if (code == NE_EXPR)
return omit_two_operands_loc (loc, type, boolean_true_node,
real0, real1);
return fold_build2_loc (loc, EQ_EXPR, type, real0, real1);
}
}
}
return NULL_TREE;
case LT_EXPR:
case GT_EXPR:
case LE_EXPR:
case GE_EXPR:
tem = fold_comparison (loc, code, type, op0, op1);
if (tem != NULL_TREE)
return tem;
if ((TREE_CODE (arg0) == PLUS_EXPR || TREE_CODE (arg0) == MINUS_EXPR)
&& operand_equal_p (TREE_OPERAND (arg0, 0), arg1, 0)
&& TREE_CODE (TREE_OPERAND (arg0, 1)) == REAL_CST
&& !HONOR_SNANS (arg0))
{
tree arg01 = TREE_OPERAND (arg0, 1);
enum tree_code code0 = TREE_CODE (arg0);
int is_positive = REAL_VALUE_NEGATIVE (TREE_REAL_CST (arg01)) ? -1 : 1;
if (code == GT_EXPR
&& ((code0 == MINUS_EXPR && is_positive >= 0)
|| (code0 == PLUS_EXPR && is_positive <= 0)))
return constant_boolean_node (0, type);
if (code == LT_EXPR
&& ((code0 == PLUS_EXPR && is_positive >= 0)
|| (code0 == MINUS_EXPR && is_positive <= 0)))
return constant_boolean_node (0, type);
if (!HONOR_NANS (arg1)
&& code == LE_EXPR
&& ((code0 == MINUS_EXPR && is_positive >= 0)
|| (code0 == PLUS_EXPR && is_positive <= 0)))
return constant_boolean_node (1, type);
if (!HONOR_NANS (arg1)
&& code == GE_EXPR
&& ((code0 == PLUS_EXPR && is_positive >= 0)
|| (code0 == MINUS_EXPR && is_positive <= 0)))
return constant_boolean_node (1, type);
}
if (code == LE_EXPR
&& TREE_CODE (arg1) == INTEGER_CST
&& TREE_CODE (arg0) == ABS_EXPR
&& ! TREE_SIDE_EFFECTS (arg0)
&& (tem = negate_expr (arg1)) != 0
&& TREE_CODE (tem) == INTEGER_CST
&& !TREE_OVERFLOW (tem))
return fold_build2_loc (loc, TRUTH_ANDIF_EXPR, type,
build2 (GE_EXPR, type,
TREE_OPERAND (arg0, 0), tem),
build2 (LE_EXPR, type,
TREE_OPERAND (arg0, 0), arg1));
strict_overflow_p = false;
if (code == GE_EXPR
&& (integer_zerop (arg1)
|| (! HONOR_NANS (arg0)
&& real_zerop (arg1)))
&& tree_expr_nonnegative_warnv_p (arg0, &strict_overflow_p))
{
if (strict_overflow_p)
fold_overflow_warning (("assuming signed overflow does not occur "
"when simplifying comparison of "
"absolute value and zero"),
WARN_STRICT_OVERFLOW_CONDITIONAL);
return omit_one_operand_loc (loc, type,
constant_boolean_node (true, type),
arg0);
}
strict_overflow_p = false;
if (code == LT_EXPR
&& (integer_zerop (arg1) || real_zerop (arg1))
&& tree_expr_nonnegative_warnv_p (arg0, &strict_overflow_p))
{
if (strict_overflow_p)
fold_overflow_warning (("assuming signed overflow does not occur "
"when simplifying comparison of "
"absolute value and zero"),
WARN_STRICT_OVERFLOW_CONDITIONAL);
return omit_one_operand_loc (loc, type,
constant_boolean_node (false, type),
arg0);
}
if ((code == LT_EXPR || code == GE_EXPR)
&& TYPE_UNSIGNED (TREE_TYPE (arg0))
&& TREE_CODE (arg1) == LSHIFT_EXPR
&& integer_onep (TREE_OPERAND (arg1, 0)))
return build2_loc (loc, code == LT_EXPR ? EQ_EXPR : NE_EXPR, type,
build2 (RSHIFT_EXPR, TREE_TYPE (arg0), arg0,
TREE_OPERAND (arg1, 1)),
build_zero_cst (TREE_TYPE (arg0)));
if ((code == LT_EXPR || code == GE_EXPR)
&& TYPE_UNSIGNED (TREE_TYPE (arg0))
&& CONVERT_EXPR_P (arg1)
&& TREE_CODE (TREE_OPERAND (arg1, 0)) == LSHIFT_EXPR
&& (element_precision (TREE_TYPE (arg1))
>= element_precision (TREE_TYPE (TREE_OPERAND (arg1, 0))))
&& (TYPE_UNSIGNED (TREE_TYPE (TREE_OPERAND (arg1, 0)))
|| (element_precision (TREE_TYPE (arg1))
== element_precision (TREE_TYPE (TREE_OPERAND (arg1, 0)))))
&& integer_onep (TREE_OPERAND (TREE_OPERAND (arg1, 0), 0)))
{
tem = build2 (RSHIFT_EXPR, TREE_TYPE (arg0), arg0,
TREE_OPERAND (TREE_OPERAND (arg1, 0), 1));
return build2_loc (loc, code == LT_EXPR ? EQ_EXPR : NE_EXPR, type,
fold_convert_loc (loc, TREE_TYPE (arg0), tem),
build_zero_cst (TREE_TYPE (arg0)));
}
return NULL_TREE;
case UNORDERED_EXPR:
case ORDERED_EXPR:
case UNLT_EXPR:
case UNLE_EXPR:
case UNGT_EXPR:
case UNGE_EXPR:
case UNEQ_EXPR:
case LTGT_EXPR:
{
tree targ0 = strip_float_extensions (arg0);
tree targ1 = strip_float_extensions (arg1);
tree newtype = TREE_TYPE (targ0);
if (TYPE_PRECISION (TREE_TYPE (targ1)) > TYPE_PRECISION (newtype))
newtype = TREE_TYPE (targ1);
if (TYPE_PRECISION (newtype) < TYPE_PRECISION (TREE_TYPE (arg0)))
return fold_build2_loc (loc, code, type,
fold_convert_loc (loc, newtype, targ0),
fold_convert_loc (loc, newtype, targ1));
}
return NULL_TREE;
case COMPOUND_EXPR:
if (TREE_SIDE_EFFECTS (arg0) || TREE_CONSTANT (arg1))
return NULL_TREE;
tem = integer_zerop (arg1) ? build1 (NOP_EXPR, type, arg1)
: fold_convert_loc (loc, type, arg1);
return pedantic_non_lvalue_loc (loc, tem);
case ASSERT_EXPR:
gcc_unreachable ();
default:
return NULL_TREE;
} 
}
struct contains_label_data
{
hash_set<tree> *pset;
bool inside_switch_p;
};
static tree
contains_label_1 (tree *tp, int *walk_subtrees, void *data)
{
contains_label_data *d = (contains_label_data *) data;
switch (TREE_CODE (*tp))
{
case LABEL_EXPR:
return *tp;
case CASE_LABEL_EXPR:
if (!d->inside_switch_p)
return *tp;
return NULL_TREE;
case SWITCH_EXPR:
if (!d->inside_switch_p)
{
if (walk_tree (&SWITCH_COND (*tp), contains_label_1, data, d->pset))
return *tp;
d->inside_switch_p = true;
if (walk_tree (&SWITCH_BODY (*tp), contains_label_1, data, d->pset))
return *tp;
d->inside_switch_p = false;
*walk_subtrees = 0;
}
return NULL_TREE;
case GOTO_EXPR:
*walk_subtrees = 0;
return NULL_TREE;
default:
return NULL_TREE;
}
}
static bool
contains_label_p (tree st)
{
hash_set<tree> pset;
contains_label_data data = { &pset, false };
return walk_tree (&st, contains_label_1, &data, &pset) != NULL_TREE;
}
tree
fold_ternary_loc (location_t loc, enum tree_code code, tree type,
tree op0, tree op1, tree op2)
{
tree tem;
tree arg0 = NULL_TREE, arg1 = NULL_TREE, arg2 = NULL_TREE;
enum tree_code_class kind = TREE_CODE_CLASS (code);
gcc_assert (IS_EXPR_CODE_CLASS (kind)
&& TREE_CODE_LENGTH (code) == 3);
if (commutative_ternary_tree_code (code)
&& tree_swap_operands_p (op0, op1))
return fold_build3_loc (loc, code, type, op1, op0, op2);
tem = generic_simplify (loc, code, type, op0, op1, op2);
if (tem)
return tem;
if (op0)
{
arg0 = op0;
STRIP_NOPS (arg0);
}
if (op1)
{
arg1 = op1;
STRIP_NOPS (arg1);
}
if (op2)
{
arg2 = op2;
STRIP_NOPS (arg2);
}
switch (code)
{
case COMPONENT_REF:
if (TREE_CODE (arg0) == CONSTRUCTOR
&& ! type_contains_placeholder_p (TREE_TYPE (arg0)))
{
unsigned HOST_WIDE_INT idx;
tree field, value;
FOR_EACH_CONSTRUCTOR_ELT (CONSTRUCTOR_ELTS (arg0), idx, field, value)
if (field == arg1)
return value;
}
return NULL_TREE;
case COND_EXPR:
case VEC_COND_EXPR:
if (TREE_CODE (arg0) == INTEGER_CST)
{
tree unused_op = integer_zerop (arg0) ? op1 : op2;
tem = integer_zerop (arg0) ? op2 : op1;
if ((!TREE_SIDE_EFFECTS (unused_op)
|| !contains_label_p (unused_op))
&& (! VOID_TYPE_P (TREE_TYPE (tem))
|| VOID_TYPE_P (type)))
return pedantic_non_lvalue_loc (loc, tem);
return NULL_TREE;
}
else if (TREE_CODE (arg0) == VECTOR_CST)
{
unsigned HOST_WIDE_INT nelts;
if ((TREE_CODE (arg1) == VECTOR_CST
|| TREE_CODE (arg1) == CONSTRUCTOR)
&& (TREE_CODE (arg2) == VECTOR_CST
|| TREE_CODE (arg2) == CONSTRUCTOR)
&& TYPE_VECTOR_SUBPARTS (type).is_constant (&nelts))
{
vec_perm_builder sel (nelts, nelts, 1);
for (unsigned int i = 0; i < nelts; i++)
{
tree val = VECTOR_CST_ELT (arg0, i);
if (integer_all_onesp (val))
sel.quick_push (i);
else if (integer_zerop (val))
sel.quick_push (nelts + i);
else 
return NULL_TREE;
}
vec_perm_indices indices (sel, 2, nelts);
tree t = fold_vec_perm (type, arg1, arg2, indices);
if (t != NULL_TREE)
return t;
}
}
if (COMPARISON_CLASS_P (arg0)
&& operand_equal_for_comparison_p (TREE_OPERAND (arg0, 0), op1)
&& !HONOR_SIGNED_ZEROS (element_mode (op1)))
{
tem = fold_cond_expr_with_comparison (loc, type, arg0, op1, op2);
if (tem)
return tem;
}
if (COMPARISON_CLASS_P (arg0)
&& operand_equal_for_comparison_p (TREE_OPERAND (arg0, 0), op2)
&& !HONOR_SIGNED_ZEROS (element_mode (op2)))
{
location_t loc0 = expr_location_or (arg0, loc);
tem = fold_invert_truthvalue (loc0, arg0);
if (tem && COMPARISON_CLASS_P (tem))
{
tem = fold_cond_expr_with_comparison (loc, type, tem, op2, op1);
if (tem)
return tem;
}
}
if (truth_value_p (TREE_CODE (arg0))
&& tree_swap_operands_p (op1, op2))
{
location_t loc0 = expr_location_or (arg0, loc);
tem = fold_invert_truthvalue (loc0, arg0);
if (tem)
return fold_build3_loc (loc, code, type, tem, op2, op1);
}
if ((code == VEC_COND_EXPR ? integer_all_onesp (op1)
: (integer_onep (op1)
&& !VECTOR_TYPE_P (type)))
&& integer_zerop (op2)
&& type == TREE_TYPE (arg0))
return pedantic_non_lvalue_loc (loc, arg0);
if (integer_zerop (op1)
&& code == COND_EXPR
&& integer_onep (op2)
&& !VECTOR_TYPE_P (type)
&& truth_value_p (TREE_CODE (arg0)))
return pedantic_non_lvalue_loc (loc,
fold_convert_loc (loc, type,
invert_truthvalue_loc (loc,
arg0)));
if (TREE_CODE (arg0) == LT_EXPR
&& integer_zerop (TREE_OPERAND (arg0, 1))
&& integer_zerop (op2)
&& (tem = sign_bit_p (TREE_OPERAND (arg0, 0), arg1)))
{
tree tem2 = TREE_OPERAND (arg0, 0);
while (tem != tem2)
{
if (TREE_CODE (tem2) != NOP_EXPR
|| TYPE_UNSIGNED (TREE_TYPE (TREE_OPERAND (tem2, 0))))
{
tem = NULL_TREE;
break;
}
tem2 = TREE_OPERAND (tem2, 0);
}
if (tem
&& TYPE_PRECISION (TREE_TYPE (tem))
< TYPE_PRECISION (TREE_TYPE (arg1))
&& TYPE_PRECISION (TREE_TYPE (tem))
< TYPE_PRECISION (type))
{
int inner_width, outer_width;
tree tem_type;
inner_width = TYPE_PRECISION (TREE_TYPE (tem));
outer_width = TYPE_PRECISION (TREE_TYPE (arg1));
if (outer_width > TYPE_PRECISION (type))
outer_width = TYPE_PRECISION (type);
wide_int mask = wi::shifted_mask
(inner_width, outer_width - inner_width, false,
TYPE_PRECISION (TREE_TYPE (arg1)));
wide_int common = mask & wi::to_wide (arg1);
if (common == mask)
{
tem_type = signed_type_for (TREE_TYPE (tem));
tem = fold_convert_loc (loc, tem_type, tem);
}
else if (common == 0)
{
tem_type = unsigned_type_for (TREE_TYPE (tem));
tem = fold_convert_loc (loc, tem_type, tem);
}
else
tem = NULL;
}
if (tem)
return
fold_convert_loc (loc, type,
fold_build2_loc (loc, BIT_AND_EXPR,
TREE_TYPE (tem), tem,
fold_convert_loc (loc,
TREE_TYPE (tem),
arg1)));
}
if (TREE_CODE (arg0) == BIT_AND_EXPR
&& integer_onep (TREE_OPERAND (arg0, 1))
&& integer_zerop (op2)
&& integer_pow2p (arg1))
{
tree tem = TREE_OPERAND (arg0, 0);
STRIP_NOPS (tem);
if (TREE_CODE (tem) == RSHIFT_EXPR
&& tree_fits_uhwi_p (TREE_OPERAND (tem, 1))
&& (unsigned HOST_WIDE_INT) tree_log2 (arg1)
== tree_to_uhwi (TREE_OPERAND (tem, 1)))
return fold_build2_loc (loc, BIT_AND_EXPR, type,
fold_convert_loc (loc, type,
TREE_OPERAND (tem, 0)),
op1);
}
if (integer_zerop (op2)
&& TREE_CODE (arg0) == NE_EXPR
&& integer_zerop (TREE_OPERAND (arg0, 1))
&& integer_pow2p (arg1)
&& TREE_CODE (TREE_OPERAND (arg0, 0)) == BIT_AND_EXPR
&& operand_equal_p (TREE_OPERAND (TREE_OPERAND (arg0, 0), 1),
arg1, OEP_ONLY_CONST)
&& integer_pow2p (TREE_OPERAND (TREE_OPERAND (arg0, 0), 1)))
return pedantic_non_lvalue_loc (loc,
fold_convert_loc (loc, type,
TREE_OPERAND (arg0,
0)));
if (code == VEC_COND_EXPR)
return NULL_TREE;
if (integer_zerop (op2)
&& truth_value_p (TREE_CODE (arg0))
&& truth_value_p (TREE_CODE (arg1))
&& (code == VEC_COND_EXPR || !VECTOR_TYPE_P (type)))
return fold_build2_loc (loc, code == VEC_COND_EXPR ? BIT_AND_EXPR
: TRUTH_ANDIF_EXPR,
type, fold_convert_loc (loc, type, arg0), op1);
if (code == VEC_COND_EXPR ? integer_all_onesp (op2) : integer_onep (op2)
&& truth_value_p (TREE_CODE (arg0))
&& truth_value_p (TREE_CODE (arg1))
&& (code == VEC_COND_EXPR || !VECTOR_TYPE_P (type)))
{
location_t loc0 = expr_location_or (arg0, loc);
tem = fold_invert_truthvalue (loc0, arg0);
if (tem)
return fold_build2_loc (loc, code == VEC_COND_EXPR
? BIT_IOR_EXPR
: TRUTH_ORIF_EXPR,
type, fold_convert_loc (loc, type, tem),
op1);
}
if (integer_zerop (arg1)
&& truth_value_p (TREE_CODE (arg0))
&& truth_value_p (TREE_CODE (op2))
&& (code == VEC_COND_EXPR || !VECTOR_TYPE_P (type)))
{
location_t loc0 = expr_location_or (arg0, loc);
tem = fold_invert_truthvalue (loc0, arg0);
if (tem)
return fold_build2_loc (loc, code == VEC_COND_EXPR
? BIT_AND_EXPR : TRUTH_ANDIF_EXPR,
type, fold_convert_loc (loc, type, tem),
op2);
}
if (code == VEC_COND_EXPR ? integer_all_onesp (arg1) : integer_onep (arg1)
&& truth_value_p (TREE_CODE (arg0))
&& truth_value_p (TREE_CODE (op2))
&& (code == VEC_COND_EXPR || !VECTOR_TYPE_P (type)))
return fold_build2_loc (loc, code == VEC_COND_EXPR
? BIT_IOR_EXPR : TRUTH_ORIF_EXPR,
type, fold_convert_loc (loc, type, arg0), op2);
return NULL_TREE;
case CALL_EXPR:
gcc_unreachable ();
case BIT_FIELD_REF:
if (TREE_CODE (arg0) == VECTOR_CST
&& (type == TREE_TYPE (TREE_TYPE (arg0))
|| (VECTOR_TYPE_P (type)
&& TREE_TYPE (type) == TREE_TYPE (TREE_TYPE (arg0))))
&& tree_fits_uhwi_p (op1)
&& tree_fits_uhwi_p (op2))
{
tree eltype = TREE_TYPE (TREE_TYPE (arg0));
unsigned HOST_WIDE_INT width = tree_to_uhwi (TYPE_SIZE (eltype));
unsigned HOST_WIDE_INT n = tree_to_uhwi (arg1);
unsigned HOST_WIDE_INT idx = tree_to_uhwi (op2);
if (n != 0
&& (idx % width) == 0
&& (n % width) == 0
&& known_le ((idx + n) / width,
TYPE_VECTOR_SUBPARTS (TREE_TYPE (arg0))))
{
idx = idx / width;
n = n / width;
if (TREE_CODE (arg0) == VECTOR_CST)
{
if (n == 1)
{
tem = VECTOR_CST_ELT (arg0, idx);
if (VECTOR_TYPE_P (type))
tem = fold_build1 (VIEW_CONVERT_EXPR, type, tem);
return tem;
}
tree_vector_builder vals (type, n, 1);
for (unsigned i = 0; i < n; ++i)
vals.quick_push (VECTOR_CST_ELT (arg0, idx + i));
return vals.build ();
}
}
}
if (CONSTANT_CLASS_P (arg0)
&& can_native_interpret_type_p (type)
&& BITS_PER_UNIT == 8
&& tree_fits_uhwi_p (op1)
&& tree_fits_uhwi_p (op2))
{
unsigned HOST_WIDE_INT bitpos = tree_to_uhwi (op2);
unsigned HOST_WIDE_INT bitsize = tree_to_uhwi (op1);
if (bitpos % BITS_PER_UNIT == 0
&& bitsize % BITS_PER_UNIT == 0
&& bitsize <= MAX_BITSIZE_MODE_ANY_MODE)
{
unsigned char b[MAX_BITSIZE_MODE_ANY_MODE / BITS_PER_UNIT];
unsigned HOST_WIDE_INT len
= native_encode_expr (arg0, b, bitsize / BITS_PER_UNIT,
bitpos / BITS_PER_UNIT);
if (len > 0
&& len * BITS_PER_UNIT >= bitsize)
{
tree v = native_interpret_expr (type, b,
bitsize / BITS_PER_UNIT);
if (v)
return v;
}
}
}
return NULL_TREE;
case FMA_EXPR:
if (TREE_CODE (arg0) == INTEGER_CST
&& TREE_CODE (arg1) == INTEGER_CST)
return fold_build2_loc (loc, PLUS_EXPR, type,
const_binop (MULT_EXPR, arg0, arg1), arg2);
if (integer_zerop (arg2))
return fold_build2_loc (loc, MULT_EXPR, type, arg0, arg1);
return fold_fma (loc, type, arg0, arg1, arg2);
case VEC_PERM_EXPR:
if (TREE_CODE (arg2) == VECTOR_CST)
{
vec_perm_builder builder;
if (!tree_to_vec_perm_builder (&builder, arg2))
return NULL_TREE;
poly_uint64 nelts = TYPE_VECTOR_SUBPARTS (type);
bool single_arg = (op0 == op1);
vec_perm_indices sel (builder, single_arg ? 1 : 2, nelts);
if (sel.series_p (0, 1, 0, 1))
return op0;
if (sel.series_p (0, 1, nelts, 1))
return op1;
if (!single_arg)
{
if (sel.all_from_input_p (0))
op1 = op0;
else if (sel.all_from_input_p (1))
{
op0 = op1;
sel.rotate_inputs (1);
}
}
if ((TREE_CODE (op0) == VECTOR_CST
|| TREE_CODE (op0) == CONSTRUCTOR)
&& (TREE_CODE (op1) == VECTOR_CST
|| TREE_CODE (op1) == CONSTRUCTOR))
{
tree t = fold_vec_perm (type, op0, op1, sel);
if (t != NULL_TREE)
return t;
}
bool changed = (op0 == op1 && !single_arg);
if (arg2 == op2 && sel.encoding () != builder)
{
if (sel.ninputs () == 2
|| can_vec_perm_const_p (TYPE_MODE (type), sel, false))
op2 = vec_perm_indices_to_tree (TREE_TYPE (arg2), sel);
else
{
vec_perm_indices sel2 (builder, 2, nelts);
if (can_vec_perm_const_p (TYPE_MODE (type), sel2, false))
op2 = vec_perm_indices_to_tree (TREE_TYPE (arg2), sel2);
else
op2 = vec_perm_indices_to_tree (TREE_TYPE (arg2), sel);
}
changed = true;
}
if (changed)
return build3_loc (loc, VEC_PERM_EXPR, type, op0, op1, op2);
}
return NULL_TREE;
case BIT_INSERT_EXPR:
if (TREE_CODE (arg0) == INTEGER_CST
&& TREE_CODE (arg1) == INTEGER_CST)
{
unsigned HOST_WIDE_INT bitpos = tree_to_uhwi (op2);
unsigned bitsize = TYPE_PRECISION (TREE_TYPE (arg1));
wide_int tem = (wi::to_wide (arg0)
& wi::shifted_mask (bitpos, bitsize, true,
TYPE_PRECISION (type)));
wide_int tem2
= wi::lshift (wi::zext (wi::to_wide (arg1, TYPE_PRECISION (type)),
bitsize), bitpos);
return wide_int_to_tree (type, wi::bit_or (tem, tem2));
}
else if (TREE_CODE (arg0) == VECTOR_CST
&& CONSTANT_CLASS_P (arg1)
&& types_compatible_p (TREE_TYPE (TREE_TYPE (arg0)),
TREE_TYPE (arg1)))
{
unsigned HOST_WIDE_INT bitpos = tree_to_uhwi (op2);
unsigned HOST_WIDE_INT elsize
= tree_to_uhwi (TYPE_SIZE (TREE_TYPE (arg1)));
if (bitpos % elsize == 0)
{
unsigned k = bitpos / elsize;
unsigned HOST_WIDE_INT nelts;
if (operand_equal_p (VECTOR_CST_ELT (arg0, k), arg1, 0))
return arg0;
else if (VECTOR_CST_NELTS (arg0).is_constant (&nelts))
{
tree_vector_builder elts (type, nelts, 1);
elts.quick_grow (nelts);
for (unsigned HOST_WIDE_INT i = 0; i < nelts; ++i)
elts[i] = (i == k ? arg1 : VECTOR_CST_ELT (arg0, i));
return elts.build ();
}
}
}
return NULL_TREE;
default:
return NULL_TREE;
} 
}
tree
get_array_ctor_element_at_index (tree ctor, offset_int access_index)
{
tree index_type = NULL_TREE;
offset_int low_bound = 0;
if (TREE_CODE (TREE_TYPE (ctor)) == ARRAY_TYPE)
{
tree domain_type = TYPE_DOMAIN (TREE_TYPE (ctor));
if (domain_type && TYPE_MIN_VALUE (domain_type))
{
gcc_assert (TREE_CODE (TYPE_MIN_VALUE (domain_type)) == INTEGER_CST);
index_type = TREE_TYPE (TYPE_MIN_VALUE (domain_type));
low_bound = wi::to_offset (TYPE_MIN_VALUE (domain_type));
}
}
if (index_type)
access_index = wi::ext (access_index, TYPE_PRECISION (index_type),
TYPE_SIGN (index_type));
offset_int index = low_bound - 1;
if (index_type)
index = wi::ext (index, TYPE_PRECISION (index_type),
TYPE_SIGN (index_type));
offset_int max_index;
unsigned HOST_WIDE_INT cnt;
tree cfield, cval;
FOR_EACH_CONSTRUCTOR_ELT (CONSTRUCTOR_ELTS (ctor), cnt, cfield, cval)
{
if (cfield)
{
if (TREE_CODE (cfield) == INTEGER_CST)
max_index = index = wi::to_offset (cfield);
else
{
gcc_assert (TREE_CODE (cfield) == RANGE_EXPR);
index = wi::to_offset (TREE_OPERAND (cfield, 0));
max_index = wi::to_offset (TREE_OPERAND (cfield, 1));
}
}
else
{
index += 1;
if (index_type)
index = wi::ext (index, TYPE_PRECISION (index_type),
TYPE_SIGN (index_type));
max_index = index;
}
if (wi::cmpu (access_index, index) >= 0
&& wi::cmpu (access_index, max_index) <= 0)
return cval;
}
return NULL_TREE;
}
#ifdef ENABLE_FOLD_CHECKING
# define fold(x) fold_1 (x)
static tree fold_1 (tree);
static
#endif
tree
fold (tree expr)
{
const tree t = expr;
enum tree_code code = TREE_CODE (t);
enum tree_code_class kind = TREE_CODE_CLASS (code);
tree tem;
location_t loc = EXPR_LOCATION (expr);
if (kind == tcc_constant)
return t;
if (kind == tcc_vl_exp)
{
if (code == CALL_EXPR)
{
tem = fold_call_expr (loc, expr, false);
return tem ? tem : expr;
}
return expr;
}
if (IS_EXPR_CODE_CLASS (kind))
{
tree type = TREE_TYPE (t);
tree op0, op1, op2;
switch (TREE_CODE_LENGTH (code))
{
case 1:
op0 = TREE_OPERAND (t, 0);
tem = fold_unary_loc (loc, code, type, op0);
return tem ? tem : expr;
case 2:
op0 = TREE_OPERAND (t, 0);
op1 = TREE_OPERAND (t, 1);
tem = fold_binary_loc (loc, code, type, op0, op1);
return tem ? tem : expr;
case 3:
op0 = TREE_OPERAND (t, 0);
op1 = TREE_OPERAND (t, 1);
op2 = TREE_OPERAND (t, 2);
tem = fold_ternary_loc (loc, code, type, op0, op1, op2);
return tem ? tem : expr;
default:
break;
}
}
switch (code)
{
case ARRAY_REF:
{
tree op0 = TREE_OPERAND (t, 0);
tree op1 = TREE_OPERAND (t, 1);
if (TREE_CODE (op1) == INTEGER_CST
&& TREE_CODE (op0) == CONSTRUCTOR
&& ! type_contains_placeholder_p (TREE_TYPE (op0)))
{
tree val = get_array_ctor_element_at_index (op0,
wi::to_offset (op1));
if (val)
return val;
}
return t;
}
case CONSTRUCTOR:
{
tree type = TREE_TYPE (t);
if (TREE_CODE (type) != VECTOR_TYPE)
return t;
unsigned i;
tree val;
FOR_EACH_CONSTRUCTOR_VALUE (CONSTRUCTOR_ELTS (t), i, val)
if (! CONSTANT_CLASS_P (val))
return t;
return build_vector_from_ctor (type, CONSTRUCTOR_ELTS (t));
}
case CONST_DECL:
return fold (DECL_INITIAL (t));
default:
return t;
} 
}
#ifdef ENABLE_FOLD_CHECKING
#undef fold
static void fold_checksum_tree (const_tree, struct md5_ctx *,
hash_table<nofree_ptr_hash<const tree_node> > *);
static void fold_check_failed (const_tree, const_tree);
void print_fold_checksum (const_tree);
tree
fold (tree expr)
{
tree ret;
struct md5_ctx ctx;
unsigned char checksum_before[16], checksum_after[16];
hash_table<nofree_ptr_hash<const tree_node> > ht (32);
md5_init_ctx (&ctx);
fold_checksum_tree (expr, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_before);
ht.empty ();
ret = fold_1 (expr);
md5_init_ctx (&ctx);
fold_checksum_tree (expr, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_after);
if (memcmp (checksum_before, checksum_after, 16))
fold_check_failed (expr, ret);
return ret;
}
void
print_fold_checksum (const_tree expr)
{
struct md5_ctx ctx;
unsigned char checksum[16], cnt;
hash_table<nofree_ptr_hash<const tree_node> > ht (32);
md5_init_ctx (&ctx);
fold_checksum_tree (expr, &ctx, &ht);
md5_finish_ctx (&ctx, checksum);
for (cnt = 0; cnt < 16; ++cnt)
fprintf (stderr, "%02x", checksum[cnt]);
putc ('\n', stderr);
}
static void
fold_check_failed (const_tree expr ATTRIBUTE_UNUSED, const_tree ret ATTRIBUTE_UNUSED)
{
internal_error ("fold check: original tree changed by fold");
}
static void
fold_checksum_tree (const_tree expr, struct md5_ctx *ctx,
hash_table<nofree_ptr_hash <const tree_node> > *ht)
{
const tree_node **slot;
enum tree_code code;
union tree_node buf;
int i, len;
recursive_label:
if (expr == NULL)
return;
slot = ht->find_slot (expr, INSERT);
if (*slot != NULL)
return;
*slot = expr;
code = TREE_CODE (expr);
if (TREE_CODE_CLASS (code) == tcc_declaration
&& HAS_DECL_ASSEMBLER_NAME_P (expr))
{
memcpy ((char *) &buf, expr, tree_size (expr));
SET_DECL_ASSEMBLER_NAME ((tree)&buf, NULL);
buf.decl_with_vis.symtab_node = NULL;
expr = (tree) &buf;
}
else if (TREE_CODE_CLASS (code) == tcc_type
&& (TYPE_POINTER_TO (expr)
|| TYPE_REFERENCE_TO (expr)
|| TYPE_CACHED_VALUES_P (expr)
|| TYPE_CONTAINS_PLACEHOLDER_INTERNAL (expr)
|| TYPE_NEXT_VARIANT (expr)
|| TYPE_ALIAS_SET_KNOWN_P (expr)))
{
tree tmp;
memcpy ((char *) &buf, expr, tree_size (expr));
expr = tmp = (tree) &buf;
TYPE_CONTAINS_PLACEHOLDER_INTERNAL (tmp) = 0;
TYPE_POINTER_TO (tmp) = NULL;
TYPE_REFERENCE_TO (tmp) = NULL;
TYPE_NEXT_VARIANT (tmp) = NULL;
TYPE_ALIAS_SET (tmp) = -1;
if (TYPE_CACHED_VALUES_P (tmp))
{
TYPE_CACHED_VALUES_P (tmp) = 0;
TYPE_CACHED_VALUES (tmp) = NULL;
}
}
md5_process_bytes (expr, tree_size (expr), ctx);
if (CODE_CONTAINS_STRUCT (code, TS_TYPED))
fold_checksum_tree (TREE_TYPE (expr), ctx, ht);
if (TREE_CODE_CLASS (code) != tcc_type
&& TREE_CODE_CLASS (code) != tcc_declaration
&& code != TREE_LIST
&& code != SSA_NAME
&& CODE_CONTAINS_STRUCT (code, TS_COMMON))
fold_checksum_tree (TREE_CHAIN (expr), ctx, ht);
switch (TREE_CODE_CLASS (code))
{
case tcc_constant:
switch (code)
{
case STRING_CST:
md5_process_bytes (TREE_STRING_POINTER (expr),
TREE_STRING_LENGTH (expr), ctx);
break;
case COMPLEX_CST:
fold_checksum_tree (TREE_REALPART (expr), ctx, ht);
fold_checksum_tree (TREE_IMAGPART (expr), ctx, ht);
break;
case VECTOR_CST:
len = vector_cst_encoded_nelts (expr);
for (i = 0; i < len; ++i)
fold_checksum_tree (VECTOR_CST_ENCODED_ELT (expr, i), ctx, ht);
break;
default:
break;
}
break;
case tcc_exceptional:
switch (code)
{
case TREE_LIST:
fold_checksum_tree (TREE_PURPOSE (expr), ctx, ht);
fold_checksum_tree (TREE_VALUE (expr), ctx, ht);
expr = TREE_CHAIN (expr);
goto recursive_label;
break;
case TREE_VEC:
for (i = 0; i < TREE_VEC_LENGTH (expr); ++i)
fold_checksum_tree (TREE_VEC_ELT (expr, i), ctx, ht);
break;
default:
break;
}
break;
case tcc_expression:
case tcc_reference:
case tcc_comparison:
case tcc_unary:
case tcc_binary:
case tcc_statement:
case tcc_vl_exp:
len = TREE_OPERAND_LENGTH (expr);
for (i = 0; i < len; ++i)
fold_checksum_tree (TREE_OPERAND (expr, i), ctx, ht);
break;
case tcc_declaration:
fold_checksum_tree (DECL_NAME (expr), ctx, ht);
fold_checksum_tree (DECL_CONTEXT (expr), ctx, ht);
if (CODE_CONTAINS_STRUCT (TREE_CODE (expr), TS_DECL_COMMON))
{
fold_checksum_tree (DECL_SIZE (expr), ctx, ht);
fold_checksum_tree (DECL_SIZE_UNIT (expr), ctx, ht);
fold_checksum_tree (DECL_INITIAL (expr), ctx, ht);
fold_checksum_tree (DECL_ABSTRACT_ORIGIN (expr), ctx, ht);
fold_checksum_tree (DECL_ATTRIBUTES (expr), ctx, ht);
}
if (CODE_CONTAINS_STRUCT (TREE_CODE (expr), TS_DECL_NON_COMMON))
{
if (TREE_CODE (expr) == FUNCTION_DECL)
{
fold_checksum_tree (DECL_VINDEX (expr), ctx, ht);
fold_checksum_tree (DECL_ARGUMENTS (expr), ctx, ht);
}
fold_checksum_tree (DECL_RESULT_FLD (expr), ctx, ht);
}
break;
case tcc_type:
if (TREE_CODE (expr) == ENUMERAL_TYPE)
fold_checksum_tree (TYPE_VALUES (expr), ctx, ht);
fold_checksum_tree (TYPE_SIZE (expr), ctx, ht);
fold_checksum_tree (TYPE_SIZE_UNIT (expr), ctx, ht);
fold_checksum_tree (TYPE_ATTRIBUTES (expr), ctx, ht);
fold_checksum_tree (TYPE_NAME (expr), ctx, ht);
if (INTEGRAL_TYPE_P (expr)
|| SCALAR_FLOAT_TYPE_P (expr))
{
fold_checksum_tree (TYPE_MIN_VALUE (expr), ctx, ht);
fold_checksum_tree (TYPE_MAX_VALUE (expr), ctx, ht);
}
fold_checksum_tree (TYPE_MAIN_VARIANT (expr), ctx, ht);
if (TREE_CODE (expr) == RECORD_TYPE
|| TREE_CODE (expr) == UNION_TYPE
|| TREE_CODE (expr) == QUAL_UNION_TYPE)
fold_checksum_tree (TYPE_BINFO (expr), ctx, ht);
fold_checksum_tree (TYPE_CONTEXT (expr), ctx, ht);
break;
default:
break;
}
}
DEBUG_FUNCTION void
debug_fold_checksum (const_tree t)
{
int i;
unsigned char checksum[16];
struct md5_ctx ctx;
hash_table<nofree_ptr_hash<const tree_node> > ht (32);
md5_init_ctx (&ctx);
fold_checksum_tree (t, &ctx, &ht);
md5_finish_ctx (&ctx, checksum);
ht.empty ();
for (i = 0; i < 16; i++)
fprintf (stderr, "%d ", checksum[i]);
fprintf (stderr, "\n");
}
#endif
tree
fold_build1_loc (location_t loc,
enum tree_code code, tree type, tree op0 MEM_STAT_DECL)
{
tree tem;
#ifdef ENABLE_FOLD_CHECKING
unsigned char checksum_before[16], checksum_after[16];
struct md5_ctx ctx;
hash_table<nofree_ptr_hash<const tree_node> > ht (32);
md5_init_ctx (&ctx);
fold_checksum_tree (op0, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_before);
ht.empty ();
#endif
tem = fold_unary_loc (loc, code, type, op0);
if (!tem)
tem = build1_loc (loc, code, type, op0 PASS_MEM_STAT);
#ifdef ENABLE_FOLD_CHECKING
md5_init_ctx (&ctx);
fold_checksum_tree (op0, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_after);
if (memcmp (checksum_before, checksum_after, 16))
fold_check_failed (op0, tem);
#endif
return tem;
}
tree
fold_build2_loc (location_t loc,
enum tree_code code, tree type, tree op0, tree op1
MEM_STAT_DECL)
{
tree tem;
#ifdef ENABLE_FOLD_CHECKING
unsigned char checksum_before_op0[16],
checksum_before_op1[16],
checksum_after_op0[16],
checksum_after_op1[16];
struct md5_ctx ctx;
hash_table<nofree_ptr_hash<const tree_node> > ht (32);
md5_init_ctx (&ctx);
fold_checksum_tree (op0, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_before_op0);
ht.empty ();
md5_init_ctx (&ctx);
fold_checksum_tree (op1, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_before_op1);
ht.empty ();
#endif
tem = fold_binary_loc (loc, code, type, op0, op1);
if (!tem)
tem = build2_loc (loc, code, type, op0, op1 PASS_MEM_STAT);
#ifdef ENABLE_FOLD_CHECKING
md5_init_ctx (&ctx);
fold_checksum_tree (op0, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_after_op0);
ht.empty ();
if (memcmp (checksum_before_op0, checksum_after_op0, 16))
fold_check_failed (op0, tem);
md5_init_ctx (&ctx);
fold_checksum_tree (op1, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_after_op1);
if (memcmp (checksum_before_op1, checksum_after_op1, 16))
fold_check_failed (op1, tem);
#endif
return tem;
}
tree
fold_build3_loc (location_t loc, enum tree_code code, tree type,
tree op0, tree op1, tree op2 MEM_STAT_DECL)
{
tree tem;
#ifdef ENABLE_FOLD_CHECKING
unsigned char checksum_before_op0[16],
checksum_before_op1[16],
checksum_before_op2[16],
checksum_after_op0[16],
checksum_after_op1[16],
checksum_after_op2[16];
struct md5_ctx ctx;
hash_table<nofree_ptr_hash<const tree_node> > ht (32);
md5_init_ctx (&ctx);
fold_checksum_tree (op0, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_before_op0);
ht.empty ();
md5_init_ctx (&ctx);
fold_checksum_tree (op1, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_before_op1);
ht.empty ();
md5_init_ctx (&ctx);
fold_checksum_tree (op2, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_before_op2);
ht.empty ();
#endif
gcc_assert (TREE_CODE_CLASS (code) != tcc_vl_exp);
tem = fold_ternary_loc (loc, code, type, op0, op1, op2);
if (!tem)
tem = build3_loc (loc, code, type, op0, op1, op2 PASS_MEM_STAT);
#ifdef ENABLE_FOLD_CHECKING
md5_init_ctx (&ctx);
fold_checksum_tree (op0, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_after_op0);
ht.empty ();
if (memcmp (checksum_before_op0, checksum_after_op0, 16))
fold_check_failed (op0, tem);
md5_init_ctx (&ctx);
fold_checksum_tree (op1, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_after_op1);
ht.empty ();
if (memcmp (checksum_before_op1, checksum_after_op1, 16))
fold_check_failed (op1, tem);
md5_init_ctx (&ctx);
fold_checksum_tree (op2, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_after_op2);
if (memcmp (checksum_before_op2, checksum_after_op2, 16))
fold_check_failed (op2, tem);
#endif
return tem;
}
tree
fold_build_call_array_loc (location_t loc, tree type, tree fn,
int nargs, tree *argarray)
{
tree tem;
#ifdef ENABLE_FOLD_CHECKING
unsigned char checksum_before_fn[16],
checksum_before_arglist[16],
checksum_after_fn[16],
checksum_after_arglist[16];
struct md5_ctx ctx;
hash_table<nofree_ptr_hash<const tree_node> > ht (32);
int i;
md5_init_ctx (&ctx);
fold_checksum_tree (fn, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_before_fn);
ht.empty ();
md5_init_ctx (&ctx);
for (i = 0; i < nargs; i++)
fold_checksum_tree (argarray[i], &ctx, &ht);
md5_finish_ctx (&ctx, checksum_before_arglist);
ht.empty ();
#endif
tem = fold_builtin_call_array (loc, type, fn, nargs, argarray);
if (!tem)
tem = build_call_array_loc (loc, type, fn, nargs, argarray);
#ifdef ENABLE_FOLD_CHECKING
md5_init_ctx (&ctx);
fold_checksum_tree (fn, &ctx, &ht);
md5_finish_ctx (&ctx, checksum_after_fn);
ht.empty ();
if (memcmp (checksum_before_fn, checksum_after_fn, 16))
fold_check_failed (fn, tem);
md5_init_ctx (&ctx);
for (i = 0; i < nargs; i++)
fold_checksum_tree (argarray[i], &ctx, &ht);
md5_finish_ctx (&ctx, checksum_after_arglist);
if (memcmp (checksum_before_arglist, checksum_after_arglist, 16))
fold_check_failed (NULL_TREE, tem);
#endif
return tem;
}
#define START_FOLD_INIT \
int saved_signaling_nans = flag_signaling_nans;\
int saved_trapping_math = flag_trapping_math;\
int saved_rounding_math = flag_rounding_math;\
int saved_trapv = flag_trapv;\
int saved_folding_initializer = folding_initializer;\
flag_signaling_nans = 0;\
flag_trapping_math = 0;\
flag_rounding_math = 0;\
flag_trapv = 0;\
folding_initializer = 1;
#define END_FOLD_INIT \
flag_signaling_nans = saved_signaling_nans;\
flag_trapping_math = saved_trapping_math;\
flag_rounding_math = saved_rounding_math;\
flag_trapv = saved_trapv;\
folding_initializer = saved_folding_initializer;
tree
fold_build1_initializer_loc (location_t loc, enum tree_code code,
tree type, tree op)
{
tree result;
START_FOLD_INIT;
result = fold_build1_loc (loc, code, type, op);
END_FOLD_INIT;
return result;
}
tree
fold_build2_initializer_loc (location_t loc, enum tree_code code,
tree type, tree op0, tree op1)
{
tree result;
START_FOLD_INIT;
result = fold_build2_loc (loc, code, type, op0, op1);
END_FOLD_INIT;
return result;
}
tree
fold_build_call_array_initializer_loc (location_t loc, tree type, tree fn,
int nargs, tree *argarray)
{
tree result;
START_FOLD_INIT;
result = fold_build_call_array_loc (loc, type, fn, nargs, argarray);
END_FOLD_INIT;
return result;
}
#undef START_FOLD_INIT
#undef END_FOLD_INIT
int
multiple_of_p (tree type, const_tree top, const_tree bottom)
{
gimple *stmt;
tree t1, op1, op2;
if (operand_equal_p (top, bottom, 0))
return 1;
if (TREE_CODE (type) != INTEGER_TYPE)
return 0;
switch (TREE_CODE (top))
{
case BIT_AND_EXPR:
if (!integer_pow2p (bottom))
return 0;
return (multiple_of_p (type, TREE_OPERAND (top, 1), bottom)
|| multiple_of_p (type, TREE_OPERAND (top, 0), bottom));
case MULT_EXPR:
if (TREE_CODE (bottom) == INTEGER_CST)
{
op1 = TREE_OPERAND (top, 0);
op2 = TREE_OPERAND (top, 1);
if (TREE_CODE (op1) == INTEGER_CST)
std::swap (op1, op2);
if (TREE_CODE (op2) == INTEGER_CST)
{
if (multiple_of_p (type, op2, bottom))
return 1;
if (multiple_of_p (type, bottom, op2))
{
widest_int w = wi::sdiv_trunc (wi::to_widest (bottom),
wi::to_widest (op2));
if (wi::fits_to_tree_p (w, TREE_TYPE (bottom)))
{
op2 = wide_int_to_tree (TREE_TYPE (bottom), w);
return multiple_of_p (type, op1, op2);
}
}
return multiple_of_p (type, op1, bottom);
}
}
return (multiple_of_p (type, TREE_OPERAND (top, 1), bottom)
|| multiple_of_p (type, TREE_OPERAND (top, 0), bottom));
case MINUS_EXPR:
return (multiple_of_p (type, TREE_OPERAND (top, 1), bottom)
&& multiple_of_p (type, TREE_OPERAND (top, 0), bottom));
case PLUS_EXPR:
op1 = TREE_OPERAND (top, 1);
if (TYPE_UNSIGNED (type)
&& TREE_CODE (op1) == INTEGER_CST && tree_int_cst_sign_bit (op1))
op1 = fold_build1 (NEGATE_EXPR, type, op1);
return (multiple_of_p (type, op1, bottom)
&& multiple_of_p (type, TREE_OPERAND (top, 0), bottom));
case LSHIFT_EXPR:
if (TREE_CODE (TREE_OPERAND (top, 1)) == INTEGER_CST)
{
op1 = TREE_OPERAND (top, 1);
if (wi::gtu_p (TYPE_PRECISION (TREE_TYPE (size_one_node)),
wi::to_wide (op1))
&& (t1 = fold_convert (type,
const_binop (LSHIFT_EXPR, size_one_node,
op1))) != 0
&& !TREE_OVERFLOW (t1))
return multiple_of_p (type, t1, bottom);
}
return 0;
case NOP_EXPR:
if ((TREE_CODE (TREE_TYPE (TREE_OPERAND (top, 0))) != INTEGER_TYPE)
|| (TYPE_PRECISION (type)
< TYPE_PRECISION (TREE_TYPE (TREE_OPERAND (top, 0)))))
return 0;
case SAVE_EXPR:
return multiple_of_p (type, TREE_OPERAND (top, 0), bottom);
case COND_EXPR:
return (multiple_of_p (type, TREE_OPERAND (top, 1), bottom)
&& multiple_of_p (type, TREE_OPERAND (top, 2), bottom));
case INTEGER_CST:
if (TREE_CODE (bottom) != INTEGER_CST
|| integer_zerop (bottom)
|| (TYPE_UNSIGNED (type)
&& (tree_int_cst_sgn (top) < 0
|| tree_int_cst_sgn (bottom) < 0)))
return 0;
return wi::multiple_of_p (wi::to_widest (top), wi::to_widest (bottom),
SIGNED);
case SSA_NAME:
if (TREE_CODE (bottom) == INTEGER_CST
&& (stmt = SSA_NAME_DEF_STMT (top)) != NULL
&& gimple_code (stmt) == GIMPLE_ASSIGN)
{
enum tree_code code = gimple_assign_rhs_code (stmt);
if (code == BIT_AND_EXPR
&& (op2 = gimple_assign_rhs2 (stmt)) != NULL_TREE
&& TREE_CODE (op2) == INTEGER_CST
&& integer_pow2p (bottom)
&& wi::multiple_of_p (wi::to_widest (op2),
wi::to_widest (bottom), UNSIGNED))
return 1;
op1 = gimple_assign_rhs1 (stmt);
if (code == MINUS_EXPR
&& (op2 = gimple_assign_rhs2 (stmt)) != NULL_TREE
&& TREE_CODE (op2) == SSA_NAME
&& (stmt = SSA_NAME_DEF_STMT (op2)) != NULL
&& gimple_code (stmt) == GIMPLE_ASSIGN
&& (code = gimple_assign_rhs_code (stmt)) == TRUNC_MOD_EXPR
&& operand_equal_p (op1, gimple_assign_rhs1 (stmt), 0)
&& operand_equal_p (bottom, gimple_assign_rhs2 (stmt), 0))
return 1;
}
default:
if (POLY_INT_CST_P (top) && poly_int_tree_p (bottom))
return multiple_p (wi::to_poly_widest (top),
wi::to_poly_widest (bottom));
return 0;
}
}
#define tree_expr_nonnegative_warnv_p(X, Y) \
_Pragma ("GCC error \"Use RECURSE for recursive calls\"") 0
#define RECURSE(X) \
((tree_expr_nonnegative_warnv_p) (X, strict_overflow_p, depth + 1))
static bool
tree_simple_nonnegative_warnv_p (enum tree_code code, tree type)
{
if ((TYPE_PRECISION (type) != 1 || TYPE_UNSIGNED (type))
&& truth_value_p (code))
return true;
return false;
}
bool
tree_unary_nonnegative_warnv_p (enum tree_code code, tree type, tree op0,
bool *strict_overflow_p, int depth)
{
if (TYPE_UNSIGNED (type))
return true;
switch (code)
{
case ABS_EXPR:
if (!ANY_INTEGRAL_TYPE_P (type))
return true;
if (TYPE_OVERFLOW_UNDEFINED (type))
{
*strict_overflow_p = true;
return true;
}
break;
case NON_LVALUE_EXPR:
case FLOAT_EXPR:
case FIX_TRUNC_EXPR:
return RECURSE (op0);
CASE_CONVERT:
{
tree inner_type = TREE_TYPE (op0);
tree outer_type = type;
if (TREE_CODE (outer_type) == REAL_TYPE)
{
if (TREE_CODE (inner_type) == REAL_TYPE)
return RECURSE (op0);
if (INTEGRAL_TYPE_P (inner_type))
{
if (TYPE_UNSIGNED (inner_type))
return true;
return RECURSE (op0);
}
}
else if (INTEGRAL_TYPE_P (outer_type))
{
if (TREE_CODE (inner_type) == REAL_TYPE)
return RECURSE (op0);
if (INTEGRAL_TYPE_P (inner_type))
return TYPE_PRECISION (inner_type) < TYPE_PRECISION (outer_type)
&& TYPE_UNSIGNED (inner_type);
}
}
break;
default:
return tree_simple_nonnegative_warnv_p (code, type);
}
return false;
}
bool
tree_binary_nonnegative_warnv_p (enum tree_code code, tree type, tree op0,
tree op1, bool *strict_overflow_p,
int depth)
{
if (TYPE_UNSIGNED (type))
return true;
switch (code)
{
case POINTER_PLUS_EXPR:
case PLUS_EXPR:
if (FLOAT_TYPE_P (type))
return RECURSE (op0) && RECURSE (op1);
if (TREE_CODE (type) == INTEGER_TYPE
&& TREE_CODE (op0) == NOP_EXPR
&& TREE_CODE (op1) == NOP_EXPR)
{
tree inner1 = TREE_TYPE (TREE_OPERAND (op0, 0));
tree inner2 = TREE_TYPE (TREE_OPERAND (op1, 0));
if (TREE_CODE (inner1) == INTEGER_TYPE && TYPE_UNSIGNED (inner1)
&& TREE_CODE (inner2) == INTEGER_TYPE && TYPE_UNSIGNED (inner2))
{
unsigned int prec = MAX (TYPE_PRECISION (inner1),
TYPE_PRECISION (inner2)) + 1;
return prec < TYPE_PRECISION (type);
}
}
break;
case MULT_EXPR:
if (FLOAT_TYPE_P (type) || TYPE_OVERFLOW_UNDEFINED (type))
{
if (operand_equal_p (op0, op1, 0)
|| (RECURSE (op0) && RECURSE (op1)))
{
if (ANY_INTEGRAL_TYPE_P (type)
&& TYPE_OVERFLOW_UNDEFINED (type))
*strict_overflow_p = true;
return true;
}
}
if (TREE_CODE (type) == INTEGER_TYPE
&& (TREE_CODE (op0) == NOP_EXPR || TREE_CODE (op0) == INTEGER_CST)
&& (TREE_CODE (op1) == NOP_EXPR || TREE_CODE (op1) == INTEGER_CST))
{
tree inner0 = (TREE_CODE (op0) == NOP_EXPR)
? TREE_TYPE (TREE_OPERAND (op0, 0))
: TREE_TYPE (op0);
tree inner1 = (TREE_CODE (op1) == NOP_EXPR)
? TREE_TYPE (TREE_OPERAND (op1, 0))
: TREE_TYPE (op1);
bool unsigned0 = TYPE_UNSIGNED (inner0);
bool unsigned1 = TYPE_UNSIGNED (inner1);
if (TREE_CODE (op0) == INTEGER_CST)
unsigned0 = unsigned0 || tree_int_cst_sgn (op0) >= 0;
if (TREE_CODE (op1) == INTEGER_CST)
unsigned1 = unsigned1 || tree_int_cst_sgn (op1) >= 0;
if (TREE_CODE (inner0) == INTEGER_TYPE && unsigned0
&& TREE_CODE (inner1) == INTEGER_TYPE && unsigned1)
{
unsigned int precision0 = (TREE_CODE (op0) == INTEGER_CST)
? tree_int_cst_min_precision (op0, UNSIGNED)
: TYPE_PRECISION (inner0);
unsigned int precision1 = (TREE_CODE (op1) == INTEGER_CST)
? tree_int_cst_min_precision (op1, UNSIGNED)
: TYPE_PRECISION (inner1);
return precision0 + precision1 < TYPE_PRECISION (type);
}
}
return false;
case BIT_AND_EXPR:
case MAX_EXPR:
return RECURSE (op0) || RECURSE (op1);
case BIT_IOR_EXPR:
case BIT_XOR_EXPR:
case MIN_EXPR:
case RDIV_EXPR:
case TRUNC_DIV_EXPR:
case CEIL_DIV_EXPR:
case FLOOR_DIV_EXPR:
case ROUND_DIV_EXPR:
return RECURSE (op0) && RECURSE (op1);
case TRUNC_MOD_EXPR:
return RECURSE (op0);
case FLOOR_MOD_EXPR:
return RECURSE (op1);
case CEIL_MOD_EXPR:
case ROUND_MOD_EXPR:
default:
return tree_simple_nonnegative_warnv_p (code, type);
}
return false;
}
bool
tree_single_nonnegative_warnv_p (tree t, bool *strict_overflow_p, int depth)
{
if (TYPE_UNSIGNED (TREE_TYPE (t)))
return true;
switch (TREE_CODE (t))
{
case INTEGER_CST:
return tree_int_cst_sgn (t) >= 0;
case REAL_CST:
return ! REAL_VALUE_NEGATIVE (TREE_REAL_CST (t));
case FIXED_CST:
return ! FIXED_VALUE_NEGATIVE (TREE_FIXED_CST (t));
case COND_EXPR:
return RECURSE (TREE_OPERAND (t, 1)) && RECURSE (TREE_OPERAND (t, 2));
case SSA_NAME:
return (!name_registered_for_update_p (t)
&& depth < PARAM_VALUE (PARAM_MAX_SSA_NAME_QUERY_DEPTH)
&& gimple_stmt_nonnegative_warnv_p (SSA_NAME_DEF_STMT (t),
strict_overflow_p, depth));
default:
return tree_simple_nonnegative_warnv_p (TREE_CODE (t), TREE_TYPE (t));
}
}
bool
tree_call_nonnegative_warnv_p (tree type, combined_fn fn, tree arg0, tree arg1,
bool *strict_overflow_p, int depth)
{
switch (fn)
{
CASE_CFN_ACOS:
CASE_CFN_ACOSH:
CASE_CFN_CABS:
CASE_CFN_COSH:
CASE_CFN_ERFC:
CASE_CFN_EXP:
CASE_CFN_EXP10:
CASE_CFN_EXP2:
CASE_CFN_FABS:
CASE_CFN_FDIM:
CASE_CFN_HYPOT:
CASE_CFN_POW10:
CASE_CFN_FFS:
CASE_CFN_PARITY:
CASE_CFN_POPCOUNT:
CASE_CFN_CLZ:
CASE_CFN_CLRSB:
case CFN_BUILT_IN_BSWAP32:
case CFN_BUILT_IN_BSWAP64:
return true;
CASE_CFN_SQRT:
CASE_CFN_SQRT_FN:
if (!HONOR_SIGNED_ZEROS (element_mode (type)))
return true;
return RECURSE (arg0);
CASE_CFN_ASINH:
CASE_CFN_ATAN:
CASE_CFN_ATANH:
CASE_CFN_CBRT:
CASE_CFN_CEIL:
CASE_CFN_CEIL_FN:
CASE_CFN_ERF:
CASE_CFN_EXPM1:
CASE_CFN_FLOOR:
CASE_CFN_FLOOR_FN:
CASE_CFN_FMOD:
CASE_CFN_FREXP:
CASE_CFN_ICEIL:
CASE_CFN_IFLOOR:
CASE_CFN_IRINT:
CASE_CFN_IROUND:
CASE_CFN_LCEIL:
CASE_CFN_LDEXP:
CASE_CFN_LFLOOR:
CASE_CFN_LLCEIL:
CASE_CFN_LLFLOOR:
CASE_CFN_LLRINT:
CASE_CFN_LLROUND:
CASE_CFN_LRINT:
CASE_CFN_LROUND:
CASE_CFN_MODF:
CASE_CFN_NEARBYINT:
CASE_CFN_NEARBYINT_FN:
CASE_CFN_RINT:
CASE_CFN_RINT_FN:
CASE_CFN_ROUND:
CASE_CFN_ROUND_FN:
CASE_CFN_SCALB:
CASE_CFN_SCALBLN:
CASE_CFN_SCALBN:
CASE_CFN_SIGNBIT:
CASE_CFN_SIGNIFICAND:
CASE_CFN_SINH:
CASE_CFN_TANH:
CASE_CFN_TRUNC:
CASE_CFN_TRUNC_FN:
return RECURSE (arg0);
CASE_CFN_FMAX:
CASE_CFN_FMAX_FN:
return RECURSE (arg0) || RECURSE (arg1);
CASE_CFN_FMIN:
CASE_CFN_FMIN_FN:
return RECURSE (arg0) && RECURSE (arg1);
CASE_CFN_COPYSIGN:
CASE_CFN_COPYSIGN_FN:
return RECURSE (arg1);
CASE_CFN_POWI:
if (TREE_CODE (arg1) == INTEGER_CST
&& (TREE_INT_CST_LOW (arg1) & 1) == 0)
return true;
return RECURSE (arg0);
CASE_CFN_POW:
if (TREE_CODE (arg1) == REAL_CST)
{
REAL_VALUE_TYPE c;
HOST_WIDE_INT n;
c = TREE_REAL_CST (arg1);
n = real_to_integer (&c);
if ((n & 1) == 0)
{
REAL_VALUE_TYPE cint;
real_from_integer (&cint, VOIDmode, n, SIGNED);
if (real_identical (&c, &cint))
return true;
}
}
return RECURSE (arg0);
default:
break;
}
return tree_simple_nonnegative_warnv_p (CALL_EXPR, type);
}
static bool
tree_invalid_nonnegative_warnv_p (tree t, bool *strict_overflow_p, int depth)
{
enum tree_code code = TREE_CODE (t);
if (TYPE_UNSIGNED (TREE_TYPE (t)))
return true;
switch (code)
{
case TARGET_EXPR:
{
tree temp = TARGET_EXPR_SLOT (t);
t = TARGET_EXPR_INITIAL (t);
if (!VOID_TYPE_P (t))
return RECURSE (t);
while (1)
{
if (TREE_CODE (t) == BIND_EXPR)
t = expr_last (BIND_EXPR_BODY (t));
else if (TREE_CODE (t) == TRY_FINALLY_EXPR
|| TREE_CODE (t) == TRY_CATCH_EXPR)
t = expr_last (TREE_OPERAND (t, 0));
else if (TREE_CODE (t) == STATEMENT_LIST)
t = expr_last (t);
else
break;
}
if (TREE_CODE (t) == MODIFY_EXPR
&& TREE_OPERAND (t, 0) == temp)
return RECURSE (TREE_OPERAND (t, 1));
return false;
}
case CALL_EXPR:
{
tree arg0 = call_expr_nargs (t) > 0 ?  CALL_EXPR_ARG (t, 0) : NULL_TREE;
tree arg1 = call_expr_nargs (t) > 1 ?  CALL_EXPR_ARG (t, 1) : NULL_TREE;
return tree_call_nonnegative_warnv_p (TREE_TYPE (t),
get_call_combined_fn (t),
arg0,
arg1,
strict_overflow_p, depth);
}
case COMPOUND_EXPR:
case MODIFY_EXPR:
return RECURSE (TREE_OPERAND (t, 1));
case BIND_EXPR:
return RECURSE (expr_last (TREE_OPERAND (t, 1)));
case SAVE_EXPR:
return RECURSE (TREE_OPERAND (t, 0));
default:
return tree_simple_nonnegative_warnv_p (TREE_CODE (t), TREE_TYPE (t));
}
}
#undef RECURSE
#undef tree_expr_nonnegative_warnv_p
bool
tree_expr_nonnegative_warnv_p (tree t, bool *strict_overflow_p, int depth)
{
enum tree_code code;
if (t == error_mark_node)
return false;
code = TREE_CODE (t);
switch (TREE_CODE_CLASS (code))
{
case tcc_binary:
case tcc_comparison:
return tree_binary_nonnegative_warnv_p (TREE_CODE (t),
TREE_TYPE (t),
TREE_OPERAND (t, 0),
TREE_OPERAND (t, 1),
strict_overflow_p, depth);
case tcc_unary:
return tree_unary_nonnegative_warnv_p (TREE_CODE (t),
TREE_TYPE (t),
TREE_OPERAND (t, 0),
strict_overflow_p, depth);
case tcc_constant:
case tcc_declaration:
case tcc_reference:
return tree_single_nonnegative_warnv_p (t, strict_overflow_p, depth);
default:
break;
}
switch (code)
{
case TRUTH_AND_EXPR:
case TRUTH_OR_EXPR:
case TRUTH_XOR_EXPR:
return tree_binary_nonnegative_warnv_p (TREE_CODE (t),
TREE_TYPE (t),
TREE_OPERAND (t, 0),
TREE_OPERAND (t, 1),
strict_overflow_p, depth);
case TRUTH_NOT_EXPR:
return tree_unary_nonnegative_warnv_p (TREE_CODE (t),
TREE_TYPE (t),
TREE_OPERAND (t, 0),
strict_overflow_p, depth);
case COND_EXPR:
case CONSTRUCTOR:
case OBJ_TYPE_REF:
case ASSERT_EXPR:
case ADDR_EXPR:
case WITH_SIZE_EXPR:
case SSA_NAME:
return tree_single_nonnegative_warnv_p (t, strict_overflow_p, depth);
default:
return tree_invalid_nonnegative_warnv_p (t, strict_overflow_p, depth);
}
}
bool
tree_expr_nonnegative_p (tree t)
{
bool ret, strict_overflow_p;
strict_overflow_p = false;
ret = tree_expr_nonnegative_warnv_p (t, &strict_overflow_p);
if (strict_overflow_p)
fold_overflow_warning (("assuming signed overflow does not occur when "
"determining that expression is always "
"non-negative"),
WARN_STRICT_OVERFLOW_MISC);
return ret;
}
bool
tree_unary_nonzero_warnv_p (enum tree_code code, tree type, tree op0,
bool *strict_overflow_p)
{
switch (code)
{
case ABS_EXPR:
return tree_expr_nonzero_warnv_p (op0,
strict_overflow_p);
case NOP_EXPR:
{
tree inner_type = TREE_TYPE (op0);
tree outer_type = type;
return (TYPE_PRECISION (outer_type) >= TYPE_PRECISION (inner_type)
&& tree_expr_nonzero_warnv_p (op0,
strict_overflow_p));
}
break;
case NON_LVALUE_EXPR:
return tree_expr_nonzero_warnv_p (op0,
strict_overflow_p);
default:
break;
}
return false;
}
bool
tree_binary_nonzero_warnv_p (enum tree_code code,
tree type,
tree op0,
tree op1, bool *strict_overflow_p)
{
bool sub_strict_overflow_p;
switch (code)
{
case POINTER_PLUS_EXPR:
case PLUS_EXPR:
if (ANY_INTEGRAL_TYPE_P (type) && TYPE_OVERFLOW_UNDEFINED (type))
{
sub_strict_overflow_p = false;
if (!tree_expr_nonnegative_warnv_p (op0,
&sub_strict_overflow_p)
|| !tree_expr_nonnegative_warnv_p (op1,
&sub_strict_overflow_p))
return false;
return (tree_expr_nonzero_warnv_p (op0,
strict_overflow_p)
|| tree_expr_nonzero_warnv_p (op1,
strict_overflow_p));
}
break;
case MULT_EXPR:
if (TYPE_OVERFLOW_UNDEFINED (type))
{
if (tree_expr_nonzero_warnv_p (op0,
strict_overflow_p)
&& tree_expr_nonzero_warnv_p (op1,
strict_overflow_p))
{
*strict_overflow_p = true;
return true;
}
}
break;
case MIN_EXPR:
sub_strict_overflow_p = false;
if (tree_expr_nonzero_warnv_p (op0,
&sub_strict_overflow_p)
&& tree_expr_nonzero_warnv_p (op1,
&sub_strict_overflow_p))
{
if (sub_strict_overflow_p)
*strict_overflow_p = true;
}
break;
case MAX_EXPR:
sub_strict_overflow_p = false;
if (tree_expr_nonzero_warnv_p (op0,
&sub_strict_overflow_p))
{
if (sub_strict_overflow_p)
*strict_overflow_p = true;
if (tree_expr_nonzero_warnv_p (op1,
strict_overflow_p))
return true;
return tree_expr_nonnegative_warnv_p (op0,
strict_overflow_p);
}
else if (tree_expr_nonzero_warnv_p (op1,
&sub_strict_overflow_p)
&& tree_expr_nonnegative_warnv_p (op1,
&sub_strict_overflow_p))
{
if (sub_strict_overflow_p)
*strict_overflow_p = true;
return true;
}
break;
case BIT_IOR_EXPR:
return (tree_expr_nonzero_warnv_p (op1,
strict_overflow_p)
|| tree_expr_nonzero_warnv_p (op0,
strict_overflow_p));
default:
break;
}
return false;
}
bool
tree_single_nonzero_warnv_p (tree t, bool *strict_overflow_p)
{
bool sub_strict_overflow_p;
switch (TREE_CODE (t))
{
case INTEGER_CST:
return !integer_zerop (t);
case ADDR_EXPR:
{
tree base = TREE_OPERAND (t, 0);
if (!DECL_P (base))
base = get_base_address (base);
if (base && TREE_CODE (base) == TARGET_EXPR)
base = TARGET_EXPR_SLOT (base);
if (!base)
return false;
int nonzero_addr = maybe_nonzero_address (base);
if (nonzero_addr >= 0)
return nonzero_addr;
if (CONSTANT_CLASS_P (base))
return true;
return false;
}
case COND_EXPR:
sub_strict_overflow_p = false;
if (tree_expr_nonzero_warnv_p (TREE_OPERAND (t, 1),
&sub_strict_overflow_p)
&& tree_expr_nonzero_warnv_p (TREE_OPERAND (t, 2),
&sub_strict_overflow_p))
{
if (sub_strict_overflow_p)
*strict_overflow_p = true;
return true;
}
break;
case SSA_NAME:
if (!INTEGRAL_TYPE_P (TREE_TYPE (t)))
break;
return expr_not_equal_to (t, wi::zero (TYPE_PRECISION (TREE_TYPE (t))));
default:
break;
}
return false;
}
#define integer_valued_real_p(X) \
_Pragma ("GCC error \"Use RECURSE for recursive calls\"") 0
#define RECURSE(X) \
((integer_valued_real_p) (X, depth + 1))
bool
integer_valued_real_unary_p (tree_code code, tree op0, int depth)
{
switch (code)
{
case FLOAT_EXPR:
return true;
case ABS_EXPR:
return RECURSE (op0);
CASE_CONVERT:
{
tree type = TREE_TYPE (op0);
if (TREE_CODE (type) == INTEGER_TYPE)
return true;
if (TREE_CODE (type) == REAL_TYPE)
return RECURSE (op0);
break;
}
default:
break;
}
return false;
}
bool
integer_valued_real_binary_p (tree_code code, tree op0, tree op1, int depth)
{
switch (code)
{
case PLUS_EXPR:
case MINUS_EXPR:
case MULT_EXPR:
case MIN_EXPR:
case MAX_EXPR:
return RECURSE (op0) && RECURSE (op1);
default:
break;
}
return false;
}
bool
integer_valued_real_call_p (combined_fn fn, tree arg0, tree arg1, int depth)
{
switch (fn)
{
CASE_CFN_CEIL:
CASE_CFN_CEIL_FN:
CASE_CFN_FLOOR:
CASE_CFN_FLOOR_FN:
CASE_CFN_NEARBYINT:
CASE_CFN_NEARBYINT_FN:
CASE_CFN_RINT:
CASE_CFN_RINT_FN:
CASE_CFN_ROUND:
CASE_CFN_ROUND_FN:
CASE_CFN_TRUNC:
CASE_CFN_TRUNC_FN:
return true;
CASE_CFN_FMIN:
CASE_CFN_FMIN_FN:
CASE_CFN_FMAX:
CASE_CFN_FMAX_FN:
return RECURSE (arg0) && RECURSE (arg1);
default:
break;
}
return false;
}
bool
integer_valued_real_single_p (tree t, int depth)
{
switch (TREE_CODE (t))
{
case REAL_CST:
return real_isinteger (TREE_REAL_CST_PTR (t), TYPE_MODE (TREE_TYPE (t)));
case COND_EXPR:
return RECURSE (TREE_OPERAND (t, 1)) && RECURSE (TREE_OPERAND (t, 2));
case SSA_NAME:
return (!name_registered_for_update_p (t)
&& depth < PARAM_VALUE (PARAM_MAX_SSA_NAME_QUERY_DEPTH)
&& gimple_stmt_integer_valued_real_p (SSA_NAME_DEF_STMT (t),
depth));
default:
break;
}
return false;
}
static bool
integer_valued_real_invalid_p (tree t, int depth)
{
switch (TREE_CODE (t))
{
case COMPOUND_EXPR:
case MODIFY_EXPR:
case BIND_EXPR:
return RECURSE (TREE_OPERAND (t, 1));
case SAVE_EXPR:
return RECURSE (TREE_OPERAND (t, 0));
default:
break;
}
return false;
}
#undef RECURSE
#undef integer_valued_real_p
bool
integer_valued_real_p (tree t, int depth)
{
if (t == error_mark_node)
return false;
tree_code code = TREE_CODE (t);
switch (TREE_CODE_CLASS (code))
{
case tcc_binary:
case tcc_comparison:
return integer_valued_real_binary_p (code, TREE_OPERAND (t, 0),
TREE_OPERAND (t, 1), depth);
case tcc_unary:
return integer_valued_real_unary_p (code, TREE_OPERAND (t, 0), depth);
case tcc_constant:
case tcc_declaration:
case tcc_reference:
return integer_valued_real_single_p (t, depth);
default:
break;
}
switch (code)
{
case COND_EXPR:
case SSA_NAME:
return integer_valued_real_single_p (t, depth);
case CALL_EXPR:
{
tree arg0 = (call_expr_nargs (t) > 0
? CALL_EXPR_ARG (t, 0)
: NULL_TREE);
tree arg1 = (call_expr_nargs (t) > 1
? CALL_EXPR_ARG (t, 1)
: NULL_TREE);
return integer_valued_real_call_p (get_call_combined_fn (t),
arg0, arg1, depth);
}
default:
return integer_valued_real_invalid_p (t, depth);
}
}
tree
fold_binary_to_constant (enum tree_code code, tree type, tree op0, tree op1)
{
tree tem = fold_binary (code, type, op0, op1);
return (tem && TREE_CONSTANT (tem)) ? tem : NULL_TREE;
}
tree
fold_unary_to_constant (enum tree_code code, tree type, tree op0)
{
tree tem = fold_unary (code, type, op0);
return (tem && TREE_CONSTANT (tem)) ? tem : NULL_TREE;
}
tree
fold_read_from_constant_string (tree exp)
{
if ((TREE_CODE (exp) == INDIRECT_REF
|| TREE_CODE (exp) == ARRAY_REF)
&& TREE_CODE (TREE_TYPE (exp)) == INTEGER_TYPE)
{
tree exp1 = TREE_OPERAND (exp, 0);
tree index;
tree string;
location_t loc = EXPR_LOCATION (exp);
if (TREE_CODE (exp) == INDIRECT_REF)
string = string_constant (exp1, &index);
else
{
tree low_bound = array_ref_low_bound (exp);
index = fold_convert_loc (loc, sizetype, TREE_OPERAND (exp, 1));
if (! integer_zerop (low_bound))
index = size_diffop_loc (loc, index,
fold_convert_loc (loc, sizetype, low_bound));
string = exp1;
}
scalar_int_mode char_mode;
if (string
&& TYPE_MODE (TREE_TYPE (exp)) == TYPE_MODE (TREE_TYPE (TREE_TYPE (string)))
&& TREE_CODE (string) == STRING_CST
&& TREE_CODE (index) == INTEGER_CST
&& compare_tree_int (index, TREE_STRING_LENGTH (string)) < 0
&& is_int_mode (TYPE_MODE (TREE_TYPE (TREE_TYPE (string))),
&char_mode)
&& GET_MODE_SIZE (char_mode) == 1)
return build_int_cst_type (TREE_TYPE (exp),
(TREE_STRING_POINTER (string)
[TREE_INT_CST_LOW (index)]));
}
return NULL;
}
static tree
fold_negate_const (tree arg0, tree type)
{
tree t = NULL_TREE;
switch (TREE_CODE (arg0))
{
case REAL_CST:
t = build_real (type, real_value_negate (&TREE_REAL_CST (arg0)));
break;
case FIXED_CST:
{
FIXED_VALUE_TYPE f;
bool overflow_p = fixed_arithmetic (&f, NEGATE_EXPR,
&(TREE_FIXED_CST (arg0)), NULL,
TYPE_SATURATING (type));
t = build_fixed (type, f);
if (overflow_p | TREE_OVERFLOW (arg0))
TREE_OVERFLOW (t) = 1;
break;
}
default:
if (poly_int_tree_p (arg0))
{
bool overflow;
poly_wide_int res = wi::neg (wi::to_poly_wide (arg0), &overflow);
t = force_fit_type (type, res, 1,
(overflow && ! TYPE_UNSIGNED (type))
|| TREE_OVERFLOW (arg0));
break;
}
gcc_unreachable ();
}
return t;
}
tree
fold_abs_const (tree arg0, tree type)
{
tree t = NULL_TREE;
switch (TREE_CODE (arg0))
{
case INTEGER_CST:
{
if (!wi::neg_p (wi::to_wide (arg0), TYPE_SIGN (type)))
t = arg0;
else
{
bool overflow;
wide_int val = wi::neg (wi::to_wide (arg0), &overflow);
t = force_fit_type (type, val, -1,
overflow | TREE_OVERFLOW (arg0));
}
}
break;
case REAL_CST:
if (REAL_VALUE_NEGATIVE (TREE_REAL_CST (arg0)))
t = build_real (type, real_value_negate (&TREE_REAL_CST (arg0)));
else
t =  arg0;
break;
default:
gcc_unreachable ();
}
return t;
}
static tree
fold_not_const (const_tree arg0, tree type)
{
gcc_assert (TREE_CODE (arg0) == INTEGER_CST);
return force_fit_type (type, ~wi::to_wide (arg0), 0, TREE_OVERFLOW (arg0));
}
static tree
fold_relational_const (enum tree_code code, tree type, tree op0, tree op1)
{
int result, invert;
if (TREE_CODE (op0) == REAL_CST && TREE_CODE (op1) == REAL_CST)
{
const REAL_VALUE_TYPE *c0 = TREE_REAL_CST_PTR (op0);
const REAL_VALUE_TYPE *c1 = TREE_REAL_CST_PTR (op1);
if (real_isnan (c0) || real_isnan (c1))
{
switch (code)
{
case EQ_EXPR:
case ORDERED_EXPR:
result = 0;
break;
case NE_EXPR:
case UNORDERED_EXPR:
case UNLT_EXPR:
case UNLE_EXPR:
case UNGT_EXPR:
case UNGE_EXPR:
case UNEQ_EXPR:
result = 1;
break;
case LT_EXPR:
case LE_EXPR:
case GT_EXPR:
case GE_EXPR:
case LTGT_EXPR:
if (flag_trapping_math)
return NULL_TREE;
result = 0;
break;
default:
gcc_unreachable ();
}
return constant_boolean_node (result, type);
}
return constant_boolean_node (real_compare (code, c0, c1), type);
}
if (TREE_CODE (op0) == FIXED_CST && TREE_CODE (op1) == FIXED_CST)
{
const FIXED_VALUE_TYPE *c0 = TREE_FIXED_CST_PTR (op0);
const FIXED_VALUE_TYPE *c1 = TREE_FIXED_CST_PTR (op1);
return constant_boolean_node (fixed_compare (code, c0, c1), type);
}
if (TREE_CODE (op0) == COMPLEX_CST && TREE_CODE (op1) == COMPLEX_CST)
{
tree rcond = fold_relational_const (code, type,
TREE_REALPART (op0),
TREE_REALPART (op1));
tree icond = fold_relational_const (code, type,
TREE_IMAGPART (op0),
TREE_IMAGPART (op1));
if (code == EQ_EXPR)
return fold_build2 (TRUTH_ANDIF_EXPR, type, rcond, icond);
else if (code == NE_EXPR)
return fold_build2 (TRUTH_ORIF_EXPR, type, rcond, icond);
else
return NULL_TREE;
}
if (TREE_CODE (op0) == VECTOR_CST && TREE_CODE (op1) == VECTOR_CST)
{
if (!VECTOR_TYPE_P (type))
{
gcc_assert ((code == EQ_EXPR || code == NE_EXPR)
&& known_eq (VECTOR_CST_NELTS (op0),
VECTOR_CST_NELTS (op1)));
unsigned HOST_WIDE_INT nunits;
if (!VECTOR_CST_NELTS (op0).is_constant (&nunits))
return NULL_TREE;
for (unsigned i = 0; i < nunits; i++)
{
tree elem0 = VECTOR_CST_ELT (op0, i);
tree elem1 = VECTOR_CST_ELT (op1, i);
tree tmp = fold_relational_const (code, type, elem0, elem1);
if (tmp == NULL_TREE)
return NULL_TREE;
if (integer_zerop (tmp))
return constant_boolean_node (false, type);
}
return constant_boolean_node (true, type);
}
tree_vector_builder elts;
if (!elts.new_binary_operation (type, op0, op1, false))
return NULL_TREE;
unsigned int count = elts.encoded_nelts ();
for (unsigned i = 0; i < count; i++)
{
tree elem_type = TREE_TYPE (type);
tree elem0 = VECTOR_CST_ELT (op0, i);
tree elem1 = VECTOR_CST_ELT (op1, i);
tree tem = fold_relational_const (code, elem_type,
elem0, elem1);
if (tem == NULL_TREE)
return NULL_TREE;
elts.quick_push (build_int_cst (elem_type,
integer_zerop (tem) ? 0 : -1));
}
return elts.build ();
}
if (code == LE_EXPR || code == GT_EXPR)
{
std::swap (op0, op1);
code = swap_tree_comparison (code);
}
invert = 0;
if (code == NE_EXPR || code == GE_EXPR)
{
invert = 1;
code = invert_tree_comparison (code, false);
}
if (TREE_CODE (op0) == INTEGER_CST && TREE_CODE (op1) == INTEGER_CST)
{
if (code == EQ_EXPR)
result = tree_int_cst_equal (op0, op1);
else
result = tree_int_cst_lt (op0, op1);
}
else
return NULL_TREE;
if (invert)
result ^= 1;
return constant_boolean_node (result, type);
}
tree
fold_build_cleanup_point_expr (tree type, tree expr)
{
if (!TREE_SIDE_EFFECTS (expr))
return expr;
if (TREE_CODE (expr) == RETURN_EXPR)
{
tree op = TREE_OPERAND (expr, 0);
if (!op || !TREE_SIDE_EFFECTS (op))
return expr;
op = TREE_OPERAND (op, 1);
if (!TREE_SIDE_EFFECTS (op))
return expr;
}
return build1_loc (EXPR_LOCATION (expr), CLEANUP_POINT_EXPR, type, expr);
}
tree
fold_indirect_ref_1 (location_t loc, tree type, tree op0)
{
tree sub = op0;
tree subtype;
poly_uint64 const_op01;
STRIP_NOPS (sub);
subtype = TREE_TYPE (sub);
if (!POINTER_TYPE_P (subtype)
|| TYPE_REF_CAN_ALIAS_ALL (TREE_TYPE (op0)))
return NULL_TREE;
if (TREE_CODE (sub) == ADDR_EXPR)
{
tree op = TREE_OPERAND (sub, 0);
tree optype = TREE_TYPE (op);
if (TREE_CODE (op) == CONST_DECL)
return DECL_INITIAL (op);
if (type == optype)
{
tree fop = fold_read_from_constant_string (op);
if (fop)
return fop;
else
return op;
}
else if (TREE_CODE (optype) == ARRAY_TYPE
&& type == TREE_TYPE (optype)
&& (!in_gimple_form
|| TREE_CODE (TYPE_SIZE (type)) == INTEGER_CST))
{
tree type_domain = TYPE_DOMAIN (optype);
tree min_val = size_zero_node;
if (type_domain && TYPE_MIN_VALUE (type_domain))
min_val = TYPE_MIN_VALUE (type_domain);
if (in_gimple_form
&& TREE_CODE (min_val) != INTEGER_CST)
return NULL_TREE;
return build4_loc (loc, ARRAY_REF, type, op, min_val,
NULL_TREE, NULL_TREE);
}
else if (TREE_CODE (optype) == COMPLEX_TYPE
&& type == TREE_TYPE (optype))
return fold_build1_loc (loc, REALPART_EXPR, type, op);
else if (VECTOR_TYPE_P (optype)
&& type == TREE_TYPE (optype))
{
tree part_width = TYPE_SIZE (type);
tree index = bitsize_int (0);
return fold_build3_loc (loc, BIT_FIELD_REF, type, op, part_width,
index);
}
}
if (TREE_CODE (sub) == POINTER_PLUS_EXPR
&& poly_int_tree_p (TREE_OPERAND (sub, 1), &const_op01))
{
tree op00 = TREE_OPERAND (sub, 0);
tree op01 = TREE_OPERAND (sub, 1);
STRIP_NOPS (op00);
if (TREE_CODE (op00) == ADDR_EXPR)
{
tree op00type;
op00 = TREE_OPERAND (op00, 0);
op00type = TREE_TYPE (op00);
if (VECTOR_TYPE_P (op00type)
&& type == TREE_TYPE (op00type)
&& tree_fits_poly_int64_p (op01))
{
tree part_width = TYPE_SIZE (type);
poly_uint64 max_offset
= (tree_to_uhwi (part_width) / BITS_PER_UNIT
* TYPE_VECTOR_SUBPARTS (op00type));
if (known_lt (const_op01, max_offset))
{
tree index = bitsize_int (const_op01 * BITS_PER_UNIT);
return fold_build3_loc (loc,
BIT_FIELD_REF, type, op00,
part_width, index);
}
}
else if (TREE_CODE (op00type) == COMPLEX_TYPE
&& type == TREE_TYPE (op00type))
{
if (known_eq (wi::to_poly_offset (TYPE_SIZE_UNIT (type)),
const_op01))
return fold_build1_loc (loc, IMAGPART_EXPR, type, op00);
}
else if (TREE_CODE (op00type) == ARRAY_TYPE
&& type == TREE_TYPE (op00type))
{
tree type_domain = TYPE_DOMAIN (op00type);
tree min_val = size_zero_node;
if (type_domain && TYPE_MIN_VALUE (type_domain))
min_val = TYPE_MIN_VALUE (type_domain);
offset_int off = wi::to_offset (op01);
offset_int el_sz = wi::to_offset (TYPE_SIZE_UNIT (type));
offset_int remainder;
off = wi::divmod_trunc (off, el_sz, SIGNED, &remainder);
if (remainder == 0 && TREE_CODE (min_val) == INTEGER_CST)
{
off = off + wi::to_offset (min_val);
op01 = wide_int_to_tree (sizetype, off);
return build4_loc (loc, ARRAY_REF, type, op00, op01,
NULL_TREE, NULL_TREE);
}
}
}
}
if (TREE_CODE (TREE_TYPE (subtype)) == ARRAY_TYPE
&& type == TREE_TYPE (TREE_TYPE (subtype))
&& (!in_gimple_form
|| TREE_CODE (TYPE_SIZE (type)) == INTEGER_CST))
{
tree type_domain;
tree min_val = size_zero_node;
sub = build_fold_indirect_ref_loc (loc, sub);
type_domain = TYPE_DOMAIN (TREE_TYPE (sub));
if (type_domain && TYPE_MIN_VALUE (type_domain))
min_val = TYPE_MIN_VALUE (type_domain);
if (in_gimple_form
&& TREE_CODE (min_val) != INTEGER_CST)
return NULL_TREE;
return build4_loc (loc, ARRAY_REF, type, sub, min_val, NULL_TREE,
NULL_TREE);
}
return NULL_TREE;
}
tree
build_fold_indirect_ref_loc (location_t loc, tree t)
{
tree type = TREE_TYPE (TREE_TYPE (t));
tree sub = fold_indirect_ref_1 (loc, type, t);
if (sub)
return sub;
return build1_loc (loc, INDIRECT_REF, type, t);
}
tree
fold_indirect_ref_loc (location_t loc, tree t)
{
tree sub = fold_indirect_ref_1 (loc, TREE_TYPE (t), TREE_OPERAND (t, 0));
if (sub)
return sub;
else
return t;
}
tree
fold_ignored_result (tree t)
{
if (!TREE_SIDE_EFFECTS (t))
return integer_zero_node;
for (;;)
switch (TREE_CODE_CLASS (TREE_CODE (t)))
{
case tcc_unary:
t = TREE_OPERAND (t, 0);
break;
case tcc_binary:
case tcc_comparison:
if (!TREE_SIDE_EFFECTS (TREE_OPERAND (t, 1)))
t = TREE_OPERAND (t, 0);
else if (!TREE_SIDE_EFFECTS (TREE_OPERAND (t, 0)))
t = TREE_OPERAND (t, 1);
else
return t;
break;
case tcc_expression:
switch (TREE_CODE (t))
{
case COMPOUND_EXPR:
if (TREE_SIDE_EFFECTS (TREE_OPERAND (t, 1)))
return t;
t = TREE_OPERAND (t, 0);
break;
case COND_EXPR:
if (TREE_SIDE_EFFECTS (TREE_OPERAND (t, 1))
|| TREE_SIDE_EFFECTS (TREE_OPERAND (t, 2)))
return t;
t = TREE_OPERAND (t, 0);
break;
default:
return t;
}
break;
default:
return t;
}
}
tree
round_up_loc (location_t loc, tree value, unsigned int divisor)
{
tree div = NULL_TREE;
if (divisor == 1)
return value;
if (TREE_CODE (value) != INTEGER_CST)
{
div = build_int_cst (TREE_TYPE (value), divisor);
if (multiple_of_p (TREE_TYPE (value), value, div))
return value;
}
if (pow2_or_zerop (divisor))
{
if (TREE_CODE (value) == INTEGER_CST)
{
wide_int val = wi::to_wide (value);
bool overflow_p;
if ((val & (divisor - 1)) == 0)
return value;
overflow_p = TREE_OVERFLOW (value);
val += divisor - 1;
val &= (int) -divisor;
if (val == 0)
overflow_p = true;
return force_fit_type (TREE_TYPE (value), val, -1, overflow_p);
}
else
{
tree t;
t = build_int_cst (TREE_TYPE (value), divisor - 1);
value = size_binop_loc (loc, PLUS_EXPR, value, t);
t = build_int_cst (TREE_TYPE (value), - (int) divisor);
value = size_binop_loc (loc, BIT_AND_EXPR, value, t);
}
}
else
{
if (!div)
div = build_int_cst (TREE_TYPE (value), divisor);
value = size_binop_loc (loc, CEIL_DIV_EXPR, value, div);
value = size_binop_loc (loc, MULT_EXPR, value, div);
}
return value;
}
tree
round_down_loc (location_t loc, tree value, int divisor)
{
tree div = NULL_TREE;
gcc_assert (divisor > 0);
if (divisor == 1)
return value;
if (TREE_CODE (value) != INTEGER_CST)
{
div = build_int_cst (TREE_TYPE (value), divisor);
if (multiple_of_p (TREE_TYPE (value), value, div))
return value;
}
if (pow2_or_zerop (divisor))
{
tree t;
t = build_int_cst (TREE_TYPE (value), -divisor);
value = size_binop_loc (loc, BIT_AND_EXPR, value, t);
}
else
{
if (!div)
div = build_int_cst (TREE_TYPE (value), divisor);
value = size_binop_loc (loc, FLOOR_DIV_EXPR, value, div);
value = size_binop_loc (loc, MULT_EXPR, value, div);
}
return value;
}
static tree
split_address_to_core_and_offset (tree exp,
poly_int64_pod *pbitpos, tree *poffset)
{
tree core;
machine_mode mode;
int unsignedp, reversep, volatilep;
poly_int64 bitsize;
location_t loc = EXPR_LOCATION (exp);
if (TREE_CODE (exp) == ADDR_EXPR)
{
core = get_inner_reference (TREE_OPERAND (exp, 0), &bitsize, pbitpos,
poffset, &mode, &unsignedp, &reversep,
&volatilep);
core = build_fold_addr_expr_loc (loc, core);
}
else if (TREE_CODE (exp) == POINTER_PLUS_EXPR)
{
core = TREE_OPERAND (exp, 0);
STRIP_NOPS (core);
*pbitpos = 0;
*poffset = TREE_OPERAND (exp, 1);
if (poly_int_tree_p (*poffset))
{
poly_offset_int tem
= wi::sext (wi::to_poly_offset (*poffset),
TYPE_PRECISION (TREE_TYPE (*poffset)));
tem <<= LOG2_BITS_PER_UNIT;
if (tem.to_shwi (pbitpos))
*poffset = NULL_TREE;
}
}
else
{
core = exp;
*pbitpos = 0;
*poffset = NULL_TREE;
}
return core;
}
bool
ptr_difference_const (tree e1, tree e2, poly_int64_pod *diff)
{
tree core1, core2;
poly_int64 bitpos1, bitpos2;
tree toffset1, toffset2, tdiff, type;
core1 = split_address_to_core_and_offset (e1, &bitpos1, &toffset1);
core2 = split_address_to_core_and_offset (e2, &bitpos2, &toffset2);
poly_int64 bytepos1, bytepos2;
if (!multiple_p (bitpos1, BITS_PER_UNIT, &bytepos1)
|| !multiple_p (bitpos2, BITS_PER_UNIT, &bytepos2)
|| !operand_equal_p (core1, core2, 0))
return false;
if (toffset1 && toffset2)
{
type = TREE_TYPE (toffset1);
if (type != TREE_TYPE (toffset2))
toffset2 = fold_convert (type, toffset2);
tdiff = fold_build2 (MINUS_EXPR, type, toffset1, toffset2);
if (!cst_and_fits_in_hwi (tdiff))
return false;
*diff = int_cst_value (tdiff);
}
else if (toffset1 || toffset2)
{
return false;
}
else
*diff = 0;
*diff += bytepos1 - bytepos2;
return true;
}
tree
convert_to_ptrofftype_loc (location_t loc, tree off)
{
return fold_convert_loc (loc, sizetype, off);
}
tree
fold_build_pointer_plus_loc (location_t loc, tree ptr, tree off)
{
return fold_build2_loc (loc, POINTER_PLUS_EXPR, TREE_TYPE (ptr),
ptr, convert_to_ptrofftype_loc (loc, off));
}
tree
fold_build_pointer_plus_hwi_loc (location_t loc, tree ptr, HOST_WIDE_INT off)
{
return fold_build2_loc (loc, POINTER_PLUS_EXPR, TREE_TYPE (ptr),
ptr, size_int (off));
}
const char *
c_getstr (tree src, unsigned HOST_WIDE_INT *strlen)
{
tree offset_node;
if (strlen)
*strlen = 0;
src = string_constant (src, &offset_node);
if (src == 0)
return NULL;
unsigned HOST_WIDE_INT offset = 0;
if (offset_node != NULL_TREE)
{
if (!tree_fits_uhwi_p (offset_node))
return NULL;
else
offset = tree_to_uhwi (offset_node);
}
unsigned HOST_WIDE_INT string_length = TREE_STRING_LENGTH (src);
const char *string = TREE_STRING_POINTER (src);
if (string_length == 0
|| string[string_length - 1] != '\0'
|| offset >= string_length)
return NULL;
if (strlen)
*strlen = string_length - offset;
return string + offset;
}
#if CHECKING_P
namespace selftest {
static void
assert_binop_folds_to_const (tree lhs, enum tree_code code, tree rhs,
tree constant)
{
ASSERT_EQ (constant, fold_build2 (code, TREE_TYPE (lhs), lhs, rhs));
}
static void
assert_binop_folds_to_nonlvalue (tree lhs, enum tree_code code, tree rhs,
tree wrapped_expr)
{
tree result = fold_build2 (code, TREE_TYPE (lhs), lhs, rhs);
ASSERT_NE (wrapped_expr, result);
ASSERT_EQ (NON_LVALUE_EXPR, TREE_CODE (result));
ASSERT_EQ (wrapped_expr, TREE_OPERAND (result, 0));
}
static void
test_arithmetic_folding ()
{
tree type = integer_type_node;
tree x = create_tmp_var_raw (type, "x");
tree zero = build_zero_cst (type);
tree one = build_int_cst (type, 1);
assert_binop_folds_to_const (zero, PLUS_EXPR, one,
one);
assert_binop_folds_to_const (one, PLUS_EXPR, zero,
one);
assert_binop_folds_to_nonlvalue (x, PLUS_EXPR, zero,
x);
assert_binop_folds_to_const (x, MINUS_EXPR, x,
zero);
assert_binop_folds_to_nonlvalue (x, MINUS_EXPR, zero,
x);
assert_binop_folds_to_const (x, MULT_EXPR, zero,
zero);
assert_binop_folds_to_nonlvalue (x, MULT_EXPR, one,
x);
}
static void
test_vector_folding ()
{
tree inner_type = integer_type_node;
tree type = build_vector_type (inner_type, 4);
tree zero = build_zero_cst (type);
tree one = build_one_cst (type);
tree res_type = boolean_type_node;
ASSERT_FALSE (integer_nonzerop (fold_build2 (EQ_EXPR, res_type, zero, one)));
ASSERT_TRUE (integer_nonzerop (fold_build2 (EQ_EXPR, res_type, zero, zero)));
ASSERT_TRUE (integer_nonzerop (fold_build2 (NE_EXPR, res_type, zero, one)));
ASSERT_FALSE (integer_nonzerop (fold_build2 (NE_EXPR, res_type, one, one)));
}
static void
test_vec_duplicate_folding ()
{
scalar_int_mode int_mode = SCALAR_INT_TYPE_MODE (ssizetype);
machine_mode vec_mode = targetm.vectorize.preferred_simd_mode (int_mode);
poly_uint64 nunits = GET_MODE_NUNITS (vec_mode);
tree type = build_vector_type (ssizetype, nunits);
tree dup5_expr = fold_unary (VEC_DUPLICATE_EXPR, type, ssize_int (5));
tree dup5_cst = build_vector_from_val (type, ssize_int (5));
ASSERT_TRUE (operand_equal_p (dup5_expr, dup5_cst, 0));
}
void
fold_const_c_tests ()
{
test_arithmetic_folding ();
test_vector_folding ();
test_vec_duplicate_folding ();
}
} 
#endif 
