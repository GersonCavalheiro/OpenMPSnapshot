#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "target.h"
#include "rtl.h"
#include "tree.h"
#include "predict.h"
#include "memmodel.h"
#include "tm_p.h"
#include "optabs.h"
#include "emit-rtl.h"
#include "fold-const.h"
#include "stor-layout.h"
#include "dojump.h"
#include "explow.h"
#include "expr.h"
#include "langhooks.h"
static bool prefer_and_bit_test (scalar_int_mode, int);
static void do_jump_by_parts_greater (scalar_int_mode, tree, tree, int,
rtx_code_label *, rtx_code_label *,
profile_probability);
static void do_jump_by_parts_equality (scalar_int_mode, tree, tree,
rtx_code_label *, rtx_code_label *,
profile_probability);
static void do_compare_and_jump	(tree, tree, enum rtx_code, enum rtx_code,
rtx_code_label *, rtx_code_label *,
profile_probability);
void
init_pending_stack_adjust (void)
{
pending_stack_adjust = 0;
}
void
discard_pending_stack_adjust (void)
{
stack_pointer_delta -= pending_stack_adjust;
pending_stack_adjust = 0;
}
void
clear_pending_stack_adjust (void)
{
if (optimize > 0
&& (! flag_omit_frame_pointer || cfun->calls_alloca)
&& EXIT_IGNORE_STACK)
discard_pending_stack_adjust ();
}
void
do_pending_stack_adjust (void)
{
if (inhibit_defer_pop == 0)
{
if (maybe_ne (pending_stack_adjust, 0))
adjust_stack (gen_int_mode (pending_stack_adjust, Pmode));
pending_stack_adjust = 0;
}
}
void
save_pending_stack_adjust (saved_pending_stack_adjust *save)
{
save->x_pending_stack_adjust = pending_stack_adjust;
save->x_stack_pointer_delta = stack_pointer_delta;
}
void
restore_pending_stack_adjust (saved_pending_stack_adjust *save)
{
if (inhibit_defer_pop == 0)
{
pending_stack_adjust = save->x_pending_stack_adjust;
stack_pointer_delta = save->x_stack_pointer_delta;
}
}

void
jumpifnot (tree exp, rtx_code_label *label, profile_probability prob)
{
do_jump (exp, label, NULL, prob.invert ());
}
void
jumpifnot_1 (enum tree_code code, tree op0, tree op1, rtx_code_label *label,
profile_probability prob)
{
do_jump_1 (code, op0, op1, label, NULL, prob.invert ());
}
void
jumpif (tree exp, rtx_code_label *label, profile_probability prob)
{
do_jump (exp, NULL, label, prob);
}
void
jumpif_1 (enum tree_code code, tree op0, tree op1,
rtx_code_label *label, profile_probability prob)
{
do_jump_1 (code, op0, op1, NULL, label, prob);
}
static GTY(()) rtx and_reg;
static GTY(()) rtx and_test;
static GTY(()) rtx shift_test;
static bool
prefer_and_bit_test (scalar_int_mode mode, int bitnum)
{
bool speed_p;
wide_int mask = wi::set_bit_in_zero (bitnum, GET_MODE_PRECISION (mode));
if (and_test == 0)
{
and_reg = gen_rtx_REG (mode, LAST_VIRTUAL_REGISTER + 1);
and_test = gen_rtx_AND (mode, and_reg, NULL);
shift_test = gen_rtx_AND (mode, gen_rtx_ASHIFTRT (mode, and_reg, NULL),
const1_rtx);
}
else
{
PUT_MODE (and_reg, mode);
PUT_MODE (and_test, mode);
PUT_MODE (shift_test, mode);
PUT_MODE (XEXP (shift_test, 0), mode);
}
XEXP (and_test, 1) = immed_wide_int_const (mask, mode);
XEXP (XEXP (shift_test, 0), 1) = GEN_INT (bitnum);
speed_p = optimize_insn_for_speed_p ();
return (rtx_cost (and_test, mode, IF_THEN_ELSE, 0, speed_p)
<= rtx_cost (shift_test, mode, IF_THEN_ELSE, 0, speed_p));
}
void
do_jump_1 (enum tree_code code, tree op0, tree op1,
rtx_code_label *if_false_label, rtx_code_label *if_true_label,
profile_probability prob)
{
machine_mode mode;
rtx_code_label *drop_through_label = 0;
scalar_int_mode int_mode;
switch (code)
{
case EQ_EXPR:
{
tree inner_type = TREE_TYPE (op0);
gcc_assert (GET_MODE_CLASS (TYPE_MODE (inner_type))
!= MODE_COMPLEX_FLOAT);
gcc_assert (GET_MODE_CLASS (TYPE_MODE (inner_type))
!= MODE_COMPLEX_INT);
if (integer_zerop (op1))
do_jump (op0, if_true_label, if_false_label,
prob.invert ());
else if (is_int_mode (TYPE_MODE (inner_type), &int_mode)
&& !can_compare_p (EQ, int_mode, ccp_jump))
do_jump_by_parts_equality (int_mode, op0, op1, if_false_label,
if_true_label, prob);
else
do_compare_and_jump (op0, op1, EQ, EQ, if_false_label, if_true_label,
prob);
break;
}
case NE_EXPR:
{
tree inner_type = TREE_TYPE (op0);
gcc_assert (GET_MODE_CLASS (TYPE_MODE (inner_type))
!= MODE_COMPLEX_FLOAT);
gcc_assert (GET_MODE_CLASS (TYPE_MODE (inner_type))
!= MODE_COMPLEX_INT);
if (integer_zerop (op1))
do_jump (op0, if_false_label, if_true_label, prob);
else if (is_int_mode (TYPE_MODE (inner_type), &int_mode)
&& !can_compare_p (NE, int_mode, ccp_jump))
do_jump_by_parts_equality (int_mode, op0, op1, if_true_label,
if_false_label, prob.invert ());
else
do_compare_and_jump (op0, op1, NE, NE, if_false_label, if_true_label,
prob);
break;
}
case LT_EXPR:
mode = TYPE_MODE (TREE_TYPE (op0));
if (is_int_mode (mode, &int_mode)
&& ! can_compare_p (LT, int_mode, ccp_jump))
do_jump_by_parts_greater (int_mode, op0, op1, 1, if_false_label,
if_true_label, prob);
else
do_compare_and_jump (op0, op1, LT, LTU, if_false_label, if_true_label,
prob);
break;
case LE_EXPR:
mode = TYPE_MODE (TREE_TYPE (op0));
if (is_int_mode (mode, &int_mode)
&& ! can_compare_p (LE, int_mode, ccp_jump))
do_jump_by_parts_greater (int_mode, op0, op1, 0, if_true_label,
if_false_label, prob.invert ());
else
do_compare_and_jump (op0, op1, LE, LEU, if_false_label, if_true_label,
prob);
break;
case GT_EXPR:
mode = TYPE_MODE (TREE_TYPE (op0));
if (is_int_mode (mode, &int_mode)
&& ! can_compare_p (GT, int_mode, ccp_jump))
do_jump_by_parts_greater (int_mode, op0, op1, 0, if_false_label,
if_true_label, prob);
else
do_compare_and_jump (op0, op1, GT, GTU, if_false_label, if_true_label,
prob);
break;
case GE_EXPR:
mode = TYPE_MODE (TREE_TYPE (op0));
if (is_int_mode (mode, &int_mode)
&& ! can_compare_p (GE, int_mode, ccp_jump))
do_jump_by_parts_greater (int_mode, op0, op1, 1, if_true_label,
if_false_label, prob.invert ());
else
do_compare_and_jump (op0, op1, GE, GEU, if_false_label, if_true_label,
prob);
break;
case ORDERED_EXPR:
do_compare_and_jump (op0, op1, ORDERED, ORDERED,
if_false_label, if_true_label, prob);
break;
case UNORDERED_EXPR:
do_compare_and_jump (op0, op1, UNORDERED, UNORDERED,
if_false_label, if_true_label, prob);
break;
case UNLT_EXPR:
do_compare_and_jump (op0, op1, UNLT, UNLT, if_false_label, if_true_label,
prob);
break;
case UNLE_EXPR:
do_compare_and_jump (op0, op1, UNLE, UNLE, if_false_label, if_true_label,
prob);
break;
case UNGT_EXPR:
do_compare_and_jump (op0, op1, UNGT, UNGT, if_false_label, if_true_label,
prob);
break;
case UNGE_EXPR:
do_compare_and_jump (op0, op1, UNGE, UNGE, if_false_label, if_true_label,
prob);
break;
case UNEQ_EXPR:
do_compare_and_jump (op0, op1, UNEQ, UNEQ, if_false_label, if_true_label,
prob);
break;
case LTGT_EXPR:
do_compare_and_jump (op0, op1, LTGT, LTGT, if_false_label, if_true_label,
prob);
break;
case TRUTH_ANDIF_EXPR:
{
profile_probability op0_prob = profile_probability::uninitialized ();
profile_probability op1_prob = profile_probability::uninitialized ();
if (prob.initialized_p ())
{
op1_prob = prob.invert ();
op0_prob = op1_prob.split (profile_probability::even ());
op0_prob = op0_prob.invert ();
op1_prob = op1_prob.invert ();
}
if (if_false_label == NULL)
{
drop_through_label = gen_label_rtx ();
do_jump (op0, drop_through_label, NULL, op0_prob);
do_jump (op1, NULL, if_true_label, op1_prob);
}
else
{
do_jump (op0, if_false_label, NULL, op0_prob);
do_jump (op1, if_false_label, if_true_label, op1_prob);
}
break;
}
case TRUTH_ORIF_EXPR:
{
profile_probability op0_prob = profile_probability::uninitialized ();
profile_probability op1_prob = profile_probability::uninitialized ();
if (prob.initialized_p ())
{
op1_prob = prob;
op0_prob = op1_prob.split (profile_probability::even ());
}
if (if_true_label == NULL)
{
drop_through_label = gen_label_rtx ();
do_jump (op0, NULL, drop_through_label, op0_prob);
do_jump (op1, if_false_label, NULL, op1_prob);
}
else
{
do_jump (op0, NULL, if_true_label, op0_prob);
do_jump (op1, if_false_label, if_true_label, op1_prob);
}
break;
}
default:
gcc_unreachable ();
}
if (drop_through_label)
{
do_pending_stack_adjust ();
emit_label (drop_through_label);
}
}
void
do_jump (tree exp, rtx_code_label *if_false_label,
rtx_code_label *if_true_label, profile_probability prob)
{
enum tree_code code = TREE_CODE (exp);
rtx temp;
int i;
tree type;
scalar_int_mode mode;
rtx_code_label *drop_through_label = NULL;
switch (code)
{
case ERROR_MARK:
break;
case INTEGER_CST:
{
rtx_code_label *lab = integer_zerop (exp) ? if_false_label
: if_true_label;
if (lab)
emit_jump (lab);
break;
}
#if 0
case ADDR_EXPR:
if (if_true_label)
emit_jump (if_true_label);
break;
#endif
case NOP_EXPR:
if (TREE_CODE (TREE_OPERAND (exp, 0)) == COMPONENT_REF
|| TREE_CODE (TREE_OPERAND (exp, 0)) == BIT_FIELD_REF
|| TREE_CODE (TREE_OPERAND (exp, 0)) == ARRAY_REF
|| TREE_CODE (TREE_OPERAND (exp, 0)) == ARRAY_RANGE_REF)
goto normal;
case CONVERT_EXPR:
if ((TYPE_PRECISION (TREE_TYPE (exp))
< TYPE_PRECISION (TREE_TYPE (TREE_OPERAND (exp, 0)))))
goto normal;
case NON_LVALUE_EXPR:
case ABS_EXPR:
case NEGATE_EXPR:
case LROTATE_EXPR:
case RROTATE_EXPR:
do_jump (TREE_OPERAND (exp, 0), if_false_label, if_true_label, prob);
break;
case TRUTH_NOT_EXPR:
do_jump (TREE_OPERAND (exp, 0), if_true_label, if_false_label,
prob.invert ());
break;
case COND_EXPR:
{
rtx_code_label *label1 = gen_label_rtx ();
if (!if_true_label || !if_false_label)
{
drop_through_label = gen_label_rtx ();
if (!if_true_label)
if_true_label = drop_through_label;
if (!if_false_label)
if_false_label = drop_through_label;
}
do_pending_stack_adjust ();
do_jump (TREE_OPERAND (exp, 0), label1, NULL,
profile_probability::uninitialized ());
do_jump (TREE_OPERAND (exp, 1), if_false_label, if_true_label, prob);
emit_label (label1);
do_jump (TREE_OPERAND (exp, 2), if_false_label, if_true_label, prob);
break;
}
case COMPOUND_EXPR:
gcc_unreachable ();
case MINUS_EXPR:
code = NE_EXPR;
case EQ_EXPR:
case NE_EXPR:
case LT_EXPR:
case LE_EXPR:
case GT_EXPR:
case GE_EXPR:
case ORDERED_EXPR:
case UNORDERED_EXPR:
case UNLT_EXPR:
case UNLE_EXPR:
case UNGT_EXPR:
case UNGE_EXPR:
case UNEQ_EXPR:
case LTGT_EXPR:
case TRUTH_ANDIF_EXPR:
case TRUTH_ORIF_EXPR:
other_code:
do_jump_1 (code, TREE_OPERAND (exp, 0), TREE_OPERAND (exp, 1),
if_false_label, if_true_label, prob);
break;
case BIT_AND_EXPR:
if (integer_onep (TREE_OPERAND (exp, 1)))
{
tree exp0 = TREE_OPERAND (exp, 0);
rtx_code_label *set_label, *clr_label;
profile_probability setclr_prob = prob;
while (CONVERT_EXPR_P (exp0)
&& TREE_OPERAND (exp0, 0) != error_mark_node
&& TYPE_PRECISION (TREE_TYPE (exp0))
<= TYPE_PRECISION (TREE_TYPE (TREE_OPERAND (exp0, 0))))
exp0 = TREE_OPERAND (exp0, 0);
if (TREE_CODE (exp0) == BIT_XOR_EXPR
&& integer_onep (TREE_OPERAND (exp0, 1)))
{
exp0 = TREE_OPERAND (exp0, 0);
clr_label = if_true_label;
set_label = if_false_label;
setclr_prob = prob.invert ();
}
else
{
clr_label = if_false_label;
set_label = if_true_label;
}
if (TREE_CODE (exp0) == RSHIFT_EXPR)
{
tree arg = TREE_OPERAND (exp0, 0);
tree shift = TREE_OPERAND (exp0, 1);
tree argtype = TREE_TYPE (arg);
if (TREE_CODE (shift) == INTEGER_CST
&& compare_tree_int (shift, 0) >= 0
&& compare_tree_int (shift, HOST_BITS_PER_WIDE_INT) < 0
&& prefer_and_bit_test (SCALAR_INT_TYPE_MODE (argtype),
TREE_INT_CST_LOW (shift)))
{
unsigned HOST_WIDE_INT mask
= HOST_WIDE_INT_1U << TREE_INT_CST_LOW (shift);
do_jump (build2 (BIT_AND_EXPR, argtype, arg,
build_int_cstu (argtype, mask)),
clr_label, set_label, setclr_prob);
break;
}
}
}
if (! SLOW_BYTE_ACCESS
&& TREE_CODE (TREE_OPERAND (exp, 1)) == INTEGER_CST
&& TYPE_PRECISION (TREE_TYPE (exp)) <= HOST_BITS_PER_WIDE_INT
&& (i = tree_floor_log2 (TREE_OPERAND (exp, 1))) >= 0
&& int_mode_for_size (i + 1, 0).exists (&mode)
&& (type = lang_hooks.types.type_for_mode (mode, 1)) != 0
&& TYPE_PRECISION (type) < TYPE_PRECISION (TREE_TYPE (exp))
&& have_insn_for (COMPARE, TYPE_MODE (type)))
{
do_jump (fold_convert (type, exp), if_false_label, if_true_label,
prob);
break;
}
if (TYPE_PRECISION (TREE_TYPE (exp)) > 1
|| TREE_CODE (TREE_OPERAND (exp, 1)) == INTEGER_CST)
goto normal;
case TRUTH_AND_EXPR:
if (BRANCH_COST (optimize_insn_for_speed_p (),
false) >= 4
|| TREE_SIDE_EFFECTS (TREE_OPERAND (exp, 1)))
goto normal;
code = TRUTH_ANDIF_EXPR;
goto other_code;
case BIT_IOR_EXPR:
case TRUTH_OR_EXPR:
if (BRANCH_COST (optimize_insn_for_speed_p (), false) >= 4
|| TREE_SIDE_EFFECTS (TREE_OPERAND (exp, 1)))
goto normal;
code = TRUTH_ORIF_EXPR;
goto other_code;
default:
normal:
temp = expand_normal (exp);
do_pending_stack_adjust ();
if (GET_CODE (temp) == SUBREG)
{
if (SUBREG_PROMOTED_VAR_P (temp)
&& REG_P (XEXP (temp, 0)))
temp = XEXP (temp, 0);
else
temp = copy_to_reg (temp);
}
do_compare_rtx_and_jump (temp, CONST0_RTX (GET_MODE (temp)),
NE, TYPE_UNSIGNED (TREE_TYPE (exp)),
GET_MODE (temp), NULL_RTX,
if_false_label, if_true_label, prob);
}
if (drop_through_label)
{
do_pending_stack_adjust ();
emit_label (drop_through_label);
}
}

static void
do_jump_by_parts_greater_rtx (scalar_int_mode mode, int unsignedp, rtx op0,
rtx op1, rtx_code_label *if_false_label,
rtx_code_label *if_true_label,
profile_probability prob)
{
int nwords = (GET_MODE_SIZE (mode) / UNITS_PER_WORD);
rtx_code_label *drop_through_label = 0;
bool drop_through_if_true = false, drop_through_if_false = false;
enum rtx_code code = GT;
int i;
if (! if_true_label || ! if_false_label)
drop_through_label = gen_label_rtx ();
if (! if_true_label)
{
if_true_label = drop_through_label;
drop_through_if_true = true;
}
if (! if_false_label)
{
if_false_label = drop_through_label;
drop_through_if_false = true;
}
if (op0 == const0_rtx && drop_through_if_true && !drop_through_if_false)
{
code = LE;
if_true_label = if_false_label;
if_false_label = drop_through_label;
drop_through_if_true = false;
drop_through_if_false = true;
prob = prob.invert ();
}
for (i = 0; i < nwords; i++)
{
rtx op0_word, op1_word;
if (WORDS_BIG_ENDIAN)
{
op0_word = operand_subword_force (op0, i, mode);
op1_word = operand_subword_force (op1, i, mode);
}
else
{
op0_word = operand_subword_force (op0, nwords - 1 - i, mode);
op1_word = operand_subword_force (op1, nwords - 1 - i, mode);
}
do_compare_rtx_and_jump (op0_word, op1_word, code, (unsignedp || i > 0),
word_mode, NULL_RTX, NULL, if_true_label,
prob);
if (op0 == const0_rtx || i == nwords - 1)
break;
do_compare_rtx_and_jump (op0_word, op1_word, NE, unsignedp, word_mode,
NULL_RTX, NULL, if_false_label,
prob.invert ());
}
if (!drop_through_if_false)
emit_jump (if_false_label);
if (drop_through_label)
emit_label (drop_through_label);
}
static void
do_jump_by_parts_greater (scalar_int_mode mode, tree treeop0, tree treeop1,
int swap, rtx_code_label *if_false_label,
rtx_code_label *if_true_label,
profile_probability prob)
{
rtx op0 = expand_normal (swap ? treeop1 : treeop0);
rtx op1 = expand_normal (swap ? treeop0 : treeop1);
int unsignedp = TYPE_UNSIGNED (TREE_TYPE (treeop0));
do_jump_by_parts_greater_rtx (mode, unsignedp, op0, op1, if_false_label,
if_true_label, prob);
}

static void
do_jump_by_parts_zero_rtx (scalar_int_mode mode, rtx op0,
rtx_code_label *if_false_label,
rtx_code_label *if_true_label,
profile_probability prob)
{
int nwords = GET_MODE_SIZE (mode) / UNITS_PER_WORD;
rtx part;
int i;
rtx_code_label *drop_through_label = NULL;
part = gen_reg_rtx (word_mode);
emit_move_insn (part, operand_subword_force (op0, 0, mode));
for (i = 1; i < nwords && part != 0; i++)
part = expand_binop (word_mode, ior_optab, part,
operand_subword_force (op0, i, mode),
part, 1, OPTAB_WIDEN);
if (part != 0)
{
do_compare_rtx_and_jump (part, const0_rtx, EQ, 1, word_mode,
NULL_RTX, if_false_label, if_true_label, prob);
return;
}
if (! if_false_label)
if_false_label = drop_through_label = gen_label_rtx ();
for (i = 0; i < nwords; i++)
do_compare_rtx_and_jump (operand_subword_force (op0, i, mode),
const0_rtx, EQ, 1, word_mode, NULL_RTX,
if_false_label, NULL, prob);
if (if_true_label)
emit_jump (if_true_label);
if (drop_through_label)
emit_label (drop_through_label);
}
static void
do_jump_by_parts_equality_rtx (scalar_int_mode mode, rtx op0, rtx op1,
rtx_code_label *if_false_label,
rtx_code_label *if_true_label,
profile_probability prob)
{
int nwords = (GET_MODE_SIZE (mode) / UNITS_PER_WORD);
rtx_code_label *drop_through_label = NULL;
int i;
if (op1 == const0_rtx)
{
do_jump_by_parts_zero_rtx (mode, op0, if_false_label, if_true_label,
prob);
return;
}
else if (op0 == const0_rtx)
{
do_jump_by_parts_zero_rtx (mode, op1, if_false_label, if_true_label,
prob);
return;
}
if (! if_false_label)
drop_through_label = if_false_label = gen_label_rtx ();
for (i = 0; i < nwords; i++)
do_compare_rtx_and_jump (operand_subword_force (op0, i, mode),
operand_subword_force (op1, i, mode),
EQ, 0, word_mode, NULL_RTX,
if_false_label, NULL, prob);
if (if_true_label)
emit_jump (if_true_label);
if (drop_through_label)
emit_label (drop_through_label);
}
static void
do_jump_by_parts_equality (scalar_int_mode mode, tree treeop0, tree treeop1,
rtx_code_label *if_false_label,
rtx_code_label *if_true_label,
profile_probability prob)
{
rtx op0 = expand_normal (treeop0);
rtx op1 = expand_normal (treeop1);
do_jump_by_parts_equality_rtx (mode, op0, op1, if_false_label,
if_true_label, prob);
}

bool
split_comparison (enum rtx_code code, machine_mode mode,
enum rtx_code *code1, enum rtx_code *code2)
{
switch (code)
{
case LT:
*code1 = ORDERED;
*code2 = UNLT;
return true;
case LE:
*code1 = ORDERED;
*code2 = UNLE;
return true;
case GT:
*code1 = ORDERED;
*code2 = UNGT;
return true;
case GE:
*code1 = ORDERED;
*code2 = UNGE;
return true;
case EQ:
*code1 = ORDERED;
*code2 = UNEQ;
return true;
case NE:
*code1 = UNORDERED;
*code2 = LTGT;
return false;
case UNLT:
*code1 = UNORDERED;
*code2 = LT;
return false;
case UNLE:
*code1 = UNORDERED;
*code2 = LE;
return false;
case UNGT:
*code1 = UNORDERED;
*code2 = GT;
return false;
case UNGE:
*code1 = UNORDERED;
*code2 = GE;
return false;
case UNEQ:
*code1 = UNORDERED;
*code2 = EQ;
return false;
case LTGT:
if (HONOR_SNANS (mode))
{
*code1 = LT;
*code2 = GT;
return false;
}
else
{
*code1 = ORDERED;
*code2 = NE;
return true;
}
default:
gcc_unreachable ();
}
}
void
do_compare_rtx_and_jump (rtx op0, rtx op1, enum rtx_code code, int unsignedp,
machine_mode mode, rtx size,
rtx_code_label *if_false_label,
rtx_code_label *if_true_label,
profile_probability prob)
{
rtx tem;
rtx_code_label *dummy_label = NULL;
if ((! if_true_label
|| ! can_compare_p (code, mode, ccp_jump))
&& (! FLOAT_MODE_P (mode)
|| code == ORDERED || code == UNORDERED
|| (! HONOR_NANS (mode) && (code == LTGT || code == UNEQ))
|| (! HONOR_SNANS (mode) && (code == EQ || code == NE))))
{
enum rtx_code rcode;
if (FLOAT_MODE_P (mode))
rcode = reverse_condition_maybe_unordered (code);
else
rcode = reverse_condition (code);
if (can_compare_p (rcode, mode, ccp_jump)
|| (code == ORDERED && ! can_compare_p (ORDERED, mode, ccp_jump)))
{
std::swap (if_true_label, if_false_label);
code = rcode;
prob = prob.invert ();
}
}
if (swap_commutative_operands_p (op0, op1))
{
std::swap (op0, op1);
code = swap_condition (code);
}
do_pending_stack_adjust ();
code = unsignedp ? unsigned_condition (code) : code;
if ((tem = simplify_relational_operation (code, mode, VOIDmode,
op0, op1)) != 0)
{
if (CONSTANT_P (tem))
{
rtx_code_label *label = (tem == const0_rtx
|| tem == CONST0_RTX (mode))
? if_false_label : if_true_label;
if (label)
emit_jump (label);
return;
}
code = GET_CODE (tem);
mode = GET_MODE (tem);
op0 = XEXP (tem, 0);
op1 = XEXP (tem, 1);
unsignedp = (code == GTU || code == LTU || code == GEU || code == LEU);
}
if (! if_true_label)
dummy_label = if_true_label = gen_label_rtx ();
scalar_int_mode int_mode;
if (is_int_mode (mode, &int_mode)
&& ! can_compare_p (code, int_mode, ccp_jump))
{
switch (code)
{
case LTU:
do_jump_by_parts_greater_rtx (int_mode, 1, op1, op0,
if_false_label, if_true_label, prob);
break;
case LEU:
do_jump_by_parts_greater_rtx (int_mode, 1, op0, op1,
if_true_label, if_false_label,
prob.invert ());
break;
case GTU:
do_jump_by_parts_greater_rtx (int_mode, 1, op0, op1,
if_false_label, if_true_label, prob);
break;
case GEU:
do_jump_by_parts_greater_rtx (int_mode, 1, op1, op0,
if_true_label, if_false_label,
prob.invert ());
break;
case LT:
do_jump_by_parts_greater_rtx (int_mode, 0, op1, op0,
if_false_label, if_true_label, prob);
break;
case LE:
do_jump_by_parts_greater_rtx (int_mode, 0, op0, op1,
if_true_label, if_false_label,
prob.invert ());
break;
case GT:
do_jump_by_parts_greater_rtx (int_mode, 0, op0, op1,
if_false_label, if_true_label, prob);
break;
case GE:
do_jump_by_parts_greater_rtx (int_mode, 0, op1, op0,
if_true_label, if_false_label,
prob.invert ());
break;
case EQ:
do_jump_by_parts_equality_rtx (int_mode, op0, op1, if_false_label,
if_true_label, prob);
break;
case NE:
do_jump_by_parts_equality_rtx (int_mode, op0, op1, if_true_label,
if_false_label,
prob.invert ());
break;
default:
gcc_unreachable ();
}
}
else
{
if (SCALAR_FLOAT_MODE_P (mode)
&& ! can_compare_p (code, mode, ccp_jump)
&& can_compare_p (swap_condition (code), mode, ccp_jump))
{
code = swap_condition (code);
std::swap (op0, op1);
}
else if (SCALAR_FLOAT_MODE_P (mode)
&& ! can_compare_p (code, mode, ccp_jump)
&& (code != ORDERED && code != UNORDERED)
&& (have_insn_for (COMPARE, mode)
|| code_to_optab (code) == unknown_optab))
{
enum rtx_code first_code;
bool and_them = split_comparison (code, mode, &first_code, &code);
if (!HONOR_NANS (mode))
gcc_assert (first_code == (and_them ? ORDERED : UNORDERED));
else
{
profile_probability cprob
= profile_probability::guessed_always ();
if (first_code == UNORDERED)
cprob = cprob.apply_scale (1, 100);
else if (first_code == ORDERED)
cprob = cprob.apply_scale (99, 100);
else
cprob = profile_probability::even ();
if (and_them)
{
rtx_code_label *dest_label;
prob = prob.invert ();
profile_probability first_prob = prob.split (cprob).invert ();
prob = prob.invert ();
if (! if_false_label)
{
if (! dummy_label)
dummy_label = gen_label_rtx ();
dest_label = dummy_label;
}
else
dest_label = if_false_label;
do_compare_rtx_and_jump (op0, op1, first_code, unsignedp, mode,
size, dest_label, NULL, first_prob);
}
else
{
profile_probability first_prob = prob.split (cprob);
do_compare_rtx_and_jump (op0, op1, first_code, unsignedp, mode,
size, NULL, if_true_label, first_prob);
}
}
}
emit_cmp_and_jump_insns (op0, op1, code, size, mode, unsignedp,
if_true_label, prob);
}
if (if_false_label)
emit_jump (if_false_label);
if (dummy_label)
emit_label (dummy_label);
}
static void
do_compare_and_jump (tree treeop0, tree treeop1, enum rtx_code signed_code,
enum rtx_code unsigned_code,
rtx_code_label *if_false_label,
rtx_code_label *if_true_label, profile_probability prob)
{
rtx op0, op1;
tree type;
machine_mode mode;
int unsignedp;
enum rtx_code code;
op0 = expand_normal (treeop0);
if (TREE_CODE (treeop0) == ERROR_MARK)
return;
op1 = expand_normal (treeop1);
if (TREE_CODE (treeop1) == ERROR_MARK)
return;
type = TREE_TYPE (treeop0);
if (TREE_CODE (treeop0) == INTEGER_CST
&& (TREE_CODE (treeop1) != INTEGER_CST
|| (GET_MODE_BITSIZE (SCALAR_INT_TYPE_MODE (type))
> GET_MODE_BITSIZE (SCALAR_INT_TYPE_MODE (TREE_TYPE (treeop1))))))
type = TREE_TYPE (treeop1);
mode = TYPE_MODE (type);
unsignedp = TYPE_UNSIGNED (type);
code = unsignedp ? unsigned_code : signed_code;
if (targetm.have_canonicalize_funcptr_for_compare ()
&& ((POINTER_TYPE_P (TREE_TYPE (treeop0))
&& FUNC_OR_METHOD_TYPE_P (TREE_TYPE (TREE_TYPE (treeop0))))
|| (POINTER_TYPE_P (TREE_TYPE (treeop1))
&& FUNC_OR_METHOD_TYPE_P (TREE_TYPE (TREE_TYPE (treeop1))))))
{
rtx new_op0 = gen_reg_rtx (mode);
rtx new_op1 = gen_reg_rtx (mode);
emit_insn (targetm.gen_canonicalize_funcptr_for_compare (new_op0, op0));
op0 = new_op0;
emit_insn (targetm.gen_canonicalize_funcptr_for_compare (new_op1, op1));
op1 = new_op1;
}
do_compare_rtx_and_jump (op0, op1, code, unsignedp, mode,
((mode == BLKmode)
? expr_size (treeop0) : NULL_RTX),
if_false_label, if_true_label, prob);
}
#include "gt-dojump.h"
