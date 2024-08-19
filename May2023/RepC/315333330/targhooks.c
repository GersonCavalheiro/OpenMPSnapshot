#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "function.h"
#include "rtl.h"
#include "tree.h"
#include "tree-ssa-alias.h"
#include "gimple-expr.h"
#include "memmodel.h"
#include "tm_p.h"
#include "stringpool.h"
#include "tree-vrp.h"
#include "tree-ssanames.h"
#include "profile-count.h"
#include "optabs.h"
#include "regs.h"
#include "recog.h"
#include "diagnostic-core.h"
#include "fold-const.h"
#include "stor-layout.h"
#include "varasm.h"
#include "flags.h"
#include "explow.h"
#include "calls.h"
#include "expr.h"
#include "output.h"
#include "common/common-target.h"
#include "reload.h"
#include "intl.h"
#include "opts.h"
#include "gimplify.h"
#include "predict.h"
#include "params.h"
#include "real.h"
#include "langhooks.h"
#include "sbitmap.h"
bool
default_legitimate_address_p (machine_mode mode ATTRIBUTE_UNUSED,
rtx addr ATTRIBUTE_UNUSED,
bool strict ATTRIBUTE_UNUSED)
{
#ifdef GO_IF_LEGITIMATE_ADDRESS
if (strict)
return strict_memory_address_p (mode, addr);
else
return memory_address_p (mode, addr);
#else
gcc_unreachable ();
#endif
}
void
default_external_libcall (rtx fun ATTRIBUTE_UNUSED)
{
#ifdef ASM_OUTPUT_EXTERNAL_LIBCALL
ASM_OUTPUT_EXTERNAL_LIBCALL (asm_out_file, fun);
#endif
}
int
default_unspec_may_trap_p (const_rtx x, unsigned flags)
{
int i;
if ((SCALAR_FLOAT_MODE_P (GET_MODE (x)) && flag_trapping_math))
return 1;
for (i = 0; i < XVECLEN (x, 0); ++i)
{
if (may_trap_p_1 (XVECEXP (x, 0, i), flags))
return 1;
}
return 0;
}
machine_mode
default_promote_function_mode (const_tree type ATTRIBUTE_UNUSED,
machine_mode mode,
int *punsignedp ATTRIBUTE_UNUSED,
const_tree funtype ATTRIBUTE_UNUSED,
int for_return ATTRIBUTE_UNUSED)
{
if (type != NULL_TREE && for_return == 2)
return promote_mode (type, mode, punsignedp);
return mode;
}
machine_mode
default_promote_function_mode_always_promote (const_tree type,
machine_mode mode,
int *punsignedp,
const_tree funtype ATTRIBUTE_UNUSED,
int for_return ATTRIBUTE_UNUSED)
{
return promote_mode (type, mode, punsignedp);
}
machine_mode
default_cc_modes_compatible (machine_mode m1, machine_mode m2)
{
if (m1 == m2)
return m1;
return VOIDmode;
}
bool
default_return_in_memory (const_tree type,
const_tree fntype ATTRIBUTE_UNUSED)
{
return (TYPE_MODE (type) == BLKmode);
}
rtx
default_legitimize_address (rtx x, rtx orig_x ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED)
{
return x;
}
bool
default_legitimize_address_displacement (rtx *, rtx *, poly_int64,
machine_mode)
{
return false;
}
bool
default_const_not_ok_for_debug_p (rtx x)
{
if (GET_CODE (x) == UNSPEC)
return true;
return false;
}
rtx
default_expand_builtin_saveregs (void)
{
error ("__builtin_saveregs not supported by this target");
return const0_rtx;
}
void
default_setup_incoming_varargs (cumulative_args_t ca ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED,
tree type ATTRIBUTE_UNUSED,
int *pretend_arg_size ATTRIBUTE_UNUSED,
int second_time ATTRIBUTE_UNUSED)
{
}
rtx
default_builtin_setjmp_frame_value (void)
{
return virtual_stack_vars_rtx;
}
bool
hook_bool_CUMULATIVE_ARGS_false (cumulative_args_t ca ATTRIBUTE_UNUSED)
{
return false;
}
bool
default_pretend_outgoing_varargs_named (cumulative_args_t ca ATTRIBUTE_UNUSED)
{
return (targetm.calls.setup_incoming_varargs
!= default_setup_incoming_varargs);
}
scalar_int_mode
default_eh_return_filter_mode (void)
{
return targetm.unwind_word_mode ();
}
scalar_int_mode
default_libgcc_cmp_return_mode (void)
{
return word_mode;
}
scalar_int_mode
default_libgcc_shift_count_mode (void)
{
return word_mode;
}
scalar_int_mode
default_unwind_word_mode (void)
{
return word_mode;
}
unsigned HOST_WIDE_INT
default_shift_truncation_mask (machine_mode mode)
{
return SHIFT_COUNT_TRUNCATED ? GET_MODE_UNIT_BITSIZE (mode) - 1 : 0;
}
unsigned int
default_min_divisions_for_recip_mul (machine_mode mode ATTRIBUTE_UNUSED)
{
return have_insn_for (DIV, mode) ? 3 : 2;
}
int
default_mode_rep_extended (scalar_int_mode, scalar_int_mode)
{
return UNKNOWN;
}
bool
hook_bool_CUMULATIVE_ARGS_true (cumulative_args_t a ATTRIBUTE_UNUSED)
{
return true;
}
machine_mode
default_mode_for_suffix (char suffix ATTRIBUTE_UNUSED)
{
return VOIDmode;
}
tree
default_cxx_guard_type (void)
{
return long_long_integer_type_node;
}
tree
default_cxx_get_cookie_size (tree type)
{
tree cookie_size;
tree sizetype_size;
tree type_align;
sizetype_size = size_in_bytes (sizetype);
type_align = size_int (TYPE_ALIGN_UNIT (type));
if (tree_int_cst_lt (type_align, sizetype_size))
cookie_size = sizetype_size;
else
cookie_size = type_align;
return cookie_size;
}
bool
hook_pass_by_reference_must_pass_in_stack (cumulative_args_t c ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED, const_tree type ATTRIBUTE_UNUSED,
bool named_arg ATTRIBUTE_UNUSED)
{
return targetm.calls.must_pass_in_stack (mode, type);
}
bool
hook_callee_copies_named (cumulative_args_t ca ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED,
const_tree type ATTRIBUTE_UNUSED, bool named)
{
return named;
}
void
default_print_operand (FILE *stream ATTRIBUTE_UNUSED, rtx x ATTRIBUTE_UNUSED,
int code ATTRIBUTE_UNUSED)
{
#ifdef PRINT_OPERAND
PRINT_OPERAND (stream, x, code);
#else
gcc_unreachable ();
#endif
}
void
default_print_operand_address (FILE *stream ATTRIBUTE_UNUSED,
machine_mode ,
rtx x ATTRIBUTE_UNUSED)
{
#ifdef PRINT_OPERAND_ADDRESS
PRINT_OPERAND_ADDRESS (stream, x);
#else
gcc_unreachable ();
#endif
}
bool
default_print_operand_punct_valid_p (unsigned char code ATTRIBUTE_UNUSED)
{
#ifdef PRINT_OPERAND_PUNCT_VALID_P
return PRINT_OPERAND_PUNCT_VALID_P (code);
#else
return false;
#endif
}
tree
default_mangle_assembler_name (const char *name ATTRIBUTE_UNUSED)
{
const char *skipped = name + (*name == '*' ? 1 : 0);
const char *stripped = targetm.strip_name_encoding (skipped);
if (*name != '*' && user_label_prefix[0])
stripped = ACONCAT ((user_label_prefix, stripped, NULL));
return get_identifier (stripped);
}
bool
default_scalar_mode_supported_p (scalar_mode mode)
{
int precision = GET_MODE_PRECISION (mode);
switch (GET_MODE_CLASS (mode))
{
case MODE_PARTIAL_INT:
case MODE_INT:
if (precision == CHAR_TYPE_SIZE)
return true;
if (precision == SHORT_TYPE_SIZE)
return true;
if (precision == INT_TYPE_SIZE)
return true;
if (precision == LONG_TYPE_SIZE)
return true;
if (precision == LONG_LONG_TYPE_SIZE)
return true;
if (precision == 2 * BITS_PER_WORD)
return true;
return false;
case MODE_FLOAT:
if (precision == FLOAT_TYPE_SIZE)
return true;
if (precision == DOUBLE_TYPE_SIZE)
return true;
if (precision == LONG_DOUBLE_TYPE_SIZE)
return true;
return false;
case MODE_DECIMAL_FLOAT:
case MODE_FRACT:
case MODE_UFRACT:
case MODE_ACCUM:
case MODE_UACCUM:
return false;
default:
gcc_unreachable ();
}
}
bool
default_libgcc_floating_mode_supported_p (scalar_float_mode mode)
{
switch (mode)
{
#ifdef HAVE_SFmode
case E_SFmode:
#endif
#ifdef HAVE_DFmode
case E_DFmode:
#endif
#ifdef HAVE_XFmode
case E_XFmode:
#endif
#ifdef HAVE_TFmode
case E_TFmode:
#endif
return true;
default:
return false;
}
}
opt_scalar_float_mode
default_floatn_mode (int n, bool extended)
{
if (extended)
{
opt_scalar_float_mode cand1, cand2;
scalar_float_mode mode;
switch (n)
{
case 32:
#ifdef HAVE_DFmode
cand1 = DFmode;
#endif
break;
case 64:
#ifdef HAVE_XFmode
cand1 = XFmode;
#endif
#ifdef HAVE_TFmode
cand2 = TFmode;
#endif
break;
case 128:
break;
default:
gcc_unreachable ();
}
if (cand1.exists (&mode)
&& REAL_MODE_FORMAT (mode)->ieee_bits > n
&& targetm.scalar_mode_supported_p (mode)
&& targetm.libgcc_floating_mode_supported_p (mode))
return cand1;
if (cand2.exists (&mode)
&& REAL_MODE_FORMAT (mode)->ieee_bits > n
&& targetm.scalar_mode_supported_p (mode)
&& targetm.libgcc_floating_mode_supported_p (mode))
return cand2;
}
else
{
opt_scalar_float_mode cand;
scalar_float_mode mode;
switch (n)
{
case 16:
#ifdef HAVE_HFmode
cand = HFmode;
#endif
break;
case 32:
#ifdef HAVE_SFmode
cand = SFmode;
#endif
break;
case 64:
#ifdef HAVE_DFmode
cand = DFmode;
#endif
break;
case 128:
#ifdef HAVE_TFmode
cand = TFmode;
#endif
break;
default:
break;
}
if (cand.exists (&mode)
&& REAL_MODE_FORMAT (mode)->ieee_bits == n
&& targetm.scalar_mode_supported_p (mode)
&& targetm.libgcc_floating_mode_supported_p (mode))
return cand;
}
return opt_scalar_float_mode ();
}
bool
default_floatn_builtin_p (int func ATTRIBUTE_UNUSED)
{
static bool first_time_p = true;
static bool c_or_objective_c;
if (first_time_p)
{
first_time_p = false;
c_or_objective_c = lang_GNU_C () || lang_GNU_OBJC ();
}
return c_or_objective_c;
}
bool
targhook_words_big_endian (void)
{
return !!WORDS_BIG_ENDIAN;
}
bool
targhook_float_words_big_endian (void)
{
return !!FLOAT_WORDS_BIG_ENDIAN;
}
bool
default_float_exceptions_rounding_supported_p (void)
{
#ifdef HAVE_adddf3
return HAVE_adddf3;
#else
return false;
#endif
}
bool
default_decimal_float_supported_p (void)
{
return ENABLE_DECIMAL_FLOAT;
}
bool
default_fixed_point_supported_p (void)
{
return ENABLE_FIXED_POINT;
}
bool
default_has_ifunc_p (void)
{
return HAVE_GNU_INDIRECT_FUNCTION;
}
const char *
default_invalid_within_doloop (const rtx_insn *insn)
{
if (CALL_P (insn))
return "Function call in loop.";
if (tablejump_p (insn, NULL, NULL) || computed_jump_p (insn))
return "Computed branch in the loop.";
return NULL;
}
tree
default_builtin_vectorized_function (unsigned int, tree, tree)
{
return NULL_TREE;
}
tree
default_builtin_md_vectorized_function (tree, tree, tree)
{
return NULL_TREE;
}
tree
default_builtin_vectorized_conversion (unsigned int code ATTRIBUTE_UNUSED,
tree dest_type ATTRIBUTE_UNUSED,
tree src_type ATTRIBUTE_UNUSED)
{
return NULL_TREE;
}
int
default_builtin_vectorization_cost (enum vect_cost_for_stmt type_of_cost,
tree vectype,
int misalign ATTRIBUTE_UNUSED)
{
switch (type_of_cost)
{
case scalar_stmt:
case scalar_load:
case scalar_store:
case vector_stmt:
case vector_load:
case vector_store:
case vec_to_scalar:
case scalar_to_vec:
case cond_branch_not_taken:
case vec_perm:
case vec_promote_demote:
return 1;
case unaligned_load:
case unaligned_store:
return 2;
case cond_branch_taken:
return 3;
case vec_construct:
return estimated_poly_value (TYPE_VECTOR_SUBPARTS (vectype)) - 1;
default:
gcc_unreachable ();
}
}
tree
default_builtin_reciprocal (tree)
{
return NULL_TREE;
}
bool
hook_bool_CUMULATIVE_ARGS_mode_tree_bool_false (
cumulative_args_t ca ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED,
const_tree type ATTRIBUTE_UNUSED, bool named ATTRIBUTE_UNUSED)
{
return false;
}
bool
hook_bool_CUMULATIVE_ARGS_mode_tree_bool_true (
cumulative_args_t ca ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED,
const_tree type ATTRIBUTE_UNUSED, bool named ATTRIBUTE_UNUSED)
{
return true;
}
int
hook_int_CUMULATIVE_ARGS_mode_tree_bool_0 (
cumulative_args_t ca ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED,
tree type ATTRIBUTE_UNUSED, bool named ATTRIBUTE_UNUSED)
{
return 0;
}
void
hook_void_CUMULATIVE_ARGS_tree (cumulative_args_t ca ATTRIBUTE_UNUSED,
tree ATTRIBUTE_UNUSED)
{
}
void
default_function_arg_advance (cumulative_args_t ca ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED,
const_tree type ATTRIBUTE_UNUSED,
bool named ATTRIBUTE_UNUSED)
{
gcc_unreachable ();
}
HOST_WIDE_INT
default_function_arg_offset (machine_mode, const_tree)
{
return 0;
}
pad_direction
default_function_arg_padding (machine_mode mode, const_tree type)
{
if (!BYTES_BIG_ENDIAN)
return PAD_UPWARD;
unsigned HOST_WIDE_INT size;
if (mode == BLKmode)
{
if (!type || TREE_CODE (TYPE_SIZE (type)) != INTEGER_CST)
return PAD_UPWARD;
size = int_size_in_bytes (type);
}
else
size = GET_MODE_SIZE (mode).to_constant ();
if (size < (PARM_BOUNDARY / BITS_PER_UNIT))
return PAD_DOWNWARD;
return PAD_UPWARD;
}
rtx
default_function_arg (cumulative_args_t ca ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED,
const_tree type ATTRIBUTE_UNUSED,
bool named ATTRIBUTE_UNUSED)
{
gcc_unreachable ();
}
rtx
default_function_incoming_arg (cumulative_args_t ca ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED,
const_tree type ATTRIBUTE_UNUSED,
bool named ATTRIBUTE_UNUSED)
{
gcc_unreachable ();
}
unsigned int
default_function_arg_boundary (machine_mode mode ATTRIBUTE_UNUSED,
const_tree type ATTRIBUTE_UNUSED)
{
return PARM_BOUNDARY;
}
unsigned int
default_function_arg_round_boundary (machine_mode mode ATTRIBUTE_UNUSED,
const_tree type ATTRIBUTE_UNUSED)
{
return PARM_BOUNDARY;
}
void
hook_void_bitmap (bitmap regs ATTRIBUTE_UNUSED)
{
}
const char *
hook_invalid_arg_for_unprototyped_fn (
const_tree typelist ATTRIBUTE_UNUSED,
const_tree funcdecl ATTRIBUTE_UNUSED,
const_tree val ATTRIBUTE_UNUSED)
{
return NULL;
}
static GTY(()) tree stack_chk_guard_decl;
tree
default_stack_protect_guard (void)
{
tree t = stack_chk_guard_decl;
if (t == NULL)
{
rtx x;
t = build_decl (UNKNOWN_LOCATION,
VAR_DECL, get_identifier ("__stack_chk_guard"),
ptr_type_node);
TREE_STATIC (t) = 1;
TREE_PUBLIC (t) = 1;
DECL_EXTERNAL (t) = 1;
TREE_USED (t) = 1;
TREE_THIS_VOLATILE (t) = 1;
DECL_ARTIFICIAL (t) = 1;
DECL_IGNORED_P (t) = 1;
x = DECL_RTL (t);
RTX_FLAG (x, used) = 1;
stack_chk_guard_decl = t;
}
return t;
}
static GTY(()) tree stack_chk_fail_decl;
tree
default_external_stack_protect_fail (void)
{
tree t = stack_chk_fail_decl;
if (t == NULL_TREE)
{
t = build_function_type_list (void_type_node, NULL_TREE);
t = build_decl (UNKNOWN_LOCATION,
FUNCTION_DECL, get_identifier ("__stack_chk_fail"), t);
TREE_STATIC (t) = 1;
TREE_PUBLIC (t) = 1;
DECL_EXTERNAL (t) = 1;
TREE_USED (t) = 1;
TREE_THIS_VOLATILE (t) = 1;
TREE_NOTHROW (t) = 1;
DECL_ARTIFICIAL (t) = 1;
DECL_IGNORED_P (t) = 1;
DECL_VISIBILITY (t) = VISIBILITY_DEFAULT;
DECL_VISIBILITY_SPECIFIED (t) = 1;
stack_chk_fail_decl = t;
}
return build_call_expr (t, 0);
}
tree
default_hidden_stack_protect_fail (void)
{
#ifndef HAVE_GAS_HIDDEN
return default_external_stack_protect_fail ();
#else
tree t = stack_chk_fail_decl;
if (!flag_pic)
return default_external_stack_protect_fail ();
if (t == NULL_TREE)
{
t = build_function_type_list (void_type_node, NULL_TREE);
t = build_decl (UNKNOWN_LOCATION, FUNCTION_DECL,
get_identifier ("__stack_chk_fail_local"), t);
TREE_STATIC (t) = 1;
TREE_PUBLIC (t) = 1;
DECL_EXTERNAL (t) = 1;
TREE_USED (t) = 1;
TREE_THIS_VOLATILE (t) = 1;
TREE_NOTHROW (t) = 1;
DECL_ARTIFICIAL (t) = 1;
DECL_IGNORED_P (t) = 1;
DECL_VISIBILITY_SPECIFIED (t) = 1;
DECL_VISIBILITY (t) = VISIBILITY_HIDDEN;
stack_chk_fail_decl = t;
}
return build_call_expr (t, 0);
#endif
}
bool
hook_bool_const_rtx_commutative_p (const_rtx x,
int outer_code ATTRIBUTE_UNUSED)
{
return COMMUTATIVE_P (x);
}
rtx
default_function_value (const_tree ret_type ATTRIBUTE_UNUSED,
const_tree fn_decl_or_type,
bool outgoing ATTRIBUTE_UNUSED)
{
if (fn_decl_or_type
&& !DECL_P (fn_decl_or_type))
fn_decl_or_type = NULL;
#ifdef FUNCTION_VALUE
return FUNCTION_VALUE (ret_type, fn_decl_or_type);
#else
gcc_unreachable ();
#endif
}
rtx
default_libcall_value (machine_mode mode ATTRIBUTE_UNUSED,
const_rtx fun ATTRIBUTE_UNUSED)
{
#ifdef LIBCALL_VALUE
return LIBCALL_VALUE (MACRO_MODE (mode));
#else
gcc_unreachable ();
#endif
}
bool
default_function_value_regno_p (const unsigned int regno ATTRIBUTE_UNUSED)
{
#ifdef FUNCTION_VALUE_REGNO_P
return FUNCTION_VALUE_REGNO_P (regno);
#else
gcc_unreachable ();
#endif
}
rtx
default_internal_arg_pointer (void)
{
if ((ARG_POINTER_REGNUM == STACK_POINTER_REGNUM
|| ! (fixed_regs[ARG_POINTER_REGNUM]
|| ARG_POINTER_REGNUM == FRAME_POINTER_REGNUM)))
return copy_to_reg (virtual_incoming_args_rtx);
else
return virtual_incoming_args_rtx;
}
rtx
default_static_chain (const_tree ARG_UNUSED (fndecl_or_type), bool incoming_p)
{
if (incoming_p)
{
#ifdef STATIC_CHAIN_INCOMING_REGNUM
return gen_rtx_REG (Pmode, STATIC_CHAIN_INCOMING_REGNUM);
#endif
}
#ifdef STATIC_CHAIN_REGNUM
return gen_rtx_REG (Pmode, STATIC_CHAIN_REGNUM);
#endif
{
static bool issued_error;
if (!issued_error)
{
issued_error = true;
sorry ("nested functions not supported on this target");
}
return gen_rtx_MEM (Pmode, stack_pointer_rtx);
}
}
void
default_trampoline_init (rtx ARG_UNUSED (m_tramp), tree ARG_UNUSED (t_func),
rtx ARG_UNUSED (r_chain))
{
sorry ("nested function trampolines not supported on this target");
}
poly_int64
default_return_pops_args (tree, tree, poly_int64)
{
return 0;
}
reg_class_t
default_branch_target_register_class (void)
{
return NO_REGS;
}
reg_class_t
default_ira_change_pseudo_allocno_class (int regno ATTRIBUTE_UNUSED,
reg_class_t cl,
reg_class_t best_cl ATTRIBUTE_UNUSED)
{
return cl;
}
extern bool
default_lra_p (void)
{
return true;
}
int
default_register_priority (int hard_regno ATTRIBUTE_UNUSED)
{
return 0;
}
extern bool
default_register_usage_leveling_p (void)
{
return false;
}
extern bool
default_different_addr_displacement_p (void)
{
return false;
}
reg_class_t
default_secondary_reload (bool in_p ATTRIBUTE_UNUSED, rtx x ATTRIBUTE_UNUSED,
reg_class_t reload_class_i ATTRIBUTE_UNUSED,
machine_mode reload_mode ATTRIBUTE_UNUSED,
secondary_reload_info *sri)
{
enum reg_class rclass = NO_REGS;
enum reg_class reload_class = (enum reg_class) reload_class_i;
if (sri->prev_sri && sri->prev_sri->t_icode != CODE_FOR_nothing)
{
sri->icode = sri->prev_sri->t_icode;
return NO_REGS;
}
#ifdef SECONDARY_INPUT_RELOAD_CLASS
if (in_p)
rclass = SECONDARY_INPUT_RELOAD_CLASS (reload_class,
MACRO_MODE (reload_mode), x);
#endif
#ifdef SECONDARY_OUTPUT_RELOAD_CLASS
if (! in_p)
rclass = SECONDARY_OUTPUT_RELOAD_CLASS (reload_class,
MACRO_MODE (reload_mode), x);
#endif
if (rclass != NO_REGS)
{
enum insn_code icode
= direct_optab_handler (in_p ? reload_in_optab : reload_out_optab,
reload_mode);
if (icode != CODE_FOR_nothing
&& !insn_operand_matches (icode, in_p, x))
icode = CODE_FOR_nothing;
else if (icode != CODE_FOR_nothing)
{
const char *insn_constraint, *scratch_constraint;
enum reg_class insn_class, scratch_class;
gcc_assert (insn_data[(int) icode].n_operands == 3);
insn_constraint = insn_data[(int) icode].operand[!in_p].constraint;
if (!*insn_constraint)
insn_class = ALL_REGS;
else
{
if (in_p)
{
gcc_assert (*insn_constraint == '=');
insn_constraint++;
}
insn_class = (reg_class_for_constraint
(lookup_constraint (insn_constraint)));
gcc_assert (insn_class != NO_REGS);
}
scratch_constraint = insn_data[(int) icode].operand[2].constraint;
gcc_assert (scratch_constraint[0] == '='
&& (in_p || scratch_constraint[1] == '&'));
scratch_constraint++;
if (*scratch_constraint == '&')
scratch_constraint++;
scratch_class = (reg_class_for_constraint
(lookup_constraint (scratch_constraint)));
if (reg_class_subset_p (reload_class, insn_class))
{
gcc_assert (scratch_class == rclass);
rclass = NO_REGS;
}
else
rclass = insn_class;
}
if (rclass == NO_REGS)
sri->icode = icode;
else
sri->t_icode = icode;
}
return rclass;
}
machine_mode
default_secondary_memory_needed_mode (machine_mode mode)
{
if (!targetm.lra_p ()
&& known_lt (GET_MODE_BITSIZE (mode), BITS_PER_WORD)
&& INTEGRAL_MODE_P (mode))
return mode_for_size (BITS_PER_WORD, GET_MODE_CLASS (mode), 0).require ();
return mode;
}
int
default_reloc_rw_mask (void)
{
return flag_pic ? 3 : 0;
}
tree default_mangle_decl_assembler_name (tree decl ATTRIBUTE_UNUSED,
tree id)
{
return id;
}
HOST_WIDE_INT
default_static_rtx_alignment (machine_mode mode)
{
return GET_MODE_ALIGNMENT (mode);
}
HOST_WIDE_INT
default_constant_alignment (const_tree, HOST_WIDE_INT align)
{
return align;
}
HOST_WIDE_INT
constant_alignment_word_strings (const_tree exp, HOST_WIDE_INT align)
{
if (TREE_CODE (exp) == STRING_CST)
return MAX (align, BITS_PER_WORD);
return align;
}
HOST_WIDE_INT
default_vector_alignment (const_tree type)
{
HOST_WIDE_INT align = tree_to_shwi (TYPE_SIZE (type));
if (align > MAX_OFILE_ALIGNMENT)
align = MAX_OFILE_ALIGNMENT;
return align;
}
HOST_WIDE_INT
default_preferred_vector_alignment (const_tree type)
{
return TYPE_ALIGN (type);
}
bool
default_builtin_vector_alignment_reachable (const_tree , bool is_packed)
{
return ! is_packed;
}
bool
default_builtin_support_vector_misalignment (machine_mode mode,
const_tree type
ATTRIBUTE_UNUSED,
int misalignment
ATTRIBUTE_UNUSED,
bool is_packed
ATTRIBUTE_UNUSED)
{
if (optab_handler (movmisalign_optab, mode) != CODE_FOR_nothing)
return true;
return false;
}
machine_mode
default_preferred_simd_mode (scalar_mode)
{
return word_mode;
}
machine_mode
default_split_reduction (machine_mode mode)
{
return mode;
}
void
default_autovectorize_vector_sizes (vector_sizes *)
{
}
opt_machine_mode
default_get_mask_mode (poly_uint64 nunits, poly_uint64 vector_size)
{
unsigned int elem_size = vector_element_size (vector_size, nunits);
scalar_int_mode elem_mode
= smallest_int_mode_for_size (elem_size * BITS_PER_UNIT);
machine_mode vector_mode;
gcc_assert (known_eq (elem_size * nunits, vector_size));
if (mode_for_vector (elem_mode, nunits).exists (&vector_mode)
&& VECTOR_MODE_P (vector_mode)
&& targetm.vector_mode_supported_p (vector_mode))
return vector_mode;
return opt_machine_mode ();
}
bool
default_empty_mask_is_expensive (unsigned ifn)
{
return ifn == IFN_MASK_STORE;
}
void *
default_init_cost (struct loop *loop_info ATTRIBUTE_UNUSED)
{
unsigned *cost = XNEWVEC (unsigned, 3);
cost[vect_prologue] = cost[vect_body] = cost[vect_epilogue] = 0;
return cost;
}
unsigned
default_add_stmt_cost (void *data, int count, enum vect_cost_for_stmt kind,
struct _stmt_vec_info *stmt_info, int misalign,
enum vect_cost_model_location where)
{
unsigned *cost = (unsigned *) data;
unsigned retval = 0;
tree vectype = stmt_info ? stmt_vectype (stmt_info) : NULL_TREE;
int stmt_cost = targetm.vectorize.builtin_vectorization_cost (kind, vectype,
misalign);
if (where == vect_body && stmt_info && stmt_in_inner_loop_p (stmt_info))
count *= 50;  
retval = (unsigned) (count * stmt_cost);
cost[where] += retval;
return retval;
}
void
default_finish_cost (void *data, unsigned *prologue_cost,
unsigned *body_cost, unsigned *epilogue_cost)
{
unsigned *cost = (unsigned *) data;
*prologue_cost = cost[vect_prologue];
*body_cost     = cost[vect_body];
*epilogue_cost = cost[vect_epilogue];
}
void
default_destroy_cost_data (void *data)
{
free (data);
}
bool
default_valid_pointer_mode (scalar_int_mode mode)
{
return (mode == ptr_mode || mode == Pmode);
}
bool
default_ref_may_alias_errno (ao_ref *ref)
{
tree base = ao_ref_base (ref);
if (TYPE_UNSIGNED (TREE_TYPE (base))
|| TYPE_MODE (TREE_TYPE (base)) != TYPE_MODE (integer_type_node))
return false;
if (DECL_P (base)
&& !TREE_STATIC (base))
return true;
else if (TREE_CODE (base) == MEM_REF
&& TREE_CODE (TREE_OPERAND (base, 0)) == SSA_NAME)
{
struct ptr_info_def *pi = SSA_NAME_PTR_INFO (TREE_OPERAND (base, 0));
return !pi || pi->pt.anything || pi->pt.nonlocal;
}
return false;
}
scalar_int_mode
default_addr_space_pointer_mode (addr_space_t addrspace ATTRIBUTE_UNUSED)
{
return ptr_mode;
}
scalar_int_mode
default_addr_space_address_mode (addr_space_t addrspace ATTRIBUTE_UNUSED)
{
return Pmode;
}
bool
default_addr_space_valid_pointer_mode (scalar_int_mode mode,
addr_space_t as ATTRIBUTE_UNUSED)
{
return targetm.valid_pointer_mode (mode);
}
bool
target_default_pointer_address_modes_p (void)
{
if (targetm.addr_space.address_mode != default_addr_space_address_mode)
return false;
if (targetm.addr_space.pointer_mode != default_addr_space_pointer_mode)
return false;
return true;
}
bool
default_addr_space_legitimate_address_p (machine_mode mode, rtx mem,
bool strict,
addr_space_t as ATTRIBUTE_UNUSED)
{
return targetm.legitimate_address_p (mode, mem, strict);
}
rtx
default_addr_space_legitimize_address (rtx x, rtx oldx, machine_mode mode,
addr_space_t as ATTRIBUTE_UNUSED)
{
return targetm.legitimize_address (x, oldx, mode);
}
bool
default_addr_space_subset_p (addr_space_t subset, addr_space_t superset)
{
return (subset == superset);
}
bool
default_addr_space_zero_address_valid (addr_space_t as ATTRIBUTE_UNUSED)
{
return false;
}
int
default_addr_space_debug (addr_space_t as)
{
return as;
}
void
default_addr_space_diagnose_usage (addr_space_t, location_t)
{
}
rtx
default_addr_space_convert (rtx op ATTRIBUTE_UNUSED,
tree from_type ATTRIBUTE_UNUSED,
tree to_type ATTRIBUTE_UNUSED)
{
gcc_unreachable ();
}
unsigned int
default_hard_regno_nregs (unsigned int, machine_mode mode)
{
return CEIL (GET_MODE_SIZE (mode).to_constant (), UNITS_PER_WORD);
}
bool
default_hard_regno_scratch_ok (unsigned int regno ATTRIBUTE_UNUSED)
{
return true;
}
bool
default_mode_dependent_address_p (const_rtx addr ATTRIBUTE_UNUSED,
addr_space_t addrspace ATTRIBUTE_UNUSED)
{
return false;
}
bool
default_target_option_valid_attribute_p (tree ARG_UNUSED (fndecl),
tree ARG_UNUSED (name),
tree ARG_UNUSED (args),
int ARG_UNUSED (flags))
{
warning (OPT_Wattributes,
"target attribute is not supported on this machine");
return false;
}
bool
default_target_option_pragma_parse (tree ARG_UNUSED (args),
tree ARG_UNUSED (pop_target))
{
if (args)
warning (OPT_Wpragmas,
"#pragma GCC target is not supported for this machine");
return false;
}
bool
default_target_can_inline_p (tree caller, tree callee)
{
tree callee_opts = DECL_FUNCTION_SPECIFIC_TARGET (callee);
tree caller_opts = DECL_FUNCTION_SPECIFIC_TARGET (caller);
if (! callee_opts)
callee_opts = target_option_default_node;
if (! caller_opts)
caller_opts = target_option_default_node;
return callee_opts == caller_opts;
}
unsigned int
default_case_values_threshold (void)
{
return (targetm.have_casesi () ? 4 : 5);
}
bool
default_have_conditional_execution (void)
{
return HAVE_conditional_execution;
}
bool
default_libc_has_function (enum function_class fn_class)
{
if (fn_class == function_c94
|| fn_class == function_c99_misc
|| fn_class == function_c99_math_complex)
return true;
return false;
}
bool
gnu_libc_has_function (enum function_class fn_class ATTRIBUTE_UNUSED)
{
return true;
}
bool
no_c99_libc_has_function (enum function_class fn_class ATTRIBUTE_UNUSED)
{
return false;
}
tree
default_builtin_tm_load_store (tree ARG_UNUSED (type))
{
return NULL_TREE;
}
int
default_memory_move_cost (machine_mode mode ATTRIBUTE_UNUSED,
reg_class_t rclass ATTRIBUTE_UNUSED,
bool in ATTRIBUTE_UNUSED)
{
#ifndef MEMORY_MOVE_COST
return (4 + memory_move_secondary_cost (mode, (enum reg_class) rclass, in));
#else
return MEMORY_MOVE_COST (MACRO_MODE (mode), (enum reg_class) rclass, in);
#endif
}
int
default_register_move_cost (machine_mode mode ATTRIBUTE_UNUSED,
reg_class_t from ATTRIBUTE_UNUSED,
reg_class_t to ATTRIBUTE_UNUSED)
{
#ifndef REGISTER_MOVE_COST
return 2;
#else
return REGISTER_MOVE_COST (MACRO_MODE (mode),
(enum reg_class) from, (enum reg_class) to);
#endif
}
bool
default_slow_unaligned_access (machine_mode, unsigned int)
{
return STRICT_ALIGNMENT;
}
HOST_WIDE_INT
default_estimated_poly_value (poly_int64 x)
{
return x.coeffs[0];
}
unsigned int
get_move_ratio (bool speed_p ATTRIBUTE_UNUSED)
{
unsigned int move_ratio;
#ifdef MOVE_RATIO
move_ratio = (unsigned int) MOVE_RATIO (speed_p);
#else
#if defined (HAVE_movmemqi) || defined (HAVE_movmemhi) || defined (HAVE_movmemsi) || defined (HAVE_movmemdi) || defined (HAVE_movmemti)
move_ratio = 2;
#else 
move_ratio = ((speed_p) ? 15 : 3);
#endif
#endif
return move_ratio;
}
bool
default_use_by_pieces_infrastructure_p (unsigned HOST_WIDE_INT size,
unsigned int alignment,
enum by_pieces_operation op,
bool speed_p)
{
unsigned int max_size = 0;
unsigned int ratio = 0;
switch (op)
{
case CLEAR_BY_PIECES:
max_size = STORE_MAX_PIECES;
ratio = CLEAR_RATIO (speed_p);
break;
case MOVE_BY_PIECES:
max_size = MOVE_MAX_PIECES;
ratio = get_move_ratio (speed_p);
break;
case SET_BY_PIECES:
max_size = STORE_MAX_PIECES;
ratio = SET_RATIO (speed_p);
break;
case STORE_BY_PIECES:
max_size = STORE_MAX_PIECES;
ratio = get_move_ratio (speed_p);
break;
case COMPARE_BY_PIECES:
max_size = COMPARE_MAX_PIECES;
ratio = speed_p ? 15 : 3;
break;
}
return by_pieces_ninsns (size, alignment, max_size + 1, op) < ratio;
}
int
default_compare_by_pieces_branch_ratio (machine_mode)
{
return 1;
}
void
default_print_patchable_function_entry (FILE *file,
unsigned HOST_WIDE_INT patch_area_size,
bool record_p)
{
const char *nop_templ = 0;
int code_num;
rtx_insn *my_nop = make_insn_raw (gen_nop ());
code_num = recog_memoized (my_nop);
nop_templ = get_insn_template (code_num, my_nop);
if (record_p && targetm_common.have_named_sections)
{
char buf[256];
static int patch_area_number;
section *previous_section = in_section;
patch_area_number++;
ASM_GENERATE_INTERNAL_LABEL (buf, "LPFE", patch_area_number);
switch_to_section (get_section ("__patchable_function_entries",
0, NULL));
fputs (integer_asm_op (POINTER_SIZE_UNITS, false), file);
assemble_name_raw (file, buf);
fputc ('\n', file);
switch_to_section (previous_section);
ASM_OUTPUT_LABEL (file, buf);
}
unsigned i;
for (i = 0; i < patch_area_size; ++i)
fprintf (file, "\t%s\n", nop_templ);
}
bool
default_profile_before_prologue (void)
{
#ifdef PROFILE_BEFORE_PROLOGUE
return true;
#else
return false;
#endif
}
reg_class_t
default_preferred_reload_class (rtx x ATTRIBUTE_UNUSED,
reg_class_t rclass)
{
#ifdef PREFERRED_RELOAD_CLASS 
return (reg_class_t) PREFERRED_RELOAD_CLASS (x, (enum reg_class) rclass);
#else
return rclass;
#endif
}
reg_class_t
default_preferred_output_reload_class (rtx x ATTRIBUTE_UNUSED,
reg_class_t rclass)
{
return rclass;
}
reg_class_t
default_preferred_rename_class (reg_class_t rclass ATTRIBUTE_UNUSED)
{
return NO_REGS;
}
bool
default_class_likely_spilled_p (reg_class_t rclass)
{
return (reg_class_size[(int) rclass] == 1);
}
unsigned char
default_class_max_nregs (reg_class_t rclass ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED)
{
#ifdef CLASS_MAX_NREGS
return (unsigned char) CLASS_MAX_NREGS ((enum reg_class) rclass,
MACRO_MODE (mode));
#else
unsigned int size = GET_MODE_SIZE (mode).to_constant ();
return (size + UNITS_PER_WORD - 1) / UNITS_PER_WORD;
#endif
}
enum unwind_info_type
default_debug_unwind_info (void)
{
#ifdef DWARF2_FRAME_INFO
if (DWARF2_FRAME_INFO)
return UI_DWARF2;
#endif
#ifdef DWARF2_DEBUGGING_INFO
if (write_symbols == DWARF2_DEBUG || write_symbols == VMS_AND_DWARF2_DEBUG)
return UI_DWARF2;
#endif
return UI_NONE;
}
unsigned int
default_dwarf_poly_indeterminate_value (unsigned int, unsigned int *, int *)
{
gcc_unreachable ();
}
machine_mode
default_dwarf_frame_reg_mode (int regno)
{
machine_mode save_mode = reg_raw_mode[regno];
if (targetm.hard_regno_call_part_clobbered (regno, save_mode))
save_mode = choose_hard_reg_mode (regno, 1, true);
return save_mode;
}
fixed_size_mode
default_get_reg_raw_mode (int regno)
{
return as_a <fixed_size_mode> (reg_raw_mode[regno]);
}
bool
default_keep_leaf_when_profiled ()
{
return false;
}
static inline bool
option_affects_pch_p (int option, struct cl_option_state *state)
{
if ((cl_options[option].flags & CL_TARGET) == 0)
return false;
if ((cl_options[option].flags & CL_PCH_IGNORE) != 0)
return false;
if (option_flag_var (option, &global_options) == &target_flags)
if (targetm.check_pch_target_flags)
return false;
return get_option_state (&global_options, option, state);
}
void *
default_get_pch_validity (size_t *sz)
{
struct cl_option_state state;
size_t i;
char *result, *r;
*sz = 2;
if (targetm.check_pch_target_flags)
*sz += sizeof (target_flags);
for (i = 0; i < cl_options_count; i++)
if (option_affects_pch_p (i, &state))
*sz += state.size;
result = r = XNEWVEC (char, *sz);
r[0] = flag_pic;
r[1] = flag_pie;
r += 2;
if (targetm.check_pch_target_flags)
{
memcpy (r, &target_flags, sizeof (target_flags));
r += sizeof (target_flags);
}
for (i = 0; i < cl_options_count; i++)
if (option_affects_pch_p (i, &state))
{
memcpy (r, state.data, state.size);
r += state.size;
}
return result;
}
static const char *
pch_option_mismatch (const char *option)
{
return xasprintf (_("created and used with differing settings of '%s'"),
option);
}
const char *
default_pch_valid_p (const void *data_p, size_t len)
{
struct cl_option_state state;
const char *data = (const char *)data_p;
size_t i;
if (data[0] != flag_pic)
return _("created and used with different settings of -fpic");
if (data[1] != flag_pie)
return _("created and used with different settings of -fpie");
data += 2;
if (targetm.check_pch_target_flags)
{
int tf;
const char *r;
memcpy (&tf, data, sizeof (target_flags));
data += sizeof (target_flags);
len -= sizeof (target_flags);
r = targetm.check_pch_target_flags (tf);
if (r != NULL)
return r;
}
for (i = 0; i < cl_options_count; i++)
if (option_affects_pch_p (i, &state))
{
if (memcmp (data, state.data, state.size) != 0)
return pch_option_mismatch (cl_options[i].opt_text);
data += state.size;
len -= state.size;
}
return NULL;
}
scalar_int_mode
default_cstore_mode (enum insn_code icode)
{
return as_a <scalar_int_mode> (insn_data[(int) icode].operand[0].mode);
}
bool
default_member_type_forces_blk (const_tree, machine_mode)
{
return false;
}
rtx
default_load_bounds_for_arg (rtx addr ATTRIBUTE_UNUSED,
rtx ptr ATTRIBUTE_UNUSED,
rtx bnd ATTRIBUTE_UNUSED)
{
gcc_unreachable ();
}
void
default_store_bounds_for_arg (rtx val ATTRIBUTE_UNUSED,
rtx addr ATTRIBUTE_UNUSED,
rtx bounds ATTRIBUTE_UNUSED,
rtx to ATTRIBUTE_UNUSED)
{
gcc_unreachable ();
}
rtx
default_load_returned_bounds (rtx slot ATTRIBUTE_UNUSED)
{
gcc_unreachable ();
}
void
default_store_returned_bounds (rtx slot ATTRIBUTE_UNUSED,
rtx bounds ATTRIBUTE_UNUSED)
{
gcc_unreachable ();
}
void
default_canonicalize_comparison (int *, rtx *, rtx *, bool)
{
}
void
default_atomic_assign_expand_fenv (tree *, tree *, tree *)
{
}
#ifndef PAD_VARARGS_DOWN
#define PAD_VARARGS_DOWN BYTES_BIG_ENDIAN
#endif
tree
build_va_arg_indirect_ref (tree addr)
{
addr = build_simple_mem_ref_loc (EXPR_LOCATION (addr), addr);
return addr;
}
tree
std_gimplify_va_arg_expr (tree valist, tree type, gimple_seq *pre_p,
gimple_seq *post_p)
{
tree addr, t, type_size, rounded_size, valist_tmp;
unsigned HOST_WIDE_INT align, boundary;
bool indirect;
if (ARGS_GROW_DOWNWARD)
gcc_unreachable ();
indirect = pass_by_reference (NULL, TYPE_MODE (type), type, false);
if (indirect)
type = build_pointer_type (type);
align = PARM_BOUNDARY / BITS_PER_UNIT;
boundary = targetm.calls.function_arg_boundary (TYPE_MODE (type), type);
if (boundary > MAX_SUPPORTED_STACK_ALIGNMENT)
boundary = MAX_SUPPORTED_STACK_ALIGNMENT;
boundary /= BITS_PER_UNIT;
valist_tmp = get_initialized_tmp_var (valist, pre_p, NULL);
if (boundary > align
&& !TYPE_EMPTY_P (type)
&& !integer_zerop (TYPE_SIZE (type)))
{
t = build2 (MODIFY_EXPR, TREE_TYPE (valist), valist_tmp,
fold_build_pointer_plus_hwi (valist_tmp, boundary - 1));
gimplify_and_add (t, pre_p);
t = build2 (MODIFY_EXPR, TREE_TYPE (valist), valist_tmp,
fold_build2 (BIT_AND_EXPR, TREE_TYPE (valist),
valist_tmp,
build_int_cst (TREE_TYPE (valist), -boundary)));
gimplify_and_add (t, pre_p);
}
else
boundary = align;
boundary *= BITS_PER_UNIT;
if (boundary < TYPE_ALIGN (type))
{
type = build_variant_type_copy (type);
SET_TYPE_ALIGN (type, boundary);
}
type_size = arg_size_in_bytes (type);
rounded_size = round_up (type_size, align);
gimplify_expr (&rounded_size, pre_p, post_p, is_gimple_val, fb_rvalue);
addr = valist_tmp;
if (PAD_VARARGS_DOWN && !integer_zerop (rounded_size))
{
t = fold_build2_loc (input_location, GT_EXPR, sizetype,
rounded_size, size_int (align));
t = fold_build3 (COND_EXPR, sizetype, t, size_zero_node,
size_binop (MINUS_EXPR, rounded_size, type_size));
addr = fold_build_pointer_plus (addr, t);
}
t = fold_build_pointer_plus (valist_tmp, rounded_size);
t = build2 (MODIFY_EXPR, TREE_TYPE (valist), valist, t);
gimplify_and_add (t, pre_p);
addr = fold_convert (build_pointer_type (type), addr);
if (indirect)
addr = build_va_arg_indirect_ref (addr);
return build_va_arg_indirect_ref (addr);
}
tree
default_chkp_bound_type (void)
{
tree res = make_node (POINTER_BOUNDS_TYPE);
TYPE_PRECISION (res) = TYPE_PRECISION (size_type_node) * 2;
TYPE_NAME (res) = get_identifier ("__bounds_type");
SET_TYPE_MODE (res, targetm.chkp_bound_mode ());
layout_type (res);
return res;
}
machine_mode
default_chkp_bound_mode (void)
{
return VOIDmode;
}
tree
default_builtin_chkp_function (unsigned int fcode ATTRIBUTE_UNUSED)
{
return NULL_TREE;
}
rtx
default_chkp_function_value_bounds (const_tree ret_type ATTRIBUTE_UNUSED,
const_tree fn_decl_or_type ATTRIBUTE_UNUSED,
bool outgoing ATTRIBUTE_UNUSED)
{
gcc_unreachable ();
}
tree
default_chkp_make_bounds_constant (HOST_WIDE_INT lb ATTRIBUTE_UNUSED,
HOST_WIDE_INT ub ATTRIBUTE_UNUSED)
{
return NULL_TREE;
}
int
default_chkp_initialize_bounds (tree var ATTRIBUTE_UNUSED,
tree lb ATTRIBUTE_UNUSED,
tree ub ATTRIBUTE_UNUSED,
tree *stmts ATTRIBUTE_UNUSED)
{
return 0;
}
void
default_setup_incoming_vararg_bounds (cumulative_args_t ca ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED,
tree type ATTRIBUTE_UNUSED,
int *pretend_arg_size ATTRIBUTE_UNUSED,
int second_time ATTRIBUTE_UNUSED)
{
}
bool
can_use_doloop_if_innermost (const widest_int &, const widest_int &,
unsigned int loop_depth, bool)
{
return loop_depth == 1;
}
bool
default_optab_supported_p (int, machine_mode, machine_mode, optimization_type)
{
return true;
}
unsigned int
default_max_noce_ifcvt_seq_cost (edge e)
{
bool predictable_p = predictable_edge_p (e);
enum compiler_param param
= (predictable_p
? PARAM_MAX_RTL_IF_CONVERSION_PREDICTABLE_COST
: PARAM_MAX_RTL_IF_CONVERSION_UNPREDICTABLE_COST);
if (global_options_set.x_param_values[param])
return PARAM_VALUE (param);
else
return BRANCH_COST (true, predictable_p) * COSTS_N_INSNS (3);
}
unsigned int
default_min_arithmetic_precision (void)
{
return WORD_REGISTER_OPERATIONS ? BITS_PER_WORD : BITS_PER_UNIT;
}
enum flt_eval_method
default_excess_precision (enum excess_precision_type ATTRIBUTE_UNUSED)
{
return FLT_EVAL_METHOD_PROMOTE_TO_FLOAT;
}
bool
default_stack_clash_protection_final_dynamic_probe (rtx residual ATTRIBUTE_UNUSED)
{
return 0;
}
void
default_select_early_remat_modes (sbitmap)
{
}
#include "gt-targhooks.h"
