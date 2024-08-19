#define IN_TARGET_CODE 1
#include "config.h"
#define INCLUDE_STRING
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "target.h"
#include "rtl.h"
#include "tree.h"
#include "memmodel.h"
#include "cfghooks.h"
#include "df.h"
#include "tm_p.h"
#include "stringpool.h"
#include "attribs.h"
#include "optabs.h"
#include "regs.h"
#include "emit-rtl.h"
#include "recog.h"
#include "cgraph.h"
#include "diagnostic-core.h"
#include "alias.h"
#include "fold-const.h"
#include "stor-layout.h"
#include "calls.h"
#include "varasm.h"
#include "output.h"
#include "insn-attr.h"
#include "flags.h"
#include "reload.h"
#include "explow.h"
#include "expr.h"
#include "cfgrtl.h"
#include "sched-int.h"
#include "common/common-target.h"
#include "langhooks.h"
#include "intl.h"
#include "libfuncs.h"
#include "params.h"
#include "opts.h"
#include "dumpfile.h"
#include "target-globals.h"
#include "builtins.h"
#include "tm-constrs.h"
#include "rtl-iter.h"
#include "optabs-libfuncs.h"
#include "gimplify.h"
#include "gimple.h"
#include "selftest.h"
#include "target-def.h"
typedef struct minipool_node    Mnode;
typedef struct minipool_fixup   Mfix;
static std::string arm_last_printed_arch_string;
static std::string arm_last_printed_fpu_string;
void (*arm_lang_output_object_attributes_hook)(void);
struct four_ints
{
int i[4];
};
static bool arm_const_not_ok_for_debug_p (rtx);
static int arm_needs_doubleword_align (machine_mode, const_tree);
static int arm_compute_static_chain_stack_bytes (void);
static arm_stack_offsets *arm_get_frame_offsets (void);
static void arm_compute_frame_layout (void);
static void arm_add_gc_roots (void);
static int arm_gen_constant (enum rtx_code, machine_mode, rtx,
unsigned HOST_WIDE_INT, rtx, rtx, int, int);
static unsigned bit_count (unsigned long);
static unsigned bitmap_popcount (const sbitmap);
static int arm_address_register_rtx_p (rtx, int);
static int arm_legitimate_index_p (machine_mode, rtx, RTX_CODE, int);
static bool is_called_in_ARM_mode (tree);
static int thumb2_legitimate_index_p (machine_mode, rtx, int);
static int thumb1_base_register_rtx_p (rtx, machine_mode, int);
static rtx arm_legitimize_address (rtx, rtx, machine_mode);
static reg_class_t arm_preferred_reload_class (rtx, reg_class_t);
static rtx thumb_legitimize_address (rtx, rtx, machine_mode);
inline static int thumb1_index_register_rtx_p (rtx, int);
static int thumb_far_jump_used_p (void);
static bool thumb_force_lr_save (void);
static unsigned arm_size_return_regs (void);
static bool arm_assemble_integer (rtx, unsigned int, int);
static void arm_print_operand (FILE *, rtx, int);
static void arm_print_operand_address (FILE *, machine_mode, rtx);
static bool arm_print_operand_punct_valid_p (unsigned char code);
static const char *fp_const_from_val (REAL_VALUE_TYPE *);
static arm_cc get_arm_condition_code (rtx);
static bool arm_fixed_condition_code_regs (unsigned int *, unsigned int *);
static const char *output_multi_immediate (rtx *, const char *, const char *,
int, HOST_WIDE_INT);
static const char *shift_op (rtx, HOST_WIDE_INT *);
static struct machine_function *arm_init_machine_status (void);
static void thumb_exit (FILE *, int);
static HOST_WIDE_INT get_jump_table_size (rtx_jump_table_data *);
static Mnode *move_minipool_fix_forward_ref (Mnode *, Mnode *, HOST_WIDE_INT);
static Mnode *add_minipool_forward_ref (Mfix *);
static Mnode *move_minipool_fix_backward_ref (Mnode *, Mnode *, HOST_WIDE_INT);
static Mnode *add_minipool_backward_ref (Mfix *);
static void assign_minipool_offsets (Mfix *);
static void arm_print_value (FILE *, rtx);
static void dump_minipool (rtx_insn *);
static int arm_barrier_cost (rtx_insn *);
static Mfix *create_fix_barrier (Mfix *, HOST_WIDE_INT);
static void push_minipool_barrier (rtx_insn *, HOST_WIDE_INT);
static void push_minipool_fix (rtx_insn *, HOST_WIDE_INT, rtx *,
machine_mode, rtx);
static void arm_reorg (void);
static void note_invalid_constants (rtx_insn *, HOST_WIDE_INT, int);
static unsigned long arm_compute_save_reg0_reg12_mask (void);
static unsigned long arm_compute_save_core_reg_mask (void);
static unsigned long arm_isr_value (tree);
static unsigned long arm_compute_func_type (void);
static tree arm_handle_fndecl_attribute (tree *, tree, tree, int, bool *);
static tree arm_handle_pcs_attribute (tree *, tree, tree, int, bool *);
static tree arm_handle_isr_attribute (tree *, tree, tree, int, bool *);
#if TARGET_DLLIMPORT_DECL_ATTRIBUTES
static tree arm_handle_notshared_attribute (tree *, tree, tree, int, bool *);
#endif
static tree arm_handle_cmse_nonsecure_entry (tree *, tree, tree, int, bool *);
static tree arm_handle_cmse_nonsecure_call (tree *, tree, tree, int, bool *);
static void arm_output_function_epilogue (FILE *);
static void arm_output_function_prologue (FILE *);
static int arm_comp_type_attributes (const_tree, const_tree);
static void arm_set_default_type_attributes (tree);
static int arm_adjust_cost (rtx_insn *, int, rtx_insn *, int, unsigned int);
static int arm_sched_reorder (FILE *, int, rtx_insn **, int *, int);
static int optimal_immediate_sequence (enum rtx_code code,
unsigned HOST_WIDE_INT val,
struct four_ints *return_sequence);
static int optimal_immediate_sequence_1 (enum rtx_code code,
unsigned HOST_WIDE_INT val,
struct four_ints *return_sequence,
int i);
static int arm_get_strip_length (int);
static bool arm_function_ok_for_sibcall (tree, tree);
static machine_mode arm_promote_function_mode (const_tree,
machine_mode, int *,
const_tree, int);
static bool arm_return_in_memory (const_tree, const_tree);
static rtx arm_function_value (const_tree, const_tree, bool);
static rtx arm_libcall_value_1 (machine_mode);
static rtx arm_libcall_value (machine_mode, const_rtx);
static bool arm_function_value_regno_p (const unsigned int);
static void arm_internal_label (FILE *, const char *, unsigned long);
static void arm_output_mi_thunk (FILE *, tree, HOST_WIDE_INT, HOST_WIDE_INT,
tree);
static bool arm_have_conditional_execution (void);
static bool arm_cannot_force_const_mem (machine_mode, rtx);
static bool arm_legitimate_constant_p (machine_mode, rtx);
static bool arm_rtx_costs (rtx, machine_mode, int, int, int *, bool);
static int arm_address_cost (rtx, machine_mode, addr_space_t, bool);
static int arm_register_move_cost (machine_mode, reg_class_t, reg_class_t);
static int arm_memory_move_cost (machine_mode, reg_class_t, bool);
static void emit_constant_insn (rtx cond, rtx pattern);
static rtx_insn *emit_set_insn (rtx, rtx);
static rtx emit_multi_reg_push (unsigned long, unsigned long);
static int arm_arg_partial_bytes (cumulative_args_t, machine_mode,
tree, bool);
static rtx arm_function_arg (cumulative_args_t, machine_mode,
const_tree, bool);
static void arm_function_arg_advance (cumulative_args_t, machine_mode,
const_tree, bool);
static pad_direction arm_function_arg_padding (machine_mode, const_tree);
static unsigned int arm_function_arg_boundary (machine_mode, const_tree);
static rtx aapcs_allocate_return_reg (machine_mode, const_tree,
const_tree);
static rtx aapcs_libcall_value (machine_mode);
static int aapcs_select_return_coproc (const_tree, const_tree);
#ifdef OBJECT_FORMAT_ELF
static void arm_elf_asm_constructor (rtx, int) ATTRIBUTE_UNUSED;
static void arm_elf_asm_destructor (rtx, int) ATTRIBUTE_UNUSED;
#endif
#ifndef ARM_PE
static void arm_encode_section_info (tree, rtx, int);
#endif
static void arm_file_end (void);
static void arm_file_start (void);
static void arm_insert_attributes (tree, tree *);
static void arm_setup_incoming_varargs (cumulative_args_t, machine_mode,
tree, int *, int);
static bool arm_pass_by_reference (cumulative_args_t,
machine_mode, const_tree, bool);
static bool arm_promote_prototypes (const_tree);
static bool arm_default_short_enums (void);
static bool arm_align_anon_bitfield (void);
static bool arm_return_in_msb (const_tree);
static bool arm_must_pass_in_stack (machine_mode, const_tree);
static bool arm_return_in_memory (const_tree, const_tree);
#if ARM_UNWIND_INFO
static void arm_unwind_emit (FILE *, rtx_insn *);
static bool arm_output_ttype (rtx);
static void arm_asm_emit_except_personality (rtx);
#endif
static void arm_asm_init_sections (void);
static rtx arm_dwarf_register_span (rtx);
static tree arm_cxx_guard_type (void);
static bool arm_cxx_guard_mask_bit (void);
static tree arm_get_cookie_size (tree);
static bool arm_cookie_has_size (void);
static bool arm_cxx_cdtor_returns_this (void);
static bool arm_cxx_key_method_may_be_inline (void);
static void arm_cxx_determine_class_data_visibility (tree);
static bool arm_cxx_class_data_always_comdat (void);
static bool arm_cxx_use_aeabi_atexit (void);
static void arm_init_libfuncs (void);
static tree arm_build_builtin_va_list (void);
static void arm_expand_builtin_va_start (tree, rtx);
static tree arm_gimplify_va_arg_expr (tree, tree, gimple_seq *, gimple_seq *);
static void arm_option_override (void);
static void arm_option_save (struct cl_target_option *, struct gcc_options *);
static void arm_option_restore (struct gcc_options *,
struct cl_target_option *);
static void arm_override_options_after_change (void);
static void arm_option_print (FILE *, int, struct cl_target_option *);
static void arm_set_current_function (tree);
static bool arm_can_inline_p (tree, tree);
static void arm_relayout_function (tree);
static bool arm_valid_target_attribute_p (tree, tree, tree, int);
static unsigned HOST_WIDE_INT arm_shift_truncation_mask (machine_mode);
static bool arm_sched_can_speculate_insn (rtx_insn *);
static bool arm_macro_fusion_p (void);
static bool arm_cannot_copy_insn_p (rtx_insn *);
static int arm_issue_rate (void);
static int arm_first_cycle_multipass_dfa_lookahead (void);
static int arm_first_cycle_multipass_dfa_lookahead_guard (rtx_insn *, int);
static void arm_output_dwarf_dtprel (FILE *, int, rtx) ATTRIBUTE_UNUSED;
static bool arm_output_addr_const_extra (FILE *, rtx);
static bool arm_allocate_stack_slots_for_args (void);
static bool arm_warn_func_return (tree);
static tree arm_promoted_type (const_tree t);
static bool arm_scalar_mode_supported_p (scalar_mode);
static bool arm_frame_pointer_required (void);
static bool arm_can_eliminate (const int, const int);
static void arm_asm_trampoline_template (FILE *);
static void arm_trampoline_init (rtx, tree, rtx);
static rtx arm_trampoline_adjust_address (rtx);
static rtx_insn *arm_pic_static_addr (rtx orig, rtx reg);
static bool cortex_a9_sched_adjust_cost (rtx_insn *, int, rtx_insn *, int *);
static bool xscale_sched_adjust_cost (rtx_insn *, int, rtx_insn *, int *);
static bool fa726te_sched_adjust_cost (rtx_insn *, int, rtx_insn *, int *);
static bool arm_array_mode_supported_p (machine_mode,
unsigned HOST_WIDE_INT);
static machine_mode arm_preferred_simd_mode (scalar_mode);
static bool arm_class_likely_spilled_p (reg_class_t);
static HOST_WIDE_INT arm_vector_alignment (const_tree type);
static bool arm_vector_alignment_reachable (const_tree type, bool is_packed);
static bool arm_builtin_support_vector_misalignment (machine_mode mode,
const_tree type,
int misalignment,
bool is_packed);
static void arm_conditional_register_usage (void);
static enum flt_eval_method arm_excess_precision (enum excess_precision_type);
static reg_class_t arm_preferred_rename_class (reg_class_t rclass);
static void arm_autovectorize_vector_sizes (vector_sizes *);
static int arm_default_branch_cost (bool, bool);
static int arm_cortex_a5_branch_cost (bool, bool);
static int arm_cortex_m_branch_cost (bool, bool);
static int arm_cortex_m7_branch_cost (bool, bool);
static bool arm_vectorize_vec_perm_const (machine_mode, rtx, rtx, rtx,
const vec_perm_indices &);
static bool aarch_macro_fusion_pair_p (rtx_insn*, rtx_insn*);
static int arm_builtin_vectorization_cost (enum vect_cost_for_stmt type_of_cost,
tree vectype,
int misalign ATTRIBUTE_UNUSED);
static unsigned arm_add_stmt_cost (void *data, int count,
enum vect_cost_for_stmt kind,
struct _stmt_vec_info *stmt_info,
int misalign,
enum vect_cost_model_location where);
static void arm_canonicalize_comparison (int *code, rtx *op0, rtx *op1,
bool op0_preserve_value);
static unsigned HOST_WIDE_INT arm_asan_shadow_offset (void);
static void arm_sched_fusion_priority (rtx_insn *, int, int *, int*);
static bool arm_can_output_mi_thunk (const_tree, HOST_WIDE_INT, HOST_WIDE_INT,
const_tree);
static section *arm_function_section (tree, enum node_frequency, bool, bool);
static bool arm_asm_elf_flags_numeric (unsigned int flags, unsigned int *num);
static unsigned int arm_elf_section_type_flags (tree decl, const char *name,
int reloc);
static void arm_expand_divmod_libfunc (rtx, machine_mode, rtx, rtx, rtx *, rtx *);
static opt_scalar_float_mode arm_floatn_mode (int, bool);
static unsigned int arm_hard_regno_nregs (unsigned int, machine_mode);
static bool arm_hard_regno_mode_ok (unsigned int, machine_mode);
static bool arm_modes_tieable_p (machine_mode, machine_mode);
static HOST_WIDE_INT arm_constant_alignment (const_tree, HOST_WIDE_INT);

static const struct attribute_spec arm_attribute_table[] =
{
{ "long_call",    0, 0, false, true,  true,  false, NULL, NULL },
{ "short_call",   0, 0, false, true,  true,  false, NULL, NULL },
{ "pcs",          1, 1, false, true,  true,  false, arm_handle_pcs_attribute,
NULL },
{ "isr",          0, 1, false, false, false, false, arm_handle_isr_attribute,
NULL },
{ "interrupt",    0, 1, false, false, false, false, arm_handle_isr_attribute,
NULL },
{ "naked",        0, 0, true,  false, false, false,
arm_handle_fndecl_attribute, NULL },
#ifdef ARM_PE
{ "dllimport",    0, 0, true,  false, false, false, NULL, NULL },
{ "dllexport",    0, 0, true,  false, false, false, NULL, NULL },
{ "interfacearm", 0, 0, true,  false, false, false,
arm_handle_fndecl_attribute, NULL },
#elif TARGET_DLLIMPORT_DECL_ATTRIBUTES
{ "dllimport",    0, 0, false, false, false, false, handle_dll_attribute,
NULL },
{ "dllexport",    0, 0, false, false, false, false, handle_dll_attribute,
NULL },
{ "notshared",    0, 0, false, true, false, false,
arm_handle_notshared_attribute, NULL },
#endif
{ "cmse_nonsecure_entry", 0, 0, true, false, false, false,
arm_handle_cmse_nonsecure_entry, NULL },
{ "cmse_nonsecure_call", 0, 0, true, false, false, true,
arm_handle_cmse_nonsecure_call, NULL },
{ NULL, 0, 0, false, false, false, false, NULL, NULL }
};

#if TARGET_DLLIMPORT_DECL_ATTRIBUTES
#undef  TARGET_MERGE_DECL_ATTRIBUTES
#define TARGET_MERGE_DECL_ATTRIBUTES merge_dllimport_decl_attributes
#endif
#undef TARGET_LEGITIMIZE_ADDRESS
#define TARGET_LEGITIMIZE_ADDRESS arm_legitimize_address
#undef  TARGET_ATTRIBUTE_TABLE
#define TARGET_ATTRIBUTE_TABLE arm_attribute_table
#undef  TARGET_INSERT_ATTRIBUTES
#define TARGET_INSERT_ATTRIBUTES arm_insert_attributes
#undef TARGET_ASM_FILE_START
#define TARGET_ASM_FILE_START arm_file_start
#undef TARGET_ASM_FILE_END
#define TARGET_ASM_FILE_END arm_file_end
#undef  TARGET_ASM_ALIGNED_SI_OP
#define TARGET_ASM_ALIGNED_SI_OP NULL
#undef  TARGET_ASM_INTEGER
#define TARGET_ASM_INTEGER arm_assemble_integer
#undef TARGET_PRINT_OPERAND
#define TARGET_PRINT_OPERAND arm_print_operand
#undef TARGET_PRINT_OPERAND_ADDRESS
#define TARGET_PRINT_OPERAND_ADDRESS arm_print_operand_address
#undef TARGET_PRINT_OPERAND_PUNCT_VALID_P
#define TARGET_PRINT_OPERAND_PUNCT_VALID_P arm_print_operand_punct_valid_p
#undef TARGET_ASM_OUTPUT_ADDR_CONST_EXTRA
#define TARGET_ASM_OUTPUT_ADDR_CONST_EXTRA arm_output_addr_const_extra
#undef  TARGET_ASM_FUNCTION_PROLOGUE
#define TARGET_ASM_FUNCTION_PROLOGUE arm_output_function_prologue
#undef  TARGET_ASM_FUNCTION_EPILOGUE
#define TARGET_ASM_FUNCTION_EPILOGUE arm_output_function_epilogue
#undef TARGET_CAN_INLINE_P
#define TARGET_CAN_INLINE_P arm_can_inline_p
#undef TARGET_RELAYOUT_FUNCTION
#define TARGET_RELAYOUT_FUNCTION arm_relayout_function
#undef  TARGET_OPTION_OVERRIDE
#define TARGET_OPTION_OVERRIDE arm_option_override
#undef TARGET_OVERRIDE_OPTIONS_AFTER_CHANGE
#define TARGET_OVERRIDE_OPTIONS_AFTER_CHANGE arm_override_options_after_change
#undef TARGET_OPTION_SAVE
#define TARGET_OPTION_SAVE arm_option_save
#undef TARGET_OPTION_RESTORE
#define TARGET_OPTION_RESTORE arm_option_restore
#undef TARGET_OPTION_PRINT
#define TARGET_OPTION_PRINT arm_option_print
#undef  TARGET_COMP_TYPE_ATTRIBUTES
#define TARGET_COMP_TYPE_ATTRIBUTES arm_comp_type_attributes
#undef TARGET_SCHED_CAN_SPECULATE_INSN
#define TARGET_SCHED_CAN_SPECULATE_INSN arm_sched_can_speculate_insn
#undef TARGET_SCHED_MACRO_FUSION_P
#define TARGET_SCHED_MACRO_FUSION_P arm_macro_fusion_p
#undef TARGET_SCHED_MACRO_FUSION_PAIR_P
#define TARGET_SCHED_MACRO_FUSION_PAIR_P aarch_macro_fusion_pair_p
#undef  TARGET_SET_DEFAULT_TYPE_ATTRIBUTES
#define TARGET_SET_DEFAULT_TYPE_ATTRIBUTES arm_set_default_type_attributes
#undef  TARGET_SCHED_ADJUST_COST
#define TARGET_SCHED_ADJUST_COST arm_adjust_cost
#undef TARGET_SET_CURRENT_FUNCTION
#define TARGET_SET_CURRENT_FUNCTION arm_set_current_function
#undef TARGET_OPTION_VALID_ATTRIBUTE_P
#define TARGET_OPTION_VALID_ATTRIBUTE_P arm_valid_target_attribute_p
#undef TARGET_SCHED_REORDER
#define TARGET_SCHED_REORDER arm_sched_reorder
#undef TARGET_REGISTER_MOVE_COST
#define TARGET_REGISTER_MOVE_COST arm_register_move_cost
#undef TARGET_MEMORY_MOVE_COST
#define TARGET_MEMORY_MOVE_COST arm_memory_move_cost
#undef TARGET_ENCODE_SECTION_INFO
#ifdef ARM_PE
#define TARGET_ENCODE_SECTION_INFO  arm_pe_encode_section_info
#else
#define TARGET_ENCODE_SECTION_INFO  arm_encode_section_info
#endif
#undef  TARGET_STRIP_NAME_ENCODING
#define TARGET_STRIP_NAME_ENCODING arm_strip_name_encoding
#undef  TARGET_ASM_INTERNAL_LABEL
#define TARGET_ASM_INTERNAL_LABEL arm_internal_label
#undef TARGET_FLOATN_MODE
#define TARGET_FLOATN_MODE arm_floatn_mode
#undef  TARGET_FUNCTION_OK_FOR_SIBCALL
#define TARGET_FUNCTION_OK_FOR_SIBCALL arm_function_ok_for_sibcall
#undef  TARGET_FUNCTION_VALUE
#define TARGET_FUNCTION_VALUE arm_function_value
#undef  TARGET_LIBCALL_VALUE
#define TARGET_LIBCALL_VALUE arm_libcall_value
#undef TARGET_FUNCTION_VALUE_REGNO_P
#define TARGET_FUNCTION_VALUE_REGNO_P arm_function_value_regno_p
#undef  TARGET_ASM_OUTPUT_MI_THUNK
#define TARGET_ASM_OUTPUT_MI_THUNK arm_output_mi_thunk
#undef  TARGET_ASM_CAN_OUTPUT_MI_THUNK
#define TARGET_ASM_CAN_OUTPUT_MI_THUNK arm_can_output_mi_thunk
#undef  TARGET_RTX_COSTS
#define TARGET_RTX_COSTS arm_rtx_costs
#undef  TARGET_ADDRESS_COST
#define TARGET_ADDRESS_COST arm_address_cost
#undef TARGET_SHIFT_TRUNCATION_MASK
#define TARGET_SHIFT_TRUNCATION_MASK arm_shift_truncation_mask
#undef TARGET_VECTOR_MODE_SUPPORTED_P
#define TARGET_VECTOR_MODE_SUPPORTED_P arm_vector_mode_supported_p
#undef TARGET_ARRAY_MODE_SUPPORTED_P
#define TARGET_ARRAY_MODE_SUPPORTED_P arm_array_mode_supported_p
#undef TARGET_VECTORIZE_PREFERRED_SIMD_MODE
#define TARGET_VECTORIZE_PREFERRED_SIMD_MODE arm_preferred_simd_mode
#undef TARGET_VECTORIZE_AUTOVECTORIZE_VECTOR_SIZES
#define TARGET_VECTORIZE_AUTOVECTORIZE_VECTOR_SIZES \
arm_autovectorize_vector_sizes
#undef  TARGET_MACHINE_DEPENDENT_REORG
#define TARGET_MACHINE_DEPENDENT_REORG arm_reorg
#undef  TARGET_INIT_BUILTINS
#define TARGET_INIT_BUILTINS  arm_init_builtins
#undef  TARGET_EXPAND_BUILTIN
#define TARGET_EXPAND_BUILTIN arm_expand_builtin
#undef  TARGET_BUILTIN_DECL
#define TARGET_BUILTIN_DECL arm_builtin_decl
#undef TARGET_INIT_LIBFUNCS
#define TARGET_INIT_LIBFUNCS arm_init_libfuncs
#undef TARGET_PROMOTE_FUNCTION_MODE
#define TARGET_PROMOTE_FUNCTION_MODE arm_promote_function_mode
#undef TARGET_PROMOTE_PROTOTYPES
#define TARGET_PROMOTE_PROTOTYPES arm_promote_prototypes
#undef TARGET_PASS_BY_REFERENCE
#define TARGET_PASS_BY_REFERENCE arm_pass_by_reference
#undef TARGET_ARG_PARTIAL_BYTES
#define TARGET_ARG_PARTIAL_BYTES arm_arg_partial_bytes
#undef TARGET_FUNCTION_ARG
#define TARGET_FUNCTION_ARG arm_function_arg
#undef TARGET_FUNCTION_ARG_ADVANCE
#define TARGET_FUNCTION_ARG_ADVANCE arm_function_arg_advance
#undef TARGET_FUNCTION_ARG_PADDING
#define TARGET_FUNCTION_ARG_PADDING arm_function_arg_padding
#undef TARGET_FUNCTION_ARG_BOUNDARY
#define TARGET_FUNCTION_ARG_BOUNDARY arm_function_arg_boundary
#undef  TARGET_SETUP_INCOMING_VARARGS
#define TARGET_SETUP_INCOMING_VARARGS arm_setup_incoming_varargs
#undef TARGET_ALLOCATE_STACK_SLOTS_FOR_ARGS
#define TARGET_ALLOCATE_STACK_SLOTS_FOR_ARGS arm_allocate_stack_slots_for_args
#undef TARGET_ASM_TRAMPOLINE_TEMPLATE
#define TARGET_ASM_TRAMPOLINE_TEMPLATE arm_asm_trampoline_template
#undef TARGET_TRAMPOLINE_INIT
#define TARGET_TRAMPOLINE_INIT arm_trampoline_init
#undef TARGET_TRAMPOLINE_ADJUST_ADDRESS
#define TARGET_TRAMPOLINE_ADJUST_ADDRESS arm_trampoline_adjust_address
#undef TARGET_WARN_FUNC_RETURN
#define TARGET_WARN_FUNC_RETURN arm_warn_func_return
#undef TARGET_DEFAULT_SHORT_ENUMS
#define TARGET_DEFAULT_SHORT_ENUMS arm_default_short_enums
#undef TARGET_ALIGN_ANON_BITFIELD
#define TARGET_ALIGN_ANON_BITFIELD arm_align_anon_bitfield
#undef TARGET_NARROW_VOLATILE_BITFIELD
#define TARGET_NARROW_VOLATILE_BITFIELD hook_bool_void_false
#undef TARGET_CXX_GUARD_TYPE
#define TARGET_CXX_GUARD_TYPE arm_cxx_guard_type
#undef TARGET_CXX_GUARD_MASK_BIT
#define TARGET_CXX_GUARD_MASK_BIT arm_cxx_guard_mask_bit
#undef TARGET_CXX_GET_COOKIE_SIZE
#define TARGET_CXX_GET_COOKIE_SIZE arm_get_cookie_size
#undef TARGET_CXX_COOKIE_HAS_SIZE
#define TARGET_CXX_COOKIE_HAS_SIZE arm_cookie_has_size
#undef TARGET_CXX_CDTOR_RETURNS_THIS
#define TARGET_CXX_CDTOR_RETURNS_THIS arm_cxx_cdtor_returns_this
#undef TARGET_CXX_KEY_METHOD_MAY_BE_INLINE
#define TARGET_CXX_KEY_METHOD_MAY_BE_INLINE arm_cxx_key_method_may_be_inline
#undef TARGET_CXX_USE_AEABI_ATEXIT
#define TARGET_CXX_USE_AEABI_ATEXIT arm_cxx_use_aeabi_atexit
#undef TARGET_CXX_DETERMINE_CLASS_DATA_VISIBILITY
#define TARGET_CXX_DETERMINE_CLASS_DATA_VISIBILITY \
arm_cxx_determine_class_data_visibility
#undef TARGET_CXX_CLASS_DATA_ALWAYS_COMDAT
#define TARGET_CXX_CLASS_DATA_ALWAYS_COMDAT arm_cxx_class_data_always_comdat
#undef TARGET_RETURN_IN_MSB
#define TARGET_RETURN_IN_MSB arm_return_in_msb
#undef TARGET_RETURN_IN_MEMORY
#define TARGET_RETURN_IN_MEMORY arm_return_in_memory
#undef TARGET_MUST_PASS_IN_STACK
#define TARGET_MUST_PASS_IN_STACK arm_must_pass_in_stack
#if ARM_UNWIND_INFO
#undef TARGET_ASM_UNWIND_EMIT
#define TARGET_ASM_UNWIND_EMIT arm_unwind_emit
#undef TARGET_ASM_TTYPE
#define TARGET_ASM_TTYPE arm_output_ttype
#undef TARGET_ARM_EABI_UNWINDER
#define TARGET_ARM_EABI_UNWINDER true
#undef TARGET_ASM_EMIT_EXCEPT_PERSONALITY
#define TARGET_ASM_EMIT_EXCEPT_PERSONALITY arm_asm_emit_except_personality
#endif 
#undef TARGET_ASM_INIT_SECTIONS
#define TARGET_ASM_INIT_SECTIONS arm_asm_init_sections
#undef TARGET_DWARF_REGISTER_SPAN
#define TARGET_DWARF_REGISTER_SPAN arm_dwarf_register_span
#undef  TARGET_CANNOT_COPY_INSN_P
#define TARGET_CANNOT_COPY_INSN_P arm_cannot_copy_insn_p
#ifdef HAVE_AS_TLS
#undef TARGET_HAVE_TLS
#define TARGET_HAVE_TLS true
#endif
#undef TARGET_HAVE_CONDITIONAL_EXECUTION
#define TARGET_HAVE_CONDITIONAL_EXECUTION arm_have_conditional_execution
#undef TARGET_LEGITIMATE_CONSTANT_P
#define TARGET_LEGITIMATE_CONSTANT_P arm_legitimate_constant_p
#undef TARGET_CANNOT_FORCE_CONST_MEM
#define TARGET_CANNOT_FORCE_CONST_MEM arm_cannot_force_const_mem
#undef TARGET_MAX_ANCHOR_OFFSET
#define TARGET_MAX_ANCHOR_OFFSET 4095
#undef TARGET_MIN_ANCHOR_OFFSET
#define TARGET_MIN_ANCHOR_OFFSET -4088
#undef TARGET_SCHED_ISSUE_RATE
#define TARGET_SCHED_ISSUE_RATE arm_issue_rate
#undef TARGET_SCHED_FIRST_CYCLE_MULTIPASS_DFA_LOOKAHEAD
#define TARGET_SCHED_FIRST_CYCLE_MULTIPASS_DFA_LOOKAHEAD \
arm_first_cycle_multipass_dfa_lookahead
#undef TARGET_SCHED_FIRST_CYCLE_MULTIPASS_DFA_LOOKAHEAD_GUARD
#define TARGET_SCHED_FIRST_CYCLE_MULTIPASS_DFA_LOOKAHEAD_GUARD \
arm_first_cycle_multipass_dfa_lookahead_guard
#undef TARGET_MANGLE_TYPE
#define TARGET_MANGLE_TYPE arm_mangle_type
#undef TARGET_ATOMIC_ASSIGN_EXPAND_FENV
#define TARGET_ATOMIC_ASSIGN_EXPAND_FENV arm_atomic_assign_expand_fenv
#undef TARGET_BUILD_BUILTIN_VA_LIST
#define TARGET_BUILD_BUILTIN_VA_LIST arm_build_builtin_va_list
#undef TARGET_EXPAND_BUILTIN_VA_START
#define TARGET_EXPAND_BUILTIN_VA_START arm_expand_builtin_va_start
#undef TARGET_GIMPLIFY_VA_ARG_EXPR
#define TARGET_GIMPLIFY_VA_ARG_EXPR arm_gimplify_va_arg_expr
#ifdef HAVE_AS_TLS
#undef TARGET_ASM_OUTPUT_DWARF_DTPREL
#define TARGET_ASM_OUTPUT_DWARF_DTPREL arm_output_dwarf_dtprel
#endif
#undef TARGET_LEGITIMATE_ADDRESS_P
#define TARGET_LEGITIMATE_ADDRESS_P	arm_legitimate_address_p
#undef TARGET_PREFERRED_RELOAD_CLASS
#define TARGET_PREFERRED_RELOAD_CLASS arm_preferred_reload_class
#undef TARGET_PROMOTED_TYPE
#define TARGET_PROMOTED_TYPE arm_promoted_type
#undef TARGET_SCALAR_MODE_SUPPORTED_P
#define TARGET_SCALAR_MODE_SUPPORTED_P arm_scalar_mode_supported_p
#undef TARGET_COMPUTE_FRAME_LAYOUT
#define TARGET_COMPUTE_FRAME_LAYOUT arm_compute_frame_layout
#undef TARGET_FRAME_POINTER_REQUIRED
#define TARGET_FRAME_POINTER_REQUIRED arm_frame_pointer_required
#undef TARGET_CAN_ELIMINATE
#define TARGET_CAN_ELIMINATE arm_can_eliminate
#undef TARGET_CONDITIONAL_REGISTER_USAGE
#define TARGET_CONDITIONAL_REGISTER_USAGE arm_conditional_register_usage
#undef TARGET_CLASS_LIKELY_SPILLED_P
#define TARGET_CLASS_LIKELY_SPILLED_P arm_class_likely_spilled_p
#undef TARGET_VECTORIZE_BUILTINS
#define TARGET_VECTORIZE_BUILTINS
#undef TARGET_VECTORIZE_BUILTIN_VECTORIZED_FUNCTION
#define TARGET_VECTORIZE_BUILTIN_VECTORIZED_FUNCTION \
arm_builtin_vectorized_function
#undef TARGET_VECTOR_ALIGNMENT
#define TARGET_VECTOR_ALIGNMENT arm_vector_alignment
#undef TARGET_VECTORIZE_VECTOR_ALIGNMENT_REACHABLE
#define TARGET_VECTORIZE_VECTOR_ALIGNMENT_REACHABLE \
arm_vector_alignment_reachable
#undef TARGET_VECTORIZE_SUPPORT_VECTOR_MISALIGNMENT
#define TARGET_VECTORIZE_SUPPORT_VECTOR_MISALIGNMENT \
arm_builtin_support_vector_misalignment
#undef TARGET_PREFERRED_RENAME_CLASS
#define TARGET_PREFERRED_RENAME_CLASS \
arm_preferred_rename_class
#undef TARGET_VECTORIZE_VEC_PERM_CONST
#define TARGET_VECTORIZE_VEC_PERM_CONST arm_vectorize_vec_perm_const
#undef TARGET_VECTORIZE_BUILTIN_VECTORIZATION_COST
#define TARGET_VECTORIZE_BUILTIN_VECTORIZATION_COST \
arm_builtin_vectorization_cost
#undef TARGET_VECTORIZE_ADD_STMT_COST
#define TARGET_VECTORIZE_ADD_STMT_COST arm_add_stmt_cost
#undef TARGET_CANONICALIZE_COMPARISON
#define TARGET_CANONICALIZE_COMPARISON \
arm_canonicalize_comparison
#undef TARGET_ASAN_SHADOW_OFFSET
#define TARGET_ASAN_SHADOW_OFFSET arm_asan_shadow_offset
#undef MAX_INSN_PER_IT_BLOCK
#define MAX_INSN_PER_IT_BLOCK (arm_restrict_it ? 1 : 4)
#undef TARGET_CAN_USE_DOLOOP_P
#define TARGET_CAN_USE_DOLOOP_P can_use_doloop_if_innermost
#undef TARGET_CONST_NOT_OK_FOR_DEBUG_P
#define TARGET_CONST_NOT_OK_FOR_DEBUG_P arm_const_not_ok_for_debug_p
#undef TARGET_CALL_FUSAGE_CONTAINS_NON_CALLEE_CLOBBERS
#define TARGET_CALL_FUSAGE_CONTAINS_NON_CALLEE_CLOBBERS true
#undef TARGET_SCHED_FUSION_PRIORITY
#define TARGET_SCHED_FUSION_PRIORITY arm_sched_fusion_priority
#undef  TARGET_ASM_FUNCTION_SECTION
#define TARGET_ASM_FUNCTION_SECTION arm_function_section
#undef TARGET_ASM_ELF_FLAGS_NUMERIC
#define TARGET_ASM_ELF_FLAGS_NUMERIC arm_asm_elf_flags_numeric
#undef TARGET_SECTION_TYPE_FLAGS
#define TARGET_SECTION_TYPE_FLAGS arm_elf_section_type_flags
#undef TARGET_EXPAND_DIVMOD_LIBFUNC
#define TARGET_EXPAND_DIVMOD_LIBFUNC arm_expand_divmod_libfunc
#undef TARGET_C_EXCESS_PRECISION
#define TARGET_C_EXCESS_PRECISION arm_excess_precision
#undef TARGET_CUSTOM_FUNCTION_DESCRIPTORS
#define TARGET_CUSTOM_FUNCTION_DESCRIPTORS 2
#undef TARGET_FIXED_CONDITION_CODE_REGS
#define TARGET_FIXED_CONDITION_CODE_REGS arm_fixed_condition_code_regs
#undef TARGET_HARD_REGNO_NREGS
#define TARGET_HARD_REGNO_NREGS arm_hard_regno_nregs
#undef TARGET_HARD_REGNO_MODE_OK
#define TARGET_HARD_REGNO_MODE_OK arm_hard_regno_mode_ok
#undef TARGET_MODES_TIEABLE_P
#define TARGET_MODES_TIEABLE_P arm_modes_tieable_p
#undef TARGET_CAN_CHANGE_MODE_CLASS
#define TARGET_CAN_CHANGE_MODE_CLASS arm_can_change_mode_class
#undef TARGET_CONSTANT_ALIGNMENT
#define TARGET_CONSTANT_ALIGNMENT arm_constant_alignment

static struct obstack minipool_obstack;
static char *         minipool_startobj;
static int max_insns_skipped = 5;
extern FILE * asm_out_file;
int making_const_table;
enum processor_type arm_tune = TARGET_CPU_arm_none;
const struct tune_params *current_tune;
int arm_fpu_attr;
rtx thumb_call_via_label[14];
static int thumb_call_reg_needed;
unsigned int tune_flags = 0;
enum base_architecture arm_base_arch = BASE_ARCH_0;
struct arm_build_target arm_active_target;
int arm_arch3m = 0;
int arm_arch4 = 0;
int arm_arch4t = 0;
int arm_arch5 = 0;
int arm_arch5e = 0;
int arm_arch5te = 0;
int arm_arch6 = 0;
int arm_arch6k = 0;
int arm_arch6kz = 0;
int arm_arch6m = 0;
int arm_arch7 = 0;
int arm_arch_lpae = 0;
int arm_arch_notm = 0;
int arm_arch7em = 0;
int arm_arch8 = 0;
int arm_arch8_1 = 0;
int arm_arch8_2 = 0;
int arm_fp16_inst = 0;
int arm_ld_sched = 0;
int arm_tune_strongarm = 0;
int arm_arch_iwmmxt = 0;
int arm_arch_iwmmxt2 = 0;
int arm_arch_xscale = 0;
int arm_tune_xscale = 0;
int arm_tune_wbuf = 0;
int arm_tune_cortex_a9 = 0;
int arm_cpp_interwork = 0;
int arm_arch_thumb1;
int arm_arch_thumb2;
int arm_arch_arm_hwdiv;
int arm_arch_thumb_hwdiv;
int arm_arch_no_volatile_ce;
int prefer_neon_for_64bits = 0;
bool arm_disable_literal_pool = false;
unsigned arm_pic_register = INVALID_REGNUM;
enum arm_pcs arm_pcs_default;
int arm_ccfsm_state;
enum arm_cond_code arm_current_cc;
rtx arm_target_insn;
int arm_target_label;
int arm_condexec_count = 0;
int arm_condexec_mask = 0;
int arm_condexec_masklen = 0;
int arm_arch_crc = 0;
int arm_arch_dotprod = 0;
int arm_arch_cmse = 0;
int arm_m_profile_small_mul = 0;
static const char * const arm_condition_codes[] =
{
"eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc",
"hi", "ls", "ge", "lt", "gt", "le", "al", "nv"
};
int arm_regs_in_sequence[] =
{
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};
#define ARM_LSL_NAME "lsl"
#define streq(string1, string2) (strcmp (string1, string2) == 0)
#define THUMB2_WORK_REGS (0xff & ~(  (1 << THUMB_HARD_FRAME_POINTER_REGNUM) \
| (1 << SP_REGNUM) | (1 << PC_REGNUM) \
| (1 << PIC_OFFSET_TABLE_REGNUM)))

struct cpu_tune
{
enum processor_type scheduler;
unsigned int tune_flags;
const struct tune_params *tune;
};
#define ARM_PREFETCH_NOT_BENEFICIAL { 0, -1, -1 }
#define ARM_PREFETCH_BENEFICIAL(num_slots,l1_size,l1_line_size) \
{								\
num_slots,							\
l1_size,							\
l1_line_size						\
}
static const
struct cpu_vec_costs arm_default_vec_cost = {
1,					
1,					
1,					
1,					
1,					
1,					
1,					
1,					
1,					
1,					
3,					
1,					
};
#include "aarch-cost-tables.h"
const struct cpu_cost_table cortexa9_extra_costs =
{
{
0,			
0,			
0,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (2),	
0,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
0,			
0,			
0,			
true		
},
{
{
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (30)	
},
{
0,			
0,			
COSTS_N_INSNS (4),	
0,			
COSTS_N_INSNS (4),	
0				
}
},
{
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
1,			
2,			
COSTS_N_INSNS (5),	
COSTS_N_INSNS (5),	
COSTS_N_INSNS (1),  
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
1,			
2,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1)	
},
{
{
COSTS_N_INSNS (14),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (7),	
COSTS_N_INSNS (30),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3)		
},
{
COSTS_N_INSNS (24),	
COSTS_N_INSNS (5),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (30),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3)		
}
},
{
COSTS_N_INSNS (1)	
}
};
const struct cpu_cost_table cortexa8_extra_costs =
{
{
0,			
0,			
COSTS_N_INSNS (1),	
0,			
COSTS_N_INSNS (1),	
0,			
COSTS_N_INSNS (1),	
0,			
0,			
0,			
0,			
0,			
0,			
0,			
0,			
true		
},
{
{
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (30)	
},
{
0,			
0,			
COSTS_N_INSNS (2),	
0,			
COSTS_N_INSNS (2),	
0				
}
},
{
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
1,			
2,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),  
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
1,			
2,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1)	
},
{
{
COSTS_N_INSNS (36),	
COSTS_N_INSNS (11),	
COSTS_N_INSNS (20),	
COSTS_N_INSNS (30),	
COSTS_N_INSNS (9),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (6),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (8)		
},
{
COSTS_N_INSNS (64),	
COSTS_N_INSNS (16),	
COSTS_N_INSNS (25),	
COSTS_N_INSNS (30),	
COSTS_N_INSNS (9),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (6),	
COSTS_N_INSNS (6),	
COSTS_N_INSNS (6),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (8)		
}
},
{
COSTS_N_INSNS (1)	
}
};
const struct cpu_cost_table cortexa5_extra_costs =
{
{
0,			
0,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
0,			
true		
},
{
{
0,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (7)		
},
{
0,			
0,			
COSTS_N_INSNS (1),	
0,			
COSTS_N_INSNS (2),	
0				
}
},
{
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (6),	
COSTS_N_INSNS (1),	
1,			
2,			
COSTS_N_INSNS (2),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (1),	
1,			
2,			
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1)	
},
{
{
COSTS_N_INSNS (15),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (7),	
COSTS_N_INSNS (7),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3)		
},
{
COSTS_N_INSNS (30),	
COSTS_N_INSNS (6),	
COSTS_N_INSNS (10),	
COSTS_N_INSNS (7),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3)		
}
},
{
COSTS_N_INSNS (1)	
}
};
const struct cpu_cost_table cortexa7_extra_costs =
{
{
0,			
0,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
0,			
true		
},
{
{
0,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (7)		
},
{
0,			
0,			
COSTS_N_INSNS (1),	
0,			
COSTS_N_INSNS (2),	
0				
}
},
{
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (1),	
1,			
2,			
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (1),	
1,			
2,			
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1)	
},
{
{
COSTS_N_INSNS (15),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (7),	
COSTS_N_INSNS (7),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3)		
},
{
COSTS_N_INSNS (30),	
COSTS_N_INSNS (6),	
COSTS_N_INSNS (10),	
COSTS_N_INSNS (7),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3)		
}
},
{
COSTS_N_INSNS (1)	
}
};
const struct cpu_cost_table cortexa12_extra_costs =
{
{
0,			
0,			
0,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
0,			
COSTS_N_INSNS (1),	
0,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
0,			
true		
},
{
{
COSTS_N_INSNS (2),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (18)	
},
{
0,			
0,			
COSTS_N_INSNS (3),	
0,			
COSTS_N_INSNS (3),	
0				
}
},
{
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
1,			
2,			
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
0,			
0,			
0,			
0,			
1,			
2,			
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
0,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1)	
},
{
{
COSTS_N_INSNS (17),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4)		
},
{
COSTS_N_INSNS (31),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4)		
}
},
{
COSTS_N_INSNS (1)	
}
};
const struct cpu_cost_table cortexa15_extra_costs =
{
{
0,			
0,			
0,			
0,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
0,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
0,			
0,			
0,			
0,			
true		
},
{
{
COSTS_N_INSNS (2),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (18)	
},
{
0,			
0,			
COSTS_N_INSNS (3),	
0,			
COSTS_N_INSNS (3),	
0				
}
},
{
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (4),	
1,			
2,			
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
0,			
0,			
0,			
COSTS_N_INSNS (1),	
1,			
2,			
0,			
0,			
0,			
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1)	
},
{
{
COSTS_N_INSNS (17),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (5),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4)		
},
{
COSTS_N_INSNS (31),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (8),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4),	
COSTS_N_INSNS (4)		
}
},
{
COSTS_N_INSNS (1)	
}
};
const struct cpu_cost_table v7m_extra_costs =
{
{
0,			
0,			
0,			
0,			
0,			
COSTS_N_INSNS (1),	
0,			
COSTS_N_INSNS (1),	
0,			
COSTS_N_INSNS (1),	
0,			
0,			
0,			
0,			
COSTS_N_INSNS (1),	
false		
},
{
{
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (8)		
},
{
0,			
0,			
COSTS_N_INSNS (2),	
0,			
COSTS_N_INSNS (3),	
0				
}
},
{
COSTS_N_INSNS (2),	
0,			
COSTS_N_INSNS (3),	
COSTS_N_INSNS (2),	
1,			
1,			
COSTS_N_INSNS (2),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (1),  
COSTS_N_INSNS (2),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (2),	
1,			
1,			
COSTS_N_INSNS (2),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1),	
COSTS_N_INSNS (1)	
},
{
{
COSTS_N_INSNS (7),	
COSTS_N_INSNS (2),	
COSTS_N_INSNS (5),	
COSTS_N_INSNS (3),	
COSTS_N_INSNS (1),	
0,			
0,			
0,			
0,			
0,			
0,			
0,			
0				
},
{
COSTS_N_INSNS (15),	
COSTS_N_INSNS (5),	
COSTS_N_INSNS (7),	
COSTS_N_INSNS (7),	
COSTS_N_INSNS (3),	
0,			
0,			
0,			
0,			
0,			
0,			
0,			
0				
}
},
{
COSTS_N_INSNS (1)	
}
};
const struct addr_mode_cost_table generic_addr_mode_costs =
{
{
COSTS_N_INSNS (0),	
COSTS_N_INSNS (0),	
COSTS_N_INSNS (0)	
},
{
COSTS_N_INSNS (0),	
COSTS_N_INSNS (0),	
COSTS_N_INSNS (0)	
},
{
COSTS_N_INSNS (0),	
COSTS_N_INSNS (0),	
COSTS_N_INSNS (0)	
}
};
const struct tune_params arm_slowmul_tune =
{
&generic_extra_costs,			
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
3,						
5,						
8,						
1,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_TRUE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_fastmul_tune =
{
&generic_extra_costs,			
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
5,						
8,						
1,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_TRUE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_strongarm_tune =
{
&generic_extra_costs,			
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
3,						
8,						
1,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_TRUE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_xscale_tune =
{
&generic_extra_costs,			
&generic_addr_mode_costs,		
xscale_sched_adjust_cost,
arm_default_branch_cost,
&arm_default_vec_cost,
2,						
3,						
8,						
1,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_TRUE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_9e_tune =
{
&generic_extra_costs,			
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
5,						
8,						
1,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_TRUE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_marvell_pj4_tune =
{
&generic_extra_costs,			
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
5,						
8,						
2,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_TRUE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_v6t2_tune =
{
&generic_extra_costs,			
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
5,						
8,						
1,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_cortex_tune =
{
&generic_extra_costs,
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
5,						
8,						
2,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_cortex_a8_tune =
{
&cortexa8_extra_costs,
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
5,						
8,						
2,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_TRUE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_cortex_a7_tune =
{
&cortexa7_extra_costs,
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
5,						
8,						
2,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_TRUE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_cortex_a15_tune =
{
&cortexa15_extra_costs,
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
2,						
8,						
3,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_TRUE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_ALL,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_TRUE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_FULL
};
const struct tune_params arm_cortex_a35_tune =
{
&cortexa53_extra_costs,
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
5,						
8,						
1,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_TRUE,
FUSE_OPS (tune_params::FUSE_MOVW_MOVT),
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_cortex_a53_tune =
{
&cortexa53_extra_costs,
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
5,						
8,						
2,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_TRUE,
FUSE_OPS (tune_params::FUSE_MOVW_MOVT | tune_params::FUSE_AES_AESMC),
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_cortex_a57_tune =
{
&cortexa57_extra_costs,
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
2,						
8,						
3,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_TRUE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_ALL,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_TRUE,
FUSE_OPS (tune_params::FUSE_MOVW_MOVT | tune_params::FUSE_AES_AESMC),
tune_params::SCHED_AUTOPREF_FULL
};
const struct tune_params arm_exynosm1_tune =
{
&exynosm1_extra_costs,
&generic_addr_mode_costs,			
NULL,						
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
2,						
8,						
3,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_TRUE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_FALSE,	
tune_params::LOG_OP_NON_SHORT_CIRCUIT_FALSE,	
tune_params::DISPARAGE_FLAGS_ALL,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_TRUE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_xgene1_tune =
{
&xgene1_extra_costs,
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
2,						
32,						
4,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_TRUE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_ALL,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_cortex_a5_tune =
{
&cortexa5_extra_costs,
&generic_addr_mode_costs,		
NULL,					
arm_cortex_a5_branch_cost,
&arm_default_vec_cost,
1,						
1,						
8,						
2,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_FALSE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_FALSE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_TRUE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_cortex_a9_tune =
{
&cortexa9_extra_costs,
&generic_addr_mode_costs,		
cortex_a9_sched_adjust_cost,
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
5,						
8,						
2,						
ARM_PREFETCH_BENEFICIAL(4,32,32),
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_cortex_a12_tune =
{
&cortexa12_extra_costs,
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,                        
1,						
2,						
8,						
2,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_TRUE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_ALL,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_TRUE,
FUSE_OPS (tune_params::FUSE_MOVW_MOVT),
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_cortex_a73_tune =
{
&cortexa57_extra_costs,
&generic_addr_mode_costs,			
NULL,						
arm_default_branch_cost,
&arm_default_vec_cost,			
1,						
2,						
8,						
2,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_TRUE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_ALL,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_TRUE,
FUSE_OPS (tune_params::FUSE_AES_AESMC | tune_params::FUSE_MOVW_MOVT),
tune_params::SCHED_AUTOPREF_FULL
};
const struct tune_params arm_v7m_tune =
{
&v7m_extra_costs,
&generic_addr_mode_costs,		
NULL,					
arm_cortex_m_branch_cost,
&arm_default_vec_cost,
1,						
2,						
8,						
1,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_TRUE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_FALSE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_FALSE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_cortex_m7_tune =
{
&v7m_extra_costs,
&generic_addr_mode_costs,		
NULL,					
arm_cortex_m7_branch_cost,
&arm_default_vec_cost,
0,						
1,						
8,						
2,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_TRUE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_v6m_tune =
{
&generic_extra_costs,			
&generic_addr_mode_costs,		
NULL,					
arm_default_branch_cost,
&arm_default_vec_cost,                        
1,						
5,						
8,						
1,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_FALSE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_FALSE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_FALSE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
const struct tune_params arm_fa726te_tune =
{
&generic_extra_costs,				
&generic_addr_mode_costs,			
fa726te_sched_adjust_cost,
arm_default_branch_cost,
&arm_default_vec_cost,
1,						
5,						
8,						
2,						
ARM_PREFETCH_NOT_BENEFICIAL,
tune_params::PREF_CONST_POOL_TRUE,
tune_params::PREF_LDRD_FALSE,
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::LOG_OP_NON_SHORT_CIRCUIT_TRUE,		
tune_params::DISPARAGE_FLAGS_NEITHER,
tune_params::PREF_NEON_64_FALSE,
tune_params::PREF_NEON_STRINGOPS_FALSE,
tune_params::FUSE_NOTHING,
tune_params::SCHED_AUTOPREF_OFF
};
#include "arm-cpu-data.h"
char arm_arch_name[] = "__ARM_ARCH_PROFILE__";
enum tls_reloc {
TLS_GD32,
TLS_LDM32,
TLS_LDO32,
TLS_IE32,
TLS_LE32,
TLS_DESCSEQ	
};
inline static int
arm_constant_limit (bool size_p)
{
return size_p ? 1 : current_tune->constant_limit;
}
inline static rtx_insn *
emit_set_insn (rtx x, rtx y)
{
return emit_insn (gen_rtx_SET (x, y));
}
static unsigned
bit_count (unsigned long value)
{
unsigned long count = 0;
while (value)
{
count++;
value &= value - 1;  
}
return count;
}
static unsigned
bitmap_popcount (const sbitmap bmap)
{
unsigned int count = 0;
unsigned int n = 0;
sbitmap_iterator sbi;
EXECUTE_IF_SET_IN_BITMAP (bmap, 0, n, sbi)
count++;
return count;
}
typedef struct
{
machine_mode mode;
const char *name;
} arm_fixed_mode_set;
static void
arm_set_fixed_optab_libfunc (optab optable, machine_mode mode,
const char *funcname, const char *modename,
int num_suffix)
{
char buffer[50];
if (num_suffix == 0)
sprintf (buffer, "__gnu_%s%s", funcname, modename);
else
sprintf (buffer, "__gnu_%s%s%d", funcname, modename, num_suffix);
set_optab_libfunc (optable, mode, buffer);
}
static void
arm_set_fixed_conv_libfunc (convert_optab optable, machine_mode to,
machine_mode from, const char *funcname,
const char *toname, const char *fromname)
{
char buffer[50];
const char *maybe_suffix_2 = "";
if (ALL_FIXED_POINT_MODE_P (from) && ALL_FIXED_POINT_MODE_P (to)
&& UNSIGNED_FIXED_POINT_MODE_P (from) == UNSIGNED_FIXED_POINT_MODE_P (to)
&& ALL_FRACT_MODE_P (from) == ALL_FRACT_MODE_P (to))
maybe_suffix_2 = "2";
sprintf (buffer, "__gnu_%s%s%s%s", funcname, fromname, toname,
maybe_suffix_2);
set_conv_libfunc (optable, to, from, buffer);
}
static void
arm_init_libfuncs (void)
{
if (arm_abi == ARM_ABI_AAPCS_LINUX)
init_sync_libfuncs (MAX_SYNC_LIBFUNC_SIZE);
if (!TARGET_BPABI)
return;
set_optab_libfunc (add_optab, DFmode, "__aeabi_dadd");
set_optab_libfunc (sdiv_optab, DFmode, "__aeabi_ddiv");
set_optab_libfunc (smul_optab, DFmode, "__aeabi_dmul");
set_optab_libfunc (neg_optab, DFmode, "__aeabi_dneg");
set_optab_libfunc (sub_optab, DFmode, "__aeabi_dsub");
set_optab_libfunc (eq_optab, DFmode, "__aeabi_dcmpeq");
set_optab_libfunc (ne_optab, DFmode, NULL);
set_optab_libfunc (lt_optab, DFmode, "__aeabi_dcmplt");
set_optab_libfunc (le_optab, DFmode, "__aeabi_dcmple");
set_optab_libfunc (ge_optab, DFmode, "__aeabi_dcmpge");
set_optab_libfunc (gt_optab, DFmode, "__aeabi_dcmpgt");
set_optab_libfunc (unord_optab, DFmode, "__aeabi_dcmpun");
set_optab_libfunc (add_optab, SFmode, "__aeabi_fadd");
set_optab_libfunc (sdiv_optab, SFmode, "__aeabi_fdiv");
set_optab_libfunc (smul_optab, SFmode, "__aeabi_fmul");
set_optab_libfunc (neg_optab, SFmode, "__aeabi_fneg");
set_optab_libfunc (sub_optab, SFmode, "__aeabi_fsub");
set_optab_libfunc (eq_optab, SFmode, "__aeabi_fcmpeq");
set_optab_libfunc (ne_optab, SFmode, NULL);
set_optab_libfunc (lt_optab, SFmode, "__aeabi_fcmplt");
set_optab_libfunc (le_optab, SFmode, "__aeabi_fcmple");
set_optab_libfunc (ge_optab, SFmode, "__aeabi_fcmpge");
set_optab_libfunc (gt_optab, SFmode, "__aeabi_fcmpgt");
set_optab_libfunc (unord_optab, SFmode, "__aeabi_fcmpun");
set_conv_libfunc (sfix_optab, SImode, DFmode, "__aeabi_d2iz");
set_conv_libfunc (ufix_optab, SImode, DFmode, "__aeabi_d2uiz");
set_conv_libfunc (sfix_optab, DImode, DFmode, "__aeabi_d2lz");
set_conv_libfunc (ufix_optab, DImode, DFmode, "__aeabi_d2ulz");
set_conv_libfunc (sfix_optab, SImode, SFmode, "__aeabi_f2iz");
set_conv_libfunc (ufix_optab, SImode, SFmode, "__aeabi_f2uiz");
set_conv_libfunc (sfix_optab, DImode, SFmode, "__aeabi_f2lz");
set_conv_libfunc (ufix_optab, DImode, SFmode, "__aeabi_f2ulz");
set_conv_libfunc (trunc_optab, SFmode, DFmode, "__aeabi_d2f");
set_conv_libfunc (sext_optab, DFmode, SFmode, "__aeabi_f2d");
set_conv_libfunc (sfloat_optab, DFmode, SImode, "__aeabi_i2d");
set_conv_libfunc (ufloat_optab, DFmode, SImode, "__aeabi_ui2d");
set_conv_libfunc (sfloat_optab, DFmode, DImode, "__aeabi_l2d");
set_conv_libfunc (ufloat_optab, DFmode, DImode, "__aeabi_ul2d");
set_conv_libfunc (sfloat_optab, SFmode, SImode, "__aeabi_i2f");
set_conv_libfunc (ufloat_optab, SFmode, SImode, "__aeabi_ui2f");
set_conv_libfunc (sfloat_optab, SFmode, DImode, "__aeabi_l2f");
set_conv_libfunc (ufloat_optab, SFmode, DImode, "__aeabi_ul2f");
set_optab_libfunc (smul_optab, DImode, "__aeabi_lmul");
set_optab_libfunc (sdivmod_optab, DImode, "__aeabi_ldivmod");
set_optab_libfunc (udivmod_optab, DImode, "__aeabi_uldivmod");
set_optab_libfunc (ashl_optab, DImode, "__aeabi_llsl");
set_optab_libfunc (lshr_optab, DImode, "__aeabi_llsr");
set_optab_libfunc (ashr_optab, DImode, "__aeabi_lasr");
set_optab_libfunc (cmp_optab, DImode, "__aeabi_lcmp");
set_optab_libfunc (ucmp_optab, DImode, "__aeabi_ulcmp");
set_optab_libfunc (sdivmod_optab, SImode, "__aeabi_idivmod");
set_optab_libfunc (udivmod_optab, SImode, "__aeabi_uidivmod");
set_optab_libfunc (sdiv_optab, DImode, "__aeabi_ldivmod");
set_optab_libfunc (udiv_optab, DImode, "__aeabi_uldivmod");
set_optab_libfunc (sdiv_optab, SImode, "__aeabi_idiv");
set_optab_libfunc (udiv_optab, SImode, "__aeabi_uidiv");
set_optab_libfunc (smod_optab, DImode, NULL);
set_optab_libfunc (umod_optab, DImode, NULL);
set_optab_libfunc (smod_optab, SImode, NULL);
set_optab_libfunc (umod_optab, SImode, NULL);
switch (arm_fp16_format)
{
case ARM_FP16_FORMAT_IEEE:
case ARM_FP16_FORMAT_ALTERNATIVE:
set_conv_libfunc (trunc_optab, HFmode, SFmode,
(arm_fp16_format == ARM_FP16_FORMAT_IEEE
? "__gnu_f2h_ieee"
: "__gnu_f2h_alternative"));
set_conv_libfunc (sext_optab, SFmode, HFmode,
(arm_fp16_format == ARM_FP16_FORMAT_IEEE
? "__gnu_h2f_ieee"
: "__gnu_h2f_alternative"));
set_conv_libfunc (trunc_optab, HFmode, DFmode,
(arm_fp16_format == ARM_FP16_FORMAT_IEEE
? "__gnu_d2h_ieee"
: "__gnu_d2h_alternative"));
set_optab_libfunc (add_optab, HFmode, NULL);
set_optab_libfunc (sdiv_optab, HFmode, NULL);
set_optab_libfunc (smul_optab, HFmode, NULL);
set_optab_libfunc (neg_optab, HFmode, NULL);
set_optab_libfunc (sub_optab, HFmode, NULL);
set_optab_libfunc (eq_optab, HFmode, NULL);
set_optab_libfunc (ne_optab, HFmode, NULL);
set_optab_libfunc (lt_optab, HFmode, NULL);
set_optab_libfunc (le_optab, HFmode, NULL);
set_optab_libfunc (ge_optab, HFmode, NULL);
set_optab_libfunc (gt_optab, HFmode, NULL);
set_optab_libfunc (unord_optab, HFmode, NULL);
break;
default:
break;
}
{
const arm_fixed_mode_set fixed_arith_modes[] =
{
{ E_QQmode, "qq" },
{ E_UQQmode, "uqq" },
{ E_HQmode, "hq" },
{ E_UHQmode, "uhq" },
{ E_SQmode, "sq" },
{ E_USQmode, "usq" },
{ E_DQmode, "dq" },
{ E_UDQmode, "udq" },
{ E_TQmode, "tq" },
{ E_UTQmode, "utq" },
{ E_HAmode, "ha" },
{ E_UHAmode, "uha" },
{ E_SAmode, "sa" },
{ E_USAmode, "usa" },
{ E_DAmode, "da" },
{ E_UDAmode, "uda" },
{ E_TAmode, "ta" },
{ E_UTAmode, "uta" }
};
const arm_fixed_mode_set fixed_conv_modes[] =
{
{ E_QQmode, "qq" },
{ E_UQQmode, "uqq" },
{ E_HQmode, "hq" },
{ E_UHQmode, "uhq" },
{ E_SQmode, "sq" },
{ E_USQmode, "usq" },
{ E_DQmode, "dq" },
{ E_UDQmode, "udq" },
{ E_TQmode, "tq" },
{ E_UTQmode, "utq" },
{ E_HAmode, "ha" },
{ E_UHAmode, "uha" },
{ E_SAmode, "sa" },
{ E_USAmode, "usa" },
{ E_DAmode, "da" },
{ E_UDAmode, "uda" },
{ E_TAmode, "ta" },
{ E_UTAmode, "uta" },
{ E_QImode, "qi" },
{ E_HImode, "hi" },
{ E_SImode, "si" },
{ E_DImode, "di" },
{ E_TImode, "ti" },
{ E_SFmode, "sf" },
{ E_DFmode, "df" }
};
unsigned int i, j;
for (i = 0; i < ARRAY_SIZE (fixed_arith_modes); i++)
{
arm_set_fixed_optab_libfunc (add_optab, fixed_arith_modes[i].mode,
"add", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (ssadd_optab, fixed_arith_modes[i].mode,
"ssadd", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (usadd_optab, fixed_arith_modes[i].mode,
"usadd", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (sub_optab, fixed_arith_modes[i].mode,
"sub", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (sssub_optab, fixed_arith_modes[i].mode,
"sssub", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (ussub_optab, fixed_arith_modes[i].mode,
"ussub", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (smul_optab, fixed_arith_modes[i].mode,
"mul", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (ssmul_optab, fixed_arith_modes[i].mode,
"ssmul", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (usmul_optab, fixed_arith_modes[i].mode,
"usmul", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (sdiv_optab, fixed_arith_modes[i].mode,
"div", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (udiv_optab, fixed_arith_modes[i].mode,
"udiv", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (ssdiv_optab, fixed_arith_modes[i].mode,
"ssdiv", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (usdiv_optab, fixed_arith_modes[i].mode,
"usdiv", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (neg_optab, fixed_arith_modes[i].mode,
"neg", fixed_arith_modes[i].name, 2);
arm_set_fixed_optab_libfunc (ssneg_optab, fixed_arith_modes[i].mode,
"ssneg", fixed_arith_modes[i].name, 2);
arm_set_fixed_optab_libfunc (usneg_optab, fixed_arith_modes[i].mode,
"usneg", fixed_arith_modes[i].name, 2);
arm_set_fixed_optab_libfunc (ashl_optab, fixed_arith_modes[i].mode,
"ashl", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (ashr_optab, fixed_arith_modes[i].mode,
"ashr", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (lshr_optab, fixed_arith_modes[i].mode,
"lshr", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (ssashl_optab, fixed_arith_modes[i].mode,
"ssashl", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (usashl_optab, fixed_arith_modes[i].mode,
"usashl", fixed_arith_modes[i].name, 3);
arm_set_fixed_optab_libfunc (cmp_optab, fixed_arith_modes[i].mode,
"cmp", fixed_arith_modes[i].name, 2);
}
for (i = 0; i < ARRAY_SIZE (fixed_conv_modes); i++)
for (j = 0; j < ARRAY_SIZE (fixed_conv_modes); j++)
{
if (i == j
|| (!ALL_FIXED_POINT_MODE_P (fixed_conv_modes[i].mode)
&& !ALL_FIXED_POINT_MODE_P (fixed_conv_modes[j].mode)))
continue;
arm_set_fixed_conv_libfunc (fract_optab, fixed_conv_modes[i].mode,
fixed_conv_modes[j].mode, "fract",
fixed_conv_modes[i].name,
fixed_conv_modes[j].name);
arm_set_fixed_conv_libfunc (satfract_optab,
fixed_conv_modes[i].mode,
fixed_conv_modes[j].mode, "satfract",
fixed_conv_modes[i].name,
fixed_conv_modes[j].name);
arm_set_fixed_conv_libfunc (fractuns_optab,
fixed_conv_modes[i].mode,
fixed_conv_modes[j].mode, "fractuns",
fixed_conv_modes[i].name,
fixed_conv_modes[j].name);
arm_set_fixed_conv_libfunc (satfractuns_optab,
fixed_conv_modes[i].mode,
fixed_conv_modes[j].mode, "satfractuns",
fixed_conv_modes[i].name,
fixed_conv_modes[j].name);
}
}
if (TARGET_AAPCS_BASED)
synchronize_libfunc = init_one_libfunc ("__sync_synchronize");
}
static GTY(()) tree va_list_type;
static tree
arm_build_builtin_va_list (void)
{
tree va_list_name;
tree ap_field;
if (!TARGET_AAPCS_BASED)
return std_build_builtin_va_list ();
va_list_type = lang_hooks.types.make_type (RECORD_TYPE);
va_list_name = build_decl (BUILTINS_LOCATION,
TYPE_DECL,
get_identifier ("__va_list"),
va_list_type);
DECL_ARTIFICIAL (va_list_name) = 1;
TYPE_NAME (va_list_type) = va_list_name;
TYPE_STUB_DECL (va_list_type) = va_list_name;
ap_field = build_decl (BUILTINS_LOCATION,
FIELD_DECL,
get_identifier ("__ap"),
ptr_type_node);
DECL_ARTIFICIAL (ap_field) = 1;
DECL_FIELD_CONTEXT (ap_field) = va_list_type;
TYPE_FIELDS (va_list_type) = ap_field;
layout_type (va_list_type);
return va_list_type;
}
static tree
arm_extract_valist_ptr (tree valist)
{
if (TREE_TYPE (valist) == error_mark_node)
return error_mark_node;
if (TARGET_AAPCS_BASED)
{
tree ap_field = TYPE_FIELDS (TREE_TYPE (valist));
valist = build3 (COMPONENT_REF, TREE_TYPE (ap_field),
valist, ap_field, NULL_TREE);
}
return valist;
}
static void
arm_expand_builtin_va_start (tree valist, rtx nextarg)
{
valist = arm_extract_valist_ptr (valist);
std_expand_builtin_va_start (valist, nextarg);
}
static tree
arm_gimplify_va_arg_expr (tree valist, tree type, gimple_seq *pre_p,
gimple_seq *post_p)
{
valist = arm_extract_valist_ptr (valist);
return std_gimplify_va_arg_expr (valist, type, pre_p, post_p);
}
static void
arm_option_check_internal (struct gcc_options *opts)
{
int flags = opts->x_target_flags;
if (TARGET_IWMMXT
&& bitmap_bit_p (arm_active_target.isa, isa_bit_neon))
error ("iWMMXt and NEON are incompatible");
if (TARGET_ARM_P (flags)
&& !bitmap_bit_p (arm_active_target.isa, isa_bit_notm))
error ("target CPU does not support ARM mode");
if ((TARGET_TPCS_FRAME || TARGET_TPCS_LEAF_FRAME) && TARGET_ARM_P (flags))
warning (0, "enabling backtrace support is only meaningful when compiling for the Thumb");
if (TARGET_ARM_P (flags) && TARGET_CALLEE_INTERWORKING)
warning (0, "enabling callee interworking support is only meaningful when compiling for the Thumb");
if (TARGET_ARM_P (flags)
&& write_symbols != NO_DEBUG
&& !TARGET_APCS_FRAME
&& (TARGET_DEFAULT & MASK_APCS_FRAME))
warning (0, "-g with -mno-apcs-frame may not give sensible debugging");
if (TARGET_THUMB_P (flags) && TARGET_IWMMXT)
error ("iWMMXt unsupported under Thumb mode");
if (TARGET_HARD_TP && TARGET_THUMB1_P (flags))
error ("can not use -mtp=cp15 with 16-bit Thumb");
if (TARGET_THUMB_P (flags) && TARGET_VXWORKS_RTP && flag_pic)
{
error ("RTP PIC is incompatible with Thumb");
flag_pic = 0;
}
if ((target_pure_code || target_slow_flash_data)
&& (!TARGET_HAVE_MOVT || arm_arch_notm || flag_pic || TARGET_NEON))
{
const char *flag = (target_pure_code ? "-mpure-code" :
"-mslow-flash-data");
error ("%s only supports non-pic code on M-profile targets with the "
"MOVT instruction", flag);
}
}
static void
arm_option_params_internal (void)
{
if (TARGET_THUMB1)
{
targetm.min_anchor_offset = 0;
targetm.max_anchor_offset = 127;
}
else if (TARGET_THUMB2)
{
targetm.min_anchor_offset = -248;
targetm.max_anchor_offset = 4095;
}
else
{
targetm.min_anchor_offset = TARGET_MIN_ANCHOR_OFFSET;
targetm.max_anchor_offset = TARGET_MAX_ANCHOR_OFFSET;
}
max_insns_skipped = optimize_size ? 4 : current_tune->max_insns_skipped;
if (TARGET_THUMB2)
max_insns_skipped = MIN (max_insns_skipped, MAX_INSN_PER_IT_BLOCK);
}
static GTY(()) bool thumb_flipper;
static GTY(()) tree init_optimize;
static void
arm_override_options_after_change_1 (struct gcc_options *opts)
{
if (opts->x_align_functions <= 0)
opts->x_align_functions = TARGET_THUMB_P (opts->x_target_flags)
&& opts->x_optimize_size ? 2 : 4;
}
static void
arm_override_options_after_change (void)
{
arm_configure_build_target (&arm_active_target,
TREE_TARGET_OPTION (target_option_default_node),
&global_options_set, false);
arm_override_options_after_change_1 (&global_options);
}
static void
arm_option_save (struct cl_target_option *ptr, struct gcc_options *opts)
{
ptr->x_arm_arch_string = opts->x_arm_arch_string;
ptr->x_arm_cpu_string = opts->x_arm_cpu_string;
ptr->x_arm_tune_string = opts->x_arm_tune_string;
}
static void
arm_option_restore (struct gcc_options *opts, struct cl_target_option *ptr)
{
opts->x_arm_arch_string = ptr->x_arm_arch_string;
opts->x_arm_cpu_string = ptr->x_arm_cpu_string;
opts->x_arm_tune_string = ptr->x_arm_tune_string;
arm_configure_build_target (&arm_active_target, ptr, &global_options_set,
false);
}
static void
arm_option_override_internal (struct gcc_options *opts,
struct gcc_options *opts_set)
{
arm_override_options_after_change_1 (opts);
if (TARGET_INTERWORK && !bitmap_bit_p (arm_active_target.isa, isa_bit_thumb))
{
opts->x_target_flags &= ~MASK_INTERWORK;
}
if (TARGET_THUMB_P (opts->x_target_flags)
&& !bitmap_bit_p (arm_active_target.isa, isa_bit_thumb))
{
warning (0, "target CPU does not support THUMB instructions");
opts->x_target_flags &= ~MASK_THUMB;
}
if (TARGET_APCS_FRAME && TARGET_THUMB_P (opts->x_target_flags))
{
opts->x_target_flags &= ~MASK_APCS_FRAME;
}
if (TARGET_THUMB_P (opts->x_target_flags) && TARGET_CALLEE_INTERWORKING)
opts->x_target_flags |= MASK_INTERWORK;
cl_optimization *to = TREE_OPTIMIZATION (init_optimize);
if (! opts_set->x_arm_restrict_it)
opts->x_arm_restrict_it = arm_arch8;
if (!TARGET_THUMB2_P (opts->x_target_flags) || !arm_arch_notm)
opts->x_arm_restrict_it = 0;
if (! opts_set->x_unaligned_access)
{
opts->x_unaligned_access = (TARGET_32BIT_P (opts->x_target_flags)
&& arm_arch6 && (arm_arch_notm || arm_arch7));
}
else if (opts->x_unaligned_access == 1
&& !(arm_arch6 && (arm_arch_notm || arm_arch7)))
{
warning (0, "target CPU does not support unaligned accesses");
opts->x_unaligned_access = 0;
}
if (TARGET_THUMB1_P (opts->x_target_flags))
opts->x_flag_schedule_insns = 0;
else
opts->x_flag_schedule_insns = to->x_flag_schedule_insns;
if (optimize_function_for_size_p (cfun)
&& TARGET_THUMB2_P (opts->x_target_flags))
opts->x_flag_shrink_wrap = false;
else
opts->x_flag_shrink_wrap = to->x_flag_shrink_wrap;
if (TARGET_THUMB1_P (opts->x_target_flags))
opts->x_flag_ipa_ra = 0;
else
opts->x_flag_ipa_ra = to->x_flag_ipa_ra;
if (TARGET_THUMB2_P (opts->x_target_flags))
opts->x_inline_asm_unified = true;
#ifdef SUBTARGET_OVERRIDE_INTERNAL_OPTIONS
SUBTARGET_OVERRIDE_INTERNAL_OPTIONS;
#endif
}
static sbitmap isa_all_fpubits;
static sbitmap isa_quirkbits;
void
arm_configure_build_target (struct arm_build_target *target,
struct cl_target_option *opts,
struct gcc_options *opts_set,
bool warn_compatible)
{
const cpu_option *arm_selected_tune = NULL;
const arch_option *arm_selected_arch = NULL;
const cpu_option *arm_selected_cpu = NULL;
const arm_fpu_desc *arm_selected_fpu = NULL;
const char *tune_opts = NULL;
const char *arch_opts = NULL;
const char *cpu_opts = NULL;
bitmap_clear (target->isa);
target->core_name = NULL;
target->arch_name = NULL;
if (opts_set->x_arm_arch_string)
{
arm_selected_arch = arm_parse_arch_option_name (all_architectures,
"-march",
opts->x_arm_arch_string);
arch_opts = strchr (opts->x_arm_arch_string, '+');
}
if (opts_set->x_arm_cpu_string)
{
arm_selected_cpu = arm_parse_cpu_option_name (all_cores, "-mcpu",
opts->x_arm_cpu_string);
cpu_opts = strchr (opts->x_arm_cpu_string, '+');
arm_selected_tune = arm_selected_cpu;
}
if (opts_set->x_arm_tune_string)
{
arm_selected_tune = arm_parse_cpu_option_name (all_cores, "-mtune",
opts->x_arm_tune_string);
tune_opts = strchr (opts->x_arm_tune_string, '+');
}
if (arm_selected_arch)
{
arm_initialize_isa (target->isa, arm_selected_arch->common.isa_bits);
arm_parse_option_features (target->isa, &arm_selected_arch->common,
arch_opts);
if (arm_selected_cpu)
{
auto_sbitmap cpu_isa (isa_num_bits);
auto_sbitmap isa_delta (isa_num_bits);
arm_initialize_isa (cpu_isa, arm_selected_cpu->common.isa_bits);
arm_parse_option_features (cpu_isa, &arm_selected_cpu->common,
cpu_opts);
bitmap_xor (isa_delta, cpu_isa, target->isa);
bitmap_and_compl (isa_delta, isa_delta, isa_quirkbits);
bitmap_and_compl (isa_delta, isa_delta, isa_all_fpubits);
if (!bitmap_empty_p (isa_delta))
{
if (warn_compatible)
warning (0, "switch -mcpu=%s conflicts with -march=%s switch",
arm_selected_cpu->common.name,
arm_selected_arch->common.name);
if (!arm_selected_tune)
arm_selected_tune = arm_selected_cpu;
arm_selected_cpu = all_cores + arm_selected_arch->tune_id;
target->arch_name = arm_selected_arch->common.name;
}
else
{
arm_selected_arch = all_architectures + arm_selected_cpu->arch;
target->core_name = arm_selected_cpu->common.name;
bitmap_copy (target->isa, cpu_isa);
}
}
else
{
arm_selected_cpu = all_cores + arm_selected_arch->tune_id;
target->arch_name = arm_selected_arch->common.name;
}
}
else if (arm_selected_cpu)
{
target->core_name = arm_selected_cpu->common.name;
arm_initialize_isa (target->isa, arm_selected_cpu->common.isa_bits);
arm_parse_option_features (target->isa, &arm_selected_cpu->common,
cpu_opts);
arm_selected_arch = all_architectures + arm_selected_cpu->arch;
}
else
{
const cpu_option *sel;
auto_sbitmap sought_isa (isa_num_bits);
bitmap_clear (sought_isa);
auto_sbitmap default_isa (isa_num_bits);
arm_selected_cpu = arm_parse_cpu_option_name (all_cores, "default CPU",
TARGET_CPU_DEFAULT);
cpu_opts = strchr (TARGET_CPU_DEFAULT, '+');
gcc_assert (arm_selected_cpu->common.name);
sel = arm_selected_cpu;
arm_initialize_isa (default_isa, sel->common.isa_bits);
arm_parse_option_features (default_isa, &arm_selected_cpu->common,
cpu_opts);
if (TARGET_INTERWORK || TARGET_THUMB)
{
bitmap_set_bit (sought_isa, isa_bit_thumb);
bitmap_set_bit (sought_isa, isa_bit_mode32);
bitmap_clear_bit (default_isa, isa_bit_mode26);
}
if (!bitmap_empty_p (sought_isa)
&& !bitmap_subset_p (sought_isa, default_isa))
{
auto_sbitmap candidate_isa (isa_num_bits);
bitmap_ior (default_isa, default_isa, sought_isa);
for (sel = all_cores; sel->common.name != NULL; sel++)
{
arm_initialize_isa (candidate_isa, sel->common.isa_bits);
if (bitmap_equal_p (default_isa, candidate_isa))
break;
}
if (sel->common.name == NULL)
{
unsigned current_bit_count = isa_num_bits;
const cpu_option *best_fit = NULL;
for (sel = all_cores; sel->common.name != NULL; sel++)
{
arm_initialize_isa (candidate_isa, sel->common.isa_bits);
if (bitmap_subset_p (default_isa, candidate_isa))
{
unsigned count;
bitmap_and_compl (candidate_isa, candidate_isa,
default_isa);
count = bitmap_popcount (candidate_isa);
if (count < current_bit_count)
{
best_fit = sel;
current_bit_count = count;
}
}
gcc_assert (best_fit);
sel = best_fit;
}
}
arm_selected_cpu = sel;
}
target->core_name = arm_selected_cpu->common.name;
arm_initialize_isa (target->isa, arm_selected_cpu->common.isa_bits);
arm_parse_option_features (target->isa, &arm_selected_cpu->common,
cpu_opts);
arm_selected_arch = all_architectures + arm_selected_cpu->arch;
}
gcc_assert (arm_selected_cpu);
gcc_assert (arm_selected_arch);
if (opts->x_arm_fpu_index != TARGET_FPU_auto)
{
arm_selected_fpu = &all_fpus[opts->x_arm_fpu_index];
auto_sbitmap fpu_bits (isa_num_bits);
arm_initialize_isa (fpu_bits, arm_selected_fpu->isa_bits);
bitmap_and_compl (target->isa, target->isa, isa_all_fpubits);
bitmap_ior (target->isa, target->isa, fpu_bits);
}
if (!arm_selected_tune)
arm_selected_tune = arm_selected_cpu;
else 
arm_parse_option_features (NULL, &arm_selected_tune->common, tune_opts);
const cpu_tune *tune_data = &all_tunes[arm_selected_tune - all_cores];
target->arch_pp_name = arm_selected_arch->arch;
target->base_arch = arm_selected_arch->base_arch;
target->profile = arm_selected_arch->profile;
target->tune_flags = tune_data->tune_flags;
target->tune = tune_data->tune;
target->tune_core = tune_data->scheduler;
arm_option_reconfigure_globals ();
}
static void
arm_option_override (void)
{
static const enum isa_feature fpu_bitlist[]
= { ISA_ALL_FPU_INTERNAL, isa_nobit };
static const enum isa_feature quirk_bitlist[] = { ISA_ALL_QUIRKS, isa_nobit};
cl_target_option opts;
isa_quirkbits = sbitmap_alloc (isa_num_bits);
arm_initialize_isa (isa_quirkbits, quirk_bitlist);
isa_all_fpubits = sbitmap_alloc (isa_num_bits);
arm_initialize_isa (isa_all_fpubits, fpu_bitlist);
arm_active_target.isa = sbitmap_alloc (isa_num_bits);
if (!global_options_set.x_arm_fpu_index)
{
bool ok;
int fpu_index;
ok = opt_enum_arg_to_value (OPT_mfpu_, FPUTYPE_AUTO, &fpu_index,
CL_TARGET);
gcc_assert (ok);
arm_fpu_index = (enum fpu_type) fpu_index;
}
cl_target_option_save (&opts, &global_options);
arm_configure_build_target (&arm_active_target, &opts, &global_options_set,
true);
#ifdef SUBTARGET_OVERRIDE_OPTIONS
SUBTARGET_OVERRIDE_OPTIONS;
#endif
arm_option_reconfigure_globals ();
arm_tune = arm_active_target.tune_core;
tune_flags = arm_active_target.tune_flags;
current_tune = arm_active_target.tune;
if (TARGET_APCS_FRAME)
flag_shrink_wrap = false;
if (TARGET_APCS_STACK && !TARGET_APCS_FRAME)
{
warning (0, "-mapcs-stack-check incompatible with -mno-apcs-frame");
target_flags |= MASK_APCS_FRAME;
}
if (TARGET_POKE_FUNCTION_NAME)
target_flags |= MASK_APCS_FRAME;
if (TARGET_APCS_REENT && flag_pic)
error ("-fpic and -mapcs-reent are incompatible");
if (TARGET_APCS_REENT)
warning (0, "APCS reentrant code not supported.  Ignored");
arm_ld_sched = (tune_flags & TF_LDSCHED) != 0;
arm_tune_strongarm = (tune_flags & TF_STRONG) != 0;
arm_tune_wbuf = (tune_flags & TF_WBUF) != 0;
arm_tune_xscale = (tune_flags & TF_XSCALE) != 0;
arm_tune_cortex_a9 = (arm_tune == TARGET_CPU_cortexa9) != 0;
arm_m_profile_small_mul = (tune_flags & TF_SMALLMUL) != 0;
if (TARGET_SOFT_FLOAT && (tune_flags & TF_NO_MODE32))
flag_schedule_insns = flag_schedule_insns_after_reload = 0;
if (!global_options_set.x_arm_structure_size_boundary)
{
if (TARGET_AAPCS_BASED)
arm_structure_size_boundary = 8;
}
else
{
warning (0, "option %<-mstructure-size-boundary%> is deprecated");
if (arm_structure_size_boundary != 8
&& arm_structure_size_boundary != 32
&& !(ARM_DOUBLEWORD_ALIGN && arm_structure_size_boundary == 64))
{
if (ARM_DOUBLEWORD_ALIGN)
warning (0,
"structure size boundary can only be set to 8, 32 or 64");
else
warning (0, "structure size boundary can only be set to 8 or 32");
arm_structure_size_boundary
= (TARGET_AAPCS_BASED ? 8 : DEFAULT_STRUCTURE_SIZE_BOUNDARY);
}
}
if (TARGET_VXWORKS_RTP)
{
if (!global_options_set.x_arm_pic_data_is_text_relative)
arm_pic_data_is_text_relative = 0;
}
else if (flag_pic
&& !arm_pic_data_is_text_relative
&& !(global_options_set.x_target_flags & MASK_SINGLE_PIC_BASE))
target_flags |= MASK_SINGLE_PIC_BASE;
if (flag_pic && TARGET_SINGLE_PIC_BASE)
{
if (TARGET_VXWORKS_RTP)
warning (0, "RTP PIC is incompatible with -msingle-pic-base");
arm_pic_register = (TARGET_APCS_STACK || TARGET_AAPCS_BASED) ? 9 : 10;
}
if (flag_pic && TARGET_VXWORKS_RTP)
arm_pic_register = 9;
if (arm_pic_register_string != NULL)
{
int pic_register = decode_reg_name (arm_pic_register_string);
if (!flag_pic)
warning (0, "-mpic-register= is useless without -fpic");
else if (pic_register < 0 || call_used_regs[pic_register]
|| pic_register == HARD_FRAME_POINTER_REGNUM
|| pic_register == STACK_POINTER_REGNUM
|| pic_register >= PC_REGNUM
|| (TARGET_VXWORKS_RTP
&& (unsigned int) pic_register != arm_pic_register))
error ("unable to use '%s' for PIC register", arm_pic_register_string);
else
arm_pic_register = pic_register;
}
if (fix_cm3_ldrd == 2)
{
if (bitmap_bit_p (arm_active_target.isa, isa_bit_quirk_cm3_ldrd))
fix_cm3_ldrd = 1;
else
fix_cm3_ldrd = 0;
}
if (flag_reorder_blocks_and_partition)
{
inform (input_location,
"-freorder-blocks-and-partition not supported on this architecture");
flag_reorder_blocks_and_partition = 0;
flag_reorder_blocks = 1;
}
if (flag_pic)
maybe_set_param_value (PARAM_GCSE_UNRESTRICTED_COST, 2,
global_options.x_param_values,
global_options_set.x_param_values);
if (TARGET_AAPCS_BASED && flag_strict_volatile_bitfields < 0
&& abi_version_at_least(2))
flag_strict_volatile_bitfields = 1;
if (flag_prefetch_loop_arrays < 0
&& HAVE_prefetch
&& optimize >= 3
&& current_tune->prefetch.num_slots > 0)
flag_prefetch_loop_arrays = 1;
if (current_tune->prefetch.num_slots > 0)
maybe_set_param_value (PARAM_SIMULTANEOUS_PREFETCHES,
current_tune->prefetch.num_slots,
global_options.x_param_values,
global_options_set.x_param_values);
if (current_tune->prefetch.l1_cache_line_size >= 0)
maybe_set_param_value (PARAM_L1_CACHE_LINE_SIZE,
current_tune->prefetch.l1_cache_line_size,
global_options.x_param_values,
global_options_set.x_param_values);
if (current_tune->prefetch.l1_cache_size >= 0)
maybe_set_param_value (PARAM_L1_CACHE_SIZE,
current_tune->prefetch.l1_cache_size,
global_options.x_param_values,
global_options_set.x_param_values);
prefer_neon_for_64bits = current_tune->prefer_neon_for_64bits;
if (use_neon_for_64bits == 1)
prefer_neon_for_64bits = true;
maybe_set_param_value (PARAM_SCHED_PRESSURE_ALGORITHM, SCHED_PRESSURE_MODEL,
global_options.x_param_values,
global_options_set.x_param_values);
int param_sched_autopref_queue_depth;
switch (current_tune->sched_autopref)
{
case tune_params::SCHED_AUTOPREF_OFF:
param_sched_autopref_queue_depth = -1;
break;
case tune_params::SCHED_AUTOPREF_RANK:
param_sched_autopref_queue_depth = 0;
break;
case tune_params::SCHED_AUTOPREF_FULL:
param_sched_autopref_queue_depth = max_insn_queue_index + 1;
break;
default:
gcc_unreachable ();
}
maybe_set_param_value (PARAM_SCHED_AUTOPREF_QUEUE_DEPTH,
param_sched_autopref_queue_depth,
global_options.x_param_values,
global_options_set.x_param_values);
if (target_slow_flash_data || target_pure_code)
arm_disable_literal_pool = true;
if (flag_schedule_fusion == 2
&& (!arm_arch7 || !current_tune->prefer_ldrd_strd))
flag_schedule_fusion = 0;
init_optimize = build_optimization_node (&global_options);
arm_options_perform_arch_sanity_checks ();
arm_option_override_internal (&global_options, &global_options_set);
arm_option_check_internal (&global_options);
arm_option_params_internal ();
target_option_default_node = target_option_current_node
= build_target_option_node (&global_options);
arm_add_gc_roots ();
thumb_flipper = TARGET_THUMB;
}
void
arm_option_reconfigure_globals (void)
{
sprintf (arm_arch_name, "__ARM_ARCH_%s__", arm_active_target.arch_pp_name);
arm_base_arch = arm_active_target.base_arch;
arm_arch3m = bitmap_bit_p (arm_active_target.isa, isa_bit_armv3m);
arm_arch4 = bitmap_bit_p (arm_active_target.isa, isa_bit_armv4);
arm_arch4t = arm_arch4 && bitmap_bit_p (arm_active_target.isa, isa_bit_thumb);
arm_arch5 = bitmap_bit_p (arm_active_target.isa, isa_bit_armv5);
arm_arch5e = bitmap_bit_p (arm_active_target.isa, isa_bit_armv5e);
arm_arch5te = arm_arch5e
&& bitmap_bit_p (arm_active_target.isa, isa_bit_thumb);
arm_arch6 = bitmap_bit_p (arm_active_target.isa, isa_bit_armv6);
arm_arch6k = bitmap_bit_p (arm_active_target.isa, isa_bit_armv6k);
arm_arch_notm = bitmap_bit_p (arm_active_target.isa, isa_bit_notm);
arm_arch6m = arm_arch6 && !arm_arch_notm;
arm_arch7 = bitmap_bit_p (arm_active_target.isa, isa_bit_armv7);
arm_arch7em = bitmap_bit_p (arm_active_target.isa, isa_bit_armv7em);
arm_arch8 = bitmap_bit_p (arm_active_target.isa, isa_bit_armv8);
arm_arch8_1 = bitmap_bit_p (arm_active_target.isa, isa_bit_armv8_1);
arm_arch8_2 = bitmap_bit_p (arm_active_target.isa, isa_bit_armv8_2);
arm_arch_thumb1 = bitmap_bit_p (arm_active_target.isa, isa_bit_thumb);
arm_arch_thumb2 = bitmap_bit_p (arm_active_target.isa, isa_bit_thumb2);
arm_arch_xscale = bitmap_bit_p (arm_active_target.isa, isa_bit_xscale);
arm_arch_iwmmxt = bitmap_bit_p (arm_active_target.isa, isa_bit_iwmmxt);
arm_arch_iwmmxt2 = bitmap_bit_p (arm_active_target.isa, isa_bit_iwmmxt2);
arm_arch_thumb_hwdiv = bitmap_bit_p (arm_active_target.isa, isa_bit_tdiv);
arm_arch_arm_hwdiv = bitmap_bit_p (arm_active_target.isa, isa_bit_adiv);
arm_arch_crc = bitmap_bit_p (arm_active_target.isa, isa_bit_crc32);
arm_arch_cmse = bitmap_bit_p (arm_active_target.isa, isa_bit_cmse);
arm_fp16_inst = bitmap_bit_p (arm_active_target.isa, isa_bit_fp16);
arm_arch_lpae = bitmap_bit_p (arm_active_target.isa, isa_bit_lpae);
if (arm_fp16_inst)
{
if (arm_fp16_format == ARM_FP16_FORMAT_ALTERNATIVE)
error ("selected fp16 options are incompatible");
arm_fp16_format = ARM_FP16_FORMAT_IEEE;
}
arm_arch_no_volatile_ce
= bitmap_bit_p (arm_active_target.isa, isa_bit_quirk_no_volatile_ce);
arm_arch6kz = arm_arch6k && bitmap_bit_p (arm_active_target.isa,
isa_bit_quirk_armv6kz);
if (target_thread_pointer == TP_AUTO)
{
if (arm_arch6k && !TARGET_THUMB1)
target_thread_pointer = TP_CP15;
else
target_thread_pointer = TP_SOFT;
}
}
void
arm_options_perform_arch_sanity_checks (void)
{
if (TARGET_INTERWORK)
arm_cpp_interwork = 1;
if (arm_arch5)
target_flags &= ~MASK_INTERWORK;
if (TARGET_IWMMXT && !ARM_DOUBLEWORD_ALIGN)
error ("iwmmxt requires an AAPCS compatible ABI for proper operation");
if (TARGET_IWMMXT_ABI && !TARGET_IWMMXT)
error ("iwmmxt abi requires an iwmmxt capable cpu");
if (TARGET_INTERWORK
&& !TARGET_BPABI
&& !bitmap_bit_p (arm_active_target.isa, isa_bit_thumb))
{
warning (0, "target CPU does not support interworking" );
target_flags &= ~MASK_INTERWORK;
}
if (TARGET_SOFT_FLOAT)
arm_fpu_attr = FPU_NONE;
else
arm_fpu_attr = FPU_VFP;
if (TARGET_AAPCS_BASED)
{
if (TARGET_CALLER_INTERWORKING)
error ("AAPCS does not support -mcaller-super-interworking");
else
if (TARGET_CALLEE_INTERWORKING)
error ("AAPCS does not support -mcallee-super-interworking");
}
if (!arm_arch4 && arm_fp16_format != ARM_FP16_FORMAT_NONE)
sorry ("__fp16 and no ldrh");
if (use_cmse && !arm_arch_cmse)
error ("target CPU does not support ARMv8-M Security Extensions");
if (use_cmse && LAST_VFP_REGNUM > LAST_LO_VFP_REGNUM)
error ("ARMv8-M Security Extensions incompatible with selected FPU");
if (TARGET_AAPCS_BASED)
{
if (arm_abi == ARM_ABI_IWMMXT)
arm_pcs_default = ARM_PCS_AAPCS_IWMMXT;
else if (TARGET_HARD_FLOAT_ABI)
{
arm_pcs_default = ARM_PCS_AAPCS_VFP;
if (!bitmap_bit_p (arm_active_target.isa, isa_bit_vfpv2))
error ("-mfloat-abi=hard: selected processor lacks an FPU");
}
else
arm_pcs_default = ARM_PCS_AAPCS;
}
else
{
if (arm_float_abi == ARM_FLOAT_ABI_HARD)
sorry ("-mfloat-abi=hard and VFP");
if (arm_abi == ARM_ABI_APCS)
arm_pcs_default = ARM_PCS_APCS;
else
arm_pcs_default = ARM_PCS_ATPCS;
}
}
static void
arm_add_gc_roots (void)
{
gcc_obstack_init(&minipool_obstack);
minipool_startobj = (char *) obstack_alloc (&minipool_obstack, 0);
}

typedef struct
{
const char *const arg;
const unsigned long return_value;
}
isr_attribute_arg;
static const isr_attribute_arg isr_attribute_args [] =
{
{ "IRQ",   ARM_FT_ISR },
{ "irq",   ARM_FT_ISR },
{ "FIQ",   ARM_FT_FIQ },
{ "fiq",   ARM_FT_FIQ },
{ "ABORT", ARM_FT_ISR },
{ "abort", ARM_FT_ISR },
{ "ABORT", ARM_FT_ISR },
{ "abort", ARM_FT_ISR },
{ "UNDEF", ARM_FT_EXCEPTION },
{ "undef", ARM_FT_EXCEPTION },
{ "SWI",   ARM_FT_EXCEPTION },
{ "swi",   ARM_FT_EXCEPTION },
{ NULL,    ARM_FT_NORMAL }
};
static unsigned long
arm_isr_value (tree argument)
{
const isr_attribute_arg * ptr;
const char *              arg;
if (!arm_arch_notm)
return ARM_FT_NORMAL | ARM_FT_STACKALIGN;
if (argument == NULL_TREE)
return ARM_FT_ISR;
if (TREE_VALUE (argument) == NULL_TREE
|| TREE_CODE (TREE_VALUE (argument)) != STRING_CST)
return ARM_FT_UNKNOWN;
arg = TREE_STRING_POINTER (TREE_VALUE (argument));
for (ptr = isr_attribute_args; ptr->arg != NULL; ptr++)
if (streq (arg, ptr->arg))
return ptr->return_value;
return ARM_FT_UNKNOWN;
}
static unsigned long
arm_compute_func_type (void)
{
unsigned long type = ARM_FT_UNKNOWN;
tree a;
tree attr;
gcc_assert (TREE_CODE (current_function_decl) == FUNCTION_DECL);
if (optimize > 0
&& (TREE_NOTHROW (current_function_decl)
|| !(flag_unwind_tables
|| (flag_exceptions
&& arm_except_unwind_info (&global_options) != UI_SJLJ)))
&& TREE_THIS_VOLATILE (current_function_decl))
type |= ARM_FT_VOLATILE;
if (cfun->static_chain_decl != NULL)
type |= ARM_FT_NESTED;
attr = DECL_ATTRIBUTES (current_function_decl);
a = lookup_attribute ("naked", attr);
if (a != NULL_TREE)
type |= ARM_FT_NAKED;
a = lookup_attribute ("isr", attr);
if (a == NULL_TREE)
a = lookup_attribute ("interrupt", attr);
if (a == NULL_TREE)
type |= TARGET_INTERWORK ? ARM_FT_INTERWORKED : ARM_FT_NORMAL;
else
type |= arm_isr_value (TREE_VALUE (a));
if (lookup_attribute ("cmse_nonsecure_entry", attr))
type |= ARM_FT_CMSE_ENTRY;
return type;
}
unsigned long
arm_current_func_type (void)
{
if (ARM_FUNC_TYPE (cfun->machine->func_type) == ARM_FT_UNKNOWN)
cfun->machine->func_type = arm_compute_func_type ();
return cfun->machine->func_type;
}
bool
arm_allocate_stack_slots_for_args (void)
{
return !IS_NAKED (arm_current_func_type ());
}
static bool
arm_warn_func_return (tree decl)
{
return lookup_attribute ("naked", DECL_ATTRIBUTES (decl)) == NULL_TREE;
}

static void
arm_asm_trampoline_template (FILE *f)
{
fprintf (f, "\t.syntax unified\n");
if (TARGET_ARM)
{
fprintf (f, "\t.arm\n");
asm_fprintf (f, "\tldr\t%r, [%r, #0]\n", STATIC_CHAIN_REGNUM, PC_REGNUM);
asm_fprintf (f, "\tldr\t%r, [%r, #0]\n", PC_REGNUM, PC_REGNUM);
}
else if (TARGET_THUMB2)
{
fprintf (f, "\t.thumb\n");
asm_fprintf (f, "\tldr.w\t%r, [%r, #4]\n",
STATIC_CHAIN_REGNUM, PC_REGNUM);
asm_fprintf (f, "\tldr.w\t%r, [%r, #4]\n", PC_REGNUM, PC_REGNUM);
}
else
{
ASM_OUTPUT_ALIGN (f, 2);
fprintf (f, "\t.code\t16\n");
fprintf (f, ".Ltrampoline_start:\n");
asm_fprintf (f, "\tpush\t{r0, r1}\n");
asm_fprintf (f, "\tldr\tr0, [%r, #8]\n", PC_REGNUM);
asm_fprintf (f, "\tmov\t%r, r0\n", STATIC_CHAIN_REGNUM);
asm_fprintf (f, "\tldr\tr0, [%r, #8]\n", PC_REGNUM);
asm_fprintf (f, "\tstr\tr0, [%r, #4]\n", SP_REGNUM);
asm_fprintf (f, "\tpop\t{r0, %r}\n", PC_REGNUM);
}
assemble_aligned_integer (UNITS_PER_WORD, const0_rtx);
assemble_aligned_integer (UNITS_PER_WORD, const0_rtx);
}
static void
arm_trampoline_init (rtx m_tramp, tree fndecl, rtx chain_value)
{
rtx fnaddr, mem, a_tramp;
emit_block_move (m_tramp, assemble_trampoline_template (),
GEN_INT (TRAMPOLINE_SIZE), BLOCK_OP_NORMAL);
mem = adjust_address (m_tramp, SImode, TARGET_32BIT ? 8 : 12);
emit_move_insn (mem, chain_value);
mem = adjust_address (m_tramp, SImode, TARGET_32BIT ? 12 : 16);
fnaddr = XEXP (DECL_RTL (fndecl), 0);
emit_move_insn (mem, fnaddr);
a_tramp = XEXP (m_tramp, 0);
emit_library_call (gen_rtx_SYMBOL_REF (Pmode, "__clear_cache"),
LCT_NORMAL, VOIDmode, a_tramp, Pmode,
plus_constant (Pmode, a_tramp, TRAMPOLINE_SIZE), Pmode);
}
static rtx
arm_trampoline_adjust_address (rtx addr)
{
if (TARGET_THUMB)
addr = expand_simple_binop (Pmode, IOR, addr, const1_rtx,
NULL, 0, OPTAB_LIB_WIDEN);
return addr;
}

int
use_return_insn (int iscond, rtx sibling)
{
int regno;
unsigned int func_type;
unsigned long saved_int_regs;
unsigned HOST_WIDE_INT stack_adjust;
arm_stack_offsets *offsets;
if (!reload_completed)
return 0;
func_type = arm_current_func_type ();
if (func_type & (ARM_FT_VOLATILE | ARM_FT_NAKED | ARM_FT_STACKALIGN))
return 0;
if (IS_INTERRUPT (func_type) && (frame_pointer_needed || TARGET_THUMB))
return 0;
if (TARGET_LDRD && current_tune->prefer_ldrd_strd
&& !optimize_function_for_size_p (cfun))
return 0;
offsets = arm_get_frame_offsets ();
stack_adjust = offsets->outgoing_args - offsets->saved_regs;
if (crtl->args.pretend_args_size
|| cfun->machine->uses_anonymous_args
|| crtl->calls_eh_return
|| cfun->calls_alloca
|| !(stack_adjust == 0 || (TARGET_APCS_FRAME && frame_pointer_needed
&& stack_adjust == 4))
|| (!(TARGET_APCS_FRAME && frame_pointer_needed)
&& arm_compute_static_chain_stack_bytes() != 0))
return 0;
saved_int_regs = offsets->saved_regs_mask;
if (stack_adjust == 4 && !arm_arch5 && TARGET_ARM)
{
if (!call_used_regs[3])
return 0;
if (arm_size_return_regs () >= (4 * UNITS_PER_WORD))
return 0;
if (sibling)
{
gcc_assert (CALL_P (sibling));
if (find_regno_fusage (sibling, USE, 3))
return 0;
}
if (saved_int_regs & 0x7)
return 0;
}
if (TARGET_INTERWORK && saved_int_regs != 0 && !IS_INTERRUPT(func_type))
return 0;
if (iscond && arm_tune_strongarm)
{
if (saved_int_regs != 0 && saved_int_regs != (1 << LR_REGNUM))
return 0;
if (flag_pic
&& arm_pic_register != INVALID_REGNUM
&& df_regs_ever_live_p (PIC_OFFSET_TABLE_REGNUM))
return 0;
}
if (saved_int_regs && IS_CMSE_ENTRY (func_type))
return 0;
if (saved_int_regs && !(saved_int_regs & (1 << LR_REGNUM)))
return 0;
if (TARGET_HARD_FLOAT)
for (regno = FIRST_VFP_REGNUM; regno <= LAST_VFP_REGNUM; regno++)
if (df_regs_ever_live_p (regno) && !call_used_regs[regno])
return 0;
if (TARGET_REALLY_IWMMXT)
for (regno = FIRST_IWMMXT_REGNUM; regno <= LAST_IWMMXT_REGNUM; regno++)
if (df_regs_ever_live_p (regno) && ! call_used_regs[regno])
return 0;
return 1;
}
bool
use_simple_return_p (void)
{
arm_stack_offsets *offsets;
if (!reload_completed)
arm_compute_frame_layout ();
offsets = arm_get_frame_offsets ();
return offsets->outgoing_args != 0;
}
int
const_ok_for_arm (HOST_WIDE_INT i)
{
int lowbit;
if ((i & ~(unsigned HOST_WIDE_INT) 0xffffffff) != 0
&& ((i & ~(unsigned HOST_WIDE_INT) 0xffffffff)
!= ((~(unsigned HOST_WIDE_INT) 0)
& ~(unsigned HOST_WIDE_INT) 0xffffffff)))
return FALSE;
i &= (unsigned HOST_WIDE_INT) 0xffffffff;
if ((i & ~(unsigned HOST_WIDE_INT) 0xff) == 0)
return TRUE;
lowbit = ffs((int) i) - 1;
if (TARGET_ARM)
lowbit &= ~1;
if ((i & ~(((unsigned HOST_WIDE_INT) 0xff) << lowbit)) == 0)
return TRUE;
if (TARGET_ARM)
{
if (lowbit <= 4
&& ((i & ~0xc000003f) == 0
|| (i & ~0xf000000f) == 0
|| (i & ~0xfc000003) == 0))
return TRUE;
}
else if (TARGET_THUMB2)
{
HOST_WIDE_INT v;
v = i & 0xff;
v |= v << 16;
if (i == v || i == (v | (v << 8)))
return TRUE;
v = i & 0xff00;
v |= v << 16;
if (i == v)
return TRUE;
}
else if (TARGET_HAVE_MOVT)
{
if (i > 0xffff)
return FALSE;
else
return TRUE;
}
return FALSE;
}
int
const_ok_for_op (HOST_WIDE_INT i, enum rtx_code code)
{
if (const_ok_for_arm (i))
return 1;
switch (code)
{
case SET:
if (TARGET_HAVE_MOVT && (i & 0xffff0000) == 0)
return 1;
else
return const_ok_for_arm (ARM_SIGN_EXTEND (~i));
case PLUS:
if (TARGET_THUMB2
&& ((i & 0xfffff000) == 0
|| ((-i) & 0xfffff000) == 0))
return 1;
case COMPARE:
case EQ:
case NE:
case GT:
case LE:
case LT:
case GE:
case GEU:
case LTU:
case GTU:
case LEU:
case UNORDERED:
case ORDERED:
case UNEQ:
case UNGE:
case UNLT:
case UNGT:
case UNLE:
return const_ok_for_arm (ARM_SIGN_EXTEND (-i));
case MINUS:		
case XOR:
return 0;
case IOR:
if (TARGET_THUMB2)
return const_ok_for_arm (ARM_SIGN_EXTEND (~i));
return 0;
case AND:
return const_ok_for_arm (ARM_SIGN_EXTEND (~i));
default:
gcc_unreachable ();
}
}
int
const_ok_for_dimode_op (HOST_WIDE_INT i, enum rtx_code code)
{
HOST_WIDE_INT hi_val = (i >> 32) & 0xFFFFFFFF;
HOST_WIDE_INT lo_val = i & 0xFFFFFFFF;
rtx hi = GEN_INT (hi_val);
rtx lo = GEN_INT (lo_val);
if (TARGET_THUMB1)
return 0;
switch (code)
{
case AND:
case IOR:
case XOR:
return (const_ok_for_op (hi_val, code) || hi_val == 0xFFFFFFFF)
&& (const_ok_for_op (lo_val, code) || lo_val == 0xFFFFFFFF);
case PLUS:
return arm_not_operand (hi, SImode) && arm_add_operand (lo, SImode);
default:
return 0;
}
}
int
arm_split_constant (enum rtx_code code, machine_mode mode, rtx insn,
HOST_WIDE_INT val, rtx target, rtx source, int subtargets)
{
rtx cond;
if (insn && GET_CODE (PATTERN (insn)) == COND_EXEC)
cond = COND_EXEC_TEST (PATTERN (insn));
else
cond = NULL_RTX;
if (subtargets || code == SET
|| (REG_P (target) && REG_P (source)
&& REGNO (target) != REGNO (source)))
{
if (!cfun->machine->after_arm_reorg
&& !cond
&& (arm_gen_constant (code, mode, NULL_RTX, val, target, source,
1, 0)
> (arm_constant_limit (optimize_function_for_size_p (cfun))
+ (code != SET))))
{
if (code == SET)
{
if (TARGET_USE_MOVT)
arm_emit_movpair (target, GEN_INT (val));
else
emit_set_insn (target, GEN_INT (val));
return 1;
}
else
{
rtx temp = subtargets ? gen_reg_rtx (mode) : target;
if (TARGET_USE_MOVT)
arm_emit_movpair (temp, GEN_INT (val));
else
emit_set_insn (temp, GEN_INT (val));
if (code == MINUS)
emit_set_insn (target, gen_rtx_MINUS (mode, temp, source));
else
emit_set_insn (target,
gen_rtx_fmt_ee (code, mode, source, temp));
return 2;
}
}
}
return arm_gen_constant (code, mode, cond, val, target, source, subtargets,
1);
}
static int
optimal_immediate_sequence (enum rtx_code code, unsigned HOST_WIDE_INT val,
struct four_ints *return_sequence)
{
int best_consecutive_zeros = 0;
int i;
int best_start = 0;
int insns1, insns2;
struct four_ints tmp_sequence;
if (TARGET_ARM)
{
for (i = 0; i < 32; i += 2)
{
int consecutive_zeros = 0;
if (!(val & (3 << i)))
{
while ((i < 32) && !(val & (3 << i)))
{
consecutive_zeros += 2;
i += 2;
}
if (consecutive_zeros > best_consecutive_zeros)
{
best_consecutive_zeros = consecutive_zeros;
best_start = i - consecutive_zeros;
}
i -= 2;
}
}
}
insns1 = optimal_immediate_sequence_1 (code, val, return_sequence, best_start);
if (best_start != 0
&& ((HOST_WIDE_INT_1U << best_start) < val))
{
insns2 = optimal_immediate_sequence_1 (code, val, &tmp_sequence, 0);
if (insns2 <= insns1)
{
*return_sequence = tmp_sequence;
insns1 = insns2;
}
}
return insns1;
}
static int
optimal_immediate_sequence_1 (enum rtx_code code, unsigned HOST_WIDE_INT val,
struct four_ints *return_sequence, int i)
{
int remainder = val & 0xffffffff;
int insns = 0;
do
{
int end;
unsigned int b1, b2, b3, b4;
unsigned HOST_WIDE_INT result;
int loc;
gcc_assert (insns < 4);
if (i <= 0)
i += 32;
if (remainder & ((TARGET_ARM ? (3 << (i - 2)) : (1 << (i - 1)))))
{
loc = i;
if (i <= 12 && TARGET_THUMB2 && code == PLUS)
result = remainder;
else
{
end = i - 8;
if (end < 0)
end += 32;
result = remainder & ((0x0ff << end)
| ((i < end) ? (0xff >> (32 - end))
: 0));
i -= 8;
}
}
else
{
i -= TARGET_ARM ? 2 : 1;
continue;
}
if (TARGET_THUMB2)
{
b1 = (remainder & 0xff000000) >> 24;
b2 = (remainder & 0x00ff0000) >> 16;
b3 = (remainder & 0x0000ff00) >> 8;
b4 = remainder & 0xff;
if (loc > 24)
{
unsigned int tmp = b1 & b2 & b3 & b4;
unsigned int tmp2 = tmp + (tmp << 8) + (tmp << 16)
+ (tmp << 24);
unsigned int matching_bytes = (tmp == b1) + (tmp == b2)
+ (tmp == b3) + (tmp == b4);
if (tmp
&& (matching_bytes >= 3
|| (matching_bytes == 2
&& const_ok_for_op (remainder & ~tmp2, code))))
{
result = tmp2;
i = tmp != b1 ? 32
: tmp != b2 ? 24
: tmp != b3 ? 16
: 8;
}
else if (b1 == b3 && (!b2 || !b4
|| (remainder & 0x00ff0000 & ~result)))
{
result = remainder & 0xff00ff00;
i = 24;
}
}
else if (loc > 16)
{
if (b2 == b4)
{
result = remainder & 0x00ff00ff;
i = 16;
}
}
}
return_sequence->i[insns++] = result;
remainder &= ~result;
if (code == SET || code == MINUS)
code = PLUS;
}
while (remainder);
return insns;
}
static void
emit_constant_insn (rtx cond, rtx pattern)
{
if (cond)
pattern = gen_rtx_COND_EXEC (VOIDmode, copy_rtx (cond), pattern);
emit_insn (pattern);
}
static int
arm_gen_constant (enum rtx_code code, machine_mode mode, rtx cond,
unsigned HOST_WIDE_INT val, rtx target, rtx source,
int subtargets, int generate)
{
int can_invert = 0;
int can_negate = 0;
int final_invert = 0;
int i;
int set_sign_bit_copies = 0;
int clear_sign_bit_copies = 0;
int clear_zero_bit_copies = 0;
int set_zero_bit_copies = 0;
int insns = 0, neg_insns, inv_insns;
unsigned HOST_WIDE_INT temp1, temp2;
unsigned HOST_WIDE_INT remainder = val & 0xffffffff;
struct four_ints *immediates;
struct four_ints pos_immediates, neg_immediates, inv_immediates;
switch (code)
{
case SET:
can_invert = 1;
break;
case PLUS:
can_negate = 1;
break;
case IOR:
if (remainder == 0xffffffff)
{
if (generate)
emit_constant_insn (cond,
gen_rtx_SET (target,
GEN_INT (ARM_SIGN_EXTEND (val))));
return 1;
}
if (remainder == 0)
{
if (reload_completed && rtx_equal_p (target, source))
return 0;
if (generate)
emit_constant_insn (cond, gen_rtx_SET (target, source));
return 1;
}
break;
case AND:
if (remainder == 0)
{
if (generate)
emit_constant_insn (cond, gen_rtx_SET (target, const0_rtx));
return 1;
}
if (remainder == 0xffffffff)
{
if (reload_completed && rtx_equal_p (target, source))
return 0;
if (generate)
emit_constant_insn (cond, gen_rtx_SET (target, source));
return 1;
}
can_invert = 1;
break;
case XOR:
if (remainder == 0)
{
if (reload_completed && rtx_equal_p (target, source))
return 0;
if (generate)
emit_constant_insn (cond, gen_rtx_SET (target, source));
return 1;
}
if (remainder == 0xffffffff)
{
if (generate)
emit_constant_insn (cond,
gen_rtx_SET (target,
gen_rtx_NOT (mode, source)));
return 1;
}
final_invert = 1;
break;
case MINUS:
if (remainder == 0)
{
if (generate)
emit_constant_insn (cond,
gen_rtx_SET (target,
gen_rtx_NEG (mode, source)));
return 1;
}
if (const_ok_for_arm (val))
{
if (generate)
emit_constant_insn (cond,
gen_rtx_SET (target,
gen_rtx_MINUS (mode, GEN_INT (val),
source)));
return 1;
}
break;
default:
gcc_unreachable ();
}
if (const_ok_for_op (val, code))
{
if (generate)
emit_constant_insn (cond,
gen_rtx_SET (target,
(source
? gen_rtx_fmt_ee (code, mode, source,
GEN_INT (val))
: GEN_INT (val))));
return 1;
}
if (code == AND && (i = exact_log2 (remainder + 1)) > 0
&& (arm_arch_thumb2 || (i == 16 && arm_arch6 && mode == SImode)))
{
if (generate)
{
if (mode == SImode && i == 16)
emit_constant_insn (cond,
gen_zero_extendhisi2
(target, gen_lowpart (HImode, source)));
else
emit_constant_insn (cond,
gen_extzv_t2 (gen_lowpart (SImode, target),
gen_lowpart (SImode, source),
GEN_INT (i), const0_rtx));
}
return 1;
}
for (i = 31; i >= 0; i--)
{
if ((remainder & (1 << i)) == 0)
clear_sign_bit_copies++;
else
break;
}
for (i = 31; i >= 0; i--)
{
if ((remainder & (1 << i)) != 0)
set_sign_bit_copies++;
else
break;
}
for (i = 0; i <= 31; i++)
{
if ((remainder & (1 << i)) == 0)
clear_zero_bit_copies++;
else
break;
}
for (i = 0; i <= 31; i++)
{
if ((remainder & (1 << i)) != 0)
set_zero_bit_copies++;
else
break;
}
switch (code)
{
case SET:
if (set_sign_bit_copies > 1)
{
if (const_ok_for_arm
(temp1 = ARM_SIGN_EXTEND (remainder
<< (set_sign_bit_copies - 1))))
{
if (generate)
{
rtx new_src = subtargets ? gen_reg_rtx (mode) : target;
emit_constant_insn (cond,
gen_rtx_SET (new_src, GEN_INT (temp1)));
emit_constant_insn (cond,
gen_ashrsi3 (target, new_src,
GEN_INT (set_sign_bit_copies - 1)));
}
return 2;
}
temp1 |= (1 << (set_sign_bit_copies - 1)) - 1;
if (const_ok_for_arm (~temp1))
{
if (generate)
{
rtx new_src = subtargets ? gen_reg_rtx (mode) : target;
emit_constant_insn (cond,
gen_rtx_SET (new_src, GEN_INT (temp1)));
emit_constant_insn (cond,
gen_ashrsi3 (target, new_src,
GEN_INT (set_sign_bit_copies - 1)));
}
return 2;
}
}
if (clear_sign_bit_copies + clear_zero_bit_copies <= 16)
{
int topshift = clear_sign_bit_copies & ~1;
temp1 = ARM_SIGN_EXTEND ((remainder + (0x00800000 >> topshift))
& (0xff000000 >> topshift));
if (temp1 == 0 && topshift != 0)
temp1 = 0x80000000 >> (topshift - 1);
temp2 = ARM_SIGN_EXTEND (temp1 - remainder);
if (const_ok_for_arm (temp2))
{
if (generate)
{
rtx new_src = subtargets ? gen_reg_rtx (mode) : target;
emit_constant_insn (cond,
gen_rtx_SET (new_src, GEN_INT (temp1)));
emit_constant_insn (cond,
gen_addsi3 (target, new_src,
GEN_INT (-temp2)));
}
return 2;
}
}
if (val & 0xffff0000)
{
temp1 = remainder & 0xffff0000;
temp2 = remainder & 0x0000ffff;
for (i = 9; i < 24; i++)
{
if ((((temp2 | (temp2 << i)) & 0xffffffff) == remainder)
&& !const_ok_for_arm (temp2))
{
rtx new_src = (subtargets
? (generate ? gen_reg_rtx (mode) : NULL_RTX)
: target);
insns = arm_gen_constant (code, mode, cond, temp2, new_src,
source, subtargets, generate);
source = new_src;
if (generate)
emit_constant_insn
(cond,
gen_rtx_SET
(target,
gen_rtx_IOR (mode,
gen_rtx_ASHIFT (mode, source,
GEN_INT (i)),
source)));
return insns + 1;
}
}
for (i = 17; i < 24; i++)
{
if (((temp1 | (temp1 >> i)) == remainder)
&& !const_ok_for_arm (temp1))
{
rtx new_src = (subtargets
? (generate ? gen_reg_rtx (mode) : NULL_RTX)
: target);
insns = arm_gen_constant (code, mode, cond, temp1, new_src,
source, subtargets, generate);
source = new_src;
if (generate)
emit_constant_insn
(cond,
gen_rtx_SET (target,
gen_rtx_IOR
(mode,
gen_rtx_LSHIFTRT (mode, source,
GEN_INT (i)),
source)));
return insns + 1;
}
}
}
break;
case IOR:
case XOR:
if (subtargets
|| (reload_completed && !reg_mentioned_p (target, source)))
{
if (const_ok_for_arm (ARM_SIGN_EXTEND (~val)))
{
if (generate)
{
rtx sub = subtargets ? gen_reg_rtx (mode) : target;
emit_constant_insn (cond,
gen_rtx_SET (sub, GEN_INT (val)));
emit_constant_insn (cond,
gen_rtx_SET (target,
gen_rtx_fmt_ee (code, mode,
source, sub)));
}
return 2;
}
}
if (code == XOR)
break;
if (set_sign_bit_copies > 8
&& (val & (HOST_WIDE_INT_M1U << (32 - set_sign_bit_copies))) == val)
{
if (generate)
{
rtx sub = subtargets ? gen_reg_rtx (mode) : target;
rtx shift = GEN_INT (set_sign_bit_copies);
emit_constant_insn
(cond,
gen_rtx_SET (sub,
gen_rtx_NOT (mode,
gen_rtx_ASHIFT (mode,
source,
shift))));
emit_constant_insn
(cond,
gen_rtx_SET (target,
gen_rtx_NOT (mode,
gen_rtx_LSHIFTRT (mode, sub,
shift))));
}
return 2;
}
if (set_zero_bit_copies > 8
&& (remainder & ((1 << set_zero_bit_copies) - 1)) == remainder)
{
if (generate)
{
rtx sub = subtargets ? gen_reg_rtx (mode) : target;
rtx shift = GEN_INT (set_zero_bit_copies);
emit_constant_insn
(cond,
gen_rtx_SET (sub,
gen_rtx_NOT (mode,
gen_rtx_LSHIFTRT (mode,
source,
shift))));
emit_constant_insn
(cond,
gen_rtx_SET (target,
gen_rtx_NOT (mode,
gen_rtx_ASHIFT (mode, sub,
shift))));
}
return 2;
}
if (const_ok_for_arm (temp1 = ARM_SIGN_EXTEND (~val)))
{
if (generate)
{
rtx sub = subtargets ? gen_reg_rtx (mode) : target;
emit_constant_insn (cond,
gen_rtx_SET (sub,
gen_rtx_NOT (mode, source)));
source = sub;
if (subtargets)
sub = gen_reg_rtx (mode);
emit_constant_insn (cond,
gen_rtx_SET (sub,
gen_rtx_AND (mode, source,
GEN_INT (temp1))));
emit_constant_insn (cond,
gen_rtx_SET (target,
gen_rtx_NOT (mode, sub)));
}
return 3;
}
break;
case AND:
if (clear_sign_bit_copies >= 16 && clear_sign_bit_copies < 24)
{
HOST_WIDE_INT shift_mask = ((0xffffffff
<< (32 - clear_sign_bit_copies))
& 0xffffffff);
if ((remainder | shift_mask) != 0xffffffff)
{
HOST_WIDE_INT new_val
= ARM_SIGN_EXTEND (remainder | shift_mask);
if (generate)
{
rtx new_src = subtargets ? gen_reg_rtx (mode) : target;
insns = arm_gen_constant (AND, SImode, cond, new_val,
new_src, source, subtargets, 1);
source = new_src;
}
else
{
rtx targ = subtargets ? NULL_RTX : target;
insns = arm_gen_constant (AND, mode, cond, new_val,
targ, source, subtargets, 0);
}
}
if (generate)
{
rtx new_src = subtargets ? gen_reg_rtx (mode) : target;
rtx shift = GEN_INT (clear_sign_bit_copies);
emit_insn (gen_ashlsi3 (new_src, source, shift));
emit_insn (gen_lshrsi3 (target, new_src, shift));
}
return insns + 2;
}
if (clear_zero_bit_copies >= 16 && clear_zero_bit_copies < 24)
{
HOST_WIDE_INT shift_mask = (1 << clear_zero_bit_copies) - 1;
if ((remainder | shift_mask) != 0xffffffff)
{
HOST_WIDE_INT new_val
= ARM_SIGN_EXTEND (remainder | shift_mask);
if (generate)
{
rtx new_src = subtargets ? gen_reg_rtx (mode) : target;
insns = arm_gen_constant (AND, mode, cond, new_val,
new_src, source, subtargets, 1);
source = new_src;
}
else
{
rtx targ = subtargets ? NULL_RTX : target;
insns = arm_gen_constant (AND, mode, cond, new_val,
targ, source, subtargets, 0);
}
}
if (generate)
{
rtx new_src = subtargets ? gen_reg_rtx (mode) : target;
rtx shift = GEN_INT (clear_zero_bit_copies);
emit_insn (gen_lshrsi3 (new_src, source, shift));
emit_insn (gen_ashlsi3 (target, new_src, shift));
}
return insns + 2;
}
break;
default:
break;
}
if (code == AND)
insns = 99;
else
insns = optimal_immediate_sequence (code, remainder, &pos_immediates);
if (can_negate)
neg_insns = optimal_immediate_sequence (code, (-remainder) & 0xffffffff,
&neg_immediates);
else
neg_insns = 99;
if (can_invert || final_invert)
inv_insns = optimal_immediate_sequence (code, remainder ^ 0xffffffff,
&inv_immediates);
else
inv_insns = 99;
immediates = &pos_immediates;
if (neg_insns < insns && neg_insns <= inv_insns)
{
insns = neg_insns;
immediates = &neg_immediates;
}
else
can_negate = 0;
if ((inv_insns + 1) < insns || (!final_invert && inv_insns < insns))
{
insns = inv_insns;
immediates = &inv_immediates;
}
else
{
can_invert = 0;
final_invert = 0;
}
if (generate)
{
for (i = 0; i < insns; i++)
{
rtx new_src, temp1_rtx;
temp1 = immediates->i[i];
if (code == SET || code == MINUS)
new_src = (subtargets ? gen_reg_rtx (mode) : target);
else if ((final_invert || i < (insns - 1)) && subtargets)
new_src = gen_reg_rtx (mode);
else
new_src = target;
if (can_invert)
temp1 = ~temp1;
else if (can_negate)
temp1 = -temp1;
temp1 = trunc_int_for_mode (temp1, mode);
temp1_rtx = GEN_INT (temp1);
if (code == SET)
;
else if (code == MINUS)
temp1_rtx = gen_rtx_MINUS (mode, temp1_rtx, source);
else
temp1_rtx = gen_rtx_fmt_ee (code, mode, source, temp1_rtx);
emit_constant_insn (cond, gen_rtx_SET (new_src, temp1_rtx));
source = new_src;
if (code == SET)
{
can_negate = can_invert;
can_invert = 0;
code = PLUS;
}
else if (code == MINUS)
code = PLUS;
}
}
if (final_invert)
{
if (generate)
emit_constant_insn (cond, gen_rtx_SET (target,
gen_rtx_NOT (mode, source)));
insns++;
}
return insns;
}
static void
arm_canonicalize_comparison (int *code, rtx *op0, rtx *op1,
bool op0_preserve_value)
{
machine_mode mode;
unsigned HOST_WIDE_INT i, maxval;
mode = GET_MODE (*op0);
if (mode == VOIDmode)
mode = GET_MODE (*op1);
maxval = (HOST_WIDE_INT_1U << (GET_MODE_BITSIZE (mode) - 1)) - 1;
if (mode == DImode)
{
if (*code == GT || *code == LE
|| (!TARGET_ARM && (*code == GTU || *code == LEU)))
{
if (CONST_INT_P (*op1))
{
i = INTVAL (*op1);
switch (*code)
{
case GT:
case LE:
if (i != maxval
&& arm_const_double_by_immediates (GEN_INT (i + 1)))
{
*op1 = GEN_INT (i + 1);
*code = *code == GT ? GE : LT;
return;
}
break;
case GTU:
case LEU:
if (i != ~((unsigned HOST_WIDE_INT) 0)
&& arm_const_double_by_immediates (GEN_INT (i + 1)))
{
*op1 = GEN_INT (i + 1);
*code = *code == GTU ? GEU : LTU;
return;
}
break;
default:
gcc_unreachable ();
}
}
if (!op0_preserve_value)
{
std::swap (*op0, *op1);
*code = (int)swap_condition ((enum rtx_code)*code);
}
}
return;
}
if (mode == SImode
&& GET_CODE (*op0) == ZERO_EXTEND
&& GET_CODE (XEXP (*op0, 0)) == SUBREG
&& GET_MODE (XEXP (*op0, 0)) == QImode
&& GET_MODE (SUBREG_REG (XEXP (*op0, 0))) == SImode
&& subreg_lowpart_p (XEXP (*op0, 0))
&& *op1 == const0_rtx)
*op0 = gen_rtx_AND (SImode, SUBREG_REG (XEXP (*op0, 0)),
GEN_INT (255));
if (!CONST_INT_P (*op1)
|| const_ok_for_arm (INTVAL (*op1))
|| const_ok_for_arm (- INTVAL (*op1)))
return;
i = INTVAL (*op1);
switch (*code)
{
case EQ:
case NE:
return;
case GT:
case LE:
if (i != maxval
&& (const_ok_for_arm (i + 1) || const_ok_for_arm (-(i + 1))))
{
*op1 = GEN_INT (ARM_SIGN_EXTEND (i + 1));
*code = *code == GT ? GE : LT;
return;
}
break;
case GE:
case LT:
if (i != ~maxval
&& (const_ok_for_arm (i - 1) || const_ok_for_arm (-(i - 1))))
{
*op1 = GEN_INT (i - 1);
*code = *code == GE ? GT : LE;
return;
}
break;
case GTU:
case LEU:
if (i != ~((unsigned HOST_WIDE_INT) 0)
&& (const_ok_for_arm (i + 1) || const_ok_for_arm (-(i + 1))))
{
*op1 = GEN_INT (ARM_SIGN_EXTEND (i + 1));
*code = *code == GTU ? GEU : LTU;
return;
}
break;
case GEU:
case LTU:
if (i != 0
&& (const_ok_for_arm (i - 1) || const_ok_for_arm (-(i - 1))))
{
*op1 = GEN_INT (i - 1);
*code = *code == GEU ? GTU : LEU;
return;
}
break;
default:
gcc_unreachable ();
}
}
static rtx
arm_function_value(const_tree type, const_tree func,
bool outgoing ATTRIBUTE_UNUSED)
{
machine_mode mode;
int unsignedp ATTRIBUTE_UNUSED;
rtx r ATTRIBUTE_UNUSED;
mode = TYPE_MODE (type);
if (TARGET_AAPCS_BASED)
return aapcs_allocate_return_reg (mode, type, func);
if (INTEGRAL_TYPE_P (type))
mode = arm_promote_function_mode (type, mode, &unsignedp, func, 1);
if (arm_return_in_msb (type))
{
HOST_WIDE_INT size = int_size_in_bytes (type);
if (size % UNITS_PER_WORD != 0)
{
size += UNITS_PER_WORD - size % UNITS_PER_WORD;
mode = int_mode_for_size (size * BITS_PER_UNIT, 0).require ();
}
}
return arm_libcall_value_1 (mode);
}
struct libcall_hasher : nofree_ptr_hash <const rtx_def>
{
static inline hashval_t hash (const rtx_def *);
static inline bool equal (const rtx_def *, const rtx_def *);
static inline void remove (rtx_def *);
};
inline bool
libcall_hasher::equal (const rtx_def *p1, const rtx_def *p2)
{
return rtx_equal_p (p1, p2);
}
inline hashval_t
libcall_hasher::hash (const rtx_def *p1)
{
return hash_rtx (p1, VOIDmode, NULL, NULL, FALSE);
}
typedef hash_table<libcall_hasher> libcall_table_type;
static void
add_libcall (libcall_table_type *htab, rtx libcall)
{
*htab->find_slot (libcall, INSERT) = libcall;
}
static bool
arm_libcall_uses_aapcs_base (const_rtx libcall)
{
static bool init_done = false;
static libcall_table_type *libcall_htab = NULL;
if (!init_done)
{
init_done = true;
libcall_htab = new libcall_table_type (31);
add_libcall (libcall_htab,
convert_optab_libfunc (sfloat_optab, SFmode, SImode));
add_libcall (libcall_htab,
convert_optab_libfunc (sfloat_optab, DFmode, SImode));
add_libcall (libcall_htab,
convert_optab_libfunc (sfloat_optab, SFmode, DImode));
add_libcall (libcall_htab,
convert_optab_libfunc (sfloat_optab, DFmode, DImode));
add_libcall (libcall_htab,
convert_optab_libfunc (ufloat_optab, SFmode, SImode));
add_libcall (libcall_htab,
convert_optab_libfunc (ufloat_optab, DFmode, SImode));
add_libcall (libcall_htab,
convert_optab_libfunc (ufloat_optab, SFmode, DImode));
add_libcall (libcall_htab,
convert_optab_libfunc (ufloat_optab, DFmode, DImode));
add_libcall (libcall_htab,
convert_optab_libfunc (sext_optab, SFmode, HFmode));
add_libcall (libcall_htab,
convert_optab_libfunc (trunc_optab, HFmode, SFmode));
add_libcall (libcall_htab,
convert_optab_libfunc (sfix_optab, SImode, DFmode));
add_libcall (libcall_htab,
convert_optab_libfunc (ufix_optab, SImode, DFmode));
add_libcall (libcall_htab,
convert_optab_libfunc (sfix_optab, DImode, DFmode));
add_libcall (libcall_htab,
convert_optab_libfunc (ufix_optab, DImode, DFmode));
add_libcall (libcall_htab,
convert_optab_libfunc (sfix_optab, DImode, SFmode));
add_libcall (libcall_htab,
convert_optab_libfunc (ufix_optab, DImode, SFmode));
add_libcall (libcall_htab, optab_libfunc (add_optab, DFmode));
add_libcall (libcall_htab, optab_libfunc (sdiv_optab, DFmode));
add_libcall (libcall_htab, optab_libfunc (smul_optab, DFmode));
add_libcall (libcall_htab, optab_libfunc (neg_optab, DFmode));
add_libcall (libcall_htab, optab_libfunc (sub_optab, DFmode));
add_libcall (libcall_htab, optab_libfunc (eq_optab, DFmode));
add_libcall (libcall_htab, optab_libfunc (lt_optab, DFmode));
add_libcall (libcall_htab, optab_libfunc (le_optab, DFmode));
add_libcall (libcall_htab, optab_libfunc (ge_optab, DFmode));
add_libcall (libcall_htab, optab_libfunc (gt_optab, DFmode));
add_libcall (libcall_htab, optab_libfunc (unord_optab, DFmode));
add_libcall (libcall_htab, convert_optab_libfunc (sext_optab, DFmode,
SFmode));
add_libcall (libcall_htab, convert_optab_libfunc (trunc_optab, SFmode,
DFmode));
add_libcall (libcall_htab,
convert_optab_libfunc (trunc_optab, HFmode, DFmode));
}
return libcall && libcall_htab->find (libcall) != NULL;
}
static rtx
arm_libcall_value_1 (machine_mode mode)
{
if (TARGET_AAPCS_BASED)
return aapcs_libcall_value (mode);
else if (TARGET_IWMMXT_ABI
&& arm_vector_mode_supported_p (mode))
return gen_rtx_REG (mode, FIRST_IWMMXT_REGNUM);
else
return gen_rtx_REG (mode, ARG_REGISTER (1));
}
static rtx
arm_libcall_value (machine_mode mode, const_rtx libcall)
{
if (TARGET_AAPCS_BASED && arm_pcs_default != ARM_PCS_AAPCS
&& GET_MODE_CLASS (mode) == MODE_FLOAT)
{
if (arm_libcall_uses_aapcs_base (libcall))
return gen_rtx_REG (mode, ARG_REGISTER(1));
}
return arm_libcall_value_1 (mode);
}
static bool
arm_function_value_regno_p (const unsigned int regno)
{
if (regno == ARG_REGISTER (1)
|| (TARGET_32BIT
&& TARGET_AAPCS_BASED
&& TARGET_HARD_FLOAT
&& regno == FIRST_VFP_REGNUM)
|| (TARGET_IWMMXT_ABI
&& regno == FIRST_IWMMXT_REGNUM))
return true;
return false;
}
int
arm_apply_result_size (void)
{
int size = 16;
if (TARGET_32BIT)
{
if (TARGET_HARD_FLOAT_ABI)
size += 32;
if (TARGET_IWMMXT_ABI)
size += 8;
}
return size;
}
static bool
arm_return_in_memory (const_tree type, const_tree fntype)
{
HOST_WIDE_INT size;
size = int_size_in_bytes (type);  
if (TARGET_AAPCS_BASED)
{
if (!AGGREGATE_TYPE_P (type)
&& TREE_CODE (type) != VECTOR_TYPE
&& TREE_CODE (type) != COMPLEX_TYPE)
return false;
if (((unsigned HOST_WIDE_INT) size) <= UNITS_PER_WORD)
return false;
if (aapcs_select_return_coproc (type, fntype) >= 0)
return false;
if (TREE_CODE (type) == VECTOR_TYPE)
return (size < 0 || size > (4 * UNITS_PER_WORD));
return true;
}
if (TREE_CODE (type) == VECTOR_TYPE)
return (size < 0 || size > (4 * UNITS_PER_WORD));
if (!AGGREGATE_TYPE_P (type) &&
(TREE_CODE (type) != VECTOR_TYPE))
return false;
if (arm_abi != ARM_ABI_APCS)
{
return (size < 0 || size > UNITS_PER_WORD);
}
#ifndef ARM_WINCE
if (size < 0 || size > UNITS_PER_WORD)
return true;
if (TREE_CODE (type) == RECORD_TYPE)
{
tree field;
for (field = TYPE_FIELDS (type);
field && TREE_CODE (field) != FIELD_DECL;
field = DECL_CHAIN (field))
continue;
if (field == NULL)
return false; 
if (FLOAT_TYPE_P (TREE_TYPE (field)))
return true;
if (arm_return_in_memory (TREE_TYPE (field), NULL_TREE))
return true;
for (field = DECL_CHAIN (field);
field;
field = DECL_CHAIN (field))
{
if (TREE_CODE (field) != FIELD_DECL)
continue;
if (!DECL_BIT_FIELD_TYPE (field))
return true;
}
return false;
}
if (TREE_CODE (type) == UNION_TYPE)
{
tree field;
for (field = TYPE_FIELDS (type);
field;
field = DECL_CHAIN (field))
{
if (TREE_CODE (field) != FIELD_DECL)
continue;
if (FLOAT_TYPE_P (TREE_TYPE (field)))
return true;
if (arm_return_in_memory (TREE_TYPE (field), NULL_TREE))
return true;
}
return false;
}
#endif 
return true;
}
const struct pcs_attribute_arg
{
const char *arg;
enum arm_pcs value;
} pcs_attribute_args[] =
{
{"aapcs", ARM_PCS_AAPCS},
{"aapcs-vfp", ARM_PCS_AAPCS_VFP},
#if 0
{"aapcs-iwmmxt", ARM_PCS_AAPCS_IWMMXT},
{"atpcs", ARM_PCS_ATPCS},
{"apcs", ARM_PCS_APCS},
#endif
{NULL, ARM_PCS_UNKNOWN}
};
static enum arm_pcs
arm_pcs_from_attribute (tree attr)
{
const struct pcs_attribute_arg *ptr;
const char *arg;
if (TREE_VALUE (attr) == NULL_TREE
|| TREE_CODE (TREE_VALUE (attr)) != STRING_CST)
return ARM_PCS_UNKNOWN;
arg = TREE_STRING_POINTER (TREE_VALUE (attr));
for (ptr = pcs_attribute_args; ptr->arg != NULL; ptr++)
if (streq (arg, ptr->arg))
return ptr->value;
return ARM_PCS_UNKNOWN;
}
static enum arm_pcs
arm_get_pcs_model (const_tree type, const_tree decl)
{
bool user_convention = false;
enum arm_pcs user_pcs = arm_pcs_default;
tree attr;
gcc_assert (type);
attr = lookup_attribute ("pcs", TYPE_ATTRIBUTES (type));
if (attr)
{
user_pcs = arm_pcs_from_attribute (TREE_VALUE (attr));
user_convention = true;
}
if (TARGET_AAPCS_BASED)
{
bool base_rules = stdarg_p (type);
if (user_convention)
{
if (user_pcs > ARM_PCS_AAPCS_LOCAL)
sorry ("non-AAPCS derived PCS variant");
else if (base_rules && user_pcs != ARM_PCS_AAPCS)
error ("variadic functions must use the base AAPCS variant");
}
if (base_rules)
return ARM_PCS_AAPCS;
else if (user_convention)
return user_pcs;
else if (decl && flag_unit_at_a_time)
{
cgraph_local_info *i = cgraph_node::local_info (CONST_CAST_TREE(decl));
if (i && i->local)
return ARM_PCS_AAPCS_LOCAL;
}
}
else if (user_convention && user_pcs != arm_pcs_default)
sorry ("PCS variant");
return arm_pcs_default;
}
static void
aapcs_vfp_cum_init (CUMULATIVE_ARGS *pcum  ATTRIBUTE_UNUSED,
const_tree fntype ATTRIBUTE_UNUSED,
rtx libcall ATTRIBUTE_UNUSED,
const_tree fndecl ATTRIBUTE_UNUSED)
{
pcum->aapcs_vfp_regs_free = (1 << NUM_VFP_ARG_REGS) - 1;
pcum->aapcs_vfp_reg_alloc = 0;
}
static int
aapcs_vfp_sub_candidate (const_tree type, machine_mode *modep)
{
machine_mode mode;
HOST_WIDE_INT size;
switch (TREE_CODE (type))
{
case REAL_TYPE:
mode = TYPE_MODE (type);
if (mode != DFmode && mode != SFmode && mode != HFmode)
return -1;
if (*modep == VOIDmode)
*modep = mode;
if (*modep == mode)
return 1;
break;
case COMPLEX_TYPE:
mode = TYPE_MODE (TREE_TYPE (type));
if (mode != DFmode && mode != SFmode)
return -1;
if (*modep == VOIDmode)
*modep = mode;
if (*modep == mode)
return 2;
break;
case VECTOR_TYPE:
size = int_size_in_bytes (type);
switch (size)
{
case 8:
mode = V2SImode;
break;
case 16:
mode = V4SImode;
break;
default:
return -1;
}
if (*modep == VOIDmode)
*modep = mode;
if (*modep == mode)
return 1;
break;
case ARRAY_TYPE:
{
int count;
tree index = TYPE_DOMAIN (type);
if (!COMPLETE_TYPE_P (type)
|| TREE_CODE (TYPE_SIZE (type)) != INTEGER_CST)
return -1;
count = aapcs_vfp_sub_candidate (TREE_TYPE (type), modep);
if (count == -1
|| !index
|| !TYPE_MAX_VALUE (index)
|| !tree_fits_uhwi_p (TYPE_MAX_VALUE (index))
|| !TYPE_MIN_VALUE (index)
|| !tree_fits_uhwi_p (TYPE_MIN_VALUE (index))
|| count < 0)
return -1;
count *= (1 + tree_to_uhwi (TYPE_MAX_VALUE (index))
- tree_to_uhwi (TYPE_MIN_VALUE (index)));
if (wi::to_wide (TYPE_SIZE (type))
!= count * GET_MODE_BITSIZE (*modep))
return -1;
return count;
}
case RECORD_TYPE:
{
int count = 0;
int sub_count;
tree field;
if (!COMPLETE_TYPE_P (type)
|| TREE_CODE (TYPE_SIZE (type)) != INTEGER_CST)
return -1;
for (field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
{
if (TREE_CODE (field) != FIELD_DECL)
continue;
sub_count = aapcs_vfp_sub_candidate (TREE_TYPE (field), modep);
if (sub_count < 0)
return -1;
count += sub_count;
}
if (wi::to_wide (TYPE_SIZE (type))
!= count * GET_MODE_BITSIZE (*modep))
return -1;
return count;
}
case UNION_TYPE:
case QUAL_UNION_TYPE:
{
int count = 0;
int sub_count;
tree field;
if (!COMPLETE_TYPE_P (type)
|| TREE_CODE (TYPE_SIZE (type)) != INTEGER_CST)
return -1;
for (field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
{
if (TREE_CODE (field) != FIELD_DECL)
continue;
sub_count = aapcs_vfp_sub_candidate (TREE_TYPE (field), modep);
if (sub_count < 0)
return -1;
count = count > sub_count ? count : sub_count;
}
if (wi::to_wide (TYPE_SIZE (type))
!= count * GET_MODE_BITSIZE (*modep))
return -1;
return count;
}
default:
break;
}
return -1;
}
static bool
use_vfp_abi (enum arm_pcs pcs_variant, bool is_double)
{
if (pcs_variant == ARM_PCS_AAPCS_VFP)
{
static bool seen_thumb1_vfp = false;
if (TARGET_THUMB1 && !seen_thumb1_vfp)
{
sorry ("Thumb-1 hard-float VFP ABI");
seen_thumb1_vfp = true;
}
return true;
}
if (pcs_variant != ARM_PCS_AAPCS_LOCAL)
return false;
return (TARGET_32BIT && TARGET_HARD_FLOAT &&
(TARGET_VFP_DOUBLE || !is_double));
}
static bool
aapcs_vfp_is_call_or_return_candidate (enum arm_pcs pcs_variant,
machine_mode mode, const_tree type,
machine_mode *base_mode, int *count)
{
machine_mode new_mode = VOIDmode;
if (type)
{
int ag_count = aapcs_vfp_sub_candidate (type, &new_mode);
if (ag_count > 0 && ag_count <= 4)
*count = ag_count;
else
return false;
}
else if (GET_MODE_CLASS (mode) == MODE_FLOAT
|| GET_MODE_CLASS (mode) == MODE_VECTOR_INT
|| GET_MODE_CLASS (mode) == MODE_VECTOR_FLOAT)
{
*count = 1;
new_mode = mode;
}
else if (GET_MODE_CLASS (mode) == MODE_COMPLEX_FLOAT)
{
*count = 2;
new_mode = (mode == DCmode ? DFmode : SFmode);
}
else
return false;
if (!use_vfp_abi (pcs_variant, ARM_NUM_REGS (new_mode) > 1))
return false;
*base_mode = new_mode;
return true;
}
static bool
aapcs_vfp_is_return_candidate (enum arm_pcs pcs_variant,
machine_mode mode, const_tree type)
{
int count ATTRIBUTE_UNUSED;
machine_mode ag_mode ATTRIBUTE_UNUSED;
if (!use_vfp_abi (pcs_variant, false))
return false;
return aapcs_vfp_is_call_or_return_candidate (pcs_variant, mode, type,
&ag_mode, &count);
}
static bool
aapcs_vfp_is_call_candidate (CUMULATIVE_ARGS *pcum, machine_mode mode,
const_tree type)
{
if (!use_vfp_abi (pcum->pcs_variant, false))
return false;
return aapcs_vfp_is_call_or_return_candidate (pcum->pcs_variant, mode, type,
&pcum->aapcs_vfp_rmode,
&pcum->aapcs_vfp_rcount);
}
static bool
aapcs_vfp_allocate (CUMULATIVE_ARGS *pcum, machine_mode mode,
const_tree type  ATTRIBUTE_UNUSED)
{
int rmode_size
= MAX (GET_MODE_SIZE (pcum->aapcs_vfp_rmode), GET_MODE_SIZE (SFmode));
int shift = rmode_size / GET_MODE_SIZE (SFmode);
unsigned mask = (1 << (shift * pcum->aapcs_vfp_rcount)) - 1;
int regno;
for (regno = 0; regno < NUM_VFP_ARG_REGS; regno += shift)
if (((pcum->aapcs_vfp_regs_free >> regno) & mask) == mask)
{
pcum->aapcs_vfp_reg_alloc = mask << regno;
if (mode == BLKmode
|| (mode == TImode && ! TARGET_NEON)
|| ! arm_hard_regno_mode_ok (FIRST_VFP_REGNUM + regno, mode))
{
int i;
int rcount = pcum->aapcs_vfp_rcount;
int rshift = shift;
machine_mode rmode = pcum->aapcs_vfp_rmode;
rtx par;
if (!TARGET_NEON)
{
if (rmode == V2SImode)
rmode = DImode;
else if (rmode == V4SImode)
{
rmode = DImode;
rcount *= 2;
rshift /= 2;
}
}
par = gen_rtx_PARALLEL (mode, rtvec_alloc (rcount));
for (i = 0; i < rcount; i++)
{
rtx tmp = gen_rtx_REG (rmode,
FIRST_VFP_REGNUM + regno + i * rshift);
tmp = gen_rtx_EXPR_LIST
(VOIDmode, tmp,
GEN_INT (i * GET_MODE_SIZE (rmode)));
XVECEXP (par, 0, i) = tmp;
}
pcum->aapcs_reg = par;
}
else
pcum->aapcs_reg = gen_rtx_REG (mode, FIRST_VFP_REGNUM + regno);
return true;
}
return false;
}
static rtx
aapcs_vfp_allocate_return_reg (enum arm_pcs pcs_variant ATTRIBUTE_UNUSED,
machine_mode mode,
const_tree type ATTRIBUTE_UNUSED)
{
if (!use_vfp_abi (pcs_variant, false))
return NULL;
if (mode == BLKmode
|| (GET_MODE_CLASS (mode) == MODE_INT
&& GET_MODE_SIZE (mode) >= GET_MODE_SIZE (TImode)
&& !TARGET_NEON))
{
int count;
machine_mode ag_mode;
int i;
rtx par;
int shift;
aapcs_vfp_is_call_or_return_candidate (pcs_variant, mode, type,
&ag_mode, &count);
if (!TARGET_NEON)
{
if (ag_mode == V2SImode)
ag_mode = DImode;
else if (ag_mode == V4SImode)
{
ag_mode = DImode;
count *= 2;
}
}
shift = GET_MODE_SIZE(ag_mode) / GET_MODE_SIZE(SFmode);
par = gen_rtx_PARALLEL (mode, rtvec_alloc (count));
for (i = 0; i < count; i++)
{
rtx tmp = gen_rtx_REG (ag_mode, FIRST_VFP_REGNUM + i * shift);
tmp = gen_rtx_EXPR_LIST (VOIDmode, tmp,
GEN_INT (i * GET_MODE_SIZE (ag_mode)));
XVECEXP (par, 0, i) = tmp;
}
return par;
}
return gen_rtx_REG (mode, FIRST_VFP_REGNUM);
}
static void
aapcs_vfp_advance (CUMULATIVE_ARGS *pcum  ATTRIBUTE_UNUSED,
machine_mode mode  ATTRIBUTE_UNUSED,
const_tree type  ATTRIBUTE_UNUSED)
{
pcum->aapcs_vfp_regs_free &= ~pcum->aapcs_vfp_reg_alloc;
pcum->aapcs_vfp_reg_alloc = 0;
return;
}
#define AAPCS_CP(X)				\
{						\
aapcs_ ## X ## _cum_init,			\
aapcs_ ## X ## _is_call_candidate,		\
aapcs_ ## X ## _allocate,			\
aapcs_ ## X ## _is_return_candidate,	\
aapcs_ ## X ## _allocate_return_reg,	\
aapcs_ ## X ## _advance			\
}
static struct
{
void (*cum_init) (CUMULATIVE_ARGS *, const_tree, rtx, const_tree);
bool (*is_call_candidate) (CUMULATIVE_ARGS *, machine_mode, const_tree);
bool (*allocate) (CUMULATIVE_ARGS *, machine_mode, const_tree);
bool (*is_return_candidate) (enum arm_pcs, machine_mode, const_tree);
rtx (*allocate_return_reg) (enum arm_pcs, machine_mode, const_tree);
void (*advance) (CUMULATIVE_ARGS *, machine_mode, const_tree);
} aapcs_cp_arg_layout[ARM_NUM_COPROC_SLOTS] =
{
AAPCS_CP(vfp)
};
#undef AAPCS_CP
static int
aapcs_select_call_coproc (CUMULATIVE_ARGS *pcum, machine_mode mode,
const_tree type)
{
int i;
for (i = 0; i < ARM_NUM_COPROC_SLOTS; i++)
if (aapcs_cp_arg_layout[i].is_call_candidate (pcum, mode, type))
return i;
return -1;
}
static int
aapcs_select_return_coproc (const_tree type, const_tree fntype)
{
enum arm_pcs pcs_variant;
if (fntype)
{
const_tree fndecl = NULL_TREE;
if (TREE_CODE (fntype) == FUNCTION_DECL)
{
fndecl = fntype;
fntype = TREE_TYPE (fntype);
}
pcs_variant = arm_get_pcs_model (fntype, fndecl);
}
else
pcs_variant = arm_pcs_default;
if (pcs_variant != ARM_PCS_AAPCS)
{
int i;
for (i = 0; i < ARM_NUM_COPROC_SLOTS; i++)
if (aapcs_cp_arg_layout[i].is_return_candidate (pcs_variant,
TYPE_MODE (type),
type))
return i;
}
return -1;
}
static rtx
aapcs_allocate_return_reg (machine_mode mode, const_tree type,
const_tree fntype)
{
enum arm_pcs pcs_variant;
int unsignedp ATTRIBUTE_UNUSED;
if (fntype)
{
const_tree fndecl = NULL_TREE;
if (TREE_CODE (fntype) == FUNCTION_DECL)
{
fndecl = fntype;
fntype = TREE_TYPE (fntype);
}
pcs_variant = arm_get_pcs_model (fntype, fndecl);
}
else
pcs_variant = arm_pcs_default;
if (type && INTEGRAL_TYPE_P (type))
mode = arm_promote_function_mode (type, mode, &unsignedp, fntype, 1);
if (pcs_variant != ARM_PCS_AAPCS)
{
int i;
for (i = 0; i < ARM_NUM_COPROC_SLOTS; i++)
if (aapcs_cp_arg_layout[i].is_return_candidate (pcs_variant, mode,
type))
return aapcs_cp_arg_layout[i].allocate_return_reg (pcs_variant,
mode, type);
}
if (type && arm_return_in_msb (type))
{
HOST_WIDE_INT size = int_size_in_bytes (type);
if (size % UNITS_PER_WORD != 0)
{
size += UNITS_PER_WORD - size % UNITS_PER_WORD;
mode = int_mode_for_size (size * BITS_PER_UNIT, 0).require ();
}
}
return gen_rtx_REG (mode, R0_REGNUM);
}
static rtx
aapcs_libcall_value (machine_mode mode)
{
if (BYTES_BIG_ENDIAN && ALL_FIXED_POINT_MODE_P (mode)
&& GET_MODE_SIZE (mode) <= 4)
mode = SImode;
return aapcs_allocate_return_reg (mode, NULL_TREE, NULL_TREE);
}
static void
aapcs_layout_arg (CUMULATIVE_ARGS *pcum, machine_mode mode,
const_tree type, bool named)
{
int nregs, nregs2;
int ncrn;
if (pcum->aapcs_arg_processed)
return;
pcum->aapcs_arg_processed = true;
if (!named)
return;
if (pcum->pcs_variant != ARM_PCS_AAPCS)
{
int slot = aapcs_select_call_coproc (pcum, mode, type);
pcum->aapcs_cprc_slot = slot;
if (slot >= 0)
{
if (!pcum->aapcs_cprc_failed[slot])
{
if (aapcs_cp_arg_layout[slot].allocate (pcum, mode, type))
return;
pcum->aapcs_cprc_failed[slot] = true;
pcum->can_split = false;
}
gcc_assert (pcum->can_split == false);
return;
}
}
ncrn = pcum->aapcs_ncrn;
if (ncrn & 1)
{
int res = arm_needs_doubleword_align (mode, type);
if (res < 0 && warn_psabi && currently_expanding_gimple_stmt)
inform (input_location, "parameter passing for argument of type "
"%qT changed in GCC 7.1", type);
else if (res > 0)
ncrn++;
}
nregs = ARM_NUM_REGS2(mode, type);
gcc_assert (nregs >= 0);
nregs2 = nregs ? nregs : 1;
if (ncrn + nregs2 <= NUM_ARG_REGS)
{
pcum->aapcs_reg = gen_rtx_REG (mode, ncrn);
pcum->aapcs_next_ncrn = ncrn + nregs;
return;
}
if (ncrn < NUM_ARG_REGS && pcum->can_split)
{
pcum->aapcs_reg = gen_rtx_REG (mode, ncrn);
pcum->aapcs_next_ncrn = NUM_ARG_REGS;
pcum->aapcs_partial = (NUM_ARG_REGS - ncrn) * UNITS_PER_WORD;
return;
}
pcum->aapcs_next_ncrn = NUM_ARG_REGS;
return;
}
void
arm_init_cumulative_args (CUMULATIVE_ARGS *pcum, tree fntype,
rtx libname,
tree fndecl ATTRIBUTE_UNUSED)
{
if (fntype)
pcum->pcs_variant = arm_get_pcs_model (fntype, fndecl);
else
pcum->pcs_variant = arm_pcs_default;
if (pcum->pcs_variant <= ARM_PCS_AAPCS_LOCAL)
{
if (arm_libcall_uses_aapcs_base (libname))
pcum->pcs_variant = ARM_PCS_AAPCS;
pcum->aapcs_ncrn = pcum->aapcs_next_ncrn = 0;
pcum->aapcs_reg = NULL_RTX;
pcum->aapcs_partial = 0;
pcum->aapcs_arg_processed = false;
pcum->aapcs_cprc_slot = -1;
pcum->can_split = true;
if (pcum->pcs_variant != ARM_PCS_AAPCS)
{
int i;
for (i = 0; i < ARM_NUM_COPROC_SLOTS; i++)
{
pcum->aapcs_cprc_failed[i] = false;
aapcs_cp_arg_layout[i].cum_init (pcum, fntype, libname, fndecl);
}
}
return;
}
pcum->nregs = 0;
pcum->iwmmxt_nregs = 0;
pcum->can_split = true;
pcum->named_count = 0;
pcum->nargs = 0;
if (TARGET_REALLY_IWMMXT && fntype)
{
tree fn_arg;
for (fn_arg = TYPE_ARG_TYPES (fntype);
fn_arg;
fn_arg = TREE_CHAIN (fn_arg))
pcum->named_count += 1;
if (! pcum->named_count)
pcum->named_count = INT_MAX;
}
}
static int
arm_needs_doubleword_align (machine_mode mode, const_tree type)
{
if (!type)
return GET_MODE_ALIGNMENT (mode) > PARM_BOUNDARY;
if (!AGGREGATE_TYPE_P (type))
return TYPE_ALIGN (TYPE_MAIN_VARIANT (type)) > PARM_BOUNDARY;
if (TREE_CODE (type) == ARRAY_TYPE)
return TYPE_ALIGN (TREE_TYPE (type)) > PARM_BOUNDARY;
int ret = 0;
for (tree field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
if (DECL_ALIGN (field) > PARM_BOUNDARY)
{
if (TREE_CODE (field) == FIELD_DECL)
return 1;
else
ret = -1;
}
return ret;
}
static rtx
arm_function_arg (cumulative_args_t pcum_v, machine_mode mode,
const_tree type, bool named)
{
CUMULATIVE_ARGS *pcum = get_cumulative_args (pcum_v);
int nregs;
if (mode == VOIDmode)
return const0_rtx;
if (pcum->pcs_variant <= ARM_PCS_AAPCS_LOCAL)
{
aapcs_layout_arg (pcum, mode, type, named);
return pcum->aapcs_reg;
}
if (TARGET_IWMMXT_ABI
&& arm_vector_mode_supported_p (mode)
&& pcum->named_count > pcum->nargs + 1)
{
if (pcum->iwmmxt_nregs <= 9)
return gen_rtx_REG (mode, pcum->iwmmxt_nregs + FIRST_IWMMXT_REGNUM);
else
{
pcum->can_split = false;
return NULL_RTX;
}
}
if ((pcum->nregs & 1) && ARM_DOUBLEWORD_ALIGN)
{
int res = arm_needs_doubleword_align (mode, type);
if (res < 0 && warn_psabi)
inform (input_location, "parameter passing for argument of type "
"%qT changed in GCC 7.1", type);
else if (res > 0)
pcum->nregs++;
}
if (pcum->can_split)
nregs = 1;
else
nregs = ARM_NUM_REGS2 (mode, type);
if (!named || pcum->nregs + nregs > NUM_ARG_REGS)
return NULL_RTX;
return gen_rtx_REG (mode, pcum->nregs);
}
static unsigned int
arm_function_arg_boundary (machine_mode mode, const_tree type)
{
if (!ARM_DOUBLEWORD_ALIGN)
return PARM_BOUNDARY;
int res = arm_needs_doubleword_align (mode, type);
if (res < 0 && warn_psabi)
inform (input_location, "parameter passing for argument of type %qT "
"changed in GCC 7.1", type);
return res > 0 ? DOUBLEWORD_ALIGNMENT : PARM_BOUNDARY;
}
static int
arm_arg_partial_bytes (cumulative_args_t pcum_v, machine_mode mode,
tree type, bool named)
{
CUMULATIVE_ARGS *pcum = get_cumulative_args (pcum_v);
int nregs = pcum->nregs;
if (pcum->pcs_variant <= ARM_PCS_AAPCS_LOCAL)
{
aapcs_layout_arg (pcum, mode, type, named);
return pcum->aapcs_partial;
}
if (TARGET_IWMMXT_ABI && arm_vector_mode_supported_p (mode))
return 0;
if (NUM_ARG_REGS > nregs
&& (NUM_ARG_REGS < nregs + ARM_NUM_REGS2 (mode, type))
&& pcum->can_split)
return (NUM_ARG_REGS - nregs) * UNITS_PER_WORD;
return 0;
}
static void
arm_function_arg_advance (cumulative_args_t pcum_v, machine_mode mode,
const_tree type, bool named)
{
CUMULATIVE_ARGS *pcum = get_cumulative_args (pcum_v);
if (pcum->pcs_variant <= ARM_PCS_AAPCS_LOCAL)
{
aapcs_layout_arg (pcum, mode, type, named);
if (pcum->aapcs_cprc_slot >= 0)
{
aapcs_cp_arg_layout[pcum->aapcs_cprc_slot].advance (pcum, mode,
type);
pcum->aapcs_cprc_slot = -1;
}
pcum->aapcs_arg_processed = false;
pcum->aapcs_ncrn = pcum->aapcs_next_ncrn;
pcum->aapcs_reg = NULL_RTX;
pcum->aapcs_partial = 0;
}
else
{
pcum->nargs += 1;
if (arm_vector_mode_supported_p (mode)
&& pcum->named_count > pcum->nargs
&& TARGET_IWMMXT_ABI)
pcum->iwmmxt_nregs += 1;
else
pcum->nregs += ARM_NUM_REGS2 (mode, type);
}
}
static bool
arm_pass_by_reference (cumulative_args_t cum ATTRIBUTE_UNUSED,
machine_mode mode ATTRIBUTE_UNUSED,
const_tree type, bool named ATTRIBUTE_UNUSED)
{
return type && TREE_CODE (TYPE_SIZE (type)) != INTEGER_CST;
}

typedef enum
{
OFF,		
LONG,		
SHORT		
} arm_pragma_enum;
static arm_pragma_enum arm_pragma_long_calls = OFF;
void
arm_pr_long_calls (struct cpp_reader * pfile ATTRIBUTE_UNUSED)
{
arm_pragma_long_calls = LONG;
}
void
arm_pr_no_long_calls (struct cpp_reader * pfile ATTRIBUTE_UNUSED)
{
arm_pragma_long_calls = SHORT;
}
void
arm_pr_long_calls_off (struct cpp_reader * pfile ATTRIBUTE_UNUSED)
{
arm_pragma_long_calls = OFF;
}

static tree
arm_handle_fndecl_attribute (tree *node, tree name, tree args ATTRIBUTE_UNUSED,
int flags ATTRIBUTE_UNUSED, bool *no_add_attrs)
{
if (TREE_CODE (*node) != FUNCTION_DECL)
{
warning (OPT_Wattributes, "%qE attribute only applies to functions",
name);
*no_add_attrs = true;
}
return NULL_TREE;
}
static tree
arm_handle_isr_attribute (tree *node, tree name, tree args, int flags,
bool *no_add_attrs)
{
if (DECL_P (*node))
{
if (TREE_CODE (*node) != FUNCTION_DECL)
{
warning (OPT_Wattributes, "%qE attribute only applies to functions",
name);
*no_add_attrs = true;
}
}
else
{
if (TREE_CODE (*node) == FUNCTION_TYPE
|| TREE_CODE (*node) == METHOD_TYPE)
{
if (arm_isr_value (args) == ARM_FT_UNKNOWN)
{
warning (OPT_Wattributes, "%qE attribute ignored",
name);
*no_add_attrs = true;
}
}
else if (TREE_CODE (*node) == POINTER_TYPE
&& (TREE_CODE (TREE_TYPE (*node)) == FUNCTION_TYPE
|| TREE_CODE (TREE_TYPE (*node)) == METHOD_TYPE)
&& arm_isr_value (args) != ARM_FT_UNKNOWN)
{
*node = build_variant_type_copy (*node);
TREE_TYPE (*node) = build_type_attribute_variant
(TREE_TYPE (*node),
tree_cons (name, args, TYPE_ATTRIBUTES (TREE_TYPE (*node))));
*no_add_attrs = true;
}
else
{
if (flags & ((int) ATTR_FLAG_DECL_NEXT
| (int) ATTR_FLAG_FUNCTION_NEXT
| (int) ATTR_FLAG_ARRAY_NEXT))
{
*no_add_attrs = true;
return tree_cons (name, args, NULL_TREE);
}
else
{
warning (OPT_Wattributes, "%qE attribute ignored",
name);
}
}
}
return NULL_TREE;
}
static tree
arm_handle_pcs_attribute (tree *node ATTRIBUTE_UNUSED, tree name, tree args,
int flags ATTRIBUTE_UNUSED, bool *no_add_attrs)
{
if (arm_pcs_from_attribute (args) == ARM_PCS_UNKNOWN)
{
warning (OPT_Wattributes, "%qE attribute ignored", name);
*no_add_attrs = true;
}
return NULL_TREE;
}
#if TARGET_DLLIMPORT_DECL_ATTRIBUTES
static tree
arm_handle_notshared_attribute (tree *node,
tree name ATTRIBUTE_UNUSED,
tree args ATTRIBUTE_UNUSED,
int flags ATTRIBUTE_UNUSED,
bool *no_add_attrs)
{
tree decl = TYPE_NAME (*node);
if (decl)
{
DECL_VISIBILITY (decl) = VISIBILITY_HIDDEN;
DECL_VISIBILITY_SPECIFIED (decl) = 1;
*no_add_attrs = false;
}
return NULL_TREE;
}
#endif
static bool
cmse_func_args_or_return_in_stack (tree fndecl, tree name, tree fntype)
{
function_args_iterator args_iter;
CUMULATIVE_ARGS args_so_far_v;
cumulative_args_t args_so_far;
bool first_param = true;
tree arg_type, prev_arg_type = NULL_TREE, ret_type;
arm_init_cumulative_args (&args_so_far_v, fntype, NULL_RTX, fndecl);
args_so_far = pack_cumulative_args (&args_so_far_v);
FOREACH_FUNCTION_ARGS (fntype, arg_type, args_iter)
{
rtx arg_rtx;
machine_mode arg_mode = TYPE_MODE (arg_type);
prev_arg_type = arg_type;
if (VOID_TYPE_P (arg_type))
continue;
if (!first_param)
arm_function_arg_advance (args_so_far, arg_mode, arg_type, true);
arg_rtx = arm_function_arg (args_so_far, arg_mode, arg_type, true);
if (!arg_rtx
|| arm_arg_partial_bytes (args_so_far, arg_mode, arg_type, true))
{
error ("%qE attribute not available to functions with arguments "
"passed on the stack", name);
return true;
}
first_param = false;
}
if (prev_arg_type != NULL_TREE && !VOID_TYPE_P (prev_arg_type))
{
error ("%qE attribute not available to functions with variable number "
"of arguments", name);
return true;
}
ret_type = TREE_TYPE (fntype);
if (arm_return_in_memory (ret_type, fntype))
{
error ("%qE attribute not available to functions that return value on "
"the stack", name);
return true;
}
return false;
}
static tree
arm_handle_cmse_nonsecure_entry (tree *node, tree name,
tree ,
int ,
bool *no_add_attrs)
{
tree fndecl;
if (!use_cmse)
{
*no_add_attrs = true;
warning (OPT_Wattributes, "%qE attribute ignored without -mcmse option.",
name);
return NULL_TREE;
}
if (TREE_CODE (*node) != FUNCTION_DECL)
{
warning (OPT_Wattributes, "%qE attribute only applies to functions",
name);
*no_add_attrs = true;
return NULL_TREE;
}
fndecl = *node;
if (!TREE_PUBLIC (fndecl))
{
warning (OPT_Wattributes, "%qE attribute has no effect on functions "
"with static linkage", name);
*no_add_attrs = true;
return NULL_TREE;
}
*no_add_attrs |= cmse_func_args_or_return_in_stack (fndecl, name,
TREE_TYPE (fndecl));
return NULL_TREE;
}
static tree
arm_handle_cmse_nonsecure_call (tree *node, tree name,
tree ,
int ,
bool *no_add_attrs)
{
tree decl = NULL_TREE, fntype = NULL_TREE;
tree type;
if (!use_cmse)
{
*no_add_attrs = true;
warning (OPT_Wattributes, "%qE attribute ignored without -mcmse option.",
name);
return NULL_TREE;
}
if (TREE_CODE (*node) == VAR_DECL || TREE_CODE (*node) == TYPE_DECL)
{
decl = *node;
fntype = TREE_TYPE (decl);
}
while (fntype != NULL_TREE && TREE_CODE (fntype) == POINTER_TYPE)
fntype = TREE_TYPE (fntype);
if (!decl || TREE_CODE (fntype) != FUNCTION_TYPE)
{
warning (OPT_Wattributes, "%qE attribute only applies to base type of a "
"function pointer", name);
*no_add_attrs = true;
return NULL_TREE;
}
*no_add_attrs |= cmse_func_args_or_return_in_stack (NULL, name, fntype);
if (*no_add_attrs)
return NULL_TREE;
type = TREE_TYPE (decl);
type = build_distinct_type_copy (type);
TREE_TYPE (decl) = type;
fntype = type;
while (TREE_CODE (fntype) != FUNCTION_TYPE)
{
type = fntype;
fntype = TREE_TYPE (fntype);
fntype = build_distinct_type_copy (fntype);
TREE_TYPE (type) = fntype;
}
tree attrs = tree_cons (get_identifier ("cmse_nonsecure_call"), NULL_TREE,
TYPE_ATTRIBUTES (fntype));
TYPE_ATTRIBUTES (fntype) = attrs;
return NULL_TREE;
}
static int
arm_comp_type_attributes (const_tree type1, const_tree type2)
{
int l1, l2, s1, s2;
if (TREE_CODE (type1) != FUNCTION_TYPE)
return 1;
l1 = lookup_attribute ("long_call", TYPE_ATTRIBUTES (type1)) != NULL;
l2 = lookup_attribute ("long_call", TYPE_ATTRIBUTES (type2)) != NULL;
s1 = lookup_attribute ("short_call", TYPE_ATTRIBUTES (type1)) != NULL;
s2 = lookup_attribute ("short_call", TYPE_ATTRIBUTES (type2)) != NULL;
if (l1 | l2 | s1 | s2)
{
if ((l1 != l2) || (s1 != s2))
return 0;
if ((l1 & s2) || (l2 & s1))
return 0;
}
l1 = lookup_attribute ("isr", TYPE_ATTRIBUTES (type1)) != NULL;
if (! l1)
l1 = lookup_attribute ("interrupt", TYPE_ATTRIBUTES (type1)) != NULL;
l2 = lookup_attribute ("isr", TYPE_ATTRIBUTES (type2)) != NULL;
if (! l2)
l1 = lookup_attribute ("interrupt", TYPE_ATTRIBUTES (type2)) != NULL;
if (l1 != l2)
return 0;
l1 = lookup_attribute ("cmse_nonsecure_call",
TYPE_ATTRIBUTES (type1)) != NULL;
l2 = lookup_attribute ("cmse_nonsecure_call",
TYPE_ATTRIBUTES (type2)) != NULL;
if (l1 != l2)
return 0;
return 1;
}
static void
arm_set_default_type_attributes (tree type)
{
if (TREE_CODE (type) == FUNCTION_TYPE || TREE_CODE (type) == METHOD_TYPE)
{
tree type_attr_list, attr_name;
type_attr_list = TYPE_ATTRIBUTES (type);
if (arm_pragma_long_calls == LONG)
attr_name = get_identifier ("long_call");
else if (arm_pragma_long_calls == SHORT)
attr_name = get_identifier ("short_call");
else
return;
type_attr_list = tree_cons (attr_name, NULL_TREE, type_attr_list);
TYPE_ATTRIBUTES (type) = type_attr_list;
}
}

static bool
arm_function_in_section_p (tree decl, section *section)
{
if (!decl_binds_to_current_def_p (decl))
return false;
if (!DECL_SECTION_NAME (decl))
{
if (flag_function_sections || DECL_COMDAT_GROUP (decl))
return false;
}
return function_section (decl) == section;
}
bool
arm_is_long_call_p (tree decl)
{
tree attrs;
if (!decl)
return TARGET_LONG_CALLS;
attrs = TYPE_ATTRIBUTES (TREE_TYPE (decl));
if (lookup_attribute ("short_call", attrs))
return false;
if (!flag_reorder_blocks_and_partition
&& TREE_CODE (decl) == FUNCTION_DECL
&& arm_function_in_section_p (decl, current_function_section ()))
return false;
if (lookup_attribute ("long_call", attrs))
return true;
return TARGET_LONG_CALLS;
}
static bool
arm_function_ok_for_sibcall (tree decl, tree exp)
{
unsigned long func_type;
if (cfun->machine->sibcall_blocked)
return false;
if (TARGET_THUMB1)
return false;
if (TARGET_VXWORKS_RTP && flag_pic && decl && !targetm.binds_local_p (decl))
return false;
if (TARGET_APCS_FRAME && TARGET_ARM
&& TARGET_HARD_FLOAT
&& decl && arm_is_long_call_p (decl))
return false;
if (TARGET_INTERWORK && decl && TREE_PUBLIC (decl)
&& !TREE_ASM_WRITTEN (decl))
return false;
func_type = arm_current_func_type ();
if (IS_INTERRUPT (func_type))
return false;
if (IS_CMSE_ENTRY (arm_current_func_type ()))
return false;
if (TREE_CODE (exp) == CALL_EXPR)
{
tree fntype = TREE_TYPE (TREE_TYPE (CALL_EXPR_FN (exp)));
if (lookup_attribute ("cmse_nonsecure_call", TYPE_ATTRIBUTES (fntype)))
return false;
}
if (!VOID_TYPE_P (TREE_TYPE (DECL_RESULT (cfun->decl))))
{
rtx a, b;
tree decl_or_type = decl;
if (!decl)
decl_or_type = TREE_TYPE (TREE_TYPE (CALL_EXPR_FN (exp)));
a = arm_function_value (TREE_TYPE (exp), decl_or_type, false);
b = arm_function_value (TREE_TYPE (DECL_RESULT (cfun->decl)),
cfun->decl, false);
if (!rtx_equal_p (a, b))
return false;
}
if (IS_STACKALIGN (func_type))
return false;
if (TARGET_AAPCS_BASED
&& arm_abi == ARM_ABI_AAPCS
&& decl
&& DECL_WEAK (decl))
return false;
if (!decl && CALL_EXPR_BY_DESCRIPTOR (exp) && !flag_trampolines)
{
tree fntype = TREE_TYPE (TREE_TYPE (CALL_EXPR_FN (exp)));
CUMULATIVE_ARGS cum;
cumulative_args_t cum_v;
arm_init_cumulative_args (&cum, fntype, NULL_RTX, NULL_TREE);
cum_v = pack_cumulative_args (&cum);
for (tree t = TYPE_ARG_TYPES (fntype); t; t = TREE_CHAIN (t))
{
tree type = TREE_VALUE (t);
if (!VOID_TYPE_P (type))
arm_function_arg_advance (cum_v, TYPE_MODE (type), type, true);
}
if (!arm_function_arg (cum_v, SImode, integer_type_node, true))
return false;
}
return true;
}

int
legitimate_pic_operand_p (rtx x)
{
if (GET_CODE (x) == SYMBOL_REF
|| (GET_CODE (x) == CONST
&& GET_CODE (XEXP (x, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == SYMBOL_REF))
return 0;
return 1;
}
static void
require_pic_register (void)
{
if (!crtl->uses_pic_offset_table)
{
gcc_assert (can_create_pseudo_p ());
if (arm_pic_register != INVALID_REGNUM
&& !(TARGET_THUMB1 && arm_pic_register > LAST_LO_REGNUM))
{
if (!cfun->machine->pic_reg)
cfun->machine->pic_reg = gen_rtx_REG (Pmode, arm_pic_register);
if (current_ir_type () != IR_GIMPLE || currently_expanding_to_rtl)
crtl->uses_pic_offset_table = 1;
}
else
{
rtx_insn *seq, *insn;
if (!cfun->machine->pic_reg)
cfun->machine->pic_reg = gen_reg_rtx (Pmode);
if (current_ir_type () != IR_GIMPLE || currently_expanding_to_rtl)
{
crtl->uses_pic_offset_table = 1;
start_sequence ();
if (TARGET_THUMB1 && arm_pic_register != INVALID_REGNUM
&& arm_pic_register > LAST_LO_REGNUM)
emit_move_insn (cfun->machine->pic_reg,
gen_rtx_REG (Pmode, arm_pic_register));
else
arm_load_pic_register (0UL);
seq = get_insns ();
end_sequence ();
for (insn = seq; insn; insn = NEXT_INSN (insn))
if (INSN_P (insn))
INSN_LOCATION (insn) = prologue_location;
insert_insn_on_edge (seq,
single_succ_edge (ENTRY_BLOCK_PTR_FOR_FN (cfun)));
}
}
}
}
rtx
legitimize_pic_address (rtx orig, machine_mode mode, rtx reg)
{
if (GET_CODE (orig) == SYMBOL_REF
|| GET_CODE (orig) == LABEL_REF)
{
if (reg == 0)
{
gcc_assert (can_create_pseudo_p ());
reg = gen_reg_rtx (Pmode);
}
rtx_insn *insn;
if ((GET_CODE (orig) == LABEL_REF
|| (GET_CODE (orig) == SYMBOL_REF
&& SYMBOL_REF_LOCAL_P (orig)
&& (SYMBOL_REF_DECL (orig)
? !DECL_WEAK (SYMBOL_REF_DECL (orig)) : 1)))
&& NEED_GOT_RELOC
&& arm_pic_data_is_text_relative)
insn = arm_pic_static_addr (orig, reg);
else
{
rtx pat;
rtx mem;
require_pic_register ();
pat = gen_calculate_pic_address (reg, cfun->machine->pic_reg, orig);
mem = SET_SRC (pat);
gcc_assert (MEM_P (mem) && !MEM_VOLATILE_P (mem));
MEM_READONLY_P (mem) = 1;
MEM_NOTRAP_P (mem) = 1;
insn = emit_insn (pat);
}
set_unique_reg_note (insn, REG_EQUAL, orig);
return reg;
}
else if (GET_CODE (orig) == CONST)
{
rtx base, offset;
if (GET_CODE (XEXP (orig, 0)) == PLUS
&& XEXP (XEXP (orig, 0), 0) == cfun->machine->pic_reg)
return orig;
if (GET_CODE (XEXP (orig, 0)) == UNSPEC
&& XINT (XEXP (orig, 0), 1) == UNSPEC_TLS)
return orig;
if (GET_CODE (XEXP (orig, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (orig, 0), 0)) == UNSPEC
&& XINT (XEXP (XEXP (orig, 0), 0), 1) == UNSPEC_TLS)
{
gcc_assert (CONST_INT_P (XEXP (XEXP (orig, 0), 1)));
return orig;
}
if (reg == 0)
{
gcc_assert (can_create_pseudo_p ());
reg = gen_reg_rtx (Pmode);
}
gcc_assert (GET_CODE (XEXP (orig, 0)) == PLUS);
base = legitimize_pic_address (XEXP (XEXP (orig, 0), 0), Pmode, reg);
offset = legitimize_pic_address (XEXP (XEXP (orig, 0), 1), Pmode,
base == reg ? 0 : reg);
if (CONST_INT_P (offset))
{
if (!arm_legitimate_index_p (mode, offset, SET, 0))
{
gcc_assert (can_create_pseudo_p ());
offset = force_reg (Pmode, offset);
}
if (CONST_INT_P (offset))
return plus_constant (Pmode, base, INTVAL (offset));
}
if (GET_MODE_SIZE (mode) > 4
&& (GET_MODE_CLASS (mode) == MODE_INT
|| TARGET_SOFT_FLOAT))
{
emit_insn (gen_addsi3 (reg, base, offset));
return reg;
}
return gen_rtx_PLUS (Pmode, base, offset);
}
return orig;
}
static int
thumb_find_work_register (unsigned long pushed_regs_mask)
{
int reg;
for (reg = LAST_ARG_REGNUM; reg >= 0; reg --)
if (!df_regs_ever_live_p (reg))
return reg;
if (cfun->machine->uses_anonymous_args
&& crtl->args.pretend_args_size > 0)
return LAST_ARG_REGNUM;
if (! cfun->machine->uses_anonymous_args
&& crtl->args.size >= 0
&& crtl->args.size <= (LAST_ARG_REGNUM * UNITS_PER_WORD)
&& (TARGET_AAPCS_BASED
? crtl->args.info.aapcs_ncrn < 4
: crtl->args.info.nregs < 4))
return LAST_ARG_REGNUM;
for (reg = LAST_LO_REGNUM; reg > LAST_ARG_REGNUM; reg --)
if (pushed_regs_mask & (1 << reg))
return reg;
if (TARGET_THUMB2)
{
for (reg = FIRST_HI_REGNUM; reg < 15; reg ++)
if (pushed_regs_mask & (1 << reg))
return reg;
}
gcc_unreachable ();
}
static GTY(()) int pic_labelno;
void
arm_load_pic_register (unsigned long saved_regs ATTRIBUTE_UNUSED)
{
rtx l1, labelno, pic_tmp, pic_rtx, pic_reg;
if (crtl->uses_pic_offset_table == 0 || TARGET_SINGLE_PIC_BASE)
return;
gcc_assert (flag_pic);
pic_reg = cfun->machine->pic_reg;
if (TARGET_VXWORKS_RTP)
{
pic_rtx = gen_rtx_SYMBOL_REF (Pmode, VXWORKS_GOTT_BASE);
pic_rtx = gen_rtx_CONST (Pmode, pic_rtx);
emit_insn (gen_pic_load_addr_32bit (pic_reg, pic_rtx));
emit_insn (gen_rtx_SET (pic_reg, gen_rtx_MEM (Pmode, pic_reg)));
pic_tmp = gen_rtx_SYMBOL_REF (Pmode, VXWORKS_GOTT_INDEX);
emit_insn (gen_pic_offset_arm (pic_reg, pic_reg, pic_tmp));
}
else
{
labelno = GEN_INT (pic_labelno++);
l1 = gen_rtx_UNSPEC (Pmode, gen_rtvec (1, labelno), UNSPEC_PIC_LABEL);
l1 = gen_rtx_CONST (VOIDmode, l1);
pic_rtx = plus_constant (Pmode, l1, TARGET_ARM ? 8 : 4);
pic_rtx = gen_rtx_UNSPEC (Pmode, gen_rtvec (1, pic_rtx),
UNSPEC_GOTSYM_OFF);
pic_rtx = gen_rtx_CONST (Pmode, pic_rtx);
if (TARGET_32BIT)
{
emit_insn (gen_pic_load_addr_unified (pic_reg, pic_rtx, labelno));
}
else 
{
if (arm_pic_register != INVALID_REGNUM
&& REGNO (pic_reg) > LAST_LO_REGNUM)
{
pic_tmp = gen_rtx_REG (SImode,
thumb_find_work_register (saved_regs));
emit_insn (gen_pic_load_addr_thumb1 (pic_tmp, pic_rtx));
emit_insn (gen_movsi (pic_offset_table_rtx, pic_tmp));
emit_insn (gen_pic_add_dot_plus_four (pic_reg, pic_reg, labelno));
}
else if (arm_pic_register != INVALID_REGNUM
&& arm_pic_register > LAST_LO_REGNUM
&& REGNO (pic_reg) <= LAST_LO_REGNUM)
{
emit_insn (gen_pic_load_addr_unified (pic_reg, pic_rtx, labelno));
emit_move_insn (gen_rtx_REG (Pmode, arm_pic_register), pic_reg);
emit_use (gen_rtx_REG (Pmode, arm_pic_register));
}
else
emit_insn (gen_pic_load_addr_unified (pic_reg, pic_rtx, labelno));
}
}
emit_use (pic_reg);
}
static rtx_insn *
arm_pic_static_addr (rtx orig, rtx reg)
{
rtx l1, labelno, offset_rtx;
gcc_assert (flag_pic);
labelno = GEN_INT (pic_labelno++);
l1 = gen_rtx_UNSPEC (Pmode, gen_rtvec (1, labelno), UNSPEC_PIC_LABEL);
l1 = gen_rtx_CONST (VOIDmode, l1);
offset_rtx = plus_constant (Pmode, l1, TARGET_ARM ? 8 : 4);
offset_rtx = gen_rtx_UNSPEC (Pmode, gen_rtvec (2, orig, offset_rtx),
UNSPEC_SYMBOL_OFFSET);
offset_rtx = gen_rtx_CONST (Pmode, offset_rtx);
return emit_insn (gen_pic_load_addr_unified (reg, offset_rtx, labelno));
}
static int
arm_address_register_rtx_p (rtx x, int strict_p)
{
int regno;
if (!REG_P (x))
return 0;
regno = REGNO (x);
if (strict_p)
return ARM_REGNO_OK_FOR_BASE_P (regno);
return (regno <= LAST_ARM_REGNUM
|| regno >= FIRST_PSEUDO_REGISTER
|| regno == FRAME_POINTER_REGNUM
|| regno == ARG_POINTER_REGNUM);
}
static int
pcrel_constant_p (rtx x)
{
if (GET_CODE (x) == MINUS)
return symbol_mentioned_p (XEXP (x, 0)) && label_mentioned_p (XEXP (x, 1));
return FALSE;
}
static bool
will_be_in_index_register (const_rtx x)
{
return GET_CODE (x) == UNSPEC && (XINT (x, 1) == UNSPEC_PIC_SYM);
}
int
arm_legitimate_address_outer_p (machine_mode mode, rtx x, RTX_CODE outer,
int strict_p)
{
bool use_ldrd;
enum rtx_code code = GET_CODE (x);
if (arm_address_register_rtx_p (x, strict_p))
return 1;
use_ldrd = (TARGET_LDRD
&& (mode == DImode || mode == DFmode));
if (code == POST_INC || code == PRE_DEC
|| ((code == PRE_INC || code == POST_DEC)
&& (use_ldrd || GET_MODE_SIZE (mode) <= 4)))
return arm_address_register_rtx_p (XEXP (x, 0), strict_p);
else if ((code == POST_MODIFY || code == PRE_MODIFY)
&& arm_address_register_rtx_p (XEXP (x, 0), strict_p)
&& GET_CODE (XEXP (x, 1)) == PLUS
&& rtx_equal_p (XEXP (XEXP (x, 1), 0), XEXP (x, 0)))
{
rtx addend = XEXP (XEXP (x, 1), 1);
if (use_ldrd
&& GET_CODE (x) == POST_MODIFY
&& REG_P (addend))
return 0;
return ((use_ldrd || GET_MODE_SIZE (mode) <= 4)
&& arm_legitimate_index_p (mode, addend, outer, strict_p));
}
else if (reload_completed
&& (code == LABEL_REF
|| (code == CONST
&& GET_CODE (XEXP (x, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == LABEL_REF
&& CONST_INT_P (XEXP (XEXP (x, 0), 1)))))
return 1;
else if (mode == TImode || (TARGET_NEON && VALID_NEON_STRUCT_MODE (mode)))
return 0;
else if (code == PLUS)
{
rtx xop0 = XEXP (x, 0);
rtx xop1 = XEXP (x, 1);
return ((arm_address_register_rtx_p (xop0, strict_p)
&& ((CONST_INT_P (xop1)
&& arm_legitimate_index_p (mode, xop1, outer, strict_p))
|| (!strict_p && will_be_in_index_register (xop1))))
|| (arm_address_register_rtx_p (xop1, strict_p)
&& arm_legitimate_index_p (mode, xop0, outer, strict_p)));
}
#if 0
else if (GET_CODE (x) == MINUS)
{
rtx xop0 = XEXP (x, 0);
rtx xop1 = XEXP (x, 1);
return (arm_address_register_rtx_p (xop0, strict_p)
&& arm_legitimate_index_p (mode, xop1, outer, strict_p));
}
#endif
else if (GET_MODE_CLASS (mode) != MODE_FLOAT
&& code == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (x)
&& ! (flag_pic
&& symbol_mentioned_p (get_pool_constant (x))
&& ! pcrel_constant_p (get_pool_constant (x))))
return 1;
return 0;
}
static bool
can_avoid_literal_pool_for_label_p (rtx x)
{
if (arm_disable_literal_pool && GET_CODE (x) == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (x))
return 1;
return 0;
}
static int
thumb2_legitimate_address_p (machine_mode mode, rtx x, int strict_p)
{
bool use_ldrd;
enum rtx_code code = GET_CODE (x);
if (arm_address_register_rtx_p (x, strict_p))
return 1;
use_ldrd = (TARGET_LDRD
&& (mode == DImode || mode == DFmode));
if (code == POST_INC || code == PRE_DEC
|| ((code == PRE_INC || code == POST_DEC)
&& (use_ldrd || GET_MODE_SIZE (mode) <= 4)))
return arm_address_register_rtx_p (XEXP (x, 0), strict_p);
else if ((code == POST_MODIFY || code == PRE_MODIFY)
&& arm_address_register_rtx_p (XEXP (x, 0), strict_p)
&& GET_CODE (XEXP (x, 1)) == PLUS
&& rtx_equal_p (XEXP (XEXP (x, 1), 0), XEXP (x, 0)))
{
rtx addend = XEXP (XEXP (x, 1), 1);
HOST_WIDE_INT offset;
if (!CONST_INT_P (addend))
return 0;
offset = INTVAL(addend);
if (GET_MODE_SIZE (mode) <= 4)
return (offset > -256 && offset < 256);
return (use_ldrd && offset > -1024 && offset < 1024
&& (offset & 3) == 0);
}
else if (reload_completed
&& (code == LABEL_REF
|| (code == CONST
&& GET_CODE (XEXP (x, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == LABEL_REF
&& CONST_INT_P (XEXP (XEXP (x, 0), 1)))))
return 1;
else if (mode == TImode || (TARGET_NEON && VALID_NEON_STRUCT_MODE (mode)))
return 0;
else if (code == PLUS)
{
rtx xop0 = XEXP (x, 0);
rtx xop1 = XEXP (x, 1);
return ((arm_address_register_rtx_p (xop0, strict_p)
&& (thumb2_legitimate_index_p (mode, xop1, strict_p)
|| (!strict_p && will_be_in_index_register (xop1))))
|| (arm_address_register_rtx_p (xop1, strict_p)
&& thumb2_legitimate_index_p (mode, xop0, strict_p)));
}
else if (can_avoid_literal_pool_for_label_p (x))
return 0;
else if (GET_MODE_CLASS (mode) != MODE_FLOAT
&& code == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (x)
&& ! (flag_pic
&& symbol_mentioned_p (get_pool_constant (x))
&& ! pcrel_constant_p (get_pool_constant (x))))
return 1;
return 0;
}
static int
arm_legitimate_index_p (machine_mode mode, rtx index, RTX_CODE outer,
int strict_p)
{
HOST_WIDE_INT range;
enum rtx_code code = GET_CODE (index);
if (TARGET_HARD_FLOAT
&& (mode == SFmode || mode == DFmode))
return (code == CONST_INT && INTVAL (index) < 1024
&& INTVAL (index) > -1024
&& (INTVAL (index) & 3) == 0);
if (TARGET_NEON && VALID_NEON_QREG_MODE (mode))
return (code == CONST_INT
&& INTVAL (index) < 1016
&& INTVAL (index) > -1024
&& (INTVAL (index) & 3) == 0);
if (TARGET_NEON && VALID_NEON_DREG_MODE (mode))
return (code == CONST_INT
&& INTVAL (index) < 1024
&& INTVAL (index) > -1024
&& (INTVAL (index) & 3) == 0);
if (TARGET_REALLY_IWMMXT && VALID_IWMMXT_REG_MODE (mode))
return (code == CONST_INT
&& INTVAL (index) < 1024
&& INTVAL (index) > -1024
&& (INTVAL (index) & 3) == 0);
if (arm_address_register_rtx_p (index, strict_p)
&& (GET_MODE_SIZE (mode) <= 4))
return 1;
if (mode == DImode || mode == DFmode)
{
if (code == CONST_INT)
{
HOST_WIDE_INT val = INTVAL (index);
if (TARGET_LDRD)
return val > -256 && val < 256;
else
return val > -4096 && val < 4092;
}
return TARGET_LDRD && arm_address_register_rtx_p (index, strict_p);
}
if (GET_MODE_SIZE (mode) <= 4
&& ! (arm_arch4
&& (mode == HImode
|| mode == HFmode
|| (mode == QImode && outer == SIGN_EXTEND))))
{
if (code == MULT)
{
rtx xiop0 = XEXP (index, 0);
rtx xiop1 = XEXP (index, 1);
return ((arm_address_register_rtx_p (xiop0, strict_p)
&& power_of_two_operand (xiop1, SImode))
|| (arm_address_register_rtx_p (xiop1, strict_p)
&& power_of_two_operand (xiop0, SImode)));
}
else if (code == LSHIFTRT || code == ASHIFTRT
|| code == ASHIFT || code == ROTATERT)
{
rtx op = XEXP (index, 1);
return (arm_address_register_rtx_p (XEXP (index, 0), strict_p)
&& CONST_INT_P (op)
&& INTVAL (op) > 0
&& INTVAL (op) <= 31);
}
}
if (arm_arch4)
{
if (mode == HImode
|| mode == HFmode
|| (outer == SIGN_EXTEND && mode == QImode))
range = 256;
else
range = 4096;
}
else
range = (mode == HImode || mode == HFmode) ? 4095 : 4096;
return (code == CONST_INT
&& INTVAL (index) < range
&& INTVAL (index) > -range);
}
static bool
thumb2_index_mul_operand (rtx op)
{
HOST_WIDE_INT val;
if (!CONST_INT_P (op))
return false;
val = INTVAL(op);
return (val == 1 || val == 2 || val == 4 || val == 8);
}
static int
thumb2_legitimate_index_p (machine_mode mode, rtx index, int strict_p)
{
enum rtx_code code = GET_CODE (index);
if (TARGET_HARD_FLOAT
&& (mode == SFmode || mode == DFmode))
return (code == CONST_INT && INTVAL (index) < 1024
&& INTVAL (index) > -256
&& (INTVAL (index) & 3) == 0);
if (TARGET_REALLY_IWMMXT && VALID_IWMMXT_REG_MODE (mode))
{
if (!TARGET_LDRD || mode != DImode)
return (code == CONST_INT
&& INTVAL (index) < 1024
&& INTVAL (index) > -1024
&& (INTVAL (index) & 3) == 0);
}
if (TARGET_NEON && VALID_NEON_QREG_MODE (mode))
return (code == CONST_INT
&& INTVAL (index) < 1016
&& INTVAL (index) > -1024
&& (INTVAL (index) & 3) == 0);
if (TARGET_NEON && VALID_NEON_DREG_MODE (mode))
return (code == CONST_INT
&& INTVAL (index) < 1024
&& INTVAL (index) > -1024
&& (INTVAL (index) & 3) == 0);
if (arm_address_register_rtx_p (index, strict_p)
&& (GET_MODE_SIZE (mode) <= 4))
return 1;
if (mode == DImode || mode == DFmode)
{
if (code == CONST_INT)
{
HOST_WIDE_INT val = INTVAL (index);
if (TARGET_LDRD)
return IN_RANGE (val, -1020, 1020) && (val & 3) == 0;
else
return IN_RANGE (val, -255, 4095 - 4);
}
else
return 0;
}
if (code == MULT)
{
rtx xiop0 = XEXP (index, 0);
rtx xiop1 = XEXP (index, 1);
return ((arm_address_register_rtx_p (xiop0, strict_p)
&& thumb2_index_mul_operand (xiop1))
|| (arm_address_register_rtx_p (xiop1, strict_p)
&& thumb2_index_mul_operand (xiop0)));
}
else if (code == ASHIFT)
{
rtx op = XEXP (index, 1);
return (arm_address_register_rtx_p (XEXP (index, 0), strict_p)
&& CONST_INT_P (op)
&& INTVAL (op) > 0
&& INTVAL (op) <= 3);
}
return (code == CONST_INT
&& INTVAL (index) < 4096
&& INTVAL (index) > -256);
}
static int
thumb1_base_register_rtx_p (rtx x, machine_mode mode, int strict_p)
{
int regno;
if (!REG_P (x))
return 0;
regno = REGNO (x);
if (strict_p)
return THUMB1_REGNO_MODE_OK_FOR_BASE_P (regno, mode);
return (regno <= LAST_LO_REGNUM
|| regno > LAST_VIRTUAL_REGISTER
|| regno == FRAME_POINTER_REGNUM
|| (GET_MODE_SIZE (mode) >= 4
&& (regno == STACK_POINTER_REGNUM
|| regno >= FIRST_PSEUDO_REGISTER
|| x == hard_frame_pointer_rtx
|| x == arg_pointer_rtx)));
}
inline static int
thumb1_index_register_rtx_p (rtx x, int strict_p)
{
return thumb1_base_register_rtx_p (x, QImode, strict_p);
}
int
thumb1_legitimate_address_p (machine_mode mode, rtx x, int strict_p)
{
if (TARGET_HAVE_MOVT && can_avoid_literal_pool_for_label_p (x))
return 0;
if (GET_MODE_SIZE (mode) < 4
&& !(reload_in_progress || reload_completed)
&& (reg_mentioned_p (frame_pointer_rtx, x)
|| reg_mentioned_p (arg_pointer_rtx, x)
|| reg_mentioned_p (virtual_incoming_args_rtx, x)
|| reg_mentioned_p (virtual_outgoing_args_rtx, x)
|| reg_mentioned_p (virtual_stack_dynamic_rtx, x)
|| reg_mentioned_p (virtual_stack_vars_rtx, x)))
return 0;
else if (thumb1_base_register_rtx_p (x, mode, strict_p))
return 1;
else if (GET_MODE_SIZE (mode) >= 4 && CONSTANT_P (x)
&& GET_CODE (x) == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (x) && !flag_pic)
return 1;
else if ((GET_MODE_SIZE (mode) >= 4 || mode == HFmode)
&& reload_completed
&& (GET_CODE (x) == LABEL_REF
|| (GET_CODE (x) == CONST
&& GET_CODE (XEXP (x, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == LABEL_REF
&& CONST_INT_P (XEXP (XEXP (x, 0), 1)))))
return 1;
else if (GET_CODE (x) == POST_INC && GET_MODE_SIZE (mode) >= 4
&& thumb1_index_register_rtx_p (XEXP (x, 0), strict_p))
return 1;
else if (GET_CODE (x) == PLUS)
{
if (GET_MODE_SIZE (mode) <= 4
&& XEXP (x, 0) != frame_pointer_rtx
&& XEXP (x, 1) != frame_pointer_rtx
&& thumb1_index_register_rtx_p (XEXP (x, 0), strict_p)
&& (thumb1_index_register_rtx_p (XEXP (x, 1), strict_p)
|| (!strict_p && will_be_in_index_register (XEXP (x, 1)))))
return 1;
else if ((thumb1_index_register_rtx_p (XEXP (x, 0), strict_p)
|| XEXP (x, 0) == arg_pointer_rtx)
&& CONST_INT_P (XEXP (x, 1))
&& thumb_legitimate_offset_p (mode, INTVAL (XEXP (x, 1))))
return 1;
else if (REG_P (XEXP (x, 0))
&& REGNO (XEXP (x, 0)) == STACK_POINTER_REGNUM
&& GET_MODE_SIZE (mode) >= 4
&& CONST_INT_P (XEXP (x, 1))
&& INTVAL (XEXP (x, 1)) >= 0
&& INTVAL (XEXP (x, 1)) + GET_MODE_SIZE (mode) <= 1024
&& (INTVAL (XEXP (x, 1)) & 3) == 0)
return 1;
else if (REG_P (XEXP (x, 0))
&& (REGNO (XEXP (x, 0)) == FRAME_POINTER_REGNUM
|| REGNO (XEXP (x, 0)) == ARG_POINTER_REGNUM
|| (REGNO (XEXP (x, 0)) >= FIRST_VIRTUAL_REGISTER
&& REGNO (XEXP (x, 0))
<= LAST_VIRTUAL_POINTER_REGISTER))
&& GET_MODE_SIZE (mode) >= 4
&& CONST_INT_P (XEXP (x, 1))
&& (INTVAL (XEXP (x, 1)) & 3) == 0)
return 1;
}
else if (GET_MODE_CLASS (mode) != MODE_FLOAT
&& GET_MODE_SIZE (mode) == 4
&& GET_CODE (x) == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (x)
&& ! (flag_pic
&& symbol_mentioned_p (get_pool_constant (x))
&& ! pcrel_constant_p (get_pool_constant (x))))
return 1;
return 0;
}
int
thumb_legitimate_offset_p (machine_mode mode, HOST_WIDE_INT val)
{
switch (GET_MODE_SIZE (mode))
{
case 1:
return val >= 0 && val < 32;
case 2:
return val >= 0 && val < 64 && (val & 1) == 0;
default:
return (val >= 0
&& (val + GET_MODE_SIZE (mode)) <= 128
&& (val & 3) == 0);
}
}
bool
arm_legitimate_address_p (machine_mode mode, rtx x, bool strict_p)
{
if (TARGET_ARM)
return arm_legitimate_address_outer_p (mode, x, SET, strict_p);
else if (TARGET_THUMB2)
return thumb2_legitimate_address_p (mode, x, strict_p);
else 
return thumb1_legitimate_address_p (mode, x, strict_p);
}
static reg_class_t
arm_preferred_reload_class (rtx x ATTRIBUTE_UNUSED, reg_class_t rclass)
{
if (TARGET_32BIT)
return rclass;
else
{
if (rclass == GENERAL_REGS)
return LO_REGS;
else
return rclass;
}
}
static GTY(()) rtx tls_get_addr_libfunc;
static rtx
get_tls_get_addr (void)
{
if (!tls_get_addr_libfunc)
tls_get_addr_libfunc = init_one_libfunc ("__tls_get_addr");
return tls_get_addr_libfunc;
}
rtx
arm_load_tp (rtx target)
{
if (!target)
target = gen_reg_rtx (SImode);
if (TARGET_HARD_TP)
{
emit_insn (gen_load_tp_hard (target));
}
else
{
rtx tmp;
emit_insn (gen_load_tp_soft ());
tmp = gen_rtx_REG (SImode, R0_REGNUM);
emit_move_insn (target, tmp);
}
return target;
}
static rtx
load_tls_operand (rtx x, rtx reg)
{
rtx tmp;
if (reg == NULL_RTX)
reg = gen_reg_rtx (SImode);
tmp = gen_rtx_CONST (SImode, x);
emit_move_insn (reg, tmp);
return reg;
}
static rtx_insn *
arm_call_tls_get_addr (rtx x, rtx reg, rtx *valuep, int reloc)
{
rtx label, labelno, sum;
gcc_assert (reloc != TLS_DESCSEQ);
start_sequence ();
labelno = GEN_INT (pic_labelno++);
label = gen_rtx_UNSPEC (Pmode, gen_rtvec (1, labelno), UNSPEC_PIC_LABEL);
label = gen_rtx_CONST (VOIDmode, label);
sum = gen_rtx_UNSPEC (Pmode,
gen_rtvec (4, x, GEN_INT (reloc), label,
GEN_INT (TARGET_ARM ? 8 : 4)),
UNSPEC_TLS);
reg = load_tls_operand (sum, reg);
if (TARGET_ARM)
emit_insn (gen_pic_add_dot_plus_eight (reg, reg, labelno));
else
emit_insn (gen_pic_add_dot_plus_four (reg, reg, labelno));
*valuep = emit_library_call_value (get_tls_get_addr (), NULL_RTX,
LCT_PURE, 
Pmode, reg, Pmode);
rtx_insn *insns = get_insns ();
end_sequence ();
return insns;
}
static rtx
arm_tls_descseq_addr (rtx x, rtx reg)
{
rtx labelno = GEN_INT (pic_labelno++);
rtx label = gen_rtx_UNSPEC (Pmode, gen_rtvec (1, labelno), UNSPEC_PIC_LABEL);
rtx sum = gen_rtx_UNSPEC (Pmode,
gen_rtvec (4, x, GEN_INT (TLS_DESCSEQ),
gen_rtx_CONST (VOIDmode, label),
GEN_INT (!TARGET_ARM)),
UNSPEC_TLS);
rtx reg0 = load_tls_operand (sum, gen_rtx_REG (SImode, R0_REGNUM));
emit_insn (gen_tlscall (x, labelno));
if (!reg)
reg = gen_reg_rtx (SImode);
else
gcc_assert (REGNO (reg) != R0_REGNUM);
emit_move_insn (reg, reg0);
return reg;
}
rtx
legitimize_tls_address (rtx x, rtx reg)
{
rtx dest, tp, label, labelno, sum, ret, eqv, addend;
rtx_insn *insns;
unsigned int model = SYMBOL_REF_TLS_MODEL (x);
switch (model)
{
case TLS_MODEL_GLOBAL_DYNAMIC:
if (TARGET_GNU2_TLS)
{
reg = arm_tls_descseq_addr (x, reg);
tp = arm_load_tp (NULL_RTX);
dest = gen_rtx_PLUS (Pmode, tp, reg);
}
else
{
insns = arm_call_tls_get_addr (x, reg, &ret, TLS_GD32);
dest = gen_reg_rtx (Pmode);
emit_libcall_block (insns, dest, ret, x);
}
return dest;
case TLS_MODEL_LOCAL_DYNAMIC:
if (TARGET_GNU2_TLS)
{
reg = arm_tls_descseq_addr (x, reg);
tp = arm_load_tp (NULL_RTX);
dest = gen_rtx_PLUS (Pmode, tp, reg);
}
else
{
insns = arm_call_tls_get_addr (x, reg, &ret, TLS_LDM32);
eqv = gen_rtx_UNSPEC (Pmode, gen_rtvec (1, const1_rtx),
UNSPEC_TLS);
dest = gen_reg_rtx (Pmode);
emit_libcall_block (insns, dest, ret, eqv);
addend = gen_rtx_UNSPEC (Pmode, gen_rtvec (2, x,
GEN_INT (TLS_LDO32)),
UNSPEC_TLS);
addend = force_reg (SImode, gen_rtx_CONST (SImode, addend));
dest = gen_rtx_PLUS (Pmode, dest, addend);
}
return dest;
case TLS_MODEL_INITIAL_EXEC:
labelno = GEN_INT (pic_labelno++);
label = gen_rtx_UNSPEC (Pmode, gen_rtvec (1, labelno), UNSPEC_PIC_LABEL);
label = gen_rtx_CONST (VOIDmode, label);
sum = gen_rtx_UNSPEC (Pmode,
gen_rtvec (4, x, GEN_INT (TLS_IE32), label,
GEN_INT (TARGET_ARM ? 8 : 4)),
UNSPEC_TLS);
reg = load_tls_operand (sum, reg);
if (TARGET_ARM)
emit_insn (gen_tls_load_dot_plus_eight (reg, reg, labelno));
else if (TARGET_THUMB2)
emit_insn (gen_tls_load_dot_plus_four (reg, NULL, reg, labelno));
else
{
emit_insn (gen_pic_add_dot_plus_four (reg, reg, labelno));
emit_move_insn (reg, gen_const_mem (SImode, reg));
}
tp = arm_load_tp (NULL_RTX);
return gen_rtx_PLUS (Pmode, tp, reg);
case TLS_MODEL_LOCAL_EXEC:
tp = arm_load_tp (NULL_RTX);
reg = gen_rtx_UNSPEC (Pmode,
gen_rtvec (2, x, GEN_INT (TLS_LE32)),
UNSPEC_TLS);
reg = force_reg (SImode, gen_rtx_CONST (SImode, reg));
return gen_rtx_PLUS (Pmode, tp, reg);
default:
abort ();
}
}
rtx
arm_legitimize_address (rtx x, rtx orig_x, machine_mode mode)
{
if (arm_tls_referenced_p (x))
{
rtx addend = NULL;
if (GET_CODE (x) == CONST && GET_CODE (XEXP (x, 0)) == PLUS)
{
addend = XEXP (XEXP (x, 0), 1);
x = XEXP (XEXP (x, 0), 0);
}
if (GET_CODE (x) != SYMBOL_REF)
return x;
gcc_assert (SYMBOL_REF_TLS_MODEL (x) != 0);
x = legitimize_tls_address (x, NULL_RTX);
if (addend)
{
x = gen_rtx_PLUS (SImode, x, addend);
orig_x = x;
}
else
return x;
}
if (!TARGET_ARM)
{
if (TARGET_THUMB2)
return x;
return thumb_legitimize_address (x, orig_x, mode);
}
if (GET_CODE (x) == PLUS)
{
rtx xop0 = XEXP (x, 0);
rtx xop1 = XEXP (x, 1);
if (CONSTANT_P (xop0) && !symbol_mentioned_p (xop0))
xop0 = force_reg (SImode, xop0);
if (CONSTANT_P (xop1) && !CONST_INT_P (xop1)
&& !symbol_mentioned_p (xop1))
xop1 = force_reg (SImode, xop1);
if (ARM_BASE_REGISTER_RTX_P (xop0)
&& CONST_INT_P (xop1))
{
HOST_WIDE_INT n, low_n;
rtx base_reg, val;
n = INTVAL (xop1);
if (mode == DImode || mode == DFmode)
{
low_n = n & 0x0f;
n &= ~0x0f;
if (low_n > 4)
{
n += 16;
low_n -= 16;
}
}
else
{
low_n = ((mode) == TImode ? 0
: n >= 0 ? (n & 0xfff) : -((-n) & 0xfff));
n -= low_n;
}
base_reg = gen_reg_rtx (SImode);
val = force_operand (plus_constant (Pmode, xop0, n), NULL_RTX);
emit_move_insn (base_reg, val);
x = plus_constant (Pmode, base_reg, low_n);
}
else if (xop0 != XEXP (x, 0) || xop1 != XEXP (x, 1))
x = gen_rtx_PLUS (SImode, xop0, xop1);
}
else if (GET_CODE (x) == MINUS)
{
rtx xop0 = XEXP (x, 0);
rtx xop1 = XEXP (x, 1);
if (CONSTANT_P (xop0))
xop0 = force_reg (SImode, xop0);
if (CONSTANT_P (xop1) && ! symbol_mentioned_p (xop1))
xop1 = force_reg (SImode, xop1);
if (xop0 != XEXP (x, 0) || xop1 != XEXP (x, 1))
x = gen_rtx_MINUS (SImode, xop0, xop1);
}
else if (CONST_INT_P (x) && optimize > 0)
{
unsigned int bits;
HOST_WIDE_INT mask, base, index;
rtx base_reg;
bits = (mode == SImode) ? 12 : 8;
mask = (1 << bits) - 1;
base = INTVAL (x) & ~mask;
index = INTVAL (x) & mask;
if (bit_count (base & 0xffffffff) > (32 - bits)/2)
{
base |= mask;
index -= mask;
}
base_reg = force_reg (SImode, GEN_INT (base));
x = plus_constant (Pmode, base_reg, index);
}
if (flag_pic)
{
rtx new_x = legitimize_pic_address (orig_x, mode, NULL_RTX);
if (new_x != orig_x)
x = new_x;
}
return x;
}
rtx
thumb_legitimize_address (rtx x, rtx orig_x, machine_mode mode)
{
if (GET_CODE (x) == PLUS
&& CONST_INT_P (XEXP (x, 1))
&& (INTVAL (XEXP (x, 1)) >= 32 * GET_MODE_SIZE (mode)
|| INTVAL (XEXP (x, 1)) < 0))
{
rtx xop0 = XEXP (x, 0);
rtx xop1 = XEXP (x, 1);
HOST_WIDE_INT offset = INTVAL (xop1);
if (optimize_size && offset >= 0
&& offset < 256 + 31 * GET_MODE_SIZE (mode))
{
HOST_WIDE_INT delta;
if (offset >= 256)
delta = offset - (256 - GET_MODE_SIZE (mode));
else if (offset < 32 * GET_MODE_SIZE (mode) + 8)
delta = 31 * GET_MODE_SIZE (mode);
else
delta = offset & (~31 * GET_MODE_SIZE (mode));
xop0 = force_operand (plus_constant (Pmode, xop0, offset - delta),
NULL_RTX);
x = plus_constant (Pmode, xop0, delta);
}
else if (offset < 0 && offset > -256)
x = force_operand (x, NULL_RTX);
else
{
xop1 = force_reg (SImode, xop1);
x = gen_rtx_PLUS (SImode, xop0, xop1);
}
}
else if (GET_CODE (x) == PLUS
&& s_register_operand (XEXP (x, 1), SImode)
&& !s_register_operand (XEXP (x, 0), SImode))
{
rtx xop0 = force_operand (XEXP (x, 0), NULL_RTX);
x = gen_rtx_PLUS (SImode, xop0, XEXP (x, 1));
}
if (flag_pic)
{
rtx new_x = legitimize_pic_address (orig_x, mode, NULL_RTX);
if (new_x != orig_x)
x = new_x;
}
return x;
}
bool
arm_tls_referenced_p (rtx x)
{
if (! TARGET_HAVE_TLS)
return false;
subrtx_iterator::array_type array;
FOR_EACH_SUBRTX (iter, array, x, ALL)
{
const_rtx x = *iter;
if (GET_CODE (x) == SYMBOL_REF && SYMBOL_REF_TLS_MODEL (x) != 0)
{
if (arm_disable_literal_pool)
sorry ("accessing thread-local storage is not currently supported "
"with -mpure-code or -mslow-flash-data");
return true;
}
if (GET_CODE (x) == UNSPEC && XINT (x, 1) == UNSPEC_TLS)
iter.skip_subrtxes ();
}
return false;
}
static bool
arm_legitimate_constant_p_1 (machine_mode, rtx x)
{
return flag_pic || !label_mentioned_p (x);
}
static bool
thumb_legitimate_constant_p (machine_mode mode ATTRIBUTE_UNUSED, rtx x)
{
if (TARGET_HAVE_MOVT && GET_CODE (x) == HIGH)
x = XEXP (x, 0);
return (CONST_INT_P (x)
|| CONST_DOUBLE_P (x)
|| CONSTANT_ADDRESS_P (x)
|| (TARGET_HAVE_MOVT && GET_CODE (x) == SYMBOL_REF)
|| flag_pic);
}
static bool
arm_legitimate_constant_p (machine_mode mode, rtx x)
{
return (!arm_cannot_force_const_mem (mode, x)
&& (TARGET_32BIT
? arm_legitimate_constant_p_1 (mode, x)
: thumb_legitimate_constant_p (mode, x)));
}
static bool
arm_cannot_force_const_mem (machine_mode mode ATTRIBUTE_UNUSED, rtx x)
{
rtx base, offset;
if (ARM_OFFSETS_MUST_BE_WITHIN_SECTIONS_P)
{
split_const (x, &base, &offset);
if (GET_CODE (base) == SYMBOL_REF
&& !offset_within_block_p (base, INTVAL (offset)))
return true;
}
return arm_tls_referenced_p (x);
}

#define REG_OR_SUBREG_REG(X)						\
(REG_P (X)							\
|| (GET_CODE (X) == SUBREG && REG_P (SUBREG_REG (X))))
#define REG_OR_SUBREG_RTX(X)			\
(REG_P (X) ? (X) : SUBREG_REG (X))
static inline int
thumb1_rtx_costs (rtx x, enum rtx_code code, enum rtx_code outer)
{
machine_mode mode = GET_MODE (x);
int total, words;
switch (code)
{
case ASHIFT:
case ASHIFTRT:
case LSHIFTRT:
case ROTATERT:
return (mode == SImode) ? COSTS_N_INSNS (1) : COSTS_N_INSNS (2);
case PLUS:
case MINUS:
case COMPARE:
case NEG:
case NOT:
return COSTS_N_INSNS (1);
case MULT:
if (arm_arch6m && arm_m_profile_small_mul)
return COSTS_N_INSNS (32);
if (CONST_INT_P (XEXP (x, 1)))
{
int cycles = 0;
unsigned HOST_WIDE_INT i = INTVAL (XEXP (x, 1));
while (i)
{
i >>= 2;
cycles++;
}
return COSTS_N_INSNS (2) + cycles;
}
return COSTS_N_INSNS (1) + 16;
case SET:
words = ARM_NUM_INTS (GET_MODE_SIZE (GET_MODE (SET_DEST (x))));
return (COSTS_N_INSNS (words)
+ 4 * ((MEM_P (SET_SRC (x)))
+ MEM_P (SET_DEST (x))));
case CONST_INT:
if (outer == SET)
{
if (UINTVAL (x) < 256
|| (TARGET_HAVE_MOVT && !(INTVAL (x) & 0xffff0000)))
return 0;
if (thumb_shiftable_const (INTVAL (x)))
return COSTS_N_INSNS (2);
return COSTS_N_INSNS (3);
}
else if ((outer == PLUS || outer == COMPARE)
&& INTVAL (x) < 256 && INTVAL (x) > -256)
return 0;
else if ((outer == IOR || outer == XOR || outer == AND)
&& INTVAL (x) < 256 && INTVAL (x) >= -256)
return COSTS_N_INSNS (1);
else if (outer == AND)
{
int i;
for (i = 9; i <= 31; i++)
if ((HOST_WIDE_INT_1 << i) - 1 == INTVAL (x)
|| (HOST_WIDE_INT_1 << i) - 1 == ~INTVAL (x))
return COSTS_N_INSNS (2);
}
else if (outer == ASHIFT || outer == ASHIFTRT
|| outer == LSHIFTRT)
return 0;
return COSTS_N_INSNS (2);
case CONST:
case CONST_DOUBLE:
case LABEL_REF:
case SYMBOL_REF:
return COSTS_N_INSNS (3);
case UDIV:
case UMOD:
case DIV:
case MOD:
return 100;
case TRUNCATE:
return 99;
case AND:
case XOR:
case IOR:
return 8;
case MEM:
return (10 + 4 * ((GET_MODE_SIZE (mode) - 1) / UNITS_PER_WORD)
+ ((GET_CODE (x) == SYMBOL_REF && CONSTANT_POOL_ADDRESS_P (x))
? 4 : 0));
case IF_THEN_ELSE:
if (GET_CODE (XEXP (x, 1)) == PC || GET_CODE (XEXP (x, 2)) == PC)
return 14;
return 2;
case SIGN_EXTEND:
case ZERO_EXTEND:
total = mode == DImode ? COSTS_N_INSNS (1) : 0;
total += thumb1_rtx_costs (XEXP (x, 0), GET_CODE (XEXP (x, 0)), code);
if (mode == SImode)
return total;
if (arm_arch6)
return total + COSTS_N_INSNS (1);
return total + 1 + COSTS_N_INSNS (2);
default:
return 99;
}
}
static inline int
thumb1_size_rtx_costs (rtx x, enum rtx_code code, enum rtx_code outer)
{
machine_mode mode = GET_MODE (x);
int words, cost;
switch (code)
{
case ASHIFT:
case ASHIFTRT:
case LSHIFTRT:
case ROTATERT:
return (mode == SImode) ? COSTS_N_INSNS (1) : COSTS_N_INSNS (2);
case PLUS:
case MINUS:
if ((GET_CODE (XEXP (x, 0)) == MULT
&& power_of_two_operand (XEXP (XEXP (x,0),1), SImode))
|| (GET_CODE (XEXP (x, 1)) == MULT
&& power_of_two_operand (XEXP (XEXP (x, 1), 1), SImode)))
return COSTS_N_INSNS (2);
case COMPARE:
case NEG:
case NOT:
return COSTS_N_INSNS (1);
case MULT:
if (CONST_INT_P (XEXP (x, 1)))
{
int const_size = thumb1_size_rtx_costs (XEXP (x, 1), CONST_INT, SET);
if (arm_arch6m && arm_m_profile_small_mul)
return COSTS_N_INSNS (5);
else
return COSTS_N_INSNS (1) + const_size;
}
return COSTS_N_INSNS (1);
case SET:
words = ARM_NUM_INTS (GET_MODE_SIZE (GET_MODE (SET_DEST (x))));
cost = COSTS_N_INSNS (words);
if (satisfies_constraint_J (SET_SRC (x))
|| satisfies_constraint_K (SET_SRC (x))
|| (CONST_INT_P (SET_SRC (x))
&& UINTVAL (SET_SRC (x)) >= 256
&& TARGET_HAVE_MOVT
&& satisfies_constraint_j (SET_SRC (x)))
|| ((words > 1) && MEM_P (SET_SRC (x))))
cost += COSTS_N_INSNS (1);
return cost;
case CONST_INT:
if (outer == SET)
{
if (UINTVAL (x) < 256)
return COSTS_N_INSNS (1);
if (TARGET_HAVE_MOVT && !(INTVAL (x) & 0xffff0000))
return COSTS_N_INSNS (2);
if (INTVAL (x) >= -255 && INTVAL (x) <= -1)
return COSTS_N_INSNS (2);
if (thumb_shiftable_const (INTVAL (x)))
return COSTS_N_INSNS (2);
return COSTS_N_INSNS (3);
}
else if ((outer == PLUS || outer == COMPARE)
&& INTVAL (x) < 256 && INTVAL (x) > -256)
return 0;
else if ((outer == IOR || outer == XOR || outer == AND)
&& INTVAL (x) < 256 && INTVAL (x) >= -256)
return COSTS_N_INSNS (1);
else if (outer == AND)
{
int i;
for (i = 9; i <= 31; i++)
if ((HOST_WIDE_INT_1 << i) - 1 == INTVAL (x)
|| (HOST_WIDE_INT_1 << i) - 1 == ~INTVAL (x))
return COSTS_N_INSNS (2);
}
else if (outer == ASHIFT || outer == ASHIFTRT
|| outer == LSHIFTRT)
return 0;
return COSTS_N_INSNS (2);
case CONST:
case CONST_DOUBLE:
case LABEL_REF:
case SYMBOL_REF:
return COSTS_N_INSNS (3);
case UDIV:
case UMOD:
case DIV:
case MOD:
return 100;
case TRUNCATE:
return 99;
case AND:
case XOR:
case IOR:
return COSTS_N_INSNS (1);
case MEM:
return (COSTS_N_INSNS (1)
+ COSTS_N_INSNS (1)
* ((GET_MODE_SIZE (mode) - 1) / UNITS_PER_WORD)
+ ((GET_CODE (x) == SYMBOL_REF && CONSTANT_POOL_ADDRESS_P (x))
? COSTS_N_INSNS (1) : 0));
case IF_THEN_ELSE:
if (GET_CODE (XEXP (x, 1)) == PC || GET_CODE (XEXP (x, 2)) == PC)
return 14;
return 2;
case ZERO_EXTEND:
switch (GET_MODE (XEXP (x, 0)))
{
case E_QImode:
return (1 + (mode == DImode ? 4 : 0)
+ (MEM_P (XEXP (x, 0)) ? 10 : 0));
case E_HImode:
return (4 + (mode == DImode ? 4 : 0)
+ (MEM_P (XEXP (x, 0)) ? 10 : 0));
case E_SImode:
return (1 + (MEM_P (XEXP (x, 0)) ? 10 : 0));
default:
return 99;
}
default:
return 99;
}
}
static rtx
shifter_op_p (rtx op, rtx *shift_reg)
{
enum rtx_code code = GET_CODE (op);
if (code == MULT && CONST_INT_P (XEXP (op, 1))
&& exact_log2 (INTVAL (XEXP (op, 1))) > 0)
return XEXP (op, 0);
else if (code == ROTATE && CONST_INT_P (XEXP (op, 1)))
return XEXP (op, 0);
else if (code == ROTATERT || code == ASHIFT || code == LSHIFTRT
|| code == ASHIFTRT)
{
if (!CONST_INT_P (XEXP (op, 1)))
*shift_reg = XEXP (op, 1);
return XEXP (op, 0);
}
return NULL;
}
static bool
arm_unspec_cost (rtx x, enum rtx_code , bool speed_p, int *cost)
{
const struct cpu_cost_table *extra_cost = current_tune->insn_extra_cost;
rtx_code code = GET_CODE (x);
gcc_assert (code == UNSPEC || code == UNSPEC_VOLATILE);
switch (XINT (x, 1))
{
case UNSPEC_UNALIGNED_LOAD:
*cost = COSTS_N_INSNS (ARM_NUM_REGS (GET_MODE (x)));
if (speed_p)
*cost += (ARM_NUM_REGS (GET_MODE (x)) * extra_cost->ldst.load
+ extra_cost->ldst.load_unaligned);
#ifdef NOT_YET
*cost += arm_address_cost (XEXP (XVECEXP (x, 0, 0), 0), GET_MODE (x),
ADDR_SPACE_GENERIC, speed_p);
#endif
return true;
case UNSPEC_UNALIGNED_STORE:
*cost = COSTS_N_INSNS (ARM_NUM_REGS (GET_MODE (x)));
if (speed_p)
*cost += (ARM_NUM_REGS (GET_MODE (x)) * extra_cost->ldst.store
+ extra_cost->ldst.store_unaligned);
*cost += rtx_cost (XVECEXP (x, 0, 0), VOIDmode, UNSPEC, 0, speed_p);
#ifdef NOT_YET
*cost += arm_address_cost (XEXP (XVECEXP (x, 0, 0), 0), GET_MODE (x),
ADDR_SPACE_GENERIC, speed_p);
#endif
return true;
case UNSPEC_VRINTZ:
case UNSPEC_VRINTP:
case UNSPEC_VRINTM:
case UNSPEC_VRINTR:
case UNSPEC_VRINTX:
case UNSPEC_VRINTA:
if (speed_p)
*cost += extra_cost->fp[GET_MODE (x) == DFmode].roundint;
return true;
default:
*cost = COSTS_N_INSNS (2);
break;
}
return true;
}
#define LIBCALL_COST(N) COSTS_N_INSNS (N + (speed_p ? 18 : 2))
#define HANDLE_NARROW_SHIFT_ARITH(OP, IDX)				\
do								\
{								\
shift_op = shifter_op_p (XEXP (x, IDX), &shift_reg);	\
if (shift_op != NULL					\
&& arm_rtx_shift_left_p (XEXP (x, IDX)))		\
{								\
if (shift_reg)						\
{							\
if (speed_p)					\
*cost += extra_cost->alu.arith_shift_reg;		\
*cost += rtx_cost (shift_reg, GET_MODE (shift_reg),	\
ASHIFT, 1, speed_p);		\
}							\
else if (speed_p)					\
*cost += extra_cost->alu.arith_shift;			\
\
*cost += (rtx_cost (shift_op, GET_MODE (shift_op),	\
ASHIFT, 0, speed_p)			\
+ rtx_cost (XEXP (x, 1 - IDX),		\
GET_MODE (shift_op),		\
OP, 1, speed_p));			\
return true;						\
}								\
}								\
while (0)
static bool
arm_mem_costs (rtx x, const struct cpu_cost_table *extra_cost,
int *cost, bool speed_p)
{
machine_mode mode = GET_MODE (x);
*cost = COSTS_N_INSNS (1);
if (flag_pic
&& GET_CODE (XEXP (x, 0)) == PLUS
&& will_be_in_index_register (XEXP (XEXP (x, 0), 1)))
*cost += COSTS_N_INSNS (1);
if (speed_p)
{
arm_addr_mode_op op_type;
switch (GET_CODE (XEXP (x, 0)))
{
default:
case REG:
op_type = AMO_DEFAULT;
break;
case MINUS:
case PLUS:
op_type = AMO_NO_WB;
break;
case PRE_INC:
case PRE_DEC:
case POST_INC:
case POST_DEC:
case PRE_MODIFY:
case POST_MODIFY:
op_type = AMO_WB;
break;
}
if (VECTOR_MODE_P (mode))
*cost += current_tune->addr_mode_costs->vector[op_type];
else if (FLOAT_MODE_P (mode))
*cost += current_tune->addr_mode_costs->fp[op_type];
else
*cost += current_tune->addr_mode_costs->integer[op_type];
}
if (speed_p)
{
if (FLOAT_MODE_P (mode))
{
if (GET_MODE_SIZE (mode) == 8)
*cost += extra_cost->ldst.loadd;
else
*cost += extra_cost->ldst.loadf;
}
else if (VECTOR_MODE_P (mode))
*cost += extra_cost->ldst.loadv;
else
{
if (GET_MODE_SIZE (mode) == 8)
*cost += extra_cost->ldst.ldrd;
else
*cost += extra_cost->ldst.load;
}
}
return true;
}
static bool
arm_rtx_costs_internal (rtx x, enum rtx_code code, enum rtx_code outer_code,
const struct cpu_cost_table *extra_cost,
int *cost, bool speed_p)
{
machine_mode mode = GET_MODE (x);
*cost = COSTS_N_INSNS (1);
if (TARGET_THUMB1)
{
if (speed_p)
*cost = thumb1_rtx_costs (x, code, outer_code);
else
*cost = thumb1_size_rtx_costs (x, code, outer_code);
return true;
}
switch (code)
{
case SET:
*cost = 0;
mode = GET_MODE (SET_DEST (x));
if (REG_P (SET_SRC (x))
&& REG_P (SET_DEST (x)))
{
*cost = COSTS_N_INSNS (((!TARGET_HARD_FLOAT
&& GET_MODE_SIZE (mode) > 4)
|| mode == DImode)
? 2 : 1);
if (!speed_p && TARGET_THUMB && outer_code == COND_EXEC)
*cost >>= 1;
return true;
}
if (CONST_INT_P (SET_SRC (x)))
{
*cost = rtx_cost (SET_DEST (x), GET_MODE (SET_DEST (x)), SET,
0, speed_p);
outer_code = SET;
if (REG_P (SET_DEST (x))
&& REGNO (SET_DEST (x)) <= LR_REGNUM)
*cost -= 1;
x = SET_SRC (x);
if (!speed_p && TARGET_THUMB && GET_MODE (x) == SImode
&& INTVAL (x) >= 0 && INTVAL (x) <=255)
*cost >>= 1;
goto const_int_cost;
}
return false;
case MEM:
return arm_mem_costs (x, extra_cost, cost, speed_p);
case PARALLEL:
{
bool is_ldm = load_multiple_operation (x, SImode);
bool is_stm = store_multiple_operation (x, SImode);
if (is_ldm || is_stm)
{
if (speed_p)
{
HOST_WIDE_INT nregs = XVECLEN (x, 0);
HOST_WIDE_INT regs_per_insn_1st = is_ldm
? extra_cost->ldst.ldm_regs_per_insn_1st
: extra_cost->ldst.stm_regs_per_insn_1st;
HOST_WIDE_INT regs_per_insn_sub = is_ldm
? extra_cost->ldst.ldm_regs_per_insn_subsequent
: extra_cost->ldst.stm_regs_per_insn_subsequent;
*cost += regs_per_insn_1st
+ COSTS_N_INSNS (((MAX (nregs - regs_per_insn_1st, 0))
+ regs_per_insn_sub - 1)
/ regs_per_insn_sub);
return true;
}
}
return false;
}
case DIV:
case UDIV:
if (TARGET_HARD_FLOAT && GET_MODE_CLASS (mode) == MODE_FLOAT
&& (mode == SFmode || !TARGET_VFP_SINGLE))
*cost += COSTS_N_INSNS (speed_p
? extra_cost->fp[mode != SFmode].div : 0);
else if (mode == SImode && TARGET_IDIV)
*cost += COSTS_N_INSNS (speed_p ? extra_cost->mult[0].idiv : 0);
else
*cost = LIBCALL_COST (2);
*cost += (code == DIV ? COSTS_N_INSNS (1) : 0);
return false;	
case MOD:
if (CONST_INT_P (XEXP (x, 1))
&& exact_log2 (INTVAL (XEXP (x, 1))) > 0
&& mode == SImode)
{
*cost += COSTS_N_INSNS (3);
if (speed_p)
*cost += 2 * extra_cost->alu.logical
+ extra_cost->alu.arith;
return true;
}
case UMOD:
*cost = LIBCALL_COST (2) + (code == MOD ? COSTS_N_INSNS (1) : 0);
return false;	
case ROTATE:
if (mode == SImode && REG_P (XEXP (x, 1)))
{
*cost += (COSTS_N_INSNS (1)
+ rtx_cost (XEXP (x, 0), mode, code, 0, speed_p));
if (speed_p)
*cost += extra_cost->alu.shift_reg;
return true;
}
case ROTATERT:
case ASHIFT:
case LSHIFTRT:
case ASHIFTRT:
if (mode == DImode && CONST_INT_P (XEXP (x, 1)))
{
*cost += (COSTS_N_INSNS (2)
+ rtx_cost (XEXP (x, 0), mode, code, 0, speed_p));
if (speed_p)
*cost += 2 * extra_cost->alu.shift;
if (code == ASHIFT && XEXP (x, 1) == CONST1_RTX (SImode))
*cost += 1;
return true;
}
else if (mode == SImode)
{
*cost += rtx_cost (XEXP (x, 0), mode, code, 0, speed_p);
if (!CONST_INT_P (XEXP (x, 1)))
*cost += (speed_p ? extra_cost->alu.shift_reg : 1
+ rtx_cost (XEXP (x, 1), mode, code, 1, speed_p));
return true;
}
else if (GET_MODE_CLASS (mode) == MODE_INT
&& GET_MODE_SIZE (mode) < 4)
{
if (code == ASHIFT)
{
*cost += rtx_cost (XEXP (x, 0), mode, code, 0, speed_p);
if (!CONST_INT_P (XEXP (x, 1)))
*cost += (speed_p ? extra_cost->alu.shift_reg : 1
+ rtx_cost (XEXP (x, 1), mode, code, 1, speed_p));
}
else if (code == LSHIFTRT || code == ASHIFTRT)
{
if (arm_arch_thumb2 && CONST_INT_P (XEXP (x, 1)))
{
if (speed_p)
*cost += extra_cost->alu.bfx;
*cost += rtx_cost (XEXP (x, 0), mode, code, 0, speed_p);
}
else
{
*cost += COSTS_N_INSNS (1);
*cost += rtx_cost (XEXP (x, 0), mode, code, 0, speed_p);
if (speed_p)
{
if (CONST_INT_P (XEXP (x, 1)))
*cost += 2 * extra_cost->alu.shift;
else
*cost += (extra_cost->alu.shift
+ extra_cost->alu.shift_reg);
}
else
*cost += !CONST_INT_P (XEXP (x, 1));
}
}
else 
{
*cost = COSTS_N_INSNS (2 + !CONST_INT_P (XEXP (x, 1)));
*cost += rtx_cost (XEXP (x, 0), mode, code, 0, speed_p);
if (speed_p)
{
if (CONST_INT_P (XEXP (x, 1)))
*cost += (2 * extra_cost->alu.shift
+ extra_cost->alu.log_shift);
else
*cost += (extra_cost->alu.shift
+ extra_cost->alu.shift_reg
+ extra_cost->alu.log_shift_reg);
}
}
return true;
}
*cost = LIBCALL_COST (2);
return false;
case BSWAP:
if (arm_arch6)
{
if (mode == SImode)
{
if (speed_p)
*cost += extra_cost->alu.rev;
return false;
}
}
else
{
if (TARGET_THUMB)
{
*cost += COSTS_N_INSNS (9);
if (speed_p)
{
*cost += 6 * extra_cost->alu.shift;
*cost += 3 * extra_cost->alu.logical;
}
}
else
{
*cost += COSTS_N_INSNS (4);
if (speed_p)
{
*cost += 2 * extra_cost->alu.shift;
*cost += extra_cost->alu.arith_shift;
*cost += 2 * extra_cost->alu.logical;
}
}
return true;
}
return false;
case MINUS:
if (TARGET_HARD_FLOAT && GET_MODE_CLASS (mode) == MODE_FLOAT
&& (mode == SFmode || !TARGET_VFP_SINGLE))
{
if (GET_CODE (XEXP (x, 0)) == MULT
|| GET_CODE (XEXP (x, 1)) == MULT)
{
rtx mul_op0, mul_op1, sub_op;
if (speed_p)
*cost += extra_cost->fp[mode != SFmode].mult_addsub;
if (GET_CODE (XEXP (x, 0)) == MULT)
{
mul_op0 = XEXP (XEXP (x, 0), 0);
mul_op1 = XEXP (XEXP (x, 0), 1);
sub_op = XEXP (x, 1);
}
else
{
mul_op0 = XEXP (XEXP (x, 1), 0);
mul_op1 = XEXP (XEXP (x, 1), 1);
sub_op = XEXP (x, 0);
}
if (GET_CODE (mul_op0) == NEG)
mul_op0 = XEXP (mul_op0, 0);
*cost += (rtx_cost (mul_op0, mode, code, 0, speed_p)
+ rtx_cost (mul_op1, mode, code, 0, speed_p)
+ rtx_cost (sub_op, mode, code, 0, speed_p));
return true;
}
if (speed_p)
*cost += extra_cost->fp[mode != SFmode].addsub;
return false;
}
if (mode == SImode)
{
rtx shift_by_reg = NULL;
rtx shift_op;
rtx non_shift_op;
shift_op = shifter_op_p (XEXP (x, 0), &shift_by_reg);
if (shift_op == NULL)
{
shift_op = shifter_op_p (XEXP (x, 1), &shift_by_reg);
non_shift_op = XEXP (x, 0);
}
else
non_shift_op = XEXP (x, 1);
if (shift_op != NULL)
{
if (shift_by_reg != NULL)
{
if (speed_p)
*cost += extra_cost->alu.arith_shift_reg;
*cost += rtx_cost (shift_by_reg, mode, code, 0, speed_p);
}
else if (speed_p)
*cost += extra_cost->alu.arith_shift;
*cost += rtx_cost (shift_op, mode, code, 0, speed_p);
*cost += rtx_cost (non_shift_op, mode, code, 0, speed_p);
return true;
}
if (arm_arch_thumb2
&& GET_CODE (XEXP (x, 1)) == MULT)
{
if (speed_p)
*cost += extra_cost->mult[0].add;
*cost += rtx_cost (XEXP (x, 0), mode, MINUS, 0, speed_p);
*cost += rtx_cost (XEXP (XEXP (x, 1), 0), mode, MULT, 0, speed_p);
*cost += rtx_cost (XEXP (XEXP (x, 1), 1), mode, MULT, 1, speed_p);
return true;
}
if (CONST_INT_P (XEXP (x, 0)))
{
int insns = arm_gen_constant (MINUS, SImode, NULL_RTX,
INTVAL (XEXP (x, 0)), NULL_RTX,
NULL_RTX, 1, 0);
*cost = COSTS_N_INSNS (insns);
if (speed_p)
*cost += insns * extra_cost->alu.arith;
*cost += rtx_cost (XEXP (x, 1), mode, code, 1, speed_p);
return true;
}
else if (speed_p)
*cost += extra_cost->alu.arith;
return false;
}
if (GET_MODE_CLASS (mode) == MODE_INT
&& GET_MODE_SIZE (mode) < 4)
{
rtx shift_op, shift_reg;
shift_reg = NULL;
HANDLE_NARROW_SHIFT_ARITH (MINUS, 0);
HANDLE_NARROW_SHIFT_ARITH (MINUS, 1);
*cost += 1;
if (speed_p)
*cost += extra_cost->alu.arith;
if (CONST_INT_P (XEXP (x, 0)))
{
*cost += rtx_cost (XEXP (x, 1), mode, code, 1, speed_p);
return true;
}
return false;
}
if (mode == DImode)
{
*cost += COSTS_N_INSNS (1);
if (GET_CODE (XEXP (x, 0)) == ZERO_EXTEND)
{
rtx op1 = XEXP (x, 1);
if (speed_p)
*cost += 2 * extra_cost->alu.arith;
if (GET_CODE (op1) == ZERO_EXTEND)
*cost += rtx_cost (XEXP (op1, 0), VOIDmode, ZERO_EXTEND,
0, speed_p);
else
*cost += rtx_cost (op1, mode, MINUS, 1, speed_p);
*cost += rtx_cost (XEXP (XEXP (x, 0), 0), VOIDmode, ZERO_EXTEND,
0, speed_p);
return true;
}
else if (GET_CODE (XEXP (x, 0)) == SIGN_EXTEND)
{
if (speed_p)
*cost += extra_cost->alu.arith + extra_cost->alu.arith_shift;
*cost += (rtx_cost (XEXP (XEXP (x, 0), 0), VOIDmode, SIGN_EXTEND,
0, speed_p)
+ rtx_cost (XEXP (x, 1), mode, MINUS, 1, speed_p));
return true;
}
else if (GET_CODE (XEXP (x, 1)) == ZERO_EXTEND
|| GET_CODE (XEXP (x, 1)) == SIGN_EXTEND)
{
if (speed_p)
*cost += (extra_cost->alu.arith
+ (GET_CODE (XEXP (x, 1)) == ZERO_EXTEND
? extra_cost->alu.arith
: extra_cost->alu.arith_shift));
*cost += (rtx_cost (XEXP (x, 0), mode, MINUS, 0, speed_p)
+ rtx_cost (XEXP (XEXP (x, 1), 0), VOIDmode,
GET_CODE (XEXP (x, 1)), 0, speed_p));
return true;
}
if (speed_p)
*cost += 2 * extra_cost->alu.arith;
return false;
}
*cost = LIBCALL_COST (2);
return false;
case PLUS:
if (TARGET_HARD_FLOAT && GET_MODE_CLASS (mode) == MODE_FLOAT
&& (mode == SFmode || !TARGET_VFP_SINGLE))
{
if (GET_CODE (XEXP (x, 0)) == MULT)
{
rtx mul_op0, mul_op1, add_op;
if (speed_p)
*cost += extra_cost->fp[mode != SFmode].mult_addsub;
mul_op0 = XEXP (XEXP (x, 0), 0);
mul_op1 = XEXP (XEXP (x, 0), 1);
add_op = XEXP (x, 1);
*cost += (rtx_cost (mul_op0, mode, code, 0, speed_p)
+ rtx_cost (mul_op1, mode, code, 0, speed_p)
+ rtx_cost (add_op, mode, code, 0, speed_p));
return true;
}
if (speed_p)
*cost += extra_cost->fp[mode != SFmode].addsub;
return false;
}
else if (GET_MODE_CLASS (mode) == MODE_FLOAT)
{
*cost = LIBCALL_COST (2);
return false;
}
if (GET_MODE_CLASS (mode) == MODE_INT
&& GET_MODE_SIZE (mode) < 4)
{
rtx shift_op, shift_reg;
shift_reg = NULL;
HANDLE_NARROW_SHIFT_ARITH (PLUS, 0);
if (CONST_INT_P (XEXP (x, 1)))
{
int insns = arm_gen_constant (PLUS, SImode, NULL_RTX,
INTVAL (XEXP (x, 1)), NULL_RTX,
NULL_RTX, 1, 0);
*cost = COSTS_N_INSNS (insns);
if (speed_p)
*cost += insns * extra_cost->alu.arith;
*cost += 1 + rtx_cost (XEXP (x, 0), mode, PLUS, 0, speed_p);
return true;
}
*cost += 1;
if (speed_p)
*cost += extra_cost->alu.arith;
return false;
}
if (mode == SImode)
{
rtx shift_op, shift_reg;
if (TARGET_INT_SIMD
&& (GET_CODE (XEXP (x, 0)) == ZERO_EXTEND
|| GET_CODE (XEXP (x, 0)) == SIGN_EXTEND))
{
if (speed_p)
*cost += extra_cost->alu.extend_arith;
*cost += (rtx_cost (XEXP (XEXP (x, 0), 0), VOIDmode, ZERO_EXTEND,
0, speed_p)
+ rtx_cost (XEXP (x, 1), mode, PLUS, 0, speed_p));
return true;
}
shift_reg = NULL;
shift_op = shifter_op_p (XEXP (x, 0), &shift_reg);
if (shift_op != NULL)
{
if (shift_reg)
{
if (speed_p)
*cost += extra_cost->alu.arith_shift_reg;
*cost += rtx_cost (shift_reg, mode, ASHIFT, 1, speed_p);
}
else if (speed_p)
*cost += extra_cost->alu.arith_shift;
*cost += (rtx_cost (shift_op, mode, ASHIFT, 0, speed_p)
+ rtx_cost (XEXP (x, 1), mode, PLUS, 1, speed_p));
return true;
}
if (GET_CODE (XEXP (x, 0)) == MULT)
{
rtx mul_op = XEXP (x, 0);
if (TARGET_DSP_MULTIPLY
&& ((GET_CODE (XEXP (mul_op, 0)) == SIGN_EXTEND
&& (GET_CODE (XEXP (mul_op, 1)) == SIGN_EXTEND
|| (GET_CODE (XEXP (mul_op, 1)) == ASHIFTRT
&& CONST_INT_P (XEXP (XEXP (mul_op, 1), 1))
&& INTVAL (XEXP (XEXP (mul_op, 1), 1)) == 16)))
|| (GET_CODE (XEXP (mul_op, 0)) == ASHIFTRT
&& CONST_INT_P (XEXP (XEXP (mul_op, 0), 1))
&& INTVAL (XEXP (XEXP (mul_op, 0), 1)) == 16
&& (GET_CODE (XEXP (mul_op, 1)) == SIGN_EXTEND
|| (GET_CODE (XEXP (mul_op, 1)) == ASHIFTRT
&& CONST_INT_P (XEXP (XEXP (mul_op, 1), 1))
&& (INTVAL (XEXP (XEXP (mul_op, 1), 1))
== 16))))))
{
if (speed_p)
*cost += extra_cost->mult[0].extend_add;
*cost += (rtx_cost (XEXP (XEXP (mul_op, 0), 0), mode,
SIGN_EXTEND, 0, speed_p)
+ rtx_cost (XEXP (XEXP (mul_op, 1), 0), mode,
SIGN_EXTEND, 0, speed_p)
+ rtx_cost (XEXP (x, 1), mode, PLUS, 1, speed_p));
return true;
}
if (speed_p)
*cost += extra_cost->mult[0].add;
*cost += (rtx_cost (XEXP (mul_op, 0), mode, MULT, 0, speed_p)
+ rtx_cost (XEXP (mul_op, 1), mode, MULT, 1, speed_p)
+ rtx_cost (XEXP (x, 1), mode, PLUS, 1, speed_p));
return true;
}
if (CONST_INT_P (XEXP (x, 1)))
{
int insns = arm_gen_constant (PLUS, SImode, NULL_RTX,
INTVAL (XEXP (x, 1)), NULL_RTX,
NULL_RTX, 1, 0);
*cost = COSTS_N_INSNS (insns);
if (speed_p)
*cost += insns * extra_cost->alu.arith;
*cost += rtx_cost (XEXP (x, 0), mode, PLUS, 0, speed_p);
return true;
}
else if (speed_p)
*cost += extra_cost->alu.arith;
return false;
}
if (mode == DImode)
{
if (arm_arch3m
&& GET_CODE (XEXP (x, 0)) == MULT
&& ((GET_CODE (XEXP (XEXP (x, 0), 0)) == ZERO_EXTEND
&& GET_CODE (XEXP (XEXP (x, 0), 1)) == ZERO_EXTEND)
|| (GET_CODE (XEXP (XEXP (x, 0), 0)) == SIGN_EXTEND
&& GET_CODE (XEXP (XEXP (x, 0), 1)) == SIGN_EXTEND)))
{
if (speed_p)
*cost += extra_cost->mult[1].extend_add;
*cost += (rtx_cost (XEXP (XEXP (XEXP (x, 0), 0), 0), mode,
ZERO_EXTEND, 0, speed_p)
+ rtx_cost (XEXP (XEXP (XEXP (x, 0), 1), 0), mode,
ZERO_EXTEND, 0, speed_p)
+ rtx_cost (XEXP (x, 1), mode, PLUS, 1, speed_p));
return true;
}
*cost += COSTS_N_INSNS (1);
if (GET_CODE (XEXP (x, 0)) == ZERO_EXTEND
|| GET_CODE (XEXP (x, 0)) == SIGN_EXTEND)
{
if (speed_p)
*cost += (extra_cost->alu.arith
+ (GET_CODE (XEXP (x, 0)) == ZERO_EXTEND
? extra_cost->alu.arith
: extra_cost->alu.arith_shift));
*cost += (rtx_cost (XEXP (XEXP (x, 0), 0), VOIDmode, ZERO_EXTEND,
0, speed_p)
+ rtx_cost (XEXP (x, 1), mode, PLUS, 1, speed_p));
return true;
}
if (speed_p)
*cost += 2 * extra_cost->alu.arith;
return false;
}
*cost = LIBCALL_COST (2);
return false;
case IOR:
if (mode == SImode && arm_arch6 && aarch_rev16_p (x))
{
if (speed_p)
*cost += extra_cost->alu.rev;
return true;
}
case AND: case XOR:
if (mode == SImode)
{
enum rtx_code subcode = GET_CODE (XEXP (x, 0));
rtx op0 = XEXP (x, 0);
rtx shift_op, shift_reg;
if (subcode == NOT
&& (code == AND
|| (code == IOR && TARGET_THUMB2)))
op0 = XEXP (op0, 0);
shift_reg = NULL;
shift_op = shifter_op_p (op0, &shift_reg);
if (shift_op != NULL)
{
if (shift_reg)
{
if (speed_p)
*cost += extra_cost->alu.log_shift_reg;
*cost += rtx_cost (shift_reg, mode, ASHIFT, 1, speed_p);
}
else if (speed_p)
*cost += extra_cost->alu.log_shift;
*cost += (rtx_cost (shift_op, mode, ASHIFT, 0, speed_p)
+ rtx_cost (XEXP (x, 1), mode, code, 1, speed_p));
return true;
}
if (CONST_INT_P (XEXP (x, 1)))
{
int insns = arm_gen_constant (code, SImode, NULL_RTX,
INTVAL (XEXP (x, 1)), NULL_RTX,
NULL_RTX, 1, 0);
*cost = COSTS_N_INSNS (insns);
if (speed_p)
*cost += insns * extra_cost->alu.logical;
*cost += rtx_cost (op0, mode, code, 0, speed_p);
return true;
}
if (speed_p)
*cost += extra_cost->alu.logical;
*cost += (rtx_cost (op0, mode, code, 0, speed_p)
+ rtx_cost (XEXP (x, 1), mode, code, 1, speed_p));
return true;
}
if (mode == DImode)
{
rtx op0 = XEXP (x, 0);
enum rtx_code subcode = GET_CODE (op0);
*cost += COSTS_N_INSNS (1);
if (subcode == NOT
&& (code == AND
|| (code == IOR && TARGET_THUMB2)))
op0 = XEXP (op0, 0);
if (GET_CODE (op0) == ZERO_EXTEND)
{
if (speed_p)
*cost += 2 * extra_cost->alu.logical;
*cost += (rtx_cost (XEXP (op0, 0), VOIDmode, ZERO_EXTEND,
0, speed_p)
+ rtx_cost (XEXP (x, 1), mode, code, 0, speed_p));
return true;
}
else if (GET_CODE (op0) == SIGN_EXTEND)
{
if (speed_p)
*cost += extra_cost->alu.logical + extra_cost->alu.log_shift;
*cost += (rtx_cost (XEXP (op0, 0), VOIDmode, SIGN_EXTEND,
0, speed_p)
+ rtx_cost (XEXP (x, 1), mode, code, 0, speed_p));
return true;
}
if (speed_p)
*cost += 2 * extra_cost->alu.logical;
return true;
}
*cost = LIBCALL_COST (2);
return false;
case MULT:
if (TARGET_HARD_FLOAT && GET_MODE_CLASS (mode) == MODE_FLOAT
&& (mode == SFmode || !TARGET_VFP_SINGLE))
{
rtx op0 = XEXP (x, 0);
if (GET_CODE (op0) == NEG && !flag_rounding_math)
op0 = XEXP (op0, 0);
if (speed_p)
*cost += extra_cost->fp[mode != SFmode].mult;
*cost += (rtx_cost (op0, mode, MULT, 0, speed_p)
+ rtx_cost (XEXP (x, 1), mode, MULT, 1, speed_p));
return true;
}
else if (GET_MODE_CLASS (mode) == MODE_FLOAT)
{
*cost = LIBCALL_COST (2);
return false;
}
if (mode == SImode)
{
if (TARGET_DSP_MULTIPLY
&& ((GET_CODE (XEXP (x, 0)) == SIGN_EXTEND
&& (GET_CODE (XEXP (x, 1)) == SIGN_EXTEND
|| (GET_CODE (XEXP (x, 1)) == ASHIFTRT
&& CONST_INT_P (XEXP (XEXP (x, 1), 1))
&& INTVAL (XEXP (XEXP (x, 1), 1)) == 16)))
|| (GET_CODE (XEXP (x, 0)) == ASHIFTRT
&& CONST_INT_P (XEXP (XEXP (x, 0), 1))
&& INTVAL (XEXP (XEXP (x, 0), 1)) == 16
&& (GET_CODE (XEXP (x, 1)) == SIGN_EXTEND
|| (GET_CODE (XEXP (x, 1)) == ASHIFTRT
&& CONST_INT_P (XEXP (XEXP (x, 1), 1))
&& (INTVAL (XEXP (XEXP (x, 1), 1))
== 16))))))
{
if (speed_p)
*cost += extra_cost->mult[0].extend;
*cost += rtx_cost (XEXP (XEXP (x, 0), 0), mode,
SIGN_EXTEND, 0, speed_p);
*cost += rtx_cost (XEXP (XEXP (x, 1), 0), mode,
SIGN_EXTEND, 1, speed_p);
return true;
}
if (speed_p)
*cost += extra_cost->mult[0].simple;
return false;
}
if (mode == DImode)
{
if (arm_arch3m
&& ((GET_CODE (XEXP (x, 0)) == ZERO_EXTEND
&& GET_CODE (XEXP (x, 1)) == ZERO_EXTEND)
|| (GET_CODE (XEXP (x, 0)) == SIGN_EXTEND
&& GET_CODE (XEXP (x, 1)) == SIGN_EXTEND)))
{
if (speed_p)
*cost += extra_cost->mult[1].extend;
*cost += (rtx_cost (XEXP (XEXP (x, 0), 0), VOIDmode,
ZERO_EXTEND, 0, speed_p)
+ rtx_cost (XEXP (XEXP (x, 1), 0), VOIDmode,
ZERO_EXTEND, 0, speed_p));
return true;
}
*cost = LIBCALL_COST (2);
return false;
}
*cost = LIBCALL_COST (2);
return false;
case NEG:
if (TARGET_HARD_FLOAT && GET_MODE_CLASS (mode) == MODE_FLOAT
&& (mode == SFmode || !TARGET_VFP_SINGLE))
{
if (GET_CODE (XEXP (x, 0)) == MULT)
{
*cost = rtx_cost (XEXP (x, 0), mode, NEG, 0, speed_p);
return true;
}
if (speed_p)
*cost += extra_cost->fp[mode != SFmode].neg;
return false;
}
else if (GET_MODE_CLASS (mode) == MODE_FLOAT)
{
*cost = LIBCALL_COST (1);
return false;
}
if (mode == SImode)
{
if (GET_CODE (XEXP (x, 0)) == ABS)
{
*cost += COSTS_N_INSNS (1);
if (speed_p)
*cost += (extra_cost->alu.log_shift
+ extra_cost->alu.arith_shift);
*cost += rtx_cost (XEXP (XEXP (x, 0), 0), mode, ABS, 0, speed_p);
return true;
}
if (GET_RTX_CLASS (GET_CODE (XEXP (x, 0))) == RTX_COMPARE
|| GET_RTX_CLASS (GET_CODE (XEXP (x, 0))) == RTX_COMM_COMPARE)
{
*cost += COSTS_N_INSNS (1);
if (!(REG_P (XEXP (XEXP (x, 0), 0))
&& REGNO (XEXP (XEXP (x, 0), 0)) == CC_REGNUM
&& XEXP (XEXP (x, 0), 1) == const0_rtx))
{
mode = GET_MODE (XEXP (XEXP (x, 0), 0));
*cost += (COSTS_N_INSNS (1)
+ rtx_cost (XEXP (XEXP (x, 0), 0), mode, COMPARE,
0, speed_p)
+ rtx_cost (XEXP (XEXP (x, 0), 1), mode, COMPARE,
1, speed_p));
if (speed_p)
*cost += extra_cost->alu.arith;
}
return true;
}
if (speed_p)
*cost += extra_cost->alu.arith;
return false;
}
if (GET_MODE_CLASS (mode) == MODE_INT
&& GET_MODE_SIZE (mode) < 4)
{
*cost += 1;
if (speed_p)
*cost += extra_cost->alu.arith;
return false;
}
if (mode == DImode)
{
*cost += COSTS_N_INSNS (1);
if (speed_p)
*cost += 2 * extra_cost->alu.arith;
return false;
}
*cost = LIBCALL_COST (1);
return false;
case NOT:
if (mode == SImode)
{
rtx shift_op;
rtx shift_reg = NULL;
shift_op = shifter_op_p (XEXP (x, 0), &shift_reg);
if (shift_op)
{
if (shift_reg != NULL)
{
if (speed_p)
*cost += extra_cost->alu.log_shift_reg;
*cost += rtx_cost (shift_reg, mode, ASHIFT, 1, speed_p);
}
else if (speed_p)
*cost += extra_cost->alu.log_shift;
*cost += rtx_cost (shift_op, mode, ASHIFT, 0, speed_p);
return true;
}
if (speed_p)
*cost += extra_cost->alu.logical;
return false;
}
if (mode == DImode)
{
*cost += COSTS_N_INSNS (1);
return false;
}
*cost += LIBCALL_COST (1);
return false;
case IF_THEN_ELSE:
{
if (GET_CODE (XEXP (x, 1)) == PC || GET_CODE (XEXP (x, 2)) == PC)
{
*cost += COSTS_N_INSNS (3);
return true;
}
int op1cost = rtx_cost (XEXP (x, 1), mode, SET, 1, speed_p);
int op2cost = rtx_cost (XEXP (x, 2), mode, SET, 1, speed_p);
*cost = rtx_cost (XEXP (x, 0), mode, IF_THEN_ELSE, 0, speed_p);
if (REG_P (XEXP (x, 1)))
*cost += op2cost;
else if (REG_P (XEXP (x, 2)))
*cost += op1cost;
else
{
if (speed_p)
{
if (extra_cost->alu.non_exec_costs_exec)
*cost += op1cost + op2cost + extra_cost->alu.non_exec;
else
*cost += MAX (op1cost, op2cost) + extra_cost->alu.non_exec;
}
else
*cost += op1cost + op2cost;
}
}
return true;
case COMPARE:
if (cc_register (XEXP (x, 0), VOIDmode) && XEXP (x, 1) == const0_rtx)
*cost = 0;
else
{
machine_mode op0mode;
op0mode = GET_MODE (XEXP (x, 0));
if (TARGET_HARD_FLOAT && GET_MODE_CLASS (op0mode) == MODE_FLOAT
&& (op0mode == SFmode || !TARGET_VFP_SINGLE))
{
if (speed_p)
*cost += extra_cost->fp[op0mode != SFmode].compare;
if (XEXP (x, 1) == CONST0_RTX (op0mode))
{
*cost += rtx_cost (XEXP (x, 0), op0mode, code, 0, speed_p);
return true;
}
return false;
}
else if (GET_MODE_CLASS (op0mode) == MODE_FLOAT)
{
*cost = LIBCALL_COST (2);
return false;
}
if (op0mode == DImode)
{
*cost += COSTS_N_INSNS (1);
if (speed_p)
*cost += 2 * extra_cost->alu.arith;
return false;
}
if (op0mode == SImode)
{
rtx shift_op;
rtx shift_reg;
if (XEXP (x, 1) == const0_rtx
&& !(REG_P (XEXP (x, 0))
|| (GET_CODE (XEXP (x, 0)) == SUBREG
&& REG_P (SUBREG_REG (XEXP (x, 0))))))
{
*cost = rtx_cost (XEXP (x, 0), op0mode, COMPARE, 0, speed_p);
if (speed_p
&& GET_CODE (XEXP (x, 0)) == MULT
&& !power_of_two_operand (XEXP (XEXP (x, 0), 1), mode))
*cost += extra_cost->mult[0].flag_setting;
if (speed_p
&& GET_CODE (XEXP (x, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == MULT
&& !power_of_two_operand (XEXP (XEXP (XEXP (x, 0),
0), 1), mode))
*cost += extra_cost->mult[0].flag_setting;
return true;
}
shift_reg = NULL;
shift_op = shifter_op_p (XEXP (x, 0), &shift_reg);
if (shift_op != NULL)
{
if (shift_reg != NULL)
{
*cost += rtx_cost (shift_reg, op0mode, ASHIFT,
1, speed_p);
if (speed_p)
*cost += extra_cost->alu.arith_shift_reg;
}
else if (speed_p)
*cost += extra_cost->alu.arith_shift;
*cost += rtx_cost (shift_op, op0mode, ASHIFT, 0, speed_p);
*cost += rtx_cost (XEXP (x, 1), op0mode, COMPARE, 1, speed_p);
return true;
}
if (speed_p)
*cost += extra_cost->alu.arith;
if (CONST_INT_P (XEXP (x, 1))
&& const_ok_for_op (INTVAL (XEXP (x, 1)), COMPARE))
{
*cost += rtx_cost (XEXP (x, 0), op0mode, COMPARE, 0, speed_p);
return true;
}
return false;
}
*cost = LIBCALL_COST (2);
return false;
}
return true;
case EQ:
case NE:
case LT:
case LE:
case GT:
case GE:
case LTU:
case LEU:
case GEU:
case GTU:
case ORDERED:
case UNORDERED:
case UNEQ:
case UNLE:
case UNLT:
case UNGE:
case UNGT:
case LTGT:
if (outer_code == SET)
{
if (REG_P (XEXP (x, 0)) && REGNO (XEXP (x, 0)) == CC_REGNUM
&& XEXP (x, 1) == const0_rtx)
{
*cost += COSTS_N_INSNS (TARGET_THUMB ? 2 : 1);
return true;
}
if (XEXP (x, 1) == const0_rtx)
{
switch (code)
{
case LT:
if (speed_p)
*cost += extra_cost->alu.shift;
break;
case EQ:
case NE:
*cost += COSTS_N_INSNS (1);
break;
case LE:
*cost += COSTS_N_INSNS (1);
if (speed_p)
*cost += extra_cost->alu.arith_shift;
break;
case GT:
*cost += COSTS_N_INSNS (1);
if (speed_p)
*cost += (extra_cost->alu.arith_shift
+ extra_cost->alu.shift);
break;
case GE:
*cost += COSTS_N_INSNS (1);
if (speed_p)
*cost += extra_cost->alu.shift;
break;
default:
*cost = COSTS_N_INSNS (3);
break;
}
*cost += rtx_cost (XEXP (x, 0), mode, code, 0, speed_p);
return true;
}
else
{
*cost += COSTS_N_INSNS (TARGET_THUMB ? 3 : 2);
if (CONST_INT_P (XEXP (x, 1))
&& const_ok_for_op (INTVAL (XEXP (x, 1)), COMPARE))
{
*cost += rtx_cost (XEXP (x, 0), mode, code, 0, speed_p);
return true;
}
return false;
}
}
else if (REG_P (XEXP (x, 0)) && REGNO (XEXP (x, 0)) == CC_REGNUM
&& XEXP (x, 1) == const0_rtx)
{
*cost = 0;
return true;
}
return false;
case ABS:
if (TARGET_HARD_FLOAT && GET_MODE_CLASS (mode) == MODE_FLOAT
&& (mode == SFmode || !TARGET_VFP_SINGLE))
{
if (speed_p)
*cost += extra_cost->fp[mode != SFmode].neg;
return false;
}
else if (GET_MODE_CLASS (mode) == MODE_FLOAT)
{
*cost = LIBCALL_COST (1);
return false;
}
if (mode == SImode)
{
if (speed_p)
*cost += extra_cost->alu.log_shift + extra_cost->alu.arith_shift;
return false;
}
*cost = LIBCALL_COST (1);
return false;
case SIGN_EXTEND:
if ((arm_arch4 || GET_MODE (XEXP (x, 0)) == SImode)
&& MEM_P (XEXP (x, 0)))
{
if (mode == DImode)
*cost += COSTS_N_INSNS (1);
if (!speed_p)
return true;
if (GET_MODE (XEXP (x, 0)) == SImode)
*cost += extra_cost->ldst.load;
else
*cost += extra_cost->ldst.load_sign_extend;
if (mode == DImode)
*cost += extra_cost->alu.shift;
return true;
}
if (GET_MODE (XEXP (x, 0)) != SImode && arm_arch6)
{
*cost += rtx_cost (XEXP (x, 0), VOIDmode, code, 0, speed_p);
if (speed_p)
*cost += extra_cost->alu.extend;
}
else if (GET_MODE (XEXP (x, 0)) != SImode)
{
*cost += COSTS_N_INSNS (1);
*cost += rtx_cost (XEXP (x, 0), VOIDmode, code, 0, speed_p);
if (speed_p)
*cost += 2 * extra_cost->alu.shift;
}
if (mode == DImode)
{
*cost += COSTS_N_INSNS (1);
if (speed_p)
*cost += extra_cost->alu.shift;
}
return true;
case ZERO_EXTEND:
if ((arm_arch4
|| GET_MODE (XEXP (x, 0)) == SImode
|| GET_MODE (XEXP (x, 0)) == QImode)
&& MEM_P (XEXP (x, 0)))
{
*cost = rtx_cost (XEXP (x, 0), VOIDmode, code, 0, speed_p);
if (mode == DImode)
*cost += COSTS_N_INSNS (1);  
return true;
}
if (GET_MODE (XEXP (x, 0)) == QImode)
{
if (speed_p)
*cost += extra_cost->alu.logical;
}
else if (GET_MODE (XEXP (x, 0)) != SImode && arm_arch6)
{
*cost += rtx_cost (XEXP (x, 0), VOIDmode, code, 0, speed_p);
if (speed_p)
*cost += extra_cost->alu.extend;
}
else if (GET_MODE (XEXP (x, 0)) != SImode)
{
*cost = COSTS_N_INSNS (2);
*cost += rtx_cost (XEXP (x, 0), VOIDmode, code, 0, speed_p);
if (speed_p)
*cost += 2 * extra_cost->alu.shift;
}
if (mode == DImode)
{
*cost += COSTS_N_INSNS (1);	
}
return true;
case CONST_INT:
*cost = 0;
if (INTVAL (x) == trunc_int_for_mode (INTVAL (x), SImode))
mode = SImode;
else
mode = DImode;
if (!(outer_code == PLUS
|| outer_code == AND
|| outer_code == IOR
|| outer_code == XOR
|| outer_code == MINUS))
outer_code = SET;
const_int_cost:
if (mode == SImode)
{
*cost += COSTS_N_INSNS (arm_gen_constant (outer_code, SImode, NULL,
INTVAL (x), NULL, NULL,
0, 0));
}
else
{
*cost += COSTS_N_INSNS (arm_gen_constant
(outer_code, SImode, NULL,
trunc_int_for_mode (INTVAL (x), SImode),
NULL, NULL, 0, 0)
+ arm_gen_constant (outer_code, SImode, NULL,
INTVAL (x) >> 32, NULL,
NULL, 0, 0));
}
return true;
case CONST:
case LABEL_REF:
case SYMBOL_REF:
if (speed_p)
{
if (arm_arch_thumb2 && !flag_pic)
*cost += COSTS_N_INSNS (1);
else
*cost += extra_cost->ldst.load;
}
else
*cost += COSTS_N_INSNS (1);
if (flag_pic)
{
*cost += COSTS_N_INSNS (1);
if (speed_p)
*cost += extra_cost->alu.arith;
}
return true;
case CONST_FIXED:
*cost = COSTS_N_INSNS (4);
return true;
case CONST_DOUBLE:
if (TARGET_HARD_FLOAT && GET_MODE_CLASS (mode) == MODE_FLOAT
&& (mode == SFmode || !TARGET_VFP_SINGLE))
{
if (vfp3_const_double_rtx (x))
{
if (speed_p)
*cost += extra_cost->fp[mode == DFmode].fpconst;
return true;
}
if (speed_p)
{
if (mode == DFmode)
*cost += extra_cost->ldst.loadd;
else
*cost += extra_cost->ldst.loadf;
}
else
*cost += COSTS_N_INSNS (1 + (mode == DFmode));
return true;
}
*cost = COSTS_N_INSNS (4);
return true;
case CONST_VECTOR:
if (TARGET_NEON
&& TARGET_HARD_FLOAT
&& (VALID_NEON_DREG_MODE (mode) || VALID_NEON_QREG_MODE (mode))
&& neon_immediate_valid_for_move (x, mode, NULL, NULL))
*cost = COSTS_N_INSNS (1);
else
*cost = COSTS_N_INSNS (4);
return true;
case HIGH:
case LO_SUM:
if (!speed_p)
*cost += 1;
return true;
case CLZ:
if (speed_p)
*cost += extra_cost->alu.clz;
return false;
case SMIN:
if (XEXP (x, 1) == const0_rtx)
{
if (speed_p)
*cost += extra_cost->alu.log_shift;
*cost += rtx_cost (XEXP (x, 0), mode, code, 0, speed_p);
return true;
}
case SMAX:
case UMIN:
case UMAX:
*cost += COSTS_N_INSNS (1);
return false;
case TRUNCATE:
if (GET_CODE (XEXP (x, 0)) == ASHIFTRT
&& CONST_INT_P (XEXP (XEXP (x, 0), 1))
&& INTVAL (XEXP (XEXP (x, 0), 1)) == 32
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == MULT
&& ((GET_CODE (XEXP (XEXP (XEXP (x, 0), 0), 0)) == SIGN_EXTEND
&& GET_CODE (XEXP (XEXP (XEXP (x, 0), 0), 1)) == SIGN_EXTEND)
|| (GET_CODE (XEXP (XEXP (XEXP (x, 0), 0), 0)) == ZERO_EXTEND
&& (GET_CODE (XEXP (XEXP (XEXP (x, 0), 0), 1))
== ZERO_EXTEND))))
{
if (speed_p)
*cost += extra_cost->mult[1].extend;
*cost += (rtx_cost (XEXP (XEXP (XEXP (x, 0), 0), 0), VOIDmode,
ZERO_EXTEND, 0, speed_p)
+ rtx_cost (XEXP (XEXP (XEXP (x, 0), 0), 1), VOIDmode,
ZERO_EXTEND, 0, speed_p));
return true;
}
*cost = LIBCALL_COST (1);
return false;
case UNSPEC_VOLATILE:
case UNSPEC:
return arm_unspec_cost (x, outer_code, speed_p, cost);
case PC:
*cost = 0;
return true;
case ZERO_EXTRACT:
case SIGN_EXTRACT:
if (arm_arch6
&& mode == SImode
&& CONST_INT_P (XEXP (x, 1))
&& CONST_INT_P (XEXP (x, 2)))
{
if (speed_p)
*cost += extra_cost->alu.bfx;
*cost += rtx_cost (XEXP (x, 0), mode, code, 0, speed_p);
return true;
}
*cost += COSTS_N_INSNS (1);
if (speed_p)
*cost += 2 * extra_cost->alu.shift;
*cost += rtx_cost (XEXP (x, 0), mode, ASHIFT, 0, speed_p);
return true;
case FLOAT_EXTEND:
if (TARGET_HARD_FLOAT)
{
if (speed_p)
*cost += extra_cost->fp[mode == DFmode].widen;
if (!TARGET_VFP5
&& GET_MODE (XEXP (x, 0)) == HFmode)
{
*cost += COSTS_N_INSNS (1);
if (speed_p)
*cost += extra_cost->fp[0].widen;
}
*cost += rtx_cost (XEXP (x, 0), VOIDmode, code, 0, speed_p);
return true;
}
*cost = LIBCALL_COST (1);
return false;
case FLOAT_TRUNCATE:
if (TARGET_HARD_FLOAT)
{
if (speed_p)
*cost += extra_cost->fp[mode == DFmode].narrow;
*cost += rtx_cost (XEXP (x, 0), VOIDmode, code, 0, speed_p);
return true;
}
*cost = LIBCALL_COST (1);
return false;
case FMA:
if (TARGET_32BIT && TARGET_HARD_FLOAT && TARGET_FMA)
{
rtx op0 = XEXP (x, 0);
rtx op1 = XEXP (x, 1);
rtx op2 = XEXP (x, 2);
if (GET_CODE (op0) == NEG)
op0 = XEXP (op0, 0);
if (GET_CODE (op2) == NEG)
op2 = XEXP (op2, 0);
*cost += rtx_cost (op0, mode, FMA, 0, speed_p);
*cost += rtx_cost (op1, mode, FMA, 1, speed_p);
*cost += rtx_cost (op2, mode, FMA, 2, speed_p);
if (speed_p)
*cost += extra_cost->fp[mode ==DFmode].fma;
return true;
}
*cost = LIBCALL_COST (3);
return false;
case FIX:
case UNSIGNED_FIX:
if (TARGET_HARD_FLOAT)
{
if (code == FIX && mode == SImode
&& GET_CODE (XEXP (x, 0)) == FIX
&& GET_MODE (XEXP (x, 0)) == SFmode
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == MULT
&& vfp3_const_double_for_bits (XEXP (XEXP (XEXP (x, 0), 0), 1))
> 0)
{
if (speed_p)
*cost += extra_cost->fp[0].toint;
*cost += rtx_cost (XEXP (XEXP (XEXP (x, 0), 0), 0), mode,
code, 0, speed_p);
return true;
}
if (GET_MODE_CLASS (mode) == MODE_INT)
{
mode = GET_MODE (XEXP (x, 0));
if (speed_p)
*cost += extra_cost->fp[mode == DFmode].toint;
if (GET_CODE (XEXP (x, 0)) == FIX)
*cost += rtx_cost (XEXP (XEXP (x, 0), 0), mode, code,
0, speed_p);
else
*cost += rtx_cost (XEXP (x, 0), mode, code, 0, speed_p);
return true;
}
else if (GET_MODE_CLASS (mode) == MODE_FLOAT
&& TARGET_VFP5)
{
if (speed_p)
*cost += extra_cost->fp[mode == DFmode].roundint;
return false;
}
}
*cost = LIBCALL_COST (1);
return false;
case FLOAT:
case UNSIGNED_FLOAT:
if (TARGET_HARD_FLOAT)
{
if (speed_p)
*cost += extra_cost->fp[mode == DFmode].fromint;
return false;
}
*cost = LIBCALL_COST (1);
return false;
case CALL:
return true;
case ASM_OPERANDS:
{
int asm_length = MAX (1, asm_str_count (ASM_OPERANDS_TEMPLATE (x)));
int num_operands = ASM_OPERANDS_INPUT_LENGTH (x);
*cost = COSTS_N_INSNS (asm_length + num_operands);
return true;
}
default:
if (mode != VOIDmode)
*cost = COSTS_N_INSNS (ARM_NUM_REGS (mode));
else
*cost = COSTS_N_INSNS (4); 
return false;
}
}
#undef HANDLE_NARROW_SHIFT_ARITH
static bool
arm_rtx_costs (rtx x, machine_mode mode ATTRIBUTE_UNUSED, int outer_code,
int opno ATTRIBUTE_UNUSED, int *total, bool speed)
{
bool result;
int code = GET_CODE (x);
gcc_assert (current_tune->insn_extra_cost);
result =  arm_rtx_costs_internal (x, (enum rtx_code) code,
(enum rtx_code) outer_code,
current_tune->insn_extra_cost,
total, speed);
if (dump_file && arm_verbose_cost)
{
print_rtl_single (dump_file, x);
fprintf (dump_file, "\n%s cost: %d (%s)\n", speed ? "Hot" : "Cold",
*total, result ? "final" : "partial");
}
return result;
}
static inline int
arm_arm_address_cost (rtx x)
{
enum rtx_code c  = GET_CODE (x);
if (c == PRE_INC || c == PRE_DEC || c == POST_INC || c == POST_DEC)
return 0;
if (c == MEM || c == LABEL_REF || c == SYMBOL_REF)
return 10;
if (c == PLUS)
{
if (CONST_INT_P (XEXP (x, 1)))
return 2;
if (ARITHMETIC_P (XEXP (x, 0)) || ARITHMETIC_P (XEXP (x, 1)))
return 3;
return 4;
}
return 6;
}
static inline int
arm_thumb_address_cost (rtx x)
{
enum rtx_code c  = GET_CODE (x);
if (c == REG)
return 1;
if (c == PLUS
&& REG_P (XEXP (x, 0))
&& CONST_INT_P (XEXP (x, 1)))
return 1;
return 2;
}
static int
arm_address_cost (rtx x, machine_mode mode ATTRIBUTE_UNUSED,
addr_space_t as ATTRIBUTE_UNUSED, bool speed ATTRIBUTE_UNUSED)
{
return TARGET_32BIT ? arm_arm_address_cost (x) : arm_thumb_address_cost (x);
}
static bool
xscale_sched_adjust_cost (rtx_insn *insn, int dep_type, rtx_insn *dep,
int * cost)
{
if (dep_type == 0
&& recog_memoized (insn) >= 0
&& recog_memoized (dep) >= 0)
{
int shift_opnum = get_attr_shift (insn);
enum attr_type attr_type = get_attr_type (dep);
if (shift_opnum != 0
&& (attr_type == TYPE_ALU_SHIFT_IMM
|| attr_type == TYPE_ALUS_SHIFT_IMM
|| attr_type == TYPE_LOGIC_SHIFT_IMM
|| attr_type == TYPE_LOGICS_SHIFT_IMM
|| attr_type == TYPE_ALU_SHIFT_REG
|| attr_type == TYPE_ALUS_SHIFT_REG
|| attr_type == TYPE_LOGIC_SHIFT_REG
|| attr_type == TYPE_LOGICS_SHIFT_REG
|| attr_type == TYPE_MOV_SHIFT
|| attr_type == TYPE_MVN_SHIFT
|| attr_type == TYPE_MOV_SHIFT_REG
|| attr_type == TYPE_MVN_SHIFT_REG))
{
rtx shifted_operand;
int opno;
extract_insn (insn);
shifted_operand = recog_data.operand[shift_opnum];
extract_insn (dep);
preprocess_constraints (dep);
for (opno = 0; opno < recog_data.n_operands; opno++)
{
if (recog_data.operand_type[opno] == OP_IN)
continue;
if (reg_overlap_mentioned_p (recog_data.operand[opno],
shifted_operand))
{
*cost = 2;
return false;
}
}
}
}
return true;
}
static bool
cortex_a9_sched_adjust_cost (rtx_insn *insn, int dep_type, rtx_insn *dep,
int * cost)
{
switch (dep_type)
{
case REG_DEP_ANTI:
*cost = 0;
return false;
case REG_DEP_TRUE:
case REG_DEP_OUTPUT:
if (recog_memoized (insn) >= 0
&& recog_memoized (dep) >= 0)
{
if (GET_CODE (PATTERN (insn)) == SET)
{
if (GET_MODE_CLASS
(GET_MODE (SET_DEST (PATTERN (insn)))) == MODE_FLOAT
|| GET_MODE_CLASS
(GET_MODE (SET_SRC (PATTERN (insn)))) == MODE_FLOAT)
{
enum attr_type attr_type_insn = get_attr_type (insn);
enum attr_type attr_type_dep = get_attr_type (dep);
if (REG_P (SET_DEST (PATTERN (insn)))
&& reg_set_p (SET_DEST (PATTERN (insn)), dep))
{
if ((attr_type_insn == TYPE_FMACS
|| attr_type_insn == TYPE_FMACD)
&& (attr_type_dep == TYPE_FMACS
|| attr_type_dep == TYPE_FMACD))
{
if (dep_type == REG_DEP_OUTPUT)
*cost = insn_default_latency (dep) - 3;
else
*cost = insn_default_latency (dep);
return false;
}
else
{
if (dep_type == REG_DEP_OUTPUT)
*cost = insn_default_latency (dep) + 1;
else
*cost = insn_default_latency (dep);
}
return false;
}
}
}
}
break;
default:
gcc_unreachable ();
}
return true;
}
static bool
fa726te_sched_adjust_cost (rtx_insn *insn, int dep_type, rtx_insn *dep,
int * cost)
{
if (dep_type == REG_DEP_TRUE
&& recog_memoized (insn) >= 0
&& recog_memoized (dep) >= 0
&& get_attr_conds (dep) == CONDS_SET)
{
if (get_attr_conds (insn) == CONDS_USE
&& get_attr_type (insn) != TYPE_BRANCH)
{
*cost = 3;
return false;
}
if (GET_CODE (PATTERN (insn)) == COND_EXEC
|| get_attr_conds (insn) == CONDS_USE)
{
*cost = 0;
return false;
}
}
return true;
}
int
arm_register_move_cost (machine_mode mode ATTRIBUTE_UNUSED,
reg_class_t from, reg_class_t to)
{
if (TARGET_32BIT)
{
if ((IS_VFP_CLASS (from) && !IS_VFP_CLASS (to))
|| (!IS_VFP_CLASS (from) && IS_VFP_CLASS (to)))
return 15;
else if ((from == IWMMXT_REGS && to != IWMMXT_REGS)
|| (from != IWMMXT_REGS && to == IWMMXT_REGS))
return 4;
else if (from == IWMMXT_GR_REGS || to == IWMMXT_GR_REGS)
return 20;
else
return 2;
}
else
{
if (from == HI_REGS || to == HI_REGS)
return 4;
else
return 2;
}
}
int
arm_memory_move_cost (machine_mode mode, reg_class_t rclass,
bool in ATTRIBUTE_UNUSED)
{
if (TARGET_32BIT)
return 10;
else
{
if (GET_MODE_SIZE (mode) < 4)
return 8;
else
return ((2 * GET_MODE_SIZE (mode)) * (rclass == LO_REGS ? 1 : 2));
}
}
static int
arm_builtin_vectorization_cost (enum vect_cost_for_stmt type_of_cost,
tree vectype,
int misalign ATTRIBUTE_UNUSED)
{
unsigned elements;
switch (type_of_cost)
{
case scalar_stmt:
return current_tune->vec_costs->scalar_stmt_cost;
case scalar_load:
return current_tune->vec_costs->scalar_load_cost;
case scalar_store:
return current_tune->vec_costs->scalar_store_cost;
case vector_stmt:
return current_tune->vec_costs->vec_stmt_cost;
case vector_load:
return current_tune->vec_costs->vec_align_load_cost;
case vector_store:
return current_tune->vec_costs->vec_store_cost;
case vec_to_scalar:
return current_tune->vec_costs->vec_to_scalar_cost;
case scalar_to_vec:
return current_tune->vec_costs->scalar_to_vec_cost;
case unaligned_load:
case vector_gather_load:
return current_tune->vec_costs->vec_unalign_load_cost;
case unaligned_store:
case vector_scatter_store:
return current_tune->vec_costs->vec_unalign_store_cost;
case cond_branch_taken:
return current_tune->vec_costs->cond_taken_branch_cost;
case cond_branch_not_taken:
return current_tune->vec_costs->cond_not_taken_branch_cost;
case vec_perm:
case vec_promote_demote:
return current_tune->vec_costs->vec_stmt_cost;
case vec_construct:
elements = TYPE_VECTOR_SUBPARTS (vectype);
return elements / 2 + 1;
default:
gcc_unreachable ();
}
}
static unsigned
arm_add_stmt_cost (void *data, int count, enum vect_cost_for_stmt kind,
struct _stmt_vec_info *stmt_info, int misalign,
enum vect_cost_model_location where)
{
unsigned *cost = (unsigned *) data;
unsigned retval = 0;
if (flag_vect_cost_model)
{
tree vectype = stmt_info ? stmt_vectype (stmt_info) : NULL_TREE;
int stmt_cost = arm_builtin_vectorization_cost (kind, vectype, misalign);
if (where == vect_body && stmt_info && stmt_in_inner_loop_p (stmt_info))
count *= 50;  
retval = (unsigned) (count * stmt_cost);
cost[where] += retval;
}
return retval;
}
static bool
cortexa7_older_only (rtx_insn *insn)
{
if (recog_memoized (insn) < 0)
return false;
switch (get_attr_type (insn))
{
case TYPE_ALU_DSP_REG:
case TYPE_ALU_SREG:
case TYPE_ALUS_SREG:
case TYPE_LOGIC_REG:
case TYPE_LOGICS_REG:
case TYPE_ADC_REG:
case TYPE_ADCS_REG:
case TYPE_ADR:
case TYPE_BFM:
case TYPE_REV:
case TYPE_MVN_REG:
case TYPE_SHIFT_IMM:
case TYPE_SHIFT_REG:
case TYPE_LOAD_BYTE:
case TYPE_LOAD_4:
case TYPE_STORE_4:
case TYPE_FFARITHS:
case TYPE_FADDS:
case TYPE_FFARITHD:
case TYPE_FADDD:
case TYPE_FMOV:
case TYPE_F_CVT:
case TYPE_FCMPS:
case TYPE_FCMPD:
case TYPE_FCONSTS:
case TYPE_FCONSTD:
case TYPE_FMULS:
case TYPE_FMACS:
case TYPE_FMULD:
case TYPE_FMACD:
case TYPE_FDIVS:
case TYPE_FDIVD:
case TYPE_F_MRC:
case TYPE_F_MRRC:
case TYPE_F_FLAG:
case TYPE_F_LOADS:
case TYPE_F_STORES:
return true;
default:
return false;
}
}
static bool
cortexa7_younger (FILE *file, int verbose, rtx_insn *insn)
{
if (recog_memoized (insn) < 0)
{
if (verbose > 5)
fprintf (file, ";; not cortexa7_younger %d\n", INSN_UID (insn));
return false;
}
switch (get_attr_type (insn))
{
case TYPE_ALU_IMM:
case TYPE_ALUS_IMM:
case TYPE_LOGIC_IMM:
case TYPE_LOGICS_IMM:
case TYPE_EXTEND:
case TYPE_MVN_IMM:
case TYPE_MOV_IMM:
case TYPE_MOV_REG:
case TYPE_MOV_SHIFT:
case TYPE_MOV_SHIFT_REG:
case TYPE_BRANCH:
case TYPE_CALL:
return true;
default:
return false;
}
}
static void
cortexa7_sched_reorder (FILE *file, int verbose, rtx_insn **ready,
int *n_readyp, int clock)
{
int i;
int first_older_only = -1, first_younger = -1;
if (verbose > 5)
fprintf (file,
";; sched_reorder for cycle %d with %d insns in ready list\n",
clock,
*n_readyp);
for (i = *n_readyp - 1; i >= 0; i--)
{
rtx_insn *insn = ready[i];
if (cortexa7_older_only (insn))
{
first_older_only = i;
if (verbose > 5)
fprintf (file, ";; reorder older found %d\n", INSN_UID (insn));
break;
}
else if (cortexa7_younger (file, verbose, insn) && first_younger == -1)
first_younger = i;
}
if (first_younger == -1)
{
if (verbose > 5)
fprintf (file, ";; sched_reorder nothing to reorder as no younger\n");
return;
}
if (first_older_only == -1)
{
if (verbose > 5)
fprintf (file, ";; sched_reorder nothing to reorder as no older_only\n");
return;
}
if (verbose > 5)
fprintf (file, ";; cortexa7_sched_reorder insn %d before %d\n",
INSN_UID(ready [first_older_only]),
INSN_UID(ready [first_younger]));
rtx_insn *first_older_only_insn = ready [first_older_only];
for (i = first_older_only; i < first_younger; i++)
{
ready[i] = ready[i+1];
}
ready[i] = first_older_only_insn;
return;
}
static int
arm_sched_reorder (FILE *file, int verbose, rtx_insn **ready, int *n_readyp,
int clock)
{
switch (arm_tune)
{
case TARGET_CPU_cortexa7:
cortexa7_sched_reorder (file, verbose, ready, n_readyp, clock);
break;
default:
break;
}
return arm_issue_rate ();
}
static int
arm_adjust_cost (rtx_insn *insn, int dep_type, rtx_insn *dep, int cost,
unsigned int)
{
rtx i_pat, d_pat;
if (TARGET_THUMB1
&& dep_type == 0
&& recog_memoized (insn) == CODE_FOR_cbranchsi4_insn
&& recog_memoized (dep) >= 0
&& get_attr_conds (dep) == CONDS_SET)
return 0;
if (current_tune->sched_adjust_cost != NULL)
{
if (!current_tune->sched_adjust_cost (insn, dep_type, dep, &cost))
return cost;
}
if (dep_type == REG_DEP_ANTI
|| dep_type == REG_DEP_OUTPUT)
return 0;
if (dep_type == 0
&& CALL_P (insn))
return 1;
if ((i_pat = single_set (insn)) != NULL
&& MEM_P (SET_SRC (i_pat))
&& (d_pat = single_set (dep)) != NULL
&& MEM_P (SET_DEST (d_pat)))
{
rtx src_mem = XEXP (SET_SRC (i_pat), 0);
if ((GET_CODE (src_mem) == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (src_mem))
|| reg_mentioned_p (stack_pointer_rtx, src_mem)
|| reg_mentioned_p (frame_pointer_rtx, src_mem)
|| reg_mentioned_p (hard_frame_pointer_rtx, src_mem))
return 1;
}
return cost;
}
int
arm_max_conditional_execute (void)
{
return max_insns_skipped;
}
static int
arm_default_branch_cost (bool speed_p, bool predictable_p ATTRIBUTE_UNUSED)
{
if (TARGET_32BIT)
return (TARGET_THUMB2 && !speed_p) ? 1 : 4;
else
return (optimize > 0) ? 2 : 0;
}
static int
arm_cortex_a5_branch_cost (bool speed_p, bool predictable_p)
{
return speed_p ? 0 : arm_default_branch_cost (speed_p, predictable_p);
}
static int
arm_cortex_m_branch_cost (bool speed_p, bool predictable_p)
{
return (TARGET_32BIT && speed_p) ? 1
: arm_default_branch_cost (speed_p, predictable_p);
}
static int
arm_cortex_m7_branch_cost (bool speed_p, bool predictable_p)
{
return speed_p ? 0 : arm_default_branch_cost (speed_p, predictable_p);
}
static bool fp_consts_inited = false;
static REAL_VALUE_TYPE value_fp0;
static void
init_fp_table (void)
{
REAL_VALUE_TYPE r;
r = REAL_VALUE_ATOF ("0", DFmode);
value_fp0 = r;
fp_consts_inited = true;
}
int
arm_const_double_rtx (rtx x)
{
const REAL_VALUE_TYPE *r;
if (!fp_consts_inited)
init_fp_table ();
r = CONST_DOUBLE_REAL_VALUE (x);
if (REAL_VALUE_MINUS_ZERO (*r))
return 0;
if (real_equal (r, &value_fp0))
return 1;
return 0;
}
static int
vfp3_const_double_index (rtx x)
{
REAL_VALUE_TYPE r, m;
int sign, exponent;
unsigned HOST_WIDE_INT mantissa, mant_hi;
unsigned HOST_WIDE_INT mask;
int point_pos = 2 * HOST_BITS_PER_WIDE_INT - 1;
bool fail;
if (!TARGET_VFP3 || !CONST_DOUBLE_P (x))
return -1;
r = *CONST_DOUBLE_REAL_VALUE (x);
if (REAL_VALUE_ISINF (r) || REAL_VALUE_ISNAN (r) || REAL_VALUE_MINUS_ZERO (r))
return -1;
sign = REAL_VALUE_NEGATIVE (r) ? 1 : 0;
r = real_value_abs (&r);
exponent = REAL_EXP (&r);
real_ldexp (&m, &r, point_pos - exponent);
wide_int w = real_to_integer (&m, &fail, HOST_BITS_PER_WIDE_INT * 2);
mantissa = w.elt (0);
mant_hi = w.elt (1);
if (mantissa != 0)
return -1;
point_pos -= HOST_BITS_PER_WIDE_INT;
mantissa = mant_hi;
mask = (HOST_WIDE_INT_1U << (point_pos - 5)) - 1;
if ((mantissa & mask) != 0)
return -1;
mantissa >>= point_pos - 5;
if (mantissa == 0)
return -1;
gcc_assert (mantissa >= 16 && mantissa <= 31);
exponent = 5 - exponent;
if (exponent < 0 || exponent > 7)
return -1;
return (sign << 7) | ((exponent ^ 3) << 4) | (mantissa - 16);
}
int
vfp3_const_double_rtx (rtx x)
{
if (!TARGET_VFP3)
return 0;
return vfp3_const_double_index (x) != -1;
}
static int
neon_valid_immediate (rtx op, machine_mode mode, int inverse,
rtx *modconst, int *elementwidth)
{
#define CHECK(STRIDE, ELSIZE, CLASS, TEST)	\
matches = 1;					\
for (i = 0; i < idx; i += (STRIDE))		\
if (!(TEST))				\
matches = 0;				\
if (matches)					\
{						\
immtype = (CLASS);			\
elsize = (ELSIZE);			\
break;					\
}
unsigned int i, elsize = 0, idx = 0, n_elts;
unsigned int innersize;
unsigned char bytes[16];
int immtype = -1, matches;
unsigned int invmask = inverse ? 0xff : 0;
bool vector = GET_CODE (op) == CONST_VECTOR;
if (vector)
n_elts = CONST_VECTOR_NUNITS (op);
else
{
n_elts = 1;
if (mode == VOIDmode)
mode = DImode;
}
innersize = GET_MODE_UNIT_SIZE (mode);
if (GET_MODE_CLASS (mode) == MODE_VECTOR_FLOAT)
{
rtx el0 = CONST_VECTOR_ELT (op, 0);
if (!vfp3_const_double_rtx (el0) && el0 != CONST0_RTX (GET_MODE (el0)))
return -1;
if (GET_MODE_INNER (mode) == HFmode)
return -1;
if (!const_vec_duplicate_p (op))
return -1;
if (modconst)
*modconst = CONST_VECTOR_ELT (op, 0);
if (elementwidth)
*elementwidth = 0;
if (el0 == CONST0_RTX (GET_MODE (el0)))
return 19;
else
return 18;
}
if (BYTES_BIG_ENDIAN && vector && !const_vec_duplicate_p (op))
return -1;
for (i = 0; i < n_elts; i++)
{
rtx el = vector ? CONST_VECTOR_ELT (op, i) : op;
unsigned HOST_WIDE_INT elpart;
gcc_assert (CONST_INT_P (el));
elpart = INTVAL (el);
for (unsigned int byte = 0; byte < innersize; byte++)
{
bytes[idx++] = (elpart & 0xff) ^ invmask;
elpart >>= BITS_PER_UNIT;
}
}
gcc_assert (idx == GET_MODE_SIZE (mode));
do
{
CHECK (4, 32, 0, bytes[i] == bytes[0] && bytes[i + 1] == 0
&& bytes[i + 2] == 0 && bytes[i + 3] == 0);
CHECK (4, 32, 1, bytes[i] == 0 && bytes[i + 1] == bytes[1]
&& bytes[i + 2] == 0 && bytes[i + 3] == 0);
CHECK (4, 32, 2, bytes[i] == 0 && bytes[i + 1] == 0
&& bytes[i + 2] == bytes[2] && bytes[i + 3] == 0);
CHECK (4, 32, 3, bytes[i] == 0 && bytes[i + 1] == 0
&& bytes[i + 2] == 0 && bytes[i + 3] == bytes[3]);
CHECK (2, 16, 4, bytes[i] == bytes[0] && bytes[i + 1] == 0);
CHECK (2, 16, 5, bytes[i] == 0 && bytes[i + 1] == bytes[1]);
CHECK (4, 32, 6, bytes[i] == bytes[0] && bytes[i + 1] == 0xff
&& bytes[i + 2] == 0xff && bytes[i + 3] == 0xff);
CHECK (4, 32, 7, bytes[i] == 0xff && bytes[i + 1] == bytes[1]
&& bytes[i + 2] == 0xff && bytes[i + 3] == 0xff);
CHECK (4, 32, 8, bytes[i] == 0xff && bytes[i + 1] == 0xff
&& bytes[i + 2] == bytes[2] && bytes[i + 3] == 0xff);
CHECK (4, 32, 9, bytes[i] == 0xff && bytes[i + 1] == 0xff
&& bytes[i + 2] == 0xff && bytes[i + 3] == bytes[3]);
CHECK (2, 16, 10, bytes[i] == bytes[0] && bytes[i + 1] == 0xff);
CHECK (2, 16, 11, bytes[i] == 0xff && bytes[i + 1] == bytes[1]);
CHECK (4, 32, 12, bytes[i] == 0xff && bytes[i + 1] == bytes[1]
&& bytes[i + 2] == 0 && bytes[i + 3] == 0);
CHECK (4, 32, 13, bytes[i] == 0 && bytes[i + 1] == bytes[1]
&& bytes[i + 2] == 0xff && bytes[i + 3] == 0xff);
CHECK (4, 32, 14, bytes[i] == 0xff && bytes[i + 1] == 0xff
&& bytes[i + 2] == bytes[2] && bytes[i + 3] == 0);
CHECK (4, 32, 15, bytes[i] == 0 && bytes[i + 1] == 0
&& bytes[i + 2] == bytes[2] && bytes[i + 3] == 0xff);
CHECK (1, 8, 16, bytes[i] == bytes[0]);
CHECK (1, 64, 17, (bytes[i] == 0 || bytes[i] == 0xff)
&& bytes[i] == bytes[(i + 8) % idx]);
}
while (0);
if (immtype == -1)
return -1;
if (elementwidth)
*elementwidth = elsize;
if (modconst)
{
unsigned HOST_WIDE_INT imm = 0;
if (invmask != 0)
for (i = 0; i < idx; i++)
bytes[i] ^= invmask;
if (immtype == 17)
{
gcc_assert (sizeof (HOST_WIDE_INT) == 8);
for (i = 0; i < 8; i++)
imm |= (unsigned HOST_WIDE_INT) (bytes[i] ? 0xff : 0)
<< (i * BITS_PER_UNIT);
*modconst = GEN_INT (imm);
}
else
{
unsigned HOST_WIDE_INT imm = 0;
for (i = 0; i < elsize / BITS_PER_UNIT; i++)
imm |= (unsigned HOST_WIDE_INT) bytes[i] << (i * BITS_PER_UNIT);
*modconst = GEN_INT (imm);
}
}
return immtype;
#undef CHECK
}
int
neon_immediate_valid_for_move (rtx op, machine_mode mode,
rtx *modconst, int *elementwidth)
{
rtx tmpconst;
int tmpwidth;
int retval = neon_valid_immediate (op, mode, 0, &tmpconst, &tmpwidth);
if (retval == -1)
return 0;
if (modconst)
*modconst = tmpconst;
if (elementwidth)
*elementwidth = tmpwidth;
return 1;
}
int
neon_immediate_valid_for_logic (rtx op, machine_mode mode, int inverse,
rtx *modconst, int *elementwidth)
{
rtx tmpconst;
int tmpwidth;
int retval = neon_valid_immediate (op, mode, inverse, &tmpconst, &tmpwidth);
if (retval < 0 || retval > 5)
return 0;
if (modconst)
*modconst = tmpconst;
if (elementwidth)
*elementwidth = tmpwidth;
return 1;
}
int
neon_immediate_valid_for_shift (rtx op, machine_mode mode,
rtx *modconst, int *elementwidth,
bool isleftshift)
{
unsigned int innersize = GET_MODE_UNIT_SIZE (mode);
unsigned int n_elts = CONST_VECTOR_NUNITS (op), i;
unsigned HOST_WIDE_INT last_elt = 0;
unsigned HOST_WIDE_INT maxshift;
for (i = 0; i < n_elts; i++)
{
rtx el = CONST_VECTOR_ELT (op, i);
unsigned HOST_WIDE_INT elpart;
if (CONST_INT_P (el))
elpart = INTVAL (el);
else if (CONST_DOUBLE_P (el))
return 0;
else
gcc_unreachable ();
if (i != 0 && elpart != last_elt)
return 0;
last_elt = elpart;
}
maxshift = innersize * 8;
if (isleftshift)
{
if (last_elt >= maxshift)
return 0;
}
else
{
if (last_elt == 0 || last_elt > maxshift)
return 0;
}
if (elementwidth)
*elementwidth = innersize * 8;
if (modconst)
*modconst = CONST_VECTOR_ELT (op, 0);
return 1;
}
char *
neon_output_logic_immediate (const char *mnem, rtx *op2, machine_mode mode,
int inverse, int quad)
{
int width, is_valid;
static char templ[40];
is_valid = neon_immediate_valid_for_logic (*op2, mode, inverse, op2, &width);
gcc_assert (is_valid != 0);
if (quad)
sprintf (templ, "%s.i%d\t%%q0, %%2", mnem, width);
else
sprintf (templ, "%s.i%d\t%%P0, %%2", mnem, width);
return templ;
}
char *
neon_output_shift_immediate (const char *mnem, char sign, rtx *op2,
machine_mode mode, int quad,
bool isleftshift)
{
int width, is_valid;
static char templ[40];
is_valid = neon_immediate_valid_for_shift (*op2, mode, op2, &width, isleftshift);
gcc_assert (is_valid != 0);
if (quad)
sprintf (templ, "%s.%c%d\t%%q0, %%q1, %%2", mnem, sign, width);
else
sprintf (templ, "%s.%c%d\t%%P0, %%P1, %%2", mnem, sign, width);
return templ;
}
void
neon_pairwise_reduce (rtx op0, rtx op1, machine_mode mode,
rtx (*reduc) (rtx, rtx, rtx))
{
unsigned int i, parts = GET_MODE_SIZE (mode) / GET_MODE_UNIT_SIZE (mode);
rtx tmpsum = op1;
for (i = parts / 2; i >= 1; i /= 2)
{
rtx dest = (i == 1) ? op0 : gen_reg_rtx (mode);
emit_insn (reduc (dest, tmpsum, tmpsum));
tmpsum = dest;
}
}
static rtx
neon_vdup_constant (rtx vals)
{
machine_mode mode = GET_MODE (vals);
machine_mode inner_mode = GET_MODE_INNER (mode);
rtx x;
if (GET_CODE (vals) != CONST_VECTOR || GET_MODE_SIZE (inner_mode) > 4)
return NULL_RTX;
if (!const_vec_duplicate_p (vals, &x))
return NULL_RTX;
x = copy_to_mode_reg (inner_mode, x);
return gen_vec_duplicate (mode, x);
}
rtx
neon_make_constant (rtx vals)
{
machine_mode mode = GET_MODE (vals);
rtx target;
rtx const_vec = NULL_RTX;
int n_elts = GET_MODE_NUNITS (mode);
int n_const = 0;
int i;
if (GET_CODE (vals) == CONST_VECTOR)
const_vec = vals;
else if (GET_CODE (vals) == PARALLEL)
{
for (i = 0; i < n_elts; ++i)
{
rtx x = XVECEXP (vals, 0, i);
if (CONST_INT_P (x) || CONST_DOUBLE_P (x))
n_const++;
}
if (n_const == n_elts)
const_vec = gen_rtx_CONST_VECTOR (mode, XVEC (vals, 0));
}
else
gcc_unreachable ();
if (const_vec != NULL
&& neon_immediate_valid_for_move (const_vec, mode, NULL, NULL))
return const_vec;
else if ((target = neon_vdup_constant (vals)) != NULL_RTX)
return target;
else if (const_vec != NULL_RTX)
return const_vec;
else
return NULL_RTX;
}
void
neon_expand_vector_init (rtx target, rtx vals)
{
machine_mode mode = GET_MODE (target);
machine_mode inner_mode = GET_MODE_INNER (mode);
int n_elts = GET_MODE_NUNITS (mode);
int n_var = 0, one_var = -1;
bool all_same = true;
rtx x, mem;
int i;
for (i = 0; i < n_elts; ++i)
{
x = XVECEXP (vals, 0, i);
if (!CONSTANT_P (x))
++n_var, one_var = i;
if (i > 0 && !rtx_equal_p (x, XVECEXP (vals, 0, 0)))
all_same = false;
}
if (n_var == 0)
{
rtx constant = neon_make_constant (vals);
if (constant != NULL_RTX)
{
emit_move_insn (target, constant);
return;
}
}
if (all_same && GET_MODE_SIZE (inner_mode) <= 4)
{
x = copy_to_mode_reg (inner_mode, XVECEXP (vals, 0, 0));
emit_insn (gen_rtx_SET (target, gen_vec_duplicate (mode, x)));
return;
}
if (n_var == 1)
{
rtx copy = copy_rtx (vals);
rtx index = GEN_INT (one_var);
XVECEXP (copy, 0, one_var) = XVECEXP (vals, 0, (one_var + 1) % n_elts);
neon_expand_vector_init (target, copy);
x = copy_to_mode_reg (inner_mode, XVECEXP (vals, 0, one_var));
switch (mode)
{
case E_V8QImode:
emit_insn (gen_neon_vset_lanev8qi (target, x, target, index));
break;
case E_V16QImode:
emit_insn (gen_neon_vset_lanev16qi (target, x, target, index));
break;
case E_V4HImode:
emit_insn (gen_neon_vset_lanev4hi (target, x, target, index));
break;
case E_V8HImode:
emit_insn (gen_neon_vset_lanev8hi (target, x, target, index));
break;
case E_V2SImode:
emit_insn (gen_neon_vset_lanev2si (target, x, target, index));
break;
case E_V4SImode:
emit_insn (gen_neon_vset_lanev4si (target, x, target, index));
break;
case E_V2SFmode:
emit_insn (gen_neon_vset_lanev2sf (target, x, target, index));
break;
case E_V4SFmode:
emit_insn (gen_neon_vset_lanev4sf (target, x, target, index));
break;
case E_V2DImode:
emit_insn (gen_neon_vset_lanev2di (target, x, target, index));
break;
default:
gcc_unreachable ();
}
return;
}
mem = assign_stack_temp (mode, GET_MODE_SIZE (mode));
for (i = 0; i < n_elts; i++)
emit_move_insn (adjust_address_nv (mem, inner_mode,
i * GET_MODE_SIZE (inner_mode)),
XVECEXP (vals, 0, i));
emit_move_insn (target, mem);
}
static void
bounds_check (rtx operand, HOST_WIDE_INT low, HOST_WIDE_INT high,
const_tree exp, const char *desc)
{
HOST_WIDE_INT lane;
gcc_assert (CONST_INT_P (operand));
lane = INTVAL (operand);
if (lane < low || lane >= high)
{
if (exp)
error ("%K%s %wd out of range %wd - %wd",
exp, desc, lane, low, high - 1);
else
error ("%s %wd out of range %wd - %wd", desc, lane, low, high - 1);
}
}
void
neon_lane_bounds (rtx operand, HOST_WIDE_INT low, HOST_WIDE_INT high,
const_tree exp)
{
bounds_check (operand, low, high, exp, "lane");
}
void
arm_const_bounds (rtx operand, HOST_WIDE_INT low, HOST_WIDE_INT high)
{
bounds_check (operand, low, high, NULL_TREE, "constant");
}
HOST_WIDE_INT
neon_element_bits (machine_mode mode)
{
return GET_MODE_UNIT_BITSIZE (mode);
}

int
arm_coproc_mem_operand (rtx op, bool wb)
{
rtx ind;
if (! (reload_in_progress || reload_completed || lra_in_progress)
&& (   reg_mentioned_p (frame_pointer_rtx, op)
|| reg_mentioned_p (arg_pointer_rtx, op)
|| reg_mentioned_p (virtual_incoming_args_rtx, op)
|| reg_mentioned_p (virtual_outgoing_args_rtx, op)
|| reg_mentioned_p (virtual_stack_dynamic_rtx, op)
|| reg_mentioned_p (virtual_stack_vars_rtx, op)))
return FALSE;
if (!MEM_P (op))
return FALSE;
ind = XEXP (op, 0);
if (reload_completed
&& (GET_CODE (ind) == LABEL_REF
|| (GET_CODE (ind) == CONST
&& GET_CODE (XEXP (ind, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (ind, 0), 0)) == LABEL_REF
&& CONST_INT_P (XEXP (XEXP (ind, 0), 1)))))
return TRUE;
if (REG_P (ind))
return arm_address_register_rtx_p (ind, 0);
if (GET_CODE (ind) == POST_INC
|| GET_CODE (ind) == PRE_DEC
|| (wb
&& (GET_CODE (ind) == PRE_INC
|| GET_CODE (ind) == POST_DEC)))
return arm_address_register_rtx_p (XEXP (ind, 0), 0);
if (wb
&& (GET_CODE (ind) == POST_MODIFY || GET_CODE (ind) == PRE_MODIFY)
&& arm_address_register_rtx_p (XEXP (ind, 0), 0)
&& GET_CODE (XEXP (ind, 1)) == PLUS
&& rtx_equal_p (XEXP (XEXP (ind, 1), 0), XEXP (ind, 0)))
ind = XEXP (ind, 1);
if (GET_CODE (ind) == PLUS
&& REG_P (XEXP (ind, 0))
&& REG_MODE_OK_FOR_BASE_P (XEXP (ind, 0), VOIDmode)
&& CONST_INT_P (XEXP (ind, 1))
&& INTVAL (XEXP (ind, 1)) > -1024
&& INTVAL (XEXP (ind, 1)) <  1024
&& (INTVAL (XEXP (ind, 1)) & 3) == 0)
return TRUE;
return FALSE;
}
int
neon_vector_mem_operand (rtx op, int type, bool strict)
{
rtx ind;
if (strict && ! (reload_in_progress || reload_completed)
&& (reg_mentioned_p (frame_pointer_rtx, op)
|| reg_mentioned_p (arg_pointer_rtx, op)
|| reg_mentioned_p (virtual_incoming_args_rtx, op)
|| reg_mentioned_p (virtual_outgoing_args_rtx, op)
|| reg_mentioned_p (virtual_stack_dynamic_rtx, op)
|| reg_mentioned_p (virtual_stack_vars_rtx, op)))
return FALSE;
if (!MEM_P (op))
return FALSE;
ind = XEXP (op, 0);
if (reload_completed
&& (GET_CODE (ind) == LABEL_REF
|| (GET_CODE (ind) == CONST
&& GET_CODE (XEXP (ind, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (ind, 0), 0)) == LABEL_REF
&& CONST_INT_P (XEXP (XEXP (ind, 0), 1)))))
return TRUE;
if (REG_P (ind))
return arm_address_register_rtx_p (ind, 0);
if ((type != 1 && GET_CODE (ind) == POST_INC)
|| (type == 0 && GET_CODE (ind) == PRE_DEC))
return arm_address_register_rtx_p (XEXP (ind, 0), 0);
if (type == 2 && GET_CODE (ind) == POST_MODIFY
&& GET_CODE (XEXP (ind, 1)) == PLUS
&& REG_P (XEXP (XEXP (ind, 1), 1)))
return true;
if (type == 0
&& GET_CODE (ind) == PLUS
&& REG_P (XEXP (ind, 0))
&& REG_MODE_OK_FOR_BASE_P (XEXP (ind, 0), VOIDmode)
&& CONST_INT_P (XEXP (ind, 1))
&& INTVAL (XEXP (ind, 1)) > -1024
&& (INTVAL (XEXP (ind, 1))
< (VALID_NEON_QREG_MODE (GET_MODE (op))? 1016 : 1024))
&& (INTVAL (XEXP (ind, 1)) & 3) == 0)
return TRUE;
return FALSE;
}
int
neon_struct_mem_operand (rtx op)
{
rtx ind;
if (! (reload_in_progress || reload_completed)
&& (   reg_mentioned_p (frame_pointer_rtx, op)
|| reg_mentioned_p (arg_pointer_rtx, op)
|| reg_mentioned_p (virtual_incoming_args_rtx, op)
|| reg_mentioned_p (virtual_outgoing_args_rtx, op)
|| reg_mentioned_p (virtual_stack_dynamic_rtx, op)
|| reg_mentioned_p (virtual_stack_vars_rtx, op)))
return FALSE;
if (!MEM_P (op))
return FALSE;
ind = XEXP (op, 0);
if (reload_completed
&& (GET_CODE (ind) == LABEL_REF
|| (GET_CODE (ind) == CONST
&& GET_CODE (XEXP (ind, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (ind, 0), 0)) == LABEL_REF
&& CONST_INT_P (XEXP (XEXP (ind, 0), 1)))))
return TRUE;
if (REG_P (ind))
return arm_address_register_rtx_p (ind, 0);
if (GET_CODE (ind) == POST_INC
|| GET_CODE (ind) == PRE_DEC)
return arm_address_register_rtx_p (XEXP (ind, 0), 0);
return FALSE;
}
int
arm_eliminable_register (rtx x)
{
return REG_P (x) && (REGNO (x) == FRAME_POINTER_REGNUM
|| REGNO (x) == ARG_POINTER_REGNUM
|| (REGNO (x) >= FIRST_VIRTUAL_REGISTER
&& REGNO (x) <= LAST_VIRTUAL_REGISTER));
}
enum reg_class
coproc_secondary_reload_class (machine_mode mode, rtx x, bool wb)
{
if (mode == HFmode)
{
if (!TARGET_NEON_FP16 && !TARGET_VFP_FP16INST)
return GENERAL_REGS;
if (s_register_operand (x, mode) || neon_vector_mem_operand (x, 2, true))
return NO_REGS;
return GENERAL_REGS;
}
if (TARGET_NEON
&& (MEM_P (x) || GET_CODE (x) == CONST_VECTOR)
&& (GET_MODE_CLASS (mode) == MODE_VECTOR_INT
|| GET_MODE_CLASS (mode) == MODE_VECTOR_FLOAT
|| VALID_NEON_STRUCT_MODE (mode)))
return NO_REGS;
if (arm_coproc_mem_operand (x, wb) || s_register_operand (x, mode))
return NO_REGS;
return GENERAL_REGS;
}
static bool
arm_return_in_msb (const_tree valtype)
{
return (TARGET_AAPCS_BASED
&& BYTES_BIG_ENDIAN
&& (AGGREGATE_TYPE_P (valtype)
|| TREE_CODE (valtype) == COMPLEX_TYPE
|| FIXED_POINT_TYPE_P (valtype)));
}
int
symbol_mentioned_p (rtx x)
{
const char * fmt;
int i;
if (GET_CODE (x) == SYMBOL_REF)
return 1;
if (GET_CODE (x) == UNSPEC && XINT (x, 1) == UNSPEC_TLS)
return 0;
fmt = GET_RTX_FORMAT (GET_CODE (x));
for (i = GET_RTX_LENGTH (GET_CODE (x)) - 1; i >= 0; i--)
{
if (fmt[i] == 'E')
{
int j;
for (j = XVECLEN (x, i) - 1; j >= 0; j--)
if (symbol_mentioned_p (XVECEXP (x, i, j)))
return 1;
}
else if (fmt[i] == 'e' && symbol_mentioned_p (XEXP (x, i)))
return 1;
}
return 0;
}
int
label_mentioned_p (rtx x)
{
const char * fmt;
int i;
if (GET_CODE (x) == LABEL_REF)
return 1;
if (GET_CODE (x) == UNSPEC && XINT (x, 1) == UNSPEC_TLS)
return 0;
fmt = GET_RTX_FORMAT (GET_CODE (x));
for (i = GET_RTX_LENGTH (GET_CODE (x)) - 1; i >= 0; i--)
{
if (fmt[i] == 'E')
{
int j;
for (j = XVECLEN (x, i) - 1; j >= 0; j--)
if (label_mentioned_p (XVECEXP (x, i, j)))
return 1;
}
else if (fmt[i] == 'e' && label_mentioned_p (XEXP (x, i)))
return 1;
}
return 0;
}
int
tls_mentioned_p (rtx x)
{
switch (GET_CODE (x))
{
case CONST:
return tls_mentioned_p (XEXP (x, 0));
case UNSPEC:
if (XINT (x, 1) == UNSPEC_TLS)
return 1;
default:
return 0;
}
}
static bool
arm_cannot_copy_insn_p (rtx_insn *insn)
{
if (recog_memoized (insn) == CODE_FOR_tlscall)
return true;
subrtx_iterator::array_type array;
FOR_EACH_SUBRTX (iter, array, PATTERN (insn), ALL)
{
const_rtx x = *iter;
if (GET_CODE (x) == UNSPEC
&& (XINT (x, 1) == UNSPEC_PIC_BASE
|| XINT (x, 1) == UNSPEC_PIC_UNIFIED))
return true;
}
rtx set = single_set (insn);
if (set)
{
rtx src = SET_SRC (set);
if (GET_CODE (src) == ZERO_EXTEND)
src = XEXP (src, 0);
if (GET_CODE (src) == UNSPEC_VOLATILE
&& (XINT (src, 1) == VUNSPEC_LL
|| XINT (src, 1) == VUNSPEC_LAX))
return true;
}
return false;
}
enum rtx_code
minmax_code (rtx x)
{
enum rtx_code code = GET_CODE (x);
switch (code)
{
case SMAX:
return GE;
case SMIN:
return LE;
case UMIN:
return LEU;
case UMAX:
return GEU;
default:
gcc_unreachable ();
}
}
bool
arm_sat_operator_match (rtx lo_bound, rtx hi_bound,
int *mask, bool *signed_sat)
{
int log = exact_log2 (INTVAL (hi_bound) + 1);
if (log == -1)
return false;
if (INTVAL (lo_bound) == 0)
{
if (mask)
*mask = log;
if (signed_sat)
*signed_sat = false;
return true;
}
if (INTVAL (lo_bound) == -INTVAL (hi_bound) - 1)
{
if (mask)
*mask = log + 1;
if (signed_sat)
*signed_sat = true;
return true;
}
return false;
}
int
adjacent_mem_locations (rtx a, rtx b)
{
if (volatile_refs_p (a) || volatile_refs_p (b))
return 0;
if ((REG_P (XEXP (a, 0))
|| (GET_CODE (XEXP (a, 0)) == PLUS
&& CONST_INT_P (XEXP (XEXP (a, 0), 1))))
&& (REG_P (XEXP (b, 0))
|| (GET_CODE (XEXP (b, 0)) == PLUS
&& CONST_INT_P (XEXP (XEXP (b, 0), 1)))))
{
HOST_WIDE_INT val0 = 0, val1 = 0;
rtx reg0, reg1;
int val_diff;
if (GET_CODE (XEXP (a, 0)) == PLUS)
{
reg0 = XEXP (XEXP (a, 0), 0);
val0 = INTVAL (XEXP (XEXP (a, 0), 1));
}
else
reg0 = XEXP (a, 0);
if (GET_CODE (XEXP (b, 0)) == PLUS)
{
reg1 = XEXP (XEXP (b, 0), 0);
val1 = INTVAL (XEXP (XEXP (b, 0), 1));
}
else
reg1 = XEXP (b, 0);
if (!const_ok_for_op (val0, PLUS) || !const_ok_for_op (val1, PLUS))
return 0;
if (arm_eliminable_register (reg0))
return 0;
val_diff = val1 - val0;
if (arm_ld_sched)
{
return (optimize_size && (REGNO (reg0) == REGNO (reg1))
&& (val0 == 0 || val1 == 0 || val0 == 4 || val1 == 4)
&& (val_diff == 4 || val_diff == -4));
}
return ((REGNO (reg0) == REGNO (reg1))
&& (val_diff == 4 || val_diff == -4));
}
return 0;
}
bool
ldm_stm_operation_p (rtx op, bool load, machine_mode mode,
bool consecutive, bool return_pc)
{
HOST_WIDE_INT count = XVECLEN (op, 0);
rtx reg, mem, addr;
unsigned regno;
unsigned first_regno;
HOST_WIDE_INT i = 1, base = 0, offset = 0;
rtx elt;
bool addr_reg_in_reglist = false;
bool update = false;
int reg_increment;
int offset_adj;
int regs_per_val;
gcc_assert ((mode == SImode) || consecutive);
gcc_assert (!return_pc || load);
reg_increment = GET_MODE_SIZE (mode);
regs_per_val = reg_increment / 4;
offset_adj = return_pc ? 1 : 0;
if (count <= 1
|| GET_CODE (XVECEXP (op, 0, offset_adj)) != SET
|| (load && !REG_P (SET_DEST (XVECEXP (op, 0, offset_adj)))))
return false;
elt = XVECEXP (op, 0, offset_adj);
if (GET_CODE (SET_SRC (elt)) == PLUS)
{
i++;
base = 1;
update = true;
if (!REG_P (SET_DEST (elt))
|| !REG_P (XEXP (SET_SRC (elt), 0))
|| (REGNO (SET_DEST (elt)) != REGNO (XEXP (SET_SRC (elt), 0)))
|| !CONST_INT_P (XEXP (SET_SRC (elt), 1))
|| INTVAL (XEXP (SET_SRC (elt), 1)) !=
((count - 1 - offset_adj) * reg_increment))
return false;
}
i = i + offset_adj;
base = base + offset_adj;
if ((count <= i) && (mode == SImode))
return false;
elt = XVECEXP (op, 0, i - 1);
if (GET_CODE (elt) != SET)
return false;
if (load)
{
reg = SET_DEST (elt);
mem = SET_SRC (elt);
}
else
{
reg = SET_SRC (elt);
mem = SET_DEST (elt);
}
if (!REG_P (reg) || !MEM_P (mem))
return false;
regno = REGNO (reg);
first_regno = regno;
addr = XEXP (mem, 0);
if (GET_CODE (addr) == PLUS)
{
if (!CONST_INT_P (XEXP (addr, 1)))
return false;
offset = INTVAL (XEXP (addr, 1));
addr = XEXP (addr, 0);
}
if (!REG_P (addr))
return false;
if (load && (REGNO (reg) == SP_REGNUM) && (REGNO (addr) != SP_REGNUM))
return false;
for (; i < count; i++)
{
elt = XVECEXP (op, 0, i);
if (GET_CODE (elt) != SET)
return false;
if (load)
{
reg = SET_DEST (elt);
mem = SET_SRC (elt);
}
else
{
reg = SET_SRC (elt);
mem = SET_DEST (elt);
}
if (!REG_P (reg)
|| GET_MODE (reg) != mode
|| REGNO (reg) <= regno
|| (consecutive
&& (REGNO (reg) !=
(unsigned int) (first_regno + regs_per_val * (i - base))))
|| (load && (REGNO (reg) == SP_REGNUM) && (REGNO (addr) != SP_REGNUM))
|| !MEM_P (mem)
|| GET_MODE (mem) != mode
|| ((GET_CODE (XEXP (mem, 0)) != PLUS
|| !rtx_equal_p (XEXP (XEXP (mem, 0), 0), addr)
|| !CONST_INT_P (XEXP (XEXP (mem, 0), 1))
|| (INTVAL (XEXP (XEXP (mem, 0), 1)) !=
offset + (i - base) * reg_increment))
&& (!REG_P (XEXP (mem, 0))
|| offset + (i - base) * reg_increment != 0)))
return false;
regno = REGNO (reg);
if (regno == REGNO (addr))
addr_reg_in_reglist = true;
}
if (load)
{
if (update && addr_reg_in_reglist)
return false;
if (TARGET_THUMB1)
return update || addr_reg_in_reglist;
}
return true;
}
static bool
multiple_operation_profitable_p (bool is_store ATTRIBUTE_UNUSED,
int nops, HOST_WIDE_INT add_offset)
{
if (nops == 2 && arm_ld_sched && add_offset != 0)
return false;
if (nops <= 2 && arm_tune_xscale && !optimize_size)
return false;
return true;
}
static bool
compute_offset_order (int nops, HOST_WIDE_INT *unsorted_offsets, int *order,
int *unsorted_regs)
{
int i;
for (i = 1; i < nops; i++)
{
int j;
order[i] = order[i - 1];
for (j = 0; j < nops; j++)
if (unsorted_offsets[j] == unsorted_offsets[order[i - 1]] + 4)
{
if (order[i] != order[i - 1])
return false;
order[i] = j;
}
if (order[i] == order[i - 1])
return false;
if (unsorted_regs != NULL
&& unsorted_regs[order[i]] <= unsorted_regs[order[i - 1]])
return false;
}
return true;
}
static int
load_multiple_sequence (rtx *operands, int nops, int *regs, int *saved_order,
int *base, HOST_WIDE_INT *load_offset, bool check_regs)
{
int unsorted_regs[MAX_LDM_STM_OPS];
HOST_WIDE_INT unsorted_offsets[MAX_LDM_STM_OPS];
int order[MAX_LDM_STM_OPS];
rtx base_reg_rtx = NULL;
int base_reg = -1;
int i, ldm_case;
gcc_assert (nops >= 2 && nops <= MAX_LDM_STM_OPS);
memset (order, 0, MAX_LDM_STM_OPS * sizeof (int));
for (i = 0; i < nops; i++)
{
rtx reg;
rtx offset;
if (GET_CODE (operands[nops + i]) == SUBREG)
operands[nops + i] = alter_subreg (operands + (nops + i), true);
gcc_assert (MEM_P (operands[nops + i]));
if (MEM_VOLATILE_P (operands[nops + i]))
return 0;
offset = const0_rtx;
if ((REG_P (reg = XEXP (operands[nops + i], 0))
|| (GET_CODE (reg) == SUBREG
&& REG_P (reg = SUBREG_REG (reg))))
|| (GET_CODE (XEXP (operands[nops + i], 0)) == PLUS
&& ((REG_P (reg = XEXP (XEXP (operands[nops + i], 0), 0)))
|| (GET_CODE (reg) == SUBREG
&& REG_P (reg = SUBREG_REG (reg))))
&& (CONST_INT_P (offset
= XEXP (XEXP (operands[nops + i], 0), 1)))))
{
if (i == 0)
{
base_reg = REGNO (reg);
base_reg_rtx = reg;
if (TARGET_THUMB1 && base_reg > LAST_LO_REGNUM)
return 0;
}
else if (base_reg != (int) REGNO (reg))
return 0;
unsorted_regs[i] = (REG_P (operands[i])
? REGNO (operands[i])
: REGNO (SUBREG_REG (operands[i])));
if (unsorted_regs[i] < 0
|| (TARGET_THUMB1 && unsorted_regs[i] > LAST_LO_REGNUM)
|| unsorted_regs[i] > 14
|| (i != nops - 1 && unsorted_regs[i] == base_reg))
return 0;
if (unsorted_regs[i] == SP_REGNUM && base_reg != SP_REGNUM)
return 0;
unsorted_offsets[i] = INTVAL (offset);
if (i == 0 || unsorted_offsets[i] < unsorted_offsets[order[0]])
order[0] = i;
}
else
return 0;
}
if (!compute_offset_order (nops, unsorted_offsets, order,
check_regs ? unsorted_regs : NULL))
return 0;
if (saved_order)
memcpy (saved_order, order, sizeof order);
if (base)
{
*base = base_reg;
for (i = 0; i < nops; i++)
regs[i] = unsorted_regs[check_regs ? order[i] : i];
*load_offset = unsorted_offsets[order[0]];
}
if (TARGET_THUMB1
&& !peep2_reg_dead_p (nops, base_reg_rtx))
return 0;
if (unsorted_offsets[order[0]] == 0)
ldm_case = 1; 
else if (TARGET_ARM && unsorted_offsets[order[0]] == 4)
ldm_case = 2; 
else if (TARGET_ARM && unsorted_offsets[order[nops - 1]] == 0)
ldm_case = 3; 
else if (TARGET_32BIT && unsorted_offsets[order[nops - 1]] == -4)
ldm_case = 4; 
else if (const_ok_for_arm (unsorted_offsets[order[0]])
|| const_ok_for_arm (-unsorted_offsets[order[0]]))
ldm_case = 5;
else
return 0;
if (!multiple_operation_profitable_p (false, nops,
ldm_case == 5
? unsorted_offsets[order[0]] : 0))
return 0;
return ldm_case;
}
static int
store_multiple_sequence (rtx *operands, int nops, int nops_total,
int *regs, rtx *reg_rtxs, int *saved_order, int *base,
HOST_WIDE_INT *load_offset, bool check_regs)
{
int unsorted_regs[MAX_LDM_STM_OPS];
rtx unsorted_reg_rtxs[MAX_LDM_STM_OPS];
HOST_WIDE_INT unsorted_offsets[MAX_LDM_STM_OPS];
int order[MAX_LDM_STM_OPS];
int base_reg = -1;
rtx base_reg_rtx = NULL;
int i, stm_case;
int base_writeback = TARGET_THUMB1;
gcc_assert (nops >= 2 && nops <= MAX_LDM_STM_OPS);
memset (order, 0, MAX_LDM_STM_OPS * sizeof (int));
for (i = 0; i < nops; i++)
{
rtx reg;
rtx offset;
if (GET_CODE (operands[nops + i]) == SUBREG)
operands[nops + i] = alter_subreg (operands + (nops + i), true);
gcc_assert (MEM_P (operands[nops + i]));
if (MEM_VOLATILE_P (operands[nops + i]))
return 0;
offset = const0_rtx;
if ((REG_P (reg = XEXP (operands[nops + i], 0))
|| (GET_CODE (reg) == SUBREG
&& REG_P (reg = SUBREG_REG (reg))))
|| (GET_CODE (XEXP (operands[nops + i], 0)) == PLUS
&& ((REG_P (reg = XEXP (XEXP (operands[nops + i], 0), 0)))
|| (GET_CODE (reg) == SUBREG
&& REG_P (reg = SUBREG_REG (reg))))
&& (CONST_INT_P (offset
= XEXP (XEXP (operands[nops + i], 0), 1)))))
{
unsorted_reg_rtxs[i] = (REG_P (operands[i])
? operands[i] : SUBREG_REG (operands[i]));
unsorted_regs[i] = REGNO (unsorted_reg_rtxs[i]);
if (i == 0)
{
base_reg = REGNO (reg);
base_reg_rtx = reg;
if (TARGET_THUMB1 && base_reg > LAST_LO_REGNUM)
return 0;
}
else if (base_reg != (int) REGNO (reg))
return 0;
if (unsorted_regs[i] < 0
|| (TARGET_THUMB1 && unsorted_regs[i] > LAST_LO_REGNUM)
|| (base_writeback && unsorted_regs[i] == base_reg)
|| (TARGET_THUMB2 && unsorted_regs[i] == SP_REGNUM)
|| unsorted_regs[i] > 14)
return 0;
unsorted_offsets[i] = INTVAL (offset);
if (i == 0 || unsorted_offsets[i] < unsorted_offsets[order[0]])
order[0] = i;
}
else
return 0;
}
if (!compute_offset_order (nops, unsorted_offsets, order,
check_regs ? unsorted_regs : NULL))
return 0;
if (saved_order)
memcpy (saved_order, order, sizeof order);
if (base)
{
*base = base_reg;
for (i = 0; i < nops; i++)
{
regs[i] = unsorted_regs[check_regs ? order[i] : i];
if (reg_rtxs)
reg_rtxs[i] = unsorted_reg_rtxs[check_regs ? order[i] : i];
}
*load_offset = unsorted_offsets[order[0]];
}
if (TARGET_THUMB1
&& !peep2_reg_dead_p (nops_total, base_reg_rtx))
return 0;
if (unsorted_offsets[order[0]] == 0)
stm_case = 1; 
else if (TARGET_ARM && unsorted_offsets[order[0]] == 4)
stm_case = 2; 
else if (TARGET_ARM && unsorted_offsets[order[nops - 1]] == 0)
stm_case = 3; 
else if (TARGET_32BIT && unsorted_offsets[order[nops - 1]] == -4)
stm_case = 4; 
else
return 0;
if (!multiple_operation_profitable_p (false, nops, 0))
return 0;
return stm_case;
}

static rtx
arm_gen_load_multiple_1 (int count, int *regs, rtx *mems, rtx basereg,
HOST_WIDE_INT wback_offset)
{
int i = 0, j;
rtx result;
if (!multiple_operation_profitable_p (false, count, 0))
{
rtx seq;
start_sequence ();
for (i = 0; i < count; i++)
emit_move_insn (gen_rtx_REG (SImode, regs[i]), mems[i]);
if (wback_offset != 0)
emit_move_insn (basereg, plus_constant (Pmode, basereg, wback_offset));
seq = get_insns ();
end_sequence ();
return seq;
}
result = gen_rtx_PARALLEL (VOIDmode,
rtvec_alloc (count + (wback_offset != 0 ? 1 : 0)));
if (wback_offset != 0)
{
XVECEXP (result, 0, 0)
= gen_rtx_SET (basereg, plus_constant (Pmode, basereg, wback_offset));
i = 1;
count++;
}
for (j = 0; i < count; i++, j++)
XVECEXP (result, 0, i)
= gen_rtx_SET (gen_rtx_REG (SImode, regs[j]), mems[j]);
return result;
}
static rtx
arm_gen_store_multiple_1 (int count, int *regs, rtx *mems, rtx basereg,
HOST_WIDE_INT wback_offset)
{
int i = 0, j;
rtx result;
if (GET_CODE (basereg) == PLUS)
basereg = XEXP (basereg, 0);
if (!multiple_operation_profitable_p (false, count, 0))
{
rtx seq;
start_sequence ();
for (i = 0; i < count; i++)
emit_move_insn (mems[i], gen_rtx_REG (SImode, regs[i]));
if (wback_offset != 0)
emit_move_insn (basereg, plus_constant (Pmode, basereg, wback_offset));
seq = get_insns ();
end_sequence ();
return seq;
}
result = gen_rtx_PARALLEL (VOIDmode,
rtvec_alloc (count + (wback_offset != 0 ? 1 : 0)));
if (wback_offset != 0)
{
XVECEXP (result, 0, 0)
= gen_rtx_SET (basereg, plus_constant (Pmode, basereg, wback_offset));
i = 1;
count++;
}
for (j = 0; i < count; i++, j++)
XVECEXP (result, 0, i)
= gen_rtx_SET (mems[j], gen_rtx_REG (SImode, regs[j]));
return result;
}
static rtx
arm_gen_multiple_op (bool is_load, int *regs, int count, rtx basereg,
bool write_back, rtx basemem, HOST_WIDE_INT *offsetp)
{
rtx mems[MAX_LDM_STM_OPS];
HOST_WIDE_INT offset = *offsetp;
int i;
gcc_assert (count <= MAX_LDM_STM_OPS);
if (GET_CODE (basereg) == PLUS)
basereg = XEXP (basereg, 0);
for (i = 0; i < count; i++)
{
rtx addr = plus_constant (Pmode, basereg, i * 4);
mems[i] = adjust_automodify_address_nv (basemem, SImode, addr, offset);
offset += 4;
}
if (write_back)
*offsetp = offset;
if (is_load)
return arm_gen_load_multiple_1 (count, regs, mems, basereg,
write_back ? 4 * count : 0);
else
return arm_gen_store_multiple_1 (count, regs, mems, basereg,
write_back ? 4 * count : 0);
}
rtx
arm_gen_load_multiple (int *regs, int count, rtx basereg, int write_back,
rtx basemem, HOST_WIDE_INT *offsetp)
{
return arm_gen_multiple_op (TRUE, regs, count, basereg, write_back, basemem,
offsetp);
}
rtx
arm_gen_store_multiple (int *regs, int count, rtx basereg, int write_back,
rtx basemem, HOST_WIDE_INT *offsetp)
{
return arm_gen_multiple_op (FALSE, regs, count, basereg, write_back, basemem,
offsetp);
}
bool
gen_ldm_seq (rtx *operands, int nops, bool sort_regs)
{
int regs[MAX_LDM_STM_OPS], mem_order[MAX_LDM_STM_OPS];
rtx mems[MAX_LDM_STM_OPS];
int i, j, base_reg;
rtx base_reg_rtx;
HOST_WIDE_INT offset;
int write_back = FALSE;
int ldm_case;
rtx addr;
ldm_case = load_multiple_sequence (operands, nops, regs, mem_order,
&base_reg, &offset, !sort_regs);
if (ldm_case == 0)
return false;
if (sort_regs)
for (i = 0; i < nops - 1; i++)
for (j = i + 1; j < nops; j++)
if (regs[i] > regs[j])
{
int t = regs[i];
regs[i] = regs[j];
regs[j] = t;
}
base_reg_rtx = gen_rtx_REG (Pmode, base_reg);
if (TARGET_THUMB1)
{
gcc_assert (peep2_reg_dead_p (nops, base_reg_rtx));
gcc_assert (ldm_case == 1 || ldm_case == 5);
write_back = TRUE;
}
if (ldm_case == 5)
{
rtx newbase = TARGET_THUMB1 ? base_reg_rtx : gen_rtx_REG (SImode, regs[0]);
emit_insn (gen_addsi3 (newbase, base_reg_rtx, GEN_INT (offset)));
offset = 0;
if (!TARGET_THUMB1)
base_reg_rtx = newbase;
}
for (i = 0; i < nops; i++)
{
addr = plus_constant (Pmode, base_reg_rtx, offset + i * 4);
mems[i] = adjust_automodify_address_nv (operands[nops + mem_order[i]],
SImode, addr, 0);
}
emit_insn (arm_gen_load_multiple_1 (nops, regs, mems, base_reg_rtx,
write_back ? offset + i * 4 : 0));
return true;
}
bool
gen_stm_seq (rtx *operands, int nops)
{
int i;
int regs[MAX_LDM_STM_OPS], mem_order[MAX_LDM_STM_OPS];
rtx mems[MAX_LDM_STM_OPS];
int base_reg;
rtx base_reg_rtx;
HOST_WIDE_INT offset;
int write_back = FALSE;
int stm_case;
rtx addr;
bool base_reg_dies;
stm_case = store_multiple_sequence (operands, nops, nops, regs, NULL,
mem_order, &base_reg, &offset, true);
if (stm_case == 0)
return false;
base_reg_rtx = gen_rtx_REG (Pmode, base_reg);
base_reg_dies = peep2_reg_dead_p (nops, base_reg_rtx);
if (TARGET_THUMB1)
{
gcc_assert (base_reg_dies);
write_back = TRUE;
}
if (stm_case == 5)
{
gcc_assert (base_reg_dies);
emit_insn (gen_addsi3 (base_reg_rtx, base_reg_rtx, GEN_INT (offset)));
offset = 0;
}
addr = plus_constant (Pmode, base_reg_rtx, offset);
for (i = 0; i < nops; i++)
{
addr = plus_constant (Pmode, base_reg_rtx, offset + i * 4);
mems[i] = adjust_automodify_address_nv (operands[nops + mem_order[i]],
SImode, addr, 0);
}
emit_insn (arm_gen_store_multiple_1 (nops, regs, mems, base_reg_rtx,
write_back ? offset + i * 4 : 0));
return true;
}
bool
gen_const_stm_seq (rtx *operands, int nops)
{
int regs[MAX_LDM_STM_OPS], sorted_regs[MAX_LDM_STM_OPS];
int reg_order[MAX_LDM_STM_OPS], mem_order[MAX_LDM_STM_OPS];
rtx reg_rtxs[MAX_LDM_STM_OPS], orig_reg_rtxs[MAX_LDM_STM_OPS];
rtx mems[MAX_LDM_STM_OPS];
int base_reg;
rtx base_reg_rtx;
HOST_WIDE_INT offset;
int write_back = FALSE;
int stm_case;
rtx addr;
bool base_reg_dies;
int i, j;
HARD_REG_SET allocated;
stm_case = store_multiple_sequence (operands, nops, 2 * nops, regs, reg_rtxs,
mem_order, &base_reg, &offset, false);
if (stm_case == 0)
return false;
memcpy (orig_reg_rtxs, reg_rtxs, sizeof orig_reg_rtxs);
CLEAR_HARD_REG_SET (allocated);
for (i = 0; i < nops; i++)
{
for (j = i + 1; j < nops; j++)
if (regs[i] == regs[j])
{
rtx t = peep2_find_free_register (0, nops * 2,
TARGET_THUMB1 ? "l" : "r",
SImode, &allocated);
if (t == NULL_RTX)
return false;
reg_rtxs[i] = t;
regs[i] = REGNO (t);
}
}
reg_order[0] = 0;
for (i = 0; i < nops; i++)
if (regs[i] < regs[reg_order[0]])
reg_order[0] = i;
for (i = 1; i < nops; i++)
{
int this_order = reg_order[i - 1];
for (j = 0; j < nops; j++)
if (regs[j] > regs[reg_order[i - 1]]
&& (this_order == reg_order[i - 1]
|| regs[j] < regs[this_order]))
this_order = j;
reg_order[i] = this_order;
}
for (i = 0; i < nops; i++)
{
int this_order = reg_order[i];
if ((this_order != mem_order[i]
|| orig_reg_rtxs[this_order] != reg_rtxs[this_order])
&& !peep2_reg_dead_p (nops * 2, orig_reg_rtxs[this_order]))
return false;
}
for (i = 0; i < nops; i++)
{
rtx op = operands[2 * nops + mem_order[i]];
sorted_regs[i] = regs[reg_order[i]];
emit_move_insn (reg_rtxs[reg_order[i]], op);
}
base_reg_rtx = gen_rtx_REG (Pmode, base_reg);
base_reg_dies = peep2_reg_dead_p (nops * 2, base_reg_rtx);
if (TARGET_THUMB1)
{
gcc_assert (base_reg_dies);
write_back = TRUE;
}
if (stm_case == 5)
{
gcc_assert (base_reg_dies);
emit_insn (gen_addsi3 (base_reg_rtx, base_reg_rtx, GEN_INT (offset)));
offset = 0;
}
addr = plus_constant (Pmode, base_reg_rtx, offset);
for (i = 0; i < nops; i++)
{
addr = plus_constant (Pmode, base_reg_rtx, offset + i * 4);
mems[i] = adjust_automodify_address_nv (operands[nops + mem_order[i]],
SImode, addr, 0);
}
emit_insn (arm_gen_store_multiple_1 (nops, sorted_regs, mems, base_reg_rtx,
write_back ? offset + i * 4 : 0));
return true;
}
static void
arm_block_move_unaligned_straight (rtx dstbase, rtx srcbase,
HOST_WIDE_INT length,
unsigned int interleave_factor)
{
rtx *regs = XALLOCAVEC (rtx, interleave_factor);
int *regnos = XALLOCAVEC (int, interleave_factor);
HOST_WIDE_INT block_size_bytes = interleave_factor * UNITS_PER_WORD;
HOST_WIDE_INT i, j;
HOST_WIDE_INT remaining = length, words;
rtx halfword_tmp = NULL, byte_tmp = NULL;
rtx dst, src;
bool src_aligned = MEM_ALIGN (srcbase) >= BITS_PER_WORD;
bool dst_aligned = MEM_ALIGN (dstbase) >= BITS_PER_WORD;
HOST_WIDE_INT srcoffset, dstoffset;
HOST_WIDE_INT src_autoinc, dst_autoinc;
rtx mem, addr;
gcc_assert (interleave_factor >= 1 && interleave_factor <= 4);
if (dst_aligned || src_aligned)
for (i = 0; i < interleave_factor; i++)
regs[i] = gen_rtx_REG (SImode, i);
else
for (i = 0; i < interleave_factor; i++)
regs[i] = gen_reg_rtx (SImode);
dst = copy_addr_to_reg (XEXP (dstbase, 0));
src = copy_addr_to_reg (XEXP (srcbase, 0));
srcoffset = dstoffset = 0;
src_autoinc = dst_autoinc = 0;
for (i = 0; i < interleave_factor; i++)
regnos[i] = i;
for (i = 0; i + block_size_bytes <= length; i += block_size_bytes)
{
if (src_aligned && interleave_factor > 1)
{
emit_insn (arm_gen_load_multiple (regnos, interleave_factor, src,
TRUE, srcbase, &srcoffset));
src_autoinc += UNITS_PER_WORD * interleave_factor;
}
else
{
for (j = 0; j < interleave_factor; j++)
{
addr = plus_constant (Pmode, src, (srcoffset + j * UNITS_PER_WORD
- src_autoinc));
mem = adjust_automodify_address (srcbase, SImode, addr,
srcoffset + j * UNITS_PER_WORD);
emit_insn (gen_unaligned_loadsi (regs[j], mem));
}
srcoffset += block_size_bytes;
}
if (dst_aligned && interleave_factor > 1)
{
emit_insn (arm_gen_store_multiple (regnos, interleave_factor, dst,
TRUE, dstbase, &dstoffset));
dst_autoinc += UNITS_PER_WORD * interleave_factor;
}
else
{
for (j = 0; j < interleave_factor; j++)
{
addr = plus_constant (Pmode, dst, (dstoffset + j * UNITS_PER_WORD
- dst_autoinc));
mem = adjust_automodify_address (dstbase, SImode, addr,
dstoffset + j * UNITS_PER_WORD);
emit_insn (gen_unaligned_storesi (mem, regs[j]));
}
dstoffset += block_size_bytes;
}
remaining -= block_size_bytes;
}
words = remaining / UNITS_PER_WORD;
gcc_assert (words < interleave_factor);
if (src_aligned && words > 1)
{
emit_insn (arm_gen_load_multiple (regnos, words, src, TRUE, srcbase,
&srcoffset));
src_autoinc += UNITS_PER_WORD * words;
}
else
{
for (j = 0; j < words; j++)
{
addr = plus_constant (Pmode, src,
srcoffset + j * UNITS_PER_WORD - src_autoinc);
mem = adjust_automodify_address (srcbase, SImode, addr,
srcoffset + j * UNITS_PER_WORD);
if (src_aligned)
emit_move_insn (regs[j], mem);
else
emit_insn (gen_unaligned_loadsi (regs[j], mem));
}
srcoffset += words * UNITS_PER_WORD;
}
if (dst_aligned && words > 1)
{
emit_insn (arm_gen_store_multiple (regnos, words, dst, TRUE, dstbase,
&dstoffset));
dst_autoinc += words * UNITS_PER_WORD;
}
else
{
for (j = 0; j < words; j++)
{
addr = plus_constant (Pmode, dst,
dstoffset + j * UNITS_PER_WORD - dst_autoinc);
mem = adjust_automodify_address (dstbase, SImode, addr,
dstoffset + j * UNITS_PER_WORD);
if (dst_aligned)
emit_move_insn (mem, regs[j]);
else
emit_insn (gen_unaligned_storesi (mem, regs[j]));
}
dstoffset += words * UNITS_PER_WORD;
}
remaining -= words * UNITS_PER_WORD;
gcc_assert (remaining < 4);
if (remaining >= 2)
{
halfword_tmp = gen_reg_rtx (SImode);
addr = plus_constant (Pmode, src, srcoffset - src_autoinc);
mem = adjust_automodify_address (srcbase, HImode, addr, srcoffset);
emit_insn (gen_unaligned_loadhiu (halfword_tmp, mem));
if (interleave_factor == 1)
{
addr = plus_constant (Pmode, dst, dstoffset - dst_autoinc);
mem = adjust_automodify_address (dstbase, HImode, addr, dstoffset);
emit_insn (gen_unaligned_storehi (mem,
gen_lowpart (HImode, halfword_tmp)));
halfword_tmp = NULL;
dstoffset += 2;
}
remaining -= 2;
srcoffset += 2;
}
gcc_assert (remaining < 2);
if ((remaining & 1) != 0)
{
byte_tmp = gen_reg_rtx (SImode);
addr = plus_constant (Pmode, src, srcoffset - src_autoinc);
mem = adjust_automodify_address (srcbase, QImode, addr, srcoffset);
emit_move_insn (gen_lowpart (QImode, byte_tmp), mem);
if (interleave_factor == 1)
{
addr = plus_constant (Pmode, dst, dstoffset - dst_autoinc);
mem = adjust_automodify_address (dstbase, QImode, addr, dstoffset);
emit_move_insn (mem, gen_lowpart (QImode, byte_tmp));
byte_tmp = NULL;
dstoffset++;
}
remaining--;
srcoffset++;
}
if (halfword_tmp)
{
addr = plus_constant (Pmode, dst, dstoffset - dst_autoinc);
mem = adjust_automodify_address (dstbase, HImode, addr, dstoffset);
emit_insn (gen_unaligned_storehi (mem,
gen_lowpart (HImode, halfword_tmp)));
dstoffset += 2;
}
if (byte_tmp)
{
addr = plus_constant (Pmode, dst, dstoffset - dst_autoinc);
mem = adjust_automodify_address (dstbase, QImode, addr, dstoffset);
emit_move_insn (mem, gen_lowpart (QImode, byte_tmp));
dstoffset++;
}
gcc_assert (remaining == 0 && srcoffset == dstoffset);
}
static void
arm_adjust_block_mem (rtx mem, HOST_WIDE_INT length, rtx *loop_reg,
rtx *loop_mem)
{
*loop_reg = copy_addr_to_reg (XEXP (mem, 0));
*loop_mem = change_address (mem, BLKmode, *loop_reg);
set_mem_align (*loop_mem, MIN (MEM_ALIGN (mem), length * BITS_PER_UNIT));
}
static void
arm_block_move_unaligned_loop (rtx dest, rtx src, HOST_WIDE_INT length,
unsigned int interleave_factor,
HOST_WIDE_INT bytes_per_iter)
{
rtx src_reg, dest_reg, final_src, test;
HOST_WIDE_INT leftover;
leftover = length % bytes_per_iter;
length -= leftover;
arm_adjust_block_mem (src, bytes_per_iter, &src_reg, &src);
arm_adjust_block_mem (dest, bytes_per_iter, &dest_reg, &dest);
final_src = expand_simple_binop (Pmode, PLUS, src_reg, GEN_INT (length),
0, 0, OPTAB_WIDEN);
rtx_code_label *label = gen_label_rtx ();
emit_label (label);
arm_block_move_unaligned_straight (dest, src, bytes_per_iter,
interleave_factor);
emit_move_insn (src_reg, plus_constant (Pmode, src_reg, bytes_per_iter));
emit_move_insn (dest_reg, plus_constant (Pmode, dest_reg, bytes_per_iter));
test = gen_rtx_NE (VOIDmode, src_reg, final_src);
emit_jump_insn (gen_cbranchsi4 (test, src_reg, final_src, label));
if (leftover)
arm_block_move_unaligned_straight (dest, src, leftover, interleave_factor);
}
static int
arm_movmemqi_unaligned (rtx *operands)
{
HOST_WIDE_INT length = INTVAL (operands[2]);
if (optimize_size)
{
bool src_aligned = MEM_ALIGN (operands[1]) >= BITS_PER_WORD;
bool dst_aligned = MEM_ALIGN (operands[0]) >= BITS_PER_WORD;
unsigned int interleave_factor = (src_aligned || dst_aligned) ? 2 : 1;
HOST_WIDE_INT bytes_per_iter = (src_aligned || dst_aligned) ? 8 : 4;
if (length > 12)
arm_block_move_unaligned_loop (operands[0], operands[1], length,
interleave_factor, bytes_per_iter);
else
arm_block_move_unaligned_straight (operands[0], operands[1], length,
interleave_factor);
}
else
{
if (length > 32)
arm_block_move_unaligned_loop (operands[0], operands[1], length, 4, 16);
else
arm_block_move_unaligned_straight (operands[0], operands[1], length, 4);
}
return 1;
}
int
arm_gen_movmemqi (rtx *operands)
{
HOST_WIDE_INT in_words_to_go, out_words_to_go, last_bytes;
HOST_WIDE_INT srcoffset, dstoffset;
rtx src, dst, srcbase, dstbase;
rtx part_bytes_reg = NULL;
rtx mem;
if (!CONST_INT_P (operands[2])
|| !CONST_INT_P (operands[3])
|| INTVAL (operands[2]) > 64)
return 0;
if (unaligned_access && (INTVAL (operands[3]) & 3) != 0)
return arm_movmemqi_unaligned (operands);
if (INTVAL (operands[3]) & 3)
return 0;
dstbase = operands[0];
srcbase = operands[1];
dst = copy_to_mode_reg (SImode, XEXP (dstbase, 0));
src = copy_to_mode_reg (SImode, XEXP (srcbase, 0));
in_words_to_go = ARM_NUM_INTS (INTVAL (operands[2]));
out_words_to_go = INTVAL (operands[2]) / 4;
last_bytes = INTVAL (operands[2]) & 3;
dstoffset = srcoffset = 0;
if (out_words_to_go != in_words_to_go && ((in_words_to_go - 1) & 3) != 0)
part_bytes_reg = gen_rtx_REG (SImode, (in_words_to_go - 1) & 3);
while (in_words_to_go >= 2)
{
if (in_words_to_go > 4)
emit_insn (arm_gen_load_multiple (arm_regs_in_sequence, 4, src,
TRUE, srcbase, &srcoffset));
else
emit_insn (arm_gen_load_multiple (arm_regs_in_sequence, in_words_to_go,
src, FALSE, srcbase,
&srcoffset));
if (out_words_to_go)
{
if (out_words_to_go > 4)
emit_insn (arm_gen_store_multiple (arm_regs_in_sequence, 4, dst,
TRUE, dstbase, &dstoffset));
else if (out_words_to_go != 1)
emit_insn (arm_gen_store_multiple (arm_regs_in_sequence,
out_words_to_go, dst,
(last_bytes == 0
? FALSE : TRUE),
dstbase, &dstoffset));
else
{
mem = adjust_automodify_address (dstbase, SImode, dst, dstoffset);
emit_move_insn (mem, gen_rtx_REG (SImode, R0_REGNUM));
if (last_bytes != 0)
{
emit_insn (gen_addsi3 (dst, dst, GEN_INT (4)));
dstoffset += 4;
}
}
}
in_words_to_go -= in_words_to_go < 4 ? in_words_to_go : 4;
out_words_to_go -= out_words_to_go < 4 ? out_words_to_go : 4;
}
if (out_words_to_go)
{
rtx sreg;
mem = adjust_automodify_address (srcbase, SImode, src, srcoffset);
sreg = copy_to_reg (mem);
mem = adjust_automodify_address (dstbase, SImode, dst, dstoffset);
emit_move_insn (mem, sreg);
in_words_to_go--;
gcc_assert (!in_words_to_go);	
}
if (in_words_to_go)
{
gcc_assert (in_words_to_go > 0);
mem = adjust_automodify_address (srcbase, SImode, src, srcoffset);
part_bytes_reg = copy_to_mode_reg (SImode, mem);
}
gcc_assert (!last_bytes || part_bytes_reg);
if (BYTES_BIG_ENDIAN && last_bytes)
{
rtx tmp = gen_reg_rtx (SImode);
emit_insn (gen_lshrsi3 (tmp, part_bytes_reg,
GEN_INT (8 * (4 - last_bytes))));
part_bytes_reg = tmp;
while (last_bytes)
{
mem = adjust_automodify_address (dstbase, QImode,
plus_constant (Pmode, dst,
last_bytes - 1),
dstoffset + last_bytes - 1);
emit_move_insn (mem, gen_lowpart (QImode, part_bytes_reg));
if (--last_bytes)
{
tmp = gen_reg_rtx (SImode);
emit_insn (gen_lshrsi3 (tmp, part_bytes_reg, GEN_INT (8)));
part_bytes_reg = tmp;
}
}
}
else
{
if (last_bytes > 1)
{
mem = adjust_automodify_address (dstbase, HImode, dst, dstoffset);
emit_move_insn (mem, gen_lowpart (HImode, part_bytes_reg));
last_bytes -= 2;
if (last_bytes)
{
rtx tmp = gen_reg_rtx (SImode);
emit_insn (gen_addsi3 (dst, dst, const2_rtx));
emit_insn (gen_lshrsi3 (tmp, part_bytes_reg, GEN_INT (16)));
part_bytes_reg = tmp;
dstoffset += 2;
}
}
if (last_bytes)
{
mem = adjust_automodify_address (dstbase, QImode, dst, dstoffset);
emit_move_insn (mem, gen_lowpart (QImode, part_bytes_reg));
}
}
return 1;
}
inline static rtx
next_consecutive_mem (rtx mem)
{
machine_mode mode = GET_MODE (mem);
HOST_WIDE_INT offset = GET_MODE_SIZE (mode);
rtx addr = plus_constant (Pmode, XEXP (mem, 0), offset);
return adjust_automodify_address (mem, mode, addr, offset);
}
bool
gen_movmem_ldrd_strd (rtx *operands)
{
unsigned HOST_WIDE_INT len;
HOST_WIDE_INT align;
rtx src, dst, base;
rtx reg0;
bool src_aligned, dst_aligned;
bool src_volatile, dst_volatile;
gcc_assert (CONST_INT_P (operands[2]));
gcc_assert (CONST_INT_P (operands[3]));
len = UINTVAL (operands[2]);
if (len > 64)
return false;
align = INTVAL (operands[3]);
if ((!unaligned_access) && (len >= 4) && ((align & 3) != 0))
return false;
dst = operands[0];
dst_volatile = MEM_VOLATILE_P (dst);
dst_aligned = MEM_ALIGN (dst) >= BITS_PER_WORD;
base = copy_to_mode_reg (SImode, XEXP (dst, 0));
dst = adjust_automodify_address (dst, VOIDmode, base, 0);
src = operands[1];
src_volatile = MEM_VOLATILE_P (src);
src_aligned = MEM_ALIGN (src) >= BITS_PER_WORD;
base = copy_to_mode_reg (SImode, XEXP (src, 0));
src = adjust_automodify_address (src, VOIDmode, base, 0);
if (!unaligned_access && !(src_aligned && dst_aligned))
return false;
if (src_volatile || dst_volatile)
return false;
if (!(dst_aligned || src_aligned))
return arm_gen_movmemqi (operands);
src = adjust_address (src, src_aligned ? DImode : SImode, 0);
dst = adjust_address (dst, dst_aligned ? DImode : SImode, 0);
while (len >= 8)
{
len -= 8;
reg0 = gen_reg_rtx (DImode);
rtx low_reg = NULL_RTX;
rtx hi_reg = NULL_RTX;
if (!src_aligned || !dst_aligned)
{
low_reg = gen_lowpart (SImode, reg0);
hi_reg = gen_highpart_mode (SImode, DImode, reg0);
}
if (src_aligned)
emit_move_insn (reg0, src);
else
{
emit_insn (gen_unaligned_loadsi (low_reg, src));
src = next_consecutive_mem (src);
emit_insn (gen_unaligned_loadsi (hi_reg, src));
}
if (dst_aligned)
emit_move_insn (dst, reg0);
else
{
emit_insn (gen_unaligned_storesi (dst, low_reg));
dst = next_consecutive_mem (dst);
emit_insn (gen_unaligned_storesi (dst, hi_reg));
}
src = next_consecutive_mem (src);
dst = next_consecutive_mem (dst);
}
gcc_assert (len < 8);
if (len >= 4)
{
reg0 = gen_reg_rtx (SImode);
src = adjust_address (src, SImode, 0);
dst = adjust_address (dst, SImode, 0);
if (src_aligned)
emit_move_insn (reg0, src);
else
emit_insn (gen_unaligned_loadsi (reg0, src));
if (dst_aligned)
emit_move_insn (dst, reg0);
else
emit_insn (gen_unaligned_storesi (dst, reg0));
src = next_consecutive_mem (src);
dst = next_consecutive_mem (dst);
len -= 4;
}
if (len == 0)
return true;
if (len >= 2)
{
dst = adjust_address (dst, HImode, 0);
src = adjust_address (src, HImode, 0);
reg0 = gen_reg_rtx (SImode);
if (src_aligned)
emit_insn (gen_zero_extendhisi2 (reg0, src));
else
emit_insn (gen_unaligned_loadhiu (reg0, src));
if (dst_aligned)
emit_insn (gen_movhi (dst, gen_lowpart(HImode, reg0)));
else
emit_insn (gen_unaligned_storehi (dst, gen_lowpart (HImode, reg0)));
src = next_consecutive_mem (src);
dst = next_consecutive_mem (dst);
if (len == 2)
return true;
}
dst = adjust_address (dst, QImode, 0);
src = adjust_address (src, QImode, 0);
reg0 = gen_reg_rtx (QImode);
emit_move_insn (reg0, src);
emit_move_insn (dst, reg0);
return true;
}
machine_mode
arm_select_dominance_cc_mode (rtx x, rtx y, HOST_WIDE_INT cond_or)
{
enum rtx_code cond1, cond2;
int swapped = 0;
if ((arm_select_cc_mode (cond1 = GET_CODE (x), XEXP (x, 0), XEXP (x, 1))
!= CCmode)
|| (arm_select_cc_mode (cond2 = GET_CODE (y), XEXP (y, 0), XEXP (y, 1))
!= CCmode))
return CCmode;
if (cond_or == DOM_CC_NX_OR_Y)
cond1 = reverse_condition (cond1);
if (cond1 != cond2
&& !comparison_dominates_p (cond1, cond2)
&& (swapped = 1, !comparison_dominates_p (cond2, cond1)))
return CCmode;
if (swapped)
std::swap (cond1, cond2);
switch (cond1)
{
case EQ:
if (cond_or == DOM_CC_X_AND_Y)
return CC_DEQmode;
switch (cond2)
{
case EQ: return CC_DEQmode;
case LE: return CC_DLEmode;
case LEU: return CC_DLEUmode;
case GE: return CC_DGEmode;
case GEU: return CC_DGEUmode;
default: gcc_unreachable ();
}
case LT:
if (cond_or == DOM_CC_X_AND_Y)
return CC_DLTmode;
switch (cond2)
{
case  LT:
return CC_DLTmode;
case LE:
return CC_DLEmode;
case NE:
return CC_DNEmode;
default:
gcc_unreachable ();
}
case GT:
if (cond_or == DOM_CC_X_AND_Y)
return CC_DGTmode;
switch (cond2)
{
case GT:
return CC_DGTmode;
case GE:
return CC_DGEmode;
case NE:
return CC_DNEmode;
default:
gcc_unreachable ();
}
case LTU:
if (cond_or == DOM_CC_X_AND_Y)
return CC_DLTUmode;
switch (cond2)
{
case LTU:
return CC_DLTUmode;
case LEU:
return CC_DLEUmode;
case NE:
return CC_DNEmode;
default:
gcc_unreachable ();
}
case GTU:
if (cond_or == DOM_CC_X_AND_Y)
return CC_DGTUmode;
switch (cond2)
{
case GTU:
return CC_DGTUmode;
case GEU:
return CC_DGEUmode;
case NE:
return CC_DNEmode;
default:
gcc_unreachable ();
}
case NE:
gcc_assert (cond1 == cond2);
return CC_DNEmode;
case LE:
gcc_assert (cond1 == cond2);
return CC_DLEmode;
case GE:
gcc_assert (cond1 == cond2);
return CC_DGEmode;
case LEU:
gcc_assert (cond1 == cond2);
return CC_DLEUmode;
case GEU:
gcc_assert (cond1 == cond2);
return CC_DGEUmode;
default:
gcc_unreachable ();
}
}
machine_mode
arm_select_cc_mode (enum rtx_code op, rtx x, rtx y)
{
if (GET_MODE_CLASS (GET_MODE (x)) == MODE_FLOAT)
{
switch (op)
{
case EQ:
case NE:
case UNORDERED:
case ORDERED:
case UNLT:
case UNLE:
case UNGT:
case UNGE:
case UNEQ:
case LTGT:
return CCFPmode;
case LT:
case LE:
case GT:
case GE:
return CCFPEmode;
default:
gcc_unreachable ();
}
}
if (GET_MODE (y) == SImode
&& (REG_P (y) || (GET_CODE (y) == SUBREG))
&& (GET_CODE (x) == ASHIFT || GET_CODE (x) == ASHIFTRT
|| GET_CODE (x) == LSHIFTRT || GET_CODE (x) == ROTATE
|| GET_CODE (x) == ROTATERT))
return CC_SWPmode;
if (GET_MODE (y) == SImode
&& (REG_P (y) || (GET_CODE (y) == SUBREG))
&& GET_CODE (x) == NEG
&& (op ==	EQ || op == NE))
return CC_Zmode;
if (GET_MODE (x) == SImode
&& GET_CODE (x) == ASHIFT
&& CONST_INT_P (XEXP (x, 1)) && INTVAL (XEXP (x, 1)) == 24
&& GET_CODE (XEXP (x, 0)) == SUBREG
&& MEM_P (SUBREG_REG (XEXP (x, 0)))
&& GET_MODE (SUBREG_REG (XEXP (x, 0))) == QImode
&& (op == EQ || op == NE
|| op == GEU || op == GTU || op == LTU || op == LEU)
&& CONST_INT_P (y))
return CC_Zmode;
if (GET_CODE (x) == IF_THEN_ELSE
&& (XEXP (x, 2) == const0_rtx
|| XEXP (x, 2) == const1_rtx)
&& COMPARISON_P (XEXP (x, 0))
&& COMPARISON_P (XEXP (x, 1)))
return arm_select_dominance_cc_mode (XEXP (x, 0), XEXP (x, 1),
INTVAL (XEXP (x, 2)));
if (GET_CODE (x) == AND
&& (op == EQ || op == NE)
&& COMPARISON_P (XEXP (x, 0))
&& COMPARISON_P (XEXP (x, 1)))
return arm_select_dominance_cc_mode (XEXP (x, 0), XEXP (x, 1),
DOM_CC_X_AND_Y);
if (GET_CODE (x) == IOR
&& (op == EQ || op == NE)
&& COMPARISON_P (XEXP (x, 0))
&& COMPARISON_P (XEXP (x, 1)))
return arm_select_dominance_cc_mode (XEXP (x, 0), XEXP (x, 1),
DOM_CC_X_OR_Y);
if (TARGET_THUMB1
&& GET_MODE (x) == SImode
&& (op == EQ || op == NE)
&& GET_CODE (x) == ZERO_EXTRACT
&& XEXP (x, 1) == const1_rtx)
return CC_Nmode;
if (GET_MODE (x) == SImode
&& y == const0_rtx
&& (op == EQ || op == NE || op == LT || op == GE)
&& (GET_CODE (x) == PLUS || GET_CODE (x) == MINUS
|| GET_CODE (x) == AND || GET_CODE (x) == IOR
|| GET_CODE (x) == XOR || GET_CODE (x) == MULT
|| GET_CODE (x) == NOT || GET_CODE (x) == NEG
|| GET_CODE (x) == LSHIFTRT
|| GET_CODE (x) == ASHIFT || GET_CODE (x) == ASHIFTRT
|| GET_CODE (x) == ROTATERT
|| (TARGET_32BIT && GET_CODE (x) == ZERO_EXTRACT)))
return CC_NOOVmode;
if (GET_MODE (x) == QImode && (op == EQ || op == NE))
return CC_Zmode;
if (GET_MODE (x) == SImode && (op == LTU || op == GEU)
&& GET_CODE (x) == PLUS
&& (rtx_equal_p (XEXP (x, 0), y) || rtx_equal_p (XEXP (x, 1), y)))
return CC_Cmode;
if (GET_MODE (x) == DImode || GET_MODE (y) == DImode)
{
switch (op)
{
case EQ:
case NE:
if (y == const0_rtx)
return CC_Zmode;
if (!TARGET_32BIT)
return CC_Zmode;
case LTU:
case LEU:
case GTU:
case GEU:
if (TARGET_32BIT)
return CC_CZmode;
case LT:
case LE:
case GT:
case GE:
gcc_assert (op != EQ && op != NE);
return CC_NCVmode;
default:
gcc_unreachable ();
}
}
if (GET_MODE_CLASS (GET_MODE (x)) == MODE_CC)
return GET_MODE (x);
return CCmode;
}
rtx
arm_gen_compare_reg (enum rtx_code code, rtx x, rtx y, rtx scratch)
{
machine_mode mode;
rtx cc_reg;
int dimode_comparison = GET_MODE (x) == DImode || GET_MODE (y) == DImode;
if (dimode_comparison && !REG_P (x))
x = force_reg (DImode, x);
mode = SELECT_CC_MODE (code, x, y);
cc_reg = gen_rtx_REG (mode, CC_REGNUM);
if (dimode_comparison
&& mode != CC_CZmode)
{
rtx clobber, set;
if (mode == CC_Zmode && y != const0_rtx)
{
gcc_assert (!reload_completed);
x = expand_binop (DImode, xor_optab, x, y, NULL_RTX, 0, OPTAB_WIDEN);
y = const0_rtx;
}
if (reload_completed)
gcc_assert (scratch != NULL && GET_MODE (scratch) == SImode);
else
scratch = gen_rtx_SCRATCH (SImode);
clobber = gen_rtx_CLOBBER (VOIDmode, scratch);
set = gen_rtx_SET (cc_reg, gen_rtx_COMPARE (mode, x, y));
emit_insn (gen_rtx_PARALLEL (VOIDmode, gen_rtvec (2, set, clobber)));
}
else
emit_set_insn (cc_reg, gen_rtx_COMPARE (mode, x, y));
return cc_reg;
}
rtx
arm_gen_return_addr_mask (void)
{
rtx reg = gen_reg_rtx (Pmode);
emit_insn (gen_return_addr_mask (reg));
return reg;
}
void
arm_reload_in_hi (rtx *operands)
{
rtx ref = operands[1];
rtx base, scratch;
HOST_WIDE_INT offset = 0;
if (GET_CODE (ref) == SUBREG)
{
offset = SUBREG_BYTE (ref);
ref = SUBREG_REG (ref);
}
if (REG_P (ref))
{
if (reg_equiv_mem (REGNO (ref)))
{
ref = reg_equiv_mem (REGNO (ref));
base = find_replacement (&XEXP (ref, 0));
}
else
base = reg_equiv_address (REGNO (ref));
if (base == NULL)
{
gcc_assert (REG_P (operands[0]));
emit_insn (gen_movsi (gen_rtx_SUBREG (SImode, operands[0], 0),
gen_rtx_SUBREG (SImode, ref, 0)));
return;
}
}
else
base = find_replacement (&XEXP (ref, 0));
if (GET_CODE (base) == MINUS
|| (GET_CODE (base) == PLUS && !CONST_INT_P (XEXP (base, 1))))
{
rtx base_plus = gen_rtx_REG (SImode, REGNO (operands[2]) + 1);
emit_set_insn (base_plus, base);
base = base_plus;
}
else if (GET_CODE (base) == PLUS)
{
HOST_WIDE_INT hi, lo;
offset += INTVAL (XEXP (base, 1));
base = XEXP (base, 0);
lo = (offset >= 0
? (offset & 0xfff)
: -((-offset) & 0xfff));
if (lo == 4095)
lo &= 0x7ff;
hi = ((((offset - lo) & (HOST_WIDE_INT) 0xffffffff)
^ (HOST_WIDE_INT) 0x80000000)
- (HOST_WIDE_INT) 0x80000000);
gcc_assert (hi + lo == offset);
if (hi != 0)
{
rtx base_plus = gen_rtx_REG (SImode, REGNO (operands[2]) + 1);
emit_insn (gen_addsi3 (base_plus, base, GEN_INT (hi)));
base = base_plus;
offset = lo;
}
}
if (REGNO (operands[2]) == REGNO (operands[0]))
scratch = gen_rtx_REG (SImode, REGNO (operands[2]) + 1);
else
scratch = gen_rtx_REG (SImode, REGNO (operands[2]));
emit_insn (gen_zero_extendqisi2 (scratch,
gen_rtx_MEM (QImode,
plus_constant (Pmode, base,
offset))));
emit_insn (gen_zero_extendqisi2 (gen_rtx_SUBREG (SImode, operands[0], 0),
gen_rtx_MEM (QImode,
plus_constant (Pmode, base,
offset + 1))));
if (!BYTES_BIG_ENDIAN)
emit_set_insn (gen_rtx_SUBREG (SImode, operands[0], 0),
gen_rtx_IOR (SImode,
gen_rtx_ASHIFT
(SImode,
gen_rtx_SUBREG (SImode, operands[0], 0),
GEN_INT (8)),
scratch));
else
emit_set_insn (gen_rtx_SUBREG (SImode, operands[0], 0),
gen_rtx_IOR (SImode,
gen_rtx_ASHIFT (SImode, scratch,
GEN_INT (8)),
gen_rtx_SUBREG (SImode, operands[0], 0)));
}
void
arm_reload_out_hi (rtx *operands)
{
rtx ref = operands[0];
rtx outval = operands[1];
rtx base, scratch;
HOST_WIDE_INT offset = 0;
if (GET_CODE (ref) == SUBREG)
{
offset = SUBREG_BYTE (ref);
ref = SUBREG_REG (ref);
}
if (REG_P (ref))
{
if (reg_equiv_mem (REGNO (ref)))
{
ref = reg_equiv_mem (REGNO (ref));
base = find_replacement (&XEXP (ref, 0));
}
else
base = reg_equiv_address (REGNO (ref));
if (base == NULL)
{
gcc_assert (REG_P (outval) || SUBREG_P (outval));
if (REG_P (outval))
{
emit_insn (gen_movsi (gen_rtx_SUBREG (SImode, ref, 0),
gen_rtx_SUBREG (SImode, outval, 0)));
}
else 
{
if (GET_MODE (SUBREG_REG (outval)) == SImode)
emit_insn (gen_movsi (gen_rtx_SUBREG (SImode, ref, 0),
SUBREG_REG (outval)));
else
gcc_unreachable ();
}
return;
}
}
else
base = find_replacement (&XEXP (ref, 0));
scratch = gen_rtx_REG (SImode, REGNO (operands[2]));
if (GET_CODE (base) == MINUS
|| (GET_CODE (base) == PLUS && !CONST_INT_P (XEXP (base, 1))))
{
rtx base_plus = gen_rtx_REG (SImode, REGNO (operands[2]) + 1);
if (reg_overlap_mentioned_p (base_plus, outval))
{
if (!reg_overlap_mentioned_p (scratch, outval))
std::swap (scratch, base_plus);
else
{
rtx scratch_hi = gen_rtx_REG (HImode, REGNO (operands[2]));
emit_insn (gen_movhi (scratch_hi, outval));
outval = scratch_hi;
}
}
emit_set_insn (base_plus, base);
base = base_plus;
}
else if (GET_CODE (base) == PLUS)
{
HOST_WIDE_INT hi, lo;
offset += INTVAL (XEXP (base, 1));
base = XEXP (base, 0);
lo = (offset >= 0
? (offset & 0xfff)
: -((-offset) & 0xfff));
if (lo == 4095)
lo &= 0x7ff;
hi = ((((offset - lo) & (HOST_WIDE_INT) 0xffffffff)
^ (HOST_WIDE_INT) 0x80000000)
- (HOST_WIDE_INT) 0x80000000);
gcc_assert (hi + lo == offset);
if (hi != 0)
{
rtx base_plus = gen_rtx_REG (SImode, REGNO (operands[2]) + 1);
if (reg_overlap_mentioned_p (base_plus, outval))
{
if (!reg_overlap_mentioned_p (scratch, outval))
std::swap (scratch, base_plus);
else
{
rtx scratch_hi = gen_rtx_REG (HImode, REGNO (operands[2]));
emit_insn (gen_movhi (scratch_hi, outval));
outval = scratch_hi;
}
}
emit_insn (gen_addsi3 (base_plus, base, GEN_INT (hi)));
base = base_plus;
offset = lo;
}
}
if (BYTES_BIG_ENDIAN)
{
emit_insn (gen_movqi (gen_rtx_MEM (QImode,
plus_constant (Pmode, base,
offset + 1)),
gen_lowpart (QImode, outval)));
emit_insn (gen_lshrsi3 (scratch,
gen_rtx_SUBREG (SImode, outval, 0),
GEN_INT (8)));
emit_insn (gen_movqi (gen_rtx_MEM (QImode, plus_constant (Pmode, base,
offset)),
gen_lowpart (QImode, scratch)));
}
else
{
emit_insn (gen_movqi (gen_rtx_MEM (QImode, plus_constant (Pmode, base,
offset)),
gen_lowpart (QImode, outval)));
emit_insn (gen_lshrsi3 (scratch,
gen_rtx_SUBREG (SImode, outval, 0),
GEN_INT (8)));
emit_insn (gen_movqi (gen_rtx_MEM (QImode,
plus_constant (Pmode, base,
offset + 1)),
gen_lowpart (QImode, scratch)));
}
}
static bool
arm_must_pass_in_stack (machine_mode mode, const_tree type)
{
if (TARGET_AAPCS_BASED)
return must_pass_in_stack_var_size (mode, type);
else
return must_pass_in_stack_var_size_or_pad (mode, type);
}
static pad_direction
arm_function_arg_padding (machine_mode mode, const_tree type)
{
if (!TARGET_AAPCS_BASED)
return default_function_arg_padding (mode, type);
if (type && BYTES_BIG_ENDIAN && INTEGRAL_TYPE_P (type))
return PAD_DOWNWARD;
return PAD_UPWARD;
}
bool
arm_pad_reg_upward (machine_mode mode,
tree type, int first ATTRIBUTE_UNUSED)
{
if (TARGET_AAPCS_BASED && BYTES_BIG_ENDIAN)
{
if (type)
{
if ((AGGREGATE_TYPE_P (type)
|| TREE_CODE (type) == COMPLEX_TYPE
|| FIXED_POINT_TYPE_P (type))
&& int_size_in_bytes (type) <= 4)
return true;
}
else
{
if ((COMPLEX_MODE_P (mode) || ALL_FIXED_POINT_MODE_P (mode))
&& GET_MODE_SIZE (mode) <= 4)
return true;
}
}
return !BYTES_BIG_ENDIAN;
}
bool
offset_ok_for_ldrd_strd (HOST_WIDE_INT offset)
{
HOST_WIDE_INT max_offset;
if (TARGET_THUMB2 && ((offset & 3) != 0))
return false;
if (TARGET_THUMB2)
max_offset = 1020;
else if (TARGET_ARM)
max_offset = 255;
else
return false;
return ((offset <= max_offset) && (offset >= -max_offset));
}
bool
operands_ok_ldrd_strd (rtx rt, rtx rt2, rtx rn, HOST_WIDE_INT offset,
bool wback, bool load)
{
unsigned int t, t2, n;
if (!reload_completed)
return true;
if (!offset_ok_for_ldrd_strd (offset))
return false;
t = REGNO (rt);
t2 = REGNO (rt2);
n = REGNO (rn);
if ((TARGET_THUMB2)
&& ((wback && (n == t || n == t2))
|| (t == SP_REGNUM)
|| (t == PC_REGNUM)
|| (t2 == SP_REGNUM)
|| (t2 == PC_REGNUM)
|| (!load && (n == PC_REGNUM))
|| (load && (t == t2))
|| (!wback && load && fix_cm3_ldrd && (n == t))))
return false;
if ((TARGET_ARM)
&& ((wback && (n == t || n == t2))
|| (t2 == PC_REGNUM)
|| (t % 2 != 0)   
|| (t2 != t + 1)
|| (n == PC_REGNUM)))
return false;
return true;
}
static bool
align_ok_ldrd_strd (HOST_WIDE_INT align, HOST_WIDE_INT offset)
{
return (unaligned_access
? (align >= BITS_PER_WORD && (offset & 3) == 0)
: (align >= 2 * BITS_PER_WORD && (offset & 7) == 0));
}
static bool
mem_ok_for_ldrd_strd (rtx mem, rtx *base, rtx *offset, HOST_WIDE_INT *align)
{
rtx addr;
gcc_assert (base != NULL && offset != NULL);
if (side_effects_p (mem))
return false;
if (GET_CODE (mem) == SUBREG)
return false;
gcc_assert (MEM_P (mem));
*offset = const0_rtx;
*align = MEM_ALIGN (mem);
addr = XEXP (mem, 0);
if (!arm_legitimate_address_p (DImode, addr,
reload_in_progress || reload_completed))
return false;
if (REG_P (addr))
{
*base = addr;
return true;
}
else if (GET_CODE (addr) == PLUS || GET_CODE (addr) == MINUS)
{
*base = XEXP (addr, 0);
*offset = XEXP (addr, 1);
return (REG_P (*base) && CONST_INT_P (*offset));
}
return false;
}
bool
gen_operands_ldrd_strd (rtx *operands, bool load,
bool const_store, bool commute)
{
int nops = 2;
HOST_WIDE_INT offsets[2], offset, align[2];
rtx base = NULL_RTX;
rtx cur_base, cur_offset, tmp;
int i, gap;
HARD_REG_SET regset;
gcc_assert (!const_store || !load);
for (i = 0; i < nops; i++)
{
if (!mem_ok_for_ldrd_strd (operands[nops+i], &cur_base, &cur_offset,
&align[i]))
return false;
if (i == 0)
base = cur_base;
else if (REGNO (base) != REGNO (cur_base))
return false;
offsets[i] = INTVAL (cur_offset);
if (GET_CODE (operands[i]) == SUBREG)
{
tmp = SUBREG_REG (operands[i]);
gcc_assert (GET_MODE (operands[i]) == GET_MODE (tmp));
operands[i] = tmp;
}
}
if (load && REGNO (operands[0]) == REGNO (base))
return false; 
if (load && REGNO (operands[0]) == REGNO (operands[1]))
return false; 
if (const_store
&& REGNO (operands[0]) == REGNO (operands[1])
&& INTVAL (operands[4]) != INTVAL (operands[5]))
{
if (TARGET_THUMB2)
{
CLEAR_HARD_REG_SET (regset);
tmp = peep2_find_free_register (0, 4, "r", SImode, &regset);
if (tmp == NULL_RTX)
return false;
operands[0] = tmp;
}
else if (TARGET_ARM)
{
int regno = REGNO (operands[0]);
if (!peep2_reg_dead_p (4, operands[0]))
{
if (regno % 2 == 0)
return false;
SET_HARD_REG_SET (regset);
CLEAR_HARD_REG_BIT(regset, regno - 1);
tmp = peep2_find_free_register (0, 4, "r", SImode, &regset);
if (tmp == NULL_RTX)
return false;
operands[0] = tmp;
}
else
{
CLEAR_HARD_REG_SET (regset);
tmp = peep2_find_free_register (0, 4, "r", DImode, &regset);
if (tmp != NULL_RTX)
{
operands[0] = simplify_gen_subreg (SImode, tmp, DImode, 0);
operands[1] = simplify_gen_subreg (SImode, tmp, DImode, 4);
}
else
{
SET_HARD_REG_SET (regset);
CLEAR_HARD_REG_BIT(regset,
regno % 2 == 0 ? regno + 1 : regno - 1);
tmp = peep2_find_free_register (0, 4, "r", SImode, &regset);
if (tmp == NULL_RTX)
return false;
operands[regno % 2 == 1 ? 0 : 1] = tmp;
}
}
gcc_assert (operands[0] != NULL_RTX);
gcc_assert (operands[1] != NULL_RTX);
gcc_assert (REGNO (operands[0]) % 2 == 0);
gcc_assert (REGNO (operands[1]) == REGNO (operands[0]) + 1);
}
}
if (offsets[0] > offsets[1])
{
gap = offsets[0] - offsets[1];
offset = offsets[1];
std::swap (operands[0], operands[1]);
std::swap (operands[2], operands[3]);
std::swap (align[0], align[1]);
if (const_store)
std::swap (operands[4], operands[5]);
}
else
{
gap = offsets[1] - offsets[0];
offset = offsets[0];
}
if (gap != 4)
return false;
if (!align_ok_ldrd_strd (align[0], offset))
return false;
if (operands_ok_ldrd_strd (operands[0], operands[1], base, offset,
false, load))
return true;
if (TARGET_THUMB2)
return false;
if (load && commute)
{
std::swap (operands[0], operands[1]);
if (operands_ok_ldrd_strd (operands[0], operands[1], base, offset,
false, load))
return true;
}
if (const_store)
{
if (!peep2_reg_dead_p (4, operands[0])
|| !peep2_reg_dead_p (4, operands[1]))
return false;
if (operands_ok_ldrd_strd (operands[1], operands[0], base, offset,
false, false))
{
std::swap (operands[0], operands[1]);
return true;
}
CLEAR_HARD_REG_SET (regset);
add_to_hard_reg_set (&regset, SImode, REGNO (operands[0]));
add_to_hard_reg_set (&regset, SImode, REGNO (operands[1]));
while (true)
{
tmp = peep2_find_free_register (0, 4, "r", DImode, &regset);
if (tmp == NULL_RTX)
return false;
operands[0] = simplify_gen_subreg (SImode, tmp, DImode, 0);
operands[1] = simplify_gen_subreg (SImode, tmp, DImode, 4);
gcc_assert (operands[0] != NULL_RTX);
gcc_assert (operands[1] != NULL_RTX);
gcc_assert (REGNO (operands[0]) % 2 == 0);
gcc_assert (REGNO (operands[0]) + 1 == REGNO (operands[1]));
return (operands_ok_ldrd_strd (operands[0], operands[1],
base, offset,
false, load));
}
}
return false;
}

static void
arm_print_value (FILE *f, rtx x)
{
switch (GET_CODE (x))
{
case CONST_INT:
fprintf (f, HOST_WIDE_INT_PRINT_HEX, INTVAL (x));
return;
case CONST_DOUBLE:
fprintf (f, "<0x%lx,0x%lx>", (long)XWINT (x, 2), (long)XWINT (x, 3));
return;
case CONST_VECTOR:
{
int i;
fprintf (f, "<");
for (i = 0; i < CONST_VECTOR_NUNITS (x); i++)
{
fprintf (f, HOST_WIDE_INT_PRINT_HEX, INTVAL (CONST_VECTOR_ELT (x, i)));
if (i < (CONST_VECTOR_NUNITS (x) - 1))
fputc (',', f);
}
fprintf (f, ">");
}
return;
case CONST_STRING:
fprintf (f, "\"%s\"", XSTR (x, 0));
return;
case SYMBOL_REF:
fprintf (f, "`%s'", XSTR (x, 0));
return;
case LABEL_REF:
fprintf (f, "L%d", INSN_UID (XEXP (x, 0)));
return;
case CONST:
arm_print_value (f, XEXP (x, 0));
return;
case PLUS:
arm_print_value (f, XEXP (x, 0));
fprintf (f, "+");
arm_print_value (f, XEXP (x, 1));
return;
case PC:
fprintf (f, "pc");
return;
default:
fprintf (f, "????");
return;
}
}

struct minipool_node
{
Mnode * next;
Mnode * prev;
HOST_WIDE_INT max_address;
HOST_WIDE_INT min_address;
int refcount;
HOST_WIDE_INT offset;
rtx value;
machine_mode mode;
int fix_size;
};
struct minipool_fixup
{
Mfix *            next;
rtx_insn *        insn;
HOST_WIDE_INT     address;
rtx *             loc;
machine_mode mode;
int               fix_size;
rtx               value;
Mnode *           minipool;
HOST_WIDE_INT     forwards;
HOST_WIDE_INT     backwards;
};
#define MINIPOOL_FIX_SIZE(mode) \
(GET_MODE_SIZE ((mode)) >= 4 ? GET_MODE_SIZE ((mode)) : 4)
static Mnode *	minipool_vector_head;
static Mnode *	minipool_vector_tail;
static rtx_code_label	*minipool_vector_label;
static int	minipool_pad;
Mfix * 		minipool_fix_head;
Mfix * 		minipool_fix_tail;
Mfix *		minipool_barrier;
#ifndef JUMP_TABLES_IN_TEXT_SECTION
#define JUMP_TABLES_IN_TEXT_SECTION 0
#endif
static HOST_WIDE_INT
get_jump_table_size (rtx_jump_table_data *insn)
{
if (JUMP_TABLES_IN_TEXT_SECTION || readonly_data_section == text_section)
{
rtx body = PATTERN (insn);
int elt = GET_CODE (body) == ADDR_DIFF_VEC ? 1 : 0;
HOST_WIDE_INT size;
HOST_WIDE_INT modesize;
modesize = GET_MODE_SIZE (GET_MODE (body));
size = modesize * XVECLEN (body, elt);
switch (modesize)
{
case 1:
size = (size + 1) & ~HOST_WIDE_INT_1;
break;
case 2:
break;
case 4:
if (TARGET_THUMB)
size += 2;
break;
default:
gcc_unreachable ();
}
return size;
}
return 0;
}
static HOST_WIDE_INT
get_label_padding (rtx label)
{
HOST_WIDE_INT align, min_insn_size;
align = 1 << label_to_alignment (label);
min_insn_size = TARGET_THUMB ? 2 : 4;
return align > min_insn_size ? align - min_insn_size : 0;
}
static Mnode *
move_minipool_fix_forward_ref (Mnode *mp, Mnode *max_mp,
HOST_WIDE_INT max_address)
{
gcc_assert (mp != max_mp);
if (max_mp == NULL)
{
if (max_address < mp->max_address)
mp->max_address = max_address;
}
else
{
if (max_address > max_mp->max_address - mp->fix_size)
mp->max_address = max_mp->max_address - mp->fix_size;
else
mp->max_address = max_address;
mp->prev->next = mp->next;
if (mp->next != NULL)
mp->next->prev = mp->prev;
else
minipool_vector_tail = mp->prev;
mp->next = max_mp;
mp->prev = max_mp->prev;
max_mp->prev = mp;
if (mp->prev != NULL)
mp->prev->next = mp;
else
minipool_vector_head = mp;
}
max_mp = mp;
while (mp->prev != NULL
&& mp->prev->max_address > mp->max_address - mp->prev->fix_size)
{
mp->prev->max_address = mp->max_address - mp->prev->fix_size;
mp = mp->prev;
}
return max_mp;
}
static Mnode *
add_minipool_forward_ref (Mfix *fix)
{
Mnode *       max_mp = NULL;
HOST_WIDE_INT max_address = fix->address + fix->forwards - minipool_pad;
Mnode *       mp;
if (minipool_vector_head &&
(fix->address + get_attr_length (fix->insn)
>= minipool_vector_head->max_address - fix->fix_size))
return NULL;
for (mp = minipool_vector_head; mp != NULL; mp = mp->next)
{
if (GET_CODE (fix->value) == GET_CODE (mp->value)
&& fix->mode == mp->mode
&& (!LABEL_P (fix->value)
|| (CODE_LABEL_NUMBER (fix->value)
== CODE_LABEL_NUMBER (mp->value)))
&& rtx_equal_p (fix->value, mp->value))
{
mp->refcount++;
return move_minipool_fix_forward_ref (mp, max_mp, max_address);
}
if (max_mp == NULL
&& mp->max_address > max_address)
max_mp = mp;
if (ARM_DOUBLEWORD_ALIGN
&& max_mp == NULL
&& fix->fix_size >= 8
&& mp->fix_size < 8)
{
max_mp = mp;
max_address = mp->max_address;
}
}
mp = XNEW (Mnode);
mp->fix_size = fix->fix_size;
mp->mode = fix->mode;
mp->value = fix->value;
mp->refcount = 1;
mp->min_address = -65536;
if (max_mp == NULL)
{
mp->max_address = max_address;
mp->next = NULL;
mp->prev = minipool_vector_tail;
if (mp->prev == NULL)
{
minipool_vector_head = mp;
minipool_vector_label = gen_label_rtx ();
}
else
mp->prev->next = mp;
minipool_vector_tail = mp;
}
else
{
if (max_address > max_mp->max_address - mp->fix_size)
mp->max_address = max_mp->max_address - mp->fix_size;
else
mp->max_address = max_address;
mp->next = max_mp;
mp->prev = max_mp->prev;
max_mp->prev = mp;
if (mp->prev != NULL)
mp->prev->next = mp;
else
minipool_vector_head = mp;
}
max_mp = mp;
while (mp->prev != NULL
&& mp->prev->max_address > mp->max_address - mp->prev->fix_size)
{
mp->prev->max_address = mp->max_address - mp->prev->fix_size;
mp = mp->prev;
}
return max_mp;
}
static Mnode *
move_minipool_fix_backward_ref (Mnode *mp, Mnode *min_mp,
HOST_WIDE_INT  min_address)
{
HOST_WIDE_INT offset;
gcc_assert (mp != min_mp);
if (min_mp == NULL)
{
if (min_address > mp->min_address)
mp->min_address = min_address;
}
else
{
mp->min_address = min_address;
mp->next->prev = mp->prev;
if (mp->prev != NULL)
mp->prev->next = mp->next;
else
minipool_vector_head = mp->next;
mp->prev = min_mp;
mp->next = min_mp->next;
min_mp->next = mp;
if (mp->next != NULL)
mp->next->prev = mp;
else
minipool_vector_tail = mp;
}
min_mp = mp;
offset = 0;
for (mp = minipool_vector_head; mp != NULL; mp = mp->next)
{
mp->offset = offset;
if (mp->refcount > 0)
offset += mp->fix_size;
if (mp->next && mp->next->min_address < mp->min_address + mp->fix_size)
mp->next->min_address = mp->min_address + mp->fix_size;
}
return min_mp;
}
static Mnode *
add_minipool_backward_ref (Mfix *fix)
{
Mnode *min_mp = NULL;
HOST_WIDE_INT  min_address = fix->address - fix->backwards;
Mnode *mp;
if (min_address >= minipool_barrier->address
|| (minipool_vector_tail->min_address + fix->fix_size
>= minipool_barrier->address))
return NULL;
for (mp = minipool_vector_tail; mp != NULL; mp = mp->prev)
{
if (GET_CODE (fix->value) == GET_CODE (mp->value)
&& fix->mode == mp->mode
&& (!LABEL_P (fix->value)
|| (CODE_LABEL_NUMBER (fix->value)
== CODE_LABEL_NUMBER (mp->value)))
&& rtx_equal_p (fix->value, mp->value)
&& (mp->max_address
> (minipool_barrier->address
+ minipool_vector_tail->offset
+ minipool_vector_tail->fix_size)))
{
mp->refcount++;
return move_minipool_fix_backward_ref (mp, min_mp, min_address);
}
if (min_mp != NULL)
mp->min_address += fix->fix_size;
else
{
if (mp->min_address < min_address)
{
if (ARM_DOUBLEWORD_ALIGN
&& fix->fix_size >= 8 && mp->fix_size < 8)
return NULL;
else
min_mp = mp;
}
else if (mp->max_address
< minipool_barrier->address + mp->offset + fix->fix_size)
{
if (ARM_DOUBLEWORD_ALIGN
&& fix->fix_size >= 8 && mp->fix_size < 8)
return NULL;
else
{
min_mp = mp;
min_address = mp->min_address + fix->fix_size;
}
}
else if (ARM_DOUBLEWORD_ALIGN
&& fix->fix_size < 8
&& mp->fix_size >= 8)
{
min_mp = mp;
min_address = mp->min_address + fix->fix_size;
}
}
}
mp = XNEW (Mnode);
mp->fix_size = fix->fix_size;
mp->mode = fix->mode;
mp->value = fix->value;
mp->refcount = 1;
mp->max_address = minipool_barrier->address + 65536;
mp->min_address = min_address;
if (min_mp == NULL)
{
mp->prev = NULL;
mp->next = minipool_vector_head;
if (mp->next == NULL)
{
minipool_vector_tail = mp;
minipool_vector_label = gen_label_rtx ();
}
else
mp->next->prev = mp;
minipool_vector_head = mp;
}
else
{
mp->next = min_mp->next;
mp->prev = min_mp;
min_mp->next = mp;
if (mp->next != NULL)
mp->next->prev = mp;
else
minipool_vector_tail = mp;
}
min_mp = mp;
if (mp->prev)
mp = mp->prev;
else
mp->offset = 0;
while (mp->next != NULL)
{
if (mp->next->min_address < mp->min_address + mp->fix_size)
mp->next->min_address = mp->min_address + mp->fix_size;
if (mp->refcount)
mp->next->offset = mp->offset + mp->fix_size;
else
mp->next->offset = mp->offset;
mp = mp->next;
}
return min_mp;
}
static void
assign_minipool_offsets (Mfix *barrier)
{
HOST_WIDE_INT offset = 0;
Mnode *mp;
minipool_barrier = barrier;
for (mp = minipool_vector_head; mp != NULL; mp = mp->next)
{
mp->offset = offset;
if (mp->refcount > 0)
offset += mp->fix_size;
}
}
static void
dump_minipool (rtx_insn *scan)
{
Mnode * mp;
Mnode * nmp;
int align64 = 0;
if (ARM_DOUBLEWORD_ALIGN)
for (mp = minipool_vector_head; mp != NULL; mp = mp->next)
if (mp->refcount > 0 && mp->fix_size >= 8)
{
align64 = 1;
break;
}
if (dump_file)
fprintf (dump_file,
";; Emitting minipool after insn %u; address %ld; align %d (bytes)\n",
INSN_UID (scan), (unsigned long) minipool_barrier->address, align64 ? 8 : 4);
scan = emit_label_after (gen_label_rtx (), scan);
scan = emit_insn_after (align64 ? gen_align_8 () : gen_align_4 (), scan);
scan = emit_label_after (minipool_vector_label, scan);
for (mp = minipool_vector_head; mp != NULL; mp = nmp)
{
if (mp->refcount > 0)
{
if (dump_file)
{
fprintf (dump_file,
";;  Offset %u, min %ld, max %ld ",
(unsigned) mp->offset, (unsigned long) mp->min_address,
(unsigned long) mp->max_address);
arm_print_value (dump_file, mp->value);
fputc ('\n', dump_file);
}
rtx val = copy_rtx (mp->value);
switch (GET_MODE_SIZE (mp->mode))
{
#ifdef HAVE_consttable_1
case 1:
scan = emit_insn_after (gen_consttable_1 (val), scan);
break;
#endif
#ifdef HAVE_consttable_2
case 2:
scan = emit_insn_after (gen_consttable_2 (val), scan);
break;
#endif
#ifdef HAVE_consttable_4
case 4:
scan = emit_insn_after (gen_consttable_4 (val), scan);
break;
#endif
#ifdef HAVE_consttable_8
case 8:
scan = emit_insn_after (gen_consttable_8 (val), scan);
break;
#endif
#ifdef HAVE_consttable_16
case 16:
scan = emit_insn_after (gen_consttable_16 (val), scan);
break;
#endif
default:
gcc_unreachable ();
}
}
nmp = mp->next;
free (mp);
}
minipool_vector_head = minipool_vector_tail = NULL;
scan = emit_insn_after (gen_consttable_end (), scan);
scan = emit_barrier_after (scan);
}
static int
arm_barrier_cost (rtx_insn *insn)
{
int base_cost = 50;
rtx_insn *next = next_nonnote_insn (insn);
if (next != NULL && LABEL_P (next))
base_cost -= 20;
switch (GET_CODE (insn))
{
case CODE_LABEL:
return 50;
case INSN:
case CALL_INSN:
return base_cost;
case JUMP_INSN:
return base_cost - 10;
default:
return base_cost + 10;
}
}
static Mfix *
create_fix_barrier (Mfix *fix, HOST_WIDE_INT max_address)
{
HOST_WIDE_INT count = 0;
rtx_barrier *barrier;
rtx_insn *from = fix->insn;
rtx_insn *selected = NULL;
int selected_cost;
HOST_WIDE_INT selected_address;
Mfix * new_fix;
HOST_WIDE_INT max_count = max_address - fix->address;
rtx_code_label *label = gen_label_rtx ();
selected_cost = arm_barrier_cost (from);
selected_address = fix->address;
while (from && count < max_count)
{
rtx_jump_table_data *tmp;
int new_cost;
gcc_assert (!BARRIER_P (from));
if (LABEL_P (from))
count += get_label_padding (from);
else
count += get_attr_length (from);
if (tablejump_p (from, NULL, &tmp))
{
count += get_jump_table_size (tmp);
new_cost = arm_barrier_cost (from);
if (count < max_count
&& (!selected || new_cost <= selected_cost))
{
selected = tmp;
selected_cost = new_cost;
selected_address = fix->address + count;
}
from = NEXT_INSN (tmp);
continue;
}
new_cost = arm_barrier_cost (from);
if (count < max_count
&& (!selected || new_cost <= selected_cost))
{
selected = from;
selected_cost = new_cost;
selected_address = fix->address + count;
}
from = NEXT_INSN (from);
}
gcc_assert (selected);
from = emit_jump_insn_after (gen_jump (label), selected);
JUMP_LABEL (from) = label;
barrier = emit_barrier_after (from);
emit_label_after (label, barrier);
new_fix = (Mfix *) obstack_alloc (&minipool_obstack, sizeof (* new_fix));
new_fix->insn = barrier;
new_fix->address = selected_address;
new_fix->next = fix->next;
fix->next = new_fix;
return new_fix;
}
static void
push_minipool_barrier (rtx_insn *insn, HOST_WIDE_INT address)
{
Mfix * fix = (Mfix *) obstack_alloc (&minipool_obstack, sizeof (* fix));
fix->insn = insn;
fix->address = address;
fix->next = NULL;
if (minipool_fix_head != NULL)
minipool_fix_tail->next = fix;
else
minipool_fix_head = fix;
minipool_fix_tail = fix;
}
static void
push_minipool_fix (rtx_insn *insn, HOST_WIDE_INT address, rtx *loc,
machine_mode mode, rtx value)
{
gcc_assert (!arm_disable_literal_pool);
Mfix * fix = (Mfix *) obstack_alloc (&minipool_obstack, sizeof (* fix));
fix->insn = insn;
fix->address = address;
fix->loc = loc;
fix->mode = mode;
fix->fix_size = MINIPOOL_FIX_SIZE (mode);
fix->value = value;
fix->forwards = get_attr_pool_range (insn);
fix->backwards = get_attr_neg_pool_range (insn);
fix->minipool = NULL;
gcc_assert (fix->forwards || fix->backwards);
if (ARM_DOUBLEWORD_ALIGN && fix->fix_size >= 8)
minipool_pad = 4;
if (dump_file)
{
fprintf (dump_file,
";; %smode fixup for i%d; addr %lu, range (%ld,%ld): ",
GET_MODE_NAME (mode),
INSN_UID (insn), (unsigned long) address,
-1 * (long)fix->backwards, (long)fix->forwards);
arm_print_value (dump_file, fix->value);
fprintf (dump_file, "\n");
}
fix->next = NULL;
if (minipool_fix_head != NULL)
minipool_fix_tail->next = fix;
else
minipool_fix_head = fix;
minipool_fix_tail = fix;
}
int
arm_max_const_double_inline_cost ()
{
return ((optimize_size || arm_ld_sched) ? 3 : 4);
}
int
arm_const_double_inline_cost (rtx val)
{
rtx lowpart, highpart;
machine_mode mode;
mode = GET_MODE (val);
if (mode == VOIDmode)
mode = DImode;
gcc_assert (GET_MODE_SIZE (mode) == 8);
lowpart = gen_lowpart (SImode, val);
highpart = gen_highpart_mode (SImode, mode, val);
gcc_assert (CONST_INT_P (lowpart));
gcc_assert (CONST_INT_P (highpart));
return (arm_gen_constant (SET, SImode, NULL_RTX, INTVAL (lowpart),
NULL_RTX, NULL_RTX, 0, 0)
+ arm_gen_constant (SET, SImode, NULL_RTX, INTVAL (highpart),
NULL_RTX, NULL_RTX, 0, 0));
}
static inline int
arm_const_inline_cost (enum rtx_code code, rtx val)
{
return arm_gen_constant (code, SImode, NULL_RTX, INTVAL (val),
NULL_RTX, NULL_RTX, 1, 0);
}
bool
arm_const_double_by_parts (rtx val)
{
machine_mode mode = GET_MODE (val);
rtx part;
if (optimize_size || arm_ld_sched)
return true;
if (mode == VOIDmode)
mode = DImode;
part = gen_highpart_mode (SImode, mode, val);
gcc_assert (CONST_INT_P (part));
if (const_ok_for_arm (INTVAL (part))
|| const_ok_for_arm (~INTVAL (part)))
return true;
part = gen_lowpart (SImode, val);
gcc_assert (CONST_INT_P (part));
if (const_ok_for_arm (INTVAL (part))
|| const_ok_for_arm (~INTVAL (part)))
return true;
return false;
}
bool
arm_const_double_by_immediates (rtx val)
{
machine_mode mode = GET_MODE (val);
rtx part;
if (mode == VOIDmode)
mode = DImode;
part = gen_highpart_mode (SImode, mode, val);
gcc_assert (CONST_INT_P (part));
if (!const_ok_for_arm (INTVAL (part)))
return false;
part = gen_lowpart (SImode, val);
gcc_assert (CONST_INT_P (part));
if (!const_ok_for_arm (INTVAL (part)))
return false;
return true;
}
static void
note_invalid_constants (rtx_insn *insn, HOST_WIDE_INT address, int do_pushes)
{
int opno;
extract_constrain_insn (insn);
if (recog_data.n_alternatives == 0)
return;
preprocess_constraints (insn);
const operand_alternative *op_alt = which_op_alt ();
for (opno = 0; opno < recog_data.n_operands; opno++)
{
if (recog_data.operand_type[opno] != OP_IN)
continue;
if (op_alt[opno].memory_ok)
{
rtx op = recog_data.operand[opno];
if (CONSTANT_P (op))
{
if (do_pushes)
push_minipool_fix (insn, address, recog_data.operand_loc[opno],
recog_data.operand_mode[opno], op);
}
else if (MEM_P (op)
&& GET_CODE (XEXP (op, 0)) == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (XEXP (op, 0)))
{
if (do_pushes)
{
rtx cop = avoid_constant_pool_reference (op);
if (op == cop)
cop = get_pool_constant (XEXP (op, 0));
push_minipool_fix (insn, address,
recog_data.operand_loc[opno],
recog_data.operand_mode[opno], cop);
}
}
}
}
return;
}
static unsigned HOST_WIDE_INT
comp_not_to_clear_mask_str_un (tree arg_type, int * regno,
uint32_t * padding_bits_to_clear,
unsigned starting_bit, int * last_used_bit)
{
unsigned HOST_WIDE_INT not_to_clear_reg_mask = 0;
if (TREE_CODE (arg_type) == RECORD_TYPE)
{
unsigned current_bit = starting_bit;
tree field;
long int offset, size;
field = TYPE_FIELDS (arg_type);
while (field)
{
offset = starting_bit;
offset += TREE_INT_CST_ELT (DECL_FIELD_BIT_OFFSET (field), 0);
offset %= 32;
size = TREE_INT_CST_ELT (DECL_SIZE (field), 0);
if (*last_used_bit != offset)
{
if (offset < *last_used_bit)
{
uint32_t mask;
mask  = ((uint32_t)-1) - ((uint32_t) 1 << *last_used_bit);
mask++;
padding_bits_to_clear[*regno] |= mask;
not_to_clear_reg_mask |= HOST_WIDE_INT_1U << *regno;
(*regno)++;
}
else
{
uint32_t mask;
mask = ((uint32_t)-1) >> (32 - offset);
mask -= ((uint32_t) 1 << *last_used_bit) - 1;
padding_bits_to_clear[*regno] |= mask;
}
current_bit = offset;
}
if (RECORD_OR_UNION_TYPE_P (TREE_TYPE (field)))
{
*last_used_bit = current_bit;
not_to_clear_reg_mask
|= comp_not_to_clear_mask_str_un (TREE_TYPE (field), regno,
padding_bits_to_clear, offset,
last_used_bit);
}
else
{
current_bit += size;
while (current_bit >= 32)
{
current_bit-=32;
not_to_clear_reg_mask |= HOST_WIDE_INT_1U << *regno;
(*regno)++;
}
*last_used_bit = current_bit;
}
field = TREE_CHAIN (field);
}
not_to_clear_reg_mask |= HOST_WIDE_INT_1U << *regno;
}
else if (TREE_CODE (arg_type) == UNION_TYPE)
{
tree field, field_t;
int i, regno_t, field_size;
int max_reg = -1;
int max_bit = -1;
uint32_t mask;
uint32_t padding_bits_to_clear_res[NUM_ARG_REGS]
= {-1, -1, -1, -1};
field = TYPE_FIELDS (arg_type);
while (field)
{
uint32_t padding_bits_to_clear_t[NUM_ARG_REGS]
= {0U, 0U, 0U, 0U};
int last_used_bit_t = *last_used_bit;
regno_t = *regno;
field_t = TREE_TYPE (field);
if (RECORD_OR_UNION_TYPE_P (field_t))
not_to_clear_reg_mask
|= comp_not_to_clear_mask_str_un (field_t, &regno_t,
&padding_bits_to_clear_t[0],
starting_bit, &last_used_bit_t);
else
{
field_size = TREE_INT_CST_ELT (DECL_SIZE (field), 0);
regno_t = (field_size / 32) + *regno;
last_used_bit_t = (starting_bit + field_size) % 32;
}
for (i = *regno; i < regno_t; i++)
{
padding_bits_to_clear_res[i] &= padding_bits_to_clear_t[i];
}
mask = (((uint32_t) -1) - ((uint32_t) 1 << last_used_bit_t)) + 1;
padding_bits_to_clear_res[regno_t]
&= padding_bits_to_clear_t[regno_t] | mask;
if (max_reg < regno_t)
{
max_reg = regno_t;
max_bit = last_used_bit_t;
}
else if (max_reg == regno_t && max_bit < last_used_bit_t)
max_bit = last_used_bit_t;
field = TREE_CHAIN (field);
}
for (i=*regno; i < max_reg; i++)
padding_bits_to_clear[i] |= padding_bits_to_clear_res[i];
mask = ((uint32_t) 1 << max_bit) - 1;
padding_bits_to_clear[max_reg]
|= padding_bits_to_clear_res[max_reg] & mask;
*regno = max_reg;
*last_used_bit = max_bit;
}
else
gcc_unreachable ();
return not_to_clear_reg_mask;
}
static unsigned HOST_WIDE_INT
compute_not_to_clear_mask (tree arg_type, rtx arg_rtx, int regno,
uint32_t * padding_bits_to_clear)
{
int last_used_bit = 0;
unsigned HOST_WIDE_INT not_to_clear_mask;
if (RECORD_OR_UNION_TYPE_P (arg_type))
{
not_to_clear_mask
= comp_not_to_clear_mask_str_un (arg_type, &regno,
padding_bits_to_clear, 0,
&last_used_bit);
if (last_used_bit != 0)
padding_bits_to_clear[regno]
|= ((uint32_t)-1) - ((uint32_t) 1 << last_used_bit) + 1;
else
not_to_clear_mask &= ~(HOST_WIDE_INT_1U << regno);
}
else
{
not_to_clear_mask = 0;
if (GET_MODE (arg_rtx) == BLKmode)
{
int i, arg_regs;
rtx reg;
gcc_assert (TARGET_HARD_FLOAT_ABI);
for (i = 0; i < XVECLEN (arg_rtx, 0); i++)
{
reg = XEXP (XVECEXP (arg_rtx, 0, i), 0);
gcc_assert (REG_P (reg));
not_to_clear_mask |= HOST_WIDE_INT_1U << REGNO (reg);
arg_regs = ARM_NUM_REGS (GET_MODE (reg));
if (arg_regs > 1)
{
unsigned HOST_WIDE_INT mask;
mask = HOST_WIDE_INT_1U << (REGNO (reg) + arg_regs);
mask -= HOST_WIDE_INT_1U << REGNO (reg);
not_to_clear_mask |= mask;
}
}
}
else
{
int arg_regs = ARM_NUM_REGS (GET_MODE (arg_rtx));
not_to_clear_mask |= HOST_WIDE_INT_1U << REGNO (arg_rtx);
if (arg_regs > 1)
{
unsigned HOST_WIDE_INT
mask = HOST_WIDE_INT_1U << (REGNO (arg_rtx) + arg_regs);
mask -= HOST_WIDE_INT_1U << REGNO (arg_rtx);
not_to_clear_mask |= mask;
}
}
}
return not_to_clear_mask;
}
static void
cmse_clear_registers (sbitmap to_clear_bitmap, uint32_t *padding_bits_to_clear,
int padding_bits_len, rtx scratch_reg, rtx clearing_reg)
{
bool saved_clearing = false;
rtx saved_clearing_reg = NULL_RTX;
int i, regno, clearing_regno, minregno = R0_REGNUM, maxregno = minregno - 1;
gcc_assert (arm_arch_cmse);
if (!bitmap_empty_p (to_clear_bitmap))
{
minregno = bitmap_first_set_bit (to_clear_bitmap);
maxregno = bitmap_last_set_bit (to_clear_bitmap);
}
clearing_regno = REGNO (clearing_reg);
gcc_assert (padding_bits_len <= NUM_ARG_REGS);
for (i = 0, regno = R0_REGNUM; i < padding_bits_len; i++, regno++)
{
uint64_t mask;
rtx rtx16, dest, cleared_reg = gen_rtx_REG (SImode, regno);
if (padding_bits_to_clear[i] == 0)
continue;
if (TARGET_THUMB1
&& REGNO (scratch_reg) > LAST_LO_REGNUM)
{
if ((clearing_regno > maxregno
|| !bitmap_bit_p (to_clear_bitmap, clearing_regno))
&& !saved_clearing)
{
gcc_assert (clearing_regno <= LAST_LO_REGNUM);
emit_move_insn (scratch_reg, clearing_reg);
saved_clearing = true;
saved_clearing_reg = scratch_reg;
}
scratch_reg = clearing_reg;
}
mask = (~padding_bits_to_clear[i]) & 0xFFFF;
emit_move_insn (scratch_reg, gen_int_mode (mask, SImode));
mask = (~padding_bits_to_clear[i]) >> 16;
rtx16 = gen_int_mode (16, SImode);
dest = gen_rtx_ZERO_EXTRACT (SImode, scratch_reg, rtx16, rtx16);
if (mask)
emit_insn (gen_rtx_SET (dest, gen_int_mode (mask, SImode)));
emit_insn (gen_andsi3 (cleared_reg, cleared_reg, scratch_reg));
}
if (saved_clearing)
emit_move_insn (clearing_reg, saved_clearing_reg);
if (clearing_regno <= maxregno
&& bitmap_bit_p (to_clear_bitmap, clearing_regno))
{
emit_move_insn (clearing_reg, const0_rtx);
emit_use (clearing_reg);
bitmap_clear_bit (to_clear_bitmap, clearing_regno);
}
for (regno = minregno; regno <= maxregno; regno++)
{
if (!bitmap_bit_p (to_clear_bitmap, regno))
continue;
if (IS_VFP_REGNUM (regno))
{
if (TARGET_VFP_DOUBLE
&& VFP_REGNO_OK_FOR_DOUBLE (regno)
&& bitmap_bit_p (to_clear_bitmap, regno + 1))
{
emit_move_insn (gen_rtx_REG (DFmode, regno),
CONST1_RTX (DFmode));
emit_use (gen_rtx_REG (DFmode, regno));
regno++;
}
else
{
emit_move_insn (gen_rtx_REG (SFmode, regno),
CONST1_RTX (SFmode));
emit_use (gen_rtx_REG (SFmode, regno));
}
}
else
{
emit_move_insn (gen_rtx_REG (SImode, regno), clearing_reg);
emit_use (gen_rtx_REG (SImode, regno));
}
}
}
static void
cmse_nonsecure_call_clear_caller_saved (void)
{
basic_block bb;
FOR_EACH_BB_FN (bb, cfun)
{
rtx_insn *insn;
FOR_BB_INSNS (bb, insn)
{
unsigned address_regnum, regno, maxregno =
TARGET_HARD_FLOAT_ABI ? D7_VFP_REGNUM : NUM_ARG_REGS - 1;
auto_sbitmap to_clear_bitmap (maxregno + 1);
rtx_insn *seq;
rtx pat, call, unspec, clearing_reg, ip_reg, shift;
rtx address;
CUMULATIVE_ARGS args_so_far_v;
cumulative_args_t args_so_far;
tree arg_type, fntype;
bool first_param = true;
function_args_iterator args_iter;
uint32_t padding_bits_to_clear[4] = {0U, 0U, 0U, 0U};
if (!NONDEBUG_INSN_P (insn))
continue;
if (!CALL_P (insn))
continue;
pat = PATTERN (insn);
gcc_assert (GET_CODE (pat) == PARALLEL && XVECLEN (pat, 0) > 0);
call = XVECEXP (pat, 0, 0);
if (GET_CODE (call) == SET)
call = SET_SRC (call);
unspec = XEXP (call, 0);
if (GET_CODE (unspec) != UNSPEC
|| XINT (unspec, 1) != UNSPEC_NONSECURE_MEM)
continue;
bitmap_clear (to_clear_bitmap);
bitmap_set_range (to_clear_bitmap, R0_REGNUM, NUM_ARG_REGS);
if (TARGET_HARD_FLOAT_ABI)
{
auto_sbitmap float_bitmap (maxregno + 1);
bitmap_clear (float_bitmap);
bitmap_set_range (float_bitmap, FIRST_VFP_REGNUM,
D7_VFP_REGNUM - FIRST_VFP_REGNUM + 1);
bitmap_ior (to_clear_bitmap, to_clear_bitmap, float_bitmap);
}
address = RTVEC_ELT (XVEC (unspec, 0), 0);
gcc_assert (MEM_P (address));
gcc_assert (REG_P (XEXP (address, 0)));
address_regnum = REGNO (XEXP (address, 0));
if (address_regnum < R0_REGNUM + NUM_ARG_REGS)
bitmap_clear_bit (to_clear_bitmap, address_regnum);
set_block_for_insn (insn, bb);
df_set_flags (DF_DEFER_INSN_RESCAN);
start_sequence ();
emit_insn (gen_blockage ());
fntype = TREE_TYPE (MEM_EXPR (address));
arm_init_cumulative_args (&args_so_far_v, fntype, NULL_RTX,
NULL_TREE);
args_so_far = pack_cumulative_args (&args_so_far_v);
FOREACH_FUNCTION_ARGS (fntype, arg_type, args_iter)
{
rtx arg_rtx;
uint64_t to_clear_args_mask;
machine_mode arg_mode = TYPE_MODE (arg_type);
if (VOID_TYPE_P (arg_type))
continue;
if (!first_param)
arm_function_arg_advance (args_so_far, arg_mode, arg_type,
true);
arg_rtx = arm_function_arg (args_so_far, arg_mode, arg_type,
true);
gcc_assert (REG_P (arg_rtx));
to_clear_args_mask
= compute_not_to_clear_mask (arg_type, arg_rtx,
REGNO (arg_rtx),
&padding_bits_to_clear[0]);
if (to_clear_args_mask)
{
for (regno = R0_REGNUM; regno <= maxregno; regno++)
{
if (to_clear_args_mask & (1ULL << regno))
bitmap_clear_bit (to_clear_bitmap, regno);
}
}
first_param = false;
}
clearing_reg = XEXP (address, 0);
shift = gen_rtx_LSHIFTRT (SImode, clearing_reg, const1_rtx);
emit_insn (gen_rtx_SET (clearing_reg, shift));
shift = gen_rtx_ASHIFT (SImode, clearing_reg, const1_rtx);
emit_insn (gen_rtx_SET (clearing_reg, shift));
ip_reg = gen_rtx_REG (SImode, IP_REGNUM);
cmse_clear_registers (to_clear_bitmap, padding_bits_to_clear,
NUM_ARG_REGS, ip_reg, clearing_reg);
seq = get_insns ();
end_sequence ();
emit_insn_before (seq, insn);
}
}
}
static void
thumb1_reorg (void)
{
basic_block bb;
FOR_EACH_BB_FN (bb, cfun)
{
rtx dest, src;
rtx cmp, op0, op1, set = NULL;
rtx_insn *prev, *insn = BB_END (bb);
bool insn_clobbered = false;
while (insn != BB_HEAD (bb) && !NONDEBUG_INSN_P (insn))
insn = PREV_INSN (insn);
if (insn == BB_HEAD (bb)
|| INSN_CODE (insn) != CODE_FOR_cbranchsi4_insn)
continue;
cmp = XEXP (SET_SRC (PATTERN (insn)), 0);
op0 = XEXP (cmp, 0);
op1 = XEXP (cmp, 1);
if (!CONST_INT_P (op1) || INTVAL (op1) != 0)
continue;
gcc_assert (insn != BB_HEAD (bb));
for (prev = PREV_INSN (insn);
(!insn_clobbered
&& prev != BB_HEAD (bb)
&& (NOTE_P (prev)
|| DEBUG_INSN_P (prev)
|| ((set = single_set (prev)) != NULL
&& get_attr_conds (prev) == CONDS_NOCOND)));
prev = PREV_INSN (prev))
{
if (reg_set_p (op0, prev))
insn_clobbered = true;
}
if (insn_clobbered)
continue;
if (!set)
continue;
dest = SET_DEST (set);
src = SET_SRC (set);
if (!low_register_operand (dest, SImode)
|| !low_register_operand (src, SImode))
continue;
if (REGNO (op0) == REGNO (src) || REGNO (op0) == REGNO (dest))
{
dest = copy_rtx (dest);
src = copy_rtx (src);
src = gen_rtx_MINUS (SImode, src, const0_rtx);
PATTERN (prev) = gen_rtx_SET (dest, src);
INSN_CODE (prev) = -1;
XEXP (cmp, 0) = copy_rtx (dest);
INSN_CODE (insn) = -1;
}
}
}
static void
thumb2_reorg (void)
{
basic_block bb;
regset_head live;
INIT_REG_SET (&live);
compute_bb_for_insn ();
df_analyze ();
enum Convert_Action {SKIP, CONV, SWAP_CONV};
FOR_EACH_BB_FN (bb, cfun)
{
if ((current_tune->disparage_flag_setting_t16_encodings
== tune_params::DISPARAGE_FLAGS_ALL)
&& optimize_bb_for_speed_p (bb))
continue;
rtx_insn *insn;
Convert_Action action = SKIP;
Convert_Action action_for_partial_flag_setting
= ((current_tune->disparage_flag_setting_t16_encodings
!= tune_params::DISPARAGE_FLAGS_NEITHER)
&& optimize_bb_for_speed_p (bb))
? SKIP : CONV;
COPY_REG_SET (&live, DF_LR_OUT (bb));
df_simulate_initialize_backwards (bb, &live);
FOR_BB_INSNS_REVERSE (bb, insn)
{
if (NONJUMP_INSN_P (insn)
&& !REGNO_REG_SET_P (&live, CC_REGNUM)
&& GET_CODE (PATTERN (insn)) == SET)
{
action = SKIP;
rtx pat = PATTERN (insn);
rtx dst = XEXP (pat, 0);
rtx src = XEXP (pat, 1);
rtx op0 = NULL_RTX, op1 = NULL_RTX;
if (UNARY_P (src) || BINARY_P (src))
op0 = XEXP (src, 0);
if (BINARY_P (src))
op1 = XEXP (src, 1);
if (low_register_operand (dst, SImode))
{
switch (GET_CODE (src))
{
case PLUS:
if (rtx_equal_p (dst, op0)
&& register_operand (op1, SImode))
break;
if (low_register_operand (op0, SImode))
{
if (low_register_operand (op1, SImode))
action = CONV;
else if (rtx_equal_p (dst, op0)
&& CONST_INT_P (op1)
&& IN_RANGE (INTVAL (op1), -255, 255))
action = CONV;
else if (CONST_INT_P (op1)
&& IN_RANGE (INTVAL (op1), -7, 7))
action = CONV;
}
else if (GET_CODE (XEXP (src, 0)) == PLUS
&& rtx_equal_p (XEXP (XEXP (src, 0), 0), dst)
&& low_register_operand (XEXP (XEXP (src, 0), 1),
SImode)
&& COMPARISON_P (op1)
&& cc_register (XEXP (op1, 0), VOIDmode)
&& maybe_get_arm_condition_code (op1) == ARM_CS
&& XEXP (op1, 1) == const0_rtx)
action = CONV;
break;
case MINUS:
if (low_register_operand (op0, SImode)
&& low_register_operand (op1, SImode))
action = CONV;
break;
case MULT:
if (!optimize_size)
break;
case AND:
case IOR:
case XOR:
if (rtx_equal_p (dst, op0)
&& low_register_operand (op1, SImode))
action = action_for_partial_flag_setting;
else if (rtx_equal_p (dst, op1)
&& low_register_operand (op0, SImode))
action = action_for_partial_flag_setting == SKIP
? SKIP : SWAP_CONV;
break;
case ASHIFTRT:
case ASHIFT:
case LSHIFTRT:
if (rtx_equal_p (dst, op0)
&& low_register_operand (op1, SImode))
action = action_for_partial_flag_setting;
else if (low_register_operand (op0, SImode)
&& CONST_INT_P (op1)
&& IN_RANGE (INTVAL (op1), 0, 31))
action = action_for_partial_flag_setting;
break;
case ROTATERT:
if (rtx_equal_p (dst, op0)
&& low_register_operand (op1, SImode))
action = action_for_partial_flag_setting;
break;
case NOT:
if (low_register_operand (op0, SImode))
action = action_for_partial_flag_setting;
break;
case NEG:
if (low_register_operand (op0, SImode))
action = CONV;
break;
case CONST_INT:
if (CONST_INT_P (src)
&& IN_RANGE (INTVAL (src), 0, 255))
action = action_for_partial_flag_setting;
break;
case REG:
break;
default:
break;
}
}
if (action != SKIP)
{
rtx ccreg = gen_rtx_REG (CCmode, CC_REGNUM);
rtx clobber = gen_rtx_CLOBBER (VOIDmode, ccreg);
rtvec vec;
if (action == SWAP_CONV)
{
src = copy_rtx (src);
XEXP (src, 0) = op1;
XEXP (src, 1) = op0;
pat = gen_rtx_SET (dst, src);
vec = gen_rtvec (2, pat, clobber);
}
else 
vec = gen_rtvec (2, pat, clobber);
PATTERN (insn) = gen_rtx_PARALLEL (VOIDmode, vec);
INSN_CODE (insn) = -1;
}
}
if (NONDEBUG_INSN_P (insn))
df_simulate_one_insn_backwards (bb, insn, &live);
}
}
CLEAR_REG_SET (&live);
}
static void
arm_reorg (void)
{
rtx_insn *insn;
HOST_WIDE_INT address = 0;
Mfix * fix;
if (use_cmse)
cmse_nonsecure_call_clear_caller_saved ();
if (cfun->is_thunk)
;
else if (TARGET_THUMB1)
thumb1_reorg ();
else if (TARGET_THUMB2)
thumb2_reorg ();
if (!optimize)
split_all_insns_noflow ();
if (arm_disable_literal_pool)
return ;
minipool_fix_head = minipool_fix_tail = NULL;
insn = get_insns ();
gcc_assert (NOTE_P (insn));
minipool_pad = 0;
for (insn = next_nonnote_insn (insn); insn; insn = next_nonnote_insn (insn))
{
if (BARRIER_P (insn))
push_minipool_barrier (insn, address);
else if (INSN_P (insn))
{
rtx_jump_table_data *table;
note_invalid_constants (insn, address, true);
address += get_attr_length (insn);
if (tablejump_p (insn, NULL, &table))
{
address += get_jump_table_size (table);
insn = table;
}
}
else if (LABEL_P (insn))
address += get_label_padding (insn);
}
fix = minipool_fix_head;
while (fix)
{
Mfix * ftmp;
Mfix * fdel;
Mfix *  last_added_fix;
Mfix * last_barrier = NULL;
Mfix * this_fix;
while (fix && BARRIER_P (fix->insn))
fix = fix->next;
if (fix == NULL)
break;
last_added_fix = NULL;
for (ftmp = fix; ftmp; ftmp = ftmp->next)
{
if (BARRIER_P (ftmp->insn))
{
if (ftmp->address >= minipool_vector_head->max_address)
break;
last_barrier = ftmp;
}
else if ((ftmp->minipool = add_minipool_forward_ref (ftmp)) == NULL)
break;
last_added_fix = ftmp;  
}
if (last_barrier != NULL)
{
for (fdel = last_barrier->next;
fdel && fdel != ftmp;
fdel = fdel->next)
{
fdel->minipool->refcount--;
fdel->minipool = NULL;
}
ftmp = last_barrier;
}
else
{
HOST_WIDE_INT max_address;
gcc_assert (ftmp);
max_address = minipool_vector_head->max_address;
if (ftmp->address < max_address)
max_address = ftmp->address + 1;
last_barrier = create_fix_barrier (last_added_fix, max_address);
}
assign_minipool_offsets (last_barrier);
while (ftmp)
{
if (!BARRIER_P (ftmp->insn)
&& ((ftmp->minipool = add_minipool_backward_ref (ftmp))
== NULL))
break;
ftmp = ftmp->next;
}
for (this_fix = fix; this_fix && ftmp != this_fix;
this_fix = this_fix->next)
if (!BARRIER_P (this_fix->insn))
{
rtx addr
= plus_constant (Pmode,
gen_rtx_LABEL_REF (VOIDmode,
minipool_vector_label),
this_fix->minipool->offset);
*this_fix->loc = gen_rtx_MEM (this_fix->mode, addr);
}
dump_minipool (last_barrier->insn);
fix = ftmp;
}
cfun->machine->after_arm_reorg = 1;
obstack_free (&minipool_obstack, minipool_startobj);
}

static const char *
fp_const_from_val (REAL_VALUE_TYPE *r)
{
if (!fp_consts_inited)
init_fp_table ();
gcc_assert (real_equal (r, &value_fp0));
return "0";
}
void
arm_output_multireg_pop (rtx *operands, bool return_pc, rtx cond, bool reverse,
bool update)
{
int i;
char pattern[100];
int offset;
const char *conditional;
int num_saves = XVECLEN (operands[0], 0);
unsigned int regno;
unsigned int regno_base = REGNO (operands[1]);
bool interrupt_p = IS_INTERRUPT (arm_current_func_type ());
offset = 0;
offset += update ? 1 : 0;
offset += return_pc ? 1 : 0;
for (i = offset; i < num_saves; i++)
{
regno = REGNO (XEXP (XVECEXP (operands[0], 0, i), 0));
gcc_assert ((regno != SP_REGNUM) || (regno_base == SP_REGNUM));
if (regno == regno_base)
gcc_assert (!update);
}
conditional = reverse ? "%?%D0" : "%?%d0";
if ((regno_base == SP_REGNUM) && update && !(interrupt_p && return_pc))
sprintf (pattern, "pop%s\t{", conditional);
else
{
if (regno_base == SP_REGNUM)
sprintf (pattern, "ldmfd%s\t", conditional);
else if (update)
sprintf (pattern, "ldmia%s\t", conditional);
else
sprintf (pattern, "ldm%s\t", conditional);
strcat (pattern, reg_names[regno_base]);
if (update)
strcat (pattern, "!, {");
else
strcat (pattern, ", {");
}
strcat (pattern,
reg_names[REGNO (XEXP (XVECEXP (operands[0], 0, offset), 0))]);
for (i = offset + 1; i < num_saves; i++)
{
strcat (pattern, ", ");
strcat (pattern,
reg_names[REGNO (XEXP (XVECEXP (operands[0], 0, i), 0))]);
}
strcat (pattern, "}");
if (interrupt_p && return_pc)
strcat (pattern, "^");
output_asm_insn (pattern, &cond);
}
const char *
vfp_output_vstmd (rtx * operands)
{
char pattern[100];
int p;
int base;
int i;
rtx addr_reg = REG_P (XEXP (operands[0], 0))
? XEXP (operands[0], 0)
: XEXP (XEXP (operands[0], 0), 0);
bool push_p =  REGNO (addr_reg) == SP_REGNUM;
if (push_p)
strcpy (pattern, "vpush%?.64\t{%P1");
else
strcpy (pattern, "vstmdb%?.64\t%m0!, {%P1");
p = strlen (pattern);
gcc_assert (REG_P (operands[1]));
base = (REGNO (operands[1]) - FIRST_VFP_REGNUM) / 2;
for (i = 1; i < XVECLEN (operands[2], 0); i++)
{
p += sprintf (&pattern[p], ", d%d", base + i);
}
strcpy (&pattern[p], "}");
output_asm_insn (pattern, operands);
return "";
}
static int
vfp_emit_fstmd (int base_reg, int count)
{
rtx par;
rtx dwarf;
rtx tmp, reg;
int i;
if (count == 2 && !arm_arch6)
{
if (base_reg == LAST_VFP_REGNUM - 3)
base_reg -= 2;
count++;
}
if (count > 16)
{
int saved;
saved = vfp_emit_fstmd (base_reg + 32, count - 16);
saved += vfp_emit_fstmd (base_reg, 16);
return saved;
}
par = gen_rtx_PARALLEL (VOIDmode, rtvec_alloc (count));
dwarf = gen_rtx_SEQUENCE (VOIDmode, rtvec_alloc (count + 1));
reg = gen_rtx_REG (DFmode, base_reg);
base_reg += 2;
XVECEXP (par, 0, 0)
= gen_rtx_SET (gen_frame_mem
(BLKmode,
gen_rtx_PRE_MODIFY (Pmode,
stack_pointer_rtx,
plus_constant
(Pmode, stack_pointer_rtx,
- (count * 8)))
),
gen_rtx_UNSPEC (BLKmode,
gen_rtvec (1, reg),
UNSPEC_PUSH_MULT));
tmp = gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx, -(count * 8)));
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (dwarf, 0, 0) = tmp;
tmp = gen_rtx_SET (gen_frame_mem (DFmode, stack_pointer_rtx), reg);
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (dwarf, 0, 1) = tmp;
for (i = 1; i < count; i++)
{
reg = gen_rtx_REG (DFmode, base_reg);
base_reg += 2;
XVECEXP (par, 0, i) = gen_rtx_USE (VOIDmode, reg);
tmp = gen_rtx_SET (gen_frame_mem (DFmode,
plus_constant (Pmode,
stack_pointer_rtx,
i * 8)),
reg);
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (dwarf, 0, i + 1) = tmp;
}
par = emit_insn (par);
add_reg_note (par, REG_FRAME_RELATED_EXPR, dwarf);
RTX_FRAME_RELATED_P (par) = 1;
return count * 8;
}
bool
detect_cmse_nonsecure_call (tree addr)
{
if (!addr)
return FALSE;
tree fntype = TREE_TYPE (addr);
if (use_cmse && lookup_attribute ("cmse_nonsecure_call",
TYPE_ATTRIBUTES (fntype)))
return TRUE;
return FALSE;
}
void
arm_emit_call_insn (rtx pat, rtx addr, bool sibcall)
{
rtx insn;
insn = emit_call_insn (pat);
if (TARGET_VXWORKS_RTP
&& flag_pic
&& !sibcall
&& GET_CODE (addr) == SYMBOL_REF
&& (SYMBOL_REF_DECL (addr)
? !targetm.binds_local_p (SYMBOL_REF_DECL (addr))
: !SYMBOL_REF_LOCAL_P (addr)))
{
require_pic_register ();
use_reg (&CALL_INSN_FUNCTION_USAGE (insn), cfun->machine->pic_reg);
}
if (TARGET_AAPCS_BASED)
{
rtx *fusage = &CALL_INSN_FUNCTION_USAGE (insn);
clobber_reg (fusage, gen_rtx_REG (word_mode, IP_REGNUM));
}
}
const char *
output_call (rtx *operands)
{
gcc_assert (!arm_arch5); 
if (REGNO (operands[0]) == LR_REGNUM)
{
operands[0] = gen_rtx_REG (SImode, IP_REGNUM);
output_asm_insn ("mov%?\t%0, %|lr", operands);
}
output_asm_insn ("mov%?\t%|lr, %|pc", operands);
if (TARGET_INTERWORK || arm_arch4t)
output_asm_insn ("bx%?\t%0", operands);
else
output_asm_insn ("mov%?\t%|pc, %0", operands);
return "";
}
const char *
output_mov_long_double_arm_from_arm (rtx *operands)
{
int dest_start = REGNO (operands[0]);
int src_start = REGNO (operands[1]);
rtx ops[2];
int i;
if (dest_start < src_start)
{
for (i = 0; i < 3; i++)
{
ops[0] = gen_rtx_REG (SImode, dest_start + i);
ops[1] = gen_rtx_REG (SImode, src_start + i);
output_asm_insn ("mov%?\t%0, %1", ops);
}
}
else
{
for (i = 2; i >= 0; i--)
{
ops[0] = gen_rtx_REG (SImode, dest_start + i);
ops[1] = gen_rtx_REG (SImode, src_start + i);
output_asm_insn ("mov%?\t%0, %1", ops);
}
}
return "";
}
void
arm_emit_movpair (rtx dest, rtx src)
{
if (CONST_INT_P (src))
{
HOST_WIDE_INT val = INTVAL (src);
emit_set_insn (dest, GEN_INT (val & 0x0000ffff));
if ((val >> 16) & 0x0000ffff)
{
emit_set_insn (gen_rtx_ZERO_EXTRACT (SImode, dest, GEN_INT (16),
GEN_INT (16)),
GEN_INT ((val >> 16) & 0x0000ffff));
rtx_insn *insn = get_last_insn ();
set_unique_reg_note (insn, REG_EQUAL, copy_rtx (src));
}
return;
}
emit_set_insn (dest, gen_rtx_HIGH (SImode, src));
emit_set_insn (dest, gen_rtx_LO_SUM (SImode, dest, src));
rtx_insn *insn = get_last_insn ();
set_unique_reg_note (insn, REG_EQUAL, copy_rtx (src));
}
const char *
output_move_double (rtx *operands, bool emit, int *count)
{
enum rtx_code code0 = GET_CODE (operands[0]);
enum rtx_code code1 = GET_CODE (operands[1]);
rtx otherops[3];
if (count)
*count = 1;
if (code0 == REG && code1 != MEM)
{
gcc_assert (!emit);
*count = 2;
return "";
}
if (code0 == REG)
{
unsigned int reg0 = REGNO (operands[0]);
otherops[0] = gen_rtx_REG (SImode, 1 + reg0);
gcc_assert (code1 == MEM);  
switch (GET_CODE (XEXP (operands[1], 0)))
{
case REG:
if (emit)
{
if (TARGET_LDRD
&& !(fix_cm3_ldrd && reg0 == REGNO(XEXP (operands[1], 0))))
output_asm_insn ("ldrd%?\t%0, [%m1]", operands);
else
output_asm_insn ("ldmia%?\t%m1, %M0", operands);
}
break;
case PRE_INC:
gcc_assert (TARGET_LDRD);
if (emit)
output_asm_insn ("ldrd%?\t%0, [%m1, #8]!", operands);
break;
case PRE_DEC:
if (emit)
{
if (TARGET_LDRD)
output_asm_insn ("ldrd%?\t%0, [%m1, #-8]!", operands);
else
output_asm_insn ("ldmdb%?\t%m1!, %M0", operands);
}
break;
case POST_INC:
if (emit)
{
if (TARGET_LDRD)
output_asm_insn ("ldrd%?\t%0, [%m1], #8", operands);
else
output_asm_insn ("ldmia%?\t%m1!, %M0", operands);
}
break;
case POST_DEC:
gcc_assert (TARGET_LDRD);
if (emit)
output_asm_insn ("ldrd%?\t%0, [%m1], #-8", operands);
break;
case PRE_MODIFY:
case POST_MODIFY:
otherops[0] = operands[0];
otherops[1] = XEXP (XEXP (XEXP (operands[1], 0), 1), 0);
otherops[2] = XEXP (XEXP (XEXP (operands[1], 0), 1), 1);
if (GET_CODE (XEXP (operands[1], 0)) == PRE_MODIFY)
{
if (reg_overlap_mentioned_p (otherops[0], otherops[2]))
{
if (emit)
{
output_asm_insn ("add%?\t%1, %1, %2", otherops);
output_asm_insn ("ldrd%?\t%0, [%1] @split", otherops);
}
if (count)
*count = 2;
}
else
{
if (TARGET_THUMB2
|| !CONST_INT_P (otherops[2])
|| (INTVAL (otherops[2]) > -256
&& INTVAL (otherops[2]) < 256))
{
if (emit)
output_asm_insn ("ldrd%?\t%0, [%1, %2]!", otherops);
}
else
{
if (emit)
{
output_asm_insn ("ldr%?\t%0, [%1, %2]!", otherops);
output_asm_insn ("ldr%?\t%H0, [%1, #4]", otherops);
}
if (count)
*count = 2;
}
}
}
else
{
if (TARGET_THUMB2
|| !CONST_INT_P (otherops[2])
|| (INTVAL (otherops[2]) > -256
&& INTVAL (otherops[2]) < 256))
{
if (emit)
output_asm_insn ("ldrd%?\t%0, [%1], %2", otherops);
}
else
{
if (emit)
{
output_asm_insn ("ldr%?\t%H0, [%1, #4]", otherops);
output_asm_insn ("ldr%?\t%0, [%1], %2", otherops);
}
if (count)
*count = 2;
}
}
break;
case LABEL_REF:
case CONST:
otherops[1] = operands[1];
if (emit)
output_asm_insn ("adr%?\t%0, %1", otherops);
operands[1] = otherops[0];
if (emit)
{
if (TARGET_LDRD)
output_asm_insn ("ldrd%?\t%0, [%1]", operands);
else
output_asm_insn ("ldmia%?\t%1, %M0", operands);
}
if (count)
*count = 2;
break;
default:
if (arm_add_operand (XEXP (XEXP (operands[1], 0), 1),
GET_MODE (XEXP (XEXP (operands[1], 0), 1))))
{
otherops[0] = operands[0];
otherops[1] = XEXP (XEXP (operands[1], 0), 0);
otherops[2] = XEXP (XEXP (operands[1], 0), 1);
if (GET_CODE (XEXP (operands[1], 0)) == PLUS)
{
if (CONST_INT_P (otherops[2]) && !TARGET_LDRD)
{
switch ((int) INTVAL (otherops[2]))
{
case -8:
if (emit)
output_asm_insn ("ldmdb%?\t%1, %M0", otherops);
return "";
case -4:
if (TARGET_THUMB2)
break;
if (emit)
output_asm_insn ("ldmda%?\t%1, %M0", otherops);
return "";
case 4:
if (TARGET_THUMB2)
break;
if (emit)
output_asm_insn ("ldmib%?\t%1, %M0", otherops);
return "";
}
}
otherops[0] = gen_rtx_REG(SImode, REGNO(operands[0]) + 1);
operands[1] = otherops[0];
if (TARGET_LDRD
&& (REG_P (otherops[2])
|| TARGET_THUMB2
|| (CONST_INT_P (otherops[2])
&& INTVAL (otherops[2]) > -256
&& INTVAL (otherops[2]) < 256)))
{
if (reg_overlap_mentioned_p (operands[0],
otherops[2]))
{
std::swap (otherops[1], otherops[2]);
}
if (reg_overlap_mentioned_p (operands[0], otherops[2])
|| (fix_cm3_ldrd && reg0 == REGNO (otherops[1])))
{
if (emit)
{
output_asm_insn ("add%?\t%0, %1, %2", otherops);
output_asm_insn ("ldrd%?\t%0, [%1]", operands);
}
if (count)
*count = 2;
}
else
{
otherops[0] = operands[0];
if (emit)
output_asm_insn ("ldrd%?\t%0, [%1, %2]", otherops);
}
return "";
}
if (CONST_INT_P (otherops[2]))
{
if (emit)
{
if (!(const_ok_for_arm (INTVAL (otherops[2]))))
output_asm_insn ("sub%?\t%0, %1, #%n2", otherops);
else
output_asm_insn ("add%?\t%0, %1, %2", otherops);
}
}
else
{
if (emit)
output_asm_insn ("add%?\t%0, %1, %2", otherops);
}
}
else
{
if (emit)
output_asm_insn ("sub%?\t%0, %1, %2", otherops);
}
if (count)
*count = 2;
if (TARGET_LDRD)
return "ldrd%?\t%0, [%1]";
return "ldmia%?\t%1, %M0";
}
else
{
otherops[1] = adjust_address (operands[1], SImode, 4);
if (reg_mentioned_p (operands[0], operands[1]))
{
if (emit)
{
output_asm_insn ("ldr%?\t%0, %1", otherops);
output_asm_insn ("ldr%?\t%0, %1", operands);
}
if (count)
*count = 2;
}
else
{
if (emit)
{
output_asm_insn ("ldr%?\t%0, %1", operands);
output_asm_insn ("ldr%?\t%0, %1", otherops);
}
if (count)
*count = 2;
}
}
}
}
else
{
gcc_assert (code0 == MEM && code1 == REG);
gcc_assert ((REGNO (operands[1]) != IP_REGNUM)
|| (TARGET_ARM && TARGET_LDRD));
bool allow_strd = TARGET_LDRD
&& !(TARGET_ARM && (REGNO (operands[1]) & 1) == 1);
switch (GET_CODE (XEXP (operands[0], 0)))
{
case REG:
if (emit)
{
if (allow_strd)
output_asm_insn ("strd%?\t%1, [%m0]", operands);
else
output_asm_insn ("stm%?\t%m0, %M1", operands);
}
break;
case PRE_INC:
gcc_assert (allow_strd);
if (emit)
output_asm_insn ("strd%?\t%1, [%m0, #8]!", operands);
break;
case PRE_DEC:
if (emit)
{
if (allow_strd)
output_asm_insn ("strd%?\t%1, [%m0, #-8]!", operands);
else
output_asm_insn ("stmdb%?\t%m0!, %M1", operands);
}
break;
case POST_INC:
if (emit)
{
if (allow_strd)
output_asm_insn ("strd%?\t%1, [%m0], #8", operands);
else
output_asm_insn ("stm%?\t%m0!, %M1", operands);
}
break;
case POST_DEC:
gcc_assert (allow_strd);
if (emit)
output_asm_insn ("strd%?\t%1, [%m0], #-8", operands);
break;
case PRE_MODIFY:
case POST_MODIFY:
otherops[0] = operands[1];
otherops[1] = XEXP (XEXP (XEXP (operands[0], 0), 1), 0);
otherops[2] = XEXP (XEXP (XEXP (operands[0], 0), 1), 1);
if (!TARGET_THUMB2
&& CONST_INT_P (otherops[2])
&& (INTVAL(otherops[2]) <= -256
|| INTVAL(otherops[2]) >= 256))
{
if (GET_CODE (XEXP (operands[0], 0)) == PRE_MODIFY)
{
if (emit)
{
output_asm_insn ("str%?\t%0, [%1, %2]!", otherops);
output_asm_insn ("str%?\t%H0, [%1, #4]", otherops);
}
if (count)
*count = 2;
}
else
{
if (emit)
{
output_asm_insn ("str%?\t%H0, [%1, #4]", otherops);
output_asm_insn ("str%?\t%0, [%1], %2", otherops);
}
if (count)
*count = 2;
}
}
else if (GET_CODE (XEXP (operands[0], 0)) == PRE_MODIFY)
{
if (emit)
output_asm_insn ("strd%?\t%0, [%1, %2]!", otherops);
}
else
{
if (emit)
output_asm_insn ("strd%?\t%0, [%1], %2", otherops);
}
break;
case PLUS:
otherops[2] = XEXP (XEXP (operands[0], 0), 1);
if (CONST_INT_P (otherops[2]) && !TARGET_LDRD)
{
switch ((int) INTVAL (XEXP (XEXP (operands[0], 0), 1)))
{
case -8:
if (emit)
output_asm_insn ("stmdb%?\t%m0, %M1", operands);
return "";
case -4:
if (TARGET_THUMB2)
break;
if (emit)
output_asm_insn ("stmda%?\t%m0, %M1", operands);
return "";
case 4:
if (TARGET_THUMB2)
break;
if (emit)
output_asm_insn ("stmib%?\t%m0, %M1", operands);
return "";
}
}
if (allow_strd
&& (REG_P (otherops[2])
|| TARGET_THUMB2
|| (CONST_INT_P (otherops[2])
&& INTVAL (otherops[2]) > -256
&& INTVAL (otherops[2]) < 256)))
{
otherops[0] = operands[1];
otherops[1] = XEXP (XEXP (operands[0], 0), 0);
if (emit)
output_asm_insn ("strd%?\t%0, [%1, %2]", otherops);
return "";
}
default:
otherops[0] = adjust_address (operands[0], SImode, 4);
otherops[1] = operands[1];
if (emit)
{
output_asm_insn ("str%?\t%1, %0", operands);
output_asm_insn ("str%?\t%H1, %0", otherops);
}
if (count)
*count = 2;
}
}
return "";
}
const char *
output_move_quad (rtx *operands)
{
if (REG_P (operands[0]))
{
if (MEM_P (operands[1]))
{
switch (GET_CODE (XEXP (operands[1], 0)))
{
case REG:
output_asm_insn ("ldmia%?\t%m1, %M0", operands);
break;
case LABEL_REF:
case CONST:
output_asm_insn ("adr%?\t%0, %1", operands);
output_asm_insn ("ldmia%?\t%0, %M0", operands);
break;
default:
gcc_unreachable ();
}
}
else
{
rtx ops[2];
int dest, src, i;
gcc_assert (REG_P (operands[1]));
dest = REGNO (operands[0]);
src = REGNO (operands[1]);
if (dest < src)
for (i = 0; i < 4; i++)
{
ops[0] = gen_rtx_REG (SImode, dest + i);
ops[1] = gen_rtx_REG (SImode, src + i);
output_asm_insn ("mov%?\t%0, %1", ops);
}
else
for (i = 3; i >= 0; i--)
{
ops[0] = gen_rtx_REG (SImode, dest + i);
ops[1] = gen_rtx_REG (SImode, src + i);
output_asm_insn ("mov%?\t%0, %1", ops);
}
}
}
else
{
gcc_assert (MEM_P (operands[0]));
gcc_assert (REG_P (operands[1]));
gcc_assert (!reg_overlap_mentioned_p (operands[1], operands[0]));
switch (GET_CODE (XEXP (operands[0], 0)))
{
case REG:
output_asm_insn ("stm%?\t%m0, %M1", operands);
break;
default:
gcc_unreachable ();
}
}
return "";
}
const char *
output_move_vfp (rtx *operands)
{
rtx reg, mem, addr, ops[2];
int load = REG_P (operands[0]);
int dp = GET_MODE_SIZE (GET_MODE (operands[0])) == 8;
int sp = (!TARGET_VFP_FP16INST
|| GET_MODE_SIZE (GET_MODE (operands[0])) == 4);
int integer_p = GET_MODE_CLASS (GET_MODE (operands[0])) == MODE_INT;
const char *templ;
char buff[50];
machine_mode mode;
reg = operands[!load];
mem = operands[load];
mode = GET_MODE (reg);
gcc_assert (REG_P (reg));
gcc_assert (IS_VFP_REGNUM (REGNO (reg)));
gcc_assert ((mode == HFmode && TARGET_HARD_FLOAT)
|| mode == SFmode
|| mode == DFmode
|| mode == HImode
|| mode == SImode
|| mode == DImode
|| (TARGET_NEON && VALID_NEON_DREG_MODE (mode)));
gcc_assert (MEM_P (mem));
addr = XEXP (mem, 0);
switch (GET_CODE (addr))
{
case PRE_DEC:
templ = "v%smdb%%?.%s\t%%0!, {%%%s1}%s";
ops[0] = XEXP (addr, 0);
ops[1] = reg;
break;
case POST_INC:
templ = "v%smia%%?.%s\t%%0!, {%%%s1}%s";
ops[0] = XEXP (addr, 0);
ops[1] = reg;
break;
default:
templ = "v%sr%%?.%s\t%%%s0, %%1%s";
ops[0] = reg;
ops[1] = mem;
break;
}
sprintf (buff, templ,
load ? "ld" : "st",
dp ? "64" : sp ? "32" : "16",
dp ? "P" : "",
integer_p ? "\t%@ int" : "");
output_asm_insn (buff, ops);
return "";
}
const char *
output_move_neon (rtx *operands)
{
rtx reg, mem, addr, ops[2];
int regno, nregs, load = REG_P (operands[0]);
const char *templ;
char buff[50];
machine_mode mode;
reg = operands[!load];
mem = operands[load];
mode = GET_MODE (reg);
gcc_assert (REG_P (reg));
regno = REGNO (reg);
nregs = REG_NREGS (reg) / 2;
gcc_assert (VFP_REGNO_OK_FOR_DOUBLE (regno)
|| NEON_REGNO_OK_FOR_QUAD (regno));
gcc_assert (VALID_NEON_DREG_MODE (mode)
|| VALID_NEON_QREG_MODE (mode)
|| VALID_NEON_STRUCT_MODE (mode));
gcc_assert (MEM_P (mem));
addr = XEXP (mem, 0);
if (GET_CODE (addr) == CONST && GET_CODE (XEXP (addr, 0)) == PLUS)
addr = XEXP (addr, 0);
switch (GET_CODE (addr))
{
case POST_INC:
if (nregs > 4)
{
templ = "v%smia%%?\t%%0!, %%h1";
ops[0] = XEXP (addr, 0);
}
else
{
templ = "v%s1.64\t%%h1, %%A0";
ops[0] = mem;
}
ops[1] = reg;
break;
case PRE_DEC:
templ = "v%smdb%%?\t%%0!, %%h1";
ops[0] = XEXP (addr, 0);
ops[1] = reg;
break;
case POST_MODIFY:
gcc_unreachable ();
case REG:
if (nregs > 1)
{
if (nregs > 4)
templ = "v%smia%%?\t%%m0, %%h1";
else
templ = "v%s1.64\t%%h1, %%A0";
ops[0] = mem;
ops[1] = reg;
break;
}
case LABEL_REF:
case PLUS:
{
int i;
int overlap = -1;
for (i = 0; i < nregs; i++)
{
ops[0] = gen_rtx_REG (DImode, REGNO (reg) + 2 * i);
ops[1] = adjust_address (mem, DImode, 8 * i);
if (reg_overlap_mentioned_p (ops[0], mem))
{
gcc_assert (overlap == -1);
overlap = i;
}
else
{
sprintf (buff, "v%sr%%?\t%%P0, %%1", load ? "ld" : "st");
output_asm_insn (buff, ops);
}
}
if (overlap != -1)
{
ops[0] = gen_rtx_REG (DImode, REGNO (reg) + 2 * overlap);
ops[1] = adjust_address (mem, SImode, 8 * overlap);
sprintf (buff, "v%sr%%?\t%%P0, %%1", load ? "ld" : "st");
output_asm_insn (buff, ops);
}
return "";
}
default:
gcc_unreachable ();
}
sprintf (buff, templ, load ? "ld" : "st");
output_asm_insn (buff, ops);
return "";
}
int
arm_attr_length_move_neon (rtx_insn *insn)
{
rtx reg, mem, addr;
int load;
machine_mode mode;
extract_insn_cached (insn);
if (REG_P (recog_data.operand[0]) && REG_P (recog_data.operand[1]))
{
mode = GET_MODE (recog_data.operand[0]);
switch (mode)
{
case E_EImode:
case E_OImode:
return 8;
case E_CImode:
return 12;
case E_XImode:
return 16;
default:
gcc_unreachable ();
}
}
load = REG_P (recog_data.operand[0]);
reg = recog_data.operand[!load];
mem = recog_data.operand[load];
gcc_assert (MEM_P (mem));
addr = XEXP (mem, 0);
if (GET_CODE (addr) == CONST && GET_CODE (XEXP (addr, 0)) == PLUS)
addr = XEXP (addr, 0);
if (GET_CODE (addr) == LABEL_REF || GET_CODE (addr) == PLUS)
{
int insns = REG_NREGS (reg) / 2;
return insns * 4;
}
else
return 4;
}
int
arm_address_offset_is_imm (rtx_insn *insn)
{
rtx mem, addr;
extract_insn_cached (insn);
if (REG_P (recog_data.operand[0]))
return 0;
mem = recog_data.operand[0];
gcc_assert (MEM_P (mem));
addr = XEXP (mem, 0);
if (REG_P (addr)
|| (GET_CODE (addr) == PLUS
&& REG_P (XEXP (addr, 0))
&& CONST_INT_P (XEXP (addr, 1))))
return 1;
else
return 0;
}
const char *
output_add_immediate (rtx *operands)
{
HOST_WIDE_INT n = INTVAL (operands[2]);
if (n != 0 || REGNO (operands[0]) != REGNO (operands[1]))
{
if (n < 0)
output_multi_immediate (operands,
"sub%?\t%0, %1, %2", "sub%?\t%0, %0, %2", 2,
-n);
else
output_multi_immediate (operands,
"add%?\t%0, %1, %2", "add%?\t%0, %0, %2", 2,
n);
}
return "";
}
static const char *
output_multi_immediate (rtx *operands, const char *instr1, const char *instr2,
int immed_op, HOST_WIDE_INT n)
{
#if HOST_BITS_PER_WIDE_INT > 32
n &= 0xffffffff;
#endif
if (n == 0)
{
operands[immed_op] = const0_rtx;
output_asm_insn (instr1, operands);
}
else
{
int i;
const char * instr = instr1;
for (i = 0; i < 32; i += 2)
{
if (n & (3 << i))
{
operands[immed_op] = GEN_INT (n & (255 << i));
output_asm_insn (instr, operands);
instr = instr2;
i += 6;
}
}
}
return "";
}
static const char *
arm_shift_nmem(enum rtx_code code)
{
switch (code)
{
case ASHIFT:
return ARM_LSL_NAME;
case ASHIFTRT:
return "asr";
case LSHIFTRT:
return "lsr";
case ROTATERT:
return "ror";
default:
abort();
}
}
const char *
arithmetic_instr (rtx op, int shift_first_arg)
{
switch (GET_CODE (op))
{
case PLUS:
return "add";
case MINUS:
return shift_first_arg ? "rsb" : "sub";
case IOR:
return "orr";
case XOR:
return "eor";
case AND:
return "and";
case ASHIFT:
case ASHIFTRT:
case LSHIFTRT:
case ROTATERT:
return arm_shift_nmem(GET_CODE(op));
default:
gcc_unreachable ();
}
}
static const char *
shift_op (rtx op, HOST_WIDE_INT *amountp)
{
const char * mnem;
enum rtx_code code = GET_CODE (op);
switch (code)
{
case ROTATE:
if (!CONST_INT_P (XEXP (op, 1)))
{
output_operand_lossage ("invalid shift operand");
return NULL;
}
code = ROTATERT;
*amountp = 32 - INTVAL (XEXP (op, 1));
mnem = "ror";
break;
case ASHIFT:
case ASHIFTRT:
case LSHIFTRT:
case ROTATERT:
mnem = arm_shift_nmem(code);
if (CONST_INT_P (XEXP (op, 1)))
{
*amountp = INTVAL (XEXP (op, 1));
}
else if (REG_P (XEXP (op, 1)))
{
*amountp = -1;
return mnem;
}
else
{
output_operand_lossage ("invalid shift operand");
return NULL;
}
break;
case MULT:
if (!CONST_INT_P (XEXP (op, 1)))
{
output_operand_lossage ("invalid shift operand");
return NULL;
}
*amountp = INTVAL (XEXP (op, 1)) & 0xFFFFFFFF;
if (*amountp & (*amountp - 1))
{
output_operand_lossage ("invalid shift operand");
return NULL;
}
*amountp = exact_log2 (*amountp);
gcc_assert (IN_RANGE (*amountp, 0, 31));
return ARM_LSL_NAME;
default:
output_operand_lossage ("invalid shift operand");
return NULL;
}
if (code == ROTATERT)
*amountp &= 31;
else if (*amountp != (*amountp & 31))
{
if (code == ASHIFT)
mnem = "lsr";
*amountp = 32;
}
if (*amountp == 0)
return NULL;
return mnem;
}
#define MAX_ASCII_LEN 51
void
output_ascii_pseudo_op (FILE *stream, const unsigned char *p, int len)
{
int i;
int len_so_far = 0;
fputs ("\t.ascii\t\"", stream);
for (i = 0; i < len; i++)
{
int c = p[i];
if (len_so_far >= MAX_ASCII_LEN)
{
fputs ("\"\n\t.ascii\t\"", stream);
len_so_far = 0;
}
if (ISPRINT (c))
{
if (c == '\\' || c == '\"')
{
putc ('\\', stream);
len_so_far++;
}
putc (c, stream);
len_so_far++;
}
else
{
fprintf (stream, "\\%03o", c);
len_so_far += 4;
}
}
fputs ("\"\n", stream);
}

#define callee_saved_reg_p(reg) \
(!call_used_regs[reg] \
|| (TARGET_THUMB1 && optimize_size \
&& reg >= FIRST_HI_REGNUM && reg <= LAST_HI_REGNUM))
static unsigned long
arm_compute_save_reg0_reg12_mask (void)
{
unsigned long func_type = arm_current_func_type ();
unsigned long save_reg_mask = 0;
unsigned int reg;
if (IS_INTERRUPT (func_type))
{
unsigned int max_reg;
if (ARM_FUNC_TYPE (func_type) == ARM_FT_FIQ)
max_reg = 7;
else
max_reg = 12;
for (reg = 0; reg <= max_reg; reg++)
if (df_regs_ever_live_p (reg)
|| (! crtl->is_leaf && call_used_regs[reg]))
save_reg_mask |= (1 << reg);
if (flag_pic
&& !TARGET_SINGLE_PIC_BASE
&& arm_pic_register != INVALID_REGNUM
&& crtl->uses_pic_offset_table)
save_reg_mask |= 1 << PIC_OFFSET_TABLE_REGNUM;
}
else if (IS_VOLATILE(func_type))
{
if (frame_pointer_needed)
save_reg_mask |= 1 << HARD_FRAME_POINTER_REGNUM;
if (df_regs_ever_live_p (ARM_HARD_FRAME_POINTER_REGNUM))
save_reg_mask |= 1 << ARM_HARD_FRAME_POINTER_REGNUM;
if (df_regs_ever_live_p (THUMB_HARD_FRAME_POINTER_REGNUM))
save_reg_mask |= 1 << THUMB_HARD_FRAME_POINTER_REGNUM;
}
else
{
for (reg = 0; reg <= 11; reg++)
if (df_regs_ever_live_p (reg) && callee_saved_reg_p (reg))
save_reg_mask |= (1 << reg);
if (frame_pointer_needed)
save_reg_mask |= 1 << HARD_FRAME_POINTER_REGNUM;
if (flag_pic
&& !TARGET_SINGLE_PIC_BASE
&& arm_pic_register != INVALID_REGNUM
&& (df_regs_ever_live_p (PIC_OFFSET_TABLE_REGNUM)
|| crtl->uses_pic_offset_table))
save_reg_mask |= 1 << PIC_OFFSET_TABLE_REGNUM;
if (IS_STACKALIGN (func_type))
save_reg_mask |= 1;
}
if (crtl->calls_eh_return)
{
unsigned int i;
for (i = 0; ; i++)
{
reg = EH_RETURN_DATA_REGNO (i);
if (reg == INVALID_REGNUM)
break;
save_reg_mask |= 1 << reg;
}
}
return save_reg_mask;
}
static bool
arm_r3_live_at_start_p (void)
{
return REGNO_REG_SET_P (df_get_live_out (ENTRY_BLOCK_PTR_FOR_FN (cfun)), 3);
}
static int
arm_compute_static_chain_stack_bytes (void)
{
if (cfun->machine->static_chain_stack_bytes != -1)
return cfun->machine->static_chain_stack_bytes;
if (IS_NESTED (arm_current_func_type ())
&& ((TARGET_APCS_FRAME && frame_pointer_needed && TARGET_ARM)
|| ((flag_stack_check == STATIC_BUILTIN_STACK_CHECK
|| flag_stack_clash_protection)
&& !df_regs_ever_live_p (LR_REGNUM)))
&& arm_r3_live_at_start_p ()
&& crtl->args.pretend_args_size == 0)
return 4;
return 0;
}
static unsigned long
arm_compute_save_core_reg_mask (void)
{
unsigned int save_reg_mask = 0;
unsigned long func_type = arm_current_func_type ();
unsigned int reg;
if (IS_NAKED (func_type))
return 0;
if (TARGET_APCS_FRAME && frame_pointer_needed && TARGET_ARM)
save_reg_mask |=
(1 << ARM_HARD_FRAME_POINTER_REGNUM)
| (1 << IP_REGNUM)
| (1 << LR_REGNUM)
| (1 << PC_REGNUM);
save_reg_mask |= arm_compute_save_reg0_reg12_mask ();
if (df_regs_ever_live_p (LR_REGNUM)
|| (save_reg_mask
&& optimize_size
&& ARM_FUNC_TYPE (func_type) == ARM_FT_NORMAL
&& !crtl->tail_call_emit
&& !crtl->calls_eh_return))
save_reg_mask |= 1 << LR_REGNUM;
if (cfun->machine->lr_save_eliminated)
save_reg_mask &= ~ (1 << LR_REGNUM);
if (TARGET_REALLY_IWMMXT
&& ((bit_count (save_reg_mask)
+ ARM_NUM_INTS (crtl->args.pretend_args_size +
arm_compute_static_chain_stack_bytes())
) % 2) != 0)
{
for (reg = 4; reg <= 12; reg++)
if ((save_reg_mask & (1 << reg)) == 0)
break;
if (reg <= 12)
save_reg_mask |= (1 << reg);
else
{
cfun->machine->sibcall_blocked = 1;
save_reg_mask |= (1 << 3);
}
}
if (TARGET_THUMB2 && IS_NESTED (func_type) && flag_pic
&& (save_reg_mask & THUMB2_WORK_REGS) == 0)
{
reg = thumb_find_work_register (1 << 4);
if (!call_used_regs[reg])
save_reg_mask |= (1 << reg);
}
return save_reg_mask;
}
static unsigned long
thumb1_compute_save_core_reg_mask (void)
{
unsigned long mask;
unsigned reg;
mask = 0;
for (reg = 0; reg < 12; reg ++)
if (df_regs_ever_live_p (reg) && callee_saved_reg_p (reg))
mask |= 1 << reg;
if (frame_pointer_needed)
mask |= 1 << HARD_FRAME_POINTER_REGNUM;
if (flag_pic
&& !TARGET_SINGLE_PIC_BASE
&& arm_pic_register != INVALID_REGNUM
&& crtl->uses_pic_offset_table)
mask |= 1 << PIC_OFFSET_TABLE_REGNUM;
if (!frame_pointer_needed && CALLER_INTERWORKING_SLOT_SIZE > 0)
mask |= 1 << ARM_HARD_FRAME_POINTER_REGNUM;
if (mask & 0xff || thumb_force_lr_save ())
mask |= (1 << LR_REGNUM);
if ((mask & 0xff) == 0
&& ((mask & 0x0f00) || TARGET_BACKTRACE))
{
reg = thumb_find_work_register (1 << LAST_LO_REGNUM);
if (reg * UNITS_PER_WORD <= (unsigned) arm_size_return_regs ())
reg = LAST_LO_REGNUM;
if (callee_saved_reg_p (reg))
mask |= 1 << reg;
}
if ((CALLER_INTERWORKING_SLOT_SIZE +
ROUND_UP_WORD (get_frame_size ()) +
crtl->outgoing_args_size) >= 504)
{
for (reg = LAST_ARG_REGNUM + 1; reg <= LAST_LO_REGNUM; reg++)
if (mask & (1 << reg))
break;
if (reg > LAST_LO_REGNUM)
{
mask |= 1 << LAST_LO_REGNUM;
}
}
return mask;
}
static int
arm_get_vfp_saved_size (void)
{
unsigned int regno;
int count;
int saved;
saved = 0;
if (TARGET_HARD_FLOAT)
{
count = 0;
for (regno = FIRST_VFP_REGNUM;
regno < LAST_VFP_REGNUM;
regno += 2)
{
if ((!df_regs_ever_live_p (regno) || call_used_regs[regno])
&& (!df_regs_ever_live_p (regno + 1) || call_used_regs[regno + 1]))
{
if (count > 0)
{
if (count == 2 && !arm_arch6)
count++;
saved += count * 8;
}
count = 0;
}
else
count++;
}
if (count > 0)
{
if (count == 2 && !arm_arch6)
count++;
saved += count * 8;
}
}
return saved;
}
const char *
output_return_instruction (rtx operand, bool really_return, bool reverse,
bool simple_return)
{
char conditional[10];
char instr[100];
unsigned reg;
unsigned long live_regs_mask;
unsigned long func_type;
arm_stack_offsets *offsets;
func_type = arm_current_func_type ();
if (IS_NAKED (func_type))
return "";
if (IS_VOLATILE (func_type) && TARGET_ABORT_NORETURN)
{
if (really_return)
{
rtx ops[2];
ops[0] = operand;
ops[1] = gen_rtx_SYMBOL_REF (Pmode, NEED_PLT_RELOC ? "abort(PLT)"
: "abort");
assemble_external_libcall (ops[1]);
output_asm_insn (reverse ? "bl%D0\t%a1" : "bl%d0\t%a1", ops);
}
return "";
}
gcc_assert (!cfun->calls_alloca || really_return);
sprintf (conditional, "%%?%%%c0", reverse ? 'D' : 'd');
cfun->machine->return_used_this_function = 1;
offsets = arm_get_frame_offsets ();
live_regs_mask = offsets->saved_regs_mask;
if (!simple_return && live_regs_mask)
{
const char * return_reg;
if (really_return
&& !IS_CMSE_ENTRY (func_type)
&& (IS_INTERRUPT (func_type) || !TARGET_INTERWORK))
return_reg = reg_names[PC_REGNUM];
else
return_reg = reg_names[LR_REGNUM];
if ((live_regs_mask & (1 << IP_REGNUM)) == (1 << IP_REGNUM))
{
if (frame_pointer_needed)
{
live_regs_mask &= ~ (1 << IP_REGNUM);
live_regs_mask |=   (1 << SP_REGNUM);
}
else
gcc_assert (IS_INTERRUPT (func_type) || TARGET_REALLY_IWMMXT);
}
for (reg = 0; reg <= LAST_ARM_REGNUM; reg++)
if (live_regs_mask == (1U << reg))
break;
if (reg <= LAST_ARM_REGNUM
&& (reg != LR_REGNUM
|| ! really_return
|| ! IS_INTERRUPT (func_type)))
{
sprintf (instr, "ldr%s\t%%|%s, [%%|sp], #4", conditional,
(reg == LR_REGNUM) ? return_reg : reg_names[reg]);
}
else
{
char *p;
int first = 1;
if (live_regs_mask & (1 << SP_REGNUM))
{
unsigned HOST_WIDE_INT stack_adjust;
stack_adjust = offsets->outgoing_args - offsets->saved_regs;
gcc_assert (stack_adjust == 0 || stack_adjust == 4);
if (stack_adjust && arm_arch5 && TARGET_ARM)
sprintf (instr, "ldmib%s\t%%|sp, {", conditional);
else
{
if (stack_adjust)
live_regs_mask |= 1 << 3;
sprintf (instr, "ldmfd%s\t%%|sp, {", conditional);
}
}
else if (IS_INTERRUPT (func_type))
sprintf (instr, "ldmfd%s\t%%|sp!, {", conditional);
else
sprintf (instr, "pop%s\t{", conditional);
p = instr + strlen (instr);
for (reg = 0; reg <= SP_REGNUM; reg++)
if (live_regs_mask & (1 << reg))
{
int l = strlen (reg_names[reg]);
if (first)
first = 0;
else
{
memcpy (p, ", ", 2);
p += 2;
}
memcpy (p, "%|", 2);
memcpy (p + 2, reg_names[reg], l);
p += l + 2;
}
if (live_regs_mask & (1 << LR_REGNUM))
{
sprintf (p, "%s%%|%s}", first ? "" : ", ", return_reg);
if (IS_INTERRUPT (func_type))
strcat (p, "^");
}
else
strcpy (p, "}");
}
output_asm_insn (instr, & operand);
if (really_return
&& func_type != ARM_FT_INTERWORKED
&& (live_regs_mask & (1 << LR_REGNUM)) != 0)
{
return "";
}
}
if (really_return)
{
switch ((int) ARM_FUNC_TYPE (func_type))
{
case ARM_FT_ISR:
case ARM_FT_FIQ:
sprintf (instr, "sub%ss\t%%|pc, %%|lr, #4", conditional);
break;
case ARM_FT_INTERWORKED:
gcc_assert (arm_arch5 || arm_arch4t);
sprintf (instr, "bx%s\t%%|lr", conditional);
break;
case ARM_FT_EXCEPTION:
sprintf (instr, "mov%ss\t%%|pc, %%|lr", conditional);
break;
default:
if (IS_CMSE_ENTRY (func_type))
{
if (TARGET_INT_SIMD)
snprintf (instr, sizeof (instr),
"msr%s\tAPSR_nzcvqg, %%|lr", conditional);
else
snprintf (instr, sizeof (instr),
"msr%s\tAPSR_nzcvq, %%|lr", conditional);
output_asm_insn (instr, & operand);
if (TARGET_HARD_FLOAT && !TARGET_THUMB1)
{
snprintf (instr, sizeof (instr), "push\t{%%|r4}");
output_asm_insn (instr, & operand);
snprintf (instr, sizeof (instr), "vmrs\t%%|ip, fpscr");
output_asm_insn (instr, & operand);
snprintf (instr, sizeof (instr), "movw\t%%|r4, #65376");
output_asm_insn (instr, & operand);
snprintf (instr, sizeof (instr), "movt\t%%|r4, #4095");
output_asm_insn (instr, & operand);
snprintf (instr, sizeof (instr), "and\t%%|ip, %%|r4");
output_asm_insn (instr, & operand);
snprintf (instr, sizeof (instr), "vmsr\tfpscr, %%|ip");
output_asm_insn (instr, & operand);
snprintf (instr, sizeof (instr), "pop\t{%%|r4}");
output_asm_insn (instr, & operand);
snprintf (instr, sizeof (instr), "mov\t%%|ip, %%|lr");
output_asm_insn (instr, & operand);
}
snprintf (instr, sizeof (instr), "bxns\t%%|lr");
}
else if (arm_arch5 || arm_arch4t)
sprintf (instr, "bx%s\t%%|lr", conditional);
else
sprintf (instr, "mov%s\t%%|pc, %%|lr", conditional);
break;
}
output_asm_insn (instr, & operand);
}
return "";
}
void
arm_asm_declare_function_name (FILE *file, const char *name, tree decl)
{
size_t cmse_name_len;
char *cmse_name = 0;
char cmse_prefix[] = "__acle_se_";
if (use_cmse && lookup_attribute ("cmse_nonsecure_entry",
DECL_ATTRIBUTES (decl)))
{
cmse_name_len = sizeof (cmse_prefix) + strlen (name);
cmse_name = XALLOCAVEC (char, cmse_name_len);
snprintf (cmse_name, cmse_name_len, "%s%s", cmse_prefix, name);
targetm.asm_out.globalize_label (file, cmse_name);
ARM_DECLARE_FUNCTION_NAME (file, cmse_name, decl);
ASM_OUTPUT_TYPE_DIRECTIVE (file, cmse_name, "function");
}
ARM_DECLARE_FUNCTION_NAME (file, name, decl);
ASM_OUTPUT_TYPE_DIRECTIVE (file, name, "function");
ASM_DECLARE_RESULT (file, DECL_RESULT (decl));
ASM_OUTPUT_LABEL (file, name);
if (cmse_name)
ASM_OUTPUT_LABEL (file, cmse_name);
ARM_OUTPUT_FN_UNWIND (file, TRUE);
}
void
arm_poke_function_name (FILE *stream, const char *name)
{
unsigned long alignlength;
unsigned long length;
rtx           x;
length      = strlen (name) + 1;
alignlength = ROUND_UP_WORD (length);
ASM_OUTPUT_ASCII (stream, name, length);
ASM_OUTPUT_ALIGN (stream, 2);
x = GEN_INT ((unsigned HOST_WIDE_INT) 0xff000000 + alignlength);
assemble_aligned_integer (UNITS_PER_WORD, x);
}
static void
arm_output_function_prologue (FILE *f)
{
unsigned long func_type;
gcc_assert (!arm_ccfsm_state && !arm_target_insn);
func_type = arm_current_func_type ();
switch ((int) ARM_FUNC_TYPE (func_type))
{
default:
case ARM_FT_NORMAL:
break;
case ARM_FT_INTERWORKED:
asm_fprintf (f, "\t%@ Function supports interworking.\n");
break;
case ARM_FT_ISR:
asm_fprintf (f, "\t%@ Interrupt Service Routine.\n");
break;
case ARM_FT_FIQ:
asm_fprintf (f, "\t%@ Fast Interrupt Service Routine.\n");
break;
case ARM_FT_EXCEPTION:
asm_fprintf (f, "\t%@ ARM Exception Handler.\n");
break;
}
if (IS_NAKED (func_type))
asm_fprintf (f, "\t%@ Naked Function: prologue and epilogue provided by programmer.\n");
if (IS_VOLATILE (func_type))
asm_fprintf (f, "\t%@ Volatile: function does not return.\n");
if (IS_NESTED (func_type))
asm_fprintf (f, "\t%@ Nested: function declared inside another function.\n");
if (IS_STACKALIGN (func_type))
asm_fprintf (f, "\t%@ Stack Align: May be called with mis-aligned SP.\n");
if (IS_CMSE_ENTRY (func_type))
asm_fprintf (f, "\t%@ Non-secure entry function: called from non-secure code.\n");
asm_fprintf (f, "\t%@ args = %wd, pretend = %d, frame = %wd\n",
(HOST_WIDE_INT) crtl->args.size,
crtl->args.pretend_args_size,
(HOST_WIDE_INT) get_frame_size ());
asm_fprintf (f, "\t%@ frame_needed = %d, uses_anonymous_args = %d\n",
frame_pointer_needed,
cfun->machine->uses_anonymous_args);
if (cfun->machine->lr_save_eliminated)
asm_fprintf (f, "\t%@ link register save eliminated.\n");
if (crtl->calls_eh_return)
asm_fprintf (f, "\t@ Calls __builtin_eh_return.\n");
}
static void
arm_output_function_epilogue (FILE *)
{
arm_stack_offsets *offsets;
if (TARGET_THUMB1)
{
int regno;
for (regno = 0; regno < LR_REGNUM; regno++)
{
rtx label = cfun->machine->call_via[regno];
if (label != NULL)
{
switch_to_section (function_section (current_function_decl));
targetm.asm_out.internal_label (asm_out_file, "L",
CODE_LABEL_NUMBER (label));
asm_fprintf (asm_out_file, "\tbx\t%r\n", regno);
}
}
cfun->machine->return_used_this_function = 0;
}
else 
{
offsets = arm_get_frame_offsets ();
gcc_assert (!use_return_insn (FALSE, NULL)
|| (cfun->machine->return_used_this_function != 0)
|| offsets->saved_regs == offsets->outgoing_args
|| frame_pointer_needed);
}
}
static void
thumb2_emit_strd_push (unsigned long saved_regs_mask)
{
int num_regs = 0;
int i;
int regno;
rtx par = NULL_RTX;
rtx dwarf = NULL_RTX;
rtx tmp;
bool first = true;
num_regs = bit_count (saved_regs_mask);
gcc_assert (num_regs > 0 && num_regs <= 14);
gcc_assert (!(saved_regs_mask & (1 << SP_REGNUM)));
gcc_assert (!(saved_regs_mask & (1 << PC_REGNUM)));
dwarf = gen_rtx_SEQUENCE (VOIDmode, rtvec_alloc (num_regs + 1));
tmp = gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx, -4 * num_regs));
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (dwarf, 0, 0) = tmp;
for (regno = 0; (saved_regs_mask & (1 << regno)) == 0; regno++)
;
i = 0;
if ((num_regs & 1) != 0)
{
rtx reg, mem, insn;
reg = gen_rtx_REG (SImode, regno);
if (num_regs == 1)
mem = gen_frame_mem (Pmode, gen_rtx_PRE_DEC (Pmode,
stack_pointer_rtx));
else
mem = gen_frame_mem (Pmode,
gen_rtx_PRE_MODIFY
(Pmode, stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx,
-4 * num_regs)));
tmp = gen_rtx_SET (mem, reg);
RTX_FRAME_RELATED_P (tmp) = 1;
insn = emit_insn (tmp);
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_FRAME_RELATED_EXPR, dwarf);
tmp = gen_rtx_SET (gen_frame_mem (Pmode, stack_pointer_rtx), reg);
RTX_FRAME_RELATED_P (tmp) = 1;
i++;
regno++;
XVECEXP (dwarf, 0, i) = tmp;
first = false;
}
while (i < num_regs)
if (saved_regs_mask & (1 << regno))
{
rtx reg1, reg2, mem1, mem2;
rtx tmp0, tmp1, tmp2;
int regno2;
for (regno2 = regno + 1; (saved_regs_mask & (1 << regno2)) == 0;
regno2++)
;
reg1 = gen_rtx_REG (SImode, regno);
reg2 = gen_rtx_REG (SImode, regno2);
if (first)
{
rtx insn;
first = false;
mem1 = gen_frame_mem (Pmode, plus_constant (Pmode,
stack_pointer_rtx,
-4 * num_regs));
mem2 = gen_frame_mem (Pmode, plus_constant (Pmode,
stack_pointer_rtx,
-4 * (num_regs - 1)));
tmp0 = gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx,
-4 * (num_regs)));
tmp1 = gen_rtx_SET (mem1, reg1);
tmp2 = gen_rtx_SET (mem2, reg2);
RTX_FRAME_RELATED_P (tmp0) = 1;
RTX_FRAME_RELATED_P (tmp1) = 1;
RTX_FRAME_RELATED_P (tmp2) = 1;
par = gen_rtx_PARALLEL (VOIDmode, rtvec_alloc (3));
XVECEXP (par, 0, 0) = tmp0;
XVECEXP (par, 0, 1) = tmp1;
XVECEXP (par, 0, 2) = tmp2;
insn = emit_insn (par);
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_FRAME_RELATED_EXPR, dwarf);
}
else
{
mem1 = gen_frame_mem (Pmode, plus_constant (Pmode,
stack_pointer_rtx,
4 * i));
mem2 = gen_frame_mem (Pmode, plus_constant (Pmode,
stack_pointer_rtx,
4 * (i + 1)));
tmp1 = gen_rtx_SET (mem1, reg1);
tmp2 = gen_rtx_SET (mem2, reg2);
RTX_FRAME_RELATED_P (tmp1) = 1;
RTX_FRAME_RELATED_P (tmp2) = 1;
par = gen_rtx_PARALLEL (VOIDmode, rtvec_alloc (2));
XVECEXP (par, 0, 0) = tmp1;
XVECEXP (par, 0, 1) = tmp2;
emit_insn (par);
}
tmp1 = gen_rtx_SET (gen_frame_mem (Pmode,
plus_constant (Pmode,
stack_pointer_rtx,
4 * i)),
reg1);
tmp2 = gen_rtx_SET (gen_frame_mem (Pmode,
plus_constant (Pmode,
stack_pointer_rtx,
4 * (i + 1))),
reg2);
RTX_FRAME_RELATED_P (tmp1) = 1;
RTX_FRAME_RELATED_P (tmp2) = 1;
XVECEXP (dwarf, 0, i + 1) = tmp1;
XVECEXP (dwarf, 0, i + 2) = tmp2;
i += 2;
regno = regno2 + 1;
}
else
regno++;
return;
}
static void
arm_emit_strd_push (unsigned long saved_regs_mask)
{
int num_regs = 0;
int i, j, dwarf_index  = 0;
int offset = 0;
rtx dwarf = NULL_RTX;
rtx insn = NULL_RTX;
rtx tmp, mem;
for (i = 0; i <= LAST_ARM_REGNUM; i++)
if (saved_regs_mask & (1 << i))
num_regs++;
gcc_assert (!(saved_regs_mask & (1 << SP_REGNUM)));
gcc_assert (!(saved_regs_mask & (1 << PC_REGNUM)));
gcc_assert (num_regs > 0);
dwarf = gen_rtx_SEQUENCE (VOIDmode, rtvec_alloc (num_regs + 1));
tmp = gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx, -4 * num_regs));
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (dwarf, 0, dwarf_index++) = tmp;
offset = - 4 * num_regs;
j = 0;
while (j <= LAST_ARM_REGNUM)
if (saved_regs_mask & (1 << j))
{
if ((j % 2 == 0)
&& (saved_regs_mask & (1 << (j + 1))))
{
if (offset < 0)
{
tmp = plus_constant (Pmode, stack_pointer_rtx, offset);
tmp = gen_rtx_PRE_MODIFY (Pmode, stack_pointer_rtx, tmp);
mem = gen_frame_mem (DImode, tmp);
offset = 0;
}
else if (offset > 0)
mem = gen_frame_mem (DImode,
plus_constant (Pmode,
stack_pointer_rtx,
offset));
else
mem = gen_frame_mem (DImode, stack_pointer_rtx);
tmp = gen_rtx_SET (mem, gen_rtx_REG (DImode, j));
RTX_FRAME_RELATED_P (tmp) = 1;
tmp = emit_insn (tmp);
if (dwarf_index == 1)
insn = tmp;
mem = gen_frame_mem (SImode,
plus_constant (Pmode,
stack_pointer_rtx,
offset));
tmp = gen_rtx_SET (mem, gen_rtx_REG (SImode, j));
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (dwarf, 0, dwarf_index++) = tmp;
mem = gen_frame_mem (SImode,
plus_constant (Pmode,
stack_pointer_rtx,
offset + 4));
tmp = gen_rtx_SET (mem, gen_rtx_REG (SImode, j + 1));
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (dwarf, 0, dwarf_index++) = tmp;
offset += 8;
j += 2;
}
else
{
if (offset < 0)
{
tmp = plus_constant (Pmode, stack_pointer_rtx, offset);
tmp = gen_rtx_PRE_MODIFY (Pmode, stack_pointer_rtx, tmp);
mem = gen_frame_mem (SImode, tmp);
offset = 0;
}
else if (offset > 0)
mem = gen_frame_mem (SImode,
plus_constant (Pmode,
stack_pointer_rtx,
offset));
else
mem = gen_frame_mem (SImode, stack_pointer_rtx);
tmp = gen_rtx_SET (mem, gen_rtx_REG (SImode, j));
RTX_FRAME_RELATED_P (tmp) = 1;
tmp = emit_insn (tmp);
if (dwarf_index == 1)
insn = tmp;
mem = gen_frame_mem (SImode,
plus_constant(Pmode,
stack_pointer_rtx,
offset));
tmp = gen_rtx_SET (mem, gen_rtx_REG (SImode, j));
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (dwarf, 0, dwarf_index++) = tmp;
offset += 4;
j += 1;
}
}
else
j++;
gcc_assert (insn != NULL_RTX);
add_reg_note (insn, REG_FRAME_RELATED_EXPR, dwarf);
RTX_FRAME_RELATED_P (insn) = 1;
}
static rtx
emit_multi_reg_push (unsigned long mask, unsigned long dwarf_regs_mask)
{
int num_regs = 0;
int num_dwarf_regs = 0;
int i, j;
rtx par;
rtx dwarf;
int dwarf_par_index;
rtx tmp, reg;
dwarf_regs_mask &= ~(1 << PC_REGNUM);
for (i = 0; i <= LAST_ARM_REGNUM; i++)
{
if (mask & (1 << i))
num_regs++;
if (dwarf_regs_mask & (1 << i))
num_dwarf_regs++;
}
gcc_assert (num_regs && num_regs <= 16);
gcc_assert ((dwarf_regs_mask & ~mask) == 0);
par = gen_rtx_PARALLEL (VOIDmode, rtvec_alloc (num_regs));
dwarf = gen_rtx_SEQUENCE (VOIDmode, rtvec_alloc (num_dwarf_regs + 1));
dwarf_par_index = 1;
for (i = 0; i <= LAST_ARM_REGNUM; i++)
{
if (mask & (1 << i))
{
reg = gen_rtx_REG (SImode, i);
XVECEXP (par, 0, 0)
= gen_rtx_SET (gen_frame_mem
(BLKmode,
gen_rtx_PRE_MODIFY (Pmode,
stack_pointer_rtx,
plus_constant
(Pmode, stack_pointer_rtx,
-4 * num_regs))
),
gen_rtx_UNSPEC (BLKmode,
gen_rtvec (1, reg),
UNSPEC_PUSH_MULT));
if (dwarf_regs_mask & (1 << i))
{
tmp = gen_rtx_SET (gen_frame_mem (SImode, stack_pointer_rtx),
reg);
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (dwarf, 0, dwarf_par_index++) = tmp;
}
break;
}
}
for (j = 1, i++; j < num_regs; i++)
{
if (mask & (1 << i))
{
reg = gen_rtx_REG (SImode, i);
XVECEXP (par, 0, j) = gen_rtx_USE (VOIDmode, reg);
if (dwarf_regs_mask & (1 << i))
{
tmp
= gen_rtx_SET (gen_frame_mem
(SImode,
plus_constant (Pmode, stack_pointer_rtx,
4 * j)),
reg);
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (dwarf, 0, dwarf_par_index++) = tmp;
}
j++;
}
}
par = emit_insn (par);
tmp = gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx, -4 * num_regs));
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (dwarf, 0, 0) = tmp;
add_reg_note (par, REG_FRAME_RELATED_EXPR, dwarf);
return par;
}
static void
arm_add_cfa_adjust_cfa_note (rtx insn, int size, rtx dest, rtx src)
{
rtx dwarf;
RTX_FRAME_RELATED_P (insn) = 1;
dwarf = gen_rtx_SET (dest, plus_constant (Pmode, src, size));
add_reg_note (insn, REG_CFA_ADJUST_CFA, dwarf);
}
static void
arm_emit_multi_reg_pop (unsigned long saved_regs_mask)
{
int num_regs = 0;
int i, j;
rtx par;
rtx dwarf = NULL_RTX;
rtx tmp, reg;
bool return_in_pc = saved_regs_mask & (1 << PC_REGNUM);
int offset_adj;
int emit_update;
offset_adj = return_in_pc ? 1 : 0;
for (i = 0; i <= LAST_ARM_REGNUM; i++)
if (saved_regs_mask & (1 << i))
num_regs++;
gcc_assert (num_regs && num_regs <= 16);
emit_update = (saved_regs_mask & (1 << SP_REGNUM)) ? 0 : 1;
par = gen_rtx_PARALLEL (VOIDmode, rtvec_alloc (num_regs + emit_update + offset_adj));
if (return_in_pc)
XVECEXP (par, 0, 0) = ret_rtx;
if (emit_update)
{
tmp = gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode,
stack_pointer_rtx,
4 * num_regs));
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (par, 0, offset_adj) = tmp;
}
for (j = 0, i = 0; j < num_regs; i++)
if (saved_regs_mask & (1 << i))
{
reg = gen_rtx_REG (SImode, i);
if ((num_regs == 1) && emit_update && !return_in_pc)
{
tmp = gen_frame_mem (SImode,
gen_rtx_POST_INC (Pmode,
stack_pointer_rtx));
tmp = emit_insn (gen_rtx_SET (reg, tmp));
REG_NOTES (tmp) = alloc_reg_note (REG_CFA_RESTORE, reg, dwarf);
return;
}
tmp = gen_rtx_SET (reg,
gen_frame_mem
(SImode,
plus_constant (Pmode, stack_pointer_rtx, 4 * j)));
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (par, 0, j + emit_update + offset_adj) = tmp;
if (i != PC_REGNUM)
dwarf = alloc_reg_note (REG_CFA_RESTORE, reg, dwarf);
j++;
}
if (return_in_pc)
par = emit_jump_insn (par);
else
par = emit_insn (par);
REG_NOTES (par) = dwarf;
if (!return_in_pc)
arm_add_cfa_adjust_cfa_note (par, UNITS_PER_WORD * num_regs,
stack_pointer_rtx, stack_pointer_rtx);
}
static void
arm_emit_vfp_multi_reg_pop (int first_reg, int num_regs, rtx base_reg)
{
int i, j;
rtx par;
rtx dwarf = NULL_RTX;
rtx tmp, reg;
gcc_assert (num_regs && num_regs <= 32);
if (num_regs == 2 && !arm_arch6)
{
if (first_reg == 15)
first_reg--;
num_regs++;
}
if (num_regs > 16)
{
arm_emit_vfp_multi_reg_pop (first_reg, 16, base_reg);
arm_emit_vfp_multi_reg_pop (first_reg + 16, num_regs - 16, base_reg);
return;
}
par = gen_rtx_PARALLEL (VOIDmode, rtvec_alloc (num_regs + 1));
tmp = gen_rtx_SET (base_reg, plus_constant (Pmode, base_reg, 8 * num_regs));
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (par, 0, 0) = tmp;
for (j = 0, i=first_reg; j < num_regs; i += 2)
{
reg = gen_rtx_REG (DFmode, i);
tmp = gen_rtx_SET (reg,
gen_frame_mem
(DFmode,
plus_constant (Pmode, base_reg, 8 * j)));
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (par, 0, j + 1) = tmp;
dwarf = alloc_reg_note (REG_CFA_RESTORE, reg, dwarf);
j++;
}
par = emit_insn (par);
REG_NOTES (par) = dwarf;
if (REGNO (base_reg) == IP_REGNUM)
{
RTX_FRAME_RELATED_P (par) = 1;
add_reg_note (par, REG_CFA_DEF_CFA, hard_frame_pointer_rtx);
}
else
arm_add_cfa_adjust_cfa_note (par, 2 * UNITS_PER_WORD * num_regs,
base_reg, base_reg);
}
static void
thumb2_emit_ldrd_pop (unsigned long saved_regs_mask)
{
int num_regs = 0;
int i, j;
rtx par = NULL_RTX;
rtx dwarf = NULL_RTX;
rtx tmp, reg, tmp1;
bool return_in_pc = saved_regs_mask & (1 << PC_REGNUM);
for (i = 0; i <= LAST_ARM_REGNUM; i++)
if (saved_regs_mask & (1 << i))
num_regs++;
gcc_assert (num_regs && num_regs <= 16);
if (return_in_pc)
num_regs--;
gcc_assert (!(saved_regs_mask & (1 << SP_REGNUM)));
for (i = 0, j = 0; i < (num_regs - (num_regs % 2)); j++)
if (saved_regs_mask & (1 << j))
{
reg = gen_rtx_REG (SImode, j);
tmp = gen_rtx_SET (reg,
gen_frame_mem (SImode,
plus_constant (Pmode,
stack_pointer_rtx, 4 * i)));
RTX_FRAME_RELATED_P (tmp) = 1;
if (i % 2 == 0)
{
par = gen_rtx_PARALLEL (VOIDmode, rtvec_alloc (2));
dwarf = NULL_RTX;
}
XVECEXP (par, 0, (i % 2)) = tmp;
dwarf = alloc_reg_note (REG_CFA_RESTORE, reg, dwarf);
if ((i % 2) == 1)
{
par = emit_insn (par);
REG_NOTES (par) = dwarf;
RTX_FRAME_RELATED_P (par) = 1;
}
i++;
}
tmp = gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx, 4 * i));
RTX_FRAME_RELATED_P (tmp) = 1;
tmp = emit_insn (tmp);
if (!return_in_pc)
{
arm_add_cfa_adjust_cfa_note (tmp, UNITS_PER_WORD * i,
stack_pointer_rtx, stack_pointer_rtx);
}
dwarf = NULL_RTX;
if (((num_regs % 2) == 1 && !return_in_pc)
|| ((num_regs % 2) == 0 && return_in_pc))
{
for (; (saved_regs_mask & (1 << j)) == 0; j++);
tmp1 = gen_rtx_MEM (SImode,
gen_rtx_POST_INC (SImode,
stack_pointer_rtx));
set_mem_alias_set (tmp1, get_frame_alias_set ());
reg = gen_rtx_REG (SImode, j);
tmp = gen_rtx_SET (reg, tmp1);
RTX_FRAME_RELATED_P (tmp) = 1;
dwarf = alloc_reg_note (REG_CFA_RESTORE, reg, dwarf);
if (return_in_pc)
{
gcc_assert (j == PC_REGNUM);
par = gen_rtx_PARALLEL (VOIDmode, rtvec_alloc (2));
XVECEXP (par, 0, 0) = ret_rtx;
XVECEXP (par, 0, 1) = tmp;
par = emit_jump_insn (par);
}
else
{
par = emit_insn (tmp);
REG_NOTES (par) = dwarf;
arm_add_cfa_adjust_cfa_note (par, UNITS_PER_WORD,
stack_pointer_rtx, stack_pointer_rtx);
}
}
else if ((num_regs % 2) == 1 && return_in_pc)
{
arm_emit_multi_reg_pop (saved_regs_mask & (~((1 << j) - 1)));
}
return;
}
static void
arm_emit_ldrd_pop (unsigned long saved_regs_mask)
{
int j = 0;
int offset = 0;
rtx par = NULL_RTX;
rtx dwarf = NULL_RTX;
rtx tmp, mem;
gcc_assert (!((saved_regs_mask & (1 << SP_REGNUM))));
j = 0;
while (j <= LAST_ARM_REGNUM)
if (saved_regs_mask & (1 << j))
{
if ((j % 2) == 0
&& (saved_regs_mask & (1 << (j + 1)))
&& (j + 1) != PC_REGNUM)
{
if (offset > 0)
mem = gen_frame_mem (DImode,
plus_constant (Pmode,
stack_pointer_rtx,
offset));
else
mem = gen_frame_mem (DImode, stack_pointer_rtx);
tmp = gen_rtx_SET (gen_rtx_REG (DImode, j), mem);
tmp = emit_insn (tmp);
RTX_FRAME_RELATED_P (tmp) = 1;
dwarf = alloc_reg_note (REG_CFA_RESTORE,
gen_rtx_REG (SImode, j),
NULL_RTX);
dwarf = alloc_reg_note (REG_CFA_RESTORE,
gen_rtx_REG (SImode, j + 1),
dwarf);
REG_NOTES (tmp) = dwarf;
offset += 8;
j += 2;
}
else if (j != PC_REGNUM)
{
if (offset > 0)
mem = gen_frame_mem (SImode,
plus_constant (Pmode,
stack_pointer_rtx,
offset));
else
mem = gen_frame_mem (SImode, stack_pointer_rtx);
tmp = gen_rtx_SET (gen_rtx_REG (SImode, j), mem);
tmp = emit_insn (tmp);
RTX_FRAME_RELATED_P (tmp) = 1;
REG_NOTES (tmp) = alloc_reg_note (REG_CFA_RESTORE,
gen_rtx_REG (SImode, j),
NULL_RTX);
offset += 4;
j += 1;
}
else 
j++;
}
else
j++;
if (offset > 0)
{
tmp = gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode,
stack_pointer_rtx,
offset));
tmp = emit_insn (tmp);
arm_add_cfa_adjust_cfa_note (tmp, offset,
stack_pointer_rtx, stack_pointer_rtx);
offset = 0;
}
if (saved_regs_mask & (1 << PC_REGNUM))
{
par = gen_rtx_PARALLEL (VOIDmode, rtvec_alloc (2));
XVECEXP (par, 0, 0) = ret_rtx;
tmp = gen_rtx_SET (gen_rtx_REG (SImode, PC_REGNUM),
gen_frame_mem (SImode,
gen_rtx_POST_INC (SImode,
stack_pointer_rtx)));
RTX_FRAME_RELATED_P (tmp) = 1;
XVECEXP (par, 0, 1) = tmp;
par = emit_jump_insn (par);
dwarf = alloc_reg_note (REG_CFA_RESTORE,
gen_rtx_REG (SImode, PC_REGNUM),
NULL_RTX);
REG_NOTES (par) = dwarf;
arm_add_cfa_adjust_cfa_note (par, UNITS_PER_WORD,
stack_pointer_rtx, stack_pointer_rtx);
}
}
static unsigned
arm_size_return_regs (void)
{
machine_mode mode;
if (crtl->return_rtx != 0)
mode = GET_MODE (crtl->return_rtx);
else
mode = DECL_MODE (DECL_RESULT (current_function_decl));
return GET_MODE_SIZE (mode);
}
static bool
thumb_force_lr_save (void)
{
return !cfun->machine->lr_save_eliminated
&& (!crtl->is_leaf
|| thumb_far_jump_used_p ()
|| df_regs_ever_live_p (LR_REGNUM));
}
static bool
is_indirect_tailcall_p (rtx call)
{
rtx pat = PATTERN (call);
pat = XVECEXP (pat, 0, 0);
if (GET_CODE (pat) == SET)
pat = SET_SRC (pat);
pat = XEXP (XEXP (pat, 0), 0);
return REG_P (pat);
}
static bool
any_sibcall_could_use_r3 (void)
{
edge_iterator ei;
edge e;
if (!crtl->tail_call_emit)
return false;
FOR_EACH_EDGE (e, ei, EXIT_BLOCK_PTR_FOR_FN (cfun)->preds)
if (e->flags & EDGE_SIBCALL)
{
rtx_insn *call = BB_END (e->src);
if (!CALL_P (call))
call = prev_nonnote_nondebug_insn (call);
gcc_assert (CALL_P (call) && SIBLING_CALL_P (call));
if (find_regno_fusage (call, USE, 3)
|| is_indirect_tailcall_p (call))
return true;
}
return false;
}
static arm_stack_offsets *
arm_get_frame_offsets (void)
{
struct arm_stack_offsets *offsets;
offsets = &cfun->machine->stack_offsets;
return offsets;
}
static void
arm_compute_frame_layout (void)
{
struct arm_stack_offsets *offsets;
unsigned long func_type;
int saved;
int core_saved;
HOST_WIDE_INT frame_size;
int i;
offsets = &cfun->machine->stack_offsets;
frame_size = ROUND_UP_WORD (get_frame_size ());
offsets->saved_args = crtl->args.pretend_args_size;
offsets->frame
= (offsets->saved_args
+ arm_compute_static_chain_stack_bytes ()
+ (frame_pointer_needed ? 4 : 0));
if (TARGET_32BIT)
{
unsigned int regno;
offsets->saved_regs_mask = arm_compute_save_core_reg_mask ();
core_saved = bit_count (offsets->saved_regs_mask) * 4;
saved = core_saved;
if (TARGET_REALLY_IWMMXT)
{
for (regno = FIRST_IWMMXT_REGNUM;
regno <= LAST_IWMMXT_REGNUM;
regno++)
if (df_regs_ever_live_p (regno) && ! call_used_regs[regno])
saved += 8;
}
func_type = arm_current_func_type ();
if (! IS_VOLATILE (func_type)
&& TARGET_HARD_FLOAT)
saved += arm_get_vfp_saved_size ();
}
else 
{
offsets->saved_regs_mask = thumb1_compute_save_core_reg_mask ();
core_saved = bit_count (offsets->saved_regs_mask) * 4;
saved = core_saved;
if (TARGET_BACKTRACE)
saved += 16;
}
offsets->saved_regs
= offsets->saved_args + arm_compute_static_chain_stack_bytes () + saved;
offsets->soft_frame = offsets->saved_regs + CALLER_INTERWORKING_SLOT_SIZE;
if (crtl->is_leaf && frame_size == 0
&& ! cfun->calls_alloca)
{
offsets->outgoing_args = offsets->soft_frame;
offsets->locals_base = offsets->soft_frame;
return;
}
if (ARM_DOUBLEWORD_ALIGN
&& (offsets->soft_frame & 7))
{
offsets->soft_frame += 4;
if (frame_size + crtl->outgoing_args_size == 0)
{
int reg = -1;
bool prefer_callee_reg_p = false;
if (! any_sibcall_could_use_r3 ()
&& arm_size_return_regs () <= 12
&& (offsets->saved_regs_mask & (1 << 3)) == 0
&& (TARGET_THUMB2
|| !(TARGET_LDRD && current_tune->prefer_ldrd_strd)))
{
reg = 3;
if (!TARGET_THUMB2)
prefer_callee_reg_p = true;
}
if (reg == -1
|| prefer_callee_reg_p)
{
for (i = 4; i <= (TARGET_THUMB1 ? LAST_LO_REGNUM : 11); i++)
{
if (!fixed_regs[i]
&& (offsets->saved_regs_mask & (1 << i)) == 0)
{
reg = i;
break;
}
}
}
if (reg != -1)
{
offsets->saved_regs += 4;
offsets->saved_regs_mask |= (1 << reg);
}
}
}
offsets->locals_base = offsets->soft_frame + frame_size;
offsets->outgoing_args = (offsets->locals_base
+ crtl->outgoing_args_size);
if (ARM_DOUBLEWORD_ALIGN)
{
if (offsets->outgoing_args & 7)
offsets->outgoing_args += 4;
gcc_assert (!(offsets->outgoing_args & 7));
}
}
HOST_WIDE_INT
arm_compute_initial_elimination_offset (unsigned int from, unsigned int to)
{
arm_stack_offsets *offsets;
offsets = arm_get_frame_offsets ();
switch (from)
{
case ARG_POINTER_REGNUM:
switch (to)
{
case THUMB_HARD_FRAME_POINTER_REGNUM:
return 0;
case FRAME_POINTER_REGNUM:
return offsets->soft_frame - offsets->saved_args;
case ARM_HARD_FRAME_POINTER_REGNUM:
return offsets->frame - offsets->saved_args - 4;
case STACK_POINTER_REGNUM:
return offsets->outgoing_args - (offsets->saved_args + 4);
default:
gcc_unreachable ();
}
gcc_unreachable ();
case FRAME_POINTER_REGNUM:
switch (to)
{
case THUMB_HARD_FRAME_POINTER_REGNUM:
return 0;
case ARM_HARD_FRAME_POINTER_REGNUM:
return offsets->frame - offsets->soft_frame;
case STACK_POINTER_REGNUM:
return offsets->outgoing_args - offsets->soft_frame;
default:
gcc_unreachable ();
}
gcc_unreachable ();
default:
gcc_unreachable ();
}
}
bool
arm_can_eliminate (const int from, const int to)
{
return ((to == FRAME_POINTER_REGNUM && from == ARG_POINTER_REGNUM) ? false :
(to == STACK_POINTER_REGNUM && frame_pointer_needed) ? false :
(to == ARM_HARD_FRAME_POINTER_REGNUM && TARGET_THUMB) ? false :
(to == THUMB_HARD_FRAME_POINTER_REGNUM && TARGET_ARM) ? false :
true);
}
static int
arm_save_coproc_regs(void)
{
int saved_size = 0;
unsigned reg;
unsigned start_reg;
rtx insn;
for (reg = LAST_IWMMXT_REGNUM; reg >= FIRST_IWMMXT_REGNUM; reg--)
if (df_regs_ever_live_p (reg) && ! call_used_regs[reg])
{
insn = gen_rtx_PRE_DEC (Pmode, stack_pointer_rtx);
insn = gen_rtx_MEM (V2SImode, insn);
insn = emit_set_insn (insn, gen_rtx_REG (V2SImode, reg));
RTX_FRAME_RELATED_P (insn) = 1;
saved_size += 8;
}
if (TARGET_HARD_FLOAT)
{
start_reg = FIRST_VFP_REGNUM;
for (reg = FIRST_VFP_REGNUM; reg < LAST_VFP_REGNUM; reg += 2)
{
if ((!df_regs_ever_live_p (reg) || call_used_regs[reg])
&& (!df_regs_ever_live_p (reg + 1) || call_used_regs[reg + 1]))
{
if (start_reg != reg)
saved_size += vfp_emit_fstmd (start_reg,
(reg - start_reg) / 2);
start_reg = reg + 2;
}
}
if (start_reg != reg)
saved_size += vfp_emit_fstmd (start_reg,
(reg - start_reg) / 2);
}
return saved_size;
}
static void
thumb_set_frame_pointer (arm_stack_offsets *offsets)
{
HOST_WIDE_INT amount;
rtx insn, dwarf;
amount = offsets->outgoing_args - offsets->locals_base;
if (amount < 1024)
insn = emit_insn (gen_addsi3 (hard_frame_pointer_rtx,
stack_pointer_rtx, GEN_INT (amount)));
else
{
emit_insn (gen_movsi (hard_frame_pointer_rtx, GEN_INT (amount)));
if (TARGET_THUMB2)
{
insn = emit_insn (gen_addsi3 (hard_frame_pointer_rtx,
stack_pointer_rtx,
hard_frame_pointer_rtx));
}
else
{
insn = emit_insn (gen_addsi3 (hard_frame_pointer_rtx,
hard_frame_pointer_rtx,
stack_pointer_rtx));
}
dwarf = gen_rtx_SET (hard_frame_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx, amount));
RTX_FRAME_RELATED_P (dwarf) = 1;
add_reg_note (insn, REG_FRAME_RELATED_EXPR, dwarf);
}
RTX_FRAME_RELATED_P (insn) = 1;
}
struct scratch_reg {
rtx reg;
bool saved;
};
static void
get_scratch_register_on_entry (struct scratch_reg *sr, unsigned int regno1,
unsigned long live_regs)
{
int regno = -1;
sr->saved = false;
if (regno1 != LR_REGNUM && (live_regs & (1 << LR_REGNUM)) != 0)
regno = LR_REGNUM;
else
{
unsigned int i;
for (i = 4; i < 11; i++)
if (regno1 != i && (live_regs & (1 << i)) != 0)
{
regno = i;
break;
}
if (regno < 0)
{
if (regno1 == IP_REGNUM && IS_NESTED (arm_current_func_type ()))
regno1 = 3;
regno = (regno1 == 3 ? 2 : 3);
sr->saved
= REGNO_REG_SET_P (df_get_live_out (ENTRY_BLOCK_PTR_FOR_FN (cfun)),
regno);
}
}
sr->reg = gen_rtx_REG (SImode, regno);
if (sr->saved)
{
rtx addr = gen_rtx_PRE_DEC (Pmode, stack_pointer_rtx);
rtx insn = emit_set_insn (gen_frame_mem (SImode, addr), sr->reg);
rtx x = gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx, -4));
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_FRAME_RELATED_EXPR, x);
}
}
static void
release_scratch_register_on_entry (struct scratch_reg *sr)
{
if (sr->saved)
{
rtx addr = gen_rtx_POST_INC (Pmode, stack_pointer_rtx);
rtx insn = emit_set_insn (sr->reg, gen_frame_mem (SImode, addr));
rtx x = gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx, 4));
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_FRAME_RELATED_EXPR, x);
}
}
#define PROBE_INTERVAL (1 << STACK_CHECK_PROBE_INTERVAL_EXP)
#if PROBE_INTERVAL > 4096
#error Cannot use indexed addressing mode for stack probing
#endif
static void
arm_emit_probe_stack_range (HOST_WIDE_INT first, HOST_WIDE_INT size,
unsigned int regno1, unsigned long live_regs)
{
rtx reg1 = gen_rtx_REG (Pmode, regno1);
if (size <= PROBE_INTERVAL)
{
emit_move_insn (reg1, GEN_INT (first + PROBE_INTERVAL));
emit_set_insn (reg1, gen_rtx_MINUS (Pmode, stack_pointer_rtx, reg1));
emit_stack_probe (plus_constant (Pmode, reg1, PROBE_INTERVAL - size));
}
else if (size <= 5 * PROBE_INTERVAL)
{
HOST_WIDE_INT i, rem;
emit_move_insn (reg1, GEN_INT (first + PROBE_INTERVAL));
emit_set_insn (reg1, gen_rtx_MINUS (Pmode, stack_pointer_rtx, reg1));
emit_stack_probe (reg1);
for (i = 2 * PROBE_INTERVAL; i < size; i += PROBE_INTERVAL)
{
emit_set_insn (reg1, plus_constant (Pmode, reg1, -PROBE_INTERVAL));
emit_stack_probe (reg1);
}
rem = size - (i - PROBE_INTERVAL);
if (rem > 4095 || (TARGET_THUMB2 && rem > 255))
{
emit_set_insn (reg1, plus_constant (Pmode, reg1, -PROBE_INTERVAL));
emit_stack_probe (plus_constant (Pmode, reg1, PROBE_INTERVAL - rem));
}
else
emit_stack_probe (plus_constant (Pmode, reg1, -rem));
}
else
{
HOST_WIDE_INT rounded_size;
struct scratch_reg sr;
get_scratch_register_on_entry (&sr, regno1, live_regs);
emit_move_insn (reg1, GEN_INT (first));
rounded_size = size & -PROBE_INTERVAL;
emit_move_insn (sr.reg, GEN_INT (rounded_size));
emit_set_insn (reg1, gen_rtx_MINUS (Pmode, stack_pointer_rtx, reg1));
emit_set_insn (sr.reg, gen_rtx_MINUS (Pmode, reg1, sr.reg));
emit_insn (gen_probe_stack_range (reg1, reg1, sr.reg));
if (size != rounded_size)
{
HOST_WIDE_INT rem = size - rounded_size;
if (rem > 4095 || (TARGET_THUMB2 && rem > 255))
{
emit_set_insn (sr.reg,
plus_constant (Pmode, sr.reg, -PROBE_INTERVAL));
emit_stack_probe (plus_constant (Pmode, sr.reg,
PROBE_INTERVAL - rem));
}
else
emit_stack_probe (plus_constant (Pmode, sr.reg, -rem));
}
release_scratch_register_on_entry (&sr);
}
emit_insn (gen_blockage ());
}
const char *
output_probe_stack_range (rtx reg1, rtx reg2)
{
static int labelno = 0;
char loop_lab[32];
rtx xops[2];
ASM_GENERATE_INTERNAL_LABEL (loop_lab, "LPSRL", labelno++);
ASM_OUTPUT_INTERNAL_LABEL (asm_out_file, loop_lab);
xops[0] = reg1;
xops[1] = GEN_INT (PROBE_INTERVAL);
output_asm_insn ("sub\t%0, %0, %1", xops);
output_asm_insn ("str\tr0, [%0, #0]", xops);
xops[1] = reg2;
output_asm_insn ("cmp\t%0, %1", xops);
fputs ("\tbne\t", asm_out_file);
assemble_name_raw (asm_out_file, loop_lab);
fputc ('\n', asm_out_file);
return "";
}
void
arm_expand_prologue (void)
{
rtx amount;
rtx insn;
rtx ip_rtx;
unsigned long live_regs_mask;
unsigned long func_type;
int fp_offset = 0;
int saved_pretend_args = 0;
int saved_regs = 0;
unsigned HOST_WIDE_INT args_to_push;
HOST_WIDE_INT size;
arm_stack_offsets *offsets;
bool clobber_ip;
func_type = arm_current_func_type ();
if (IS_NAKED (func_type))
{
if (flag_stack_usage_info)
current_function_static_stack_size = 0;
return;
}
args_to_push = crtl->args.pretend_args_size;
offsets = arm_get_frame_offsets ();
live_regs_mask = offsets->saved_regs_mask;
ip_rtx = gen_rtx_REG (SImode, IP_REGNUM);
if (IS_STACKALIGN (func_type))
{
rtx r0, r1;
gcc_assert (TARGET_THUMB2 && !arm_arch_notm && args_to_push == 0);
r0 = gen_rtx_REG (SImode, R0_REGNUM);
r1 = gen_rtx_REG (SImode, R1_REGNUM);
insn = emit_insn (gen_movsi (r0, stack_pointer_rtx));
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_CFA_REGISTER, NULL);
emit_insn (gen_andsi3 (r1, r0, GEN_INT (~(HOST_WIDE_INT)7)));
emit_insn (gen_movsi (stack_pointer_rtx, r1));
}
cfun->machine->static_chain_stack_bytes
= arm_compute_static_chain_stack_bytes ();
clobber_ip = IS_NESTED (func_type)
&& ((TARGET_APCS_FRAME && frame_pointer_needed && TARGET_ARM)
|| ((flag_stack_check == STATIC_BUILTIN_STACK_CHECK
|| flag_stack_clash_protection)
&& !df_regs_ever_live_p (LR_REGNUM)
&& arm_r3_live_at_start_p ()));
if (clobber_ip)
{
if (!arm_r3_live_at_start_p ())
insn = emit_set_insn (gen_rtx_REG (SImode, 3), ip_rtx);
else if (args_to_push == 0)
{
rtx addr, dwarf;
gcc_assert(arm_compute_static_chain_stack_bytes() == 4);
saved_regs += 4;
addr = gen_rtx_PRE_DEC (Pmode, stack_pointer_rtx);
insn = emit_set_insn (gen_frame_mem (SImode, addr), ip_rtx);
fp_offset = 4;
dwarf = gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx,
-fp_offset));
RTX_FRAME_RELATED_P (insn) = 1;
add_reg_note (insn, REG_FRAME_RELATED_EXPR, dwarf);
}
else
{
if (cfun->machine->uses_anonymous_args)
{
insn = emit_multi_reg_push ((0xf0 >> (args_to_push / 4)) & 0xf,
(0xf0 >> (args_to_push / 4)) & 0xf);
emit_set_insn (gen_rtx_REG (SImode, 3), ip_rtx);
saved_pretend_args = 1;
}
else
{
rtx addr, dwarf;
if (args_to_push == 4)
addr = gen_rtx_PRE_DEC (Pmode, stack_pointer_rtx);
else
addr = gen_rtx_PRE_MODIFY (Pmode, stack_pointer_rtx,
plus_constant (Pmode,
stack_pointer_rtx,
-args_to_push));
insn = emit_set_insn (gen_frame_mem (SImode, addr), ip_rtx);
dwarf = gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx,
-args_to_push));
add_reg_note (insn, REG_FRAME_RELATED_EXPR, dwarf);
}
RTX_FRAME_RELATED_P (insn) = 1;
fp_offset = args_to_push;
args_to_push = 0;
}
}
if (TARGET_APCS_FRAME && frame_pointer_needed && TARGET_ARM)
{
if (IS_INTERRUPT (func_type))
{
emit_multi_reg_push (1 << IP_REGNUM, 1 << IP_REGNUM);
}
insn = emit_set_insn (ip_rtx,
plus_constant (Pmode, stack_pointer_rtx,
fp_offset));
RTX_FRAME_RELATED_P (insn) = 1;
}
if (args_to_push)
{
if (cfun->machine->uses_anonymous_args)
insn = emit_multi_reg_push
((0xf0 >> (args_to_push / 4)) & 0xf,
(0xf0 >> (args_to_push / 4)) & 0xf);
else
insn = emit_insn
(gen_addsi3 (stack_pointer_rtx, stack_pointer_rtx,
GEN_INT (- args_to_push)));
RTX_FRAME_RELATED_P (insn) = 1;
}
if ((func_type == ARM_FT_ISR || func_type == ARM_FT_FIQ)
&& (live_regs_mask & (1 << LR_REGNUM)) != 0
&& !(frame_pointer_needed && TARGET_APCS_FRAME)
&& TARGET_ARM)
{
rtx lr = gen_rtx_REG (SImode, LR_REGNUM);
emit_set_insn (lr, plus_constant (SImode, lr, -4));
}
if (live_regs_mask)
{
unsigned long dwarf_regs_mask = live_regs_mask;
saved_regs += bit_count (live_regs_mask) * 4;
if (optimize_size && !frame_pointer_needed
&& saved_regs == offsets->saved_regs - offsets->saved_args)
{
int n;
int frame;
n = 0;
while (n < 8 && (live_regs_mask & (1 << n)) == 0)
n++;
frame = offsets->outgoing_args - (offsets->saved_args + saved_regs);
if (frame && n * 4 >= frame)
{
n = frame / 4;
live_regs_mask |= (1 << n) - 1;
saved_regs += frame;
}
}
if (TARGET_LDRD
&& current_tune->prefer_ldrd_strd
&& !optimize_function_for_size_p (cfun))
{
gcc_checking_assert (live_regs_mask == dwarf_regs_mask);
if (TARGET_THUMB2)
thumb2_emit_strd_push (live_regs_mask);
else if (TARGET_ARM
&& !TARGET_APCS_FRAME
&& !IS_INTERRUPT (func_type))
arm_emit_strd_push (live_regs_mask);
else
{
insn = emit_multi_reg_push (live_regs_mask, live_regs_mask);
RTX_FRAME_RELATED_P (insn) = 1;
}
}
else
{
insn = emit_multi_reg_push (live_regs_mask, dwarf_regs_mask);
RTX_FRAME_RELATED_P (insn) = 1;
}
}
if (! IS_VOLATILE (func_type))
saved_regs += arm_save_coproc_regs ();
if (frame_pointer_needed && TARGET_ARM)
{
if (TARGET_APCS_FRAME)
{
insn = GEN_INT (-(4 + args_to_push + fp_offset));
insn = emit_insn (gen_addsi3 (hard_frame_pointer_rtx, ip_rtx, insn));
RTX_FRAME_RELATED_P (insn) = 1;
}
else
{
insn = GEN_INT (saved_regs - (4 + fp_offset));
insn = emit_insn (gen_addsi3 (hard_frame_pointer_rtx,
stack_pointer_rtx, insn));
RTX_FRAME_RELATED_P (insn) = 1;
}
}
size = offsets->outgoing_args - offsets->saved_args;
if (flag_stack_usage_info)
current_function_static_stack_size = size;
if (!IS_INTERRUPT (func_type)
&& (flag_stack_check == STATIC_BUILTIN_STACK_CHECK
|| flag_stack_clash_protection))
{
unsigned int regno;
if (!IS_NESTED (func_type) || clobber_ip)
regno = IP_REGNUM;
else if (df_regs_ever_live_p (LR_REGNUM))
regno = LR_REGNUM;
else
regno = 3;
if (crtl->is_leaf && !cfun->calls_alloca)
{
if (size > PROBE_INTERVAL && size > get_stack_check_protect ())
arm_emit_probe_stack_range (get_stack_check_protect (),
size - get_stack_check_protect (),
regno, live_regs_mask);
}
else if (size > 0)
arm_emit_probe_stack_range (get_stack_check_protect (), size,
regno, live_regs_mask);
}
if (clobber_ip)
{
if (!arm_r3_live_at_start_p () || saved_pretend_args)
insn = gen_rtx_REG (SImode, 3);
else
{
insn = plus_constant (Pmode, hard_frame_pointer_rtx, 4);
insn = gen_frame_mem (SImode, insn);
}
emit_set_insn (ip_rtx, insn);
emit_insn (gen_force_register_use (ip_rtx));
}
if (offsets->outgoing_args != offsets->saved_args + saved_regs)
{
rtx_insn *last = get_last_insn ();
amount = GEN_INT (offsets->saved_args + saved_regs
- offsets->outgoing_args);
insn = emit_insn (gen_addsi3 (stack_pointer_rtx, stack_pointer_rtx,
amount));
do
{
last = last ? NEXT_INSN (last) : get_insns ();
RTX_FRAME_RELATED_P (last) = 1;
}
while (last != insn);
if (frame_pointer_needed)
emit_insn (gen_stack_tie (stack_pointer_rtx,
hard_frame_pointer_rtx));
}
if (frame_pointer_needed && TARGET_THUMB2)
thumb_set_frame_pointer (offsets);
if (flag_pic && arm_pic_register != INVALID_REGNUM)
{
unsigned long mask;
mask = live_regs_mask;
mask &= THUMB2_WORK_REGS;
if (!IS_NESTED (func_type))
mask |= (1 << IP_REGNUM);
arm_load_pic_register (mask);
}
if (crtl->profile || !TARGET_SCHED_PROLOG
|| (arm_except_unwind_info (&global_options) == UI_TARGET
&& cfun->can_throw_non_call_exceptions))
emit_insn (gen_blockage ());
if ((live_regs_mask & (1 << LR_REGNUM)) == 0)
cfun->machine->lr_save_eliminated = 1;
}

static void
arm_print_condition (FILE *stream)
{
if (arm_ccfsm_state == 3 || arm_ccfsm_state == 4)
{
if (TARGET_THUMB)
{
output_operand_lossage ("predicated Thumb instruction");
return;
}
if (current_insn_predicate != NULL)
{
output_operand_lossage
("predicated instruction in conditional sequence");
return;
}
fputs (arm_condition_codes[arm_current_cc], stream);
}
else if (current_insn_predicate)
{
enum arm_cond_code code;
if (TARGET_THUMB1)
{
output_operand_lossage ("predicated Thumb instruction");
return;
}
code = get_arm_condition_code (current_insn_predicate);
fputs (arm_condition_codes[code], stream);
}
}
static void
arm_print_operand (FILE *stream, rtx x, int code)
{
switch (code)
{
case '@':
fputs (ASM_COMMENT_START, stream);
return;
case '_':
fputs (user_label_prefix, stream);
return;
case '|':
fputs (REGISTER_PREFIX, stream);
return;
case '?':
arm_print_condition (stream);
return;
case '.':
fputc('s', stream);
arm_print_condition (stream);
return;
case '!':
gcc_assert (TARGET_THUMB2);
if (current_insn_predicate)
arm_print_condition (stream);
else
fputc('s', stream);
break;
case '#':
return;
case 'N':
{
REAL_VALUE_TYPE r;
r = real_value_negate (CONST_DOUBLE_REAL_VALUE (x));
fprintf (stream, "%s", fp_const_from_val (&r));
}
return;
case 'c':
switch (GET_CODE (x))
{
case CONST_INT:
fprintf (stream, HOST_WIDE_INT_PRINT_DEC, INTVAL (x));
break;
case SYMBOL_REF:
output_addr_const (stream, x);
break;
case CONST:
if (GET_CODE (XEXP (x, 0)) == PLUS
&& GET_CODE (XEXP (XEXP (x, 0), 0)) == SYMBOL_REF)
{
output_addr_const (stream, x);
break;
}
default:
output_operand_lossage ("Unsupported operand for code '%c'", code);
}
return;
case 'x':
switch (GET_CODE (x))
{
case CONST_INT:
fprintf (stream, "#" HOST_WIDE_INT_PRINT_HEX, INTVAL (x));
break;
default:
output_operand_lossage ("Unsupported operand for code '%c'", code);
}
return;
case 'B':
if (CONST_INT_P (x))
{
HOST_WIDE_INT val;
val = ARM_SIGN_EXTEND (~INTVAL (x));
fprintf (stream, HOST_WIDE_INT_PRINT_DEC, val);
}
else
{
putc ('~', stream);
output_addr_const (stream, x);
}
return;
case 'b':
{
HOST_WIDE_INT val;
if (!CONST_INT_P (x)
|| (val = exact_log2 (INTVAL (x) & 0xffffffff)) < 0)
output_operand_lossage ("Unsupported operand for code '%c'", code);
else
fprintf (stream, "#" HOST_WIDE_INT_PRINT_DEC, val);
}
return;
case 'L':
fprintf (stream, HOST_WIDE_INT_PRINT_DEC, INTVAL(x) & 0xffff);
return;
case 'i':
fprintf (stream, "%s", arithmetic_instr (x, 1));
return;
case 'I':
fprintf (stream, "%s", arithmetic_instr (x, 0));
return;
case 'S':
{
HOST_WIDE_INT val;
const char *shift;
shift = shift_op (x, &val);
if (shift)
{
fprintf (stream, ", %s ", shift);
if (val == -1)
arm_print_operand (stream, XEXP (x, 1), 0);
else
fprintf (stream, "#" HOST_WIDE_INT_PRINT_DEC, val);
}
}
return;
case 'Q':
if (CONST_INT_P (x) || CONST_DOUBLE_P (x))
{
rtx part = gen_lowpart (SImode, x);
fprintf (stream, "#" HOST_WIDE_INT_PRINT_DEC, INTVAL (part));
return;
}
if (!REG_P (x) || REGNO (x) > LAST_ARM_REGNUM)
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
asm_fprintf (stream, "%r", REGNO (x) + (WORDS_BIG_ENDIAN ? 1 : 0));
return;
case 'R':
if (CONST_INT_P (x) || CONST_DOUBLE_P (x))
{
machine_mode mode = GET_MODE (x);
rtx part;
if (mode == VOIDmode)
mode = DImode;
part = gen_highpart_mode (SImode, mode, x);
fprintf (stream, "#" HOST_WIDE_INT_PRINT_DEC, INTVAL (part));
return;
}
if (!REG_P (x) || REGNO (x) > LAST_ARM_REGNUM)
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
asm_fprintf (stream, "%r", REGNO (x) + (WORDS_BIG_ENDIAN ? 0 : 1));
return;
case 'H':
if (!REG_P (x) || REGNO (x) > LAST_ARM_REGNUM)
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
asm_fprintf (stream, "%r", REGNO (x) + 1);
return;
case 'J':
if (!REG_P (x) || REGNO (x) > LAST_ARM_REGNUM)
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
asm_fprintf (stream, "%r", REGNO (x) + (WORDS_BIG_ENDIAN ? 3 : 2));
return;
case 'K':
if (!REG_P (x) || REGNO (x) > LAST_ARM_REGNUM)
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
asm_fprintf (stream, "%r", REGNO (x) + (WORDS_BIG_ENDIAN ? 2 : 3));
return;
case 'm':
asm_fprintf (stream, "%r",
REG_P (XEXP (x, 0))
? REGNO (XEXP (x, 0)) : REGNO (XEXP (XEXP (x, 0), 0)));
return;
case 'M':
asm_fprintf (stream, "{%r-%r}",
REGNO (x),
REGNO (x) + ARM_NUM_REGS (GET_MODE (x)) - 1);
return;
case 'h':
{
int regno = (REGNO (x) - FIRST_VFP_REGNUM) / 2;
int numregs = ARM_NUM_REGS (GET_MODE (x)) / 2;
if (numregs == 1)
asm_fprintf (stream, "{d%d}", regno);
else
asm_fprintf (stream, "{d%d-d%d}", regno, regno + numregs - 1);
}
return;
case 'd':
if (x == const_true_rtx)
return;
if (!COMPARISON_P (x))
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
fputs (arm_condition_codes[get_arm_condition_code (x)],
stream);
return;
case 'D':
if (x == const_true_rtx)
{
output_operand_lossage ("instruction never executed");
return;
}
if (!COMPARISON_P (x))
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
fputs (arm_condition_codes[ARM_INVERSE_CONDITION_CODE
(get_arm_condition_code (x))],
stream);
return;
case 's':
case 'V':
case 'W':
case 'X':
case 'Y':
case 'Z':
output_operand_lossage ("obsolete Maverick format code '%c'", code);
return;
case 'U':
if (!REG_P (x)
|| REGNO (x) < FIRST_IWMMXT_GR_REGNUM
|| REGNO (x) > LAST_IWMMXT_GR_REGNUM)
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
else
fprintf (stream, "%d", REGNO (x) - FIRST_IWMMXT_GR_REGNUM);
return;
case 'w':
if (!CONST_INT_P (x)
|| INTVAL (x) < 0
|| INTVAL (x) >= 16)
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
else
{
static const char * wc_reg_names [16] =
{
"wCID",  "wCon",  "wCSSF", "wCASF",
"wC4",   "wC5",   "wC6",   "wC7",
"wCGR0", "wCGR1", "wCGR2", "wCGR3",
"wC12",  "wC13",  "wC14",  "wC15"
};
fputs (wc_reg_names [INTVAL (x)], stream);
}
return;
case 'p':
{
machine_mode mode = GET_MODE (x);
int regno;
if (GET_MODE_SIZE (mode) != 8 || !REG_P (x))
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
regno = REGNO (x);
if (!VFP_REGNO_OK_FOR_DOUBLE (regno))
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
fprintf (stream, "s%d", regno - FIRST_VFP_REGNUM + 1);
}
return;
case 'P':
case 'q':
{
machine_mode mode = GET_MODE (x);
int is_quad = (code == 'q');
int regno;
if (GET_MODE_SIZE (mode) != (is_quad ? 16 : 8))
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
if (!REG_P (x)
|| !IS_VFP_REGNUM (REGNO (x)))
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
regno = REGNO (x);
if ((is_quad && !NEON_REGNO_OK_FOR_QUAD (regno))
|| (!is_quad && !VFP_REGNO_OK_FOR_DOUBLE (regno)))
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
fprintf (stream, "%c%d", is_quad ? 'q' : 'd',
(regno - FIRST_VFP_REGNUM) >> (is_quad ? 2 : 1));
}
return;
case 'e':
case 'f':
{
machine_mode mode = GET_MODE (x);
int regno;
if ((GET_MODE_SIZE (mode) != 16
&& GET_MODE_SIZE (mode) != 32) || !REG_P (x))
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
regno = REGNO (x);
if (!NEON_REGNO_OK_FOR_QUAD (regno))
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
if (GET_MODE_SIZE (mode) == 16)
fprintf (stream, "d%d", ((regno - FIRST_VFP_REGNUM) >> 1)
+ (code == 'f' ? 1 : 0));
else
fprintf (stream, "q%d", ((regno - FIRST_VFP_REGNUM) >> 2)
+ (code == 'f' ? 1 : 0));
}
return;
case 'G':
{
int index = vfp3_const_double_index (x);
gcc_assert (index != -1);
fprintf (stream, "%d", index);
}
return;
case 'T':
{
HOST_WIDE_INT bits = INTVAL (x);
fputc ("uspf"[bits & 3], stream);
}
return;
case 'F':
{
HOST_WIDE_INT bits = INTVAL (x);
fputc ("iipf"[bits & 3], stream);
}
return;
case 't':
{
HOST_WIDE_INT bits = INTVAL (x);
fputc ("usuf"[bits & 3], stream);
}
return;
case 'O':
{
HOST_WIDE_INT bits = INTVAL (x);
fputs ((bits & 4) != 0 ? "r" : "", stream);
}
return;
case 'A':
{
rtx addr;
bool postinc = FALSE;
rtx postinc_reg = NULL;
unsigned align, memsize, align_bits;
gcc_assert (MEM_P (x));
addr = XEXP (x, 0);
if (GET_CODE (addr) == POST_INC)
{
postinc = 1;
addr = XEXP (addr, 0);
}
if (GET_CODE (addr) == POST_MODIFY)
{
postinc_reg = XEXP( XEXP (addr, 1), 1);
addr = XEXP (addr, 0);
}
asm_fprintf (stream, "[%r", REGNO (addr));
align = MEM_ALIGN (x) >> 3;
memsize = MEM_SIZE (x);
if (memsize == 32 && (align % 32) == 0)
align_bits = 256;
else if ((memsize == 16 || memsize == 32) && (align % 16) == 0)
align_bits = 128;
else if (memsize >= 8 && (align % 8) == 0)
align_bits = 64;
else
align_bits = 0;
if (align_bits != 0)
asm_fprintf (stream, ":%d", align_bits);
asm_fprintf (stream, "]");
if (postinc)
fputs("!", stream);
if (postinc_reg)
asm_fprintf (stream, ", %r", REGNO (postinc_reg));
}
return;
case 'C':
{
rtx addr;
gcc_assert (MEM_P (x));
addr = XEXP (x, 0);
gcc_assert (REG_P (addr));
asm_fprintf (stream, "[%r]", REGNO (addr));
}
return;
case 'y':
{
machine_mode mode = GET_MODE (x);
int regno;
if (GET_MODE_SIZE (mode) != 4 || !REG_P (x))
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
regno = REGNO (x);
if (!VFP_REGNO_OK_FOR_SINGLE (regno))
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
regno = regno - FIRST_VFP_REGNUM;
fprintf (stream, "d%d[%d]", regno / 2, regno % 2);
}
return;
case 'v':
gcc_assert (CONST_DOUBLE_P (x));
int result;
result = vfp3_const_double_for_fract_bits (x);
if (result == 0)
result = vfp3_const_double_for_bits (x);
fprintf (stream, "#%d", result);
return;
case 'z':
{
machine_mode mode = GET_MODE (x);
int regno;
if (GET_MODE_SIZE (mode) != 2 || !REG_P (x))
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
regno = REGNO (x);
if (!VFP_REGNO_OK_FOR_SINGLE (regno))
{
output_operand_lossage ("invalid operand for code '%c'", code);
return;
}
regno = regno - FIRST_VFP_REGNUM;
fprintf (stream, "d%d[%d]", regno/2, ((regno % 2) ? 2 : 0));
}
return;
default:
if (x == 0)
{
output_operand_lossage ("missing operand");
return;
}
switch (GET_CODE (x))
{
case REG:
asm_fprintf (stream, "%r", REGNO (x));
break;
case MEM:
output_address (GET_MODE (x), XEXP (x, 0));
break;
case CONST_DOUBLE:
{
char fpstr[20];
real_to_decimal (fpstr, CONST_DOUBLE_REAL_VALUE (x),
sizeof (fpstr), 0, 1);
fprintf (stream, "#%s", fpstr);
}
break;
default:
gcc_assert (GET_CODE (x) != NEG);
fputc ('#', stream);
if (GET_CODE (x) == HIGH)
{
fputs (":lower16:", stream);
x = XEXP (x, 0);
}
output_addr_const (stream, x);
break;
}
}
}

static void
arm_print_operand_address (FILE *stream, machine_mode mode, rtx x)
{
if (TARGET_32BIT)
{
int is_minus = GET_CODE (x) == MINUS;
if (REG_P (x))
asm_fprintf (stream, "[%r]", REGNO (x));
else if (GET_CODE (x) == PLUS || is_minus)
{
rtx base = XEXP (x, 0);
rtx index = XEXP (x, 1);
HOST_WIDE_INT offset = 0;
if (!REG_P (base)
|| (REG_P (index) && REGNO (index) == SP_REGNUM))
{
std::swap (base, index);
}
switch (GET_CODE (index))
{
case CONST_INT:
offset = INTVAL (index);
if (is_minus)
offset = -offset;
asm_fprintf (stream, "[%r, #%wd]",
REGNO (base), offset);
break;
case REG:
asm_fprintf (stream, "[%r, %s%r]",
REGNO (base), is_minus ? "-" : "",
REGNO (index));
break;
case MULT:
case ASHIFTRT:
case LSHIFTRT:
case ASHIFT:
case ROTATERT:
{
asm_fprintf (stream, "[%r, %s%r",
REGNO (base), is_minus ? "-" : "",
REGNO (XEXP (index, 0)));
arm_print_operand (stream, index, 'S');
fputs ("]", stream);
break;
}
default:
gcc_unreachable ();
}
}
else if (GET_CODE (x) == PRE_INC || GET_CODE (x) == POST_INC
|| GET_CODE (x) == PRE_DEC || GET_CODE (x) == POST_DEC)
{
gcc_assert (REG_P (XEXP (x, 0)));
if (GET_CODE (x) == PRE_DEC || GET_CODE (x) == PRE_INC)
asm_fprintf (stream, "[%r, #%s%d]!",
REGNO (XEXP (x, 0)),
GET_CODE (x) == PRE_DEC ? "-" : "",
GET_MODE_SIZE (mode));
else
asm_fprintf (stream, "[%r], #%s%d",
REGNO (XEXP (x, 0)),
GET_CODE (x) == POST_DEC ? "-" : "",
GET_MODE_SIZE (mode));
}
else if (GET_CODE (x) == PRE_MODIFY)
{
asm_fprintf (stream, "[%r, ", REGNO (XEXP (x, 0)));
if (CONST_INT_P (XEXP (XEXP (x, 1), 1)))
asm_fprintf (stream, "#%wd]!",
INTVAL (XEXP (XEXP (x, 1), 1)));
else
asm_fprintf (stream, "%r]!",
REGNO (XEXP (XEXP (x, 1), 1)));
}
else if (GET_CODE (x) == POST_MODIFY)
{
asm_fprintf (stream, "[%r], ", REGNO (XEXP (x, 0)));
if (CONST_INT_P (XEXP (XEXP (x, 1), 1)))
asm_fprintf (stream, "#%wd",
INTVAL (XEXP (XEXP (x, 1), 1)));
else
asm_fprintf (stream, "%r",
REGNO (XEXP (XEXP (x, 1), 1)));
}
else output_addr_const (stream, x);
}
else
{
if (REG_P (x))
asm_fprintf (stream, "[%r]", REGNO (x));
else if (GET_CODE (x) == POST_INC)
asm_fprintf (stream, "%r!", REGNO (XEXP (x, 0)));
else if (GET_CODE (x) == PLUS)
{
gcc_assert (REG_P (XEXP (x, 0)));
if (CONST_INT_P (XEXP (x, 1)))
asm_fprintf (stream, "[%r, #%wd]",
REGNO (XEXP (x, 0)),
INTVAL (XEXP (x, 1)));
else
asm_fprintf (stream, "[%r, %r]",
REGNO (XEXP (x, 0)),
REGNO (XEXP (x, 1)));
}
else
output_addr_const (stream, x);
}
}

static bool
arm_print_operand_punct_valid_p (unsigned char code)
{
return (code == '@' || code == '|' || code == '.'
|| code == '(' || code == ')' || code == '#'
|| (TARGET_32BIT && (code == '?'))
|| (TARGET_THUMB2 && (code == '!'))
|| (TARGET_THUMB && (code == '_')));
}

static bool
arm_assemble_integer (rtx x, unsigned int size, int aligned_p)
{
machine_mode mode;
if (size == UNITS_PER_WORD && aligned_p)
{
fputs ("\t.word\t", asm_out_file);
output_addr_const (asm_out_file, x);
if (NEED_GOT_RELOC && flag_pic && making_const_table &&
(GET_CODE (x) == SYMBOL_REF || GET_CODE (x) == LABEL_REF))
{
if (!arm_pic_data_is_text_relative
|| (GET_CODE (x) == SYMBOL_REF
&& (!SYMBOL_REF_LOCAL_P (x)
|| (SYMBOL_REF_DECL (x)
? DECL_WEAK (SYMBOL_REF_DECL (x)) : 0))))
fputs ("(GOT)", asm_out_file);
else
fputs ("(GOTOFF)", asm_out_file);
}
fputc ('\n', asm_out_file);
return true;
}
mode = GET_MODE (x);
if (arm_vector_mode_supported_p (mode))
{
int i, units;
gcc_assert (GET_CODE (x) == CONST_VECTOR);
units = CONST_VECTOR_NUNITS (x);
size = GET_MODE_UNIT_SIZE (mode);
if (GET_MODE_CLASS (mode) == MODE_VECTOR_INT)
for (i = 0; i < units; i++)
{
rtx elt = CONST_VECTOR_ELT (x, i);
assemble_integer
(elt, size, i == 0 ? BIGGEST_ALIGNMENT : size * BITS_PER_UNIT, 1);
}
else
for (i = 0; i < units; i++)
{
rtx elt = CONST_VECTOR_ELT (x, i);
assemble_real
(*CONST_DOUBLE_REAL_VALUE (elt),
as_a <scalar_float_mode> (GET_MODE_INNER (mode)),
i == 0 ? BIGGEST_ALIGNMENT : size * BITS_PER_UNIT);
}
return true;
}
return default_assemble_integer (x, size, aligned_p);
}
static void
arm_elf_asm_cdtor (rtx symbol, int priority, bool is_ctor)
{
section *s;
if (!TARGET_AAPCS_BASED)
{
(is_ctor ?
default_named_section_asm_out_constructor
: default_named_section_asm_out_destructor) (symbol, priority);
return;
}
if (priority != DEFAULT_INIT_PRIORITY)
{
char buf[18];
sprintf (buf, "%s.%.5u",
is_ctor ? ".init_array" : ".fini_array",
priority);
s = get_section (buf, SECTION_WRITE | SECTION_NOTYPE, NULL_TREE);
}
else if (is_ctor)
s = ctors_section;
else
s = dtors_section;
switch_to_section (s);
assemble_align (POINTER_SIZE);
fputs ("\t.word\t", asm_out_file);
output_addr_const (asm_out_file, symbol);
fputs ("(target1)\n", asm_out_file);
}
static void
arm_elf_asm_constructor (rtx symbol, int priority)
{
arm_elf_asm_cdtor (symbol, priority, true);
}
static void
arm_elf_asm_destructor (rtx symbol, int priority)
{
arm_elf_asm_cdtor (symbol, priority, false);
}

enum arm_cond_code
maybe_get_arm_condition_code (rtx comparison)
{
machine_mode mode = GET_MODE (XEXP (comparison, 0));
enum arm_cond_code code;
enum rtx_code comp_code = GET_CODE (comparison);
if (GET_MODE_CLASS (mode) != MODE_CC)
mode = SELECT_CC_MODE (comp_code, XEXP (comparison, 0),
XEXP (comparison, 1));
switch (mode)
{
case E_CC_DNEmode: code = ARM_NE; goto dominance;
case E_CC_DEQmode: code = ARM_EQ; goto dominance;
case E_CC_DGEmode: code = ARM_GE; goto dominance;
case E_CC_DGTmode: code = ARM_GT; goto dominance;
case E_CC_DLEmode: code = ARM_LE; goto dominance;
case E_CC_DLTmode: code = ARM_LT; goto dominance;
case E_CC_DGEUmode: code = ARM_CS; goto dominance;
case E_CC_DGTUmode: code = ARM_HI; goto dominance;
case E_CC_DLEUmode: code = ARM_LS; goto dominance;
case E_CC_DLTUmode: code = ARM_CC;
dominance:
if (comp_code == EQ)
return ARM_INVERSE_CONDITION_CODE (code);
if (comp_code == NE)
return code;
return ARM_NV;
case E_CC_NOOVmode:
switch (comp_code)
{
case NE: return ARM_NE;
case EQ: return ARM_EQ;
case GE: return ARM_PL;
case LT: return ARM_MI;
default: return ARM_NV;
}
case E_CC_Zmode:
switch (comp_code)
{
case NE: return ARM_NE;
case EQ: return ARM_EQ;
default: return ARM_NV;
}
case E_CC_Nmode:
switch (comp_code)
{
case NE: return ARM_MI;
case EQ: return ARM_PL;
default: return ARM_NV;
}
case E_CCFPEmode:
case E_CCFPmode:
switch (comp_code)
{
case GE: return ARM_GE;
case GT: return ARM_GT;
case LE: return ARM_LS;
case LT: return ARM_MI;
case NE: return ARM_NE;
case EQ: return ARM_EQ;
case ORDERED: return ARM_VC;
case UNORDERED: return ARM_VS;
case UNLT: return ARM_LT;
case UNLE: return ARM_LE;
case UNGT: return ARM_HI;
case UNGE: return ARM_PL;
case UNEQ: 
case LTGT: 
default: return ARM_NV;
}
case E_CC_SWPmode:
switch (comp_code)
{
case NE: return ARM_NE;
case EQ: return ARM_EQ;
case GE: return ARM_LE;
case GT: return ARM_LT;
case LE: return ARM_GE;
case LT: return ARM_GT;
case GEU: return ARM_LS;
case GTU: return ARM_CC;
case LEU: return ARM_CS;
case LTU: return ARM_HI;
default: return ARM_NV;
}
case E_CC_Cmode:
switch (comp_code)
{
case LTU: return ARM_CS;
case GEU: return ARM_CC;
case NE: return ARM_CS;
case EQ: return ARM_CC;
default: return ARM_NV;
}
case E_CC_CZmode:
switch (comp_code)
{
case NE: return ARM_NE;
case EQ: return ARM_EQ;
case GEU: return ARM_CS;
case GTU: return ARM_HI;
case LEU: return ARM_LS;
case LTU: return ARM_CC;
default: return ARM_NV;
}
case E_CC_NCVmode:
switch (comp_code)
{
case GE: return ARM_GE;
case LT: return ARM_LT;
case GEU: return ARM_CS;
case LTU: return ARM_CC;
default: return ARM_NV;
}
case E_CC_Vmode:
switch (comp_code)
{
case NE: return ARM_VS;
case EQ: return ARM_VC;
default: return ARM_NV;
}
case E_CCmode:
switch (comp_code)
{
case NE: return ARM_NE;
case EQ: return ARM_EQ;
case GE: return ARM_GE;
case GT: return ARM_GT;
case LE: return ARM_LE;
case LT: return ARM_LT;
case GEU: return ARM_CS;
case GTU: return ARM_HI;
case LEU: return ARM_LS;
case LTU: return ARM_CC;
default: return ARM_NV;
}
default: gcc_unreachable ();
}
}
static enum arm_cond_code
get_arm_condition_code (rtx comparison)
{
enum arm_cond_code code = maybe_get_arm_condition_code (comparison);
gcc_assert (code != ARM_NV);
return code;
}
static bool
arm_fixed_condition_code_regs (unsigned int *p1, unsigned int *p2)
{
if (!TARGET_32BIT)
return false;
*p1 = CC_REGNUM;
*p2 = TARGET_HARD_FLOAT ? VFPCC_REGNUM : INVALID_REGNUM;
return true;
}
void
thumb2_final_prescan_insn (rtx_insn *insn)
{
rtx_insn *first_insn = insn;
rtx body = PATTERN (insn);
rtx predicate;
enum arm_cond_code code;
int n;
int mask;
int max;
max = MAX_INSN_PER_IT_BLOCK;
if (arm_condexec_count)
arm_condexec_count--;
if (arm_condexec_count)
return;
if (GET_CODE (body) != COND_EXEC)
return;
if (JUMP_P (insn))
return;
predicate = COND_EXEC_TEST (body);
arm_current_cc = get_arm_condition_code (predicate);
n = get_attr_ce_count (insn);
arm_condexec_count = 1;
arm_condexec_mask = (1 << n) - 1;
arm_condexec_masklen = n;
for (;;)
{
insn = next_nonnote_insn (insn);
if (!NONJUMP_INSN_P (insn) && !JUMP_P (insn))
break;
body = PATTERN (insn);
if (GET_CODE (body) == USE
|| GET_CODE (body) == CLOBBER)
continue;
if (GET_CODE (body) != COND_EXEC)
break;
n = get_attr_ce_count (insn);
if (arm_condexec_masklen + n > max)
break;
predicate = COND_EXEC_TEST (body);
code = get_arm_condition_code (predicate);
mask = (1 << n) - 1;
if (arm_current_cc == code)
arm_condexec_mask |= (mask << arm_condexec_masklen);
else if (arm_current_cc != ARM_INVERSE_CONDITION_CODE(code))
break;
arm_condexec_count++;
arm_condexec_masklen += n;
if (JUMP_P (insn))
break;
}
extract_constrain_insn_cached (first_insn);
}
void
arm_final_prescan_insn (rtx_insn *insn)
{
rtx body = PATTERN (insn);
int reverse = 0;
int seeking_return = 0;
enum rtx_code return_code = UNKNOWN;
rtx_insn *start_insn = insn;
if (arm_ccfsm_state == 4)
{
if (insn == arm_target_insn)
{
arm_target_insn = NULL;
arm_ccfsm_state = 0;
}
return;
}
if (arm_ccfsm_state == 3)
{
if (simplejump_p (insn))
{
start_insn = next_nonnote_insn (start_insn);
if (BARRIER_P (start_insn))
{
start_insn = next_nonnote_insn (start_insn);
}
if (LABEL_P (start_insn)
&& CODE_LABEL_NUMBER (start_insn) == arm_target_label
&& LABEL_NUSES (start_insn) == 1)
reverse = TRUE;
else
return;
}
else if (ANY_RETURN_P (body))
{
start_insn = next_nonnote_insn (start_insn);
if (BARRIER_P (start_insn))
start_insn = next_nonnote_insn (start_insn);
if (LABEL_P (start_insn)
&& CODE_LABEL_NUMBER (start_insn) == arm_target_label
&& LABEL_NUSES (start_insn) == 1)
{
reverse = TRUE;
seeking_return = 1;
return_code = GET_CODE (body);
}
else
return;
}
else
return;
}
gcc_assert (!arm_ccfsm_state || reverse);
if (!JUMP_P (insn))
return;
if (GET_CODE (body) == PARALLEL && XVECLEN (body, 0) > 0)
body = XVECEXP (body, 0, 0);
if (reverse
|| (GET_CODE (body) == SET && GET_CODE (SET_DEST (body)) == PC
&& GET_CODE (SET_SRC (body)) == IF_THEN_ELSE))
{
int insns_skipped;
int fail = FALSE, succeed = FALSE;
int then_not_else = TRUE;
rtx_insn *this_insn = start_insn;
rtx label = 0;
if (reverse)
{
if (!seeking_return)
label = XEXP (SET_SRC (body), 0);
}
else if (GET_CODE (XEXP (SET_SRC (body), 1)) == LABEL_REF)
label = XEXP (XEXP (SET_SRC (body), 1), 0);
else if (GET_CODE (XEXP (SET_SRC (body), 2)) == LABEL_REF)
{
label = XEXP (XEXP (SET_SRC (body), 2), 0);
then_not_else = FALSE;
}
else if (ANY_RETURN_P (XEXP (SET_SRC (body), 1)))
{
seeking_return = 1;
return_code = GET_CODE (XEXP (SET_SRC (body), 1));
}
else if (ANY_RETURN_P (XEXP (SET_SRC (body), 2)))
{
seeking_return = 1;
return_code = GET_CODE (XEXP (SET_SRC (body), 2));
then_not_else = FALSE;
}
else
gcc_unreachable ();
for (insns_skipped = 0;
!fail && !succeed && insns_skipped++ < max_insns_skipped;)
{
rtx scanbody;
this_insn = next_nonnote_insn (this_insn);
if (!this_insn)
break;
switch (GET_CODE (this_insn))
{
case CODE_LABEL:
if (this_insn == label)
{
arm_ccfsm_state = 1;
succeed = TRUE;
}
else
fail = TRUE;
break;
case BARRIER:
this_insn = next_nonnote_insn (this_insn);
if (this_insn && this_insn == label)
{
arm_ccfsm_state = 1;
succeed = TRUE;
}
else
fail = TRUE;
break;
case CALL_INSN:
if (arm_arch5)
{
fail = TRUE;
break;
}
this_insn = next_nonnote_insn (this_insn);
if (this_insn && BARRIER_P (this_insn))
this_insn = next_nonnote_insn (this_insn);
if (this_insn && this_insn == label
&& insns_skipped < max_insns_skipped)
{
arm_ccfsm_state = 1;
succeed = TRUE;
}
else
fail = TRUE;
break;
case JUMP_INSN:
scanbody = PATTERN (this_insn);
if (GET_CODE (scanbody) == SET
&& GET_CODE (SET_DEST (scanbody)) == PC)
{
if (GET_CODE (SET_SRC (scanbody)) == LABEL_REF
&& XEXP (SET_SRC (scanbody), 0) == label && !reverse)
{
arm_ccfsm_state = 2;
succeed = TRUE;
}
else if (GET_CODE (SET_SRC (scanbody)) == IF_THEN_ELSE)
fail = TRUE;
}
else if (GET_CODE (scanbody) == return_code
&& !use_return_insn (TRUE, NULL)
&& !optimize_size)
fail = TRUE;
else if (GET_CODE (scanbody) == return_code)
{
arm_ccfsm_state = 2;
succeed = TRUE;
}
else if (GET_CODE (scanbody) == PARALLEL)
{
switch (get_attr_conds (this_insn))
{
case CONDS_NOCOND:
break;
default:
fail = TRUE;
break;
}
}
else
fail = TRUE;	
break;
case INSN:
scanbody = PATTERN (this_insn);
if (!(GET_CODE (scanbody) == SET
|| GET_CODE (scanbody) == PARALLEL)
|| get_attr_conds (this_insn) != CONDS_NOCOND)
fail = TRUE;
break;
default:
break;
}
}
if (succeed)
{
if ((!seeking_return) && (arm_ccfsm_state == 1 || reverse))
arm_target_label = CODE_LABEL_NUMBER (label);
else
{
gcc_assert (seeking_return || arm_ccfsm_state == 2);
while (this_insn && GET_CODE (PATTERN (this_insn)) == USE)
{
this_insn = next_nonnote_insn (this_insn);
gcc_assert (!this_insn
|| (!BARRIER_P (this_insn)
&& !LABEL_P (this_insn)));
}
if (!this_insn)
{
extract_constrain_insn_cached (insn);
arm_ccfsm_state = 0;
arm_target_insn = NULL;
return;
}
arm_target_insn = this_insn;
}
if (!reverse)
arm_current_cc = get_arm_condition_code (XEXP (SET_SRC (body), 0));
if (reverse || then_not_else)
arm_current_cc = ARM_INVERSE_CONDITION_CODE (arm_current_cc);
}
extract_constrain_insn_cached (insn);
}
}
void
thumb2_asm_output_opcode (FILE * stream)
{
char buff[5];
int n;
if (arm_condexec_mask)
{
for (n = 0; n < arm_condexec_masklen; n++)
buff[n] = (arm_condexec_mask & (1 << n)) ? 't' : 'e';
buff[n] = 0;
asm_fprintf(stream, "i%s\t%s\n\t", buff,
arm_condition_codes[arm_current_cc]);
arm_condexec_mask = 0;
}
}
static unsigned int
arm_hard_regno_nregs (unsigned int regno, machine_mode mode)
{
if (TARGET_32BIT
&& regno > PC_REGNUM
&& regno != FRAME_POINTER_REGNUM
&& regno != ARG_POINTER_REGNUM
&& !IS_VFP_REGNUM (regno))
return 1;
return ARM_NUM_REGS (mode);
}
static bool
arm_hard_regno_mode_ok (unsigned int regno, machine_mode mode)
{
if (GET_MODE_CLASS (mode) == MODE_CC)
return (regno == CC_REGNUM
|| (TARGET_HARD_FLOAT
&& regno == VFPCC_REGNUM));
if (regno == CC_REGNUM && GET_MODE_CLASS (mode) != MODE_CC)
return false;
if (TARGET_THUMB1)
return (ARM_NUM_REGS (mode) < 2) || (regno < LAST_LO_REGNUM);
if (TARGET_HARD_FLOAT && IS_VFP_REGNUM (regno))
{
if (mode == SFmode || mode == SImode)
return VFP_REGNO_OK_FOR_SINGLE (regno);
if (mode == DFmode)
return VFP_REGNO_OK_FOR_DOUBLE (regno);
if (mode == HFmode)
return VFP_REGNO_OK_FOR_SINGLE (regno);
if (mode == HImode)
return VFP_REGNO_OK_FOR_SINGLE (regno);
if (TARGET_NEON)
return (VALID_NEON_DREG_MODE (mode) && VFP_REGNO_OK_FOR_DOUBLE (regno))
|| (VALID_NEON_QREG_MODE (mode)
&& NEON_REGNO_OK_FOR_QUAD (regno))
|| (mode == TImode && NEON_REGNO_OK_FOR_NREGS (regno, 2))
|| (mode == EImode && NEON_REGNO_OK_FOR_NREGS (regno, 3))
|| (mode == OImode && NEON_REGNO_OK_FOR_NREGS (regno, 4))
|| (mode == CImode && NEON_REGNO_OK_FOR_NREGS (regno, 6))
|| (mode == XImode && NEON_REGNO_OK_FOR_NREGS (regno, 8));
return false;
}
if (TARGET_REALLY_IWMMXT)
{
if (IS_IWMMXT_GR_REGNUM (regno))
return mode == SImode;
if (IS_IWMMXT_REGNUM (regno))
return VALID_IWMMXT_REG_MODE (mode);
}
if (regno <= LAST_ARM_REGNUM)
{
if (ARM_NUM_REGS (mode) > 4)
return false;
if (TARGET_THUMB2)
return true;
return !(TARGET_LDRD && GET_MODE_SIZE (mode) > 4 && (regno & 1) != 0);
}
if (regno == FRAME_POINTER_REGNUM
|| regno == ARG_POINTER_REGNUM)
return GET_MODE_CLASS (mode) == MODE_INT;
return false;
}
static bool
arm_modes_tieable_p (machine_mode mode1, machine_mode mode2)
{
if (GET_MODE_CLASS (mode1) == GET_MODE_CLASS (mode2))
return true;
if (TARGET_NEON
&& (VALID_NEON_DREG_MODE (mode1)
|| VALID_NEON_QREG_MODE (mode1)
|| VALID_NEON_STRUCT_MODE (mode1))
&& (VALID_NEON_DREG_MODE (mode2)
|| VALID_NEON_QREG_MODE (mode2)
|| VALID_NEON_STRUCT_MODE (mode2)))
return true;
return false;
}
enum reg_class
arm_regno_class (int regno)
{
if (regno == PC_REGNUM)
return NO_REGS;
if (TARGET_THUMB1)
{
if (regno == STACK_POINTER_REGNUM)
return STACK_REG;
if (regno == CC_REGNUM)
return CC_REG;
if (regno < 8)
return LO_REGS;
return HI_REGS;
}
if (TARGET_THUMB2 && regno < 8)
return LO_REGS;
if (   regno <= LAST_ARM_REGNUM
|| regno == FRAME_POINTER_REGNUM
|| regno == ARG_POINTER_REGNUM)
return TARGET_THUMB2 ? HI_REGS : GENERAL_REGS;
if (regno == CC_REGNUM || regno == VFPCC_REGNUM)
return TARGET_THUMB2 ? CC_REG : NO_REGS;
if (IS_VFP_REGNUM (regno))
{
if (regno <= D7_VFP_REGNUM)
return VFP_D0_D7_REGS;
else if (regno <= LAST_LO_VFP_REGNUM)
return VFP_LO_REGS;
else
return VFP_HI_REGS;
}
if (IS_IWMMXT_REGNUM (regno))
return IWMMXT_REGS;
if (IS_IWMMXT_GR_REGNUM (regno))
return IWMMXT_GR_REGS;
return NO_REGS;
}
int
arm_debugger_arg_offset (int value, rtx addr)
{
rtx_insn *insn;
if (value != 0)
return 0;
if (!REG_P (addr))
return 0;
if (REGNO (addr) == (unsigned) HARD_FRAME_POINTER_REGNUM)
return 0;
if ((TARGET_THUMB || !frame_pointer_needed)
&& REGNO (addr) == SP_REGNUM)
return 0;
for (insn = get_insns (); insn; insn = NEXT_INSN (insn))
{
if (   NONJUMP_INSN_P (insn)
&& GET_CODE (PATTERN (insn)) == SET
&& REGNO    (XEXP (PATTERN (insn), 0)) == REGNO (addr)
&& GET_CODE (XEXP (PATTERN (insn), 1)) == PLUS
&& REG_P (XEXP (XEXP (PATTERN (insn), 1), 0))
&& REGNO    (XEXP (XEXP (PATTERN (insn), 1), 0)) == (unsigned) HARD_FRAME_POINTER_REGNUM
&& CONST_INT_P (XEXP (XEXP (PATTERN (insn), 1), 1))
)
{
value = INTVAL (XEXP (XEXP (PATTERN (insn), 1), 1));
break;
}
}
if (value == 0)
{
debug_rtx (addr);
warning (0, "unable to compute real location of stacked parameter");
value = 8; 
}
return value;
}

static tree
arm_promoted_type (const_tree t)
{
if (SCALAR_FLOAT_TYPE_P (t)
&& TYPE_PRECISION (t) == 16
&& TYPE_MAIN_VARIANT (t) == arm_fp16_type_node)
return float_type_node;
return NULL_TREE;
}
static bool
arm_scalar_mode_supported_p (scalar_mode mode)
{
if (mode == HFmode)
return (arm_fp16_format != ARM_FP16_FORMAT_NONE);
else if (ALL_FIXED_POINT_MODE_P (mode))
return true;
else
return default_scalar_mode_supported_p (mode);
}
static enum flt_eval_method
arm_excess_precision (enum excess_precision_type type)
{
switch (type)
{
case EXCESS_PRECISION_TYPE_FAST:
case EXCESS_PRECISION_TYPE_STANDARD:
return (TARGET_VFP_FP16INST
? FLT_EVAL_METHOD_PROMOTE_TO_FLOAT16
: FLT_EVAL_METHOD_PROMOTE_TO_FLOAT);
case EXCESS_PRECISION_TYPE_IMPLICIT:
return FLT_EVAL_METHOD_PROMOTE_TO_FLOAT16;
default:
gcc_unreachable ();
}
return FLT_EVAL_METHOD_UNPREDICTABLE;
}
static opt_scalar_float_mode
arm_floatn_mode (int n, bool extended)
{
if (!extended && n == 16)
{
if (arm_fp16_format == ARM_FP16_FORMAT_IEEE)
return HFmode;
return opt_scalar_float_mode ();
}
return default_floatn_mode (n, extended);
}
void
neon_disambiguate_copy (rtx *operands, rtx *dest, rtx *src, unsigned int count)
{
unsigned int i;
if (!reg_overlap_mentioned_p (operands[0], operands[1])
|| REGNO (operands[0]) < REGNO (operands[1]))
{
for (i = 0; i < count; i++)
{
operands[2 * i] = dest[i];
operands[2 * i + 1] = src[i];
}
}
else
{
for (i = 0; i < count; i++)
{
operands[2 * i] = dest[count - i - 1];
operands[2 * i + 1] = src[count - i - 1];
}
}
}
void
neon_split_vcombine (rtx operands[3])
{
unsigned int dest = REGNO (operands[0]);
unsigned int src1 = REGNO (operands[1]);
unsigned int src2 = REGNO (operands[2]);
machine_mode halfmode = GET_MODE (operands[1]);
unsigned int halfregs = REG_NREGS (operands[1]);
rtx destlo, desthi;
if (src1 == dest && src2 == dest + halfregs)
{
emit_note (NOTE_INSN_DELETED);
return;
}
destlo = gen_rtx_REG_offset (operands[0], halfmode, dest, 0);
desthi = gen_rtx_REG_offset (operands[0], halfmode, dest + halfregs,
GET_MODE_SIZE (halfmode));
if (src2 == dest && src1 == dest + halfregs)
{
rtx x = gen_rtx_SET (destlo, operands[1]);
rtx y = gen_rtx_SET (desthi, operands[2]);
emit_insn (gen_rtx_PARALLEL (VOIDmode, gen_rtvec (2, x, y)));
return;
}
if (!reg_overlap_mentioned_p (operands[2], destlo))
{
if (src1 != dest)
emit_move_insn (destlo, operands[1]);
if (src2 != dest + halfregs)
emit_move_insn (desthi, operands[2]);
}
else
{
if (src2 != dest + halfregs)
emit_move_insn (desthi, operands[2]);
if (src1 != dest)
emit_move_insn (destlo, operands[1]);
}
}

inline static int
number_of_first_bit_set (unsigned mask)
{
return ctz_hwi (mask);
}
static rtx_insn *
thumb1_emit_multi_reg_push (unsigned long mask, unsigned long real_regs)
{
unsigned long regno;
rtx par[10], tmp, reg;
rtx_insn *insn;
int i, j;
for (i = 0; mask; ++i, mask &= mask - 1)
{
regno = ctz_hwi (mask);
reg = gen_rtx_REG (SImode, regno);
if (i == 0)
tmp = gen_rtx_UNSPEC (BLKmode, gen_rtvec (1, reg), UNSPEC_PUSH_MULT);
else
tmp = gen_rtx_USE (VOIDmode, reg);
par[i] = tmp;
}
tmp = plus_constant (Pmode, stack_pointer_rtx, -4 * i);
tmp = gen_rtx_PRE_MODIFY (Pmode, stack_pointer_rtx, tmp);
tmp = gen_frame_mem (BLKmode, tmp);
tmp = gen_rtx_SET (tmp, par[0]);
par[0] = tmp;
tmp = gen_rtx_PARALLEL (VOIDmode, gen_rtvec_v (i, par));
insn = emit_insn (tmp);
tmp = plus_constant (Pmode, stack_pointer_rtx, -4 * i);
tmp = gen_rtx_SET (stack_pointer_rtx, tmp);
par[0] = tmp;
for (j = 0; real_regs; ++j, real_regs &= real_regs - 1)
{
regno = ctz_hwi (real_regs);
reg = gen_rtx_REG (SImode, regno);
tmp = plus_constant (Pmode, stack_pointer_rtx, j * 4);
tmp = gen_frame_mem (SImode, tmp);
tmp = gen_rtx_SET (tmp, reg);
RTX_FRAME_RELATED_P (tmp) = 1;
par[j + 1] = tmp;
}
if (j == 0)
tmp = par[0];
else
{
RTX_FRAME_RELATED_P (par[0]) = 1;
tmp = gen_rtx_SEQUENCE (VOIDmode, gen_rtvec_v (j + 1, par));
}
add_reg_note (insn, REG_FRAME_RELATED_EXPR, tmp);
return insn;
}
static void
thumb_pop (FILE *f, unsigned long mask)
{
int regno;
int lo_mask = mask & 0xFF;
gcc_assert (mask);
if (lo_mask == 0 && (mask & (1 << PC_REGNUM)))
{
thumb_exit (f, -1);
return;
}
fprintf (f, "\tpop\t{");
for (regno = 0; regno <= LAST_LO_REGNUM; regno++, lo_mask >>= 1)
{
if (lo_mask & 1)
{
asm_fprintf (f, "%r", regno);
if ((lo_mask & ~1) != 0)
fprintf (f, ", ");
}
}
if (mask & (1 << PC_REGNUM))
{
if (TARGET_INTERWORK || TARGET_BACKTRACE || crtl->calls_eh_return
|| IS_CMSE_ENTRY (arm_current_func_type ()))
{
fprintf (f, "}\n");
thumb_exit (f, -1);
return;
}
else
{
if (mask & 0xFF)
fprintf (f, ", ");
asm_fprintf (f, "%r", PC_REGNUM);
}
}
fprintf (f, "}\n");
}
static void
thumb_exit (FILE *f, int reg_containing_return_addr)
{
unsigned regs_available_for_popping;
unsigned regs_to_pop;
int pops_needed;
unsigned available;
unsigned required;
machine_mode mode;
int size;
int restore_a4 = FALSE;
regs_to_pop = 0;
pops_needed = 0;
if (reg_containing_return_addr == -1)
{
regs_to_pop |= 1 << LR_REGNUM;
++pops_needed;
}
if (TARGET_BACKTRACE)
{
regs_to_pop |= (1 << ARM_HARD_FRAME_POINTER_REGNUM) | (1 << SP_REGNUM);
pops_needed += 2;
}
if (pops_needed == 0)
{
if (crtl->calls_eh_return)
asm_fprintf (f, "\tadd\t%r, %r\n", SP_REGNUM, ARM_EH_STACKADJ_REGNUM);
if (IS_CMSE_ENTRY (arm_current_func_type ()))
{
asm_fprintf (f, "\tmsr\tAPSR_nzcvq, %r\n",
reg_containing_return_addr);
asm_fprintf (f, "\tbxns\t%r\n", reg_containing_return_addr);
}
else
asm_fprintf (f, "\tbx\t%r\n", reg_containing_return_addr);
return;
}
else if (!TARGET_INTERWORK
&& !TARGET_BACKTRACE
&& !is_called_in_ARM_mode (current_function_decl)
&& !crtl->calls_eh_return
&& !IS_CMSE_ENTRY (arm_current_func_type ()))
{
asm_fprintf (f, "\tpop\t{%r}\n", PC_REGNUM);
return;
}
regs_available_for_popping = 0;
if (crtl->calls_eh_return)
size = 12;
else
{
if (crtl->return_rtx != 0)
mode = GET_MODE (crtl->return_rtx);
else
mode = DECL_MODE (DECL_RESULT (current_function_decl));
size = GET_MODE_SIZE (mode);
if (size == 0)
{
if (mode == VOIDmode)
regs_available_for_popping =
(1 << ARG_REGISTER (1))
| (1 << ARG_REGISTER (2))
| (1 << ARG_REGISTER (3));
else
regs_available_for_popping =
(1 << ARG_REGISTER (2))
| (1 << ARG_REGISTER (3));
}
else if (size <= 4)
regs_available_for_popping =
(1 << ARG_REGISTER (2))
| (1 << ARG_REGISTER (3));
else if (size <= 8)
regs_available_for_popping =
(1 << ARG_REGISTER (3));
}
for (available = regs_available_for_popping,
required  = regs_to_pop;
required != 0 && available != 0;
available &= ~(available & - available),
required  &= ~(required  & - required))
-- pops_needed;
if (available > 0)
regs_available_for_popping &= ~available;
else if (pops_needed)
{
if (regs_available_for_popping == 0
&& reg_containing_return_addr == LAST_ARG_REGNUM)
{
asm_fprintf (f, "\tmov\t%r, %r\n", LR_REGNUM, LAST_ARG_REGNUM);
reg_containing_return_addr = LR_REGNUM;
}
else if (size > 12)
{
restore_a4 = TRUE;
asm_fprintf (f, "\tmov\t%r, %r\n",IP_REGNUM, LAST_ARG_REGNUM);
}
if (reg_containing_return_addr != LAST_ARG_REGNUM)
{
regs_available_for_popping |= 1 << LAST_ARG_REGNUM;
--pops_needed;
}
}
thumb_pop (f, regs_available_for_popping);
if (reg_containing_return_addr == -1)
{
regs_to_pop &= ~(1 << LR_REGNUM);
reg_containing_return_addr =
number_of_first_bit_set (regs_available_for_popping);
regs_available_for_popping &= ~(1 << reg_containing_return_addr);
}
if (regs_available_for_popping)
{
int frame_pointer;
frame_pointer = number_of_first_bit_set (regs_available_for_popping);
asm_fprintf (f, "\tmov\t%r, %r\n",
ARM_HARD_FRAME_POINTER_REGNUM, frame_pointer);
regs_available_for_popping &= ~(1 << frame_pointer);
regs_to_pop &= ~(1 << ARM_HARD_FRAME_POINTER_REGNUM);
if (regs_available_for_popping)
{
int stack_pointer;
stack_pointer = number_of_first_bit_set (regs_available_for_popping);
asm_fprintf (f, "\tmov\t%r, %r\n", SP_REGNUM, stack_pointer);
}
else
{
regs_available_for_popping |= (1 << frame_pointer);
}
}
if (regs_available_for_popping == 0 && pops_needed > 0)
{
regs_available_for_popping |= 1 << reg_containing_return_addr;
asm_fprintf (f, "\tmov\t%r, %r\n", LR_REGNUM,
reg_containing_return_addr);
reg_containing_return_addr = LR_REGNUM;
}
if (pops_needed > 0)
{
int  popped_into;
int  move_to;
thumb_pop (f, regs_available_for_popping);
popped_into = number_of_first_bit_set (regs_available_for_popping);
move_to     = number_of_first_bit_set (regs_to_pop);
asm_fprintf (f, "\tmov\t%r, %r\n", move_to, popped_into);
--pops_needed;
}
if (pops_needed > 0)
{
int  popped_into;
thumb_pop (f, regs_available_for_popping);
popped_into = number_of_first_bit_set (regs_available_for_popping);
asm_fprintf (f, "\tmov\t%r, %r\n", SP_REGNUM, popped_into);
}
if (restore_a4)
{
if (reg_containing_return_addr != LR_REGNUM)
{
asm_fprintf (f, "\tmov\t%r, %r\n", LR_REGNUM, LAST_ARG_REGNUM);
reg_containing_return_addr = LR_REGNUM;
}
asm_fprintf (f, "\tmov\t%r, %r\n", LAST_ARG_REGNUM, IP_REGNUM);
}
if (crtl->calls_eh_return)
asm_fprintf (f, "\tadd\t%r, %r\n", SP_REGNUM, ARM_EH_STACKADJ_REGNUM);
if (IS_CMSE_ENTRY (arm_current_func_type ()))
{
if (reg_containing_return_addr != LR_REGNUM)
asm_fprintf (f, "\tmov\tlr, r0\n");
asm_fprintf (f, "\tmsr\tAPSR_nzcvq, %r\n", reg_containing_return_addr);
asm_fprintf (f, "\tbxns\t%r\n", reg_containing_return_addr);
}
else
asm_fprintf (f, "\tbx\t%r\n", reg_containing_return_addr);
}

void
thumb1_final_prescan_insn (rtx_insn *insn)
{
if (flag_print_asm_name)
asm_fprintf (asm_out_file, "%@ 0x%04x\n",
INSN_ADDRESSES (INSN_UID (insn)));
if (INSN_CODE (insn) != CODE_FOR_cbranchsi4_insn)
{
enum attr_conds conds;
if (cfun->machine->thumb1_cc_insn)
{
if (modified_in_p (cfun->machine->thumb1_cc_op0, insn)
|| modified_in_p (cfun->machine->thumb1_cc_op1, insn))
CC_STATUS_INIT;
}
conds = get_attr_conds (insn);
if (conds == CONDS_SET)
{
rtx set = single_set (insn);
cfun->machine->thumb1_cc_insn = insn;
cfun->machine->thumb1_cc_op0 = SET_DEST (set);
cfun->machine->thumb1_cc_op1 = const0_rtx;
cfun->machine->thumb1_cc_mode = CC_NOOVmode;
if (INSN_CODE (insn) == CODE_FOR_thumb1_subsi3_insn)
{
rtx src1 = XEXP (SET_SRC (set), 1);
if (src1 == const0_rtx)
cfun->machine->thumb1_cc_mode = CCmode;
}
else if (REG_P (SET_DEST (set)) && REG_P (SET_SRC (set)))
{
cfun->machine->thumb1_cc_op0 = SET_SRC (set);
}
}
else if (conds != CONDS_NOCOND)
cfun->machine->thumb1_cc_insn = NULL_RTX;
}
if (cfun->machine->lr_save_eliminated
&& get_attr_far_jump (insn) == FAR_JUMP_YES)
internal_error("Unexpected thumb1 far jump");
}
int
thumb_shiftable_const (unsigned HOST_WIDE_INT val)
{
unsigned HOST_WIDE_INT mask = 0xff;
int i;
val = val & (unsigned HOST_WIDE_INT)0xffffffffu;
if (val == 0) 
return 0;
for (i = 0; i < 25; i++)
if ((val & (mask << i)) == val)
return 1;
return 0;
}
static int
thumb_far_jump_used_p (void)
{
rtx_insn *insn;
bool far_jump = false;
unsigned int func_size = 0;
if (cfun->machine->far_jump_used)
return 1;
if (!(ARM_DOUBLEWORD_ALIGN || reload_completed))
{
if (df_regs_ever_live_p (ARG_POINTER_REGNUM))
cfun->machine->arg_pointer_live = 1;
else if (!cfun->machine->arg_pointer_live)
return 0;
}
if (reload_in_progress || reload_completed)
return 0;
for (insn = get_insns (); insn; insn = NEXT_INSN (insn))
{
if (JUMP_P (insn) && get_attr_far_jump (insn) == FAR_JUMP_YES)
{
far_jump = true;
}
func_size += get_attr_length (insn);
}
if (far_jump)
{
if ((func_size * 3) >= 2048)
{
cfun->machine->far_jump_used = 1;
return 1;
}
}
return 0;
}
static bool
is_called_in_ARM_mode (tree func)
{
gcc_assert (TREE_CODE (func) == FUNCTION_DECL);
if (TARGET_CALLEE_INTERWORKING && TREE_PUBLIC (func))
return true;
#ifdef ARM_PE
return lookup_attribute ("interfacearm", DECL_ATTRIBUTES (func)) != NULL_TREE;
#else
return false;
#endif
}
static int
thumb1_extra_regs_pushed (arm_stack_offsets *offsets, bool for_prologue)
{
HOST_WIDE_INT amount;
unsigned long live_regs_mask = offsets->saved_regs_mask;
unsigned long l_mask = live_regs_mask & (for_prologue ? 0x40ff : 0xff);
unsigned long high_regs_pushed = bit_count (live_regs_mask & 0x0f00);
int n_free, reg_base, size;
if (!for_prologue && frame_pointer_needed)
amount = offsets->locals_base - offsets->saved_regs;
else
amount = offsets->outgoing_args - offsets->saved_regs;
if (!optimize_size && amount != 512)
return 0;
if (high_regs_pushed != 0)
return 0;
if  (l_mask == 0
&& (for_prologue
|| TARGET_BACKTRACE
|| (live_regs_mask & 1 << LR_REGNUM) == 0
|| TARGET_INTERWORK
|| crtl->args.pretend_args_size != 0))
return 0;
if (for_prologue
&& ((flag_pic && arm_pic_register != INVALID_REGNUM)
|| (!frame_pointer_needed && CALLER_INTERWORKING_SLOT_SIZE > 0)))
return 0;
reg_base = 0;
n_free = 0;
if (!for_prologue)
{
size = arm_size_return_regs ();
reg_base = ARM_NUM_INTS (size);
live_regs_mask >>= reg_base;
}
while (reg_base + n_free < 8 && !(live_regs_mask & 1)
&& (for_prologue || call_used_regs[reg_base + n_free]))
{
live_regs_mask >>= 1;
n_free++;
}
if (n_free == 0)
return 0;
gcc_assert (amount / 4 * 4 == amount);
if (amount >= 512 && (amount - n_free * 4) < 512)
return (amount - 508) / 4;
if (amount <= n_free * 4)
return amount / 4;
return 0;
}
const char *
thumb1_unexpanded_epilogue (void)
{
arm_stack_offsets *offsets;
int regno;
unsigned long live_regs_mask = 0;
int high_regs_pushed = 0;
int extra_pop;
int had_to_push_lr;
int size;
if (cfun->machine->return_used_this_function != 0)
return "";
if (IS_NAKED (arm_current_func_type ()))
return "";
offsets = arm_get_frame_offsets ();
live_regs_mask = offsets->saved_regs_mask;
high_regs_pushed = bit_count (live_regs_mask & 0x0f00);
size = arm_size_return_regs ();
extra_pop = thumb1_extra_regs_pushed (offsets, false);
if (extra_pop > 0)
{
unsigned long extra_mask = (1 << extra_pop) - 1;
live_regs_mask |= extra_mask << ARM_NUM_INTS (size);
}
if (high_regs_pushed)
{
unsigned long mask = live_regs_mask & 0xff;
int next_hi_reg;
if (size <= 12)
mask |=  1 << 3;
if (size <= 8)
mask |= 1 << 2;
if (mask == 0)
internal_error
("no low registers available for popping high registers");
for (next_hi_reg = 8; next_hi_reg < 13; next_hi_reg++)
if (live_regs_mask & (1 << next_hi_reg))
break;
while (high_regs_pushed)
{
for (regno = 0; regno <= LAST_LO_REGNUM; regno++)
{
if (mask & (1 << regno))
high_regs_pushed--;
if (high_regs_pushed == 0)
break;
}
mask &= (2 << regno) - 1;	
thumb_pop (asm_out_file, mask);
for (regno = 0; regno <= LAST_LO_REGNUM; regno++)
{
if (mask & (1 << regno))
{
asm_fprintf (asm_out_file, "\tmov\t%r, %r\n", next_hi_reg,
regno);
for (next_hi_reg++; next_hi_reg < 13; next_hi_reg++)
if (live_regs_mask & (1 << next_hi_reg))
break;
}
}
}
live_regs_mask &= ~0x0f00;
}
had_to_push_lr = (live_regs_mask & (1 << LR_REGNUM)) != 0;
live_regs_mask &= 0xff;
if (crtl->args.pretend_args_size == 0 || TARGET_BACKTRACE)
{
if (had_to_push_lr)
live_regs_mask |= 1 << PC_REGNUM;
if (live_regs_mask)
thumb_pop (asm_out_file, live_regs_mask);
if (!had_to_push_lr)
thumb_exit (asm_out_file, LR_REGNUM);
}
else
{
if (live_regs_mask)
thumb_pop (asm_out_file, live_regs_mask);
if (had_to_push_lr)
{
if (size > 12)
{
asm_fprintf (asm_out_file, "\tmov\t%r, %r\n", IP_REGNUM,
LAST_ARG_REGNUM);
}
thumb_pop (asm_out_file, 1 << LAST_ARG_REGNUM);
if (size > 12)
{
asm_fprintf (asm_out_file, "\tmov\t%r, %r\n", LR_REGNUM,
LAST_ARG_REGNUM);
asm_fprintf (asm_out_file, "\tmov\t%r, %r\n", LAST_ARG_REGNUM,
IP_REGNUM);
regno = LR_REGNUM;
}
else
regno = LAST_ARG_REGNUM;
}
else
regno = LR_REGNUM;
asm_fprintf (asm_out_file, "\tadd\t%r, %r, #%d\n",
SP_REGNUM, SP_REGNUM,
crtl->args.pretend_args_size);
thumb_exit (asm_out_file, regno);
}
return "";
}
static struct machine_function *
arm_init_machine_status (void)
{
struct machine_function *machine;
machine = ggc_cleared_alloc<machine_function> ();
#if ARM_FT_UNKNOWN != 0
machine->func_type = ARM_FT_UNKNOWN;
#endif
machine->static_chain_stack_bytes = -1;
return machine;
}
rtx
arm_return_addr (int count, rtx frame ATTRIBUTE_UNUSED)
{
if (count != 0)
return NULL_RTX;
return get_hard_reg_initial_val (Pmode, LR_REGNUM);
}
void
arm_init_expanders (void)
{
init_machine_status = arm_init_machine_status;
if (cfun)
mark_reg_pointer (arg_pointer_rtx, PARM_BOUNDARY);
}
bool
arm_change_mode_p (tree func)
{
if (TREE_CODE (func) != FUNCTION_DECL)
return false;
tree callee_tree = DECL_FUNCTION_SPECIFIC_TARGET (func);
if (!callee_tree)
callee_tree = target_option_default_node;
struct cl_target_option *callee_opts = TREE_TARGET_OPTION (callee_tree);
int flags = callee_opts->x_target_flags;
return (TARGET_THUMB_P (flags) != TARGET_THUMB);
}
HOST_WIDE_INT
thumb_compute_initial_elimination_offset (unsigned int from, unsigned int to)
{
arm_stack_offsets *offsets;
offsets = arm_get_frame_offsets ();
switch (from)
{
case ARG_POINTER_REGNUM:
switch (to)
{
case STACK_POINTER_REGNUM:
return offsets->outgoing_args - offsets->saved_args;
case FRAME_POINTER_REGNUM:
return offsets->soft_frame - offsets->saved_args;
case ARM_HARD_FRAME_POINTER_REGNUM:
return offsets->saved_regs - offsets->saved_args;
case THUMB_HARD_FRAME_POINTER_REGNUM:
return offsets->locals_base - offsets->saved_args;
default:
gcc_unreachable ();
}
break;
case FRAME_POINTER_REGNUM:
switch (to)
{
case STACK_POINTER_REGNUM:
return offsets->outgoing_args - offsets->soft_frame;
case ARM_HARD_FRAME_POINTER_REGNUM:
return offsets->saved_regs - offsets->soft_frame;
case THUMB_HARD_FRAME_POINTER_REGNUM:
return offsets->locals_base - offsets->soft_frame;
default:
gcc_unreachable ();
}
break;
default:
gcc_unreachable ();
}
}
void
thumb1_expand_prologue (void)
{
rtx_insn *insn;
HOST_WIDE_INT amount;
HOST_WIDE_INT size;
arm_stack_offsets *offsets;
unsigned long func_type;
int regno;
unsigned long live_regs_mask;
unsigned long l_mask;
unsigned high_regs_pushed = 0;
bool lr_needs_saving;
func_type = arm_current_func_type ();
if (IS_NAKED (func_type))
{
if (flag_stack_usage_info)
current_function_static_stack_size = 0;
return;
}
if (IS_INTERRUPT (func_type))
{
error ("interrupt Service Routines cannot be coded in Thumb mode");
return;
}
if (is_called_in_ARM_mode (current_function_decl))
emit_insn (gen_prologue_thumb1_interwork ());
offsets = arm_get_frame_offsets ();
live_regs_mask = offsets->saved_regs_mask;
lr_needs_saving = live_regs_mask & (1 << LR_REGNUM);
l_mask = live_regs_mask & 0x40ff;
high_regs_pushed = bit_count (live_regs_mask & 0x0f00);
if (crtl->args.pretend_args_size)
{
rtx x = GEN_INT (-crtl->args.pretend_args_size);
if (cfun->machine->uses_anonymous_args)
{
int num_pushes = ARM_NUM_INTS (crtl->args.pretend_args_size);
unsigned long mask;
mask = 1ul << (LAST_ARG_REGNUM + 1);
mask -= 1ul << (LAST_ARG_REGNUM + 1 - num_pushes);
insn = thumb1_emit_multi_reg_push (mask, 0);
}
else
{
insn = emit_insn (gen_addsi3 (stack_pointer_rtx,
stack_pointer_rtx, x));
}
RTX_FRAME_RELATED_P (insn) = 1;
}
if (TARGET_BACKTRACE)
{
HOST_WIDE_INT offset = 0;
unsigned work_register;
rtx work_reg, x, arm_hfp_rtx;
work_register = thumb_find_work_register (live_regs_mask);
work_reg = gen_rtx_REG (SImode, work_register);
arm_hfp_rtx = gen_rtx_REG (SImode, ARM_HARD_FRAME_POINTER_REGNUM);
insn = emit_insn (gen_addsi3 (stack_pointer_rtx,
stack_pointer_rtx, GEN_INT (-16)));
RTX_FRAME_RELATED_P (insn) = 1;
if (l_mask)
{
insn = thumb1_emit_multi_reg_push (l_mask, l_mask);
RTX_FRAME_RELATED_P (insn) = 1;
lr_needs_saving = false;
offset = bit_count (l_mask) * UNITS_PER_WORD;
}
x = GEN_INT (offset + 16 + crtl->args.pretend_args_size);
emit_insn (gen_addsi3 (work_reg, stack_pointer_rtx, x));
x = plus_constant (Pmode, stack_pointer_rtx, offset + 4);
x = gen_frame_mem (SImode, x);
emit_move_insn (x, work_reg);
if (l_mask)
{
x = gen_rtx_REG (SImode, PC_REGNUM);
emit_move_insn (work_reg, x);
x = plus_constant (Pmode, stack_pointer_rtx, offset + 12);
x = gen_frame_mem (SImode, x);
emit_move_insn (x, work_reg);
emit_move_insn (work_reg, arm_hfp_rtx);
x = plus_constant (Pmode, stack_pointer_rtx, offset);
x = gen_frame_mem (SImode, x);
emit_move_insn (x, work_reg);
}
else
{
emit_move_insn (work_reg, arm_hfp_rtx);
x = plus_constant (Pmode, stack_pointer_rtx, offset);
x = gen_frame_mem (SImode, x);
emit_move_insn (x, work_reg);
x = gen_rtx_REG (SImode, PC_REGNUM);
emit_move_insn (work_reg, x);
x = plus_constant (Pmode, stack_pointer_rtx, offset + 12);
x = gen_frame_mem (SImode, x);
emit_move_insn (x, work_reg);
}
x = gen_rtx_REG (SImode, LR_REGNUM);
emit_move_insn (work_reg, x);
x = plus_constant (Pmode, stack_pointer_rtx, offset + 8);
x = gen_frame_mem (SImode, x);
emit_move_insn (x, work_reg);
x = GEN_INT (offset + 12);
emit_insn (gen_addsi3 (work_reg, stack_pointer_rtx, x));
emit_move_insn (arm_hfp_rtx, work_reg);
}
else if ((l_mask & 0xff) != 0
|| (high_regs_pushed == 0 && lr_needs_saving))
{
unsigned long mask = l_mask;
mask |= (1 << thumb1_extra_regs_pushed (offsets, true)) - 1;
insn = thumb1_emit_multi_reg_push (mask, mask);
RTX_FRAME_RELATED_P (insn) = 1;
lr_needs_saving = false;
}
if (high_regs_pushed)
{
unsigned pushable_regs;
unsigned next_hi_reg;
unsigned arg_regs_num = TARGET_AAPCS_BASED ? crtl->args.info.aapcs_ncrn
: crtl->args.info.nregs;
unsigned arg_regs_mask = (1 << arg_regs_num) - 1;
for (next_hi_reg = 12; next_hi_reg > LAST_LO_REGNUM; next_hi_reg--)
if (live_regs_mask & (1 << next_hi_reg))
break;
pushable_regs = l_mask & (~arg_regs_mask);
if (lr_needs_saving)
pushable_regs &= ~(1 << LR_REGNUM);
if (pushable_regs == 0)
pushable_regs = 1 << thumb_find_work_register (live_regs_mask);
while (high_regs_pushed > 0)
{
unsigned long real_regs_mask = 0;
unsigned long push_mask = 0;
for (regno = LR_REGNUM; regno >= 0; regno --)
{
if (pushable_regs & (1 << regno))
{
emit_move_insn (gen_rtx_REG (SImode, regno),
gen_rtx_REG (SImode, next_hi_reg));
high_regs_pushed --;
real_regs_mask |= (1 << next_hi_reg);
push_mask |= (1 << regno);
if (high_regs_pushed)
{
for (next_hi_reg --; next_hi_reg > LAST_LO_REGNUM;
next_hi_reg --)
if (live_regs_mask & (1 << next_hi_reg))
break;
}
else
break;
}
}
if (lr_needs_saving)
{
push_mask |= 1 << LR_REGNUM;
real_regs_mask |= 1 << LR_REGNUM;
lr_needs_saving = false;
}
insn = thumb1_emit_multi_reg_push (push_mask, real_regs_mask);
RTX_FRAME_RELATED_P (insn) = 1;
}
}
if (flag_pic && arm_pic_register != INVALID_REGNUM)
arm_load_pic_register (live_regs_mask);
if (!frame_pointer_needed && CALLER_INTERWORKING_SLOT_SIZE > 0)
emit_move_insn (gen_rtx_REG (Pmode, ARM_HARD_FRAME_POINTER_REGNUM),
stack_pointer_rtx);
size = offsets->outgoing_args - offsets->saved_args;
if (flag_stack_usage_info)
current_function_static_stack_size = size;
if ((flag_stack_check == STATIC_BUILTIN_STACK_CHECK
|| flag_stack_clash_protection)
&& size)
sorry ("-fstack-check=specific for Thumb-1");
amount = offsets->outgoing_args - offsets->saved_regs;
amount -= 4 * thumb1_extra_regs_pushed (offsets, true);
if (amount)
{
if (amount < 512)
{
insn = emit_insn (gen_addsi3 (stack_pointer_rtx, stack_pointer_rtx,
GEN_INT (- amount)));
RTX_FRAME_RELATED_P (insn) = 1;
}
else
{
rtx reg, dwarf;
for (regno = LAST_ARG_REGNUM + 1; regno <= LAST_LO_REGNUM; regno++)
if (live_regs_mask & (1 << regno))
break;
gcc_assert(regno <= LAST_LO_REGNUM);
reg = gen_rtx_REG (SImode, regno);
emit_insn (gen_movsi (reg, GEN_INT (- amount)));
insn = emit_insn (gen_addsi3 (stack_pointer_rtx,
stack_pointer_rtx, reg));
dwarf = gen_rtx_SET (stack_pointer_rtx,
plus_constant (Pmode, stack_pointer_rtx,
-amount));
add_reg_note (insn, REG_FRAME_RELATED_EXPR, dwarf);
RTX_FRAME_RELATED_P (insn) = 1;
}
}
if (frame_pointer_needed)
thumb_set_frame_pointer (offsets);
if (crtl->profile || !TARGET_SCHED_PROLOG
|| (arm_except_unwind_info (&global_options) == UI_TARGET
&& cfun->can_throw_non_call_exceptions))
emit_insn (gen_blockage ());
cfun->machine->lr_save_eliminated = !thumb_force_lr_save ();
if (live_regs_mask & 0xff)
cfun->machine->lr_save_eliminated = 0;
}
void
cmse_nonsecure_entry_clear_before_return (void)
{
int regno, maxregno = TARGET_HARD_FLOAT ? LAST_VFP_REGNUM : IP_REGNUM;
uint32_t padding_bits_to_clear = 0;
auto_sbitmap to_clear_bitmap (maxregno + 1);
rtx r1_reg, result_rtl, clearing_reg = NULL_RTX;
tree result_type;
bitmap_clear (to_clear_bitmap);
bitmap_set_range (to_clear_bitmap, R0_REGNUM, NUM_ARG_REGS);
bitmap_set_bit (to_clear_bitmap, IP_REGNUM);
if (TARGET_HARD_FLOAT)
{
int float_bits = D7_VFP_REGNUM - FIRST_VFP_REGNUM + 1;
bitmap_set_range (to_clear_bitmap, FIRST_VFP_REGNUM, float_bits);
emit_use (gen_rtx_REG (SImode, IP_REGNUM));
bitmap_clear_bit (to_clear_bitmap, IP_REGNUM);
emit_use (gen_rtx_REG (SImode, 4));
bitmap_clear_bit (to_clear_bitmap, 4);
}
for (regno = NUM_ARG_REGS; regno <= maxregno; regno++)
{
if (IN_RANGE (regno, FIRST_VFP_REGNUM, D7_VFP_REGNUM))
continue;
if (IN_RANGE (regno, IP_REGNUM, PC_REGNUM))
continue;
if (call_used_regs[regno])
bitmap_set_bit (to_clear_bitmap, regno);
}
result_type = TREE_TYPE (DECL_RESULT (current_function_decl));
if (!VOID_TYPE_P (result_type))
{
uint64_t to_clear_return_mask;
result_rtl = arm_function_value (result_type, current_function_decl, 0);
gcc_assert (REG_P (result_rtl));
to_clear_return_mask
= compute_not_to_clear_mask (result_type, result_rtl, 0,
&padding_bits_to_clear);
if (to_clear_return_mask)
{
gcc_assert ((unsigned) maxregno < sizeof (long long) * __CHAR_BIT__);
for (regno = R0_REGNUM; regno <= maxregno; regno++)
{
if (to_clear_return_mask & (1ULL << regno))
bitmap_clear_bit (to_clear_bitmap, regno);
}
}
}
if (padding_bits_to_clear != 0)
{
int to_clear_bitmap_size = SBITMAP_SIZE ((sbitmap) to_clear_bitmap);
auto_sbitmap to_clear_arg_regs_bitmap (to_clear_bitmap_size);
bitmap_clear (to_clear_arg_regs_bitmap);
bitmap_set_range (to_clear_arg_regs_bitmap, R1_REGNUM, NUM_ARG_REGS - 1);
gcc_assert (bitmap_subset_p (to_clear_arg_regs_bitmap, to_clear_bitmap));
}
clearing_reg = gen_rtx_REG (SImode, TARGET_THUMB1 ? R0_REGNUM : LR_REGNUM);
r1_reg = gen_rtx_REG (SImode, R0_REGNUM + 1);
cmse_clear_registers (to_clear_bitmap, &padding_bits_to_clear, 1, r1_reg,
clearing_reg);
}
void
thumb2_expand_return (bool simple_return)
{
int i, num_regs;
unsigned long saved_regs_mask;
arm_stack_offsets *offsets;
offsets = arm_get_frame_offsets ();
saved_regs_mask = offsets->saved_regs_mask;
for (i = 0, num_regs = 0; i <= LAST_ARM_REGNUM; i++)
if (saved_regs_mask & (1 << i))
num_regs++;
if (!simple_return && saved_regs_mask)
{
gcc_assert (!IS_CMSE_ENTRY (arm_current_func_type ()));
if (num_regs == 1)
{
rtx par = gen_rtx_PARALLEL (VOIDmode, rtvec_alloc (2));
rtx reg = gen_rtx_REG (SImode, PC_REGNUM);
rtx addr = gen_rtx_MEM (SImode,
gen_rtx_POST_INC (SImode,
stack_pointer_rtx));
set_mem_alias_set (addr, get_frame_alias_set ());
XVECEXP (par, 0, 0) = ret_rtx;
XVECEXP (par, 0, 1) = gen_rtx_SET (reg, addr);
RTX_FRAME_RELATED_P (XVECEXP (par, 0, 1)) = 1;
emit_jump_insn (par);
}
else
{
saved_regs_mask &= ~ (1 << LR_REGNUM);
saved_regs_mask |=   (1 << PC_REGNUM);
arm_emit_multi_reg_pop (saved_regs_mask);
}
}
else
{
if (IS_CMSE_ENTRY (arm_current_func_type ()))
cmse_nonsecure_entry_clear_before_return ();
emit_jump_insn (simple_return_rtx);
}
}
void
thumb1_expand_epilogue (void)
{
HOST_WIDE_INT amount;
arm_stack_offsets *offsets;
int regno;
if (IS_NAKED (arm_current_func_type ()))
return;
offsets = arm_get_frame_offsets ();
amount = offsets->outgoing_args - offsets->saved_regs;
if (frame_pointer_needed)
{
emit_insn (gen_movsi (stack_pointer_rtx, hard_frame_pointer_rtx));
amount = offsets->locals_base - offsets->saved_regs;
}
amount -= 4 * thumb1_extra_regs_pushed (offsets, false);
gcc_assert (amount >= 0);
if (amount)
{
emit_insn (gen_blockage ());
if (amount < 512)
emit_insn (gen_addsi3 (stack_pointer_rtx, stack_pointer_rtx,
GEN_INT (amount)));
else
{
rtx reg = gen_rtx_REG (SImode, LAST_ARG_REGNUM);
emit_insn (gen_movsi (reg, GEN_INT (amount)));
emit_insn (gen_addsi3 (stack_pointer_rtx, stack_pointer_rtx, reg));
}
}
emit_insn (gen_force_register_use (stack_pointer_rtx));
if (crtl->profile || !TARGET_SCHED_PROLOG)
emit_insn (gen_blockage ());
for (regno = 0; regno < 13; regno++)
if (df_regs_ever_live_p (regno) && !call_used_regs[regno])
emit_clobber (gen_rtx_REG (SImode, regno));
if (! df_regs_ever_live_p (LR_REGNUM))
emit_use (gen_rtx_REG (SImode, LR_REGNUM));
if (IS_CMSE_ENTRY (arm_current_func_type ()))
cmse_nonsecure_entry_clear_before_return ();
}
static void
arm_expand_epilogue_apcs_frame (bool really_return)
{
unsigned long func_type;
unsigned long saved_regs_mask;
int num_regs = 0;
int i;
int floats_from_frame = 0;
arm_stack_offsets *offsets;
gcc_assert (TARGET_APCS_FRAME && frame_pointer_needed && TARGET_ARM);
func_type = arm_current_func_type ();
offsets = arm_get_frame_offsets ();
saved_regs_mask = offsets->saved_regs_mask;
floats_from_frame
= (offsets->saved_args
+ arm_compute_static_chain_stack_bytes ()
- offsets->frame);
for (i = 0; i <= LAST_ARM_REGNUM; i++)
if (saved_regs_mask & (1 << i))
{
num_regs++;
floats_from_frame += 4;
}
if (TARGET_HARD_FLOAT)
{
int start_reg;
rtx ip_rtx = gen_rtx_REG (SImode, IP_REGNUM);
int saved_size = arm_get_vfp_saved_size ();
if (saved_size > 0)
{
rtx_insn *insn;
floats_from_frame += saved_size;
insn = emit_insn (gen_addsi3 (ip_rtx,
hard_frame_pointer_rtx,
GEN_INT (-floats_from_frame)));
arm_add_cfa_adjust_cfa_note (insn, -floats_from_frame,
ip_rtx, hard_frame_pointer_rtx);
}
start_reg = FIRST_VFP_REGNUM;
for (i = FIRST_VFP_REGNUM; i < LAST_VFP_REGNUM; i += 2)
if ((!df_regs_ever_live_p (i) || call_used_regs[i])
&& (!df_regs_ever_live_p (i + 1)
|| call_used_regs[i + 1]))
{
if (start_reg != i)
arm_emit_vfp_multi_reg_pop (start_reg,
(i - start_reg) / 2,
gen_rtx_REG (SImode,
IP_REGNUM));
start_reg = i + 2;
}
if (start_reg != i)
arm_emit_vfp_multi_reg_pop (start_reg,
(i - start_reg) / 2,
gen_rtx_REG (SImode, IP_REGNUM));
}
if (TARGET_IWMMXT)
{
rtx_insn *insn;
int lrm_count = (num_regs % 2) ? (num_regs + 2) : (num_regs + 1);
for (i = LAST_IWMMXT_REGNUM; i >= FIRST_IWMMXT_REGNUM; i--)
if (df_regs_ever_live_p (i) && !call_used_regs[i])
{
rtx addr = gen_frame_mem (V2SImode,
plus_constant (Pmode, hard_frame_pointer_rtx,
- lrm_count * 4));
insn = emit_insn (gen_movsi (gen_rtx_REG (V2SImode, i), addr));
REG_NOTES (insn) = alloc_reg_note (REG_CFA_RESTORE,
gen_rtx_REG (V2SImode, i),
NULL_RTX);
lrm_count += 2;
}
}
gcc_assert (saved_regs_mask & (1 << IP_REGNUM));
saved_regs_mask &= ~(1 << IP_REGNUM);
saved_regs_mask |= (1 << SP_REGNUM);
if (really_return
&& ARM_FUNC_TYPE (func_type) == ARM_FT_NORMAL
&& !crtl->calls_eh_return)
saved_regs_mask &= ~(1 << LR_REGNUM);
else
saved_regs_mask &= ~(1 << PC_REGNUM);
num_regs = bit_count (saved_regs_mask);
if ((offsets->outgoing_args != (1 + num_regs)) || cfun->calls_alloca)
{
rtx_insn *insn;
emit_insn (gen_blockage ());
insn = emit_insn (gen_addsi3 (stack_pointer_rtx,
hard_frame_pointer_rtx,
GEN_INT (- 4 * num_regs)));
arm_add_cfa_adjust_cfa_note (insn, - 4 * num_regs,
stack_pointer_rtx, hard_frame_pointer_rtx);
}
arm_emit_multi_reg_pop (saved_regs_mask);
if (IS_INTERRUPT (func_type))
{
rtx_insn *insn;
rtx addr = gen_rtx_MEM (SImode,
gen_rtx_POST_INC (SImode,
stack_pointer_rtx));
set_mem_alias_set (addr, get_frame_alias_set ());
insn = emit_insn (gen_movsi (gen_rtx_REG (SImode, IP_REGNUM), addr));
REG_NOTES (insn) = alloc_reg_note (REG_CFA_RESTORE,
gen_rtx_REG (SImode, IP_REGNUM),
NULL_RTX);
}
if (!really_return || (saved_regs_mask & (1 << PC_REGNUM)))
return;
if (crtl->calls_eh_return)
emit_insn (gen_addsi3 (stack_pointer_rtx,
stack_pointer_rtx,
gen_rtx_REG (SImode, ARM_EH_STACKADJ_REGNUM)));
if (IS_STACKALIGN (func_type))
emit_insn (gen_movsi (stack_pointer_rtx, gen_rtx_REG (SImode, R0_REGNUM)));
emit_jump_insn (simple_return_rtx);
}
void
arm_expand_epilogue (bool really_return)
{
unsigned long func_type;
unsigned long saved_regs_mask;
int num_regs = 0;
int i;
int amount;
arm_stack_offsets *offsets;
func_type = arm_current_func_type ();
if (IS_NAKED (func_type)
|| (IS_VOLATILE (func_type) && TARGET_ABORT_NORETURN))
{
if (really_return)
emit_jump_insn (simple_return_rtx);
return;
}
gcc_assert (!crtl->calls_eh_return || really_return);
if (TARGET_APCS_FRAME && frame_pointer_needed && TARGET_ARM)
{
arm_expand_epilogue_apcs_frame (really_return);
return;
}
offsets = arm_get_frame_offsets ();
saved_regs_mask = offsets->saved_regs_mask;
num_regs = bit_count (saved_regs_mask);
if (frame_pointer_needed)
{
rtx_insn *insn;
if (TARGET_ARM)
{
amount = offsets->frame - offsets->saved_regs;
emit_insn (gen_blockage ());
insn = emit_insn (gen_addsi3 (stack_pointer_rtx,
hard_frame_pointer_rtx,
GEN_INT (amount)));
arm_add_cfa_adjust_cfa_note (insn, amount,
stack_pointer_rtx,
hard_frame_pointer_rtx);
emit_insn (gen_force_register_use (stack_pointer_rtx));
}
else
{
amount = offsets->locals_base - offsets->saved_regs;
if (amount)
{
insn = emit_insn (gen_addsi3 (hard_frame_pointer_rtx,
hard_frame_pointer_rtx,
GEN_INT (amount)));
arm_add_cfa_adjust_cfa_note (insn, amount,
hard_frame_pointer_rtx,
hard_frame_pointer_rtx);
}
emit_insn (gen_blockage ());
insn = emit_insn (gen_movsi (stack_pointer_rtx,
hard_frame_pointer_rtx));
arm_add_cfa_adjust_cfa_note (insn, 0,
stack_pointer_rtx,
hard_frame_pointer_rtx);
emit_insn (gen_force_register_use (stack_pointer_rtx));
}
}
else
{
amount = offsets->outgoing_args - offsets->saved_regs;
if (amount)
{
rtx_insn *tmp;
emit_insn (gen_blockage ());
tmp = emit_insn (gen_addsi3 (stack_pointer_rtx,
stack_pointer_rtx,
GEN_INT (amount)));
arm_add_cfa_adjust_cfa_note (tmp, amount,
stack_pointer_rtx, stack_pointer_rtx);
emit_insn (gen_force_register_use (stack_pointer_rtx));
}
}
if (TARGET_HARD_FLOAT)
{
int end_reg = LAST_VFP_REGNUM + 1;
for (i = LAST_VFP_REGNUM - 1; i >= FIRST_VFP_REGNUM; i -= 2)
if ((!df_regs_ever_live_p (i) || call_used_regs[i])
&& (!df_regs_ever_live_p (i + 1)
|| call_used_regs[i + 1]))
{
if (end_reg > i + 2)
arm_emit_vfp_multi_reg_pop (i + 2,
(end_reg - (i + 2)) / 2,
stack_pointer_rtx);
end_reg = i;
}
if (end_reg > i + 2)
arm_emit_vfp_multi_reg_pop (i + 2,
(end_reg - (i + 2)) / 2,
stack_pointer_rtx);
}
if (TARGET_IWMMXT)
for (i = FIRST_IWMMXT_REGNUM; i <= LAST_IWMMXT_REGNUM; i++)
if (df_regs_ever_live_p (i) && !call_used_regs[i])
{
rtx_insn *insn;
rtx addr = gen_rtx_MEM (V2SImode,
gen_rtx_POST_INC (SImode,
stack_pointer_rtx));
set_mem_alias_set (addr, get_frame_alias_set ());
insn = emit_insn (gen_movsi (gen_rtx_REG (V2SImode, i), addr));
REG_NOTES (insn) = alloc_reg_note (REG_CFA_RESTORE,
gen_rtx_REG (V2SImode, i),
NULL_RTX);
arm_add_cfa_adjust_cfa_note (insn, UNITS_PER_WORD,
stack_pointer_rtx, stack_pointer_rtx);
}
if (saved_regs_mask)
{
rtx insn;
bool return_in_pc = false;
if (ARM_FUNC_TYPE (func_type) != ARM_FT_INTERWORKED
&& (TARGET_ARM || ARM_FUNC_TYPE (func_type) == ARM_FT_NORMAL)
&& !IS_CMSE_ENTRY (func_type)
&& !IS_STACKALIGN (func_type)
&& really_return
&& crtl->args.pretend_args_size == 0
&& saved_regs_mask & (1 << LR_REGNUM)
&& !crtl->calls_eh_return)
{
saved_regs_mask &= ~(1 << LR_REGNUM);
saved_regs_mask |= (1 << PC_REGNUM);
return_in_pc = true;
}
if (num_regs == 1 && (!IS_INTERRUPT (func_type) || !return_in_pc))
{
for (i = 0; i <= LAST_ARM_REGNUM; i++)
if (saved_regs_mask & (1 << i))
{
rtx addr = gen_rtx_MEM (SImode,
gen_rtx_POST_INC (SImode,
stack_pointer_rtx));
set_mem_alias_set (addr, get_frame_alias_set ());
if (i == PC_REGNUM)
{
insn = gen_rtx_PARALLEL (VOIDmode, rtvec_alloc (2));
XVECEXP (insn, 0, 0) = ret_rtx;
XVECEXP (insn, 0, 1) = gen_rtx_SET (gen_rtx_REG (SImode, i),
addr);
RTX_FRAME_RELATED_P (XVECEXP (insn, 0, 1)) = 1;
insn = emit_jump_insn (insn);
}
else
{
insn = emit_insn (gen_movsi (gen_rtx_REG (SImode, i),
addr));
REG_NOTES (insn) = alloc_reg_note (REG_CFA_RESTORE,
gen_rtx_REG (SImode, i),
NULL_RTX);
arm_add_cfa_adjust_cfa_note (insn, UNITS_PER_WORD,
stack_pointer_rtx,
stack_pointer_rtx);
}
}
}
else
{
if (TARGET_LDRD
&& current_tune->prefer_ldrd_strd
&& !optimize_function_for_size_p (cfun))
{
if (TARGET_THUMB2)
thumb2_emit_ldrd_pop (saved_regs_mask);
else if (TARGET_ARM && !IS_INTERRUPT (func_type))
arm_emit_ldrd_pop (saved_regs_mask);
else
arm_emit_multi_reg_pop (saved_regs_mask);
}
else
arm_emit_multi_reg_pop (saved_regs_mask);
}
if (return_in_pc)
return;
}
amount
= crtl->args.pretend_args_size + arm_compute_static_chain_stack_bytes();
if (amount)
{
int i, j;
rtx dwarf = NULL_RTX;
rtx_insn *tmp =
emit_insn (gen_addsi3 (stack_pointer_rtx,
stack_pointer_rtx,
GEN_INT (amount)));
RTX_FRAME_RELATED_P (tmp) = 1;
if (cfun->machine->uses_anonymous_args)
{
int num_regs = crtl->args.pretend_args_size / 4;
saved_regs_mask = (0xf0 >> num_regs) & 0xf;
for (j = 0, i = 0; j < num_regs; i++)
if (saved_regs_mask & (1 << i))
{
rtx reg = gen_rtx_REG (SImode, i);
dwarf = alloc_reg_note (REG_CFA_RESTORE, reg, dwarf);
j++;
}
REG_NOTES (tmp) = dwarf;
}
arm_add_cfa_adjust_cfa_note (tmp, amount,
stack_pointer_rtx, stack_pointer_rtx);
}
if (IS_CMSE_ENTRY (arm_current_func_type ()))
{
gcc_assert (really_return);
cmse_nonsecure_entry_clear_before_return ();
}
if (!really_return)
return;
if (crtl->calls_eh_return)
emit_insn (gen_addsi3 (stack_pointer_rtx,
stack_pointer_rtx,
gen_rtx_REG (SImode, ARM_EH_STACKADJ_REGNUM)));
if (IS_STACKALIGN (func_type))
emit_insn (gen_movsi (stack_pointer_rtx, gen_rtx_REG (SImode, R0_REGNUM)));
emit_jump_insn (simple_return_rtx);
}
const char *
thumb1_output_interwork (void)
{
const char * name;
FILE *f = asm_out_file;
gcc_assert (MEM_P (DECL_RTL (current_function_decl)));
gcc_assert (GET_CODE (XEXP (DECL_RTL (current_function_decl), 0))
== SYMBOL_REF);
name = XSTR (XEXP (DECL_RTL (current_function_decl), 0), 0);
asm_fprintf (f, "\torr\t%r, %r, #1\n", IP_REGNUM, PC_REGNUM);
asm_fprintf (f, "\tbx\t%r\n", IP_REGNUM);
#define STUB_NAME ".real_start_of"
fprintf (f, "\t.code\t16\n");
#ifdef ARM_PE
if (arm_dllexport_name_p (name))
name = arm_strip_name_encoding (name);
#endif
asm_fprintf (f, "\t.globl %s%U%s\n", STUB_NAME, name);
fprintf (f, "\t.thumb_func\n");
asm_fprintf (f, "%s%U%s:\n", STUB_NAME, name);
return "";
}
const char *
thumb_load_double_from_address (rtx *operands)
{
rtx addr;
rtx base;
rtx offset;
rtx arg1;
rtx arg2;
gcc_assert (REG_P (operands[0]));
gcc_assert (MEM_P (operands[1]));
addr = XEXP (operands[1], 0);
switch (GET_CODE (addr))
{
case REG:
operands[2] = adjust_address (operands[1], SImode, 4);
if (REGNO (operands[0]) == REGNO (addr))
{
output_asm_insn ("ldr\t%H0, %2", operands);
output_asm_insn ("ldr\t%0, %1", operands);
}
else
{
output_asm_insn ("ldr\t%0, %1", operands);
output_asm_insn ("ldr\t%H0, %2", operands);
}
break;
case CONST:
operands[2] = adjust_address (operands[1], SImode, 4);
output_asm_insn ("ldr\t%0, %1", operands);
output_asm_insn ("ldr\t%H0, %2", operands);
break;
case PLUS:
arg1   = XEXP (addr, 0);
arg2   = XEXP (addr, 1);
if (CONSTANT_P (arg1))
base = arg2, offset = arg1;
else
base = arg1, offset = arg2;
gcc_assert (REG_P (base));
if (REG_P (offset))
{
int reg_offset = REGNO (offset);
int reg_base   = REGNO (base);
int reg_dest   = REGNO (operands[0]);
asm_fprintf (asm_out_file, "\tadd\t%r, %r, %r",
reg_dest + 1, reg_base, reg_offset);
asm_fprintf (asm_out_file, "\tldr\t%r, [%r, #0]",
reg_dest, reg_dest + 1);
asm_fprintf (asm_out_file, "\tldr\t%r, [%r, #4]",
reg_dest + 1, reg_dest + 1);
}
else
{
operands[2] = adjust_address (operands[1], SImode, 4);
if (REGNO (operands[0]) == REGNO (base))
{
output_asm_insn ("ldr\t%H0, %2", operands);
output_asm_insn ("ldr\t%0, %1", operands);
}
else
{
output_asm_insn ("ldr\t%0, %1", operands);
output_asm_insn ("ldr\t%H0, %2", operands);
}
}
break;
case LABEL_REF:
operands[2] = adjust_address (operands[1], SImode, 4);
output_asm_insn ("ldr\t%H0, %2", operands);
output_asm_insn ("ldr\t%0, %1", operands);
break;
default:
gcc_unreachable ();
}
return "";
}
const char *
thumb_output_move_mem_multiple (int n, rtx *operands)
{
switch (n)
{
case 2:
if (REGNO (operands[4]) > REGNO (operands[5]))
std::swap (operands[4], operands[5]);
output_asm_insn ("ldmia\t%1!, {%4, %5}", operands);
output_asm_insn ("stmia\t%0!, {%4, %5}", operands);
break;
case 3:
if (REGNO (operands[4]) > REGNO (operands[5]))
std::swap (operands[4], operands[5]);
if (REGNO (operands[5]) > REGNO (operands[6]))
std::swap (operands[5], operands[6]);
if (REGNO (operands[4]) > REGNO (operands[5]))
std::swap (operands[4], operands[5]);
output_asm_insn ("ldmia\t%1!, {%4, %5, %6}", operands);
output_asm_insn ("stmia\t%0!, {%4, %5, %6}", operands);
break;
default:
gcc_unreachable ();
}
return "";
}
const char *
thumb_call_via_reg (rtx reg)
{
int regno = REGNO (reg);
rtx *labelp;
gcc_assert (regno < LR_REGNUM);
if (in_section == text_section)
{
thumb_call_reg_needed = 1;
if (thumb_call_via_label[regno] == NULL)
thumb_call_via_label[regno] = gen_label_rtx ();
labelp = thumb_call_via_label + regno;
}
else
{
if (cfun->machine->call_via[regno] == NULL)
cfun->machine->call_via[regno] = gen_label_rtx ();
labelp = cfun->machine->call_via + regno;
}
output_asm_insn ("bl\t%a0", labelp);
return "";
}
void
thumb_expand_movmemqi (rtx *operands)
{
rtx out = copy_to_mode_reg (SImode, XEXP (operands[0], 0));
rtx in  = copy_to_mode_reg (SImode, XEXP (operands[1], 0));
HOST_WIDE_INT len = INTVAL (operands[2]);
HOST_WIDE_INT offset = 0;
while (len >= 12)
{
emit_insn (gen_movmem12b (out, in, out, in));
len -= 12;
}
if (len >= 8)
{
emit_insn (gen_movmem8b (out, in, out, in));
len -= 8;
}
if (len >= 4)
{
rtx reg = gen_reg_rtx (SImode);
emit_insn (gen_movsi (reg, gen_rtx_MEM (SImode, in)));
emit_insn (gen_movsi (gen_rtx_MEM (SImode, out), reg));
len -= 4;
offset += 4;
}
if (len >= 2)
{
rtx reg = gen_reg_rtx (HImode);
emit_insn (gen_movhi (reg, gen_rtx_MEM (HImode,
plus_constant (Pmode, in,
offset))));
emit_insn (gen_movhi (gen_rtx_MEM (HImode, plus_constant (Pmode, out,
offset)),
reg));
len -= 2;
offset += 2;
}
if (len)
{
rtx reg = gen_reg_rtx (QImode);
emit_insn (gen_movqi (reg, gen_rtx_MEM (QImode,
plus_constant (Pmode, in,
offset))));
emit_insn (gen_movqi (gen_rtx_MEM (QImode, plus_constant (Pmode, out,
offset)),
reg));
}
}
void
thumb_reload_out_hi (rtx *operands)
{
emit_insn (gen_thumb_movhi_clobber (operands[0], operands[1], operands[2]));
}
static int
arm_get_strip_length (int c)
{
switch (c)
{
ARM_NAME_ENCODING_LENGTHS
default: return 0;
}
}
const char *
arm_strip_name_encoding (const char *name)
{
int skip;
while ((skip = arm_get_strip_length (* name)))
name += skip;
return name;
}
void
arm_asm_output_labelref (FILE *stream, const char *name)
{
int skip;
int verbatim = 0;
while ((skip = arm_get_strip_length (* name)))
{
verbatim |= (*name == '*');
name += skip;
}
if (verbatim)
fputs (name, stream);
else
asm_fprintf (stream, "%U%s", name);
}
void
arm_emit_eabi_attribute (const char *name, int num, int val)
{
asm_fprintf (asm_out_file, "\t.eabi_attribute %d, %d", num, val);
if (flag_verbose_asm || flag_debug_asm)
asm_fprintf (asm_out_file, "\t%s %s", ASM_COMMENT_START, name);
asm_fprintf (asm_out_file, "\n");
}
void
arm_print_tune_info (void)
{
asm_fprintf (asm_out_file, "\t" ASM_COMMENT_START ".tune parameters\n");
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START "constant_limit:\t%d\n",
current_tune->constant_limit);
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START
"max_insns_skipped:\t%d\n", current_tune->max_insns_skipped);
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START
"prefetch.num_slots:\t%d\n", current_tune->prefetch.num_slots);
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START
"prefetch.l1_cache_size:\t%d\n",
current_tune->prefetch.l1_cache_size);
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START
"prefetch.l1_cache_line_size:\t%d\n",
current_tune->prefetch.l1_cache_line_size);
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START
"prefer_constant_pool:\t%d\n",
(int) current_tune->prefer_constant_pool);
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START
"branch_cost:\t(s:speed, p:predictable)\n");
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START "\t\ts&p\tcost\n");
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START "\t\t00\t%d\n",
current_tune->branch_cost (false, false));
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START "\t\t01\t%d\n",
current_tune->branch_cost (false, true));
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START "\t\t10\t%d\n",
current_tune->branch_cost (true, false));
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START "\t\t11\t%d\n",
current_tune->branch_cost (true, true));
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START
"prefer_ldrd_strd:\t%d\n",
(int) current_tune->prefer_ldrd_strd);
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START
"logical_op_non_short_circuit:\t[%d,%d]\n",
(int) current_tune->logical_op_non_short_circuit_thumb,
(int) current_tune->logical_op_non_short_circuit_arm);
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START
"prefer_neon_for_64bits:\t%d\n",
(int) current_tune->prefer_neon_for_64bits);
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START
"disparage_flag_setting_t16_encodings:\t%d\n",
(int) current_tune->disparage_flag_setting_t16_encodings);
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START
"string_ops_prefer_neon:\t%d\n",
(int) current_tune->string_ops_prefer_neon);
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START
"max_insns_inline_memset:\t%d\n",
current_tune->max_insns_inline_memset);
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START "fusible_ops:\t%u\n",
current_tune->fusible_ops);
asm_fprintf (asm_out_file, "\t\t" ASM_COMMENT_START "sched_autopref:\t%d\n",
(int) current_tune->sched_autopref);
}
static void
arm_print_asm_arch_directives ()
{
const arch_option *arch
= arm_parse_arch_option_name (all_architectures, "-march",
arm_active_target.arch_name);
auto_sbitmap opt_bits (isa_num_bits);
gcc_assert (arch);
asm_fprintf (asm_out_file, "\t.arch %s\n", arm_active_target.arch_name);
arm_last_printed_arch_string = arm_active_target.arch_name;
if (!arch->common.extensions)
return;
for (const struct cpu_arch_extension *opt = arch->common.extensions;
opt->name != NULL;
opt++)
{
if (!opt->remove)
{
arm_initialize_isa (opt_bits, opt->isa_bits);
if (bitmap_subset_p (opt_bits, arm_active_target.isa)
&& !bitmap_subset_p (opt_bits, isa_all_fpubits))
asm_fprintf (asm_out_file, "\t.arch_extension %s\n", opt->name);
}
}
}
static void
arm_file_start (void)
{
int val;
if (TARGET_BPABI)
{
if (!arm_active_target.core_name)
{
if (strcmp (arm_active_target.arch_name, "armv7ve") == 0)
{
asm_fprintf (asm_out_file, "\t.arch armv7-a\n");
asm_fprintf (asm_out_file, "\t.arch_extension virt\n");
asm_fprintf (asm_out_file, "\t.arch_extension idiv\n");
asm_fprintf (asm_out_file, "\t.arch_extension sec\n");
asm_fprintf (asm_out_file, "\t.arch_extension mp\n");
arm_last_printed_arch_string = "armv7ve";
}
else
arm_print_asm_arch_directives ();
}
else if (strncmp (arm_active_target.core_name, "generic", 7) == 0)
{
asm_fprintf (asm_out_file, "\t.arch %s\n",
arm_active_target.core_name + 8);
arm_last_printed_arch_string = arm_active_target.core_name + 8;
}
else
{
const char* truncated_name
= arm_rewrite_selected_cpu (arm_active_target.core_name);
asm_fprintf (asm_out_file, "\t.cpu %s\n", truncated_name);
}
if (print_tune_info)
arm_print_tune_info ();
if (! TARGET_SOFT_FLOAT)
{
if (TARGET_HARD_FLOAT && TARGET_VFP_SINGLE)
arm_emit_eabi_attribute ("Tag_ABI_HardFP_use", 27, 1);
if (TARGET_HARD_FLOAT_ABI)
arm_emit_eabi_attribute ("Tag_ABI_VFP_args", 28, 1);
}
if (flag_rounding_math)
arm_emit_eabi_attribute ("Tag_ABI_FP_rounding", 19, 1);
if (!flag_unsafe_math_optimizations)
{
arm_emit_eabi_attribute ("Tag_ABI_FP_denormal", 20, 1);
arm_emit_eabi_attribute ("Tag_ABI_FP_exceptions", 21, 1);
}
if (flag_signaling_nans)
arm_emit_eabi_attribute ("Tag_ABI_FP_user_exceptions", 22, 1);
arm_emit_eabi_attribute ("Tag_ABI_FP_number_model", 23,
flag_finite_math_only ? 1 : 3);
arm_emit_eabi_attribute ("Tag_ABI_align8_needed", 24, 1);
arm_emit_eabi_attribute ("Tag_ABI_align8_preserved", 25, 1);
arm_emit_eabi_attribute ("Tag_ABI_enum_size", 26,
flag_short_enums ? 1 : 2);
if (optimize_size)
val = 4;
else if (optimize >= 2)
val = 2;
else if (optimize)
val = 1;
else
val = 6;
arm_emit_eabi_attribute ("Tag_ABI_optimization_goals", 30, val);
arm_emit_eabi_attribute ("Tag_CPU_unaligned_access", 34,
unaligned_access);
if (arm_fp16_format)
arm_emit_eabi_attribute ("Tag_ABI_FP_16bit_format", 38,
(int) arm_fp16_format);
if (arm_lang_output_object_attributes_hook)
arm_lang_output_object_attributes_hook();
}
default_file_start ();
}
static void
arm_file_end (void)
{
int regno;
if (NEED_INDICATE_EXEC_STACK)
file_end_indicate_exec_stack ();
if (! thumb_call_reg_needed)
return;
switch_to_section (text_section);
asm_fprintf (asm_out_file, "\t.code 16\n");
ASM_OUTPUT_ALIGN (asm_out_file, 1);
for (regno = 0; regno < LR_REGNUM; regno++)
{
rtx label = thumb_call_via_label[regno];
if (label != 0)
{
targetm.asm_out.internal_label (asm_out_file, "L",
CODE_LABEL_NUMBER (label));
asm_fprintf (asm_out_file, "\tbx\t%r\n", regno);
}
}
}
#ifndef ARM_PE
static void
arm_encode_section_info (tree decl, rtx rtl, int first)
{
if (optimize > 0 && TREE_CONSTANT (decl))
SYMBOL_REF_FLAG (XEXP (rtl, 0)) = 1;
default_encode_section_info (decl, rtl, first);
}
#endif 
static void
arm_internal_label (FILE *stream, const char *prefix, unsigned long labelno)
{
if (arm_ccfsm_state == 3 && (unsigned) arm_target_label == labelno
&& !strcmp (prefix, "L"))
{
arm_ccfsm_state = 0;
arm_target_insn = NULL;
}
default_internal_label (stream, prefix, labelno);
}
static void
arm_thumb1_mi_thunk (FILE *file, tree, HOST_WIDE_INT delta,
HOST_WIDE_INT, tree function)
{
static int thunk_label = 0;
char label[256];
char labelpc[256];
int mi_delta = delta;
const char *const mi_op = mi_delta < 0 ? "sub" : "add";
int shift = 0;
int this_regno = (aggregate_value_p (TREE_TYPE (TREE_TYPE (function)), function)
? 1 : 0);
if (mi_delta < 0)
mi_delta = - mi_delta;
final_start_function (emit_barrier (), file, 1);
if (TARGET_THUMB1)
{
int labelno = thunk_label++;
ASM_GENERATE_INTERNAL_LABEL (label, "LTHUMBFUNC", labelno);
if (TARGET_THUMB1_ONLY)
{
fputs ("\tpush {r3}\n", file);
fputs ("\tldr\tr3, ", file);
}
else
{
fputs ("\tldr\tr12, ", file);
}
assemble_name (file, label);
fputc ('\n', file);
if (flag_pic)
{
ASM_GENERATE_INTERNAL_LABEL (labelpc, "LTHUNKPC", labelno);
assemble_name (file, labelpc);
fputs (":\n", file);
if (TARGET_THUMB1_ONLY)
{
fputs ("\tadd\tr3, pc, r3\n", file);
fputs ("\tmov r12, r3\n", file);
}
else
fputs ("\tadd\tr12, pc, r12\n", file);
}
else if (TARGET_THUMB1_ONLY)
fputs ("\tmov r12, r3\n", file);
}
if (TARGET_THUMB1_ONLY)
{
if (mi_delta > 255)
{
fputs ("\tldr\tr3, ", file);
assemble_name (file, label);
fputs ("+4\n", file);
asm_fprintf (file, "\t%ss\t%r, %r, r3\n",
mi_op, this_regno, this_regno);
}
else if (mi_delta != 0)
{
asm_fprintf (file, "\t%ss\t%r, %r, #%d\n",
mi_op, this_regno, this_regno,
mi_delta);
}
}
else
{
while (mi_delta != 0)
{
if ((mi_delta & (3 << shift)) == 0)
shift += 2;
else
{
asm_fprintf (file, "\t%s\t%r, %r, #%d\n",
mi_op, this_regno, this_regno,
mi_delta & (0xff << shift));
mi_delta &= ~(0xff << shift);
shift += 8;
}
}
}
if (TARGET_THUMB1)
{
if (TARGET_THUMB1_ONLY)
fputs ("\tpop\t{r3}\n", file);
fprintf (file, "\tbx\tr12\n");
ASM_OUTPUT_ALIGN (file, 2);
assemble_name (file, label);
fputs (":\n", file);
if (flag_pic)
{
rtx tem = XEXP (DECL_RTL (function), 0);
tem = plus_constant (GET_MODE (tem), tem,
TARGET_THUMB1_ONLY ? -3 : -7);
tem = gen_rtx_MINUS (GET_MODE (tem),
tem,
gen_rtx_SYMBOL_REF (Pmode,
ggc_strdup (labelpc)));
assemble_integer (tem, 4, BITS_PER_WORD, 1);
}
else
assemble_integer (XEXP (DECL_RTL (function), 0), 4, BITS_PER_WORD, 1);
if (TARGET_THUMB1_ONLY && mi_delta > 255)
assemble_integer (GEN_INT(mi_delta), 4, BITS_PER_WORD, 1);
}
else
{
fputs ("\tb\t", file);
assemble_name (file, XSTR (XEXP (DECL_RTL (function), 0), 0));
if (NEED_PLT_RELOC)
fputs ("(PLT)", file);
fputc ('\n', file);
}
final_end_function ();
}
static void
arm32_output_mi_thunk (FILE *file, tree, HOST_WIDE_INT delta,
HOST_WIDE_INT vcall_offset, tree function)
{
const bool long_call_p = arm_is_long_call_p (function);
int this_regno = (aggregate_value_p (TREE_TYPE (TREE_TYPE (function)),
function)
? R1_REGNUM : R0_REGNUM);
rtx temp = gen_rtx_REG (Pmode, IP_REGNUM);
rtx this_rtx = gen_rtx_REG (Pmode, this_regno);
reload_completed = 1;
emit_note (NOTE_INSN_PROLOGUE_END);
if (delta != 0)
arm_split_constant (PLUS, Pmode, NULL_RTX,
delta, this_rtx, this_rtx, false);
if (vcall_offset != 0)
{
emit_move_insn (temp, gen_rtx_MEM (Pmode, this_rtx));
arm_split_constant (PLUS, Pmode, NULL_RTX, vcall_offset, temp, temp,
false);
emit_move_insn (temp, gen_rtx_MEM (Pmode, temp));
emit_insn (gen_add3_insn (this_rtx, this_rtx, temp));
}
if (!TREE_USED (function))
{
assemble_external (function);
TREE_USED (function) = 1;
}
rtx funexp = XEXP (DECL_RTL (function), 0);
if (long_call_p)
{
emit_move_insn (temp, funexp);
funexp = temp;
}
funexp = gen_rtx_MEM (FUNCTION_MODE, funexp);
rtx_insn *insn = emit_call_insn (gen_sibcall (funexp, const0_rtx, NULL_RTX));
SIBLING_CALL_P (insn) = 1;
emit_barrier ();
if (long_call_p)
{
split_all_insns_noflow ();
arm_reorg ();
}
insn = get_insns ();
shorten_branches (insn);
final_start_function (insn, file, 1);
final (insn, file, 1);
final_end_function ();
reload_completed = 0;
}
static void
arm_output_mi_thunk (FILE *file, tree thunk, HOST_WIDE_INT delta,
HOST_WIDE_INT vcall_offset, tree function)
{
if (TARGET_32BIT)
arm32_output_mi_thunk (file, thunk, delta, vcall_offset, function);
else
arm_thumb1_mi_thunk (file, thunk, delta, vcall_offset, function);
}
int
arm_emit_vector_const (FILE *file, rtx x)
{
int i;
const char * pattern;
gcc_assert (GET_CODE (x) == CONST_VECTOR);
switch (GET_MODE (x))
{
case E_V2SImode: pattern = "%08x"; break;
case E_V4HImode: pattern = "%04x"; break;
case E_V8QImode: pattern = "%02x"; break;
default:       gcc_unreachable ();
}
fprintf (file, "0x");
for (i = CONST_VECTOR_NUNITS (x); i--;)
{
rtx element;
element = CONST_VECTOR_ELT (x, i);
fprintf (file, pattern, INTVAL (element));
}
return 1;
}
void
arm_emit_fp16_const (rtx c)
{
long bits;
bits = real_to_target (NULL, CONST_DOUBLE_REAL_VALUE (c), HFmode);
if (WORDS_BIG_ENDIAN)
assemble_zeros (2);
assemble_integer (GEN_INT (bits), 2, BITS_PER_WORD, 1);
if (!WORDS_BIG_ENDIAN)
assemble_zeros (2);
}
const char *
arm_output_load_gr (rtx *operands)
{
rtx reg;
rtx offset;
rtx wcgr;
rtx sum;
if (!MEM_P (operands [1])
|| GET_CODE (sum = XEXP (operands [1], 0)) != PLUS
|| !REG_P (reg = XEXP (sum, 0))
|| !CONST_INT_P (offset = XEXP (sum, 1))
|| ((INTVAL (offset) < 1024) && (INTVAL (offset) > -1024)))
return "wldrw%?\t%0, %1";
output_asm_insn ("str%?\t%0, [sp, #-4]!\t@ Start of GR load expansion", & reg);
wcgr = operands[0];
operands[0] = reg;
output_asm_insn ("ldr%?\t%0, %1", operands);
operands[0] = wcgr;
operands[1] = reg;
output_asm_insn ("tmcr%?\t%0, %1", operands);
output_asm_insn ("ldr%?\t%0, [sp], #4\t@ End of GR load expansion", & reg);
return "";
}
static void
arm_setup_incoming_varargs (cumulative_args_t pcum_v,
machine_mode mode,
tree type,
int *pretend_size,
int second_time ATTRIBUTE_UNUSED)
{
CUMULATIVE_ARGS *pcum = get_cumulative_args (pcum_v);
int nregs;
cfun->machine->uses_anonymous_args = 1;
if (pcum->pcs_variant <= ARM_PCS_AAPCS_LOCAL)
{
nregs = pcum->aapcs_ncrn;
if (nregs & 1)
{
int res = arm_needs_doubleword_align (mode, type);
if (res < 0 && warn_psabi)
inform (input_location, "parameter passing for argument of "
"type %qT changed in GCC 7.1", type);
else if (res > 0)
nregs++;
}
}
else
nregs = pcum->nregs;
if (nregs < NUM_ARG_REGS)
*pretend_size = (NUM_ARG_REGS - nregs) * UNITS_PER_WORD;
}
static bool
arm_promote_prototypes (const_tree t ATTRIBUTE_UNUSED)
{
return !TARGET_AAPCS_BASED;
}
static machine_mode
arm_promote_function_mode (const_tree type ATTRIBUTE_UNUSED,
machine_mode mode,
int *punsignedp ATTRIBUTE_UNUSED,
const_tree fntype ATTRIBUTE_UNUSED,
int for_return ATTRIBUTE_UNUSED)
{
if (GET_MODE_CLASS (mode) == MODE_INT
&& GET_MODE_SIZE (mode) < 4)
return SImode;
return mode;
}
static bool
arm_default_short_enums (void)
{
return ARM_DEFAULT_SHORT_ENUMS;
}
static bool
arm_align_anon_bitfield (void)
{
return TARGET_AAPCS_BASED;
}
static tree
arm_cxx_guard_type (void)
{
return TARGET_AAPCS_BASED ? integer_type_node : long_long_integer_type_node;
}
static bool
arm_cxx_guard_mask_bit (void)
{
return TARGET_AAPCS_BASED;
}
static tree
arm_get_cookie_size (tree type)
{
tree size;
if (!TARGET_AAPCS_BASED)
return default_cxx_get_cookie_size (type);
size = build_int_cst (sizetype, 8);
return size;
}
static bool
arm_cookie_has_size (void)
{
return TARGET_AAPCS_BASED;
}
static bool
arm_cxx_cdtor_returns_this (void)
{
return TARGET_AAPCS_BASED;
}
static bool
arm_cxx_key_method_may_be_inline (void)
{
return !TARGET_AAPCS_BASED;
}
static void
arm_cxx_determine_class_data_visibility (tree decl)
{
if (!TARGET_AAPCS_BASED
|| !TARGET_DLLIMPORT_DECL_ATTRIBUTES)
return;
if (!TARGET_ARM_DYNAMIC_VAGUE_LINKAGE_P && DECL_COMDAT (decl))
DECL_VISIBILITY (decl) = VISIBILITY_HIDDEN;
else
DECL_VISIBILITY (decl) = VISIBILITY_DEFAULT;
DECL_VISIBILITY_SPECIFIED (decl) = 1;
}
static bool
arm_cxx_class_data_always_comdat (void)
{
return !TARGET_AAPCS_BASED;
}
static bool
arm_cxx_use_aeabi_atexit (void)
{
return TARGET_AAPCS_BASED;
}
void
arm_set_return_address (rtx source, rtx scratch)
{
arm_stack_offsets *offsets;
HOST_WIDE_INT delta;
rtx addr, mem;
unsigned long saved_regs;
offsets = arm_get_frame_offsets ();
saved_regs = offsets->saved_regs_mask;
if ((saved_regs & (1 << LR_REGNUM)) == 0)
emit_move_insn (gen_rtx_REG (Pmode, LR_REGNUM), source);
else
{
if (frame_pointer_needed)
addr = plus_constant (Pmode, hard_frame_pointer_rtx, -4);
else
{
delta = offsets->outgoing_args - (offsets->frame + 4);
if (delta >= 4096)
{
emit_insn (gen_addsi3 (scratch, stack_pointer_rtx,
GEN_INT (delta & ~4095)));
addr = scratch;
delta &= 4095;
}
else
addr = stack_pointer_rtx;
addr = plus_constant (Pmode, addr, delta);
}
mem = gen_frame_mem (Pmode, addr);
MEM_VOLATILE_P (mem) = true;
emit_move_insn (mem, source);
}
}
void
thumb_set_return_address (rtx source, rtx scratch)
{
arm_stack_offsets *offsets;
HOST_WIDE_INT delta;
HOST_WIDE_INT limit;
int reg;
rtx addr, mem;
unsigned long mask;
emit_use (source);
offsets = arm_get_frame_offsets ();
mask = offsets->saved_regs_mask;
if (mask & (1 << LR_REGNUM))
{
limit = 1024;
if (frame_pointer_needed)
{
delta = offsets->soft_frame - offsets->saved_args;
reg = THUMB_HARD_FRAME_POINTER_REGNUM;
if (TARGET_THUMB1)
limit = 128;
}
else
{
delta = offsets->outgoing_args - offsets->saved_args;
reg = SP_REGNUM;
}
if (TARGET_THUMB1 && TARGET_BACKTRACE)
delta -= 16;
delta -= 4;
addr = gen_rtx_REG (SImode, reg);
if (delta > limit)
{
emit_insn (gen_movsi (scratch, GEN_INT (delta)));
emit_insn (gen_addsi3 (scratch, scratch, stack_pointer_rtx));
addr = scratch;
}
else
addr = plus_constant (Pmode, addr, delta);
mem = gen_frame_mem (Pmode, addr);
MEM_VOLATILE_P (mem) = true;
emit_move_insn (mem, source);
}
else
emit_move_insn (gen_rtx_REG (Pmode, LR_REGNUM), source);
}
bool
arm_vector_mode_supported_p (machine_mode mode)
{
if (TARGET_NEON && (mode == V2SFmode || mode == V4SImode || mode == V8HImode
|| mode == V4HFmode || mode == V16QImode || mode == V4SFmode
|| mode == V2DImode || mode == V8HFmode))
return true;
if ((TARGET_NEON || TARGET_IWMMXT)
&& ((mode == V2SImode)
|| (mode == V4HImode)
|| (mode == V8QImode)))
return true;
if (TARGET_INT_SIMD && (mode == V4UQQmode || mode == V4QQmode
|| mode == V2UHQmode || mode == V2HQmode || mode == V2UHAmode
|| mode == V2HAmode))
return true;
return false;
}
static bool
arm_array_mode_supported_p (machine_mode mode,
unsigned HOST_WIDE_INT nelems)
{
if (TARGET_NEON && !BYTES_BIG_ENDIAN
&& (VALID_NEON_DREG_MODE (mode) || VALID_NEON_QREG_MODE (mode))
&& (nelems >= 2 && nelems <= 4))
return true;
return false;
}
static machine_mode
arm_preferred_simd_mode (scalar_mode mode)
{
if (TARGET_NEON)
switch (mode)
{
case E_SFmode:
return TARGET_NEON_VECTORIZE_DOUBLE ? V2SFmode : V4SFmode;
case E_SImode:
return TARGET_NEON_VECTORIZE_DOUBLE ? V2SImode : V4SImode;
case E_HImode:
return TARGET_NEON_VECTORIZE_DOUBLE ? V4HImode : V8HImode;
case E_QImode:
return TARGET_NEON_VECTORIZE_DOUBLE ? V8QImode : V16QImode;
case E_DImode:
if (!TARGET_NEON_VECTORIZE_DOUBLE)
return V2DImode;
break;
default:;
}
if (TARGET_REALLY_IWMMXT)
switch (mode)
{
case E_SImode:
return V2SImode;
case E_HImode:
return V4HImode;
case E_QImode:
return V8QImode;
default:;
}
return word_mode;
}
static bool
arm_class_likely_spilled_p (reg_class_t rclass)
{
if ((TARGET_THUMB1 && rclass == LO_REGS)
|| rclass  == CC_REG)
return true;
return false;
}
bool
arm_small_register_classes_for_mode_p (machine_mode mode ATTRIBUTE_UNUSED)
{
return TARGET_THUMB1;
}
static unsigned HOST_WIDE_INT
arm_shift_truncation_mask (machine_mode mode)
{
return mode == SImode ? 255 : 0;
}
unsigned int
arm_dbx_register_number (unsigned int regno)
{
if (regno < 16)
return regno;
if (IS_VFP_REGNUM (regno))
{
if (VFP_REGNO_OK_FOR_SINGLE (regno))
return 64 + regno - FIRST_VFP_REGNUM;
else
return 256 + (regno - FIRST_VFP_REGNUM) / 2;
}
if (IS_IWMMXT_GR_REGNUM (regno))
return 104 + regno - FIRST_IWMMXT_GR_REGNUM;
if (IS_IWMMXT_REGNUM (regno))
return 112 + regno - FIRST_IWMMXT_REGNUM;
return DWARF_FRAME_REGISTERS;
}
static rtx
arm_dwarf_register_span (rtx rtl)
{
machine_mode mode;
unsigned regno;
rtx parts[16];
int nregs;
int i;
regno = REGNO (rtl);
if (!IS_VFP_REGNUM (regno))
return NULL_RTX;
mode = GET_MODE (rtl);
if (GET_MODE_SIZE (mode) < 8)
return NULL_RTX;
if (VFP_REGNO_OK_FOR_SINGLE (regno))
{
nregs = GET_MODE_SIZE (mode) / 4;
for (i = 0; i < nregs; i += 2)
if (TARGET_BIG_END)
{
parts[i] = gen_rtx_REG (SImode, regno + i + 1);
parts[i + 1] = gen_rtx_REG (SImode, regno + i);
}
else
{
parts[i] = gen_rtx_REG (SImode, regno + i);
parts[i + 1] = gen_rtx_REG (SImode, regno + i + 1);
}
}
else
{
nregs = GET_MODE_SIZE (mode) / 8;
for (i = 0; i < nregs; i++)
parts[i] = gen_rtx_REG (DImode, regno + i);
}
return gen_rtx_PARALLEL (VOIDmode, gen_rtvec_v (nregs , parts));
}
#if ARM_UNWIND_INFO
static void
arm_unwind_emit_sequence (FILE * asm_out_file, rtx p)
{
int i;
HOST_WIDE_INT offset;
HOST_WIDE_INT nregs;
int reg_size;
unsigned reg;
unsigned lastreg;
unsigned padfirst = 0, padlast = 0;
rtx e;
e = XVECEXP (p, 0, 0);
gcc_assert (GET_CODE (e) == SET);
gcc_assert (GET_CODE (e) == SET
&& REG_P (SET_DEST (e))
&& REGNO (SET_DEST (e)) == SP_REGNUM
&& GET_CODE (SET_SRC (e)) == PLUS);
offset = -INTVAL (XEXP (SET_SRC (e), 1));
nregs = XVECLEN (p, 0) - 1;
gcc_assert (nregs);
reg = REGNO (SET_SRC (XVECEXP (p, 0, 1)));
if (reg < 16)
{
e = XVECEXP (p, 0, 1);
e = XEXP (SET_DEST (e), 0);
if (GET_CODE (e) == PLUS)
padfirst = INTVAL (XEXP (e, 1));
gcc_assert (padfirst == 0 || optimize_size);
e = XVECEXP (p, 0, nregs);
e = XEXP (SET_DEST (e), 0);
if (GET_CODE (e) == PLUS)
padlast = offset - INTVAL (XEXP (e, 1)) - 4;
else
padlast = offset - 4;
gcc_assert (padlast == 0 || padlast == 4);
if (padlast == 4)
fprintf (asm_out_file, "\t.pad #4\n");
reg_size = 4;
fprintf (asm_out_file, "\t.save {");
}
else if (IS_VFP_REGNUM (reg))
{
reg_size = 8;
fprintf (asm_out_file, "\t.vsave {");
}
else
gcc_unreachable ();
gcc_assert (offset == padfirst + nregs * reg_size + padlast);
offset = padfirst;
lastreg = 0;
for (i = 1; i <= nregs; i++)
{
e = XVECEXP (p, 0, i);
gcc_assert (GET_CODE (e) == SET
&& MEM_P (SET_DEST (e))
&& REG_P (SET_SRC (e)));
reg = REGNO (SET_SRC (e));
gcc_assert (reg >= lastreg);
if (i != 1)
fprintf (asm_out_file, ", ");
if (IS_VFP_REGNUM (reg))
asm_fprintf (asm_out_file, "d%d", (reg - FIRST_VFP_REGNUM) / 2);
else
asm_fprintf (asm_out_file, "%r", reg);
if (flag_checking)
{
e = XEXP (SET_DEST (e), 0);
if (GET_CODE (e) == PLUS)
gcc_assert (REG_P (XEXP (e, 0))
&& REGNO (XEXP (e, 0)) == SP_REGNUM
&& CONST_INT_P (XEXP (e, 1))
&& offset == INTVAL (XEXP (e, 1)));
else
gcc_assert (i == 1
&& REG_P (e)
&& REGNO (e) == SP_REGNUM);
offset += reg_size;
}
}
fprintf (asm_out_file, "}\n");
if (padfirst)
fprintf (asm_out_file, "\t.pad #%d\n", padfirst);
}
static void
arm_unwind_emit_set (FILE * asm_out_file, rtx p)
{
rtx e0;
rtx e1;
unsigned reg;
e0 = XEXP (p, 0);
e1 = XEXP (p, 1);
switch (GET_CODE (e0))
{
case MEM:
if (GET_CODE (XEXP (e0, 0)) != PRE_DEC
|| !REG_P (XEXP (XEXP (e0, 0), 0))
|| REGNO (XEXP (XEXP (e0, 0), 0)) != SP_REGNUM)
abort ();
asm_fprintf (asm_out_file, "\t.save ");
if (IS_VFP_REGNUM (REGNO (e1)))
asm_fprintf(asm_out_file, "{d%d}\n",
(REGNO (e1) - FIRST_VFP_REGNUM) / 2);
else
asm_fprintf(asm_out_file, "{%r}\n", REGNO (e1));
break;
case REG:
if (REGNO (e0) == SP_REGNUM)
{
if (GET_CODE (e1) != PLUS
|| !REG_P (XEXP (e1, 0))
|| REGNO (XEXP (e1, 0)) != SP_REGNUM
|| !CONST_INT_P (XEXP (e1, 1)))
abort ();
asm_fprintf (asm_out_file, "\t.pad #%wd\n",
-INTVAL (XEXP (e1, 1)));
}
else if (REGNO (e0) == HARD_FRAME_POINTER_REGNUM)
{
HOST_WIDE_INT offset;
if (GET_CODE (e1) == PLUS)
{
if (!REG_P (XEXP (e1, 0))
|| !CONST_INT_P (XEXP (e1, 1)))
abort ();
reg = REGNO (XEXP (e1, 0));
offset = INTVAL (XEXP (e1, 1));
asm_fprintf (asm_out_file, "\t.setfp %r, %r, #%wd\n",
HARD_FRAME_POINTER_REGNUM, reg,
offset);
}
else if (REG_P (e1))
{
reg = REGNO (e1);
asm_fprintf (asm_out_file, "\t.setfp %r, %r\n",
HARD_FRAME_POINTER_REGNUM, reg);
}
else
abort ();
}
else if (REG_P (e1) && REGNO (e1) == SP_REGNUM)
{
asm_fprintf (asm_out_file, "\t.movsp %r\n", REGNO (e0));
}
else if (GET_CODE (e1) == PLUS
&& REG_P (XEXP (e1, 0))
&& REGNO (XEXP (e1, 0)) == SP_REGNUM
&& CONST_INT_P (XEXP (e1, 1)))
{
asm_fprintf (asm_out_file, "\t.movsp %r, #%d\n",
REGNO (e0), (int)INTVAL(XEXP (e1, 1)));
}
else
abort ();
break;
default:
abort ();
}
}
static void
arm_unwind_emit (FILE * asm_out_file, rtx_insn *insn)
{
rtx note, pat;
bool handled_one = false;
if (arm_except_unwind_info (&global_options) != UI_TARGET)
return;
if (!(flag_unwind_tables || crtl->uses_eh_lsda)
&& (TREE_NOTHROW (current_function_decl)
|| crtl->all_throwers_are_sibcalls))
return;
if (NOTE_P (insn) || !RTX_FRAME_RELATED_P (insn))
return;
for (note = REG_NOTES (insn); note ; note = XEXP (note, 1))
{
switch (REG_NOTE_KIND (note))
{
case REG_FRAME_RELATED_EXPR:
pat = XEXP (note, 0);
goto found;
case REG_CFA_REGISTER:
pat = XEXP (note, 0);
if (pat == NULL)
{
pat = PATTERN (insn);
if (GET_CODE (pat) == PARALLEL)
pat = XVECEXP (pat, 0, 0);
}
{
rtx dest, src;
unsigned reg;
src = SET_SRC (pat);
dest = SET_DEST (pat);
gcc_assert (src == stack_pointer_rtx);
reg = REGNO (dest);
asm_fprintf (asm_out_file, "\t.unwind_raw 0, 0x%x @ vsp = r%d\n",
reg + 0x90, reg);
}
handled_one = true;
break;
case REG_CFA_DEF_CFA:
case REG_CFA_ADJUST_CFA:
case REG_CFA_RESTORE:
return;
case REG_CFA_EXPRESSION:
case REG_CFA_OFFSET:
gcc_unreachable ();
default:
break;
}
}
if (handled_one)
return;
pat = PATTERN (insn);
found:
switch (GET_CODE (pat))
{
case SET:
arm_unwind_emit_set (asm_out_file, pat);
break;
case SEQUENCE:
arm_unwind_emit_sequence (asm_out_file, pat);
break;
default:
abort();
}
}
static bool
arm_output_ttype (rtx x)
{
fputs ("\t.word\t", asm_out_file);
output_addr_const (asm_out_file, x);
if (!CONST_INT_P (x))
fputs ("(TARGET2)", asm_out_file);
fputc ('\n', asm_out_file);
return TRUE;
}
static void
arm_asm_emit_except_personality (rtx personality)
{
fputs ("\t.personality\t", asm_out_file);
output_addr_const (asm_out_file, personality);
fputc ('\n', asm_out_file);
}
#endif 
static void
arm_asm_init_sections (void)
{
#if ARM_UNWIND_INFO
exception_section = get_unnamed_section (0, output_section_asm_op,
"\t.handlerdata");
#endif 
#ifdef OBJECT_FORMAT_ELF
if (target_pure_code)
text_section->unnamed.data = "\t.section .text,\"0x20000006\",%progbits";
#endif
}
void
arm_output_fn_unwind (FILE * f, bool prologue)
{
if (arm_except_unwind_info (&global_options) != UI_TARGET)
return;
if (prologue)
fputs ("\t.fnstart\n", f);
else
{
if (!(flag_unwind_tables || crtl->uses_eh_lsda)
&& (TREE_NOTHROW (current_function_decl)
|| crtl->all_throwers_are_sibcalls))
fputs("\t.cantunwind\n", f);
fputs ("\t.fnend\n", f);
}
}
static bool
arm_emit_tls_decoration (FILE *fp, rtx x)
{
enum tls_reloc reloc;
rtx val;
val = XVECEXP (x, 0, 0);
reloc = (enum tls_reloc) INTVAL (XVECEXP (x, 0, 1));
output_addr_const (fp, val);
switch (reloc)
{
case TLS_GD32:
fputs ("(tlsgd)", fp);
break;
case TLS_LDM32:
fputs ("(tlsldm)", fp);
break;
case TLS_LDO32:
fputs ("(tlsldo)", fp);
break;
case TLS_IE32:
fputs ("(gottpoff)", fp);
break;
case TLS_LE32:
fputs ("(tpoff)", fp);
break;
case TLS_DESCSEQ:
fputs ("(tlsdesc)", fp);
break;
default:
gcc_unreachable ();
}
switch (reloc)
{
case TLS_GD32:
case TLS_LDM32:
case TLS_IE32:
case TLS_DESCSEQ:
fputs (" + (. - ", fp);
output_addr_const (fp, XVECEXP (x, 0, 2));
fputs (reloc == TLS_DESCSEQ ? " + " : " - ", fp);
output_addr_const (fp, XVECEXP (x, 0, 3));
fputc (')', fp);
break;
default:
break;
}
return TRUE;
}
static void
arm_output_dwarf_dtprel (FILE *file, int size, rtx x)
{
gcc_assert (size == 4);
fputs ("\t.word\t", file);
output_addr_const (file, x);
fputs ("(tlsldo)", file);
}
static bool
arm_output_addr_const_extra (FILE *fp, rtx x)
{
if (GET_CODE (x) == UNSPEC && XINT (x, 1) == UNSPEC_TLS)
return arm_emit_tls_decoration (fp, x);
else if (GET_CODE (x) == UNSPEC && XINT (x, 1) == UNSPEC_PIC_LABEL)
{
char label[256];
int labelno = INTVAL (XVECEXP (x, 0, 0));
ASM_GENERATE_INTERNAL_LABEL (label, "LPIC", labelno);
assemble_name_raw (fp, label);
return TRUE;
}
else if (GET_CODE (x) == UNSPEC && XINT (x, 1) == UNSPEC_GOTSYM_OFF)
{
assemble_name (fp, "_GLOBAL_OFFSET_TABLE_");
if (GOT_PCREL)
fputs ("+.", fp);
fputs ("-(", fp);
output_addr_const (fp, XVECEXP (x, 0, 0));
fputc (')', fp);
return TRUE;
}
else if (GET_CODE (x) == UNSPEC && XINT (x, 1) == UNSPEC_SYMBOL_OFFSET)
{
output_addr_const (fp, XVECEXP (x, 0, 0));
if (GOT_PCREL)
fputs ("+.", fp);
fputs ("-(", fp);
output_addr_const (fp, XVECEXP (x, 0, 1));
fputc (')', fp);
return TRUE;
}
else if (GET_CODE (x) == CONST_VECTOR)
return arm_emit_vector_const (fp, x);
return FALSE;
}
const char *
arm_output_shift(rtx * operands, int set_flags)
{
char pattern[100];
static const char flag_chars[3] = {'?', '.', '!'};
const char *shift;
HOST_WIDE_INT val;
char c;
c = flag_chars[set_flags];
shift = shift_op(operands[3], &val);
if (shift)
{
if (val != -1)
operands[2] = GEN_INT(val);
sprintf (pattern, "%s%%%c\t%%0, %%1, %%2", shift, c);
}
else
sprintf (pattern, "mov%%%c\t%%0, %%1", c);
output_asm_insn (pattern, operands);
return "";
}
const char *
arm_output_iwmmxt_shift_immediate (const char *insn_name, rtx *operands, bool wror_or_wsra)
{
int shift = INTVAL (operands[2]);
char templ[50];
machine_mode opmode = GET_MODE (operands[0]);
gcc_assert (shift >= 0);
if (((opmode == V4HImode) && (shift > 15))
|| ((opmode == V2SImode) && (shift > 31))
|| ((opmode == DImode) && (shift > 63)))
{
if (wror_or_wsra)
{
sprintf (templ, "%s\t%%0, %%1, #%d", insn_name, 32);
output_asm_insn (templ, operands);
if (opmode == DImode)
{
sprintf (templ, "%s\t%%0, %%0, #%d", insn_name, 32);
output_asm_insn (templ, operands);
}
}
else
{
sprintf (templ, "wzero\t%%0");
output_asm_insn (templ, operands);
}
return "";
}
if ((opmode == DImode) && (shift > 32))
{
sprintf (templ, "%s\t%%0, %%1, #%d", insn_name, 32);
output_asm_insn (templ, operands);
sprintf (templ, "%s\t%%0, %%0, #%d", insn_name, shift - 32);
output_asm_insn (templ, operands);
}
else
{
sprintf (templ, "%s\t%%0, %%1, #%d", insn_name, shift);
output_asm_insn (templ, operands);
}
return "";
}
const char *
arm_output_iwmmxt_tinsr (rtx *operands)
{
int mask = INTVAL (operands[3]);
int i;
char templ[50];
int units = mode_nunits[GET_MODE (operands[0])];
gcc_assert ((mask & (mask - 1)) == 0);
for (i = 0; i < units; ++i)
{
if ((mask & 0x01) == 1)
{
break;
}
mask >>= 1;
}
gcc_assert (i < units);
{
switch (GET_MODE (operands[0]))
{
case E_V8QImode:
sprintf (templ, "tinsrb%%?\t%%0, %%2, #%d", i);
break;
case E_V4HImode:
sprintf (templ, "tinsrh%%?\t%%0, %%2, #%d", i);
break;
case E_V2SImode:
sprintf (templ, "tinsrw%%?\t%%0, %%2, #%d", i);
break;
default:
gcc_unreachable ();
break;
}
output_asm_insn (templ, operands);
}
return "";
}
const char *
thumb1_output_casesi (rtx *operands)
{
rtx diff_vec = PATTERN (NEXT_INSN (as_a <rtx_insn *> (operands[0])));
gcc_assert (GET_CODE (diff_vec) == ADDR_DIFF_VEC);
switch (GET_MODE(diff_vec))
{
case E_QImode:
return (ADDR_DIFF_VEC_FLAGS (diff_vec).offset_unsigned ?
"bl\t%___gnu_thumb1_case_uqi" : "bl\t%___gnu_thumb1_case_sqi");
case E_HImode:
return (ADDR_DIFF_VEC_FLAGS (diff_vec).offset_unsigned ?
"bl\t%___gnu_thumb1_case_uhi" : "bl\t%___gnu_thumb1_case_shi");
case E_SImode:
return "bl\t%___gnu_thumb1_case_si";
default:
gcc_unreachable ();
}
}
const char *
thumb2_output_casesi (rtx *operands)
{
rtx diff_vec = PATTERN (NEXT_INSN (as_a <rtx_insn *> (operands[2])));
gcc_assert (GET_CODE (diff_vec) == ADDR_DIFF_VEC);
output_asm_insn ("cmp\t%0, %1", operands);
output_asm_insn ("bhi\t%l3", operands);
switch (GET_MODE(diff_vec))
{
case E_QImode:
return "tbb\t[%|pc, %0]";
case E_HImode:
return "tbh\t[%|pc, %0, lsl #1]";
case E_SImode:
if (flag_pic)
{
output_asm_insn ("adr\t%4, %l2", operands);
output_asm_insn ("ldr\t%5, [%4, %0, lsl #2]", operands);
output_asm_insn ("add\t%4, %4, %5", operands);
return "bx\t%4";
}
else
{
output_asm_insn ("adr\t%4, %l2", operands);
return "ldr\t%|pc, [%4, %0, lsl #2]";
}
default:
gcc_unreachable ();
}
}
static int
arm_issue_rate (void)
{
return current_tune->issue_rate;
}
static int
arm_first_cycle_multipass_dfa_lookahead (void)
{
int issue_rate = arm_issue_rate ();
return issue_rate > 1 && !sched_fusion ? issue_rate : 0;
}
static int
arm_first_cycle_multipass_dfa_lookahead_guard (rtx_insn *insn, int ready_index)
{
return autopref_multipass_dfa_lookahead_guard (insn, ready_index);
}
const char *
arm_mangle_type (const_tree type)
{
if (TARGET_AAPCS_BASED
&& lang_hooks.types_compatible_p (CONST_CAST_TREE (type), va_list_type))
return "St9__va_list";
if (TREE_CODE (type) == REAL_TYPE && TYPE_PRECISION (type) == 16)
return "Dh";
if (TYPE_NAME (type) != NULL)
return arm_mangle_builtin_type (type);
return NULL;
}
static const int thumb_core_reg_alloc_order[] =
{
3,  2,  1,  0,  4,  5,  6,  7,
12, 14,  8,  9, 10, 11
};
void
arm_order_regs_for_local_alloc (void)
{
const int arm_reg_alloc_order[] = REG_ALLOC_ORDER;
memcpy(reg_alloc_order, arm_reg_alloc_order, sizeof (reg_alloc_order));
if (TARGET_THUMB)
memcpy (reg_alloc_order, thumb_core_reg_alloc_order,
sizeof (thumb_core_reg_alloc_order));
}
bool
arm_frame_pointer_required (void)
{
if (SUBTARGET_FRAME_POINTER_REQUIRED)
return true;
if (cfun->has_nonlocal_label)
return true;
if (TARGET_ARM && TARGET_APCS_FRAME && !crtl->is_leaf)
return true;
if (!IS_INTERRUPT (arm_current_func_type ())
&& (flag_stack_check == STATIC_BUILTIN_STACK_CHECK
|| flag_stack_clash_protection)
&& arm_except_unwind_info (&global_options) == UI_TARGET
&& cfun->can_throw_non_call_exceptions)
{
HOST_WIDE_INT size = get_frame_size ();
if (size <= 0)
return false;
if (crtl->is_leaf && !cfun->calls_alloca)
{
size += 32 * UNITS_PER_WORD;
if (size > PROBE_INTERVAL && size > get_stack_check_protect ())
return true;
}
else
return true;
}
return false;
}
static bool
arm_have_conditional_execution (void)
{
return !TARGET_THUMB1;
}
static HOST_WIDE_INT
arm_vector_alignment (const_tree type)
{
HOST_WIDE_INT align = tree_to_shwi (TYPE_SIZE (type));
if (TARGET_AAPCS_BASED)
align = MIN (align, 64);
return align;
}
static void
arm_autovectorize_vector_sizes (vector_sizes *sizes)
{
if (!TARGET_NEON_VECTORIZE_DOUBLE)
{
sizes->safe_push (16);
sizes->safe_push (8);
}
}
static bool
arm_vector_alignment_reachable (const_tree type, bool is_packed)
{
if (TARGET_NEON && !BYTES_BIG_ENDIAN && unaligned_access)
return !is_packed;
return default_builtin_vector_alignment_reachable (type, is_packed);
}
static bool
arm_builtin_support_vector_misalignment (machine_mode mode,
const_tree type, int misalignment,
bool is_packed)
{
if (TARGET_NEON && !BYTES_BIG_ENDIAN && unaligned_access)
{
HOST_WIDE_INT align = TYPE_ALIGN_UNIT (type);
if (is_packed)
return align == 1;
if (misalignment == -1)
return true;
return ((misalignment % align) == 0);
}
return default_builtin_support_vector_misalignment (mode, type, misalignment,
is_packed);
}
static void
arm_conditional_register_usage (void)
{
int regno;
if (TARGET_THUMB1 && optimize_size)
{
for (regno = FIRST_HI_REGNUM; regno <= LAST_HI_REGNUM; ++regno)
fixed_regs[regno] = call_used_regs[regno] = 1;
}
if (TARGET_THUMB1)
fixed_regs[LR_REGNUM] = call_used_regs[LR_REGNUM] = 1;
if (TARGET_32BIT && TARGET_HARD_FLOAT)
{
for (regno = FIRST_VFP_REGNUM;
regno <= LAST_VFP_REGNUM; ++ regno)
{
fixed_regs[regno] = 0;
call_used_regs[regno] = regno < FIRST_VFP_REGNUM + 16
|| regno >= FIRST_VFP_REGNUM + 32;
}
}
if (TARGET_REALLY_IWMMXT)
{
regno = FIRST_IWMMXT_GR_REGNUM;
for (regno = FIRST_IWMMXT_GR_REGNUM;
regno <= LAST_IWMMXT_GR_REGNUM; ++ regno)
fixed_regs[regno] = 0;
for (regno = FIRST_IWMMXT_REGNUM;
regno <= LAST_IWMMXT_REGNUM; ++ regno)
{
fixed_regs[regno] = 0;
call_used_regs[regno] = regno < FIRST_IWMMXT_REGNUM + 10;
}
}
if ((unsigned) PIC_OFFSET_TABLE_REGNUM != INVALID_REGNUM)
{
fixed_regs[PIC_OFFSET_TABLE_REGNUM] = 1;
call_used_regs[PIC_OFFSET_TABLE_REGNUM] = 1;
}
else if (TARGET_APCS_STACK)
{
fixed_regs[10]     = 1;
call_used_regs[10] = 1;
}
if (TARGET_APCS_FRAME || TARGET_CALLER_INTERWORKING
|| TARGET_TPCS_FRAME || TARGET_TPCS_LEAF_FRAME)
{
fixed_regs[ARM_HARD_FRAME_POINTER_REGNUM] = 1;
call_used_regs[ARM_HARD_FRAME_POINTER_REGNUM] = 1;
if (TARGET_CALLER_INTERWORKING)
global_regs[ARM_HARD_FRAME_POINTER_REGNUM] = 1;
}
SUBTARGET_CONDITIONAL_REGISTER_USAGE
}
static reg_class_t
arm_preferred_rename_class (reg_class_t rclass)
{
if (TARGET_THUMB2 && rclass == GENERAL_REGS)
return LO_REGS;
else
return NO_REGS;
}
int
arm_attr_length_push_multi(rtx parallel_op, rtx first_op)
{
int i, regno, hi_reg;
int num_saves = XVECLEN (parallel_op, 0);
if (TARGET_ARM)
return 4;
if (TARGET_THUMB1)
return 2;
regno = REGNO (first_op);
hi_reg = (REGNO_REG_CLASS (regno) == HI_REGS) && (regno != LR_REGNUM);
for (i = 1; i < num_saves && !hi_reg; i++)
{
regno = REGNO (XEXP (XVECEXP (parallel_op, 0, i), 0));
hi_reg |= (REGNO_REG_CLASS (regno) == HI_REGS) && (regno != LR_REGNUM);
}
if (!hi_reg)
return 2;
return 4;
}
int
arm_attr_length_pop_multi (rtx *operands, bool return_pc, bool write_back_p)
{
if (TARGET_ARM)
return 4;
if (TARGET_THUMB1)
return 2;
rtx parallel_op = operands[0];
unsigned indx = XVECLEN (parallel_op, 0) - 1;
unsigned regno = REGNO (operands[1]);
unsigned first_indx = 0;
first_indx += return_pc ? 1 : 0;
first_indx += write_back_p ? 1 : 0;
bool pop_p = (regno == SP_REGNUM && write_back_p);
bool ldm_p = !pop_p;
if (ldm_p && REGNO_REG_CLASS (regno) == HI_REGS)
return 4;
for (; indx >= first_indx; indx--)
{
regno = REGNO (XEXP (XVECEXP (parallel_op, 0, indx), 0));
if (REGNO_REG_CLASS (regno) == HI_REGS
&& (regno != PC_REGNUM || ldm_p))
return 4;
}
return 2;
}
int
arm_count_output_move_double_insns (rtx *operands)
{
int count;
rtx ops[2];
ops[0] = operands[0];
ops[1] = operands[1];
output_move_double (ops, false, &count);
return count;
}
int
vfp3_const_double_for_fract_bits (rtx operand)
{
REAL_VALUE_TYPE r0;
if (!CONST_DOUBLE_P (operand))
return 0;
r0 = *CONST_DOUBLE_REAL_VALUE (operand);
if (exact_real_inverse (DFmode, &r0)
&& !REAL_VALUE_NEGATIVE (r0))
{
if (exact_real_truncate (DFmode, &r0))
{
HOST_WIDE_INT value = real_to_integer (&r0);
value = value & 0xffffffff;
if ((value != 0) && ( (value & (value - 1)) == 0))
{
int ret = exact_log2 (value);
gcc_assert (IN_RANGE (ret, 0, 31));
return ret;
}
}
}
return 0;
}
int
vfp3_const_double_for_bits (rtx x)
{
const REAL_VALUE_TYPE *r;
if (!CONST_DOUBLE_P (x))
return -1;
r = CONST_DOUBLE_REAL_VALUE (x);
if (REAL_VALUE_NEGATIVE (*r)
|| REAL_VALUE_ISNAN (*r)
|| REAL_VALUE_ISINF (*r)
|| !real_isinteger (r, SFmode))
return -1;
HOST_WIDE_INT hwint = exact_log2 (real_to_integer (r));
if (!IN_RANGE (hwint, 1, 32))
return -1;
return hwint;
}

static void
arm_pre_atomic_barrier (enum memmodel model)
{
if (need_atomic_barrier_p (model, true))
emit_insn (gen_memory_barrier ());
}
static void
arm_post_atomic_barrier (enum memmodel model)
{
if (need_atomic_barrier_p (model, false))
emit_insn (gen_memory_barrier ());
}
static void
arm_emit_load_exclusive (machine_mode mode, rtx rval, rtx mem, bool acq)
{
rtx (*gen) (rtx, rtx);
if (acq)
{
switch (mode)
{
case E_QImode: gen = gen_arm_load_acquire_exclusiveqi; break;
case E_HImode: gen = gen_arm_load_acquire_exclusivehi; break;
case E_SImode: gen = gen_arm_load_acquire_exclusivesi; break;
case E_DImode: gen = gen_arm_load_acquire_exclusivedi; break;
default:
gcc_unreachable ();
}
}
else
{
switch (mode)
{
case E_QImode: gen = gen_arm_load_exclusiveqi; break;
case E_HImode: gen = gen_arm_load_exclusivehi; break;
case E_SImode: gen = gen_arm_load_exclusivesi; break;
case E_DImode: gen = gen_arm_load_exclusivedi; break;
default:
gcc_unreachable ();
}
}
emit_insn (gen (rval, mem));
}
static void
arm_emit_store_exclusive (machine_mode mode, rtx bval, rtx rval,
rtx mem, bool rel)
{
rtx (*gen) (rtx, rtx, rtx);
if (rel)
{
switch (mode)
{
case E_QImode: gen = gen_arm_store_release_exclusiveqi; break;
case E_HImode: gen = gen_arm_store_release_exclusivehi; break;
case E_SImode: gen = gen_arm_store_release_exclusivesi; break;
case E_DImode: gen = gen_arm_store_release_exclusivedi; break;
default:
gcc_unreachable ();
}
}
else
{
switch (mode)
{
case E_QImode: gen = gen_arm_store_exclusiveqi; break;
case E_HImode: gen = gen_arm_store_exclusivehi; break;
case E_SImode: gen = gen_arm_store_exclusivesi; break;
case E_DImode: gen = gen_arm_store_exclusivedi; break;
default:
gcc_unreachable ();
}
}
emit_insn (gen (bval, rval, mem));
}
static void
emit_unlikely_jump (rtx insn)
{
rtx_insn *jump = emit_jump_insn (insn);
add_reg_br_prob_note (jump, profile_probability::very_unlikely ());
}
void
arm_expand_compare_and_swap (rtx operands[])
{
rtx bval, bdst, rval, mem, oldval, newval, is_weak, mod_s, mod_f, x;
machine_mode mode;
rtx (*gen) (rtx, rtx, rtx, rtx, rtx, rtx, rtx, rtx);
bval = operands[0];
rval = operands[1];
mem = operands[2];
oldval = operands[3];
newval = operands[4];
is_weak = operands[5];
mod_s = operands[6];
mod_f = operands[7];
mode = GET_MODE (mem);
if (TARGET_HAVE_LDACQ
&& is_mm_acquire (memmodel_from_int (INTVAL (mod_f)))
&& is_mm_release (memmodel_from_int (INTVAL (mod_s))))
mod_s = GEN_INT (MEMMODEL_ACQ_REL);
switch (mode)
{
case E_QImode:
case E_HImode:
rval = gen_reg_rtx (SImode);
oldval = convert_modes (SImode, mode, oldval, true);
case E_SImode:
if (!arm_add_operand (oldval, SImode))
oldval = force_reg (SImode, oldval);
break;
case E_DImode:
if (!cmpdi_operand (oldval, mode))
oldval = force_reg (mode, oldval);
break;
default:
gcc_unreachable ();
}
if (TARGET_THUMB1)
{
switch (mode)
{
case E_QImode: gen = gen_atomic_compare_and_swapt1qi_1; break;
case E_HImode: gen = gen_atomic_compare_and_swapt1hi_1; break;
case E_SImode: gen = gen_atomic_compare_and_swapt1si_1; break;
case E_DImode: gen = gen_atomic_compare_and_swapt1di_1; break;
default:
gcc_unreachable ();
}
}
else
{
switch (mode)
{
case E_QImode: gen = gen_atomic_compare_and_swap32qi_1; break;
case E_HImode: gen = gen_atomic_compare_and_swap32hi_1; break;
case E_SImode: gen = gen_atomic_compare_and_swap32si_1; break;
case E_DImode: gen = gen_atomic_compare_and_swap32di_1; break;
default:
gcc_unreachable ();
}
}
bdst = TARGET_THUMB1 ? bval : gen_rtx_REG (CC_Zmode, CC_REGNUM);
emit_insn (gen (bdst, rval, mem, oldval, newval, is_weak, mod_s, mod_f));
if (mode == QImode || mode == HImode)
emit_move_insn (operands[1], gen_lowpart (mode, rval));
if (TARGET_THUMB1)
emit_insn (gen_cstoresi_eq0_thumb1 (bval, bdst));
else
{
x = gen_rtx_EQ (SImode, bdst, const0_rtx);
emit_insn (gen_rtx_SET (bval, x));
}
}
void
arm_split_compare_and_swap (rtx operands[])
{
rtx rval, mem, oldval, newval, neg_bval;
machine_mode mode;
enum memmodel mod_s, mod_f;
bool is_weak;
rtx_code_label *label1, *label2;
rtx x, cond;
rval = operands[1];
mem = operands[2];
oldval = operands[3];
newval = operands[4];
is_weak = (operands[5] != const0_rtx);
mod_s = memmodel_from_int (INTVAL (operands[6]));
mod_f = memmodel_from_int (INTVAL (operands[7]));
neg_bval = TARGET_THUMB1 ? operands[0] : operands[8];
mode = GET_MODE (mem);
bool is_armv8_sync = arm_arch8 && is_mm_sync (mod_s);
bool use_acquire = TARGET_HAVE_LDACQ
&& !(is_mm_relaxed (mod_s) || is_mm_consume (mod_s)
|| is_mm_release (mod_s));
bool use_release = TARGET_HAVE_LDACQ
&& !(is_mm_relaxed (mod_s) || is_mm_consume (mod_s)
|| is_mm_acquire (mod_s));
if (is_armv8_sync)
use_acquire = false;
if (!(use_acquire || use_release))
arm_pre_atomic_barrier (mod_s);
label1 = NULL;
if (!is_weak)
{
label1 = gen_label_rtx ();
emit_label (label1);
}
label2 = gen_label_rtx ();
arm_emit_load_exclusive (mode, rval, mem, use_acquire);
if (TARGET_32BIT)
{
cond = arm_gen_compare_reg (NE, rval, oldval, neg_bval);
x = gen_rtx_NE (VOIDmode, cond, const0_rtx);
x = gen_rtx_IF_THEN_ELSE (VOIDmode, x,
gen_rtx_LABEL_REF (Pmode, label2), pc_rtx);
emit_unlikely_jump (gen_rtx_SET (pc_rtx, x));
}
else
{
emit_move_insn (neg_bval, const1_rtx);
cond = gen_rtx_NE (VOIDmode, rval, oldval);
if (thumb1_cmpneg_operand (oldval, SImode))
emit_unlikely_jump (gen_cbranchsi4_scratch (neg_bval, rval, oldval,
label2, cond));
else
emit_unlikely_jump (gen_cbranchsi4_insn (cond, rval, oldval, label2));
}
arm_emit_store_exclusive (mode, neg_bval, mem, newval, use_release);
if (TARGET_32BIT)
{
cond = gen_rtx_REG (CCmode, CC_REGNUM);
x = gen_rtx_COMPARE (CCmode, neg_bval, const0_rtx);
emit_insn (gen_rtx_SET (cond, x));
}
if (!is_weak)
{
x = gen_rtx_NE (VOIDmode, neg_bval, const0_rtx);
emit_unlikely_jump (gen_cbranchsi4 (x, neg_bval, const0_rtx, label1));
}
if (!is_mm_relaxed (mod_f))
emit_label (label2);
if (is_armv8_sync
|| !(use_acquire || use_release))
arm_post_atomic_barrier (mod_s);
if (is_mm_relaxed (mod_f))
emit_label (label2);
}
void
arm_split_atomic_op (enum rtx_code code, rtx old_out, rtx new_out, rtx mem,
rtx value, rtx model_rtx, rtx cond)
{
enum memmodel model = memmodel_from_int (INTVAL (model_rtx));
machine_mode mode = GET_MODE (mem);
machine_mode wmode = (mode == DImode ? DImode : SImode);
rtx_code_label *label;
bool all_low_regs, bind_old_new;
rtx x;
bool is_armv8_sync = arm_arch8 && is_mm_sync (model);
bool use_acquire = TARGET_HAVE_LDACQ
&& !(is_mm_relaxed (model) || is_mm_consume (model)
|| is_mm_release (model));
bool use_release = TARGET_HAVE_LDACQ
&& !(is_mm_relaxed (model) || is_mm_consume (model)
|| is_mm_acquire (model));
if (is_armv8_sync)
use_acquire = false;
if (!(use_acquire || use_release))
arm_pre_atomic_barrier (model);
label = gen_label_rtx ();
emit_label (label);
if (new_out)
new_out = gen_lowpart (wmode, new_out);
if (old_out)
old_out = gen_lowpart (wmode, old_out);
else
old_out = new_out;
value = simplify_gen_subreg (wmode, value, mode, 0);
arm_emit_load_exclusive (mode, old_out, mem, use_acquire);
gcc_assert (!new_out || REG_P (new_out));
all_low_regs = REG_P (value) && REGNO_REG_CLASS (REGNO (value)) == LO_REGS
&& new_out && REGNO_REG_CLASS (REGNO (new_out)) == LO_REGS
&& REGNO_REG_CLASS (REGNO (old_out)) == LO_REGS;
bind_old_new =
(TARGET_THUMB1
&& code != SET
&& code != MINUS
&& (code != PLUS || (!all_low_regs && !satisfies_constraint_L (value))));
if (old_out && bind_old_new)
{
emit_move_insn (new_out, old_out);
old_out = new_out;
}
switch (code)
{
case SET:
new_out = value;
break;
case NOT:
x = gen_rtx_AND (wmode, old_out, value);
emit_insn (gen_rtx_SET (new_out, x));
x = gen_rtx_NOT (wmode, new_out);
emit_insn (gen_rtx_SET (new_out, x));
break;
case MINUS:
if (CONST_INT_P (value))
{
value = GEN_INT (-INTVAL (value));
code = PLUS;
}
case PLUS:
if (mode == DImode)
{
emit_move_insn (new_out, old_out);
if (code == PLUS)
x = gen_adddi3 (new_out, new_out, value);
else
x = gen_subdi3 (new_out, new_out, value);
emit_insn (x);
break;
}
default:
x = gen_rtx_fmt_ee (code, wmode, old_out, value);
emit_insn (gen_rtx_SET (new_out, x));
break;
}
arm_emit_store_exclusive (mode, cond, mem, gen_lowpart (mode, new_out),
use_release);
x = gen_rtx_NE (VOIDmode, cond, const0_rtx);
emit_unlikely_jump (gen_cbranchsi4 (x, cond, const0_rtx, label));
if (is_armv8_sync
|| !(use_acquire || use_release))
arm_post_atomic_barrier (model);
}

#define MAX_VECT_LEN 16
struct expand_vec_perm_d
{
rtx target, op0, op1;
vec_perm_indices perm;
machine_mode vmode;
bool one_vector_p;
bool testing_p;
};
static void
arm_expand_vec_perm_1 (rtx target, rtx op0, rtx op1, rtx sel)
{
machine_mode vmode = GET_MODE (target);
bool one_vector_p = rtx_equal_p (op0, op1);
gcc_checking_assert (vmode == V8QImode || vmode == V16QImode);
gcc_checking_assert (GET_MODE (op0) == vmode);
gcc_checking_assert (GET_MODE (op1) == vmode);
gcc_checking_assert (GET_MODE (sel) == vmode);
gcc_checking_assert (TARGET_NEON);
if (one_vector_p)
{
if (vmode == V8QImode)
emit_insn (gen_neon_vtbl1v8qi (target, op0, sel));
else
emit_insn (gen_neon_vtbl1v16qi (target, op0, sel));
}
else
{
rtx pair;
if (vmode == V8QImode)
{
pair = gen_reg_rtx (V16QImode);
emit_insn (gen_neon_vcombinev8qi (pair, op0, op1));
pair = gen_lowpart (TImode, pair);
emit_insn (gen_neon_vtbl2v8qi (target, pair, sel));
}
else
{
pair = gen_reg_rtx (OImode);
emit_insn (gen_neon_vcombinev16qi (pair, op0, op1));
emit_insn (gen_neon_vtbl2v16qi (target, pair, sel));
}
}
}
void
arm_expand_vec_perm (rtx target, rtx op0, rtx op1, rtx sel)
{
machine_mode vmode = GET_MODE (target);
unsigned int nelt = GET_MODE_NUNITS (vmode);
bool one_vector_p = rtx_equal_p (op0, op1);
rtx mask;
gcc_checking_assert (!BYTES_BIG_ENDIAN);
mask = GEN_INT (one_vector_p ? nelt - 1 : 2 * nelt - 1);
mask = gen_const_vec_duplicate (vmode, mask);
sel = expand_simple_binop (vmode, AND, sel, mask, NULL, 0, OPTAB_LIB_WIDEN);
arm_expand_vec_perm_1 (target, op0, op1, sel);
}
static int
neon_endian_lane_map (machine_mode mode, int lane)
{
if (BYTES_BIG_ENDIAN)
{
int nelems = GET_MODE_NUNITS (mode);
lane = (nelems - 1 - lane);
if (GET_MODE_SIZE (mode) == 16)
lane = lane ^ (nelems / 2);
}
return lane;
}
static int
neon_pair_endian_lane_map (machine_mode mode, int lane)
{
int nelem = GET_MODE_NUNITS (mode);
if (BYTES_BIG_ENDIAN)
lane =
neon_endian_lane_map (mode, lane & (nelem - 1)) + (lane & nelem);
return lane;
}
static bool
arm_evpc_neon_vuzp (struct expand_vec_perm_d *d)
{
unsigned int i, odd, mask, nelt = d->perm.length ();
rtx out0, out1, in0, in1;
rtx (*gen)(rtx, rtx, rtx, rtx);
int first_elem;
int swap_nelt;
if (GET_MODE_UNIT_SIZE (d->vmode) >= 8)
return false;
swap_nelt = BYTES_BIG_ENDIAN && !d->one_vector_p
&& GET_MODE_SIZE (d->vmode) == 8 ? nelt : 0;
first_elem = d->perm[neon_endian_lane_map (d->vmode, 0)] ^ swap_nelt;
if (first_elem == neon_endian_lane_map (d->vmode, 0))
odd = 0;
else if (first_elem == neon_endian_lane_map (d->vmode, 1))
odd = 1;
else
return false;
mask = (d->one_vector_p ? nelt - 1 : 2 * nelt - 1);
for (i = 0; i < nelt; i++)
{
unsigned elt =
(neon_pair_endian_lane_map (d->vmode, i) * 2 + odd) & mask;
if ((d->perm[i] ^ swap_nelt) != neon_pair_endian_lane_map (d->vmode, elt))
return false;
}
if (d->testing_p)
return true;
switch (d->vmode)
{
case E_V16QImode: gen = gen_neon_vuzpv16qi_internal; break;
case E_V8QImode:  gen = gen_neon_vuzpv8qi_internal;  break;
case E_V8HImode:  gen = gen_neon_vuzpv8hi_internal;  break;
case E_V4HImode:  gen = gen_neon_vuzpv4hi_internal;  break;
case E_V8HFmode:  gen = gen_neon_vuzpv8hf_internal;  break;
case E_V4HFmode:  gen = gen_neon_vuzpv4hf_internal;  break;
case E_V4SImode:  gen = gen_neon_vuzpv4si_internal;  break;
case E_V2SImode:  gen = gen_neon_vuzpv2si_internal;  break;
case E_V2SFmode:  gen = gen_neon_vuzpv2sf_internal;  break;
case E_V4SFmode:  gen = gen_neon_vuzpv4sf_internal;  break;
default:
gcc_unreachable ();
}
in0 = d->op0;
in1 = d->op1;
if (swap_nelt != 0)
std::swap (in0, in1);
out0 = d->target;
out1 = gen_reg_rtx (d->vmode);
if (odd)
std::swap (out0, out1);
emit_insn (gen (out0, in0, in1, out1));
return true;
}
static bool
arm_evpc_neon_vzip (struct expand_vec_perm_d *d)
{
unsigned int i, high, mask, nelt = d->perm.length ();
rtx out0, out1, in0, in1;
rtx (*gen)(rtx, rtx, rtx, rtx);
int first_elem;
bool is_swapped;
if (GET_MODE_UNIT_SIZE (d->vmode) >= 8)
return false;
is_swapped = BYTES_BIG_ENDIAN;
first_elem = d->perm[neon_endian_lane_map (d->vmode, 0) ^ is_swapped];
high = nelt / 2;
if (first_elem == neon_endian_lane_map (d->vmode, high))
;
else if (first_elem == neon_endian_lane_map (d->vmode, 0))
high = 0;
else
return false;
mask = (d->one_vector_p ? nelt - 1 : 2 * nelt - 1);
for (i = 0; i < nelt / 2; i++)
{
unsigned elt =
neon_pair_endian_lane_map (d->vmode, i + high) & mask;
if (d->perm[neon_pair_endian_lane_map (d->vmode, 2 * i + is_swapped)]
!= elt)
return false;
elt =
neon_pair_endian_lane_map (d->vmode, i + nelt + high) & mask;
if (d->perm[neon_pair_endian_lane_map (d->vmode, 2 * i + !is_swapped)]
!= elt)
return false;
}
if (d->testing_p)
return true;
switch (d->vmode)
{
case E_V16QImode: gen = gen_neon_vzipv16qi_internal; break;
case E_V8QImode:  gen = gen_neon_vzipv8qi_internal;  break;
case E_V8HImode:  gen = gen_neon_vzipv8hi_internal;  break;
case E_V4HImode:  gen = gen_neon_vzipv4hi_internal;  break;
case E_V8HFmode:  gen = gen_neon_vzipv8hf_internal;  break;
case E_V4HFmode:  gen = gen_neon_vzipv4hf_internal;  break;
case E_V4SImode:  gen = gen_neon_vzipv4si_internal;  break;
case E_V2SImode:  gen = gen_neon_vzipv2si_internal;  break;
case E_V2SFmode:  gen = gen_neon_vzipv2sf_internal;  break;
case E_V4SFmode:  gen = gen_neon_vzipv4sf_internal;  break;
default:
gcc_unreachable ();
}
in0 = d->op0;
in1 = d->op1;
if (is_swapped)
std::swap (in0, in1);
out0 = d->target;
out1 = gen_reg_rtx (d->vmode);
if (high)
std::swap (out0, out1);
emit_insn (gen (out0, in0, in1, out1));
return true;
}
static bool
arm_evpc_neon_vrev (struct expand_vec_perm_d *d)
{
unsigned int i, j, diff, nelt = d->perm.length ();
rtx (*gen)(rtx, rtx);
if (!d->one_vector_p)
return false;
diff = d->perm[0];
switch (diff)
{
case 7:
switch (d->vmode)
{
case E_V16QImode: gen = gen_neon_vrev64v16qi; break;
case E_V8QImode:  gen = gen_neon_vrev64v8qi;  break;
default:
return false;
}
break;
case 3:
switch (d->vmode)
{
case E_V16QImode: gen = gen_neon_vrev32v16qi; break;
case E_V8QImode:  gen = gen_neon_vrev32v8qi;  break;
case E_V8HImode:  gen = gen_neon_vrev64v8hi;  break;
case E_V4HImode:  gen = gen_neon_vrev64v4hi;  break;
case E_V8HFmode:  gen = gen_neon_vrev64v8hf;  break;
case E_V4HFmode:  gen = gen_neon_vrev64v4hf;  break;
default:
return false;
}
break;
case 1:
switch (d->vmode)
{
case E_V16QImode: gen = gen_neon_vrev16v16qi; break;
case E_V8QImode:  gen = gen_neon_vrev16v8qi;  break;
case E_V8HImode:  gen = gen_neon_vrev32v8hi;  break;
case E_V4HImode:  gen = gen_neon_vrev32v4hi;  break;
case E_V4SImode:  gen = gen_neon_vrev64v4si;  break;
case E_V2SImode:  gen = gen_neon_vrev64v2si;  break;
case E_V4SFmode:  gen = gen_neon_vrev64v4sf;  break;
case E_V2SFmode:  gen = gen_neon_vrev64v2sf;  break;
default:
return false;
}
break;
default:
return false;
}
for (i = 0; i < nelt ; i += diff + 1)
for (j = 0; j <= diff; j += 1)
{
gcc_assert (i + j < nelt);
if (d->perm[i + j] != i + diff - j)
return false;
}
if (d->testing_p)
return true;
emit_insn (gen (d->target, d->op0));
return true;
}
static bool
arm_evpc_neon_vtrn (struct expand_vec_perm_d *d)
{
unsigned int i, odd, mask, nelt = d->perm.length ();
rtx out0, out1, in0, in1;
rtx (*gen)(rtx, rtx, rtx, rtx);
if (GET_MODE_UNIT_SIZE (d->vmode) >= 8)
return false;
if (d->perm[0] == 0)
odd = 0;
else if (d->perm[0] == 1)
odd = 1;
else
return false;
mask = (d->one_vector_p ? nelt - 1 : 2 * nelt - 1);
for (i = 0; i < nelt; i += 2)
{
if (d->perm[i] != i + odd)
return false;
if (d->perm[i + 1] != ((i + nelt + odd) & mask))
return false;
}
if (d->testing_p)
return true;
switch (d->vmode)
{
case E_V16QImode: gen = gen_neon_vtrnv16qi_internal; break;
case E_V8QImode:  gen = gen_neon_vtrnv8qi_internal;  break;
case E_V8HImode:  gen = gen_neon_vtrnv8hi_internal;  break;
case E_V4HImode:  gen = gen_neon_vtrnv4hi_internal;  break;
case E_V8HFmode:  gen = gen_neon_vtrnv8hf_internal;  break;
case E_V4HFmode:  gen = gen_neon_vtrnv4hf_internal;  break;
case E_V4SImode:  gen = gen_neon_vtrnv4si_internal;  break;
case E_V2SImode:  gen = gen_neon_vtrnv2si_internal;  break;
case E_V2SFmode:  gen = gen_neon_vtrnv2sf_internal;  break;
case E_V4SFmode:  gen = gen_neon_vtrnv4sf_internal;  break;
default:
gcc_unreachable ();
}
in0 = d->op0;
in1 = d->op1;
if (BYTES_BIG_ENDIAN)
{
std::swap (in0, in1);
odd = !odd;
}
out0 = d->target;
out1 = gen_reg_rtx (d->vmode);
if (odd)
std::swap (out0, out1);
emit_insn (gen (out0, in0, in1, out1));
return true;
}
static bool
arm_evpc_neon_vext (struct expand_vec_perm_d *d)
{
unsigned int i, nelt = d->perm.length ();
rtx (*gen) (rtx, rtx, rtx, rtx);
rtx offset;
unsigned int location;
unsigned int next  = d->perm[0] + 1;
if (BYTES_BIG_ENDIAN)
return false;
for (i = 1; i < nelt; next++, i++)
{
if (next == 2 * nelt)
return false;
if (d->one_vector_p && (next == nelt))
{
if ((nelt == 2) && (d->vmode != V2DImode))
return false;
else
next = 0;
}
if (d->perm[i] != next)
return false;
}
location = d->perm[0];
switch (d->vmode)
{
case E_V16QImode: gen = gen_neon_vextv16qi; break;
case E_V8QImode: gen = gen_neon_vextv8qi; break;
case E_V4HImode: gen = gen_neon_vextv4hi; break;
case E_V8HImode: gen = gen_neon_vextv8hi; break;
case E_V2SImode: gen = gen_neon_vextv2si; break;
case E_V4SImode: gen = gen_neon_vextv4si; break;
case E_V4HFmode: gen = gen_neon_vextv4hf; break;
case E_V8HFmode: gen = gen_neon_vextv8hf; break;
case E_V2SFmode: gen = gen_neon_vextv2sf; break;
case E_V4SFmode: gen = gen_neon_vextv4sf; break;
case E_V2DImode: gen = gen_neon_vextv2di; break;
default:
return false;
}
if (d->testing_p)
return true;
offset = GEN_INT (location);
emit_insn (gen (d->target, d->op0, d->op1, offset));
return true;
}
static bool
arm_evpc_neon_vtbl (struct expand_vec_perm_d *d)
{
rtx rperm[MAX_VECT_LEN], sel;
machine_mode vmode = d->vmode;
unsigned int i, nelt = d->perm.length ();
if (BYTES_BIG_ENDIAN)
return false;
if (d->testing_p)
return true;
if (vmode != V8QImode && vmode != V16QImode)
return false;
for (i = 0; i < nelt; ++i)
rperm[i] = GEN_INT (d->perm[i]);
sel = gen_rtx_CONST_VECTOR (vmode, gen_rtvec_v (nelt, rperm));
sel = force_reg (vmode, sel);
arm_expand_vec_perm_1 (d->target, d->op0, d->op1, sel);
return true;
}
static bool
arm_expand_vec_perm_const_1 (struct expand_vec_perm_d *d)
{
if (TARGET_NEON)
if (arm_evpc_neon_vext (d))
return true;
unsigned int nelt = d->perm.length ();
if (d->perm[0] >= nelt)
{
d->perm.rotate_inputs (1);
std::swap (d->op0, d->op1);
}
if (TARGET_NEON)
{
if (arm_evpc_neon_vuzp (d))
return true;
if (arm_evpc_neon_vzip (d))
return true;
if (arm_evpc_neon_vrev (d))
return true;
if (arm_evpc_neon_vtrn (d))
return true;
return arm_evpc_neon_vtbl (d);
}
return false;
}
static bool
arm_vectorize_vec_perm_const (machine_mode vmode, rtx target, rtx op0, rtx op1,
const vec_perm_indices &sel)
{
struct expand_vec_perm_d d;
int i, nelt, which;
if (!VALID_NEON_DREG_MODE (vmode) && !VALID_NEON_QREG_MODE (vmode))
return false;
d.target = target;
d.op0 = op0;
d.op1 = op1;
d.vmode = vmode;
gcc_assert (VECTOR_MODE_P (d.vmode));
d.testing_p = !target;
nelt = GET_MODE_NUNITS (d.vmode);
for (i = which = 0; i < nelt; ++i)
{
int ei = sel[i] & (2 * nelt - 1);
which |= (ei < nelt ? 1 : 2);
}
switch (which)
{
default:
gcc_unreachable();
case 3:
d.one_vector_p = false;
if (d.testing_p || !rtx_equal_p (op0, op1))
break;
case 2:
d.op0 = op1;
d.one_vector_p = true;
break;
case 1:
d.op1 = op0;
d.one_vector_p = true;
break;
}
d.perm.new_vector (sel.encoding (), d.one_vector_p ? 1 : 2, nelt);
if (!d.testing_p)
return arm_expand_vec_perm_const_1 (&d);
d.target = gen_raw_REG (d.vmode, LAST_VIRTUAL_REGISTER + 1);
d.op1 = d.op0 = gen_raw_REG (d.vmode, LAST_VIRTUAL_REGISTER + 2);
if (!d.one_vector_p)
d.op1 = gen_raw_REG (d.vmode, LAST_VIRTUAL_REGISTER + 3);
start_sequence ();
bool ret = arm_expand_vec_perm_const_1 (&d);
end_sequence ();
return ret;
}
bool
arm_autoinc_modes_ok_p (machine_mode mode, enum arm_auto_incmodes code)
{
if (TARGET_SOFT_FLOAT && (TARGET_LDRD || GET_MODE_SIZE (mode) <= 4))
return true;
switch (code)
{
case ARM_POST_INC:
case ARM_PRE_DEC:
if (VECTOR_MODE_P (mode))
{
if (code != ARM_PRE_DEC)
return true;
else
return false;
}
return true;
case ARM_POST_DEC:
case ARM_PRE_INC:
if (!TARGET_LDRD && GET_MODE_SIZE (mode) > 4)
return false;
if (FLOAT_MODE_P (mode) || VECTOR_MODE_P (mode))
return false;
return true;
default:
return false;
}
return false;
}
void
arm_emit_coreregs_64bit_shift (enum rtx_code code, rtx out, rtx in,
rtx amount, rtx scratch1, rtx scratch2)
{
rtx out_high = gen_highpart (SImode, out);
rtx out_low = gen_lowpart (SImode, out);
rtx in_high = gen_highpart (SImode, in);
rtx in_low = gen_lowpart (SImode, in);
rtx out_up   = code == ASHIFT ? out_low : out_high;
rtx out_down = code == ASHIFT ? out_high : out_low;
rtx in_up   = code == ASHIFT ? in_low : in_high;
rtx in_down = code == ASHIFT ? in_high : in_low;
gcc_assert (code == ASHIFT || code == ASHIFTRT || code == LSHIFTRT);
gcc_assert (out
&& (REG_P (out) || GET_CODE (out) == SUBREG)
&& GET_MODE (out) == DImode);
gcc_assert (in
&& (REG_P (in) || GET_CODE (in) == SUBREG)
&& GET_MODE (in) == DImode);
gcc_assert (amount
&& (((REG_P (amount) || GET_CODE (amount) == SUBREG)
&& GET_MODE (amount) == SImode)
|| CONST_INT_P (amount)));
gcc_assert (scratch1 == NULL
|| (GET_CODE (scratch1) == SCRATCH)
|| (GET_MODE (scratch1) == SImode
&& REG_P (scratch1)));
gcc_assert (scratch2 == NULL
|| (GET_CODE (scratch2) == SCRATCH)
|| (GET_MODE (scratch2) == SImode
&& REG_P (scratch2)));
gcc_assert (!REG_P (out) || !REG_P (amount)
|| !HARD_REGISTER_P (out)
|| (REGNO (out) != REGNO (amount)
&& REGNO (out) + 1 != REGNO (amount)));
#define SUB_32(DEST,SRC) \
gen_addsi3 ((DEST), (SRC), GEN_INT (-32))
#define RSB_32(DEST,SRC) \
gen_subsi3 ((DEST), GEN_INT (32), (SRC))
#define SUB_S_32(DEST,SRC) \
gen_addsi3_compare0 ((DEST), (SRC), \
GEN_INT (-32))
#define SET(DEST,SRC) \
gen_rtx_SET ((DEST), (SRC))
#define SHIFT(CODE,SRC,AMOUNT) \
gen_rtx_fmt_ee ((CODE), SImode, (SRC), (AMOUNT))
#define LSHIFT(CODE,SRC,AMOUNT) \
gen_rtx_fmt_ee ((CODE) == ASHIFT ? ASHIFT : LSHIFTRT, \
SImode, (SRC), (AMOUNT))
#define REV_LSHIFT(CODE,SRC,AMOUNT) \
gen_rtx_fmt_ee ((CODE) == ASHIFT ? LSHIFTRT : ASHIFT, \
SImode, (SRC), (AMOUNT))
#define ORR(A,B) \
gen_rtx_IOR (SImode, (A), (B))
#define BRANCH(COND,LABEL) \
gen_arm_cond_branch ((LABEL), \
gen_rtx_ ## COND (CCmode, cc_reg, \
const0_rtx), \
cc_reg)
if (CONST_INT_P (amount))
{
if (INTVAL (amount) <= 0)
emit_insn (gen_movdi (out, in));
else if (INTVAL (amount) >= 64)
{
if (code == ASHIFTRT)
{
rtx const31_rtx = GEN_INT (31);
emit_insn (SET (out_down, SHIFT (code, in_up, const31_rtx)));
emit_insn (SET (out_up, SHIFT (code, in_up, const31_rtx)));
}
else
emit_insn (gen_movdi (out, const0_rtx));
}
else if (INTVAL (amount) < 32)
{
rtx reverse_amount = GEN_INT (32 - INTVAL (amount));
if (REG_P (out) && REG_P (in) && REGNO (out) != REGNO (in))
emit_insn (SET (out, const0_rtx));
emit_insn (SET (out_down, LSHIFT (code, in_down, amount)));
emit_insn (SET (out_down,
ORR (REV_LSHIFT (code, in_up, reverse_amount),
out_down)));
emit_insn (SET (out_up, SHIFT (code, in_up, amount)));
}
else
{
rtx adj_amount = GEN_INT (INTVAL (amount) - 32);
if (REG_P (out) && REG_P (in) && REGNO (out) != REGNO (in))
emit_insn (SET (out, const0_rtx));
emit_insn (SET (out_down, SHIFT (code, in_up, adj_amount)));
if (code == ASHIFTRT)
emit_insn (gen_ashrsi3 (out_up, in_up,
GEN_INT (31)));
else
emit_insn (SET (out_up, const0_rtx));
}
}
else
{
rtx cc_reg = gen_rtx_REG (CC_NOOVmode, CC_REGNUM);
gcc_assert (scratch1 && REG_P (scratch1));
gcc_assert (scratch2 && REG_P (scratch2));
switch (code)
{
case ASHIFT:
emit_insn (SUB_32 (scratch1, amount));
emit_insn (RSB_32 (scratch2, amount));
break;
case ASHIFTRT:
emit_insn (RSB_32 (scratch1, amount));
emit_insn (SUB_S_32 (scratch2, amount));
break;
case LSHIFTRT:
emit_insn (RSB_32 (scratch1, amount));
emit_insn (SUB_32 (scratch2, amount));
break;
default:
gcc_unreachable ();
}
emit_insn (SET (out_down, LSHIFT (code, in_down, amount)));
if (!TARGET_THUMB2)
{
emit_insn (SET (out_down,
ORR (SHIFT (ASHIFT, in_up, scratch1), out_down)));
if (code == ASHIFTRT)
{
rtx_code_label *done_label = gen_label_rtx ();
emit_jump_insn (BRANCH (LT, done_label));
emit_insn (SET (out_down, ORR (SHIFT (ASHIFTRT, in_up, scratch2),
out_down)));
emit_label (done_label);
}
else
emit_insn (SET (out_down, ORR (SHIFT (LSHIFTRT, in_up, scratch2),
out_down)));
}
else
{
emit_insn (SET (scratch1, SHIFT (ASHIFT, in_up, scratch1)));
emit_insn (gen_iorsi3 (out_down, out_down, scratch1));
if (code == ASHIFTRT)
{
rtx_code_label *done_label = gen_label_rtx ();
emit_jump_insn (BRANCH (LT, done_label));
emit_insn (SET (scratch2, SHIFT (ASHIFTRT, in_up, scratch2)));
emit_insn (SET (out_down, ORR (out_down, scratch2)));
emit_label (done_label);
}
else
{
emit_insn (SET (scratch2, SHIFT (LSHIFTRT, in_up, scratch2)));
emit_insn (gen_iorsi3 (out_down, out_down, scratch2));
}
}
emit_insn (SET (out_up, SHIFT (code, in_up, amount)));
}
#undef SUB_32
#undef RSB_32
#undef SUB_S_32
#undef SET
#undef SHIFT
#undef LSHIFT
#undef REV_LSHIFT
#undef ORR
#undef BRANCH
}
bool
arm_valid_symbolic_address_p (rtx addr)
{
rtx xop0, xop1 = NULL_RTX;
rtx tmp = addr;
if (target_word_relocations)
return false;
if (GET_CODE (tmp) == SYMBOL_REF || GET_CODE (tmp) == LABEL_REF)
return true;
if (GET_CODE (addr) == CONST)
tmp = XEXP (addr, 0);
if (GET_CODE (tmp) == PLUS)
{
xop0 = XEXP (tmp, 0);
xop1 = XEXP (tmp, 1);
if (GET_CODE (xop0) == SYMBOL_REF && CONST_INT_P (xop1))
return IN_RANGE (INTVAL (xop1), -0x8000, 0x7fff);
}
return false;
}
bool
arm_validize_comparison (rtx *comparison, rtx * op1, rtx * op2)
{
enum rtx_code code = GET_CODE (*comparison);
int code_int;
machine_mode mode = (GET_MODE (*op1) == VOIDmode)
? GET_MODE (*op2) : GET_MODE (*op1);
gcc_assert (GET_MODE (*op1) != VOIDmode || GET_MODE (*op2) != VOIDmode);
if (code == UNEQ || code == LTGT)
return false;
code_int = (int)code;
arm_canonicalize_comparison (&code_int, op1, op2, 0);
PUT_CODE (*comparison, (enum rtx_code)code_int);
switch (mode)
{
case E_SImode:
if (!arm_add_operand (*op1, mode))
*op1 = force_reg (mode, *op1);
if (!arm_add_operand (*op2, mode))
*op2 = force_reg (mode, *op2);
return true;
case E_DImode:
if (!cmpdi_operand (*op1, mode))
*op1 = force_reg (mode, *op1);
if (!cmpdi_operand (*op2, mode))
*op2 = force_reg (mode, *op2);
return true;
case E_HFmode:
if (!TARGET_VFP_FP16INST)
break;
mode = SFmode;
*op1 = convert_to_mode (mode, *op1, 1);
*op2 = convert_to_mode (mode, *op2, 1);
case E_SFmode:
case E_DFmode:
if (!vfp_compare_operand (*op1, mode))
*op1 = force_reg (mode, *op1);
if (!vfp_compare_operand (*op2, mode))
*op2 = force_reg (mode, *op2);
return true;
default:
break;
}
return false;
}
static int
arm_block_set_max_insns (void)
{
if (optimize_function_for_size_p (cfun))
return 4;
else
return current_tune->max_insns_inline_memset;
}
static bool
arm_block_set_non_vect_profit_p (rtx val,
unsigned HOST_WIDE_INT length,
unsigned HOST_WIDE_INT align,
bool unaligned_p, bool use_strd_p)
{
int num = 0;
const int leftover[8] = {0, 1, 1, 2, 1, 2, 2, 3};
if (unaligned_p)
{
num = arm_const_inline_cost (SET, val);
num += length / align + length % align;
}
else if (use_strd_p)
{
num = arm_const_double_inline_cost (val);
num += (length >> 3) + leftover[length & 7];
}
else
{
num = arm_const_inline_cost (SET, val);
num += (length >> 2) + leftover[length & 3];
}
if (unaligned_access && length > 3 && (length & 3) == 3)
num--;
return (num <= arm_block_set_max_insns ());
}
static bool
arm_block_set_vect_profit_p (unsigned HOST_WIDE_INT length,
unsigned HOST_WIDE_INT align,
machine_mode mode)
{
int num;
bool unaligned_p = ((align & 3) != 0);
unsigned int nelt = GET_MODE_NUNITS (mode);
num = 1;
num += (length + nelt - 1) / nelt;
if (!unaligned_p && (length & 3) != 0)
num++;
if (!unaligned_p && mode == V16QImode)
num--;
return (num <= arm_block_set_max_insns ());
}
static bool
arm_block_set_unaligned_vect (rtx dstbase,
unsigned HOST_WIDE_INT length,
unsigned HOST_WIDE_INT value,
unsigned HOST_WIDE_INT align)
{
unsigned int i, nelt_v16, nelt_v8, nelt_mode;
rtx dst, mem;
rtx val_vec, reg;
rtx (*gen_func) (rtx, rtx);
machine_mode mode;
unsigned HOST_WIDE_INT v = value;
unsigned int offset = 0;
gcc_assert ((align & 0x3) != 0);
nelt_v8 = GET_MODE_NUNITS (V8QImode);
nelt_v16 = GET_MODE_NUNITS (V16QImode);
if (length >= nelt_v16)
{
mode = V16QImode;
gen_func = gen_movmisalignv16qi;
}
else
{
mode = V8QImode;
gen_func = gen_movmisalignv8qi;
}
nelt_mode = GET_MODE_NUNITS (mode);
gcc_assert (length >= nelt_mode);
if (!arm_block_set_vect_profit_p (length, align, mode))
return false;
dst = copy_addr_to_reg (XEXP (dstbase, 0));
mem = adjust_automodify_address (dstbase, mode, dst, offset);
v = sext_hwi (v, BITS_PER_WORD);
reg = gen_reg_rtx (mode);
val_vec = gen_const_vec_duplicate (mode, GEN_INT (v));
emit_move_insn (reg, val_vec);
for (i = 0; (i + nelt_mode <= length); i += nelt_mode)
{
emit_insn ((*gen_func) (mem, reg));
if (i + 2 * nelt_mode <= length)
{
emit_insn (gen_add2_insn (dst, GEN_INT (nelt_mode)));
offset += nelt_mode;
mem = adjust_automodify_address (dstbase, mode, dst, offset);
}
}
gcc_assert ((i + nelt_v8) > length || mode == V16QImode);
if (i + nelt_v8 < length)
{
emit_insn (gen_add2_insn (dst, GEN_INT (length - i)));
offset += length - i;
mem = adjust_automodify_address (dstbase, mode, dst, offset);
if ((length & 1) != 0 && align >= 2)
set_mem_align (mem, BITS_PER_UNIT);
emit_insn (gen_movmisalignv16qi (mem, reg));
}
else if (i < length && i + nelt_v8 >= length)
{
if (mode == V16QImode)
reg = gen_lowpart (V8QImode, reg);
emit_insn (gen_add2_insn (dst, GEN_INT ((length - i)
+ (nelt_mode - nelt_v8))));
offset += (length - i) + (nelt_mode - nelt_v8);
mem = adjust_automodify_address (dstbase, V8QImode, dst, offset);
if ((length & 1) != 0 && align >= 2)
set_mem_align (mem, BITS_PER_UNIT);
emit_insn (gen_movmisalignv8qi (mem, reg));
}
return true;
}
static bool
arm_block_set_aligned_vect (rtx dstbase,
unsigned HOST_WIDE_INT length,
unsigned HOST_WIDE_INT value,
unsigned HOST_WIDE_INT align)
{
unsigned int i, nelt_v8, nelt_v16, nelt_mode;
rtx dst, addr, mem;
rtx val_vec, reg;
machine_mode mode;
unsigned int offset = 0;
gcc_assert ((align & 0x3) == 0);
nelt_v8 = GET_MODE_NUNITS (V8QImode);
nelt_v16 = GET_MODE_NUNITS (V16QImode);
if (length >= nelt_v16 && unaligned_access && !BYTES_BIG_ENDIAN)
mode = V16QImode;
else
mode = V8QImode;
nelt_mode = GET_MODE_NUNITS (mode);
gcc_assert (length >= nelt_mode);
if (!arm_block_set_vect_profit_p (length, align, mode))
return false;
dst = copy_addr_to_reg (XEXP (dstbase, 0));
reg = gen_reg_rtx (mode);
val_vec = gen_const_vec_duplicate (mode, gen_int_mode (value, QImode));
emit_move_insn (reg, val_vec);
i = 0;
if (mode == V16QImode)
{
mem = adjust_automodify_address (dstbase, mode, dst, offset);
emit_insn (gen_movmisalignv16qi (mem, reg));
i += nelt_mode;
if (i + nelt_v8 < length && i + nelt_v16 > length)
{
emit_insn (gen_add2_insn (dst, GEN_INT (length - nelt_mode)));
offset += length - nelt_mode;
mem = adjust_automodify_address (dstbase, mode, dst, offset);
if ((length & 0x3) == 0)
set_mem_align (mem, BITS_PER_UNIT * 4);
else if ((length & 0x1) == 0)
set_mem_align (mem, BITS_PER_UNIT * 2);
else
set_mem_align (mem, BITS_PER_UNIT);
emit_insn (gen_movmisalignv16qi (mem, reg));
return true;
}
mode = V8QImode;
nelt_mode = GET_MODE_NUNITS (mode);
reg = gen_lowpart (V8QImode, reg);
}
for (; (i + nelt_mode <= length); i += nelt_mode)
{
addr = plus_constant (Pmode, dst, i);
mem = adjust_automodify_address (dstbase, mode, addr, offset + i);
emit_move_insn (mem, reg);
}
if (i + UNITS_PER_WORD == length)
{
addr = plus_constant (Pmode, dst, i - UNITS_PER_WORD);
offset += i - UNITS_PER_WORD;
mem = adjust_automodify_address (dstbase, mode, addr, offset);
if (align > UNITS_PER_WORD)
set_mem_align (mem, BITS_PER_UNIT * UNITS_PER_WORD);
emit_move_insn (mem, reg);
}
else if (i < length)
{
emit_insn (gen_add2_insn (dst, GEN_INT (length - nelt_mode)));
offset += length - nelt_mode;
mem = adjust_automodify_address (dstbase, mode, dst, offset);
if ((length & 1) == 0)
set_mem_align (mem, BITS_PER_UNIT * 2);
else
set_mem_align (mem, BITS_PER_UNIT);
emit_insn (gen_movmisalignv8qi (mem, reg));
}
return true;
}
static bool
arm_block_set_unaligned_non_vect (rtx dstbase,
unsigned HOST_WIDE_INT length,
unsigned HOST_WIDE_INT value,
unsigned HOST_WIDE_INT align)
{
unsigned int i;
rtx dst, addr, mem;
rtx val_exp, val_reg, reg;
machine_mode mode;
HOST_WIDE_INT v = value;
gcc_assert (align == 1 || align == 2);
if (align == 2)
v |= (value << BITS_PER_UNIT);
v = sext_hwi (v, BITS_PER_WORD);
val_exp = GEN_INT (v);
if (!arm_block_set_non_vect_profit_p (val_exp, length,
align, true, false))
return false;
dst = copy_addr_to_reg (XEXP (dstbase, 0));
mode = (align == 2 ? HImode : QImode);
val_reg = force_reg (SImode, val_exp);
reg = gen_lowpart (mode, val_reg);
for (i = 0; (i + GET_MODE_SIZE (mode) <= length); i += GET_MODE_SIZE (mode))
{
addr = plus_constant (Pmode, dst, i);
mem = adjust_automodify_address (dstbase, mode, addr, i);
emit_move_insn (mem, reg);
}
if (i + 1 == length)
{
reg = gen_lowpart (QImode, val_reg);
addr = plus_constant (Pmode, dst, i);
mem = adjust_automodify_address (dstbase, QImode, addr, i);
emit_move_insn (mem, reg);
i++;
}
gcc_assert (i == length);
return true;
}
static bool
arm_block_set_aligned_non_vect (rtx dstbase,
unsigned HOST_WIDE_INT length,
unsigned HOST_WIDE_INT value,
unsigned HOST_WIDE_INT align)
{
unsigned int i;
rtx dst, addr, mem;
rtx val_exp, val_reg, reg;
unsigned HOST_WIDE_INT v;
bool use_strd_p;
use_strd_p = (length >= 2 * UNITS_PER_WORD && (align & 3) == 0
&& TARGET_LDRD && current_tune->prefer_ldrd_strd);
v = (value | (value << 8) | (value << 16) | (value << 24));
if (length < UNITS_PER_WORD)
v &= (0xFFFFFFFF >> (UNITS_PER_WORD - length) * BITS_PER_UNIT);
if (use_strd_p)
v |= (v << BITS_PER_WORD);
else
v = sext_hwi (v, BITS_PER_WORD);
val_exp = GEN_INT (v);
if (!arm_block_set_non_vect_profit_p (val_exp, length,
align, false, use_strd_p))
{
if (!use_strd_p)
return false;
v = (v >> BITS_PER_WORD);
v = sext_hwi (v, BITS_PER_WORD);
val_exp = GEN_INT (v);
use_strd_p = false;
if (!arm_block_set_non_vect_profit_p (val_exp, length,
align, false, use_strd_p))
return false;
}
i = 0;
dst = copy_addr_to_reg (XEXP (dstbase, 0));
if (use_strd_p)
{
val_reg = force_reg (DImode, val_exp);
reg = val_reg;
for (; (i + 8 <= length); i += 8)
{
addr = plus_constant (Pmode, dst, i);
mem = adjust_automodify_address (dstbase, DImode, addr, i);
emit_move_insn (mem, reg);
}
}
else
val_reg = force_reg (SImode, val_exp);
reg = (use_strd_p ? gen_lowpart (SImode, val_reg) : val_reg);
for (; (i + 4 <= length); i += 4)
{
addr = plus_constant (Pmode, dst, i);
mem = adjust_automodify_address (dstbase, SImode, addr, i);
if ((align & 3) == 0)
emit_move_insn (mem, reg);
else
emit_insn (gen_unaligned_storesi (mem, reg));
}
if (unaligned_access && i > 0 && (i + 3) == length)
{
addr = plus_constant (Pmode, dst, i - 1);
mem = adjust_automodify_address (dstbase, SImode, addr, i - 1);
if ((align & 1) == 0)
set_mem_align (mem, BITS_PER_UNIT);
emit_insn (gen_unaligned_storesi (mem, reg));
return true;
}
if (i + 2 <= length)
{
reg = gen_lowpart (HImode, val_reg);
addr = plus_constant (Pmode, dst, i);
mem = adjust_automodify_address (dstbase, HImode, addr, i);
if ((align & 1) == 0)
emit_move_insn (mem, reg);
else
emit_insn (gen_unaligned_storehi (mem, reg));
i += 2;
}
if (i + 1 == length)
{
reg = gen_lowpart (QImode, val_reg);
addr = plus_constant (Pmode, dst, i);
mem = adjust_automodify_address (dstbase, QImode, addr, i);
emit_move_insn (mem, reg);
}
return true;
}
static bool
arm_block_set_vect (rtx dstbase,
unsigned HOST_WIDE_INT length,
unsigned HOST_WIDE_INT value,
unsigned HOST_WIDE_INT align)
{
if (((align & 3) != 0 || (length & 3) != 0)
&& (!unaligned_access || BYTES_BIG_ENDIAN))
return false;
if ((align & 3) == 0)
return arm_block_set_aligned_vect (dstbase, length, value, align);
else
return arm_block_set_unaligned_vect (dstbase, length, value, align);
}
bool
arm_gen_setmem (rtx *operands)
{
rtx dstbase = operands[0];
unsigned HOST_WIDE_INT length;
unsigned HOST_WIDE_INT value;
unsigned HOST_WIDE_INT align;
if (!CONST_INT_P (operands[2]) || !CONST_INT_P (operands[1]))
return false;
length = UINTVAL (operands[1]);
if (length > 64)
return false;
value = (UINTVAL (operands[2]) & 0xFF);
align = UINTVAL (operands[3]);
if (TARGET_NEON && length >= 8
&& current_tune->string_ops_prefer_neon
&& arm_block_set_vect (dstbase, length, value, align))
return true;
if (!unaligned_access && (align & 3) != 0)
return arm_block_set_unaligned_non_vect (dstbase, length, value, align);
return arm_block_set_aligned_non_vect (dstbase, length, value, align);
}
static bool
arm_macro_fusion_p (void)
{
return current_tune->fusible_ops != tune_params::FUSE_NOTHING;
}
static bool
arm_sets_movw_movt_fusible_p (rtx prev_set, rtx curr_set)
{
rtx set_dest = SET_DEST (curr_set);
if (GET_MODE (set_dest) != SImode)
return false;
if (GET_CODE (set_dest) == ZERO_EXTRACT)
{
if (CONST_INT_P (SET_SRC (curr_set))
&& CONST_INT_P (SET_SRC (prev_set))
&& REG_P (XEXP (set_dest, 0))
&& REG_P (SET_DEST (prev_set))
&& REGNO (XEXP (set_dest, 0)) == REGNO (SET_DEST (prev_set)))
return true;
}
else if (GET_CODE (SET_SRC (curr_set)) == LO_SUM
&& REG_P (SET_DEST (curr_set))
&& REG_P (SET_DEST (prev_set))
&& GET_CODE (SET_SRC (prev_set)) == HIGH
&& REGNO (SET_DEST (curr_set)) == REGNO (SET_DEST (prev_set)))
return true;
return false;
}
static bool
aarch_macro_fusion_pair_p (rtx_insn* prev, rtx_insn* curr)
{
rtx prev_set = single_set (prev);
rtx curr_set = single_set (curr);
if (!prev_set
|| !curr_set)
return false;
if (any_condjump_p (curr))
return false;
if (!arm_macro_fusion_p ())
return false;
if (current_tune->fusible_ops & tune_params::FUSE_AES_AESMC
&& aarch_crypto_can_dual_issue (prev, curr))
return true;
if (current_tune->fusible_ops & tune_params::FUSE_MOVW_MOVT
&& arm_sets_movw_movt_fusible_p (prev_set, curr_set))
return true;
return false;
}
bool
arm_fusion_enabled_p (tune_params::fuse_ops op)
{
return current_tune->fusible_ops & op;
}
static bool
arm_sched_can_speculate_insn (rtx_insn *insn)
{
switch (get_attr_type (insn))
{
case TYPE_SDIV:
case TYPE_UDIV:
case TYPE_FDIVS:
case TYPE_FDIVD:
case TYPE_FSQRTS:
case TYPE_FSQRTD:
case TYPE_NEON_FP_SQRT_S:
case TYPE_NEON_FP_SQRT_D:
case TYPE_NEON_FP_SQRT_S_Q:
case TYPE_NEON_FP_SQRT_D_Q:
case TYPE_NEON_FP_DIV_S:
case TYPE_NEON_FP_DIV_D:
case TYPE_NEON_FP_DIV_S_Q:
case TYPE_NEON_FP_DIV_D_Q:
return false;
default:
return true;
}
}
static unsigned HOST_WIDE_INT
arm_asan_shadow_offset (void)
{
return HOST_WIDE_INT_1U << 29;
}
static bool
arm_const_not_ok_for_debug_p (rtx p)
{
tree decl_op0 = NULL;
tree decl_op1 = NULL;
if (GET_CODE (p) == UNSPEC)
return true;
if (GET_CODE (p) == MINUS)
{
if (GET_CODE (XEXP (p, 1)) == SYMBOL_REF)
{
decl_op1 = SYMBOL_REF_DECL (XEXP (p, 1));
if (decl_op1
&& GET_CODE (XEXP (p, 0)) == SYMBOL_REF
&& (decl_op0 = SYMBOL_REF_DECL (XEXP (p, 0))))
{
if ((VAR_P (decl_op1)
|| TREE_CODE (decl_op1) == CONST_DECL)
&& (VAR_P (decl_op0)
|| TREE_CODE (decl_op0) == CONST_DECL))
return (get_variable_section (decl_op1, false)
!= get_variable_section (decl_op0, false));
if (TREE_CODE (decl_op1) == LABEL_DECL
&& TREE_CODE (decl_op0) == LABEL_DECL)
return (DECL_CONTEXT (decl_op1)
!= DECL_CONTEXT (decl_op0));
}
return true;
}
}
return false;
}
extern bool
arm_is_constant_pool_ref (rtx x)
{
return (MEM_P (x)
&& GET_CODE (XEXP (x, 0)) == SYMBOL_REF
&& CONSTANT_POOL_ADDRESS_P (XEXP (x, 0)));
}
static GTY(()) tree arm_previous_fndecl;
void
save_restore_target_globals (tree new_tree)
{
if (TREE_TARGET_GLOBALS (new_tree))
restore_target_globals (TREE_TARGET_GLOBALS (new_tree));
else if (new_tree == target_option_default_node)
restore_target_globals (&default_target_globals);
else
{
TREE_TARGET_GLOBALS (new_tree) = save_target_globals_default_opts ();
}
arm_option_params_internal ();
}
void
arm_reset_previous_fndecl (void)
{
arm_previous_fndecl = NULL_TREE;
}
static void
arm_set_current_function (tree fndecl)
{
if (!fndecl || fndecl == arm_previous_fndecl)
return;
tree old_tree = (arm_previous_fndecl
? DECL_FUNCTION_SPECIFIC_TARGET (arm_previous_fndecl)
: NULL_TREE);
tree new_tree = DECL_FUNCTION_SPECIFIC_TARGET (fndecl);
if (! new_tree && old_tree)
new_tree = target_option_default_node;
if (old_tree == new_tree)
return;
arm_previous_fndecl = fndecl;
cl_target_option_restore (&global_options, TREE_TARGET_OPTION (new_tree));
save_restore_target_globals (new_tree);
}
static void
arm_option_print (FILE *file, int indent, struct cl_target_option *ptr)
{
int flags = ptr->x_target_flags;
const char *fpu_name;
fpu_name = (ptr->x_arm_fpu_index == TARGET_FPU_auto
? "auto" : all_fpus[ptr->x_arm_fpu_index].name);
fprintf (file, "%*sselected isa %s\n", indent, "",
TARGET_THUMB2_P (flags) ? "thumb2" :
TARGET_THUMB_P (flags) ? "thumb1" :
"arm");
if (ptr->x_arm_arch_string)
fprintf (file, "%*sselected architecture %s\n", indent, "",
ptr->x_arm_arch_string);
if (ptr->x_arm_cpu_string)
fprintf (file, "%*sselected CPU %s\n", indent, "",
ptr->x_arm_cpu_string);
if (ptr->x_arm_tune_string)
fprintf (file, "%*sselected tune %s\n", indent, "",
ptr->x_arm_tune_string);
fprintf (file, "%*sselected fpu %s\n", indent, "", fpu_name);
}
static bool
arm_can_inline_p (tree caller, tree callee)
{
tree caller_tree = DECL_FUNCTION_SPECIFIC_TARGET (caller);
tree callee_tree = DECL_FUNCTION_SPECIFIC_TARGET (callee);
bool can_inline = true;
struct cl_target_option *caller_opts
= TREE_TARGET_OPTION (caller_tree ? caller_tree
: target_option_default_node);
struct cl_target_option *callee_opts
= TREE_TARGET_OPTION (callee_tree ? callee_tree
: target_option_default_node);
if (callee_opts == caller_opts)
return true;
struct arm_build_target caller_target;
struct arm_build_target callee_target;
caller_target.isa = sbitmap_alloc (isa_num_bits);
callee_target.isa = sbitmap_alloc (isa_num_bits);
arm_configure_build_target (&caller_target, caller_opts, &global_options_set,
false);
arm_configure_build_target (&callee_target, callee_opts, &global_options_set,
false);
if (!bitmap_subset_p (callee_target.isa, caller_target.isa))
can_inline = false;
sbitmap_free (caller_target.isa);
sbitmap_free (callee_target.isa);
return can_inline;
}
static void
arm_relayout_function (tree fndecl)
{
if (DECL_USER_ALIGN (fndecl))
return;
tree callee_tree = DECL_FUNCTION_SPECIFIC_TARGET (fndecl);
if (!callee_tree)
callee_tree = target_option_default_node;
struct cl_target_option *opts = TREE_TARGET_OPTION (callee_tree);
SET_DECL_ALIGN
(fndecl,
FUNCTION_ALIGNMENT (FUNCTION_BOUNDARY_P (opts->x_target_flags)));
}
static bool
arm_valid_target_attribute_rec (tree args, struct gcc_options *opts)
{
if (TREE_CODE (args) == TREE_LIST)
{
bool ret = true;
for (; args; args = TREE_CHAIN (args))
if (TREE_VALUE (args)
&& !arm_valid_target_attribute_rec (TREE_VALUE (args), opts))
ret = false;
return ret;
}
else if (TREE_CODE (args) != STRING_CST)
{
error ("attribute %<target%> argument not a string");
return false;
}
char *argstr = ASTRDUP (TREE_STRING_POINTER (args));
char *q;
while ((q = strtok (argstr, ",")) != NULL)
{
while (ISSPACE (*q)) ++q;
argstr = NULL;
if (!strncmp (q, "thumb", 5))
opts->x_target_flags |= MASK_THUMB;
else if (!strncmp (q, "arm", 3))
opts->x_target_flags &= ~MASK_THUMB;
else if (!strncmp (q, "fpu=", 4))
{
int fpu_index;
if (! opt_enum_arg_to_value (OPT_mfpu_, q+4,
&fpu_index, CL_TARGET))
{
error ("invalid fpu for target attribute or pragma %qs", q);
return false;
}
if (fpu_index == TARGET_FPU_auto)
{
sorry ("auto fpu selection not currently permitted here");
return false;
}
opts->x_arm_fpu_index = (enum fpu_type) fpu_index;
}
else if (!strncmp (q, "arch=", 5))
{
char* arch = q+5;
const arch_option *arm_selected_arch
= arm_parse_arch_option_name (all_architectures, "arch", arch);
if (!arm_selected_arch)
{
error ("invalid architecture for target attribute or pragma %qs",
q);
return false;
}
opts->x_arm_arch_string = xstrndup (arch, strlen (arch));
}
else if (q[0] == '+')
{
opts->x_arm_arch_string
= xasprintf ("%s%s", opts->x_arm_arch_string, q);
}
else
{
error ("unknown target attribute or pragma %qs", q);
return false;
}
}
return true;
}
tree
arm_valid_target_attribute_tree (tree args, struct gcc_options *opts,
struct gcc_options *opts_set)
{
struct cl_target_option cl_opts;
if (!arm_valid_target_attribute_rec (args, opts))
return NULL_TREE;
cl_target_option_save (&cl_opts, opts);
arm_configure_build_target (&arm_active_target, &cl_opts, opts_set, false);
arm_option_check_internal (opts);
arm_option_reconfigure_globals ();
arm_options_perform_arch_sanity_checks ();
arm_option_override_internal (opts, opts_set);
return build_target_option_node (opts);
}
static void 
add_attribute  (const char * mode, tree *attributes)
{
size_t len = strlen (mode);
tree value = build_string (len, mode);
TREE_TYPE (value) = build_array_type (char_type_node,
build_index_type (size_int (len)));
*attributes = tree_cons (get_identifier ("target"),
build_tree_list (NULL_TREE, value),
*attributes);
}
static void
arm_insert_attributes (tree fndecl, tree * attributes)
{
const char *mode;
if (! TARGET_FLIP_THUMB)
return;
if (TREE_CODE (fndecl) != FUNCTION_DECL || DECL_EXTERNAL(fndecl)
|| DECL_BUILT_IN (fndecl) || DECL_ARTIFICIAL (fndecl))
return;
if (current_function_decl)
{
mode = TARGET_THUMB ? "thumb" : "arm";      
add_attribute (mode, attributes);
return;
}
if (lookup_attribute ("target", *attributes) != NULL)
return;
mode = thumb_flipper ? "thumb" : "arm";
add_attribute (mode, attributes);
thumb_flipper = !thumb_flipper;
}
static bool
arm_valid_target_attribute_p (tree fndecl, tree ARG_UNUSED (name),
tree args, int ARG_UNUSED (flags))
{
bool ret = true;
struct gcc_options func_options;
tree cur_tree, new_optimize;
gcc_assert ((fndecl != NULL_TREE) && (args != NULL_TREE));
tree func_optimize = DECL_FUNCTION_SPECIFIC_OPTIMIZATION (fndecl);
if (!func_optimize)
func_optimize = optimization_default_node;
memset (&func_options, 0, sizeof (func_options));
init_options_struct (&func_options, NULL);
lang_hooks.init_options_struct (&func_options);
cl_optimization_restore (&func_options,
TREE_OPTIMIZATION (func_optimize));
cl_target_option_restore (&func_options,
TREE_TARGET_OPTION (target_option_default_node));
cur_tree = arm_valid_target_attribute_tree (args, &func_options,
&global_options_set);
if (cur_tree == NULL_TREE)
ret = false;
new_optimize = build_optimization_node (&func_options);
DECL_FUNCTION_SPECIFIC_TARGET (fndecl) = cur_tree;
DECL_FUNCTION_SPECIFIC_OPTIMIZATION (fndecl) = new_optimize;
finalize_options_struct (&func_options);
return ret;
}
static const char*
arm_identify_fpu_from_isa (sbitmap isa)
{
auto_sbitmap fpubits (isa_num_bits);
auto_sbitmap cand_fpubits (isa_num_bits);
bitmap_and (fpubits, isa, isa_all_fpubits);
if (bitmap_empty_p (fpubits))
return "softvfp";
for (unsigned int i = 0; i < TARGET_FPU_auto; i++)
{
arm_initialize_isa (cand_fpubits, all_fpus[i].isa_bits);
if (bitmap_equal_p (fpubits, cand_fpubits))
return all_fpus[i].name;
}
gcc_unreachable ();
}
void
arm_declare_function_name (FILE *stream, const char *name, tree decl)
{
tree target_parts = DECL_FUNCTION_SPECIFIC_TARGET (decl);
struct cl_target_option *targ_options;
if (target_parts)
targ_options = TREE_TARGET_OPTION (target_parts);
else
targ_options = TREE_TARGET_OPTION (target_option_current_node);
gcc_assert (targ_options);
std::string arch_to_print;
if (targ_options->x_arm_arch_string)
arch_to_print = targ_options->x_arm_arch_string;
if (arch_to_print != arm_last_printed_arch_string)
{
std::string arch_name
= arch_to_print.substr (0, arch_to_print.find ("+"));
asm_fprintf (asm_out_file, "\t.arch %s\n", arch_name.c_str ());
const arch_option *arch
= arm_parse_arch_option_name (all_architectures, "-march",
targ_options->x_arm_arch_string);
auto_sbitmap opt_bits (isa_num_bits);
gcc_assert (arch);
if (arch->common.extensions)
{
for (const struct cpu_arch_extension *opt = arch->common.extensions;
opt->name != NULL;
opt++)
{
if (!opt->remove)
{
arm_initialize_isa (opt_bits, opt->isa_bits);
if (bitmap_subset_p (opt_bits, arm_active_target.isa)
&& !bitmap_subset_p (opt_bits, isa_all_fpubits))
asm_fprintf (asm_out_file, "\t.arch_extension %s\n",
opt->name);
}
}
}
arm_last_printed_arch_string = arch_to_print;
}
fprintf (stream, "\t.syntax unified\n");
if (TARGET_THUMB)
{
if (is_called_in_ARM_mode (decl)
|| (TARGET_THUMB1 && !TARGET_THUMB1_ONLY
&& cfun->is_thunk))
fprintf (stream, "\t.code 32\n");
else if (TARGET_THUMB1)
fprintf (stream, "\t.code\t16\n\t.thumb_func\n");
else
fprintf (stream, "\t.thumb\n\t.thumb_func\n");
}
else
fprintf (stream, "\t.arm\n");
std::string fpu_to_print
= TARGET_SOFT_FLOAT
? "softvfp" : arm_identify_fpu_from_isa (arm_active_target.isa);
if (fpu_to_print != arm_last_printed_arch_string)
{
asm_fprintf (asm_out_file, "\t.fpu %s\n", fpu_to_print.c_str ());
arm_last_printed_fpu_string = fpu_to_print;
}
if (TARGET_POKE_FUNCTION_NAME)
arm_poke_function_name (stream, (const char *) name);
}
static bool
extract_base_offset_in_addr (rtx mem, rtx *base, rtx *offset)
{
rtx addr;
gcc_assert (MEM_P (mem));
addr = XEXP (mem, 0);
if (GET_CODE (addr) == CONST)
addr = XEXP (addr, 0);
if (GET_CODE (addr) == REG)
{
*base = addr;
*offset = const0_rtx;
return true;
}
if (GET_CODE (addr) == PLUS
&& GET_CODE (XEXP (addr, 0)) == REG
&& CONST_INT_P (XEXP (addr, 1)))
{
*base = XEXP (addr, 0);
*offset = XEXP (addr, 1);
return true;
}
*base = NULL_RTX;
*offset = NULL_RTX;
return false;
}
static bool
fusion_load_store (rtx_insn *insn, rtx *base, rtx *offset, bool *is_load)
{
rtx x, dest, src;
gcc_assert (INSN_P (insn));
x = PATTERN (insn);
if (GET_CODE (x) != SET)
return false;
src = SET_SRC (x);
dest = SET_DEST (x);
if (GET_CODE (src) == REG && GET_CODE (dest) == MEM)
{
*is_load = false;
extract_base_offset_in_addr (dest, base, offset);
}
else if (GET_CODE (src) == MEM && GET_CODE (dest) == REG)
{
*is_load = true;
extract_base_offset_in_addr (src, base, offset);
}
else
return false;
return (*base != NULL_RTX && *offset != NULL_RTX);
}
static void
arm_sched_fusion_priority (rtx_insn *insn, int max_pri,
int *fusion_pri, int *pri)
{
int tmp, off_val;
bool is_load;
rtx base, offset;
gcc_assert (INSN_P (insn));
tmp = max_pri - 1;
if (!fusion_load_store (insn, &base, &offset, &is_load))
{
*pri = tmp;
*fusion_pri = tmp;
return;
}
if (is_load)
*fusion_pri = tmp - 1;
else
*fusion_pri = tmp - 2;
tmp /= 2;
tmp -= ((REGNO (base) & 0xff) << 20);
off_val = (int)(INTVAL (offset));
if (off_val >= 0)
tmp -= (off_val & 0xfffff);
else
tmp += ((- off_val) & 0xfffff);
*pri = tmp;
return;
}
rtx
arm_simd_vect_par_cnst_half (machine_mode mode, bool high)
{
int nunits = GET_MODE_NUNITS (mode);
rtvec v = rtvec_alloc (nunits / 2);
int high_base = nunits / 2;
int low_base = 0;
int base;
rtx t1;
int i;
if (BYTES_BIG_ENDIAN)
base = high ? low_base : high_base;
else
base = high ? high_base : low_base;
for (i = 0; i < nunits / 2; i++)
RTVEC_ELT (v, i) = GEN_INT (base + i);
t1 = gen_rtx_PARALLEL (mode, v);
return t1;
}
bool
arm_simd_check_vect_par_cnst_half_p (rtx op, machine_mode mode,
bool high)
{
rtx ideal = arm_simd_vect_par_cnst_half (mode, high);
HOST_WIDE_INT count_op = XVECLEN (op, 0);
HOST_WIDE_INT count_ideal = XVECLEN (ideal, 0);
int i = 0;
if (!VECTOR_MODE_P (mode))
return false;
if (count_op != count_ideal)
return false;
for (i = 0; i < count_ideal; i++)
{
rtx elt_op = XVECEXP (op, 0, i);
rtx elt_ideal = XVECEXP (ideal, 0, i);
if (!CONST_INT_P (elt_op)
|| INTVAL (elt_ideal) != INTVAL (elt_op))
return false;
}
return true;
}
static bool
arm_can_output_mi_thunk (const_tree, HOST_WIDE_INT, HOST_WIDE_INT vcall_offset,
const_tree)
{
if (vcall_offset && TARGET_THUMB1)
return false;
return true;
}
void
arm_gen_unlikely_cbranch (enum rtx_code code, machine_mode cc_mode,
rtx label_ref)
{
rtx x;
x = gen_rtx_fmt_ee (code, VOIDmode,
gen_rtx_REG (cc_mode, CC_REGNUM),
const0_rtx);
x = gen_rtx_IF_THEN_ELSE (VOIDmode, x,
gen_rtx_LABEL_REF (VOIDmode, label_ref),
pc_rtx);
emit_unlikely_jump (gen_rtx_SET (pc_rtx, x));
}
static bool
arm_asm_elf_flags_numeric (unsigned int flags, unsigned int *num)
{
if (flags & SECTION_ARM_PURECODE)
{
*num = 0x20000000;
if (!(flags & SECTION_DEBUG))
*num |= 0x2;
if (flags & SECTION_EXCLUDE)
*num |= 0x80000000;
if (flags & SECTION_WRITE)
*num |= 0x1;
if (flags & SECTION_CODE)
*num |= 0x4;
if (flags & SECTION_MERGE)
*num |= 0x10;
if (flags & SECTION_STRINGS)
*num |= 0x20;
if (flags & SECTION_TLS)
*num |= 0x400;
if (HAVE_COMDAT_GROUP && (flags & SECTION_LINKONCE))
*num |= 0x200;
return true;
}
return false;
}
static section *
arm_function_section (tree decl, enum node_frequency freq,
bool startup, bool exit)
{
const char * section_name;
section * sec;
if (!decl || TREE_CODE (decl) != FUNCTION_DECL)
return default_function_section (decl, freq, startup, exit);
if (!target_pure_code)
return default_function_section (decl, freq, startup, exit);
section_name = DECL_SECTION_NAME (decl);
if (!section_name)
{
section *default_sec = default_function_section (decl, freq, startup,
exit);
if (default_sec)
default_sec->common.flags |= SECTION_ARM_PURECODE;
return default_sec;
}
sec = get_named_section (decl, section_name, 0);
if (!sec)
sec = get_named_section (decl, NULL, 0);
sec->common.flags |= SECTION_ARM_PURECODE;
return sec;
}
static unsigned int
arm_elf_section_type_flags (tree decl, const char *name, int reloc)
{
unsigned int flags = default_section_type_flags (decl, name, reloc);
if (decl && TREE_CODE (decl) == FUNCTION_DECL && target_pure_code)
flags |= SECTION_ARM_PURECODE;
return flags;
}
static void
arm_expand_divmod_libfunc (rtx libfunc, machine_mode mode,
rtx op0, rtx op1,
rtx *quot_p, rtx *rem_p)
{
if (mode == SImode)
gcc_assert (!TARGET_IDIV);
scalar_int_mode libval_mode
= smallest_int_mode_for_size (2 * GET_MODE_BITSIZE (mode));
rtx libval = emit_library_call_value (libfunc, NULL_RTX, LCT_CONST,
libval_mode,
op0, GET_MODE (op0),
op1, GET_MODE (op1));
rtx quotient = simplify_gen_subreg (mode, libval, libval_mode, 0);
rtx remainder = simplify_gen_subreg (mode, libval, libval_mode,
GET_MODE_SIZE (mode));
gcc_assert (quotient);
gcc_assert (remainder);
*quot_p = quotient;
*rem_p = remainder;
}
bool
arm_coproc_builtin_available (enum unspecv builtin)
{
if (TARGET_THUMB1)
return false;
switch (builtin)
{
case VUNSPEC_CDP:
case VUNSPEC_LDC:
case VUNSPEC_LDCL:
case VUNSPEC_STC:
case VUNSPEC_STCL:
case VUNSPEC_MCR:
case VUNSPEC_MRC:
if (arm_arch4)
return true;
break;
case VUNSPEC_CDP2:
case VUNSPEC_LDC2:
case VUNSPEC_LDC2L:
case VUNSPEC_STC2:
case VUNSPEC_STC2L:
case VUNSPEC_MCR2:
case VUNSPEC_MRC2:
if (arm_arch5)
return true;
break;
case VUNSPEC_MCRR:
case VUNSPEC_MRRC:
if (arm_arch6 || arm_arch5te)
return true;
break;
case VUNSPEC_MCRR2:
case VUNSPEC_MRRC2:
if (arm_arch6)
return true;
break;
default:
gcc_unreachable ();
}
return false;
}
bool
arm_coproc_ldc_stc_legitimate_address (rtx op)
{
HOST_WIDE_INT range;
if (!MEM_P (op))
return false;
op = XEXP (op, 0);
if (REG_P (op))
return true;
switch GET_CODE (op)
{
case PLUS:
{
if (!REG_P (XEXP (op, 0)))
return false;
op = XEXP (op, 1);
if (!CONST_INT_P (op))
return false;
range = INTVAL (op);
if (!IN_RANGE (range, -1020, 1020))
return false;
return (range % 4) == 0;
}
case PRE_INC:
case POST_INC:
case PRE_DEC:
case POST_DEC:
return REG_P (XEXP (op, 0));
default:
gcc_unreachable ();
}
return false;
}
static bool
arm_can_change_mode_class (machine_mode from, machine_mode to,
reg_class_t rclass)
{
if (TARGET_BIG_END
&& !(GET_MODE_SIZE (from) == 16 && GET_MODE_SIZE (to) == 8)
&& (GET_MODE_SIZE (from) > UNITS_PER_WORD
|| GET_MODE_SIZE (to) > UNITS_PER_WORD)
&& reg_classes_intersect_p (VFP_REGS, rclass))
return false;
return true;
}
static HOST_WIDE_INT
arm_constant_alignment (const_tree exp, HOST_WIDE_INT align)
{
unsigned int factor = (TARGET_THUMB || ! arm_tune_xscale ? 1 : 2);
if (TREE_CODE (exp) == STRING_CST && !optimize_size)
return MAX (align, BITS_PER_WORD * factor);
return align;
}
#if CHECKING_P
namespace selftest {
static void
arm_test_cpu_arch_data (void)
{
const arch_option *arch;
const cpu_option *cpu;
auto_sbitmap target_isa (isa_num_bits);
auto_sbitmap isa1 (isa_num_bits);
auto_sbitmap isa2 (isa_num_bits);
for (arch = all_architectures; arch->common.name != NULL; ++arch)
{
const cpu_arch_extension *ext1, *ext2;
if (arch->common.extensions == NULL)
continue;
arm_initialize_isa (target_isa, arch->common.isa_bits);
for (ext1 = arch->common.extensions; ext1->name != NULL; ++ext1)
{
if (ext1->alias)
continue;
arm_initialize_isa (isa1, ext1->isa_bits);
for (ext2 = ext1 + 1; ext2->name != NULL; ++ext2)
{
if (ext2->alias || ext1->remove != ext2->remove)
continue;
arm_initialize_isa (isa2, ext2->isa_bits);
ASSERT_TRUE (!bitmap_subset_p (isa2, isa1));
ASSERT_TRUE (!bitmap_intersect_p (isa2, target_isa));
}
}
}
for (cpu = all_cores; cpu->common.name != NULL; ++cpu)
{
const cpu_arch_extension *ext1, *ext2;
if (cpu->common.extensions == NULL)
continue;
arm_initialize_isa (target_isa, arch->common.isa_bits);
for (ext1 = cpu->common.extensions; ext1->name != NULL; ++ext1)
{
if (ext1->alias)
continue;
arm_initialize_isa (isa1, ext1->isa_bits);
for (ext2 = ext1 + 1; ext2->name != NULL; ++ext2)
{
if (ext2->alias || ext1->remove != ext2->remove)
continue;
arm_initialize_isa (isa2, ext2->isa_bits);
ASSERT_TRUE (!bitmap_subset_p (isa2, isa1));
ASSERT_TRUE (!bitmap_intersect_p (isa2, target_isa));
}
}
}
}
static void
arm_test_fpu_data (void)
{
auto_sbitmap isa_all_fpubits (isa_num_bits);
auto_sbitmap fpubits (isa_num_bits);
auto_sbitmap tmpset (isa_num_bits);
static const enum isa_feature fpu_bitlist[]
= { ISA_ALL_FPU_INTERNAL, isa_nobit };
arm_initialize_isa (isa_all_fpubits, fpu_bitlist);
for (unsigned int i = 0; i < TARGET_FPU_auto; i++)
{
arm_initialize_isa (fpubits, all_fpus[i].isa_bits);
bitmap_and_compl (tmpset, isa_all_fpubits, fpubits);
bitmap_clear (isa_all_fpubits);
bitmap_copy (isa_all_fpubits, tmpset);
}
if (!bitmap_empty_p (isa_all_fpubits))
{
fprintf (stderr, "Error: found feature bits in the ALL_FPU_INTERAL"
" group that are not defined by any FPU.\n"
"       Check your arm-cpus.in.\n");
ASSERT_TRUE (bitmap_empty_p (isa_all_fpubits));
}
}
static void
arm_run_selftests (void)
{
arm_test_cpu_arch_data ();
arm_test_fpu_data ();
}
} 
#undef TARGET_RUN_TARGET_SELFTESTS
#define TARGET_RUN_TARGET_SELFTESTS selftest::arm_run_selftests
#endif 
struct gcc_target targetm = TARGET_INITIALIZER;
#include "gt-arm.h"
