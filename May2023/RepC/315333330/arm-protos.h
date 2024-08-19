#ifndef GCC_ARM_PROTOS_H
#define GCC_ARM_PROTOS_H
#include "sbitmap.h"
extern enum unwind_info_type arm_except_unwind_info (struct gcc_options *);
extern int use_return_insn (int, rtx);
extern bool use_simple_return_p (void);
extern enum reg_class arm_regno_class (int);
extern void arm_load_pic_register (unsigned long);
extern int arm_volatile_func (void);
extern void arm_expand_prologue (void);
extern void arm_expand_epilogue (bool);
extern void arm_declare_function_name (FILE *, const char *, tree);
extern void arm_asm_declare_function_name (FILE *, const char *, tree);
extern void thumb2_expand_return (bool);
extern const char *arm_strip_name_encoding (const char *);
extern void arm_asm_output_labelref (FILE *, const char *);
extern void thumb2_asm_output_opcode (FILE *);
extern unsigned long arm_current_func_type (void);
extern HOST_WIDE_INT arm_compute_initial_elimination_offset (unsigned int,
unsigned int);
extern HOST_WIDE_INT thumb_compute_initial_elimination_offset (unsigned int,
unsigned int);
extern unsigned int arm_dbx_register_number (unsigned int);
extern void arm_output_fn_unwind (FILE *, bool);
extern rtx arm_expand_builtin (tree exp, rtx target, rtx subtarget
ATTRIBUTE_UNUSED, machine_mode mode
ATTRIBUTE_UNUSED, int ignore ATTRIBUTE_UNUSED);
extern tree arm_builtin_decl (unsigned code, bool initialize_p
ATTRIBUTE_UNUSED);
extern void arm_init_builtins (void);
extern void arm_atomic_assign_expand_fenv (tree *hold, tree *clear, tree *update);
extern rtx arm_simd_vect_par_cnst_half (machine_mode mode, bool high);
extern bool arm_simd_check_vect_par_cnst_half_p (rtx op, machine_mode mode,
bool high);
#ifdef RTX_CODE
extern void arm_gen_unlikely_cbranch (enum rtx_code, machine_mode cc_mode,
rtx label_ref);
extern bool arm_vector_mode_supported_p (machine_mode);
extern bool arm_small_register_classes_for_mode_p (machine_mode);
extern int const_ok_for_arm (HOST_WIDE_INT);
extern int const_ok_for_op (HOST_WIDE_INT, enum rtx_code);
extern int const_ok_for_dimode_op (HOST_WIDE_INT, enum rtx_code);
extern int arm_split_constant (RTX_CODE, machine_mode, rtx,
HOST_WIDE_INT, rtx, rtx, int);
extern int legitimate_pic_operand_p (rtx);
extern rtx legitimize_pic_address (rtx, machine_mode, rtx);
extern rtx legitimize_tls_address (rtx, rtx);
extern bool arm_legitimate_address_p (machine_mode, rtx, bool);
extern int arm_legitimate_address_outer_p (machine_mode, rtx, RTX_CODE, int);
extern int thumb_legitimate_offset_p (machine_mode, HOST_WIDE_INT);
extern int thumb1_legitimate_address_p (machine_mode, rtx, int);
extern bool ldm_stm_operation_p (rtx, bool, machine_mode mode,
bool, bool);
extern int arm_const_double_rtx (rtx);
extern int vfp3_const_double_rtx (rtx);
extern int neon_immediate_valid_for_move (rtx, machine_mode, rtx *, int *);
extern int neon_immediate_valid_for_logic (rtx, machine_mode, int, rtx *,
int *);
extern int neon_immediate_valid_for_shift (rtx, machine_mode, rtx *,
int *, bool);
extern char *neon_output_logic_immediate (const char *, rtx *,
machine_mode, int, int);
extern char *neon_output_shift_immediate (const char *, char, rtx *,
machine_mode, int, bool);
extern void neon_pairwise_reduce (rtx, rtx, machine_mode,
rtx (*) (rtx, rtx, rtx));
extern rtx neon_make_constant (rtx);
extern tree arm_builtin_vectorized_function (unsigned int, tree, tree);
extern void neon_expand_vector_init (rtx, rtx);
extern void neon_lane_bounds (rtx, HOST_WIDE_INT, HOST_WIDE_INT, const_tree);
extern void arm_const_bounds (rtx, HOST_WIDE_INT, HOST_WIDE_INT);
extern HOST_WIDE_INT neon_element_bits (machine_mode);
extern void neon_emit_pair_result_insn (machine_mode,
rtx (*) (rtx, rtx, rtx, rtx),
rtx, rtx, rtx);
extern void neon_disambiguate_copy (rtx *, rtx *, rtx *, unsigned int);
extern void neon_split_vcombine (rtx op[3]);
extern enum reg_class coproc_secondary_reload_class (machine_mode, rtx,
bool);
extern bool arm_tls_referenced_p (rtx);
extern int arm_coproc_mem_operand (rtx, bool);
extern int neon_vector_mem_operand (rtx, int, bool);
extern int neon_struct_mem_operand (rtx);
extern int tls_mentioned_p (rtx);
extern int symbol_mentioned_p (rtx);
extern int label_mentioned_p (rtx);
extern RTX_CODE minmax_code (rtx);
extern bool arm_sat_operator_match (rtx, rtx, int *, bool *);
extern int adjacent_mem_locations (rtx, rtx);
extern bool gen_ldm_seq (rtx *, int, bool);
extern bool gen_stm_seq (rtx *, int);
extern bool gen_const_stm_seq (rtx *, int);
extern rtx arm_gen_load_multiple (int *, int, rtx, int, rtx, HOST_WIDE_INT *);
extern rtx arm_gen_store_multiple (int *, int, rtx, int, rtx, HOST_WIDE_INT *);
extern bool offset_ok_for_ldrd_strd (HOST_WIDE_INT);
extern bool operands_ok_ldrd_strd (rtx, rtx, rtx, HOST_WIDE_INT, bool, bool);
extern bool gen_operands_ldrd_strd (rtx *, bool, bool, bool);
extern int arm_gen_movmemqi (rtx *);
extern bool gen_movmem_ldrd_strd (rtx *);
extern machine_mode arm_select_cc_mode (RTX_CODE, rtx, rtx);
extern machine_mode arm_select_dominance_cc_mode (rtx, rtx,
HOST_WIDE_INT);
extern rtx arm_gen_compare_reg (RTX_CODE, rtx, rtx, rtx);
extern rtx arm_gen_return_addr_mask (void);
extern void arm_reload_in_hi (rtx *);
extern void arm_reload_out_hi (rtx *);
extern int arm_max_const_double_inline_cost (void);
extern int arm_const_double_inline_cost (rtx);
extern bool arm_const_double_by_parts (rtx);
extern bool arm_const_double_by_immediates (rtx);
extern void arm_emit_call_insn (rtx, rtx, bool);
bool detect_cmse_nonsecure_call (tree);
extern const char *output_call (rtx *);
void arm_emit_movpair (rtx, rtx);
extern const char *output_mov_long_double_arm_from_arm (rtx *);
extern const char *output_move_double (rtx *, bool, int *count);
extern const char *output_move_quad (rtx *);
extern int arm_count_output_move_double_insns (rtx *);
extern const char *output_move_vfp (rtx *operands);
extern const char *output_move_neon (rtx *operands);
extern int arm_attr_length_move_neon (rtx_insn *);
extern int arm_address_offset_is_imm (rtx_insn *);
extern const char *output_add_immediate (rtx *);
extern const char *arithmetic_instr (rtx, int);
extern void output_ascii_pseudo_op (FILE *, const unsigned char *, int);
extern const char *output_return_instruction (rtx, bool, bool, bool);
extern const char *output_probe_stack_range (rtx, rtx);
extern void arm_poke_function_name (FILE *, const char *);
extern void arm_final_prescan_insn (rtx_insn *);
extern int arm_debugger_arg_offset (int, rtx);
extern bool arm_is_long_call_p (tree);
extern int    arm_emit_vector_const (FILE *, rtx);
extern void arm_emit_fp16_const (rtx c);
extern const char * arm_output_load_gr (rtx *);
extern const char *vfp_output_vstmd (rtx *);
extern void arm_output_multireg_pop (rtx *, bool, rtx, bool, bool);
extern void arm_set_return_address (rtx, rtx);
extern int arm_eliminable_register (rtx);
extern const char *arm_output_shift(rtx *, int);
extern const char *arm_output_iwmmxt_shift_immediate (const char *, rtx *, bool);
extern const char *arm_output_iwmmxt_tinsr (rtx *);
extern unsigned int arm_sync_loop_insns (rtx , rtx *);
extern int arm_attr_length_push_multi(rtx, rtx);
extern int arm_attr_length_pop_multi(rtx *, bool, bool);
extern void arm_expand_compare_and_swap (rtx op[]);
extern void arm_split_compare_and_swap (rtx op[]);
extern void arm_split_atomic_op (enum rtx_code, rtx, rtx, rtx, rtx, rtx, rtx);
extern rtx arm_load_tp (rtx);
extern bool arm_coproc_builtin_available (enum unspecv);
extern bool arm_coproc_ldc_stc_legitimate_address (rtx);
#if defined TREE_CODE
extern void arm_init_cumulative_args (CUMULATIVE_ARGS *, tree, rtx, tree);
extern bool arm_pad_reg_upward (machine_mode, tree, int);
#endif
extern int arm_apply_result_size (void);
#endif 
extern void arm_init_expanders (void);
extern const char *thumb1_unexpanded_epilogue (void);
extern void thumb1_expand_prologue (void);
extern void thumb1_expand_epilogue (void);
extern const char *thumb1_output_interwork (void);
extern int thumb_shiftable_const (unsigned HOST_WIDE_INT);
#ifdef RTX_CODE
extern enum arm_cond_code maybe_get_arm_condition_code (rtx);
extern void thumb1_final_prescan_insn (rtx_insn *);
extern void thumb2_final_prescan_insn (rtx_insn *);
extern const char *thumb_load_double_from_address (rtx *);
extern const char *thumb_output_move_mem_multiple (int, rtx *);
extern const char *thumb_call_via_reg (rtx);
extern void thumb_expand_movmemqi (rtx *);
extern rtx arm_return_addr (int, rtx);
extern void thumb_reload_out_hi (rtx *);
extern void thumb_set_return_address (rtx, rtx);
extern const char *thumb1_output_casesi (rtx *);
extern const char *thumb2_output_casesi (rtx *);
#endif
extern int arm_dllexport_name_p (const char *);
extern int arm_dllimport_name_p (const char *);
#ifdef TREE_CODE
extern void arm_pe_unique_section (tree, int);
extern void arm_pe_encode_section_info (tree, rtx, int);
extern int arm_dllexport_p (tree);
extern int arm_dllimport_p (tree);
extern void arm_mark_dllexport (tree);
extern void arm_mark_dllimport (tree);
extern bool arm_change_mode_p (tree);
#endif
extern tree arm_valid_target_attribute_tree (tree, struct gcc_options *,
struct gcc_options *);
extern void arm_configure_build_target (struct arm_build_target *,
struct cl_target_option *,
struct gcc_options *, bool);
extern void arm_option_reconfigure_globals (void);
extern void arm_options_perform_arch_sanity_checks (void);
extern void arm_pr_long_calls (struct cpp_reader *);
extern void arm_pr_no_long_calls (struct cpp_reader *);
extern void arm_pr_long_calls_off (struct cpp_reader *);
extern const char *arm_mangle_type (const_tree);
extern const char *arm_mangle_builtin_type (const_tree);
extern void arm_order_regs_for_local_alloc (void);
extern int arm_max_conditional_execute ();
struct cpu_vec_costs {
const int scalar_stmt_cost;   
const int scalar_load_cost;   
const int scalar_store_cost;  
const int vec_stmt_cost;      
const int vec_to_scalar_cost;    
const int scalar_to_vec_cost;    
const int vec_align_load_cost;   
const int vec_unalign_load_cost; 
const int vec_unalign_store_cost; 
const int vec_store_cost;        
const int cond_taken_branch_cost;    
const int cond_not_taken_branch_cost;
};
#ifdef RTX_CODE
struct cpu_cost_table;
enum arm_addr_mode_op
{
AMO_DEFAULT,
AMO_NO_WB,	
AMO_WB,	
AMO_MAX	
};
struct addr_mode_cost_table
{
const int integer[AMO_MAX];
const int fp[AMO_MAX];
const int vector[AMO_MAX];
};
struct tune_params
{
const struct cpu_cost_table *insn_extra_cost;
const struct addr_mode_cost_table *addr_mode_costs;
bool (*sched_adjust_cost) (rtx_insn *, int, rtx_insn *, int *);
int (*branch_cost) (bool, bool);
const struct cpu_vec_costs* vec_costs;
int constant_limit;
int max_insns_skipped;
int max_insns_inline_memset;
unsigned int issue_rate;
struct
{
int num_slots;
int l1_cache_size;
int l1_cache_line_size;
} prefetch;
enum {PREF_CONST_POOL_FALSE, PREF_CONST_POOL_TRUE}
prefer_constant_pool: 1;
enum {PREF_LDRD_FALSE, PREF_LDRD_TRUE} prefer_ldrd_strd: 1;
enum log_op_non_short_circuit {LOG_OP_NON_SHORT_CIRCUIT_FALSE,
LOG_OP_NON_SHORT_CIRCUIT_TRUE};
log_op_non_short_circuit logical_op_non_short_circuit_thumb: 1;
log_op_non_short_circuit logical_op_non_short_circuit_arm: 1;
enum {DISPARAGE_FLAGS_NEITHER, DISPARAGE_FLAGS_PARTIAL, DISPARAGE_FLAGS_ALL}
disparage_flag_setting_t16_encodings: 2;
enum {PREF_NEON_64_FALSE, PREF_NEON_64_TRUE} prefer_neon_for_64bits: 1;
enum {PREF_NEON_STRINGOPS_FALSE, PREF_NEON_STRINGOPS_TRUE}
string_ops_prefer_neon: 1;
enum fuse_ops
{
FUSE_NOTHING   = 0,
FUSE_MOVW_MOVT = 1 << 0,
FUSE_AES_AESMC = 1 << 1
} fusible_ops: 2;
enum {SCHED_AUTOPREF_OFF, SCHED_AUTOPREF_RANK, SCHED_AUTOPREF_FULL}
sched_autopref: 2;
};
#define FUSE_OPS(x) ((tune_params::fuse_ops) (x))
extern const struct tune_params *current_tune;
extern int vfp3_const_double_for_fract_bits (rtx);
extern int vfp3_const_double_for_bits (rtx);
extern void arm_emit_coreregs_64bit_shift (enum rtx_code, rtx, rtx, rtx, rtx,
rtx);
extern bool arm_fusion_enabled_p (tune_params::fuse_ops);
extern bool arm_valid_symbolic_address_p (rtx);
extern bool arm_validize_comparison (rtx *, rtx *, rtx *);
#endif 
extern bool arm_gen_setmem (rtx *);
extern void arm_expand_vec_perm (rtx target, rtx op0, rtx op1, rtx sel);
extern bool arm_autoinc_modes_ok_p (machine_mode, enum arm_auto_incmodes);
extern void arm_emit_eabi_attribute (const char *, int, int);
extern void arm_reset_previous_fndecl (void);
extern void save_restore_target_globals (tree);
extern const char *arm_rewrite_selected_cpu (const char *name);
extern void arm_lang_object_attributes_init (void);
extern void arm_register_target_pragmas (void);
extern void arm_cpu_cpp_builtins (struct cpp_reader *);
extern bool arm_is_constant_pool_ref (rtx);
extern unsigned int tune_flags;
extern int arm_arch3m;
extern int arm_arch4;
extern int arm_arch4t;
extern int arm_arch5;
extern int arm_arch5e;
extern int arm_arch6;
extern int arm_arch6k;
extern int arm_arch6kz;
extern int arm_arch6m;
extern int arm_arch7;
extern int arm_arch_lpae;
extern int arm_arch_notm;
extern int arm_arch7em;
extern int arm_arch8;
extern int arm_ld_sched;
extern int arm_tune_strongarm;
extern int arm_arch_iwmmxt;
extern int arm_arch_iwmmxt2;
extern int arm_arch_xscale;
extern int arm_tune_xscale;
extern int arm_tune_wbuf;
extern int arm_tune_cortex_a9;
extern int arm_cpp_interwork;
extern int arm_arch_thumb1;
extern int arm_arch_thumb2;
extern int arm_arch_arm_hwdiv;
extern int arm_arch_thumb_hwdiv;
extern int arm_arch_no_volatile_ce;
extern int prefer_neon_for_64bits;
struct arm_build_target
{
const char *core_name;
const char *arch_name;
const char *arch_pp_name;
enum base_architecture base_arch;
char profile;
sbitmap isa;
unsigned int tune_flags;
const struct tune_params *tune;
enum processor_type tune_core;
};
extern struct arm_build_target arm_active_target;
struct cpu_arch_extension
{
const char *const name;
bool remove;
bool alias;
const enum isa_feature isa_bits[isa_num_bits];
};
struct cpu_arch_option
{
const char *name;
const struct cpu_arch_extension *extensions;
enum isa_feature isa_bits[isa_num_bits];
};
struct arch_option
{
cpu_arch_option common;
const char *arch;
enum base_architecture base_arch;
const char profile;
enum processor_type tune_id;
};
struct cpu_option
{
cpu_arch_option common;
enum arch_type arch;
};
extern const arch_option all_architectures[];
extern const cpu_option all_cores[];
const cpu_option *arm_parse_cpu_option_name (const cpu_option *, const char *,
const char *, bool = true);
const arch_option *arm_parse_arch_option_name (const arch_option *,
const char *, const char *, bool = true);
void arm_parse_option_features (sbitmap, const cpu_arch_option *,
const char *);
void arm_initialize_isa (sbitmap, const enum isa_feature *);
#endif 
