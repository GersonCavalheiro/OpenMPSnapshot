#ifndef GCC_FUNCTION_H
#define GCC_FUNCTION_H
struct GTY(()) sequence_stack {
rtx_insn *first;
rtx_insn *last;
struct sequence_stack *next;
};

struct GTY(()) emit_status {
void ensure_regno_capacity ();
int x_reg_rtx_no;
int x_first_label_num;
struct sequence_stack seq;
int x_cur_insn_uid;
int x_cur_debug_insn_uid;
int regno_pointer_align_length;
unsigned char * GTY((skip)) regno_pointer_align;
};
extern GTY ((length ("crtl->emit.x_reg_rtx_no"))) rtx * regno_reg_rtx;
#define reg_rtx_no (crtl->emit.x_reg_rtx_no)
#define REGNO_POINTER_ALIGN(REGNO) (crtl->emit.regno_pointer_align[REGNO])
struct GTY(()) expr_status {
poly_int64_pod x_pending_stack_adjust;
int x_inhibit_defer_pop;
poly_int64_pod x_stack_pointer_delta;
rtx x_saveregs_value;
rtx x_apply_args_value;
vec<rtx_insn *, va_gc> *x_forced_labels;
};
typedef struct call_site_record_d *call_site_record;
struct GTY(()) rtl_eh {
rtx ehr_stackadj;
rtx ehr_handler;
rtx_code_label *ehr_label;
rtx sjlj_fc;
rtx_insn *sjlj_exit_after;
vec<uchar, va_gc> *action_record_data;
vec<call_site_record, va_gc> *call_site_record_v[2];
};
#define pending_stack_adjust (crtl->expr.x_pending_stack_adjust)
#define inhibit_defer_pop (crtl->expr.x_inhibit_defer_pop)
#define saveregs_value (crtl->expr.x_saveregs_value)
#define apply_args_value (crtl->expr.x_apply_args_value)
#define forced_labels (crtl->expr.x_forced_labels)
#define stack_pointer_delta (crtl->expr.x_stack_pointer_delta)
struct gimple_df;
struct call_site_record_d;
struct dw_fde_node;
struct GTY(()) varasm_status {
struct rtx_constant_pool *pool;
unsigned int deferred_constants;
};
struct GTY(()) function_subsections {
const char *hot_section_label;
const char *cold_section_label;
const char *hot_section_end_label;
const char *cold_section_end_label;
};
struct GTY(()) frame_space
{
struct frame_space *next;
poly_int64 start;
poly_int64 length;
};
struct GTY(()) stack_usage
{
HOST_WIDE_INT static_stack_size;
HOST_WIDE_INT dynamic_stack_size;
poly_int64 pushed_stack_size;
unsigned int has_unbounded_dynamic_stack_size : 1;
};
#define current_function_static_stack_size (cfun->su->static_stack_size)
#define current_function_dynamic_stack_size (cfun->su->dynamic_stack_size)
#define current_function_pushed_stack_size (cfun->su->pushed_stack_size)
#define current_function_has_unbounded_dynamic_stack_size \
(cfun->su->has_unbounded_dynamic_stack_size)
#define current_function_allocates_dynamic_stack_space    \
(current_function_dynamic_stack_size != 0               \
|| current_function_has_unbounded_dynamic_stack_size)
struct GTY(()) function {
struct eh_status *eh;
struct control_flow_graph *cfg;
gimple_seq gimple_body;
struct gimple_df *gimple_df;
struct loops *x_current_loops;
char *pass_startwith;
struct stack_usage *su;
htab_t GTY((skip)) value_histograms;
tree decl;
tree static_chain_decl;
tree nonlocal_goto_save_area;
vec<tree, va_gc> *local_decls;
struct machine_function * GTY ((maybe_undef)) machine;
struct language_function * language;
hash_set<tree> *GTY (()) used_types_hash;
struct dw_fde_node *fde;
int last_stmt_uid;
int debug_marker_count;
int funcdef_no;
location_t function_start_locus;
location_t function_end_locus;
unsigned int curr_properties;
unsigned int last_verified;
const char * GTY((skip)) cannot_be_copied_reason;
unsigned short last_clique;
unsigned int va_list_gpr_size : 8;
unsigned int va_list_fpr_size : 8;
unsigned int calls_setjmp : 1;
unsigned int calls_alloca : 1;
unsigned int has_nonlocal_label : 1;
unsigned int has_forced_label_in_static : 1;
unsigned int cannot_be_copied_set : 1;
unsigned int stdarg : 1;
unsigned int after_inlining : 1;
unsigned int always_inline_functions_inlined : 1;
unsigned int can_throw_non_call_exceptions : 1;
unsigned int can_delete_dead_exceptions : 1;
unsigned int returns_struct : 1;
unsigned int returns_pcc_struct : 1;
unsigned int has_local_explicit_reg_vars : 1;
unsigned int is_thunk : 1;
unsigned int has_force_vectorize_loops : 1;
unsigned int has_simduid_loops : 1;
unsigned int tail_call_marked : 1;
unsigned int has_unroll : 1;
unsigned int debug_nonbind_markers : 1;
};
void add_local_decl (struct function *fun, tree d);
#define FOR_EACH_LOCAL_DECL(FUN, I, D)		\
FOR_EACH_VEC_SAFE_ELT_REVERSE ((FUN)->local_decls, I, D)
#define VA_LIST_MAX_GPR_SIZE	255
#define VA_LIST_MAX_FPR_SIZE	255
extern GTY(()) struct function *cfun;
#define cfun (cfun + 0)
extern int virtuals_instantiated;
extern int trampolines_created;
struct GTY((for_user)) types_used_by_vars_entry {
tree type;
tree var_decl;
};
struct used_type_hasher : ggc_ptr_hash<types_used_by_vars_entry>
{
static hashval_t hash (types_used_by_vars_entry *);
static bool equal (types_used_by_vars_entry *, types_used_by_vars_entry *);
};
extern GTY(()) hash_table<used_type_hasher> *types_used_by_vars_hash;
void types_used_by_var_decl_insert (tree type, tree var_decl);
extern GTY(()) vec<tree, va_gc> *types_used_by_cur_var_decl;
inline struct loops *
loops_for_fn (struct function *fn)
{
return fn->x_current_loops;
}
inline void
set_loops_for_fn (struct function *fn, struct loops *loops)
{
gcc_checking_assert (fn->x_current_loops == NULL || loops == NULL);
fn->x_current_loops = loops;
}
#define current_function_funcdef_no (cfun->funcdef_no)
#define current_loops (cfun->x_current_loops)
#define dom_computed (cfun->cfg->x_dom_computed)
#define n_bbs_in_dom_tree (cfun->cfg->x_n_bbs_in_dom_tree)
#define VALUE_HISTOGRAMS(fun) (fun)->value_histograms
extern struct machine_function * (*init_machine_status) (void);
struct args_size
{
poly_int64_pod constant;
tree var;
};
struct locate_and_pad_arg_data
{
struct args_size size;
struct args_size offset;
struct args_size slot_offset;
struct args_size alignment_pad;
pad_direction where_pad;
unsigned int boundary;
};
#define ADD_PARM_SIZE(TO, INC)					\
do {								\
tree inc = (INC);						\
if (tree_fits_shwi_p (inc))					\
(TO).constant += tree_to_shwi (inc);			\
else if ((TO).var == 0)					\
(TO).var = fold_convert (ssizetype, inc);			\
else								\
(TO).var = size_binop (PLUS_EXPR, (TO).var,			\
fold_convert (ssizetype, inc));	\
} while (0)
#define SUB_PARM_SIZE(TO, DEC)					\
do {								\
tree dec = (DEC);						\
if (tree_fits_shwi_p (dec))					\
(TO).constant -= tree_to_shwi (dec);			\
else if ((TO).var == 0)					\
(TO).var = size_binop (MINUS_EXPR, ssize_int (0),		\
fold_convert (ssizetype, dec));	\
else								\
(TO).var = size_binop (MINUS_EXPR, (TO).var,		\
fold_convert (ssizetype, dec));	\
} while (0)
#define ARGS_SIZE_TREE(SIZE)					\
((SIZE).var == 0 ? ssize_int ((SIZE).constant)			\
: size_binop (PLUS_EXPR, fold_convert (ssizetype, (SIZE).var),	\
ssize_int ((SIZE).constant)))
#define ARGS_SIZE_RTX(SIZE)					\
((SIZE).var == 0 ? gen_int_mode ((SIZE).constant, Pmode)	\
: expand_normal (ARGS_SIZE_TREE (SIZE)))
#define ASLK_REDUCE_ALIGN 1
#define ASLK_RECORD_PAD 2
#define MINIMUM_METHOD_BOUNDARY \
((TARGET_PTRMEMFUNC_VBIT_LOCATION == ptrmemfunc_vbit_in_pfn)	     \
? MAX (FUNCTION_BOUNDARY, 2 * BITS_PER_UNIT) : FUNCTION_BOUNDARY)
enum stack_clash_probes {
NO_PROBE_NO_FRAME,
NO_PROBE_SMALL_FRAME,
PROBE_INLINE,
PROBE_LOOP
};
extern void dump_stack_clash_frame_info (enum stack_clash_probes, bool);

extern void push_function_context (void);
extern void pop_function_context (void);
extern void free_after_parsing (struct function *);
extern void free_after_compilation (struct function *);
extern poly_int64 get_frame_size (void);
extern bool frame_offset_overflow (poly_int64, tree);
extern unsigned int spill_slot_alignment (machine_mode);
extern rtx assign_stack_local_1 (machine_mode, poly_int64, int, int);
extern rtx assign_stack_local (machine_mode, poly_int64, int);
extern rtx assign_stack_temp_for_type (machine_mode, poly_int64, tree);
extern rtx assign_stack_temp (machine_mode, poly_int64);
extern rtx assign_temp (tree, int, int);
extern void update_temp_slot_address (rtx, rtx);
extern void preserve_temp_slots (rtx);
extern void free_temp_slots (void);
extern void push_temp_slots (void);
extern void pop_temp_slots (void);
extern void init_temp_slots (void);
extern rtx get_hard_reg_initial_reg (rtx);
extern rtx get_hard_reg_initial_val (machine_mode, unsigned int);
extern rtx has_hard_reg_initial_val (machine_mode, unsigned int);
extern unsigned int emit_initial_value_sets (void);
extern bool initial_value_entry (int i, rtx *, rtx *);
extern void instantiate_decl_rtl (rtx x);
extern int aggregate_value_p (const_tree, const_tree);
extern bool use_register_for_decl (const_tree);
extern gimple_seq gimplify_parameters (gimple_seq *);
extern void locate_and_pad_parm (machine_mode, tree, int, int, int,
tree, struct args_size *,
struct locate_and_pad_arg_data *);
extern void generate_setjmp_warnings (void);
extern void reorder_blocks (void);
extern void clear_block_marks (tree);
extern tree blocks_nreverse (tree);
extern tree block_chainon (tree, tree);
extern void number_blocks (tree);
extern void set_cfun (struct function *new_cfun, bool force = false);
extern void push_cfun (struct function *new_cfun);
extern void pop_cfun (void);
extern int get_next_funcdef_no (void);
extern int get_last_funcdef_no (void);
extern void allocate_struct_function (tree, bool);
extern void push_struct_function (tree fndecl);
extern void push_dummy_function (bool);
extern void pop_dummy_function (void);
extern void init_dummy_function_start (void);
extern void init_function_start (tree);
extern void stack_protect_epilogue (void);
extern void expand_function_start (tree);
extern void expand_dummy_function_end (void);
extern void thread_prologue_and_epilogue_insns (void);
extern void diddle_return_value (void (*)(rtx, void*), void*);
extern void clobber_return_register (void);
extern void expand_function_end (void);
extern rtx get_arg_pointer_save_area (void);
extern void maybe_copy_prologue_epilogue_insn (rtx, rtx);
extern int prologue_contains (const rtx_insn *);
extern int epilogue_contains (const rtx_insn *);
extern int prologue_epilogue_contains (const rtx_insn *);
extern void record_prologue_seq (rtx_insn *);
extern void record_epilogue_seq (rtx_insn *);
extern void emit_return_into_block (bool simple_p, basic_block bb);
extern void set_return_jump_label (rtx_insn *);
extern bool active_insn_between (rtx_insn *head, rtx_insn *tail);
extern vec<edge> convert_jumps_to_returns (basic_block last_bb, bool simple_p,
vec<edge> unconverted);
extern basic_block emit_return_for_exit (edge exit_fallthru_edge,
bool simple_p);
extern void reposition_prologue_and_epilogue_notes (void);
extern const char *fndecl_name (tree);
extern const char *function_name (struct function *);
extern const char *current_function_name (void);
extern void used_types_insert (tree);
#endif  
