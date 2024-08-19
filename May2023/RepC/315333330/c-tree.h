#ifndef GCC_C_TREE_H
#define GCC_C_TREE_H
#include "c-family/c-common.h"
#include "diagnostic.h"
#define C_SIZEOF_STRUCT_LANG_IDENTIFIER \
(sizeof (struct c_common_identifier) + 3 * sizeof (void *))
#define C_TYPE_FIELDS_READONLY(TYPE) TREE_LANG_FLAG_1 (TYPE)
#define C_TYPE_FIELDS_VOLATILE(TYPE) TREE_LANG_FLAG_2 (TYPE)
#define C_TYPE_BEING_DEFINED(TYPE) TYPE_LANG_FLAG_0 (TYPE)
#define C_TYPE_INCOMPLETE_VARS(TYPE) TYPE_VFIELD (TYPE)
#define C_IS_RESERVED_WORD(ID) TREE_LANG_FLAG_0 (ID)
#define C_TYPE_VARIABLE_SIZE(TYPE) TYPE_LANG_FLAG_1 (TYPE)
#define C_DECL_VARIABLE_SIZE(TYPE) DECL_LANG_FLAG_0 (TYPE)
#define C_TYPE_DEFINED_IN_STRUCT(TYPE) TYPE_LANG_FLAG_2 (TYPE)
#define C_TYPE_ERROR_REPORTED(TYPE) TYPE_LANG_FLAG_3 (TYPE)
#define C_TYPEDEF_EXPLICITLY_SIGNED(EXP) DECL_LANG_FLAG_1 (EXP)
#define C_FUNCTION_IMPLICIT_INT(EXP) DECL_LANG_FLAG_1 (EXP)
#define C_DECL_IMPLICIT(EXP) DECL_LANG_FLAG_2 (EXP)
#define C_ARRAY_PARAMETER(NODE) DECL_LANG_FLAG_0 (NODE)
#define C_DECL_DECLARED_BUILTIN(EXP)		\
DECL_LANG_FLAG_3 (FUNCTION_DECL_CHECK (EXP))
#define C_DECL_BUILTIN_PROTOTYPE(EXP)		\
DECL_LANG_FLAG_6 (FUNCTION_DECL_CHECK (EXP))
#define C_DECL_REGISTER(EXP) DECL_LANG_FLAG_4 (EXP)
#define C_DECL_USED(EXP) DECL_LANG_FLAG_5 (FUNCTION_DECL_CHECK (EXP))
#define C_DECL_THREADPRIVATE_P(DECL) DECL_LANG_FLAG_3 (VAR_DECL_CHECK (DECL))
#define C_DECL_ISNT_PROTOTYPE(EXP)			\
(EXP == 0					\
|| (!prototype_p (TREE_TYPE (EXP))	\
&& !DECL_BUILT_IN (EXP)))
#define TYPE_ACTUAL_ARG_TYPES(NODE) TYPE_LANG_SLOT_1 (NODE)
#define CONSTRUCTOR_NON_CONST(EXPR) TREE_LANG_FLAG_1 (CONSTRUCTOR_CHECK (EXPR))
#define SAVE_EXPR_FOLDED_P(EXP)	TREE_LANG_FLAG_1 (SAVE_EXPR_CHECK (EXP))
struct c_expr
{
tree value;
enum tree_code original_code;
tree original_type;
source_range src_range;
location_t get_start () const { return src_range.m_start; }
location_t get_finish () const { return src_range.m_finish; }
location_t get_location () const
{
if (EXPR_HAS_LOCATION (value))
return EXPR_LOCATION (value);
else
return make_location (get_start (), get_start (), get_finish ());
}
void set_error ()
{
value = error_mark_node;
src_range.m_start = UNKNOWN_LOCATION;
src_range.m_finish = UNKNOWN_LOCATION;
}
};
typedef struct c_expr c_expr_t;
enum c_typespec_kind {
ctsk_none,
ctsk_resword,
ctsk_tagref,
ctsk_tagfirstref,
ctsk_tagdef,
ctsk_typedef,
ctsk_objc,
ctsk_typeof
};
struct c_typespec {
enum c_typespec_kind kind;
bool expr_const_operands;
tree spec;
tree expr;
};
enum c_storage_class {
csc_none,
csc_auto,
csc_extern,
csc_register,
csc_static,
csc_typedef
};
enum c_typespec_keyword {
cts_none,
cts_void,
cts_bool,
cts_char,
cts_int,
cts_float,
cts_int_n,
cts_double,
cts_dfloat32,
cts_dfloat64,
cts_dfloat128,
cts_floatn_nx,
cts_fract,
cts_accum,
cts_auto_type
};
enum c_declspec_word {
cdw_typespec ,
cdw_storage_class  ,
cdw_attributes,
cdw_typedef,
cdw_explicit_signed,
cdw_deprecated,
cdw_default_int,
cdw_long,
cdw_long_long,
cdw_short,
cdw_signed,
cdw_unsigned,
cdw_complex,
cdw_inline,
cdw_noreturn,
cdw_thread,
cdw_const,
cdw_volatile,
cdw_restrict,
cdw_atomic,
cdw_saturating,
cdw_alignas,
cdw_address_space,
cdw_gimple,
cdw_rtl,
cdw_number_of_elements 
};
struct c_declspecs {
source_location locations[cdw_number_of_elements];
tree type;
tree expr;
tree decl_attr;
tree attrs;
char *gimple_or_rtl_pass;
int align_log;
int int_n_idx;
int floatn_nx_idx;
enum c_storage_class storage_class;
ENUM_BITFIELD (c_typespec_keyword) typespec_word : 8;
ENUM_BITFIELD (c_typespec_kind) typespec_kind : 3;
BOOL_BITFIELD expr_const_operands : 1;
BOOL_BITFIELD declspecs_seen_p : 1;
BOOL_BITFIELD non_sc_seen_p : 1;
BOOL_BITFIELD typedef_p : 1;
BOOL_BITFIELD explicit_signed_p : 1;
BOOL_BITFIELD deprecated_p : 1;
BOOL_BITFIELD default_int_p : 1;
BOOL_BITFIELD long_p : 1;
BOOL_BITFIELD long_long_p : 1;
BOOL_BITFIELD short_p : 1;
BOOL_BITFIELD signed_p : 1;
BOOL_BITFIELD unsigned_p : 1;
BOOL_BITFIELD complex_p : 1;
BOOL_BITFIELD inline_p : 1;
BOOL_BITFIELD noreturn_p : 1;
BOOL_BITFIELD thread_p : 1;
BOOL_BITFIELD thread_gnu_p : 1;
BOOL_BITFIELD const_p : 1;
BOOL_BITFIELD volatile_p : 1;
BOOL_BITFIELD restrict_p : 1;
BOOL_BITFIELD atomic_p : 1;
BOOL_BITFIELD saturating_p : 1;
BOOL_BITFIELD alignas_p : 1;
BOOL_BITFIELD gimple_p : 1;
BOOL_BITFIELD rtl_p : 1;
addr_space_t address_space;
};
enum c_declarator_kind {
cdk_id,
cdk_function,
cdk_array,
cdk_pointer,
cdk_attrs
};
struct c_arg_tag {
tree id;
tree type;
};
struct c_arg_info {
tree parms;
vec<c_arg_tag, va_gc> *tags;
tree types;
tree others;
tree pending_sizes;
BOOL_BITFIELD had_vla_unspec : 1;
};
struct c_declarator {
enum c_declarator_kind kind;
location_t id_loc; 
struct c_declarator *declarator;
union {
tree id;
struct c_arg_info *arg_info;
struct {
tree dimen;
int quals;
tree attrs;
BOOL_BITFIELD static_p : 1;
BOOL_BITFIELD vla_unspec_p : 1;
} array;
int pointer_quals;
tree attrs;
} u;
};
struct c_type_name {
struct c_declspecs *specs;
struct c_declarator *declarator;
};
struct c_parm {
struct c_declspecs *specs;
tree attrs;
struct c_declarator *declarator;
location_t loc;
};
struct c_enum_contents
{
tree enum_next_value;
int enum_overflow;
};
enum c_inline_static_type {
csi_internal,
csi_modifiable
};

extern void c_parse_init (void);
extern bool c_keyword_starts_typename (enum rid keyword);
extern void gen_aux_info_record (tree, int, int, int);
struct c_spot_bindings;
struct c_struct_parse_info;
extern struct obstack parser_obstack;
extern tree c_break_label;
extern tree c_cont_label;
extern bool global_bindings_p (void);
extern tree pushdecl (tree);
extern void push_scope (void);
extern tree pop_scope (void);
extern void c_bindings_start_stmt_expr (struct c_spot_bindings *);
extern void c_bindings_end_stmt_expr (struct c_spot_bindings *);
extern void record_inline_static (location_t, tree, tree,
enum c_inline_static_type);
extern void c_init_decl_processing (void);
extern void c_print_identifier (FILE *, tree, int);
extern int quals_from_declspecs (const struct c_declspecs *);
extern struct c_declarator *build_array_declarator (location_t, tree,
struct c_declspecs *,
bool, bool);
extern tree build_enumerator (location_t, location_t, struct c_enum_contents *,
tree, tree);
extern tree check_for_loop_decls (location_t, bool);
extern void mark_forward_parm_decls (void);
extern void declare_parm_level (void);
extern void undeclared_variable (location_t, tree);
extern tree lookup_label_for_goto (location_t, tree);
extern tree declare_label (tree);
extern tree define_label (location_t, tree);
extern struct c_spot_bindings *c_get_switch_bindings (void);
extern void c_release_switch_bindings (struct c_spot_bindings *);
extern bool c_check_switch_jump_warnings (struct c_spot_bindings *,
location_t, location_t);
extern void finish_decl (tree, location_t, tree, tree, tree);
extern tree finish_enum (tree, tree, tree);
extern void finish_function (void);
extern tree finish_struct (location_t, tree, tree, tree,
struct c_struct_parse_info *);
extern struct c_arg_info *build_arg_info (void);
extern struct c_arg_info *get_parm_info (bool, tree);
extern tree grokfield (location_t, struct c_declarator *,
struct c_declspecs *, tree, tree *);
extern tree groktypename (struct c_type_name *, tree *, bool *);
extern tree grokparm (const struct c_parm *, tree *);
extern tree implicitly_declare (location_t, tree);
extern void keep_next_level (void);
extern void pending_xref_error (void);
extern void c_push_function_context (void);
extern void c_pop_function_context (void);
extern void push_parm_decl (const struct c_parm *, tree *);
extern struct c_declarator *set_array_declarator_inner (struct c_declarator *,
struct c_declarator *);
extern tree c_builtin_function (tree);
extern tree c_builtin_function_ext_scope (tree);
extern void shadow_tag (const struct c_declspecs *);
extern void shadow_tag_warned (const struct c_declspecs *, int);
extern tree start_enum (location_t, struct c_enum_contents *, tree);
extern bool start_function (struct c_declspecs *, struct c_declarator *, tree);
extern tree start_decl (struct c_declarator *, struct c_declspecs *, bool,
tree);
extern tree start_struct (location_t, enum tree_code, tree,
struct c_struct_parse_info **);
extern void store_parm_decls (void);
extern void store_parm_decls_from (struct c_arg_info *);
extern void temp_store_parm_decls (tree, tree);
extern void temp_pop_parm_decls (void);
extern tree xref_tag (enum tree_code, tree);
extern struct c_typespec parser_xref_tag (location_t, enum tree_code, tree);
extern struct c_parm *build_c_parm (struct c_declspecs *, tree,
struct c_declarator *, location_t);
extern struct c_declarator *build_attrs_declarator (tree,
struct c_declarator *);
extern struct c_declarator *build_function_declarator (struct c_arg_info *,
struct c_declarator *);
extern struct c_declarator *build_id_declarator (tree);
extern struct c_declarator *make_pointer_declarator (struct c_declspecs *,
struct c_declarator *);
extern struct c_declspecs *build_null_declspecs (void);
extern struct c_declspecs *declspecs_add_qual (source_location,
struct c_declspecs *, tree);
extern struct c_declspecs *declspecs_add_type (location_t,
struct c_declspecs *,
struct c_typespec);
extern struct c_declspecs *declspecs_add_scspec (source_location,
struct c_declspecs *, tree);
extern struct c_declspecs *declspecs_add_attrs (source_location,
struct c_declspecs *, tree);
extern struct c_declspecs *declspecs_add_addrspace (source_location,
struct c_declspecs *,
addr_space_t);
extern struct c_declspecs *declspecs_add_alignas (source_location,
struct c_declspecs *, tree);
extern struct c_declspecs *finish_declspecs (struct c_declspecs *);
extern bool c_objc_common_init (void);
extern bool c_missing_noreturn_ok_p (tree);
extern bool c_warn_unused_global_decl (const_tree);
extern void c_initialize_diagnostics (diagnostic_context *);
extern bool c_vla_unspec_p (tree x, tree fn);
extern int in_alignof;
extern int in_sizeof;
extern int in_typeof;
extern tree c_last_sizeof_arg;
extern location_t c_last_sizeof_loc;
extern struct c_switch *c_switch_stack;
extern tree c_objc_common_truthvalue_conversion (location_t, tree);
extern tree require_complete_type (location_t, tree);
extern bool same_translation_unit_p (const_tree, const_tree);
extern int comptypes (tree, tree);
extern int comptypes_check_different_types (tree, tree, bool *);
extern bool c_vla_type_p (const_tree);
extern bool c_mark_addressable (tree, bool = false);
extern void c_incomplete_type_error (location_t, const_tree, const_tree);
extern tree c_type_promotes_to (tree);
extern struct c_expr default_function_array_conversion (location_t,
struct c_expr);
extern struct c_expr default_function_array_read_conversion (location_t,
struct c_expr);
extern struct c_expr convert_lvalue_to_rvalue (location_t, struct c_expr,
bool, bool);
extern tree decl_constant_value_1 (tree, bool);
extern void mark_exp_read (tree);
extern tree composite_type (tree, tree);
extern tree build_component_ref (location_t, tree, tree, location_t);
extern tree build_array_ref (location_t, tree, tree);
extern tree build_external_ref (location_t, tree, bool, tree *);
extern void pop_maybe_used (bool);
extern struct c_expr c_expr_sizeof_expr (location_t, struct c_expr);
extern struct c_expr c_expr_sizeof_type (location_t, struct c_type_name *);
extern struct c_expr parser_build_unary_op (location_t, enum tree_code,
struct c_expr);
extern struct c_expr parser_build_binary_op (location_t,
enum tree_code, struct c_expr,
struct c_expr);
extern tree build_conditional_expr (location_t, tree, bool, tree, tree,
location_t, tree, tree, location_t);
extern tree build_compound_expr (location_t, tree, tree);
extern tree c_cast_expr (location_t, struct c_type_name *, tree);
extern tree build_c_cast (location_t, tree, tree);
extern void store_init_value (location_t, tree, tree, tree);
extern void maybe_warn_string_init (location_t, tree, struct c_expr);
extern void start_init (tree, tree, int, rich_location *);
extern void finish_init (void);
extern void really_start_incremental_init (tree);
extern void finish_implicit_inits (location_t, struct obstack *);
extern void push_init_level (location_t, int, struct obstack *);
extern struct c_expr pop_init_level (location_t, int, struct obstack *,
location_t);
extern void set_init_index (location_t, tree, tree, struct obstack *);
extern void set_init_label (location_t, tree, location_t, struct obstack *);
extern void process_init_element (location_t, struct c_expr, bool,
struct obstack *);
extern tree build_compound_literal (location_t, tree, tree, bool,
unsigned int);
extern void check_compound_literal_type (location_t, struct c_type_name *);
extern tree c_start_case (location_t, location_t, tree, bool);
extern void c_finish_case (tree, tree);
extern tree build_asm_expr (location_t, tree, tree, tree, tree, tree, bool,
bool);
extern tree build_asm_stmt (bool, tree);
extern int c_types_compatible_p (tree, tree);
extern tree c_begin_compound_stmt (bool);
extern tree c_end_compound_stmt (location_t, tree, bool);
extern void c_finish_if_stmt (location_t, tree, tree, tree);
extern void c_finish_loop (location_t, tree, tree, tree, tree, tree, bool);
extern tree c_begin_stmt_expr (void);
extern tree c_finish_stmt_expr (location_t, tree);
extern tree c_process_expr_stmt (location_t, tree);
extern tree c_finish_expr_stmt (location_t, tree);
extern tree c_finish_return (location_t, tree, tree);
extern tree c_finish_bc_stmt (location_t, tree *, bool);
extern tree c_finish_goto_label (location_t, tree);
extern tree c_finish_goto_ptr (location_t, tree);
extern tree c_expr_to_decl (tree, bool *, bool *);
extern tree c_finish_omp_construct (location_t, enum tree_code, tree, tree);
extern tree c_finish_oacc_data (location_t, tree, tree);
extern tree c_finish_oacc_host_data (location_t, tree, tree);
extern tree c_begin_omp_parallel (void);
extern tree c_finish_omp_parallel (location_t, tree, tree);
extern tree c_begin_omp_task (void);
extern tree c_finish_omp_task (location_t, tree, tree);
extern void c_finish_omp_cancel (location_t, tree);
extern void c_finish_omp_cancellation_point (location_t, tree);
extern tree c_finish_omp_clauses (tree, enum c_omp_region_type);
extern tree c_build_va_arg (location_t, tree, location_t, tree);
extern tree c_finish_transaction (location_t, tree, int);
extern bool c_tree_equal (tree, tree);
extern tree c_build_function_call_vec (location_t, vec<location_t>, tree,
vec<tree, va_gc> *, vec<tree, va_gc> *);
extern tree c_omp_clause_copy_ctor (tree, tree, tree);
extern int current_function_returns_value;
extern int current_function_returns_null;
extern int current_function_returns_abnormally;
enum c_oracle_request
{
C_ORACLE_SYMBOL,
C_ORACLE_TAG,
C_ORACLE_LABEL
};
typedef void c_binding_oracle_function (enum c_oracle_request, tree identifier);
extern c_binding_oracle_function *c_binding_oracle;
extern void c_finish_incomplete_decl (tree);
extern tree c_omp_reduction_id (enum tree_code, tree);
extern tree c_omp_reduction_decl (tree);
extern tree c_omp_reduction_lookup (tree, tree);
extern tree c_check_omp_declare_reduction_r (tree *, int *, void *);
extern void c_pushtag (location_t, tree, tree);
extern void c_bind (location_t, tree, bool);
extern bool tag_exists_p (enum tree_code, tree);
extern bool pedwarn_c90 (location_t, int opt, const char *, ...)
ATTRIBUTE_GCC_DIAG(3,4);
extern bool pedwarn_c99 (location_t, int opt, const char *, ...)
ATTRIBUTE_GCC_DIAG(3,4);
extern void
set_c_expr_source_range (c_expr *expr,
location_t start, location_t finish);
extern void
set_c_expr_source_range (c_expr *expr,
source_range src_range);
extern vec<tree> incomplete_record_decls;
#if CHECKING_P
namespace selftest {
extern void run_c_tests (void);
} 
#endif 
#endif 
