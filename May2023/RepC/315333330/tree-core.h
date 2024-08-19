#ifndef GCC_TREE_CORE_H
#define GCC_TREE_CORE_H
#include "symtab.h"
struct function;
struct real_value;
struct fixed_value;
struct ptr_info_def;
struct range_info_def;
struct die_struct;
#define ECF_CONST		  (1 << 0)
#define ECF_PURE		  (1 << 1)
#define ECF_LOOPING_CONST_OR_PURE (1 << 2)
#define ECF_NORETURN		  (1 << 3)
#define ECF_MALLOC		  (1 << 4)
#define ECF_MAY_BE_ALLOCA	  (1 << 5)
#define ECF_NOTHROW		  (1 << 6)
#define ECF_RETURNS_TWICE	  (1 << 7)
#define ECF_SIBCALL		  (1 << 8)
#define ECF_NOVOPS		  (1 << 9)
#define ECF_LEAF		  (1 << 10)
#define ECF_RET1		  (1 << 11)
#define ECF_TM_PURE		  (1 << 12)
#define ECF_TM_BUILTIN		  (1 << 13)
#define ECF_BY_DESCRIPTOR	  (1 << 14)
#define ECF_COLD		  (1 << 15)
#define EAF_DIRECT		(1 << 0)
#define EAF_NOCLOBBER		(1 << 1)
#define EAF_NOESCAPE		(1 << 2)
#define EAF_UNUSED		(1 << 3)
#define ERF_RETURN_ARG_MASK	(3)
#define ERF_RETURNS_ARG		(1 << 2)
#define ERF_NOALIAS		(1 << 3)
#define DEFTREECODE(SYM, STRING, TYPE, NARGS)   SYM,
#define END_OF_BASE_TREE_CODES LAST_AND_UNUSED_TREE_CODE,
enum tree_code {
#include "all-tree.def"
MAX_TREE_CODES
};
#undef DEFTREECODE
#undef END_OF_BASE_TREE_CODES
#define NUM_TREE_CODES \
((int) LAST_AND_UNUSED_TREE_CODE)
#define CODE_CONTAINS_STRUCT(CODE, STRUCT) \
(tree_contains_struct[(CODE)][(STRUCT)])
enum built_in_class {
NOT_BUILT_IN = 0,
BUILT_IN_FRONTEND,
BUILT_IN_MD,
BUILT_IN_NORMAL
};
#define BUILT_IN_LAST (BUILT_IN_NORMAL + 1)
#define DEF_BUILTIN(ENUM, N, C, T, LT, B, F, NA, AT, IM, COND) ENUM,
enum built_in_function {
#include "builtins.def"
BEGIN_CHKP_BUILTINS,
#define DEF_BUILTIN(ENUM, N, C, T, LT, B, F, NA, AT, IM, COND)
#define DEF_BUILTIN_CHKP(ENUM, N, C, T, LT, B, F, NA, AT, IM, COND) \
ENUM##_CHKP = ENUM + BEGIN_CHKP_BUILTINS + 1,
#include "builtins.def"
END_CHKP_BUILTINS = BEGIN_CHKP_BUILTINS * 2 + 1,
BUILT_IN_COMPLEX_MUL_MIN,
BUILT_IN_COMPLEX_MUL_MAX
= BUILT_IN_COMPLEX_MUL_MIN
+ MAX_MODE_COMPLEX_FLOAT
- MIN_MODE_COMPLEX_FLOAT,
BUILT_IN_COMPLEX_DIV_MIN,
BUILT_IN_COMPLEX_DIV_MAX
= BUILT_IN_COMPLEX_DIV_MIN
+ MAX_MODE_COMPLEX_FLOAT
- MIN_MODE_COMPLEX_FLOAT,
END_BUILTINS
};
enum internal_fn {
#define DEF_INTERNAL_FN(CODE, FLAGS, FNSPEC) IFN_##CODE,
#include "internal-fn.def"
IFN_LAST
};
enum combined_fn {
#define DEF_BUILTIN(ENUM, N, C, T, LT, B, F, NA, AT, IM, COND) \
CFN_##ENUM = int (ENUM),
#include "builtins.def"
#define DEF_BUILTIN(ENUM, N, C, T, LT, B, F, NA, AT, IM, COND)
#define DEF_BUILTIN_CHKP(ENUM, N, C, T, LT, B, F, NA, AT, IM, COND) \
CFN_##ENUM##_CHKP = int (ENUM##_CHKP),
#include "builtins.def"
#define DEF_INTERNAL_FN(CODE, FLAGS, FNSPEC) \
CFN_##CODE = int (END_BUILTINS) + int (IFN_##CODE),
#include "internal-fn.def"
CFN_LAST
};
enum tree_code_class {
tcc_exceptional, 
tcc_constant,    
tcc_type,        
tcc_declaration, 
tcc_reference,   
tcc_comparison,  
tcc_unary,       
tcc_binary,      
tcc_statement,   
tcc_vl_exp,      
tcc_expression   
};
enum omp_clause_code {
OMP_CLAUSE_ERROR = 0,
OMP_CLAUSE_PRIVATE,
OMP_CLAUSE_SHARED,
OMP_CLAUSE_FIRSTPRIVATE,
OMP_CLAUSE_LASTPRIVATE,
OMP_CLAUSE_REDUCTION,
OMP_CLAUSE_COPYIN,
OMP_CLAUSE_COPYPRIVATE,
OMP_CLAUSE_LINEAR,
OMP_CLAUSE_ALIGNED,
OMP_CLAUSE_DEPEND,
OMP_CLAUSE_UNIFORM,
OMP_CLAUSE_TO_DECLARE,
OMP_CLAUSE_LINK,
OMP_CLAUSE_FROM,
OMP_CLAUSE_TO,
OMP_CLAUSE_MAP,
OMP_CLAUSE_USE_DEVICE_PTR,
OMP_CLAUSE_IS_DEVICE_PTR,
OMP_CLAUSE__CACHE_,
OMP_CLAUSE_GANG,
OMP_CLAUSE_ASYNC,
OMP_CLAUSE_WAIT,
OMP_CLAUSE_AUTO,
OMP_CLAUSE_SEQ,
OMP_CLAUSE__LOOPTEMP_,
OMP_CLAUSE_IF,
OMP_CLAUSE_NUM_THREADS,
OMP_CLAUSE_SCHEDULE,
OMP_CLAUSE_NOWAIT,
OMP_CLAUSE_ORDERED,
OMP_CLAUSE_DEFAULT,
OMP_CLAUSE_COLLAPSE,
OMP_CLAUSE_UNTIED,
OMP_CLAUSE_FINAL,
OMP_CLAUSE_MERGEABLE,
OMP_CLAUSE_DEVICE,
OMP_CLAUSE_DIST_SCHEDULE,
OMP_CLAUSE_INBRANCH,
OMP_CLAUSE_NOTINBRANCH,
OMP_CLAUSE_NUM_TEAMS,
OMP_CLAUSE_THREAD_LIMIT,
OMP_CLAUSE_PROC_BIND,
OMP_CLAUSE_SAFELEN,
OMP_CLAUSE_SIMDLEN,
OMP_CLAUSE_FOR,
OMP_CLAUSE_PARALLEL,
OMP_CLAUSE_SECTIONS,
OMP_CLAUSE_TASKGROUP,
OMP_CLAUSE_PRIORITY,
OMP_CLAUSE_GRAINSIZE,
OMP_CLAUSE_NUM_TASKS,
OMP_CLAUSE_NOGROUP,
OMP_CLAUSE_THREADS,
OMP_CLAUSE_SIMD,
OMP_CLAUSE_HINT,
OMP_CLAUSE_DEFAULTMAP,
OMP_CLAUSE__SIMDUID_,
OMP_CLAUSE__SIMT_,
OMP_CLAUSE_INDEPENDENT,
OMP_CLAUSE_WORKER,
OMP_CLAUSE_VECTOR,
OMP_CLAUSE_NUM_GANGS,
OMP_CLAUSE_NUM_WORKERS,
OMP_CLAUSE_VECTOR_LENGTH,
OMP_CLAUSE_TILE,
OMP_CLAUSE__GRIDDIM_
};
#undef DEFTREESTRUCT
#define DEFTREESTRUCT(ENUM, NAME) ENUM,
enum tree_node_structure_enum {
#include "treestruct.def"
LAST_TS_ENUM
};
#undef DEFTREESTRUCT
enum omp_clause_schedule_kind {
OMP_CLAUSE_SCHEDULE_STATIC,
OMP_CLAUSE_SCHEDULE_DYNAMIC,
OMP_CLAUSE_SCHEDULE_GUIDED,
OMP_CLAUSE_SCHEDULE_AUTO,
OMP_CLAUSE_SCHEDULE_RUNTIME,
OMP_CLAUSE_SCHEDULE_MASK = (1 << 3) - 1,
OMP_CLAUSE_SCHEDULE_MONOTONIC = (1 << 3),
OMP_CLAUSE_SCHEDULE_NONMONOTONIC = (1 << 4),
OMP_CLAUSE_SCHEDULE_LAST = 2 * OMP_CLAUSE_SCHEDULE_NONMONOTONIC - 1
};
enum omp_clause_default_kind {
OMP_CLAUSE_DEFAULT_UNSPECIFIED,
OMP_CLAUSE_DEFAULT_SHARED,
OMP_CLAUSE_DEFAULT_NONE,
OMP_CLAUSE_DEFAULT_PRIVATE,
OMP_CLAUSE_DEFAULT_FIRSTPRIVATE,
OMP_CLAUSE_DEFAULT_PRESENT,
OMP_CLAUSE_DEFAULT_LAST
};
enum cv_qualifier {
TYPE_UNQUALIFIED   = 0x0,
TYPE_QUAL_CONST    = 0x1,
TYPE_QUAL_VOLATILE = 0x2,
TYPE_QUAL_RESTRICT = 0x4,
TYPE_QUAL_ATOMIC   = 0x8
};
enum tree_index {
TI_ERROR_MARK,
TI_INTQI_TYPE,
TI_INTHI_TYPE,
TI_INTSI_TYPE,
TI_INTDI_TYPE,
TI_INTTI_TYPE,
TI_UINTQI_TYPE,
TI_UINTHI_TYPE,
TI_UINTSI_TYPE,
TI_UINTDI_TYPE,
TI_UINTTI_TYPE,
TI_ATOMICQI_TYPE,
TI_ATOMICHI_TYPE,
TI_ATOMICSI_TYPE,
TI_ATOMICDI_TYPE,
TI_ATOMICTI_TYPE,
TI_UINT16_TYPE,
TI_UINT32_TYPE,
TI_UINT64_TYPE,
TI_VOID,
TI_INTEGER_ZERO,
TI_INTEGER_ONE,
TI_INTEGER_THREE,
TI_INTEGER_MINUS_ONE,
TI_NULL_POINTER,
TI_SIZE_ZERO,
TI_SIZE_ONE,
TI_BITSIZE_ZERO,
TI_BITSIZE_ONE,
TI_BITSIZE_UNIT,
TI_PUBLIC,
TI_PROTECTED,
TI_PRIVATE,
TI_BOOLEAN_FALSE,
TI_BOOLEAN_TRUE,
TI_FLOAT_TYPE,
TI_DOUBLE_TYPE,
TI_LONG_DOUBLE_TYPE,
TI_FLOAT16_TYPE,
TI_FLOATN_TYPE_FIRST = TI_FLOAT16_TYPE,
TI_FLOATN_NX_TYPE_FIRST = TI_FLOAT16_TYPE,
TI_FLOAT32_TYPE,
TI_FLOAT64_TYPE,
TI_FLOAT128_TYPE,
TI_FLOATN_TYPE_LAST = TI_FLOAT128_TYPE,
#define NUM_FLOATN_TYPES (TI_FLOATN_TYPE_LAST - TI_FLOATN_TYPE_FIRST + 1)
TI_FLOAT32X_TYPE,
TI_FLOATNX_TYPE_FIRST = TI_FLOAT32X_TYPE,
TI_FLOAT64X_TYPE,
TI_FLOAT128X_TYPE,
TI_FLOATNX_TYPE_LAST = TI_FLOAT128X_TYPE,
TI_FLOATN_NX_TYPE_LAST = TI_FLOAT128X_TYPE,
#define NUM_FLOATNX_TYPES (TI_FLOATNX_TYPE_LAST - TI_FLOATNX_TYPE_FIRST + 1)
#define NUM_FLOATN_NX_TYPES (TI_FLOATN_NX_TYPE_LAST		\
- TI_FLOATN_NX_TYPE_FIRST		\
+ 1)
TI_COMPLEX_INTEGER_TYPE,
TI_COMPLEX_FLOAT_TYPE,
TI_COMPLEX_DOUBLE_TYPE,
TI_COMPLEX_LONG_DOUBLE_TYPE,
TI_COMPLEX_FLOAT16_TYPE,
TI_COMPLEX_FLOATN_NX_TYPE_FIRST = TI_COMPLEX_FLOAT16_TYPE,
TI_COMPLEX_FLOAT32_TYPE,
TI_COMPLEX_FLOAT64_TYPE,
TI_COMPLEX_FLOAT128_TYPE,
TI_COMPLEX_FLOAT32X_TYPE,
TI_COMPLEX_FLOAT64X_TYPE,
TI_COMPLEX_FLOAT128X_TYPE,
TI_FLOAT_PTR_TYPE,
TI_DOUBLE_PTR_TYPE,
TI_LONG_DOUBLE_PTR_TYPE,
TI_INTEGER_PTR_TYPE,
TI_VOID_TYPE,
TI_PTR_TYPE,
TI_CONST_PTR_TYPE,
TI_SIZE_TYPE,
TI_PID_TYPE,
TI_PTRDIFF_TYPE,
TI_VA_LIST_TYPE,
TI_VA_LIST_GPR_COUNTER_FIELD,
TI_VA_LIST_FPR_COUNTER_FIELD,
TI_BOOLEAN_TYPE,
TI_FILEPTR_TYPE,
TI_CONST_TM_PTR_TYPE,
TI_FENV_T_PTR_TYPE,
TI_CONST_FENV_T_PTR_TYPE,
TI_FEXCEPT_T_PTR_TYPE,
TI_CONST_FEXCEPT_T_PTR_TYPE,
TI_POINTER_SIZED_TYPE,
TI_POINTER_BOUNDS_TYPE,
TI_DFLOAT32_TYPE,
TI_DFLOAT64_TYPE,
TI_DFLOAT128_TYPE,
TI_DFLOAT32_PTR_TYPE,
TI_DFLOAT64_PTR_TYPE,
TI_DFLOAT128_PTR_TYPE,
TI_VOID_LIST_NODE,
TI_MAIN_IDENTIFIER,
TI_SAT_SFRACT_TYPE,
TI_SAT_FRACT_TYPE,
TI_SAT_LFRACT_TYPE,
TI_SAT_LLFRACT_TYPE,
TI_SAT_USFRACT_TYPE,
TI_SAT_UFRACT_TYPE,
TI_SAT_ULFRACT_TYPE,
TI_SAT_ULLFRACT_TYPE,
TI_SFRACT_TYPE,
TI_FRACT_TYPE,
TI_LFRACT_TYPE,
TI_LLFRACT_TYPE,
TI_USFRACT_TYPE,
TI_UFRACT_TYPE,
TI_ULFRACT_TYPE,
TI_ULLFRACT_TYPE,
TI_SAT_SACCUM_TYPE,
TI_SAT_ACCUM_TYPE,
TI_SAT_LACCUM_TYPE,
TI_SAT_LLACCUM_TYPE,
TI_SAT_USACCUM_TYPE,
TI_SAT_UACCUM_TYPE,
TI_SAT_ULACCUM_TYPE,
TI_SAT_ULLACCUM_TYPE,
TI_SACCUM_TYPE,
TI_ACCUM_TYPE,
TI_LACCUM_TYPE,
TI_LLACCUM_TYPE,
TI_USACCUM_TYPE,
TI_UACCUM_TYPE,
TI_ULACCUM_TYPE,
TI_ULLACCUM_TYPE,
TI_QQ_TYPE,
TI_HQ_TYPE,
TI_SQ_TYPE,
TI_DQ_TYPE,
TI_TQ_TYPE,
TI_UQQ_TYPE,
TI_UHQ_TYPE,
TI_USQ_TYPE,
TI_UDQ_TYPE,
TI_UTQ_TYPE,
TI_SAT_QQ_TYPE,
TI_SAT_HQ_TYPE,
TI_SAT_SQ_TYPE,
TI_SAT_DQ_TYPE,
TI_SAT_TQ_TYPE,
TI_SAT_UQQ_TYPE,
TI_SAT_UHQ_TYPE,
TI_SAT_USQ_TYPE,
TI_SAT_UDQ_TYPE,
TI_SAT_UTQ_TYPE,
TI_HA_TYPE,
TI_SA_TYPE,
TI_DA_TYPE,
TI_TA_TYPE,
TI_UHA_TYPE,
TI_USA_TYPE,
TI_UDA_TYPE,
TI_UTA_TYPE,
TI_SAT_HA_TYPE,
TI_SAT_SA_TYPE,
TI_SAT_DA_TYPE,
TI_SAT_TA_TYPE,
TI_SAT_UHA_TYPE,
TI_SAT_USA_TYPE,
TI_SAT_UDA_TYPE,
TI_SAT_UTA_TYPE,
TI_OPTIMIZATION_DEFAULT,
TI_OPTIMIZATION_CURRENT,
TI_TARGET_OPTION_DEFAULT,
TI_TARGET_OPTION_CURRENT,
TI_CURRENT_TARGET_PRAGMA,
TI_CURRENT_OPTIMIZE_PRAGMA,
TI_MAX
};
enum integer_type_kind {
itk_char,
itk_signed_char,
itk_unsigned_char,
itk_short,
itk_unsigned_short,
itk_int,
itk_unsigned_int,
itk_long,
itk_unsigned_long,
itk_long_long,
itk_unsigned_long_long,
itk_intN_0,
itk_unsigned_intN_0,
itk_intN_1,
itk_unsigned_intN_1,
itk_intN_2,
itk_unsigned_intN_2,
itk_intN_3,
itk_unsigned_intN_3,
itk_none
};
enum ptrmemfunc_vbit_where_t {
ptrmemfunc_vbit_in_pfn,
ptrmemfunc_vbit_in_delta
};
enum attribute_flags {
ATTR_FLAG_DECL_NEXT = 1,
ATTR_FLAG_FUNCTION_NEXT = 2,
ATTR_FLAG_ARRAY_NEXT = 4,
ATTR_FLAG_TYPE_IN_PLACE = 8,
ATTR_FLAG_BUILT_IN = 16,
ATTR_FLAG_CXX11 = 32
};
enum size_type_kind {
stk_sizetype,		
stk_ssizetype,	
stk_bitsizetype,	
stk_sbitsizetype,	
stk_type_kind_last
};
enum operand_equal_flag {
OEP_ONLY_CONST = 1,
OEP_PURE_SAME = 2,
OEP_MATCH_SIDE_EFFECTS = 4,
OEP_ADDRESS_OF = 8,
OEP_NO_HASH_CHECK = 16,
OEP_HASH_CHECK = 32,
OEP_LEXICOGRAPHIC = 64
};
enum tree_node_kind {
d_kind,
t_kind,
b_kind,
s_kind,
r_kind,
e_kind,
c_kind,
id_kind,
vec_kind,
binfo_kind,
ssa_name_kind,
constr_kind,
x_kind,
lang_decl,
lang_type,
omp_clause_kind,
all_kinds
};
enum annot_expr_kind {
annot_expr_ivdep_kind,
annot_expr_unroll_kind,
annot_expr_no_vector_kind,
annot_expr_vector_kind,
annot_expr_parallel_kind,
annot_expr_kind_last
};
struct GTY(()) alias_pair {
tree decl;
tree target;
};
typedef unsigned short priority_type;
typedef tree (*walk_tree_fn) (tree *, int *, void *);
typedef tree (*walk_tree_lh) (tree *, int *, tree (*) (tree *, int *, void *),
void *, hash_set<tree> *);
struct GTY(()) tree_base {
ENUM_BITFIELD(tree_code) code : 16;
unsigned side_effects_flag : 1;
unsigned constant_flag : 1;
unsigned addressable_flag : 1;
unsigned volatile_flag : 1;
unsigned readonly_flag : 1;
unsigned asm_written_flag: 1;
unsigned nowarning_flag : 1;
unsigned visited : 1;
unsigned used_flag : 1;
unsigned nothrow_flag : 1;
unsigned static_flag : 1;
unsigned public_flag : 1;
unsigned private_flag : 1;
unsigned protected_flag : 1;
unsigned deprecated_flag : 1;
unsigned default_def_flag : 1;
union {
struct {
unsigned lang_flag_0 : 1;
unsigned lang_flag_1 : 1;
unsigned lang_flag_2 : 1;
unsigned lang_flag_3 : 1;
unsigned lang_flag_4 : 1;
unsigned lang_flag_5 : 1;
unsigned lang_flag_6 : 1;
unsigned saturating_flag : 1;
unsigned unsigned_flag : 1;
unsigned packed_flag : 1;
unsigned user_align : 1;
unsigned nameless_flag : 1;
unsigned atomic_flag : 1;
unsigned spare0 : 3;
unsigned spare1 : 8;
unsigned address_space : 8;
} bits;
struct {
unsigned char unextended;
unsigned char extended;
unsigned char offset;
} int_length;
int length;
struct {
unsigned int log2_npatterns : 8;
unsigned int nelts_per_pattern : 8;
unsigned int unused : 16;
} vector_cst;
unsigned int version;
unsigned int chrec_var;
enum internal_fn ifn;
struct {
unsigned short clique;
unsigned short base;
} dependence_info;
} GTY((skip(""))) u;
};
struct GTY(()) tree_typed {
struct tree_base base;
tree type;
};
struct GTY(()) tree_common {
struct tree_typed typed;
tree chain;
};
struct GTY(()) tree_int_cst {
struct tree_typed typed;
HOST_WIDE_INT val[1];
};
struct GTY(()) tree_real_cst {
struct tree_typed typed;
struct real_value * real_cst_ptr;
};
struct GTY(()) tree_fixed_cst {
struct tree_typed typed;
struct fixed_value * fixed_cst_ptr;
};
struct GTY(()) tree_string {
struct tree_typed typed;
int length;
char str[1];
};
struct GTY(()) tree_complex {
struct tree_typed typed;
tree real;
tree imag;
};
struct GTY(()) tree_vector {
struct tree_typed typed;
tree GTY ((length ("vector_cst_encoded_nelts ((tree) &%h)"))) elts[1];
};
struct GTY(()) tree_poly_int_cst {
struct tree_typed typed;
tree coeffs[NUM_POLY_INT_COEFFS];
};
struct GTY(()) tree_identifier {
struct tree_common common;
struct ht_identifier id;
};
struct GTY(()) tree_list {
struct tree_common common;
tree purpose;
tree value;
};
struct GTY(()) tree_vec {
struct tree_common common;
tree GTY ((length ("TREE_VEC_LENGTH ((tree)&%h)"))) a[1];
};
struct GTY(()) constructor_elt {
tree index;
tree value;
};
struct GTY(()) tree_constructor {
struct tree_typed typed;
vec<constructor_elt, va_gc> *elts;
};
enum omp_clause_depend_kind
{
OMP_CLAUSE_DEPEND_IN,
OMP_CLAUSE_DEPEND_OUT,
OMP_CLAUSE_DEPEND_INOUT,
OMP_CLAUSE_DEPEND_SOURCE,
OMP_CLAUSE_DEPEND_SINK,
OMP_CLAUSE_DEPEND_LAST
};
enum omp_clause_proc_bind_kind
{
OMP_CLAUSE_PROC_BIND_FALSE = 0,
OMP_CLAUSE_PROC_BIND_TRUE = 1,
OMP_CLAUSE_PROC_BIND_MASTER = 2,
OMP_CLAUSE_PROC_BIND_CLOSE = 3,
OMP_CLAUSE_PROC_BIND_SPREAD = 4,
OMP_CLAUSE_PROC_BIND_LAST
};
enum omp_clause_linear_kind
{
OMP_CLAUSE_LINEAR_DEFAULT,
OMP_CLAUSE_LINEAR_REF,
OMP_CLAUSE_LINEAR_VAL,
OMP_CLAUSE_LINEAR_UVAL
};
struct GTY(()) tree_exp {
struct tree_typed typed;
location_t locus;
tree GTY ((special ("tree_exp"),
desc ("TREE_CODE ((tree) &%0)")))
operands[1];
};
struct GTY(()) ssa_use_operand_t {
struct ssa_use_operand_t* GTY((skip(""))) prev;
struct ssa_use_operand_t* GTY((skip(""))) next;
union { gimple *stmt; tree ssa_name; } GTY((skip(""))) loc;
tree *GTY((skip(""))) use;
};
struct GTY(()) tree_ssa_name {
struct tree_typed typed;
tree var;
gimple *def_stmt;
union ssa_name_info_type {
struct GTY ((tag ("0"))) ptr_info_def *ptr_info;
struct GTY ((tag ("1"))) range_info_def *range_info;
} GTY ((desc ("%1.typed.type ?" \
"!POINTER_TYPE_P (TREE_TYPE ((tree)&%1)) : 2"))) info;
struct ssa_use_operand_t imm_uses;
};
struct GTY(()) phi_arg_d {
struct ssa_use_operand_t imm_use;
tree def;
location_t locus;
};
struct GTY(()) tree_omp_clause {
struct tree_common common;
location_t locus;
enum omp_clause_code code;
union omp_clause_subcode {
enum omp_clause_default_kind   default_kind;
enum omp_clause_schedule_kind  schedule_kind;
enum omp_clause_depend_kind    depend_kind;
unsigned int		   map_kind;
enum omp_clause_proc_bind_kind proc_bind_kind;
enum tree_code                 reduction_code;
enum omp_clause_linear_kind    linear_kind;
enum tree_code                 if_modifier;
unsigned int		   dimension;
} GTY ((skip)) subcode;
gimple_seq gimple_reduction_init;
gimple_seq gimple_reduction_merge;
tree GTY ((length ("omp_clause_num_ops[OMP_CLAUSE_CODE ((tree)&%h)]")))
ops[1];
};
struct GTY(()) tree_block {
struct tree_base base;
tree chain;
unsigned abstract_flag : 1;
unsigned block_num : 31;
location_t locus;
location_t end_locus;
tree vars;
vec<tree, va_gc> *nonlocalized_vars;
tree subblocks;
tree supercontext;
tree abstract_origin;
tree fragment_origin;
tree fragment_chain;
struct die_struct *die;
};
struct GTY(()) tree_type_common {
struct tree_common common;
tree size;
tree size_unit;
tree attributes;
unsigned int uid;
unsigned int precision : 10;
unsigned no_force_blk_flag : 1;
unsigned needs_constructing_flag : 1;
unsigned transparent_aggr_flag : 1;
unsigned restrict_flag : 1;
unsigned contains_placeholder_bits : 2;
ENUM_BITFIELD(machine_mode) mode : 8;
unsigned string_flag : 1;
unsigned lang_flag_0 : 1;
unsigned lang_flag_1 : 1;
unsigned lang_flag_2 : 1;
unsigned lang_flag_3 : 1;
unsigned lang_flag_4 : 1;
unsigned lang_flag_5 : 1;
unsigned lang_flag_6 : 1;
unsigned lang_flag_7 : 1;
unsigned align : 6;
unsigned warn_if_not_align : 6;
unsigned typeless_storage : 1;
unsigned empty_flag : 1;
unsigned spare : 17;
alias_set_type alias_set;
tree pointer_to;
tree reference_to;
union tree_type_symtab {
int GTY ((tag ("TYPE_SYMTAB_IS_ADDRESS"))) address;
struct die_struct * GTY ((tag ("TYPE_SYMTAB_IS_DIE"))) die;
} GTY ((desc ("debug_hooks->tree_type_symtab_field"))) symtab;
tree canonical;
tree next_variant;
tree main_variant;
tree context;
tree name;
};
struct GTY(()) tree_type_with_lang_specific {
struct tree_type_common common;
struct lang_type *lang_specific;
};
struct GTY(()) tree_type_non_common {
struct tree_type_with_lang_specific with_lang_specific;
tree values;
tree minval;
tree maxval;
tree lang_1;
};
struct GTY (()) tree_binfo {
struct tree_common common;
tree offset;
tree vtable;
tree virtuals;
tree vptr_field;
vec<tree, va_gc> *base_accesses;
tree inheritance;
tree vtt_subvtt;
tree vtt_vptr;
vec<tree, va_gc> base_binfos;
};
struct GTY(()) tree_decl_minimal {
struct tree_common common;
location_t locus;
unsigned int uid;
tree name;
tree context;
};
struct GTY(()) tree_decl_common {
struct tree_decl_minimal common;
tree size;
ENUM_BITFIELD(machine_mode) mode : 8;
unsigned nonlocal_flag : 1;
unsigned virtual_flag : 1;
unsigned ignored_flag : 1;
unsigned abstract_flag : 1;
unsigned artificial_flag : 1;
unsigned preserve_flag: 1;
unsigned debug_expr_is_from : 1;
unsigned lang_flag_0 : 1;
unsigned lang_flag_1 : 1;
unsigned lang_flag_2 : 1;
unsigned lang_flag_3 : 1;
unsigned lang_flag_4 : 1;
unsigned lang_flag_5 : 1;
unsigned lang_flag_6 : 1;
unsigned lang_flag_7 : 1;
unsigned lang_flag_8 : 1;
unsigned decl_flag_0 : 1;
unsigned decl_flag_1 : 1;
unsigned decl_flag_2 : 1;
unsigned decl_flag_3 : 1;
unsigned gimple_reg_flag : 1;
unsigned decl_by_reference_flag : 1;
unsigned decl_read_flag : 1;
unsigned decl_nonshareable_flag : 1;
unsigned int off_align : 6;
unsigned int align : 6;
unsigned int warn_if_not_align : 6;
unsigned int pt_uid;
tree size_unit;
tree initial;
tree attributes;
tree abstract_origin;
struct lang_decl *lang_specific;
};
struct GTY(()) tree_decl_with_rtl {
struct tree_decl_common common;
rtx rtl;
};
struct GTY(()) tree_field_decl {
struct tree_decl_common common;
tree offset;
tree bit_field_type;
tree qualifier;
tree bit_offset;
tree fcontext;
};
struct GTY(()) tree_label_decl {
struct tree_decl_with_rtl common;
int label_decl_uid;
int eh_landing_pad_nr;
};
struct GTY(()) tree_result_decl {
struct tree_decl_with_rtl common;
};
struct GTY(()) tree_const_decl {
struct tree_decl_common common;
};
struct GTY(()) tree_parm_decl {
struct tree_decl_with_rtl common;
rtx incoming_rtl;
};
struct GTY(()) tree_decl_with_vis {
struct tree_decl_with_rtl common;
tree assembler_name;
struct symtab_node *symtab_node;
unsigned defer_output : 1;
unsigned hard_register : 1;
unsigned common_flag : 1;
unsigned in_text_section : 1;
unsigned in_constant_pool : 1;
unsigned dllimport_flag : 1;
unsigned weak_flag : 1;
unsigned seen_in_bind_expr : 1;
unsigned comdat_flag : 1;
ENUM_BITFIELD(symbol_visibility) visibility : 2;
unsigned visibility_specified : 1;
unsigned init_priority_p : 1;
unsigned shadowed_for_var_p : 1;
unsigned cxx_constructor : 1;
unsigned cxx_destructor : 1;
unsigned final : 1;
unsigned regdecl_flag : 1;
};
struct GTY(()) tree_var_decl {
struct tree_decl_with_vis common;
};
struct GTY(()) tree_decl_non_common {
struct tree_decl_with_vis common;
tree result;
};
struct GTY(()) tree_function_decl {
struct tree_decl_non_common common;
struct function *f;
tree arguments;
tree personality;
tree function_specific_target;	
tree function_specific_optimization;	
tree saved_tree;
tree vindex;
ENUM_BITFIELD(built_in_function) function_code : 12;
ENUM_BITFIELD(built_in_class) built_in_class : 2;
unsigned static_ctor_flag : 1;
unsigned static_dtor_flag : 1;
unsigned uninlinable : 1;
unsigned possibly_inlined : 1;
unsigned novops_flag : 1;
unsigned returns_twice_flag : 1;
unsigned malloc_flag : 1;
unsigned operator_new_flag : 1;
unsigned declared_inline_flag : 1;
unsigned no_inline_warning_flag : 1;
unsigned no_instrument_function_entry_exit : 1;
unsigned no_limit_stack : 1;
unsigned disregard_inline_limits : 1;
unsigned pure_flag : 1;
unsigned looping_const_or_pure_flag : 1;
unsigned has_debug_args_flag : 1;
unsigned versioned_function : 1;
unsigned lambda_function: 1;
};
struct GTY(()) tree_translation_unit_decl {
struct tree_decl_common common;
const char * GTY((skip(""))) language;
};
struct GTY(()) tree_type_decl {
struct tree_decl_non_common common;
};
struct GTY ((chain_next ("%h.next"), chain_prev ("%h.prev"))) tree_statement_list_node
{
struct tree_statement_list_node *prev;
struct tree_statement_list_node *next;
tree stmt;
};
struct GTY(()) tree_statement_list
{
struct tree_typed typed;
struct tree_statement_list_node *head;
struct tree_statement_list_node *tail;
};
struct GTY(()) tree_optimization_option {
struct tree_base base;
struct cl_optimization *opts;
void *GTY ((atomic)) optabs;
struct target_optabs *GTY ((skip)) base_optabs;
};
struct GTY(()) target_globals;
struct GTY(()) tree_target_option {
struct tree_base base;
struct target_globals *globals;
struct cl_target_option *opts;
};
union GTY ((ptr_alias (union lang_tree_node),
desc ("tree_node_structure (&%h)"), variable_size)) tree_node {
struct tree_base GTY ((tag ("TS_BASE"))) base;
struct tree_typed GTY ((tag ("TS_TYPED"))) typed;
struct tree_common GTY ((tag ("TS_COMMON"))) common;
struct tree_int_cst GTY ((tag ("TS_INT_CST"))) int_cst;
struct tree_poly_int_cst GTY ((tag ("TS_POLY_INT_CST"))) poly_int_cst;
struct tree_real_cst GTY ((tag ("TS_REAL_CST"))) real_cst;
struct tree_fixed_cst GTY ((tag ("TS_FIXED_CST"))) fixed_cst;
struct tree_vector GTY ((tag ("TS_VECTOR"))) vector;
struct tree_string GTY ((tag ("TS_STRING"))) string;
struct tree_complex GTY ((tag ("TS_COMPLEX"))) complex;
struct tree_identifier GTY ((tag ("TS_IDENTIFIER"))) identifier;
struct tree_decl_minimal GTY((tag ("TS_DECL_MINIMAL"))) decl_minimal;
struct tree_decl_common GTY ((tag ("TS_DECL_COMMON"))) decl_common;
struct tree_decl_with_rtl GTY ((tag ("TS_DECL_WRTL"))) decl_with_rtl;
struct tree_decl_non_common  GTY ((tag ("TS_DECL_NON_COMMON")))
decl_non_common;
struct tree_parm_decl  GTY  ((tag ("TS_PARM_DECL"))) parm_decl;
struct tree_decl_with_vis GTY ((tag ("TS_DECL_WITH_VIS"))) decl_with_vis;
struct tree_var_decl GTY ((tag ("TS_VAR_DECL"))) var_decl;
struct tree_field_decl GTY ((tag ("TS_FIELD_DECL"))) field_decl;
struct tree_label_decl GTY ((tag ("TS_LABEL_DECL"))) label_decl;
struct tree_result_decl GTY ((tag ("TS_RESULT_DECL"))) result_decl;
struct tree_const_decl GTY ((tag ("TS_CONST_DECL"))) const_decl;
struct tree_type_decl GTY ((tag ("TS_TYPE_DECL"))) type_decl;
struct tree_function_decl GTY ((tag ("TS_FUNCTION_DECL"))) function_decl;
struct tree_translation_unit_decl GTY ((tag ("TS_TRANSLATION_UNIT_DECL")))
translation_unit_decl;
struct tree_type_common GTY ((tag ("TS_TYPE_COMMON"))) type_common;
struct tree_type_with_lang_specific GTY ((tag ("TS_TYPE_WITH_LANG_SPECIFIC")))
type_with_lang_specific;
struct tree_type_non_common GTY ((tag ("TS_TYPE_NON_COMMON")))
type_non_common;
struct tree_list GTY ((tag ("TS_LIST"))) list;
struct tree_vec GTY ((tag ("TS_VEC"))) vec;
struct tree_exp GTY ((tag ("TS_EXP"))) exp;
struct tree_ssa_name GTY ((tag ("TS_SSA_NAME"))) ssa_name;
struct tree_block GTY ((tag ("TS_BLOCK"))) block;
struct tree_binfo GTY ((tag ("TS_BINFO"))) binfo;
struct tree_statement_list GTY ((tag ("TS_STATEMENT_LIST"))) stmt_list;
struct tree_constructor GTY ((tag ("TS_CONSTRUCTOR"))) constructor;
struct tree_omp_clause GTY ((tag ("TS_OMP_CLAUSE"))) omp_clause;
struct tree_optimization_option GTY ((tag ("TS_OPTIMIZATION"))) optimization;
struct tree_target_option GTY ((tag ("TS_TARGET_OPTION"))) target_option;
};
struct attribute_spec {
const char *name;
int min_length;
int max_length;
bool decl_required;
bool type_required;
bool function_type_required;
bool affects_type_identity;
tree (*handler) (tree *node, tree name, tree args,
int flags, bool *no_add_attrs);
struct exclusions {
const char *name;
bool function;
bool variable;
bool type;
};
const exclusions *exclude;
};
typedef struct record_layout_info_s {
tree t;
tree offset;
unsigned int offset_align;
tree bitpos;
unsigned int record_align;
unsigned int unpacked_align;
tree prev_field;
vec<tree, va_gc> *pending_statics;
int remaining_in_alignment;
int packed_maybe_necessary;
} *record_layout_info;
struct function_args_iterator {
tree next;			
};
struct GTY(()) tree_map_base {
tree from;
};
struct GTY((for_user)) tree_map {
struct tree_map_base base;
unsigned int hash;
tree to;
};
struct GTY((for_user)) tree_decl_map {
struct tree_map_base base;
tree to;
};
struct GTY((for_user)) tree_int_map {
struct tree_map_base base;
unsigned int to;
};
struct GTY((for_user)) tree_vec_map {
struct tree_map_base base;
vec<tree, va_gc> *to;
};
struct call_expr_arg_iterator {
tree t;	
int n;	
int i;	
};
struct const_call_expr_arg_iterator {
const_tree t;	
int n;	
int i;	
};
struct GTY(()) builtin_info_type {
tree decl;
unsigned implicit_p : 1;
unsigned declared_p : 1;
};
struct floatn_type_info {
int n;
bool extended;
};
extern bool tree_contains_struct[MAX_TREE_CODES][64];
extern const enum tree_code_class tree_code_type[];
extern const char *const tree_code_class_strings[];
extern const unsigned char tree_code_length[];
extern GTY(()) vec<alias_pair, va_gc> *alias_pairs;
extern const char *const built_in_class_names[BUILT_IN_LAST];
extern const char * built_in_names[(int) END_BUILTINS];
extern unsigned const char omp_clause_num_ops[];
extern const char * const omp_clause_code_name[];
extern GTY (()) vec<tree, va_gc> *all_translation_units;
extern GTY(()) tree global_trees[TI_MAX];
extern GTY(()) tree integer_types[itk_none];
extern GTY(()) tree sizetype_tab[(int) stk_type_kind_last];
extern uint64_t tree_node_counts[];
extern uint64_t tree_node_sizes[];
extern bool in_gimple_form;
extern GTY(()) builtin_info_type builtin_info[(int)END_BUILTINS];
extern unsigned int maximum_field_alignment;
extern GTY(()) tree current_function_decl;
extern GTY(()) const char * current_function_func_begin_label;
extern const floatn_type_info floatn_nx_types[NUM_FLOATN_NX_TYPES];
#endif  
