#ifndef GCC_TREE_VECTORIZER_H
#define GCC_TREE_VECTORIZER_H
#include "tree-data-ref.h"
#include "tree-hash-traits.h"
#include "target.h"
enum vect_var_kind {
vect_simple_var,
vect_pointer_var,
vect_scalar_var,
vect_mask_var
};
enum operation_type {
unary_op = 1,
binary_op,
ternary_op
};
enum dr_alignment_support {
dr_unaligned_unsupported,
dr_unaligned_supported,
dr_explicit_realign,
dr_explicit_realign_optimized,
dr_aligned
};
enum vect_def_type {
vect_uninitialized_def = 0,
vect_constant_def = 1,
vect_external_def,
vect_internal_def,
vect_induction_def,
vect_reduction_def,
vect_double_reduction_def,
vect_nested_cycle,
vect_unknown_def_type
};
enum vect_reduction_type {
TREE_CODE_REDUCTION,
COND_REDUCTION,
INTEGER_INDUC_COND_REDUCTION,
CONST_COND_REDUCTION,
EXTRACT_LAST_REDUCTION,
FOLD_LEFT_REDUCTION
};
#define VECTORIZABLE_CYCLE_DEF(D) (((D) == vect_reduction_def)           \
|| ((D) == vect_double_reduction_def) \
|| ((D) == vect_nested_cycle))
struct stmt_info_for_cost {
int count;
enum vect_cost_for_stmt kind;
gimple *stmt;
int misalign;
};
typedef vec<stmt_info_for_cost> stmt_vector_for_cost;
typedef hash_map<tree_operand_hash,
innermost_loop_behavior *> vec_base_alignments;
typedef struct _slp_tree *slp_tree;
struct _slp_tree {
vec<slp_tree> children;
vec<gimple *> stmts;
vec<unsigned> load_permutation;
vec<gimple *> vec_stmts;
unsigned int vec_stmts_size;
bool two_operators;
enum vect_def_type def_type;
};
typedef struct _slp_instance {
slp_tree root;
unsigned int group_size;
poly_uint64 unrolling_factor;
vec<slp_tree> loads;
slp_tree reduc_phis;
} *slp_instance;
#define SLP_INSTANCE_TREE(S)                     (S)->root
#define SLP_INSTANCE_GROUP_SIZE(S)               (S)->group_size
#define SLP_INSTANCE_UNROLLING_FACTOR(S)         (S)->unrolling_factor
#define SLP_INSTANCE_LOADS(S)                    (S)->loads
#define SLP_TREE_CHILDREN(S)                     (S)->children
#define SLP_TREE_SCALAR_STMTS(S)                 (S)->stmts
#define SLP_TREE_VEC_STMTS(S)                    (S)->vec_stmts
#define SLP_TREE_NUMBER_OF_VEC_STMTS(S)          (S)->vec_stmts_size
#define SLP_TREE_LOAD_PERMUTATION(S)             (S)->load_permutation
#define SLP_TREE_TWO_OPERATORS(S)		 (S)->two_operators
#define SLP_TREE_DEF_TYPE(S)			 (S)->def_type
typedef std::pair<tree, tree> vec_object_pair;
struct vec_lower_bound {
vec_lower_bound () {}
vec_lower_bound (tree e, bool u, poly_uint64 m)
: expr (e), unsigned_p (u), min_value (m) {}
tree expr;
bool unsigned_p;
poly_uint64 min_value;
};
struct vec_info {
enum vec_kind { bb, loop };
vec_info (vec_kind, void *);
~vec_info ();
vec_kind kind;
auto_vec<slp_instance> slp_instances;
vec<data_reference_p> datarefs;
vec_base_alignments base_alignments;
vec<ddr_p> ddrs;
auto_vec<gimple *> grouped_stores;
void *target_cost_data;
};
struct _loop_vec_info;
struct _bb_vec_info;
template<>
template<>
inline bool
is_a_helper <_loop_vec_info *>::test (vec_info *i)
{
return i->kind == vec_info::loop;
}
template<>
template<>
inline bool
is_a_helper <_bb_vec_info *>::test (vec_info *i)
{
return i->kind == vec_info::bb;
}
struct rgroup_masks {
unsigned int max_nscalars_per_iter;
tree mask_type;
vec<tree> masks;
};
typedef auto_vec<rgroup_masks> vec_loop_masks;
typedef struct _loop_vec_info : public vec_info {
_loop_vec_info (struct loop *);
~_loop_vec_info ();
struct loop *loop;
basic_block *bbs;
tree num_itersm1;
tree num_iters;
tree num_iters_unchanged;
tree num_iters_assumptions;
unsigned int th;
poly_uint64 versioning_threshold;
poly_uint64 vectorization_factor;
unsigned HOST_WIDE_INT max_vectorization_factor;
vec_loop_masks masks;
tree mask_skip_niters;
tree mask_compare_type;
struct data_reference *unaligned_dr;
int peeling_for_alignment;
int ptr_mask;
auto_vec<loop_p> loop_nest;
auto_vec<ddr_p> may_alias_ddrs;
auto_vec<dr_with_seg_len_pair_t> comp_alias_ddrs;
auto_vec<vec_object_pair> check_unequal_addrs;
auto_vec<tree> check_nonzero;
auto_vec<vec_lower_bound> lower_bounds;
auto_vec<gimple *> may_misalign_stmts;
auto_vec<gimple *> reductions;
auto_vec<gimple *> reduction_chains;
auto_vec<stmt_info_for_cost> scalar_cost_vec;
hash_map<tree_operand_hash, tree> *ivexpr_map;
poly_uint64 slp_unrolling_factor;
int single_scalar_iteration_cost;
bool vectorizable;
bool can_fully_mask_p;
bool fully_masked_p;
bool peeling_for_gaps;
bool peeling_for_niter;
bool operands_swapped;
bool no_data_dependencies;
bool has_mask_store;
struct loop *scalar_loop;
_loop_vec_info *orig_loop_info;
} *loop_vec_info;
#define LOOP_VINFO_LOOP(L)                 (L)->loop
#define LOOP_VINFO_BBS(L)                  (L)->bbs
#define LOOP_VINFO_NITERSM1(L)             (L)->num_itersm1
#define LOOP_VINFO_NITERS(L)               (L)->num_iters
#define LOOP_VINFO_NITERS_UNCHANGED(L)     (L)->num_iters_unchanged
#define LOOP_VINFO_NITERS_ASSUMPTIONS(L)   (L)->num_iters_assumptions
#define LOOP_VINFO_COST_MODEL_THRESHOLD(L) (L)->th
#define LOOP_VINFO_VERSIONING_THRESHOLD(L) (L)->versioning_threshold
#define LOOP_VINFO_VECTORIZABLE_P(L)       (L)->vectorizable
#define LOOP_VINFO_CAN_FULLY_MASK_P(L)     (L)->can_fully_mask_p
#define LOOP_VINFO_FULLY_MASKED_P(L)       (L)->fully_masked_p
#define LOOP_VINFO_VECT_FACTOR(L)          (L)->vectorization_factor
#define LOOP_VINFO_MAX_VECT_FACTOR(L)      (L)->max_vectorization_factor
#define LOOP_VINFO_MASKS(L)                (L)->masks
#define LOOP_VINFO_MASK_SKIP_NITERS(L)     (L)->mask_skip_niters
#define LOOP_VINFO_MASK_COMPARE_TYPE(L)    (L)->mask_compare_type
#define LOOP_VINFO_PTR_MASK(L)             (L)->ptr_mask
#define LOOP_VINFO_LOOP_NEST(L)            (L)->loop_nest
#define LOOP_VINFO_DATAREFS(L)             (L)->datarefs
#define LOOP_VINFO_DDRS(L)                 (L)->ddrs
#define LOOP_VINFO_INT_NITERS(L)           (TREE_INT_CST_LOW ((L)->num_iters))
#define LOOP_VINFO_PEELING_FOR_ALIGNMENT(L) (L)->peeling_for_alignment
#define LOOP_VINFO_UNALIGNED_DR(L)         (L)->unaligned_dr
#define LOOP_VINFO_MAY_MISALIGN_STMTS(L)   (L)->may_misalign_stmts
#define LOOP_VINFO_MAY_ALIAS_DDRS(L)       (L)->may_alias_ddrs
#define LOOP_VINFO_COMP_ALIAS_DDRS(L)      (L)->comp_alias_ddrs
#define LOOP_VINFO_CHECK_UNEQUAL_ADDRS(L)  (L)->check_unequal_addrs
#define LOOP_VINFO_CHECK_NONZERO(L)        (L)->check_nonzero
#define LOOP_VINFO_LOWER_BOUNDS(L)         (L)->lower_bounds
#define LOOP_VINFO_GROUPED_STORES(L)       (L)->grouped_stores
#define LOOP_VINFO_SLP_INSTANCES(L)        (L)->slp_instances
#define LOOP_VINFO_SLP_UNROLLING_FACTOR(L) (L)->slp_unrolling_factor
#define LOOP_VINFO_REDUCTIONS(L)           (L)->reductions
#define LOOP_VINFO_REDUCTION_CHAINS(L)     (L)->reduction_chains
#define LOOP_VINFO_TARGET_COST_DATA(L)     (L)->target_cost_data
#define LOOP_VINFO_PEELING_FOR_GAPS(L)     (L)->peeling_for_gaps
#define LOOP_VINFO_OPERANDS_SWAPPED(L)     (L)->operands_swapped
#define LOOP_VINFO_PEELING_FOR_NITER(L)    (L)->peeling_for_niter
#define LOOP_VINFO_NO_DATA_DEPENDENCIES(L) (L)->no_data_dependencies
#define LOOP_VINFO_SCALAR_LOOP(L)	   (L)->scalar_loop
#define LOOP_VINFO_HAS_MASK_STORE(L)       (L)->has_mask_store
#define LOOP_VINFO_SCALAR_ITERATION_COST(L) (L)->scalar_cost_vec
#define LOOP_VINFO_SINGLE_SCALAR_ITERATION_COST(L) (L)->single_scalar_iteration_cost
#define LOOP_VINFO_ORIG_LOOP_INFO(L)       (L)->orig_loop_info
#define LOOP_REQUIRES_VERSIONING_FOR_ALIGNMENT(L)	\
((L)->may_misalign_stmts.length () > 0)
#define LOOP_REQUIRES_VERSIONING_FOR_ALIAS(L)		\
((L)->comp_alias_ddrs.length () > 0 \
|| (L)->check_unequal_addrs.length () > 0 \
|| (L)->lower_bounds.length () > 0)
#define LOOP_REQUIRES_VERSIONING_FOR_NITERS(L)		\
(LOOP_VINFO_NITERS_ASSUMPTIONS (L))
#define LOOP_REQUIRES_VERSIONING(L)			\
(LOOP_REQUIRES_VERSIONING_FOR_ALIGNMENT (L)		\
|| LOOP_REQUIRES_VERSIONING_FOR_ALIAS (L)		\
|| LOOP_REQUIRES_VERSIONING_FOR_NITERS (L))
#define LOOP_VINFO_NITERS_KNOWN_P(L)          \
(tree_fits_shwi_p ((L)->num_iters) && tree_to_shwi ((L)->num_iters) > 0)
#define LOOP_VINFO_EPILOGUE_P(L) \
(LOOP_VINFO_ORIG_LOOP_INFO (L) != NULL)
#define LOOP_VINFO_ORIG_MAX_VECT_FACTOR(L) \
(LOOP_VINFO_MAX_VECT_FACTOR (LOOP_VINFO_ORIG_LOOP_INFO (L)))
static inline loop_vec_info
loop_vec_info_for_loop (struct loop *loop)
{
return (loop_vec_info) loop->aux;
}
static inline bool
nested_in_vect_loop_p (struct loop *loop, gimple *stmt)
{
return (loop->inner
&& (loop->inner == (gimple_bb (stmt))->loop_father));
}
typedef struct _bb_vec_info : public vec_info
{
_bb_vec_info (gimple_stmt_iterator, gimple_stmt_iterator);
~_bb_vec_info ();
basic_block bb;
gimple_stmt_iterator region_begin;
gimple_stmt_iterator region_end;
} *bb_vec_info;
#define BB_VINFO_BB(B)               (B)->bb
#define BB_VINFO_GROUPED_STORES(B)   (B)->grouped_stores
#define BB_VINFO_SLP_INSTANCES(B)    (B)->slp_instances
#define BB_VINFO_DATAREFS(B)         (B)->datarefs
#define BB_VINFO_DDRS(B)             (B)->ddrs
#define BB_VINFO_TARGET_COST_DATA(B) (B)->target_cost_data
static inline bb_vec_info
vec_info_for_bb (basic_block bb)
{
return (bb_vec_info) bb->aux;
}
enum stmt_vec_info_type {
undef_vec_info_type = 0,
load_vec_info_type,
store_vec_info_type,
shift_vec_info_type,
op_vec_info_type,
call_vec_info_type,
call_simd_clone_vec_info_type,
assignment_vec_info_type,
condition_vec_info_type,
comparison_vec_info_type,
reduc_vec_info_type,
induc_vec_info_type,
type_promotion_vec_info_type,
type_demotion_vec_info_type,
type_conversion_vec_info_type,
loop_exit_ctrl_vec_info_type
};
enum vect_relevant {
vect_unused_in_scope = 0,
vect_used_only_live,
vect_used_in_outer_by_reduction,
vect_used_in_outer,
vect_used_by_reduction,
vect_used_in_scope
};
enum slp_vect_type {
loop_vect = 0,
pure_slp,
hybrid
};
enum vec_load_store_type {
VLS_LOAD,
VLS_STORE,
VLS_STORE_INVARIANT
};
enum vect_memory_access_type {
VMAT_INVARIANT,
VMAT_CONTIGUOUS,
VMAT_CONTIGUOUS_DOWN,
VMAT_CONTIGUOUS_PERMUTE,
VMAT_CONTIGUOUS_REVERSE,
VMAT_LOAD_STORE_LANES,
VMAT_ELEMENTWISE,
VMAT_STRIDED_SLP,
VMAT_GATHER_SCATTER
};
typedef struct data_reference *dr_p;
typedef struct _stmt_vec_info {
enum stmt_vec_info_type type;
bool live;
bool in_pattern_p;
bool vectorizable;
gimple *stmt;
vec_info *vinfo;
tree vectype;
gimple *vectorized_stmt;
struct data_reference *data_ref_info;
innermost_loop_behavior dr_wrt_vec_loop;
tree loop_phi_evolution_base_unchanged;
tree loop_phi_evolution_part;
gimple *related_stmt;
gimple_seq pattern_def_seq;
vec<dr_p> same_align_refs;
vec<tree> simd_clone_info;
enum vect_def_type def_type;
enum slp_vect_type slp_type;
gimple *first_element;
gimple *next_element;
gimple *same_dr_stmt;
unsigned int size;
unsigned int store_count;
unsigned int gap;
unsigned int min_neg_dist;
enum vect_relevant relevant;
bool gather_scatter_p;
bool strided_p;
bool simd_lane_access_p;
vect_memory_access_type memory_access_type;
enum vect_reduction_type v_reduc_type;
enum tree_code const_cond_reduc_code;
enum vect_reduction_type reduc_type;
gimple *reduc_def;
unsigned int num_slp_uses;
} *stmt_vec_info;
struct gather_scatter_info {
internal_fn ifn;
tree decl;
tree base;
tree offset;
int scale;
enum vect_def_type offset_dt;
tree offset_vectype;
tree element_type;
tree memory_type;
};
#define STMT_VINFO_TYPE(S)                 (S)->type
#define STMT_VINFO_STMT(S)                 (S)->stmt
inline loop_vec_info
STMT_VINFO_LOOP_VINFO (stmt_vec_info stmt_vinfo)
{
if (loop_vec_info loop_vinfo = dyn_cast <loop_vec_info> (stmt_vinfo->vinfo))
return loop_vinfo;
return NULL;
}
inline bb_vec_info
STMT_VINFO_BB_VINFO (stmt_vec_info stmt_vinfo)
{
if (bb_vec_info bb_vinfo = dyn_cast <bb_vec_info> (stmt_vinfo->vinfo))
return bb_vinfo;
return NULL;
}
#define STMT_VINFO_RELEVANT(S)             (S)->relevant
#define STMT_VINFO_LIVE_P(S)               (S)->live
#define STMT_VINFO_VECTYPE(S)              (S)->vectype
#define STMT_VINFO_VEC_STMT(S)             (S)->vectorized_stmt
#define STMT_VINFO_VECTORIZABLE(S)         (S)->vectorizable
#define STMT_VINFO_DATA_REF(S)             (S)->data_ref_info
#define STMT_VINFO_GATHER_SCATTER_P(S)	   (S)->gather_scatter_p
#define STMT_VINFO_STRIDED_P(S)	   	   (S)->strided_p
#define STMT_VINFO_MEMORY_ACCESS_TYPE(S)   (S)->memory_access_type
#define STMT_VINFO_SIMD_LANE_ACCESS_P(S)   (S)->simd_lane_access_p
#define STMT_VINFO_VEC_REDUCTION_TYPE(S)   (S)->v_reduc_type
#define STMT_VINFO_VEC_CONST_COND_REDUC_CODE(S) (S)->const_cond_reduc_code
#define STMT_VINFO_DR_WRT_VEC_LOOP(S)      (S)->dr_wrt_vec_loop
#define STMT_VINFO_DR_BASE_ADDRESS(S)      (S)->dr_wrt_vec_loop.base_address
#define STMT_VINFO_DR_INIT(S)              (S)->dr_wrt_vec_loop.init
#define STMT_VINFO_DR_OFFSET(S)            (S)->dr_wrt_vec_loop.offset
#define STMT_VINFO_DR_STEP(S)              (S)->dr_wrt_vec_loop.step
#define STMT_VINFO_DR_BASE_ALIGNMENT(S)    (S)->dr_wrt_vec_loop.base_alignment
#define STMT_VINFO_DR_BASE_MISALIGNMENT(S) \
(S)->dr_wrt_vec_loop.base_misalignment
#define STMT_VINFO_DR_OFFSET_ALIGNMENT(S) \
(S)->dr_wrt_vec_loop.offset_alignment
#define STMT_VINFO_DR_STEP_ALIGNMENT(S) \
(S)->dr_wrt_vec_loop.step_alignment
#define STMT_VINFO_IN_PATTERN_P(S)         (S)->in_pattern_p
#define STMT_VINFO_RELATED_STMT(S)         (S)->related_stmt
#define STMT_VINFO_PATTERN_DEF_SEQ(S)      (S)->pattern_def_seq
#define STMT_VINFO_SAME_ALIGN_REFS(S)      (S)->same_align_refs
#define STMT_VINFO_SIMD_CLONE_INFO(S)	   (S)->simd_clone_info
#define STMT_VINFO_DEF_TYPE(S)             (S)->def_type
#define STMT_VINFO_GROUP_FIRST_ELEMENT(S)  (S)->first_element
#define STMT_VINFO_GROUP_NEXT_ELEMENT(S)   (S)->next_element
#define STMT_VINFO_GROUP_SIZE(S)           (S)->size
#define STMT_VINFO_GROUP_STORE_COUNT(S)    (S)->store_count
#define STMT_VINFO_GROUP_GAP(S)            (S)->gap
#define STMT_VINFO_GROUP_SAME_DR_STMT(S)   (S)->same_dr_stmt
#define STMT_VINFO_GROUPED_ACCESS(S)      ((S)->first_element != NULL && (S)->data_ref_info)
#define STMT_VINFO_LOOP_PHI_EVOLUTION_BASE_UNCHANGED(S) (S)->loop_phi_evolution_base_unchanged
#define STMT_VINFO_LOOP_PHI_EVOLUTION_PART(S) (S)->loop_phi_evolution_part
#define STMT_VINFO_MIN_NEG_DIST(S)	(S)->min_neg_dist
#define STMT_VINFO_NUM_SLP_USES(S)	(S)->num_slp_uses
#define STMT_VINFO_REDUC_TYPE(S)	(S)->reduc_type
#define STMT_VINFO_REDUC_DEF(S)		(S)->reduc_def
#define GROUP_FIRST_ELEMENT(S)          (S)->first_element
#define GROUP_NEXT_ELEMENT(S)           (S)->next_element
#define GROUP_SIZE(S)                   (S)->size
#define GROUP_STORE_COUNT(S)            (S)->store_count
#define GROUP_GAP(S)                    (S)->gap
#define GROUP_SAME_DR_STMT(S)           (S)->same_dr_stmt
#define STMT_VINFO_RELEVANT_P(S)          ((S)->relevant != vect_unused_in_scope)
#define HYBRID_SLP_STMT(S)                ((S)->slp_type == hybrid)
#define PURE_SLP_STMT(S)                  ((S)->slp_type == pure_slp)
#define STMT_SLP_TYPE(S)                   (S)->slp_type
struct dataref_aux {
int misalignment;
int target_alignment;
bool base_misaligned;
tree base_decl;
};
#define DR_VECT_AUX(dr) ((dataref_aux *)(dr)->aux)
#define VECT_MAX_COST 1000
#define MAX_INTERM_CVT_STEPS         3
#define MAX_VECTORIZATION_FACTOR INT_MAX
#define VECT_SCALAR_BOOLEAN_TYPE_P(TYPE) \
(TREE_CODE (TYPE) == BOOLEAN_TYPE		\
|| ((TREE_CODE (TYPE) == INTEGER_TYPE	\
|| TREE_CODE (TYPE) == ENUMERAL_TYPE)	\
&& TYPE_PRECISION (TYPE) == 1		\
&& TYPE_UNSIGNED (TYPE)))
extern vec<stmt_vec_info> stmt_vec_info_vec;
void init_stmt_vec_info_vec (void);
void free_stmt_vec_info_vec (void);
static inline stmt_vec_info
vinfo_for_stmt (gimple *stmt)
{
int uid = gimple_uid (stmt);
if (uid <= 0)
return NULL;
return stmt_vec_info_vec[uid - 1];
}
static inline void
set_vinfo_for_stmt (gimple *stmt, stmt_vec_info info)
{
unsigned int uid = gimple_uid (stmt);
if (uid == 0)
{
gcc_checking_assert (info);
uid = stmt_vec_info_vec.length () + 1;
gimple_set_uid (stmt, uid);
stmt_vec_info_vec.safe_push (info);
}
else
{
gcc_checking_assert (info == NULL);
stmt_vec_info_vec[uid - 1] = info;
}
}
static inline bool
is_pattern_stmt_p (stmt_vec_info stmt_info)
{
gimple *related_stmt;
stmt_vec_info related_stmt_info;
related_stmt = STMT_VINFO_RELATED_STMT (stmt_info);
if (related_stmt
&& (related_stmt_info = vinfo_for_stmt (related_stmt))
&& STMT_VINFO_IN_PATTERN_P (related_stmt_info))
return true;
return false;
}
static inline gimple *
get_later_stmt (gimple *stmt1, gimple *stmt2)
{
unsigned int uid1, uid2;
if (stmt1 == NULL)
return stmt2;
if (stmt2 == NULL)
return stmt1;
stmt_vec_info stmt_info1 = vinfo_for_stmt (stmt1);
stmt_vec_info stmt_info2 = vinfo_for_stmt (stmt2);
uid1 = gimple_uid (is_pattern_stmt_p (stmt_info1)
? STMT_VINFO_RELATED_STMT (stmt_info1) : stmt1);
uid2 = gimple_uid (is_pattern_stmt_p (stmt_info2)
? STMT_VINFO_RELATED_STMT (stmt_info2) : stmt2);
if (uid1 == 0 || uid2 == 0)
return NULL;
gcc_assert (uid1 <= stmt_vec_info_vec.length ());
gcc_assert (uid2 <= stmt_vec_info_vec.length ());
if (uid1 > uid2)
return stmt1;
else
return stmt2;
}
static inline bool
is_loop_header_bb_p (basic_block bb)
{
if (bb == (bb->loop_father)->header)
return true;
gcc_checking_assert (EDGE_COUNT (bb->preds) == 1);
return false;
}
static inline int
vect_pow2 (int x)
{
int i, res = 1;
for (i = 0; i < x; i++)
res *= 2;
return res;
}
static inline int
builtin_vectorization_cost (enum vect_cost_for_stmt type_of_cost,
tree vectype, int misalign)
{
return targetm.vectorize.builtin_vectorization_cost (type_of_cost,
vectype, misalign);
}
static inline
int vect_get_stmt_cost (enum vect_cost_for_stmt type_of_cost)
{
return builtin_vectorization_cost (type_of_cost, NULL, 0);
}
static inline void *
init_cost (struct loop *loop_info)
{
return targetm.vectorize.init_cost (loop_info);
}
static inline unsigned
add_stmt_cost (void *data, int count, enum vect_cost_for_stmt kind,
stmt_vec_info stmt_info, int misalign,
enum vect_cost_model_location where)
{
return targetm.vectorize.add_stmt_cost (data, count, kind,
stmt_info, misalign, where);
}
static inline void
finish_cost (void *data, unsigned *prologue_cost,
unsigned *body_cost, unsigned *epilogue_cost)
{
targetm.vectorize.finish_cost (data, prologue_cost, body_cost, epilogue_cost);
}
static inline void
destroy_cost_data (void *data)
{
targetm.vectorize.destroy_cost_data (data);
}
inline void
set_dr_misalignment (struct data_reference *dr, int val)
{
dataref_aux *data_aux = DR_VECT_AUX (dr);
if (!data_aux)
{
data_aux = XCNEW (dataref_aux);
dr->aux = data_aux;
}
data_aux->misalignment = val;
}
inline int
dr_misalignment (struct data_reference *dr)
{
return DR_VECT_AUX (dr)->misalignment;
}
#define DR_MISALIGNMENT(DR) dr_misalignment (DR)
#define SET_DR_MISALIGNMENT(DR, VAL) set_dr_misalignment (DR, VAL)
#define DR_MISALIGNMENT_UNKNOWN (-1)
#define DR_TARGET_ALIGNMENT(DR) DR_VECT_AUX (DR)->target_alignment
static inline bool
aligned_access_p (struct data_reference *data_ref_info)
{
return (DR_MISALIGNMENT (data_ref_info) == 0);
}
static inline bool
known_alignment_for_access_p (struct data_reference *data_ref_info)
{
return (DR_MISALIGNMENT (data_ref_info) != DR_MISALIGNMENT_UNKNOWN);
}
static inline unsigned int
vect_known_alignment_in_bytes (struct data_reference *dr)
{
if (DR_MISALIGNMENT (dr) == DR_MISALIGNMENT_UNKNOWN)
return TYPE_ALIGN_UNIT (TREE_TYPE (DR_REF (dr)));
if (DR_MISALIGNMENT (dr) == 0)
return DR_TARGET_ALIGNMENT (dr);
return DR_MISALIGNMENT (dr) & -DR_MISALIGNMENT (dr);
}
static inline innermost_loop_behavior *
vect_dr_behavior (data_reference *dr)
{
gimple *stmt = DR_STMT (dr);
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
loop_vec_info loop_vinfo = STMT_VINFO_LOOP_VINFO (stmt_info);
if (loop_vinfo == NULL
|| !nested_in_vect_loop_p (LOOP_VINFO_LOOP (loop_vinfo), stmt))
return &DR_INNERMOST (dr);
else
return &STMT_VINFO_DR_WRT_VEC_LOOP (stmt_info);
}
static inline bool
unlimited_cost_model (loop_p loop)
{
if (loop != NULL && loop->force_vectorize
&& flag_simd_cost_model != VECT_COST_MODEL_DEFAULT)
return flag_simd_cost_model == VECT_COST_MODEL_UNLIMITED;
return (flag_vect_cost_model == VECT_COST_MODEL_UNLIMITED);
}
static inline bool
vect_use_loop_mask_for_alignment_p (loop_vec_info loop_vinfo)
{
return (LOOP_VINFO_FULLY_MASKED_P (loop_vinfo)
&& LOOP_VINFO_PEELING_FOR_ALIGNMENT (loop_vinfo));
}
static inline unsigned int
vect_get_num_vectors (poly_uint64 nunits, tree vectype)
{
return exact_div (nunits, TYPE_VECTOR_SUBPARTS (vectype)).to_constant ();
}
static inline unsigned int
vect_get_num_copies (loop_vec_info loop_vinfo, tree vectype)
{
return vect_get_num_vectors (LOOP_VINFO_VECT_FACTOR (loop_vinfo), vectype);
}
static inline void
vect_update_max_nunits (poly_uint64 *max_nunits, tree vectype)
{
poly_uint64 nunits = TYPE_VECTOR_SUBPARTS (vectype);
*max_nunits = force_common_multiple (*max_nunits, nunits);
}
static inline unsigned int
vect_vf_for_cost (loop_vec_info loop_vinfo)
{
return estimated_poly_value (LOOP_VINFO_VECT_FACTOR (loop_vinfo));
}
static inline unsigned int
vect_nunits_for_cost (tree vec_type)
{
return estimated_poly_value (TYPE_VECTOR_SUBPARTS (vec_type));
}
static inline unsigned HOST_WIDE_INT
vect_max_vf (loop_vec_info loop_vinfo)
{
unsigned HOST_WIDE_INT vf;
if (LOOP_VINFO_VECT_FACTOR (loop_vinfo).is_constant (&vf))
return vf;
return MAX_VECTORIZATION_FACTOR;
}
inline unsigned int
vect_get_scalar_dr_size (struct data_reference *dr)
{
return tree_to_uhwi (TYPE_SIZE_UNIT (TREE_TYPE (DR_REF (dr))));
}
extern source_location vect_location;
extern void vect_set_loop_condition (struct loop *, loop_vec_info,
tree, tree, tree, bool);
extern bool slpeel_can_duplicate_loop_p (const struct loop *, const_edge);
struct loop *slpeel_tree_duplicate_loop_to_edge_cfg (struct loop *,
struct loop *, edge);
extern void vect_loop_versioning (loop_vec_info, unsigned int, bool,
poly_uint64);
extern struct loop *vect_do_peeling (loop_vec_info, tree, tree,
tree *, tree *, tree *, int, bool, bool);
extern void vect_prepare_for_masked_peels (loop_vec_info);
extern source_location find_loop_location (struct loop *);
extern bool vect_can_advance_ivs_p (loop_vec_info);
extern poly_uint64 current_vector_size;
extern tree get_vectype_for_scalar_type (tree);
extern tree get_vectype_for_scalar_type_and_size (tree, poly_uint64);
extern tree get_mask_type_for_scalar_type (tree);
extern tree get_same_sized_vectype (tree, tree);
extern bool vect_get_loop_mask_type (loop_vec_info);
extern bool vect_is_simple_use (tree, vec_info *, gimple **,
enum vect_def_type *);
extern bool vect_is_simple_use (tree, vec_info *, gimple **,
enum vect_def_type *, tree *);
extern bool supportable_widening_operation (enum tree_code, gimple *, tree,
tree, enum tree_code *,
enum tree_code *, int *,
vec<tree> *);
extern bool supportable_narrowing_operation (enum tree_code, tree, tree,
enum tree_code *,
int *, vec<tree> *);
extern stmt_vec_info new_stmt_vec_info (gimple *stmt, vec_info *);
extern void free_stmt_vec_info (gimple *stmt);
extern void vect_model_simple_cost (stmt_vec_info, int, enum vect_def_type *,
int, stmt_vector_for_cost *,
stmt_vector_for_cost *);
extern void vect_model_store_cost (stmt_vec_info, int, vect_memory_access_type,
vec_load_store_type, slp_tree,
stmt_vector_for_cost *,
stmt_vector_for_cost *);
extern void vect_model_load_cost (stmt_vec_info, int, vect_memory_access_type,
slp_tree, stmt_vector_for_cost *,
stmt_vector_for_cost *);
extern unsigned record_stmt_cost (stmt_vector_for_cost *, int,
enum vect_cost_for_stmt, stmt_vec_info,
int, enum vect_cost_model_location);
extern void vect_finish_replace_stmt (gimple *, gimple *);
extern void vect_finish_stmt_generation (gimple *, gimple *,
gimple_stmt_iterator *);
extern bool vect_mark_stmts_to_be_vectorized (loop_vec_info);
extern tree vect_get_store_rhs (gimple *);
extern tree vect_get_vec_def_for_operand_1 (gimple *, enum vect_def_type);
extern tree vect_get_vec_def_for_operand (tree, gimple *, tree = NULL);
extern void vect_get_vec_defs (tree, tree, gimple *, vec<tree> *,
vec<tree> *, slp_tree);
extern void vect_get_vec_defs_for_stmt_copy (enum vect_def_type *,
vec<tree> *, vec<tree> *);
extern tree vect_init_vector (gimple *, tree, tree,
gimple_stmt_iterator *);
extern tree vect_get_vec_def_for_stmt_copy (enum vect_def_type, tree);
extern bool vect_transform_stmt (gimple *, gimple_stmt_iterator *,
bool *, slp_tree, slp_instance);
extern void vect_remove_stores (gimple *);
extern bool vect_analyze_stmt (gimple *, bool *, slp_tree, slp_instance);
extern bool vectorizable_condition (gimple *, gimple_stmt_iterator *,
gimple **, tree, int, slp_tree);
extern void vect_get_load_cost (struct data_reference *, int, bool,
unsigned int *, unsigned int *,
stmt_vector_for_cost *,
stmt_vector_for_cost *, bool);
extern void vect_get_store_cost (struct data_reference *, int,
unsigned int *, stmt_vector_for_cost *);
extern bool vect_supportable_shift (enum tree_code, tree);
extern tree vect_gen_perm_mask_any (tree, const vec_perm_indices &);
extern tree vect_gen_perm_mask_checked (tree, const vec_perm_indices &);
extern void optimize_mask_stores (struct loop*);
extern gcall *vect_gen_while (tree, tree, tree);
extern tree vect_gen_while_not (gimple_seq *, tree, tree, tree);
extern bool vect_can_force_dr_alignment_p (const_tree, unsigned int);
extern enum dr_alignment_support vect_supportable_dr_alignment
(struct data_reference *, bool);
extern tree vect_get_smallest_scalar_type (gimple *, HOST_WIDE_INT *,
HOST_WIDE_INT *);
extern bool vect_analyze_data_ref_dependences (loop_vec_info, unsigned int *);
extern bool vect_slp_analyze_instance_dependence (slp_instance);
extern bool vect_enhance_data_refs_alignment (loop_vec_info);
extern bool vect_analyze_data_refs_alignment (loop_vec_info);
extern bool vect_verify_datarefs_alignment (loop_vec_info);
extern bool vect_slp_analyze_and_verify_instance_alignment (slp_instance);
extern bool vect_analyze_data_ref_accesses (vec_info *);
extern bool vect_prune_runtime_alias_test_list (loop_vec_info);
extern bool vect_gather_scatter_fn_p (bool, bool, tree, tree, unsigned int,
signop, int, internal_fn *, tree *);
extern bool vect_check_gather_scatter (gimple *, loop_vec_info,
gather_scatter_info *);
extern bool vect_analyze_data_refs (vec_info *, poly_uint64 *);
extern void vect_record_base_alignments (vec_info *);
extern tree vect_create_data_ref_ptr (gimple *, tree, struct loop *, tree,
tree *, gimple_stmt_iterator *,
gimple **, bool, bool *,
tree = NULL_TREE, tree = NULL_TREE);
extern tree bump_vector_ptr (tree, gimple *, gimple_stmt_iterator *, gimple *,
tree);
extern void vect_copy_ref_info (tree, tree);
extern tree vect_create_destination_var (tree, tree);
extern bool vect_grouped_store_supported (tree, unsigned HOST_WIDE_INT);
extern bool vect_store_lanes_supported (tree, unsigned HOST_WIDE_INT, bool);
extern bool vect_grouped_load_supported (tree, bool, unsigned HOST_WIDE_INT);
extern bool vect_load_lanes_supported (tree, unsigned HOST_WIDE_INT, bool);
extern void vect_permute_store_chain (vec<tree> ,unsigned int, gimple *,
gimple_stmt_iterator *, vec<tree> *);
extern tree vect_setup_realignment (gimple *, gimple_stmt_iterator *, tree *,
enum dr_alignment_support, tree,
struct loop **);
extern void vect_transform_grouped_load (gimple *, vec<tree> , int,
gimple_stmt_iterator *);
extern void vect_record_grouped_load_vectors (gimple *, vec<tree> );
extern tree vect_get_new_vect_var (tree, enum vect_var_kind, const char *);
extern tree vect_get_new_ssa_name (tree, enum vect_var_kind,
const char * = NULL);
extern tree vect_create_addr_base_for_vector_ref (gimple *, gimple_seq *,
tree, tree = NULL_TREE);
extern gimple *vect_force_simple_reduction (loop_vec_info, gimple *,
bool *, bool);
extern bool check_reduction_path (location_t, loop_p, gphi *, tree,
enum tree_code);
extern loop_vec_info vect_analyze_loop (struct loop *, loop_vec_info);
extern tree vect_build_loop_niters (loop_vec_info, bool * = NULL);
extern void vect_gen_vector_loop_niters (loop_vec_info, tree, tree *,
tree *, bool);
extern tree vect_halve_mask_nunits (tree);
extern tree vect_double_mask_nunits (tree);
extern void vect_record_loop_mask (loop_vec_info, vec_loop_masks *,
unsigned int, tree);
extern tree vect_get_loop_mask (gimple_stmt_iterator *, vec_loop_masks *,
unsigned int, tree, unsigned int);
extern struct loop *vect_transform_loop (loop_vec_info);
extern loop_vec_info vect_analyze_loop_form (struct loop *);
extern bool vectorizable_live_operation (gimple *, gimple_stmt_iterator *,
slp_tree, int, gimple **);
extern bool vectorizable_reduction (gimple *, gimple_stmt_iterator *,
gimple **, slp_tree, slp_instance);
extern bool vectorizable_induction (gimple *, gimple_stmt_iterator *,
gimple **, slp_tree);
extern tree get_initial_def_for_reduction (gimple *, tree, tree *);
extern bool vect_worthwhile_without_simd_p (vec_info *, tree_code);
extern int vect_get_known_peeling_cost (loop_vec_info, int, int *,
stmt_vector_for_cost *,
stmt_vector_for_cost *,
stmt_vector_for_cost *);
extern tree cse_and_gimplify_to_preheader (loop_vec_info, tree);
extern void vect_free_slp_instance (slp_instance);
extern bool vect_transform_slp_perm_load (slp_tree, vec<tree> ,
gimple_stmt_iterator *, poly_uint64,
slp_instance, bool, unsigned *);
extern bool vect_slp_analyze_operations (vec_info *);
extern bool vect_schedule_slp (vec_info *);
extern bool vect_analyze_slp (vec_info *, unsigned);
extern bool vect_make_slp_decision (loop_vec_info);
extern void vect_detect_hybrid_slp (loop_vec_info);
extern void vect_get_slp_defs (vec<tree> , slp_tree, vec<vec<tree> > *);
extern bool vect_slp_bb (basic_block);
extern gimple *vect_find_last_scalar_stmt_in_slp (slp_tree);
extern bool is_simple_and_all_uses_invariant (gimple *, loop_vec_info);
extern bool can_duplicate_and_interleave_p (unsigned int, machine_mode,
unsigned int * = NULL,
tree * = NULL, tree * = NULL);
extern void duplicate_and_interleave (gimple_seq *, tree, vec<tree>,
unsigned int, vec<tree> &);
extern int vect_get_place_in_interleaving_chain (gimple *, gimple *);
typedef gimple *(* vect_recog_func_ptr) (vec<gimple *> *, tree *, tree *);
#define NUM_PATTERNS 15
void vect_pattern_recog (vec_info *);
unsigned vectorize_loops (void);
bool vect_stmt_in_region_p (vec_info *, gimple *);
void vect_free_loop_info_assumptions (struct loop *);
#endif  
