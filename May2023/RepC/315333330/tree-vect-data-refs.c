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
#include "ssa.h"
#include "optabs-tree.h"
#include "cgraph.h"
#include "dumpfile.h"
#include "alias.h"
#include "fold-const.h"
#include "stor-layout.h"
#include "tree-eh.h"
#include "gimplify.h"
#include "gimple-iterator.h"
#include "gimplify-me.h"
#include "tree-ssa-loop-ivopts.h"
#include "tree-ssa-loop-manip.h"
#include "tree-ssa-loop.h"
#include "cfgloop.h"
#include "tree-scalar-evolution.h"
#include "tree-vectorizer.h"
#include "expr.h"
#include "builtins.h"
#include "params.h"
#include "tree-cfg.h"
#include "tree-hash-traits.h"
#include "vec-perm-indices.h"
#include "internal-fn.h"
static bool
vect_lanes_optab_supported_p (const char *name, convert_optab optab,
tree vectype, unsigned HOST_WIDE_INT count)
{
machine_mode mode, array_mode;
bool limit_p;
mode = TYPE_MODE (vectype);
if (!targetm.array_mode (mode, count).exists (&array_mode))
{
poly_uint64 bits = count * GET_MODE_BITSIZE (mode);
limit_p = !targetm.array_mode_supported_p (mode, count);
if (!int_mode_for_size (bits, limit_p).exists (&array_mode))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"no array mode for %s["
HOST_WIDE_INT_PRINT_DEC "]\n",
GET_MODE_NAME (mode), count);
return false;
}
}
if (convert_optab_handler (optab, array_mode, mode) == CODE_FOR_nothing)
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"cannot use %s<%s><%s>\n", name,
GET_MODE_NAME (array_mode), GET_MODE_NAME (mode));
return false;
}
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"can use %s<%s><%s>\n", name, GET_MODE_NAME (array_mode),
GET_MODE_NAME (mode));
return true;
}
tree
vect_get_smallest_scalar_type (gimple *stmt, HOST_WIDE_INT *lhs_size_unit,
HOST_WIDE_INT *rhs_size_unit)
{
tree scalar_type = gimple_expr_type (stmt);
HOST_WIDE_INT lhs, rhs;
if (!tree_fits_uhwi_p (TYPE_SIZE_UNIT (scalar_type)))
return scalar_type;
lhs = rhs = TREE_INT_CST_LOW (TYPE_SIZE_UNIT (scalar_type));
if (is_gimple_assign (stmt)
&& (gimple_assign_cast_p (stmt)
|| gimple_assign_rhs_code (stmt) == DOT_PROD_EXPR
|| gimple_assign_rhs_code (stmt) == WIDEN_SUM_EXPR
|| gimple_assign_rhs_code (stmt) == WIDEN_MULT_EXPR
|| gimple_assign_rhs_code (stmt) == WIDEN_LSHIFT_EXPR
|| gimple_assign_rhs_code (stmt) == FLOAT_EXPR))
{
tree rhs_type = TREE_TYPE (gimple_assign_rhs1 (stmt));
rhs = TREE_INT_CST_LOW (TYPE_SIZE_UNIT (rhs_type));
if (rhs < lhs)
scalar_type = rhs_type;
}
*lhs_size_unit = lhs;
*rhs_size_unit = rhs;
return scalar_type;
}
static bool
vect_mark_for_runtime_alias_test (ddr_p ddr, loop_vec_info loop_vinfo)
{
struct loop *loop = LOOP_VINFO_LOOP (loop_vinfo);
if ((unsigned) PARAM_VALUE (PARAM_VECT_MAX_VERSION_FOR_ALIAS_CHECKS) == 0)
return false;
if (!runtime_alias_check_p (ddr, loop,
optimize_loop_nest_for_speed_p (loop)))
return false;
LOOP_VINFO_MAY_ALIAS_DDRS (loop_vinfo).safe_push (ddr);
return true;
}
static void
vect_check_nonzero_value (loop_vec_info loop_vinfo, tree value)
{
vec<tree> checks = LOOP_VINFO_CHECK_NONZERO (loop_vinfo);
for (unsigned int i = 0; i < checks.length(); ++i)
if (checks[i] == value)
return;
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location, "need run-time check that ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, value);
dump_printf (MSG_NOTE, " is nonzero\n");
}
LOOP_VINFO_CHECK_NONZERO (loop_vinfo).safe_push (value);
}
static bool
vect_preserves_scalar_order_p (gimple *stmt_a, gimple *stmt_b)
{
stmt_vec_info stmtinfo_a = vinfo_for_stmt (stmt_a);
stmt_vec_info stmtinfo_b = vinfo_for_stmt (stmt_b);
if (!STMT_VINFO_GROUPED_ACCESS (stmtinfo_a)
&& !STMT_VINFO_GROUPED_ACCESS (stmtinfo_b))
return true;
gimple *last_a = GROUP_FIRST_ELEMENT (stmtinfo_a);
if (last_a)
for (gimple *s = GROUP_NEXT_ELEMENT (vinfo_for_stmt (last_a)); s;
s = GROUP_NEXT_ELEMENT (vinfo_for_stmt (s)))
last_a = get_later_stmt (last_a, s);
else
last_a = stmt_a;
gimple *last_b = GROUP_FIRST_ELEMENT (stmtinfo_b);
if (last_b)
for (gimple *s = GROUP_NEXT_ELEMENT (vinfo_for_stmt (last_b)); s;
s = GROUP_NEXT_ELEMENT (vinfo_for_stmt (s)))
last_b = get_later_stmt (last_b, s);
else
last_b = stmt_b;
return ((get_later_stmt (last_a, last_b) == last_a)
== (get_later_stmt (stmt_a, stmt_b) == stmt_a));
}
static bool
vect_analyze_possibly_independent_ddr (data_dependence_relation *ddr,
loop_vec_info loop_vinfo,
int loop_depth, unsigned int *max_vf)
{
struct loop *loop = LOOP_VINFO_LOOP (loop_vinfo);
lambda_vector dist_v;
unsigned int i;
FOR_EACH_VEC_ELT (DDR_DIST_VECTS (ddr), i, dist_v)
{
int dist = dist_v[loop_depth];
if (dist != 0 && !(dist > 0 && DDR_REVERSED_P (ddr)))
{
if (loop->safelen >= 2 && abs_hwi (dist) <= loop->safelen)
{
if ((unsigned int) loop->safelen < *max_vf)
*max_vf = loop->safelen;
LOOP_VINFO_NO_DATA_DEPENDENCIES (loop_vinfo) = false;
continue;
}
return vect_mark_for_runtime_alias_test (ddr, loop_vinfo);
}
}
return true;
}
static bool
vect_analyze_data_ref_dependence (struct data_dependence_relation *ddr,
loop_vec_info loop_vinfo,
unsigned int *max_vf)
{
unsigned int i;
struct loop *loop = LOOP_VINFO_LOOP (loop_vinfo);
struct data_reference *dra = DDR_A (ddr);
struct data_reference *drb = DDR_B (ddr);
stmt_vec_info stmtinfo_a = vinfo_for_stmt (DR_STMT (dra));
stmt_vec_info stmtinfo_b = vinfo_for_stmt (DR_STMT (drb));
lambda_vector dist_v;
unsigned int loop_depth;
if (!STMT_VINFO_VECTORIZABLE (stmtinfo_a)
|| !STMT_VINFO_VECTORIZABLE (stmtinfo_b))
gcc_unreachable ();
if (DDR_ARE_DEPENDENT (ddr) == chrec_known)
return false;
if (dra == drb
|| (DR_IS_READ (dra) && DR_IS_READ (drb)))
return false;
if (GROUP_FIRST_ELEMENT (stmtinfo_a)
&& GROUP_FIRST_ELEMENT (stmtinfo_a) == GROUP_FIRST_ELEMENT (stmtinfo_b)
&& !STMT_VINFO_STRIDED_P (stmtinfo_a))
return false;
if (((DR_IS_READ (dra) && DR_IS_WRITE (drb))
|| (DR_IS_WRITE (dra) && DR_IS_READ (drb)))
&& !alias_sets_conflict_p (get_alias_set (DR_REF (dra)),
get_alias_set (DR_REF (drb))))
return false;
if (DDR_ARE_DEPENDENT (ddr) == chrec_dont_know)
{
if (loop->safelen >= 2)
{
if ((unsigned int) loop->safelen < *max_vf)
*max_vf = loop->safelen;
LOOP_VINFO_NO_DATA_DEPENDENCIES (loop_vinfo) = false;
return false;
}
if (STMT_VINFO_GATHER_SCATTER_P (stmtinfo_a)
|| STMT_VINFO_GATHER_SCATTER_P (stmtinfo_b))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"versioning for alias not supported for: "
"can't determine dependence between ");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM,
DR_REF (dra));
dump_printf (MSG_MISSED_OPTIMIZATION, " and ");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM,
DR_REF (drb));
dump_printf (MSG_MISSED_OPTIMIZATION, "\n");
}
return true;
}
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"versioning for alias required: "
"can't determine dependence between ");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM,
DR_REF (dra));
dump_printf (MSG_MISSED_OPTIMIZATION, " and ");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM,
DR_REF (drb));
dump_printf (MSG_MISSED_OPTIMIZATION, "\n");
}
return !vect_mark_for_runtime_alias_test (ddr, loop_vinfo);
}
if (DDR_NUM_DIST_VECTS (ddr) == 0)
{
if (loop->safelen >= 2)
{
if ((unsigned int) loop->safelen < *max_vf)
*max_vf = loop->safelen;
LOOP_VINFO_NO_DATA_DEPENDENCIES (loop_vinfo) = false;
return false;
}
if (STMT_VINFO_GATHER_SCATTER_P (stmtinfo_a)
|| STMT_VINFO_GATHER_SCATTER_P (stmtinfo_b))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"versioning for alias not supported for: "
"bad dist vector for ");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM,
DR_REF (dra));
dump_printf (MSG_MISSED_OPTIMIZATION, " and ");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM,
DR_REF (drb));
dump_printf (MSG_MISSED_OPTIMIZATION, "\n");
}
return true;
}
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"versioning for alias required: "
"bad dist vector for ");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM, DR_REF (dra));
dump_printf (MSG_MISSED_OPTIMIZATION,  " and ");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM, DR_REF (drb));
dump_printf (MSG_MISSED_OPTIMIZATION, "\n");
}
return !vect_mark_for_runtime_alias_test (ddr, loop_vinfo);
}
loop_depth = index_in_loop_nest (loop->num, DDR_LOOP_NEST (ddr));
if (DDR_COULD_BE_INDEPENDENT_P (ddr)
&& vect_analyze_possibly_independent_ddr (ddr, loop_vinfo,
loop_depth, max_vf))
return false;
FOR_EACH_VEC_ELT (DDR_DIST_VECTS (ddr), i, dist_v)
{
int dist = dist_v[loop_depth];
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"dependence distance  = %d.\n", dist);
if (dist == 0)
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"dependence distance == 0 between ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (dra));
dump_printf (MSG_NOTE, " and ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (drb));
dump_printf (MSG_MISSED_OPTIMIZATION, "\n");
}
if (!vect_preserves_scalar_order_p (DR_STMT (dra), DR_STMT (drb)))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"READ_WRITE dependence in interleaving.\n");
return true;
}
if (loop->safelen < 2)
{
tree indicator = dr_zero_step_indicator (dra);
if (TREE_CODE (indicator) != INTEGER_CST)
vect_check_nonzero_value (loop_vinfo, indicator);
else if (integer_zerop (indicator))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"access also has a zero step\n");
return true;
}
}
continue;
}
if (dist > 0 && DDR_REVERSED_P (ddr))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"dependence distance negative.\n");
if (DR_IS_READ (drb)
&& (STMT_VINFO_MIN_NEG_DIST (stmtinfo_b) == 0
|| STMT_VINFO_MIN_NEG_DIST (stmtinfo_b) > (unsigned)dist))
STMT_VINFO_MIN_NEG_DIST (stmtinfo_b) = dist;
continue;
}
unsigned int abs_dist = abs (dist);
if (abs_dist >= 2 && abs_dist < *max_vf)
{
*max_vf = abs (dist);
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"adjusting maximal vectorization factor to %i\n",
*max_vf);
}
if (abs_dist >= *max_vf)
{
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"dependence distance >= VF.\n");
continue;
}
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized, possible dependence "
"between data-refs ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (dra));
dump_printf (MSG_NOTE,  " and ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (drb));
dump_printf (MSG_NOTE,  "\n");
}
return true;
}
return false;
}
bool
vect_analyze_data_ref_dependences (loop_vec_info loop_vinfo,
unsigned int *max_vf)
{
unsigned int i;
struct data_dependence_relation *ddr;
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"=== vect_analyze_data_ref_dependences ===\n");
LOOP_VINFO_DDRS (loop_vinfo)
.create (LOOP_VINFO_DATAREFS (loop_vinfo).length ()
* LOOP_VINFO_DATAREFS (loop_vinfo).length ());
LOOP_VINFO_NO_DATA_DEPENDENCIES (loop_vinfo) = true;
if (!compute_all_dependences (LOOP_VINFO_DATAREFS (loop_vinfo),
&LOOP_VINFO_DDRS (loop_vinfo),
LOOP_VINFO_LOOP_NEST (loop_vinfo), true))
return false;
if (LOOP_VINFO_EPILOGUE_P (loop_vinfo))
*max_vf = LOOP_VINFO_ORIG_MAX_VECT_FACTOR (loop_vinfo);
else
FOR_EACH_VEC_ELT (LOOP_VINFO_DDRS (loop_vinfo), i, ddr)
if (vect_analyze_data_ref_dependence (ddr, loop_vinfo, max_vf))
return false;
return true;
}
static bool
vect_slp_analyze_data_ref_dependence (struct data_dependence_relation *ddr)
{
struct data_reference *dra = DDR_A (ddr);
struct data_reference *drb = DDR_B (ddr);
if (DDR_ARE_DEPENDENT (ddr) == chrec_known)
return false;
if (dra == drb)
return false;
if (DR_IS_READ (dra) && DR_IS_READ (drb))
return false;
if (STMT_VINFO_GROUPED_ACCESS (vinfo_for_stmt (DR_STMT (dra)))
&& (GROUP_FIRST_ELEMENT (vinfo_for_stmt (DR_STMT (dra)))
== GROUP_FIRST_ELEMENT (vinfo_for_stmt (DR_STMT (drb)))))
return false;
if (DDR_ARE_DEPENDENT (ddr) == chrec_dont_know)
{
if  (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"can't determine dependence between ");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM, DR_REF (dra));
dump_printf (MSG_MISSED_OPTIMIZATION,  " and ");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM, DR_REF (drb));
dump_printf (MSG_MISSED_OPTIMIZATION,  "\n");
}
}
else if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"determined dependence between ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (dra));
dump_printf (MSG_NOTE, " and ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (drb));
dump_printf (MSG_NOTE,  "\n");
}
return true;
}
static bool
vect_slp_analyze_node_dependences (slp_instance instance, slp_tree node,
vec<gimple *> stores, gimple *last_store)
{
gimple *last_access = vect_find_last_scalar_stmt_in_slp (node);
for (unsigned k = 0; k < SLP_INSTANCE_GROUP_SIZE (instance); ++k)
{
gimple *access = SLP_TREE_SCALAR_STMTS (node)[k];
if (access == last_access)
continue;
data_reference *dr_a = STMT_VINFO_DATA_REF (vinfo_for_stmt (access));
for (gimple_stmt_iterator gsi = gsi_for_stmt (access);
gsi_stmt (gsi) != last_access; gsi_next (&gsi))
{
gimple *stmt = gsi_stmt (gsi);
if (! gimple_vuse (stmt)
|| (DR_IS_READ (dr_a) && ! gimple_vdef (stmt)))
continue;
data_reference *dr_b = STMT_VINFO_DATA_REF (vinfo_for_stmt (stmt));
if (!dr_b)
return false;
bool dependent = false;
if (gimple_visited_p (stmt))
{
if (stmt != last_store)
continue;
unsigned i;
gimple *store;
FOR_EACH_VEC_ELT (stores, i, store)
{
data_reference *store_dr
= STMT_VINFO_DATA_REF (vinfo_for_stmt (store));
ddr_p ddr = initialize_data_dependence_relation
(dr_a, store_dr, vNULL);
dependent = vect_slp_analyze_data_ref_dependence (ddr);
free_dependence_relation (ddr);
if (dependent)
break;
}
}
else
{
ddr_p ddr = initialize_data_dependence_relation (dr_a,
dr_b, vNULL);
dependent = vect_slp_analyze_data_ref_dependence (ddr);
free_dependence_relation (ddr);
}
if (dependent)
return false;
}
}
return true;
}
bool
vect_slp_analyze_instance_dependence (slp_instance instance)
{
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"=== vect_slp_analyze_instance_dependence ===\n");
slp_tree store = SLP_INSTANCE_TREE (instance);
if (! STMT_VINFO_DATA_REF (vinfo_for_stmt (SLP_TREE_SCALAR_STMTS (store)[0])))
store = NULL;
gimple *last_store = NULL;
if (store)
{
if (! vect_slp_analyze_node_dependences (instance, store, vNULL, NULL))
return false;
last_store = vect_find_last_scalar_stmt_in_slp (store);
for (unsigned k = 0; k < SLP_INSTANCE_GROUP_SIZE (instance); ++k)
gimple_set_visited (SLP_TREE_SCALAR_STMTS (store)[k], true);
}
bool res = true;
slp_tree load;
unsigned int i;
FOR_EACH_VEC_ELT (SLP_INSTANCE_LOADS (instance), i, load)
if (! vect_slp_analyze_node_dependences (instance, load,
store
? SLP_TREE_SCALAR_STMTS (store)
: vNULL, last_store))
{
res = false;
break;
}
if (store)
for (unsigned k = 0; k < SLP_INSTANCE_GROUP_SIZE (instance); ++k)
gimple_set_visited (SLP_TREE_SCALAR_STMTS (store)[k], false);
return res;
}
static void
vect_record_base_alignment (vec_info *vinfo, gimple *stmt,
innermost_loop_behavior *drb)
{
bool existed;
innermost_loop_behavior *&entry
= vinfo->base_alignments.get_or_insert (drb->base_address, &existed);
if (!existed || entry->base_alignment < drb->base_alignment)
{
entry = drb;
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"recording new base alignment for ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, drb->base_address);
dump_printf (MSG_NOTE, "\n");
dump_printf_loc (MSG_NOTE, vect_location,
"  alignment:    %d\n", drb->base_alignment);
dump_printf_loc (MSG_NOTE, vect_location,
"  misalignment: %d\n", drb->base_misalignment);
dump_printf_loc (MSG_NOTE, vect_location,
"  based on:     ");
dump_gimple_stmt (MSG_NOTE, TDF_SLIM, stmt, 0);
}
}
}
void
vect_record_base_alignments (vec_info *vinfo)
{
loop_vec_info loop_vinfo = dyn_cast <loop_vec_info> (vinfo);
struct loop *loop = loop_vinfo ? LOOP_VINFO_LOOP (loop_vinfo) : NULL;
data_reference *dr;
unsigned int i;
FOR_EACH_VEC_ELT (vinfo->datarefs, i, dr)
if (!DR_IS_CONDITIONAL_IN_STMT (dr))
{
gimple *stmt = DR_STMT (dr);
vect_record_base_alignment (vinfo, stmt, &DR_INNERMOST (dr));
if (loop && nested_in_vect_loop_p (loop, stmt))
{
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
vect_record_base_alignment
(vinfo, stmt, &STMT_VINFO_DR_WRT_VEC_LOOP (stmt_info));
}
}
}
static unsigned int
vect_calculate_target_alignment (struct data_reference *dr)
{
gimple *stmt = DR_STMT (dr);
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
tree vectype = STMT_VINFO_VECTYPE (stmt_info);
return targetm.vectorize.preferred_vector_alignment (vectype);
}
bool
vect_compute_data_ref_alignment (struct data_reference *dr)
{
gimple *stmt = DR_STMT (dr);
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
vec_base_alignments *base_alignments = &stmt_info->vinfo->base_alignments;
loop_vec_info loop_vinfo = STMT_VINFO_LOOP_VINFO (stmt_info);
struct loop *loop = NULL;
tree ref = DR_REF (dr);
tree vectype = STMT_VINFO_VECTYPE (stmt_info);
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"vect_compute_data_ref_alignment:\n");
if (loop_vinfo)
loop = LOOP_VINFO_LOOP (loop_vinfo);
SET_DR_MISALIGNMENT (dr, DR_MISALIGNMENT_UNKNOWN);
innermost_loop_behavior *drb = vect_dr_behavior (dr);
bool step_preserves_misalignment_p;
unsigned HOST_WIDE_INT vector_alignment
= vect_calculate_target_alignment (dr) / BITS_PER_UNIT;
DR_TARGET_ALIGNMENT (dr) = vector_alignment;
if (!loop)
{
gcc_assert (integer_zerop (drb->step));
step_preserves_misalignment_p = true;
}
else if (nested_in_vect_loop_p (loop, stmt))
{
step_preserves_misalignment_p
= (DR_STEP_ALIGNMENT (dr) % vector_alignment) == 0;
if (dump_enabled_p ())
{
if (step_preserves_misalignment_p)
dump_printf_loc (MSG_NOTE, vect_location,
"inner step divides the vector alignment.\n");
else
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"inner step doesn't divide the vector"
" alignment.\n");
}
}
else
{
poly_uint64 vf = LOOP_VINFO_VECT_FACTOR (loop_vinfo);
step_preserves_misalignment_p
= multiple_p (DR_STEP_ALIGNMENT (dr) * vf, vector_alignment);
if (!step_preserves_misalignment_p && dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"step doesn't divide the vector alignment.\n");
}
unsigned int base_alignment = drb->base_alignment;
unsigned int base_misalignment = drb->base_misalignment;
innermost_loop_behavior **entry = base_alignments->get (drb->base_address);
if (entry && base_alignment < (*entry)->base_alignment)
{
base_alignment = (*entry)->base_alignment;
base_misalignment = (*entry)->base_misalignment;
}
if (drb->offset_alignment < vector_alignment
|| !step_preserves_misalignment_p
|| TREE_CODE (drb->step) != INTEGER_CST)
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"Unknown alignment for access: ");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM, ref);
dump_printf (MSG_MISSED_OPTIMIZATION, "\n");
}
return true;
}
if (base_alignment < vector_alignment)
{
unsigned int max_alignment;
tree base = get_base_for_alignment (drb->base_address, &max_alignment);
if (max_alignment < vector_alignment
|| !vect_can_force_dr_alignment_p (base,
vector_alignment * BITS_PER_UNIT))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"can't force alignment of ref: ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, ref);
dump_printf (MSG_NOTE, "\n");
}
return true;
}
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location, "force alignment of ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, ref);
dump_printf (MSG_NOTE, "\n");
}
DR_VECT_AUX (dr)->base_decl = base;
DR_VECT_AUX (dr)->base_misaligned = true;
base_misalignment = 0;
}
poly_int64 misalignment
= base_misalignment + wi::to_poly_offset (drb->init).force_shwi ();
if (tree_int_cst_sgn (drb->step) < 0)
misalignment += ((TYPE_VECTOR_SUBPARTS (vectype) - 1)
* TREE_INT_CST_LOW (drb->step));
unsigned int const_misalignment;
if (!known_misalignment (misalignment, vector_alignment,
&const_misalignment))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"Non-constant misalignment for access: ");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM, ref);
dump_printf (MSG_MISSED_OPTIMIZATION, "\n");
}
return true;
}
SET_DR_MISALIGNMENT (dr, const_misalignment);
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"misalign = %d bytes of ref ", DR_MISALIGNMENT (dr));
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM, ref);
dump_printf (MSG_MISSED_OPTIMIZATION, "\n");
}
return true;
}
static void
vect_update_misalignment_for_peel (struct data_reference *dr,
struct data_reference *dr_peel, int npeel)
{
unsigned int i;
vec<dr_p> same_aligned_drs;
struct data_reference *current_dr;
int dr_size = vect_get_scalar_dr_size (dr);
int dr_peel_size = vect_get_scalar_dr_size (dr_peel);
stmt_vec_info stmt_info = vinfo_for_stmt (DR_STMT (dr));
stmt_vec_info peel_stmt_info = vinfo_for_stmt (DR_STMT (dr_peel));
if (STMT_VINFO_GROUPED_ACCESS (stmt_info))
dr_size *= GROUP_SIZE (vinfo_for_stmt (GROUP_FIRST_ELEMENT (stmt_info)));
if (STMT_VINFO_GROUPED_ACCESS (peel_stmt_info))
dr_peel_size *= GROUP_SIZE (peel_stmt_info);
same_aligned_drs
= STMT_VINFO_SAME_ALIGN_REFS (vinfo_for_stmt (DR_STMT (dr_peel)));
FOR_EACH_VEC_ELT (same_aligned_drs, i, current_dr)
{
if (current_dr != dr)
continue;
gcc_assert (!known_alignment_for_access_p (dr)
|| !known_alignment_for_access_p (dr_peel)
|| (DR_MISALIGNMENT (dr) / dr_size
== DR_MISALIGNMENT (dr_peel) / dr_peel_size));
SET_DR_MISALIGNMENT (dr, 0);
return;
}
if (known_alignment_for_access_p (dr)
&& known_alignment_for_access_p (dr_peel))
{
bool negative = tree_int_cst_compare (DR_STEP (dr), size_zero_node) < 0;
int misal = DR_MISALIGNMENT (dr);
misal += negative ? -npeel * dr_size : npeel * dr_size;
misal &= DR_TARGET_ALIGNMENT (dr) - 1;
SET_DR_MISALIGNMENT (dr, misal);
return;
}
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location, "Setting misalignment " \
"to unknown (-1).\n");
SET_DR_MISALIGNMENT (dr, DR_MISALIGNMENT_UNKNOWN);
}
static bool
verify_data_ref_alignment (data_reference_p dr)
{
enum dr_alignment_support supportable_dr_alignment
= vect_supportable_dr_alignment (dr, false);
if (!supportable_dr_alignment)
{
if (dump_enabled_p ())
{
if (DR_IS_READ (dr))
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: unsupported unaligned load.");
else
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: unsupported unaligned "
"store.");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_SLIM,
DR_REF (dr));
dump_printf (MSG_MISSED_OPTIMIZATION, "\n");
}
return false;
}
if (supportable_dr_alignment != dr_aligned && dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"Vectorizing an unaligned access.\n");
return true;
}
bool
vect_verify_datarefs_alignment (loop_vec_info vinfo)
{
vec<data_reference_p> datarefs = vinfo->datarefs;
struct data_reference *dr;
unsigned int i;
FOR_EACH_VEC_ELT (datarefs, i, dr)
{
gimple *stmt = DR_STMT (dr);
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
if (!STMT_VINFO_RELEVANT_P (stmt_info))
continue;
if (STMT_VINFO_GROUPED_ACCESS (stmt_info)
&& GROUP_FIRST_ELEMENT (stmt_info) != stmt)
continue;
if (STMT_VINFO_STRIDED_P (stmt_info)
&& !STMT_VINFO_GROUPED_ACCESS (stmt_info))
continue;
if (! verify_data_ref_alignment (dr))
return false;
}
return true;
}
static bool
not_size_aligned (tree exp)
{
if (!tree_fits_uhwi_p (TYPE_SIZE (TREE_TYPE (exp))))
return true;
return (tree_to_uhwi (TYPE_SIZE (TREE_TYPE (exp)))
> get_object_alignment (exp));
}
static bool
vector_alignment_reachable_p (struct data_reference *dr)
{
gimple *stmt = DR_STMT (dr);
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
tree vectype = STMT_VINFO_VECTYPE (stmt_info);
if (STMT_VINFO_GROUPED_ACCESS (stmt_info))
{
int elem_size, mis_in_elements;
if (!known_alignment_for_access_p (dr))
return false;
poly_uint64 nelements = TYPE_VECTOR_SUBPARTS (vectype);
poly_uint64 vector_size = GET_MODE_SIZE (TYPE_MODE (vectype));
elem_size = vector_element_size (vector_size, nelements);
mis_in_elements = DR_MISALIGNMENT (dr) / elem_size;
if (!multiple_p (nelements - mis_in_elements, GROUP_SIZE (stmt_info)))
return false;
}
if (known_alignment_for_access_p (dr) && !aligned_access_p (dr))
{
HOST_WIDE_INT elmsize =
int_cst_value (TYPE_SIZE_UNIT (TREE_TYPE (vectype)));
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"data size =" HOST_WIDE_INT_PRINT_DEC, elmsize);
dump_printf (MSG_NOTE,
". misalignment = %d.\n", DR_MISALIGNMENT (dr));
}
if (DR_MISALIGNMENT (dr) % elmsize)
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"data size does not divide the misalignment.\n");
return false;
}
}
if (!known_alignment_for_access_p (dr))
{
tree type = TREE_TYPE (DR_REF (dr));
bool is_packed = not_size_aligned (DR_REF (dr));
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"Unknown misalignment, %snaturally aligned\n",
is_packed ? "not " : "");
return targetm.vectorize.vector_alignment_reachable (type, is_packed);
}
return true;
}
static void
vect_get_data_access_cost (struct data_reference *dr,
unsigned int *inside_cost,
unsigned int *outside_cost,
stmt_vector_for_cost *body_cost_vec)
{
gimple *stmt = DR_STMT (dr);
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
loop_vec_info loop_vinfo = STMT_VINFO_LOOP_VINFO (stmt_info);
int ncopies;
if (PURE_SLP_STMT (stmt_info))
ncopies = 1;
else
ncopies = vect_get_num_copies (loop_vinfo, STMT_VINFO_VECTYPE (stmt_info));
if (DR_IS_READ (dr))
vect_get_load_cost (dr, ncopies, true, inside_cost, outside_cost,
NULL, body_cost_vec, false);
else
vect_get_store_cost (dr, ncopies, inside_cost, body_cost_vec);
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"vect_get_data_access_cost: inside_cost = %d, "
"outside_cost = %d.\n", *inside_cost, *outside_cost);
}
typedef struct _vect_peel_info
{
struct data_reference *dr;
int npeel;
unsigned int count;
} *vect_peel_info;
typedef struct _vect_peel_extended_info
{
struct _vect_peel_info peel_info;
unsigned int inside_cost;
unsigned int outside_cost;
} *vect_peel_extended_info;
struct peel_info_hasher : free_ptr_hash <_vect_peel_info>
{
static inline hashval_t hash (const _vect_peel_info *);
static inline bool equal (const _vect_peel_info *, const _vect_peel_info *);
};
inline hashval_t
peel_info_hasher::hash (const _vect_peel_info *peel_info)
{
return (hashval_t) peel_info->npeel;
}
inline bool
peel_info_hasher::equal (const _vect_peel_info *a, const _vect_peel_info *b)
{
return (a->npeel == b->npeel);
}
static void
vect_peeling_hash_insert (hash_table<peel_info_hasher> *peeling_htab,
loop_vec_info loop_vinfo, struct data_reference *dr,
int npeel)
{
struct _vect_peel_info elem, *slot;
_vect_peel_info **new_slot;
bool supportable_dr_alignment = vect_supportable_dr_alignment (dr, true);
elem.npeel = npeel;
slot = peeling_htab->find (&elem);
if (slot)
slot->count++;
else
{
slot = XNEW (struct _vect_peel_info);
slot->npeel = npeel;
slot->dr = dr;
slot->count = 1;
new_slot = peeling_htab->find_slot (slot, INSERT);
*new_slot = slot;
}
if (!supportable_dr_alignment
&& unlimited_cost_model (LOOP_VINFO_LOOP (loop_vinfo)))
slot->count += VECT_MAX_COST;
}
int
vect_peeling_hash_get_most_frequent (_vect_peel_info **slot,
_vect_peel_extended_info *max)
{
vect_peel_info elem = *slot;
if (elem->count > max->peel_info.count
|| (elem->count == max->peel_info.count
&& max->peel_info.npeel > elem->npeel))
{
max->peel_info.npeel = elem->npeel;
max->peel_info.count = elem->count;
max->peel_info.dr = elem->dr;
}
return 1;
}
static void
vect_get_peeling_costs_all_drs (vec<data_reference_p> datarefs,
struct data_reference *dr0,
unsigned int *inside_cost,
unsigned int *outside_cost,
stmt_vector_for_cost *body_cost_vec,
unsigned int npeel,
bool unknown_misalignment)
{
unsigned i;
data_reference *dr;
FOR_EACH_VEC_ELT (datarefs, i, dr)
{
gimple *stmt = DR_STMT (dr);
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
if (!STMT_VINFO_RELEVANT_P (stmt_info))
continue;
if (STMT_VINFO_GROUPED_ACCESS (stmt_info)
&& GROUP_FIRST_ELEMENT (stmt_info) != stmt)
continue;
if (STMT_VINFO_STRIDED_P (stmt_info)
&& !STMT_VINFO_GROUPED_ACCESS (stmt_info))
continue;
int save_misalignment;
save_misalignment = DR_MISALIGNMENT (dr);
if (npeel == 0)
;
else if (unknown_misalignment && dr == dr0)
SET_DR_MISALIGNMENT (dr, 0);
else
vect_update_misalignment_for_peel (dr, dr0, npeel);
vect_get_data_access_cost (dr, inside_cost, outside_cost,
body_cost_vec);
SET_DR_MISALIGNMENT (dr, save_misalignment);
}
}
int
vect_peeling_hash_get_lowest_cost (_vect_peel_info **slot,
_vect_peel_extended_info *min)
{
vect_peel_info elem = *slot;
int dummy;
unsigned int inside_cost = 0, outside_cost = 0;
gimple *stmt = DR_STMT (elem->dr);
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
loop_vec_info loop_vinfo = STMT_VINFO_LOOP_VINFO (stmt_info);
stmt_vector_for_cost prologue_cost_vec, body_cost_vec,
epilogue_cost_vec;
prologue_cost_vec.create (2);
body_cost_vec.create (2);
epilogue_cost_vec.create (2);
vect_get_peeling_costs_all_drs (LOOP_VINFO_DATAREFS (loop_vinfo),
elem->dr, &inside_cost, &outside_cost,
&body_cost_vec, elem->npeel, false);
body_cost_vec.release ();
outside_cost += vect_get_known_peeling_cost
(loop_vinfo, elem->npeel, &dummy,
&LOOP_VINFO_SCALAR_ITERATION_COST (loop_vinfo),
&prologue_cost_vec, &epilogue_cost_vec);
prologue_cost_vec.release ();
epilogue_cost_vec.release ();
if (inside_cost < min->inside_cost
|| (inside_cost == min->inside_cost
&& outside_cost < min->outside_cost))
{
min->inside_cost = inside_cost;
min->outside_cost = outside_cost;
min->peel_info.dr = elem->dr;
min->peel_info.npeel = elem->npeel;
min->peel_info.count = elem->count;
}
return 1;
}
static struct _vect_peel_extended_info
vect_peeling_hash_choose_best_peeling (hash_table<peel_info_hasher> *peeling_htab,
loop_vec_info loop_vinfo)
{
struct _vect_peel_extended_info res;
res.peel_info.dr = NULL;
if (!unlimited_cost_model (LOOP_VINFO_LOOP (loop_vinfo)))
{
res.inside_cost = INT_MAX;
res.outside_cost = INT_MAX;
peeling_htab->traverse <_vect_peel_extended_info *,
vect_peeling_hash_get_lowest_cost> (&res);
}
else
{
res.peel_info.count = 0;
peeling_htab->traverse <_vect_peel_extended_info *,
vect_peeling_hash_get_most_frequent> (&res);
res.inside_cost = 0;
res.outside_cost = 0;
}
return res;
}
static bool
vect_peeling_supportable (loop_vec_info loop_vinfo, struct data_reference *dr0,
unsigned npeel)
{
unsigned i;
struct data_reference *dr = NULL;
vec<data_reference_p> datarefs = LOOP_VINFO_DATAREFS (loop_vinfo);
gimple *stmt;
stmt_vec_info stmt_info;
enum dr_alignment_support supportable_dr_alignment;
FOR_EACH_VEC_ELT (datarefs, i, dr)
{
int save_misalignment;
if (dr == dr0)
continue;
stmt = DR_STMT (dr);
stmt_info = vinfo_for_stmt (stmt);
if (STMT_VINFO_GROUPED_ACCESS (stmt_info)
&& GROUP_FIRST_ELEMENT (stmt_info) != stmt)
continue;
if (STMT_VINFO_STRIDED_P (stmt_info)
&& !STMT_VINFO_GROUPED_ACCESS (stmt_info))
continue;
save_misalignment = DR_MISALIGNMENT (dr);
vect_update_misalignment_for_peel (dr, dr0, npeel);
supportable_dr_alignment = vect_supportable_dr_alignment (dr, false);
SET_DR_MISALIGNMENT (dr, save_misalignment);
if (!supportable_dr_alignment)
return false;
}
return true;
}
bool
vect_enhance_data_refs_alignment (loop_vec_info loop_vinfo)
{
vec<data_reference_p> datarefs = LOOP_VINFO_DATAREFS (loop_vinfo);
struct loop *loop = LOOP_VINFO_LOOP (loop_vinfo);
enum dr_alignment_support supportable_dr_alignment;
struct data_reference *dr0 = NULL, *first_store = NULL;
struct data_reference *dr;
unsigned int i, j;
bool do_peeling = false;
bool do_versioning = false;
bool stat;
gimple *stmt;
stmt_vec_info stmt_info;
unsigned int npeel = 0;
bool one_misalignment_known = false;
bool one_misalignment_unknown = false;
bool one_dr_unsupportable = false;
struct data_reference *unsupportable_dr = NULL;
poly_uint64 vf = LOOP_VINFO_VECT_FACTOR (loop_vinfo);
unsigned possible_npeel_number = 1;
tree vectype;
unsigned int mis, same_align_drs_max = 0;
hash_table<peel_info_hasher> peeling_htab (1);
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"=== vect_enhance_data_refs_alignment ===\n");
LOOP_VINFO_MAY_MISALIGN_STMTS (loop_vinfo).truncate (0);
LOOP_VINFO_PEELING_FOR_ALIGNMENT (loop_vinfo) = 0;
FOR_EACH_VEC_ELT (datarefs, i, dr)
{
stmt = DR_STMT (dr);
stmt_info = vinfo_for_stmt (stmt);
if (!STMT_VINFO_RELEVANT_P (stmt_info))
continue;
if (STMT_VINFO_GROUPED_ACCESS (stmt_info)
&& GROUP_FIRST_ELEMENT (stmt_info) != stmt)
continue;
if (integer_zerop (DR_STEP (dr)))
continue;
if (STMT_VINFO_STRIDED_P (stmt_info)
&& !STMT_VINFO_GROUPED_ACCESS (stmt_info))
continue;
supportable_dr_alignment = vect_supportable_dr_alignment (dr, true);
do_peeling = vector_alignment_reachable_p (dr);
if (do_peeling)
{
if (known_alignment_for_access_p (dr))
{
unsigned int npeel_tmp = 0;
bool negative = tree_int_cst_compare (DR_STEP (dr),
size_zero_node) < 0;
vectype = STMT_VINFO_VECTYPE (stmt_info);
unsigned int target_align = DR_TARGET_ALIGNMENT (dr);
unsigned int dr_size = vect_get_scalar_dr_size (dr);
mis = (negative ? DR_MISALIGNMENT (dr) : -DR_MISALIGNMENT (dr));
if (DR_MISALIGNMENT (dr) != 0)
npeel_tmp = (mis & (target_align - 1)) / dr_size;
if (unlimited_cost_model (LOOP_VINFO_LOOP (loop_vinfo)))
{
poly_uint64 nscalars = (STMT_SLP_TYPE (stmt_info)
? vf * GROUP_SIZE (stmt_info) : vf);
possible_npeel_number
= vect_get_num_vectors (nscalars, vectype);
if (DR_MISALIGNMENT (dr) == 0)
possible_npeel_number++;
}
for (j = 0; j < possible_npeel_number; j++)
{
vect_peeling_hash_insert (&peeling_htab, loop_vinfo,
dr, npeel_tmp);
npeel_tmp += target_align / dr_size;
}
one_misalignment_known = true;
}
else
{
unsigned same_align_drs
= STMT_VINFO_SAME_ALIGN_REFS (stmt_info).length ();
if (!dr0
|| same_align_drs_max < same_align_drs)
{
same_align_drs_max = same_align_drs;
dr0 = dr;
}
else if (same_align_drs_max == same_align_drs)
{
struct loop *ivloop0, *ivloop;
ivloop0 = outermost_invariant_loop_for_expr
(loop, DR_BASE_ADDRESS (dr0));
ivloop = outermost_invariant_loop_for_expr
(loop, DR_BASE_ADDRESS (dr));
if ((ivloop && !ivloop0)
|| (ivloop && ivloop0
&& flow_loop_nested_p (ivloop, ivloop0)))
dr0 = dr;
}
one_misalignment_unknown = true;
if (!supportable_dr_alignment)
{
one_dr_unsupportable = true;
unsupportable_dr = dr;
}
if (!first_store && DR_IS_WRITE (dr))
first_store = dr;
}
}
else
{
if (!aligned_access_p (dr))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"vector alignment may not be reachable\n");
break;
}
}
}
if (!vect_can_advance_ivs_p (loop_vinfo)
|| !slpeel_can_duplicate_loop_p (loop, single_exit (loop))
|| loop->inner)
do_peeling = false;
struct _vect_peel_extended_info peel_for_known_alignment;
struct _vect_peel_extended_info peel_for_unknown_alignment;
struct _vect_peel_extended_info best_peel;
peel_for_unknown_alignment.inside_cost = INT_MAX;
peel_for_unknown_alignment.outside_cost = INT_MAX;
peel_for_unknown_alignment.peel_info.count = 0;
if (do_peeling
&& one_misalignment_unknown)
{
unsigned int load_inside_cost = 0;
unsigned int load_outside_cost = 0;
unsigned int store_inside_cost = 0;
unsigned int store_outside_cost = 0;
unsigned int estimated_npeels = vect_vf_for_cost (loop_vinfo) / 2;
stmt_vector_for_cost dummy;
dummy.create (2);
vect_get_peeling_costs_all_drs (datarefs, dr0,
&load_inside_cost,
&load_outside_cost,
&dummy, estimated_npeels, true);
dummy.release ();
if (first_store)
{
dummy.create (2);
vect_get_peeling_costs_all_drs (datarefs, first_store,
&store_inside_cost,
&store_outside_cost,
&dummy, estimated_npeels, true);
dummy.release ();
}
else
{
store_inside_cost = INT_MAX;
store_outside_cost = INT_MAX;
}
if (load_inside_cost > store_inside_cost
|| (load_inside_cost == store_inside_cost
&& load_outside_cost > store_outside_cost))
{
dr0 = first_store;
peel_for_unknown_alignment.inside_cost = store_inside_cost;
peel_for_unknown_alignment.outside_cost = store_outside_cost;
}
else
{
peel_for_unknown_alignment.inside_cost = load_inside_cost;
peel_for_unknown_alignment.outside_cost = load_outside_cost;
}
stmt_vector_for_cost prologue_cost_vec, epilogue_cost_vec;
prologue_cost_vec.create (2);
epilogue_cost_vec.create (2);
int dummy2;
peel_for_unknown_alignment.outside_cost += vect_get_known_peeling_cost
(loop_vinfo, estimated_npeels, &dummy2,
&LOOP_VINFO_SCALAR_ITERATION_COST (loop_vinfo),
&prologue_cost_vec, &epilogue_cost_vec);
prologue_cost_vec.release ();
epilogue_cost_vec.release ();
peel_for_unknown_alignment.peel_info.count = 1
+ STMT_VINFO_SAME_ALIGN_REFS
(vinfo_for_stmt (DR_STMT (dr0))).length ();
}
peel_for_unknown_alignment.peel_info.npeel = 0;
peel_for_unknown_alignment.peel_info.dr = dr0;
best_peel = peel_for_unknown_alignment;
peel_for_known_alignment.inside_cost = INT_MAX;
peel_for_known_alignment.outside_cost = INT_MAX;
peel_for_known_alignment.peel_info.count = 0;
peel_for_known_alignment.peel_info.dr = NULL;
if (do_peeling && one_misalignment_known)
{
peel_for_known_alignment = vect_peeling_hash_choose_best_peeling
(&peeling_htab, loop_vinfo);
}
if (peel_for_known_alignment.peel_info.dr != NULL
&& peel_for_unknown_alignment.inside_cost
>= peel_for_known_alignment.inside_cost)
{
best_peel = peel_for_known_alignment;
if (best_peel.peel_info.npeel == 0 && !one_dr_unsupportable)
do_peeling = false;
}
if (one_dr_unsupportable)
dr0 = unsupportable_dr;
else if (do_peeling)
{
unsigned nopeel_inside_cost = 0;
unsigned nopeel_outside_cost = 0;
stmt_vector_for_cost dummy;
dummy.create (2);
vect_get_peeling_costs_all_drs (datarefs, NULL, &nopeel_inside_cost,
&nopeel_outside_cost, &dummy, 0, false);
dummy.release ();
stmt_vector_for_cost prologue_cost_vec, epilogue_cost_vec;
prologue_cost_vec.create (2);
epilogue_cost_vec.create (2);
int dummy2;
nopeel_outside_cost += vect_get_known_peeling_cost
(loop_vinfo, 0, &dummy2,
&LOOP_VINFO_SCALAR_ITERATION_COST (loop_vinfo),
&prologue_cost_vec, &epilogue_cost_vec);
prologue_cost_vec.release ();
epilogue_cost_vec.release ();
npeel = best_peel.peel_info.npeel;
dr0 = best_peel.peel_info.dr;
if (nopeel_inside_cost <= best_peel.inside_cost)
do_peeling = false;
}
if (do_peeling)
{
stmt = DR_STMT (dr0);
stmt_info = vinfo_for_stmt (stmt);
vectype = STMT_VINFO_VECTYPE (stmt_info);
if (known_alignment_for_access_p (dr0))
{
bool negative = tree_int_cst_compare (DR_STEP (dr0),
size_zero_node) < 0;
if (!npeel)
{
mis = negative ? DR_MISALIGNMENT (dr0) : -DR_MISALIGNMENT (dr0);
unsigned int target_align = DR_TARGET_ALIGNMENT (dr0);
npeel = ((mis & (target_align - 1))
/ vect_get_scalar_dr_size (dr0));
}
stmt_info = vinfo_for_stmt (DR_STMT (dr0));
if (STMT_VINFO_GROUPED_ACCESS (stmt_info))
npeel /= GROUP_SIZE (stmt_info);
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"Try peeling by %d\n", npeel);
}
if (!vect_peeling_supportable (loop_vinfo, dr0, npeel))
do_peeling = false;
if (do_peeling && known_alignment_for_access_p (dr0) && npeel == 0)
{
stat = vect_verify_datarefs_alignment (loop_vinfo);
if (!stat)
do_peeling = false;
else
return stat;
}
if (do_peeling)
{
unsigned max_allowed_peel
= PARAM_VALUE (PARAM_VECT_MAX_PEELING_FOR_ALIGNMENT);
if (max_allowed_peel != (unsigned)-1)
{
unsigned max_peel = npeel;
if (max_peel == 0)
{
unsigned int target_align = DR_TARGET_ALIGNMENT (dr0);
max_peel = target_align / vect_get_scalar_dr_size (dr0) - 1;
}
if (max_peel > max_allowed_peel)
{
do_peeling = false;
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"Disable peeling, max peels reached: %d\n", max_peel);
}
}
}
if (do_peeling
&& LOOP_VINFO_NITERS_KNOWN_P (loop_vinfo))
{
unsigned int assumed_vf = vect_vf_for_cost (loop_vinfo);
unsigned int max_peel = npeel == 0 ? assumed_vf - 1 : npeel;
if ((unsigned HOST_WIDE_INT) LOOP_VINFO_INT_NITERS (loop_vinfo)
< assumed_vf + max_peel)
do_peeling = false;
}
if (do_peeling)
{
FOR_EACH_VEC_ELT (datarefs, i, dr)
if (dr != dr0)
{
stmt_info = vinfo_for_stmt (DR_STMT (dr));
if (STMT_VINFO_STRIDED_P (stmt_info)
&& !STMT_VINFO_GROUPED_ACCESS (stmt_info))
continue;
vect_update_misalignment_for_peel (dr, dr0, npeel);
}
LOOP_VINFO_UNALIGNED_DR (loop_vinfo) = dr0;
if (npeel)
LOOP_VINFO_PEELING_FOR_ALIGNMENT (loop_vinfo) = npeel;
else
LOOP_VINFO_PEELING_FOR_ALIGNMENT (loop_vinfo)
= DR_MISALIGNMENT (dr0);
SET_DR_MISALIGNMENT (dr0, 0);
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"Alignment of access forced using peeling.\n");
dump_printf_loc (MSG_NOTE, vect_location,
"Peeling for alignment will be applied.\n");
}
stat = vect_verify_datarefs_alignment (loop_vinfo);
gcc_assert (stat);
return stat;
}
}
do_versioning =
optimize_loop_nest_for_speed_p (loop)
&& (!loop->inner); 
if (do_versioning)
{
FOR_EACH_VEC_ELT (datarefs, i, dr)
{
stmt = DR_STMT (dr);
stmt_info = vinfo_for_stmt (stmt);
if (aligned_access_p (dr)
|| (STMT_VINFO_GROUPED_ACCESS (stmt_info)
&& GROUP_FIRST_ELEMENT (stmt_info) != stmt))
continue;
if (STMT_VINFO_STRIDED_P (stmt_info))
{
if (!STMT_VINFO_GROUPED_ACCESS (stmt_info))
continue;
do_versioning = false;
break;
}
supportable_dr_alignment = vect_supportable_dr_alignment (dr, false);
if (!supportable_dr_alignment)
{
gimple *stmt;
int mask;
tree vectype;
if (known_alignment_for_access_p (dr)
|| LOOP_VINFO_MAY_MISALIGN_STMTS (loop_vinfo).length ()
>= (unsigned) PARAM_VALUE (PARAM_VECT_MAX_VERSION_FOR_ALIGNMENT_CHECKS))
{
do_versioning = false;
break;
}
stmt = DR_STMT (dr);
vectype = STMT_VINFO_VECTYPE (vinfo_for_stmt (stmt));
gcc_assert (vectype);
unsigned HOST_WIDE_INT size;
if (!GET_MODE_SIZE (TYPE_MODE (vectype)).is_constant (&size))
{
do_versioning = false;
break;
}
mask = size - 1;
gcc_assert (!LOOP_VINFO_PTR_MASK (loop_vinfo)
|| LOOP_VINFO_PTR_MASK (loop_vinfo) == mask);
LOOP_VINFO_PTR_MASK (loop_vinfo) = mask;
LOOP_VINFO_MAY_MISALIGN_STMTS (loop_vinfo).safe_push (
DR_STMT (dr));
}
}
if (!LOOP_REQUIRES_VERSIONING_FOR_ALIGNMENT (loop_vinfo))
do_versioning = false;
else if (!do_versioning)
LOOP_VINFO_MAY_MISALIGN_STMTS (loop_vinfo).truncate (0);
}
if (do_versioning)
{
vec<gimple *> may_misalign_stmts
= LOOP_VINFO_MAY_MISALIGN_STMTS (loop_vinfo);
gimple *stmt;
FOR_EACH_VEC_ELT (may_misalign_stmts, i, stmt)
{
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
dr = STMT_VINFO_DATA_REF (stmt_info);
SET_DR_MISALIGNMENT (dr, 0);
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"Alignment of access forced using versioning.\n");
}
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"Versioning for alignment will be applied.\n");
gcc_assert (! (do_peeling && do_versioning));
stat = vect_verify_datarefs_alignment (loop_vinfo);
gcc_assert (stat);
return stat;
}
gcc_assert (! (do_peeling || do_versioning));
stat = vect_verify_datarefs_alignment (loop_vinfo);
return stat;
}
static void
vect_find_same_alignment_drs (struct data_dependence_relation *ddr)
{
struct data_reference *dra = DDR_A (ddr);
struct data_reference *drb = DDR_B (ddr);
stmt_vec_info stmtinfo_a = vinfo_for_stmt (DR_STMT (dra));
stmt_vec_info stmtinfo_b = vinfo_for_stmt (DR_STMT (drb));
if (DDR_ARE_DEPENDENT (ddr) == chrec_known)
return;
if (dra == drb)
return;
if (!operand_equal_p (DR_BASE_ADDRESS (dra), DR_BASE_ADDRESS (drb), 0)
|| !operand_equal_p (DR_OFFSET (dra), DR_OFFSET (drb), 0)
|| !operand_equal_p (DR_STEP (dra), DR_STEP (drb), 0))
return;
poly_offset_int diff = (wi::to_poly_offset (DR_INIT (dra))
- wi::to_poly_offset (DR_INIT (drb)));
if (maybe_ne (diff, 0))
{
unsigned int align_a = (vect_calculate_target_alignment (dra)
/ BITS_PER_UNIT);
unsigned int align_b = (vect_calculate_target_alignment (drb)
/ BITS_PER_UNIT);
unsigned int max_align = MAX (align_a, align_b);
if (!multiple_p (diff, max_align))
return;
}
STMT_VINFO_SAME_ALIGN_REFS (stmtinfo_a).safe_push (drb);
STMT_VINFO_SAME_ALIGN_REFS (stmtinfo_b).safe_push (dra);
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"accesses have the same alignment: ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (dra));
dump_printf (MSG_NOTE,  " and ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (drb));
dump_printf (MSG_NOTE, "\n");
}
}
bool
vect_analyze_data_refs_alignment (loop_vec_info vinfo)
{
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"=== vect_analyze_data_refs_alignment ===\n");
vec<ddr_p> ddrs = vinfo->ddrs;
struct data_dependence_relation *ddr;
unsigned int i;
FOR_EACH_VEC_ELT (ddrs, i, ddr)
vect_find_same_alignment_drs (ddr);
vec<data_reference_p> datarefs = vinfo->datarefs;
struct data_reference *dr;
vect_record_base_alignments (vinfo);
FOR_EACH_VEC_ELT (datarefs, i, dr)
{
stmt_vec_info stmt_info = vinfo_for_stmt (DR_STMT (dr));
if (STMT_VINFO_VECTORIZABLE (stmt_info)
&& !vect_compute_data_ref_alignment (dr))
{
if (STMT_VINFO_STRIDED_P (stmt_info)
&& !STMT_VINFO_GROUPED_ACCESS (stmt_info))
continue;
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: can't calculate alignment "
"for data ref.\n");
return false;
}
}
return true;
}
static bool
vect_slp_analyze_and_verify_node_alignment (slp_tree node)
{
gimple *first_stmt = SLP_TREE_SCALAR_STMTS (node)[0];
data_reference_p first_dr = STMT_VINFO_DATA_REF (vinfo_for_stmt (first_stmt));
if (SLP_TREE_LOAD_PERMUTATION (node).exists ())
first_stmt = GROUP_FIRST_ELEMENT (vinfo_for_stmt (first_stmt));
data_reference_p dr = STMT_VINFO_DATA_REF (vinfo_for_stmt (first_stmt));
if (! vect_compute_data_ref_alignment (dr)
|| (dr != first_dr
&& ! vect_compute_data_ref_alignment (first_dr))
|| ! verify_data_ref_alignment (dr))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: bad data alignment in basic "
"block.\n");
return false;
}
return true;
}
bool
vect_slp_analyze_and_verify_instance_alignment (slp_instance instance)
{
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"=== vect_slp_analyze_and_verify_instance_alignment ===\n");
slp_tree node;
unsigned i;
FOR_EACH_VEC_ELT (SLP_INSTANCE_LOADS (instance), i, node)
if (! vect_slp_analyze_and_verify_node_alignment (node))
return false;
node = SLP_INSTANCE_TREE (instance);
if (STMT_VINFO_DATA_REF (vinfo_for_stmt (SLP_TREE_SCALAR_STMTS (node)[0]))
&& ! vect_slp_analyze_and_verify_node_alignment
(SLP_INSTANCE_TREE (instance)))
return false;
return true;
}
static bool
vect_analyze_group_access_1 (struct data_reference *dr)
{
tree step = DR_STEP (dr);
tree scalar_type = TREE_TYPE (DR_REF (dr));
HOST_WIDE_INT type_size = TREE_INT_CST_LOW (TYPE_SIZE_UNIT (scalar_type));
gimple *stmt = DR_STMT (dr);
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
loop_vec_info loop_vinfo = STMT_VINFO_LOOP_VINFO (stmt_info);
bb_vec_info bb_vinfo = STMT_VINFO_BB_VINFO (stmt_info);
HOST_WIDE_INT dr_step = -1;
HOST_WIDE_INT groupsize, last_accessed_element = 1;
bool slp_impossible = false;
if (tree_fits_shwi_p (step))
{
dr_step = tree_to_shwi (step);
if ((dr_step % type_size) != 0)
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"Step ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, step);
dump_printf (MSG_NOTE,
" is not a multiple of the element size for ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (dr));
dump_printf (MSG_NOTE, "\n");
}
return false;
}
groupsize = absu_hwi (dr_step) / type_size;
}
else
groupsize = 0;
if (!GROUP_FIRST_ELEMENT (vinfo_for_stmt (stmt)))
{
if (DR_IS_READ (dr)
&& (dr_step % type_size) == 0
&& groupsize > 0)
{
GROUP_FIRST_ELEMENT (vinfo_for_stmt (stmt)) = stmt;
GROUP_SIZE (vinfo_for_stmt (stmt)) = groupsize;
GROUP_GAP (stmt_info) = groupsize - 1;
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"Detected single element interleaving ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (dr));
dump_printf (MSG_NOTE, " step ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, step);
dump_printf (MSG_NOTE, "\n");
}
return true;
}
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not consecutive access ");
dump_gimple_stmt (MSG_MISSED_OPTIMIZATION, TDF_SLIM, stmt, 0);
}
if (bb_vinfo)
{
STMT_VINFO_VECTORIZABLE (vinfo_for_stmt (DR_STMT (dr))) = false;
return true;
}
dump_printf_loc (MSG_NOTE, vect_location, "using strided accesses\n");
STMT_VINFO_STRIDED_P (stmt_info) = true;
return true;
}
if (GROUP_FIRST_ELEMENT (vinfo_for_stmt (stmt)) == stmt)
{
gimple *next = GROUP_NEXT_ELEMENT (vinfo_for_stmt (stmt));
struct data_reference *data_ref = dr;
unsigned int count = 1;
tree prev_init = DR_INIT (data_ref);
gimple *prev = stmt;
HOST_WIDE_INT diff, gaps = 0;
while (next)
{
if (!tree_int_cst_compare (DR_INIT (data_ref),
DR_INIT (STMT_VINFO_DATA_REF (
vinfo_for_stmt (next)))))
{
if (DR_IS_WRITE (data_ref))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"Two store stmts share the same dr.\n");
return false;
}
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"Two or more load stmts share the same dr.\n");
GROUP_SAME_DR_STMT (vinfo_for_stmt (next)) = prev;
prev = next;
next = GROUP_NEXT_ELEMENT (vinfo_for_stmt (next));
continue;
}
prev = next;
data_ref = STMT_VINFO_DATA_REF (vinfo_for_stmt (next));
gcc_checking_assert (operand_equal_p (DR_STEP (data_ref), step, 0));
diff = (TREE_INT_CST_LOW (DR_INIT (data_ref))
- TREE_INT_CST_LOW (prev_init)) / type_size;
if (diff != 1)
{
slp_impossible = true;
if (DR_IS_WRITE (data_ref))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"interleaved store with gaps\n");
return false;
}
gaps += diff - 1;
}
last_accessed_element += diff;
GROUP_GAP (vinfo_for_stmt (next)) = diff;
prev_init = DR_INIT (data_ref);
next = GROUP_NEXT_ELEMENT (vinfo_for_stmt (next));
count++;
}
if (groupsize == 0)
groupsize = count + gaps;
if (groupsize > 4096)
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"group is too large\n");
return false;
}
if (groupsize != count
&& !DR_IS_READ (dr))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"interleaved store with gaps\n");
return false;
}
GROUP_GAP (vinfo_for_stmt (stmt)) = groupsize - last_accessed_element;
GROUP_SIZE (vinfo_for_stmt (stmt)) = groupsize;
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"Detected interleaving ");
if (DR_IS_READ (dr))
dump_printf (MSG_NOTE, "load ");
else
dump_printf (MSG_NOTE, "store ");
dump_printf (MSG_NOTE, "of size %u starting with ",
(unsigned)groupsize);
dump_gimple_stmt (MSG_NOTE, TDF_SLIM, stmt, 0);
if (GROUP_GAP (vinfo_for_stmt (stmt)) != 0)
dump_printf_loc (MSG_NOTE, vect_location,
"There is a gap of %u elements after the group\n",
GROUP_GAP (vinfo_for_stmt (stmt)));
}
if (DR_IS_WRITE (dr) && !slp_impossible)
{
if (loop_vinfo)
LOOP_VINFO_GROUPED_STORES (loop_vinfo).safe_push (stmt);
if (bb_vinfo)
BB_VINFO_GROUPED_STORES (bb_vinfo).safe_push (stmt);
}
}
return true;
}
static bool
vect_analyze_group_access (struct data_reference *dr)
{
if (!vect_analyze_group_access_1 (dr))
{
gimple *next;
gimple *stmt = GROUP_FIRST_ELEMENT (vinfo_for_stmt (DR_STMT (dr)));
while (stmt)
{
stmt_vec_info vinfo = vinfo_for_stmt (stmt);
next = GROUP_NEXT_ELEMENT (vinfo);
GROUP_FIRST_ELEMENT (vinfo) = NULL;
GROUP_NEXT_ELEMENT (vinfo) = NULL;
stmt = next;
}
return false;
}
return true;
}
static bool
vect_analyze_data_ref_access (struct data_reference *dr)
{
tree step = DR_STEP (dr);
tree scalar_type = TREE_TYPE (DR_REF (dr));
gimple *stmt = DR_STMT (dr);
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
loop_vec_info loop_vinfo = STMT_VINFO_LOOP_VINFO (stmt_info);
struct loop *loop = NULL;
if (STMT_VINFO_GATHER_SCATTER_P (stmt_info))
return true;
if (loop_vinfo)
loop = LOOP_VINFO_LOOP (loop_vinfo);
if (loop_vinfo && !step)
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"bad data-ref access in loop\n");
return false;
}
if (loop_vinfo && integer_zerop (step))
{
GROUP_FIRST_ELEMENT (vinfo_for_stmt (stmt)) = NULL;
if (!nested_in_vect_loop_p (loop, stmt))
return DR_IS_READ (dr);
if (loop->safelen < 2)
{
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"zero step in inner loop of nest\n");
return false;
}
}
if (loop && nested_in_vect_loop_p (loop, stmt))
{
GROUP_FIRST_ELEMENT (vinfo_for_stmt (stmt)) = NULL;
step = STMT_VINFO_DR_STEP (stmt_info);
if (integer_zerop (step))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"zero step in outer loop.\n");
return DR_IS_READ (dr);
}
}
if (TREE_CODE (step) == INTEGER_CST)
{
HOST_WIDE_INT dr_step = TREE_INT_CST_LOW (step);
if (!tree_int_cst_compare (step, TYPE_SIZE_UNIT (scalar_type))
|| (dr_step < 0
&& !compare_tree_int (TYPE_SIZE_UNIT (scalar_type), -dr_step)))
{
GROUP_FIRST_ELEMENT (vinfo_for_stmt (stmt)) = NULL;
return true;
}
}
if (loop && nested_in_vect_loop_p (loop, stmt))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"grouped access in outer loop.\n");
return false;
}
if (TREE_CODE (step) != INTEGER_CST)
return (STMT_VINFO_STRIDED_P (stmt_info)
&& (!STMT_VINFO_GROUPED_ACCESS (stmt_info)
|| vect_analyze_group_access (dr)));
return vect_analyze_group_access (dr);
}
static int
dr_group_sort_cmp (const void *dra_, const void *drb_)
{
data_reference_p dra = *(data_reference_p *)const_cast<void *>(dra_);
data_reference_p drb = *(data_reference_p *)const_cast<void *>(drb_);
int cmp;
if (dra == drb)
return 0;
loop_p loopa = gimple_bb (DR_STMT (dra))->loop_father;
loop_p loopb = gimple_bb (DR_STMT (drb))->loop_father;
if (loopa != loopb)
return loopa->num < loopb->num ? -1 : 1;
cmp = data_ref_compare_tree (DR_BASE_ADDRESS (dra),
DR_BASE_ADDRESS (drb));
if (cmp != 0)
return cmp;
cmp = data_ref_compare_tree (DR_OFFSET (dra), DR_OFFSET (drb));
if (cmp != 0)
return cmp;
if (DR_IS_READ (dra) != DR_IS_READ (drb))
return DR_IS_READ (dra) ? -1 : 1;
cmp = data_ref_compare_tree (TYPE_SIZE_UNIT (TREE_TYPE (DR_REF (dra))),
TYPE_SIZE_UNIT (TREE_TYPE (DR_REF (drb))));
if (cmp != 0)
return cmp;
cmp = data_ref_compare_tree (DR_STEP (dra), DR_STEP (drb));
if (cmp != 0)
return cmp;
cmp = data_ref_compare_tree (DR_INIT (dra), DR_INIT (drb));
if (cmp == 0)
return gimple_uid (DR_STMT (dra)) < gimple_uid (DR_STMT (drb)) ? -1 : 1;
return cmp;
}
static tree
strip_conversion (tree op)
{
if (TREE_CODE (op) != SSA_NAME)
return NULL_TREE;
gimple *stmt = SSA_NAME_DEF_STMT (op);
if (!is_gimple_assign (stmt)
|| !CONVERT_EXPR_CODE_P (gimple_assign_rhs_code (stmt)))
return NULL_TREE;
return gimple_assign_rhs1 (stmt);
}
static bool
can_group_stmts_p (gimple *stmt1, gimple *stmt2)
{
if (gimple_assign_single_p (stmt1))
return gimple_assign_single_p (stmt2);
if (is_gimple_call (stmt1) && gimple_call_internal_p (stmt1))
{
if (!is_gimple_call (stmt2) || !gimple_call_internal_p (stmt2))
return false;
internal_fn ifn = gimple_call_internal_fn (stmt1);
if (ifn != IFN_MASK_LOAD && ifn != IFN_MASK_STORE)
return false;
if (ifn != gimple_call_internal_fn (stmt2))
return false;
tree mask1 = gimple_call_arg (stmt1, 2);
tree mask2 = gimple_call_arg (stmt2, 2);
if (!operand_equal_p (mask1, mask2, 0))
{
mask1 = strip_conversion (mask1);
if (!mask1)
return false;
mask2 = strip_conversion (mask2);
if (!mask2)
return false;
if (!operand_equal_p (mask1, mask2, 0))
return false;
}
return true;
}
return false;
}
bool
vect_analyze_data_ref_accesses (vec_info *vinfo)
{
unsigned int i;
vec<data_reference_p> datarefs = vinfo->datarefs;
struct data_reference *dr;
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"=== vect_analyze_data_ref_accesses ===\n");
if (datarefs.is_empty ())
return true;
vec<data_reference_p> datarefs_copy = datarefs.copy ();
datarefs_copy.qsort (dr_group_sort_cmp);
for (i = 0; i < datarefs_copy.length () - 1;)
{
data_reference_p dra = datarefs_copy[i];
stmt_vec_info stmtinfo_a = vinfo_for_stmt (DR_STMT (dra));
stmt_vec_info lastinfo = NULL;
if (!STMT_VINFO_VECTORIZABLE (stmtinfo_a)
|| STMT_VINFO_GATHER_SCATTER_P (stmtinfo_a))
{
++i;
continue;
}
for (i = i + 1; i < datarefs_copy.length (); ++i)
{
data_reference_p drb = datarefs_copy[i];
stmt_vec_info stmtinfo_b = vinfo_for_stmt (DR_STMT (drb));
if (!STMT_VINFO_VECTORIZABLE (stmtinfo_b)
|| STMT_VINFO_GATHER_SCATTER_P (stmtinfo_b))
break;
if (gimple_bb (DR_STMT (dra))->loop_father
!= gimple_bb (DR_STMT (drb))->loop_father)
break;
if (DR_IS_READ (dra) != DR_IS_READ (drb)
|| data_ref_compare_tree (DR_BASE_ADDRESS (dra),
DR_BASE_ADDRESS (drb)) != 0
|| data_ref_compare_tree (DR_OFFSET (dra), DR_OFFSET (drb)) != 0
|| !can_group_stmts_p (DR_STMT (dra), DR_STMT (drb)))
break;
tree sza = TYPE_SIZE_UNIT (TREE_TYPE (DR_REF (dra)));
tree szb = TYPE_SIZE_UNIT (TREE_TYPE (DR_REF (drb)));
if (!tree_fits_uhwi_p (sza)
|| !tree_fits_uhwi_p (szb)
|| !tree_int_cst_equal (sza, szb))
break;
if (data_ref_compare_tree (DR_STEP (dra), DR_STEP (drb)) != 0)
break;
if (!types_compatible_p (TREE_TYPE (DR_REF (dra)),
TREE_TYPE (DR_REF (drb))))
break;
if (TREE_CODE (DR_INIT (dra)) != INTEGER_CST
|| TREE_CODE (DR_INIT (drb)) != INTEGER_CST)
break;
HOST_WIDE_INT init_a = TREE_INT_CST_LOW (DR_INIT (dra));
HOST_WIDE_INT init_b = TREE_INT_CST_LOW (DR_INIT (drb));
HOST_WIDE_INT init_prev
= TREE_INT_CST_LOW (DR_INIT (datarefs_copy[i-1]));
gcc_assert (init_a <= init_b
&& init_a <= init_prev
&& init_prev <= init_b);
if (init_b == init_prev)
{
gcc_assert (gimple_uid (DR_STMT (datarefs_copy[i-1]))
< gimple_uid (DR_STMT (drb)));
continue;
}
HOST_WIDE_INT type_size_a = tree_to_uhwi (sza);
if (type_size_a == 0
|| (init_b - init_a) % type_size_a != 0)
break;
if (!DR_IS_READ (dra) && init_b - init_prev != type_size_a)
break;
if (tree_fits_shwi_p (DR_STEP (dra)))
{
HOST_WIDE_INT step = tree_to_shwi (DR_STEP (dra));
if (step != 0 && step <= (init_b - init_a))
break;
}
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"Detected interleaving ");
if (DR_IS_READ (dra))
dump_printf (MSG_NOTE, "load ");
else
dump_printf (MSG_NOTE, "store ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (dra));
dump_printf (MSG_NOTE,  " and ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (drb));
dump_printf (MSG_NOTE, "\n");
}
if (!GROUP_FIRST_ELEMENT (stmtinfo_a))
{
GROUP_FIRST_ELEMENT (stmtinfo_a) = DR_STMT (dra);
lastinfo = stmtinfo_a;
}
GROUP_FIRST_ELEMENT (stmtinfo_b) = DR_STMT (dra);
GROUP_NEXT_ELEMENT (lastinfo) = DR_STMT (drb);
lastinfo = stmtinfo_b;
}
}
FOR_EACH_VEC_ELT (datarefs_copy, i, dr)
if (STMT_VINFO_VECTORIZABLE (vinfo_for_stmt (DR_STMT (dr))) 
&& !vect_analyze_data_ref_access (dr))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: complicated access pattern.\n");
if (is_a <bb_vec_info> (vinfo))
{
STMT_VINFO_VECTORIZABLE (vinfo_for_stmt (DR_STMT (dr))) = false;
continue;
}
else
{
datarefs_copy.release ();
return false;
}
}
datarefs_copy.release ();
return true;
}
static tree
vect_vfa_segment_size (struct data_reference *dr, tree length_factor)
{
length_factor = size_binop (MINUS_EXPR,
fold_convert (sizetype, length_factor),
size_one_node);
return size_binop (MULT_EXPR, fold_convert (sizetype, DR_STEP (dr)),
length_factor);
}
static unsigned HOST_WIDE_INT
vect_vfa_access_size (data_reference *dr)
{
stmt_vec_info stmt_vinfo = vinfo_for_stmt (DR_STMT (dr));
tree ref_type = TREE_TYPE (DR_REF (dr));
unsigned HOST_WIDE_INT ref_size = tree_to_uhwi (TYPE_SIZE_UNIT (ref_type));
unsigned HOST_WIDE_INT access_size = ref_size;
if (GROUP_FIRST_ELEMENT (stmt_vinfo))
{
gcc_assert (GROUP_FIRST_ELEMENT (stmt_vinfo) == DR_STMT (dr));
access_size *= GROUP_SIZE (stmt_vinfo) - GROUP_GAP (stmt_vinfo);
}
if (STMT_VINFO_VEC_STMT (stmt_vinfo)
&& (vect_supportable_dr_alignment (dr, false)
== dr_explicit_realign_optimized))
{
tree vectype = STMT_VINFO_VECTYPE (stmt_vinfo);
access_size += tree_to_uhwi (TYPE_SIZE_UNIT (vectype)) - ref_size;
}
return access_size;
}
static unsigned int
vect_vfa_align (const data_reference *dr)
{
return TYPE_ALIGN_UNIT (TREE_TYPE (DR_REF (dr)));
}
static int
vect_compile_time_alias (struct data_reference *a, struct data_reference *b,
tree segment_length_a, tree segment_length_b,
unsigned HOST_WIDE_INT access_size_a,
unsigned HOST_WIDE_INT access_size_b)
{
poly_offset_int offset_a = wi::to_poly_offset (DR_INIT (a));
poly_offset_int offset_b = wi::to_poly_offset (DR_INIT (b));
poly_uint64 const_length_a;
poly_uint64 const_length_b;
if (tree_int_cst_compare (DR_STEP (a), size_zero_node) < 0)
{
const_length_a = (-wi::to_poly_wide (segment_length_a)).force_uhwi ();
offset_a = (offset_a + access_size_a) - const_length_a;
}
else
const_length_a = tree_to_poly_uint64 (segment_length_a);
if (tree_int_cst_compare (DR_STEP (b), size_zero_node) < 0)
{
const_length_b = (-wi::to_poly_wide (segment_length_b)).force_uhwi ();
offset_b = (offset_b + access_size_b) - const_length_b;
}
else
const_length_b = tree_to_poly_uint64 (segment_length_b);
const_length_a += access_size_a;
const_length_b += access_size_b;
if (ranges_known_overlap_p (offset_a, const_length_a,
offset_b, const_length_b))
return 1;
if (!ranges_maybe_overlap_p (offset_a, const_length_a,
offset_b, const_length_b))
return 0;
return -1;
}
static bool
dependence_distance_ge_vf (data_dependence_relation *ddr,
unsigned int loop_depth, poly_uint64 vf)
{
if (DDR_ARE_DEPENDENT (ddr) != NULL_TREE
|| DDR_NUM_DIST_VECTS (ddr) == 0)
return false;
gcc_checking_assert (DDR_COULD_BE_INDEPENDENT_P (ddr));
unsigned int i;
lambda_vector dist_v;
FOR_EACH_VEC_ELT (DDR_DIST_VECTS (ddr), i, dist_v)
{
HOST_WIDE_INT dist = dist_v[loop_depth];
if (dist != 0
&& !(dist > 0 && DDR_REVERSED_P (ddr))
&& maybe_lt ((unsigned HOST_WIDE_INT) abs_hwi (dist), vf))
return false;
}
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"dependence distance between ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (DDR_A (ddr)));
dump_printf (MSG_NOTE,  " and ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (DDR_B (ddr)));
dump_printf (MSG_NOTE,  " is >= VF\n");
}
return true;
}
static void
dump_lower_bound (int dump_kind, const vec_lower_bound &lower_bound)
{
dump_printf (dump_kind, "%s (", lower_bound.unsigned_p ? "unsigned" : "abs");
dump_generic_expr (dump_kind, TDF_SLIM, lower_bound.expr);
dump_printf (dump_kind, ") >= ");
dump_dec (dump_kind, lower_bound.min_value);
}
static void
vect_check_lower_bound (loop_vec_info loop_vinfo, tree expr, bool unsigned_p,
poly_uint64 min_value)
{
vec<vec_lower_bound> lower_bounds = LOOP_VINFO_LOWER_BOUNDS (loop_vinfo);
for (unsigned int i = 0; i < lower_bounds.length (); ++i)
if (operand_equal_p (lower_bounds[i].expr, expr, 0))
{
unsigned_p &= lower_bounds[i].unsigned_p;
min_value = upper_bound (lower_bounds[i].min_value, min_value);
if (lower_bounds[i].unsigned_p != unsigned_p
|| maybe_lt (lower_bounds[i].min_value, min_value))
{
lower_bounds[i].unsigned_p = unsigned_p;
lower_bounds[i].min_value = min_value;
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"updating run-time check to ");
dump_lower_bound (MSG_NOTE, lower_bounds[i]);
dump_printf (MSG_NOTE, "\n");
}
}
return;
}
vec_lower_bound lower_bound (expr, unsigned_p, min_value);
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location, "need a run-time check that ");
dump_lower_bound (MSG_NOTE, lower_bound);
dump_printf (MSG_NOTE, "\n");
}
LOOP_VINFO_LOWER_BOUNDS (loop_vinfo).safe_push (lower_bound);
}
static bool
vect_small_gap_p (loop_vec_info loop_vinfo, data_reference *dr, poly_int64 gap)
{
stmt_vec_info stmt_info = vinfo_for_stmt (DR_STMT (dr));
HOST_WIDE_INT count
= estimated_poly_value (LOOP_VINFO_VECT_FACTOR (loop_vinfo));
if (GROUP_FIRST_ELEMENT (stmt_info))
count *= GROUP_SIZE (vinfo_for_stmt (GROUP_FIRST_ELEMENT (stmt_info)));
return estimated_poly_value (gap) <= count * vect_get_scalar_dr_size (dr);
}
static bool
vectorizable_with_step_bound_p (data_reference *dr_a, data_reference *dr_b,
poly_uint64 *lower_bound_out)
{
poly_int64 init_a, init_b;
if (!operand_equal_p (DR_BASE_ADDRESS (dr_a), DR_BASE_ADDRESS (dr_b), 0)
|| !operand_equal_p (DR_OFFSET (dr_a), DR_OFFSET (dr_b), 0)
|| !operand_equal_p (DR_STEP (dr_a), DR_STEP (dr_b), 0)
|| !poly_int_tree_p (DR_INIT (dr_a), &init_a)
|| !poly_int_tree_p (DR_INIT (dr_b), &init_b)
|| !ordered_p (init_a, init_b))
return false;
if (maybe_lt (init_b, init_a))
{
std::swap (init_a, init_b);
std::swap (dr_a, dr_b);
}
if (maybe_gt (init_a + vect_get_scalar_dr_size (dr_a), init_b)
&& !vect_preserves_scalar_order_p (DR_STMT (dr_a), DR_STMT (dr_b)))
return false;
*lower_bound_out = init_b + vect_get_scalar_dr_size (dr_b) - init_a;
return true;
}
bool
vect_prune_runtime_alias_test_list (loop_vec_info loop_vinfo)
{
typedef pair_hash <tree_operand_hash, tree_operand_hash> tree_pair_hash;
hash_set <tree_pair_hash> compared_objects;
vec<ddr_p> may_alias_ddrs = LOOP_VINFO_MAY_ALIAS_DDRS (loop_vinfo);
vec<dr_with_seg_len_pair_t> &comp_alias_ddrs
= LOOP_VINFO_COMP_ALIAS_DDRS (loop_vinfo);
vec<vec_object_pair> &check_unequal_addrs
= LOOP_VINFO_CHECK_UNEQUAL_ADDRS (loop_vinfo);
poly_uint64 vect_factor = LOOP_VINFO_VECT_FACTOR (loop_vinfo);
tree scalar_loop_iters = LOOP_VINFO_NITERS (loop_vinfo);
ddr_p ddr;
unsigned int i;
tree length_factor;
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"=== vect_prune_runtime_alias_test_list ===\n");
bool ignore_step_p = known_eq (LOOP_VINFO_VECT_FACTOR (loop_vinfo), 1U);
if (!ignore_step_p)
{
tree value;
FOR_EACH_VEC_ELT (LOOP_VINFO_CHECK_NONZERO (loop_vinfo), i, value)
vect_check_lower_bound (loop_vinfo, value, true, 1);
}
if (may_alias_ddrs.is_empty ())
return true;
comp_alias_ddrs.create (may_alias_ddrs.length ());
unsigned int loop_depth
= index_in_loop_nest (LOOP_VINFO_LOOP (loop_vinfo)->num,
LOOP_VINFO_LOOP_NEST (loop_vinfo));
FOR_EACH_VEC_ELT (may_alias_ddrs, i, ddr)
{
int comp_res;
poly_uint64 lower_bound;
struct data_reference *dr_a, *dr_b;
gimple *dr_group_first_a, *dr_group_first_b;
tree segment_length_a, segment_length_b;
unsigned HOST_WIDE_INT access_size_a, access_size_b;
unsigned int align_a, align_b;
gimple *stmt_a, *stmt_b;
if (dependence_distance_ge_vf (ddr, loop_depth, vect_factor))
continue;
if (DDR_OBJECT_A (ddr))
{
vec_object_pair new_pair (DDR_OBJECT_A (ddr), DDR_OBJECT_B (ddr));
if (!compared_objects.add (new_pair))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location, "checking that ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, new_pair.first);
dump_printf (MSG_NOTE, " and ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, new_pair.second);
dump_printf (MSG_NOTE, " have different addresses\n");
}
LOOP_VINFO_CHECK_UNEQUAL_ADDRS (loop_vinfo).safe_push (new_pair);
}
continue;
}
dr_a = DDR_A (ddr);
stmt_a = DR_STMT (DDR_A (ddr));
dr_b = DDR_B (ddr);
stmt_b = DR_STMT (DDR_B (ddr));
if (ignore_step_p
&& (vect_preserves_scalar_order_p (stmt_a, stmt_b)
|| vectorizable_with_step_bound_p (dr_a, dr_b, &lower_bound)))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"no need for alias check between ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (dr_a));
dump_printf (MSG_NOTE, " and ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (dr_b));
dump_printf (MSG_NOTE, " when VF is 1\n");
}
continue;
}
if (!ignore_step_p
&& TREE_CODE (DR_STEP (dr_a)) != INTEGER_CST
&& vectorizable_with_step_bound_p (dr_a, dr_b, &lower_bound)
&& (vect_small_gap_p (loop_vinfo, dr_a, lower_bound)
|| vect_small_gap_p (loop_vinfo, dr_b, lower_bound)))
{
bool unsigned_p = dr_known_forward_stride_p (dr_a);
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location, "no alias between ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (dr_a));
dump_printf (MSG_NOTE, " and ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (dr_b));
dump_printf (MSG_NOTE, " when the step ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_STEP (dr_a));
dump_printf (MSG_NOTE, " is outside ");
if (unsigned_p)
dump_printf (MSG_NOTE, "[0");
else
{
dump_printf (MSG_NOTE, "(");
dump_dec (MSG_NOTE, poly_int64 (-lower_bound));
}
dump_printf (MSG_NOTE, ", ");
dump_dec (MSG_NOTE, lower_bound);
dump_printf (MSG_NOTE, ")\n");
}
vect_check_lower_bound (loop_vinfo, DR_STEP (dr_a), unsigned_p,
lower_bound);
continue;
}
dr_group_first_a = GROUP_FIRST_ELEMENT (vinfo_for_stmt (stmt_a));
if (dr_group_first_a)
{
stmt_a = dr_group_first_a;
dr_a = STMT_VINFO_DATA_REF (vinfo_for_stmt (stmt_a));
}
dr_group_first_b = GROUP_FIRST_ELEMENT (vinfo_for_stmt (stmt_b));
if (dr_group_first_b)
{
stmt_b = dr_group_first_b;
dr_b = STMT_VINFO_DATA_REF (vinfo_for_stmt (stmt_b));
}
if (ignore_step_p)
{
segment_length_a = size_zero_node;
segment_length_b = size_zero_node;
}
else
{
if (!operand_equal_p (DR_STEP (dr_a), DR_STEP (dr_b), 0))
length_factor = scalar_loop_iters;
else
length_factor = size_int (vect_factor);
segment_length_a = vect_vfa_segment_size (dr_a, length_factor);
segment_length_b = vect_vfa_segment_size (dr_b, length_factor);
}
access_size_a = vect_vfa_access_size (dr_a);
access_size_b = vect_vfa_access_size (dr_b);
align_a = vect_vfa_align (dr_a);
align_b = vect_vfa_align (dr_b);
comp_res = data_ref_compare_tree (DR_BASE_ADDRESS (dr_a),
DR_BASE_ADDRESS (dr_b));
if (comp_res == 0)
comp_res = data_ref_compare_tree (DR_OFFSET (dr_a),
DR_OFFSET (dr_b));
if (comp_res == 0
&& TREE_CODE (DR_STEP (dr_a)) == INTEGER_CST
&& TREE_CODE (DR_STEP (dr_b)) == INTEGER_CST
&& poly_int_tree_p (segment_length_a)
&& poly_int_tree_p (segment_length_b))
{
int res = vect_compile_time_alias (dr_a, dr_b,
segment_length_a,
segment_length_b,
access_size_a,
access_size_b);
if (res >= 0 && dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"can tell at compile time that ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (dr_a));
dump_printf (MSG_NOTE, " and ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_REF (dr_b));
if (res == 0)
dump_printf (MSG_NOTE, " do not alias\n");
else
dump_printf (MSG_NOTE, " alias\n");
}
if (res == 0)
continue;
if (res == 1)
{
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"not vectorized: compilation time alias.\n");
return false;
}
}
dr_with_seg_len_pair_t dr_with_seg_len_pair
(dr_with_seg_len (dr_a, segment_length_a, access_size_a, align_a),
dr_with_seg_len (dr_b, segment_length_b, access_size_b, align_b));
if (comp_res > 0)
std::swap (dr_with_seg_len_pair.first, dr_with_seg_len_pair.second);
comp_alias_ddrs.safe_push (dr_with_seg_len_pair);
}
prune_runtime_alias_test_list (&comp_alias_ddrs, vect_factor);
unsigned int count = (comp_alias_ddrs.length ()
+ check_unequal_addrs.length ());
dump_printf_loc (MSG_NOTE, vect_location,
"improved number of alias checks from %d to %d\n",
may_alias_ddrs.length (), count);
if ((int) count > PARAM_VALUE (PARAM_VECT_MAX_VERSION_FOR_ALIAS_CHECKS))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"number of versioning for alias "
"run-time tests exceeds %d "
"(--param vect-max-version-for-alias-checks)\n",
PARAM_VALUE (PARAM_VECT_MAX_VERSION_FOR_ALIAS_CHECKS));
return false;
}
return true;
}
bool
vect_gather_scatter_fn_p (bool read_p, bool masked_p, tree vectype,
tree memory_type, unsigned int offset_bits,
signop offset_sign, int scale,
internal_fn *ifn_out, tree *element_type_out)
{
unsigned int memory_bits = tree_to_uhwi (TYPE_SIZE (memory_type));
unsigned int element_bits = tree_to_uhwi (TYPE_SIZE (TREE_TYPE (vectype)));
if (offset_bits > element_bits)
return false;
if (element_bits != memory_bits)
return false;
internal_fn ifn;
if (read_p)
ifn = masked_p ? IFN_MASK_GATHER_LOAD : IFN_GATHER_LOAD;
else
ifn = masked_p ? IFN_MASK_SCATTER_STORE : IFN_SCATTER_STORE;
if (!internal_gather_scatter_fn_supported_p (ifn, vectype, memory_type,
offset_sign, scale))
return false;
*ifn_out = ifn;
*element_type_out = TREE_TYPE (vectype);
return true;
}
static void
vect_describe_gather_scatter_call (gcall *call, gather_scatter_info *info)
{
stmt_vec_info stmt_info = vinfo_for_stmt (call);
tree vectype = STMT_VINFO_VECTYPE (stmt_info);
data_reference *dr = STMT_VINFO_DATA_REF (stmt_info);
info->ifn = gimple_call_internal_fn (call);
info->decl = NULL_TREE;
info->base = gimple_call_arg (call, 0);
info->offset = gimple_call_arg (call, 1);
info->offset_dt = vect_unknown_def_type;
info->offset_vectype = NULL_TREE;
info->scale = TREE_INT_CST_LOW (gimple_call_arg (call, 2));
info->element_type = TREE_TYPE (vectype);
info->memory_type = TREE_TYPE (DR_REF (dr));
}
bool
vect_check_gather_scatter (gimple *stmt, loop_vec_info loop_vinfo,
gather_scatter_info *info)
{
HOST_WIDE_INT scale = 1;
poly_int64 pbitpos, pbitsize;
struct loop *loop = LOOP_VINFO_LOOP (loop_vinfo);
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
struct data_reference *dr = STMT_VINFO_DATA_REF (stmt_info);
tree offtype = NULL_TREE;
tree decl = NULL_TREE, base, off;
tree vectype = STMT_VINFO_VECTYPE (stmt_info);
tree memory_type = TREE_TYPE (DR_REF (dr));
machine_mode pmode;
int punsignedp, reversep, pvolatilep = 0;
internal_fn ifn;
tree element_type;
bool masked_p = false;
gcall *call = dyn_cast <gcall *> (stmt);
if (call && gimple_call_internal_p (call))
{
ifn = gimple_call_internal_fn (stmt);
if (internal_gather_scatter_fn_p (ifn))
{
vect_describe_gather_scatter_call (call, info);
return true;
}
masked_p = (ifn == IFN_MASK_LOAD || ifn == IFN_MASK_STORE);
}
bool use_ifn_p = (DR_IS_READ (dr)
? supports_vec_gather_load_p ()
: supports_vec_scatter_store_p ());
base = DR_REF (dr);
if (masked_p
&& TREE_CODE (base) == MEM_REF
&& TREE_CODE (TREE_OPERAND (base, 0)) == SSA_NAME
&& integer_zerop (TREE_OPERAND (base, 1))
&& !expr_invariant_in_loop_p (loop, TREE_OPERAND (base, 0)))
{
gimple *def_stmt = SSA_NAME_DEF_STMT (TREE_OPERAND (base, 0));
if (is_gimple_assign (def_stmt)
&& gimple_assign_rhs_code (def_stmt) == ADDR_EXPR)
base = TREE_OPERAND (gimple_assign_rhs1 (def_stmt), 0);
}
base = get_inner_reference (base, &pbitsize, &pbitpos, &off, &pmode,
&punsignedp, &reversep, &pvolatilep);
gcc_assert (base && !reversep);
poly_int64 pbytepos = exact_div (pbitpos, BITS_PER_UNIT);
if (TREE_CODE (base) == MEM_REF)
{
if (!integer_zerop (TREE_OPERAND (base, 1)))
{
if (off == NULL_TREE)
off = wide_int_to_tree (sizetype, mem_ref_offset (base));
else
off = size_binop (PLUS_EXPR, off,
fold_convert (sizetype, TREE_OPERAND (base, 1)));
}
base = TREE_OPERAND (base, 0);
}
else
base = build_fold_addr_expr (base);
if (off == NULL_TREE)
off = size_zero_node;
if (!expr_invariant_in_loop_p (loop, base))
{
if (!integer_zerop (off))
return false;
off = base;
base = size_int (pbytepos);
}
else
{
base = fold_convert (sizetype, base);
base = size_binop (PLUS_EXPR, base, size_int (pbytepos));
}
STRIP_NOPS (off);
while (offtype == NULL_TREE)
{
enum tree_code code;
tree op0, op1, add = NULL_TREE;
if (TREE_CODE (off) == SSA_NAME)
{
gimple *def_stmt = SSA_NAME_DEF_STMT (off);
if (expr_invariant_in_loop_p (loop, off))
return false;
if (gimple_code (def_stmt) != GIMPLE_ASSIGN)
break;
op0 = gimple_assign_rhs1 (def_stmt);
code = gimple_assign_rhs_code (def_stmt);
op1 = gimple_assign_rhs2 (def_stmt);
}
else
{
if (get_gimple_rhs_class (TREE_CODE (off)) == GIMPLE_TERNARY_RHS)
return false;
code = TREE_CODE (off);
extract_ops_from_tree (off, &code, &op0, &op1);
}
switch (code)
{
case POINTER_PLUS_EXPR:
case PLUS_EXPR:
if (expr_invariant_in_loop_p (loop, op0))
{
add = op0;
off = op1;
do_add:
add = fold_convert (sizetype, add);
if (scale != 1)
add = size_binop (MULT_EXPR, add, size_int (scale));
base = size_binop (PLUS_EXPR, base, add);
continue;
}
if (expr_invariant_in_loop_p (loop, op1))
{
add = op1;
off = op0;
goto do_add;
}
break;
case MINUS_EXPR:
if (expr_invariant_in_loop_p (loop, op1))
{
add = fold_convert (sizetype, op1);
add = size_binop (MINUS_EXPR, size_zero_node, add);
off = op0;
goto do_add;
}
break;
case MULT_EXPR:
if (scale == 1 && tree_fits_shwi_p (op1))
{
int new_scale = tree_to_shwi (op1);
if (use_ifn_p
&& !vect_gather_scatter_fn_p (DR_IS_READ (dr), masked_p,
vectype, memory_type, 1,
TYPE_SIGN (TREE_TYPE (op0)),
new_scale, &ifn,
&element_type))
break;
scale = new_scale;
off = op0;
continue;
}
break;
case SSA_NAME:
off = op0;
continue;
CASE_CONVERT:
if (!POINTER_TYPE_P (TREE_TYPE (op0))
&& !INTEGRAL_TYPE_P (TREE_TYPE (op0)))
break;
if (TYPE_PRECISION (TREE_TYPE (op0))
== TYPE_PRECISION (TREE_TYPE (off)))
{
off = op0;
continue;
}
if (use_ifn_p
&& (int_size_in_bytes (TREE_TYPE (vectype))
== int_size_in_bytes (TREE_TYPE (off))))
break;
if (TYPE_PRECISION (TREE_TYPE (op0))
< TYPE_PRECISION (TREE_TYPE (off)))
{
off = op0;
offtype = TREE_TYPE (off);
STRIP_NOPS (off);
continue;
}
break;
default:
break;
}
break;
}
if (TREE_CODE (off) != SSA_NAME
|| expr_invariant_in_loop_p (loop, off))
return false;
if (offtype == NULL_TREE)
offtype = TREE_TYPE (off);
if (use_ifn_p)
{
if (!vect_gather_scatter_fn_p (DR_IS_READ (dr), masked_p, vectype,
memory_type, TYPE_PRECISION (offtype),
TYPE_SIGN (offtype), scale, &ifn,
&element_type))
return false;
}
else
{
if (DR_IS_READ (dr))
{
if (targetm.vectorize.builtin_gather)
decl = targetm.vectorize.builtin_gather (vectype, offtype, scale);
}
else
{
if (targetm.vectorize.builtin_scatter)
decl = targetm.vectorize.builtin_scatter (vectype, offtype, scale);
}
if (!decl)
return false;
ifn = IFN_LAST;
element_type = TREE_TYPE (vectype);
}
info->ifn = ifn;
info->decl = decl;
info->base = base;
info->offset = off;
info->offset_dt = vect_unknown_def_type;
info->offset_vectype = NULL_TREE;
info->scale = scale;
info->element_type = element_type;
info->memory_type = memory_type;
return true;
}
bool
vect_analyze_data_refs (vec_info *vinfo, poly_uint64 *min_vf)
{
struct loop *loop = NULL;
unsigned int i;
struct data_reference *dr;
tree scalar_type;
if (dump_enabled_p ())
dump_printf_loc (MSG_NOTE, vect_location,
"=== vect_analyze_data_refs ===\n");
if (loop_vec_info loop_vinfo = dyn_cast <loop_vec_info> (vinfo))
loop = LOOP_VINFO_LOOP (loop_vinfo);
vec<data_reference_p> datarefs = vinfo->datarefs;
FOR_EACH_VEC_ELT (datarefs, i, dr)
{
gimple *stmt;
stmt_vec_info stmt_info;
tree base, offset, init;
enum { SG_NONE, GATHER, SCATTER } gatherscatter = SG_NONE;
bool simd_lane_access = false;
poly_uint64 vf;
again:
if (!dr || !DR_REF (dr))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: unhandled data-ref\n");
return false;
}
stmt = DR_STMT (dr);
stmt_info = vinfo_for_stmt (stmt);
if (gimple_clobber_p (stmt))
{
free_data_ref (dr);
if (i == datarefs.length () - 1)
{
datarefs.pop ();
break;
}
datarefs.ordered_remove (i);
dr = datarefs[i];
goto again;
}
if (!DR_BASE_ADDRESS (dr) || !DR_OFFSET (dr) || !DR_INIT (dr)
|| !DR_STEP (dr))
{
bool maybe_gather
= DR_IS_READ (dr)
&& !TREE_THIS_VOLATILE (DR_REF (dr))
&& (targetm.vectorize.builtin_gather != NULL
|| supports_vec_gather_load_p ());
bool maybe_scatter
= DR_IS_WRITE (dr)
&& !TREE_THIS_VOLATILE (DR_REF (dr))
&& (targetm.vectorize.builtin_scatter != NULL
|| supports_vec_scatter_store_p ());
bool maybe_simd_lane_access
= is_a <loop_vec_info> (vinfo) && loop->simduid;
if (is_a <loop_vec_info> (vinfo)
&& (maybe_gather || maybe_scatter || maybe_simd_lane_access)
&& !nested_in_vect_loop_p (loop, stmt))
{
struct data_reference *newdr
= create_data_ref (NULL, loop_containing_stmt (stmt),
DR_REF (dr), stmt, !maybe_scatter,
DR_IS_CONDITIONAL_IN_STMT (dr));
gcc_assert (newdr != NULL && DR_REF (newdr));
if (DR_BASE_ADDRESS (newdr)
&& DR_OFFSET (newdr)
&& DR_INIT (newdr)
&& DR_STEP (newdr)
&& integer_zerop (DR_STEP (newdr)))
{
if (maybe_simd_lane_access)
{
tree off = DR_OFFSET (newdr);
STRIP_NOPS (off);
if (TREE_CODE (DR_INIT (newdr)) == INTEGER_CST
&& TREE_CODE (off) == MULT_EXPR
&& tree_fits_uhwi_p (TREE_OPERAND (off, 1)))
{
tree step = TREE_OPERAND (off, 1);
off = TREE_OPERAND (off, 0);
STRIP_NOPS (off);
if (CONVERT_EXPR_P (off)
&& TYPE_PRECISION (TREE_TYPE (TREE_OPERAND (off,
0)))
< TYPE_PRECISION (TREE_TYPE (off)))
off = TREE_OPERAND (off, 0);
if (TREE_CODE (off) == SSA_NAME)
{
gimple *def = SSA_NAME_DEF_STMT (off);
tree reft = TREE_TYPE (DR_REF (newdr));
if (is_gimple_call (def)
&& gimple_call_internal_p (def)
&& (gimple_call_internal_fn (def)
== IFN_GOMP_SIMD_LANE))
{
tree arg = gimple_call_arg (def, 0);
gcc_assert (TREE_CODE (arg) == SSA_NAME);
arg = SSA_NAME_VAR (arg);
if (arg == loop->simduid
&& tree_int_cst_equal
(TYPE_SIZE_UNIT (reft),
step))
{
DR_OFFSET (newdr) = ssize_int (0);
DR_STEP (newdr) = step;
DR_OFFSET_ALIGNMENT (newdr)
= BIGGEST_ALIGNMENT;
DR_STEP_ALIGNMENT (newdr)
= highest_pow2_factor (step);
dr = newdr;
simd_lane_access = true;
}
}
}
}
}
if (!simd_lane_access && (maybe_gather || maybe_scatter))
{
dr = newdr;
if (maybe_gather)
gatherscatter = GATHER;
else
gatherscatter = SCATTER;
}
}
if (gatherscatter == SG_NONE && !simd_lane_access)
free_data_ref (newdr);
}
if (gatherscatter == SG_NONE && !simd_lane_access)
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: data ref analysis "
"failed ");
dump_gimple_stmt (MSG_MISSED_OPTIMIZATION, TDF_SLIM, stmt, 0);
}
if (is_a <bb_vec_info> (vinfo))
break;
return false;
}
}
if (TREE_CODE (DR_BASE_ADDRESS (dr)) == INTEGER_CST)
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: base addr of dr is a "
"constant\n");
if (is_a <bb_vec_info> (vinfo))
break;
if (gatherscatter != SG_NONE || simd_lane_access)
free_data_ref (dr);
return false;
}
if (TREE_THIS_VOLATILE (DR_REF (dr)))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: volatile type ");
dump_gimple_stmt (MSG_MISSED_OPTIMIZATION, TDF_SLIM, stmt, 0);
}
if (is_a <bb_vec_info> (vinfo))
break;
return false;
}
if (stmt_can_throw_internal (stmt))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: statement can throw an "
"exception ");
dump_gimple_stmt (MSG_MISSED_OPTIMIZATION, TDF_SLIM, stmt, 0);
}
if (is_a <bb_vec_info> (vinfo))
break;
if (gatherscatter != SG_NONE || simd_lane_access)
free_data_ref (dr);
return false;
}
if (TREE_CODE (DR_REF (dr)) == COMPONENT_REF
&& DECL_BIT_FIELD (TREE_OPERAND (DR_REF (dr), 1)))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: statement is bitfield "
"access ");
dump_gimple_stmt (MSG_MISSED_OPTIMIZATION, TDF_SLIM, stmt, 0);
}
if (is_a <bb_vec_info> (vinfo))
break;
if (gatherscatter != SG_NONE || simd_lane_access)
free_data_ref (dr);
return false;
}
base = unshare_expr (DR_BASE_ADDRESS (dr));
offset = unshare_expr (DR_OFFSET (dr));
init = unshare_expr (DR_INIT (dr));
if (is_gimple_call (stmt)
&& (!gimple_call_internal_p (stmt)
|| (gimple_call_internal_fn (stmt) != IFN_MASK_LOAD
&& gimple_call_internal_fn (stmt) != IFN_MASK_STORE)))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION,  vect_location,
"not vectorized: dr in a call ");
dump_gimple_stmt (MSG_MISSED_OPTIMIZATION, TDF_SLIM, stmt, 0);
}
if (is_a <bb_vec_info> (vinfo))
break;
if (gatherscatter != SG_NONE || simd_lane_access)
free_data_ref (dr);
return false;
}
if (loop && nested_in_vect_loop_p (loop, stmt))
{
tree init_offset = fold_build2 (PLUS_EXPR, TREE_TYPE (offset),
init, offset);
tree init_addr = fold_build_pointer_plus (base, init_offset);
tree init_ref = build_fold_indirect_ref (init_addr);
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"analyze in outer loop: ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, init_ref);
dump_printf (MSG_NOTE, "\n");
}
if (!dr_analyze_innermost (&STMT_VINFO_DR_WRT_VEC_LOOP (stmt_info),
init_ref, loop))
return false;
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"\touter base_address: ");
dump_generic_expr (MSG_NOTE, TDF_SLIM,
STMT_VINFO_DR_BASE_ADDRESS (stmt_info));
dump_printf (MSG_NOTE, "\n\touter offset from base address: ");
dump_generic_expr (MSG_NOTE, TDF_SLIM,
STMT_VINFO_DR_OFFSET (stmt_info));
dump_printf (MSG_NOTE,
"\n\touter constant offset from base address: ");
dump_generic_expr (MSG_NOTE, TDF_SLIM,
STMT_VINFO_DR_INIT (stmt_info));
dump_printf (MSG_NOTE, "\n\touter step: ");
dump_generic_expr (MSG_NOTE, TDF_SLIM,
STMT_VINFO_DR_STEP (stmt_info));
dump_printf (MSG_NOTE, "\n\touter base alignment: %d\n",
STMT_VINFO_DR_BASE_ALIGNMENT (stmt_info));
dump_printf (MSG_NOTE, "\n\touter base misalignment: %d\n",
STMT_VINFO_DR_BASE_MISALIGNMENT (stmt_info));
dump_printf (MSG_NOTE, "\n\touter offset alignment: %d\n",
STMT_VINFO_DR_OFFSET_ALIGNMENT (stmt_info));
dump_printf (MSG_NOTE, "\n\touter step alignment: %d\n",
STMT_VINFO_DR_STEP_ALIGNMENT (stmt_info));
}
}
if (STMT_VINFO_DATA_REF (stmt_info))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: more than one data ref "
"in stmt: ");
dump_gimple_stmt (MSG_MISSED_OPTIMIZATION, TDF_SLIM, stmt, 0);
}
if (is_a <bb_vec_info> (vinfo))
break;
if (gatherscatter != SG_NONE || simd_lane_access)
free_data_ref (dr);
return false;
}
STMT_VINFO_DATA_REF (stmt_info) = dr;
if (simd_lane_access)
{
STMT_VINFO_SIMD_LANE_ACCESS_P (stmt_info) = true;
free_data_ref (datarefs[i]);
datarefs[i] = dr;
}
if (TREE_CODE (DR_BASE_ADDRESS (dr)) == ADDR_EXPR
&& VAR_P (TREE_OPERAND (DR_BASE_ADDRESS (dr), 0))
&& DECL_NONALIASED (TREE_OPERAND (DR_BASE_ADDRESS (dr), 0)))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: base object not addressable "
"for stmt: ");
dump_gimple_stmt (MSG_MISSED_OPTIMIZATION, TDF_SLIM, stmt, 0);
}
if (is_a <bb_vec_info> (vinfo))
{
STMT_VINFO_VECTORIZABLE (stmt_info) = false;
continue;
}
return false;
}
scalar_type = TREE_TYPE (DR_REF (dr));
STMT_VINFO_VECTYPE (stmt_info)
= get_vectype_for_scalar_type (scalar_type);
if (!STMT_VINFO_VECTYPE (stmt_info))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"not vectorized: no vectype for stmt: ");
dump_gimple_stmt (MSG_MISSED_OPTIMIZATION, TDF_SLIM, stmt, 0);
dump_printf (MSG_MISSED_OPTIMIZATION, " scalar_type: ");
dump_generic_expr (MSG_MISSED_OPTIMIZATION, TDF_DETAILS,
scalar_type);
dump_printf (MSG_MISSED_OPTIMIZATION, "\n");
}
if (is_a <bb_vec_info> (vinfo))
{
STMT_VINFO_VECTORIZABLE (stmt_info) = false;
continue;
}
if (gatherscatter != SG_NONE || simd_lane_access)
{
STMT_VINFO_DATA_REF (stmt_info) = NULL;
if (gatherscatter != SG_NONE)
free_data_ref (dr);
}
return false;
}
else
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location,
"got vectype for stmt: ");
dump_gimple_stmt (MSG_NOTE, TDF_SLIM, stmt, 0);
dump_generic_expr (MSG_NOTE, TDF_SLIM,
STMT_VINFO_VECTYPE (stmt_info));
dump_printf (MSG_NOTE, "\n");
}
}
vf = TYPE_VECTOR_SUBPARTS (STMT_VINFO_VECTYPE (stmt_info));
*min_vf = upper_bound (*min_vf, vf);
if (gatherscatter != SG_NONE)
{
gather_scatter_info gs_info;
if (!vect_check_gather_scatter (stmt, as_a <loop_vec_info> (vinfo),
&gs_info)
|| !get_vectype_for_scalar_type (TREE_TYPE (gs_info.offset)))
{
STMT_VINFO_DATA_REF (stmt_info) = NULL;
free_data_ref (dr);
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
(gatherscatter == GATHER) ?
"not vectorized: not suitable for gather "
"load " :
"not vectorized: not suitable for scatter "
"store ");
dump_gimple_stmt (MSG_MISSED_OPTIMIZATION, TDF_SLIM, stmt, 0);
}
return false;
}
free_data_ref (datarefs[i]);
datarefs[i] = dr;
STMT_VINFO_GATHER_SCATTER_P (stmt_info) = gatherscatter;
}
else if (is_a <loop_vec_info> (vinfo)
&& TREE_CODE (DR_STEP (dr)) != INTEGER_CST)
{
if (nested_in_vect_loop_p (loop, stmt))
{
if (dump_enabled_p ())
{
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location, 
"not vectorized: not suitable for strided "
"load ");
dump_gimple_stmt (MSG_MISSED_OPTIMIZATION, TDF_SLIM, stmt, 0);
}
return false;
}
STMT_VINFO_STRIDED_P (stmt_info) = true;
}
}
if (i != datarefs.length ())
{
gcc_assert (is_a <bb_vec_info> (vinfo));
for (unsigned j = i; j < datarefs.length (); ++j)
{
data_reference_p dr = datarefs[j];
STMT_VINFO_VECTORIZABLE (vinfo_for_stmt (DR_STMT (dr))) = false;
free_data_ref (dr);
}
datarefs.truncate (i);
}
return true;
}
tree
vect_get_new_vect_var (tree type, enum vect_var_kind var_kind, const char *name)
{
const char *prefix;
tree new_vect_var;
switch (var_kind)
{
case vect_simple_var:
prefix = "vect";
break;
case vect_scalar_var:
prefix = "stmp";
break;
case vect_mask_var:
prefix = "mask";
break;
case vect_pointer_var:
prefix = "vectp";
break;
default:
gcc_unreachable ();
}
if (name)
{
char* tmp = concat (prefix, "_", name, NULL);
new_vect_var = create_tmp_reg (type, tmp);
free (tmp);
}
else
new_vect_var = create_tmp_reg (type, prefix);
return new_vect_var;
}
tree
vect_get_new_ssa_name (tree type, enum vect_var_kind var_kind, const char *name)
{
const char *prefix;
tree new_vect_var;
switch (var_kind)
{
case vect_simple_var:
prefix = "vect";
break;
case vect_scalar_var:
prefix = "stmp";
break;
case vect_pointer_var:
prefix = "vectp";
break;
default:
gcc_unreachable ();
}
if (name)
{
char* tmp = concat (prefix, "_", name, NULL);
new_vect_var = make_temp_ssa_name (type, NULL, tmp);
free (tmp);
}
else
new_vect_var = make_temp_ssa_name (type, NULL, prefix);
return new_vect_var;
}
static void
vect_duplicate_ssa_name_ptr_info (tree name, data_reference *dr)
{
duplicate_ssa_name_ptr_info (name, DR_PTR_INFO (dr));
int misalign = DR_MISALIGNMENT (dr);
if (misalign == DR_MISALIGNMENT_UNKNOWN)
mark_ptr_info_alignment_unknown (SSA_NAME_PTR_INFO (name));
else
set_ptr_info_alignment (SSA_NAME_PTR_INFO (name),
DR_TARGET_ALIGNMENT (dr), misalign);
}
tree
vect_create_addr_base_for_vector_ref (gimple *stmt,
gimple_seq *new_stmt_list,
tree offset,
tree byte_offset)
{
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
struct data_reference *dr = STMT_VINFO_DATA_REF (stmt_info);
const char *base_name;
tree addr_base;
tree dest;
gimple_seq seq = NULL;
tree vect_ptr_type;
tree step = TYPE_SIZE_UNIT (TREE_TYPE (DR_REF (dr)));
loop_vec_info loop_vinfo = STMT_VINFO_LOOP_VINFO (stmt_info);
innermost_loop_behavior *drb = vect_dr_behavior (dr);
tree data_ref_base = unshare_expr (drb->base_address);
tree base_offset = unshare_expr (drb->offset);
tree init = unshare_expr (drb->init);
if (loop_vinfo)
base_name = get_name (data_ref_base);
else
{
base_offset = ssize_int (0);
init = ssize_int (0);
base_name = get_name (DR_REF (dr));
}
base_offset = size_binop (PLUS_EXPR,
fold_convert (sizetype, base_offset),
fold_convert (sizetype, init));
if (offset)
{
offset = fold_build2 (MULT_EXPR, sizetype,
fold_convert (sizetype, offset), step);
base_offset = fold_build2 (PLUS_EXPR, sizetype,
base_offset, offset);
}
if (byte_offset)
{
byte_offset = fold_convert (sizetype, byte_offset);
base_offset = fold_build2 (PLUS_EXPR, sizetype,
base_offset, byte_offset);
}
if (loop_vinfo)
addr_base = fold_build_pointer_plus (data_ref_base, base_offset);
else
{
addr_base = build1 (ADDR_EXPR,
build_pointer_type (TREE_TYPE (DR_REF (dr))),
unshare_expr (DR_REF (dr)));
}
vect_ptr_type = build_pointer_type (STMT_VINFO_VECTYPE (stmt_info));
dest = vect_get_new_vect_var (vect_ptr_type, vect_pointer_var, base_name);
addr_base = force_gimple_operand (addr_base, &seq, true, dest);
gimple_seq_add_seq (new_stmt_list, seq);
if (DR_PTR_INFO (dr)
&& TREE_CODE (addr_base) == SSA_NAME
&& !SSA_NAME_PTR_INFO (addr_base))
{
vect_duplicate_ssa_name_ptr_info (addr_base, dr);
if (offset || byte_offset)
mark_ptr_info_alignment_unknown (SSA_NAME_PTR_INFO (addr_base));
}
if (dump_enabled_p ())
{
dump_printf_loc (MSG_NOTE, vect_location, "created ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, addr_base);
dump_printf (MSG_NOTE, "\n");
}
return addr_base;
}
tree
vect_create_data_ref_ptr (gimple *stmt, tree aggr_type, struct loop *at_loop,
tree offset, tree *initial_address,
gimple_stmt_iterator *gsi, gimple **ptr_incr,
bool only_init, bool *inv_p, tree byte_offset,
tree iv_step)
{
const char *base_name;
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
loop_vec_info loop_vinfo = STMT_VINFO_LOOP_VINFO (stmt_info);
struct loop *loop = NULL;
bool nested_in_vect_loop = false;
struct loop *containing_loop = NULL;
tree aggr_ptr_type;
tree aggr_ptr;
tree new_temp;
gimple_seq new_stmt_list = NULL;
edge pe = NULL;
basic_block new_bb;
tree aggr_ptr_init;
struct data_reference *dr = STMT_VINFO_DATA_REF (stmt_info);
tree aptr;
gimple_stmt_iterator incr_gsi;
bool insert_after;
tree indx_before_incr, indx_after_incr;
gimple *incr;
tree step;
bb_vec_info bb_vinfo = STMT_VINFO_BB_VINFO (stmt_info);
gcc_assert (iv_step != NULL_TREE
|| TREE_CODE (aggr_type) == ARRAY_TYPE
|| TREE_CODE (aggr_type) == VECTOR_TYPE);
if (loop_vinfo)
{
loop = LOOP_VINFO_LOOP (loop_vinfo);
nested_in_vect_loop = nested_in_vect_loop_p (loop, stmt);
containing_loop = (gimple_bb (stmt))->loop_father;
pe = loop_preheader_edge (loop);
}
else
{
gcc_assert (bb_vinfo);
only_init = true;
*ptr_incr = NULL;
}
step = vect_dr_behavior (dr)->step;
if (integer_zerop (step))
*inv_p = true;
else
*inv_p = false;
base_name = get_name (DR_BASE_ADDRESS (dr));
if (dump_enabled_p ())
{
tree dr_base_type = TREE_TYPE (DR_BASE_OBJECT (dr));
dump_printf_loc (MSG_NOTE, vect_location,
"create %s-pointer variable to type: ",
get_tree_code_name (TREE_CODE (aggr_type)));
dump_generic_expr (MSG_NOTE, TDF_SLIM, aggr_type);
if (TREE_CODE (dr_base_type) == ARRAY_TYPE)
dump_printf (MSG_NOTE, "  vectorizing an array ref: ");
else if (TREE_CODE (dr_base_type) == VECTOR_TYPE)
dump_printf (MSG_NOTE, "  vectorizing a vector ref: ");
else if (TREE_CODE (dr_base_type) == RECORD_TYPE)
dump_printf (MSG_NOTE, "  vectorizing a record based array ref: ");
else
dump_printf (MSG_NOTE, "  vectorizing a pointer ref: ");
dump_generic_expr (MSG_NOTE, TDF_SLIM, DR_BASE_OBJECT (dr));
dump_printf (MSG_NOTE, "\n");
}
bool need_ref_all = false;
if (!alias_sets_conflict_p (get_alias_set (aggr_type),
get_alias_set (DR_REF (dr))))
need_ref_all = true;
else if (STMT_VINFO_GROUP_SIZE (stmt_info) > 1)
{
gimple *orig_stmt = STMT_VINFO_GROUP_FIRST_ELEMENT (stmt_info);
do
{
stmt_vec_info sinfo = vinfo_for_stmt (orig_stmt);
struct data_reference *sdr = STMT_VINFO_DATA_REF (sinfo);
if (!alias_sets_conflict_p (get_alias_set (aggr_type),
get_alias_set (DR_REF (sdr))))
{
need_ref_all = true;
break;
}
orig_stmt = STMT_VINFO_GROUP_NEXT_ELEMENT (sinfo);
}
while (orig_stmt);
}
aggr_ptr_type = build_pointer_type_for_mode (aggr_type, ptr_mode,
need_ref_all);
aggr_ptr = vect_get_new_vect_var (aggr_ptr_type, vect_pointer_var, base_name);
new_temp = vect_create_addr_base_for_vector_ref (stmt, &new_stmt_list,
offset, byte_offset);
if (new_stmt_list)
{
if (pe)
{
new_bb = gsi_insert_seq_on_edge_immediate (pe, new_stmt_list);
gcc_assert (!new_bb);
}
else
gsi_insert_seq_before (gsi, new_stmt_list, GSI_SAME_STMT);
}
*initial_address = new_temp;
aggr_ptr_init = new_temp;
if (only_init && (!loop_vinfo || at_loop == loop))
aptr = aggr_ptr_init;
else
{
if (iv_step == NULL_TREE)
{
iv_step = TYPE_SIZE_UNIT (aggr_type);
if (*inv_p)
iv_step = size_zero_node;
else if (tree_int_cst_sgn (step) == -1)
iv_step = fold_build1 (NEGATE_EXPR, TREE_TYPE (iv_step), iv_step);
}
standard_iv_increment_position (loop, &incr_gsi, &insert_after);
create_iv (aggr_ptr_init,
fold_convert (aggr_ptr_type, iv_step),
aggr_ptr, loop, &incr_gsi, insert_after,
&indx_before_incr, &indx_after_incr);
incr = gsi_stmt (incr_gsi);
set_vinfo_for_stmt (incr, new_stmt_vec_info (incr, loop_vinfo));
if (DR_PTR_INFO (dr))
{
vect_duplicate_ssa_name_ptr_info (indx_before_incr, dr);
vect_duplicate_ssa_name_ptr_info (indx_after_incr, dr);
}
if (ptr_incr)
*ptr_incr = incr;
aptr = indx_before_incr;
}
if (!nested_in_vect_loop || only_init)
return aptr;
gcc_assert (nested_in_vect_loop);
if (!only_init)
{
standard_iv_increment_position (containing_loop, &incr_gsi,
&insert_after);
create_iv (aptr, fold_convert (aggr_ptr_type, DR_STEP (dr)), aggr_ptr,
containing_loop, &incr_gsi, insert_after, &indx_before_incr,
&indx_after_incr);
incr = gsi_stmt (incr_gsi);
set_vinfo_for_stmt (incr, new_stmt_vec_info (incr, loop_vinfo));
if (DR_PTR_INFO (dr))
{
vect_duplicate_ssa_name_ptr_info (indx_before_incr, dr);
vect_duplicate_ssa_name_ptr_info (indx_after_incr, dr);
}
if (ptr_incr)
*ptr_incr = incr;
return indx_before_incr;
}
else
gcc_unreachable ();
}
tree
bump_vector_ptr (tree dataref_ptr, gimple *ptr_incr, gimple_stmt_iterator *gsi,
gimple *stmt, tree bump)
{
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
struct data_reference *dr = STMT_VINFO_DATA_REF (stmt_info);
tree vectype = STMT_VINFO_VECTYPE (stmt_info);
tree update = TYPE_SIZE_UNIT (vectype);
gassign *incr_stmt;
ssa_op_iter iter;
use_operand_p use_p;
tree new_dataref_ptr;
if (bump)
update = bump;
if (TREE_CODE (dataref_ptr) == SSA_NAME)
new_dataref_ptr = copy_ssa_name (dataref_ptr);
else
new_dataref_ptr = make_ssa_name (TREE_TYPE (dataref_ptr));
incr_stmt = gimple_build_assign (new_dataref_ptr, POINTER_PLUS_EXPR,
dataref_ptr, update);
vect_finish_stmt_generation (stmt, incr_stmt, gsi);
if (DR_PTR_INFO (dr))
{
duplicate_ssa_name_ptr_info (new_dataref_ptr, DR_PTR_INFO (dr));
mark_ptr_info_alignment_unknown (SSA_NAME_PTR_INFO (new_dataref_ptr));
}
if (!ptr_incr)
return new_dataref_ptr;
FOR_EACH_SSA_USE_OPERAND (use_p, ptr_incr, iter, SSA_OP_USE)
{
tree use = USE_FROM_PTR (use_p);
if (use == dataref_ptr)
SET_USE (use_p, new_dataref_ptr);
else
gcc_assert (operand_equal_p (use, update, 0));
}
return new_dataref_ptr;
}
void
vect_copy_ref_info (tree dest, tree src)
{
if (TREE_CODE (dest) != MEM_REF)
return;
tree src_base = src;
while (handled_component_p (src_base))
src_base = TREE_OPERAND (src_base, 0);
if (TREE_CODE (src_base) != MEM_REF
&& TREE_CODE (src_base) != TARGET_MEM_REF)
return;
MR_DEPENDENCE_CLIQUE (dest) = MR_DEPENDENCE_CLIQUE (src_base);
MR_DEPENDENCE_BASE (dest) = MR_DEPENDENCE_BASE (src_base);
}
tree
vect_create_destination_var (tree scalar_dest, tree vectype)
{
tree vec_dest;
const char *name;
char *new_name;
tree type;
enum vect_var_kind kind;
kind = vectype
? VECTOR_BOOLEAN_TYPE_P (vectype)
? vect_mask_var
: vect_simple_var
: vect_scalar_var;
type = vectype ? vectype : TREE_TYPE (scalar_dest);
gcc_assert (TREE_CODE (scalar_dest) == SSA_NAME);
name = get_name (scalar_dest);
if (name)
new_name = xasprintf ("%s_%u", name, SSA_NAME_VERSION (scalar_dest));
else
new_name = xasprintf ("_%u", SSA_NAME_VERSION (scalar_dest));
vec_dest = vect_get_new_vect_var (type, kind, new_name);
free (new_name);
return vec_dest;
}
bool
vect_grouped_store_supported (tree vectype, unsigned HOST_WIDE_INT count)
{
machine_mode mode = TYPE_MODE (vectype);
if (count != 3 && exact_log2 (count) == -1)
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"the size of the group of accesses"
" is not a power of 2 or not eqaul to 3\n");
return false;
}
if (VECTOR_MODE_P (mode))
{
unsigned int i;
if (count == 3)
{
unsigned int j0 = 0, j1 = 0, j2 = 0;
unsigned int i, j;
unsigned int nelt;
if (!GET_MODE_NUNITS (mode).is_constant (&nelt))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"cannot handle groups of 3 stores for"
" variable-length vectors\n");
return false;
}
vec_perm_builder sel (nelt, nelt, 1);
sel.quick_grow (nelt);
vec_perm_indices indices;
for (j = 0; j < 3; j++)
{
int nelt0 = ((3 - j) * nelt) % 3;
int nelt1 = ((3 - j) * nelt + 1) % 3;
int nelt2 = ((3 - j) * nelt + 2) % 3;
for (i = 0; i < nelt; i++)
{
if (3 * i + nelt0 < nelt)
sel[3 * i + nelt0] = j0++;
if (3 * i + nelt1 < nelt)
sel[3 * i + nelt1] = nelt + j1++;
if (3 * i + nelt2 < nelt)
sel[3 * i + nelt2] = 0;
}
indices.new_vector (sel, 2, nelt);
if (!can_vec_perm_const_p (mode, indices))
{
if (dump_enabled_p ())
dump_printf (MSG_MISSED_OPTIMIZATION,
"permutation op not supported by target.\n");
return false;
}
for (i = 0; i < nelt; i++)
{
if (3 * i + nelt0 < nelt)
sel[3 * i + nelt0] = 3 * i + nelt0;
if (3 * i + nelt1 < nelt)
sel[3 * i + nelt1] = 3 * i + nelt1;
if (3 * i + nelt2 < nelt)
sel[3 * i + nelt2] = nelt + j2++;
}
indices.new_vector (sel, 2, nelt);
if (!can_vec_perm_const_p (mode, indices))
{
if (dump_enabled_p ())
dump_printf (MSG_MISSED_OPTIMIZATION,
"permutation op not supported by target.\n");
return false;
}
}
return true;
}
else
{
gcc_assert (pow2p_hwi (count));
poly_uint64 nelt = GET_MODE_NUNITS (mode);
vec_perm_builder sel (nelt, 2, 3);
sel.quick_grow (6);
for (i = 0; i < 3; i++)
{
sel[i * 2] = i;
sel[i * 2 + 1] = i + nelt;
}
vec_perm_indices indices (sel, 2, nelt);
if (can_vec_perm_const_p (mode, indices))
{
for (i = 0; i < 6; i++)
sel[i] += exact_div (nelt, 2);
indices.new_vector (sel, 2, nelt);
if (can_vec_perm_const_p (mode, indices))
return true;
}
}
}
if (dump_enabled_p ())
dump_printf (MSG_MISSED_OPTIMIZATION,
"permutaion op not supported by target.\n");
return false;
}
bool
vect_store_lanes_supported (tree vectype, unsigned HOST_WIDE_INT count,
bool masked_p)
{
if (masked_p)
return vect_lanes_optab_supported_p ("vec_mask_store_lanes",
vec_mask_store_lanes_optab,
vectype, count);
else
return vect_lanes_optab_supported_p ("vec_store_lanes",
vec_store_lanes_optab,
vectype, count);
}
void
vect_permute_store_chain (vec<tree> dr_chain,
unsigned int length,
gimple *stmt,
gimple_stmt_iterator *gsi,
vec<tree> *result_chain)
{
tree vect1, vect2, high, low;
gimple *perm_stmt;
tree vectype = STMT_VINFO_VECTYPE (vinfo_for_stmt (stmt));
tree perm_mask_low, perm_mask_high;
tree data_ref;
tree perm3_mask_low, perm3_mask_high;
unsigned int i, j, n, log_length = exact_log2 (length);
result_chain->quick_grow (length);
memcpy (result_chain->address (), dr_chain.address (),
length * sizeof (tree));
if (length == 3)
{
unsigned int nelt = TYPE_VECTOR_SUBPARTS (vectype).to_constant ();
unsigned int j0 = 0, j1 = 0, j2 = 0;
vec_perm_builder sel (nelt, nelt, 1);
sel.quick_grow (nelt);
vec_perm_indices indices;
for (j = 0; j < 3; j++)
{
int nelt0 = ((3 - j) * nelt) % 3;
int nelt1 = ((3 - j) * nelt + 1) % 3;
int nelt2 = ((3 - j) * nelt + 2) % 3;
for (i = 0; i < nelt; i++)
{
if (3 * i + nelt0 < nelt)
sel[3 * i + nelt0] = j0++;
if (3 * i + nelt1 < nelt)
sel[3 * i + nelt1] = nelt + j1++;
if (3 * i + nelt2 < nelt)
sel[3 * i + nelt2] = 0;
}
indices.new_vector (sel, 2, nelt);
perm3_mask_low = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0; i < nelt; i++)
{
if (3 * i + nelt0 < nelt)
sel[3 * i + nelt0] = 3 * i + nelt0;
if (3 * i + nelt1 < nelt)
sel[3 * i + nelt1] = 3 * i + nelt1;
if (3 * i + nelt2 < nelt)
sel[3 * i + nelt2] = nelt + j2++;
}
indices.new_vector (sel, 2, nelt);
perm3_mask_high = vect_gen_perm_mask_checked (vectype, indices);
vect1 = dr_chain[0];
vect2 = dr_chain[1];
data_ref = make_temp_ssa_name (vectype, NULL, "vect_shuffle3_low");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR, vect1,
vect2, perm3_mask_low);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
vect1 = data_ref;
vect2 = dr_chain[2];
data_ref = make_temp_ssa_name (vectype, NULL, "vect_shuffle3_high");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR, vect1,
vect2, perm3_mask_high);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
(*result_chain)[j] = data_ref;
}
}
else
{
gcc_assert (pow2p_hwi (length));
poly_uint64 nelt = TYPE_VECTOR_SUBPARTS (vectype);
vec_perm_builder sel (nelt, 2, 3);
sel.quick_grow (6);
for (i = 0; i < 3; i++)
{
sel[i * 2] = i;
sel[i * 2 + 1] = i + nelt;
}
vec_perm_indices indices (sel, 2, nelt);
perm_mask_high = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0; i < 6; i++)
sel[i] += exact_div (nelt, 2);
indices.new_vector (sel, 2, nelt);
perm_mask_low = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0, n = log_length; i < n; i++)
{
for (j = 0; j < length/2; j++)
{
vect1 = dr_chain[j];
vect2 = dr_chain[j+length/2];
high = make_temp_ssa_name (vectype, NULL, "vect_inter_high");
perm_stmt = gimple_build_assign (high, VEC_PERM_EXPR, vect1,
vect2, perm_mask_high);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
(*result_chain)[2*j] = high;
low = make_temp_ssa_name (vectype, NULL, "vect_inter_low");
perm_stmt = gimple_build_assign (low, VEC_PERM_EXPR, vect1,
vect2, perm_mask_low);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
(*result_chain)[2*j+1] = low;
}
memcpy (dr_chain.address (), result_chain->address (),
length * sizeof (tree));
}
}
}
tree
vect_setup_realignment (gimple *stmt, gimple_stmt_iterator *gsi,
tree *realignment_token,
enum dr_alignment_support alignment_support_scheme,
tree init_addr,
struct loop **at_loop)
{
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
tree vectype = STMT_VINFO_VECTYPE (stmt_info);
loop_vec_info loop_vinfo = STMT_VINFO_LOOP_VINFO (stmt_info);
struct data_reference *dr = STMT_VINFO_DATA_REF (stmt_info);
struct loop *loop = NULL;
edge pe = NULL;
tree scalar_dest = gimple_assign_lhs (stmt);
tree vec_dest;
gimple *inc;
tree ptr;
tree data_ref;
basic_block new_bb;
tree msq_init = NULL_TREE;
tree new_temp;
gphi *phi_stmt;
tree msq = NULL_TREE;
gimple_seq stmts = NULL;
bool inv_p;
bool compute_in_loop = false;
bool nested_in_vect_loop = false;
struct loop *containing_loop = (gimple_bb (stmt))->loop_father;
struct loop *loop_for_initial_load = NULL;
if (loop_vinfo)
{
loop = LOOP_VINFO_LOOP (loop_vinfo);
nested_in_vect_loop = nested_in_vect_loop_p (loop, stmt);
}
gcc_assert (alignment_support_scheme == dr_explicit_realign
|| alignment_support_scheme == dr_explicit_realign_optimized);
if (init_addr != NULL_TREE || !loop_vinfo)
{
compute_in_loop = true;
gcc_assert (alignment_support_scheme == dr_explicit_realign);
}
if (nested_in_vect_loop)
{
tree outerloop_step = STMT_VINFO_DR_STEP (stmt_info);
bool invariant_in_outerloop =
(tree_int_cst_compare (outerloop_step, size_zero_node) == 0);
loop_for_initial_load = (invariant_in_outerloop ? loop : loop->inner);
}
else
loop_for_initial_load = loop;
if (at_loop)
*at_loop = loop_for_initial_load;
if (loop_for_initial_load)
pe = loop_preheader_edge (loop_for_initial_load);
if (alignment_support_scheme == dr_explicit_realign_optimized)
{
gassign *new_stmt;
gcc_assert (!compute_in_loop);
vec_dest = vect_create_destination_var (scalar_dest, vectype);
ptr = vect_create_data_ref_ptr (stmt, vectype, loop_for_initial_load,
NULL_TREE, &init_addr, NULL, &inc,
true, &inv_p);
if (TREE_CODE (ptr) == SSA_NAME)
new_temp = copy_ssa_name (ptr);
else
new_temp = make_ssa_name (TREE_TYPE (ptr));
unsigned int align = DR_TARGET_ALIGNMENT (dr);
new_stmt = gimple_build_assign
(new_temp, BIT_AND_EXPR, ptr,
build_int_cst (TREE_TYPE (ptr), -(HOST_WIDE_INT) align));
new_bb = gsi_insert_on_edge_immediate (pe, new_stmt);
gcc_assert (!new_bb);
data_ref
= build2 (MEM_REF, TREE_TYPE (vec_dest), new_temp,
build_int_cst (reference_alias_ptr_type (DR_REF (dr)), 0));
vect_copy_ref_info (data_ref, DR_REF (dr));
new_stmt = gimple_build_assign (vec_dest, data_ref);
new_temp = make_ssa_name (vec_dest, new_stmt);
gimple_assign_set_lhs (new_stmt, new_temp);
if (pe)
{
new_bb = gsi_insert_on_edge_immediate (pe, new_stmt);
gcc_assert (!new_bb);
}
else
gsi_insert_before (gsi, new_stmt, GSI_SAME_STMT);
msq_init = gimple_assign_lhs (new_stmt);
}
if (targetm.vectorize.builtin_mask_for_load)
{
gcall *new_stmt;
tree builtin_decl;
if (!init_addr)
{
init_addr = vect_create_addr_base_for_vector_ref (stmt, &stmts,
NULL_TREE);
if (loop)
{
pe = loop_preheader_edge (loop);
new_bb = gsi_insert_seq_on_edge_immediate (pe, stmts);
gcc_assert (!new_bb);
}
else
gsi_insert_seq_before (gsi, stmts, GSI_SAME_STMT);
}
builtin_decl = targetm.vectorize.builtin_mask_for_load ();
new_stmt = gimple_build_call (builtin_decl, 1, init_addr);
vec_dest =
vect_create_destination_var (scalar_dest,
gimple_call_return_type (new_stmt));
new_temp = make_ssa_name (vec_dest, new_stmt);
gimple_call_set_lhs (new_stmt, new_temp);
if (compute_in_loop)
gsi_insert_before (gsi, new_stmt, GSI_SAME_STMT);
else
{
pe = loop_preheader_edge (loop);
new_bb = gsi_insert_on_edge_immediate (pe, new_stmt);
gcc_assert (!new_bb);
}
*realignment_token = gimple_call_lhs (new_stmt);
gcc_assert (TREE_READONLY (builtin_decl));
}
if (alignment_support_scheme == dr_explicit_realign)
return msq;
gcc_assert (!compute_in_loop);
gcc_assert (alignment_support_scheme == dr_explicit_realign_optimized);
pe = loop_preheader_edge (containing_loop);
vec_dest = vect_create_destination_var (scalar_dest, vectype);
msq = make_ssa_name (vec_dest);
phi_stmt = create_phi_node (msq, containing_loop->header);
add_phi_arg (phi_stmt, msq_init, pe, UNKNOWN_LOCATION);
return msq;
}
bool
vect_grouped_load_supported (tree vectype, bool single_element_p,
unsigned HOST_WIDE_INT count)
{
machine_mode mode = TYPE_MODE (vectype);
if (single_element_p && maybe_gt (count, TYPE_VECTOR_SUBPARTS (vectype)))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"single-element interleaving not supported "
"for not adjacent vector loads\n");
return false;
}
if (count != 3 && exact_log2 (count) == -1)
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"the size of the group of accesses"
" is not a power of 2 or not equal to 3\n");
return false;
}
if (VECTOR_MODE_P (mode))
{
unsigned int i, j;
if (count == 3)
{
unsigned int nelt;
if (!GET_MODE_NUNITS (mode).is_constant (&nelt))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"cannot handle groups of 3 loads for"
" variable-length vectors\n");
return false;
}
vec_perm_builder sel (nelt, nelt, 1);
sel.quick_grow (nelt);
vec_perm_indices indices;
unsigned int k;
for (k = 0; k < 3; k++)
{
for (i = 0; i < nelt; i++)
if (3 * i + k < 2 * nelt)
sel[i] = 3 * i + k;
else
sel[i] = 0;
indices.new_vector (sel, 2, nelt);
if (!can_vec_perm_const_p (mode, indices))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"shuffle of 3 loads is not supported by"
" target\n");
return false;
}
for (i = 0, j = 0; i < nelt; i++)
if (3 * i + k < 2 * nelt)
sel[i] = i;
else
sel[i] = nelt + ((nelt + k) % 3) + 3 * (j++);
indices.new_vector (sel, 2, nelt);
if (!can_vec_perm_const_p (mode, indices))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"shuffle of 3 loads is not supported by"
" target\n");
return false;
}
}
return true;
}
else
{
gcc_assert (pow2p_hwi (count));
poly_uint64 nelt = GET_MODE_NUNITS (mode);
vec_perm_builder sel (nelt, 1, 3);
sel.quick_grow (3);
for (i = 0; i < 3; i++)
sel[i] = i * 2;
vec_perm_indices indices (sel, 2, nelt);
if (can_vec_perm_const_p (mode, indices))
{
for (i = 0; i < 3; i++)
sel[i] = i * 2 + 1;
indices.new_vector (sel, 2, nelt);
if (can_vec_perm_const_p (mode, indices))
return true;
}
}
}
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"extract even/odd not supported by target\n");
return false;
}
bool
vect_load_lanes_supported (tree vectype, unsigned HOST_WIDE_INT count,
bool masked_p)
{
if (masked_p)
return vect_lanes_optab_supported_p ("vec_mask_load_lanes",
vec_mask_load_lanes_optab,
vectype, count);
else
return vect_lanes_optab_supported_p ("vec_load_lanes",
vec_load_lanes_optab,
vectype, count);
}
static void
vect_permute_load_chain (vec<tree> dr_chain,
unsigned int length,
gimple *stmt,
gimple_stmt_iterator *gsi,
vec<tree> *result_chain)
{
tree data_ref, first_vect, second_vect;
tree perm_mask_even, perm_mask_odd;
tree perm3_mask_low, perm3_mask_high;
gimple *perm_stmt;
tree vectype = STMT_VINFO_VECTYPE (vinfo_for_stmt (stmt));
unsigned int i, j, log_length = exact_log2 (length);
result_chain->quick_grow (length);
memcpy (result_chain->address (), dr_chain.address (),
length * sizeof (tree));
if (length == 3)
{
unsigned nelt = TYPE_VECTOR_SUBPARTS (vectype).to_constant ();
unsigned int k;
vec_perm_builder sel (nelt, nelt, 1);
sel.quick_grow (nelt);
vec_perm_indices indices;
for (k = 0; k < 3; k++)
{
for (i = 0; i < nelt; i++)
if (3 * i + k < 2 * nelt)
sel[i] = 3 * i + k;
else
sel[i] = 0;
indices.new_vector (sel, 2, nelt);
perm3_mask_low = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0, j = 0; i < nelt; i++)
if (3 * i + k < 2 * nelt)
sel[i] = i;
else
sel[i] = nelt + ((nelt + k) % 3) + 3 * (j++);
indices.new_vector (sel, 2, nelt);
perm3_mask_high = vect_gen_perm_mask_checked (vectype, indices);
first_vect = dr_chain[0];
second_vect = dr_chain[1];
data_ref = make_temp_ssa_name (vectype, NULL, "vect_shuffle3_low");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR, first_vect,
second_vect, perm3_mask_low);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
first_vect = data_ref;
second_vect = dr_chain[2];
data_ref = make_temp_ssa_name (vectype, NULL, "vect_shuffle3_high");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR, first_vect,
second_vect, perm3_mask_high);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
(*result_chain)[k] = data_ref;
}
}
else
{
gcc_assert (pow2p_hwi (length));
poly_uint64 nelt = TYPE_VECTOR_SUBPARTS (vectype);
vec_perm_builder sel (nelt, 1, 3);
sel.quick_grow (3);
for (i = 0; i < 3; ++i)
sel[i] = i * 2;
vec_perm_indices indices (sel, 2, nelt);
perm_mask_even = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0; i < 3; ++i)
sel[i] = i * 2 + 1;
indices.new_vector (sel, 2, nelt);
perm_mask_odd = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0; i < log_length; i++)
{
for (j = 0; j < length; j += 2)
{
first_vect = dr_chain[j];
second_vect = dr_chain[j+1];
data_ref = make_temp_ssa_name (vectype, NULL, "vect_perm_even");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR,
first_vect, second_vect,
perm_mask_even);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
(*result_chain)[j/2] = data_ref;
data_ref = make_temp_ssa_name (vectype, NULL, "vect_perm_odd");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR,
first_vect, second_vect,
perm_mask_odd);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
(*result_chain)[j/2+length/2] = data_ref;
}
memcpy (dr_chain.address (), result_chain->address (),
length * sizeof (tree));
}
}
}
static bool
vect_shift_permute_load_chain (vec<tree> dr_chain,
unsigned int length,
gimple *stmt,
gimple_stmt_iterator *gsi,
vec<tree> *result_chain)
{
tree vect[3], vect_shift[3], data_ref, first_vect, second_vect;
tree perm2_mask1, perm2_mask2, perm3_mask;
tree select_mask, shift1_mask, shift2_mask, shift3_mask, shift4_mask;
gimple *perm_stmt;
tree vectype = STMT_VINFO_VECTYPE (vinfo_for_stmt (stmt));
unsigned int i;
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
loop_vec_info loop_vinfo = STMT_VINFO_LOOP_VINFO (stmt_info);
unsigned HOST_WIDE_INT nelt, vf;
if (!TYPE_VECTOR_SUBPARTS (vectype).is_constant (&nelt)
|| !LOOP_VINFO_VECT_FACTOR (loop_vinfo).is_constant (&vf))
return false;
vec_perm_builder sel (nelt, nelt, 1);
sel.quick_grow (nelt);
result_chain->quick_grow (length);
memcpy (result_chain->address (), dr_chain.address (),
length * sizeof (tree));
if (pow2p_hwi (length) && vf > 4)
{
unsigned int j, log_length = exact_log2 (length);
for (i = 0; i < nelt / 2; ++i)
sel[i] = i * 2;
for (i = 0; i < nelt / 2; ++i)
sel[nelt / 2 + i] = i * 2 + 1;
vec_perm_indices indices (sel, 2, nelt);
if (!can_vec_perm_const_p (TYPE_MODE (vectype), indices))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"shuffle of 2 fields structure is not \
supported by target\n");
return false;
}
perm2_mask1 = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0; i < nelt / 2; ++i)
sel[i] = i * 2 + 1;
for (i = 0; i < nelt / 2; ++i)
sel[nelt / 2 + i] = i * 2;
indices.new_vector (sel, 2, nelt);
if (!can_vec_perm_const_p (TYPE_MODE (vectype), indices))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"shuffle of 2 fields structure is not \
supported by target\n");
return false;
}
perm2_mask2 = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0; i < nelt; i++)
sel[i] = nelt / 2 + i;
indices.new_vector (sel, 2, nelt);
if (!can_vec_perm_const_p (TYPE_MODE (vectype), indices))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"shift permutation is not supported by target\n");
return false;
}
shift1_mask = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0; i < nelt / 2; i++)
sel[i] = i;
for (i = nelt / 2; i < nelt; i++)
sel[i] = nelt + i;
indices.new_vector (sel, 2, nelt);
if (!can_vec_perm_const_p (TYPE_MODE (vectype), indices))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"select is not supported by target\n");
return false;
}
select_mask = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0; i < log_length; i++)
{
for (j = 0; j < length; j += 2)
{
first_vect = dr_chain[j];
second_vect = dr_chain[j + 1];
data_ref = make_temp_ssa_name (vectype, NULL, "vect_shuffle2");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR,
first_vect, first_vect,
perm2_mask1);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
vect[0] = data_ref;
data_ref = make_temp_ssa_name (vectype, NULL, "vect_shuffle2");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR,
second_vect, second_vect,
perm2_mask2);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
vect[1] = data_ref;
data_ref = make_temp_ssa_name (vectype, NULL, "vect_shift");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR,
vect[0], vect[1], shift1_mask);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
(*result_chain)[j/2 + length/2] = data_ref;
data_ref = make_temp_ssa_name (vectype, NULL, "vect_select");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR,
vect[0], vect[1], select_mask);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
(*result_chain)[j/2] = data_ref;
}
memcpy (dr_chain.address (), result_chain->address (),
length * sizeof (tree));
}
return true;
}
if (length == 3 && vf > 2)
{
unsigned int k = 0, l = 0;
for (i = 0; i < nelt; i++)
{
if (3 * k + (l % 3) >= nelt)
{
k = 0;
l += (3 - (nelt % 3));
}
sel[i] = 3 * k + (l % 3);
k++;
}
vec_perm_indices indices (sel, 2, nelt);
if (!can_vec_perm_const_p (TYPE_MODE (vectype), indices))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"shuffle of 3 fields structure is not \
supported by target\n");
return false;
}
perm3_mask = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0; i < nelt; i++)
sel[i] = 2 * (nelt / 3) + (nelt % 3) + i;
indices.new_vector (sel, 2, nelt);
if (!can_vec_perm_const_p (TYPE_MODE (vectype), indices))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"shift permutation is not supported by target\n");
return false;
}
shift1_mask = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0; i < nelt; i++)
sel[i] = 2 * (nelt / 3) + 1 + i;
indices.new_vector (sel, 2, nelt);
if (!can_vec_perm_const_p (TYPE_MODE (vectype), indices))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"shift permutation is not supported by target\n");
return false;
}
shift2_mask = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0; i < nelt; i++)
sel[i] = (nelt / 3) + (nelt % 3) / 2 + i;
indices.new_vector (sel, 2, nelt);
if (!can_vec_perm_const_p (TYPE_MODE (vectype), indices))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"shift permutation is not supported by target\n");
return false;
}
shift3_mask = vect_gen_perm_mask_checked (vectype, indices);
for (i = 0; i < nelt; i++)
sel[i] = 2 * (nelt / 3) + (nelt % 3) / 2 + i;
indices.new_vector (sel, 2, nelt);
if (!can_vec_perm_const_p (TYPE_MODE (vectype), indices))
{
if (dump_enabled_p ())
dump_printf_loc (MSG_MISSED_OPTIMIZATION, vect_location,
"shift permutation is not supported by target\n");
return false;
}
shift4_mask = vect_gen_perm_mask_checked (vectype, indices);
for (k = 0; k < 3; k++)
{
data_ref = make_temp_ssa_name (vectype, NULL, "vect_shuffle3");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR,
dr_chain[k], dr_chain[k],
perm3_mask);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
vect[k] = data_ref;
}
for (k = 0; k < 3; k++)
{
data_ref = make_temp_ssa_name (vectype, NULL, "vect_shift1");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR,
vect[k % 3], vect[(k + 1) % 3],
shift1_mask);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
vect_shift[k] = data_ref;
}
for (k = 0; k < 3; k++)
{
data_ref = make_temp_ssa_name (vectype, NULL, "vect_shift2");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR,
vect_shift[(4 - k) % 3],
vect_shift[(3 - k) % 3],
shift2_mask);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
vect[k] = data_ref;
}
(*result_chain)[3 - (nelt % 3)] = vect[2];
data_ref = make_temp_ssa_name (vectype, NULL, "vect_shift3");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR, vect[0],
vect[0], shift3_mask);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
(*result_chain)[nelt % 3] = data_ref;
data_ref = make_temp_ssa_name (vectype, NULL, "vect_shift4");
perm_stmt = gimple_build_assign (data_ref, VEC_PERM_EXPR, vect[1],
vect[1], shift4_mask);
vect_finish_stmt_generation (stmt, perm_stmt, gsi);
(*result_chain)[0] = data_ref;
return true;
}
return false;
}
void
vect_transform_grouped_load (gimple *stmt, vec<tree> dr_chain, int size,
gimple_stmt_iterator *gsi)
{
machine_mode mode;
vec<tree> result_chain = vNULL;
result_chain.create (size);
mode = TYPE_MODE (STMT_VINFO_VECTYPE (vinfo_for_stmt (stmt)));
if (targetm.sched.reassociation_width (VEC_PERM_EXPR, mode) > 1
|| pow2p_hwi (size)
|| !vect_shift_permute_load_chain (dr_chain, size, stmt,
gsi, &result_chain))
vect_permute_load_chain (dr_chain, size, stmt, gsi, &result_chain);
vect_record_grouped_load_vectors (stmt, result_chain);
result_chain.release ();
}
void
vect_record_grouped_load_vectors (gimple *stmt, vec<tree> result_chain)
{
gimple *first_stmt = GROUP_FIRST_ELEMENT (vinfo_for_stmt (stmt));
gimple *next_stmt, *new_stmt;
unsigned int i, gap_count;
tree tmp_data_ref;
next_stmt = first_stmt;
gap_count = 1;
FOR_EACH_VEC_ELT (result_chain, i, tmp_data_ref)
{
if (!next_stmt)
break;
if (next_stmt != first_stmt
&& gap_count < GROUP_GAP (vinfo_for_stmt (next_stmt)))
{
gap_count++;
continue;
}
while (next_stmt)
{
new_stmt = SSA_NAME_DEF_STMT (tmp_data_ref);
if (!STMT_VINFO_VEC_STMT (vinfo_for_stmt (next_stmt)))
STMT_VINFO_VEC_STMT (vinfo_for_stmt (next_stmt)) = new_stmt;
else
{
if (!GROUP_SAME_DR_STMT (vinfo_for_stmt (next_stmt)))
{
gimple *prev_stmt =
STMT_VINFO_VEC_STMT (vinfo_for_stmt (next_stmt));
gimple *rel_stmt =
STMT_VINFO_RELATED_STMT (vinfo_for_stmt (prev_stmt));
while (rel_stmt)
{
prev_stmt = rel_stmt;
rel_stmt =
STMT_VINFO_RELATED_STMT (vinfo_for_stmt (rel_stmt));
}
STMT_VINFO_RELATED_STMT (vinfo_for_stmt (prev_stmt)) =
new_stmt;
}
}
next_stmt = GROUP_NEXT_ELEMENT (vinfo_for_stmt (next_stmt));
gap_count = 1;
if (!next_stmt || !GROUP_SAME_DR_STMT (vinfo_for_stmt (next_stmt)))
break;
}
}
}
bool
vect_can_force_dr_alignment_p (const_tree decl, unsigned int alignment)
{
if (!VAR_P (decl))
return false;
if (decl_in_symtab_p (decl)
&& !symtab_node::get (decl)->can_increase_alignment_p ())
return false;
if (TREE_STATIC (decl))
return (alignment <= MAX_OFILE_ALIGNMENT);
else
return (alignment <= MAX_STACK_ALIGNMENT);
}
enum dr_alignment_support
vect_supportable_dr_alignment (struct data_reference *dr,
bool check_aligned_accesses)
{
gimple *stmt = DR_STMT (dr);
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
tree vectype = STMT_VINFO_VECTYPE (stmt_info);
machine_mode mode = TYPE_MODE (vectype);
loop_vec_info loop_vinfo = STMT_VINFO_LOOP_VINFO (stmt_info);
struct loop *vect_loop = NULL;
bool nested_in_vect_loop = false;
if (aligned_access_p (dr) && !check_aligned_accesses)
return dr_aligned;
if (is_gimple_call (stmt)
&& gimple_call_internal_p (stmt)
&& (gimple_call_internal_fn (stmt) == IFN_MASK_LOAD
|| gimple_call_internal_fn (stmt) == IFN_MASK_STORE))
return dr_unaligned_supported;
if (loop_vinfo)
{
vect_loop = LOOP_VINFO_LOOP (loop_vinfo);
nested_in_vect_loop = nested_in_vect_loop_p (vect_loop, stmt);
}
if (DR_IS_READ (dr))
{
bool is_packed = false;
tree type = (TREE_TYPE (DR_REF (dr)));
if (optab_handler (vec_realign_load_optab, mode) != CODE_FOR_nothing
&& (!targetm.vectorize.builtin_mask_for_load
|| targetm.vectorize.builtin_mask_for_load ()))
{
tree vectype = STMT_VINFO_VECTYPE (stmt_info);
if (loop_vinfo
&& STMT_SLP_TYPE (stmt_info)
&& !multiple_p (LOOP_VINFO_VECT_FACTOR (loop_vinfo)
* GROUP_SIZE (vinfo_for_stmt
(GROUP_FIRST_ELEMENT (stmt_info))),
TYPE_VECTOR_SUBPARTS (vectype)))
;
else if (!loop_vinfo
|| (nested_in_vect_loop
&& maybe_ne (TREE_INT_CST_LOW (DR_STEP (dr)),
GET_MODE_SIZE (TYPE_MODE (vectype)))))
return dr_explicit_realign;
else
return dr_explicit_realign_optimized;
}
if (!known_alignment_for_access_p (dr))
is_packed = not_size_aligned (DR_REF (dr));
if (targetm.vectorize.support_vector_misalignment
(mode, type, DR_MISALIGNMENT (dr), is_packed))
return dr_unaligned_supported;
}
else
{
bool is_packed = false;
tree type = (TREE_TYPE (DR_REF (dr)));
if (!known_alignment_for_access_p (dr))
is_packed = not_size_aligned (DR_REF (dr));
if (targetm.vectorize.support_vector_misalignment
(mode, type, DR_MISALIGNMENT (dr), is_packed))
return dr_unaligned_supported;
}
return dr_unaligned_unsupported;
}
