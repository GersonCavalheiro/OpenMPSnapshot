#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "tree.h"
#include "gimple.h"
#include "cfghooks.h"
#include "tree-pass.h"
#include "ssa.h"
#include "cgraph.h"
#include "gimple-pretty-print.h"
#include "fold-const.h"
#include "gimplify.h"
#include "gimple-iterator.h"
#include "gimplify-me.h"
#include "gimple-walk.h"
#include "stor-layout.h"
#include "tree-nested.h"
#include "tree-cfg.h"
#include "tree-ssa-loop-ivopts.h"
#include "tree-ssa-loop-manip.h"
#include "tree-ssa-loop-niter.h"
#include "tree-ssa-loop.h"
#include "tree-into-ssa.h"
#include "cfgloop.h"
#include "tree-scalar-evolution.h"
#include "langhooks.h"
#include "tree-vectorizer.h"
#include "tree-hasher.h"
#include "tree-parloops.h"
#include "omp-general.h"
#include "omp-low.h"
#include "tree-ssa.h"
#include "params.h"
#include "params-enum.h"
#include "tree-ssa-alias.h"
#include "tree-eh.h"
#include "gomp-constants.h"
#include "tree-dfa.h"
#include "stringpool.h"
#include "attribs.h"
#define MIN_PER_THREAD PARAM_VALUE (PARAM_PARLOOPS_MIN_PER_THREAD)
struct reduction_info
{
gimple *reduc_stmt;		
gimple *reduc_phi;		
enum tree_code reduction_code;
unsigned reduc_version;	
gphi *keep_res;		
tree initial_value;		
tree field;			
tree reduc_addr;		
tree init;			
gphi *new_phi;		
};
struct reduction_hasher : free_ptr_hash <reduction_info>
{
static inline hashval_t hash (const reduction_info *);
static inline bool equal (const reduction_info *, const reduction_info *);
};
inline bool
reduction_hasher::equal (const reduction_info *a, const reduction_info *b)
{
return (a->reduc_phi == b->reduc_phi);
}
inline hashval_t
reduction_hasher::hash (const reduction_info *a)
{
return a->reduc_version;
}
typedef hash_table<reduction_hasher> reduction_info_table_type;
static struct reduction_info *
reduction_phi (reduction_info_table_type *reduction_list, gimple *phi)
{
struct reduction_info tmpred, *red;
if (reduction_list->elements () == 0 || phi == NULL)
return NULL;
if (gimple_uid (phi) == (unsigned int)-1
|| gimple_uid (phi) == 0)
return NULL;
tmpred.reduc_phi = phi;
tmpred.reduc_version = gimple_uid (phi);
red = reduction_list->find (&tmpred);
gcc_assert (red == NULL || red->reduc_phi == phi);
return red;
}
struct name_to_copy_elt
{
unsigned version;	
tree new_name;	
tree field;		
};
struct name_to_copy_hasher : free_ptr_hash <name_to_copy_elt>
{
static inline hashval_t hash (const name_to_copy_elt *);
static inline bool equal (const name_to_copy_elt *, const name_to_copy_elt *);
};
inline bool
name_to_copy_hasher::equal (const name_to_copy_elt *a, const name_to_copy_elt *b)
{
return a->version == b->version;
}
inline hashval_t
name_to_copy_hasher::hash (const name_to_copy_elt *a)
{
return (hashval_t) a->version;
}
typedef hash_table<name_to_copy_hasher> name_to_copy_table_type;
typedef struct lambda_trans_matrix_s
{
lambda_matrix matrix;
int rowsize;
int colsize;
int denominator;
} *lambda_trans_matrix;
#define LTM_MATRIX(T) ((T)->matrix)
#define LTM_ROWSIZE(T) ((T)->rowsize)
#define LTM_COLSIZE(T) ((T)->colsize)
#define LTM_DENOMINATOR(T) ((T)->denominator)
static lambda_trans_matrix
lambda_trans_matrix_new (int colsize, int rowsize,
struct obstack * lambda_obstack)
{
lambda_trans_matrix ret;
ret = (lambda_trans_matrix)
obstack_alloc (lambda_obstack, sizeof (struct lambda_trans_matrix_s));
LTM_MATRIX (ret) = lambda_matrix_new (rowsize, colsize, lambda_obstack);
LTM_ROWSIZE (ret) = rowsize;
LTM_COLSIZE (ret) = colsize;
LTM_DENOMINATOR (ret) = 1;
return ret;
}
static void
lambda_matrix_vector_mult (lambda_matrix matrix, int m, int n,
lambda_vector vec, lambda_vector dest)
{
int i, j;
lambda_vector_clear (dest, m);
for (i = 0; i < m; i++)
for (j = 0; j < n; j++)
dest[i] += matrix[i][j] * vec[j];
}
static bool
lambda_transform_legal_p (lambda_trans_matrix trans,
int nb_loops,
vec<ddr_p> dependence_relations)
{
unsigned int i, j;
lambda_vector distres;
struct data_dependence_relation *ddr;
gcc_assert (LTM_COLSIZE (trans) == nb_loops
&& LTM_ROWSIZE (trans) == nb_loops);
if (dependence_relations.length () == 0)
return true;
ddr = dependence_relations[0];
if (ddr == NULL)
return true;
if (DDR_ARE_DEPENDENT (ddr) == chrec_dont_know)
return false;
distres = lambda_vector_new (nb_loops);
FOR_EACH_VEC_ELT (dependence_relations, i, ddr)
{
if (DDR_ARE_DEPENDENT (ddr) == chrec_known
|| (DR_IS_READ (DDR_A (ddr)) && DR_IS_READ (DDR_B (ddr))))
continue;
if (DDR_ARE_DEPENDENT (ddr) == chrec_dont_know)
return false;
if (DDR_NUM_DIST_VECTS (ddr) == 0)
return false;
for (j = 0; j < DDR_NUM_DIST_VECTS (ddr); j++)
{
lambda_matrix_vector_mult (LTM_MATRIX (trans), nb_loops, nb_loops,
DDR_DIST_VECT (ddr, j), distres);
if (!lambda_vector_lexico_pos (distres, nb_loops))
return false;
}
}
return true;
}
static bool
loop_parallel_p (struct loop *loop, struct obstack * parloop_obstack)
{
vec<ddr_p> dependence_relations;
vec<data_reference_p> datarefs;
lambda_trans_matrix trans;
bool ret = false;
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "Considering loop %d\n", loop->num);
if (!loop->inner)
fprintf (dump_file, "loop is innermost\n");
else
fprintf (dump_file, "loop NOT innermost\n");
}
auto_vec<loop_p, 3> loop_nest;
datarefs.create (10);
dependence_relations.create (100);
if (! compute_data_dependences_for_loop (loop, true, &loop_nest, &datarefs,
&dependence_relations))
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "  FAILED: cannot analyze data dependencies\n");
ret = false;
goto end;
}
if (dump_file && (dump_flags & TDF_DETAILS))
dump_data_dependence_relations (dump_file, dependence_relations);
trans = lambda_trans_matrix_new (1, 1, parloop_obstack);
LTM_MATRIX (trans)[0][0] = -1;
if (lambda_transform_legal_p (trans, 1, dependence_relations))
{
ret = true;
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "  SUCCESS: may be parallelized\n");
}
else if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file,
"  FAILED: data dependencies exist across iterations\n");
end:
free_dependence_relations (dependence_relations);
free_data_refs (datarefs);
return ret;
}
static inline bool
loop_has_blocks_with_irreducible_flag (struct loop *loop)
{
unsigned i;
basic_block *bbs = get_loop_body_in_dom_order (loop);
bool res = true;
for (i = 0; i < loop->num_nodes; i++)
if (bbs[i]->flags & BB_IRREDUCIBLE_LOOP)
goto end;
res = false;
end:
free (bbs);
return res;
}
static tree
take_address_of (tree obj, tree type, edge entry,
int_tree_htab_type *decl_address, gimple_stmt_iterator *gsi)
{
int uid;
tree *var_p, name, addr;
gassign *stmt;
gimple_seq stmts;
obj = unshare_expr (obj);
for (var_p = &obj;
handled_component_p (*var_p);
var_p = &TREE_OPERAND (*var_p, 0))
continue;
if (DECL_P (*var_p))
*var_p = build_simple_mem_ref (build_fold_addr_expr (*var_p));
uid = DECL_UID (TREE_OPERAND (TREE_OPERAND (*var_p, 0), 0));
int_tree_map elt;
elt.uid = uid;
int_tree_map *slot = decl_address->find_slot (elt, INSERT);
if (!slot->to)
{
if (gsi == NULL)
return NULL;
addr = TREE_OPERAND (*var_p, 0);
const char *obj_name
= get_name (TREE_OPERAND (TREE_OPERAND (*var_p, 0), 0));
if (obj_name)
name = make_temp_ssa_name (TREE_TYPE (addr), NULL, obj_name);
else
name = make_ssa_name (TREE_TYPE (addr));
stmt = gimple_build_assign (name, addr);
gsi_insert_on_edge_immediate (entry, stmt);
slot->uid = uid;
slot->to = name;
}
else
name = slot->to;
TREE_OPERAND (*var_p, 0) = name;
if (gsi == NULL)
return build_fold_addr_expr_with_type (obj, type);
name = force_gimple_operand (build_addr (obj),
&stmts, true, NULL_TREE);
if (!gimple_seq_empty_p (stmts))
gsi_insert_seq_before (gsi, stmts, GSI_SAME_STMT);
if (!useless_type_conversion_p (type, TREE_TYPE (name)))
{
name = force_gimple_operand (fold_convert (type, name), &stmts, true,
NULL_TREE);
if (!gimple_seq_empty_p (stmts))
gsi_insert_seq_before (gsi, stmts, GSI_SAME_STMT);
}
return name;
}
static tree
reduc_stmt_res (gimple *stmt)
{
return (gimple_code (stmt) == GIMPLE_PHI
? gimple_phi_result (stmt)
: gimple_assign_lhs (stmt));
}
int
initialize_reductions (reduction_info **slot, struct loop *loop)
{
tree init;
tree type, arg;
edge e;
struct reduction_info *const reduc = *slot;
type = TREE_TYPE (PHI_RESULT (reduc->reduc_phi));
init = omp_reduction_init_op (gimple_location (reduc->reduc_stmt),
reduc->reduction_code, type);
reduc->init = init;
e = loop_preheader_edge (loop);
arg = PHI_ARG_DEF_FROM_EDGE (reduc->reduc_phi, e);
SET_USE (PHI_ARG_DEF_PTR_FROM_EDGE
(reduc->reduc_phi, loop_preheader_edge (loop)), init);
reduc->initial_value = arg;
return 1;
}
struct elv_data
{
struct walk_stmt_info info;
edge entry;
int_tree_htab_type *decl_address;
gimple_stmt_iterator *gsi;
bool changed;
bool reset;
};
static tree
eliminate_local_variables_1 (tree *tp, int *walk_subtrees, void *data)
{
struct elv_data *const dta = (struct elv_data *) data;
tree t = *tp, var, addr, addr_type, type, obj;
if (DECL_P (t))
{
*walk_subtrees = 0;
if (!SSA_VAR_P (t) || DECL_EXTERNAL (t))
return NULL_TREE;
type = TREE_TYPE (t);
addr_type = build_pointer_type (type);
addr = take_address_of (t, addr_type, dta->entry, dta->decl_address,
dta->gsi);
if (dta->gsi == NULL && addr == NULL_TREE)
{
dta->reset = true;
return NULL_TREE;
}
*tp = build_simple_mem_ref (addr);
dta->changed = true;
return NULL_TREE;
}
if (TREE_CODE (t) == ADDR_EXPR)
{
if (!is_gimple_val (t))
return NULL_TREE;
*walk_subtrees = 0;
obj = TREE_OPERAND (t, 0);
var = get_base_address (obj);
if (!var || !SSA_VAR_P (var) || DECL_EXTERNAL (var))
return NULL_TREE;
addr_type = TREE_TYPE (t);
addr = take_address_of (obj, addr_type, dta->entry, dta->decl_address,
dta->gsi);
if (dta->gsi == NULL && addr == NULL_TREE)
{
dta->reset = true;
return NULL_TREE;
}
*tp = addr;
dta->changed = true;
return NULL_TREE;
}
if (!EXPR_P (t))
*walk_subtrees = 0;
return NULL_TREE;
}
static void
eliminate_local_variables_stmt (edge entry, gimple_stmt_iterator *gsi,
int_tree_htab_type *decl_address)
{
struct elv_data dta;
gimple *stmt = gsi_stmt (*gsi);
memset (&dta.info, '\0', sizeof (dta.info));
dta.entry = entry;
dta.decl_address = decl_address;
dta.changed = false;
dta.reset = false;
if (gimple_debug_bind_p (stmt))
{
dta.gsi = NULL;
walk_tree (gimple_debug_bind_get_value_ptr (stmt),
eliminate_local_variables_1, &dta.info, NULL);
if (dta.reset)
{
gimple_debug_bind_reset_value (stmt);
dta.changed = true;
}
}
else if (gimple_clobber_p (stmt))
{
unlink_stmt_vdef (stmt);
stmt = gimple_build_nop ();
gsi_replace (gsi, stmt, false);
dta.changed = true;
}
else
{
dta.gsi = gsi;
walk_gimple_op (stmt, eliminate_local_variables_1, &dta.info);
}
if (dta.changed)
update_stmt (stmt);
}
static void
eliminate_local_variables (edge entry, edge exit)
{
basic_block bb;
auto_vec<basic_block, 3> body;
unsigned i;
gimple_stmt_iterator gsi;
bool has_debug_stmt = false;
int_tree_htab_type decl_address (10);
basic_block entry_bb = entry->src;
basic_block exit_bb = exit->dest;
gather_blocks_in_sese_region (entry_bb, exit_bb, &body);
FOR_EACH_VEC_ELT (body, i, bb)
if (bb != entry_bb && bb != exit_bb)
{
for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi); gsi_next (&gsi))
if (is_gimple_debug (gsi_stmt (gsi)))
{
if (gimple_debug_bind_p (gsi_stmt (gsi)))
has_debug_stmt = true;
}
else
eliminate_local_variables_stmt (entry, &gsi, &decl_address);
}
if (has_debug_stmt)
FOR_EACH_VEC_ELT (body, i, bb)
if (bb != entry_bb && bb != exit_bb)
for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi); gsi_next (&gsi))
if (gimple_debug_bind_p (gsi_stmt (gsi)))
eliminate_local_variables_stmt (entry, &gsi, &decl_address);
}
static bool
expr_invariant_in_region_p (edge entry, edge exit, tree expr)
{
basic_block entry_bb = entry->src;
basic_block exit_bb = exit->dest;
basic_block def_bb;
if (is_gimple_min_invariant (expr))
return true;
if (TREE_CODE (expr) == SSA_NAME)
{
def_bb = gimple_bb (SSA_NAME_DEF_STMT (expr));
if (def_bb
&& dominated_by_p (CDI_DOMINATORS, def_bb, entry_bb)
&& !dominated_by_p (CDI_DOMINATORS, def_bb, exit_bb))
return false;
return true;
}
return false;
}
static tree
separate_decls_in_region_name (tree name, name_to_copy_table_type *name_copies,
int_tree_htab_type *decl_copies,
bool copy_name_p)
{
tree copy, var, var_copy;
unsigned idx, uid, nuid;
struct int_tree_map ielt;
struct name_to_copy_elt elt, *nelt;
name_to_copy_elt **slot;
int_tree_map *dslot;
if (TREE_CODE (name) != SSA_NAME)
return name;
idx = SSA_NAME_VERSION (name);
elt.version = idx;
slot = name_copies->find_slot_with_hash (&elt, idx,
copy_name_p ? INSERT : NO_INSERT);
if (slot && *slot)
return (*slot)->new_name;
if (copy_name_p)
{
copy = duplicate_ssa_name (name, NULL);
nelt = XNEW (struct name_to_copy_elt);
nelt->version = idx;
nelt->new_name = copy;
nelt->field = NULL_TREE;
*slot = nelt;
}
else
{
gcc_assert (!slot);
copy = name;
}
var = SSA_NAME_VAR (name);
if (!var)
return copy;
uid = DECL_UID (var);
ielt.uid = uid;
dslot = decl_copies->find_slot_with_hash (ielt, uid, INSERT);
if (!dslot->to)
{
var_copy = create_tmp_var (TREE_TYPE (var), get_name (var));
DECL_GIMPLE_REG_P (var_copy) = DECL_GIMPLE_REG_P (var);
dslot->uid = uid;
dslot->to = var_copy;
nuid = DECL_UID (var_copy);
ielt.uid = nuid;
dslot = decl_copies->find_slot_with_hash (ielt, nuid, INSERT);
gcc_assert (!dslot->to);
dslot->uid = nuid;
dslot->to = var_copy;
}
else
var_copy = dslot->to;
replace_ssa_name_symbol (copy, var_copy);
return copy;
}
static void
separate_decls_in_region_stmt (edge entry, edge exit, gimple *stmt,
name_to_copy_table_type *name_copies,
int_tree_htab_type *decl_copies)
{
use_operand_p use;
def_operand_p def;
ssa_op_iter oi;
tree name, copy;
bool copy_name_p;
FOR_EACH_PHI_OR_STMT_DEF (def, stmt, oi, SSA_OP_DEF)
{
name = DEF_FROM_PTR (def);
gcc_assert (TREE_CODE (name) == SSA_NAME);
copy = separate_decls_in_region_name (name, name_copies, decl_copies,
false);
gcc_assert (copy == name);
}
FOR_EACH_PHI_OR_STMT_USE (use, stmt, oi, SSA_OP_USE)
{
name = USE_FROM_PTR (use);
if (TREE_CODE (name) != SSA_NAME)
continue;
copy_name_p = expr_invariant_in_region_p (entry, exit, name);
copy = separate_decls_in_region_name (name, name_copies, decl_copies,
copy_name_p);
SET_USE (use, copy);
}
}
static bool
separate_decls_in_region_debug (gimple *stmt,
name_to_copy_table_type *name_copies,
int_tree_htab_type *decl_copies)
{
use_operand_p use;
ssa_op_iter oi;
tree var, name;
struct int_tree_map ielt;
struct name_to_copy_elt elt;
name_to_copy_elt **slot;
int_tree_map *dslot;
if (gimple_debug_bind_p (stmt))
var = gimple_debug_bind_get_var (stmt);
else if (gimple_debug_source_bind_p (stmt))
var = gimple_debug_source_bind_get_var (stmt);
else
return true;
if (TREE_CODE (var) == DEBUG_EXPR_DECL || TREE_CODE (var) == LABEL_DECL)
return true;
gcc_assert (DECL_P (var) && SSA_VAR_P (var));
ielt.uid = DECL_UID (var);
dslot = decl_copies->find_slot_with_hash (ielt, ielt.uid, NO_INSERT);
if (!dslot)
return true;
if (gimple_debug_bind_p (stmt))
gimple_debug_bind_set_var (stmt, dslot->to);
else if (gimple_debug_source_bind_p (stmt))
gimple_debug_source_bind_set_var (stmt, dslot->to);
FOR_EACH_PHI_OR_STMT_USE (use, stmt, oi, SSA_OP_USE)
{
name = USE_FROM_PTR (use);
if (TREE_CODE (name) != SSA_NAME)
continue;
elt.version = SSA_NAME_VERSION (name);
slot = name_copies->find_slot_with_hash (&elt, elt.version, NO_INSERT);
if (!slot)
{
gimple_debug_bind_reset_value (stmt);
update_stmt (stmt);
break;
}
SET_USE (use, (*slot)->new_name);
}
return false;
}
int
add_field_for_reduction (reduction_info **slot, tree type)
{
struct reduction_info *const red = *slot;
tree var = reduc_stmt_res (red->reduc_stmt);
tree field = build_decl (gimple_location (red->reduc_stmt), FIELD_DECL,
SSA_NAME_IDENTIFIER (var), TREE_TYPE (var));
insert_field_into_struct (type, field);
red->field = field;
return 1;
}
int
add_field_for_name (name_to_copy_elt **slot, tree type)
{
struct name_to_copy_elt *const elt = *slot;
tree name = ssa_name (elt->version);
tree field = build_decl (UNKNOWN_LOCATION,
FIELD_DECL, SSA_NAME_IDENTIFIER (name),
TREE_TYPE (name));
insert_field_into_struct (type, field);
elt->field = field;
return 1;
}
int
create_phi_for_local_result (reduction_info **slot, struct loop *loop)
{
struct reduction_info *const reduc = *slot;
edge e;
gphi *new_phi;
basic_block store_bb, continue_bb;
tree local_res;
source_location locus;
continue_bb = single_pred (loop->latch);
store_bb = FALLTHRU_EDGE (continue_bb)->dest;
if (EDGE_PRED (store_bb, 0) == FALLTHRU_EDGE (continue_bb))
e = EDGE_PRED (store_bb, 1);
else
e = EDGE_PRED (store_bb, 0);
tree lhs = reduc_stmt_res (reduc->reduc_stmt);
local_res = copy_ssa_name (lhs);
locus = gimple_location (reduc->reduc_stmt);
new_phi = create_phi_node (local_res, store_bb);
add_phi_arg (new_phi, reduc->init, e, locus);
add_phi_arg (new_phi, lhs, FALLTHRU_EDGE (continue_bb), locus);
reduc->new_phi = new_phi;
return 1;
}
struct clsn_data
{
tree store;
tree load;
basic_block store_bb;
basic_block load_bb;
};
int
create_call_for_reduction_1 (reduction_info **slot, struct clsn_data *clsn_data)
{
struct reduction_info *const reduc = *slot;
gimple_stmt_iterator gsi;
tree type = TREE_TYPE (PHI_RESULT (reduc->reduc_phi));
tree load_struct;
basic_block bb;
basic_block new_bb;
edge e;
tree t, addr, ref, x;
tree tmp_load, name;
gimple *load;
if (reduc->reduc_addr == NULL_TREE)
{
load_struct = build_simple_mem_ref (clsn_data->load);
t = build3 (COMPONENT_REF, type, load_struct, reduc->field, NULL_TREE);
addr = build_addr (t);
}
else
{
addr = reduc->reduc_addr;
tree res = PHI_RESULT (reduc->keep_res);
use_operand_p use_p;
gimple *stmt;
bool single_use_p = single_imm_use (res, &use_p, &stmt);
gcc_assert (single_use_p);
replace_uses_by (gimple_vdef (stmt),
gimple_vuse (stmt));
gimple_stmt_iterator gsi = gsi_for_stmt (stmt);
gsi_remove (&gsi, true);
}
bb = clsn_data->load_bb;
gsi = gsi_last_bb (bb);
e = split_block (bb, gsi_stmt (gsi));
new_bb = e->dest;
tmp_load = create_tmp_var (TREE_TYPE (TREE_TYPE (addr)));
tmp_load = make_ssa_name (tmp_load);
load = gimple_build_omp_atomic_load (tmp_load, addr);
SSA_NAME_DEF_STMT (tmp_load) = load;
gsi = gsi_start_bb (new_bb);
gsi_insert_after (&gsi, load, GSI_NEW_STMT);
e = split_block (new_bb, load);
new_bb = e->dest;
gsi = gsi_start_bb (new_bb);
ref = tmp_load;
x = fold_build2 (reduc->reduction_code,
TREE_TYPE (PHI_RESULT (reduc->new_phi)), ref,
PHI_RESULT (reduc->new_phi));
name = force_gimple_operand_gsi (&gsi, x, true, NULL_TREE, true,
GSI_CONTINUE_LINKING);
gsi_insert_after (&gsi, gimple_build_omp_atomic_store (name), GSI_NEW_STMT);
return 1;
}
static void
create_call_for_reduction (struct loop *loop,
reduction_info_table_type *reduction_list,
struct clsn_data *ld_st_data)
{
reduction_list->traverse <struct loop *, create_phi_for_local_result> (loop);
basic_block continue_bb = single_pred (loop->latch);
ld_st_data->load_bb = FALLTHRU_EDGE (continue_bb)->dest;
reduction_list
->traverse <struct clsn_data *, create_call_for_reduction_1> (ld_st_data);
}
int
create_loads_for_reductions (reduction_info **slot, struct clsn_data *clsn_data)
{
struct reduction_info *const red = *slot;
gimple *stmt;
gimple_stmt_iterator gsi;
tree type = TREE_TYPE (reduc_stmt_res (red->reduc_stmt));
tree load_struct;
tree name;
tree x;
if (red->keep_res == NULL)
return 1;
gsi = gsi_after_labels (clsn_data->load_bb);
load_struct = build_simple_mem_ref (clsn_data->load);
load_struct = build3 (COMPONENT_REF, type, load_struct, red->field,
NULL_TREE);
x = load_struct;
name = PHI_RESULT (red->keep_res);
stmt = gimple_build_assign (name, x);
gsi_insert_after (&gsi, stmt, GSI_NEW_STMT);
for (gsi = gsi_start_phis (gimple_bb (red->keep_res));
!gsi_end_p (gsi); gsi_next (&gsi))
if (gsi_stmt (gsi) == red->keep_res)
{
remove_phi_node (&gsi, false);
return 1;
}
gcc_unreachable ();
}
static void
create_final_loads_for_reduction (reduction_info_table_type *reduction_list,
struct clsn_data *ld_st_data)
{
gimple_stmt_iterator gsi;
tree t;
gimple *stmt;
gsi = gsi_after_labels (ld_st_data->load_bb);
t = build_fold_addr_expr (ld_st_data->store);
stmt = gimple_build_assign (ld_st_data->load, t);
gsi_insert_before (&gsi, stmt, GSI_NEW_STMT);
reduction_list
->traverse <struct clsn_data *, create_loads_for_reductions> (ld_st_data);
}
int
create_stores_for_reduction (reduction_info **slot, struct clsn_data *clsn_data)
{
struct reduction_info *const red = *slot;
tree t;
gimple *stmt;
gimple_stmt_iterator gsi;
tree type = TREE_TYPE (reduc_stmt_res (red->reduc_stmt));
gsi = gsi_last_bb (clsn_data->store_bb);
t = build3 (COMPONENT_REF, type, clsn_data->store, red->field, NULL_TREE);
stmt = gimple_build_assign (t, red->initial_value);
gsi_insert_after (&gsi, stmt, GSI_NEW_STMT);
return 1;
}
int
create_loads_and_stores_for_name (name_to_copy_elt **slot,
struct clsn_data *clsn_data)
{
struct name_to_copy_elt *const elt = *slot;
tree t;
gimple *stmt;
gimple_stmt_iterator gsi;
tree type = TREE_TYPE (elt->new_name);
tree load_struct;
gsi = gsi_last_bb (clsn_data->store_bb);
t = build3 (COMPONENT_REF, type, clsn_data->store, elt->field, NULL_TREE);
stmt = gimple_build_assign (t, ssa_name (elt->version));
gsi_insert_after (&gsi, stmt, GSI_NEW_STMT);
gsi = gsi_last_bb (clsn_data->load_bb);
load_struct = build_simple_mem_ref (clsn_data->load);
t = build3 (COMPONENT_REF, type, load_struct, elt->field, NULL_TREE);
stmt = gimple_build_assign (elt->new_name, t);
gsi_insert_after (&gsi, stmt, GSI_NEW_STMT);
return 1;
}
static void
separate_decls_in_region (edge entry, edge exit,
reduction_info_table_type *reduction_list,
tree *arg_struct, tree *new_arg_struct,
struct clsn_data *ld_st_data)
{
basic_block bb1 = split_edge (entry);
basic_block bb0 = single_pred (bb1);
name_to_copy_table_type name_copies (10);
int_tree_htab_type decl_copies (10);
unsigned i;
tree type, type_name, nvar;
gimple_stmt_iterator gsi;
struct clsn_data clsn_data;
auto_vec<basic_block, 3> body;
basic_block bb;
basic_block entry_bb = bb1;
basic_block exit_bb = exit->dest;
bool has_debug_stmt = false;
entry = single_succ_edge (entry_bb);
gather_blocks_in_sese_region (entry_bb, exit_bb, &body);
FOR_EACH_VEC_ELT (body, i, bb)
{
if (bb != entry_bb && bb != exit_bb)
{
for (gsi = gsi_start_phis (bb); !gsi_end_p (gsi); gsi_next (&gsi))
separate_decls_in_region_stmt (entry, exit, gsi_stmt (gsi),
&name_copies, &decl_copies);
for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi); gsi_next (&gsi))
{
gimple *stmt = gsi_stmt (gsi);
if (is_gimple_debug (stmt))
has_debug_stmt = true;
else
separate_decls_in_region_stmt (entry, exit, stmt,
&name_copies, &decl_copies);
}
}
}
if (has_debug_stmt)
FOR_EACH_VEC_ELT (body, i, bb)
if (bb != entry_bb && bb != exit_bb)
{
for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi);)
{
gimple *stmt = gsi_stmt (gsi);
if (is_gimple_debug (stmt))
{
if (separate_decls_in_region_debug (stmt, &name_copies,
&decl_copies))
{
gsi_remove (&gsi, true);
continue;
}
}
gsi_next (&gsi);
}
}
if (name_copies.elements () == 0 && reduction_list->elements () == 0)
{
*arg_struct = NULL;
*new_arg_struct = NULL;
}
else
{
type = lang_hooks.types.make_type (RECORD_TYPE);
type_name = build_decl (UNKNOWN_LOCATION,
TYPE_DECL, create_tmp_var_name (".paral_data"),
type);
TYPE_NAME (type) = type_name;
name_copies.traverse <tree, add_field_for_name> (type);
if (reduction_list && reduction_list->elements () > 0)
{
reduction_list->traverse <tree, add_field_for_reduction> (type);
}
layout_type (type);
*arg_struct = create_tmp_var (type, ".paral_data_store");
nvar = create_tmp_var (build_pointer_type (type), ".paral_data_load");
*new_arg_struct = make_ssa_name (nvar);
ld_st_data->store = *arg_struct;
ld_st_data->load = *new_arg_struct;
ld_st_data->store_bb = bb0;
ld_st_data->load_bb = bb1;
name_copies
.traverse <struct clsn_data *, create_loads_and_stores_for_name>
(ld_st_data);
if (reduction_list && reduction_list->elements () > 0)
{
reduction_list
->traverse <struct clsn_data *, create_stores_for_reduction>
(ld_st_data);
clsn_data.load = make_ssa_name (nvar);
clsn_data.load_bb = exit->dest;
clsn_data.store = ld_st_data->store;
create_final_loads_for_reduction (reduction_list, &clsn_data);
}
}
}
bool
parallelized_function_p (tree fndecl)
{
cgraph_node *node = cgraph_node::get (fndecl);
gcc_assert (node != NULL);
return node->parallelized_function;
}
static tree
create_loop_fn (location_t loc)
{
char buf[100];
char *tname;
tree decl, type, name, t;
struct function *act_cfun = cfun;
static unsigned loopfn_num;
loc = LOCATION_LOCUS (loc);
snprintf (buf, 100, "%s.$loopfn", current_function_name ());
ASM_FORMAT_PRIVATE_NAME (tname, buf, loopfn_num++);
clean_symbol_name (tname);
name = get_identifier (tname);
type = build_function_type_list (void_type_node, ptr_type_node, NULL_TREE);
decl = build_decl (loc, FUNCTION_DECL, name, type);
TREE_STATIC (decl) = 1;
TREE_USED (decl) = 1;
DECL_ARTIFICIAL (decl) = 1;
DECL_IGNORED_P (decl) = 0;
TREE_PUBLIC (decl) = 0;
DECL_UNINLINABLE (decl) = 1;
DECL_EXTERNAL (decl) = 0;
DECL_CONTEXT (decl) = NULL_TREE;
DECL_INITIAL (decl) = make_node (BLOCK);
BLOCK_SUPERCONTEXT (DECL_INITIAL (decl)) = decl;
t = build_decl (loc, RESULT_DECL, NULL_TREE, void_type_node);
DECL_ARTIFICIAL (t) = 1;
DECL_IGNORED_P (t) = 1;
DECL_RESULT (decl) = t;
t = build_decl (loc, PARM_DECL, get_identifier (".paral_data_param"),
ptr_type_node);
DECL_ARTIFICIAL (t) = 1;
DECL_ARG_TYPE (t) = ptr_type_node;
DECL_CONTEXT (t) = decl;
TREE_USED (t) = 1;
DECL_ARGUMENTS (decl) = t;
allocate_struct_function (decl, false);
set_cfun (act_cfun);
return decl;
}
static void
replace_uses_in_bb_by (tree name, tree val, basic_block bb)
{
gimple *use_stmt;
imm_use_iterator imm_iter;
FOR_EACH_IMM_USE_STMT (use_stmt, imm_iter, name)
{
if (gimple_bb (use_stmt) != bb)
continue;
use_operand_p use_p;
FOR_EACH_IMM_USE_ON_STMT (use_p, imm_iter)
SET_USE (use_p, val);
}
}
static void
transform_to_exit_first_loop_alt (struct loop *loop,
reduction_info_table_type *reduction_list,
tree bound)
{
basic_block header = loop->header;
basic_block latch = loop->latch;
edge exit = single_dom_exit (loop);
basic_block exit_block = exit->dest;
gcond *cond_stmt = as_a <gcond *> (last_stmt (exit->src));
tree control = gimple_cond_lhs (cond_stmt);
edge e;
rewrite_virtuals_into_loop_closed_ssa (loop);
basic_block new_header = split_block_before_cond_jump (exit->src);
edge edge_at_split = single_pred_edge (new_header);
edge entry = loop_preheader_edge (loop);
e = redirect_edge_and_branch (entry, new_header);
gcc_assert (e == entry);
edge post_inc_edge = single_succ_edge (latch);
e = redirect_edge_and_branch (post_inc_edge, new_header);
gcc_assert (e == post_inc_edge);
edge post_cond_edge = single_pred_edge (latch);
e = redirect_edge_and_branch (post_cond_edge, header);
gcc_assert (e == post_cond_edge);
e = redirect_edge_and_branch (edge_at_split, latch);
gcc_assert (e == edge_at_split);
gimple_cond_set_rhs (cond_stmt, bound);
update_stmt (cond_stmt);
vec<edge_var_map> *v = redirect_edge_var_map_vector (post_inc_edge);
edge_var_map *vm;
gphi_iterator gsi;
int i;
for (gsi = gsi_start_phis (header), i = 0;
!gsi_end_p (gsi) && v->iterate (i, &vm);
gsi_next (&gsi), i++)
{
gphi *phi = gsi.phi ();
tree res_a = PHI_RESULT (phi);
tree res_c = copy_ssa_name (res_a, phi);
gphi *nphi = create_phi_node (res_c, new_header);
replace_uses_in_bb_by (res_a, res_c, new_header);
add_phi_arg (phi, res_c, post_cond_edge, UNKNOWN_LOCATION);
tree res_b = redirect_edge_var_map_def (vm);
replace_uses_in_bb_by (res_b, res_c, exit_block);
struct reduction_info *red = reduction_phi (reduction_list, phi);
gcc_assert (virtual_operand_p (res_a)
|| res_a == control
|| red != NULL);
if (red)
{
red->reduc_phi = nphi;
gimple_set_uid (red->reduc_phi, red->reduc_version);
}
}
gcc_assert (gsi_end_p (gsi) && !v->iterate (i, &vm));
flush_pending_stmts (entry);
flush_pending_stmts (post_inc_edge);
basic_block new_exit_block = NULL;
if (!single_pred_p (exit->dest))
{
new_exit_block = split_edge (exit);
}
for (gphi_iterator gsi = gsi_start_phis (exit_block);
!gsi_end_p (gsi);
gsi_next (&gsi))
{
gphi *phi = gsi.phi ();
gphi *nphi = NULL;
tree res_z = PHI_RESULT (phi);
tree res_c;
if (new_exit_block != NULL)
{
edge succ_new_exit_block = single_succ_edge (new_exit_block);
edge pred_new_exit_block = single_pred_edge (new_exit_block);
tree res_y = copy_ssa_name (res_z, phi);
nphi = create_phi_node (res_y, new_exit_block);
res_c = PHI_ARG_DEF_FROM_EDGE (phi, succ_new_exit_block);
add_phi_arg (nphi, res_c, pred_new_exit_block, UNKNOWN_LOCATION);
add_phi_arg (phi, res_y, succ_new_exit_block, UNKNOWN_LOCATION);
}
else
res_c = PHI_ARG_DEF_FROM_EDGE (phi, exit);
if (virtual_operand_p (res_z))
continue;
gimple *reduc_phi = SSA_NAME_DEF_STMT (res_c);
struct reduction_info *red = reduction_phi (reduction_list, reduc_phi);
if (red != NULL)
red->keep_res = (nphi != NULL
? nphi
: phi);
}
loop->header = new_header;
free_dominance_info (CDI_DOMINATORS);
calculate_dominance_info (CDI_DOMINATORS);
checking_verify_ssa (true, true);
}
static bool
try_transform_to_exit_first_loop_alt (struct loop *loop,
reduction_info_table_type *reduction_list,
tree nit)
{
if (!gimple_seq_nondebug_singleton_p (bb_seq (loop->latch)))
return false;
if (phi_nodes (loop->latch) != NULL)
return false;
edge back = single_succ_edge (loop->latch);
edge exit = single_dom_exit (loop);
gcond *cond_stmt = as_a <gcond *> (last_stmt (exit->src));
tree control = gimple_cond_lhs (cond_stmt);
gphi *phi = as_a <gphi *> (SSA_NAME_DEF_STMT (control));
tree inc_res = gimple_phi_arg_def (phi, back->dest_idx);
if (gimple_bb (SSA_NAME_DEF_STMT (inc_res)) != loop->latch)
return false;
if (!single_pred_p (loop->latch)
|| single_pred (loop->latch) != exit->src)
return false;
tree alt_bound = NULL_TREE;
tree nit_type = TREE_TYPE (nit);
if (TREE_CODE (nit) == INTEGER_CST)
{
if (!tree_int_cst_equal (nit, TYPE_MAX_VALUE (nit_type)))
{
alt_bound = fold_build2_loc (UNKNOWN_LOCATION, PLUS_EXPR, nit_type,
nit, build_one_cst (nit_type));
gcc_assert (TREE_CODE (alt_bound) == INTEGER_CST);
transform_to_exit_first_loop_alt (loop, reduction_list, alt_bound);
return true;
}
else
{
return false;
}
}
gcc_assert (TREE_CODE (nit) == SSA_NAME);
widest_int nit_max;
if (!max_loop_iterations (loop, &nit_max))
return false;
widest_int type_max = wi::to_widest (TYPE_MAX_VALUE (nit_type));
if (nit_max >= type_max)
return false;
gimple *def = SSA_NAME_DEF_STMT (nit);
if (def
&& is_gimple_assign (def)
&& gimple_assign_rhs_code (def) == PLUS_EXPR)
{
tree op1 = gimple_assign_rhs1 (def);
tree op2 = gimple_assign_rhs2 (def);
if (integer_minus_onep (op1))
alt_bound = op2;
else if (integer_minus_onep (op2))
alt_bound = op1;
}
if (alt_bound == NULL_TREE)
{
alt_bound = fold_build2 (PLUS_EXPR, nit_type, nit,
build_int_cst_type (nit_type, 1));
gimple_stmt_iterator gsi = gsi_last_bb (loop_preheader_edge (loop)->src);
alt_bound
= force_gimple_operand_gsi (&gsi, alt_bound, true, NULL_TREE, false,
GSI_CONTINUE_LINKING);
}
transform_to_exit_first_loop_alt (loop, reduction_list, alt_bound);
return true;
}
static void
transform_to_exit_first_loop (struct loop *loop,
reduction_info_table_type *reduction_list,
tree nit)
{
basic_block *bbs, *nbbs, ex_bb, orig_header;
unsigned n;
bool ok;
edge exit = single_dom_exit (loop), hpred;
tree control, control_name, res, t;
gphi *phi, *nphi;
gassign *stmt;
gcond *cond_stmt, *cond_nit;
tree nit_1;
split_block_after_labels (loop->header);
orig_header = single_succ (loop->header);
hpred = single_succ_edge (loop->header);
cond_stmt = as_a <gcond *> (last_stmt (exit->src));
control = gimple_cond_lhs (cond_stmt);
gcc_assert (gimple_cond_rhs (cond_stmt) == nit);
for (gphi_iterator gsi = gsi_start_phis (loop->header);
!gsi_end_p (gsi);
gsi_next (&gsi))
{
phi = gsi.phi ();
res = PHI_RESULT (phi);
t = copy_ssa_name (res, phi);
SET_PHI_RESULT (phi, t);
nphi = create_phi_node (res, orig_header);
add_phi_arg (nphi, t, hpred, UNKNOWN_LOCATION);
if (res == control)
{
gimple_cond_set_lhs (cond_stmt, t);
update_stmt (cond_stmt);
control = t;
}
}
bbs = get_loop_body_in_dom_order (loop);
for (n = 0; bbs[n] != exit->src; n++)
continue;
nbbs = XNEWVEC (basic_block, n);
ok = gimple_duplicate_sese_tail (single_succ_edge (loop->header), exit,
bbs + 1, n, nbbs);
gcc_assert (ok);
free (bbs);
ex_bb = nbbs[0];
free (nbbs);
exit = single_dom_exit (loop);
control_name = NULL_TREE;
for (gphi_iterator gsi = gsi_start_phis (ex_bb);
!gsi_end_p (gsi); )
{
phi = gsi.phi ();
res = PHI_RESULT (phi);
if (virtual_operand_p (res))
{
gsi_next (&gsi);
continue;
}
if (reduction_list->elements () > 0)
{
struct reduction_info *red;
tree val = PHI_ARG_DEF_FROM_EDGE (phi, exit);
red = reduction_phi (reduction_list, SSA_NAME_DEF_STMT (val));
if (red)
{
red->keep_res = phi;
gsi_next (&gsi);
continue;
}
}
gcc_assert (control_name == NULL_TREE
&& SSA_NAME_VAR (res) == SSA_NAME_VAR (control));
control_name = res;
remove_phi_node (&gsi, false);
}
gcc_assert (control_name != NULL_TREE);
gimple_stmt_iterator gsi = gsi_after_labels (ex_bb);
cond_nit = as_a <gcond *> (last_stmt (exit->src));
nit_1 =  gimple_cond_rhs (cond_nit);
nit_1 = force_gimple_operand_gsi (&gsi,
fold_convert (TREE_TYPE (control_name), nit_1),
false, NULL_TREE, false, GSI_SAME_STMT);
stmt = gimple_build_assign (control_name, nit_1);
gsi_insert_before (&gsi, stmt, GSI_NEW_STMT);
}
static void
create_parallel_loop (struct loop *loop, tree loop_fn, tree data,
tree new_data, unsigned n_threads, location_t loc,
bool oacc_kernels_p)
{
gimple_stmt_iterator gsi;
basic_block for_bb, ex_bb, continue_bb;
tree t, param;
gomp_parallel *omp_par_stmt;
gimple *omp_return_stmt1, *omp_return_stmt2;
gimple *phi;
gcond *cond_stmt;
gomp_for *for_stmt;
gomp_continue *omp_cont_stmt;
tree cvar, cvar_init, initvar, cvar_next, cvar_base, type;
edge exit, nexit, guard, end, e;
if (oacc_kernels_p)
{
gcc_checking_assert (lookup_attribute ("oacc kernels",
DECL_ATTRIBUTES (cfun->decl)));
DECL_ATTRIBUTES (cfun->decl)
= tree_cons (get_identifier ("oacc kernels parallelized"),
NULL_TREE, DECL_ATTRIBUTES (cfun->decl));
}
else
{
basic_block bb = loop_preheader_edge (loop)->src;
basic_block paral_bb = single_pred (bb);
gsi = gsi_last_bb (paral_bb);
gcc_checking_assert (n_threads != 0);
t = build_omp_clause (loc, OMP_CLAUSE_NUM_THREADS);
OMP_CLAUSE_NUM_THREADS_EXPR (t)
= build_int_cst (integer_type_node, n_threads);
omp_par_stmt = gimple_build_omp_parallel (NULL, t, loop_fn, data);
gimple_set_location (omp_par_stmt, loc);
gsi_insert_after (&gsi, omp_par_stmt, GSI_NEW_STMT);
if (data)
{
gassign *assign_stmt;
gsi = gsi_after_labels (bb);
param = make_ssa_name (DECL_ARGUMENTS (loop_fn));
assign_stmt = gimple_build_assign (param, build_fold_addr_expr (data));
gsi_insert_before (&gsi, assign_stmt, GSI_SAME_STMT);
assign_stmt = gimple_build_assign (new_data,
fold_convert (TREE_TYPE (new_data), param));
gsi_insert_before (&gsi, assign_stmt, GSI_SAME_STMT);
}
bb = split_loop_exit_edge (single_dom_exit (loop));
gsi = gsi_last_bb (bb);
omp_return_stmt1 = gimple_build_omp_return (false);
gimple_set_location (omp_return_stmt1, loc);
gsi_insert_after (&gsi, omp_return_stmt1, GSI_NEW_STMT);
}
gcc_assert (loop->header == single_dom_exit (loop)->src);
cond_stmt = as_a <gcond *> (last_stmt (loop->header));
cvar = gimple_cond_lhs (cond_stmt);
cvar_base = SSA_NAME_VAR (cvar);
phi = SSA_NAME_DEF_STMT (cvar);
cvar_init = PHI_ARG_DEF_FROM_EDGE (phi, loop_preheader_edge (loop));
initvar = copy_ssa_name (cvar);
SET_USE (PHI_ARG_DEF_PTR_FROM_EDGE (phi, loop_preheader_edge (loop)),
initvar);
cvar_next = PHI_ARG_DEF_FROM_EDGE (phi, loop_latch_edge (loop));
gsi = gsi_last_nondebug_bb (loop->latch);
gcc_assert (gsi_stmt (gsi) == SSA_NAME_DEF_STMT (cvar_next));
gsi_remove (&gsi, true);
for_bb = split_edge (loop_preheader_edge (loop));
ex_bb = split_loop_exit_edge (single_dom_exit (loop));
extract_true_false_edges_from_block (loop->header, &nexit, &exit);
gcc_assert (exit == single_dom_exit (loop));
guard = make_edge (for_bb, ex_bb, 0);
guard->probability = profile_probability::guessed_never ();
loop->latch = split_edge (single_succ_edge (loop->latch));
single_pred_edge (loop->latch)->flags = 0;
end = make_single_succ_edge (single_pred (loop->latch), ex_bb, EDGE_FALLTHRU);
rescan_loop_exit (end, true, false);
for (gphi_iterator gpi = gsi_start_phis (ex_bb);
!gsi_end_p (gpi); gsi_next (&gpi))
{
source_location locus;
gphi *phi = gpi.phi ();
tree def = PHI_ARG_DEF_FROM_EDGE (phi, exit);
gimple *def_stmt = SSA_NAME_DEF_STMT (def);
if (!(gimple_code (def_stmt) == GIMPLE_PHI
&& gimple_bb (def_stmt) == loop->header))
{
locus = gimple_phi_arg_location_from_edge (phi, exit);
add_phi_arg (phi, def, guard, locus);
add_phi_arg (phi, def, end, locus);
continue;
}
gphi *stmt = as_a <gphi *> (def_stmt);
def = PHI_ARG_DEF_FROM_EDGE (stmt, loop_preheader_edge (loop));
locus = gimple_phi_arg_location_from_edge (stmt,
loop_preheader_edge (loop));
add_phi_arg (phi, def, guard, locus);
def = PHI_ARG_DEF_FROM_EDGE (stmt, loop_latch_edge (loop));
locus = gimple_phi_arg_location_from_edge (stmt, loop_latch_edge (loop));
add_phi_arg (phi, def, end, locus);
}
e = redirect_edge_and_branch (exit, nexit->dest);
PENDING_STMT (e) = NULL;
if (oacc_kernels_p)
t = build_omp_clause (loc, OMP_CLAUSE_GANG);
else
{
t = build_omp_clause (loc, OMP_CLAUSE_SCHEDULE);
int chunk_size = PARAM_VALUE (PARAM_PARLOOPS_CHUNK_SIZE);
enum PARAM_PARLOOPS_SCHEDULE_KIND schedule_type \
= (enum PARAM_PARLOOPS_SCHEDULE_KIND) PARAM_VALUE (PARAM_PARLOOPS_SCHEDULE);
switch (schedule_type)
{
case PARAM_PARLOOPS_SCHEDULE_KIND_static:
OMP_CLAUSE_SCHEDULE_KIND (t) = OMP_CLAUSE_SCHEDULE_STATIC;
break;
case PARAM_PARLOOPS_SCHEDULE_KIND_dynamic:
OMP_CLAUSE_SCHEDULE_KIND (t) = OMP_CLAUSE_SCHEDULE_DYNAMIC;
break;
case PARAM_PARLOOPS_SCHEDULE_KIND_guided:
OMP_CLAUSE_SCHEDULE_KIND (t) = OMP_CLAUSE_SCHEDULE_GUIDED;
break;
case PARAM_PARLOOPS_SCHEDULE_KIND_auto:
OMP_CLAUSE_SCHEDULE_KIND (t) = OMP_CLAUSE_SCHEDULE_AUTO;
chunk_size = 0;
break;
case PARAM_PARLOOPS_SCHEDULE_KIND_runtime:
OMP_CLAUSE_SCHEDULE_KIND (t) = OMP_CLAUSE_SCHEDULE_RUNTIME;
chunk_size = 0;
break;
default:
gcc_unreachable ();
}
if (chunk_size != 0)
OMP_CLAUSE_SCHEDULE_CHUNK_EXPR (t)
= build_int_cst (integer_type_node, chunk_size);
}
for_stmt = gimple_build_omp_for (NULL,
(oacc_kernels_p
? GF_OMP_FOR_KIND_OACC_LOOP
: GF_OMP_FOR_KIND_FOR),
t, 1, NULL);
gimple_cond_set_lhs (cond_stmt, cvar_base);
type = TREE_TYPE (cvar);
gimple_set_location (for_stmt, loc);
gimple_omp_for_set_index (for_stmt, 0, initvar);
gimple_omp_for_set_initial (for_stmt, 0, cvar_init);
gimple_omp_for_set_final (for_stmt, 0, gimple_cond_rhs (cond_stmt));
gimple_omp_for_set_cond (for_stmt, 0, gimple_cond_code (cond_stmt));
gimple_omp_for_set_incr (for_stmt, 0, build2 (PLUS_EXPR, type,
cvar_base,
build_int_cst (type, 1)));
gsi = gsi_last_bb (for_bb);
gsi_insert_after (&gsi, for_stmt, GSI_NEW_STMT);
SSA_NAME_DEF_STMT (initvar) = for_stmt;
continue_bb = single_pred (loop->latch);
gsi = gsi_last_bb (continue_bb);
omp_cont_stmt = gimple_build_omp_continue (cvar_next, cvar);
gimple_set_location (omp_cont_stmt, loc);
gsi_insert_after (&gsi, omp_cont_stmt, GSI_NEW_STMT);
SSA_NAME_DEF_STMT (cvar_next) = omp_cont_stmt;
gsi = gsi_last_bb (ex_bb);
omp_return_stmt2 = gimple_build_omp_return (true);
gimple_set_location (omp_return_stmt2, loc);
gsi_insert_after (&gsi, omp_return_stmt2, GSI_NEW_STMT);
free_dominance_info (CDI_DOMINATORS);
calculate_dominance_info (CDI_DOMINATORS);
}
static unsigned int
num_phis (basic_block bb, bool count_virtual_p)
{
unsigned int nr_phis = 0;
gphi_iterator gsi;
for (gsi = gsi_start_phis (bb); !gsi_end_p (gsi); gsi_next (&gsi))
{
if (!count_virtual_p && virtual_operand_p (PHI_RESULT (gsi.phi ())))
continue;
nr_phis++;
}
return nr_phis;
}
static void
gen_parallel_loop (struct loop *loop,
reduction_info_table_type *reduction_list,
unsigned n_threads, struct tree_niter_desc *niter,
bool oacc_kernels_p)
{
tree many_iterations_cond, type, nit;
tree arg_struct, new_arg_struct;
gimple_seq stmts;
edge entry, exit;
struct clsn_data clsn_data;
location_t loc;
gimple *cond_stmt;
unsigned int m_p_thread=2;
type = TREE_TYPE (niter->niter);
nit = force_gimple_operand (unshare_expr (niter->niter), &stmts, true,
NULL_TREE);
if (stmts)
gsi_insert_seq_on_edge_immediate (loop_preheader_edge (loop), stmts);
if (!oacc_kernels_p)
{
if (loop->inner)
m_p_thread=2;
else
m_p_thread=MIN_PER_THREAD;
gcc_checking_assert (n_threads != 0);
many_iterations_cond =
fold_build2 (GE_EXPR, boolean_type_node,
nit, build_int_cst (type, m_p_thread * n_threads - 1));
many_iterations_cond
= fold_build2 (TRUTH_AND_EXPR, boolean_type_node,
invert_truthvalue (unshare_expr (niter->may_be_zero)),
many_iterations_cond);
many_iterations_cond
= force_gimple_operand (many_iterations_cond, &stmts, false, NULL_TREE);
if (stmts)
gsi_insert_seq_on_edge_immediate (loop_preheader_edge (loop), stmts);
if (!is_gimple_condexpr (many_iterations_cond))
{
many_iterations_cond
= force_gimple_operand (many_iterations_cond, &stmts,
true, NULL_TREE);
if (stmts)
gsi_insert_seq_on_edge_immediate (loop_preheader_edge (loop),
stmts);
}
initialize_original_copy_tables ();
loop_version (loop, many_iterations_cond, NULL,
profile_probability::likely (),
profile_probability::unlikely (),
profile_probability::likely (),
profile_probability::unlikely (), true);
update_ssa (TODO_update_ssa);
free_original_copy_tables ();
}
canonicalize_loop_ivs (loop, &nit, true);
if (num_phis (loop->header, false) != reduction_list->elements () + 1)
{
basic_block preheader = loop_preheader_edge (loop)->src;
basic_block cond_bb = single_pred (preheader);
gcond *cond = as_a <gcond *> (gsi_stmt (gsi_last_bb (cond_bb)));
gimple_cond_make_true (cond);
update_stmt (cond);
if (dump_file
&& (dump_flags & TDF_DETAILS))
fprintf (dump_file, "canonicalize_loop_ivs failed for loop %d,"
" aborting transformation\n", loop->num);
return;
}
if (try_transform_to_exit_first_loop_alt (loop, reduction_list, nit))
{
if (dump_file
&& (dump_flags & TDF_DETAILS))
fprintf (dump_file,
"alternative exit-first loop transform succeeded"
" for loop %d\n", loop->num);
}
else
{
if (oacc_kernels_p)
n_threads = 1;
transform_to_exit_first_loop (loop, reduction_list, nit);
}
if (reduction_list->elements () > 0)
reduction_list->traverse <struct loop *, initialize_reductions> (loop);
gcc_assert (single_exit (loop));
entry = loop_preheader_edge (loop);
exit = single_dom_exit (loop);
if (!oacc_kernels_p)
{
eliminate_local_variables (entry, exit);
separate_decls_in_region (entry, exit, reduction_list, &arg_struct,
&new_arg_struct, &clsn_data);
}
else
{
arg_struct = NULL_TREE;
new_arg_struct = NULL_TREE;
clsn_data.load = NULL_TREE;
clsn_data.load_bb = exit->dest;
clsn_data.store = NULL_TREE;
clsn_data.store_bb = NULL;
}
loc = UNKNOWN_LOCATION;
cond_stmt = last_stmt (loop->header);
if (cond_stmt)
loc = gimple_location (cond_stmt);
create_parallel_loop (loop, create_loop_fn (loc), arg_struct, new_arg_struct,
n_threads, loc, oacc_kernels_p);
if (reduction_list->elements () > 0)
create_call_for_reduction (loop, reduction_list, &clsn_data);
scev_reset ();
free_numbers_of_iterations_estimates (cfun);
}
static bool
loop_has_vector_phi_nodes (struct loop *loop ATTRIBUTE_UNUSED)
{
unsigned i;
basic_block *bbs = get_loop_body_in_dom_order (loop);
gphi_iterator gsi;
bool res = true;
for (i = 0; i < loop->num_nodes; i++)
for (gsi = gsi_start_phis (bbs[i]); !gsi_end_p (gsi); gsi_next (&gsi))
if (TREE_CODE (TREE_TYPE (PHI_RESULT (gsi.phi ()))) == VECTOR_TYPE)
goto end;
res = false;
end:
free (bbs);
return res;
}
static void
build_new_reduction (reduction_info_table_type *reduction_list,
gimple *reduc_stmt, gphi *phi)
{
reduction_info **slot;
struct reduction_info *new_reduction;
enum tree_code reduction_code;
gcc_assert (reduc_stmt);
if (gimple_code (reduc_stmt) == GIMPLE_PHI)
{
tree op1 = PHI_ARG_DEF (reduc_stmt, 0);
gimple *def1 = SSA_NAME_DEF_STMT (op1);
reduction_code = gimple_assign_rhs_code (def1);
}
else
reduction_code = gimple_assign_rhs_code (reduc_stmt);
switch (reduction_code)
{
case PLUS_EXPR:
case MULT_EXPR:
case MAX_EXPR:
case MIN_EXPR:
case BIT_IOR_EXPR:
case BIT_XOR_EXPR:
case BIT_AND_EXPR:
case TRUTH_OR_EXPR:
case TRUTH_XOR_EXPR:
case TRUTH_AND_EXPR:
break;
default:
return;
}
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file,
"Detected reduction. reduction stmt is:\n");
print_gimple_stmt (dump_file, reduc_stmt, 0);
fprintf (dump_file, "\n");
}
new_reduction = XCNEW (struct reduction_info);
new_reduction->reduc_stmt = reduc_stmt;
new_reduction->reduc_phi = phi;
new_reduction->reduc_version = SSA_NAME_VERSION (gimple_phi_result (phi));
new_reduction->reduction_code = reduction_code;
slot = reduction_list->find_slot (new_reduction, INSERT);
*slot = new_reduction;
}
int
set_reduc_phi_uids (reduction_info **slot, void *data ATTRIBUTE_UNUSED)
{
struct reduction_info *const red = *slot;
gimple_set_uid (red->reduc_phi, red->reduc_version);
return 1;
}
static bool
valid_reduction_p (gimple *stmt)
{
stmt_vec_info stmt_info = vinfo_for_stmt (stmt);
vect_reduction_type reduc_type = STMT_VINFO_REDUC_TYPE (stmt_info);
return reduc_type != FOLD_LEFT_REDUCTION;
}
static void
gather_scalar_reductions (loop_p loop, reduction_info_table_type *reduction_list)
{
gphi_iterator gsi;
loop_vec_info simple_loop_info;
auto_vec<gphi *, 4> double_reduc_phis;
auto_vec<gimple *, 4> double_reduc_stmts;
if (!stmt_vec_info_vec.exists ())
init_stmt_vec_info_vec ();
simple_loop_info = vect_analyze_loop_form (loop);
if (simple_loop_info == NULL)
goto gather_done;
for (gsi = gsi_start_phis (loop->header); !gsi_end_p (gsi); gsi_next (&gsi))
{
gphi *phi = gsi.phi ();
affine_iv iv;
tree res = PHI_RESULT (phi);
bool double_reduc;
if (virtual_operand_p (res))
continue;
if (simple_iv (loop, loop, res, &iv, true))
continue;
gimple *reduc_stmt
= vect_force_simple_reduction (simple_loop_info, phi,
&double_reduc, true);
if (!reduc_stmt || !valid_reduction_p (reduc_stmt))
continue;
if (double_reduc)
{
if (loop->inner->inner != NULL)
continue;
double_reduc_phis.safe_push (phi);
double_reduc_stmts.safe_push (reduc_stmt);
continue;
}
build_new_reduction (reduction_list, reduc_stmt, phi);
}
delete simple_loop_info;
if (!double_reduc_phis.is_empty ())
{
simple_loop_info = vect_analyze_loop_form (loop->inner);
if (simple_loop_info)
{
gphi *phi;
unsigned int i;
FOR_EACH_VEC_ELT (double_reduc_phis, i, phi)
{
affine_iv iv;
tree res = PHI_RESULT (phi);
bool double_reduc;
use_operand_p use_p;
gimple *inner_stmt;
bool single_use_p = single_imm_use (res, &use_p, &inner_stmt);
gcc_assert (single_use_p);
if (gimple_code (inner_stmt) != GIMPLE_PHI)
continue;
gphi *inner_phi = as_a <gphi *> (inner_stmt);
if (simple_iv (loop->inner, loop->inner, PHI_RESULT (inner_phi),
&iv, true))
continue;
gimple *inner_reduc_stmt
= vect_force_simple_reduction (simple_loop_info, inner_phi,
&double_reduc, true);
gcc_assert (!double_reduc);
if (inner_reduc_stmt == NULL
|| !valid_reduction_p (inner_reduc_stmt))
continue;
build_new_reduction (reduction_list, double_reduc_stmts[i], phi);
}
delete simple_loop_info;
}
}
gather_done:
free_stmt_vec_info_vec ();
if (reduction_list->elements () == 0)
return;
basic_block bb;
FOR_EACH_BB_FN (bb, cfun)
for (gsi = gsi_start_phis (bb); !gsi_end_p (gsi); gsi_next (&gsi))
gimple_set_uid (gsi_stmt (gsi), (unsigned int)-1);
reduction_list->traverse <void *, set_reduc_phi_uids> (NULL);
}
static bool
try_get_loop_niter (loop_p loop, struct tree_niter_desc *niter)
{
edge exit = single_dom_exit (loop);
gcc_assert (exit);
if (!number_of_iterations_exit (loop, exit, niter, false))
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "  FAILED: number of iterations not known\n");
return false;
}
return true;
}
static tree
get_omp_data_i_param (void)
{
tree decl = DECL_ARGUMENTS (cfun->decl);
gcc_assert (DECL_CHAIN (decl) == NULL_TREE);
return ssa_default_def (cfun, decl);
}
static tree
find_reduc_addr (struct loop *loop, gphi *phi)
{
edge e = loop_preheader_edge (loop);
tree arg = PHI_ARG_DEF_FROM_EDGE (phi, e);
gimple *stmt = SSA_NAME_DEF_STMT (arg);
if (!gimple_assign_single_p (stmt))
return NULL_TREE;
tree memref = gimple_assign_rhs1 (stmt);
if (TREE_CODE (memref) != MEM_REF)
return NULL_TREE;
tree addr = TREE_OPERAND (memref, 0);
gimple *stmt2 = SSA_NAME_DEF_STMT (addr);
if (!gimple_assign_single_p (stmt2))
return NULL_TREE;
tree compref = gimple_assign_rhs1 (stmt2);
if (TREE_CODE (compref) != COMPONENT_REF)
return NULL_TREE;
tree addr2 = TREE_OPERAND (compref, 0);
if (TREE_CODE (addr2) != MEM_REF)
return NULL_TREE;
addr2 = TREE_OPERAND (addr2, 0);
if (TREE_CODE (addr2) != SSA_NAME
|| addr2 != get_omp_data_i_param ())
return NULL_TREE;
return addr;
}
static bool
try_create_reduction_list (loop_p loop,
reduction_info_table_type *reduction_list,
bool oacc_kernels_p)
{
edge exit = single_dom_exit (loop);
gphi_iterator gsi;
gcc_assert (exit);
final_value_replacement_loop (loop);
gather_scalar_reductions (loop, reduction_list);
for (gsi = gsi_start_phis (exit->dest); !gsi_end_p (gsi); gsi_next (&gsi))
{
gphi *phi = gsi.phi ();
struct reduction_info *red;
imm_use_iterator imm_iter;
use_operand_p use_p;
gimple *reduc_phi;
tree val = PHI_ARG_DEF_FROM_EDGE (phi, exit);
if (!virtual_operand_p (val))
{
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "phi is ");
print_gimple_stmt (dump_file, phi, 0);
fprintf (dump_file, "arg of phi to exit:   value ");
print_generic_expr (dump_file, val);
fprintf (dump_file, " used outside loop\n");
fprintf (dump_file,
"  checking if it is part of reduction pattern:\n");
}
if (reduction_list->elements () == 0)
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file,
"  FAILED: it is not a part of reduction.\n");
return false;
}
reduc_phi = NULL;
FOR_EACH_IMM_USE_FAST (use_p, imm_iter, val)
{
if (!gimple_debug_bind_p (USE_STMT (use_p))
&& flow_bb_inside_loop_p (loop, gimple_bb (USE_STMT (use_p))))
{
reduc_phi = USE_STMT (use_p);
break;
}
}
red = reduction_phi (reduction_list, reduc_phi);
if (red == NULL)
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file,
"  FAILED: it is not a part of reduction.\n");
return false;
}
if (red->keep_res != NULL)
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file,
"  FAILED: reduction has multiple exit phis.\n");
return false;
}
red->keep_res = phi;
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "reduction phi is  ");
print_gimple_stmt (dump_file, red->reduc_phi, 0);
fprintf (dump_file, "reduction stmt is  ");
print_gimple_stmt (dump_file, red->reduc_stmt, 0);
}
}
}
for (gsi = gsi_start_phis (loop->header); !gsi_end_p (gsi); gsi_next (&gsi))
{
gphi *phi = gsi.phi ();
tree def = PHI_RESULT (phi);
affine_iv iv;
if (!virtual_operand_p (def) && !simple_iv (loop, loop, def, &iv, true))
{
struct reduction_info *red;
red = reduction_phi (reduction_list, phi);
if (red == NULL)
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file,
"  FAILED: scalar dependency between iterations\n");
return false;
}
}
}
if (oacc_kernels_p)
{
for (gsi = gsi_start_phis (loop->header); !gsi_end_p (gsi);
gsi_next (&gsi))
{
gphi *phi = gsi.phi ();
tree def = PHI_RESULT (phi);
affine_iv iv;
if (!virtual_operand_p (def)
&& !simple_iv (loop, loop, def, &iv, true))
{
tree addr = find_reduc_addr (loop, phi);
if (addr == NULL_TREE)
return false;
struct reduction_info *red = reduction_phi (reduction_list, phi);
red->reduc_addr = addr;
}
}
}
return true;
}
static bool
loop_has_phi_with_address_arg (struct loop *loop)
{
basic_block *bbs = get_loop_body (loop);
bool res = false;
unsigned i, j;
gphi_iterator gsi;
for (i = 0; i < loop->num_nodes; i++)
for (gsi = gsi_start_phis (bbs[i]); !gsi_end_p (gsi); gsi_next (&gsi))
{
gphi *phi = gsi.phi ();
for (j = 0; j < gimple_phi_num_args (phi); j++)
{
tree arg = gimple_phi_arg_def (phi, j);
if (TREE_CODE (arg) == ADDR_EXPR)
{
res = true;
goto end;
}
}
}
end:
free (bbs);
return res;
}
static bool
ref_conflicts_with_region (gimple_stmt_iterator gsi, ao_ref *ref,
bool ref_is_store, vec<basic_block> region_bbs,
unsigned int i, gimple *skip_stmt)
{
basic_block bb = region_bbs[i];
gsi_next (&gsi);
while (true)
{
for (; !gsi_end_p (gsi);
gsi_next (&gsi))
{
gimple *stmt = gsi_stmt (gsi);
if (stmt == skip_stmt)
{
if (dump_file)
{
fprintf (dump_file, "skipping reduction store: ");
print_gimple_stmt (dump_file, stmt, 0);
}
continue;
}
if (!gimple_vdef (stmt)
&& !gimple_vuse (stmt))
continue;
if (gimple_code (stmt) == GIMPLE_RETURN)
continue;
if (ref_is_store)
{
if (ref_maybe_used_by_stmt_p (stmt, ref))
{
if (dump_file)
{
fprintf (dump_file, "Stmt ");
print_gimple_stmt (dump_file, stmt, 0);
}
return true;
}
}
else
{
if (stmt_may_clobber_ref_p_1 (stmt, ref))
{
if (dump_file)
{
fprintf (dump_file, "Stmt ");
print_gimple_stmt (dump_file, stmt, 0);
}
return true;
}
}
}
i++;
if (i == region_bbs.length ())
break;
bb = region_bbs[i];
gsi = gsi_start_bb (bb);
}
return false;
}
static bool
oacc_entry_exit_ok_1 (bitmap in_loop_bbs, vec<basic_block> region_bbs,
reduction_info_table_type *reduction_list,
bitmap reduction_stores)
{
tree omp_data_i = get_omp_data_i_param ();
unsigned i;
basic_block bb;
FOR_EACH_VEC_ELT (region_bbs, i, bb)
{
if (bitmap_bit_p (in_loop_bbs, bb->index))
continue;
gimple_stmt_iterator gsi;
for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi);
gsi_next (&gsi))
{
gimple *stmt = gsi_stmt (gsi);
gimple *skip_stmt = NULL;
if (is_gimple_debug (stmt)
|| gimple_code (stmt) == GIMPLE_COND)
continue;
ao_ref ref;
bool ref_is_store = false;
if (gimple_assign_load_p (stmt))
{
tree rhs = gimple_assign_rhs1 (stmt);
tree base = get_base_address (rhs);
if (TREE_CODE (base) == MEM_REF
&& operand_equal_p (TREE_OPERAND (base, 0), omp_data_i, 0))
continue;
tree lhs = gimple_assign_lhs (stmt);
if (TREE_CODE (lhs) == SSA_NAME
&& has_single_use (lhs))
{
use_operand_p use_p;
gimple *use_stmt;
single_imm_use (lhs, &use_p, &use_stmt);
if (gimple_code (use_stmt) == GIMPLE_PHI)
{
struct reduction_info *red;
red = reduction_phi (reduction_list, use_stmt);
tree val = PHI_RESULT (red->keep_res);
if (has_single_use (val))
{
single_imm_use (val, &use_p, &use_stmt);
if (gimple_store_p (use_stmt))
{
unsigned int id
= SSA_NAME_VERSION (gimple_vdef (use_stmt));
bitmap_set_bit (reduction_stores, id);
skip_stmt = use_stmt;
if (dump_file)
{
fprintf (dump_file, "found reduction load: ");
print_gimple_stmt (dump_file, stmt, 0);
}
}
}
}
}
ao_ref_init (&ref, rhs);
}
else if (gimple_store_p (stmt))
{
ao_ref_init (&ref, gimple_assign_lhs (stmt));
ref_is_store = true;
}
else if (gimple_code (stmt) == GIMPLE_OMP_RETURN)
continue;
else if (!gimple_has_side_effects (stmt)
&& !gimple_could_trap_p (stmt)
&& !stmt_could_throw_p (stmt)
&& !gimple_vdef (stmt)
&& !gimple_vuse (stmt))
continue;
else if (gimple_call_internal_p (stmt, IFN_GOACC_DIM_POS))
continue;
else if (gimple_code (stmt) == GIMPLE_RETURN)
continue;
else
{
if (dump_file)
{
fprintf (dump_file, "Unhandled stmt in entry/exit: ");
print_gimple_stmt (dump_file, stmt, 0);
}
return false;
}
if (ref_conflicts_with_region (gsi, &ref, ref_is_store, region_bbs,
i, skip_stmt))
{
if (dump_file)
{
fprintf (dump_file, "conflicts with entry/exit stmt: ");
print_gimple_stmt (dump_file, stmt, 0);
}
return false;
}
}
}
return true;
}
static bool
oacc_entry_exit_single_gang (bitmap in_loop_bbs, vec<basic_block> region_bbs,
bitmap reduction_stores)
{
tree gang_pos = NULL_TREE;
bool changed = false;
unsigned i;
basic_block bb;
FOR_EACH_VEC_ELT (region_bbs, i, bb)
{
if (bitmap_bit_p (in_loop_bbs, bb->index))
continue;
gimple_stmt_iterator gsi;
for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi);)
{
gimple *stmt = gsi_stmt (gsi);
if (!gimple_store_p (stmt))
{
gsi_next (&gsi);
continue;
}
if (bitmap_bit_p (reduction_stores,
SSA_NAME_VERSION (gimple_vdef (stmt))))
{
if (dump_file)
{
fprintf (dump_file,
"skipped reduction store for single-gang"
" neutering: ");
print_gimple_stmt (dump_file, stmt, 0);
}
gsi_next (&gsi);
continue;
}
changed = true;
if (gang_pos == NULL_TREE)
{
tree arg = build_int_cst (integer_type_node, GOMP_DIM_GANG);
gcall *gang_single
= gimple_build_call_internal (IFN_GOACC_DIM_POS, 1, arg);
gang_pos = make_ssa_name (integer_type_node);
gimple_call_set_lhs (gang_single, gang_pos);
gimple_stmt_iterator start
= gsi_start_bb (single_succ (ENTRY_BLOCK_PTR_FOR_FN (cfun)));
tree vuse = ssa_default_def (cfun, gimple_vop (cfun));
gimple_set_vuse (gang_single, vuse);
gsi_insert_before (&start, gang_single, GSI_SAME_STMT);
}
if (dump_file)
{
fprintf (dump_file,
"found store that needs single-gang neutering: ");
print_gimple_stmt (dump_file, stmt, 0);
}
{
gimple_stmt_iterator gsi2 = gsi;
gsi_prev (&gsi2);
edge e;
if (gsi_end_p (gsi2))
{
e = split_block_after_labels (bb);
gsi2 = gsi_last_bb (bb);
}
else
e = split_block (bb, gsi_stmt (gsi2));
basic_block bb2 = e->dest;
gimple_stmt_iterator gsi3 = gsi_start_bb (bb2);
edge e2 = split_block (bb2, gsi_stmt (gsi3));
basic_block bb3 = e2->dest;
gimple *cond
= gimple_build_cond (EQ_EXPR, gang_pos, integer_zero_node,
NULL_TREE, NULL_TREE);
gsi_insert_after (&gsi2, cond, GSI_NEW_STMT);
edge e3 = make_edge (bb, bb3, EDGE_FALSE_VALUE);
e3->probability = profile_probability::guessed_never ();
e->flags = EDGE_TRUE_VALUE;
tree vdef = gimple_vdef (stmt);
tree vuse = gimple_vuse (stmt);
tree phi_res = copy_ssa_name (vdef);
gphi *new_phi = create_phi_node (phi_res, bb3);
replace_uses_by (vdef, phi_res);
add_phi_arg (new_phi, vuse, e3, UNKNOWN_LOCATION);
add_phi_arg (new_phi, vdef, e2, UNKNOWN_LOCATION);
bb = bb3;
gsi = gsi_start_bb (bb);
}
}
}
return changed;
}
static bool
oacc_entry_exit_ok (struct loop *loop,
reduction_info_table_type *reduction_list)
{
basic_block *loop_bbs = get_loop_body_in_dom_order (loop);
vec<basic_block> region_bbs
= get_all_dominated_blocks (CDI_DOMINATORS, ENTRY_BLOCK_PTR_FOR_FN (cfun));
bitmap in_loop_bbs = BITMAP_ALLOC (NULL);
bitmap_clear (in_loop_bbs);
for (unsigned int i = 0; i < loop->num_nodes; i++)
bitmap_set_bit (in_loop_bbs, loop_bbs[i]->index);
bitmap reduction_stores = BITMAP_ALLOC (NULL);
bool res = oacc_entry_exit_ok_1 (in_loop_bbs, region_bbs, reduction_list,
reduction_stores);
if (res)
{
bool changed = oacc_entry_exit_single_gang (in_loop_bbs, region_bbs,
reduction_stores);
if (changed)
{
free_dominance_info (CDI_DOMINATORS);
calculate_dominance_info (CDI_DOMINATORS);
}
}
region_bbs.release ();
free (loop_bbs);
BITMAP_FREE (in_loop_bbs);
BITMAP_FREE (reduction_stores);
return res;
}
static bool
parallelize_loops (bool oacc_kernels_p)
{
unsigned n_threads;
bool changed = false;
struct loop *loop;
struct loop *skip_loop = NULL;
struct tree_niter_desc niter_desc;
struct obstack parloop_obstack;
HOST_WIDE_INT estimated;
source_location loop_loc;
if (!oacc_kernels_p
&& parallelized_function_p (cfun->decl))
return false;
if (!oacc_kernels_p
&& oacc_get_fn_attrib (cfun->decl) != NULL)
return false;
if (cfun->has_nonlocal_label)
return false;
if (oacc_kernels_p)
n_threads = 0;
else
n_threads = flag_tree_parallelize_loops;
gcc_obstack_init (&parloop_obstack);
reduction_info_table_type reduction_list (10);
calculate_dominance_info (CDI_DOMINATORS);
FOR_EACH_LOOP (loop, 0)
{
if (loop == skip_loop)
{
if (!loop->in_oacc_kernels_region
&& dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file,
"Skipping loop %d as inner loop of parallelized loop\n",
loop->num);
skip_loop = loop->inner;
continue;
}
else
skip_loop = NULL;
reduction_list.empty ();
if (oacc_kernels_p)
{
if (!loop->in_oacc_kernels_region)
continue;
if (loop->inner)
skip_loop = loop->inner;
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file,
"Trying loop %d with header bb %d in oacc kernels"
" region\n", loop->num, loop->header->index);
}
if (dump_file && (dump_flags & TDF_DETAILS))
{
fprintf (dump_file, "Trying loop %d as candidate\n",loop->num);
if (loop->inner)
fprintf (dump_file, "loop %d is not innermost\n",loop->num);
else
fprintf (dump_file, "loop %d is innermost\n",loop->num);
}
if (!single_dom_exit (loop))
{
if (dump_file && (dump_flags & TDF_DETAILS))
fprintf (dump_file, "loop is !single_dom_exit\n");
continue;
}
if (
!can_duplicate_loop_p (loop)
|| loop_has_blocks_with_irreducible_flag (loop)
|| (loop_preheader_edge (loop)->src->flags & BB_IRREDUCIBLE_LOOP)
|| loop_has_vector_phi_nodes (loop))
continue;
estimated = estimated_loop_iterations_int (loop);
if (estimated == -1)
estimated = get_likely_max_loop_iterations_int (loop);
if (!flag_loop_parallelize_all
&& !oacc_kernels_p
&& ((estimated != -1
&& (estimated
< ((HOST_WIDE_INT) n_threads
* (loop->inner ? 2 : MIN_PER_THREAD) - 1)))
|| optimize_loop_nest_for_size_p (loop)))
continue;
if (!try_get_loop_niter (loop, &niter_desc))
continue;
if (!try_create_reduction_list (loop, &reduction_list, oacc_kernels_p))
continue;
if (loop_has_phi_with_address_arg (loop))
continue;
if (!loop->can_be_parallel
&& !loop_parallel_p (loop, &parloop_obstack))
continue;
if (oacc_kernels_p
&& !oacc_entry_exit_ok (loop, &reduction_list))
{
if (dump_file)
fprintf (dump_file, "entry/exit not ok: FAILED\n");
continue;
}
changed = true;
skip_loop = loop->inner;
loop_loc = find_loop_location (loop);
if (loop->inner)
dump_printf_loc (MSG_OPTIMIZED_LOCATIONS, loop_loc,
"parallelizing outer loop %d\n", loop->num);
else
dump_printf_loc (MSG_OPTIMIZED_LOCATIONS, loop_loc,
"parallelizing inner loop %d\n", loop->num);
gen_parallel_loop (loop, &reduction_list,
n_threads, &niter_desc, oacc_kernels_p);
}
obstack_free (&parloop_obstack, NULL);
if (changed)
pt_solution_reset (&cfun->gimple_df->escaped);
return changed;
}
namespace {
const pass_data pass_data_parallelize_loops =
{
GIMPLE_PASS, 
"parloops", 
OPTGROUP_LOOP, 
TV_TREE_PARALLELIZE_LOOPS, 
( PROP_cfg | PROP_ssa ), 
0, 
0, 
0, 
0, 
};
class pass_parallelize_loops : public gimple_opt_pass
{
public:
pass_parallelize_loops (gcc::context *ctxt)
: gimple_opt_pass (pass_data_parallelize_loops, ctxt),
oacc_kernels_p (false)
{}
virtual bool gate (function *)
{
if (oacc_kernels_p)
return flag_openacc;
else
return flag_tree_parallelize_loops > 1;
}
virtual unsigned int execute (function *);
opt_pass * clone () { return new pass_parallelize_loops (m_ctxt); }
void set_pass_param (unsigned int n, bool param)
{
gcc_assert (n == 0);
oacc_kernels_p = param;
}
private:
bool oacc_kernels_p;
}; 
unsigned
pass_parallelize_loops::execute (function *fun)
{
tree nthreads = builtin_decl_explicit (BUILT_IN_OMP_GET_NUM_THREADS);
if (nthreads == NULL_TREE)
return 0;
bool in_loop_pipeline = scev_initialized_p ();
if (!in_loop_pipeline)
loop_optimizer_init (LOOPS_NORMAL
| LOOPS_HAVE_RECORDED_EXITS);
if (number_of_loops (fun) <= 1)
return 0;
if (!in_loop_pipeline)
{
rewrite_into_loop_closed_ssa (NULL, TODO_update_ssa);
scev_initialize ();
}
unsigned int todo = 0;
if (parallelize_loops (oacc_kernels_p))
{
fun->curr_properties &= ~(PROP_gimple_eomp);
checking_verify_loop_structure ();
todo |= TODO_update_ssa;
}
if (!in_loop_pipeline)
{
scev_finalize ();
loop_optimizer_finalize ();
}
return todo;
}
} 
gimple_opt_pass *
make_pass_parallelize_loops (gcc::context *ctxt)
{
return new pass_parallelize_loops (ctxt);
}
