#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "target.h"
#include "rtl.h"
#include "tree.h"
#include "memmodel.h"
#include "tm_p.h"
#include "gimple.h"
#include "gimple-predict.h"
#include "tree-pass.h"		
#include "ssa.h"
#include "cgraph.h"
#include "tree-pretty-print.h"
#include "diagnostic-core.h"
#include "alias.h"
#include "fold-const.h"
#include "calls.h"
#include "varasm.h"
#include "stmt.h"
#include "expr.h"
#include "gimple-fold.h"
#include "tree-eh.h"
#include "gimplify.h"
#include "gimple-iterator.h"
#include "stor-layout.h"
#include "print-tree.h"
#include "tree-iterator.h"
#include "tree-inline.h"
#include "langhooks.h"
#include "tree-cfg.h"
#include "tree-ssa.h"
#include "omp-general.h"
#include "omp-low.h"
#include "gimple-low.h"
#include "gomp-constants.h"
#include "splay-tree.h"
#include "gimple-walk.h"
#include "langhooks-def.h"	
#include "builtins.h"
#include "stringpool.h"
#include "attribs.h"
#include "asan.h"
#include "dbgcnt.h"
static hash_set<tree> *asan_poisoned_variables = NULL;
enum gimplify_omp_var_data
{
GOVD_SEEN = 1,
GOVD_EXPLICIT = 2,
GOVD_SHARED = 4,
GOVD_PRIVATE = 8,
GOVD_FIRSTPRIVATE = 16,
GOVD_LASTPRIVATE = 32,
GOVD_REDUCTION = 64,
GOVD_LOCAL = 128,
GOVD_MAP = 256,
GOVD_DEBUG_PRIVATE = 512,
GOVD_PRIVATE_OUTER_REF = 1024,
GOVD_LINEAR = 2048,
GOVD_ALIGNED = 4096,
GOVD_MAP_TO_ONLY = 8192,
GOVD_LINEAR_LASTPRIVATE_NO_OUTER = 16384,
GOVD_MAP_0LEN_ARRAY = 32768,
GOVD_MAP_ALWAYS_TO = 65536,
GOVD_WRITTEN = 131072,
GOVD_MAP_FORCE = 262144,
GOVD_MAP_FORCE_PRESENT = 524288,
GOVD_DATA_SHARE_CLASS = (GOVD_SHARED | GOVD_PRIVATE | GOVD_FIRSTPRIVATE
| GOVD_LASTPRIVATE | GOVD_REDUCTION | GOVD_LINEAR
| GOVD_LOCAL)
};
enum omp_region_type
{
ORT_WORKSHARE = 0x00,
ORT_SIMD 	= 0x01,
ORT_PARALLEL	= 0x02,
ORT_COMBINED_PARALLEL = 0x03,
ORT_TASK	= 0x04,
ORT_UNTIED_TASK = 0x05,
ORT_TEAMS	= 0x08,
ORT_COMBINED_TEAMS = 0x09,
ORT_TARGET_DATA = 0x10,
ORT_TARGET	= 0x20,
ORT_COMBINED_TARGET = 0x21,
ORT_ACC	= 0x40,  
ORT_ACC_DATA	= ORT_ACC | ORT_TARGET_DATA, 
ORT_ACC_PARALLEL = ORT_ACC | ORT_TARGET,  
ORT_ACC_KERNELS  = ORT_ACC | ORT_TARGET | 0x80,  
ORT_ACC_HOST_DATA = ORT_ACC | ORT_TARGET_DATA | 0x80,  
ORT_NONE	= 0x100
};
struct gimplify_hasher : free_ptr_hash <elt_t>
{
static inline hashval_t hash (const elt_t *);
static inline bool equal (const elt_t *, const elt_t *);
};
struct gimplify_ctx
{
struct gimplify_ctx *prev_context;
vec<gbind *> bind_expr_stack;
tree temps;
gimple_seq conditional_cleanups;
tree exit_label;
tree return_temp;
vec<tree> case_labels;
hash_set<tree> *live_switch_vars;
hash_table<gimplify_hasher> *temp_htab;
int conditions;
unsigned into_ssa : 1;
unsigned allow_rhs_cond_expr : 1;
unsigned in_cleanup_point_expr : 1;
unsigned keep_stack : 1;
unsigned save_stack : 1;
unsigned in_switch_expr : 1;
};
struct gimplify_omp_ctx
{
struct gimplify_omp_ctx *outer_context;
splay_tree variables;
hash_set<tree> *privatized_types;
vec<tree> loop_iter_var;
location_t location;
enum omp_clause_default_kind default_kind;
enum omp_region_type region_type;
bool combined_loop;
bool distribute;
bool target_map_scalars_firstprivate;
bool target_map_pointers_as_0len_arrays;
bool target_firstprivatize_array_bases;
};
static struct gimplify_ctx *gimplify_ctxp;
static struct gimplify_omp_ctx *gimplify_omp_ctxp;
static enum gimplify_status gimplify_compound_expr (tree *, gimple_seq *, bool);
static hash_map<tree, tree> *oacc_declare_returns;
static enum gimplify_status gimplify_expr (tree *, gimple_seq *, gimple_seq *,
bool (*) (tree), fallback_t, bool);
static inline void
gimplify_seq_add_stmt (gimple_seq *seq_p, gimple *gs)
{
gimple_seq_add_stmt_without_update (seq_p, gs);
}
static void
gimplify_seq_add_seq (gimple_seq *dst_p, gimple_seq src)
{
gimple_stmt_iterator si;
if (src == NULL)
return;
si = gsi_last (*dst_p);
gsi_insert_seq_after_without_update (&si, src, GSI_NEW_STMT);
}
static struct gimplify_ctx *ctx_pool = NULL;
static inline struct gimplify_ctx *
ctx_alloc (void)
{
struct gimplify_ctx * c = ctx_pool;
if (c)
ctx_pool = c->prev_context;
else
c = XNEW (struct gimplify_ctx);
memset (c, '\0', sizeof (*c));
return c;
}
static inline void
ctx_free (struct gimplify_ctx *c)
{
c->prev_context = ctx_pool;
ctx_pool = c;
}
void
free_gimplify_stack (void)
{
struct gimplify_ctx *c;
while ((c = ctx_pool))
{
ctx_pool = c->prev_context;
free (c);
}
}
void
push_gimplify_context (bool in_ssa, bool rhs_cond_ok)
{
struct gimplify_ctx *c = ctx_alloc ();
c->prev_context = gimplify_ctxp;
gimplify_ctxp = c;
gimplify_ctxp->into_ssa = in_ssa;
gimplify_ctxp->allow_rhs_cond_expr = rhs_cond_ok;
}
void
pop_gimplify_context (gimple *body)
{
struct gimplify_ctx *c = gimplify_ctxp;
gcc_assert (c
&& (!c->bind_expr_stack.exists ()
|| c->bind_expr_stack.is_empty ()));
c->bind_expr_stack.release ();
gimplify_ctxp = c->prev_context;
if (body)
declare_vars (c->temps, body, false);
else
record_vars (c->temps);
delete c->temp_htab;
c->temp_htab = NULL;
ctx_free (c);
}
static void
gimple_push_bind_expr (gbind *bind_stmt)
{
gimplify_ctxp->bind_expr_stack.reserve (8);
gimplify_ctxp->bind_expr_stack.safe_push (bind_stmt);
}
static void
gimple_pop_bind_expr (void)
{
gimplify_ctxp->bind_expr_stack.pop ();
}
gbind *
gimple_current_bind_expr (void)
{
return gimplify_ctxp->bind_expr_stack.last ();
}
vec<gbind *>
gimple_bind_expr_stack (void)
{
return gimplify_ctxp->bind_expr_stack;
}
static bool
gimple_conditional_context (void)
{
return gimplify_ctxp->conditions > 0;
}
static void
gimple_push_condition (void)
{
#ifdef ENABLE_GIMPLE_CHECKING
if (gimplify_ctxp->conditions == 0)
gcc_assert (gimple_seq_empty_p (gimplify_ctxp->conditional_cleanups));
#endif
++(gimplify_ctxp->conditions);
}
static void
gimple_pop_condition (gimple_seq *pre_p)
{
int conds = --(gimplify_ctxp->conditions);
gcc_assert (conds >= 0);
if (conds == 0)
{
gimplify_seq_add_seq (pre_p, gimplify_ctxp->conditional_cleanups);
gimplify_ctxp->conditional_cleanups = NULL;
}
}
static int
splay_tree_compare_decl_uid (splay_tree_key xa, splay_tree_key xb)
{
tree a = (tree) xa;
tree b = (tree) xb;
return DECL_UID (a) - DECL_UID (b);
}
static struct gimplify_omp_ctx *
new_omp_context (enum omp_region_type region_type)
{
struct gimplify_omp_ctx *c;
c = XCNEW (struct gimplify_omp_ctx);
c->outer_context = gimplify_omp_ctxp;
c->variables = splay_tree_new (splay_tree_compare_decl_uid, 0, 0);
c->privatized_types = new hash_set<tree>;
c->location = input_location;
c->region_type = region_type;
if ((region_type & ORT_TASK) == 0)
c->default_kind = OMP_CLAUSE_DEFAULT_SHARED;
else
c->default_kind = OMP_CLAUSE_DEFAULT_UNSPECIFIED;
return c;
}
static void
delete_omp_context (struct gimplify_omp_ctx *c)
{
splay_tree_delete (c->variables);
delete c->privatized_types;
c->loop_iter_var.release ();
XDELETE (c);
}
static void omp_add_variable (struct gimplify_omp_ctx *, tree, unsigned int);
static bool omp_notice_variable (struct gimplify_omp_ctx *, tree, bool);
void
gimplify_and_add (tree t, gimple_seq *seq_p)
{
gimplify_stmt (&t, seq_p);
}
static gimple *
gimplify_and_return_first (tree t, gimple_seq *seq_p)
{
gimple_stmt_iterator last = gsi_last (*seq_p);
gimplify_and_add (t, seq_p);
if (!gsi_end_p (last))
{
gsi_next (&last);
return gsi_stmt (last);
}
else
return gimple_seq_first_stmt (*seq_p);
}
static bool
is_gimple_mem_rhs (tree t)
{
if (is_gimple_reg_type (TREE_TYPE (t)))
return is_gimple_val (t);
else
return is_gimple_val (t) || is_gimple_lvalue (t);
}
static bool
is_gimple_reg_rhs_or_call (tree t)
{
return (get_gimple_rhs_class (TREE_CODE (t)) != GIMPLE_INVALID_RHS
|| TREE_CODE (t) == CALL_EXPR);
}
static bool
is_gimple_mem_rhs_or_call (tree t)
{
if (is_gimple_reg_type (TREE_TYPE (t)))
return is_gimple_val (t);
else
return (is_gimple_val (t)
|| is_gimple_lvalue (t)
|| TREE_CLOBBER_P (t)
|| TREE_CODE (t) == CALL_EXPR);
}
static inline tree
create_tmp_from_val (tree val)
{
tree type = TYPE_MAIN_VARIANT (TREE_TYPE (val));
tree var = create_tmp_var (type, get_name (val));
if (TREE_CODE (TREE_TYPE (var)) == COMPLEX_TYPE
|| TREE_CODE (TREE_TYPE (var)) == VECTOR_TYPE)
DECL_GIMPLE_REG_P (var) = 1;
return var;
}
static tree
lookup_tmp_var (tree val, bool is_formal)
{
tree ret;
if (!optimize || !is_formal || TREE_SIDE_EFFECTS (val))
ret = create_tmp_from_val (val);
else
{
elt_t elt, *elt_p;
elt_t **slot;
elt.val = val;
if (!gimplify_ctxp->temp_htab)
gimplify_ctxp->temp_htab = new hash_table<gimplify_hasher> (1000);
slot = gimplify_ctxp->temp_htab->find_slot (&elt, INSERT);
if (*slot == NULL)
{
elt_p = XNEW (elt_t);
elt_p->val = val;
elt_p->temp = ret = create_tmp_from_val (val);
*slot = elt_p;
}
else
{
elt_p = *slot;
ret = elt_p->temp;
}
}
return ret;
}
static tree
internal_get_tmp_var (tree val, gimple_seq *pre_p, gimple_seq *post_p,
bool is_formal, bool allow_ssa)
{
tree t, mod;
gimplify_expr (&val, pre_p, post_p, is_gimple_reg_rhs_or_call,
fb_rvalue);
if (allow_ssa
&& gimplify_ctxp->into_ssa
&& is_gimple_reg_type (TREE_TYPE (val)))
{
t = make_ssa_name (TYPE_MAIN_VARIANT (TREE_TYPE (val)));
if (! gimple_in_ssa_p (cfun))
{
const char *name = get_name (val);
if (name)
SET_SSA_NAME_VAR_OR_IDENTIFIER (t, create_tmp_var_name (name));
}
}
else
t = lookup_tmp_var (val, is_formal);
mod = build2 (INIT_EXPR, TREE_TYPE (t), t, unshare_expr (val));
SET_EXPR_LOCATION (mod, EXPR_LOC_OR_LOC (val, input_location));
gimplify_and_add (mod, pre_p);
ggc_free (mod);
return t;
}
tree
get_formal_tmp_var (tree val, gimple_seq *pre_p)
{
return internal_get_tmp_var (val, pre_p, NULL, true, true);
}
tree
get_initialized_tmp_var (tree val, gimple_seq *pre_p, gimple_seq *post_p,
bool allow_ssa)
{
return internal_get_tmp_var (val, pre_p, post_p, false, allow_ssa);
}
void
declare_vars (tree vars, gimple *gs, bool debug_info)
{
tree last = vars;
if (last)
{
tree temps, block;
gbind *scope = as_a <gbind *> (gs);
temps = nreverse (last);
block = gimple_bind_block (scope);
gcc_assert (!block || TREE_CODE (block) == BLOCK);
if (!block || !debug_info)
{
DECL_CHAIN (last) = gimple_bind_vars (scope);
gimple_bind_set_vars (scope, temps);
}
else
{
if (BLOCK_VARS (block))
BLOCK_VARS (block) = chainon (BLOCK_VARS (block), temps);
else
{
gimple_bind_set_vars (scope,
chainon (gimple_bind_vars (scope), temps));
BLOCK_VARS (block) = temps;
}
}
}
}
static void
force_constant_size (tree var)
{
HOST_WIDE_INT max_size;
gcc_assert (VAR_P (var));
max_size = max_int_size_in_bytes (TREE_TYPE (var));
gcc_assert (max_size >= 0);
DECL_SIZE_UNIT (var)
= build_int_cst (TREE_TYPE (DECL_SIZE_UNIT (var)), max_size);
DECL_SIZE (var)
= build_int_cst (TREE_TYPE (DECL_SIZE (var)), max_size * BITS_PER_UNIT);
}
void
gimple_add_tmp_var_fn (struct function *fn, tree tmp)
{
gcc_assert (!DECL_CHAIN (tmp) && !DECL_SEEN_IN_BIND_EXPR_P (tmp));
if (!tree_fits_poly_uint64_p (DECL_SIZE_UNIT (tmp)))
force_constant_size (tmp);
DECL_CONTEXT (tmp) = fn->decl;
DECL_SEEN_IN_BIND_EXPR_P (tmp) = 1;
record_vars_into (tmp, fn->decl);
}
void
gimple_add_tmp_var (tree tmp)
{
gcc_assert (!DECL_CHAIN (tmp) && !DECL_SEEN_IN_BIND_EXPR_P (tmp));
if (!tree_fits_poly_uint64_p (DECL_SIZE_UNIT (tmp)))
force_constant_size (tmp);
DECL_CONTEXT (tmp) = current_function_decl;
DECL_SEEN_IN_BIND_EXPR_P (tmp) = 1;
if (gimplify_ctxp)
{
DECL_CHAIN (tmp) = gimplify_ctxp->temps;
gimplify_ctxp->temps = tmp;
if (gimplify_omp_ctxp)
{
struct gimplify_omp_ctx *ctx = gimplify_omp_ctxp;
while (ctx
&& (ctx->region_type == ORT_WORKSHARE
|| ctx->region_type == ORT_SIMD
|| ctx->region_type == ORT_ACC))
ctx = ctx->outer_context;
if (ctx)
omp_add_variable (ctx, tmp, GOVD_LOCAL | GOVD_SEEN);
}
}
else if (cfun)
record_vars (tmp);
else
{
gimple_seq body_seq;
body_seq = gimple_body (current_function_decl);
declare_vars (tmp, gimple_seq_first_stmt (body_seq), false);
}
}

static tree
mostly_copy_tree_r (tree *tp, int *walk_subtrees, void *data)
{
tree t = *tp;
enum tree_code code = TREE_CODE (t);
if (code == SAVE_EXPR || code == TARGET_EXPR || code == BIND_EXPR)
{
if (data && !((hash_set<tree> *)data)->add (t))
;
else
*walk_subtrees = 0;
}
else if (TREE_CODE_CLASS (code) == tcc_type
|| TREE_CODE_CLASS (code) == tcc_declaration
|| TREE_CODE_CLASS (code) == tcc_constant)
*walk_subtrees = 0;
else if (code == STATEMENT_LIST)
;
else
copy_tree_r (tp, walk_subtrees, NULL);
return NULL_TREE;
}
static tree
copy_if_shared_r (tree *tp, int *walk_subtrees, void *data)
{
tree t = *tp;
enum tree_code code = TREE_CODE (t);
if (TREE_CODE_CLASS (code) == tcc_type
|| TREE_CODE_CLASS (code) == tcc_declaration
|| TREE_CODE_CLASS (code) == tcc_constant)
{
if (TREE_VISITED (t))
*walk_subtrees = 0;
else
TREE_VISITED (t) = 1;
}
else if (TREE_VISITED (t))
{
walk_tree (tp, mostly_copy_tree_r, data, NULL);
*walk_subtrees = 0;
}
else
TREE_VISITED (t) = 1;
return NULL_TREE;
}
static inline void
copy_if_shared (tree *tp, void *data)
{
walk_tree (tp, copy_if_shared_r, data, NULL);
}
static void
unshare_body (tree fndecl)
{
struct cgraph_node *cgn = cgraph_node::get (fndecl);
hash_set<tree> *visited
= lang_hooks.deep_unsharing ? new hash_set<tree> : NULL;
copy_if_shared (&DECL_SAVED_TREE (fndecl), visited);
copy_if_shared (&DECL_SIZE (DECL_RESULT (fndecl)), visited);
copy_if_shared (&DECL_SIZE_UNIT (DECL_RESULT (fndecl)), visited);
delete visited;
if (cgn)
for (cgn = cgn->nested; cgn; cgn = cgn->next_nested)
unshare_body (cgn->decl);
}
static tree
unmark_visited_r (tree *tp, int *walk_subtrees, void *data ATTRIBUTE_UNUSED)
{
tree t = *tp;
if (TREE_VISITED (t))
TREE_VISITED (t) = 0;
else
*walk_subtrees = 0;
return NULL_TREE;
}
static inline void
unmark_visited (tree *tp)
{
walk_tree (tp, unmark_visited_r, NULL, NULL);
}
static void
unvisit_body (tree fndecl)
{
struct cgraph_node *cgn = cgraph_node::get (fndecl);
unmark_visited (&DECL_SAVED_TREE (fndecl));
unmark_visited (&DECL_SIZE (DECL_RESULT (fndecl)));
unmark_visited (&DECL_SIZE_UNIT (DECL_RESULT (fndecl)));
if (cgn)
for (cgn = cgn->nested; cgn; cgn = cgn->next_nested)
unvisit_body (cgn->decl);
}
tree
unshare_expr (tree expr)
{
walk_tree (&expr, mostly_copy_tree_r, NULL, NULL);
return expr;
}
static tree
prune_expr_location (tree *tp, int *walk_subtrees, void *)
{
if (EXPR_P (*tp))
SET_EXPR_LOCATION (*tp, UNKNOWN_LOCATION);
else
*walk_subtrees = 0;
return NULL_TREE;
}
tree
unshare_expr_without_location (tree expr)
{
walk_tree (&expr, mostly_copy_tree_r, NULL, NULL);
if (EXPR_P (expr))
walk_tree (&expr, prune_expr_location, NULL, NULL);
return expr;
}
static location_t
rexpr_location (tree expr, location_t or_else = UNKNOWN_LOCATION)
{
if (!expr)
return or_else;
if (EXPR_HAS_LOCATION (expr))
return EXPR_LOCATION (expr);
if (TREE_CODE (expr) != STATEMENT_LIST)
return or_else;
tree_stmt_iterator i = tsi_start (expr);
bool found = false;
while (!tsi_end_p (i) && TREE_CODE (tsi_stmt (i)) == DEBUG_BEGIN_STMT)
{
found = true;
tsi_next (&i);
}
if (!found || !tsi_one_before_end_p (i))
return or_else;
return rexpr_location (tsi_stmt (i), or_else);
}
static inline bool
rexpr_has_location (tree expr)
{
return rexpr_location (expr) != UNKNOWN_LOCATION;
}

tree
voidify_wrapper_expr (tree wrapper, tree temp)
{
tree type = TREE_TYPE (wrapper);
if (type && !VOID_TYPE_P (type))
{
tree *p;
for (p = &wrapper; p && *p; )
{
switch (TREE_CODE (*p))
{
case BIND_EXPR:
TREE_SIDE_EFFECTS (*p) = 1;
TREE_TYPE (*p) = void_type_node;
p = &BIND_EXPR_BODY (*p);
break;
case CLEANUP_POINT_EXPR:
case TRY_FINALLY_EXPR:
case TRY_CATCH_EXPR:
TREE_SIDE_EFFECTS (*p) = 1;
TREE_TYPE (*p) = void_type_node;
p = &TREE_OPERAND (*p, 0);
break;
case STATEMENT_LIST:
{
tree_stmt_iterator i = tsi_last (*p);
TREE_SIDE_EFFECTS (*p) = 1;
TREE_TYPE (*p) = void_type_node;
p = tsi_end_p (i) ? NULL : tsi_stmt_ptr (i);
}
break;
case COMPOUND_EXPR:
for (; TREE_CODE (*p) == COMPOUND_EXPR; p = &TREE_OPERAND (*p, 1))
{
TREE_SIDE_EFFECTS (*p) = 1;
TREE_TYPE (*p) = void_type_node;
}
break;
case TRANSACTION_EXPR:
TREE_SIDE_EFFECTS (*p) = 1;
TREE_TYPE (*p) = void_type_node;
p = &TRANSACTION_EXPR_BODY (*p);
break;
default:
if (p == &wrapper)
{
TREE_SIDE_EFFECTS (*p) = 1;
TREE_TYPE (*p) = void_type_node;
p = &TREE_OPERAND (*p, 0);
break;
}
goto out;
}
}
out:
if (p == NULL || IS_EMPTY_STMT (*p))
temp = NULL_TREE;
else if (temp)
{
gcc_assert (TREE_CODE (temp) == INIT_EXPR
|| TREE_CODE (temp) == MODIFY_EXPR);
TREE_OPERAND (temp, 1) = *p;
*p = temp;
}
else
{
temp = create_tmp_var (type, "retval");
*p = build2 (INIT_EXPR, type, temp, *p);
}
return temp;
}
return NULL_TREE;
}
static void
build_stack_save_restore (gcall **save, gcall **restore)
{
tree tmp_var;
*save = gimple_build_call (builtin_decl_implicit (BUILT_IN_STACK_SAVE), 0);
tmp_var = create_tmp_var (ptr_type_node, "saved_stack");
gimple_call_set_lhs (*save, tmp_var);
*restore
= gimple_build_call (builtin_decl_implicit (BUILT_IN_STACK_RESTORE),
1, tmp_var);
}
static tree
build_asan_poison_call_expr (tree decl)
{
tree unit_size = DECL_SIZE_UNIT (decl);
if (zerop (unit_size))
return NULL_TREE;
tree base = build_fold_addr_expr (decl);
return build_call_expr_internal_loc (UNKNOWN_LOCATION, IFN_ASAN_MARK,
void_type_node, 3,
build_int_cst (integer_type_node,
ASAN_MARK_POISON),
base, unit_size);
}
static void
asan_poison_variable (tree decl, bool poison, gimple_stmt_iterator *it,
bool before)
{
tree unit_size = DECL_SIZE_UNIT (decl);
tree base = build_fold_addr_expr (decl);
if (zerop (unit_size))
return;
if (DECL_ALIGN_UNIT (decl) <= ASAN_SHADOW_GRANULARITY)
SET_DECL_ALIGN (decl, BITS_PER_UNIT * ASAN_SHADOW_GRANULARITY);
HOST_WIDE_INT flags = poison ? ASAN_MARK_POISON : ASAN_MARK_UNPOISON;
gimple *g
= gimple_build_call_internal (IFN_ASAN_MARK, 3,
build_int_cst (integer_type_node, flags),
base, unit_size);
if (before)
gsi_insert_before (it, g, GSI_NEW_STMT);
else
gsi_insert_after (it, g, GSI_NEW_STMT);
}
static void
asan_poison_variable (tree decl, bool poison, gimple_seq *seq_p)
{
gimple_stmt_iterator it = gsi_last (*seq_p);
bool before = false;
if (gsi_end_p (it))
before = true;
asan_poison_variable (decl, poison, &it, before);
}
static int
sort_by_decl_uid (const void *a, const void *b)
{
const tree *t1 = (const tree *)a;
const tree *t2 = (const tree *)b;
int uid1 = DECL_UID (*t1);
int uid2 = DECL_UID (*t2);
if (uid1 < uid2)
return -1;
else if (uid1 > uid2)
return 1;
else
return 0;
}
static void
asan_poison_variables (hash_set<tree> *variables, bool poison, gimple_seq *seq_p)
{
unsigned c = variables->elements ();
if (c == 0)
return;
auto_vec<tree> sorted_variables (c);
for (hash_set<tree>::iterator it = variables->begin ();
it != variables->end (); ++it)
sorted_variables.safe_push (*it);
sorted_variables.qsort (sort_by_decl_uid);
unsigned i;
tree var;
FOR_EACH_VEC_ELT (sorted_variables, i, var)
{
asan_poison_variable (var, poison, seq_p);
if (!lookup_attribute (ASAN_USE_AFTER_SCOPE_ATTRIBUTE,
DECL_ATTRIBUTES (var)))
DECL_ATTRIBUTES (var)
= tree_cons (get_identifier (ASAN_USE_AFTER_SCOPE_ATTRIBUTE),
integer_one_node,
DECL_ATTRIBUTES (var));
}
}
static enum gimplify_status
gimplify_bind_expr (tree *expr_p, gimple_seq *pre_p)
{
tree bind_expr = *expr_p;
bool old_keep_stack = gimplify_ctxp->keep_stack;
bool old_save_stack = gimplify_ctxp->save_stack;
tree t;
gbind *bind_stmt;
gimple_seq body, cleanup;
gcall *stack_save;
location_t start_locus = 0, end_locus = 0;
tree ret_clauses = NULL;
tree temp = voidify_wrapper_expr (bind_expr, NULL);
for (t = BIND_EXPR_VARS (bind_expr); t ; t = DECL_CHAIN (t))
{
if (VAR_P (t))
{
struct gimplify_omp_ctx *ctx = gimplify_omp_ctxp;
if (ctx && ctx->region_type != ORT_NONE && !DECL_EXTERNAL (t)
&& (! DECL_SEEN_IN_BIND_EXPR_P (t)
|| splay_tree_lookup (ctx->variables,
(splay_tree_key) t) == NULL))
{
if (ctx->region_type == ORT_SIMD
&& TREE_ADDRESSABLE (t)
&& !TREE_STATIC (t))
omp_add_variable (ctx, t, GOVD_PRIVATE | GOVD_SEEN);
else
omp_add_variable (ctx, t, GOVD_LOCAL | GOVD_SEEN);
}
DECL_SEEN_IN_BIND_EXPR_P (t) = 1;
if (DECL_HARD_REGISTER (t) && !is_global_var (t) && cfun)
cfun->has_local_explicit_reg_vars = true;
}
if ((TREE_CODE (TREE_TYPE (t)) == COMPLEX_TYPE
|| TREE_CODE (TREE_TYPE (t)) == VECTOR_TYPE)
&& !TREE_THIS_VOLATILE (t)
&& (VAR_P (t) && !DECL_HARD_REGISTER (t))
&& !needs_to_live_in_memory (t))
DECL_GIMPLE_REG_P (t) = 1;
}
bind_stmt = gimple_build_bind (BIND_EXPR_VARS (bind_expr), NULL,
BIND_EXPR_BLOCK (bind_expr));
gimple_push_bind_expr (bind_stmt);
gimplify_ctxp->keep_stack = false;
gimplify_ctxp->save_stack = false;
body = NULL;
gimplify_stmt (&BIND_EXPR_BODY (bind_expr), &body);
gimple_bind_set_body (bind_stmt, body);
if (BIND_EXPR_BLOCK (bind_expr))
{
end_locus = BLOCK_SOURCE_END_LOCATION (BIND_EXPR_BLOCK (bind_expr));
start_locus = BLOCK_SOURCE_LOCATION (BIND_EXPR_BLOCK (bind_expr));
}
if (start_locus == 0)
start_locus = EXPR_LOCATION (bind_expr);
cleanup = NULL;
stack_save = NULL;
if (gimplify_ctxp->save_stack && !gimplify_ctxp->keep_stack)
{
gcall *stack_restore;
build_stack_save_restore (&stack_save, &stack_restore);
gimple_set_location (stack_save, start_locus);
gimple_set_location (stack_restore, end_locus);
gimplify_seq_add_stmt (&cleanup, stack_restore);
}
for (t = BIND_EXPR_VARS (bind_expr); t ; t = DECL_CHAIN (t))
{
if (VAR_P (t)
&& !is_global_var (t)
&& DECL_CONTEXT (t) == current_function_decl)
{
if (!DECL_HARD_REGISTER (t)
&& !TREE_THIS_VOLATILE (t)
&& !DECL_HAS_VALUE_EXPR_P (t)
&& !is_gimple_reg (t)
&& flag_stack_reuse != SR_NONE)
{
tree clobber = build_constructor (TREE_TYPE (t), NULL);
gimple *clobber_stmt;
TREE_THIS_VOLATILE (clobber) = 1;
clobber_stmt = gimple_build_assign (t, clobber);
gimple_set_location (clobber_stmt, end_locus);
gimplify_seq_add_stmt (&cleanup, clobber_stmt);
}
if (flag_openacc && oacc_declare_returns != NULL)
{
tree *c = oacc_declare_returns->get (t);
if (c != NULL)
{
if (ret_clauses)
OMP_CLAUSE_CHAIN (*c) = ret_clauses;
ret_clauses = *c;
oacc_declare_returns->remove (t);
if (oacc_declare_returns->elements () == 0)
{
delete oacc_declare_returns;
oacc_declare_returns = NULL;
}
}
}
}
if (asan_poisoned_variables != NULL
&& asan_poisoned_variables->contains (t))
{
asan_poisoned_variables->remove (t);
asan_poison_variable (t, true, &cleanup);
}
if (gimplify_ctxp->live_switch_vars != NULL
&& gimplify_ctxp->live_switch_vars->contains (t))
gimplify_ctxp->live_switch_vars->remove (t);
}
if (ret_clauses)
{
gomp_target *stmt;
gimple_stmt_iterator si = gsi_start (cleanup);
stmt = gimple_build_omp_target (NULL, GF_OMP_TARGET_KIND_OACC_DECLARE,
ret_clauses);
gsi_insert_seq_before_without_update (&si, stmt, GSI_NEW_STMT);
}
if (cleanup)
{
gtry *gs;
gimple_seq new_body;
new_body = NULL;
gs = gimple_build_try (gimple_bind_body (bind_stmt), cleanup,
GIMPLE_TRY_FINALLY);
if (stack_save)
gimplify_seq_add_stmt (&new_body, stack_save);
gimplify_seq_add_stmt (&new_body, gs);
gimple_bind_set_body (bind_stmt, new_body);
}
if (!gimplify_ctxp->keep_stack)
gimplify_ctxp->keep_stack = old_keep_stack;
gimplify_ctxp->save_stack = old_save_stack;
gimple_pop_bind_expr ();
gimplify_seq_add_stmt (pre_p, bind_stmt);
if (temp)
{
*expr_p = temp;
return GS_OK;
}
*expr_p = NULL_TREE;
return GS_ALL_DONE;
}
static void
maybe_add_early_return_predict_stmt (gimple_seq *pre_p)
{
if (gimple_conditional_context ())
{
gimple *predict = gimple_build_predict (PRED_TREE_EARLY_RETURN,
NOT_TAKEN);
gimplify_seq_add_stmt (pre_p, predict);
}
}
static enum gimplify_status
gimplify_return_expr (tree stmt, gimple_seq *pre_p)
{
greturn *ret;
tree ret_expr = TREE_OPERAND (stmt, 0);
tree result_decl, result;
if (ret_expr == error_mark_node)
return GS_ERROR;
if (!ret_expr
|| TREE_CODE (ret_expr) == RESULT_DECL)
{
maybe_add_early_return_predict_stmt (pre_p);
greturn *ret = gimple_build_return (ret_expr);
gimple_set_no_warning (ret, TREE_NO_WARNING (stmt));
gimplify_seq_add_stmt (pre_p, ret);
return GS_ALL_DONE;
}
if (VOID_TYPE_P (TREE_TYPE (TREE_TYPE (current_function_decl))))
result_decl = NULL_TREE;
else
{
result_decl = TREE_OPERAND (ret_expr, 0);
if (TREE_CODE (result_decl) == INDIRECT_REF)
result_decl = TREE_OPERAND (result_decl, 0);
gcc_assert ((TREE_CODE (ret_expr) == MODIFY_EXPR
|| TREE_CODE (ret_expr) == INIT_EXPR)
&& TREE_CODE (result_decl) == RESULT_DECL);
}
if (!result_decl)
result = NULL_TREE;
else if (aggregate_value_p (result_decl, TREE_TYPE (current_function_decl)))
{
if (TREE_CODE (DECL_SIZE (result_decl)) != INTEGER_CST)
{
if (!TYPE_SIZES_GIMPLIFIED (TREE_TYPE (result_decl)))
gimplify_type_sizes (TREE_TYPE (result_decl), pre_p);
gimplify_one_sizepos (&DECL_SIZE (result_decl), pre_p);
gimplify_one_sizepos (&DECL_SIZE_UNIT (result_decl), pre_p);
}
result = result_decl;
}
else if (gimplify_ctxp->return_temp)
result = gimplify_ctxp->return_temp;
else
{
result = create_tmp_reg (TREE_TYPE (result_decl));
TREE_NO_WARNING (result) = 1;
gimplify_ctxp->return_temp = result;
}
if (result != result_decl)
TREE_OPERAND (ret_expr, 0) = result;
gimplify_and_add (TREE_OPERAND (stmt, 0), pre_p);
maybe_add_early_return_predict_stmt (pre_p);
ret = gimple_build_return (result);
gimple_set_no_warning (ret, TREE_NO_WARNING (stmt));
gimplify_seq_add_stmt (pre_p, ret);
return GS_ALL_DONE;
}
static void
gimplify_vla_decl (tree decl, gimple_seq *seq_p)
{
tree t, addr, ptr_type;
gimplify_one_sizepos (&DECL_SIZE (decl), seq_p);
gimplify_one_sizepos (&DECL_SIZE_UNIT (decl), seq_p);
if (DECL_HAS_VALUE_EXPR_P (decl))
return;
ptr_type = build_pointer_type (TREE_TYPE (decl));
addr = create_tmp_var (ptr_type, get_name (decl));
DECL_IGNORED_P (addr) = 0;
t = build_fold_indirect_ref (addr);
TREE_THIS_NOTRAP (t) = 1;
SET_DECL_VALUE_EXPR (decl, t);
DECL_HAS_VALUE_EXPR_P (decl) = 1;
t = build_alloca_call_expr (DECL_SIZE_UNIT (decl), DECL_ALIGN (decl),
max_int_size_in_bytes (TREE_TYPE (decl)));
CALL_ALLOCA_FOR_VAR_P (t) = 1;
t = fold_convert (ptr_type, t);
t = build2 (MODIFY_EXPR, TREE_TYPE (addr), addr, t);
gimplify_and_add (t, seq_p);
}
static tree
force_labels_r (tree *tp, int *walk_subtrees, void *data ATTRIBUTE_UNUSED)
{
if (TYPE_P (*tp))
*walk_subtrees = 0;
if (TREE_CODE (*tp) == LABEL_DECL)
{
FORCED_LABEL (*tp) = 1;
cfun->has_forced_label_in_static = 1;
}
return NULL_TREE;
}
static enum gimplify_status
gimplify_decl_expr (tree *stmt_p, gimple_seq *seq_p)
{
tree stmt = *stmt_p;
tree decl = DECL_EXPR_DECL (stmt);
*stmt_p = NULL_TREE;
if (TREE_TYPE (decl) == error_mark_node)
return GS_ERROR;
if ((TREE_CODE (decl) == TYPE_DECL
|| VAR_P (decl))
&& !TYPE_SIZES_GIMPLIFIED (TREE_TYPE (decl)))
{
gimplify_type_sizes (TREE_TYPE (decl), seq_p);
if (TREE_CODE (TREE_TYPE (decl)) == REFERENCE_TYPE)
gimplify_type_sizes (TREE_TYPE (TREE_TYPE (decl)), seq_p);
}
if (TREE_CODE (decl) == TYPE_DECL
&& DECL_ORIGINAL_TYPE (decl)
&& !TYPE_SIZES_GIMPLIFIED (DECL_ORIGINAL_TYPE (decl)))
{
gimplify_type_sizes (DECL_ORIGINAL_TYPE (decl), seq_p);
if (TREE_CODE (DECL_ORIGINAL_TYPE (decl)) == REFERENCE_TYPE)
gimplify_type_sizes (TREE_TYPE (DECL_ORIGINAL_TYPE (decl)), seq_p);
}
if (VAR_P (decl) && !DECL_EXTERNAL (decl))
{
tree init = DECL_INITIAL (decl);
bool is_vla = false;
if (TREE_CODE (DECL_SIZE_UNIT (decl)) != INTEGER_CST
|| (!TREE_STATIC (decl)
&& flag_stack_check == GENERIC_STACK_CHECK
&& compare_tree_int (DECL_SIZE_UNIT (decl),
STACK_CHECK_MAX_VAR_SIZE) > 0))
{
gimplify_vla_decl (decl, seq_p);
is_vla = true;
}
if (asan_poisoned_variables
&& !is_vla
&& TREE_ADDRESSABLE (decl)
&& !TREE_STATIC (decl)
&& !DECL_HAS_VALUE_EXPR_P (decl)
&& DECL_ALIGN (decl) <= MAX_SUPPORTED_STACK_ALIGNMENT
&& dbg_cnt (asan_use_after_scope)
&& !gimplify_omp_ctxp)
{
asan_poisoned_variables->add (decl);
asan_poison_variable (decl, false, seq_p);
if (!DECL_ARTIFICIAL (decl) && gimplify_ctxp->live_switch_vars)
gimplify_ctxp->live_switch_vars->add (decl);
}
if (!DECL_SEEN_IN_BIND_EXPR_P (decl)
&& DECL_ARTIFICIAL (decl) && DECL_NAME (decl) == NULL_TREE)
gimple_add_tmp_var (decl);
if (init && init != error_mark_node)
{
if (!TREE_STATIC (decl))
{
DECL_INITIAL (decl) = NULL_TREE;
init = build2 (INIT_EXPR, void_type_node, decl, init);
gimplify_and_add (init, seq_p);
ggc_free (init);
}
else
walk_tree (&init, force_labels_r, NULL, NULL);
}
}
return GS_ALL_DONE;
}
static enum gimplify_status
gimplify_loop_expr (tree *expr_p, gimple_seq *pre_p)
{
tree saved_label = gimplify_ctxp->exit_label;
tree start_label = create_artificial_label (UNKNOWN_LOCATION);
gimplify_seq_add_stmt (pre_p, gimple_build_label (start_label));
gimplify_ctxp->exit_label = NULL_TREE;
gimplify_and_add (LOOP_EXPR_BODY (*expr_p), pre_p);
gimplify_seq_add_stmt (pre_p, gimple_build_goto (start_label));
if (gimplify_ctxp->exit_label)
gimplify_seq_add_stmt (pre_p,
gimple_build_label (gimplify_ctxp->exit_label));
gimplify_ctxp->exit_label = saved_label;
*expr_p = NULL;
return GS_ALL_DONE;
}
static enum gimplify_status
gimplify_statement_list (tree *expr_p, gimple_seq *pre_p)
{
tree temp = voidify_wrapper_expr (*expr_p, NULL);
tree_stmt_iterator i = tsi_start (*expr_p);
while (!tsi_end_p (i))
{
gimplify_stmt (tsi_stmt_ptr (i), pre_p);
tsi_delink (&i);
}
if (temp)
{
*expr_p = temp;
return GS_OK;
}
return GS_ALL_DONE;
}
static tree
warn_switch_unreachable_r (gimple_stmt_iterator *gsi_p, bool *handled_ops_p,
struct walk_stmt_info *wi)
{
gimple *stmt = gsi_stmt (*gsi_p);
*handled_ops_p = true;
switch (gimple_code (stmt))
{
case GIMPLE_TRY:
if (gimple_try_eval (stmt) == NULL)
{
wi->info = stmt;
return integer_zero_node;
}
case GIMPLE_BIND:
case GIMPLE_CATCH:
case GIMPLE_EH_FILTER:
case GIMPLE_TRANSACTION:
*handled_ops_p = false;
break;
case GIMPLE_DEBUG:
break;
case GIMPLE_CALL:
if (gimple_call_internal_p (stmt, IFN_ASAN_MARK))
{
*handled_ops_p = false;
break;
}
default:
wi->info = stmt;
return integer_zero_node;
}
return NULL_TREE;
}
static void
maybe_warn_switch_unreachable (gimple_seq seq)
{
if (!warn_switch_unreachable
|| lang_GNU_Fortran ()
|| seq == NULL)
return;
struct walk_stmt_info wi;
memset (&wi, 0, sizeof (wi));
walk_gimple_seq (seq, warn_switch_unreachable_r, NULL, &wi);
gimple *stmt = (gimple *) wi.info;
if (stmt && gimple_code (stmt) != GIMPLE_LABEL)
{
if (gimple_code (stmt) == GIMPLE_GOTO
&& TREE_CODE (gimple_goto_dest (stmt)) == LABEL_DECL
&& DECL_ARTIFICIAL (gimple_goto_dest (stmt)))
;
else
warning_at (gimple_location (stmt), OPT_Wswitch_unreachable,
"statement will never be executed");
}
}
struct label_entry
{
tree label;
location_t loc;
};
static struct label_entry *
find_label_entry (const auto_vec<struct label_entry> *vec, tree label)
{
unsigned int i;
struct label_entry *l;
FOR_EACH_VEC_ELT (*vec, i, l)
if (l->label == label)
return l;
return NULL;
}
static bool
case_label_p (const vec<tree> *cases, tree label)
{
unsigned int i;
tree l;
FOR_EACH_VEC_ELT (*cases, i, l)
if (CASE_LABEL (l) == label)
return true;
return false;
}
static gimple *
last_stmt_in_scope (gimple *stmt)
{
if (!stmt)
return NULL;
switch (gimple_code (stmt))
{
case GIMPLE_BIND:
{
gbind *bind = as_a <gbind *> (stmt);
stmt = gimple_seq_last_nondebug_stmt (gimple_bind_body (bind));
return last_stmt_in_scope (stmt);
}
case GIMPLE_TRY:
{
gtry *try_stmt = as_a <gtry *> (stmt);
stmt = gimple_seq_last_nondebug_stmt (gimple_try_eval (try_stmt));
gimple *last_eval = last_stmt_in_scope (stmt);
if (gimple_stmt_may_fallthru (last_eval)
&& (last_eval == NULL
|| !gimple_call_internal_p (last_eval, IFN_FALLTHROUGH))
&& gimple_try_kind (try_stmt) == GIMPLE_TRY_FINALLY)
{
stmt = gimple_seq_last_nondebug_stmt (gimple_try_cleanup (try_stmt));
return last_stmt_in_scope (stmt);
}
else
return last_eval;
}
case GIMPLE_DEBUG:
gcc_unreachable ();
default:
return stmt;
}
}
static gimple *
collect_fallthrough_labels (gimple_stmt_iterator *gsi_p,
auto_vec <struct label_entry> *labels)
{
gimple *prev = NULL;
do
{
if (gimple_code (gsi_stmt (*gsi_p)) == GIMPLE_BIND)
{
gbind *bind = as_a <gbind *> (gsi_stmt (*gsi_p));
gimple *first = gimple_seq_first_stmt (gimple_bind_body (bind));
gimple *last = gimple_seq_last_stmt (gimple_bind_body (bind));
if (last
&& gimple_code (first) == GIMPLE_SWITCH
&& gimple_code (last) == GIMPLE_LABEL)
{
tree label = gimple_label_label (as_a <glabel *> (last));
if (SWITCH_BREAK_LABEL_P (label))
{
prev = bind;
gsi_next (gsi_p);
continue;
}
}
}
if (gimple_code (gsi_stmt (*gsi_p)) == GIMPLE_BIND
|| gimple_code (gsi_stmt (*gsi_p)) == GIMPLE_TRY)
{
location_t bind_loc = gimple_location (gsi_stmt (*gsi_p));
gimple *last = last_stmt_in_scope (gsi_stmt (*gsi_p));
if (last)
{
prev = last;
if (!gimple_has_location (prev))
gimple_set_location (prev, bind_loc);
}
gsi_next (gsi_p);
continue;
}
if (gimple_code (gsi_stmt (*gsi_p)) == GIMPLE_COND)
{
gcond *cond_stmt = as_a <gcond *> (gsi_stmt (*gsi_p));
tree false_lab = gimple_cond_false_label (cond_stmt);
location_t if_loc = gimple_location (cond_stmt);
if (!DECL_ARTIFICIAL (false_lab))
break;
for (; !gsi_end_p (*gsi_p); gsi_next (gsi_p))
{
gimple *stmt = gsi_stmt (*gsi_p);
if (gimple_code (stmt) == GIMPLE_LABEL
&& gimple_label_label (as_a <glabel *> (stmt)) == false_lab)
break;
}
if (gsi_end_p (*gsi_p))
break;
struct label_entry l = { false_lab, if_loc };
labels->safe_push (l);
gsi_prev (gsi_p);
if (gimple_code (gsi_stmt (*gsi_p)) == GIMPLE_GOTO
&& !gimple_has_location (gsi_stmt (*gsi_p)))
{
gsi_prev (gsi_p);
bool fallthru_before_dest
= gimple_call_internal_p (gsi_stmt (*gsi_p), IFN_FALLTHROUGH);
gsi_next (gsi_p);
tree goto_dest = gimple_goto_dest (gsi_stmt (*gsi_p));
if (!fallthru_before_dest)
{
struct label_entry l = { goto_dest, if_loc };
labels->safe_push (l);
}
}
gsi_next (gsi_p);
}
if (gimple_code (gsi_stmt (*gsi_p)) == GIMPLE_LABEL)
{
tree label = gimple_label_label (as_a <glabel *> (gsi_stmt (*gsi_p)));
if (find_label_entry (labels, label))
prev = gsi_stmt (*gsi_p);
}
else if (gimple_call_internal_p (gsi_stmt (*gsi_p), IFN_ASAN_MARK))
;
else if (!is_gimple_debug (gsi_stmt (*gsi_p)))
prev = gsi_stmt (*gsi_p);
gsi_next (gsi_p);
}
while (!gsi_end_p (*gsi_p)
&& (gimple_code (gsi_stmt (*gsi_p)) != GIMPLE_LABEL
|| !gimple_has_location (gsi_stmt (*gsi_p))));
return prev;
}
static bool
should_warn_for_implicit_fallthrough (gimple_stmt_iterator *gsi_p, tree label)
{
gimple_stmt_iterator gsi = *gsi_p;
if (FALLTHROUGH_LABEL_P (label))
return false;
if (!case_label_p (&gimplify_ctxp->case_labels, label))
{
tree l;
while (!gsi_end_p (gsi)
&& gimple_code (gsi_stmt (gsi)) == GIMPLE_LABEL
&& (l = gimple_label_label (as_a <glabel *> (gsi_stmt (gsi))))
&& !case_label_p (&gimplify_ctxp->case_labels, l))
gsi_next_nondebug (&gsi);
if (gsi_end_p (gsi) || gimple_code (gsi_stmt (gsi)) != GIMPLE_LABEL)
return false;
}
gsi = *gsi_p;
while (!gsi_end_p (gsi)
&& (gimple_code (gsi_stmt (gsi)) == GIMPLE_LABEL
|| gimple_code (gsi_stmt (gsi)) == GIMPLE_PREDICT))
gsi_next_nondebug (&gsi);
if (gsi_end_p (gsi)
|| gimple_code (gsi_stmt (gsi)) == GIMPLE_GOTO
|| gimple_code (gsi_stmt (gsi)) == GIMPLE_RETURN)
return false;
return true;
}
static tree
warn_implicit_fallthrough_r (gimple_stmt_iterator *gsi_p, bool *handled_ops_p,
struct walk_stmt_info *)
{
gimple *stmt = gsi_stmt (*gsi_p);
*handled_ops_p = true;
switch (gimple_code (stmt))
{
case GIMPLE_TRY:
case GIMPLE_BIND:
case GIMPLE_CATCH:
case GIMPLE_EH_FILTER:
case GIMPLE_TRANSACTION:
*handled_ops_p = false;
break;
case GIMPLE_LABEL:
{
while (!gsi_end_p (*gsi_p)
&& gimple_code (gsi_stmt (*gsi_p)) == GIMPLE_LABEL)
gsi_next_nondebug (gsi_p);
if (gsi_end_p (*gsi_p))
return integer_zero_node;
auto_vec <struct label_entry> labels;
gimple *prev = collect_fallthrough_labels (gsi_p, &labels);
if (gsi_end_p (*gsi_p))
return integer_zero_node;
gimple *next = gsi_stmt (*gsi_p);
tree label;
if (gimple_code (next) == GIMPLE_LABEL
&& gimple_has_location (next)
&& (label = gimple_label_label (as_a <glabel *> (next)))
&& prev != NULL)
{
struct label_entry *l;
bool warned_p = false;
if (!should_warn_for_implicit_fallthrough (gsi_p, label))
;
else if (gimple_code (prev) == GIMPLE_LABEL
&& (label = gimple_label_label (as_a <glabel *> (prev)))
&& (l = find_label_entry (&labels, label)))
warned_p = warning_at (l->loc, OPT_Wimplicit_fallthrough_,
"this statement may fall through");
else if (!gimple_call_internal_p (prev, IFN_FALLTHROUGH)
&& gimple_stmt_may_fallthru (prev)
&& gimple_has_location (prev))
warned_p = warning_at (gimple_location (prev),
OPT_Wimplicit_fallthrough_,
"this statement may fall through");
if (warned_p)
inform (gimple_location (next), "here");
FALLTHROUGH_LABEL_P (label) = true;
gsi_prev (gsi_p);
}
}
break;
default:
break;
}
return NULL_TREE;
}
static void
maybe_warn_implicit_fallthrough (gimple_seq seq)
{
if (!warn_implicit_fallthrough)
return;
if (!(lang_GNU_C ()
|| lang_GNU_CXX ()
|| lang_GNU_OBJC ()))
return;
struct walk_stmt_info wi;
memset (&wi, 0, sizeof (wi));
walk_gimple_seq (seq, warn_implicit_fallthrough_r, NULL, &wi);
}
static tree
expand_FALLTHROUGH_r (gimple_stmt_iterator *gsi_p, bool *handled_ops_p,
struct walk_stmt_info *)
{
gimple *stmt = gsi_stmt (*gsi_p);
*handled_ops_p = true;
switch (gimple_code (stmt))
{
case GIMPLE_TRY:
case GIMPLE_BIND:
case GIMPLE_CATCH:
case GIMPLE_EH_FILTER:
case GIMPLE_TRANSACTION:
*handled_ops_p = false;
break;
case GIMPLE_CALL:
if (gimple_call_internal_p (stmt, IFN_FALLTHROUGH))
{
gsi_remove (gsi_p, true);
if (gsi_end_p (*gsi_p))
return integer_zero_node;
bool found = false;
location_t loc = gimple_location (stmt);
gimple_stmt_iterator gsi2 = *gsi_p;
stmt = gsi_stmt (gsi2);
if (gimple_code (stmt) == GIMPLE_GOTO && !gimple_has_location (stmt))
{
tree goto_dest = gimple_goto_dest (stmt);
for (; !gsi_end_p (gsi2); gsi_next (&gsi2))
{
if (gimple_code (gsi_stmt (gsi2)) == GIMPLE_LABEL
&& gimple_label_label (as_a <glabel *> (gsi_stmt (gsi2)))
== goto_dest)
break;
}
if (gsi_end_p (gsi2))
break;
gsi_next (&gsi2);
}
while (!gsi_end_p (gsi2))
{
stmt = gsi_stmt (gsi2);
if (gimple_code (stmt) == GIMPLE_LABEL)
{
tree label = gimple_label_label (as_a <glabel *> (stmt));
if (gimple_has_location (stmt) && DECL_ARTIFICIAL (label))
{
found = true;
break;
}
}
else if (gimple_call_internal_p (stmt, IFN_ASAN_MARK))
;
else if (!is_gimple_debug (stmt))
break;
gsi_next (&gsi2);
}
if (!found)
warning_at (loc, 0, "attribute %<fallthrough%> not preceding "
"a case label or default label");
}
break;
default:
break;
}
return NULL_TREE;
}
static void
expand_FALLTHROUGH (gimple_seq *seq_p)
{
struct walk_stmt_info wi;
memset (&wi, 0, sizeof (wi));
walk_gimple_seq_mod (seq_p, expand_FALLTHROUGH_r, NULL, &wi);
}

static enum gimplify_status
gimplify_switch_expr (tree *expr_p, gimple_seq *pre_p)
{
tree switch_expr = *expr_p;
gimple_seq switch_body_seq = NULL;
enum gimplify_status ret;
tree index_type = TREE_TYPE (switch_expr);
if (index_type == NULL_TREE)
index_type = TREE_TYPE (SWITCH_COND (switch_expr));
ret = gimplify_expr (&SWITCH_COND (switch_expr), pre_p, NULL, is_gimple_val,
fb_rvalue);
if (ret == GS_ERROR || ret == GS_UNHANDLED)
return ret;
if (SWITCH_BODY (switch_expr))
{
vec<tree> labels;
vec<tree> saved_labels;
hash_set<tree> *saved_live_switch_vars = NULL;
tree default_case = NULL_TREE;
gswitch *switch_stmt;
saved_labels = gimplify_ctxp->case_labels;
gimplify_ctxp->case_labels.create (8);
saved_live_switch_vars = gimplify_ctxp->live_switch_vars;
tree_code body_type = TREE_CODE (SWITCH_BODY (switch_expr));
if (body_type == BIND_EXPR || body_type == STATEMENT_LIST)
gimplify_ctxp->live_switch_vars = new hash_set<tree> (4);
else
gimplify_ctxp->live_switch_vars = NULL;
bool old_in_switch_expr = gimplify_ctxp->in_switch_expr;
gimplify_ctxp->in_switch_expr = true;
gimplify_stmt (&SWITCH_BODY (switch_expr), &switch_body_seq);
gimplify_ctxp->in_switch_expr = old_in_switch_expr;
maybe_warn_switch_unreachable (switch_body_seq);
maybe_warn_implicit_fallthrough (switch_body_seq);
if (!gimplify_ctxp->in_switch_expr)
expand_FALLTHROUGH (&switch_body_seq);
labels = gimplify_ctxp->case_labels;
gimplify_ctxp->case_labels = saved_labels;
if (gimplify_ctxp->live_switch_vars)
{
gcc_assert (gimplify_ctxp->live_switch_vars->elements () == 0);
delete gimplify_ctxp->live_switch_vars;
}
gimplify_ctxp->live_switch_vars = saved_live_switch_vars;
preprocess_case_label_vec_for_gimple (labels, index_type,
&default_case);
bool add_bind = false;
if (!default_case)
{
glabel *new_default;
default_case
= build_case_label (NULL_TREE, NULL_TREE,
create_artificial_label (UNKNOWN_LOCATION));
if (old_in_switch_expr)
{
SWITCH_BREAK_LABEL_P (CASE_LABEL (default_case)) = 1;
add_bind = true;
}
new_default = gimple_build_label (CASE_LABEL (default_case));
gimplify_seq_add_stmt (&switch_body_seq, new_default);
}
else if (old_in_switch_expr)
{
gimple *last = gimple_seq_last_stmt (switch_body_seq);
if (last && gimple_code (last) == GIMPLE_LABEL)
{
tree label = gimple_label_label (as_a <glabel *> (last));
if (SWITCH_BREAK_LABEL_P (label))
add_bind = true;
}
}
switch_stmt = gimple_build_switch (SWITCH_COND (switch_expr),
default_case, labels);
if (add_bind)
{
gimple_seq bind_body = NULL;
gimplify_seq_add_stmt (&bind_body, switch_stmt);
gimple_seq_add_seq (&bind_body, switch_body_seq);
gbind *bind = gimple_build_bind (NULL_TREE, bind_body, NULL_TREE);
gimple_set_location (bind, EXPR_LOCATION (switch_expr));
gimplify_seq_add_stmt (pre_p, bind);
}
else
{
gimplify_seq_add_stmt (pre_p, switch_stmt);
gimplify_seq_add_seq (pre_p, switch_body_seq);
}
labels.release ();
}
else
gcc_unreachable ();
return GS_ALL_DONE;
}
static enum gimplify_status
gimplify_label_expr (tree *expr_p, gimple_seq *pre_p)
{
gcc_assert (decl_function_context (LABEL_EXPR_LABEL (*expr_p))
== current_function_decl);
tree label = LABEL_EXPR_LABEL (*expr_p);
glabel *label_stmt = gimple_build_label (label);
gimple_set_location (label_stmt, EXPR_LOCATION (*expr_p));
gimplify_seq_add_stmt (pre_p, label_stmt);
if (lookup_attribute ("cold", DECL_ATTRIBUTES (label)))
gimple_seq_add_stmt (pre_p, gimple_build_predict (PRED_COLD_LABEL,
NOT_TAKEN));
else if (lookup_attribute ("hot", DECL_ATTRIBUTES (label)))
gimple_seq_add_stmt (pre_p, gimple_build_predict (PRED_HOT_LABEL,
TAKEN));
return GS_ALL_DONE;
}
static enum gimplify_status
gimplify_case_label_expr (tree *expr_p, gimple_seq *pre_p)
{
struct gimplify_ctx *ctxp;
glabel *label_stmt;
for (ctxp = gimplify_ctxp; ; ctxp = ctxp->prev_context)
if (ctxp->case_labels.exists ())
break;
label_stmt = gimple_build_label (CASE_LABEL (*expr_p));
gimple_set_location (label_stmt, EXPR_LOCATION (*expr_p));
ctxp->case_labels.safe_push (*expr_p);
gimplify_seq_add_stmt (pre_p, label_stmt);
return GS_ALL_DONE;
}
tree
build_and_jump (tree *label_p)
{
if (label_p == NULL)
return NULL_TREE;
if (*label_p == NULL_TREE)
{
tree label = create_artificial_label (UNKNOWN_LOCATION);
*label_p = label;
}
return build1 (GOTO_EXPR, void_type_node, *label_p);
}
static enum gimplify_status
gimplify_exit_expr (tree *expr_p)
{
tree cond = TREE_OPERAND (*expr_p, 0);
tree expr;
expr = build_and_jump (&gimplify_ctxp->exit_label);
expr = build3 (COND_EXPR, void_type_node, cond, expr, NULL_TREE);
*expr_p = expr;
return GS_OK;
}
static void
canonicalize_component_ref (tree *expr_p)
{
tree expr = *expr_p;
tree type;
gcc_assert (TREE_CODE (expr) == COMPONENT_REF);
if (INTEGRAL_TYPE_P (TREE_TYPE (expr)))
type = TREE_TYPE (get_unwidened (expr, NULL_TREE));
else
type = TREE_TYPE (TREE_OPERAND (expr, 1));
if (TREE_TYPE (expr) != type)
{
#ifdef ENABLE_TYPES_CHECKING
tree old_type = TREE_TYPE (expr);
#endif
int type_quals;
type_quals = TYPE_QUALS (type)
| TYPE_QUALS (TREE_TYPE (TREE_OPERAND (expr, 0)));
if (TYPE_QUALS (type) != type_quals)
type = build_qualified_type (TYPE_MAIN_VARIANT (type), type_quals);
TREE_TYPE (expr) = type;
#ifdef ENABLE_TYPES_CHECKING
gcc_assert (useless_type_conversion_p (old_type, type));
#endif
}
}
static void
canonicalize_addr_expr (tree *expr_p)
{
tree expr = *expr_p;
tree addr_expr = TREE_OPERAND (expr, 0);
tree datype, ddatype, pddatype;
if (!POINTER_TYPE_P (TREE_TYPE (expr))
|| TREE_CODE (addr_expr) != ADDR_EXPR)
return;
datype = TREE_TYPE (TREE_TYPE (addr_expr));
if (TREE_CODE (datype) != ARRAY_TYPE)
return;
ddatype = TREE_TYPE (datype);
pddatype = build_pointer_type (ddatype);
if (!useless_type_conversion_p (TYPE_MAIN_VARIANT (TREE_TYPE (expr)),
pddatype))
return;
if (!TYPE_SIZE_UNIT (ddatype)
|| TREE_CODE (TYPE_SIZE_UNIT (ddatype)) != INTEGER_CST
|| !TYPE_DOMAIN (datype) || !TYPE_MIN_VALUE (TYPE_DOMAIN (datype))
|| TREE_CODE (TYPE_MIN_VALUE (TYPE_DOMAIN (datype))) != INTEGER_CST)
return;
*expr_p = build4 (ARRAY_REF, ddatype, TREE_OPERAND (addr_expr, 0),
TYPE_MIN_VALUE (TYPE_DOMAIN (datype)),
NULL_TREE, NULL_TREE);
*expr_p = build1 (ADDR_EXPR, pddatype, *expr_p);
if (!useless_type_conversion_p (TREE_TYPE (expr), TREE_TYPE (*expr_p)))
*expr_p = fold_convert (TREE_TYPE (expr), *expr_p);
}
static enum gimplify_status
gimplify_conversion (tree *expr_p)
{
location_t loc = EXPR_LOCATION (*expr_p);
gcc_assert (CONVERT_EXPR_P (*expr_p));
STRIP_SIGN_NOPS (TREE_OPERAND (*expr_p, 0));
if (tree_ssa_useless_type_conversion (*expr_p))
*expr_p = TREE_OPERAND (*expr_p, 0);
if (CONVERT_EXPR_P (*expr_p))
{
tree sub = TREE_OPERAND (*expr_p, 0);
if (TREE_CODE (sub) == COMPONENT_REF)
canonicalize_component_ref (&TREE_OPERAND (*expr_p, 0));
else if (TREE_CODE (sub) == ADDR_EXPR)
canonicalize_addr_expr (expr_p);
}
if (CONVERT_EXPR_P (*expr_p) && !is_gimple_reg_type (TREE_TYPE (*expr_p)))
*expr_p = fold_build1_loc (loc, VIEW_CONVERT_EXPR, TREE_TYPE (*expr_p),
TREE_OPERAND (*expr_p, 0));
if (TREE_CODE (*expr_p) == CONVERT_EXPR)
TREE_SET_CODE (*expr_p, NOP_EXPR);
return GS_OK;
}
static hash_set<tree> *nonlocal_vlas;
static tree nonlocal_vla_vars;
static enum gimplify_status
gimplify_var_or_parm_decl (tree *expr_p)
{
tree decl = *expr_p;
if (VAR_P (decl)
&& !DECL_SEEN_IN_BIND_EXPR_P (decl)
&& !TREE_STATIC (decl) && !DECL_EXTERNAL (decl)
&& decl_function_context (decl) == current_function_decl)
{
gcc_assert (seen_error ());
return GS_ERROR;
}
if (gimplify_omp_ctxp && omp_notice_variable (gimplify_omp_ctxp, decl, true))
return GS_ALL_DONE;
if (DECL_HAS_VALUE_EXPR_P (decl))
{
tree value_expr = DECL_VALUE_EXPR (decl);
if (VAR_P (decl)
&& TREE_CODE (DECL_SIZE_UNIT (decl)) != INTEGER_CST
&& nonlocal_vlas != NULL
&& TREE_CODE (value_expr) == INDIRECT_REF
&& TREE_CODE (TREE_OPERAND (value_expr, 0)) == VAR_DECL
&& decl_function_context (decl) != current_function_decl)
{
struct gimplify_omp_ctx *ctx = gimplify_omp_ctxp;
while (ctx
&& (ctx->region_type == ORT_WORKSHARE
|| ctx->region_type == ORT_SIMD
|| ctx->region_type == ORT_ACC))
ctx = ctx->outer_context;
if (!ctx && !nonlocal_vlas->add (decl))
{
tree copy = copy_node (decl);
lang_hooks.dup_lang_specific_decl (copy);
SET_DECL_RTL (copy, 0);
TREE_USED (copy) = 1;
DECL_CHAIN (copy) = nonlocal_vla_vars;
nonlocal_vla_vars = copy;
SET_DECL_VALUE_EXPR (copy, unshare_expr (value_expr));
DECL_HAS_VALUE_EXPR_P (copy) = 1;
}
}
*expr_p = unshare_expr (value_expr);
return GS_OK;
}
return GS_ALL_DONE;
}
static void
recalculate_side_effects (tree t)
{
enum tree_code code = TREE_CODE (t);
int len = TREE_OPERAND_LENGTH (t);
int i;
switch (TREE_CODE_CLASS (code))
{
case tcc_expression:
switch (code)
{
case INIT_EXPR:
case MODIFY_EXPR:
case VA_ARG_EXPR:
case PREDECREMENT_EXPR:
case PREINCREMENT_EXPR:
case POSTDECREMENT_EXPR:
case POSTINCREMENT_EXPR:
return;
default:
break;
}
case tcc_comparison:  
case tcc_unary:       
case tcc_binary:      
case tcc_reference:   
case tcc_vl_exp:        
TREE_SIDE_EFFECTS (t) = TREE_THIS_VOLATILE (t);
for (i = 0; i < len; ++i)
{
tree op = TREE_OPERAND (t, i);
if (op && TREE_SIDE_EFFECTS (op))
TREE_SIDE_EFFECTS (t) = 1;
}
break;
case tcc_constant:
return;
default:
gcc_unreachable ();
}
}
static enum gimplify_status
gimplify_compound_lval (tree *expr_p, gimple_seq *pre_p, gimple_seq *post_p,
fallback_t fallback)
{
tree *p;
enum gimplify_status ret = GS_ALL_DONE, tret;
int i;
location_t loc = EXPR_LOCATION (*expr_p);
tree expr = *expr_p;
auto_vec<tree, 10> expr_stack;
for (p = expr_p; ; p = &TREE_OPERAND (*p, 0))
{
restart:
if (TREE_CODE (*p) == INDIRECT_REF)
*p = fold_indirect_ref_loc (loc, *p);
if (handled_component_p (*p))
;
else if ((VAR_P (*p) || TREE_CODE (*p) == PARM_DECL)
&& gimplify_var_or_parm_decl (p) == GS_OK)
goto restart;
else
break;
expr_stack.safe_push (*p);
}
gcc_assert (expr_stack.length ());
for (i = expr_stack.length () - 1; i >= 0; i--)
{
tree t = expr_stack[i];
if (TREE_CODE (t) == ARRAY_REF || TREE_CODE (t) == ARRAY_RANGE_REF)
{
if (TREE_OPERAND (t, 2) == NULL_TREE)
{
tree low = unshare_expr (array_ref_low_bound (t));
if (!is_gimple_min_invariant (low))
{
TREE_OPERAND (t, 2) = low;
tret = gimplify_expr (&TREE_OPERAND (t, 2), pre_p,
post_p, is_gimple_reg,
fb_rvalue);
ret = MIN (ret, tret);
}
}
else
{
tret = gimplify_expr (&TREE_OPERAND (t, 2), pre_p, post_p,
is_gimple_reg, fb_rvalue);
ret = MIN (ret, tret);
}
if (TREE_OPERAND (t, 3) == NULL_TREE)
{
tree elmt_type = TREE_TYPE (TREE_TYPE (TREE_OPERAND (t, 0)));
tree elmt_size = unshare_expr (array_ref_element_size (t));
tree factor = size_int (TYPE_ALIGN_UNIT (elmt_type));
elmt_size
= size_binop_loc (loc, EXACT_DIV_EXPR, elmt_size, factor);
if (!is_gimple_min_invariant (elmt_size))
{
TREE_OPERAND (t, 3) = elmt_size;
tret = gimplify_expr (&TREE_OPERAND (t, 3), pre_p,
post_p, is_gimple_reg,
fb_rvalue);
ret = MIN (ret, tret);
}
}
else
{
tret = gimplify_expr (&TREE_OPERAND (t, 3), pre_p, post_p,
is_gimple_reg, fb_rvalue);
ret = MIN (ret, tret);
}
}
else if (TREE_CODE (t) == COMPONENT_REF)
{
if (TREE_OPERAND (t, 2) == NULL_TREE)
{
tree offset = unshare_expr (component_ref_field_offset (t));
tree field = TREE_OPERAND (t, 1);
tree factor
= size_int (DECL_OFFSET_ALIGN (field) / BITS_PER_UNIT);
offset = size_binop_loc (loc, EXACT_DIV_EXPR, offset, factor);
if (!is_gimple_min_invariant (offset))
{
TREE_OPERAND (t, 2) = offset;
tret = gimplify_expr (&TREE_OPERAND (t, 2), pre_p,
post_p, is_gimple_reg,
fb_rvalue);
ret = MIN (ret, tret);
}
}
else
{
tret = gimplify_expr (&TREE_OPERAND (t, 2), pre_p, post_p,
is_gimple_reg, fb_rvalue);
ret = MIN (ret, tret);
}
}
}
tret = gimplify_expr (p, pre_p, post_p, is_gimple_min_lval,
fallback | fb_lvalue);
ret = MIN (ret, tret);
for (; expr_stack.length () > 0; )
{
tree t = expr_stack.pop ();
if (TREE_CODE (t) == ARRAY_REF || TREE_CODE (t) == ARRAY_RANGE_REF)
{
if (!is_gimple_min_invariant (TREE_OPERAND (t, 1)))
{
tret = gimplify_expr (&TREE_OPERAND (t, 1), pre_p, post_p,
is_gimple_val, fb_rvalue);
ret = MIN (ret, tret);
}
}
STRIP_USELESS_TYPE_CONVERSION (TREE_OPERAND (t, 0));
recalculate_side_effects (t);
}
if ((fallback & fb_rvalue) && TREE_CODE (*expr_p) == COMPONENT_REF)
{
canonicalize_component_ref (expr_p);
}
expr_stack.release ();
gcc_assert (*expr_p == expr || ret != GS_ALL_DONE);
return ret;
}
enum gimplify_status
gimplify_self_mod_expr (tree *expr_p, gimple_seq *pre_p, gimple_seq *post_p,
bool want_value, tree arith_type)
{
enum tree_code code;
tree lhs, lvalue, rhs, t1;
gimple_seq post = NULL, *orig_post_p = post_p;
bool postfix;
enum tree_code arith_code;
enum gimplify_status ret;
location_t loc = EXPR_LOCATION (*expr_p);
code = TREE_CODE (*expr_p);
gcc_assert (code == POSTINCREMENT_EXPR || code == POSTDECREMENT_EXPR
|| code == PREINCREMENT_EXPR || code == PREDECREMENT_EXPR);
if (code == POSTINCREMENT_EXPR || code == POSTDECREMENT_EXPR)
postfix = want_value;
else
postfix = false;
if (postfix)
post_p = &post;
if (code == PREINCREMENT_EXPR || code == POSTINCREMENT_EXPR)
arith_code = PLUS_EXPR;
else
arith_code = MINUS_EXPR;
lvalue = TREE_OPERAND (*expr_p, 0);
ret = gimplify_expr (&lvalue, pre_p, post_p, is_gimple_lvalue, fb_lvalue);
if (ret == GS_ERROR)
return ret;
lhs = lvalue;
rhs = TREE_OPERAND (*expr_p, 1);
if (postfix)
{
ret = gimplify_expr (&lhs, pre_p, post_p, is_gimple_val, fb_rvalue);
if (ret == GS_ERROR)
return ret;
lhs = get_initialized_tmp_var (lhs, pre_p, NULL);
}
if (POINTER_TYPE_P (TREE_TYPE (lhs)))
{
rhs = convert_to_ptrofftype_loc (loc, rhs);
if (arith_code == MINUS_EXPR)
rhs = fold_build1_loc (loc, NEGATE_EXPR, TREE_TYPE (rhs), rhs);
t1 = fold_build2 (POINTER_PLUS_EXPR, TREE_TYPE (*expr_p), lhs, rhs);
}
else
t1 = fold_convert (TREE_TYPE (*expr_p),
fold_build2 (arith_code, arith_type,
fold_convert (arith_type, lhs),
fold_convert (arith_type, rhs)));
if (postfix)
{
gimplify_assign (lvalue, t1, pre_p);
gimplify_seq_add_seq (orig_post_p, post);
*expr_p = lhs;
return GS_ALL_DONE;
}
else
{
*expr_p = build2 (MODIFY_EXPR, TREE_TYPE (lvalue), lvalue, t1);
return GS_OK;
}
}
static void
maybe_with_size_expr (tree *expr_p)
{
tree expr = *expr_p;
tree type = TREE_TYPE (expr);
tree size;
if (TREE_CODE (expr) == WITH_SIZE_EXPR
|| type == error_mark_node)
return;
size = TYPE_SIZE_UNIT (type);
if (!size || poly_int_tree_p (size))
return;
size = unshare_expr (size);
size = SUBSTITUTE_PLACEHOLDER_IN_EXPR (size, expr);
*expr_p = build2 (WITH_SIZE_EXPR, type, expr, size);
}
enum gimplify_status
gimplify_arg (tree *arg_p, gimple_seq *pre_p, location_t call_location,
bool allow_ssa)
{
bool (*test) (tree);
fallback_t fb;
if (is_gimple_reg_type (TREE_TYPE (*arg_p)))
test = is_gimple_val, fb = fb_rvalue;
else
{
test = is_gimple_lvalue, fb = fb_either;
if (TREE_CODE (*arg_p) == TARGET_EXPR)
{
tree init = TARGET_EXPR_INITIAL (*arg_p);
if (init
&& !VOID_TYPE_P (TREE_TYPE (init)))
*arg_p = init;
}
}
maybe_with_size_expr (arg_p);
protected_set_expr_location (*arg_p, call_location);
return gimplify_expr (arg_p, pre_p, NULL, test, fb, allow_ssa);
}
static bool
maybe_fold_stmt (gimple_stmt_iterator *gsi)
{
struct gimplify_omp_ctx *ctx;
for (ctx = gimplify_omp_ctxp; ctx; ctx = ctx->outer_context)
if ((ctx->region_type & (ORT_TARGET | ORT_PARALLEL | ORT_TASK)) != 0)
return false;
return fold_stmt (gsi);
}
static enum gimplify_status
gimplify_call_expr (tree *expr_p, gimple_seq *pre_p, bool want_value)
{
tree fndecl, parms, p, fnptrtype;
enum gimplify_status ret;
int i, nargs;
gcall *call;
bool builtin_va_start_p = false;
location_t loc = EXPR_LOCATION (*expr_p);
gcc_assert (TREE_CODE (*expr_p) == CALL_EXPR);
if (! EXPR_HAS_LOCATION (*expr_p))
SET_EXPR_LOCATION (*expr_p, input_location);
if (CALL_EXPR_FN (*expr_p) == NULL_TREE)
{
if (want_value)
return GS_ALL_DONE;
nargs = call_expr_nargs (*expr_p);
enum internal_fn ifn = CALL_EXPR_IFN (*expr_p);
auto_vec<tree> vargs (nargs);
for (i = 0; i < nargs; i++)
{
gimplify_arg (&CALL_EXPR_ARG (*expr_p, i), pre_p,
EXPR_LOCATION (*expr_p));
vargs.quick_push (CALL_EXPR_ARG (*expr_p, i));
}
gcall *call = gimple_build_call_internal_vec (ifn, vargs);
gimple_call_set_nothrow (call, TREE_NOTHROW (*expr_p));
gimplify_seq_add_stmt (pre_p, call);
return GS_ALL_DONE;
}
fndecl = get_callee_fndecl (*expr_p);
if (fndecl
&& DECL_BUILT_IN_CLASS (fndecl) == BUILT_IN_NORMAL)
switch (DECL_FUNCTION_CODE (fndecl))
{
CASE_BUILT_IN_ALLOCA:
if (CALL_ALLOCA_FOR_VAR_P (*expr_p))
gimplify_ctxp->save_stack = true;
else
gimplify_ctxp->keep_stack = true;
break;
case BUILT_IN_VA_START:
{
builtin_va_start_p = TRUE;
if (call_expr_nargs (*expr_p) < 2)
{
error ("too few arguments to function %<va_start%>");
*expr_p = build_empty_stmt (EXPR_LOCATION (*expr_p));
return GS_OK;
}
if (fold_builtin_next_arg (*expr_p, true))
{
*expr_p = build_empty_stmt (EXPR_LOCATION (*expr_p));
return GS_OK;
}
break;
}
default:
;
}
if (fndecl && DECL_BUILT_IN (fndecl))
{
tree new_tree = fold_call_expr (input_location, *expr_p, !want_value);
if (new_tree && new_tree != *expr_p)
{
*expr_p = new_tree;
return GS_OK;
}
}
fnptrtype = TREE_TYPE (CALL_EXPR_FN (*expr_p));
ret = gimplify_expr (&CALL_EXPR_FN (*expr_p), pre_p, NULL,
is_gimple_call_addr, fb_rvalue);
nargs = call_expr_nargs (*expr_p);
fndecl = get_callee_fndecl (*expr_p);
parms = NULL_TREE;
if (fndecl)
parms = TYPE_ARG_TYPES (TREE_TYPE (fndecl));
else
parms = TYPE_ARG_TYPES (TREE_TYPE (fnptrtype));
if (fndecl && DECL_ARGUMENTS (fndecl))
p = DECL_ARGUMENTS (fndecl);
else if (parms)
p = parms;
else
p = NULL_TREE;
for (i = 0; i < nargs && p; i++, p = TREE_CHAIN (p))
;
if (!p
&& i < nargs
&& TREE_CODE (CALL_EXPR_ARG (*expr_p, nargs - 1)) == CALL_EXPR)
{
tree last_arg = CALL_EXPR_ARG (*expr_p, nargs - 1);
tree last_arg_fndecl = get_callee_fndecl (last_arg);
if (last_arg_fndecl
&& TREE_CODE (last_arg_fndecl) == FUNCTION_DECL
&& DECL_BUILT_IN_CLASS (last_arg_fndecl) == BUILT_IN_NORMAL
&& DECL_FUNCTION_CODE (last_arg_fndecl) == BUILT_IN_VA_ARG_PACK)
{
tree call = *expr_p;
--nargs;
*expr_p = build_call_array_loc (loc, TREE_TYPE (call),
CALL_EXPR_FN (call),
nargs, CALL_EXPR_ARGP (call));
CALL_EXPR_STATIC_CHAIN (*expr_p) = CALL_EXPR_STATIC_CHAIN (call);
CALL_EXPR_TAILCALL (*expr_p) = CALL_EXPR_TAILCALL (call);
CALL_EXPR_RETURN_SLOT_OPT (*expr_p)
= CALL_EXPR_RETURN_SLOT_OPT (call);
CALL_FROM_THUNK_P (*expr_p) = CALL_FROM_THUNK_P (call);
SET_EXPR_LOCATION (*expr_p, EXPR_LOCATION (call));
CALL_EXPR_VA_ARG_PACK (*expr_p) = 1;
}
}
bool returns_twice = call_expr_flags (*expr_p) & ECF_RETURNS_TWICE;
if (nargs > 0)
{
for (i = (PUSH_ARGS_REVERSED ? nargs - 1 : 0);
PUSH_ARGS_REVERSED ? i >= 0 : i < nargs;
PUSH_ARGS_REVERSED ? i-- : i++)
{
enum gimplify_status t;
if ((i != 1) || !builtin_va_start_p)
{
t = gimplify_arg (&CALL_EXPR_ARG (*expr_p, i), pre_p,
EXPR_LOCATION (*expr_p), ! returns_twice);
if (t == GS_ERROR)
ret = GS_ERROR;
}
}
}
if (CALL_EXPR_STATIC_CHAIN (*expr_p))
{
if (fndecl && !DECL_STATIC_CHAIN (fndecl))
CALL_EXPR_STATIC_CHAIN (*expr_p) = NULL;
else
{
enum gimplify_status t;
t = gimplify_arg (&CALL_EXPR_STATIC_CHAIN (*expr_p), pre_p,
EXPR_LOCATION (*expr_p), ! returns_twice);
if (t == GS_ERROR)
ret = GS_ERROR;
}
}
if (want_value && fndecl
&& VOID_TYPE_P (TREE_TYPE (TREE_TYPE (fnptrtype))))
{
error_at (loc, "using result of function returning %<void%>");
ret = GS_ERROR;
}
if (ret != GS_ERROR)
{
tree new_tree = fold_call_expr (input_location, *expr_p, !want_value);
if (new_tree && new_tree != *expr_p)
{
*expr_p = new_tree;
return GS_OK;
}
}
else
{
*expr_p = error_mark_node;
return GS_ERROR;
}
if (TREE_CODE (*expr_p) == CALL_EXPR)
{
int flags = call_expr_flags (*expr_p);
if (flags & (ECF_CONST | ECF_PURE)
&& !(flags & (ECF_LOOPING_CONST_OR_PURE)))
TREE_SIDE_EFFECTS (*expr_p) = 0;
}
if (!want_value)
{
gimple_stmt_iterator gsi;
call = gimple_build_call_from_tree (*expr_p, fnptrtype);
notice_special_calls (call);
gimplify_seq_add_stmt (pre_p, call);
gsi = gsi_last (*pre_p);
maybe_fold_stmt (&gsi);
*expr_p = NULL_TREE;
}
else
CALL_EXPR_FN (*expr_p) = build1 (NOP_EXPR, fnptrtype,
CALL_EXPR_FN (*expr_p));
return ret;
}
static tree
shortcut_cond_r (tree pred, tree *true_label_p, tree *false_label_p,
location_t locus)
{
tree local_label = NULL_TREE;
tree t, expr = NULL;
if (TREE_CODE (pred) == TRUTH_ANDIF_EXPR)
{
location_t new_locus;
if (false_label_p == NULL)
false_label_p = &local_label;
t = shortcut_cond_r (TREE_OPERAND (pred, 0), NULL, false_label_p, locus);
append_to_statement_list (t, &expr);
new_locus = rexpr_location (pred, locus);
t = shortcut_cond_r (TREE_OPERAND (pred, 1), true_label_p, false_label_p,
new_locus);
append_to_statement_list (t, &expr);
}
else if (TREE_CODE (pred) == TRUTH_ORIF_EXPR)
{
location_t new_locus;
if (true_label_p == NULL)
true_label_p = &local_label;
t = shortcut_cond_r (TREE_OPERAND (pred, 0), true_label_p, NULL, locus);
append_to_statement_list (t, &expr);
new_locus = rexpr_location (pred, locus);
t = shortcut_cond_r (TREE_OPERAND (pred, 1), true_label_p, false_label_p,
new_locus);
append_to_statement_list (t, &expr);
}
else if (TREE_CODE (pred) == COND_EXPR
&& !VOID_TYPE_P (TREE_TYPE (TREE_OPERAND (pred, 1)))
&& !VOID_TYPE_P (TREE_TYPE (TREE_OPERAND (pred, 2))))
{
location_t new_locus;
new_locus = rexpr_location (pred, locus);
expr = build3 (COND_EXPR, void_type_node, TREE_OPERAND (pred, 0),
shortcut_cond_r (TREE_OPERAND (pred, 1), true_label_p,
false_label_p, locus),
shortcut_cond_r (TREE_OPERAND (pred, 2), true_label_p,
false_label_p, new_locus));
}
else
{
expr = build3 (COND_EXPR, void_type_node, pred,
build_and_jump (true_label_p),
build_and_jump (false_label_p));
SET_EXPR_LOCATION (expr, locus);
}
if (local_label)
{
t = build1 (LABEL_EXPR, void_type_node, local_label);
append_to_statement_list (t, &expr);
}
return expr;
}
static tree
find_goto (tree expr)
{
if (!expr)
return NULL_TREE;
if (TREE_CODE (expr) == GOTO_EXPR)
return expr;
if (TREE_CODE (expr) != STATEMENT_LIST)
return NULL_TREE;
tree_stmt_iterator i = tsi_start (expr);
while (!tsi_end_p (i) && TREE_CODE (tsi_stmt (i)) == DEBUG_BEGIN_STMT)
tsi_next (&i);
if (!tsi_one_before_end_p (i))
return NULL_TREE;
return find_goto (tsi_stmt (i));
}
static inline tree
find_goto_label (tree expr)
{
tree dest = find_goto (expr);
if (dest && TREE_CODE (GOTO_DESTINATION (dest)) == LABEL_DECL)
return dest;
return NULL_TREE;
}
static tree
shortcut_cond_expr (tree expr)
{
tree pred = TREE_OPERAND (expr, 0);
tree then_ = TREE_OPERAND (expr, 1);
tree else_ = TREE_OPERAND (expr, 2);
tree true_label, false_label, end_label, t;
tree *true_label_p;
tree *false_label_p;
bool emit_end, emit_false, jump_over_else;
bool then_se = then_ && TREE_SIDE_EFFECTS (then_);
bool else_se = else_ && TREE_SIDE_EFFECTS (else_);
if (!else_se)
{
while (TREE_CODE (pred) == TRUTH_ANDIF_EXPR)
{
location_t locus = EXPR_LOC_OR_LOC (expr, input_location);
TREE_OPERAND (expr, 0) = TREE_OPERAND (pred, 1);
if (rexpr_has_location (pred))
SET_EXPR_LOCATION (expr, rexpr_location (pred));
then_ = shortcut_cond_expr (expr);
then_se = then_ && TREE_SIDE_EFFECTS (then_);
pred = TREE_OPERAND (pred, 0);
expr = build3 (COND_EXPR, void_type_node, pred, then_, NULL_TREE);
SET_EXPR_LOCATION (expr, locus);
}
}
if (!then_se)
{
while (TREE_CODE (pred) == TRUTH_ORIF_EXPR)
{
location_t locus = EXPR_LOC_OR_LOC (expr, input_location);
TREE_OPERAND (expr, 0) = TREE_OPERAND (pred, 1);
if (rexpr_has_location (pred))
SET_EXPR_LOCATION (expr, rexpr_location (pred));
else_ = shortcut_cond_expr (expr);
else_se = else_ && TREE_SIDE_EFFECTS (else_);
pred = TREE_OPERAND (pred, 0);
expr = build3 (COND_EXPR, void_type_node, pred, NULL_TREE, else_);
SET_EXPR_LOCATION (expr, locus);
}
}
if (TREE_CODE (pred) != TRUTH_ANDIF_EXPR
&& TREE_CODE (pred) != TRUTH_ORIF_EXPR)
return expr;
true_label = false_label = end_label = NULL_TREE;
if (tree then_goto = find_goto_label (then_))
{
true_label = GOTO_DESTINATION (then_goto);
then_ = NULL;
then_se = false;
}
if (tree else_goto = find_goto_label (else_))
{
false_label = GOTO_DESTINATION (else_goto);
else_ = NULL;
else_se = false;
}
if (true_label)
true_label_p = &true_label;
else
true_label_p = NULL;
if (false_label || else_se)
false_label_p = &false_label;
else
false_label_p = NULL;
if (!then_se && !else_se)
return shortcut_cond_r (pred, true_label_p, false_label_p,
EXPR_LOC_OR_LOC (expr, input_location));
if (else_se)
t = expr_last (else_);
else if (then_se)
t = expr_last (then_);
else
t = NULL;
if (t && TREE_CODE (t) == LABEL_EXPR)
end_label = LABEL_EXPR_LABEL (t);
if (!false_label_p)
false_label_p = &end_label;
emit_end = (end_label == NULL_TREE);
emit_false = (false_label == NULL_TREE);
jump_over_else = block_may_fallthru (then_);
pred = shortcut_cond_r (pred, true_label_p, false_label_p,
EXPR_LOC_OR_LOC (expr, input_location));
expr = NULL;
append_to_statement_list (pred, &expr);
append_to_statement_list (then_, &expr);
if (else_se)
{
if (jump_over_else)
{
tree last = expr_last (expr);
t = build_and_jump (&end_label);
if (rexpr_has_location (last))
SET_EXPR_LOCATION (t, rexpr_location (last));
append_to_statement_list (t, &expr);
}
if (emit_false)
{
t = build1 (LABEL_EXPR, void_type_node, false_label);
append_to_statement_list (t, &expr);
}
append_to_statement_list (else_, &expr);
}
if (emit_end && end_label)
{
t = build1 (LABEL_EXPR, void_type_node, end_label);
append_to_statement_list (t, &expr);
}
return expr;
}
tree
gimple_boolify (tree expr)
{
tree type = TREE_TYPE (expr);
location_t loc = EXPR_LOCATION (expr);
if (TREE_CODE (expr) == NE_EXPR
&& TREE_CODE (TREE_OPERAND (expr, 0)) == CALL_EXPR
&& integer_zerop (TREE_OPERAND (expr, 1)))
{
tree call = TREE_OPERAND (expr, 0);
tree fn = get_callee_fndecl (call);
if (fn
&& DECL_BUILT_IN_CLASS (fn) == BUILT_IN_NORMAL
&& DECL_FUNCTION_CODE (fn) == BUILT_IN_EXPECT
&& call_expr_nargs (call) == 2)
{
tree arg = CALL_EXPR_ARG (call, 0);
if (arg)
{
if (TREE_CODE (arg) == NOP_EXPR
&& TREE_TYPE (arg) == TREE_TYPE (call))
arg = TREE_OPERAND (arg, 0);
if (truth_value_p (TREE_CODE (arg)))
{
arg = gimple_boolify (arg);
CALL_EXPR_ARG (call, 0)
= fold_convert_loc (loc, TREE_TYPE (call), arg);
}
}
}
}
switch (TREE_CODE (expr))
{
case TRUTH_AND_EXPR:
case TRUTH_OR_EXPR:
case TRUTH_XOR_EXPR:
case TRUTH_ANDIF_EXPR:
case TRUTH_ORIF_EXPR:
TREE_OPERAND (expr, 1) = gimple_boolify (TREE_OPERAND (expr, 1));
case TRUTH_NOT_EXPR:
TREE_OPERAND (expr, 0) = gimple_boolify (TREE_OPERAND (expr, 0));
if (TREE_CODE (type) != BOOLEAN_TYPE)
TREE_TYPE (expr) = boolean_type_node;
return expr;
case ANNOTATE_EXPR:
switch ((enum annot_expr_kind) TREE_INT_CST_LOW (TREE_OPERAND (expr, 1)))
{
case annot_expr_ivdep_kind:
case annot_expr_unroll_kind:
case annot_expr_no_vector_kind:
case annot_expr_vector_kind:
case annot_expr_parallel_kind:
TREE_OPERAND (expr, 0) = gimple_boolify (TREE_OPERAND (expr, 0));
if (TREE_CODE (type) != BOOLEAN_TYPE)
TREE_TYPE (expr) = boolean_type_node;
return expr;
default:
gcc_unreachable ();
}
default:
if (COMPARISON_CLASS_P (expr))
{
if (TREE_CODE (type) != BOOLEAN_TYPE)
TREE_TYPE (expr) = boolean_type_node;
return expr;
}
if (TREE_CODE (type) == BOOLEAN_TYPE)
return expr;
return fold_convert_loc (loc, boolean_type_node, expr);
}
}
static enum gimplify_status
gimplify_pure_cond_expr (tree *expr_p, gimple_seq *pre_p)
{
tree expr = *expr_p, cond;
enum gimplify_status ret, tret;
enum tree_code code;
cond = gimple_boolify (COND_EXPR_COND (expr));
code = TREE_CODE (cond);
if (code == TRUTH_ANDIF_EXPR)
TREE_SET_CODE (cond, TRUTH_AND_EXPR);
else if (code == TRUTH_ORIF_EXPR)
TREE_SET_CODE (cond, TRUTH_OR_EXPR);
ret = gimplify_expr (&cond, pre_p, NULL, is_gimple_condexpr, fb_rvalue);
COND_EXPR_COND (*expr_p) = cond;
tret = gimplify_expr (&COND_EXPR_THEN (expr), pre_p, NULL,
is_gimple_val, fb_rvalue);
ret = MIN (ret, tret);
tret = gimplify_expr (&COND_EXPR_ELSE (expr), pre_p, NULL,
is_gimple_val, fb_rvalue);
return MIN (ret, tret);
}
static bool
generic_expr_could_trap_p (tree expr)
{
unsigned i, n;
if (!expr || is_gimple_val (expr))
return false;
if (!EXPR_P (expr) || tree_could_trap_p (expr))
return true;
n = TREE_OPERAND_LENGTH (expr);
for (i = 0; i < n; i++)
if (generic_expr_could_trap_p (TREE_OPERAND (expr, i)))
return true;
return false;
}
static enum gimplify_status
gimplify_cond_expr (tree *expr_p, gimple_seq *pre_p, fallback_t fallback)
{
tree expr = *expr_p;
tree type = TREE_TYPE (expr);
location_t loc = EXPR_LOCATION (expr);
tree tmp, arm1, arm2;
enum gimplify_status ret;
tree label_true, label_false, label_cont;
bool have_then_clause_p, have_else_clause_p;
gcond *cond_stmt;
enum tree_code pred_code;
gimple_seq seq = NULL;
if (!VOID_TYPE_P (type))
{
tree then_ = TREE_OPERAND (expr, 1), else_ = TREE_OPERAND (expr, 2);
tree result;
if (((fallback & fb_rvalue) || !(fallback & fb_lvalue))
&& !TREE_ADDRESSABLE (type))
{
if (gimplify_ctxp->allow_rhs_cond_expr
&& !TREE_SIDE_EFFECTS (then_)
&& !generic_expr_could_trap_p (then_)
&& !TREE_SIDE_EFFECTS (else_)
&& !generic_expr_could_trap_p (else_))
return gimplify_pure_cond_expr (expr_p, pre_p);
tmp = create_tmp_var (type, "iftmp");
result = tmp;
}
else
{
type = build_pointer_type (type);
if (!VOID_TYPE_P (TREE_TYPE (then_)))
then_ = build_fold_addr_expr_loc (loc, then_);
if (!VOID_TYPE_P (TREE_TYPE (else_)))
else_ = build_fold_addr_expr_loc (loc, else_);
expr
= build3 (COND_EXPR, type, TREE_OPERAND (expr, 0), then_, else_);
tmp = create_tmp_var (type, "iftmp");
result = build_simple_mem_ref_loc (loc, tmp);
}
if (!VOID_TYPE_P (TREE_TYPE (then_)))
TREE_OPERAND (expr, 1) = build2 (MODIFY_EXPR, type, tmp, then_);
if (!VOID_TYPE_P (TREE_TYPE (else_)))
TREE_OPERAND (expr, 2) = build2 (MODIFY_EXPR, type, tmp, else_);
TREE_TYPE (expr) = void_type_node;
recalculate_side_effects (expr);
gimplify_stmt (&expr, pre_p);
*expr_p = result;
return GS_ALL_DONE;
}
STRIP_TYPE_NOPS (TREE_OPERAND (expr, 0));
if (TREE_CODE (TREE_OPERAND (expr, 0)) == COMPOUND_EXPR)
gimplify_compound_expr (&TREE_OPERAND (expr, 0), pre_p, true);
TREE_OPERAND (expr, 0) = gimple_boolify (TREE_OPERAND (expr, 0));
if (TREE_CODE (TREE_OPERAND (expr, 0)) == TRUTH_ANDIF_EXPR
|| TREE_CODE (TREE_OPERAND (expr, 0)) == TRUTH_ORIF_EXPR)
{
expr = shortcut_cond_expr (expr);
if (expr != *expr_p)
{
*expr_p = expr;
gimple_push_condition ();
gimplify_stmt (expr_p, &seq);
gimple_pop_condition (pre_p);
gimple_seq_add_seq (pre_p, seq);
return GS_ALL_DONE;
}
}
ret = gimplify_expr (&TREE_OPERAND (expr, 0), pre_p, NULL, is_gimple_condexpr,
fb_rvalue);
if (ret == GS_ERROR)
return GS_ERROR;
gcc_assert (TREE_OPERAND (expr, 0) != NULL_TREE);
gimple_push_condition ();
have_then_clause_p = have_else_clause_p = false;
label_true = find_goto_label (TREE_OPERAND (expr, 1));
if (label_true
&& DECL_CONTEXT (GOTO_DESTINATION (label_true)) == current_function_decl
&& (optimize
|| !EXPR_HAS_LOCATION (expr)
|| !rexpr_has_location (label_true)
|| EXPR_LOCATION (expr) == rexpr_location (label_true)))
{
have_then_clause_p = true;
label_true = GOTO_DESTINATION (label_true);
}
else
label_true = create_artificial_label (UNKNOWN_LOCATION);
label_false = find_goto_label (TREE_OPERAND (expr, 2));
if (label_false
&& DECL_CONTEXT (GOTO_DESTINATION (label_false)) == current_function_decl
&& (optimize
|| !EXPR_HAS_LOCATION (expr)
|| !rexpr_has_location (label_false)
|| EXPR_LOCATION (expr) == rexpr_location (label_false)))
{
have_else_clause_p = true;
label_false = GOTO_DESTINATION (label_false);
}
else
label_false = create_artificial_label (UNKNOWN_LOCATION);
gimple_cond_get_ops_from_tree (COND_EXPR_COND (expr), &pred_code, &arm1,
&arm2);
cond_stmt = gimple_build_cond (pred_code, arm1, arm2, label_true,
label_false);
gimple_set_no_warning (cond_stmt, TREE_NO_WARNING (COND_EXPR_COND (expr)));
gimplify_seq_add_stmt (&seq, cond_stmt);
gimple_stmt_iterator gsi = gsi_last (seq);
maybe_fold_stmt (&gsi);
label_cont = NULL_TREE;
if (!have_then_clause_p)
{
if (TREE_OPERAND (expr, 1) == NULL_TREE
&& !have_else_clause_p
&& TREE_OPERAND (expr, 2) != NULL_TREE)
label_cont = label_true;
else
{
gimplify_seq_add_stmt (&seq, gimple_build_label (label_true));
have_then_clause_p = gimplify_stmt (&TREE_OPERAND (expr, 1), &seq);
if (!have_else_clause_p
&& TREE_OPERAND (expr, 2) != NULL_TREE
&& gimple_seq_may_fallthru (seq))
{
gimple *g;
label_cont = create_artificial_label (UNKNOWN_LOCATION);
g = gimple_build_goto (label_cont);
gimple_set_do_not_emit_location (g);
gimplify_seq_add_stmt (&seq, g);
}
}
}
if (!have_else_clause_p)
{
gimplify_seq_add_stmt (&seq, gimple_build_label (label_false));
have_else_clause_p = gimplify_stmt (&TREE_OPERAND (expr, 2), &seq);
}
if (label_cont)
gimplify_seq_add_stmt (&seq, gimple_build_label (label_cont));
gimple_pop_condition (pre_p);
gimple_seq_add_seq (pre_p, seq);
if (ret == GS_ERROR)
; 
else if (have_then_clause_p || have_else_clause_p)
ret = GS_ALL_DONE;
else
{
expr = TREE_OPERAND (expr, 0);
gimplify_stmt (&expr, pre_p);
}
*expr_p = NULL;
return ret;
}
static void
prepare_gimple_addressable (tree *expr_p, gimple_seq *seq_p)
{
while (handled_component_p (*expr_p))
expr_p = &TREE_OPERAND (*expr_p, 0);
if (is_gimple_reg (*expr_p))
{
tree var = get_initialized_tmp_var (*expr_p, seq_p, NULL, false);
DECL_GIMPLE_REG_P (var) = 0;
*expr_p = var;
}
}
static enum gimplify_status
gimplify_modify_expr_to_memcpy (tree *expr_p, tree size, bool want_value,
gimple_seq *seq_p)
{
tree t, to, to_ptr, from, from_ptr;
gcall *gs;
location_t loc = EXPR_LOCATION (*expr_p);
to = TREE_OPERAND (*expr_p, 0);
from = TREE_OPERAND (*expr_p, 1);
prepare_gimple_addressable (&from, seq_p);
mark_addressable (from);
from_ptr = build_fold_addr_expr_loc (loc, from);
gimplify_arg (&from_ptr, seq_p, loc);
mark_addressable (to);
to_ptr = build_fold_addr_expr_loc (loc, to);
gimplify_arg (&to_ptr, seq_p, loc);
t = builtin_decl_implicit (BUILT_IN_MEMCPY);
gs = gimple_build_call (t, 3, to_ptr, from_ptr, size);
if (want_value)
{
t = create_tmp_var (TREE_TYPE (to_ptr));
gimple_call_set_lhs (gs, t);
gimplify_seq_add_stmt (seq_p, gs);
*expr_p = build_simple_mem_ref (t);
return GS_ALL_DONE;
}
gimplify_seq_add_stmt (seq_p, gs);
*expr_p = NULL;
return GS_ALL_DONE;
}
static enum gimplify_status
gimplify_modify_expr_to_memset (tree *expr_p, tree size, bool want_value,
gimple_seq *seq_p)
{
tree t, from, to, to_ptr;
gcall *gs;
location_t loc = EXPR_LOCATION (*expr_p);
from = TREE_OPERAND (*expr_p, 1);
if (TREE_CODE (from) == WITH_SIZE_EXPR)
from = TREE_OPERAND (from, 0);
gcc_assert (TREE_CODE (from) == CONSTRUCTOR
&& vec_safe_is_empty (CONSTRUCTOR_ELTS (from)));
to = TREE_OPERAND (*expr_p, 0);
to_ptr = build_fold_addr_expr_loc (loc, to);
gimplify_arg (&to_ptr, seq_p, loc);
t = builtin_decl_implicit (BUILT_IN_MEMSET);
gs = gimple_build_call (t, 3, to_ptr, integer_zero_node, size);
if (want_value)
{
t = create_tmp_var (TREE_TYPE (to_ptr));
gimple_call_set_lhs (gs, t);
gimplify_seq_add_stmt (seq_p, gs);
*expr_p = build1 (INDIRECT_REF, TREE_TYPE (to), t);
return GS_ALL_DONE;
}
gimplify_seq_add_stmt (seq_p, gs);
*expr_p = NULL;
return GS_ALL_DONE;
}
struct gimplify_init_ctor_preeval_data
{
tree lhs_base_decl;
alias_set_type lhs_alias_set;
};
static tree
gimplify_init_ctor_preeval_1 (tree *tp, int *walk_subtrees, void *xdata)
{
struct gimplify_init_ctor_preeval_data *data
= (struct gimplify_init_ctor_preeval_data *) xdata;
tree t = *tp;
if (data->lhs_base_decl == t)
return t;
if ((INDIRECT_REF_P (t)
|| TREE_CODE (t) == MEM_REF)
&& (!data->lhs_base_decl || TREE_ADDRESSABLE (data->lhs_base_decl))
&& alias_sets_conflict_p (data->lhs_alias_set, get_alias_set (t)))
return t;
if (TREE_CODE (t) == CALL_EXPR)
{
tree type, fntype = TREE_TYPE (TREE_TYPE (CALL_EXPR_FN (t)));
for (type = TYPE_ARG_TYPES (fntype); type; type = TREE_CHAIN (type))
if (POINTER_TYPE_P (TREE_VALUE (type))
&& (!data->lhs_base_decl || TREE_ADDRESSABLE (data->lhs_base_decl))
&& alias_sets_conflict_p (data->lhs_alias_set,
get_alias_set
(TREE_TYPE (TREE_VALUE (type)))))
return t;
}
if (IS_TYPE_OR_DECL_P (t))
*walk_subtrees = 0;
return NULL;
}
static void
gimplify_init_ctor_preeval (tree *expr_p, gimple_seq *pre_p, gimple_seq *post_p,
struct gimplify_init_ctor_preeval_data *data)
{
enum gimplify_status one;
if (TREE_CONSTANT (*expr_p))
{
gcc_assert (!TREE_SIDE_EFFECTS (*expr_p));
return;
}
if (TREE_ADDRESSABLE (TREE_TYPE (*expr_p)))
return;
if (TREE_CODE (*expr_p) == CONSTRUCTOR)
{
unsigned HOST_WIDE_INT ix;
constructor_elt *ce;
vec<constructor_elt, va_gc> *v = CONSTRUCTOR_ELTS (*expr_p);
FOR_EACH_VEC_SAFE_ELT (v, ix, ce)
gimplify_init_ctor_preeval (&ce->value, pre_p, post_p, data);
return;
}
maybe_with_size_expr (expr_p);
one = gimplify_expr (expr_p, pre_p, post_p, is_gimple_mem_rhs, fb_rvalue);
if (one == GS_ERROR)
{
*expr_p = NULL;
return;
}
if (DECL_P (*expr_p))
return;
if (TREE_CODE (TYPE_SIZE (TREE_TYPE (*expr_p))) != INTEGER_CST)
return;
if (!walk_tree (expr_p, gimplify_init_ctor_preeval_1, data, NULL))
return;
*expr_p = get_formal_tmp_var (*expr_p, pre_p);
}
static void gimplify_init_ctor_eval (tree, vec<constructor_elt, va_gc> *,
gimple_seq *, bool);
static void
gimplify_init_ctor_eval_range (tree object, tree lower, tree upper,
tree value, tree array_elt_type,
gimple_seq *pre_p, bool cleared)
{
tree loop_entry_label, loop_exit_label, fall_thru_label;
tree var, var_type, cref, tmp;
loop_entry_label = create_artificial_label (UNKNOWN_LOCATION);
loop_exit_label = create_artificial_label (UNKNOWN_LOCATION);
fall_thru_label = create_artificial_label (UNKNOWN_LOCATION);
var_type = TREE_TYPE (upper);
var = create_tmp_var (var_type);
gimplify_seq_add_stmt (pre_p, gimple_build_assign (var, lower));
gimplify_seq_add_stmt (pre_p, gimple_build_label (loop_entry_label));
cref = build4 (ARRAY_REF, array_elt_type, unshare_expr (object),
var, NULL_TREE, NULL_TREE);
if (TREE_CODE (value) == CONSTRUCTOR)
gimplify_init_ctor_eval (cref, CONSTRUCTOR_ELTS (value),
pre_p, cleared);
else
gimplify_seq_add_stmt (pre_p, gimple_build_assign (cref, value));
gimplify_seq_add_stmt (pre_p,
gimple_build_cond (EQ_EXPR, var, upper,
loop_exit_label, fall_thru_label));
gimplify_seq_add_stmt (pre_p, gimple_build_label (fall_thru_label));
tmp = build2 (PLUS_EXPR, var_type, var,
fold_convert (var_type, integer_one_node));
gimplify_seq_add_stmt (pre_p, gimple_build_assign (var, tmp));
gimplify_seq_add_stmt (pre_p, gimple_build_goto (loop_entry_label));
gimplify_seq_add_stmt (pre_p, gimple_build_label (loop_exit_label));
}
static bool
zero_sized_field_decl (const_tree fdecl)
{
if (TREE_CODE (fdecl) == FIELD_DECL && DECL_SIZE (fdecl)
&& integer_zerop (DECL_SIZE (fdecl)))
return true;
return false;
}
static bool
zero_sized_type (const_tree type)
{
if (AGGREGATE_TYPE_P (type) && TYPE_SIZE (type)
&& integer_zerop (TYPE_SIZE (type)))
return true;
return false;
}
static void
gimplify_init_ctor_eval (tree object, vec<constructor_elt, va_gc> *elts,
gimple_seq *pre_p, bool cleared)
{
tree array_elt_type = NULL;
unsigned HOST_WIDE_INT ix;
tree purpose, value;
if (TREE_CODE (TREE_TYPE (object)) == ARRAY_TYPE)
array_elt_type = TYPE_MAIN_VARIANT (TREE_TYPE (TREE_TYPE (object)));
FOR_EACH_CONSTRUCTOR_ELT (elts, ix, purpose, value)
{
tree cref;
if (value == NULL)
continue;
if (cleared && initializer_zerop (value))
continue;
gcc_assert (purpose);
if (! TREE_SIDE_EFFECTS (value) && zero_sized_field_decl (purpose))
continue;
if (TREE_CODE (purpose) == RANGE_EXPR)
{
tree lower = TREE_OPERAND (purpose, 0);
tree upper = TREE_OPERAND (purpose, 1);
if (simple_cst_equal (lower, upper))
purpose = upper;
else
{
gimplify_init_ctor_eval_range (object, lower, upper, value,
array_elt_type, pre_p, cleared);
continue;
}
}
if (array_elt_type)
{
if (TYPE_DOMAIN (TREE_TYPE (object)))
purpose
= fold_convert (TREE_TYPE (TYPE_DOMAIN (TREE_TYPE (object))),
purpose);
cref = build4 (ARRAY_REF, array_elt_type, unshare_expr (object),
purpose, NULL_TREE, NULL_TREE);
}
else
{
gcc_assert (TREE_CODE (purpose) == FIELD_DECL);
cref = build3 (COMPONENT_REF, TREE_TYPE (purpose),
unshare_expr (object), purpose, NULL_TREE);
}
if (TREE_CODE (value) == CONSTRUCTOR
&& TREE_CODE (TREE_TYPE (value)) != VECTOR_TYPE)
gimplify_init_ctor_eval (cref, CONSTRUCTOR_ELTS (value),
pre_p, cleared);
else
{
tree init = build2 (INIT_EXPR, TREE_TYPE (cref), cref, value);
gimplify_and_add (init, pre_p);
ggc_free (init);
}
}
}
gimple_predicate
rhs_predicate_for (tree lhs)
{
if (is_gimple_reg (lhs))
return is_gimple_reg_rhs_or_call;
else
return is_gimple_mem_rhs_or_call;
}
static gimple_predicate
initial_rhs_predicate_for (tree lhs)
{
if (is_gimple_reg_type (TREE_TYPE (lhs)))
return is_gimple_reg_rhs_or_call;
else
return is_gimple_mem_rhs_or_call;
}
static enum gimplify_status
gimplify_compound_literal_expr (tree *expr_p, gimple_seq *pre_p,
bool (*gimple_test_f) (tree),
fallback_t fallback)
{
tree decl_s = COMPOUND_LITERAL_EXPR_DECL_EXPR (*expr_p);
tree decl = DECL_EXPR_DECL (decl_s);
tree init = DECL_INITIAL (decl);
if (TREE_ADDRESSABLE (*expr_p))
TREE_ADDRESSABLE (decl) = 1;
else if (!TREE_ADDRESSABLE (decl)
&& init
&& (fallback & fb_lvalue) == 0
&& gimple_test_f (init))
{
*expr_p = init;
return GS_OK;
}
if ((TREE_CODE (TREE_TYPE (decl)) == COMPLEX_TYPE
|| TREE_CODE (TREE_TYPE (decl)) == VECTOR_TYPE)
&& !TREE_THIS_VOLATILE (decl)
&& !needs_to_live_in_memory (decl))
DECL_GIMPLE_REG_P (decl) = 1;
if (!TREE_ADDRESSABLE (decl) && (fallback & fb_lvalue) == 0)
TREE_READONLY (decl) = 1;
if (DECL_NAME (decl) == NULL_TREE && !DECL_SEEN_IN_BIND_EXPR_P (decl))
gimple_add_tmp_var (decl);
gimplify_and_add (decl_s, pre_p);
*expr_p = decl;
return GS_OK;
}
static tree
optimize_compound_literals_in_ctor (tree orig_ctor)
{
tree ctor = orig_ctor;
vec<constructor_elt, va_gc> *elts = CONSTRUCTOR_ELTS (ctor);
unsigned int idx, num = vec_safe_length (elts);
for (idx = 0; idx < num; idx++)
{
tree value = (*elts)[idx].value;
tree newval = value;
if (TREE_CODE (value) == CONSTRUCTOR)
newval = optimize_compound_literals_in_ctor (value);
else if (TREE_CODE (value) == COMPOUND_LITERAL_EXPR)
{
tree decl_s = COMPOUND_LITERAL_EXPR_DECL_EXPR (value);
tree decl = DECL_EXPR_DECL (decl_s);
tree init = DECL_INITIAL (decl);
if (!TREE_ADDRESSABLE (value)
&& !TREE_ADDRESSABLE (decl)
&& init
&& TREE_CODE (init) == CONSTRUCTOR)
newval = optimize_compound_literals_in_ctor (init);
}
if (newval == value)
continue;
if (ctor == orig_ctor)
{
ctor = copy_node (orig_ctor);
CONSTRUCTOR_ELTS (ctor) = vec_safe_copy (elts);
elts = CONSTRUCTOR_ELTS (ctor);
}
(*elts)[idx].value = newval;
}
return ctor;
}
static enum gimplify_status
gimplify_init_constructor (tree *expr_p, gimple_seq *pre_p, gimple_seq *post_p,
bool want_value, bool notify_temp_creation)
{
tree object, ctor, type;
enum gimplify_status ret;
vec<constructor_elt, va_gc> *elts;
gcc_assert (TREE_CODE (TREE_OPERAND (*expr_p, 1)) == CONSTRUCTOR);
if (!notify_temp_creation)
{
ret = gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p, post_p,
is_gimple_lvalue, fb_lvalue);
if (ret == GS_ERROR)
return ret;
}
object = TREE_OPERAND (*expr_p, 0);
ctor = TREE_OPERAND (*expr_p, 1)
= optimize_compound_literals_in_ctor (TREE_OPERAND (*expr_p, 1));
type = TREE_TYPE (ctor);
elts = CONSTRUCTOR_ELTS (ctor);
ret = GS_ALL_DONE;
switch (TREE_CODE (type))
{
case RECORD_TYPE:
case UNION_TYPE:
case QUAL_UNION_TYPE:
case ARRAY_TYPE:
{
struct gimplify_init_ctor_preeval_data preeval_data;
HOST_WIDE_INT num_ctor_elements, num_nonzero_elements;
HOST_WIDE_INT num_unique_nonzero_elements;
bool cleared, complete_p, valid_const_initializer;
const HOST_WIDE_INT min_unique_size = 64;
const int unique_nonzero_ratio = 8;
if (vec_safe_is_empty (elts))
{
if (notify_temp_creation)
return GS_OK;
break;
}
valid_const_initializer
= categorize_ctor_elements (ctor, &num_nonzero_elements,
&num_unique_nonzero_elements,
&num_ctor_elements, &complete_p);
if (valid_const_initializer
&& num_nonzero_elements > 1
&& TREE_READONLY (object)
&& VAR_P (object)
&& (flag_merge_constants >= 2 || !TREE_ADDRESSABLE (object))
&& (num_unique_nonzero_elements
> num_nonzero_elements / unique_nonzero_ratio
|| ((unsigned HOST_WIDE_INT) int_size_in_bytes (type)
<= (unsigned HOST_WIDE_INT) min_unique_size)))
{
if (notify_temp_creation)
return GS_ERROR;
DECL_INITIAL (object) = ctor;
TREE_STATIC (object) = 1;
if (!DECL_NAME (object))
DECL_NAME (object) = create_tmp_var_name ("C");
walk_tree (&DECL_INITIAL (object), force_labels_r, NULL, NULL);
lhd_set_decl_assembler_name (object);
*expr_p = NULL_TREE;
break;
}
if (int_size_in_bytes (TREE_TYPE (ctor)) < 0)
cleared = false;
else if (!complete_p)
cleared = !CONSTRUCTOR_NO_CLEARING (ctor);
else if (num_ctor_elements - num_nonzero_elements
> CLEAR_RATIO (optimize_function_for_speed_p (cfun))
&& num_nonzero_elements < num_ctor_elements / 4)
cleared = true;
else
cleared = false;
if (valid_const_initializer
&& !(cleared || num_nonzero_elements == 0)
&& !TREE_ADDRESSABLE (type)
&& (!current_function_decl
|| !lookup_attribute ("chkp ctor",
DECL_ATTRIBUTES (current_function_decl))))
{
HOST_WIDE_INT size = int_size_in_bytes (type);
unsigned int align;
if (size < 0)
{
size = int_size_in_bytes (TREE_TYPE (object));
if (size >= 0)
TREE_TYPE (ctor) = type = TREE_TYPE (object);
}
if (DECL_P (object))
align = DECL_ALIGN (object);
else
align = TYPE_ALIGN (type);
if (size > 0
&& num_nonzero_elements > 1
&& (num_unique_nonzero_elements
> num_nonzero_elements / unique_nonzero_ratio
|| size <= min_unique_size)
&& (size < num_nonzero_elements
|| !can_move_by_pieces (size, align)))
{
if (notify_temp_creation)
return GS_ERROR;
walk_tree (&ctor, force_labels_r, NULL, NULL);
ctor = tree_output_constant_def (ctor);
if (!useless_type_conversion_p (type, TREE_TYPE (ctor)))
ctor = build1 (VIEW_CONVERT_EXPR, type, ctor);
TREE_OPERAND (*expr_p, 1) = ctor;
return GS_UNHANDLED;
}
}
if (TREE_THIS_VOLATILE (object)
&& !TREE_ADDRESSABLE (type)
&& num_nonzero_elements > 0
&& vec_safe_length (elts) > 1)
{
tree temp = create_tmp_var (TYPE_MAIN_VARIANT (type));
TREE_OPERAND (*expr_p, 0) = temp;
*expr_p = build2 (COMPOUND_EXPR, TREE_TYPE (*expr_p),
*expr_p,
build2 (MODIFY_EXPR, void_type_node,
object, temp));
return GS_OK;
}
if (notify_temp_creation)
return GS_OK;
if (num_nonzero_elements > 0 && TREE_CODE (*expr_p) != INIT_EXPR)
{
preeval_data.lhs_base_decl = get_base_address (object);
if (!DECL_P (preeval_data.lhs_base_decl))
preeval_data.lhs_base_decl = NULL;
preeval_data.lhs_alias_set = get_alias_set (object);
gimplify_init_ctor_preeval (&TREE_OPERAND (*expr_p, 1),
pre_p, post_p, &preeval_data);
}
bool ctor_has_side_effects_p
= TREE_SIDE_EFFECTS (TREE_OPERAND (*expr_p, 1));
if (cleared)
{
CONSTRUCTOR_ELTS (ctor) = NULL;
TREE_SIDE_EFFECTS (ctor) = 0;
object = unshare_expr (object);
gimplify_stmt (expr_p, pre_p);
}
if (!cleared
|| num_nonzero_elements > 0
|| ctor_has_side_effects_p)
gimplify_init_ctor_eval (object, elts, pre_p, cleared);
*expr_p = NULL_TREE;
}
break;
case COMPLEX_TYPE:
{
tree r, i;
if (notify_temp_creation)
return GS_OK;
gcc_assert (elts->length () == 2);
r = (*elts)[0].value;
i = (*elts)[1].value;
if (r == NULL || i == NULL)
{
tree zero = build_zero_cst (TREE_TYPE (type));
if (r == NULL)
r = zero;
if (i == NULL)
i = zero;
}
if (TREE_CONSTANT (r) && TREE_CONSTANT (i))
{
ctor = build_complex (type, r, i);
TREE_OPERAND (*expr_p, 1) = ctor;
}
else
{
ctor = build2 (COMPLEX_EXPR, type, r, i);
TREE_OPERAND (*expr_p, 1) = ctor;
ret = gimplify_expr (&TREE_OPERAND (*expr_p, 1),
pre_p,
post_p,
rhs_predicate_for (TREE_OPERAND (*expr_p, 0)),
fb_rvalue);
}
}
break;
case VECTOR_TYPE:
{
unsigned HOST_WIDE_INT ix;
constructor_elt *ce;
if (notify_temp_creation)
return GS_OK;
if (TREE_CONSTANT (ctor))
{
bool constant_p = true;
tree value;
FOR_EACH_CONSTRUCTOR_VALUE (elts, ix, value)
if (!CONSTANT_CLASS_P (value))
{
constant_p = false;
break;
}
if (constant_p)
{
TREE_OPERAND (*expr_p, 1) = build_vector_from_ctor (type, elts);
break;
}
TREE_CONSTANT (ctor) = 0;
}
FOR_EACH_VEC_SAFE_ELT (elts, ix, ce)
{
enum gimplify_status tret;
tret = gimplify_expr (&ce->value, pre_p, post_p, is_gimple_val,
fb_rvalue);
if (tret == GS_ERROR)
ret = GS_ERROR;
else if (TREE_STATIC (ctor)
&& !initializer_constant_valid_p (ce->value,
TREE_TYPE (ce->value)))
TREE_STATIC (ctor) = 0;
}
if (!is_gimple_reg (TREE_OPERAND (*expr_p, 0)))
TREE_OPERAND (*expr_p, 1) = get_formal_tmp_var (ctor, pre_p);
}
break;
default:
gcc_unreachable ();
}
if (ret == GS_ERROR)
return GS_ERROR;
if (*expr_p)
{
tree lhs = TREE_OPERAND (*expr_p, 0);
tree rhs = TREE_OPERAND (*expr_p, 1);
if (want_value && object == lhs)
lhs = unshare_expr (lhs);
gassign *init = gimple_build_assign (lhs, rhs);
gimplify_seq_add_stmt (pre_p, init);
}
if (want_value)
{
*expr_p = object;
return GS_OK;
}
else
{
*expr_p = NULL;
return GS_ALL_DONE;
}
}
static tree
gimple_fold_indirect_ref_rhs (tree t)
{
return gimple_fold_indirect_ref (t);
}
static enum gimplify_status
gimplify_modify_expr_rhs (tree *expr_p, tree *from_p, tree *to_p,
gimple_seq *pre_p, gimple_seq *post_p,
bool want_value)
{
enum gimplify_status ret = GS_UNHANDLED;
bool changed;
do
{
changed = false;
switch (TREE_CODE (*from_p))
{
case VAR_DECL:
if (DECL_INITIAL (*from_p)
&& TREE_READONLY (*from_p)
&& !TREE_THIS_VOLATILE (*from_p)
&& !TREE_THIS_VOLATILE (*to_p)
&& TREE_CODE (DECL_INITIAL (*from_p)) == CONSTRUCTOR)
{
tree old_from = *from_p;
enum gimplify_status subret;
*from_p = unshare_expr (DECL_INITIAL (*from_p));
subret = gimplify_init_constructor (expr_p, NULL, NULL,
false, true);
if (subret == GS_ERROR)
{
*from_p = old_from;
}
else
{
ret = GS_OK;
changed = true;
}
}
break;
case INDIRECT_REF:
{
bool volatile_p = TREE_THIS_VOLATILE (*from_p);
tree t = gimple_fold_indirect_ref_rhs (TREE_OPERAND (*from_p, 0));
if (t)
{
if (TREE_THIS_VOLATILE (t) != volatile_p)
{
if (DECL_P (t))
t = build_simple_mem_ref_loc (EXPR_LOCATION (*from_p),
build_fold_addr_expr (t));
if (REFERENCE_CLASS_P (t))
TREE_THIS_VOLATILE (t) = volatile_p;
}
*from_p = t;
ret = GS_OK;
changed = true;
}
break;
}
case TARGET_EXPR:
{
tree init = TARGET_EXPR_INITIAL (*from_p);
if (init
&& (TREE_CODE (*expr_p) != MODIFY_EXPR
|| !TARGET_EXPR_NO_ELIDE (*from_p))
&& !VOID_TYPE_P (TREE_TYPE (init)))
{
*from_p = init;
ret = GS_OK;
changed = true;
}
}
break;
case COMPOUND_EXPR:
gimplify_compound_expr (from_p, pre_p, true);
ret = GS_OK;
changed = true;
break;
case CONSTRUCTOR:
if (ret != GS_UNHANDLED)
break;
return gimplify_init_constructor (expr_p, pre_p, post_p, want_value,
false);
case COND_EXPR:
if (!is_gimple_reg_type (TREE_TYPE (*from_p)))
{
enum tree_code code = TREE_CODE (*expr_p);
tree cond = *from_p;
tree result = *to_p;
ret = gimplify_expr (&result, pre_p, post_p,
is_gimple_lvalue, fb_lvalue);
if (ret != GS_ERROR)
ret = GS_OK;
if (VAR_P (result)
&& TREE_TYPE (TREE_OPERAND (cond, 1)) != void_type_node
&& TREE_TYPE (TREE_OPERAND (cond, 2)) != void_type_node)
TREE_READONLY (result) = 0;
if (TREE_TYPE (TREE_OPERAND (cond, 1)) != void_type_node)
TREE_OPERAND (cond, 1)
= build2 (code, void_type_node, result,
TREE_OPERAND (cond, 1));
if (TREE_TYPE (TREE_OPERAND (cond, 2)) != void_type_node)
TREE_OPERAND (cond, 2)
= build2 (code, void_type_node, unshare_expr (result),
TREE_OPERAND (cond, 2));
TREE_TYPE (cond) = void_type_node;
recalculate_side_effects (cond);
if (want_value)
{
gimplify_and_add (cond, pre_p);
*expr_p = unshare_expr (result);
}
else
*expr_p = cond;
return ret;
}
break;
case CALL_EXPR:
if (!CALL_EXPR_RETURN_SLOT_OPT (*from_p)
&& aggregate_value_p (*from_p, *from_p))
{
bool use_target;
if (!(rhs_predicate_for (*to_p))(*from_p))
use_target = false;
else if (TREE_CODE (*to_p) == RESULT_DECL
&& DECL_NAME (*to_p) == NULL_TREE
&& needs_to_live_in_memory (*to_p))
use_target = true;
else if (is_gimple_reg_type (TREE_TYPE (*to_p))
|| (DECL_P (*to_p) && DECL_REGISTER (*to_p)))
use_target = false;
else if (TREE_CODE (*expr_p) == INIT_EXPR)
use_target = true;
else if (TREE_CODE (TYPE_SIZE_UNIT (TREE_TYPE (*to_p)))
!= INTEGER_CST)
use_target = true;
else if (TREE_CODE (*to_p) != SSA_NAME
&& (!is_gimple_variable (*to_p)
|| needs_to_live_in_memory (*to_p)))
use_target = false;
else
use_target = true;
if (use_target)
{
CALL_EXPR_RETURN_SLOT_OPT (*from_p) = 1;
mark_addressable (*to_p);
}
}
break;
case WITH_SIZE_EXPR:
if (TREE_CODE (TREE_OPERAND (*from_p, 0)) == CALL_EXPR)
{
*from_p = TREE_OPERAND (*from_p, 0);
changed = true;
}
break;
case CLEANUP_POINT_EXPR:
case BIND_EXPR:
case STATEMENT_LIST:
{
tree wrap = *from_p;
tree t;
ret = gimplify_expr (to_p, pre_p, post_p, is_gimple_min_lval,
fb_lvalue);
if (ret != GS_ERROR)
ret = GS_OK;
t = voidify_wrapper_expr (wrap, *expr_p);
gcc_assert (t == *expr_p);
if (want_value)
{
gimplify_and_add (wrap, pre_p);
*expr_p = unshare_expr (*to_p);
}
else
*expr_p = wrap;
return GS_OK;
}
case COMPOUND_LITERAL_EXPR:
{
tree complit = TREE_OPERAND (*expr_p, 1);
tree decl_s = COMPOUND_LITERAL_EXPR_DECL_EXPR (complit);
tree decl = DECL_EXPR_DECL (decl_s);
tree init = DECL_INITIAL (decl);
if (!TREE_ADDRESSABLE (complit)
&& !TREE_ADDRESSABLE (decl)
&& init)
{
*expr_p = copy_node (*expr_p);
TREE_OPERAND (*expr_p, 1) = init;
return GS_OK;
}
}
default:
break;
}
}
while (changed);
return ret;
}
static bool
is_gimple_stmt (tree t)
{
const enum tree_code code = TREE_CODE (t);
switch (code)
{
case NOP_EXPR:
return IS_EMPTY_STMT (t);
case BIND_EXPR:
case COND_EXPR:
return TREE_TYPE (t) == NULL || VOID_TYPE_P (TREE_TYPE (t));
case SWITCH_EXPR:
case GOTO_EXPR:
case RETURN_EXPR:
case LABEL_EXPR:
case CASE_LABEL_EXPR:
case TRY_CATCH_EXPR:
case TRY_FINALLY_EXPR:
case EH_FILTER_EXPR:
case CATCH_EXPR:
case ASM_EXPR:
case STATEMENT_LIST:
case OACC_PARALLEL:
case OACC_KERNELS:
case OACC_DATA:
case OACC_HOST_DATA:
case OACC_DECLARE:
case OACC_UPDATE:
case OACC_ENTER_DATA:
case OACC_EXIT_DATA:
case OACC_CACHE:
case OMP_PARALLEL:
case OMP_FOR:
case OMP_SIMD:
case OMP_DISTRIBUTE:
case OACC_LOOP:
case OMP_SECTIONS:
case OMP_SECTION:
case OMP_SINGLE:
case OMP_MASTER:
case OMP_TASKGROUP:
case OMP_ORDERED:
case OMP_CRITICAL:
case OMP_TASK:
case OMP_TARGET:
case OMP_TARGET_DATA:
case OMP_TARGET_UPDATE:
case OMP_TARGET_ENTER_DATA:
case OMP_TARGET_EXIT_DATA:
case OMP_TASKLOOP:
case OMP_TEAMS:
return true;
case CALL_EXPR:
case MODIFY_EXPR:
case PREDICT_EXPR:
return true;
default:
return false;
}
}
static enum gimplify_status
gimplify_modify_expr_complex_part (tree *expr_p, gimple_seq *pre_p,
bool want_value)
{
enum tree_code code, ocode;
tree lhs, rhs, new_rhs, other, realpart, imagpart;
lhs = TREE_OPERAND (*expr_p, 0);
rhs = TREE_OPERAND (*expr_p, 1);
code = TREE_CODE (lhs);
lhs = TREE_OPERAND (lhs, 0);
ocode = code == REALPART_EXPR ? IMAGPART_EXPR : REALPART_EXPR;
other = build1 (ocode, TREE_TYPE (rhs), lhs);
TREE_NO_WARNING (other) = 1;
other = get_formal_tmp_var (other, pre_p);
realpart = code == REALPART_EXPR ? rhs : other;
imagpart = code == REALPART_EXPR ? other : rhs;
if (TREE_CONSTANT (realpart) && TREE_CONSTANT (imagpart))
new_rhs = build_complex (TREE_TYPE (lhs), realpart, imagpart);
else
new_rhs = build2 (COMPLEX_EXPR, TREE_TYPE (lhs), realpart, imagpart);
gimplify_seq_add_stmt (pre_p, gimple_build_assign (lhs, new_rhs));
*expr_p = (want_value) ? rhs : NULL_TREE;
return GS_ALL_DONE;
}
static enum gimplify_status
gimplify_modify_expr (tree *expr_p, gimple_seq *pre_p, gimple_seq *post_p,
bool want_value)
{
tree *from_p = &TREE_OPERAND (*expr_p, 1);
tree *to_p = &TREE_OPERAND (*expr_p, 0);
enum gimplify_status ret = GS_UNHANDLED;
gimple *assign;
location_t loc = EXPR_LOCATION (*expr_p);
gimple_stmt_iterator gsi;
gcc_assert (TREE_CODE (*expr_p) == MODIFY_EXPR
|| TREE_CODE (*expr_p) == INIT_EXPR);
if (TREE_CLOBBER_P (*from_p))
{
ret = gimplify_expr (to_p, pre_p, post_p, is_gimple_lvalue, fb_lvalue);
if (ret == GS_ERROR)
return ret;
gcc_assert (!want_value
&& (VAR_P (*to_p) || TREE_CODE (*to_p) == MEM_REF));
gimplify_seq_add_stmt (pre_p, gimple_build_assign (*to_p, *from_p));
*expr_p = NULL;
return GS_ALL_DONE;
}
if (POINTER_TYPE_P (TREE_TYPE (*to_p)))
{
STRIP_USELESS_TYPE_CONVERSION (*from_p);
if (!useless_type_conversion_p (TREE_TYPE (*to_p), TREE_TYPE (*from_p)))
*from_p = fold_convert_loc (loc, TREE_TYPE (*to_p), *from_p);
}
ret = gimplify_modify_expr_rhs (expr_p, from_p, to_p, pre_p, post_p,
want_value);
if (ret != GS_UNHANDLED)
return ret;
if (zero_sized_type (TREE_TYPE (*from_p))
&& !want_value
&& !(TREE_ADDRESSABLE (TREE_TYPE (*from_p))
&& TREE_CODE (*from_p) == CALL_EXPR))
{
gimplify_stmt (from_p, pre_p);
gimplify_stmt (to_p, pre_p);
*expr_p = NULL_TREE;
return GS_ALL_DONE;
}
maybe_with_size_expr (from_p);
gimple_predicate initial_pred = initial_rhs_predicate_for (*to_p);
ret = gimplify_expr (from_p, pre_p, post_p, initial_pred, fb_rvalue);
if (ret == GS_ERROR)
return ret;
bool saved_into_ssa = gimplify_ctxp->into_ssa;
if (saved_into_ssa
&& TREE_CODE (*from_p) == CALL_EXPR
&& call_expr_flags (*from_p) & ECF_RETURNS_TWICE)
gimplify_ctxp->into_ssa = false;
ret = gimplify_expr (to_p, pre_p, post_p, is_gimple_lvalue, fb_lvalue);
gimplify_ctxp->into_ssa = saved_into_ssa;
if (ret == GS_ERROR)
return ret;
gimple_predicate final_pred = rhs_predicate_for (*to_p);
if (final_pred != initial_pred)
{
ret = gimplify_expr (from_p, pre_p, post_p, final_pred, fb_rvalue);
if (ret == GS_ERROR)
return ret;
}
if (TREE_CODE (*from_p) == WITH_SIZE_EXPR)
{
tree call = TREE_OPERAND (*from_p, 0);
tree vlasize = TREE_OPERAND (*from_p, 1);
if (TREE_CODE (call) == CALL_EXPR
&& CALL_EXPR_IFN (call) == IFN_VA_ARG)
{
int nargs = call_expr_nargs (call);
tree type = TREE_TYPE (call);
tree ap = CALL_EXPR_ARG (call, 0);
tree tag = CALL_EXPR_ARG (call, 1);
tree aptag = CALL_EXPR_ARG (call, 2);
tree newcall = build_call_expr_internal_loc (EXPR_LOCATION (call),
IFN_VA_ARG, type,
nargs + 1, ap, tag,
aptag, vlasize);
TREE_OPERAND (*from_p, 0) = newcall;
}
}
ret = gimplify_modify_expr_rhs (expr_p, from_p, to_p, pre_p, post_p,
want_value);
if (ret != GS_UNHANDLED)
return ret;
if (TREE_CODE (*from_p) == WITH_SIZE_EXPR)
{
tree from = TREE_OPERAND (*from_p, 0);
tree size = TREE_OPERAND (*from_p, 1);
if (TREE_CODE (from) == CONSTRUCTOR)
return gimplify_modify_expr_to_memset (expr_p, size, want_value, pre_p);
if (is_gimple_addressable (from))
{
*from_p = from;
return gimplify_modify_expr_to_memcpy (expr_p, size, want_value,
pre_p);
}
}
if ((TREE_CODE (*to_p) == REALPART_EXPR
|| TREE_CODE (*to_p) == IMAGPART_EXPR)
&& is_gimple_reg (TREE_OPERAND (*to_p, 0)))
return gimplify_modify_expr_complex_part (expr_p, pre_p, want_value);
if (!gimplify_ctxp->into_ssa
&& VAR_P (*from_p)
&& DECL_IGNORED_P (*from_p)
&& DECL_P (*to_p)
&& !DECL_IGNORED_P (*to_p)
&& decl_function_context (*to_p) == current_function_decl
&& decl_function_context (*from_p) == current_function_decl)
{
if (!DECL_NAME (*from_p) && DECL_NAME (*to_p))
DECL_NAME (*from_p)
= create_tmp_var_name (IDENTIFIER_POINTER (DECL_NAME (*to_p)));
DECL_HAS_DEBUG_EXPR_P (*from_p) = 1;
SET_DECL_DEBUG_EXPR (*from_p, *to_p);
}
if (want_value && TREE_THIS_VOLATILE (*to_p))
*from_p = get_initialized_tmp_var (*from_p, pre_p, post_p);
if (TREE_CODE (*from_p) == CALL_EXPR)
{
gcall *call_stmt;
if (CALL_EXPR_FN (*from_p) == NULL_TREE)
{
int nargs = call_expr_nargs (*from_p), i;
enum internal_fn ifn = CALL_EXPR_IFN (*from_p);
auto_vec<tree> vargs (nargs);
for (i = 0; i < nargs; i++)
{
gimplify_arg (&CALL_EXPR_ARG (*from_p, i), pre_p,
EXPR_LOCATION (*from_p));
vargs.quick_push (CALL_EXPR_ARG (*from_p, i));
}
call_stmt = gimple_build_call_internal_vec (ifn, vargs);
gimple_call_set_nothrow (call_stmt, TREE_NOTHROW (*from_p));
gimple_set_location (call_stmt, EXPR_LOCATION (*expr_p));
}
else
{
tree fnptrtype = TREE_TYPE (CALL_EXPR_FN (*from_p));
CALL_EXPR_FN (*from_p) = TREE_OPERAND (CALL_EXPR_FN (*from_p), 0);
STRIP_USELESS_TYPE_CONVERSION (CALL_EXPR_FN (*from_p));
tree fndecl = get_callee_fndecl (*from_p);
if (fndecl
&& DECL_BUILT_IN_CLASS (fndecl) == BUILT_IN_NORMAL
&& DECL_FUNCTION_CODE (fndecl) == BUILT_IN_EXPECT
&& call_expr_nargs (*from_p) == 3)
call_stmt = gimple_build_call_internal (IFN_BUILTIN_EXPECT, 3,
CALL_EXPR_ARG (*from_p, 0),
CALL_EXPR_ARG (*from_p, 1),
CALL_EXPR_ARG (*from_p, 2));
else
{
call_stmt = gimple_build_call_from_tree (*from_p, fnptrtype);
}
}
notice_special_calls (call_stmt);
if (!gimple_call_noreturn_p (call_stmt) || !should_remove_lhs_p (*to_p))
gimple_call_set_lhs (call_stmt, *to_p);
else if (TREE_CODE (*to_p) == SSA_NAME)
SSA_NAME_DEF_STMT (*to_p) = gimple_build_nop ();
assign = call_stmt;
}
else
{
assign = gimple_build_assign (*to_p, *from_p);
gimple_set_location (assign, EXPR_LOCATION (*expr_p));
if (COMPARISON_CLASS_P (*from_p))
gimple_set_no_warning (assign, TREE_NO_WARNING (*from_p));
}
if (gimplify_ctxp->into_ssa && is_gimple_reg (*to_p))
{
gcc_assert (TREE_CODE (*to_p) == SSA_NAME
|| ! gimple_in_ssa_p (cfun));
}
gimplify_seq_add_stmt (pre_p, assign);
gsi = gsi_last (*pre_p);
maybe_fold_stmt (&gsi);
if (want_value)
{
*expr_p = TREE_THIS_VOLATILE (*to_p) ? *from_p : unshare_expr (*to_p);
return GS_OK;
}
else
*expr_p = NULL;
return GS_ALL_DONE;
}
static enum gimplify_status
gimplify_variable_sized_compare (tree *expr_p)
{
location_t loc = EXPR_LOCATION (*expr_p);
tree op0 = TREE_OPERAND (*expr_p, 0);
tree op1 = TREE_OPERAND (*expr_p, 1);
tree t, arg, dest, src, expr;
arg = TYPE_SIZE_UNIT (TREE_TYPE (op0));
arg = unshare_expr (arg);
arg = SUBSTITUTE_PLACEHOLDER_IN_EXPR (arg, op0);
src = build_fold_addr_expr_loc (loc, op1);
dest = build_fold_addr_expr_loc (loc, op0);
t = builtin_decl_implicit (BUILT_IN_MEMCMP);
t = build_call_expr_loc (loc, t, 3, dest, src, arg);
expr
= build2 (TREE_CODE (*expr_p), TREE_TYPE (*expr_p), t, integer_zero_node);
SET_EXPR_LOCATION (expr, loc);
*expr_p = expr;
return GS_OK;
}
static enum gimplify_status
gimplify_scalar_mode_aggregate_compare (tree *expr_p)
{
location_t loc = EXPR_LOCATION (*expr_p);
tree op0 = TREE_OPERAND (*expr_p, 0);
tree op1 = TREE_OPERAND (*expr_p, 1);
tree type = TREE_TYPE (op0);
tree scalar_type = lang_hooks.types.type_for_mode (TYPE_MODE (type), 1);
op0 = fold_build1_loc (loc, VIEW_CONVERT_EXPR, scalar_type, op0);
op1 = fold_build1_loc (loc, VIEW_CONVERT_EXPR, scalar_type, op1);
*expr_p
= fold_build2_loc (loc, TREE_CODE (*expr_p), TREE_TYPE (*expr_p), op0, op1);
return GS_OK;
}
static enum gimplify_status
gimplify_compound_expr (tree *expr_p, gimple_seq *pre_p, bool want_value)
{
tree t = *expr_p;
do
{
tree *sub_p = &TREE_OPERAND (t, 0);
if (TREE_CODE (*sub_p) == COMPOUND_EXPR)
gimplify_compound_expr (sub_p, pre_p, false);
else
gimplify_stmt (sub_p, pre_p);
t = TREE_OPERAND (t, 1);
}
while (TREE_CODE (t) == COMPOUND_EXPR);
*expr_p = t;
if (want_value)
return GS_OK;
else
{
gimplify_stmt (expr_p, pre_p);
return GS_ALL_DONE;
}
}
static enum gimplify_status
gimplify_save_expr (tree *expr_p, gimple_seq *pre_p, gimple_seq *post_p)
{
enum gimplify_status ret = GS_ALL_DONE;
tree val;
gcc_assert (TREE_CODE (*expr_p) == SAVE_EXPR);
val = TREE_OPERAND (*expr_p, 0);
if (!SAVE_EXPR_RESOLVED_P (*expr_p))
{
if (TREE_TYPE (val) == void_type_node)
{
ret = gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p, post_p,
is_gimple_stmt, fb_none);
val = NULL;
}
else
val = get_initialized_tmp_var (val, pre_p, post_p,
gimple_in_ssa_p (cfun));
TREE_OPERAND (*expr_p, 0) = val;
SAVE_EXPR_RESOLVED_P (*expr_p) = 1;
}
*expr_p = val;
return ret;
}
static enum gimplify_status
gimplify_addr_expr (tree *expr_p, gimple_seq *pre_p, gimple_seq *post_p)
{
tree expr = *expr_p;
tree op0 = TREE_OPERAND (expr, 0);
enum gimplify_status ret;
location_t loc = EXPR_LOCATION (*expr_p);
switch (TREE_CODE (op0))
{
case INDIRECT_REF:
do_indirect_ref:
{
tree op00 = TREE_OPERAND (op0, 0);
tree t_expr = TREE_TYPE (expr);
tree t_op00 = TREE_TYPE (op00);
if (!useless_type_conversion_p (t_expr, t_op00))
op00 = fold_convert_loc (loc, TREE_TYPE (expr), op00);
*expr_p = op00;
ret = GS_OK;
}
break;
case VIEW_CONVERT_EXPR:
if (tree_ssa_useless_type_conversion (TREE_OPERAND (op0, 0)))
op0 = TREE_OPERAND (op0, 0);
*expr_p = fold_convert_loc (loc, TREE_TYPE (expr),
build_fold_addr_expr_loc (loc,
TREE_OPERAND (op0, 0)));
ret = GS_OK;
break;
case MEM_REF:
if (integer_zerop (TREE_OPERAND (op0, 1)))
goto do_indirect_ref;
default:
if (TREE_CODE (op0) == FUNCTION_DECL
&& DECL_BUILT_IN_CLASS (op0) == BUILT_IN_NORMAL
&& builtin_decl_declared_p (DECL_FUNCTION_CODE (op0)))
set_builtin_decl_implicit_p (DECL_FUNCTION_CODE (op0), true);
ret = gimplify_expr (&TREE_OPERAND (expr, 0), pre_p, post_p,
is_gimple_addressable, fb_either);
if (ret == GS_ERROR)
break;
prepare_gimple_addressable (&TREE_OPERAND (expr, 0), pre_p);
op0 = TREE_OPERAND (expr, 0);
if (TREE_CODE (op0) == INDIRECT_REF)
goto do_indirect_ref;
mark_addressable (TREE_OPERAND (expr, 0));
if (!types_compatible_p (TREE_TYPE (op0), TREE_TYPE (TREE_TYPE (expr))))
*expr_p = build_fold_addr_expr (op0);
recompute_tree_invariant_for_addr_expr (*expr_p);
if (!useless_type_conversion_p (TREE_TYPE (expr), TREE_TYPE (*expr_p)))
*expr_p = fold_convert (TREE_TYPE (expr), *expr_p);
break;
}
return ret;
}
static enum gimplify_status
gimplify_asm_expr (tree *expr_p, gimple_seq *pre_p, gimple_seq *post_p)
{
tree expr;
int noutputs;
const char **oconstraints;
int i;
tree link;
const char *constraint;
bool allows_mem, allows_reg, is_inout;
enum gimplify_status ret, tret;
gasm *stmt;
vec<tree, va_gc> *inputs;
vec<tree, va_gc> *outputs;
vec<tree, va_gc> *clobbers;
vec<tree, va_gc> *labels;
tree link_next;
expr = *expr_p;
noutputs = list_length (ASM_OUTPUTS (expr));
oconstraints = (const char **) alloca ((noutputs) * sizeof (const char *));
inputs = NULL;
outputs = NULL;
clobbers = NULL;
labels = NULL;
ret = GS_ALL_DONE;
link_next = NULL_TREE;
for (i = 0, link = ASM_OUTPUTS (expr); link; ++i, link = link_next)
{
bool ok;
size_t constraint_len;
link_next = TREE_CHAIN (link);
oconstraints[i]
= constraint
= TREE_STRING_POINTER (TREE_VALUE (TREE_PURPOSE (link)));
constraint_len = strlen (constraint);
if (constraint_len == 0)
continue;
ok = parse_output_constraint (&constraint, i, 0, 0,
&allows_mem, &allows_reg, &is_inout);
if (!ok)
{
ret = GS_ERROR;
is_inout = false;
}
if (!allows_reg && allows_mem)
mark_addressable (TREE_VALUE (link));
tret = gimplify_expr (&TREE_VALUE (link), pre_p, post_p,
is_inout ? is_gimple_min_lval : is_gimple_lvalue,
fb_lvalue | fb_mayfail);
if (tret == GS_ERROR)
{
error ("invalid lvalue in asm output %d", i);
ret = tret;
}
if (!allows_mem)
{
tree op = TREE_VALUE (link);
if (! is_gimple_val (op)
&& is_gimple_reg_type (TREE_TYPE (op))
&& is_gimple_reg (get_base_address (op)))
{
tree tem = create_tmp_reg (TREE_TYPE (op));
tree ass;
if (is_inout)
{
ass = build2 (MODIFY_EXPR, TREE_TYPE (tem),
tem, unshare_expr (op));
gimplify_and_add (ass, pre_p);
}
ass = build2 (MODIFY_EXPR, TREE_TYPE (tem), op, tem);
gimplify_and_add (ass, post_p);
TREE_VALUE (link) = tem;
tret = GS_OK;
}
}
vec_safe_push (outputs, link);
TREE_CHAIN (link) = NULL_TREE;
if (is_inout)
{
tree input;
char buf[11];
char *p = xstrdup (constraint);
p[0] = '=';
TREE_VALUE (TREE_PURPOSE (link)) = build_string (constraint_len, p);
if (allows_reg)
{
sprintf (buf, "%u", i);
if (strchr (p, ',') != NULL)
{
size_t len = 0, buflen = strlen (buf);
char *beg, *end, *str, *dst;
for (beg = p + 1;;)
{
end = strchr (beg, ',');
if (end == NULL)
end = strchr (beg, '\0');
if ((size_t) (end - beg) < buflen)
len += buflen + 1;
else
len += end - beg + 1;
if (*end)
beg = end + 1;
else
break;
}
str = (char *) alloca (len);
for (beg = p + 1, dst = str;;)
{
const char *tem;
bool mem_p, reg_p, inout_p;
end = strchr (beg, ',');
if (end)
*end = '\0';
beg[-1] = '=';
tem = beg - 1;
parse_output_constraint (&tem, i, 0, 0,
&mem_p, &reg_p, &inout_p);
if (dst != str)
*dst++ = ',';
if (reg_p)
{
memcpy (dst, buf, buflen);
dst += buflen;
}
else
{
if (end)
len = end - beg;
else
len = strlen (beg);
memcpy (dst, beg, len);
dst += len;
}
if (end)
beg = end + 1;
else
break;
}
*dst = '\0';
input = build_string (dst - str, str);
}
else
input = build_string (strlen (buf), buf);
}
else
input = build_string (constraint_len - 1, constraint + 1);
free (p);
input = build_tree_list (build_tree_list (NULL_TREE, input),
unshare_expr (TREE_VALUE (link)));
ASM_INPUTS (expr) = chainon (ASM_INPUTS (expr), input);
}
}
link_next = NULL_TREE;
for (link = ASM_INPUTS (expr); link; ++i, link = link_next)
{
link_next = TREE_CHAIN (link);
constraint = TREE_STRING_POINTER (TREE_VALUE (TREE_PURPOSE (link)));
parse_input_constraint (&constraint, 0, 0, noutputs, 0,
oconstraints, &allows_mem, &allows_reg);
if (TREE_ADDRESSABLE (TREE_TYPE (TREE_VALUE (link))))
{
if (allows_mem)
allows_reg = 0;
else
{
error ("impossible constraint in %<asm%>");
error ("non-memory input %d must stay in memory", i);
return GS_ERROR;
}
}
if (!allows_reg && allows_mem)
{
tree inputv = TREE_VALUE (link);
STRIP_NOPS (inputv);
if (TREE_CODE (inputv) == PREDECREMENT_EXPR
|| TREE_CODE (inputv) == PREINCREMENT_EXPR
|| TREE_CODE (inputv) == POSTDECREMENT_EXPR
|| TREE_CODE (inputv) == POSTINCREMENT_EXPR
|| TREE_CODE (inputv) == MODIFY_EXPR)
TREE_VALUE (link) = error_mark_node;
tret = gimplify_expr (&TREE_VALUE (link), pre_p, post_p,
is_gimple_lvalue, fb_lvalue | fb_mayfail);
if (tret != GS_ERROR)
{
tree x = TREE_VALUE (link);
while (handled_component_p (x))
x = TREE_OPERAND (x, 0);
if (TREE_CODE (x) == MEM_REF
&& TREE_CODE (TREE_OPERAND (x, 0)) == ADDR_EXPR)
x = TREE_OPERAND (TREE_OPERAND (x, 0), 0);
if ((VAR_P (x)
|| TREE_CODE (x) == PARM_DECL
|| TREE_CODE (x) == RESULT_DECL)
&& !TREE_ADDRESSABLE (x)
&& is_gimple_reg (x))
{
warning_at (EXPR_LOC_OR_LOC (TREE_VALUE (link),
input_location), 0,
"memory input %d is not directly addressable",
i);
prepare_gimple_addressable (&TREE_VALUE (link), pre_p);
}
}
mark_addressable (TREE_VALUE (link));
if (tret == GS_ERROR)
{
error_at (EXPR_LOC_OR_LOC (TREE_VALUE (link), input_location),
"memory input %d is not directly addressable", i);
ret = tret;
}
}
else
{
tret = gimplify_expr (&TREE_VALUE (link), pre_p, post_p,
is_gimple_asm_val, fb_rvalue);
if (tret == GS_ERROR)
ret = tret;
}
TREE_CHAIN (link) = NULL_TREE;
vec_safe_push (inputs, link);
}
link_next = NULL_TREE;
for (link = ASM_CLOBBERS (expr); link; ++i, link = link_next)
{
link_next = TREE_CHAIN (link);
TREE_CHAIN (link) = NULL_TREE;
vec_safe_push (clobbers, link);
}
link_next = NULL_TREE;
for (link = ASM_LABELS (expr); link; ++i, link = link_next)
{
link_next = TREE_CHAIN (link);
TREE_CHAIN (link) = NULL_TREE;
vec_safe_push (labels, link);
}
if (ret != GS_ERROR)
{
stmt = gimple_build_asm_vec (TREE_STRING_POINTER (ASM_STRING (expr)),
inputs, outputs, clobbers, labels);
gimple_asm_set_volatile (stmt, ASM_VOLATILE_P (expr) || noutputs == 0);
gimple_asm_set_input (stmt, ASM_INPUT_P (expr));
gimple_asm_set_inline (stmt, ASM_INLINE_P (expr));
gimplify_seq_add_stmt (pre_p, stmt);
}
return ret;
}
static enum gimplify_status
gimplify_cleanup_point_expr (tree *expr_p, gimple_seq *pre_p)
{
gimple_stmt_iterator iter;
gimple_seq body_sequence = NULL;
tree temp = voidify_wrapper_expr (*expr_p, NULL);
int old_conds = gimplify_ctxp->conditions;
gimple_seq old_cleanups = gimplify_ctxp->conditional_cleanups;
bool old_in_cleanup_point_expr = gimplify_ctxp->in_cleanup_point_expr;
gimplify_ctxp->conditions = 0;
gimplify_ctxp->conditional_cleanups = NULL;
gimplify_ctxp->in_cleanup_point_expr = true;
gimplify_stmt (&TREE_OPERAND (*expr_p, 0), &body_sequence);
gimplify_ctxp->conditions = old_conds;
gimplify_ctxp->conditional_cleanups = old_cleanups;
gimplify_ctxp->in_cleanup_point_expr = old_in_cleanup_point_expr;
for (iter = gsi_start (body_sequence); !gsi_end_p (iter); )
{
gimple *wce = gsi_stmt (iter);
if (gimple_code (wce) == GIMPLE_WITH_CLEANUP_EXPR)
{
if (gsi_one_before_end_p (iter))
{
if (!gimple_wce_cleanup_eh_only (wce))
gsi_insert_seq_before_without_update (&iter,
gimple_wce_cleanup (wce),
GSI_SAME_STMT);
gsi_remove (&iter, true);
break;
}
else
{
gtry *gtry;
gimple_seq seq;
enum gimple_try_flags kind;
if (gimple_wce_cleanup_eh_only (wce))
kind = GIMPLE_TRY_CATCH;
else
kind = GIMPLE_TRY_FINALLY;
seq = gsi_split_seq_after (iter);
gtry = gimple_build_try (seq, gimple_wce_cleanup (wce), kind);
gsi_set_stmt (&iter, gtry);
iter = gsi_start (gtry->eval);
}
}
else
gsi_next (&iter);
}
gimplify_seq_add_seq (pre_p, body_sequence);
if (temp)
{
*expr_p = temp;
return GS_OK;
}
else
{
*expr_p = NULL;
return GS_ALL_DONE;
}
}
static void
gimple_push_cleanup (tree var, tree cleanup, bool eh_only, gimple_seq *pre_p,
bool force_uncond = false)
{
gimple *wce;
gimple_seq cleanup_stmts = NULL;
if (seen_error ())
return;
if (gimple_conditional_context ())
{
if (force_uncond)
{
gimplify_stmt (&cleanup, &cleanup_stmts);
wce = gimple_build_wce (cleanup_stmts);
gimplify_seq_add_stmt (&gimplify_ctxp->conditional_cleanups, wce);
}
else
{
tree flag = create_tmp_var (boolean_type_node, "cleanup");
gassign *ffalse = gimple_build_assign (flag, boolean_false_node);
gassign *ftrue = gimple_build_assign (flag, boolean_true_node);
cleanup = build3 (COND_EXPR, void_type_node, flag, cleanup, NULL);
gimplify_stmt (&cleanup, &cleanup_stmts);
wce = gimple_build_wce (cleanup_stmts);
gimplify_seq_add_stmt (&gimplify_ctxp->conditional_cleanups, ffalse);
gimplify_seq_add_stmt (&gimplify_ctxp->conditional_cleanups, wce);
gimplify_seq_add_stmt (pre_p, ftrue);
TREE_NO_WARNING (var) = 1;
}
}
else
{
gimplify_stmt (&cleanup, &cleanup_stmts);
wce = gimple_build_wce (cleanup_stmts);
gimple_wce_set_cleanup_eh_only (wce, eh_only);
gimplify_seq_add_stmt (pre_p, wce);
}
}
static enum gimplify_status
gimplify_target_expr (tree *expr_p, gimple_seq *pre_p, gimple_seq *post_p)
{
tree targ = *expr_p;
tree temp = TARGET_EXPR_SLOT (targ);
tree init = TARGET_EXPR_INITIAL (targ);
enum gimplify_status ret;
bool unpoison_empty_seq = false;
gimple_stmt_iterator unpoison_it;
if (init)
{
tree cleanup = NULL_TREE;
if (TREE_CODE (DECL_SIZE (temp)) != INTEGER_CST)
{
if (!TYPE_SIZES_GIMPLIFIED (TREE_TYPE (temp)))
gimplify_type_sizes (TREE_TYPE (temp), pre_p);
gimplify_vla_decl (temp, pre_p);
}
else
{
unpoison_it = gsi_last (*pre_p);
unpoison_empty_seq = gsi_end_p (unpoison_it);
gimple_add_tmp_var (temp);
}
if (VOID_TYPE_P (TREE_TYPE (init)))
ret = gimplify_expr (&init, pre_p, post_p, is_gimple_stmt, fb_none);
else
{
tree init_expr = build2 (INIT_EXPR, void_type_node, temp, init);
init = init_expr;
ret = gimplify_expr (&init, pre_p, post_p, is_gimple_stmt, fb_none);
init = NULL;
ggc_free (init_expr);
}
if (ret == GS_ERROR)
{
TARGET_EXPR_INITIAL (targ) = NULL_TREE;
return GS_ERROR;
}
if (init)
gimplify_and_add (init, pre_p);
if (TARGET_EXPR_CLEANUP (targ))
{
if (CLEANUP_EH_ONLY (targ))
gimple_push_cleanup (temp, TARGET_EXPR_CLEANUP (targ),
CLEANUP_EH_ONLY (targ), pre_p);
else
cleanup = TARGET_EXPR_CLEANUP (targ);
}
if (gimplify_ctxp->in_cleanup_point_expr
&& needs_to_live_in_memory (temp))
{
if (flag_stack_reuse == SR_ALL)
{
tree clobber = build_constructor (TREE_TYPE (temp),
NULL);
TREE_THIS_VOLATILE (clobber) = true;
clobber = build2 (MODIFY_EXPR, TREE_TYPE (temp), temp, clobber);
gimple_push_cleanup (temp, clobber, false, pre_p, true);
}
if (asan_poisoned_variables
&& DECL_ALIGN (temp) <= MAX_SUPPORTED_STACK_ALIGNMENT
&& dbg_cnt (asan_use_after_scope)
&& !gimplify_omp_ctxp)
{
tree asan_cleanup = build_asan_poison_call_expr (temp);
if (asan_cleanup)
{
if (unpoison_empty_seq)
unpoison_it = gsi_start (*pre_p);
asan_poison_variable (temp, false, &unpoison_it,
unpoison_empty_seq);
gimple_push_cleanup (temp, asan_cleanup, false, pre_p);
}
}
}
if (cleanup)
gimple_push_cleanup (temp, cleanup, false, pre_p);
TREE_OPERAND (targ, 3) = init;
TARGET_EXPR_INITIAL (targ) = NULL_TREE;
}
else
gcc_assert (DECL_SEEN_IN_BIND_EXPR_P (temp));
*expr_p = temp;
return GS_OK;
}
bool
gimplify_stmt (tree *stmt_p, gimple_seq *seq_p)
{
gimple_seq_node last;
last = gimple_seq_last (*seq_p);
gimplify_expr (stmt_p, seq_p, NULL, is_gimple_stmt, fb_none);
return last != gimple_seq_last (*seq_p);
}
void
omp_firstprivatize_variable (struct gimplify_omp_ctx *ctx, tree decl)
{
splay_tree_node n;
if (decl == NULL || !DECL_P (decl) || ctx->region_type == ORT_NONE)
return;
do
{
n = splay_tree_lookup (ctx->variables, (splay_tree_key)decl);
if (n != NULL)
{
if (n->value & GOVD_SHARED)
n->value = GOVD_FIRSTPRIVATE | (n->value & GOVD_SEEN);
else if (n->value & GOVD_MAP)
n->value |= GOVD_MAP_TO_ONLY;
else
return;
}
else if ((ctx->region_type & ORT_TARGET) != 0)
{
if (ctx->target_map_scalars_firstprivate)
omp_add_variable (ctx, decl, GOVD_FIRSTPRIVATE);
else
omp_add_variable (ctx, decl, GOVD_MAP | GOVD_MAP_TO_ONLY);
}
else if (ctx->region_type != ORT_WORKSHARE
&& ctx->region_type != ORT_SIMD
&& ctx->region_type != ORT_ACC
&& !(ctx->region_type & ORT_TARGET_DATA))
omp_add_variable (ctx, decl, GOVD_FIRSTPRIVATE);
ctx = ctx->outer_context;
}
while (ctx);
}
static void
omp_firstprivatize_type_sizes (struct gimplify_omp_ctx *ctx, tree type)
{
if (type == NULL || type == error_mark_node)
return;
type = TYPE_MAIN_VARIANT (type);
if (ctx->privatized_types->add (type))
return;
switch (TREE_CODE (type))
{
case INTEGER_TYPE:
case ENUMERAL_TYPE:
case BOOLEAN_TYPE:
case REAL_TYPE:
case FIXED_POINT_TYPE:
omp_firstprivatize_variable (ctx, TYPE_MIN_VALUE (type));
omp_firstprivatize_variable (ctx, TYPE_MAX_VALUE (type));
break;
case ARRAY_TYPE:
omp_firstprivatize_type_sizes (ctx, TREE_TYPE (type));
omp_firstprivatize_type_sizes (ctx, TYPE_DOMAIN (type));
break;
case RECORD_TYPE:
case UNION_TYPE:
case QUAL_UNION_TYPE:
{
tree field;
for (field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
if (TREE_CODE (field) == FIELD_DECL)
{
omp_firstprivatize_variable (ctx, DECL_FIELD_OFFSET (field));
omp_firstprivatize_type_sizes (ctx, TREE_TYPE (field));
}
}
break;
case POINTER_TYPE:
case REFERENCE_TYPE:
omp_firstprivatize_type_sizes (ctx, TREE_TYPE (type));
break;
default:
break;
}
omp_firstprivatize_variable (ctx, TYPE_SIZE (type));
omp_firstprivatize_variable (ctx, TYPE_SIZE_UNIT (type));
lang_hooks.types.omp_firstprivatize_type_sizes (ctx, type);
}
static void
omp_add_variable (struct gimplify_omp_ctx *ctx, tree decl, unsigned int flags)
{
splay_tree_node n;
unsigned int nflags;
tree t;
if (error_operand_p (decl) || ctx->region_type == ORT_NONE)
return;
if ((flags & GOVD_SHARED) == 0
&& (TREE_ADDRESSABLE (TREE_TYPE (decl))
|| TYPE_NEEDS_CONSTRUCTING (TREE_TYPE (decl))))
flags |= GOVD_SEEN;
n = splay_tree_lookup (ctx->variables, (splay_tree_key)decl);
if (n != NULL && (n->value & GOVD_DATA_SHARE_CLASS) != 0)
{
gcc_assert ((n->value & GOVD_DATA_SHARE_CLASS & flags) == 0);
nflags = n->value | flags;
gcc_assert ((ctx->region_type & ORT_ACC) != 0
|| ((nflags & GOVD_DATA_SHARE_CLASS)
== (GOVD_FIRSTPRIVATE | GOVD_LASTPRIVATE))
|| (flags & GOVD_DATA_SHARE_CLASS) == 0);
n->value = nflags;
return;
}
if (DECL_SIZE (decl) && TREE_CODE (DECL_SIZE (decl)) != INTEGER_CST)
{
if (!(flags & GOVD_LOCAL))
{
if (flags & GOVD_MAP)
nflags = GOVD_MAP | GOVD_MAP_TO_ONLY | GOVD_EXPLICIT;
else if (flags & GOVD_PRIVATE)
nflags = GOVD_PRIVATE;
else if ((ctx->region_type & (ORT_TARGET | ORT_TARGET_DATA)) != 0
&& (flags & GOVD_FIRSTPRIVATE))
nflags = GOVD_PRIVATE | GOVD_EXPLICIT;
else
nflags = GOVD_FIRSTPRIVATE;
nflags |= flags & GOVD_SEEN;
t = DECL_VALUE_EXPR (decl);
gcc_assert (TREE_CODE (t) == INDIRECT_REF);
t = TREE_OPERAND (t, 0);
gcc_assert (DECL_P (t));
omp_add_variable (ctx, t, nflags);
}
omp_firstprivatize_variable (ctx, DECL_SIZE_UNIT (decl));
omp_firstprivatize_variable (ctx, DECL_SIZE (decl));
omp_firstprivatize_type_sizes (ctx, TREE_TYPE (decl));
if (flags & GOVD_SHARED)
flags = GOVD_SHARED | GOVD_DEBUG_PRIVATE
| (flags & (GOVD_SEEN | GOVD_EXPLICIT));
else if (! (flags & (GOVD_LOCAL | GOVD_MAP))
&& DECL_P (TYPE_SIZE_UNIT (TREE_TYPE (decl))))
omp_notice_variable (ctx, TYPE_SIZE_UNIT (TREE_TYPE (decl)), true);
}
else if ((flags & (GOVD_MAP | GOVD_LOCAL)) == 0
&& lang_hooks.decls.omp_privatize_by_reference (decl))
{
omp_firstprivatize_type_sizes (ctx, TREE_TYPE (decl));
if ((flags & GOVD_SHARED) == 0)
{
t = TYPE_SIZE_UNIT (TREE_TYPE (TREE_TYPE (decl)));
if (DECL_P (t))
omp_notice_variable (ctx, t, true);
}
}
if (n != NULL)
n->value |= flags;
else
splay_tree_insert (ctx->variables, (splay_tree_key)decl, flags);
if (ctx->region_type == ORT_ACC && (flags & GOVD_REDUCTION))
{
struct gimplify_omp_ctx *outer_ctx = ctx->outer_context;
while (outer_ctx)
{
n = splay_tree_lookup (outer_ctx->variables, (splay_tree_key)decl);
if (n != NULL)
{
if (n->value & (GOVD_LOCAL | GOVD_EXPLICIT))
break;
else if (outer_ctx->region_type == ORT_ACC_KERNELS)
{
gcc_assert (!(n->value & GOVD_FIRSTPRIVATE)
&& (n->value & GOVD_MAP));
}
else if (outer_ctx->region_type == ORT_ACC_PARALLEL)
{
n->value &= ~GOVD_FIRSTPRIVATE;
n->value |= GOVD_MAP;
}
}
else if (outer_ctx->region_type == ORT_ACC_PARALLEL)
{
splay_tree_insert (outer_ctx->variables, (splay_tree_key)decl,
GOVD_MAP | GOVD_SEEN);
break;
}
outer_ctx = outer_ctx->outer_context;
}
}
}
static bool
omp_notice_threadprivate_variable (struct gimplify_omp_ctx *ctx, tree decl,
tree decl2)
{
splay_tree_node n;
struct gimplify_omp_ctx *octx;
for (octx = ctx; octx; octx = octx->outer_context)
if ((octx->region_type & ORT_TARGET) != 0)
{
n = splay_tree_lookup (octx->variables, (splay_tree_key)decl);
if (n == NULL)
{
error ("threadprivate variable %qE used in target region",
DECL_NAME (decl));
error_at (octx->location, "enclosing target region");
splay_tree_insert (octx->variables, (splay_tree_key)decl, 0);
}
if (decl2)
splay_tree_insert (octx->variables, (splay_tree_key)decl2, 0);
}
if (ctx->region_type != ORT_UNTIED_TASK)
return false;
n = splay_tree_lookup (ctx->variables, (splay_tree_key)decl);
if (n == NULL)
{
error ("threadprivate variable %qE used in untied task",
DECL_NAME (decl));
error_at (ctx->location, "enclosing task");
splay_tree_insert (ctx->variables, (splay_tree_key)decl, 0);
}
if (decl2)
splay_tree_insert (ctx->variables, (splay_tree_key)decl2, 0);
return false;
}
static bool
device_resident_p (tree decl)
{
tree attr = lookup_attribute ("oacc declare target", DECL_ATTRIBUTES (decl));
if (!attr)
return false;
for (tree t = TREE_VALUE (attr); t; t = TREE_PURPOSE (t))
{
tree c = TREE_VALUE (t);
if (OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_DEVICE_RESIDENT)
return true;
}
return false;
}
static bool
is_oacc_declared (tree decl)
{
tree t = TREE_CODE (decl) == MEM_REF ? TREE_OPERAND (decl, 0) : decl;
tree declared = lookup_attribute ("oacc declare target", DECL_ATTRIBUTES (t));
return declared != NULL_TREE;
}
static unsigned
omp_default_clause (struct gimplify_omp_ctx *ctx, tree decl,
bool in_code, unsigned flags)
{
enum omp_clause_default_kind default_kind = ctx->default_kind;
enum omp_clause_default_kind kind;
kind = lang_hooks.decls.omp_predetermined_sharing (decl);
if (kind != OMP_CLAUSE_DEFAULT_UNSPECIFIED)
default_kind = kind;
switch (default_kind)
{
case OMP_CLAUSE_DEFAULT_NONE:
{
const char *rtype;
if (ctx->region_type & ORT_PARALLEL)
rtype = "parallel";
else if (ctx->region_type & ORT_TASK)
rtype = "task";
else if (ctx->region_type & ORT_TEAMS)
rtype = "teams";
else
gcc_unreachable ();
error ("%qE not specified in enclosing %qs",
DECL_NAME (lang_hooks.decls.omp_report_decl (decl)), rtype);
error_at (ctx->location, "enclosing %qs", rtype);
}
case OMP_CLAUSE_DEFAULT_SHARED:
flags |= GOVD_SHARED;
break;
case OMP_CLAUSE_DEFAULT_PRIVATE:
flags |= GOVD_PRIVATE;
break;
case OMP_CLAUSE_DEFAULT_FIRSTPRIVATE:
flags |= GOVD_FIRSTPRIVATE;
break;
case OMP_CLAUSE_DEFAULT_UNSPECIFIED:
gcc_assert ((ctx->region_type & ORT_TASK) != 0);
if (struct gimplify_omp_ctx *octx = ctx->outer_context)
{
omp_notice_variable (octx, decl, in_code);
for (; octx; octx = octx->outer_context)
{
splay_tree_node n2;
n2 = splay_tree_lookup (octx->variables, (splay_tree_key) decl);
if ((octx->region_type & (ORT_TARGET_DATA | ORT_TARGET)) != 0
&& (n2 == NULL || (n2->value & GOVD_DATA_SHARE_CLASS) == 0))
continue;
if (n2 && (n2->value & GOVD_DATA_SHARE_CLASS) != GOVD_SHARED)
{
flags |= GOVD_FIRSTPRIVATE;
goto found_outer;
}
if ((octx->region_type & (ORT_PARALLEL | ORT_TEAMS)) != 0)
{
flags |= GOVD_SHARED;
goto found_outer;
}
}
}
if (TREE_CODE (decl) == PARM_DECL
|| (!is_global_var (decl)
&& DECL_CONTEXT (decl) == current_function_decl))
flags |= GOVD_FIRSTPRIVATE;
else
flags |= GOVD_SHARED;
found_outer:
break;
default:
gcc_unreachable ();
}
return flags;
}
static unsigned
oacc_default_clause (struct gimplify_omp_ctx *ctx, tree decl, unsigned flags)
{
const char *rkind;
bool on_device = false;
bool declared = is_oacc_declared (decl);
tree type = TREE_TYPE (decl);
if (lang_hooks.decls.omp_privatize_by_reference (decl))
type = TREE_TYPE (type);
if ((ctx->region_type & (ORT_ACC_PARALLEL | ORT_ACC_KERNELS)) != 0
&& is_global_var (decl)
&& device_resident_p (decl))
{
on_device = true;
flags |= GOVD_MAP_TO_ONLY;
}
switch (ctx->region_type)
{
case ORT_ACC_KERNELS:
rkind = "kernels";
if (AGGREGATE_TYPE_P (type))
{
if (ctx->default_kind != OMP_CLAUSE_DEFAULT_PRESENT)
flags |= GOVD_MAP;
else
flags |= GOVD_MAP | GOVD_MAP_FORCE_PRESENT;
}
else
flags |= GOVD_MAP | GOVD_MAP_FORCE;
break;
case ORT_ACC_PARALLEL:
rkind = "parallel";
if (on_device || declared)
flags |= GOVD_MAP;
else if (AGGREGATE_TYPE_P (type))
{
if (ctx->default_kind != OMP_CLAUSE_DEFAULT_PRESENT)
flags |= GOVD_MAP;
else
flags |= GOVD_MAP | GOVD_MAP_FORCE_PRESENT;
}
else
flags |= GOVD_FIRSTPRIVATE;
break;
default:
gcc_unreachable ();
}
if (DECL_ARTIFICIAL (decl))
; 
else if (ctx->default_kind == OMP_CLAUSE_DEFAULT_NONE)
{
error ("%qE not specified in enclosing OpenACC %qs construct",
DECL_NAME (lang_hooks.decls.omp_report_decl (decl)), rkind);
inform (ctx->location, "enclosing OpenACC %qs construct", rkind);
}
else if (ctx->default_kind == OMP_CLAUSE_DEFAULT_PRESENT)
; 
else
gcc_checking_assert (ctx->default_kind == OMP_CLAUSE_DEFAULT_SHARED);
return flags;
}
static bool
omp_notice_variable (struct gimplify_omp_ctx *ctx, tree decl, bool in_code)
{
splay_tree_node n;
unsigned flags = in_code ? GOVD_SEEN : 0;
bool ret = false, shared;
if (error_operand_p (decl))
return false;
if (ctx->region_type == ORT_NONE)
return lang_hooks.decls.omp_disregard_value_expr (decl, false);
if (is_global_var (decl))
{
if (DECL_THREAD_LOCAL_P (decl))
return omp_notice_threadprivate_variable (ctx, decl, NULL_TREE);
if (DECL_HAS_VALUE_EXPR_P (decl))
{
tree value = get_base_address (DECL_VALUE_EXPR (decl));
if (value && DECL_P (value) && DECL_THREAD_LOCAL_P (value))
return omp_notice_threadprivate_variable (ctx, decl, value);
}
if (gimplify_omp_ctxp->outer_context == NULL
&& VAR_P (decl)
&& oacc_get_fn_attrib (current_function_decl))
{
location_t loc = DECL_SOURCE_LOCATION (decl);
if (lookup_attribute ("omp declare target link",
DECL_ATTRIBUTES (decl)))
{
error_at (loc,
"%qE with %<link%> clause used in %<routine%> function",
DECL_NAME (decl));
return false;
}
else if (!lookup_attribute ("omp declare target",
DECL_ATTRIBUTES (decl)))
{
error_at (loc,
"%qE requires a %<declare%> directive for use "
"in a %<routine%> function", DECL_NAME (decl));
return false;
}
}
}
n = splay_tree_lookup (ctx->variables, (splay_tree_key)decl);
if ((ctx->region_type & ORT_TARGET) != 0)
{
ret = lang_hooks.decls.omp_disregard_value_expr (decl, true);
if (n == NULL)
{
unsigned nflags = flags;
if (ctx->target_map_pointers_as_0len_arrays
|| ctx->target_map_scalars_firstprivate)
{
bool is_declare_target = false;
bool is_scalar = false;
if (is_global_var (decl)
&& varpool_node::get_create (decl)->offloadable)
{
struct gimplify_omp_ctx *octx;
for (octx = ctx->outer_context;
octx; octx = octx->outer_context)
{
n = splay_tree_lookup (octx->variables,
(splay_tree_key)decl);
if (n
&& (n->value & GOVD_DATA_SHARE_CLASS) != GOVD_SHARED
&& (n->value & GOVD_DATA_SHARE_CLASS) != 0)
break;
}
is_declare_target = octx == NULL;
}
if (!is_declare_target && ctx->target_map_scalars_firstprivate)
is_scalar = lang_hooks.decls.omp_scalar_p (decl);
if (is_declare_target)
;
else if (ctx->target_map_pointers_as_0len_arrays
&& (TREE_CODE (TREE_TYPE (decl)) == POINTER_TYPE
|| (TREE_CODE (TREE_TYPE (decl)) == REFERENCE_TYPE
&& TREE_CODE (TREE_TYPE (TREE_TYPE (decl)))
== POINTER_TYPE)))
nflags |= GOVD_MAP | GOVD_MAP_0LEN_ARRAY;
else if (is_scalar)
nflags |= GOVD_FIRSTPRIVATE;
}
struct gimplify_omp_ctx *octx = ctx->outer_context;
if ((ctx->region_type & ORT_ACC) && octx)
{
omp_notice_variable (octx, decl, in_code);
for (; octx; octx = octx->outer_context)
{
if (!(octx->region_type & (ORT_TARGET_DATA | ORT_TARGET)))
break;
splay_tree_node n2
= splay_tree_lookup (octx->variables,
(splay_tree_key) decl);
if (n2)
{
if (octx->region_type == ORT_ACC_HOST_DATA)
error ("variable %qE declared in enclosing "
"%<host_data%> region", DECL_NAME (decl));
nflags |= GOVD_MAP;
if (octx->region_type == ORT_ACC_DATA
&& (n2->value & GOVD_MAP_0LEN_ARRAY))
nflags |= GOVD_MAP_0LEN_ARRAY;
goto found_outer;
}
}
}
{
tree type = TREE_TYPE (decl);
if (nflags == flags
&& gimplify_omp_ctxp->target_firstprivatize_array_bases
&& lang_hooks.decls.omp_privatize_by_reference (decl))
type = TREE_TYPE (type);
if (nflags == flags
&& !lang_hooks.types.omp_mappable_type (type))
{
error ("%qD referenced in target region does not have "
"a mappable type", decl);
nflags |= GOVD_MAP | GOVD_EXPLICIT;
}
else if (nflags == flags)
{
if ((ctx->region_type & ORT_ACC) != 0)
nflags = oacc_default_clause (ctx, decl, flags);
else
nflags |= GOVD_MAP;
}
}
found_outer:
omp_add_variable (ctx, decl, nflags);
}
else
{
if ((n->value & flags) == flags)
return ret;
flags |= n->value;
n->value = flags;
}
goto do_outer;
}
if (n == NULL)
{
if (ctx->region_type == ORT_WORKSHARE
|| ctx->region_type == ORT_SIMD
|| ctx->region_type == ORT_ACC
|| (ctx->region_type & ORT_TARGET_DATA) != 0)
goto do_outer;
flags = omp_default_clause (ctx, decl, in_code, flags);
if ((flags & GOVD_PRIVATE)
&& lang_hooks.decls.omp_private_outer_ref (decl))
flags |= GOVD_PRIVATE_OUTER_REF;
omp_add_variable (ctx, decl, flags);
shared = (flags & GOVD_SHARED) != 0;
ret = lang_hooks.decls.omp_disregard_value_expr (decl, shared);
goto do_outer;
}
if ((n->value & (GOVD_SEEN | GOVD_LOCAL)) == 0
&& (flags & (GOVD_SEEN | GOVD_LOCAL)) == GOVD_SEEN
&& DECL_SIZE (decl))
{
if (TREE_CODE (DECL_SIZE (decl)) != INTEGER_CST)
{
splay_tree_node n2;
tree t = DECL_VALUE_EXPR (decl);
gcc_assert (TREE_CODE (t) == INDIRECT_REF);
t = TREE_OPERAND (t, 0);
gcc_assert (DECL_P (t));
n2 = splay_tree_lookup (ctx->variables, (splay_tree_key) t);
n2->value |= GOVD_SEEN;
}
else if (lang_hooks.decls.omp_privatize_by_reference (decl)
&& TYPE_SIZE_UNIT (TREE_TYPE (TREE_TYPE (decl)))
&& (TREE_CODE (TYPE_SIZE_UNIT (TREE_TYPE (TREE_TYPE (decl))))
!= INTEGER_CST))
{
splay_tree_node n2;
tree t = TYPE_SIZE_UNIT (TREE_TYPE (TREE_TYPE (decl)));
gcc_assert (DECL_P (t));
n2 = splay_tree_lookup (ctx->variables, (splay_tree_key) t);
if (n2)
omp_notice_variable (ctx, t, true);
}
}
shared = ((flags | n->value) & GOVD_SHARED) != 0;
ret = lang_hooks.decls.omp_disregard_value_expr (decl, shared);
if ((n->value & flags) == flags)
return ret;
flags |= n->value;
n->value = flags;
do_outer:
if ((flags & GOVD_PRIVATE) && !(flags & GOVD_PRIVATE_OUTER_REF))
return ret;
if ((flags & (GOVD_LINEAR | GOVD_LINEAR_LASTPRIVATE_NO_OUTER))
== (GOVD_LINEAR | GOVD_LINEAR_LASTPRIVATE_NO_OUTER))
return ret;
if ((flags & (GOVD_FIRSTPRIVATE | GOVD_LASTPRIVATE
| GOVD_LINEAR_LASTPRIVATE_NO_OUTER))
== (GOVD_LASTPRIVATE | GOVD_LINEAR_LASTPRIVATE_NO_OUTER))
return ret;
if (ctx->outer_context
&& omp_notice_variable (ctx->outer_context, decl, in_code))
return true;
return ret;
}
static bool
omp_is_private (struct gimplify_omp_ctx *ctx, tree decl, int simd)
{
splay_tree_node n;
n = splay_tree_lookup (ctx->variables, (splay_tree_key)decl);
if (n != NULL)
{
if (n->value & GOVD_SHARED)
{
if (ctx == gimplify_omp_ctxp)
{
if (simd)
error ("iteration variable %qE is predetermined linear",
DECL_NAME (decl));
else
error ("iteration variable %qE should be private",
DECL_NAME (decl));
n->value = GOVD_PRIVATE;
return true;
}
else
return false;
}
else if ((n->value & GOVD_EXPLICIT) != 0
&& (ctx == gimplify_omp_ctxp
|| (ctx->region_type == ORT_COMBINED_PARALLEL
&& gimplify_omp_ctxp->outer_context == ctx)))
{
if ((n->value & GOVD_FIRSTPRIVATE) != 0)
error ("iteration variable %qE should not be firstprivate",
DECL_NAME (decl));
else if ((n->value & GOVD_REDUCTION) != 0)
error ("iteration variable %qE should not be reduction",
DECL_NAME (decl));
else if (simd == 0 && (n->value & GOVD_LINEAR) != 0)
error ("iteration variable %qE should not be linear",
DECL_NAME (decl));
else if (simd == 1 && (n->value & GOVD_LASTPRIVATE) != 0)
error ("iteration variable %qE should not be lastprivate",
DECL_NAME (decl));
else if (simd && (n->value & GOVD_PRIVATE) != 0)
error ("iteration variable %qE should not be private",
DECL_NAME (decl));
else if (simd == 2 && (n->value & GOVD_LINEAR) != 0)
error ("iteration variable %qE is predetermined linear",
DECL_NAME (decl));
}
return (ctx == gimplify_omp_ctxp
|| (ctx->region_type == ORT_COMBINED_PARALLEL
&& gimplify_omp_ctxp->outer_context == ctx));
}
if (ctx->region_type != ORT_WORKSHARE
&& ctx->region_type != ORT_SIMD
&& ctx->region_type != ORT_ACC)
return false;
else if (ctx->outer_context)
return omp_is_private (ctx->outer_context, decl, simd);
return false;
}
static bool
omp_check_private (struct gimplify_omp_ctx *ctx, tree decl, bool copyprivate)
{
splay_tree_node n;
do
{
ctx = ctx->outer_context;
if (ctx == NULL)
{
if (is_global_var (decl))
return false;
if (copyprivate)
return true;
if (lang_hooks.decls.omp_privatize_by_reference (decl))
return false;
if (omp_member_access_dummy_var (decl))
return false;
return true;
}
n = splay_tree_lookup (ctx->variables, (splay_tree_key) decl);
if ((ctx->region_type & (ORT_TARGET | ORT_TARGET_DATA)) != 0
&& (n == NULL || (n->value & GOVD_DATA_SHARE_CLASS) == 0))
continue;
if (n != NULL)
{
if ((n->value & GOVD_LOCAL) != 0
&& omp_member_access_dummy_var (decl))
return false;
return (n->value & GOVD_SHARED) == 0;
}
}
while (ctx->region_type == ORT_WORKSHARE
|| ctx->region_type == ORT_SIMD
|| ctx->region_type == ORT_ACC);
return false;
}
static tree
find_decl_expr (tree *tp, int *walk_subtrees, void *data)
{
tree t = *tp;
if (TREE_CODE (t) == DECL_EXPR && DECL_EXPR_DECL (t) == (tree) data)
return t;
if (IS_TYPE_OR_DECL_P (t))
*walk_subtrees = 0;
return NULL_TREE;
}
static void
gimplify_scan_omp_clauses (tree *list_p, gimple_seq *pre_p,
enum omp_region_type region_type,
enum tree_code code)
{
struct gimplify_omp_ctx *ctx, *outer_ctx;
tree c;
hash_map<tree, tree> *struct_map_to_clause = NULL;
tree *prev_list_p = NULL;
ctx = new_omp_context (region_type);
outer_ctx = ctx->outer_context;
if (code == OMP_TARGET)
{
if (!lang_GNU_Fortran ())
ctx->target_map_pointers_as_0len_arrays = true;
ctx->target_map_scalars_firstprivate = true;
}
if (!lang_GNU_Fortran ())
switch (code)
{
case OMP_TARGET:
case OMP_TARGET_DATA:
case OMP_TARGET_ENTER_DATA:
case OMP_TARGET_EXIT_DATA:
case OACC_DECLARE:
case OACC_HOST_DATA:
ctx->target_firstprivatize_array_bases = true;
default:
break;
}
while ((c = *list_p) != NULL)
{
bool remove = false;
bool notice_outer = true;
const char *check_non_private = NULL;
unsigned int flags;
tree decl;
switch (OMP_CLAUSE_CODE (c))
{
case OMP_CLAUSE_PRIVATE:
flags = GOVD_PRIVATE | GOVD_EXPLICIT;
if (lang_hooks.decls.omp_private_outer_ref (OMP_CLAUSE_DECL (c)))
{
flags |= GOVD_PRIVATE_OUTER_REF;
OMP_CLAUSE_PRIVATE_OUTER_REF (c) = 1;
}
else
notice_outer = false;
goto do_add;
case OMP_CLAUSE_SHARED:
flags = GOVD_SHARED | GOVD_EXPLICIT;
goto do_add;
case OMP_CLAUSE_FIRSTPRIVATE:
flags = GOVD_FIRSTPRIVATE | GOVD_EXPLICIT;
check_non_private = "firstprivate";
goto do_add;
case OMP_CLAUSE_LASTPRIVATE:
flags = GOVD_LASTPRIVATE | GOVD_SEEN | GOVD_EXPLICIT;
check_non_private = "lastprivate";
decl = OMP_CLAUSE_DECL (c);
if (error_operand_p (decl))
goto do_add;
else if (outer_ctx
&& (outer_ctx->region_type == ORT_COMBINED_PARALLEL
|| outer_ctx->region_type == ORT_COMBINED_TEAMS)
&& splay_tree_lookup (outer_ctx->variables,
(splay_tree_key) decl) == NULL)
{
omp_add_variable (outer_ctx, decl, GOVD_SHARED | GOVD_SEEN);
if (outer_ctx->outer_context)
omp_notice_variable (outer_ctx->outer_context, decl, true);
}
else if (outer_ctx
&& (outer_ctx->region_type & ORT_TASK) != 0
&& outer_ctx->combined_loop
&& splay_tree_lookup (outer_ctx->variables,
(splay_tree_key) decl) == NULL)
{
omp_add_variable (outer_ctx, decl, GOVD_LASTPRIVATE | GOVD_SEEN);
if (outer_ctx->outer_context)
omp_notice_variable (outer_ctx->outer_context, decl, true);
}
else if (outer_ctx
&& (outer_ctx->region_type == ORT_WORKSHARE
|| outer_ctx->region_type == ORT_ACC)
&& outer_ctx->combined_loop
&& splay_tree_lookup (outer_ctx->variables,
(splay_tree_key) decl) == NULL
&& !omp_check_private (outer_ctx, decl, false))
{
omp_add_variable (outer_ctx, decl, GOVD_LASTPRIVATE | GOVD_SEEN);
if (outer_ctx->outer_context
&& (outer_ctx->outer_context->region_type
== ORT_COMBINED_PARALLEL)
&& splay_tree_lookup (outer_ctx->outer_context->variables,
(splay_tree_key) decl) == NULL)
{
struct gimplify_omp_ctx *octx = outer_ctx->outer_context;
omp_add_variable (octx, decl, GOVD_SHARED | GOVD_SEEN);
if (octx->outer_context)
{
octx = octx->outer_context;
if (octx->region_type == ORT_WORKSHARE
&& octx->combined_loop
&& splay_tree_lookup (octx->variables,
(splay_tree_key) decl) == NULL
&& !omp_check_private (octx, decl, false))
{
omp_add_variable (octx, decl,
GOVD_LASTPRIVATE | GOVD_SEEN);
octx = octx->outer_context;
if (octx
&& octx->region_type == ORT_COMBINED_TEAMS
&& (splay_tree_lookup (octx->variables,
(splay_tree_key) decl)
== NULL))
{
omp_add_variable (octx, decl,
GOVD_SHARED | GOVD_SEEN);
octx = octx->outer_context;
}
}
if (octx)
omp_notice_variable (octx, decl, true);
}
}
else if (outer_ctx->outer_context)
omp_notice_variable (outer_ctx->outer_context, decl, true);
}
goto do_add;
case OMP_CLAUSE_REDUCTION:
flags = GOVD_REDUCTION | GOVD_SEEN | GOVD_EXPLICIT;
if (!(region_type & ORT_ACC))
check_non_private = "reduction";
decl = OMP_CLAUSE_DECL (c);
if (TREE_CODE (decl) == MEM_REF)
{
tree type = TREE_TYPE (decl);
if (gimplify_expr (&TYPE_MAX_VALUE (TYPE_DOMAIN (type)), pre_p,
NULL, is_gimple_val, fb_rvalue, false)
== GS_ERROR)
{
remove = true;
break;
}
tree v = TYPE_MAX_VALUE (TYPE_DOMAIN (type));
if (DECL_P (v))
{
omp_firstprivatize_variable (ctx, v);
omp_notice_variable (ctx, v, true);
}
decl = TREE_OPERAND (decl, 0);
if (TREE_CODE (decl) == POINTER_PLUS_EXPR)
{
if (gimplify_expr (&TREE_OPERAND (decl, 1), pre_p,
NULL, is_gimple_val, fb_rvalue, false)
== GS_ERROR)
{
remove = true;
break;
}
v = TREE_OPERAND (decl, 1);
if (DECL_P (v))
{
omp_firstprivatize_variable (ctx, v);
omp_notice_variable (ctx, v, true);
}
decl = TREE_OPERAND (decl, 0);
}
if (TREE_CODE (decl) == ADDR_EXPR
|| TREE_CODE (decl) == INDIRECT_REF)
decl = TREE_OPERAND (decl, 0);
}
goto do_add_decl;
case OMP_CLAUSE_LINEAR:
if (gimplify_expr (&OMP_CLAUSE_LINEAR_STEP (c), pre_p, NULL,
is_gimple_val, fb_rvalue) == GS_ERROR)
{
remove = true;
break;
}
else
{
if (code == OMP_SIMD
&& !OMP_CLAUSE_LINEAR_NO_COPYIN (c))
{
struct gimplify_omp_ctx *octx = outer_ctx;
if (octx
&& octx->region_type == ORT_WORKSHARE
&& octx->combined_loop
&& !octx->distribute)
{
if (octx->outer_context
&& (octx->outer_context->region_type
== ORT_COMBINED_PARALLEL))
octx = octx->outer_context->outer_context;
else
octx = octx->outer_context;
}
if (octx
&& octx->region_type == ORT_WORKSHARE
&& octx->combined_loop
&& octx->distribute)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<linear%> clause for variable other than "
"loop iterator specified on construct "
"combined with %<distribute%>");
remove = true;
break;
}
}
struct gimplify_omp_ctx *octx = outer_ctx;
decl = NULL_TREE;
do
{
if (OMP_CLAUSE_LINEAR_NO_COPYIN (c)
&& OMP_CLAUSE_LINEAR_NO_COPYOUT (c))
break;
decl = OMP_CLAUSE_DECL (c);
if (error_operand_p (decl))
{
decl = NULL_TREE;
break;
}
flags = GOVD_SEEN;
if (!OMP_CLAUSE_LINEAR_NO_COPYIN (c))
flags |= GOVD_FIRSTPRIVATE;
if (!OMP_CLAUSE_LINEAR_NO_COPYOUT (c))
flags |= GOVD_LASTPRIVATE;
if (octx
&& octx->region_type == ORT_WORKSHARE
&& octx->combined_loop)
{
if (octx->outer_context
&& (octx->outer_context->region_type
== ORT_COMBINED_PARALLEL))
octx = octx->outer_context;
else if (omp_check_private (octx, decl, false))
break;
}
else if (octx
&& (octx->region_type & ORT_TASK) != 0
&& octx->combined_loop)
;
else if (octx
&& octx->region_type == ORT_COMBINED_PARALLEL
&& ctx->region_type == ORT_WORKSHARE
&& octx == outer_ctx)
flags = GOVD_SEEN | GOVD_SHARED;
else if (octx
&& octx->region_type == ORT_COMBINED_TEAMS)
flags = GOVD_SEEN | GOVD_SHARED;
else if (octx
&& octx->region_type == ORT_COMBINED_TARGET)
{
flags &= ~GOVD_LASTPRIVATE;
if (flags == GOVD_SEEN)
break;
}
else
break;
splay_tree_node on
= splay_tree_lookup (octx->variables,
(splay_tree_key) decl);
if (on && (on->value & GOVD_DATA_SHARE_CLASS) != 0)
{
octx = NULL;
break;
}
omp_add_variable (octx, decl, flags);
if (octx->outer_context == NULL)
break;
octx = octx->outer_context;
}
while (1);
if (octx
&& decl
&& (!OMP_CLAUSE_LINEAR_NO_COPYIN (c)
|| !OMP_CLAUSE_LINEAR_NO_COPYOUT (c)))
omp_notice_variable (octx, decl, true);
}
flags = GOVD_LINEAR | GOVD_EXPLICIT;
if (OMP_CLAUSE_LINEAR_NO_COPYIN (c)
&& OMP_CLAUSE_LINEAR_NO_COPYOUT (c))
{
notice_outer = false;
flags |= GOVD_LINEAR_LASTPRIVATE_NO_OUTER;
}
goto do_add;
case OMP_CLAUSE_MAP:
decl = OMP_CLAUSE_DECL (c);
if (error_operand_p (decl))
remove = true;
switch (code)
{
case OMP_TARGET:
break;
case OACC_DATA:
if (TREE_CODE (TREE_TYPE (decl)) != ARRAY_TYPE)
break;
case OMP_TARGET_DATA:
case OMP_TARGET_ENTER_DATA:
case OMP_TARGET_EXIT_DATA:
case OACC_ENTER_DATA:
case OACC_EXIT_DATA:
case OACC_HOST_DATA:
if (OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_FIRSTPRIVATE_POINTER
|| (OMP_CLAUSE_MAP_KIND (c)
== GOMP_MAP_FIRSTPRIVATE_REFERENCE))
remove = true;
break;
default:
break;
}
if (remove)
break;
if (DECL_P (decl) && outer_ctx && (region_type & ORT_ACC))
{
struct gimplify_omp_ctx *octx;
for (octx = outer_ctx; octx; octx = octx->outer_context)
{
if (octx->region_type != ORT_ACC_HOST_DATA)
break;
splay_tree_node n2
= splay_tree_lookup (octx->variables,
(splay_tree_key) decl);
if (n2)
error_at (OMP_CLAUSE_LOCATION (c), "variable %qE "
"declared in enclosing %<host_data%> region",
DECL_NAME (decl));
}
}
if (OMP_CLAUSE_SIZE (c) == NULL_TREE)
OMP_CLAUSE_SIZE (c) = DECL_P (decl) ? DECL_SIZE_UNIT (decl)
: TYPE_SIZE_UNIT (TREE_TYPE (decl));
if (gimplify_expr (&OMP_CLAUSE_SIZE (c), pre_p,
NULL, is_gimple_val, fb_rvalue) == GS_ERROR)
{
remove = true;
break;
}
else if ((OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_FIRSTPRIVATE_POINTER
|| (OMP_CLAUSE_MAP_KIND (c)
== GOMP_MAP_FIRSTPRIVATE_REFERENCE))
&& TREE_CODE (OMP_CLAUSE_SIZE (c)) != INTEGER_CST)
{
OMP_CLAUSE_SIZE (c)
= get_initialized_tmp_var (OMP_CLAUSE_SIZE (c), pre_p, NULL,
false);
omp_add_variable (ctx, OMP_CLAUSE_SIZE (c),
GOVD_FIRSTPRIVATE | GOVD_SEEN);
}
if (!DECL_P (decl))
{
tree d = decl, *pd;
if (TREE_CODE (d) == ARRAY_REF)
{
while (TREE_CODE (d) == ARRAY_REF)
d = TREE_OPERAND (d, 0);
if (TREE_CODE (d) == COMPONENT_REF
&& TREE_CODE (TREE_TYPE (d)) == ARRAY_TYPE)
decl = d;
}
pd = &OMP_CLAUSE_DECL (c);
if (d == decl
&& TREE_CODE (decl) == INDIRECT_REF
&& TREE_CODE (TREE_OPERAND (decl, 0)) == COMPONENT_REF
&& (TREE_CODE (TREE_TYPE (TREE_OPERAND (decl, 0)))
== REFERENCE_TYPE))
{
pd = &TREE_OPERAND (decl, 0);
decl = TREE_OPERAND (decl, 0);
}
if (TREE_CODE (decl) == COMPONENT_REF)
{
while (TREE_CODE (decl) == COMPONENT_REF)
decl = TREE_OPERAND (decl, 0);
if (TREE_CODE (decl) == INDIRECT_REF
&& DECL_P (TREE_OPERAND (decl, 0))
&& (TREE_CODE (TREE_TYPE (TREE_OPERAND (decl, 0)))
== REFERENCE_TYPE))
decl = TREE_OPERAND (decl, 0);
}
if (gimplify_expr (pd, pre_p, NULL, is_gimple_lvalue, fb_lvalue)
== GS_ERROR)
{
remove = true;
break;
}
if (DECL_P (decl))
{
if (error_operand_p (decl))
{
remove = true;
break;
}
tree stype = TREE_TYPE (decl);
if (TREE_CODE (stype) == REFERENCE_TYPE)
stype = TREE_TYPE (stype);
if (TYPE_SIZE_UNIT (stype) == NULL
|| TREE_CODE (TYPE_SIZE_UNIT (stype)) != INTEGER_CST)
{
error_at (OMP_CLAUSE_LOCATION (c),
"mapping field %qE of variable length "
"structure", OMP_CLAUSE_DECL (c));
remove = true;
break;
}
if (OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_ALWAYS_POINTER)
{
if (prev_list_p == NULL)
{
remove = true;
break;
}
if (OMP_CLAUSE_CHAIN (*prev_list_p) != c)
{
tree ch = OMP_CLAUSE_CHAIN (*prev_list_p);
if (ch == NULL_TREE || OMP_CLAUSE_CHAIN (ch) != c)
{
remove = true;
break;
}
}
}
tree offset;
poly_int64 bitsize, bitpos;
machine_mode mode;
int unsignedp, reversep, volatilep = 0;
tree base = OMP_CLAUSE_DECL (c);
while (TREE_CODE (base) == ARRAY_REF)
base = TREE_OPERAND (base, 0);
if (TREE_CODE (base) == INDIRECT_REF)
base = TREE_OPERAND (base, 0);
base = get_inner_reference (base, &bitsize, &bitpos, &offset,
&mode, &unsignedp, &reversep,
&volatilep);
tree orig_base = base;
if ((TREE_CODE (base) == INDIRECT_REF
|| (TREE_CODE (base) == MEM_REF
&& integer_zerop (TREE_OPERAND (base, 1))))
&& DECL_P (TREE_OPERAND (base, 0))
&& (TREE_CODE (TREE_TYPE (TREE_OPERAND (base, 0)))
== REFERENCE_TYPE))
base = TREE_OPERAND (base, 0);
gcc_assert (base == decl
&& (offset == NULL_TREE
|| poly_int_tree_p (offset)));
splay_tree_node n
= splay_tree_lookup (ctx->variables, (splay_tree_key)decl);
bool ptr = (OMP_CLAUSE_MAP_KIND (c)
== GOMP_MAP_ALWAYS_POINTER);
if (n == NULL || (n->value & GOVD_MAP) == 0)
{
tree l = build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_MAP);
OMP_CLAUSE_SET_MAP_KIND (l, GOMP_MAP_STRUCT);
if (orig_base != base)
OMP_CLAUSE_DECL (l) = unshare_expr (orig_base);
else
OMP_CLAUSE_DECL (l) = decl;
OMP_CLAUSE_SIZE (l) = size_int (1);
if (struct_map_to_clause == NULL)
struct_map_to_clause = new hash_map<tree, tree>;
struct_map_to_clause->put (decl, l);
if (ptr)
{
enum gomp_map_kind mkind
= code == OMP_TARGET_EXIT_DATA
? GOMP_MAP_RELEASE : GOMP_MAP_ALLOC;
tree c2 = build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_MAP);
OMP_CLAUSE_SET_MAP_KIND (c2, mkind);
OMP_CLAUSE_DECL (c2)
= unshare_expr (OMP_CLAUSE_DECL (c));
OMP_CLAUSE_CHAIN (c2) = *prev_list_p;
OMP_CLAUSE_SIZE (c2)
= TYPE_SIZE_UNIT (ptr_type_node);
OMP_CLAUSE_CHAIN (l) = c2;
if (OMP_CLAUSE_CHAIN (*prev_list_p) != c)
{
tree c4 = OMP_CLAUSE_CHAIN (*prev_list_p);
tree c3
= build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_MAP);
OMP_CLAUSE_SET_MAP_KIND (c3, mkind);
OMP_CLAUSE_DECL (c3)
= unshare_expr (OMP_CLAUSE_DECL (c4));
OMP_CLAUSE_SIZE (c3)
= TYPE_SIZE_UNIT (ptr_type_node);
OMP_CLAUSE_CHAIN (c3) = *prev_list_p;
OMP_CLAUSE_CHAIN (c2) = c3;
}
*prev_list_p = l;
prev_list_p = NULL;
}
else
{
OMP_CLAUSE_CHAIN (l) = c;
*list_p = l;
list_p = &OMP_CLAUSE_CHAIN (l);
}
if (orig_base != base && code == OMP_TARGET)
{
tree c2 = build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_MAP);
enum gomp_map_kind mkind
= GOMP_MAP_FIRSTPRIVATE_REFERENCE;
OMP_CLAUSE_SET_MAP_KIND (c2, mkind);
OMP_CLAUSE_DECL (c2) = decl;
OMP_CLAUSE_SIZE (c2) = size_zero_node;
OMP_CLAUSE_CHAIN (c2) = OMP_CLAUSE_CHAIN (l);
OMP_CLAUSE_CHAIN (l) = c2;
}
flags = GOVD_MAP | GOVD_EXPLICIT;
if (GOMP_MAP_ALWAYS_P (OMP_CLAUSE_MAP_KIND (c)) || ptr)
flags |= GOVD_SEEN;
goto do_add_decl;
}
else
{
tree *osc = struct_map_to_clause->get (decl);
tree *sc = NULL, *scp = NULL;
if (GOMP_MAP_ALWAYS_P (OMP_CLAUSE_MAP_KIND (c)) || ptr)
n->value |= GOVD_SEEN;
poly_offset_int o1, o2;
if (offset)
o1 = wi::to_poly_offset (offset);
else
o1 = 0;
if (maybe_ne (bitpos, 0))
o1 += bits_to_bytes_round_down (bitpos);
sc = &OMP_CLAUSE_CHAIN (*osc);
if (*sc != c
&& (OMP_CLAUSE_MAP_KIND (*sc)
== GOMP_MAP_FIRSTPRIVATE_REFERENCE)) 
sc = &OMP_CLAUSE_CHAIN (*sc);
for (; *sc != c; sc = &OMP_CLAUSE_CHAIN (*sc))
if (ptr && sc == prev_list_p)
break;
else if (TREE_CODE (OMP_CLAUSE_DECL (*sc))
!= COMPONENT_REF
&& (TREE_CODE (OMP_CLAUSE_DECL (*sc))
!= INDIRECT_REF)
&& (TREE_CODE (OMP_CLAUSE_DECL (*sc))
!= ARRAY_REF))
break;
else
{
tree offset2;
poly_int64 bitsize2, bitpos2;
base = OMP_CLAUSE_DECL (*sc);
if (TREE_CODE (base) == ARRAY_REF)
{
while (TREE_CODE (base) == ARRAY_REF)
base = TREE_OPERAND (base, 0);
if (TREE_CODE (base) != COMPONENT_REF
|| (TREE_CODE (TREE_TYPE (base))
!= ARRAY_TYPE))
break;
}
else if (TREE_CODE (base) == INDIRECT_REF
&& (TREE_CODE (TREE_OPERAND (base, 0))
== COMPONENT_REF)
&& (TREE_CODE (TREE_TYPE
(TREE_OPERAND (base, 0)))
== REFERENCE_TYPE))
base = TREE_OPERAND (base, 0);
base = get_inner_reference (base, &bitsize2,
&bitpos2, &offset2,
&mode, &unsignedp,
&reversep, &volatilep);
if ((TREE_CODE (base) == INDIRECT_REF
|| (TREE_CODE (base) == MEM_REF
&& integer_zerop (TREE_OPERAND (base,
1))))
&& DECL_P (TREE_OPERAND (base, 0))
&& (TREE_CODE (TREE_TYPE (TREE_OPERAND (base,
0)))
== REFERENCE_TYPE))
base = TREE_OPERAND (base, 0);
if (base != decl)
break;
if (scp)
continue;
gcc_assert (offset == NULL_TREE
|| poly_int_tree_p (offset));
tree d1 = OMP_CLAUSE_DECL (*sc);
tree d2 = OMP_CLAUSE_DECL (c);
while (TREE_CODE (d1) == ARRAY_REF)
d1 = TREE_OPERAND (d1, 0);
while (TREE_CODE (d2) == ARRAY_REF)
d2 = TREE_OPERAND (d2, 0);
if (TREE_CODE (d1) == INDIRECT_REF)
d1 = TREE_OPERAND (d1, 0);
if (TREE_CODE (d2) == INDIRECT_REF)
d2 = TREE_OPERAND (d2, 0);
while (TREE_CODE (d1) == COMPONENT_REF)
if (TREE_CODE (d2) == COMPONENT_REF
&& TREE_OPERAND (d1, 1)
== TREE_OPERAND (d2, 1))
{
d1 = TREE_OPERAND (d1, 0);
d2 = TREE_OPERAND (d2, 0);
}
else
break;
if (d1 == d2)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE appears more than once in map "
"clauses", OMP_CLAUSE_DECL (c));
remove = true;
break;
}
if (offset2)
o2 = wi::to_poly_offset (offset2);
else
o2 = 0;
o2 += bits_to_bytes_round_down (bitpos2);
if (maybe_lt (o1, o2)
|| (known_eq (o1, o2)
&& maybe_lt (bitpos, bitpos2)))
{
if (ptr)
scp = sc;
else
break;
}
}
if (remove)
break;
OMP_CLAUSE_SIZE (*osc)
= size_binop (PLUS_EXPR, OMP_CLAUSE_SIZE (*osc),
size_one_node);
if (ptr)
{
tree c2 = build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_MAP);
tree cl = NULL_TREE;
enum gomp_map_kind mkind
= code == OMP_TARGET_EXIT_DATA
? GOMP_MAP_RELEASE : GOMP_MAP_ALLOC;
OMP_CLAUSE_SET_MAP_KIND (c2, mkind);
OMP_CLAUSE_DECL (c2)
= unshare_expr (OMP_CLAUSE_DECL (c));
OMP_CLAUSE_CHAIN (c2) = scp ? *scp : *prev_list_p;
OMP_CLAUSE_SIZE (c2)
= TYPE_SIZE_UNIT (ptr_type_node);
cl = scp ? *prev_list_p : c2;
if (OMP_CLAUSE_CHAIN (*prev_list_p) != c)
{
tree c4 = OMP_CLAUSE_CHAIN (*prev_list_p);
tree c3
= build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_MAP);
OMP_CLAUSE_SET_MAP_KIND (c3, mkind);
OMP_CLAUSE_DECL (c3)
= unshare_expr (OMP_CLAUSE_DECL (c4));
OMP_CLAUSE_SIZE (c3)
= TYPE_SIZE_UNIT (ptr_type_node);
OMP_CLAUSE_CHAIN (c3) = *prev_list_p;
if (!scp)
OMP_CLAUSE_CHAIN (c2) = c3;
else
cl = c3;
}
if (scp)
*scp = c2;
if (sc == prev_list_p)
{
*sc = cl;
prev_list_p = NULL;
}
else
{
*prev_list_p = OMP_CLAUSE_CHAIN (c);
list_p = prev_list_p;
prev_list_p = NULL;
OMP_CLAUSE_CHAIN (c) = *sc;
*sc = cl;
continue;
}
}
else if (*sc != c)
{
*list_p = OMP_CLAUSE_CHAIN (c);
OMP_CLAUSE_CHAIN (c) = *sc;
*sc = c;
continue;
}
}
}
if (!remove
&& OMP_CLAUSE_MAP_KIND (c) != GOMP_MAP_ALWAYS_POINTER
&& OMP_CLAUSE_CHAIN (c)
&& OMP_CLAUSE_CODE (OMP_CLAUSE_CHAIN (c)) == OMP_CLAUSE_MAP
&& (OMP_CLAUSE_MAP_KIND (OMP_CLAUSE_CHAIN (c))
== GOMP_MAP_ALWAYS_POINTER))
prev_list_p = list_p;
break;
}
flags = GOVD_MAP | GOVD_EXPLICIT;
if (OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_ALWAYS_TO
|| OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_ALWAYS_TOFROM)
flags |= GOVD_MAP_ALWAYS_TO;
goto do_add;
case OMP_CLAUSE_DEPEND:
if (OMP_CLAUSE_DEPEND_KIND (c) == OMP_CLAUSE_DEPEND_SINK)
{
tree deps = OMP_CLAUSE_DECL (c);
while (deps && TREE_CODE (deps) == TREE_LIST)
{
if (TREE_CODE (TREE_PURPOSE (deps)) == TRUNC_DIV_EXPR
&& DECL_P (TREE_OPERAND (TREE_PURPOSE (deps), 1)))
gimplify_expr (&TREE_OPERAND (TREE_PURPOSE (deps), 1),
pre_p, NULL, is_gimple_val, fb_rvalue);
deps = TREE_CHAIN (deps);
}
break;
}
else if (OMP_CLAUSE_DEPEND_KIND (c) == OMP_CLAUSE_DEPEND_SOURCE)
break;
if (TREE_CODE (OMP_CLAUSE_DECL (c)) == COMPOUND_EXPR)
{
gimplify_expr (&TREE_OPERAND (OMP_CLAUSE_DECL (c), 0), pre_p,
NULL, is_gimple_val, fb_rvalue);
OMP_CLAUSE_DECL (c) = TREE_OPERAND (OMP_CLAUSE_DECL (c), 1);
}
if (error_operand_p (OMP_CLAUSE_DECL (c)))
{
remove = true;
break;
}
OMP_CLAUSE_DECL (c) = build_fold_addr_expr (OMP_CLAUSE_DECL (c));
if (gimplify_expr (&OMP_CLAUSE_DECL (c), pre_p, NULL,
is_gimple_val, fb_rvalue) == GS_ERROR)
{
remove = true;
break;
}
break;
case OMP_CLAUSE_TO:
case OMP_CLAUSE_FROM:
case OMP_CLAUSE__CACHE_:
decl = OMP_CLAUSE_DECL (c);
if (error_operand_p (decl))
{
remove = true;
break;
}
if (OMP_CLAUSE_SIZE (c) == NULL_TREE)
OMP_CLAUSE_SIZE (c) = DECL_P (decl) ? DECL_SIZE_UNIT (decl)
: TYPE_SIZE_UNIT (TREE_TYPE (decl));
if (gimplify_expr (&OMP_CLAUSE_SIZE (c), pre_p,
NULL, is_gimple_val, fb_rvalue) == GS_ERROR)
{
remove = true;
break;
}
if (!DECL_P (decl))
{
if (gimplify_expr (&OMP_CLAUSE_DECL (c), pre_p,
NULL, is_gimple_lvalue, fb_lvalue)
== GS_ERROR)
{
remove = true;
break;
}
break;
}
goto do_notice;
case OMP_CLAUSE_USE_DEVICE_PTR:
flags = GOVD_FIRSTPRIVATE | GOVD_EXPLICIT;
goto do_add;
case OMP_CLAUSE_IS_DEVICE_PTR:
flags = GOVD_FIRSTPRIVATE | GOVD_EXPLICIT;
goto do_add;
do_add:
decl = OMP_CLAUSE_DECL (c);
do_add_decl:
if (error_operand_p (decl))
{
remove = true;
break;
}
if (DECL_NAME (decl) == NULL_TREE && (flags & GOVD_SHARED) == 0)
{
tree t = omp_member_access_dummy_var (decl);
if (t)
{
tree v = DECL_VALUE_EXPR (decl);
DECL_NAME (decl) = DECL_NAME (TREE_OPERAND (v, 1));
if (outer_ctx)
omp_notice_variable (outer_ctx, t, true);
}
}
if (code == OACC_DATA
&& OMP_CLAUSE_CODE (c) == OMP_CLAUSE_MAP
&& OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_FIRSTPRIVATE_POINTER)
flags |= GOVD_MAP_0LEN_ARRAY;
omp_add_variable (ctx, decl, flags);
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION
&& OMP_CLAUSE_REDUCTION_PLACEHOLDER (c))
{
omp_add_variable (ctx, OMP_CLAUSE_REDUCTION_PLACEHOLDER (c),
GOVD_LOCAL | GOVD_SEEN);
if (OMP_CLAUSE_REDUCTION_DECL_PLACEHOLDER (c)
&& walk_tree (&OMP_CLAUSE_REDUCTION_INIT (c),
find_decl_expr,
OMP_CLAUSE_REDUCTION_DECL_PLACEHOLDER (c),
NULL) == NULL_TREE)
omp_add_variable (ctx,
OMP_CLAUSE_REDUCTION_DECL_PLACEHOLDER (c),
GOVD_LOCAL | GOVD_SEEN);
gimplify_omp_ctxp = ctx;
push_gimplify_context ();
OMP_CLAUSE_REDUCTION_GIMPLE_INIT (c) = NULL;
OMP_CLAUSE_REDUCTION_GIMPLE_MERGE (c) = NULL;
gimplify_and_add (OMP_CLAUSE_REDUCTION_INIT (c),
&OMP_CLAUSE_REDUCTION_GIMPLE_INIT (c));
pop_gimplify_context
(gimple_seq_first_stmt (OMP_CLAUSE_REDUCTION_GIMPLE_INIT (c)));
push_gimplify_context ();
gimplify_and_add (OMP_CLAUSE_REDUCTION_MERGE (c),
&OMP_CLAUSE_REDUCTION_GIMPLE_MERGE (c));
pop_gimplify_context
(gimple_seq_first_stmt (OMP_CLAUSE_REDUCTION_GIMPLE_MERGE (c)));
OMP_CLAUSE_REDUCTION_INIT (c) = NULL_TREE;
OMP_CLAUSE_REDUCTION_MERGE (c) = NULL_TREE;
gimplify_omp_ctxp = outer_ctx;
}
else if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LASTPRIVATE
&& OMP_CLAUSE_LASTPRIVATE_STMT (c))
{
gimplify_omp_ctxp = ctx;
push_gimplify_context ();
if (TREE_CODE (OMP_CLAUSE_LASTPRIVATE_STMT (c)) != BIND_EXPR)
{
tree bind = build3 (BIND_EXPR, void_type_node, NULL,
NULL, NULL);
TREE_SIDE_EFFECTS (bind) = 1;
BIND_EXPR_BODY (bind) = OMP_CLAUSE_LASTPRIVATE_STMT (c);
OMP_CLAUSE_LASTPRIVATE_STMT (c) = bind;
}
gimplify_and_add (OMP_CLAUSE_LASTPRIVATE_STMT (c),
&OMP_CLAUSE_LASTPRIVATE_GIMPLE_SEQ (c));
pop_gimplify_context
(gimple_seq_first_stmt (OMP_CLAUSE_LASTPRIVATE_GIMPLE_SEQ (c)));
OMP_CLAUSE_LASTPRIVATE_STMT (c) = NULL_TREE;
gimplify_omp_ctxp = outer_ctx;
}
else if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LINEAR
&& OMP_CLAUSE_LINEAR_STMT (c))
{
gimplify_omp_ctxp = ctx;
push_gimplify_context ();
if (TREE_CODE (OMP_CLAUSE_LINEAR_STMT (c)) != BIND_EXPR)
{
tree bind = build3 (BIND_EXPR, void_type_node, NULL,
NULL, NULL);
TREE_SIDE_EFFECTS (bind) = 1;
BIND_EXPR_BODY (bind) = OMP_CLAUSE_LINEAR_STMT (c);
OMP_CLAUSE_LINEAR_STMT (c) = bind;
}
gimplify_and_add (OMP_CLAUSE_LINEAR_STMT (c),
&OMP_CLAUSE_LINEAR_GIMPLE_SEQ (c));
pop_gimplify_context
(gimple_seq_first_stmt (OMP_CLAUSE_LINEAR_GIMPLE_SEQ (c)));
OMP_CLAUSE_LINEAR_STMT (c) = NULL_TREE;
gimplify_omp_ctxp = outer_ctx;
}
if (notice_outer)
goto do_notice;
break;
case OMP_CLAUSE_COPYIN:
case OMP_CLAUSE_COPYPRIVATE:
decl = OMP_CLAUSE_DECL (c);
if (error_operand_p (decl))
{
remove = true;
break;
}
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_COPYPRIVATE
&& !remove
&& !omp_check_private (ctx, decl, true))
{
remove = true;
if (is_global_var (decl))
{
if (DECL_THREAD_LOCAL_P (decl))
remove = false;
else if (DECL_HAS_VALUE_EXPR_P (decl))
{
tree value = get_base_address (DECL_VALUE_EXPR (decl));
if (value
&& DECL_P (value)
&& DECL_THREAD_LOCAL_P (value))
remove = false;
}
}
if (remove)
error_at (OMP_CLAUSE_LOCATION (c),
"copyprivate variable %qE is not threadprivate"
" or private in outer context", DECL_NAME (decl));
}
do_notice:
if (outer_ctx)
omp_notice_variable (outer_ctx, decl, true);
if (check_non_private
&& region_type == ORT_WORKSHARE
&& (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_REDUCTION
|| decl == OMP_CLAUSE_DECL (c)
|| (TREE_CODE (OMP_CLAUSE_DECL (c)) == MEM_REF
&& (TREE_CODE (TREE_OPERAND (OMP_CLAUSE_DECL (c), 0))
== ADDR_EXPR
|| (TREE_CODE (TREE_OPERAND (OMP_CLAUSE_DECL (c), 0))
== POINTER_PLUS_EXPR
&& (TREE_CODE (TREE_OPERAND (TREE_OPERAND
(OMP_CLAUSE_DECL (c), 0), 0))
== ADDR_EXPR)))))
&& omp_check_private (ctx, decl, false))
{
error ("%s variable %qE is private in outer context",
check_non_private, DECL_NAME (decl));
remove = true;
}
break;
case OMP_CLAUSE_IF:
if (OMP_CLAUSE_IF_MODIFIER (c) != ERROR_MARK
&& OMP_CLAUSE_IF_MODIFIER (c) != code)
{
const char *p[2];
for (int i = 0; i < 2; i++)
switch (i ? OMP_CLAUSE_IF_MODIFIER (c) : code)
{
case OMP_PARALLEL: p[i] = "parallel"; break;
case OMP_TASK: p[i] = "task"; break;
case OMP_TASKLOOP: p[i] = "taskloop"; break;
case OMP_TARGET_DATA: p[i] = "target data"; break;
case OMP_TARGET: p[i] = "target"; break;
case OMP_TARGET_UPDATE: p[i] = "target update"; break;
case OMP_TARGET_ENTER_DATA:
p[i] = "target enter data"; break;
case OMP_TARGET_EXIT_DATA: p[i] = "target exit data"; break;
default: gcc_unreachable ();
}
error_at (OMP_CLAUSE_LOCATION (c),
"expected %qs %<if%> clause modifier rather than %qs",
p[0], p[1]);
remove = true;
}
case OMP_CLAUSE_FINAL:
OMP_CLAUSE_OPERAND (c, 0)
= gimple_boolify (OMP_CLAUSE_OPERAND (c, 0));
case OMP_CLAUSE_SCHEDULE:
case OMP_CLAUSE_NUM_THREADS:
case OMP_CLAUSE_NUM_TEAMS:
case OMP_CLAUSE_THREAD_LIMIT:
case OMP_CLAUSE_DIST_SCHEDULE:
case OMP_CLAUSE_DEVICE:
case OMP_CLAUSE_PRIORITY:
case OMP_CLAUSE_GRAINSIZE:
case OMP_CLAUSE_NUM_TASKS:
case OMP_CLAUSE_HINT:
case OMP_CLAUSE_ASYNC:
case OMP_CLAUSE_WAIT:
case OMP_CLAUSE_NUM_GANGS:
case OMP_CLAUSE_NUM_WORKERS:
case OMP_CLAUSE_VECTOR_LENGTH:
case OMP_CLAUSE_WORKER:
case OMP_CLAUSE_VECTOR:
if (gimplify_expr (&OMP_CLAUSE_OPERAND (c, 0), pre_p, NULL,
is_gimple_val, fb_rvalue) == GS_ERROR)
remove = true;
break;
case OMP_CLAUSE_GANG:
if (gimplify_expr (&OMP_CLAUSE_OPERAND (c, 0), pre_p, NULL,
is_gimple_val, fb_rvalue) == GS_ERROR)
remove = true;
if (gimplify_expr (&OMP_CLAUSE_OPERAND (c, 1), pre_p, NULL,
is_gimple_val, fb_rvalue) == GS_ERROR)
remove = true;
break;
case OMP_CLAUSE_NOWAIT:
case OMP_CLAUSE_ORDERED:
case OMP_CLAUSE_UNTIED:
case OMP_CLAUSE_COLLAPSE:
case OMP_CLAUSE_TILE:
case OMP_CLAUSE_AUTO:
case OMP_CLAUSE_SEQ:
case OMP_CLAUSE_INDEPENDENT:
case OMP_CLAUSE_MERGEABLE:
case OMP_CLAUSE_PROC_BIND:
case OMP_CLAUSE_SAFELEN:
case OMP_CLAUSE_SIMDLEN:
case OMP_CLAUSE_NOGROUP:
case OMP_CLAUSE_THREADS:
case OMP_CLAUSE_SIMD:
break;
case OMP_CLAUSE_DEFAULTMAP:
ctx->target_map_scalars_firstprivate = false;
break;
case OMP_CLAUSE_ALIGNED:
decl = OMP_CLAUSE_DECL (c);
if (error_operand_p (decl))
{
remove = true;
break;
}
if (gimplify_expr (&OMP_CLAUSE_ALIGNED_ALIGNMENT (c), pre_p, NULL,
is_gimple_val, fb_rvalue) == GS_ERROR)
{
remove = true;
break;
}
if (!is_global_var (decl)
&& TREE_CODE (TREE_TYPE (decl)) == POINTER_TYPE)
omp_add_variable (ctx, decl, GOVD_ALIGNED);
break;
case OMP_CLAUSE_DEFAULT:
ctx->default_kind = OMP_CLAUSE_DEFAULT_KIND (c);
break;
default:
gcc_unreachable ();
}
if (code == OACC_DATA
&& OMP_CLAUSE_CODE (c) == OMP_CLAUSE_MAP
&& OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_FIRSTPRIVATE_POINTER)
remove = true;
if (remove)
*list_p = OMP_CLAUSE_CHAIN (c);
else
list_p = &OMP_CLAUSE_CHAIN (c);
}
gimplify_omp_ctxp = ctx;
if (struct_map_to_clause)
delete struct_map_to_clause;
}
static bool
omp_shared_to_firstprivate_optimizable_decl_p (tree decl)
{
if (TREE_ADDRESSABLE (decl))
return false;
tree type = TREE_TYPE (decl);
if (!is_gimple_reg_type (type)
|| TREE_CODE (type) == REFERENCE_TYPE
|| TREE_ADDRESSABLE (type))
return false;
HOST_WIDE_INT len = int_size_in_bytes (type);
if (len == -1 || len > 4 * POINTER_SIZE / BITS_PER_UNIT)
return false;
if (lang_hooks.decls.omp_privatize_by_reference (decl))
return false;
return true;
}
static void
omp_mark_stores (struct gimplify_omp_ctx *ctx, tree decl)
{
for (; ctx; ctx = ctx->outer_context)
{
splay_tree_node n = splay_tree_lookup (ctx->variables,
(splay_tree_key) decl);
if (n == NULL)
continue;
else if (n->value & GOVD_SHARED)
{
n->value |= GOVD_WRITTEN;
return;
}
else if (n->value & GOVD_DATA_SHARE_CLASS)
return;
}
}
static tree
omp_find_stores_op (tree *tp, int *walk_subtrees, void *data)
{
struct walk_stmt_info *wi = (struct walk_stmt_info *) data;
*walk_subtrees = 0;
if (!wi->is_lhs)
return NULL_TREE;
tree op = *tp;
do
{
if (handled_component_p (op))
op = TREE_OPERAND (op, 0);
else if ((TREE_CODE (op) == MEM_REF || TREE_CODE (op) == TARGET_MEM_REF)
&& TREE_CODE (TREE_OPERAND (op, 0)) == ADDR_EXPR)
op = TREE_OPERAND (TREE_OPERAND (op, 0), 0);
else
break;
}
while (1);
if (!DECL_P (op) || !omp_shared_to_firstprivate_optimizable_decl_p (op))
return NULL_TREE;
omp_mark_stores (gimplify_omp_ctxp, op);
return NULL_TREE;
}
static tree
omp_find_stores_stmt (gimple_stmt_iterator *gsi_p,
bool *handled_ops_p,
struct walk_stmt_info *wi)
{
gimple *stmt = gsi_stmt (*gsi_p);
switch (gimple_code (stmt))
{
case GIMPLE_OMP_FOR:
*handled_ops_p = true;
if (gimple_omp_for_pre_body (stmt))
walk_gimple_seq (gimple_omp_for_pre_body (stmt),
omp_find_stores_stmt, omp_find_stores_op, wi);
break;
case GIMPLE_OMP_PARALLEL:
case GIMPLE_OMP_TASK:
case GIMPLE_OMP_SECTIONS:
case GIMPLE_OMP_SINGLE:
case GIMPLE_OMP_TARGET:
case GIMPLE_OMP_TEAMS:
case GIMPLE_OMP_CRITICAL:
*handled_ops_p = true;
break;
default:
break;
}
return NULL_TREE;
}
struct gimplify_adjust_omp_clauses_data
{
tree *list_p;
gimple_seq *pre_p;
};
static int
gimplify_adjust_omp_clauses_1 (splay_tree_node n, void *data)
{
tree *list_p = ((struct gimplify_adjust_omp_clauses_data *) data)->list_p;
gimple_seq *pre_p
= ((struct gimplify_adjust_omp_clauses_data *) data)->pre_p;
tree decl = (tree) n->key;
unsigned flags = n->value;
enum omp_clause_code code;
tree clause;
bool private_debug;
if (flags & (GOVD_EXPLICIT | GOVD_LOCAL))
return 0;
if ((flags & GOVD_SEEN) == 0)
return 0;
if (flags & GOVD_DEBUG_PRIVATE)
{
gcc_assert ((flags & GOVD_DATA_SHARE_CLASS) == GOVD_SHARED);
private_debug = true;
}
else if (flags & GOVD_MAP)
private_debug = false;
else
private_debug
= lang_hooks.decls.omp_private_debug_clause (decl,
!!(flags & GOVD_SHARED));
if (private_debug)
code = OMP_CLAUSE_PRIVATE;
else if (flags & GOVD_MAP)
{
code = OMP_CLAUSE_MAP;
if ((gimplify_omp_ctxp->region_type & ORT_ACC) == 0
&& TYPE_ATOMIC (strip_array_types (TREE_TYPE (decl))))
{
error ("%<_Atomic%> %qD in implicit %<map%> clause", decl);
return 0;
}
}
else if (flags & GOVD_SHARED)
{
if (is_global_var (decl))
{
struct gimplify_omp_ctx *ctx = gimplify_omp_ctxp->outer_context;
while (ctx != NULL)
{
splay_tree_node on
= splay_tree_lookup (ctx->variables, (splay_tree_key) decl);
if (on && (on->value & (GOVD_FIRSTPRIVATE | GOVD_LASTPRIVATE
| GOVD_PRIVATE | GOVD_REDUCTION
| GOVD_LINEAR | GOVD_MAP)) != 0)
break;
ctx = ctx->outer_context;
}
if (ctx == NULL)
return 0;
}
code = OMP_CLAUSE_SHARED;
}
else if (flags & GOVD_PRIVATE)
code = OMP_CLAUSE_PRIVATE;
else if (flags & GOVD_FIRSTPRIVATE)
{
code = OMP_CLAUSE_FIRSTPRIVATE;
if ((gimplify_omp_ctxp->region_type & ORT_TARGET)
&& (gimplify_omp_ctxp->region_type & ORT_ACC) == 0
&& TYPE_ATOMIC (strip_array_types (TREE_TYPE (decl))))
{
error ("%<_Atomic%> %qD in implicit %<firstprivate%> clause on "
"%<target%> construct", decl);
return 0;
}
}
else if (flags & GOVD_LASTPRIVATE)
code = OMP_CLAUSE_LASTPRIVATE;
else if (flags & GOVD_ALIGNED)
return 0;
else
gcc_unreachable ();
if (((flags & GOVD_LASTPRIVATE)
|| (code == OMP_CLAUSE_SHARED && (flags & GOVD_WRITTEN)))
&& omp_shared_to_firstprivate_optimizable_decl_p (decl))
omp_mark_stores (gimplify_omp_ctxp->outer_context, decl);
tree chain = *list_p;
clause = build_omp_clause (input_location, code);
OMP_CLAUSE_DECL (clause) = decl;
OMP_CLAUSE_CHAIN (clause) = chain;
if (private_debug)
OMP_CLAUSE_PRIVATE_DEBUG (clause) = 1;
else if (code == OMP_CLAUSE_PRIVATE && (flags & GOVD_PRIVATE_OUTER_REF))
OMP_CLAUSE_PRIVATE_OUTER_REF (clause) = 1;
else if (code == OMP_CLAUSE_SHARED
&& (flags & GOVD_WRITTEN) == 0
&& omp_shared_to_firstprivate_optimizable_decl_p (decl))
OMP_CLAUSE_SHARED_READONLY (clause) = 1;
else if (code == OMP_CLAUSE_FIRSTPRIVATE && (flags & GOVD_EXPLICIT) == 0)
OMP_CLAUSE_FIRSTPRIVATE_IMPLICIT (clause) = 1;
else if (code == OMP_CLAUSE_MAP && (flags & GOVD_MAP_0LEN_ARRAY) != 0)
{
tree nc = build_omp_clause (input_location, OMP_CLAUSE_MAP);
OMP_CLAUSE_DECL (nc) = decl;
if (TREE_CODE (TREE_TYPE (decl)) == REFERENCE_TYPE
&& TREE_CODE (TREE_TYPE (TREE_TYPE (decl))) == POINTER_TYPE)
OMP_CLAUSE_DECL (clause)
= build_simple_mem_ref_loc (input_location, decl);
OMP_CLAUSE_DECL (clause)
= build2 (MEM_REF, char_type_node, OMP_CLAUSE_DECL (clause),
build_int_cst (build_pointer_type (char_type_node), 0));
OMP_CLAUSE_SIZE (clause) = size_zero_node;
OMP_CLAUSE_SIZE (nc) = size_zero_node;
OMP_CLAUSE_SET_MAP_KIND (clause, GOMP_MAP_ALLOC);
OMP_CLAUSE_MAP_MAYBE_ZERO_LENGTH_ARRAY_SECTION (clause) = 1;
OMP_CLAUSE_SET_MAP_KIND (nc, GOMP_MAP_FIRSTPRIVATE_POINTER);
OMP_CLAUSE_CHAIN (nc) = chain;
OMP_CLAUSE_CHAIN (clause) = nc;
struct gimplify_omp_ctx *ctx = gimplify_omp_ctxp;
gimplify_omp_ctxp = ctx->outer_context;
gimplify_expr (&TREE_OPERAND (OMP_CLAUSE_DECL (clause), 0),
pre_p, NULL, is_gimple_val, fb_rvalue);
gimplify_omp_ctxp = ctx;
}
else if (code == OMP_CLAUSE_MAP)
{
int kind;
switch (flags & (GOVD_MAP_TO_ONLY
| GOVD_MAP_FORCE
| GOVD_MAP_FORCE_PRESENT))
{
case 0:
kind = GOMP_MAP_TOFROM;
break;
case GOVD_MAP_FORCE:
kind = GOMP_MAP_TOFROM | GOMP_MAP_FLAG_FORCE;
break;
case GOVD_MAP_TO_ONLY:
kind = GOMP_MAP_TO;
break;
case GOVD_MAP_TO_ONLY | GOVD_MAP_FORCE:
kind = GOMP_MAP_TO | GOMP_MAP_FLAG_FORCE;
break;
case GOVD_MAP_FORCE_PRESENT:
kind = GOMP_MAP_FORCE_PRESENT;
break;
default:
gcc_unreachable ();
}
OMP_CLAUSE_SET_MAP_KIND (clause, kind);
if (DECL_SIZE (decl)
&& TREE_CODE (DECL_SIZE (decl)) != INTEGER_CST)
{
tree decl2 = DECL_VALUE_EXPR (decl);
gcc_assert (TREE_CODE (decl2) == INDIRECT_REF);
decl2 = TREE_OPERAND (decl2, 0);
gcc_assert (DECL_P (decl2));
tree mem = build_simple_mem_ref (decl2);
OMP_CLAUSE_DECL (clause) = mem;
OMP_CLAUSE_SIZE (clause) = TYPE_SIZE_UNIT (TREE_TYPE (decl));
if (gimplify_omp_ctxp->outer_context)
{
struct gimplify_omp_ctx *ctx = gimplify_omp_ctxp->outer_context;
omp_notice_variable (ctx, decl2, true);
omp_notice_variable (ctx, OMP_CLAUSE_SIZE (clause), true);
}
tree nc = build_omp_clause (OMP_CLAUSE_LOCATION (clause),
OMP_CLAUSE_MAP);
OMP_CLAUSE_DECL (nc) = decl;
OMP_CLAUSE_SIZE (nc) = size_zero_node;
if (gimplify_omp_ctxp->target_firstprivatize_array_bases)
OMP_CLAUSE_SET_MAP_KIND (nc, GOMP_MAP_FIRSTPRIVATE_POINTER);
else
OMP_CLAUSE_SET_MAP_KIND (nc, GOMP_MAP_POINTER);
OMP_CLAUSE_CHAIN (nc) = OMP_CLAUSE_CHAIN (clause);
OMP_CLAUSE_CHAIN (clause) = nc;
}
else if (gimplify_omp_ctxp->target_firstprivatize_array_bases
&& lang_hooks.decls.omp_privatize_by_reference (decl))
{
OMP_CLAUSE_DECL (clause) = build_simple_mem_ref (decl);
OMP_CLAUSE_SIZE (clause)
= unshare_expr (TYPE_SIZE_UNIT (TREE_TYPE (TREE_TYPE (decl))));
struct gimplify_omp_ctx *ctx = gimplify_omp_ctxp;
gimplify_omp_ctxp = ctx->outer_context;
gimplify_expr (&OMP_CLAUSE_SIZE (clause),
pre_p, NULL, is_gimple_val, fb_rvalue);
gimplify_omp_ctxp = ctx;
tree nc = build_omp_clause (OMP_CLAUSE_LOCATION (clause),
OMP_CLAUSE_MAP);
OMP_CLAUSE_DECL (nc) = decl;
OMP_CLAUSE_SIZE (nc) = size_zero_node;
OMP_CLAUSE_SET_MAP_KIND (nc, GOMP_MAP_FIRSTPRIVATE_REFERENCE);
OMP_CLAUSE_CHAIN (nc) = OMP_CLAUSE_CHAIN (clause);
OMP_CLAUSE_CHAIN (clause) = nc;
}
else
OMP_CLAUSE_SIZE (clause) = DECL_SIZE_UNIT (decl);
}
if (code == OMP_CLAUSE_FIRSTPRIVATE && (flags & GOVD_LASTPRIVATE) != 0)
{
tree nc = build_omp_clause (input_location, OMP_CLAUSE_LASTPRIVATE);
OMP_CLAUSE_DECL (nc) = decl;
OMP_CLAUSE_LASTPRIVATE_FIRSTPRIVATE (nc) = 1;
OMP_CLAUSE_CHAIN (nc) = chain;
OMP_CLAUSE_CHAIN (clause) = nc;
struct gimplify_omp_ctx *ctx = gimplify_omp_ctxp;
gimplify_omp_ctxp = ctx->outer_context;
lang_hooks.decls.omp_finish_clause (nc, pre_p);
gimplify_omp_ctxp = ctx;
}
*list_p = clause;
struct gimplify_omp_ctx *ctx = gimplify_omp_ctxp;
gimplify_omp_ctxp = ctx->outer_context;
lang_hooks.decls.omp_finish_clause (clause, pre_p);
if (gimplify_omp_ctxp)
for (; clause != chain; clause = OMP_CLAUSE_CHAIN (clause))
if (OMP_CLAUSE_CODE (clause) == OMP_CLAUSE_MAP
&& DECL_P (OMP_CLAUSE_SIZE (clause)))
omp_notice_variable (gimplify_omp_ctxp, OMP_CLAUSE_SIZE (clause),
true);
gimplify_omp_ctxp = ctx;
return 0;
}
static void
gimplify_adjust_omp_clauses (gimple_seq *pre_p, gimple_seq body, tree *list_p,
enum tree_code code)
{
struct gimplify_omp_ctx *ctx = gimplify_omp_ctxp;
tree c, decl;
if (body)
{
struct gimplify_omp_ctx *octx;
for (octx = ctx; octx; octx = octx->outer_context)
if ((octx->region_type & (ORT_PARALLEL | ORT_TASK | ORT_TEAMS)) != 0)
break;
if (octx)
{
struct walk_stmt_info wi;
memset (&wi, 0, sizeof (wi));
walk_gimple_seq (body, omp_find_stores_stmt,
omp_find_stores_op, &wi);
}
}
while ((c = *list_p) != NULL)
{
splay_tree_node n;
bool remove = false;
switch (OMP_CLAUSE_CODE (c))
{
case OMP_CLAUSE_FIRSTPRIVATE:
if ((ctx->region_type & ORT_TARGET)
&& (ctx->region_type & ORT_ACC) == 0
&& TYPE_ATOMIC (strip_array_types
(TREE_TYPE (OMP_CLAUSE_DECL (c)))))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<_Atomic%> %qD in %<firstprivate%> clause on "
"%<target%> construct", OMP_CLAUSE_DECL (c));
remove = true;
break;
}
case OMP_CLAUSE_PRIVATE:
case OMP_CLAUSE_SHARED:
case OMP_CLAUSE_LINEAR:
decl = OMP_CLAUSE_DECL (c);
n = splay_tree_lookup (ctx->variables, (splay_tree_key) decl);
remove = !(n->value & GOVD_SEEN);
if (! remove)
{
bool shared = OMP_CLAUSE_CODE (c) == OMP_CLAUSE_SHARED;
if ((n->value & GOVD_DEBUG_PRIVATE)
|| lang_hooks.decls.omp_private_debug_clause (decl, shared))
{
gcc_assert ((n->value & GOVD_DEBUG_PRIVATE) == 0
|| ((n->value & GOVD_DATA_SHARE_CLASS)
== GOVD_SHARED));
OMP_CLAUSE_SET_CODE (c, OMP_CLAUSE_PRIVATE);
OMP_CLAUSE_PRIVATE_DEBUG (c) = 1;
}
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_SHARED
&& (n->value & GOVD_WRITTEN) == 0
&& DECL_P (decl)
&& omp_shared_to_firstprivate_optimizable_decl_p (decl))
OMP_CLAUSE_SHARED_READONLY (c) = 1;
else if (DECL_P (decl)
&& ((OMP_CLAUSE_CODE (c) == OMP_CLAUSE_SHARED
&& (n->value & GOVD_WRITTEN) != 0)
|| (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LINEAR
&& !OMP_CLAUSE_LINEAR_NO_COPYOUT (c)))
&& omp_shared_to_firstprivate_optimizable_decl_p (decl))
omp_mark_stores (gimplify_omp_ctxp->outer_context, decl);
}
break;
case OMP_CLAUSE_LASTPRIVATE:
decl = OMP_CLAUSE_DECL (c);
n = splay_tree_lookup (ctx->variables, (splay_tree_key) decl);
OMP_CLAUSE_LASTPRIVATE_FIRSTPRIVATE (c)
= (n->value & GOVD_FIRSTPRIVATE) != 0;
if (code == OMP_DISTRIBUTE
&& OMP_CLAUSE_LASTPRIVATE_FIRSTPRIVATE (c))
{
remove = true;
error_at (OMP_CLAUSE_LOCATION (c),
"same variable used in %<firstprivate%> and "
"%<lastprivate%> clauses on %<distribute%> "
"construct");
}
if (!remove
&& OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LASTPRIVATE
&& DECL_P (decl)
&& omp_shared_to_firstprivate_optimizable_decl_p (decl))
omp_mark_stores (gimplify_omp_ctxp->outer_context, decl);
break;
case OMP_CLAUSE_ALIGNED:
decl = OMP_CLAUSE_DECL (c);
if (!is_global_var (decl))
{
n = splay_tree_lookup (ctx->variables, (splay_tree_key) decl);
remove = n == NULL || !(n->value & GOVD_SEEN);
if (!remove && TREE_CODE (TREE_TYPE (decl)) == POINTER_TYPE)
{
struct gimplify_omp_ctx *octx;
if (n != NULL
&& (n->value & (GOVD_DATA_SHARE_CLASS
& ~GOVD_FIRSTPRIVATE)))
remove = true;
else
for (octx = ctx->outer_context; octx;
octx = octx->outer_context)
{
n = splay_tree_lookup (octx->variables,
(splay_tree_key) decl);
if (n == NULL)
continue;
if (n->value & GOVD_LOCAL)
break;
if (n->value & GOVD_SHARED)
{
remove = true;
break;
}
}
}
}
else if (TREE_CODE (TREE_TYPE (decl)) == ARRAY_TYPE)
{
n = splay_tree_lookup (ctx->variables, (splay_tree_key) decl);
if (n != NULL && (n->value & GOVD_DATA_SHARE_CLASS) != 0)
remove = true;
}
break;
case OMP_CLAUSE_MAP:
if (code == OMP_TARGET_EXIT_DATA
&& OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_ALWAYS_POINTER)
{
remove = true;
break;
}
decl = OMP_CLAUSE_DECL (c);
if (ctx->region_type == ORT_ACC_PARALLEL)
{
tree t = DECL_P (decl) ? decl : TREE_OPERAND (decl, 0);
n = NULL;
if (DECL_P (t))
n = splay_tree_lookup (ctx->variables, (splay_tree_key) t);
if (n && (n->value & GOVD_REDUCTION))
{
enum gomp_map_kind kind = OMP_CLAUSE_MAP_KIND (c);
OMP_CLAUSE_MAP_IN_REDUCTION (c) = 1;
if ((kind & GOMP_MAP_TOFROM) != GOMP_MAP_TOFROM
&& kind != GOMP_MAP_FORCE_PRESENT
&& kind != GOMP_MAP_POINTER)
{
warning_at (OMP_CLAUSE_LOCATION (c), 0,
"incompatible data clause with reduction "
"on %qE; promoting to present_or_copy",
DECL_NAME (t));
OMP_CLAUSE_SET_MAP_KIND (c, GOMP_MAP_TOFROM);
}
}
}
if (!DECL_P (decl))
{
if ((ctx->region_type & ORT_TARGET) != 0
&& OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_FIRSTPRIVATE_POINTER)
{
if (TREE_CODE (decl) == INDIRECT_REF
&& TREE_CODE (TREE_OPERAND (decl, 0)) == COMPONENT_REF
&& (TREE_CODE (TREE_TYPE (TREE_OPERAND (decl, 0)))
== REFERENCE_TYPE))
decl = TREE_OPERAND (decl, 0);
if (TREE_CODE (decl) == COMPONENT_REF)
{
while (TREE_CODE (decl) == COMPONENT_REF)
decl = TREE_OPERAND (decl, 0);
if (DECL_P (decl))
{
n = splay_tree_lookup (ctx->variables,
(splay_tree_key) decl);
if (!(n->value & GOVD_SEEN))
remove = true;
}
}
}
break;
}
n = splay_tree_lookup (ctx->variables, (splay_tree_key) decl);
if ((ctx->region_type & ORT_TARGET) != 0
&& !(n->value & GOVD_SEEN)
&& GOMP_MAP_ALWAYS_P (OMP_CLAUSE_MAP_KIND (c)) == 0
&& (!is_global_var (decl)
|| !lookup_attribute ("omp declare target link",
DECL_ATTRIBUTES (decl))))
{
remove = true;
if (OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_STRUCT)
{
HOST_WIDE_INT cnt = tree_to_shwi (OMP_CLAUSE_SIZE (c));
while (cnt--)
OMP_CLAUSE_CHAIN (c)
= OMP_CLAUSE_CHAIN (OMP_CLAUSE_CHAIN (c));
}
}
else if (OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_STRUCT
&& code == OMP_TARGET_EXIT_DATA)
remove = true;
else if (DECL_SIZE (decl)
&& TREE_CODE (DECL_SIZE (decl)) != INTEGER_CST
&& OMP_CLAUSE_MAP_KIND (c) != GOMP_MAP_POINTER
&& OMP_CLAUSE_MAP_KIND (c) != GOMP_MAP_FIRSTPRIVATE_POINTER
&& (OMP_CLAUSE_MAP_KIND (c)
!= GOMP_MAP_FIRSTPRIVATE_REFERENCE))
{
gcc_assert (OMP_CLAUSE_MAP_KIND (c) != GOMP_MAP_FORCE_DEVICEPTR);
tree decl2 = DECL_VALUE_EXPR (decl);
gcc_assert (TREE_CODE (decl2) == INDIRECT_REF);
decl2 = TREE_OPERAND (decl2, 0);
gcc_assert (DECL_P (decl2));
tree mem = build_simple_mem_ref (decl2);
OMP_CLAUSE_DECL (c) = mem;
OMP_CLAUSE_SIZE (c) = TYPE_SIZE_UNIT (TREE_TYPE (decl));
if (ctx->outer_context)
{
omp_notice_variable (ctx->outer_context, decl2, true);
omp_notice_variable (ctx->outer_context,
OMP_CLAUSE_SIZE (c), true);
}
if (((ctx->region_type & ORT_TARGET) != 0
|| !ctx->target_firstprivatize_array_bases)
&& ((n->value & GOVD_SEEN) == 0
|| (n->value & (GOVD_PRIVATE | GOVD_FIRSTPRIVATE)) == 0))
{
tree nc = build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_MAP);
OMP_CLAUSE_DECL (nc) = decl;
OMP_CLAUSE_SIZE (nc) = size_zero_node;
if (ctx->target_firstprivatize_array_bases)
OMP_CLAUSE_SET_MAP_KIND (nc,
GOMP_MAP_FIRSTPRIVATE_POINTER);
else
OMP_CLAUSE_SET_MAP_KIND (nc, GOMP_MAP_POINTER);
OMP_CLAUSE_CHAIN (nc) = OMP_CLAUSE_CHAIN (c);
OMP_CLAUSE_CHAIN (c) = nc;
c = nc;
}
}
else
{
if (OMP_CLAUSE_SIZE (c) == NULL_TREE)
OMP_CLAUSE_SIZE (c) = DECL_SIZE_UNIT (decl);
gcc_assert ((n->value & GOVD_SEEN) == 0
|| ((n->value & (GOVD_PRIVATE | GOVD_FIRSTPRIVATE))
== 0));
}
break;
case OMP_CLAUSE_TO:
case OMP_CLAUSE_FROM:
case OMP_CLAUSE__CACHE_:
decl = OMP_CLAUSE_DECL (c);
if (!DECL_P (decl))
break;
if (DECL_SIZE (decl)
&& TREE_CODE (DECL_SIZE (decl)) != INTEGER_CST)
{
tree decl2 = DECL_VALUE_EXPR (decl);
gcc_assert (TREE_CODE (decl2) == INDIRECT_REF);
decl2 = TREE_OPERAND (decl2, 0);
gcc_assert (DECL_P (decl2));
tree mem = build_simple_mem_ref (decl2);
OMP_CLAUSE_DECL (c) = mem;
OMP_CLAUSE_SIZE (c) = TYPE_SIZE_UNIT (TREE_TYPE (decl));
if (ctx->outer_context)
{
omp_notice_variable (ctx->outer_context, decl2, true);
omp_notice_variable (ctx->outer_context,
OMP_CLAUSE_SIZE (c), true);
}
}
else if (OMP_CLAUSE_SIZE (c) == NULL_TREE)
OMP_CLAUSE_SIZE (c) = DECL_SIZE_UNIT (decl);
break;
case OMP_CLAUSE_REDUCTION:
decl = OMP_CLAUSE_DECL (c);
if (ctx->region_type == ORT_ACC_PARALLEL)
{
n = splay_tree_lookup (ctx->variables, (splay_tree_key) decl);
if (n->value & (GOVD_PRIVATE | GOVD_FIRSTPRIVATE))
error_at (OMP_CLAUSE_LOCATION (c), "invalid private "
"reduction on %qE", DECL_NAME (decl));
else if ((n->value & GOVD_MAP) == 0)
{
tree next = OMP_CLAUSE_CHAIN (c);
tree nc = build_omp_clause (UNKNOWN_LOCATION, OMP_CLAUSE_MAP);
OMP_CLAUSE_SET_MAP_KIND (nc, GOMP_MAP_TOFROM);
OMP_CLAUSE_DECL (nc) = decl;
OMP_CLAUSE_CHAIN (c) = nc;
lang_hooks.decls.omp_finish_clause (nc, pre_p);
while (1)
{
OMP_CLAUSE_MAP_IN_REDUCTION (nc) = 1;
if (OMP_CLAUSE_CHAIN (nc) == NULL)
break;
nc = OMP_CLAUSE_CHAIN (nc);
}
OMP_CLAUSE_CHAIN (nc) = next;
n->value |= GOVD_MAP;
}
}
if (DECL_P (decl)
&& omp_shared_to_firstprivate_optimizable_decl_p (decl))
omp_mark_stores (gimplify_omp_ctxp->outer_context, decl);
break;
case OMP_CLAUSE_COPYIN:
case OMP_CLAUSE_COPYPRIVATE:
case OMP_CLAUSE_IF:
case OMP_CLAUSE_NUM_THREADS:
case OMP_CLAUSE_NUM_TEAMS:
case OMP_CLAUSE_THREAD_LIMIT:
case OMP_CLAUSE_DIST_SCHEDULE:
case OMP_CLAUSE_DEVICE:
case OMP_CLAUSE_SCHEDULE:
case OMP_CLAUSE_NOWAIT:
case OMP_CLAUSE_ORDERED:
case OMP_CLAUSE_DEFAULT:
case OMP_CLAUSE_UNTIED:
case OMP_CLAUSE_COLLAPSE:
case OMP_CLAUSE_FINAL:
case OMP_CLAUSE_MERGEABLE:
case OMP_CLAUSE_PROC_BIND:
case OMP_CLAUSE_SAFELEN:
case OMP_CLAUSE_SIMDLEN:
case OMP_CLAUSE_DEPEND:
case OMP_CLAUSE_PRIORITY:
case OMP_CLAUSE_GRAINSIZE:
case OMP_CLAUSE_NUM_TASKS:
case OMP_CLAUSE_NOGROUP:
case OMP_CLAUSE_THREADS:
case OMP_CLAUSE_SIMD:
case OMP_CLAUSE_HINT:
case OMP_CLAUSE_DEFAULTMAP:
case OMP_CLAUSE_USE_DEVICE_PTR:
case OMP_CLAUSE_IS_DEVICE_PTR:
case OMP_CLAUSE_ASYNC:
case OMP_CLAUSE_WAIT:
case OMP_CLAUSE_INDEPENDENT:
case OMP_CLAUSE_NUM_GANGS:
case OMP_CLAUSE_NUM_WORKERS:
case OMP_CLAUSE_VECTOR_LENGTH:
case OMP_CLAUSE_GANG:
case OMP_CLAUSE_WORKER:
case OMP_CLAUSE_VECTOR:
case OMP_CLAUSE_AUTO:
case OMP_CLAUSE_SEQ:
case OMP_CLAUSE_TILE:
break;
default:
gcc_unreachable ();
}
if (remove)
*list_p = OMP_CLAUSE_CHAIN (c);
else
list_p = &OMP_CLAUSE_CHAIN (c);
}
struct gimplify_adjust_omp_clauses_data data;
data.list_p = list_p;
data.pre_p = pre_p;
splay_tree_foreach (ctx->variables, gimplify_adjust_omp_clauses_1, &data);
gimplify_omp_ctxp = ctx->outer_context;
delete_omp_context (ctx);
}
static void
gimplify_oacc_cache (tree *expr_p, gimple_seq *pre_p)
{
tree expr = *expr_p;
gimplify_scan_omp_clauses (&OACC_CACHE_CLAUSES (expr), pre_p, ORT_ACC,
OACC_CACHE);
gimplify_adjust_omp_clauses (pre_p, NULL, &OACC_CACHE_CLAUSES (expr),
OACC_CACHE);
*expr_p = NULL_TREE;
}
static tree
gimplify_oacc_declare_1 (tree clause)
{
HOST_WIDE_INT kind, new_op;
bool ret = false;
tree c = NULL;
kind = OMP_CLAUSE_MAP_KIND (clause);
switch (kind)
{
case GOMP_MAP_ALLOC:
case GOMP_MAP_FORCE_ALLOC:
case GOMP_MAP_FORCE_TO:
new_op = GOMP_MAP_DELETE;
ret = true;
break;
case GOMP_MAP_FORCE_FROM:
OMP_CLAUSE_SET_MAP_KIND (clause, GOMP_MAP_FORCE_ALLOC);
new_op = GOMP_MAP_FORCE_FROM;
ret = true;
break;
case GOMP_MAP_FORCE_TOFROM:
OMP_CLAUSE_SET_MAP_KIND (clause, GOMP_MAP_FORCE_TO);
new_op = GOMP_MAP_FORCE_FROM;
ret = true;
break;
case GOMP_MAP_FROM:
OMP_CLAUSE_SET_MAP_KIND (clause, GOMP_MAP_FORCE_ALLOC);
new_op = GOMP_MAP_FROM;
ret = true;
break;
case GOMP_MAP_TOFROM:
OMP_CLAUSE_SET_MAP_KIND (clause, GOMP_MAP_TO);
new_op = GOMP_MAP_FROM;
ret = true;
break;
case GOMP_MAP_DEVICE_RESIDENT:
case GOMP_MAP_FORCE_DEVICEPTR:
case GOMP_MAP_FORCE_PRESENT:
case GOMP_MAP_LINK:
case GOMP_MAP_POINTER:
case GOMP_MAP_TO:
break;
default:
gcc_unreachable ();
break;
}
if (ret)
{
c = build_omp_clause (OMP_CLAUSE_LOCATION (clause), OMP_CLAUSE_MAP);
OMP_CLAUSE_SET_MAP_KIND (c, new_op);
OMP_CLAUSE_DECL (c) = OMP_CLAUSE_DECL (clause);
}
return c;
}
static void
gimplify_oacc_declare (tree *expr_p, gimple_seq *pre_p)
{
tree expr = *expr_p;
gomp_target *stmt;
tree clauses, t, decl;
clauses = OACC_DECLARE_CLAUSES (expr);
gimplify_scan_omp_clauses (&clauses, pre_p, ORT_TARGET_DATA, OACC_DECLARE);
gimplify_adjust_omp_clauses (pre_p, NULL, &clauses, OACC_DECLARE);
for (t = clauses; t; t = OMP_CLAUSE_CHAIN (t))
{
decl = OMP_CLAUSE_DECL (t);
if (TREE_CODE (decl) == MEM_REF)
decl = TREE_OPERAND (decl, 0);
if (VAR_P (decl) && !is_oacc_declared (decl))
{
tree attr = get_identifier ("oacc declare target");
DECL_ATTRIBUTES (decl) = tree_cons (attr, NULL_TREE,
DECL_ATTRIBUTES (decl));
}
if (VAR_P (decl)
&& !is_global_var (decl)
&& DECL_CONTEXT (decl) == current_function_decl)
{
tree c = gimplify_oacc_declare_1 (t);
if (c)
{
if (oacc_declare_returns == NULL)
oacc_declare_returns = new hash_map<tree, tree>;
oacc_declare_returns->put (decl, c);
}
}
if (gimplify_omp_ctxp)
omp_add_variable (gimplify_omp_ctxp, decl, GOVD_SEEN);
}
stmt = gimple_build_omp_target (NULL, GF_OMP_TARGET_KIND_OACC_DECLARE,
clauses);
gimplify_seq_add_stmt (pre_p, stmt);
*expr_p = NULL_TREE;
}
static void
gimplify_omp_parallel (tree *expr_p, gimple_seq *pre_p)
{
tree expr = *expr_p;
gimple *g;
gimple_seq body = NULL;
gimplify_scan_omp_clauses (&OMP_PARALLEL_CLAUSES (expr), pre_p,
OMP_PARALLEL_COMBINED (expr)
? ORT_COMBINED_PARALLEL
: ORT_PARALLEL, OMP_PARALLEL);
push_gimplify_context ();
g = gimplify_and_return_first (OMP_PARALLEL_BODY (expr), &body);
if (gimple_code (g) == GIMPLE_BIND)
pop_gimplify_context (g);
else
pop_gimplify_context (NULL);
gimplify_adjust_omp_clauses (pre_p, body, &OMP_PARALLEL_CLAUSES (expr),
OMP_PARALLEL);
g = gimple_build_omp_parallel (body,
OMP_PARALLEL_CLAUSES (expr),
NULL_TREE, NULL_TREE);
if (OMP_PARALLEL_COMBINED (expr))
gimple_omp_set_subcode (g, GF_OMP_PARALLEL_COMBINED);
gimplify_seq_add_stmt (pre_p, g);
*expr_p = NULL_TREE;
}
static void
gimplify_omp_task (tree *expr_p, gimple_seq *pre_p)
{
tree expr = *expr_p;
gimple *g;
gimple_seq body = NULL;
gimplify_scan_omp_clauses (&OMP_TASK_CLAUSES (expr), pre_p,
omp_find_clause (OMP_TASK_CLAUSES (expr),
OMP_CLAUSE_UNTIED)
? ORT_UNTIED_TASK : ORT_TASK, OMP_TASK);
push_gimplify_context ();
g = gimplify_and_return_first (OMP_TASK_BODY (expr), &body);
if (gimple_code (g) == GIMPLE_BIND)
pop_gimplify_context (g);
else
pop_gimplify_context (NULL);
gimplify_adjust_omp_clauses (pre_p, body, &OMP_TASK_CLAUSES (expr),
OMP_TASK);
g = gimple_build_omp_task (body,
OMP_TASK_CLAUSES (expr),
NULL_TREE, NULL_TREE,
NULL_TREE, NULL_TREE, NULL_TREE);
gimplify_seq_add_stmt (pre_p, g);
*expr_p = NULL_TREE;
}
static tree
find_combined_omp_for (tree *tp, int *walk_subtrees, void *)
{
*walk_subtrees = 0;
switch (TREE_CODE (*tp))
{
case OMP_FOR:
*walk_subtrees = 1;
case OMP_SIMD:
if (OMP_FOR_INIT (*tp) != NULL_TREE)
return *tp;
break;
case BIND_EXPR:
case STATEMENT_LIST:
case OMP_PARALLEL:
*walk_subtrees = 1;
break;
default:
break;
}
return NULL_TREE;
}
static enum gimplify_status
gimplify_omp_for (tree *expr_p, gimple_seq *pre_p)
{
tree for_stmt, orig_for_stmt, inner_for_stmt = NULL_TREE, decl, var, t;
enum gimplify_status ret = GS_ALL_DONE;
enum gimplify_status tret;
gomp_for *gfor;
gimple_seq for_body, for_pre_body;
int i;
bitmap has_decl_expr = NULL;
enum omp_region_type ort = ORT_WORKSHARE;
orig_for_stmt = for_stmt = *expr_p;
switch (TREE_CODE (for_stmt))
{
case OMP_FOR:
case OMP_DISTRIBUTE:
break;
case OACC_LOOP:
ort = ORT_ACC;
break;
case OMP_TASKLOOP:
if (omp_find_clause (OMP_FOR_CLAUSES (for_stmt), OMP_CLAUSE_UNTIED))
ort = ORT_UNTIED_TASK;
else
ort = ORT_TASK;
break;
case OMP_SIMD:
ort = ORT_SIMD;
break;
default:
gcc_unreachable ();
}
if (ort == ORT_SIMD && TREE_VEC_LENGTH (OMP_FOR_INIT (for_stmt)) == 1)
{
t = TREE_VEC_ELT (OMP_FOR_INIT (for_stmt), 0);
gcc_assert (TREE_CODE (t) == MODIFY_EXPR);
decl = TREE_OPERAND (t, 0);
for (tree c = OMP_FOR_CLAUSES (for_stmt); c; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LINEAR
&& OMP_CLAUSE_DECL (c) == decl)
{
OMP_CLAUSE_LINEAR_NO_COPYIN (c) = 1;
break;
}
}
if (OMP_FOR_INIT (for_stmt) == NULL_TREE)
{
gcc_assert (TREE_CODE (for_stmt) != OACC_LOOP);
inner_for_stmt = walk_tree (&OMP_FOR_BODY (for_stmt),
find_combined_omp_for, NULL, NULL);
if (inner_for_stmt == NULL_TREE)
{
gcc_assert (seen_error ());
*expr_p = NULL_TREE;
return GS_ERROR;
}
}
if (TREE_CODE (for_stmt) != OMP_TASKLOOP)
gimplify_scan_omp_clauses (&OMP_FOR_CLAUSES (for_stmt), pre_p, ort,
TREE_CODE (for_stmt));
if (TREE_CODE (for_stmt) == OMP_DISTRIBUTE)
gimplify_omp_ctxp->distribute = true;
for_pre_body = NULL;
if (ort == ORT_SIMD && OMP_FOR_PRE_BODY (for_stmt))
{
has_decl_expr = BITMAP_ALLOC (NULL);
if (TREE_CODE (OMP_FOR_PRE_BODY (for_stmt)) == DECL_EXPR
&& TREE_CODE (DECL_EXPR_DECL (OMP_FOR_PRE_BODY (for_stmt)))
== VAR_DECL)
{
t = OMP_FOR_PRE_BODY (for_stmt);
bitmap_set_bit (has_decl_expr, DECL_UID (DECL_EXPR_DECL (t)));
}
else if (TREE_CODE (OMP_FOR_PRE_BODY (for_stmt)) == STATEMENT_LIST)
{
tree_stmt_iterator si;
for (si = tsi_start (OMP_FOR_PRE_BODY (for_stmt)); !tsi_end_p (si);
tsi_next (&si))
{
t = tsi_stmt (si);
if (TREE_CODE (t) == DECL_EXPR
&& TREE_CODE (DECL_EXPR_DECL (t)) == VAR_DECL)
bitmap_set_bit (has_decl_expr, DECL_UID (DECL_EXPR_DECL (t)));
}
}
}
if (OMP_FOR_PRE_BODY (for_stmt))
{
if (TREE_CODE (for_stmt) != OMP_TASKLOOP || gimplify_omp_ctxp)
gimplify_and_add (OMP_FOR_PRE_BODY (for_stmt), &for_pre_body);
else
{
struct gimplify_omp_ctx ctx;
memset (&ctx, 0, sizeof (ctx));
ctx.region_type = ORT_NONE;
gimplify_omp_ctxp = &ctx;
gimplify_and_add (OMP_FOR_PRE_BODY (for_stmt), &for_pre_body);
gimplify_omp_ctxp = NULL;
}
}
OMP_FOR_PRE_BODY (for_stmt) = NULL_TREE;
if (OMP_FOR_INIT (for_stmt) == NULL_TREE)
for_stmt = inner_for_stmt;
if (TREE_CODE (orig_for_stmt) == OMP_TASKLOOP)
{
for (i = 0; i < TREE_VEC_LENGTH (OMP_FOR_INIT (for_stmt)); i++)
{
t = TREE_VEC_ELT (OMP_FOR_INIT (for_stmt), i);
if (!is_gimple_constant (TREE_OPERAND (t, 1)))
{
tree type = TREE_TYPE (TREE_OPERAND (t, 0));
TREE_OPERAND (t, 1)
= get_initialized_tmp_var (TREE_OPERAND (t, 1),
gimple_seq_empty_p (for_pre_body)
? pre_p : &for_pre_body, NULL,
false);
if (TREE_CODE (type) == POINTER_TYPE
&& (TREE_CODE (TREE_TYPE (TREE_OPERAND (t, 1)))
== REFERENCE_TYPE))
{
tree v = create_tmp_var (TYPE_MAIN_VARIANT (type));
tree m = build2 (INIT_EXPR, TREE_TYPE (v), v,
TREE_OPERAND (t, 1));
gimplify_and_add (m, gimple_seq_empty_p (for_pre_body)
? pre_p : &for_pre_body);
TREE_OPERAND (t, 1) = v;
}
tree c = build_omp_clause (input_location,
OMP_CLAUSE_FIRSTPRIVATE);
OMP_CLAUSE_DECL (c) = TREE_OPERAND (t, 1);
OMP_CLAUSE_CHAIN (c) = OMP_FOR_CLAUSES (orig_for_stmt);
OMP_FOR_CLAUSES (orig_for_stmt) = c;
}
t = TREE_VEC_ELT (OMP_FOR_COND (for_stmt), i);
if (!is_gimple_constant (TREE_OPERAND (t, 1)))
{
tree type = TREE_TYPE (TREE_OPERAND (t, 0));
TREE_OPERAND (t, 1)
= get_initialized_tmp_var (TREE_OPERAND (t, 1),
gimple_seq_empty_p (for_pre_body)
? pre_p : &for_pre_body, NULL,
false);
if (TREE_CODE (type) == POINTER_TYPE
&& (TREE_CODE (TREE_TYPE (TREE_OPERAND (t, 1)))
== REFERENCE_TYPE))
{
tree v = create_tmp_var (TYPE_MAIN_VARIANT (type));
tree m = build2 (INIT_EXPR, TREE_TYPE (v), v,
TREE_OPERAND (t, 1));
gimplify_and_add (m, gimple_seq_empty_p (for_pre_body)
? pre_p : &for_pre_body);
TREE_OPERAND (t, 1) = v;
}
tree c = build_omp_clause (input_location,
OMP_CLAUSE_FIRSTPRIVATE);
OMP_CLAUSE_DECL (c) = TREE_OPERAND (t, 1);
OMP_CLAUSE_CHAIN (c) = OMP_FOR_CLAUSES (orig_for_stmt);
OMP_FOR_CLAUSES (orig_for_stmt) = c;
}
t = TREE_VEC_ELT (OMP_FOR_INCR (for_stmt), i);
if (TREE_CODE (t) == MODIFY_EXPR)
{
decl = TREE_OPERAND (t, 0);
t = TREE_OPERAND (t, 1);
tree *tp = &TREE_OPERAND (t, 1);
if (TREE_CODE (t) == PLUS_EXPR && *tp == decl)
tp = &TREE_OPERAND (t, 0);
if (!is_gimple_constant (*tp))
{
gimple_seq *seq = gimple_seq_empty_p (for_pre_body)
? pre_p : &for_pre_body;
*tp = get_initialized_tmp_var (*tp, seq, NULL, false);
tree c = build_omp_clause (input_location,
OMP_CLAUSE_FIRSTPRIVATE);
OMP_CLAUSE_DECL (c) = *tp;
OMP_CLAUSE_CHAIN (c) = OMP_FOR_CLAUSES (orig_for_stmt);
OMP_FOR_CLAUSES (orig_for_stmt) = c;
}
}
}
gimplify_scan_omp_clauses (&OMP_FOR_CLAUSES (orig_for_stmt), pre_p, ort,
OMP_TASKLOOP);
}
if (orig_for_stmt != for_stmt)
gimplify_omp_ctxp->combined_loop = true;
for_body = NULL;
gcc_assert (TREE_VEC_LENGTH (OMP_FOR_INIT (for_stmt))
== TREE_VEC_LENGTH (OMP_FOR_COND (for_stmt)));
gcc_assert (TREE_VEC_LENGTH (OMP_FOR_INIT (for_stmt))
== TREE_VEC_LENGTH (OMP_FOR_INCR (for_stmt)));
tree c = omp_find_clause (OMP_FOR_CLAUSES (for_stmt), OMP_CLAUSE_ORDERED);
bool is_doacross = false;
if (c && OMP_CLAUSE_ORDERED_EXPR (c))
{
is_doacross = true;
gimplify_omp_ctxp->loop_iter_var.create (TREE_VEC_LENGTH
(OMP_FOR_INIT (for_stmt))
* 2);
}
int collapse = 1, tile = 0;
c = omp_find_clause (OMP_FOR_CLAUSES (for_stmt), OMP_CLAUSE_COLLAPSE);
if (c)
collapse = tree_to_shwi (OMP_CLAUSE_COLLAPSE_EXPR (c));
c = omp_find_clause (OMP_FOR_CLAUSES (for_stmt), OMP_CLAUSE_TILE);
if (c)
tile = list_length (OMP_CLAUSE_TILE_LIST (c));
for (i = 0; i < TREE_VEC_LENGTH (OMP_FOR_INIT (for_stmt)); i++)
{
t = TREE_VEC_ELT (OMP_FOR_INIT (for_stmt), i);
gcc_assert (TREE_CODE (t) == MODIFY_EXPR);
decl = TREE_OPERAND (t, 0);
gcc_assert (DECL_P (decl));
gcc_assert (INTEGRAL_TYPE_P (TREE_TYPE (decl))
|| POINTER_TYPE_P (TREE_TYPE (decl)));
if (is_doacross)
{
if (TREE_CODE (for_stmt) == OMP_FOR && OMP_FOR_ORIG_DECLS (for_stmt))
gimplify_omp_ctxp->loop_iter_var.quick_push
(TREE_VEC_ELT (OMP_FOR_ORIG_DECLS (for_stmt), i));
else
gimplify_omp_ctxp->loop_iter_var.quick_push (decl);
gimplify_omp_ctxp->loop_iter_var.quick_push (decl);
}
tree c = NULL_TREE;
tree c2 = NULL_TREE;
if (orig_for_stmt != for_stmt)
;
else if (ort == ORT_SIMD)
{
splay_tree_node n = splay_tree_lookup (gimplify_omp_ctxp->variables,
(splay_tree_key) decl);
omp_is_private (gimplify_omp_ctxp, decl,
1 + (TREE_VEC_LENGTH (OMP_FOR_INIT (for_stmt))
!= 1));
if (n != NULL && (n->value & GOVD_DATA_SHARE_CLASS) != 0)
omp_notice_variable (gimplify_omp_ctxp, decl, true);
else if (TREE_VEC_LENGTH (OMP_FOR_INIT (for_stmt)) == 1)
{
c = build_omp_clause (input_location, OMP_CLAUSE_LINEAR);
OMP_CLAUSE_LINEAR_NO_COPYIN (c) = 1;
unsigned int flags = GOVD_LINEAR | GOVD_EXPLICIT | GOVD_SEEN;
if (has_decl_expr
&& bitmap_bit_p (has_decl_expr, DECL_UID (decl)))
{
OMP_CLAUSE_LINEAR_NO_COPYOUT (c) = 1;
flags |= GOVD_LINEAR_LASTPRIVATE_NO_OUTER;
}
struct gimplify_omp_ctx *outer
= gimplify_omp_ctxp->outer_context;
if (outer && !OMP_CLAUSE_LINEAR_NO_COPYOUT (c))
{
if (outer->region_type == ORT_WORKSHARE
&& outer->combined_loop)
{
n = splay_tree_lookup (outer->variables,
(splay_tree_key)decl);
if (n != NULL && (n->value & GOVD_LOCAL) != 0)
{
OMP_CLAUSE_LINEAR_NO_COPYOUT (c) = 1;
flags |= GOVD_LINEAR_LASTPRIVATE_NO_OUTER;
}
else
{
struct gimplify_omp_ctx *octx = outer->outer_context;
if (octx
&& octx->region_type == ORT_COMBINED_PARALLEL
&& octx->outer_context
&& (octx->outer_context->region_type
== ORT_WORKSHARE)
&& octx->outer_context->combined_loop)
{
octx = octx->outer_context;
n = splay_tree_lookup (octx->variables,
(splay_tree_key)decl);
if (n != NULL && (n->value & GOVD_LOCAL) != 0)
{
OMP_CLAUSE_LINEAR_NO_COPYOUT (c) = 1;
flags |= GOVD_LINEAR_LASTPRIVATE_NO_OUTER;
}
}
}
}
}
OMP_CLAUSE_DECL (c) = decl;
OMP_CLAUSE_CHAIN (c) = OMP_FOR_CLAUSES (for_stmt);
OMP_FOR_CLAUSES (for_stmt) = c;
omp_add_variable (gimplify_omp_ctxp, decl, flags);
if (outer && !OMP_CLAUSE_LINEAR_NO_COPYOUT (c))
{
if (outer->region_type == ORT_WORKSHARE
&& outer->combined_loop)
{
if (outer->outer_context
&& (outer->outer_context->region_type
== ORT_COMBINED_PARALLEL))
outer = outer->outer_context;
else if (omp_check_private (outer, decl, false))
outer = NULL;
}
else if (((outer->region_type & ORT_TASK) != 0)
&& outer->combined_loop
&& !omp_check_private (gimplify_omp_ctxp,
decl, false))
;
else if (outer->region_type != ORT_COMBINED_PARALLEL)
{
omp_notice_variable (outer, decl, true);
outer = NULL;
}
if (outer)
{
n = splay_tree_lookup (outer->variables,
(splay_tree_key)decl);
if (n == NULL || (n->value & GOVD_DATA_SHARE_CLASS) == 0)
{
omp_add_variable (outer, decl,
GOVD_LASTPRIVATE | GOVD_SEEN);
if (outer->region_type == ORT_COMBINED_PARALLEL
&& outer->outer_context
&& (outer->outer_context->region_type
== ORT_WORKSHARE)
&& outer->outer_context->combined_loop)
{
outer = outer->outer_context;
n = splay_tree_lookup (outer->variables,
(splay_tree_key)decl);
if (omp_check_private (outer, decl, false))
outer = NULL;
else if (n == NULL
|| ((n->value & GOVD_DATA_SHARE_CLASS)
== 0))
omp_add_variable (outer, decl,
GOVD_LASTPRIVATE
| GOVD_SEEN);
else
outer = NULL;
}
if (outer && outer->outer_context
&& (outer->outer_context->region_type
== ORT_COMBINED_TEAMS))
{
outer = outer->outer_context;
n = splay_tree_lookup (outer->variables,
(splay_tree_key)decl);
if (n == NULL
|| (n->value & GOVD_DATA_SHARE_CLASS) == 0)
omp_add_variable (outer, decl,
GOVD_SHARED | GOVD_SEEN);
else
outer = NULL;
}
if (outer && outer->outer_context)
omp_notice_variable (outer->outer_context, decl,
true);
}
}
}
}
else
{
bool lastprivate
= (!has_decl_expr
|| !bitmap_bit_p (has_decl_expr, DECL_UID (decl)));
struct gimplify_omp_ctx *outer
= gimplify_omp_ctxp->outer_context;
if (outer && lastprivate)
{
if (outer->region_type == ORT_WORKSHARE
&& outer->combined_loop)
{
n = splay_tree_lookup (outer->variables,
(splay_tree_key)decl);
if (n != NULL && (n->value & GOVD_LOCAL) != 0)
{
lastprivate = false;
outer = NULL;
}
else if (outer->outer_context
&& (outer->outer_context->region_type
== ORT_COMBINED_PARALLEL))
outer = outer->outer_context;
else if (omp_check_private (outer, decl, false))
outer = NULL;
}
else if (((outer->region_type & ORT_TASK) != 0)
&& outer->combined_loop
&& !omp_check_private (gimplify_omp_ctxp,
decl, false))
;
else if (outer->region_type != ORT_COMBINED_PARALLEL)
{
omp_notice_variable (outer, decl, true);
outer = NULL;
}
if (outer)
{
n = splay_tree_lookup (outer->variables,
(splay_tree_key)decl);
if (n == NULL || (n->value & GOVD_DATA_SHARE_CLASS) == 0)
{
omp_add_variable (outer, decl,
GOVD_LASTPRIVATE | GOVD_SEEN);
if (outer->region_type == ORT_COMBINED_PARALLEL
&& outer->outer_context
&& (outer->outer_context->region_type
== ORT_WORKSHARE)
&& outer->outer_context->combined_loop)
{
outer = outer->outer_context;
n = splay_tree_lookup (outer->variables,
(splay_tree_key)decl);
if (omp_check_private (outer, decl, false))
outer = NULL;
else if (n == NULL
|| ((n->value & GOVD_DATA_SHARE_CLASS)
== 0))
omp_add_variable (outer, decl,
GOVD_LASTPRIVATE
| GOVD_SEEN);
else
outer = NULL;
}
if (outer && outer->outer_context
&& (outer->outer_context->region_type
== ORT_COMBINED_TEAMS))
{
outer = outer->outer_context;
n = splay_tree_lookup (outer->variables,
(splay_tree_key)decl);
if (n == NULL
|| (n->value & GOVD_DATA_SHARE_CLASS) == 0)
omp_add_variable (outer, decl,
GOVD_SHARED | GOVD_SEEN);
else
outer = NULL;
}
if (outer && outer->outer_context)
omp_notice_variable (outer->outer_context, decl,
true);
}
}
}
c = build_omp_clause (input_location,
lastprivate ? OMP_CLAUSE_LASTPRIVATE
: OMP_CLAUSE_PRIVATE);
OMP_CLAUSE_DECL (c) = decl;
OMP_CLAUSE_CHAIN (c) = OMP_FOR_CLAUSES (for_stmt);
OMP_FOR_CLAUSES (for_stmt) = c;
omp_add_variable (gimplify_omp_ctxp, decl,
(lastprivate ? GOVD_LASTPRIVATE : GOVD_PRIVATE)
| GOVD_EXPLICIT | GOVD_SEEN);
c = NULL_TREE;
}
}
else if (omp_is_private (gimplify_omp_ctxp, decl, 0))
omp_notice_variable (gimplify_omp_ctxp, decl, true);
else
omp_add_variable (gimplify_omp_ctxp, decl, GOVD_PRIVATE | GOVD_SEEN);
if (orig_for_stmt != for_stmt)
var = decl;
else if (!is_gimple_reg (decl)
|| (ort == ORT_SIMD
&& TREE_VEC_LENGTH (OMP_FOR_INIT (for_stmt)) > 1))
{
struct gimplify_omp_ctx *ctx = gimplify_omp_ctxp;
gimplify_omp_ctxp = NULL;
var = create_tmp_var (TREE_TYPE (decl), get_name (decl));
gimplify_omp_ctxp = ctx;
TREE_OPERAND (t, 0) = var;
gimplify_seq_add_stmt (&for_body, gimple_build_assign (decl, var));
if (ort == ORT_SIMD
&& TREE_VEC_LENGTH (OMP_FOR_INIT (for_stmt)) == 1)
{
c2 = build_omp_clause (input_location, OMP_CLAUSE_LINEAR);
OMP_CLAUSE_LINEAR_NO_COPYIN (c2) = 1;
OMP_CLAUSE_LINEAR_NO_COPYOUT (c2) = 1;
OMP_CLAUSE_DECL (c2) = var;
OMP_CLAUSE_CHAIN (c2) = OMP_FOR_CLAUSES (for_stmt);
OMP_FOR_CLAUSES (for_stmt) = c2;
omp_add_variable (gimplify_omp_ctxp, var,
GOVD_LINEAR | GOVD_EXPLICIT | GOVD_SEEN);
if (c == NULL_TREE)
{
c = c2;
c2 = NULL_TREE;
}
}
else
omp_add_variable (gimplify_omp_ctxp, var,
GOVD_PRIVATE | GOVD_SEEN);
}
else
var = decl;
tret = gimplify_expr (&TREE_OPERAND (t, 1), &for_pre_body, NULL,
is_gimple_val, fb_rvalue, false);
ret = MIN (ret, tret);
if (ret == GS_ERROR)
return ret;
t = TREE_VEC_ELT (OMP_FOR_COND (for_stmt), i);
gcc_assert (COMPARISON_CLASS_P (t));
gcc_assert (TREE_OPERAND (t, 0) == decl);
tret = gimplify_expr (&TREE_OPERAND (t, 1), &for_pre_body, NULL,
is_gimple_val, fb_rvalue, false);
ret = MIN (ret, tret);
t = TREE_VEC_ELT (OMP_FOR_INCR (for_stmt), i);
switch (TREE_CODE (t))
{
case PREINCREMENT_EXPR:
case POSTINCREMENT_EXPR:
{
tree decl = TREE_OPERAND (t, 0);
gcc_assert (!POINTER_TYPE_P (TREE_TYPE (decl)));
if (orig_for_stmt != for_stmt)
break;
t = build_int_cst (TREE_TYPE (decl), 1);
if (c)
OMP_CLAUSE_LINEAR_STEP (c) = t;
t = build2 (PLUS_EXPR, TREE_TYPE (decl), var, t);
t = build2 (MODIFY_EXPR, TREE_TYPE (var), var, t);
TREE_VEC_ELT (OMP_FOR_INCR (for_stmt), i) = t;
break;
}
case PREDECREMENT_EXPR:
case POSTDECREMENT_EXPR:
gcc_assert (!POINTER_TYPE_P (TREE_TYPE (decl)));
if (orig_for_stmt != for_stmt)
break;
t = build_int_cst (TREE_TYPE (decl), -1);
if (c)
OMP_CLAUSE_LINEAR_STEP (c) = t;
t = build2 (PLUS_EXPR, TREE_TYPE (decl), var, t);
t = build2 (MODIFY_EXPR, TREE_TYPE (var), var, t);
TREE_VEC_ELT (OMP_FOR_INCR (for_stmt), i) = t;
break;
case MODIFY_EXPR:
gcc_assert (TREE_OPERAND (t, 0) == decl);
TREE_OPERAND (t, 0) = var;
t = TREE_OPERAND (t, 1);
switch (TREE_CODE (t))
{
case PLUS_EXPR:
if (TREE_OPERAND (t, 1) == decl)
{
TREE_OPERAND (t, 1) = TREE_OPERAND (t, 0);
TREE_OPERAND (t, 0) = var;
break;
}
case MINUS_EXPR:
case POINTER_PLUS_EXPR:
gcc_assert (TREE_OPERAND (t, 0) == decl);
TREE_OPERAND (t, 0) = var;
break;
default:
gcc_unreachable ();
}
tret = gimplify_expr (&TREE_OPERAND (t, 1), &for_pre_body, NULL,
is_gimple_val, fb_rvalue, false);
ret = MIN (ret, tret);
if (c)
{
tree step = TREE_OPERAND (t, 1);
tree stept = TREE_TYPE (decl);
if (POINTER_TYPE_P (stept))
stept = sizetype;
step = fold_convert (stept, step);
if (TREE_CODE (t) == MINUS_EXPR)
step = fold_build1 (NEGATE_EXPR, stept, step);
OMP_CLAUSE_LINEAR_STEP (c) = step;
if (step != TREE_OPERAND (t, 1))
{
tret = gimplify_expr (&OMP_CLAUSE_LINEAR_STEP (c),
&for_pre_body, NULL,
is_gimple_val, fb_rvalue, false);
ret = MIN (ret, tret);
}
}
break;
default:
gcc_unreachable ();
}
if (c2)
{
gcc_assert (c);
OMP_CLAUSE_LINEAR_STEP (c2) = OMP_CLAUSE_LINEAR_STEP (c);
}
if ((var != decl || collapse > 1 || tile) && orig_for_stmt == for_stmt)
{
for (c = OMP_FOR_CLAUSES (for_stmt); c ; c = OMP_CLAUSE_CHAIN (c))
if (((OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LASTPRIVATE
&& OMP_CLAUSE_LASTPRIVATE_GIMPLE_SEQ (c) == NULL)
|| (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LINEAR
&& !OMP_CLAUSE_LINEAR_NO_COPYOUT (c)
&& OMP_CLAUSE_LINEAR_GIMPLE_SEQ (c) == NULL))
&& OMP_CLAUSE_DECL (c) == decl)
{
if (is_doacross && (collapse == 1 || i >= collapse))
t = var;
else
{
t = TREE_VEC_ELT (OMP_FOR_INCR (for_stmt), i);
gcc_assert (TREE_CODE (t) == MODIFY_EXPR);
gcc_assert (TREE_OPERAND (t, 0) == var);
t = TREE_OPERAND (t, 1);
gcc_assert (TREE_CODE (t) == PLUS_EXPR
|| TREE_CODE (t) == MINUS_EXPR
|| TREE_CODE (t) == POINTER_PLUS_EXPR);
gcc_assert (TREE_OPERAND (t, 0) == var);
t = build2 (TREE_CODE (t), TREE_TYPE (decl),
is_doacross ? var : decl,
TREE_OPERAND (t, 1));
}
gimple_seq *seq;
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LASTPRIVATE)
seq = &OMP_CLAUSE_LASTPRIVATE_GIMPLE_SEQ (c);
else
seq = &OMP_CLAUSE_LINEAR_GIMPLE_SEQ (c);
push_gimplify_context ();
gimplify_assign (decl, t, seq);
gimple *bind = NULL;
if (gimplify_ctxp->temps)
{
bind = gimple_build_bind (NULL_TREE, *seq, NULL_TREE);
*seq = NULL;
gimplify_seq_add_stmt (seq, bind);
}
pop_gimplify_context (bind);
}
}
}
BITMAP_FREE (has_decl_expr);
if (TREE_CODE (orig_for_stmt) == OMP_TASKLOOP)
{
push_gimplify_context ();
if (TREE_CODE (OMP_FOR_BODY (orig_for_stmt)) != BIND_EXPR)
{
OMP_FOR_BODY (orig_for_stmt)
= build3 (BIND_EXPR, void_type_node, NULL,
OMP_FOR_BODY (orig_for_stmt), NULL);
TREE_SIDE_EFFECTS (OMP_FOR_BODY (orig_for_stmt)) = 1;
}
}
gimple *g = gimplify_and_return_first (OMP_FOR_BODY (orig_for_stmt),
&for_body);
if (TREE_CODE (orig_for_stmt) == OMP_TASKLOOP)
{
if (gimple_code (g) == GIMPLE_BIND)
pop_gimplify_context (g);
else
pop_gimplify_context (NULL);
}
if (orig_for_stmt != for_stmt)
for (i = 0; i < TREE_VEC_LENGTH (OMP_FOR_INIT (for_stmt)); i++)
{
t = TREE_VEC_ELT (OMP_FOR_INIT (for_stmt), i);
decl = TREE_OPERAND (t, 0);
struct gimplify_omp_ctx *ctx = gimplify_omp_ctxp;
if (TREE_CODE (orig_for_stmt) == OMP_TASKLOOP)
gimplify_omp_ctxp = ctx->outer_context;
var = create_tmp_var (TREE_TYPE (decl), get_name (decl));
gimplify_omp_ctxp = ctx;
omp_add_variable (gimplify_omp_ctxp, var, GOVD_PRIVATE | GOVD_SEEN);
TREE_OPERAND (t, 0) = var;
t = TREE_VEC_ELT (OMP_FOR_INCR (for_stmt), i);
TREE_OPERAND (t, 1) = copy_node (TREE_OPERAND (t, 1));
TREE_OPERAND (TREE_OPERAND (t, 1), 0) = var;
}
gimplify_adjust_omp_clauses (pre_p, for_body,
&OMP_FOR_CLAUSES (orig_for_stmt),
TREE_CODE (orig_for_stmt));
int kind;
switch (TREE_CODE (orig_for_stmt))
{
case OMP_FOR: kind = GF_OMP_FOR_KIND_FOR; break;
case OMP_SIMD: kind = GF_OMP_FOR_KIND_SIMD; break;
case OMP_DISTRIBUTE: kind = GF_OMP_FOR_KIND_DISTRIBUTE; break;
case OMP_TASKLOOP: kind = GF_OMP_FOR_KIND_TASKLOOP; break;
case OACC_LOOP: kind = GF_OMP_FOR_KIND_OACC_LOOP; break;
default:
gcc_unreachable ();
}
gfor = gimple_build_omp_for (for_body, kind, OMP_FOR_CLAUSES (orig_for_stmt),
TREE_VEC_LENGTH (OMP_FOR_INIT (for_stmt)),
for_pre_body);
if (orig_for_stmt != for_stmt)
gimple_omp_for_set_combined_p (gfor, true);
if (gimplify_omp_ctxp
&& (gimplify_omp_ctxp->combined_loop
|| (gimplify_omp_ctxp->region_type == ORT_COMBINED_PARALLEL
&& gimplify_omp_ctxp->outer_context
&& gimplify_omp_ctxp->outer_context->combined_loop)))
{
gimple_omp_for_set_combined_into_p (gfor, true);
if (gimplify_omp_ctxp->combined_loop)
gcc_assert (TREE_CODE (orig_for_stmt) == OMP_SIMD);
else
gcc_assert (TREE_CODE (orig_for_stmt) == OMP_FOR);
}
for (i = 0; i < TREE_VEC_LENGTH (OMP_FOR_INIT (for_stmt)); i++)
{
t = TREE_VEC_ELT (OMP_FOR_INIT (for_stmt), i);
gimple_omp_for_set_index (gfor, i, TREE_OPERAND (t, 0));
gimple_omp_for_set_initial (gfor, i, TREE_OPERAND (t, 1));
t = TREE_VEC_ELT (OMP_FOR_COND (for_stmt), i);
gimple_omp_for_set_cond (gfor, i, TREE_CODE (t));
gimple_omp_for_set_final (gfor, i, TREE_OPERAND (t, 1));
t = TREE_VEC_ELT (OMP_FOR_INCR (for_stmt), i);
gimple_omp_for_set_incr (gfor, i, TREE_OPERAND (t, 1));
}
if (TREE_CODE (orig_for_stmt) == OMP_TASKLOOP)
{
tree *gfor_clauses_ptr = gimple_omp_for_clauses_ptr (gfor);
tree task_clauses = NULL_TREE;
tree c = *gfor_clauses_ptr;
tree *gtask_clauses_ptr = &task_clauses;
tree outer_for_clauses = NULL_TREE;
tree *gforo_clauses_ptr = &outer_for_clauses;
for (; c; c = OMP_CLAUSE_CHAIN (c))
switch (OMP_CLAUSE_CODE (c))
{
case OMP_CLAUSE_SHARED:
case OMP_CLAUSE_FIRSTPRIVATE:
case OMP_CLAUSE_DEFAULT:
case OMP_CLAUSE_IF:
case OMP_CLAUSE_UNTIED:
case OMP_CLAUSE_FINAL:
case OMP_CLAUSE_MERGEABLE:
case OMP_CLAUSE_PRIORITY:
*gtask_clauses_ptr = c;
gtask_clauses_ptr = &OMP_CLAUSE_CHAIN (c);
break;
case OMP_CLAUSE_PRIVATE:
if (OMP_CLAUSE_PRIVATE_TASKLOOP_IV (c))
{
*gtask_clauses_ptr
= build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_FIRSTPRIVATE);
OMP_CLAUSE_DECL (*gtask_clauses_ptr) = OMP_CLAUSE_DECL (c);
lang_hooks.decls.omp_finish_clause (*gtask_clauses_ptr, NULL);
gtask_clauses_ptr = &OMP_CLAUSE_CHAIN (*gtask_clauses_ptr);
*gforo_clauses_ptr = c;
gforo_clauses_ptr = &OMP_CLAUSE_CHAIN (c);
}
else
{
*gtask_clauses_ptr = c;
gtask_clauses_ptr = &OMP_CLAUSE_CHAIN (c);
}
break;
case OMP_CLAUSE_GRAINSIZE:
case OMP_CLAUSE_NUM_TASKS:
case OMP_CLAUSE_NOGROUP:
*gforo_clauses_ptr = c;
gforo_clauses_ptr = &OMP_CLAUSE_CHAIN (c);
break;
case OMP_CLAUSE_COLLAPSE:
*gfor_clauses_ptr = c;
gfor_clauses_ptr = &OMP_CLAUSE_CHAIN (c);
*gforo_clauses_ptr = copy_node (c);
gforo_clauses_ptr = &OMP_CLAUSE_CHAIN (*gforo_clauses_ptr);
break;
case OMP_CLAUSE_LASTPRIVATE:
if (OMP_CLAUSE_LASTPRIVATE_TASKLOOP_IV (c))
{
*gtask_clauses_ptr
= build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_FIRSTPRIVATE);
OMP_CLAUSE_DECL (*gtask_clauses_ptr) = OMP_CLAUSE_DECL (c);
lang_hooks.decls.omp_finish_clause (*gtask_clauses_ptr, NULL);
gtask_clauses_ptr = &OMP_CLAUSE_CHAIN (*gtask_clauses_ptr);
OMP_CLAUSE_LASTPRIVATE_FIRSTPRIVATE (c) = 1;
*gforo_clauses_ptr = build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_PRIVATE);
OMP_CLAUSE_DECL (*gforo_clauses_ptr) = OMP_CLAUSE_DECL (c);
OMP_CLAUSE_PRIVATE_TASKLOOP_IV (*gforo_clauses_ptr) = 1;
TREE_TYPE (*gforo_clauses_ptr) = TREE_TYPE (c);
gforo_clauses_ptr = &OMP_CLAUSE_CHAIN (*gforo_clauses_ptr);
}
*gfor_clauses_ptr = c;
gfor_clauses_ptr = &OMP_CLAUSE_CHAIN (c);
*gtask_clauses_ptr
= build_omp_clause (OMP_CLAUSE_LOCATION (c), OMP_CLAUSE_SHARED);
OMP_CLAUSE_DECL (*gtask_clauses_ptr) = OMP_CLAUSE_DECL (c);
if (OMP_CLAUSE_LASTPRIVATE_FIRSTPRIVATE (c))
OMP_CLAUSE_SHARED_FIRSTPRIVATE (*gtask_clauses_ptr) = 1;
gtask_clauses_ptr
= &OMP_CLAUSE_CHAIN (*gtask_clauses_ptr);
break;
default:
gcc_unreachable ();
}
*gfor_clauses_ptr = NULL_TREE;
*gtask_clauses_ptr = NULL_TREE;
*gforo_clauses_ptr = NULL_TREE;
g = gimple_build_bind (NULL_TREE, gfor, NULL_TREE);
g = gimple_build_omp_task (g, task_clauses, NULL_TREE, NULL_TREE,
NULL_TREE, NULL_TREE, NULL_TREE);
gimple_omp_task_set_taskloop_p (g, true);
g = gimple_build_bind (NULL_TREE, g, NULL_TREE);
gomp_for *gforo
= gimple_build_omp_for (g, GF_OMP_FOR_KIND_TASKLOOP, outer_for_clauses,
gimple_omp_for_collapse (gfor),
gimple_omp_for_pre_body (gfor));
gimple_omp_for_set_pre_body (gfor, NULL);
gimple_omp_for_set_combined_p (gforo, true);
gimple_omp_for_set_combined_into_p (gfor, true);
for (i = 0; i < (int) gimple_omp_for_collapse (gfor); i++)
{
tree type = TREE_TYPE (gimple_omp_for_index (gfor, i));
tree v = create_tmp_var (type);
gimple_omp_for_set_index (gforo, i, v);
t = unshare_expr (gimple_omp_for_initial (gfor, i));
gimple_omp_for_set_initial (gforo, i, t);
gimple_omp_for_set_cond (gforo, i,
gimple_omp_for_cond (gfor, i));
t = unshare_expr (gimple_omp_for_final (gfor, i));
gimple_omp_for_set_final (gforo, i, t);
t = unshare_expr (gimple_omp_for_incr (gfor, i));
gcc_assert (TREE_OPERAND (t, 0) == gimple_omp_for_index (gfor, i));
TREE_OPERAND (t, 0) = v;
gimple_omp_for_set_incr (gforo, i, t);
t = build_omp_clause (input_location, OMP_CLAUSE_PRIVATE);
OMP_CLAUSE_DECL (t) = v;
OMP_CLAUSE_CHAIN (t) = gimple_omp_for_clauses (gforo);
gimple_omp_for_set_clauses (gforo, t);
}
gimplify_seq_add_stmt (pre_p, gforo);
}
else
gimplify_seq_add_stmt (pre_p, gfor);
if (ret != GS_ALL_DONE)
return GS_ERROR;
*expr_p = NULL_TREE;
return GS_ALL_DONE;
}
static tree
find_omp_teams (tree *tp, int *walk_subtrees, void *)
{
*walk_subtrees = 0;
switch (TREE_CODE (*tp))
{
case OMP_TEAMS:
return *tp;
case BIND_EXPR:
case STATEMENT_LIST:
*walk_subtrees = 1;
break;
default:
break;
}
return NULL_TREE;
}
static tree
computable_teams_clause (tree *tp, int *walk_subtrees, void *)
{
splay_tree_node n;
if (TYPE_P (*tp))
{
*walk_subtrees = 0;
return NULL_TREE;
}
switch (TREE_CODE (*tp))
{
case VAR_DECL:
case PARM_DECL:
case RESULT_DECL:
*walk_subtrees = 0;
if (error_operand_p (*tp)
|| !INTEGRAL_TYPE_P (TREE_TYPE (*tp))
|| DECL_HAS_VALUE_EXPR_P (*tp)
|| DECL_THREAD_LOCAL_P (*tp)
|| TREE_SIDE_EFFECTS (*tp)
|| TREE_THIS_VOLATILE (*tp))
return *tp;
if (is_global_var (*tp)
&& (lookup_attribute ("omp declare target", DECL_ATTRIBUTES (*tp))
|| lookup_attribute ("omp declare target link",
DECL_ATTRIBUTES (*tp))))
return *tp;
if (VAR_P (*tp)
&& !DECL_SEEN_IN_BIND_EXPR_P (*tp)
&& !is_global_var (*tp)
&& decl_function_context (*tp) == current_function_decl)
return *tp;
n = splay_tree_lookup (gimplify_omp_ctxp->variables,
(splay_tree_key) *tp);
if (n == NULL)
{
if (gimplify_omp_ctxp->target_map_scalars_firstprivate)
return NULL_TREE;
return *tp;
}
else if (n->value & GOVD_LOCAL)
return *tp;
else if (n->value & GOVD_FIRSTPRIVATE)
return NULL_TREE;
else if ((n->value & (GOVD_MAP | GOVD_MAP_ALWAYS_TO))
== (GOVD_MAP | GOVD_MAP_ALWAYS_TO))
return NULL_TREE;
return *tp;
case INTEGER_CST:
if (!INTEGRAL_TYPE_P (TREE_TYPE (*tp)))
return *tp;
return NULL_TREE;
case TARGET_EXPR:
if (TARGET_EXPR_INITIAL (*tp)
|| TREE_CODE (TARGET_EXPR_SLOT (*tp)) != VAR_DECL)
return *tp;
return computable_teams_clause (&TARGET_EXPR_SLOT (*tp),
walk_subtrees, NULL);
case PLUS_EXPR:
case MINUS_EXPR:
case MULT_EXPR:
case TRUNC_DIV_EXPR:
case CEIL_DIV_EXPR:
case FLOOR_DIV_EXPR:
case ROUND_DIV_EXPR:
case TRUNC_MOD_EXPR:
case CEIL_MOD_EXPR:
case FLOOR_MOD_EXPR:
case ROUND_MOD_EXPR:
case RDIV_EXPR:
case EXACT_DIV_EXPR:
case MIN_EXPR:
case MAX_EXPR:
case LSHIFT_EXPR:
case RSHIFT_EXPR:
case BIT_IOR_EXPR:
case BIT_XOR_EXPR:
case BIT_AND_EXPR:
case NEGATE_EXPR:
case ABS_EXPR:
case BIT_NOT_EXPR:
case NON_LVALUE_EXPR:
CASE_CONVERT:
if (!INTEGRAL_TYPE_P (TREE_TYPE (*tp)))
return *tp;
return NULL_TREE;
default:
if (COMPARISON_CLASS_P (*tp))
return NULL_TREE;
return *tp;
}
}
static void
optimize_target_teams (tree target, gimple_seq *pre_p)
{
tree body = OMP_BODY (target);
tree teams = walk_tree (&body, find_omp_teams, NULL, NULL);
tree num_teams = integer_zero_node;
tree thread_limit = integer_zero_node;
location_t num_teams_loc = EXPR_LOCATION (target);
location_t thread_limit_loc = EXPR_LOCATION (target);
tree c, *p, expr;
struct gimplify_omp_ctx *target_ctx = gimplify_omp_ctxp;
if (teams == NULL_TREE)
num_teams = integer_one_node;
else
for (c = OMP_TEAMS_CLAUSES (teams); c; c = OMP_CLAUSE_CHAIN (c))
{
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_NUM_TEAMS)
{
p = &num_teams;
num_teams_loc = OMP_CLAUSE_LOCATION (c);
}
else if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_THREAD_LIMIT)
{
p = &thread_limit;
thread_limit_loc = OMP_CLAUSE_LOCATION (c);
}
else
continue;
expr = OMP_CLAUSE_OPERAND (c, 0);
if (TREE_CODE (expr) == INTEGER_CST)
{
*p = expr;
continue;
}
if (walk_tree (&expr, computable_teams_clause, NULL, NULL))
{
*p = integer_minus_one_node;
continue;
}
*p = expr;
gimplify_omp_ctxp = gimplify_omp_ctxp->outer_context;
if (gimplify_expr (p, pre_p, NULL, is_gimple_val, fb_rvalue, false)
== GS_ERROR)
{
gimplify_omp_ctxp = target_ctx;
*p = integer_minus_one_node;
continue;
}
gimplify_omp_ctxp = target_ctx;
if (!DECL_P (expr) && TREE_CODE (expr) != TARGET_EXPR)
OMP_CLAUSE_OPERAND (c, 0) = *p;
}
c = build_omp_clause (thread_limit_loc, OMP_CLAUSE_THREAD_LIMIT);
OMP_CLAUSE_THREAD_LIMIT_EXPR (c) = thread_limit;
OMP_CLAUSE_CHAIN (c) = OMP_TARGET_CLAUSES (target);
OMP_TARGET_CLAUSES (target) = c;
c = build_omp_clause (num_teams_loc, OMP_CLAUSE_NUM_TEAMS);
OMP_CLAUSE_NUM_TEAMS_EXPR (c) = num_teams;
OMP_CLAUSE_CHAIN (c) = OMP_TARGET_CLAUSES (target);
OMP_TARGET_CLAUSES (target) = c;
}
static void
gimplify_omp_workshare (tree *expr_p, gimple_seq *pre_p)
{
tree expr = *expr_p;
gimple *stmt;
gimple_seq body = NULL;
enum omp_region_type ort;
switch (TREE_CODE (expr))
{
case OMP_SECTIONS:
case OMP_SINGLE:
ort = ORT_WORKSHARE;
break;
case OMP_TARGET:
ort = OMP_TARGET_COMBINED (expr) ? ORT_COMBINED_TARGET : ORT_TARGET;
break;
case OACC_KERNELS:
ort = ORT_ACC_KERNELS;
break;
case OACC_PARALLEL:
ort = ORT_ACC_PARALLEL;
break;
case OACC_DATA:
ort = ORT_ACC_DATA;
break;
case OMP_TARGET_DATA:
ort = ORT_TARGET_DATA;
break;
case OMP_TEAMS:
ort = OMP_TEAMS_COMBINED (expr) ? ORT_COMBINED_TEAMS : ORT_TEAMS;
break;
case OACC_HOST_DATA:
ort = ORT_ACC_HOST_DATA;
break;
default:
gcc_unreachable ();
}
gimplify_scan_omp_clauses (&OMP_CLAUSES (expr), pre_p, ort,
TREE_CODE (expr));
if (TREE_CODE (expr) == OMP_TARGET)
optimize_target_teams (expr, pre_p);
if ((ort & (ORT_TARGET | ORT_TARGET_DATA)) != 0)
{
push_gimplify_context ();
gimple *g = gimplify_and_return_first (OMP_BODY (expr), &body);
if (gimple_code (g) == GIMPLE_BIND)
pop_gimplify_context (g);
else
pop_gimplify_context (NULL);
if ((ort & ORT_TARGET_DATA) != 0)
{
enum built_in_function end_ix;
switch (TREE_CODE (expr))
{
case OACC_DATA:
case OACC_HOST_DATA:
end_ix = BUILT_IN_GOACC_DATA_END;
break;
case OMP_TARGET_DATA:
end_ix = BUILT_IN_GOMP_TARGET_END_DATA;
break;
default:
gcc_unreachable ();
}
tree fn = builtin_decl_explicit (end_ix);
g = gimple_build_call (fn, 0);
gimple_seq cleanup = NULL;
gimple_seq_add_stmt (&cleanup, g);
g = gimple_build_try (body, cleanup, GIMPLE_TRY_FINALLY);
body = NULL;
gimple_seq_add_stmt (&body, g);
}
}
else
gimplify_and_add (OMP_BODY (expr), &body);
gimplify_adjust_omp_clauses (pre_p, body, &OMP_CLAUSES (expr),
TREE_CODE (expr));
switch (TREE_CODE (expr))
{
case OACC_DATA:
stmt = gimple_build_omp_target (body, GF_OMP_TARGET_KIND_OACC_DATA,
OMP_CLAUSES (expr));
break;
case OACC_KERNELS:
stmt = gimple_build_omp_target (body, GF_OMP_TARGET_KIND_OACC_KERNELS,
OMP_CLAUSES (expr));
break;
case OACC_HOST_DATA:
stmt = gimple_build_omp_target (body, GF_OMP_TARGET_KIND_OACC_HOST_DATA,
OMP_CLAUSES (expr));
break;
case OACC_PARALLEL:
stmt = gimple_build_omp_target (body, GF_OMP_TARGET_KIND_OACC_PARALLEL,
OMP_CLAUSES (expr));
break;
case OMP_SECTIONS:
stmt = gimple_build_omp_sections (body, OMP_CLAUSES (expr));
break;
case OMP_SINGLE:
stmt = gimple_build_omp_single (body, OMP_CLAUSES (expr));
break;
case OMP_TARGET:
stmt = gimple_build_omp_target (body, GF_OMP_TARGET_KIND_REGION,
OMP_CLAUSES (expr));
break;
case OMP_TARGET_DATA:
stmt = gimple_build_omp_target (body, GF_OMP_TARGET_KIND_DATA,
OMP_CLAUSES (expr));
break;
case OMP_TEAMS:
stmt = gimple_build_omp_teams (body, OMP_CLAUSES (expr));
break;
default:
gcc_unreachable ();
}
gimplify_seq_add_stmt (pre_p, stmt);
*expr_p = NULL_TREE;
}
static void
gimplify_omp_target_update (tree *expr_p, gimple_seq *pre_p)
{
tree expr = *expr_p;
int kind;
gomp_target *stmt;
enum omp_region_type ort = ORT_WORKSHARE;
switch (TREE_CODE (expr))
{
case OACC_ENTER_DATA:
case OACC_EXIT_DATA:
kind = GF_OMP_TARGET_KIND_OACC_ENTER_EXIT_DATA;
ort = ORT_ACC;
break;
case OACC_UPDATE:
kind = GF_OMP_TARGET_KIND_OACC_UPDATE;
ort = ORT_ACC;
break;
case OMP_TARGET_UPDATE:
kind = GF_OMP_TARGET_KIND_UPDATE;
break;
case OMP_TARGET_ENTER_DATA:
kind = GF_OMP_TARGET_KIND_ENTER_DATA;
break;
case OMP_TARGET_EXIT_DATA:
kind = GF_OMP_TARGET_KIND_EXIT_DATA;
break;
default:
gcc_unreachable ();
}
gimplify_scan_omp_clauses (&OMP_STANDALONE_CLAUSES (expr), pre_p,
ort, TREE_CODE (expr));
gimplify_adjust_omp_clauses (pre_p, NULL, &OMP_STANDALONE_CLAUSES (expr),
TREE_CODE (expr));
stmt = gimple_build_omp_target (NULL, kind, OMP_STANDALONE_CLAUSES (expr));
gimplify_seq_add_stmt (pre_p, stmt);
*expr_p = NULL_TREE;
}
static bool
goa_lhs_expr_p (tree expr, tree addr)
{
STRIP_USELESS_TYPE_CONVERSION (expr);
if (TREE_CODE (expr) == INDIRECT_REF)
{
expr = TREE_OPERAND (expr, 0);
while (expr != addr
&& (CONVERT_EXPR_P (expr)
|| TREE_CODE (expr) == NON_LVALUE_EXPR)
&& TREE_CODE (expr) == TREE_CODE (addr)
&& types_compatible_p (TREE_TYPE (expr), TREE_TYPE (addr)))
{
expr = TREE_OPERAND (expr, 0);
addr = TREE_OPERAND (addr, 0);
}
if (expr == addr)
return true;
return (TREE_CODE (addr) == ADDR_EXPR
&& TREE_CODE (expr) == ADDR_EXPR
&& TREE_OPERAND (addr, 0) == TREE_OPERAND (expr, 0));
}
if (TREE_CODE (addr) == ADDR_EXPR && expr == TREE_OPERAND (addr, 0))
return true;
return false;
}
static int
goa_stabilize_expr (tree *expr_p, gimple_seq *pre_p, tree lhs_addr,
tree lhs_var)
{
tree expr = *expr_p;
int saw_lhs;
if (goa_lhs_expr_p (expr, lhs_addr))
{
*expr_p = lhs_var;
return 1;
}
if (is_gimple_val (expr))
return 0;
saw_lhs = 0;
switch (TREE_CODE_CLASS (TREE_CODE (expr)))
{
case tcc_binary:
case tcc_comparison:
saw_lhs |= goa_stabilize_expr (&TREE_OPERAND (expr, 1), pre_p, lhs_addr,
lhs_var);
case tcc_unary:
saw_lhs |= goa_stabilize_expr (&TREE_OPERAND (expr, 0), pre_p, lhs_addr,
lhs_var);
break;
case tcc_expression:
switch (TREE_CODE (expr))
{
case TRUTH_ANDIF_EXPR:
case TRUTH_ORIF_EXPR:
case TRUTH_AND_EXPR:
case TRUTH_OR_EXPR:
case TRUTH_XOR_EXPR:
case BIT_INSERT_EXPR:
saw_lhs |= goa_stabilize_expr (&TREE_OPERAND (expr, 1), pre_p,
lhs_addr, lhs_var);
case TRUTH_NOT_EXPR:
saw_lhs |= goa_stabilize_expr (&TREE_OPERAND (expr, 0), pre_p,
lhs_addr, lhs_var);
break;
case COMPOUND_EXPR:
for (; TREE_CODE (expr) == COMPOUND_EXPR;
expr = TREE_OPERAND (expr, 1))
gimplify_stmt (&TREE_OPERAND (expr, 0), pre_p);
*expr_p = expr;
return goa_stabilize_expr (expr_p, pre_p, lhs_addr, lhs_var);
default:
break;
}
break;
case tcc_reference:
if (TREE_CODE (expr) == BIT_FIELD_REF)
saw_lhs |= goa_stabilize_expr (&TREE_OPERAND (expr, 0), pre_p,
lhs_addr, lhs_var);
break;
default:
break;
}
if (saw_lhs == 0)
{
enum gimplify_status gs;
gs = gimplify_expr (expr_p, pre_p, NULL, is_gimple_val, fb_rvalue);
if (gs != GS_ALL_DONE)
saw_lhs = -1;
}
return saw_lhs;
}
static enum gimplify_status
gimplify_omp_atomic (tree *expr_p, gimple_seq *pre_p)
{
tree addr = TREE_OPERAND (*expr_p, 0);
tree rhs = TREE_CODE (*expr_p) == OMP_ATOMIC_READ
? NULL : TREE_OPERAND (*expr_p, 1);
tree type = TYPE_MAIN_VARIANT (TREE_TYPE (TREE_TYPE (addr)));
tree tmp_load;
gomp_atomic_load *loadstmt;
gomp_atomic_store *storestmt;
tmp_load = create_tmp_reg (type);
if (rhs && goa_stabilize_expr (&rhs, pre_p, addr, tmp_load) < 0)
return GS_ERROR;
if (gimplify_expr (&addr, pre_p, NULL, is_gimple_val, fb_rvalue)
!= GS_ALL_DONE)
return GS_ERROR;
loadstmt = gimple_build_omp_atomic_load (tmp_load, addr);
gimplify_seq_add_stmt (pre_p, loadstmt);
if (rhs)
{
if (TREE_CODE (rhs) == BIT_INSERT_EXPR
&& !INTEGRAL_TYPE_P (TREE_TYPE (tmp_load)))
{
tree bitpos = TREE_OPERAND (rhs, 2);
tree op1 = TREE_OPERAND (rhs, 1);
tree bitsize;
tree tmp_store = tmp_load;
if (TREE_CODE (*expr_p) == OMP_ATOMIC_CAPTURE_OLD)
tmp_store = get_initialized_tmp_var (tmp_load, pre_p, NULL);
if (INTEGRAL_TYPE_P (TREE_TYPE (op1)))
bitsize = bitsize_int (TYPE_PRECISION (TREE_TYPE (op1)));
else
bitsize = TYPE_SIZE (TREE_TYPE (op1));
gcc_assert (TREE_OPERAND (rhs, 0) == tmp_load);
tree t = build2_loc (EXPR_LOCATION (rhs),
MODIFY_EXPR, void_type_node,
build3_loc (EXPR_LOCATION (rhs), BIT_FIELD_REF,
TREE_TYPE (op1), tmp_store, bitsize,
bitpos), op1);
gimplify_and_add (t, pre_p);
rhs = tmp_store;
}
if (gimplify_expr (&rhs, pre_p, NULL, is_gimple_val, fb_rvalue)
!= GS_ALL_DONE)
return GS_ERROR;
}
if (TREE_CODE (*expr_p) == OMP_ATOMIC_READ)
rhs = tmp_load;
storestmt = gimple_build_omp_atomic_store (rhs);
gimplify_seq_add_stmt (pre_p, storestmt);
if (OMP_ATOMIC_SEQ_CST (*expr_p))
{
gimple_omp_atomic_set_seq_cst (loadstmt);
gimple_omp_atomic_set_seq_cst (storestmt);
}
switch (TREE_CODE (*expr_p))
{
case OMP_ATOMIC_READ:
case OMP_ATOMIC_CAPTURE_OLD:
*expr_p = tmp_load;
gimple_omp_atomic_set_need_value (loadstmt);
break;
case OMP_ATOMIC_CAPTURE_NEW:
*expr_p = rhs;
gimple_omp_atomic_set_need_value (storestmt);
break;
default:
*expr_p = NULL;
break;
}
return GS_ALL_DONE;
}
static enum gimplify_status
gimplify_transaction (tree *expr_p, gimple_seq *pre_p)
{
tree expr = *expr_p, temp, tbody = TRANSACTION_EXPR_BODY (expr);
gimple *body_stmt;
gtransaction *trans_stmt;
gimple_seq body = NULL;
int subcode = 0;
if (TREE_CODE (tbody) != BIND_EXPR)
{
tree bind = build3 (BIND_EXPR, void_type_node, NULL, tbody, NULL);
TREE_SIDE_EFFECTS (bind) = 1;
SET_EXPR_LOCATION (bind, EXPR_LOCATION (tbody));
TRANSACTION_EXPR_BODY (expr) = bind;
}
push_gimplify_context ();
temp = voidify_wrapper_expr (*expr_p, NULL);
body_stmt = gimplify_and_return_first (TRANSACTION_EXPR_BODY (expr), &body);
pop_gimplify_context (body_stmt);
trans_stmt = gimple_build_transaction (body);
if (TRANSACTION_EXPR_OUTER (expr))
subcode = GTMA_IS_OUTER;
else if (TRANSACTION_EXPR_RELAXED (expr))
subcode = GTMA_IS_RELAXED;
gimple_transaction_set_subcode (trans_stmt, subcode);
gimplify_seq_add_stmt (pre_p, trans_stmt);
if (temp)
{
*expr_p = temp;
return GS_OK;
}
*expr_p = NULL_TREE;
return GS_ALL_DONE;
}
static gimple *
gimplify_omp_ordered (tree expr, gimple_seq body)
{
tree c, decls;
int failures = 0;
unsigned int i;
tree source_c = NULL_TREE;
tree sink_c = NULL_TREE;
if (gimplify_omp_ctxp)
{
for (c = OMP_ORDERED_CLAUSES (expr); c; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_DEPEND
&& gimplify_omp_ctxp->loop_iter_var.is_empty ()
&& (OMP_CLAUSE_DEPEND_KIND (c) == OMP_CLAUSE_DEPEND_SINK
|| OMP_CLAUSE_DEPEND_KIND (c) == OMP_CLAUSE_DEPEND_SOURCE))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<ordered%> construct with %<depend%> clause must be "
"closely nested inside a loop with %<ordered%> clause "
"with a parameter");
failures++;
}
else if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_DEPEND
&& OMP_CLAUSE_DEPEND_KIND (c) == OMP_CLAUSE_DEPEND_SINK)
{
bool fail = false;
for (decls = OMP_CLAUSE_DECL (c), i = 0;
decls && TREE_CODE (decls) == TREE_LIST;
decls = TREE_CHAIN (decls), ++i)
if (i >= gimplify_omp_ctxp->loop_iter_var.length () / 2)
continue;
else if (TREE_VALUE (decls)
!= gimplify_omp_ctxp->loop_iter_var[2 * i])
{
error_at (OMP_CLAUSE_LOCATION (c),
"variable %qE is not an iteration "
"of outermost loop %d, expected %qE",
TREE_VALUE (decls), i + 1,
gimplify_omp_ctxp->loop_iter_var[2 * i]);
fail = true;
failures++;
}
else
TREE_VALUE (decls)
= gimplify_omp_ctxp->loop_iter_var[2 * i + 1];
if (!fail && i != gimplify_omp_ctxp->loop_iter_var.length () / 2)
{
error_at (OMP_CLAUSE_LOCATION (c),
"number of variables in %<depend(sink)%> "
"clause does not match number of "
"iteration variables");
failures++;
}
sink_c = c;
}
else if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_DEPEND
&& OMP_CLAUSE_DEPEND_KIND (c) == OMP_CLAUSE_DEPEND_SOURCE)
{
if (source_c)
{
error_at (OMP_CLAUSE_LOCATION (c),
"more than one %<depend(source)%> clause on an "
"%<ordered%> construct");
failures++;
}
else
source_c = c;
}
}
if (source_c && sink_c)
{
error_at (OMP_CLAUSE_LOCATION (source_c),
"%<depend(source)%> clause specified together with "
"%<depend(sink:)%> clauses on the same construct");
failures++;
}
if (failures)
return gimple_build_nop ();
return gimple_build_omp_ordered (body, OMP_ORDERED_CLAUSES (expr));
}
enum gimplify_status
gimplify_expr (tree *expr_p, gimple_seq *pre_p, gimple_seq *post_p,
bool (*gimple_test_f) (tree), fallback_t fallback)
{
tree tmp;
gimple_seq internal_pre = NULL;
gimple_seq internal_post = NULL;
tree save_expr;
bool is_statement;
location_t saved_location;
enum gimplify_status ret;
gimple_stmt_iterator pre_last_gsi, post_last_gsi;
tree label;
save_expr = *expr_p;
if (save_expr == NULL_TREE)
return GS_ALL_DONE;
is_statement = gimple_test_f == is_gimple_stmt;
if (is_statement)
gcc_assert (pre_p);
if (gimple_test_f == is_gimple_reg)
gcc_assert (fallback & (fb_rvalue | fb_lvalue));
else if (gimple_test_f == is_gimple_val
|| gimple_test_f == is_gimple_call_addr
|| gimple_test_f == is_gimple_condexpr
|| gimple_test_f == is_gimple_mem_rhs
|| gimple_test_f == is_gimple_mem_rhs_or_call
|| gimple_test_f == is_gimple_reg_rhs
|| gimple_test_f == is_gimple_reg_rhs_or_call
|| gimple_test_f == is_gimple_asm_val
|| gimple_test_f == is_gimple_mem_ref_addr)
gcc_assert (fallback & fb_rvalue);
else if (gimple_test_f == is_gimple_min_lval
|| gimple_test_f == is_gimple_lvalue)
gcc_assert (fallback & fb_lvalue);
else if (gimple_test_f == is_gimple_addressable)
gcc_assert (fallback & fb_either);
else if (gimple_test_f == is_gimple_stmt)
gcc_assert (fallback == fb_none);
else
{
gcc_unreachable ();
}
if (pre_p == NULL)
pre_p = &internal_pre;
if (post_p == NULL)
post_p = &internal_post;
pre_last_gsi = gsi_last (*pre_p);
post_last_gsi = gsi_last (*post_p);
saved_location = input_location;
if (save_expr != error_mark_node
&& EXPR_HAS_LOCATION (*expr_p))
input_location = EXPR_LOCATION (*expr_p);
do
{
STRIP_USELESS_TYPE_CONVERSION (*expr_p);
save_expr = *expr_p;
if (error_operand_p (save_expr))
{
ret = GS_ERROR;
break;
}
ret = ((enum gimplify_status)
lang_hooks.gimplify_expr (expr_p, pre_p, post_p));
if (ret == GS_OK)
{
if (*expr_p == NULL_TREE)
break;
if (*expr_p != save_expr)
continue;
}
else if (ret != GS_UNHANDLED)
break;
ret = GS_UNHANDLED;
switch (TREE_CODE (*expr_p))
{
case POSTINCREMENT_EXPR:
case POSTDECREMENT_EXPR:
case PREINCREMENT_EXPR:
case PREDECREMENT_EXPR:
ret = gimplify_self_mod_expr (expr_p, pre_p, post_p,
fallback != fb_none,
TREE_TYPE (*expr_p));
break;
case VIEW_CONVERT_EXPR:
if (is_gimple_reg_type (TREE_TYPE (*expr_p))
&& is_gimple_reg_type (TREE_TYPE (TREE_OPERAND (*expr_p, 0))))
{
ret = gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p,
post_p, is_gimple_val, fb_rvalue);
recalculate_side_effects (*expr_p);
break;
}
case ARRAY_REF:
case ARRAY_RANGE_REF:
case REALPART_EXPR:
case IMAGPART_EXPR:
case COMPONENT_REF:
ret = gimplify_compound_lval (expr_p, pre_p, post_p,
fallback ? fallback : fb_rvalue);
break;
case COND_EXPR:
ret = gimplify_cond_expr (expr_p, pre_p, fallback);
if (fallback == fb_lvalue)
{
*expr_p = get_initialized_tmp_var (*expr_p, pre_p, post_p, false);
mark_addressable (*expr_p);
ret = GS_OK;
}
break;
case CALL_EXPR:
ret = gimplify_call_expr (expr_p, pre_p, fallback != fb_none);
if (fallback == fb_lvalue)
{
*expr_p = get_initialized_tmp_var (*expr_p, pre_p, post_p, false);
mark_addressable (*expr_p);
ret = GS_OK;
}
break;
case TREE_LIST:
gcc_unreachable ();
case COMPOUND_EXPR:
ret = gimplify_compound_expr (expr_p, pre_p, fallback != fb_none);
break;
case COMPOUND_LITERAL_EXPR:
ret = gimplify_compound_literal_expr (expr_p, pre_p,
gimple_test_f, fallback);
break;
case MODIFY_EXPR:
case INIT_EXPR:
ret = gimplify_modify_expr (expr_p, pre_p, post_p,
fallback != fb_none);
break;
case TRUTH_ANDIF_EXPR:
case TRUTH_ORIF_EXPR:
{
tree org_type = TREE_TYPE (*expr_p);
*expr_p = gimple_boolify (*expr_p);
*expr_p = build3_loc (input_location, COND_EXPR,
org_type, *expr_p,
fold_convert_loc
(input_location,
org_type, boolean_true_node),
fold_convert_loc
(input_location,
org_type, boolean_false_node));
ret = GS_OK;
break;
}
case TRUTH_NOT_EXPR:
{
tree type = TREE_TYPE (*expr_p);
*expr_p = gimple_boolify (*expr_p);
if (TYPE_PRECISION (TREE_TYPE (*expr_p)) == 1)
*expr_p = build1_loc (input_location, BIT_NOT_EXPR,
TREE_TYPE (*expr_p),
TREE_OPERAND (*expr_p, 0));
else
*expr_p = build2_loc (input_location, BIT_XOR_EXPR,
TREE_TYPE (*expr_p),
TREE_OPERAND (*expr_p, 0),
build_int_cst (TREE_TYPE (*expr_p), 1));
if (!useless_type_conversion_p (type, TREE_TYPE (*expr_p)))
*expr_p = fold_convert_loc (input_location, type, *expr_p);
ret = GS_OK;
break;
}
case ADDR_EXPR:
ret = gimplify_addr_expr (expr_p, pre_p, post_p);
break;
case ANNOTATE_EXPR:
{
tree cond = TREE_OPERAND (*expr_p, 0);
tree kind = TREE_OPERAND (*expr_p, 1);
tree data = TREE_OPERAND (*expr_p, 2);
tree type = TREE_TYPE (cond);
if (!INTEGRAL_TYPE_P (type))
{
*expr_p = cond;
ret = GS_OK;
break;
}
tree tmp = create_tmp_var (type);
gimplify_arg (&cond, pre_p, EXPR_LOCATION (*expr_p));
gcall *call
= gimple_build_call_internal (IFN_ANNOTATE, 3, cond, kind, data);
gimple_call_set_lhs (call, tmp);
gimplify_seq_add_stmt (pre_p, call);
*expr_p = tmp;
ret = GS_ALL_DONE;
break;
}
case VA_ARG_EXPR:
ret = gimplify_va_arg_expr (expr_p, pre_p, post_p);
break;
CASE_CONVERT:
if (IS_EMPTY_STMT (*expr_p))
{
ret = GS_ALL_DONE;
break;
}
if (VOID_TYPE_P (TREE_TYPE (*expr_p))
|| fallback == fb_none)
{
*expr_p = TREE_OPERAND (*expr_p, 0);
ret = GS_OK;
break;
}
ret = gimplify_conversion (expr_p);
if (ret == GS_ERROR)
break;
if (*expr_p != save_expr)
break;
case FIX_TRUNC_EXPR:
ret = gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p, post_p,
is_gimple_val, fb_rvalue);
recalculate_side_effects (*expr_p);
break;
case INDIRECT_REF:
{
bool volatilep = TREE_THIS_VOLATILE (*expr_p);
bool notrap = TREE_THIS_NOTRAP (*expr_p);
tree saved_ptr_type = TREE_TYPE (TREE_OPERAND (*expr_p, 0));
*expr_p = fold_indirect_ref_loc (input_location, *expr_p);
if (*expr_p != save_expr)
{
ret = GS_OK;
break;
}
ret = gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p, post_p,
is_gimple_reg, fb_rvalue);
if (ret == GS_ERROR)
break;
recalculate_side_effects (*expr_p);
*expr_p = fold_build2_loc (input_location, MEM_REF,
TREE_TYPE (*expr_p),
TREE_OPERAND (*expr_p, 0),
build_int_cst (saved_ptr_type, 0));
TREE_THIS_VOLATILE (*expr_p) = volatilep;
TREE_THIS_NOTRAP (*expr_p) = notrap;
ret = GS_OK;
break;
}
case MEM_REF:
tmp = fold_binary (MEM_REF, TREE_TYPE (*expr_p),
TREE_OPERAND (*expr_p, 0),
TREE_OPERAND (*expr_p, 1));
if (tmp)
{
REF_REVERSE_STORAGE_ORDER (tmp)
= REF_REVERSE_STORAGE_ORDER (*expr_p);
*expr_p = tmp;
recalculate_side_effects (*expr_p);
ret = GS_OK;
break;
}
if (!gimplify_ctxp || !gimple_in_ssa_p (cfun)
|| !is_gimple_mem_ref_addr (TREE_OPERAND (*expr_p, 0)))
{
ret = gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p, post_p,
is_gimple_mem_ref_addr, fb_rvalue);
if (ret == GS_ERROR)
break;
}
recalculate_side_effects (*expr_p);
ret = GS_ALL_DONE;
break;
case INTEGER_CST:
case REAL_CST:
case FIXED_CST:
case STRING_CST:
case COMPLEX_CST:
case VECTOR_CST:
if (TREE_OVERFLOW_P (*expr_p))
*expr_p = drop_tree_overflow (*expr_p);
ret = GS_ALL_DONE;
break;
case CONST_DECL:
if (fallback & fb_lvalue)
ret = GS_ALL_DONE;
else
{
*expr_p = DECL_INITIAL (*expr_p);
ret = GS_OK;
}
break;
case DECL_EXPR:
ret = gimplify_decl_expr (expr_p, pre_p);
break;
case BIND_EXPR:
ret = gimplify_bind_expr (expr_p, pre_p);
break;
case LOOP_EXPR:
ret = gimplify_loop_expr (expr_p, pre_p);
break;
case SWITCH_EXPR:
ret = gimplify_switch_expr (expr_p, pre_p);
break;
case EXIT_EXPR:
ret = gimplify_exit_expr (expr_p);
break;
case GOTO_EXPR:
if (TREE_CODE (GOTO_DESTINATION (*expr_p)) != LABEL_DECL)
{
ret = gimplify_expr (&GOTO_DESTINATION (*expr_p), pre_p,
NULL, is_gimple_val, fb_rvalue);
if (ret == GS_ERROR)
break;
}
gimplify_seq_add_stmt (pre_p,
gimple_build_goto (GOTO_DESTINATION (*expr_p)));
ret = GS_ALL_DONE;
break;
case PREDICT_EXPR:
gimplify_seq_add_stmt (pre_p,
gimple_build_predict (PREDICT_EXPR_PREDICTOR (*expr_p),
PREDICT_EXPR_OUTCOME (*expr_p)));
ret = GS_ALL_DONE;
break;
case LABEL_EXPR:
ret = gimplify_label_expr (expr_p, pre_p);
label = LABEL_EXPR_LABEL (*expr_p);
gcc_assert (decl_function_context (label) == current_function_decl);
if (asan_poisoned_variables
&& asan_used_labels != NULL
&& asan_used_labels->contains (label))
asan_poison_variables (asan_poisoned_variables, false, pre_p);
break;
case CASE_LABEL_EXPR:
ret = gimplify_case_label_expr (expr_p, pre_p);
if (gimplify_ctxp->live_switch_vars)
asan_poison_variables (gimplify_ctxp->live_switch_vars, false,
pre_p);
break;
case RETURN_EXPR:
ret = gimplify_return_expr (*expr_p, pre_p);
break;
case CONSTRUCTOR:
if (fallback == fb_none)
{
unsigned HOST_WIDE_INT ix;
tree val;
tree temp = NULL_TREE;
FOR_EACH_CONSTRUCTOR_VALUE (CONSTRUCTOR_ELTS (*expr_p), ix, val)
if (TREE_SIDE_EFFECTS (val))
append_to_statement_list (val, &temp);
*expr_p = temp;
ret = temp ? GS_OK : GS_ALL_DONE;
}
else if (fallback == fb_lvalue)
{
*expr_p = get_initialized_tmp_var (*expr_p, pre_p, post_p, false);
mark_addressable (*expr_p);
ret = GS_OK;
}
else
ret = GS_ALL_DONE;
break;
case SAVE_EXPR:
ret = gimplify_save_expr (expr_p, pre_p, post_p);
break;
case BIT_FIELD_REF:
ret = gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p,
post_p, is_gimple_lvalue, fb_either);
recalculate_side_effects (*expr_p);
break;
case TARGET_MEM_REF:
{
enum gimplify_status r0 = GS_ALL_DONE, r1 = GS_ALL_DONE;
if (TMR_BASE (*expr_p))
r0 = gimplify_expr (&TMR_BASE (*expr_p), pre_p,
post_p, is_gimple_mem_ref_addr, fb_either);
if (TMR_INDEX (*expr_p))
r1 = gimplify_expr (&TMR_INDEX (*expr_p), pre_p,
post_p, is_gimple_val, fb_rvalue);
if (TMR_INDEX2 (*expr_p))
r1 = gimplify_expr (&TMR_INDEX2 (*expr_p), pre_p,
post_p, is_gimple_val, fb_rvalue);
ret = MIN (r0, r1);
}
break;
case NON_LVALUE_EXPR:
gcc_unreachable ();
case ASM_EXPR:
ret = gimplify_asm_expr (expr_p, pre_p, post_p);
break;
case TRY_FINALLY_EXPR:
case TRY_CATCH_EXPR:
{
gimple_seq eval, cleanup;
gtry *try_;
input_location = UNKNOWN_LOCATION;
eval = cleanup = NULL;
gimplify_and_add (TREE_OPERAND (*expr_p, 0), &eval);
gimplify_and_add (TREE_OPERAND (*expr_p, 1), &cleanup);
if (gimple_seq_empty_p (cleanup))
{
gimple_seq_add_seq (pre_p, eval);
ret = GS_ALL_DONE;
break;
}
try_ = gimple_build_try (eval, cleanup,
TREE_CODE (*expr_p) == TRY_FINALLY_EXPR
? GIMPLE_TRY_FINALLY
: GIMPLE_TRY_CATCH);
if (EXPR_HAS_LOCATION (save_expr))
gimple_set_location (try_, EXPR_LOCATION (save_expr));
else if (LOCATION_LOCUS (saved_location) != UNKNOWN_LOCATION)
gimple_set_location (try_, saved_location);
if (TREE_CODE (*expr_p) == TRY_CATCH_EXPR)
gimple_try_set_catch_is_cleanup (try_,
TRY_CATCH_IS_CLEANUP (*expr_p));
gimplify_seq_add_stmt (pre_p, try_);
ret = GS_ALL_DONE;
break;
}
case CLEANUP_POINT_EXPR:
ret = gimplify_cleanup_point_expr (expr_p, pre_p);
break;
case TARGET_EXPR:
ret = gimplify_target_expr (expr_p, pre_p, post_p);
break;
case CATCH_EXPR:
{
gimple *c;
gimple_seq handler = NULL;
gimplify_and_add (CATCH_BODY (*expr_p), &handler);
c = gimple_build_catch (CATCH_TYPES (*expr_p), handler);
gimplify_seq_add_stmt (pre_p, c);
ret = GS_ALL_DONE;
break;
}
case EH_FILTER_EXPR:
{
gimple *ehf;
gimple_seq failure = NULL;
gimplify_and_add (EH_FILTER_FAILURE (*expr_p), &failure);
ehf = gimple_build_eh_filter (EH_FILTER_TYPES (*expr_p), failure);
gimple_set_no_warning (ehf, TREE_NO_WARNING (*expr_p));
gimplify_seq_add_stmt (pre_p, ehf);
ret = GS_ALL_DONE;
break;
}
case OBJ_TYPE_REF:
{
enum gimplify_status r0, r1;
r0 = gimplify_expr (&OBJ_TYPE_REF_OBJECT (*expr_p), pre_p,
post_p, is_gimple_val, fb_rvalue);
r1 = gimplify_expr (&OBJ_TYPE_REF_EXPR (*expr_p), pre_p,
post_p, is_gimple_val, fb_rvalue);
TREE_SIDE_EFFECTS (*expr_p) = 0;
ret = MIN (r0, r1);
}
break;
case LABEL_DECL:
FORCED_LABEL (*expr_p) = 1;
ret = GS_ALL_DONE;
break;
case STATEMENT_LIST:
ret = gimplify_statement_list (expr_p, pre_p);
break;
case WITH_SIZE_EXPR:
{
gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p,
post_p == &internal_post ? NULL : post_p,
gimple_test_f, fallback);
gimplify_expr (&TREE_OPERAND (*expr_p, 1), pre_p, post_p,
is_gimple_val, fb_rvalue);
ret = GS_ALL_DONE;
}
break;
case VAR_DECL:
case PARM_DECL:
ret = gimplify_var_or_parm_decl (expr_p);
break;
case RESULT_DECL:
if (gimplify_omp_ctxp)
omp_notice_variable (gimplify_omp_ctxp, *expr_p, true);
ret = GS_ALL_DONE;
break;
case DEBUG_EXPR_DECL:
gcc_unreachable ();
case DEBUG_BEGIN_STMT:
gimplify_seq_add_stmt (pre_p,
gimple_build_debug_begin_stmt
(TREE_BLOCK (*expr_p),
EXPR_LOCATION (*expr_p)));
ret = GS_ALL_DONE;
*expr_p = NULL;
break;
case SSA_NAME:
ret = GS_ALL_DONE;
break;
case OMP_PARALLEL:
gimplify_omp_parallel (expr_p, pre_p);
ret = GS_ALL_DONE;
break;
case OMP_TASK:
gimplify_omp_task (expr_p, pre_p);
ret = GS_ALL_DONE;
break;
case OMP_FOR:
case OMP_SIMD:
case OMP_DISTRIBUTE:
case OMP_TASKLOOP:
case OACC_LOOP:
ret = gimplify_omp_for (expr_p, pre_p);
break;
case OACC_CACHE:
gimplify_oacc_cache (expr_p, pre_p);
ret = GS_ALL_DONE;
break;
case OACC_DECLARE:
gimplify_oacc_declare (expr_p, pre_p);
ret = GS_ALL_DONE;
break;
case OACC_HOST_DATA:
case OACC_DATA:
case OACC_KERNELS:
case OACC_PARALLEL:
case OMP_SECTIONS:
case OMP_SINGLE:
case OMP_TARGET:
case OMP_TARGET_DATA:
case OMP_TEAMS:
gimplify_omp_workshare (expr_p, pre_p);
ret = GS_ALL_DONE;
break;
case OACC_ENTER_DATA:
case OACC_EXIT_DATA:
case OACC_UPDATE:
case OMP_TARGET_UPDATE:
case OMP_TARGET_ENTER_DATA:
case OMP_TARGET_EXIT_DATA:
gimplify_omp_target_update (expr_p, pre_p);
ret = GS_ALL_DONE;
break;
case OMP_SECTION:
case OMP_MASTER:
case OMP_TASKGROUP:
case OMP_ORDERED:
case OMP_CRITICAL:
{
gimple_seq body = NULL;
gimple *g;
gimplify_and_add (OMP_BODY (*expr_p), &body);
switch (TREE_CODE (*expr_p))
{
case OMP_SECTION:
g = gimple_build_omp_section (body);
break;
case OMP_MASTER:
g = gimple_build_omp_master (body);
break;
case OMP_TASKGROUP:
{
gimple_seq cleanup = NULL;
tree fn
= builtin_decl_explicit (BUILT_IN_GOMP_TASKGROUP_END);
g = gimple_build_call (fn, 0);
gimple_seq_add_stmt (&cleanup, g);
g = gimple_build_try (body, cleanup, GIMPLE_TRY_FINALLY);
body = NULL;
gimple_seq_add_stmt (&body, g);
g = gimple_build_omp_taskgroup (body);
}
break;
case OMP_ORDERED:
g = gimplify_omp_ordered (*expr_p, body);
break;
case OMP_CRITICAL:
gimplify_scan_omp_clauses (&OMP_CRITICAL_CLAUSES (*expr_p),
pre_p, ORT_WORKSHARE, OMP_CRITICAL);
gimplify_adjust_omp_clauses (pre_p, body,
&OMP_CRITICAL_CLAUSES (*expr_p),
OMP_CRITICAL);
g = gimple_build_omp_critical (body,
OMP_CRITICAL_NAME (*expr_p),
OMP_CRITICAL_CLAUSES (*expr_p));
break;
default:
gcc_unreachable ();
}
gimplify_seq_add_stmt (pre_p, g);
ret = GS_ALL_DONE;
break;
}
case OMP_ATOMIC:
case OMP_ATOMIC_READ:
case OMP_ATOMIC_CAPTURE_OLD:
case OMP_ATOMIC_CAPTURE_NEW:
ret = gimplify_omp_atomic (expr_p, pre_p);
break;
case TRANSACTION_EXPR:
ret = gimplify_transaction (expr_p, pre_p);
break;
case TRUTH_AND_EXPR:
case TRUTH_OR_EXPR:
case TRUTH_XOR_EXPR:
{
tree orig_type = TREE_TYPE (*expr_p);
tree new_type, xop0, xop1;
*expr_p = gimple_boolify (*expr_p);
new_type = TREE_TYPE (*expr_p);
if (!useless_type_conversion_p (orig_type, new_type))
{
*expr_p = fold_convert_loc (input_location, orig_type, *expr_p);
ret = GS_OK;
break;
}
switch (TREE_CODE (*expr_p))
{
case TRUTH_AND_EXPR:
TREE_SET_CODE (*expr_p, BIT_AND_EXPR);
break;
case TRUTH_OR_EXPR:
TREE_SET_CODE (*expr_p, BIT_IOR_EXPR);
break;
case TRUTH_XOR_EXPR:
TREE_SET_CODE (*expr_p, BIT_XOR_EXPR);
break;
default:
break;
}
xop0 = TREE_OPERAND (*expr_p, 0);
xop1 = TREE_OPERAND (*expr_p, 1);
if (!useless_type_conversion_p (new_type, TREE_TYPE (xop0)))
TREE_OPERAND (*expr_p, 0) = fold_convert_loc (input_location,
new_type,
xop0);
if (!useless_type_conversion_p (new_type, TREE_TYPE (xop1)))
TREE_OPERAND (*expr_p, 1) = fold_convert_loc (input_location,
new_type,
xop1);
goto expr_2;
}
case VEC_COND_EXPR:
{
enum gimplify_status r0, r1, r2;
r0 = gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p,
post_p, is_gimple_condexpr, fb_rvalue);
r1 = gimplify_expr (&TREE_OPERAND (*expr_p, 1), pre_p,
post_p, is_gimple_val, fb_rvalue);
r2 = gimplify_expr (&TREE_OPERAND (*expr_p, 2), pre_p,
post_p, is_gimple_val, fb_rvalue);
ret = MIN (MIN (r0, r1), r2);
recalculate_side_effects (*expr_p);
}
break;
case FMA_EXPR:
case VEC_PERM_EXPR:
goto expr_3;
case BIT_INSERT_EXPR:
goto expr_2;
case POINTER_PLUS_EXPR:
{
enum gimplify_status r0, r1;
r0 = gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p,
post_p, is_gimple_val, fb_rvalue);
r1 = gimplify_expr (&TREE_OPERAND (*expr_p, 1), pre_p,
post_p, is_gimple_val, fb_rvalue);
recalculate_side_effects (*expr_p);
ret = MIN (r0, r1);
break;
}
default:
switch (TREE_CODE_CLASS (TREE_CODE (*expr_p)))
{
case tcc_comparison:
{
tree type = TREE_TYPE (TREE_OPERAND (*expr_p, 1));
if (TREE_CODE (type) == VECTOR_TYPE)
goto expr_2;
else if (!AGGREGATE_TYPE_P (type))
{
tree org_type = TREE_TYPE (*expr_p);
*expr_p = gimple_boolify (*expr_p);
if (!useless_type_conversion_p (org_type,
TREE_TYPE (*expr_p)))
{
*expr_p = fold_convert_loc (input_location,
org_type, *expr_p);
ret = GS_OK;
}
else
goto expr_2;
}
else if (TYPE_MODE (type) != BLKmode)
ret = gimplify_scalar_mode_aggregate_compare (expr_p);
else
ret = gimplify_variable_sized_compare (expr_p);
break;
}
case tcc_unary:
ret = gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p,
post_p, is_gimple_val, fb_rvalue);
break;
case tcc_binary:
expr_2:
{
enum gimplify_status r0, r1;
r0 = gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p,
post_p, is_gimple_val, fb_rvalue);
r1 = gimplify_expr (&TREE_OPERAND (*expr_p, 1), pre_p,
post_p, is_gimple_val, fb_rvalue);
ret = MIN (r0, r1);
break;
}
expr_3:
{
enum gimplify_status r0, r1, r2;
r0 = gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p,
post_p, is_gimple_val, fb_rvalue);
r1 = gimplify_expr (&TREE_OPERAND (*expr_p, 1), pre_p,
post_p, is_gimple_val, fb_rvalue);
r2 = gimplify_expr (&TREE_OPERAND (*expr_p, 2), pre_p,
post_p, is_gimple_val, fb_rvalue);
ret = MIN (MIN (r0, r1), r2);
break;
}
case tcc_declaration:
case tcc_constant:
ret = GS_ALL_DONE;
goto dont_recalculate;
default:
gcc_unreachable ();
}
recalculate_side_effects (*expr_p);
dont_recalculate:
break;
}
gcc_assert (*expr_p || ret != GS_OK);
}
while (ret == GS_OK);
if (ret == GS_ERROR)
{
if (is_statement)
*expr_p = NULL;
goto out;
}
gcc_assert (ret != GS_UNHANDLED);
if (fallback == fb_none && *expr_p && !is_gimple_stmt (*expr_p))
{
if (TREE_CODE (*expr_p) == LABEL_DECL
|| !TREE_SIDE_EFFECTS (*expr_p))
*expr_p = NULL;
else if (!TREE_THIS_VOLATILE (*expr_p))
{
enum tree_code code = TREE_CODE (*expr_p);
switch (code)
{
case COMPONENT_REF:
case REALPART_EXPR:
case IMAGPART_EXPR:
case VIEW_CONVERT_EXPR:
gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p, post_p,
gimple_test_f, fallback);
break;
case ARRAY_REF:
case ARRAY_RANGE_REF:
gimplify_expr (&TREE_OPERAND (*expr_p, 0), pre_p, post_p,
gimple_test_f, fallback);
gimplify_expr (&TREE_OPERAND (*expr_p, 1), pre_p, post_p,
gimple_test_f, fallback);
break;
default:
gcc_unreachable ();
}
*expr_p = NULL;
}
else if (COMPLETE_TYPE_P (TREE_TYPE (*expr_p))
&& TYPE_MODE (TREE_TYPE (*expr_p)) != BLKmode)
{
tree type = TYPE_MAIN_VARIANT (TREE_TYPE (*expr_p));
tree tmp = create_tmp_var_raw (type, "vol");
gimple_add_tmp_var (tmp);
gimplify_assign (tmp, *expr_p, pre_p);
*expr_p = NULL;
}
else
*expr_p = NULL;
}
if (fallback == fb_none || is_statement)
{
*expr_p = NULL_TREE;
if (!gimple_seq_empty_p (internal_pre)
|| !gimple_seq_empty_p (internal_post))
{
gimplify_seq_add_seq (&internal_pre, internal_post);
gimplify_seq_add_seq (pre_p, internal_pre);
}
if (!gimple_seq_empty_p (*pre_p))
annotate_all_with_location_after (*pre_p, pre_last_gsi, input_location);
if (!gimple_seq_empty_p (*post_p))
annotate_all_with_location_after (*post_p, post_last_gsi,
input_location);
goto out;
}
#ifdef ENABLE_GIMPLE_CHECKING
if (*expr_p)
{
enum tree_code code = TREE_CODE (*expr_p);
gcc_assert (code != MODIFY_EXPR
&& code != ASM_EXPR
&& code != BIND_EXPR
&& code != CATCH_EXPR
&& (code != COND_EXPR || gimplify_ctxp->allow_rhs_cond_expr)
&& code != EH_FILTER_EXPR
&& code != GOTO_EXPR
&& code != LABEL_EXPR
&& code != LOOP_EXPR
&& code != SWITCH_EXPR
&& code != TRY_FINALLY_EXPR
&& code != OACC_PARALLEL
&& code != OACC_KERNELS
&& code != OACC_DATA
&& code != OACC_HOST_DATA
&& code != OACC_DECLARE
&& code != OACC_UPDATE
&& code != OACC_ENTER_DATA
&& code != OACC_EXIT_DATA
&& code != OACC_CACHE
&& code != OMP_CRITICAL
&& code != OMP_FOR
&& code != OACC_LOOP
&& code != OMP_MASTER
&& code != OMP_TASKGROUP
&& code != OMP_ORDERED
&& code != OMP_PARALLEL
&& code != OMP_SECTIONS
&& code != OMP_SECTION
&& code != OMP_SINGLE);
}
#endif
if (gimple_seq_empty_p (internal_post) && (*gimple_test_f) (*expr_p))
goto out;
if ((fallback & fb_lvalue)
&& gimple_seq_empty_p (internal_post)
&& is_gimple_addressable (*expr_p))
{
tmp = build_fold_addr_expr_loc (input_location, *expr_p);
gimplify_expr (&tmp, pre_p, post_p, is_gimple_reg, fb_rvalue);
*expr_p = build_simple_mem_ref (tmp);
}
else if ((fallback & fb_rvalue) && is_gimple_reg_rhs_or_call (*expr_p))
{
gcc_assert (!VOID_TYPE_P (TREE_TYPE (*expr_p)));
*expr_p = get_formal_tmp_var (*expr_p, pre_p);
}
else
{
#ifdef ENABLE_GIMPLE_CHECKING
if (!(fallback & fb_mayfail))
{
fprintf (stderr, "gimplification failed:\n");
print_generic_expr (stderr, *expr_p);
debug_tree (*expr_p);
internal_error ("gimplification failed");
}
#endif
gcc_assert (fallback & fb_mayfail);
ret = GS_ERROR;
goto out;
}
gcc_assert ((*gimple_test_f) (*expr_p));
if (!gimple_seq_empty_p (internal_post))
{
annotate_all_with_location (internal_post, input_location);
gimplify_seq_add_seq (pre_p, internal_post);
}
out:
input_location = saved_location;
return ret;
}
static enum gimplify_status
gimplify_expr (tree *expr_p, gimple_seq *pre_p, gimple_seq *post_p,
bool (*gimple_test_f) (tree), fallback_t fallback,
bool allow_ssa)
{
bool was_ssa_name_p = TREE_CODE (*expr_p) == SSA_NAME;
enum gimplify_status ret = gimplify_expr (expr_p, pre_p, post_p,
gimple_test_f, fallback);
if (! allow_ssa
&& TREE_CODE (*expr_p) == SSA_NAME)
{
tree name = *expr_p;
if (was_ssa_name_p)
*expr_p = get_initialized_tmp_var (*expr_p, pre_p, NULL, false);
else
{
*expr_p = create_tmp_reg (TREE_TYPE (name));
gimple_set_lhs (SSA_NAME_DEF_STMT (name), *expr_p);
release_ssa_name (name);
}
}
return ret;
}
void
gimplify_type_sizes (tree type, gimple_seq *list_p)
{
tree field, t;
if (type == NULL || type == error_mark_node)
return;
type = TYPE_MAIN_VARIANT (type);
if (TYPE_SIZES_GIMPLIFIED (type))
return;
TYPE_SIZES_GIMPLIFIED (type) = 1;
switch (TREE_CODE (type))
{
case INTEGER_TYPE:
case ENUMERAL_TYPE:
case BOOLEAN_TYPE:
case REAL_TYPE:
case FIXED_POINT_TYPE:
gimplify_one_sizepos (&TYPE_MIN_VALUE (type), list_p);
gimplify_one_sizepos (&TYPE_MAX_VALUE (type), list_p);
for (t = TYPE_NEXT_VARIANT (type); t; t = TYPE_NEXT_VARIANT (t))
{
TYPE_MIN_VALUE (t) = TYPE_MIN_VALUE (type);
TYPE_MAX_VALUE (t) = TYPE_MAX_VALUE (type);
}
break;
case ARRAY_TYPE:
gimplify_type_sizes (TREE_TYPE (type), list_p);
gimplify_type_sizes (TYPE_DOMAIN (type), list_p);
if (!(TYPE_NAME (type)
&& TREE_CODE (TYPE_NAME (type)) == TYPE_DECL
&& DECL_IGNORED_P (TYPE_NAME (type)))
&& TYPE_DOMAIN (type)
&& INTEGRAL_TYPE_P (TYPE_DOMAIN (type)))
{
t = TYPE_MIN_VALUE (TYPE_DOMAIN (type));
if (t && VAR_P (t) && DECL_ARTIFICIAL (t))
DECL_IGNORED_P (t) = 0;
t = TYPE_MAX_VALUE (TYPE_DOMAIN (type));
if (t && VAR_P (t) && DECL_ARTIFICIAL (t))
DECL_IGNORED_P (t) = 0;
}
break;
case RECORD_TYPE:
case UNION_TYPE:
case QUAL_UNION_TYPE:
for (field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
if (TREE_CODE (field) == FIELD_DECL)
{
gimplify_one_sizepos (&DECL_FIELD_OFFSET (field), list_p);
gimplify_one_sizepos (&DECL_SIZE (field), list_p);
gimplify_one_sizepos (&DECL_SIZE_UNIT (field), list_p);
gimplify_type_sizes (TREE_TYPE (field), list_p);
}
break;
case POINTER_TYPE:
case REFERENCE_TYPE:
break;
default:
break;
}
gimplify_one_sizepos (&TYPE_SIZE (type), list_p);
gimplify_one_sizepos (&TYPE_SIZE_UNIT (type), list_p);
for (t = TYPE_NEXT_VARIANT (type); t; t = TYPE_NEXT_VARIANT (t))
{
TYPE_SIZE (t) = TYPE_SIZE (type);
TYPE_SIZE_UNIT (t) = TYPE_SIZE_UNIT (type);
TYPE_SIZES_GIMPLIFIED (t) = 1;
}
}
void
gimplify_one_sizepos (tree *expr_p, gimple_seq *stmt_p)
{
tree expr = *expr_p;
if (expr == NULL_TREE
|| is_gimple_constant (expr)
|| TREE_CODE (expr) == VAR_DECL
|| CONTAINS_PLACEHOLDER_P (expr))
return;
*expr_p = unshare_expr (expr);
gimplify_expr (expr_p, stmt_p, NULL, is_gimple_val, fb_rvalue, false);
if (is_gimple_constant (*expr_p))
*expr_p = get_initialized_tmp_var (*expr_p, stmt_p, NULL, false);
}
gbind *
gimplify_body (tree fndecl, bool do_parms)
{
location_t saved_location = input_location;
gimple_seq parm_stmts, parm_cleanup = NULL, seq;
gimple *outer_stmt;
gbind *outer_bind;
struct cgraph_node *cgn;
timevar_push (TV_TREE_GIMPLIFY);
init_tree_ssa (cfun);
default_rtl_profile ();
gcc_assert (gimplify_ctxp == NULL);
push_gimplify_context (true);
if (flag_openacc || flag_openmp)
{
gcc_assert (gimplify_omp_ctxp == NULL);
if (lookup_attribute ("omp declare target", DECL_ATTRIBUTES (fndecl)))
gimplify_omp_ctxp = new_omp_context (ORT_TARGET);
}
unshare_body (fndecl);
unvisit_body (fndecl);
cgn = cgraph_node::get (fndecl);
if (cgn && cgn->origin)
nonlocal_vlas = new hash_set<tree>;
input_location = DECL_SOURCE_LOCATION (fndecl);
parm_stmts = do_parms ? gimplify_parameters (&parm_cleanup) : NULL;
seq = NULL;
gimplify_stmt (&DECL_SAVED_TREE (fndecl), &seq);
outer_stmt = gimple_seq_first_stmt (seq);
if (!outer_stmt)
{
outer_stmt = gimple_build_nop ();
gimplify_seq_add_stmt (&seq, outer_stmt);
}
if (gimple_code (outer_stmt) == GIMPLE_BIND
&& gimple_seq_first (seq) == gimple_seq_last (seq))
outer_bind = as_a <gbind *> (outer_stmt);
else
outer_bind = gimple_build_bind (NULL_TREE, seq, NULL);
DECL_SAVED_TREE (fndecl) = NULL_TREE;
if (!gimple_seq_empty_p (parm_stmts))
{
tree parm;
gimplify_seq_add_seq (&parm_stmts, gimple_bind_body (outer_bind));
if (parm_cleanup)
{
gtry *g = gimple_build_try (parm_stmts, parm_cleanup,
GIMPLE_TRY_FINALLY);
parm_stmts = NULL;
gimple_seq_add_stmt (&parm_stmts, g);
}
gimple_bind_set_body (outer_bind, parm_stmts);
for (parm = DECL_ARGUMENTS (current_function_decl);
parm; parm = DECL_CHAIN (parm))
if (DECL_HAS_VALUE_EXPR_P (parm))
{
DECL_HAS_VALUE_EXPR_P (parm) = 0;
DECL_IGNORED_P (parm) = 0;
}
}
if (nonlocal_vlas)
{
if (nonlocal_vla_vars)
{
if (gimple_bind_block (outer_bind)
== DECL_INITIAL (current_function_decl))
declare_vars (nonlocal_vla_vars, outer_bind, true);
else
BLOCK_VARS (DECL_INITIAL (current_function_decl))
= chainon (BLOCK_VARS (DECL_INITIAL (current_function_decl)),
nonlocal_vla_vars);
nonlocal_vla_vars = NULL_TREE;
}
delete nonlocal_vlas;
nonlocal_vlas = NULL;
}
if ((flag_openacc || flag_openmp || flag_openmp_simd)
&& gimplify_omp_ctxp)
{
delete_omp_context (gimplify_omp_ctxp);
gimplify_omp_ctxp = NULL;
}
pop_gimplify_context (outer_bind);
gcc_assert (gimplify_ctxp == NULL);
if (flag_checking && !seen_error ())
verify_gimple_in_seq (gimple_bind_body (outer_bind));
timevar_pop (TV_TREE_GIMPLIFY);
input_location = saved_location;
return outer_bind;
}
typedef char *char_p; 
static bool
flag_instrument_functions_exclude_p (tree fndecl)
{
vec<char_p> *v;
v = (vec<char_p> *) flag_instrument_functions_exclude_functions;
if (v && v->length () > 0)
{
const char *name;
int i;
char *s;
name = lang_hooks.decl_printable_name (fndecl, 0);
FOR_EACH_VEC_ELT (*v, i, s)
if (strstr (name, s) != NULL)
return true;
}
v = (vec<char_p> *) flag_instrument_functions_exclude_files;
if (v && v->length () > 0)
{
const char *name;
int i;
char *s;
name = DECL_SOURCE_FILE (fndecl);
FOR_EACH_VEC_ELT (*v, i, s)
if (strstr (name, s) != NULL)
return true;
}
return false;
}
void
gimplify_function_tree (tree fndecl)
{
tree parm, ret;
gimple_seq seq;
gbind *bind;
gcc_assert (!gimple_body (fndecl));
if (DECL_STRUCT_FUNCTION (fndecl))
push_cfun (DECL_STRUCT_FUNCTION (fndecl));
else
push_struct_function (fndecl);
cfun->curr_properties |= PROP_gimple_lva;
for (parm = DECL_ARGUMENTS (fndecl); parm ; parm = DECL_CHAIN (parm))
{
if ((TREE_CODE (TREE_TYPE (parm)) == COMPLEX_TYPE
|| TREE_CODE (TREE_TYPE (parm)) == VECTOR_TYPE)
&& !TREE_THIS_VOLATILE (parm)
&& !needs_to_live_in_memory (parm))
DECL_GIMPLE_REG_P (parm) = 1;
}
ret = DECL_RESULT (fndecl);
if ((TREE_CODE (TREE_TYPE (ret)) == COMPLEX_TYPE
|| TREE_CODE (TREE_TYPE (ret)) == VECTOR_TYPE)
&& !needs_to_live_in_memory (ret))
DECL_GIMPLE_REG_P (ret) = 1;
if (asan_sanitize_use_after_scope () && sanitize_flags_p (SANITIZE_ADDRESS))
asan_poisoned_variables = new hash_set<tree> ();
bind = gimplify_body (fndecl, true);
if (asan_poisoned_variables)
{
delete asan_poisoned_variables;
asan_poisoned_variables = NULL;
}
seq = NULL;
gimple_seq_add_stmt (&seq, bind);
gimple_set_body (fndecl, seq);
if (flag_instrument_function_entry_exit
&& !DECL_NO_INSTRUMENT_FUNCTION_ENTRY_EXIT (fndecl)
&& !(DECL_DECLARED_INLINE_P (fndecl)
&& DECL_EXTERNAL (fndecl)
&& DECL_DISREGARD_INLINE_LIMITS (fndecl))
&& !flag_instrument_functions_exclude_p (fndecl))
{
tree x;
gbind *new_bind;
gimple *tf;
gimple_seq cleanup = NULL, body = NULL;
tree tmp_var;
gcall *call;
x = builtin_decl_implicit (BUILT_IN_RETURN_ADDRESS);
call = gimple_build_call (x, 1, integer_zero_node);
tmp_var = create_tmp_var (ptr_type_node, "return_addr");
gimple_call_set_lhs (call, tmp_var);
gimplify_seq_add_stmt (&cleanup, call);
x = builtin_decl_implicit (BUILT_IN_PROFILE_FUNC_EXIT);
call = gimple_build_call (x, 2,
build_fold_addr_expr (current_function_decl),
tmp_var);
gimplify_seq_add_stmt (&cleanup, call);
tf = gimple_build_try (seq, cleanup, GIMPLE_TRY_FINALLY);
x = builtin_decl_implicit (BUILT_IN_RETURN_ADDRESS);
call = gimple_build_call (x, 1, integer_zero_node);
tmp_var = create_tmp_var (ptr_type_node, "return_addr");
gimple_call_set_lhs (call, tmp_var);
gimplify_seq_add_stmt (&body, call);
x = builtin_decl_implicit (BUILT_IN_PROFILE_FUNC_ENTER);
call = gimple_build_call (x, 2,
build_fold_addr_expr (current_function_decl),
tmp_var);
gimplify_seq_add_stmt (&body, call);
gimplify_seq_add_stmt (&body, tf);
new_bind = gimple_build_bind (NULL, body, NULL);
seq = NULL;
gimple_seq_add_stmt (&seq, new_bind);
gimple_set_body (fndecl, seq);
bind = new_bind;
}
if (sanitize_flags_p (SANITIZE_THREAD))
{
gcall *call = gimple_build_call_internal (IFN_TSAN_FUNC_EXIT, 0);
gimple *tf = gimple_build_try (seq, call, GIMPLE_TRY_FINALLY);
gbind *new_bind = gimple_build_bind (NULL, tf, NULL);
seq = NULL;
gimple_seq_add_stmt (&seq, new_bind);
gimple_set_body (fndecl, seq);
}
DECL_SAVED_TREE (fndecl) = NULL_TREE;
cfun->curr_properties |= PROP_gimple_any;
pop_cfun ();
dump_function (TDI_gimple, fndecl);
}
static tree
dummy_object (tree type)
{
tree t = build_int_cst (build_pointer_type (type), 0);
return build2 (MEM_REF, type, t, t);
}
enum gimplify_status
gimplify_va_arg_expr (tree *expr_p, gimple_seq *pre_p,
gimple_seq *post_p ATTRIBUTE_UNUSED)
{
tree promoted_type, have_va_type;
tree valist = TREE_OPERAND (*expr_p, 0);
tree type = TREE_TYPE (*expr_p);
tree t, tag, aptag;
location_t loc = EXPR_LOCATION (*expr_p);
have_va_type = TREE_TYPE (valist);
if (have_va_type == error_mark_node)
return GS_ERROR;
have_va_type = targetm.canonical_va_list_type (have_va_type);
if (have_va_type == NULL_TREE
&& POINTER_TYPE_P (TREE_TYPE (valist)))
have_va_type
= targetm.canonical_va_list_type (TREE_TYPE (TREE_TYPE (valist)));
gcc_assert (have_va_type != NULL_TREE);
if ((promoted_type = lang_hooks.types.type_promotes_to (type))
!= type)
{
static bool gave_help;
bool warned;
source_location xloc
= expansion_point_location_if_in_system_header (loc);
warned = warning_at (xloc, 0,
"%qT is promoted to %qT when passed through %<...%>",
type, promoted_type);
if (!gave_help && warned)
{
gave_help = true;
inform (xloc, "(so you should pass %qT not %qT to %<va_arg%>)",
promoted_type, type);
}
if (warned)
inform (xloc, "if this code is reached, the program will abort");
gimplify_and_add (valist, pre_p);
t = build_call_expr_loc (loc,
builtin_decl_implicit (BUILT_IN_TRAP), 0);
gimplify_and_add (t, pre_p);
*expr_p = dummy_object (type);
return GS_ALL_DONE;
}
tag = build_int_cst (build_pointer_type (type), 0);
aptag = build_int_cst (TREE_TYPE (valist), 0);
*expr_p = build_call_expr_internal_loc (loc, IFN_VA_ARG, type, 3,
valist, tag, aptag);
cfun->curr_properties &= ~PROP_gimple_lva;
return GS_OK;
}
gimple *
gimplify_assign (tree dst, tree src, gimple_seq *seq_p)
{
tree t = build2 (MODIFY_EXPR, TREE_TYPE (dst), dst, src);
gimplify_and_add (t, seq_p);
ggc_free (t);
return gimple_seq_last_stmt (*seq_p);
}
inline hashval_t
gimplify_hasher::hash (const elt_t *p)
{
tree t = p->val;
return iterative_hash_expr (t, 0);
}
inline bool
gimplify_hasher::equal (const elt_t *p1, const elt_t *p2)
{
tree t1 = p1->val;
tree t2 = p2->val;
enum tree_code code = TREE_CODE (t1);
if (TREE_CODE (t2) != code
|| TREE_TYPE (t1) != TREE_TYPE (t2))
return false;
if (!operand_equal_p (t1, t2, 0))
return false;
gcc_checking_assert (hash (p1) == hash (p2));
return true;
}
