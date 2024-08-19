#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "cp-tree.h"
#include "timevar.h"
#include "stringpool.h"
#include "varasm.h"
#include "attribs.h"
#include "stor-layout.h"
#include "intl.h"
#include "c-family/c-objc.h"
#include "cp-objcp-common.h"
#include "toplev.h"
#include "tree-iterator.h"
#include "type-utils.h"
#include "gimplify.h"
#include "gcc-rich-location.h"
#include "selftest.h"
typedef int (*tree_fn_t) (tree, void*);
struct GTY ((chain_next ("%h.next"))) pending_template
{
struct pending_template *next;
struct tinst_level *tinst;
};
static GTY(()) struct pending_template *pending_templates;
static GTY(()) struct pending_template *last_pending_template;
int processing_template_parmlist;
static int template_header_count;
static GTY(()) tree saved_trees;
static vec<int> inline_parm_levels;
static GTY(()) struct tinst_level *current_tinst_level;
static GTY(()) tree saved_access_scope;
static tree cur_stmt_expr;
local_specialization_stack::local_specialization_stack (lss_policy policy)
: saved (local_specializations)
{
if (policy == lss_blank || !saved)
local_specializations = new hash_map<tree, tree>;
else
local_specializations = new hash_map<tree, tree>(*saved);
}
local_specialization_stack::~local_specialization_stack ()
{
delete local_specializations;
local_specializations = saved;
}
static bool excessive_deduction_depth;
struct GTY((for_user)) spec_entry
{
tree tmpl;
tree args;
tree spec;
};
struct spec_hasher : ggc_ptr_hash<spec_entry>
{
static hashval_t hash (spec_entry *);
static bool equal (spec_entry *, spec_entry *);
};
static GTY (()) hash_table<spec_hasher> *decl_specializations;
static GTY (()) hash_table<spec_hasher> *type_specializations;
static GTY(()) vec<tree, va_gc> *canonical_template_parms;
#define UNIFY_ALLOW_NONE 0
#define UNIFY_ALLOW_MORE_CV_QUAL 1
#define UNIFY_ALLOW_LESS_CV_QUAL 2
#define UNIFY_ALLOW_DERIVED 4
#define UNIFY_ALLOW_INTEGER 8
#define UNIFY_ALLOW_OUTER_LEVEL 16
#define UNIFY_ALLOW_OUTER_MORE_CV_QUAL 32
#define UNIFY_ALLOW_OUTER_LESS_CV_QUAL 64
enum template_base_result {
tbr_incomplete_type,
tbr_ambiguous_baseclass,
tbr_success
};
static void push_access_scope (tree);
static void pop_access_scope (tree);
static bool resolve_overloaded_unification (tree, tree, tree, tree,
unification_kind_t, int,
bool);
static int try_one_overload (tree, tree, tree, tree, tree,
unification_kind_t, int, bool, bool);
static int unify (tree, tree, tree, tree, int, bool);
static void add_pending_template (tree);
static tree reopen_tinst_level (struct tinst_level *);
static tree tsubst_initializer_list (tree, tree);
static tree get_partial_spec_bindings (tree, tree, tree);
static tree coerce_template_parms (tree, tree, tree, tsubst_flags_t,
bool, bool);
static tree coerce_innermost_template_parms (tree, tree, tree, tsubst_flags_t,
bool, bool);
static void tsubst_enum	(tree, tree, tree);
static tree add_to_template_args (tree, tree);
static tree add_outermost_template_args (tree, tree);
static bool check_instantiated_args (tree, tree, tsubst_flags_t);
static int maybe_adjust_types_for_deduction (unification_kind_t, tree*, tree*,
tree);
static int type_unification_real (tree, tree, tree, const tree *,
unsigned int, int, unification_kind_t, int,
vec<deferred_access_check, va_gc> **,
bool);
static void note_template_header (int);
static tree convert_nontype_argument_function (tree, tree, tsubst_flags_t);
static tree convert_nontype_argument (tree, tree, tsubst_flags_t);
static tree convert_template_argument (tree, tree, tree,
tsubst_flags_t, int, tree);
static tree for_each_template_parm (tree, tree_fn_t, void*,
hash_set<tree> *, bool, tree_fn_t = NULL);
static tree expand_template_argument_pack (tree);
static tree build_template_parm_index (int, int, int, tree, tree);
static bool inline_needs_template_parms (tree, bool);
static void push_inline_template_parms_recursive (tree, int);
static tree reduce_template_parm_level (tree, tree, int, tree, tsubst_flags_t);
static int mark_template_parm (tree, void *);
static int template_parm_this_level_p (tree, void *);
static tree tsubst_friend_function (tree, tree);
static tree tsubst_friend_class (tree, tree);
static int can_complete_type_without_circularity (tree);
static tree get_bindings (tree, tree, tree, bool);
static int template_decl_level (tree);
static int check_cv_quals_for_unify (int, tree, tree);
static void template_parm_level_and_index (tree, int*, int*);
static int unify_pack_expansion (tree, tree, tree,
tree, unification_kind_t, bool, bool);
static tree copy_template_args (tree);
static tree tsubst_template_arg (tree, tree, tsubst_flags_t, tree);
static tree tsubst_template_args (tree, tree, tsubst_flags_t, tree);
static tree tsubst_template_parms (tree, tree, tsubst_flags_t);
static tree most_specialized_partial_spec (tree, tsubst_flags_t);
static tree tsubst_aggr_type (tree, tree, tsubst_flags_t, tree, int);
static tree tsubst_arg_types (tree, tree, tree, tsubst_flags_t, tree);
static tree tsubst_function_type (tree, tree, tsubst_flags_t, tree);
static bool check_specialization_scope (void);
static tree process_partial_specialization (tree);
static void set_current_access_from_decl (tree);
static enum template_base_result get_template_base (tree, tree, tree, tree,
bool , tree *);
static tree try_class_unification (tree, tree, tree, tree, bool);
static int coerce_template_template_parms (tree, tree, tsubst_flags_t,
tree, tree);
static bool template_template_parm_bindings_ok_p (tree, tree);
static void tsubst_default_arguments (tree, tsubst_flags_t);
static tree for_each_template_parm_r (tree *, int *, void *);
static tree copy_default_args_to_explicit_spec_1 (tree, tree);
static void copy_default_args_to_explicit_spec (tree);
static bool invalid_nontype_parm_type_p (tree, tsubst_flags_t);
static bool dependent_template_arg_p (tree);
static bool any_template_arguments_need_structural_equality_p (tree);
static bool dependent_type_p_r (tree);
static tree tsubst_copy	(tree, tree, tsubst_flags_t, tree);
static tree tsubst_decl (tree, tree, tsubst_flags_t);
static void perform_typedefs_access_check (tree tmpl, tree targs);
static void append_type_to_template_for_access_check_1 (tree, tree, tree,
location_t);
static tree listify (tree);
static tree listify_autos (tree, tree);
static tree tsubst_template_parm (tree, tree, tsubst_flags_t);
static tree instantiate_alias_template (tree, tree, tsubst_flags_t);
static bool complex_alias_template_p (const_tree tmpl);
static tree tsubst_attributes (tree, tree, tsubst_flags_t, tree);
static tree canonicalize_expr_argument (tree, tsubst_flags_t);
static tree make_argument_pack (tree);
static void register_parameter_specializations (tree, tree);
static tree enclosing_instantiation_of (tree tctx);
static void
push_access_scope (tree t)
{
gcc_assert (VAR_OR_FUNCTION_DECL_P (t)
|| TREE_CODE (t) == TYPE_DECL);
if (DECL_FRIEND_CONTEXT (t))
push_nested_class (DECL_FRIEND_CONTEXT (t));
else if (DECL_CLASS_SCOPE_P (t))
push_nested_class (DECL_CONTEXT (t));
else
push_to_top_level ();
if (TREE_CODE (t) == FUNCTION_DECL)
{
saved_access_scope = tree_cons
(NULL_TREE, current_function_decl, saved_access_scope);
current_function_decl = t;
}
}
static void
pop_access_scope (tree t)
{
if (TREE_CODE (t) == FUNCTION_DECL)
{
current_function_decl = TREE_VALUE (saved_access_scope);
saved_access_scope = TREE_CHAIN (saved_access_scope);
}
if (DECL_FRIEND_CONTEXT (t) || DECL_CLASS_SCOPE_P (t))
pop_nested_class ();
else
pop_from_top_level ();
}
tree
finish_member_template_decl (tree decl)
{
if (decl == error_mark_node)
return error_mark_node;
gcc_assert (DECL_P (decl));
if (TREE_CODE (decl) == TYPE_DECL)
{
tree type;
type = TREE_TYPE (decl);
if (type == error_mark_node)
return error_mark_node;
if (MAYBE_CLASS_TYPE_P (type)
&& CLASSTYPE_TEMPLATE_INFO (type)
&& !CLASSTYPE_TEMPLATE_SPECIALIZATION (type))
{
tree tmpl = CLASSTYPE_TI_TEMPLATE (type);
check_member_template (tmpl);
return tmpl;
}
return NULL_TREE;
}
else if (TREE_CODE (decl) == FIELD_DECL)
error ("data member %qD cannot be a member template", decl);
else if (DECL_TEMPLATE_INFO (decl))
{
if (!DECL_TEMPLATE_SPECIALIZATION (decl))
{
check_member_template (DECL_TI_TEMPLATE (decl));
return DECL_TI_TEMPLATE (decl);
}
else
return decl;
}
else
error ("invalid member template declaration %qD", decl);
return error_mark_node;
}
tree
build_template_info (tree template_decl, tree template_args)
{
tree result = make_node (TEMPLATE_INFO);
TI_TEMPLATE (result) = template_decl;
TI_ARGS (result) = template_args;
return result;
}
tree
get_template_info (const_tree t)
{
tree tinfo = NULL_TREE;
if (!t || t == error_mark_node)
return NULL;
if (TREE_CODE (t) == NAMESPACE_DECL
|| TREE_CODE (t) == PARM_DECL)
return NULL;
if (DECL_P (t) && DECL_LANG_SPECIFIC (t))
tinfo = DECL_TEMPLATE_INFO (t);
if (!tinfo && DECL_IMPLICIT_TYPEDEF_P (t))
t = TREE_TYPE (t);
if (OVERLOAD_TYPE_P (t))
tinfo = TYPE_TEMPLATE_INFO (t);
else if (TREE_CODE (t) == BOUND_TEMPLATE_TEMPLATE_PARM)
tinfo = TEMPLATE_TEMPLATE_PARM_TEMPLATE_INFO (t);
return tinfo;
}
int
template_class_depth (tree type)
{
int depth;
for (depth = 0; type && TREE_CODE (type) != NAMESPACE_DECL; )
{
tree tinfo = get_template_info (type);
if (tinfo && PRIMARY_TEMPLATE_P (TI_TEMPLATE (tinfo))
&& uses_template_parms (INNERMOST_TEMPLATE_ARGS (TI_ARGS (tinfo))))
++depth;
if (DECL_P (type))
type = CP_DECL_CONTEXT (type);
else if (LAMBDA_TYPE_P (type))
type = LAMBDA_TYPE_EXTRA_SCOPE (type);
else
type = CP_TYPE_CONTEXT (type);
}
return depth;
}
static bool
inline_needs_template_parms (tree decl, bool nsdmi)
{
if (!decl || (!nsdmi && ! DECL_TEMPLATE_INFO (decl)))
return false;
return (TMPL_PARMS_DEPTH (DECL_TEMPLATE_PARMS (most_general_template (decl)))
> (processing_template_decl + DECL_TEMPLATE_SPECIALIZATION (decl)));
}
static void
push_inline_template_parms_recursive (tree parmlist, int levels)
{
tree parms = TREE_VALUE (parmlist);
int i;
if (levels > 1)
push_inline_template_parms_recursive (TREE_CHAIN (parmlist), levels - 1);
++processing_template_decl;
current_template_parms
= tree_cons (size_int (processing_template_decl),
parms, current_template_parms);
TEMPLATE_PARMS_FOR_INLINE (current_template_parms) = 1;
begin_scope (TREE_VEC_LENGTH (parms) ? sk_template_parms : sk_template_spec,
NULL);
for (i = 0; i < TREE_VEC_LENGTH (parms); ++i)
{
tree parm = TREE_VALUE (TREE_VEC_ELT (parms, i));
if (error_operand_p (parm))
continue;
gcc_assert (DECL_P (parm));
switch (TREE_CODE (parm))
{
case TYPE_DECL:
case TEMPLATE_DECL:
pushdecl (parm);
break;
case PARM_DECL:
pushdecl (TEMPLATE_PARM_DECL (DECL_INITIAL (parm)));
break;
default:
gcc_unreachable ();
}
}
}
void
maybe_begin_member_template_processing (tree decl)
{
tree parms;
int levels = 0;
bool nsdmi = TREE_CODE (decl) == FIELD_DECL;
if (nsdmi)
{
tree ctx = DECL_CONTEXT (decl);
decl = (CLASSTYPE_TEMPLATE_INFO (ctx)
&& uses_template_parms (ctx)
? CLASSTYPE_TI_TEMPLATE (ctx) : NULL_TREE);
}
if (inline_needs_template_parms (decl, nsdmi))
{
parms = DECL_TEMPLATE_PARMS (most_general_template (decl));
levels = TMPL_PARMS_DEPTH (parms) - processing_template_decl;
if (DECL_TEMPLATE_SPECIALIZATION (decl))
{
--levels;
parms = TREE_CHAIN (parms);
}
push_inline_template_parms_recursive (parms, levels);
}
inline_parm_levels.safe_push (levels);
}
void
maybe_end_member_template_processing (void)
{
int i;
int last;
if (inline_parm_levels.length () == 0)
return;
last = inline_parm_levels.pop ();
for (i = 0; i < last; ++i)
{
--processing_template_decl;
current_template_parms = TREE_CHAIN (current_template_parms);
poplevel (0, 0, 0);
}
}
static tree
add_to_template_args (tree args, tree extra_args)
{
tree new_args;
int extra_depth;
int i;
int j;
if (args == NULL_TREE || extra_args == error_mark_node)
return extra_args;
extra_depth = TMPL_ARGS_DEPTH (extra_args);
new_args = make_tree_vec (TMPL_ARGS_DEPTH (args) + extra_depth);
for (i = 1; i <= TMPL_ARGS_DEPTH (args); ++i)
SET_TMPL_ARGS_LEVEL (new_args, i, TMPL_ARGS_LEVEL (args, i));
for (j = 1; j <= extra_depth; ++j, ++i)
SET_TMPL_ARGS_LEVEL (new_args, i, TMPL_ARGS_LEVEL (extra_args, j));
return new_args;
}
static tree
add_outermost_template_args (tree args, tree extra_args)
{
tree new_args;
gcc_assert (TMPL_ARGS_DEPTH (args) >= TMPL_ARGS_DEPTH (extra_args));
if (TMPL_ARGS_DEPTH (args) == TMPL_ARGS_DEPTH (extra_args))
return extra_args;
TREE_VEC_LENGTH (args) -= TMPL_ARGS_DEPTH (extra_args);
new_args = add_to_template_args (args, extra_args);
TREE_VEC_LENGTH (args) += TMPL_ARGS_DEPTH (extra_args);
return new_args;
}
tree
get_innermost_template_args (tree args, int n)
{
tree new_args;
int extra_levels;
int i;
gcc_assert (n >= 0);
if (n == 1)
return TMPL_ARGS_LEVEL (args, TMPL_ARGS_DEPTH (args));
extra_levels = TMPL_ARGS_DEPTH (args) - n;
gcc_assert (extra_levels >= 0);
if (extra_levels == 0)
return args;
new_args = make_tree_vec (n);
for (i = 1; i <= n; ++i)
SET_TMPL_ARGS_LEVEL (new_args, i,
TMPL_ARGS_LEVEL (args, i + extra_levels));
return new_args;
}
static tree
strip_innermost_template_args (tree args, int extra_levels)
{
tree new_args;
int n = TMPL_ARGS_DEPTH (args) - extra_levels;
int i;
gcc_assert (n >= 0);
if (n == 1)
return TMPL_ARGS_LEVEL (args, 1);
gcc_assert (extra_levels >= 0);
if (extra_levels == 0)
return args;
new_args = make_tree_vec (n);
for (i = 1; i <= n; ++i)
SET_TMPL_ARGS_LEVEL (new_args, i,
TMPL_ARGS_LEVEL (args, i));
return new_args;
}
void
begin_template_parm_list (void)
{
begin_scope (sk_template_parms, NULL);
++processing_template_decl;
++processing_template_parmlist;
note_template_header (0);
current_template_parms
= tree_cons (size_int (processing_template_decl),
make_tree_vec (0),
current_template_parms);
}
static bool
check_specialization_scope (void)
{
tree scope = current_scope ();
if (scope && TREE_CODE (scope) != NAMESPACE_DECL)
{
error ("explicit specialization in non-namespace scope %qD", scope);
return false;
}
if (current_template_parms)
{
error ("enclosing class templates are not explicitly specialized");
return false;
}
return true;
}
bool
begin_specialization (void)
{
begin_scope (sk_template_spec, NULL);
note_template_header (1);
return check_specialization_scope ();
}
void
end_specialization (void)
{
finish_scope ();
reset_specialization ();
}
void
reset_specialization (void)
{
processing_specialization = 0;
template_header_count = 0;
}
static void
note_template_header (int specialization)
{
processing_specialization = specialization;
template_header_count++;
}
void
begin_explicit_instantiation (void)
{
gcc_assert (!processing_explicit_instantiation);
processing_explicit_instantiation = true;
}
void
end_explicit_instantiation (void)
{
gcc_assert (processing_explicit_instantiation);
processing_explicit_instantiation = false;
}
static bool
check_specialization_namespace (tree tmpl)
{
tree tpl_ns = decl_namespace_context (tmpl);
if (current_scope() != DECL_CONTEXT (tmpl)
&& !at_namespace_scope_p ())
{
error ("specialization of %qD must appear at namespace scope", tmpl);
return false;
}
if (is_nested_namespace (current_namespace, tpl_ns, cxx_dialect < cxx11))
return true;
else
{
permerror (input_location,
"specialization of %qD in different namespace", tmpl);
inform (DECL_SOURCE_LOCATION (tmpl),
"  from definition of %q#D", tmpl);
return false;
}
}
static void
check_explicit_instantiation_namespace (tree spec)
{
tree ns;
ns = decl_namespace_context (spec);
if (!is_nested_namespace (current_namespace, ns))
permerror (input_location, "explicit instantiation of %qD in namespace %qD "
"(which does not enclose namespace %qD)",
spec, current_namespace, ns);
}
static tree
maybe_new_partial_specialization (tree type)
{
if (CLASSTYPE_IMPLICIT_INSTANTIATION (type) && !COMPLETE_TYPE_P (type))
return type;
if (flag_concepts && CLASSTYPE_TEMPLATE_SPECIALIZATION (type))
{
tree tmpl = CLASSTYPE_TI_TEMPLATE (type);
tree args = CLASSTYPE_TI_ARGS (type);
if (!current_template_parms)
return NULL_TREE;
if (DECL_SELF_REFERENCE_P (TYPE_NAME (type)))
return NULL_TREE;
tree type_constr = current_template_constraints ();
if (type == TREE_TYPE (tmpl))
{
tree main_constr = get_constraints (tmpl);
if (equivalent_constraints (type_constr, main_constr))
return NULL_TREE;
}
tree specs = DECL_TEMPLATE_SPECIALIZATIONS (tmpl);
while (specs)
{
tree spec_tmpl = TREE_VALUE (specs);
tree spec_args = TREE_PURPOSE (specs);
tree spec_constr = get_constraints (spec_tmpl);
if (comp_template_args (args, spec_args)
&& equivalent_constraints (type_constr, spec_constr))
return NULL_TREE;
specs = TREE_CHAIN (specs);
}
tree t = make_class_type (TREE_CODE (type));
CLASSTYPE_DECLARED_CLASS (t) = CLASSTYPE_DECLARED_CLASS (type);
SET_TYPE_TEMPLATE_INFO (t, build_template_info (tmpl, args));
TYPE_CANONICAL (t) = TYPE_CANONICAL (type);
tree d = create_implicit_typedef (DECL_NAME (tmpl), t);
DECL_CONTEXT (d) = TYPE_CONTEXT (t);
DECL_SOURCE_LOCATION (d) = input_location;
return t;
}
return NULL_TREE;
}
tree
maybe_process_partial_specialization (tree type)
{
tree context;
if (type == error_mark_node)
return error_mark_node;
if (CLASS_TYPE_P (type) && CLASSTYPE_LAMBDA_EXPR (type))
return type;
if (TREE_CODE (type) == BOUND_TEMPLATE_TEMPLATE_PARM)
{
error ("name of class shadows template template parameter %qD",
TYPE_NAME (type));
return error_mark_node;
}
context = TYPE_CONTEXT (type);
if (TYPE_ALIAS_P (type))
{
tree tinfo = TYPE_ALIAS_TEMPLATE_INFO (type);
if (tinfo && DECL_ALIAS_TEMPLATE_P (TI_TEMPLATE (tinfo)))
error ("specialization of alias template %qD",
TI_TEMPLATE (tinfo));
else
error ("explicit specialization of non-template %qT", type);
return error_mark_node;
}
else if (CLASS_TYPE_P (type) && CLASSTYPE_USE_TEMPLATE (type))
{
if (tree t = maybe_new_partial_specialization (type))
{
if (!check_specialization_namespace (CLASSTYPE_TI_TEMPLATE (t))
&& !at_namespace_scope_p ())
return error_mark_node;
SET_CLASSTYPE_TEMPLATE_SPECIALIZATION (t);
DECL_SOURCE_LOCATION (TYPE_MAIN_DECL (t)) = input_location;
if (processing_template_decl)
{
tree decl = push_template_decl (TYPE_MAIN_DECL (t));
if (decl == error_mark_node)
return error_mark_node;
return TREE_TYPE (decl);
}
}
else if (CLASSTYPE_TEMPLATE_INSTANTIATION (type))
error ("specialization of %qT after instantiation", type);
else if (errorcount && !processing_specialization
&& CLASSTYPE_TEMPLATE_SPECIALIZATION (type)
&& !uses_template_parms (CLASSTYPE_TI_ARGS (type)))
return error_mark_node;
}
else if (CLASS_TYPE_P (type)
&& !CLASSTYPE_USE_TEMPLATE (type)
&& CLASSTYPE_TEMPLATE_INFO (type)
&& context && CLASS_TYPE_P (context)
&& CLASSTYPE_TEMPLATE_INFO (context))
{
if (CLASSTYPE_IMPLICIT_INSTANTIATION (context)
&& !COMPLETE_TYPE_P (type))
{
tree t;
tree tmpl = CLASSTYPE_TI_TEMPLATE (type);
if (current_namespace
!= decl_namespace_context (tmpl))
{
permerror (input_location,
"specializing %q#T in different namespace", type);
permerror (DECL_SOURCE_LOCATION (tmpl),
"  from definition of %q#D", tmpl);
}
for (t = DECL_TEMPLATE_INSTANTIATIONS (tmpl);
t; t = TREE_CHAIN (t))
{
tree inst = TREE_VALUE (t);
if (CLASSTYPE_TEMPLATE_SPECIALIZATION (inst)
|| !COMPLETE_OR_OPEN_TYPE_P (inst))
{
spec_entry elt;
spec_entry *entry;
elt.tmpl = most_general_template (tmpl);
elt.args = CLASSTYPE_TI_ARGS (inst);
elt.spec = inst;
type_specializations->remove_elt (&elt);
elt.tmpl = tmpl;
elt.args = INNERMOST_TEMPLATE_ARGS (elt.args);
spec_entry **slot
= type_specializations->find_slot (&elt, INSERT);
entry = ggc_alloc<spec_entry> ();
*entry = elt;
*slot = entry;
}
else
error ("specialization %qT after instantiation %qT",
type, inst);
}
SET_CLASSTYPE_TEMPLATE_SPECIALIZATION (type);
DECL_SOURCE_LOCATION (TYPE_MAIN_DECL (type)) = input_location;
CLASSTYPE_TI_ARGS (type)
= INNERMOST_TEMPLATE_ARGS (CLASSTYPE_TI_ARGS (type));
}
}
else if (processing_specialization)
{
if (cxx_dialect > cxx98 && TREE_CODE (type) == ENUMERAL_TYPE
&& CLASS_TYPE_P (context) && CLASSTYPE_USE_TEMPLATE (context))
pedwarn (input_location, OPT_Wpedantic, "template specialization "
"of %qD not allowed by ISO C++", type);
else
{
error ("explicit specialization of non-template %qT", type);
return error_mark_node;
}
}
return type;
}
static inline bool
optimize_specialization_lookup_p (tree tmpl)
{
return (DECL_FUNCTION_TEMPLATE_P (tmpl)
&& DECL_CLASS_SCOPE_P (tmpl)
&& CLASS_TYPE_P (DECL_CONTEXT (tmpl))
&& !CLASSTYPE_TEMPLATE_SPECIALIZATION (DECL_CONTEXT (tmpl))
&& !DECL_MEMBER_TEMPLATE_P (tmpl)
&& !DECL_CONV_FN_P (tmpl)
&& !DECL_FRIEND_P (DECL_TEMPLATE_RESULT (tmpl)));
}
static void
verify_unstripped_args_1 (tree inner)
{
for (int i = 0; i < TREE_VEC_LENGTH (inner); ++i)
{
tree arg = TREE_VEC_ELT (inner, i);
if (TREE_CODE (arg) == TEMPLATE_DECL)
;
else if (TYPE_P (arg))
gcc_assert (strip_typedefs (arg, NULL) == arg);
else if (ARGUMENT_PACK_P (arg))
verify_unstripped_args_1 (ARGUMENT_PACK_ARGS (arg));
else if (strip_typedefs (TREE_TYPE (arg), NULL) != TREE_TYPE (arg))
;
else
gcc_assert (strip_typedefs_expr (arg, NULL) == arg);
}
}
static void
verify_unstripped_args (tree args)
{
++processing_template_decl;
if (!any_dependent_template_arguments_p (args))
verify_unstripped_args_1 (INNERMOST_TEMPLATE_ARGS (args));
--processing_template_decl;
}
static tree
retrieve_specialization (tree tmpl, tree args, hashval_t hash)
{
if (tmpl == NULL_TREE)
return NULL_TREE;
if (args == error_mark_node)
return NULL_TREE;
gcc_assert (TREE_CODE (tmpl) == TEMPLATE_DECL
|| TREE_CODE (tmpl) == FIELD_DECL);
gcc_assert (TMPL_ARGS_DEPTH (args)
== (TREE_CODE (tmpl) == TEMPLATE_DECL
? TMPL_PARMS_DEPTH (DECL_TEMPLATE_PARMS (tmpl))
: template_class_depth (DECL_CONTEXT (tmpl))));
if (flag_checking)
verify_unstripped_args (args);
if (lambda_fn_in_template_p (tmpl))
return NULL_TREE;
if (optimize_specialization_lookup_p (tmpl))
{
tree class_template = CLASSTYPE_TI_TEMPLATE (DECL_CONTEXT (tmpl));
tree class_specialization
= retrieve_specialization (class_template, args, 0);
if (!class_specialization)
return NULL_TREE;
tree fns = get_class_binding (class_specialization, DECL_NAME (tmpl));
for (ovl_iterator iter (fns); iter; ++iter)
{
tree fn = *iter;
if (DECL_TEMPLATE_INFO (fn) && DECL_TI_TEMPLATE (fn) == tmpl
&& DECL_CONTEXT (fn) == class_specialization)
return fn;
}
return NULL_TREE;
}
else
{
spec_entry *found;
spec_entry elt;
hash_table<spec_hasher> *specializations;
elt.tmpl = tmpl;
elt.args = args;
elt.spec = NULL_TREE;
if (DECL_CLASS_TEMPLATE_P (tmpl))
specializations = type_specializations;
else
specializations = decl_specializations;
if (hash == 0)
hash = spec_hasher::hash (&elt);
found = specializations->find_with_hash (&elt, hash);
if (found)
return found->spec;
}
return NULL_TREE;
}
tree
retrieve_local_specialization (tree tmpl)
{
if (local_specializations == NULL)
return NULL_TREE;
tree *slot = local_specializations->get (tmpl);
return slot ? *slot : NULL_TREE;
}
int
is_specialization_of (tree decl, tree tmpl)
{
tree t;
if (TREE_CODE (decl) == FUNCTION_DECL)
{
for (t = decl;
t != NULL_TREE;
t = DECL_TEMPLATE_INFO (t) ? DECL_TI_TEMPLATE (t) : NULL_TREE)
if (t == tmpl)
return 1;
}
else
{
gcc_assert (TREE_CODE (decl) == TYPE_DECL);
for (t = TREE_TYPE (decl);
t != NULL_TREE;
t = CLASSTYPE_USE_TEMPLATE (t)
? TREE_TYPE (CLASSTYPE_TI_TEMPLATE (t)) : NULL_TREE)
if (same_type_ignoring_top_level_qualifiers_p (t, TREE_TYPE (tmpl)))
return 1;
}
return 0;
}
bool
is_specialization_of_friend (tree decl, tree friend_decl)
{
bool need_template = true;
int template_depth;
gcc_assert (TREE_CODE (decl) == FUNCTION_DECL
|| TREE_CODE (decl) == TYPE_DECL);
if (TREE_CODE (friend_decl) == FUNCTION_DECL
&& DECL_TEMPLATE_INFO (friend_decl)
&& !DECL_USE_TEMPLATE (friend_decl))
{
friend_decl = DECL_TI_TEMPLATE (friend_decl);
need_template = false;
}
else if (TREE_CODE (friend_decl) == TEMPLATE_DECL
&& !PRIMARY_TEMPLATE_P (friend_decl))
need_template = false;
if (TREE_CODE (friend_decl) != TEMPLATE_DECL)
return false;
if (is_specialization_of (decl, friend_decl))
return true;
template_depth = template_class_depth (CP_DECL_CONTEXT (friend_decl));
if (template_depth
&& DECL_CLASS_SCOPE_P (decl)
&& is_specialization_of (TYPE_NAME (DECL_CONTEXT (decl)),
CLASSTYPE_TI_TEMPLATE (DECL_CONTEXT (friend_decl))))
{
tree context = DECL_CONTEXT (decl);
tree args = NULL_TREE;
int current_depth = 0;
while (current_depth < template_depth)
{
if (CLASSTYPE_TEMPLATE_INFO (context))
{
if (current_depth == 0)
args = TYPE_TI_ARGS (context);
else
args = add_to_template_args (TYPE_TI_ARGS (context), args);
current_depth++;
}
context = TYPE_CONTEXT (context);
}
if (TREE_CODE (decl) == FUNCTION_DECL)
{
bool is_template;
tree friend_type;
tree decl_type;
tree friend_args_type;
tree decl_args_type;
is_template = DECL_TEMPLATE_INFO (decl)
&& PRIMARY_TEMPLATE_P (DECL_TI_TEMPLATE (decl));
if (need_template ^ is_template)
return false;
else if (is_template)
{
tree friend_parms
= tsubst_template_parms (DECL_TEMPLATE_PARMS (friend_decl),
args, tf_none);
if (!comp_template_parms
(DECL_TEMPLATE_PARMS (DECL_TI_TEMPLATE (decl)),
friend_parms))
return false;
decl_type = TREE_TYPE (DECL_TI_TEMPLATE (decl));
}
else
decl_type = TREE_TYPE (decl);
friend_type = tsubst_function_type (TREE_TYPE (friend_decl), args,
tf_none, NULL_TREE);
if (friend_type == error_mark_node)
return false;
if (!same_type_p (TREE_TYPE (decl_type), TREE_TYPE (friend_type)))
return false;
friend_args_type = TYPE_ARG_TYPES (friend_type);
decl_args_type = TYPE_ARG_TYPES (decl_type);
if (DECL_NONSTATIC_MEMBER_FUNCTION_P (friend_decl))
friend_args_type = TREE_CHAIN (friend_args_type);
if (DECL_NONSTATIC_MEMBER_FUNCTION_P (decl))
decl_args_type = TREE_CHAIN (decl_args_type);
return compparms (decl_args_type, friend_args_type);
}
else
{
bool is_template;
tree decl_type = TREE_TYPE (decl);
is_template
= CLASSTYPE_TEMPLATE_INFO (decl_type)
&& PRIMARY_TEMPLATE_P (CLASSTYPE_TI_TEMPLATE (decl_type));
if (need_template ^ is_template)
return false;
else if (is_template)
{
tree friend_parms;
if (DECL_NAME (CLASSTYPE_TI_TEMPLATE (decl_type))
!= DECL_NAME (friend_decl))
return false;
friend_parms
= tsubst_template_parms (DECL_TEMPLATE_PARMS (friend_decl),
args, tf_none);
return comp_template_parms
(DECL_TEMPLATE_PARMS (CLASSTYPE_TI_TEMPLATE (decl_type)),
friend_parms);
}
else
return (DECL_NAME (decl)
== DECL_NAME (friend_decl));
}
}
return false;
}
static tree
register_specialization (tree spec, tree tmpl, tree args, bool is_friend,
hashval_t hash)
{
tree fn;
spec_entry **slot = NULL;
spec_entry elt;
gcc_assert ((TREE_CODE (tmpl) == TEMPLATE_DECL && DECL_P (spec))
|| (TREE_CODE (tmpl) == FIELD_DECL
&& TREE_CODE (spec) == NONTYPE_ARGUMENT_PACK));
if (TREE_CODE (spec) == FUNCTION_DECL
&& uses_template_parms (DECL_TI_ARGS (spec)))
return spec;
if (optimize_specialization_lookup_p (tmpl))
fn = retrieve_specialization (tmpl, args, 0);
else
{
elt.tmpl = tmpl;
elt.args = args;
elt.spec = spec;
if (hash == 0)
hash = spec_hasher::hash (&elt);
slot =
decl_specializations->find_slot_with_hash (&elt, hash, INSERT);
if (*slot)
fn = ((spec_entry *) *slot)->spec;
else
fn = NULL_TREE;
}
if (fn == spec)
return spec;
else if (fn && DECL_TEMPLATE_SPECIALIZATION (spec))
{
if (DECL_TEMPLATE_INSTANTIATION (fn))
{
if (DECL_ODR_USED (fn)
|| DECL_EXPLICIT_INSTANTIATION (fn))
{
error ("specialization of %qD after instantiation",
fn);
return error_mark_node;
}
else
{
tree clone;
DECL_INITIAL (fn) = NULL_TREE;
duplicate_decls (spec, fn, is_friend);
FOR_EACH_CLONE (clone, fn)
{
DECL_DECLARED_INLINE_P (clone)
= DECL_DECLARED_INLINE_P (fn);
DECL_SOURCE_LOCATION (clone)
= DECL_SOURCE_LOCATION (fn);
DECL_DELETED_FN (clone)
= DECL_DELETED_FN (fn);
}
check_specialization_namespace (tmpl);
return fn;
}
}
else if (DECL_TEMPLATE_SPECIALIZATION (fn))
{
tree dd = duplicate_decls (spec, fn, is_friend);
if (dd == error_mark_node)
return error_mark_node;
if (dd == NULL_TREE && DECL_INITIAL (spec))
DECL_SOURCE_LOCATION (fn) = DECL_SOURCE_LOCATION (spec);
return fn;
}
}
else if (fn)
return duplicate_decls (spec, fn, is_friend);
if (DECL_P (spec) && DECL_TEMPLATE_SPECIALIZATION (spec)
&& !check_specialization_namespace (tmpl))
DECL_CONTEXT (spec) = DECL_CONTEXT (tmpl);
if (slot != NULL )
{
spec_entry *entry = ggc_alloc<spec_entry> ();
gcc_assert (tmpl && args && spec);
*entry = elt;
*slot = entry;
if ((TREE_CODE (spec) == FUNCTION_DECL && DECL_NAMESPACE_SCOPE_P (spec)
&& PRIMARY_TEMPLATE_P (tmpl)
&& DECL_SAVED_TREE (DECL_TEMPLATE_RESULT (tmpl)) == NULL_TREE)
|| variable_template_p (tmpl))
DECL_TEMPLATE_INSTANTIATIONS (tmpl)
= tree_cons (args, spec, DECL_TEMPLATE_INSTANTIATIONS (tmpl));
}
return spec;
}
int comparing_specializations;
bool
spec_hasher::equal (spec_entry *e1, spec_entry *e2)
{
int equal;
++comparing_specializations;
equal = (e1->tmpl == e2->tmpl
&& comp_template_args (e1->args, e2->args));
if (equal && flag_concepts
&& TREE_CODE (e1->tmpl) == TEMPLATE_DECL
&& VAR_P (DECL_TEMPLATE_RESULT (e1->tmpl))
&& uses_template_parms (e1->args))
{
tree c1 = e1->spec ? get_constraints (e1->spec) : NULL_TREE;
tree c2 = e2->spec ? get_constraints (e2->spec) : NULL_TREE;
equal = equivalent_constraints (c1, c2);
}
--comparing_specializations;
return equal;
}
static hashval_t
hash_tmpl_and_args (tree tmpl, tree args)
{
hashval_t val = iterative_hash_object (DECL_UID (tmpl), 0);
return iterative_hash_template_arg (args, val);
}
hashval_t
spec_hasher::hash (spec_entry *e)
{
return hash_tmpl_and_args (e->tmpl, e->args);
}
hashval_t
iterative_hash_template_arg (tree arg, hashval_t val)
{
unsigned HOST_WIDE_INT i;
enum tree_code code;
char tclass;
if (arg == NULL_TREE)
return iterative_hash_object (arg, val);
if (!TYPE_P (arg))
STRIP_NOPS (arg);
if (TREE_CODE (arg) == ARGUMENT_PACK_SELECT)
gcc_unreachable ();
code = TREE_CODE (arg);
tclass = TREE_CODE_CLASS (code);
val = iterative_hash_object (code, val);
switch (code)
{
case ERROR_MARK:
return val;
case IDENTIFIER_NODE:
return iterative_hash_object (IDENTIFIER_HASH_VALUE (arg), val);
case TREE_VEC:
{
int i, len = TREE_VEC_LENGTH (arg);
for (i = 0; i < len; ++i)
val = iterative_hash_template_arg (TREE_VEC_ELT (arg, i), val);
return val;
}
case TYPE_PACK_EXPANSION:
case EXPR_PACK_EXPANSION:
val = iterative_hash_template_arg (PACK_EXPANSION_PATTERN (arg), val);
return iterative_hash_template_arg (PACK_EXPANSION_EXTRA_ARGS (arg), val);
case TYPE_ARGUMENT_PACK:
case NONTYPE_ARGUMENT_PACK:
return iterative_hash_template_arg (ARGUMENT_PACK_ARGS (arg), val);
case TREE_LIST:
for (; arg; arg = TREE_CHAIN (arg))
val = iterative_hash_template_arg (TREE_VALUE (arg), val);
return val;
case OVERLOAD:
for (lkp_iterator iter (arg); iter; ++iter)
val = iterative_hash_template_arg (*iter, val);
return val;
case CONSTRUCTOR:
{
tree field, value;
iterative_hash_template_arg (TREE_TYPE (arg), val);
FOR_EACH_CONSTRUCTOR_ELT (CONSTRUCTOR_ELTS (arg), i, field, value)
{
val = iterative_hash_template_arg (field, val);
val = iterative_hash_template_arg (value, val);
}
return val;
}
case PARM_DECL:
if (!DECL_ARTIFICIAL (arg))
{
val = iterative_hash_object (DECL_PARM_INDEX (arg), val);
val = iterative_hash_object (DECL_PARM_LEVEL (arg), val);
}
return iterative_hash_template_arg (TREE_TYPE (arg), val);
case TARGET_EXPR:
return iterative_hash_template_arg (TARGET_EXPR_INITIAL (arg), val);
case PTRMEM_CST:
val = iterative_hash_template_arg (PTRMEM_CST_CLASS (arg), val);
return iterative_hash_template_arg (PTRMEM_CST_MEMBER (arg), val);
case TEMPLATE_PARM_INDEX:
val = iterative_hash_template_arg
(TREE_TYPE (TEMPLATE_PARM_DECL (arg)), val);
val = iterative_hash_object (TEMPLATE_PARM_LEVEL (arg), val);
return iterative_hash_object (TEMPLATE_PARM_IDX (arg), val);
case TRAIT_EXPR:
val = iterative_hash_object (TRAIT_EXPR_KIND (arg), val);
val = iterative_hash_template_arg (TRAIT_EXPR_TYPE1 (arg), val);
return iterative_hash_template_arg (TRAIT_EXPR_TYPE2 (arg), val);
case BASELINK:
val = iterative_hash_template_arg (BINFO_TYPE (BASELINK_BINFO (arg)),
val);
return iterative_hash_template_arg (DECL_NAME (get_first_fn (arg)),
val);
case MODOP_EXPR:
val = iterative_hash_template_arg (TREE_OPERAND (arg, 0), val);
code = TREE_CODE (TREE_OPERAND (arg, 1));
val = iterative_hash_object (code, val);
return iterative_hash_template_arg (TREE_OPERAND (arg, 2), val);
case LAMBDA_EXPR:
gcc_assert (seen_error ());
return val;
case CAST_EXPR:
case IMPLICIT_CONV_EXPR:
case STATIC_CAST_EXPR:
case REINTERPRET_CAST_EXPR:
case CONST_CAST_EXPR:
case DYNAMIC_CAST_EXPR:
case NEW_EXPR:
val = iterative_hash_template_arg (TREE_TYPE (arg), val);
break;
default:
break;
}
switch (tclass)
{
case tcc_type:
if (alias_template_specialization_p (arg))
{
tree ti = TYPE_ALIAS_TEMPLATE_INFO (arg);
return hash_tmpl_and_args (TI_TEMPLATE (ti), TI_ARGS (ti));
}
if (TYPE_CANONICAL (arg))
return iterative_hash_object (TYPE_HASH (TYPE_CANONICAL (arg)),
val);
else if (TREE_CODE (arg) == DECLTYPE_TYPE)
return iterative_hash_template_arg (DECLTYPE_TYPE_EXPR (arg), val);
return val;
case tcc_declaration:
case tcc_constant:
return iterative_hash_expr (arg, val);
default:
gcc_assert (IS_EXPR_CODE_CLASS (tclass));
{
unsigned n = cp_tree_operand_length (arg);
for (i = 0; i < n; ++i)
val = iterative_hash_template_arg (TREE_OPERAND (arg, i), val);
return val;
}
}
gcc_unreachable ();
return 0;
}
bool
reregister_specialization (tree spec, tree tinfo, tree new_spec)
{
spec_entry *entry;
spec_entry elt;
elt.tmpl = most_general_template (TI_TEMPLATE (tinfo));
elt.args = TI_ARGS (tinfo);
elt.spec = NULL_TREE;
entry = decl_specializations->find (&elt);
if (entry != NULL)
{
gcc_assert (entry->spec == spec || entry->spec == new_spec);
gcc_assert (new_spec != NULL_TREE);
entry->spec = new_spec;
return 1;
}
return 0;
}
void
register_local_specialization (tree spec, tree tmpl)
{
gcc_assert (tmpl != spec);
local_specializations->put (tmpl, spec);
}
bool
explicit_class_specialization_p (tree type)
{
if (!CLASSTYPE_TEMPLATE_SPECIALIZATION (type))
return false;
return !uses_template_parms (CLASSTYPE_TI_ARGS (type));
}
static void
print_candidates_1 (tree fns, char **str, bool more = false)
{
if (TREE_CODE (fns) == TREE_LIST)
for (; fns; fns = TREE_CHAIN (fns))
print_candidates_1 (TREE_VALUE (fns), str, more || TREE_CHAIN (fns));
else
for (lkp_iterator iter (fns); iter;)
{
tree cand = *iter;
++iter;
const char *pfx = *str;
if (!pfx)
{
if (more || iter)
pfx = _("candidates are:");
else
pfx = _("candidate is:");
*str = get_spaces (pfx);
}
inform (DECL_SOURCE_LOCATION (cand), "%s %#qD", pfx, cand);
}
}
void
print_candidates (tree fns)
{
char *str = NULL;
print_candidates_1 (fns, &str);
free (str);
}
static tree
get_template_for_ordering (tree list)
{
gcc_assert (TREE_CODE (list) == TREE_LIST);
tree f = TREE_VALUE (list);
if (tree ti = DECL_TEMPLATE_INFO (f))
return TI_TEMPLATE (ti);
return f;
}
static tree
most_constrained_function (tree candidates)
{
tree champ = candidates;
for (tree c = TREE_CHAIN (champ); c; c = TREE_CHAIN (c))
{
int winner = more_constrained (get_template_for_ordering (champ),
get_template_for_ordering (c));
if (winner == -1)
champ = c; 
else if (winner == 0)
return NULL_TREE; 
}
for (tree c = candidates; c != champ; c = TREE_CHAIN (c)) {
if (!more_constrained (get_template_for_ordering (champ),
get_template_for_ordering (c)))
return NULL_TREE;
}
return champ;
}
static tree
determine_specialization (tree template_id,
tree decl,
tree* targs_out,
int need_member_template,
int template_count,
tmpl_spec_kind tsk)
{
tree fns;
tree targs;
tree explicit_targs;
tree candidates = NULL_TREE;
tree templates = NULL_TREE;
int header_count;
cp_binding_level *b;
*targs_out = NULL_TREE;
if (template_id == error_mark_node || decl == error_mark_node)
return error_mark_node;
if (!VAR_P (decl)
&& template_count && DECL_CLASS_SCOPE_P (decl)
&& template_class_depth (DECL_CONTEXT (decl)) > 0)
{
gcc_assert (errorcount);
return error_mark_node;
}
fns = TREE_OPERAND (template_id, 0);
explicit_targs = TREE_OPERAND (template_id, 1);
if (fns == error_mark_node)
return error_mark_node;
if (BASELINK_P (fns))
fns = BASELINK_FUNCTIONS (fns);
if (TREE_CODE (decl) == FUNCTION_DECL && !is_overloaded_fn (fns))
{
error ("%qD is not a function template", fns);
return error_mark_node;
}
else if (VAR_P (decl) && !variable_template_p (fns))
{
error ("%qD is not a variable template", fns);
return error_mark_node;
}
header_count = 0;
for (b = current_binding_level;
b->kind == sk_template_parms;
b = b->level_chain)
++header_count;
tree orig_fns = fns;
if (variable_template_p (fns))
{
tree parms = INNERMOST_TEMPLATE_PARMS (DECL_TEMPLATE_PARMS (fns));
targs = coerce_template_parms (parms, explicit_targs, fns,
tf_warning_or_error,
true, true);
if (targs != error_mark_node)
templates = tree_cons (targs, fns, templates);
}
else for (lkp_iterator iter (fns); iter; ++iter)
{
tree fn = *iter;
if (TREE_CODE (fn) == TEMPLATE_DECL)
{
tree decl_arg_types;
tree fn_arg_types;
tree insttype;
if (header_count && header_count != template_count + 1)
continue;
if (current_binding_level->kind == sk_template_parms
&& !current_binding_level->explicit_spec_p
&& (TREE_VEC_LENGTH (DECL_INNERMOST_TEMPLATE_PARMS (fn))
!= TREE_VEC_LENGTH (INNERMOST_TEMPLATE_PARMS
(current_template_parms))))
continue;
decl_arg_types = TYPE_ARG_TYPES (TREE_TYPE (decl));
fn_arg_types = TYPE_ARG_TYPES (TREE_TYPE (fn));
if (DECL_NONSTATIC_MEMBER_FUNCTION_P (fn))
{
if (!same_type_p (TREE_VALUE (fn_arg_types),
TREE_VALUE (decl_arg_types)))
continue;
if (type_memfn_rqual (TREE_TYPE (decl))
!= type_memfn_rqual (TREE_TYPE (fn)))
continue;
}
decl_arg_types 
= skip_artificial_parms_for (decl, decl_arg_types);
fn_arg_types 
= skip_artificial_parms_for (fn, fn_arg_types);
if (tsk == tsk_template)
{
if (compparms (fn_arg_types, decl_arg_types))
candidates = tree_cons (NULL_TREE, fn, candidates);
continue;
}
push_deferring_access_checks (dk_no_check);
targs = get_bindings (fn, decl, explicit_targs, true);
pop_deferring_access_checks ();
if (!targs)
continue;
if (uses_template_parms (targs))
continue;
if (flag_concepts && !constraints_satisfied_p (fn, targs))
continue;
insttype = tsubst (TREE_TYPE (fn), targs, tf_fndecl_type, NULL_TREE);
if (insttype == error_mark_node)
continue;
fn_arg_types
= skip_artificial_parms_for (fn, TYPE_ARG_TYPES (insttype));
if (!compparms (fn_arg_types, decl_arg_types))
continue;
templates = tree_cons (targs, fn, templates);
}
else if (need_member_template)
;
else if (TREE_CODE (fn) != FUNCTION_DECL)
;
else if (!DECL_FUNCTION_MEMBER_P (fn))
;
else if (DECL_ARTIFICIAL (fn))
;
else
{
tree decl_arg_types;
if (!DECL_TEMPLATE_INFO (fn))
continue;
if (!same_type_p (TREE_TYPE (TREE_TYPE (decl)),
TREE_TYPE (TREE_TYPE (fn))))
continue;
decl_arg_types = TYPE_ARG_TYPES (TREE_TYPE (decl));
if (DECL_STATIC_FUNCTION_P (fn)
&& DECL_NONSTATIC_MEMBER_FUNCTION_P (decl))
decl_arg_types = TREE_CHAIN (decl_arg_types);
if (!compparms (TYPE_ARG_TYPES (TREE_TYPE (fn)),
decl_arg_types))
continue;
if (DECL_NONSTATIC_MEMBER_FUNCTION_P (fn)
&& (type_memfn_rqual (TREE_TYPE (decl))
!= type_memfn_rqual (TREE_TYPE (fn))))
continue;
if (flag_concepts && !constraints_satisfied_p (fn))
continue;
candidates = tree_cons (NULL_TREE, fn, candidates);
}
}
if (templates && TREE_CHAIN (templates))
{
tree tmpl = most_specialized_instantiation (templates);
if (tmpl != error_mark_node)
{
templates = tmpl;
TREE_CHAIN (templates) = NULL_TREE;
}
}
if (flag_concepts && candidates && TREE_CHAIN (candidates))
{
if (tree cand = most_constrained_function (candidates))
{
candidates = cand;
TREE_CHAIN (cand) = NULL_TREE;
}
}
if (templates == NULL_TREE && candidates == NULL_TREE)
{
error ("template-id %qD for %q+D does not match any template "
"declaration", template_id, decl);
if (header_count && header_count != template_count + 1)
inform (input_location, "saw %d %<template<>%>, need %d for "
"specializing a member function template",
header_count, template_count + 1);
else
print_candidates (orig_fns);
return error_mark_node;
}
else if ((templates && TREE_CHAIN (templates))
|| (candidates && TREE_CHAIN (candidates))
|| (templates && candidates))
{
error ("ambiguous template specialization %qD for %q+D",
template_id, decl);
candidates = chainon (candidates, templates);
print_candidates (candidates);
return error_mark_node;
}
if (candidates)
{
tree fn = TREE_VALUE (candidates);
*targs_out = copy_node (DECL_TI_ARGS (fn));
set_constraints (decl, get_constraints (fn));
if (TREE_CODE (fn) == TEMPLATE_DECL)
return fn;
return DECL_TI_TEMPLATE (fn);
}
targs = DECL_TI_ARGS (DECL_TEMPLATE_RESULT (TREE_VALUE (templates)));
if (TMPL_ARGS_HAVE_MULTIPLE_LEVELS (targs))
{
*targs_out = copy_node (targs);
SET_TMPL_ARGS_LEVEL (*targs_out,
TMPL_ARGS_DEPTH (*targs_out),
TREE_PURPOSE (templates));
}
else
*targs_out = TREE_PURPOSE (templates);
return TREE_VALUE (templates);
}
static tree
copy_default_args_to_explicit_spec_1 (tree spec_types,
tree tmpl_types)
{
tree new_spec_types;
if (!spec_types)
return NULL_TREE;
if (spec_types == void_list_node)
return void_list_node;
new_spec_types =
copy_default_args_to_explicit_spec_1 (TREE_CHAIN (spec_types),
TREE_CHAIN (tmpl_types));
return hash_tree_cons (TREE_PURPOSE (tmpl_types),
TREE_VALUE (spec_types),
new_spec_types);
}
static void
copy_default_args_to_explicit_spec (tree decl)
{
tree tmpl;
tree spec_types;
tree tmpl_types;
tree new_spec_types;
tree old_type;
tree new_type;
tree t;
tree object_type = NULL_TREE;
tree in_charge = NULL_TREE;
tree vtt = NULL_TREE;
tmpl = DECL_TI_TEMPLATE (decl);
tmpl_types = TYPE_ARG_TYPES (TREE_TYPE (DECL_TEMPLATE_RESULT (tmpl)));
for (t = tmpl_types; t; t = TREE_CHAIN (t))
if (TREE_PURPOSE (t))
break;
if (!t)
return;
old_type = TREE_TYPE (decl);
spec_types = TYPE_ARG_TYPES (old_type);
if (DECL_NONSTATIC_MEMBER_FUNCTION_P (decl))
{
object_type = TREE_TYPE (TREE_VALUE (spec_types));
spec_types = TREE_CHAIN (spec_types);
tmpl_types = TREE_CHAIN (tmpl_types);
if (DECL_HAS_IN_CHARGE_PARM_P (decl))
{
in_charge = spec_types;
spec_types = TREE_CHAIN (spec_types);
}
if (DECL_HAS_VTT_PARM_P (decl))
{
vtt = spec_types;
spec_types = TREE_CHAIN (spec_types);
}
}
new_spec_types =
copy_default_args_to_explicit_spec_1 (spec_types, tmpl_types);
if (object_type)
{
if (vtt)
new_spec_types = hash_tree_cons (TREE_PURPOSE (vtt),
TREE_VALUE (vtt),
new_spec_types);
if (in_charge)
new_spec_types = hash_tree_cons (TREE_PURPOSE (in_charge),
TREE_VALUE (in_charge),
new_spec_types);
new_type = build_method_type_directly (object_type,
TREE_TYPE (old_type),
new_spec_types);
}
else
new_type = build_function_type (TREE_TYPE (old_type),
new_spec_types);
new_type = cp_build_type_attribute_variant (new_type,
TYPE_ATTRIBUTES (old_type));
new_type = build_exception_variant (new_type,
TYPE_RAISES_EXCEPTIONS (old_type));
if (TYPE_HAS_LATE_RETURN_TYPE (old_type))
TYPE_HAS_LATE_RETURN_TYPE (new_type) = 1;
TREE_TYPE (decl) = new_type;
}
int
num_template_headers_for_class (tree ctype)
{
int num_templates = 0;
while (ctype && CLASS_TYPE_P (ctype))
{
if (!CLASSTYPE_TEMPLATE_INFO (ctype))
break;
if (explicit_class_specialization_p (ctype))
break;
if (PRIMARY_TEMPLATE_P (CLASSTYPE_TI_TEMPLATE (ctype)))
++num_templates;
ctype = TYPE_CONTEXT (ctype);
}
return num_templates;
}
void
check_template_variable (tree decl)
{
tree ctx = CP_DECL_CONTEXT (decl);
int wanted = num_template_headers_for_class (ctx);
if (DECL_LANG_SPECIFIC (decl) && DECL_TEMPLATE_INFO (decl)
&& PRIMARY_TEMPLATE_P (DECL_TI_TEMPLATE (decl)))
{
if (cxx_dialect < cxx14)
pedwarn (DECL_SOURCE_LOCATION (decl), 0,
"variable templates only available with "
"-std=c++14 or -std=gnu++14");
++wanted;
}
if (template_header_count > wanted)
{
bool warned = pedwarn (DECL_SOURCE_LOCATION (decl), 0,
"too many template headers for %qD "
"(should be %d)",
decl, wanted);
if (warned && CLASS_TYPE_P (ctx)
&& CLASSTYPE_TEMPLATE_SPECIALIZATION (ctx))
inform (DECL_SOURCE_LOCATION (decl),
"members of an explicitly specialized class are defined "
"without a template header");
}
}
void
check_unqualified_spec_or_inst (tree t, location_t loc)
{
tree tmpl = most_general_template (t);
if (DECL_NAMESPACE_SCOPE_P (tmpl)
&& !is_nested_namespace (current_namespace,
CP_DECL_CONTEXT (tmpl), true))
{
if (processing_specialization)
permerror (loc, "explicit specialization of %qD outside its "
"namespace must use a nested-name-specifier", tmpl);
else if (processing_explicit_instantiation
&& cxx_dialect >= cxx11)
pedwarn (loc, OPT_Wpedantic, "explicit instantiation of %qD "
"outside its namespace must use a nested-name-"
"specifier", tmpl);
}
}
static void
warn_spec_missing_attributes (tree tmpl, tree spec, tree attrlist)
{
if (DECL_FUNCTION_TEMPLATE_P (tmpl))
tmpl = DECL_TEMPLATE_RESULT (tmpl);
if (TREE_CODE (tmpl) != FUNCTION_DECL)
return;
if (TREE_DEPRECATED (tmpl)
|| TREE_DEPRECATED (spec))
return;
tree tmpl_type = TREE_TYPE (tmpl);
tree spec_type = TREE_TYPE (spec);
if (TREE_DEPRECATED (tmpl_type)
|| TREE_DEPRECATED (spec_type)
|| TREE_DEPRECATED (TREE_TYPE (tmpl_type))
|| TREE_DEPRECATED (TREE_TYPE (spec_type)))
return;
tree tmpl_attrs[] = { DECL_ATTRIBUTES (tmpl), TYPE_ATTRIBUTES (tmpl_type) };
tree spec_attrs[] = { DECL_ATTRIBUTES (spec), TYPE_ATTRIBUTES (spec_type) };
if (!spec_attrs[0])
spec_attrs[0] = attrlist;
else if (!spec_attrs[1])
spec_attrs[1] = attrlist;
if (!tmpl_attrs[0] && !tmpl_attrs[1])
return;
const char* const whitelist[] = {
"error", "warning"
};
for (unsigned i = 0; i != 2; ++i)
for (unsigned j = 0; j != sizeof whitelist / sizeof *whitelist; ++j)
if (lookup_attribute (whitelist[j], tmpl_attrs[i])
|| lookup_attribute (whitelist[j], spec_attrs[i]))
return;
const char* const blacklist[] = {
"alloc_align", "alloc_size", "assume_aligned", "format",
"format_arg", "malloc", "nonnull"
};
unsigned nattrs = 0;
pretty_printer str;
for (unsigned i = 0; i != sizeof blacklist / sizeof *blacklist; ++i)
{
for (unsigned j = 0; j != 2; ++j)
{
if (!lookup_attribute (blacklist[i], tmpl_attrs[j]))
continue;
for (unsigned k = 0; k != 1 + !!spec_attrs[1]; ++k)
{
if (lookup_attribute (blacklist[i], spec_attrs[k]))
break;
if (nattrs)
pp_string (&str, ", ");
pp_begin_quote (&str, pp_show_color (global_dc->printer));
pp_string (&str, blacklist[i]);
pp_end_quote (&str, pp_show_color (global_dc->printer));
++nattrs;
}
}
}
if (!nattrs)
return;
if (warning_at (DECL_SOURCE_LOCATION (spec), OPT_Wmissing_attributes,
"explicit specialization %q#D may be missing attributes",
spec))
inform (DECL_SOURCE_LOCATION (tmpl),
nattrs > 1
? G_("missing primary template attributes %s")
: G_("missing primary template attribute %s"),
pp_formatted_text (&str));
}
tree
check_explicit_specialization (tree declarator,
tree decl,
int template_count,
int flags,
tree attrlist)
{
int have_def = flags & 2;
int is_friend = flags & 4;
bool is_concept = flags & 8;
int specialization = 0;
int explicit_instantiation = 0;
int member_specialization = 0;
tree ctype = DECL_CLASS_CONTEXT (decl);
tree dname = DECL_NAME (decl);
tmpl_spec_kind tsk;
if (is_friend)
{
if (!processing_specialization)
tsk = tsk_none;
else
tsk = tsk_excessive_parms;
}
else
tsk = current_tmpl_spec_kind (template_count);
switch (tsk)
{
case tsk_none:
if (processing_specialization && !VAR_P (decl))
{
specialization = 1;
SET_DECL_TEMPLATE_SPECIALIZATION (decl);
}
else if (TREE_CODE (declarator) == TEMPLATE_ID_EXPR)
{
if (is_friend)
specialization = 1;
else
{
error ("template-id %qD in declaration of primary template",
declarator);
return decl;
}
}
break;
case tsk_invalid_member_spec:
return error_mark_node;
case tsk_invalid_expl_inst:
error ("template parameter list used in explicit instantiation");
case tsk_expl_inst:
if (have_def)
error ("definition provided for explicit instantiation");
explicit_instantiation = 1;
break;
case tsk_excessive_parms:
case tsk_insufficient_parms:
if (tsk == tsk_excessive_parms)
error ("too many template parameter lists in declaration of %qD",
decl);
else if (template_header_count)
error("too few template parameter lists in declaration of %qD", decl);
else
error("explicit specialization of %qD must be introduced by "
"%<template <>%>", decl);
case tsk_expl_spec:
if (is_concept)
error ("explicit specialization declared %<concept%>");
if (VAR_P (decl) && TREE_CODE (declarator) != TEMPLATE_ID_EXPR)
break;
SET_DECL_TEMPLATE_SPECIALIZATION (decl);
if (ctype)
member_specialization = 1;
else
specialization = 1;
break;
case tsk_template:
if (TREE_CODE (declarator) == TEMPLATE_ID_EXPR)
{
if (!uses_template_parms (declarator))
error ("template-id %qD in declaration of primary template",
declarator);
else if (variable_template_p (TREE_OPERAND (declarator, 0)))
{
SET_DECL_TEMPLATE_SPECIALIZATION (decl);
specialization = 1;
goto ok;
}
else if (cxx_dialect < cxx14)
error ("non-type partial specialization %qD "
"is not allowed", declarator);
else
error ("non-class, non-variable partial specialization %qD "
"is not allowed", declarator);
return decl;
ok:;
}
if (ctype && CLASSTYPE_TEMPLATE_INSTANTIATION (ctype))
specialization = 1;
break;
default:
gcc_unreachable ();
}
if ((specialization || member_specialization)
&& (TREE_CODE (TREE_TYPE (decl)) == FUNCTION_TYPE
|| TREE_CODE (TREE_TYPE (decl)) == METHOD_TYPE))
{
tree t = TYPE_ARG_TYPES (TREE_TYPE (decl));
for (; t; t = TREE_CHAIN (t))
if (TREE_PURPOSE (t))
{
permerror (input_location, 
"default argument specified in explicit specialization");
break;
}
}
if (specialization || member_specialization || explicit_instantiation)
{
tree tmpl = NULL_TREE;
tree targs = NULL_TREE;
bool was_template_id = (TREE_CODE (declarator) == TEMPLATE_ID_EXPR);
if (!was_template_id)
{
tree fns;
gcc_assert (identifier_p (declarator));
if (ctype)
fns = dname;
else
{
gcc_assert (DECL_NAMESPACE_SCOPE_P (decl));
fns = lookup_qualified_name (CP_DECL_CONTEXT (decl), dname,
false, true);
if (fns == error_mark_node)
fns = lookup_qualified_name (CP_DECL_CONTEXT (decl), dname,
false, true,
true);
if (fns == error_mark_node || !is_overloaded_fn (fns))
{
error ("%qD is not a template function", dname);
fns = error_mark_node;
}
}
declarator = lookup_template_function (fns, NULL_TREE);
}
if (declarator == error_mark_node)
return error_mark_node;
if (ctype != NULL_TREE && TYPE_BEING_DEFINED (ctype))
{
if (!explicit_instantiation)
return error_mark_node;
else
{
;
}
return decl;
}
else if (ctype != NULL_TREE
&& (identifier_p (TREE_OPERAND (declarator, 0))))
{
if (VAR_P (decl))
return decl;
tree name = TREE_OPERAND (declarator, 0);
if (constructor_name_p (name, ctype))
{
if (DECL_CONSTRUCTOR_P (decl)
? !TYPE_HAS_USER_CONSTRUCTOR (ctype)
: !CLASSTYPE_DESTRUCTOR (ctype))
{
error ("specialization of implicitly-declared special member function");
return error_mark_node;
}
name = DECL_NAME (decl);
}
tree fns = get_class_binding (ctype, IDENTIFIER_CONV_OP_P (name)
? conv_op_identifier : name);
if (fns == NULL_TREE)
{
error ("no member function %qD declared in %qT", name, ctype);
return error_mark_node;
}
else
TREE_OPERAND (declarator, 0) = fns;
}
tmpl = determine_specialization (declarator, decl,
&targs,
member_specialization,
template_count,
tsk);
if (!tmpl || tmpl == error_mark_node)
return error_mark_node;
else
{
if (TREE_CODE (decl) == FUNCTION_DECL
&& DECL_HIDDEN_FRIEND_P (tmpl))
{
if (pedwarn (DECL_SOURCE_LOCATION (decl), 0,
"friend declaration %qD is not visible to "
"explicit specialization", tmpl))
inform (DECL_SOURCE_LOCATION (tmpl),
"friend declaration here");
}
else if (!ctype && !is_friend
&& CP_DECL_CONTEXT (decl) == current_namespace)
check_unqualified_spec_or_inst (tmpl, DECL_SOURCE_LOCATION (decl));
tree gen_tmpl = most_general_template (tmpl);
if (explicit_instantiation)
{
int arg_depth = TMPL_ARGS_DEPTH (targs);
int parm_depth = TMPL_PARMS_DEPTH (DECL_TEMPLATE_PARMS (tmpl));
if (arg_depth > parm_depth)
{
int i;
tree new_targs;
new_targs = make_tree_vec (parm_depth);
for (i = arg_depth - parm_depth; i < arg_depth; ++i)
TREE_VEC_ELT (new_targs, i - (arg_depth - parm_depth))
= TREE_VEC_ELT (targs, i);
targs = new_targs;
}
return instantiate_template (tmpl, targs, tf_error);
}
if (DECL_FUNCTION_TEMPLATE_P (tmpl)
&& DECL_STATIC_FUNCTION_P (tmpl)
&& DECL_NONSTATIC_MEMBER_FUNCTION_P (decl))
revert_static_member_fn (decl);
if (tsk == tsk_template && !was_template_id)
{
tree result = DECL_TEMPLATE_RESULT (tmpl);
SET_DECL_TEMPLATE_SPECIALIZATION (tmpl);
DECL_INITIAL (result) = NULL_TREE;
if (have_def)
{
tree parm;
DECL_SOURCE_LOCATION (tmpl) = DECL_SOURCE_LOCATION (decl);
DECL_SOURCE_LOCATION (result)
= DECL_SOURCE_LOCATION (decl);
DECL_ARGUMENTS (result) = DECL_ARGUMENTS (decl);
for (parm = DECL_ARGUMENTS (result); parm;
parm = DECL_CHAIN (parm))
DECL_CONTEXT (parm) = result;
}
return register_specialization (tmpl, gen_tmpl, targs,
is_friend, 0);
}
DECL_TEMPLATE_INFO (decl) = build_template_info (tmpl, targs);
if (was_template_id)
TINFO_USED_TEMPLATE_ID (DECL_TEMPLATE_INFO (decl)) = true;
if (DECL_FUNCTION_TEMPLATE_P (tmpl))
copy_default_args_to_explicit_spec (decl);
TREE_PRIVATE (decl) = TREE_PRIVATE (gen_tmpl);
TREE_PROTECTED (decl) = TREE_PROTECTED (gen_tmpl);
if (tsk == tsk_expl_spec && DECL_FUNCTION_TEMPLATE_P (gen_tmpl))
{
tree tmpl_func = DECL_TEMPLATE_RESULT (gen_tmpl);
gcc_assert (TREE_CODE (tmpl_func) == FUNCTION_DECL);
if (DECL_DECLARED_CONCEPT_P (tmpl_func))
{
error ("explicit specialization of function concept %qD",
gen_tmpl);
return error_mark_node;
}
TREE_PUBLIC (decl) = TREE_PUBLIC (tmpl_func);
if (! TREE_PUBLIC (decl))
{
DECL_INTERFACE_KNOWN (decl) = 1;
DECL_NOT_REALLY_EXTERN (decl) = 1;
}
DECL_THIS_STATIC (decl) = DECL_THIS_STATIC (tmpl_func);
if (DECL_VISIBILITY_SPECIFIED (tmpl_func))
{
DECL_VISIBILITY_SPECIFIED (decl) = 1;
DECL_VISIBILITY (decl) = DECL_VISIBILITY (tmpl_func);
}
}
if (DECL_NAMESPACE_SCOPE_P (decl))
DECL_CONTEXT (decl) = DECL_CONTEXT (tmpl);
if (is_friend && !have_def)
SET_DECL_IMPLICIT_INSTANTIATION (decl);
else if (TREE_CODE (decl) == FUNCTION_DECL)
DECL_COMDAT (decl) = (TREE_PUBLIC (decl)
&& DECL_DECLARED_INLINE_P (decl));
else if (VAR_P (decl))
DECL_COMDAT (decl) = false;
if (!processing_template_decl)
{
warn_spec_missing_attributes (gen_tmpl, decl, attrlist);
decl = register_specialization (decl, gen_tmpl, targs,
is_friend, 0);
}
gcc_assert (decl == error_mark_node
|| variable_template_p (tmpl)
|| !(DECL_CONSTRUCTOR_P (decl)
|| DECL_DESTRUCTOR_P (decl))
|| DECL_CLONED_FUNCTION_P (DECL_CHAIN (decl)));
}
}
return decl;
}
int
comp_template_parms (const_tree parms1, const_tree parms2)
{
const_tree p1;
const_tree p2;
if (parms1 == parms2)
return 1;
for (p1 = parms1, p2 = parms2;
p1 != NULL_TREE && p2 != NULL_TREE;
p1 = TREE_CHAIN (p1), p2 = TREE_CHAIN (p2))
{
tree t1 = TREE_VALUE (p1);
tree t2 = TREE_VALUE (p2);
int i;
gcc_assert (TREE_CODE (t1) == TREE_VEC);
gcc_assert (TREE_CODE (t2) == TREE_VEC);
if (TREE_VEC_LENGTH (t1) != TREE_VEC_LENGTH (t2))
return 0;
for (i = 0; i < TREE_VEC_LENGTH (t2); ++i)
{
tree parm1 = TREE_VALUE (TREE_VEC_ELT (t1, i));
tree parm2 = TREE_VALUE (TREE_VEC_ELT (t2, i));
if (error_operand_p (parm1) || error_operand_p (parm2))
return 1;
if (TREE_CODE (parm1) != TREE_CODE (parm2))
return 0;
if (TREE_CODE (parm1) == TEMPLATE_TYPE_PARM
&& (TEMPLATE_TYPE_PARAMETER_PACK (parm1)
== TEMPLATE_TYPE_PARAMETER_PACK (parm2)))
continue;
else if (!same_type_p (TREE_TYPE (parm1), TREE_TYPE (parm2)))
return 0;
}
}
if ((p1 != NULL_TREE) != (p2 != NULL_TREE))
return 0;
return 1;
}
bool 
template_parameter_pack_p (const_tree parm)
{
if (TREE_CODE (parm) == PARM_DECL)
return (DECL_TEMPLATE_PARM_P (parm) 
&& TEMPLATE_PARM_PARAMETER_PACK (DECL_INITIAL (parm)));
if (TREE_CODE (parm) == TEMPLATE_PARM_INDEX)
return TEMPLATE_PARM_PARAMETER_PACK (parm);
if (TREE_CODE (parm) == TYPE_DECL || TREE_CODE (parm) == TEMPLATE_DECL)
parm = TREE_TYPE (parm);
return ((TREE_CODE (parm) == TEMPLATE_TYPE_PARM
|| TREE_CODE (parm) == TEMPLATE_TEMPLATE_PARM)
&& TEMPLATE_TYPE_PARAMETER_PACK (parm));
}
bool
function_parameter_pack_p (const_tree t)
{
if (t && TREE_CODE (t) == PARM_DECL)
return DECL_PACK_P (t);
return false;
}
tree
get_function_template_decl (const_tree primary_func_tmpl_inst)
{
if (! primary_func_tmpl_inst
|| TREE_CODE (primary_func_tmpl_inst) != FUNCTION_DECL
|| ! primary_template_specialization_p (primary_func_tmpl_inst))
return NULL;
return DECL_TEMPLATE_RESULT (DECL_TI_TEMPLATE (primary_func_tmpl_inst));
}
bool
function_parameter_expanded_from_pack_p (tree param_decl, tree pack)
{
if (DECL_ARTIFICIAL (param_decl)
|| !function_parameter_pack_p (pack))
return false;
return DECL_PARM_INDEX (pack) == DECL_PARM_INDEX (param_decl);
}
static bool 
template_args_variadic_p (tree args)
{
int nargs;
tree last_parm;
if (args == NULL_TREE)
return false;
args = INNERMOST_TEMPLATE_ARGS (args);
nargs = TREE_VEC_LENGTH (args);
if (nargs == 0)
return false;
last_parm = TREE_VEC_ELT (args, nargs - 1);
return ARGUMENT_PACK_P (last_parm);
}
static tree
make_ith_pack_parameter_name (tree name, int i)
{
#define NUMBUF_LEN 128
char numbuf[NUMBUF_LEN];
char* newname;
int newname_len;
if (name == NULL_TREE)
return name;
snprintf (numbuf, NUMBUF_LEN, "%i", i);
newname_len = IDENTIFIER_LENGTH (name)
+ strlen (numbuf) + 2;
newname = (char*)alloca (newname_len);
snprintf (newname, newname_len,
"%s#%i", IDENTIFIER_POINTER (name), i);
return get_identifier (newname);
}
bool
primary_template_specialization_p (const_tree t)
{
if (!t)
return false;
if (TREE_CODE (t) == FUNCTION_DECL || VAR_P (t))
return (DECL_LANG_SPECIFIC (t)
&& DECL_USE_TEMPLATE (t)
&& DECL_TEMPLATE_INFO (t)
&& PRIMARY_TEMPLATE_P (DECL_TI_TEMPLATE (t)));
else if (CLASS_TYPE_P (t) && !TYPE_DECL_ALIAS_P (TYPE_NAME (t)))
return (CLASSTYPE_TEMPLATE_INFO (t)
&& CLASSTYPE_USE_TEMPLATE (t)
&& PRIMARY_TEMPLATE_P (CLASSTYPE_TI_TEMPLATE (t)));
else if (alias_template_specialization_p (t))
return true;
return false;
}
bool
template_template_parameter_p (const_tree parm)
{
return DECL_TEMPLATE_TEMPLATE_PARM_P (parm);
}
bool
template_type_parameter_p (const_tree parm)
{
return (parm
&& (TREE_CODE (parm) == TYPE_DECL
|| TREE_CODE (parm) == TEMPLATE_DECL)
&& DECL_TEMPLATE_PARM_P (parm));
}
tree
get_primary_template_innermost_parameters (const_tree t)
{
tree parms = NULL, template_info = NULL;
if ((template_info = get_template_info (t))
&& primary_template_specialization_p (t))
parms = INNERMOST_TEMPLATE_PARMS
(DECL_TEMPLATE_PARMS (TI_TEMPLATE (template_info)));
return parms;
}
tree
get_template_parms_at_level (tree parms, int level)
{
tree p;
if (!parms
|| TREE_CODE (parms) != TREE_LIST
|| level > TMPL_PARMS_DEPTH (parms))
return NULL_TREE;
for (p = parms; p; p = TREE_CHAIN (p))
if (TMPL_PARMS_DEPTH (p) == level)
return p;
return NULL_TREE;
}
tree
get_template_innermost_arguments (const_tree t)
{
tree args = NULL, template_info = NULL;
if ((template_info = get_template_info (t))
&& TI_ARGS (template_info))
args = INNERMOST_TEMPLATE_ARGS (TI_ARGS (template_info));
return args;
}
tree
get_template_argument_pack_elems (const_tree t)
{
if (TREE_CODE (t) != TYPE_ARGUMENT_PACK
&& TREE_CODE (t) != NONTYPE_ARGUMENT_PACK)
return NULL;
return ARGUMENT_PACK_ARGS (t);
}
static tree
argument_pack_select_arg (tree t)
{
tree args = ARGUMENT_PACK_ARGS (ARGUMENT_PACK_SELECT_FROM_PACK (t));
tree arg = TREE_VEC_ELT (args, ARGUMENT_PACK_SELECT_INDEX (t));
if (PACK_EXPANSION_P (arg))
{
gcc_assert (!PACK_EXPANSION_EXTRA_ARGS (arg));
arg = PACK_EXPANSION_PATTERN (arg);
}
return arg;
}
bool
builtin_pack_fn_p (tree fn)
{
if (!fn
|| TREE_CODE (fn) != FUNCTION_DECL
|| !DECL_IS_BUILTIN (fn))
return false;
if (id_equal (DECL_NAME (fn), "__integer_pack"))
return true;
return false;
}
static bool
builtin_pack_call_p (tree call)
{
if (TREE_CODE (call) != CALL_EXPR)
return false;
return builtin_pack_fn_p (CALL_EXPR_FN (call));
}
static tree
expand_integer_pack (tree call, tree args, tsubst_flags_t complain,
tree in_decl)
{
tree ohi = CALL_EXPR_ARG (call, 0);
tree hi = tsubst_copy_and_build (ohi, args, complain, in_decl,
false, true);
if (value_dependent_expression_p (hi))
{
if (hi != ohi)
{
call = copy_node (call);
CALL_EXPR_ARG (call, 0) = hi;
}
tree ex = make_pack_expansion (call, complain);
tree vec = make_tree_vec (1);
TREE_VEC_ELT (vec, 0) = ex;
return vec;
}
else
{
hi = cxx_constant_value (hi);
int len = valid_constant_size_p (hi) ? tree_to_shwi (hi) : -1;
int max = ((INT_MAX - sizeof (tree_vec)) / sizeof (tree)) + 1;
if (len < 0 || len > max)
{
if ((complain & tf_error)
&& hi != error_mark_node)
error ("argument to __integer_pack must be between 0 and %d", max);
return error_mark_node;
}
tree vec = make_tree_vec (len);
for (int i = 0; i < len; ++i)
TREE_VEC_ELT (vec, i) = size_int (i);
return vec;
}
}
static tree
expand_builtin_pack_call (tree call, tree args, tsubst_flags_t complain,
tree in_decl)
{
if (!builtin_pack_call_p (call))
return NULL_TREE;
tree fn = CALL_EXPR_FN (call);
if (id_equal (DECL_NAME (fn), "__integer_pack"))
return expand_integer_pack (call, args, complain, in_decl);
return NULL_TREE;
}
struct find_parameter_pack_data 
{
tree* parameter_packs;
hash_set<tree> *visited;
bool type_pack_expansion_p;
};
static tree
find_parameter_packs_r (tree *tp, int *walk_subtrees, void* data)
{
tree t = *tp;
struct find_parameter_pack_data* ppd = 
(struct find_parameter_pack_data*)data;
bool parameter_pack_p = false;
if (TYPE_ALIAS_P (t))
{
if (tree tinfo = TYPE_ALIAS_TEMPLATE_INFO (t))
cp_walk_tree (&TI_ARGS (tinfo),
&find_parameter_packs_r,
ppd, ppd->visited);
*walk_subtrees = 0;
return NULL_TREE;
}
switch (TREE_CODE (t))
{
case TEMPLATE_PARM_INDEX:
if (TEMPLATE_PARM_PARAMETER_PACK (t))
parameter_pack_p = true;
break;
case TEMPLATE_TYPE_PARM:
t = TYPE_MAIN_VARIANT (t);
case TEMPLATE_TEMPLATE_PARM:
if (ppd->type_pack_expansion_p && is_auto (t))
TEMPLATE_TYPE_PARAMETER_PACK (t) = true;
if (TEMPLATE_TYPE_PARAMETER_PACK (t))
parameter_pack_p = true;
break;
case FIELD_DECL:
case PARM_DECL:
if (DECL_PACK_P (t))
{
*walk_subtrees = 0;
parameter_pack_p = true;
}
break;
case VAR_DECL:
if (DECL_PACK_P (t))
{
*walk_subtrees = 0;
parameter_pack_p = true;
}
else if (variable_template_specialization_p (t))
{
cp_walk_tree (&DECL_TI_ARGS (t),
find_parameter_packs_r,
ppd, ppd->visited);
*walk_subtrees = 0;
}
break;
case CALL_EXPR:
if (builtin_pack_call_p (t))
parameter_pack_p = true;
break;
case BASES:
parameter_pack_p = true;
break;
default:
break;
}
if (parameter_pack_p)
{
*ppd->parameter_packs = tree_cons (NULL_TREE, t, *ppd->parameter_packs);
}
if (TYPE_P (t))
cp_walk_tree (&TYPE_CONTEXT (t), 
&find_parameter_packs_r, ppd, ppd->visited);
switch (TREE_CODE (t)) 
{
case TEMPLATE_PARM_INDEX:
return NULL_TREE;
case BOUND_TEMPLATE_TEMPLATE_PARM:
cp_walk_tree (&TREE_TYPE (TYPE_TI_TEMPLATE (t)), 
&find_parameter_packs_r, ppd, ppd->visited);
cp_walk_tree (&TYPE_TI_ARGS (t), &find_parameter_packs_r, ppd, 
ppd->visited);
*walk_subtrees = 0;
return NULL_TREE;
case TEMPLATE_TYPE_PARM:
case TEMPLATE_TEMPLATE_PARM:
return NULL_TREE;
case PARM_DECL:
return NULL_TREE;
case DECL_EXPR:
if (is_capture_proxy (DECL_EXPR_DECL (t)))
*walk_subtrees = 0;
return NULL_TREE;
case RECORD_TYPE:
if (TYPE_PTRMEMFUNC_P (t))
return NULL_TREE;
case UNION_TYPE:
case ENUMERAL_TYPE:
if (TYPE_TEMPLATE_INFO (t))
cp_walk_tree (&TYPE_TI_ARGS (t),
&find_parameter_packs_r, ppd, ppd->visited);
*walk_subtrees = 0;
return NULL_TREE;
case TEMPLATE_DECL:
if (!DECL_TEMPLATE_TEMPLATE_PARM_P (t))
return NULL_TREE;
gcc_fallthrough();
case CONSTRUCTOR:
cp_walk_tree (&TREE_TYPE (t),
&find_parameter_packs_r, ppd, ppd->visited);
return NULL_TREE;
case TYPENAME_TYPE:
cp_walk_tree (&TYPENAME_TYPE_FULLNAME (t), &find_parameter_packs_r,
ppd, ppd->visited);
*walk_subtrees = 0;
return NULL_TREE;
case TYPE_PACK_EXPANSION:
case EXPR_PACK_EXPANSION:
*walk_subtrees = 0;
return NULL_TREE;
case INTEGER_TYPE:
cp_walk_tree (&TYPE_MAX_VALUE (t), &find_parameter_packs_r, 
ppd, ppd->visited);
*walk_subtrees = 0;
return NULL_TREE;
case IDENTIFIER_NODE:
cp_walk_tree (&TREE_TYPE (t), &find_parameter_packs_r, ppd, 
ppd->visited);
*walk_subtrees = 0;
return NULL_TREE;
case LAMBDA_EXPR:
{
for (tree cap = LAMBDA_EXPR_CAPTURE_LIST (t);
cap; cap = TREE_CHAIN (cap))
cp_walk_tree (&TREE_VALUE (cap), &find_parameter_packs_r, ppd,
ppd->visited);
tree fn = lambda_function (t);
cp_walk_tree (&TREE_TYPE (fn), &find_parameter_packs_r, ppd,
ppd->visited);
cp_walk_tree (&DECL_SAVED_TREE (fn), &find_parameter_packs_r, ppd,
ppd->visited);
*walk_subtrees = 0;
return NULL_TREE;
}
case DECLTYPE_TYPE:
{
bool type_pack_expansion_p = ppd->type_pack_expansion_p;
ppd->type_pack_expansion_p = false;
cp_walk_tree (&DECLTYPE_TYPE_EXPR (t), &find_parameter_packs_r,
ppd, ppd->visited);
ppd->type_pack_expansion_p = type_pack_expansion_p;
*walk_subtrees = 0;
return NULL_TREE;
}
case IF_STMT:
cp_walk_tree (&IF_COND (t), &find_parameter_packs_r,
ppd, ppd->visited);
cp_walk_tree (&THEN_CLAUSE (t), &find_parameter_packs_r,
ppd, ppd->visited);
cp_walk_tree (&ELSE_CLAUSE (t), &find_parameter_packs_r,
ppd, ppd->visited);
*walk_subtrees = 0;
return NULL_TREE;
default:
return NULL_TREE;
}
return NULL_TREE;
}
bool
uses_parameter_packs (tree t)
{
tree parameter_packs = NULL_TREE;
struct find_parameter_pack_data ppd;
ppd.parameter_packs = &parameter_packs;
ppd.visited = new hash_set<tree>;
ppd.type_pack_expansion_p = false;
cp_walk_tree (&t, &find_parameter_packs_r, &ppd, ppd.visited);
delete ppd.visited;
return parameter_packs != NULL_TREE;
}
tree 
make_pack_expansion (tree arg, tsubst_flags_t complain)
{
tree result;
tree parameter_packs = NULL_TREE;
bool for_types = false;
struct find_parameter_pack_data ppd;
if (!arg || arg == error_mark_node)
return arg;
if (TREE_CODE (arg) == TREE_LIST && TREE_PURPOSE (arg))
{
tree purpose;
tree value;
tree parameter_packs = NULL_TREE;
ppd.visited = new hash_set<tree>;
ppd.parameter_packs = &parameter_packs;
ppd.type_pack_expansion_p = true;
gcc_assert (TYPE_P (TREE_PURPOSE (arg)));
cp_walk_tree (&TREE_PURPOSE (arg), &find_parameter_packs_r, 
&ppd, ppd.visited);
if (parameter_packs == NULL_TREE)
{
if (complain & tf_error)
error ("base initializer expansion %qT contains no parameter packs",
arg);
delete ppd.visited;
return error_mark_node;
}
if (TREE_VALUE (arg) != void_type_node)
{
for (value = TREE_VALUE (arg); value; value = TREE_CHAIN (value))
{
cp_walk_tree (&TREE_VALUE (value), &find_parameter_packs_r, 
&ppd, ppd.visited);
}
}
delete ppd.visited;
purpose = cxx_make_type (TYPE_PACK_EXPANSION);
SET_PACK_EXPANSION_PATTERN (purpose, TREE_PURPOSE (arg));
PACK_EXPANSION_PARAMETER_PACKS (purpose) = parameter_packs;
PACK_EXPANSION_LOCAL_P (purpose) = at_function_scope_p ();
SET_TYPE_STRUCTURAL_EQUALITY (purpose);
return tree_cons (purpose, TREE_VALUE (arg), NULL_TREE);
}
if (TYPE_P (arg) || TREE_CODE (arg) == TEMPLATE_DECL)
for_types = true;
result = for_types
? cxx_make_type (TYPE_PACK_EXPANSION)
: make_node (EXPR_PACK_EXPANSION);
SET_PACK_EXPANSION_PATTERN (result, arg);
if (TREE_CODE (result) == EXPR_PACK_EXPANSION)
{
TREE_TYPE (result) = TREE_TYPE (arg);
TREE_CONSTANT (result) = TREE_CONSTANT (arg);
mark_exp_read (arg);
}
else
SET_TYPE_STRUCTURAL_EQUALITY (result);
ppd.parameter_packs = &parameter_packs;
ppd.visited = new hash_set<tree>;
ppd.type_pack_expansion_p = TYPE_P (arg);
cp_walk_tree (&arg, &find_parameter_packs_r, &ppd, ppd.visited);
delete ppd.visited;
if (parameter_packs == NULL_TREE)
{
if (complain & tf_error)
{
if (TYPE_P (arg))
error ("expansion pattern %qT contains no argument packs", arg);
else
error ("expansion pattern %qE contains no argument packs", arg);
}
return error_mark_node;
}
PACK_EXPANSION_PARAMETER_PACKS (result) = parameter_packs;
PACK_EXPANSION_LOCAL_P (result) = at_function_scope_p ();
return result;
}
bool 
check_for_bare_parameter_packs (tree t, location_t loc )
{
tree parameter_packs = NULL_TREE;
struct find_parameter_pack_data ppd;
if (!processing_template_decl || !t || t == error_mark_node)
return false;
if (current_class_type && LAMBDA_TYPE_P (current_class_type)
&& CLASSTYPE_TEMPLATE_INFO (current_class_type))
return false;
if (TREE_CODE (t) == TYPE_DECL)
t = TREE_TYPE (t);
ppd.parameter_packs = &parameter_packs;
ppd.visited = new hash_set<tree>;
ppd.type_pack_expansion_p = false;
cp_walk_tree (&t, &find_parameter_packs_r, &ppd, ppd.visited);
delete ppd.visited;
if (parameter_packs) 
{
if (loc == UNKNOWN_LOCATION)
loc = EXPR_LOC_OR_LOC (t, input_location);
error_at (loc, "parameter packs not expanded with %<...%>:");
while (parameter_packs)
{
tree pack = TREE_VALUE (parameter_packs);
tree name = NULL_TREE;
if (TREE_CODE (pack) == TEMPLATE_TYPE_PARM
|| TREE_CODE (pack) == TEMPLATE_TEMPLATE_PARM)
name = TYPE_NAME (pack);
else if (TREE_CODE (pack) == TEMPLATE_PARM_INDEX)
name = DECL_NAME (TEMPLATE_PARM_DECL (pack));
else if (TREE_CODE (pack) == CALL_EXPR)
name = DECL_NAME (CALL_EXPR_FN (pack));
else
name = DECL_NAME (pack);
if (name)
inform (loc, "        %qD", name);
else
inform (loc, "        <anonymous>");
parameter_packs = TREE_CHAIN (parameter_packs);
}
return true;
}
return false;
}
tree
expand_template_argument_pack (tree args)
{
if (args == error_mark_node)
return error_mark_node;
tree result_args = NULL_TREE;
int in_arg, out_arg = 0, nargs = args ? TREE_VEC_LENGTH (args) : 0;
int num_result_args = -1;
int non_default_args_count = -1;
for (in_arg = 0; in_arg < nargs; ++in_arg)
{
tree arg = TREE_VEC_ELT (args, in_arg);
if (arg == NULL_TREE)
return args;
if (ARGUMENT_PACK_P (arg))
{
int num_packed = TREE_VEC_LENGTH (ARGUMENT_PACK_ARGS (arg));
if (num_result_args < 0)
num_result_args = in_arg + num_packed;
else
num_result_args += num_packed;
}
else
{
if (num_result_args >= 0)
num_result_args++;
}
}
if (num_result_args < 0)
return args;
result_args = make_tree_vec (num_result_args);
if (NON_DEFAULT_TEMPLATE_ARGS_COUNT (args))
non_default_args_count =
GET_NON_DEFAULT_TEMPLATE_ARGS_COUNT (args);
for (in_arg = 0; in_arg < nargs; ++in_arg)
{
tree arg = TREE_VEC_ELT (args, in_arg);
if (ARGUMENT_PACK_P (arg))
{
tree packed = ARGUMENT_PACK_ARGS (arg);
int i, num_packed = TREE_VEC_LENGTH (packed);
for (i = 0; i < num_packed; ++i, ++out_arg)
TREE_VEC_ELT (result_args, out_arg) = TREE_VEC_ELT(packed, i);
if (non_default_args_count > 0)
non_default_args_count += num_packed - 1;
}
else
{
TREE_VEC_ELT (result_args, out_arg) = arg;
++out_arg;
}
}
if (non_default_args_count >= 0)
SET_NON_DEFAULT_TEMPLATE_ARGS_COUNT (result_args, non_default_args_count);
return result_args;
}
bool
check_template_shadow (tree decl)
{
tree olddecl;
if (!current_template_parms)
return true;
decl = OVL_FIRST (decl);
olddecl = innermost_non_namespace_value (DECL_NAME (decl));
if (!olddecl)
return true;
if (!DECL_P (olddecl) || !DECL_TEMPLATE_PARM_P (olddecl))
return true;
if (decl == olddecl
|| (DECL_TEMPLATE_PARM_P (decl)
&& TEMPLATE_PARMS_FOR_INLINE (current_template_parms)))
return true;
if (DECL_SELF_REFERENCE_P (decl))
return false;
if (DECL_TEMPLATE_PARM_P (decl))
error ("declaration of template parameter %q+D shadows "
"template parameter", decl);
else
error ("declaration of %q+#D shadows template parameter", decl);
inform (DECL_SOURCE_LOCATION (olddecl),
"template parameter %qD declared here", olddecl);
return false;
}
static tree
build_template_parm_index (int index,
int level,
int orig_level,
tree decl,
tree type)
{
tree t = make_node (TEMPLATE_PARM_INDEX);
TEMPLATE_PARM_IDX (t) = index;
TEMPLATE_PARM_LEVEL (t) = level;
TEMPLATE_PARM_ORIG_LEVEL (t) = orig_level;
TEMPLATE_PARM_DECL (t) = decl;
TREE_TYPE (t) = type;
TREE_CONSTANT (t) = TREE_CONSTANT (decl);
TREE_READONLY (t) = TREE_READONLY (decl);
return t;
}
static tree
canonical_type_parameter (tree type)
{
tree list;
int idx = TEMPLATE_TYPE_IDX (type);
if (!canonical_template_parms)
vec_alloc (canonical_template_parms, idx + 1);
if (canonical_template_parms->length () <= (unsigned) idx)
vec_safe_grow_cleared (canonical_template_parms, idx + 1);
list = (*canonical_template_parms)[idx];
while (list && !comptypes (type, TREE_VALUE (list), COMPARE_STRUCTURAL))
list = TREE_CHAIN (list);
if (list)
return TREE_VALUE (list);
else
{
(*canonical_template_parms)[idx]
= tree_cons (NULL_TREE, type, (*canonical_template_parms)[idx]);
return type;
}
}
static tree
reduce_template_parm_level (tree index, tree type, int levels, tree args,
tsubst_flags_t complain)
{
if (TEMPLATE_PARM_DESCENDANTS (index) == NULL_TREE
|| (TEMPLATE_PARM_LEVEL (TEMPLATE_PARM_DESCENDANTS (index))
!= TEMPLATE_PARM_LEVEL (index) - levels)
|| !same_type_p (type, TREE_TYPE (TEMPLATE_PARM_DESCENDANTS (index))))
{
tree orig_decl = TEMPLATE_PARM_DECL (index);
tree decl, t;
decl = build_decl (DECL_SOURCE_LOCATION (orig_decl),
TREE_CODE (orig_decl), DECL_NAME (orig_decl), type);
TREE_CONSTANT (decl) = TREE_CONSTANT (orig_decl);
TREE_READONLY (decl) = TREE_READONLY (orig_decl);
DECL_ARTIFICIAL (decl) = 1;
SET_DECL_TEMPLATE_PARM_P (decl);
t = build_template_parm_index (TEMPLATE_PARM_IDX (index),
TEMPLATE_PARM_LEVEL (index) - levels,
TEMPLATE_PARM_ORIG_LEVEL (index),
decl, type);
TEMPLATE_PARM_DESCENDANTS (index) = t;
TEMPLATE_PARM_PARAMETER_PACK (t) 
= TEMPLATE_PARM_PARAMETER_PACK (index);
if (TREE_CODE (decl) == TEMPLATE_DECL)
{
DECL_TEMPLATE_RESULT (decl)
= build_decl (DECL_SOURCE_LOCATION (decl),
TYPE_DECL, DECL_NAME (decl), type);
DECL_ARTIFICIAL (DECL_TEMPLATE_RESULT (decl)) = true;
DECL_TEMPLATE_PARMS (decl) = tsubst_template_parms
(DECL_TEMPLATE_PARMS (orig_decl), args, complain);
}
}
return TEMPLATE_PARM_DESCENDANTS (index);
}
tree
process_template_parm (tree list, location_t parm_loc, tree parm,
bool is_non_type, bool is_parameter_pack)
{
tree decl = 0;
int idx = 0;
gcc_assert (TREE_CODE (parm) == TREE_LIST);
tree defval = TREE_PURPOSE (parm);
tree constr = TREE_TYPE (parm);
if (list)
{
tree p = tree_last (list);
if (p && TREE_VALUE (p) != error_mark_node)
{
p = TREE_VALUE (p);
if (TREE_CODE (p) == TYPE_DECL || TREE_CODE (p) == TEMPLATE_DECL)
idx = TEMPLATE_TYPE_IDX (TREE_TYPE (p));
else
idx = TEMPLATE_PARM_IDX (DECL_INITIAL (p));
}
++idx;
}
if (is_non_type)
{
parm = TREE_VALUE (parm);
SET_DECL_TEMPLATE_PARM_P (parm);
if (TREE_TYPE (parm) != error_mark_node)
{
TREE_TYPE (parm) = TYPE_MAIN_VARIANT (TREE_TYPE (parm));
if (invalid_nontype_parm_type_p (TREE_TYPE (parm), 1))
TREE_TYPE (parm) = error_mark_node;
else if (uses_parameter_packs (TREE_TYPE (parm))
&& !is_parameter_pack
&& processing_template_parmlist == 1)
{
check_for_bare_parameter_packs (TREE_TYPE (parm));
is_parameter_pack = true;
}
}
TREE_CONSTANT (parm) = 1;
TREE_READONLY (parm) = 1;
decl = build_decl (parm_loc,
CONST_DECL, DECL_NAME (parm), TREE_TYPE (parm));
TREE_CONSTANT (decl) = 1;
TREE_READONLY (decl) = 1;
DECL_INITIAL (parm) = DECL_INITIAL (decl)
= build_template_parm_index (idx, processing_template_decl,
processing_template_decl,
decl, TREE_TYPE (parm));
TEMPLATE_PARM_PARAMETER_PACK (DECL_INITIAL (parm)) 
= is_parameter_pack;
}
else
{
tree t;
parm = TREE_VALUE (TREE_VALUE (parm));
if (parm && TREE_CODE (parm) == TEMPLATE_DECL)
{
t = cxx_make_type (TEMPLATE_TEMPLATE_PARM);
TREE_TYPE (parm) = t;
TREE_TYPE (DECL_TEMPLATE_RESULT (parm)) = t;
decl = parm;
}
else
{
t = cxx_make_type (TEMPLATE_TYPE_PARM);
decl = build_decl (parm_loc,
TYPE_DECL, parm, t);
}
TYPE_NAME (t) = decl;
TYPE_STUB_DECL (t) = decl;
parm = decl;
TEMPLATE_TYPE_PARM_INDEX (t)
= build_template_parm_index (idx, processing_template_decl,
processing_template_decl,
decl, TREE_TYPE (parm));
TEMPLATE_TYPE_PARAMETER_PACK (t) = is_parameter_pack;
TYPE_CANONICAL (t) = canonical_type_parameter (t);
}
DECL_ARTIFICIAL (decl) = 1;
SET_DECL_TEMPLATE_PARM_P (decl);
tree reqs = finish_shorthand_constraint (parm, constr);
pushdecl (decl);
if (defval && TREE_CODE (defval) == OVERLOAD)
lookup_keep (defval, true);
parm = build_tree_list (defval, parm);
TEMPLATE_PARM_CONSTRAINTS (parm) = reqs;
return chainon (list, parm);
}
tree
end_template_parm_list (tree parms)
{
int nparms;
tree parm, next;
tree saved_parmlist = make_tree_vec (list_length (parms));
current_template_parms = TREE_CHAIN (current_template_parms);
current_template_parms
= tree_cons (size_int (processing_template_decl),
saved_parmlist, current_template_parms);
for (parm = parms, nparms = 0; parm; parm = next, nparms++)
{
next = TREE_CHAIN (parm);
TREE_VEC_ELT (saved_parmlist, nparms) = parm;
TREE_CHAIN (parm) = NULL_TREE;
}
--processing_template_parmlist;
return saved_parmlist;
}
void
end_template_parm_list ()
{
--processing_template_parmlist;
}
void
end_template_decl (void)
{
reset_specialization ();
if (! processing_template_decl)
return;
finish_scope ();
--processing_template_decl;
current_template_parms = TREE_CHAIN (current_template_parms);
}
tree
template_parm_to_arg (tree t)
{
if (t == NULL_TREE
|| TREE_CODE (t) != TREE_LIST)
return t;
if (error_operand_p (TREE_VALUE (t)))
return error_mark_node;
t = TREE_VALUE (t);
if (TREE_CODE (t) == TYPE_DECL
|| TREE_CODE (t) == TEMPLATE_DECL)
{
t = TREE_TYPE (t);
if (TEMPLATE_TYPE_PARAMETER_PACK (t))
{
tree vec = make_tree_vec (1);
if (CHECKING_P)
SET_NON_DEFAULT_TEMPLATE_ARGS_COUNT (vec, TREE_VEC_LENGTH (vec));
TREE_VEC_ELT (vec, 0) = make_pack_expansion (t);
t = cxx_make_type (TYPE_ARGUMENT_PACK);
SET_ARGUMENT_PACK_ARGS (t, vec);
}
}
else
{
t = DECL_INITIAL (t);
if (TEMPLATE_PARM_PARAMETER_PACK (t))
{
tree vec = make_tree_vec (1);
if (CHECKING_P)
SET_NON_DEFAULT_TEMPLATE_ARGS_COUNT (vec, TREE_VEC_LENGTH (vec));
t = convert_from_reference (t);
TREE_VEC_ELT (vec, 0) = make_pack_expansion (t);
t  = make_node (NONTYPE_ARGUMENT_PACK);
SET_ARGUMENT_PACK_ARGS (t, vec);
}
else
t = convert_from_reference (t);
}
return t;
}
static tree
template_parms_level_to_args (tree parms)
{
tree a = copy_node (parms);
TREE_TYPE (a) = NULL_TREE;
for (int i = TREE_VEC_LENGTH (a) - 1; i >= 0; --i)
TREE_VEC_ELT (a, i) = template_parm_to_arg (TREE_VEC_ELT (a, i));
if (CHECKING_P)
SET_NON_DEFAULT_TEMPLATE_ARGS_COUNT (a, TREE_VEC_LENGTH (a));
return a;
}
static tree
template_parms_to_args (tree parms)
{
tree header;
tree args = NULL_TREE;
int length = TMPL_PARMS_DEPTH (parms);
int l = length;
if (length > 1)
args = make_tree_vec (length);
for (header = parms; header; header = TREE_CHAIN (header))
{
tree a = template_parms_level_to_args (TREE_VALUE (header));
if (length > 1)
TREE_VEC_ELT (args, --l) = a;
else
args = a;
}
return args;
}
static tree
current_template_args (void)
{
return template_parms_to_args (current_template_parms);
}
tree
maybe_update_decl_type (tree orig_type, tree scope)
{
tree type = orig_type;
if (type == NULL_TREE)
return type;
if (TREE_CODE (orig_type) == TYPE_DECL)
type = TREE_TYPE (type);
if (scope && TYPE_P (scope) && dependent_type_p (scope)
&& dependent_type_p (type)
&& TREE_CODE (type) != TEMPLATE_TYPE_PARM)
{
tree args = current_template_args ();
tree auto_node = type_uses_auto (type);
tree pushed;
if (auto_node)
{
tree auto_vec = make_tree_vec (1);
TREE_VEC_ELT (auto_vec, 0) = auto_node;
args = add_to_template_args (args, auto_vec);
}
pushed = push_scope (scope);
type = tsubst (type, args, tf_warning_or_error, NULL_TREE);
if (pushed)
pop_scope (scope);
}
if (type == error_mark_node)
return orig_type;
if (TREE_CODE (orig_type) == TYPE_DECL)
{
if (same_type_p (type, TREE_TYPE (orig_type)))
type = orig_type;
else
type = TYPE_NAME (type);
}
return type;
}
tree
build_template_decl (tree decl, tree parms, bool member_template_p)
{
tree tmpl = build_lang_decl (TEMPLATE_DECL, DECL_NAME (decl), NULL_TREE);
SET_DECL_LANGUAGE (tmpl, DECL_LANGUAGE (decl));
DECL_TEMPLATE_PARMS (tmpl) = parms;
DECL_CONTEXT (tmpl) = DECL_CONTEXT (decl);
DECL_SOURCE_LOCATION (tmpl) = DECL_SOURCE_LOCATION (decl);
DECL_MEMBER_TEMPLATE_P (tmpl) = member_template_p;
return tmpl;
}
struct template_parm_data
{
int level;
int current_arg;
int* parms;
int* arg_uses_template_parms;
};
static int
mark_template_parm (tree t, void* data)
{
int level;
int idx;
struct template_parm_data* tpd = (struct template_parm_data*) data;
template_parm_level_and_index (t, &level, &idx);
if (level == tpd->level)
{
tpd->parms[idx] = 1;
tpd->arg_uses_template_parms[tpd->current_arg] = 1;
}
if (cxx_dialect >= cxx17
&& TREE_CODE (t) == TEMPLATE_PARM_INDEX)
for_each_template_parm (TREE_TYPE (t),
&mark_template_parm,
data,
NULL,
false);
return 0;
}
static tree
process_partial_specialization (tree decl)
{
tree type = TREE_TYPE (decl);
tree tinfo = get_template_info (decl);
tree maintmpl = TI_TEMPLATE (tinfo);
tree specargs = TI_ARGS (tinfo);
tree inner_args = INNERMOST_TEMPLATE_ARGS (specargs);
tree main_inner_parms = DECL_INNERMOST_TEMPLATE_PARMS (maintmpl);
tree inner_parms;
tree inst;
int nargs = TREE_VEC_LENGTH (inner_args);
int ntparms;
int  i;
bool did_error_intro = false;
struct template_parm_data tpd;
struct template_parm_data tpd2;
gcc_assert (current_template_parms);
if (flag_concepts && variable_concept_p (maintmpl))
{
error ("specialization of variable concept %q#D", maintmpl);
return error_mark_node;
}
inner_parms = INNERMOST_TEMPLATE_PARMS (current_template_parms);
ntparms = TREE_VEC_LENGTH (inner_parms);
tpd.level = TMPL_PARMS_DEPTH (current_template_parms);
tpd.parms = XALLOCAVEC (int, ntparms);
memset (tpd.parms, 0, sizeof (int) * ntparms);
tpd.arg_uses_template_parms = XALLOCAVEC (int, nargs);
memset (tpd.arg_uses_template_parms, 0, sizeof (int) * nargs);
for (i = 0; i < nargs; ++i)
{
tpd.current_arg = i;
for_each_template_parm (TREE_VEC_ELT (inner_args, i),
&mark_template_parm,
&tpd,
NULL,
false);
}
for (i = 0; i < ntparms; ++i)
if (tpd.parms[i] == 0)
{
if (!did_error_intro)
{
error ("template parameters not deducible in "
"partial specialization:");
did_error_intro = true;
}
inform (input_location, "        %qD",
TREE_VALUE (TREE_VEC_ELT (inner_parms, i)));
}
if (did_error_intro)
return error_mark_node;
tree main_args
= TI_ARGS (get_template_info (DECL_TEMPLATE_RESULT (maintmpl)));
if (comp_template_args (inner_args, INNERMOST_TEMPLATE_ARGS (main_args))
&& (!flag_concepts
|| !strictly_subsumes (current_template_constraints (),
get_constraints (maintmpl))))
{
if (!flag_concepts)
error ("partial specialization %q+D does not specialize "
"any template arguments; to define the primary template, "
"remove the template argument list", decl);
else
error ("partial specialization %q+D does not specialize any "
"template arguments and is not more constrained than "
"the primary template; to define the primary template, "
"remove the template argument list", decl);
inform (DECL_SOURCE_LOCATION (maintmpl), "primary template here");
}
if (nargs < DECL_NTPARMS (maintmpl))
{
error ("partial specialization is not more specialized than the "
"primary template because it replaces multiple parameters "
"with a pack expansion");
inform (DECL_SOURCE_LOCATION (maintmpl), "primary template here");
return decl;
}
else if (tpd.level == 1
&& TMPL_ARGS_DEPTH (specargs) == 1
&& !get_partial_spec_bindings (maintmpl, maintmpl, specargs))
{
if (permerror (input_location, "partial specialization %qD is not "
"more specialized than", decl))
inform (DECL_SOURCE_LOCATION (maintmpl), "primary template %qD",
maintmpl);
}
gcc_assert (nargs == DECL_NTPARMS (maintmpl));
tpd2.parms = 0;
for (i = 0; i < nargs; ++i)
{
tree parm = TREE_VALUE (TREE_VEC_ELT (main_inner_parms, i));
tree arg = TREE_VEC_ELT (inner_args, i);
tree packed_args = NULL_TREE;
int j, len = 1;
if (ARGUMENT_PACK_P (arg))
{
packed_args = ARGUMENT_PACK_ARGS (arg);
len = TREE_VEC_LENGTH (packed_args);
}
for (j = 0; j < len; j++)
{
if (packed_args)
arg = TREE_VEC_ELT (packed_args, j);
if (PACK_EXPANSION_P (arg))
{
if ((packed_args && j < len - 1)
|| (!packed_args && i < nargs - 1))
{
if (TREE_CODE (arg) == EXPR_PACK_EXPANSION)
error ("parameter pack argument %qE must be at the "
"end of the template argument list", arg);
else
error ("parameter pack argument %qT must be at the "
"end of the template argument list", arg);
}
}
if (TREE_CODE (arg) == EXPR_PACK_EXPANSION)
arg = PACK_EXPANSION_PATTERN (arg);
if (
!TYPE_P (arg)
&& TREE_CODE (arg) != TEMPLATE_DECL
&& TREE_CODE (arg) != TEMPLATE_PARM_INDEX
&& !(REFERENCE_REF_P (arg)
&& TREE_CODE (TREE_OPERAND (arg, 0)) == TEMPLATE_PARM_INDEX))
{
if ((!packed_args && tpd.arg_uses_template_parms[i])
|| (packed_args && uses_template_parms (arg)))
error ("template argument %qE involves template parameter(s)",
arg);
else 
{
tree type = TREE_TYPE (parm);
if (!tpd2.parms)
{
tpd2.arg_uses_template_parms = XALLOCAVEC (int, nargs);
tpd2.parms = XALLOCAVEC (int, nargs);
tpd2.level = 
TMPL_PARMS_DEPTH (DECL_TEMPLATE_PARMS (maintmpl));
}
tpd2.current_arg = i;
tpd2.arg_uses_template_parms[i] = 0;
memset (tpd2.parms, 0, sizeof (int) * nargs);
for_each_template_parm (type,
&mark_template_parm,
&tpd2,
NULL,
false);
if (tpd2.arg_uses_template_parms [i])
{
int j;
int count = 0;
for (j = 0; j < nargs; ++j)
if (tpd2.parms[j] != 0
&& tpd.arg_uses_template_parms [j])
++count;
if (count != 0)
error_n (input_location, count,
"type %qT of template argument %qE depends "
"on a template parameter",
"type %qT of template argument %qE depends "
"on template parameters",
type,
arg);
}
}
}
}
}
if (TREE_CODE (decl) == TYPE_DECL)
gcc_assert (!COMPLETE_TYPE_P (type));
tree tmpl = build_template_decl (decl, current_template_parms,
DECL_MEMBER_TEMPLATE_P (maintmpl));
TREE_TYPE (tmpl) = type;
DECL_TEMPLATE_RESULT (tmpl) = decl;
SET_DECL_TEMPLATE_SPECIALIZATION (tmpl);
DECL_TEMPLATE_INFO (tmpl) = build_template_info (maintmpl, specargs);
DECL_PRIMARY_TEMPLATE (tmpl) = maintmpl;
for (i = 0; i < ntparms; ++i)
{
tree parm = TREE_VALUE (TREE_VEC_ELT (inner_parms, i));
if (TREE_CODE (parm) == TEMPLATE_DECL)
DECL_CONTEXT (parm) = tmpl;
}
if (VAR_P (decl))
decl = register_specialization (decl, maintmpl, specargs, false, 0);
else
associate_classtype_constraints (type);
DECL_TEMPLATE_SPECIALIZATIONS (maintmpl)
= tree_cons (specargs, tmpl,
DECL_TEMPLATE_SPECIALIZATIONS (maintmpl));
TREE_TYPE (DECL_TEMPLATE_SPECIALIZATIONS (maintmpl)) = type;
for (inst = DECL_TEMPLATE_INSTANTIATIONS (maintmpl); inst;
inst = TREE_CHAIN (inst))
{
tree instance = TREE_VALUE (inst);
if (TYPE_P (instance)
? (COMPLETE_TYPE_P (instance)
&& CLASSTYPE_IMPLICIT_INSTANTIATION (instance))
: DECL_TEMPLATE_INSTANTIATION (instance))
{
tree spec = most_specialized_partial_spec (instance, tf_none);
tree inst_decl = (DECL_P (instance)
? instance : TYPE_NAME (instance));
if (!spec)
;
else if (spec == error_mark_node)
permerror (input_location,
"declaration of %qD ambiguates earlier template "
"instantiation for %qD", decl, inst_decl);
else if (TREE_VALUE (spec) == tmpl)
permerror (input_location,
"partial specialization of %qD after instantiation "
"of %qD", decl, inst_decl);
}
}
return decl;
}
static tree
get_template_parm_index (tree parm)
{
if (TREE_CODE (parm) == PARM_DECL
|| TREE_CODE (parm) == CONST_DECL)
parm = DECL_INITIAL (parm);
else if (TREE_CODE (parm) == TYPE_DECL
|| TREE_CODE (parm) == TEMPLATE_DECL)
parm = TREE_TYPE (parm);
if (TREE_CODE (parm) == TEMPLATE_TYPE_PARM
|| TREE_CODE (parm) == BOUND_TEMPLATE_TEMPLATE_PARM
|| TREE_CODE (parm) == TEMPLATE_TEMPLATE_PARM)
parm = TEMPLATE_TYPE_PARM_INDEX (parm);
gcc_assert (TREE_CODE (parm) == TEMPLATE_PARM_INDEX);
return parm;
}
static void
fixed_parameter_pack_p_1 (tree parm, struct find_parameter_pack_data *ppd)
{
if (TREE_CODE (parm) == TYPE_DECL || parm == error_mark_node)
return;
else if (TREE_CODE (parm) == PARM_DECL)
{
cp_walk_tree (&TREE_TYPE (parm), &find_parameter_packs_r,
ppd, ppd->visited);
return;
}
gcc_assert (TREE_CODE (parm) == TEMPLATE_DECL);
tree vec = INNERMOST_TEMPLATE_PARMS (DECL_TEMPLATE_PARMS (parm));
for (int i = 0; i < TREE_VEC_LENGTH (vec); ++i)
fixed_parameter_pack_p_1 (TREE_VALUE (TREE_VEC_ELT (vec, i)), ppd);
}
tree
fixed_parameter_pack_p (tree parm)
{
if (TEMPLATE_PARM_ORIG_LEVEL (get_template_parm_index (parm)) < 2)
return NULL_TREE;
if (!template_parameter_pack_p (parm))
return NULL_TREE;
if (TREE_CODE (parm) == TYPE_DECL)
return NULL_TREE;
tree parameter_packs = NULL_TREE;
struct find_parameter_pack_data ppd;
ppd.parameter_packs = &parameter_packs;
ppd.visited = new hash_set<tree>;
ppd.type_pack_expansion_p = false;
fixed_parameter_pack_p_1 (parm, &ppd);
delete ppd.visited;
return parameter_packs;
}
bool
check_default_tmpl_args (tree decl, tree parms, bool is_primary,
bool is_partial, int is_friend_decl)
{
const char *msg;
int last_level_to_check;
tree parm_level;
bool no_errors = true;
if (TREE_CODE (CP_DECL_CONTEXT (decl)) == FUNCTION_DECL
|| (TREE_CODE (decl) == FUNCTION_DECL && DECL_LOCAL_FUNCTION_P (decl)))
return true;
if ((TREE_CODE (decl) == TYPE_DECL
&& TREE_TYPE (decl)
&& LAMBDA_TYPE_P (TREE_TYPE (decl)))
|| (TREE_CODE (decl) == FUNCTION_DECL
&& LAMBDA_FUNCTION_P (decl)))
return true;
if (current_class_type
&& !TYPE_BEING_DEFINED (current_class_type)
&& DECL_LANG_SPECIFIC (decl)
&& DECL_DECLARES_FUNCTION_P (decl)
&& (DECL_FUNCTION_MEMBER_P (decl)
? same_type_p (DECL_CONTEXT (decl), current_class_type)
: DECL_FRIEND_CONTEXT (decl)
? same_type_p (DECL_FRIEND_CONTEXT (decl), current_class_type)
: false)
&& (!DECL_FUNCTION_MEMBER_P (decl)
|| DECL_INITIALIZED_IN_CLASS_P (decl)))
return true;
if (is_primary
&& ((cxx_dialect == cxx98) || TREE_CODE (decl) != FUNCTION_DECL))
{
for (parm_level = parms; parm_level; parm_level = TREE_CHAIN (parm_level))
{
tree inner_parms = TREE_VALUE (parm_level);
int ntparms = TREE_VEC_LENGTH (inner_parms);
int seen_def_arg_p = 0;
int i;
for (i = 0; i < ntparms; ++i)
{
tree parm = TREE_VEC_ELT (inner_parms, i);
if (parm == error_mark_node)
continue;
if (TREE_PURPOSE (parm))
seen_def_arg_p = 1;
else if (seen_def_arg_p
&& !template_parameter_pack_p (TREE_VALUE (parm)))
{
error ("no default argument for %qD", TREE_VALUE (parm));
TREE_PURPOSE (parm) = error_mark_node;
no_errors = false;
}
else if (!is_partial
&& !is_friend_decl
&& parm_level == parms
&& TREE_CODE (decl) == TYPE_DECL
&& i < ntparms - 1
&& template_parameter_pack_p (TREE_VALUE (parm))
&& !fixed_parameter_pack_p (TREE_VALUE (parm)))
{
error ("parameter pack %q+D must be at the end of the"
" template parameter list", TREE_VALUE (parm));
TREE_VALUE (TREE_VEC_ELT (inner_parms, i)) 
= error_mark_node;
no_errors = false;
}
}
}
}
if (((cxx_dialect == cxx98) && TREE_CODE (decl) != TYPE_DECL)
|| is_partial 
|| !is_primary
|| is_friend_decl)
;
else
parms = TREE_CHAIN (parms);
if (is_friend_decl == 2)
msg = G_("default template arguments may not be used in function template "
"friend re-declaration");
else if (is_friend_decl)
msg = G_("default template arguments may not be used in template "
"friend declarations");
else if (TREE_CODE (decl) == FUNCTION_DECL && (cxx_dialect == cxx98))
msg = G_("default template arguments may not be used in function templates "
"without -std=c++11 or -std=gnu++11");
else if (is_partial)
msg = G_("default template arguments may not be used in "
"partial specializations");
else if (current_class_type && CLASSTYPE_IS_TEMPLATE (current_class_type))
msg = G_("default argument for template parameter for class enclosing %qD");
else
return true;
if (current_class_type && TYPE_BEING_DEFINED (current_class_type))
last_level_to_check = template_class_depth (current_class_type) + 1;
else
last_level_to_check = 0;
for (parm_level = parms;
parm_level && TMPL_PARMS_DEPTH (parm_level) >= last_level_to_check;
parm_level = TREE_CHAIN (parm_level))
{
tree inner_parms = TREE_VALUE (parm_level);
int i;
int ntparms;
ntparms = TREE_VEC_LENGTH (inner_parms);
for (i = 0; i < ntparms; ++i)
{
if (TREE_VEC_ELT (inner_parms, i) == error_mark_node)
continue;
if (TREE_PURPOSE (TREE_VEC_ELT (inner_parms, i)))
{
if (msg)
{
no_errors = false;
if (is_friend_decl == 2)
return no_errors;
error (msg, decl);
msg = 0;
}
TREE_PURPOSE (TREE_VEC_ELT (inner_parms, i)) = NULL_TREE;
}
}
if (msg)
msg = G_("default argument for template parameter for class "
"enclosing %qD");
}
return no_errors;
}
static int
template_parm_this_level_p (tree t, void* data)
{
int this_level = *(int *)data;
int level;
if (TREE_CODE (t) == TEMPLATE_PARM_INDEX)
level = TEMPLATE_PARM_LEVEL (t);
else
level = TEMPLATE_TYPE_LEVEL (t);
return level == this_level;
}
static int
template_parm_outer_level (tree t, void *data)
{
int this_level = *(int *)data;
int level;
if (TREE_CODE (t) == TEMPLATE_PARM_INDEX)
level = TEMPLATE_PARM_LEVEL (t);
else
level = TEMPLATE_TYPE_LEVEL (t);
return level <= this_level;
}
tree
push_template_decl_real (tree decl, bool is_friend)
{
tree tmpl;
tree args;
tree info;
tree ctx;
bool is_primary;
bool is_partial;
int new_template_p = 0;
bool member_template_p = false;
if (decl == error_mark_node || !current_template_parms)
return error_mark_node;
is_partial = ((DECL_IMPLICIT_TYPEDEF_P (decl)
&& TREE_CODE (TREE_TYPE (decl)) != ENUMERAL_TYPE
&& CLASSTYPE_TEMPLATE_SPECIALIZATION (TREE_TYPE (decl)))
|| (VAR_P (decl)
&& DECL_LANG_SPECIFIC (decl)
&& DECL_TEMPLATE_SPECIALIZATION (decl)
&& TINFO_USED_TEMPLATE_ID (DECL_TEMPLATE_INFO (decl))));
if (TREE_CODE (decl) == FUNCTION_DECL && DECL_FRIEND_P (decl))
is_friend = true;
if (is_friend)
ctx = CP_DECL_CONTEXT (decl);
else if (CP_DECL_CONTEXT (decl)
&& TREE_CODE (CP_DECL_CONTEXT (decl)) != NAMESPACE_DECL)
ctx = CP_DECL_CONTEXT (decl);
else
ctx = current_scope ();
if (ctx && TREE_CODE (ctx) == NAMESPACE_DECL)
ctx = NULL_TREE;
if (!DECL_CONTEXT (decl))
DECL_CONTEXT (decl) = FROB_CONTEXT (current_namespace);
if (is_friend && ctx
&& uses_template_parms_level (ctx, processing_template_decl))
is_primary = false;
else if (TREE_CODE (decl) == TYPE_DECL
&& LAMBDA_TYPE_P (TREE_TYPE (decl)))
is_primary = false;
else
is_primary = template_parm_scope_p ();
if (is_primary)
{
warning (OPT_Wtemplates, "template %qD declared", decl);
if (DECL_CLASS_SCOPE_P (decl))
member_template_p = true;
if (TREE_CODE (decl) == TYPE_DECL
&& anon_aggrname_p (DECL_NAME (decl)))
{
error ("template class without a name");
return error_mark_node;
}
else if (TREE_CODE (decl) == FUNCTION_DECL)
{
if (member_template_p)
{
if (DECL_OVERRIDE_P (decl) || DECL_FINAL_P (decl))
error ("member template %qD may not have virt-specifiers", decl);
}
if (DECL_DESTRUCTOR_P (decl))
{
error ("destructor %qD declared as member template", decl);
return error_mark_node;
}
if (IDENTIFIER_NEWDEL_OP_P (DECL_NAME (decl))
&& (!prototype_p (TREE_TYPE (decl))
|| TYPE_ARG_TYPES (TREE_TYPE (decl)) == void_list_node
|| !TREE_CHAIN (TYPE_ARG_TYPES (TREE_TYPE (decl)))
|| (TREE_CHAIN (TYPE_ARG_TYPES (TREE_TYPE (decl)))
== void_list_node)))
{
error ("invalid template declaration of %qD", decl);
return error_mark_node;
}
}
else if (DECL_IMPLICIT_TYPEDEF_P (decl)
&& CLASS_TYPE_P (TREE_TYPE (decl)))
{
tree parms = INNERMOST_TEMPLATE_PARMS (current_template_parms);
for (int i = 0; i < TREE_VEC_LENGTH (parms); ++i)
{
tree t = TREE_VALUE (TREE_VEC_ELT (parms, i));
if (TREE_CODE (t) == TYPE_DECL)
t = TREE_TYPE (t);
if (TREE_CODE (t) == TEMPLATE_TYPE_PARM)
TEMPLATE_TYPE_PARM_FOR_CLASS (t) = true;
}
}
else if (TREE_CODE (decl) == TYPE_DECL
&& TYPE_DECL_ALIAS_P (decl))
gcc_assert (!DECL_ARTIFICIAL (decl));
else if (VAR_P (decl))
;
else
{
error ("template declaration of %q#D", decl);
return error_mark_node;
}
}
if (!is_friend || TREE_CODE (decl) != FUNCTION_DECL)
check_default_tmpl_args (decl, current_template_parms,
is_primary, is_partial, is_friend);
if (TREE_CODE (decl) == FUNCTION_DECL)
{
tree type = TREE_TYPE (decl);
tree arg = DECL_ARGUMENTS (decl);
tree argtype = TYPE_ARG_TYPES (type);
while (arg && argtype)
{
if (!DECL_PACK_P (arg)
&& check_for_bare_parameter_packs (TREE_TYPE (arg)))
{
TREE_TYPE (arg) = error_mark_node;
TREE_VALUE (argtype) = error_mark_node;
}
arg = DECL_CHAIN (arg);
argtype = TREE_CHAIN (argtype);
}
if (check_for_bare_parameter_packs (TREE_TYPE (type)))
TREE_TYPE (type) = integer_type_node;
if (check_for_bare_parameter_packs (TYPE_RAISES_EXCEPTIONS (type)))
TYPE_RAISES_EXCEPTIONS (type) = NULL_TREE;
}
else if (check_for_bare_parameter_packs ((TREE_CODE (decl) == TYPE_DECL
&& TYPE_DECL_ALIAS_P (decl))
? DECL_ORIGINAL_TYPE (decl)
: TREE_TYPE (decl)))
{
TREE_TYPE (decl) = error_mark_node;
return error_mark_node;
}
if (is_partial)
return process_partial_specialization (decl);
args = current_template_args ();
if (!ctx
|| TREE_CODE (ctx) == FUNCTION_DECL
|| (CLASS_TYPE_P (ctx) && TYPE_BEING_DEFINED (ctx))
|| (TREE_CODE (decl) == TYPE_DECL
&& LAMBDA_TYPE_P (TREE_TYPE (decl)))
|| (is_friend && !DECL_TEMPLATE_INFO (decl)))
{
if (DECL_LANG_SPECIFIC (decl)
&& DECL_TEMPLATE_INFO (decl)
&& DECL_TI_TEMPLATE (decl))
tmpl = DECL_TI_TEMPLATE (decl);
else if (DECL_IMPLICIT_TYPEDEF_P (decl)
&& TYPE_TEMPLATE_INFO (TREE_TYPE (decl))
&& TYPE_TI_TEMPLATE (TREE_TYPE (decl)))
{
redeclare_class_template (TREE_TYPE (decl),
current_template_parms,
current_template_constraints ());
tmpl = TYPE_TI_TEMPLATE (TREE_TYPE (decl));
}
else
{
tmpl = build_template_decl (decl, current_template_parms,
member_template_p);
new_template_p = 1;
if (DECL_LANG_SPECIFIC (decl)
&& DECL_TEMPLATE_SPECIALIZATION (decl))
{
SET_DECL_TEMPLATE_SPECIALIZATION (tmpl);
DECL_TEMPLATE_INFO (tmpl) = DECL_TEMPLATE_INFO (decl);
DECL_TEMPLATE_INFO (decl) = NULL_TREE;
}
}
}
else
{
tree a, t, current, parms;
int i;
tree tinfo = get_template_info (decl);
if (!tinfo)
{
error ("template definition of non-template %q#D", decl);
return error_mark_node;
}
tmpl = TI_TEMPLATE (tinfo);
if (DECL_FUNCTION_TEMPLATE_P (tmpl)
&& DECL_TEMPLATE_INFO (decl) && DECL_TI_ARGS (decl)
&& DECL_TEMPLATE_SPECIALIZATION (decl)
&& DECL_MEMBER_TEMPLATE_P (tmpl))
{
tree new_tmpl;
args = DECL_TI_ARGS (decl);
new_tmpl
= build_template_decl (decl, current_template_parms,
member_template_p);
DECL_TEMPLATE_RESULT (new_tmpl) = decl;
TREE_TYPE (new_tmpl) = TREE_TYPE (decl);
DECL_TI_TEMPLATE (decl) = new_tmpl;
SET_DECL_TEMPLATE_SPECIALIZATION (new_tmpl);
DECL_TEMPLATE_INFO (new_tmpl)
= build_template_info (tmpl, args);
register_specialization (new_tmpl,
most_general_template (tmpl),
args,
is_friend, 0);
return decl;
}
parms = DECL_TEMPLATE_PARMS (tmpl);
i = TMPL_PARMS_DEPTH (parms);
if (TMPL_ARGS_DEPTH (args) != i)
{
error ("expected %d levels of template parms for %q#D, got %d",
i, decl, TMPL_ARGS_DEPTH (args));
DECL_INTERFACE_KNOWN (decl) = 1;
return error_mark_node;
}
else
for (current = decl; i > 0; --i, parms = TREE_CHAIN (parms))
{
a = TMPL_ARGS_LEVEL (args, i);
t = INNERMOST_TEMPLATE_PARMS (parms);
if (TREE_VEC_LENGTH (t) != TREE_VEC_LENGTH (a))
{
if (current == decl)
error ("got %d template parameters for %q#D",
TREE_VEC_LENGTH (a), decl);
else
error ("got %d template parameters for %q#T",
TREE_VEC_LENGTH (a), current);
error ("  but %d required", TREE_VEC_LENGTH (t));
DECL_INTERFACE_KNOWN (decl) = 1;
return error_mark_node;
}
if (current == decl)
current = ctx;
else if (current == NULL_TREE)
break;
else
current = get_containing_scope (current);
}
if (!comp_template_args
(TI_ARGS (tinfo),
TI_ARGS (get_template_info (DECL_TEMPLATE_RESULT (tmpl)))))
{
error ("template arguments to %qD do not match original "
"template %qD", decl, DECL_TEMPLATE_RESULT (tmpl));
if (!uses_template_parms (TI_ARGS (tinfo)))
inform (input_location, "use %<template<>%> for"
" an explicit specialization");
DECL_INTERFACE_KNOWN (decl) = 1;
return error_mark_node;
}
}
DECL_TEMPLATE_RESULT (tmpl) = decl;
TREE_TYPE (tmpl) = TREE_TYPE (decl);
if (new_template_p && !ctx
&& !(is_friend && template_class_depth (current_class_type) > 0))
{
tmpl = pushdecl_namespace_level (tmpl, is_friend);
if (tmpl == error_mark_node)
return error_mark_node;
if (is_friend && TREE_CODE (decl) == TYPE_DECL)
{
DECL_ANTICIPATED (tmpl) = 1;
DECL_FRIEND_P (tmpl) = 1;
}
}
if (is_primary)
{
tree parms = DECL_TEMPLATE_PARMS (tmpl);
DECL_PRIMARY_TEMPLATE (tmpl) = tmpl;
parms = INNERMOST_TEMPLATE_PARMS (parms);
for (int i = TREE_VEC_LENGTH (parms) - 1; i >= 0; --i)
{
tree parm = TREE_VALUE (TREE_VEC_ELT (parms, i));
if (TREE_CODE (parm) == TEMPLATE_DECL)
DECL_CONTEXT (parm) = tmpl;
}
if (TREE_CODE (decl) == TYPE_DECL
&& TYPE_DECL_ALIAS_P (decl)
&& complex_alias_template_p (tmpl))
TEMPLATE_DECL_COMPLEX_ALIAS_P (tmpl) = true;
}
if (DECL_TEMPLATE_INFO (tmpl))
args = add_outermost_template_args (DECL_TI_ARGS (tmpl), args);
info = build_template_info (tmpl, args);
if (DECL_IMPLICIT_TYPEDEF_P (decl))
SET_TYPE_TEMPLATE_INFO (TREE_TYPE (tmpl), info);
else
{
if (is_primary)
retrofit_lang_decl (decl);
if (DECL_LANG_SPECIFIC (decl))
DECL_TEMPLATE_INFO (decl) = info;
}
if (flag_implicit_templates
&& !is_friend
&& TREE_PUBLIC (decl)
&& VAR_OR_FUNCTION_DECL_P (decl))
DECL_COMDAT (decl) = true;
return DECL_TEMPLATE_RESULT (tmpl);
}
tree
push_template_decl (tree decl)
{
return push_template_decl_real (decl, false);
}
tree
add_inherited_template_parms (tree fn, tree inherited)
{
tree inner_parms
= INNERMOST_TEMPLATE_PARMS (DECL_TEMPLATE_PARMS (inherited));
inner_parms = copy_node (inner_parms);
tree parms
= tree_cons (size_int (processing_template_decl + 1),
inner_parms, current_template_parms);
tree tmpl = build_template_decl (fn, parms, true);
tree args = template_parms_to_args (parms);
DECL_TEMPLATE_INFO (fn) = build_template_info (tmpl, args);
TREE_TYPE (tmpl) = TREE_TYPE (fn);
DECL_TEMPLATE_RESULT (tmpl) = fn;
DECL_ARTIFICIAL (tmpl) = true;
DECL_PRIMARY_TEMPLATE (tmpl) = tmpl;
return tmpl;
}
bool
redeclare_class_template (tree type, tree parms, tree cons)
{
tree tmpl;
tree tmpl_parms;
int i;
if (!TYPE_TEMPLATE_INFO (type))
{
error ("%qT is not a template type", type);
return false;
}
tmpl = TYPE_TI_TEMPLATE (type);
if (!PRIMARY_TEMPLATE_P (tmpl))
return true;
if (!parms)
{
error ("template specifiers not specified in declaration of %qD",
tmpl);
return false;
}
parms = INNERMOST_TEMPLATE_PARMS (parms);
tmpl_parms = DECL_INNERMOST_TEMPLATE_PARMS (tmpl);
if (TREE_VEC_LENGTH (parms) != TREE_VEC_LENGTH (tmpl_parms))
{
error_n (input_location, TREE_VEC_LENGTH (parms),
"redeclared with %d template parameter",
"redeclared with %d template parameters",
TREE_VEC_LENGTH (parms));
inform_n (DECL_SOURCE_LOCATION (tmpl), TREE_VEC_LENGTH (tmpl_parms),
"previous declaration %qD used %d template parameter",
"previous declaration %qD used %d template parameters",
tmpl, TREE_VEC_LENGTH (tmpl_parms));
return false;
}
for (i = 0; i < TREE_VEC_LENGTH (tmpl_parms); ++i)
{
tree tmpl_parm;
tree parm;
tree tmpl_default;
tree parm_default;
if (TREE_VEC_ELT (tmpl_parms, i) == error_mark_node
|| TREE_VEC_ELT (parms, i) == error_mark_node)
continue;
tmpl_parm = TREE_VALUE (TREE_VEC_ELT (tmpl_parms, i));
if (error_operand_p (tmpl_parm))
return false;
parm = TREE_VALUE (TREE_VEC_ELT (parms, i));
tmpl_default = TREE_PURPOSE (TREE_VEC_ELT (tmpl_parms, i));
parm_default = TREE_PURPOSE (TREE_VEC_ELT (parms, i));
if (TREE_CODE (tmpl_parm) != TREE_CODE (parm)
|| (TREE_CODE (tmpl_parm) != TYPE_DECL
&& !same_type_p (TREE_TYPE (tmpl_parm), TREE_TYPE (parm)))
|| (TREE_CODE (tmpl_parm) != PARM_DECL
&& (TEMPLATE_TYPE_PARAMETER_PACK (TREE_TYPE (tmpl_parm))
!= TEMPLATE_TYPE_PARAMETER_PACK (TREE_TYPE (parm))))
|| (TREE_CODE (tmpl_parm) == PARM_DECL
&& (TEMPLATE_PARM_PARAMETER_PACK (DECL_INITIAL (tmpl_parm))
!= TEMPLATE_PARM_PARAMETER_PACK (DECL_INITIAL (parm)))))
{
error ("template parameter %q+#D", tmpl_parm);
error ("redeclared here as %q#D", parm);
return false;
}
if (tmpl_default != NULL_TREE && parm_default != NULL_TREE)
{
error_at (input_location, "redefinition of default argument for %q#D", parm);
inform (DECL_SOURCE_LOCATION (tmpl_parm),
"original definition appeared here");
return false;
}
if (parm_default != NULL_TREE)
TREE_PURPOSE (TREE_VEC_ELT (tmpl_parms, i)) = parm_default;
else if (tmpl_default != NULL_TREE)
TREE_PURPOSE (TREE_VEC_ELT (parms, i)) = tmpl_default;
if (TREE_CODE (parm) == TEMPLATE_DECL)
{
gcc_assert (DECL_CONTEXT (parm) == NULL_TREE);
DECL_CONTEXT (parm) = tmpl;
}
if (TREE_CODE (parm) == TYPE_DECL)
TEMPLATE_TYPE_PARM_FOR_CLASS (TREE_TYPE (parm)) = true;
}
if (!equivalent_constraints (get_constraints (tmpl), cons))
{
error_at (input_location, "redeclaration %q#D with different "
"constraints", tmpl);
inform (DECL_SOURCE_LOCATION (tmpl),
"original declaration appeared here");
}
return true;
}
tree
instantiate_non_dependent_expr_internal (tree expr, tsubst_flags_t complain)
{
return tsubst_copy_and_build (expr,
NULL_TREE,
complain,
NULL_TREE,
false,
true);
}
tree
instantiate_non_dependent_expr_sfinae (tree expr, tsubst_flags_t complain)
{
if (expr == NULL_TREE)
return NULL_TREE;
if (processing_template_decl
&& is_nondependent_constant_expression (expr))
{
processing_template_decl_sentinel s;
expr = instantiate_non_dependent_expr_internal (expr, complain);
}
return expr;
}
tree
instantiate_non_dependent_expr (tree expr)
{
return instantiate_non_dependent_expr_sfinae (expr, tf_error);
}
tree
instantiate_non_dependent_or_null (tree expr)
{
if (expr == NULL_TREE)
return NULL_TREE;
if (processing_template_decl)
{
if (!is_nondependent_constant_expression (expr))
expr = NULL_TREE;
else
{
processing_template_decl_sentinel s;
expr = instantiate_non_dependent_expr_internal (expr, tf_error);
}
}
return expr;
}
bool
variable_template_specialization_p (tree t)
{
if (!VAR_P (t) || !DECL_LANG_SPECIFIC (t) || !DECL_TEMPLATE_INFO (t))
return false;
tree tmpl = DECL_TI_TEMPLATE (t);
return variable_template_p (tmpl);
}
bool
alias_type_or_template_p (tree t)
{
if (t == NULL_TREE)
return false;
return ((TREE_CODE (t) == TYPE_DECL && TYPE_DECL_ALIAS_P (t))
|| (TYPE_P (t)
&& TYPE_NAME (t)
&& TYPE_DECL_ALIAS_P (TYPE_NAME (t)))
|| DECL_ALIAS_TEMPLATE_P (t));
}
bool
alias_template_specialization_p (const_tree t)
{
if (TYPE_ALIAS_P (t))
if (tree tinfo = TYPE_ALIAS_TEMPLATE_INFO (t))
return PRIMARY_TEMPLATE_P (TI_TEMPLATE (tinfo));
return false;
}
struct uses_all_template_parms_data
{
int level;
bool *seen;
};
static int
uses_all_template_parms_r (tree t, void *data_)
{
struct uses_all_template_parms_data &data
= *(struct uses_all_template_parms_data*)data_;
tree idx = get_template_parm_index (t);
if (TEMPLATE_PARM_LEVEL (idx) == data.level)
data.seen[TEMPLATE_PARM_IDX (idx)] = true;
return 0;
}
static bool
complex_alias_template_p (const_tree tmpl)
{
struct uses_all_template_parms_data data;
tree pat = DECL_ORIGINAL_TYPE (DECL_TEMPLATE_RESULT (tmpl));
tree parms = DECL_TEMPLATE_PARMS (tmpl);
data.level = TMPL_PARMS_DEPTH (parms);
int len = TREE_VEC_LENGTH (INNERMOST_TEMPLATE_PARMS (parms));
data.seen = XALLOCAVEC (bool, len);
for (int i = 0; i < len; ++i)
data.seen[i] = false;
for_each_template_parm (pat, uses_all_template_parms_r, &data, NULL, true);
for (int i = 0; i < len; ++i)
if (!data.seen[i])
return true;
return false;
}
bool
dependent_alias_template_spec_p (const_tree t)
{
if (!alias_template_specialization_p (t))
return false;
tree tinfo = TYPE_ALIAS_TEMPLATE_INFO (t);
if (!TEMPLATE_DECL_COMPLEX_ALIAS_P (TI_TEMPLATE (tinfo)))
return false;
tree args = INNERMOST_TEMPLATE_ARGS (TI_ARGS (tinfo));
if (!any_dependent_template_arguments_p (args))
return false;
return true;
}
static int
num_innermost_template_parms (tree tmpl)
{
tree parms = INNERMOST_TEMPLATE_PARMS (DECL_TEMPLATE_PARMS (tmpl));
return TREE_VEC_LENGTH (parms);
}
static tree
get_underlying_template (tree tmpl)
{
gcc_assert (TREE_CODE (tmpl) == TEMPLATE_DECL);
while (DECL_ALIAS_TEMPLATE_P (tmpl))
{
tree orig_type = DECL_ORIGINAL_TYPE (DECL_TEMPLATE_RESULT (tmpl));
tree tinfo = TYPE_TEMPLATE_INFO_MAYBE_ALIAS (orig_type);
if (!tinfo)
break;
tree underlying = TI_TEMPLATE (tinfo);
if (!PRIMARY_TEMPLATE_P (underlying)
|| (num_innermost_template_parms (tmpl)
!= num_innermost_template_parms (underlying)))
break;
tree alias_args = INNERMOST_TEMPLATE_ARGS
(template_parms_to_args (DECL_TEMPLATE_PARMS (tmpl)));
if (!comp_template_args (TI_ARGS (tinfo), alias_args))
break;
tmpl = underlying;
}
return tmpl;
}
static tree
convert_nontype_argument_function (tree type, tree expr,
tsubst_flags_t complain)
{
tree fns = expr;
tree fn, fn_no_ptr;
linkage_kind linkage;
fn = instantiate_type (type, fns, tf_none);
if (fn == error_mark_node)
return error_mark_node;
if (value_dependent_expression_p (fn))
goto accept;
fn_no_ptr = strip_fnptr_conv (fn);
if (TREE_CODE (fn_no_ptr) == ADDR_EXPR)
fn_no_ptr = TREE_OPERAND (fn_no_ptr, 0);
if (BASELINK_P (fn_no_ptr))
fn_no_ptr = BASELINK_FUNCTIONS (fn_no_ptr);
if (TREE_CODE (fn_no_ptr) != FUNCTION_DECL)
{
if (complain & tf_error)
{
error ("%qE is not a valid template argument for type %qT",
expr, type);
if (TYPE_PTR_P (type))
inform (input_location, "it must be the address of a function "
"with external linkage");
else
inform (input_location, "it must be the name of a function with "
"external linkage");
}
return NULL_TREE;
}
linkage = decl_linkage (fn_no_ptr);
if (cxx_dialect >= cxx11 ? linkage == lk_none : linkage != lk_external)
{
if (complain & tf_error)
{
if (cxx_dialect >= cxx11)
error ("%qE is not a valid template argument for type %qT "
"because %qD has no linkage",
expr, type, fn_no_ptr);
else
error ("%qE is not a valid template argument for type %qT "
"because %qD does not have external linkage",
expr, type, fn_no_ptr);
}
return NULL_TREE;
}
accept:
if (TREE_CODE (type) == REFERENCE_TYPE)
{
if (REFERENCE_REF_P (fn))
fn = TREE_OPERAND (fn, 0);
else
fn = build_address (fn);
}
if (!same_type_ignoring_top_level_qualifiers_p (type, TREE_TYPE (fn)))
fn = build_nop (type, fn);
return fn;
}
static bool
check_valid_ptrmem_cst_expr (tree type, tree expr,
tsubst_flags_t complain)
{
location_t loc = EXPR_LOC_OR_LOC (expr, input_location);
tree orig_expr = expr;
STRIP_NOPS (expr);
if (null_ptr_cst_p (expr))
return true;
if (TREE_CODE (expr) == PTRMEM_CST
&& same_type_p (TYPE_PTRMEM_CLASS_TYPE (type),
PTRMEM_CST_CLASS (expr)))
return true;
if (cxx_dialect >= cxx11 && null_member_pointer_value_p (expr))
return true;
if (processing_template_decl
&& TREE_CODE (expr) == ADDR_EXPR
&& TREE_CODE (TREE_OPERAND (expr, 0)) == OFFSET_REF)
return true;
if (complain & tf_error)
{
error_at (loc, "%qE is not a valid template argument for type %qT",
orig_expr, type);
if (TREE_CODE (expr) != PTRMEM_CST)
inform (loc, "it must be a pointer-to-member of the form %<&X::Y%>");
else
inform (loc, "because it is a member of %qT", PTRMEM_CST_CLASS (expr));
}
return false;
}
static bool
has_value_dependent_address (tree op)
{
if (DECL_P (op))
{
tree ctx = CP_DECL_CONTEXT (op);
if (TYPE_P (ctx) && dependent_type_p (ctx))
return true;
}
return false;
}
static int
unify_success (bool )
{
return 0;
}
static int
unify_invalid (bool )
{
return 1;
}
static int
unify_parameter_deduction_failure (bool explain_p, tree parm)
{
if (explain_p)
inform (input_location,
"  couldn't deduce template parameter %qD", parm);
return unify_invalid (explain_p);
}
static int
unify_cv_qual_mismatch (bool explain_p, tree parm, tree arg)
{
if (explain_p)
inform (input_location,
"  types %qT and %qT have incompatible cv-qualifiers",
parm, arg);
return unify_invalid (explain_p);
}
static int
unify_type_mismatch (bool explain_p, tree parm, tree arg)
{
if (explain_p)
inform (input_location, "  mismatched types %qT and %qT", parm, arg);
return unify_invalid (explain_p);
}
static int
unify_parameter_pack_mismatch (bool explain_p, tree parm, tree arg)
{
if (explain_p)
inform (input_location,
"  template parameter %qD is not a parameter pack, but "
"argument %qD is",
parm, arg);
return unify_invalid (explain_p);
}
static int
unify_ptrmem_cst_mismatch (bool explain_p, tree parm, tree arg)
{
if (explain_p)
inform (input_location,
"  template argument %qE does not match "
"pointer-to-member constant %qE",
arg, parm);
return unify_invalid (explain_p);
}
static int
unify_expression_unequal (bool explain_p, tree parm, tree arg)
{
if (explain_p)
inform (input_location, "  %qE is not equivalent to %qE", parm, arg);
return unify_invalid (explain_p);
}
static int
unify_parameter_pack_inconsistent (bool explain_p, tree old_arg, tree new_arg)
{
if (explain_p)
inform (input_location,
"  inconsistent parameter pack deduction with %qT and %qT",
old_arg, new_arg);
return unify_invalid (explain_p);
}
static int
unify_inconsistency (bool explain_p, tree parm, tree first, tree second)
{
if (explain_p)
{
if (TYPE_P (parm))
inform (input_location,
"  deduced conflicting types for parameter %qT (%qT and %qT)",
parm, first, second);
else
inform (input_location,
"  deduced conflicting values for non-type parameter "
"%qE (%qE and %qE)", parm, first, second);
}
return unify_invalid (explain_p);
}
static int
unify_vla_arg (bool explain_p, tree arg)
{
if (explain_p)
inform (input_location,
"  variable-sized array type %qT is not "
"a valid template argument",
arg);
return unify_invalid (explain_p);
}
static int
unify_method_type_error (bool explain_p, tree arg)
{
if (explain_p)
inform (input_location,
"  member function type %qT is not a valid template argument",
arg);
return unify_invalid (explain_p);
}
static int
unify_arity (bool explain_p, int have, int wanted, bool least_p = false)
{
if (explain_p)
{
if (least_p)
inform_n (input_location, wanted,
"  candidate expects at least %d argument, %d provided",
"  candidate expects at least %d arguments, %d provided",
wanted, have);
else
inform_n (input_location, wanted,
"  candidate expects %d argument, %d provided",
"  candidate expects %d arguments, %d provided",
wanted, have);
}
return unify_invalid (explain_p);
}
static int
unify_too_many_arguments (bool explain_p, int have, int wanted)
{
return unify_arity (explain_p, have, wanted);
}
static int
unify_too_few_arguments (bool explain_p, int have, int wanted,
bool least_p = false)
{
return unify_arity (explain_p, have, wanted, least_p);
}
static int
unify_arg_conversion (bool explain_p, tree to_type,
tree from_type, tree arg)
{
if (explain_p)
inform (EXPR_LOC_OR_LOC (arg, input_location),
"  cannot convert %qE (type %qT) to type %qT",
arg, from_type, to_type);
return unify_invalid (explain_p);
}
static int
unify_no_common_base (bool explain_p, enum template_base_result r,
tree parm, tree arg)
{
if (explain_p)
switch (r)
{
case tbr_ambiguous_baseclass:
inform (input_location, "  %qT is an ambiguous base class of %qT",
parm, arg);
break;
default:
inform (input_location, "  %qT is not derived from %qT", arg, parm);
break;
}
return unify_invalid (explain_p);
}
static int
unify_inconsistent_template_template_parameters (bool explain_p)
{
if (explain_p)
inform (input_location,
"  template parameters of a template template argument are "
"inconsistent with other deduced template arguments");
return unify_invalid (explain_p);
}
static int
unify_template_deduction_failure (bool explain_p, tree parm, tree arg)
{
if (explain_p)
inform (input_location,
"  can't deduce a template for %qT from non-template type %qT",
parm, arg);
return unify_invalid (explain_p);
}
static int
unify_template_argument_mismatch (bool explain_p, tree parm, tree arg)
{
if (explain_p)
inform (input_location,
"  template argument %qE does not match %qE", arg, parm);
return unify_invalid (explain_p);
}
static tree
convert_nontype_argument (tree type, tree expr, tsubst_flags_t complain)
{
tree expr_type;
location_t loc = EXPR_LOC_OR_LOC (expr, input_location);
tree orig_expr = expr;
if (TREE_CODE (expr) == STRING_CST)
{
if (complain & tf_error)
error ("%qE is not a valid template argument for type %qT "
"because string literals can never be used in this context",
expr, type);
return NULL_TREE;
}
if (TYPE_PTROBV_P (type)
&& TREE_CODE (TREE_TYPE (expr)) == ARRAY_TYPE)
{
expr = decay_conversion (expr, complain);
if (expr == error_mark_node)
return error_mark_node;
}
bool non_dep = false;
if (TYPE_REF_OBJ_P (type)
&& has_value_dependent_address (expr))
;
else if (processing_template_decl
&& is_nondependent_constant_expression (expr))
non_dep = true;
if (error_operand_p (expr))
return error_mark_node;
expr_type = TREE_TYPE (expr);
processing_template_decl_sentinel s (non_dep);
if (non_dep)
expr = instantiate_non_dependent_expr_internal (expr, complain);
if (value_dependent_expression_p (expr))
expr = canonicalize_expr_argument (expr, complain);
if (NULLPTR_TYPE_P (expr_type) && TYPE_PTR_OR_PTRMEM_P (type))
expr = fold_simple (convert (type, expr));
if (cxx_dialect >= cxx11)
{
if (TREE_CODE (expr) == PTRMEM_CST)
;
else if (INTEGRAL_OR_ENUMERATION_TYPE_P (type)
|| cxx_dialect >= cxx17)
{
expr = build_converted_constant_expr (type, expr, complain);
if (expr == error_mark_node)
return error_mark_node;
expr = maybe_constant_value (expr);
expr = convert_from_reference (expr);
}
else if (TYPE_PTR_OR_PTRMEM_P (type))
{
tree folded = maybe_constant_value (expr);
if (TYPE_PTR_P (type) ? integer_zerop (folded)
: null_member_pointer_value_p (folded))
expr = folded;
}
}
if (TREE_CODE (type) == REFERENCE_TYPE)
expr = mark_lvalue_use (expr);
else
expr = mark_rvalue_use (expr);
if (TYPE_REF_OBJ_P (type) || TYPE_REFFN_P (type))
{
tree probe_type, probe = expr;
if (REFERENCE_REF_P (probe))
probe = TREE_OPERAND (probe, 0);
probe_type = TREE_TYPE (probe);
if (TREE_CODE (probe) == NOP_EXPR)
{
tree addr = TREE_OPERAND (probe, 0);
if (TREE_CODE (probe_type) == REFERENCE_TYPE
&& TREE_CODE (addr) == ADDR_EXPR
&& TYPE_PTR_P (TREE_TYPE (addr))
&& (same_type_ignoring_top_level_qualifiers_p
(TREE_TYPE (probe_type),
TREE_TYPE (TREE_TYPE (addr)))))
{
expr = TREE_OPERAND (addr, 0);
expr_type = TREE_TYPE (probe_type);
}
}
}
if (INTEGRAL_OR_ENUMERATION_TYPE_P (type))
{
if (cxx_dialect < cxx11)
{
tree t = build_converted_constant_expr (type, expr, complain);
t = maybe_constant_value (t);
if (t != error_mark_node)
expr = t;
}
if (!same_type_ignoring_top_level_qualifiers_p (type, TREE_TYPE (expr)))
return error_mark_node;
if (TREE_CODE (expr) != INTEGER_CST
&& !value_dependent_expression_p (expr))
{
if (complain & tf_error)
{
int errs = errorcount, warns = warningcount + werrorcount;
if (!require_potential_constant_expression (expr))
expr = error_mark_node;
else
expr = cxx_constant_value (expr);
if (errorcount > errs || warningcount + werrorcount > warns)
inform (loc, "in template argument for type %qT", type);
if (expr == error_mark_node)
return NULL_TREE;
if (TREE_CODE (expr) != INTEGER_CST)
{
gcc_checking_assert (reduced_constant_expression_p (expr));
error_at (loc, "template argument %qE for type %qT not "
"a constant integer", expr, type);
return NULL_TREE;
}
}
else
return NULL_TREE;
}
if (TREE_TYPE (expr) != type)
expr = fold_convert (type, expr);
}
else if (TYPE_PTROBV_P (type))
{
tree decayed = expr;
if (TREE_CODE (expr) == NOP_EXPR)
{
tree probe = expr;
STRIP_NOPS (probe);
if (TREE_CODE (probe) == ADDR_EXPR
&& TYPE_PTR_P (TREE_TYPE (probe)))
{
expr = probe;
expr_type = TREE_TYPE (expr);
}
}
if (value_dependent_expression_p (expr))
;
else if (cxx_dialect >= cxx11 && integer_zerop (expr))
;
else if (TREE_CODE (expr) != ADDR_EXPR)
{
if (VAR_P (expr))
{
if (complain & tf_error)
error ("%qD is not a valid template argument "
"because %qD is a variable, not the address of "
"a variable", orig_expr, expr);
return NULL_TREE;
}
if (POINTER_TYPE_P (expr_type))
{
if (complain & tf_error)
error ("%qE is not a valid template argument for %qT "
"because it is not the address of a variable",
orig_expr, type);
return NULL_TREE;
}
return error_mark_node;
}
else
{
tree decl = TREE_OPERAND (expr, 0);
if (!VAR_P (decl))
{
if (complain & tf_error)
error ("%qE is not a valid template argument of type %qT "
"because %qE is not a variable", orig_expr, type, decl);
return NULL_TREE;
}
else if (cxx_dialect < cxx11 && !DECL_EXTERNAL_LINKAGE_P (decl))
{
if (complain & tf_error)
error ("%qE is not a valid template argument of type %qT "
"because %qD does not have external linkage",
orig_expr, type, decl);
return NULL_TREE;
}
else if ((cxx_dialect >= cxx11 && cxx_dialect < cxx17)
&& decl_linkage (decl) == lk_none)
{
if (complain & tf_error)
error ("%qE is not a valid template argument of type %qT "
"because %qD has no linkage", orig_expr, type, decl);
return NULL_TREE;
}
else if (DECL_ARTIFICIAL (decl))
{
if (complain & tf_error)
error ("the address of %qD is not a valid template argument",
decl);
return NULL_TREE;
}
else if (!same_type_ignoring_top_level_qualifiers_p
(strip_array_types (TREE_TYPE (type)),
strip_array_types (TREE_TYPE (decl))))
{
if (complain & tf_error)
error ("the address of the %qT subobject of %qD is not a "
"valid template argument", TREE_TYPE (type), decl);
return NULL_TREE;
}
else if (!TREE_STATIC (decl) && !DECL_EXTERNAL (decl))
{
if (complain & tf_error)
error ("the address of %qD is not a valid template argument "
"because it does not have static storage duration",
decl);
return NULL_TREE;
}
}
expr = decayed;
expr = perform_qualification_conversions (type, expr);
if (expr == error_mark_node)
return error_mark_node;
}
else if (TYPE_REF_OBJ_P (type))
{
if (!same_type_ignoring_top_level_qualifiers_p (TREE_TYPE (type),
expr_type))
return error_mark_node;
if (!at_least_as_qualified_p (TREE_TYPE (type), expr_type))
{
if (complain & tf_error)
error ("%qE is not a valid template argument for type %qT "
"because of conflicts in cv-qualification", expr, type);
return NULL_TREE;
}
if (!lvalue_p (expr))
{
if (complain & tf_error)
error ("%qE is not a valid template argument for type %qT "
"because it is not an lvalue", expr, type);
return NULL_TREE;
}
if (INDIRECT_REF_P (expr)
&& TYPE_REF_OBJ_P (TREE_TYPE (TREE_OPERAND (expr, 0))))
{
expr = TREE_OPERAND (expr, 0);
if (DECL_P (expr))
{
if (complain & tf_error)
error ("%q#D is not a valid template argument for type %qT "
"because a reference variable does not have a constant "
"address", expr, type);
return NULL_TREE;
}
}
if (TYPE_REF_OBJ_P (TREE_TYPE (expr))
&& value_dependent_expression_p (expr))
;
else
{
if (!DECL_P (expr))
{
if (complain & tf_error)
error ("%qE is not a valid template argument for type %qT "
"because it is not an object with linkage",
expr, type);
return NULL_TREE;
}
linkage_kind linkage = decl_linkage (expr);
if (linkage < (cxx_dialect >= cxx11 ? lk_internal : lk_external))
{
if (complain & tf_error)
error ("%qE is not a valid template argument for type %qT "
"because object %qD does not have linkage",
expr, type, expr);
return NULL_TREE;
}
expr = build_address (expr);
}
if (!same_type_p (type, TREE_TYPE (expr)))
expr = build_nop (type, expr);
}
else if (TYPE_PTRFN_P (type))
{
if (!type_unknown_p (expr_type))
{
expr = decay_conversion (expr, complain);
if (expr == error_mark_node)
return error_mark_node;
}
if (cxx_dialect >= cxx11 && integer_zerop (expr))
return perform_qualification_conversions (type, expr);
expr = convert_nontype_argument_function (type, expr, complain);
if (!expr || expr == error_mark_node)
return expr;
}
else if (TYPE_REFFN_P (type))
{
if (TREE_CODE (expr) == ADDR_EXPR)
{
if (complain & tf_error)
{
error ("%qE is not a valid template argument for type %qT "
"because it is a pointer", expr, type);
inform (input_location, "try using %qE instead",
TREE_OPERAND (expr, 0));
}
return NULL_TREE;
}
expr = convert_nontype_argument_function (type, expr, complain);
if (!expr || expr == error_mark_node)
return expr;
}
else if (TYPE_PTRMEMFUNC_P (type))
{
expr = instantiate_type (type, expr, tf_none);
if (expr == error_mark_node)
return error_mark_node;
if (!value_dependent_expression_p (expr)
&& !check_valid_ptrmem_cst_expr (type, expr, complain))
return NULL_TREE;
if (fnptr_conv_p (type, TREE_TYPE (expr)))
expr = make_ptrmem_cst (type, PTRMEM_CST_MEMBER (expr));
}
else if (TYPE_PTRDATAMEM_P (type))
{
if (!value_dependent_expression_p (expr)
&& !check_valid_ptrmem_cst_expr (type, expr, complain))
return NULL_TREE;
expr = perform_qualification_conversions (type, expr);
if (expr == error_mark_node)
return expr;
}
else if (NULLPTR_TYPE_P (type))
{
if (!NULLPTR_TYPE_P (TREE_TYPE (expr)))
{
if (complain & tf_error)
error ("%qE is not a valid template argument for type %qT "
"because it is of type %qT", expr, type, TREE_TYPE (expr));
return NULL_TREE;
}
return expr;
}
else
gcc_unreachable ();
gcc_assert (same_type_ignoring_top_level_qualifiers_p
(type, TREE_TYPE (expr)));
return convert_from_reference (expr);
}
static int
coerce_template_template_parm (tree parm,
tree arg,
tsubst_flags_t complain,
tree in_decl,
tree outer_args)
{
if (arg == NULL_TREE || error_operand_p (arg)
|| parm == NULL_TREE || error_operand_p (parm))
return 0;
if (TREE_CODE (arg) != TREE_CODE (parm))
return 0;
switch (TREE_CODE (parm))
{
case TEMPLATE_DECL:
{
tree parmparm = DECL_INNERMOST_TEMPLATE_PARMS (parm);
tree argparm = DECL_INNERMOST_TEMPLATE_PARMS (arg);
if (!coerce_template_template_parms
(parmparm, argparm, complain, in_decl, outer_args))
return 0;
}
case TYPE_DECL:
if (TEMPLATE_TYPE_PARAMETER_PACK (TREE_TYPE (arg))
&& !TEMPLATE_TYPE_PARAMETER_PACK (TREE_TYPE (parm)))
return 0;
break;
case PARM_DECL:
if (!uses_template_parms (TREE_TYPE (arg)))
{
tree t = tsubst (TREE_TYPE (parm), outer_args, complain, in_decl);
if (!uses_template_parms (t)
&& !same_type_p (t, TREE_TYPE (arg)))
return 0;
}
if (TEMPLATE_PARM_PARAMETER_PACK (DECL_INITIAL (arg))
&& !TEMPLATE_PARM_PARAMETER_PACK (DECL_INITIAL (parm)))
return 0;
break;
default:
gcc_unreachable ();
}
return 1;
}
static tree
coerce_template_args_for_ttp (tree templ, tree arglist,
tsubst_flags_t complain)
{
tree outer = DECL_CONTEXT (templ);
if (outer)
{
if (DECL_TEMPLATE_SPECIALIZATION (outer))
outer = template_parms_to_args (DECL_TEMPLATE_PARMS (outer));
else
outer = TI_ARGS (get_template_info (DECL_TEMPLATE_RESULT (outer)));
}
else if (current_template_parms)
{
tree relevant_template_parms;
relevant_template_parms = current_template_parms;
while (TMPL_PARMS_DEPTH (relevant_template_parms)
!= TEMPLATE_TYPE_LEVEL (TREE_TYPE (templ)))
relevant_template_parms = TREE_CHAIN (relevant_template_parms);
outer = template_parms_to_args (relevant_template_parms);
}
if (outer)
arglist = add_to_template_args (outer, arglist);
tree parmlist = DECL_INNERMOST_TEMPLATE_PARMS (templ);
return coerce_template_parms (parmlist, arglist, templ,
complain,
true,
true);
}
static GTY((deletable)) hash_map<tree,tree> *defaulted_ttp_cache;
static void
store_defaulted_ttp (tree v, tree t)
{
if (!defaulted_ttp_cache)
defaulted_ttp_cache = hash_map<tree,tree>::create_ggc (13);
defaulted_ttp_cache->put (v, t);
}
static tree
lookup_defaulted_ttp (tree v)
{
if (defaulted_ttp_cache)
if (tree *p = defaulted_ttp_cache->get (v))
return *p;
return NULL_TREE;
}
static tree
add_defaults_to_ttp (tree otmpl)
{
if (tree c = lookup_defaulted_ttp (otmpl))
return c;
tree ntmpl = copy_node (otmpl);
tree ntype = copy_node (TREE_TYPE (otmpl));
TYPE_STUB_DECL (ntype) = TYPE_NAME (ntype) = ntmpl;
TYPE_MAIN_VARIANT (ntype) = ntype;
TYPE_POINTER_TO (ntype) = TYPE_REFERENCE_TO (ntype) = NULL_TREE;
TYPE_NAME (ntype) = ntmpl;
SET_TYPE_STRUCTURAL_EQUALITY (ntype);
tree idx = TEMPLATE_TYPE_PARM_INDEX (ntype)
= copy_node (TEMPLATE_TYPE_PARM_INDEX (ntype));
TEMPLATE_PARM_DECL (idx) = ntmpl;
TREE_TYPE (ntmpl) = TREE_TYPE (idx) = ntype;
tree oparms = DECL_TEMPLATE_PARMS (otmpl);
tree parms = DECL_TEMPLATE_PARMS (ntmpl) = copy_node (oparms);
TREE_CHAIN (parms) = TREE_CHAIN (oparms);
tree vec = TREE_VALUE (parms) = copy_node (TREE_VALUE (parms));
for (int i = 0; i < TREE_VEC_LENGTH (vec); ++i)
{
tree o = TREE_VEC_ELT (vec, i);
if (!template_parameter_pack_p (TREE_VALUE (o)))
{
tree n = TREE_VEC_ELT (vec, i) = copy_node (o);
TREE_PURPOSE (n) = any_targ_node;
}
}
store_defaulted_ttp (otmpl, ntmpl);
return ntmpl;
}
static tree
coerce_ttp_args_for_tta (tree& arg, tree pargs, tsubst_flags_t complain)
{
++processing_template_decl;
tree arg_tmpl = TYPE_TI_TEMPLATE (arg);
if (DECL_TEMPLATE_TEMPLATE_PARM_P (arg_tmpl))
{
arg_tmpl = add_defaults_to_ttp (arg_tmpl);
pargs = coerce_template_args_for_ttp (arg_tmpl, pargs, complain);
if (pargs != error_mark_node)
arg = bind_template_template_parm (TREE_TYPE (arg_tmpl),
TYPE_TI_ARGS (arg));
}
else
{
tree aparms
= INNERMOST_TEMPLATE_PARMS (DECL_TEMPLATE_PARMS (arg_tmpl));
pargs = coerce_template_parms (aparms, pargs, arg_tmpl, complain,
true,
true);
}
--processing_template_decl;
return pargs;
}
static int
unify_bound_ttp_args (tree tparms, tree targs, tree parm, tree& arg,
bool explain_p)
{
tree parmvec = TYPE_TI_ARGS (parm);
tree argvec = INNERMOST_TEMPLATE_ARGS (TYPE_TI_ARGS (arg));
parmvec = expand_template_argument_pack (parmvec);
argvec = expand_template_argument_pack (argvec);
if (flag_new_ttp)
{
tree nparmvec = parmvec;
nparmvec = coerce_ttp_args_for_tta (arg, parmvec, tf_none);
nparmvec = expand_template_argument_pack (nparmvec);
if (unify (tparms, targs, nparmvec, argvec,
UNIFY_ALLOW_NONE, explain_p))
return 1;
if (flag_new_ttp
&& TREE_VEC_LENGTH (nparmvec) < TREE_VEC_LENGTH (parmvec)
&& unify_pack_expansion (tparms, targs, parmvec, argvec,
DEDUCE_EXACT, true, explain_p))
return 1;
}
else
{
int len = TREE_VEC_LENGTH (parmvec);
int parm_variadic_p = 0;
if (len > 0
&& PACK_EXPANSION_P (TREE_VEC_ELT (parmvec, len - 1)))
parm_variadic_p = 1;
for (int i = 0; i < len - parm_variadic_p; ++i)
if (PACK_EXPANSION_P (TREE_VEC_ELT (parmvec, i)))
return unify_success (explain_p);
if (TREE_VEC_LENGTH (argvec) < len - parm_variadic_p)
return unify_too_few_arguments (explain_p,
TREE_VEC_LENGTH (argvec), len);
for (int i = 0; i < len - parm_variadic_p; ++i)
if (unify (tparms, targs,
TREE_VEC_ELT (parmvec, i),
TREE_VEC_ELT (argvec, i),
UNIFY_ALLOW_NONE, explain_p))
return 1;
if (parm_variadic_p
&& unify_pack_expansion (tparms, targs,
parmvec, argvec,
DEDUCE_EXACT,
true, explain_p))
return 1;
}
return 0;
}
static int
coerce_template_template_parms (tree parm_parms,
tree arg_parms,
tsubst_flags_t complain,
tree in_decl,
tree outer_args)
{
int nparms, nargs, i;
tree parm, arg;
int variadic_p = 0;
gcc_assert (TREE_CODE (parm_parms) == TREE_VEC);
gcc_assert (TREE_CODE (arg_parms) == TREE_VEC);
nparms = TREE_VEC_LENGTH (parm_parms);
nargs = TREE_VEC_LENGTH (arg_parms);
if (flag_new_ttp)
{
tree pargs = template_parms_level_to_args (parm_parms);
++processing_template_decl;
pargs = coerce_template_parms (arg_parms, pargs, NULL_TREE, tf_none,
true, true);
--processing_template_decl;
if (pargs != error_mark_node)
{
tree targs = make_tree_vec (nargs);
tree aargs = template_parms_level_to_args (arg_parms);
if (!unify (arg_parms, targs, aargs, pargs, UNIFY_ALLOW_NONE,
false))
return 1;
}
}
if (TREE_VEC_ELT (parm_parms, nparms - 1) != error_mark_node)
{
parm = TREE_VALUE (TREE_VEC_ELT (parm_parms, nparms - 1));
if (error_operand_p (parm))
return 0;
switch (TREE_CODE (parm))
{
case TEMPLATE_DECL:
case TYPE_DECL:
if (TEMPLATE_TYPE_PARAMETER_PACK (TREE_TYPE (parm)))
variadic_p = 1;
break;
case PARM_DECL:
if (TEMPLATE_PARM_PARAMETER_PACK (DECL_INITIAL (parm)))
variadic_p = 1;
break;
default:
gcc_unreachable ();
}
}
if (nargs != nparms
&& !(variadic_p && nargs >= nparms - 1))
return 0;
for (i = 0; i < nparms - variadic_p; ++i)
{
if (TREE_VEC_ELT (parm_parms, i) == error_mark_node
|| TREE_VEC_ELT (arg_parms, i) == error_mark_node)
continue;
parm = TREE_VALUE (TREE_VEC_ELT (parm_parms, i));
arg = TREE_VALUE (TREE_VEC_ELT (arg_parms, i));
if (!coerce_template_template_parm (parm, arg, complain, in_decl,
outer_args))
return 0;
}
if (variadic_p)
{
if (TREE_VEC_ELT (parm_parms, i) == error_mark_node)
return 0;
parm = TREE_VALUE (TREE_VEC_ELT (parm_parms, i));
for (; i < nargs; ++i)
{
if (TREE_VEC_ELT (arg_parms, i) == error_mark_node)
continue;
arg = TREE_VALUE (TREE_VEC_ELT (arg_parms, i));
if (!coerce_template_template_parm (parm, arg, complain, in_decl,
outer_args))
return 0;
}
}
return 1;
}
bool 
template_template_parm_bindings_ok_p (tree tparms, tree targs)
{
int i, ntparms = TREE_VEC_LENGTH (tparms);
bool ret = true;
++processing_template_decl;
targs = INNERMOST_TEMPLATE_ARGS (targs);
for (i = 0; i < ntparms; ++i)
{
tree tparm = TREE_VALUE (TREE_VEC_ELT (tparms, i));
tree targ = TREE_VEC_ELT (targs, i);
if (TREE_CODE (tparm) == TEMPLATE_DECL && targ)
{
tree packed_args = NULL_TREE;
int idx, len = 1;
if (ARGUMENT_PACK_P (targ))
{
packed_args = ARGUMENT_PACK_ARGS (targ);
len = TREE_VEC_LENGTH (packed_args);
}
for (idx = 0; idx < len; ++idx)
{
tree targ_parms = NULL_TREE;
if (packed_args)
targ = TREE_VEC_ELT (packed_args, idx);
if (PACK_EXPANSION_P (targ))
targ = PACK_EXPANSION_PATTERN (targ);
if (TREE_CODE (targ) == TEMPLATE_DECL)
targ_parms = DECL_INNERMOST_TEMPLATE_PARMS (targ);
else if (TREE_CODE (targ) == TEMPLATE_TEMPLATE_PARM)
targ_parms = DECL_INNERMOST_TEMPLATE_PARMS (TYPE_NAME (targ));
if (targ_parms
&& !coerce_template_template_parms
(DECL_INNERMOST_TEMPLATE_PARMS (tparm),
targ_parms,
tf_none,
tparm,
targs))
{
ret = false;
goto out;
}
}
}
}
out:
--processing_template_decl;
return ret;
}
static tree
canonicalize_type_argument (tree arg, tsubst_flags_t complain)
{
if (!arg || arg == error_mark_node || arg == TYPE_CANONICAL (arg))
return arg;
bool removed_attributes = false;
tree canon = strip_typedefs (arg, &removed_attributes);
if (removed_attributes
&& (complain & tf_warning))
warning (OPT_Wignored_attributes,
"ignoring attributes on template argument %qT", arg);
return canon;
}
static tree
canonicalize_expr_argument (tree arg, tsubst_flags_t complain)
{
if (!arg || arg == error_mark_node)
return arg;
bool removed_attributes = false;
tree canon = strip_typedefs_expr (arg, &removed_attributes);
if (removed_attributes
&& (complain & tf_warning))
warning (OPT_Wignored_attributes,
"ignoring attributes in template argument %qE", arg);
return canon;
}
static bool
is_compatible_template_arg (tree parm, tree arg)
{
tree parm_cons = get_constraints (parm);
if (parm_cons == NULL_TREE)
return true;
tree arg_cons = get_constraints (arg);
if (parm_cons)
{
tree args = template_parms_to_args (DECL_TEMPLATE_PARMS (arg));
parm_cons = tsubst_constraint_info (parm_cons,
INNERMOST_TEMPLATE_ARGS (args),
tf_none, NULL_TREE);
if (parm_cons == error_mark_node)
return false;
}
return subsumes (parm_cons, arg_cons);
}
static inline tree
convert_wildcard_argument (tree parm, tree arg)
{
TREE_TYPE (arg) = parm;
return arg;
}
static tree
maybe_convert_nontype_argument (tree type, tree arg)
{
if (type_uses_auto (type))
return arg;
if (value_dependent_expression_p (arg))
return arg;
type = cv_unqualified (type);
tree argtype = TREE_TYPE (arg);
if (same_type_p (type, argtype))
return arg;
arg = build1 (IMPLICIT_CONV_EXPR, type, arg);
IMPLICIT_CONV_EXPR_NONTYPE_ARG (arg) = true;
return arg;
}
static tree
convert_template_argument (tree parm,
tree arg,
tree args,
tsubst_flags_t complain,
int i,
tree in_decl)
{
tree orig_arg;
tree val;
int is_type, requires_type, is_tmpl_type, requires_tmpl_type;
if (parm == error_mark_node)
return error_mark_node;
if (TREE_CODE (arg) == WILDCARD_DECL)
return convert_wildcard_argument (parm, arg);
if (arg == any_targ_node)
return arg;
if (TREE_CODE (arg) == TREE_LIST
&& TREE_CODE (TREE_VALUE (arg)) == OFFSET_REF)
{
orig_arg = TREE_VALUE (arg);
TREE_TYPE (arg) = unknown_type_node;
}
orig_arg = arg;
requires_tmpl_type = TREE_CODE (parm) == TEMPLATE_DECL;
requires_type = (TREE_CODE (parm) == TYPE_DECL
|| requires_tmpl_type);
if (TREE_CODE (arg) == TYPE_PACK_EXPANSION)
arg = PACK_EXPANSION_PATTERN (arg);
if (requires_tmpl_type && CLASS_TYPE_P (arg))
{
tree t = maybe_get_template_decl_from_type_decl (TYPE_NAME (arg));
if (TREE_CODE (t) == TEMPLATE_DECL)
{
if (cxx_dialect >= cxx11)
;
else if (complain & tf_warning_or_error)
pedwarn (input_location, OPT_Wpedantic, "injected-class-name %qD"
" used as template template argument", TYPE_NAME (arg));
else if (flag_pedantic_errors)
t = arg;
arg = t;
}
}
is_tmpl_type = 
((TREE_CODE (arg) == TEMPLATE_DECL
&& TREE_CODE (DECL_TEMPLATE_RESULT (arg)) == TYPE_DECL)
|| (requires_tmpl_type && TREE_CODE (arg) == TYPE_ARGUMENT_PACK)
|| TREE_CODE (arg) == TEMPLATE_TEMPLATE_PARM
|| TREE_CODE (arg) == UNBOUND_CLASS_TEMPLATE);
if (is_tmpl_type
&& (TREE_CODE (arg) == TEMPLATE_TEMPLATE_PARM
|| TREE_CODE (arg) == UNBOUND_CLASS_TEMPLATE))
arg = TYPE_STUB_DECL (arg);
is_type = TYPE_P (arg) || is_tmpl_type;
if (requires_type && ! is_type && TREE_CODE (arg) == SCOPE_REF
&& TREE_CODE (TREE_OPERAND (arg, 0)) == TEMPLATE_TYPE_PARM)
{
if (TREE_CODE (TREE_OPERAND (arg, 1)) == BIT_NOT_EXPR)
{
if (complain & tf_error)
error ("invalid use of destructor %qE as a type", orig_arg);
return error_mark_node;
}
permerror (input_location,
"to refer to a type member of a template parameter, "
"use %<typename %E%>", orig_arg);
orig_arg = make_typename_type (TREE_OPERAND (arg, 0),
TREE_OPERAND (arg, 1),
typename_type,
complain);
arg = orig_arg;
is_type = 1;
}
if (is_type != requires_type)
{
if (in_decl)
{
if (complain & tf_error)
{
error ("type/value mismatch at argument %d in template "
"parameter list for %qD",
i + 1, in_decl);
if (is_type)
inform (input_location,
"  expected a constant of type %qT, got %qT",
TREE_TYPE (parm),
(DECL_P (arg) ? DECL_NAME (arg) : orig_arg));
else if (requires_tmpl_type)
inform (input_location,
"  expected a class template, got %qE", orig_arg);
else
inform (input_location,
"  expected a type, got %qE", orig_arg);
}
}
return error_mark_node;
}
if (is_tmpl_type ^ requires_tmpl_type)
{
if (in_decl && (complain & tf_error))
{
error ("type/value mismatch at argument %d in template "
"parameter list for %qD",
i + 1, in_decl);
if (is_tmpl_type)
inform (input_location,
"  expected a type, got %qT", DECL_NAME (arg));
else
inform (input_location,
"  expected a class template, got %qT", orig_arg);
}
return error_mark_node;
}
if (template_parameter_pack_p (parm) && ARGUMENT_PACK_P (orig_arg))
val = orig_arg;
else if (is_type)
{
if (requires_tmpl_type)
{
if (TREE_CODE (TREE_TYPE (arg)) == UNBOUND_CLASS_TEMPLATE)
val = orig_arg;
else
{
tree parmparm = DECL_INNERMOST_TEMPLATE_PARMS (parm);
tree argparm;
arg = get_underlying_template (arg);
argparm = DECL_INNERMOST_TEMPLATE_PARMS (arg);
if (coerce_template_template_parms (parmparm, argparm,
complain, in_decl,
args))
{
val = arg;
if (val != error_mark_node)
{
if (DECL_TEMPLATE_TEMPLATE_PARM_P (val))
val = TREE_TYPE (val);
if (TREE_CODE (orig_arg) == TYPE_PACK_EXPANSION)
val = make_pack_expansion (val, complain);
}
}
else
{
if (in_decl && (complain & tf_error))
{
error ("type/value mismatch at argument %d in "
"template parameter list for %qD",
i + 1, in_decl);
inform (input_location,
"  expected a template of type %qD, got %qT",
parm, orig_arg);
}
val = error_mark_node;
}
if (val != error_mark_node)
if (!is_compatible_template_arg (parm, arg))
{
if (in_decl && (complain & tf_error))
{
error ("constraint mismatch at argument %d in "
"template parameter list for %qD",
i + 1, in_decl);
inform (input_location, "  expected %qD but got %qD",
parm, arg);
}
val = error_mark_node;
}
}
}
else
val = orig_arg;
if (TYPE_P (val))
val = canonicalize_type_argument (val, complain);
}
else
{
tree t = TREE_TYPE (parm);
if (TEMPLATE_PARM_LEVEL (get_template_parm_index (parm))
> TMPL_ARGS_DEPTH (args))
;
else if (tree a = type_uses_auto (t))
{
t = do_auto_deduction (t, arg, a, complain, adc_unify, args);
if (t == error_mark_node)
return error_mark_node;
}
else
t = tsubst (t, args, complain, in_decl);
if (invalid_nontype_parm_type_p (t, complain))
return error_mark_node;
if (!type_dependent_expression_p (orig_arg)
&& !uses_template_parms (t))
val = convert_nontype_argument (t, orig_arg, complain);
else
{
val = canonicalize_expr_argument (orig_arg, complain);
val = maybe_convert_nontype_argument (t, val);
}
if (val == NULL_TREE)
val = error_mark_node;
else if (val == error_mark_node && (complain & tf_error))
error ("could not convert template argument %qE from %qT to %qT",
orig_arg, TREE_TYPE (orig_arg), t);
if (INDIRECT_REF_P (val))
{
const_tree inner = TREE_OPERAND (val, 0);
const_tree innertype = TREE_TYPE (inner);
if (innertype
&& TREE_CODE (innertype) == REFERENCE_TYPE
&& TREE_CODE (TREE_TYPE (innertype)) == FUNCTION_TYPE
&& TREE_OPERAND_LENGTH (inner) > 0
&& reject_gcc_builtin (TREE_OPERAND (inner, 0)))
return error_mark_node;
}
if (TREE_CODE (val) == SCOPE_REF)
{
tree type = canonicalize_type_argument (TREE_TYPE (val), complain);
tree scope = canonicalize_type_argument (TREE_OPERAND (val, 0),
complain);
val = build_qualified_name (type, scope, TREE_OPERAND (val, 1),
QUALIFIED_NAME_IS_TEMPLATE (val));
}
}
return val;
}
static tree
coerce_template_parameter_pack (tree parms,
int parm_idx,
tree args,
tree inner_args,
int arg_idx,
tree new_args,
int* lost,
tree in_decl,
tsubst_flags_t complain)
{
tree parm = TREE_VEC_ELT (parms, parm_idx);
int nargs = inner_args ? NUM_TMPL_ARGS (inner_args) : 0;
tree packed_args;
tree argument_pack;
tree packed_parms = NULL_TREE;
if (arg_idx > nargs)
arg_idx = nargs;
if (tree packs = fixed_parameter_pack_p (TREE_VALUE (parm)))
{
tree decl = TREE_VALUE (parm);
tree exp = cxx_make_type (TYPE_PACK_EXPANSION);
SET_PACK_EXPANSION_PATTERN (exp, decl);
PACK_EXPANSION_PARAMETER_PACKS (exp) = packs;
SET_TYPE_STRUCTURAL_EQUALITY (exp);
TREE_VEC_LENGTH (args)--;
packed_parms = tsubst_pack_expansion (exp, args, complain, decl);
TREE_VEC_LENGTH (args)++;
if (packed_parms == error_mark_node)
return error_mark_node;
if (arg_idx < nargs
&& PACK_EXPANSION_P (TREE_VEC_ELT (inner_args, arg_idx)))
{
int j, len = TREE_VEC_LENGTH (packed_parms);
for (j = 0; j < len; ++j)
{
tree t = TREE_TYPE (TREE_VEC_ELT (packed_parms, j));
if (invalid_nontype_parm_type_p (t, complain))
return error_mark_node;
}
return NULL_TREE;
}
packed_args = make_tree_vec (TREE_VEC_LENGTH (packed_parms));
}
else if (arg_idx < nargs
&& TREE_CODE (TREE_VEC_ELT (inner_args, arg_idx)) == WILDCARD_DECL
&& WILDCARD_PACK_P (TREE_VEC_ELT (inner_args, arg_idx)))
{
nargs = arg_idx + 1;
packed_args = make_tree_vec (1);
}
else
packed_args = make_tree_vec (nargs - arg_idx);
int first_pack_arg = arg_idx;
for (; arg_idx < nargs; ++arg_idx)
{
tree arg = TREE_VEC_ELT (inner_args, arg_idx);
tree actual_parm = TREE_VALUE (parm);
int pack_idx = arg_idx - first_pack_arg;
if (packed_parms)
{
if (pack_idx >= TREE_VEC_LENGTH (packed_parms))
break;
else if (PACK_EXPANSION_P (arg))
return NULL_TREE;
else
actual_parm = TREE_VEC_ELT (packed_parms, pack_idx);
}
if (arg == error_mark_node)
{
if (complain & tf_error)
error ("template argument %d is invalid", arg_idx + 1);
}
else
arg = convert_template_argument (actual_parm, 
arg, new_args, complain, parm_idx,
in_decl);
if (arg == error_mark_node)
(*lost)++;
TREE_VEC_ELT (packed_args, pack_idx) = arg;
}
if (arg_idx - first_pack_arg < TREE_VEC_LENGTH (packed_args)
&& TREE_VEC_LENGTH (packed_args) > 0)
{
if (complain & tf_error)
error ("wrong number of template arguments (%d, should be %d)",
arg_idx - first_pack_arg, TREE_VEC_LENGTH (packed_args));
return error_mark_node;
}
if (TREE_CODE (TREE_VALUE (parm)) == TYPE_DECL
|| TREE_CODE (TREE_VALUE (parm)) == TEMPLATE_DECL)
argument_pack = cxx_make_type (TYPE_ARGUMENT_PACK);
else
{
argument_pack = make_node (NONTYPE_ARGUMENT_PACK);
TREE_CONSTANT (argument_pack) = 1;
}
SET_ARGUMENT_PACK_ARGS (argument_pack, packed_args);
if (CHECKING_P)
SET_NON_DEFAULT_TEMPLATE_ARGS_COUNT (packed_args,
TREE_VEC_LENGTH (packed_args));
return argument_pack;
}
static int
pack_expansion_args_count (tree args)
{
int i;
int count = 0;
if (args)
for (i = 0; i < TREE_VEC_LENGTH (args); ++i)
{
tree elt = TREE_VEC_ELT (args, i);
if (elt && PACK_EXPANSION_P (elt))
++count;
}
return count;
}
static tree
coerce_template_parms (tree parms,
tree args,
tree in_decl,
tsubst_flags_t complain,
bool require_all_args,
bool use_default_args)
{
int nparms, nargs, parm_idx, arg_idx, lost = 0;
tree orig_inner_args;
tree inner_args;
tree new_args;
tree new_inner_args;
int saved_unevaluated_operand;
int saved_inhibit_evaluation_warnings;
int variadic_p = 0;
int variadic_args_p = 0;
int post_variadic_parms = 0;
int fixed_pack_adjust = 0;
int fixed_packs = 0;
int missing = 0;
int default_p = 0;
if (args == error_mark_node)
return error_mark_node;
nparms = TREE_VEC_LENGTH (parms);
for (parm_idx = 0; parm_idx < nparms; ++parm_idx)
{
tree parm = TREE_VEC_ELT (parms, parm_idx);
if (variadic_p)
++post_variadic_parms;
if (template_parameter_pack_p (TREE_VALUE (parm)))
++variadic_p;
if (TREE_PURPOSE (parm))
++default_p;
}
inner_args = orig_inner_args = INNERMOST_TEMPLATE_ARGS (args);
if (!post_variadic_parms)
inner_args = expand_template_argument_pack (inner_args);
variadic_args_p = pack_expansion_args_count (inner_args);
nargs = inner_args ? NUM_TMPL_ARGS (inner_args) : 0;
if ((nargs - variadic_args_p > nparms && !variadic_p)
|| (nargs < nparms - variadic_p
&& require_all_args
&& !variadic_args_p
&& (!use_default_args
|| (TREE_VEC_ELT (parms, nargs) != error_mark_node
&& !TREE_PURPOSE (TREE_VEC_ELT (parms, nargs))))))
{
bad_nargs:
if (complain & tf_error)
{
if (variadic_p || default_p)
{
nparms -= variadic_p + default_p;
error ("wrong number of template arguments "
"(%d, should be at least %d)", nargs, nparms);
}
else
error ("wrong number of template arguments "
"(%d, should be %d)", nargs, nparms);
if (in_decl)
inform (DECL_SOURCE_LOCATION (in_decl),
"provided for %qD", in_decl);
}
return error_mark_node;
}
else if (in_decl
&& (DECL_ALIAS_TEMPLATE_P (in_decl)
|| concept_template_p (in_decl))
&& variadic_args_p
&& nargs - variadic_args_p < nparms - variadic_p)
{
if (complain & tf_error)
{
for (int i = 0; i < TREE_VEC_LENGTH (inner_args); ++i)
{
tree arg = TREE_VEC_ELT (inner_args, i);
tree parm = TREE_VALUE (TREE_VEC_ELT (parms, i));
if (PACK_EXPANSION_P (arg)
&& !template_parameter_pack_p (parm))
{
if (DECL_ALIAS_TEMPLATE_P (in_decl))
error_at (location_of (arg),
"pack expansion argument for non-pack parameter "
"%qD of alias template %qD", parm, in_decl);
else
error_at (location_of (arg),
"pack expansion argument for non-pack parameter "
"%qD of concept %qD", parm, in_decl);
inform (DECL_SOURCE_LOCATION (parm), "declared here");
goto found;
}
}
gcc_unreachable ();
found:;
}
return error_mark_node;
}
saved_unevaluated_operand = cp_unevaluated_operand;
cp_unevaluated_operand = 0;
saved_inhibit_evaluation_warnings = c_inhibit_evaluation_warnings;
c_inhibit_evaluation_warnings = 0;
new_inner_args = make_tree_vec (nparms);
new_args = add_outermost_template_args (args, new_inner_args);
int pack_adjust = 0;
for (parm_idx = 0, arg_idx = 0; parm_idx < nparms; parm_idx++, arg_idx++)
{
tree arg;
tree parm;
parm = TREE_VEC_ELT (parms, parm_idx);
if (parm == error_mark_node)
{
TREE_VEC_ELT (new_inner_args, arg_idx) = error_mark_node;
continue;
}
if (arg_idx < nargs)
arg = TREE_VEC_ELT (inner_args, arg_idx);
else
arg = NULL_TREE;
if (template_parameter_pack_p (TREE_VALUE (parm))
&& !(arg && ARGUMENT_PACK_P (arg)))
{
arg = coerce_template_parameter_pack (parms, parm_idx, args, 
inner_args, arg_idx,
new_args, &lost,
in_decl, complain);
if (arg == NULL_TREE)
{
new_inner_args = orig_inner_args;
arg_idx = nargs;
break;
}
TREE_VEC_ELT (new_inner_args, parm_idx) = arg;
if (arg == error_mark_node)
{
lost++;
arg_idx = nargs;
break;
}
else
{
pack_adjust = TREE_VEC_LENGTH (ARGUMENT_PACK_ARGS (arg)) - 1;
arg_idx += pack_adjust;
if (fixed_parameter_pack_p (TREE_VALUE (parm)))
{
++fixed_packs;
fixed_pack_adjust += pack_adjust;
}
}
continue;
}
else if (arg)
{
if (PACK_EXPANSION_P (arg))
{
tree pattern = PACK_EXPANSION_PATTERN (arg);
tree conv = convert_template_argument (TREE_VALUE (parm),
pattern, new_args,
complain, parm_idx,
in_decl);
if (conv == error_mark_node)
{
if (complain & tf_error)
inform (input_location, "so any instantiation with a "
"non-empty parameter pack would be ill-formed");
++lost;
}
else if (TYPE_P (conv) && !TYPE_P (pattern))
TREE_VEC_ELT (inner_args, arg_idx)
= make_pack_expansion (conv, complain);
new_inner_args = inner_args;
arg_idx = nargs;
break;
}
}
else if (require_all_args)
{
arg = tsubst_template_arg (TREE_PURPOSE (parm), new_args,
complain, in_decl);
if (!NON_DEFAULT_TEMPLATE_ARGS_COUNT (new_inner_args))
SET_NON_DEFAULT_TEMPLATE_ARGS_COUNT (new_inner_args,
arg_idx - pack_adjust);
}
else
break;
if (arg == error_mark_node)
{
if (complain & tf_error)
error ("template argument %d is invalid", arg_idx + 1);
}
else if (!arg)
{
++lost;
if (arg_idx >= nargs)
++missing;
}
else
arg = convert_template_argument (TREE_VALUE (parm),
arg, new_args, complain, 
parm_idx, in_decl);
if (arg == error_mark_node)
lost++;
TREE_VEC_ELT (new_inner_args, arg_idx - pack_adjust) = arg;
}
cp_unevaluated_operand = saved_unevaluated_operand;
c_inhibit_evaluation_warnings = saved_inhibit_evaluation_warnings;
if (missing || arg_idx < nargs - variadic_args_p)
{
nparms += fixed_pack_adjust;
variadic_p -= fixed_packs;
goto bad_nargs;
}
if (arg_idx < nargs)
{
int len = nparms + (nargs - arg_idx);
tree args = make_tree_vec (len);
int i = 0;
for (; i < nparms; ++i)
TREE_VEC_ELT (args, i) = TREE_VEC_ELT (new_inner_args, i);
for (; i < len; ++i, ++arg_idx)
TREE_VEC_ELT (args, i) = TREE_VEC_ELT (inner_args,
arg_idx - pack_adjust);
new_inner_args = args;
}
if (lost)
{
gcc_assert (!(complain & tf_error) || seen_error ());
return error_mark_node;
}
if (CHECKING_P && !NON_DEFAULT_TEMPLATE_ARGS_COUNT (new_inner_args))
SET_NON_DEFAULT_TEMPLATE_ARGS_COUNT (new_inner_args,
TREE_VEC_LENGTH (new_inner_args));
return new_inner_args;
}
tree
coerce_template_parms (tree parms, tree args, tree in_decl)
{
return coerce_template_parms (parms, args, in_decl, tf_none, true, true);
}
tree
coerce_template_parms (tree parms, tree args, tree in_decl,
tsubst_flags_t complain)
{
return coerce_template_parms (parms, args, in_decl, complain, true, true);
}
static tree
coerce_innermost_template_parms (tree parms,
tree args,
tree in_decl,
tsubst_flags_t complain,
bool require_all_args,
bool use_default_args)
{
int parms_depth = TMPL_PARMS_DEPTH (parms);
int args_depth = TMPL_ARGS_DEPTH (args);
tree coerced_args;
if (parms_depth > 1)
{
coerced_args = make_tree_vec (parms_depth);
tree level;
int cur_depth;
for (level = parms, cur_depth = parms_depth;
parms_depth > 0 && level != NULL_TREE;
level = TREE_CHAIN (level), --cur_depth)
{
tree l;
if (cur_depth == args_depth)
l = coerce_template_parms (TREE_VALUE (level),
args, in_decl, complain,
require_all_args,
use_default_args);
else
l = TMPL_ARGS_LEVEL (args, cur_depth);
if (l == error_mark_node)
return error_mark_node;
SET_TMPL_ARGS_LEVEL (coerced_args, cur_depth, l);
}
}
else
coerced_args = coerce_template_parms (INNERMOST_TEMPLATE_PARMS (parms),
args, in_decl, complain,
require_all_args,
use_default_args);
return coerced_args;
}
int
template_args_equal (tree ot, tree nt, bool partial_order )
{
if (nt == ot)
return 1;
if (nt == NULL_TREE || ot == NULL_TREE)
return false;
if (nt == any_targ_node || ot == any_targ_node)
return true;
if (TREE_CODE (nt) == TREE_VEC)
return TREE_CODE (ot) == TREE_VEC && comp_template_args (ot, nt);
else if (PACK_EXPANSION_P (ot))
return (PACK_EXPANSION_P (nt)
&& template_args_equal (PACK_EXPANSION_PATTERN (ot),
PACK_EXPANSION_PATTERN (nt))
&& template_args_equal (PACK_EXPANSION_EXTRA_ARGS (ot),
PACK_EXPANSION_EXTRA_ARGS (nt)));
else if (ARGUMENT_PACK_P (ot))
{
int i, len;
tree opack, npack;
if (!ARGUMENT_PACK_P (nt))
return 0;
opack = ARGUMENT_PACK_ARGS (ot);
npack = ARGUMENT_PACK_ARGS (nt);
len = TREE_VEC_LENGTH (opack);
if (TREE_VEC_LENGTH (npack) != len)
return 0;
for (i = 0; i < len; ++i)
if (!template_args_equal (TREE_VEC_ELT (opack, i),
TREE_VEC_ELT (npack, i)))
return 0;
return 1;
}
else if (ot && TREE_CODE (ot) == ARGUMENT_PACK_SELECT)
gcc_unreachable ();
else if (TYPE_P (nt))
{
if (!TYPE_P (ot))
return false;
if (!partial_order
&& (TYPE_ALIAS_P (nt) || TYPE_ALIAS_P (ot)))
return false;
else
return same_type_p (ot, nt);
}
else if (TREE_CODE (ot) == TREE_VEC || TYPE_P (ot))
return 0;
else
{
for (enum tree_code code1 = TREE_CODE (ot);
CONVERT_EXPR_CODE_P (code1)
|| code1 == NON_LVALUE_EXPR;
code1 = TREE_CODE (ot))
ot = TREE_OPERAND (ot, 0);
for (enum tree_code code2 = TREE_CODE (nt);
CONVERT_EXPR_CODE_P (code2)
|| code2 == NON_LVALUE_EXPR;
code2 = TREE_CODE (nt))
nt = TREE_OPERAND (nt, 0);
return cp_tree_equal (ot, nt);
}
}
int
comp_template_args (tree oldargs, tree newargs,
tree *oldarg_ptr, tree *newarg_ptr,
bool partial_order)
{
int i;
if (oldargs == newargs)
return 1;
if (!oldargs || !newargs)
return 0;
if (TREE_VEC_LENGTH (oldargs) != TREE_VEC_LENGTH (newargs))
return 0;
for (i = 0; i < TREE_VEC_LENGTH (oldargs); ++i)
{
tree nt = TREE_VEC_ELT (newargs, i);
tree ot = TREE_VEC_ELT (oldargs, i);
if (! template_args_equal (ot, nt, partial_order))
{
if (oldarg_ptr != NULL)
*oldarg_ptr = ot;
if (newarg_ptr != NULL)
*newarg_ptr = nt;
return 0;
}
}
return 1;
}
inline bool
comp_template_args_porder (tree oargs, tree nargs)
{
return comp_template_args (oargs, nargs, NULL, NULL, true);
}
template <typename T>
class freelist
{
static T *&next (T *obj) { return obj->next; }
static T *anew () { return ggc_alloc<T> (); }
static void poison (T *obj) {
T *p ATTRIBUTE_UNUSED = obj;
T **q ATTRIBUTE_UNUSED = &next (obj);
#ifdef ENABLE_GC_CHECKING
VALGRIND_DISCARD (VALGRIND_MAKE_MEM_UNDEFINED (p, sizeof (*p)));
memset (p, 0xa5, sizeof (*p));
#endif
VALGRIND_DISCARD (VALGRIND_MAKE_MEM_NOACCESS (p, sizeof (*p)));
VALGRIND_DISCARD (VALGRIND_MAKE_MEM_UNDEFINED (q, sizeof (*q)));
}
static void reinit (T *obj) {
T **q ATTRIBUTE_UNUSED = &next (obj);
#ifdef ENABLE_GC_CHECKING
memset (q, 0xa5, sizeof (*q));
#endif
VALGRIND_DISCARD (VALGRIND_MAKE_MEM_UNDEFINED (obj, sizeof (*obj)));
}
T *&head;
public:
freelist (T *&head) : head(head) {}
void free (T *obj)
{
poison (obj);
next (obj) = head;
head = obj;
}
T *alloc ()
{
if (head)
{
T *obj = head;
head = next (head);
reinit (obj);
return obj;
}
else
return anew ();
}
};
template <>
inline tree &
freelist<tree_node>::next (tree obj)
{
return TREE_CHAIN (obj);
}
template <>
inline tree
freelist<tree_node>::anew ()
{
return build_tree_list (NULL, NULL);
}
template <>
inline void
freelist<tree_node>::poison (tree obj ATTRIBUTE_UNUSED)
{
int size ATTRIBUTE_UNUSED = sizeof (tree_list);
tree p ATTRIBUTE_UNUSED = obj;
tree_base *b ATTRIBUTE_UNUSED = &obj->base;
tree *q ATTRIBUTE_UNUSED = &next (obj);
#ifdef ENABLE_GC_CHECKING
gcc_checking_assert (TREE_CODE (obj) == TREE_LIST);
VALGRIND_DISCARD (VALGRIND_MAKE_MEM_UNDEFINED (p, size));
memset (p, 0xa5, size);
#endif
VALGRIND_DISCARD (VALGRIND_MAKE_MEM_NOACCESS (p, size));
VALGRIND_DISCARD (VALGRIND_MAKE_MEM_DEFINED (b, sizeof (*b)));
VALGRIND_DISCARD (VALGRIND_MAKE_MEM_UNDEFINED (q, sizeof (*q)));
#ifdef ENABLE_GC_CHECKING
VALGRIND_DISCARD (VALGRIND_MAKE_MEM_UNDEFINED (b, sizeof (*b)));
TREE_SET_CODE (obj, TREE_LIST);
#else
VALGRIND_DISCARD (VALGRIND_MAKE_MEM_DEFINED (b, sizeof (*b)));
#endif
}
template <>
inline void
freelist<tree_node>::reinit (tree obj ATTRIBUTE_UNUSED)
{
tree_base *b ATTRIBUTE_UNUSED = &obj->base;
#ifdef ENABLE_GC_CHECKING
gcc_checking_assert (TREE_CODE (obj) == TREE_LIST);
VALGRIND_DISCARD (VALGRIND_MAKE_MEM_UNDEFINED (obj, sizeof (tree_list)));
memset (obj, 0, sizeof (tree_list));
#endif
VALGRIND_DISCARD (VALGRIND_MAKE_MEM_UNDEFINED (obj, sizeof (tree_list)));
#ifdef ENABLE_GC_CHECKING
TREE_SET_CODE (obj, TREE_LIST);
#else
VALGRIND_DISCARD (VALGRIND_MAKE_MEM_DEFINED (b, sizeof (*b)));
#endif
}
static GTY((deletable)) tree tree_list_freelist_head;
static inline freelist<tree_node>
tree_list_freelist ()
{
return tree_list_freelist_head;
}
static GTY((deletable)) tinst_level *tinst_level_freelist_head;
static inline freelist<tinst_level>
tinst_level_freelist ()
{
return tinst_level_freelist_head;
}
static GTY((deletable)) pending_template *pending_template_freelist_head;
static inline freelist<pending_template>
pending_template_freelist ()
{
return pending_template_freelist_head;
}
tree
tinst_level::to_list ()
{
gcc_assert (split_list_p ());
tree ret = tree_list_freelist ().alloc ();
TREE_PURPOSE (ret) = tldcl;
TREE_VALUE (ret) = targs;
tldcl = ret;
targs = NULL;
gcc_assert (tree_list_p ());
return ret;
}
const unsigned short tinst_level::refcount_infinity;
static tinst_level *
inc_refcount_use (tinst_level *obj)
{
if (obj && obj->refcount != tinst_level::refcount_infinity)
++obj->refcount;
return obj;
}
void
tinst_level::free (tinst_level *obj)
{
if (obj->tree_list_p ())
tree_list_freelist ().free (obj->get_node ());
tinst_level_freelist ().free (obj);
}
static void
dec_refcount_use (tinst_level *obj)
{
while (obj
&& obj->refcount != tinst_level::refcount_infinity
&& !--obj->refcount)
{
tinst_level *next = obj->next;
tinst_level::free (obj);
obj = next;
}
}
template <typename T>
static void
set_refcount_ptr (T *& ptr, T *obj = NULL)
{
T *save = ptr;
ptr = inc_refcount_use (obj);
dec_refcount_use (save);
}
static void
add_pending_template (tree d)
{
tree ti = (TYPE_P (d)
? CLASSTYPE_TEMPLATE_INFO (d)
: DECL_TEMPLATE_INFO (d));
struct pending_template *pt;
int level;
if (TI_PENDING_TEMPLATE_FLAG (ti))
return;
gcc_assert (TREE_CODE (d) != TREE_LIST);
level = !current_tinst_level
|| current_tinst_level->maybe_get_node () != d;
if (level)
push_tinst_level (d);
pt = pending_template_freelist ().alloc ();
pt->next = NULL;
pt->tinst = NULL;
set_refcount_ptr (pt->tinst, current_tinst_level);
if (last_pending_template)
last_pending_template->next = pt;
else
pending_templates = pt;
last_pending_template = pt;
TI_PENDING_TEMPLATE_FLAG (ti) = 1;
if (level)
pop_tinst_level ();
}
tree
lookup_template_function (tree fns, tree arglist)
{
tree type;
if (fns == error_mark_node || arglist == error_mark_node)
return error_mark_node;
gcc_assert (!arglist || TREE_CODE (arglist) == TREE_VEC);
if (!is_overloaded_fn (fns) && !identifier_p (fns))
{
error ("%q#D is not a function template", fns);
return error_mark_node;
}
if (BASELINK_P (fns))
{
BASELINK_FUNCTIONS (fns) = build2 (TEMPLATE_ID_EXPR,
unknown_type_node,
BASELINK_FUNCTIONS (fns),
arglist);
return fns;
}
type = TREE_TYPE (fns);
if (TREE_CODE (fns) == OVERLOAD || !type)
type = unknown_type_node;
return build2 (TEMPLATE_ID_EXPR, type, fns, arglist);
}
tree
maybe_get_template_decl_from_type_decl (tree decl)
{
if (decl == NULL_TREE)
return decl;
if (TREE_CODE (decl) == TREE_LIST)
{
tree t, tmpl = NULL_TREE;
for (t = decl; t; t = TREE_CHAIN (t))
{
tree elt = maybe_get_template_decl_from_type_decl (TREE_VALUE (t));
if (!tmpl)
tmpl = elt;
else if (tmpl != elt)
break;
}
if (tmpl && t == NULL_TREE)
return tmpl;
else
return decl;
}
return (decl != NULL_TREE
&& DECL_SELF_REFERENCE_P (decl)
&& CLASSTYPE_TEMPLATE_INFO (TREE_TYPE (decl)))
? CLASSTYPE_TI_TEMPLATE (TREE_TYPE (decl)) : decl;
}
static tree
lookup_template_class_1 (tree d1, tree arglist, tree in_decl, tree context,
int entering_scope, tsubst_flags_t complain)
{
tree templ = NULL_TREE, parmlist;
tree t;
spec_entry **slot;
spec_entry *entry;
spec_entry elt;
hashval_t hash;
if (identifier_p (d1))
{
tree value = innermost_non_namespace_value (d1);
if (value && DECL_TEMPLATE_TEMPLATE_PARM_P (value))
templ = value;
else
{
if (context)
push_decl_namespace (context);
templ = lookup_name (d1);
templ = maybe_get_template_decl_from_type_decl (templ);
if (context)
pop_decl_namespace ();
}
if (templ)
context = DECL_CONTEXT (templ);
}
else if (TREE_CODE (d1) == TYPE_DECL && MAYBE_CLASS_TYPE_P (TREE_TYPE (d1)))
{
tree type = TREE_TYPE (d1);
if (TREE_CODE (type) == TYPENAME_TYPE && TREE_TYPE (type))
type = TREE_TYPE (type);
if (CLASSTYPE_TEMPLATE_INFO (type))
{
templ = CLASSTYPE_TI_TEMPLATE (type);
d1 = DECL_NAME (templ);
}
}
else if (TREE_CODE (d1) == ENUMERAL_TYPE
|| (TYPE_P (d1) && MAYBE_CLASS_TYPE_P (d1)))
{
templ = TYPE_TI_TEMPLATE (d1);
d1 = DECL_NAME (templ);
}
else if (DECL_TYPE_TEMPLATE_P (d1))
{
templ = d1;
d1 = DECL_NAME (templ);
context = DECL_CONTEXT (templ);
}
else if (DECL_TEMPLATE_TEMPLATE_PARM_P (d1))
{
templ = d1;
d1 = DECL_NAME (templ);
}
if (! templ)
{
if (complain & tf_error)
error ("%qT is not a template", d1);
return error_mark_node;
}
if (TREE_CODE (templ) != TEMPLATE_DECL
|| ((complain & tf_user) && !DECL_TEMPLATE_PARM_P (templ)
&& !PRIMARY_TEMPLATE_P (templ)))
{
if (complain & tf_error)
{
error ("non-template type %qT used as a template", d1);
if (in_decl)
error ("for template declaration %q+D", in_decl);
}
return error_mark_node;
}
complain &= ~tf_user;
if (pack_expansion_args_count (INNERMOST_TEMPLATE_ARGS (arglist)))
templ = get_underlying_template (templ);
if (DECL_TEMPLATE_TEMPLATE_PARM_P (templ))
{
tree parm;
tree arglist2 = coerce_template_args_for_ttp (templ, arglist, complain);
if (arglist2 == error_mark_node
|| (!uses_template_parms (arglist2)
&& check_instantiated_args (templ, arglist2, complain)))
return error_mark_node;
parm = bind_template_template_parm (TREE_TYPE (templ), arglist2);
return parm;
}
else
{
tree template_type = TREE_TYPE (templ);
tree gen_tmpl;
tree type_decl;
tree found = NULL_TREE;
int arg_depth;
int parm_depth;
int is_dependent_type;
int use_partial_inst_tmpl = false;
if (template_type == error_mark_node)
return error_mark_node;
gen_tmpl = most_general_template (templ);
parmlist = DECL_TEMPLATE_PARMS (gen_tmpl);
parm_depth = TMPL_PARMS_DEPTH (parmlist);
arg_depth = TMPL_ARGS_DEPTH (arglist);
if (arg_depth == 1 && parm_depth > 1)
{
tree ti = TYPE_TEMPLATE_INFO_MAYBE_ALIAS (TREE_TYPE (templ));
arglist = add_outermost_template_args (TI_ARGS (ti), arglist);
arg_depth = TMPL_ARGS_DEPTH (arglist);
}
gcc_assert (parm_depth == arg_depth);
arglist = coerce_innermost_template_parms (parmlist, arglist, gen_tmpl,
complain,
true,
true);
if (arglist == error_mark_node)
return error_mark_node;
if (entering_scope
|| !PRIMARY_TEMPLATE_P (gen_tmpl)
|| currently_open_class (template_type))
{
tree tinfo = TYPE_TEMPLATE_INFO (template_type);
if (tinfo && comp_template_args (TI_ARGS (tinfo), arglist))
return template_type;
}
elt.tmpl = gen_tmpl;
elt.args = arglist;
elt.spec = NULL_TREE;
hash = spec_hasher::hash (&elt);
entry = type_specializations->find_with_hash (&elt, hash);
if (entry)
return entry->spec;
if (flag_concepts && !constraints_satisfied_p (gen_tmpl, arglist))
{
if (complain & tf_error)
{
error ("template constraint failure");
diagnose_constraints (input_location, gen_tmpl, arglist);
}
return error_mark_node;
}
is_dependent_type = uses_template_parms (arglist);
if (!is_dependent_type
&& check_instantiated_args (gen_tmpl,
INNERMOST_TEMPLATE_ARGS (arglist),
complain))
return error_mark_node;
if (!is_dependent_type
&& !PRIMARY_TEMPLATE_P (gen_tmpl)
&& !LAMBDA_TYPE_P (TREE_TYPE (gen_tmpl))
&& TREE_CODE (CP_DECL_CONTEXT (gen_tmpl)) == NAMESPACE_DECL)
{
found = xref_tag_from_type (TREE_TYPE (gen_tmpl),
DECL_NAME (gen_tmpl),
ts_global);
return found;
}
context = DECL_CONTEXT (gen_tmpl);
if (context && TYPE_P (context))
{
context = tsubst_aggr_type (context, arglist, complain, in_decl, true);
context = complete_type (context);
}
else
context = tsubst (context, arglist, complain, in_decl);
if (context == error_mark_node)
return error_mark_node;
if (!context)
context = global_namespace;
if (DECL_ALIAS_TEMPLATE_P (gen_tmpl))
{
t = tsubst (TREE_TYPE (gen_tmpl), arglist, complain, in_decl);
if (t == error_mark_node)
return t;
}
else if (TREE_CODE (template_type) == ENUMERAL_TYPE)
{
if (!is_dependent_type)
{
set_current_access_from_decl (TYPE_NAME (template_type));
t = start_enum (TYPE_IDENTIFIER (template_type), NULL_TREE,
tsubst (ENUM_UNDERLYING_TYPE (template_type),
arglist, complain, in_decl),
tsubst_attributes (TYPE_ATTRIBUTES (template_type),
arglist, complain, in_decl),
SCOPED_ENUM_P (template_type), NULL);
if (t == error_mark_node)
return t;
}
else
{
t = cxx_make_type (ENUMERAL_TYPE);
SET_SCOPED_ENUM_P (t, SCOPED_ENUM_P (template_type));
}
SET_OPAQUE_ENUM_P (t, OPAQUE_ENUM_P (template_type));
ENUM_FIXED_UNDERLYING_TYPE_P (t)
= ENUM_FIXED_UNDERLYING_TYPE_P (template_type);
}
else if (CLASS_TYPE_P (template_type))
{
gcc_assert (!LAMBDA_TYPE_P (template_type));
t = make_class_type (TREE_CODE (template_type));
CLASSTYPE_DECLARED_CLASS (t)
= CLASSTYPE_DECLARED_CLASS (template_type);
SET_CLASSTYPE_IMPLICIT_INSTANTIATION (t);
if (context == current_function_decl)
if (pushtag (DECL_NAME (gen_tmpl), t, ts_current)
== error_mark_node)
return error_mark_node;
if (comp_template_args (CLASSTYPE_TI_ARGS (template_type), arglist))
TYPE_CANONICAL (t) = template_type;
else if (any_template_arguments_need_structural_equality_p (arglist))
SET_TYPE_STRUCTURAL_EQUALITY (t);
}
else
gcc_unreachable ();
if (!TYPE_NAME (t))
{
TYPE_CONTEXT (t) = FROB_CONTEXT (context);
type_decl = create_implicit_typedef (DECL_NAME (gen_tmpl), t);
DECL_CONTEXT (type_decl) = TYPE_CONTEXT (t);
DECL_SOURCE_LOCATION (type_decl)
= DECL_SOURCE_LOCATION (TYPE_STUB_DECL (template_type));
}
else
type_decl = TYPE_NAME (t);
if (CLASS_TYPE_P (template_type))
{
TREE_PRIVATE (type_decl)
= TREE_PRIVATE (TYPE_MAIN_DECL (template_type));
TREE_PROTECTED (type_decl)
= TREE_PROTECTED (TYPE_MAIN_DECL (template_type));
if (CLASSTYPE_VISIBILITY_SPECIFIED (template_type))
{
DECL_VISIBILITY_SPECIFIED (type_decl) = 1;
DECL_VISIBILITY (type_decl) = CLASSTYPE_VISIBILITY (template_type);
}
}
if (OVERLOAD_TYPE_P (t)
&& !DECL_ALIAS_TEMPLATE_P (gen_tmpl))
{
static const char *tags[] = {"abi_tag", "may_alias"};
for (unsigned ix = 0; ix != 2; ix++)
{
tree attributes
= lookup_attribute (tags[ix], TYPE_ATTRIBUTES (template_type));
if (attributes)
TYPE_ATTRIBUTES (t)
= tree_cons (TREE_PURPOSE (attributes),
TREE_VALUE (attributes),
TYPE_ATTRIBUTES (t));
}
}
if (PRIMARY_TEMPLATE_P (gen_tmpl)
&& TMPL_ARGS_HAVE_MULTIPLE_LEVELS (arglist)
&& CLASS_TYPE_P (context)
&& !same_type_p (context, DECL_CONTEXT (gen_tmpl)))
{
TREE_VEC_LENGTH (arglist)--;
++processing_template_decl;
tree tinfo = TYPE_TEMPLATE_INFO_MAYBE_ALIAS (TREE_TYPE (gen_tmpl));
tree partial_inst_args =
tsubst (INNERMOST_TEMPLATE_ARGS (TI_ARGS (tinfo)),
arglist, complain, NULL_TREE);
--processing_template_decl;
TREE_VEC_LENGTH (arglist)++;
if (partial_inst_args == error_mark_node)
return error_mark_node;
use_partial_inst_tmpl =
!comp_template_args (INNERMOST_TEMPLATE_ARGS (arglist),
partial_inst_args);
}
if (!use_partial_inst_tmpl)
found = gen_tmpl;
else
{
TREE_VEC_LENGTH (arglist)--;
found = tsubst (gen_tmpl, arglist, complain, NULL_TREE);
TREE_VEC_LENGTH (arglist)++;
found = (TREE_CODE (found) == TEMPLATE_DECL
? found
: (TREE_CODE (found) == TYPE_DECL
? DECL_TI_TEMPLATE (found)
: CLASSTYPE_TI_TEMPLATE (found)));
}
SET_TYPE_TEMPLATE_INFO (t, build_template_info (found, arglist));
elt.spec = t;
slot = type_specializations->find_slot_with_hash (&elt, hash, INSERT);
entry = ggc_alloc<spec_entry> ();
*entry = elt;
*slot = entry;
DECL_TEMPLATE_INSTANTIATIONS (found)
= tree_cons (arglist, t,
DECL_TEMPLATE_INSTANTIATIONS (found));
if (TREE_CODE (template_type) == ENUMERAL_TYPE && !is_dependent_type
&& !DECL_ALIAS_TEMPLATE_P (gen_tmpl))
tsubst_enum (template_type, t, arglist);
if (CLASS_TYPE_P (template_type) && is_dependent_type)
DECL_IGNORED_P (TYPE_MAIN_DECL (t)) = 1;
TREE_PUBLIC (type_decl) = 1;
determine_visibility (type_decl);
inherit_targ_abi_tags (t);
return t;
}
}
tree
lookup_template_class (tree d1, tree arglist, tree in_decl, tree context,
int entering_scope, tsubst_flags_t complain)
{
tree ret;
timevar_push (TV_TEMPLATE_INST);
ret = lookup_template_class_1 (d1, arglist, in_decl, context,
entering_scope, complain);
timevar_pop (TV_TEMPLATE_INST);
return ret;
}
tree
lookup_template_variable (tree templ, tree arglist)
{
tree type = NULL_TREE;
if (flag_concepts && variable_concept_p (templ))
type = boolean_type_node;
return build2 (TEMPLATE_ID_EXPR, type, templ, arglist);
}
tree
finish_template_variable (tree var, tsubst_flags_t complain)
{
tree templ = TREE_OPERAND (var, 0);
tree arglist = TREE_OPERAND (var, 1);
bool concept_p = flag_concepts && variable_concept_p (templ);
if (concept_p && processing_template_decl)
return var;
tree tmpl_args = DECL_TI_ARGS (DECL_TEMPLATE_RESULT (templ));
arglist = add_outermost_template_args (tmpl_args, arglist);
templ = most_general_template (templ);
tree parms = DECL_TEMPLATE_PARMS (templ);
arglist = coerce_innermost_template_parms (parms, arglist, templ, complain,
true,
true);
if (flag_concepts && !constraints_satisfied_p (templ, arglist))
{
if (complain & tf_error)
{
error ("use of invalid variable template %qE", var);
diagnose_constraints (location_of (var), templ, arglist);
}
return error_mark_node;
}
if (concept_p)
{
tree decl = DECL_TEMPLATE_RESULT (templ);
return evaluate_variable_concept (decl, arglist);
}
return instantiate_template (templ, arglist, complain);
}
tree
lookup_and_finish_template_variable (tree templ, tree targs,
tsubst_flags_t complain)
{
templ = lookup_template_variable (templ, targs);
if (!any_dependent_template_arguments_p (targs))
{
templ = finish_template_variable (templ, complain);
mark_used (templ);
}
return convert_from_reference (templ);
}

struct pair_fn_data
{
tree_fn_t fn;
tree_fn_t any_fn;
void *data;
bool include_nondeduced_p;
hash_set<tree> *visited;
};
static tree
for_each_template_parm_r (tree *tp, int *walk_subtrees, void *d)
{
tree t = *tp;
struct pair_fn_data *pfd = (struct pair_fn_data *) d;
tree_fn_t fn = pfd->fn;
void *data = pfd->data;
tree result = NULL_TREE;
#define WALK_SUBTREE(NODE)						\
do									\
{									\
result = for_each_template_parm (NODE, fn, data, pfd->visited,	\
pfd->include_nondeduced_p,	\
pfd->any_fn);			\
if (result) goto out;						\
}									\
while (0)
if (pfd->any_fn && (*pfd->any_fn)(t, data))
return t;
if (TYPE_P (t)
&& (pfd->include_nondeduced_p || TREE_CODE (t) != TYPENAME_TYPE))
WALK_SUBTREE (TYPE_CONTEXT (t));
switch (TREE_CODE (t))
{
case RECORD_TYPE:
if (TYPE_PTRMEMFUNC_P (t))
break;
case UNION_TYPE:
case ENUMERAL_TYPE:
if (!TYPE_TEMPLATE_INFO (t))
*walk_subtrees = 0;
else
WALK_SUBTREE (TYPE_TI_ARGS (t));
break;
case INTEGER_TYPE:
WALK_SUBTREE (TYPE_MIN_VALUE (t));
WALK_SUBTREE (TYPE_MAX_VALUE (t));
break;
case METHOD_TYPE:
WALK_SUBTREE (TYPE_METHOD_BASETYPE (t));
case FUNCTION_TYPE:
WALK_SUBTREE (TREE_TYPE (t));
{
tree parm;
for (parm = TYPE_ARG_TYPES (t); parm; parm = TREE_CHAIN (parm))
WALK_SUBTREE (TREE_VALUE (parm));
*walk_subtrees = 0;
}
if (flag_noexcept_type)
{
tree spec = TYPE_RAISES_EXCEPTIONS (t);
if (spec)
WALK_SUBTREE (TREE_PURPOSE (spec));
}
break;
case TYPEOF_TYPE:
case DECLTYPE_TYPE:
case UNDERLYING_TYPE:
if (pfd->include_nondeduced_p
&& for_each_template_parm (TYPE_VALUES_RAW (t), fn, data,
pfd->visited, 
pfd->include_nondeduced_p,
pfd->any_fn))
return error_mark_node;
*walk_subtrees = false;
break;
case FUNCTION_DECL:
case VAR_DECL:
if (DECL_LANG_SPECIFIC (t) && DECL_TEMPLATE_INFO (t))
WALK_SUBTREE (DECL_TI_ARGS (t));
case PARM_DECL:
case CONST_DECL:
if (TREE_CODE (t) == CONST_DECL && DECL_TEMPLATE_PARM_P (t))
WALK_SUBTREE (DECL_INITIAL (t));
if (DECL_CONTEXT (t)
&& pfd->include_nondeduced_p)
WALK_SUBTREE (DECL_CONTEXT (t));
break;
case BOUND_TEMPLATE_TEMPLATE_PARM:
WALK_SUBTREE (TYPE_TI_ARGS (t));
case TEMPLATE_TEMPLATE_PARM:
case TEMPLATE_TYPE_PARM:
case TEMPLATE_PARM_INDEX:
if (fn && (*fn)(t, data))
return t;
else if (!fn)
return t;
break;
case TEMPLATE_DECL:
if (DECL_TEMPLATE_TEMPLATE_PARM_P (t))
WALK_SUBTREE (TREE_TYPE (t));
*walk_subtrees = 0;
break;
case TYPENAME_TYPE:
WALK_SUBTREE (TYPENAME_TYPE_FULLNAME (t));
break;
case CONSTRUCTOR:
if (TREE_TYPE (t) && TYPE_PTRMEMFUNC_P (TREE_TYPE (t))
&& pfd->include_nondeduced_p)
WALK_SUBTREE (TYPE_PTRMEMFUNC_FN_TYPE (TREE_TYPE (t)));
break;
case INDIRECT_REF:
case COMPONENT_REF:
if (!fn && !TREE_TYPE (t))
return error_mark_node;
break;
case MODOP_EXPR:
case CAST_EXPR:
case IMPLICIT_CONV_EXPR:
case REINTERPRET_CAST_EXPR:
case CONST_CAST_EXPR:
case STATIC_CAST_EXPR:
case DYNAMIC_CAST_EXPR:
case ARROW_EXPR:
case DOTSTAR_EXPR:
case TYPEID_EXPR:
case PSEUDO_DTOR_EXPR:
if (!fn)
return error_mark_node;
break;
default:
break;
}
#undef WALK_SUBTREE
out:
return result;
}
static tree
for_each_template_parm (tree t, tree_fn_t fn, void* data,
hash_set<tree> *visited,
bool include_nondeduced_p,
tree_fn_t any_fn)
{
struct pair_fn_data pfd;
tree result;
pfd.fn = fn;
pfd.any_fn = any_fn;
pfd.data = data;
pfd.include_nondeduced_p = include_nondeduced_p;
if (visited)
pfd.visited = visited;
else
pfd.visited = new hash_set<tree>;
result = cp_walk_tree (&t,
for_each_template_parm_r,
&pfd,
pfd.visited);
if (!visited)
{
delete pfd.visited;
pfd.visited = 0;
}
return result;
}
int
uses_template_parms (tree t)
{
if (t == NULL_TREE)
return false;
bool dependent_p;
int saved_processing_template_decl;
saved_processing_template_decl = processing_template_decl;
if (!saved_processing_template_decl)
processing_template_decl = 1;
if (TYPE_P (t))
dependent_p = dependent_type_p (t);
else if (TREE_CODE (t) == TREE_VEC)
dependent_p = any_dependent_template_arguments_p (t);
else if (TREE_CODE (t) == TREE_LIST)
dependent_p = (uses_template_parms (TREE_VALUE (t))
|| uses_template_parms (TREE_CHAIN (t)));
else if (TREE_CODE (t) == TYPE_DECL)
dependent_p = dependent_type_p (TREE_TYPE (t));
else if (DECL_P (t)
|| EXPR_P (t)
|| TREE_CODE (t) == TEMPLATE_PARM_INDEX
|| TREE_CODE (t) == OVERLOAD
|| BASELINK_P (t)
|| identifier_p (t)
|| TREE_CODE (t) == TRAIT_EXPR
|| TREE_CODE (t) == CONSTRUCTOR
|| CONSTANT_CLASS_P (t))
dependent_p = (type_dependent_expression_p (t)
|| value_dependent_expression_p (t));
else
{
gcc_assert (t == error_mark_node);
dependent_p = false;
}
processing_template_decl = saved_processing_template_decl;
return dependent_p;
}
bool
in_template_function (void)
{
tree fn = current_function_decl;
bool ret;
++processing_template_decl;
ret = (fn && DECL_LANG_SPECIFIC (fn)
&& DECL_TEMPLATE_INFO (fn)
&& any_dependent_template_arguments_p (DECL_TI_ARGS (fn)));
--processing_template_decl;
return ret;
}
bool
uses_template_parms_level (tree t, int level)
{
return for_each_template_parm (t, template_parm_this_level_p, &level, NULL,
true);
}
bool
uses_outer_template_parms (tree decl)
{
int depth = template_class_depth (CP_DECL_CONTEXT (decl));
if (depth == 0)
return false;
if (for_each_template_parm (TREE_TYPE (decl), template_parm_outer_level,
&depth, NULL, true))
return true;
if (PRIMARY_TEMPLATE_P (decl)
&& for_each_template_parm (INNERMOST_TEMPLATE_PARMS
(DECL_TEMPLATE_PARMS (decl)),
template_parm_outer_level,
&depth, NULL, true))
return true;
tree ci = get_constraints (decl);
if (ci)
ci = CI_ASSOCIATED_CONSTRAINTS (ci);
if (ci && for_each_template_parm (ci, template_parm_outer_level,
&depth, NULL, true))
return true;
return false;
}
static inline bool
neglectable_inst_p (tree d)
{
return (d && DECL_P (d)
&& !undeduced_auto_decl (d)
&& !(TREE_CODE (d) == FUNCTION_DECL ? DECL_DECLARED_CONSTEXPR_P (d)
: decl_maybe_constant_var_p (d)));
}
static bool
limit_bad_template_recursion (tree decl)
{
struct tinst_level *lev = current_tinst_level;
int errs = errorcount + sorrycount;
if (lev == NULL || errs == 0 || !neglectable_inst_p (decl))
return false;
for (; lev; lev = lev->next)
if (neglectable_inst_p (lev->maybe_get_node ()))
break;
return (lev && errs > lev->errors);
}
static int tinst_depth;
extern int max_tinst_depth;
int depth_reached;
static GTY(()) struct tinst_level *last_error_tinst_level;
static bool
push_tinst_level_loc (tree tldcl, tree targs, location_t loc)
{
struct tinst_level *new_level;
if (tinst_depth >= max_tinst_depth)
{
at_eof = 2;
fatal_error (input_location,
"template instantiation depth exceeds maximum of %d"
" (use -ftemplate-depth= to increase the maximum)",
max_tinst_depth);
return false;
}
if (!targs && limit_bad_template_recursion (tldcl))
return false;
if (!quiet_flag && !targs
&& TREE_CODE (tldcl) != TREE_LIST
&& TREE_CODE (tldcl) != FUNCTION_DECL)
fprintf (stderr, " %s", decl_as_string (tldcl, TFF_DECL_SPECIFIERS));
new_level = tinst_level_freelist ().alloc ();
new_level->tldcl = tldcl;
new_level->targs = targs;
new_level->locus = loc;
new_level->errors = errorcount + sorrycount;
new_level->next = NULL;
new_level->refcount = 0;
set_refcount_ptr (new_level->next, current_tinst_level);
set_refcount_ptr (current_tinst_level, new_level);
++tinst_depth;
if (GATHER_STATISTICS && (tinst_depth > depth_reached))
depth_reached = tinst_depth;
return true;
}
static bool
push_tinst_level (tree tmpl, tree args)
{
return push_tinst_level_loc (tmpl, args, input_location);
}
bool
push_tinst_level (tree d)
{
return push_tinst_level_loc (d, input_location);
}
bool
push_tinst_level_loc (tree d, location_t loc)
{
gcc_assert (TREE_CODE (d) != TREE_LIST);
return push_tinst_level_loc (d, NULL, loc);
}
void
pop_tinst_level (void)
{
input_location = current_tinst_level->locus;
set_refcount_ptr (current_tinst_level, current_tinst_level->next);
--tinst_depth;
}
static tree
reopen_tinst_level (struct tinst_level *level)
{
struct tinst_level *t;
tinst_depth = 0;
for (t = level; t; t = t->next)
++tinst_depth;
set_refcount_ptr (current_tinst_level, level);
pop_tinst_level ();
if (current_tinst_level)
current_tinst_level->errors = errorcount+sorrycount;
return level->maybe_get_node ();
}
struct tinst_level *
outermost_tinst_level (void)
{
struct tinst_level *level = current_tinst_level;
if (level)
while (level->next)
level = level->next;
return level;
}
static tree
tsubst_friend_function (tree decl, tree args)
{
tree new_friend;
if (TREE_CODE (decl) == FUNCTION_DECL
&& DECL_TEMPLATE_INSTANTIATION (decl)
&& TREE_CODE (DECL_TI_TEMPLATE (decl)) != TEMPLATE_DECL)
{
tree template_id, arglist, fns;
tree new_args;
tree tmpl;
tree ns = decl_namespace_context (TYPE_MAIN_DECL (current_class_type));
push_nested_namespace (ns);
fns = tsubst_expr (DECL_TI_TEMPLATE (decl), args,
tf_warning_or_error, NULL_TREE,
false);
pop_nested_namespace (ns);
arglist = tsubst (DECL_TI_ARGS (decl), args,
tf_warning_or_error, NULL_TREE);
template_id = lookup_template_function (fns, arglist);
new_friend = tsubst (decl, args, tf_warning_or_error, NULL_TREE);
tmpl = determine_specialization (template_id, new_friend,
&new_args,
0,
TREE_VEC_LENGTH (args),
tsk_none);
return instantiate_template (tmpl, new_args, tf_error);
}
new_friend = tsubst (decl, args, tf_warning_or_error, NULL_TREE);
if (new_friend == error_mark_node)
return error_mark_node;
DECL_USE_TEMPLATE (new_friend) = 0;
if (TREE_CODE (decl) == TEMPLATE_DECL)
{
DECL_USE_TEMPLATE (DECL_TEMPLATE_RESULT (new_friend)) = 0;
DECL_SAVED_TREE (DECL_TEMPLATE_RESULT (new_friend))
= DECL_SAVED_TREE (DECL_TEMPLATE_RESULT (decl));
}
if (TREE_CODE (new_friend) != TEMPLATE_DECL)
{
SET_DECL_RTL (new_friend, NULL);
SET_DECL_ASSEMBLER_NAME (new_friend, NULL_TREE);
}
if (DECL_NAMESPACE_SCOPE_P (new_friend))
{
tree old_decl;
tree new_friend_template_info;
tree new_friend_result_template_info;
tree ns;
int  new_friend_is_defn;
new_friend_template_info = DECL_TEMPLATE_INFO (new_friend);
new_friend_is_defn =
(DECL_INITIAL (DECL_TEMPLATE_RESULT
(template_for_substitution (new_friend)))
!= NULL_TREE);
if (TREE_CODE (new_friend) == TEMPLATE_DECL)
{
DECL_PRIMARY_TEMPLATE (new_friend) = new_friend;
new_friend_result_template_info
= DECL_TEMPLATE_INFO (DECL_TEMPLATE_RESULT (new_friend));
}
else
new_friend_result_template_info = NULL_TREE;
ns = decl_namespace_context (new_friend);
push_nested_namespace (ns);
old_decl = pushdecl_namespace_level (new_friend, true);
pop_nested_namespace (ns);
if (old_decl == error_mark_node)
return error_mark_node;
if (old_decl != new_friend)
{
if (!new_friend_is_defn)
;
else
{
tree new_template = TI_TEMPLATE (new_friend_template_info);
tree new_args = TI_ARGS (new_friend_template_info);
DECL_TEMPLATE_INFO (old_decl) = new_friend_template_info;
if (TREE_CODE (old_decl) != TEMPLATE_DECL)
{
gcc_assert (retrieve_specialization (new_template,
new_args, 0)
== old_decl);
if (DECL_ODR_USED (old_decl))
instantiate_decl (old_decl, true,
false);
}
else
{
tree t;
DECL_TEMPLATE_INFO (DECL_TEMPLATE_RESULT (old_decl))
= new_friend_result_template_info;
gcc_assert (new_template
== most_general_template (new_template));
gcc_assert (new_template != old_decl);
for (t = DECL_TEMPLATE_INSTANTIATIONS (old_decl);
t != NULL_TREE;
t = TREE_CHAIN (t))
{
tree spec = TREE_VALUE (t);
spec_entry elt;
elt.tmpl = old_decl;
elt.args = DECL_TI_ARGS (spec);
elt.spec = NULL_TREE;
decl_specializations->remove_elt (&elt);
DECL_TI_ARGS (spec)
= add_outermost_template_args (new_args,
DECL_TI_ARGS (spec));
register_specialization
(spec, new_template, DECL_TI_ARGS (spec), true, 0);
}
DECL_TEMPLATE_INSTANTIATIONS (old_decl) = NULL_TREE;
}
}
new_friend = old_decl;
}
}
else
{
tree context = DECL_CONTEXT (new_friend);
bool dependent_p;
++processing_template_decl;
dependent_p = dependent_type_p (context);
--processing_template_decl;
if (!dependent_p
&& !complete_type_or_else (context, NULL_TREE))
return error_mark_node;
if (COMPLETE_TYPE_P (context))
{
tree fn = new_friend;
if (TREE_CODE (fn) == TEMPLATE_DECL
&& !PRIMARY_TEMPLATE_P (fn))
fn = DECL_TEMPLATE_RESULT (fn);
fn = check_classfn (context, fn, NULL_TREE);
if (fn)
new_friend = fn;
}
}
return new_friend;
}
static tree
tsubst_friend_class (tree friend_tmpl, tree args)
{
tree tmpl;
if (DECL_TEMPLATE_TEMPLATE_PARM_P (friend_tmpl))
{
tmpl = tsubst (TREE_TYPE (friend_tmpl), args, tf_none, NULL_TREE);
return TREE_TYPE (tmpl);
}
tree context = CP_DECL_CONTEXT (friend_tmpl);
if (TREE_CODE (context) == NAMESPACE_DECL)
push_nested_namespace (context);
else
push_nested_class (context);
tmpl = lookup_name_real (DECL_NAME (friend_tmpl), false,
false, false,
false, LOOKUP_HIDDEN);
if (tmpl && DECL_CLASS_TEMPLATE_P (tmpl))
{
if (TMPL_PARMS_DEPTH (DECL_TEMPLATE_PARMS (friend_tmpl))
> TMPL_ARGS_DEPTH (args))
{
tree parms = tsubst_template_parms (DECL_TEMPLATE_PARMS (friend_tmpl),
args, tf_warning_or_error);
location_t saved_input_location = input_location;
input_location = DECL_SOURCE_LOCATION (friend_tmpl);
tree cons = get_constraints (tmpl);
redeclare_class_template (TREE_TYPE (tmpl), parms, cons);
input_location = saved_input_location;
}
}
else
{
tmpl = tsubst (friend_tmpl, args, tf_warning_or_error, NULL_TREE);
if (tmpl != error_mark_node)
{
DECL_USE_TEMPLATE (tmpl) = 0;
DECL_TEMPLATE_INFO (tmpl) = NULL_TREE;
CLASSTYPE_USE_TEMPLATE (TREE_TYPE (tmpl)) = 0;
CLASSTYPE_TI_ARGS (TREE_TYPE (tmpl))
= INNERMOST_TEMPLATE_ARGS (CLASSTYPE_TI_ARGS (TREE_TYPE (tmpl)));
retrofit_lang_decl (DECL_TEMPLATE_RESULT (tmpl));
DECL_ANTICIPATED (tmpl)
= DECL_ANTICIPATED (DECL_TEMPLATE_RESULT (tmpl)) = true;
tmpl = pushdecl_namespace_level (tmpl, true);
}
}
if (TREE_CODE (context) == NAMESPACE_DECL)
pop_nested_namespace (context);
else
pop_nested_class ();
return TREE_TYPE (tmpl);
}
static int
can_complete_type_without_circularity (tree type)
{
if (type == NULL_TREE || type == error_mark_node)
return 0;
else if (COMPLETE_TYPE_P (type))
return 1;
else if (TREE_CODE (type) == ARRAY_TYPE)
return can_complete_type_without_circularity (TREE_TYPE (type));
else if (CLASS_TYPE_P (type)
&& TYPE_BEING_DEFINED (TYPE_MAIN_VARIANT (type)))
return 0;
else
return 1;
}
static tree tsubst_omp_clauses (tree, enum c_omp_region_type, tree,
tsubst_flags_t, tree);
static tree
tsubst_attribute (tree t, tree *decl_p, tree args,
tsubst_flags_t complain, tree in_decl)
{
gcc_assert (ATTR_IS_DEPENDENT (t));
tree val = TREE_VALUE (t);
if (val == NULL_TREE)
;
else if ((flag_openmp || flag_openmp_simd)
&& is_attribute_p ("omp declare simd",
get_attribute_name (t)))
{
tree clauses = TREE_VALUE (val);
clauses = tsubst_omp_clauses (clauses, C_ORT_OMP_DECLARE_SIMD, args,
complain, in_decl);
c_omp_declare_simd_clauses_to_decls (*decl_p, clauses);
clauses = finish_omp_clauses (clauses, C_ORT_OMP_DECLARE_SIMD);
tree parms = DECL_ARGUMENTS (*decl_p);
clauses
= c_omp_declare_simd_clauses_to_numbers (parms, clauses);
if (clauses)
val = build_tree_list (NULL_TREE, clauses);
else
val = NULL_TREE;
}
else if (attribute_takes_identifier_p (get_attribute_name (t)))
{
tree chain
= tsubst_expr (TREE_CHAIN (val), args, complain, in_decl,
false);
if (chain != TREE_CHAIN (val))
val = tree_cons (NULL_TREE, TREE_VALUE (val), chain);
}
else if (PACK_EXPANSION_P (val))
{
tree purp = TREE_PURPOSE (t);
tree pack = tsubst_pack_expansion (val, args, complain, in_decl);
if (pack == error_mark_node)
return error_mark_node;
int len = TREE_VEC_LENGTH (pack);
tree list = NULL_TREE;
tree *q = &list;
for (int i = 0; i < len; ++i)
{
tree elt = TREE_VEC_ELT (pack, i);
*q = build_tree_list (purp, elt);
q = &TREE_CHAIN (*q);
}
return list;
}
else
val = tsubst_expr (val, args, complain, in_decl,
false);
if (val != TREE_VALUE (t))
return build_tree_list (TREE_PURPOSE (t), val);
return t;
}
static tree
tsubst_attributes (tree attributes, tree args,
tsubst_flags_t complain, tree in_decl)
{
tree last_dep = NULL_TREE;
for (tree t = attributes; t; t = TREE_CHAIN (t))
if (ATTR_IS_DEPENDENT (t))
{
last_dep = t;
attributes = copy_list (attributes);
break;
}
if (last_dep)
for (tree *p = &attributes; *p; )
{
tree t = *p;
if (ATTR_IS_DEPENDENT (t))
{
tree subst = tsubst_attribute (t, NULL, args, complain, in_decl);
if (subst != t)
{
*p = subst;
while (*p)
p = &TREE_CHAIN (*p);
*p = TREE_CHAIN (t);
continue;
}
}
p = &TREE_CHAIN (*p);
}
return attributes;
}
static void
apply_late_template_attributes (tree *decl_p, tree attributes, int attr_flags,
tree args, tsubst_flags_t complain, tree in_decl)
{
tree last_dep = NULL_TREE;
tree t;
tree *p;
if (attributes == NULL_TREE)
return;
if (DECL_P (*decl_p))
{
if (TREE_TYPE (*decl_p) == error_mark_node)
return;
p = &DECL_ATTRIBUTES (*decl_p);
gcc_assert (*p == attributes);
}
else
{
p = &TYPE_ATTRIBUTES (*decl_p);
gcc_assert (*p != attributes);
while (*p)
p = &TREE_CHAIN (*p);
}
for (t = attributes; t; t = TREE_CHAIN (t))
if (ATTR_IS_DEPENDENT (t))
{
last_dep = t;
attributes = copy_list (attributes);
break;
}
*p = attributes;
if (last_dep)
{
tree late_attrs = NULL_TREE;
tree *q = &late_attrs;
for (; *p; )
{
t = *p;
if (ATTR_IS_DEPENDENT (t))
{
*p = TREE_CHAIN (t);
TREE_CHAIN (t) = NULL_TREE;
*q = tsubst_attribute (t, decl_p, args, complain, in_decl);
while (*q)
q = &TREE_CHAIN (*q);
}
else
p = &TREE_CHAIN (t);
}
cplus_decl_attributes (decl_p, late_attrs, attr_flags);
}
}
static void
perform_typedefs_access_check (tree tmpl, tree targs)
{
location_t saved_location;
unsigned i;
qualified_typedef_usage_t *iter;
if (!tmpl
|| (!CLASS_TYPE_P (tmpl)
&& TREE_CODE (tmpl) != FUNCTION_DECL))
return;
saved_location = input_location;
FOR_EACH_VEC_SAFE_ELT (get_types_needing_access_check (tmpl), i, iter)
{
tree type_decl = iter->typedef_decl;
tree type_scope = iter->context;
if (!type_decl || !type_scope || !CLASS_TYPE_P (type_scope))
continue;
if (uses_template_parms (type_decl))
type_decl = tsubst (type_decl, targs, tf_error, NULL_TREE);
if (uses_template_parms (type_scope))
type_scope = tsubst (type_scope, targs, tf_error, NULL_TREE);
input_location = iter->locus;
perform_or_defer_access_check (TYPE_BINFO (type_scope),
type_decl, type_decl,
tf_warning_or_error);
}
input_location = saved_location;
}
static tree
instantiate_class_template_1 (tree type)
{
tree templ, args, pattern, t, member;
tree typedecl;
tree pbinfo;
tree base_list;
unsigned int saved_maximum_field_alignment;
tree fn_context;
if (type == error_mark_node)
return error_mark_node;
if (COMPLETE_OR_OPEN_TYPE_P (type)
|| uses_template_parms (type))
return type;
templ = most_general_template (CLASSTYPE_TI_TEMPLATE (type));
gcc_assert (TREE_CODE (templ) == TEMPLATE_DECL);
TYPE_BEING_DEFINED (type) = 1;
deferring_access_check_sentinel acs (dk_no_deferred);
t = most_specialized_partial_spec (type, tf_warning_or_error);
if (t == error_mark_node)
return error_mark_node;
else if (t)
{
pattern = TREE_TYPE (t);
args = TREE_PURPOSE (t);
}
else
{
pattern = TREE_TYPE (templ);
args = CLASSTYPE_TI_ARGS (type);
}
if (!COMPLETE_TYPE_P (pattern))
{
TYPE_BEING_DEFINED (type) = 0;
return type;
}
if (! push_tinst_level (type))
return type;
int saved_unevaluated_operand = cp_unevaluated_operand;
int saved_inhibit_evaluation_warnings = c_inhibit_evaluation_warnings;
fn_context = decl_function_context (TYPE_MAIN_DECL (type));
if (!fn_context && LAMBDA_TYPE_P (type) && TYPE_CLASS_SCOPE_P (type))
fn_context = error_mark_node;
if (!fn_context)
push_to_top_level ();
else
{
cp_unevaluated_operand = 0;
c_inhibit_evaluation_warnings = 0;
}
saved_maximum_field_alignment = maximum_field_alignment;
maximum_field_alignment = TYPE_PRECISION (pattern);
SET_CLASSTYPE_INTERFACE_UNKNOWN (type);
typedecl = TYPE_MAIN_DECL (pattern);
input_location = DECL_SOURCE_LOCATION (TYPE_NAME (type)) =
DECL_SOURCE_LOCATION (typedecl);
TYPE_PACKED (type) = TYPE_PACKED (pattern);
SET_TYPE_ALIGN (type, TYPE_ALIGN (pattern));
TYPE_USER_ALIGN (type) = TYPE_USER_ALIGN (pattern);
CLASSTYPE_NON_AGGREGATE (type) = CLASSTYPE_NON_AGGREGATE (pattern);
if (ANON_AGGR_TYPE_P (pattern))
SET_ANON_AGGR_TYPE_P (type);
if (CLASSTYPE_VISIBILITY_SPECIFIED (pattern))
{
CLASSTYPE_VISIBILITY_SPECIFIED (type) = 1;
CLASSTYPE_VISIBILITY (type) = CLASSTYPE_VISIBILITY (pattern);
determine_visibility (TYPE_MAIN_DECL (type));
}
if (CLASS_TYPE_P (type))
CLASSTYPE_FINAL (type) = CLASSTYPE_FINAL (pattern);
pbinfo = TYPE_BINFO (pattern);
gcc_assert (!DECL_CLASS_SCOPE_P (TYPE_MAIN_DECL (pattern))
|| COMPLETE_OR_OPEN_TYPE_P (TYPE_CONTEXT (type)));
base_list = NULL_TREE;
if (BINFO_N_BASE_BINFOS (pbinfo))
{
tree pbase_binfo;
tree pushed_scope;
int i;
pushed_scope = push_scope (CP_TYPE_CONTEXT (type));
for (i = 0; BINFO_BASE_ITERATE (pbinfo, i, pbase_binfo); i++)
{
tree base;
tree access = BINFO_BASE_ACCESS (pbinfo, i);
tree expanded_bases = NULL_TREE;
int idx, len = 1;
if (PACK_EXPANSION_P (BINFO_TYPE (pbase_binfo)))
{
expanded_bases = 
tsubst_pack_expansion (BINFO_TYPE (pbase_binfo),
args, tf_error, NULL_TREE);
if (expanded_bases == error_mark_node)
continue;
len = TREE_VEC_LENGTH (expanded_bases);
}
for (idx = 0; idx < len; idx++)
{
if (expanded_bases)
base = TREE_VEC_ELT (expanded_bases, idx);
else
base = tsubst (BINFO_TYPE (pbase_binfo), args, tf_error, 
NULL_TREE);
if (base == error_mark_node)
continue;
base_list = tree_cons (access, base, base_list);
if (BINFO_VIRTUAL_P (pbase_binfo))
TREE_TYPE (base_list) = integer_type_node;
}
}
base_list = nreverse (base_list);
if (pushed_scope)
pop_scope (pushed_scope);
}
xref_basetypes (type, base_list);
apply_late_template_attributes (&type, TYPE_ATTRIBUTES (pattern),
(int) ATTR_FLAG_TYPE_IN_PLACE,
args, tf_error, NULL_TREE);
fixup_attribute_variants (type);
push_nested_class (type);
for (member = CLASSTYPE_DECL_LIST (pattern);
member; member = TREE_CHAIN (member))
{
tree t = TREE_VALUE (member);
if (TREE_PURPOSE (member))
{
if (TYPE_P (t))
{
if (LAMBDA_TYPE_P (t))
continue;
tree newtag;
bool class_template_p;
class_template_p = (TREE_CODE (t) != ENUMERAL_TYPE
&& TYPE_LANG_SPECIFIC (t)
&& CLASSTYPE_IS_TEMPLATE (t));
if (class_template_p)
++processing_template_decl;
newtag = tsubst (t, args, tf_error, NULL_TREE);
if (class_template_p)
--processing_template_decl;
if (newtag == error_mark_node)
continue;
if (TREE_CODE (newtag) != ENUMERAL_TYPE)
{
tree name = TYPE_IDENTIFIER (t);
if (class_template_p)
CLASSTYPE_USE_TEMPLATE (newtag) = 0;
if (name)
SET_IDENTIFIER_TYPE_VALUE (name, newtag);
pushtag (name, newtag, ts_current);
}
}
else if (DECL_DECLARES_FUNCTION_P (t))
{
tree r;
if (TREE_CODE (t) == TEMPLATE_DECL)
++processing_template_decl;
r = tsubst (t, args, tf_error, NULL_TREE);
if (TREE_CODE (t) == TEMPLATE_DECL)
--processing_template_decl;
set_current_access_from_decl (r);
finish_member_declaration (r);
if (r != error_mark_node && DECL_PRESERVE_P (r))
mark_used (r);
if (TREE_CODE (r) == FUNCTION_DECL
&& DECL_OMP_DECLARE_REDUCTION_P (r))
cp_check_omp_declare_reduction (r);
}
else if ((DECL_CLASS_TEMPLATE_P (t) || DECL_IMPLICIT_TYPEDEF_P (t))
&& LAMBDA_TYPE_P (TREE_TYPE (t)))
;
else
{
if (TREE_CODE (t) == STATIC_ASSERT)
{
tree condition;
++c_inhibit_evaluation_warnings;
condition =
tsubst_expr (STATIC_ASSERT_CONDITION (t), args, 
tf_warning_or_error, NULL_TREE,
true);
--c_inhibit_evaluation_warnings;
finish_static_assert (condition,
STATIC_ASSERT_MESSAGE (t), 
STATIC_ASSERT_SOURCE_LOCATION (t),
true);
}
else if (TREE_CODE (t) != CONST_DECL)
{
tree r;
tree vec = NULL_TREE;
int len = 1;
input_location = DECL_SOURCE_LOCATION (t);
if (TREE_CODE (t) == TEMPLATE_DECL)
++processing_template_decl;
r = tsubst (t, args, tf_warning_or_error, NULL_TREE);
if (TREE_CODE (t) == TEMPLATE_DECL)
--processing_template_decl;
if (TREE_CODE (r) == TREE_VEC)
{
vec = r;
len = TREE_VEC_LENGTH (vec);
}
for (int i = 0; i < len; ++i)
{
if (vec)
r = TREE_VEC_ELT (vec, i);
if (VAR_P (r))
{
finish_static_data_member_decl
(r,
NULL_TREE,
false,
NULL_TREE,
0);
if (r != error_mark_node && DECL_PRESERVE_P (r))
mark_used (r);
}
else if (TREE_CODE (r) == FIELD_DECL)
{
tree rtype = TREE_TYPE (r);
if (can_complete_type_without_circularity (rtype))
complete_type (rtype);
if (!complete_or_array_type_p (rtype))
{
cxx_incomplete_type_error (r, rtype);
TREE_TYPE (r) = error_mark_node;
}
else if (TREE_CODE (rtype) == ARRAY_TYPE
&& TYPE_DOMAIN (rtype) == NULL_TREE
&& (TREE_CODE (type) == UNION_TYPE
|| TREE_CODE (type) == QUAL_UNION_TYPE))
{
error ("flexible array member %qD in union", r);
TREE_TYPE (r) = error_mark_node;
}
}
if (!(TREE_CODE (r) == TYPE_DECL
&& TREE_CODE (TREE_TYPE (r)) == ENUMERAL_TYPE
&& DECL_ARTIFICIAL (r)))
{
set_current_access_from_decl (r);
finish_member_declaration (r);
}
}
}
}
}
else
{
if (TYPE_P (t) || DECL_CLASS_TEMPLATE_P (t)
|| DECL_TEMPLATE_TEMPLATE_PARM_P (t))
{
tree friend_type = t;
bool adjust_processing_template_decl = false;
if (TREE_CODE (friend_type) == TEMPLATE_DECL)
{
friend_type = tsubst_friend_class (friend_type, args);
adjust_processing_template_decl = true;
}
else if (TREE_CODE (friend_type) == UNBOUND_CLASS_TEMPLATE)
{
friend_type = tsubst (friend_type, args,
tf_warning_or_error, NULL_TREE);
if (TREE_CODE (friend_type) == TEMPLATE_DECL)
friend_type = TREE_TYPE (friend_type);
adjust_processing_template_decl = true;
}
else if (TREE_CODE (friend_type) == TYPENAME_TYPE
|| TREE_CODE (friend_type) == TEMPLATE_TYPE_PARM)
{
++processing_template_decl;
friend_type = tsubst (friend_type, args,
tf_warning_or_error, NULL_TREE);
if (dependent_type_p (friend_type))
adjust_processing_template_decl = true;
--processing_template_decl;
}
else if (TREE_CODE (friend_type) != BOUND_TEMPLATE_TEMPLATE_PARM
&& !CLASSTYPE_USE_TEMPLATE (friend_type)
&& TYPE_HIDDEN_P (friend_type))
{
tree ns = decl_namespace_context (TYPE_MAIN_DECL (friend_type));
push_nested_namespace (ns);
friend_type =
xref_tag_from_type (friend_type, NULL_TREE,
ts_current);
pop_nested_namespace (ns);
}
else if (uses_template_parms (friend_type))
friend_type = tsubst (friend_type, args,
tf_warning_or_error, NULL_TREE);
if (adjust_processing_template_decl)
++processing_template_decl;
if (friend_type != error_mark_node)
make_friend_class (type, friend_type, false);
if (adjust_processing_template_decl)
--processing_template_decl;
}
else
{
tree r;
input_location = DECL_SOURCE_LOCATION (t);
if (TREE_CODE (t) == TEMPLATE_DECL)
{
++processing_template_decl;
push_deferring_access_checks (dk_no_check);
}
r = tsubst_friend_function (t, args);
add_friend (type, r, false);
if (TREE_CODE (t) == TEMPLATE_DECL)
{
pop_deferring_access_checks ();
--processing_template_decl;
}
}
}
}
if (fn_context)
{
cp_unevaluated_operand = saved_unevaluated_operand;
c_inhibit_evaluation_warnings = saved_inhibit_evaluation_warnings;
}
input_location = DECL_SOURCE_LOCATION (typedecl);
unreverse_member_declarations (type);
finish_struct_1 (type);
TYPE_BEING_DEFINED (type) = 0;
perform_typedefs_access_check (pattern, args);
perform_deferred_access_checks (tf_warning_or_error);
pop_nested_class ();
maximum_field_alignment = saved_maximum_field_alignment;
if (!fn_context)
pop_from_top_level ();
pop_tinst_level ();
if (TYPE_CONTAINS_VPTR_P (type) && CLASSTYPE_KEY_METHOD (type))
vec_safe_push (keyed_classes, type);
return type;
}
tree
instantiate_class_template (tree type)
{
tree ret;
timevar_push (TV_TEMPLATE_INST);
ret = instantiate_class_template_1 (type);
timevar_pop (TV_TEMPLATE_INST);
return ret;
}
static tree
tsubst_template_arg (tree t, tree args, tsubst_flags_t complain, tree in_decl)
{
tree r;
if (!t)
r = t;
else if (TYPE_P (t))
r = tsubst (t, args, complain, in_decl);
else
{
if (!(complain & tf_warning))
++c_inhibit_evaluation_warnings;
r = tsubst_expr (t, args, complain, in_decl,
true);
if (!(complain & tf_warning))
--c_inhibit_evaluation_warnings;
}
return r;
}
tree
extract_fnparm_pack (tree tmpl_parm, tree *spec_p)
{
tree parmvec;
tree argpack = make_node (NONTYPE_ARGUMENT_PACK);
tree spec_parm = *spec_p;
int i, len;
for (len = 0; spec_parm; ++len, spec_parm = TREE_CHAIN (spec_parm))
if (tmpl_parm
&& !function_parameter_expanded_from_pack_p (spec_parm, tmpl_parm))
break;
parmvec = make_tree_vec (len);
spec_parm = *spec_p;
for (i = 0; i < len; i++, spec_parm = DECL_CHAIN (spec_parm))
{
tree elt = spec_parm;
if (DECL_PACK_P (elt))
elt = make_pack_expansion (elt);
TREE_VEC_ELT (parmvec, i) = elt;
}
SET_ARGUMENT_PACK_ARGS (argpack, parmvec);
*spec_p = spec_parm;
return argpack;
}
static tree
make_fnparm_pack (tree spec_parm)
{
return extract_fnparm_pack (NULL_TREE, &spec_parm);
}
static int
argument_pack_element_is_expansion_p (tree arg_pack, int i)
{
tree vec = ARGUMENT_PACK_ARGS (arg_pack);
if (i >= TREE_VEC_LENGTH (vec))
return 0;
tree elt = TREE_VEC_ELT (vec, i);
if (DECL_P (elt))
elt = TREE_TYPE (elt);
if (!PACK_EXPANSION_P (elt))
return 0;
if (PACK_EXPANSION_EXTRA_ARGS (elt))
return 2;
return 1;
}
static tree
make_argument_pack_select (tree arg_pack, unsigned index)
{
tree aps = make_node (ARGUMENT_PACK_SELECT);
ARGUMENT_PACK_SELECT_FROM_PACK (aps) = arg_pack;
ARGUMENT_PACK_SELECT_INDEX (aps) = index;
return aps;
}
static bool
use_pack_expansion_extra_args_p (tree parm_packs,
int arg_pack_len,
bool has_empty_arg)
{
if (parm_packs == NULL_TREE)
return false;
else if (has_empty_arg)
return true;
bool has_expansion_arg = false;
for (int i = 0 ; i < arg_pack_len; ++i)
{
bool has_non_expansion_arg = false;
for (tree parm_pack = parm_packs;
parm_pack;
parm_pack = TREE_CHAIN (parm_pack))
{
tree arg = TREE_VALUE (parm_pack);
int exp = argument_pack_element_is_expansion_p (arg, i);
if (exp == 2)
return true;
else if (exp)
has_expansion_arg = true;
else
has_non_expansion_arg = true;
}
if (has_expansion_arg && has_non_expansion_arg)
return true;
}
return false;
}
static tree
gen_elem_of_pack_expansion_instantiation (tree pattern,
tree parm_packs,
unsigned index,
tree args ,
tsubst_flags_t complain,
tree in_decl)
{
tree t;
bool ith_elem_is_expansion = false;
for (tree pack = parm_packs; pack; pack = TREE_CHAIN (pack))
{
tree parm = TREE_PURPOSE (pack);
tree arg_pack = TREE_VALUE (pack);
tree aps;			
ith_elem_is_expansion |=
argument_pack_element_is_expansion_p (arg_pack, index);
if (TREE_CODE (parm) == PARM_DECL
|| VAR_P (parm)
|| TREE_CODE (parm) == FIELD_DECL)
{
if (index == 0)
{
aps = make_argument_pack_select (arg_pack, index);
if (!mark_used (parm, complain) && !(complain & tf_error))
return error_mark_node;
register_local_specialization (aps, parm);
}
else
aps = retrieve_local_specialization (parm);
}
else
{
int idx, level;
template_parm_level_and_index (parm, &level, &idx);
if (index == 0)
{
aps = make_argument_pack_select (arg_pack, index);
TMPL_ARG (args, level, idx) = aps;
}
else
aps = TMPL_ARG (args, level, idx);
}
ARGUMENT_PACK_SELECT_INDEX (aps) = index;
}
if (pattern == in_decl)
t = tsubst_decl (pattern, args, complain);
else if (pattern == error_mark_node)
t = error_mark_node;
else if (constraint_p (pattern))
{
if (processing_template_decl)
t = tsubst_constraint (pattern, args, complain, in_decl);
else
t = (constraints_satisfied_p (pattern, args)
? boolean_true_node : boolean_false_node);
}
else if (!TYPE_P (pattern))
t = tsubst_expr (pattern, args, complain, in_decl,
false);
else
t = tsubst (pattern, args, complain, in_decl);
if (ith_elem_is_expansion)
t = make_pack_expansion (t, complain);
return t;
}
tree
expand_empty_fold (tree t, tsubst_flags_t complain)
{
tree_code code = (tree_code)TREE_INT_CST_LOW (TREE_OPERAND (t, 0));
if (!FOLD_EXPR_MODIFY_P (t))
switch (code)
{
case TRUTH_ANDIF_EXPR:
return boolean_true_node;
case TRUTH_ORIF_EXPR:
return boolean_false_node;
case COMPOUND_EXPR:
return void_node;
default:
break;
}
if (complain & tf_error)
error_at (location_of (t),
"fold of empty expansion over %O", code);
return error_mark_node;
}
static tree
fold_expression (tree t, tree left, tree right, tsubst_flags_t complain)
{
tree op = FOLD_EXPR_OP (t);
tree_code code = (tree_code)TREE_INT_CST_LOW (op);
if (FOLD_EXPR_MODIFY_P (t))
return build_x_modify_expr (input_location, left, code, right, complain);
switch (code)
{
case COMPOUND_EXPR:
return build_x_compound_expr (input_location, left, right, complain);
case DOTSTAR_EXPR:
return build_m_component_ref (left, right, complain);
default:
return build_x_binary_op (input_location, code,
left, TREE_CODE (left),
right, TREE_CODE (right),
NULL,
complain);
}
}
static inline tree
tsubst_fold_expr_pack (tree t, tree args, tsubst_flags_t complain, tree in_decl)
{
return tsubst_pack_expansion (FOLD_EXPR_PACK (t), args, complain, in_decl);
}
static inline tree
tsubst_fold_expr_init (tree t, tree args, tsubst_flags_t complain, tree in_decl)
{
return tsubst_expr (FOLD_EXPR_INIT (t), args, complain, in_decl, false);
}
static tree
expand_left_fold (tree t, tree pack, tsubst_flags_t complain)
{
tree left = TREE_VEC_ELT (pack, 0);
for (int i = 1; i < TREE_VEC_LENGTH (pack); ++i)
{
tree right = TREE_VEC_ELT (pack, i);
left = fold_expression (t, left, right, complain);
}
return left;
}
static tree
tsubst_unary_left_fold (tree t, tree args, tsubst_flags_t complain,
tree in_decl)
{
tree pack = tsubst_fold_expr_pack (t, args, complain, in_decl);
if (pack == error_mark_node)
return error_mark_node;
if (PACK_EXPANSION_P (pack))
{
tree r = copy_node (t);
FOLD_EXPR_PACK (r) = pack;
return r;
}
if (TREE_VEC_LENGTH (pack) == 0)
return expand_empty_fold (t, complain);
else
return expand_left_fold (t, pack, complain);
}
static tree
tsubst_binary_left_fold (tree t, tree args, tsubst_flags_t complain,
tree in_decl)
{
tree pack = tsubst_fold_expr_pack (t, args, complain, in_decl);
if (pack == error_mark_node)
return error_mark_node;
tree init = tsubst_fold_expr_init (t, args, complain, in_decl);
if (init == error_mark_node)
return error_mark_node;
if (PACK_EXPANSION_P (pack))
{
tree r = copy_node (t);
FOLD_EXPR_PACK (r) = pack;
FOLD_EXPR_INIT (r) = init;
return r;
}
tree vec = make_tree_vec (TREE_VEC_LENGTH (pack) + 1);
TREE_VEC_ELT (vec, 0) = init;
for (int i = 0; i < TREE_VEC_LENGTH (pack); ++i)
TREE_VEC_ELT (vec, i + 1) = TREE_VEC_ELT (pack, i);
return expand_left_fold (t, vec, complain);
}
tree
expand_right_fold (tree t, tree pack, tsubst_flags_t complain)
{
int n = TREE_VEC_LENGTH (pack);
tree right = TREE_VEC_ELT (pack, n - 1);
for (--n; n != 0; --n)
{
tree left = TREE_VEC_ELT (pack, n - 1);
right = fold_expression (t, left, right, complain);
}
return right;
}
static tree
tsubst_unary_right_fold (tree t, tree args, tsubst_flags_t complain,
tree in_decl)
{
tree pack = tsubst_fold_expr_pack (t, args, complain, in_decl);
if (pack == error_mark_node)
return error_mark_node;
if (PACK_EXPANSION_P (pack))
{
tree r = copy_node (t);
FOLD_EXPR_PACK (r) = pack;
return r;
}
if (TREE_VEC_LENGTH (pack) == 0)
return expand_empty_fold (t, complain);
else
return expand_right_fold (t, pack, complain);
}
static tree
tsubst_binary_right_fold (tree t, tree args, tsubst_flags_t complain,
tree in_decl)
{
tree pack = tsubst_fold_expr_pack (t, args, complain, in_decl);
if (pack == error_mark_node)
return error_mark_node;
tree init = tsubst_fold_expr_init (t, args, complain, in_decl);
if (init == error_mark_node)
return error_mark_node;
if (PACK_EXPANSION_P (pack))
{
tree r = copy_node (t);
FOLD_EXPR_PACK (r) = pack;
FOLD_EXPR_INIT (r) = init;
return r;
}
int n = TREE_VEC_LENGTH (pack);
tree vec = make_tree_vec (n + 1);
for (int i = 0; i < n; ++i)
TREE_VEC_ELT (vec, i) = TREE_VEC_ELT (pack, i);
TREE_VEC_ELT (vec, n) = init;
return expand_right_fold (t, vec, complain);
}
struct el_data
{
hash_set<tree> internal;
tree extra;
tsubst_flags_t complain;
el_data (tsubst_flags_t c)
: extra (NULL_TREE), complain (c) {}
};
static tree
extract_locals_r (tree *tp, int *, void *data_)
{
el_data &data = *reinterpret_cast<el_data*>(data_);
tree *extra = &data.extra;
tsubst_flags_t complain = data.complain;
if (TYPE_P (*tp) && typedef_variant_p (*tp))
tp = &TYPE_NAME (*tp);
if (TREE_CODE (*tp) == DECL_EXPR)
data.internal.add (DECL_EXPR_DECL (*tp));
else if (tree spec = retrieve_local_specialization (*tp))
{
if (data.internal.contains (*tp))
return NULL_TREE;
if (TREE_CODE (spec) == NONTYPE_ARGUMENT_PACK)
{
tree args = ARGUMENT_PACK_ARGS (spec);
if (TREE_VEC_LENGTH (args) == 1)
{
tree elt = TREE_VEC_ELT (args, 0);
if (PACK_EXPANSION_P (elt))
elt = PACK_EXPANSION_PATTERN (elt);
if (DECL_PACK_P (elt))
spec = elt;
}
if (TREE_CODE (spec) == NONTYPE_ARGUMENT_PACK)
{
tree args = ARGUMENT_PACK_ARGS (spec);
int len = TREE_VEC_LENGTH (args);
for (int i = 0; i < len; ++i)
{
tree arg = TREE_VEC_ELT (args, i);
tree carg = arg;
if (outer_automatic_var_p (arg))
carg = process_outer_var_ref (arg, complain);
if (carg != arg)
{
if (i == 0)
{
spec = copy_node (spec);
args = copy_node (args);
SET_ARGUMENT_PACK_ARGS (spec, args);
register_local_specialization (spec, *tp);
}
TREE_VEC_ELT (args, i) = carg;
}
}
}
}
if (outer_automatic_var_p (spec))
spec = process_outer_var_ref (spec, complain);
*extra = tree_cons (*tp, spec, *extra);
}
return NULL_TREE;
}
static tree
extract_local_specs (tree pattern, tsubst_flags_t complain)
{
el_data data (complain);
cp_walk_tree_without_duplicates (&pattern, extract_locals_r, &data);
return data.extra;
}
tree
build_extra_args (tree pattern, tree args, tsubst_flags_t complain)
{
tree extra = args;
if (local_specializations)
if (tree locals = extract_local_specs (pattern, complain))
extra = tree_cons (NULL_TREE, extra, locals);
return extra;
}
tree
add_extra_args (tree extra, tree args)
{
if (extra && TREE_CODE (extra) == TREE_LIST)
{
for (tree elt = TREE_CHAIN (extra); elt; elt = TREE_CHAIN (elt))
{
tree gen = TREE_PURPOSE (elt);
tree inst = TREE_VALUE (elt);
if (DECL_P (inst))
if (tree local = retrieve_local_specialization (inst))
inst = local;
register_local_specialization (inst, gen);
}
gcc_assert (!TREE_PURPOSE (extra));
extra = TREE_VALUE (extra);
}
return add_to_template_args (extra, args);
}
tree
tsubst_pack_expansion (tree t, tree args, tsubst_flags_t complain,
tree in_decl)
{
tree pattern;
tree pack, packs = NULL_TREE;
bool unsubstituted_packs = false;
bool unsubstituted_fn_pack = false;
int i, len = -1;
tree result;
hash_map<tree, tree> *saved_local_specializations = NULL;
bool need_local_specializations = false;
int levels;
gcc_assert (PACK_EXPANSION_P (t));
pattern = PACK_EXPANSION_PATTERN (t);
args = add_extra_args (PACK_EXPANSION_EXTRA_ARGS (t), args);
levels = TMPL_ARGS_DEPTH (args);
for (pack = PACK_EXPANSION_PARAMETER_PACKS (t); pack; 
pack = TREE_CHAIN (pack))
{
tree parm_pack = TREE_VALUE (pack);
tree arg_pack = NULL_TREE;
tree orig_arg = NULL_TREE;
int level = 0;
if (TREE_CODE (parm_pack) == BASES)
{
gcc_assert (parm_pack == pattern);
if (BASES_DIRECT (parm_pack))
return calculate_direct_bases (tsubst_expr (BASES_TYPE (parm_pack),
args, complain,
in_decl, false),
complain);
else
return calculate_bases (tsubst_expr (BASES_TYPE (parm_pack),
args, complain, in_decl,
false), complain);
}
else if (builtin_pack_call_p (parm_pack))
{
gcc_assert (parm_pack == pattern);
return expand_builtin_pack_call (parm_pack, args,
complain, in_decl);
}
else if (TREE_CODE (parm_pack) == PARM_DECL)
{
if (PACK_EXPANSION_LOCAL_P (t) || CONSTRAINT_VAR_P (parm_pack))
arg_pack = retrieve_local_specialization (parm_pack);
else
need_local_specializations = true;
if (!arg_pack)
{
++cp_unevaluated_operand;
arg_pack = tsubst_decl (parm_pack, args, complain);
--cp_unevaluated_operand;
if (arg_pack && DECL_PACK_P (arg_pack))
arg_pack = NULL_TREE;
else
arg_pack = make_fnparm_pack (arg_pack);
}
else if (argument_pack_element_is_expansion_p (arg_pack, 0))
unsubstituted_fn_pack = true;
}
else if (is_normal_capture_proxy (parm_pack))
{
arg_pack = retrieve_local_specialization (parm_pack);
if (argument_pack_element_is_expansion_p (arg_pack, 0))
unsubstituted_fn_pack = true;
}
else
{
int idx;
template_parm_level_and_index (parm_pack, &level, &idx);
if (level <= levels)
arg_pack = TMPL_ARG (args, level, idx);
}
orig_arg = arg_pack;
if (arg_pack && TREE_CODE (arg_pack) == ARGUMENT_PACK_SELECT)
arg_pack = ARGUMENT_PACK_SELECT_FROM_PACK (arg_pack);
if (arg_pack && !ARGUMENT_PACK_P (arg_pack))
{
result = make_tree_vec (1);
TREE_VEC_ELT (result, 0) = error_mark_node;
return result;
}
if (arg_pack)
{
int my_len = 
TREE_VEC_LENGTH (ARGUMENT_PACK_ARGS (arg_pack));
if (ARGUMENT_PACK_INCOMPLETE_P (arg_pack))
return t;
if (len < 0)
len = my_len;
else if (len != my_len
&& !unsubstituted_fn_pack)
{
if (!(complain & tf_error))
;
else if (TREE_CODE (t) == TYPE_PACK_EXPANSION)
error ("mismatched argument pack lengths while expanding %qT",
pattern);
else
error ("mismatched argument pack lengths while expanding %qE",
pattern);
return error_mark_node;
}
packs = tree_cons (parm_pack, arg_pack, packs);
TREE_TYPE (packs) = orig_arg;
}
else
{
gcc_assert (processing_template_decl || is_auto (parm_pack));
unsubstituted_packs = true;
}
}
if (!unsubstituted_packs
&& TREE_PURPOSE (packs) == pattern)
{
tree args = ARGUMENT_PACK_ARGS (TREE_VALUE (packs));
if (TREE_VEC_LENGTH (args) == 1
&& pack_expansion_args_count (args))
return TREE_VEC_ELT (args, 0);
if (TREE_CODE (t) == TYPE_PACK_EXPANSION
|| PACK_EXPANSION_SIZEOF_P (t)
|| pack_expansion_args_count (args))
return args;
tree type = TREE_TYPE (pattern);
if (type && TREE_CODE (type) != REFERENCE_TYPE
&& !PACK_EXPANSION_P (type)
&& !WILDCARD_TYPE_P (type))
return args;
}
if (use_pack_expansion_extra_args_p (packs, len, unsubstituted_packs))
{
t = make_pack_expansion (pattern, complain);
PACK_EXPANSION_EXTRA_ARGS (t)
= build_extra_args (pattern, args, complain);
return t;
}
else if (unsubstituted_packs)
{
if (TREE_CODE (t) == EXPR_PACK_EXPANSION)
t = tsubst_expr (pattern, args, complain, in_decl,
false);
else
t = tsubst (pattern, args, complain, in_decl);
t = make_pack_expansion (t, complain);
return t;
}
gcc_assert (len >= 0);
if (need_local_specializations)
{
saved_local_specializations = local_specializations;
local_specializations = new hash_map<tree, tree>;
}
result = make_tree_vec (len);
tree elem_args = copy_template_args (args);
for (i = 0; i < len; ++i)
{
t = gen_elem_of_pack_expansion_instantiation (pattern, packs,
i,
elem_args, complain,
in_decl);
TREE_VEC_ELT (result, i) = t;
if (t == error_mark_node)
{
result = error_mark_node;
break;
}
}
for (pack = packs; pack; pack = TREE_CHAIN (pack))
{
tree parm = TREE_PURPOSE (pack);
if (TREE_CODE (parm) == PARM_DECL
|| VAR_P (parm)
|| TREE_CODE (parm) == FIELD_DECL)
register_local_specialization (TREE_TYPE (pack), parm);
else
{
int idx, level;
if (TREE_VALUE (pack) == NULL_TREE)
continue;
template_parm_level_and_index (parm, &level, &idx);
if (TMPL_ARGS_HAVE_MULTIPLE_LEVELS (args))
TREE_VEC_ELT (TREE_VEC_ELT (args, level -1 ), idx) =
TREE_TYPE (pack);
else
TREE_VEC_ELT (args, idx) = TREE_TYPE (pack);
}
}
if (need_local_specializations)
{
delete local_specializations;
local_specializations = saved_local_specializations;
}
if (len == 1 && TREE_CODE (result) == TREE_VEC
&& PACK_EXPANSION_P (TREE_VEC_ELT (result, 0)))
return TREE_VEC_ELT (result, 0);
return result;
}
tree
get_pattern_parm (tree parm, tree tmpl)
{
tree pattern = DECL_TEMPLATE_RESULT (tmpl);
tree patparm;
if (DECL_ARTIFICIAL (parm))
{
for (patparm = DECL_ARGUMENTS (pattern);
patparm; patparm = DECL_CHAIN (patparm))
if (DECL_ARTIFICIAL (patparm)
&& DECL_NAME (parm) == DECL_NAME (patparm))
break;
}
else
{
patparm = FUNCTION_FIRST_USER_PARM (DECL_TEMPLATE_RESULT (tmpl));
patparm = chain_index (DECL_PARM_INDEX (parm)-1, patparm);
gcc_assert (DECL_PARM_INDEX (patparm)
== DECL_PARM_INDEX (parm));
}
return patparm;
}
static tree
make_argument_pack (tree vec)
{
tree pack;
tree elt = TREE_VEC_ELT (vec, 0);
if (TYPE_P (elt))
pack = cxx_make_type (TYPE_ARGUMENT_PACK);
else
{
pack = make_node (NONTYPE_ARGUMENT_PACK);
TREE_CONSTANT (pack) = 1;
}
SET_ARGUMENT_PACK_ARGS (pack, vec);
return pack;
}
static tree
copy_template_args (tree t)
{
if (t == error_mark_node)
return t;
int len = TREE_VEC_LENGTH (t);
tree new_vec = make_tree_vec (len);
for (int i = 0; i < len; ++i)
{
tree elt = TREE_VEC_ELT (t, i);
if (elt && TREE_CODE (elt) == TREE_VEC)
elt = copy_template_args (elt);
TREE_VEC_ELT (new_vec, i) = elt;
}
NON_DEFAULT_TEMPLATE_ARGS_COUNT (new_vec)
= NON_DEFAULT_TEMPLATE_ARGS_COUNT (t);
return new_vec;
}
static tree
tsubst_template_args (tree t, tree args, tsubst_flags_t complain, tree in_decl)
{
tree orig_t = t;
int len, need_new = 0, i, expanded_len_adjust = 0, out;
tree *elts;
if (t == error_mark_node)
return error_mark_node;
len = TREE_VEC_LENGTH (t);
elts = XALLOCAVEC (tree, len);
for (i = 0; i < len; i++)
{
tree orig_arg = TREE_VEC_ELT (t, i);
tree new_arg;
if (TREE_CODE (orig_arg) == TREE_VEC)
new_arg = tsubst_template_args (orig_arg, args, complain, in_decl);
else if (PACK_EXPANSION_P (orig_arg))
{
new_arg = tsubst_pack_expansion (orig_arg, args, complain, in_decl);
if (TREE_CODE (new_arg) == TREE_VEC)
expanded_len_adjust += TREE_VEC_LENGTH (new_arg) - 1;
}
else if (ARGUMENT_PACK_P (orig_arg))
{
new_arg = TYPE_P (orig_arg)
? cxx_make_type (TREE_CODE (orig_arg))
: make_node (TREE_CODE (orig_arg));
tree pack_args = tsubst_template_args (ARGUMENT_PACK_ARGS (orig_arg),
args, complain, in_decl);
if (pack_args == error_mark_node)
new_arg = error_mark_node;
else
SET_ARGUMENT_PACK_ARGS (new_arg, pack_args);
if (TREE_CODE (new_arg) == NONTYPE_ARGUMENT_PACK)
TREE_CONSTANT (new_arg) = TREE_CONSTANT (orig_arg);
}
else
new_arg = tsubst_template_arg (orig_arg, args, complain, in_decl);
if (new_arg == error_mark_node)
return error_mark_node;
elts[i] = new_arg;
if (new_arg != orig_arg)
need_new = 1;
}
if (!need_new)
return t;
t = make_tree_vec (len + expanded_len_adjust);
if (NON_DEFAULT_TEMPLATE_ARGS_COUNT (orig_t))
{
int count = GET_NON_DEFAULT_TEMPLATE_ARGS_COUNT (orig_t);
count += expanded_len_adjust;
SET_NON_DEFAULT_TEMPLATE_ARGS_COUNT (t, count);
}
for (i = 0, out = 0; i < len; i++)
{
if ((PACK_EXPANSION_P (TREE_VEC_ELT (orig_t, i))
|| ARGUMENT_PACK_P (TREE_VEC_ELT (orig_t, i)))
&& TREE_CODE (elts[i]) == TREE_VEC)
{
int idx;
for (idx = 0; idx < TREE_VEC_LENGTH (elts[i]); idx++, out++)
TREE_VEC_ELT (t, out) = TREE_VEC_ELT (elts[i], idx);
}
else
{
TREE_VEC_ELT (t, out) = elts[i];
out++;
}
}
return t;
}
static tree
tsubst_template_parms_level (tree parms, tree args, tsubst_flags_t complain)
{
if (parms == error_mark_node)
return error_mark_node;
tree new_vec = make_tree_vec (TREE_VEC_LENGTH (parms));
for (int i = 0; i < TREE_VEC_LENGTH (new_vec); ++i)
{
tree tuple = TREE_VEC_ELT (parms, i);
if (tuple == error_mark_node)
continue;
TREE_VEC_ELT (new_vec, i) =
tsubst_template_parm (tuple, args, complain);
}
return new_vec;
}
static tree
tsubst_template_parms (tree parms, tree args, tsubst_flags_t complain)
{
tree r = NULL_TREE;
tree* new_parms;
++processing_template_decl;
for (new_parms = &r;
parms && TMPL_PARMS_DEPTH (parms) > TMPL_ARGS_DEPTH (args);
new_parms = &(TREE_CHAIN (*new_parms)),
parms = TREE_CHAIN (parms))
{
tree new_vec = tsubst_template_parms_level (TREE_VALUE (parms),
args, complain);
*new_parms =
tree_cons (size_int (TMPL_PARMS_DEPTH (parms)
- TMPL_ARGS_DEPTH (args)),
new_vec, NULL_TREE);
}
--processing_template_decl;
return r;
}
static tree
tsubst_template_parm (tree t, tree args, tsubst_flags_t complain)
{
tree default_value, parm_decl;
if (args == NULL_TREE
|| t == NULL_TREE
|| t == error_mark_node)
return t;
gcc_assert (TREE_CODE (t) == TREE_LIST);
default_value = TREE_PURPOSE (t);
parm_decl = TREE_VALUE (t);
parm_decl = tsubst (parm_decl, args, complain, NULL_TREE);
if (TREE_CODE (parm_decl) == PARM_DECL
&& invalid_nontype_parm_type_p (TREE_TYPE (parm_decl), complain))
parm_decl = error_mark_node;
default_value = tsubst_template_arg (default_value, args,
complain, NULL_TREE);
return build_tree_list (default_value, parm_decl);
}
static tree
tsubst_aggr_type (tree t,
tree args,
tsubst_flags_t complain,
tree in_decl,
int entering_scope)
{
if (t == NULL_TREE)
return NULL_TREE;
switch (TREE_CODE (t))
{
case RECORD_TYPE:
if (TYPE_PTRMEMFUNC_P (t))
return tsubst (TYPE_PTRMEMFUNC_FN_TYPE (t), args, complain, in_decl);
case ENUMERAL_TYPE:
case UNION_TYPE:
if (TYPE_TEMPLATE_INFO (t) && uses_template_parms (t))
{
tree argvec;
tree context;
tree r;
int saved_unevaluated_operand;
int saved_inhibit_evaluation_warnings;
saved_unevaluated_operand = cp_unevaluated_operand;
cp_unevaluated_operand = 0;
saved_inhibit_evaluation_warnings = c_inhibit_evaluation_warnings;
c_inhibit_evaluation_warnings = 0;
context = TYPE_CONTEXT (t);
if (context && TYPE_P (context))
{
context = tsubst_aggr_type (context, args, complain,
in_decl, 1);
context = complete_type (context);
}
argvec = tsubst_template_args (TYPE_TI_ARGS (t), args,
complain, in_decl);
if (argvec == error_mark_node)
r = error_mark_node;
else
{
r = lookup_template_class (t, argvec, in_decl, context,
entering_scope, complain);
r = cp_build_qualified_type_real (r, cp_type_quals (t), complain);
}
cp_unevaluated_operand = saved_unevaluated_operand;
c_inhibit_evaluation_warnings = saved_inhibit_evaluation_warnings;
return r;
}
else
return t;
default:
return tsubst (t, args, complain, in_decl);
}
}
static GTY((cache)) tree_cache_map *defarg_inst;
tree
tsubst_default_argument (tree fn, int parmnum, tree type, tree arg,
tsubst_flags_t complain)
{
tree saved_class_ptr = NULL_TREE;
tree saved_class_ref = NULL_TREE;
int errs = errorcount + sorrycount;
if (TREE_CODE (arg) == DEFAULT_ARG)
return arg;
tree parm = FUNCTION_FIRST_USER_PARM (fn);
parm = chain_index (parmnum, parm);
tree parmtype = TREE_TYPE (parm);
if (DECL_BY_REFERENCE (parm))
parmtype = TREE_TYPE (parmtype);
if (parmtype == error_mark_node)
return error_mark_node;
gcc_assert (same_type_ignoring_top_level_qualifiers_p (type, parmtype));
tree *slot;
if (defarg_inst && (slot = defarg_inst->get (parm)))
return *slot;
push_access_scope (fn);
if (cfun)
{
saved_class_ptr = current_class_ptr;
cp_function_chain->x_current_class_ptr = NULL_TREE;
saved_class_ref = current_class_ref;
cp_function_chain->x_current_class_ref = NULL_TREE;
}
start_lambda_scope (parm);
push_deferring_access_checks(dk_no_deferred);
++function_depth;
arg = tsubst_expr (arg, DECL_TI_ARGS (fn),
complain, NULL_TREE,
false);
--function_depth;
pop_deferring_access_checks();
finish_lambda_scope ();
if (cfun)
{
cp_function_chain->x_current_class_ptr = saved_class_ptr;
cp_function_chain->x_current_class_ref = saved_class_ref;
}
if (errorcount+sorrycount > errs
&& (complain & tf_warning_or_error))
inform (input_location,
"  when instantiating default argument for call to %qD", fn);
arg = check_default_argument (type, arg, complain);
pop_access_scope (fn);
if (arg != error_mark_node && !cp_unevaluated_operand)
{
if (!defarg_inst)
defarg_inst = tree_cache_map::create_ggc (37);
defarg_inst->put (parm, arg);
}
return arg;
}
static void
tsubst_default_arguments (tree fn, tsubst_flags_t complain)
{
tree arg;
tree tmpl_args;
tmpl_args = DECL_TI_ARGS (fn);
if (uses_template_parms (tmpl_args))
return;
if (DECL_CLONED_FUNCTION_P (fn))
return;
int i = 0;
for (arg = TYPE_ARG_TYPES (TREE_TYPE (fn));
arg;
arg = TREE_CHAIN (arg), ++i)
if (TREE_PURPOSE (arg))
TREE_PURPOSE (arg) = tsubst_default_argument (fn, i,
TREE_VALUE (arg),
TREE_PURPOSE (arg),
complain);
}
static tree
tsubst_function_decl (tree t, tree args, tsubst_flags_t complain,
tree lambda_fntype)
{
tree gen_tmpl, argvec;
hashval_t hash = 0;
tree in_decl = t;
gcc_assert (DECL_TEMPLATE_INFO (t) != NULL_TREE);
if (TREE_CODE (DECL_TI_TEMPLATE (t)) == TEMPLATE_DECL)
{
if (!uses_template_parms (DECL_TI_ARGS (t)))
return t;
gen_tmpl = most_general_template (DECL_TI_TEMPLATE (t));
if (LAMBDA_FUNCTION_P (t) && !lambda_fntype
&& (!generic_lambda_fn_p (t)
|| TMPL_PARMS_DEPTH (DECL_TEMPLATE_PARMS (gen_tmpl)) > 1))
return enclosing_instantiation_of (t);
argvec = tsubst_template_args (DECL_TI_ARGS
(DECL_TEMPLATE_RESULT
(DECL_TI_TEMPLATE (t))),
args, complain, in_decl);
if (argvec == error_mark_node)
return error_mark_node;
if (!lambda_fntype)
{
hash = hash_tmpl_and_args (gen_tmpl, argvec);
if (tree spec = retrieve_specialization (gen_tmpl, argvec, hash))
return spec;
}
int args_depth = TMPL_ARGS_DEPTH (args);
int parms_depth =
TMPL_PARMS_DEPTH (DECL_TEMPLATE_PARMS (DECL_TI_TEMPLATE (t)));
if (args_depth > parms_depth && !DECL_TEMPLATE_SPECIALIZATION (t))
args = get_innermost_template_args (args, parms_depth);
}
else
{
gen_tmpl = NULL_TREE;
argvec = NULL_TREE;
}
tree closure = (lambda_fntype ? TYPE_METHOD_BASETYPE (lambda_fntype)
: NULL_TREE);
tree ctx = closure ? closure : DECL_CONTEXT (t);
bool member = ctx && TYPE_P (ctx);
if (member && !closure)
ctx = tsubst_aggr_type (ctx, args,
complain, t, 1);
tree type = (lambda_fntype ? lambda_fntype
: tsubst (TREE_TYPE (t), args,
complain | tf_fndecl_type, in_decl));
if (type == error_mark_node)
return error_mark_node;
if (excessive_deduction_depth)
return error_mark_node;
tree r = copy_decl (t);
DECL_USE_TEMPLATE (r) = 0;
TREE_TYPE (r) = type;
SET_DECL_ASSEMBLER_NAME (r, NULL_TREE);
SET_DECL_RTL (r, NULL);
if (!DECL_DELETED_FN (r))
DECL_INITIAL (r) = NULL_TREE;
DECL_CONTEXT (r) = ctx;
if (DECL_OMP_DECLARE_REDUCTION_P (t))
{
tree argtype
= TREE_TYPE (TREE_VALUE (TYPE_ARG_TYPES (TREE_TYPE (t))));
argtype = tsubst (argtype, args, complain, in_decl);
if (TREE_CODE (argtype) == REFERENCE_TYPE)
error_at (DECL_SOURCE_LOCATION (t),
"reference type %qT in "
"%<#pragma omp declare reduction%>", argtype);
if (strchr (IDENTIFIER_POINTER (DECL_NAME (t)), '~') == NULL)
DECL_NAME (r) = omp_reduction_id (ERROR_MARK, DECL_NAME (t),
argtype);
}
if (member && DECL_CONV_FN_P (r))
DECL_NAME (r) = make_conv_op_name (TREE_TYPE (type));
tree parms = DECL_ARGUMENTS (t);
if (closure)
parms = DECL_CHAIN (parms);
parms = tsubst (parms, args, complain, t);
for (tree parm = parms; parm; parm = DECL_CHAIN (parm))
DECL_CONTEXT (parm) = r;
if (closure)
{
tree tparm = build_this_parm (r, closure, type_memfn_quals (type));
DECL_CHAIN (tparm) = parms;
parms = tparm;
}
DECL_ARGUMENTS (r) = parms;
DECL_RESULT (r) = NULL_TREE;
TREE_STATIC (r) = 0;
TREE_PUBLIC (r) = TREE_PUBLIC (t);
DECL_EXTERNAL (r) = 1;
DECL_INTERFACE_KNOWN (r) = !TREE_PUBLIC (r);
DECL_DEFER_OUTPUT (r) = 0;
DECL_CHAIN (r) = NULL_TREE;
DECL_PENDING_INLINE_INFO (r) = 0;
DECL_PENDING_INLINE_P (r) = 0;
DECL_SAVED_TREE (r) = NULL_TREE;
DECL_STRUCT_FUNCTION (r) = NULL;
TREE_USED (r) = 0;
DECL_CLONED_FUNCTION (r) = NULL_TREE;
if ((complain & tf_error) == 0
&& IDENTIFIER_ANY_OP_P (DECL_NAME (r))
&& !grok_op_properties (r, false))
return error_mark_node;
if (tree ci = get_constraints (t))
if (member)
{
ci = tsubst_constraint_info (ci, argvec, complain, NULL_TREE);
set_constraints (r, ci);
}
if (gen_tmpl && !closure)
{
DECL_TEMPLATE_INFO (r)
= build_template_info (gen_tmpl, argvec);
SET_DECL_IMPLICIT_INSTANTIATION (r);
tree new_r
= register_specialization (r, gen_tmpl, argvec, false, hash);
if (new_r != r)
return new_r;
if (!member
&& !PRIMARY_TEMPLATE_P (gen_tmpl)
&& !uses_template_parms (argvec))
tsubst_default_arguments (r, complain);
}
else
DECL_TEMPLATE_INFO (r) = NULL_TREE;
for (tree *friends = &DECL_BEFRIENDING_CLASSES (r);
*friends;
friends = &TREE_CHAIN (*friends))
{
*friends = copy_node (*friends);
TREE_VALUE (*friends)
= tsubst (TREE_VALUE (*friends), args, complain, in_decl);
}
if (DECL_CONSTRUCTOR_P (r) || DECL_DESTRUCTOR_P (r))
{
maybe_retrofit_in_chrg (r);
if (DECL_CONSTRUCTOR_P (r) && !grok_ctor_properties (ctx, r))
return error_mark_node;
if (PRIMARY_TEMPLATE_P (gen_tmpl))
clone_function_decl (r, false);
}
else if ((complain & tf_error) != 0
&& IDENTIFIER_ANY_OP_P (DECL_NAME (r))
&& !grok_op_properties (r, true))
return error_mark_node;
if (DECL_FRIEND_P (t) && DECL_FRIEND_CONTEXT (t))
SET_DECL_FRIEND_CONTEXT (r,
tsubst (DECL_FRIEND_CONTEXT (t),
args, complain, in_decl));
DECL_VISIBILITY (r) = VISIBILITY_DEFAULT;
if (DECL_VISIBILITY_SPECIFIED (t))
{
DECL_VISIBILITY_SPECIFIED (r) = 0;
DECL_ATTRIBUTES (r)
= remove_attribute ("visibility", DECL_ATTRIBUTES (r));
}
determine_visibility (r);
if (DECL_DEFAULTED_OUTSIDE_CLASS_P (r)
&& !processing_template_decl)
defaulted_late_check (r);
apply_late_template_attributes (&r, DECL_ATTRIBUTES (r), 0,
args, complain, in_decl);
return r;
}
static tree
tsubst_template_decl (tree t, tree args, tsubst_flags_t complain,
tree lambda_fntype)
{
tree decl = DECL_TEMPLATE_RESULT (t);
tree in_decl = t;
tree spec;
tree tmpl_args;
tree full_args;
tree r;
hashval_t hash = 0;
if (DECL_TEMPLATE_TEMPLATE_PARM_P (t))
{
tree new_type = tsubst (TREE_TYPE (t), args, complain, in_decl);
if (new_type == error_mark_node)
r = error_mark_node;
else if (TREE_CODE (new_type) == TEMPLATE_DECL)
r = new_type;
else
r = TEMPLATE_TEMPLATE_PARM_TEMPLATE_DECL (new_type);
return r;
}
if (!lambda_fntype)
{
tmpl_args = DECL_CLASS_TEMPLATE_P (t)
? CLASSTYPE_TI_ARGS (TREE_TYPE (t))
: DECL_TI_ARGS (DECL_TEMPLATE_RESULT (t));
++processing_template_decl;
full_args = tsubst_template_args (tmpl_args, args,
complain, in_decl);
--processing_template_decl;
if (full_args == error_mark_node)
return error_mark_node;
if (full_args == tmpl_args)
return t;
hash = hash_tmpl_and_args (t, full_args);
spec = retrieve_specialization (t, full_args, hash);
if (spec != NULL_TREE)
return spec;
}
r = copy_decl (t);
gcc_assert (DECL_LANG_SPECIFIC (r) != 0);
DECL_CHAIN (r) = NULL_TREE;
if (!lambda_fntype)
{
DECL_TEMPLATE_INFO (r) = build_template_info (t, args);
SET_DECL_IMPLICIT_INSTANTIATION (r);
}
else
DECL_TEMPLATE_INFO (r) = NULL_TREE;
DECL_TEMPLATE_PARMS (r)
= tsubst_template_parms (DECL_TEMPLATE_PARMS (t), args,
complain);
if (TREE_CODE (decl) == TYPE_DECL
&& !TYPE_DECL_ALIAS_P (decl))
{
tree new_type;
++processing_template_decl;
new_type = tsubst (TREE_TYPE (t), args, complain, in_decl);
--processing_template_decl;
if (new_type == error_mark_node)
return error_mark_node;
TREE_TYPE (r) = new_type;
if (!DECL_TEMPLATE_SPECIALIZATION (t))
CLASSTYPE_TI_TEMPLATE (new_type) = r;
DECL_TEMPLATE_RESULT (r) = TYPE_MAIN_DECL (new_type);
DECL_TI_ARGS (r) = CLASSTYPE_TI_ARGS (new_type);
DECL_CONTEXT (r) = TYPE_CONTEXT (new_type);
}
else
{
tree new_decl;
++processing_template_decl;
if (TREE_CODE (decl) == FUNCTION_DECL)
new_decl = tsubst_function_decl (decl, args, complain, lambda_fntype);
else
new_decl = tsubst (decl, args, complain, in_decl);
--processing_template_decl;
if (new_decl == error_mark_node)
return error_mark_node;
DECL_TEMPLATE_RESULT (r) = new_decl;
TREE_TYPE (r) = TREE_TYPE (new_decl);
DECL_CONTEXT (r) = DECL_CONTEXT (new_decl);
if (lambda_fntype)
{
tree args = template_parms_to_args (DECL_TEMPLATE_PARMS (r));
DECL_TEMPLATE_INFO (new_decl) = build_template_info (r, args);
}
else
{
DECL_TI_TEMPLATE (new_decl) = r;
DECL_TI_ARGS (r) = DECL_TI_ARGS (new_decl);
}
}
DECL_TEMPLATE_INSTANTIATIONS (r) = NULL_TREE;
DECL_TEMPLATE_SPECIALIZATIONS (r) = NULL_TREE;
if (PRIMARY_TEMPLATE_P (t))
DECL_PRIMARY_TEMPLATE (r) = r;
if (TREE_CODE (decl) != TYPE_DECL && !VAR_P (decl)
&& !lambda_fntype)
register_specialization (r, t,
DECL_TI_ARGS (DECL_TEMPLATE_RESULT (r)),
false, hash);
return r;
}
bool
lambda_fn_in_template_p (tree fn)
{
if (!fn || !LAMBDA_FUNCTION_P (fn))
return false;
tree closure = DECL_CONTEXT (fn);
return CLASSTYPE_TEMPLATE_INFO (closure) != NULL_TREE;
}
static tree
enclosing_instantiation_of (tree otctx)
{
tree tctx = otctx;
tree fn = current_function_decl;
int lambda_count = 0;
for (; tctx && lambda_fn_in_template_p (tctx);
tctx = decl_function_context (tctx))
++lambda_count;
for (; fn; fn = decl_function_context (fn))
{
tree ofn = fn;
int flambda_count = 0;
for (; flambda_count < lambda_count && fn && LAMBDA_FUNCTION_P (fn);
fn = decl_function_context (fn))
++flambda_count;
if ((fn && DECL_TEMPLATE_INFO (fn))
? most_general_template (fn) != most_general_template (tctx)
: fn != tctx)
continue;
gcc_assert (DECL_NAME (ofn) == DECL_NAME (otctx)
|| DECL_CONV_FN_P (ofn));
return ofn;
}
gcc_unreachable ();
}
static tree
tsubst_decl (tree t, tree args, tsubst_flags_t complain)
{
#define RETURN(EXP) do { r = (EXP); goto out; } while(0)
location_t saved_loc;
tree r = NULL_TREE;
tree in_decl = t;
hashval_t hash = 0;
saved_loc = input_location;
input_location = DECL_SOURCE_LOCATION (t);
switch (TREE_CODE (t))
{
case TEMPLATE_DECL:
r = tsubst_template_decl (t, args, complain, NULL_TREE);
break;
case FUNCTION_DECL:
r = tsubst_function_decl (t, args, complain, NULL_TREE);
break;
case PARM_DECL:
{
tree type = NULL_TREE;
int i, len = 1;
tree expanded_types = NULL_TREE;
tree prev_r = NULL_TREE;
tree first_r = NULL_TREE;
if (DECL_PACK_P (t))
{
tree spec = retrieve_local_specialization (t);
if (spec 
&& TREE_CODE (spec) == PARM_DECL
&& TREE_CODE (TREE_TYPE (spec)) != TYPE_PACK_EXPANSION)
RETURN (spec);
expanded_types = tsubst_pack_expansion (TREE_TYPE (t), args,
complain, in_decl);
if (TREE_CODE (expanded_types) == TREE_VEC)
{
len = TREE_VEC_LENGTH (expanded_types);
if (len == 0)
RETURN (tsubst (TREE_CHAIN (t), args, complain,
TREE_CHAIN (t)));
}
else
{
type = expanded_types;
expanded_types = NULL_TREE;
}
}
r = NULL_TREE;
for (i = 0; i < len; ++i)
{
prev_r = r;
r = copy_node (t);
if (DECL_TEMPLATE_PARM_P (t))
SET_DECL_TEMPLATE_PARM_P (r);
if (expanded_types)
{
type = TREE_VEC_ELT (expanded_types, i);
DECL_NAME (r)
= make_ith_pack_parameter_name (DECL_NAME (r), i);
}
else if (!type)
type = tsubst (TREE_TYPE (t), args, complain, in_decl);
type = type_decays_to (type);
TREE_TYPE (r) = type;
cp_apply_type_quals_to_decl (cp_type_quals (type), r);
if (DECL_INITIAL (r))
{
if (TREE_CODE (DECL_INITIAL (r)) != TEMPLATE_PARM_INDEX)
DECL_INITIAL (r) = TREE_TYPE (r);
else
DECL_INITIAL (r) = tsubst (DECL_INITIAL (r), args,
complain, in_decl);
}
DECL_CONTEXT (r) = NULL_TREE;
if (!DECL_TEMPLATE_PARM_P (r))
DECL_ARG_TYPE (r) = type_passed_as (type);
apply_late_template_attributes (&r, DECL_ATTRIBUTES (r), 0,
args, complain, in_decl);
if (!first_r)
first_r = r;
if (prev_r)
DECL_CHAIN (prev_r) = r;
}
if (DECL_CHAIN (t) && !cp_unevaluated_operand)
DECL_CHAIN (r) = tsubst (DECL_CHAIN (t), args,
complain, DECL_CHAIN (t));
r = first_r;
}
break;
case FIELD_DECL:
{
tree type = NULL_TREE;
tree vec = NULL_TREE;
tree expanded_types = NULL_TREE;
int len = 1;
if (PACK_EXPANSION_P (TREE_TYPE (t)))
{
expanded_types = tsubst_pack_expansion (TREE_TYPE (t), args,
complain, in_decl);
if (TREE_CODE (expanded_types) == TREE_VEC)
{
len = TREE_VEC_LENGTH (expanded_types);
vec = make_tree_vec (len);
}
else
{
type = expanded_types;
expanded_types = NULL_TREE;
}
}
for (int i = 0; i < len; ++i)
{
r = copy_decl (t);
if (expanded_types)
{
type = TREE_VEC_ELT (expanded_types, i);
DECL_NAME (r)
= make_ith_pack_parameter_name (DECL_NAME (r), i);
}
else if (!type)
type = tsubst (TREE_TYPE (t), args, complain, in_decl);
if (type == error_mark_node)
RETURN (error_mark_node);
TREE_TYPE (r) = type;
cp_apply_type_quals_to_decl (cp_type_quals (type), r);
if (DECL_C_BIT_FIELD (r))
DECL_BIT_FIELD_REPRESENTATIVE (r)
= tsubst_expr (DECL_BIT_FIELD_REPRESENTATIVE (t), args,
complain, in_decl,
true);
if (DECL_INITIAL (t))
{
DECL_INITIAL (r) = void_node;
gcc_assert (DECL_LANG_SPECIFIC (r) == NULL);
retrofit_lang_decl (r);
DECL_TEMPLATE_INFO (r) = build_template_info (t, args);
}
DECL_CHAIN (r) = NULL_TREE;
apply_late_template_attributes (&r, DECL_ATTRIBUTES (r), 0,
args, complain, in_decl);
if (vec)
TREE_VEC_ELT (vec, i) = r;
}
if (vec)
r = vec;
}
break;
case USING_DECL:
if (DECL_DEPENDENT_P (t)
|| uses_template_parms (USING_DECL_SCOPE (t)))
{
tree scope = USING_DECL_SCOPE (t);
tree name = tsubst_copy (DECL_NAME (t), args, complain, in_decl);
if (PACK_EXPANSION_P (scope))
{
tree vec = tsubst_pack_expansion (scope, args, complain, in_decl);
int len = TREE_VEC_LENGTH (vec);
r = make_tree_vec (len);
for (int i = 0; i < len; ++i)
{
tree escope = TREE_VEC_ELT (vec, i);
tree elt = do_class_using_decl (escope, name);
if (!elt)
{
r = error_mark_node;
break;
}
else
{
TREE_PROTECTED (elt) = TREE_PROTECTED (t);
TREE_PRIVATE (elt) = TREE_PRIVATE (t);
}
TREE_VEC_ELT (r, i) = elt;
}
}
else
{
tree inst_scope = tsubst_copy (USING_DECL_SCOPE (t), args,
complain, in_decl);
r = do_class_using_decl (inst_scope, name);
if (!r)
r = error_mark_node;
else
{
TREE_PROTECTED (r) = TREE_PROTECTED (t);
TREE_PRIVATE (r) = TREE_PRIVATE (t);
}
}
}
else
{
r = copy_node (t);
DECL_CHAIN (r) = NULL_TREE;
}
break;
case TYPE_DECL:
case VAR_DECL:
{
tree argvec = NULL_TREE;
tree gen_tmpl = NULL_TREE;
tree spec;
tree tmpl = NULL_TREE;
tree ctx;
tree type = NULL_TREE;
bool local_p;
if (TREE_TYPE (t) == error_mark_node)
RETURN (error_mark_node);
if (TREE_CODE (t) == TYPE_DECL
&& t == TYPE_MAIN_DECL (TREE_TYPE (t)))
{
type = tsubst (TREE_TYPE (t), args, complain, in_decl);
if (type == error_mark_node)
RETURN (error_mark_node);
r = TYPE_NAME (type);
break;
}
spec = NULL_TREE;
if (DECL_CLASS_SCOPE_P (t) || DECL_NAMESPACE_SCOPE_P (t))
{
local_p = false;
ctx = DECL_CONTEXT (t);
if (DECL_CLASS_SCOPE_P (t))
{
ctx = tsubst_aggr_type (ctx, args,
complain,
in_decl, 1);
if (ctx == DECL_CONTEXT (t)
&& !(DECL_TI_TEMPLATE (t)
&& DECL_MEMBER_TEMPLATE_P (DECL_TI_TEMPLATE (t))))
spec = t;
}
if (!spec)
{
tmpl = DECL_TI_TEMPLATE (t);
gen_tmpl = most_general_template (tmpl);
argvec = tsubst (DECL_TI_ARGS (t), args, complain, in_decl);
if (argvec != error_mark_node)
argvec = (coerce_innermost_template_parms
(DECL_TEMPLATE_PARMS (gen_tmpl),
argvec, t, complain,
true, true));
if (argvec == error_mark_node)
RETURN (error_mark_node);
hash = hash_tmpl_and_args (gen_tmpl, argvec);
spec = retrieve_specialization (gen_tmpl, argvec, hash);
}
}
else
{
local_p = true;
ctx = NULL_TREE;
if (TREE_STATIC (t))
{
tree fn = enclosing_instantiation_of (DECL_CONTEXT (t));
if (fn != current_function_decl)
ctx = fn;
}
spec = retrieve_local_specialization (t);
}
if (spec)
{
r = spec;
break;
}
r = copy_decl (t);
if (type == NULL_TREE)
{
if (is_typedef_decl (t))
type = DECL_ORIGINAL_TYPE (t);
else
type = TREE_TYPE (t);
if (VAR_P (t)
&& VAR_HAD_UNKNOWN_BOUND (t)
&& type != error_mark_node)
type = strip_array_domain (type);
tree sub_args = args;
if (tree auto_node = type_uses_auto (type))
{
int nouter = TEMPLATE_TYPE_LEVEL (auto_node) - 1;
int extra = TMPL_ARGS_DEPTH (args) - nouter;
if (extra > 0)
gcc_assert (!CHECKING_P),
sub_args = strip_innermost_template_args (args, extra);
}
type = tsubst (type, sub_args, complain, in_decl);
}
if (VAR_P (r))
{
DECL_DEAD_FOR_LOCAL (r) = 0;
DECL_INITIALIZED_P (r) = 0;
DECL_TEMPLATE_INSTANTIATED (r) = 0;
if (type == error_mark_node)
RETURN (error_mark_node);
if (TREE_CODE (type) == FUNCTION_TYPE)
{
error ("variable %qD has function type",
DECL_NAME (r));
RETURN (error_mark_node);
}
type = complete_type (type);
DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P (r) = 0;
type = check_var_type (DECL_NAME (r), type);
if (DECL_HAS_VALUE_EXPR_P (t))
{
tree ve = DECL_VALUE_EXPR (t);
ve = tsubst_expr (ve, args, complain, in_decl,
false);
if (REFERENCE_REF_P (ve))
{
gcc_assert (TREE_CODE (type) == REFERENCE_TYPE);
ve = TREE_OPERAND (ve, 0);
}
SET_DECL_VALUE_EXPR (r, ve);
}
if (CP_DECL_THREAD_LOCAL_P (r)
&& !processing_template_decl)
set_decl_tls_model (r, decl_default_tls_model (r));
}
else if (DECL_SELF_REFERENCE_P (t))
SET_DECL_SELF_REFERENCE_P (r);
TREE_TYPE (r) = type;
cp_apply_type_quals_to_decl (cp_type_quals (type), r);
DECL_CONTEXT (r) = ctx;
SET_DECL_ASSEMBLER_NAME (r, NULL_TREE);
if (CODE_CONTAINS_STRUCT (TREE_CODE (t), TS_DECL_WRTL))
SET_DECL_RTL (r, NULL);
DECL_INITIAL (r) = NULL_TREE;
DECL_SIZE (r) = DECL_SIZE_UNIT (r) = 0;
if (VAR_P (r))
{
if (DECL_LANG_SPECIFIC (r))
SET_DECL_DEPENDENT_INIT_P (r, false);
SET_DECL_MODE (r, VOIDmode);
DECL_VISIBILITY (r) = VISIBILITY_DEFAULT;
if (DECL_VISIBILITY_SPECIFIED (t))
{
DECL_VISIBILITY_SPECIFIED (r) = 0;
DECL_ATTRIBUTES (r)
= remove_attribute ("visibility", DECL_ATTRIBUTES (r));
}
determine_visibility (r);
}
if (!local_p)
{
DECL_EXTERNAL (r) = 1;
if (DECL_NAMESPACE_SCOPE_P (t))
DECL_NOT_REALLY_EXTERN (r) = 1;
DECL_TEMPLATE_INFO (r) = build_template_info (tmpl, argvec);
SET_DECL_IMPLICIT_INSTANTIATION (r);
register_specialization (r, gen_tmpl, argvec, false, hash);
}
else
{
if (DECL_LANG_SPECIFIC (r))
DECL_TEMPLATE_INFO (r) = NULL_TREE;
if (!cp_unevaluated_operand)
register_local_specialization (r, t);
}
DECL_CHAIN (r) = NULL_TREE;
apply_late_template_attributes (&r, DECL_ATTRIBUTES (r),
0,
args, complain, in_decl);
if (is_typedef_decl (r) && type != error_mark_node)
{
DECL_ORIGINAL_TYPE (r) = NULL_TREE;
set_underlying_type (r);
if (TYPE_DECL_ALIAS_P (r))
TYPE_DEPENDENT_P_VALID (TREE_TYPE (r)) = false;
}
layout_decl (r, 0);
}
break;
default:
gcc_unreachable ();
}
#undef RETURN
out:
input_location = saved_loc;
return r;
}
static tree
tsubst_arg_types (tree arg_types,
tree args,
tree end,
tsubst_flags_t complain,
tree in_decl)
{
tree remaining_arg_types;
tree type = NULL_TREE;
int i = 1;
tree expanded_args = NULL_TREE;
tree default_arg;
if (!arg_types || arg_types == void_list_node || arg_types == end)
return arg_types;
remaining_arg_types = tsubst_arg_types (TREE_CHAIN (arg_types),
args, end, complain, in_decl);
if (remaining_arg_types == error_mark_node)
return error_mark_node;
if (PACK_EXPANSION_P (TREE_VALUE (arg_types)))
{
expanded_args = tsubst_pack_expansion (TREE_VALUE (arg_types),
args, complain, in_decl);
if (TREE_CODE (expanded_args) == TREE_VEC)
i = TREE_VEC_LENGTH (expanded_args);
else
{
type = expanded_args;
expanded_args = NULL_TREE;
}
}
while (i > 0) {
--i;
if (expanded_args)
type = TREE_VEC_ELT (expanded_args, i);
else if (!type)
type = tsubst (TREE_VALUE (arg_types), args, complain, in_decl);
if (type == error_mark_node)
return error_mark_node;
if (VOID_TYPE_P (type))
{
if (complain & tf_error)
{
error ("invalid parameter type %qT", type);
if (in_decl)
error ("in declaration %q+D", in_decl);
}
return error_mark_node;
}
if (abstract_virtuals_error_sfinae (ACU_PARM, type, complain))
return error_mark_node;
type = cv_unqualified (type_decays_to (type));
default_arg = TREE_PURPOSE (arg_types);
if (lambda_fn_in_template_p (in_decl))
default_arg = tsubst_copy_and_build (default_arg, args, complain, in_decl,
false, false);
if (default_arg && TREE_CODE (default_arg) == DEFAULT_ARG)
{
remaining_arg_types = 
tree_cons (default_arg, type, remaining_arg_types);
vec_safe_push (DEFARG_INSTANTIATIONS(default_arg), remaining_arg_types);
}
else
remaining_arg_types = 
hash_tree_cons (default_arg, type, remaining_arg_types);
}
return remaining_arg_types;
}
static tree
tsubst_function_type (tree t,
tree args,
tsubst_flags_t complain,
tree in_decl)
{
tree return_type;
tree arg_types = NULL_TREE;
tree fntype;
gcc_assert (TYPE_CONTEXT (t) == NULL_TREE);
bool late_return_type_p = TYPE_HAS_LATE_RETURN_TYPE (t);
if (late_return_type_p)
{
arg_types = tsubst_arg_types (TYPE_ARG_TYPES (t), args, NULL_TREE,
complain, in_decl);
if (arg_types == error_mark_node)
return error_mark_node;
tree save_ccp = current_class_ptr;
tree save_ccr = current_class_ref;
tree this_type = (TREE_CODE (t) == METHOD_TYPE
? TREE_TYPE (TREE_VALUE (arg_types)) : NULL_TREE);
bool do_inject = this_type && CLASS_TYPE_P (this_type);
if (do_inject)
{
inject_this_parameter (this_type, cp_type_quals (this_type));
}
return_type = tsubst (TREE_TYPE (t), args, complain, in_decl);
if (do_inject)
{
current_class_ptr = save_ccp;
current_class_ref = save_ccr;
}
}
else
return_type = tsubst (TREE_TYPE (t), args, complain, in_decl);
if (return_type == error_mark_node)
return error_mark_node;
if (TREE_CODE (return_type) == ARRAY_TYPE
|| TREE_CODE (return_type) == FUNCTION_TYPE)
{
if (complain & tf_error)
{
if (TREE_CODE (return_type) == ARRAY_TYPE)
error ("function returning an array");
else
error ("function returning a function");
}
return error_mark_node;
}
if (abstract_virtuals_error_sfinae (ACU_RETURN, return_type, complain))
return error_mark_node;
if (!late_return_type_p)
{
arg_types = tsubst_arg_types (TYPE_ARG_TYPES (t), args, NULL_TREE,
complain, in_decl);
if (arg_types == error_mark_node)
return error_mark_node;
}
if (TREE_CODE (t) == FUNCTION_TYPE)
{
fntype = build_function_type (return_type, arg_types);
fntype = apply_memfn_quals (fntype,
type_memfn_quals (t),
type_memfn_rqual (t));
}
else
{
tree r = TREE_TYPE (TREE_VALUE (arg_types));
r = cp_build_qualified_type_real (r, type_memfn_quals (t), complain);
if (! MAYBE_CLASS_TYPE_P (r))
{
if (complain & tf_error)
error ("creating pointer to member function of non-class type %qT",
r);
return error_mark_node;
}
fntype = build_method_type_directly (r, return_type,
TREE_CHAIN (arg_types));
fntype = build_ref_qualified_type (fntype, type_memfn_rqual (t));
}
fntype = cp_build_type_attribute_variant (fntype, TYPE_ATTRIBUTES (t));
if (late_return_type_p)
TYPE_HAS_LATE_RETURN_TYPE (fntype) = 1;
return fntype;
}
static tree
tsubst_exception_specification (tree fntype,
tree args,
tsubst_flags_t complain,
tree in_decl,
bool defer_ok)
{
tree specs;
tree new_specs;
specs = TYPE_RAISES_EXCEPTIONS (fntype);
new_specs = NULL_TREE;
if (specs && TREE_PURPOSE (specs))
{
tree expr = TREE_PURPOSE (specs);
if (TREE_CODE (expr) == INTEGER_CST)
new_specs = expr;
else if (defer_ok)
{
new_specs = make_node (DEFERRED_NOEXCEPT);
if (DEFERRED_NOEXCEPT_SPEC_P (specs))
{
DEFERRED_NOEXCEPT_PATTERN (new_specs)
= DEFERRED_NOEXCEPT_PATTERN (expr);
DEFERRED_NOEXCEPT_ARGS (new_specs)
= add_to_template_args (DEFERRED_NOEXCEPT_ARGS (expr), args);
}
else
{
DEFERRED_NOEXCEPT_PATTERN (new_specs) = expr;
DEFERRED_NOEXCEPT_ARGS (new_specs) = args;
}
}
else
new_specs = tsubst_copy_and_build
(expr, args, complain, in_decl, false,
true);
new_specs = build_noexcept_spec (new_specs, complain);
}
else if (specs)
{
if (! TREE_VALUE (specs))
new_specs = specs;
else
while (specs)
{
tree spec;
int i, len = 1;
tree expanded_specs = NULL_TREE;
if (PACK_EXPANSION_P (TREE_VALUE (specs)))
{
expanded_specs = tsubst_pack_expansion (TREE_VALUE (specs),
args, complain,
in_decl);
if (expanded_specs == error_mark_node)
return error_mark_node;
else if (TREE_CODE (expanded_specs) == TREE_VEC)
len = TREE_VEC_LENGTH (expanded_specs);
else
{
gcc_assert (TREE_CODE (expanded_specs) 
== TYPE_PACK_EXPANSION);
new_specs = add_exception_specifier (new_specs,
expanded_specs,
complain);
specs = TREE_CHAIN (specs);
continue;
}
}
for (i = 0; i < len; ++i)
{
if (expanded_specs)
spec = TREE_VEC_ELT (expanded_specs, i);
else
spec = tsubst (TREE_VALUE (specs), args, complain, in_decl);
if (spec == error_mark_node)
return spec;
new_specs = add_exception_specifier (new_specs, spec, 
complain);
}
specs = TREE_CHAIN (specs);
}
}
return new_specs;
}
tree
tsubst (tree t, tree args, tsubst_flags_t complain, tree in_decl)
{
enum tree_code code;
tree type, r = NULL_TREE;
if (t == NULL_TREE || t == error_mark_node
|| t == integer_type_node
|| t == void_type_node
|| t == char_type_node
|| t == unknown_type_node
|| TREE_CODE (t) == NAMESPACE_DECL
|| TREE_CODE (t) == TRANSLATION_UNIT_DECL)
return t;
if (DECL_P (t))
return tsubst_decl (t, args, complain);
if (args == NULL_TREE)
return t;
code = TREE_CODE (t);
if (code == IDENTIFIER_NODE)
type = IDENTIFIER_TYPE_VALUE (t);
else
type = TREE_TYPE (t);
gcc_assert (type != unknown_type_node);
if (TYPE_P (t)
&& typedef_variant_p (t))
{
tree decl = TYPE_NAME (t);
if (alias_template_specialization_p (t))
{
tree tmpl = most_general_template (DECL_TI_TEMPLATE (decl));
tree gen_args = tsubst (DECL_TI_ARGS (decl), args, complain, in_decl);
r = instantiate_alias_template (tmpl, gen_args, complain);
}
else if (DECL_CLASS_SCOPE_P (decl)
&& CLASSTYPE_TEMPLATE_INFO (DECL_CONTEXT (decl))
&& uses_template_parms (DECL_CONTEXT (decl)))
{
tree tmpl = most_general_template (DECL_TI_TEMPLATE (decl));
tree gen_args = tsubst (DECL_TI_ARGS (decl), args, complain, in_decl);
r = retrieve_specialization (tmpl, gen_args, 0);
}
else if (DECL_FUNCTION_SCOPE_P (decl)
&& DECL_TEMPLATE_INFO (DECL_CONTEXT (decl))
&& uses_template_parms (DECL_TI_ARGS (DECL_CONTEXT (decl))))
r = retrieve_local_specialization (decl);
else
return t;
if (r)
{
r = TREE_TYPE (r);
r = cp_build_qualified_type_real
(r, cp_type_quals (t) | cp_type_quals (r),
complain | tf_ignore_bad_quals);
return r;
}
else
{
int quals = cp_type_quals (t);
t = DECL_ORIGINAL_TYPE (decl);
t = cp_build_qualified_type_real (t, quals,
complain | tf_ignore_bad_quals);
}
}
bool fndecl_type = (complain & tf_fndecl_type);
complain &= ~tf_fndecl_type;
if (type
&& code != TYPENAME_TYPE
&& code != TEMPLATE_TYPE_PARM
&& code != TEMPLATE_PARM_INDEX
&& code != IDENTIFIER_NODE
&& code != FUNCTION_TYPE
&& code != METHOD_TYPE)
type = tsubst (type, args, complain, in_decl);
if (type == error_mark_node)
return error_mark_node;
switch (code)
{
case RECORD_TYPE:
case UNION_TYPE:
case ENUMERAL_TYPE:
return tsubst_aggr_type (t, args, complain, in_decl,
0);
case ERROR_MARK:
case IDENTIFIER_NODE:
case VOID_TYPE:
case REAL_TYPE:
case COMPLEX_TYPE:
case VECTOR_TYPE:
case BOOLEAN_TYPE:
case NULLPTR_TYPE:
case LANG_TYPE:
return t;
case INTEGER_TYPE:
if (t == integer_type_node)
return t;
if (TREE_CODE (TYPE_MIN_VALUE (t)) == INTEGER_CST
&& TREE_CODE (TYPE_MAX_VALUE (t)) == INTEGER_CST)
return t;
{
tree max, omax = TREE_OPERAND (TYPE_MAX_VALUE (t), 0);
max = tsubst_expr (omax, args, complain, in_decl,
false);
if (TREE_CODE (max) == NOP_EXPR
&& TREE_SIDE_EFFECTS (omax)
&& !TREE_TYPE (max))
TREE_TYPE (max) = TREE_TYPE (TREE_OPERAND (max, 0));
if (processing_template_decl
&& TREE_SIDE_EFFECTS (omax) && TREE_CODE (omax) == NOP_EXPR)
{
gcc_assert (TREE_CODE (max) == NOP_EXPR);
TREE_SIDE_EFFECTS (max) = 1;
}
return compute_array_index_type (NULL_TREE, max, complain);
}
case TEMPLATE_TYPE_PARM:
case TEMPLATE_TEMPLATE_PARM:
case BOUND_TEMPLATE_TEMPLATE_PARM:
case TEMPLATE_PARM_INDEX:
{
int idx;
int level;
int levels;
tree arg = NULL_TREE;
if (is_auto (t) && (complain & tf_partial))
return t;
r = NULL_TREE;
gcc_assert (TREE_VEC_LENGTH (args) > 0);
template_parm_level_and_index (t, &level, &idx); 
levels = TMPL_ARGS_DEPTH (args);
if (level <= levels
&& TREE_VEC_LENGTH (TMPL_ARGS_LEVEL (args, level)) > 0)
{
arg = TMPL_ARG (args, level, idx);
if (arg && TREE_CODE (arg) == ARGUMENT_PACK_SELECT)
arg = argument_pack_select_arg (arg);
}
if (arg == error_mark_node)
return error_mark_node;
else if (arg != NULL_TREE)
{
if (ARGUMENT_PACK_P (arg))
return t;
if (code == TEMPLATE_TYPE_PARM)
{
int quals;
gcc_assert (TYPE_P (arg));
quals = cp_type_quals (arg) | cp_type_quals (t);
return cp_build_qualified_type_real
(arg, quals, complain | tf_ignore_bad_quals);
}
else if (code == BOUND_TEMPLATE_TEMPLATE_PARM)
{
tree argvec = tsubst (TYPE_TI_ARGS (t),
args, complain, in_decl);
if (argvec == error_mark_node)
return error_mark_node;
gcc_assert (TREE_CODE (arg) == TEMPLATE_TEMPLATE_PARM
|| TREE_CODE (arg) == TEMPLATE_DECL
|| TREE_CODE (arg) == UNBOUND_CLASS_TEMPLATE);
if (TREE_CODE (arg) == UNBOUND_CLASS_TEMPLATE)
return make_typename_type (TYPE_CONTEXT (arg),
build_nt (TEMPLATE_ID_EXPR,
TYPE_IDENTIFIER (arg),
argvec),
typename_type,
complain);
if (TREE_CODE (arg) == TEMPLATE_TEMPLATE_PARM)
arg = TYPE_NAME (arg);
r = lookup_template_class (arg,
argvec, in_decl,
DECL_CONTEXT (arg),
0,
complain);
return cp_build_qualified_type_real
(r, cp_type_quals (t) | cp_type_quals (r), complain);
}
else if (code == TEMPLATE_TEMPLATE_PARM)
return arg;
else
return convert_from_reference (unshare_expr (arg));
}
if (level == 1)
return t;
switch (code)
{
case TEMPLATE_TYPE_PARM:
case TEMPLATE_TEMPLATE_PARM:
case BOUND_TEMPLATE_TEMPLATE_PARM:
if (cp_type_quals (t))
{
r = tsubst (TYPE_MAIN_VARIANT (t), args, complain, in_decl);
r = cp_build_qualified_type_real
(r, cp_type_quals (t),
complain | (code == TEMPLATE_TYPE_PARM
? tf_ignore_bad_quals : 0));
}
else if (TREE_CODE (t) == TEMPLATE_TYPE_PARM
&& PLACEHOLDER_TYPE_CONSTRAINTS (t)
&& (r = (TEMPLATE_PARM_DESCENDANTS
(TEMPLATE_TYPE_PARM_INDEX (t))))
&& (r = TREE_TYPE (r))
&& !PLACEHOLDER_TYPE_CONSTRAINTS (r))
;
else if (TREE_CODE (t) == TEMPLATE_TYPE_PARM
&& !PLACEHOLDER_TYPE_CONSTRAINTS (t)
&& !CLASS_PLACEHOLDER_TEMPLATE (t)
&& (arg = TEMPLATE_TYPE_PARM_INDEX (t),
r = TEMPLATE_PARM_DESCENDANTS (arg))
&& (TEMPLATE_PARM_LEVEL (r)
== TEMPLATE_PARM_LEVEL (arg) - levels))
r = TREE_TYPE (r);
else
{
r = copy_type (t);
TEMPLATE_TYPE_PARM_INDEX (r)
= reduce_template_parm_level (TEMPLATE_TYPE_PARM_INDEX (t),
r, levels, args, complain);
TYPE_STUB_DECL (r) = TYPE_NAME (r) = TEMPLATE_TYPE_DECL (r);
TYPE_MAIN_VARIANT (r) = r;
TYPE_POINTER_TO (r) = NULL_TREE;
TYPE_REFERENCE_TO (r) = NULL_TREE;
if (TREE_CODE (t) == TEMPLATE_TYPE_PARM)
{
if (tree constr = PLACEHOLDER_TYPE_CONSTRAINTS (t))
PLACEHOLDER_TYPE_CONSTRAINTS (r)
= tsubst_constraint (constr, args, complain, in_decl);
else if (tree pl = CLASS_PLACEHOLDER_TEMPLATE (t))
{
pl = tsubst_copy (pl, args, complain, in_decl);
CLASS_PLACEHOLDER_TEMPLATE (r) = pl;
}
}
if (TREE_CODE (r) == TEMPLATE_TEMPLATE_PARM)
SET_TYPE_STRUCTURAL_EQUALITY (r);
else if (TYPE_STRUCTURAL_EQUALITY_P (t))
SET_TYPE_STRUCTURAL_EQUALITY (r);
else
TYPE_CANONICAL (r) = canonical_type_parameter (r);
if (code == BOUND_TEMPLATE_TEMPLATE_PARM)
{
tree tinfo = TYPE_TEMPLATE_INFO (t);
tree tmpl = tsubst (TI_TEMPLATE (tinfo), args,
complain, in_decl);
if (tmpl == error_mark_node)
return error_mark_node;
tree argvec = tsubst (TI_ARGS (tinfo), args,
complain, in_decl);
if (argvec == error_mark_node)
return error_mark_node;
TEMPLATE_TEMPLATE_PARM_TEMPLATE_INFO (r)
= build_template_info (tmpl, argvec);
}
}
break;
case TEMPLATE_PARM_INDEX:
type = tsubst (type, args, complain, in_decl);
if (type == error_mark_node)
return error_mark_node;
r = reduce_template_parm_level (t, type, levels, args, complain);
break;
default:
gcc_unreachable ();
}
return r;
}
case TREE_LIST:
{
tree purpose, value, chain;
if (t == void_list_node)
return t;
purpose = TREE_PURPOSE (t);
if (purpose)
{
purpose = tsubst (purpose, args, complain, in_decl);
if (purpose == error_mark_node)
return error_mark_node;
}
value = TREE_VALUE (t);
if (value)
{
value = tsubst (value, args, complain, in_decl);
if (value == error_mark_node)
return error_mark_node;
}
chain = TREE_CHAIN (t);
if (chain && chain != void_type_node)
{
chain = tsubst (chain, args, complain, in_decl);
if (chain == error_mark_node)
return error_mark_node;
}
if (purpose == TREE_PURPOSE (t)
&& value == TREE_VALUE (t)
&& chain == TREE_CHAIN (t))
return t;
return hash_tree_cons (purpose, value, chain);
}
case TREE_BINFO:
gcc_unreachable ();
case TREE_VEC:
gcc_assert (!type);
return tsubst_template_args (t, args, complain, in_decl);
case POINTER_TYPE:
case REFERENCE_TYPE:
{
if (type == TREE_TYPE (t) && TREE_CODE (type) != METHOD_TYPE)
return t;
if ((TREE_CODE (type) == REFERENCE_TYPE
&& (((cxx_dialect == cxx98) && flag_iso) || code != REFERENCE_TYPE))
|| (code == REFERENCE_TYPE && VOID_TYPE_P (type)))
{
static location_t last_loc;
if (complain & tf_error
&& last_loc != input_location)
{
if (VOID_TYPE_P (type))
error ("forming reference to void");
else if (code == POINTER_TYPE)
error ("forming pointer to reference type %qT", type);
else
error ("forming reference to reference type %qT", type);
last_loc = input_location;
}
return error_mark_node;
}
else if (TREE_CODE (type) == FUNCTION_TYPE
&& (type_memfn_quals (type) != TYPE_UNQUALIFIED
|| type_memfn_rqual (type) != REF_QUAL_NONE))
{
if (complain & tf_error)
{
if (code == POINTER_TYPE)
error ("forming pointer to qualified function type %qT",
type);
else
error ("forming reference to qualified function type %qT",
type);
}
return error_mark_node;
}
else if (code == POINTER_TYPE)
{
r = build_pointer_type (type);
if (TREE_CODE (type) == METHOD_TYPE)
r = build_ptrmemfunc_type (r);
}
else if (TREE_CODE (type) == REFERENCE_TYPE)
r = cp_build_reference_type
(TREE_TYPE (type),
TYPE_REF_IS_RVALUE (t) && TYPE_REF_IS_RVALUE (type));
else
r = cp_build_reference_type (type, TYPE_REF_IS_RVALUE (t));
r = cp_build_qualified_type_real (r, cp_type_quals (t), complain);
if (r != error_mark_node)
layout_type (r);
return r;
}
case OFFSET_TYPE:
{
r = tsubst (TYPE_OFFSET_BASETYPE (t), args, complain, in_decl);
if (r == error_mark_node || !MAYBE_CLASS_TYPE_P (r))
{
if (complain & tf_error)
error ("creating pointer to member of non-class type %qT", r);
return error_mark_node;
}
if (TREE_CODE (type) == REFERENCE_TYPE)
{
if (complain & tf_error)
error ("creating pointer to member reference type %qT", type);
return error_mark_node;
}
if (VOID_TYPE_P (type))
{
if (complain & tf_error)
error ("creating pointer to member of type void");
return error_mark_node;
}
gcc_assert (TREE_CODE (type) != METHOD_TYPE);
if (TREE_CODE (type) == FUNCTION_TYPE)
{
tree memptr;
tree method_type
= build_memfn_type (type, r, type_memfn_quals (type),
type_memfn_rqual (type));
memptr = build_ptrmemfunc_type (build_pointer_type (method_type));
return cp_build_qualified_type_real (memptr, cp_type_quals (t),
complain);
}
else
return cp_build_qualified_type_real (build_ptrmem_type (r, type),
cp_type_quals (t),
complain);
}
case FUNCTION_TYPE:
case METHOD_TYPE:
{
tree fntype;
tree specs;
fntype = tsubst_function_type (t, args, complain, in_decl);
if (fntype == error_mark_node)
return error_mark_node;
specs = tsubst_exception_specification (t, args, complain, in_decl,
fndecl_type);
if (specs == error_mark_node)
return error_mark_node;
if (specs)
fntype = build_exception_variant (fntype, specs);
return fntype;
}
case ARRAY_TYPE:
{
tree domain = tsubst (TYPE_DOMAIN (t), args, complain, in_decl);
if (domain == error_mark_node)
return error_mark_node;
if (type == TREE_TYPE (t) && domain == TYPE_DOMAIN (t))
return t;
if (VOID_TYPE_P (type)
|| TREE_CODE (type) == FUNCTION_TYPE
|| (TREE_CODE (type) == ARRAY_TYPE
&& TYPE_DOMAIN (type) == NULL_TREE)
|| TREE_CODE (type) == REFERENCE_TYPE)
{
if (complain & tf_error)
error ("creating array of %qT", type);
return error_mark_node;
}
if (abstract_virtuals_error_sfinae (ACU_ARRAY, type, complain))
return error_mark_node;
r = build_cplus_array_type (type, domain);
if (TYPE_USER_ALIGN (t))
{
SET_TYPE_ALIGN (r, TYPE_ALIGN (t));
TYPE_USER_ALIGN (r) = 1;
}
return r;
}
case TYPENAME_TYPE:
{
tree ctx = tsubst_aggr_type (TYPE_CONTEXT (t), args, complain,
in_decl, 1);
if (ctx == error_mark_node)
return error_mark_node;
tree f = tsubst_copy (TYPENAME_TYPE_FULLNAME (t), args,
complain, in_decl);
if (f == error_mark_node)
return error_mark_node;
if (!MAYBE_CLASS_TYPE_P (ctx))
{
if (complain & tf_error)
error ("%qT is not a class, struct, or union type", ctx);
return error_mark_node;
}
else if (!uses_template_parms (ctx) && !TYPE_BEING_DEFINED (ctx))
{
ctx = complete_type (ctx);
if (!COMPLETE_TYPE_P (ctx))
{
if (complain & tf_error)
cxx_incomplete_type_error (NULL_TREE, ctx);
return error_mark_node;
}
}
f = make_typename_type (ctx, f, typename_type,
complain | tf_keep_type_decl);
if (f == error_mark_node)
return f;
if (TREE_CODE (f) == TYPE_DECL)
{
complain |= tf_ignore_bad_quals;
f = TREE_TYPE (f);
}
if (TREE_CODE (f) != TYPENAME_TYPE)
{
if (TYPENAME_IS_ENUM_P (t) && TREE_CODE (f) != ENUMERAL_TYPE)
{
if (complain & tf_error)
error ("%qT resolves to %qT, which is not an enumeration type",
t, f);
else
return error_mark_node;
}
else if (TYPENAME_IS_CLASS_P (t) && !CLASS_TYPE_P (f))
{
if (complain & tf_error)
error ("%qT resolves to %qT, which is is not a class type",
t, f);
else
return error_mark_node;
}
}
return cp_build_qualified_type_real
(f, cp_type_quals (f) | cp_type_quals (t), complain);
}
case UNBOUND_CLASS_TEMPLATE:
{
tree ctx = tsubst_aggr_type (TYPE_CONTEXT (t), args, complain,
in_decl, 1);
tree name = TYPE_IDENTIFIER (t);
tree parm_list = DECL_TEMPLATE_PARMS (TYPE_NAME (t));
if (ctx == error_mark_node || name == error_mark_node)
return error_mark_node;
if (parm_list)
parm_list = tsubst_template_parms (parm_list, args, complain);
return make_unbound_class_template (ctx, name, parm_list, complain);
}
case TYPEOF_TYPE:
{
tree type;
++cp_unevaluated_operand;
++c_inhibit_evaluation_warnings;
type = tsubst_expr (TYPEOF_TYPE_EXPR (t), args,
complain, in_decl,
false);
--cp_unevaluated_operand;
--c_inhibit_evaluation_warnings;
type = finish_typeof (type);
return cp_build_qualified_type_real (type,
cp_type_quals (t)
| cp_type_quals (type),
complain);
}
case DECLTYPE_TYPE:
{
tree type;
++cp_unevaluated_operand;
++c_inhibit_evaluation_warnings;
type = tsubst_copy_and_build (DECLTYPE_TYPE_EXPR (t), args,
complain|tf_decltype, in_decl,
false,
false);
if (DECLTYPE_FOR_INIT_CAPTURE (t))
{
if (type == NULL_TREE)
{
if (complain & tf_error)
error ("empty initializer in lambda init-capture");
type = error_mark_node;
}
else if (TREE_CODE (type) == TREE_LIST)
type = build_x_compound_expr_from_list (type, ELK_INIT, complain);
}
--cp_unevaluated_operand;
--c_inhibit_evaluation_warnings;
if (DECLTYPE_FOR_LAMBDA_CAPTURE (t))
type = lambda_capture_field_type (type,
DECLTYPE_FOR_INIT_CAPTURE (t),
DECLTYPE_FOR_REF_CAPTURE (t));
else if (DECLTYPE_FOR_LAMBDA_PROXY (t))
type = lambda_proxy_type (type);
else
{
bool id = DECLTYPE_TYPE_ID_EXPR_OR_MEMBER_ACCESS_P (t);
if (id && TREE_CODE (DECLTYPE_TYPE_EXPR (t)) == BIT_NOT_EXPR
&& EXPR_P (type))
id = false;
type = finish_decltype_type (type, id, complain);
}
return cp_build_qualified_type_real (type,
cp_type_quals (t)
| cp_type_quals (type),
complain | tf_ignore_bad_quals);
}
case UNDERLYING_TYPE:
{
tree type = tsubst (UNDERLYING_TYPE_TYPE (t), args,
complain, in_decl);
return finish_underlying_type (type);
}
case TYPE_ARGUMENT_PACK:
case NONTYPE_ARGUMENT_PACK:
{
tree r;
if (code == NONTYPE_ARGUMENT_PACK)
r = make_node (code);
else
r = cxx_make_type (code);
tree pack_args = ARGUMENT_PACK_ARGS (t);
pack_args = tsubst_template_args (pack_args, args, complain, in_decl);
SET_ARGUMENT_PACK_ARGS (r, pack_args);
return r;
}
case VOID_CST:
case INTEGER_CST:
case REAL_CST:
case STRING_CST:
case PLUS_EXPR:
case MINUS_EXPR:
case NEGATE_EXPR:
case NOP_EXPR:
case INDIRECT_REF:
case ADDR_EXPR:
case CALL_EXPR:
case ARRAY_REF:
case SCOPE_REF:
gcc_unreachable ();
default:
sorry ("use of %qs in template", get_tree_code_name (code));
return error_mark_node;
}
}
static tree
tsubst_baselink (tree baselink, tree object_type,
tree args, tsubst_flags_t complain, tree in_decl)
{
bool qualified_p = BASELINK_QUALIFIED_P (baselink);
tree qualifying_scope = BINFO_TYPE (BASELINK_ACCESS_BINFO (baselink));
qualifying_scope = tsubst (qualifying_scope, args, complain, in_decl);
tree optype = BASELINK_OPTYPE (baselink);
optype = tsubst (optype, args, complain, in_decl);
tree template_args = NULL_TREE;
bool template_id_p = false;
tree fns = BASELINK_FUNCTIONS (baselink);
if (TREE_CODE (fns) == TEMPLATE_ID_EXPR)
{
template_id_p = true;
template_args = TREE_OPERAND (fns, 1);
fns = TREE_OPERAND (fns, 0);
if (template_args)
template_args = tsubst_template_args (template_args, args,
complain, in_decl);
}
tree binfo_type = BINFO_TYPE (BASELINK_BINFO (baselink));
binfo_type = tsubst (binfo_type, args, complain, in_decl);
bool dependent_p = binfo_type != BINFO_TYPE (BASELINK_BINFO (baselink));
if (dependent_p)
{
tree name = OVL_NAME (fns);
if (IDENTIFIER_CONV_OP_P (name))
name = make_conv_op_name (optype);
if (name == complete_dtor_identifier)
dependent_p = false;
baselink = lookup_fnfields (qualifying_scope, name, 1);
if (!baselink)
{
if ((complain & tf_error)
&& constructor_name_p (name, qualifying_scope))
error ("cannot call constructor %<%T::%D%> directly",
qualifying_scope, name);
return error_mark_node;
}
if (BASELINK_P (baselink))
fns = BASELINK_FUNCTIONS (baselink);
}
else
baselink = copy_node (baselink);
if (!template_id_p && !really_overloaded_fn (fns))
{
tree fn = OVL_FIRST (fns);
bool ok = mark_used (fn, complain);
if (!ok && !(complain & tf_error))
return error_mark_node;
if (ok && BASELINK_P (baselink))
TREE_TYPE (baselink) = TREE_TYPE (fn);
}
if (BASELINK_P (baselink))
{
if (template_id_p)
BASELINK_FUNCTIONS (baselink)
= build2 (TEMPLATE_ID_EXPR, unknown_type_node, fns, template_args);
BASELINK_OPTYPE (baselink) = optype;
}
if (!object_type)
object_type = current_class_type;
if (qualified_p || !dependent_p)
{
baselink = adjust_result_of_qualified_name_lookup (baselink,
qualifying_scope,
object_type);
if (!qualified_p)
BASELINK_QUALIFIED_P (baselink) = false;
}
return baselink;
}
static tree
tsubst_qualified_id (tree qualified_id, tree args,
tsubst_flags_t complain, tree in_decl,
bool done, bool address_p)
{
tree expr;
tree scope;
tree name;
bool is_template;
tree template_args;
location_t loc = UNKNOWN_LOCATION;
gcc_assert (TREE_CODE (qualified_id) == SCOPE_REF);
name = TREE_OPERAND (qualified_id, 1);
if (TREE_CODE (name) == TEMPLATE_ID_EXPR)
{
is_template = true;
loc = EXPR_LOCATION (name);
template_args = TREE_OPERAND (name, 1);
if (template_args)
template_args = tsubst_template_args (template_args, args,
complain, in_decl);
if (template_args == error_mark_node)
return error_mark_node;
name = TREE_OPERAND (name, 0);
}
else
{
is_template = false;
template_args = NULL_TREE;
}
scope = TREE_OPERAND (qualified_id, 0);
if (args)
{
scope = tsubst (scope, args, complain, in_decl);
expr = tsubst_copy (name, args, complain, in_decl);
}
else
expr = name;
if (dependent_scope_p (scope))
{
if (is_template)
expr = build_min_nt_loc (loc, TEMPLATE_ID_EXPR, expr, template_args);
tree r = build_qualified_name (NULL_TREE, scope, expr,
QUALIFIED_NAME_IS_TEMPLATE (qualified_id));
REF_PARENTHESIZED_P (r) = REF_PARENTHESIZED_P (qualified_id);
return r;
}
if (!BASELINK_P (name) && !DECL_P (expr))
{
if (TREE_CODE (expr) == BIT_NOT_EXPR)
{
if (!check_dtor_name (scope, TREE_OPERAND (expr, 0)))
{
error ("qualifying type %qT does not match destructor name ~%qT",
scope, TREE_OPERAND (expr, 0));
expr = error_mark_node;
}
else
expr = lookup_qualified_name (scope, complete_dtor_identifier,
0, false);
}
else
expr = lookup_qualified_name (scope, expr, 0, false);
if (TREE_CODE (TREE_CODE (expr) == TEMPLATE_DECL
? DECL_TEMPLATE_RESULT (expr) : expr) == TYPE_DECL)
{
if (complain & tf_error)
{
error ("dependent-name %qE is parsed as a non-type, but "
"instantiation yields a type", qualified_id);
inform (input_location, "say %<typename %E%> if a type is meant", qualified_id);
}
return error_mark_node;
}
}
if (DECL_P (expr))
{
check_accessibility_of_qualified_id (expr, NULL_TREE,
scope);
if (!mark_used (expr, complain) && !(complain & tf_error))
return error_mark_node;
}
if (expr == error_mark_node || TREE_CODE (expr) == TREE_LIST)
{
if (complain & tf_error)
qualified_name_lookup_error (scope,
TREE_OPERAND (qualified_id, 1),
expr, input_location);
return error_mark_node;
}
if (is_template)
{
if (flag_concepts && check_auto_in_tmpl_args (expr, template_args))
return error_mark_node;
if (variable_template_p (expr))
expr = lookup_and_finish_template_variable (expr, template_args,
complain);
else
expr = lookup_template_function (expr, template_args);
}
if (expr == error_mark_node && complain & tf_error)
qualified_name_lookup_error (scope, TREE_OPERAND (qualified_id, 1),
expr, input_location);
else if (TYPE_P (scope))
{
expr = (adjust_result_of_qualified_name_lookup
(expr, scope, current_nonlambda_class_type ()));
expr = (finish_qualified_id_expr
(scope, expr, done, address_p && PTRMEM_OK_P (qualified_id),
QUALIFIED_NAME_IS_TEMPLATE (qualified_id),
false, complain));
}
if (TREE_CODE (expr) != SCOPE_REF
&& TREE_CODE (expr) != OFFSET_REF)
expr = convert_from_reference (expr);
if (REF_PARENTHESIZED_P (qualified_id))
expr = force_paren_expr (expr);
return expr;
}
static tree
tsubst_init (tree init, tree decl, tree args,
tsubst_flags_t complain, tree in_decl)
{
if (!init)
return NULL_TREE;
init = tsubst_expr (init, args, complain, in_decl, false);
if (!init && TREE_TYPE (decl) != error_mark_node)
{
init = build_value_init (TREE_TYPE (decl),
complain);
if (TREE_CODE (init) == AGGR_INIT_EXPR)
init = get_target_expr_sfinae (init, complain);
if (TREE_CODE (init) == TARGET_EXPR)
TARGET_EXPR_DIRECT_INIT_P (init) = true;
}
return init;
}
static tree
tsubst_copy (tree t, tree args, tsubst_flags_t complain, tree in_decl)
{
enum tree_code code;
tree r;
if (t == NULL_TREE || t == error_mark_node || args == NULL_TREE)
return t;
code = TREE_CODE (t);
switch (code)
{
case PARM_DECL:
r = retrieve_local_specialization (t);
if (r == NULL_TREE)
{
if (DECL_NAME (t) == this_identifier && current_class_ptr)
return current_class_ptr;
gcc_assert (cp_unevaluated_operand != 0);
r = tsubst_decl (t, args, complain);
DECL_CONTEXT (r) = DECL_CONTEXT (t);
}
if (TREE_CODE (r) == ARGUMENT_PACK_SELECT)
r = argument_pack_select_arg (r);
if (!mark_used (r, complain) && !(complain & tf_error))
return error_mark_node;
return r;
case CONST_DECL:
{
tree enum_type;
tree v;
if (DECL_TEMPLATE_PARM_P (t))
return tsubst_copy (DECL_INITIAL (t), args, complain, in_decl);
if (DECL_NAMESPACE_SCOPE_P (t))
return t;
if (args == NULL_TREE)
return scalar_constant_value (t);
enum_type
= tsubst_aggr_type (DECL_CONTEXT (t), args, complain, in_decl,
0);
for (v = TYPE_VALUES (enum_type);
v != NULL_TREE;
v = TREE_CHAIN (v))
if (TREE_PURPOSE (v) == DECL_NAME (t))
return TREE_VALUE (v);
gcc_unreachable ();
}
return t;
case FIELD_DECL:
if (DECL_CONTEXT (t))
{
tree ctx;
ctx = tsubst_aggr_type (DECL_CONTEXT (t), args, complain, in_decl,
1);
if (ctx != DECL_CONTEXT (t))
{
tree r = lookup_field (ctx, DECL_NAME (t), 0, false);
if (!r)
{
if (complain & tf_error)
error ("using invalid field %qD", t);
return error_mark_node;
}
return r;
}
}
return t;
case VAR_DECL:
case FUNCTION_DECL:
if (DECL_LANG_SPECIFIC (t) && DECL_TEMPLATE_INFO (t))
r = tsubst (t, args, complain, in_decl);
else if (local_variable_p (t)
&& uses_template_parms (DECL_CONTEXT (t)))
{
r = retrieve_local_specialization (t);
if (r == NULL_TREE)
{
r = lookup_name (DECL_NAME (t));
if (r && !is_capture_proxy (r))
{
tree ctx = enclosing_instantiation_of (DECL_CONTEXT (t));
if (ctx != DECL_CONTEXT (r))
r = NULL_TREE;
}
if (r)
;
else
{
r = tsubst_decl (t, args, complain);
if (local_specializations)
register_local_specialization (r, t);
if (decl_maybe_constant_var_p (r))
{
tree init = tsubst_init (DECL_INITIAL (t), r, args,
complain, in_decl);
if (!processing_template_decl)
init = maybe_constant_init (init);
if (processing_template_decl
? potential_constant_expression (init)
: reduced_constant_expression_p (init))
DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P (r)
= TREE_CONSTANT (r) = true;
DECL_INITIAL (r) = init;
if (tree auto_node = type_uses_auto (TREE_TYPE (r)))
TREE_TYPE (r)
= do_auto_deduction (TREE_TYPE (r), init, auto_node,
complain, adc_variable_type);
}
gcc_assert (cp_unevaluated_operand || TREE_STATIC (r)
|| decl_constant_var_p (r)
|| errorcount || sorrycount);
if (!processing_template_decl
&& !TREE_STATIC (r))
r = process_outer_var_ref (r, complain);
}
if (local_specializations)
register_local_specialization (r, t);
}
if (TREE_CODE (r) == ARGUMENT_PACK_SELECT)
r = argument_pack_select_arg (r);
}
else
r = t;
if (!mark_used (r, complain))
return error_mark_node;
return r;
case NAMESPACE_DECL:
return t;
case OVERLOAD:
gcc_assert (!uses_template_parms (t));
gcc_assert (!OVL_LOOKUP_P (t) || OVL_USED_P (t));
return t;
case BASELINK:
return tsubst_baselink (t, current_nonlambda_class_type (),
args, complain, in_decl);
case TEMPLATE_DECL:
if (DECL_TEMPLATE_TEMPLATE_PARM_P (t))
return tsubst (TREE_TYPE (DECL_TEMPLATE_RESULT (t)),
args, complain, in_decl);
else if (DECL_FUNCTION_TEMPLATE_P (t) && DECL_MEMBER_TEMPLATE_P (t))
return tsubst (t, args, complain, in_decl);
else if (DECL_CLASS_SCOPE_P (t)
&& uses_template_parms (DECL_CONTEXT (t)))
{
tree context = tsubst (DECL_CONTEXT (t), args, complain, in_decl);
if (dependent_scope_p (context))
{
tree type = tsubst (TREE_TYPE (t), args, complain, in_decl);
return build_qualified_name (type, context, DECL_NAME (t),
true);
}
return lookup_field (context, DECL_NAME(t), 0, false);
}
else
return t;
case NON_LVALUE_EXPR:
case VIEW_CONVERT_EXPR:
{
gcc_assert (location_wrapper_p (t));
tree op0 = tsubst_copy (TREE_OPERAND (t, 0), args, complain, in_decl);
return maybe_wrap_with_location (op0, EXPR_LOCATION (t));
}
case CAST_EXPR:
case REINTERPRET_CAST_EXPR:
case CONST_CAST_EXPR:
case STATIC_CAST_EXPR:
case DYNAMIC_CAST_EXPR:
case IMPLICIT_CONV_EXPR:
case CONVERT_EXPR:
case NOP_EXPR:
{
tree type = tsubst (TREE_TYPE (t), args, complain, in_decl);
tree op0 = tsubst_copy (TREE_OPERAND (t, 0), args, complain, in_decl);
return build1 (code, type, op0);
}
case SIZEOF_EXPR:
if (PACK_EXPANSION_P (TREE_OPERAND (t, 0))
|| ARGUMENT_PACK_P (TREE_OPERAND (t, 0)))
{
tree expanded, op = TREE_OPERAND (t, 0);
int len = 0;
if (SIZEOF_EXPR_TYPE_P (t))
op = TREE_TYPE (op);
++cp_unevaluated_operand;
++c_inhibit_evaluation_warnings;
if (PACK_EXPANSION_P (op))
expanded = tsubst_pack_expansion (op, args, complain, in_decl);
else
expanded = tsubst_template_args (ARGUMENT_PACK_ARGS (op),
args, complain, in_decl);
--cp_unevaluated_operand;
--c_inhibit_evaluation_warnings;
if (TREE_CODE (expanded) == TREE_VEC)
{
len = TREE_VEC_LENGTH (expanded);
for (int i = 0; i < len; i++)
if (DECL_P (TREE_VEC_ELT (expanded, i)))
TREE_USED (TREE_VEC_ELT (expanded, i)) = true;
}
if (expanded == error_mark_node)
return error_mark_node;
else if (PACK_EXPANSION_P (expanded)
|| (TREE_CODE (expanded) == TREE_VEC
&& pack_expansion_args_count (expanded)))
{
if (PACK_EXPANSION_P (expanded))
;
else if (TREE_VEC_LENGTH (expanded) == 1)
expanded = TREE_VEC_ELT (expanded, 0);
else
expanded = make_argument_pack (expanded);
if (TYPE_P (expanded))
return cxx_sizeof_or_alignof_type (expanded, SIZEOF_EXPR,
false,
complain & tf_error);
else
return cxx_sizeof_or_alignof_expr (expanded, SIZEOF_EXPR,
complain & tf_error);
}
else
return build_int_cst (size_type_node, len);
}
if (SIZEOF_EXPR_TYPE_P (t))
{
r = tsubst (TREE_TYPE (TREE_OPERAND (t, 0)),
args, complain, in_decl);
r = build1 (NOP_EXPR, r, error_mark_node);
r = build1 (SIZEOF_EXPR,
tsubst (TREE_TYPE (t), args, complain, in_decl), r);
SIZEOF_EXPR_TYPE_P (r) = 1;
return r;
}
case INDIRECT_REF:
case NEGATE_EXPR:
case TRUTH_NOT_EXPR:
case BIT_NOT_EXPR:
case ADDR_EXPR:
case UNARY_PLUS_EXPR:      
case ALIGNOF_EXPR:
case AT_ENCODE_EXPR:
case ARROW_EXPR:
case THROW_EXPR:
case TYPEID_EXPR:
case REALPART_EXPR:
case IMAGPART_EXPR:
case PAREN_EXPR:
{
tree type = tsubst (TREE_TYPE (t), args, complain, in_decl);
tree op0 = tsubst_copy (TREE_OPERAND (t, 0), args, complain, in_decl);
r = build1 (code, type, op0);
if (code == ALIGNOF_EXPR)
ALIGNOF_EXPR_STD_P (r) = ALIGNOF_EXPR_STD_P (t);
return r;
}
case COMPONENT_REF:
{
tree object;
tree name;
object = tsubst_copy (TREE_OPERAND (t, 0), args, complain, in_decl);
name = TREE_OPERAND (t, 1);
if (TREE_CODE (name) == BIT_NOT_EXPR)
{
name = tsubst_copy (TREE_OPERAND (name, 0), args,
complain, in_decl);
name = build1 (BIT_NOT_EXPR, NULL_TREE, name);
}
else if (TREE_CODE (name) == SCOPE_REF
&& TREE_CODE (TREE_OPERAND (name, 1)) == BIT_NOT_EXPR)
{
tree base = tsubst_copy (TREE_OPERAND (name, 0), args,
complain, in_decl);
name = TREE_OPERAND (name, 1);
name = tsubst_copy (TREE_OPERAND (name, 0), args,
complain, in_decl);
name = build1 (BIT_NOT_EXPR, NULL_TREE, name);
name = build_qualified_name (NULL_TREE,
base, name,
false);
}
else if (BASELINK_P (name))
name = tsubst_baselink (name,
non_reference (TREE_TYPE (object)),
args, complain,
in_decl);
else
name = tsubst_copy (name, args, complain, in_decl);
return build_nt (COMPONENT_REF, object, name, NULL_TREE);
}
case PLUS_EXPR:
case MINUS_EXPR:
case MULT_EXPR:
case TRUNC_DIV_EXPR:
case CEIL_DIV_EXPR:
case FLOOR_DIV_EXPR:
case ROUND_DIV_EXPR:
case EXACT_DIV_EXPR:
case BIT_AND_EXPR:
case BIT_IOR_EXPR:
case BIT_XOR_EXPR:
case TRUNC_MOD_EXPR:
case FLOOR_MOD_EXPR:
case TRUTH_ANDIF_EXPR:
case TRUTH_ORIF_EXPR:
case TRUTH_AND_EXPR:
case TRUTH_OR_EXPR:
case RSHIFT_EXPR:
case LSHIFT_EXPR:
case RROTATE_EXPR:
case LROTATE_EXPR:
case EQ_EXPR:
case NE_EXPR:
case MAX_EXPR:
case MIN_EXPR:
case LE_EXPR:
case GE_EXPR:
case LT_EXPR:
case GT_EXPR:
case COMPOUND_EXPR:
case DOTSTAR_EXPR:
case MEMBER_REF:
case PREDECREMENT_EXPR:
case PREINCREMENT_EXPR:
case POSTDECREMENT_EXPR:
case POSTINCREMENT_EXPR:
{
tree op0 = tsubst_copy (TREE_OPERAND (t, 0), args, complain, in_decl);
tree op1 = tsubst_copy (TREE_OPERAND (t, 1), args, complain, in_decl);
return build_nt (code, op0, op1);
}
case SCOPE_REF:
{
tree op0 = tsubst_copy (TREE_OPERAND (t, 0), args, complain, in_decl);
tree op1 = tsubst_copy (TREE_OPERAND (t, 1), args, complain, in_decl);
return build_qualified_name (NULL_TREE, op0, op1,
QUALIFIED_NAME_IS_TEMPLATE (t));
}
case ARRAY_REF:
{
tree op0 = tsubst_copy (TREE_OPERAND (t, 0), args, complain, in_decl);
tree op1 = tsubst_copy (TREE_OPERAND (t, 1), args, complain, in_decl);
return build_nt (ARRAY_REF, op0, op1, NULL_TREE, NULL_TREE);
}
case CALL_EXPR:
{
int n = VL_EXP_OPERAND_LENGTH (t);
tree result = build_vl_exp (CALL_EXPR, n);
int i;
for (i = 0; i < n; i++)
TREE_OPERAND (t, i) = tsubst_copy (TREE_OPERAND (t, i), args,
complain, in_decl);
return result;
}
case COND_EXPR:
case MODOP_EXPR:
case PSEUDO_DTOR_EXPR:
case VEC_PERM_EXPR:
{
tree op0 = tsubst_copy (TREE_OPERAND (t, 0), args, complain, in_decl);
tree op1 = tsubst_copy (TREE_OPERAND (t, 1), args, complain, in_decl);
tree op2 = tsubst_copy (TREE_OPERAND (t, 2), args, complain, in_decl);
r = build_nt (code, op0, op1, op2);
TREE_NO_WARNING (r) = TREE_NO_WARNING (t);
return r;
}
case NEW_EXPR:
{
tree op0 = tsubst_copy (TREE_OPERAND (t, 0), args, complain, in_decl);
tree op1 = tsubst_copy (TREE_OPERAND (t, 1), args, complain, in_decl);
tree op2 = tsubst_copy (TREE_OPERAND (t, 2), args, complain, in_decl);
r = build_nt (code, op0, op1, op2);
NEW_EXPR_USE_GLOBAL (r) = NEW_EXPR_USE_GLOBAL (t);
return r;
}
case DELETE_EXPR:
{
tree op0 = tsubst_copy (TREE_OPERAND (t, 0), args, complain, in_decl);
tree op1 = tsubst_copy (TREE_OPERAND (t, 1), args, complain, in_decl);
r = build_nt (code, op0, op1);
DELETE_EXPR_USE_GLOBAL (r) = DELETE_EXPR_USE_GLOBAL (t);
DELETE_EXPR_USE_VEC (r) = DELETE_EXPR_USE_VEC (t);
return r;
}
case TEMPLATE_ID_EXPR:
{
tree fn = TREE_OPERAND (t, 0);
tree targs = TREE_OPERAND (t, 1);
fn = tsubst_copy (fn, args, complain, in_decl);
if (targs)
targs = tsubst_template_args (targs, args, complain, in_decl);
return lookup_template_function (fn, targs);
}
case TREE_LIST:
{
tree purpose, value, chain;
if (t == void_list_node)
return t;
purpose = TREE_PURPOSE (t);
if (purpose)
purpose = tsubst_copy (purpose, args, complain, in_decl);
value = TREE_VALUE (t);
if (value)
value = tsubst_copy (value, args, complain, in_decl);
chain = TREE_CHAIN (t);
if (chain && chain != void_type_node)
chain = tsubst_copy (chain, args, complain, in_decl);
if (purpose == TREE_PURPOSE (t)
&& value == TREE_VALUE (t)
&& chain == TREE_CHAIN (t))
return t;
return tree_cons (purpose, value, chain);
}
case RECORD_TYPE:
case UNION_TYPE:
case ENUMERAL_TYPE:
case INTEGER_TYPE:
case TEMPLATE_TYPE_PARM:
case TEMPLATE_TEMPLATE_PARM:
case BOUND_TEMPLATE_TEMPLATE_PARM:
case TEMPLATE_PARM_INDEX:
case POINTER_TYPE:
case REFERENCE_TYPE:
case OFFSET_TYPE:
case FUNCTION_TYPE:
case METHOD_TYPE:
case ARRAY_TYPE:
case TYPENAME_TYPE:
case UNBOUND_CLASS_TEMPLATE:
case TYPEOF_TYPE:
case DECLTYPE_TYPE:
case TYPE_DECL:
return tsubst (t, args, complain, in_decl);
case USING_DECL:
t = DECL_NAME (t);
case IDENTIFIER_NODE:
if (IDENTIFIER_CONV_OP_P (t))
{
tree new_type = tsubst (TREE_TYPE (t), args, complain, in_decl);
return make_conv_op_name (new_type);
}
else
return t;
case CONSTRUCTOR:
gcc_unreachable ();
case VA_ARG_EXPR:
{
tree op0 = tsubst_copy (TREE_OPERAND (t, 0), args, complain, in_decl);
tree type = tsubst (TREE_TYPE (t), args, complain, in_decl);
return build_x_va_arg (EXPR_LOCATION (t), op0, type);
}
case CLEANUP_POINT_EXPR:
gcc_unreachable ();
case OFFSET_REF:
{
tree type = tsubst (TREE_TYPE (t), args, complain, in_decl);
tree op0 = tsubst_copy (TREE_OPERAND (t, 0), args, complain, in_decl);
tree op1 = tsubst_copy (TREE_OPERAND (t, 1), args, complain, in_decl);
r = build2 (code, type, op0, op1);
PTRMEM_OK_P (r) = PTRMEM_OK_P (t);
if (!mark_used (TREE_OPERAND (r, 1), complain)
&& !(complain & tf_error))
return error_mark_node;
return r;
}
case EXPR_PACK_EXPANSION:
error ("invalid use of pack expansion expression");
return error_mark_node;
case NONTYPE_ARGUMENT_PACK:
error ("use %<...%> to expand argument pack");
return error_mark_node;
case VOID_CST:
gcc_checking_assert (t == void_node && VOID_TYPE_P (TREE_TYPE (t)));
return t;
case INTEGER_CST:
case REAL_CST:
case STRING_CST:
case COMPLEX_CST:
{
tree type = tsubst (TREE_TYPE (t), args, complain, in_decl);
r = fold_convert (type, t);
gcc_assert (TREE_CODE (r) == code);
return r;
}
case PTRMEM_CST:
gcc_assert (!uses_template_parms (t));
return t;
case UNARY_LEFT_FOLD_EXPR:
return tsubst_unary_left_fold (t, args, complain, in_decl);
case UNARY_RIGHT_FOLD_EXPR:
return tsubst_unary_right_fold (t, args, complain, in_decl);
case BINARY_LEFT_FOLD_EXPR:
return tsubst_binary_left_fold (t, args, complain, in_decl);
case BINARY_RIGHT_FOLD_EXPR:
return tsubst_binary_right_fold (t, args, complain, in_decl);
case PREDICT_EXPR:
return t;
case DEBUG_BEGIN_STMT:
return t;
default:
if (flag_checking)
gcc_unreachable ();
return t;
}
}
static tree
tsubst_omp_clause_decl (tree decl, tree args, tsubst_flags_t complain,
tree in_decl)
{
if (decl == NULL_TREE)
return NULL_TREE;
if (TREE_CODE (decl) == TREE_LIST)
{
tree low_bound
= tsubst_expr (TREE_PURPOSE (decl), args, complain, in_decl,
false);
tree length = tsubst_expr (TREE_VALUE (decl), args, complain, in_decl,
false);
tree chain = tsubst_omp_clause_decl (TREE_CHAIN (decl), args, complain,
in_decl);
if (TREE_PURPOSE (decl) == low_bound
&& TREE_VALUE (decl) == length
&& TREE_CHAIN (decl) == chain)
return decl;
tree ret = tree_cons (low_bound, length, chain);
OMP_CLAUSE_DEPEND_SINK_NEGATIVE (ret)
= OMP_CLAUSE_DEPEND_SINK_NEGATIVE (decl);
return ret;
}
tree ret = tsubst_expr (decl, args, complain, in_decl,
false);
if (decl
&& REFERENCE_REF_P (ret)
&& !REFERENCE_REF_P (decl))
ret = TREE_OPERAND (ret, 0);
return ret;
}
static tree
tsubst_omp_clauses (tree clauses, enum c_omp_region_type ort,
tree args, tsubst_flags_t complain, tree in_decl)
{
tree new_clauses = NULL_TREE, nc, oc;
tree linear_no_step = NULL_TREE;
for (oc = clauses; oc ; oc = OMP_CLAUSE_CHAIN (oc))
{
nc = copy_node (oc);
OMP_CLAUSE_CHAIN (nc) = new_clauses;
new_clauses = nc;
switch (OMP_CLAUSE_CODE (nc))
{
case OMP_CLAUSE_LASTPRIVATE:
if (OMP_CLAUSE_LASTPRIVATE_STMT (oc))
{
OMP_CLAUSE_LASTPRIVATE_STMT (nc) = push_stmt_list ();
tsubst_expr (OMP_CLAUSE_LASTPRIVATE_STMT (oc), args, complain,
in_decl, false);
OMP_CLAUSE_LASTPRIVATE_STMT (nc)
= pop_stmt_list (OMP_CLAUSE_LASTPRIVATE_STMT (nc));
}
case OMP_CLAUSE_PRIVATE:
case OMP_CLAUSE_SHARED:
case OMP_CLAUSE_FIRSTPRIVATE:
case OMP_CLAUSE_COPYIN:
case OMP_CLAUSE_COPYPRIVATE:
case OMP_CLAUSE_UNIFORM:
case OMP_CLAUSE_DEPEND:
case OMP_CLAUSE_FROM:
case OMP_CLAUSE_TO:
case OMP_CLAUSE_MAP:
case OMP_CLAUSE_USE_DEVICE_PTR:
case OMP_CLAUSE_IS_DEVICE_PTR:
OMP_CLAUSE_DECL (nc)
= tsubst_omp_clause_decl (OMP_CLAUSE_DECL (oc), args, complain,
in_decl);
break;
case OMP_CLAUSE_TILE:
case OMP_CLAUSE_IF:
case OMP_CLAUSE_NUM_THREADS:
case OMP_CLAUSE_SCHEDULE:
case OMP_CLAUSE_COLLAPSE:
case OMP_CLAUSE_FINAL:
case OMP_CLAUSE_DEVICE:
case OMP_CLAUSE_DIST_SCHEDULE:
case OMP_CLAUSE_NUM_TEAMS:
case OMP_CLAUSE_THREAD_LIMIT:
case OMP_CLAUSE_SAFELEN:
case OMP_CLAUSE_SIMDLEN:
case OMP_CLAUSE_NUM_TASKS:
case OMP_CLAUSE_GRAINSIZE:
case OMP_CLAUSE_PRIORITY:
case OMP_CLAUSE_ORDERED:
case OMP_CLAUSE_HINT:
case OMP_CLAUSE_NUM_GANGS:
case OMP_CLAUSE_NUM_WORKERS:
case OMP_CLAUSE_VECTOR_LENGTH:
case OMP_CLAUSE_WORKER:
case OMP_CLAUSE_VECTOR:
case OMP_CLAUSE_ASYNC:
case OMP_CLAUSE_WAIT:
OMP_CLAUSE_OPERAND (nc, 0)
= tsubst_expr (OMP_CLAUSE_OPERAND (oc, 0), args, complain, 
in_decl, false);
break;
case OMP_CLAUSE_REDUCTION:
if (OMP_CLAUSE_REDUCTION_PLACEHOLDER (oc))
{
tree placeholder = OMP_CLAUSE_REDUCTION_PLACEHOLDER (oc);
if (TREE_CODE (placeholder) == SCOPE_REF)
{
tree scope = tsubst (TREE_OPERAND (placeholder, 0), args,
complain, in_decl);
OMP_CLAUSE_REDUCTION_PLACEHOLDER (nc)
= build_qualified_name (NULL_TREE, scope,
TREE_OPERAND (placeholder, 1),
false);
}
else
gcc_assert (identifier_p (placeholder));
}
OMP_CLAUSE_DECL (nc)
= tsubst_omp_clause_decl (OMP_CLAUSE_DECL (oc), args, complain,
in_decl);
break;
case OMP_CLAUSE_GANG:
case OMP_CLAUSE_ALIGNED:
OMP_CLAUSE_DECL (nc)
= tsubst_omp_clause_decl (OMP_CLAUSE_DECL (oc), args, complain,
in_decl);
OMP_CLAUSE_OPERAND (nc, 1)
= tsubst_expr (OMP_CLAUSE_OPERAND (oc, 1), args, complain,
in_decl, false);
break;
case OMP_CLAUSE_LINEAR:
OMP_CLAUSE_DECL (nc)
= tsubst_omp_clause_decl (OMP_CLAUSE_DECL (oc), args, complain,
in_decl);
if (OMP_CLAUSE_LINEAR_STEP (oc) == NULL_TREE)
{
gcc_assert (!linear_no_step);
linear_no_step = nc;
}
else if (OMP_CLAUSE_LINEAR_VARIABLE_STRIDE (oc))
OMP_CLAUSE_LINEAR_STEP (nc)
= tsubst_omp_clause_decl (OMP_CLAUSE_LINEAR_STEP (oc), args,
complain, in_decl);
else
OMP_CLAUSE_LINEAR_STEP (nc)
= tsubst_expr (OMP_CLAUSE_LINEAR_STEP (oc), args, complain,
in_decl,
false);
break;
case OMP_CLAUSE_NOWAIT:
case OMP_CLAUSE_DEFAULT:
case OMP_CLAUSE_UNTIED:
case OMP_CLAUSE_MERGEABLE:
case OMP_CLAUSE_INBRANCH:
case OMP_CLAUSE_NOTINBRANCH:
case OMP_CLAUSE_PROC_BIND:
case OMP_CLAUSE_FOR:
case OMP_CLAUSE_PARALLEL:
case OMP_CLAUSE_SECTIONS:
case OMP_CLAUSE_TASKGROUP:
case OMP_CLAUSE_NOGROUP:
case OMP_CLAUSE_THREADS:
case OMP_CLAUSE_SIMD:
case OMP_CLAUSE_DEFAULTMAP:
case OMP_CLAUSE_INDEPENDENT:
case OMP_CLAUSE_AUTO:
case OMP_CLAUSE_SEQ:
break;
default:
gcc_unreachable ();
}
if ((ort & C_ORT_OMP_DECLARE_SIMD) == C_ORT_OMP)
switch (OMP_CLAUSE_CODE (nc))
{
case OMP_CLAUSE_SHARED:
case OMP_CLAUSE_PRIVATE:
case OMP_CLAUSE_FIRSTPRIVATE:
case OMP_CLAUSE_LASTPRIVATE:
case OMP_CLAUSE_COPYPRIVATE:
case OMP_CLAUSE_LINEAR:
case OMP_CLAUSE_REDUCTION:
case OMP_CLAUSE_USE_DEVICE_PTR:
case OMP_CLAUSE_IS_DEVICE_PTR:
if (TREE_CODE (OMP_CLAUSE_DECL (oc)) == SCOPE_REF
&& (TREE_CODE (TREE_OPERAND (OMP_CLAUSE_DECL (oc), 1))
== IDENTIFIER_NODE))
{
tree t = OMP_CLAUSE_DECL (nc);
tree v = t;
while (v)
switch (TREE_CODE (v))
{
case COMPONENT_REF:
case MEM_REF:
case INDIRECT_REF:
CASE_CONVERT:
case POINTER_PLUS_EXPR:
v = TREE_OPERAND (v, 0);
continue;
case PARM_DECL:
if (DECL_CONTEXT (v) == current_function_decl
&& DECL_ARTIFICIAL (v)
&& DECL_NAME (v) == this_identifier)
OMP_CLAUSE_DECL (nc) = TREE_OPERAND (t, 1);
default:
v = NULL_TREE;
break;
}
}
else if (VAR_P (OMP_CLAUSE_DECL (oc))
&& DECL_HAS_VALUE_EXPR_P (OMP_CLAUSE_DECL (oc))
&& DECL_ARTIFICIAL (OMP_CLAUSE_DECL (oc))
&& DECL_LANG_SPECIFIC (OMP_CLAUSE_DECL (oc))
&& DECL_OMP_PRIVATIZED_MEMBER (OMP_CLAUSE_DECL (oc)))
{
tree decl = OMP_CLAUSE_DECL (nc);
if (VAR_P (decl))
{
retrofit_lang_decl (decl);
DECL_OMP_PRIVATIZED_MEMBER (decl) = 1;
}
}
break;
default:
break;
}
}
new_clauses = nreverse (new_clauses);
if (ort != C_ORT_OMP_DECLARE_SIMD)
{
new_clauses = finish_omp_clauses (new_clauses, ort);
if (linear_no_step)
for (nc = new_clauses; nc; nc = OMP_CLAUSE_CHAIN (nc))
if (nc == linear_no_step)
{
OMP_CLAUSE_LINEAR_STEP (nc) = NULL_TREE;
break;
}
}
return new_clauses;
}
static tree
tsubst_copy_asm_operands (tree t, tree args, tsubst_flags_t complain,
tree in_decl)
{
#define RECUR(t) tsubst_copy_asm_operands (t, args, complain, in_decl)
tree purpose, value, chain;
if (t == NULL)
return t;
if (TREE_CODE (t) != TREE_LIST)
return tsubst_copy_and_build (t, args, complain, in_decl,
false,
false);
if (t == void_list_node)
return t;
purpose = TREE_PURPOSE (t);
if (purpose)
purpose = RECUR (purpose);
value = TREE_VALUE (t);
if (value)
{
if (TREE_CODE (value) != LABEL_DECL)
value = RECUR (value);
else
{
value = lookup_label (DECL_NAME (value));
gcc_assert (TREE_CODE (value) == LABEL_DECL);
TREE_USED (value) = 1;
}
}
chain = TREE_CHAIN (t);
if (chain && chain != void_type_node)
chain = RECUR (chain);
return tree_cons (purpose, value, chain);
#undef RECUR
}
static tree *omp_parallel_combined_clauses;
static void
tsubst_omp_for_iterator (tree t, int i, tree declv, tree orig_declv,
tree initv, tree condv, tree incrv, tree *clauses,
tree args, tsubst_flags_t complain, tree in_decl,
bool integral_constant_expression_p)
{
#define RECUR(NODE)				\
tsubst_expr ((NODE), args, complain, in_decl,	\
integral_constant_expression_p)
tree decl, init, cond, incr;
init = TREE_VEC_ELT (OMP_FOR_INIT (t), i);
gcc_assert (TREE_CODE (init) == MODIFY_EXPR);
if (orig_declv && OMP_FOR_ORIG_DECLS (t))
{
tree o = TREE_VEC_ELT (OMP_FOR_ORIG_DECLS (t), i);
TREE_VEC_ELT (orig_declv, i) = RECUR (o);
}
decl = TREE_OPERAND (init, 0);
init = TREE_OPERAND (init, 1);
tree decl_expr = NULL_TREE;
if (init && TREE_CODE (init) == DECL_EXPR)
{
decl_expr = init;
init = DECL_INITIAL (DECL_EXPR_DECL (init));
decl = tsubst_decl (decl, args, complain);
}
else
{
if (TREE_CODE (decl) == SCOPE_REF)
{
decl = RECUR (decl);
if (TREE_CODE (decl) == COMPONENT_REF)
{
tree v = decl;
while (v)
switch (TREE_CODE (v))
{
case COMPONENT_REF:
case MEM_REF:
case INDIRECT_REF:
CASE_CONVERT:
case POINTER_PLUS_EXPR:
v = TREE_OPERAND (v, 0);
continue;
case PARM_DECL:
if (DECL_CONTEXT (v) == current_function_decl
&& DECL_ARTIFICIAL (v)
&& DECL_NAME (v) == this_identifier)
{
decl = TREE_OPERAND (decl, 1);
decl = omp_privatize_field (decl, false);
}
default:
v = NULL_TREE;
break;
}
}
}
else
decl = RECUR (decl);
}
init = RECUR (init);
tree auto_node = type_uses_auto (TREE_TYPE (decl));
if (auto_node && init)
TREE_TYPE (decl)
= do_auto_deduction (TREE_TYPE (decl), init, auto_node, complain);
gcc_assert (!type_dependent_expression_p (decl));
if (!CLASS_TYPE_P (TREE_TYPE (decl)))
{
if (decl_expr)
{
tree init_sav = DECL_INITIAL (DECL_EXPR_DECL (decl_expr));
DECL_INITIAL (DECL_EXPR_DECL (decl_expr)) = NULL_TREE;
RECUR (decl_expr);
DECL_INITIAL (DECL_EXPR_DECL (decl_expr)) = init_sav;
}
cond = RECUR (TREE_VEC_ELT (OMP_FOR_COND (t), i));
incr = TREE_VEC_ELT (OMP_FOR_INCR (t), i);
if (TREE_CODE (incr) == MODIFY_EXPR)
{
tree lhs = RECUR (TREE_OPERAND (incr, 0));
tree rhs = RECUR (TREE_OPERAND (incr, 1));
incr = build_x_modify_expr (EXPR_LOCATION (incr), lhs,
NOP_EXPR, rhs, complain);
}
else
incr = RECUR (incr);
TREE_VEC_ELT (declv, i) = decl;
TREE_VEC_ELT (initv, i) = init;
TREE_VEC_ELT (condv, i) = cond;
TREE_VEC_ELT (incrv, i) = incr;
return;
}
if (decl_expr)
{
RECUR (decl_expr);
init = NULL_TREE;
}
else if (init)
{
tree *pc;
int j;
for (j = (omp_parallel_combined_clauses == NULL ? 1 : 0); j < 2; j++)
{
for (pc = j ? clauses : omp_parallel_combined_clauses; *pc; )
{
if (OMP_CLAUSE_CODE (*pc) == OMP_CLAUSE_PRIVATE
&& OMP_CLAUSE_DECL (*pc) == decl)
break;
else if (OMP_CLAUSE_CODE (*pc) == OMP_CLAUSE_LASTPRIVATE
&& OMP_CLAUSE_DECL (*pc) == decl)
{
if (j)
break;
tree c = *pc;
*pc = OMP_CLAUSE_CHAIN (c);
OMP_CLAUSE_CHAIN (c) = *clauses;
*clauses = c;
}
else if (OMP_CLAUSE_CODE (*pc) == OMP_CLAUSE_FIRSTPRIVATE
&& OMP_CLAUSE_DECL (*pc) == decl)
{
error ("iteration variable %qD should not be firstprivate",
decl);
*pc = OMP_CLAUSE_CHAIN (*pc);
}
else if (OMP_CLAUSE_CODE (*pc) == OMP_CLAUSE_REDUCTION
&& OMP_CLAUSE_DECL (*pc) == decl)
{
error ("iteration variable %qD should not be reduction",
decl);
*pc = OMP_CLAUSE_CHAIN (*pc);
}
else
pc = &OMP_CLAUSE_CHAIN (*pc);
}
if (*pc)
break;
}
if (*pc == NULL_TREE)
{
tree c = build_omp_clause (input_location, OMP_CLAUSE_PRIVATE);
OMP_CLAUSE_DECL (c) = decl;
c = finish_omp_clauses (c, C_ORT_OMP);
if (c)
{
OMP_CLAUSE_CHAIN (c) = *clauses;
*clauses = c;
}
}
}
cond = TREE_VEC_ELT (OMP_FOR_COND (t), i);
if (COMPARISON_CLASS_P (cond))
{
tree op0 = RECUR (TREE_OPERAND (cond, 0));
tree op1 = RECUR (TREE_OPERAND (cond, 1));
cond = build2 (TREE_CODE (cond), boolean_type_node, op0, op1);
}
else
cond = RECUR (cond);
incr = TREE_VEC_ELT (OMP_FOR_INCR (t), i);
switch (TREE_CODE (incr))
{
case PREINCREMENT_EXPR:
case PREDECREMENT_EXPR:
case POSTINCREMENT_EXPR:
case POSTDECREMENT_EXPR:
incr = build2 (TREE_CODE (incr), TREE_TYPE (decl),
RECUR (TREE_OPERAND (incr, 0)), NULL_TREE);
break;
case MODIFY_EXPR:
if (TREE_CODE (TREE_OPERAND (incr, 1)) == PLUS_EXPR
|| TREE_CODE (TREE_OPERAND (incr, 1)) == MINUS_EXPR)
{
tree rhs = TREE_OPERAND (incr, 1);
tree lhs = RECUR (TREE_OPERAND (incr, 0));
tree rhs0 = RECUR (TREE_OPERAND (rhs, 0));
tree rhs1 = RECUR (TREE_OPERAND (rhs, 1));
incr = build2 (MODIFY_EXPR, TREE_TYPE (decl), lhs,
build2 (TREE_CODE (rhs), TREE_TYPE (decl),
rhs0, rhs1));
}
else
incr = RECUR (incr);
break;
case MODOP_EXPR:
if (TREE_CODE (TREE_OPERAND (incr, 1)) == PLUS_EXPR
|| TREE_CODE (TREE_OPERAND (incr, 1)) == MINUS_EXPR)
{
tree lhs = RECUR (TREE_OPERAND (incr, 0));
incr = build2 (MODIFY_EXPR, TREE_TYPE (decl), lhs,
build2 (TREE_CODE (TREE_OPERAND (incr, 1)),
TREE_TYPE (decl), lhs,
RECUR (TREE_OPERAND (incr, 2))));
}
else if (TREE_CODE (TREE_OPERAND (incr, 1)) == NOP_EXPR
&& (TREE_CODE (TREE_OPERAND (incr, 2)) == PLUS_EXPR
|| (TREE_CODE (TREE_OPERAND (incr, 2)) == MINUS_EXPR)))
{
tree rhs = TREE_OPERAND (incr, 2);
tree lhs = RECUR (TREE_OPERAND (incr, 0));
tree rhs0 = RECUR (TREE_OPERAND (rhs, 0));
tree rhs1 = RECUR (TREE_OPERAND (rhs, 1));
incr = build2 (MODIFY_EXPR, TREE_TYPE (decl), lhs,
build2 (TREE_CODE (rhs), TREE_TYPE (decl),
rhs0, rhs1));
}
else
incr = RECUR (incr);
break;
default:
incr = RECUR (incr);
break;
}
TREE_VEC_ELT (declv, i) = decl;
TREE_VEC_ELT (initv, i) = init;
TREE_VEC_ELT (condv, i) = cond;
TREE_VEC_ELT (incrv, i) = incr;
#undef RECUR
}
static tree
tsubst_find_omp_teams (tree *tp, int *walk_subtrees, void *)
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
tsubst_decomp_names (tree decl, tree pattern_decl, tree args,
tsubst_flags_t complain, tree in_decl, tree *first,
unsigned int *cnt)
{
tree decl2, decl3, prev = decl;
*cnt = 0;
gcc_assert (DECL_NAME (decl) == NULL_TREE);
for (decl2 = DECL_CHAIN (pattern_decl);
decl2
&& VAR_P (decl2)
&& DECL_DECOMPOSITION_P (decl2)
&& DECL_NAME (decl2);
decl2 = DECL_CHAIN (decl2))
{
if (TREE_TYPE (decl2) == error_mark_node && *cnt == 0)
{
gcc_assert (errorcount);
return error_mark_node;
}
(*cnt)++;
gcc_assert (DECL_DECOMP_BASE (decl2) == pattern_decl);
gcc_assert (DECL_HAS_VALUE_EXPR_P (decl2));
tree v = DECL_VALUE_EXPR (decl2);
DECL_HAS_VALUE_EXPR_P (decl2) = 0;
SET_DECL_VALUE_EXPR (decl2, NULL_TREE);
decl3 = tsubst (decl2, args, complain, in_decl);
SET_DECL_VALUE_EXPR (decl2, v);
DECL_HAS_VALUE_EXPR_P (decl2) = 1;
if (VAR_P (decl3))
DECL_TEMPLATE_INSTANTIATED (decl3) = 1;
else
{
gcc_assert (errorcount);
decl = error_mark_node;
continue;
}
maybe_push_decl (decl3);
if (error_operand_p (decl3))
decl = error_mark_node;
else if (decl != error_mark_node
&& DECL_CHAIN (decl3) != prev
&& decl != prev)
{
gcc_assert (errorcount);
decl = error_mark_node;
}
else
prev = decl3;
}
*first = prev;
return decl;
}
tree
tsubst_expr (tree t, tree args, tsubst_flags_t complain, tree in_decl,
bool integral_constant_expression_p)
{
#define RETURN(EXP) do { r = (EXP); goto out; } while(0)
#define RECUR(NODE)				\
tsubst_expr ((NODE), args, complain, in_decl,	\
integral_constant_expression_p)
tree stmt, tmp;
tree r;
location_t loc;
if (t == NULL_TREE || t == error_mark_node)
return t;
loc = input_location;
if (EXPR_HAS_LOCATION (t))
input_location = EXPR_LOCATION (t);
if (STATEMENT_CODE_P (TREE_CODE (t)))
current_stmt_tree ()->stmts_are_full_exprs_p = STMT_IS_FULL_EXPR_P (t);
switch (TREE_CODE (t))
{
case STATEMENT_LIST:
{
tree_stmt_iterator i;
for (i = tsi_start (t); !tsi_end_p (i); tsi_next (&i))
RECUR (tsi_stmt (i));
break;
}
case CTOR_INITIALIZER:
finish_mem_initializers (tsubst_initializer_list
(TREE_OPERAND (t, 0), args));
break;
case RETURN_EXPR:
finish_return_stmt (RECUR (TREE_OPERAND (t, 0)));
break;
case EXPR_STMT:
tmp = RECUR (EXPR_STMT_EXPR (t));
if (EXPR_STMT_STMT_EXPR_RESULT (t))
finish_stmt_expr_expr (tmp, cur_stmt_expr);
else
finish_expr_stmt (tmp);
break;
case USING_STMT:
finish_local_using_directive (USING_STMT_NAMESPACE (t),
NULL_TREE);
break;
case DECL_EXPR:
{
tree decl, pattern_decl;
tree init;
pattern_decl = decl = DECL_EXPR_DECL (t);
if (TREE_CODE (decl) == LABEL_DECL)
finish_label_decl (DECL_NAME (decl));
else if (TREE_CODE (decl) == USING_DECL)
{
tree scope = USING_DECL_SCOPE (decl);
tree name = DECL_NAME (decl);
scope = tsubst (scope, args, complain, in_decl);
decl = lookup_qualified_name (scope, name,
false,
false);
if (decl == error_mark_node || TREE_CODE (decl) == TREE_LIST)
qualified_name_lookup_error (scope, name, decl, input_location);
else
finish_local_using_decl (decl, scope, name);
}
else if (is_capture_proxy (decl)
&& !DECL_TEMPLATE_INSTANTIATION (current_function_decl))
{
tree inst;
if (DECL_PACK_P (decl))
{
inst = (retrieve_local_specialization
(DECL_CAPTURED_VARIABLE (decl)));
gcc_assert (TREE_CODE (inst) == NONTYPE_ARGUMENT_PACK);
}
else
{
inst = lookup_name_real (DECL_NAME (decl), 0, 0,
true, 0, LOOKUP_HIDDEN);
gcc_assert (inst != decl && is_capture_proxy (inst));
}
register_local_specialization (inst, decl);
break;
}
else if (DECL_IMPLICIT_TYPEDEF_P (decl)
&& LAMBDA_TYPE_P (TREE_TYPE (decl)))
break;
else
{
init = DECL_INITIAL (decl);
decl = tsubst (decl, args, complain, in_decl);
if (decl != error_mark_node)
{
if (VAR_P (decl))
DECL_TEMPLATE_INSTANTIATED (decl) = 1;
if (VAR_P (decl) && !DECL_NAME (decl)
&& ANON_AGGR_TYPE_P (TREE_TYPE (decl)))
finish_anon_union (decl);
else if (is_capture_proxy (DECL_EXPR_DECL (t)))
{
DECL_CONTEXT (decl) = current_function_decl;
if (DECL_NAME (decl) == this_identifier)
{
tree lam = DECL_CONTEXT (current_function_decl);
lam = CLASSTYPE_LAMBDA_EXPR (lam);
LAMBDA_EXPR_THIS_CAPTURE (lam) = decl;
}
insert_capture_proxy (decl);
}
else if (DECL_IMPLICIT_TYPEDEF_P (t))
;
else if (TREE_CODE (decl) == FUNCTION_DECL
&& DECL_OMP_DECLARE_REDUCTION_P (decl)
&& DECL_FUNCTION_SCOPE_P (pattern_decl))
{
DECL_CONTEXT (decl) = NULL_TREE;
pushdecl (decl);
DECL_CONTEXT (decl) = current_function_decl;
cp_check_omp_declare_reduction (decl);
}
else
{
int const_init = false;
unsigned int cnt = 0;
tree first = NULL_TREE, ndecl = error_mark_node;
maybe_push_decl (decl);
if (VAR_P (decl)
&& DECL_DECOMPOSITION_P (decl)
&& TREE_TYPE (pattern_decl) != error_mark_node)
ndecl = tsubst_decomp_names (decl, pattern_decl, args,
complain, in_decl, &first,
&cnt);
if (VAR_P (decl)
&& DECL_PRETTY_FUNCTION_P (decl))
{
const char *const name
= cxx_printable_name (current_function_decl, 2);
init = cp_fname_init (name, &TREE_TYPE (decl));
}
else
init = tsubst_init (init, decl, args, complain, in_decl);
if (VAR_P (decl))
const_init = (DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P
(pattern_decl));
if (ndecl != error_mark_node)
cp_maybe_mangle_decomp (ndecl, first, cnt);
cp_finish_decl (decl, init, const_init, NULL_TREE, 0);
if (ndecl != error_mark_node)
cp_finish_decomp (ndecl, first, cnt);
}
}
}
break;
}
case FOR_STMT:
stmt = begin_for_stmt (NULL_TREE, NULL_TREE);
RECUR (FOR_INIT_STMT (t));
finish_init_stmt (stmt);
tmp = RECUR (FOR_COND (t));
finish_for_cond (tmp, stmt, false, 0);
tmp = RECUR (FOR_EXPR (t));
finish_for_expr (tmp, stmt);
{
bool prev = note_iteration_stmt_body_start ();
RECUR (FOR_BODY (t));
note_iteration_stmt_body_end (prev);
}
finish_for_stmt (stmt);
break;
case RANGE_FOR_STMT:
{
tree decl, expr;
stmt = (processing_template_decl
? begin_range_for_stmt (NULL_TREE, NULL_TREE)
: begin_for_stmt (NULL_TREE, NULL_TREE));
decl = RANGE_FOR_DECL (t);
decl = tsubst (decl, args, complain, in_decl);
maybe_push_decl (decl);
expr = RECUR (RANGE_FOR_EXPR (t));
tree decomp_first = NULL_TREE;
unsigned decomp_cnt = 0;
if (VAR_P (decl) && DECL_DECOMPOSITION_P (decl))
decl = tsubst_decomp_names (decl, RANGE_FOR_DECL (t), args,
complain, in_decl,
&decomp_first, &decomp_cnt);
if (processing_template_decl)
{
RANGE_FOR_IVDEP (stmt) = RANGE_FOR_IVDEP (t);
RANGE_FOR_UNROLL (stmt) = RANGE_FOR_UNROLL (t);
finish_range_for_decl (stmt, decl, expr);
if (decomp_first && decl != error_mark_node)
cp_finish_decomp (decl, decomp_first, decomp_cnt);
}
else
{
unsigned short unroll = (RANGE_FOR_UNROLL (t)
? tree_to_uhwi (RANGE_FOR_UNROLL (t)) : 0);
stmt = cp_convert_range_for (stmt, decl, expr,
decomp_first, decomp_cnt,
RANGE_FOR_IVDEP (t), unroll);
}
bool prev = note_iteration_stmt_body_start ();
RECUR (RANGE_FOR_BODY (t));
note_iteration_stmt_body_end (prev);
finish_for_stmt (stmt);
}
break;
case WHILE_STMT:
stmt = begin_while_stmt ();
tmp = RECUR (WHILE_COND (t));
finish_while_stmt_cond (tmp, stmt, false, 0);
{
bool prev = note_iteration_stmt_body_start ();
RECUR (WHILE_BODY (t));
note_iteration_stmt_body_end (prev);
}
finish_while_stmt (stmt);
break;
case DO_STMT:
stmt = begin_do_stmt ();
{
bool prev = note_iteration_stmt_body_start ();
RECUR (DO_BODY (t));
note_iteration_stmt_body_end (prev);
}
finish_do_body (stmt);
tmp = RECUR (DO_COND (t));
finish_do_stmt (tmp, stmt, false, 0);
break;
case IF_STMT:
stmt = begin_if_stmt ();
IF_STMT_CONSTEXPR_P (stmt) = IF_STMT_CONSTEXPR_P (t);
if (IF_STMT_CONSTEXPR_P (t))
args = add_extra_args (IF_STMT_EXTRA_ARGS (t), args);
tmp = RECUR (IF_COND (t));
tmp = finish_if_stmt_cond (tmp, stmt);
if (IF_STMT_CONSTEXPR_P (t)
&& instantiation_dependent_expression_p (tmp))
{
do_poplevel (IF_SCOPE (stmt));
IF_COND (stmt) = IF_COND (t);
THEN_CLAUSE (stmt) = THEN_CLAUSE (t);
ELSE_CLAUSE (stmt) = ELSE_CLAUSE (t);
IF_STMT_EXTRA_ARGS (stmt) = build_extra_args (t, args, complain);
add_stmt (stmt);
break;
}
if (IF_STMT_CONSTEXPR_P (t) && integer_zerop (tmp))
;
else
{
bool inhibit = integer_zerop (fold_non_dependent_expr (tmp));
if (inhibit)
++c_inhibit_evaluation_warnings;
RECUR (THEN_CLAUSE (t));
if (inhibit)
--c_inhibit_evaluation_warnings;
}
finish_then_clause (stmt);
if (IF_STMT_CONSTEXPR_P (t) && integer_nonzerop (tmp))
;
else if (ELSE_CLAUSE (t))
{
bool inhibit = integer_nonzerop (fold_non_dependent_expr (tmp));
begin_else_clause (stmt);
if (inhibit)
++c_inhibit_evaluation_warnings;
RECUR (ELSE_CLAUSE (t));
if (inhibit)
--c_inhibit_evaluation_warnings;
finish_else_clause (stmt);
}
finish_if_stmt (stmt);
break;
case BIND_EXPR:
if (BIND_EXPR_BODY_BLOCK (t))
stmt = begin_function_body ();
else
stmt = begin_compound_stmt (BIND_EXPR_TRY_BLOCK (t)
? BCS_TRY_BLOCK : 0);
RECUR (BIND_EXPR_BODY (t));
if (BIND_EXPR_BODY_BLOCK (t))
finish_function_body (stmt);
else
finish_compound_stmt (stmt);
break;
case BREAK_STMT:
finish_break_stmt ();
break;
case CONTINUE_STMT:
finish_continue_stmt ();
break;
case SWITCH_STMT:
stmt = begin_switch_stmt ();
tmp = RECUR (SWITCH_STMT_COND (t));
finish_switch_cond (tmp, stmt);
RECUR (SWITCH_STMT_BODY (t));
finish_switch_stmt (stmt);
break;
case CASE_LABEL_EXPR:
{
tree low = RECUR (CASE_LOW (t));
tree high = RECUR (CASE_HIGH (t));
tree l = finish_case_label (EXPR_LOCATION (t), low, high);
if (l && TREE_CODE (l) == CASE_LABEL_EXPR)
FALLTHROUGH_LABEL_P (CASE_LABEL (l))
= FALLTHROUGH_LABEL_P (CASE_LABEL (t));
}
break;
case LABEL_EXPR:
{
tree decl = LABEL_EXPR_LABEL (t);
tree label;
label = finish_label_stmt (DECL_NAME (decl));
if (TREE_CODE (label) == LABEL_DECL)
FALLTHROUGH_LABEL_P (label) = FALLTHROUGH_LABEL_P (decl);
if (DECL_ATTRIBUTES (decl) != NULL_TREE)
cplus_decl_attributes (&label, DECL_ATTRIBUTES (decl), 0);
}
break;
case GOTO_EXPR:
tmp = GOTO_DESTINATION (t);
if (TREE_CODE (tmp) != LABEL_DECL)
tmp = RECUR (tmp);
else
tmp = DECL_NAME (tmp);
finish_goto_stmt (tmp);
break;
case ASM_EXPR:
{
tree string = RECUR (ASM_STRING (t));
tree outputs = tsubst_copy_asm_operands (ASM_OUTPUTS (t), args,
complain, in_decl);
tree inputs = tsubst_copy_asm_operands (ASM_INPUTS (t), args,
complain, in_decl);
tree clobbers = tsubst_copy_asm_operands (ASM_CLOBBERS (t), args,
complain, in_decl);
tree labels = tsubst_copy_asm_operands (ASM_LABELS (t), args,
complain, in_decl);
tmp = finish_asm_stmt (ASM_VOLATILE_P (t), string, outputs, inputs,
clobbers, labels, ASM_INLINE_P (t));
tree asm_expr = tmp;
if (TREE_CODE (asm_expr) == CLEANUP_POINT_EXPR)
asm_expr = TREE_OPERAND (asm_expr, 0);
ASM_INPUT_P (asm_expr) = ASM_INPUT_P (t);
}
break;
case TRY_BLOCK:
if (CLEANUP_P (t))
{
stmt = begin_try_block ();
RECUR (TRY_STMTS (t));
finish_cleanup_try_block (stmt);
finish_cleanup (RECUR (TRY_HANDLERS (t)), stmt);
}
else
{
tree compound_stmt = NULL_TREE;
if (FN_TRY_BLOCK_P (t))
stmt = begin_function_try_block (&compound_stmt);
else
stmt = begin_try_block ();
RECUR (TRY_STMTS (t));
if (FN_TRY_BLOCK_P (t))
finish_function_try_block (stmt);
else
finish_try_block (stmt);
RECUR (TRY_HANDLERS (t));
if (FN_TRY_BLOCK_P (t))
finish_function_handler_sequence (stmt, compound_stmt);
else
finish_handler_sequence (stmt);
}
break;
case HANDLER:
{
tree decl = HANDLER_PARMS (t);
if (decl)
{
decl = tsubst (decl, args, complain, in_decl);
if (decl != error_mark_node)
DECL_TEMPLATE_INSTANTIATED (decl) = 1;
}
stmt = begin_handler ();
finish_handler_parms (decl, stmt);
RECUR (HANDLER_BODY (t));
finish_handler (stmt);
}
break;
case TAG_DEFN:
tmp = tsubst (TREE_TYPE (t), args, complain, NULL_TREE);
if (CLASS_TYPE_P (tmp))
{
gcc_assert (!LAMBDA_TYPE_P (TREE_TYPE (t)));
complete_type (tmp);
for (tree fld = TYPE_FIELDS (tmp); fld; fld = DECL_CHAIN (fld))
if ((VAR_P (fld)
|| (TREE_CODE (fld) == FUNCTION_DECL
&& !DECL_ARTIFICIAL (fld)))
&& DECL_TEMPLATE_INSTANTIATION (fld))
instantiate_decl (fld, false,
false);
}
break;
case STATIC_ASSERT:
{
tree condition;
++c_inhibit_evaluation_warnings;
condition = 
tsubst_expr (STATIC_ASSERT_CONDITION (t), 
args,
complain, in_decl,
true);
--c_inhibit_evaluation_warnings;
finish_static_assert (condition,
STATIC_ASSERT_MESSAGE (t),
STATIC_ASSERT_SOURCE_LOCATION (t),
false);
}
break;
case OACC_KERNELS:
case OACC_PARALLEL:
tmp = tsubst_omp_clauses (OMP_CLAUSES (t), C_ORT_ACC, args, complain,
in_decl);
stmt = begin_omp_parallel ();
RECUR (OMP_BODY (t));
finish_omp_construct (TREE_CODE (t), stmt, tmp);
break;
case OMP_PARALLEL:
r = push_omp_privatization_clauses (OMP_PARALLEL_COMBINED (t));
tmp = tsubst_omp_clauses (OMP_PARALLEL_CLAUSES (t), C_ORT_OMP, args,
complain, in_decl);
if (OMP_PARALLEL_COMBINED (t))
omp_parallel_combined_clauses = &tmp;
stmt = begin_omp_parallel ();
RECUR (OMP_PARALLEL_BODY (t));
gcc_assert (omp_parallel_combined_clauses == NULL);
OMP_PARALLEL_COMBINED (finish_omp_parallel (tmp, stmt))
= OMP_PARALLEL_COMBINED (t);
pop_omp_privatization_clauses (r);
break;
case OMP_TASK:
r = push_omp_privatization_clauses (false);
tmp = tsubst_omp_clauses (OMP_TASK_CLAUSES (t), C_ORT_OMP, args,
complain, in_decl);
stmt = begin_omp_task ();
RECUR (OMP_TASK_BODY (t));
finish_omp_task (tmp, stmt);
pop_omp_privatization_clauses (r);
break;
case OMP_FOR:
case OMP_SIMD:
case OMP_DISTRIBUTE:
case OMP_TASKLOOP:
case OACC_LOOP:
{
tree clauses, body, pre_body;
tree declv = NULL_TREE, initv = NULL_TREE, condv = NULL_TREE;
tree orig_declv = NULL_TREE;
tree incrv = NULL_TREE;
enum c_omp_region_type ort = C_ORT_OMP;
int i;
if (TREE_CODE (t) == OACC_LOOP)
ort = C_ORT_ACC;
r = push_omp_privatization_clauses (OMP_FOR_INIT (t) == NULL_TREE);
clauses = tsubst_omp_clauses (OMP_FOR_CLAUSES (t), ort, args, complain,
in_decl);
if (OMP_FOR_INIT (t) != NULL_TREE)
{
declv = make_tree_vec (TREE_VEC_LENGTH (OMP_FOR_INIT (t)));
if (OMP_FOR_ORIG_DECLS (t))
orig_declv = make_tree_vec (TREE_VEC_LENGTH (OMP_FOR_INIT (t)));
initv = make_tree_vec (TREE_VEC_LENGTH (OMP_FOR_INIT (t)));
condv = make_tree_vec (TREE_VEC_LENGTH (OMP_FOR_INIT (t)));
incrv = make_tree_vec (TREE_VEC_LENGTH (OMP_FOR_INIT (t)));
}
stmt = begin_omp_structured_block ();
pre_body = push_stmt_list ();
RECUR (OMP_FOR_PRE_BODY (t));
pre_body = pop_stmt_list (pre_body);
if (OMP_FOR_INIT (t) != NULL_TREE)
for (i = 0; i < TREE_VEC_LENGTH (OMP_FOR_INIT (t)); i++)
tsubst_omp_for_iterator (t, i, declv, orig_declv, initv, condv,
incrv, &clauses, args, complain, in_decl,
integral_constant_expression_p);
omp_parallel_combined_clauses = NULL;
body = push_stmt_list ();
RECUR (OMP_FOR_BODY (t));
body = pop_stmt_list (body);
if (OMP_FOR_INIT (t) != NULL_TREE)
t = finish_omp_for (EXPR_LOCATION (t), TREE_CODE (t), declv,
orig_declv, initv, condv, incrv, body, pre_body,
NULL, clauses);
else
{
t = make_node (TREE_CODE (t));
TREE_TYPE (t) = void_type_node;
OMP_FOR_BODY (t) = body;
OMP_FOR_PRE_BODY (t) = pre_body;
OMP_FOR_CLAUSES (t) = clauses;
SET_EXPR_LOCATION (t, EXPR_LOCATION (t));
add_stmt (t);
}
add_stmt (finish_omp_structured_block (stmt));
pop_omp_privatization_clauses (r);
}
break;
case OMP_SECTIONS:
omp_parallel_combined_clauses = NULL;
case OMP_SINGLE:
case OMP_TEAMS:
case OMP_CRITICAL:
r = push_omp_privatization_clauses (TREE_CODE (t) == OMP_TEAMS
&& OMP_TEAMS_COMBINED (t));
tmp = tsubst_omp_clauses (OMP_CLAUSES (t), C_ORT_OMP, args, complain,
in_decl);
stmt = push_stmt_list ();
RECUR (OMP_BODY (t));
stmt = pop_stmt_list (stmt);
t = copy_node (t);
OMP_BODY (t) = stmt;
OMP_CLAUSES (t) = tmp;
add_stmt (t);
pop_omp_privatization_clauses (r);
break;
case OACC_DATA:
case OMP_TARGET_DATA:
case OMP_TARGET:
tmp = tsubst_omp_clauses (OMP_CLAUSES (t), (TREE_CODE (t) == OACC_DATA)
? C_ORT_ACC : C_ORT_OMP, args, complain,
in_decl);
keep_next_level (true);
stmt = begin_omp_structured_block ();
RECUR (OMP_BODY (t));
stmt = finish_omp_structured_block (stmt);
t = copy_node (t);
OMP_BODY (t) = stmt;
OMP_CLAUSES (t) = tmp;
if (TREE_CODE (t) == OMP_TARGET && OMP_TARGET_COMBINED (t))
{
tree teams = cp_walk_tree (&stmt, tsubst_find_omp_teams, NULL, NULL);
if (teams)
{
tree c;
for (c = OMP_TEAMS_CLAUSES (teams);
c; c = OMP_CLAUSE_CHAIN (c))
if ((OMP_CLAUSE_CODE (c) == OMP_CLAUSE_NUM_TEAMS
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_THREAD_LIMIT)
&& TREE_CODE (OMP_CLAUSE_OPERAND (c, 0)) != INTEGER_CST)
{
tree expr = OMP_CLAUSE_OPERAND (c, 0);
expr = force_target_expr (TREE_TYPE (expr), expr, tf_none);
if (expr == error_mark_node)
continue;
tmp = TARGET_EXPR_SLOT (expr);
add_stmt (expr);
OMP_CLAUSE_OPERAND (c, 0) = expr;
tree tc = build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_FIRSTPRIVATE);
OMP_CLAUSE_DECL (tc) = tmp;
OMP_CLAUSE_CHAIN (tc) = OMP_TARGET_CLAUSES (t);
OMP_TARGET_CLAUSES (t) = tc;
}
}
}
add_stmt (t);
break;
case OACC_DECLARE:
t = copy_node (t);
tmp = tsubst_omp_clauses (OACC_DECLARE_CLAUSES (t), C_ORT_ACC, args,
complain, in_decl);
OACC_DECLARE_CLAUSES (t) = tmp;
add_stmt (t);
break;
case OMP_TARGET_UPDATE:
case OMP_TARGET_ENTER_DATA:
case OMP_TARGET_EXIT_DATA:
tmp = tsubst_omp_clauses (OMP_STANDALONE_CLAUSES (t), C_ORT_OMP, args,
complain, in_decl);
t = copy_node (t);
OMP_STANDALONE_CLAUSES (t) = tmp;
add_stmt (t);
break;
case OACC_ENTER_DATA:
case OACC_EXIT_DATA:
case OACC_UPDATE:
tmp = tsubst_omp_clauses (OMP_STANDALONE_CLAUSES (t), C_ORT_ACC, args,
complain, in_decl);
t = copy_node (t);
OMP_STANDALONE_CLAUSES (t) = tmp;
add_stmt (t);
break;
case OMP_ORDERED:
tmp = tsubst_omp_clauses (OMP_ORDERED_CLAUSES (t), C_ORT_OMP, args,
complain, in_decl);
stmt = push_stmt_list ();
RECUR (OMP_BODY (t));
stmt = pop_stmt_list (stmt);
t = copy_node (t);
OMP_BODY (t) = stmt;
OMP_ORDERED_CLAUSES (t) = tmp;
add_stmt (t);
break;
case OMP_SECTION:
case OMP_MASTER:
case OMP_TASKGROUP:
stmt = push_stmt_list ();
RECUR (OMP_BODY (t));
stmt = pop_stmt_list (stmt);
t = copy_node (t);
OMP_BODY (t) = stmt;
add_stmt (t);
break;
case OMP_ATOMIC:
gcc_assert (OMP_ATOMIC_DEPENDENT_P (t));
if (TREE_CODE (TREE_OPERAND (t, 1)) != MODIFY_EXPR)
{
tree op1 = TREE_OPERAND (t, 1);
tree rhs1 = NULL_TREE;
tree lhs, rhs;
if (TREE_CODE (op1) == COMPOUND_EXPR)
{
rhs1 = RECUR (TREE_OPERAND (op1, 0));
op1 = TREE_OPERAND (op1, 1);
}
lhs = RECUR (TREE_OPERAND (op1, 0));
rhs = RECUR (TREE_OPERAND (op1, 1));
finish_omp_atomic (OMP_ATOMIC, TREE_CODE (op1), lhs, rhs,
NULL_TREE, NULL_TREE, rhs1,
OMP_ATOMIC_SEQ_CST (t));
}
else
{
tree op1 = TREE_OPERAND (t, 1);
tree v = NULL_TREE, lhs, rhs = NULL_TREE, lhs1 = NULL_TREE;
tree rhs1 = NULL_TREE;
enum tree_code code = TREE_CODE (TREE_OPERAND (op1, 1));
enum tree_code opcode = NOP_EXPR;
if (code == OMP_ATOMIC_READ)
{
v = RECUR (TREE_OPERAND (op1, 0));
lhs = RECUR (TREE_OPERAND (TREE_OPERAND (op1, 1), 0));
}
else if (code == OMP_ATOMIC_CAPTURE_OLD
|| code == OMP_ATOMIC_CAPTURE_NEW)
{
tree op11 = TREE_OPERAND (TREE_OPERAND (op1, 1), 1);
v = RECUR (TREE_OPERAND (op1, 0));
lhs1 = RECUR (TREE_OPERAND (TREE_OPERAND (op1, 1), 0));
if (TREE_CODE (op11) == COMPOUND_EXPR)
{
rhs1 = RECUR (TREE_OPERAND (op11, 0));
op11 = TREE_OPERAND (op11, 1);
}
lhs = RECUR (TREE_OPERAND (op11, 0));
rhs = RECUR (TREE_OPERAND (op11, 1));
opcode = TREE_CODE (op11);
if (opcode == MODIFY_EXPR)
opcode = NOP_EXPR;
}
else
{
code = OMP_ATOMIC;
lhs = RECUR (TREE_OPERAND (op1, 0));
rhs = RECUR (TREE_OPERAND (op1, 1));
}
finish_omp_atomic (code, opcode, lhs, rhs, v, lhs1, rhs1,
OMP_ATOMIC_SEQ_CST (t));
}
break;
case TRANSACTION_EXPR:
{
int flags = 0;
flags |= (TRANSACTION_EXPR_OUTER (t) ? TM_STMT_ATTR_OUTER : 0);
flags |= (TRANSACTION_EXPR_RELAXED (t) ? TM_STMT_ATTR_RELAXED : 0);
if (TRANSACTION_EXPR_IS_STMT (t))
{
tree body = TRANSACTION_EXPR_BODY (t);
tree noex = NULL_TREE;
if (TREE_CODE (body) == MUST_NOT_THROW_EXPR)
{
noex = MUST_NOT_THROW_COND (body);
if (noex == NULL_TREE)
noex = boolean_true_node;
body = TREE_OPERAND (body, 0);
}
stmt = begin_transaction_stmt (input_location, NULL, flags);
RECUR (body);
finish_transaction_stmt (stmt, NULL, flags, RECUR (noex));
}
else
{
stmt = build_transaction_expr (EXPR_LOCATION (t),
RECUR (TRANSACTION_EXPR_BODY (t)),
flags, NULL_TREE);
RETURN (stmt);
}
}
break;
case MUST_NOT_THROW_EXPR:
{
tree op0 = RECUR (TREE_OPERAND (t, 0));
tree cond = RECUR (MUST_NOT_THROW_COND (t));
RETURN (build_must_not_throw_expr (op0, cond));
}
case EXPR_PACK_EXPANSION:
error ("invalid use of pack expansion expression");
RETURN (error_mark_node);
case NONTYPE_ARGUMENT_PACK:
error ("use %<...%> to expand argument pack");
RETURN (error_mark_node);
case COMPOUND_EXPR:
tmp = RECUR (TREE_OPERAND (t, 0));
if (tmp == NULL_TREE)
RETURN (RECUR (TREE_OPERAND (t, 1)));
RETURN (build_x_compound_expr (EXPR_LOCATION (t), tmp,
RECUR (TREE_OPERAND (t, 1)),
complain));
case ANNOTATE_EXPR:
tmp = RECUR (TREE_OPERAND (t, 0));
RETURN (build3_loc (EXPR_LOCATION (t), ANNOTATE_EXPR,
TREE_TYPE (tmp), tmp,
RECUR (TREE_OPERAND (t, 1)),
RECUR (TREE_OPERAND (t, 2))));
default:
gcc_assert (!STATEMENT_CODE_P (TREE_CODE (t)));
RETURN (tsubst_copy_and_build (t, args, complain, in_decl,
false,
integral_constant_expression_p));
}
RETURN (NULL_TREE);
out:
input_location = loc;
return r;
#undef RECUR
#undef RETURN
}
static void
tsubst_omp_udr (tree t, tree args, tsubst_flags_t complain, tree in_decl)
{
if (t == NULL_TREE || t == error_mark_node)
return;
gcc_assert (TREE_CODE (t) == STATEMENT_LIST);
tree_stmt_iterator tsi;
int i;
tree stmts[7];
memset (stmts, 0, sizeof stmts);
for (i = 0, tsi = tsi_start (t);
i < 7 && !tsi_end_p (tsi);
i++, tsi_next (&tsi))
stmts[i] = tsi_stmt (tsi);
gcc_assert (tsi_end_p (tsi));
if (i >= 3)
{
gcc_assert (TREE_CODE (stmts[0]) == DECL_EXPR
&& TREE_CODE (stmts[1]) == DECL_EXPR);
tree omp_out = tsubst (DECL_EXPR_DECL (stmts[0]),
args, complain, in_decl);
tree omp_in = tsubst (DECL_EXPR_DECL (stmts[1]),
args, complain, in_decl);
DECL_CONTEXT (omp_out) = current_function_decl;
DECL_CONTEXT (omp_in) = current_function_decl;
keep_next_level (true);
tree block = begin_omp_structured_block ();
tsubst_expr (stmts[2], args, complain, in_decl, false);
block = finish_omp_structured_block (block);
block = maybe_cleanup_point_expr_void (block);
add_decl_expr (omp_out);
if (TREE_NO_WARNING (DECL_EXPR_DECL (stmts[0])))
TREE_NO_WARNING (omp_out) = 1;
add_decl_expr (omp_in);
finish_expr_stmt (block);
}
if (i >= 6)
{
gcc_assert (TREE_CODE (stmts[3]) == DECL_EXPR
&& TREE_CODE (stmts[4]) == DECL_EXPR);
tree omp_priv = tsubst (DECL_EXPR_DECL (stmts[3]),
args, complain, in_decl);
tree omp_orig = tsubst (DECL_EXPR_DECL (stmts[4]),
args, complain, in_decl);
DECL_CONTEXT (omp_priv) = current_function_decl;
DECL_CONTEXT (omp_orig) = current_function_decl;
keep_next_level (true);
tree block = begin_omp_structured_block ();
tsubst_expr (stmts[5], args, complain, in_decl, false);
block = finish_omp_structured_block (block);
block = maybe_cleanup_point_expr_void (block);
cp_walk_tree (&block, cp_remove_omp_priv_cleanup_stmt, omp_priv, NULL);
add_decl_expr (omp_priv);
add_decl_expr (omp_orig);
finish_expr_stmt (block);
if (i == 7)
add_decl_expr (omp_orig);
}
}
static tree
tsubst_non_call_postfix_expression (tree t, tree args,
tsubst_flags_t complain,
tree in_decl)
{
if (TREE_CODE (t) == SCOPE_REF)
t = tsubst_qualified_id (t, args, complain, in_decl,
false, false);
else
t = tsubst_copy_and_build (t, args, complain, in_decl,
false,
false);
return t;
}
tree
tsubst_lambda_expr (tree t, tree args, tsubst_flags_t complain, tree in_decl)
{
tree oldfn = lambda_function (t);
in_decl = oldfn;
tree r = build_lambda_expr ();
LAMBDA_EXPR_LOCATION (r)
= LAMBDA_EXPR_LOCATION (t);
LAMBDA_EXPR_DEFAULT_CAPTURE_MODE (r)
= LAMBDA_EXPR_DEFAULT_CAPTURE_MODE (t);
LAMBDA_EXPR_MUTABLE_P (r) = LAMBDA_EXPR_MUTABLE_P (t);
if (LAMBDA_EXPR_EXTRA_SCOPE (t) == NULL_TREE)
record_null_lambda_scope (r);
else
record_lambda_scope (r);
gcc_assert (LAMBDA_EXPR_THIS_CAPTURE (t) == NULL_TREE
&& LAMBDA_EXPR_PENDING_PROXIES (t) == NULL);
for (tree cap = LAMBDA_EXPR_CAPTURE_LIST (t); cap;
cap = TREE_CHAIN (cap))
{
tree field = TREE_PURPOSE (cap);
if (PACK_EXPANSION_P (field))
field = PACK_EXPANSION_PATTERN (field);
field = tsubst_decl (field, args, complain);
if (field == error_mark_node)
return error_mark_node;
tree init = TREE_VALUE (cap);
if (PACK_EXPANSION_P (init))
init = tsubst_pack_expansion (init, args, complain, in_decl);
else
init = tsubst_copy_and_build (init, args, complain, in_decl,
false, false);
if (TREE_CODE (field) == TREE_VEC)
{
int len = TREE_VEC_LENGTH (field);
gcc_assert (TREE_CODE (init) == TREE_VEC
&& TREE_VEC_LENGTH (init) == len);
for (int i = 0; i < len; ++i)
LAMBDA_EXPR_CAPTURE_LIST (r)
= tree_cons (TREE_VEC_ELT (field, i),
TREE_VEC_ELT (init, i),
LAMBDA_EXPR_CAPTURE_LIST (r));
}
else
{
LAMBDA_EXPR_CAPTURE_LIST (r)
= tree_cons (field, init, LAMBDA_EXPR_CAPTURE_LIST (r));
if (id_equal (DECL_NAME (field), "__this"))
LAMBDA_EXPR_THIS_CAPTURE (r) = field;
}
}
tree type = begin_lambda_type (r);
if (type == error_mark_node)
return error_mark_node;
determine_visibility (TYPE_NAME (type));
register_capture_members (LAMBDA_EXPR_CAPTURE_LIST (r));
tree oldtmpl = (generic_lambda_fn_p (oldfn)
? DECL_TI_TEMPLATE (oldfn)
: NULL_TREE);
tree fntype = static_fn_type (oldfn);
if (oldtmpl)
++processing_template_decl;
fntype = tsubst (fntype, args, complain, in_decl);
if (oldtmpl)
--processing_template_decl;
if (fntype == error_mark_node)
r = error_mark_node;
else
{
fntype = build_memfn_type (fntype, type,
type_memfn_quals (fntype),
type_memfn_rqual (fntype));
tree fn, tmpl;
if (oldtmpl)
{
tmpl = tsubst_template_decl (oldtmpl, args, complain, fntype);
fn = DECL_TEMPLATE_RESULT (tmpl);
finish_member_declaration (tmpl);
}
else
{
tmpl = NULL_TREE;
fn = tsubst_function_decl (oldfn, args, complain, fntype);
finish_member_declaration (fn);
}
DECL_DECLARED_CONSTEXPR_P (fn) = false;
bool nested = cfun;
if (nested)
push_function_context ();
else
++function_depth;
local_specialization_stack s (lss_copy);
tree body = start_lambda_function (fn, r);
register_parameter_specializations (oldfn, fn);
if (oldtmpl)
{
language_function *ol = DECL_STRUCT_FUNCTION (oldfn)->language;
current_function_returns_value = ol->returns_value;
current_function_returns_null = ol->returns_null;
current_function_returns_abnormally = ol->returns_abnormally;
current_function_infinite_loop = ol->infinite_loop;
}
tsubst_expr (DECL_SAVED_TREE (oldfn), args, complain, r,
false);
finish_lambda_function (body);
if (nested)
pop_function_context ();
else
--function_depth;
LAMBDA_EXPR_CAPTURE_LIST (r)
= nreverse (LAMBDA_EXPR_CAPTURE_LIST (r));
LAMBDA_EXPR_THIS_CAPTURE (r) = NULL_TREE;
maybe_add_lambda_conv_op (type);
}
finish_struct (type, NULL_TREE);
insert_pending_capture_proxies ();
return r;
}
tree
tsubst_copy_and_build (tree t,
tree args,
tsubst_flags_t complain,
tree in_decl,
bool function_p,
bool integral_constant_expression_p)
{
#define RETURN(EXP) do { retval = (EXP); goto out; } while(0)
#define RECUR(NODE)						\
tsubst_copy_and_build (NODE, args, complain, in_decl, 	\
false,			\
integral_constant_expression_p)
tree retval, op1;
location_t loc;
if (t == NULL_TREE || t == error_mark_node)
return t;
loc = input_location;
if (EXPR_HAS_LOCATION (t))
input_location = EXPR_LOCATION (t);
tsubst_flags_t decltype_flag = (complain & tf_decltype);
complain &= ~tf_decltype;
switch (TREE_CODE (t))
{
case USING_DECL:
t = DECL_NAME (t);
case IDENTIFIER_NODE:
{
tree decl;
cp_id_kind idk;
bool non_integral_constant_expression_p;
const char *error_msg;
if (IDENTIFIER_CONV_OP_P (t))
{
tree new_type = tsubst (TREE_TYPE (t), args, complain, in_decl);
t = make_conv_op_name (new_type);
}
decl = lookup_name (t);
if (decl == NULL_TREE)
decl = error_mark_node;
decl = finish_id_expression (t, decl, NULL_TREE,
&idk,
integral_constant_expression_p,
(cxx_dialect >= cxx11),
&non_integral_constant_expression_p,
false,
true,
false,
false,
&error_msg,
input_location);
if (error_msg)
error (error_msg);
if (!function_p && identifier_p (decl))
{
if (complain & tf_error)
unqualified_name_lookup_error (decl);
decl = error_mark_node;
}
RETURN (decl);
}
case TEMPLATE_ID_EXPR:
{
tree object;
tree templ = RECUR (TREE_OPERAND (t, 0));
tree targs = TREE_OPERAND (t, 1);
if (targs)
targs = tsubst_template_args (targs, args, complain, in_decl);
if (targs == error_mark_node)
RETURN (error_mark_node);
if (TREE_CODE (templ) == SCOPE_REF)
{
tree name = TREE_OPERAND (templ, 1);
tree tid = lookup_template_function (name, targs);
TREE_OPERAND (templ, 1) = tid;
RETURN (templ);
}
if (variable_template_p (templ))
RETURN (lookup_and_finish_template_variable (templ, targs, complain));
if (TREE_CODE (templ) == COMPONENT_REF)
{
object = TREE_OPERAND (templ, 0);
templ = TREE_OPERAND (templ, 1);
}
else
object = NULL_TREE;
templ = lookup_template_function (templ, targs);
if (object)
RETURN (build3 (COMPONENT_REF, TREE_TYPE (templ),
object, templ, NULL_TREE));
else
RETURN (baselink_for_fns (templ));
}
case INDIRECT_REF:
{
tree r = RECUR (TREE_OPERAND (t, 0));
if (REFERENCE_REF_P (t))
{
r = convert_from_reference (r);
}
else
r = build_x_indirect_ref (input_location, r, RO_UNARY_STAR,
complain|decltype_flag);
if (REF_PARENTHESIZED_P (t))
r = force_paren_expr (r);
RETURN (r);
}
case NOP_EXPR:
{
tree type = tsubst (TREE_TYPE (t), args, complain, in_decl);
tree op0 = RECUR (TREE_OPERAND (t, 0));
RETURN (build_nop (type, op0));
}
case IMPLICIT_CONV_EXPR:
{
tree type = tsubst (TREE_TYPE (t), args, complain, in_decl);
tree expr = RECUR (TREE_OPERAND (t, 0));
if (dependent_type_p (type) || type_dependent_expression_p (expr))
{
retval = copy_node (t);
TREE_TYPE (retval) = type;
TREE_OPERAND (retval, 0) = expr;
RETURN (retval);
}
if (IMPLICIT_CONV_EXPR_NONTYPE_ARG (t))
RETURN (expr);
int flags = LOOKUP_IMPLICIT;
if (IMPLICIT_CONV_EXPR_DIRECT_INIT (t))
flags = LOOKUP_NORMAL;
RETURN (perform_implicit_conversion_flags (type, expr, complain,
flags));
}
case CONVERT_EXPR:
{
tree type = tsubst (TREE_TYPE (t), args, complain, in_decl);
tree op0 = RECUR (TREE_OPERAND (t, 0));
if (op0 == error_mark_node)
RETURN (error_mark_node);
RETURN (build1 (CONVERT_EXPR, type, op0));
}
case CAST_EXPR:
case REINTERPRET_CAST_EXPR:
case CONST_CAST_EXPR:
case DYNAMIC_CAST_EXPR:
case STATIC_CAST_EXPR:
{
tree type;
tree op, r = NULL_TREE;
type = tsubst (TREE_TYPE (t), args, complain, in_decl);
if (integral_constant_expression_p
&& !cast_valid_in_integral_constant_expression_p (type))
{
if (complain & tf_error)
error ("a cast to a type other than an integral or "
"enumeration type cannot appear in a constant-expression");
RETURN (error_mark_node);
}
op = RECUR (TREE_OPERAND (t, 0));
warning_sentinel s(warn_useless_cast);
warning_sentinel s2(warn_ignored_qualifiers);
switch (TREE_CODE (t))
{
case CAST_EXPR:
r = build_functional_cast (type, op, complain);
break;
case REINTERPRET_CAST_EXPR:
r = build_reinterpret_cast (type, op, complain);
break;
case CONST_CAST_EXPR:
r = build_const_cast (type, op, complain);
break;
case DYNAMIC_CAST_EXPR:
r = build_dynamic_cast (type, op, complain);
break;
case STATIC_CAST_EXPR:
r = build_static_cast (type, op, complain);
break;
default:
gcc_unreachable ();
}
RETURN (r);
}
case POSTDECREMENT_EXPR:
case POSTINCREMENT_EXPR:
op1 = tsubst_non_call_postfix_expression (TREE_OPERAND (t, 0),
args, complain, in_decl);
RETURN (build_x_unary_op (input_location, TREE_CODE (t), op1,
complain|decltype_flag));
case PREDECREMENT_EXPR:
case PREINCREMENT_EXPR:
case NEGATE_EXPR:
case BIT_NOT_EXPR:
case ABS_EXPR:
case TRUTH_NOT_EXPR:
case UNARY_PLUS_EXPR:  
case REALPART_EXPR:
case IMAGPART_EXPR:
RETURN (build_x_unary_op (input_location, TREE_CODE (t),
RECUR (TREE_OPERAND (t, 0)),
complain|decltype_flag));
case FIX_TRUNC_EXPR:
gcc_unreachable ();    
case ADDR_EXPR:
op1 = TREE_OPERAND (t, 0);
if (TREE_CODE (op1) == LABEL_DECL)
RETURN (finish_label_address_expr (DECL_NAME (op1),
EXPR_LOCATION (op1)));
if (TREE_CODE (op1) == SCOPE_REF)
op1 = tsubst_qualified_id (op1, args, complain, in_decl,
true, true);
else
op1 = tsubst_non_call_postfix_expression (op1, args, complain,
in_decl);
RETURN (build_x_unary_op (input_location, ADDR_EXPR, op1,
complain|decltype_flag));
case PLUS_EXPR:
case MINUS_EXPR:
case MULT_EXPR:
case TRUNC_DIV_EXPR:
case CEIL_DIV_EXPR:
case FLOOR_DIV_EXPR:
case ROUND_DIV_EXPR:
case EXACT_DIV_EXPR:
case BIT_AND_EXPR:
case BIT_IOR_EXPR:
case BIT_XOR_EXPR:
case TRUNC_MOD_EXPR:
case FLOOR_MOD_EXPR:
case TRUTH_ANDIF_EXPR:
case TRUTH_ORIF_EXPR:
case TRUTH_AND_EXPR:
case TRUTH_OR_EXPR:
case RSHIFT_EXPR:
case LSHIFT_EXPR:
case RROTATE_EXPR:
case LROTATE_EXPR:
case EQ_EXPR:
case NE_EXPR:
case MAX_EXPR:
case MIN_EXPR:
case LE_EXPR:
case GE_EXPR:
case LT_EXPR:
case GT_EXPR:
case MEMBER_REF:
case DOTSTAR_EXPR:
{
warning_sentinel s1(warn_type_limits);
warning_sentinel s2(warn_div_by_zero);
warning_sentinel s3(warn_logical_op);
warning_sentinel s4(warn_tautological_compare);
tree op0 = RECUR (TREE_OPERAND (t, 0));
tree op1 = RECUR (TREE_OPERAND (t, 1));
tree r = build_x_binary_op
(input_location, TREE_CODE (t),
op0,
(TREE_NO_WARNING (TREE_OPERAND (t, 0))
? ERROR_MARK
: TREE_CODE (TREE_OPERAND (t, 0))),
op1,
(TREE_NO_WARNING (TREE_OPERAND (t, 1))
? ERROR_MARK
: TREE_CODE (TREE_OPERAND (t, 1))),
NULL,
complain|decltype_flag);
if (EXPR_P (r) && TREE_NO_WARNING (t))
TREE_NO_WARNING (r) = TREE_NO_WARNING (t);
RETURN (r);
}
case POINTER_PLUS_EXPR:
{
tree op0 = RECUR (TREE_OPERAND (t, 0));
tree op1 = RECUR (TREE_OPERAND (t, 1));
RETURN (fold_build_pointer_plus (op0, op1));
}
case SCOPE_REF:
RETURN (tsubst_qualified_id (t, args, complain, in_decl, true,
false));
case ARRAY_REF:
op1 = tsubst_non_call_postfix_expression (TREE_OPERAND (t, 0),
args, complain, in_decl);
RETURN (build_x_array_ref (EXPR_LOCATION (t), op1,
RECUR (TREE_OPERAND (t, 1)),
complain|decltype_flag));
case SIZEOF_EXPR:
if (PACK_EXPANSION_P (TREE_OPERAND (t, 0))
|| ARGUMENT_PACK_P (TREE_OPERAND (t, 0)))
RETURN (tsubst_copy (t, args, complain, in_decl));
case ALIGNOF_EXPR:
{
tree r;
op1 = TREE_OPERAND (t, 0);
if (TREE_CODE (t) == SIZEOF_EXPR && SIZEOF_EXPR_TYPE_P (t))
op1 = TREE_TYPE (op1);
bool std_alignof = (TREE_CODE (t) == ALIGNOF_EXPR
&& ALIGNOF_EXPR_STD_P (t));
if (!args)
{
if (!TYPE_P (op1))
op1 = TREE_TYPE (op1);
}
else
{
++cp_unevaluated_operand;
++c_inhibit_evaluation_warnings;
if (TYPE_P (op1))
op1 = tsubst (op1, args, complain, in_decl);
else
op1 = tsubst_copy_and_build (op1, args, complain, in_decl,
false,
false);
--cp_unevaluated_operand;
--c_inhibit_evaluation_warnings;
}
if (TYPE_P (op1))
r = cxx_sizeof_or_alignof_type (op1, TREE_CODE (t), std_alignof,
complain & tf_error);
else
r = cxx_sizeof_or_alignof_expr (op1, TREE_CODE (t),
complain & tf_error);
if (TREE_CODE (t) == SIZEOF_EXPR && r != error_mark_node)
{
if (TREE_CODE (r) != SIZEOF_EXPR || TYPE_P (op1))
{
if (!processing_template_decl && TYPE_P (op1))
{
r = build_min (SIZEOF_EXPR, size_type_node,
build1 (NOP_EXPR, op1, error_mark_node));
SIZEOF_EXPR_TYPE_P (r) = 1;
}
else
r = build_min (SIZEOF_EXPR, size_type_node, op1);
TREE_SIDE_EFFECTS (r) = 0;
TREE_READONLY (r) = 1;
}
SET_EXPR_LOCATION (r, EXPR_LOCATION (t));
}
RETURN (r);
}
case AT_ENCODE_EXPR:
{
op1 = TREE_OPERAND (t, 0);
++cp_unevaluated_operand;
++c_inhibit_evaluation_warnings;
op1 = tsubst_copy_and_build (op1, args, complain, in_decl,
false,
false);
--cp_unevaluated_operand;
--c_inhibit_evaluation_warnings;
RETURN (objc_build_encode_expr (op1));
}
case NOEXCEPT_EXPR:
op1 = TREE_OPERAND (t, 0);
++cp_unevaluated_operand;
++c_inhibit_evaluation_warnings;
++cp_noexcept_operand;
op1 = tsubst_copy_and_build (op1, args, complain, in_decl,
false,
false);
--cp_unevaluated_operand;
--c_inhibit_evaluation_warnings;
--cp_noexcept_operand;
RETURN (finish_noexcept_expr (op1, complain));
case MODOP_EXPR:
{
warning_sentinel s(warn_div_by_zero);
tree lhs = RECUR (TREE_OPERAND (t, 0));
tree rhs = RECUR (TREE_OPERAND (t, 2));
tree r = build_x_modify_expr
(EXPR_LOCATION (t), lhs, TREE_CODE (TREE_OPERAND (t, 1)), rhs,
complain|decltype_flag);
if (TREE_NO_WARNING (t))
TREE_NO_WARNING (r) = TREE_NO_WARNING (t);
RETURN (r);
}
case ARROW_EXPR:
op1 = tsubst_non_call_postfix_expression (TREE_OPERAND (t, 0),
args, complain, in_decl);
if (DECL_P (op1)
&& !mark_used (op1, complain) && !(complain & tf_error))
RETURN (error_mark_node);
RETURN (build_x_arrow (input_location, op1, complain));
case NEW_EXPR:
{
tree placement = RECUR (TREE_OPERAND (t, 0));
tree init = RECUR (TREE_OPERAND (t, 3));
vec<tree, va_gc> *placement_vec;
vec<tree, va_gc> *init_vec;
tree ret;
if (placement == NULL_TREE)
placement_vec = NULL;
else
{
placement_vec = make_tree_vector ();
for (; placement != NULL_TREE; placement = TREE_CHAIN (placement))
vec_safe_push (placement_vec, TREE_VALUE (placement));
}
if (init == NULL_TREE && TREE_OPERAND (t, 3) == NULL_TREE)
init_vec = NULL;
else
{
init_vec = make_tree_vector ();
if (init == void_node)
gcc_assert (init_vec != NULL);
else
{
for (; init != NULL_TREE; init = TREE_CHAIN (init))
vec_safe_push (init_vec, TREE_VALUE (init));
}
}
tree op1 = tsubst (TREE_OPERAND (t, 1), args, complain, in_decl);
tree op2 = RECUR (TREE_OPERAND (t, 2));
ret = build_new (&placement_vec, op1, op2, &init_vec,
NEW_EXPR_USE_GLOBAL (t),
complain);
if (placement_vec != NULL)
release_tree_vector (placement_vec);
if (init_vec != NULL)
release_tree_vector (init_vec);
RETURN (ret);
}
case DELETE_EXPR:
{
tree op0 = RECUR (TREE_OPERAND (t, 0));
tree op1 = RECUR (TREE_OPERAND (t, 1));
RETURN (delete_sanity (op0, op1,
DELETE_EXPR_USE_VEC (t),
DELETE_EXPR_USE_GLOBAL (t),
complain));
}
case COMPOUND_EXPR:
{
tree op0 = tsubst_copy_and_build (TREE_OPERAND (t, 0), args,
complain & ~tf_decltype, in_decl,
false,
integral_constant_expression_p);
RETURN (build_x_compound_expr (EXPR_LOCATION (t),
op0,
RECUR (TREE_OPERAND (t, 1)),
complain|decltype_flag));
}
case CALL_EXPR:
{
tree function;
vec<tree, va_gc> *call_args;
unsigned int nargs, i;
bool qualified_p;
bool koenig_p;
tree ret;
function = CALL_EXPR_FN (t);
if (function == NULL_TREE && call_expr_nargs (t) == 0)
RETURN (t);
koenig_p = KOENIG_LOOKUP_P (t);
if (function == NULL_TREE)
{
koenig_p = false;
qualified_p = false;
}
else if (TREE_CODE (function) == SCOPE_REF)
{
qualified_p = true;
function = tsubst_qualified_id (function, args, complain, in_decl,
false,
false);
}
else if (koenig_p && identifier_p (function))
{
qualified_p = false;
}
else
{
if (TREE_CODE (function) == COMPONENT_REF)
{
tree op = TREE_OPERAND (function, 1);
qualified_p = (TREE_CODE (op) == SCOPE_REF
|| (BASELINK_P (op)
&& BASELINK_QUALIFIED_P (op)));
}
else
qualified_p = false;
if (TREE_CODE (function) == ADDR_EXPR
&& TREE_CODE (TREE_OPERAND (function, 0)) == FUNCTION_DECL)
function = TREE_OPERAND (function, 0);
function = tsubst_copy_and_build (function, args, complain,
in_decl,
!qualified_p,
integral_constant_expression_p);
if (BASELINK_P (function))
qualified_p = true;
}
nargs = call_expr_nargs (t);
call_args = make_tree_vector ();
for (i = 0; i < nargs; ++i)
{
tree arg = CALL_EXPR_ARG (t, i);
if (!PACK_EXPANSION_P (arg))
vec_safe_push (call_args, RECUR (CALL_EXPR_ARG (t, i)));
else
{
arg = tsubst_pack_expansion (arg, args, complain, in_decl);
if (TREE_CODE (arg) == TREE_VEC)
{
unsigned int len, j;
len = TREE_VEC_LENGTH (arg);
for (j = 0; j < len; ++j)
{
tree value = TREE_VEC_ELT (arg, j);
if (value != NULL_TREE)
value = convert_from_reference (value);
vec_safe_push (call_args, value);
}
}
else
{
vec_safe_push (call_args, arg);
}
}
}
if (koenig_p
&& ((is_overloaded_fn (function)
&& !DECL_FUNCTION_MEMBER_P (get_first_fn (function)))
|| identifier_p (function))
&& type_dependent_expression_p_push (t)
&& !any_type_dependent_arguments_p (call_args))
function = perform_koenig_lookup (function, call_args, tf_none);
if (function != NULL_TREE
&& identifier_p (function)
&& !any_type_dependent_arguments_p (call_args))
{
if (koenig_p && (complain & tf_warning_or_error))
{
tree unq = (tsubst_copy_and_build
(function, args, complain, in_decl, true,
integral_constant_expression_p));
if (unq == error_mark_node)
{
release_tree_vector (call_args);
RETURN (error_mark_node);
}
if (unq != function)
{
bool in_lambda = (current_class_type
&& LAMBDA_TYPE_P (current_class_type));
char const *const msg
= G_("%qD was not declared in this scope, "
"and no declarations were found by "
"argument-dependent lookup at the point "
"of instantiation");
bool diag = true;
if (in_lambda)
error_at (EXPR_LOC_OR_LOC (t, input_location),
msg, function);
else
diag = permerror (EXPR_LOC_OR_LOC (t, input_location),
msg, function);
if (diag)
{
tree fn = unq;
if (INDIRECT_REF_P (fn))
fn = TREE_OPERAND (fn, 0);
if (is_overloaded_fn (fn))
fn = get_first_fn (fn);
if (!DECL_P (fn))
;
else if (DECL_CLASS_SCOPE_P (fn))
{
location_t loc = EXPR_LOC_OR_LOC (t,
input_location);
inform (loc,
"declarations in dependent base %qT are "
"not found by unqualified lookup",
DECL_CLASS_CONTEXT (fn));
if (current_class_ptr)
inform (loc,
"use %<this->%D%> instead", function);
else
inform (loc,
"use %<%T::%D%> instead",
current_class_name, function);
}
else
inform (DECL_SOURCE_LOCATION (fn),
"%qD declared here, later in the "
"translation unit", fn);
if (in_lambda)
{
release_tree_vector (call_args);
RETURN (error_mark_node);
}
}
function = unq;
}
}
if (identifier_p (function))
{
if (complain & tf_error)
unqualified_name_lookup_error (function);
release_tree_vector (call_args);
RETURN (error_mark_node);
}
}
if (function != NULL_TREE
&& DECL_P (function)
&& !mark_used (function, complain) && !(complain & tf_error))
{
release_tree_vector (call_args);
RETURN (error_mark_node);
}
complain |= decltype_flag;
if (function == NULL_TREE)
switch (CALL_EXPR_IFN (t))
{
case IFN_LAUNDER:
gcc_assert (nargs == 1);
if (vec_safe_length (call_args) != 1)
{
error_at (EXPR_LOC_OR_LOC (t, input_location),
"wrong number of arguments to "
"%<__builtin_launder%>");
ret = error_mark_node;
}
else
ret = finish_builtin_launder (EXPR_LOC_OR_LOC (t,
input_location),
(*call_args)[0], complain);
break;
default:
gcc_unreachable ();
}
else if (TREE_CODE (function) == OFFSET_REF
|| TREE_CODE (function) == DOTSTAR_EXPR
|| TREE_CODE (function) == MEMBER_REF)
ret = build_offset_ref_call_from_tree (function, &call_args,
complain);
else if (TREE_CODE (function) == COMPONENT_REF)
{
tree instance = TREE_OPERAND (function, 0);
tree fn = TREE_OPERAND (function, 1);
if (processing_template_decl
&& (type_dependent_expression_p (instance)
|| (!BASELINK_P (fn)
&& TREE_CODE (fn) != FIELD_DECL)
|| type_dependent_expression_p (fn)
|| any_type_dependent_arguments_p (call_args)))
ret = build_min_nt_call_vec (function, call_args);
else if (!BASELINK_P (fn))
ret = finish_call_expr (function, &call_args,
false,
false,
complain);
else
ret = (build_new_method_call
(instance, fn,
&call_args, NULL_TREE,
qualified_p ? LOOKUP_NONVIRTUAL : LOOKUP_NORMAL,
NULL,
complain));
}
else
ret = finish_call_expr (function, &call_args,
qualified_p,
koenig_p,
complain);
release_tree_vector (call_args);
if (ret != error_mark_node)
{
bool op = CALL_EXPR_OPERATOR_SYNTAX (t);
bool ord = CALL_EXPR_ORDERED_ARGS (t);
bool rev = CALL_EXPR_REVERSE_ARGS (t);
bool thk = CALL_FROM_THUNK_P (t);
if (op || ord || rev || thk)
{
function = extract_call_expr (ret);
CALL_EXPR_OPERATOR_SYNTAX (function) = op;
CALL_EXPR_ORDERED_ARGS (function) = ord;
CALL_EXPR_REVERSE_ARGS (function) = rev;
if (thk)
{
SET_EXPR_LOCATION (function, UNKNOWN_LOCATION);
}
}
}
RETURN (ret);
}
case COND_EXPR:
{
tree cond = RECUR (TREE_OPERAND (t, 0));
cond = mark_rvalue_use (cond);
tree folded_cond = fold_non_dependent_expr (cond);
tree exp1, exp2;
if (TREE_CODE (folded_cond) == INTEGER_CST)
{
if (integer_zerop (folded_cond))
{
++c_inhibit_evaluation_warnings;
exp1 = RECUR (TREE_OPERAND (t, 1));
--c_inhibit_evaluation_warnings;
exp2 = RECUR (TREE_OPERAND (t, 2));
}
else
{
exp1 = RECUR (TREE_OPERAND (t, 1));
++c_inhibit_evaluation_warnings;
exp2 = RECUR (TREE_OPERAND (t, 2));
--c_inhibit_evaluation_warnings;
}
cond = folded_cond;
}
else
{
exp1 = RECUR (TREE_OPERAND (t, 1));
exp2 = RECUR (TREE_OPERAND (t, 2));
}
warning_sentinel s(warn_duplicated_branches);
RETURN (build_x_conditional_expr (EXPR_LOCATION (t),
cond, exp1, exp2, complain));
}
case PSEUDO_DTOR_EXPR:
{
tree op0 = RECUR (TREE_OPERAND (t, 0));
tree op1 = RECUR (TREE_OPERAND (t, 1));
tree op2 = tsubst (TREE_OPERAND (t, 2), args, complain, in_decl);
RETURN (finish_pseudo_destructor_expr (op0, op1, op2,
input_location));
}
case TREE_LIST:
{
tree purpose, value, chain;
if (t == void_list_node)
RETURN (t);
if ((TREE_PURPOSE (t) && PACK_EXPANSION_P (TREE_PURPOSE (t)))
|| (TREE_VALUE (t) && PACK_EXPANSION_P (TREE_VALUE (t))))
{
tree purposevec = NULL_TREE;
tree valuevec = NULL_TREE;
tree chain;
int i, len = -1;
if (TREE_PURPOSE (t))
purposevec = tsubst_pack_expansion (TREE_PURPOSE (t), args,
complain, in_decl);
if (TREE_VALUE (t))
valuevec = tsubst_pack_expansion (TREE_VALUE (t), args,
complain, in_decl);
chain = TREE_CHAIN (t);
if (chain && chain != void_type_node)
chain = RECUR (chain);
if (purposevec && TREE_CODE (purposevec) == TREE_VEC)
{
len = TREE_VEC_LENGTH (purposevec);
gcc_assert (!valuevec || len == TREE_VEC_LENGTH (valuevec));
}
else if (TREE_CODE (valuevec) == TREE_VEC)
len = TREE_VEC_LENGTH (valuevec);
else
{
if (purposevec == TREE_PURPOSE (t)
&& valuevec == TREE_VALUE (t)
&& chain == TREE_CHAIN (t))
RETURN (t);
RETURN (tree_cons (purposevec, valuevec, chain));
}
i = len;
while (i > 0)
{
i--;
purpose = purposevec ? TREE_VEC_ELT (purposevec, i) 
: NULL_TREE;
value 
= valuevec ? convert_from_reference (TREE_VEC_ELT (valuevec, i)) 
: NULL_TREE;
chain = tree_cons (purpose, value, chain);
}
RETURN (chain);
}
purpose = TREE_PURPOSE (t);
if (purpose)
purpose = RECUR (purpose);
value = TREE_VALUE (t);
if (value)
value = RECUR (value);
chain = TREE_CHAIN (t);
if (chain && chain != void_type_node)
chain = RECUR (chain);
if (purpose == TREE_PURPOSE (t)
&& value == TREE_VALUE (t)
&& chain == TREE_CHAIN (t))
RETURN (t);
RETURN (tree_cons (purpose, value, chain));
}
case COMPONENT_REF:
{
tree object;
tree object_type;
tree member;
tree r;
object = tsubst_non_call_postfix_expression (TREE_OPERAND (t, 0),
args, complain, in_decl);
if (DECL_P (object)
&& !mark_used (object, complain) && !(complain & tf_error))
RETURN (error_mark_node);
object_type = TREE_TYPE (object);
member = TREE_OPERAND (t, 1);
if (BASELINK_P (member))
member = tsubst_baselink (member,
non_reference (TREE_TYPE (object)),
args, complain, in_decl);
else
member = tsubst_copy (member, args, complain, in_decl);
if (member == error_mark_node)
RETURN (error_mark_node);
if (TREE_CODE (member) == FIELD_DECL)
{
r = finish_non_static_data_member (member, object, NULL_TREE);
if (TREE_CODE (r) == COMPONENT_REF)
REF_PARENTHESIZED_P (r) = REF_PARENTHESIZED_P (t);
RETURN (r);
}
else if (type_dependent_expression_p (object))
;
else if (!CLASS_TYPE_P (object_type))
{
if (scalarish_type_p (object_type))
{
tree s = NULL_TREE;
tree dtor = member;
if (TREE_CODE (dtor) == SCOPE_REF)
{
s = TREE_OPERAND (dtor, 0);
dtor = TREE_OPERAND (dtor, 1);
}
if (TREE_CODE (dtor) == BIT_NOT_EXPR)
{
dtor = TREE_OPERAND (dtor, 0);
if (TYPE_P (dtor))
RETURN (finish_pseudo_destructor_expr
(object, s, dtor, input_location));
}
}
}
else if (TREE_CODE (member) == SCOPE_REF
&& TREE_CODE (TREE_OPERAND (member, 1)) == TEMPLATE_ID_EXPR)
{
tree scope = TREE_OPERAND (member, 0);
tree tmpl = TREE_OPERAND (TREE_OPERAND (member, 1), 0);
tree args = TREE_OPERAND (TREE_OPERAND (member, 1), 1);
member = lookup_qualified_name (scope, tmpl,
false,
false);
if (BASELINK_P (member))
{
BASELINK_FUNCTIONS (member)
= build_nt (TEMPLATE_ID_EXPR, BASELINK_FUNCTIONS (member),
args);
member = (adjust_result_of_qualified_name_lookup
(member, BINFO_TYPE (BASELINK_BINFO (member)),
object_type));
}
else
{
qualified_name_lookup_error (scope, tmpl, member,
input_location);
RETURN (error_mark_node);
}
}
else if (TREE_CODE (member) == SCOPE_REF
&& !CLASS_TYPE_P (TREE_OPERAND (member, 0))
&& TREE_CODE (TREE_OPERAND (member, 0)) != NAMESPACE_DECL)
{
if (complain & tf_error)
{
if (TYPE_P (TREE_OPERAND (member, 0)))
error ("%qT is not a class or namespace",
TREE_OPERAND (member, 0));
else
error ("%qD is not a class or namespace",
TREE_OPERAND (member, 0));
}
RETURN (error_mark_node);
}
r = finish_class_member_access_expr (object, member,
false,
complain);
if (TREE_CODE (r) == COMPONENT_REF)
REF_PARENTHESIZED_P (r) = REF_PARENTHESIZED_P (t);
RETURN (r);
}
case THROW_EXPR:
RETURN (build_throw
(RECUR (TREE_OPERAND (t, 0))));
case CONSTRUCTOR:
{
vec<constructor_elt, va_gc> *n;
constructor_elt *ce;
unsigned HOST_WIDE_INT idx;
tree type = tsubst (TREE_TYPE (t), args, complain, in_decl);
bool process_index_p;
int newlen;
bool need_copy_p = false;
tree r;
if (type == error_mark_node)
RETURN (error_mark_node);
process_index_p = !(type && MAYBE_CLASS_TYPE_P (type));
n = vec_safe_copy (CONSTRUCTOR_ELTS (t));
newlen = vec_safe_length (n);
FOR_EACH_VEC_SAFE_ELT (n, idx, ce)
{
if (ce->index && process_index_p
&& TREE_CODE (ce->index) != IDENTIFIER_NODE)
ce->index = RECUR (ce->index);
if (PACK_EXPANSION_P (ce->value))
{
ce->value = tsubst_pack_expansion (ce->value, args, complain,
in_decl);
if (ce->value == error_mark_node
|| PACK_EXPANSION_P (ce->value))
;
else if (TREE_VEC_LENGTH (ce->value) == 1)
ce->value = TREE_VEC_ELT (ce->value, 0);
else
{
newlen = newlen + TREE_VEC_LENGTH (ce->value) - 1;
need_copy_p = true;
}
}
else
ce->value = RECUR (ce->value);
}
if (need_copy_p)
{
vec<constructor_elt, va_gc> *old_n = n;
vec_alloc (n, newlen);
FOR_EACH_VEC_ELT (*old_n, idx, ce)
{
if (TREE_CODE (ce->value) == TREE_VEC)
{
int i, len = TREE_VEC_LENGTH (ce->value);
for (i = 0; i < len; ++i)
CONSTRUCTOR_APPEND_ELT (n, 0,
TREE_VEC_ELT (ce->value, i));
}
else
CONSTRUCTOR_APPEND_ELT (n, 0, ce->value);
}
}
r = build_constructor (init_list_type_node, n);
CONSTRUCTOR_IS_DIRECT_INIT (r) = CONSTRUCTOR_IS_DIRECT_INIT (t);
if (TREE_HAS_CONSTRUCTOR (t))
{
fcl_t cl = fcl_functional;
if (CONSTRUCTOR_C99_COMPOUND_LITERAL (t))
cl = fcl_c99;
RETURN (finish_compound_literal (type, r, complain, cl));
}
TREE_TYPE (r) = type;
RETURN (r);
}
case TYPEID_EXPR:
{
tree operand_0 = TREE_OPERAND (t, 0);
if (TYPE_P (operand_0))
{
operand_0 = tsubst (operand_0, args, complain, in_decl);
RETURN (get_typeid (operand_0, complain));
}
else
{
operand_0 = RECUR (operand_0);
RETURN (build_typeid (operand_0, complain));
}
}
case VAR_DECL:
if (!args)
RETURN (t);
case PARM_DECL:
{
tree r = tsubst_copy (t, args, complain, in_decl);
if (VAR_P (r)
&& !processing_template_decl
&& !cp_unevaluated_operand
&& (TREE_STATIC (r) || DECL_EXTERNAL (r))
&& CP_DECL_THREAD_LOCAL_P (r))
{
if (tree wrap = get_tls_wrapper_fn (r))
r = build_cxx_call (wrap, 0, NULL, tf_warning_or_error);
}
else if (outer_automatic_var_p (r))
r = process_outer_var_ref (r, complain);
if (TREE_CODE (TREE_TYPE (t)) != REFERENCE_TYPE)
r = convert_from_reference (r);
RETURN (r);
}
case VA_ARG_EXPR:
{
tree op0 = RECUR (TREE_OPERAND (t, 0));
tree type = tsubst (TREE_TYPE (t), args, complain, in_decl);
RETURN (build_x_va_arg (EXPR_LOCATION (t), op0, type));
}
case OFFSETOF_EXPR:
{
tree object_ptr
= tsubst_copy_and_build (TREE_OPERAND (t, 1), args, complain,
in_decl, false,
false);
RETURN (finish_offsetof (object_ptr,
RECUR (TREE_OPERAND (t, 0)),
EXPR_LOCATION (t)));
}
case ADDRESSOF_EXPR:
RETURN (cp_build_addressof (EXPR_LOCATION (t),
RECUR (TREE_OPERAND (t, 0)), complain));
case TRAIT_EXPR:
{
tree type1 = tsubst (TRAIT_EXPR_TYPE1 (t), args,
complain, in_decl);
tree type2 = TRAIT_EXPR_TYPE2 (t);
if (type2 && TREE_CODE (type2) == TREE_LIST)
type2 = RECUR (type2);
else if (type2)
type2 = tsubst (type2, args, complain, in_decl);
RETURN (finish_trait_expr (TRAIT_EXPR_KIND (t), type1, type2));
}
case STMT_EXPR:
{
tree old_stmt_expr = cur_stmt_expr;
tree stmt_expr = begin_stmt_expr ();
cur_stmt_expr = stmt_expr;
tsubst_expr (STMT_EXPR_STMT (t), args, complain, in_decl,
integral_constant_expression_p);
stmt_expr = finish_stmt_expr (stmt_expr, false);
cur_stmt_expr = old_stmt_expr;
if (empty_expr_stmt_p (stmt_expr))
stmt_expr = void_node;
RETURN (stmt_expr);
}
case LAMBDA_EXPR:
{
tree r = tsubst_lambda_expr (t, args, complain, in_decl);
RETURN (build_lambda_object (r));
}
case TARGET_EXPR:
{
tree r = get_target_expr_sfinae (RECUR (TARGET_EXPR_INITIAL (t)),
complain);
RETURN (r);
}
case TRANSACTION_EXPR:
RETURN (tsubst_expr(t, args, complain, in_decl,
integral_constant_expression_p));
case PAREN_EXPR:
RETURN (finish_parenthesized_expr (RECUR (TREE_OPERAND (t, 0))));
case VEC_PERM_EXPR:
{
tree op0 = RECUR (TREE_OPERAND (t, 0));
tree op1 = RECUR (TREE_OPERAND (t, 1));
tree op2 = RECUR (TREE_OPERAND (t, 2));
RETURN (build_x_vec_perm_expr (input_location, op0, op1, op2,
complain));
}
case REQUIRES_EXPR:
RETURN (tsubst_requires_expr (t, args, complain, in_decl));
case RANGE_EXPR:
RETURN (t);
case NON_LVALUE_EXPR:
case VIEW_CONVERT_EXPR:
gcc_assert (location_wrapper_p (t) || args == NULL_TREE);
if (location_wrapper_p (t))
RETURN (maybe_wrap_with_location (RECUR (TREE_OPERAND (t, 0)),
EXPR_LOCATION (t)));
default:
{
tree subst
= objcp_tsubst_copy_and_build (t, args, complain,
in_decl, false);
if (subst)
RETURN (subst);
}
RETURN (tsubst_copy (t, args, complain, in_decl));
}
#undef RECUR
#undef RETURN
out:
input_location = loc;
return retval;
}
static bool
check_instantiated_arg (tree tmpl, tree t, tsubst_flags_t complain)
{
if (dependent_template_arg_p (t))
return false;
if (ARGUMENT_PACK_P (t))
{
tree vec = ARGUMENT_PACK_ARGS (t);
int len = TREE_VEC_LENGTH (vec);
bool result = false;
int i;
for (i = 0; i < len; ++i)
if (check_instantiated_arg (tmpl, TREE_VEC_ELT (vec, i), complain))
result = true;
return result;
}
else if (TYPE_P (t))
{
tree nt = (cxx_dialect > cxx98 ? NULL_TREE
: no_linkage_check (t, false));
if (nt)
{
if (complain & tf_error)
{
if (TYPE_UNNAMED_P (nt))
error ("%qT is/uses unnamed type", t);
else
error ("template argument for %qD uses local type %qT",
tmpl, t);
}
return true;
}
else if (variably_modified_type_p (t, NULL_TREE))
{
if (complain & tf_error)
error ("%qT is a variably modified type", t);
return true;
}
}
else if (DECL_TYPE_TEMPLATE_P (t))
;
else if (TREE_TYPE (t)
&& INTEGRAL_OR_ENUMERATION_TYPE_P (TREE_TYPE (t))
&& !REFERENCE_REF_P (t)
&& !TREE_CONSTANT (t))
{
if (complain & tf_error)
error ("integral expression %qE is not constant", t);
return true;
}
return false;
}
static bool
check_instantiated_args (tree tmpl, tree args, tsubst_flags_t complain)
{
int ix, len = DECL_NTPARMS (tmpl);
bool result = false;
for (ix = 0; ix != len; ix++)
{
if (check_instantiated_arg (tmpl, TREE_VEC_ELT (args, ix), complain))
result = true;
}
if (result && (complain & tf_error))
error ("  trying to instantiate %qD", tmpl);
return result;
}
static void
recheck_decl_substitution (tree d, tree tmpl, tree args)
{
tree pattern = DECL_TEMPLATE_RESULT (tmpl);
tree type = TREE_TYPE (pattern);
location_t loc = input_location;
push_access_scope (d);
push_deferring_access_checks (dk_no_deferred);
input_location = DECL_SOURCE_LOCATION (pattern);
tsubst (type, args, tf_warning_or_error, d);
input_location = loc;
pop_deferring_access_checks ();
pop_access_scope (d);
}
static tree
instantiate_template_1 (tree tmpl, tree orig_args, tsubst_flags_t complain)
{
tree targ_ptr = orig_args;
tree fndecl;
tree gen_tmpl;
tree spec;
bool access_ok = true;
if (tmpl == error_mark_node)
return error_mark_node;
gcc_assert (TREE_CODE (tmpl) == TEMPLATE_DECL);
if (DECL_CLONED_FUNCTION_P (tmpl))
{
tree spec;
tree clone;
spec = instantiate_template (DECL_ABSTRACT_ORIGIN (tmpl),
targ_ptr, complain);
if (spec == error_mark_node)
return error_mark_node;
FOR_EACH_CLONE (clone, spec)
if (DECL_NAME (clone) == DECL_NAME (tmpl))
return clone;
gcc_unreachable ();
return NULL_TREE;
}
if (targ_ptr == error_mark_node)
return error_mark_node;
gen_tmpl = most_general_template (tmpl);
if (TMPL_ARGS_DEPTH (targ_ptr)
< TMPL_PARMS_DEPTH (DECL_TEMPLATE_PARMS (gen_tmpl)))
targ_ptr = (add_outermost_template_args
(DECL_TI_ARGS (DECL_TEMPLATE_RESULT (tmpl)),
targ_ptr));
spec = retrieve_specialization (gen_tmpl, targ_ptr, 0);
gcc_assert (tmpl == gen_tmpl
|| ((fndecl = retrieve_specialization (tmpl, orig_args, 0))
== spec)
|| fndecl == NULL_TREE);
if (spec != NULL_TREE)
{
if (FNDECL_HAS_ACCESS_ERRORS (spec))
{
if (complain & tf_error)
recheck_decl_substitution (spec, gen_tmpl, targ_ptr);
return error_mark_node;
}
return spec;
}
if (check_instantiated_args (gen_tmpl, INNERMOST_TEMPLATE_ARGS (targ_ptr),
complain))
return error_mark_node;
push_deferring_access_checks (dk_deferred);
push_to_top_level ();
if (uses_template_parms (targ_ptr))
++processing_template_decl;
if (DECL_CLASS_SCOPE_P (gen_tmpl))
{
tree ctx = tsubst_aggr_type (DECL_CONTEXT (gen_tmpl), targ_ptr,
complain, gen_tmpl, true);
push_nested_class (ctx);
}
tree pattern = DECL_TEMPLATE_RESULT (gen_tmpl);
fndecl = NULL_TREE;
if (VAR_P (pattern))
{
tree tid = lookup_template_variable (gen_tmpl, targ_ptr);
tree elt = most_specialized_partial_spec (tid, complain);
if (elt == error_mark_node)
pattern = error_mark_node;
else if (elt)
{
tree partial_tmpl = TREE_VALUE (elt);
tree partial_args = TREE_PURPOSE (elt);
tree partial_pat = DECL_TEMPLATE_RESULT (partial_tmpl);
fndecl = tsubst (partial_pat, partial_args, complain, gen_tmpl);
}
}
if (fndecl == NULL_TREE)
fndecl = tsubst (pattern, targ_ptr, complain, gen_tmpl);
if (DECL_CLASS_SCOPE_P (gen_tmpl))
pop_nested_class ();
pop_from_top_level ();
if (fndecl == error_mark_node)
{
pop_deferring_access_checks ();
return error_mark_node;
}
DECL_TI_TEMPLATE (fndecl) = tmpl;
DECL_TI_ARGS (fndecl) = targ_ptr;
if (!(flag_new_inheriting_ctors
&& DECL_INHERITED_CTOR (fndecl)))
{
push_access_scope (fndecl);
if (!perform_deferred_access_checks (complain))
access_ok = false;
pop_access_scope (fndecl);
}
pop_deferring_access_checks ();
if (DECL_CHAIN (gen_tmpl) && DECL_CLONED_FUNCTION_P (DECL_CHAIN (gen_tmpl)))
clone_function_decl (fndecl, false);
if (!access_ok)
{
if (!(complain & tf_error))
{
FNDECL_HAS_ACCESS_ERRORS (fndecl) = true;
}
return error_mark_node;
}
return fndecl;
}
tree
instantiate_template (tree tmpl, tree orig_args, tsubst_flags_t complain)
{
tree ret;
timevar_push (TV_TEMPLATE_INST);
ret = instantiate_template_1 (tmpl, orig_args,  complain);
timevar_pop (TV_TEMPLATE_INST);
return ret;
}
static tree
instantiate_alias_template (tree tmpl, tree args, tsubst_flags_t complain)
{
if (tmpl == error_mark_node || args == error_mark_node)
return error_mark_node;
if (!push_tinst_level (tmpl, args))
return error_mark_node;
args =
coerce_innermost_template_parms (DECL_TEMPLATE_PARMS (tmpl),
args, tmpl, complain,
true,
true);
tree r = instantiate_template (tmpl, args, complain);
pop_tinst_level ();
return r;
}
static bool
pack_deducible_p (tree parm, tree fn)
{
tree t = FUNCTION_FIRST_USER_PARMTYPE (fn);
for (; t; t = TREE_CHAIN (t))
{
tree type = TREE_VALUE (t);
tree packs;
if (!PACK_EXPANSION_P (type))
continue;
for (packs = PACK_EXPANSION_PARAMETER_PACKS (type);
packs; packs = TREE_CHAIN (packs))
if (template_args_equal (TREE_VALUE (packs), parm))
{
if (TREE_CHAIN (t) == void_list_node)
return true;
else
return false;
}
}
return true;
}
tree
fn_type_unification (tree fn,
tree explicit_targs,
tree targs,
const tree *args,
unsigned int nargs,
tree return_type,
unification_kind_t strict,
int flags,
bool explain_p,
bool decltype_p)
{
tree parms;
tree fntype;
tree decl = NULL_TREE;
tsubst_flags_t complain = (explain_p ? tf_warning_or_error : tf_none);
bool ok;
static int deduction_depth;
tree orig_fn = fn;
if (flag_new_inheriting_ctors)
fn = strip_inheriting_ctors (fn);
tree tparms = DECL_INNERMOST_TEMPLATE_PARMS (fn);
tree r = error_mark_node;
tree full_targs = targs;
if (TMPL_ARGS_DEPTH (targs)
< TMPL_PARMS_DEPTH (DECL_TEMPLATE_PARMS (fn)))
full_targs = (add_outermost_template_args
(DECL_TI_ARGS (DECL_TEMPLATE_RESULT (fn)),
targs));
if (decltype_p)
complain |= tf_decltype;
if (excessive_deduction_depth)
return error_mark_node;
++deduction_depth;
gcc_assert (TREE_CODE (fn) == TEMPLATE_DECL);
fntype = TREE_TYPE (fn);
if (explicit_targs)
{
int i, len = TREE_VEC_LENGTH (tparms);
location_t loc = input_location;
bool incomplete = false;
if (explicit_targs == error_mark_node)
goto fail;
if (TMPL_ARGS_DEPTH (explicit_targs)
< TMPL_ARGS_DEPTH (full_targs))
explicit_targs = add_outermost_template_args (full_targs,
explicit_targs);
explicit_targs
= (coerce_template_parms (tparms, explicit_targs, NULL_TREE,
complain,
false,
false));
if (explicit_targs == error_mark_node)
goto fail;
for (i = 0; i < len; i++)
{
tree parm = TREE_VALUE (TREE_VEC_ELT (tparms, i));
bool parameter_pack = false;
tree targ = TREE_VEC_ELT (explicit_targs, i);
if (TREE_CODE (parm) == TYPE_DECL
|| TREE_CODE (parm) == TEMPLATE_DECL)
{
parm = TREE_TYPE (parm);
parameter_pack = TEMPLATE_TYPE_PARAMETER_PACK (parm);
}
else if (TREE_CODE (parm) == PARM_DECL)
{
parm = DECL_INITIAL (parm);
parameter_pack = TEMPLATE_PARM_PARAMETER_PACK (parm);
}
if (!parameter_pack && targ == NULL_TREE)
incomplete = true;
if (parameter_pack && pack_deducible_p (parm, fn))
{
if (targ)
{
ARGUMENT_PACK_INCOMPLETE_P(targ) = 1;
ARGUMENT_PACK_EXPLICIT_ARGS (targ) 
= ARGUMENT_PACK_ARGS (targ);
}
incomplete = true;
}
}
if (!push_tinst_level (fn, explicit_targs))
{
excessive_deduction_depth = true;
goto fail;
}
processing_template_decl += incomplete;
input_location = DECL_SOURCE_LOCATION (fn);
push_deferring_access_checks (dk_deferred);
fntype = tsubst (TREE_TYPE (fn), explicit_targs,
complain | tf_partial | tf_fndecl_type, NULL_TREE);
pop_deferring_access_checks ();
input_location = loc;
processing_template_decl -= incomplete;
pop_tinst_level ();
if (fntype == error_mark_node)
goto fail;
explicit_targs = INNERMOST_TEMPLATE_ARGS (explicit_targs);
for (i = NUM_TMPL_ARGS (explicit_targs); i--;)
TREE_VEC_ELT (targs, i) = TREE_VEC_ELT (explicit_targs, i);
}
parms = skip_artificial_parms_for (fn, TYPE_ARG_TYPES (fntype));
if (return_type && strict == DEDUCE_CALL)
{
if (POINTER_TYPE_P (return_type))
return_type = TREE_TYPE (return_type);
parms = TYPE_ARG_TYPES (return_type);
}
else if (return_type)
{
tree *new_args;
parms = tree_cons (NULL_TREE, TREE_TYPE (fntype), parms);
new_args = XALLOCAVEC (tree, nargs + 1);
new_args[0] = return_type;
memcpy (new_args + 1, args, nargs * sizeof (tree));
args = new_args;
++nargs;
}
if (!explain_p && !push_tinst_level (fn, targs))
{
excessive_deduction_depth = true;
goto fail;
}
vec<deferred_access_check, va_gc> *checks;
checks = NULL;
ok = !type_unification_real (DECL_INNERMOST_TEMPLATE_PARMS (fn),
full_targs, parms, args, nargs, 0,
strict, flags, &checks, explain_p);
if (!explain_p)
pop_tinst_level ();
if (!ok)
goto fail;
if (!template_template_parm_bindings_ok_p
(DECL_INNERMOST_TEMPLATE_PARMS (fn), targs))
{
unify_inconsistent_template_template_parameters (explain_p);
goto fail;
}
if (!push_tinst_level (fn, targs))
{
excessive_deduction_depth = true;
goto fail;
}
reopen_deferring_access_checks (checks);
decl = instantiate_template (fn, targs, complain);
checks = get_deferred_access_checks ();
pop_deferring_access_checks ();
pop_tinst_level ();
if (decl == error_mark_node)
goto fail;
push_access_scope (decl);
ok = perform_access_checks (checks, complain);
pop_access_scope (decl);
if (!ok)
goto fail;
if (strict == DEDUCE_EXACT && !any_dependent_template_arguments_p (targs))
{
tree substed = TREE_TYPE (decl);
unsigned int i;
tree sarg
= skip_artificial_parms_for (decl, TYPE_ARG_TYPES (substed));
if (return_type)
sarg = tree_cons (NULL_TREE, TREE_TYPE (substed), sarg);
for (i = 0; i < nargs && sarg; ++i, sarg = TREE_CHAIN (sarg))
if (!same_type_p (args[i], TREE_VALUE (sarg)))
{
unify_type_mismatch (explain_p, args[i],
TREE_VALUE (sarg));
goto fail;
}
}
if (orig_fn != fn)
decl = instantiate_template (orig_fn, targs, complain);
r = decl;
fail:
--deduction_depth;
if (excessive_deduction_depth)
{
if (deduction_depth == 0)
excessive_deduction_depth = false;
}
return r;
}
static int
maybe_adjust_types_for_deduction (unification_kind_t strict,
tree* parm,
tree* arg,
tree arg_expr)
{
int result = 0;
switch (strict)
{
case DEDUCE_CALL:
break;
case DEDUCE_CONV:
std::swap (parm, arg);
break;
case DEDUCE_EXACT:
if (TREE_CODE (*parm) == REFERENCE_TYPE
&& TYPE_REF_IS_RVALUE (*parm)
&& TREE_CODE (TREE_TYPE (*parm)) == TEMPLATE_TYPE_PARM
&& cp_type_quals (TREE_TYPE (*parm)) == TYPE_UNQUALIFIED
&& TREE_CODE (*arg) == REFERENCE_TYPE
&& !TYPE_REF_IS_RVALUE (*arg))
*parm = TREE_TYPE (*parm);
return 0;
default:
gcc_unreachable ();
}
if (TREE_CODE (*parm) != REFERENCE_TYPE)
{
if (TREE_CODE (*arg) == ARRAY_TYPE)
*arg = build_pointer_type (TREE_TYPE (*arg));
else if (TREE_CODE (*arg) == FUNCTION_TYPE)
*arg = build_pointer_type (*arg);
else
*arg = TYPE_MAIN_VARIANT (*arg);
}
if (TREE_CODE (*parm) == REFERENCE_TYPE
&& TYPE_REF_IS_RVALUE (*parm)
&& TREE_CODE (TREE_TYPE (*parm)) == TEMPLATE_TYPE_PARM
&& !TEMPLATE_TYPE_PARM_FOR_CLASS (TREE_TYPE (*parm))
&& cp_type_quals (TREE_TYPE (*parm)) == TYPE_UNQUALIFIED
&& (arg_expr ? lvalue_p (arg_expr)
: TREE_CODE (*arg) == FUNCTION_TYPE))
*arg = build_reference_type (*arg);
*parm = TYPE_MAIN_VARIANT (*parm);
if (TREE_CODE (*parm) == REFERENCE_TYPE)
{
*parm = TREE_TYPE (*parm);
result |= UNIFY_ALLOW_OUTER_MORE_CV_QUAL;
}
if (strict == DEDUCE_CONV && TREE_CODE (*arg) == REFERENCE_TYPE)
*arg = TREE_TYPE (*arg);
return result;
}
static int
check_non_deducible_conversion (tree parm, tree arg, int strict,
int flags, bool explain_p)
{
tree type;
if (!TYPE_P (arg))
type = TREE_TYPE (arg);
else
type = arg;
if (same_type_p (parm, type))
return unify_success (explain_p);
if (strict == DEDUCE_CONV)
{
if (can_convert_arg (type, parm, NULL_TREE, flags,
explain_p ? tf_warning_or_error : tf_none))
return unify_success (explain_p);
}
else if (strict != DEDUCE_EXACT)
{
if (can_convert_arg (parm, type,
TYPE_P (arg) ? NULL_TREE : arg,
flags, explain_p ? tf_warning_or_error : tf_none))
return unify_success (explain_p);
}
if (strict == DEDUCE_EXACT)
return unify_type_mismatch (explain_p, parm, arg);
else
return unify_arg_conversion (explain_p, parm, type, arg);
}
static bool uses_deducible_template_parms (tree type);
static bool
deducible_expression (tree expr)
{
while (CONVERT_EXPR_P (expr))
expr = TREE_OPERAND (expr, 0);
return (TREE_CODE (expr) == TEMPLATE_PARM_INDEX);
}
static bool
deducible_array_bound (tree domain)
{
if (domain == NULL_TREE)
return false;
tree max = TYPE_MAX_VALUE (domain);
if (TREE_CODE (max) != MINUS_EXPR)
return false;
return deducible_expression (TREE_OPERAND (max, 0));
}
static bool
deducible_template_args (tree args)
{
for (int i = 0; i < TREE_VEC_LENGTH (args); ++i)
{
bool deducible;
tree elt = TREE_VEC_ELT (args, i);
if (ARGUMENT_PACK_P (elt))
deducible = deducible_template_args (ARGUMENT_PACK_ARGS (elt));
else
{
if (PACK_EXPANSION_P (elt))
elt = PACK_EXPANSION_PATTERN (elt);
if (TREE_CODE (elt) == TEMPLATE_TEMPLATE_PARM)
deducible = true;
else if (TYPE_P (elt))
deducible = uses_deducible_template_parms (elt);
else
deducible = deducible_expression (elt);
}
if (deducible)
return true;
}
return false;
}
static bool
uses_deducible_template_parms (tree type)
{
if (PACK_EXPANSION_P (type))
type = PACK_EXPANSION_PATTERN (type);
if (TREE_CODE (type) == TEMPLATE_TYPE_PARM
|| TREE_CODE (type) == BOUND_TEMPLATE_TEMPLATE_PARM)
return true;
if (POINTER_TYPE_P (type))
return uses_deducible_template_parms (TREE_TYPE (type));
if (TREE_CODE (type) == ARRAY_TYPE)
return (uses_deducible_template_parms (TREE_TYPE (type))
|| deducible_array_bound (TYPE_DOMAIN (type)));
if (TYPE_PTRMEM_P (type))
return (uses_deducible_template_parms (TYPE_PTRMEM_CLASS_TYPE (type))
|| (uses_deducible_template_parms
(TYPE_PTRMEM_POINTED_TO_TYPE (type))));
if (CLASS_TYPE_P (type)
&& CLASSTYPE_TEMPLATE_INFO (type)
&& PRIMARY_TEMPLATE_P (CLASSTYPE_TI_TEMPLATE (type)))
return deducible_template_args (INNERMOST_TEMPLATE_ARGS
(CLASSTYPE_TI_ARGS (type)));
if (TREE_CODE (type) == FUNCTION_TYPE
|| TREE_CODE (type) == METHOD_TYPE)
{
if (uses_deducible_template_parms (TREE_TYPE (type)))
return true;
tree parm = TYPE_ARG_TYPES (type);
if (TREE_CODE (type) == METHOD_TYPE)
parm = TREE_CHAIN (parm);
for (; parm; parm = TREE_CHAIN (parm))
if (uses_deducible_template_parms (TREE_VALUE (parm)))
return true;
}
return false;
}
static int
unify_one_argument (tree tparms, tree targs, tree parm, tree arg,
int subr, unification_kind_t strict,
bool explain_p)
{
tree arg_expr = NULL_TREE;
int arg_strict;
if (arg == error_mark_node || parm == error_mark_node)
return unify_invalid (explain_p);
if (arg == unknown_type_node)
return unify_success (explain_p);
if (strict != DEDUCE_EXACT
&& TYPE_P (parm) && !uses_deducible_template_parms (parm))
return unify_success (explain_p);
switch (strict)
{
case DEDUCE_CALL:
arg_strict = (UNIFY_ALLOW_OUTER_LEVEL
| UNIFY_ALLOW_MORE_CV_QUAL
| UNIFY_ALLOW_DERIVED);
break;
case DEDUCE_CONV:
arg_strict = UNIFY_ALLOW_LESS_CV_QUAL;
break;
case DEDUCE_EXACT:
arg_strict = UNIFY_ALLOW_NONE;
break;
default:
gcc_unreachable ();
}
if (!subr)
{
if (!TYPE_P (arg))
{
gcc_assert (TREE_TYPE (arg) != NULL_TREE);
if (type_unknown_p (arg))
{
resolve_overloaded_unification (tparms, targs, parm,
arg, strict,
arg_strict, explain_p);
return unify_success (explain_p);
}
arg_expr = arg;
arg = unlowered_expr_type (arg);
if (arg == error_mark_node)
return unify_invalid (explain_p);
}
arg_strict |=
maybe_adjust_types_for_deduction (strict, &parm, &arg, arg_expr);
}
else
if ((TYPE_P (parm) || TREE_CODE (parm) == TEMPLATE_DECL)
!= (TYPE_P (arg) || TREE_CODE (arg) == TEMPLATE_DECL))
return unify_template_argument_mismatch (explain_p, parm, arg);
if (arg_expr && BRACE_ENCLOSED_INITIALIZER_P (arg_expr))
arg = arg_expr;
return unify (tparms, targs, parm, arg, arg_strict, explain_p);
}
static int
zero_r (tree, void *)
{
return 0;
}
static int
array_deduction_r (tree t, void *data)
{
tree_pair_p d = (tree_pair_p)data;
tree &tparms = d->purpose;
tree &targs = d->value;
if (TREE_CODE (t) == ARRAY_TYPE)
if (tree dom = TYPE_DOMAIN (t))
if (tree max = TYPE_MAX_VALUE (dom))
{
if (TREE_CODE (max) == MINUS_EXPR)
max = TREE_OPERAND (max, 0);
if (TREE_CODE (max) == TEMPLATE_PARM_INDEX)
unify (tparms, targs, TREE_TYPE (max), size_type_node,
UNIFY_ALLOW_NONE, false);
}
return 0;
}
static void
try_array_deduction (tree tparms, tree targs, tree parm)
{
tree_pair_s data = { tparms, targs };
hash_set<tree> visited;
for_each_template_parm (parm, zero_r, &data, &visited,
false, array_deduction_r);
}
static int
braced_init_depth (tree init)
{
if (!init || !BRACE_ENCLOSED_INITIALIZER_P (init))
return 0;
unsigned i; tree val;
unsigned max = 0;
FOR_EACH_CONSTRUCTOR_VALUE (CONSTRUCTOR_ELTS (init), i, val)
{
unsigned elt_d = braced_init_depth (val);
if (elt_d > max)
max = elt_d;
}
return max + 1;
}
static int
type_unification_real (tree tparms,
tree full_targs,
tree xparms,
const tree *xargs,
unsigned int xnargs,
int subr,
unification_kind_t strict,
int flags,
vec<deferred_access_check, va_gc> **checks,
bool explain_p)
{
tree parm, arg;
int i;
int ntparms = TREE_VEC_LENGTH (tparms);
int saw_undeduced = 0;
tree parms;
const tree *args;
unsigned int nargs;
unsigned int ia;
gcc_assert (TREE_CODE (tparms) == TREE_VEC);
gcc_assert (xparms == NULL_TREE || TREE_CODE (xparms) == TREE_LIST);
gcc_assert (ntparms > 0);
tree targs = INNERMOST_TEMPLATE_ARGS (full_targs);
NON_DEFAULT_TEMPLATE_ARGS_COUNT (targs) = NULL_TREE;
again:
parms = xparms;
args = xargs;
nargs = xnargs;
ia = 0;
while (parms && parms != void_list_node
&& ia < nargs)
{
parm = TREE_VALUE (parms);
if (TREE_CODE (parm) == TYPE_PACK_EXPANSION
&& (!TREE_CHAIN (parms) || TREE_CHAIN (parms) == void_list_node))
break;
parms = TREE_CHAIN (parms);
if (TREE_CODE (parm) == TYPE_PACK_EXPANSION)
continue;
arg = args[ia];
++ia;
if (unify_one_argument (tparms, full_targs, parm, arg, subr, strict,
explain_p))
return 1;
}
if (parms 
&& parms != void_list_node
&& TREE_CODE (TREE_VALUE (parms)) == TYPE_PACK_EXPANSION)
{
tree argvec;
tree parmvec = make_tree_vec (1);
argvec = make_tree_vec (nargs - ia);
for (i = 0; ia < nargs; ++ia, ++i)
TREE_VEC_ELT (argvec, i) = args[ia];
TREE_VEC_ELT (parmvec, 0) = TREE_VALUE (parms);
if (unify_pack_expansion (tparms, full_targs, parmvec, argvec, strict,
subr, explain_p))
return 1;
parms = TREE_CHAIN (parms);
}
if (ia < nargs && parms == void_list_node)
return unify_too_many_arguments (explain_p, nargs, ia);
if (parms && parms != void_list_node
&& TREE_PURPOSE (parms) == NULL_TREE)
{
unsigned int count = nargs;
tree p = parms;
bool type_pack_p;
do
{
type_pack_p = TREE_CODE (TREE_VALUE (p)) == TYPE_PACK_EXPANSION;
if (!type_pack_p)
count++;
p = TREE_CHAIN (p);
}
while (p && p != void_list_node);
if (count != nargs)
return unify_too_few_arguments (explain_p, ia, count,
type_pack_p);
}
if (!subr)
{
tsubst_flags_t complain = (explain_p
? tf_warning_or_error
: tf_none);
bool tried_array_deduction = (cxx_dialect < cxx17);
for (i = 0; i < ntparms; i++)
{
tree targ = TREE_VEC_ELT (targs, i);
tree tparm = TREE_VEC_ELT (tparms, i);
if (targ && ARGUMENT_PACK_P (targ))
{
ARGUMENT_PACK_INCOMPLETE_P (targ) = 0;
ARGUMENT_PACK_EXPLICIT_ARGS (targ) = NULL_TREE;
}
if (targ || tparm == error_mark_node)
continue;
tparm = TREE_VALUE (tparm);
if (TREE_CODE (tparm) == TYPE_DECL
&& !tried_array_deduction)
{
try_array_deduction (tparms, targs, xparms);
tried_array_deduction = true;
if (TREE_VEC_ELT (targs, i))
continue;
}
if (TREE_CODE (tparm) == PARM_DECL
&& uses_template_parms (TREE_TYPE (tparm))
&& saw_undeduced < 2)
{
saw_undeduced = 1;
continue;
}
if (TREE_PURPOSE (TREE_VEC_ELT (tparms, i)))
continue;
if (template_parameter_pack_p (tparm))
{
tree arg;
if (TREE_CODE (tparm) == TEMPLATE_PARM_INDEX)
{
arg = make_node (NONTYPE_ARGUMENT_PACK);
TREE_CONSTANT (arg) = 1;
}
else
arg = cxx_make_type (TYPE_ARGUMENT_PACK);
SET_ARGUMENT_PACK_ARGS (arg, make_tree_vec (0));
TREE_VEC_ELT (targs, i) = arg;
continue;
}
return unify_parameter_deduction_failure (explain_p, tparm);
}
if (saw_undeduced < 2)
for (ia = 0, parms = xparms, args = xargs, nargs = xnargs;
parms && parms != void_list_node && ia < nargs; )
{
parm = TREE_VALUE (parms);
if (TREE_CODE (parm) == TYPE_PACK_EXPANSION
&& (!TREE_CHAIN (parms)
|| TREE_CHAIN (parms) == void_list_node))
break;
parms = TREE_CHAIN (parms);
if (TREE_CODE (parm) == TYPE_PACK_EXPANSION)
continue;
arg = args[ia];
++ia;
if (uses_template_parms (parm))
continue;
if (braced_init_depth (arg) > 2)
continue;
if (check_non_deducible_conversion (parm, arg, strict, flags,
explain_p))
return 1;
}
for (i = 0; i < ntparms; i++)
{
tree targ = TREE_VEC_ELT (targs, i);
tree tparm = TREE_VEC_ELT (tparms, i);
if (targ || tparm == error_mark_node)
continue;
tree parm = TREE_VALUE (tparm);
tree arg = TREE_PURPOSE (tparm);
reopen_deferring_access_checks (*checks);
location_t save_loc = input_location;
if (DECL_P (parm))
input_location = DECL_SOURCE_LOCATION (parm);
if (saw_undeduced == 1
&& TREE_CODE (parm) == PARM_DECL
&& uses_template_parms (TREE_TYPE (parm)))
{
++processing_template_decl;
tree type = tsubst (TREE_TYPE (parm), full_targs, complain,
NULL_TREE);
--processing_template_decl;
if (type == error_mark_node)
arg = error_mark_node;
else
arg = NULL_TREE;
}
else
{
tree substed = NULL_TREE;
if (saw_undeduced == 1 && processing_template_decl == 0)
{
++processing_template_decl;
substed = tsubst_template_arg (arg, full_targs, complain,
NULL_TREE);
--processing_template_decl;
if (substed != error_mark_node
&& !uses_template_parms (substed))
substed = NULL_TREE;
}
if (!substed)
substed = tsubst_template_arg (arg, full_targs, complain,
NULL_TREE);
if (!uses_template_parms (substed))
arg = convert_template_argument (parm, substed, full_targs,
complain, i, NULL_TREE);
else if (saw_undeduced == 1)
arg = NULL_TREE;
else
arg = error_mark_node;
}
input_location = save_loc;
*checks = get_deferred_access_checks ();
pop_deferring_access_checks ();
if (arg == error_mark_node)
return 1;
else if (arg)
{
TREE_VEC_ELT (targs, i) = arg;
if (!NON_DEFAULT_TEMPLATE_ARGS_COUNT (targs))
SET_NON_DEFAULT_TEMPLATE_ARGS_COUNT (targs, i);
}
}
if (saw_undeduced++ == 1)
goto again;
}
if (CHECKING_P && !NON_DEFAULT_TEMPLATE_ARGS_COUNT (targs))
SET_NON_DEFAULT_TEMPLATE_ARGS_COUNT (targs, TREE_VEC_LENGTH (targs));
return unify_success (explain_p);
}
static bool
resolve_overloaded_unification (tree tparms,
tree targs,
tree parm,
tree arg,
unification_kind_t strict,
int sub_strict,
bool explain_p)
{
tree tempargs = copy_node (targs);
int good = 0;
tree goodfn = NULL_TREE;
bool addr_p;
if (TREE_CODE (arg) == ADDR_EXPR)
{
arg = TREE_OPERAND (arg, 0);
addr_p = true;
}
else
addr_p = false;
if (TREE_CODE (arg) == COMPONENT_REF)
arg = TREE_OPERAND (arg, 1);
if (TREE_CODE (arg) == OFFSET_REF)
arg = TREE_OPERAND (arg, 1);
if (BASELINK_P (arg))
arg = BASELINK_FUNCTIONS (arg);
if (TREE_CODE (arg) == TEMPLATE_ID_EXPR)
{
int ok = 0;
tree expl_subargs = TREE_OPERAND (arg, 1);
arg = TREE_OPERAND (arg, 0);
for (lkp_iterator iter (arg); iter; ++iter)
{
tree fn = *iter;
tree subargs, elem;
if (TREE_CODE (fn) != TEMPLATE_DECL)
continue;
subargs = coerce_template_parms (DECL_INNERMOST_TEMPLATE_PARMS (fn),
expl_subargs, NULL_TREE, tf_none,
true,
true);
if (subargs != error_mark_node
&& !any_dependent_template_arguments_p (subargs))
{
elem = TREE_TYPE (instantiate_template (fn, subargs, tf_none));
if (try_one_overload (tparms, targs, tempargs, parm,
elem, strict, sub_strict, addr_p, explain_p)
&& (!goodfn || !same_type_p (goodfn, elem)))
{
goodfn = elem;
++good;
}
}
else if (subargs)
++ok;
}
if (good != 1)
good = ok;
}
else if (TREE_CODE (arg) != OVERLOAD
&& TREE_CODE (arg) != FUNCTION_DECL)
return false;
else
for (lkp_iterator iter (arg); iter; ++iter)
{
tree fn = *iter;
if (try_one_overload (tparms, targs, tempargs, parm, TREE_TYPE (fn),
strict, sub_strict, addr_p, explain_p)
&& (!goodfn || !decls_match (goodfn, fn)))
{
goodfn = fn;
++good;
}
}
if (good == 1)
{
int i = TREE_VEC_LENGTH (targs);
for (; i--; )
if (TREE_VEC_ELT (tempargs, i))
{
tree old = TREE_VEC_ELT (targs, i);
tree new_ = TREE_VEC_ELT (tempargs, i);
if (new_ && old && ARGUMENT_PACK_P (old)
&& ARGUMENT_PACK_EXPLICIT_ARGS (old))
ARGUMENT_PACK_EXPLICIT_ARGS (new_)
= ARGUMENT_PACK_EXPLICIT_ARGS (old);
TREE_VEC_ELT (targs, i) = new_;
}
}
if (good)
return true;
return false;
}
tree
resolve_nondeduced_context (tree orig_expr, tsubst_flags_t complain)
{
tree expr, offset, baselink;
bool addr;
if (!type_unknown_p (orig_expr))
return orig_expr;
expr = orig_expr;
addr = false;
offset = NULL_TREE;
baselink = NULL_TREE;
if (TREE_CODE (expr) == ADDR_EXPR)
{
expr = TREE_OPERAND (expr, 0);
addr = true;
}
if (TREE_CODE (expr) == OFFSET_REF)
{
offset = expr;
expr = TREE_OPERAND (expr, 1);
}
if (BASELINK_P (expr))
{
baselink = expr;
expr = BASELINK_FUNCTIONS (expr);
}
if (TREE_CODE (expr) == TEMPLATE_ID_EXPR)
{
int good = 0;
tree goodfn = NULL_TREE;
tree expl_subargs = TREE_OPERAND (expr, 1);
tree arg = TREE_OPERAND (expr, 0);
tree badfn = NULL_TREE;
tree badargs = NULL_TREE;
for (lkp_iterator iter (arg); iter; ++iter)
{
tree fn = *iter;
tree subargs, elem;
if (TREE_CODE (fn) != TEMPLATE_DECL)
continue;
subargs = coerce_template_parms (DECL_INNERMOST_TEMPLATE_PARMS (fn),
expl_subargs, NULL_TREE, tf_none,
true,
true);
if (subargs != error_mark_node
&& !any_dependent_template_arguments_p (subargs))
{
elem = instantiate_template (fn, subargs, tf_none);
if (elem == error_mark_node)
{
badfn = fn;
badargs = subargs;
}
else if (elem && (!goodfn || !decls_match (goodfn, elem)))
{
goodfn = elem;
++good;
}
}
}
if (good == 1)
{
mark_used (goodfn);
expr = goodfn;
if (baselink)
expr = build_baselink (BASELINK_BINFO (baselink),
BASELINK_ACCESS_BINFO (baselink),
expr, BASELINK_OPTYPE (baselink));
if (offset)
{
tree base
= TYPE_MAIN_VARIANT (TREE_TYPE (TREE_OPERAND (offset, 0)));
expr = build_offset_ref (base, expr, addr, complain);
}
if (addr)
expr = cp_build_addr_expr (expr, complain);
return expr;
}
else if (good == 0 && badargs && (complain & tf_error))
instantiate_template (badfn, badargs, complain);
}
return orig_expr;
}
static int
try_one_overload (tree tparms,
tree orig_targs,
tree targs,
tree parm,
tree arg,
unification_kind_t strict,
int sub_strict,
bool addr_p,
bool explain_p)
{
int nargs;
tree tempargs;
int i;
if (arg == error_mark_node)
return 0;
if (uses_template_parms (arg))
return 1;
if (TREE_CODE (arg) == METHOD_TYPE)
arg = build_ptrmemfunc_type (build_pointer_type (arg));
else if (addr_p)
arg = build_pointer_type (arg);
sub_strict |= maybe_adjust_types_for_deduction (strict, &parm, &arg, NULL);
nargs = TREE_VEC_LENGTH (targs);
tempargs = make_tree_vec (nargs);
if (unify (tparms, tempargs, parm, arg, sub_strict, explain_p))
return 0;
for (i = nargs; i--; )
{
tree elt = TREE_VEC_ELT (tempargs, i);
tree oldelt = TREE_VEC_ELT (orig_targs, i);
if (!elt)
;
else if (uses_template_parms (elt))
TREE_VEC_ELT (tempargs, i) = NULL_TREE;
else if (oldelt && ARGUMENT_PACK_P (oldelt))
{
tree explicit_pack = ARGUMENT_PACK_ARGS (oldelt);
tree deduced_pack = ARGUMENT_PACK_ARGS (elt);
if (TREE_VEC_LENGTH (deduced_pack)
< TREE_VEC_LENGTH (explicit_pack))
return 0;
for (int j = 0; j < TREE_VEC_LENGTH (explicit_pack); j++)
if (!template_args_equal (TREE_VEC_ELT (explicit_pack, j),
TREE_VEC_ELT (deduced_pack, j)))
return 0;
}
else if (oldelt && !template_args_equal (oldelt, elt))
return 0;
}
for (i = nargs; i--; )
{
tree elt = TREE_VEC_ELT (tempargs, i);
if (elt)
TREE_VEC_ELT (targs, i) = elt;
}
return 1;
}
static tree
try_class_unification (tree tparms, tree targs, tree parm, tree arg,
bool explain_p)
{
tree copy_of_targs;
if (!CLASSTYPE_SPECIALIZATION_OF_PRIMARY_TEMPLATE_P (arg))
return NULL_TREE;
else if (TREE_CODE (parm) == BOUND_TEMPLATE_TEMPLATE_PARM)
;
else if (most_general_template (CLASSTYPE_TI_TEMPLATE (arg))
!= most_general_template (CLASSTYPE_TI_TEMPLATE (parm)))
return NULL_TREE;
copy_of_targs = make_tree_vec (TREE_VEC_LENGTH (targs));
if (TREE_CODE (parm) == BOUND_TEMPLATE_TEMPLATE_PARM)
{
if (unify_bound_ttp_args (tparms, copy_of_targs, parm, arg, explain_p))
return NULL_TREE;
return arg;
}
if (unify (tparms, copy_of_targs, CLASSTYPE_TI_ARGS (parm),
CLASSTYPE_TI_ARGS (arg), UNIFY_ALLOW_NONE, explain_p))
return NULL_TREE;
return arg;
}
static enum template_base_result
get_template_base (tree tparms, tree targs, tree parm, tree arg,
bool explain_p, tree *result)
{
tree rval = NULL_TREE;
tree binfo;
gcc_assert (RECORD_OR_UNION_CODE_P (TREE_CODE (arg)));
binfo = TYPE_BINFO (complete_type (arg));
if (!binfo)
{
*result = NULL_TREE;
return tbr_incomplete_type;
}
for (binfo = TREE_CHAIN (binfo); binfo; binfo = TREE_CHAIN (binfo))
{
tree r = try_class_unification (tparms, targs, parm,
BINFO_TYPE (binfo), explain_p);
if (r)
{
if (rval && !same_type_p (r, rval))
{
*result = NULL_TREE;
return tbr_ambiguous_baseclass;
}
rval = r;
}
}
*result = rval;
return tbr_success;
}
static int
template_decl_level (tree decl)
{
switch (TREE_CODE (decl))
{
case TYPE_DECL:
case TEMPLATE_DECL:
return TEMPLATE_TYPE_LEVEL (TREE_TYPE (decl));
case PARM_DECL:
return TEMPLATE_PARM_LEVEL (DECL_INITIAL (decl));
default:
gcc_unreachable ();
}
return 0;
}
static int
check_cv_quals_for_unify (int strict, tree arg, tree parm)
{
int arg_quals = cp_type_quals (arg);
int parm_quals = cp_type_quals (parm);
if (TREE_CODE (parm) == TEMPLATE_TYPE_PARM
&& !(strict & UNIFY_ALLOW_OUTER_MORE_CV_QUAL))
{
if ((TREE_CODE (arg) == REFERENCE_TYPE
|| TREE_CODE (arg) == FUNCTION_TYPE
|| TREE_CODE (arg) == METHOD_TYPE)
&& (parm_quals & (TYPE_QUAL_CONST | TYPE_QUAL_VOLATILE)))
return 0;
if ((!POINTER_TYPE_P (arg) && TREE_CODE (arg) != TEMPLATE_TYPE_PARM)
&& (parm_quals & TYPE_QUAL_RESTRICT))
return 0;
}
if (!(strict & (UNIFY_ALLOW_MORE_CV_QUAL | UNIFY_ALLOW_OUTER_MORE_CV_QUAL))
&& (arg_quals & parm_quals) != parm_quals)
return 0;
if (!(strict & (UNIFY_ALLOW_LESS_CV_QUAL | UNIFY_ALLOW_OUTER_LESS_CV_QUAL))
&& (parm_quals & arg_quals) != arg_quals)
return 0;
return 1;
}
void 
template_parm_level_and_index (tree parm, int* level, int* index)
{
if (TREE_CODE (parm) == TEMPLATE_TYPE_PARM
|| TREE_CODE (parm) == TEMPLATE_TEMPLATE_PARM
|| TREE_CODE (parm) == BOUND_TEMPLATE_TEMPLATE_PARM)
{
*index = TEMPLATE_TYPE_IDX (parm);
*level = TEMPLATE_TYPE_LEVEL (parm);
}
else
{
*index = TEMPLATE_PARM_IDX (parm);
*level = TEMPLATE_PARM_LEVEL (parm);
}
}
#define RECUR_AND_CHECK_FAILURE(TP, TA, P, A, S, EP)			\
do {									\
if (unify (TP, TA, P, A, S, EP))					\
return 1;								\
} while (0)
static int
unify_pack_expansion (tree tparms, tree targs, tree packed_parms, 
tree packed_args, unification_kind_t strict,
bool subr, bool explain_p)
{
tree parm 
= TREE_VEC_ELT (packed_parms, TREE_VEC_LENGTH (packed_parms) - 1);
tree pattern = PACK_EXPANSION_PATTERN (parm);
tree pack, packs = NULL_TREE;
int i, start = TREE_VEC_LENGTH (packed_parms) - 1;
targs = add_to_template_args (PACK_EXPANSION_EXTRA_ARGS (parm), targs);
int levels = TMPL_ARGS_DEPTH (targs);
packed_args = expand_template_argument_pack (packed_args);
int len = TREE_VEC_LENGTH (packed_args);
for (pack = PACK_EXPANSION_PARAMETER_PACKS (parm); 
pack; pack = TREE_CHAIN (pack))
{
tree parm_pack = TREE_VALUE (pack);
int idx, level;
if (!TEMPLATE_PARM_P (parm_pack))
continue;
template_parm_level_and_index (parm_pack, &level, &idx);
if (level < levels)
continue;
packs = tree_cons (parm_pack, TMPL_ARG (targs, level, idx), packs);
TREE_TYPE (packs) = make_tree_vec (len - start);
}
for (i = start; i < len; i++)
{
tree parm;
bool any_explicit = false;
tree arg = TREE_VEC_ELT (packed_args, i);
for (pack = packs; pack; pack = TREE_CHAIN (pack))
{
int idx, level;
tree arg, pargs;
template_parm_level_and_index (TREE_PURPOSE (pack), &level, &idx);
arg = NULL_TREE;
if (TREE_VALUE (pack)
&& (pargs = ARGUMENT_PACK_EXPLICIT_ARGS (TREE_VALUE (pack)))
&& (i - start < TREE_VEC_LENGTH (pargs)))
{
any_explicit = true;
arg = TREE_VEC_ELT (pargs, i - start);
}
TMPL_ARG (targs, level, idx) = arg;
}
if (any_explicit)
{
bool dependent;
++processing_template_decl;
dependent = any_dependent_template_arguments_p (targs);
if (!dependent)
--processing_template_decl;
parm = tsubst (pattern, targs,
explain_p ? tf_warning_or_error : tf_none,
NULL_TREE);
if (dependent)
--processing_template_decl;
if (parm == error_mark_node)
return 1;
}
else
parm = pattern;
if (unify_one_argument (tparms, targs, parm, arg, subr, strict,
explain_p))
return 1;
for (pack = packs; pack; pack = TREE_CHAIN (pack))
{
int idx, level;
template_parm_level_and_index (TREE_PURPOSE (pack), &level, &idx);
TREE_VEC_ELT (TREE_TYPE (pack), i - start) = 
TMPL_ARG (targs, level, idx);
}
}
for (pack = packs; pack; pack = TREE_CHAIN (pack))
{
tree old_pack = TREE_VALUE (pack);
tree new_args = TREE_TYPE (pack);
int i, len = TREE_VEC_LENGTH (new_args);
int idx, level;
bool nondeduced_p = false;
template_parm_level_and_index (TREE_PURPOSE (pack), &level, &idx);
TMPL_ARG (targs, level, idx) = old_pack;
for (i = 0; i < len && !nondeduced_p; ++i)
if (TREE_VEC_ELT (new_args, i) == NULL_TREE)
nondeduced_p = true;
if (nondeduced_p)
continue;
if (old_pack && ARGUMENT_PACK_INCOMPLETE_P (old_pack))
{
tree explicit_args = ARGUMENT_PACK_EXPLICIT_ARGS (old_pack);
int explicit_len = TREE_VEC_LENGTH (explicit_args);
if (len < explicit_len)
new_args = explicit_args;
}
if (!old_pack)
{
tree result;
if (TREE_CODE (TREE_PURPOSE (pack)) == TEMPLATE_PARM_INDEX)
{
result = make_node (NONTYPE_ARGUMENT_PACK);
TREE_CONSTANT (result) = 1;
}
else
result = cxx_make_type (TYPE_ARGUMENT_PACK);
SET_ARGUMENT_PACK_ARGS (result, new_args);
TMPL_ARG (targs, level, idx) = result;
}
else if (ARGUMENT_PACK_INCOMPLETE_P (old_pack)
&& (ARGUMENT_PACK_ARGS (old_pack) 
== ARGUMENT_PACK_EXPLICIT_ARGS (old_pack)))
{
tree explicit_args = ARGUMENT_PACK_EXPLICIT_ARGS (old_pack);
SET_ARGUMENT_PACK_ARGS (old_pack, new_args);
ARGUMENT_PACK_INCOMPLETE_P (old_pack) = 1;
ARGUMENT_PACK_EXPLICIT_ARGS (old_pack) = explicit_args;
}
else
{
tree bad_old_arg = NULL_TREE, bad_new_arg = NULL_TREE;
tree old_args = ARGUMENT_PACK_ARGS (old_pack);
if (!comp_template_args (old_args, new_args,
&bad_old_arg, &bad_new_arg))
return unify_parameter_pack_inconsistent (explain_p,
bad_old_arg,
bad_new_arg);
}
}
return unify_success (explain_p);
}
static int
unify_array_domain (tree tparms, tree targs,
tree parm_dom, tree arg_dom,
bool explain_p)
{
tree parm_max;
tree arg_max;
bool parm_cst;
bool arg_cst;
parm_max = TYPE_MAX_VALUE (parm_dom);
parm_cst = TREE_CODE (parm_max) == INTEGER_CST;
if (!parm_cst)
{
gcc_assert (TREE_CODE (parm_max) == MINUS_EXPR);
parm_max = TREE_OPERAND (parm_max, 0);
}
arg_max = TYPE_MAX_VALUE (arg_dom);
arg_cst = TREE_CODE (arg_max) == INTEGER_CST;
if (!arg_cst)
{
if (TREE_CODE (arg_max) != MINUS_EXPR)
return unify_vla_arg (explain_p, arg_dom);
arg_max = TREE_OPERAND (arg_max, 0);
}
if (parm_cst && !arg_cst)
parm_max = fold_build2_loc (input_location, PLUS_EXPR,
integer_type_node,
parm_max,
integer_one_node);
else if (arg_cst && !parm_cst)
arg_max = fold_build2_loc (input_location, PLUS_EXPR,
integer_type_node,
arg_max,
integer_one_node);
return unify (tparms, targs, parm_max, arg_max,
UNIFY_ALLOW_INTEGER, explain_p);
}
enum pa_kind_t { pa_type, pa_tmpl, pa_expr };
static pa_kind_t
pa_kind (tree t)
{
if (PACK_EXPANSION_P (t))
t = PACK_EXPANSION_PATTERN (t);
if (TREE_CODE (t) == TEMPLATE_TEMPLATE_PARM
|| TREE_CODE (t) == UNBOUND_CLASS_TEMPLATE
|| DECL_TYPE_TEMPLATE_P (t))
return pa_tmpl;
else if (TYPE_P (t))
return pa_type;
else
return pa_expr;
}
static int
unify (tree tparms, tree targs, tree parm, tree arg, int strict,
bool explain_p)
{
int idx;
tree targ;
tree tparm;
int strict_in = strict;
tsubst_flags_t complain = (explain_p
? tf_warning_or_error
: tf_none);
while (CONVERT_EXPR_P (parm))
parm = TREE_OPERAND (parm, 0);
if (arg == error_mark_node)
return unify_invalid (explain_p);
if (arg == unknown_type_node
|| arg == init_list_type_node)
return unify_success (explain_p);
if (parm == any_targ_node || arg == any_targ_node)
return unify_success (explain_p);
if (arg == parm && !uses_template_parms (parm))
return unify_success (explain_p);
if (BRACE_ENCLOSED_INITIALIZER_P (arg))
{
tree elt, elttype;
unsigned i;
tree orig_parm = parm;
if (TREE_CODE (parm) == TEMPLATE_TYPE_PARM
&& flag_deduce_init_list)
parm = listify (parm);
if (!is_std_init_list (parm)
&& TREE_CODE (parm) != ARRAY_TYPE)
return unify_success (explain_p);
if (TREE_CODE (parm) == ARRAY_TYPE)
elttype = TREE_TYPE (parm);
else
{
elttype = TREE_VEC_ELT (CLASSTYPE_TI_ARGS (parm), 0);
if (PACK_EXPANSION_P (elttype))
return unify_success (explain_p);
}
FOR_EACH_CONSTRUCTOR_VALUE (CONSTRUCTOR_ELTS (arg), i, elt)
{
int elt_strict = strict;
if (elt == error_mark_node)
return unify_invalid (explain_p);
if (!BRACE_ENCLOSED_INITIALIZER_P (elt))
{
tree type = TREE_TYPE (elt);
if (type == error_mark_node)
return unify_invalid (explain_p);
gcc_assert (elt_strict & UNIFY_ALLOW_OUTER_LEVEL);
elt_strict |= maybe_adjust_types_for_deduction
(DEDUCE_CALL, &elttype, &type, elt);
elt = type;
}
RECUR_AND_CHECK_FAILURE (tparms, targs, elttype, elt, elt_strict,
explain_p);
}
if (TREE_CODE (parm) == ARRAY_TYPE
&& deducible_array_bound (TYPE_DOMAIN (parm)))
{
tree max = size_int (CONSTRUCTOR_NELTS (arg));
tree idx = compute_array_index_type (NULL_TREE, max, tf_none);
if (idx == error_mark_node)
return unify_invalid (explain_p);
return unify_array_domain (tparms, targs, TYPE_DOMAIN (parm),
idx, explain_p);
}
if (orig_parm != parm)
{
idx = TEMPLATE_TYPE_IDX (orig_parm);
targ = TREE_VEC_ELT (INNERMOST_TEMPLATE_ARGS (targs), idx);
targ = listify (targ);
TREE_VEC_ELT (INNERMOST_TEMPLATE_ARGS (targs), idx) = targ;
}
return unify_success (explain_p);
}
if (pa_kind (parm) != pa_kind (arg))
return unify_invalid (explain_p);
if (TREE_CODE (arg) == TREE_CODE (parm)
&& TYPE_P (arg)
&& TREE_CODE (arg) != ARRAY_TYPE
&& TREE_CODE (arg) != TEMPLATE_TYPE_PARM
&& !check_cv_quals_for_unify (strict_in, arg, parm))
return unify_cv_qual_mismatch (explain_p, parm, arg);
if (!(strict & UNIFY_ALLOW_OUTER_LEVEL)
&& TYPE_P (parm) && !CP_TYPE_CONST_P (parm))
strict &= ~UNIFY_ALLOW_MORE_CV_QUAL;
strict &= ~UNIFY_ALLOW_OUTER_LEVEL;
strict &= ~UNIFY_ALLOW_DERIVED;
strict &= ~UNIFY_ALLOW_OUTER_MORE_CV_QUAL;
strict &= ~UNIFY_ALLOW_OUTER_LESS_CV_QUAL;
switch (TREE_CODE (parm))
{
case TYPENAME_TYPE:
case SCOPE_REF:
case UNBOUND_CLASS_TEMPLATE:
return unify_success (explain_p);
case TEMPLATE_TYPE_PARM:
case TEMPLATE_TEMPLATE_PARM:
case BOUND_TEMPLATE_TEMPLATE_PARM:
tparm = TREE_VALUE (TREE_VEC_ELT (tparms, 0));
if (error_operand_p (tparm))
return unify_invalid (explain_p);
if (TEMPLATE_TYPE_LEVEL (parm)
!= template_decl_level (tparm))
{
if (TREE_CODE (arg) == TREE_CODE (parm)
&& (is_auto (parm) ? is_auto (arg)
: same_type_p (parm, arg)))
return unify_success (explain_p);
else
return unify_type_mismatch (explain_p, parm, arg);
}
idx = TEMPLATE_TYPE_IDX (parm);
targ = TREE_VEC_ELT (INNERMOST_TEMPLATE_ARGS (targs), idx);
tparm = TREE_VALUE (TREE_VEC_ELT (tparms, idx));
if (error_operand_p (tparm))
return unify_invalid (explain_p);
if ((TREE_CODE (parm) == TEMPLATE_TYPE_PARM
&& TREE_CODE (tparm) != TYPE_DECL)
|| (TREE_CODE (parm) == TEMPLATE_TEMPLATE_PARM
&& TREE_CODE (tparm) != TEMPLATE_DECL))
gcc_unreachable ();
if (TREE_CODE (parm) == BOUND_TEMPLATE_TEMPLATE_PARM)
{
if ((strict_in & UNIFY_ALLOW_DERIVED)
&& CLASS_TYPE_P (arg))
{
tree t = try_class_unification (tparms, targs, parm, arg,
explain_p);
if (!t)
{
enum template_base_result r;
r = get_template_base (tparms, targs, parm, arg,
explain_p, &t);
if (!t)
return unify_no_common_base (explain_p, r, parm, arg);
arg = t;
}
}
else if (TREE_CODE (arg) != BOUND_TEMPLATE_TEMPLATE_PARM
&& !CLASSTYPE_SPECIALIZATION_OF_PRIMARY_TEMPLATE_P (arg))
return unify_template_deduction_failure (explain_p, parm, arg);
if (unify_bound_ttp_args (tparms, targs, parm, arg, explain_p))
return 1;
arg = TYPE_TI_TEMPLATE (arg);
}
if (TREE_CODE (parm) == TEMPLATE_TEMPLATE_PARM
|| TREE_CODE (parm) == BOUND_TEMPLATE_TEMPLATE_PARM)
{
if (targ != NULL_TREE && template_args_equal (targ, arg))
return unify_success (explain_p);
else if (targ)
return unify_inconsistency (explain_p, parm, targ, arg);
}
else
{
if (!check_cv_quals_for_unify (strict_in | UNIFY_ALLOW_LESS_CV_QUAL,
arg, parm))
return unify_cv_qual_mismatch (explain_p, parm, arg);
arg = cp_build_qualified_type_real
(arg, cp_type_quals (arg) & ~cp_type_quals (parm), tf_none);
if (arg == error_mark_node)
return unify_invalid (explain_p);
if (targ != NULL_TREE && same_type_p (targ, arg))
return unify_success (explain_p);
else if (targ)
return unify_inconsistency (explain_p, parm, targ, arg);
if (!is_auto (parm) && variably_modified_type_p (arg, NULL_TREE))
return unify_vla_arg (explain_p, arg);
arg = canonicalize_type_argument (arg, tf_none);
}
if ((template_parameter_pack_p (arg) || PACK_EXPANSION_P (arg))
&& !template_parameter_pack_p (parm))
return unify_parameter_pack_mismatch (explain_p, parm, arg);
if (TREE_CODE (arg) == METHOD_TYPE)
return unify_method_type_error (explain_p, arg);
TREE_VEC_ELT (INNERMOST_TEMPLATE_ARGS (targs), idx) = arg;
return unify_success (explain_p);
case TEMPLATE_PARM_INDEX:
tparm = TREE_VALUE (TREE_VEC_ELT (tparms, 0));
if (error_operand_p (tparm))
return unify_invalid (explain_p);
if (TEMPLATE_PARM_LEVEL (parm)
!= template_decl_level (tparm))
{
int result = !(TREE_CODE (arg) == TREE_CODE (parm)
&& cp_tree_equal (parm, arg));
if (result)
unify_expression_unequal (explain_p, parm, arg);
return result;
}
idx = TEMPLATE_PARM_IDX (parm);
targ = TREE_VEC_ELT (INNERMOST_TEMPLATE_ARGS (targs), idx);
if (targ)
{
if ((strict & UNIFY_ALLOW_INTEGER)
&& TREE_TYPE (targ) && TREE_TYPE (arg)
&& CP_INTEGRAL_TYPE_P (TREE_TYPE (targ)))
arg = fold_convert (TREE_TYPE (targ), arg);
int x = !cp_tree_equal (targ, arg);
if (x)
unify_inconsistency (explain_p, parm, targ, arg);
return x;
}
tparm = TREE_TYPE (parm);
if (TEMPLATE_PARM_LEVEL (parm) > TMPL_ARGS_DEPTH (targs))
;
else
{
++processing_template_decl;
tparm = tsubst (tparm, targs, tf_none, NULL_TREE);
--processing_template_decl;
if (tree a = type_uses_auto (tparm))
{
tparm = do_auto_deduction (tparm, arg, a, complain, adc_unify);
if (tparm == error_mark_node)
return 1;
}
}
if (!TREE_TYPE (arg))
;
else if (same_type_p (non_reference (TREE_TYPE (arg)),
non_reference (tparm)))
;
else if ((strict & UNIFY_ALLOW_INTEGER)
&& CP_INTEGRAL_TYPE_P (tparm))
arg = fold (build_nop (tparm, arg));
else if (uses_template_parms (tparm))
{
if (cxx_dialect >= cxx17
&& !(strict & UNIFY_ALLOW_INTEGER))
{
tree atype = TREE_TYPE (arg);
RECUR_AND_CHECK_FAILURE (tparms, targs,
tparm, atype,
UNIFY_ALLOW_NONE, explain_p);
}
else
return unify_success (explain_p);
}
else
return unify_type_mismatch (explain_p, tparm, TREE_TYPE (arg));
if ((template_parameter_pack_p (arg) || PACK_EXPANSION_P (arg))
&& !TEMPLATE_PARM_PARAMETER_PACK (parm))
return unify_parameter_pack_mismatch (explain_p, parm, arg);
{
bool removed_attr = false;
arg = strip_typedefs_expr (arg, &removed_attr);
}
TREE_VEC_ELT (INNERMOST_TEMPLATE_ARGS (targs), idx) = arg;
return unify_success (explain_p);
case PTRMEM_CST:
{
if (TREE_CODE (arg) != PTRMEM_CST)
return unify_ptrmem_cst_mismatch (explain_p, parm, arg);
return unify (tparms, targs, PTRMEM_CST_MEMBER (parm),
PTRMEM_CST_MEMBER (arg), strict, explain_p);
}
case POINTER_TYPE:
{
if (!TYPE_PTR_P (arg))
return unify_type_mismatch (explain_p, parm, arg);
if (TREE_CODE (TREE_TYPE (arg)) == RECORD_TYPE)
strict |= (strict_in & UNIFY_ALLOW_DERIVED);
return unify (tparms, targs, TREE_TYPE (parm),
TREE_TYPE (arg), strict, explain_p);
}
case REFERENCE_TYPE:
if (TREE_CODE (arg) != REFERENCE_TYPE)
return unify_type_mismatch (explain_p, parm, arg);
return unify (tparms, targs, TREE_TYPE (parm), TREE_TYPE (arg),
strict & UNIFY_ALLOW_MORE_CV_QUAL, explain_p);
case ARRAY_TYPE:
if (TREE_CODE (arg) != ARRAY_TYPE)
return unify_type_mismatch (explain_p, parm, arg);
if ((TYPE_DOMAIN (parm) == NULL_TREE)
!= (TYPE_DOMAIN (arg) == NULL_TREE))
return unify_type_mismatch (explain_p, parm, arg);
RECUR_AND_CHECK_FAILURE (tparms, targs, TREE_TYPE (parm), TREE_TYPE (arg),
strict & UNIFY_ALLOW_MORE_CV_QUAL, explain_p);
if (TYPE_DOMAIN (parm) != NULL_TREE)
return unify_array_domain (tparms, targs, TYPE_DOMAIN (parm),
TYPE_DOMAIN (arg), explain_p);
return unify_success (explain_p);
case REAL_TYPE:
case COMPLEX_TYPE:
case VECTOR_TYPE:
case INTEGER_TYPE:
case BOOLEAN_TYPE:
case ENUMERAL_TYPE:
case VOID_TYPE:
case NULLPTR_TYPE:
if (TREE_CODE (arg) != TREE_CODE (parm))
return unify_type_mismatch (explain_p, parm, arg);
if (!same_type_ignoring_top_level_qualifiers_p (arg, parm))
return unify_type_mismatch (explain_p, parm, arg);
return unify_success (explain_p);
case INTEGER_CST:
while (CONVERT_EXPR_P (arg))
arg = TREE_OPERAND (arg, 0);
if (TREE_CODE (arg) != INTEGER_CST)
return unify_template_argument_mismatch (explain_p, parm, arg);
return (tree_int_cst_equal (parm, arg)
? unify_success (explain_p)
: unify_template_argument_mismatch (explain_p, parm, arg));
case TREE_VEC:
{
int i, len, argslen;
int parm_variadic_p = 0;
if (TREE_CODE (arg) != TREE_VEC)
return unify_template_argument_mismatch (explain_p, parm, arg);
len = TREE_VEC_LENGTH (parm);
argslen = TREE_VEC_LENGTH (arg);
for (i = 0; i < len; ++i)
{
if (PACK_EXPANSION_P (TREE_VEC_ELT (parm, i)))
{
if (i == len - 1)
parm_variadic_p = 1;
else
return unify_success (explain_p);
}
}
if (parm_variadic_p
? argslen < len - parm_variadic_p
: argslen != len)
return unify_arity (explain_p, TREE_VEC_LENGTH (arg), len);
for (i = 0; i < len - parm_variadic_p; ++i)
{
RECUR_AND_CHECK_FAILURE (tparms, targs,
TREE_VEC_ELT (parm, i),
TREE_VEC_ELT (arg, i),
UNIFY_ALLOW_NONE, explain_p);
}
if (parm_variadic_p)
return unify_pack_expansion (tparms, targs, parm, arg,
DEDUCE_EXACT,
true, explain_p);
return unify_success (explain_p);
}
case RECORD_TYPE:
case UNION_TYPE:
if (TREE_CODE (arg) != TREE_CODE (parm))
return unify_type_mismatch (explain_p, parm, arg);
if (TYPE_PTRMEMFUNC_P (parm))
{
if (!TYPE_PTRMEMFUNC_P (arg))
return unify_type_mismatch (explain_p, parm, arg);
return unify (tparms, targs,
TYPE_PTRMEMFUNC_FN_TYPE (parm),
TYPE_PTRMEMFUNC_FN_TYPE (arg),
strict, explain_p);
}
else if (TYPE_PTRMEMFUNC_P (arg))
return unify_type_mismatch (explain_p, parm, arg);
if (CLASSTYPE_TEMPLATE_INFO (parm))
{
tree t = NULL_TREE;
if (strict_in & UNIFY_ALLOW_DERIVED)
{
t = try_class_unification (tparms, targs,
parm, arg, explain_p);
if (!t)
{
enum template_base_result r;
r = get_template_base (tparms, targs, parm, arg,
explain_p, &t);
if (!t)
{
bool same_template
= (CLASSTYPE_TEMPLATE_INFO (arg)
&& (CLASSTYPE_TI_TEMPLATE (parm)
== CLASSTYPE_TI_TEMPLATE (arg)));
return unify_no_common_base (explain_p && !same_template,
r, parm, arg);
}
}
}
else if (CLASSTYPE_TEMPLATE_INFO (arg)
&& (CLASSTYPE_TI_TEMPLATE (parm)
== CLASSTYPE_TI_TEMPLATE (arg)))
t = arg;
else
return unify_type_mismatch (explain_p, parm, arg);
return unify (tparms, targs, CLASSTYPE_TI_ARGS (parm),
CLASSTYPE_TI_ARGS (t), UNIFY_ALLOW_NONE, explain_p);
}
else if (!same_type_ignoring_top_level_qualifiers_p (parm, arg))
return unify_type_mismatch (explain_p, parm, arg);
return unify_success (explain_p);
case METHOD_TYPE:
case FUNCTION_TYPE:
{
unsigned int nargs;
tree *args;
tree a;
unsigned int i;
if (TREE_CODE (arg) != TREE_CODE (parm))
return unify_type_mismatch (explain_p, parm, arg);
if (TREE_CODE (parm) == METHOD_TYPE
&& (!check_cv_quals_for_unify
(UNIFY_ALLOW_NONE,
class_of_this_parm (arg),
class_of_this_parm (parm))))
return unify_cv_qual_mismatch (explain_p, parm, arg);
if (TREE_CODE (arg) == FUNCTION_TYPE
&& type_memfn_quals (parm) != type_memfn_quals (arg))
return unify_cv_qual_mismatch (explain_p, parm, arg);
if (type_memfn_rqual (parm) != type_memfn_rqual (arg))
return unify_type_mismatch (explain_p, parm, arg);
RECUR_AND_CHECK_FAILURE (tparms, targs, TREE_TYPE (parm),
TREE_TYPE (arg), UNIFY_ALLOW_NONE, explain_p);
nargs = list_length (TYPE_ARG_TYPES (arg));
args = XALLOCAVEC (tree, nargs);
for (a = TYPE_ARG_TYPES (arg), i = 0;
a != NULL_TREE && a != void_list_node;
a = TREE_CHAIN (a), ++i)
args[i] = TREE_VALUE (a);
nargs = i;
if (type_unification_real (tparms, targs, TYPE_ARG_TYPES (parm),
args, nargs, 1, DEDUCE_EXACT,
LOOKUP_NORMAL, NULL, explain_p))
return 1;
if (flag_noexcept_type)
{
tree pspec = TYPE_RAISES_EXCEPTIONS (parm);
tree aspec = canonical_eh_spec (TYPE_RAISES_EXCEPTIONS (arg));
if (pspec == NULL_TREE) pspec = noexcept_false_spec;
if (aspec == NULL_TREE) aspec = noexcept_false_spec;
if (TREE_PURPOSE (pspec) && TREE_PURPOSE (aspec)
&& uses_template_parms (TREE_PURPOSE (pspec)))
RECUR_AND_CHECK_FAILURE (tparms, targs, TREE_PURPOSE (pspec),
TREE_PURPOSE (aspec),
UNIFY_ALLOW_NONE, explain_p);
else if (nothrow_spec_p (pspec) && !nothrow_spec_p (aspec))
return unify_type_mismatch (explain_p, parm, arg);
}
return 0;
}
case OFFSET_TYPE:
if (TYPE_PTRMEMFUNC_P (arg))
{
if (!check_cv_quals_for_unify (UNIFY_ALLOW_NONE, arg, parm))
return unify_cv_qual_mismatch (explain_p, parm, arg);
RECUR_AND_CHECK_FAILURE (tparms, targs, TYPE_OFFSET_BASETYPE (parm),
TYPE_PTRMEMFUNC_OBJECT_TYPE (arg),
UNIFY_ALLOW_NONE, explain_p);
tree fntype = static_fn_type (arg);
return unify (tparms, targs, TREE_TYPE (parm), fntype, strict, explain_p);
}
if (TREE_CODE (arg) != OFFSET_TYPE)
return unify_type_mismatch (explain_p, parm, arg);
RECUR_AND_CHECK_FAILURE (tparms, targs, TYPE_OFFSET_BASETYPE (parm),
TYPE_OFFSET_BASETYPE (arg),
UNIFY_ALLOW_NONE, explain_p);
return unify (tparms, targs, TREE_TYPE (parm), TREE_TYPE (arg),
strict, explain_p);
case CONST_DECL:
if (DECL_TEMPLATE_PARM_P (parm))
return unify (tparms, targs, DECL_INITIAL (parm), arg, strict, explain_p);
if (arg != scalar_constant_value (parm))
return unify_template_argument_mismatch (explain_p, parm, arg);
return unify_success (explain_p);
case FIELD_DECL:
case TEMPLATE_DECL:
return unify_template_argument_mismatch (explain_p, parm, arg);
case VAR_DECL:
if (CONSTANT_CLASS_P (arg))
parm = fold_non_dependent_expr (parm);
else if (REFERENCE_REF_P (arg))
{
tree sub = TREE_OPERAND (arg, 0);
STRIP_NOPS (sub);
if (TREE_CODE (sub) == ADDR_EXPR)
arg = TREE_OPERAND (sub, 0);
}
goto expr;
case TYPE_ARGUMENT_PACK:
case NONTYPE_ARGUMENT_PACK:
return unify (tparms, targs, ARGUMENT_PACK_ARGS (parm),
ARGUMENT_PACK_ARGS (arg), strict, explain_p);
case TYPEOF_TYPE:
case DECLTYPE_TYPE:
case UNDERLYING_TYPE:
return unify_success (explain_p);
case ERROR_MARK:
return unify_invalid (explain_p);
case INDIRECT_REF:
if (REFERENCE_REF_P (parm))
{
bool pexp = PACK_EXPANSION_P (arg);
if (pexp)
arg = PACK_EXPANSION_PATTERN (arg);
if (REFERENCE_REF_P (arg))
arg = TREE_OPERAND (arg, 0);
if (pexp)
arg = make_pack_expansion (arg, complain);
return unify (tparms, targs, TREE_OPERAND (parm, 0), arg,
strict, explain_p);
}
default:
if (is_overloaded_fn (parm) || type_unknown_p (parm))
return unify_success (explain_p);
gcc_assert (EXPR_P (parm) || TREE_CODE (parm) == TRAIT_EXPR);
expr:
if (!uses_template_parms (parm)
&& !template_args_equal (parm, arg))
return unify_expression_unequal (explain_p, parm, arg);
else
return unify_success (explain_p);
}
}
#undef RECUR_AND_CHECK_FAILURE

static void
mark_definable (tree decl)
{
tree clone;
DECL_NOT_REALLY_EXTERN (decl) = 1;
FOR_EACH_CLONE (clone, decl)
DECL_NOT_REALLY_EXTERN (clone) = 1;
}
void
mark_decl_instantiated (tree result, int extern_p)
{
SET_DECL_EXPLICIT_INSTANTIATION (result);
if (TREE_ASM_WRITTEN (result))
return;
if (decl_anon_ns_mem_p (result))
{
gcc_assert (!TREE_PUBLIC (result));
return;
}
if (TREE_CODE (result) != FUNCTION_DECL)
TREE_PUBLIC (result) = 1;
DECL_COMDAT (result) = 0;
if (extern_p)
DECL_NOT_REALLY_EXTERN (result) = 0;
else
{
mark_definable (result);
mark_needed (result);
if (DECL_ARTIFICIAL (result) && flag_weak)
comdat_linkage (result);
else if (TREE_PUBLIC (result))
maybe_make_one_only (result);
}
DECL_INTERFACE_KNOWN (result) = 1;
}
static bool
check_undeduced_parms (tree targs, tree args, tree end)
{
bool found = false;
int i;
for (i = TREE_VEC_LENGTH (targs) - 1; i >= 0; --i)
if (TREE_VEC_ELT (targs, i) == NULL_TREE)
{
found = true;
TREE_VEC_ELT (targs, i) = error_mark_node;
}
if (found)
{
tree substed = tsubst_arg_types (args, targs, end, tf_none, NULL_TREE);
if (substed == error_mark_node)
return true;
}
return false;
}
int
more_specialized_fn (tree pat1, tree pat2, int len)
{
tree decl1 = DECL_TEMPLATE_RESULT (pat1);
tree decl2 = DECL_TEMPLATE_RESULT (pat2);
tree targs1 = make_tree_vec (DECL_NTPARMS (pat1));
tree targs2 = make_tree_vec (DECL_NTPARMS (pat2));
tree tparms1 = DECL_INNERMOST_TEMPLATE_PARMS (pat1);
tree tparms2 = DECL_INNERMOST_TEMPLATE_PARMS (pat2);
tree args1 = TYPE_ARG_TYPES (TREE_TYPE (decl1));
tree args2 = TYPE_ARG_TYPES (TREE_TYPE (decl2));
tree origs1, origs2;
bool lose1 = false;
bool lose2 = false;
if (DECL_NONSTATIC_MEMBER_FUNCTION_P (decl1))
{
len--; 
args1 = TREE_CHAIN (args1);
if (!DECL_STATIC_FUNCTION_P (decl2))
args2 = TREE_CHAIN (args2);
}
else if (DECL_NONSTATIC_MEMBER_FUNCTION_P (decl2))
{
args2 = TREE_CHAIN (args2);
if (!DECL_STATIC_FUNCTION_P (decl1))
{
len--;
args1 = TREE_CHAIN (args1);
}
}
if (DECL_CONV_FN_P (decl1) != DECL_CONV_FN_P (decl2))
return 0;
if (DECL_CONV_FN_P (decl1))
{
args1 = tree_cons (NULL_TREE, TREE_TYPE (TREE_TYPE (decl1)), args1);
args2 = tree_cons (NULL_TREE, TREE_TYPE (TREE_TYPE (decl2)), args2);
len++;
}
processing_template_decl++;
origs1 = args1;
origs2 = args2;
while (len--
&& args1 != NULL_TREE && args2 != NULL_TREE)
{
tree arg1 = TREE_VALUE (args1);
tree arg2 = TREE_VALUE (args2);
int deduce1, deduce2;
int quals1 = -1;
int quals2 = -1;
int ref1 = 0;
int ref2 = 0;
if (TREE_CODE (arg1) == TYPE_PACK_EXPANSION
&& TREE_CODE (arg2) == TYPE_PACK_EXPANSION)
{
arg1 = PACK_EXPANSION_PATTERN (arg1);
arg2 = PACK_EXPANSION_PATTERN (arg2);
len = 0;
}
if (!uses_deducible_template_parms (arg1)
&& !uses_deducible_template_parms (arg2))
goto next;
if (TREE_CODE (arg1) == REFERENCE_TYPE)
{
ref1 = TYPE_REF_IS_RVALUE (arg1) + 1;
arg1 = TREE_TYPE (arg1);
quals1 = cp_type_quals (arg1);
}
if (TREE_CODE (arg2) == REFERENCE_TYPE)
{
ref2 = TYPE_REF_IS_RVALUE (arg2) + 1;
arg2 = TREE_TYPE (arg2);
quals2 = cp_type_quals (arg2);
}
arg1 = TYPE_MAIN_VARIANT (arg1);
arg2 = TYPE_MAIN_VARIANT (arg2);
if (TREE_CODE (arg1) == TYPE_PACK_EXPANSION)
{
int i, len2 = remaining_arguments (args2);
tree parmvec = make_tree_vec (1);
tree argvec = make_tree_vec (len2);
tree ta = args2;
TREE_VEC_ELT (parmvec, 0) = arg1;
for (i = 0; i < len2; i++, ta = TREE_CHAIN (ta))
TREE_VEC_ELT (argvec, i) = TREE_VALUE (ta);
deduce1 = (unify_pack_expansion (tparms1, targs1, parmvec,
argvec, DEDUCE_EXACT,
true, false)
== 0);
deduce2 = 0;
}
else if (TREE_CODE (arg2) == TYPE_PACK_EXPANSION)
{
int i, len1 = remaining_arguments (args1);
tree parmvec = make_tree_vec (1);
tree argvec = make_tree_vec (len1);
tree ta = args1;
TREE_VEC_ELT (parmvec, 0) = arg2;
for (i = 0; i < len1; i++, ta = TREE_CHAIN (ta))
TREE_VEC_ELT (argvec, i) = TREE_VALUE (ta);
deduce2 = (unify_pack_expansion (tparms2, targs2, parmvec,
argvec, DEDUCE_EXACT,
true, false)
== 0);
deduce1 = 0;
}
else
{
deduce1 = (unify (tparms1, targs1, arg1, arg2,
UNIFY_ALLOW_NONE, false)
== 0);
deduce2 = (unify (tparms2, targs2, arg2, arg1,
UNIFY_ALLOW_NONE, false)
== 0);
}
if (!deduce1)
lose2 = true;
if (!deduce2)
lose1 = true;
if (deduce1 && deduce2)
{
if (ref1 && ref2 && ref1 != ref2)
{
if (ref1 > ref2)
lose1 = true;
else
lose2 = true;
}
else if (quals1 != quals2 && quals1 >= 0 && quals2 >= 0)
{
if ((quals1 & quals2) == quals2)
lose2 = true;
if ((quals1 & quals2) == quals1)
lose1 = true;
}
}
if (lose1 && lose2)
break;
next:
if (TREE_CODE (arg1) == TYPE_PACK_EXPANSION
|| TREE_CODE (arg2) == TYPE_PACK_EXPANSION)
len = 0;
args1 = TREE_CHAIN (args1);
args2 = TREE_CHAIN (args2);
}
if (!lose2 && check_undeduced_parms (targs1, origs1, args1))
lose2 = true;
if (!lose1 && check_undeduced_parms (targs2, origs2, args2))
lose1 = true;
processing_template_decl--;
if (!lose1 && !lose2)
{
tree c1 = get_constraints (DECL_TEMPLATE_RESULT (pat1));
tree c2 = get_constraints (DECL_TEMPLATE_RESULT (pat2));
lose1 = !subsumes_constraints (c1, c2);
lose2 = !subsumes_constraints (c2, c1);
}
if (lose1 == lose2
&& args1 && TREE_VALUE (args1)
&& args2 && TREE_VALUE (args2))
{
lose1 = TREE_CODE (TREE_VALUE (args1)) == TYPE_PACK_EXPANSION;
lose2 = TREE_CODE (TREE_VALUE (args2)) == TYPE_PACK_EXPANSION;
}
if (lose1 == lose2)
return 0;
else if (!lose1)
return 1;
else
return -1;
}
static int
more_specialized_partial_spec (tree tmpl, tree pat1, tree pat2)
{
tree targs;
int winner = 0;
bool any_deductions = false;
tree tmpl1 = TREE_VALUE (pat1);
tree tmpl2 = TREE_VALUE (pat2);
tree specargs1 = TI_ARGS (get_template_info (DECL_TEMPLATE_RESULT (tmpl1)));
tree specargs2 = TI_ARGS (get_template_info (DECL_TEMPLATE_RESULT (tmpl2)));
++processing_template_decl;
targs = get_partial_spec_bindings (tmpl, tmpl1, specargs2);
if (targs)
{
--winner;
any_deductions = true;
}
targs = get_partial_spec_bindings (tmpl, tmpl2, specargs1);
if (targs)
{
++winner;
any_deductions = true;
}
--processing_template_decl;
if (!winner && any_deductions)
return more_constrained (tmpl1, tmpl2);
if (winner == 0
&& any_deductions
&& (template_args_variadic_p (TREE_PURPOSE (pat1))
|| template_args_variadic_p (TREE_PURPOSE (pat2))))
{
tree args1 = INNERMOST_TEMPLATE_ARGS (TREE_PURPOSE (pat1));
tree args2 = INNERMOST_TEMPLATE_ARGS (TREE_PURPOSE (pat2));
int len1 = TREE_VEC_LENGTH (args1);
int len2 = TREE_VEC_LENGTH (args2);
if (template_args_variadic_p (TREE_PURPOSE (pat1)))
--len1;
if (template_args_variadic_p (TREE_PURPOSE (pat2)))
--len2;
if (len1 > len2)
return 1;
else if (len1 < len2)
return -1;
}
return winner;
}
static tree
get_bindings (tree fn, tree decl, tree explicit_args, bool check_rettype)
{
int ntparms = DECL_NTPARMS (fn);
tree targs = make_tree_vec (ntparms);
tree decl_type = TREE_TYPE (decl);
tree decl_arg_types;
tree *args;
unsigned int nargs, ix;
tree arg;
gcc_assert (decl != DECL_TEMPLATE_RESULT (fn));
decl_arg_types = skip_artificial_parms_for (decl, 
TYPE_ARG_TYPES (decl_type));
nargs = list_length (decl_arg_types);
args = XALLOCAVEC (tree, nargs);
for (arg = decl_arg_types, ix = 0;
arg != NULL_TREE && arg != void_list_node;
arg = TREE_CHAIN (arg), ++ix)
args[ix] = TREE_VALUE (arg);
if (fn_type_unification (fn, explicit_args, targs,
args, ix,
(check_rettype || DECL_CONV_FN_P (fn)
? TREE_TYPE (decl_type) : NULL_TREE),
DEDUCE_EXACT, LOOKUP_NORMAL, false,
false)
== error_mark_node)
return NULL_TREE;
return targs;
}
static tree
get_partial_spec_bindings (tree tmpl, tree spec_tmpl, tree args)
{
tree tparms = DECL_INNERMOST_TEMPLATE_PARMS (spec_tmpl);
tree spec_args
= TI_ARGS (get_template_info (DECL_TEMPLATE_RESULT (spec_tmpl)));
int i, ntparms = TREE_VEC_LENGTH (tparms);
tree deduced_args;
tree innermost_deduced_args;
innermost_deduced_args = make_tree_vec (ntparms);
if (TMPL_ARGS_HAVE_MULTIPLE_LEVELS (args))
{
deduced_args = copy_node (args);
SET_TMPL_ARGS_LEVEL (deduced_args,
TMPL_ARGS_DEPTH (deduced_args),
innermost_deduced_args);
}
else
deduced_args = innermost_deduced_args;
bool tried_array_deduction = (cxx_dialect < cxx17);
again:
if (unify (tparms, deduced_args,
INNERMOST_TEMPLATE_ARGS (spec_args),
INNERMOST_TEMPLATE_ARGS (args),
UNIFY_ALLOW_NONE, false))
return NULL_TREE;
for (i =  0; i < ntparms; ++i)
if (! TREE_VEC_ELT (innermost_deduced_args, i))
{
if (!tried_array_deduction)
{
try_array_deduction (tparms, innermost_deduced_args,
INNERMOST_TEMPLATE_ARGS (spec_args));
tried_array_deduction = true;
if (TREE_VEC_ELT (innermost_deduced_args, i))
goto again;
}
return NULL_TREE;
}
if (!push_tinst_level (spec_tmpl, deduced_args))
{
excessive_deduction_depth = true;
return NULL_TREE;
}
spec_args = tsubst (spec_args, deduced_args, tf_none, NULL_TREE);
if (spec_args != error_mark_node)
spec_args = coerce_template_parms (DECL_INNERMOST_TEMPLATE_PARMS (tmpl),
INNERMOST_TEMPLATE_ARGS (spec_args),
tmpl, tf_none, false, false);
pop_tinst_level ();
if (spec_args == error_mark_node
|| !comp_template_args_porder (INNERMOST_TEMPLATE_ARGS (spec_args),
INNERMOST_TEMPLATE_ARGS (args)))
return NULL_TREE;
if (!template_template_parm_bindings_ok_p (tparms, deduced_args))
return NULL_TREE;
return deduced_args;
}
static int
more_specialized_inst (tree t1, tree t2)
{
int fate = 0;
int count = 0;
if (get_bindings (t1, DECL_TEMPLATE_RESULT (t2), NULL_TREE, true))
{
--fate;
++count;
}
if (get_bindings (t2, DECL_TEMPLATE_RESULT (t1), NULL_TREE, true))
{
++fate;
++count;
}
if (count == 2 && fate == 0)
fate = more_constrained (t1, t2);
return fate;
}
tree
most_specialized_instantiation (tree templates)
{
tree fn, champ;
++processing_template_decl;
champ = templates;
for (fn = TREE_CHAIN (templates); fn; fn = TREE_CHAIN (fn))
{
gcc_assert (TREE_VALUE (champ) != TREE_VALUE (fn));
int fate = more_specialized_inst (TREE_VALUE (champ), TREE_VALUE (fn));
if (fate == -1)
champ = fn;
else if (!fate)
{
fn = TREE_CHAIN (fn);
champ = fn;
if (!fn)
break;
}
}
if (champ)
for (fn = templates; fn != champ; fn = TREE_CHAIN (fn)) {
if (more_specialized_inst (TREE_VALUE (champ), TREE_VALUE (fn)) != 1)
{
champ = NULL_TREE;
break;
}
}
processing_template_decl--;
if (!champ)
return error_mark_node;
return champ;
}
tree
most_general_template (tree decl)
{
if (TREE_CODE (decl) != TEMPLATE_DECL)
{
if (tree tinfo = get_template_info (decl))
decl = TI_TEMPLATE (tinfo);
if (TREE_CODE (decl) != TEMPLATE_DECL)
return NULL_TREE;
}
while (DECL_LANG_SPECIFIC (decl) && DECL_TEMPLATE_INFO (decl))
{
if (TREE_CODE (DECL_TI_TEMPLATE (decl)) != TEMPLATE_DECL)
break;
if (CLASS_TYPE_P (TREE_TYPE (decl))
&& !TYPE_DECL_ALIAS_P (TYPE_NAME (TREE_TYPE (decl)))
&& CLASSTYPE_TEMPLATE_SPECIALIZATION (TREE_TYPE (decl)))
break;
if (!DECL_NAMESPACE_SCOPE_P (decl)
&& DECL_CONTEXT (decl)
&& CLASSTYPE_TEMPLATE_SPECIALIZATION (DECL_CONTEXT (decl)))
break;
decl = DECL_TI_TEMPLATE (decl);
}
return decl;
}
static tree
most_specialized_partial_spec (tree target, tsubst_flags_t complain)
{
tree list = NULL_TREE;
tree t;
tree champ;
int fate;
bool ambiguous_p;
tree outer_args = NULL_TREE;
tree tmpl, args;
if (TYPE_P (target))
{
tree tinfo = CLASSTYPE_TEMPLATE_INFO (target);
tmpl = TI_TEMPLATE (tinfo);
args = TI_ARGS (tinfo);
}
else if (TREE_CODE (target) == TEMPLATE_ID_EXPR)
{
tmpl = TREE_OPERAND (target, 0);
args = TREE_OPERAND (target, 1);
}
else if (VAR_P (target))
{
tree tinfo = DECL_TEMPLATE_INFO (target);
tmpl = TI_TEMPLATE (tinfo);
args = TI_ARGS (tinfo);
}
else
gcc_unreachable ();
tree main_tmpl = most_general_template (tmpl);
if (TMPL_ARGS_HAVE_MULTIPLE_LEVELS (args))
{
outer_args = strip_innermost_template_args (args, 1);
args = INNERMOST_TEMPLATE_ARGS (args);
}
for (t = DECL_TEMPLATE_SPECIALIZATIONS (main_tmpl); t; t = TREE_CHAIN (t))
{
tree spec_args;
tree spec_tmpl = TREE_VALUE (t);
if (outer_args)
{
++processing_template_decl;
spec_tmpl = tsubst (spec_tmpl, outer_args, tf_none, NULL_TREE);
--processing_template_decl;
}
if (spec_tmpl == error_mark_node)
return error_mark_node;
spec_args = get_partial_spec_bindings (tmpl, spec_tmpl, args);
if (spec_args)
{
if (outer_args)
spec_args = add_to_template_args (outer_args, spec_args);
if (!flag_concepts
|| constraints_satisfied_p (spec_tmpl, spec_args))
{
list = tree_cons (spec_args, TREE_VALUE (t), list);
TREE_TYPE (list) = TREE_TYPE (t);
}
}
}
if (! list)
return NULL_TREE;
ambiguous_p = false;
t = list;
champ = t;
t = TREE_CHAIN (t);
for (; t; t = TREE_CHAIN (t))
{
fate = more_specialized_partial_spec (tmpl, champ, t);
if (fate == 1)
;
else
{
if (fate == 0)
{
t = TREE_CHAIN (t);
if (! t)
{
ambiguous_p = true;
break;
}
}
champ = t;
}
}
if (!ambiguous_p)
for (t = list; t && t != champ; t = TREE_CHAIN (t))
{
fate = more_specialized_partial_spec (tmpl, champ, t);
if (fate != 1)
{
ambiguous_p = true;
break;
}
}
if (ambiguous_p)
{
const char *str;
char *spaces = NULL;
if (!(complain & tf_error))
return error_mark_node;
if (TYPE_P (target))
error ("ambiguous template instantiation for %q#T", target);
else
error ("ambiguous template instantiation for %q#D", target);
str = ngettext ("candidate is:", "candidates are:", list_length (list));
for (t = list; t; t = TREE_CHAIN (t))
{
tree subst = build_tree_list (TREE_VALUE (t), TREE_PURPOSE (t));
inform (DECL_SOURCE_LOCATION (TREE_VALUE (t)),
"%s %#qS", spaces ? spaces : str, subst);
spaces = spaces ? spaces : get_spaces (str);
}
free (spaces);
return error_mark_node;
}
return champ;
}
void
do_decl_instantiation (tree decl, tree storage)
{
tree result = NULL_TREE;
int extern_p = 0;
if (!decl || decl == error_mark_node)
return;
else if (! DECL_LANG_SPECIFIC (decl))
{
error ("explicit instantiation of non-template %q#D", decl);
return;
}
bool var_templ = (DECL_TEMPLATE_INFO (decl)
&& variable_template_p (DECL_TI_TEMPLATE (decl)));
if (VAR_P (decl) && !var_templ)
{
if (!DECL_CLASS_SCOPE_P (decl))
{
error ("%qD is not a static data member of a class template", decl);
return;
}
result = lookup_field (DECL_CONTEXT (decl), DECL_NAME (decl), 0, false);
if (!result || !VAR_P (result))
{
error ("no matching template for %qD found", decl);
return;
}
if (!same_type_p (TREE_TYPE (result), TREE_TYPE (decl)))
{
error ("type %qT for explicit instantiation %qD does not match "
"declared type %qT", TREE_TYPE (result), decl,
TREE_TYPE (decl));
return;
}
}
else if (TREE_CODE (decl) != FUNCTION_DECL && !var_templ)
{
error ("explicit instantiation of %q#D", decl);
return;
}
else
result = decl;
if (DECL_TEMPLATE_SPECIALIZATION (result))
{
return;
}
else if (DECL_EXPLICIT_INSTANTIATION (result))
{
if (DECL_NOT_REALLY_EXTERN (result) && !extern_p)
permerror (input_location, "duplicate explicit instantiation of %q#D", result);
if (extern_p)
return;
}
else if (!DECL_IMPLICIT_INSTANTIATION (result))
{
error ("no matching template for %qD found", result);
return;
}
else if (!DECL_TEMPLATE_INFO (result))
{
permerror (input_location, "explicit instantiation of non-template %q#D", result);
return;
}
if (storage == NULL_TREE)
;
else if (storage == ridpointers[(int) RID_EXTERN])
{
if (!in_system_header_at (input_location) && (cxx_dialect == cxx98))
pedwarn (input_location, OPT_Wpedantic, 
"ISO C++ 1998 forbids the use of %<extern%> on explicit "
"instantiations");
extern_p = 1;
}
else
error ("storage class %qD applied to template instantiation", storage);
check_explicit_instantiation_namespace (result);
mark_decl_instantiated (result, extern_p);
if (! extern_p)
instantiate_decl (result, true,
false);
}
static void
mark_class_instantiated (tree t, int extern_p)
{
SET_CLASSTYPE_EXPLICIT_INSTANTIATION (t);
SET_CLASSTYPE_INTERFACE_KNOWN (t);
CLASSTYPE_INTERFACE_ONLY (t) = extern_p;
TYPE_DECL_SUPPRESS_DEBUG (TYPE_NAME (t)) = extern_p;
if (! extern_p)
{
CLASSTYPE_DEBUG_REQUESTED (t) = 1;
rest_of_type_compilation (t, 1);
}
}
static void
bt_instantiate_type_proc (binding_entry entry, void *data)
{
tree storage = *(tree *) data;
if (MAYBE_CLASS_TYPE_P (entry->type)
&& CLASSTYPE_TEMPLATE_INFO (entry->type)
&& !uses_template_parms (CLASSTYPE_TI_ARGS (entry->type)))
do_type_instantiation (TYPE_MAIN_DECL (entry->type), storage, 0);
}
void
do_type_instantiation (tree t, tree storage, tsubst_flags_t complain)
{
int extern_p = 0;
int nomem_p = 0;
int static_p = 0;
int previous_instantiation_extern_p = 0;
if (TREE_CODE (t) == TYPE_DECL)
t = TREE_TYPE (t);
if (! CLASS_TYPE_P (t) || ! CLASSTYPE_TEMPLATE_INFO (t))
{
tree tmpl =
(TYPE_TEMPLATE_INFO (t)) ? TYPE_TI_TEMPLATE (t) : NULL;
if (tmpl)
error ("explicit instantiation of non-class template %qD", tmpl);
else
error ("explicit instantiation of non-template type %qT", t);
return;
}
complete_type (t);
if (!COMPLETE_TYPE_P (t))
{
if (complain & tf_error)
error ("explicit instantiation of %q#T before definition of template",
t);
return;
}
if (storage != NULL_TREE)
{
if (!in_system_header_at (input_location))
{
if (storage == ridpointers[(int) RID_EXTERN])
{
if (cxx_dialect == cxx98)
pedwarn (input_location, OPT_Wpedantic, 
"ISO C++ 1998 forbids the use of %<extern%> on "
"explicit instantiations");
}
else
pedwarn (input_location, OPT_Wpedantic, 
"ISO C++ forbids the use of %qE"
" on explicit instantiations", storage);
}
if (storage == ridpointers[(int) RID_INLINE])
nomem_p = 1;
else if (storage == ridpointers[(int) RID_EXTERN])
extern_p = 1;
else if (storage == ridpointers[(int) RID_STATIC])
static_p = 1;
else
{
error ("storage class %qD applied to template instantiation",
storage);
extern_p = 0;
}
}
if (CLASSTYPE_TEMPLATE_SPECIALIZATION (t))
{
return;
}
else if (CLASSTYPE_EXPLICIT_INSTANTIATION (t))
{
previous_instantiation_extern_p = CLASSTYPE_INTERFACE_ONLY (t);
if (!previous_instantiation_extern_p && !extern_p
&& (complain & tf_error))
permerror (input_location, "duplicate explicit instantiation of %q#T", t);
if (!CLASSTYPE_INTERFACE_ONLY (t))
return;
}
check_explicit_instantiation_namespace (TYPE_NAME (t));
mark_class_instantiated (t, extern_p);
if (nomem_p)
return;
for (tree fld = TYPE_FIELDS (t); fld; fld = DECL_CHAIN (fld))
if ((VAR_P (fld)
|| (TREE_CODE (fld) == FUNCTION_DECL
&& !static_p
&& user_provided_p (fld)))
&& DECL_TEMPLATE_INSTANTIATION (fld))
{
mark_decl_instantiated (fld, extern_p);
if (! extern_p)
instantiate_decl (fld, true,
true);
}
if (CLASSTYPE_NESTED_UTDS (t))
binding_table_foreach (CLASSTYPE_NESTED_UTDS (t),
bt_instantiate_type_proc, &storage);
}
static void
regenerate_decl_from_template (tree decl, tree tmpl, tree args)
{
tree code_pattern;
code_pattern = DECL_TEMPLATE_RESULT (tmpl);
push_access_scope (decl);
if (TREE_CODE (decl) == FUNCTION_DECL)
{
tree decl_parm;
tree pattern_parm;
tree specs;
int args_depth;
int parms_depth;
args_depth = TMPL_ARGS_DEPTH (args);
parms_depth = TMPL_PARMS_DEPTH (DECL_TEMPLATE_PARMS (tmpl));
if (args_depth > parms_depth)
args = get_innermost_template_args (args, parms_depth);
specs = tsubst_exception_specification (TREE_TYPE (code_pattern),
args, tf_error, NULL_TREE,
false);
if (specs && specs != error_mark_node)
TREE_TYPE (decl) = build_exception_variant (TREE_TYPE (decl),
specs);
decl_parm = skip_artificial_parms_for (decl,
DECL_ARGUMENTS (decl));
pattern_parm
= skip_artificial_parms_for (code_pattern,
DECL_ARGUMENTS (code_pattern));
while (decl_parm && !DECL_PACK_P (pattern_parm))
{
tree parm_type;
tree attributes;
if (DECL_NAME (decl_parm) != DECL_NAME (pattern_parm))
DECL_NAME (decl_parm) = DECL_NAME (pattern_parm);
parm_type = tsubst (TREE_TYPE (pattern_parm), args, tf_error,
NULL_TREE);
parm_type = type_decays_to (parm_type);
if (!same_type_p (TREE_TYPE (decl_parm), parm_type))
TREE_TYPE (decl_parm) = parm_type;
attributes = DECL_ATTRIBUTES (pattern_parm);
if (DECL_ATTRIBUTES (decl_parm) != attributes)
{
DECL_ATTRIBUTES (decl_parm) = attributes;
cplus_decl_attributes (&decl_parm, attributes, 0);
}
decl_parm = DECL_CHAIN (decl_parm);
pattern_parm = DECL_CHAIN (pattern_parm);
}
if (pattern_parm && DECL_PACK_P (pattern_parm))
{
int i, len;
tree expanded_types;
expanded_types = tsubst_pack_expansion (TREE_TYPE (pattern_parm), 
args, tf_error, NULL_TREE);
len = TREE_VEC_LENGTH (expanded_types);
for (i = 0; i < len; i++)
{
tree parm_type;
tree attributes;
if (DECL_NAME (decl_parm) != DECL_NAME (pattern_parm))
DECL_NAME (decl_parm) = 
make_ith_pack_parameter_name (DECL_NAME (pattern_parm), i);
parm_type = TREE_VEC_ELT (expanded_types, i);
parm_type = type_decays_to (parm_type);
if (!same_type_p (TREE_TYPE (decl_parm), parm_type))
TREE_TYPE (decl_parm) = parm_type;
attributes = DECL_ATTRIBUTES (pattern_parm);
if (DECL_ATTRIBUTES (decl_parm) != attributes)
{
DECL_ATTRIBUTES (decl_parm) = attributes;
cplus_decl_attributes (&decl_parm, attributes, 0);
}
decl_parm = DECL_CHAIN (decl_parm);
}
}
if (DECL_DECLARED_INLINE_P (code_pattern)
&& !DECL_DECLARED_INLINE_P (decl))
DECL_DECLARED_INLINE_P (decl) = 1;
}
else if (VAR_P (decl))
{
start_lambda_scope (decl);
DECL_INITIAL (decl) =
tsubst_expr (DECL_INITIAL (code_pattern), args,
tf_error, DECL_TI_TEMPLATE (decl),
false);
finish_lambda_scope ();
if (VAR_HAD_UNKNOWN_BOUND (decl))
TREE_TYPE (decl) = tsubst (TREE_TYPE (code_pattern), args,
tf_error, DECL_TI_TEMPLATE (decl));
}
else
gcc_unreachable ();
pop_access_scope (decl);
}
tree
template_for_substitution (tree decl)
{
tree tmpl = DECL_TI_TEMPLATE (decl);
while (
DECL_TEMPLATE_INSTANTIATION (tmpl)
|| (TREE_CODE (decl) == FUNCTION_DECL
&& DECL_FRIEND_PSEUDO_TEMPLATE_INSTANTIATION (tmpl)
&& !DECL_INITIAL (DECL_TEMPLATE_RESULT (tmpl))))
{
tmpl = DECL_TI_TEMPLATE (tmpl);
}
return tmpl;
}
bool
always_instantiate_p (tree decl)
{
return ((TREE_CODE (decl) == FUNCTION_DECL
&& (DECL_DECLARED_INLINE_P (decl)
|| type_uses_auto (TREE_TYPE (TREE_TYPE (decl)))))
|| (VAR_P (decl)
&& decl_maybe_constant_var_p (decl)));
}
bool
maybe_instantiate_noexcept (tree fn, tsubst_flags_t complain)
{
tree fntype, spec, noex, clone;
if (processing_template_decl
&& (!flag_noexcept_type || type_dependent_expression_p (fn)))
return true;
if (DECL_CLONED_FUNCTION_P (fn))
fn = DECL_CLONED_FUNCTION (fn);
fntype = TREE_TYPE (fn);
spec = TYPE_RAISES_EXCEPTIONS (fntype);
if (!spec || !TREE_PURPOSE (spec))
return true;
noex = TREE_PURPOSE (spec);
if (TREE_CODE (noex) == DEFERRED_NOEXCEPT)
{
static hash_set<tree>* fns = new hash_set<tree>;
bool added = false;
if (DEFERRED_NOEXCEPT_PATTERN (noex) == NULL_TREE)
spec = get_defaulted_eh_spec (fn, complain);
else if (!(added = !fns->add (fn)))
{
location_t loc = EXPR_LOC_OR_LOC (DEFERRED_NOEXCEPT_PATTERN (noex),
DECL_SOURCE_LOCATION (fn));
error_at (loc,
"exception specification of %qD depends on itself",
fn);
spec = noexcept_false_spec;
}
else if (push_tinst_level (fn))
{
push_access_scope (fn);
push_deferring_access_checks (dk_no_deferred);
input_location = DECL_SOURCE_LOCATION (fn);
noex = tsubst_copy_and_build (DEFERRED_NOEXCEPT_PATTERN (noex),
DEFERRED_NOEXCEPT_ARGS (noex),
tf_warning_or_error, fn,
false,
true);
spec = build_noexcept_spec (noex, tf_warning_or_error);
pop_deferring_access_checks ();
pop_access_scope (fn);
pop_tinst_level ();
if (spec == error_mark_node)
spec = noexcept_false_spec;
}
else
spec = noexcept_false_spec;
if (added)
fns->remove (fn);
if (spec == error_mark_node)
return false;
TREE_TYPE (fn) = build_exception_variant (fntype, spec);
}
FOR_EACH_CLONE (clone, fn)
{
if (TREE_TYPE (clone) == fntype)
TREE_TYPE (clone) = TREE_TYPE (fn);
else
TREE_TYPE (clone) = build_exception_variant (TREE_TYPE (clone), spec);
}
return true;
}
static void
register_parameter_specializations (tree pattern, tree inst)
{
tree tmpl_parm = DECL_ARGUMENTS (pattern);
tree spec_parm = DECL_ARGUMENTS (inst);
if (DECL_NONSTATIC_MEMBER_FUNCTION_P (inst))
{
register_local_specialization (spec_parm, tmpl_parm);
spec_parm = skip_artificial_parms_for (inst, spec_parm);
tmpl_parm = skip_artificial_parms_for (pattern, tmpl_parm);
}
for (; tmpl_parm; tmpl_parm = DECL_CHAIN (tmpl_parm))
{
if (!DECL_PACK_P (tmpl_parm))
{
register_local_specialization (spec_parm, tmpl_parm);
spec_parm = DECL_CHAIN (spec_parm);
}
else
{
tree argpack = extract_fnparm_pack (tmpl_parm, &spec_parm);
register_local_specialization (argpack, tmpl_parm);
}
}
gcc_assert (!spec_parm);
}
tree
instantiate_decl (tree d, bool defer_ok, bool expl_inst_class_mem_p)
{
tree tmpl = DECL_TI_TEMPLATE (d);
tree gen_args;
tree args;
tree td;
tree code_pattern;
tree spec;
tree gen_tmpl;
bool pattern_defined;
location_t saved_loc = input_location;
int saved_unevaluated_operand = cp_unevaluated_operand;
int saved_inhibit_evaluation_warnings = c_inhibit_evaluation_warnings;
bool external_p;
bool deleted_p;
gcc_assert (VAR_OR_FUNCTION_DECL_P (d));
gcc_assert (!DECL_DECLARED_CONCEPT_P (d));
if (VAR_P (d))
defer_ok = false;
if (TREE_CODE (d) == FUNCTION_DECL && DECL_CLONED_FUNCTION_P (d))
d = DECL_CLONED_FUNCTION (d);
if (DECL_TEMPLATE_INSTANTIATED (d)
|| (TREE_CODE (d) == FUNCTION_DECL
&& DECL_DEFAULTED_FN (d) && DECL_INITIAL (d))
|| DECL_TEMPLATE_SPECIALIZATION (d))
return d;
external_p = (DECL_INTERFACE_KNOWN (d) && DECL_REALLY_EXTERN (d));
if (external_p && !always_instantiate_p (d))
return d;
gen_tmpl = most_general_template (tmpl);
gen_args = DECL_TI_ARGS (d);
if (tmpl != gen_tmpl)
gcc_assert (TMPL_PARMS_DEPTH (DECL_TEMPLATE_PARMS (gen_tmpl))
== TMPL_ARGS_DEPTH (gen_args));
gcc_assert ((spec = retrieve_specialization (gen_tmpl, gen_args, 0)) == d
|| spec == NULL_TREE);
if (! push_tinst_level (d))
return d;
timevar_push (TV_TEMPLATE_INST);
td = template_for_substitution (d);
args = gen_args;
if (VAR_P (d))
{
tree tid = lookup_template_variable (gen_tmpl, gen_args);
tree elt = most_specialized_partial_spec (tid, tf_warning_or_error);
if (elt && elt != error_mark_node)
{
td = TREE_VALUE (elt);
args = TREE_PURPOSE (elt);
}
}
code_pattern = DECL_TEMPLATE_RESULT (td);
gcc_assert (d != code_pattern);
if ((DECL_NAMESPACE_SCOPE_P (d) && !DECL_INITIALIZED_IN_CLASS_P (d))
|| DECL_TEMPLATE_SPECIALIZATION (td))
args = get_innermost_template_args
(args, TMPL_PARMS_DEPTH (DECL_TEMPLATE_PARMS (td)));
if (TREE_CODE (d) == FUNCTION_DECL)
{
deleted_p = DECL_DELETED_FN (code_pattern);
pattern_defined = ((DECL_SAVED_TREE (code_pattern) != NULL_TREE
&& DECL_INITIAL (code_pattern) != error_mark_node)
|| DECL_DEFAULTED_OUTSIDE_CLASS_P (code_pattern)
|| deleted_p);
}
else
{
deleted_p = false;
if (DECL_CLASS_SCOPE_P (code_pattern))
pattern_defined = (! DECL_IN_AGGR_P (code_pattern)
|| DECL_INLINE_VAR_P (code_pattern));
else
pattern_defined = ! DECL_EXTERNAL (code_pattern);
}
push_deferring_access_checks (dk_no_deferred);
if (pattern_defined
&& !DECL_INTERFACE_KNOWN (d)
&& !DECL_NOT_REALLY_EXTERN (d))
mark_definable (d);
DECL_SOURCE_LOCATION (td) = DECL_SOURCE_LOCATION (code_pattern);
DECL_SOURCE_LOCATION (d) = DECL_SOURCE_LOCATION (code_pattern);
input_location = DECL_SOURCE_LOCATION (d);
if (!pattern_defined && expl_inst_class_mem_p
&& DECL_EXPLICIT_INSTANTIATION (d))
{
if (TREE_PUBLIC (d))
{
DECL_NOT_REALLY_EXTERN (d) = 0;
DECL_INTERFACE_KNOWN (d) = 0;
}
SET_DECL_IMPLICIT_INSTANTIATION (d);
}
if (
! pattern_defined
|| defer_ok
|| (external_p && VAR_P (d))
|| deleted_p)
{
if (VAR_P (d)
&& !DECL_INITIAL (d)
&& DECL_INITIAL (code_pattern))
{
tree ns;
tree init;
bool const_init = false;
bool enter_context = DECL_CLASS_SCOPE_P (d);
ns = decl_namespace_context (d);
push_nested_namespace (ns);
if (enter_context)
push_nested_class (DECL_CONTEXT (d));
init = tsubst_expr (DECL_INITIAL (code_pattern),
args,
tf_warning_or_error, NULL_TREE,
false);
if (!DECL_INITIAL (d))
{
const_init
= DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P (code_pattern);
cp_finish_decl (d, init, const_init,
NULL_TREE,
LOOKUP_ONLYCONVERTING);
}
if (enter_context)
pop_nested_class ();
pop_nested_namespace (ns);
}
input_location = saved_loc;
if (at_eof && !pattern_defined
&& DECL_EXPLICIT_INSTANTIATION (d)
&& DECL_NOT_REALLY_EXTERN (d))
permerror (input_location,  "explicit instantiation of %qD "
"but no definition available", d);
if (cp_unevaluated_operand != 0)
goto out;
if (!(external_p && VAR_P (d)))
add_pending_template (d);
goto out;
}
if (TREE_PUBLIC (d) && !DECL_REALLY_EXTERN (d) && !repo_emit_p (d))
{
if (pch_file)
add_pending_template (d);
if (!(TREE_CODE (d) == FUNCTION_DECL && possibly_inlined_p (d)))
goto out;
}
bool push_to_top, nested;
tree fn_context;
fn_context = decl_function_context (d);
if (LAMBDA_FUNCTION_P (d))
fn_context = NULL_TREE;
nested = current_function_decl != NULL_TREE;
push_to_top = !(nested && fn_context == current_function_decl);
vec<tree> omp_privatization_save;
if (nested)
save_omp_privatization_clauses (omp_privatization_save);
if (push_to_top)
push_to_top_level ();
else
{
push_function_context ();
cp_unevaluated_operand = 0;
c_inhibit_evaluation_warnings = 0;
}
DECL_TEMPLATE_INSTANTIATED (d) = 1;
regenerate_decl_from_template (d, td, args);
input_location = DECL_SOURCE_LOCATION (d);
if (VAR_P (d))
{
tree init;
bool const_init = false;
SET_DECL_RTL (d, NULL);
DECL_IN_AGGR_P (d) = 0;
init = DECL_INITIAL (d);
DECL_INITIAL (d) = NULL_TREE;
DECL_INITIALIZED_P (d) = 0;
DECL_EXTERNAL (d) = 0;
bool enter_context = DECL_CLASS_SCOPE_P (d);
if (enter_context)
push_nested_class (DECL_CONTEXT (d));
const_init = DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P (code_pattern);
cp_finish_decl (d, init, const_init, NULL_TREE, 0);
if (enter_context)
pop_nested_class ();
if (variable_template_p (gen_tmpl))
note_variable_template_instantiation (d);
}
else if (TREE_CODE (d) == FUNCTION_DECL && DECL_DEFAULTED_FN (code_pattern))
synthesize_method (d);
else if (TREE_CODE (d) == FUNCTION_DECL)
{
local_specialization_stack lss (push_to_top ? lss_blank : lss_copy);
tree block = NULL_TREE;
if (DECL_OMP_DECLARE_REDUCTION_P (code_pattern)
&& TREE_CODE (DECL_CONTEXT (code_pattern)) == FUNCTION_DECL)
block = push_stmt_list ();
else
start_preparsed_function (d, NULL_TREE, SF_PRE_PARSED);
perform_typedefs_access_check (DECL_TEMPLATE_RESULT (td),
args);
register_parameter_specializations (code_pattern, d);
if (DECL_OMP_DECLARE_REDUCTION_P (code_pattern))
tsubst_omp_udr (DECL_SAVED_TREE (code_pattern), args,
tf_warning_or_error, tmpl);
else
{
tsubst_expr (DECL_SAVED_TREE (code_pattern), args,
tf_warning_or_error, tmpl,
false);
input_location
= DECL_STRUCT_FUNCTION (code_pattern)->function_end_locus;
current_function_infinite_loop
= DECL_STRUCT_FUNCTION (code_pattern)->language->infinite_loop;
}
if (DECL_OMP_DECLARE_REDUCTION_P (code_pattern)
&& TREE_CODE (DECL_CONTEXT (code_pattern)) == FUNCTION_DECL)
DECL_SAVED_TREE (d) = pop_stmt_list (block);
else
{
d = finish_function (false);
expand_or_defer_fn (d);
}
if (DECL_OMP_DECLARE_REDUCTION_P (code_pattern))
cp_check_omp_declare_reduction (d);
}
TI_PENDING_TEMPLATE_FLAG (DECL_TEMPLATE_INFO (d)) = 0;
if (push_to_top)
pop_from_top_level ();
else
pop_function_context ();
if (nested)
restore_omp_privatization_clauses (omp_privatization_save);
out:
pop_deferring_access_checks ();
timevar_pop (TV_TEMPLATE_INST);
pop_tinst_level ();
input_location = saved_loc;
cp_unevaluated_operand = saved_unevaluated_operand;
c_inhibit_evaluation_warnings = saved_inhibit_evaluation_warnings;
return d;
}
void
instantiate_pending_templates (int retries)
{
int reconsider;
location_t saved_loc = input_location;
if (pending_templates && retries >= max_tinst_depth)
{
tree decl = pending_templates->tinst->maybe_get_node ();
fatal_error (input_location,
"template instantiation depth exceeds maximum of %d"
" instantiating %q+D, possibly from virtual table generation"
" (use -ftemplate-depth= to increase the maximum)",
max_tinst_depth, decl);
if (TREE_CODE (decl) == FUNCTION_DECL)
DECL_INITIAL (decl) = error_mark_node;
return;
}
do
{
struct pending_template **t = &pending_templates;
struct pending_template *last = NULL;
reconsider = 0;
while (*t)
{
tree instantiation = reopen_tinst_level ((*t)->tinst);
bool complete = false;
if (TYPE_P (instantiation))
{
if (!COMPLETE_TYPE_P (instantiation))
{
instantiate_class_template (instantiation);
if (CLASSTYPE_TEMPLATE_INSTANTIATION (instantiation))
for (tree fld = TYPE_FIELDS (instantiation);
fld; fld = TREE_CHAIN (fld))
if ((VAR_P (fld)
|| (TREE_CODE (fld) == FUNCTION_DECL
&& !DECL_ARTIFICIAL (fld)))
&& DECL_TEMPLATE_INSTANTIATION (fld))
instantiate_decl (fld,
false,
false);
if (COMPLETE_TYPE_P (instantiation))
reconsider = 1;
}
complete = COMPLETE_TYPE_P (instantiation);
}
else
{
if (!DECL_TEMPLATE_SPECIALIZATION (instantiation)
&& !DECL_TEMPLATE_INSTANTIATED (instantiation))
{
instantiation
= instantiate_decl (instantiation,
false,
false);
if (DECL_TEMPLATE_INSTANTIATED (instantiation))
reconsider = 1;
}
complete = (DECL_TEMPLATE_SPECIALIZATION (instantiation)
|| DECL_TEMPLATE_INSTANTIATED (instantiation));
}
if (complete)
{
struct pending_template *drop = *t;
*t = (*t)->next;
set_refcount_ptr (drop->tinst);
pending_template_freelist ().free (drop);
}
else
{
last = *t;
t = &(*t)->next;
}
tinst_depth = 0;
set_refcount_ptr (current_tinst_level);
}
last_pending_template = last;
}
while (reconsider);
input_location = saved_loc;
}
static tree
tsubst_initializer_list (tree t, tree argvec)
{
tree inits = NULL_TREE;
tree target_ctor = error_mark_node;
for (; t; t = TREE_CHAIN (t))
{
tree decl;
tree init;
tree expanded_bases = NULL_TREE;
tree expanded_arguments = NULL_TREE;
int i, len = 1;
if (TREE_CODE (TREE_PURPOSE (t)) == TYPE_PACK_EXPANSION)
{
tree expr;
tree arg;
expanded_bases = tsubst_pack_expansion (TREE_PURPOSE (t), argvec,
tf_warning_or_error,
NULL_TREE);
if (expanded_bases == error_mark_node)
continue;
len = TREE_VEC_LENGTH (expanded_bases);
expanded_arguments = make_tree_vec (len);
for (i = 0; i < len; i++)
TREE_VEC_ELT (expanded_arguments, i) = NULL_TREE;
expr = make_node (EXPR_PACK_EXPANSION);
PACK_EXPANSION_LOCAL_P (expr) = true;
PACK_EXPANSION_PARAMETER_PACKS (expr) =
PACK_EXPANSION_PARAMETER_PACKS (TREE_PURPOSE (t));
if (TREE_VALUE (t) == void_type_node)
{
for (i = 0; i < len; i++)
TREE_VEC_ELT (expanded_arguments, i) = void_type_node;
}
else
{
in_base_initializer = 1;
for (arg = TREE_VALUE (t); arg; arg = TREE_CHAIN (arg))
{
tree expanded_exprs;
SET_PACK_EXPANSION_PATTERN (expr, TREE_VALUE (arg));
expanded_exprs 
= tsubst_pack_expansion (expr, argvec,
tf_warning_or_error,
NULL_TREE);
if (expanded_exprs == error_mark_node)
continue;
for (i = 0; i < len; i++)
{
TREE_VEC_ELT (expanded_arguments, i) = 
tree_cons (NULL_TREE, 
TREE_VEC_ELT (expanded_exprs, i),
TREE_VEC_ELT (expanded_arguments, i));
}
}
in_base_initializer = 0;
for (i = 0; i < len; i++)
{
TREE_VEC_ELT (expanded_arguments, i) = 
nreverse (TREE_VEC_ELT (expanded_arguments, i));
}
}
}
for (i = 0; i < len; ++i)
{
if (expanded_bases)
{
decl = TREE_VEC_ELT (expanded_bases, i);
decl = expand_member_init (decl);
init = TREE_VEC_ELT (expanded_arguments, i);
}
else
{
tree tmp;
decl = tsubst_copy (TREE_PURPOSE (t), argvec, 
tf_warning_or_error, NULL_TREE);
decl = expand_member_init (decl);
if (decl && !DECL_P (decl))
in_base_initializer = 1;
init = TREE_VALUE (t);
tmp = init;
if (init != void_type_node)
init = tsubst_expr (init, argvec,
tf_warning_or_error, NULL_TREE,
false);
if (init == NULL_TREE && tmp != NULL_TREE)
init = void_type_node;
in_base_initializer = 0;
}
if (target_ctor != error_mark_node
&& init != error_mark_node)
{
error ("mem-initializer for %qD follows constructor delegation",
decl);
return inits;
}
if (init != error_mark_node
&& decl && CLASS_TYPE_P (decl)
&& same_type_p (decl, current_class_type))
{
maybe_warn_cpp0x (CPP0X_DELEGATING_CTORS);
if (inits)
{
error ("constructor delegation follows mem-initializer for %qD",
TREE_PURPOSE (inits));
continue;
}
target_ctor = init;
}
if (decl)
{
init = build_tree_list (decl, init);
TREE_CHAIN (init) = inits;
inits = init;
}
}
}
return inits;
}
static void
set_current_access_from_decl (tree decl)
{
if (TREE_PRIVATE (decl))
current_access_specifier = access_private_node;
else if (TREE_PROTECTED (decl))
current_access_specifier = access_protected_node;
else
current_access_specifier = access_public_node;
}
static void
tsubst_enum (tree tag, tree newtag, tree args)
{
tree e;
if (SCOPED_ENUM_P (newtag))
begin_scope (sk_scoped_enum, newtag);
for (e = TYPE_VALUES (tag); e; e = TREE_CHAIN (e))
{
tree value;
tree decl;
decl = TREE_VALUE (e);
value = tsubst_expr (DECL_INITIAL (decl),
args, tf_warning_or_error, NULL_TREE,
true);
set_current_access_from_decl (decl);
build_enumerator (DECL_NAME (decl), value, newtag,
DECL_ATTRIBUTES (decl), DECL_SOURCE_LOCATION (decl));
}
if (SCOPED_ENUM_P (newtag))
finish_scope ();
finish_enum_value_list (newtag);
finish_enum (newtag);
DECL_SOURCE_LOCATION (TYPE_NAME (newtag))
= DECL_SOURCE_LOCATION (TYPE_NAME (tag));
}
tree
get_mostly_instantiated_function_type (tree decl)
{
return TREE_TYPE (DECL_TI_TEMPLATE (decl));
}
bool
problematic_instantiation_changed (void)
{
return current_tinst_level != last_error_tinst_level;
}
void
record_last_problematic_instantiation (void)
{
set_refcount_ptr (last_error_tinst_level, current_tinst_level);
}
struct tinst_level *
current_instantiation (void)
{
return current_tinst_level;
}
bool
instantiating_current_function_p (void)
{
return (current_instantiation ()
&& (current_instantiation ()->maybe_get_node ()
== current_function_decl));
}
static bool
invalid_nontype_parm_type_p (tree type, tsubst_flags_t complain)
{
if (INTEGRAL_OR_ENUMERATION_TYPE_P (type))
return false;
else if (TYPE_PTR_P (type))
return false;
else if (TREE_CODE (type) == REFERENCE_TYPE
&& !TYPE_REF_IS_RVALUE (type))
return false;
else if (TYPE_PTRMEM_P (type))
return false;
else if (TREE_CODE (type) == TEMPLATE_TYPE_PARM)
return false;
else if (TREE_CODE (type) == TYPENAME_TYPE)
return false;
else if (TREE_CODE (type) == DECLTYPE_TYPE)
return false;
else if (TREE_CODE (type) == NULLPTR_TYPE)
return false;
else if (cxx_dialect >= cxx11
&& TREE_CODE (type) == BOUND_TEMPLATE_TEMPLATE_PARM)
return false;
if (complain & tf_error)
{
if (type == error_mark_node)
inform (input_location, "invalid template non-type parameter");
else
error ("%q#T is not a valid type for a template non-type parameter",
type);
}
return true;
}
static bool
dependent_type_p_r (tree type)
{
tree scope;
if (TREE_CODE (type) == TEMPLATE_TYPE_PARM
|| TREE_CODE (type) == TEMPLATE_TEMPLATE_PARM)
return true;
if (TREE_CODE (type) == TYPENAME_TYPE)
return true;
if (dependent_alias_template_spec_p (type))
return true;
if (TYPE_PTRMEM_P (type))
return (dependent_type_p (TYPE_PTRMEM_CLASS_TYPE (type))
|| dependent_type_p (TYPE_PTRMEM_POINTED_TO_TYPE
(type)));
else if (TYPE_PTR_P (type)
|| TREE_CODE (type) == REFERENCE_TYPE)
return dependent_type_p (TREE_TYPE (type));
else if (TREE_CODE (type) == FUNCTION_TYPE
|| TREE_CODE (type) == METHOD_TYPE)
{
tree arg_type;
if (dependent_type_p (TREE_TYPE (type)))
return true;
for (arg_type = TYPE_ARG_TYPES (type);
arg_type;
arg_type = TREE_CHAIN (arg_type))
if (dependent_type_p (TREE_VALUE (arg_type)))
return true;
if (cxx_dialect >= cxx17)
if (tree spec = TYPE_RAISES_EXCEPTIONS (type))
if (tree noex = TREE_PURPOSE (spec))
if (TREE_CODE (noex) != DEFERRED_NOEXCEPT
&& value_dependent_expression_p (noex))
return true;
return false;
}
if (TREE_CODE (type) == ARRAY_TYPE)
{
if (TYPE_DOMAIN (type)
&& dependent_type_p (TYPE_DOMAIN (type)))
return true;
return dependent_type_p (TREE_TYPE (type));
}
if (TREE_CODE (type) == BOUND_TEMPLATE_TEMPLATE_PARM)
return true;
else if (CLASS_TYPE_P (type) && CLASSTYPE_TEMPLATE_INFO (type)
&& (any_dependent_template_arguments_p
(INNERMOST_TEMPLATE_ARGS (CLASSTYPE_TI_ARGS (type)))))
return true;
if (TREE_CODE (type) == TYPEOF_TYPE
|| TREE_CODE (type) == DECLTYPE_TYPE
|| TREE_CODE (type) == UNDERLYING_TYPE)
return true;
if (TREE_CODE (type) == TYPE_ARGUMENT_PACK)
{
tree args = ARGUMENT_PACK_ARGS (type);
int i, len = TREE_VEC_LENGTH (args);
for (i = 0; i < len; ++i)
if (dependent_template_arg_p (TREE_VEC_ELT (args, i)))
return true;
}
if (TREE_CODE (type) == TYPE_PACK_EXPANSION)
return true;
if (any_dependent_type_attributes_p (TYPE_ATTRIBUTES (type)))
return true;
scope = TYPE_CONTEXT (type);
if (scope && TYPE_P (scope))
return dependent_type_p (scope);
else if (scope && TREE_CODE (scope) == FUNCTION_DECL
&& DECL_LANG_SPECIFIC (scope)
&& DECL_TEMPLATE_INFO (scope)
&& (any_dependent_template_arguments_p
(INNERMOST_TEMPLATE_ARGS (DECL_TI_ARGS (scope)))))
return true;
return false;
}
bool
dependent_type_p (tree type)
{
if (!processing_template_decl)
{
gcc_assert (type);
gcc_assert (TREE_CODE (type) != TEMPLATE_TYPE_PARM || is_auto (type));
return false;
}
if (!type)
return true;
if (type == error_mark_node)
return false;
gcc_checking_assert (type != global_type_node);
if (!TYPE_DEPENDENT_P_VALID (type))
{
TYPE_DEPENDENT_P (type) = dependent_type_p_r (type);
TYPE_DEPENDENT_P_VALID (type) = 1;
}
return TYPE_DEPENDENT_P (type);
}
bool
dependent_scope_p (tree scope)
{
return (scope && TYPE_P (scope) && dependent_type_p (scope)
&& !currently_open_class (scope));
}
static bool
unknown_base_ref_p (tree t)
{
if (!current_class_ptr)
return false;
tree mem = TREE_OPERAND (t, 1);
if (shared_member_p (mem))
return false;
tree cur = current_nonlambda_class_type ();
if (!any_dependent_bases_p (cur))
return false;
tree ctx = TREE_OPERAND (t, 0);
if (DERIVED_FROM_P (ctx, cur))
return false;
return true;
}
static bool
instantiation_dependent_scope_ref_p (tree t)
{
if (DECL_P (TREE_OPERAND (t, 1))
&& CLASS_TYPE_P (TREE_OPERAND (t, 0))
&& !unknown_base_ref_p (t)
&& accessible_in_template_p (TREE_OPERAND (t, 0),
TREE_OPERAND (t, 1)))
return false;
else
return true;
}
bool
value_dependent_expression_p (tree expression)
{
if (!processing_template_decl || expression == NULL_TREE)
return false;
if (type_dependent_expression_p (expression))
return true;
switch (TREE_CODE (expression))
{
case BASELINK:
return dependent_type_p (BINFO_TYPE (BASELINK_BINFO (expression)));
case FUNCTION_DECL:
if (DECL_CLASS_SCOPE_P (expression)
&& dependent_type_p (DECL_CONTEXT (expression)))
return true;
break;
case IDENTIFIER_NODE:
return true;
case TEMPLATE_PARM_INDEX:
return true;
case CONST_DECL:
if (DECL_TEMPLATE_PARM_P (expression))
return true;
return value_dependent_expression_p (DECL_INITIAL (expression));
case VAR_DECL:
if (DECL_DEPENDENT_INIT_P (expression)
|| TREE_CODE (TREE_TYPE (expression)) == REFERENCE_TYPE)
return true;
if (DECL_HAS_VALUE_EXPR_P (expression))
{
tree value_expr = DECL_VALUE_EXPR (expression);
if (value_dependent_expression_p (value_expr))
return true;
}
return false;
case DYNAMIC_CAST_EXPR:
case STATIC_CAST_EXPR:
case CONST_CAST_EXPR:
case REINTERPRET_CAST_EXPR:
case CAST_EXPR:
case IMPLICIT_CONV_EXPR:
{
tree type = TREE_TYPE (expression);
if (dependent_type_p (type))
return true;
expression = TREE_OPERAND (expression, 0);
if (!expression)
{
gcc_assert (cxx_dialect >= cxx11
|| INTEGRAL_OR_ENUMERATION_TYPE_P (type));
return false;
}
if (TREE_CODE (expression) == TREE_LIST)
return any_value_dependent_elements_p (expression);
return value_dependent_expression_p (expression);
}
case SIZEOF_EXPR:
if (SIZEOF_EXPR_TYPE_P (expression))
return dependent_type_p (TREE_TYPE (TREE_OPERAND (expression, 0)));
case ALIGNOF_EXPR:
case TYPEID_EXPR:
expression = TREE_OPERAND (expression, 0);
if (PACK_EXPANSION_P (expression))
return true;
else if (TYPE_P (expression))
return dependent_type_p (expression);
return instantiation_dependent_uneval_expression_p (expression);
case AT_ENCODE_EXPR:
expression = TREE_OPERAND (expression, 0);
return dependent_type_p (expression);
case NOEXCEPT_EXPR:
expression = TREE_OPERAND (expression, 0);
return instantiation_dependent_uneval_expression_p (expression);
case SCOPE_REF:
return instantiation_dependent_scope_ref_p (expression);
case COMPONENT_REF:
return (value_dependent_expression_p (TREE_OPERAND (expression, 0))
|| value_dependent_expression_p (TREE_OPERAND (expression, 1)));
case NONTYPE_ARGUMENT_PACK:
{
tree values = ARGUMENT_PACK_ARGS (expression);
int i, len = TREE_VEC_LENGTH (values);
for (i = 0; i < len; ++i)
if (value_dependent_expression_p (TREE_VEC_ELT (values, i)))
return true;
return false;
}
case TRAIT_EXPR:
{
tree type2 = TRAIT_EXPR_TYPE2 (expression);
if (dependent_type_p (TRAIT_EXPR_TYPE1 (expression)))
return true;
if (!type2)
return false;
if (TREE_CODE (type2) != TREE_LIST)
return dependent_type_p (type2);
for (; type2; type2 = TREE_CHAIN (type2))
if (dependent_type_p (TREE_VALUE (type2)))
return true;
return false;
}
case MODOP_EXPR:
return ((value_dependent_expression_p (TREE_OPERAND (expression, 0)))
|| (value_dependent_expression_p (TREE_OPERAND (expression, 2))));
case ARRAY_REF:
return ((value_dependent_expression_p (TREE_OPERAND (expression, 0)))
|| (value_dependent_expression_p (TREE_OPERAND (expression, 1))));
case ADDR_EXPR:
{
tree op = TREE_OPERAND (expression, 0);
return (value_dependent_expression_p (op)
|| has_value_dependent_address (op));
}
case REQUIRES_EXPR:
return true;
case TYPE_REQ:
return dependent_type_p (TREE_OPERAND (expression, 0));
case CALL_EXPR:
{
if (value_dependent_expression_p (CALL_EXPR_FN (expression)))
return true;
tree fn = get_callee_fndecl (expression);
int i, nargs;
nargs = call_expr_nargs (expression);
for (i = 0; i < nargs; ++i)
{
tree op = CALL_EXPR_ARG (expression, i);
if (i == 0 && fn && DECL_DECLARED_CONSTEXPR_P (fn)
&& DECL_NONSTATIC_MEMBER_FUNCTION_P (fn)
&& TREE_CODE (op) == ADDR_EXPR)
op = TREE_OPERAND (op, 0);
if (value_dependent_expression_p (op))
return true;
}
return false;
}
case TEMPLATE_ID_EXPR:
return variable_concept_p (TREE_OPERAND (expression, 0));
case CONSTRUCTOR:
{
unsigned ix;
tree val;
if (dependent_type_p (TREE_TYPE (expression)))
return true;
FOR_EACH_CONSTRUCTOR_VALUE (CONSTRUCTOR_ELTS (expression), ix, val)
if (value_dependent_expression_p (val))
return true;
return false;
}
case STMT_EXPR:
return true;
default:
switch (TREE_CODE_CLASS (TREE_CODE (expression)))
{
case tcc_reference:
case tcc_unary:
case tcc_comparison:
case tcc_binary:
case tcc_expression:
case tcc_vl_exp:
{
int i, len = cp_tree_operand_length (expression);
for (i = 0; i < len; i++)
{
tree t = TREE_OPERAND (expression, i);
if (t && value_dependent_expression_p (t))
return true;
}
}
break;
default:
break;
}
break;
}
return false;
}
bool
type_dependent_expression_p (tree expression)
{
if (!processing_template_decl)
return false;
if (expression == NULL_TREE || expression == error_mark_node)
return false;
STRIP_ANY_LOCATION_WRAPPER (expression);
if (identifier_p (expression)
|| TREE_CODE (expression) == USING_DECL
|| TREE_CODE (expression) == WILDCARD_DECL)
return true;
if (TREE_CODE (expression) == UNARY_LEFT_FOLD_EXPR
|| TREE_CODE (expression) == UNARY_RIGHT_FOLD_EXPR
|| TREE_CODE (expression) == BINARY_LEFT_FOLD_EXPR
|| TREE_CODE (expression) == BINARY_RIGHT_FOLD_EXPR)
return true;
if (TREE_CODE (expression) == PSEUDO_DTOR_EXPR
|| TREE_CODE (expression) == SIZEOF_EXPR
|| TREE_CODE (expression) == ALIGNOF_EXPR
|| TREE_CODE (expression) == AT_ENCODE_EXPR
|| TREE_CODE (expression) == NOEXCEPT_EXPR
|| TREE_CODE (expression) == TRAIT_EXPR
|| TREE_CODE (expression) == TYPEID_EXPR
|| TREE_CODE (expression) == DELETE_EXPR
|| TREE_CODE (expression) == VEC_DELETE_EXPR
|| TREE_CODE (expression) == THROW_EXPR
|| TREE_CODE (expression) == REQUIRES_EXPR)
return false;
if (TREE_CODE (expression) == DYNAMIC_CAST_EXPR
|| TREE_CODE (expression) == STATIC_CAST_EXPR
|| TREE_CODE (expression) == CONST_CAST_EXPR
|| TREE_CODE (expression) == REINTERPRET_CAST_EXPR
|| TREE_CODE (expression) == IMPLICIT_CONV_EXPR
|| TREE_CODE (expression) == CAST_EXPR)
return dependent_type_p (TREE_TYPE (expression));
if (TREE_CODE (expression) == NEW_EXPR
|| TREE_CODE (expression) == VEC_NEW_EXPR)
{
tree type = TREE_OPERAND (expression, 1);
if (TREE_CODE (type) == TREE_LIST)
return dependent_type_p (TREE_VALUE (TREE_PURPOSE (type)))
|| value_dependent_expression_p
(TREE_OPERAND (TREE_VALUE (type), 1));
else
return dependent_type_p (type);
}
if (TREE_CODE (expression) == SCOPE_REF)
{
tree scope = TREE_OPERAND (expression, 0);
tree name = TREE_OPERAND (expression, 1);
return (type_dependent_expression_p (name)
|| dependent_scope_p (scope));
}
if (TREE_CODE (expression) == TEMPLATE_DECL
&& !DECL_TEMPLATE_TEMPLATE_PARM_P (expression))
return uses_outer_template_parms (expression);
if (TREE_CODE (expression) == STMT_EXPR)
expression = stmt_expr_value_expr (expression);
if (BRACE_ENCLOSED_INITIALIZER_P (expression))
{
tree elt;
unsigned i;
FOR_EACH_CONSTRUCTOR_VALUE (CONSTRUCTOR_ELTS (expression), i, elt)
{
if (type_dependent_expression_p (elt))
return true;
}
return false;
}
if (VAR_P (expression)
&& DECL_CLASS_SCOPE_P (expression)
&& dependent_type_p (DECL_CONTEXT (expression))
&& VAR_HAD_UNKNOWN_BOUND (expression))
return true;
if (VAR_P (expression)
&& TREE_TYPE (expression) != NULL_TREE
&& TREE_CODE (TREE_TYPE (expression)) == ARRAY_TYPE
&& !TYPE_DOMAIN (TREE_TYPE (expression))
&& DECL_INITIAL (expression))
return true;
if (VAR_OR_FUNCTION_DECL_P (expression)
&& DECL_LANG_SPECIFIC (expression)
&& DECL_TEMPLATE_INFO (expression))
{
if (PRIMARY_TEMPLATE_P (DECL_TI_TEMPLATE (expression))
&& (any_dependent_template_arguments_p
(INNERMOST_TEMPLATE_ARGS (DECL_TI_ARGS (expression)))))
return true;
}
if (TREE_CODE (expression) == FUNCTION_DECL
&& !(DECL_CLASS_SCOPE_P (expression)
&& dependent_type_p (DECL_CONTEXT (expression)))
&& !(DECL_LANG_SPECIFIC (expression)
&& DECL_FRIEND_P (expression)
&& (!DECL_FRIEND_CONTEXT (expression)
|| dependent_type_p (DECL_FRIEND_CONTEXT (expression))))
&& !DECL_LOCAL_FUNCTION_P (expression))
{
gcc_assert (!dependent_type_p (TREE_TYPE (expression))
|| undeduced_auto_decl (expression));
return false;
}
if (TREE_CODE (expression) == EXPR_PACK_EXPANSION)
return true;
if (TREE_TYPE (expression) == unknown_type_node)
{
if (TREE_CODE (expression) == ADDR_EXPR)
return type_dependent_expression_p (TREE_OPERAND (expression, 0));
if (TREE_CODE (expression) == COMPONENT_REF
|| TREE_CODE (expression) == OFFSET_REF)
{
if (type_dependent_expression_p (TREE_OPERAND (expression, 0)))
return true;
expression = TREE_OPERAND (expression, 1);
if (identifier_p (expression))
return false;
}
if (TREE_CODE (expression) == SCOPE_REF)
return false;
if (BASELINK_P (expression))
{
if (BASELINK_OPTYPE (expression)
&& dependent_type_p (BASELINK_OPTYPE (expression)))
return true;
expression = BASELINK_FUNCTIONS (expression);
}
if (TREE_CODE (expression) == TEMPLATE_ID_EXPR)
{
if (any_dependent_template_arguments_p
(TREE_OPERAND (expression, 1)))
return true;
expression = TREE_OPERAND (expression, 0);
if (identifier_p (expression))
return true;
}
gcc_assert (TREE_CODE (expression) == OVERLOAD
|| TREE_CODE (expression) == FUNCTION_DECL);
for (lkp_iterator iter (expression); iter; ++iter)
if (type_dependent_expression_p (*iter))
return true;
return false;
}
gcc_assert (TREE_CODE (expression) != TYPE_DECL);
if (DECL_P (expression)
&& any_dependent_type_attributes_p (DECL_ATTRIBUTES (expression)))
return true;
return (dependent_type_p (TREE_TYPE (expression)));
}
bool
type_dependent_object_expression_p (tree object)
{
if (TREE_CODE (object) == IDENTIFIER_NODE)
return true;
tree scope = TREE_TYPE (object);
return (!scope || dependent_scope_p (scope));
}
static tree
instantiation_dependent_r (tree *tp, int *walk_subtrees,
void * )
{
if (TYPE_P (*tp))
{
*walk_subtrees = false;
return NULL_TREE;
}
enum tree_code code = TREE_CODE (*tp);
switch (code)
{
case TREE_LIST:
case TREE_VEC:
case NONTYPE_ARGUMENT_PACK:
return NULL_TREE;
case TEMPLATE_PARM_INDEX:
return *tp;
case SIZEOF_EXPR:
case ALIGNOF_EXPR:
case TYPEID_EXPR:
case AT_ENCODE_EXPR:
{
tree op = TREE_OPERAND (*tp, 0);
if (code == SIZEOF_EXPR && SIZEOF_EXPR_TYPE_P (*tp))
op = TREE_TYPE (op);
if (TYPE_P (op))
{
if (dependent_type_p (op))
return *tp;
else
{
*walk_subtrees = false;
return NULL_TREE;
}
}
break;
}
case COMPONENT_REF:
if (identifier_p (TREE_OPERAND (*tp, 1)))
return *tp;
break;
case SCOPE_REF:
if (instantiation_dependent_scope_ref_p (*tp))
return *tp;
else
break;
case BIND_EXPR:
return *tp;
case REQUIRES_EXPR:
return *tp;
case CALL_EXPR:
if (function_concept_check_p (*tp))
return *tp;
break;
case TEMPLATE_ID_EXPR:
if (variable_concept_p (TREE_OPERAND (*tp, 0)))
return *tp;
break;
default:
break;
}
if (type_dependent_expression_p (*tp))
return *tp;
else
return NULL_TREE;
}
bool
instantiation_dependent_uneval_expression_p (tree expression)
{
tree result;
if (!processing_template_decl)
return false;
if (expression == error_mark_node)
return false;
result = cp_walk_tree_without_duplicates (&expression,
instantiation_dependent_r, NULL);
return result != NULL_TREE;
}
bool
instantiation_dependent_expression_p (tree expression)
{
return (instantiation_dependent_uneval_expression_p (expression)
|| value_dependent_expression_p (expression));
}
bool
type_dependent_expression_p_push (tree expr)
{
bool b;
++processing_template_decl;
b = type_dependent_expression_p (expr);
--processing_template_decl;
return b;
}
bool
any_type_dependent_arguments_p (const vec<tree, va_gc> *args)
{
unsigned int i;
tree arg;
FOR_EACH_VEC_SAFE_ELT (args, i, arg)
{
if (type_dependent_expression_p (arg))
return true;
}
return false;
}
bool
any_type_dependent_elements_p (const_tree list)
{
for (; list; list = TREE_CHAIN (list))
if (type_dependent_expression_p (TREE_VALUE (list)))
return true;
return false;
}
bool
any_value_dependent_elements_p (const_tree list)
{
for (; list; list = TREE_CHAIN (list))
if (value_dependent_expression_p (TREE_VALUE (list)))
return true;
return false;
}
bool
dependent_template_arg_p (tree arg)
{
if (!processing_template_decl)
return false;
if (!arg || arg == error_mark_node)
return true;
if (TREE_CODE (arg) == ARGUMENT_PACK_SELECT)
arg = argument_pack_select_arg (arg);
if (TREE_CODE (arg) == TEMPLATE_TEMPLATE_PARM)
return true;
if (TREE_CODE (arg) == TEMPLATE_DECL)
{
if (DECL_TEMPLATE_PARM_P (arg))
return true;
tree scope = CP_DECL_CONTEXT (arg);
return TYPE_P (scope) && dependent_type_p (scope);
}
else if (ARGUMENT_PACK_P (arg))
{
tree args = ARGUMENT_PACK_ARGS (arg);
int i, len = TREE_VEC_LENGTH (args);
for (i = 0; i < len; ++i)
{
if (dependent_template_arg_p (TREE_VEC_ELT (args, i)))
return true;
}
return false;
}
else if (TYPE_P (arg))
return dependent_type_p (arg);
else
return (type_dependent_expression_p (arg)
|| value_dependent_expression_p (arg));
}
bool
any_template_arguments_need_structural_equality_p (tree args)
{
int i;
int j;
if (!args)
return false;
if (args == error_mark_node)
return true;
for (i = 0; i < TMPL_ARGS_DEPTH (args); ++i)
{
tree level = TMPL_ARGS_LEVEL (args, i + 1);
for (j = 0; j < TREE_VEC_LENGTH (level); ++j)
{
tree arg = TREE_VEC_ELT (level, j);
tree packed_args = NULL_TREE;
int k, len = 1;
if (ARGUMENT_PACK_P (arg))
{
packed_args = ARGUMENT_PACK_ARGS (arg);
len = TREE_VEC_LENGTH (packed_args);
}
for (k = 0; k < len; ++k)
{
if (packed_args)
arg = TREE_VEC_ELT (packed_args, k);
if (error_operand_p (arg))
return true;
else if (TREE_CODE (arg) == TEMPLATE_DECL)
continue;
else if (TYPE_P (arg) && TYPE_STRUCTURAL_EQUALITY_P (arg))
return true;
else if (!TYPE_P (arg) && TREE_TYPE (arg)
&& TYPE_STRUCTURAL_EQUALITY_P (TREE_TYPE (arg)))
return true;
}
}
}
return false;
}
bool
any_dependent_template_arguments_p (const_tree args)
{
int i;
int j;
if (!args)
return false;
if (args == error_mark_node)
return true;
for (i = 0; i < TMPL_ARGS_DEPTH (args); ++i)
{
const_tree level = TMPL_ARGS_LEVEL (args, i + 1);
for (j = 0; j < TREE_VEC_LENGTH (level); ++j)
if (dependent_template_arg_p (TREE_VEC_ELT (level, j)))
return true;
}
return false;
}
bool
any_erroneous_template_args_p (const_tree args)
{
int i;
int j;
if (args == error_mark_node)
return true;
if (args && TREE_CODE (args) != TREE_VEC)
{
if (tree ti = get_template_info (args))
args = TI_ARGS (ti);
else
args = NULL_TREE;
}
if (!args)
return false;
for (i = 0; i < TMPL_ARGS_DEPTH (args); ++i)
{
const_tree level = TMPL_ARGS_LEVEL (args, i + 1);
for (j = 0; j < TREE_VEC_LENGTH (level); ++j)
if (error_operand_p (TREE_VEC_ELT (level, j)))
return true;
}
return false;
}
bool
dependent_template_p (tree tmpl)
{
if (TREE_CODE (tmpl) == OVERLOAD)
{
for (lkp_iterator iter (tmpl); iter; ++iter)
if (dependent_template_p (*iter))
return true;
return false;
}
if (DECL_TEMPLATE_TEMPLATE_PARM_P (tmpl)
|| TREE_CODE (tmpl) == TEMPLATE_TEMPLATE_PARM)
return true;
if (TREE_CODE (tmpl) == SCOPE_REF || identifier_p (tmpl))
return true;
return false;
}
bool
dependent_template_id_p (tree tmpl, tree args)
{
return (dependent_template_p (tmpl)
|| any_dependent_template_arguments_p (args));
}
bool
dependent_omp_for_p (tree declv, tree initv, tree condv, tree incrv)
{
int i;
if (!processing_template_decl)
return false;
for (i = 0; i < TREE_VEC_LENGTH (declv); i++)
{
tree decl = TREE_VEC_ELT (declv, i);
tree init = TREE_VEC_ELT (initv, i);
tree cond = TREE_VEC_ELT (condv, i);
tree incr = TREE_VEC_ELT (incrv, i);
if (type_dependent_expression_p (decl)
|| TREE_CODE (decl) == SCOPE_REF)
return true;
if (init && type_dependent_expression_p (init))
return true;
if (type_dependent_expression_p (cond))
return true;
if (COMPARISON_CLASS_P (cond)
&& (type_dependent_expression_p (TREE_OPERAND (cond, 0))
|| type_dependent_expression_p (TREE_OPERAND (cond, 1))))
return true;
if (TREE_CODE (incr) == MODOP_EXPR)
{
if (type_dependent_expression_p (TREE_OPERAND (incr, 0))
|| type_dependent_expression_p (TREE_OPERAND (incr, 2)))
return true;
}
else if (type_dependent_expression_p (incr))
return true;
else if (TREE_CODE (incr) == MODIFY_EXPR)
{
if (type_dependent_expression_p (TREE_OPERAND (incr, 0)))
return true;
else if (BINARY_CLASS_P (TREE_OPERAND (incr, 1)))
{
tree t = TREE_OPERAND (incr, 1);
if (type_dependent_expression_p (TREE_OPERAND (t, 0))
|| type_dependent_expression_p (TREE_OPERAND (t, 1)))
return true;
}
}
}
return false;
}
tree
resolve_typename_type (tree type, bool only_current_p)
{
tree scope;
tree name;
tree decl;
int quals;
tree pushed_scope;
tree result;
gcc_assert (TREE_CODE (type) == TYPENAME_TYPE);
scope = TYPE_CONTEXT (type);
gcc_checking_assert (uses_template_parms (scope));
name = TYPE_IDENTIFIER (TYPE_MAIN_VARIANT (type));
if (TREE_CODE (scope) == TYPENAME_TYPE)
{
if (TYPENAME_IS_RESOLVING_P (scope))
return type;
else
scope = resolve_typename_type (scope, only_current_p);
}
if (!CLASS_TYPE_P (scope))
return type;
if (typedef_variant_p (type))
return type;
if (same_type_p (scope, CLASSTYPE_PRIMARY_TEMPLATE_TYPE (scope)))
scope = CLASSTYPE_PRIMARY_TEMPLATE_TYPE (scope);
if (!TYPE_FIELDS (scope))
return type;
if (only_current_p && !currently_open_class (scope))
return type;
pushed_scope = push_scope (scope);
decl = lookup_member (scope, name, 0, true,
tf_warning_or_error);
result = NULL_TREE;
tree fullname = TYPENAME_TYPE_FULLNAME (type);
if (!decl)
;
else if (identifier_p (fullname)
&& TREE_CODE (decl) == TYPE_DECL)
{
result = TREE_TYPE (decl);
if (result == error_mark_node)
result = NULL_TREE;
}
else if (TREE_CODE (fullname) == TEMPLATE_ID_EXPR
&& DECL_CLASS_TEMPLATE_P (decl))
{
tree tmpl = TREE_OPERAND (fullname, 0);
if (TREE_CODE (tmpl) == IDENTIFIER_NODE)
{
pedwarn (EXPR_LOC_OR_LOC (fullname, input_location), OPT_Wpedantic,
"keyword %<template%> not allowed in declarator-id");
tmpl = decl;
}
tree args = TREE_OPERAND (fullname, 1);
result = lookup_template_class (tmpl, args, NULL_TREE, NULL_TREE,
true,
tf_error | tf_user);
if (result == error_mark_node)
result = NULL_TREE;
}
if (pushed_scope)
pop_scope (pushed_scope);
if (!result)
return type;
if (TREE_CODE (result) == TYPENAME_TYPE && !TYPENAME_IS_RESOLVING_P (result))
{
TYPENAME_IS_RESOLVING_P (result) = 1;
result = resolve_typename_type (result, only_current_p);
TYPENAME_IS_RESOLVING_P (result) = 0;
}
quals = cp_type_quals (type);
if (quals)
result = cp_build_qualified_type (result, cp_type_quals (result) | quals);
return result;
}
tree
build_non_dependent_expr (tree expr)
{
tree orig_expr = expr;
tree inner_expr;
if (flag_checking > 1
&& cxx_dialect >= cxx11
&& !parsing_nsdmi ()
&& !expanding_concept ())
fold_non_dependent_expr (expr);
STRIP_ANY_LOCATION_WRAPPER (expr);
inner_expr = expr;
if (TREE_CODE (inner_expr) == STMT_EXPR)
inner_expr = stmt_expr_value_expr (inner_expr);
if (TREE_CODE (inner_expr) == ADDR_EXPR)
inner_expr = TREE_OPERAND (inner_expr, 0);
if (TREE_CODE (inner_expr) == COMPONENT_REF)
inner_expr = TREE_OPERAND (inner_expr, 1);
if (is_overloaded_fn (inner_expr)
|| TREE_CODE (inner_expr) == OFFSET_REF)
return orig_expr;
if (VAR_P (expr))
return orig_expr;
if (TREE_CODE (expr) == STRING_CST)
return orig_expr;
if (TREE_CODE (expr) == VOID_CST
|| TREE_CODE (expr) == INTEGER_CST
|| TREE_CODE (expr) == REAL_CST)
return orig_expr;
if (TREE_CODE (expr) == THROW_EXPR)
return orig_expr;
if (BRACE_ENCLOSED_INITIALIZER_P (expr))
return orig_expr;
if (is_dummy_object (expr))
return orig_expr;
if (TREE_CODE (expr) == COND_EXPR)
return build3 (COND_EXPR,
TREE_TYPE (expr),
TREE_OPERAND (expr, 0),
(TREE_OPERAND (expr, 1)
? build_non_dependent_expr (TREE_OPERAND (expr, 1))
: build_non_dependent_expr (TREE_OPERAND (expr, 0))),
build_non_dependent_expr (TREE_OPERAND (expr, 2)));
if (TREE_CODE (expr) == COMPOUND_EXPR
&& !COMPOUND_EXPR_OVERLOADED (expr))
return build2 (COMPOUND_EXPR,
TREE_TYPE (expr),
TREE_OPERAND (expr, 0),
build_non_dependent_expr (TREE_OPERAND (expr, 1)));
gcc_assert (TREE_TYPE (expr) != unknown_type_node);
return build1_loc (EXPR_LOCATION (orig_expr), NON_DEPENDENT_EXPR,
TREE_TYPE (expr), expr);
}
void
make_args_non_dependent (vec<tree, va_gc> *args)
{
unsigned int ix;
tree arg;
FOR_EACH_VEC_SAFE_ELT (args, ix, arg)
{
tree newarg = build_non_dependent_expr (arg);
if (newarg != arg)
(*args)[ix] = newarg;
}
}
static tree
make_auto_1 (tree name, bool set_canonical)
{
tree au = cxx_make_type (TEMPLATE_TYPE_PARM);
TYPE_NAME (au) = build_decl (input_location,
TYPE_DECL, name, au);
TYPE_STUB_DECL (au) = TYPE_NAME (au);
TEMPLATE_TYPE_PARM_INDEX (au) = build_template_parm_index
(0, processing_template_decl + 1, processing_template_decl + 1,
TYPE_NAME (au), NULL_TREE);
if (set_canonical)
TYPE_CANONICAL (au) = canonical_type_parameter (au);
DECL_ARTIFICIAL (TYPE_NAME (au)) = 1;
SET_DECL_TEMPLATE_PARM_P (TYPE_NAME (au));
return au;
}
tree
make_decltype_auto (void)
{
return make_auto_1 (decltype_auto_identifier, true);
}
tree
make_auto (void)
{
return make_auto_1 (auto_identifier, true);
}
tree
make_template_placeholder (tree tmpl)
{
tree t = make_auto_1 (DECL_NAME (tmpl), true);
CLASS_PLACEHOLDER_TEMPLATE (t) = tmpl;
return t;
}
bool
template_placeholder_p (tree t)
{
return is_auto (t) && CLASS_PLACEHOLDER_TEMPLATE (t);
}
tree
make_constrained_auto (tree con, tree args)
{
tree type = make_auto_1 (auto_identifier, false);
tree tmpl = DECL_TI_TEMPLATE (con);
tree expr = VAR_P (con) ? tmpl : ovl_make (tmpl);
expr = build_concept_check (expr, type, args);
tree constr = normalize_expression (expr);
PLACEHOLDER_TYPE_CONSTRAINTS (type) = constr;
TYPE_CANONICAL (type) = canonical_type_parameter (type);
tree decl = TYPE_NAME (type);
return decl;
}
static tree
listify (tree arg)
{
tree std_init_list = get_namespace_binding (std_node, init_list_identifier);
if (!std_init_list || !DECL_CLASS_TEMPLATE_P (std_init_list))
{    
gcc_rich_location richloc (input_location);
maybe_add_include_fixit (&richloc, "<initializer_list>");
error_at (&richloc,
"deducing from brace-enclosed initializer list"
" requires %<#include <initializer_list>%>");
return error_mark_node;
}
tree argvec = make_tree_vec (1);
TREE_VEC_ELT (argvec, 0) = arg;
return lookup_template_class (std_init_list, argvec, NULL_TREE,
NULL_TREE, 0, tf_warning_or_error);
}
static tree
listify_autos (tree type, tree auto_node)
{
tree init_auto = listify (auto_node);
tree argvec = make_tree_vec (1);
TREE_VEC_ELT (argvec, 0) = init_auto;
if (processing_template_decl)
argvec = add_to_template_args (current_template_args (), argvec);
return tsubst (type, argvec, tf_warning_or_error, NULL_TREE);
}
struct auto_hash : default_hash_traits<tree>
{
static inline hashval_t hash (tree);
static inline bool equal (tree, tree);
};
inline hashval_t
auto_hash::hash (tree t)
{
if (tree c = PLACEHOLDER_TYPE_CONSTRAINTS (t))
return hash_placeholder_constraint (c);
else
return iterative_hash_object (t, 0);
}
inline bool
auto_hash::equal (tree t1, tree t2)
{
if (t1 == t2)
return true;
tree c1 = PLACEHOLDER_TYPE_CONSTRAINTS (t1);
tree c2 = PLACEHOLDER_TYPE_CONSTRAINTS (t2);
if (!c1 || !c2)
return false;
return equivalent_placeholder_constraints (c1, c2);
}
static int
extract_autos_r (tree t, void *data)
{
hash_table<auto_hash> &hash = *(hash_table<auto_hash>*)data;
if (is_auto (t))
{
tree *p = hash.find_slot (t, INSERT);
unsigned idx;
if (*p)
idx = TEMPLATE_PARM_IDX (TEMPLATE_TYPE_PARM_INDEX (*p));
else
{
*p = t;
idx = hash.elements () - 1;
}
TEMPLATE_PARM_IDX (TEMPLATE_TYPE_PARM_INDEX (t)) = idx;
}
return 0;
}
static tree
extract_autos (tree type)
{
hash_set<tree> visited;
hash_table<auto_hash> hash (2);
for_each_template_parm (type, extract_autos_r, &hash, &visited, true);
tree tree_vec = make_tree_vec (hash.elements());
for (hash_table<auto_hash>::iterator iter = hash.begin();
iter != hash.end(); ++iter)
{
tree elt = *iter;
unsigned i = TEMPLATE_PARM_IDX (TEMPLATE_TYPE_PARM_INDEX (elt));
TREE_VEC_ELT (tree_vec, i)
= build_tree_list (NULL_TREE, TYPE_NAME (elt));
}
return tree_vec;
}
const char *const dguide_base = "__dguide_";
tree
dguide_name (tree tmpl)
{
tree type = (TYPE_P (tmpl) ? tmpl : TREE_TYPE (tmpl));
tree tname = TYPE_IDENTIFIER (type);
char *buf = (char *) alloca (1 + strlen (dguide_base)
+ IDENTIFIER_LENGTH (tname));
memcpy (buf, dguide_base, strlen (dguide_base));
memcpy (buf + strlen (dguide_base), IDENTIFIER_POINTER (tname),
IDENTIFIER_LENGTH (tname) + 1);
tree dname = get_identifier (buf);
TREE_TYPE (dname) = type;
return dname;
}
bool
dguide_name_p (tree name)
{
return (TREE_CODE (name) == IDENTIFIER_NODE
&& TREE_TYPE (name)
&& !strncmp (IDENTIFIER_POINTER (name), dguide_base,
strlen (dguide_base)));
}
bool
deduction_guide_p (const_tree fn)
{
if (DECL_P (fn))
if (tree name = DECL_NAME (fn))
return dguide_name_p (name);
return false;
}
bool
copy_guide_p (const_tree fn)
{
gcc_assert (deduction_guide_p (fn));
if (!DECL_ARTIFICIAL (fn))
return false;
tree parms = FUNCTION_FIRST_USER_PARMTYPE (DECL_TI_TEMPLATE (fn));
return (TREE_CHAIN (parms) == void_list_node
&& same_type_p (TREE_VALUE (parms), TREE_TYPE (DECL_NAME (fn))));
}
bool
template_guide_p (const_tree fn)
{
gcc_assert (deduction_guide_p (fn));
if (!DECL_ARTIFICIAL (fn))
return false;
tree tmpl = DECL_TI_TEMPLATE (fn);
if (tree org = DECL_ABSTRACT_ORIGIN (tmpl))
return PRIMARY_TEMPLATE_P (org);
return false;
}
static tree
rewrite_template_parm (tree olddecl, unsigned index, unsigned level,
tree tsubst_args, tsubst_flags_t complain)
{
if (olddecl == error_mark_node)
return error_mark_node;
tree oldidx = get_template_parm_index (olddecl);
tree newtype;
if (TREE_CODE (olddecl) == TYPE_DECL
|| TREE_CODE (olddecl) == TEMPLATE_DECL)
{
tree oldtype = TREE_TYPE (olddecl);
newtype = cxx_make_type (TREE_CODE (oldtype));
TYPE_MAIN_VARIANT (newtype) = newtype;
if (TREE_CODE (oldtype) == TEMPLATE_TYPE_PARM)
TEMPLATE_TYPE_PARM_FOR_CLASS (newtype)
= TEMPLATE_TYPE_PARM_FOR_CLASS (oldtype);
}
else
{
newtype = TREE_TYPE (olddecl);
if (type_uses_auto (newtype))
{
newtype = tsubst (newtype, tsubst_args,
complain|tf_partial, NULL_TREE);
newtype = tsubst (newtype, current_template_args (),
complain, NULL_TREE);
}
else
newtype = tsubst (newtype, tsubst_args,
complain, NULL_TREE);
}
tree newdecl
= build_decl (DECL_SOURCE_LOCATION (olddecl), TREE_CODE (olddecl),
DECL_NAME (olddecl), newtype);
SET_DECL_TEMPLATE_PARM_P (newdecl);
tree newidx;
if (TREE_CODE (olddecl) == TYPE_DECL
|| TREE_CODE (olddecl) == TEMPLATE_DECL)
{
newidx = TEMPLATE_TYPE_PARM_INDEX (newtype)
= build_template_parm_index (index, level, level,
newdecl, newtype);
TEMPLATE_PARM_PARAMETER_PACK (newidx)
= TEMPLATE_PARM_PARAMETER_PACK (oldidx);
TYPE_STUB_DECL (newtype) = TYPE_NAME (newtype) = newdecl;
TYPE_CANONICAL (newtype) = canonical_type_parameter (newtype);
if (TREE_CODE (olddecl) == TEMPLATE_DECL)
{
DECL_TEMPLATE_RESULT (newdecl)
= build_decl (DECL_SOURCE_LOCATION (olddecl), TYPE_DECL,
DECL_NAME (olddecl), newtype);
DECL_ARTIFICIAL (DECL_TEMPLATE_RESULT (newdecl)) = true;
tree ttparms = (INNERMOST_TEMPLATE_PARMS
(DECL_TEMPLATE_PARMS (olddecl)));
const int depth = TMPL_ARGS_DEPTH (tsubst_args);
tree ttargs = make_tree_vec (depth + 1);
for (int i = 0; i < depth; ++i)
TREE_VEC_ELT (ttargs, i) = TREE_VEC_ELT (tsubst_args, i);
TREE_VEC_ELT (ttargs, depth)
= template_parms_level_to_args (ttparms);
ttparms = tsubst_template_parms_level (ttparms, ttargs,
complain|tf_partial);
ttargs = current_template_args ();
ttparms = tsubst_template_parms_level (ttparms, ttargs,
complain);
ttparms = tree_cons (size_int (depth), ttparms,
current_template_parms);
DECL_TEMPLATE_PARMS (newdecl) = ttparms;
}
}
else
{
tree oldconst = TEMPLATE_PARM_DECL (oldidx);
tree newconst
= build_decl (DECL_SOURCE_LOCATION (oldconst),
TREE_CODE (oldconst),
DECL_NAME (oldconst), newtype);
TREE_CONSTANT (newconst) = TREE_CONSTANT (newdecl)
= TREE_READONLY (newconst) = TREE_READONLY (newdecl) = true;
SET_DECL_TEMPLATE_PARM_P (newconst);
newidx = build_template_parm_index (index, level, level,
newconst, newtype);
TEMPLATE_PARM_PARAMETER_PACK (newidx)
= TEMPLATE_PARM_PARAMETER_PACK (oldidx);
DECL_INITIAL (newdecl) = DECL_INITIAL (newconst) = newidx;
}
return newdecl;
}
static tree
build_deduction_guide (tree ctor, tree outer_args, tsubst_flags_t complain)
{
tree type, tparms, targs, fparms, fargs, ci;
bool memtmpl = false;
bool explicit_p;
location_t loc;
tree fn_tmpl = NULL_TREE;
if (TYPE_P (ctor))
{
type = ctor;
bool copy_p = TREE_CODE (type) == REFERENCE_TYPE;
if (copy_p)
{
type = TREE_TYPE (type);
fparms = tree_cons (NULL_TREE, type, void_list_node);
}
else
fparms = void_list_node;
tree ctmpl = CLASSTYPE_TI_TEMPLATE (type);
tparms = DECL_TEMPLATE_PARMS (ctmpl);
targs = CLASSTYPE_TI_ARGS (type);
ci = NULL_TREE;
fargs = NULL_TREE;
loc = DECL_SOURCE_LOCATION (ctmpl);
explicit_p = false;
}
else
{
++processing_template_decl;
bool ok = true;
fn_tmpl
= (TREE_CODE (ctor) == TEMPLATE_DECL ? ctor
: DECL_TI_TEMPLATE (ctor));
if (outer_args)
fn_tmpl = tsubst (fn_tmpl, outer_args, complain, ctor);
ctor = DECL_TEMPLATE_RESULT (fn_tmpl);
type = DECL_CONTEXT (ctor);
tparms = DECL_TEMPLATE_PARMS (fn_tmpl);
targs = get_innermost_template_args (DECL_TI_ARGS (ctor),
TMPL_PARMS_DEPTH (tparms));
fparms = FUNCTION_ARG_CHAIN (ctor);
fargs = TREE_CHAIN (DECL_ARGUMENTS (ctor));
ci = get_constraints (ctor);
loc = DECL_SOURCE_LOCATION (ctor);
explicit_p = DECL_NONCONVERTING_P (ctor);
if (PRIMARY_TEMPLATE_P (fn_tmpl))
{
memtmpl = true;
tree save_parms = current_template_parms;
const int depth = 2;
gcc_assert (TMPL_ARGS_DEPTH (targs) == depth);
tree tsubst_args = copy_node (targs);
TMPL_ARGS_LEVEL (tsubst_args, depth)
= copy_node (TMPL_ARGS_LEVEL (tsubst_args, depth));
tree ftparms = TREE_VALUE (tparms);
unsigned flen = TREE_VEC_LENGTH (ftparms);
tparms = TREE_CHAIN (tparms);
tree ctparms = TREE_VALUE (tparms);
unsigned clen = TREE_VEC_LENGTH (ctparms);
current_template_parms = tparms = copy_node (tparms);
tree new_vec = TREE_VALUE (tparms) = make_tree_vec (flen + clen);
for (unsigned i = 0; i < clen; ++i)
TREE_VEC_ELT (new_vec, i) = TREE_VEC_ELT (ctparms, i);
for (unsigned i = 0; i < flen; ++i)
{
unsigned index = i + clen;
unsigned level = 1;
tree oldelt = TREE_VEC_ELT (ftparms, i);
tree olddecl = TREE_VALUE (oldelt);
tree newdecl = rewrite_template_parm (olddecl, index, level,
tsubst_args, complain);
if (newdecl == error_mark_node)
ok = false;
tree newdef = tsubst_template_arg (TREE_PURPOSE (oldelt),
tsubst_args, complain, ctor);
tree list = build_tree_list (newdef, newdecl);
TEMPLATE_PARM_CONSTRAINTS (list)
= tsubst_constraint_info (TEMPLATE_PARM_CONSTRAINTS (oldelt),
tsubst_args, complain, ctor);
TREE_VEC_ELT (new_vec, index) = list;
TMPL_ARG (tsubst_args, depth, i) = template_parm_to_arg (list);
}
targs = template_parms_to_args (tparms);
fparms = tsubst_arg_types (fparms, tsubst_args, NULL_TREE,
complain, ctor);
if (fparms == error_mark_node)
ok = false;
fargs = tsubst (fargs, tsubst_args, complain, ctor);
if (ci)
ci = tsubst_constraint_info (ci, tsubst_args, complain, ctor);
current_template_parms = save_parms;
}
--processing_template_decl;
if (!ok)
return error_mark_node;
}
if (!memtmpl)
{
tparms = copy_node (tparms);
INNERMOST_TEMPLATE_PARMS (tparms)
= copy_node (INNERMOST_TEMPLATE_PARMS (tparms));
}
tree fntype = build_function_type (type, fparms);
tree ded_fn = build_lang_decl_loc (loc,
FUNCTION_DECL,
dguide_name (type), fntype);
DECL_ARGUMENTS (ded_fn) = fargs;
DECL_ARTIFICIAL (ded_fn) = true;
DECL_NONCONVERTING_P (ded_fn) = explicit_p;
tree ded_tmpl = build_template_decl (ded_fn, tparms, false);
DECL_ARTIFICIAL (ded_tmpl) = true;
DECL_TEMPLATE_RESULT (ded_tmpl) = ded_fn;
TREE_TYPE (ded_tmpl) = TREE_TYPE (ded_fn);
DECL_TEMPLATE_INFO (ded_fn) = build_template_info (ded_tmpl, targs);
DECL_PRIMARY_TEMPLATE (ded_tmpl) = ded_tmpl;
if (DECL_P (ctor))
DECL_ABSTRACT_ORIGIN (ded_tmpl) = fn_tmpl;
if (ci)
set_constraints (ded_tmpl, ci);
return ded_tmpl;
}
static tree
do_class_deduction (tree ptype, tree tmpl, tree init, int flags,
tsubst_flags_t complain)
{
if (!DECL_CLASS_TEMPLATE_P (tmpl))
{
if (DECL_TEMPLATE_TEMPLATE_PARM_P (tmpl))
return ptype;
if (complain & tf_error)
error ("non-class template %qT used without template arguments", tmpl);
return error_mark_node;
}
tree type = TREE_TYPE (tmpl);
bool try_list_ctor = false;
vec<tree,va_gc> *args;
if (init == NULL_TREE
|| TREE_CODE (init) == TREE_LIST)
args = make_tree_vector_from_list (init);
else if (BRACE_ENCLOSED_INITIALIZER_P (init))
{
try_list_ctor = TYPE_HAS_LIST_CTOR (type);
if (try_list_ctor && CONSTRUCTOR_NELTS (init) == 1)
{
tree elt = CONSTRUCTOR_ELT (init, 0)->value;
tree etype = TREE_TYPE (elt);
tree tparms = INNERMOST_TEMPLATE_PARMS (DECL_TEMPLATE_PARMS (tmpl));
tree targs = make_tree_vec (TREE_VEC_LENGTH (tparms));
int err = unify (tparms, targs, type, etype,
UNIFY_ALLOW_DERIVED, false);
if (err == 0)
try_list_ctor = false;
ggc_free (targs);
}
if (try_list_ctor || is_std_init_list (type))
args = make_tree_vector_single (init);
else
args = make_tree_vector_from_ctor (init);
}
else
args = make_tree_vector_single (init);
tree dname = dguide_name (tmpl);
tree cands = lookup_qualified_name (CP_DECL_CONTEXT (tmpl), dname,
false, false,
false);
bool elided = false;
if (cands == error_mark_node)
cands = NULL_TREE;
if (flags & LOOKUP_ONLYCONVERTING)
{
for (lkp_iterator iter (cands); !elided && iter; ++iter)
if (DECL_NONCONVERTING_P (STRIP_TEMPLATE (*iter)))
elided = true;
if (elided)
{
tree pruned = NULL_TREE;
for (lkp_iterator iter (cands); iter; ++iter)
if (!DECL_NONCONVERTING_P (STRIP_TEMPLATE (*iter)))
pruned = lookup_add (*iter, pruned);
cands = pruned;
}
}
tree outer_args = NULL_TREE;
if (DECL_CLASS_SCOPE_P (tmpl)
&& CLASSTYPE_TEMPLATE_INFO (DECL_CONTEXT (tmpl)))
{
outer_args = CLASSTYPE_TI_ARGS (DECL_CONTEXT (tmpl));
type = TREE_TYPE (most_general_template (tmpl));
}
bool saw_ctor = false;
for (ovl_iterator iter (CLASSTYPE_CONSTRUCTORS (type)); iter; ++iter)
{
if (iter.using_p ())
continue;
tree guide = build_deduction_guide (*iter, outer_args, complain);
if (guide == error_mark_node)
return error_mark_node;
if ((flags & LOOKUP_ONLYCONVERTING)
&& DECL_NONCONVERTING_P (STRIP_TEMPLATE (guide)))
elided = true;
else
cands = lookup_add (guide, cands);
saw_ctor = true;
}
tree call = error_mark_node;
tree list_cands = NULL_TREE;
if (try_list_ctor && cands)
for (lkp_iterator iter (cands); iter; ++iter)
{
tree dg = *iter;
if (is_list_ctor (dg))
list_cands = lookup_add (dg, list_cands);
}
if (list_cands)
{
++cp_unevaluated_operand;
call = build_new_function_call (list_cands, &args, tf_decltype);
--cp_unevaluated_operand;
if (call == error_mark_node)
{
release_tree_vector (args);
args = make_tree_vector_from_ctor (init);
}
}
if (call == error_mark_node && args->length () < 2)
{
tree gtype = NULL_TREE;
if (args->length () == 1)
gtype = build_reference_type (type);
else if (!saw_ctor)
gtype = type;
if (gtype)
{
tree guide = build_deduction_guide (gtype, outer_args, complain);
if (guide == error_mark_node)
return error_mark_node;
cands = lookup_add (guide, cands);
}
}
if (elided && !cands)
{
error ("cannot deduce template arguments for copy-initialization"
" of %qT, as it has no non-explicit deduction guides or "
"user-declared constructors", type);
return error_mark_node;
}
else if (!cands && call == error_mark_node)
{
error ("cannot deduce template arguments of %qT, as it has no viable "
"deduction guides", type);
return error_mark_node;
}
if (call == error_mark_node)
{
++cp_unevaluated_operand;
call = build_new_function_call (cands, &args, tf_decltype);
--cp_unevaluated_operand;
}
if (call == error_mark_node && (complain & tf_warning_or_error))
{
error ("class template argument deduction failed:");
++cp_unevaluated_operand;
call = build_new_function_call (cands, &args, complain | tf_decltype);
--cp_unevaluated_operand;
if (elided)
inform (input_location, "explicit deduction guides not considered "
"for copy-initialization");
}
release_tree_vector (args);
return cp_build_qualified_type (TREE_TYPE (call), cp_type_quals (ptype));
}
tree
do_auto_deduction (tree type, tree init, tree auto_node,
tsubst_flags_t complain, auto_deduction_context context,
tree outer_targs, int flags)
{
tree targs;
if (init == error_mark_node)
return error_mark_node;
if (init && type_dependent_expression_p (init)
&& context != adc_unify)
return type;
if (init && undeduced_auto_decl (init))
return type;
if (tree tmpl = CLASS_PLACEHOLDER_TEMPLATE (auto_node))
return do_class_deduction (type, tmpl, init, flags, complain);
if (init == NULL_TREE || TREE_TYPE (init) == NULL_TREE)
return type;
if (BRACE_ENCLOSED_INITIALIZER_P (init))
{
if (!DIRECT_LIST_INIT_P (init))
type = listify_autos (type, auto_node);
else if (CONSTRUCTOR_NELTS (init) == 1)
init = CONSTRUCTOR_ELT (init, 0)->value;
else
{
if (complain & tf_warning_or_error)
{
if (permerror (input_location, "direct-list-initialization of "
"%<auto%> requires exactly one element"))
inform (input_location,
"for deduction to %<std::initializer_list%>, use copy-"
"list-initialization (i.e. add %<=%> before the %<{%>)");
}
type = listify_autos (type, auto_node);
}
}
if (type == error_mark_node)
return error_mark_node;
init = resolve_nondeduced_context (init, complain);
if (context == adc_decomp_type
&& auto_node == type
&& init != error_mark_node
&& TREE_CODE (TREE_TYPE (init)) == ARRAY_TYPE)
return cp_build_qualified_type_real (TREE_TYPE (init), TYPE_QUALS (type),
complain);
else if (AUTO_IS_DECLTYPE (auto_node))
{
bool id = (DECL_P (init)
|| ((TREE_CODE (init) == COMPONENT_REF
|| TREE_CODE (init) == SCOPE_REF)
&& !REF_PARENTHESIZED_P (init)));
targs = make_tree_vec (1);
TREE_VEC_ELT (targs, 0)
= finish_decltype_type (init, id, tf_warning_or_error);
if (type != auto_node)
{
if (complain & tf_error)
error ("%qT as type rather than plain %<decltype(auto)%>", type);
return error_mark_node;
}
}
else
{
tree parms = build_tree_list (NULL_TREE, type);
tree tparms;
if (flag_concepts)
tparms = extract_autos (type);
else
{
tparms = make_tree_vec (1);
TREE_VEC_ELT (tparms, 0)
= build_tree_list (NULL_TREE, TYPE_NAME (auto_node));
}
targs = make_tree_vec (TREE_VEC_LENGTH (tparms));
int val = type_unification_real (tparms, targs, parms, &init, 1, 0,
DEDUCE_CALL, LOOKUP_NORMAL,
NULL, false);
if (val > 0)
{
if (processing_template_decl)
return type;
if (type && type != error_mark_node
&& (complain & tf_error))
{
if (cfun && auto_node == current_function_auto_return_pattern
&& LAMBDA_FUNCTION_P (current_function_decl))
error ("unable to deduce lambda return type from %qE", init);
else
error ("unable to deduce %qT from %qE", type, init);
type_unification_real (tparms, targs, parms, &init, 1, 0,
DEDUCE_CALL, LOOKUP_NORMAL,
NULL, true);
}
return error_mark_node;
}
}
if (flag_concepts && !processing_template_decl)
if (tree constr = PLACEHOLDER_TYPE_CONSTRAINTS (auto_node))
{
gcc_assert (TREE_CODE (constr) == CHECK_CONSTR);
tree cargs = CHECK_CONSTR_ARGS (constr);
if (TREE_VEC_LENGTH (cargs) > 1)
{
cargs = copy_node (cargs);
TREE_VEC_ELT (cargs, 0) = TREE_VEC_ELT (targs, 0);
}
else
cargs = targs;
if (!constraints_satisfied_p (constr, cargs))
{
if (complain & tf_warning_or_error)
{
switch (context)
{
case adc_unspecified:
case adc_unify:
error("placeholder constraints not satisfied");
break;
case adc_variable_type:
case adc_decomp_type:
error ("deduced initializer does not satisfy "
"placeholder constraints");
break;
case adc_return_type:
error ("deduced return type does not satisfy "
"placeholder constraints");
break;
case adc_requirement:
error ("deduced expression type does not satisfy "
"placeholder constraints");
break;
}
diagnose_constraints (input_location, constr, targs);
}
return error_mark_node;
}
}
if (processing_template_decl && context != adc_unify)
outer_targs = current_template_args ();
targs = add_to_template_args (outer_targs, targs);
return tsubst (type, targs, complain, NULL_TREE);
}
tree
splice_late_return_type (tree type, tree late_return_type)
{
if (is_auto (type))
{
if (late_return_type)
return late_return_type;
tree idx = get_template_parm_index (type);
if (TEMPLATE_PARM_LEVEL (idx) <= processing_template_decl)
return make_auto_1 (TYPE_IDENTIFIER (type), true);
}
return type;
}
bool
is_auto (const_tree type)
{
if (TREE_CODE (type) == TEMPLATE_TYPE_PARM
&& (TYPE_IDENTIFIER (type) == auto_identifier
|| TYPE_IDENTIFIER (type) == decltype_auto_identifier
|| CLASS_PLACEHOLDER_TEMPLATE (type)))
return true;
else
return false;
}
int
is_auto_r (tree tp, void *)
{
return is_auto (tp);
}
tree
type_uses_auto (tree type)
{
if (type == NULL_TREE)
return NULL_TREE;
else if (flag_concepts)
{
if (uses_template_parms (type))
return for_each_template_parm (type, is_auto_r, NULL,
NULL, false);
else
return NULL_TREE;
}
else
return find_type_usage (type, is_auto);
}
bool
check_auto_in_tmpl_args (tree tmpl, tree args)
{
if (!args || TREE_CODE (args) != TREE_VEC)
return false;
if (flag_concepts
&& (identifier_p (tmpl)
|| (DECL_P (tmpl)
&&  (DECL_TYPE_TEMPLATE_P (tmpl)
|| DECL_TEMPLATE_TEMPLATE_PARM_P (tmpl)))))
return false;
if (!type_uses_auto (args))
return false;
bool errors = false;
tree vec = extract_autos (args);
for (int i = 0; i < TREE_VEC_LENGTH (vec); i++)
{
tree xauto = TREE_VALUE (TREE_VEC_ELT (vec, i));
error_at (DECL_SOURCE_LOCATION (xauto),
"invalid use of %qT in template argument", xauto);
errors = true;
}
return errors;
}
vec<qualified_typedef_usage_t, va_gc> *
get_types_needing_access_check (tree t)
{
tree ti;
vec<qualified_typedef_usage_t, va_gc> *result = NULL;
if (!t || t == error_mark_node)
return NULL;
if (!(ti = get_template_info (t)))
return NULL;
if (CLASS_TYPE_P (t)
|| TREE_CODE (t) == FUNCTION_DECL)
{
if (!TI_TEMPLATE (ti))
return NULL;
result = TI_TYPEDEFS_NEEDING_ACCESS_CHECKING (ti);
}
return result;
}
static void
append_type_to_template_for_access_check_1 (tree t,
tree type_decl,
tree scope,
location_t location)
{
qualified_typedef_usage_t typedef_usage;
tree ti;
if (!t || t == error_mark_node)
return;
gcc_assert ((TREE_CODE (t) == FUNCTION_DECL
|| CLASS_TYPE_P (t))
&& type_decl
&& TREE_CODE (type_decl) == TYPE_DECL
&& scope);
if (!(ti = get_template_info (t)))
return;
gcc_assert (TI_TEMPLATE (ti));
typedef_usage.typedef_decl = type_decl;
typedef_usage.context = scope;
typedef_usage.locus = location;
vec_safe_push (TI_TYPEDEFS_NEEDING_ACCESS_CHECKING (ti), typedef_usage);
}
void
append_type_to_template_for_access_check (tree templ,
tree type_decl,
tree scope,
location_t location)
{
qualified_typedef_usage_t *iter;
unsigned i;
gcc_assert (type_decl && (TREE_CODE (type_decl) == TYPE_DECL));
FOR_EACH_VEC_SAFE_ELT (get_types_needing_access_check (templ), i, iter)
if (iter->typedef_decl == type_decl && scope == iter->context)
return;
append_type_to_template_for_access_check_1 (templ, type_decl,
scope, location);
}
tree
convert_generic_types_to_packs (tree parm, int start_idx, int end_idx)
{
tree current = current_template_parms;
int depth = TMPL_PARMS_DEPTH (current);
current = INNERMOST_TEMPLATE_PARMS (current);
tree replacement = make_tree_vec (TREE_VEC_LENGTH (current));
for (int i = 0; i < start_idx; ++i)
TREE_VEC_ELT (replacement, i)
= TREE_TYPE (TREE_VALUE (TREE_VEC_ELT (current, i)));
for (int i = start_idx; i < end_idx; ++i)
{
tree o = TREE_TYPE (TREE_VALUE
(TREE_VEC_ELT (current, i)));
tree t = copy_type (o);
TEMPLATE_TYPE_PARM_INDEX (t)
= reduce_template_parm_level (TEMPLATE_TYPE_PARM_INDEX (o),
o, 0, 0, tf_none);
TREE_TYPE (TEMPLATE_TYPE_DECL (t)) = t;
TYPE_STUB_DECL (t) = TYPE_NAME (t) = TEMPLATE_TYPE_DECL (t);
TYPE_MAIN_VARIANT (t) = t;
TEMPLATE_TYPE_PARAMETER_PACK (t) = true;
TYPE_CANONICAL (t) = canonical_type_parameter (t);
TREE_VEC_ELT (replacement, i) = t;
TREE_VALUE (TREE_VEC_ELT (current, i)) = TREE_CHAIN (t);
}
for (int i = end_idx, e = TREE_VEC_LENGTH (current); i < e; ++i)
TREE_VEC_ELT (replacement, i)
= TREE_TYPE (TREE_VALUE (TREE_VEC_ELT (current, i)));
if (depth > 1)
replacement = add_to_template_args (template_parms_to_args
(TREE_CHAIN (current_template_parms)),
replacement);
return tsubst (parm, replacement, tf_none, NULL_TREE);
}
struct GTY((for_user)) constr_entry
{
tree decl;
tree ci;
};
struct constr_hasher : ggc_ptr_hash<constr_entry>
{
static hashval_t hash (constr_entry *e)
{
return (hashval_t)DECL_UID (e->decl);
}
static bool equal (constr_entry *e1, constr_entry *e2)
{
return e1->decl == e2->decl;
}
};
static GTY (()) hash_table<constr_hasher> *decl_constraints;
tree
get_constraints (tree t)
{
if (!flag_concepts)
return NULL_TREE;
gcc_assert (DECL_P (t));
if (TREE_CODE (t) == TEMPLATE_DECL)
t = DECL_TEMPLATE_RESULT (t);
constr_entry elt = { t, NULL_TREE };
constr_entry* found = decl_constraints->find (&elt);
if (found)
return found->ci;
else
return NULL_TREE;
}
void
set_constraints (tree t, tree ci)
{
if (!ci)
return;
gcc_assert (t && flag_concepts);
if (TREE_CODE (t) == TEMPLATE_DECL)
t = DECL_TEMPLATE_RESULT (t);
gcc_assert (!get_constraints (t));
constr_entry elt = {t, ci};
constr_entry** slot = decl_constraints->find_slot (&elt, INSERT);
constr_entry* entry = ggc_alloc<constr_entry> ();
*entry = elt;
*slot = entry;
}
void
remove_constraints (tree t)
{
gcc_assert (DECL_P (t));
if (TREE_CODE (t) == TEMPLATE_DECL)
t = DECL_TEMPLATE_RESULT (t);
constr_entry elt = {t, NULL_TREE};
constr_entry** slot = decl_constraints->find_slot (&elt, NO_INSERT);
if (slot)
decl_constraints->clear_slot (slot);
}
struct GTY((for_user)) constraint_sat_entry
{
tree ci;
tree args;
tree result;
};
struct constraint_sat_hasher : ggc_ptr_hash<constraint_sat_entry>
{
static hashval_t hash (constraint_sat_entry *e)
{
hashval_t val = iterative_hash_object(e->ci, 0);
return iterative_hash_template_arg (e->args, val);
}
static bool equal (constraint_sat_entry *e1, constraint_sat_entry *e2)
{
return e1->ci == e2->ci && comp_template_args (e1->args, e2->args);
}
};
struct GTY((for_user)) concept_spec_entry
{
tree tmpl;
tree args;
tree result;
};
struct concept_spec_hasher : ggc_ptr_hash<concept_spec_entry>
{
static hashval_t hash (concept_spec_entry *e)
{
return hash_tmpl_and_args (e->tmpl, e->args);
}
static bool equal (concept_spec_entry *e1, concept_spec_entry *e2)
{
++comparing_specializations;
bool eq = e1->tmpl == e2->tmpl && comp_template_args (e1->args, e2->args);
--comparing_specializations;
return eq;
}
};
static GTY (()) hash_table<constraint_sat_hasher> *constraint_memos;
static GTY (()) hash_table<concept_spec_hasher> *concept_memos;
tree
lookup_constraint_satisfaction (tree ci, tree args)
{
constraint_sat_entry elt = { ci, args, NULL_TREE };
constraint_sat_entry* found = constraint_memos->find (&elt);
if (found)
return found->result;
else
return NULL_TREE;
}
tree
memoize_constraint_satisfaction (tree ci, tree args, tree result)
{
constraint_sat_entry elt = {ci, args, result};
constraint_sat_entry** slot = constraint_memos->find_slot (&elt, INSERT);
constraint_sat_entry* entry = ggc_alloc<constraint_sat_entry> ();
*entry = elt;
*slot = entry;
return result;
}
tree
lookup_concept_satisfaction (tree tmpl, tree args)
{
concept_spec_entry elt = { tmpl, args, NULL_TREE };
concept_spec_entry* found = concept_memos->find (&elt);
if (found)
return found->result;
else
return NULL_TREE;
}
tree
memoize_concept_satisfaction (tree tmpl, tree args, tree result)
{
concept_spec_entry elt = {tmpl, args, result};
concept_spec_entry** slot = concept_memos->find_slot (&elt, INSERT);
concept_spec_entry* entry = ggc_alloc<concept_spec_entry> ();
*entry = elt;
*slot = entry;
return result;
}
static GTY (()) hash_table<concept_spec_hasher> *concept_expansions;
tree
get_concept_expansion (tree tmpl, tree args)
{
concept_spec_entry elt = { tmpl, args, NULL_TREE };
concept_spec_entry* found = concept_expansions->find (&elt);
if (found)
return found->result;
else
return NULL_TREE;
}
tree
save_concept_expansion (tree tmpl, tree args, tree def)
{
concept_spec_entry elt = {tmpl, args, def};
concept_spec_entry** slot = concept_expansions->find_slot (&elt, INSERT);
concept_spec_entry* entry = ggc_alloc<concept_spec_entry> ();
*entry = elt;
*slot = entry;
return def;
}
static hashval_t
hash_subsumption_args (tree t1, tree t2)
{
gcc_assert (TREE_CODE (t1) == CHECK_CONSTR);
gcc_assert (TREE_CODE (t2) == CHECK_CONSTR);
int val = 0;
val = iterative_hash_object (CHECK_CONSTR_CONCEPT (t1), val);
val = iterative_hash_template_arg (CHECK_CONSTR_ARGS (t1), val);
val = iterative_hash_object (CHECK_CONSTR_CONCEPT (t2), val);
val = iterative_hash_template_arg (CHECK_CONSTR_ARGS (t2), val);
return val;
}
static bool
comp_subsumption_args (tree left1, tree left2, tree right1, tree right2)
{
if (CHECK_CONSTR_CONCEPT (left1) == CHECK_CONSTR_CONCEPT (right1))
if (CHECK_CONSTR_CONCEPT (left2) == CHECK_CONSTR_CONCEPT (right2))
if (comp_template_args (CHECK_CONSTR_ARGS (left1),
CHECK_CONSTR_ARGS (right1)))
return comp_template_args (CHECK_CONSTR_ARGS (left2),
CHECK_CONSTR_ARGS (right2));
return false;
}
struct GTY((for_user)) subsumption_entry
{
tree t1;
tree t2;
bool result;
};
struct subsumption_hasher : ggc_ptr_hash<subsumption_entry>
{
static hashval_t hash (subsumption_entry *e)
{
return hash_subsumption_args (e->t1, e->t2);
}
static bool equal (subsumption_entry *e1, subsumption_entry *e2)
{
++comparing_specializations;
bool eq = comp_subsumption_args(e1->t1, e1->t2, e2->t1, e2->t2);
--comparing_specializations;
return eq;
}
};
static GTY (()) hash_table<subsumption_hasher> *subsumption_table;
bool*
lookup_subsumption_result (tree t1, tree t2)
{
subsumption_entry elt = { t1, t2, false };
subsumption_entry* found = subsumption_table->find (&elt);
if (found)
return &found->result;
else
return 0;
}
bool
save_subsumption_result (tree t1, tree t2, bool result)
{
subsumption_entry elt = {t1, t2, result};
subsumption_entry** slot = subsumption_table->find_slot (&elt, INSERT);
subsumption_entry* entry = ggc_alloc<subsumption_entry> ();
*entry = elt;
*slot = entry;
return result;
}
void
init_constraint_processing (void)
{
if (!flag_concepts)
return;
decl_constraints = hash_table<constr_hasher>::create_ggc(37);
constraint_memos = hash_table<constraint_sat_hasher>::create_ggc(37);
concept_memos = hash_table<concept_spec_hasher>::create_ggc(37);
concept_expansions = hash_table<concept_spec_hasher>::create_ggc(37);
subsumption_table = hash_table<subsumption_hasher>::create_ggc(37);
}
void
declare_integer_pack (void)
{
tree ipfn = push_library_fn (get_identifier ("__integer_pack"),
build_function_type_list (integer_type_node,
integer_type_node,
NULL_TREE),
NULL_TREE, ECF_CONST);
DECL_DECLARED_CONSTEXPR_P (ipfn) = true;
DECL_BUILT_IN_CLASS (ipfn) = BUILT_IN_FRONTEND;
}
void
init_template_processing (void)
{
decl_specializations = hash_table<spec_hasher>::create_ggc (37);
type_specializations = hash_table<spec_hasher>::create_ggc (37);
if (cxx_dialect >= cxx11)
declare_integer_pack ();
}
void
print_template_statistics (void)
{
fprintf (stderr, "decl_specializations: size %ld, %ld elements, "
"%f collisions\n", (long) decl_specializations->size (),
(long) decl_specializations->elements (),
decl_specializations->collisions ());
fprintf (stderr, "type_specializations: size %ld, %ld elements, "
"%f collisions\n", (long) type_specializations->size (),
(long) type_specializations->elements (),
type_specializations->collisions ());
}
#if CHECKING_P
namespace selftest {
static void
test_build_non_dependent_expr ()
{
location_t loc = BUILTINS_LOCATION;
tree int_cst = build_int_cst (integer_type_node, 42);
ASSERT_EQ (int_cst, build_non_dependent_expr (int_cst));
tree wrapped_int_cst = maybe_wrap_with_location (int_cst, loc);
ASSERT_TRUE (location_wrapper_p (wrapped_int_cst));
ASSERT_EQ (wrapped_int_cst, build_non_dependent_expr (wrapped_int_cst));
tree string_lit = build_string (4, "foo");
TREE_TYPE (string_lit) = char_array_type_node;
string_lit = fix_string_type (string_lit);
ASSERT_EQ (string_lit, build_non_dependent_expr (string_lit));
tree wrapped_string_lit = maybe_wrap_with_location (string_lit, loc);
ASSERT_TRUE (location_wrapper_p (wrapped_string_lit));
ASSERT_EQ (wrapped_string_lit,
build_non_dependent_expr (wrapped_string_lit));
}
static void
test_type_dependent_expression_p ()
{
location_t loc = BUILTINS_LOCATION;
tree name = get_identifier ("foo");
gcc_assert (!processing_template_decl);
ASSERT_FALSE (type_dependent_expression_p (name));
++processing_template_decl;
ASSERT_TRUE (type_dependent_expression_p (name));
ASSERT_FALSE (type_dependent_expression_p (NULL_TREE));
ASSERT_FALSE (type_dependent_expression_p (error_mark_node));
tree using_decl = build_lang_decl (USING_DECL, name, NULL_TREE);
TREE_TYPE (using_decl) = integer_type_node;
ASSERT_TRUE (type_dependent_expression_p (using_decl));
tree wrapped_using_decl = maybe_wrap_with_location (using_decl, loc);
ASSERT_TRUE (location_wrapper_p (wrapped_using_decl));
ASSERT_TRUE (type_dependent_expression_p (wrapped_using_decl));
--processing_template_decl;
}
void
cp_pt_c_tests ()
{
test_build_non_dependent_expr ();
test_type_dependent_expression_p ();
}
} 
#endif 
#include "gt-cp-pt.h"
