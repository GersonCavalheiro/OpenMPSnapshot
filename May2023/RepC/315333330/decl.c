#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "c-family/c-target.h"
#include "cp-tree.h"
#include "timevar.h"
#include "stringpool.h"
#include "cgraph.h"
#include "stor-layout.h"
#include "varasm.h"
#include "attribs.h"
#include "flags.h"
#include "tree-iterator.h"
#include "decl.h"
#include "intl.h"
#include "toplev.h"
#include "c-family/c-objc.h"
#include "c-family/c-pragma.h"
#include "c-family/c-ubsan.h"
#include "debug.h"
#include "plugin.h"
#include "builtins.h"
#include "gimplify.h"
#include "asan.h"
enum bad_spec_place {
BSP_VAR,    
BSP_PARM,   
BSP_TYPE,   
BSP_FIELD   
};
static const char *redeclaration_error_message (tree, tree);
static int decl_jump_unsafe (tree);
static void require_complete_types_for_parms (tree);
static void push_local_name (tree);
static tree grok_reference_init (tree, tree, tree, int);
static tree grokvardecl (tree, tree, tree, const cp_decl_specifier_seq *,
int, int, int, bool, int, tree);
static int check_static_variable_definition (tree, tree);
static void record_unknown_type (tree, const char *);
static tree builtin_function_1 (tree, tree, bool);
static int member_function_or_else (tree, tree, enum overload_flags);
static tree local_variable_p_walkfn (tree *, int *, void *);
static const char *tag_name (enum tag_types);
static tree lookup_and_check_tag (enum tag_types, tree, tag_scope, bool);
static void maybe_deduce_size_from_array_init (tree, tree);
static void layout_var_decl (tree);
static tree check_initializer (tree, tree, int, vec<tree, va_gc> **);
static void make_rtl_for_nonlocal_decl (tree, tree, const char *);
static void save_function_data (tree);
static void copy_type_enum (tree , tree);
static void check_function_type (tree, tree);
static void finish_constructor_body (void);
static void begin_destructor_body (void);
static void finish_destructor_body (void);
static void record_key_method_defined (tree);
static tree create_array_type_for_decl (tree, tree, tree);
static tree get_atexit_node (void);
static tree get_dso_handle_node (void);
static tree start_cleanup_fn (void);
static void end_cleanup_fn (void);
static tree cp_make_fname_decl (location_t, tree, int);
static void initialize_predefined_identifiers (void);
static tree check_special_function_return_type
(special_function_kind, tree, tree, int, const location_t*);
static tree push_cp_library_fn (enum tree_code, tree, int);
static tree build_cp_library_fn (tree, enum tree_code, tree, int);
static void store_parm_decls (tree);
static void initialize_local_var (tree, tree);
static void expand_static_init (tree, tree);
tree cp_global_trees[CPTI_MAX];
#define local_names cp_function_chain->x_local_names
tree static_aggregates;
tree tls_aggregates;
tree integer_two_node;
vec<tree, va_gc> *static_decls;
vec<tree, va_gc> *keyed_classes;
struct GTY((chain_next ("%h.next"))) named_label_use_entry {
struct named_label_use_entry *next;
cp_binding_level *binding_level;
tree names_in_scope;
location_t o_goto_locus;
bool in_omp_scope;
};
struct GTY((for_user)) named_label_entry {
tree name;  
tree label_decl; 
named_label_entry *outer; 
cp_binding_level *binding_level;
tree names_in_scope;
vec<tree, va_gc> *bad_decls;
named_label_use_entry *uses;
bool in_try_scope;
bool in_catch_scope;
bool in_omp_scope;
bool in_transaction_scope;
bool in_constexpr_if;
};
#define named_labels cp_function_chain->x_named_labels

int function_depth;
bool flag_noexcept_type;
enum deprecated_states deprecated_state = DEPRECATED_NORMAL;

struct GTY(()) incomplete_var {
tree decl;
tree incomplete_type;
};
static GTY(()) vec<incomplete_var, va_gc> *incomplete_vars;

tmpl_spec_kind
current_tmpl_spec_kind (int n_class_scopes)
{
int n_template_parm_scopes = 0;
int seen_specialization_p = 0;
int innermost_specialization_p = 0;
cp_binding_level *b;
for (b = current_binding_level;
b->kind == sk_template_parms;
b = b->level_chain)
{
if (b->explicit_spec_p)
{
if (n_template_parm_scopes == 0)
innermost_specialization_p = 1;
else
seen_specialization_p = 1;
}
else if (seen_specialization_p == 1)
return tsk_invalid_member_spec;
++n_template_parm_scopes;
}
if (processing_explicit_instantiation)
{
if (n_template_parm_scopes != 0)
return tsk_invalid_expl_inst;
else
return tsk_expl_inst;
}
if (n_template_parm_scopes < n_class_scopes)
return tsk_insufficient_parms;
else if (n_template_parm_scopes == n_class_scopes)
return tsk_none;
else if (n_template_parm_scopes > n_class_scopes + 1)
return tsk_excessive_parms;
else
return innermost_specialization_p ? tsk_expl_spec : tsk_template;
}
void
finish_scope (void)
{
poplevel (0, 0, 0);
}
static void
check_label_used (tree label)
{
if (!processing_template_decl)
{
if (DECL_INITIAL (label) == NULL_TREE)
{
location_t location;
error ("label %q+D used but not defined", label);
location = input_location;
define_label (location, DECL_NAME (label));
}
else 
warn_for_unused_label (label);
}
}
static int
sort_labels (const void *a, const void *b)
{
tree label1 = *(tree const *) a;
tree label2 = *(tree const *) b;
return DECL_UID (label1) > DECL_UID (label2) ? -1 : +1;
}
static void
pop_labels (tree block)
{
if (!named_labels)
return;
auto_vec<tree, 32> labels (named_labels->elements ());
hash_table<named_label_hash>::iterator end (named_labels->end ());
for (hash_table<named_label_hash>::iterator iter
(named_labels->begin ()); iter != end; ++iter)
{
named_label_entry *ent = *iter;
gcc_checking_assert (!ent->outer);
if (ent->label_decl)
labels.quick_push (ent->label_decl);
ggc_free (ent);
}
named_labels = NULL;
labels.qsort (sort_labels);
while (labels.length ())
{
tree label = labels.pop ();
DECL_CHAIN (label) = BLOCK_VARS (block);
BLOCK_VARS (block) = label;
check_label_used (label);
}
}
static void
pop_local_label (tree id, tree label)
{
check_label_used (label);
named_label_entry **slot = named_labels->find_slot_with_hash
(id, IDENTIFIER_HASH_VALUE (id), NO_INSERT);
named_label_entry *ent = *slot;
if (ent->outer)
ent = ent->outer;
else
{
ent = ggc_cleared_alloc<named_label_entry> ();
ent->name = id;
}
*slot = ent;
}
void *
objc_get_current_scope (void)
{
return current_binding_level;
}
void
objc_mark_locals_volatile (void *enclosing_blk)
{
cp_binding_level *scope;
for (scope = current_binding_level;
scope && scope != enclosing_blk;
scope = scope->level_chain)
{
tree decl;
for (decl = scope->names; decl; decl = TREE_CHAIN (decl))
objc_volatilize_decl (decl);
if (scope->kind == sk_function_parms)
break;
}
}
static bool
level_for_constexpr_if (cp_binding_level *b)
{
return (b->kind == sk_cond && b->this_entity
&& TREE_CODE (b->this_entity) == IF_STMT
&& IF_STMT_CONSTEXPR_P (b->this_entity));
}
int
poplevel_named_label_1 (named_label_entry **slot, cp_binding_level *bl)
{
named_label_entry *ent = *slot;
cp_binding_level *obl = bl->level_chain;
if (ent->binding_level == bl)
{
tree decl;
for (decl = ent->names_in_scope; decl; decl = (DECL_P (decl)
? DECL_CHAIN (decl)
: TREE_CHAIN (decl)))
if (decl_jump_unsafe (decl))
vec_safe_push (ent->bad_decls, decl);
ent->binding_level = obl;
ent->names_in_scope = obl->names;
switch (bl->kind)
{
case sk_try:
ent->in_try_scope = true;
break;
case sk_catch:
ent->in_catch_scope = true;
break;
case sk_omp:
ent->in_omp_scope = true;
break;
case sk_transaction:
ent->in_transaction_scope = true;
break;
case sk_block:
if (level_for_constexpr_if (bl->level_chain))
ent->in_constexpr_if = true;
break;
default:
break;
}
}
else if (ent->uses)
{
struct named_label_use_entry *use;
for (use = ent->uses; use ; use = use->next)
if (use->binding_level == bl)
{
use->binding_level = obl;
use->names_in_scope = obl->names;
if (bl->kind == sk_omp)
use->in_omp_scope = true;
}
}
return 1;
}
static int unused_but_set_errorcount;
tree
poplevel (int keep, int reverse, int functionbody)
{
tree link;
tree decls;
tree subblocks;
tree block;
tree decl;
int leaving_for_scope;
scope_kind kind;
unsigned ix;
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
restart:
block = NULL_TREE;
gcc_assert (current_binding_level->kind != sk_class
&& current_binding_level->kind != sk_namespace);
if (current_binding_level->kind == sk_cleanup)
functionbody = 0;
subblocks = functionbody >= 0 ? current_binding_level->blocks : 0;
gcc_assert (!vec_safe_length (current_binding_level->class_shadowed));
gcc_assert (keep == 0 || keep == 1);
if (current_binding_level->keep)
keep = 1;
if (cfun && !functionbody && named_labels)
named_labels->traverse<cp_binding_level *, poplevel_named_label_1>
(current_binding_level);
decls = current_binding_level->names;
if (reverse)
{
decls = nreverse (decls);
current_binding_level->names = decls;
}
block = NULL_TREE;
if (functionbody && subblocks && BLOCK_CHAIN (subblocks) == NULL_TREE)
keep = 0;
else if (keep == 1 || functionbody)
block = make_node (BLOCK);
if (block != NULL_TREE)
{
BLOCK_VARS (block) = decls;
BLOCK_SUBBLOCKS (block) = subblocks;
}
if (keep >= 0)
for (link = subblocks; link; link = BLOCK_CHAIN (link))
BLOCK_SUPERCONTEXT (link) = block;
leaving_for_scope
= current_binding_level->kind == sk_for && flag_new_for_scope;
if ((warn_unused_variable || warn_unused_but_set_variable)
&& current_binding_level->kind != sk_template_parms
&& !processing_template_decl)
for (tree d = get_local_decls (); d; d = TREE_CHAIN (d))
{
decl = TREE_CODE (d) == TREE_LIST ? TREE_VALUE (d) : d;
tree type = TREE_TYPE (decl);
if (VAR_P (decl)
&& (! TREE_USED (decl) || !DECL_READ_P (decl))
&& ! DECL_IN_SYSTEM_HEADER (decl)
&& (DECL_DECOMPOSITION_P (decl) ? !DECL_DECOMP_BASE (decl)
: (DECL_NAME (decl) && !DECL_ARTIFICIAL (decl)))
&& type != error_mark_node
&& (!CLASS_TYPE_P (type)
|| !TYPE_HAS_NONTRIVIAL_DESTRUCTOR (type)
|| lookup_attribute ("warn_unused",
TYPE_ATTRIBUTES (TREE_TYPE (decl)))))
{
if (! TREE_USED (decl))
{
if (!DECL_NAME (decl) && DECL_DECOMPOSITION_P (decl))
warning_at (DECL_SOURCE_LOCATION (decl),
OPT_Wunused_variable,
"unused structured binding declaration");
else
warning_at (DECL_SOURCE_LOCATION (decl),
OPT_Wunused_variable, "unused variable %qD", decl);
}
else if (DECL_CONTEXT (decl) == current_function_decl
&& TREE_CODE (TREE_TYPE (decl)) != REFERENCE_TYPE
&& errorcount == unused_but_set_errorcount)
{
if (!DECL_NAME (decl) && DECL_DECOMPOSITION_P (decl))
warning_at (DECL_SOURCE_LOCATION (decl),
OPT_Wunused_but_set_variable, "structured "
"binding declaration set but not used");
else
warning_at (DECL_SOURCE_LOCATION (decl),
OPT_Wunused_but_set_variable,
"variable %qD set but not used", decl);
unused_but_set_errorcount = errorcount;
}
}
}
for (link = decls; link; link = TREE_CHAIN (link))
{
decl = TREE_CODE (link) == TREE_LIST ? TREE_VALUE (link) : link;
tree name = OVL_NAME (decl);
if (leaving_for_scope && VAR_P (decl)
&& cxx_dialect < cxx11
&& name)
{
cxx_binding *ob = outer_binding (name,
IDENTIFIER_BINDING (name),
true);
tree ns_binding = NULL_TREE;
if (!ob)
ns_binding = get_namespace_binding (current_namespace, name);
if (ob && ob->scope == current_binding_level->level_chain)
;
else if ((ob && (TREE_CODE (ob->value) == TYPE_DECL))
|| (ns_binding && TREE_CODE (ns_binding) == TYPE_DECL))
;
else
{
DECL_DEAD_FOR_LOCAL (link) = 1;
if (ob && ob->value)
{
SET_DECL_SHADOWED_FOR_VAR (link, ob->value);
DECL_HAS_SHADOWED_FOR_VAR_P (link) = 1;
}
vec_safe_push (
current_binding_level->level_chain->dead_vars_from_for,
link);
IDENTIFIER_BINDING (name)->scope
= current_binding_level->level_chain;
name = NULL_TREE;
}
}
if (TREE_CODE (decl) == LABEL_DECL)
pop_local_label (name, decl);
else
pop_local_binding (name, decl);
}
FOR_EACH_VEC_SAFE_ELT_REVERSE (current_binding_level->dead_vars_from_for,
ix, decl)
pop_local_binding (DECL_NAME (decl), decl);
for (link = current_binding_level->type_shadowed;
link; link = TREE_CHAIN (link))
SET_IDENTIFIER_TYPE_VALUE (TREE_PURPOSE (link), TREE_VALUE (link));
if (block)
{
tree* d;
for (d = &BLOCK_VARS (block); *d; )
{
if (TREE_CODE (*d) == TREE_LIST
|| (!processing_template_decl
&& undeduced_auto_decl (*d)))
*d = TREE_CHAIN (*d);
else
d = &DECL_CHAIN (*d);
}
}
if (functionbody)
{
if (block)
{
BLOCK_VARS (block) = 0;
pop_labels (block);
}
else
pop_labels (subblocks);
}
kind = current_binding_level->kind;
if (kind == sk_cleanup)
{
tree stmt;
stmt = pop_stmt_list (current_binding_level->statement_list);
stmt = c_build_bind_expr (input_location, block, stmt);
add_stmt (stmt);
}
leave_scope ();
if (functionbody)
{
gcc_assert (DECL_INITIAL (current_function_decl) == error_mark_node);
DECL_INITIAL (current_function_decl) = block ? block : subblocks;
if (subblocks)
{
if (FUNCTION_NEEDS_BODY_BLOCK (current_function_decl))
{
if (BLOCK_SUBBLOCKS (subblocks))
BLOCK_OUTER_CURLY_BRACE_P (BLOCK_SUBBLOCKS (subblocks)) = 1;
}
else
BLOCK_OUTER_CURLY_BRACE_P (subblocks) = 1;
}
}
else if (block)
current_binding_level->blocks
= block_chainon (current_binding_level->blocks, block);
else if (subblocks)
current_binding_level->blocks
= block_chainon (current_binding_level->blocks, subblocks);
if (block)
TREE_USED (block) = 1;
if (kind == sk_cleanup)
goto restart;
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return block;
}
int
wrapup_namespace_globals ()
{
if (vec<tree, va_gc> *statics = static_decls)
{
tree decl;
unsigned int i;
FOR_EACH_VEC_ELT (*statics, i, decl)
{
if (warn_unused_function
&& TREE_CODE (decl) == FUNCTION_DECL
&& DECL_INITIAL (decl) == 0
&& DECL_EXTERNAL (decl)
&& !TREE_PUBLIC (decl)
&& !DECL_ARTIFICIAL (decl)
&& !DECL_FRIEND_PSEUDO_TEMPLATE_INSTANTIATION (decl)
&& !TREE_NO_WARNING (decl))
warning_at (DECL_SOURCE_LOCATION (decl),
OPT_Wunused_function,
"%qF declared %<static%> but never defined", decl);
if (VAR_P (decl)
&& DECL_EXTERNAL (decl)
&& DECL_INLINE_VAR_P (decl)
&& DECL_ODR_USED (decl))
error_at (DECL_SOURCE_LOCATION (decl),
"odr-used inline variable %qD is not defined", decl);
}
static_decls = NULL;
return wrapup_global_declarations (statics->address (),
statics->length ());
}
return 0;
}

tree
create_implicit_typedef (tree name, tree type)
{
tree decl;
decl = build_decl (input_location, TYPE_DECL, name, type);
DECL_ARTIFICIAL (decl) = 1;
SET_DECL_IMPLICIT_TYPEDEF_P (decl);
TYPE_NAME (type) = decl;
TYPE_STUB_DECL (type) = decl;
return decl;
}
static void
push_local_name (tree decl)
{
size_t i, nelts;
tree t, name;
timevar_start (TV_NAME_LOOKUP);
name = DECL_NAME (decl);
nelts = vec_safe_length (local_names);
for (i = 0; i < nelts; i++)
{
t = (*local_names)[i];
if (DECL_NAME (t) == name)
{
retrofit_lang_decl (decl);
DECL_LANG_SPECIFIC (decl)->u.base.u2sel = 1;
if (DECL_DISCRIMINATOR_SET_P (t))
DECL_DISCRIMINATOR (decl) = DECL_DISCRIMINATOR (t) + 1;
else
DECL_DISCRIMINATOR (decl) = 1;
(*local_names)[i] = decl;
timevar_stop (TV_NAME_LOOKUP);
return;
}
}
vec_safe_push (local_names, decl);
timevar_stop (TV_NAME_LOOKUP);
}

int
decls_match (tree newdecl, tree olddecl, bool record_versions )
{
int types_match;
if (newdecl == olddecl)
return 1;
if (TREE_CODE (newdecl) != TREE_CODE (olddecl))
return 0;
gcc_assert (DECL_P (newdecl));
if (TREE_CODE (newdecl) == FUNCTION_DECL)
{
tree f1 = TREE_TYPE (newdecl);
tree f2 = TREE_TYPE (olddecl);
tree p1 = TYPE_ARG_TYPES (f1);
tree p2 = TYPE_ARG_TYPES (f2);
tree r2;
tree t1 = (DECL_USE_TEMPLATE (newdecl)
? DECL_TI_TEMPLATE (newdecl)
: NULL_TREE);
tree t2 = (DECL_USE_TEMPLATE (olddecl)
? DECL_TI_TEMPLATE (olddecl)
: NULL_TREE);
if (t1 != t2)
return 0;
if (CP_DECL_CONTEXT (newdecl) != CP_DECL_CONTEXT (olddecl)
&& ! (DECL_EXTERN_C_P (newdecl)
&& DECL_EXTERN_C_P (olddecl)))
return 0;
if (DECL_IS_BUILTIN (olddecl)
&& DECL_EXTERN_C_P (olddecl) && !DECL_EXTERN_C_P (newdecl))
return 0;
if (TREE_CODE (f1) != TREE_CODE (f2))
return 0;
r2 = fndecl_declared_return_type (olddecl);
if (same_type_p (TREE_TYPE (f1), r2))
{
if (!prototype_p (f2) && DECL_EXTERN_C_P (olddecl)
&& (DECL_BUILT_IN (olddecl)
#ifndef NO_IMPLICIT_EXTERN_C
|| (DECL_IN_SYSTEM_HEADER (newdecl) && !DECL_CLASS_SCOPE_P (newdecl))
|| (DECL_IN_SYSTEM_HEADER (olddecl) && !DECL_CLASS_SCOPE_P (olddecl))
#endif
))
{
types_match = self_promoting_args_p (p1);
if (p1 == void_list_node)
TREE_TYPE (newdecl) = TREE_TYPE (olddecl);
}
#ifndef NO_IMPLICIT_EXTERN_C
else if (!prototype_p (f1)
&& (DECL_EXTERN_C_P (olddecl)
&& DECL_IN_SYSTEM_HEADER (olddecl)
&& !DECL_CLASS_SCOPE_P (olddecl))
&& (DECL_EXTERN_C_P (newdecl)
&& DECL_IN_SYSTEM_HEADER (newdecl)
&& !DECL_CLASS_SCOPE_P (newdecl)))
{
types_match = self_promoting_args_p (p2);
TREE_TYPE (newdecl) = TREE_TYPE (olddecl);
}
#endif
else
types_match =
compparms (p1, p2)
&& type_memfn_rqual (f1) == type_memfn_rqual (f2)
&& (TYPE_ATTRIBUTES (TREE_TYPE (newdecl)) == NULL_TREE
|| comp_type_attributes (TREE_TYPE (newdecl),
TREE_TYPE (olddecl)) != 0);
}
else
types_match = 0;
if (types_match
&& !DECL_EXTERN_C_P (newdecl)
&& !DECL_EXTERN_C_P (olddecl)
&& record_versions
&& maybe_version_functions (newdecl, olddecl,
(!DECL_FUNCTION_VERSIONED (newdecl)
|| !DECL_FUNCTION_VERSIONED (olddecl))))
return 0;
}
else if (TREE_CODE (newdecl) == TEMPLATE_DECL)
{
tree oldres = DECL_TEMPLATE_RESULT (olddecl);
tree newres = DECL_TEMPLATE_RESULT (newdecl);
if (TREE_CODE (newres) != TREE_CODE (oldres))
return 0;
if (!comp_template_parms (DECL_TEMPLATE_PARMS (newdecl),
DECL_TEMPLATE_PARMS (olddecl)))
return 0;
if (TREE_CODE (DECL_TEMPLATE_RESULT (newdecl)) == TYPE_DECL)
types_match = (same_type_p (TREE_TYPE (oldres), TREE_TYPE (newres))
&& equivalently_constrained (olddecl, newdecl));
else
types_match = decls_match (oldres, newres);
}
else
{
if (VAR_P (newdecl)
&& CP_DECL_CONTEXT (newdecl) != CP_DECL_CONTEXT (olddecl)
&& !(DECL_EXTERN_C_P (olddecl) && DECL_EXTERN_C_P (newdecl)))
return 0;
if (TREE_TYPE (newdecl) == error_mark_node)
types_match = TREE_TYPE (olddecl) == error_mark_node;
else if (TREE_TYPE (olddecl) == NULL_TREE)
types_match = TREE_TYPE (newdecl) == NULL_TREE;
else if (TREE_TYPE (newdecl) == NULL_TREE)
types_match = 0;
else
types_match = comptypes (TREE_TYPE (newdecl),
TREE_TYPE (olddecl),
COMPARE_REDECLARATION);
}
if (types_match && VAR_OR_FUNCTION_DECL_P (newdecl))
types_match = equivalently_constrained (newdecl, olddecl);
return types_match;
}
bool
maybe_version_functions (tree newdecl, tree olddecl, bool record)
{
if (!targetm.target_option.function_versions (newdecl, olddecl))
return false;
if (!DECL_FUNCTION_VERSIONED (olddecl))
{
DECL_FUNCTION_VERSIONED (olddecl) = 1;
if (DECL_ASSEMBLER_NAME_SET_P (olddecl))
mangle_decl (olddecl);
}
if (!DECL_FUNCTION_VERSIONED (newdecl))
{
DECL_FUNCTION_VERSIONED (newdecl) = 1;
if (DECL_ASSEMBLER_NAME_SET_P (newdecl))
mangle_decl (newdecl);
}
if (record)
cgraph_node::record_function_versions (olddecl, newdecl);
return true;
}
void
warn_extern_redeclared_static (tree newdecl, tree olddecl)
{
if (TREE_CODE (newdecl) == TYPE_DECL
|| TREE_CODE (newdecl) == TEMPLATE_DECL
|| TREE_CODE (newdecl) == CONST_DECL
|| TREE_CODE (newdecl) == NAMESPACE_DECL)
return;
if (TREE_CODE (newdecl) == FUNCTION_DECL
&& DECL_STATIC_FUNCTION_P (newdecl))
return;
if (DECL_THIS_STATIC (olddecl) || !DECL_THIS_STATIC (newdecl))
return;
if (TREE_CODE (olddecl) == FUNCTION_DECL
&& DECL_ARTIFICIAL (olddecl))
return;
if (permerror (DECL_SOURCE_LOCATION (newdecl),
"%qD was declared %<extern%> and later %<static%>", newdecl))
inform (DECL_SOURCE_LOCATION (olddecl),
"previous declaration of %qD", olddecl);
}
static void
check_redeclaration_exception_specification (tree new_decl,
tree old_decl)
{
tree new_exceptions = TYPE_RAISES_EXCEPTIONS (TREE_TYPE (new_decl));
tree old_exceptions = TYPE_RAISES_EXCEPTIONS (TREE_TYPE (old_decl));
if (UNEVALUATED_NOEXCEPT_SPEC_P (new_exceptions)
&& UNEVALUATED_NOEXCEPT_SPEC_P (old_exceptions))
return;
if (!type_dependent_expression_p (old_decl))
{
maybe_instantiate_noexcept (new_decl);
maybe_instantiate_noexcept (old_decl);
}
new_exceptions = TYPE_RAISES_EXCEPTIONS (TREE_TYPE (new_decl));
old_exceptions = TYPE_RAISES_EXCEPTIONS (TREE_TYPE (old_decl));
if (! DECL_IS_BUILTIN (old_decl)
&& !comp_except_specs (new_exceptions, old_exceptions, ce_normal))
{
const char *const msg
= G_("declaration of %qF has a different exception specifier");
bool complained = true;
location_t new_loc = DECL_SOURCE_LOCATION (new_decl);
if (DECL_IN_SYSTEM_HEADER (old_decl))
complained = pedwarn (new_loc, OPT_Wsystem_headers, msg, new_decl);
else if (!flag_exceptions)
complained = pedwarn (new_loc, OPT_Wpedantic, msg, new_decl);
else
error_at (new_loc, msg, new_decl);
if (complained)
inform (DECL_SOURCE_LOCATION (old_decl),
"from previous declaration %qF", old_decl);
}
}
static bool
validate_constexpr_redeclaration (tree old_decl, tree new_decl)
{
old_decl = STRIP_TEMPLATE (old_decl);
new_decl = STRIP_TEMPLATE (new_decl);
if (!VAR_OR_FUNCTION_DECL_P (old_decl)
|| !VAR_OR_FUNCTION_DECL_P (new_decl))
return true;
if (DECL_DECLARED_CONSTEXPR_P (old_decl)
== DECL_DECLARED_CONSTEXPR_P (new_decl))
return true;
if (TREE_CODE (old_decl) == FUNCTION_DECL)
{
if (DECL_BUILT_IN (old_decl))
{
DECL_DECLARED_CONSTEXPR_P (old_decl)
= DECL_DECLARED_CONSTEXPR_P (new_decl);
return true;
}
if (! DECL_TEMPLATE_SPECIALIZATION (old_decl)
&& DECL_TEMPLATE_SPECIALIZATION (new_decl))
return true;
error_at (DECL_SOURCE_LOCATION (new_decl),
"redeclaration %qD differs in %<constexpr%> "
"from previous declaration", new_decl);
inform (DECL_SOURCE_LOCATION (old_decl),
"previous declaration %qD", old_decl);
return false;
}
return true;
}
static inline bool
check_concept_refinement (tree olddecl, tree newdecl)
{
if (!DECL_DECLARED_CONCEPT_P (olddecl) || !DECL_DECLARED_CONCEPT_P (newdecl))
return false;
tree d1 = DECL_TEMPLATE_RESULT (olddecl);
tree d2 = DECL_TEMPLATE_RESULT (newdecl);
if (TREE_CODE (d1) != TREE_CODE (d2))
return false;
tree t1 = TREE_TYPE (d1);
tree t2 = TREE_TYPE (d2);
if (TREE_CODE (d1) == FUNCTION_DECL)
{
if (compparms (TYPE_ARG_TYPES (t1), TYPE_ARG_TYPES (t2))
&& comp_template_parms (DECL_TEMPLATE_PARMS (olddecl),
DECL_TEMPLATE_PARMS (newdecl))
&& !equivalently_constrained (olddecl, newdecl))
{
error ("cannot specialize concept %q#D", olddecl);
return true;
}
}
return false;
}
static void
check_redeclaration_no_default_args (tree decl)
{
gcc_assert (DECL_DECLARES_FUNCTION_P (decl));
for (tree t = FUNCTION_FIRST_USER_PARMTYPE (decl);
t && t != void_list_node; t = TREE_CHAIN (t))
if (TREE_PURPOSE (t))
{
permerror (DECL_SOURCE_LOCATION (decl),
"redeclaration of %q#D may not have default "
"arguments", decl);
return;
}
}
static void
merge_attribute_bits (tree newdecl, tree olddecl)
{
TREE_THIS_VOLATILE (newdecl) |= TREE_THIS_VOLATILE (olddecl);
TREE_THIS_VOLATILE (olddecl) |= TREE_THIS_VOLATILE (newdecl);
TREE_NOTHROW (newdecl) |= TREE_NOTHROW (olddecl);
TREE_NOTHROW (olddecl) |= TREE_NOTHROW (newdecl);
TREE_READONLY (newdecl) |= TREE_READONLY (olddecl);
TREE_READONLY (olddecl) |= TREE_READONLY (newdecl);
DECL_IS_MALLOC (newdecl) |= DECL_IS_MALLOC (olddecl);
DECL_IS_MALLOC (olddecl) |= DECL_IS_MALLOC (newdecl);
DECL_PURE_P (newdecl) |= DECL_PURE_P (olddecl);
DECL_PURE_P (olddecl) |= DECL_PURE_P (newdecl);
DECL_UNINLINABLE (newdecl) |= DECL_UNINLINABLE (olddecl);
DECL_UNINLINABLE (olddecl) |= DECL_UNINLINABLE (newdecl);
}
#define GNU_INLINE_P(fn) (DECL_DECLARED_INLINE_P (fn)			\
&& lookup_attribute ("gnu_inline",		\
DECL_ATTRIBUTES (fn)))
tree
duplicate_decls (tree newdecl, tree olddecl, bool newdecl_is_friend)
{
unsigned olddecl_uid = DECL_UID (olddecl);
int olddecl_friend = 0, types_match = 0, hidden_friend = 0;
int new_defines_function = 0;
tree new_template_info;
if (newdecl == olddecl)
return olddecl;
types_match = decls_match (newdecl, olddecl);
if (TREE_TYPE (newdecl) == error_mark_node
|| TREE_TYPE (olddecl) == error_mark_node)
return error_mark_node;
if (DECL_NAME (newdecl)
&& DECL_NAME (olddecl)
&& UDLIT_OPER_P (DECL_NAME (newdecl))
&& UDLIT_OPER_P (DECL_NAME (olddecl)))
{
if (TREE_CODE (newdecl) == TEMPLATE_DECL
&& TREE_CODE (olddecl) != TEMPLATE_DECL
&& check_raw_literal_operator (olddecl))
error ("literal operator template %q+D conflicts with"
" raw literal operator %qD", newdecl, olddecl);
else if (TREE_CODE (newdecl) != TEMPLATE_DECL
&& TREE_CODE (olddecl) == TEMPLATE_DECL
&& check_raw_literal_operator (newdecl))
error ("raw literal operator %q+D conflicts with"
" literal operator template %qD", newdecl, olddecl);
}
const bool merge_attr = (TREE_CODE (newdecl) != FUNCTION_DECL
|| !DECL_TEMPLATE_SPECIALIZATION (newdecl)
|| DECL_TEMPLATE_SPECIALIZATION (olddecl));
if (DECL_P (olddecl)
&& TREE_CODE (newdecl) == FUNCTION_DECL
&& TREE_CODE (olddecl) == FUNCTION_DECL
&& merge_attr
&& diagnose_mismatched_attributes (olddecl, newdecl))
{
if (DECL_INITIAL (olddecl))
inform (DECL_SOURCE_LOCATION (olddecl),
"previous definition of %qD was here", olddecl);
else
inform (DECL_SOURCE_LOCATION (olddecl),
"previous declaration of %qD was here", olddecl);
}
if (TREE_CODE (olddecl) == FUNCTION_DECL
&& DECL_ARTIFICIAL (olddecl))
{
gcc_assert (!DECL_HIDDEN_FRIEND_P (olddecl));
if (TREE_CODE (newdecl) != FUNCTION_DECL)
{
if (DECL_ANTICIPATED (olddecl))
{
if (TREE_PUBLIC (newdecl)
&& CP_DECL_CONTEXT (newdecl) == global_namespace)
warning_at (DECL_SOURCE_LOCATION (newdecl),
OPT_Wbuiltin_declaration_mismatch,
"built-in function %qD declared as non-function",
newdecl);
return NULL_TREE;
}
if (! TREE_PUBLIC (newdecl))
{
warning (OPT_Wshadow, 
DECL_BUILT_IN (olddecl)
? G_("shadowing built-in function %q#D")
: G_("shadowing library function %q#D"), olddecl);
return NULL_TREE;
}
else if (! DECL_BUILT_IN (olddecl))
warning_at (DECL_SOURCE_LOCATION (newdecl), 0,
"library function %q#D redeclared as non-function %q#D",
olddecl, newdecl);
else
error ("declaration of %q+#D conflicts with built-in "
"declaration %q#D", newdecl, olddecl);
return NULL_TREE;
}
else if (DECL_OMP_DECLARE_REDUCTION_P (olddecl))
{
gcc_assert (DECL_OMP_DECLARE_REDUCTION_P (newdecl));
error_at (DECL_SOURCE_LOCATION (newdecl),
"redeclaration of %<pragma omp declare reduction%>");
inform (DECL_SOURCE_LOCATION (olddecl),
"previous %<pragma omp declare reduction%> declaration");
return error_mark_node;
}
else if (!types_match)
{
if (DECL_ANTICIPATED (olddecl))
{
tree t1, t2;
gcc_assert (DECL_IS_BUILTIN (olddecl));
gcc_assert (DECL_EXTERN_C_P (olddecl));
if (!DECL_EXTERN_C_P (newdecl))
return NULL_TREE;
for (t1 = TYPE_ARG_TYPES (TREE_TYPE (newdecl)),
t2 = TYPE_ARG_TYPES (TREE_TYPE (olddecl));
t1 || t2;
t1 = TREE_CHAIN (t1), t2 = TREE_CHAIN (t2))
{
if (!t1 || !t2)
break;
for (unsigned i = 0;
i < sizeof (builtin_structptr_types)
/ sizeof (builtin_structptr_type);
++i)
if (TREE_VALUE (t2) == builtin_structptr_types[i].node)
{
tree t = TREE_VALUE (t1);
if (TYPE_PTR_P (t)
&& TYPE_IDENTIFIER (TREE_TYPE (t))
== get_identifier (builtin_structptr_types[i].str)
&& compparms (TREE_CHAIN (t1), TREE_CHAIN (t2)))
{
tree oldargs = TYPE_ARG_TYPES (TREE_TYPE (olddecl));
TYPE_ARG_TYPES (TREE_TYPE (olddecl))
= TYPE_ARG_TYPES (TREE_TYPE (newdecl));
types_match = decls_match (newdecl, olddecl);
if (types_match)
return duplicate_decls (newdecl, olddecl,
newdecl_is_friend);
TYPE_ARG_TYPES (TREE_TYPE (olddecl)) = oldargs;
}
goto next_arg;
}
if (! same_type_p (TREE_VALUE (t1), TREE_VALUE (t2)))
break;
next_arg:;
}
warning_at (DECL_SOURCE_LOCATION (newdecl),
OPT_Wbuiltin_declaration_mismatch,
"declaration of %q#D conflicts with built-in "
"declaration %q#D", newdecl, olddecl);
}
else if ((DECL_EXTERN_C_P (newdecl)
&& DECL_EXTERN_C_P (olddecl))
|| compparms (TYPE_ARG_TYPES (TREE_TYPE (newdecl)),
TYPE_ARG_TYPES (TREE_TYPE (olddecl))))
{
if (DECL_BUILT_IN (olddecl))
{
tree id = DECL_NAME (olddecl);
const char *name = IDENTIFIER_POINTER (id);
size_t len;
if (name[0] == '_'
&& name[1] == '_'
&& (strncmp (name + 2, "builtin_",
strlen ("builtin_")) == 0
|| (len = strlen (name)) <= strlen ("___chk")
|| memcmp (name + len - strlen ("_chk"),
"_chk", strlen ("_chk") + 1) != 0))
{
if (DECL_INITIAL (newdecl))
{
error_at (DECL_SOURCE_LOCATION (newdecl),
"definition of %q#D ambiguates built-in "
"declaration %q#D", newdecl, olddecl);
return error_mark_node;
}
if (permerror (DECL_SOURCE_LOCATION (newdecl),
"new declaration %q#D ambiguates built-in"
" declaration %q#D", newdecl, olddecl)
&& flag_permissive)
inform (DECL_SOURCE_LOCATION (newdecl),
"ignoring the %q#D declaration", newdecl);
return flag_permissive ? olddecl : error_mark_node;
}
}
if (TREE_PUBLIC (newdecl))
warning_at (DECL_SOURCE_LOCATION (newdecl),
OPT_Wbuiltin_declaration_mismatch,
"new declaration %q#D ambiguates built-in "
"declaration %q#D", newdecl, olddecl);
else
warning (OPT_Wshadow, 
DECL_BUILT_IN (olddecl)
? G_("shadowing built-in function %q#D")
: G_("shadowing library function %q#D"), olddecl);
}
else
return NULL_TREE;
COPY_DECL_RTL (newdecl, olddecl);
}
else if (DECL_IS_BUILTIN (olddecl))
{
tree type = TREE_TYPE (newdecl);
tree attribs = (*targetm.merge_type_attributes)
(TREE_TYPE (olddecl), type);
type = cp_build_type_attribute_variant (type, attribs);
TREE_TYPE (newdecl) = TREE_TYPE (olddecl) = type;
}
if (DECL_BUILT_IN_CLASS (olddecl) == BUILT_IN_NORMAL
&& DECL_ANTICIPATED (olddecl)
&& TREE_NOTHROW (newdecl)
&& !TREE_NOTHROW (olddecl))
{
enum built_in_function fncode = DECL_FUNCTION_CODE (olddecl);
tree tmpdecl = builtin_decl_explicit (fncode);
if (tmpdecl && tmpdecl != olddecl && types_match)
TREE_NOTHROW (tmpdecl)  = 1;
}
TREE_NOTHROW (olddecl) = 0;
if (DECL_THIS_STATIC (newdecl) && !DECL_THIS_STATIC (olddecl))
{
DECL_THIS_STATIC (olddecl) = 1;
TREE_PUBLIC (olddecl) = 0;
SET_DECL_LANGUAGE (olddecl, DECL_LANGUAGE (newdecl));
COPY_DECL_RTL (newdecl, olddecl);
}
}
else if (TREE_CODE (olddecl) != TREE_CODE (newdecl))
{
if (TREE_CODE (olddecl) != NAMESPACE_DECL
&& TREE_CODE (newdecl) != NAMESPACE_DECL
&& (TREE_CODE (olddecl) != TEMPLATE_DECL
|| TREE_CODE (DECL_TEMPLATE_RESULT (olddecl)) != TYPE_DECL)
&& (TREE_CODE (newdecl) != TEMPLATE_DECL
|| TREE_CODE (DECL_TEMPLATE_RESULT (newdecl)) != TYPE_DECL))
{
if ((TREE_CODE (olddecl) == TYPE_DECL && DECL_ARTIFICIAL (olddecl)
&& TREE_CODE (newdecl) != TYPE_DECL)
|| (TREE_CODE (newdecl) == TYPE_DECL && DECL_ARTIFICIAL (newdecl)
&& TREE_CODE (olddecl) != TYPE_DECL))
{
return NULL_TREE;
}
if ((TREE_CODE (newdecl) == FUNCTION_DECL
&& DECL_FUNCTION_TEMPLATE_P (olddecl))
|| (TREE_CODE (olddecl) == FUNCTION_DECL
&& DECL_FUNCTION_TEMPLATE_P (newdecl)))
return NULL_TREE;
}
error ("%q#D redeclared as different kind of symbol", newdecl);
if (TREE_CODE (olddecl) == TREE_LIST)
olddecl = TREE_VALUE (olddecl);
inform (DECL_SOURCE_LOCATION (olddecl),
"previous declaration %q#D", olddecl);
return error_mark_node;
}
else if (!types_match)
{
if (CP_DECL_CONTEXT (newdecl) != CP_DECL_CONTEXT (olddecl))
return NULL_TREE;
if (TREE_CODE (newdecl) == TEMPLATE_DECL)
{
if (TREE_CODE (DECL_TEMPLATE_RESULT (olddecl)) == TYPE_DECL
|| TREE_CODE (DECL_TEMPLATE_RESULT (newdecl)) == TYPE_DECL)
{
error ("conflicting declaration of template %q+#D", newdecl);
inform (DECL_SOURCE_LOCATION (olddecl),
"previous declaration %q#D", olddecl);
return error_mark_node;
}
else if (TREE_CODE (DECL_TEMPLATE_RESULT (olddecl)) == FUNCTION_DECL
&& TREE_CODE (DECL_TEMPLATE_RESULT (newdecl)) == FUNCTION_DECL
&& compparms (TYPE_ARG_TYPES (TREE_TYPE (DECL_TEMPLATE_RESULT (olddecl))),
TYPE_ARG_TYPES (TREE_TYPE (DECL_TEMPLATE_RESULT (newdecl))))
&& comp_template_parms (DECL_TEMPLATE_PARMS (newdecl),
DECL_TEMPLATE_PARMS (olddecl))
&& same_type_p (TREE_TYPE (TREE_TYPE (newdecl)),
TREE_TYPE (TREE_TYPE (olddecl)))
&& equivalently_constrained (olddecl, newdecl))
{
error ("ambiguating new declaration %q+#D", newdecl);
inform (DECL_SOURCE_LOCATION (olddecl),
"old declaration %q#D", olddecl);
}
else if (check_concept_refinement (olddecl, newdecl))
return error_mark_node;
return NULL_TREE;
}
if (TREE_CODE (newdecl) == FUNCTION_DECL)
{
if (DECL_EXTERN_C_P (newdecl) && DECL_EXTERN_C_P (olddecl))
{
error ("conflicting declaration of C function %q+#D",
newdecl);
inform (DECL_SOURCE_LOCATION (olddecl),
"previous declaration %q#D", olddecl);
return NULL_TREE;
}
else if ((!DECL_FUNCTION_VERSIONED (newdecl)
&& !DECL_FUNCTION_VERSIONED (olddecl))
&& compparms (TYPE_ARG_TYPES (TREE_TYPE (newdecl)),
TYPE_ARG_TYPES (TREE_TYPE (olddecl)))
&& equivalently_constrained (newdecl, olddecl))
{
error ("ambiguating new declaration of %q+#D", newdecl);
inform (DECL_SOURCE_LOCATION (olddecl),
"old declaration %q#D", olddecl);
return error_mark_node;
}
else
return NULL_TREE;
}
else
{
error ("conflicting declaration %q+#D", newdecl);
inform (DECL_SOURCE_LOCATION (olddecl),
"previous declaration as %q#D", olddecl);
return error_mark_node;
}
}
else if (TREE_CODE (newdecl) == FUNCTION_DECL
&& ((DECL_TEMPLATE_SPECIALIZATION (olddecl)
&& (!DECL_TEMPLATE_INFO (newdecl)
|| (DECL_TI_TEMPLATE (newdecl)
!= DECL_TI_TEMPLATE (olddecl))))
|| (DECL_TEMPLATE_SPECIALIZATION (newdecl)
&& (!DECL_TEMPLATE_INFO (olddecl)
|| (DECL_TI_TEMPLATE (olddecl)
!= DECL_TI_TEMPLATE (newdecl))))))
return NULL_TREE;
else if (TREE_CODE (newdecl) == FUNCTION_DECL
&& ((DECL_TEMPLATE_INSTANTIATION (olddecl)
&& !DECL_USE_TEMPLATE (newdecl))
|| (DECL_TEMPLATE_INSTANTIATION (newdecl)
&& !DECL_USE_TEMPLATE (olddecl))))
return NULL_TREE;
else if (TREE_CODE (newdecl) == NAMESPACE_DECL)
{
if (DECL_NAMESPACE_ALIAS (newdecl)
&& (DECL_NAMESPACE_ALIAS (newdecl)
== DECL_NAMESPACE_ALIAS (olddecl)))
return olddecl;
return NULL_TREE;
}
else
{
const char *errmsg = redeclaration_error_message (newdecl, olddecl);
if (errmsg)
{
error_at (DECL_SOURCE_LOCATION (newdecl), errmsg, newdecl);
if (DECL_NAME (olddecl) != NULL_TREE)
inform (DECL_SOURCE_LOCATION (olddecl),
(DECL_INITIAL (olddecl) && namespace_bindings_p ())
? G_("%q#D previously defined here")
: G_("%q#D previously declared here"), olddecl);
return error_mark_node;
}
else if (TREE_CODE (olddecl) == FUNCTION_DECL
&& DECL_INITIAL (olddecl) != NULL_TREE
&& !prototype_p (TREE_TYPE (olddecl))
&& prototype_p (TREE_TYPE (newdecl)))
{
if (warning_at (DECL_SOURCE_LOCATION (newdecl), 0,
"prototype specified for %q#D", newdecl))
inform (DECL_SOURCE_LOCATION (olddecl),
"previous non-prototype definition here");
}
else if (VAR_OR_FUNCTION_DECL_P (olddecl)
&& DECL_LANGUAGE (newdecl) != DECL_LANGUAGE (olddecl))
{
if (current_lang_depth () == 0)
{
retrofit_lang_decl (newdecl);
SET_DECL_LANGUAGE (newdecl, DECL_LANGUAGE (olddecl));
}
else
{
error ("conflicting declaration of %q+#D with %qL linkage",
newdecl, DECL_LANGUAGE (newdecl));
inform (DECL_SOURCE_LOCATION (olddecl),
"previous declaration with %qL linkage",
DECL_LANGUAGE (olddecl));
}
}
if (DECL_LANG_SPECIFIC (olddecl) && DECL_USE_TEMPLATE (olddecl))
;
else if (TREE_CODE (olddecl) == FUNCTION_DECL)
{
if (DECL_FUNCTION_MEMBER_P (olddecl)
&& (
DECL_TEMPLATE_INFO (olddecl)
|| CLASSTYPE_TEMPLATE_INFO (CP_DECL_CONTEXT (olddecl))))
check_redeclaration_no_default_args (newdecl);
else
{
tree t1 = FUNCTION_FIRST_USER_PARMTYPE (olddecl);
tree t2 = FUNCTION_FIRST_USER_PARMTYPE (newdecl);
int i = 1;
for (; t1 && t1 != void_list_node;
t1 = TREE_CHAIN (t1), t2 = TREE_CHAIN (t2), i++)
if (TREE_PURPOSE (t1) && TREE_PURPOSE (t2))
{
if (simple_cst_equal (TREE_PURPOSE (t1),
TREE_PURPOSE (t2)) == 1)
{
if (permerror (input_location,
"default argument given for parameter "
"%d of %q#D", i, newdecl))
inform (DECL_SOURCE_LOCATION (olddecl),
"previous specification in %q#D here",
olddecl);
}
else
{
error ("default argument given for parameter %d "
"of %q#D", i, newdecl);
inform (DECL_SOURCE_LOCATION (olddecl),
"previous specification in %q#D here",
olddecl);
}
}
}
}
}
if (TREE_CODE (olddecl) == TYPE_DECL
&& (DECL_IMPLICIT_TYPEDEF_P (olddecl)
|| DECL_IMPLICIT_TYPEDEF_P (newdecl)))
return NULL_TREE;
warn_extern_redeclared_static (newdecl, olddecl);
if (!validate_constexpr_redeclaration (olddecl, newdecl))
return error_mark_node;
if (TREE_CODE (newdecl) == FUNCTION_DECL)
{
if (DECL_VINDEX (olddecl))
DECL_VINDEX (newdecl) = DECL_VINDEX (olddecl);
if (DECL_CONTEXT (olddecl))
DECL_CONTEXT (newdecl) = DECL_CONTEXT (olddecl);
DECL_STATIC_CONSTRUCTOR (newdecl) |= DECL_STATIC_CONSTRUCTOR (olddecl);
DECL_STATIC_DESTRUCTOR (newdecl) |= DECL_STATIC_DESTRUCTOR (olddecl);
DECL_PURE_VIRTUAL_P (newdecl) |= DECL_PURE_VIRTUAL_P (olddecl);
DECL_VIRTUAL_P (newdecl) |= DECL_VIRTUAL_P (olddecl);
DECL_INVALID_OVERRIDER_P (newdecl) |= DECL_INVALID_OVERRIDER_P (olddecl);
DECL_FINAL_P (newdecl) |= DECL_FINAL_P (olddecl);
DECL_OVERRIDE_P (newdecl) |= DECL_OVERRIDE_P (olddecl);
DECL_THIS_STATIC (newdecl) |= DECL_THIS_STATIC (olddecl);
if (DECL_OVERLOADED_OPERATOR_P (olddecl))
DECL_OVERLOADED_OPERATOR_CODE_RAW (newdecl)
= DECL_OVERLOADED_OPERATOR_CODE_RAW (olddecl);
new_defines_function = DECL_INITIAL (newdecl) != NULL_TREE;
if (warn_redundant_decls && ! DECL_ARTIFICIAL (olddecl)
&& !(new_defines_function && DECL_INITIAL (olddecl) == NULL_TREE)
&& !(DECL_EXTERNAL (olddecl) && ! DECL_EXTERNAL (newdecl))
&& ! (newdecl_is_friend || DECL_FRIEND_P (olddecl))
&& (! DECL_TEMPLATE_SPECIALIZATION (newdecl)
|| DECL_TEMPLATE_SPECIALIZATION (olddecl)))
{
if (warning_at (DECL_SOURCE_LOCATION (newdecl),
OPT_Wredundant_decls,
"redundant redeclaration of %qD in same scope",
newdecl))
inform (DECL_SOURCE_LOCATION (olddecl),
"previous declaration of %qD", olddecl);
}
if (!(DECL_TEMPLATE_INSTANTIATION (olddecl)
&& DECL_TEMPLATE_SPECIALIZATION (newdecl)))
{
if (DECL_DELETED_FN (newdecl))
{
error ("deleted definition of %q+D", newdecl);
inform (DECL_SOURCE_LOCATION (olddecl),
"previous declaration of %qD", olddecl);
}
DECL_DELETED_FN (newdecl) |= DECL_DELETED_FN (olddecl);
}
}
if (TREE_CODE (olddecl) == TYPE_DECL)
{
tree newtype = TREE_TYPE (newdecl);
tree oldtype = TREE_TYPE (olddecl);
if (newtype != error_mark_node && oldtype != error_mark_node
&& TYPE_LANG_SPECIFIC (newtype) && TYPE_LANG_SPECIFIC (oldtype))
CLASSTYPE_FRIEND_CLASSES (newtype)
= CLASSTYPE_FRIEND_CLASSES (oldtype);
DECL_ORIGINAL_TYPE (newdecl) = DECL_ORIGINAL_TYPE (olddecl);
}
if (merge_attr)
DECL_ATTRIBUTES (newdecl)
= (*targetm.merge_decl_attributes) (olddecl, newdecl);
else
DECL_ATTRIBUTES (olddecl) = DECL_ATTRIBUTES (newdecl);
if (DECL_DECLARES_FUNCTION_P (olddecl) && DECL_DECLARES_FUNCTION_P (newdecl))
{
olddecl_friend = DECL_FRIEND_P (olddecl);
hidden_friend = (DECL_ANTICIPATED (olddecl)
&& DECL_HIDDEN_FRIEND_P (olddecl)
&& newdecl_is_friend);
if (!hidden_friend)
{
DECL_ANTICIPATED (olddecl) = 0;
DECL_HIDDEN_FRIEND_P (olddecl) = 0;
}
}
if (TREE_CODE (newdecl) == TEMPLATE_DECL)
{
tree old_result;
tree new_result;
old_result = DECL_TEMPLATE_RESULT (olddecl);
new_result = DECL_TEMPLATE_RESULT (newdecl);
TREE_TYPE (olddecl) = TREE_TYPE (old_result);
DECL_TEMPLATE_SPECIALIZATIONS (olddecl)
= chainon (DECL_TEMPLATE_SPECIALIZATIONS (olddecl),
DECL_TEMPLATE_SPECIALIZATIONS (newdecl));
DECL_ATTRIBUTES (old_result)
= (*targetm.merge_decl_attributes) (old_result, new_result);
if (DECL_FUNCTION_TEMPLATE_P (newdecl))
{
if (DECL_SOURCE_LOCATION (newdecl)
!= DECL_SOURCE_LOCATION (olddecl))
check_redeclaration_no_default_args (newdecl);
check_default_args (newdecl);
if (GNU_INLINE_P (old_result) != GNU_INLINE_P (new_result)
&& DECL_INITIAL (new_result))
{
if (DECL_INITIAL (old_result))
DECL_UNINLINABLE (old_result) = 1;
else
DECL_UNINLINABLE (old_result) = DECL_UNINLINABLE (new_result);
DECL_EXTERNAL (old_result) = DECL_EXTERNAL (new_result);
DECL_NOT_REALLY_EXTERN (old_result)
= DECL_NOT_REALLY_EXTERN (new_result);
DECL_INTERFACE_KNOWN (old_result)
= DECL_INTERFACE_KNOWN (new_result);
DECL_DECLARED_INLINE_P (old_result)
= DECL_DECLARED_INLINE_P (new_result);
DECL_DISREGARD_INLINE_LIMITS (old_result)
|= DECL_DISREGARD_INLINE_LIMITS (new_result);
}
else
{
DECL_DECLARED_INLINE_P (old_result)
|= DECL_DECLARED_INLINE_P (new_result);
DECL_DISREGARD_INLINE_LIMITS (old_result)
|= DECL_DISREGARD_INLINE_LIMITS (new_result);
check_redeclaration_exception_specification (newdecl, olddecl);
merge_attribute_bits (new_result, old_result);
}
}
if (DECL_INITIAL (new_result) != NULL_TREE)
{
DECL_SOURCE_LOCATION (olddecl)
= DECL_SOURCE_LOCATION (old_result)
= DECL_SOURCE_LOCATION (newdecl);
DECL_INITIAL (old_result) = DECL_INITIAL (new_result);
if (DECL_FUNCTION_TEMPLATE_P (newdecl))
{
tree parm;
DECL_ARGUMENTS (old_result)
= DECL_ARGUMENTS (new_result);
for (parm = DECL_ARGUMENTS (old_result); parm;
parm = DECL_CHAIN (parm))
DECL_CONTEXT (parm) = old_result;
}
}
return olddecl;
}
if (types_match)
{
if (TREE_CODE (newdecl) == FUNCTION_DECL)
check_redeclaration_exception_specification (newdecl, olddecl);
tree oldtype = TREE_TYPE (olddecl);
tree newtype;
if (TREE_CODE (newdecl) == TYPE_DECL)
{
tree tem = TREE_TYPE (newdecl);
newtype = oldtype;
if (TYPE_USER_ALIGN (tem))
{
if (TYPE_ALIGN (tem) > TYPE_ALIGN (newtype))
SET_TYPE_ALIGN (newtype, TYPE_ALIGN (tem));
TYPE_USER_ALIGN (newtype) = true;
}
if (TYPE_NAME (TREE_TYPE (newdecl)) == newdecl)
{
tree remove = TREE_TYPE (newdecl);
for (tree t = TYPE_MAIN_VARIANT (remove); ;
t = TYPE_NEXT_VARIANT (t))
if (TYPE_NEXT_VARIANT (t) == remove)
{
TYPE_NEXT_VARIANT (t) = TYPE_NEXT_VARIANT (remove);
break;
}
}
}
else if (merge_attr)
newtype = merge_types (TREE_TYPE (newdecl), TREE_TYPE (olddecl));
else
newtype = TREE_TYPE (newdecl);
if (VAR_P (newdecl))
{
DECL_THIS_EXTERN (newdecl) |= DECL_THIS_EXTERN (olddecl);
if (DECL_INITIALIZED_P (olddecl)
&& !DECL_EXTERNAL (olddecl)
&& !TREE_READONLY (olddecl))
TREE_READONLY (newdecl) = 0;
DECL_INITIALIZED_P (newdecl) |= DECL_INITIALIZED_P (olddecl);
DECL_NONTRIVIALLY_INITIALIZED_P (newdecl)
|= DECL_NONTRIVIALLY_INITIALIZED_P (olddecl);
if (DECL_DEPENDENT_INIT_P (olddecl))
SET_DECL_DEPENDENT_INIT_P (newdecl, true);
DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P (newdecl)
|= DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P (olddecl);
if (DECL_CLASS_SCOPE_P (olddecl))
DECL_DECLARED_CONSTEXPR_P (newdecl)
|= DECL_DECLARED_CONSTEXPR_P (olddecl);
if (DECL_LANG_SPECIFIC (olddecl)
&& CP_DECL_THREADPRIVATE_P (olddecl))
{
retrofit_lang_decl (newdecl);
CP_DECL_THREADPRIVATE_P (newdecl) = 1;
}
}
if (flag_tm && TREE_CODE (newdecl) == FUNCTION_DECL
&& DECL_TEMPLATE_INSTANTIATION (olddecl)
&& DECL_TEMPLATE_SPECIALIZATION (newdecl)
&& tx_safe_fn_type_p (newtype)
&& !tx_safe_fn_type_p (TREE_TYPE (newdecl)))
newtype = tx_unsafe_fn_variant (newtype);
TREE_TYPE (newdecl) = TREE_TYPE (olddecl) = newtype;
if (TREE_CODE (newdecl) == FUNCTION_DECL)
check_default_args (newdecl);
if (! same_type_p (newtype, oldtype)
&& TREE_TYPE (newdecl) != error_mark_node
&& !(processing_template_decl && uses_template_parms (newdecl)))
layout_type (TREE_TYPE (newdecl));
if ((VAR_P (newdecl)
|| TREE_CODE (newdecl) == PARM_DECL
|| TREE_CODE (newdecl) == RESULT_DECL
|| TREE_CODE (newdecl) == FIELD_DECL
|| TREE_CODE (newdecl) == TYPE_DECL)
&& !(processing_template_decl && uses_template_parms (newdecl)))
layout_decl (newdecl, 0);
if (TREE_DEPRECATED (newdecl))
TREE_DEPRECATED (olddecl) = 1;
if (TREE_CODE (newdecl) == FUNCTION_DECL)
{
if (DECL_FUNCTION_SPECIFIC_TARGET (olddecl)
&& !DECL_FUNCTION_SPECIFIC_TARGET (newdecl))
DECL_FUNCTION_SPECIFIC_TARGET (newdecl)
= DECL_FUNCTION_SPECIFIC_TARGET (olddecl);
if (DECL_FUNCTION_SPECIFIC_OPTIMIZATION (olddecl)
&& !DECL_FUNCTION_SPECIFIC_OPTIMIZATION (newdecl))
DECL_FUNCTION_SPECIFIC_OPTIMIZATION (newdecl)
= DECL_FUNCTION_SPECIFIC_OPTIMIZATION (olddecl);
}
else
{
if (TREE_READONLY (newdecl))
TREE_READONLY (olddecl) = 1;
if (TREE_THIS_VOLATILE (newdecl))
TREE_THIS_VOLATILE (olddecl) = 1;
}
if (DECL_INITIAL (newdecl) == NULL_TREE
&& DECL_INITIAL (olddecl) != NULL_TREE)
{
DECL_INITIAL (newdecl) = DECL_INITIAL (olddecl);
DECL_SOURCE_LOCATION (newdecl) = DECL_SOURCE_LOCATION (olddecl);
if (TREE_CODE (newdecl) == FUNCTION_DECL)
{
DECL_SAVED_TREE (newdecl) = DECL_SAVED_TREE (olddecl);
DECL_STRUCT_FUNCTION (newdecl) = DECL_STRUCT_FUNCTION (olddecl);
}
}
if (TREE_CODE (newdecl) == FUNCTION_DECL)
{
DECL_NO_INSTRUMENT_FUNCTION_ENTRY_EXIT (newdecl)
|= DECL_NO_INSTRUMENT_FUNCTION_ENTRY_EXIT (olddecl);
DECL_NO_LIMIT_STACK (newdecl) |= DECL_NO_LIMIT_STACK (olddecl);
DECL_IS_OPERATOR_NEW (newdecl) |= DECL_IS_OPERATOR_NEW (olddecl);
DECL_LOOPING_CONST_OR_PURE_P (newdecl)
|= DECL_LOOPING_CONST_OR_PURE_P (olddecl);
if (merge_attr)
merge_attribute_bits (newdecl, olddecl);
else
{
TREE_THIS_VOLATILE (olddecl) = TREE_THIS_VOLATILE (newdecl);
TREE_READONLY (olddecl) = TREE_READONLY (newdecl);
TREE_NOTHROW (olddecl) = TREE_NOTHROW (newdecl);
DECL_IS_MALLOC (olddecl) = DECL_IS_MALLOC (newdecl);
DECL_PURE_P (olddecl) = DECL_PURE_P (newdecl);
}
COPY_DECL_RTL (olddecl, newdecl);
}
else if (VAR_P (newdecl)
&& (DECL_SIZE (olddecl) || !DECL_SIZE (newdecl)))
{
COPY_DECL_RTL (olddecl, newdecl);
}
}
else
{
tree oldstatic = value_member (olddecl, static_aggregates);
if (oldstatic)
TREE_VALUE (oldstatic) = error_mark_node;
TREE_TYPE (olddecl) = TREE_TYPE (newdecl);
TREE_READONLY (olddecl) = TREE_READONLY (newdecl);
TREE_THIS_VOLATILE (olddecl) = TREE_THIS_VOLATILE (newdecl);
TREE_NOTHROW (olddecl) = TREE_NOTHROW (newdecl);
TREE_SIDE_EFFECTS (olddecl) = TREE_SIDE_EFFECTS (newdecl);
}
merge_weak (newdecl, olddecl);
DECL_DEFER_OUTPUT (newdecl) |= DECL_DEFER_OUTPUT (olddecl);
TREE_PUBLIC (newdecl) = TREE_PUBLIC (olddecl);
TREE_STATIC (olddecl) = TREE_STATIC (newdecl) |= TREE_STATIC (olddecl);
if (! DECL_EXTERNAL (olddecl))
DECL_EXTERNAL (newdecl) = 0;
if (! DECL_COMDAT (olddecl))
DECL_COMDAT (newdecl) = 0;
new_template_info = NULL_TREE;
if (DECL_LANG_SPECIFIC (newdecl) && DECL_LANG_SPECIFIC (olddecl))
{
bool new_redefines_gnu_inline = false;
if (new_defines_function
&& ((DECL_INTERFACE_KNOWN (olddecl)
&& TREE_CODE (olddecl) == FUNCTION_DECL)
|| (TREE_CODE (olddecl) == TEMPLATE_DECL
&& (TREE_CODE (DECL_TEMPLATE_RESULT (olddecl))
== FUNCTION_DECL))))
{
tree fn = olddecl;
if (TREE_CODE (fn) == TEMPLATE_DECL)
fn = DECL_TEMPLATE_RESULT (olddecl);
new_redefines_gnu_inline = GNU_INLINE_P (fn) && DECL_INITIAL (fn);
}
if (!new_redefines_gnu_inline)
{
DECL_INTERFACE_KNOWN (newdecl) |= DECL_INTERFACE_KNOWN (olddecl);
DECL_NOT_REALLY_EXTERN (newdecl) |= DECL_NOT_REALLY_EXTERN (olddecl);
DECL_COMDAT (newdecl) |= DECL_COMDAT (olddecl);
}
DECL_TEMPLATE_INSTANTIATED (newdecl)
|= DECL_TEMPLATE_INSTANTIATED (olddecl);
DECL_ODR_USED (newdecl) |= DECL_ODR_USED (olddecl);
if (!DECL_USE_TEMPLATE (newdecl))
DECL_USE_TEMPLATE (newdecl) = DECL_USE_TEMPLATE (olddecl);
DECL_IN_AGGR_P (newdecl) = DECL_IN_AGGR_P (olddecl);
DECL_REPO_AVAILABLE_P (newdecl) = DECL_REPO_AVAILABLE_P (olddecl);
DECL_INITIALIZED_IN_CLASS_P (newdecl)
|= DECL_INITIALIZED_IN_CLASS_P (olddecl);
if (LANG_DECL_HAS_MIN (newdecl))
{
DECL_LANG_SPECIFIC (newdecl)->u.min.u2 =
DECL_LANG_SPECIFIC (olddecl)->u.min.u2;
if (DECL_TEMPLATE_INFO (newdecl))
{
new_template_info = DECL_TEMPLATE_INFO (newdecl);
if (DECL_TEMPLATE_INSTANTIATION (olddecl)
&& DECL_TEMPLATE_SPECIALIZATION (newdecl))
TINFO_USED_TEMPLATE_ID (DECL_TEMPLATE_INFO (olddecl))
= TINFO_USED_TEMPLATE_ID (new_template_info);
}
DECL_TEMPLATE_INFO (newdecl) = DECL_TEMPLATE_INFO (olddecl);
}
if (DECL_DECLARES_FUNCTION_P (newdecl))
{
DECL_NONCONVERTING_P (newdecl) = DECL_NONCONVERTING_P (olddecl);
DECL_BEFRIENDING_CLASSES (newdecl)
= chainon (DECL_BEFRIENDING_CLASSES (newdecl),
DECL_BEFRIENDING_CLASSES (olddecl));
if (DECL_VIRTUAL_P (newdecl))
SET_DECL_THUNKS (newdecl, DECL_THUNKS (olddecl));
}
else if (VAR_P (newdecl)
&& VAR_HAD_UNKNOWN_BOUND (olddecl))
SET_VAR_HAD_UNKNOWN_BOUND (newdecl);
}
if (TREE_CODE (newdecl) == FUNCTION_DECL)
{
tree parm;
tree oldarg, newarg;
for (oldarg = DECL_ARGUMENTS(olddecl), 
newarg = DECL_ARGUMENTS(newdecl);
oldarg && newarg;
oldarg = DECL_CHAIN(oldarg), newarg = DECL_CHAIN(newarg)) {
DECL_ATTRIBUTES (newarg)
= (*targetm.merge_decl_attributes) (oldarg, newarg);
DECL_ATTRIBUTES (oldarg) = DECL_ATTRIBUTES (newarg);
}
if (DECL_TEMPLATE_INSTANTIATION (olddecl)
&& !DECL_TEMPLATE_INSTANTIATION (newdecl))
{
gcc_assert (DECL_TEMPLATE_SPECIALIZATION (newdecl));
if (DECL_ODR_USED (olddecl))
error ("explicit specialization of %qD after first use",
olddecl);
SET_DECL_TEMPLATE_SPECIALIZATION (olddecl);
DECL_COMDAT (newdecl) = (TREE_PUBLIC (newdecl)
&& DECL_DECLARED_INLINE_P (newdecl));
DECL_VISIBILITY_SPECIFIED (olddecl) = 0;
gcc_assert (!merge_attr);
DECL_DECLARED_INLINE_P (olddecl)
= DECL_DECLARED_INLINE_P (newdecl);
DECL_DISREGARD_INLINE_LIMITS (olddecl)
= DECL_DISREGARD_INLINE_LIMITS (newdecl);
DECL_UNINLINABLE (olddecl) = DECL_UNINLINABLE (newdecl);
}
else if (new_defines_function && DECL_INITIAL (olddecl))
{
DECL_UNINLINABLE (newdecl) = 1;
}
else
{
if (DECL_PENDING_INLINE_P (olddecl))
{
DECL_PENDING_INLINE_P (newdecl) = 1;
DECL_PENDING_INLINE_INFO (newdecl)
= DECL_PENDING_INLINE_INFO (olddecl);
}
else if (DECL_PENDING_INLINE_P (newdecl))
;
else if (DECL_SAVED_FUNCTION_DATA (newdecl) == NULL)
DECL_SAVED_FUNCTION_DATA (newdecl)
= DECL_SAVED_FUNCTION_DATA (olddecl);
DECL_DECLARED_INLINE_P (newdecl) |= DECL_DECLARED_INLINE_P (olddecl);
DECL_UNINLINABLE (newdecl) = DECL_UNINLINABLE (olddecl)
= (DECL_UNINLINABLE (newdecl) || DECL_UNINLINABLE (olddecl));
DECL_DISREGARD_INLINE_LIMITS (newdecl)
= DECL_DISREGARD_INLINE_LIMITS (olddecl)
= (DECL_DISREGARD_INLINE_LIMITS (newdecl)
|| DECL_DISREGARD_INLINE_LIMITS (olddecl));
}
DECL_ABSTRACT_P (newdecl) = DECL_ABSTRACT_P (olddecl);
for (parm = DECL_ARGUMENTS (newdecl); parm;
parm = DECL_CHAIN (parm))
DECL_CONTEXT (parm) = olddecl;
if (! types_match)
{
SET_DECL_LANGUAGE (olddecl, DECL_LANGUAGE (newdecl));
COPY_DECL_ASSEMBLER_NAME (newdecl, olddecl);
COPY_DECL_RTL (newdecl, olddecl);
}
if (! types_match || new_defines_function)
{
DECL_ARGUMENTS (olddecl) = DECL_ARGUMENTS (newdecl);
DECL_RESULT (olddecl) = DECL_RESULT (newdecl);
}
if (DECL_BUILT_IN (olddecl)
&& (new_defines_function ? GNU_INLINE_P (newdecl) : types_match))
{
DECL_BUILT_IN_CLASS (newdecl) = DECL_BUILT_IN_CLASS (olddecl);
DECL_FUNCTION_CODE (newdecl) = DECL_FUNCTION_CODE (olddecl);
COPY_DECL_RTL (olddecl, newdecl);
if (DECL_BUILT_IN_CLASS (newdecl) == BUILT_IN_NORMAL)
{
enum built_in_function fncode = DECL_FUNCTION_CODE (newdecl);
switch (fncode)
{
case BUILT_IN_STPCPY:
if (builtin_decl_explicit_p (fncode))
set_builtin_decl_implicit_p (fncode, true);
break;
default:
if (builtin_decl_explicit_p (fncode))
set_builtin_decl_declared_p (fncode, true);
break;
}
}
copy_attributes_to_builtin (newdecl);
}
if (new_defines_function)
SET_DECL_LANGUAGE (newdecl, DECL_LANGUAGE (olddecl));
else if (types_match)
{
DECL_RESULT (newdecl) = DECL_RESULT (olddecl);
if (DECL_ARGUMENTS (olddecl))
DECL_ARGUMENTS (newdecl) = DECL_ARGUMENTS (olddecl);
}
}
else if (TREE_CODE (newdecl) == NAMESPACE_DECL)
NAMESPACE_LEVEL (newdecl) = NAMESPACE_LEVEL (olddecl);
TREE_ADDRESSABLE (newdecl) = TREE_ADDRESSABLE (olddecl);
TREE_ASM_WRITTEN (newdecl) = TREE_ASM_WRITTEN (olddecl);
DECL_COMMON (newdecl) = DECL_COMMON (olddecl);
COPY_DECL_ASSEMBLER_NAME (olddecl, newdecl);
if (DECL_VISIBILITY_SPECIFIED (olddecl)
&& DECL_VISIBILITY_SPECIFIED (newdecl)
&& DECL_VISIBILITY (newdecl) != DECL_VISIBILITY (olddecl))
{
if (warning_at (DECL_SOURCE_LOCATION (newdecl), OPT_Wattributes,
"%qD: visibility attribute ignored because it "
"conflicts with previous declaration", newdecl))
inform (DECL_SOURCE_LOCATION (olddecl),
"previous declaration of %qD", olddecl);
}
if (DECL_VISIBILITY_SPECIFIED (olddecl))
{
DECL_VISIBILITY (newdecl) = DECL_VISIBILITY (olddecl);
DECL_VISIBILITY_SPECIFIED (newdecl) = 1;
}
if (VAR_P (newdecl) && DECL_HAS_INIT_PRIORITY_P (newdecl))
{
SET_DECL_INIT_PRIORITY (olddecl, DECL_INIT_PRIORITY (newdecl));
DECL_HAS_INIT_PRIORITY_P (olddecl) = 1;
}
if (DECL_ALIGN (olddecl) > DECL_ALIGN (newdecl))
{
SET_DECL_ALIGN (newdecl, DECL_ALIGN (olddecl));
DECL_USER_ALIGN (newdecl) |= DECL_USER_ALIGN (olddecl);
}
DECL_USER_ALIGN (olddecl) = DECL_USER_ALIGN (newdecl);
if (DECL_WARN_IF_NOT_ALIGN (olddecl)
> DECL_WARN_IF_NOT_ALIGN (newdecl))
SET_DECL_WARN_IF_NOT_ALIGN (newdecl,
DECL_WARN_IF_NOT_ALIGN (olddecl));
if (TREE_CODE (newdecl) == FIELD_DECL)
DECL_PACKED (olddecl) = DECL_PACKED (newdecl);
if (DECL_LANG_SPECIFIC (olddecl))
{
gcc_assert (DECL_LANG_SPECIFIC (olddecl)
!= DECL_LANG_SPECIFIC (newdecl));
ggc_free (DECL_LANG_SPECIFIC (olddecl));
}
if (TREE_USED (olddecl))
TREE_USED (newdecl) = 1;
else if (TREE_USED (newdecl))
TREE_USED (olddecl) = 1;
if (VAR_P (newdecl))
{
if (DECL_READ_P (olddecl))
DECL_READ_P (newdecl) = 1;
else if (DECL_READ_P (newdecl))
DECL_READ_P (olddecl) = 1;
}
if (DECL_PRESERVE_P (olddecl))
DECL_PRESERVE_P (newdecl) = 1;
else if (DECL_PRESERVE_P (newdecl))
DECL_PRESERVE_P (olddecl) = 1;
if (TREE_CODE (newdecl) == FUNCTION_DECL
&& DECL_FUNCTION_VERSIONED (olddecl))
{
DECL_FUNCTION_VERSIONED (newdecl) = 1;
cgraph_node::delete_function_version_by_decl (newdecl);
}
if (TREE_CODE (newdecl) == FUNCTION_DECL)
{
int function_size;
struct symtab_node *snode = symtab_node::get (olddecl);
function_size = sizeof (struct tree_decl_common);
memcpy ((char *) olddecl + sizeof (struct tree_common),
(char *) newdecl + sizeof (struct tree_common),
function_size - sizeof (struct tree_common));
memcpy ((char *) olddecl + sizeof (struct tree_decl_common),
(char *) newdecl + sizeof (struct tree_decl_common),
sizeof (struct tree_function_decl) - sizeof (struct tree_decl_common));
olddecl->decl_with_vis.symtab_node = snode;
if (new_template_info)
reregister_specialization (newdecl,
new_template_info,
olddecl);
}
else
{
size_t size = tree_code_size (TREE_CODE (newdecl));
memcpy ((char *) olddecl + sizeof (struct tree_common),
(char *) newdecl + sizeof (struct tree_common),
sizeof (struct tree_decl_common) - sizeof (struct tree_common));
switch (TREE_CODE (newdecl))
{
case LABEL_DECL:
case VAR_DECL:
case RESULT_DECL:
case PARM_DECL:
case FIELD_DECL:
case TYPE_DECL:
case CONST_DECL:
{
struct symtab_node *snode = NULL;
if (VAR_P (newdecl)
&& (TREE_STATIC (olddecl) || TREE_PUBLIC (olddecl)
|| DECL_EXTERNAL (olddecl)))
snode = symtab_node::get (olddecl);
memcpy ((char *) olddecl + sizeof (struct tree_decl_common),
(char *) newdecl + sizeof (struct tree_decl_common),
size - sizeof (struct tree_decl_common)
+ TREE_CODE_LENGTH (TREE_CODE (newdecl)) * sizeof (char *));
if (VAR_P (newdecl))
olddecl->decl_with_vis.symtab_node = snode;
}
break;
default:
memcpy ((char *) olddecl + sizeof (struct tree_decl_common),
(char *) newdecl + sizeof (struct tree_decl_common),
sizeof (struct tree_decl_non_common) - sizeof (struct tree_decl_common)
+ TREE_CODE_LENGTH (TREE_CODE (newdecl)) * sizeof (char *));
break;
}
}
if (VAR_OR_FUNCTION_DECL_P (newdecl))
{
if (DECL_EXTERNAL (olddecl)
|| TREE_PUBLIC (olddecl)
|| TREE_STATIC (olddecl))
{
if (DECL_SECTION_NAME (newdecl) != NULL)
set_decl_section_name (olddecl, DECL_SECTION_NAME (newdecl));
if (DECL_ONE_ONLY (newdecl))
{
struct symtab_node *oldsym, *newsym;
if (TREE_CODE (olddecl) == FUNCTION_DECL)
oldsym = cgraph_node::get_create (olddecl);
else
oldsym = varpool_node::get_create (olddecl);
newsym = symtab_node::get (newdecl);
oldsym->set_comdat_group (newsym->get_comdat_group ());
}
}
if (VAR_P (newdecl)
&& CP_DECL_THREAD_LOCAL_P (newdecl))
{
CP_DECL_THREAD_LOCAL_P (olddecl) = true;
if (!processing_template_decl)
set_decl_tls_model (olddecl, DECL_TLS_MODEL (newdecl));
}
}
DECL_UID (olddecl) = olddecl_uid;
if (olddecl_friend)
DECL_FRIEND_P (olddecl) = 1;
if (hidden_friend)
{
DECL_ANTICIPATED (olddecl) = 1;
DECL_HIDDEN_FRIEND_P (olddecl) = 1;
}
DECL_ATTRIBUTES (olddecl) = DECL_ATTRIBUTES (newdecl);
if (DECL_RTL_SET_P (olddecl)
&& (TREE_CODE (olddecl) == FUNCTION_DECL
|| (VAR_P (olddecl)
&& TREE_STATIC (olddecl))))
make_decl_rtl (olddecl);
if (TREE_CODE (newdecl) == FUNCTION_DECL)
DECL_STRUCT_FUNCTION (newdecl) = NULL;
if (VAR_OR_FUNCTION_DECL_P (newdecl))
{
struct symtab_node *snode = symtab_node::get (newdecl);
if (snode)
snode->remove ();
}
if (flag_concepts)
remove_constraints (newdecl);
ggc_free (newdecl);
return olddecl;
}

static const char *
redeclaration_error_message (tree newdecl, tree olddecl)
{
if (TREE_CODE (newdecl) == TYPE_DECL)
{
if (same_type_p (TREE_TYPE (newdecl), TREE_TYPE (olddecl)))
return NULL;
else
return G_("redefinition of %q#D");
}
else if (TREE_CODE (newdecl) == FUNCTION_DECL)
{
if (DECL_LANG_SPECIFIC (olddecl) && DECL_PURE_VIRTUAL_P (olddecl)
&& DECL_INITIAL (olddecl) == NULL_TREE)
return NULL;
if (DECL_NAMESPACE_SCOPE_P (olddecl)
&& DECL_CONTEXT (olddecl) != DECL_CONTEXT (newdecl)
&& ! decls_match (olddecl, newdecl))
return G_("%qD conflicts with used function");
if (decl_defined_p (olddecl)
&& decl_defined_p (newdecl))
{
if (DECL_NAME (olddecl) == NULL_TREE)
return G_("%q#D not declared in class");
else if (!GNU_INLINE_P (olddecl)
|| GNU_INLINE_P (newdecl))
return G_("redefinition of %q#D");
}
if (DECL_DECLARED_INLINE_P (olddecl) && DECL_DECLARED_INLINE_P (newdecl))
{
bool olda = GNU_INLINE_P (olddecl);
bool newa = GNU_INLINE_P (newdecl);
if (olda != newa)
{
if (newa)
return G_("%q+D redeclared inline with "
"%<gnu_inline%> attribute");
else
return G_("%q+D redeclared inline without "
"%<gnu_inline%> attribute");
}
}
check_abi_tag_redeclaration
(olddecl, lookup_attribute ("abi_tag", DECL_ATTRIBUTES (olddecl)),
lookup_attribute ("abi_tag", DECL_ATTRIBUTES (newdecl)));
return NULL;
}
else if (TREE_CODE (newdecl) == TEMPLATE_DECL)
{
tree nt, ot;
if (TREE_CODE (DECL_TEMPLATE_RESULT (newdecl)) == TYPE_DECL)
{
if (COMPLETE_TYPE_P (TREE_TYPE (newdecl))
&& COMPLETE_TYPE_P (TREE_TYPE (olddecl)))
return G_("redefinition of %q#D");
return NULL;
}
if (TREE_CODE (DECL_TEMPLATE_RESULT (newdecl)) != FUNCTION_DECL
|| (DECL_TEMPLATE_RESULT (newdecl)
== DECL_TEMPLATE_RESULT (olddecl)))
return NULL;
nt = DECL_TEMPLATE_RESULT (newdecl);
if (DECL_TEMPLATE_INFO (nt))
nt = DECL_TEMPLATE_RESULT (template_for_substitution (nt));
ot = DECL_TEMPLATE_RESULT (olddecl);
if (DECL_TEMPLATE_INFO (ot))
ot = DECL_TEMPLATE_RESULT (template_for_substitution (ot));
if (DECL_INITIAL (nt) && DECL_INITIAL (ot)
&& (!GNU_INLINE_P (ot) || GNU_INLINE_P (nt)))
return G_("redefinition of %q#D");
if (DECL_DECLARED_INLINE_P (ot) && DECL_DECLARED_INLINE_P (nt))
{
bool olda = GNU_INLINE_P (ot);
bool newa = GNU_INLINE_P (nt);
if (olda != newa)
{
if (newa)
return G_("%q+D redeclared inline with "
"%<gnu_inline%> attribute");
else
return G_("%q+D redeclared inline without "
"%<gnu_inline%> attribute");
}
}
if ((cxx_dialect != cxx98) 
&& TREE_CODE (ot) == FUNCTION_DECL && DECL_FRIEND_P (ot)
&& !check_default_tmpl_args (nt, DECL_TEMPLATE_PARMS (newdecl), 
true,
false,
2))
return G_("redeclaration of friend %q#D "
"may not have default template arguments");
return NULL;
}
else if (VAR_P (newdecl)
&& CP_DECL_THREAD_LOCAL_P (newdecl) != CP_DECL_THREAD_LOCAL_P (olddecl)
&& (! DECL_LANG_SPECIFIC (olddecl)
|| ! CP_DECL_THREADPRIVATE_P (olddecl)
|| CP_DECL_THREAD_LOCAL_P (newdecl)))
{
if (CP_DECL_THREAD_LOCAL_P (newdecl))
return G_("thread-local declaration of %q#D follows "
"non-thread-local declaration");
else
return G_("non-thread-local declaration of %q#D follows "
"thread-local declaration");
}
else if (toplevel_bindings_p () || DECL_NAMESPACE_SCOPE_P (newdecl))
{
if ((VAR_P (newdecl) && DECL_ANON_UNION_VAR_P (newdecl))
|| (VAR_P (olddecl) && DECL_ANON_UNION_VAR_P (olddecl)))
return G_("redeclaration of %q#D");
if (DECL_EXTERNAL (newdecl) || DECL_EXTERNAL (olddecl))
return NULL;
if (cxx_dialect >= cxx17
&& VAR_P (olddecl)
&& DECL_CLASS_SCOPE_P (olddecl)
&& DECL_DECLARED_CONSTEXPR_P (olddecl)
&& !DECL_INITIAL (newdecl))
{
DECL_EXTERNAL (newdecl) = 1;
if (global_options_set.x_warn_deprecated
&& warning_at (DECL_SOURCE_LOCATION (newdecl), OPT_Wdeprecated,
"redundant redeclaration of %<constexpr%> static "
"data member %qD", newdecl))
inform (DECL_SOURCE_LOCATION (olddecl),
"previous declaration of %qD", olddecl);
return NULL;
}
return G_("redefinition of %q#D");
}
else
{
if (!(DECL_EXTERNAL (newdecl) && DECL_EXTERNAL (olddecl)))
return G_("redeclaration of %q#D");
return NULL;
}
}

hashval_t
named_label_hash::hash (const value_type entry)
{
return IDENTIFIER_HASH_VALUE (entry->name);
}
bool
named_label_hash::equal (const value_type entry, compare_type name)
{
return name == entry->name;
}
static named_label_entry *
lookup_label_1 (tree id, bool making_local_p)
{
if (current_function_decl == NULL_TREE)
{
error ("label %qE referenced outside of any function", id);
return NULL;
}
if (!named_labels)
named_labels = hash_table<named_label_hash>::create_ggc (13);
hashval_t hash = IDENTIFIER_HASH_VALUE (id);
named_label_entry **slot
= named_labels->find_slot_with_hash (id, hash, INSERT);
named_label_entry *old = *slot;
if (old && old->label_decl)
{
if (!making_local_p)
return old;
if (old->binding_level == current_binding_level)
{
error ("local label %qE conflicts with existing label", id);
inform (DECL_SOURCE_LOCATION (old->label_decl), "previous label");
return NULL;
}
}
named_label_entry *ent = NULL;
if (old && !old->label_decl)
ent = old;
else
{
ent = ggc_cleared_alloc<named_label_entry> ();
ent->name = id;
ent->outer = old;
*slot = ent;
}
tree decl = build_decl (input_location, LABEL_DECL, id, void_type_node);
DECL_CONTEXT (decl) = current_function_decl;
SET_DECL_MODE (decl, VOIDmode);
if (making_local_p)
{
C_DECLARED_LABEL_FLAG (decl) = true;
DECL_CHAIN (decl) = current_binding_level->names;
current_binding_level->names = decl;
}
ent->label_decl = decl;
return ent;
}
tree
lookup_label (tree id)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
named_label_entry *ent = lookup_label_1 (id, false);
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return ent ? ent->label_decl : NULL_TREE;
}
tree
declare_local_label (tree id)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
named_label_entry *ent = lookup_label_1 (id, true);
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return ent ? ent->label_decl : NULL_TREE;
}
static int
decl_jump_unsafe (tree decl)
{
tree type = TREE_TYPE (decl);
if (!VAR_P (decl) || TREE_STATIC (decl)
|| type == error_mark_node)
return 0;
if (DECL_NONTRIVIALLY_INITIALIZED_P (decl)
|| variably_modified_type_p (type, NULL_TREE))
return 2;
if (TYPE_HAS_NONTRIVIAL_DESTRUCTOR (type))
return 1;
return 0;
}
static bool
identify_goto (tree decl, location_t loc, const location_t *locus,
diagnostic_t diag_kind)
{
bool complained
= emit_diagnostic (diag_kind, loc, 0,
decl ? N_("jump to label %qD")
: N_("jump to case label"), decl);
if (complained && locus)
inform (*locus, "  from here");
return complained;
}
static bool
check_previous_goto_1 (tree decl, cp_binding_level* level, tree names,
bool exited_omp, const location_t *locus)
{
cp_binding_level *b;
bool complained = false;
int identified = 0;
bool saw_eh = false, saw_omp = false, saw_tm = false, saw_cxif = false;
if (exited_omp)
{
complained = identify_goto (decl, input_location, locus, DK_ERROR);
if (complained)
inform (input_location, "  exits OpenMP structured block");
saw_omp = true;
identified = 2;
}
for (b = current_binding_level; b ; b = b->level_chain)
{
tree new_decls, old_decls = (b == level ? names : NULL_TREE);
for (new_decls = b->names; new_decls != old_decls;
new_decls = (DECL_P (new_decls) ? DECL_CHAIN (new_decls)
: TREE_CHAIN (new_decls)))
{
int problem = decl_jump_unsafe (new_decls);
if (! problem)
continue;
if (!identified)
{
complained = identify_goto (decl, input_location, locus,
DK_PERMERROR);
identified = 1;
}
if (complained)
{
if (problem > 1)
inform (DECL_SOURCE_LOCATION (new_decls),
"  crosses initialization of %q#D", new_decls);
else
inform (DECL_SOURCE_LOCATION (new_decls),
"  enters scope of %q#D, which has "
"non-trivial destructor", new_decls);
}
}
if (b == level)
break;
const char *inf = NULL;
location_t loc = input_location;
switch (b->kind)
{
case sk_try:
if (!saw_eh)
inf = N_("enters try block");
saw_eh = true;
break;
case sk_catch:
if (!saw_eh)
inf = N_("enters catch block");
saw_eh = true;
break;
case sk_omp:
if (!saw_omp)
inf = N_("enters OpenMP structured block");
saw_omp = true;
break;
case sk_transaction:
if (!saw_tm)
inf = N_("enters synchronized or atomic statement");
saw_tm = true;
break;
case sk_block:
if (!saw_cxif && level_for_constexpr_if (b->level_chain))
{
inf = N_("enters constexpr if statement");
loc = EXPR_LOCATION (b->level_chain->this_entity);
saw_cxif = true;
}
break;
default:
break;
}
if (inf)
{
if (identified < 2)
complained = identify_goto (decl, input_location, locus, DK_ERROR);
identified = 2;
if (complained)
inform (loc, "  %s", inf);
}
}
return !identified;
}
static void
check_previous_goto (tree decl, struct named_label_use_entry *use)
{
check_previous_goto_1 (decl, use->binding_level,
use->names_in_scope, use->in_omp_scope,
&use->o_goto_locus);
}
static bool
check_switch_goto (cp_binding_level* level)
{
return check_previous_goto_1 (NULL_TREE, level, level->names, false, NULL);
}
void
check_goto (tree decl)
{
if (TREE_CODE (decl) != LABEL_DECL)
return;
if (decl == cdtor_label)
return;
hashval_t hash = IDENTIFIER_HASH_VALUE (DECL_NAME (decl));
named_label_entry **slot
= named_labels->find_slot_with_hash (DECL_NAME (decl), hash, NO_INSERT);
named_label_entry *ent = *slot;
if (! DECL_INITIAL (decl))
{
if (ent->uses
&& ent->uses->names_in_scope == current_binding_level->names)
return;
named_label_use_entry *new_use
= ggc_alloc<named_label_use_entry> ();
new_use->binding_level = current_binding_level;
new_use->names_in_scope = current_binding_level->names;
new_use->o_goto_locus = input_location;
new_use->in_omp_scope = false;
new_use->next = ent->uses;
ent->uses = new_use;
return;
}
bool saw_catch = false, complained = false;
int identified = 0;
tree bad;
unsigned ix;
if (ent->in_try_scope || ent->in_catch_scope || ent->in_transaction_scope
|| ent->in_constexpr_if
|| ent->in_omp_scope || !vec_safe_is_empty (ent->bad_decls))
{
diagnostic_t diag_kind = DK_PERMERROR;
if (ent->in_try_scope || ent->in_catch_scope || ent->in_constexpr_if
|| ent->in_transaction_scope || ent->in_omp_scope)
diag_kind = DK_ERROR;
complained = identify_goto (decl, DECL_SOURCE_LOCATION (decl),
&input_location, diag_kind);
identified = 1 + (diag_kind == DK_ERROR);
}
FOR_EACH_VEC_SAFE_ELT (ent->bad_decls, ix, bad)
{
int u = decl_jump_unsafe (bad);
if (u > 1 && DECL_ARTIFICIAL (bad))
{
if (identified == 1)
{
complained = identify_goto (decl, DECL_SOURCE_LOCATION (decl),
&input_location, DK_ERROR);
identified = 2;
}
if (complained)
inform (DECL_SOURCE_LOCATION (bad), "  enters catch block");
saw_catch = true;
}
else if (complained)
{
if (u > 1)
inform (DECL_SOURCE_LOCATION (bad),
"  skips initialization of %q#D", bad);
else
inform (DECL_SOURCE_LOCATION (bad),
"  enters scope of %q#D which has "
"non-trivial destructor", bad);
}
}
if (complained)
{
if (ent->in_try_scope)
inform (input_location, "  enters try block");
else if (ent->in_catch_scope && !saw_catch)
inform (input_location, "  enters catch block");
else if (ent->in_transaction_scope)
inform (input_location, "  enters synchronized or atomic statement");
else if (ent->in_constexpr_if)
inform (input_location, "  enters %<constexpr%> if statement");
}
if (ent->in_omp_scope)
{
if (complained)
inform (input_location, "  enters OpenMP structured block");
}
else if (flag_openmp)
for (cp_binding_level *b = current_binding_level; b ; b = b->level_chain)
{
if (b == ent->binding_level)
break;
if (b->kind == sk_omp)
{
if (identified < 2)
{
complained = identify_goto (decl,
DECL_SOURCE_LOCATION (decl),
&input_location, DK_ERROR);
identified = 2;
}
if (complained)
inform (input_location, "  exits OpenMP structured block");
break;
}
}
}
bool
check_omp_return (void)
{
for (cp_binding_level *b = current_binding_level; b ; b = b->level_chain)
if (b->kind == sk_omp)
{
error ("invalid exit from OpenMP structured block");
return false;
}
else if (b->kind == sk_function_parms)
break;
return true;
}
static tree
define_label_1 (location_t location, tree name)
{
for (cp_binding_level *p = current_binding_level;
p->kind != sk_function_parms;
p = p->level_chain)
p->more_cleanups_ok = 0;
named_label_entry *ent = lookup_label_1 (name, false);
tree decl = ent->label_decl;
if (DECL_INITIAL (decl) != NULL_TREE)
{
error ("duplicate label %qD", decl);
return error_mark_node;
}
else
{
DECL_INITIAL (decl) = error_mark_node;
DECL_SOURCE_LOCATION (decl) = location;
ent->binding_level = current_binding_level;
ent->names_in_scope = current_binding_level->names;
for (named_label_use_entry *use = ent->uses; use; use = use->next)
check_previous_goto (decl, use);
ent->uses = NULL;
}
return decl;
}
tree
define_label (location_t location, tree name)
{
bool running = timevar_cond_start (TV_NAME_LOOKUP);
tree ret = define_label_1 (location, name);
timevar_cond_stop (TV_NAME_LOOKUP, running);
return ret;
}
struct cp_switch
{
cp_binding_level *level;
struct cp_switch *next;
tree switch_stmt;
splay_tree cases;
bool outside_range_p;
bool has_default_p;
bool break_stmt_seen_p;
bool in_loop_body_p;
};
static struct cp_switch *switch_stack;
void
push_switch (tree switch_stmt)
{
struct cp_switch *p = XNEW (struct cp_switch);
p->level = current_binding_level;
p->next = switch_stack;
p->switch_stmt = switch_stmt;
p->cases = splay_tree_new (case_compare, NULL, NULL);
p->outside_range_p = false;
p->has_default_p = false;
p->break_stmt_seen_p = false;
p->in_loop_body_p = false;
switch_stack = p;
}
void
pop_switch (void)
{
struct cp_switch *cs = switch_stack;
location_t switch_location;
switch_location = EXPR_LOC_OR_LOC (cs->switch_stmt, input_location);
const bool bool_cond_p
= (SWITCH_STMT_TYPE (cs->switch_stmt)
&& TREE_CODE (SWITCH_STMT_TYPE (cs->switch_stmt)) == BOOLEAN_TYPE);
if (!processing_template_decl)
c_do_switch_warnings (cs->cases, switch_location,
SWITCH_STMT_TYPE (cs->switch_stmt),
SWITCH_STMT_COND (cs->switch_stmt),
bool_cond_p, cs->outside_range_p);
if (cs->has_default_p
|| (!processing_template_decl
&& c_switch_covers_all_cases_p (cs->cases,
SWITCH_STMT_TYPE (cs->switch_stmt))))
SWITCH_STMT_ALL_CASES_P (cs->switch_stmt) = 1;
if (!cs->break_stmt_seen_p)
SWITCH_STMT_NO_BREAK_P (cs->switch_stmt) = 1;
gcc_assert (!cs->in_loop_body_p);
splay_tree_delete (cs->cases);
switch_stack = switch_stack->next;
free (cs);
}
void
note_break_stmt (void)
{
if (switch_stack && !switch_stack->in_loop_body_p)
switch_stack->break_stmt_seen_p = true;
}
bool
note_iteration_stmt_body_start (void)
{
if (!switch_stack)
return false;
bool ret = switch_stack->in_loop_body_p;
switch_stack->in_loop_body_p = true;
return ret;
}
void
note_iteration_stmt_body_end (bool prev)
{
if (switch_stack)
switch_stack->in_loop_body_p = prev;
}
static tree
case_conversion (tree type, tree value)
{
if (value == NULL_TREE)
return value;
value = mark_rvalue_use (value);
if (cxx_dialect >= cxx11
&& (SCOPED_ENUM_P (type)
|| !INTEGRAL_OR_UNSCOPED_ENUMERATION_TYPE_P (TREE_TYPE (value))))
{
if (INTEGRAL_OR_UNSCOPED_ENUMERATION_TYPE_P (type))
type = type_promotes_to (type);
value = (perform_implicit_conversion_flags
(type, value, tf_warning_or_error,
LOOKUP_IMPLICIT | LOOKUP_NO_NON_INTEGRAL));
}
return cxx_constant_value (value);
}
tree
finish_case_label (location_t loc, tree low_value, tree high_value)
{
tree cond, r;
cp_binding_level *p;
tree type;
if (low_value == NULL_TREE && high_value == NULL_TREE)
switch_stack->has_default_p = true;
if (processing_template_decl)
{
tree label;
label = build_decl (loc, LABEL_DECL, NULL_TREE, NULL_TREE);
return add_stmt (build_case_label (low_value, high_value, label));
}
cond = SWITCH_STMT_COND (switch_stack->switch_stmt);
if (cond && TREE_CODE (cond) == TREE_LIST)
cond = TREE_VALUE (cond);
if (!check_switch_goto (switch_stack->level))
return error_mark_node;
type = SWITCH_STMT_TYPE (switch_stack->switch_stmt);
low_value = case_conversion (type, low_value);
high_value = case_conversion (type, high_value);
r = c_add_case_label (loc, switch_stack->cases, cond, type,
low_value, high_value,
&switch_stack->outside_range_p);
for (p = current_binding_level;
p->kind != sk_function_parms;
p = p->level_chain)
p->more_cleanups_ok = 0;
return r;
}

struct typename_info {
tree scope;
tree name;
tree template_id;
bool enum_p;
bool class_p;
};
struct typename_hasher : ggc_ptr_hash<tree_node>
{
typedef typename_info *compare_type;
static hashval_t
hash (tree t)
{
hashval_t hash;
hash = (htab_hash_pointer (TYPE_CONTEXT (t))
^ htab_hash_pointer (TYPE_IDENTIFIER (t)));
return hash;
}
static bool
equal (tree t1, const typename_info *t2)
{
return (TYPE_IDENTIFIER (t1) == t2->name
&& TYPE_CONTEXT (t1) == t2->scope
&& TYPENAME_TYPE_FULLNAME (t1) == t2->template_id
&& TYPENAME_IS_ENUM_P (t1) == t2->enum_p
&& TYPENAME_IS_CLASS_P (t1) == t2->class_p);
}
};
static GTY (()) hash_table<typename_hasher> *typename_htab;
tree
build_typename_type (tree context, tree name, tree fullname,
enum tag_types tag_type)
{
tree t;
tree d;
typename_info ti;
tree *e;
hashval_t hash;
if (typename_htab == NULL)
typename_htab = hash_table<typename_hasher>::create_ggc (61);
ti.scope = FROB_CONTEXT (context);
ti.name = name;
ti.template_id = fullname;
ti.enum_p = tag_type == enum_type;
ti.class_p = (tag_type == class_type
|| tag_type == record_type
|| tag_type == union_type);
hash =  (htab_hash_pointer (ti.scope)
^ htab_hash_pointer (ti.name));
e = typename_htab->find_slot_with_hash (&ti, hash, INSERT);
if (*e)
t = *e;
else
{
t = cxx_make_type (TYPENAME_TYPE);
TYPE_CONTEXT (t) = ti.scope;
TYPENAME_TYPE_FULLNAME (t) = ti.template_id;
TYPENAME_IS_ENUM_P (t) = ti.enum_p;
TYPENAME_IS_CLASS_P (t) = ti.class_p;
d = build_decl (input_location, TYPE_DECL, name, t);
TYPE_NAME (TREE_TYPE (d)) = d;
TYPE_STUB_DECL (TREE_TYPE (d)) = d;
DECL_CONTEXT (d) = FROB_CONTEXT (context);
DECL_ARTIFICIAL (d) = 1;
*e = t;
SET_TYPE_STRUCTURAL_EQUALITY (t);
}
return t;
}
tree
make_typename_type (tree context, tree name, enum tag_types tag_type,
tsubst_flags_t complain)
{
tree fullname;
tree t;
bool want_template;
if (name == error_mark_node
|| context == NULL_TREE
|| context == error_mark_node)
return error_mark_node;
if (TYPE_P (name))
{
if (!(TYPE_LANG_SPECIFIC (name)
&& (CLASSTYPE_IS_TEMPLATE (name)
|| CLASSTYPE_USE_TEMPLATE (name))))
name = TYPE_IDENTIFIER (name);
else
name = build_nt (TEMPLATE_ID_EXPR,
CLASSTYPE_TI_TEMPLATE (name),
CLASSTYPE_TI_ARGS (name));
}
else if (TREE_CODE (name) == TYPE_DECL)
name = DECL_NAME (name);
fullname = name;
if (TREE_CODE (name) == TEMPLATE_ID_EXPR)
{
name = TREE_OPERAND (name, 0);
if (DECL_TYPE_TEMPLATE_P (name))
name = TREE_OPERAND (fullname, 0) = DECL_NAME (name);
if (TREE_CODE (name) != IDENTIFIER_NODE)
{
if (complain & tf_error)
error ("%qD is not a type", name);
return error_mark_node;
}
}
if (TREE_CODE (name) == TEMPLATE_DECL)
{
if (complain & tf_error)
error ("%qD used without template parameters", name);
return error_mark_node;
}
gcc_assert (identifier_p (name));
gcc_assert (TYPE_P (context));
if (!MAYBE_CLASS_TYPE_P (context))
{
if (complain & tf_error)
error ("%q#T is not a class", context);
return error_mark_node;
}
if (!dependent_scope_p (context))
t = lookup_field (context, name, 2, true);
else
t = NULL_TREE;
if ((!t || TREE_CODE (t) == TREE_LIST) && dependent_type_p (context))
return build_typename_type (context, name, fullname, tag_type);
want_template = TREE_CODE (fullname) == TEMPLATE_ID_EXPR;
if (!t)
{
if (complain & tf_error)
{
if (!COMPLETE_TYPE_P (context))
cxx_incomplete_type_error (NULL_TREE, context);
else
error (want_template ? G_("no class template named %q#T in %q#T")
: G_("no type named %q#T in %q#T"), name, context);
}
return error_mark_node;
}
if (want_template)
t = maybe_get_template_decl_from_type_decl (t);
if (TREE_CODE (t) == TREE_LIST)
{
if (complain & tf_error)
{
error ("lookup of %qT in %qT is ambiguous", name, context);
print_candidates (t);
}
return error_mark_node;
}
if (want_template && !DECL_TYPE_TEMPLATE_P (t))
{
if (complain & tf_error)
error ("%<typename %T::%D%> names %q#T, which is not a class template",
context, name, t);
return error_mark_node;
}
if (!want_template && TREE_CODE (t) != TYPE_DECL)
{
if (complain & tf_error)
error ("%<typename %T::%D%> names %q#T, which is not a type",
context, name, t);
return error_mark_node;
}
if (!perform_or_defer_access_check (TYPE_BINFO (context), t, t, complain))
return error_mark_node;
add_typedef_to_current_template_for_access_check (t, context, input_location);
if (want_template)
return lookup_template_class (t, TREE_OPERAND (fullname, 1),
NULL_TREE, context,
0,
complain | tf_user);
if (DECL_ARTIFICIAL (t) || !(complain & tf_keep_type_decl))
t = TREE_TYPE (t);
maybe_record_typedef_use (t);
return t;
}
tree
make_unbound_class_template (tree context, tree name, tree parm_list,
tsubst_flags_t complain)
{
tree t;
tree d;
if (TYPE_P (name))
name = TYPE_IDENTIFIER (name);
else if (DECL_P (name))
name = DECL_NAME (name);
gcc_assert (identifier_p (name));
if (!dependent_type_p (context)
|| currently_open_class (context))
{
tree tmpl = NULL_TREE;
if (MAYBE_CLASS_TYPE_P (context))
tmpl = lookup_field (context, name, 0, false);
if (tmpl && TREE_CODE (tmpl) == TYPE_DECL)
tmpl = maybe_get_template_decl_from_type_decl (tmpl);
if (!tmpl || !DECL_TYPE_TEMPLATE_P (tmpl))
{
if (complain & tf_error)
error ("no class template named %q#T in %q#T", name, context);
return error_mark_node;
}
if (parm_list
&& !comp_template_parms (DECL_TEMPLATE_PARMS (tmpl), parm_list))
{
if (complain & tf_error)
{
error ("template parameters do not match template %qD", tmpl);
inform (DECL_SOURCE_LOCATION (tmpl),
"%qD declared here", tmpl);
}
return error_mark_node;
}
if (!perform_or_defer_access_check (TYPE_BINFO (context), tmpl, tmpl,
complain))
return error_mark_node;
return tmpl;
}
t = cxx_make_type (UNBOUND_CLASS_TEMPLATE);
TYPE_CONTEXT (t) = FROB_CONTEXT (context);
TREE_TYPE (t) = NULL_TREE;
SET_TYPE_STRUCTURAL_EQUALITY (t);
d = build_decl (input_location, TEMPLATE_DECL, name, t);
TYPE_NAME (TREE_TYPE (d)) = d;
TYPE_STUB_DECL (TREE_TYPE (d)) = d;
DECL_CONTEXT (d) = FROB_CONTEXT (context);
DECL_ARTIFICIAL (d) = 1;
DECL_TEMPLATE_PARMS (d) = parm_list;
return t;
}

void
record_builtin_type (enum rid rid_index,
const char* name,
tree type)
{
tree decl = NULL_TREE;
if (name)
{
tree tname = get_identifier (name);
tree tdecl = build_decl (BUILTINS_LOCATION, TYPE_DECL, tname, type);
DECL_ARTIFICIAL (tdecl) = 1;
set_global_binding (tdecl);
decl = tdecl;
}
if ((int) rid_index < (int) RID_MAX)
if (tree rname = ridpointers[(int) rid_index])
if (!decl || DECL_NAME (decl) != rname)
{
tree rdecl = build_decl (BUILTINS_LOCATION, TYPE_DECL, rname, type);
DECL_ARTIFICIAL (rdecl) = 1;
set_global_binding (rdecl);
if (!decl)
decl = rdecl;
}
if (decl)
{
if (!TYPE_NAME (type))
TYPE_NAME (type) = decl;
debug_hooks->type_decl (decl, 0);
}
}
static void
record_unknown_type (tree type, const char* name)
{
tree decl = pushdecl (build_decl (UNKNOWN_LOCATION,
TYPE_DECL, get_identifier (name), type));
DECL_IGNORED_P (decl) = 1;
TYPE_DECL_SUPPRESS_DEBUG (decl) = 1;
TYPE_SIZE (type) = TYPE_SIZE (void_type_node);
SET_TYPE_ALIGN (type, 1);
TYPE_USER_ALIGN (type) = 0;
SET_TYPE_MODE (type, TYPE_MODE (void_type_node));
}
static void
initialize_predefined_identifiers (void)
{
struct predefined_identifier
{
const char *name; 
tree *node;  
cp_identifier_kind kind;  
};
static const predefined_identifier predefined_identifiers[] = {
{"C++", &lang_name_cplusplus, cik_normal},
{"C", &lang_name_c, cik_normal},
{"__ct ", &ctor_identifier, cik_ctor},
{"__ct_base ", &base_ctor_identifier, cik_ctor},
{"__ct_comp ", &complete_ctor_identifier, cik_ctor},
{"__dt ", &dtor_identifier, cik_dtor},
{"__dt_base ", &base_dtor_identifier, cik_dtor},
{"__dt_comp ", &complete_dtor_identifier, cik_dtor},
{"__dt_del ", &deleting_dtor_identifier, cik_dtor},
{"__conv_op ", &conv_op_identifier, cik_conv_op},
{"__in_chrg", &in_charge_identifier, cik_normal},
{"this", &this_identifier, cik_normal},
{"__delta", &delta_identifier, cik_normal},
{"__pfn", &pfn_identifier, cik_normal},
{"_vptr", &vptr_identifier, cik_normal},
{"__vtt_parm", &vtt_parm_identifier, cik_normal},
{"::", &global_identifier, cik_normal},
{"std", &std_identifier, cik_normal},
{"_GLOBAL__N_1", &anon_identifier, cik_normal},
{"auto", &auto_identifier, cik_normal},
{"decltype(auto)", &decltype_auto_identifier, cik_normal},
{"initializer_list", &init_list_identifier, cik_normal},
{NULL, NULL, cik_normal}
};
for (const predefined_identifier *pid = predefined_identifiers;
pid->name; ++pid)
{
*pid->node = get_identifier (pid->name);
if (pid->kind != cik_normal)
set_identifier_kind (*pid->node, pid->kind);
}
}
void
cxx_init_decl_processing (void)
{
tree void_ftype;
tree void_ftype_ptr;
initialize_predefined_identifiers ();
push_to_top_level ();
current_function_decl = NULL_TREE;
current_binding_level = NULL;
gcc_assert (global_namespace == NULL_TREE);
global_namespace = build_lang_decl (NAMESPACE_DECL, global_identifier,
void_type_node);
TREE_PUBLIC (global_namespace) = 1;
DECL_CONTEXT (global_namespace)
= build_translation_unit_decl (get_identifier (main_input_filename));
TRANSLATION_UNIT_WARN_EMPTY_P (DECL_CONTEXT (global_namespace))
= warn_abi && abi_version_crosses (12);
debug_hooks->register_main_translation_unit
(DECL_CONTEXT (global_namespace));
begin_scope (sk_namespace, global_namespace);
current_namespace = global_namespace;
if (flag_visibility_ms_compat)
default_visibility = VISIBILITY_HIDDEN;
current_lang_name = lang_name_c;
push_namespace (std_identifier);
std_node = current_namespace;
pop_namespace ();
flag_noexcept_type = (cxx_dialect >= cxx17);
if (!flag_new_for_scope)
warning_at (UNKNOWN_LOCATION, OPT_Wdeprecated,
"%<-fno-for-scope%> is deprecated");
if (flag_friend_injection)
warning_at (UNKNOWN_LOCATION, OPT_Wdeprecated,
"%<-ffriend-injection%> is deprecated");
c_common_nodes_and_builtins ();
integer_two_node = build_int_cst (NULL_TREE, 2);
vec_alloc (static_decls, 500);
vec_alloc (keyed_classes, 100);
record_builtin_type (RID_BOOL, "bool", boolean_type_node);
truthvalue_type_node = boolean_type_node;
truthvalue_false_node = boolean_false_node;
truthvalue_true_node = boolean_true_node;
empty_except_spec = build_tree_list (NULL_TREE, NULL_TREE);
noexcept_true_spec = build_tree_list (boolean_true_node, NULL_TREE);
noexcept_false_spec = build_tree_list (boolean_false_node, NULL_TREE);
noexcept_deferred_spec = build_tree_list (make_node (DEFERRED_NOEXCEPT),
NULL_TREE);
#if 0
record_builtin_type (RID_MAX, NULL, string_type_node);
#endif
delta_type_node = ptrdiff_type_node;
vtable_index_type = ptrdiff_type_node;
vtt_parm_type = build_pointer_type (const_ptr_type_node);
void_ftype = build_function_type_list (void_type_node, NULL_TREE);
void_ftype_ptr = build_function_type_list (void_type_node,
ptr_type_node, NULL_TREE);
void_ftype_ptr
= build_exception_variant (void_ftype_ptr, empty_except_spec);
conv_op_marker = build_lang_decl (FUNCTION_DECL, conv_op_identifier,
void_ftype);
unknown_type_node = make_node (LANG_TYPE);
record_unknown_type (unknown_type_node, "unknown type");
TREE_TYPE (unknown_type_node) = unknown_type_node;
TYPE_POINTER_TO (unknown_type_node) = unknown_type_node;
TYPE_REFERENCE_TO (unknown_type_node) = unknown_type_node;
init_list_type_node = make_node (LANG_TYPE);
record_unknown_type (init_list_type_node, "init list");
{
tree vfunc_type = make_node (FUNCTION_TYPE);
TREE_TYPE (vfunc_type) = integer_type_node;
TYPE_ARG_TYPES (vfunc_type) = NULL_TREE;
layout_type (vfunc_type);
vtable_entry_type = build_pointer_type (vfunc_type);
}
record_builtin_type (RID_MAX, "__vtbl_ptr_type", vtable_entry_type);
vtbl_type_node
= build_cplus_array_type (vtable_entry_type, NULL_TREE);
layout_type (vtbl_type_node);
vtbl_type_node = cp_build_qualified_type (vtbl_type_node, TYPE_QUAL_CONST);
record_builtin_type (RID_MAX, NULL, vtbl_type_node);
vtbl_ptr_type_node = build_pointer_type (vtable_entry_type);
layout_type (vtbl_ptr_type_node);
record_builtin_type (RID_MAX, NULL, vtbl_ptr_type_node);
push_namespace (get_identifier ("__cxxabiv1"));
abi_node = current_namespace;
pop_namespace ();
global_type_node = make_node (LANG_TYPE);
record_unknown_type (global_type_node, "global type");
any_targ_node = make_node (LANG_TYPE);
record_unknown_type (any_targ_node, "any type");
current_lang_name = lang_name_cplusplus;
if (aligned_new_threshold > 1
&& !pow2p_hwi (aligned_new_threshold))
{
error ("-faligned-new=%d is not a power of two", aligned_new_threshold);
aligned_new_threshold = 1;
}
if (aligned_new_threshold == -1)
aligned_new_threshold = (cxx_dialect >= cxx17) ? 1 : 0;
if (aligned_new_threshold == 1)
aligned_new_threshold = malloc_alignment () / BITS_PER_UNIT;
{
tree newattrs, extvisattr;
tree newtype, deltype;
tree ptr_ftype_sizetype;
tree new_eh_spec;
ptr_ftype_sizetype
= build_function_type_list (ptr_type_node, size_type_node, NULL_TREE);
if (cxx_dialect == cxx98)
{
tree bad_alloc_id;
tree bad_alloc_type_node;
tree bad_alloc_decl;
push_namespace (std_identifier);
bad_alloc_id = get_identifier ("bad_alloc");
bad_alloc_type_node = make_class_type (RECORD_TYPE);
TYPE_CONTEXT (bad_alloc_type_node) = current_namespace;
bad_alloc_decl
= create_implicit_typedef (bad_alloc_id, bad_alloc_type_node);
DECL_CONTEXT (bad_alloc_decl) = current_namespace;
pop_namespace ();
new_eh_spec
= add_exception_specifier (NULL_TREE, bad_alloc_type_node, -1);
}
else
new_eh_spec = noexcept_false_spec;
init_attributes ();
init_constraint_processing ();
extvisattr = build_tree_list (get_identifier ("externally_visible"),
NULL_TREE);
newattrs = tree_cons (get_identifier ("alloc_size"),
build_tree_list (NULL_TREE, integer_one_node),
extvisattr);
newtype = cp_build_type_attribute_variant (ptr_ftype_sizetype, newattrs);
newtype = build_exception_variant (newtype, new_eh_spec);
deltype = cp_build_type_attribute_variant (void_ftype_ptr, extvisattr);
deltype = build_exception_variant (deltype, empty_except_spec);
tree opnew = push_cp_library_fn (NEW_EXPR, newtype, 0);
DECL_IS_MALLOC (opnew) = 1;
DECL_IS_OPERATOR_NEW (opnew) = 1;
opnew = push_cp_library_fn (VEC_NEW_EXPR, newtype, 0);
DECL_IS_MALLOC (opnew) = 1;
DECL_IS_OPERATOR_NEW (opnew) = 1;
push_cp_library_fn (DELETE_EXPR, deltype, ECF_NOTHROW);
push_cp_library_fn (VEC_DELETE_EXPR, deltype, ECF_NOTHROW);
if (flag_sized_deallocation)
{
tree void_ftype_ptr_size
= build_function_type_list (void_type_node, ptr_type_node,
size_type_node, NULL_TREE);
deltype = cp_build_type_attribute_variant (void_ftype_ptr_size,
extvisattr);
deltype = build_exception_variant (deltype, empty_except_spec);
push_cp_library_fn (DELETE_EXPR, deltype, ECF_NOTHROW);
push_cp_library_fn (VEC_DELETE_EXPR, deltype, ECF_NOTHROW);
}
if (aligned_new_threshold)
{
push_namespace (std_identifier);
tree align_id = get_identifier ("align_val_t");
align_type_node = start_enum (align_id, NULL_TREE, size_type_node,
NULL_TREE, true, NULL);
pop_namespace ();
newtype = build_function_type_list (ptr_type_node, size_type_node,
align_type_node, NULL_TREE);
newtype = cp_build_type_attribute_variant (newtype, newattrs);
newtype = build_exception_variant (newtype, new_eh_spec);
opnew = push_cp_library_fn (NEW_EXPR, newtype, 0);
DECL_IS_MALLOC (opnew) = 1;
DECL_IS_OPERATOR_NEW (opnew) = 1;
opnew = push_cp_library_fn (VEC_NEW_EXPR, newtype, 0);
DECL_IS_MALLOC (opnew) = 1;
DECL_IS_OPERATOR_NEW (opnew) = 1;
deltype = build_function_type_list (void_type_node, ptr_type_node,
align_type_node, NULL_TREE);
deltype = cp_build_type_attribute_variant (deltype, extvisattr);
deltype = build_exception_variant (deltype, empty_except_spec);
push_cp_library_fn (DELETE_EXPR, deltype, ECF_NOTHROW);
push_cp_library_fn (VEC_DELETE_EXPR, deltype, ECF_NOTHROW);
if (flag_sized_deallocation)
{
deltype = build_function_type_list (void_type_node, ptr_type_node,
size_type_node, align_type_node,
NULL_TREE);
deltype = cp_build_type_attribute_variant (deltype, extvisattr);
deltype = build_exception_variant (deltype, empty_except_spec);
push_cp_library_fn (DELETE_EXPR, deltype, ECF_NOTHROW);
push_cp_library_fn (VEC_DELETE_EXPR, deltype, ECF_NOTHROW);
}
}
nullptr_type_node = make_node (NULLPTR_TYPE);
TYPE_SIZE (nullptr_type_node) = bitsize_int (GET_MODE_BITSIZE (ptr_mode));
TYPE_SIZE_UNIT (nullptr_type_node) = size_int (GET_MODE_SIZE (ptr_mode));
TYPE_UNSIGNED (nullptr_type_node) = 1;
TYPE_PRECISION (nullptr_type_node) = GET_MODE_BITSIZE (ptr_mode);
if (abi_version_at_least (9))
SET_TYPE_ALIGN (nullptr_type_node, GET_MODE_ALIGNMENT (ptr_mode));
SET_TYPE_MODE (nullptr_type_node, ptr_mode);
record_builtin_type (RID_MAX, "decltype(nullptr)", nullptr_type_node);
nullptr_node = build_int_cst (nullptr_type_node, 0);
}
abort_fndecl
= build_library_fn_ptr ("__cxa_pure_virtual", void_ftype,
ECF_NORETURN | ECF_NOTHROW | ECF_COLD);
init_class_processing ();
init_rtti_processing ();
init_template_processing ();
if (flag_exceptions)
init_exception_processing ();
if (! supports_one_only ())
flag_weak = 0;
make_fname_decl = cp_make_fname_decl;
start_fname_decls ();
if (flag_exceptions)
using_eh_for_cleanups ();
}
tree
cp_fname_init (const char* name, tree *type_p)
{
tree domain = NULL_TREE;
tree type;
tree init = NULL_TREE;
size_t length = 0;
if (name)
{
length = strlen (name);
domain = build_index_type (size_int (length));
init = build_string (length + 1, name);
}
type = cp_build_qualified_type (char_type_node, TYPE_QUAL_CONST);
type = build_cplus_array_type (type, domain);
*type_p = type;
if (init)
TREE_TYPE (init) = type;
else
init = error_mark_node;
return init;
}
static tree
cp_make_fname_decl (location_t loc, tree id, int type_dep)
{
const char *const name = (type_dep && processing_template_decl
? NULL : fname_as_string (type_dep));
tree type;
tree init = cp_fname_init (name, &type);
tree decl = build_decl (loc, VAR_DECL, id, type);
if (name)
free (CONST_CAST (char *, name));
TREE_STATIC (decl) = 1;
TREE_READONLY (decl) = 1;
DECL_ARTIFICIAL (decl) = 1;
TREE_USED (decl) = 1;
if (current_function_decl)
{
DECL_CONTEXT (decl) = current_function_decl;
decl = pushdecl_outermost_localscope (decl);
cp_finish_decl (decl, init, false, NULL_TREE,
LOOKUP_ONLYCONVERTING);
}
else
{
DECL_THIS_STATIC (decl) = true;
pushdecl_top_level_and_finish (decl, init);
}
return decl;
}
static tree
builtin_function_1 (tree decl, tree context, bool is_global)
{
tree          id = DECL_NAME (decl);
const char *name = IDENTIFIER_POINTER (id);
retrofit_lang_decl (decl);
DECL_ARTIFICIAL (decl) = 1;
SET_DECL_LANGUAGE (decl, lang_c);
DECL_VISIBILITY (decl) = VISIBILITY_DEFAULT;
DECL_VISIBILITY_SPECIFIED (decl) = 1;
DECL_CONTEXT (decl) = context;
if (name[0] != '_' || name[1] != '_')
DECL_ANTICIPATED (decl) = 1;
else if (strncmp (name + 2, "builtin_", strlen ("builtin_")) != 0)
{
size_t len = strlen (name);
if (len > strlen ("___chk")
&& memcmp (name + len - strlen ("_chk"),
"_chk", strlen ("_chk") + 1) == 0)
DECL_ANTICIPATED (decl) = 1;
}
if (is_global)
pushdecl_top_level (decl);
else
pushdecl (decl);
return decl;
}
tree
cxx_builtin_function (tree decl)
{
tree          id = DECL_NAME (decl);
const char *name = IDENTIFIER_POINTER (id);
if (name[0] != '_')
{
tree decl2 = copy_node(decl);
push_namespace (std_identifier);
builtin_function_1 (decl2, std_node, false);
pop_namespace ();
}
return builtin_function_1 (decl, NULL_TREE, false);
}
tree
cxx_builtin_function_ext_scope (tree decl)
{
tree          id = DECL_NAME (decl);
const char *name = IDENTIFIER_POINTER (id);
if (name[0] != '_')
{
tree decl2 = copy_node(decl);
push_namespace (std_identifier);
builtin_function_1 (decl2, std_node, true);
pop_namespace ();
}
return builtin_function_1 (decl, NULL_TREE, true);
}
static tree
build_library_fn (tree name, enum tree_code operator_code, tree type,
int ecf_flags)
{
tree fn = build_lang_decl (FUNCTION_DECL, name, type);
DECL_EXTERNAL (fn) = 1;
TREE_PUBLIC (fn) = 1;
DECL_ARTIFICIAL (fn) = 1;
DECL_OVERLOADED_OPERATOR_CODE_RAW (fn)
= OVL_OP_INFO (false, operator_code)->ovl_op_code;
SET_DECL_LANGUAGE (fn, lang_c);
DECL_VISIBILITY (fn) = VISIBILITY_DEFAULT;
DECL_VISIBILITY_SPECIFIED (fn) = 1;
set_call_expr_flags (fn, ecf_flags);
return fn;
}
static tree
build_cp_library_fn (tree name, enum tree_code operator_code, tree type,
int ecf_flags)
{
tree fn = build_library_fn (name, operator_code, type, ecf_flags);
DECL_CONTEXT (fn) = FROB_CONTEXT (current_namespace);
SET_DECL_LANGUAGE (fn, lang_cplusplus);
return fn;
}
tree
build_library_fn_ptr (const char* name, tree type, int ecf_flags)
{
return build_library_fn (get_identifier (name), ERROR_MARK, type, ecf_flags);
}
tree
build_cp_library_fn_ptr (const char* name, tree type, int ecf_flags)
{
return build_cp_library_fn (get_identifier (name), ERROR_MARK, type,
ecf_flags);
}
tree
push_library_fn (tree name, tree type, tree raises, int ecf_flags)
{
tree fn;
if (raises)
type = build_exception_variant (type, raises);
fn = build_library_fn (name, ERROR_MARK, type, ecf_flags);
pushdecl_top_level (fn);
return fn;
}
static tree
push_cp_library_fn (enum tree_code operator_code, tree type,
int ecf_flags)
{
tree fn = build_cp_library_fn (ovl_op_identifier (false, operator_code),
operator_code, type, ecf_flags);
pushdecl (fn);
if (flag_tm)
apply_tm_attr (fn, get_identifier ("transaction_safe"));
return fn;
}
tree
push_void_library_fn (tree name, tree parmtypes, int ecf_flags)
{
tree type = build_function_type (void_type_node, parmtypes);
return push_library_fn (name, type, NULL_TREE, ecf_flags);
}
tree
push_throw_library_fn (tree name, tree type)
{
tree fn = push_library_fn (name, type, NULL_TREE, ECF_NORETURN | ECF_COLD);
return fn;
}

void
fixup_anonymous_aggr (tree t)
{
TYPE_HAS_USER_CONSTRUCTOR (t) = 0;
TYPE_HAS_DEFAULT_CONSTRUCTOR (t) = 0;
TYPE_HAS_COPY_CTOR (t) = 0;
TYPE_HAS_CONST_COPY_CTOR (t) = 0;
TYPE_HAS_COPY_ASSIGN (t) = 0;
TYPE_HAS_CONST_COPY_ASSIGN (t) = 0;
for (tree probe, *prev_p = &TYPE_FIELDS (t); (probe = *prev_p);)
if (TREE_CODE (probe) == FUNCTION_DECL && DECL_ARTIFICIAL (probe))
*prev_p = DECL_CHAIN (probe);
else
prev_p = &DECL_CHAIN (probe);
if (TREE_CODE (t) != UNION_TYPE)
{
tree field, type;
for (field = TYPE_FIELDS (t); field; field = DECL_CHAIN (field))
if (TREE_CODE (field) == FIELD_DECL)
{
type = TREE_TYPE (field);
if (CLASS_TYPE_P (type))
{
if (TYPE_NEEDS_CONSTRUCTING (type))
error ("member %q+#D with constructor not allowed "
"in anonymous aggregate", field);
if (TYPE_HAS_NONTRIVIAL_DESTRUCTOR (type))
error ("member %q+#D with destructor not allowed "
"in anonymous aggregate", field);
if (TYPE_HAS_COMPLEX_COPY_ASSIGN (type))
error ("member %q+#D with copy assignment operator "
"not allowed in anonymous aggregate", field);
}
}
}
}
void
warn_misplaced_attr_for_class_type (source_location location,
tree class_type)
{
gcc_assert (OVERLOAD_TYPE_P (class_type));
if (warning_at (location, OPT_Wattributes,
"attribute ignored in declaration "
"of %q#T", class_type))
inform (location,
"attribute for %q#T must follow the %qs keyword",
class_type, class_key_or_enum_as_string (class_type));
}
tree
check_tag_decl (cp_decl_specifier_seq *declspecs,
bool explicit_type_instantiation_p)
{
int saw_friend = decl_spec_seq_has_spec_p (declspecs, ds_friend);
int saw_typedef = decl_spec_seq_has_spec_p (declspecs, ds_typedef);
tree declared_type = NULL_TREE;
bool error_p = false;
if (declspecs->multiple_types_p)
error ("multiple types in one declaration");
else if (declspecs->redefined_builtin_type)
{
if (!in_system_header_at (input_location))
permerror (declspecs->locations[ds_redefined_builtin_type_spec],
"redeclaration of C++ built-in type %qT",
declspecs->redefined_builtin_type);
return NULL_TREE;
}
if (declspecs->type
&& TYPE_P (declspecs->type)
&& ((TREE_CODE (declspecs->type) != TYPENAME_TYPE
&& MAYBE_CLASS_TYPE_P (declspecs->type))
|| TREE_CODE (declspecs->type) == ENUMERAL_TYPE))
declared_type = declspecs->type;
else if (declspecs->type == error_mark_node)
error_p = true;
if (declared_type == NULL_TREE && ! saw_friend && !error_p)
permerror (input_location, "declaration does not declare anything");
else if (declared_type != NULL_TREE && type_uses_auto (declared_type))
{
error_at (declspecs->locations[ds_type_spec],
"%<auto%> can only be specified for variables "
"or function declarations");
return error_mark_node;
}
else if (declared_type && RECORD_OR_UNION_CODE_P (TREE_CODE (declared_type))
&& TYPE_UNNAMED_P (declared_type))
{
if (saw_typedef)
{
error ("missing type-name in typedef-declaration");
return NULL_TREE;
}
;
SET_ANON_AGGR_TYPE_P (declared_type);
if (TREE_CODE (declared_type) != UNION_TYPE
&& !in_system_header_at (input_location))
pedwarn (input_location, OPT_Wpedantic, "ISO C++ prohibits anonymous structs");
}
else
{
if (decl_spec_seq_has_spec_p (declspecs, ds_inline))
error_at (declspecs->locations[ds_inline],
"%<inline%> can only be specified for functions");
else if (decl_spec_seq_has_spec_p (declspecs, ds_virtual))
error_at (declspecs->locations[ds_virtual],
"%<virtual%> can only be specified for functions");
else if (saw_friend
&& (!current_class_type
|| current_scope () != current_class_type))
error_at (declspecs->locations[ds_friend],
"%<friend%> can only be specified inside a class");
else if (decl_spec_seq_has_spec_p (declspecs, ds_explicit))
error_at (declspecs->locations[ds_explicit],
"%<explicit%> can only be specified for constructors");
else if (declspecs->storage_class)
error_at (declspecs->locations[ds_storage_class],
"a storage class can only be specified for objects "
"and functions");
else if (decl_spec_seq_has_spec_p (declspecs, ds_const))
error_at (declspecs->locations[ds_const],
"%<const%> can only be specified for objects and "
"functions");
else if (decl_spec_seq_has_spec_p (declspecs, ds_volatile))
error_at (declspecs->locations[ds_volatile],
"%<volatile%> can only be specified for objects and "
"functions");
else if (decl_spec_seq_has_spec_p (declspecs, ds_restrict))
error_at (declspecs->locations[ds_restrict],
"%<__restrict%> can only be specified for objects and "
"functions");
else if (decl_spec_seq_has_spec_p (declspecs, ds_thread))
error_at (declspecs->locations[ds_thread],
"%<__thread%> can only be specified for objects "
"and functions");
else if (saw_typedef)
warning_at (declspecs->locations[ds_typedef], 0,
"%<typedef%> was ignored in this declaration");
else if (decl_spec_seq_has_spec_p (declspecs,  ds_constexpr))
error_at (declspecs->locations[ds_constexpr],
"%<constexpr%> cannot be used for type declarations");
}
if (declspecs->attributes && warn_attributes && declared_type)
{
location_t loc;
if (!CLASS_TYPE_P (declared_type)
|| !CLASSTYPE_TEMPLATE_INSTANTIATION (declared_type))
loc = location_of (declared_type);
else
loc = input_location;
if (explicit_type_instantiation_p)
{
if (warning_at (loc, OPT_Wattributes,
"attribute ignored in explicit instantiation %q#T",
declared_type))
inform (loc,
"no attribute can be applied to "
"an explicit instantiation");
}
else
warn_misplaced_attr_for_class_type (loc, declared_type);
}
return declared_type;
}
tree
shadow_tag (cp_decl_specifier_seq *declspecs)
{
tree t = check_tag_decl (declspecs,
false);
if (!t)
return NULL_TREE;
if (maybe_process_partial_specialization (t) == error_mark_node)
return NULL_TREE;
if (ANON_AGGR_TYPE_P (t))
{
fixup_anonymous_aggr (t);
if (TYPE_FIELDS (t))
{
tree decl = grokdeclarator (NULL,
declspecs, NORMAL, 0, NULL);
finish_anon_union (decl);
}
}
return t;
}

tree
groktypename (cp_decl_specifier_seq *type_specifiers,
const cp_declarator *declarator,
bool is_template_arg)
{
tree attrs;
tree type;
enum decl_context context
= is_template_arg ? TEMPLATE_TYPE_ARG : TYPENAME;
attrs = type_specifiers->attributes;
type_specifiers->attributes = NULL_TREE;
type = grokdeclarator (declarator, type_specifiers, context, 0, &attrs);
if (attrs && type != error_mark_node)
{
if (CLASS_TYPE_P (type))
warning (OPT_Wattributes, "ignoring attributes applied to class type %qT "
"outside of definition", type);
else if (MAYBE_CLASS_TYPE_P (type))
warning (OPT_Wattributes, "ignoring attributes applied to dependent "
"type %qT without an associated declaration", type);
else
cplus_decl_attributes (&type, attrs, 0);
}
return type;
}
tree
start_decl (const cp_declarator *declarator,
cp_decl_specifier_seq *declspecs,
int initialized,
tree attributes,
tree prefix_attributes,
tree *pushed_scope_p)
{
tree decl;
tree context;
bool was_public;
int flags;
bool alias;
*pushed_scope_p = NULL_TREE;
if (lookup_attribute ("deprecated", attributes))
deprecated_state = DEPRECATED_SUPPRESS;
attributes = chainon (attributes, prefix_attributes);
decl = grokdeclarator (declarator, declspecs, NORMAL, initialized,
&attributes);
deprecated_state = DEPRECATED_NORMAL;
if (decl == NULL_TREE || VOID_TYPE_P (decl)
|| decl == error_mark_node)
return error_mark_node;
context = CP_DECL_CONTEXT (decl);
if (context != global_namespace)
*pushed_scope_p = push_scope (context);
if (initialized
&& TREE_CODE (decl) == TYPE_DECL)
{
error ("typedef %qD is initialized (use decltype instead)", decl);
return error_mark_node;
}
if (initialized)
{
if (! toplevel_bindings_p ()
&& DECL_EXTERNAL (decl))
warning (0, "declaration of %q#D has %<extern%> and is initialized",
decl);
DECL_EXTERNAL (decl) = 0;
if (toplevel_bindings_p ())
TREE_STATIC (decl) = 1;
}
alias = lookup_attribute ("alias", DECL_ATTRIBUTES (decl)) != 0;
if (alias && TREE_CODE (decl) == FUNCTION_DECL)
record_key_method_defined (decl);
if (TREE_CODE (decl) == TYPE_DECL
&& OVERLOAD_TYPE_P (TREE_TYPE (decl))
&& decl == TYPE_NAME (TYPE_MAIN_VARIANT (TREE_TYPE (decl))))
flags = ATTR_FLAG_TYPE_IN_PLACE;
else
flags = 0;
cplus_decl_attributes (&decl, attributes, flags);
if (initialized && DECL_DLLIMPORT_P (decl))
{
error ("definition of %q#D is marked %<dllimport%>", decl);
DECL_DLLIMPORT_P (decl) = 0;
}
if (!processing_template_decl && !DECL_DECOMPOSITION_P (decl))
maybe_apply_pragma_weak (decl);
if (TREE_CODE (decl) == FUNCTION_DECL
&& DECL_DECLARED_INLINE_P (decl)
&& DECL_UNINLINABLE (decl)
&& lookup_attribute ("noinline", DECL_ATTRIBUTES (decl)))
warning_at (DECL_SOURCE_LOCATION (decl), 0,
"inline function %qD given attribute noinline", decl);
if (TYPE_P (context) && COMPLETE_TYPE_P (complete_type (context)))
{
bool this_tmpl = (processing_template_decl
> template_class_depth (context));
if (VAR_P (decl))
{
tree field = lookup_field (context, DECL_NAME (decl), 0, false);
if (field == NULL_TREE
|| !(VAR_P (field) || variable_template_p (field)))
error ("%q+#D is not a static data member of %q#T", decl, context);
else if (variable_template_p (field)
&& (DECL_LANG_SPECIFIC (decl)
&& DECL_TEMPLATE_SPECIALIZATION (decl)))
;
else if (variable_template_p (field) && !this_tmpl)
{
error_at (DECL_SOURCE_LOCATION (decl),
"non-member-template declaration of %qD", decl);
inform (DECL_SOURCE_LOCATION (field), "does not match "
"member template declaration here");
return error_mark_node;
}
else
{
if (variable_template_p (field))
field = DECL_TEMPLATE_RESULT (field);
if (DECL_CONTEXT (field) != context)
{
if (!same_type_p (DECL_CONTEXT (field), context))
permerror (input_location, "ISO C++ does not permit %<%T::%D%> "
"to be defined as %<%T::%D%>",
DECL_CONTEXT (field), DECL_NAME (decl),
context, DECL_NAME (decl));
DECL_CONTEXT (decl) = DECL_CONTEXT (field);
}
if (initialized && DECL_INITIALIZED_IN_CLASS_P (field))
error ("duplicate initialization of %qD", decl);
field = duplicate_decls (decl, field,
false);
if (field == error_mark_node)
return error_mark_node;
else if (field)
decl = field;
}
}
else
{
tree field = check_classfn (context, decl,
this_tmpl
? current_template_parms
: NULL_TREE);
if (field && field != error_mark_node
&& duplicate_decls (decl, field,
false))
decl = field;
}
DECL_IN_AGGR_P (decl) = 0;
if (DECL_LANG_SPECIFIC (decl) && DECL_USE_TEMPLATE (decl))
{
SET_DECL_TEMPLATE_SPECIALIZATION (decl);
if (TREE_CODE (decl) == FUNCTION_DECL)
DECL_COMDAT (decl) = (TREE_PUBLIC (decl)
&& DECL_DECLARED_INLINE_P (decl));
else
DECL_COMDAT (decl) = false;
if (!initialized && processing_specialization)
DECL_EXTERNAL (decl) = 1;
}
if (DECL_EXTERNAL (decl) && ! DECL_TEMPLATE_SPECIALIZATION (decl)
&& !alias)
permerror (input_location, "declaration of %q#D outside of class is not definition",
decl);
}
was_public = TREE_PUBLIC (decl);
if (!template_parm_scope_p ()
|| !VAR_P (decl))
decl = maybe_push_decl (decl);
if (processing_template_decl)
decl = push_template_decl (decl);
if (decl == error_mark_node)
return error_mark_node;
if (VAR_P (decl)
&& DECL_NAMESPACE_SCOPE_P (decl) && !TREE_PUBLIC (decl) && !was_public
&& !DECL_THIS_STATIC (decl) && !DECL_ARTIFICIAL (decl))
{
gcc_assert (CP_TYPE_CONST_P (TREE_TYPE (decl)) || errorcount);
DECL_THIS_STATIC (decl) = 1;
}
if (current_function_decl && VAR_P (decl)
&& DECL_DECLARED_CONSTEXPR_P (current_function_decl))
{
bool ok = false;
if (CP_DECL_THREAD_LOCAL_P (decl))
error ("%qD declared %<thread_local%> in %<constexpr%> function",
decl);
else if (TREE_STATIC (decl))
error ("%qD declared %<static%> in %<constexpr%> function", decl);
else
ok = true;
if (!ok)
cp_function_chain->invalid_constexpr = true;
}
if (!processing_template_decl && VAR_P (decl))
start_decl_1 (decl, initialized);
return decl;
}
void
start_decl_1 (tree decl, bool initialized)
{
tree type;
bool complete_p;
bool aggregate_definition_p;
gcc_assert (!processing_template_decl);
if (error_operand_p (decl))
return;
gcc_assert (VAR_P (decl));
type = TREE_TYPE (decl);
complete_p = COMPLETE_TYPE_P (type);
aggregate_definition_p = MAYBE_CLASS_TYPE_P (type) && !DECL_EXTERNAL (decl);
if ((initialized || aggregate_definition_p) 
&& !complete_p
&& COMPLETE_TYPE_P (complete_type (type)))
{
complete_p = true;
cp_apply_type_quals_to_decl (cp_type_quals (type), decl);
}
if (initialized)
{
if (complete_p)
;			
else if (type_uses_auto (type))
; 			
else if (TREE_CODE (type) != ARRAY_TYPE)
{
error ("variable %q#D has initializer but incomplete type", decl);
type = TREE_TYPE (decl) = error_mark_node;
}
else if (!COMPLETE_TYPE_P (complete_type (TREE_TYPE (type))))
{
if (DECL_LANG_SPECIFIC (decl) && DECL_TEMPLATE_INFO (decl))
error ("elements of array %q#D have incomplete type", decl);
}
}
else if (aggregate_definition_p && !complete_p)
{
if (type_uses_auto (type))
gcc_assert (CLASS_PLACEHOLDER_TEMPLATE (type));
else
{
error ("aggregate %q#D has incomplete type and cannot be defined",
decl);
type = TREE_TYPE (decl) = error_mark_node;
}
}
maybe_push_cleanup_level (type);
}
static tree
grok_reference_init (tree decl, tree type, tree init, int flags)
{
if (init == NULL_TREE)
{
if ((DECL_LANG_SPECIFIC (decl) == 0
|| DECL_IN_AGGR_P (decl) == 0)
&& ! DECL_THIS_EXTERN (decl))
error ("%qD declared as reference but not initialized", decl);
return NULL_TREE;
}
if (TREE_CODE (init) == TREE_LIST)
init = build_x_compound_expr_from_list (init, ELK_INIT,
tf_warning_or_error);
tree ttype = TREE_TYPE (type);
if (TREE_CODE (ttype) != ARRAY_TYPE
&& TREE_CODE (TREE_TYPE (init)) == ARRAY_TYPE)
init = decay_conversion (init, tf_warning_or_error);
if (TREE_CODE (ttype) == ARRAY_TYPE
&& TYPE_DOMAIN (ttype) == NULL_TREE
&& (BRACE_ENCLOSED_INITIALIZER_P (init)
|| TREE_CODE (init) == STRING_CST))
{
cp_complete_array_type (&ttype, init, false);
if (ttype != TREE_TYPE (type))
type = cp_build_reference_type (ttype, TYPE_REF_IS_RVALUE (type));
}
return initialize_reference (type, init, flags,
tf_warning_or_error);
}
bool
check_array_designated_initializer (constructor_elt *ce,
unsigned HOST_WIDE_INT index)
{
if (ce->index)
{
if (ce->index == error_mark_node)
{
error ("name used in a GNU-style designated "
"initializer for an array");
return false;
}
else if (identifier_p (ce->index))
{
error ("name %qD used in a GNU-style designated "
"initializer for an array", ce->index);
return false;
}
tree ce_index = build_expr_type_conversion (WANT_INT | WANT_ENUM,
ce->index, true);
if (ce_index
&& INTEGRAL_OR_UNSCOPED_ENUMERATION_TYPE_P (TREE_TYPE (ce_index))
&& (TREE_CODE (ce_index = fold_non_dependent_expr (ce_index))
== INTEGER_CST))
{
if (wi::to_wide (ce_index) == index)
{
ce->index = ce_index;
return true;
}
else
sorry ("non-trivial designated initializers not supported");
}
else
error ("C99 designator %qE is not an integral constant-expression",
ce->index);
return false;
}
return true;
}
static void
maybe_deduce_size_from_array_init (tree decl, tree init)
{
tree type = TREE_TYPE (decl);
if (TREE_CODE (type) == ARRAY_TYPE
&& TYPE_DOMAIN (type) == NULL_TREE
&& TREE_CODE (decl) != TYPE_DECL)
{
int do_default = !DECL_EXTERNAL (decl);
tree initializer = init ? init : DECL_INITIAL (decl);
int failure = 0;
if (initializer && BRACE_ENCLOSED_INITIALIZER_P (initializer))
{
vec<constructor_elt, va_gc> *v = CONSTRUCTOR_ELTS (initializer);
constructor_elt *ce;
HOST_WIDE_INT i;
FOR_EACH_VEC_SAFE_ELT (v, i, ce)
{
if (instantiation_dependent_expression_p (ce->index))
return;
if (!check_array_designated_initializer (ce, i))
failure = 1;
}
}
if (failure)
TREE_TYPE (decl) = error_mark_node;
else
{
failure = cp_complete_array_type (&TREE_TYPE (decl), initializer,
do_default);
if (failure == 1)
{
error_at (EXPR_LOC_OR_LOC (initializer,
DECL_SOURCE_LOCATION (decl)),
"initializer fails to determine size of %qD", decl);
}
else if (failure == 2)
{
if (do_default)
{
error_at (DECL_SOURCE_LOCATION (decl),
"array size missing in %qD", decl);
}
else if (!pedantic && TREE_STATIC (decl) && !TREE_PUBLIC (decl))
DECL_EXTERNAL (decl) = 1;
}
else if (failure == 3)
{
error_at (DECL_SOURCE_LOCATION (decl),
"zero-size array %qD", decl);
}
}
cp_apply_type_quals_to_decl (cp_type_quals (TREE_TYPE (decl)), decl);
relayout_decl (decl);
}
}
static void
layout_var_decl (tree decl)
{
tree type;
type = TREE_TYPE (decl);
if (type == error_mark_node)
return;
if (!DECL_EXTERNAL (decl))
complete_type (type);
if (!DECL_SIZE (decl)
&& TREE_TYPE (decl) != error_mark_node
&& complete_or_array_type_p (type))
layout_decl (decl, 0);
if (!DECL_EXTERNAL (decl) && DECL_SIZE (decl) == NULL_TREE)
{
error_at (DECL_SOURCE_LOCATION (decl),
"storage size of %qD isn%'t known", decl);
TREE_TYPE (decl) = error_mark_node;
}
#if 0
else if (!DECL_EXTERNAL (decl) && MAYBE_CLASS_TYPE_P (ttype))
note_debug_info_needed (ttype);
if (TREE_STATIC (decl) && DECL_CLASS_SCOPE_P (decl))
note_debug_info_needed (DECL_CONTEXT (decl));
#endif
if ((DECL_EXTERNAL (decl) || TREE_STATIC (decl))
&& DECL_SIZE (decl) != NULL_TREE
&& ! TREE_CONSTANT (DECL_SIZE (decl)))
{
if (TREE_CODE (DECL_SIZE (decl)) == INTEGER_CST)
constant_expression_warning (DECL_SIZE (decl));
else
{
error_at (DECL_SOURCE_LOCATION (decl),
"storage size of %qD isn%'t constant", decl);
TREE_TYPE (decl) = error_mark_node;
}
}
}
void
maybe_commonize_var (tree decl)
{
if ((TREE_STATIC (decl)
&& ! DECL_ARTIFICIAL (decl)
&& DECL_FUNCTION_SCOPE_P (decl)
&& vague_linkage_p (DECL_CONTEXT (decl)))
|| (TREE_PUBLIC (decl) && DECL_INLINE_VAR_P (decl)))
{
if (flag_weak)
{
comdat_linkage (decl);
}
else
{
if (DECL_INITIAL (decl) == NULL_TREE
|| DECL_INITIAL (decl) == error_mark_node)
{
TREE_PUBLIC (decl) = 1;
DECL_COMMON (decl) = 1;
}
else
{
TREE_PUBLIC (decl) = 0;
DECL_COMMON (decl) = 0;
const char *msg;
if (DECL_INLINE_VAR_P (decl))
msg = G_("sorry: semantics of inline variable "
"%q#D are wrong (you%'ll wind up with "
"multiple copies)");
else
msg = G_("sorry: semantics of inline function "
"static data %q#D are wrong (you%'ll wind "
"up with multiple copies)");
if (warning_at (DECL_SOURCE_LOCATION (decl), 0,
msg, decl))
inform (DECL_SOURCE_LOCATION (decl),
"you can work around this by removing the initializer");
}
}
}
}
bool
check_for_uninitialized_const_var (tree decl, bool constexpr_context_p,
tsubst_flags_t complain)
{
tree type = strip_array_types (TREE_TYPE (decl));
if (VAR_P (decl)
&& TREE_CODE (type) != REFERENCE_TYPE
&& (constexpr_context_p
|| CP_TYPE_CONST_P (type) || var_in_constexpr_fn (decl))
&& !DECL_NONTRIVIALLY_INITIALIZED_P (decl))
{
tree field = default_init_uninitialized_part (type);
if (!field)
return true;
if (!constexpr_context_p)
{
if (CP_TYPE_CONST_P (type))
{
if (complain & tf_error)
permerror (DECL_SOURCE_LOCATION (decl),
"uninitialized const %qD", decl);
}
else
{
if (!is_instantiation_of_constexpr (current_function_decl)
&& (complain & tf_error))
error_at (DECL_SOURCE_LOCATION (decl),
"uninitialized variable %qD in %<constexpr%> "
"function", decl);
cp_function_chain->invalid_constexpr = true;
}
}
else if (complain & tf_error)
error_at (DECL_SOURCE_LOCATION (decl),
"uninitialized variable %qD in %<constexpr%> context",
decl);
if (CLASS_TYPE_P (type) && (complain & tf_error))
{
tree defaulted_ctor;
inform (DECL_SOURCE_LOCATION (TYPE_MAIN_DECL (type)),
"%q#T has no user-provided default constructor", type);
defaulted_ctor = in_class_defaulted_default_constructor (type);
if (defaulted_ctor)
inform (DECL_SOURCE_LOCATION (defaulted_ctor),
"constructor is not user-provided because it is "
"explicitly defaulted in the class body");
inform (DECL_SOURCE_LOCATION (field),
"and the implicitly-defined constructor does not "
"initialize %q#D", field);
}
return false;
}
return true;
}

struct reshape_iter
{
constructor_elt *cur;
constructor_elt *end;
};
static tree reshape_init_r (tree, reshape_iter *, bool, tsubst_flags_t);
tree
next_initializable_field (tree field)
{
while (field
&& (TREE_CODE (field) != FIELD_DECL
|| DECL_UNNAMED_BIT_FIELD (field)
|| (DECL_ARTIFICIAL (field)
&& !(cxx_dialect >= cxx17 && DECL_FIELD_IS_BASE (field)))))
field = DECL_CHAIN (field);
return field;
}
bool
is_direct_enum_init (tree type, tree init)
{
if (cxx_dialect >= cxx17
&& TREE_CODE (type) == ENUMERAL_TYPE
&& ENUM_FIXED_UNDERLYING_TYPE_P (type)
&& TREE_CODE (init) == CONSTRUCTOR
&& CONSTRUCTOR_IS_DIRECT_INIT (init)
&& CONSTRUCTOR_NELTS (init) == 1)
return true;
return false;
}
static tree
reshape_init_array_1 (tree elt_type, tree max_index, reshape_iter *d,
tsubst_flags_t complain)
{
tree new_init;
bool sized_array_p = (max_index && TREE_CONSTANT (max_index));
unsigned HOST_WIDE_INT max_index_cst = 0;
unsigned HOST_WIDE_INT index;
new_init = build_constructor (init_list_type_node, NULL);
if (sized_array_p)
{
if (integer_all_onesp (max_index))
return new_init;
if (tree_fits_uhwi_p (max_index))
max_index_cst = tree_to_uhwi (max_index);
else
max_index_cst = tree_to_uhwi (fold_convert (size_type_node, max_index));
}
for (index = 0;
d->cur != d->end && (!sized_array_p || index <= max_index_cst);
++index)
{
tree elt_init;
constructor_elt *old_cur = d->cur;
check_array_designated_initializer (d->cur, index);
elt_init = reshape_init_r (elt_type, d, false,
complain);
if (elt_init == error_mark_node)
return error_mark_node;
CONSTRUCTOR_APPEND_ELT (CONSTRUCTOR_ELTS (new_init),
size_int (index), elt_init);
if (!TREE_CONSTANT (elt_init))
TREE_CONSTANT (new_init) = false;
if (d->cur == old_cur && !sized_array_p)
break;
}
return new_init;
}
static tree
reshape_init_array (tree type, reshape_iter *d, tsubst_flags_t complain)
{
tree max_index = NULL_TREE;
gcc_assert (TREE_CODE (type) == ARRAY_TYPE);
if (TYPE_DOMAIN (type))
max_index = array_type_nelts (type);
return reshape_init_array_1 (TREE_TYPE (type), max_index, d, complain);
}
static tree
reshape_init_vector (tree type, reshape_iter *d, tsubst_flags_t complain)
{
tree max_index = NULL_TREE;
gcc_assert (VECTOR_TYPE_P (type));
if (COMPOUND_LITERAL_P (d->cur->value))
{
tree value = d->cur->value;
if (!same_type_p (TREE_TYPE (value), type))
{
if (complain & tf_error)
error ("invalid type %qT as initializer for a vector of type %qT",
TREE_TYPE (d->cur->value), type);
value = error_mark_node;
}
++d->cur;
return value;
}
if (VECTOR_TYPE_P (type))
max_index = size_int (TYPE_VECTOR_SUBPARTS (type) - 1);
return reshape_init_array_1 (TREE_TYPE (type), max_index, d, complain);
}
static tree
reshape_init_class (tree type, reshape_iter *d, bool first_initializer_p,
tsubst_flags_t complain)
{
tree field;
tree new_init;
gcc_assert (CLASS_TYPE_P (type));
new_init = build_constructor (init_list_type_node, NULL);
field = next_initializable_field (TYPE_FIELDS (type));
if (!field)
{
if (!first_initializer_p)
{
if (complain & tf_error)
error ("initializer for %qT must be brace-enclosed", type);
return error_mark_node;
}
return new_init;
}
while (d->cur != d->end)
{
tree field_init;
constructor_elt *old_cur = d->cur;
if (d->cur->index)
{
if (d->cur->index == error_mark_node)
return error_mark_node;
if (TREE_CODE (d->cur->index) == FIELD_DECL)
{
if (field != d->cur->index)
{
tree id = DECL_NAME (d->cur->index);
gcc_assert (id);
gcc_checking_assert (d->cur->index
== get_class_binding (type, id, false));
field = d->cur->index;
}
}
else if (TREE_CODE (d->cur->index) == IDENTIFIER_NODE)
field = get_class_binding (type, d->cur->index, false);
else
{
if (complain & tf_error)
error ("%<[%E] =%> used in a GNU-style designated initializer"
" for class %qT", d->cur->index, type);
return error_mark_node;
}
if (!field || TREE_CODE (field) != FIELD_DECL)
{
if (complain & tf_error)
error ("%qT has no non-static data member named %qD", type,
d->cur->index);
return error_mark_node;
}
}
if (!field)
break;
field_init = reshape_init_r (TREE_TYPE (field), d,
false, complain);
if (field_init == error_mark_node)
return error_mark_node;
if (d->cur == old_cur && d->cur->index)
{
if (complain & tf_error)
error ("invalid initializer for %q#D", field);
return error_mark_node;
}
CONSTRUCTOR_APPEND_ELT (CONSTRUCTOR_ELTS (new_init), field, field_init);
if (TREE_CODE (type) == UNION_TYPE)
break;
field = next_initializable_field (DECL_CHAIN (field));
}
return new_init;
}
static bool
has_designator_problem (reshape_iter *d, tsubst_flags_t complain)
{
if (d->cur->index)
{
if (complain & tf_error)
error ("C99 designator %qE outside aggregate initializer",
d->cur->index);
else
return true;
}
return false;
}
static tree
reshape_init_r (tree type, reshape_iter *d, bool first_initializer_p,
tsubst_flags_t complain)
{
tree init = d->cur->value;
if (error_operand_p (init))
return error_mark_node;
if (first_initializer_p && !CP_AGGREGATE_TYPE_P (type)
&& has_designator_problem (d, complain))
return error_mark_node;
if (TREE_CODE (type) == COMPLEX_TYPE)
{
d->cur++;
if (BRACE_ENCLOSED_INITIALIZER_P (init))
{
if (CONSTRUCTOR_NELTS (init) > 2)
{
if (complain & tf_error)
error ("too many initializers for %qT", type);
else
return error_mark_node;
}
}
else if (first_initializer_p && d->cur != d->end)
{
vec<constructor_elt, va_gc> *v = 0;
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE, init);
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE, d->cur->value);
if (has_designator_problem (d, complain))
return error_mark_node;
d->cur++;
init = build_constructor (init_list_type_node, v);
}
return init;
}
if (!CP_AGGREGATE_TYPE_P (type))
{
if (TREE_CODE (init) == CONSTRUCTOR
&& !CONSTRUCTOR_IS_DIRECT_INIT (init)
&& BRACE_ENCLOSED_INITIALIZER_P (init))  
{
if (SCALAR_TYPE_P (type))
{
if (cxx_dialect < cxx11
|| CONSTRUCTOR_NELTS (init) > 0)
{
if (complain & tf_error)
error ("braces around scalar initializer for type %qT",
type);
init = error_mark_node;
}
}
else
maybe_warn_cpp0x (CPP0X_INITIALIZER_LISTS);
}
d->cur++;
return init;
}
if (cxx_dialect >= cxx11 && (CLASS_TYPE_P (type) || VECTOR_TYPE_P (type))
&& first_initializer_p
&& d->end - d->cur == 1
&& reference_related_p (type, TREE_TYPE (init)))
{
d->cur++;
return init;
}
if (TREE_CODE (init) != CONSTRUCTOR
&& !first_initializer_p
&& (same_type_ignoring_top_level_qualifiers_p (type, TREE_TYPE (init))
|| can_convert_arg (type, TREE_TYPE (init), init, LOOKUP_NORMAL,
complain)))
{
d->cur++;
return init;
}
if (TREE_CODE (type) == ARRAY_TYPE
&& char_type_p (TYPE_MAIN_VARIANT (TREE_TYPE (type))))
{
tree str_init = init;
if (!first_initializer_p
&& TREE_CODE (str_init) == CONSTRUCTOR
&& CONSTRUCTOR_NELTS (str_init) == 1)
{
str_init = (*CONSTRUCTOR_ELTS (str_init))[0].value;
}
if (TREE_CODE (str_init) == STRING_CST)
{
if (has_designator_problem (d, complain))
return error_mark_node;
d->cur++;
return str_init;
}
}
if (!first_initializer_p)
{
if (TREE_CODE (init) == CONSTRUCTOR)
{
tree init_type = TREE_TYPE (init);
if (init_type && TYPE_PTRMEMFUNC_P (init_type))
;
else if (COMPOUND_LITERAL_P (init)
|| same_type_ignoring_top_level_qualifiers_p (type,
init_type))
{
++d->cur;
gcc_assert (!BRACE_ENCLOSED_INITIALIZER_P (init));
return init;
}
else
{
++d->cur;
gcc_assert (BRACE_ENCLOSED_INITIALIZER_P (init));
return reshape_init (type, init, complain);
}
}
if (complain & tf_warning)
warning (OPT_Wmissing_braces,
"missing braces around initializer for %qT",
type);
}
if (CLASS_TYPE_P (type))
return reshape_init_class (type, d, first_initializer_p, complain);
else if (TREE_CODE (type) == ARRAY_TYPE)
return reshape_init_array (type, d, complain);
else if (VECTOR_TYPE_P (type))
return reshape_init_vector (type, d, complain);
else
gcc_unreachable();
}
tree
reshape_init (tree type, tree init, tsubst_flags_t complain)
{
vec<constructor_elt, va_gc> *v;
reshape_iter d;
tree new_init;
gcc_assert (BRACE_ENCLOSED_INITIALIZER_P (init));
v = CONSTRUCTOR_ELTS (init);
if (vec_safe_is_empty (v))
return init;
if (is_direct_enum_init (type, init))
{
tree elt = CONSTRUCTOR_ELT (init, 0)->value;
type = cv_unqualified (type);
if (check_narrowing (ENUM_UNDERLYING_TYPE (type), elt, complain))
{
warning_sentinel w (warn_useless_cast);
warning_sentinel w2 (warn_ignored_qualifiers);
return cp_build_c_cast (type, elt, tf_warning_or_error);
}
else
return error_mark_node;
}
d.cur = &(*v)[0];
d.end = d.cur + v->length ();
new_init = reshape_init_r (type, &d, true, complain);
if (new_init == error_mark_node)
return error_mark_node;
if (d.cur != d.end)
{
if (complain & tf_error)
error ("too many initializers for %qT", type);
return error_mark_node;
}
if (CONSTRUCTOR_IS_DIRECT_INIT (init)
&& BRACE_ENCLOSED_INITIALIZER_P (new_init))
CONSTRUCTOR_IS_DIRECT_INIT (new_init) = true;
return new_init;
}
bool
check_array_initializer (tree decl, tree type, tree init)
{
tree element_type = TREE_TYPE (type);
if (!COMPLETE_TYPE_P (complete_type (element_type)))
{
if (decl)
error_at (DECL_SOURCE_LOCATION (decl),
"elements of array %q#D have incomplete type", decl);
else
error ("elements of array %q#T have incomplete type", type);
return true;
}
if (init && !decl
&& ((COMPLETE_TYPE_P (type) && !TREE_CONSTANT (TYPE_SIZE (type)))
|| !TREE_CONSTANT (TYPE_SIZE (element_type))))
{
error ("variable-sized compound literal");
return true;
}
return false;
}
static tree
build_aggr_init_full_exprs (tree decl, tree init, int flags)
{
gcc_assert (stmts_are_full_exprs_p ());
return build_aggr_init (decl, init, flags, tf_warning_or_error);
}
static tree
check_initializer (tree decl, tree init, int flags, vec<tree, va_gc> **cleanups)
{
tree type = TREE_TYPE (decl);
tree init_code = NULL;
tree core_type;
TREE_TYPE (decl) = type = complete_type (TREE_TYPE (decl));
if (DECL_HAS_VALUE_EXPR_P (decl))
{
gcc_assert (init == NULL_TREE);
return NULL_TREE;
}
if (type == error_mark_node)
return NULL_TREE;
if (TREE_CODE (type) == ARRAY_TYPE)
{
if (check_array_initializer (decl, type, init))
return NULL_TREE;
}
else if (!COMPLETE_TYPE_P (type))
{
error_at (DECL_SOURCE_LOCATION (decl),
"%q#D has incomplete type", decl);
TREE_TYPE (decl) = error_mark_node;
return NULL_TREE;
}
else
gcc_assert (TREE_CONSTANT (TYPE_SIZE (type)));
if (init && BRACE_ENCLOSED_INITIALIZER_P (init))
{
int init_len = CONSTRUCTOR_NELTS (init);
if (SCALAR_TYPE_P (type))
{
if (init_len == 0)
{
maybe_warn_cpp0x (CPP0X_INITIALIZER_LISTS);
init = build_zero_init (type, NULL_TREE, false);
}
else if (init_len != 1 && TREE_CODE (type) != COMPLEX_TYPE)
{
error_at (EXPR_LOC_OR_LOC (init, DECL_SOURCE_LOCATION (decl)),
"scalar object %qD requires one element in "
"initializer", decl);
TREE_TYPE (decl) = error_mark_node;
return NULL_TREE;
}
}
}
if (TREE_CODE (decl) == CONST_DECL)
{
gcc_assert (TREE_CODE (type) != REFERENCE_TYPE);
DECL_INITIAL (decl) = init;
gcc_assert (init != NULL_TREE);
init = NULL_TREE;
}
else if (!init && DECL_REALLY_EXTERN (decl))
;
else if (init || type_build_ctor_call (type)
|| TREE_CODE (type) == REFERENCE_TYPE)
{
if (TREE_CODE (type) == REFERENCE_TYPE)
{
init = grok_reference_init (decl, type, init, flags);
flags |= LOOKUP_ALREADY_DIGESTED;
}
else if (!init)
check_for_uninitialized_const_var (decl, false,
tf_warning_or_error);
else if (BRACE_ENCLOSED_INITIALIZER_P (init))
{
if (is_std_init_list (type))
{
init = perform_implicit_conversion (type, init,
tf_warning_or_error);
flags |= LOOKUP_ALREADY_DIGESTED;
}
else if (TYPE_NON_AGGREGATE_CLASS (type))
{
if (cxx_dialect == cxx98)
error_at (EXPR_LOC_OR_LOC (init, DECL_SOURCE_LOCATION (decl)),
"in C++98 %qD must be initialized by "
"constructor, not by %<{...}%>",
decl);
}
else if (VECTOR_TYPE_P (type) && TYPE_VECTOR_OPAQUE (type))
{
error ("opaque vector types cannot be initialized");
init = error_mark_node;
}
else
{
init = reshape_init (type, init, tf_warning_or_error);
flags |= LOOKUP_NO_NARROWING;
}
}
else if (TREE_CODE (init) == TREE_LIST
&& TREE_TYPE (init) != unknown_type_node
&& !MAYBE_CLASS_TYPE_P (type))
{
gcc_assert (TREE_CODE (decl) != RESULT_DECL);
init = build_x_compound_expr_from_list (init, ELK_INIT,
tf_warning_or_error);
}
maybe_deduce_size_from_array_init (decl, init);
type = TREE_TYPE (decl);
if (type == error_mark_node)
return NULL_TREE;
if (((type_build_ctor_call (type) || CLASS_TYPE_P (type))
&& !(flags & LOOKUP_ALREADY_DIGESTED)
&& !(init && BRACE_ENCLOSED_INITIALIZER_P (init)
&& CP_AGGREGATE_TYPE_P (type)
&& (CLASS_TYPE_P (type)
|| !TYPE_NEEDS_CONSTRUCTING (type)
|| type_has_extended_temps (type))))
|| (DECL_DECOMPOSITION_P (decl) && TREE_CODE (type) == ARRAY_TYPE))
{
init_code = build_aggr_init_full_exprs (decl, init, flags);
if (TREE_SIDE_EFFECTS (init_code))
DECL_NONTRIVIALLY_INITIALIZED_P (decl) = true;
while (TREE_CODE (init_code) == EXPR_STMT
|| TREE_CODE (init_code) == CONVERT_EXPR)
init_code = TREE_OPERAND (init_code, 0);
if (TREE_CODE (init_code) == INIT_EXPR)
{
init = TREE_OPERAND (init_code, 1);
init_code = NULL_TREE;
flags |= LOOKUP_ALREADY_DIGESTED;
}
else if (DECL_DECLARED_CONSTEXPR_P (decl))
{
if (CLASS_TYPE_P (type)
&& (!init || TREE_CODE (init) == TREE_LIST))
{
init = build_functional_cast (type, init, tf_none);
if (TREE_CODE (init) == TARGET_EXPR)
TARGET_EXPR_DIRECT_INIT_P (init) = true;
}
init_code = NULL_TREE;
}
else
init = NULL_TREE;
}
if (init && TREE_CODE (init) != TREE_VEC)
{
gcc_assert (stmts_are_full_exprs_p ());
init_code = store_init_value (decl, init, cleanups, flags);
if (pedantic && TREE_CODE (type) == ARRAY_TYPE
&& DECL_INITIAL (decl)
&& TREE_CODE (DECL_INITIAL (decl)) == STRING_CST
&& PAREN_STRING_LITERAL_P (DECL_INITIAL (decl)))
warning_at (EXPR_LOC_OR_LOC (DECL_INITIAL (decl),
DECL_SOURCE_LOCATION (decl)),
0, "array %qD initialized by parenthesized "
"string literal %qE",
decl, DECL_INITIAL (decl));
init = NULL;
}
}
else
{
if (CLASS_TYPE_P (core_type = strip_array_types (type))
&& (CLASSTYPE_READONLY_FIELDS_NEED_INIT (core_type)
|| CLASSTYPE_REF_FIELDS_NEED_INIT (core_type)))
diagnose_uninitialized_cst_or_ref_member (core_type, false,
true);
check_for_uninitialized_const_var (decl, false,
tf_warning_or_error);
}
if (init && init != error_mark_node)
init_code = build2 (INIT_EXPR, type, decl, init);
if (init_code)
{
DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P (decl) = false;
TREE_CONSTANT (decl) = false;
}
if (init_code
&& (DECL_IN_AGGR_P (decl)
&& DECL_INITIALIZED_IN_CLASS_P (decl)
&& !DECL_VAR_DECLARED_INLINE_P (decl)))
{
static int explained = 0;
if (cxx_dialect < cxx11)
error ("initializer invalid for static member with constructor");
else if (cxx_dialect < cxx17)
error ("non-constant in-class initialization invalid for static "
"member %qD", decl);
else
error ("non-constant in-class initialization invalid for non-inline "
"static member %qD", decl);
if (!explained)
{
inform (input_location,
"(an out of class initialization is required)");
explained = 1;
}
return NULL_TREE;
}
return init_code;
}
static void
make_rtl_for_nonlocal_decl (tree decl, tree init, const char* asmspec)
{
int toplev = toplevel_bindings_p ();
int defer_p;
if (asmspec)
{
if (VAR_P (decl) && DECL_REGISTER (decl))
{
set_user_assembler_name (decl, asmspec);
DECL_HARD_REGISTER (decl) = 1;
}
else
{
if (TREE_CODE (decl) == FUNCTION_DECL
&& DECL_BUILT_IN_CLASS (decl) == BUILT_IN_NORMAL)
set_builtin_user_assembler_name (decl, asmspec);
set_user_assembler_name (decl, asmspec);
}
}
if (!VAR_P (decl))
{
rest_of_decl_compilation (decl, toplev, at_eof);
return;
}
if (DECL_LANG_SPECIFIC (decl) && DECL_IN_AGGR_P (decl))
{
gcc_assert (TREE_STATIC (decl));
if (init == NULL_TREE)
gcc_assert (DECL_EXTERNAL (decl)
|| !TREE_PUBLIC (decl)
|| DECL_INLINE_VAR_P (decl));
}
if (DECL_FUNCTION_SCOPE_P (decl) && !TREE_STATIC (decl))
return;
defer_p = ((DECL_FUNCTION_SCOPE_P (decl)
&& !var_in_maybe_constexpr_fn (decl))
|| DECL_VIRTUAL_P (decl));
if (DECL_LANG_SPECIFIC (decl)
&& DECL_IMPLICIT_INSTANTIATION (decl))
defer_p = 1;
if (!defer_p)
rest_of_decl_compilation (decl, toplev, at_eof);
}
static tree
wrap_cleanups_r (tree *stmt_p, int *walk_subtrees, void *data)
{
if (TYPE_P (*stmt_p)
|| TREE_CODE (*stmt_p) == CLEANUP_POINT_EXPR)
{
*walk_subtrees = 0;
return NULL_TREE;
}
if (TREE_CODE (*stmt_p) == TARGET_EXPR)
{
tree guard = (tree)data;
tree tcleanup = TARGET_EXPR_CLEANUP (*stmt_p);
tcleanup = build2 (TRY_CATCH_EXPR, void_type_node, tcleanup, guard);
TRY_CATCH_IS_CLEANUP (tcleanup) = 1;
TARGET_EXPR_CLEANUP (*stmt_p) = tcleanup;
}
return NULL_TREE;
}
static void
wrap_temporary_cleanups (tree init, tree guard)
{
cp_walk_tree_without_duplicates (&init, wrap_cleanups_r, (void *)guard);
}
static void
initialize_local_var (tree decl, tree init)
{
tree type = TREE_TYPE (decl);
tree cleanup;
int already_used;
gcc_assert (VAR_P (decl)
|| TREE_CODE (decl) == RESULT_DECL);
gcc_assert (!TREE_STATIC (decl));
if (DECL_SIZE (decl) == NULL_TREE)
{
DECL_INITIAL (decl) = NULL_TREE;
TREE_ADDRESSABLE (decl) = TREE_USED (decl);
return;
}
if (type == error_mark_node)
return;
already_used = TREE_USED (decl) || TREE_USED (type);
if (TREE_USED (type))
DECL_READ_P (decl) = 1;
cleanup = cxx_maybe_build_cleanup (decl, tf_warning_or_error);
if (init)
{
tree rinit = (TREE_CODE (init) == INIT_EXPR
? TREE_OPERAND (init, 1) : NULL_TREE);
if (rinit && !TREE_SIDE_EFFECTS (rinit))
{
gcc_assert (TREE_OPERAND (init, 0) == decl);
DECL_INITIAL (decl) = rinit;
if (warn_init_self && TREE_CODE (type) == REFERENCE_TYPE)
{
STRIP_NOPS (rinit);
if (rinit == decl)
warning_at (DECL_SOURCE_LOCATION (decl),
OPT_Winit_self,
"reference %qD is initialized with itself", decl);
}
}
else
{
int saved_stmts_are_full_exprs_p;
if (cleanup && TREE_CODE (type) != ARRAY_TYPE)
wrap_temporary_cleanups (init, cleanup);
gcc_assert (building_stmt_list_p ());
saved_stmts_are_full_exprs_p = stmts_are_full_exprs_p ();
current_stmt_tree ()->stmts_are_full_exprs_p = 1;
finish_expr_stmt (init);
current_stmt_tree ()->stmts_are_full_exprs_p =
saved_stmts_are_full_exprs_p;
}
}
if (TYPE_NEEDS_CONSTRUCTING (type)
&& ! already_used
&& TYPE_HAS_TRIVIAL_DESTRUCTOR (type)
&& DECL_NAME (decl))
TREE_USED (decl) = 0;
else if (already_used)
TREE_USED (decl) = 1;
if (cleanup)
finish_decl_cleanup (decl, cleanup);
}
void
initialize_artificial_var (tree decl, vec<constructor_elt, va_gc> *v)
{
tree init;
gcc_assert (DECL_ARTIFICIAL (decl));
init = build_constructor (TREE_TYPE (decl), v);
gcc_assert (TREE_CODE (init) == CONSTRUCTOR);
DECL_INITIAL (decl) = init;
DECL_INITIALIZED_P (decl) = 1;
determine_visibility (decl);
layout_var_decl (decl);
maybe_commonize_var (decl);
make_rtl_for_nonlocal_decl (decl, init, NULL);
}
static bool
value_dependent_init_p (tree init)
{
if (TREE_CODE (init) == TREE_LIST)
return any_value_dependent_elements_p (init);
else if (TREE_CODE (init) == CONSTRUCTOR)
{
if (dependent_type_p (TREE_TYPE (init)))
return true;
vec<constructor_elt, va_gc> *elts;
size_t nelts;
size_t i;
elts = CONSTRUCTOR_ELTS (init);
nelts = vec_safe_length (elts);
for (i = 0; i < nelts; ++i)
if (value_dependent_init_p ((*elts)[i].value))
return true;
}
else
return value_dependent_expression_p (init);
return false;
}
static inline bool
is_concept_var (tree decl)
{
return (VAR_P (decl)
&& DECL_LANG_SPECIFIC (decl)
&& DECL_DECLARED_CONCEPT_P (decl));
}
static tree
notice_forced_label_r (tree *tp, int *walk_subtrees, void *)
{
if (TYPE_P (*tp))
*walk_subtrees = 0;
if (TREE_CODE (*tp) == LABEL_DECL)
cfun->has_forced_label_in_static = 1;
return NULL_TREE;
}
void
cp_finish_decl (tree decl, tree init, bool init_const_expr_p,
tree asmspec_tree, int flags)
{
tree type;
vec<tree, va_gc> *cleanups = NULL;
const char *asmspec = NULL;
int was_readonly = 0;
bool var_definition_p = false;
tree auto_node;
if (decl == error_mark_node)
return;
else if (! decl)
{
if (init)
error ("assignment (not initialization) in declaration");
return;
}
gcc_assert (TREE_CODE (decl) != RESULT_DECL);
gcc_assert (TREE_CODE (decl) != PARM_DECL);
type = TREE_TYPE (decl);
if (type == error_mark_node)
return;
if (VAR_P (decl) && DECL_REGISTER (decl) && asmspec_tree == NULL_TREE)
{
if (cxx_dialect >= cxx17)
pedwarn (DECL_SOURCE_LOCATION (decl), OPT_Wregister,
"ISO C++17 does not allow %<register%> storage "
"class specifier");
else
warning_at (DECL_SOURCE_LOCATION (decl), OPT_Wregister,
"%<register%> storage class specifier used");
}
if (at_namespace_scope_p ())
asmspec_tree = maybe_apply_renaming_pragma (decl, asmspec_tree);
if (asmspec_tree && asmspec_tree != error_mark_node)
asmspec = TREE_STRING_POINTER (asmspec_tree);
if (current_class_type
&& CP_DECL_CONTEXT (decl) == current_class_type
&& TYPE_BEING_DEFINED (current_class_type)
&& !CLASSTYPE_TEMPLATE_INSTANTIATION (current_class_type)
&& (DECL_INITIAL (decl) || init))
DECL_INITIALIZED_IN_CLASS_P (decl) = 1;
if (TREE_CODE (decl) != FUNCTION_DECL
&& (auto_node = type_uses_auto (type)))
{
tree d_init;
if (init == NULL_TREE)
{
if (DECL_LANG_SPECIFIC (decl)
&& DECL_TEMPLATE_INSTANTIATION (decl)
&& !DECL_TEMPLATE_INSTANTIATED (decl))
{
instantiate_decl (decl, true, false);
return;
}
gcc_assert (CLASS_PLACEHOLDER_TEMPLATE (auto_node));
}
d_init = init;
if (d_init)
{
if (TREE_CODE (d_init) == TREE_LIST
&& !CLASS_PLACEHOLDER_TEMPLATE (auto_node))
d_init = build_x_compound_expr_from_list (d_init, ELK_INIT,
tf_warning_or_error);
d_init = resolve_nondeduced_context (d_init, tf_warning_or_error);
}
enum auto_deduction_context adc = adc_variable_type;
if (VAR_P (decl) && DECL_DECOMPOSITION_P (decl))
adc = adc_decomp_type;
type = TREE_TYPE (decl) = do_auto_deduction (type, d_init, auto_node,
tf_warning_or_error, adc,
NULL_TREE, flags);
if (type == error_mark_node)
return;
if (TREE_CODE (type) == FUNCTION_TYPE)
{
error ("initializer for %<decltype(auto) %D%> has function type "
"(did you forget the %<()%> ?)", decl);
TREE_TYPE (decl) = error_mark_node;
return;
}
cp_apply_type_quals_to_decl (cp_type_quals (type), decl);
}
if (ensure_literal_type_for_constexpr_object (decl) == error_mark_node)
{
DECL_DECLARED_CONSTEXPR_P (decl) = 0;
if (VAR_P (decl) && DECL_CLASS_SCOPE_P (decl))
{
init = NULL_TREE;
DECL_EXTERNAL (decl) = 1;
}
}
if (VAR_P (decl)
&& DECL_CLASS_SCOPE_P (decl)
&& DECL_INITIALIZED_IN_CLASS_P (decl))
check_static_variable_definition (decl, type);
if (init && TREE_CODE (decl) == FUNCTION_DECL)
{
tree clone;
if (init == ridpointers[(int)RID_DELETE])
{
DECL_DELETED_FN (decl) = 1;
DECL_DECLARED_INLINE_P (decl) = 1;
DECL_INITIAL (decl) = error_mark_node;
FOR_EACH_CLONE (clone, decl)
{
DECL_DELETED_FN (clone) = 1;
DECL_DECLARED_INLINE_P (clone) = 1;
DECL_INITIAL (clone) = error_mark_node;
}
init = NULL_TREE;
}
else if (init == ridpointers[(int)RID_DEFAULT])
{
if (defaultable_fn_check (decl))
DECL_DEFAULTED_FN (decl) = 1;
else
DECL_INITIAL (decl) = NULL_TREE;
}
}
if (init && VAR_P (decl))
{
DECL_NONTRIVIALLY_INITIALIZED_P (decl) = 1;
if (TREE_CODE (type) == REFERENCE_TYPE)
init_const_expr_p = potential_constant_expression (init);
if (init_const_expr_p)
{
DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P (decl) = 1;
if (decl_maybe_constant_var_p (decl)
&& TREE_CODE (type) != REFERENCE_TYPE)
TREE_CONSTANT (decl) = 1;
}
}
if (processing_template_decl)
{
bool type_dependent_p;
if (at_function_scope_p ())
add_decl_expr (decl);
type_dependent_p = dependent_type_p (type);
if (check_for_bare_parameter_packs (init))
{
init = NULL_TREE;
DECL_INITIAL (decl) = NULL_TREE;
}
bool dep_init = false;
if (!VAR_P (decl) || type_dependent_p)
;
else if (!init && is_concept_var (decl))
{
error ("variable concept has no initializer");
init = boolean_true_node;
}
else if (init
&& init_const_expr_p
&& TREE_CODE (type) != REFERENCE_TYPE
&& decl_maybe_constant_var_p (decl)
&& !(dep_init = value_dependent_init_p (init)))
{
tree init_code;
cleanups = make_tree_vector ();
init_code = check_initializer (decl, init, flags, &cleanups);
if (init_code == NULL_TREE)
init = NULL_TREE;
release_tree_vector (cleanups);
}
else if (!DECL_PRETTY_FUNCTION_P (decl))
{
maybe_deduce_size_from_array_init (decl, init);
if (init && TREE_CODE (init) == TREE_LIST && TREE_CHAIN (init)
&& !MAYBE_CLASS_TYPE_P (type))
init = build_x_compound_expr_from_list (init, ELK_INIT,
tf_warning_or_error);
}
if (init)
{
if (TREE_CODE (init) == TREE_LIST)
lookup_list_keep (init, true);
DECL_INITIAL (decl) = init;
}
if (dep_init)
{
retrofit_lang_decl (decl);
SET_DECL_DEPENDENT_INIT_P (decl, true);
}
return;
}
if (init && TREE_CODE (decl) == FIELD_DECL)
DECL_INITIAL (decl) = init;
if (TREE_CODE (decl) == TYPE_DECL)
{
if (type != error_mark_node
&& MAYBE_CLASS_TYPE_P (type) && DECL_NAME (decl))
{
if (TREE_TYPE (DECL_NAME (decl)) && TREE_TYPE (decl) != type)
warning (0, "shadowing previous type declaration of %q#D", decl);
set_identifier_type_value (DECL_NAME (decl), decl);
}
if (TYPE_MAIN_DECL (TREE_TYPE (decl)) == decl
&& !COMPLETE_TYPE_P (TREE_TYPE (decl)))
TYPE_DECL_SUPPRESS_DEBUG (decl) = 1;
rest_of_decl_compilation (decl, DECL_FILE_SCOPE_P (decl),
at_eof);
return;
}
if (! DECL_EXTERNAL (decl)
&& TREE_READONLY (decl)
&& TREE_CODE (type) == REFERENCE_TYPE)
{
was_readonly = 1;
TREE_READONLY (decl) = 0;
}
if (VAR_P (decl))
{
if (DECL_FUNCTION_SCOPE_P (decl)
&& TREE_STATIC (decl)
&& !DECL_ARTIFICIAL (decl))
{
push_local_name (decl);
if ((DECL_CONSTRUCTOR_P (current_function_decl)
|| DECL_DESTRUCTOR_P (current_function_decl))
&& init)
{
walk_tree (&init, notice_forced_label_r, NULL, NULL);
add_local_decl (cfun, decl);
}
varpool_node::get_create (decl);
}
if (!DECL_INITIALIZED_P (decl)
&& (!DECL_EXTERNAL (decl) || init))
{
cleanups = make_tree_vector ();
init = check_initializer (decl, init, flags, &cleanups);
if (TREE_STATIC (decl) && !DECL_INITIAL (decl))
DECL_INITIAL (decl) = build_zero_init (TREE_TYPE (decl),
NULL_TREE,
true);
DECL_INITIALIZED_P (decl) = 1;
if (!DECL_EXTERNAL (decl))
var_definition_p = true;
}
else if (TREE_CODE (type) == ARRAY_TYPE)
layout_type (type);
if (TREE_STATIC (decl)
&& !at_function_scope_p ()
&& current_function_decl == NULL)
record_types_used_by_current_var_decl (decl);
}
if (DECL_FUNCTION_SCOPE_P (decl))
{
if (VAR_P (decl)
&& DECL_SIZE (decl)
&& !TREE_CONSTANT (DECL_SIZE (decl))
&& STATEMENT_LIST_HAS_LABEL (cur_stmt_list))
{
tree bind;
bind = build3 (BIND_EXPR, void_type_node, NULL, NULL, NULL);
TREE_SIDE_EFFECTS (bind) = 1;
add_stmt (bind);
BIND_EXPR_BODY (bind) = push_stmt_list ();
}
add_decl_expr (decl);
}
if (VAR_OR_FUNCTION_DECL_P (decl))
{
if (VAR_P (decl))
{
layout_var_decl (decl);
maybe_commonize_var (decl);
}
determine_visibility (decl);
if (var_definition_p && TREE_STATIC (decl))
{
if (init)
{
if (TREE_READONLY (decl))
TREE_READONLY (decl) = 0;
was_readonly = 0;
}
else if (was_readonly)
TREE_READONLY (decl) = 1;
if (TYPE_HAS_NONTRIVIAL_DESTRUCTOR (type))
TREE_READONLY (decl) = 0;
}
make_rtl_for_nonlocal_decl (decl, init, asmspec);
abstract_virtuals_error (decl, type);
if (TREE_TYPE (decl) == error_mark_node)
;
else if (TREE_CODE (decl) == FUNCTION_DECL)
{
if (init)
{
if (init == ridpointers[(int)RID_DEFAULT])
{
if (DECL_DELETED_FN (decl))
maybe_explain_implicit_delete (decl);
else if (DECL_INITIAL (decl) == error_mark_node)
synthesize_method (decl);
}
else
error ("function %q#D is initialized like a variable", decl);
}
}
else if (DECL_EXTERNAL (decl)
&& ! (DECL_LANG_SPECIFIC (decl)
&& DECL_NOT_REALLY_EXTERN (decl)))
{
if (init)
DECL_INITIAL (decl) = init;
}
else if (DECL_FUNCTION_SCOPE_P (decl) && !TREE_STATIC (decl))
initialize_local_var (decl, init);
else if (var_definition_p && TREE_STATIC (decl))
expand_static_init (decl, init);
}
if (cleanups)
{
unsigned i; tree t;
FOR_EACH_VEC_ELT (*cleanups, i, t)
push_cleanup (decl, t, false);
release_tree_vector (cleanups);
}
if (was_readonly)
TREE_READONLY (decl) = 1;
if (flag_openmp
&& VAR_P (decl)
&& lookup_attribute ("omp declare target implicit",
DECL_ATTRIBUTES (decl)))
{
DECL_ATTRIBUTES (decl)
= remove_attribute ("omp declare target implicit",
DECL_ATTRIBUTES (decl));
complete_type (TREE_TYPE (decl));
if (!cp_omp_mappable_type (TREE_TYPE (decl)))
error ("%q+D in declare target directive does not have mappable type",
decl);
else if (!lookup_attribute ("omp declare target",
DECL_ATTRIBUTES (decl))
&& !lookup_attribute ("omp declare target link",
DECL_ATTRIBUTES (decl)))
DECL_ATTRIBUTES (decl)
= tree_cons (get_identifier ("omp declare target"),
NULL_TREE, DECL_ATTRIBUTES (decl));
}
invoke_plugin_callbacks (PLUGIN_FINISH_DECL, decl);
}
static tree
find_decomp_class_base (location_t loc, tree type, tree ret)
{
bool member_seen = false;
for (tree field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
if (TREE_CODE (field) != FIELD_DECL
|| DECL_ARTIFICIAL (field)
|| DECL_UNNAMED_BIT_FIELD (field))
continue;
else if (ret)
return type;
else if (ANON_AGGR_TYPE_P (TREE_TYPE (field)))
{
if (TREE_CODE (TREE_TYPE (field)) == RECORD_TYPE)
error_at (loc, "cannot decompose class type %qT because it has an "
"anonymous struct member", type);
else
error_at (loc, "cannot decompose class type %qT because it has an "
"anonymous union member", type);
inform (DECL_SOURCE_LOCATION (field), "declared here");
return error_mark_node;
}
else if (!accessible_p (type, field, true))
{
error_at (loc, "cannot decompose inaccessible member %qD of %qT",
field, type);
inform (DECL_SOURCE_LOCATION (field),
TREE_PRIVATE (field)
? G_("declared private here")
: G_("declared protected here"));
return error_mark_node;
}
else
member_seen = true;
tree base_binfo, binfo;
tree orig_ret = ret;
int i;
if (member_seen)
ret = type;
for (binfo = TYPE_BINFO (type), i = 0;
BINFO_BASE_ITERATE (binfo, i, base_binfo); i++)
{
tree t = find_decomp_class_base (loc, TREE_TYPE (base_binfo), ret);
if (t == error_mark_node)
return error_mark_node;
if (t != NULL_TREE && t != ret)
{
if (ret == type)
{
error_at (loc, "cannot decompose class type %qT: both it and "
"its base class %qT have non-static data members",
type, t);
return error_mark_node;
}
else if (orig_ret != NULL_TREE)
return t;
else if (ret != NULL_TREE)
{
error_at (loc, "cannot decompose class type %qT: its base "
"classes %qT and %qT have non-static data "
"members", type, ret, t);
return error_mark_node;
}
else
ret = t;
}
}
return ret;
}
static tree
get_tuple_size (tree type)
{
tree args = make_tree_vec (1);
TREE_VEC_ELT (args, 0) = type;
tree inst = lookup_template_class (get_identifier ("tuple_size"), args,
NULL_TREE,
std_node,
false, tf_none);
inst = complete_type (inst);
if (inst == error_mark_node || !COMPLETE_TYPE_P (inst))
return NULL_TREE;
tree val = lookup_qualified_name (inst, get_identifier ("value"),
false, false);
if (TREE_CODE (val) == VAR_DECL || TREE_CODE (val) == CONST_DECL)
val = maybe_constant_value (val);
if (TREE_CODE (val) == INTEGER_CST)
return val;
else
return error_mark_node;
}
static tree
get_tuple_element_type (tree type, unsigned i)
{
tree args = make_tree_vec (2);
TREE_VEC_ELT (args, 0) = build_int_cst (integer_type_node, i);
TREE_VEC_ELT (args, 1) = type;
tree inst = lookup_template_class (get_identifier ("tuple_element"), args,
NULL_TREE,
std_node,
false,
tf_warning_or_error);
return make_typename_type (inst, get_identifier ("type"),
none_type, tf_warning_or_error);
}
static tree
get_tuple_decomp_init (tree decl, unsigned i)
{
tree get_id = get_identifier ("get");
tree targs = make_tree_vec (1);
TREE_VEC_ELT (targs, 0) = build_int_cst (integer_type_node, i);
tree etype = TREE_TYPE (decl);
tree e = convert_from_reference (decl);
if (TREE_CODE (etype) != REFERENCE_TYPE
|| TYPE_REF_IS_RVALUE (etype))
e = move (e);
tree fns = lookup_qualified_name (TREE_TYPE (e), get_id,
false, false);
bool use_member_get = false;
for (lkp_iterator iter (MAYBE_BASELINK_FUNCTIONS (fns)); iter; ++iter)
{
tree fn = *iter;
if (TREE_CODE (fn) == TEMPLATE_DECL)
{
tree tparms = DECL_TEMPLATE_PARMS (fn);
tree parm = TREE_VEC_ELT (INNERMOST_TEMPLATE_PARMS (tparms), 0);
if (TREE_CODE (TREE_VALUE (parm)) == PARM_DECL)
{
use_member_get = true;
break;
}
}
}
if (use_member_get)
{
fns = lookup_template_function (fns, targs);
return build_new_method_call (e, fns, NULL,
NULL_TREE, LOOKUP_NORMAL,
NULL, tf_warning_or_error);
}
else
{
vec<tree,va_gc> *args = make_tree_vector_single (e);
fns = lookup_template_function (get_id, targs);
fns = perform_koenig_lookup (fns, args, tf_warning_or_error);
return finish_call_expr (fns, &args, false,
true, tf_warning_or_error);
}
}
static GTY((cache)) tree_cache_map *decomp_type_table;
static void
store_decomp_type (tree v, tree t)
{
if (!decomp_type_table)
decomp_type_table = tree_cache_map::create_ggc (13);
decomp_type_table->put (v, t);
}
tree
lookup_decomp_type (tree v)
{
return *decomp_type_table->get (v);
}
void
cp_maybe_mangle_decomp (tree decl, tree first, unsigned int count)
{
if (!processing_template_decl
&& !error_operand_p (decl)
&& DECL_NAMESPACE_SCOPE_P (decl))
{
auto_vec<tree, 16> v;
v.safe_grow (count);
tree d = first;
for (unsigned int i = 0; i < count; i++, d = DECL_CHAIN (d))
v[count - i - 1] = d;
SET_DECL_ASSEMBLER_NAME (decl, mangle_decomp (decl, v));
maybe_apply_pragma_weak (decl);
}
}
void
cp_finish_decomp (tree decl, tree first, unsigned int count)
{
if (error_operand_p (decl))
{
error_out:
while (count--)
{
TREE_TYPE (first) = error_mark_node;
if (DECL_HAS_VALUE_EXPR_P (first))
{
SET_DECL_VALUE_EXPR (first, NULL_TREE);
DECL_HAS_VALUE_EXPR_P (first) = 0;
}
first = DECL_CHAIN (first);
}
if (DECL_P (decl) && DECL_NAMESPACE_SCOPE_P (decl))
SET_DECL_ASSEMBLER_NAME (decl, get_identifier ("<decomp>"));
return;
}
location_t loc = DECL_SOURCE_LOCATION (decl);
if (type_dependent_expression_p (decl)
|| (!processing_template_decl
&& type_uses_auto (TREE_TYPE (decl))))
{
for (unsigned int i = 0; i < count; i++)
{
if (!DECL_HAS_VALUE_EXPR_P (first))
{
tree v = build_nt (ARRAY_REF, decl,
size_int (count - i - 1),
NULL_TREE, NULL_TREE);
SET_DECL_VALUE_EXPR (first, v);
DECL_HAS_VALUE_EXPR_P (first) = 1;
}
if (processing_template_decl)
fit_decomposition_lang_decl (first, decl);
first = DECL_CHAIN (first);
}
return;
}
auto_vec<tree, 16> v;
v.safe_grow (count);
tree d = first;
for (unsigned int i = 0; i < count; i++, d = DECL_CHAIN (d))
{
v[count - i - 1] = d;
fit_decomposition_lang_decl (d, decl);
}
tree type = TREE_TYPE (decl);
tree dexp = decl;
if (TREE_CODE (type) == REFERENCE_TYPE)
{
dexp = convert_from_reference (dexp);
type = complete_type (TREE_TYPE (type));
if (type == error_mark_node)
goto error_out;
if (!COMPLETE_TYPE_P (type))
{
error_at (loc, "structured binding refers to incomplete type %qT",
type);
goto error_out;
}
}
tree eltype = NULL_TREE;
unsigned HOST_WIDE_INT eltscnt = 0;
if (TREE_CODE (type) == ARRAY_TYPE)
{
tree nelts;
nelts = array_type_nelts_top (type);
if (nelts == error_mark_node)
goto error_out;
if (!tree_fits_uhwi_p (nelts))
{
error_at (loc, "cannot decompose variable length array %qT", type);
goto error_out;
}
eltscnt = tree_to_uhwi (nelts);
if (count != eltscnt)
{
cnt_mismatch:
if (count > eltscnt)
error_n (loc, count,
"%u name provided for structured binding",
"%u names provided for structured binding", count);
else
error_n (loc, count,
"only %u name provided for structured binding",
"only %u names provided for structured binding", count);
inform_n (loc, eltscnt,
"while %qT decomposes into %wu element",
"while %qT decomposes into %wu elements",
type, eltscnt);
goto error_out;
}
eltype = TREE_TYPE (type);
for (unsigned int i = 0; i < count; i++)
{
TREE_TYPE (v[i]) = eltype;
layout_decl (v[i], 0);
if (processing_template_decl)
continue;
tree t = unshare_expr (dexp);
t = build4_loc (DECL_SOURCE_LOCATION (v[i]), ARRAY_REF,
eltype, t, size_int (i), NULL_TREE,
NULL_TREE);
SET_DECL_VALUE_EXPR (v[i], t);
DECL_HAS_VALUE_EXPR_P (v[i]) = 1;
}
}
else if (TREE_CODE (type) == COMPLEX_TYPE)
{
eltscnt = 2;
if (count != eltscnt)
goto cnt_mismatch;
eltype = cp_build_qualified_type (TREE_TYPE (type), TYPE_QUALS (type));
for (unsigned int i = 0; i < count; i++)
{
TREE_TYPE (v[i]) = eltype;
layout_decl (v[i], 0);
if (processing_template_decl)
continue;
tree t = unshare_expr (dexp);
t = build1_loc (DECL_SOURCE_LOCATION (v[i]),
i ? IMAGPART_EXPR : REALPART_EXPR, eltype,
t);
SET_DECL_VALUE_EXPR (v[i], t);
DECL_HAS_VALUE_EXPR_P (v[i]) = 1;
}
}
else if (TREE_CODE (type) == VECTOR_TYPE)
{
if (!TYPE_VECTOR_SUBPARTS (type).is_constant (&eltscnt))
{
error_at (loc, "cannot decompose variable length vector %qT", type);
goto error_out;
}
if (count != eltscnt)
goto cnt_mismatch;
eltype = cp_build_qualified_type (TREE_TYPE (type), TYPE_QUALS (type));
for (unsigned int i = 0; i < count; i++)
{
TREE_TYPE (v[i]) = eltype;
layout_decl (v[i], 0);
if (processing_template_decl)
continue;
tree t = unshare_expr (dexp);
convert_vector_to_array_for_subscript (DECL_SOURCE_LOCATION (v[i]),
&t, size_int (i));
t = build4_loc (DECL_SOURCE_LOCATION (v[i]), ARRAY_REF,
eltype, t, size_int (i), NULL_TREE,
NULL_TREE);
SET_DECL_VALUE_EXPR (v[i], t);
DECL_HAS_VALUE_EXPR_P (v[i]) = 1;
}
}
else if (tree tsize = get_tuple_size (type))
{
if (tsize == error_mark_node)
{
error_at (loc, "%<std::tuple_size<%T>::value%> is not an integral "
"constant expression", type);
goto error_out;
}
if (!tree_fits_uhwi_p (tsize))
{
error_n (loc, count,
"%u name provided for structured binding",
"%u names provided for structured binding", count);
inform (loc, "while %qT decomposes into %E elements",
type, tsize);
goto error_out;
}
eltscnt = tree_to_uhwi (tsize);
if (count != eltscnt)
goto cnt_mismatch;
int save_read = DECL_READ_P (decl);	
for (unsigned i = 0; i < count; ++i)
{
location_t sloc = input_location;
location_t dloc = DECL_SOURCE_LOCATION (v[i]);
input_location = dloc;
tree init = get_tuple_decomp_init (decl, i);
tree eltype = (init == error_mark_node ? error_mark_node
: get_tuple_element_type (type, i));
input_location = sloc;
if (init == error_mark_node || eltype == error_mark_node)
{
inform (dloc, "in initialization of structured binding "
"variable %qD", v[i]);
goto error_out;
}
store_decomp_type (v[i], eltype);
eltype = cp_build_reference_type (eltype, !lvalue_p (init));
TREE_TYPE (v[i]) = eltype;
layout_decl (v[i], 0);
if (DECL_HAS_VALUE_EXPR_P (v[i]))
{
SET_DECL_VALUE_EXPR (v[i], NULL_TREE);
DECL_HAS_VALUE_EXPR_P (v[i]) = 0;
}
if (!processing_template_decl)
cp_finish_decl (v[i], init, false,
NULL_TREE, LOOKUP_NORMAL);
}
DECL_READ_P (decl) = save_read;
}
else if (TREE_CODE (type) == UNION_TYPE)
{
error_at (loc, "cannot decompose union type %qT", type);
goto error_out;
}
else if (!CLASS_TYPE_P (type))
{
error_at (loc, "cannot decompose non-array non-class type %qT", type);
goto error_out;
}
else if (LAMBDA_TYPE_P (type))
{
error_at (loc, "cannot decompose lambda closure type %qT", type);
goto error_out;
}
else if (processing_template_decl && !COMPLETE_TYPE_P (type))
pedwarn (loc, 0, "structured binding refers to incomplete class type %qT",
type);
else
{
tree btype = find_decomp_class_base (loc, type, NULL_TREE);
if (btype == error_mark_node)
goto error_out;
else if (btype == NULL_TREE)
{
error_at (loc, "cannot decompose class type %qT without non-static "
"data members", type);
goto error_out;
}
for (tree field = TYPE_FIELDS (btype); field; field = TREE_CHAIN (field))
if (TREE_CODE (field) != FIELD_DECL
|| DECL_ARTIFICIAL (field)
|| DECL_UNNAMED_BIT_FIELD (field))
continue;
else
eltscnt++;
if (count != eltscnt)
goto cnt_mismatch;
tree t = dexp;
if (type != btype)
{
t = convert_to_base (t, btype, true,
false, tf_warning_or_error);
type = btype;
}
unsigned int i = 0;
for (tree field = TYPE_FIELDS (btype); field; field = TREE_CHAIN (field))
if (TREE_CODE (field) != FIELD_DECL
|| DECL_ARTIFICIAL (field)
|| DECL_UNNAMED_BIT_FIELD (field))
continue;
else
{
tree tt = finish_non_static_data_member (field, unshare_expr (t),
NULL_TREE);
if (REFERENCE_REF_P (tt))
tt = TREE_OPERAND (tt, 0);
TREE_TYPE (v[i]) = TREE_TYPE (tt);
layout_decl (v[i], 0);
if (!processing_template_decl)
{
SET_DECL_VALUE_EXPR (v[i], tt);
DECL_HAS_VALUE_EXPR_P (v[i]) = 1;
}
i++;
}
}
if (processing_template_decl)
{
for (unsigned int i = 0; i < count; i++)
if (!DECL_HAS_VALUE_EXPR_P (v[i]))
{
tree a = build_nt (ARRAY_REF, decl, size_int (i),
NULL_TREE, NULL_TREE);
SET_DECL_VALUE_EXPR (v[i], a);
DECL_HAS_VALUE_EXPR_P (v[i]) = 1;
}
}
}
static tree
declare_global_var (tree name, tree type)
{
tree decl;
push_to_top_level ();
decl = build_decl (input_location, VAR_DECL, name, type);
TREE_PUBLIC (decl) = 1;
DECL_EXTERNAL (decl) = 1;
DECL_ARTIFICIAL (decl) = 1;
DECL_CONTEXT (decl) = FROB_CONTEXT (global_namespace);
decl = pushdecl (decl);
cp_finish_decl (decl, NULL_TREE, false, NULL_TREE, 0);
pop_from_top_level ();
return decl;
}
static tree
get_atexit_fn_ptr_type (void)
{
tree fn_type;
if (!atexit_fn_ptr_type_node)
{
tree arg_type;
if (flag_use_cxa_atexit 
&& !targetm.cxx.use_atexit_for_cxa_atexit ())
arg_type = ptr_type_node;
else
arg_type = NULL_TREE;
fn_type = build_function_type_list (void_type_node,
arg_type, NULL_TREE);
atexit_fn_ptr_type_node = build_pointer_type (fn_type);
}
return atexit_fn_ptr_type_node;
}
static tree
get_atexit_node (void)
{
tree atexit_fndecl;
tree fn_type;
tree fn_ptr_type;
const char *name;
bool use_aeabi_atexit;
if (atexit_node)
return atexit_node;
if (flag_use_cxa_atexit && !targetm.cxx.use_atexit_for_cxa_atexit ())
{
tree argtype0, argtype1, argtype2;
use_aeabi_atexit = targetm.cxx.use_aeabi_atexit ();
fn_ptr_type = get_atexit_fn_ptr_type ();
argtype2 = ptr_type_node;
if (use_aeabi_atexit)
{
argtype1 = fn_ptr_type;
argtype0 = ptr_type_node;
}
else
{
argtype1 = ptr_type_node;
argtype0 = fn_ptr_type;
}
fn_type = build_function_type_list (integer_type_node,
argtype0, argtype1, argtype2,
NULL_TREE);
if (use_aeabi_atexit)
name = "__aeabi_atexit";
else
name = "__cxa_atexit";
}
else
{
fn_ptr_type = get_atexit_fn_ptr_type ();
fn_type = build_function_type_list (integer_type_node,
fn_ptr_type, NULL_TREE);
name = "atexit";
}
push_lang_context (lang_name_c);
atexit_fndecl = build_library_fn_ptr (name, fn_type, ECF_LEAF | ECF_NOTHROW);
mark_used (atexit_fndecl);
pop_lang_context ();
atexit_node = decay_conversion (atexit_fndecl, tf_warning_or_error);
return atexit_node;
}
static tree
get_thread_atexit_node (void)
{
tree fn_type = build_function_type_list (integer_type_node,
get_atexit_fn_ptr_type (),
ptr_type_node, ptr_type_node,
NULL_TREE);
tree atexit_fndecl = build_library_fn_ptr ("__cxa_thread_atexit", fn_type,
ECF_LEAF | ECF_NOTHROW);
return decay_conversion (atexit_fndecl, tf_warning_or_error);
}
static tree
get_dso_handle_node (void)
{
if (dso_handle_node)
return dso_handle_node;
dso_handle_node = declare_global_var (get_identifier ("__dso_handle"),
ptr_type_node);
#ifdef HAVE_GAS_HIDDEN
if (dso_handle_node != error_mark_node)
{
DECL_VISIBILITY (dso_handle_node) = VISIBILITY_HIDDEN;
DECL_VISIBILITY_SPECIFIED (dso_handle_node) = 1;
}
#endif
return dso_handle_node;
}
static GTY(()) int start_cleanup_cnt;
static tree
start_cleanup_fn (void)
{
char name[32];
tree fntype;
tree fndecl;
bool use_cxa_atexit = flag_use_cxa_atexit
&& !targetm.cxx.use_atexit_for_cxa_atexit ();
push_to_top_level ();
push_lang_context (lang_name_c);
sprintf (name, "__tcf_%d", start_cleanup_cnt++);
fntype = TREE_TYPE (get_atexit_fn_ptr_type ());
fndecl = build_lang_decl (FUNCTION_DECL, get_identifier (name), fntype);
TREE_PUBLIC (fndecl) = 0;
DECL_ARTIFICIAL (fndecl) = 1;
DECL_DECLARED_INLINE_P (fndecl) = 1;
DECL_INTERFACE_KNOWN (fndecl) = 1;
if (use_cxa_atexit)
{
tree parmdecl = cp_build_parm_decl (fndecl, NULL_TREE, ptr_type_node);
TREE_USED (parmdecl) = 1;
DECL_READ_P (parmdecl) = 1;
DECL_ARGUMENTS (fndecl) = parmdecl;
}
pushdecl (fndecl);
start_preparsed_function (fndecl, NULL_TREE, SF_PRE_PARSED);
pop_lang_context ();
return current_function_decl;
}
static void
end_cleanup_fn (void)
{
expand_or_defer_fn (finish_function (false));
pop_from_top_level ();
}
tree
register_dtor_fn (tree decl)
{
tree cleanup;
tree addr;
tree compound_stmt;
tree fcall;
tree type;
bool ob_parm, dso_parm, use_dtor;
tree arg0, arg1, arg2;
tree atex_node;
type = TREE_TYPE (decl);
if (TYPE_HAS_TRIVIAL_DESTRUCTOR (type))
return void_node;
dso_parm = (flag_use_cxa_atexit
&& !targetm.cxx.use_atexit_for_cxa_atexit ());
ob_parm = (CP_DECL_THREAD_LOCAL_P (decl) || dso_parm);
use_dtor = ob_parm && CLASS_TYPE_P (type);
if (use_dtor)
{
cleanup = get_class_binding (type, complete_dtor_identifier);
perform_or_defer_access_check (TYPE_BINFO (type), cleanup, cleanup,
tf_warning_or_error);
}
else
{
build_cleanup (decl);
cleanup = start_cleanup_fn ();
push_deferring_access_checks (dk_no_check);
fcall = build_cleanup (decl);
pop_deferring_access_checks ();
compound_stmt = begin_compound_stmt (BCS_FN_BODY);
finish_expr_stmt (fcall);
finish_compound_stmt (compound_stmt);
end_cleanup_fn ();
}
mark_used (cleanup);
cleanup = build_address (cleanup);
if (CP_DECL_THREAD_LOCAL_P (decl))
atex_node = get_thread_atexit_node ();
else
atex_node = get_atexit_node ();
if (use_dtor)
{
cleanup = build_nop (get_atexit_fn_ptr_type (), cleanup);
mark_used (decl);
addr = build_address (decl);
addr = build_nop (ptr_type_node, addr);
}
else
addr = null_pointer_node;
if (dso_parm)
arg2 = cp_build_addr_expr (get_dso_handle_node (),
tf_warning_or_error);
else if (ob_parm)
arg2 = null_pointer_node;
else
arg2 = NULL_TREE;
if (ob_parm)
{
if (!CP_DECL_THREAD_LOCAL_P (decl)
&& targetm.cxx.use_aeabi_atexit ())
{
arg1 = cleanup;
arg0 = addr;
}
else
{
arg1 = addr;
arg0 = cleanup;
}
}
else
{
arg0 = cleanup;
arg1 = NULL_TREE;
}
return cp_build_function_call_nary (atex_node, tf_warning_or_error,
arg0, arg1, arg2, NULL_TREE);
}
static void
expand_static_init (tree decl, tree init)
{
gcc_assert (VAR_P (decl));
gcc_assert (TREE_STATIC (decl));
if (TYPE_HAS_TRIVIAL_DESTRUCTOR (TREE_TYPE (decl)))
{
cxx_maybe_build_cleanup (decl, tf_warning_or_error);
if (!init)
return;
}
if (CP_DECL_THREAD_LOCAL_P (decl) && DECL_GNU_TLS_P (decl)
&& !DECL_FUNCTION_SCOPE_P (decl))
{
if (init)
error ("non-local variable %qD declared %<__thread%> "
"needs dynamic initialization", decl);
else
error ("non-local variable %qD declared %<__thread%> "
"has a non-trivial destructor", decl);
static bool informed;
if (!informed)
{
inform (DECL_SOURCE_LOCATION (decl),
"C++11 %<thread_local%> allows dynamic initialization "
"and destruction");
informed = true;
}
return;
}
if (DECL_FUNCTION_SCOPE_P (decl))
{
tree if_stmt = NULL_TREE, inner_if_stmt = NULL_TREE;
tree then_clause = NULL_TREE, inner_then_clause = NULL_TREE;
tree guard, guard_addr;
tree flag, begin;
bool thread_guard = (flag_threadsafe_statics
&& !CP_DECL_THREAD_LOCAL_P (decl));
guard = get_guard (decl);
if_stmt = begin_if_stmt ();
finish_if_stmt_cond (get_guard_cond (guard, thread_guard), if_stmt);
then_clause = begin_compound_stmt (BCS_NO_SCOPE);
if (thread_guard)
{
tree vfntype = NULL_TREE;
tree acquire_name, release_name, abort_name;
tree acquire_fn, release_fn, abort_fn;
guard_addr = build_address (guard);
acquire_name = get_identifier ("__cxa_guard_acquire");
release_name = get_identifier ("__cxa_guard_release");
abort_name = get_identifier ("__cxa_guard_abort");
acquire_fn = get_global_binding (acquire_name);
release_fn = get_global_binding (release_name);
abort_fn = get_global_binding (abort_name);
if (!acquire_fn)
acquire_fn = push_library_fn
(acquire_name, build_function_type_list (integer_type_node,
TREE_TYPE (guard_addr),
NULL_TREE),
NULL_TREE, ECF_NOTHROW | ECF_LEAF);
if (!release_fn || !abort_fn)
vfntype = build_function_type_list (void_type_node,
TREE_TYPE (guard_addr),
NULL_TREE);
if (!release_fn)
release_fn = push_library_fn (release_name, vfntype, NULL_TREE,
ECF_NOTHROW | ECF_LEAF);
if (!abort_fn)
abort_fn = push_library_fn (abort_name, vfntype, NULL_TREE,
ECF_NOTHROW | ECF_LEAF);
inner_if_stmt = begin_if_stmt ();
finish_if_stmt_cond (build_call_n (acquire_fn, 1, guard_addr),
inner_if_stmt);
inner_then_clause = begin_compound_stmt (BCS_NO_SCOPE);
begin = get_target_expr (boolean_false_node);
flag = TARGET_EXPR_SLOT (begin);
TARGET_EXPR_CLEANUP (begin)
= build3 (COND_EXPR, void_type_node, flag,
void_node,
build_call_n (abort_fn, 1, guard_addr));
CLEANUP_EH_ONLY (begin) = 1;
init = add_stmt_to_compound (begin, init);
init = add_stmt_to_compound
(init, build2 (MODIFY_EXPR, void_type_node, flag, boolean_true_node));
init = add_stmt_to_compound
(init, build_call_n (release_fn, 1, guard_addr));
}
else
init = add_stmt_to_compound (init, set_guard (guard));
init = add_stmt_to_compound (init, register_dtor_fn (decl));
finish_expr_stmt (init);
if (thread_guard)
{
finish_compound_stmt (inner_then_clause);
finish_then_clause (inner_if_stmt);
finish_if_stmt (inner_if_stmt);
}
finish_compound_stmt (then_clause);
finish_then_clause (if_stmt);
finish_if_stmt (if_stmt);
}
else if (CP_DECL_THREAD_LOCAL_P (decl))
tls_aggregates = tree_cons (init, decl, tls_aggregates);
else
static_aggregates = tree_cons (init, decl, static_aggregates);
}

int
cp_complete_array_type (tree *ptype, tree initial_value, bool do_default)
{
int failure;
tree type, elt_type;
if (initial_value && TREE_CODE (initial_value) == CONSTRUCTOR
&& !BRACE_ENCLOSED_INITIALIZER_P (initial_value)
&& TREE_CODE (TREE_TYPE (initial_value)) != ARRAY_TYPE)
return 1;
if (initial_value)
{
unsigned HOST_WIDE_INT i;
tree value;
if (char_type_p (TYPE_MAIN_VARIANT (TREE_TYPE (*ptype)))
&& TREE_CODE (initial_value) == CONSTRUCTOR
&& !vec_safe_is_empty (CONSTRUCTOR_ELTS (initial_value)))
{
vec<constructor_elt, va_gc> *v = CONSTRUCTOR_ELTS (initial_value);
tree value = (*v)[0].value;
if (TREE_CODE (value) == STRING_CST
&& v->length () == 1)
initial_value = value;
}
if (TREE_CODE (initial_value) == CONSTRUCTOR)
{
FOR_EACH_CONSTRUCTOR_VALUE (CONSTRUCTOR_ELTS (initial_value), 
i, value)
{
if (PACK_EXPANSION_P (value))
return 0;
}
}
}
failure = complete_array_type (ptype, initial_value, do_default);
type = *ptype;
if (type != error_mark_node && TYPE_DOMAIN (type))
{
elt_type = TREE_TYPE (type);
TYPE_NEEDS_CONSTRUCTING (type) = TYPE_NEEDS_CONSTRUCTING (elt_type);
TYPE_HAS_NONTRIVIAL_DESTRUCTOR (type)
= TYPE_HAS_NONTRIVIAL_DESTRUCTOR (elt_type);
}
return failure;
}
int
cp_complete_array_type_or_error (tree *ptype, tree initial_value,
bool do_default, tsubst_flags_t complain)
{
int failure;
bool sfinae = !(complain & tf_error);
if (sfinae)
++pedantic;
failure = cp_complete_array_type (ptype, initial_value, do_default);
if (sfinae)
--pedantic;
if (failure)
{
if (sfinae)
;
else if (failure == 1)
error ("initializer fails to determine size of %qT", *ptype);
else if (failure == 2)
{
if (do_default)
error ("array size missing in %qT", *ptype);
}
else if (failure == 3)
error ("zero-size array %qT", *ptype);
*ptype = error_mark_node;
}
return failure;
}

static int
member_function_or_else (tree ctype, tree cur_type, enum overload_flags flags)
{
if (ctype && ctype != cur_type)
{
if (flags == DTOR_FLAG)
error ("destructor for alien class %qT cannot be a member", ctype);
else
error ("constructor for alien class %qT cannot be a member", ctype);
return 0;
}
return 1;
}

static void
bad_specifiers (tree object,
enum bad_spec_place type,
int virtualp,
int quals,
int inlinep,
int friendp,
int raises)
{
switch (type)
{
case BSP_VAR:
if (virtualp)
error ("%qD declared as a %<virtual%> variable", object);
if (quals)
error ("%<const%> and %<volatile%> function specifiers on "
"%qD invalid in variable declaration", object);
break;
case BSP_PARM:
if (virtualp)
error ("%qD declared as a %<virtual%> parameter", object);
if (inlinep)
error ("%qD declared as an %<inline%> parameter", object);
if (quals)
error ("%<const%> and %<volatile%> function specifiers on "
"%qD invalid in parameter declaration", object);
break;
case BSP_TYPE:
if (virtualp)
error ("%qD declared as a %<virtual%> type", object);
if (inlinep)
error ("%qD declared as an %<inline%> type", object);
if (quals)
error ("%<const%> and %<volatile%> function specifiers on "
"%qD invalid in type declaration", object);
break;
case BSP_FIELD:
if (virtualp)
error ("%qD declared as a %<virtual%> field", object);
if (inlinep)
error ("%qD declared as an %<inline%> field", object);
if (quals)
error ("%<const%> and %<volatile%> function specifiers on "
"%qD invalid in field declaration", object);
break;
default:
gcc_unreachable();
}
if (friendp)
error ("%q+D declared as a friend", object);
if (raises
&& !flag_noexcept_type
&& (TREE_CODE (object) == TYPE_DECL
|| (!TYPE_PTRFN_P (TREE_TYPE (object))
&& !TYPE_REFFN_P (TREE_TYPE (object))
&& !TYPE_PTRMEMFUNC_P (TREE_TYPE (object)))))
error ("%q+D declared with an exception specification", object);
}
static void
check_class_member_definition_namespace (tree decl)
{
gcc_assert (VAR_OR_FUNCTION_DECL_P (decl));
if (processing_specialization)
return;
if (processing_explicit_instantiation)
return;
if (!is_ancestor (current_namespace, DECL_CONTEXT (decl)))
permerror (input_location, "definition of %qD is not in namespace enclosing %qT",
decl, DECL_CONTEXT (decl));
}
tree
build_this_parm (tree fn, tree type, cp_cv_quals quals)
{
tree this_type;
tree qual_type;
tree parm;
cp_cv_quals this_quals;
if (CLASS_TYPE_P (type))
{
this_type
= cp_build_qualified_type (type, quals & ~TYPE_QUAL_RESTRICT);
this_type = build_pointer_type (this_type);
}
else
this_type = type_of_this_parm (type);
this_quals = (quals & TYPE_QUAL_RESTRICT) | TYPE_QUAL_CONST;
qual_type = cp_build_qualified_type (this_type, this_quals);
parm = build_artificial_parm (fn, this_identifier, qual_type);
cp_apply_type_quals_to_decl (this_quals, parm);
return parm;
}
static void
check_static_quals (tree decl, cp_cv_quals quals)
{
if (quals != TYPE_UNQUALIFIED)
error ("static member function %q#D declared with type qualifiers",
decl);
}
static void
check_concept_fn (tree fn)
{
if (DECL_ARGUMENTS (fn))
error ("concept %q#D declared with function parameters", fn);
tree type = TREE_TYPE (TREE_TYPE (fn));
if (is_auto (type))
error ("concept %q#D declared with a deduced return type", fn);
else if (type != boolean_type_node)
error ("concept %q#D with non-%<bool%> return type %qT", fn, type);
}
static tree
declare_simd_adjust_this (tree *tp, int *walk_subtrees, void *data)
{
tree this_parm = (tree) data;
if (TREE_CODE (*tp) == PARM_DECL
&& DECL_NAME (*tp) == this_identifier
&& *tp != this_parm)
*tp = this_parm;
else if (TYPE_P (*tp))
*walk_subtrees = 0;
return NULL_TREE;
}
static tree
grokfndecl (tree ctype,
tree type,
tree declarator,
tree parms,
tree orig_declarator,
tree decl_reqs,
int virtualp,
enum overload_flags flags,
cp_cv_quals quals,
cp_ref_qualifier rqual,
tree raises,
int check,
int friendp,
int publicp,
int inlinep,
bool deletedp,
special_function_kind sfk,
bool funcdef_flag,
int template_count,
tree in_namespace,
tree* attrlist,
location_t location)
{
tree decl;
int staticp = ctype && TREE_CODE (type) == FUNCTION_TYPE;
tree t;
bool concept_p = inlinep & 4;
if (concept_p && !funcdef_flag)
{
error ("concept %qD has no definition", declarator);
return NULL_TREE;
}
if (rqual)
type = build_ref_qualified_type (type, rqual);
if (raises)
type = build_exception_variant (type, raises);
decl = build_lang_decl (FUNCTION_DECL, declarator, type);
if (flag_concepts)
{
tree tmpl_reqs = NULL_TREE;
if (processing_template_decl > template_class_depth (ctype))
tmpl_reqs = TEMPLATE_PARMS_CONSTRAINTS (current_template_parms);
if (decl_reqs)
decl_reqs = normalize_expression (decl_reqs);
tree ci = build_constraints (tmpl_reqs, decl_reqs);
set_constraints (decl, ci);
}
if (location != UNKNOWN_LOCATION)
DECL_SOURCE_LOCATION (decl) = location;
if (TREE_CODE (type) == METHOD_TYPE)
{
tree parm = build_this_parm (decl, type, quals);
DECL_CHAIN (parm) = parms;
parms = parm;
SET_DECL_ALIGN (decl, MINIMUM_METHOD_BOUNDARY);
}
DECL_ARGUMENTS (decl) = parms;
for (t = parms; t; t = DECL_CHAIN (t))
DECL_CONTEXT (t) = decl;
if (TYPE_VOLATILE (type))
TREE_THIS_VOLATILE (decl) = 1;
switch (sfk)
{
case sfk_constructor:
case sfk_copy_constructor:
case sfk_move_constructor:
DECL_CXX_CONSTRUCTOR_P (decl) = 1;
DECL_NAME (decl) = ctor_identifier;
break;
case sfk_destructor:
DECL_CXX_DESTRUCTOR_P (decl) = 1;
DECL_NAME (decl) = dtor_identifier;
break;
default:
break;
}
if (friendp && TREE_CODE (orig_declarator) == TEMPLATE_ID_EXPR)
{
if (funcdef_flag)
error ("defining explicit specialization %qD in friend declaration",
orig_declarator);
else
{
tree fns = TREE_OPERAND (orig_declarator, 0);
tree args = TREE_OPERAND (orig_declarator, 1);
if (PROCESSING_REAL_TEMPLATE_DECL_P ())
{
error ("invalid use of template-id %qD in declaration "
"of primary template",
orig_declarator);
return NULL_TREE;
}
SET_DECL_IMPLICIT_INSTANTIATION (decl);
gcc_assert (identifier_p (fns) || TREE_CODE (fns) == OVERLOAD);
DECL_TEMPLATE_INFO (decl) = build_template_info (fns, args);
for (t = TYPE_ARG_TYPES (TREE_TYPE (decl)); t; t = TREE_CHAIN (t))
if (TREE_PURPOSE (t)
&& TREE_CODE (TREE_PURPOSE (t)) == DEFAULT_ARG)
{
error ("default arguments are not allowed in declaration "
"of friend template specialization %qD",
decl);
return NULL_TREE;
}
if (inlinep & 1)
{
error ("%<inline%> is not allowed in declaration of friend "
"template specialization %qD",
decl);
return NULL_TREE;
}
}
}
if (in_namespace)
set_decl_namespace (decl, in_namespace, friendp);
else if (!ctype)
DECL_CONTEXT (decl) = FROB_CONTEXT (current_decl_namespace ());
if (ctype == NULL_TREE
&& DECL_FILE_SCOPE_P (decl)
&& current_lang_name == lang_name_cplusplus
&& (MAIN_NAME_P (declarator)
|| (IDENTIFIER_LENGTH (declarator) > 10
&& IDENTIFIER_POINTER (declarator)[0] == '_'
&& IDENTIFIER_POINTER (declarator)[1] == '_'
&& strncmp (IDENTIFIER_POINTER (declarator)+2,
"builtin_", 8) == 0)
|| (targetcm.cxx_implicit_extern_c
&& (targetcm.cxx_implicit_extern_c
(IDENTIFIER_POINTER (declarator))))))
SET_DECL_LANGUAGE (decl, lang_c);
if (staticp)
{
DECL_STATIC_FUNCTION_P (decl) = 1;
DECL_CONTEXT (decl) = ctype;
}
if (deletedp)
DECL_DELETED_FN (decl) = 1;
if (ctype)
{
DECL_CONTEXT (decl) = ctype;
if (funcdef_flag)
check_class_member_definition_namespace (decl);
}
if (ctype == NULL_TREE && DECL_MAIN_P (decl))
{
if (PROCESSING_REAL_TEMPLATE_DECL_P())
error ("cannot declare %<::main%> to be a template");
if (inlinep & 1)
error ("cannot declare %<::main%> to be inline");
if (inlinep & 2)
error ("cannot declare %<::main%> to be %<constexpr%>");
if (!publicp)
error ("cannot declare %<::main%> to be static");
inlinep = 0;
publicp = 1;
}
if (ctype && (!TREE_PUBLIC (TYPE_MAIN_DECL (ctype))
|| decl_function_context (TYPE_MAIN_DECL (ctype))))
publicp = 0;
if (publicp && cxx_dialect == cxx98)
{
no_linkage_error (decl);
}
TREE_PUBLIC (decl) = publicp;
if (! publicp)
{
DECL_INTERFACE_KNOWN (decl) = 1;
DECL_NOT_REALLY_EXTERN (decl) = 1;
}
if (inlinep)
{
DECL_DECLARED_INLINE_P (decl) = 1;
if (publicp)
DECL_COMDAT (decl) = 1;
}
if (inlinep & 2)
DECL_DECLARED_CONSTEXPR_P (decl) = true;
if (concept_p)
{
DECL_DECLARED_CONCEPT_P (decl) = true;
check_concept_fn (decl);
}
DECL_EXTERNAL (decl) = 1;
if (TREE_CODE (type) == FUNCTION_TYPE)
{
if (quals || rqual)
TREE_TYPE (decl) = apply_memfn_quals (TREE_TYPE (decl),
TYPE_UNQUALIFIED,
REF_QUAL_NONE);
if (quals)
{
error (ctype
? G_("static member function %qD cannot have cv-qualifier")
: G_("non-member function %qD cannot have cv-qualifier"),
decl);
quals = TYPE_UNQUALIFIED;
}
if (rqual)
{
error (ctype
? G_("static member function %qD cannot have ref-qualifier")
: G_("non-member function %qD cannot have ref-qualifier"),
decl);
rqual = REF_QUAL_NONE;
}
}
if (deduction_guide_p (decl))
{
if (!DECL_NAMESPACE_SCOPE_P (decl))
{
error_at (location, "deduction guide %qD must be declared at "
"namespace scope", decl);
return NULL_TREE;
}
if (funcdef_flag)
error_at (location,
"deduction guide %qD must not have a function body", decl);
}
else if (IDENTIFIER_ANY_OP_P (DECL_NAME (decl))
&& !grok_op_properties (decl, true))
return NULL_TREE;
else if (UDLIT_OPER_P (DECL_NAME (decl)))
{
bool long_long_unsigned_p;
bool long_double_p;
const char *suffix = NULL;
if (DECL_LANGUAGE (decl) == lang_c)
{
error ("literal operator with C linkage");
maybe_show_extern_c_location ();
return NULL_TREE;
}
if (DECL_NAMESPACE_SCOPE_P (decl))
{
if (!check_literal_operator_args (decl, &long_long_unsigned_p,
&long_double_p))
{
error ("%qD has invalid argument list", decl);
return NULL_TREE;
}
suffix = UDLIT_OP_SUFFIX (DECL_NAME (decl));
if (long_long_unsigned_p)
{
if (cpp_interpret_int_suffix (parse_in, suffix, strlen (suffix)))
warning (0, "integer suffix %qs"
" shadowed by implementation", suffix);
}
else if (long_double_p)
{
if (cpp_interpret_float_suffix (parse_in, suffix, strlen (suffix)))
warning (0, "floating point suffix %qs"
" shadowed by implementation", suffix);
}
if (suffix[0] != '_'
&& !in_system_header_at (DECL_SOURCE_LOCATION (decl))
&& !current_function_decl && !(friendp && !funcdef_flag))
warning (OPT_Wliteral_suffix,
"literal operator suffixes not preceded by %<_%>"
" are reserved for future standardization");
}
else
{
error ("%qD must be a non-member function", decl);
return NULL_TREE;
}
}
if (funcdef_flag)
DECL_INITIAL (decl) = error_mark_node;
if (TYPE_NOTHROW_P (type) || nothrow_libfn_p (decl))
TREE_NOTHROW (decl) = 1;
if (flag_openmp || flag_openmp_simd)
{
tree ods = lookup_attribute ("omp declare simd", *attrlist);
if (ods)
{
tree attr;
for (attr = ods; attr;
attr = lookup_attribute ("omp declare simd", TREE_CHAIN (attr)))
{
if (TREE_CODE (type) == METHOD_TYPE)
walk_tree (&TREE_VALUE (attr), declare_simd_adjust_this,
DECL_ARGUMENTS (decl), NULL);
if (TREE_VALUE (attr) != NULL_TREE)
{
tree cl = TREE_VALUE (TREE_VALUE (attr));
cl = c_omp_declare_simd_clauses_to_numbers
(DECL_ARGUMENTS (decl), cl);
if (cl)
TREE_VALUE (TREE_VALUE (attr)) = cl;
else
TREE_VALUE (attr) = NULL_TREE;
}
}
}
}
if (check < 0)
return decl;
if (ctype != NULL_TREE)
grokclassfn (ctype, decl, flags);
if (cxx_dialect >= cxx11
&& DECL_DESTRUCTOR_P (decl)
&& !TYPE_BEING_DEFINED (DECL_CONTEXT (decl))
&& !processing_template_decl)
deduce_noexcept_on_destructor (decl);
decl = check_explicit_specialization (orig_declarator, decl,
template_count,
2 * funcdef_flag +
4 * (friendp != 0) +
8 * concept_p,
*attrlist);
if (decl == error_mark_node)
return NULL_TREE;
if (DECL_STATIC_FUNCTION_P (decl))
check_static_quals (decl, quals);
if (attrlist)
{
cplus_decl_attributes (&decl, *attrlist, 0);
*attrlist = NULL_TREE;
}
if (ctype == NULL_TREE && DECL_MAIN_P (decl))
{
if (!same_type_p (TREE_TYPE (TREE_TYPE (decl)),
integer_type_node))
{
tree oldtypeargs = TYPE_ARG_TYPES (TREE_TYPE (decl));
tree newtype;
error ("%<::main%> must return %<int%>");
newtype = build_function_type (integer_type_node, oldtypeargs);
TREE_TYPE (decl) = newtype;
}
if (warn_main)
check_main_parameter_types (decl);
}
if (ctype != NULL_TREE && check)
{
tree old_decl = check_classfn (ctype, decl,
(processing_template_decl
> template_class_depth (ctype))
? current_template_parms
: NULL_TREE);
if (old_decl == error_mark_node)
return NULL_TREE;
if (old_decl)
{
tree ok;
tree pushed_scope;
if (TREE_CODE (old_decl) == TEMPLATE_DECL)
old_decl = DECL_TEMPLATE_RESULT (old_decl);
if (DECL_STATIC_FUNCTION_P (old_decl)
&& TREE_CODE (TREE_TYPE (decl)) == METHOD_TYPE)
{
revert_static_member_fn (decl);
check_static_quals (decl, quals);
}
if (DECL_ARTIFICIAL (old_decl))
{
error ("definition of implicitly-declared %qD", old_decl);
return NULL_TREE;
}
else if (DECL_DEFAULTED_FN (old_decl))
{
error ("definition of explicitly-defaulted %q+D", decl);
inform (DECL_SOURCE_LOCATION (old_decl),
"%q#D explicitly defaulted here", old_decl);
return NULL_TREE;
}
if (TREE_CODE (decl) == TEMPLATE_DECL)
decl = DECL_TEMPLATE_RESULT (decl);
pushed_scope = push_scope (ctype);
ok = duplicate_decls (decl, old_decl, friendp);
if (pushed_scope)
pop_scope (pushed_scope);
if (!ok)
{
error ("no %q#D member function declared in class %qT",
decl, ctype);
return NULL_TREE;
}
if (ok == error_mark_node)
return NULL_TREE;
return old_decl;
}
}
if (DECL_CONSTRUCTOR_P (decl) && !grok_ctor_properties (ctype, decl))
return NULL_TREE;
if (ctype == NULL_TREE || check)
return decl;
if (virtualp)
DECL_VIRTUAL_P (decl) = 1;
return decl;
}
static tree
set_virt_specifiers (tree decl, cp_virt_specifiers specifiers)
{
if (decl == NULL_TREE)
return decl;
if (specifiers & VIRT_SPEC_OVERRIDE)
DECL_OVERRIDE_P (decl) = 1;
if (specifiers & VIRT_SPEC_FINAL)
DECL_FINAL_P (decl) = 1;
return decl;
}
static void
set_linkage_for_static_data_member (tree decl)
{
TREE_PUBLIC (decl) = 1;
TREE_STATIC (decl) = 1;
if (!processing_template_decl)
DECL_INTERFACE_KNOWN (decl) = 1;
}
static tree
grokvardecl (tree type,
tree name,
tree orig_declarator,
const cp_decl_specifier_seq *declspecs,
int initialized,
int type_quals,
int inlinep,
bool conceptp,
int template_count,
tree scope)
{
tree decl;
tree explicit_scope;
gcc_assert (!name || identifier_p (name));
bool constp = (type_quals & TYPE_QUAL_CONST) != 0;
bool volatilep = (type_quals & TYPE_QUAL_VOLATILE) != 0;
explicit_scope = scope;
if (!scope)
{
if (declspecs->storage_class == sc_extern)
scope = current_decl_namespace ();
else if (!at_function_scope_p ())
scope = current_scope ();
}
if (scope
&& (
(TREE_CODE (scope) == NAMESPACE_DECL && processing_template_decl)
|| (TREE_CODE (scope) == NAMESPACE_DECL
&& current_lang_name != lang_name_cplusplus)
|| TYPE_P (scope)
|| (orig_declarator
&& TREE_CODE (orig_declarator) == TEMPLATE_ID_EXPR)))
decl = build_lang_decl (VAR_DECL, name, type);
else
decl = build_decl (input_location, VAR_DECL, name, type);
if (explicit_scope && TREE_CODE (explicit_scope) == NAMESPACE_DECL)
set_decl_namespace (decl, explicit_scope, 0);
else
DECL_CONTEXT (decl) = FROB_CONTEXT (scope);
if (declspecs->storage_class == sc_extern)
{
DECL_THIS_EXTERN (decl) = 1;
DECL_EXTERNAL (decl) = !initialized;
}
if (DECL_CLASS_SCOPE_P (decl))
{
set_linkage_for_static_data_member (decl);
DECL_EXTERNAL (decl) = 0;
check_class_member_definition_namespace (decl);
}
else if (toplevel_bindings_p ())
{
TREE_PUBLIC (decl) = (declspecs->storage_class != sc_static
&& (DECL_THIS_EXTERN (decl)
|| ! constp
|| volatilep
|| inlinep));
TREE_STATIC (decl) = ! DECL_EXTERNAL (decl);
}
else
{
TREE_STATIC (decl) = declspecs->storage_class == sc_static;
TREE_PUBLIC (decl) = DECL_EXTERNAL (decl);
}
if (decl_spec_seq_has_spec_p (declspecs, ds_thread))
{
if (DECL_EXTERNAL (decl) || TREE_STATIC (decl))
{
CP_DECL_THREAD_LOCAL_P (decl) = true;
if (!processing_template_decl)
set_decl_tls_model (decl, decl_default_tls_model (decl));
}
if (declspecs->gnu_thread_keyword_p)
SET_DECL_GNU_TLS_P (decl);
}
if (cxx_dialect > cxx98
&& decl_linkage (decl) != lk_none
&& DECL_LANG_SPECIFIC (decl) == NULL
&& !DECL_EXTERN_C_P (decl)
&& no_linkage_check (TREE_TYPE (decl), false))
retrofit_lang_decl (decl);
if (TREE_PUBLIC (decl))
{
if (cxx_dialect < cxx11)
no_linkage_error (decl);
}
else
DECL_INTERFACE_KNOWN (decl) = 1;
if (DECL_NAME (decl)
&& MAIN_NAME_P (DECL_NAME (decl))
&& scope == global_namespace)
error ("cannot declare %<::main%> to be a global variable");
if (conceptp)
{
if (!processing_template_decl)
{
error ("a non-template variable cannot be %<concept%>");
return NULL_TREE;
}
else
DECL_DECLARED_CONCEPT_P (decl) = true;
if (!same_type_ignoring_top_level_qualifiers_p (type, boolean_type_node))
error_at (declspecs->locations[ds_type_spec],
"concept must have type %<bool%>");
}
else if (flag_concepts
&& processing_template_decl > template_class_depth (scope))
{
tree reqs = TEMPLATE_PARMS_CONSTRAINTS (current_template_parms);
tree ci = build_constraints (reqs, NULL_TREE);
set_constraints (decl, ci);
}
if (orig_declarator)
decl = check_explicit_specialization (orig_declarator, decl,
template_count, conceptp * 8);
return decl != error_mark_node ? decl : NULL_TREE;
}
tree
build_ptrmemfunc_type (tree type)
{
tree field, fields;
tree t;
if (type == error_mark_node)
return type;
if (cp_cv_quals quals = cp_type_quals (type))
{
tree unqual = build_ptrmemfunc_type (TYPE_MAIN_VARIANT (type));
return cp_build_qualified_type (unqual, quals);
}
t = TYPE_PTRMEMFUNC_TYPE (type);
if (t)
return t;
t = make_node (RECORD_TYPE);
TYPE_PTRMEMFUNC_FLAG (t) = 1;
field = build_decl (input_location, FIELD_DECL, pfn_identifier, type);
fields = field;
field = build_decl (input_location, FIELD_DECL, delta_identifier, 
delta_type_node);
DECL_CHAIN (field) = fields;
fields = field;
finish_builtin_struct (t, "__ptrmemfunc_type", fields, ptr_type_node);
TYPE_NAME (t) = NULL_TREE;
TYPE_PTRMEMFUNC_TYPE (type) = t;
if (TYPE_STRUCTURAL_EQUALITY_P (type))
SET_TYPE_STRUCTURAL_EQUALITY (t);
else if (TYPE_CANONICAL (type) != type)
TYPE_CANONICAL (t) = build_ptrmemfunc_type (TYPE_CANONICAL (type));
return t;
}
tree
build_ptrmem_type (tree class_type, tree member_type)
{
if (TREE_CODE (member_type) == METHOD_TYPE)
{
cp_cv_quals quals = type_memfn_quals (member_type);
cp_ref_qualifier rqual = type_memfn_rqual (member_type);
member_type = build_memfn_type (member_type, class_type, quals, rqual);
return build_ptrmemfunc_type (build_pointer_type (member_type));
}
else
{
gcc_assert (TREE_CODE (member_type) != FUNCTION_TYPE);
return build_offset_type (class_type, member_type);
}
}
static int
check_static_variable_definition (tree decl, tree type)
{
if (!current_class_type || !TYPE_BEING_DEFINED (current_class_type))
return 0;
if (dependent_type_p (type))
return 0;
if (DECL_P (decl)
&& (DECL_DECLARED_CONSTEXPR_P (decl)
|| DECL_VAR_DECLARED_INLINE_P (decl)))
return 0;
else if (cxx_dialect >= cxx11 && !INTEGRAL_OR_ENUMERATION_TYPE_P (type))
{
if (!COMPLETE_TYPE_P (type))
error_at (DECL_SOURCE_LOCATION (decl),
"in-class initialization of static data member %q#D of "
"incomplete type", decl);
else if (literal_type_p (type))
permerror (DECL_SOURCE_LOCATION (decl),
"%<constexpr%> needed for in-class initialization of "
"static data member %q#D of non-integral type", decl);
else
error_at (DECL_SOURCE_LOCATION (decl),
"in-class initialization of static data member %q#D of "
"non-literal type", decl);
return 1;
}
if (!ARITHMETIC_TYPE_P (type) && TREE_CODE (type) != ENUMERAL_TYPE)
{
error_at (DECL_SOURCE_LOCATION (decl),
"invalid in-class initialization of static data member "
"of non-integral type %qT",
type);
return 1;
}
else if (!CP_TYPE_CONST_P (type))
error_at (DECL_SOURCE_LOCATION (decl),
"ISO C++ forbids in-class initialization of non-const "
"static member %qD",
decl);
else if (!INTEGRAL_OR_ENUMERATION_TYPE_P (type))
pedwarn (DECL_SOURCE_LOCATION (decl), OPT_Wpedantic,
"ISO C++ forbids initialization of member constant "
"%qD of non-integral type %qT", decl, type);
return 0;
}
static tree
stabilize_save_expr_r (tree *expr_p, int *walk_subtrees, void *data)
{
hash_set<tree> *pset = (hash_set<tree> *)data;
tree expr = *expr_p;
if (TREE_CODE (expr) == SAVE_EXPR)
{
tree op = TREE_OPERAND (expr, 0);
cp_walk_tree (&op, stabilize_save_expr_r, data, pset);
if (TREE_SIDE_EFFECTS (op))
TREE_OPERAND (expr, 0) = get_temp_regvar (TREE_TYPE (op), op);
*walk_subtrees = 0;
}
else if (!EXPR_P (expr) || !TREE_SIDE_EFFECTS (expr))
*walk_subtrees = 0;
return NULL;
}
static void
stabilize_vla_size (tree size)
{
hash_set<tree> pset;
cp_walk_tree (&size, stabilize_save_expr_r, &pset, &pset);
}
tree
fold_sizeof_expr (tree t)
{
tree r;
if (SIZEOF_EXPR_TYPE_P (t))
r = cxx_sizeof_or_alignof_type (TREE_TYPE (TREE_OPERAND (t, 0)),
SIZEOF_EXPR, false, false);
else if (TYPE_P (TREE_OPERAND (t, 0)))
r = cxx_sizeof_or_alignof_type (TREE_OPERAND (t, 0), SIZEOF_EXPR,
false, false);
else
r = cxx_sizeof_or_alignof_expr (TREE_OPERAND (t, 0), SIZEOF_EXPR,
false);
if (r == error_mark_node)
r = size_one_node;
return r;
}
tree
compute_array_index_type (tree name, tree size, tsubst_flags_t complain)
{
tree itype;
tree osize = size;
if (error_operand_p (size))
return error_mark_node;
if (!type_dependent_expression_p (size))
{
osize = size = mark_rvalue_use (size);
if (cxx_dialect < cxx11 && TREE_CODE (size) == NOP_EXPR
&& TREE_SIDE_EFFECTS (size))
;
else
{
size = instantiate_non_dependent_expr_sfinae (size, complain);
size = build_converted_constant_expr (size_type_node, size, complain);
size = maybe_constant_value (size);
if (!TREE_CONSTANT (size))
size = osize;
}
if (error_operand_p (size))
return error_mark_node;
tree type = TREE_TYPE (size);
if (!INTEGRAL_OR_UNSCOPED_ENUMERATION_TYPE_P (type))
{
if (!(complain & tf_error))
return error_mark_node;
if (name)
error ("size of array %qD has non-integral type %qT", name, type);
else
error ("size of array has non-integral type %qT", type);
size = integer_one_node;
}
}
if (processing_template_decl
&& (type_dependent_expression_p (size)
|| !TREE_CONSTANT (size) || value_dependent_expression_p (size)))
{
itype = build_index_type (build_min (MINUS_EXPR, sizetype,
size, size_one_node));
TYPE_DEPENDENT_P (itype) = 1;
TYPE_DEPENDENT_P_VALID (itype) = 1;
SET_TYPE_STRUCTURAL_EQUALITY (itype);
return itype;
}
if (TREE_CODE (size) != INTEGER_CST)
{
tree folded = cp_fully_fold (size);
if (TREE_CODE (folded) == INTEGER_CST)
pedwarn (location_of (size), OPT_Wpedantic,
"size of array is not an integral constant-expression");
size = folded;
}
if (TREE_CODE (size) == INTEGER_CST)
{
if (!valid_constant_size_p (size))
{
if (!(complain & tf_error))
return error_mark_node;
if (name)
error ("size of array %qD is negative", name);
else
error ("size of array is negative");
size = integer_one_node;
}
else if (integer_zerop (size))
{
if (!(complain & tf_error))
return error_mark_node;
else if (in_system_header_at (input_location))
;
else if (name)
pedwarn (input_location, OPT_Wpedantic, "ISO C++ forbids zero-size array %qD", name);
else
pedwarn (input_location, OPT_Wpedantic, "ISO C++ forbids zero-size array");
}
}
else if (TREE_CONSTANT (size)
|| !at_function_scope_p ()
|| !(complain & tf_error))
{
if (!(complain & tf_error))
return error_mark_node;
if (name)
error ("size of array %qD is not an integral constant-expression",
name);
else
error ("size of array is not an integral constant-expression");
size = integer_one_node;
}
else if (pedantic && warn_vla != 0)
{
if (name)
pedwarn (input_location, OPT_Wvla, "ISO C++ forbids variable length array %qD", name);
else
pedwarn (input_location, OPT_Wvla, "ISO C++ forbids variable length array");
}
else if (warn_vla > 0)
{
if (name)
warning (OPT_Wvla, 
"variable length array %qD is used", name);
else
warning (OPT_Wvla, 
"variable length array is used");
}
if (processing_template_decl && !TREE_CONSTANT (size))
itype = build_min (MINUS_EXPR, sizetype, size, integer_one_node);
else
{
{
processing_template_decl_sentinel s;
itype = cp_build_binary_op (input_location,
MINUS_EXPR,
cp_convert (ssizetype, size, complain),
cp_convert (ssizetype, integer_one_node,
complain),
complain);
itype = maybe_constant_value (itype);
}
if (!TREE_CONSTANT (itype))
{
itype = variable_size (itype);
stabilize_vla_size (itype);
if (sanitize_flags_p (SANITIZE_VLA)
&& current_function_decl != NULL_TREE)
{
tree t = fold_build2 (PLUS_EXPR, TREE_TYPE (itype), itype,
build_one_cst (TREE_TYPE (itype)));
t = ubsan_instrument_vla (input_location, t);
finish_expr_stmt (t);
}
}
else if (TREE_CODE (itype) == INTEGER_CST
&& TREE_OVERFLOW (itype))
{
if (!(complain & tf_error))
return error_mark_node;
error ("overflow in array dimension");
TREE_OVERFLOW (itype) = 0;
}
}
itype = build_index_type (itype);
TYPE_DEPENDENT_P (itype) = 0;
TYPE_DEPENDENT_P_VALID (itype) = 1;
return itype;
}
tree
get_scope_of_declarator (const cp_declarator *declarator)
{
while (declarator && declarator->kind != cdk_id)
declarator = declarator->declarator;
if (declarator
&& declarator->u.id.qualifying_scope)
return declarator->u.id.qualifying_scope;
return NULL_TREE;
}
static tree
create_array_type_for_decl (tree name, tree type, tree size)
{
tree itype = NULL_TREE;
if (type == error_mark_node || size == error_mark_node)
return error_mark_node;
if (type_uses_auto (type))
{
error ("%qD declared as array of %qT", name, type);
return error_mark_node;
}
switch (TREE_CODE (type))
{
case VOID_TYPE:
if (name)
error ("declaration of %qD as array of void", name);
else
error ("creating array of void");
return error_mark_node;
case FUNCTION_TYPE:
if (name)
error ("declaration of %qD as array of functions", name);
else
error ("creating array of functions");
return error_mark_node;
case REFERENCE_TYPE:
if (name)
error ("declaration of %qD as array of references", name);
else
error ("creating array of references");
return error_mark_node;
case METHOD_TYPE:
if (name)
error ("declaration of %qD as array of function members", name);
else
error ("creating array of function members");
return error_mark_node;
default:
break;
}
if (TREE_CODE (type) == ARRAY_TYPE && !TYPE_DOMAIN (type))
{
if (name)
error ("declaration of %qD as multidimensional array must "
"have bounds for all dimensions except the first",
name);
else
error ("multidimensional array must have bounds for all "
"dimensions except the first");
return error_mark_node;
}
if (size)
itype = compute_array_index_type (name, size, tf_warning_or_error);
abstract_virtuals_error (name, type);
return build_cplus_array_type (type, itype);
}
static location_t
smallest_type_quals_location (int type_quals, const location_t* locations)
{
location_t loc = UNKNOWN_LOCATION;
if (type_quals & TYPE_QUAL_CONST)
loc = locations[ds_const];
if ((type_quals & TYPE_QUAL_VOLATILE)
&& (loc == UNKNOWN_LOCATION || locations[ds_volatile] < loc))
loc = locations[ds_volatile];
if ((type_quals & TYPE_QUAL_RESTRICT)
&& (loc == UNKNOWN_LOCATION || locations[ds_restrict] < loc))
loc = locations[ds_restrict];
return loc;
}
static tree
check_special_function_return_type (special_function_kind sfk,
tree type,
tree optype,
int type_quals,
const location_t* locations)
{
switch (sfk)
{
case sfk_constructor:
if (type)
error ("return type specification for constructor invalid");
else if (type_quals != TYPE_UNQUALIFIED)
error_at (smallest_type_quals_location (type_quals, locations),
"qualifiers are not allowed on constructor declaration");
if (targetm.cxx.cdtor_returns_this ())
type = build_pointer_type (optype);
else
type = void_type_node;
break;
case sfk_destructor:
if (type)
error ("return type specification for destructor invalid");
else if (type_quals != TYPE_UNQUALIFIED)
error_at (smallest_type_quals_location (type_quals, locations),
"qualifiers are not allowed on destructor declaration");
if (targetm.cxx.cdtor_returns_this ())
type = build_pointer_type (void_type_node);
else
type = void_type_node;
break;
case sfk_conversion:
if (type)
error ("return type specified for %<operator %T%>", optype);
else if (type_quals != TYPE_UNQUALIFIED)
error_at (smallest_type_quals_location (type_quals, locations),
"qualifiers are not allowed on declaration of "
"%<operator %T%>", optype);
type = optype;
break;
case sfk_deduction_guide:
if (type)
error ("return type specified for deduction guide");
else if (type_quals != TYPE_UNQUALIFIED)
error_at (smallest_type_quals_location (type_quals, locations),
"qualifiers are not allowed on declaration of "
"deduction guide");
if (TREE_CODE (optype) == TEMPLATE_TEMPLATE_PARM)
{
error ("template template parameter %qT in declaration of "
"deduction guide", optype);
type = error_mark_node;
}
else
type = make_template_placeholder (CLASSTYPE_TI_TEMPLATE (optype));
for (int i = 0; i < ds_last; ++i)
if (i != ds_explicit && locations[i])
error_at (locations[i],
"decl-specifier in declaration of deduction guide");
break;
default:
gcc_unreachable ();
}
return type;
}
tree
check_var_type (tree identifier, tree type)
{
if (VOID_TYPE_P (type))
{
if (!identifier)
error ("unnamed variable or field declared void");
else if (identifier_p (identifier))
{
gcc_assert (!IDENTIFIER_ANY_OP_P (identifier));
error ("variable or field %qE declared void", identifier);
}
else
error ("variable or field declared void");
type = error_mark_node;
}
return type;
}
static void
mark_inline_variable (tree decl)
{
bool inlinep = true;
if (! toplevel_bindings_p ())
{
error ("%<inline%> specifier invalid for variable "
"%qD declared at block scope", decl);
inlinep = false;
}
else if (cxx_dialect < cxx17)
pedwarn (DECL_SOURCE_LOCATION (decl), 0,
"inline variables are only available "
"with -std=c++17 or -std=gnu++17");
if (inlinep)
{
retrofit_lang_decl (decl);
SET_DECL_VAR_DECLARED_INLINE_P (decl);
}
}
void
name_unnamed_type (tree type, tree decl)
{
gcc_assert (TYPE_UNNAMED_P (type));
for (tree t = TYPE_MAIN_VARIANT (type); t; t = TYPE_NEXT_VARIANT (t))
{
if (anon_aggrname_p (TYPE_IDENTIFIER (t)))
TYPE_NAME (t) = decl;
}
if (TYPE_LANG_SPECIFIC (type))
TYPE_WAS_UNNAMED (type) = 1;
if (TYPE_LANG_SPECIFIC (type) && CLASSTYPE_TEMPLATE_INFO (type))
DECL_NAME (CLASSTYPE_TI_TEMPLATE (type))
= TYPE_IDENTIFIER (type);
reset_type_linkage (type);
gcc_assert (!TYPE_UNNAMED_P (type));
}
tree
grokdeclarator (const cp_declarator *declarator,
cp_decl_specifier_seq *declspecs,
enum decl_context decl_context,
int initialized,
tree* attrlist)
{
tree type = NULL_TREE;
int longlong = 0;
int explicit_intN = 0;
int virtualp, explicitp, friendp, inlinep, staticp;
int explicit_int = 0;
int explicit_char = 0;
int defaulted_int = 0;
tree typedef_decl = NULL_TREE;
const char *name = NULL;
tree typedef_type = NULL_TREE;
bool funcdef_flag = false;
cp_declarator_kind innermost_code = cdk_error;
int bitfield = 0;
#if 0
tree decl_attr = NULL_TREE;
#endif
special_function_kind sfk = sfk_none;
tree dname = NULL_TREE;
tree ctor_return_type = NULL_TREE;
enum overload_flags flags = NO_SPECIAL;
cp_cv_quals memfn_quals = TYPE_UNQUALIFIED;
cp_virt_specifiers virt_specifiers = VIRT_SPEC_UNSPECIFIED;
cp_ref_qualifier rqual = REF_QUAL_NONE;
int type_quals = TYPE_UNQUALIFIED;
tree raises = NULL_TREE;
int template_count = 0;
tree returned_attrs = NULL_TREE;
tree parms = NULL_TREE;
const cp_declarator *id_declarator;
tree unqualified_id;
tree ctype = current_class_type;
tree in_namespace = NULL_TREE;
cp_storage_class storage_class;
bool unsigned_p, signed_p, short_p, long_p, thread_p;
bool type_was_error_mark_node = false;
bool parameter_pack_p = declarator ? declarator->parameter_pack_p : false;
bool template_type_arg = false;
bool template_parm_flag = false;
bool typedef_p = decl_spec_seq_has_spec_p (declspecs, ds_typedef);
bool constexpr_p = decl_spec_seq_has_spec_p (declspecs, ds_constexpr);
bool late_return_type_p = false;
bool array_parameter_p = false;
source_location saved_loc = input_location;
tree reqs = NULL_TREE;
signed_p = decl_spec_seq_has_spec_p (declspecs, ds_signed);
unsigned_p = decl_spec_seq_has_spec_p (declspecs, ds_unsigned);
short_p = decl_spec_seq_has_spec_p (declspecs, ds_short);
long_p = decl_spec_seq_has_spec_p (declspecs, ds_long);
longlong = decl_spec_seq_has_spec_p (declspecs, ds_long_long);
explicit_intN = declspecs->explicit_intN_p;
thread_p = decl_spec_seq_has_spec_p (declspecs, ds_thread);
bool concept_p = decl_spec_seq_has_spec_p (declspecs, ds_concept);
if (concept_p)
constexpr_p = true;
if (decl_spec_seq_has_spec_p (declspecs, ds_const))
type_quals |= TYPE_QUAL_CONST;
if (decl_spec_seq_has_spec_p (declspecs, ds_volatile))
type_quals |= TYPE_QUAL_VOLATILE;
if (decl_spec_seq_has_spec_p (declspecs, ds_restrict))
type_quals |= TYPE_QUAL_RESTRICT;
if (decl_context == FUNCDEF)
funcdef_flag = true, decl_context = NORMAL;
else if (decl_context == MEMFUNCDEF)
funcdef_flag = true, decl_context = FIELD;
else if (decl_context == BITFIELD)
bitfield = 1, decl_context = FIELD;
else if (decl_context == TEMPLATE_TYPE_ARG)
template_type_arg = true, decl_context = TYPENAME;
else if (decl_context == TPARM)
template_parm_flag = true, decl_context = PARM;
if (initialized > 1)
funcdef_flag = true;
location_t typespec_loc = smallest_type_quals_location (type_quals,
declspecs->locations);
if (typespec_loc == UNKNOWN_LOCATION)
typespec_loc = declspecs->locations[ds_type_spec];
if (typespec_loc == UNKNOWN_LOCATION)
typespec_loc = input_location;
for (id_declarator = declarator;
id_declarator;
id_declarator = id_declarator->declarator)
{
if (id_declarator->kind != cdk_id)
innermost_code = id_declarator->kind;
switch (id_declarator->kind)
{
case cdk_function:
if (id_declarator->declarator
&& id_declarator->declarator->kind == cdk_id)
{
sfk = id_declarator->declarator->u.id.sfk;
if (sfk == sfk_destructor)
flags = DTOR_FLAG;
}
break;
case cdk_id:
{
tree qualifying_scope = id_declarator->u.id.qualifying_scope;
tree decl = id_declarator->u.id.unqualified_name;
if (!decl)
break;
if (qualifying_scope)
{
if (check_for_bare_parameter_packs (qualifying_scope,
id_declarator->id_loc))
return error_mark_node;
if (at_function_scope_p ())
{
if (qualifying_scope == global_namespace)
error ("invalid use of qualified-name %<::%D%>",
decl);
else if (TYPE_P (qualifying_scope))
error ("invalid use of qualified-name %<%T::%D%>",
qualifying_scope, decl);
else 
error ("invalid use of qualified-name %<%D::%D%>",
qualifying_scope, decl);
return error_mark_node;
}
else if (TYPE_P (qualifying_scope))
{
ctype = qualifying_scope;
if (!MAYBE_CLASS_TYPE_P (ctype))
{
error ("%q#T is not a class or a namespace", ctype);
ctype = NULL_TREE;
}
else if (innermost_code != cdk_function
&& current_class_type
&& !uniquely_derived_from_p (ctype,
current_class_type))
{
error ("invalid use of qualified-name %<%T::%D%>",
qualifying_scope, decl);
return error_mark_node;
}
}
else if (TREE_CODE (qualifying_scope) == NAMESPACE_DECL)
in_namespace = qualifying_scope;
}
switch (TREE_CODE (decl))
{
case BIT_NOT_EXPR:
{
if (innermost_code != cdk_function)
{
error ("declaration of %qD as non-function", decl);
return error_mark_node;
}
else if (!qualifying_scope
&& !(current_class_type && at_class_scope_p ()))
{
error ("declaration of %qD as non-member", decl);
return error_mark_node;
}
tree type = TREE_OPERAND (decl, 0);
if (TYPE_P (type))
type = constructor_name (type);
name = identifier_to_locale (IDENTIFIER_POINTER (type));
dname = decl;
}
break;
case TEMPLATE_ID_EXPR:
{
tree fns = TREE_OPERAND (decl, 0);
dname = fns;
if (!identifier_p (dname))
dname = OVL_NAME (dname);
}
case IDENTIFIER_NODE:
if (identifier_p (decl))
dname = decl;
if (IDENTIFIER_KEYWORD_P (dname))
{
error ("declarator-id missing; using reserved word %qD",
dname);
name = identifier_to_locale (IDENTIFIER_POINTER (dname));
}
else if (!IDENTIFIER_CONV_OP_P (dname))
name = identifier_to_locale (IDENTIFIER_POINTER (dname));
else
{
gcc_assert (flags == NO_SPECIAL);
flags = TYPENAME_FLAG;
sfk = sfk_conversion;
tree glob = get_global_binding (dname);
if (glob && TREE_CODE (glob) == TYPE_DECL)
name = identifier_to_locale (IDENTIFIER_POINTER (dname));
else
name = "<invalid operator>";
}
break;
default:
gcc_unreachable ();
}
break;
}
case cdk_array:
case cdk_pointer:
case cdk_reference:
case cdk_ptrmem:
break;
case cdk_decomp:
name = "structured binding";
break;
case cdk_error:
return error_mark_node;
default:
gcc_unreachable ();
}
if (id_declarator->kind == cdk_id)
break;
}
if (funcdef_flag && innermost_code != cdk_function)
{
error ("function definition does not declare parameters");
return error_mark_node;
}
if (flags == TYPENAME_FLAG
&& innermost_code != cdk_function
&& ! (ctype && !declspecs->any_specifiers_p))
{
error ("declaration of %qD as non-function", dname);
return error_mark_node;
}
if (dname && identifier_p (dname))
{
if (UDLIT_OPER_P (dname)
&& innermost_code != cdk_function)
{
error ("declaration of %qD as non-function", dname);
return error_mark_node;
}
if (IDENTIFIER_ANY_OP_P (dname))
{
if (typedef_p)
{
error ("declaration of %qD as %<typedef%>", dname);
return error_mark_node;
}
else if (decl_context == PARM || decl_context == CATCHPARM)
{
error ("declaration of %qD as parameter", dname);
return error_mark_node;
}
}
}
if (decl_context == NORMAL && !toplevel_bindings_p ())
{
cp_binding_level *b = current_binding_level;
current_binding_level = b->level_chain;
if (current_binding_level != 0 && toplevel_bindings_p ())
decl_context = PARM;
current_binding_level = b;
}
if (name == NULL)
name = decl_context == PARM ? "parameter" : "type name";
if (concept_p && typedef_p)
{
error ("%<concept%> cannot appear in a typedef declaration");
return error_mark_node;
}
if (constexpr_p && typedef_p)
{
error ("%<constexpr%> cannot appear in a typedef declaration");
return error_mark_node;
}
if (declspecs->multiple_types_p)
{
error ("two or more data types in declaration of %qs", name);
return error_mark_node;
}
if (declspecs->conflicting_specifiers_p)
{
error ("conflicting specifiers in declaration of %qs", name);
return error_mark_node;
}
type = declspecs->type;
if (type == error_mark_node)
{
type = NULL_TREE;
type_was_error_mark_node = true;
}
if (type && TREE_DEPRECATED (type)
&& deprecated_state != DEPRECATED_SUPPRESS)
cp_warn_deprecated_use (type);
if (type && TREE_CODE (type) == TYPE_DECL)
{
typedef_decl = type;
type = TREE_TYPE (typedef_decl);
if (TREE_DEPRECATED (type)
&& DECL_ARTIFICIAL (typedef_decl)
&& deprecated_state != DEPRECATED_SUPPRESS)
cp_warn_deprecated_use (type);
}
if (type == NULL_TREE)
{
if (signed_p || unsigned_p || long_p || short_p)
{
type = integer_type_node;
defaulted_int = 1;
}
else if (!longlong && !explicit_intN
&& decl_spec_seq_has_spec_p (declspecs, ds_complex))
{
type = double_type_node;
pedwarn (declspecs->locations[ds_complex], OPT_Wpedantic,
"ISO C++ does not support plain %<complex%> meaning "
"%<double complex%>");
}
}
explicit_int = declspecs->explicit_int_p;
explicit_char = declspecs->explicit_char_p;
#if 0
if (typedef_decl)
decl_attr = DECL_ATTRIBUTES (typedef_decl);
#endif
typedef_type = type;
if (sfk == sfk_conversion || sfk == sfk_deduction_guide)
ctor_return_type = TREE_TYPE (dname);
else
ctor_return_type = ctype;
if (sfk != sfk_none)
{
type = check_special_function_return_type (sfk, type,
ctor_return_type,
type_quals,
declspecs->locations);
type_quals = TYPE_UNQUALIFIED;
}
else if (type == NULL_TREE)
{
int is_main;
explicit_int = -1;
is_main = (funcdef_flag
&& dname && identifier_p (dname)
&& MAIN_NAME_P (dname)
&& ctype == NULL_TREE
&& in_namespace == NULL_TREE
&& current_namespace == global_namespace);
if (type_was_error_mark_node)
;
else if (in_system_header_at (input_location) || flag_ms_extensions)
;
else if (! is_main)
permerror (input_location, "ISO C++ forbids declaration of %qs with no type", name);
else if (pedantic)
pedwarn (input_location, OPT_Wpedantic,
"ISO C++ forbids declaration of %qs with no type", name);
else
warning (OPT_Wreturn_type,
"ISO C++ forbids declaration of %qs with no type", name);
if (type_was_error_mark_node && template_parm_flag)
type = error_mark_node;
else
type = integer_type_node;
}
ctype = NULL_TREE;
if (explicit_intN)
{
if (! int_n_enabled_p[declspecs->int_n_idx])
{
error ("%<__int%d%> is not supported by this target",
int_n_data[declspecs->int_n_idx].bitsize);
explicit_intN = false;
}
else if (pedantic && ! in_system_header_at (input_location))
pedwarn (input_location, OPT_Wpedantic,
"ISO C++ does not support %<__int%d%> for %qs",
int_n_data[declspecs->int_n_idx].bitsize, name);
}
if (long_p && !longlong && TYPE_MAIN_VARIANT (type) == double_type_node)
{
long_p = false;
type = cp_build_qualified_type (long_double_type_node,
cp_type_quals (type));
}
if (unsigned_p || signed_p || long_p || short_p)
{
int ok = 0;
if ((signed_p || unsigned_p) && TREE_CODE (type) != INTEGER_TYPE)
error ("%<signed%> or %<unsigned%> invalid for %qs", name);
else if (signed_p && unsigned_p)
error ("%<signed%> and %<unsigned%> specified together for %qs", name);
else if (longlong && TREE_CODE (type) != INTEGER_TYPE)
error ("%<long long%> invalid for %qs", name);
else if (long_p && TREE_CODE (type) == REAL_TYPE)
error ("%<long%> invalid for %qs", name);
else if (short_p && TREE_CODE (type) == REAL_TYPE)
error ("%<short%> invalid for %qs", name);
else if ((long_p || short_p) && TREE_CODE (type) != INTEGER_TYPE)
error ("%<long%> or %<short%> invalid for %qs", name);
else if ((long_p || short_p || explicit_char || explicit_int) && explicit_intN)
error ("%<long%>, %<int%>, %<short%>, or %<char%> invalid for %qs", name);
else if ((long_p || short_p) && explicit_char)
error ("%<long%> or %<short%> specified with char for %qs", name);
else if (long_p && short_p)
error ("%<long%> and %<short%> specified together for %qs", name);
else if (type == char16_type_node || type == char32_type_node)
{
if (signed_p || unsigned_p)
error ("%<signed%> or %<unsigned%> invalid for %qs", name);
else if (short_p || long_p)
error ("%<short%> or %<long%> invalid for %qs", name);
}
else
{
ok = 1;
if (!explicit_int && !defaulted_int && !explicit_char && !explicit_intN && pedantic)
{
pedwarn (input_location, OPT_Wpedantic, 
"long, short, signed or unsigned used invalidly for %qs",
name);
if (flag_pedantic_errors)
ok = 0;
}
}
if (! ok)
{
unsigned_p = false;
signed_p = false;
long_p = false;
short_p = false;
longlong = 0;
}
}
if (unsigned_p
|| (bitfield && !flag_signed_bitfields
&& !signed_p
&& !(typedef_decl
&& C_TYPEDEF_EXPLICITLY_SIGNED (typedef_decl))
&& TREE_CODE (type) == INTEGER_TYPE
&& !same_type_p (TYPE_MAIN_VARIANT (type), wchar_type_node)))
{
if (explicit_intN)
type = int_n_trees[declspecs->int_n_idx].unsigned_type;
else if (longlong)
type = long_long_unsigned_type_node;
else if (long_p)
type = long_unsigned_type_node;
else if (short_p)
type = short_unsigned_type_node;
else if (type == char_type_node)
type = unsigned_char_type_node;
else if (typedef_decl)
type = unsigned_type_for (type);
else
type = unsigned_type_node;
}
else if (signed_p && type == char_type_node)
type = signed_char_type_node;
else if (explicit_intN)
type = int_n_trees[declspecs->int_n_idx].signed_type;
else if (longlong)
type = long_long_integer_type_node;
else if (long_p)
type = long_integer_type_node;
else if (short_p)
type = short_integer_type_node;
if (decl_spec_seq_has_spec_p (declspecs, ds_complex))
{
if (TREE_CODE (type) != INTEGER_TYPE && TREE_CODE (type) != REAL_TYPE)
error ("complex invalid for %qs", name);
else if (type == integer_type_node)
type = complex_integer_type_node;
else if (type == float_type_node)
type = complex_float_type_node;
else if (type == double_type_node)
type = complex_double_type_node;
else if (type == long_double_type_node)
type = complex_long_double_type_node;
else
type = build_complex_type (type);
}
if (CLASS_TYPE_P (type)
&& DECL_SELF_REFERENCE_P (TYPE_NAME (type))
&& type == TREE_TYPE (TYPE_NAME (type))
&& (declarator || type_quals))
type = DECL_ORIGINAL_TYPE (TYPE_NAME (type));
type_quals |= cp_type_quals (type);
type = cp_build_qualified_type_real
(type, type_quals, ((((typedef_decl && !DECL_ARTIFICIAL (typedef_decl))
|| declspecs->decltype_p)
? tf_ignore_bad_quals : 0) | tf_warning_or_error));
type_quals = cp_type_quals (type);
if (cxx_dialect >= cxx17 && type && is_auto (type)
&& innermost_code != cdk_function
&& id_declarator && declarator != id_declarator)
if (tree tmpl = CLASS_PLACEHOLDER_TEMPLATE (type))
{
error_at (typespec_loc, "template placeholder type %qT must be followed "
"by a simple declarator-id", type);
inform (DECL_SOURCE_LOCATION (tmpl), "%qD declared here", tmpl);
}
staticp = 0;
inlinep = decl_spec_seq_has_spec_p (declspecs, ds_inline);
virtualp =  decl_spec_seq_has_spec_p (declspecs, ds_virtual);
explicitp = decl_spec_seq_has_spec_p (declspecs, ds_explicit);
storage_class = declspecs->storage_class;
if (storage_class == sc_static)
staticp = 1 + (decl_context == FIELD);
if (virtualp)
{
if (staticp == 2)
{
error ("member %qD cannot be declared both %<virtual%> "
"and %<static%>", dname);
storage_class = sc_none;
staticp = 0;
}
if (constexpr_p)
error ("member %qD cannot be declared both %<virtual%> "
"and %<constexpr%>", dname);
}
friendp = decl_spec_seq_has_spec_p (declspecs, ds_friend);
if (decl_context == PARM)
{
if (typedef_p)
{
error ("typedef declaration invalid in parameter declaration");
return error_mark_node;
}
else if (template_parm_flag && storage_class != sc_none)
{
error ("storage class specified for template parameter %qs", name);
return error_mark_node;
}
else if (storage_class == sc_static
|| storage_class == sc_extern
|| thread_p)
error ("storage class specifiers invalid in parameter declarations");
if (concept_p)
error ("a parameter cannot be declared %<concept%>");
else if (constexpr_p)
{
error ("a parameter cannot be declared %<constexpr%>");
constexpr_p = 0;
}
}
if (virtualp
&& (current_class_name == NULL_TREE || decl_context != FIELD))
{
error_at (declspecs->locations[ds_virtual],
"%<virtual%> outside class declaration");
virtualp = 0;
}
if (innermost_code == cdk_decomp)
{
location_t loc = (declarator->kind == cdk_reference
? declarator->declarator->id_loc : declarator->id_loc);
if (inlinep)
error_at (declspecs->locations[ds_inline],
"structured binding declaration cannot be %<inline%>");
if (typedef_p)
error_at (declspecs->locations[ds_typedef],
"structured binding declaration cannot be %<typedef%>");
if (constexpr_p)
error_at (declspecs->locations[ds_constexpr], "structured "
"binding declaration cannot be %<constexpr%>");
if (thread_p)
error_at (declspecs->locations[ds_thread],
"structured binding declaration cannot be %qs",
declspecs->gnu_thread_keyword_p
? "__thread" : "thread_local");
if (concept_p)
error_at (declspecs->locations[ds_concept],
"structured binding declaration cannot be %<concept%>");
switch (storage_class)
{
case sc_none:
break;
case sc_register:
error_at (loc, "structured binding declaration cannot be "
"%<register%>");
break;
case sc_static:
error_at (loc, "structured binding declaration cannot be "
"%<static%>");
break;
case sc_extern:
error_at (loc, "structured binding declaration cannot be "
"%<extern%>");
break;
case sc_mutable:
error_at (loc, "structured binding declaration cannot be "
"%<mutable%>");
break;
case sc_auto:
error_at (loc, "structured binding declaration cannot be "
"C++98 %<auto%>");
break;
default:
gcc_unreachable ();
}
if (TREE_CODE (type) != TEMPLATE_TYPE_PARM
|| TYPE_IDENTIFIER (type) != auto_identifier)
{
if (type != error_mark_node)
{
error_at (loc, "structured binding declaration cannot have "
"type %qT", type);
inform (loc,
"type must be cv-qualified %<auto%> or reference to "
"cv-qualified %<auto%>");
}
type = build_qualified_type (make_auto (), type_quals);
declspecs->type = type;
}
inlinep = 0;
typedef_p = 0;
constexpr_p = 0;
thread_p = 0;
concept_p = 0;
storage_class = sc_none;
staticp = 0;
declspecs->storage_class = sc_none;
declspecs->locations[ds_thread] = UNKNOWN_LOCATION;
}
if (staticp && decl_context == TYPENAME
&& declspecs->type
&& ANON_AGGR_TYPE_P (declspecs->type))
decl_context = FIELD;
if (thread_p
&& ((storage_class
&& storage_class != sc_extern
&& storage_class != sc_static)
|| typedef_p))
{
error ("multiple storage classes in declaration of %qs", name);
thread_p = false;
}
if (decl_context != NORMAL
&& ((storage_class != sc_none
&& storage_class != sc_mutable)
|| thread_p))
{
if ((decl_context == PARM || decl_context == CATCHPARM)
&& (storage_class == sc_register
|| storage_class == sc_auto))
;
else if (typedef_p)
;
else if (decl_context == FIELD
&& storage_class == sc_static)
;
else
{
if (decl_context == FIELD)
error ("storage class specified for %qs", name);
else
{
if (decl_context == PARM || decl_context == CATCHPARM)
error ("storage class specified for parameter %qs", name);
else
error ("storage class specified for typename");
}
if (storage_class == sc_register
|| storage_class == sc_auto
|| storage_class == sc_extern
|| thread_p)
storage_class = sc_none;
}
}
else if (storage_class == sc_extern && funcdef_flag
&& ! toplevel_bindings_p ())
error ("nested function %qs declared %<extern%>", name);
else if (toplevel_bindings_p ())
{
if (storage_class == sc_auto)
error ("top-level declaration of %qs specifies %<auto%>", name);
}
else if (thread_p
&& storage_class != sc_extern
&& storage_class != sc_static)
{
if (declspecs->gnu_thread_keyword_p)
pedwarn (input_location, 0, "function-scope %qs implicitly auto and "
"declared %<__thread%>", name);
storage_class = declspecs->storage_class = sc_static;
staticp = 1;
}
if (storage_class && friendp)
{
error ("storage class specifiers invalid in friend function declarations");
storage_class = sc_none;
staticp = 0;
}
if (!id_declarator)
unqualified_id = NULL_TREE;
else
{
unqualified_id = id_declarator->u.id.unqualified_name;
switch (TREE_CODE (unqualified_id))
{
case BIT_NOT_EXPR:
unqualified_id = TREE_OPERAND (unqualified_id, 0);
if (TYPE_P (unqualified_id))
unqualified_id = constructor_name (unqualified_id);
break;
case IDENTIFIER_NODE:
case TEMPLATE_ID_EXPR:
break;
default:
gcc_unreachable ();
}
}
if (declspecs->std_attributes)
{
location_t attr_loc = declspecs->locations[ds_std_attribute];
if (warning_at (attr_loc, OPT_Wattributes, "attribute ignored"))
inform (attr_loc, "an attribute that appertains to a type-specifier "
"is ignored");
}
for (; declarator; declarator = declarator->declarator)
{
const cp_declarator *inner_declarator;
tree attrs;
if (type == error_mark_node)
return error_mark_node;
attrs = declarator->attributes;
if (attrs)
{
int attr_flags;
attr_flags = 0;
if (declarator == NULL || declarator->kind == cdk_id)
attr_flags |= (int) ATTR_FLAG_DECL_NEXT;
if (declarator->kind == cdk_function)
attr_flags |= (int) ATTR_FLAG_FUNCTION_NEXT;
if (declarator->kind == cdk_array)
attr_flags |= (int) ATTR_FLAG_ARRAY_NEXT;
returned_attrs = decl_attributes (&type,
chainon (returned_attrs, attrs),
attr_flags);
}
inner_declarator = declarator->declarator;
if (decl_context != PARM
&& decl_context != TYPENAME
&& !typedef_p
&& declarator->parenthesized != UNKNOWN_LOCATION
&& !(inner_declarator
&& inner_declarator->kind == cdk_id
&& inner_declarator->u.id.qualifying_scope
&& (MAYBE_CLASS_TYPE_P (type)
|| TREE_CODE (type) == ENUMERAL_TYPE)))
warning_at (declarator->parenthesized, OPT_Wparentheses,
"unnecessary parentheses in declaration of %qs", name);
if (declarator->kind == cdk_id || declarator->kind == cdk_decomp)
break;
switch (declarator->kind)
{
case cdk_array:
type = create_array_type_for_decl (dname, type,
declarator->u.array.bounds);
if (!valid_array_size_p (input_location, type, dname))
type = error_mark_node;
if (declarator->std_attributes)
returned_attrs = chainon (returned_attrs,
declarator->std_attributes);
break;
case cdk_function:
{
tree arg_types;
int funcdecl_p;
input_location = declspecs->locations[ds_type_spec];
abstract_virtuals_error (ACU_RETURN, type);
input_location = saved_loc;
memfn_quals = declarator->u.function.qualifiers;
virt_specifiers = declarator->u.function.virt_specifiers;
rqual = declarator->u.function.ref_qualifier;
tree tx_qual = declarator->u.function.tx_qualifier;
raises = declarator->u.function.exception_specification;
if (raises == error_mark_node)
raises = NULL_TREE;
if (reqs)
error_at (location_of (reqs), "requires-clause on return type");
reqs = declarator->u.function.requires_clause;
funcdecl_p = inner_declarator && inner_declarator->kind == cdk_id;
tree late_return_type = declarator->u.function.late_return_type;
if (funcdecl_p)
{
if (tree auto_node = type_uses_auto (type))
{
if (!late_return_type)
{
if (current_class_type
&& LAMBDA_TYPE_P (current_class_type))
;
else if (cxx_dialect < cxx14)
{
error ("%qs function uses "
"%<auto%> type specifier without trailing "
"return type", name);
inform (input_location, "deduced return type "
"only available with -std=c++14 or "
"-std=gnu++14");
}
else if (virtualp)
{
error ("virtual function cannot "
"have deduced return type");
virtualp = false;
}
}
else if (!is_auto (type) && sfk != sfk_conversion)
{
error ("%qs function with trailing return type has"
" %qT as its type rather than plain %<auto%>",
name, type);
return error_mark_node;
}
tree tmpl = CLASS_PLACEHOLDER_TEMPLATE (auto_node);
if (!tmpl)
if (tree late_auto = type_uses_auto (late_return_type))
tmpl = CLASS_PLACEHOLDER_TEMPLATE (late_auto);
if (tmpl)
{
if (!dguide_name_p (unqualified_id))
{
error_at (declarator->id_loc, "deduced class "
"type %qD in function return type",
DECL_NAME (tmpl));
inform (DECL_SOURCE_LOCATION (tmpl),
"%qD declared here", tmpl);
return error_mark_node;
}
else if (!late_return_type)
{
error_at (declarator->id_loc, "deduction guide "
"for %qT must have trailing return "
"type", TREE_TYPE (tmpl));
inform (DECL_SOURCE_LOCATION (tmpl),
"%qD declared here", tmpl);
return error_mark_node;
}
else if (CLASS_TYPE_P (late_return_type)
&& CLASSTYPE_TEMPLATE_INFO (late_return_type)
&& (CLASSTYPE_TI_TEMPLATE (late_return_type)
== tmpl))
;
else
error ("trailing return type %qT of deduction guide "
"is not a specialization of %qT",
late_return_type, TREE_TYPE (tmpl));
}
}
else if (late_return_type
&& sfk != sfk_conversion)
{
if (cxx_dialect < cxx11)
error ("trailing return type only available with "
"-std=c++11 or -std=gnu++11");
else
error ("%qs function with trailing return type not "
"declared with %<auto%> type specifier", name);
return error_mark_node;
}
}
type = splice_late_return_type (type, late_return_type);
if (type == error_mark_node)
return error_mark_node;
if (late_return_type)
{
late_return_type_p = true;
type_quals = cp_type_quals (type);
}
if (type_quals != TYPE_UNQUALIFIED)
{
if (SCALAR_TYPE_P (type) || VOID_TYPE_P (type))
warning_at (typespec_loc, OPT_Wignored_qualifiers, "type "
"qualifiers ignored on function return type");
type_quals = TYPE_UNQUALIFIED;
}
if (TREE_CODE (type) == FUNCTION_TYPE)
{
error_at (typespec_loc, "%qs declared as function returning "
"a function", name);
return error_mark_node;
}
if (TREE_CODE (type) == ARRAY_TYPE)
{
error_at (typespec_loc, "%qs declared as function returning "
"an array", name);
return error_mark_node;
}
if (ctype == NULL_TREE
&& decl_context == FIELD
&& funcdecl_p
&& friendp == 0)
ctype = current_class_type;
if (ctype && (sfk == sfk_constructor
|| sfk == sfk_destructor))
{
if (staticp == 2)
error ((flags == DTOR_FLAG)
? G_("destructor cannot be static member function")
: G_("constructor cannot be static member function"));
if (memfn_quals)
{
error ((flags == DTOR_FLAG)
? G_("destructors may not be cv-qualified")
: G_("constructors may not be cv-qualified"));
memfn_quals = TYPE_UNQUALIFIED;
}
if (rqual)
{
maybe_warn_cpp0x (CPP0X_REF_QUALIFIER);
error ((flags == DTOR_FLAG)
? G_("destructors may not be ref-qualified")
: G_("constructors may not be ref-qualified"));
rqual = REF_QUAL_NONE;
}
if (decl_context == FIELD
&& !member_function_or_else (ctype,
current_class_type,
flags))
return error_mark_node;
if (flags != DTOR_FLAG)
{
if (explicitp == 1)
explicitp = 2;
if (virtualp)
{
permerror (input_location,
"constructors cannot be declared %<virtual%>");
virtualp = 0;
}
if (decl_context == FIELD
&& sfk != sfk_constructor)
return error_mark_node;
}
if (decl_context == FIELD)
staticp = 0;
}
else if (friendp)
{
if (virtualp)
{
error ("virtual functions cannot be friends");
friendp = 0;
}
if (decl_context == NORMAL)
error ("friend declaration not in class definition");
if (current_function_decl && funcdef_flag)
error ("can%'t define friend function %qs in a local "
"class definition",
name);
}
else if (ctype && sfk == sfk_conversion)
{
if (explicitp == 1)
{
maybe_warn_cpp0x (CPP0X_EXPLICIT_CONVERSION);
explicitp = 2;
}
if (late_return_type_p)
error ("a conversion function cannot have a trailing return type");
}
else if (sfk == sfk_deduction_guide)
{
if (explicitp == 1)
explicitp = 2;
}
tree pushed_scope = NULL_TREE;
if (funcdecl_p
&& decl_context != FIELD
&& inner_declarator->u.id.qualifying_scope
&& CLASS_TYPE_P (inner_declarator->u.id.qualifying_scope))
pushed_scope
= push_scope (inner_declarator->u.id.qualifying_scope);
arg_types = grokparms (declarator->u.function.parameters, &parms);
if (pushed_scope)
pop_scope (pushed_scope);
if (inner_declarator
&& inner_declarator->kind == cdk_id
&& inner_declarator->u.id.sfk == sfk_destructor
&& arg_types != void_list_node)
{
error ("destructors may not have parameters");
arg_types = void_list_node;
parms = NULL_TREE;
}
type = build_function_type (type, arg_types);
tree attrs = declarator->std_attributes;
if (tx_qual)
{
tree att = build_tree_list (tx_qual, NULL_TREE);
if (is_attribute_p ("transaction_safe", tx_qual))
attrs = chainon (attrs, att);
else
returned_attrs = chainon (returned_attrs, att);
}
if (attrs)
decl_attributes (&type, attrs, 0);
if (raises)
type = build_exception_variant (type, raises);
}
break;
case cdk_pointer:
case cdk_reference:
case cdk_ptrmem:
if (TREE_CODE (type) == REFERENCE_TYPE)
{
if (declarator->kind != cdk_reference)
{
error ("cannot declare pointer to %q#T", type);
type = TREE_TYPE (type);
}
else if (cxx_dialect == cxx98)
{
error ("cannot declare reference to %q#T", type);
type = TREE_TYPE (type);
}
}
else if (VOID_TYPE_P (type))
{
if (declarator->kind == cdk_reference)
error ("cannot declare reference to %q#T", type);
else if (declarator->kind == cdk_ptrmem)
error ("cannot declare pointer to %q#T member", type);
}
type_quals = TYPE_UNQUALIFIED;
gcc_assert (TREE_CODE (type) != METHOD_TYPE);
if (declarator->kind == cdk_ptrmem
&& TREE_CODE (type) == FUNCTION_TYPE)
{
memfn_quals |= type_memfn_quals (type);
type = build_memfn_type (type,
declarator->u.pointer.class_type,
memfn_quals,
rqual);
if (type == error_mark_node)
return error_mark_node;
rqual = REF_QUAL_NONE;
memfn_quals = TYPE_UNQUALIFIED;
}
if (TREE_CODE (type) == FUNCTION_TYPE
&& (type_memfn_quals (type) != TYPE_UNQUALIFIED
|| type_memfn_rqual (type) != REF_QUAL_NONE))
error (declarator->kind == cdk_reference
? G_("cannot declare reference to qualified function type %qT")
: G_("cannot declare pointer to qualified function type %qT"),
type);
if (!TYPE_NAME (type)
&& (decl_context == NORMAL || decl_context == FIELD)
&& at_function_scope_p ()
&& variably_modified_type_p (type, NULL_TREE))
{
TYPE_NAME (type) = build_decl (UNKNOWN_LOCATION, TYPE_DECL,
NULL_TREE, type);
add_decl_expr (TYPE_NAME (type));
}
if (declarator->kind == cdk_reference)
{
if (VOID_TYPE_P (type))
;
else if (TREE_CODE (type) == REFERENCE_TYPE)
{
if (declarator->u.reference.rvalue_ref)
;
else
type = cp_build_reference_type (TREE_TYPE (type), false);
}
else
type = cp_build_reference_type
(type, declarator->u.reference.rvalue_ref);
if (inner_declarator && inner_declarator->kind == cdk_reference)
error ("cannot declare reference to %q#T, which is not "
"a typedef or a template type argument", type);
}
else if (TREE_CODE (type) == METHOD_TYPE)
type = build_ptrmemfunc_type (build_pointer_type (type));
else if (declarator->kind == cdk_ptrmem)
{
gcc_assert (TREE_CODE (declarator->u.pointer.class_type)
!= NAMESPACE_DECL);
if (declarator->u.pointer.class_type == error_mark_node)
type = error_mark_node;
else
type = build_ptrmem_type (declarator->u.pointer.class_type,
type);
}
else
type = build_pointer_type (type);
if (declarator->u.pointer.qualifiers)
{
type
= cp_build_qualified_type (type,
declarator->u.pointer.qualifiers);
type_quals = cp_type_quals (type);
}
if (declarator->std_attributes)
decl_attributes (&type, declarator->std_attributes,
0);
ctype = NULL_TREE;
break;
case cdk_error:
break;
default:
gcc_unreachable ();
}
}
if (constexpr_p && innermost_code != cdk_function)
{
if (TREE_CODE (type) != REFERENCE_TYPE)
{
type_quals |= TYPE_QUAL_CONST;
type = cp_build_qualified_type (type, type_quals);
}
}
if (unqualified_id && TREE_CODE (unqualified_id) == TEMPLATE_ID_EXPR
&& TREE_CODE (type) != FUNCTION_TYPE
&& TREE_CODE (type) != METHOD_TYPE
&& !variable_template_p (TREE_OPERAND (unqualified_id, 0)))
{
error ("template-id %qD used as a declarator",
unqualified_id);
unqualified_id = dname;
}
if (declarator
&& declarator->kind == cdk_id
&& declarator->u.id.qualifying_scope
&& MAYBE_CLASS_TYPE_P (declarator->u.id.qualifying_scope))
{
ctype = declarator->u.id.qualifying_scope;
ctype = TYPE_MAIN_VARIANT (ctype);
template_count = num_template_headers_for_class (ctype);
if (ctype == current_class_type)
{
if (friendp)
{
permerror (input_location, "member functions are implicitly "
"friends of their class");
friendp = 0;
}
else
permerror (declarator->id_loc, 
"extra qualification %<%T::%> on member %qs",
ctype, name);
}
else if (
!COMPLETE_TYPE_P (ctype)
&& (
funcdef_flag
|| (!friendp && !CLASS_TYPE_P (ctype))
|| !(dependent_type_p (ctype)
|| currently_open_class (ctype)))
&& !complete_type_or_else (ctype, NULL_TREE))
return error_mark_node;
else if (TREE_CODE (type) == FUNCTION_TYPE)
{
if (current_class_type
&& (!friendp || funcdef_flag || initialized))
{
error (funcdef_flag || initialized
? G_("cannot define member function %<%T::%s%> "
"within %qT")
: G_("cannot declare member function %<%T::%s%> "
"within %qT"),
ctype, name, current_class_type);
return error_mark_node;
}
}
else if (typedef_p && current_class_type)
{
error ("cannot declare member %<%T::%s%> within %qT",
ctype, name, current_class_type);
return error_mark_node;
}
}
if (ctype == NULL_TREE && decl_context == FIELD && friendp == 0)
ctype = current_class_type;
if (returned_attrs)
{
if (attrlist)
*attrlist = chainon (returned_attrs, *attrlist);
else
attrlist = &returned_attrs;
}
if (declarator
&& declarator->kind == cdk_id
&& declarator->std_attributes
&& attrlist != NULL)
{
if (declarator->std_attributes != error_mark_node)
*attrlist = chainon (*attrlist, declarator->std_attributes);
else
gcc_assert (seen_error ());
}
if (parameter_pack_p)
{
if (decl_context == PARM)
type = make_pack_expansion (type);
else
error ("non-parameter %qs cannot be a parameter pack", name);
}
if ((decl_context == FIELD || decl_context == PARM)
&& !processing_template_decl
&& variably_modified_type_p (type, NULL_TREE))
{
if (decl_context == FIELD)
error ("data member may not have variably modified type %qT", type);
else
error ("parameter may not have variably modified type %qT", type);
type = error_mark_node;
}
if (explicitp == 1 || (explicitp && friendp))
{
if (!current_class_type)
error_at (declspecs->locations[ds_explicit],
"%<explicit%> outside class declaration");
else if (friendp)
error_at (declspecs->locations[ds_explicit],
"%<explicit%> in friend declaration");
else
error_at (declspecs->locations[ds_explicit],
"only declarations of constructors and conversion operators "
"can be %<explicit%>");
explicitp = 0;
}
if (storage_class == sc_mutable)
{
if (decl_context != FIELD || friendp)
{
error ("non-member %qs cannot be declared %<mutable%>", name);
storage_class = sc_none;
}
else if (decl_context == TYPENAME || typedef_p)
{
error ("non-object member %qs cannot be declared %<mutable%>", name);
storage_class = sc_none;
}
else if (TREE_CODE (type) == FUNCTION_TYPE
|| TREE_CODE (type) == METHOD_TYPE)
{
error ("function %qs cannot be declared %<mutable%>", name);
storage_class = sc_none;
}
else if (staticp)
{
error ("static %qs cannot be declared %<mutable%>", name);
storage_class = sc_none;
}
else if (type_quals & TYPE_QUAL_CONST)
{
error ("const %qs cannot be declared %<mutable%>", name);
storage_class = sc_none;
}
else if (TREE_CODE (type) == REFERENCE_TYPE)
{
permerror (input_location, "reference %qs cannot be declared "
"%<mutable%>", name);
storage_class = sc_none;
}
}
if (typedef_p && decl_context != TYPENAME)
{
tree decl;
if ((rqual || memfn_quals) && TREE_CODE (type) == FUNCTION_TYPE)
{
type = apply_memfn_quals (type, memfn_quals, rqual);
memfn_quals = TYPE_UNQUALIFIED;
rqual = REF_QUAL_NONE;
}
if (type_uses_auto (type))
{
error ("typedef declared %<auto%>");
type = error_mark_node;
}
if (reqs)
error_at (location_of (reqs), "requires-clause on typedef");
if (id_declarator && declarator->u.id.qualifying_scope)
{
error ("typedef name may not be a nested-name-specifier");
type = error_mark_node;
}
if (decl_context == FIELD)
decl = build_lang_decl (TYPE_DECL, unqualified_id, type);
else
decl = build_decl (input_location, TYPE_DECL, unqualified_id, type);
if (decl_context != FIELD)
{
if (!current_function_decl)
DECL_CONTEXT (decl) = FROB_CONTEXT (current_namespace);
else if (DECL_MAYBE_IN_CHARGE_CONSTRUCTOR_P (current_function_decl)
|| (DECL_MAYBE_IN_CHARGE_DESTRUCTOR_P
(current_function_decl)))
DECL_ABSTRACT_P (decl) = true;
}
else if (current_class_type
&& constructor_name_p (unqualified_id, current_class_type))
permerror (input_location, "ISO C++ forbids nested type %qD with same name "
"as enclosing class",
unqualified_id);
if (type != error_mark_node
&& unqualified_id
&& TYPE_NAME (type)
&& TREE_CODE (TYPE_NAME (type)) == TYPE_DECL
&& TYPE_UNNAMED_P (type)
&& declspecs->type_definition_p
&& attributes_naming_typedef_ok (*attrlist)
&& cp_type_quals (type) == TYPE_UNQUALIFIED)
name_unnamed_type (type, decl);
if (signed_p
|| (typedef_decl && C_TYPEDEF_EXPLICITLY_SIGNED (typedef_decl)))
C_TYPEDEF_EXPLICITLY_SIGNED (decl) = 1;
bad_specifiers (decl, BSP_TYPE, virtualp,
memfn_quals != TYPE_UNQUALIFIED,
inlinep, friendp, raises != NULL_TREE);
if (decl_spec_seq_has_spec_p (declspecs, ds_alias))
TYPE_DECL_ALIAS_P (decl) = 1;
return decl;
}
if (type && typedef_type
&& TREE_CODE (type) == ARRAY_TYPE && !TYPE_DOMAIN (type)
&& TYPE_MAIN_VARIANT (type) == TYPE_MAIN_VARIANT (typedef_type))
type = build_cplus_array_type (TREE_TYPE (type), NULL_TREE);
if (type == typedef_type && TREE_CODE (type) == FUNCTION_TYPE)
{
tree decls = NULL_TREE;
tree args;
for (args = TYPE_ARG_TYPES (type);
args && args != void_list_node;
args = TREE_CHAIN (args))
{
tree decl = cp_build_parm_decl (NULL_TREE, NULL_TREE,
TREE_VALUE (args));
DECL_CHAIN (decl) = decls;
decls = decl;
}
parms = nreverse (decls);
if (decl_context != TYPENAME)
{
memfn_quals |= type_memfn_quals (type);
rqual = type_memfn_rqual (type);
type_quals = TYPE_UNQUALIFIED;
}
}
if (decl_context == TYPENAME)
{
if (friendp)
{
if (inlinep)
{
error ("%<inline%> specified for friend class declaration");
inlinep = 0;
}
if (!current_aggr)
{
if (TREE_CODE (type) == TEMPLATE_TYPE_PARM)
permerror (input_location, "template parameters cannot be friends");
else if (TREE_CODE (type) == TYPENAME_TYPE)
permerror (input_location, "friend declaration requires class-key, "
"i.e. %<friend class %T::%D%>",
TYPE_CONTEXT (type), TYPENAME_TYPE_FULLNAME (type));
else
permerror (input_location, "friend declaration requires class-key, "
"i.e. %<friend %#T%>",
type);
}
if (type != integer_type_node)
{
if (current_class_type)
make_friend_class (current_class_type, TYPE_MAIN_VARIANT (type),
true);
else
error ("trying to make class %qT a friend of global scope",
type);
type = void_type_node;
}
}
else if (memfn_quals || rqual)
{
if (ctype == NULL_TREE
&& TREE_CODE (type) == METHOD_TYPE)
ctype = TYPE_METHOD_BASETYPE (type);
if (ctype)
type = build_memfn_type (type, ctype, memfn_quals, rqual);
else if ((template_type_arg || cxx_dialect >= cxx11)
&& TREE_CODE (type) == FUNCTION_TYPE)
type = apply_memfn_quals (type, memfn_quals, rqual);
else
error ("invalid qualifiers on non-member function type");
}
if (reqs)
error_at (location_of (reqs), "requires-clause on type-id");
return type;
}
else if (unqualified_id == NULL_TREE && decl_context != PARM
&& decl_context != CATCHPARM
&& TREE_CODE (type) != UNION_TYPE
&& ! bitfield
&& innermost_code != cdk_decomp)
{
error ("abstract declarator %qT used as declaration", type);
return error_mark_node;
}
if (!FUNC_OR_METHOD_TYPE_P (type))
{
if (dname && IDENTIFIER_ANY_OP_P (dname))
{
error ("declaration of %qD as non-function", dname);
return error_mark_node;
}
if (reqs)
error_at (location_of (reqs),
"requires-clause on declaration of non-function type %qT",
type);
}
if (decl_context != PARM)
{
type = check_var_type (unqualified_id, type);
if (type == error_mark_node)
return error_mark_node;
}
if (decl_context == PARM || decl_context == CATCHPARM)
{
if (ctype || in_namespace)
error ("cannot use %<::%> in parameter declaration");
if (type_uses_auto (type)
&& !(cxx_dialect >= cxx17 && template_parm_flag))
{
if (cxx_dialect >= cxx14)
error ("%<auto%> parameter not permitted in this context");
else
error ("parameter declared %<auto%>");
type = error_mark_node;
}
if (TREE_CODE (type) == ARRAY_TYPE)
{
type = build_pointer_type (TREE_TYPE (type));
type_quals = TYPE_UNQUALIFIED;
array_parameter_p = true;
}
else if (TREE_CODE (type) == FUNCTION_TYPE)
type = build_pointer_type (type);
}
if (ctype && TREE_CODE (type) == FUNCTION_TYPE && staticp < 2
&& !(identifier_p (unqualified_id)
&& IDENTIFIER_NEWDEL_OP_P (unqualified_id)))
{
cp_cv_quals real_quals = memfn_quals;
if (cxx_dialect < cxx14 && constexpr_p
&& sfk != sfk_constructor && sfk != sfk_destructor)
real_quals |= TYPE_QUAL_CONST;
type = build_memfn_type (type, ctype, real_quals, rqual);
}
{
tree decl = NULL_TREE;
if (decl_context == PARM)
{
decl = cp_build_parm_decl (NULL_TREE, unqualified_id, type);
DECL_ARRAY_PARAMETER_P (decl) = array_parameter_p;
bad_specifiers (decl, BSP_PARM, virtualp,
memfn_quals != TYPE_UNQUALIFIED,
inlinep, friendp, raises != NULL_TREE);
}
else if (decl_context == FIELD)
{
if (!staticp && !friendp && TREE_CODE (type) != METHOD_TYPE)
if (tree auto_node = type_uses_auto (type))
{
location_t loc = declspecs->locations[ds_type_spec];
if (CLASS_PLACEHOLDER_TEMPLATE (auto_node))
error_at (loc, "invalid use of template-name %qE without an "
"argument list",
CLASS_PLACEHOLDER_TEMPLATE (auto_node));
else
error_at (loc, "non-static data member declared with "
"placeholder %qT", auto_node);
type = error_mark_node;
}
if (!staticp && TREE_CODE (type) == ARRAY_TYPE
&& TYPE_DOMAIN (type) == NULL_TREE)
{
if (ctype
&& (TREE_CODE (ctype) == UNION_TYPE
|| TREE_CODE (ctype) == QUAL_UNION_TYPE))
{
error ("flexible array member in union");
type = error_mark_node;
}
else
{
if (in_system_header_at (input_location))
;
else if (name)
pedwarn (input_location, OPT_Wpedantic,
"ISO C++ forbids flexible array member %qs", name);
else
pedwarn (input_location, OPT_Wpedantic,
"ISO C++ forbids flexible array members");
type = build_cplus_array_type (TREE_TYPE (type), NULL_TREE);
}
}
if (type == error_mark_node)
{
decl = NULL_TREE;
}
else if (in_namespace && !friendp)
{
error ("invalid use of %<::%>");
return error_mark_node;
}
else if (TREE_CODE (type) == FUNCTION_TYPE
|| TREE_CODE (type) == METHOD_TYPE)
{
int publicp = 0;
tree function_context;
if (friendp == 0)
{
if (!ctype)
{
error ("declaration of function %qD in invalid context",
unqualified_id);
return error_mark_node;
}
if (virtualp && TREE_CODE (ctype) == UNION_TYPE)
{
error ("function %qD declared %<virtual%> inside a union",
unqualified_id);
return error_mark_node;
}
if (virtualp
&& identifier_p (unqualified_id)
&& IDENTIFIER_NEWDEL_OP_P (unqualified_id))
{
error ("%qD cannot be declared %<virtual%>, since it "
"is always static", unqualified_id);
virtualp = 0;
}
}
if (sfk == sfk_destructor)
{
tree uqname = id_declarator->u.id.unqualified_name;
if (!ctype)
{
gcc_assert (friendp);
error ("expected qualified name in friend declaration "
"for destructor %qD", uqname);
return error_mark_node;
}
if (!check_dtor_name (ctype, TREE_OPERAND (uqname, 0)))
{
error ("declaration of %qD as member of %qT",
uqname, ctype);
return error_mark_node;
}
if (concept_p)
{
error ("a destructor cannot be %<concept%>");
return error_mark_node;
}
if (constexpr_p)
{
error ("a destructor cannot be %<constexpr%>");
return error_mark_node;
}
}
else if (sfk == sfk_constructor && friendp && !ctype)
{
error ("expected qualified name in friend declaration "
"for constructor %qD",
id_declarator->u.id.unqualified_name);
return error_mark_node;
}
if (sfk == sfk_constructor)
if (concept_p)
{
error ("a constructor cannot be %<concept%>");
return error_mark_node;
}
if (concept_p)
{
error ("a concept cannot be a member function");
concept_p = false;
}
if (TREE_CODE (unqualified_id) == TEMPLATE_ID_EXPR)
{
tree tmpl = TREE_OPERAND (unqualified_id, 0);
if (variable_template_p (tmpl))
{
error ("specialization of variable template %qD "
"declared as function", tmpl);
inform (DECL_SOURCE_LOCATION (tmpl),
"variable template declared here");
return error_mark_node;
}
}
function_context = (ctype != NULL_TREE) ?
decl_function_context (TYPE_MAIN_DECL (ctype)) : NULL_TREE;
publicp = (! friendp || ! staticp)
&& function_context == NULL_TREE;
if (late_return_type_p)
TYPE_HAS_LATE_RETURN_TYPE (type) = 1;
decl = grokfndecl (ctype, type,
TREE_CODE (unqualified_id) != TEMPLATE_ID_EXPR
? unqualified_id : dname,
parms,
unqualified_id,
reqs,
virtualp, flags, memfn_quals, rqual, raises,
friendp ? -1 : 0, friendp, publicp,
inlinep | (2 * constexpr_p) | (4 * concept_p),
initialized == SD_DELETED, sfk,
funcdef_flag, template_count, in_namespace,
attrlist, declarator->id_loc);
decl = set_virt_specifiers (decl, virt_specifiers);
if (decl == NULL_TREE)
return error_mark_node;
#if 0
decl = build_decl_attribute_variant (decl, decl_attr);
#endif
if (explicitp == 2)
DECL_NONCONVERTING_P (decl) = 1;
}
else if (!staticp && !dependent_type_p (type)
&& !COMPLETE_TYPE_P (complete_type (type))
&& (!complete_or_array_type_p (type)
|| initialized == 0))
{
if (TREE_CODE (type) != ARRAY_TYPE
|| !COMPLETE_TYPE_P (TREE_TYPE (type)))
{
if (unqualified_id)
{
error ("field %qD has incomplete type %qT",
unqualified_id, type);
cxx_incomplete_type_inform (strip_array_types (type));
}
else
error ("name %qT has incomplete type", type);
type = error_mark_node;
decl = NULL_TREE;
}
}
else
{
if (friendp)
{
error ("%qE is neither function nor member function; "
"cannot be declared friend", unqualified_id);
return error_mark_node;
}
decl = NULL_TREE;
}
if (friendp)
{
if (ctype == current_class_type)
;  
else if (decl && DECL_NAME (decl))
{
if (template_class_depth (current_class_type) == 0)
{
decl = check_explicit_specialization
(unqualified_id, decl, template_count,
2 * funcdef_flag + 4);
if (decl == error_mark_node)
return error_mark_node;
}
decl = do_friend (ctype, unqualified_id, decl,
*attrlist, flags,
funcdef_flag);
return decl;
}
else
return error_mark_node;
}
if (decl == NULL_TREE)
{
if (staticp)
{
decl = build_lang_decl_loc (declarator
? declarator->id_loc
: input_location,
VAR_DECL, unqualified_id, type);
set_linkage_for_static_data_member (decl);
if (concept_p)
error ("static data member %qE declared %<concept%>",
unqualified_id);
else if (constexpr_p && !initialized)
{
error ("%<constexpr%> static data member %qD must have an "
"initializer", decl);
constexpr_p = false;
}
if (inlinep)
mark_inline_variable (decl);
if (!DECL_VAR_DECLARED_INLINE_P (decl)
&& !(cxx_dialect >= cxx17 && constexpr_p))
DECL_EXTERNAL (decl) = 1;
if (thread_p)
{
CP_DECL_THREAD_LOCAL_P (decl) = true;
if (!processing_template_decl)
set_decl_tls_model (decl, decl_default_tls_model (decl));
if (declspecs->gnu_thread_keyword_p)
SET_DECL_GNU_TLS_P (decl);
}
}
else
{
if (concept_p)
error ("non-static data member %qE declared %<concept%>",
unqualified_id);
else if (constexpr_p)
{
error ("non-static data member %qE declared %<constexpr%>",
unqualified_id);
constexpr_p = false;
}
decl = build_decl (input_location,
FIELD_DECL, unqualified_id, type);
DECL_NONADDRESSABLE_P (decl) = bitfield;
if (bitfield && !unqualified_id)
{
TREE_NO_WARNING (decl) = 1;
DECL_PADDING_P (decl) = 1;
}
if (storage_class == sc_mutable)
{
DECL_MUTABLE_P (decl) = 1;
storage_class = sc_none;
}
if (initialized)
{
maybe_warn_cpp0x (CPP0X_NSDMI);
if (declspecs->storage_class == sc_static)
DECL_INITIAL (decl) = error_mark_node;
}
}
bad_specifiers (decl, BSP_FIELD, virtualp,
memfn_quals != TYPE_UNQUALIFIED,
staticp ? false : inlinep, friendp,
raises != NULL_TREE);
}
}
else if (TREE_CODE (type) == FUNCTION_TYPE
|| TREE_CODE (type) == METHOD_TYPE)
{
tree original_name;
int publicp = 0;
if (!unqualified_id)
return error_mark_node;
if (TREE_CODE (unqualified_id) == TEMPLATE_ID_EXPR)
original_name = dname;
else
original_name = unqualified_id;
if (storage_class == sc_auto)
error ("storage class %<auto%> invalid for function %qs", name);
else if (storage_class == sc_register)
error ("storage class %<register%> invalid for function %qs", name);
else if (thread_p)
{
if (declspecs->gnu_thread_keyword_p)
error ("storage class %<__thread%> invalid for function %qs",
name);
else
error ("storage class %<thread_local%> invalid for function %qs",
name);
}
if (virt_specifiers)
error ("virt-specifiers in %qs not allowed outside a class definition", name);
if (! toplevel_bindings_p ()
&& (storage_class == sc_static
|| decl_spec_seq_has_spec_p (declspecs, ds_inline))
&& pedantic)
{
if (storage_class == sc_static)
pedwarn (input_location, OPT_Wpedantic, 
"%<static%> specifier invalid for function %qs "
"declared out of global scope", name);
else
pedwarn (input_location, OPT_Wpedantic, 
"%<inline%> specifier invalid for function %qs "
"declared out of global scope", name);
}
if (ctype == NULL_TREE)
{
if (virtualp)
{
error ("virtual non-class function %qs", name);
virtualp = 0;
}
else if (sfk == sfk_constructor
|| sfk == sfk_destructor)
{
error (funcdef_flag
? G_("%qs defined in a non-class scope")
: G_("%qs declared in a non-class scope"), name);
sfk = sfk_none;
}
}
publicp = (ctype != NULL_TREE
|| storage_class != sc_static);
if (late_return_type_p)
TYPE_HAS_LATE_RETURN_TYPE (type) = 1;
decl = grokfndecl (ctype, type, original_name, parms, unqualified_id,
reqs, virtualp, flags, memfn_quals, rqual, raises,
1, friendp,
publicp,
inlinep | (2 * constexpr_p) | (4 * concept_p),
initialized == SD_DELETED,
sfk,
funcdef_flag,
template_count, in_namespace, attrlist,
declarator->id_loc);
if (decl == NULL_TREE)
return error_mark_node;
if (explicitp == 2)
DECL_NONCONVERTING_P (decl) = 1;
if (staticp == 1)
{
int invalid_static = 0;
if (TREE_CODE (type) == METHOD_TYPE)
{
permerror (input_location, "cannot declare member function %qD to have "
"static linkage", decl);
invalid_static = 1;
}
else if (current_function_decl)
{
error ("cannot declare static function inside another function");
invalid_static = 1;
}
if (invalid_static)
{
staticp = 0;
storage_class = sc_none;
}
}
}
else
{
decl = grokvardecl (type, dname, unqualified_id,
declspecs,
initialized,
type_quals,
inlinep,
concept_p,
template_count,
ctype ? ctype : in_namespace);
if (decl == NULL_TREE)
return error_mark_node;
bad_specifiers (decl, BSP_VAR, virtualp,
memfn_quals != TYPE_UNQUALIFIED,
inlinep, friendp, raises != NULL_TREE);
if (ctype)
{
DECL_CONTEXT (decl) = ctype;
if (staticp == 1)
{
permerror (input_location, "%<static%> may not be used when defining "
"(as opposed to declaring) a static data member");
staticp = 0;
storage_class = sc_none;
}
if (storage_class == sc_register && TREE_STATIC (decl))
{
error ("static member %qD declared %<register%>", decl);
storage_class = sc_none;
}
if (storage_class == sc_extern && pedantic)
{
pedwarn (input_location, OPT_Wpedantic, 
"cannot explicitly declare member %q#D to have "
"extern linkage", decl);
storage_class = sc_none;
}
}
else if (constexpr_p && DECL_EXTERNAL (decl))
{
error ("declaration of %<constexpr%> variable %qD "
"is not a definition", decl);
constexpr_p = false;
}
if (inlinep)
mark_inline_variable (decl);
if (innermost_code == cdk_decomp)
{
gcc_assert (declarator && declarator->kind == cdk_decomp);
DECL_SOURCE_LOCATION (decl) = declarator->id_loc;
DECL_ARTIFICIAL (decl) = 1;
fit_decomposition_lang_decl (decl, NULL_TREE);
}
}
if (VAR_P (decl) && !initialized)
if (tree auto_node = type_uses_auto (type))
if (!CLASS_PLACEHOLDER_TEMPLATE (auto_node))
{
location_t loc = declspecs->locations[ds_type_spec];
error_at (loc, "declaration of %q#D has no initializer", decl);
TREE_TYPE (decl) = error_mark_node;
}
if (storage_class == sc_extern && initialized && !funcdef_flag)
{
if (toplevel_bindings_p ())
{
if (!(type_quals & TYPE_QUAL_CONST))
warning (0, "%qs initialized and declared %<extern%>", name);
}
else
{
error ("%qs has both %<extern%> and initializer", name);
return error_mark_node;
}
}
if (storage_class == sc_register)
{
DECL_REGISTER (decl) = 1;
if (TREE_CODE (decl) == PARM_DECL)
{
if (cxx_dialect >= cxx17)
pedwarn (DECL_SOURCE_LOCATION (decl), OPT_Wregister,
"ISO C++17 does not allow %<register%> storage "
"class specifier");
else
warning_at (DECL_SOURCE_LOCATION (decl), OPT_Wregister,
"%<register%> storage class specifier used");
}
}
else if (storage_class == sc_extern)
DECL_THIS_EXTERN (decl) = 1;
else if (storage_class == sc_static)
DECL_THIS_STATIC (decl) = 1;
if (constexpr_p && VAR_P (decl))
DECL_DECLARED_CONSTEXPR_P (decl) = true;
if (!processing_template_decl)
cp_apply_type_quals_to_decl (type_quals, decl);
return decl;
}
}

static void
require_complete_types_for_parms (tree parms)
{
for (; parms; parms = DECL_CHAIN (parms))
{
if (dependent_type_p (TREE_TYPE (parms)))
continue;
if (!VOID_TYPE_P (TREE_TYPE (parms))
&& complete_type_or_else (TREE_TYPE (parms), parms))
{
relayout_decl (parms);
DECL_ARG_TYPE (parms) = type_passed_as (TREE_TYPE (parms));
maybe_warn_parm_abi (TREE_TYPE (parms),
DECL_SOURCE_LOCATION (parms));
}
else
TREE_TYPE (parms) = error_mark_node;
}
}
int
local_variable_p (const_tree t)
{
if ((VAR_P (t)
&& !TYPE_P (CP_DECL_CONTEXT (t))
&& !DECL_NAMESPACE_SCOPE_P (t))
|| (TREE_CODE (t) == PARM_DECL))
return 1;
return 0;
}
static tree
local_variable_p_walkfn (tree *tp, int *walk_subtrees,
void * )
{
if (local_variable_p (*tp)
&& (!DECL_ARTIFICIAL (*tp) || DECL_NAME (*tp) == this_identifier))
return *tp;
else if (TYPE_P (*tp))
*walk_subtrees = 0;
return NULL_TREE;
}
tree
check_default_argument (tree decl, tree arg, tsubst_flags_t complain)
{
tree var;
tree decl_type;
if (TREE_CODE (arg) == DEFAULT_ARG)
return arg;
if (TYPE_P (decl))
{
decl_type = decl;
decl = NULL_TREE;
}
else
decl_type = TREE_TYPE (decl);
if (arg == error_mark_node
|| decl == error_mark_node
|| TREE_TYPE (arg) == error_mark_node
|| decl_type == error_mark_node)
return error_mark_node;
++cp_unevaluated_operand;
tree carg = BRACE_ENCLOSED_INITIALIZER_P (arg) ? unshare_expr (arg): arg;
perform_implicit_conversion_flags (decl_type, carg, complain,
LOOKUP_IMPLICIT);
--cp_unevaluated_operand;
if (TYPE_PTR_OR_PTRMEM_P (decl_type)
&& null_ptr_cst_p (arg))
return nullptr_node;
var = cp_walk_tree_without_duplicates (&arg, local_variable_p_walkfn, NULL);
if (var)
{
if (complain & tf_warning_or_error)
{
if (DECL_NAME (var) == this_identifier)
permerror (input_location, "default argument %qE uses %qD",
arg, var);
else
error ("default argument %qE uses local variable %qD", arg, var);
}
return error_mark_node;
}
return arg;
}
static tree
type_is_deprecated (tree type)
{
enum tree_code code;
if (TREE_DEPRECATED (type))
return type;
if (TYPE_NAME (type))
{
if (TREE_DEPRECATED (TYPE_NAME (type)))
return type;
else
return NULL_TREE;
}
if (OVERLOAD_TYPE_P (type) && type != TYPE_MAIN_VARIANT (type))
return type_is_deprecated (TYPE_MAIN_VARIANT (type));
code = TREE_CODE (type);
if (code == POINTER_TYPE || code == REFERENCE_TYPE
|| code == OFFSET_TYPE || code == FUNCTION_TYPE
|| code == METHOD_TYPE || code == ARRAY_TYPE)
return type_is_deprecated (TREE_TYPE (type));
if (TYPE_PTRMEMFUNC_P (type))
return type_is_deprecated
(TREE_TYPE (TREE_TYPE (TYPE_PTRMEMFUNC_FN_TYPE (type))));
return NULL_TREE;
}
tree
grokparms (tree parmlist, tree *parms)
{
tree result = NULL_TREE;
tree decls = NULL_TREE;
tree parm;
int any_error = 0;
for (parm = parmlist; parm != NULL_TREE; parm = TREE_CHAIN (parm))
{
tree type = NULL_TREE;
tree init = TREE_PURPOSE (parm);
tree decl = TREE_VALUE (parm);
if (parm == void_list_node)
break;
if (! decl || TREE_TYPE (decl) == error_mark_node)
continue;
type = TREE_TYPE (decl);
if (VOID_TYPE_P (type))
{
if (same_type_p (type, void_type_node)
&& !init
&& !DECL_NAME (decl) && !result
&& TREE_CHAIN (parm) == void_list_node)
break;
else if (cv_qualified_p (type))
error_at (DECL_SOURCE_LOCATION (decl),
"invalid use of cv-qualified type %qT in "
"parameter declaration", type);
else
error_at (DECL_SOURCE_LOCATION (decl),
"invalid use of type %<void%> in parameter "
"declaration");
type = error_mark_node;
TREE_TYPE (decl) = error_mark_node;
}
if (type != error_mark_node)
{
if (deprecated_state != DEPRECATED_SUPPRESS)
{
tree deptype = type_is_deprecated (type);
if (deptype)
cp_warn_deprecated_use (deptype);
}
type = cp_build_qualified_type (type, 0);
if (TREE_CODE (type) == METHOD_TYPE)
{
error ("parameter %qD invalidly declared method type", decl);
type = build_pointer_type (type);
TREE_TYPE (decl) = type;
}
else if (abstract_virtuals_error (decl, type))
any_error = 1;  
else if (cxx_dialect < cxx17 && POINTER_TYPE_P (type))
{
tree t = TREE_TYPE (type);
int ptr = TYPE_PTR_P (type);
while (1)
{
if (TYPE_PTR_P (t))
ptr = 1;
else if (TREE_CODE (t) != ARRAY_TYPE)
break;
else if (!TYPE_DOMAIN (t))
break;
t = TREE_TYPE (t);
}
if (TREE_CODE (t) == ARRAY_TYPE)
pedwarn (DECL_SOURCE_LOCATION (decl), OPT_Wpedantic,
ptr
? G_("parameter %qD includes pointer to array of "
"unknown bound %qT")
: G_("parameter %qD includes reference to array of "
"unknown bound %qT"),
decl, t);
}
if (any_error)
init = NULL_TREE;
else if (init && !processing_template_decl)
init = check_default_argument (decl, init, tf_warning_or_error);
}
DECL_CHAIN (decl) = decls;
decls = decl;
result = tree_cons (init, type, result);
}
decls = nreverse (decls);
result = nreverse (result);
if (parm)
result = chainon (result, void_list_node);
*parms = decls;
return result;
}

int
copy_fn_p (const_tree d)
{
tree args;
tree arg_type;
int result = 1;
gcc_assert (DECL_FUNCTION_MEMBER_P (d));
if (TREE_CODE (d) == TEMPLATE_DECL
|| (DECL_TEMPLATE_INFO (d)
&& DECL_MEMBER_TEMPLATE_P (DECL_TI_TEMPLATE (d))))
return 0;
args = FUNCTION_FIRST_USER_PARMTYPE (d);
if (!args)
return 0;
arg_type = TREE_VALUE (args);
if (arg_type == error_mark_node)
return 0;
if (TYPE_MAIN_VARIANT (arg_type) == DECL_CONTEXT (d))
{
result = -1;
}
else if (TREE_CODE (arg_type) == REFERENCE_TYPE
&& !TYPE_REF_IS_RVALUE (arg_type)
&& TYPE_MAIN_VARIANT (TREE_TYPE (arg_type)) == DECL_CONTEXT (d))
{
if (CP_TYPE_CONST_P (TREE_TYPE (arg_type)))
result = 2;
}
else
return 0;
args = TREE_CHAIN (args);
if (args && args != void_list_node && !TREE_PURPOSE (args))
return 0;
return result;
}
bool
move_fn_p (const_tree d)
{
gcc_assert (DECL_FUNCTION_MEMBER_P (d));
if (cxx_dialect == cxx98)
return false;
if (TREE_CODE (d) == TEMPLATE_DECL
|| (DECL_TEMPLATE_INFO (d)
&& DECL_MEMBER_TEMPLATE_P (DECL_TI_TEMPLATE (d))))
return 0;
return move_signature_fn_p (d);
}
bool
move_signature_fn_p (const_tree d)
{
tree args;
tree arg_type;
bool result = false;
args = FUNCTION_FIRST_USER_PARMTYPE (d);
if (!args)
return 0;
arg_type = TREE_VALUE (args);
if (arg_type == error_mark_node)
return 0;
if (TREE_CODE (arg_type) == REFERENCE_TYPE
&& TYPE_REF_IS_RVALUE (arg_type)
&& same_type_p (TYPE_MAIN_VARIANT (TREE_TYPE (arg_type)),
DECL_CONTEXT (d)))
result = true;
args = TREE_CHAIN (args);
if (args && args != void_list_node && !TREE_PURPOSE (args))
return false;
return result;
}
void
grok_special_member_properties (tree decl)
{
tree class_type;
if (!DECL_NONSTATIC_MEMBER_FUNCTION_P (decl))
return;
class_type = DECL_CONTEXT (decl);
if (IDENTIFIER_CTOR_P (DECL_NAME (decl)))
{
int ctor = copy_fn_p (decl);
if (!DECL_ARTIFICIAL (decl))
TYPE_HAS_USER_CONSTRUCTOR (class_type) = 1;
if (ctor > 0)
{
TYPE_HAS_COPY_CTOR (class_type) = 1;
if (user_provided_p (decl))
TYPE_HAS_COMPLEX_COPY_CTOR (class_type) = 1;
if (ctor > 1)
TYPE_HAS_CONST_COPY_CTOR (class_type) = 1;
}
else if (sufficient_parms_p (FUNCTION_FIRST_USER_PARMTYPE (decl)))
TYPE_HAS_DEFAULT_CONSTRUCTOR (class_type) = 1;
else if (move_fn_p (decl) && user_provided_p (decl))
TYPE_HAS_COMPLEX_MOVE_CTOR (class_type) = 1;
else if (is_list_ctor (decl))
TYPE_HAS_LIST_CTOR (class_type) = 1;
if (DECL_DECLARED_CONSTEXPR_P (decl)
&& !ctor && !move_fn_p (decl))
TYPE_HAS_CONSTEXPR_CTOR (class_type) = 1;
}
else if (DECL_NAME (decl) == assign_op_identifier)
{
int assop = copy_fn_p (decl);
if (assop)
{
TYPE_HAS_COPY_ASSIGN (class_type) = 1;
if (user_provided_p (decl))
TYPE_HAS_COMPLEX_COPY_ASSIGN (class_type) = 1;
if (assop != 1)
TYPE_HAS_CONST_COPY_ASSIGN (class_type) = 1;
}
else if (move_fn_p (decl) && user_provided_p (decl))
TYPE_HAS_COMPLEX_MOVE_ASSIGN (class_type) = 1;
}
else if (IDENTIFIER_CONV_OP_P (DECL_NAME (decl)))
TYPE_HAS_CONVERSION (class_type) = true;
}
bool
grok_ctor_properties (const_tree ctype, const_tree decl)
{
int ctor_parm = copy_fn_p (decl);
if (ctor_parm < 0)
{
error ("invalid constructor; you probably meant %<%T (const %T&)%>",
ctype, ctype);
return false;
}
return true;
}
bool
grok_op_properties (tree decl, bool complain)
{
tree argtypes = TYPE_ARG_TYPES (TREE_TYPE (decl));
bool methodp = TREE_CODE (TREE_TYPE (decl)) == METHOD_TYPE;
tree name = DECL_NAME (decl);
tree class_type = DECL_CONTEXT (decl);
if (class_type && !CLASS_TYPE_P (class_type))
class_type = NULL_TREE;
tree_code operator_code;
unsigned op_flags;
if (IDENTIFIER_CONV_OP_P (name))
{
operator_code = TYPE_EXPR;
op_flags = OVL_OP_FLAG_UNARY;
}
else
{
const ovl_op_info_t *ovl_op = IDENTIFIER_OVL_OP_INFO (name);
operator_code = ovl_op->tree_code;
op_flags = ovl_op->flags;
gcc_checking_assert (operator_code != ERROR_MARK);
DECL_OVERLOADED_OPERATOR_CODE_RAW (decl) = ovl_op->ovl_op_code;
}
if (op_flags & OVL_OP_FLAG_ALLOC)
{
if (class_type)
switch (op_flags)
{
case OVL_OP_FLAG_ALLOC:
TYPE_HAS_NEW_OPERATOR (class_type) = 1;
break;
case OVL_OP_FLAG_ALLOC | OVL_OP_FLAG_DELETE:
TYPE_GETS_DELETE (class_type) |= 1;
break;
case OVL_OP_FLAG_ALLOC | OVL_OP_FLAG_VEC:
TYPE_HAS_ARRAY_NEW_OPERATOR (class_type) = 1;
break;
case OVL_OP_FLAG_ALLOC | OVL_OP_FLAG_DELETE | OVL_OP_FLAG_VEC:
TYPE_GETS_DELETE (class_type) |= 2;
break;
default:
gcc_unreachable ();
}
if (DECL_NAMESPACE_SCOPE_P (decl))
{
if (CP_DECL_CONTEXT (decl) != global_namespace)
{
error ("%qD may not be declared within a namespace", decl);
return false;
}
if (!TREE_PUBLIC (decl))
{
error ("%qD may not be declared as static", decl);
return false;
}
}
if (op_flags & OVL_OP_FLAG_DELETE)
TREE_TYPE (decl) = coerce_delete_type (TREE_TYPE (decl));
else
{
DECL_IS_OPERATOR_NEW (decl) = 1;
TREE_TYPE (decl) = coerce_new_type (TREE_TYPE (decl));
}
return true;
}
if (! methodp || DECL_STATIC_FUNCTION_P (decl))
{
if (operator_code == TYPE_EXPR
|| operator_code == CALL_EXPR
|| operator_code == COMPONENT_REF
|| operator_code == ARRAY_REF
|| operator_code == NOP_EXPR)
{
error ("%qD must be a nonstatic member function", decl);
return false;
}
if (DECL_STATIC_FUNCTION_P (decl))
{
error ("%qD must be either a non-static member "
"function or a non-member function", decl);
return false;
}
for (tree arg = argtypes; ; arg = TREE_CHAIN (arg))
{
if (!arg || arg == void_list_node)
{
if (complain)
error ("%qD must have an argument of class or "
"enumerated type", decl);
return false;
}
tree type = non_reference (TREE_VALUE (arg));
if (type == error_mark_node)
return false;
if (MAYBE_CLASS_TYPE_P (type)
|| TREE_CODE (type) == ENUMERAL_TYPE)
break;
}
}
if (operator_code == CALL_EXPR)
return true;
if (operator_code == COND_EXPR)
{
error ("ISO C++ prohibits overloading operator ?:");
return false;
}
int arity = 0;
for (tree arg = argtypes; arg != void_list_node; arg = TREE_CHAIN (arg))
{
if (!arg)
{
error ("%qD must not have variable number of arguments", decl);
return false;
}
++arity;
}
switch (op_flags)
{
case OVL_OP_FLAG_AMBIARY:
if (arity == 1)
{
unsigned alt = ovl_op_alternate[ovl_op_mapping [operator_code]];
const ovl_op_info_t *ovl_op = &ovl_op_info[false][alt];
gcc_checking_assert (ovl_op->flags == OVL_OP_FLAG_UNARY);
operator_code = ovl_op->tree_code;
DECL_OVERLOADED_OPERATOR_CODE_RAW (decl) = ovl_op->ovl_op_code;
}
else if (arity != 2)
{
error (methodp
? G_("%qD must have either zero or one argument")
: G_("%qD must have either one or two arguments"), decl);
return false;
}
else if ((operator_code == POSTINCREMENT_EXPR
|| operator_code == POSTDECREMENT_EXPR)
&& ! processing_template_decl
&& ! same_type_p (TREE_VALUE (TREE_CHAIN (argtypes)),
integer_type_node))
{
error (methodp
? G_("postfix %qD must have %<int%> as its argument")
: G_("postfix %qD must have %<int%> as its second argument"),
decl);
return false;
}
break;
case OVL_OP_FLAG_UNARY:
if (arity != 1)
{
error (methodp
? G_("%qD must have no arguments")
: G_("%qD must have exactly one argument"), decl);
return false;
}
break;
case OVL_OP_FLAG_BINARY:
if (arity != 2)
{
error (methodp
? G_("%qD must have exactly one argument")
: G_("%qD must have exactly two arguments"), decl);
return false;
}
break;
default:
gcc_unreachable ();
}
for (tree arg = argtypes; arg != void_list_node; arg = TREE_CHAIN (arg))
if (TREE_PURPOSE (arg))
{
TREE_PURPOSE (arg) = NULL_TREE;
if (operator_code == POSTINCREMENT_EXPR
|| operator_code == POSTDECREMENT_EXPR)
pedwarn (input_location, OPT_Wpedantic,
"%qD cannot have default arguments", decl);
else
{
error ("%qD cannot have default arguments", decl);
return false;
}
}
if (class_type && class_type != current_class_type)
return true;
if (IDENTIFIER_CONV_OP_P (name)
&& ! DECL_TEMPLATE_INFO (decl)
&& warn_conversion)
{
tree t = TREE_TYPE (name);
int ref = (TREE_CODE (t) == REFERENCE_TYPE);
if (ref)
t = TYPE_MAIN_VARIANT (TREE_TYPE (t));
if (VOID_TYPE_P (t))
warning (OPT_Wconversion,
ref
? G_("conversion to a reference to void "
"will never use a type conversion operator")
: G_("conversion to void "
"will never use a type conversion operator"));
else if (class_type)
{
if (t == class_type)
warning (OPT_Wconversion,
ref
? G_("conversion to a reference to the same type "
"will never use a type conversion operator")
: G_("conversion to the same type "
"will never use a type conversion operator"));
else if (MAYBE_CLASS_TYPE_P (t)
&& COMPLETE_TYPE_P (t)
&& DERIVED_FROM_P (t, class_type))
warning (OPT_Wconversion,
ref
? G_("conversion to a reference to a base class "
"will never use a type conversion operator")
: G_("conversion to a base class "
"will never use a type conversion operator"));
}
}
if (!warn_ecpp)
return true;
if (operator_code == TRUTH_ANDIF_EXPR
|| operator_code == TRUTH_ORIF_EXPR
|| operator_code == COMPOUND_EXPR)
warning (OPT_Weffc__,
"user-defined %qD always evaluates both arguments", decl);
if (operator_code == POSTINCREMENT_EXPR
|| operator_code == POSTDECREMENT_EXPR
|| operator_code == PREINCREMENT_EXPR
|| operator_code == PREDECREMENT_EXPR)
{
tree arg = TREE_VALUE (argtypes);
tree ret = TREE_TYPE (TREE_TYPE (decl));
if (methodp || TREE_CODE (arg) == REFERENCE_TYPE)
arg = TREE_TYPE (arg);
arg = TYPE_MAIN_VARIANT (arg);
if (operator_code == PREINCREMENT_EXPR
|| operator_code == PREDECREMENT_EXPR)
{
if (TREE_CODE (ret) != REFERENCE_TYPE
|| !same_type_p (TYPE_MAIN_VARIANT (TREE_TYPE (ret)), arg))
warning (OPT_Weffc__, "prefix %qD should return %qT", decl,
build_reference_type (arg));
}
else
{
if (!same_type_p (TYPE_MAIN_VARIANT (ret), arg))
warning (OPT_Weffc__, "postfix %qD should return %qT", decl, arg);
}
}
if (!DECL_ASSIGNMENT_OPERATOR_P (decl)
&& (operator_code == PLUS_EXPR
|| operator_code == MINUS_EXPR
|| operator_code == TRUNC_DIV_EXPR
|| operator_code == MULT_EXPR
|| operator_code == TRUNC_MOD_EXPR)
&& TREE_CODE (TREE_TYPE (TREE_TYPE (decl))) == REFERENCE_TYPE)
warning (OPT_Weffc__, "%qD should return by value", decl);
return true;
}

static const char *
tag_name (enum tag_types code)
{
switch (code)
{
case record_type:
return "struct";
case class_type:
return "class";
case union_type:
return "union";
case enum_type:
return "enum";
case typename_type:
return "typename";
default:
gcc_unreachable ();
}
}
tree
check_elaborated_type_specifier (enum tag_types tag_code,
tree decl,
bool allow_template_p)
{
tree type;
if (DECL_SELF_REFERENCE_P (decl))
decl = TYPE_NAME (TREE_TYPE (decl));
type = TREE_TYPE (decl);
if (TREE_CODE (type) == TEMPLATE_TYPE_PARM)
{
error ("using template type parameter %qT after %qs",
type, tag_name (tag_code));
return error_mark_node;
}
else if (allow_template_p
&& (TREE_CODE (type) == BOUND_TEMPLATE_TEMPLATE_PARM
|| TREE_CODE (type) == TEMPLATE_TEMPLATE_PARM))
;
else if (!DECL_IMPLICIT_TYPEDEF_P (decl)
&& !DECL_SELF_REFERENCE_P (decl)
&& tag_code != typename_type)
{
if (alias_template_specialization_p (type))
error ("using alias template specialization %qT after %qs",
type, tag_name (tag_code));
else
error ("using typedef-name %qD after %qs", decl, tag_name (tag_code));
inform (DECL_SOURCE_LOCATION (decl),
"%qD has a previous declaration here", decl);
return error_mark_node;
}
else if (TREE_CODE (type) != RECORD_TYPE
&& TREE_CODE (type) != UNION_TYPE
&& tag_code != enum_type
&& tag_code != typename_type)
{
error ("%qT referred to as %qs", type, tag_name (tag_code));
inform (location_of (type), "%qT has a previous declaration here", type);
return error_mark_node;
}
else if (TREE_CODE (type) != ENUMERAL_TYPE
&& tag_code == enum_type)
{
error ("%qT referred to as enum", type);
inform (location_of (type), "%qT has a previous declaration here", type);
return error_mark_node;
}
else if (!allow_template_p
&& TREE_CODE (type) == RECORD_TYPE
&& CLASSTYPE_IS_TEMPLATE (type))
{
error ("template argument required for %<%s %T%>",
tag_name (tag_code),
DECL_NAME (CLASSTYPE_TI_TEMPLATE (type)));
return error_mark_node;
}
return type;
}
static tree
lookup_and_check_tag (enum tag_types tag_code, tree name,
tag_scope scope, bool template_header_p)
{
tree t;
tree decl;
if (scope == ts_global)
{
decl = lookup_name_prefer_type (name, 2);
decl = strip_using_decl (decl);
if (!decl)
decl = lookup_type_scope (name, ts_within_enclosing_non_class);
}
else
decl = lookup_type_scope (name, scope);
if (decl
&& (DECL_CLASS_TEMPLATE_P (decl)
|| (scope != ts_current
&& DECL_TEMPLATE_TEMPLATE_PARM_P (decl))))
decl = DECL_TEMPLATE_RESULT (decl);
if (decl && TREE_CODE (decl) == TYPE_DECL)
{
if (scope == ts_current && DECL_SELF_REFERENCE_P (decl))
{
error ("%qD has the same name as the class in which it is "
"declared",
decl);
return error_mark_node;
}
t = check_elaborated_type_specifier (tag_code,
decl,
template_header_p
| DECL_SELF_REFERENCE_P (decl));
if (template_header_p && t && CLASS_TYPE_P (t)
&& (!CLASSTYPE_TEMPLATE_INFO (t)
|| (!PRIMARY_TEMPLATE_P (CLASSTYPE_TI_TEMPLATE (t)))))
{
error ("%qT is not a template", t);
inform (location_of (t), "previous declaration here");
if (TYPE_CLASS_SCOPE_P (t)
&& CLASSTYPE_TEMPLATE_INFO (TYPE_CONTEXT (t)))
inform (input_location,
"perhaps you want to explicitly add %<%T::%>",
TYPE_CONTEXT (t));
t = error_mark_node;
}
return t;
}
else if (decl && TREE_CODE (decl) == TREE_LIST)
{
error ("reference to %qD is ambiguous", name);
print_candidates (decl);
return error_mark_node;
}
else
return NULL_TREE;
}
static tree
xref_tag_1 (enum tag_types tag_code, tree name,
tag_scope scope, bool template_header_p)
{
enum tree_code code;
tree context = NULL_TREE;
gcc_assert (identifier_p (name));
switch (tag_code)
{
case record_type:
case class_type:
code = RECORD_TYPE;
break;
case union_type:
code = UNION_TYPE;
break;
case enum_type:
code = ENUMERAL_TYPE;
break;
default:
gcc_unreachable ();
}
tree t = NULL_TREE;
if (scope != ts_lambda && !anon_aggrname_p (name))
t = lookup_and_check_tag  (tag_code, name, scope, template_header_p);
if (t == error_mark_node)
return error_mark_node;
if (scope != ts_current && t && current_class_type
&& template_class_depth (current_class_type)
&& template_header_p)
{
if (TREE_CODE (t) == TEMPLATE_TEMPLATE_PARM)
return t;
context = TYPE_CONTEXT (t);
t = NULL_TREE;
}
if (! t)
{
if (code == ENUMERAL_TYPE)
{
error ("use of enum %q#D without previous declaration", name);
return error_mark_node;
}
else
{
t = make_class_type (code);
TYPE_CONTEXT (t) = context;
if (scope == ts_lambda)
{
CLASSTYPE_LAMBDA_EXPR (t) = error_mark_node;
scope = ts_current;
}
t = pushtag (name, t, scope);
}
}
else
{
if (template_header_p && MAYBE_CLASS_TYPE_P (t))
{
tree constr = NULL_TREE;
if (current_template_parms)
{
tree reqs = TEMPLATE_PARMS_CONSTRAINTS (current_template_parms);
constr = build_constraints (reqs, NULL_TREE);
}
if (!redeclare_class_template (t, current_template_parms, constr))
return error_mark_node;
}
else if (!processing_template_decl
&& CLASS_TYPE_P (t)
&& CLASSTYPE_IS_TEMPLATE (t))
{
error ("redeclaration of %qT as a non-template", t);
inform (location_of (t), "previous declaration %qD", t);
return error_mark_node;
}
if (scope != ts_within_enclosing_non_class && TYPE_HIDDEN_P (t))
{
tree decl = TYPE_NAME (t);
DECL_ANTICIPATED (decl) = false;
DECL_FRIEND_P (decl) = false;
if (TYPE_TEMPLATE_INFO (t))
{
tree tmpl = TYPE_TI_TEMPLATE (t);
DECL_ANTICIPATED (tmpl) = false;
DECL_FRIEND_P (tmpl) = false;
}
}
}
return t;
}
tree
xref_tag (enum tag_types tag_code, tree name,
tag_scope scope, bool template_header_p)
{
tree ret;
bool subtime;
subtime = timevar_cond_start (TV_NAME_LOOKUP);
ret = xref_tag_1 (tag_code, name, scope, template_header_p);
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return ret;
}
tree
xref_tag_from_type (tree old, tree id, tag_scope scope)
{
enum tag_types tag_kind;
if (TREE_CODE (old) == RECORD_TYPE)
tag_kind = (CLASSTYPE_DECLARED_CLASS (old) ? class_type : record_type);
else
tag_kind  = union_type;
if (id == NULL_TREE)
id = TYPE_IDENTIFIER (old);
return xref_tag (tag_kind, id, scope, false);
}
void
xref_basetypes (tree ref, tree base_list)
{
tree *basep;
tree binfo, base_binfo;
unsigned max_vbases = 0; 
unsigned max_bases = 0;  
unsigned max_dvbases = 0; 
int i;
tree default_access;
tree igo_prev; 
if (ref == error_mark_node)
return;
default_access = (TREE_CODE (ref) == RECORD_TYPE
&& CLASSTYPE_DECLARED_CLASS (ref)
? access_private_node : access_public_node);
basep = &base_list;
while (*basep)
{
tree basetype = TREE_VALUE (*basep);
if (processing_template_decl
&& CLASS_TYPE_P (basetype) && TYPE_BEING_DEFINED (basetype))
cxx_incomplete_type_diagnostic (NULL_TREE, basetype, DK_PEDWARN);
if (!dependent_type_p (basetype)
&& !complete_type_or_else (basetype, NULL))
*basep = TREE_CHAIN (*basep);
else
{
max_bases++;
if (TREE_TYPE (*basep))
max_dvbases++;
if (CLASS_TYPE_P (basetype))
max_vbases += vec_safe_length (CLASSTYPE_VBASECLASSES (basetype));
basep = &TREE_CHAIN (*basep);
}
}
max_vbases += max_dvbases;
TYPE_MARKED_P (ref) = 1;
gcc_assert (!TYPE_BINFO (ref) || TYPE_SIZE (ref));
gcc_assert (TYPE_MAIN_VARIANT (ref) == ref);
binfo = make_tree_binfo (max_bases);
TYPE_BINFO (ref) = binfo;
BINFO_OFFSET (binfo) = size_zero_node;
BINFO_TYPE (binfo) = ref;
fixup_type_variants (ref);
if (max_bases)
{
vec_alloc (BINFO_BASE_ACCESSES (binfo), max_bases);
CLASSTYPE_NON_LAYOUT_POD_P (ref) = true;
if (TREE_CODE (ref) == UNION_TYPE)
{
error ("derived union %qT invalid", ref);
return;
}
}
if (max_bases > 1)
warning (OPT_Wmultiple_inheritance,
"%qT defined with multiple direct bases", ref);
if (max_vbases)
{
CLASSTYPE_NON_AGGREGATE (ref) = true;
vec_alloc (CLASSTYPE_VBASECLASSES (ref), max_vbases);
if (max_dvbases)
warning (OPT_Wvirtual_inheritance,
"%qT defined with direct virtual base", ref);
}
for (igo_prev = binfo; base_list; base_list = TREE_CHAIN (base_list))
{
tree access = TREE_PURPOSE (base_list);
int via_virtual = TREE_TYPE (base_list) != NULL_TREE;
tree basetype = TREE_VALUE (base_list);
if (access == access_default_node)
access = default_access;
if (cxx_dialect < cxx17
|| access != access_public_node
|| via_virtual)
CLASSTYPE_NON_AGGREGATE (ref) = true;
if (PACK_EXPANSION_P (basetype))
basetype = PACK_EXPANSION_PATTERN (basetype);
if (TREE_CODE (basetype) == TYPE_DECL)
basetype = TREE_TYPE (basetype);
if (!MAYBE_CLASS_TYPE_P (basetype) || TREE_CODE (basetype) == UNION_TYPE)
{
error ("base type %qT fails to be a struct or class type",
basetype);
goto dropped_base;
}
base_binfo = NULL_TREE;
if (CLASS_TYPE_P (basetype) && !dependent_scope_p (basetype))
{
base_binfo = TYPE_BINFO (basetype);
basetype = BINFO_TYPE (base_binfo);
TYPE_HAS_NEW_OPERATOR (ref)
|= TYPE_HAS_NEW_OPERATOR (basetype);
TYPE_HAS_ARRAY_NEW_OPERATOR (ref)
|= TYPE_HAS_ARRAY_NEW_OPERATOR (basetype);
TYPE_GETS_DELETE (ref) |= TYPE_GETS_DELETE (basetype);
TYPE_HAS_CONVERSION (ref) |= TYPE_HAS_CONVERSION (basetype);
CLASSTYPE_DIAMOND_SHAPED_P (ref)
|= CLASSTYPE_DIAMOND_SHAPED_P (basetype);
CLASSTYPE_REPEATED_BASE_P (ref)
|= CLASSTYPE_REPEATED_BASE_P (basetype);
}
if (TYPE_MARKED_P (basetype))
{
if (basetype == ref)
error ("recursive type %qT undefined", basetype);
else
error ("duplicate base type %qT invalid", basetype);
goto dropped_base;
}
if (PACK_EXPANSION_P (TREE_VALUE (base_list)))
basetype = make_pack_expansion (basetype);
TYPE_MARKED_P (basetype) = 1;
base_binfo = copy_binfo (base_binfo, basetype, ref,
&igo_prev, via_virtual);
if (!BINFO_INHERITANCE_CHAIN (base_binfo))
BINFO_INHERITANCE_CHAIN (base_binfo) = binfo;
BINFO_BASE_APPEND (binfo, base_binfo);
BINFO_BASE_ACCESS_APPEND (binfo, access);
continue;
dropped_base:
if (via_virtual)
max_vbases--;
if (CLASS_TYPE_P (basetype))
max_vbases
-= vec_safe_length (CLASSTYPE_VBASECLASSES (basetype));
}
if (CLASSTYPE_VBASECLASSES (ref)
&& max_vbases == 0)
vec_free (CLASSTYPE_VBASECLASSES (ref));
if (vec_safe_length (CLASSTYPE_VBASECLASSES (ref)) < max_vbases)
CLASSTYPE_DIAMOND_SHAPED_P (ref) = 1;
for (i = 0; BINFO_BASE_ITERATE (binfo, i, base_binfo); i++)
TYPE_MARKED_P (BINFO_TYPE (base_binfo)) = 0;
TYPE_MARKED_P (ref) = 0;
if (!CLASSTYPE_REPEATED_BASE_P (ref))
{
for (base_binfo = binfo; base_binfo;
base_binfo = TREE_CHAIN (base_binfo))
{
if (TYPE_MARKED_P (BINFO_TYPE (base_binfo)))
{
CLASSTYPE_REPEATED_BASE_P (ref) = 1;
break;
}
TYPE_MARKED_P (BINFO_TYPE (base_binfo)) = 1;
}
for (base_binfo = binfo; base_binfo;
base_binfo = TREE_CHAIN (base_binfo))
if (TYPE_MARKED_P (BINFO_TYPE (base_binfo)))
TYPE_MARKED_P (BINFO_TYPE (base_binfo)) = 0;
else
break;
}
}

static void
copy_type_enum (tree dst, tree src)
{
tree t;
for (t = dst; t; t = TYPE_NEXT_VARIANT (t))
{
TYPE_MIN_VALUE (t) = TYPE_MIN_VALUE (src);
TYPE_MAX_VALUE (t) = TYPE_MAX_VALUE (src);
TYPE_SIZE (t) = TYPE_SIZE (src);
TYPE_SIZE_UNIT (t) = TYPE_SIZE_UNIT (src);
SET_TYPE_MODE (dst, TYPE_MODE (src));
TYPE_PRECISION (t) = TYPE_PRECISION (src);
unsigned valign = TYPE_ALIGN (src);
if (TYPE_USER_ALIGN (t))
valign = MAX (valign, TYPE_ALIGN (t));
else
TYPE_USER_ALIGN (t) = TYPE_USER_ALIGN (src);
SET_TYPE_ALIGN (t, valign);
TYPE_UNSIGNED (t) = TYPE_UNSIGNED (src);
}
}
tree
start_enum (tree name, tree enumtype, tree underlying_type,
tree attributes, bool scoped_enum_p, bool *is_new)
{
tree prevtype = NULL_TREE;
gcc_assert (identifier_p (name));
if (is_new)
*is_new = false;
if (!underlying_type && scoped_enum_p)
underlying_type = integer_type_node;
if (underlying_type)
underlying_type = cv_unqualified (underlying_type);
if (!enumtype)
enumtype = lookup_and_check_tag (enum_type, name,
ts_current,
false);
if (enumtype && TREE_CODE (enumtype) == ENUMERAL_TYPE)
{
if (scoped_enum_p != SCOPED_ENUM_P (enumtype))
{
error_at (input_location, "scoped/unscoped mismatch "
"in enum %q#T", enumtype);
inform (DECL_SOURCE_LOCATION (TYPE_MAIN_DECL (enumtype)),
"previous definition here");
enumtype = error_mark_node;
}
else if (ENUM_FIXED_UNDERLYING_TYPE_P (enumtype) != !! underlying_type)
{
error_at (input_location, "underlying type mismatch "
"in enum %q#T", enumtype);
inform (DECL_SOURCE_LOCATION (TYPE_MAIN_DECL (enumtype)),
"previous definition here");
enumtype = error_mark_node;
}
else if (underlying_type && ENUM_UNDERLYING_TYPE (enumtype)
&& !dependent_type_p (underlying_type)
&& !dependent_type_p (ENUM_UNDERLYING_TYPE (enumtype))
&& !same_type_p (underlying_type,
ENUM_UNDERLYING_TYPE (enumtype)))
{
error_at (input_location, "different underlying type "
"in enum %q#T", enumtype);
inform (DECL_SOURCE_LOCATION (TYPE_MAIN_DECL (enumtype)),
"previous definition here");
underlying_type = NULL_TREE;
}
}
if (!enumtype || TREE_CODE (enumtype) != ENUMERAL_TYPE
|| processing_template_decl)
{
if (enumtype == error_mark_node)
{
name = make_anon_name ();
enumtype = NULL_TREE;
}
if (!enumtype)
{
if (is_new)
*is_new = true;
}
prevtype = enumtype;
if (!enumtype
|| TREE_CODE (enumtype) != ENUMERAL_TYPE
|| (underlying_type
&& dependent_type_p (underlying_type))
|| (ENUM_UNDERLYING_TYPE (enumtype)
&& dependent_type_p (ENUM_UNDERLYING_TYPE (enumtype))))
{
enumtype = cxx_make_type (ENUMERAL_TYPE);
enumtype = pushtag (name, enumtype, ts_current);
if (enumtype != error_mark_node
&& TYPE_CONTEXT (enumtype) == std_node
&& !strcmp ("byte", TYPE_NAME_STRING (enumtype)))
TYPE_ALIAS_SET (enumtype) = 0;
}
else
enumtype = xref_tag (enum_type, name, ts_current,
false);
if (enumtype == error_mark_node)
return error_mark_node;
SET_OPAQUE_ENUM_P (enumtype, true);
ENUM_FIXED_UNDERLYING_TYPE_P (enumtype) = !! underlying_type;
}
SET_SCOPED_ENUM_P (enumtype, scoped_enum_p);
cplus_decl_attributes (&enumtype, attributes, (int)ATTR_FLAG_TYPE_IN_PLACE);
if (underlying_type)
{
if (ENUM_UNDERLYING_TYPE (enumtype))
;
else if (CP_INTEGRAL_TYPE_P (underlying_type))
{
copy_type_enum (enumtype, underlying_type);
ENUM_UNDERLYING_TYPE (enumtype) = underlying_type;
}
else if (dependent_type_p (underlying_type))
ENUM_UNDERLYING_TYPE (enumtype) = underlying_type;
else
error ("underlying type %qT of %qT must be an integral type", 
underlying_type, enumtype);
}
if (prevtype && processing_template_decl)
return prevtype;
else
return enumtype;
}
void
finish_enum_value_list (tree enumtype)
{
tree values;
tree underlying_type;
tree decl;
tree value;
tree minnode, maxnode;
tree t;
bool fixed_underlying_type_p 
= ENUM_UNDERLYING_TYPE (enumtype) != NULL_TREE;
TYPE_VALUES (enumtype) = nreverse (TYPE_VALUES (enumtype));
if (processing_template_decl)
{
for (values = TYPE_VALUES (enumtype);
values;
values = TREE_CHAIN (values))
TREE_TYPE (TREE_VALUE (values)) = enumtype;
return;
}
if (TYPE_VALUES (enumtype))
{
minnode = maxnode = NULL_TREE;
for (values = TYPE_VALUES (enumtype);
values;
values = TREE_CHAIN (values))
{
decl = TREE_VALUE (values);
TREE_TYPE (decl) = enumtype;
value = DECL_INITIAL (decl);
if (value == error_mark_node)
value = integer_zero_node;
if (!minnode)
minnode = maxnode = value;
else if (tree_int_cst_lt (maxnode, value))
maxnode = value;
else if (tree_int_cst_lt (value, minnode))
minnode = value;
}
}
else
minnode = maxnode = integer_zero_node;
if (!fixed_underlying_type_p)
{
signop sgn = tree_int_cst_sgn (minnode) >= 0 ? UNSIGNED : SIGNED;
int lowprec = tree_int_cst_min_precision (minnode, sgn);
int highprec = tree_int_cst_min_precision (maxnode, sgn);
int precision = MAX (lowprec, highprec);
unsigned int itk;
bool use_short_enum;
use_short_enum = flag_short_enums
|| lookup_attribute ("packed", TYPE_ATTRIBUTES (enumtype));
if (TYPE_PRECISION (enumtype))
{
if (precision > TYPE_PRECISION (enumtype))
error ("specified mode too small for enumeral values");
else
{
use_short_enum = true;
precision = TYPE_PRECISION (enumtype);
}
}
for (itk = (use_short_enum ? itk_char : itk_int);
itk != itk_none;
itk++)
{
underlying_type = integer_types[itk];
if (underlying_type != NULL_TREE
&& TYPE_PRECISION (underlying_type) >= precision
&& TYPE_SIGN (underlying_type) == sgn)
break;
}
if (itk == itk_none)
{
error ("no integral type can represent all of the enumerator values "
"for %qT", enumtype);
precision = TYPE_PRECISION (long_long_integer_type_node);
underlying_type = integer_types[itk_unsigned_long_long];
}
copy_type_enum (enumtype, underlying_type);
ENUM_UNDERLYING_TYPE (enumtype)
= build_distinct_type_copy (underlying_type);
TYPE_PRECISION (ENUM_UNDERLYING_TYPE (enumtype)) = precision;
set_min_and_max_values_for_integral_type
(ENUM_UNDERLYING_TYPE (enumtype), precision, sgn);
if (flag_strict_enums)
set_min_and_max_values_for_integral_type (enumtype, precision, sgn);
}
else
underlying_type = ENUM_UNDERLYING_TYPE (enumtype);
for (values = TYPE_VALUES (enumtype); values; values = TREE_CHAIN (values))
{
location_t saved_location;
decl = TREE_VALUE (values);
saved_location = input_location;
input_location = DECL_SOURCE_LOCATION (decl);
if (fixed_underlying_type_p)
value = DECL_INITIAL (decl);
else
value = perform_implicit_conversion (underlying_type,
DECL_INITIAL (decl),
tf_warning_or_error);
input_location = saved_location;
if (value != error_mark_node)
{
value = copy_node (value);
TREE_TYPE (value) = enumtype;
}
DECL_INITIAL (decl) = value;
}
for (t = TYPE_MAIN_VARIANT (enumtype); t; t = TYPE_NEXT_VARIANT (t))
TYPE_VALUES (t) = TYPE_VALUES (enumtype);
if (at_class_scope_p ()
&& COMPLETE_TYPE_P (current_class_type)
&& UNSCOPED_ENUM_P (enumtype))
{
insert_late_enum_def_bindings (current_class_type, enumtype);
fixup_type_variants (current_class_type);
}
rest_of_type_compilation (enumtype, namespace_bindings_p ());
clear_cv_and_fold_caches ();
}
void
finish_enum (tree enumtype)
{
if (processing_template_decl)
{
if (at_function_scope_p ())
add_stmt (build_min (TAG_DEFN, enumtype));
return;
}
gcc_assert (enumtype == TYPE_MAIN_VARIANT (enumtype)
&& (TYPE_VALUES (enumtype)
|| !TYPE_NEXT_VARIANT (enumtype)));
}
void
build_enumerator (tree name, tree value, tree enumtype, tree attributes,
location_t loc)
{
tree decl;
tree context;
tree type;
if (processing_template_decl)
value = fold_non_dependent_expr (value);
if (value == error_mark_node)
value = NULL_TREE;
if (value)
STRIP_TYPE_NOPS (value);
if (! processing_template_decl)
{
if (value != NULL_TREE)
{
if (!ENUM_UNDERLYING_TYPE (enumtype))
{
tree tmp_value = build_expr_type_conversion (WANT_INT | WANT_ENUM,
value, true);
if (tmp_value)
value = tmp_value;
}
else if (! INTEGRAL_OR_UNSCOPED_ENUMERATION_TYPE_P
(TREE_TYPE (value)))
value = perform_implicit_conversion_flags
(ENUM_UNDERLYING_TYPE (enumtype), value, tf_warning_or_error,
LOOKUP_IMPLICIT | LOOKUP_NO_NARROWING);
if (value == error_mark_node)
value = NULL_TREE;
if (value != NULL_TREE)
{
if (! INTEGRAL_OR_UNSCOPED_ENUMERATION_TYPE_P
(TREE_TYPE (value)))
{
error ("enumerator value for %qD must have integral or "
"unscoped enumeration type", name);
value = NULL_TREE;
}
else
{
value = cxx_constant_value (value);
if (TREE_CODE (value) != INTEGER_CST)
{
error ("enumerator value for %qD is not an integer "
"constant", name);
value = NULL_TREE;
}
}
}
}
if (value == NULL_TREE)
{
if (TYPE_VALUES (enumtype))
{
tree prev_value;
bool overflowed;
prev_value = DECL_INITIAL (TREE_VALUE (TYPE_VALUES (enumtype)));
if (error_operand_p (prev_value))
value = error_mark_node;
else
{
tree type = TREE_TYPE (prev_value);
signop sgn = TYPE_SIGN (type);
widest_int wi = wi::add (wi::to_widest (prev_value), 1, sgn,
&overflowed);
if (!overflowed)
{
bool pos = !wi::neg_p (wi, sgn);
if (!wi::fits_to_tree_p (wi, type))
{
unsigned int itk;
for (itk = itk_int; itk != itk_none; itk++)
{
type = integer_types[itk];
if (type != NULL_TREE
&& (pos || !TYPE_UNSIGNED (type))
&& wi::fits_to_tree_p (wi, type))
break;
}
if (type && cxx_dialect < cxx11
&& itk > itk_unsigned_long)
pedwarn (input_location, OPT_Wlong_long,
pos ? G_("\
incremented enumerator value is too large for %<unsigned long%>") : G_("\
incremented enumerator value is too large for %<long%>"));
}
if (type == NULL_TREE)
overflowed = true;
else
value = wide_int_to_tree (type, wi);
}
if (overflowed)
{
error ("overflow in enumeration values at %qD", name);
value = error_mark_node;
}
}
}
else
value = integer_zero_node;
}
STRIP_TYPE_NOPS (value);
if (ENUM_UNDERLYING_TYPE (enumtype)
&& value
&& TREE_CODE (value) == INTEGER_CST)
{
if (!int_fits_type_p (value, ENUM_UNDERLYING_TYPE (enumtype)))
error ("enumerator value %qE is outside the range of underlying "
"type %qT", value, ENUM_UNDERLYING_TYPE (enumtype));
value = fold_convert (ENUM_UNDERLYING_TYPE (enumtype), value);
}
}
context = current_scope ();
type = value ? TREE_TYPE (value) : NULL_TREE;
decl = build_decl (loc, CONST_DECL, name, type);
DECL_CONTEXT (decl) = enumtype;
TREE_CONSTANT (decl) = 1;
TREE_READONLY (decl) = 1;
DECL_INITIAL (decl) = value;
if (attributes)
cplus_decl_attributes (&decl, attributes, 0);
if (context && context == current_class_type && !SCOPED_ENUM_P (enumtype))
{
tree saved_cas = current_access_specifier;
if (TREE_PRIVATE (TYPE_NAME (enumtype)))
current_access_specifier = access_private_node;
else if (TREE_PROTECTED (TYPE_NAME (enumtype)))
current_access_specifier = access_protected_node;
else
current_access_specifier = access_public_node;
finish_member_declaration (decl);
current_access_specifier = saved_cas;
}
else
pushdecl (decl);
TYPE_VALUES (enumtype) = tree_cons (name, decl, TYPE_VALUES (enumtype));
}
tree
lookup_enumerator (tree enumtype, tree name)
{
tree e;
gcc_assert (enumtype && TREE_CODE (enumtype) == ENUMERAL_TYPE);
e = purpose_member (name, TYPE_VALUES (enumtype));
return e? TREE_VALUE (e) : NULL_TREE;
}

static void
check_function_type (tree decl, tree current_function_parms)
{
tree fntype = TREE_TYPE (decl);
tree return_type = complete_type (TREE_TYPE (fntype));
require_complete_types_for_parms (current_function_parms);
if (dependent_type_p (return_type)
|| type_uses_auto (return_type))
return;
if (!COMPLETE_OR_VOID_TYPE_P (return_type))
{
tree args = TYPE_ARG_TYPES (fntype);
error ("return type %q#T is incomplete", return_type);
if (TREE_CODE (fntype) == METHOD_TYPE)
fntype = build_method_type_directly (TREE_TYPE (TREE_VALUE (args)),
void_type_node,
TREE_CHAIN (args));
else
fntype = build_function_type (void_type_node, args);
fntype
= build_exception_variant (fntype,
TYPE_RAISES_EXCEPTIONS (TREE_TYPE (decl)));
fntype = (cp_build_type_attribute_variant
(fntype, TYPE_ATTRIBUTES (TREE_TYPE (decl))));
TREE_TYPE (decl) = fntype;
}
else
{
abstract_virtuals_error (decl, TREE_TYPE (fntype));
maybe_warn_parm_abi (TREE_TYPE (fntype),
DECL_SOURCE_LOCATION (decl));
}
}
static bool
implicit_default_ctor_p (tree fn)
{
return (DECL_CONSTRUCTOR_P (fn)
&& !user_provided_p (fn)
&& sufficient_parms_p (FUNCTION_FIRST_USER_PARMTYPE (fn)));
}
static tree
build_clobber_this ()
{
if (is_empty_class (current_class_type))
return void_node;
bool vbases = CLASSTYPE_VBASECLASSES (current_class_type);
tree ctype = current_class_type;
if (!vbases)
ctype = CLASSTYPE_AS_BASE (ctype);
tree clobber = build_constructor (ctype, NULL);
TREE_THIS_VOLATILE (clobber) = true;
tree thisref = current_class_ref;
if (ctype != current_class_type)
{
thisref = build_nop (build_reference_type (ctype), current_class_ptr);
thisref = convert_from_reference (thisref);
}
tree exprstmt = build2 (MODIFY_EXPR, void_type_node, thisref, clobber);
if (vbases)
exprstmt = build_if_in_charge (exprstmt);
return exprstmt;
}
bool
start_preparsed_function (tree decl1, tree attrs, int flags)
{
tree ctype = NULL_TREE;
tree fntype;
tree restype;
int doing_friend = 0;
cp_binding_level *bl;
tree current_function_parms;
struct c_fileinfo *finfo
= get_fileinfo (LOCATION_FILE (DECL_SOURCE_LOCATION (decl1)));
bool honor_interface;
gcc_assert (VOID_TYPE_P (TREE_VALUE (void_list_node)));
gcc_assert (TREE_CHAIN (void_list_node) == NULL_TREE);
fntype = TREE_TYPE (decl1);
if (TREE_CODE (fntype) == METHOD_TYPE)
ctype = TYPE_METHOD_BASETYPE (fntype);
if (!ctype && DECL_FRIEND_P (decl1))
{
ctype = DECL_FRIEND_CONTEXT (decl1);
if (ctype && TREE_CODE (ctype) != RECORD_TYPE)
ctype = NULL_TREE;
else
doing_friend = 1;
}
if (DECL_DECLARED_INLINE_P (decl1)
&& lookup_attribute ("noinline", attrs))
warning_at (DECL_SOURCE_LOCATION (decl1), 0,
"inline function %qD given attribute noinline", decl1);
if (GNU_INLINE_P (decl1))
{
DECL_EXTERNAL (decl1) = 1;
DECL_NOT_REALLY_EXTERN (decl1) = 0;
DECL_INTERFACE_KNOWN (decl1) = 1;
DECL_DISREGARD_INLINE_LIMITS (decl1) = 1;
}
if (DECL_MAYBE_IN_CHARGE_CONSTRUCTOR_P (decl1))
adjust_clone_args (decl1);
gcc_assert (!(ctype != NULL_TREE && DECL_STATIC_FUNCTION_P (decl1)
&& TREE_CODE (TREE_TYPE (decl1)) == METHOD_TYPE));
if (ctype)
push_nested_class (ctype);
else if (DECL_STATIC_FUNCTION_P (decl1))
push_nested_class (DECL_CONTEXT (decl1));
if (flags & SF_INCLASS_INLINE)
maybe_begin_member_template_processing (decl1);
if (warn_ecpp
&& DECL_ASSIGNMENT_OPERATOR_P (decl1)
&& DECL_OVERLOADED_OPERATOR_IS (decl1, NOP_EXPR)
&& VOID_TYPE_P (TREE_TYPE (fntype)))
warning (OPT_Weffc__,
"%<operator=%> should return a reference to %<*this%>");
if (!DECL_INITIAL (decl1))
DECL_INITIAL (decl1) = error_mark_node;
TREE_STATIC (decl1) = 1;
if (processing_template_decl)
{
tree newdecl1 = push_template_decl (decl1);
if (newdecl1 == error_mark_node)
{
if (ctype || DECL_STATIC_FUNCTION_P (decl1))
pop_nested_class ();
return false;
}
decl1 = newdecl1;
}
check_function_type (decl1, DECL_ARGUMENTS (decl1));
restype = TREE_TYPE (fntype);
if (DECL_RESULT (decl1) == NULL_TREE)
{
tree resdecl;
resdecl = build_decl (input_location, RESULT_DECL, 0, restype);
DECL_ARTIFICIAL (resdecl) = 1;
DECL_IGNORED_P (resdecl) = 1;
DECL_RESULT (decl1) = resdecl;
cp_apply_type_quals_to_decl (cp_type_quals (restype), resdecl);
}
if (!processing_template_decl && !(flags & SF_PRE_PARSED))
{
if (!DECL_FUNCTION_MEMBER_P (decl1)
&& !(DECL_USE_TEMPLATE (decl1) &&
PRIMARY_TEMPLATE_P (DECL_TI_TEMPLATE (decl1))))
{
tree olddecl = pushdecl (decl1);
if (olddecl == error_mark_node)
;
else
{
if (warn_missing_declarations
&& olddecl == decl1
&& !DECL_MAIN_P (decl1)
&& TREE_PUBLIC (decl1)
&& !DECL_DECLARED_INLINE_P (decl1))
{
tree context;
for (context = DECL_CONTEXT (decl1);
context;
context = DECL_CONTEXT (context))
{
if (TREE_CODE (context) == NAMESPACE_DECL
&& DECL_NAME (context) == NULL_TREE)
break;
}
if (context == NULL)
warning_at (DECL_SOURCE_LOCATION (decl1),
OPT_Wmissing_declarations,
"no previous declaration for %qD", decl1);
}
decl1 = olddecl;
}
}
else
{
if (!DECL_CONTEXT (decl1) && DECL_TEMPLATE_INFO (decl1))
DECL_CONTEXT (decl1) = DECL_CONTEXT (DECL_TI_TEMPLATE (decl1));
}
fntype = TREE_TYPE (decl1);
restype = TREE_TYPE (fntype);
if (DECL_FILE_SCOPE_P (decl1))
maybe_apply_pragma_weak (decl1);
}
current_function_decl = decl1;
current_function_parms = DECL_ARGUMENTS (decl1);
announce_function (decl1);
gcc_assert (DECL_INITIAL (decl1));
if (DECL_INITIAL (decl1) != error_mark_node)
return true;
bl = current_binding_level;
allocate_struct_function (decl1, processing_template_decl);
cfun->language = ggc_cleared_alloc<language_function> ();
current_stmt_tree ()->stmts_are_full_exprs_p = 1;
current_binding_level = bl;
if (!processing_template_decl && type_uses_auto (restype))
{
FNDECL_USED_AUTO (decl1) = true;
current_function_auto_return_pattern = restype;
}
DECL_SAVED_TREE (decl1) = push_stmt_list ();
if (!DECL_PENDING_INLINE_P (decl1))
DECL_SAVED_FUNCTION_DATA (decl1) = NULL;
if (ctype && !doing_friend && !DECL_STATIC_FUNCTION_P (decl1))
{
tree t = DECL_ARGUMENTS (decl1);
gcc_assert (t != NULL_TREE && TREE_CODE (t) == PARM_DECL);
gcc_assert (TYPE_PTR_P (TREE_TYPE (t)));
cp_function_chain->x_current_class_ref
= cp_build_fold_indirect_ref (t);
cp_function_chain->x_current_class_ptr = t;
t = DECL_CHAIN (t);
if (DECL_HAS_IN_CHARGE_PARM_P (decl1))
{
current_in_charge_parm = t;
t = DECL_CHAIN (t);
}
if (DECL_HAS_VTT_PARM_P (decl1))
{
gcc_assert (DECL_NAME (t) == vtt_parm_identifier);
current_vtt_parm = t;
}
}
honor_interface = (!DECL_TEMPLATE_INSTANTIATION (decl1)
&& !DECL_ARTIFICIAL (decl1));
if (processing_template_decl)
;
else if (DECL_INTERFACE_KNOWN (decl1))
{
tree ctx = decl_function_context (decl1);
if (DECL_NOT_REALLY_EXTERN (decl1))
DECL_EXTERNAL (decl1) = 0;
if (ctx != NULL_TREE && vague_linkage_p (ctx))
comdat_linkage (decl1);
}
else if (!finfo->interface_unknown && honor_interface)
{
if (DECL_DECLARED_INLINE_P (decl1)
|| DECL_TEMPLATE_INSTANTIATION (decl1))
{
DECL_EXTERNAL (decl1)
= (finfo->interface_only
|| (DECL_DECLARED_INLINE_P (decl1)
&& ! flag_implement_inlines
&& !DECL_VINDEX (decl1)));
maybe_make_one_only (decl1);
}
else
DECL_EXTERNAL (decl1) = 0;
DECL_INTERFACE_KNOWN (decl1) = 1;
if (!DECL_EXTERNAL (decl1))
mark_needed (decl1);
}
else if (finfo->interface_unknown && finfo->interface_only
&& honor_interface)
{
comdat_linkage (decl1);
DECL_EXTERNAL (decl1) = 0;
DECL_INTERFACE_KNOWN (decl1) = 1;
DECL_DEFER_OUTPUT (decl1) = 1;
}
else
{
if (!GNU_INLINE_P (decl1))
DECL_EXTERNAL (decl1) = 0;
if ((DECL_DECLARED_INLINE_P (decl1)
|| DECL_TEMPLATE_INSTANTIATION (decl1))
&& ! DECL_INTERFACE_KNOWN (decl1))
DECL_DEFER_OUTPUT (decl1) = 1;
else
DECL_INTERFACE_KNOWN (decl1) = 1;
}
if (!DECL_CLONED_FUNCTION_P (decl1))
determine_visibility (decl1);
if (!processing_template_decl)
maybe_instantiate_noexcept (decl1);
begin_scope (sk_function_parms, decl1);
++function_depth;
if (DECL_DESTRUCTOR_P (decl1)
|| (DECL_CONSTRUCTOR_P (decl1)
&& targetm.cxx.cdtor_returns_this ()))
{
cdtor_label = create_artificial_label (input_location);
LABEL_DECL_CDTOR (cdtor_label) = true;
}
start_fname_decls ();
store_parm_decls (current_function_parms);
if (!processing_template_decl
&& (flag_lifetime_dse > 1)
&& DECL_CONSTRUCTOR_P (decl1)
&& !DECL_CLONED_FUNCTION_P (decl1)
&& !is_empty_class (current_class_type)
&& !implicit_default_ctor_p (decl1))
finish_expr_stmt (build_clobber_this ());
if (!processing_template_decl
&& DECL_CONSTRUCTOR_P (decl1)
&& sanitize_flags_p (SANITIZE_VPTR)
&& !DECL_CLONED_FUNCTION_P (decl1)
&& !implicit_default_ctor_p (decl1))
cp_ubsan_maybe_initialize_vtbl_ptrs (current_class_ptr);
start_lambda_scope (decl1);
return true;
}
bool
start_function (cp_decl_specifier_seq *declspecs,
const cp_declarator *declarator,
tree attrs)
{
tree decl1;
decl1 = grokdeclarator (declarator, declspecs, FUNCDEF, 1, &attrs);
invoke_plugin_callbacks (PLUGIN_START_PARSE_FUNCTION, decl1);
if (decl1 == error_mark_node)
return false;
if (decl1 == NULL_TREE || TREE_CODE (decl1) != FUNCTION_DECL)
{
error ("invalid function declaration");
return false;
}
if (DECL_MAIN_P (decl1))
gcc_assert (same_type_p (TREE_TYPE (TREE_TYPE (decl1)),
integer_type_node));
return start_preparsed_function (decl1, attrs, SF_DEFAULT);
}

static bool
use_eh_spec_block (tree fn)
{
return (flag_exceptions && flag_enforce_eh_specs
&& !processing_template_decl
&& !type_throw_all_p (TREE_TYPE (fn))
&& !DECL_CLONED_FUNCTION_P (fn)
&& !DECL_DEFAULTED_FN (fn));
}
static void
store_parm_decls (tree current_function_parms)
{
tree fndecl = current_function_decl;
tree parm;
tree nonparms = NULL_TREE;
if (current_function_parms)
{
tree specparms = current_function_parms;
tree next;
current_binding_level->names = NULL;
specparms = nreverse (specparms);
for (parm = specparms; parm; parm = next)
{
next = DECL_CHAIN (parm);
if (TREE_CODE (parm) == PARM_DECL)
pushdecl (parm);
else
{
TREE_CHAIN (parm) = NULL_TREE;
nonparms = chainon (nonparms, parm);
}
}
DECL_ARGUMENTS (fndecl) = get_local_decls ();
}
else
DECL_ARGUMENTS (fndecl) = NULL_TREE;
current_binding_level->names = chainon (nonparms, DECL_ARGUMENTS (fndecl));
if (use_eh_spec_block (current_function_decl))
current_eh_spec_block = begin_eh_spec_block ();
}

static void
save_function_data (tree decl)
{
struct language_function *f;
gcc_assert (!DECL_PENDING_INLINE_P (decl));
f = ggc_alloc<language_function> ();
memcpy (f, cp_function_chain, sizeof (struct language_function));
DECL_SAVED_FUNCTION_DATA (decl) = f;
f->base.x_stmt_tree.x_cur_stmt_list = NULL;
f->bindings = NULL;
f->x_local_names = NULL;
f->base.local_typedefs = NULL;
}
static void
finish_constructor_body (void)
{
tree val;
tree exprstmt;
if (targetm.cxx.cdtor_returns_this ())
{
add_stmt (build_stmt (input_location, LABEL_EXPR, cdtor_label));
val = DECL_ARGUMENTS (current_function_decl);
val = build2 (MODIFY_EXPR, TREE_TYPE (val),
DECL_RESULT (current_function_decl), val);
exprstmt = build_stmt (input_location, RETURN_EXPR, val);
add_stmt (exprstmt);
}
}
static void
begin_destructor_body (void)
{
tree compound_stmt;
if (COMPLETE_TYPE_P (current_class_type))
{
compound_stmt = begin_compound_stmt (0);
initialize_vtbl_ptrs (current_class_ptr);
finish_compound_stmt (compound_stmt);
if (flag_lifetime_dse
&& !is_empty_class (current_class_type))
{
if (sanitize_flags_p (SANITIZE_VPTR)
&& (flag_sanitize_recover & SANITIZE_VPTR) == 0
&& TYPE_CONTAINS_VPTR_P (current_class_type))
{
tree binfo = TYPE_BINFO (current_class_type);
tree ref
= cp_build_fold_indirect_ref (current_class_ptr);
tree vtbl_ptr = build_vfield_ref (ref, TREE_TYPE (binfo));
tree vtbl = build_zero_cst (TREE_TYPE (vtbl_ptr));
tree stmt = cp_build_modify_expr (input_location, vtbl_ptr,
NOP_EXPR, vtbl,
tf_warning_or_error);
if (CLASSTYPE_VBASECLASSES (current_class_type)
&& CLASSTYPE_PRIMARY_BINFO (current_class_type)
&& BINFO_VIRTUAL_P
(CLASSTYPE_PRIMARY_BINFO (current_class_type)))
{
stmt = convert_to_void (stmt, ICV_STATEMENT,
tf_warning_or_error);
stmt = build_if_in_charge (stmt);
}
finish_decl_cleanup (NULL_TREE, stmt);
}
else
finish_decl_cleanup (NULL_TREE, build_clobber_this ());
}
push_base_cleanups ();
}
}
static void
finish_destructor_body (void)
{
tree exprstmt;
add_stmt (build_stmt (input_location, LABEL_EXPR, cdtor_label));
if (targetm.cxx.cdtor_returns_this ())
{
tree val;
val = DECL_ARGUMENTS (current_function_decl);
val = build2 (MODIFY_EXPR, TREE_TYPE (val),
DECL_RESULT (current_function_decl), val);
exprstmt = build_stmt (input_location, RETURN_EXPR, val);
add_stmt (exprstmt);
}
}
tree
begin_function_body (void)
{
tree stmt;
if (! FUNCTION_NEEDS_BODY_BLOCK (current_function_decl))
return NULL_TREE;
if (processing_template_decl)
;
else
keep_next_level (true);
stmt = begin_compound_stmt (BCS_FN_BODY);
if (processing_template_decl)
;
else if (DECL_DESTRUCTOR_P (current_function_decl))
begin_destructor_body ();
return stmt;
}
void
finish_function_body (tree compstmt)
{
if (compstmt == NULL_TREE)
return;
finish_compound_stmt (compstmt);
if (processing_template_decl)
;
else if (DECL_CONSTRUCTOR_P (current_function_decl))
finish_constructor_body ();
else if (DECL_DESTRUCTOR_P (current_function_decl))
finish_destructor_body ();
}
tree
outer_curly_brace_block (tree fndecl)
{
tree block = DECL_INITIAL (fndecl);
if (BLOCK_OUTER_CURLY_BRACE_P (block))
return block;
block = BLOCK_SUBBLOCKS (block);
if (BLOCK_OUTER_CURLY_BRACE_P (block))
return block;
block = BLOCK_SUBBLOCKS (block);
gcc_assert (BLOCK_OUTER_CURLY_BRACE_P (block));
return block;
}
static void
record_key_method_defined (tree fndecl)
{
if (DECL_NONSTATIC_MEMBER_FUNCTION_P (fndecl)
&& DECL_VIRTUAL_P (fndecl)
&& !processing_template_decl)
{
tree fnclass = DECL_CONTEXT (fndecl);
if (fndecl == CLASSTYPE_KEY_METHOD (fnclass))
vec_safe_push (keyed_classes, fnclass);
}
}
static void
maybe_save_function_definition (tree fun)
{
if (!processing_template_decl
&& DECL_DECLARED_CONSTEXPR_P (fun)
&& !cp_function_chain->invalid_constexpr
&& !DECL_CLONED_FUNCTION_P (fun))
register_constexpr_fundef (fun, DECL_SAVED_TREE (fun));
}
tree
finish_function (bool inline_p)
{
tree fndecl = current_function_decl;
tree fntype, ctype = NULL_TREE;
if (fndecl == NULL_TREE)
return error_mark_node;
finish_lambda_scope ();
if (c_dialect_objc ())
objc_finish_function ();
record_key_method_defined (fndecl);
fntype = TREE_TYPE (fndecl);
gcc_assert (building_stmt_list_p ());
gcc_assert (DECL_INITIAL (fndecl) == error_mark_node);
if (!DECL_CLONED_FUNCTION_P (fndecl))
{
if (DECL_MAIN_P (current_function_decl))
finish_return_stmt (integer_zero_node);
if (use_eh_spec_block (current_function_decl))
finish_eh_spec_block (TYPE_RAISES_EXCEPTIONS
(TREE_TYPE (current_function_decl)),
current_eh_spec_block);
}
DECL_SAVED_TREE (fndecl) = pop_stmt_list (DECL_SAVED_TREE (fndecl));
finish_fname_decls ();
if (!processing_template_decl
&& !cp_function_chain->can_throw
&& !flag_non_call_exceptions
&& !decl_replaceable_p (fndecl))
TREE_NOTHROW (fndecl) = 1;
if (current_binding_level->kind != sk_function_parms)
{
gcc_assert (errorcount);
DECL_SAVED_TREE (fndecl) = alloc_stmt_list ();
while (current_binding_level->kind != sk_function_parms)
{
if (current_binding_level->kind == sk_class)
pop_nested_class ();
else
poplevel (0, 0, 0);
}
}
poplevel (1, 0, 1);
gcc_assert (stmts_are_full_exprs_p ());
if (!processing_template_decl && FNDECL_USED_AUTO (fndecl)
&& TREE_TYPE (fntype) == current_function_auto_return_pattern)
{
if (is_auto (current_function_auto_return_pattern))
{
apply_deduced_return_type (fndecl, void_type_node);
fntype = TREE_TYPE (fndecl);
}
else if (!current_function_returns_value
&& !current_function_returns_null)
{
error ("no return statements in function returning %qT",
current_function_auto_return_pattern);
inform (input_location, "only plain %<auto%> return type can be "
"deduced to %<void%>");
}
}
if (DECL_DECLARED_CONCEPT_P (fndecl))
check_function_concept (fndecl);
if (cxx_dialect >= cxx17
&& LAMBDA_TYPE_P (CP_DECL_CONTEXT (fndecl)))
DECL_DECLARED_CONSTEXPR_P (fndecl)
= ((processing_template_decl
|| is_valid_constexpr_fn (fndecl, false))
&& potential_constant_expression (DECL_SAVED_TREE (fndecl)));
maybe_save_function_definition (fndecl);
if (!processing_template_decl)
invoke_plugin_callbacks (PLUGIN_PRE_GENERICIZE, fndecl);
if (!processing_template_decl)
cp_fold_function (fndecl);
if (current_function_return_value)
{
tree r = current_function_return_value;
tree outer;
if (r != error_mark_node
&& aggregate_value_p (TREE_TYPE (TREE_TYPE (fndecl)), fndecl)
&& (outer = outer_curly_brace_block (fndecl))
&& chain_member (r, BLOCK_VARS (outer)))
finalize_nrv (&DECL_SAVED_TREE (fndecl), r, DECL_RESULT (fndecl));
current_function_return_value = NULL_TREE;
}
if (current_class_name)
ctype = current_class_type;
DECL_CONTEXT (DECL_RESULT (fndecl)) = fndecl;
BLOCK_SUPERCONTEXT (DECL_INITIAL (fndecl)) = fndecl;
if (!processing_template_decl)
save_function_data (fndecl);
if (warn_return_type
&& !VOID_TYPE_P (TREE_TYPE (fntype))
&& !dependent_type_p (TREE_TYPE (fntype))
&& !current_function_returns_value && !current_function_returns_null
&& !current_function_returns_abnormally
&& !current_function_infinite_loop
&& !TREE_THIS_VOLATILE (fndecl)
&& !DECL_NAME (DECL_RESULT (fndecl))
&& !TREE_NO_WARNING (fndecl)
&& !DECL_CONSTRUCTOR_P (fndecl)
&& !DECL_DESTRUCTOR_P (fndecl)
&& targetm.warn_func_return (fndecl))
{
warning (OPT_Wreturn_type,
"no return statement in function returning non-void");
TREE_NO_WARNING (fndecl) = 1;
}
cfun->function_end_locus = input_location;
if (warn_unused_but_set_parameter
&& !processing_template_decl
&& errorcount == unused_but_set_errorcount
&& !DECL_CLONED_FUNCTION_P (fndecl))
{
tree decl;
for (decl = DECL_ARGUMENTS (fndecl);
decl;
decl = DECL_CHAIN (decl))
if (TREE_USED (decl)
&& TREE_CODE (decl) == PARM_DECL
&& !DECL_READ_P (decl)
&& DECL_NAME (decl)
&& !DECL_ARTIFICIAL (decl)
&& !TREE_NO_WARNING (decl)
&& !DECL_IN_SYSTEM_HEADER (decl)
&& TREE_TYPE (decl) != error_mark_node
&& TREE_CODE (TREE_TYPE (decl)) != REFERENCE_TYPE
&& (!CLASS_TYPE_P (TREE_TYPE (decl))
|| !TYPE_HAS_NONTRIVIAL_DESTRUCTOR (TREE_TYPE (decl))))
warning_at (DECL_SOURCE_LOCATION (decl),
OPT_Wunused_but_set_parameter,
"parameter %qD set but not used", decl);
unused_but_set_errorcount = errorcount;
}
maybe_warn_unused_local_typedefs ();
if (warn_unused_parameter
&& !processing_template_decl 
&& !DECL_CLONED_FUNCTION_P (fndecl))
do_warn_unused_parameter (fndecl);
if (!processing_template_decl)
{
struct language_function *f = DECL_SAVED_FUNCTION_DATA (fndecl);
cp_genericize (fndecl);
f->x_current_class_ptr = NULL;
f->x_current_class_ref = NULL;
f->x_eh_spec_block = NULL;
f->x_in_charge_parm = NULL;
f->x_vtt_parm = NULL;
f->x_return_value = NULL;
f->bindings = NULL;
f->extern_decl_map = NULL;
f->infinite_loops = NULL;
}
local_names = NULL;
set_cfun (NULL);
current_function_decl = NULL;
if (inline_p)
maybe_end_member_template_processing ();
if (ctype)
pop_nested_class ();
--function_depth;
current_function_decl = NULL_TREE;
invoke_plugin_callbacks (PLUGIN_FINISH_PARSE_FUNCTION, fndecl);
return fndecl;
}

tree
grokmethod (cp_decl_specifier_seq *declspecs,
const cp_declarator *declarator, tree attrlist)
{
tree fndecl = grokdeclarator (declarator, declspecs, MEMFUNCDEF, 0,
&attrlist);
if (fndecl == error_mark_node)
return error_mark_node;
if (fndecl == NULL || TREE_CODE (fndecl) != FUNCTION_DECL)
{
error ("invalid member function declaration");
return error_mark_node;
}
if (attrlist)
cplus_decl_attributes (&fndecl, attrlist, 0);
if (fndecl == void_type_node)
return fndecl;
if (DECL_IN_AGGR_P (fndecl))
{
if (DECL_CLASS_SCOPE_P (fndecl))
error ("%qD is already defined in class %qT", fndecl,
DECL_CONTEXT (fndecl));
return error_mark_node;
}
check_template_shadow (fndecl);
if (TREE_PUBLIC (fndecl))
DECL_COMDAT (fndecl) = 1;
DECL_DECLARED_INLINE_P (fndecl) = 1;
DECL_NO_INLINE_WARNING_P (fndecl) = 1;
if (processing_template_decl && !DECL_TEMPLATE_SPECIALIZATION (fndecl))
{
fndecl = push_template_decl (fndecl);
if (fndecl == error_mark_node)
return fndecl;
}
if (! DECL_FRIEND_P (fndecl))
{
if (DECL_CHAIN (fndecl))
{
fndecl = copy_node (fndecl);
TREE_CHAIN (fndecl) = NULL_TREE;
}
}
cp_finish_decl (fndecl, NULL_TREE, false, NULL_TREE, 0);
DECL_IN_AGGR_P (fndecl) = 1;
return fndecl;
}

void
maybe_register_incomplete_var (tree var)
{
gcc_assert (VAR_P (var));
if (!processing_template_decl && TREE_TYPE (var) != error_mark_node
&& DECL_EXTERNAL (var))
{
tree inner_type = TREE_TYPE (var);
while (TREE_CODE (inner_type) == ARRAY_TYPE)
inner_type = TREE_TYPE (inner_type);
inner_type = TYPE_MAIN_VARIANT (inner_type);
if ((!COMPLETE_TYPE_P (inner_type) && CLASS_TYPE_P (inner_type))
|| (TYPE_LANG_SPECIFIC (inner_type)
&& TYPE_BEING_DEFINED (inner_type)))
{
incomplete_var iv = {var, inner_type};
vec_safe_push (incomplete_vars, iv);
}
else if (!(DECL_LANG_SPECIFIC (var) && DECL_TEMPLATE_INFO (var))
&& decl_constant_var_p (var)
&& (TYPE_PTRMEM_P (inner_type) || CLASS_TYPE_P (inner_type)))
{
tree context = outermost_open_class ();
incomplete_var iv = {var, context};
vec_safe_push (incomplete_vars, iv);
}
}
}
void
complete_vars (tree type)
{
unsigned ix;
incomplete_var *iv;
for (ix = 0; vec_safe_iterate (incomplete_vars, ix, &iv); )
{
if (same_type_p (type, iv->incomplete_type))
{
tree var = iv->decl;
tree type = TREE_TYPE (var);
if (type != error_mark_node
&& (TYPE_MAIN_VARIANT (strip_array_types (type))
== iv->incomplete_type))
{
complete_type (type);
cp_apply_type_quals_to_decl (cp_type_quals (type), var);
}
incomplete_vars->unordered_remove (ix);
}
else
ix++;
}
complete_type_check_abstract (type);
}
tree
cxx_maybe_build_cleanup (tree decl, tsubst_flags_t complain)
{
tree type;
tree attr;
tree cleanup;
cleanup = NULL_TREE;
if (error_operand_p (decl))
return cleanup;
if (DECL_P (decl))
attr = lookup_attribute ("cleanup", DECL_ATTRIBUTES (decl));
else
attr = NULL_TREE;
if (attr)
{
tree id;
tree fn;
tree arg;
id = TREE_VALUE (TREE_VALUE (attr));
fn = lookup_name (id);
arg = build_address (decl);
if (!mark_used (decl, complain) && !(complain & tf_error))
return error_mark_node;
cleanup = cp_build_function_call_nary (fn, complain, arg, NULL_TREE);
if (cleanup == error_mark_node)
return error_mark_node;
}
type = TREE_TYPE (decl);
if (type_build_dtor_call (type))
{
int flags = LOOKUP_NORMAL|LOOKUP_NONVIRTUAL|LOOKUP_DESTRUCTOR;
tree addr;
tree call;
if (TREE_CODE (type) == ARRAY_TYPE)
addr = decl;
else
addr = build_address (decl);
call = build_delete (TREE_TYPE (addr), addr,
sfk_complete_destructor, flags, 0, complain);
if (call == error_mark_node)
cleanup = error_mark_node;
else if (TYPE_HAS_TRIVIAL_DESTRUCTOR (type))
;
else if (cleanup)
cleanup = cp_build_compound_expr (cleanup, call, complain);
else
cleanup = call;
}
protected_set_expr_location (cleanup, UNKNOWN_LOCATION);
if (cleanup
&& DECL_P (decl)
&& !lookup_attribute ("warn_unused", TYPE_ATTRIBUTES (TREE_TYPE (decl)))
&& !mark_used (decl, complain) && !(complain & tf_error))
return error_mark_node;
return cleanup;
}

tree
static_fn_type (tree memfntype)
{
tree fntype;
tree args;
if (TYPE_PTRMEMFUNC_P (memfntype))
memfntype = TYPE_PTRMEMFUNC_FN_TYPE (memfntype);
if (POINTER_TYPE_P (memfntype)
|| TREE_CODE (memfntype) == FUNCTION_DECL)
memfntype = TREE_TYPE (memfntype);
if (TREE_CODE (memfntype) == FUNCTION_TYPE)
return memfntype;
gcc_assert (TREE_CODE (memfntype) == METHOD_TYPE);
args = TYPE_ARG_TYPES (memfntype);
cp_ref_qualifier rqual = type_memfn_rqual (memfntype);
fntype = build_function_type (TREE_TYPE (memfntype), TREE_CHAIN (args));
fntype = apply_memfn_quals (fntype, type_memfn_quals (memfntype), rqual);
fntype = (cp_build_type_attribute_variant
(fntype, TYPE_ATTRIBUTES (memfntype)));
fntype = (build_exception_variant
(fntype, TYPE_RAISES_EXCEPTIONS (memfntype)));
if (TYPE_HAS_LATE_RETURN_TYPE (memfntype))
TYPE_HAS_LATE_RETURN_TYPE (fntype) = 1;
return fntype;
}
void
revert_static_member_fn (tree decl)
{
tree stype = static_fn_type (decl);
cp_cv_quals quals = type_memfn_quals (stype);
cp_ref_qualifier rqual = type_memfn_rqual (stype);
if (quals != TYPE_UNQUALIFIED || rqual != REF_QUAL_NONE)
stype = apply_memfn_quals (stype, TYPE_UNQUALIFIED, REF_QUAL_NONE);
TREE_TYPE (decl) = stype;
if (DECL_ARGUMENTS (decl))
DECL_ARGUMENTS (decl) = DECL_CHAIN (DECL_ARGUMENTS (decl));
DECL_STATIC_FUNCTION_P (decl) = 1;
}
enum cp_tree_node_structure_enum
cp_tree_node_structure (union lang_tree_node * t)
{
switch (TREE_CODE (&t->generic))
{
case DEFAULT_ARG:		return TS_CP_DEFAULT_ARG;
case DEFERRED_NOEXCEPT:	return TS_CP_DEFERRED_NOEXCEPT;
case IDENTIFIER_NODE:	return TS_CP_IDENTIFIER;
case OVERLOAD:		return TS_CP_OVERLOAD;
case TEMPLATE_PARM_INDEX:	return TS_CP_TPI;
case PTRMEM_CST:		return TS_CP_PTRMEM;
case BASELINK:		return TS_CP_BASELINK;
case TEMPLATE_DECL:		return TS_CP_TEMPLATE_DECL;
case STATIC_ASSERT:		return TS_CP_STATIC_ASSERT;
case ARGUMENT_PACK_SELECT:  return TS_CP_ARGUMENT_PACK_SELECT;
case TRAIT_EXPR:		return TS_CP_TRAIT_EXPR;
case LAMBDA_EXPR:		return TS_CP_LAMBDA_EXPR;
case TEMPLATE_INFO:		return TS_CP_TEMPLATE_INFO;
case CONSTRAINT_INFO:       return TS_CP_CONSTRAINT_INFO;
case USERDEF_LITERAL:	return TS_CP_USERDEF_LITERAL;
default:			return TS_CP_GENERIC;
}
}
tree
build_void_list_node (void)
{
tree t = build_tree_list (NULL_TREE, void_type_node);
return t;
}
bool
cp_missing_noreturn_ok_p (tree decl)
{
return DECL_MAIN_P (decl);
}
tree
cxx_comdat_group (tree decl)
{
if (VAR_P (decl) && DECL_VTABLE_OR_VTT_P (decl))
decl = CLASSTYPE_VTABLES (DECL_CONTEXT (decl));
else
{
while (DECL_THUNK_P (decl))
{
tree target = THUNK_TARGET (decl);
if (TARGET_USE_LOCAL_THUNK_ALIAS_P (target)
&& DECL_SECTION_NAME (target) != NULL
&& DECL_ONE_ONLY (target))
decl = target;
else
break;
}
}
return decl;
}
tree
fndecl_declared_return_type (tree fn)
{
fn = STRIP_TEMPLATE (fn);
if (FNDECL_USED_AUTO (fn))
{
struct language_function *f = NULL;
if (DECL_STRUCT_FUNCTION (fn))
f = DECL_STRUCT_FUNCTION (fn)->language;
if (f == NULL)
f = DECL_SAVED_FUNCTION_DATA (fn);
return f->x_auto_return_pattern;
}
return TREE_TYPE (TREE_TYPE (fn));
}
bool
undeduced_auto_decl (tree decl)
{
if (cxx_dialect < cxx11)
return false;
return ((VAR_OR_FUNCTION_DECL_P (decl)
|| TREE_CODE (decl) == TEMPLATE_DECL)
&& type_uses_auto (TREE_TYPE (decl)));
}
bool
require_deduced_type (tree decl, tsubst_flags_t complain)
{
if (undeduced_auto_decl (decl))
{
if (complain & tf_error)
error ("use of %qD before deduction of %<auto%>", decl);
return false;
}
return true;
}
#include "gt-cp-decl.h"
