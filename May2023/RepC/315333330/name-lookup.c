#include "config.h"
#define INCLUDE_UNIQUE_PTR
#include "system.h"
#include "coretypes.h"
#include "cp-tree.h"
#include "timevar.h"
#include "stringpool.h"
#include "print-tree.h"
#include "attribs.h"
#include "debug.h"
#include "c-family/c-pragma.h"
#include "params.h"
#include "gcc-rich-location.h"
#include "spellcheck-tree.h"
#include "parser.h"
#include "c-family/name-hint.h"
#include "c-family/known-headers.h"
#include "c-family/c-spellcheck.h"
static cxx_binding *cxx_binding_make (tree value, tree type);
static cp_binding_level *innermost_nonclass_level (void);
static void set_identifier_type_value_with_scope (tree id, tree decl,
cp_binding_level *b);
static bool maybe_suggest_missing_std_header (location_t location, tree name);
#define STAT_HACK_P(N) ((N) && TREE_CODE (N) == OVERLOAD && OVL_LOOKUP_P (N))
#define STAT_TYPE(N) TREE_TYPE (N)
#define STAT_DECL(N) OVL_FUNCTION (N)
#define MAYBE_STAT_DECL(N) (STAT_HACK_P (N) ? STAT_DECL (N) : N)
#define MAYBE_STAT_TYPE(N) (STAT_HACK_P (N) ? STAT_TYPE (N) : NULL_TREE)
static tree
stat_hack (tree decl = NULL_TREE, tree type = NULL_TREE)
{
tree result = make_node (OVERLOAD);
OVL_LOOKUP_P (result) = true;
STAT_DECL (result) = decl;
STAT_TYPE (result) = type;
return result;
}
static cxx_binding *
create_local_binding (cp_binding_level *level, tree name)
{
cxx_binding *binding = cxx_binding_make (NULL, NULL);
INHERITED_VALUE_BINDING_P (binding) = false;
LOCAL_BINDING_P (binding) = true;
binding->scope = level;
binding->previous = IDENTIFIER_BINDING (name);
IDENTIFIER_BINDING (name) = binding;
return binding;
}
static tree *
find_namespace_slot (tree ns, tree name, bool create_p = false)
{
tree *slot = DECL_NAMESPACE_BINDINGS (ns)
->find_slot_with_hash (name, name ? IDENTIFIER_HASH_VALUE (name) : 0,
create_p ? INSERT : NO_INSERT);
return slot;
}
static tree
find_namespace_value (tree ns, tree name)
{
tree *b = find_namespace_slot (ns, name);
return b ? MAYBE_STAT_DECL (*b) : NULL_TREE;
}
static void
add_decl_to_level (cp_binding_level *b, tree decl)
{
gcc_assert (b->kind != sk_class);
gcc_assert (b->names != decl);
TREE_CHAIN (decl) = b->names;
b->names = decl;
if (b->kind == sk_namespace
&& ((VAR_P (decl)
&& (TREE_STATIC (decl) || DECL_EXTERNAL (decl)))
|| (TREE_CODE (decl) == FUNCTION_DECL
&& (!TREE_PUBLIC (decl)
|| decl_anon_ns_mem_p (decl)
|| DECL_DECLARED_INLINE_P (decl)))))
vec_safe_push (static_decls, decl);
}
static cxx_binding *
find_local_binding (cp_binding_level *b, tree name)
{
if (cxx_binding *binding = IDENTIFIER_BINDING (name))
for (;; b = b->level_chain)
{
if (binding->scope == b
&& !(VAR_P (binding->value)
&& DECL_DEAD_FOR_LOCAL (binding->value)))
return binding;
if (b->kind != sk_cleanup)
break;
}
return NULL;
}
struct name_lookup
{
public:
typedef std::pair<tree, tree> using_pair;
typedef vec<using_pair, va_heap, vl_embed> using_queue;
public:
tree name;	
tree value;	
tree type;	
int flags;	
bool deduping; 
vec<tree, va_heap, vl_embed> *scopes;
name_lookup *previous; 
protected:
static vec<tree, va_heap, vl_embed> *shared_scopes;
static name_lookup *active;
public:
name_lookup (tree n, int f = 0)
: name (n), value (NULL_TREE), type (NULL_TREE), flags (f),
deduping (false), scopes (NULL), previous (NULL)
{
preserve_state ();
}
~name_lookup ()
{
restore_state ();
}
private: 
name_lookup (const name_lookup &);
name_lookup &operator= (const name_lookup &);
protected:
static bool seen_p (tree scope)
{
return LOOKUP_SEEN_P (scope);
}
static bool found_p (tree scope)
{
return LOOKUP_FOUND_P (scope);
}
void mark_seen (tree scope); 
static void mark_found (tree scope)
{
gcc_checking_assert (seen_p (scope));
LOOKUP_FOUND_P (scope) = true;
}
bool see_and_mark (tree scope)
{
bool ret = seen_p (scope);
if (!ret)
mark_seen (scope);
return ret;
}
bool find_and_mark (tree scope);
private:
void preserve_state ();
void restore_state ();
private:
static tree ambiguous (tree thing, tree current);
void add_overload (tree fns);
void add_value (tree new_val);
void add_type (tree new_type);
bool process_binding (tree val_bind, tree type_bind);
bool search_namespace_only (tree scope);
bool search_namespace (tree scope);
bool search_usings (tree scope);
private:
using_queue *queue_namespace (using_queue *queue, int depth, tree scope);
using_queue *do_queue_usings (using_queue *queue, int depth,
vec<tree, va_gc> *usings);
using_queue *queue_usings (using_queue *queue, int depth,
vec<tree, va_gc> *usings)
{
if (usings)
queue = do_queue_usings (queue, depth, usings);
return queue;
}
private:
void add_fns (tree);
void adl_expr (tree);
void adl_type (tree);
void adl_template_arg (tree);
void adl_class (tree);
void adl_bases (tree);
void adl_class_only (tree);
void adl_namespace (tree);
void adl_namespace_only (tree);
public:
bool search_qualified (tree scope, bool usings = true);
bool search_unqualified (tree scope, cp_binding_level *);
tree search_adl (tree fns, vec<tree, va_gc> *args);
};
vec<tree, va_heap, vl_embed> *name_lookup::shared_scopes;
name_lookup *name_lookup::active;
void
name_lookup::preserve_state ()
{
previous = active;
if (previous)
{
unsigned length = vec_safe_length (previous->scopes);
vec_safe_reserve (previous->scopes, length * 2);
for (unsigned ix = length; ix--;)
{
tree decl = (*previous->scopes)[ix];
gcc_checking_assert (LOOKUP_SEEN_P (decl));
LOOKUP_SEEN_P (decl) = false;
if (LOOKUP_FOUND_P (decl))
{
LOOKUP_FOUND_P (decl) = false;
previous->scopes->quick_push (decl);
}
}
if (previous->deduping)
lookup_mark (previous->value, false);
}
else
scopes = shared_scopes;
active = this;
}
void
name_lookup::restore_state ()
{
if (deduping)
lookup_mark (value, false);
for (unsigned ix = vec_safe_length (scopes); ix--;)
{
tree decl = scopes->pop ();
gcc_checking_assert (LOOKUP_SEEN_P (decl));
LOOKUP_SEEN_P (decl) = false;
LOOKUP_FOUND_P (decl) = false;
}
active = previous;
if (previous)
{
free (scopes);
unsigned length = vec_safe_length (previous->scopes);
for (unsigned ix = 0; ix != length; ix++)
{
tree decl = (*previous->scopes)[ix];
if (LOOKUP_SEEN_P (decl))
{
do
{
tree decl = previous->scopes->pop ();
gcc_checking_assert (LOOKUP_SEEN_P (decl)
&& !LOOKUP_FOUND_P (decl));
LOOKUP_FOUND_P (decl) = true;
}
while (++ix != length);
break;
}
gcc_checking_assert (!LOOKUP_FOUND_P (decl));
LOOKUP_SEEN_P (decl) = true;
}
if (previous->deduping)
lookup_mark (previous->value, true);
}
else
shared_scopes = scopes;
}
void
name_lookup::mark_seen (tree scope)
{
gcc_checking_assert (!seen_p (scope));
LOOKUP_SEEN_P (scope) = true;
vec_safe_push (scopes, scope);
}
bool
name_lookup::find_and_mark (tree scope)
{
bool result = LOOKUP_FOUND_P (scope);
if (!result)
{
LOOKUP_FOUND_P (scope) = true;
if (!LOOKUP_SEEN_P (scope))
vec_safe_push (scopes, scope);
}
return result;
}
tree
name_lookup::ambiguous (tree thing, tree current)
{
if (TREE_CODE (current) != TREE_LIST)
{
current = build_tree_list (NULL_TREE, current);
TREE_TYPE (current) = error_mark_node;
}
current = tree_cons (NULL_TREE, thing, current);
TREE_TYPE (current) = error_mark_node;
return current;
}
void
name_lookup::add_overload (tree fns)
{
if (!deduping && TREE_CODE (fns) == OVERLOAD)
{
tree probe = fns;
if (flags & LOOKUP_HIDDEN)
probe = ovl_skip_hidden (probe);
if (probe && TREE_CODE (probe) == OVERLOAD && OVL_USING_P (probe))
{
lookup_mark (value, true);
deduping = true;
}
}
value = lookup_maybe_add (fns, value, deduping);
}
void
name_lookup::add_value (tree new_val)
{
if (OVL_P (new_val) && (!value || OVL_P (value)))
add_overload (new_val);
else if (!value)
value = new_val;
else if (value == new_val)
;
else if ((TREE_CODE (value) == TYPE_DECL
&& TREE_CODE (new_val) == TYPE_DECL
&& same_type_p (TREE_TYPE (value), TREE_TYPE (new_val))))
;
else if (TREE_CODE (value) == NAMESPACE_DECL
&& TREE_CODE (new_val) == NAMESPACE_DECL
&& ORIGINAL_NAMESPACE (value) == ORIGINAL_NAMESPACE (new_val))
value = ORIGINAL_NAMESPACE (value);
else
{
if (deduping)
{
lookup_mark (value, false);
deduping = false;
}
value = ambiguous (new_val, value);
}
}
void
name_lookup::add_type (tree new_type)
{
if (!type)
type = new_type;
else if (TREE_CODE (type) == TREE_LIST
|| !same_type_p (TREE_TYPE (type), TREE_TYPE (new_type)))
type = ambiguous (new_type, type);
}
bool
name_lookup::process_binding (tree new_val, tree new_type)
{
if (new_type
&& (LOOKUP_NAMESPACES_ONLY (flags)
|| (!(flags & LOOKUP_HIDDEN)
&& DECL_LANG_SPECIFIC (new_type)
&& DECL_ANTICIPATED (new_type))))
new_type = NULL_TREE;
if (new_val && !(flags & LOOKUP_HIDDEN))
new_val = ovl_skip_hidden (new_val);
if (new_val)
switch (TREE_CODE (new_val))
{
case TEMPLATE_DECL:
if ((LOOKUP_QUALIFIERS_ONLY (flags)
&& !DECL_TYPE_TEMPLATE_P (new_val)))
new_val = NULL_TREE;
break;
case TYPE_DECL:
if (LOOKUP_NAMESPACES_ONLY (flags)
|| (new_type && (flags & LOOKUP_PREFER_TYPES)))
new_val = NULL_TREE;
break;
case NAMESPACE_DECL:
if (LOOKUP_TYPES_ONLY (flags))
new_val = NULL_TREE;
break;
default:
if (LOOKUP_QUALIFIERS_ONLY (flags))
new_val = NULL_TREE;
}
if (!new_val)
{
new_val = new_type;
new_type = NULL_TREE;
}
if (new_val)
add_value (new_val);
if (new_type)
add_type (new_type);
return new_val != NULL_TREE;
}
bool
name_lookup::search_namespace_only (tree scope)
{
bool found = false;
if (tree *binding = find_namespace_slot (scope, name))
found |= process_binding (MAYBE_STAT_DECL (*binding),
MAYBE_STAT_TYPE (*binding));
return found;
}
bool
name_lookup::search_namespace (tree scope)
{
if (see_and_mark (scope))
return found_p (scope);
bool found = search_namespace_only (scope);
if (name)
if (vec<tree, va_gc> *inlinees = DECL_NAMESPACE_INLINEES (scope))
for (unsigned ix = inlinees->length (); ix--;)
found |= search_namespace ((*inlinees)[ix]);
if (found)
mark_found (scope);
return found;
}
bool
name_lookup::search_usings (tree scope)
{
if (found_p (scope))
return true;
bool found = false;
if (vec<tree, va_gc> *usings = DECL_NAMESPACE_USING (scope))
for (unsigned ix = usings->length (); ix--;)
found |= search_qualified ((*usings)[ix], true);
if (vec<tree, va_gc> *inlinees = DECL_NAMESPACE_INLINEES (scope))
for (unsigned ix = inlinees->length (); ix--;)
found |= search_usings ((*inlinees)[ix]);
if (found)
mark_found (scope);
return found;
}
bool
name_lookup::search_qualified (tree scope, bool usings)
{
bool found = false;
if (seen_p (scope))
found = found_p (scope);
else 
{
found = search_namespace (scope);
if (!found && usings)
found = search_usings (scope);
}
return found;
}
name_lookup::using_queue *
name_lookup::queue_namespace (using_queue *queue, int depth, tree scope)
{
if (see_and_mark (scope))
return queue;
tree common = scope;
while (SCOPE_DEPTH (common) > depth)
common = CP_DECL_CONTEXT (common);
vec_safe_push (queue, using_pair (common, scope));
if (vec<tree, va_gc> *inlinees = DECL_NAMESPACE_INLINEES (scope))
for (unsigned ix = inlinees->length (); ix--;)
queue = queue_namespace (queue, depth, (*inlinees)[ix]);
queue = queue_usings (queue, depth, DECL_NAMESPACE_USING (scope));
return queue;
}
name_lookup::using_queue *
name_lookup::do_queue_usings (using_queue *queue, int depth,
vec<tree, va_gc> *usings)
{
for (unsigned ix = usings->length (); ix--;)
queue = queue_namespace (queue, depth, (*usings)[ix]);
return queue;
}
bool
name_lookup::search_unqualified (tree scope, cp_binding_level *level)
{
static using_queue *queue = NULL;
bool found = false;
int length = vec_safe_length (queue);
for (; level->kind != sk_namespace; level = level->level_chain)
queue = queue_usings (queue, SCOPE_DEPTH (scope), level->using_directives);
for (; !found; scope = CP_DECL_CONTEXT (scope))
{
gcc_assert (!DECL_NAMESPACE_ALIAS (scope));
int depth = SCOPE_DEPTH (scope);
queue = queue_namespace (queue, depth, scope);
unsigned ix = length;
do
{
using_pair &pair = (*queue)[ix];
while (pair.first == scope)
{
found |= search_namespace_only (pair.second);
pair = queue->pop ();
if (ix == queue->length ())
goto done;
}
if (SCOPE_DEPTH (pair.first) == depth)
pair.first = CP_DECL_CONTEXT (pair.first);
ix++;
}
while (ix < queue->length ());
done:;
if (scope == global_namespace)
break;
if (flags & LOOKUP_HIDDEN)
break;
}
vec_safe_truncate (queue, length);
return found;
}
void
name_lookup::add_fns (tree fns)
{
if (!fns)
return;
else if (TREE_CODE (fns) == OVERLOAD)
{
if (TREE_TYPE (fns) != unknown_type_node)
fns = OVL_FUNCTION (fns);
}
else if (!DECL_DECLARES_FUNCTION_P (fns))
return;
add_overload (fns);
}
void
name_lookup::adl_namespace_only (tree scope)
{
mark_seen (scope);
if (vec<tree, va_gc> *inlinees = DECL_NAMESPACE_INLINEES (scope))
for (unsigned ix = inlinees->length (); ix--;)
adl_namespace_only ((*inlinees)[ix]);
if (tree fns = find_namespace_value (scope, name))
add_fns (ovl_skip_hidden (fns));
}
void
name_lookup::adl_namespace (tree scope)
{
if (seen_p (scope))
return;
while (DECL_NAMESPACE_INLINE_P (scope))
scope = CP_DECL_CONTEXT (scope);
adl_namespace_only (scope);
}
void
name_lookup::adl_class_only (tree type)
{
if (!CLASS_TYPE_P (type))
return;
type = TYPE_MAIN_VARIANT (type);
if (see_and_mark (type))
return;
tree context = decl_namespace_context (type);
adl_namespace (context);
complete_type (type);
for (tree list = DECL_FRIENDLIST (TYPE_MAIN_DECL (type)); list;
list = TREE_CHAIN (list))
if (name == FRIEND_NAME (list))
for (tree friends = FRIEND_DECLS (list); friends;
friends = TREE_CHAIN (friends))
{
tree fn = TREE_VALUE (friends);
if (CP_DECL_CONTEXT (fn) != context)
continue;
if (!DECL_ANTICIPATED (fn))
continue;
if (TREE_CODE (fn) == FUNCTION_DECL && DECL_USE_TEMPLATE (fn))
continue;
add_fns (fn);
}
}
void
name_lookup::adl_bases (tree type)
{
adl_class_only (type);
if (tree binfo = TYPE_BINFO (type))
{
tree base_binfo;
int i;
for (i = 0; BINFO_BASE_ITERATE (binfo, i, base_binfo); i++)
adl_bases (BINFO_TYPE (base_binfo));
}
}
void
name_lookup::adl_class (tree type)
{
if (!CLASS_TYPE_P (type))
return;
type = TYPE_MAIN_VARIANT (type);
if (found_p (type))
return;
adl_bases (type);
mark_found (type);
if (TYPE_CLASS_SCOPE_P (type))
adl_class_only (TYPE_CONTEXT (type));
if (CLASSTYPE_TEMPLATE_INFO (type)
&& PRIMARY_TEMPLATE_P (CLASSTYPE_TI_TEMPLATE (type)))
{
tree list = INNERMOST_TEMPLATE_ARGS (CLASSTYPE_TI_ARGS (type));
for (int i = 0; i < TREE_VEC_LENGTH (list); ++i)
adl_template_arg (TREE_VEC_ELT (list, i));
}
}
void
name_lookup::adl_expr (tree expr)
{
if (!expr)
return;
gcc_assert (!TYPE_P (expr));
if (TREE_TYPE (expr) != unknown_type_node)
{
adl_type (TREE_TYPE (expr));
return;
}
if (TREE_CODE (expr) == ADDR_EXPR)
expr = TREE_OPERAND (expr, 0);
if (TREE_CODE (expr) == COMPONENT_REF
|| TREE_CODE (expr) == OFFSET_REF)
expr = TREE_OPERAND (expr, 1);
expr = MAYBE_BASELINK_FUNCTIONS (expr);
if (OVL_P (expr))
for (lkp_iterator iter (expr); iter; ++iter)
adl_type (TREE_TYPE (*iter));
else if (TREE_CODE (expr) == TEMPLATE_ID_EXPR)
{
adl_expr (TREE_OPERAND (expr, 0));
if (tree args = TREE_OPERAND (expr, 1))
for (int ix = TREE_VEC_LENGTH (args); ix--;)
adl_template_arg (TREE_VEC_ELT (args, ix));
}
}
void
name_lookup::adl_type (tree type)
{
if (!type)
return;
if (TYPE_PTRDATAMEM_P (type))
{
adl_type (TYPE_PTRMEM_CLASS_TYPE (type));
adl_type (TYPE_PTRMEM_POINTED_TO_TYPE (type));
return;
}
switch (TREE_CODE (type))
{
case RECORD_TYPE:
if (TYPE_PTRMEMFUNC_P (type))
{
adl_type (TYPE_PTRMEMFUNC_FN_TYPE (type));
return;
}
case UNION_TYPE:
adl_class (type);
return;
case METHOD_TYPE:
case FUNCTION_TYPE:
for (tree args = TYPE_ARG_TYPES (type); args; args = TREE_CHAIN (args))
adl_type (TREE_VALUE (args));
case POINTER_TYPE:
case REFERENCE_TYPE:
case ARRAY_TYPE:
adl_type (TREE_TYPE (type));
return;
case ENUMERAL_TYPE:
if (TYPE_CLASS_SCOPE_P (type))
adl_class_only (TYPE_CONTEXT (type));
adl_namespace (decl_namespace_context (type));
return;
case LANG_TYPE:
gcc_assert (type == unknown_type_node
|| type == init_list_type_node);
return;
case TYPE_PACK_EXPANSION:
adl_type (PACK_EXPANSION_PATTERN (type));
return;
default:
break;
}
}
void
name_lookup::adl_template_arg (tree arg)
{
if (TREE_CODE (arg) == TEMPLATE_TEMPLATE_PARM
|| TREE_CODE (arg) == UNBOUND_CLASS_TEMPLATE)
;
else if (TREE_CODE (arg) == TEMPLATE_DECL)
{
tree ctx = CP_DECL_CONTEXT (arg);
if (TREE_CODE (ctx) == NAMESPACE_DECL)
adl_namespace (ctx);
else
adl_class_only (ctx);
}
else if (ARGUMENT_PACK_P (arg))
{
tree args = ARGUMENT_PACK_ARGS (arg);
int i, len = TREE_VEC_LENGTH (args);
for (i = 0; i < len; ++i) 
adl_template_arg (TREE_VEC_ELT (args, i));
}
else if (TYPE_P (arg))
adl_type (arg);
}
tree
name_lookup::search_adl (tree fns, vec<tree, va_gc> *args)
{
if (fns)
{
deduping = true;
lookup_mark (fns, true);
}
value = fns;
unsigned ix;
tree arg;
FOR_EACH_VEC_ELT_REVERSE (*args, ix, arg)
if (TYPE_P (arg))
adl_type (arg);
else
adl_expr (arg);
fns = value;
return fns;
}
static bool qualified_namespace_lookup (tree, name_lookup *);
static void consider_binding_level (tree name,
best_match <tree, const char *> &bm,
cp_binding_level *lvl,
bool look_within_fields,
enum lookup_name_fuzzy_kind kind);
static void diagnose_name_conflict (tree, tree);
tree
lookup_arg_dependent (tree name, tree fns, vec<tree, va_gc> *args)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
name_lookup lookup (name);
fns = lookup.search_adl (fns, args);
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return fns;
}
static tree
extract_conversion_operator (tree fns, tree type)
{
tree convs = NULL_TREE;
tree tpls = NULL_TREE;
for (ovl_iterator iter (fns); iter; ++iter)
{
if (same_type_p (DECL_CONV_FN_TYPE (*iter), type))
convs = lookup_add (*iter, convs);
if (TREE_CODE (*iter) == TEMPLATE_DECL)
tpls = lookup_add (*iter, tpls);
}
if (!convs)
convs = tpls;
return convs;
}
static tree
member_vec_binary_search (vec<tree, va_gc> *member_vec, tree name)
{
for (unsigned lo = 0, hi = member_vec->length (); lo < hi;)
{
unsigned mid = (lo + hi) / 2;
tree binding = (*member_vec)[mid];
tree binding_name = OVL_NAME (binding);
if (binding_name > name)
hi = mid;
else if (binding_name < name)
lo = mid + 1;
else
return binding;
}
return NULL_TREE;
}
static tree
member_vec_linear_search (vec<tree, va_gc> *member_vec, tree name)
{
for (int ix = member_vec->length (); ix--;)
if (tree binding = (*member_vec)[ix])
if (OVL_NAME (binding) == name)
return binding;
return NULL_TREE;
}
static tree
fields_linear_search (tree klass, tree name, bool want_type)
{
for (tree fields = TYPE_FIELDS (klass); fields; fields = DECL_CHAIN (fields))
{
tree decl = fields;
if (TREE_CODE (decl) == FIELD_DECL
&& ANON_AGGR_TYPE_P (TREE_TYPE (decl)))
{
if (tree temp = search_anon_aggr (TREE_TYPE (decl), name, want_type))
return temp;
}
if (DECL_NAME (decl) != name)
continue;
if (TREE_CODE (decl) == USING_DECL)
{
decl = strip_using_decl (decl);
if (is_overloaded_fn (decl))
continue;
}
if (DECL_DECLARES_FUNCTION_P (decl))
continue;
if (!want_type || DECL_DECLARES_TYPE_P (decl))
return decl;
}
return NULL_TREE;
}
tree
search_anon_aggr (tree anon, tree name, bool want_type)
{
gcc_assert (COMPLETE_TYPE_P (anon));
tree ret = get_class_binding_direct (anon, name, want_type);
return ret;
}
tree
get_class_binding_direct (tree klass, tree name, int type_or_fns)
{
gcc_checking_assert (RECORD_OR_UNION_TYPE_P (klass));
bool conv_op = IDENTIFIER_CONV_OP_P (name);
tree lookup = conv_op ? conv_op_identifier : name;
tree val = NULL_TREE;
vec<tree, va_gc> *member_vec = CLASSTYPE_MEMBER_VEC (klass);
if (COMPLETE_TYPE_P (klass) && member_vec)
{
val = member_vec_binary_search (member_vec, lookup);
if (!val)
;
else if (type_or_fns > 0)
{
if (STAT_HACK_P (val))
val = STAT_TYPE (val);
else if (!DECL_DECLARES_TYPE_P (val))
val = NULL_TREE;
}
else if (STAT_HACK_P (val))
val = STAT_DECL (val);
}
else
{
if (member_vec && type_or_fns <= 0)
val = member_vec_linear_search (member_vec, lookup);
if (type_or_fns < 0)
;
else if (!val || (TREE_CODE (val) == OVERLOAD && OVL_USING_P (val)))
if (tree field_val = fields_linear_search (klass, lookup,
type_or_fns > 0))
if (!val || TREE_CODE (field_val) == USING_DECL)
val = field_val;
}
if (val && conv_op)
{
gcc_checking_assert (OVL_FUNCTION (val) == conv_op_marker);
val = OVL_CHAIN (val);
if (tree type = TREE_TYPE (name))
val = extract_conversion_operator (val, type);
}
return val;
}
tree
get_class_binding (tree klass, tree name, int type_or_fns)
{
klass = complete_type (klass);
if (COMPLETE_TYPE_P (klass))
{
if (IDENTIFIER_CTOR_P (name))
{
if (CLASSTYPE_LAZY_DEFAULT_CTOR (klass))
lazily_declare_fn (sfk_constructor, klass);
if (CLASSTYPE_LAZY_COPY_CTOR (klass))
lazily_declare_fn (sfk_copy_constructor, klass);
if (CLASSTYPE_LAZY_MOVE_CTOR (klass))
lazily_declare_fn (sfk_move_constructor, klass);
}
else if (IDENTIFIER_DTOR_P (name))
{
if (CLASSTYPE_LAZY_DESTRUCTOR (klass))
lazily_declare_fn (sfk_destructor, klass);
}
else if (name == assign_op_identifier)
{
if (CLASSTYPE_LAZY_COPY_ASSIGN (klass))
lazily_declare_fn (sfk_copy_assignment, klass);
if (CLASSTYPE_LAZY_MOVE_ASSIGN (klass))
lazily_declare_fn (sfk_move_assignment, klass);
}
}
return get_class_binding_direct (klass, name, type_or_fns);
}
tree *
find_member_slot (tree klass, tree name)
{
bool complete_p = COMPLETE_TYPE_P (klass);
vec<tree, va_gc> *member_vec = CLASSTYPE_MEMBER_VEC (klass);
if (!member_vec)
{
vec_alloc (member_vec, 8);
CLASSTYPE_MEMBER_VEC (klass) = member_vec;
if (complete_p)
{
set_class_bindings (klass, 6);
member_vec = CLASSTYPE_MEMBER_VEC (klass);
}
}
if (IDENTIFIER_CONV_OP_P (name))
name = conv_op_identifier;
unsigned ix, length = member_vec->length ();
for (ix = 0; ix < length; ix++)
{
tree *slot = &(*member_vec)[ix];
tree fn_name = OVL_NAME (*slot);
if (fn_name == name)
{
gcc_checking_assert (OVL_P (*slot));
if (name == conv_op_identifier)
{
gcc_checking_assert (OVL_FUNCTION (*slot) == conv_op_marker);
slot = &OVL_CHAIN (*slot);
}
return slot;
}
if (complete_p && fn_name > name)
break;
}
if (complete_p)
{
gcc_assert (name != conv_op_identifier);
vec_safe_reserve_exact (member_vec, 1);
CLASSTYPE_MEMBER_VEC (klass) = member_vec;
member_vec->quick_insert (ix, NULL_TREE);
return &(*member_vec)[ix];
}
return NULL;
}
tree *
add_member_slot (tree klass, tree name)
{
gcc_assert (!COMPLETE_TYPE_P (klass));
vec<tree, va_gc> *member_vec = CLASSTYPE_MEMBER_VEC (klass);
vec_safe_push (member_vec, NULL_TREE);
CLASSTYPE_MEMBER_VEC (klass) = member_vec;
tree *slot = &member_vec->last ();
if (IDENTIFIER_CONV_OP_P (name))
{
*slot = ovl_make (conv_op_marker, NULL_TREE);
slot = &OVL_CHAIN (*slot);
}
return slot;
}
static int
member_name_cmp (const void *a_p, const void *b_p)
{
tree a = *(const tree *)a_p;
tree b = *(const tree *)b_p;
tree name_a = DECL_NAME (TREE_CODE (a) == OVERLOAD ? OVL_FUNCTION (a) : a);
tree name_b = DECL_NAME (TREE_CODE (b) == OVERLOAD ? OVL_FUNCTION (b) : b);
gcc_checking_assert (name_a && name_b);
if (name_a != name_b)
return name_a < name_b ? -1 : +1;
if (name_a == conv_op_identifier)
{
gcc_checking_assert (OVL_FUNCTION (a) == conv_op_marker
&& OVL_FUNCTION (b) == conv_op_marker);
a = OVL_CHAIN (a);
b = OVL_CHAIN (b);
}
if (TREE_CODE (a) == OVERLOAD)
a = OVL_FUNCTION (a);
if (TREE_CODE (b) == OVERLOAD)
b = OVL_FUNCTION (b);
if (TREE_CODE (a) != TREE_CODE (b))
{
if (TREE_CODE (a) == TYPE_DECL)
return +1;
else if (TREE_CODE (b) == TYPE_DECL)
return -1;
if (TREE_CODE (a) == USING_DECL)
return +1;
else if (TREE_CODE (b) == USING_DECL)
return -1;
gcc_assert (errorcount);
}
if (DECL_UID (a) != DECL_UID (b))
return DECL_UID (a) < DECL_UID (b) ? -1 : +1;
gcc_assert (a == b);
return 0;
}
static struct {
gt_pointer_operator new_value;
void *cookie;
} resort_data;
static int
resort_member_name_cmp (const void *a_p, const void *b_p)
{
tree a = *(const tree *)a_p;
tree b = *(const tree *)b_p;
tree name_a = OVL_NAME (a);
tree name_b = OVL_NAME (b);
resort_data.new_value (&name_a, resort_data.cookie);
resort_data.new_value (&name_b, resort_data.cookie);
gcc_checking_assert (name_a != name_b);
return name_a < name_b ? -1 : +1;
}
void
resort_type_member_vec (void *obj, void *,
gt_pointer_operator new_value, void* cookie)
{
if (vec<tree, va_gc> *member_vec = (vec<tree, va_gc> *) obj)
{
resort_data.new_value = new_value;
resort_data.cookie = cookie;
member_vec->qsort (resort_member_name_cmp);
}
}
static unsigned
count_class_fields (tree klass)
{
unsigned n_fields = 0;
for (tree fields = TYPE_FIELDS (klass); fields; fields = DECL_CHAIN (fields))
if (DECL_DECLARES_FUNCTION_P (fields))
;
else if (TREE_CODE (fields) == FIELD_DECL
&& ANON_AGGR_TYPE_P (TREE_TYPE (fields)))
n_fields += count_class_fields (TREE_TYPE (fields));
else if (DECL_NAME (fields))
n_fields += 1;
return n_fields;
}
static void
member_vec_append_class_fields (vec<tree, va_gc> *member_vec, tree klass)
{
for (tree fields = TYPE_FIELDS (klass); fields; fields = DECL_CHAIN (fields))
if (DECL_DECLARES_FUNCTION_P (fields))
;
else if (TREE_CODE (fields) == FIELD_DECL
&& ANON_AGGR_TYPE_P (TREE_TYPE (fields)))
member_vec_append_class_fields (member_vec, TREE_TYPE (fields));
else if (DECL_NAME (fields))
{
tree field = fields;
if (TREE_CODE (field) == USING_DECL
&& IDENTIFIER_CONV_OP_P (DECL_NAME (field)))
field = ovl_make (conv_op_marker, field);
member_vec->quick_push (field);
}
}
static void
member_vec_append_enum_values (vec<tree, va_gc> *member_vec, tree enumtype)
{
for (tree values = TYPE_VALUES (enumtype);
values; values = TREE_CHAIN (values))
member_vec->quick_push (TREE_VALUE (values));
}
static void
member_vec_dedup (vec<tree, va_gc> *member_vec)
{
unsigned len = member_vec->length ();
unsigned store = 0;
if (!len)
return;
tree name = OVL_NAME ((*member_vec)[0]);
for (unsigned jx, ix = 0; ix < len; ix = jx)
{
tree current = NULL_TREE;
tree to_type = NULL_TREE;
tree to_using = NULL_TREE;
tree marker = NULL_TREE;
for (jx = ix; jx < len; jx++)
{
tree next = (*member_vec)[jx];
if (jx != ix)
{
tree next_name = OVL_NAME (next);
if (next_name != name)
{
name = next_name;
break;
}
}
if (IDENTIFIER_CONV_OP_P (name))
{
marker = next;
next = OVL_CHAIN (next);
}
if (TREE_CODE (next) == USING_DECL)
{
if (IDENTIFIER_CTOR_P (name))
continue;
next = strip_using_decl (next);
if (TREE_CODE (next) == USING_DECL)
{
to_using = next;
continue;
}
if (is_overloaded_fn (next))
continue;
}
if (DECL_DECLARES_TYPE_P (next))
{
to_type = next;
continue;
}
if (!current)
current = next;
}
if (to_using)
{
if (!current)
current = to_using;
else
current = ovl_make (to_using, current);
}
if (to_type)
{
if (!current)
current = to_type;
else
current = stat_hack (current, to_type);
}
if (current)
{
if (marker)
{
OVL_CHAIN (marker) = current;
current = marker;
}
(*member_vec)[store++] = current;
}
}
while (store++ < len)
member_vec->pop ();
}
void 
set_class_bindings (tree klass, unsigned extra)
{
unsigned n_fields = count_class_fields (klass);
vec<tree, va_gc> *member_vec = CLASSTYPE_MEMBER_VEC (klass);
if (member_vec || n_fields >= 8)
{
vec_safe_reserve_exact (member_vec, extra + n_fields);
member_vec_append_class_fields (member_vec, klass);
}
if (member_vec)
{
CLASSTYPE_MEMBER_VEC (klass) = member_vec;
member_vec->qsort (member_name_cmp);
member_vec_dedup (member_vec);
}
}
void
insert_late_enum_def_bindings (tree klass, tree enumtype)
{
int n_fields;
vec<tree, va_gc> *member_vec = CLASSTYPE_MEMBER_VEC (klass);
if (!member_vec)
n_fields = count_class_fields (klass);
else
n_fields = list_length (TYPE_VALUES (enumtype));
if (member_vec || n_fields >= 8)
{
vec_safe_reserve_exact (member_vec, n_fields);
if (CLASSTYPE_MEMBER_VEC (klass))
member_vec_append_enum_values (member_vec, enumtype);
else
member_vec_append_class_fields (member_vec, klass);
CLASSTYPE_MEMBER_VEC (klass) = member_vec;
member_vec->qsort (member_name_cmp);
member_vec_dedup (member_vec);
}
}
#define ENTRY_INDEX(HASH, COUNT) (((HASH) >> 3) & ((COUNT) - 1))
static GTY((deletable)) binding_entry free_binding_entry = NULL;
cp_binding_oracle_function *cp_binding_oracle;
static inline void
query_oracle (tree name)
{
if (!cp_binding_oracle)
return;
static hash_set<tree> looked_up;
if (looked_up.add (name))
return;
cp_binding_oracle (CP_ORACLE_IDENTIFIER, name);
}
static inline binding_entry
binding_entry_make (tree name, tree type)
{
binding_entry entry;
if (free_binding_entry)
{
entry = free_binding_entry;
free_binding_entry = entry->chain;
}
else
entry = ggc_alloc<binding_entry_s> ();
entry->name = name;
entry->type = type;
entry->chain = NULL;
return entry;
}
#if 0
static inline void
binding_entry_free (binding_entry entry)
{
entry->name = NULL;
entry->type = NULL;
entry->chain = free_binding_entry;
free_binding_entry = entry;
}
#endif
struct GTY(()) binding_table_s {
binding_entry * GTY((length ("%h.chain_count"))) chain;
size_t chain_count;
size_t entry_count;
};
static inline void
binding_table_construct (binding_table table, size_t chain_count)
{
table->chain_count = chain_count;
table->entry_count = 0;
table->chain = ggc_cleared_vec_alloc<binding_entry> (table->chain_count);
}
#if 0
static void
binding_table_free (binding_table table)
{
size_t i;
size_t count;
if (table == NULL)
return;
for (i = 0, count = table->chain_count; i < count; ++i)
{
binding_entry temp = table->chain[i];
while (temp != NULL)
{
binding_entry entry = temp;
temp = entry->chain;
binding_entry_free (entry);
}
table->chain[i] = NULL;
}
table->entry_count = 0;
}
#endif
static inline binding_table
binding_table_new (size_t chain_count)
{
binding_table table = ggc_alloc<binding_table_s> ();
table->chain = NULL;
binding_table_construct (table, chain_count);
return table;
}
static void
binding_table_expand (binding_table table)
{
const size_t old_chain_count = table->chain_count;
const size_t old_entry_count = table->entry_count;
const size_t new_chain_count = 2 * old_chain_count;
binding_entry *old_chains = table->chain;
size_t i;
binding_table_construct (table, new_chain_count);
for (i = 0; i < old_chain_count; ++i)
{
binding_entry entry = old_chains[i];
for (; entry != NULL; entry = old_chains[i])
{
const unsigned int hash = IDENTIFIER_HASH_VALUE (entry->name);
const size_t j = ENTRY_INDEX (hash, new_chain_count);
old_chains[i] = entry->chain;
entry->chain = table->chain[j];
table->chain[j] = entry;
}
}
table->entry_count = old_entry_count;
}
static void
binding_table_insert (binding_table table, tree name, tree type)
{
const unsigned int hash = IDENTIFIER_HASH_VALUE (name);
const size_t i = ENTRY_INDEX (hash, table->chain_count);
binding_entry entry = binding_entry_make (name, type);
entry->chain = table->chain[i];
table->chain[i] = entry;
++table->entry_count;
if (3 * table->chain_count < 5 * table->entry_count)
binding_table_expand (table);
}
binding_entry
binding_table_find (binding_table table, tree name)
{
const unsigned int hash = IDENTIFIER_HASH_VALUE (name);
binding_entry entry = table->chain[ENTRY_INDEX (hash, table->chain_count)];
while (entry != NULL && entry->name != name)
entry = entry->chain;
return entry;
}
void
binding_table_foreach (binding_table table, bt_foreach_proc proc, void *data)
{
size_t chain_count;
size_t i;
if (!table)
return;
chain_count = table->chain_count;
for (i = 0; i < chain_count; ++i)
{
binding_entry entry = table->chain[i];
for (; entry != NULL; entry = entry->chain)
proc (entry, data);
}
}

#ifndef ENABLE_SCOPE_CHECKING
#  define ENABLE_SCOPE_CHECKING 0
#else
#  define ENABLE_SCOPE_CHECKING 1
#endif
static GTY((deletable)) cxx_binding *free_bindings;
static inline void
cxx_binding_init (cxx_binding *binding, tree value, tree type)
{
binding->value = value;
binding->type = type;
binding->previous = NULL;
}
static cxx_binding *
cxx_binding_make (tree value, tree type)
{
cxx_binding *binding;
if (free_bindings)
{
binding = free_bindings;
free_bindings = binding->previous;
}
else
binding = ggc_alloc<cxx_binding> ();
cxx_binding_init (binding, value, type);
return binding;
}
static inline void
cxx_binding_free (cxx_binding *binding)
{
binding->scope = NULL;
binding->previous = free_bindings;
free_bindings = binding;
}
static cxx_binding *
new_class_binding (tree name, tree value, tree type, cp_binding_level *scope)
{
cp_class_binding cb = {cxx_binding_make (value, type), name};
cxx_binding *binding = cb.base;
vec_safe_push (scope->class_shadowed, cb);
binding->scope = scope;
return binding;
}
void
push_binding (tree id, tree decl, cp_binding_level* level)
{
cxx_binding *binding;
if (level != class_binding_level)
{
binding = cxx_binding_make (decl, NULL_TREE);
binding->scope = level;
}
else
binding = new_class_binding (id, decl, NULL_TREE, level);
binding->previous = IDENTIFIER_BINDING (id);
INHERITED_VALUE_BINDING_P (binding) = 0;
LOCAL_BINDING_P (binding) = (level != class_binding_level);
IDENTIFIER_BINDING (id) = binding;
}
void
pop_local_binding (tree id, tree decl)
{
cxx_binding *binding;
if (id == NULL_TREE)
return;
binding = IDENTIFIER_BINDING (id);
gcc_assert (binding != NULL);
if (binding->value == decl)
binding->value = NULL_TREE;
else
{
gcc_assert (binding->type == decl);
binding->type = NULL_TREE;
}
if (!binding->value && !binding->type)
{
IDENTIFIER_BINDING (id) = binding->previous;
cxx_binding_free (binding);
}
}
void
pop_bindings_and_leave_scope (void)
{
for (tree t = get_local_decls (); t; t = DECL_CHAIN (t))
{
tree decl = TREE_CODE (t) == TREE_LIST ? TREE_VALUE (t) : t;
tree name = OVL_NAME (decl);
pop_local_binding (name, decl);
}
leave_scope ();
}
tree
strip_using_decl (tree decl)
{
if (decl == NULL_TREE)
return NULL_TREE;
while (TREE_CODE (decl) == USING_DECL && !DECL_DEPENDENT_P (decl))
decl = USING_DECL_DECLS (decl);
if (TREE_CODE (decl) == USING_DECL && DECL_DEPENDENT_P (decl)
&& USING_DECL_TYPENAME_P (decl))
{
decl = make_typename_type (TREE_TYPE (decl),
DECL_NAME (decl),
typename_type, tf_error);
if (decl != error_mark_node)
decl = TYPE_NAME (decl);
}
return decl;
}
static bool
anticipated_builtin_p (tree ovl)
{
if (TREE_CODE (ovl) != OVERLOAD)
return false;
if (!OVL_HIDDEN_P (ovl))
return false;
tree fn = OVL_FUNCTION (ovl);
gcc_checking_assert (DECL_ANTICIPATED (fn));
if (DECL_HIDDEN_FRIEND_P (fn))
return false;
return true;
}
static bool
supplement_binding_1 (cxx_binding *binding, tree decl)
{
tree bval = binding->value;
bool ok = true;
tree target_bval = strip_using_decl (bval);
tree target_decl = strip_using_decl (decl);
if (TREE_CODE (target_decl) == TYPE_DECL && DECL_ARTIFICIAL (target_decl)
&& target_decl != target_bval
&& (TREE_CODE (target_bval) != TYPE_DECL
|| (processing_template_decl
&& TREE_CODE (TREE_TYPE (target_decl)) == ENUMERAL_TYPE
&& TREE_CODE (TREE_TYPE (target_bval)) == ENUMERAL_TYPE
&& (dependent_type_p (ENUM_UNDERLYING_TYPE
(TREE_TYPE (target_decl)))
|| dependent_type_p (ENUM_UNDERLYING_TYPE
(TREE_TYPE (target_bval)))))))
binding->type = decl;
else if (
!target_bval
|| target_bval == error_mark_node
|| anticipated_builtin_p (target_bval))
binding->value = decl;
else if (TREE_CODE (target_bval) == TYPE_DECL
&& DECL_ARTIFICIAL (target_bval)
&& target_decl != target_bval
&& (TREE_CODE (target_decl) != TYPE_DECL
|| same_type_p (TREE_TYPE (target_decl),
TREE_TYPE (target_bval))))
{
binding->type = bval;
binding->value = decl;
binding->value_is_inherited = false;
}
else if (TREE_CODE (target_bval) == TYPE_DECL
&& TREE_CODE (target_decl) == TYPE_DECL
&& DECL_NAME (target_decl) == DECL_NAME (target_bval)
&& binding->scope->kind != sk_class
&& (same_type_p (TREE_TYPE (target_decl), TREE_TYPE (target_bval))
|| uses_template_parms (TREE_TYPE (target_decl))
|| uses_template_parms (TREE_TYPE (target_bval))))
ok = false;
else if (VAR_P (target_decl)
&& VAR_P (target_bval)
&& DECL_EXTERNAL (target_decl) && DECL_EXTERNAL (target_bval)
&& !DECL_CLASS_SCOPE_P (target_decl))
{
duplicate_decls (decl, binding->value, false);
ok = false;
}
else if (TREE_CODE (decl) == NAMESPACE_DECL
&& TREE_CODE (bval) == NAMESPACE_DECL
&& DECL_NAMESPACE_ALIAS (decl)
&& DECL_NAMESPACE_ALIAS (bval)
&& ORIGINAL_NAMESPACE (bval) == ORIGINAL_NAMESPACE (decl))
ok = false;
else
{
if (!error_operand_p (bval))
diagnose_name_conflict (decl, bval);
ok = false;
}
return ok;
}
static void
diagnose_name_conflict (tree decl, tree bval)
{
if (TREE_CODE (decl) == TREE_CODE (bval)
&& TREE_CODE (decl) != NAMESPACE_DECL
&& !DECL_DECLARES_FUNCTION_P (decl)
&& (TREE_CODE (decl) != TYPE_DECL
|| DECL_ARTIFICIAL (decl) == DECL_ARTIFICIAL (bval))
&& CP_DECL_CONTEXT (decl) == CP_DECL_CONTEXT (bval))
error ("redeclaration of %q#D", decl);
else
error ("%q#D conflicts with a previous declaration", decl);
inform (location_of (bval), "previous declaration %q#D", bval);
}
static bool
supplement_binding (cxx_binding *binding, tree decl)
{
bool ret;
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
ret = supplement_binding_1 (binding, decl);
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return ret;
}
static void
update_local_overload (cxx_binding *binding, tree newval)
{
tree *d;
for (d = &binding->scope->names; ; d = &TREE_CHAIN (*d))
if (*d == binding->value)
{
*d = tree_cons (NULL_TREE, NULL_TREE, TREE_CHAIN (*d));
break;
}
else if (TREE_CODE (*d) == TREE_LIST && TREE_VALUE (*d) == binding->value)
break;
TREE_VALUE (*d) = newval;
}
static bool
matching_fn_p (tree one, tree two)
{
if (!compparms (TYPE_ARG_TYPES (TREE_TYPE (one)),
TYPE_ARG_TYPES (TREE_TYPE (two))))
return false;
if (TREE_CODE (one) == TEMPLATE_DECL
&& TREE_CODE (two) == TEMPLATE_DECL)
{
if (!comp_template_parms (DECL_TEMPLATE_PARMS (one),
DECL_TEMPLATE_PARMS (two)))
return false;
if (!same_type_p (TREE_TYPE (TREE_TYPE (one)),
TREE_TYPE (TREE_TYPE (two))))
return false;
}
return true;
}
static tree
update_binding (cp_binding_level *level, cxx_binding *binding, tree *slot,
tree old, tree decl, bool is_friend)
{
tree to_val = decl;
tree old_type = slot ? MAYBE_STAT_TYPE (*slot) : binding->type;
tree to_type = old_type;
gcc_assert (level->kind == sk_namespace ? !binding
: level->kind != sk_class && !slot);
if (old == error_mark_node)
old = NULL_TREE;
if (TREE_CODE (decl) == TYPE_DECL && DECL_ARTIFICIAL (decl))
{
tree other = to_type;
if (old && TREE_CODE (old) == TYPE_DECL && DECL_ARTIFICIAL (old))
other = old;
if (!other)
;
else if (same_type_p (TREE_TYPE (other), TREE_TYPE (decl)))
return other;
else
goto conflict;
if (old)
{
to_type = decl;
to_val = old;
goto done;
}
}
if (old && TREE_CODE (old) == TYPE_DECL && DECL_ARTIFICIAL (old))
{
to_type = old;
old = NULL_TREE;
}
if (DECL_DECLARES_FUNCTION_P (decl))
{
if (!old)
;
else if (OVL_P (old))
{
for (ovl_iterator iter (old); iter; ++iter)
{
tree fn = *iter;
if (iter.using_p () && matching_fn_p (fn, decl))
{
if (tree match = duplicate_decls (decl, fn, is_friend))
return match;
else
diagnose_name_conflict (decl, fn);
}
}
}
else
goto conflict;
if (to_type != old_type
&& warn_shadow
&& MAYBE_CLASS_TYPE_P (TREE_TYPE (to_type))
&& !(DECL_IN_SYSTEM_HEADER (decl)
&& DECL_IN_SYSTEM_HEADER (to_type)))
warning (OPT_Wshadow, "%q#D hides constructor for %q#D",
decl, to_type);
to_val = ovl_insert (decl, old);
}
else if (!old)
;
else if (TREE_CODE (old) != TREE_CODE (decl))
goto conflict;
else if (TREE_CODE (old) == TYPE_DECL)
{
if (same_type_p (TREE_TYPE (old), TREE_TYPE (decl)))
return old;
else
goto conflict;
}
else if (TREE_CODE (old) == NAMESPACE_DECL)
{
if (ORIGINAL_NAMESPACE (old) != ORIGINAL_NAMESPACE (decl))
goto conflict;
gcc_assert (DECL_NAMESPACE_ALIAS (decl));
return old;
}
else if (TREE_CODE (old) == VAR_DECL)
{
if (!DECL_EXTERNAL (old) || !DECL_EXTERNAL (decl))
goto conflict;
else if (tree match = duplicate_decls (decl, old, false))
return match;
else
goto conflict;
}
else
{
conflict:
diagnose_name_conflict (decl, old);
to_val = NULL_TREE;
}
done:
if (to_val)
{
if (level->kind == sk_namespace || to_type == decl || to_val == decl)
add_decl_to_level (level, decl);
else
{
gcc_checking_assert (binding->value && OVL_P (binding->value));
update_local_overload (binding, to_val);
}
if (slot)
{
if (STAT_HACK_P (*slot))
{
STAT_TYPE (*slot) = to_type;
STAT_DECL (*slot) = to_val;
}
else if (to_type)
*slot = stat_hack (to_val, to_type);
else
*slot = to_val;
}
else
{
binding->type = to_type;
binding->value = to_val;
}
}
return decl;
}
static GTY(()) hash_table<named_decl_hash> *extern_c_decls;
static void
check_extern_c_conflict (tree decl)
{
if (DECL_ARTIFICIAL (decl) || DECL_IN_SYSTEM_HEADER (decl))
return;
if (!DECL_NAMESPACE_SCOPE_P (decl))
return;
if (!extern_c_decls)
extern_c_decls = hash_table<named_decl_hash>::create_ggc (127);
tree *slot = extern_c_decls
->find_slot_with_hash (DECL_NAME (decl),
IDENTIFIER_HASH_VALUE (DECL_NAME (decl)), INSERT);
if (tree old = *slot)
{
if (TREE_CODE (old) == OVERLOAD)
old = OVL_FUNCTION (old);
int mismatch = 0;
if (DECL_CONTEXT (old) == DECL_CONTEXT (decl))
; 
else if (!decls_match (decl, old))
mismatch = 1;
else if (TREE_CODE (decl) == FUNCTION_DECL
&& !comp_except_specs (TYPE_RAISES_EXCEPTIONS (TREE_TYPE (old)),
TYPE_RAISES_EXCEPTIONS (TREE_TYPE (decl)),
ce_normal))
mismatch = -1;
else if (DECL_ASSEMBLER_NAME_SET_P (old))
SET_DECL_ASSEMBLER_NAME (decl, DECL_ASSEMBLER_NAME (old));
if (mismatch)
{
pedwarn (input_location, 0,
"conflicting C language linkage declaration %q#D", decl);
inform (DECL_SOURCE_LOCATION (old),
"previous declaration %q#D", old);
if (mismatch < 0)
inform (input_location,
"due to different exception specifications");
}
else
{
if (old == *slot)
*slot = ovl_make (old, old);
slot = &OVL_CHAIN (*slot);
*slot = tree_cons (NULL_TREE, decl, *slot);
}
}
else
*slot = decl;
}
tree
c_linkage_bindings (tree name)
{
if (extern_c_decls)
if (tree *slot = extern_c_decls
->find_slot_with_hash (name, IDENTIFIER_HASH_VALUE (name), NO_INSERT))
{
tree result = *slot;
if (TREE_CODE (result) == OVERLOAD)
result = OVL_CHAIN (result);
return result;
}
return NULL_TREE;
}
static void
check_local_shadow (tree decl)
{
if (TREE_CODE (decl) == PARM_DECL && !DECL_CONTEXT (decl))
return;
if (DECL_FROM_INLINE (decl))
return;
if (DECL_EXTERNAL (decl))
return;
tree old = NULL_TREE;
cp_binding_level *old_scope = NULL;
if (cxx_binding *binding = outer_binding (DECL_NAME (decl), NULL, true))
{
old = binding->value;
old_scope = binding->scope;
}
while (old && VAR_P (old) && DECL_DEAD_FOR_LOCAL (old))
old = DECL_SHADOWED_FOR_VAR (old);
tree shadowed = NULL_TREE;
if (old
&& (TREE_CODE (old) == PARM_DECL
|| VAR_P (old)
|| (TREE_CODE (old) == TYPE_DECL
&& (!DECL_ARTIFICIAL (old)
|| TREE_CODE (decl) == TYPE_DECL)))
&& (!DECL_ARTIFICIAL (decl)
|| DECL_IMPLICIT_TYPEDEF_P (decl)
|| (VAR_P (decl) && DECL_ANON_UNION_VAR_P (decl))))
{
if (DECL_CONTEXT (old) == current_function_decl
&& TREE_CODE (decl) != PARM_DECL
&& TREE_CODE (old) == PARM_DECL)
{
cp_binding_level *b = current_binding_level->level_chain;
if (FUNCTION_NEEDS_BODY_BLOCK (current_function_decl))
b = b->level_chain;
if (b->kind == sk_function_parms)
{
error ("declaration of %q#D shadows a parameter", decl);
return;
}
}
if (DECL_CONTEXT (old) != current_function_decl)
{
for (cp_binding_level *scope = current_binding_level;
scope != old_scope; scope = scope->level_chain)
if (scope->kind == sk_class
&& !LAMBDA_TYPE_P (scope->this_entity))
return;
}
else if (VAR_P (old)
&& old_scope == current_binding_level->level_chain
&& (old_scope->kind == sk_cond || old_scope->kind == sk_for))
{
error ("redeclaration of %q#D", decl);
inform (DECL_SOURCE_LOCATION (old),
"%q#D previously declared here", old);
return;
}
else if ((TREE_CODE (old) == VAR_DECL
&& old_scope == current_binding_level->level_chain
&& old_scope->kind == sk_catch)
|| (TREE_CODE (old) == PARM_DECL
&& (current_binding_level->kind == sk_catch
|| current_binding_level->level_chain->kind == sk_catch)
&& in_function_try_handler))
{
if (permerror (input_location, "redeclaration of %q#D", decl))
inform (DECL_SOURCE_LOCATION (old),
"%q#D previously declared here", old);
return;
}
enum opt_code warning_code;
if (warn_shadow)
warning_code = OPT_Wshadow;
else if (warn_shadow_local)
warning_code = OPT_Wshadow_local;
else if (warn_shadow_compatible_local
&& (same_type_p (TREE_TYPE (old), TREE_TYPE (decl))
|| (!dependent_type_p (TREE_TYPE (decl))
&& !dependent_type_p (TREE_TYPE (old))
&& !type_uses_auto (TREE_TYPE (decl))
&& can_convert (TREE_TYPE (old), TREE_TYPE (decl),
tf_none))))
warning_code = OPT_Wshadow_compatible_local;
else
return;
const char *msg;
if (TREE_CODE (old) == PARM_DECL)
msg = "declaration of %q#D shadows a parameter";
else if (is_capture_proxy (old))
msg = "declaration of %qD shadows a lambda capture";
else
msg = "declaration of %qD shadows a previous local";
if (warning_at (input_location, warning_code, msg, decl))
{
shadowed = old;
goto inform_shadowed;
}
return;
}
if (!warn_shadow)
return;
if (DECL_ARTIFICIAL (decl) && !DECL_IMPLICIT_TYPEDEF_P (decl))
return;
if (nonlambda_method_basetype ())
if (tree member = lookup_member (current_nonlambda_class_type (),
DECL_NAME (decl), 0,
false, tf_warning_or_error))
{
member = MAYBE_BASELINK_FUNCTIONS (member);
if (!OVL_P (member)
|| TREE_CODE (decl) == FUNCTION_DECL
|| TYPE_PTRFN_P (TREE_TYPE (decl))
|| TYPE_PTRMEMFUNC_P (TREE_TYPE (decl)))
{
if (warning_at (input_location, OPT_Wshadow,
"declaration of %qD shadows a member of %qT",
decl, current_nonlambda_class_type ())
&& DECL_P (member))
{
shadowed = member;
goto inform_shadowed;
}
}
return;
}
old = find_namespace_value (current_namespace, DECL_NAME (decl));
if (old
&& (VAR_P (old)
|| (TREE_CODE (old) == TYPE_DECL
&& (!DECL_ARTIFICIAL (old)
|| TREE_CODE (decl) == TYPE_DECL)))
&& !instantiating_current_function_p ())
{
if (warning_at (input_location, OPT_Wshadow,
"declaration of %qD shadows a global declaration",
decl))
{
shadowed = old;
goto inform_shadowed;
}
return;
}
return;
inform_shadowed:
inform (DECL_SOURCE_LOCATION (shadowed), "shadowed declaration is here");
}
static void
set_decl_context_in_fn (tree ctx, tree decl)
{
if (!DECL_CONTEXT (decl)
&& TREE_CODE (decl) != FUNCTION_DECL
&& !(VAR_P (decl) && DECL_EXTERNAL (decl))
&& !(TREE_CODE (decl) == PARM_DECL
&& current_binding_level->kind == sk_function_parms
&& current_binding_level->this_entity == NULL))
DECL_CONTEXT (decl) = ctx;
if (TREE_CODE (decl) == FUNCTION_DECL && DECL_NAMESPACE_SCOPE_P (decl))
DECL_LOCAL_FUNCTION_P (decl) = 1;
}
static void
set_local_extern_decl_linkage (tree decl, bool shadowed)
{
tree ns_value = decl; 
if (!shadowed)
{
tree loc_value = innermost_non_namespace_value (DECL_NAME (decl));
if (!loc_value)
{
ns_value
= find_namespace_value (current_namespace, DECL_NAME (decl));
loc_value = ns_value;
}
if (loc_value == error_mark_node
|| (loc_value && TREE_CODE (loc_value) == TREE_LIST))
loc_value = NULL_TREE;
for (ovl_iterator iter (loc_value); iter; ++iter)
if (!iter.hidden_p ()
&& (TREE_STATIC (*iter) || DECL_EXTERNAL (*iter))
&& decls_match (*iter, decl))
{
struct cxx_int_tree_map *h;
TREE_PUBLIC (decl) = TREE_PUBLIC (*iter);
if (cp_function_chain->extern_decl_map == NULL)
cp_function_chain->extern_decl_map
= hash_table<cxx_int_tree_map_hasher>::create_ggc (20);
h = ggc_alloc<cxx_int_tree_map> ();
h->uid = DECL_UID (decl);
h->to = *iter;
cxx_int_tree_map **loc = cp_function_chain->extern_decl_map
->find_slot (h, INSERT);
*loc = h;
break;
}
}
if (TREE_PUBLIC (decl))
{
if (ns_value == decl)
ns_value = find_namespace_value (current_namespace, DECL_NAME (decl));
if (ns_value == error_mark_node
|| (ns_value && TREE_CODE (ns_value) == TREE_LIST))
ns_value = NULL_TREE;
for (ovl_iterator iter (ns_value); iter; ++iter)
{
tree other = *iter;
if (!(TREE_PUBLIC (other) || DECL_EXTERNAL (other)))
; 
else if (DECL_EXTERN_C_P (decl) && DECL_EXTERN_C_P (other))
; 
else if (TREE_CODE (other) != TREE_CODE (decl)
|| ((VAR_P (decl) || matching_fn_p (other, decl))
&& !comptypes (TREE_TYPE (decl), TREE_TYPE (other),
COMPARE_REDECLARATION)))
{
if (permerror (DECL_SOURCE_LOCATION (decl),
"local external declaration %q#D", decl))
inform (DECL_SOURCE_LOCATION (other),
"does not match previous declaration %q#D", other);
break;
}
}
}
}
static tree
do_pushdecl (tree decl, bool is_friend)
{
if (decl == error_mark_node)
return error_mark_node;
if (!DECL_TEMPLATE_PARM_P (decl) && current_function_decl)
set_decl_context_in_fn (current_function_decl, decl);
cp_binding_level *level = current_binding_level;
while (level->kind == sk_class)
level = level->level_chain;
tree name = DECL_NAME (decl);
if (name || TREE_CODE (decl) == NAMESPACE_DECL)
{
cxx_binding *binding = NULL; 
tree ns = NULL_TREE; 
tree *slot = NULL; 
tree old = NULL_TREE;
if (level->kind == sk_namespace)
{
ns = (DECL_NAMESPACE_SCOPE_P (decl)
? CP_DECL_CONTEXT (decl) : current_namespace);
slot = find_namespace_slot (ns, name, ns == current_namespace);
if (slot)
old = MAYBE_STAT_DECL (*slot);
}
else
{
binding = find_local_binding (level, name);
if (binding)
old = binding->value;
}
if (current_function_decl && VAR_OR_FUNCTION_DECL_P (decl)
&& DECL_EXTERNAL (decl))
set_local_extern_decl_linkage (decl, old != NULL_TREE);
if (old == error_mark_node)
old = NULL_TREE;
for (ovl_iterator iter (old); iter; ++iter)
if (iter.using_p ())
; 
else if (tree match = duplicate_decls (decl, *iter, is_friend))
{
if (match == error_mark_node)
;
else if (TREE_CODE (match) == TYPE_DECL)
SET_IDENTIFIER_TYPE_VALUE (name, TREE_TYPE (match));
else if (iter.hidden_p () && !DECL_HIDDEN_P (match))
{
tree head = iter.reveal_node (old);
if (head != old)
{
if (!ns)
{
update_local_overload (binding, head);
binding->value = head;
}
else if (STAT_HACK_P (*slot))
STAT_DECL (*slot) = head;
else
*slot = head;
}
if (DECL_EXTERN_C_P (match))
check_extern_c_conflict (match);
}
return match;
}
if (old && anticipated_builtin_p (old))
old = OVL_CHAIN (old);
check_template_shadow (decl);
bool visible_injection = false;
if (DECL_DECLARES_FUNCTION_P (decl))
{
check_default_args (decl);
if (is_friend)
{
if (level->kind != sk_namespace)
{
error ("friend declaration %qD in local class without "
"prior local declaration", decl);
return error_mark_node;
}
if (!flag_friend_injection)
DECL_ANTICIPATED (decl) = DECL_HIDDEN_FRIEND_P (decl) = true;
else
visible_injection = true;
}
}
if (level->kind != sk_namespace)
{
check_local_shadow (decl);
if (TREE_CODE (decl) == NAMESPACE_DECL)
set_identifier_type_value (name, NULL_TREE);
if (!binding)
binding = create_local_binding (level, name);
}
else if (!slot)
{
ns = current_namespace;
slot = find_namespace_slot (ns, name, true);
old = MAYBE_STAT_DECL (*slot);
}
old = update_binding (level, binding, slot, old, decl, is_friend);
if (old != decl)
decl = old;
else if (TREE_CODE (decl) == TYPE_DECL)
{
tree type = TREE_TYPE (decl);
if (type != error_mark_node)
{
if (TYPE_NAME (type) != decl)
set_underlying_type (decl);
if (!ns)
set_identifier_type_value_with_scope (name, decl, level);
else
SET_IDENTIFIER_TYPE_VALUE (name, global_type_node);
}
if (!instantiating_current_function_p ())
record_locally_defined_typedef (decl);
}
else if (VAR_P (decl))
maybe_register_incomplete_var (decl);
else if (visible_injection)
warning (0, "injected friend %qD is visible"
" due to %<-ffriend-injection%>", decl);
if ((VAR_P (decl) || TREE_CODE (decl) == FUNCTION_DECL)
&& DECL_EXTERN_C_P (decl))
check_extern_c_conflict (decl);
}
else
add_decl_to_level (level, decl);
return decl;
}
tree
pushdecl (tree x, bool is_friend)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
tree ret = do_pushdecl (x, is_friend);
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return ret;
}
tree
maybe_push_decl (tree decl)
{
tree type = TREE_TYPE (decl);
if (decl == error_mark_node
|| (TREE_CODE (decl) != PARM_DECL
&& DECL_CONTEXT (decl) != NULL_TREE
&& !DECL_NAMESPACE_SCOPE_P (decl))
|| (TREE_CODE (decl) == TEMPLATE_DECL && !namespace_bindings_p ())
|| type == unknown_type_node
|| (TREE_CODE (decl) == FUNCTION_DECL
&& DECL_TEMPLATE_SPECIALIZATION (decl)))
return decl;
else
return pushdecl (decl);
}
static void
push_local_binding (tree id, tree decl, bool is_using)
{
cp_binding_level *b = innermost_nonclass_level ();
gcc_assert (b->kind != sk_namespace);
if (find_local_binding (b, id))
{
if (!supplement_binding (IDENTIFIER_BINDING (id), decl))
return;
}
else
push_binding (id, decl, b);
if (TREE_CODE (decl) == OVERLOAD || is_using)
decl = build_tree_list (NULL_TREE, decl);
add_decl_to_level (b, decl);
}
tree
check_for_out_of_scope_variable (tree decl)
{
tree shadowed;
if (!(VAR_P (decl) && DECL_DEAD_FOR_LOCAL (decl)))
return decl;
shadowed = DECL_HAS_SHADOWED_FOR_VAR_P (decl)
? DECL_SHADOWED_FOR_VAR (decl) : NULL_TREE ;
while (shadowed != NULL_TREE && VAR_P (shadowed)
&& DECL_DEAD_FOR_LOCAL (shadowed))
shadowed = DECL_HAS_SHADOWED_FOR_VAR_P (shadowed)
? DECL_SHADOWED_FOR_VAR (shadowed) : NULL_TREE;
if (!shadowed)
shadowed = find_namespace_value (current_namespace, DECL_NAME (decl));
if (shadowed)
{
if (!DECL_ERROR_REPORTED (decl)
&& flag_permissive
&& warning (0, "name lookup of %qD changed", DECL_NAME (decl)))
{
inform (DECL_SOURCE_LOCATION (shadowed),
"matches this %qD under ISO standard rules", shadowed);
inform (DECL_SOURCE_LOCATION (decl),
"  matches this %qD under old rules", decl);
}
DECL_ERROR_REPORTED (decl) = 1;
return shadowed;
}
if (DECL_ERROR_REPORTED (decl))
return decl;
DECL_ERROR_REPORTED (decl) = 1;
if (TREE_TYPE (decl) == error_mark_node)
return decl;
if (TYPE_HAS_NONTRIVIAL_DESTRUCTOR (TREE_TYPE (decl)))
{
error ("name lookup of %qD changed for ISO %<for%> scoping",
DECL_NAME (decl));
inform (DECL_SOURCE_LOCATION (decl),
"cannot use obsolete binding %qD because it has a destructor",
decl);
return error_mark_node;
}
else
{
permerror (input_location,
"name lookup of %qD changed for ISO %<for%> scoping",
DECL_NAME (decl));
if (flag_permissive)
inform (DECL_SOURCE_LOCATION (decl),
"using obsolete binding %qD", decl);
static bool hint;
if (!hint)
inform (input_location, flag_permissive
? "this flexibility is deprecated and will be removed"
: "if you use %<-fpermissive%> G++ will accept your code");
hint = true;
}
return decl;
}

static bool keep_next_level_flag;
static int binding_depth = 0;
static void
indent (int depth)
{
int i;
for (i = 0; i < depth * 2; i++)
putc (' ', stderr);
}
static const char *
cp_binding_level_descriptor (cp_binding_level *scope)
{
static const char* scope_kind_names[] = {
"block-scope",
"cleanup-scope",
"try-scope",
"catch-scope",
"for-scope",
"function-parameter-scope",
"class-scope",
"namespace-scope",
"template-parameter-scope",
"template-explicit-spec-scope"
};
const scope_kind kind = scope->explicit_spec_p
? sk_template_spec : scope->kind;
return scope_kind_names[kind];
}
static void
cp_binding_level_debug (cp_binding_level *scope, int line, const char *action)
{
const char *desc = cp_binding_level_descriptor (scope);
if (scope->this_entity)
verbatim ("%s %<%s(%E)%> %p %d\n", action, desc,
scope->this_entity, (void *) scope, line);
else
verbatim ("%s %s %p %d\n", action, desc, (void *) scope, line);
}
static inline size_t
namespace_scope_ht_size (tree ns)
{
tree name = DECL_NAME (ns);
return name == std_identifier
? NAMESPACE_STD_HT_SIZE
: (name == global_identifier
? GLOBAL_SCOPE_HT_SIZE
: NAMESPACE_ORDINARY_HT_SIZE);
}
static GTY((deletable)) cp_binding_level *free_binding_level;
void
push_binding_level (cp_binding_level *scope)
{
scope->level_chain = current_binding_level;
current_binding_level = scope;
keep_next_level_flag = false;
if (ENABLE_SCOPE_CHECKING)
{
scope->binding_depth = binding_depth;
indent (binding_depth);
cp_binding_level_debug (scope, LOCATION_LINE (input_location),
"push");
binding_depth++;
}
}
cp_binding_level *
begin_scope (scope_kind kind, tree entity)
{
cp_binding_level *scope;
if (!ENABLE_SCOPE_CHECKING && free_binding_level)
{
scope = free_binding_level;
free_binding_level = scope->level_chain;
memset (scope, 0, sizeof (cp_binding_level));
}
else
scope = ggc_cleared_alloc<cp_binding_level> ();
scope->this_entity = entity;
scope->more_cleanups_ok = true;
switch (kind)
{
case sk_cleanup:
scope->keep = true;
break;
case sk_template_spec:
scope->explicit_spec_p = true;
kind = sk_template_parms;
case sk_template_parms:
case sk_block:
case sk_try:
case sk_catch:
case sk_for:
case sk_cond:
case sk_class:
case sk_scoped_enum:
case sk_function_parms:
case sk_transaction:
case sk_omp:
scope->keep = keep_next_level_flag;
break;
case sk_namespace:
NAMESPACE_LEVEL (entity) = scope;
break;
default:
gcc_unreachable ();
break;
}
scope->kind = kind;
push_binding_level (scope);
return scope;
}
cp_binding_level *
leave_scope (void)
{
cp_binding_level *scope = current_binding_level;
if (scope->kind == sk_namespace && class_binding_level)
current_binding_level = class_binding_level;
if (NAMESPACE_LEVEL (global_namespace))
gcc_assert (!global_scope_p (scope));
if (ENABLE_SCOPE_CHECKING)
{
indent (--binding_depth);
cp_binding_level_debug (scope, LOCATION_LINE (input_location),
"leave");
}
current_binding_level = scope->level_chain;
if (scope->kind != sk_namespace
&& scope->kind != sk_class)
{
scope->level_chain = free_binding_level;
gcc_assert (!ENABLE_SCOPE_CHECKING
|| scope->binding_depth == binding_depth);
free_binding_level = scope;
}
if (scope->kind == sk_class)
{
scope->defining_class_p = 0;
class_binding_level = NULL;
for (scope = current_binding_level; scope; scope = scope->level_chain)
if (scope->kind == sk_class)
{
class_binding_level = scope;
break;
}
}
return current_binding_level;
}
static void
resume_scope (cp_binding_level* b)
{
gcc_assert (!class_binding_level);
gcc_assert (b->level_chain == current_binding_level);
current_binding_level = b;
if (ENABLE_SCOPE_CHECKING)
{
b->binding_depth = binding_depth;
indent (binding_depth);
cp_binding_level_debug (b, LOCATION_LINE (input_location), "resume");
binding_depth++;
}
}
static cp_binding_level *
innermost_nonclass_level (void)
{
cp_binding_level *b;
b = current_binding_level;
while (b->kind == sk_class)
b = b->level_chain;
return b;
}
void
maybe_push_cleanup_level (tree type)
{
if (type != error_mark_node
&& TYPE_HAS_NONTRIVIAL_DESTRUCTOR (type)
&& current_binding_level->more_cleanups_ok == 0)
{
begin_scope (sk_cleanup, NULL);
current_binding_level->statement_list = push_stmt_list ();
}
}
bool
global_bindings_p (void)
{
return global_scope_p (current_binding_level);
}
bool
toplevel_bindings_p (void)
{
cp_binding_level *b = innermost_nonclass_level ();
return b->kind == sk_namespace || b->kind == sk_template_parms;
}
bool
namespace_bindings_p (void)
{
cp_binding_level *b = innermost_nonclass_level ();
return b->kind == sk_namespace;
}
bool
local_bindings_p (void)
{
cp_binding_level *b = innermost_nonclass_level ();
return b->kind < sk_function_parms || b->kind == sk_omp;
}
bool
kept_level_p (void)
{
return (current_binding_level->blocks != NULL_TREE
|| current_binding_level->keep
|| current_binding_level->kind == sk_cleanup
|| current_binding_level->names != NULL_TREE
|| current_binding_level->using_directives);
}
scope_kind
innermost_scope_kind (void)
{
return current_binding_level->kind;
}
bool
template_parm_scope_p (void)
{
return innermost_scope_kind () == sk_template_parms;
}
void
keep_next_level (bool keep)
{
keep_next_level_flag = keep;
}
tree
get_local_decls (void)
{
gcc_assert (current_binding_level->kind != sk_namespace
&& current_binding_level->kind != sk_class);
return current_binding_level->names;
}
int
function_parm_depth (void)
{
int level = 0;
cp_binding_level *b;
for (b = current_binding_level;
b->kind == sk_function_parms;
b = b->level_chain)
++level;
return level;
}
static int no_print_functions = 0;
static int no_print_builtins = 0;
static void
print_binding_level (cp_binding_level* lvl)
{
tree t;
int i = 0, len;
fprintf (stderr, " blocks=%p", (void *) lvl->blocks);
if (lvl->more_cleanups_ok)
fprintf (stderr, " more-cleanups-ok");
if (lvl->have_cleanups)
fprintf (stderr, " have-cleanups");
fprintf (stderr, "\n");
if (lvl->names)
{
fprintf (stderr, " names:\t");
for (t = lvl->names; t; t = TREE_CHAIN (t))
{
if (no_print_functions && (TREE_CODE (t) == FUNCTION_DECL))
continue;
if (no_print_builtins
&& (TREE_CODE (t) == TYPE_DECL)
&& DECL_IS_BUILTIN (t))
continue;
if (TREE_CODE (t) == FUNCTION_DECL)
len = 3;
else
len = 2;
i += len;
if (i > 6)
{
fprintf (stderr, "\n\t");
i = len;
}
print_node_brief (stderr, "", t, 0);
if (t == error_mark_node)
break;
}
if (i)
fprintf (stderr, "\n");
}
if (vec_safe_length (lvl->class_shadowed))
{
size_t i;
cp_class_binding *b;
fprintf (stderr, " class-shadowed:");
FOR_EACH_VEC_ELT (*lvl->class_shadowed, i, b)
fprintf (stderr, " %s ", IDENTIFIER_POINTER (b->identifier));
fprintf (stderr, "\n");
}
if (lvl->type_shadowed)
{
fprintf (stderr, " type-shadowed:");
for (t = lvl->type_shadowed; t; t = TREE_CHAIN (t))
{
fprintf (stderr, " %s ", IDENTIFIER_POINTER (TREE_PURPOSE (t)));
}
fprintf (stderr, "\n");
}
}
DEBUG_FUNCTION void
debug (cp_binding_level &ref)
{
print_binding_level (&ref);
}
DEBUG_FUNCTION void
debug (cp_binding_level *ptr)
{
if (ptr)
debug (*ptr);
else
fprintf (stderr, "<nil>\n");
}
void
print_other_binding_stack (cp_binding_level *stack)
{
cp_binding_level *level;
for (level = stack; !global_scope_p (level); level = level->level_chain)
{
fprintf (stderr, "binding level %p\n", (void *) level);
print_binding_level (level);
}
}
void
print_binding_stack (void)
{
cp_binding_level *b;
fprintf (stderr, "current_binding_level=%p\n"
"class_binding_level=%p\n"
"NAMESPACE_LEVEL (global_namespace)=%p\n",
(void *) current_binding_level, (void *) class_binding_level,
(void *) NAMESPACE_LEVEL (global_namespace));
if (class_binding_level)
{
for (b = class_binding_level; b; b = b->level_chain)
if (b == current_binding_level)
break;
if (b)
b = class_binding_level;
else
b = current_binding_level;
}
else
b = current_binding_level;
print_other_binding_stack (b);
fprintf (stderr, "global:\n");
print_binding_level (NAMESPACE_LEVEL (global_namespace));
}

static tree
identifier_type_value_1 (tree id)
{
if (REAL_IDENTIFIER_TYPE_VALUE (id) == NULL_TREE)
return NULL_TREE;
if (REAL_IDENTIFIER_TYPE_VALUE (id) != global_type_node)
return REAL_IDENTIFIER_TYPE_VALUE (id);
id = lookup_name_real (id, 2, 1, true, 0, 0);
if (id)
return TREE_TYPE (id);
return NULL_TREE;
}
tree
identifier_type_value (tree id)
{
tree ret;
timevar_start (TV_NAME_LOOKUP);
ret = identifier_type_value_1 (id);
timevar_stop (TV_NAME_LOOKUP);
return ret;
}
static void
set_identifier_type_value_with_scope (tree id, tree decl, cp_binding_level *b)
{
tree type;
if (b->kind != sk_namespace)
{
tree old_type_value = REAL_IDENTIFIER_TYPE_VALUE (id);
b->type_shadowed
= tree_cons (id, old_type_value, b->type_shadowed);
type = decl ? TREE_TYPE (decl) : NULL_TREE;
TREE_TYPE (b->type_shadowed) = type;
}
else
{
tree *slot = find_namespace_slot (current_namespace, id, true);
gcc_assert (decl);
update_binding (b, NULL, slot, MAYBE_STAT_DECL (*slot), decl, false);
type = global_type_node;
}
SET_IDENTIFIER_TYPE_VALUE (id, type);
}
void
set_identifier_type_value (tree id, tree decl)
{
set_identifier_type_value_with_scope (id, decl, current_binding_level);
}
tree
constructor_name (tree type)
{
tree decl = TYPE_NAME (TYPE_MAIN_VARIANT (type));
return decl ? DECL_NAME (decl) : NULL_TREE;
}
bool
constructor_name_p (tree name, tree type)
{
gcc_assert (MAYBE_CLASS_TYPE_P (type));
if (TREE_CODE (type) == DECLTYPE_TYPE
|| TREE_CODE (type) == TYPEOF_TYPE)
return false;
if (name && name == constructor_name (type))
return true;
return false;
}
static GTY(()) int anon_cnt;
tree
make_anon_name (void)
{
char buf[32];
sprintf (buf, anon_aggrname_format (), anon_cnt++);
return get_identifier (buf);
}
static GTY(()) int lambda_cnt = 0;
tree
make_lambda_name (void)
{
char buf[32];
sprintf (buf, LAMBDANAME_FORMAT, lambda_cnt++);
return get_identifier (buf);
}
static tree
push_using_decl_1 (tree scope, tree name)
{
tree decl;
gcc_assert (TREE_CODE (scope) == NAMESPACE_DECL);
gcc_assert (identifier_p (name));
for (decl = current_binding_level->usings; decl; decl = DECL_CHAIN (decl))
if (USING_DECL_SCOPE (decl) == scope && DECL_NAME (decl) == name)
break;
if (decl)
return namespace_bindings_p () ? decl : NULL_TREE;
decl = build_lang_decl (USING_DECL, name, NULL_TREE);
USING_DECL_SCOPE (decl) = scope;
DECL_CHAIN (decl) = current_binding_level->usings;
current_binding_level->usings = decl;
return decl;
}
static tree
push_using_decl (tree scope, tree name)
{
tree ret;
timevar_start (TV_NAME_LOOKUP);
ret = push_using_decl_1 (scope, name);
timevar_stop (TV_NAME_LOOKUP);
return ret;
}
static tree
do_pushdecl_with_scope (tree x, cp_binding_level *level, bool is_friend)
{
cp_binding_level *b;
if (level->kind == sk_class)
{
b = class_binding_level;
class_binding_level = level;
pushdecl_class_level (x);
class_binding_level = b;
}
else
{
tree function_decl = current_function_decl;
if (level->kind == sk_namespace)
current_function_decl = NULL_TREE;
b = current_binding_level;
current_binding_level = level;
x = pushdecl (x, is_friend);
current_binding_level = b;
current_function_decl = function_decl;
}
return x;
}
tree
pushdecl_outermost_localscope (tree x)
{
cp_binding_level *b = NULL;
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
for (cp_binding_level *n = current_binding_level;
n->kind != sk_function_parms; n = b->level_chain)
b = n;
tree ret = b ? do_pushdecl_with_scope (x, b, false) : error_mark_node;
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return ret;
}
static tree
validate_nonmember_using_decl (tree decl, tree scope, tree name)
{
if (TYPE_P (scope))
{
error ("%qT is not a namespace or unscoped enum", scope);
return NULL_TREE;
}
else if (scope == error_mark_node)
return NULL_TREE;
if (TREE_CODE (decl) == TEMPLATE_ID_EXPR)
{
error ("a using-declaration cannot specify a template-id.  "
"Try %<using %D%>", name);
return NULL_TREE;
}
if (TREE_CODE (decl) == NAMESPACE_DECL)
{
error ("namespace %qD not allowed in using-declaration", decl);
return NULL_TREE;
}
if (TREE_CODE (decl) == SCOPE_REF)
{
error ("%qT is not a namespace", TREE_OPERAND (decl, 0));
return NULL_TREE;
}
decl = OVL_FIRST (decl);
tree using_decl = push_using_decl (scope, name);
if (using_decl == NULL_TREE
&& at_function_scope_p ()
&& VAR_P (decl))
error ("%qD is already declared in this scope", name);
return using_decl;
}
static void
do_nonmember_using_decl (tree scope, tree name, tree *value_p, tree *type_p)
{
name_lookup lookup (name, 0);
if (!qualified_namespace_lookup (scope, &lookup))
{
error ("%qD not declared", name);
return;
}
else if (TREE_CODE (lookup.value) == TREE_LIST)
{
error ("reference to %qD is ambiguous", name);
print_candidates (lookup.value);
lookup.value = NULL_TREE;
}
if (lookup.type && TREE_CODE (lookup.type) == TREE_LIST)
{
error ("reference to %qD is ambiguous", name);
print_candidates (lookup.type);
lookup.type = NULL_TREE;
}
tree value = *value_p;
tree type = *type_p;
if (value && DECL_IMPLICIT_TYPEDEF_P (value))
{
type = value;
value = NULL_TREE;
}
if (lookup.value && DECL_IMPLICIT_TYPEDEF_P (lookup.value))
{
lookup.type = lookup.value;
lookup.value = NULL_TREE;
}
if (lookup.value && lookup.value != value)
{
if (OVL_P (lookup.value) && (!value || OVL_P (value)))
{
for (lkp_iterator usings (lookup.value); usings; ++usings)
{
tree new_fn = *usings;
bool found = false;
for (ovl_iterator old (value); !found && old; ++old)
{
tree old_fn = *old;
if (new_fn == old_fn)
found = true;
else if (old.using_p ())
continue; 
else if (old.hidden_p () && !DECL_HIDDEN_FRIEND_P (old_fn))
continue; 
else if (!matching_fn_p (new_fn, old_fn))
continue; 
else if (decls_match (new_fn, old_fn))
found = true;
else
{
diagnose_name_conflict (new_fn, old_fn);
found = true;
}
}
if (!found)
value = ovl_insert (new_fn, value, true);
}
}
else if (value
&& !anticipated_builtin_p (value)
&& !decls_match (lookup.value, value))
diagnose_name_conflict (lookup.value, value);
else
value = lookup.value;
}
if (lookup.type && lookup.type != type)
{
if (type && !decls_match (lookup.type, type))
diagnose_name_conflict (lookup.type, type);
else
type = lookup.type;
}
if (!value)
{
value = type;
type = NULL_TREE;
}
*value_p = value;
*type_p = type;
}
bool
is_nested_namespace (tree ancestor, tree descendant, bool inline_only)
{
int depth = SCOPE_DEPTH (ancestor);
if (!depth && !inline_only)
return true;
while (SCOPE_DEPTH (descendant) > depth
&& (!inline_only || DECL_NAMESPACE_INLINE_P (descendant)))
descendant = CP_DECL_CONTEXT (descendant);
return ancestor == descendant;
}
bool
is_ancestor (tree root, tree child)
{
gcc_assert ((TREE_CODE (root) == NAMESPACE_DECL
|| TREE_CODE (root) == FUNCTION_DECL
|| CLASS_TYPE_P (root)));
gcc_assert ((TREE_CODE (child) == NAMESPACE_DECL
|| CLASS_TYPE_P (child)));
if (root == global_namespace)
return true;
while (TREE_CODE (child) != NAMESPACE_DECL)
{
if (root == child)
return true;
if (TYPE_P (child))
child = TYPE_NAME (child);
child = CP_DECL_CONTEXT (child);
}
if (TREE_CODE (root) == NAMESPACE_DECL)
return is_nested_namespace (root, child);
return false;
}
tree
push_scope (tree t)
{
if (TREE_CODE (t) == NAMESPACE_DECL)
push_decl_namespace (t);
else if (CLASS_TYPE_P (t))
{
if (!at_class_scope_p ()
|| !same_type_p (current_class_type, t))
push_nested_class (t);
else
t = NULL_TREE;
}
return t;
}
void
pop_scope (tree t)
{
if (t == NULL_TREE)
return;
if (TREE_CODE (t) == NAMESPACE_DECL)
pop_decl_namespace ();
else if CLASS_TYPE_P (t)
pop_nested_class ();
}
static void
push_inner_scope_r (tree outer, tree inner)
{
tree prev;
if (outer == inner
|| (TREE_CODE (inner) != NAMESPACE_DECL && !CLASS_TYPE_P (inner)))
return;
prev = CP_DECL_CONTEXT (TREE_CODE (inner) == NAMESPACE_DECL ? inner : TYPE_NAME (inner));
if (outer != prev)
push_inner_scope_r (outer, prev);
if (TREE_CODE (inner) == NAMESPACE_DECL)
{
cp_binding_level *save_template_parm = 0;
while (current_binding_level->kind == sk_template_parms)
{
cp_binding_level *b = current_binding_level;
current_binding_level = b->level_chain;
b->level_chain = save_template_parm;
save_template_parm = b;
}
resume_scope (NAMESPACE_LEVEL (inner));
current_namespace = inner;
while (save_template_parm)
{
cp_binding_level *b = save_template_parm;
save_template_parm = b->level_chain;
b->level_chain = current_binding_level;
current_binding_level = b;
}
}
else
pushclass (inner);
}
tree
push_inner_scope (tree inner)
{
tree outer = current_scope ();
if (!outer)
outer = current_namespace;
push_inner_scope_r (outer, inner);
return outer;
}
void
pop_inner_scope (tree outer, tree inner)
{
if (outer == inner
|| (TREE_CODE (inner) != NAMESPACE_DECL && !CLASS_TYPE_P (inner)))
return;
while (outer != inner)
{
if (TREE_CODE (inner) == NAMESPACE_DECL)
{
cp_binding_level *save_template_parm = 0;
while (current_binding_level->kind == sk_template_parms)
{
cp_binding_level *b = current_binding_level;
current_binding_level = b->level_chain;
b->level_chain = save_template_parm;
save_template_parm = b;
}
pop_namespace ();
while (save_template_parm)
{
cp_binding_level *b = save_template_parm;
save_template_parm = b->level_chain;
b->level_chain = current_binding_level;
current_binding_level = b;
}
}
else
popclass ();
inner = CP_DECL_CONTEXT (TREE_CODE (inner) == NAMESPACE_DECL ? inner : TYPE_NAME (inner));
}
}

void
pushlevel_class (void)
{
class_binding_level = begin_scope (sk_class, current_class_type);
}
void
poplevel_class (void)
{
cp_binding_level *level = class_binding_level;
cp_class_binding *cb;
size_t i;
tree shadowed;
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
gcc_assert (level != 0);
if (current_class_depth == 1)
previous_class_level = level;
for (shadowed = level->type_shadowed;
shadowed;
shadowed = TREE_CHAIN (shadowed))
SET_IDENTIFIER_TYPE_VALUE (TREE_PURPOSE (shadowed), TREE_VALUE (shadowed));
if (level->class_shadowed)
{
FOR_EACH_VEC_ELT (*level->class_shadowed, i, cb)
{
IDENTIFIER_BINDING (cb->identifier) = cb->base->previous;
cxx_binding_free (cb->base);
}
ggc_free (level->class_shadowed);
level->class_shadowed = NULL;
}
gcc_assert (current_binding_level == level);
leave_scope ();
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
}
static void
set_inherited_value_binding_p (cxx_binding *binding, tree decl,
tree class_type)
{
if (binding->value == decl && TREE_CODE (decl) != TREE_LIST)
{
tree context;
if (TREE_CODE (decl) == OVERLOAD)
context = ovl_scope (decl);
else
{
gcc_assert (DECL_P (decl));
context = context_for_name_lookup (decl);
}
if (is_properly_derived_from (class_type, context))
INHERITED_VALUE_BINDING_P (binding) = 1;
else
INHERITED_VALUE_BINDING_P (binding) = 0;
}
else if (binding->value == decl)
INHERITED_VALUE_BINDING_P (binding) = 1;
else
INHERITED_VALUE_BINDING_P (binding) = 0;
}
bool
pushdecl_class_level (tree x)
{
bool is_valid = true;
bool subtime;
if (current_class_type != class_binding_level->this_entity)
return true;
subtime = timevar_cond_start (TV_NAME_LOOKUP);
tree name = OVL_NAME (x);
if (name)
{
is_valid = push_class_level_binding (name, x);
if (TREE_CODE (x) == TYPE_DECL)
set_identifier_type_value (name, x);
}
else if (ANON_AGGR_TYPE_P (TREE_TYPE (x)))
{
location_t save_location = input_location;
tree anon = TREE_TYPE (x);
if (vec<tree, va_gc> *member_vec = CLASSTYPE_MEMBER_VEC (anon))
for (unsigned ix = member_vec->length (); ix--;)
{
tree binding = (*member_vec)[ix];
if (STAT_HACK_P (binding))
{
if (!pushdecl_class_level (STAT_TYPE (binding)))
is_valid = false;
binding = STAT_DECL (binding);
}
if (!pushdecl_class_level (binding))
is_valid = false;
}
else
for (tree f = TYPE_FIELDS (anon); f; f = DECL_CHAIN (f))
if (TREE_CODE (f) == FIELD_DECL)
{
input_location = DECL_SOURCE_LOCATION (f);
if (!pushdecl_class_level (f))
is_valid = false;
}
input_location = save_location;
}
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return is_valid;
}
static cxx_binding *
get_class_binding (tree name, cp_binding_level *scope)
{
tree class_type;
tree type_binding;
tree value_binding;
cxx_binding *binding;
class_type = scope->this_entity;
type_binding = lookup_member (class_type, name,
2, true,
tf_warning_or_error);
value_binding = lookup_member (class_type, name,
2, false,
tf_warning_or_error);
if (value_binding
&& (TREE_CODE (value_binding) == TYPE_DECL
|| DECL_CLASS_TEMPLATE_P (value_binding)
|| (TREE_CODE (value_binding) == TREE_LIST
&& TREE_TYPE (value_binding) == error_mark_node
&& (TREE_CODE (TREE_VALUE (value_binding))
== TYPE_DECL))))
;
else if (value_binding)
{
if (TREE_CODE (value_binding) == TREE_LIST
&& TREE_TYPE (value_binding) == error_mark_node)
;
else if (BASELINK_P (value_binding))
value_binding = BASELINK_FUNCTIONS (value_binding);
}
if (type_binding || value_binding)
{
binding = new_class_binding (name,
value_binding,
type_binding,
scope);
LOCAL_BINDING_P (binding) = 0;
set_inherited_value_binding_p (binding, value_binding, class_type);
}
else
binding = NULL;
return binding;
}
static bool
push_class_level_binding_1 (tree name, tree x)
{
cxx_binding *binding;
tree decl = x;
bool ok;
if (!class_binding_level)
return true;
if (name == error_mark_node)
return false;
if (!identifier_p (name))
{
gcc_assert (errorcount || sorrycount);
return false;
}
gcc_assert (TYPE_BEING_DEFINED (current_class_type)
|| LAMBDA_TYPE_P (TREE_TYPE (decl)));
gcc_assert (current_class_type == class_binding_level->this_entity);
if (TREE_CODE (decl) == TREE_LIST
&& TREE_TYPE (decl) == error_mark_node)
decl = TREE_VALUE (decl);
if (!check_template_shadow (decl))
return false;
if ((VAR_P (x)
|| TREE_CODE (x) == CONST_DECL
|| (TREE_CODE (x) == TYPE_DECL
&& !DECL_SELF_REFERENCE_P (x))
|| (TREE_CODE (x) == FIELD_DECL
&& DECL_CONTEXT (x) != current_class_type))
&& DECL_NAME (x) == DECL_NAME (TYPE_NAME (current_class_type)))
{
tree scope = context_for_name_lookup (x);
if (TYPE_P (scope) && same_type_p (scope, current_class_type))
{
error ("%qD has the same name as the class in which it is "
"declared",
x);
return false;
}
}
binding = IDENTIFIER_BINDING (name);
if (!binding || binding->scope != class_binding_level)
{
binding = get_class_binding (name, class_binding_level);
if (binding)
{
binding->previous = IDENTIFIER_BINDING (name);
IDENTIFIER_BINDING (name) = binding;
}
}
if (binding && binding->value)
{
tree bval = binding->value;
tree old_decl = NULL_TREE;
tree target_decl = strip_using_decl (decl);
tree target_bval = strip_using_decl (bval);
if (INHERITED_VALUE_BINDING_P (binding))
{
if (TREE_CODE (target_bval) == TYPE_DECL
&& DECL_ARTIFICIAL (target_bval)
&& !(TREE_CODE (target_decl) == TYPE_DECL
&& DECL_ARTIFICIAL (target_decl)))
{
old_decl = binding->type;
binding->type = bval;
binding->value = NULL_TREE;
INHERITED_VALUE_BINDING_P (binding) = 0;
}
else
{
old_decl = bval;
if (TREE_CODE (target_decl) == TYPE_DECL
&& DECL_ARTIFICIAL (target_decl))
binding->type = NULL_TREE;
}
}
else if (TREE_CODE (target_decl) == OVERLOAD
&& OVL_P (target_bval))
old_decl = bval;
else if (TREE_CODE (decl) == USING_DECL
&& TREE_CODE (bval) == USING_DECL
&& same_type_p (USING_DECL_SCOPE (decl),
USING_DECL_SCOPE (bval)))
;
else if (TREE_CODE (decl) == USING_DECL
&& TREE_CODE (bval) == USING_DECL
&& DECL_DEPENDENT_P (decl)
&& DECL_DEPENDENT_P (bval))
return true;
else if (TREE_CODE (decl) == USING_DECL
&& OVL_P (target_bval))
old_decl = bval;
else if (TREE_CODE (bval) == USING_DECL
&& OVL_P (target_decl))
return true;
if (old_decl && binding->scope == class_binding_level)
{
binding->value = x;
INHERITED_VALUE_BINDING_P (binding) = 0;
return true;
}
}
note_name_declared_in_class (name, decl);
if (binding && binding->scope == class_binding_level)
ok = supplement_binding (binding, decl);
else
{
push_binding (name, decl, class_binding_level);
ok = true;
}
return ok;
}
bool
push_class_level_binding (tree name, tree x)
{
bool ret;
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
ret = push_class_level_binding_1 (name, x);
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return ret;
}
tree
do_class_using_decl (tree scope, tree name)
{
if (name == error_mark_node)
return NULL_TREE;
if (!scope || !TYPE_P (scope))
{
error ("using-declaration for non-member at class scope");
return NULL_TREE;
}
if (TREE_CODE (name) == BIT_NOT_EXPR)
{
error ("%<%T::%D%> names destructor", scope, name);
return NULL_TREE;
}
if (MAYBE_CLASS_TYPE_P (scope)
&& (name == TYPE_IDENTIFIER (scope)
|| constructor_name_p (name, scope)))
{
maybe_warn_cpp0x (CPP0X_INHERITING_CTORS);
name = ctor_identifier;
CLASSTYPE_NON_AGGREGATE (current_class_type) = true;
}
if (constructor_name_p (name, current_class_type))
{
error ("%<%T::%D%> names constructor in %qT",
scope, name, current_class_type);
return NULL_TREE;
}
tree decl = NULL_TREE;
if (!dependent_scope_p (scope))
{
base_kind b_kind;
tree binfo = lookup_base (current_class_type, scope, ba_any, &b_kind,
tf_warning_or_error);
if (b_kind < bk_proper_base)
{
if (b_kind == bk_same_type || !any_dependent_bases_p ())
{
error_not_base_type (scope, current_class_type);
return NULL_TREE;
}
}
else if (name == ctor_identifier && !binfo_direct_p (binfo))
{
error ("cannot inherit constructors from indirect base %qT", scope);
return NULL_TREE;
}
else if (!IDENTIFIER_CONV_OP_P (name)
|| !dependent_type_p (TREE_TYPE (name)))
{
decl = lookup_member (binfo, name, 0, false, tf_warning_or_error);
if (!decl)
{
error ("no members matching %<%T::%D%> in %q#T", scope, name,
scope);
return NULL_TREE;
}
if (BASELINK_P (decl))
decl = BASELINK_FUNCTIONS (decl);
}
}
tree value = build_lang_decl (USING_DECL, name, NULL_TREE);
USING_DECL_DECLS (value) = decl;
USING_DECL_SCOPE (value) = scope;
DECL_DEPENDENT_P (value) = !decl;
return value;
}

tree
get_namespace_binding (tree ns, tree name)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
if (!ns)
ns = global_namespace;
gcc_checking_assert (!DECL_NAMESPACE_ALIAS (ns));
tree ret = find_namespace_value (ns, name);
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return ret;
}
void
set_global_binding (tree decl)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
tree *slot = find_namespace_slot (global_namespace, DECL_NAME (decl), true);
if (*slot)
diagnose_name_conflict (decl, MAYBE_STAT_DECL (*slot));
*slot = decl;
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
}
void
set_decl_namespace (tree decl, tree scope, bool friendp)
{
scope = ORIGINAL_NAMESPACE (scope);
if (!friendp && !is_nested_namespace (current_namespace, scope))
error ("declaration of %qD not in a namespace surrounding %qD",
decl, scope);
DECL_CONTEXT (decl) = FROB_CONTEXT (scope);
tree old = NULL_TREE;
{
name_lookup lookup (DECL_NAME (decl), LOOKUP_HIDDEN);
if (!lookup.search_qualified (scope, false))
goto not_found;
old = lookup.value;
}
if (TREE_CODE (old) == TREE_LIST)
{
ambiguous:
DECL_CONTEXT (decl) = FROB_CONTEXT (scope);
error ("reference to %qD is ambiguous", decl);
print_candidates (old);
return;
}
if (!DECL_DECLARES_FUNCTION_P (decl))
{
if (TREE_CODE (decl) == TREE_CODE (old))
DECL_CONTEXT (decl) = DECL_CONTEXT (old);
found:
if (CP_DECL_CONTEXT (decl) == current_namespace
&& at_namespace_scope_p ())
error ("explicit qualification in declaration of %qD", decl);
return;
}
if (!OVL_P (old))
goto not_found;
if (processing_explicit_instantiation)
return;
if (processing_template_decl || processing_specialization)
return;
if (friendp && DECL_USE_TEMPLATE (decl))
return;
tree found;
found = NULL_TREE;
for (lkp_iterator iter (old); iter; ++iter)
{
if (iter.using_p ())
continue;
tree ofn = *iter;
DECL_CONTEXT (decl) = DECL_CONTEXT (ofn);
if (decls_match (decl, ofn))
{
if (found)
{
DECL_CONTEXT (decl) = FROB_CONTEXT (scope);
goto ambiguous;
}
found = ofn;
}
}
if (found)
{
if (DECL_HIDDEN_FRIEND_P (found))
{
pedwarn (DECL_SOURCE_LOCATION (decl), 0,
"%qD has not been declared within %qD", decl, scope);
inform (DECL_SOURCE_LOCATION (found),
"only here as a %<friend%>");
}
DECL_CONTEXT (decl) = DECL_CONTEXT (found);
goto found;
}
not_found:
DECL_CONTEXT (decl) = FROB_CONTEXT (scope);
error ("%qD should have been declared inside %qD", decl, scope);
}
tree
current_decl_namespace (void)
{
tree result;
if (!vec_safe_is_empty (decl_namespace_list))
return decl_namespace_list->last ();
if (current_class_type)
result = decl_namespace_context (current_class_type);
else if (current_function_decl)
result = decl_namespace_context (current_function_decl);
else
result = current_namespace;
return result;
}
bool
handle_namespace_attrs (tree ns, tree attributes)
{
tree d;
bool saw_vis = false;
if (attributes == error_mark_node)
return false;
for (d = attributes; d; d = TREE_CHAIN (d))
{
tree name = get_attribute_name (d);
tree args = TREE_VALUE (d);
if (is_attribute_p ("visibility", name))
{
tree x = args ? TREE_VALUE (args) : NULL_TREE;
if (x == NULL_TREE || TREE_CODE (x) != STRING_CST || TREE_CHAIN (args))
{
warning (OPT_Wattributes,
"%qD attribute requires a single NTBS argument",
name);
continue;
}
if (!TREE_PUBLIC (ns))
warning (OPT_Wattributes,
"%qD attribute is meaningless since members of the "
"anonymous namespace get local symbols", name);
push_visibility (TREE_STRING_POINTER (x), 1);
saw_vis = true;
}
else if (is_attribute_p ("abi_tag", name))
{
if (!DECL_NAME (ns))
{
warning (OPT_Wattributes, "ignoring %qD attribute on anonymous "
"namespace", name);
continue;
}
if (!DECL_NAMESPACE_INLINE_P (ns))
{
warning (OPT_Wattributes, "ignoring %qD attribute on non-inline "
"namespace", name);
continue;
}
if (!args)
{
tree dn = DECL_NAME (ns);
args = build_string (IDENTIFIER_LENGTH (dn) + 1,
IDENTIFIER_POINTER (dn));
TREE_TYPE (args) = char_array_type_node;
args = fix_string_type (args);
args = build_tree_list (NULL_TREE, args);
}
if (check_abi_tag_args (args, name))
DECL_ATTRIBUTES (ns) = tree_cons (name, args,
DECL_ATTRIBUTES (ns));
}
else
{
warning (OPT_Wattributes, "%qD attribute directive ignored",
name);
continue;
}
}
return saw_vis;
}
void
push_decl_namespace (tree decl)
{
if (TREE_CODE (decl) != NAMESPACE_DECL)
decl = decl_namespace_context (decl);
vec_safe_push (decl_namespace_list, ORIGINAL_NAMESPACE (decl));
}
void
pop_decl_namespace (void)
{
decl_namespace_list->pop ();
}
void
do_namespace_alias (tree alias, tree name_space)
{
if (name_space == error_mark_node)
return;
gcc_assert (TREE_CODE (name_space) == NAMESPACE_DECL);
name_space = ORIGINAL_NAMESPACE (name_space);
alias = build_lang_decl (NAMESPACE_DECL, alias, void_type_node);
DECL_NAMESPACE_ALIAS (alias) = name_space;
DECL_EXTERNAL (alias) = 1;
DECL_CONTEXT (alias) = FROB_CONTEXT (current_scope ());
pushdecl (alias);
if (!building_stmt_list_p ())
(*debug_hooks->early_global_decl) (alias);
}
tree
pushdecl_namespace_level (tree x, bool is_friend)
{
cp_binding_level *b = current_binding_level;
tree t;
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
t = do_pushdecl_with_scope
(x, NAMESPACE_LEVEL (current_namespace), is_friend);
if (TREE_CODE (t) == TYPE_DECL)
{
tree name = DECL_NAME (t);
tree newval;
tree *ptr = (tree *)0;
for (; !global_scope_p (b); b = b->level_chain)
{
tree shadowed = b->type_shadowed;
for (; shadowed; shadowed = TREE_CHAIN (shadowed))
if (TREE_PURPOSE (shadowed) == name)
{
ptr = &TREE_VALUE (shadowed);
}
}
newval = TREE_TYPE (t);
if (ptr == (tree *)0)
{
SET_IDENTIFIER_TYPE_VALUE (name, t);
}
else
{
*ptr = newval;
}
}
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return t;
}
void
finish_namespace_using_decl (tree decl, tree scope, tree name)
{
tree orig_decl = decl;
gcc_checking_assert (current_binding_level->kind == sk_namespace
&& !processing_template_decl);
decl = validate_nonmember_using_decl (decl, scope, name);
if (decl == NULL_TREE)
return;
tree *slot = find_namespace_slot (current_namespace, name, true);
tree val = slot ? MAYBE_STAT_DECL (*slot) : NULL_TREE;
tree type = slot ? MAYBE_STAT_TYPE (*slot) : NULL_TREE;
do_nonmember_using_decl (scope, name, &val, &type);
if (STAT_HACK_P (*slot))
{
STAT_DECL (*slot) = val;
STAT_TYPE (*slot) = type;
}
else if (type)
*slot = stat_hack (val, type);
else
*slot = val;
cp_emit_debug_info_for_using (orig_decl, current_namespace);
}
void
finish_local_using_decl (tree decl, tree scope, tree name)
{
tree orig_decl = decl;
gcc_checking_assert (current_binding_level->kind != sk_class
&& current_binding_level->kind != sk_namespace);
decl = validate_nonmember_using_decl (decl, scope, name);
if (decl == NULL_TREE)
return;
add_decl_expr (decl);
cxx_binding *binding = find_local_binding (current_binding_level, name);
tree value = binding ? binding->value : NULL_TREE;
tree type = binding ? binding->type : NULL_TREE;
do_nonmember_using_decl (scope, name, &value, &type);
if (!value)
;
else if (binding && value == binding->value)
;
else if (binding && binding->value && TREE_CODE (value) == OVERLOAD)
{
update_local_overload (IDENTIFIER_BINDING (name), value);
IDENTIFIER_BINDING (name)->value = value;
}
else
push_local_binding (name, value, true);
if (!type)
;
else if (binding && type == binding->type)
;
else
{
push_local_binding (name, type, true);
set_identifier_type_value (name, type);
}
if (!processing_template_decl)
cp_emit_debug_info_for_using (orig_decl, current_scope ());
}
tree
cp_namespace_decls (tree ns)
{
return NAMESPACE_LEVEL (ns)->names;
}
static int
lookup_flags (int prefer_type, int namespaces_only)
{
if (namespaces_only)
return LOOKUP_PREFER_NAMESPACES;
if (prefer_type > 1)
return LOOKUP_PREFER_TYPES;
if (prefer_type > 0)
return LOOKUP_PREFER_BOTH;
return 0;
}
static bool
qualify_lookup (tree val, int flags)
{
if (val == NULL_TREE)
return false;
if ((flags & LOOKUP_PREFER_NAMESPACES) && TREE_CODE (val) == NAMESPACE_DECL)
return true;
if (flags & LOOKUP_PREFER_TYPES)
{
tree target_val = strip_using_decl (val);
if (TREE_CODE (target_val) == TYPE_DECL
|| TREE_CODE (target_val) == TEMPLATE_DECL)
return true;
}
if (flags & (LOOKUP_PREFER_NAMESPACES | LOOKUP_PREFER_TYPES))
return false;
if (!(flags & LOOKUP_HIDDEN) && is_lambda_ignored_entity (val))
return false;
return true;
}
static bool
using_directives_contain_std_p (vec<tree, va_gc> *usings)
{
if (!usings)
return false;
for (unsigned ix = usings->length (); ix--;)
if ((*usings)[ix] == std_node)
return true;
return false;
}
static bool
has_using_namespace_std_directive_p ()
{
for (cp_binding_level *level = current_binding_level;
level->kind != sk_namespace;
level = level->level_chain)
if (using_directives_contain_std_p (level->using_directives))
return true;
for (tree scope = current_namespace; scope; scope = CP_DECL_CONTEXT (scope))
{
if (using_directives_contain_std_p (DECL_NAMESPACE_USING (scope)))
return true;
if (scope == global_namespace)
break;
}
return false;
}
void
suggest_alternatives_for (location_t location, tree name,
bool suggest_misspellings)
{
vec<tree> candidates = vNULL;
vec<tree> worklist = vNULL;
unsigned limit = PARAM_VALUE (CXX_MAX_NAMESPACES_FOR_DIAGNOSTIC_HELP);
bool limited = false;
worklist.safe_push (global_namespace);
for (unsigned ix = 0; ix != worklist.length (); ix++)
{
tree ns = worklist[ix];
name_lookup lookup (name);
if (lookup.search_qualified (ns, false))
candidates.safe_push (lookup.value);
if (!limited)
{
vec<tree> children = vNULL;
for (tree decl = NAMESPACE_LEVEL (ns)->names;
decl; decl = TREE_CHAIN (decl))
if (TREE_CODE (decl) == NAMESPACE_DECL
&& !DECL_NAMESPACE_ALIAS (decl)
&& !DECL_NAMESPACE_INLINE_P (decl))
children.safe_push (decl);
while (!limited && !children.is_empty ())
{
if (worklist.length () == limit)
{
inform (location,
"maximum limit of %d namespaces searched for %qE",
limit, name);
limited = true;
}
else
worklist.safe_push (children.pop ());
}
children.release ();
}
}
worklist.release ();
if (candidates.length ())
{
inform_n (location, candidates.length (),
"suggested alternative:",
"suggested alternatives:");
for (unsigned ix = 0; ix != candidates.length (); ix++)
{
tree val = candidates[ix];
inform (location_of (val), "  %qE", val);
}
candidates.release ();
return;
}
if (has_using_namespace_std_directive_p ())
if (maybe_suggest_missing_std_header (location, name))
return;
if (!suggest_misspellings)
return;
if (name_hint hint = lookup_name_fuzzy (name, FUZZY_LOOKUP_NAME,
location))
{
gcc_rich_location richloc (location);
richloc.add_fixit_replace (hint.suggestion ());
inform (&richloc, "suggested alternative: %qs", hint.suggestion ());
}
}
struct std_name_hint
{
const char *name;
const char *header;
enum cxx_dialect min_dialect;
};
static const std_name_hint *
get_std_name_hint (const char *name)
{
static const std_name_hint hints[] = {
{"any", "<any>", cxx17},
{"any_cast", "<any>", cxx17},
{"make_any", "<any>", cxx17},
{"array", "<array>", cxx11},
{"atomic", "<atomic>", cxx11},
{"atomic_flag", "<atomic>", cxx11},
{"bitset", "<bitset>", cxx11},
{"complex", "<complex>", cxx98},
{"complex_literals", "<complex>", cxx98},
{"condition_variable", "<condition_variable>", cxx11},
{"condition_variable_any", "<condition_variable>", cxx11},
{"deque", "<deque>", cxx98},
{"forward_list", "<forward_list>", cxx11},
{"basic_filebuf", "<fstream>", cxx98},
{"basic_ifstream", "<fstream>", cxx98},
{"basic_ofstream", "<fstream>", cxx98},
{"basic_fstream", "<fstream>", cxx98},
{"fstream", "<fstream>", cxx98},
{"ifstream", "<fstream>", cxx98},
{"ofstream", "<fstream>", cxx98},
{"bind", "<functional>", cxx11},
{"function", "<functional>", cxx11},
{"hash", "<functional>", cxx11},
{"mem_fn", "<functional>", cxx11},
{"async", "<future>", cxx11},
{"future", "<future>", cxx11},
{"packaged_task", "<future>", cxx11},
{"promise", "<future>", cxx11},
{"cin", "<iostream>", cxx98},
{"cout", "<iostream>", cxx98},
{"cerr", "<iostream>", cxx98},
{"clog", "<iostream>", cxx98},
{"wcin", "<iostream>", cxx98},
{"wcout", "<iostream>", cxx98},
{"wclog", "<iostream>", cxx98},
{"istream", "<istream>", cxx98},
{"advance", "<iterator>", cxx98},
{"back_inserter", "<iterator>", cxx98},
{"begin", "<iterator>", cxx11},
{"distance", "<iterator>", cxx98},
{"end", "<iterator>", cxx11},
{"front_inserter", "<iterator>", cxx98},
{"inserter", "<iterator>", cxx98},
{"istream_iterator", "<iterator>", cxx98},
{"istreambuf_iterator", "<iterator>", cxx98},
{"iterator_traits", "<iterator>", cxx98},
{"move_iterator", "<iterator>", cxx11},
{"next", "<iterator>", cxx11},
{"ostream_iterator", "<iterator>", cxx98},
{"ostreambuf_iterator", "<iterator>", cxx98},
{"prev", "<iterator>", cxx11},
{"reverse_iterator", "<iterator>", cxx98},
{"ostream", "<ostream>", cxx98},
{"list", "<list>", cxx98},
{"map", "<map>", cxx98},
{"multimap", "<map>", cxx98},
{"make_shared", "<memory>", cxx11},
{"make_unique", "<memory>", cxx11},
{"shared_ptr", "<memory>", cxx11},
{"unique_ptr", "<memory>", cxx11},
{"weak_ptr", "<memory>", cxx11},
{"mutex", "<mutex>", cxx11},
{"timed_mutex", "<mutex>", cxx11},
{"recursive_mutex", "<mutex>", cxx11},
{"recursive_timed_mutex", "<mutex>", cxx11},
{"once_flag", "<mutex>", cxx11},
{"call_once,", "<mutex>", cxx11},
{"lock", "<mutex>", cxx11},
{"scoped_lock", "<mutex>", cxx17},
{"try_lock", "<mutex>", cxx11},
{"lock_guard", "<mutex>", cxx11},
{"unique_lock", "<mutex>", cxx11},
{"optional", "<optional>", cxx17},
{"make_optional", "<optional>", cxx17},
{"ostream", "<ostream>", cxx98},
{"wostream", "<ostream>", cxx98},
{"ends", "<ostream>", cxx98},
{"flush", "<ostream>", cxx98},
{"endl", "<ostream>", cxx98},
{"queue", "<queue>", cxx98},
{"priority_queue", "<queue>", cxx98},
{"set", "<set>", cxx98},
{"multiset", "<set>", cxx98},
{"shared_lock", "<shared_mutex>", cxx14},
{"shared_mutex", "<shared_mutex>", cxx17},
{"shared_timed_mutex", "<shared_mutex>", cxx14},
{"basic_stringbuf", "<sstream>", cxx98},
{"basic_istringstream", "<sstream>", cxx98},
{"basic_ostringstream", "<sstream>", cxx98},
{"basic_stringstream", "<sstream>", cxx98},
{"istringstream", "<sstream>", cxx98},
{"ostringstream", "<sstream>", cxx98},
{"stringstream", "<sstream>", cxx98},
{"stack", "<stack>", cxx98},
{"basic_string", "<string>", cxx98},
{"string", "<string>", cxx98},
{"wstring", "<string>", cxx98},
{"u16string", "<string>", cxx11},
{"u32string", "<string>", cxx11},
{"string_view", "<string_view>", cxx17},
{"thread", "<thread>", cxx11},
{"make_tuple", "<tuple>", cxx11},
{"tuple", "<tuple>", cxx11},
{"tuple_element", "<tuple>", cxx11},
{"tuple_size", "<tuple>", cxx11},
{"unordered_map", "<unordered_map>", cxx11},
{"unordered_multimap", "<unordered_map>", cxx11},
{"unordered_set", "<unordered_set>", cxx11},
{"unordered_multiset", "<unordered_set>", cxx11},
{"declval", "<utility>", cxx11},
{"forward", "<utility>", cxx11},
{"make_pair", "<utility>", cxx98},
{"move", "<utility>", cxx11},
{"pair", "<utility>", cxx98},
{"variant", "<variant>", cxx17},
{"visit", "<variant>", cxx17},
{"vector", "<vector>", cxx98},
};
const size_t num_hints = sizeof (hints) / sizeof (hints[0]);
for (size_t i = 0; i < num_hints; i++)
{
if (strcmp (name, hints[i].name) == 0)
return &hints[i];
}
return NULL;
}
static const char *
get_cxx_dialect_name (enum cxx_dialect dialect)
{
switch (dialect)
{
default:
gcc_unreachable ();
case cxx98:
return "C++98";
case cxx11:
return "C++11";
case cxx14:
return "C++14";
case cxx17:
return "C++17";
case cxx2a:
return "C++2a";
}
}
static bool
maybe_suggest_missing_std_header (location_t location, tree name)
{
gcc_assert (TREE_CODE (name) == IDENTIFIER_NODE);
const char *name_str = IDENTIFIER_POINTER (name);
const std_name_hint *header_hint = get_std_name_hint (name_str);
if (!header_hint)
return false;
gcc_rich_location richloc (location);
if (cxx_dialect >= header_hint->min_dialect)
{
const char *header = header_hint->header;
maybe_add_include_fixit (&richloc, header);
inform (&richloc,
"%<std::%s%> is defined in header %qs;"
" did you forget to %<#include %s%>?",
name_str, header, header);
}
else
{
inform (&richloc,
"%<std::%s%> is only available from %s onwards",
name_str, get_cxx_dialect_name (header_hint->min_dialect));
}
return true;
}
static bool
maybe_suggest_missing_header (location_t location, tree name, tree scope)
{
if (scope == NULL_TREE)
return false;
if (TREE_CODE (scope) != NAMESPACE_DECL)
return false;
if (scope != std_node)
return false;
return maybe_suggest_missing_std_header (location, name);
}
bool
suggest_alternative_in_explicit_scope (location_t location, tree name,
tree scope)
{
if (name == error_mark_node)
return false;
scope = ORIGINAL_NAMESPACE (scope);
if (maybe_suggest_missing_header (location, name, scope))
return true;
cp_binding_level *level = NAMESPACE_LEVEL (scope);
best_match <tree, const char *> bm (name);
consider_binding_level (name, bm, level, false, FUZZY_LOOKUP_NAME);
const char *fuzzy_name = bm.get_best_meaningful_candidate ();
if (fuzzy_name)
{
gcc_rich_location richloc (location);
richloc.add_fixit_replace (fuzzy_name);
inform (&richloc, "suggested alternative: %qs",
fuzzy_name);
return true;
}
return false;
}
tree
lookup_qualified_name (tree scope, tree name, int prefer_type, bool complain,
bool find_hidden)
{
tree t = NULL_TREE;
if (TREE_CODE (scope) == NAMESPACE_DECL)
{
int flags = lookup_flags (prefer_type, false);
if (find_hidden)
flags |= LOOKUP_HIDDEN;
name_lookup lookup (name, flags);
if (qualified_namespace_lookup (scope, &lookup))
t = lookup.value;
}
else if (cxx_dialect != cxx98 && TREE_CODE (scope) == ENUMERAL_TYPE)
t = lookup_enumerator (scope, name);
else if (is_class_type (scope, complain))
t = lookup_member (scope, name, 2, prefer_type, tf_warning_or_error);
if (!t)
return error_mark_node;
return t;
}
static bool
qualified_namespace_lookup (tree scope, name_lookup *lookup)
{
timevar_start (TV_NAME_LOOKUP);
query_oracle (lookup->name);
bool found = lookup->search_qualified (ORIGINAL_NAMESPACE (scope));
timevar_stop (TV_NAME_LOOKUP);
return found;
}
static void
consider_binding_level (tree name, best_match <tree, const char *> &bm,
cp_binding_level *lvl, bool look_within_fields,
enum lookup_name_fuzzy_kind kind)
{
if (look_within_fields)
if (lvl->this_entity && TREE_CODE (lvl->this_entity) == RECORD_TYPE)
{
tree type = lvl->this_entity;
bool want_type_p = (kind == FUZZY_LOOKUP_TYPENAME);
tree best_matching_field
= lookup_member_fuzzy (type, name, want_type_p);
if (best_matching_field)
bm.consider (IDENTIFIER_POINTER (best_matching_field));
}
bool consider_implementation_names = (IDENTIFIER_POINTER (name)[0] == '_');
for (tree t = lvl->names; t; t = TREE_CHAIN (t))
{
tree d = t;
if (TREE_CODE (d) == TREE_LIST)
d = OVL_FIRST (TREE_VALUE (d));
if (TREE_TYPE (d) == error_mark_node)
continue;
if (TREE_CODE (d) == FUNCTION_DECL
&& DECL_BUILT_IN (d)
&& DECL_ANTICIPATED (d))
continue;
if (TREE_CODE (d) == VAR_DECL
&& DECL_ARTIFICIAL (d))
continue;
tree suggestion = DECL_NAME (d);
if (!suggestion)
continue;
if (anon_aggrname_p (suggestion))
continue;
const char *suggestion_str = IDENTIFIER_POINTER (suggestion);
if (strchr (suggestion_str, ' '))
continue;
if (name_reserved_for_implementation_p (suggestion_str)
&& !consider_implementation_names)
continue;
bm.consider (suggestion_str);
}
}
class macro_use_before_def : public deferred_diagnostic
{
public:
static macro_use_before_def *
maybe_make (location_t use_loc, cpp_hashnode *macro)
{
source_location def_loc = cpp_macro_definition_location (macro);
if (def_loc == UNKNOWN_LOCATION)
return NULL;
if (!linemap_location_before_p (line_table, use_loc, def_loc))
return NULL;
return new macro_use_before_def (use_loc, macro);
}
private:
macro_use_before_def (location_t loc, cpp_hashnode *macro)
: deferred_diagnostic (loc), m_macro (macro)
{
gcc_assert (macro);
}
~macro_use_before_def ()
{
if (is_suppressed_p ())
return;
inform (get_location (), "the macro %qs had not yet been defined",
(const char *)m_macro->ident.str);
inform (cpp_macro_definition_location (m_macro),
"it was later defined here");
}
private:
cpp_hashnode *m_macro;
};
static bool
suggest_rid_p  (enum rid rid)
{
switch (rid)
{
case RID_STATIC_ASSERT:
return true;
default:
if (cp_keyword_starts_decl_specifier_p (rid))
return true;
return false;
}
}
name_hint
lookup_name_fuzzy (tree name, enum lookup_name_fuzzy_kind kind, location_t loc)
{
gcc_assert (TREE_CODE (name) == IDENTIFIER_NODE);
const char *header_hint
= get_cp_stdlib_header_for_name (IDENTIFIER_POINTER (name));
if (header_hint)
return name_hint (NULL,
new suggest_missing_header (loc,
IDENTIFIER_POINTER (name),
header_hint));
best_match <tree, const char *> bm (name);
cp_binding_level *lvl;
for (lvl = scope_chain->class_bindings; lvl; lvl = lvl->level_chain)
consider_binding_level (name, bm, lvl, true, kind);
for (lvl = current_binding_level; lvl; lvl = lvl->level_chain)
consider_binding_level (name, bm, lvl, false, kind);
best_macro_match bmm (name, bm.get_best_distance (), parse_in);
cpp_hashnode *best_macro = bmm.get_best_meaningful_candidate ();
if (best_macro)
bm.consider ((const char *)best_macro->ident.str);
else if (bmm.get_best_distance () == 0)
{
cpp_hashnode *macro = bmm.blithely_get_best_candidate ();
if (macro && (macro->flags & NODE_BUILTIN) == 0)
return name_hint (NULL,
macro_use_before_def::maybe_make (loc, macro));
}
for (unsigned i = 0; i < num_c_common_reswords; i++)
{
const c_common_resword *resword = &c_common_reswords[i];
if (!suggest_rid_p (resword->rid))
continue;
tree resword_identifier = ridpointers [resword->rid];
if (!resword_identifier)
continue;
gcc_assert (TREE_CODE (resword_identifier) == IDENTIFIER_NODE);
if (!IDENTIFIER_KEYWORD_P (resword_identifier))
continue;
bm.consider (IDENTIFIER_POINTER (resword_identifier));
}
return name_hint (bm.get_best_meaningful_candidate (), NULL);
}
static bool
binding_to_template_parms_of_scope_p (cxx_binding *binding,
cp_binding_level *scope)
{
tree binding_value, tmpl, tinfo;
int level;
if (!binding || !scope || !scope->this_entity)
return false;
binding_value = binding->value ?  binding->value : binding->type;
tinfo = get_template_info (scope->this_entity);
if (binding_value == NULL_TREE
|| (!DECL_P (binding_value)
|| !DECL_TEMPLATE_PARM_P (binding_value)))
return false;
level =
template_type_parameter_p (binding_value)
? TEMPLATE_PARM_LEVEL (TEMPLATE_TYPE_PARM_INDEX
(TREE_TYPE (binding_value)))
: TEMPLATE_PARM_LEVEL (DECL_INITIAL (binding_value));
tmpl = (tinfo
&& PRIMARY_TEMPLATE_P (TI_TEMPLATE (tinfo))
? TI_TEMPLATE (tinfo)
: NULL_TREE);
return (tmpl && level == TMPL_PARMS_DEPTH (DECL_TEMPLATE_PARMS (tmpl)));
}
cxx_binding *
outer_binding (tree name,
cxx_binding *binding,
bool class_p)
{
cxx_binding *outer;
cp_binding_level *scope;
cp_binding_level *outer_scope;
if (binding)
{
scope = binding->scope->level_chain;
outer = binding->previous;
}
else
{
scope = current_binding_level;
outer = IDENTIFIER_BINDING (name);
}
outer_scope = outer ? outer->scope : NULL;
if (class_p)
while (scope && scope != outer_scope && scope->kind != sk_namespace)
{
if (scope->kind == sk_class)
{
cxx_binding *class_binding;
class_binding = get_class_binding (name, scope);
if (class_binding)
{
class_binding->previous = outer;
if (binding)
binding->previous = class_binding;
else
IDENTIFIER_BINDING (name) = class_binding;
return class_binding;
}
}
if (outer_scope && outer_scope->kind == sk_template_parms
&& binding_to_template_parms_of_scope_p (outer, scope))
return outer;
scope = scope->level_chain;
}
return outer;
}
tree
innermost_non_namespace_value (tree name)
{
cxx_binding *binding;
binding = outer_binding (name, NULL, true);
return binding ? binding->value : NULL_TREE;
}
static tree
lookup_name_real_1 (tree name, int prefer_type, int nonclass, bool block_p,
int namespaces_only, int flags)
{
cxx_binding *iter;
tree val = NULL_TREE;
query_oracle (name);
if (IDENTIFIER_CONV_OP_P (name))
{
cp_binding_level *level;
for (level = current_binding_level;
level && level->kind != sk_namespace;
level = level->level_chain)
{
tree class_type;
tree operators;
if (level->kind != sk_class)
continue;
class_type = level->this_entity;
operators = lookup_fnfields (class_type, name, 0);
if (operators)
return operators;
}
return NULL_TREE;
}
flags |= lookup_flags (prefer_type, namespaces_only);
if (current_class_type == NULL_TREE)
nonclass = 1;
if (block_p || !nonclass)
for (iter = outer_binding (name, NULL, !nonclass);
iter;
iter = outer_binding (name, iter, !nonclass))
{
tree binding;
if (LOCAL_BINDING_P (iter) ? !block_p : nonclass)
continue;
if (qualify_lookup (iter->value, flags))
binding = iter->value;
else if ((flags & LOOKUP_PREFER_TYPES)
&& qualify_lookup (iter->type, flags))
binding = iter->type;
else
binding = NULL_TREE;
if (binding)
{
if (TREE_CODE (binding) == TYPE_DECL && DECL_HIDDEN_P (binding))
{
gcc_assert (TREE_CODE (binding) == TYPE_DECL);
continue;
}
val = binding;
break;
}
}
if (!val)
{
name_lookup lookup (name, flags);
if (lookup.search_unqualified
(current_decl_namespace (), current_binding_level))
val = lookup.value;
}
if (val && TREE_CODE (val) == OVERLOAD && !really_overloaded_fn (val))
val = OVL_FUNCTION (val);
return val;
}
tree
lookup_name_real (tree name, int prefer_type, int nonclass, bool block_p,
int namespaces_only, int flags)
{
tree ret;
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
ret = lookup_name_real_1 (name, prefer_type, nonclass, block_p,
namespaces_only, flags);
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return ret;
}
tree
lookup_name_nonclass (tree name)
{
return lookup_name_real (name, 0, 1, true, 0, 0);
}
tree
lookup_name (tree name)
{
return lookup_name_real (name, 0, 0, true, 0, 0);
}
tree
lookup_name_prefer_type (tree name, int prefer_type)
{
return lookup_name_real (name, prefer_type, 0, true, 0, 0);
}
static tree
lookup_type_scope_1 (tree name, tag_scope scope)
{
cxx_binding *iter = NULL;
tree val = NULL_TREE;
cp_binding_level *level = NULL;
if (current_binding_level->kind != sk_namespace)
iter = outer_binding (name, NULL,  true);
for (; iter; iter = outer_binding (name, iter,  true))
{
if (qualify_lookup (iter->type, LOOKUP_PREFER_TYPES)
&& (scope != ts_current
|| LOCAL_BINDING_P (iter)
|| DECL_CONTEXT (iter->type) == iter->scope->this_entity))
val = iter->type;
else if ((scope != ts_current
|| !INHERITED_VALUE_BINDING_P (iter))
&& qualify_lookup (iter->value, LOOKUP_PREFER_TYPES))
val = iter->value;
if (val)
break;
}
if (val)
level = iter->scope;
else
{
tree ns = current_decl_namespace ();
if (tree *slot = find_namespace_slot (ns, name))
{
if (tree type = MAYBE_STAT_TYPE (*slot))
if (qualify_lookup (type, LOOKUP_PREFER_TYPES))
val = type;
if (!val)
{
if (tree decl = MAYBE_STAT_DECL (*slot))
if (qualify_lookup (decl, LOOKUP_PREFER_TYPES))
val = decl;
}
level = NAMESPACE_LEVEL (ns);
}
}
if (val)
{
cp_binding_level *b = current_binding_level;
while (b)
{
if (level == b)
return val;
if (b->kind == sk_cleanup || b->kind == sk_template_parms
|| b->kind == sk_function_parms)
b = b->level_chain;
else if (b->kind == sk_class
&& scope == ts_within_enclosing_non_class)
b = b->level_chain;
else
break;
}
}
return NULL_TREE;
}
tree
lookup_type_scope (tree name, tag_scope scope)
{
tree ret;
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
ret = lookup_type_scope_1 (name, scope);
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return ret;
}
bool
is_local_extern (tree decl)
{
cxx_binding *binding;
if (TREE_CODE (decl) == FUNCTION_DECL)
return DECL_LOCAL_FUNCTION_P (decl);
if (!VAR_P (decl))
return false;
if (!current_function_decl)
return false;
for (binding = IDENTIFIER_BINDING (DECL_NAME (decl));
binding && binding->scope->kind != sk_namespace;
binding = binding->previous)
if (binding->value == decl)
return LOCAL_BINDING_P (binding);
return false;
}
static tree
maybe_process_template_type_declaration (tree type, int is_friend,
cp_binding_level *b)
{
tree decl = TYPE_NAME (type);
if (processing_template_parmlist)
;
else if (b->kind == sk_namespace
&& current_binding_level->kind != sk_namespace)
;
else
{
gcc_assert (MAYBE_CLASS_TYPE_P (type)
|| TREE_CODE (type) == ENUMERAL_TYPE);
if (processing_template_decl)
{
tree name = DECL_NAME (decl);
decl = push_template_decl_real (decl, is_friend);
if (decl == error_mark_node)
return error_mark_node;
if (TREE_CODE (type) != ENUMERAL_TYPE
&& !is_friend && b->kind == sk_template_parms
&& b->level_chain->kind == sk_class)
{
finish_member_declaration (CLASSTYPE_TI_TEMPLATE (type));
if (!COMPLETE_TYPE_P (current_class_type))
{
maybe_add_class_template_decl_list (current_class_type,
type, 0);
if (CLASSTYPE_NESTED_UTDS (current_class_type) == NULL)
CLASSTYPE_NESTED_UTDS (current_class_type) =
binding_table_new (SCOPE_DEFAULT_HT_SIZE);
binding_table_insert
(CLASSTYPE_NESTED_UTDS (current_class_type), name, type);
}
}
}
}
return decl;
}
static tree
do_pushtag (tree name, tree type, tag_scope scope)
{
tree decl;
cp_binding_level *b = current_binding_level;
while (
b->kind == sk_cleanup
|| b->kind == sk_function_parms
|| (b->kind == sk_template_parms
&& (b->explicit_spec_p || scope == ts_global))
|| (b->kind == sk_class
&& (scope != ts_current
|| COMPLETE_TYPE_P (b->this_entity))))
b = b->level_chain;
gcc_assert (identifier_p (name));
if (identifier_type_value_1 (name) != type)
{
tree tdef;
int in_class = 0;
tree context = TYPE_CONTEXT (type);
if (! context)
{
tree cs = current_scope ();
if (cs && TREE_CODE (cs) == FUNCTION_DECL
&& LAMBDA_TYPE_P (type)
&& !at_function_scope_p ())
cs = DECL_CONTEXT (cs);
if (scope == ts_current
|| (cs && TREE_CODE (cs) == FUNCTION_DECL))
context = cs;
else if (cs && TYPE_P (cs))
context = decl_function_context (get_type_decl (cs));
}
if (!context)
context = current_namespace;
if (b->kind == sk_class
|| (b->kind == sk_template_parms
&& b->level_chain->kind == sk_class))
in_class = 1;
tdef = create_implicit_typedef (name, type);
DECL_CONTEXT (tdef) = FROB_CONTEXT (context);
if (scope == ts_within_enclosing_non_class)
{
retrofit_lang_decl (tdef);
DECL_ANTICIPATED (tdef) = 1;
DECL_FRIEND_P (tdef) = 1;
}
decl = maybe_process_template_type_declaration
(type, scope == ts_within_enclosing_non_class, b);
if (decl == error_mark_node)
return decl;
if (b->kind == sk_class)
{
if (!TYPE_BEING_DEFINED (current_class_type)
&& !LAMBDA_TYPE_P (type))
return error_mark_node;
if (!PROCESSING_REAL_TEMPLATE_DECL_P ())
finish_member_declaration (decl);
else
pushdecl_class_level (decl);
}
else if (b->kind != sk_template_parms)
{
decl = do_pushdecl_with_scope (decl, b, false);
if (decl == error_mark_node)
return decl;
if (DECL_CONTEXT (decl) == std_node
&& init_list_identifier == DECL_NAME (TYPE_NAME (type))
&& !CLASSTYPE_TEMPLATE_INFO (type))
{
error ("declaration of %<std::initializer_list%> does not match "
"%<#include <initializer_list>%>, isn't a template");
return error_mark_node;
}
}
if (! in_class)
set_identifier_type_value_with_scope (name, tdef, b);
TYPE_CONTEXT (type) = DECL_CONTEXT (decl);
if (TYPE_FUNCTION_SCOPE_P (type))
{
if (processing_template_decl)
{
add_decl_expr (decl);
}
else if (!LAMBDA_TYPE_P (type))
vec_safe_push (local_classes, type);
}
}
if (b->kind == sk_class
&& !COMPLETE_TYPE_P (current_class_type))
{
maybe_add_class_template_decl_list (current_class_type,
type, 0);
if (CLASSTYPE_NESTED_UTDS (current_class_type) == NULL)
CLASSTYPE_NESTED_UTDS (current_class_type)
= binding_table_new (SCOPE_DEFAULT_HT_SIZE);
binding_table_insert
(CLASSTYPE_NESTED_UTDS (current_class_type), name, type);
}
decl = TYPE_NAME (type);
gcc_assert (TREE_CODE (decl) == TYPE_DECL);
TREE_PUBLIC (decl) = 1;
determine_visibility (decl);
return type;
}
tree
pushtag (tree name, tree type, tag_scope scope)
{
tree ret;
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
ret = do_pushtag (name, type, scope);
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return ret;
}

struct saved_scope *scope_chain;
static inline bool
store_binding_p (tree id)
{
if (!id || !IDENTIFIER_BINDING (id))
return false;
if (IDENTIFIER_MARKED (id))
return false;
return true;
}
static void
store_binding (tree id, vec<cxx_saved_binding, va_gc> **old_bindings)
{
cxx_saved_binding saved;
gcc_checking_assert (store_binding_p (id));
IDENTIFIER_MARKED (id) = 1;
saved.identifier = id;
saved.binding = IDENTIFIER_BINDING (id);
saved.real_type_value = REAL_IDENTIFIER_TYPE_VALUE (id);
(*old_bindings)->quick_push (saved);
IDENTIFIER_BINDING (id) = NULL;
}
static void
store_bindings (tree names, vec<cxx_saved_binding, va_gc> **old_bindings)
{
static vec<tree> bindings_need_stored;
tree t, id;
size_t i;
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
for (t = names; t; t = TREE_CHAIN (t))
{
if (TREE_CODE (t) == TREE_LIST)
id = TREE_PURPOSE (t);
else
id = DECL_NAME (t);
if (store_binding_p (id))
bindings_need_stored.safe_push (id);
}
if (!bindings_need_stored.is_empty ())
{
vec_safe_reserve_exact (*old_bindings, bindings_need_stored.length ());
for (i = 0; bindings_need_stored.iterate (i, &id); ++i)
{
if (store_binding_p (id))
store_binding (id, old_bindings);
}
bindings_need_stored.truncate (0);
}
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
}
static void
store_class_bindings (vec<cp_class_binding, va_gc> *names,
vec<cxx_saved_binding, va_gc> **old_bindings)
{
static vec<tree> bindings_need_stored;
size_t i;
cp_class_binding *cb;
for (i = 0; vec_safe_iterate (names, i, &cb); ++i)
if (store_binding_p (cb->identifier))
bindings_need_stored.safe_push (cb->identifier);
if (!bindings_need_stored.is_empty ())
{
tree id;
vec_safe_reserve_exact (*old_bindings, bindings_need_stored.length ());
for (i = 0; bindings_need_stored.iterate (i, &id); ++i)
store_binding (id, old_bindings);
bindings_need_stored.truncate (0);
}
}
static GTY((deletable)) struct saved_scope *free_saved_scope;
static void
do_push_to_top_level (void)
{
struct saved_scope *s;
cp_binding_level *b;
cxx_saved_binding *sb;
size_t i;
bool need_pop;
if (free_saved_scope != NULL)
{
s = free_saved_scope;
free_saved_scope = s->prev;
vec<cxx_saved_binding, va_gc> *old_bindings = s->old_bindings;
memset (s, 0, sizeof (*s));
vec_safe_truncate (old_bindings, 0);
s->old_bindings = old_bindings;
}
else
s = ggc_cleared_alloc<saved_scope> ();
b = scope_chain ? current_binding_level : 0;
if (cfun)
{
need_pop = true;
push_function_context ();
}
else
need_pop = false;
if (scope_chain && previous_class_level)
store_class_bindings (previous_class_level->class_shadowed,
&s->old_bindings);
for (; b; b = b->level_chain)
{
tree t;
if (global_scope_p (b))
break;
store_bindings (b->names, &s->old_bindings);
if (b->kind == sk_class)
store_class_bindings (b->class_shadowed, &s->old_bindings);
for (t = b->type_shadowed; t; t = TREE_CHAIN (t))
SET_IDENTIFIER_TYPE_VALUE (TREE_PURPOSE (t), TREE_VALUE (t));
}
FOR_EACH_VEC_SAFE_ELT (s->old_bindings, i, sb)
IDENTIFIER_MARKED (sb->identifier) = 0;
s->prev = scope_chain;
s->bindings = b;
s->need_pop_function_context = need_pop;
s->function_decl = current_function_decl;
s->unevaluated_operand = cp_unevaluated_operand;
s->inhibit_evaluation_warnings = c_inhibit_evaluation_warnings;
s->x_stmt_tree.stmts_are_full_exprs_p = true;
scope_chain = s;
current_function_decl = NULL_TREE;
current_lang_base = NULL;
current_lang_name = lang_name_cplusplus;
current_namespace = global_namespace;
push_class_stack ();
cp_unevaluated_operand = 0;
c_inhibit_evaluation_warnings = 0;
}
static void
do_pop_from_top_level (void)
{
struct saved_scope *s = scope_chain;
cxx_saved_binding *saved;
size_t i;
if (previous_class_level)
invalidate_class_lookup_cache ();
pop_class_stack ();
release_tree_vector (current_lang_base);
scope_chain = s->prev;
FOR_EACH_VEC_SAFE_ELT (s->old_bindings, i, saved)
{
tree id = saved->identifier;
IDENTIFIER_BINDING (id) = saved->binding;
SET_IDENTIFIER_TYPE_VALUE (id, saved->real_type_value);
}
if (s->need_pop_function_context)
pop_function_context ();
current_function_decl = s->function_decl;
cp_unevaluated_operand = s->unevaluated_operand;
c_inhibit_evaluation_warnings = s->inhibit_evaluation_warnings;
s->prev = free_saved_scope;
free_saved_scope = s;
}
static void
do_push_nested_namespace (tree ns)
{
if (ns == global_namespace)
do_push_to_top_level ();
else
{
do_push_nested_namespace (CP_DECL_CONTEXT (ns));
gcc_checking_assert
(find_namespace_value (current_namespace, DECL_NAME (ns)) == ns);
resume_scope (NAMESPACE_LEVEL (ns));
current_namespace = ns;
}
}
static void
do_pop_nested_namespace (tree ns)
{
while (ns != global_namespace)
{
ns = CP_DECL_CONTEXT (ns);
current_namespace = ns;
leave_scope ();
}
do_pop_from_top_level ();
}
static void
add_using_namespace (vec<tree, va_gc> *&usings, tree target)
{
if (usings)
for (unsigned ix = usings->length (); ix--;)
if ((*usings)[ix] == target)
return;
vec_safe_push (usings, target);
}
static void
emit_debug_info_using_namespace (tree from, tree target, bool implicit)
{
tree context = from != global_namespace ? from : NULL_TREE;
debug_hooks->imported_module_or_decl (target, NULL_TREE, context, false,
implicit);
}
void
finish_namespace_using_directive (tree target, tree attribs)
{
gcc_checking_assert (namespace_bindings_p ());
if (target == error_mark_node)
return;
add_using_namespace (DECL_NAMESPACE_USING (current_namespace),
ORIGINAL_NAMESPACE (target));
emit_debug_info_using_namespace (current_namespace,
ORIGINAL_NAMESPACE (target), false);
if (attribs == error_mark_node)
return;
for (tree a = attribs; a; a = TREE_CHAIN (a))
{
tree name = get_attribute_name (a);
if (is_attribute_p ("strong", name))
{
warning (0, "strong using directive no longer supported");
if (CP_DECL_CONTEXT (target) == current_namespace)
inform (DECL_SOURCE_LOCATION (target),
"you may use an inline namespace instead");
}
else
warning (OPT_Wattributes, "%qD attribute directive ignored", name);
}
}
void
finish_local_using_directive (tree target, tree attribs)
{
gcc_checking_assert (local_bindings_p ());
if (target == error_mark_node)
return;
if (attribs)
warning (OPT_Wattributes, "attributes ignored on local using directive");
add_stmt (build_stmt (input_location, USING_STMT, target));
add_using_namespace (current_binding_level->using_directives,
ORIGINAL_NAMESPACE (target));
}
tree
pushdecl_top_level (tree x, bool is_friend)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
do_push_to_top_level ();
x = pushdecl_namespace_level (x, is_friend);
do_pop_from_top_level ();
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return x;
}
tree
pushdecl_top_level_and_finish (tree x, tree init)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
do_push_to_top_level ();
x = pushdecl_namespace_level (x, false);
cp_finish_decl (x, init, false, NULL_TREE, 0);
do_pop_from_top_level ();
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return x;
}
static int
push_inline_namespaces (tree ns)
{
int count = 0;
if (ns != current_namespace)
{
gcc_assert (ns != global_namespace);
count += push_inline_namespaces (CP_DECL_CONTEXT (ns));
resume_scope (NAMESPACE_LEVEL (ns));
current_namespace = ns;
count++;
}
return count;
}
int
push_namespace (tree name, bool make_inline)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
int count = 0;
gcc_checking_assert (global_namespace != NULL && name != global_identifier);
tree ns = NULL_TREE;
{
name_lookup lookup (name, 0);
if (!lookup.search_qualified (current_namespace, false))
;
else if (TREE_CODE (lookup.value) != NAMESPACE_DECL)
;
else if (tree dna = DECL_NAMESPACE_ALIAS (lookup.value))
{
if (is_nested_namespace (current_namespace, CP_DECL_CONTEXT (dna)))
{
error ("namespace alias %qD not allowed here, "
"assuming %qD", lookup.value, dna);
ns = dna;
}
}
else
ns = lookup.value;
}
bool new_ns = false;
if (ns)
count += push_inline_namespaces (CP_DECL_CONTEXT (ns));
else
{
ns = build_lang_decl (NAMESPACE_DECL, name, void_type_node);
SCOPE_DEPTH (ns) = SCOPE_DEPTH (current_namespace) + 1;
if (!SCOPE_DEPTH (ns))
sorry ("cannot nest more than %d namespaces",
SCOPE_DEPTH (current_namespace));
DECL_CONTEXT (ns) = FROB_CONTEXT (current_namespace);
new_ns = true;
if (pushdecl (ns) == error_mark_node)
ns = NULL_TREE;
else
{
if (!name)
{
SET_DECL_ASSEMBLER_NAME (ns, anon_identifier);
if (!make_inline)
add_using_namespace (DECL_NAMESPACE_USING (current_namespace),
ns);
}
else if (TREE_PUBLIC (current_namespace))
TREE_PUBLIC (ns) = 1;
if (make_inline)
{
DECL_NAMESPACE_INLINE_P (ns) = true;
vec_safe_push (DECL_NAMESPACE_INLINEES (current_namespace), ns);
}
if (!name || make_inline)
emit_debug_info_using_namespace (current_namespace, ns, true);
}
}
if (ns)
{
if (make_inline && !DECL_NAMESPACE_INLINE_P (ns))
{
error ("inline namespace must be specified at initial definition");
inform (DECL_SOURCE_LOCATION (ns), "%qD defined here", ns);
}
if (new_ns)
begin_scope (sk_namespace, ns);
else
resume_scope (NAMESPACE_LEVEL (ns));
current_namespace = ns;
count++;
}
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
return count;
}
void
pop_namespace (void)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
gcc_assert (current_namespace != global_namespace);
current_namespace = CP_DECL_CONTEXT (current_namespace);
leave_scope ();
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
}
void
push_to_top_level (void)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
do_push_to_top_level ();
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
}
void
pop_from_top_level (void)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
do_pop_from_top_level ();
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
}
void
push_nested_namespace (tree ns)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
do_push_nested_namespace (ns);
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
}
void
pop_nested_namespace (tree ns)
{
bool subtime = timevar_cond_start (TV_NAME_LOOKUP);
gcc_assert (current_namespace == ns);
do_pop_nested_namespace (ns);
timevar_cond_stop (TV_NAME_LOOKUP, subtime);
}
void
pop_everything (void)
{
if (ENABLE_SCOPE_CHECKING)
verbatim ("XXX entering pop_everything ()\n");
while (!namespace_bindings_p ())
{
if (current_binding_level->kind == sk_class)
pop_nested_class ();
else
poplevel (0, 0, 0);
}
if (ENABLE_SCOPE_CHECKING)
verbatim ("XXX leaving pop_everything ()\n");
}
void
cp_emit_debug_info_for_using (tree t, tree context)
{
if (seen_error ())
return;
if (TREE_CODE (t) == FUNCTION_DECL
&& DECL_EXTERNAL (t)
&& DECL_BUILT_IN (t))
return;
if (context == global_namespace)
context = NULL_TREE;
t = MAYBE_BASELINK_FUNCTIONS (t);
for (lkp_iterator iter (t); iter; ++iter)
{
tree fn = *iter;
if (TREE_CODE (fn) != TEMPLATE_DECL)
{
if (building_stmt_list_p ())
add_stmt (build_stmt (input_location, USING_STMT, fn));
else
debug_hooks->imported_module_or_decl (fn, NULL_TREE, context,
false, false);
}
}
}
#include "gt-cp-name-lookup.h"
