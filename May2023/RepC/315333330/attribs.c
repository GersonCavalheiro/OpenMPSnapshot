#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "tree.h"
#include "stringpool.h"
#include "diagnostic-core.h"
#include "attribs.h"
#include "stor-layout.h"
#include "langhooks.h"
#include "plugin.h"
#include "selftest.h"
#include "hash-set.h"
static const struct attribute_spec *attribute_tables[4];
struct substring
{
const char *str;
int length;
};
static inline hashval_t
substring_hash (const char *str, int l)
{
return str[0] + str[l - 1] * 256 + l * 65536;
}
struct attribute_hasher : nofree_ptr_hash <attribute_spec>
{
typedef substring *compare_type;
static inline hashval_t hash (const attribute_spec *);
static inline bool equal (const attribute_spec *, const substring *);
};
inline hashval_t
attribute_hasher::hash (const attribute_spec *spec)
{
const int l = strlen (spec->name);
return substring_hash (spec->name, l);
}
inline bool
attribute_hasher::equal (const attribute_spec *spec, const substring *str)
{
return (strncmp (spec->name, str->str, str->length) == 0
&& !spec->name[str->length]);
}
struct scoped_attributes
{
const char *ns;
vec<attribute_spec> attributes;
hash_table<attribute_hasher> *attribute_hash;
};
static vec<scoped_attributes> attributes_table;
static scoped_attributes* find_attribute_namespace (const char*);
static void register_scoped_attribute (const struct attribute_spec *,
scoped_attributes *);
static bool attributes_initialized = false;
static const struct attribute_spec empty_attribute_table[] =
{
{ NULL, 0, 0, false, false, false, false, NULL, NULL }
};
static void
extract_attribute_substring (struct substring *str)
{
if (str->length > 4 && str->str[0] == '_' && str->str[1] == '_'
&& str->str[str->length - 1] == '_' && str->str[str->length - 2] == '_')
{
str->length -= 4;
str->str += 2;
}
}
scoped_attributes *
register_scoped_attributes (const struct attribute_spec *attributes,
const char *ns)
{
scoped_attributes *result = NULL;
result = find_attribute_namespace (ns);
if (result == NULL)
{
scoped_attributes sa;
if (attributes_table.is_empty ())
attributes_table.create (64);
memset (&sa, 0, sizeof (sa));
sa.ns = ns;
sa.attributes.create (64);
result = attributes_table.safe_push (sa);
result->attribute_hash = new hash_table<attribute_hasher> (200);
}
for (unsigned i = 0; attributes[i].name != NULL; ++i)
{
result->attributes.safe_push (attributes[i]);
register_scoped_attribute (&attributes[i], result);
}
gcc_assert (result != NULL);
return result;
}
static scoped_attributes*
find_attribute_namespace (const char* ns)
{
unsigned ix;
scoped_attributes *iter;
FOR_EACH_VEC_ELT (attributes_table, ix, iter)
if (ns == iter->ns
|| (iter->ns != NULL
&& ns != NULL
&& !strcmp (iter->ns, ns)))
return iter;
return NULL;
}
static void
check_attribute_tables (void)
{
for (size_t i = 0; i < ARRAY_SIZE (attribute_tables); i++)
for (size_t j = 0; attribute_tables[i][j].name != NULL; j++)
{
const char *name = attribute_tables[i][j].name;
int len = strlen (name);
gcc_assert (!(name[0] == '_' && name[1] == '_'
&& name[len - 1] == '_' && name[len - 2] == '_'));
gcc_assert (attribute_tables[i][j].min_length >= 0);
gcc_assert (attribute_tables[i][j].max_length == -1
|| (attribute_tables[i][j].max_length
>= attribute_tables[i][j].min_length));
gcc_assert (!attribute_tables[i][j].decl_required
|| !attribute_tables[i][j].type_required);
gcc_assert (!attribute_tables[i][j].function_type_required
|| attribute_tables[i][j].type_required);
}
for (size_t i = 0; i < ARRAY_SIZE (attribute_tables); i++)
for (size_t j = 0; attribute_tables[i][j].name != NULL; j++)
for (size_t k = j + 1; attribute_tables[i][k].name != NULL; k++)
gcc_assert (strcmp (attribute_tables[i][j].name,
attribute_tables[i][k].name));
for (size_t i = 0; i < ARRAY_SIZE (attribute_tables); i++)
for (size_t j = i + 1; j < ARRAY_SIZE (attribute_tables); j++)
for (size_t k = 0; attribute_tables[i][k].name != NULL; k++)
for (size_t l = 0; attribute_tables[j][l].name != NULL; l++)
gcc_assert (attribute_tables[i][k].name[0] == '*'
|| strcmp (attribute_tables[i][k].name,
attribute_tables[j][l].name));
}
void
init_attributes (void)
{
size_t i;
if (attributes_initialized)
return;
attribute_tables[0] = lang_hooks.common_attribute_table;
attribute_tables[1] = lang_hooks.attribute_table;
attribute_tables[2] = lang_hooks.format_attribute_table;
attribute_tables[3] = targetm.attribute_table;
for (i = 0; i < ARRAY_SIZE (attribute_tables); i++)
if (attribute_tables[i] == NULL)
attribute_tables[i] = empty_attribute_table;
if (flag_checking)
check_attribute_tables ();
for (i = 0; i < ARRAY_SIZE (attribute_tables); ++i)
register_scoped_attributes (attribute_tables[i], "gnu");
invoke_plugin_callbacks (PLUGIN_ATTRIBUTES, NULL);
attributes_initialized = true;
}
void
register_attribute (const struct attribute_spec *attr)
{
register_scoped_attribute (attr, find_attribute_namespace ("gnu"));
}
static void
register_scoped_attribute (const struct attribute_spec *attr,
scoped_attributes *name_space)
{
struct substring str;
attribute_spec **slot;
gcc_assert (attr != NULL && name_space != NULL);
gcc_assert (name_space->attribute_hash);
str.str = attr->name;
str.length = strlen (str.str);
gcc_assert (str.length > 0 && str.str[0] != '_');
slot = name_space->attribute_hash
->find_slot_with_hash (&str, substring_hash (str.str, str.length),
INSERT);
gcc_assert (!*slot || attr->name[0] == '*');
*slot = CONST_CAST (struct attribute_spec *, attr);
}
static const struct attribute_spec *
lookup_scoped_attribute_spec (const_tree ns, const_tree name)
{
struct substring attr;
scoped_attributes *attrs;
const char *ns_str = (ns != NULL_TREE) ? IDENTIFIER_POINTER (ns): NULL;
attrs = find_attribute_namespace (ns_str);
if (attrs == NULL)
return NULL;
attr.str = IDENTIFIER_POINTER (name);
attr.length = IDENTIFIER_LENGTH (name);
extract_attribute_substring (&attr);
return attrs->attribute_hash->find_with_hash (&attr,
substring_hash (attr.str,
attr.length));
}
const struct attribute_spec *
lookup_attribute_spec (const_tree name)
{
tree ns;
if (TREE_CODE (name) == TREE_LIST)
{
ns = TREE_PURPOSE (name);
name = TREE_VALUE (name);
}
else
ns = get_identifier ("gnu");
return lookup_scoped_attribute_spec (ns, name);
}
static tree
get_attribute_namespace (const_tree attr)
{
if (cxx11_attribute_p (attr))
return TREE_PURPOSE (TREE_PURPOSE (attr));
return get_identifier ("gnu");
}
static bool
diag_attr_exclusions (tree last_decl, tree node, tree attrname,
const attribute_spec *spec)
{
const attribute_spec::exclusions *excl = spec->exclude;
tree_code code = TREE_CODE (node);
if ((code == FUNCTION_DECL && !excl->function
&& (!excl->type || !spec->affects_type_identity))
|| (code == VAR_DECL && !excl->variable
&& (!excl->type || !spec->affects_type_identity))
|| (((code == TYPE_DECL || RECORD_OR_UNION_TYPE_P (node)) && !excl->type)))
return false;
bool found = false;
if (last_decl && last_decl != node && TREE_TYPE (last_decl) != node)
{
found |= diag_attr_exclusions (last_decl, last_decl, attrname, spec);
tree decl_type = TREE_TYPE (last_decl);
found |= diag_attr_exclusions (last_decl, decl_type, attrname, spec);
}
tree attrs[2];
if (DECL_P (node))
{
attrs[0] = DECL_ATTRIBUTES (node);
attrs[1] = TYPE_ATTRIBUTES (TREE_TYPE (node));
}
else
{
attrs[0] = TYPE_ATTRIBUTES (node);
attrs[1] = NULL_TREE;
}
for (unsigned i = 0; i != sizeof attrs / sizeof *attrs; ++i)
{
if (!attrs[i])
continue;
for ( ; excl->name; ++excl)
{
if (is_attribute_p (excl->name, attrname))
continue;
if (!lookup_attribute (excl->name, attrs[i]))
continue;
if (TREE_CODE (node) == FUNCTION_DECL
&& !excl->function)
continue;
if (TREE_CODE (node) == TYPE_DECL
&& !excl->type)
continue;
if ((TREE_CODE (node) == FIELD_DECL
|| TREE_CODE (node) == VAR_DECL)
&& !excl->variable)
continue;
found = true;
bool note = last_decl != NULL_TREE;
if (TREE_CODE (node) == FUNCTION_DECL
&& DECL_BUILT_IN (node))
note &= warning (OPT_Wattributes,
"ignoring attribute %qE in declaration of "
"a built-in function %qD because it conflicts "
"with attribute %qs",
attrname, node, excl->name);
else
note &= warning (OPT_Wattributes,
"ignoring attribute %qE because "
"it conflicts with attribute %qs",
attrname, excl->name);
if (note)
inform (DECL_SOURCE_LOCATION (last_decl),
"previous declaration here");
}
}
return found;
}
tree
decl_attributes (tree *node, tree attributes, int flags,
tree last_decl )
{
tree a;
tree returned_attrs = NULL_TREE;
if (TREE_TYPE (*node) == error_mark_node || attributes == error_mark_node)
return NULL_TREE;
if (!attributes_initialized)
init_attributes ();
if (TREE_CODE (*node) == FUNCTION_DECL && current_optimize_pragma)
{
tree cur_attr = lookup_attribute ("optimize", attributes);
tree opts = copy_list (current_optimize_pragma);
if (! cur_attr)
attributes
= tree_cons (get_identifier ("optimize"), opts, attributes);
else
TREE_VALUE (cur_attr) = chainon (opts, TREE_VALUE (cur_attr));
}
if (TREE_CODE (*node) == FUNCTION_DECL
&& optimization_current_node != optimization_default_node
&& !DECL_FUNCTION_SPECIFIC_OPTIMIZATION (*node))
DECL_FUNCTION_SPECIFIC_OPTIMIZATION (*node) = optimization_current_node;
if (TREE_CODE (*node) == FUNCTION_DECL
&& current_target_pragma
&& targetm.target_option.valid_attribute_p (*node, NULL_TREE,
current_target_pragma, 0))
{
tree cur_attr = lookup_attribute ("target", attributes);
tree opts = copy_list (current_target_pragma);
if (! cur_attr)
attributes = tree_cons (get_identifier ("target"), opts, attributes);
else
TREE_VALUE (cur_attr) = chainon (opts, TREE_VALUE (cur_attr));
}
if (TREE_CODE (*node) == FUNCTION_DECL
&& attributes
&& lookup_attribute ("naked", attributes) != NULL
&& lookup_attribute_spec (get_identifier ("naked")))
{
if (lookup_attribute ("noinline", attributes) == NULL)
attributes = tree_cons (get_identifier ("noinline"), NULL, attributes);
if (lookup_attribute ("noclone", attributes) == NULL)
attributes = tree_cons (get_identifier ("noclone"),  NULL, attributes);
}
if (TREE_CODE (*node) == FUNCTION_DECL
&& attributes
&& lookup_attribute ("noipa", attributes) != NULL
&& lookup_attribute_spec (get_identifier ("noipa")))
{
if (lookup_attribute ("noinline", attributes) == NULL)
attributes = tree_cons (get_identifier ("noinline"), NULL, attributes);
if (lookup_attribute ("noclone", attributes) == NULL)
attributes = tree_cons (get_identifier ("noclone"),  NULL, attributes);
if (lookup_attribute ("no_icf", attributes) == NULL)
attributes = tree_cons (get_identifier ("no_icf"),  NULL, attributes);
}
targetm.insert_attributes (*node, &attributes);
for (a = attributes; a; a = TREE_CHAIN (a))
{
tree ns = get_attribute_namespace (a);
tree name = get_attribute_name (a);
tree args = TREE_VALUE (a);
tree *anode = node;
const struct attribute_spec *spec
= lookup_scoped_attribute_spec (ns, name);
int fn_ptr_quals = 0;
tree fn_ptr_tmp = NULL_TREE;
if (spec == NULL)
{
if (!(flags & (int) ATTR_FLAG_BUILT_IN))
{
if (ns == NULL_TREE || !cxx11_attribute_p (a))
warning (OPT_Wattributes, "%qE attribute directive ignored",
name);
else
warning (OPT_Wattributes,
"%<%E::%E%> scoped attribute directive ignored",
ns, name);
}
continue;
}
else if (list_length (args) < spec->min_length
|| (spec->max_length >= 0
&& list_length (args) > spec->max_length))
{
error ("wrong number of arguments specified for %qE attribute",
name);
continue;
}
gcc_assert (is_attribute_p (spec->name, name));
if (TYPE_P (*node)
&& cxx11_attribute_p (a)
&& !(flags & ATTR_FLAG_TYPE_IN_PLACE))
{
if (warning (OPT_Wattributes, "attribute ignored"))
inform (input_location,
"an attribute that appertains to a type-specifier "
"is ignored");
continue;
}
if (spec->decl_required && !DECL_P (*anode))
{
if (flags & ((int) ATTR_FLAG_DECL_NEXT
| (int) ATTR_FLAG_FUNCTION_NEXT
| (int) ATTR_FLAG_ARRAY_NEXT))
{
tree attr = tree_cons (name, args, NULL_TREE);
returned_attrs = chainon (returned_attrs, attr);
continue;
}
else
{
warning (OPT_Wattributes, "%qE attribute does not apply to types",
name);
continue;
}
}
if (spec->type_required && DECL_P (*anode))
{
anode = &TREE_TYPE (*anode);
flags &= ~(int) ATTR_FLAG_TYPE_IN_PLACE;
}
if (spec->function_type_required && TREE_CODE (*anode) != FUNCTION_TYPE
&& TREE_CODE (*anode) != METHOD_TYPE)
{
if (TREE_CODE (*anode) == POINTER_TYPE
&& (TREE_CODE (TREE_TYPE (*anode)) == FUNCTION_TYPE
|| TREE_CODE (TREE_TYPE (*anode)) == METHOD_TYPE))
{
fn_ptr_tmp = TREE_TYPE (*anode);
fn_ptr_quals = TYPE_QUALS (*anode);
anode = &fn_ptr_tmp;
flags &= ~(int) ATTR_FLAG_TYPE_IN_PLACE;
}
else if (flags & (int) ATTR_FLAG_FUNCTION_NEXT)
{
tree attr = tree_cons (name, args, NULL_TREE);
returned_attrs = chainon (returned_attrs, attr);
continue;
}
if (TREE_CODE (*anode) != FUNCTION_TYPE
&& TREE_CODE (*anode) != METHOD_TYPE)
{
warning (OPT_Wattributes,
"%qE attribute only applies to function types",
name);
continue;
}
}
if (TYPE_P (*anode)
&& (flags & (int) ATTR_FLAG_TYPE_IN_PLACE)
&& TYPE_SIZE (*anode) != NULL_TREE)
{
warning (OPT_Wattributes, "type attributes ignored after type is already defined");
continue;
}
bool no_add_attrs = false;
if (spec->handler != NULL)
{
int cxx11_flag =
cxx11_attribute_p (a) ? ATTR_FLAG_CXX11 : 0;
tree cur_and_last_decl[] = { *anode, last_decl };
tree ret = (spec->handler) (cur_and_last_decl, name, args,
flags|cxx11_flag, &no_add_attrs);
*anode = cur_and_last_decl[0];
if (ret == error_mark_node)
{
warning (OPT_Wattributes, "%qE attribute ignored", name);
no_add_attrs = true;
}
else
returned_attrs = chainon (ret, returned_attrs);
}
bool built_in = flags & ATTR_FLAG_BUILT_IN;
if (spec->exclude
&& !no_add_attrs
&& (flag_checking || !built_in))
{
if (!built_in
|| !DECL_P (*anode)
|| (DECL_FUNCTION_CODE (*anode) != BUILT_IN_UNREACHABLE
&& (DECL_FUNCTION_CODE (*anode)
!= BUILT_IN_UBSAN_HANDLE_BUILTIN_UNREACHABLE)))
{
bool no_add = diag_attr_exclusions (last_decl, *anode, name, spec);
if (!no_add && anode != node)
no_add = diag_attr_exclusions (last_decl, *node, name, spec);
no_add_attrs |= no_add;
}
}
if (spec->type_required && DECL_P (*node)
&& (VAR_P (*node)
|| TREE_CODE (*node) == PARM_DECL
|| TREE_CODE (*node) == RESULT_DECL))
relayout_decl (*node);
if (!no_add_attrs)
{
tree old_attrs;
tree a;
if (DECL_P (*anode))
old_attrs = DECL_ATTRIBUTES (*anode);
else
old_attrs = TYPE_ATTRIBUTES (*anode);
for (a = lookup_attribute (spec->name, old_attrs);
a != NULL_TREE;
a = lookup_attribute (spec->name, TREE_CHAIN (a)))
{
if (simple_cst_equal (TREE_VALUE (a), args) == 1)
break;
}
if (a == NULL_TREE)
{
if (DECL_P (*anode))
DECL_ATTRIBUTES (*anode) = tree_cons (name, args, old_attrs);
else if (flags & (int) ATTR_FLAG_TYPE_IN_PLACE)
{
TYPE_ATTRIBUTES (*anode) = tree_cons (name, args, old_attrs);
if (*anode == TYPE_MAIN_VARIANT (*anode))
{
tree variant;
for (variant = *anode; variant;
variant = TYPE_NEXT_VARIANT (variant))
{
if (TYPE_ATTRIBUTES (variant) == old_attrs)
TYPE_ATTRIBUTES (variant)
= TYPE_ATTRIBUTES (*anode);
else if (!lookup_attribute
(spec->name, TYPE_ATTRIBUTES (variant)))
TYPE_ATTRIBUTES (variant) = tree_cons
(name, args, TYPE_ATTRIBUTES (variant));
}
}
}
else
*anode = build_type_attribute_variant (*anode,
tree_cons (name, args,
old_attrs));
}
}
if (fn_ptr_tmp)
{
fn_ptr_tmp = build_pointer_type (fn_ptr_tmp);
if (fn_ptr_quals)
fn_ptr_tmp = build_qualified_type (fn_ptr_tmp, fn_ptr_quals);
if (DECL_P (*node))
TREE_TYPE (*node) = fn_ptr_tmp;
else
{
gcc_assert (TREE_CODE (*node) == POINTER_TYPE);
*node = fn_ptr_tmp;
}
}
}
return returned_attrs;
}
bool
cxx11_attribute_p (const_tree attr)
{
if (attr == NULL_TREE
|| TREE_CODE (attr) != TREE_LIST)
return false;
return (TREE_CODE (TREE_PURPOSE (attr)) == TREE_LIST);
}
tree
get_attribute_name (const_tree attr)
{
if (cxx11_attribute_p (attr))
return TREE_VALUE (TREE_PURPOSE (attr));
return TREE_PURPOSE (attr);
}
void
apply_tm_attr (tree fndecl, tree attr)
{
decl_attributes (&TREE_TYPE (fndecl), tree_cons (attr, NULL, NULL), 0);
}
tree
make_attribute (const char *name, const char *arg_name, tree chain)
{
tree attr_name;
tree attr_arg_name;
tree attr_args;
tree attr;
attr_name = get_identifier (name);
attr_arg_name = build_string (strlen (arg_name), arg_name);
attr_args = tree_cons (NULL_TREE, attr_arg_name, NULL_TREE);
attr = tree_cons (attr_name, attr_args, chain);
return attr;
}

static int
attr_strcmp (const void *v1, const void *v2)
{
const char *c1 = *(char *const*)v1;
const char *c2 = *(char *const*)v2;
return strcmp (c1, c2);
}
char *
sorted_attr_string (tree arglist)
{
tree arg;
size_t str_len_sum = 0;
char **args = NULL;
char *attr_str, *ret_str;
char *attr = NULL;
unsigned int argnum = 1;
unsigned int i;
for (arg = arglist; arg; arg = TREE_CHAIN (arg))
{
const char *str = TREE_STRING_POINTER (TREE_VALUE (arg));
size_t len = strlen (str);
str_len_sum += len + 1;
if (arg != arglist)
argnum++;
for (i = 0; i < strlen (str); i++)
if (str[i] == ',')
argnum++;
}
attr_str = XNEWVEC (char, str_len_sum);
str_len_sum = 0;
for (arg = arglist; arg; arg = TREE_CHAIN (arg))
{
const char *str = TREE_STRING_POINTER (TREE_VALUE (arg));
size_t len = strlen (str);
memcpy (attr_str + str_len_sum, str, len);
attr_str[str_len_sum + len] = TREE_CHAIN (arg) ? ',' : '\0';
str_len_sum += len + 1;
}
for (i = 0; i < strlen (attr_str); i++)
if (attr_str[i] == '=' || attr_str[i]== '-')
attr_str[i] = '_';
if (argnum == 1)
return attr_str;
args = XNEWVEC (char *, argnum);
i = 0;
attr = strtok (attr_str, ",");
while (attr != NULL)
{
args[i] = attr;
i++;
attr = strtok (NULL, ",");
}
qsort (args, argnum, sizeof (char *), attr_strcmp);
ret_str = XNEWVEC (char, str_len_sum);
str_len_sum = 0;
for (i = 0; i < argnum; i++)
{
size_t len = strlen (args[i]);
memcpy (ret_str + str_len_sum, args[i], len);
ret_str[str_len_sum + len] = i < argnum - 1 ? '_' : '\0';
str_len_sum += len + 1;
}
XDELETEVEC (args);
XDELETEVEC (attr_str);
return ret_str;
}
bool
common_function_versions (tree fn1, tree fn2)
{
tree attr1, attr2;
char *target1, *target2;
bool result;
if (TREE_CODE (fn1) != FUNCTION_DECL
|| TREE_CODE (fn2) != FUNCTION_DECL)
return false;
attr1 = lookup_attribute ("target", DECL_ATTRIBUTES (fn1));
attr2 = lookup_attribute ("target", DECL_ATTRIBUTES (fn2));
if (attr1 == NULL_TREE && attr2 == NULL_TREE)
return false;
if (attr1 == NULL_TREE || attr2 == NULL_TREE)
{
if (DECL_FUNCTION_VERSIONED (fn1) || DECL_FUNCTION_VERSIONED (fn2))
{
if (attr2 != NULL_TREE)
{
std::swap (fn1, fn2);
attr1 = attr2;
}
error_at (DECL_SOURCE_LOCATION (fn2),
"missing %<target%> attribute for multi-versioned %qD",
fn2);
inform (DECL_SOURCE_LOCATION (fn1),
"previous declaration of %qD", fn1);
DECL_ATTRIBUTES (fn2)
= tree_cons (get_identifier ("target"),
copy_node (TREE_VALUE (attr1)),
DECL_ATTRIBUTES (fn2));
}
return false;
}
target1 = sorted_attr_string (TREE_VALUE (attr1));
target2 = sorted_attr_string (TREE_VALUE (attr2));
if (strcmp (target1, target2) == 0)
result = false;
else
result = true;
XDELETEVEC (target1);
XDELETEVEC (target2);
return result;
}
char *
make_unique_name (tree decl, const char *suffix, bool make_unique)
{
char *global_var_name;
int name_len;
const char *name;
const char *unique_name = NULL;
name = IDENTIFIER_POINTER (DECL_ASSEMBLER_NAME (decl));
if (make_unique)
unique_name = IDENTIFIER_POINTER (get_file_function_name ("\0"));
name_len = strlen (name) + strlen (suffix) + 2;
if (make_unique)
name_len += strlen (unique_name) + 1;
global_var_name = XNEWVEC (char, name_len);
if (make_unique)
snprintf (global_var_name, name_len, "%s.%s.%s", name, unique_name,
suffix);
else
snprintf (global_var_name, name_len, "%s.%s", name, suffix);
return global_var_name;
}
tree
make_dispatcher_decl (const tree decl)
{
tree func_decl;
char *func_name;
tree fn_type, func_type;
func_name = xstrdup (IDENTIFIER_POINTER (DECL_ASSEMBLER_NAME (decl)));
fn_type = TREE_TYPE (decl);
func_type = build_function_type (TREE_TYPE (fn_type),
TYPE_ARG_TYPES (fn_type));
func_decl = build_fn_decl (func_name, func_type);
XDELETEVEC (func_name);
TREE_USED (func_decl) = 1;
DECL_CONTEXT (func_decl) = NULL_TREE;
DECL_INITIAL (func_decl) = error_mark_node;
DECL_ARTIFICIAL (func_decl) = 1;
DECL_EXTERNAL (func_decl) = 1;
TREE_PUBLIC (func_decl) = 1;
return func_decl;  
}
bool
is_function_default_version (const tree decl)
{
if (TREE_CODE (decl) != FUNCTION_DECL
|| !DECL_FUNCTION_VERSIONED (decl))
return false;
tree attr = lookup_attribute ("target", DECL_ATTRIBUTES (decl));
gcc_assert (attr);
attr = TREE_VALUE (TREE_VALUE (attr));
return (TREE_CODE (attr) == STRING_CST
&& strcmp (TREE_STRING_POINTER (attr), "default") == 0);
}
tree
build_decl_attribute_variant (tree ddecl, tree attribute)
{
DECL_ATTRIBUTES (ddecl) = attribute;
return ddecl;
}
tree
build_type_attribute_qual_variant (tree otype, tree attribute, int quals)
{
tree ttype = otype;
if (! attribute_list_equal (TYPE_ATTRIBUTES (ttype), attribute))
{
tree ntype;
if (TREE_CODE (ttype) == RECORD_TYPE
|| TREE_CODE (ttype) == UNION_TYPE
|| TREE_CODE (ttype) == QUAL_UNION_TYPE
|| TREE_CODE (ttype) == ENUMERAL_TYPE)
{
warning (OPT_Wattributes,
"ignoring attributes applied to %qT after definition",
TYPE_MAIN_VARIANT (ttype));
return build_qualified_type (ttype, quals);
}
ttype = build_qualified_type (ttype, TYPE_UNQUALIFIED);
if (lang_hooks.types.copy_lang_qualifiers
&& otype != TYPE_MAIN_VARIANT (otype))
ttype = (lang_hooks.types.copy_lang_qualifiers
(ttype, TYPE_MAIN_VARIANT (otype)));
tree dtype = ntype = build_distinct_type_copy (ttype);
TYPE_ATTRIBUTES (ntype) = attribute;
hashval_t hash = type_hash_canon_hash (ntype);
ntype = type_hash_canon (hash, ntype);
if (ntype != dtype)
;
else if (TYPE_STRUCTURAL_EQUALITY_P (ttype)
|| !comp_type_attributes (ntype, ttype))
SET_TYPE_STRUCTURAL_EQUALITY (ntype);
else if (TYPE_CANONICAL (ntype) == ntype)
TYPE_CANONICAL (ntype) = TYPE_CANONICAL (ttype);
ttype = build_qualified_type (ntype, quals);
if (lang_hooks.types.copy_lang_qualifiers
&& otype != TYPE_MAIN_VARIANT (otype))
ttype = lang_hooks.types.copy_lang_qualifiers (ttype, otype);
}
else if (TYPE_QUALS (ttype) != quals)
ttype = build_qualified_type (ttype, quals);
return ttype;
}
static bool
cmp_attrib_identifiers (const_tree attr1, const_tree attr2)
{
gcc_checking_assert (TREE_CODE (attr1) == IDENTIFIER_NODE
&& TREE_CODE (attr2) == IDENTIFIER_NODE);
if (attr1 == attr2)
return true;
return cmp_attribs (IDENTIFIER_POINTER (attr1), IDENTIFIER_LENGTH (attr1),
IDENTIFIER_POINTER (attr2), IDENTIFIER_LENGTH (attr2));
}
static bool
simple_cst_list_equal (const_tree l1, const_tree l2)
{
while (l1 != NULL_TREE && l2 != NULL_TREE)
{
if (simple_cst_equal (TREE_VALUE (l1), TREE_VALUE (l2)) != 1)
return false;
l1 = TREE_CHAIN (l1);
l2 = TREE_CHAIN (l2);
}
return l1 == l2;
}
static bool
omp_declare_simd_clauses_equal (tree clauses1, tree clauses2)
{
tree cl1, cl2;
for (cl1 = clauses1, cl2 = clauses2;
cl1 && cl2;
cl1 = OMP_CLAUSE_CHAIN (cl1), cl2 = OMP_CLAUSE_CHAIN (cl2))
{
if (OMP_CLAUSE_CODE (cl1) != OMP_CLAUSE_CODE (cl2))
return false;
if (OMP_CLAUSE_CODE (cl1) != OMP_CLAUSE_SIMDLEN)
{
if (simple_cst_equal (OMP_CLAUSE_DECL (cl1),
OMP_CLAUSE_DECL (cl2)) != 1)
return false;
}
switch (OMP_CLAUSE_CODE (cl1))
{
case OMP_CLAUSE_ALIGNED:
if (simple_cst_equal (OMP_CLAUSE_ALIGNED_ALIGNMENT (cl1),
OMP_CLAUSE_ALIGNED_ALIGNMENT (cl2)) != 1)
return false;
break;
case OMP_CLAUSE_LINEAR:
if (simple_cst_equal (OMP_CLAUSE_LINEAR_STEP (cl1),
OMP_CLAUSE_LINEAR_STEP (cl2)) != 1)
return false;
break;
case OMP_CLAUSE_SIMDLEN:
if (simple_cst_equal (OMP_CLAUSE_SIMDLEN_EXPR (cl1),
OMP_CLAUSE_SIMDLEN_EXPR (cl2)) != 1)
return false;
default:
break;
}
}
return true;
}
bool
attribute_value_equal (const_tree attr1, const_tree attr2)
{
if (TREE_VALUE (attr1) == TREE_VALUE (attr2))
return true;
if (TREE_VALUE (attr1) != NULL_TREE
&& TREE_CODE (TREE_VALUE (attr1)) == TREE_LIST
&& TREE_VALUE (attr2) != NULL_TREE
&& TREE_CODE (TREE_VALUE (attr2)) == TREE_LIST)
{
if (is_attribute_p ("format", get_attribute_name (attr1)))
{
attr1 = TREE_VALUE (attr1);
attr2 = TREE_VALUE (attr2);
if (!cmp_attrib_identifiers (TREE_VALUE (attr1), TREE_VALUE (attr2)))
return false;
return (simple_cst_list_equal (TREE_CHAIN (attr1),
TREE_CHAIN (attr2)) == 1);
}
return (simple_cst_list_equal (TREE_VALUE (attr1),
TREE_VALUE (attr2)) == 1);
}
if (TREE_VALUE (attr1)
&& TREE_CODE (TREE_VALUE (attr1)) == OMP_CLAUSE
&& TREE_VALUE (attr2)
&& TREE_CODE (TREE_VALUE (attr2)) == OMP_CLAUSE)
return omp_declare_simd_clauses_equal (TREE_VALUE (attr1),
TREE_VALUE (attr2));
return (simple_cst_equal (TREE_VALUE (attr1), TREE_VALUE (attr2)) == 1);
}
int
comp_type_attributes (const_tree type1, const_tree type2)
{
const_tree a1 = TYPE_ATTRIBUTES (type1);
const_tree a2 = TYPE_ATTRIBUTES (type2);
const_tree a;
if (a1 == a2)
return 1;
for (a = a1; a != NULL_TREE; a = TREE_CHAIN (a))
{
const struct attribute_spec *as;
const_tree attr;
as = lookup_attribute_spec (get_attribute_name (a));
if (!as || as->affects_type_identity == false)
continue;
attr = lookup_attribute (as->name, CONST_CAST_TREE (a2));
if (!attr || !attribute_value_equal (a, attr))
break;
}
if (!a)
{
for (a = a2; a != NULL_TREE; a = TREE_CHAIN (a))
{
const struct attribute_spec *as;
as = lookup_attribute_spec (get_attribute_name (a));
if (!as || as->affects_type_identity == false)
continue;
if (!lookup_attribute (as->name, CONST_CAST_TREE (a1)))
break;
}
if (!a)
return 1;
}
if (lookup_attribute ("transaction_safe", CONST_CAST_TREE (a)))
return 0;
if ((lookup_attribute ("nocf_check", TYPE_ATTRIBUTES (type1)) != NULL)
^ (lookup_attribute ("nocf_check", TYPE_ATTRIBUTES (type2)) != NULL))
return 0;
return targetm.comp_type_attributes (type1, type2);
}
tree
build_type_attribute_variant (tree ttype, tree attribute)
{
return build_type_attribute_qual_variant (ttype, attribute,
TYPE_QUALS (ttype));
}

static tree
lookup_ident_attribute (tree attr_identifier, tree list)
{
gcc_checking_assert (TREE_CODE (attr_identifier) == IDENTIFIER_NODE);
while (list)
{
gcc_checking_assert (TREE_CODE (get_attribute_name (list))
== IDENTIFIER_NODE);
if (cmp_attrib_identifiers (attr_identifier,
get_attribute_name (list)))
break;
list = TREE_CHAIN (list);
}
return list;
}
tree
remove_attribute (const char *attr_name, tree list)
{
tree *p;
gcc_checking_assert (attr_name[0] != '_');
for (p = &list; *p;)
{
tree l = *p;
tree attr = get_attribute_name (l);
if (is_attribute_p (attr_name, attr))
*p = TREE_CHAIN (l);
else
p = &TREE_CHAIN (l);
}
return list;
}
tree
merge_attributes (tree a1, tree a2)
{
tree attributes;
if ((attributes = a1) == 0)
attributes = a2;
else if (a2 != 0 && ! attribute_list_contained (a1, a2))
{
if (attribute_list_contained (a2, a1))
attributes = a2;
else
{
if (list_length (a1) < list_length (a2))
attributes = a2, a2 = a1;
for (; a2 != 0; a2 = TREE_CHAIN (a2))
{
tree a;
for (a = lookup_ident_attribute (get_attribute_name (a2),
attributes);
a != NULL_TREE && !attribute_value_equal (a, a2);
a = lookup_ident_attribute (get_attribute_name (a2),
TREE_CHAIN (a)))
;
if (a == NULL_TREE)
{
a1 = copy_node (a2);
TREE_CHAIN (a1) = attributes;
attributes = a1;
}
}
}
}
return attributes;
}
tree
merge_type_attributes (tree t1, tree t2)
{
return merge_attributes (TYPE_ATTRIBUTES (t1),
TYPE_ATTRIBUTES (t2));
}
tree
merge_decl_attributes (tree olddecl, tree newdecl)
{
return merge_attributes (DECL_ATTRIBUTES (olddecl),
DECL_ATTRIBUTES (newdecl));
}
void
duplicate_one_attribute (tree *attrs, tree attr, const char *name)
{
attr = lookup_attribute (name, attr);
if (!attr)
return;
tree a = lookup_attribute (name, *attrs);
while (attr)
{
tree a2;
for (a2 = a; a2; a2 = lookup_attribute (name, TREE_CHAIN (a2)))
if (attribute_value_equal (attr, a2))
break;
if (!a2)
{
a2 = copy_node (attr);
TREE_CHAIN (a2) = *attrs;
*attrs = a2;
}
attr = lookup_attribute (name, TREE_CHAIN (attr));
}
}
void
copy_attributes_to_builtin (tree decl)
{
tree b = builtin_decl_explicit (DECL_FUNCTION_CODE (decl));
if (b)
duplicate_one_attribute (&DECL_ATTRIBUTES (b),
DECL_ATTRIBUTES (decl), "omp declare simd");
}
#if TARGET_DLLIMPORT_DECL_ATTRIBUTES
tree
merge_dllimport_decl_attributes (tree old, tree new_tree)
{
tree a;
int delete_dllimport_p = 1;
if (!VAR_OR_FUNCTION_DECL_P (new_tree))
delete_dllimport_p = 0;
else if (DECL_DLLIMPORT_P (new_tree)
&& lookup_attribute ("dllexport", DECL_ATTRIBUTES (old)))
{
DECL_DLLIMPORT_P (new_tree) = 0;
warning (OPT_Wattributes, "%q+D already declared with dllexport "
"attribute: dllimport ignored", new_tree);
}
else if (DECL_DLLIMPORT_P (old) && !DECL_DLLIMPORT_P (new_tree))
{
if (TREE_USED (old))
{
warning (0, "%q+D redeclared without dllimport attribute "
"after being referenced with dll linkage", new_tree);
if (VAR_P (old) && TREE_ADDRESSABLE (old))
DECL_DLLIMPORT_P (new_tree) = 1;
}
else if (VAR_P (new_tree) || !DECL_DECLARED_INLINE_P (new_tree))
warning (OPT_Wattributes, "%q+D redeclared without dllimport "
"attribute: previous dllimport ignored", new_tree);
}
else
delete_dllimport_p = 0;
a = merge_attributes (DECL_ATTRIBUTES (old), DECL_ATTRIBUTES (new_tree));
if (delete_dllimport_p)
a = remove_attribute ("dllimport", a);
return a;
}
tree
handle_dll_attribute (tree * pnode, tree name, tree args, int flags,
bool *no_add_attrs)
{
tree node = *pnode;
bool is_dllimport;
if (!DECL_P (node))
{
if (flags & ((int) ATTR_FLAG_DECL_NEXT | (int) ATTR_FLAG_FUNCTION_NEXT
| (int) ATTR_FLAG_ARRAY_NEXT))
{
*no_add_attrs = true;
return tree_cons (name, args, NULL_TREE);
}
if (TREE_CODE (node) == RECORD_TYPE
|| TREE_CODE (node) == UNION_TYPE)
{
node = TYPE_NAME (node);
if (!node)
return NULL_TREE;
}
else
{
warning (OPT_Wattributes, "%qE attribute ignored",
name);
*no_add_attrs = true;
return NULL_TREE;
}
}
if (!VAR_OR_FUNCTION_DECL_P (node) && TREE_CODE (node) != TYPE_DECL)
{
*no_add_attrs = true;
warning (OPT_Wattributes, "%qE attribute ignored",
name);
return NULL_TREE;
}
if (TREE_CODE (node) == TYPE_DECL
&& TREE_CODE (TREE_TYPE (node)) != RECORD_TYPE
&& TREE_CODE (TREE_TYPE (node)) != UNION_TYPE)
{
*no_add_attrs = true;
warning (OPT_Wattributes, "%qE attribute ignored",
name);
return NULL_TREE;
}
is_dllimport = is_attribute_p ("dllimport", name);
if (is_dllimport)
{
if (!targetm.valid_dllimport_attribute_p (node))
*no_add_attrs = true;
else if (TREE_CODE (node) == FUNCTION_DECL
&& DECL_DECLARED_INLINE_P (node))
{
warning (OPT_Wattributes, "inline function %q+D declared as "
" dllimport: attribute ignored", node);
*no_add_attrs = true;
}
else if (TREE_CODE (node) == FUNCTION_DECL && DECL_INITIAL (node))
{
error ("function %q+D definition is marked dllimport", node);
*no_add_attrs = true;
}
else if (VAR_P (node))
{
if (DECL_INITIAL (node))
{
error ("variable %q+D definition is marked dllimport",
node);
*no_add_attrs = true;
}
DECL_EXTERNAL (node) = 1;
if (current_function_decl != NULL_TREE && !TREE_STATIC (node))
TREE_PUBLIC (node) = 1;
TREE_STATIC (node) = 0;
}
if (*no_add_attrs == false)
DECL_DLLIMPORT_P (node) = 1;
}
else if (TREE_CODE (node) == FUNCTION_DECL
&& DECL_DECLARED_INLINE_P (node)
&& flag_keep_inline_dllexport)
DECL_EXTERNAL (node) = 0;
if (!TREE_PUBLIC (node) && VAR_OR_FUNCTION_DECL_P (node))
{
error ("external linkage required for symbol %q+D because of "
"%qE attribute", node, name);
*no_add_attrs = true;
}
if (!*no_add_attrs)
{
if (DECL_VISIBILITY_SPECIFIED (node)
&& DECL_VISIBILITY (node) != VISIBILITY_DEFAULT)
error ("%qE implies default visibility, but %qD has already "
"been declared with a different visibility",
name, node);
DECL_VISIBILITY (node) = VISIBILITY_DEFAULT;
DECL_VISIBILITY_SPECIFIED (node) = 1;
}
return NULL_TREE;
}
#endif 
int
attribute_list_equal (const_tree l1, const_tree l2)
{
if (l1 == l2)
return 1;
return attribute_list_contained (l1, l2)
&& attribute_list_contained (l2, l1);
}
int
attribute_list_contained (const_tree l1, const_tree l2)
{
const_tree t1, t2;
if (l1 == l2)
return 1;
for (t1 = l1, t2 = l2;
t1 != 0 && t2 != 0
&& get_attribute_name (t1) == get_attribute_name (t2)
&& TREE_VALUE (t1) == TREE_VALUE (t2);
t1 = TREE_CHAIN (t1), t2 = TREE_CHAIN (t2))
;
if (t1 == 0 && t2 == 0)
return 1;
for (; t2 != 0; t2 = TREE_CHAIN (t2))
{
const_tree attr;
for (attr = lookup_ident_attribute (get_attribute_name (t2),
CONST_CAST_TREE (l1));
attr != NULL_TREE && !attribute_value_equal (t2, attr);
attr = lookup_ident_attribute (get_attribute_name (t2),
TREE_CHAIN (attr)))
;
if (attr == NULL_TREE)
return 0;
}
return 1;
}
tree
private_lookup_attribute (const char *attr_name, size_t attr_len, tree list)
{
while (list)
{
tree attr = get_attribute_name (list);
size_t ident_len = IDENTIFIER_LENGTH (attr);
if (cmp_attribs (attr_name, attr_len, IDENTIFIER_POINTER (attr),
ident_len))
break;
list = TREE_CHAIN (list);
}
return list;
}
#if CHECKING_P
namespace selftest
{
typedef std::pair<const char *, const char *> excl_pair;
struct excl_hash_traits: typed_noop_remove<excl_pair>
{
typedef excl_pair  value_type;
typedef value_type compare_type;
static hashval_t hash (const value_type &x)
{
hashval_t h1 = htab_hash_string (x.first);
hashval_t h2 = htab_hash_string (x.second);
return h1 ^ h2;
}
static bool equal (const value_type &x, const value_type &y)
{
return !strcmp (x.first, y.first) && !strcmp (x.second, y.second);
}
static void mark_deleted (value_type &x)
{
x = value_type (NULL, NULL);
}
static void mark_empty (value_type &x)
{
x = value_type ("", "");
}
static bool is_deleted (const value_type &x)
{
return !x.first && !x.second;
}
static bool is_empty (const value_type &x)
{
return !*x.first && !*x.second;
}
};
static void
test_attribute_exclusions ()
{
const size_t ntables = ARRAY_SIZE (attribute_tables);
typedef hash_set<excl_pair, excl_hash_traits> exclusion_set;
exclusion_set excl_set;
for (size_t ti0 = 0; ti0 != ntables; ++ti0)
for (size_t s0 = 0; attribute_tables[ti0][s0].name; ++s0)
{
const attribute_spec::exclusions *excl
= attribute_tables[ti0][s0].exclude;
if (!excl)
continue;
const char *attr_name = attribute_tables[ti0][s0].name;
for (size_t ei0 = 0; excl[ei0].name; ++ei0)
{
const char *excl_name = excl[ei0].name;
if (!strcmp (attr_name, excl_name))
continue;
excl_set.add (excl_pair (attr_name, excl_name));
}
}
for (exclusion_set::iterator it = excl_set.begin ();
it != excl_set.end ();
++it)
{
if (!excl_set.contains (excl_pair ((*it).second, (*it).first)))
{
char desc[120];
sprintf (desc, "'%s' attribute exclusion '%s' must be symmetric",
(*it).first, (*it).second);
fail (SELFTEST_LOCATION, desc);
}
}
}
void
attribute_c_tests ()
{
test_attribute_exclusions ();
}
} 
#endif 
