#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "cp-tree.h"
#include "stringpool.h"
#include "c-family/c-pragma.h"
#include "c-family/c-objc.h"
static int interface_strcmp (const char *);
static void init_cp_pragma (void);
static tree parse_strconst_pragma (const char *, int);
static void handle_pragma_vtable (cpp_reader *);
static void handle_pragma_unit (cpp_reader *);
static void handle_pragma_interface (cpp_reader *);
static void handle_pragma_implementation (cpp_reader *);
static void init_operators (void);
static void copy_lang_type (tree);
#define CONSTRAINT(name, expr) extern int constraint_##name [(expr) ? 1 : -1]
struct impl_files
{
const char *filename;
struct impl_files *next;
};
static struct impl_files *impl_file_chain;

void
cxx_finish (void)
{
c_common_finish ();
}
ovl_op_info_t ovl_op_info[2][OVL_OP_MAX] = 
{
{
{NULL_TREE, NULL, NULL, ERROR_MARK, OVL_OP_ERROR_MARK, 0},
{NULL_TREE, NULL, NULL, NOP_EXPR, OVL_OP_NOP_EXPR, 0},
#define DEF_OPERATOR(NAME, CODE, MANGLING, FLAGS) \
{NULL_TREE, NAME, MANGLING, CODE, OVL_OP_##CODE, FLAGS},
#define OPERATOR_TRANSITION }, {			\
{NULL_TREE, NULL, NULL, ERROR_MARK, OVL_OP_ERROR_MARK, 0},
#include "operators.def"
}
};
unsigned char ovl_op_mapping[MAX_TREE_CODES];
unsigned char ovl_op_alternate[OVL_OP_MAX];
const char *
get_identifier_kind_name (tree id)
{
static const char *const names[cik_max] = {
"normal", "keyword", "constructor", "destructor",
"simple-op", "assign-op", "conv-op", "<reserved>udlit-op"
};
unsigned kind = 0;
kind |= IDENTIFIER_KIND_BIT_2 (id) << 2;
kind |= IDENTIFIER_KIND_BIT_1 (id) << 1;
kind |= IDENTIFIER_KIND_BIT_0 (id) << 0;
return names[kind];
}
void
set_identifier_kind (tree id, cp_identifier_kind kind)
{
gcc_checking_assert (!IDENTIFIER_KIND_BIT_2 (id)
& !IDENTIFIER_KIND_BIT_1 (id)
& !IDENTIFIER_KIND_BIT_0 (id));
IDENTIFIER_KIND_BIT_2 (id) |= (kind >> 2) & 1;
IDENTIFIER_KIND_BIT_1 (id) |= (kind >> 1) & 1;
IDENTIFIER_KIND_BIT_0 (id) |= (kind >> 0) & 1;
}
static tree
set_operator_ident (ovl_op_info_t *ptr)
{
char buffer[32];
size_t len = snprintf (buffer, sizeof (buffer), "operator%s%s",
&" "[ptr->name[0] && ptr->name[0] != '_'
&& !ISALPHA (ptr->name[0])],
ptr->name);
gcc_checking_assert (len < sizeof (buffer));
tree ident = get_identifier_with_length (buffer, len);
ptr->identifier = ident;
return ident;
}
static void
init_operators (void)
{
gcc_checking_assert (!OVL_OP_ERROR_MARK && !ERROR_MARK);
for (unsigned ix = OVL_OP_MAX; --ix;)
{
ovl_op_info_t *op_ptr = &ovl_op_info[false][ix];
if (op_ptr->name)
{
gcc_checking_assert (op_ptr->ovl_op_code < (1 << 6));
tree ident = set_operator_ident (op_ptr);
if (unsigned index = IDENTIFIER_CP_INDEX (ident))
{
ovl_op_info_t *bin_ptr = &ovl_op_info[false][index];
gcc_checking_assert ((op_ptr->flags ^ bin_ptr->flags)
== OVL_OP_FLAG_AMBIARY);
bin_ptr->flags |= op_ptr->flags;
ovl_op_alternate[index] = ix;
}
else
{
IDENTIFIER_CP_INDEX (ident) = ix;
set_identifier_kind (ident, cik_simple_op);
}
}
if (op_ptr->tree_code)
{
gcc_checking_assert (op_ptr->ovl_op_code == ix
&& !ovl_op_mapping[op_ptr->tree_code]);
ovl_op_mapping[op_ptr->tree_code] = op_ptr->ovl_op_code;
}
ovl_op_info_t *as_ptr = &ovl_op_info[true][ix];
if (as_ptr->name)
{
if (as_ptr->ovl_op_code != ix)
{
ovl_op_info_t *dst_ptr = &ovl_op_info[true][as_ptr->ovl_op_code];
gcc_assert (as_ptr->ovl_op_code > ix && !dst_ptr->tree_code);
memcpy (dst_ptr, as_ptr, sizeof (*dst_ptr));
memset (as_ptr, 0, sizeof (*as_ptr));
as_ptr = dst_ptr;
}
tree ident = set_operator_ident (as_ptr);
gcc_checking_assert (!IDENTIFIER_CP_INDEX (ident));
IDENTIFIER_CP_INDEX (ident) = as_ptr->ovl_op_code;
set_identifier_kind (ident, cik_assign_op);
gcc_checking_assert (!ovl_op_mapping[as_ptr->tree_code]
|| (ovl_op_mapping[as_ptr->tree_code]
== as_ptr->ovl_op_code));
ovl_op_mapping[as_ptr->tree_code] = as_ptr->ovl_op_code;
}
}
}
void
init_reswords (void)
{
unsigned int i;
tree id;
int mask = 0;
if (cxx_dialect < cxx11)
mask |= D_CXX11;
if (!flag_concepts)
mask |= D_CXX_CONCEPTS;
if (!flag_tm)
mask |= D_TRANSMEM;
if (flag_no_asm)
mask |= D_ASM | D_EXT;
if (flag_no_gnu_keywords)
mask |= D_EXT;
mask |= D_OBJC;
ridpointers = ggc_cleared_vec_alloc<tree> ((int) RID_MAX);
for (i = 0; i < num_c_common_reswords; i++)
{
if (c_common_reswords[i].disable & D_CONLY)
continue;
id = get_identifier (c_common_reswords[i].word);
C_SET_RID_CODE (id, c_common_reswords[i].rid);
ridpointers [(int) c_common_reswords[i].rid] = id;
if (! (c_common_reswords[i].disable & mask))
set_identifier_kind (id, cik_keyword);
}
for (i = 0; i < NUM_INT_N_ENTS; i++)
{
char name[50];
sprintf (name, "__int%d", int_n_data[i].bitsize);
id = get_identifier (name);
C_SET_RID_CODE (id, RID_FIRST_INT_N + i);
set_identifier_kind (id, cik_keyword);
}
}
static void
init_cp_pragma (void)
{
c_register_pragma (0, "vtable", handle_pragma_vtable);
c_register_pragma (0, "unit", handle_pragma_unit);
c_register_pragma (0, "interface", handle_pragma_interface);
c_register_pragma (0, "implementation", handle_pragma_implementation);
c_register_pragma ("GCC", "interface", handle_pragma_interface);
c_register_pragma ("GCC", "implementation", handle_pragma_implementation);
}

bool statement_code_p[MAX_TREE_CODES];
bool
cxx_init (void)
{
location_t saved_loc;
unsigned int i;
static const enum tree_code stmt_codes[] = {
CTOR_INITIALIZER,	TRY_BLOCK,	HANDLER,
EH_SPEC_BLOCK,	USING_STMT,	TAG_DEFN,
IF_STMT,		CLEANUP_STMT,	FOR_STMT,
RANGE_FOR_STMT,	WHILE_STMT,	DO_STMT,
BREAK_STMT,		CONTINUE_STMT,	SWITCH_STMT,
EXPR_STMT
};
memset (&statement_code_p, 0, sizeof (statement_code_p));
for (i = 0; i < ARRAY_SIZE (stmt_codes); i++)
statement_code_p[stmt_codes[i]] = true;
saved_loc = input_location;
input_location = BUILTINS_LOCATION;
init_reswords ();
init_tree ();
init_cp_semantics ();
init_operators ();
init_method ();
current_function_decl = NULL;
class_type_node = ridpointers[(int) RID_CLASS];
cxx_init_decl_processing ();
if (c_common_init () == false)
{
input_location = saved_loc;
return false;
}
init_cp_pragma ();
init_repo ();
input_location = saved_loc;
return true;
}

static int
interface_strcmp (const char* s)
{
struct impl_files *ifiles;
const char *s1;
for (ifiles = impl_file_chain; ifiles; ifiles = ifiles->next)
{
const char *t1 = ifiles->filename;
s1 = s;
if (*s1 == 0 || filename_ncmp (s1, t1, 1) != 0)
continue;
while (*s1 != 0 && filename_ncmp (s1, t1, 1) == 0)
s1++, t1++;
if (*s1 == *t1)
return 0;
if (strchr (s1, '.') || strchr (t1, '.'))
continue;
if (*s1 == '\0' || s1[-1] != '.' || t1[-1] != '.')
continue;
return 0;
}
return 1;
}

static tree
parse_strconst_pragma (const char* name, int opt)
{
tree result, x;
enum cpp_ttype t;
t = pragma_lex (&result);
if (t == CPP_STRING)
{
if (pragma_lex (&x) != CPP_EOF)
warning (0, "junk at end of #pragma %s", name);
return result;
}
if (t == CPP_EOF && opt)
return NULL_TREE;
error ("invalid #pragma %s", name);
return error_mark_node;
}
static void
handle_pragma_vtable (cpp_reader* )
{
parse_strconst_pragma ("vtable", 0);
sorry ("#pragma vtable no longer supported");
}
static void
handle_pragma_unit (cpp_reader* )
{
parse_strconst_pragma ("unit", 0);
}
static void
handle_pragma_interface (cpp_reader* )
{
tree fname = parse_strconst_pragma ("interface", 1);
struct c_fileinfo *finfo;
const char *filename;
if (fname == error_mark_node)
return;
else if (fname == 0)
filename = lbasename (LOCATION_FILE (input_location));
else
filename = TREE_STRING_POINTER (fname);
finfo = get_fileinfo (LOCATION_FILE (input_location));
if (impl_file_chain == 0)
{
if (main_input_filename == 0)
main_input_filename = LOCATION_FILE (input_location);
}
finfo->interface_only = interface_strcmp (filename);
if (!MULTIPLE_SYMBOL_SPACES || !finfo->interface_only)
finfo->interface_unknown = 0;
}
static void
handle_pragma_implementation (cpp_reader* )
{
tree fname = parse_strconst_pragma ("implementation", 1);
const char *filename;
struct impl_files *ifiles = impl_file_chain;
if (fname == error_mark_node)
return;
if (fname == 0)
{
if (main_input_filename)
filename = main_input_filename;
else
filename = LOCATION_FILE (input_location);
filename = lbasename (filename);
}
else
{
filename = TREE_STRING_POINTER (fname);
if (cpp_included_before (parse_in, filename, input_location))
warning (0, "#pragma implementation for %qs appears after "
"file is included", filename);
}
for (; ifiles; ifiles = ifiles->next)
{
if (! filename_cmp (ifiles->filename, filename))
break;
}
if (ifiles == 0)
{
ifiles = XNEW (struct impl_files);
ifiles->filename = xstrdup (filename);
ifiles->next = impl_file_chain;
impl_file_chain = ifiles;
}
}
tree
unqualified_name_lookup_error (tree name, location_t loc)
{
if (loc == UNKNOWN_LOCATION)
loc = EXPR_LOC_OR_LOC (name, input_location);
if (IDENTIFIER_ANY_OP_P (name))
error_at (loc, "%qD not defined", name);
else
{
if (!objc_diagnose_private_ivar (name))
{
error_at (loc, "%qD was not declared in this scope", name);
suggest_alternatives_for (loc, name, true);
}
if (local_bindings_p ())
{
tree decl = build_decl (loc, VAR_DECL, name, error_mark_node);
TREE_USED (decl) = true;
pushdecl (decl);
}
}
return error_mark_node;
}
tree
unqualified_fn_lookup_error (cp_expr name_expr)
{
tree name = name_expr.get_value ();
location_t loc = name_expr.get_location ();
if (loc == UNKNOWN_LOCATION)
loc = input_location;
if (processing_template_decl)
{
permerror (loc, "there are no arguments to %qD that depend on a template "
"parameter, so a declaration of %qD must be available",
name, name);
if (!flag_permissive)
{
static bool hint;
if (!hint)
{
inform (loc, "(if you use %<-fpermissive%>, G++ will accept your "
"code, but allowing the use of an undeclared name is "
"deprecated)");
hint = true;
}
}
return name;
}
return unqualified_name_lookup_error (name, loc);
}
struct conv_type_hasher : ggc_ptr_hash<tree_node>
{
static hashval_t hash (tree node)
{
return (hashval_t) TYPE_UID (TREE_TYPE (node));
}
static bool equal (tree node, tree type)
{
return TREE_TYPE (node) == type;
}
};
static GTY (()) hash_table<conv_type_hasher> *conv_type_names;
tree
make_conv_op_name (tree type)
{
if (type == error_mark_node)
return error_mark_node;
if (conv_type_names == NULL)
conv_type_names = hash_table<conv_type_hasher>::create_ggc (31);
tree *slot = conv_type_names->find_slot_with_hash
(type, (hashval_t) TYPE_UID (type), INSERT);
tree identifier = *slot;
if (!identifier)
{
identifier = copy_node (conv_op_identifier);
IDENTIFIER_BINDING (identifier) = NULL;
TREE_TYPE (identifier) = type;
*slot = identifier;
}
return identifier;
}
tree
build_lang_decl (enum tree_code code, tree name, tree type)
{
return build_lang_decl_loc (input_location, code, name, type);
}
tree
build_lang_decl_loc (location_t loc, enum tree_code code, tree name, tree type)
{
tree t;
t = build_decl (loc, code, name, type);
retrofit_lang_decl (t);
return t;
}
static bool
maybe_add_lang_decl_raw (tree t, bool decomp_p)
{
size_t size;
lang_decl_selector sel;
if (decomp_p)
sel = lds_decomp, size = sizeof (struct lang_decl_decomp);
else if (TREE_CODE (t) == FUNCTION_DECL)
sel = lds_fn, size = sizeof (struct lang_decl_fn);
else if (TREE_CODE (t) == NAMESPACE_DECL)
sel = lds_ns, size = sizeof (struct lang_decl_ns);
else if (TREE_CODE (t) == PARM_DECL)
sel = lds_parm, size = sizeof (struct lang_decl_parm);
else if (LANG_DECL_HAS_MIN (t))
sel = lds_min, size = sizeof (struct lang_decl_min);
else
return false;
struct lang_decl *ld
= (struct lang_decl *) ggc_internal_cleared_alloc (size);
ld->u.base.selector = sel;
DECL_LANG_SPECIFIC (t) = ld;
if (sel == lds_ns)
ld->u.ns.bindings = hash_table<named_decl_hash>::create_ggc (499);
if (GATHER_STATISTICS)
{
tree_node_counts[(int)lang_decl] += 1;
tree_node_sizes[(int)lang_decl] += size;
}
return true;
}
static void
set_decl_linkage (tree t)
{
if (current_lang_name == lang_name_cplusplus
|| decl_linkage (t) == lk_none)
SET_DECL_LANGUAGE (t, lang_cplusplus);
else if (current_lang_name == lang_name_c)
SET_DECL_LANGUAGE (t, lang_c);
else
gcc_unreachable ();
}
void
fit_decomposition_lang_decl (tree t, tree base)
{
if (struct lang_decl *orig_ld = DECL_LANG_SPECIFIC (t))
{
if (orig_ld->u.base.selector == lds_min)
{
maybe_add_lang_decl_raw (t, true);
memcpy (DECL_LANG_SPECIFIC (t), orig_ld,
sizeof (struct lang_decl_min));
DECL_LANG_SPECIFIC (t)->u.base.selector = lds_decomp;
}
else
gcc_checking_assert (orig_ld->u.base.selector == lds_decomp);
}
else
{
maybe_add_lang_decl_raw (t, true);
set_decl_linkage (t);
}
DECL_DECOMP_BASE (t) = base;
}
void
retrofit_lang_decl (tree t)
{
if (DECL_LANG_SPECIFIC (t))
return;
if (maybe_add_lang_decl_raw (t, false))
set_decl_linkage (t);
}
void
cxx_dup_lang_specific_decl (tree node)
{
int size;
if (! DECL_LANG_SPECIFIC (node))
return;
switch (DECL_LANG_SPECIFIC (node)->u.base.selector)
{
case lds_min:
size = sizeof (struct lang_decl_min);
break;
case lds_fn:
size = sizeof (struct lang_decl_fn);
break;
case lds_ns:
size = sizeof (struct lang_decl_ns);
break;
case lds_parm:
size = sizeof (struct lang_decl_parm);
break;
case lds_decomp:
size = sizeof (struct lang_decl_decomp);
break;
default:
gcc_unreachable ();
}
struct lang_decl *ld = (struct lang_decl *) ggc_internal_alloc (size);
memcpy (ld, DECL_LANG_SPECIFIC (node), size);
DECL_LANG_SPECIFIC (node) = ld;
if (GATHER_STATISTICS)
{
tree_node_counts[(int)lang_decl] += 1;
tree_node_sizes[(int)lang_decl] += size;
}
}
tree
copy_decl (tree decl MEM_STAT_DECL)
{
tree copy;
copy = copy_node (decl PASS_MEM_STAT);
cxx_dup_lang_specific_decl (copy);
return copy;
}
static void
copy_lang_type (tree node)
{
if (! TYPE_LANG_SPECIFIC (node))
return;
struct lang_type *lt
= (struct lang_type *) ggc_internal_alloc (sizeof (struct lang_type));
memcpy (lt, TYPE_LANG_SPECIFIC (node), (sizeof (struct lang_type)));
TYPE_LANG_SPECIFIC (node) = lt;
if (GATHER_STATISTICS)
{
tree_node_counts[(int)lang_type] += 1;
tree_node_sizes[(int)lang_type] += sizeof (struct lang_type);
}
}
tree
copy_type (tree type MEM_STAT_DECL)
{
tree copy;
copy = copy_node (type PASS_MEM_STAT);
copy_lang_type (copy);
return copy;
}
static bool
maybe_add_lang_type_raw (tree t)
{
if (!RECORD_OR_UNION_CODE_P (TREE_CODE (t)))
return false;
TYPE_LANG_SPECIFIC (t)
= (struct lang_type *) (ggc_internal_cleared_alloc
(sizeof (struct lang_type)));
if (GATHER_STATISTICS)
{
tree_node_counts[(int)lang_type] += 1;
tree_node_sizes[(int)lang_type] += sizeof (struct lang_type);
}
return true;
}
tree
cxx_make_type (enum tree_code code)
{
tree t = make_node (code);
if (maybe_add_lang_type_raw (t))
{
struct c_fileinfo *finfo =
get_fileinfo (LOCATION_FILE (input_location));
SET_CLASSTYPE_INTERFACE_UNKNOWN_X (t, finfo->interface_unknown);
CLASSTYPE_INTERFACE_ONLY (t) = finfo->interface_only;
}
return t;
}
tree
make_class_type (enum tree_code code)
{
tree t = cxx_make_type (code);
SET_CLASS_TYPE_P (t, 1);
return t;
}
bool
in_main_input_context (void)
{
struct tinst_level *tl = outermost_tinst_level();
if (tl)
return filename_cmp (main_input_filename,
LOCATION_FILE (tl->locus)) == 0;
else
return filename_cmp (main_input_filename, LOCATION_FILE (input_location)) == 0;
}
#include "gt-cp-lex.h"
