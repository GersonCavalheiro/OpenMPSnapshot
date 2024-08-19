#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "cp-tree.h"
#include "memmodel.h"
#include "tm_p.h"
#include "stringpool.h"
#include "intl.h"
#include "stor-layout.h"
#include "c-family/c-pragma.h"
#include "gcc-rich-location.h"
struct GTY (()) tinfo_s {
tree type;  
tree vtable; 
tree name;  
};
enum tinfo_kind
{
TK_TYPE_INFO_TYPE,    
TK_BASE_TYPE,		
TK_DERIVED_TYPES,	
TK_BUILTIN_TYPE = TK_DERIVED_TYPES,	
TK_ARRAY_TYPE,	
TK_FUNCTION_TYPE,	
TK_ENUMERAL_TYPE,	
TK_POINTER_TYPE,	
TK_POINTER_MEMBER_TYPE, 
TK_CLASS_TYPE,	
TK_SI_CLASS_TYPE,	
TK_VMI_CLASS_TYPES,	
TK_MAX
};
static const char *const tinfo_names[TK_MAX] =
{
"__type_info",
"__base_class_type_info",
"__fundamental_type_info",
"__array_type_info",
"__function_type_info",
"__enum_type_info",
"__pointer_type_info",
"__pointer_to_member_type_info",
"__class_type_info",
"__si_class_type_info",
"__vmi_class_type_info"
};
#define LONGPTR_T \
integer_types[(POINTER_SIZE <= TYPE_PRECISION (integer_types[itk_long]) \
? itk_long : itk_long_long)]
vec<tree, va_gc> *unemitted_tinfo_decls;
static GTY (()) vec<tinfo_s, va_gc> *tinfo_descs;
static tree ifnonnull (tree, tree, tsubst_flags_t);
static tree tinfo_name (tree, bool);
static tree build_dynamic_cast_1 (tree, tree, tsubst_flags_t);
static tree throw_bad_cast (void);
static tree throw_bad_typeid (void);
static tree get_tinfo_ptr (tree);
static bool typeid_ok_p (void);
static int qualifier_flags (tree);
static bool target_incomplete_p (tree);
static tree tinfo_base_init (tinfo_s *, tree);
static tree generic_initializer (tinfo_s *, tree);
static tree ptr_initializer (tinfo_s *, tree);
static tree ptm_initializer (tinfo_s *, tree);
static tree class_initializer (tinfo_s *, tree, unsigned, ...);
static tree get_pseudo_ti_init (tree, unsigned);
static unsigned get_pseudo_ti_index (tree);
static tinfo_s *get_tinfo_desc (unsigned);
static void create_tinfo_types (void);
static bool typeinfo_in_lib_p (tree);
static int doing_runtime = 0;

static void
push_abi_namespace (void)
{
push_nested_namespace (abi_node);
push_visibility ("default", 2);
}
static void
pop_abi_namespace (void)
{
pop_visibility (2);
pop_nested_namespace (abi_node);
}
void
init_rtti_processing (void)
{
tree type_info_type;
push_namespace (std_identifier);
type_info_type = xref_tag (class_type, get_identifier ("type_info"),
ts_current, false);
pop_namespace ();
const_type_info_type_node
= cp_build_qualified_type (type_info_type, TYPE_QUAL_CONST);
type_info_ptr_type = build_pointer_type (const_type_info_type_node);
vec_alloc (unemitted_tinfo_decls, 124);
create_tinfo_types ();
}
tree
build_headof (tree exp)
{
tree type = TREE_TYPE (exp);
tree offset;
tree index;
gcc_assert (TYPE_PTR_P (type));
type = TREE_TYPE (type);
if (!TYPE_POLYMORPHIC_P (type))
return exp;
exp = save_expr (exp);
index = build_int_cst (NULL_TREE,
-2 * TARGET_VTABLE_DATA_ENTRY_DISTANCE);
offset = build_vtbl_ref (cp_build_fold_indirect_ref (exp),
index);
type = cp_build_qualified_type (ptr_type_node,
cp_type_quals (TREE_TYPE (exp)));
return fold_build_pointer_plus (exp, offset);
}
static tree
throw_bad_cast (void)
{
static tree fn;
if (!fn)
{
tree name = get_identifier ("__cxa_bad_cast");
fn = get_global_binding (name);
if (!fn)
fn = push_throw_library_fn
(name, build_function_type_list (ptr_type_node, NULL_TREE));
}
return build_cxx_call (fn, 0, NULL, tf_warning_or_error);
}
static tree
throw_bad_typeid (void)
{
static tree fn;
if (!fn)
{
tree name = get_identifier ("__cxa_bad_typeid");
fn = get_global_binding (name);
if (!fn)
{
tree t = build_reference_type (const_type_info_type_node);
t = build_function_type_list (t, NULL_TREE);
fn = push_throw_library_fn (name, t);
}
}
return build_cxx_call (fn, 0, NULL, tf_warning_or_error);
}

static tree
get_tinfo_decl_dynamic (tree exp, tsubst_flags_t complain)
{
tree type;
tree t;
if (error_operand_p (exp))
return error_mark_node;
exp = resolve_nondeduced_context (exp, complain);
type = non_reference (TREE_TYPE (exp));
type = TYPE_MAIN_VARIANT (type);
if (CLASS_TYPE_P (type) || type == unknown_type_node
|| type == init_list_type_node)
type = complete_type_or_maybe_complain (type, exp, complain);
if (!type)
return error_mark_node;
if (TYPE_POLYMORPHIC_P (type) && ! resolves_to_fixed_type_p (exp, 0))
{
tree index;
index = build_int_cst (NULL_TREE,
-1 * TARGET_VTABLE_DATA_ENTRY_DISTANCE);
t = build_vtbl_ref (exp, index);
t = convert (type_info_ptr_type, t);
}
else
t = get_tinfo_ptr (TYPE_MAIN_VARIANT (type));
return cp_build_fold_indirect_ref (t);
}
static bool
typeid_ok_p (void)
{
if (! flag_rtti)
{
error ("cannot use %<typeid%> with -fno-rtti");
return false;
}
if (!COMPLETE_TYPE_P (const_type_info_type_node))
{
gcc_rich_location richloc (input_location);
maybe_add_include_fixit (&richloc, "<typeinfo>");
error_at (&richloc,
"must %<#include <typeinfo>%> before using"
" %<typeid%>");
return false;
}
tree pseudo = TYPE_MAIN_VARIANT (get_tinfo_desc (TK_TYPE_INFO_TYPE)->type);
tree real = TYPE_MAIN_VARIANT (const_type_info_type_node);
if (! TYPE_ALIAS_SET_KNOWN_P (pseudo))
TYPE_ALIAS_SET (pseudo) = get_alias_set (real);
else
gcc_assert (TYPE_ALIAS_SET (pseudo) == get_alias_set (real));
return true;
}
tree
build_typeid (tree exp, tsubst_flags_t complain)
{
tree cond = NULL_TREE, initial_expr = exp;
int nonnull = 0;
if (exp == error_mark_node || !typeid_ok_p ())
return error_mark_node;
if (processing_template_decl)
return build_min (TYPEID_EXPR, const_type_info_type_node, exp);
if (TYPE_POLYMORPHIC_P (TREE_TYPE (exp))
&& ! resolves_to_fixed_type_p (exp, &nonnull)
&& ! nonnull)
{
exp = cp_build_addr_expr (exp, complain);
exp = save_expr (exp);
cond = cp_convert (boolean_type_node, exp, complain);
exp = cp_build_fold_indirect_ref (exp);
}
exp = get_tinfo_decl_dynamic (exp, complain);
if (exp == error_mark_node)
return error_mark_node;
if (cond)
{
tree bad = throw_bad_typeid ();
exp = build3 (COND_EXPR, TREE_TYPE (exp), cond, exp, bad);
}
else
mark_type_use (initial_expr);
return exp;
}
static tree
tinfo_name (tree type, bool mark_private)
{
const char *name;
int length;
tree name_string;
name = mangle_type_string (type);
length = strlen (name);
if (mark_private)
{
char* buf = (char*) XALLOCAVEC (char, length + 2);
buf[0] = '*';
memcpy (buf + 1, name, length + 1);
name_string = build_string (length + 2, buf);
}
else
name_string = build_string (length + 1, name);
return fix_string_type (name_string);
}
tree
get_tinfo_decl (tree type)
{
tree name;
tree d;
if (variably_modified_type_p (type, NULL_TREE))
{
error ("cannot create type information for type %qT because "
"it involves types of variable size",
type);
return error_mark_node;
}
if (TREE_CODE (type) == METHOD_TYPE)
type = build_function_type (TREE_TYPE (type),
TREE_CHAIN (TYPE_ARG_TYPES (type)));
type = complete_type (type);
if (CLASS_TYPE_P (type))
{
d = CLASSTYPE_TYPEINFO_VAR (TYPE_MAIN_VARIANT (type));
if (d)
return d;
}
name = mangle_typeinfo_for_type (type);
d = get_global_binding (name);
if (!d)
{
int ix = get_pseudo_ti_index (type);
const tinfo_s *ti = get_tinfo_desc (ix);
d = build_lang_decl (VAR_DECL, name, ti->type);
SET_DECL_ASSEMBLER_NAME (d, name);
TREE_TYPE (name) = type;
DECL_TINFO_P (d) = 1;
DECL_ARTIFICIAL (d) = 1;
DECL_IGNORED_P (d) = 1;
TREE_READONLY (d) = 1;
TREE_STATIC (d) = 1;
DECL_EXTERNAL (d) = 1;
DECL_NOT_REALLY_EXTERN (d) = 1;
set_linkage_according_to_type (type, d);
d = pushdecl_top_level_and_finish (d, NULL_TREE);
if (CLASS_TYPE_P (type))
CLASSTYPE_TYPEINFO_VAR (TYPE_MAIN_VARIANT (type)) = d;
vec_safe_push (unemitted_tinfo_decls, d);
}
return d;
}
static tree
get_tinfo_ptr (tree type)
{
tree decl = get_tinfo_decl (type);
mark_used (decl);
return build_nop (type_info_ptr_type,
build_address (decl));
}
tree
get_typeid (tree type, tsubst_flags_t complain)
{
if (type == error_mark_node || !typeid_ok_p ())
return error_mark_node;
if (processing_template_decl)
return build_min (TYPEID_EXPR, const_type_info_type_node, type);
type = non_reference (type);
if (TREE_CODE (type) == FUNCTION_TYPE
&& (type_memfn_quals (type) != TYPE_UNQUALIFIED
|| type_memfn_rqual (type) != REF_QUAL_NONE))
{
if (complain & tf_error)
error ("typeid of qualified function type %qT", type);
return error_mark_node;
}
type = TYPE_MAIN_VARIANT (type);
if (CLASS_TYPE_P (type) || type == unknown_type_node
|| type == init_list_type_node)
type = complete_type_or_maybe_complain (type, NULL_TREE, complain);
if (!type)
return error_mark_node;
return cp_build_fold_indirect_ref (get_tinfo_ptr (type));
}
static tree
ifnonnull (tree test, tree result, tsubst_flags_t complain)
{
tree cond = build2 (NE_EXPR, boolean_type_node, test,
cp_convert (TREE_TYPE (test), nullptr_node, complain));
TREE_NO_WARNING (cond) = 1;
return build3 (COND_EXPR, TREE_TYPE (result), cond, result,
cp_convert (TREE_TYPE (result), nullptr_node, complain));
}
static tree
build_dynamic_cast_1 (tree type, tree expr, tsubst_flags_t complain)
{
enum tree_code tc = TREE_CODE (type);
tree exprtype;
tree dcast_fn;
tree old_expr = expr;
const char *errstr = NULL;
used_types_insert (type);
switch (tc)
{
case POINTER_TYPE:
if (VOID_TYPE_P (TREE_TYPE (type)))
break;
case REFERENCE_TYPE:
if (! MAYBE_CLASS_TYPE_P (TREE_TYPE (type)))
{
errstr = _("target is not pointer or reference to class");
goto fail;
}
if (!COMPLETE_TYPE_P (complete_type (TREE_TYPE (type))))
{
errstr = _("target is not pointer or reference to complete type");
goto fail;
}
break;
default:
errstr = _("target is not pointer or reference");
goto fail;
}
if (tc == POINTER_TYPE)
{
expr = decay_conversion (expr, complain);
exprtype = TREE_TYPE (expr);
expr = mark_rvalue_use (expr);
if (!TYPE_PTR_P (exprtype))
{
errstr = _("source is not a pointer");
goto fail;
}
if (! MAYBE_CLASS_TYPE_P (TREE_TYPE (exprtype)))
{
errstr = _("source is not a pointer to class");
goto fail;
}
if (!COMPLETE_TYPE_P (complete_type (TREE_TYPE (exprtype))))
{
errstr = _("source is a pointer to incomplete type");
goto fail;
}
}
else
{
expr = mark_lvalue_use (expr);
exprtype = build_reference_type (TREE_TYPE (expr));
if (! MAYBE_CLASS_TYPE_P (TREE_TYPE (exprtype)))
{
errstr = _("source is not of class type");
goto fail;
}
if (!COMPLETE_TYPE_P (complete_type (TREE_TYPE (exprtype))))
{
errstr = _("source is of incomplete class type");
goto fail;
}
}
if (!at_least_as_qualified_p (TREE_TYPE (type),
TREE_TYPE (exprtype)))
{
errstr = _("conversion casts away constness");
goto fail;
}
{
tree binfo = lookup_base (TREE_TYPE (exprtype), TREE_TYPE (type),
ba_check, NULL, complain);
if (binfo)
return build_static_cast (type, expr, complain);
}
if (tc == REFERENCE_TYPE)
expr = convert_to_reference (exprtype, expr, CONV_IMPLICIT,
LOOKUP_NORMAL, NULL_TREE, complain);
if (TYPE_POLYMORPHIC_P (TREE_TYPE (exprtype)))
{
tree expr1;
if (tc == POINTER_TYPE && VOID_TYPE_P (TREE_TYPE (type)))
{
if (TREE_CODE (expr) == ADDR_EXPR
&& VAR_P (TREE_OPERAND (expr, 0))
&& TREE_CODE (TREE_TYPE (TREE_OPERAND (expr, 0))) == RECORD_TYPE)
return build1 (NOP_EXPR, type, expr);
expr = save_expr (expr);
expr1 = build_headof (expr);
if (TREE_TYPE (expr1) != type)
expr1 = build1 (NOP_EXPR, type, expr1);
return ifnonnull (expr, expr1, complain);
}
else
{
tree retval;
tree result, td2, td3;
tree elems[4];
tree static_type, target_type, boff;
if (tc == REFERENCE_TYPE)
{
if (VAR_P (old_expr)
&& TREE_CODE (TREE_TYPE (old_expr)) == RECORD_TYPE)
{
tree expr = throw_bad_cast ();
if (complain & tf_warning)
warning (0, "dynamic_cast of %q#D to %q#T can never succeed",
old_expr, type);
TREE_TYPE (expr) = type;
return expr;
}
}
else if (TREE_CODE (expr) == ADDR_EXPR)
{
tree op = TREE_OPERAND (expr, 0);
if (VAR_P (op)
&& TREE_CODE (TREE_TYPE (op)) == RECORD_TYPE)
{
if (complain & tf_warning)
warning (0, "dynamic_cast of %q#D to %q#T can never succeed",
op, type);
retval = build_int_cst (type, 0);
return retval;
}
}
if (!flag_rtti)
{
if (complain & tf_error)
error ("%<dynamic_cast%> not permitted with -fno-rtti");
return error_mark_node;
}
target_type = TYPE_MAIN_VARIANT (TREE_TYPE (type));
static_type = TYPE_MAIN_VARIANT (TREE_TYPE (exprtype));
td2 = get_tinfo_decl (target_type);
if (!mark_used (td2, complain) && !(complain & tf_error))
return error_mark_node;
td2 = cp_build_addr_expr (td2, complain);
td3 = get_tinfo_decl (static_type);
if (!mark_used (td3, complain) && !(complain & tf_error))
return error_mark_node;
td3 = cp_build_addr_expr (td3, complain);
boff = dcast_base_hint (static_type, target_type);
expr = save_expr (expr);
expr1 = expr;
if (tc == REFERENCE_TYPE)
expr1 = cp_build_addr_expr (expr1, complain);
elems[0] = expr1;
elems[1] = td3;
elems[2] = td2;
elems[3] = boff;
dcast_fn = dynamic_cast_node;
if (!dcast_fn)
{
tree tmp;
tree tinfo_ptr;
const char *name;
push_abi_namespace ();
tinfo_ptr = xref_tag (class_type,
get_identifier ("__class_type_info"),
ts_current, false);
tinfo_ptr = build_pointer_type
(cp_build_qualified_type
(tinfo_ptr, TYPE_QUAL_CONST));
name = "__dynamic_cast";
tmp = build_function_type_list (ptr_type_node,
const_ptr_type_node,
tinfo_ptr, tinfo_ptr,
ptrdiff_type_node, NULL_TREE);
dcast_fn = build_library_fn_ptr (name, tmp,
ECF_LEAF | ECF_PURE | ECF_NOTHROW);
pop_abi_namespace ();
dynamic_cast_node = dcast_fn;
}
result = build_cxx_call (dcast_fn, 4, elems, complain);
if (tc == REFERENCE_TYPE)
{
tree bad = throw_bad_cast ();
tree neq;
result = save_expr (result);
neq = cp_truthvalue_conversion (result);
return cp_convert (type,
build3 (COND_EXPR, TREE_TYPE (result),
neq, result, bad), complain);
}
result = cp_convert (type, result, complain);
return ifnonnull (expr, result, complain);
}
}
else
errstr = _("source type is not polymorphic");
fail:
if (complain & tf_error)
error ("cannot dynamic_cast %qE (of type %q#T) to type %q#T (%s)",
old_expr, TREE_TYPE (old_expr), type, errstr);
return error_mark_node;
}
tree
build_dynamic_cast (tree type, tree expr, tsubst_flags_t complain)
{
tree r;
if (type == error_mark_node || expr == error_mark_node)
return error_mark_node;
if (processing_template_decl)
{
expr = build_min (DYNAMIC_CAST_EXPR, type, expr);
TREE_SIDE_EFFECTS (expr) = 1;
return convert_from_reference (expr);
}
r = convert_from_reference (build_dynamic_cast_1 (type, expr, complain));
if (r != error_mark_node)
maybe_warn_about_useless_cast (type, expr, complain);
return r;
}
static int
qualifier_flags (tree type)
{
int flags = 0;
int quals = cp_type_quals (type);
if (quals & TYPE_QUAL_CONST)
flags |= 1;
if (quals & TYPE_QUAL_VOLATILE)
flags |= 2;
if (quals & TYPE_QUAL_RESTRICT)
flags |= 4;
return flags;
}
static bool
target_incomplete_p (tree type)
{
while (true)
if (TYPE_PTRDATAMEM_P (type))
{
if (!COMPLETE_TYPE_P (TYPE_PTRMEM_CLASS_TYPE (type)))
return true;
type = TYPE_PTRMEM_POINTED_TO_TYPE (type);
}
else if (TYPE_PTR_P (type))
type = TREE_TYPE (type);
else
return !COMPLETE_OR_VOID_TYPE_P (type);
}
static bool
involves_incomplete_p (tree type)
{
switch (TREE_CODE (type))
{
case POINTER_TYPE:
return target_incomplete_p (TREE_TYPE (type));
case OFFSET_TYPE:
ptrmem:
return
(target_incomplete_p (TYPE_PTRMEM_POINTED_TO_TYPE (type))
|| !COMPLETE_TYPE_P (TYPE_PTRMEM_CLASS_TYPE (type)));
case RECORD_TYPE:
if (TYPE_PTRMEMFUNC_P (type))
goto ptrmem;
case UNION_TYPE:
if (!COMPLETE_TYPE_P (type))
return true;
default:
return false;
}
}
static tree
tinfo_base_init (tinfo_s *ti, tree target)
{
tree init;
tree name_decl;
tree vtable_ptr;
vec<constructor_elt, va_gc> *v;
{
tree name_name, name_string;
tree name_type = build_cplus_array_type
(cp_build_qualified_type (char_type_node, TYPE_QUAL_CONST),
NULL_TREE);
name_name = mangle_typeinfo_string_for_type (target);
TREE_TYPE (name_name) = target;
name_decl = build_lang_decl (VAR_DECL, name_name, name_type);
SET_DECL_ASSEMBLER_NAME (name_decl, name_name);
DECL_ARTIFICIAL (name_decl) = 1;
DECL_IGNORED_P (name_decl) = 1;
TREE_READONLY (name_decl) = 1;
TREE_STATIC (name_decl) = 1;
DECL_EXTERNAL (name_decl) = 0;
DECL_TINFO_P (name_decl) = 1;
set_linkage_according_to_type (target, name_decl);
import_export_decl (name_decl);
name_string = tinfo_name (target, !TREE_PUBLIC (name_decl));
DECL_INITIAL (name_decl) = name_string;
mark_used (name_decl);
pushdecl_top_level_and_finish (name_decl, name_string);
}
vtable_ptr = ti->vtable;
if (!vtable_ptr)
{
tree real_type;
push_abi_namespace ();
real_type = xref_tag (class_type, ti->name,
ts_current, false);
pop_abi_namespace ();
if (!COMPLETE_TYPE_P (real_type))
{
SET_CLASSTYPE_INTERFACE_KNOWN (real_type);
CLASSTYPE_INTERFACE_ONLY (real_type) = 1;
}
vtable_ptr = get_vtable_decl (real_type, 1);
vtable_ptr = cp_build_addr_expr (vtable_ptr, tf_warning_or_error);
vtable_ptr = fold_build_pointer_plus
(vtable_ptr,
size_binop (MULT_EXPR,
size_int (2 * TARGET_VTABLE_DATA_ENTRY_DISTANCE),
TYPE_SIZE_UNIT (vtable_entry_type)));
ti->vtable = vtable_ptr;
}
vec_alloc (v, 2);
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE, vtable_ptr);
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE,
decay_conversion (name_decl, tf_warning_or_error));
init = build_constructor (init_list_type_node, v);
TREE_CONSTANT (init) = 1;
TREE_STATIC (init) = 1;
return init;
}
static tree
generic_initializer (tinfo_s *ti, tree target)
{
tree init = tinfo_base_init (ti, target);
init = build_constructor_single (init_list_type_node, NULL_TREE, init);
TREE_CONSTANT (init) = 1;
TREE_STATIC (init) = 1;
return init;
}
static tree
ptr_initializer (tinfo_s *ti, tree target)
{
tree init = tinfo_base_init (ti, target);
tree to = TREE_TYPE (target);
int flags = qualifier_flags (to);
bool incomplete = target_incomplete_p (to);
vec<constructor_elt, va_gc> *v;
vec_alloc (v, 3);
if (incomplete)
flags |= 8;
if (tx_safe_fn_type_p (to))
{
flags |= 0x20;
to = tx_unsafe_fn_variant (to);
}
if (flag_noexcept_type
&& (TREE_CODE (to) == FUNCTION_TYPE
|| TREE_CODE (to) == METHOD_TYPE)
&& TYPE_NOTHROW_P (to))
{
flags |= 0x40;
to = build_exception_variant (to, NULL_TREE);
}
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE, init);
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE, build_int_cst (NULL_TREE, flags));
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE,
get_tinfo_ptr (TYPE_MAIN_VARIANT (to)));
init = build_constructor (init_list_type_node, v);
TREE_CONSTANT (init) = 1;
TREE_STATIC (init) = 1;
return init;
}
static tree
ptm_initializer (tinfo_s *ti, tree target)
{
tree init = tinfo_base_init (ti, target);
tree to = TYPE_PTRMEM_POINTED_TO_TYPE (target);
tree klass = TYPE_PTRMEM_CLASS_TYPE (target);
int flags = qualifier_flags (to);
bool incomplete = target_incomplete_p (to);
vec<constructor_elt, va_gc> *v;
vec_alloc (v, 4);
if (incomplete)
flags |= 0x8;
if (!COMPLETE_TYPE_P (klass))
flags |= 0x10;
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE, init);
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE, build_int_cst (NULL_TREE, flags));
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE,
get_tinfo_ptr (TYPE_MAIN_VARIANT (to)));
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE, get_tinfo_ptr (klass));
init = build_constructor (init_list_type_node, v);
TREE_CONSTANT (init) = 1;
TREE_STATIC (init) = 1;
return init;
}
static tree
class_initializer (tinfo_s *ti, tree target, unsigned n, ...)
{
tree init = tinfo_base_init (ti, target);
va_list extra_inits;
unsigned i;
vec<constructor_elt, va_gc> *v;
vec_alloc (v, n+1);
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE, init);
va_start (extra_inits, n);
for (i = 0; i < n; i++)
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE, va_arg (extra_inits, tree));
va_end (extra_inits);
init = build_constructor (init_list_type_node, v);
TREE_CONSTANT (init) = 1;
TREE_STATIC (init) = 1;
return init;
}
static bool
typeinfo_in_lib_p (tree type)
{
if (TYPE_PTR_P (type)
&& (cp_type_quals (TREE_TYPE (type)) == TYPE_QUAL_CONST
|| cp_type_quals (TREE_TYPE (type)) == TYPE_UNQUALIFIED))
type = TREE_TYPE (type);
switch (TREE_CODE (type))
{
case INTEGER_TYPE:
case BOOLEAN_TYPE:
case REAL_TYPE:
case VOID_TYPE:
case NULLPTR_TYPE:
return true;
case LANG_TYPE:
default:
return false;
}
}
static tree
get_pseudo_ti_init (tree type, unsigned tk_index)
{
tinfo_s *ti = get_tinfo_desc (tk_index);
gcc_assert (at_eof);
switch (tk_index)
{
case TK_POINTER_MEMBER_TYPE:
return ptm_initializer (ti, type);
case TK_POINTER_TYPE:
return ptr_initializer (ti, type);
case TK_BUILTIN_TYPE:
case TK_ENUMERAL_TYPE:
case TK_FUNCTION_TYPE:
case TK_ARRAY_TYPE:
return generic_initializer (ti, type);
case TK_CLASS_TYPE:
return class_initializer (ti, type, 0);
case TK_SI_CLASS_TYPE:
{
tree base_binfo = BINFO_BASE_BINFO (TYPE_BINFO (type), 0);
tree tinfo = get_tinfo_ptr (BINFO_TYPE (base_binfo));
ti = &(*tinfo_descs)[tk_index];
return class_initializer (ti, type, 1, tinfo);
}
default:
{
int hint = ((CLASSTYPE_REPEATED_BASE_P (type) << 0)
| (CLASSTYPE_DIAMOND_SHAPED_P (type) << 1));
tree binfo = TYPE_BINFO (type);
unsigned nbases = BINFO_N_BASE_BINFOS (binfo);
vec<tree, va_gc> *base_accesses = BINFO_BASE_ACCESSES (binfo);
tree offset_type = LONGPTR_T;
vec<constructor_elt, va_gc> *init_vec = NULL;
gcc_assert (tk_index - TK_VMI_CLASS_TYPES + 1 == nbases);
vec_safe_grow (init_vec, nbases);
for (unsigned ix = nbases; ix--;)
{
tree base_binfo = BINFO_BASE_BINFO (binfo, ix);
int flags = 0;
tree tinfo;
tree offset;
vec<constructor_elt, va_gc> *v;
if ((*base_accesses)[ix] == access_public_node)
flags |= 2;
tinfo = get_tinfo_ptr (BINFO_TYPE (base_binfo));
if (BINFO_VIRTUAL_P (base_binfo))
{
offset = BINFO_VPTR_FIELD (base_binfo);
flags |= 1;
}
else
offset = BINFO_OFFSET (base_binfo);
offset = fold_convert (offset_type, offset);
offset = fold_build2_loc (input_location,
LSHIFT_EXPR, offset_type, offset,
build_int_cst (offset_type, 8));
offset = fold_build2_loc (input_location,
BIT_IOR_EXPR, offset_type, offset,
build_int_cst (offset_type, flags));
vec_alloc (v, 2);
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE, tinfo);
CONSTRUCTOR_APPEND_ELT (v, NULL_TREE, offset);
tree base_init = build_constructor (init_list_type_node, v);
constructor_elt *e = &(*init_vec)[ix];
e->index = NULL_TREE;
e->value = base_init;
}
tree base_inits = build_constructor (init_list_type_node, init_vec);
ti = &(*tinfo_descs)[tk_index];
return class_initializer (ti, type, 3,
build_int_cst (NULL_TREE, hint),
build_int_cst (NULL_TREE, nbases),
base_inits);
}
}
}
static unsigned
get_pseudo_ti_index (tree type)
{
unsigned ix;
switch (TREE_CODE (type))
{
case OFFSET_TYPE:
ix = TK_POINTER_MEMBER_TYPE;
break;
case POINTER_TYPE:
ix = TK_POINTER_TYPE;
break;
case ENUMERAL_TYPE:
ix = TK_ENUMERAL_TYPE;
break;
case FUNCTION_TYPE:
ix = TK_FUNCTION_TYPE;
break;
case ARRAY_TYPE:
ix = TK_ARRAY_TYPE;
break;
case UNION_TYPE:
case RECORD_TYPE:
if (TYPE_PTRMEMFUNC_P (type))
ix = TK_POINTER_MEMBER_TYPE;
else if (!COMPLETE_TYPE_P (type))
{
if (!at_eof)
cxx_incomplete_type_error (NULL_TREE, type);
ix = TK_CLASS_TYPE;
}
else if (!TYPE_BINFO (type)
|| !BINFO_N_BASE_BINFOS (TYPE_BINFO (type)))
ix = TK_CLASS_TYPE;
else
{
tree binfo = TYPE_BINFO (type);
vec<tree, va_gc> *base_accesses = BINFO_BASE_ACCESSES (binfo);
tree base_binfo = BINFO_BASE_BINFO (binfo, 0);
int num_bases = BINFO_N_BASE_BINFOS (binfo);
if (num_bases == 1
&& (*base_accesses)[0] == access_public_node
&& !BINFO_VIRTUAL_P (base_binfo)
&& integer_zerop (BINFO_OFFSET (base_binfo)))
ix = TK_SI_CLASS_TYPE;
else
ix = TK_VMI_CLASS_TYPES + num_bases - 1;
}
break;
default:
ix = TK_BUILTIN_TYPE;
break;
}
return ix;
}
static tinfo_s *
get_tinfo_desc (unsigned ix)
{
unsigned len = tinfo_descs->length ();
if (len <= ix)
{
len = ix + 1 - len;
vec_safe_reserve (tinfo_descs, len);
tinfo_s elt;
elt.type = elt.vtable = elt.name = NULL_TREE;
while (len--)
tinfo_descs->quick_push (elt);
}
tinfo_s *res = &(*tinfo_descs)[ix];
if (res->type)
return res;
tree fields = NULL_TREE;
if (ix >= TK_DERIVED_TYPES)
{
tree fld_base = build_decl (BUILTINS_LOCATION, FIELD_DECL, NULL_TREE,
get_tinfo_desc (TK_TYPE_INFO_TYPE)->type);
DECL_CHAIN (fld_base) = fields;
fields = fld_base;
}
switch (ix)
{
case TK_TYPE_INFO_TYPE:
{
tree fld_ptr = build_decl (BUILTINS_LOCATION, FIELD_DECL,
NULL_TREE, const_ptr_type_node);
fields = fld_ptr;
tree fld_str = build_decl (BUILTINS_LOCATION, FIELD_DECL,
NULL_TREE, const_string_type_node);
DECL_CHAIN (fld_str) = fields;
fields = fld_str;
break;
}
case TK_BASE_TYPE:
{
tree fld_ptr = build_decl (BUILTINS_LOCATION, FIELD_DECL,
NULL_TREE, type_info_ptr_type);
DECL_CHAIN (fld_ptr) = fields;
fields = fld_ptr;
tree fld_flag = build_decl (BUILTINS_LOCATION, FIELD_DECL,
NULL_TREE, LONGPTR_T);
DECL_CHAIN (fld_flag) = fields;
fields = fld_flag;
break;
}
case TK_BUILTIN_TYPE:
break;
case TK_ARRAY_TYPE:
break;
case TK_FUNCTION_TYPE:
break;
case TK_ENUMERAL_TYPE:
break;
case TK_POINTER_TYPE:
case TK_POINTER_MEMBER_TYPE:
{
tree fld_mask = build_decl (BUILTINS_LOCATION, FIELD_DECL,
NULL_TREE, integer_type_node);
DECL_CHAIN (fld_mask) = fields;
fields = fld_mask;
tree fld_ptr = build_decl (BUILTINS_LOCATION, FIELD_DECL,
NULL_TREE, type_info_ptr_type);
DECL_CHAIN (fld_ptr) = fields;
fields = fld_ptr;
if (ix == TK_POINTER_MEMBER_TYPE)
{
tree fld_cls = build_decl (BUILTINS_LOCATION, FIELD_DECL,
NULL_TREE, type_info_ptr_type);
DECL_CHAIN (fld_cls) = fields;
fields = fld_cls;
}
break;
}
case TK_CLASS_TYPE:
break;
case TK_SI_CLASS_TYPE:
{
tree fld_ptr = build_decl (BUILTINS_LOCATION, FIELD_DECL,
NULL_TREE, type_info_ptr_type);
DECL_CHAIN (fld_ptr) = fields;
fields = fld_ptr;
break;
}
default: 
{
unsigned num_bases = ix - TK_VMI_CLASS_TYPES + 1;
tree fld_flg = build_decl (BUILTINS_LOCATION, FIELD_DECL,
NULL_TREE, integer_type_node);
DECL_CHAIN (fld_flg) = fields;
fields = fld_flg;
tree fld_cnt = build_decl (BUILTINS_LOCATION, FIELD_DECL,
NULL_TREE, integer_type_node);
DECL_CHAIN (fld_cnt) = fields;
fields = fld_cnt;
tree domain = build_index_type (size_int (num_bases - 1));
tree array = build_array_type (get_tinfo_desc (TK_BASE_TYPE)->type,
domain);
tree fld_ary = build_decl (BUILTINS_LOCATION, FIELD_DECL,
NULL_TREE, array);
DECL_CHAIN (fld_ary) = fields;
fields = fld_ary;
break;
}
}
push_abi_namespace ();
const char *real_name = tinfo_names[ix < TK_VMI_CLASS_TYPES
? ix : unsigned (TK_VMI_CLASS_TYPES)];
size_t name_len = strlen (real_name);
char *pseudo_name = (char *) alloca (name_len + 30);
memcpy (pseudo_name, real_name, name_len);
sprintf (pseudo_name + name_len, "_pseudo_%d", ix);
tree pseudo_type = make_class_type (RECORD_TYPE);
finish_builtin_struct (pseudo_type, pseudo_name, fields, NULL_TREE);
CLASSTYPE_AS_BASE (pseudo_type) = pseudo_type;
res->type = cp_build_qualified_type (pseudo_type, TYPE_QUAL_CONST);
res->name = get_identifier (real_name);
TREE_PUBLIC (TYPE_MAIN_DECL (res->type)) = 1;
pop_abi_namespace ();
return res;
}
static void
create_tinfo_types (void)
{
gcc_assert (!tinfo_descs);
vec_alloc (tinfo_descs, TK_MAX + 20);
}
void
emit_support_tinfo_1 (tree bltn)
{
tree types[3];
if (bltn == NULL_TREE)
return;
types[0] = bltn;
types[1] = build_pointer_type (bltn);
types[2] = build_pointer_type (cp_build_qualified_type (bltn,
TYPE_QUAL_CONST));
for (int i = 0; i < 3; ++i)
{
tree tinfo = get_tinfo_decl (types[i]);
TREE_USED (tinfo) = 1;
mark_needed (tinfo);
if (!flag_weak || ! targetm.cxx.library_rtti_comdat ())
{
gcc_assert (TREE_PUBLIC (tinfo) && !DECL_COMDAT (tinfo));
DECL_INTERFACE_KNOWN (tinfo) = 1;
}
}
}
void
emit_support_tinfos (void)
{
static tree *const fundamentals[] =
{
&void_type_node,
&boolean_type_node,
&wchar_type_node, &char16_type_node, &char32_type_node,
&char_type_node, &signed_char_type_node, &unsigned_char_type_node,
&short_integer_type_node, &short_unsigned_type_node,
&integer_type_node, &unsigned_type_node,
&long_integer_type_node, &long_unsigned_type_node,
&long_long_integer_type_node, &long_long_unsigned_type_node,
&float_type_node, &double_type_node, &long_double_type_node,
&dfloat32_type_node, &dfloat64_type_node, &dfloat128_type_node,
&nullptr_type_node,
0
};
int ix;
tree bltn_type = lookup_qualified_name
(abi_node, get_identifier ("__fundamental_type_info"), true, false, false);
if (TREE_CODE (bltn_type) != TYPE_DECL)
return;
bltn_type = TREE_TYPE (bltn_type);
if (!COMPLETE_TYPE_P (bltn_type))
return;
tree dtor = CLASSTYPE_DESTRUCTOR (bltn_type);
if (!dtor || DECL_EXTERNAL (dtor))
return;
location_t saved_loc = input_location;
input_location = BUILTINS_LOCATION;
doing_runtime = 1;
for (ix = 0; fundamentals[ix]; ix++)
emit_support_tinfo_1 (*fundamentals[ix]);
for (ix = 0; ix < NUM_INT_N_ENTS; ix ++)
if (int_n_enabled_p[ix])
{
emit_support_tinfo_1 (int_n_trees[ix].signed_type);
emit_support_tinfo_1 (int_n_trees[ix].unsigned_type);
}
for (tree t = registered_builtin_types; t; t = TREE_CHAIN (t))
emit_support_tinfo_1 (TREE_VALUE (t));
input_location = saved_loc;
}
bool
emit_tinfo_decl (tree decl)
{
tree type = TREE_TYPE (DECL_NAME (decl));
int in_library = typeinfo_in_lib_p (type);
gcc_assert (DECL_TINFO_P (decl));
if (in_library)
{
if (doing_runtime)
DECL_EXTERNAL (decl) = 0;
else
{
DECL_INTERFACE_KNOWN (decl) = 1;
return false;
}
}
else if (involves_incomplete_p (type))
{
if (!decl_needed_p (decl))
return false;
TREE_PUBLIC (decl) = 0;
DECL_EXTERNAL (decl) = 0;
DECL_INTERFACE_KNOWN (decl) = 1;
}
import_export_decl (decl);
if (DECL_NOT_REALLY_EXTERN (decl) && decl_needed_p (decl))
{
tree init;
DECL_EXTERNAL (decl) = 0;
init = get_pseudo_ti_init (type, get_pseudo_ti_index (type));
DECL_INITIAL (decl) = init;
mark_used (decl);
cp_finish_decl (decl, init, false, NULL_TREE, 0);
#ifdef DATA_ABI_ALIGNMENT
SET_DECL_ALIGN (decl, DATA_ABI_ALIGNMENT (decl, TYPE_ALIGN (TREE_TYPE (decl))));
DECL_USER_ALIGN (decl) = true;
#endif
return true;
}
else
return false;
}
#include "gt-cp-rtti.h"
