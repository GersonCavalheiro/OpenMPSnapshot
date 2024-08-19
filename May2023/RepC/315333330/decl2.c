#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "memmodel.h"
#include "target.h"
#include "cp-tree.h"
#include "c-family/c-common.h"
#include "timevar.h"
#include "stringpool.h"
#include "cgraph.h"
#include "varasm.h"
#include "attribs.h"
#include "stor-layout.h"
#include "calls.h"
#include "decl.h"
#include "toplev.h"
#include "c-family/c-objc.h"
#include "c-family/c-pragma.h"
#include "dumpfile.h"
#include "intl.h"
#include "c-family/c-ada-spec.h"
#include "asan.h"
int raw_dump_id;
extern cpp_reader *parse_in;
typedef struct priority_info_s {
int initializations_p;
int destructions_p;
} *priority_info;
static void mark_vtable_entries (tree);
static bool maybe_emit_vtables (tree);
static tree start_objects (int, int);
static void finish_objects (int, int, tree);
static tree start_static_storage_duration_function (unsigned);
static void finish_static_storage_duration_function (tree);
static priority_info get_priority_info (int);
static void do_static_initialization_or_destruction (tree, bool);
static void one_static_initialization_or_destruction (tree, tree, bool);
static void generate_ctor_or_dtor_function (bool, int, location_t *);
static int generate_ctor_and_dtor_functions_for_priority (splay_tree_node,
void *);
static tree prune_vars_needing_no_initialization (tree *);
static void write_out_vars (tree);
static void import_export_class (tree);
static tree get_guard_bits (tree);
static void determine_visibility_from_class (tree, tree);
static bool determine_hidden_inline (tree);
static void maybe_instantiate_decl (tree);
static GTY(()) vec<tree, va_gc> *pending_statics;
static GTY(()) vec<tree, va_gc> *deferred_fns;
static GTY(()) vec<tree, va_gc> *no_linkage_decls;
static GTY(()) vec<tree, va_gc> *mangling_aliases;
struct mangled_decl_hash : ggc_remove <tree>
{
typedef tree value_type; 
typedef tree compare_type; 
static hashval_t hash (const value_type decl)
{
return IDENTIFIER_HASH_VALUE (DECL_ASSEMBLER_NAME_RAW (decl));
}
static bool equal (const value_type existing, compare_type candidate)
{
tree name = DECL_ASSEMBLER_NAME_RAW (existing);
return candidate == name;
}
static inline void mark_empty (value_type &p) {p = NULL_TREE;}
static inline bool is_empty (value_type p) {return !p;}
static bool is_deleted (value_type e)
{
return e == reinterpret_cast <value_type> (1);
}
static void mark_deleted (value_type &e)
{
e = reinterpret_cast <value_type> (1);
}
};
static GTY(()) hash_table<mangled_decl_hash> *mangled_decls;
int at_eof;
bool defer_mangling_aliases = true;

tree
build_memfn_type (tree fntype, tree ctype, cp_cv_quals quals,
cp_ref_qualifier rqual)
{
tree raises;
tree attrs;
int type_quals;
bool late_return_type_p;
if (fntype == error_mark_node || ctype == error_mark_node)
return error_mark_node;
gcc_assert (TREE_CODE (fntype) == FUNCTION_TYPE
|| TREE_CODE (fntype) == METHOD_TYPE);
type_quals = quals & ~TYPE_QUAL_RESTRICT;
ctype = cp_build_qualified_type (ctype, type_quals);
raises = TYPE_RAISES_EXCEPTIONS (fntype);
attrs = TYPE_ATTRIBUTES (fntype);
late_return_type_p = TYPE_HAS_LATE_RETURN_TYPE (fntype);
fntype = build_method_type_directly (ctype, TREE_TYPE (fntype),
(TREE_CODE (fntype) == METHOD_TYPE
? TREE_CHAIN (TYPE_ARG_TYPES (fntype))
: TYPE_ARG_TYPES (fntype)));
if (attrs)
fntype = cp_build_type_attribute_variant (fntype, attrs);
if (rqual)
fntype = build_ref_qualified_type (fntype, rqual);
if (raises)
fntype = build_exception_variant (fntype, raises);
if (late_return_type_p)
TYPE_HAS_LATE_RETURN_TYPE (fntype) = 1;
return fntype;
}
tree
change_return_type (tree new_ret, tree fntype)
{
tree newtype;
tree args = TYPE_ARG_TYPES (fntype);
tree raises = TYPE_RAISES_EXCEPTIONS (fntype);
tree attrs = TYPE_ATTRIBUTES (fntype);
bool late_return_type_p = TYPE_HAS_LATE_RETURN_TYPE (fntype);
if (new_ret == error_mark_node)
return fntype;
if (same_type_p (new_ret, TREE_TYPE (fntype)))
return fntype;
if (TREE_CODE (fntype) == FUNCTION_TYPE)
{
newtype = build_function_type (new_ret, args);
newtype = apply_memfn_quals (newtype,
type_memfn_quals (fntype),
type_memfn_rqual (fntype));
}
else
newtype = build_method_type_directly
(class_of_this_parm (fntype), new_ret, TREE_CHAIN (args));
if (FUNCTION_REF_QUALIFIED (fntype))
newtype = build_ref_qualified_type (newtype, type_memfn_rqual (fntype));
if (raises)
newtype = build_exception_variant (newtype, raises);
if (attrs)
newtype = cp_build_type_attribute_variant (newtype, attrs);
if (late_return_type_p)
TYPE_HAS_LATE_RETURN_TYPE (newtype) = 1;
return newtype;
}
tree
cp_build_parm_decl (tree fn, tree name, tree type)
{
tree parm = build_decl (input_location,
PARM_DECL, name, type);
DECL_CONTEXT (parm) = fn;
if (!processing_template_decl)
DECL_ARG_TYPE (parm) = type_passed_as (type);
return parm;
}
tree
build_artificial_parm (tree fn, tree name, tree type)
{
tree parm = cp_build_parm_decl (fn, name, type);
DECL_ARTIFICIAL (parm) = 1;
TREE_READONLY (parm) = 1;
return parm;
}
void
maybe_retrofit_in_chrg (tree fn)
{
tree basetype, arg_types, parms, parm, fntype;
if (DECL_HAS_IN_CHARGE_PARM_P (fn))
return;
if (processing_template_decl)
return;
if (DECL_CONSTRUCTOR_P (fn)
&& !CLASSTYPE_VBASECLASSES (DECL_CONTEXT (fn)))
return;
arg_types = TYPE_ARG_TYPES (TREE_TYPE (fn));
basetype = TREE_TYPE (TREE_VALUE (arg_types));
arg_types = TREE_CHAIN (arg_types);
parms = DECL_CHAIN (DECL_ARGUMENTS (fn));
if (CLASSTYPE_VBASECLASSES (DECL_CONTEXT (fn)))
{
parm = build_artificial_parm (fn, vtt_parm_identifier, vtt_parm_type);
DECL_CHAIN (parm) = parms;
parms = parm;
arg_types = hash_tree_chain (vtt_parm_type, arg_types);
DECL_HAS_VTT_PARM_P (fn) = 1;
}
parm = build_artificial_parm (fn, in_charge_identifier, integer_type_node);
DECL_CHAIN (parm) = parms;
parms = parm;
arg_types = hash_tree_chain (integer_type_node, arg_types);
DECL_CHAIN (DECL_ARGUMENTS (fn)) = parms;
fntype = build_method_type_directly (basetype, TREE_TYPE (TREE_TYPE (fn)),
arg_types);
if (TYPE_RAISES_EXCEPTIONS (TREE_TYPE (fn)))
fntype = build_exception_variant (fntype,
TYPE_RAISES_EXCEPTIONS (TREE_TYPE (fn)));
if (TYPE_ATTRIBUTES (TREE_TYPE (fn)))
fntype = (cp_build_type_attribute_variant
(fntype, TYPE_ATTRIBUTES (TREE_TYPE (fn))));
TREE_TYPE (fn) = fntype;
DECL_HAS_IN_CHARGE_PARM_P (fn) = 1;
}
void
grokclassfn (tree ctype, tree function, enum overload_flags flags)
{
tree fn_name = DECL_NAME (function);
SET_DECL_LANGUAGE (function, lang_cplusplus);
if (fn_name == NULL_TREE)
{
error ("name missing for member function");
fn_name = get_identifier ("<anonymous>");
DECL_NAME (function) = fn_name;
}
DECL_CONTEXT (function) = ctype;
if (flags == DTOR_FLAG)
DECL_CXX_DESTRUCTOR_P (function) = 1;
if (flags == DTOR_FLAG || DECL_CONSTRUCTOR_P (function))
maybe_retrofit_in_chrg (function);
}
tree
grok_array_decl (location_t loc, tree array_expr, tree index_exp,
bool decltype_p)
{
tree type;
tree expr;
tree orig_array_expr = array_expr;
tree orig_index_exp = index_exp;
tree overload = NULL_TREE;
if (error_operand_p (array_expr) || error_operand_p (index_exp))
return error_mark_node;
if (processing_template_decl)
{
if (type_dependent_expression_p (array_expr)
|| type_dependent_expression_p (index_exp))
return build_min_nt_loc (loc, ARRAY_REF, array_expr, index_exp,
NULL_TREE, NULL_TREE);
array_expr = build_non_dependent_expr (array_expr);
index_exp = build_non_dependent_expr (index_exp);
}
type = TREE_TYPE (array_expr);
gcc_assert (type);
type = non_reference (type);
if (MAYBE_CLASS_TYPE_P (type) || MAYBE_CLASS_TYPE_P (TREE_TYPE (index_exp)))
{
tsubst_flags_t complain = tf_warning_or_error;
if (decltype_p)
complain |= tf_decltype;
expr = build_new_op (loc, ARRAY_REF, LOOKUP_NORMAL, array_expr,
index_exp, NULL_TREE, &overload, complain);
}
else
{
tree p1, p2, i1, i2;
if (TREE_CODE (type) == ARRAY_TYPE || VECTOR_TYPE_P (type))
p1 = array_expr;
else
p1 = build_expr_type_conversion (WANT_POINTER, array_expr, false);
if (TREE_CODE (TREE_TYPE (index_exp)) == ARRAY_TYPE)
p2 = index_exp;
else
p2 = build_expr_type_conversion (WANT_POINTER, index_exp, false);
i1 = build_expr_type_conversion (WANT_INT | WANT_ENUM, array_expr,
false);
i2 = build_expr_type_conversion (WANT_INT | WANT_ENUM, index_exp,
false);
if ((p1 && i2) && (i1 && p2))
error ("ambiguous conversion for array subscript");
if (p1 && i2)
array_expr = p1, index_exp = i2;
else if (i1 && p2)
array_expr = p2, index_exp = i1;
else
{
error ("invalid types %<%T[%T]%> for array subscript",
type, TREE_TYPE (index_exp));
return error_mark_node;
}
if (array_expr == error_mark_node || index_exp == error_mark_node)
error ("ambiguous conversion for array subscript");
if (TREE_CODE (TREE_TYPE (array_expr)) == POINTER_TYPE)
array_expr = mark_rvalue_use (array_expr);
else
array_expr = mark_lvalue_use_nonread (array_expr);
index_exp = mark_rvalue_use (index_exp);
expr = build_array_ref (input_location, array_expr, index_exp);
}
if (processing_template_decl && expr != error_mark_node)
{
if (overload != NULL_TREE)
return (build_min_non_dep_op_overload
(ARRAY_REF, expr, overload, orig_array_expr, orig_index_exp));
return build_min_non_dep (ARRAY_REF, expr, orig_array_expr, orig_index_exp,
NULL_TREE, NULL_TREE);
}
return expr;
}
tree
delete_sanity (tree exp, tree size, bool doing_vec, int use_global_delete,
tsubst_flags_t complain)
{
tree t, type;
if (exp == error_mark_node)
return exp;
if (processing_template_decl)
{
t = build_min (DELETE_EXPR, void_type_node, exp, size);
DELETE_EXPR_USE_GLOBAL (t) = use_global_delete;
DELETE_EXPR_USE_VEC (t) = doing_vec;
TREE_SIDE_EFFECTS (t) = 1;
return t;
}
if (TREE_CODE (TREE_TYPE (exp)) == ARRAY_TYPE)
warning (0, "deleting array %q#E", exp);
t = build_expr_type_conversion (WANT_POINTER, exp, true);
if (t == NULL_TREE || t == error_mark_node)
{
error ("type %q#T argument given to %<delete%>, expected pointer",
TREE_TYPE (exp));
return error_mark_node;
}
type = TREE_TYPE (t);
if (TREE_CODE (TREE_TYPE (type)) == FUNCTION_TYPE)
{
error ("cannot delete a function.  Only pointer-to-objects are "
"valid arguments to %<delete%>");
return error_mark_node;
}
if (VOID_TYPE_P (TREE_TYPE (type)))
{
warning (OPT_Wdelete_incomplete, "deleting %qT is undefined", type);
doing_vec = 0;
}
if (integer_zerop (t))
return build1 (NOP_EXPR, void_type_node, t);
if (doing_vec)
return build_vec_delete (t, NULL_TREE,
sfk_deleting_destructor,
use_global_delete, complain);
else
return build_delete (type, t, sfk_deleting_destructor,
LOOKUP_NORMAL, use_global_delete,
complain);
}
void
check_member_template (tree tmpl)
{
tree decl;
gcc_assert (TREE_CODE (tmpl) == TEMPLATE_DECL);
decl = DECL_TEMPLATE_RESULT (tmpl);
if (TREE_CODE (decl) == FUNCTION_DECL
|| DECL_ALIAS_TEMPLATE_P (tmpl)
|| (TREE_CODE (decl) == TYPE_DECL
&& MAYBE_CLASS_TYPE_P (TREE_TYPE (decl))))
{
gcc_assert (!current_function_decl || LAMBDA_FUNCTION_P (decl));
gcc_assert (!(TREE_CODE (decl) == FUNCTION_DECL
&& DECL_VIRTUAL_P (decl)));
DECL_IGNORED_P (tmpl) = 1;
}
else if (variable_template_p (tmpl))
;
else
error ("template declaration of %q#D", decl);
}
tree
check_classfn (tree ctype, tree function, tree template_parms)
{
if (DECL_USE_TEMPLATE (function)
&& !(TREE_CODE (function) == TEMPLATE_DECL
&& DECL_TEMPLATE_SPECIALIZATION (function))
&& DECL_MEMBER_TEMPLATE_P (DECL_TI_TEMPLATE (function)))
return NULL_TREE;
if (TREE_CODE (function) == TEMPLATE_DECL)
{
if (template_parms
&& !comp_template_parms (template_parms,
DECL_TEMPLATE_PARMS (function)))
{
error ("template parameter lists provided don%'t match the "
"template parameters of %qD", function);
return error_mark_node;
}
template_parms = DECL_TEMPLATE_PARMS (function);
}
bool is_template = (template_parms != NULL_TREE);
if (DECL_DESTRUCTOR_P (function) && is_template)
{
error ("destructor %qD declared as member template", function);
return error_mark_node;
}
tree pushed_scope = push_scope (ctype);
tree matched = NULL_TREE;
tree fns = get_class_binding (ctype, DECL_NAME (function));
for (ovl_iterator iter (fns); !matched && iter; ++iter)
{
tree fndecl = *iter;
if (is_template != (TREE_CODE (fndecl) == TEMPLATE_DECL))
continue;
if (!DECL_DECLARES_FUNCTION_P (fndecl))
continue;
tree p1 = TYPE_ARG_TYPES (TREE_TYPE (function));
tree p2 = TYPE_ARG_TYPES (TREE_TYPE (fndecl));
if (DECL_STATIC_FUNCTION_P (fndecl)
&& TREE_CODE (TREE_TYPE (function)) == METHOD_TYPE)
p1 = TREE_CHAIN (p1);
if (type_memfn_rqual (TREE_TYPE (function))
!= type_memfn_rqual (TREE_TYPE (fndecl)))
continue;
tree c1 = get_constraints (function);
tree c2 = get_constraints (fndecl);
if (same_type_p (TREE_TYPE (TREE_TYPE (function)),
TREE_TYPE (TREE_TYPE (fndecl)))
&& compparms (p1, p2)
&& !targetm.target_option.function_versions (function, fndecl)
&& (!is_template
|| comp_template_parms (template_parms,
DECL_TEMPLATE_PARMS (fndecl)))
&& equivalent_constraints (c1, c2)
&& (DECL_TEMPLATE_SPECIALIZATION (function)
== DECL_TEMPLATE_SPECIALIZATION (fndecl))
&& (!DECL_TEMPLATE_SPECIALIZATION (function)
|| (DECL_TI_TEMPLATE (function) == DECL_TI_TEMPLATE (fndecl))))
matched = fndecl;
}
if (!matched)
{
if (!COMPLETE_TYPE_P (ctype))
cxx_incomplete_type_error (function, ctype);
else
{
if (DECL_CONV_FN_P (function))
fns = get_class_binding (ctype, conv_op_identifier);
error_at (DECL_SOURCE_LOCATION (function),
"no declaration matches %q#D", function);
if (fns)
print_candidates (fns);
else if (DECL_CONV_FN_P (function))
inform (DECL_SOURCE_LOCATION (function),
"no conversion operators declared");
else
inform (DECL_SOURCE_LOCATION (function),
"no functions named %qD", function);
inform (DECL_SOURCE_LOCATION (TYPE_NAME (ctype)),
"%#qT defined here", ctype);
}
matched = error_mark_node;
}
if (pushed_scope)
pop_scope (pushed_scope);
return matched;
}
void
note_vague_linkage_fn (tree decl)
{
if (processing_template_decl)
return;
DECL_DEFER_OUTPUT (decl) = 1;
vec_safe_push (deferred_fns, decl);
}
void
note_variable_template_instantiation (tree decl)
{
vec_safe_push (pending_statics, decl);
}
void
finish_static_data_member_decl (tree decl,
tree init, bool init_const_expr_p,
tree asmspec_tree,
int flags)
{
DECL_CONTEXT (decl) = current_class_type;
if (! processing_template_decl)
vec_safe_push (pending_statics, decl);
if (LOCAL_CLASS_P (current_class_type)
&& !DECL_TEMPLATE_INSTANTIATION (decl))
permerror (input_location, "local class %q#T shall not have static data member %q#D",
current_class_type, decl);
else
for (tree t = current_class_type; TYPE_P (t);
t = CP_TYPE_CONTEXT (t))
if (TYPE_UNNAMED_P (t))
{
if (permerror (DECL_SOURCE_LOCATION (decl),
"static data member %qD in unnamed class", decl))
inform (DECL_SOURCE_LOCATION (TYPE_NAME (t)),
"unnamed class defined here");
break;
}
DECL_IN_AGGR_P (decl) = 1;
if (TREE_CODE (TREE_TYPE (decl)) == ARRAY_TYPE
&& TYPE_DOMAIN (TREE_TYPE (decl)) == NULL_TREE)
SET_VAR_HAD_UNKNOWN_BOUND (decl);
if (init)
{
tree type = TREE_TYPE (decl) = complete_type (TREE_TYPE (decl));
cp_apply_type_quals_to_decl (cp_type_quals (type), decl);
}
cp_finish_decl (decl, init, init_const_expr_p, asmspec_tree, flags);
}
tree
grokfield (const cp_declarator *declarator,
cp_decl_specifier_seq *declspecs,
tree init, bool init_const_expr_p,
tree asmspec_tree,
tree attrlist)
{
tree value;
const char *asmspec = 0;
int flags;
tree name;
if (init
&& TREE_CODE (init) == TREE_LIST
&& TREE_VALUE (init) == error_mark_node
&& TREE_CHAIN (init) == NULL_TREE)
init = NULL_TREE;
value = grokdeclarator (declarator, declspecs, FIELD, init != 0, &attrlist);
if (! value || value == error_mark_node)
return error_mark_node;
if (TREE_TYPE (value) == error_mark_node)
return value;
if (TREE_CODE (value) == TYPE_DECL && init)
{
error ("typedef %qD is initialized (use decltype instead)", value);
init = NULL_TREE;
}
if (value == void_type_node)
return value;
name = DECL_NAME (value);
if (name != NULL_TREE)
{
if (TREE_CODE (name) == TEMPLATE_ID_EXPR)
{
error ("explicit template argument list not allowed");
return error_mark_node;
}
if (IDENTIFIER_POINTER (name)[0] == '_'
&& id_equal (name, "_vptr"))
error ("member %qD conflicts with virtual function table field name",
value);
}
if (TREE_CODE (value) == TYPE_DECL)
{
DECL_NONLOCAL (value) = 1;
DECL_CONTEXT (value) = current_class_type;
if (attrlist)
{
int attrflags = 0;
if (OVERLOAD_TYPE_P (TREE_TYPE (value))
&& value == TYPE_NAME (TYPE_MAIN_VARIANT (TREE_TYPE (value))))
attrflags = ATTR_FLAG_TYPE_IN_PLACE;
cplus_decl_attributes (&value, attrlist, attrflags);
}
if (decl_spec_seq_has_spec_p (declspecs, ds_typedef)
&& TREE_TYPE (value) != error_mark_node
&& TYPE_NAME (TYPE_MAIN_VARIANT (TREE_TYPE (value))) != value)
set_underlying_type (value);
if (processing_template_decl)
value = push_template_decl (value);
record_locally_defined_typedef (value);
return value;
}
int friendp = decl_spec_seq_has_spec_p (declspecs, ds_friend);
if (!friendp && DECL_IN_AGGR_P (value))
{
error ("%qD is already defined in %qT", value, DECL_CONTEXT (value));
return void_type_node;
}
if (asmspec_tree && asmspec_tree != error_mark_node)
asmspec = TREE_STRING_POINTER (asmspec_tree);
if (init)
{
if (TREE_CODE (value) == FUNCTION_DECL)
{
if (init == ridpointers[(int)RID_DELETE])
{
if (friendp && decl_defined_p (value))
{
error ("redefinition of %q#D", value);
inform (DECL_SOURCE_LOCATION (value),
"%q#D previously defined here", value);
}
else
{
DECL_DELETED_FN (value) = 1;
DECL_DECLARED_INLINE_P (value) = 1;
DECL_INITIAL (value) = error_mark_node;
}
}
else if (init == ridpointers[(int)RID_DEFAULT])
{
if (defaultable_fn_check (value))
{
DECL_DEFAULTED_FN (value) = 1;
DECL_INITIALIZED_IN_CLASS_P (value) = 1;
DECL_DECLARED_INLINE_P (value) = 1;
}
}
else if (TREE_CODE (init) == DEFAULT_ARG)
error ("invalid initializer for member function %qD", value);
else if (TREE_CODE (TREE_TYPE (value)) == METHOD_TYPE)
{
if (integer_zerop (init))
DECL_PURE_VIRTUAL_P (value) = 1;
else if (error_operand_p (init))
; 
else
error ("invalid initializer for member function %qD",
value);
}
else
{
gcc_assert (TREE_CODE (TREE_TYPE (value)) == FUNCTION_TYPE);
if (friendp)
error ("initializer specified for friend function %qD",
value);
else
error ("initializer specified for static member function %qD",
value);
}
}
else if (TREE_CODE (value) == FIELD_DECL)
;
else if (!VAR_P (value))
gcc_unreachable ();
}
if ((TREE_CODE (value) == FUNCTION_DECL
|| TREE_CODE (value) == TEMPLATE_DECL)
&& DECL_CONTEXT (value) != current_class_type)
return value;
if (VAR_P (value))
DECL_CONTEXT (value) = current_class_type;
if (processing_template_decl && VAR_OR_FUNCTION_DECL_P (value))
{
value = push_template_decl (value);
if (error_operand_p (value))
return error_mark_node;
}
if (attrlist)
cplus_decl_attributes (&value, attrlist, 0);
if (init && DIRECT_LIST_INIT_P (init))
flags = LOOKUP_NORMAL;
else
flags = LOOKUP_IMPLICIT;
switch (TREE_CODE (value))
{
case VAR_DECL:
finish_static_data_member_decl (value, init, init_const_expr_p,
asmspec_tree, flags);
return value;
case FIELD_DECL:
if (asmspec)
error ("%<asm%> specifiers are not permitted on non-static data members");
if (DECL_INITIAL (value) == error_mark_node)
init = error_mark_node;
cp_finish_decl (value, init, false,
NULL_TREE, flags);
DECL_IN_AGGR_P (value) = 1;
return value;
case  FUNCTION_DECL:
if (asmspec)
set_user_assembler_name (value, asmspec);
cp_finish_decl (value,
NULL_TREE,
false,
asmspec_tree, flags);
if (DECL_FRIEND_P (value))
return void_type_node;
DECL_IN_AGGR_P (value) = 1;
return value;
default:
gcc_unreachable ();
}
return NULL_TREE;
}
tree
grokbitfield (const cp_declarator *declarator,
cp_decl_specifier_seq *declspecs, tree width, tree init,
tree attrlist)
{
tree value = grokdeclarator (declarator, declspecs, BITFIELD,
init != NULL_TREE, &attrlist);
if (value == error_mark_node)
return NULL_TREE; 
if (TREE_TYPE (value) == error_mark_node)
return value;
if (VOID_TYPE_P (value))
return void_type_node;
if (!INTEGRAL_OR_ENUMERATION_TYPE_P (TREE_TYPE (value))
&& (POINTER_TYPE_P (value)
|| !dependent_type_p (TREE_TYPE (value))))
{
error ("bit-field %qD with non-integral type", value);
return error_mark_node;
}
if (TREE_CODE (value) == TYPE_DECL)
{
error ("cannot declare %qD to be a bit-field type", value);
return NULL_TREE;
}
if (TREE_CODE (value) == FUNCTION_DECL)
{
error ("cannot declare bit-field %qD with function type",
DECL_NAME (value));
return NULL_TREE;
}
if (width && TYPE_WARN_IF_NOT_ALIGN (TREE_TYPE (value)))
{
error ("cannot declare bit-field %qD with %<warn_if_not_aligned%> type",
DECL_NAME (value));
return NULL_TREE;
}
if (DECL_IN_AGGR_P (value))
{
error ("%qD is already defined in the class %qT", value,
DECL_CONTEXT (value));
return void_type_node;
}
if (TREE_STATIC (value))
{
error ("static member %qD cannot be a bit-field", value);
return NULL_TREE;
}
int flags = LOOKUP_IMPLICIT;
if (init && DIRECT_LIST_INIT_P (init))
flags = LOOKUP_NORMAL;
cp_finish_decl (value, init, false, NULL_TREE, flags);
if (width != error_mark_node)
{
if (!type_dependent_expression_p (width)
&& !INTEGRAL_OR_UNSCOPED_ENUMERATION_TYPE_P (TREE_TYPE (width)))
error ("width of bit-field %qD has non-integral type %qT", value,
TREE_TYPE (width));
else
{
DECL_BIT_FIELD_REPRESENTATIVE (value) = width;
SET_DECL_C_BIT_FIELD (value);
}
}
DECL_IN_AGGR_P (value) = 1;
if (attrlist)
cplus_decl_attributes (&value, attrlist, 0);
return value;
}

static bool
is_late_template_attribute (tree attr, tree decl)
{
tree name = get_attribute_name (attr);
tree args = TREE_VALUE (attr);
const struct attribute_spec *spec = lookup_attribute_spec (name);
tree arg;
if (!spec)
return false;
if (is_attribute_p ("weak", name))
return true;
if (TREE_CODE (decl) == TYPE_DECL
&& (is_attribute_p ("unused", name)
|| is_attribute_p ("used", name)))
return false;
if (is_attribute_p ("tls_model", name))
return true;
if (flag_openmp
&& is_attribute_p ("omp declare simd", name))
return true;
if (args && PACK_EXPANSION_P (args))
return true;
for (arg = args; arg; arg = TREE_CHAIN (arg))
{
tree t = TREE_VALUE (arg);
if (arg == args && attribute_takes_identifier_p (name)
&& identifier_p (t))
continue;
if (value_dependent_expression_p (t)
|| type_dependent_expression_p (t))
return true;
}
if (TREE_CODE (decl) == TYPE_DECL
|| TYPE_P (decl)
|| spec->type_required)
{
tree type = TYPE_P (decl) ? decl : TREE_TYPE (decl);
enum tree_code code = TREE_CODE (type);
if (code == TEMPLATE_TYPE_PARM
|| code == BOUND_TEMPLATE_TEMPLATE_PARM
|| code == TYPENAME_TYPE)
return true;
else if (dependent_type_p (type)
&& !is_attribute_p ("abi_tag", name)
&& !is_attribute_p ("deprecated", name)
&& !is_attribute_p ("visibility", name))
return true;
else
return false;
}
else
return false;
}
static tree
splice_template_attributes (tree *attr_p, tree decl)
{
tree *p = attr_p;
tree late_attrs = NULL_TREE;
tree *q = &late_attrs;
if (!p)
return NULL_TREE;
for (; *p; )
{
if (is_late_template_attribute (*p, decl))
{
ATTR_IS_DEPENDENT (*p) = 1;
*q = *p;
*p = TREE_CHAIN (*p);
q = &TREE_CHAIN (*q);
*q = NULL_TREE;
}
else
p = &TREE_CHAIN (*p);
}
return late_attrs;
}
static void
save_template_attributes (tree *attr_p, tree *decl_p, int flags)
{
tree *q;
if (attr_p && *attr_p == error_mark_node)
return;
tree late_attrs = splice_template_attributes (attr_p, *decl_p);
if (!late_attrs)
return;
if (DECL_P (*decl_p))
q = &DECL_ATTRIBUTES (*decl_p);
else
q = &TYPE_ATTRIBUTES (*decl_p);
tree old_attrs = *q;
late_attrs = merge_attributes (late_attrs, *q);
if (*q != late_attrs
&& !DECL_P (*decl_p)
&& !(flags & ATTR_FLAG_TYPE_IN_PLACE))
{
if (!dependent_type_p (*decl_p))
*decl_p = cp_build_type_attribute_variant (*decl_p, late_attrs);
else
{
*decl_p = build_variant_type_copy (*decl_p);
TYPE_ATTRIBUTES (*decl_p) = late_attrs;
}
}
else
*q = late_attrs;
if (!DECL_P (*decl_p) && *decl_p == TYPE_MAIN_VARIANT (*decl_p))
{
tree variant;
for (variant = TYPE_NEXT_VARIANT (*decl_p); variant;
variant = TYPE_NEXT_VARIANT (variant))
{
gcc_assert (TYPE_ATTRIBUTES (variant) == old_attrs);
TYPE_ATTRIBUTES (variant) = TYPE_ATTRIBUTES (*decl_p);
}
}
}
bool
any_dependent_type_attributes_p (tree attrs)
{
for (tree a = attrs; a; a = TREE_CHAIN (a))
if (ATTR_IS_DEPENDENT (a))
{
const attribute_spec *as = lookup_attribute_spec (TREE_PURPOSE (a));
if (as && as->affects_type_identity)
return true;
}
return false;
}
bool
attributes_naming_typedef_ok (tree attrs)
{
for (; attrs; attrs = TREE_CHAIN (attrs))
{
tree name = get_attribute_name (attrs);
if (is_attribute_p ("vector_size", name))
return false;
}
return true;
}
tree
cp_reconstruct_complex_type (tree type, tree bottom)
{
tree inner, outer;
bool late_return_type_p = false;
if (TYPE_PTR_P (type))
{
inner = cp_reconstruct_complex_type (TREE_TYPE (type), bottom);
outer = build_pointer_type_for_mode (inner, TYPE_MODE (type),
TYPE_REF_CAN_ALIAS_ALL (type));
}
else if (TREE_CODE (type) == REFERENCE_TYPE)
{
inner = cp_reconstruct_complex_type (TREE_TYPE (type), bottom);
outer = build_reference_type_for_mode (inner, TYPE_MODE (type),
TYPE_REF_CAN_ALIAS_ALL (type));
}
else if (TREE_CODE (type) == ARRAY_TYPE)
{
inner = cp_reconstruct_complex_type (TREE_TYPE (type), bottom);
outer = build_cplus_array_type (inner, TYPE_DOMAIN (type));
return outer;
}
else if (TREE_CODE (type) == FUNCTION_TYPE)
{
late_return_type_p = TYPE_HAS_LATE_RETURN_TYPE (type);
inner = cp_reconstruct_complex_type (TREE_TYPE (type), bottom);
outer = build_function_type (inner, TYPE_ARG_TYPES (type));
outer = apply_memfn_quals (outer,
type_memfn_quals (type),
type_memfn_rqual (type));
}
else if (TREE_CODE (type) == METHOD_TYPE)
{
late_return_type_p = TYPE_HAS_LATE_RETURN_TYPE (type);
inner = cp_reconstruct_complex_type (TREE_TYPE (type), bottom);
outer
= build_method_type_directly
(class_of_this_parm (type), inner,
TREE_CHAIN (TYPE_ARG_TYPES (type)));
}
else if (TREE_CODE (type) == OFFSET_TYPE)
{
inner = cp_reconstruct_complex_type (TREE_TYPE (type), bottom);
outer = build_offset_type (TYPE_OFFSET_BASETYPE (type), inner);
}
else
return bottom;
if (TYPE_ATTRIBUTES (type))
outer = cp_build_type_attribute_variant (outer, TYPE_ATTRIBUTES (type));
outer = cp_build_qualified_type (outer, cp_type_quals (type));
if (late_return_type_p)
TYPE_HAS_LATE_RETURN_TYPE (outer) = 1;
return outer;
}
static void
cp_check_const_attributes (tree attributes)
{
if (attributes == error_mark_node)
return;
tree attr;
for (attr = attributes; attr; attr = TREE_CHAIN (attr))
{
tree arg;
for (arg = TREE_VALUE (attr); arg; arg = TREE_CHAIN (arg))
{
tree expr = TREE_VALUE (arg);
if (EXPR_P (expr))
TREE_VALUE (arg) = fold_non_dependent_expr (expr);
}
}
}
bool
cp_omp_mappable_type (tree type)
{
if (type == error_mark_node || !COMPLETE_TYPE_P (type))
return false;
while (TREE_CODE (type) == ARRAY_TYPE)
type = TREE_TYPE (type);
if (CLASS_TYPE_P (type) && CLASSTYPE_VTABLES (type))
return false;
if (CLASS_TYPE_P (type))
{
tree field;
for (field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
if (VAR_P (field))
return false;
else if (TREE_CODE (field) == FIELD_DECL
&& !cp_omp_mappable_type (TREE_TYPE (field)))
return false;
}
return true;
}
static tree
find_last_decl (tree decl)
{
tree last_decl = NULL_TREE;
if (tree name = DECL_P (decl) ? DECL_NAME (decl) : NULL_TREE)
{
tree pushed_scope = NULL_TREE;
if (tree ctype = DECL_CONTEXT (decl))
pushed_scope = push_scope (ctype);
last_decl = lookup_name (name);
if (pushed_scope)
pop_scope (pushed_scope);
if (last_decl && BASELINK_P (last_decl))
last_decl = BASELINK_FUNCTIONS (last_decl);
}
if (!last_decl)
return NULL_TREE;
if (DECL_P (last_decl) || TREE_CODE (last_decl) == OVERLOAD)
{
for (lkp_iterator iter (last_decl); iter; ++iter)
{
if (TREE_CODE (*iter) == OVERLOAD)
continue;
if (decls_match (decl, *iter, false))
return *iter;
}
return NULL_TREE;
}
return NULL_TREE;
}
void
cplus_decl_attributes (tree *decl, tree attributes, int flags)
{
if (*decl == NULL_TREE || *decl == void_type_node
|| *decl == error_mark_node)
return;
if (scope_chain->omp_declare_target_attribute
&& ((VAR_P (*decl)
&& (TREE_STATIC (*decl) || DECL_EXTERNAL (*decl)))
|| TREE_CODE (*decl) == FUNCTION_DECL))
{
if (VAR_P (*decl)
&& DECL_CLASS_SCOPE_P (*decl))
error ("%q+D static data member inside of declare target directive",
*decl);
else if (VAR_P (*decl)
&& (processing_template_decl
|| !cp_omp_mappable_type (TREE_TYPE (*decl))))
attributes = tree_cons (get_identifier ("omp declare target implicit"),
NULL_TREE, attributes);
else
attributes = tree_cons (get_identifier ("omp declare target"),
NULL_TREE, attributes);
}
if (processing_template_decl)
{
if (check_for_bare_parameter_packs (attributes))
return;
save_template_attributes (&attributes, decl, flags);
}
cp_check_const_attributes (attributes);
if (TREE_CODE (*decl) == TEMPLATE_DECL)
decl = &DECL_TEMPLATE_RESULT (*decl);
if (TREE_TYPE (*decl) && TYPE_PTRMEMFUNC_P (TREE_TYPE (*decl)))
{
attributes
= decl_attributes (decl, attributes, flags | ATTR_FLAG_FUNCTION_NEXT);
decl_attributes (&TYPE_PTRMEMFUNC_FN_TYPE_RAW (TREE_TYPE (*decl)),
attributes, flags);
}
else
{
tree last_decl = find_last_decl (*decl);
decl_attributes (decl, attributes, flags, last_decl);
}
if (TREE_CODE (*decl) == TYPE_DECL)
SET_IDENTIFIER_TYPE_VALUE (DECL_NAME (*decl), TREE_TYPE (*decl));
if (TREE_DEPRECATED (*decl))
if (tree ti = get_template_info (*decl))
{
tree tmpl = TI_TEMPLATE (ti);
tree pattern = (TYPE_P (*decl) ? TREE_TYPE (tmpl)
: DECL_TEMPLATE_RESULT (tmpl));
if (*decl == pattern)
TREE_DEPRECATED (tmpl) = true;
}
}

static tree
build_anon_union_vars (tree type, tree object)
{
tree main_decl = NULL_TREE;
tree field;
if (TREE_CODE (type) != UNION_TYPE)
{
error ("anonymous struct not inside named type");
return error_mark_node;
}
for (field = TYPE_FIELDS (type);
field != NULL_TREE;
field = DECL_CHAIN (field))
{
tree decl;
tree ref;
if (DECL_ARTIFICIAL (field))
continue;
if (TREE_CODE (field) != FIELD_DECL)
{
permerror (DECL_SOURCE_LOCATION (field),
"%q#D invalid; an anonymous union can only "
"have non-static data members", field);
continue;
}
if (TREE_PRIVATE (field))
permerror (DECL_SOURCE_LOCATION (field),
"private member %q#D in anonymous union", field);
else if (TREE_PROTECTED (field))
permerror (DECL_SOURCE_LOCATION (field),
"protected member %q#D in anonymous union", field);
if (processing_template_decl)
ref = build_min_nt_loc (UNKNOWN_LOCATION, COMPONENT_REF, object,
DECL_NAME (field), NULL_TREE);
else
ref = build_class_member_access_expr (object, field, NULL_TREE,
false, tf_warning_or_error);
if (DECL_NAME (field))
{
tree base;
decl = build_decl (input_location,
VAR_DECL, DECL_NAME (field), TREE_TYPE (field));
DECL_ANON_UNION_VAR_P (decl) = 1;
DECL_ARTIFICIAL (decl) = 1;
base = get_base_address (object);
TREE_PUBLIC (decl) = TREE_PUBLIC (base);
TREE_STATIC (decl) = TREE_STATIC (base);
DECL_EXTERNAL (decl) = DECL_EXTERNAL (base);
SET_DECL_VALUE_EXPR (decl, ref);
DECL_HAS_VALUE_EXPR_P (decl) = 1;
decl = pushdecl (decl);
}
else if (ANON_AGGR_TYPE_P (TREE_TYPE (field)))
decl = build_anon_union_vars (TREE_TYPE (field), ref);
else
decl = 0;
if (main_decl == NULL_TREE)
main_decl = decl;
}
return main_decl;
}
void
finish_anon_union (tree anon_union_decl)
{
tree type;
tree main_decl;
bool public_p;
if (anon_union_decl == error_mark_node)
return;
type = TREE_TYPE (anon_union_decl);
public_p = TREE_PUBLIC (anon_union_decl);
DECL_CONTEXT (anon_union_decl) = DECL_CONTEXT (TYPE_NAME (type));
if (TYPE_FIELDS (type) == NULL_TREE)
return;
if (public_p)
{
error ("namespace-scope anonymous aggregates must be static");
return;
}
main_decl = build_anon_union_vars (type, anon_union_decl);
if (main_decl == error_mark_node)
return;
if (main_decl == NULL_TREE)
{
pedwarn (input_location, 0, "anonymous union with no members");
return;
}
if (!processing_template_decl)
{
DECL_NAME (anon_union_decl) = DECL_NAME (main_decl);
maybe_commonize_var (anon_union_decl);
if (TREE_STATIC (anon_union_decl) || DECL_EXTERNAL (anon_union_decl))
mangle_decl (anon_union_decl);
DECL_NAME (anon_union_decl) = NULL_TREE;
}
pushdecl (anon_union_decl);
cp_finish_decl (anon_union_decl, NULL_TREE, false, NULL_TREE, 0);
}

tree
coerce_new_type (tree type)
{
int e = 0;
tree args = TYPE_ARG_TYPES (type);
gcc_assert (TREE_CODE (type) == FUNCTION_TYPE);
if (!same_type_p (TREE_TYPE (type), ptr_type_node))
{
e = 1;
error ("%<operator new%> must return type %qT", ptr_type_node);
}
if (args && args != void_list_node)
{
if (TREE_PURPOSE (args))
{
error ("the first parameter of %<operator new%> cannot "
"have a default argument");
TREE_PURPOSE (args) = NULL_TREE;
}
if (!same_type_p (TREE_VALUE (args), size_type_node))
{
e = 2;
args = TREE_CHAIN (args);
}
}
else
e = 2;
if (e == 2)
permerror (input_location, "%<operator new%> takes type %<size_t%> (%qT) "
"as first parameter", size_type_node);
switch (e)
{
case 2:
args = tree_cons (NULL_TREE, size_type_node, args);
case 1:
type = build_exception_variant
(build_function_type (ptr_type_node, args),
TYPE_RAISES_EXCEPTIONS (type));
default:;
}
return type;
}
tree
coerce_delete_type (tree type)
{
int e = 0;
tree args = TYPE_ARG_TYPES (type);
gcc_assert (TREE_CODE (type) == FUNCTION_TYPE);
if (!same_type_p (TREE_TYPE (type), void_type_node))
{
e = 1;
error ("%<operator delete%> must return type %qT", void_type_node);
}
if (!args || args == void_list_node
|| !same_type_p (TREE_VALUE (args), ptr_type_node))
{
e = 2;
if (args && args != void_list_node)
args = TREE_CHAIN (args);
error ("%<operator delete%> takes type %qT as first parameter",
ptr_type_node);
}
switch (e)
{
case 2:
args = tree_cons (NULL_TREE, ptr_type_node, args);
case 1:
type = build_exception_variant
(build_function_type (void_type_node, args),
TYPE_RAISES_EXCEPTIONS (type));
default:;
}
return type;
}

static void
mark_vtable_entries (tree decl)
{
tree fnaddr;
unsigned HOST_WIDE_INT idx;
warning_sentinel w(warn_deprecated_decl);
FOR_EACH_CONSTRUCTOR_VALUE (CONSTRUCTOR_ELTS (DECL_INITIAL (decl)),
idx, fnaddr)
{
tree fn;
STRIP_NOPS (fnaddr);
if (TREE_CODE (fnaddr) != ADDR_EXPR
&& TREE_CODE (fnaddr) != FDESC_EXPR)
continue;
fn = TREE_OPERAND (fnaddr, 0);
TREE_ADDRESSABLE (fn) = 1;
if (DECL_THUNK_P (fn))
use_thunk (fn, 0);
input_location = DECL_SOURCE_LOCATION (fn);
mark_used (fn);
}
}
void
comdat_linkage (tree decl)
{
if (flag_weak)
make_decl_one_only (decl, cxx_comdat_group (decl));
else if (TREE_CODE (decl) == FUNCTION_DECL
|| (VAR_P (decl) && DECL_ARTIFICIAL (decl)))
TREE_PUBLIC (decl) = 0;
else
{
if (DECL_INITIAL (decl) == 0
|| DECL_INITIAL (decl) == error_mark_node)
DECL_COMMON (decl) = 1;
else if (EMPTY_CONSTRUCTOR_P (DECL_INITIAL (decl)))
{
DECL_COMMON (decl) = 1;
DECL_INITIAL (decl) = error_mark_node;
}
else if (!DECL_EXPLICIT_INSTANTIATION (decl))
{
DECL_EXTERNAL (decl) = 1;
DECL_NOT_REALLY_EXTERN (decl) = 0;
}
}
if (TREE_PUBLIC (decl))
DECL_COMDAT (decl) = 1;
}
void
maybe_make_one_only (tree decl)
{
if (! flag_weak)
return;
if (!TARGET_WEAK_NOT_IN_ARCHIVE_TOC
|| (! DECL_EXPLICIT_INSTANTIATION (decl)
&& ! DECL_TEMPLATE_SPECIALIZATION (decl)))
{
make_decl_one_only (decl, cxx_comdat_group (decl));
if (VAR_P (decl))
{
varpool_node *node = varpool_node::get_create (decl);
DECL_COMDAT (decl) = 1;
node->forced_by_abi = true;
TREE_USED (decl) = 1;
}
}
}
bool
vague_linkage_p (tree decl)
{
if (!TREE_PUBLIC (decl))
{
if ((DECL_MAYBE_IN_CHARGE_DESTRUCTOR_P (decl)
|| DECL_MAYBE_IN_CHARGE_CONSTRUCTOR_P (decl))
&& !DECL_ABSTRACT_P (decl)
&& DECL_CHAIN (decl)
&& DECL_CLONED_FUNCTION_P (DECL_CHAIN (decl)))
return vague_linkage_p (DECL_CHAIN (decl));
gcc_checking_assert (!DECL_COMDAT (decl));
return false;
}
if (DECL_COMDAT (decl)
|| (TREE_CODE (decl) == FUNCTION_DECL
&& DECL_DECLARED_INLINE_P (decl))
|| (DECL_LANG_SPECIFIC (decl)
&& DECL_TEMPLATE_INSTANTIATION (decl))
|| (VAR_P (decl) && DECL_INLINE_VAR_P (decl)))
return true;
else if (DECL_FUNCTION_SCOPE_P (decl))
return (TREE_STATIC (decl)
&& vague_linkage_p (DECL_CONTEXT (decl)));
else
return false;
}
static void
import_export_class (tree ctype)
{
int import_export = 0;
gcc_assert (at_eof);
if (CLASSTYPE_INTERFACE_KNOWN (ctype))
return;
if (CLASSTYPE_INTERFACE_ONLY (ctype))
return;
if (lookup_attribute ("dllimport", TYPE_ATTRIBUTES (ctype)))
import_export = -1;
else if (lookup_attribute ("dllexport", TYPE_ATTRIBUTES (ctype)))
import_export = 1;
else if (CLASSTYPE_IMPLICIT_INSTANTIATION (ctype)
&& !flag_implicit_templates)
import_export = repo_export_class_p (ctype) ? 1 : -1;
else if (TYPE_POLYMORPHIC_P (ctype))
{
tree method = CLASSTYPE_KEY_METHOD (ctype);
if (method && (flag_weak || ! DECL_DECLARED_INLINE_P (method)))
import_export = (DECL_REALLY_EXTERN (method) ? -1 : 1);
}
if (MULTIPLE_SYMBOL_SPACES && import_export == -1)
import_export = 0;
if (targetm.cxx.import_export_class)
import_export = targetm.cxx.import_export_class (ctype, import_export);
if (import_export)
{
SET_CLASSTYPE_INTERFACE_KNOWN (ctype);
CLASSTYPE_INTERFACE_ONLY (ctype) = (import_export < 0);
}
}
static bool
var_finalized_p (tree var)
{
return varpool_node::get_create (var)->definition;
}
void
mark_needed (tree decl)
{
TREE_USED (decl) = 1;
if (TREE_CODE (decl) == FUNCTION_DECL)
{
struct cgraph_node *node = cgraph_node::get_create (decl);
node->forced_by_abi = true;
tree clone;
FOR_EACH_CLONE (clone, decl)
mark_needed (clone);
}
else if (VAR_P (decl))
{
varpool_node *node = varpool_node::get_create (decl);
node->forced_by_abi = true;
}
}
bool
decl_needed_p (tree decl)
{
gcc_assert (VAR_OR_FUNCTION_DECL_P (decl));
gcc_assert (at_eof);
if (TREE_PUBLIC (decl) && !DECL_COMDAT (decl) && !DECL_REALLY_EXTERN (decl))
return true;
if (flag_keep_inline_dllexport
&& lookup_attribute ("dllexport", DECL_ATTRIBUTES (decl)))
return true;
if (DECL_REALLY_EXTERN (decl)
&& ((TREE_CODE (decl) != FUNCTION_DECL
&& !optimize)
|| (TREE_CODE (decl) == FUNCTION_DECL
&& !opt_for_fn (decl, optimize)))
&& !lookup_attribute ("always_inline", decl))
return false;
if (TREE_USED (decl))
return true;
if (flag_devirtualize
&& TREE_CODE (decl) == FUNCTION_DECL
&& DECL_VIRTUAL_P (decl))
return true;
return false;
}
static bool
maybe_emit_vtables (tree ctype)
{
tree vtbl;
tree primary_vtbl;
int needed = 0;
varpool_node *current = NULL, *last = NULL;
primary_vtbl = CLASSTYPE_VTABLES (ctype);
if (var_finalized_p (primary_vtbl))
return false;
if (TREE_TYPE (primary_vtbl) == void_type_node)
return false;
if (!targetm.cxx.key_method_may_be_inline ())
determine_key_method (ctype);
for (vtbl = CLASSTYPE_VTABLES (ctype); vtbl; vtbl = DECL_CHAIN (vtbl))
{
import_export_decl (vtbl);
if (DECL_NOT_REALLY_EXTERN (vtbl) && decl_needed_p (vtbl))
needed = 1;
}
if (!needed)
{
if (DECL_COMDAT (primary_vtbl)
&& CLASSTYPE_DEBUG_REQUESTED (ctype))
note_debug_info_needed (ctype);
return false;
}
for (vtbl = CLASSTYPE_VTABLES (ctype); vtbl; vtbl = DECL_CHAIN (vtbl))
{
mark_vtable_entries (vtbl);
if (TREE_TYPE (DECL_INITIAL (vtbl)) == 0)
{
vec<tree, va_gc> *cleanups = NULL;
tree expr = store_init_value (vtbl, DECL_INITIAL (vtbl), &cleanups,
LOOKUP_NORMAL);
gcc_assert (!expr && !cleanups);
}
DECL_EXTERNAL (vtbl) = 0;
rest_of_decl_compilation (vtbl, 1, 1);
if (flag_syntax_only)
TREE_ASM_WRITTEN (vtbl) = 1;
else if (DECL_ONE_ONLY (vtbl))
{
current = varpool_node::get_create (vtbl);
if (last)
current->add_to_same_comdat_group (last);
last = current;
}
}
note_debug_info_needed (ctype);
return true;
}
enum { VISIBILITY_ANON = VISIBILITY_INTERNAL+1 };
static tree
min_vis_r (tree *tp, int *walk_subtrees, void *data)
{
int *vis_p = (int *)data;
if (! TYPE_P (*tp))
{
*walk_subtrees = 0;
}
else if (OVERLOAD_TYPE_P (*tp)
&& !TREE_PUBLIC (TYPE_MAIN_DECL (*tp)))
{
*vis_p = VISIBILITY_ANON;
return *tp;
}
else if (CLASS_TYPE_P (*tp)
&& CLASSTYPE_VISIBILITY (*tp) > *vis_p)
*vis_p = CLASSTYPE_VISIBILITY (*tp);
return NULL;
}
static int
type_visibility (tree type)
{
int vis = VISIBILITY_DEFAULT;
cp_walk_tree_without_duplicates (&type, min_vis_r, &vis);
return vis;
}
static void
constrain_visibility (tree decl, int visibility, bool tmpl)
{
if (visibility == VISIBILITY_ANON)
{
if (!DECL_EXTERN_C_P (decl))
{
TREE_PUBLIC (decl) = 0;
DECL_WEAK (decl) = 0;
DECL_COMMON (decl) = 0;
DECL_COMDAT (decl) = false;
if (VAR_OR_FUNCTION_DECL_P (decl))
{
struct symtab_node *snode = symtab_node::get (decl);
if (snode)
snode->set_comdat_group (NULL);
}
DECL_INTERFACE_KNOWN (decl) = 1;
if (DECL_LANG_SPECIFIC (decl))
DECL_NOT_REALLY_EXTERN (decl) = 1;
}
}
else if (visibility > DECL_VISIBILITY (decl)
&& (tmpl || !DECL_VISIBILITY_SPECIFIED (decl)))
{
DECL_VISIBILITY (decl) = (enum symbol_visibility) visibility;
DECL_VISIBILITY_SPECIFIED (decl) = false;
}
}
static void
constrain_visibility_for_template (tree decl, tree targs)
{
tree args = INNERMOST_TEMPLATE_ARGS (targs);
int i;
for (i = TREE_VEC_LENGTH (args); i > 0; --i)
{
int vis = 0;
tree arg = TREE_VEC_ELT (args, i-1);
if (TYPE_P (arg))
vis = type_visibility (arg);
else
{
if (REFERENCE_REF_P (arg))
arg = TREE_OPERAND (arg, 0);
if (TREE_TYPE (arg))
STRIP_NOPS (arg);
if (TREE_CODE (arg) == ADDR_EXPR)
arg = TREE_OPERAND (arg, 0);
if (VAR_OR_FUNCTION_DECL_P (arg))
{
if (! TREE_PUBLIC (arg))
vis = VISIBILITY_ANON;
else
vis = DECL_VISIBILITY (arg);
}
}
if (vis)
constrain_visibility (decl, vis, true);
}
}
void
determine_visibility (tree decl)
{
if (!TREE_PUBLIC (decl))
return;
gcc_assert (!DECL_CLONED_FUNCTION_P (decl));
bool orig_visibility_specified = DECL_VISIBILITY_SPECIFIED (decl);
enum symbol_visibility orig_visibility = DECL_VISIBILITY (decl);
tree template_decl = NULL_TREE;
if (TREE_CODE (decl) == TYPE_DECL)
{
if (CLASS_TYPE_P (TREE_TYPE (decl)))
{
if (CLASSTYPE_USE_TEMPLATE (TREE_TYPE (decl)))
template_decl = decl;
}
else if (TYPE_TEMPLATE_INFO (TREE_TYPE (decl)))
template_decl = decl;
}
else if (DECL_LANG_SPECIFIC (decl) && DECL_USE_TEMPLATE (decl))
template_decl = decl;
tree class_type = NULL_TREE;
if (DECL_CLASS_SCOPE_P (decl))
class_type = DECL_CONTEXT (decl);
else
{
gcc_assert (!VAR_P (decl)
|| !DECL_VTABLE_OR_VTT_P (decl));
if (DECL_FUNCTION_SCOPE_P (decl) && ! DECL_VISIBILITY_SPECIFIED (decl))
{
tree fn = DECL_CONTEXT (decl);
if (DECL_VISIBILITY_SPECIFIED (fn))
{
DECL_VISIBILITY (decl) = DECL_VISIBILITY (fn);
DECL_VISIBILITY_SPECIFIED (decl) = 
DECL_VISIBILITY_SPECIFIED (fn);
}
else
{
if (DECL_CLASS_SCOPE_P (fn))
determine_visibility_from_class (decl, DECL_CONTEXT (fn));
else if (determine_hidden_inline (fn))
{
DECL_VISIBILITY (decl) = default_visibility;
DECL_VISIBILITY_SPECIFIED (decl) =
visibility_options.inpragma;
}
else
{
DECL_VISIBILITY (decl) = DECL_VISIBILITY (fn);
DECL_VISIBILITY_SPECIFIED (decl) =
DECL_VISIBILITY_SPECIFIED (fn);
}
}
template_decl = NULL_TREE;
}
else if (VAR_P (decl) && DECL_TINFO_P (decl)
&& flag_visibility_ms_compat)
{
tree underlying_type = TREE_TYPE (DECL_NAME (decl));
int underlying_vis = type_visibility (underlying_type);
if (underlying_vis == VISIBILITY_ANON
|| (CLASS_TYPE_P (underlying_type)
&& CLASSTYPE_VISIBILITY_SPECIFIED (underlying_type)))
constrain_visibility (decl, underlying_vis, false);
else
DECL_VISIBILITY (decl) = VISIBILITY_DEFAULT;
}
else if (VAR_P (decl) && DECL_TINFO_P (decl))
{
constrain_visibility
(decl, type_visibility (TREE_TYPE (DECL_NAME (decl))), false);
if (TREE_PUBLIC (decl)
&& !DECL_REALLY_EXTERN (decl)
&& CLASS_TYPE_P (TREE_TYPE (DECL_NAME (decl)))
&& !CLASSTYPE_VISIBILITY_SPECIFIED (TREE_TYPE (DECL_NAME (decl))))
targetm.cxx.determine_class_data_visibility (decl);
}
else if (template_decl)
;
else if (! DECL_VISIBILITY_SPECIFIED (decl))
{
if (determine_hidden_inline (decl))
DECL_VISIBILITY (decl) = VISIBILITY_HIDDEN;
else
{
DECL_VISIBILITY (decl) = default_visibility;
DECL_VISIBILITY_SPECIFIED (decl) = visibility_options.inpragma;
}
}
}
if (template_decl)
{
tree tinfo = get_template_info (template_decl);
tree args = TI_ARGS (tinfo);
tree attribs = (TREE_CODE (decl) == TYPE_DECL
? TYPE_ATTRIBUTES (TREE_TYPE (decl))
: DECL_ATTRIBUTES (decl));
if (args != error_mark_node)
{
tree pattern = DECL_TEMPLATE_RESULT (TI_TEMPLATE (tinfo));
if (!DECL_VISIBILITY_SPECIFIED (decl))
{
if (!DECL_VISIBILITY_SPECIFIED (pattern)
&& determine_hidden_inline (decl))
DECL_VISIBILITY (decl) = VISIBILITY_HIDDEN;
else
{
DECL_VISIBILITY (decl) = DECL_VISIBILITY (pattern);
DECL_VISIBILITY_SPECIFIED (decl)
= DECL_VISIBILITY_SPECIFIED (pattern);
}
}
if (args
&& !lookup_attribute ("visibility", attribs))
{
int depth = TMPL_ARGS_DEPTH (args);
if (DECL_VISIBILITY_SPECIFIED (decl))
{
int i;
for (i = 1; i <= depth; ++i)
{
tree lev = TMPL_ARGS_LEVEL (args, i);
constrain_visibility_for_template (decl, lev);
}
}
else if (PRIMARY_TEMPLATE_P (TI_TEMPLATE (tinfo)))
constrain_visibility_for_template (decl, args);
}
}
}
if (class_type)
determine_visibility_from_class (decl, class_type);
if (decl_anon_ns_mem_p (decl))
constrain_visibility (decl, VISIBILITY_ANON, false);
else if (TREE_CODE (decl) != TYPE_DECL)
{
int tvis = type_visibility (TREE_TYPE (decl));
if (tvis == VISIBILITY_ANON
|| ! DECL_VISIBILITY_SPECIFIED (decl))
constrain_visibility (decl, tvis, false);
}
else if (no_linkage_check (TREE_TYPE (decl), true))
constrain_visibility (decl, VISIBILITY_ANON, false);
if ((DECL_VISIBILITY (decl) != orig_visibility
|| DECL_VISIBILITY_SPECIFIED (decl) != orig_visibility_specified)
&& ((VAR_P (decl) && TREE_STATIC (decl))
|| TREE_CODE (decl) == FUNCTION_DECL)
&& DECL_RTL_SET_P (decl))
make_decl_rtl (decl);
}
static void
determine_visibility_from_class (tree decl, tree class_type)
{
if (DECL_VISIBILITY_SPECIFIED (decl))
return;
if (determine_hidden_inline (decl))
DECL_VISIBILITY (decl) = VISIBILITY_HIDDEN;
else
{
DECL_VISIBILITY (decl) = CLASSTYPE_VISIBILITY (class_type);
DECL_VISIBILITY_SPECIFIED (decl)
= CLASSTYPE_VISIBILITY_SPECIFIED (class_type);
}
if (VAR_P (decl)
&& (DECL_TINFO_P (decl)
|| (DECL_VTABLE_OR_VTT_P (decl)
&& !DECL_CONSTRUCTION_VTABLE_P (decl)))
&& TREE_PUBLIC (decl)
&& !DECL_REALLY_EXTERN (decl)
&& !CLASSTYPE_VISIBILITY_SPECIFIED (class_type))
targetm.cxx.determine_class_data_visibility (decl);
}
static bool
determine_hidden_inline (tree decl)
{
return (visibility_options.inlines_hidden
&& !processing_template_decl
&& TREE_CODE (decl) == FUNCTION_DECL
&& DECL_DECLARED_INLINE_P (decl)
&& (! DECL_LANG_SPECIFIC (decl)
|| ! DECL_EXPLICIT_INSTANTIATION (decl)));
}
void
constrain_class_visibility (tree type)
{
tree binfo;
tree t;
int i;
int vis = type_visibility (type);
if (vis == VISIBILITY_ANON
|| DECL_IN_SYSTEM_HEADER (TYPE_MAIN_DECL (type)))
return;
if (CLASSTYPE_VISIBILITY_SPECIFIED (type))
vis = VISIBILITY_INTERNAL;
for (t = TYPE_FIELDS (type); t; t = DECL_CHAIN (t))
if (TREE_CODE (t) == FIELD_DECL && TREE_TYPE (t) != error_mark_node
&& !DECL_ARTIFICIAL (t))
{
tree ftype = strip_pointer_or_array_types (TREE_TYPE (t));
int subvis = type_visibility (ftype);
if (subvis == VISIBILITY_ANON)
{
if (!in_main_input_context())
{
tree nlt = no_linkage_check (ftype, false);
if (nlt)
{
if (same_type_p (TREE_TYPE (t), nlt))
warning (OPT_Wsubobject_linkage, "\
%qT has a field %qD whose type has no linkage",
type, t);
else
warning (OPT_Wsubobject_linkage, "\
%qT has a field %qD whose type depends on the type %qT which has no linkage",
type, t, nlt);
}
else
warning (OPT_Wsubobject_linkage, "\
%qT has a field %qD whose type uses the anonymous namespace",
type, t);
}
}
else if (MAYBE_CLASS_TYPE_P (ftype)
&& vis < VISIBILITY_HIDDEN
&& subvis >= VISIBILITY_HIDDEN)
warning (OPT_Wattributes, "\
%qT declared with greater visibility than the type of its field %qD",
type, t);
}
binfo = TYPE_BINFO (type);
for (i = 0; BINFO_BASE_ITERATE (binfo, i, t); ++i)
{
int subvis = type_visibility (TREE_TYPE (t));
if (subvis == VISIBILITY_ANON)
{
if (!in_main_input_context())
{
tree nlt = no_linkage_check (TREE_TYPE (t), false);
if (nlt)
{
if (same_type_p (TREE_TYPE (t), nlt))
warning (OPT_Wsubobject_linkage, "\
%qT has a base %qT whose type has no linkage",
type, TREE_TYPE (t));
else
warning (OPT_Wsubobject_linkage, "\
%qT has a base %qT whose type depends on the type %qT which has no linkage",
type, TREE_TYPE (t), nlt);
}
else
warning (OPT_Wsubobject_linkage, "\
%qT has a base %qT whose type uses the anonymous namespace",
type, TREE_TYPE (t));
}
}
else if (vis < VISIBILITY_HIDDEN
&& subvis >= VISIBILITY_HIDDEN)
warning (OPT_Wattributes, "\
%qT declared with greater visibility than its base %qT",
type, TREE_TYPE (t));
}
}
static void bt_reset_linkage_1 (binding_entry, void *);
static void bt_reset_linkage_2 (binding_entry, void *);
static void
reset_type_linkage_1 (tree type)
{
set_linkage_according_to_type (type, TYPE_MAIN_DECL (type));
if (CLASS_TYPE_P (type))
binding_table_foreach (CLASSTYPE_NESTED_UTDS (type),
bt_reset_linkage_1, NULL);
}
static void
bt_reset_linkage_1 (binding_entry b, void *)
{
reset_type_linkage_1 (b->type);
}
static void
reset_decl_linkage (tree decl)
{
if (TREE_PUBLIC (decl))
return;
if (DECL_CLONED_FUNCTION_P (decl))
return;
TREE_PUBLIC (decl) = true;
DECL_INTERFACE_KNOWN (decl) = false;
determine_visibility (decl);
tentative_decl_linkage (decl);
}
static void
reset_type_linkage_2 (tree type)
{
if (CLASS_TYPE_P (type))
{
if (tree vt = CLASSTYPE_VTABLES (type))
{
tree name = mangle_vtbl_for_type (type);
DECL_NAME (vt) = name;
SET_DECL_ASSEMBLER_NAME (vt, name);
reset_decl_linkage (vt);
}
if (tree ti = CLASSTYPE_TYPEINFO_VAR (type))
{
tree name = mangle_typeinfo_for_type (type);
DECL_NAME (ti) = name;
SET_DECL_ASSEMBLER_NAME (ti, name);
TREE_TYPE (name) = type;
reset_decl_linkage (ti);
}
for (tree m = TYPE_FIELDS (type); m; m = DECL_CHAIN (m))
{
tree mem = STRIP_TEMPLATE (m);
if (TREE_CODE (mem) == VAR_DECL || TREE_CODE (mem) == FUNCTION_DECL)
reset_decl_linkage (mem);
}
binding_table_foreach (CLASSTYPE_NESTED_UTDS (type),
bt_reset_linkage_2, NULL);
}
}
static void
bt_reset_linkage_2 (binding_entry b, void *)
{
reset_type_linkage_2 (b->type);
}
void
reset_type_linkage (tree type)
{
reset_type_linkage_1 (type);
reset_type_linkage_2 (type);
}
void
tentative_decl_linkage (tree decl)
{
if (DECL_INTERFACE_KNOWN (decl))
;
else if (vague_linkage_p (decl))
{
if (TREE_CODE (decl) == FUNCTION_DECL
&& decl_defined_p (decl))
{
DECL_EXTERNAL (decl) = 1;
DECL_NOT_REALLY_EXTERN (decl) = 1;
note_vague_linkage_fn (decl);
if (DECL_DECLARED_INLINE_P (decl)
&& (!DECL_IMPLICIT_INSTANTIATION (decl)
|| DECL_DEFAULTED_FN (decl)))
{
gcc_assert (TREE_PUBLIC (decl));
comdat_linkage (decl);
DECL_INTERFACE_KNOWN (decl) = 1;
}
}
else if (VAR_P (decl))
maybe_commonize_var (decl);
}
}
void
import_export_decl (tree decl)
{
int emit_p;
bool comdat_p;
bool import_p;
tree class_type = NULL_TREE;
if (DECL_INTERFACE_KNOWN (decl))
return;
gcc_assert (at_eof);
gcc_assert (VAR_OR_FUNCTION_DECL_P (decl));
gcc_assert (TREE_PUBLIC (decl));
if (TREE_CODE (decl) == FUNCTION_DECL)
gcc_assert (DECL_IMPLICIT_INSTANTIATION (decl)
|| DECL_FRIEND_PSEUDO_TEMPLATE_INSTANTIATION (decl)
|| DECL_DECLARED_INLINE_P (decl));
else
gcc_assert (DECL_IMPLICIT_INSTANTIATION (decl)
|| DECL_VTABLE_OR_VTT_P (decl)
|| DECL_TINFO_P (decl));
gcc_assert (!DECL_REALLY_EXTERN (decl));
comdat_p = false;
import_p = false;
emit_p = repo_emit_p (decl);
if (emit_p == 0)
import_p = true;
else if (emit_p == 1)
{
mark_needed (decl);
DECL_EXTERNAL (decl) = 0;
DECL_INTERFACE_KNOWN (decl) = 1;
return;
}
if (import_p)
;
else if (VAR_P (decl) && DECL_VTABLE_OR_VTT_P (decl))
{
class_type = DECL_CONTEXT (decl);
import_export_class (class_type);
if (CLASSTYPE_INTERFACE_KNOWN (class_type)
&& CLASSTYPE_INTERFACE_ONLY (class_type))
import_p = true;
else if ((!flag_weak || TARGET_WEAK_NOT_IN_ARCHIVE_TOC)
&& !CLASSTYPE_USE_TEMPLATE (class_type)
&& CLASSTYPE_KEY_METHOD (class_type)
&& !DECL_DECLARED_INLINE_P (CLASSTYPE_KEY_METHOD (class_type)))
DECL_EXTERNAL (decl) = 0;
else if (CLASSTYPE_INTERFACE_KNOWN (class_type))
{
if (!flag_weak && CLASSTYPE_EXPLICIT_INSTANTIATION (class_type))
DECL_EXTERNAL (decl) = 0;
else
{
if (!CLASSTYPE_KEY_METHOD (class_type)
|| DECL_DECLARED_INLINE_P (CLASSTYPE_KEY_METHOD (class_type))
|| targetm.cxx.class_data_always_comdat ())
{
comdat_p = true;
mark_needed (decl);
}
}
}
else if (!flag_implicit_templates
&& CLASSTYPE_IMPLICIT_INSTANTIATION (class_type))
import_p = true;
else
comdat_p = true;
}
else if (VAR_P (decl) && DECL_TINFO_P (decl))
{
tree type = TREE_TYPE (DECL_NAME (decl));
if (CLASS_TYPE_P (type))
{
class_type = type;
import_export_class (type);
if (CLASSTYPE_INTERFACE_KNOWN (type)
&& TYPE_POLYMORPHIC_P (type)
&& CLASSTYPE_INTERFACE_ONLY (type)
&& flag_rtti)
import_p = true;
else
{
if (CLASSTYPE_INTERFACE_KNOWN (type)
&& !CLASSTYPE_INTERFACE_ONLY (type))
{
comdat_p = (targetm.cxx.class_data_always_comdat ()
|| (CLASSTYPE_KEY_METHOD (type)
&& DECL_DECLARED_INLINE_P (CLASSTYPE_KEY_METHOD (type))));
mark_needed (decl);
if (!flag_weak)
{
comdat_p = false;
DECL_EXTERNAL (decl) = 0;
}
}
else
comdat_p = true;
}
}
else
comdat_p = true;
}
else if (DECL_TEMPLOID_INSTANTIATION (decl))
{
if ((flag_implicit_templates
&& !flag_use_repository)
|| (flag_implicit_inline_templates
&& TREE_CODE (decl) == FUNCTION_DECL
&& DECL_DECLARED_INLINE_P (decl)))
comdat_p = true;
else
import_p = true;
}
else if (DECL_FUNCTION_MEMBER_P (decl))
{
if (!DECL_DECLARED_INLINE_P (decl))
{
tree ctype = DECL_CONTEXT (decl);
import_export_class (ctype);
if (CLASSTYPE_INTERFACE_KNOWN (ctype))
{
DECL_NOT_REALLY_EXTERN (decl)
= ! (CLASSTYPE_INTERFACE_ONLY (ctype)
|| (DECL_DECLARED_INLINE_P (decl)
&& ! flag_implement_inlines
&& !DECL_VINDEX (decl)));
if (!DECL_NOT_REALLY_EXTERN (decl))
DECL_EXTERNAL (decl) = 1;
if (DECL_ARTIFICIAL (decl) && flag_weak)
comdat_p = true;
else
maybe_make_one_only (decl);
}
}
else
comdat_p = true;
}
else
comdat_p = true;
if (import_p)
{
DECL_EXTERNAL (decl) = 1;
DECL_NOT_REALLY_EXTERN (decl) = 0;
}
else if (comdat_p)
{
comdat_linkage (decl);
}
DECL_INTERFACE_KNOWN (decl) = 1;
}
tree
build_cleanup (tree decl)
{
tree clean = cxx_maybe_build_cleanup (decl, tf_warning_or_error);
gcc_assert (clean != NULL_TREE);
return clean;
}
tree
get_guard (tree decl)
{
tree sname;
tree guard;
sname = mangle_guard_variable (decl);
guard = get_global_binding (sname);
if (! guard)
{
tree guard_type;
guard_type = targetm.cxx.guard_type ();
guard = build_decl (DECL_SOURCE_LOCATION (decl),
VAR_DECL, sname, guard_type);
TREE_PUBLIC (guard) = TREE_PUBLIC (decl);
TREE_STATIC (guard) = TREE_STATIC (decl);
DECL_COMMON (guard) = DECL_COMMON (decl);
DECL_COMDAT (guard) = DECL_COMDAT (decl);
CP_DECL_THREAD_LOCAL_P (guard) = CP_DECL_THREAD_LOCAL_P (decl);
set_decl_tls_model (guard, DECL_TLS_MODEL (decl));
if (DECL_ONE_ONLY (decl))
make_decl_one_only (guard, cxx_comdat_group (guard));
if (TREE_PUBLIC (decl))
DECL_WEAK (guard) = DECL_WEAK (decl);
DECL_VISIBILITY (guard) = DECL_VISIBILITY (decl);
DECL_VISIBILITY_SPECIFIED (guard) = DECL_VISIBILITY_SPECIFIED (decl);
DECL_ARTIFICIAL (guard) = 1;
DECL_IGNORED_P (guard) = 1;
TREE_USED (guard) = 1;
pushdecl_top_level_and_finish (guard, NULL_TREE);
}
return guard;
}
static tree
build_atomic_load_byte (tree src, HOST_WIDE_INT model)
{
tree ptr_type = build_pointer_type (char_type_node);
tree mem_model = build_int_cst (integer_type_node, model);
tree t, addr, val;
unsigned int size;
int fncode;
size = tree_to_uhwi (TYPE_SIZE_UNIT (char_type_node));
fncode = BUILT_IN_ATOMIC_LOAD_N + exact_log2 (size) + 1;
t = builtin_decl_implicit ((enum built_in_function) fncode);
addr = build1 (ADDR_EXPR, ptr_type, src);
val = build_call_expr (t, 2, addr, mem_model);
return val;
}
static tree
get_guard_bits (tree guard)
{
if (!targetm.cxx.guard_mask_bit ())
{
guard = build1 (ADDR_EXPR,
build_pointer_type (TREE_TYPE (guard)),
guard);
guard = build1 (NOP_EXPR,
build_pointer_type (char_type_node),
guard);
guard = build1 (INDIRECT_REF, char_type_node, guard);
}
return guard;
}
tree
get_guard_cond (tree guard, bool thread_safe)
{
tree guard_value;
if (!thread_safe)
guard = get_guard_bits (guard);
else
guard = build_atomic_load_byte (guard, MEMMODEL_ACQUIRE);
if (targetm.cxx.guard_mask_bit ())
{
guard_value = integer_one_node;
if (!same_type_p (TREE_TYPE (guard_value), TREE_TYPE (guard)))
guard_value = fold_convert (TREE_TYPE (guard), guard_value);
guard = cp_build_binary_op (input_location,
BIT_AND_EXPR, guard, guard_value,
tf_warning_or_error);
}
guard_value = integer_zero_node;
if (!same_type_p (TREE_TYPE (guard_value), TREE_TYPE (guard)))
guard_value = fold_convert (TREE_TYPE (guard), guard_value);
return cp_build_binary_op (input_location,
EQ_EXPR, guard, guard_value,
tf_warning_or_error);
}
tree
set_guard (tree guard)
{
tree guard_init;
guard = get_guard_bits (guard);
guard_init = integer_one_node;
if (!same_type_p (TREE_TYPE (guard_init), TREE_TYPE (guard)))
guard_init = fold_convert (TREE_TYPE (guard), guard_init);
return cp_build_modify_expr (input_location, guard, NOP_EXPR, guard_init,
tf_warning_or_error);
}
static bool
var_defined_without_dynamic_init (tree var)
{
if (DECL_EXTERNAL (var))
return false;
if (TYPE_HAS_NONTRIVIAL_DESTRUCTOR (TREE_TYPE (var)))
return false;
if (!DECL_INITIALIZED_P (var))
return false;
return (!DECL_NONTRIVIALLY_INITIALIZED_P (var)
|| DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P (var));
}
static bool
var_needs_tls_wrapper (tree var)
{
return (!error_operand_p (var)
&& CP_DECL_THREAD_LOCAL_P (var)
&& !DECL_GNU_TLS_P (var)
&& !DECL_FUNCTION_SCOPE_P (var)
&& !var_defined_without_dynamic_init (var));
}
static tree
get_local_tls_init_fn (void)
{
tree sname = get_identifier ("__tls_init");
tree fn = get_global_binding (sname);
if (!fn)
{
fn = build_lang_decl (FUNCTION_DECL, sname,
build_function_type (void_type_node,
void_list_node));
SET_DECL_LANGUAGE (fn, lang_c);
TREE_PUBLIC (fn) = false;
DECL_ARTIFICIAL (fn) = true;
mark_used (fn);
set_global_binding (fn);
}
return fn;
}
static tree
get_tls_init_fn (tree var)
{
if (!var_needs_tls_wrapper (var))
return NULL_TREE;
if (!flag_extern_tls_init && DECL_EXTERNAL (var))
return NULL_TREE;
if (!TREE_PUBLIC (var) || !TARGET_SUPPORTS_ALIASES)
return get_local_tls_init_fn ();
tree sname = mangle_tls_init_fn (var);
tree fn = get_global_binding (sname);
if (!fn)
{
fn = build_lang_decl (FUNCTION_DECL, sname,
build_function_type (void_type_node,
void_list_node));
SET_DECL_LANGUAGE (fn, lang_c);
TREE_PUBLIC (fn) = TREE_PUBLIC (var);
DECL_ARTIFICIAL (fn) = true;
DECL_COMDAT (fn) = DECL_COMDAT (var);
DECL_EXTERNAL (fn) = DECL_EXTERNAL (var);
if (DECL_ONE_ONLY (var))
make_decl_one_only (fn, cxx_comdat_group (fn));
if (TREE_PUBLIC (var))
{
tree obtype = strip_array_types (non_reference (TREE_TYPE (var)));
if ((!TYPE_NEEDS_CONSTRUCTING (obtype)
|| TYPE_HAS_CONSTEXPR_CTOR (obtype)
|| TYPE_HAS_TRIVIAL_DFLT (obtype))
&& TYPE_HAS_TRIVIAL_DESTRUCTOR (obtype)
&& DECL_EXTERNAL (var))
declare_weak (fn);
else
DECL_WEAK (fn) = DECL_WEAK (var);
}
DECL_VISIBILITY (fn) = DECL_VISIBILITY (var);
DECL_VISIBILITY_SPECIFIED (fn) = DECL_VISIBILITY_SPECIFIED (var);
DECL_DLLIMPORT_P (fn) = DECL_DLLIMPORT_P (var);
DECL_IGNORED_P (fn) = 1;
mark_used (fn);
DECL_BEFRIENDING_CLASSES (fn) = var;
set_global_binding (fn);
}
return fn;
}
tree
get_tls_wrapper_fn (tree var)
{
if (!var_needs_tls_wrapper (var))
return NULL_TREE;
tree sname = mangle_tls_wrapper_fn (var);
tree fn = get_global_binding (sname);
if (!fn)
{
tree type = non_reference (TREE_TYPE (var));
type = build_reference_type (type);
tree fntype = build_function_type (type, void_list_node);
fn = build_lang_decl (FUNCTION_DECL, sname, fntype);
SET_DECL_LANGUAGE (fn, lang_c);
TREE_PUBLIC (fn) = TREE_PUBLIC (var);
DECL_ARTIFICIAL (fn) = true;
DECL_IGNORED_P (fn) = 1;
DECL_DECLARED_INLINE_P (fn) = true;
if (TREE_PUBLIC (var))
{
comdat_linkage (fn);
#ifdef HAVE_GAS_HIDDEN
DECL_VISIBILITY (fn) = VISIBILITY_INTERNAL;
DECL_VISIBILITY_SPECIFIED (fn) = true;
#endif
}
if (!TREE_PUBLIC (fn))
DECL_INTERFACE_KNOWN (fn) = true;
mark_used (fn);
note_vague_linkage_fn (fn);
#if 0
DECL_PURE_P (fn) = true;
#endif
DECL_BEFRIENDING_CLASSES (fn) = var;
set_global_binding (fn);
}
return fn;
}
static void
generate_tls_wrapper (tree fn)
{
tree var = DECL_BEFRIENDING_CLASSES (fn);
start_preparsed_function (fn, NULL_TREE, SF_DEFAULT | SF_PRE_PARSED);
tree body = begin_function_body ();
if (tree init_fn = get_tls_init_fn (var))
{
tree if_stmt = NULL_TREE;
if (lookup_attribute ("weak", DECL_ATTRIBUTES (init_fn)))
{
if_stmt = begin_if_stmt ();
tree addr = cp_build_addr_expr (init_fn, tf_warning_or_error);
tree cond = cp_build_binary_op (DECL_SOURCE_LOCATION (var),
NE_EXPR, addr, nullptr_node,
tf_warning_or_error);
finish_if_stmt_cond (cond, if_stmt);
}
finish_expr_stmt (build_cxx_call
(init_fn, 0, NULL, tf_warning_or_error));
if (if_stmt)
{
finish_then_clause (if_stmt);
finish_if_stmt (if_stmt);
}
}
else
TREE_READONLY (fn) = true;
finish_return_stmt (convert_from_reference (var));
finish_function_body (body);
expand_or_defer_fn (finish_function (false));
}
static tree
start_objects (int method_type, int initp)
{
tree body;
tree fndecl;
char type[14];
if (initp != DEFAULT_INIT_PRIORITY)
{
char joiner;
#ifdef JOINER
joiner = JOINER;
#else
joiner = '_';
#endif
sprintf (type, "sub_%c%c%.5u", method_type, joiner, initp);
}
else
sprintf (type, "sub_%c", method_type);
fndecl = build_lang_decl (FUNCTION_DECL,
get_file_function_name (type),
build_function_type_list (void_type_node,
NULL_TREE));
start_preparsed_function (fndecl, NULL_TREE, SF_PRE_PARSED);
TREE_PUBLIC (current_function_decl) = 0;
DECL_ARTIFICIAL (current_function_decl) = 1;
TREE_USED (current_function_decl) = 1;
if (method_type == 'I')
DECL_GLOBAL_CTOR_P (current_function_decl) = 1;
else
DECL_GLOBAL_DTOR_P (current_function_decl) = 1;
body = begin_compound_stmt (BCS_FN_BODY);
return body;
}
static void
finish_objects (int method_type, int initp, tree body)
{
tree fn;
finish_compound_stmt (body);
fn = finish_function (false);
if (method_type == 'I')
{
DECL_STATIC_CONSTRUCTOR (fn) = 1;
decl_init_priority_insert (fn, initp);
}
else
{
DECL_STATIC_DESTRUCTOR (fn) = 1;
decl_fini_priority_insert (fn, initp);
}
expand_or_defer_fn (fn);
}
#define INITIALIZE_P_IDENTIFIER "__initialize_p"
#define PRIORITY_IDENTIFIER "__priority"
#define SSDF_IDENTIFIER "__static_initialization_and_destruction"
static GTY(()) tree initialize_p_decl;
static GTY(()) tree priority_decl;
static GTY(()) tree ssdf_decl;
static GTY(()) vec<tree, va_gc> *ssdf_decls;
static splay_tree priority_info_map;
static tree
start_static_storage_duration_function (unsigned count)
{
tree type;
tree body;
char id[sizeof (SSDF_IDENTIFIER) + 1  + 32];
sprintf (id, "%s_%u", SSDF_IDENTIFIER, count);
type = build_function_type_list (void_type_node,
integer_type_node, integer_type_node,
NULL_TREE);
ssdf_decl = build_lang_decl (FUNCTION_DECL,
get_identifier (id),
type);
TREE_PUBLIC (ssdf_decl) = 0;
DECL_ARTIFICIAL (ssdf_decl) = 1;
if (!ssdf_decls)
{
vec_alloc (ssdf_decls, 32);
priority_info_map = splay_tree_new (splay_tree_compare_ints,
0,
(splay_tree_delete_value_fn)
(void (*) (void)) free);
get_priority_info (DEFAULT_INIT_PRIORITY);
}
vec_safe_push (ssdf_decls, ssdf_decl);
initialize_p_decl = cp_build_parm_decl
(ssdf_decl, get_identifier (INITIALIZE_P_IDENTIFIER), integer_type_node);
TREE_USED (initialize_p_decl) = 1;
priority_decl = cp_build_parm_decl
(ssdf_decl, get_identifier (PRIORITY_IDENTIFIER), integer_type_node);
TREE_USED (priority_decl) = 1;
DECL_CHAIN (initialize_p_decl) = priority_decl;
DECL_ARGUMENTS (ssdf_decl) = initialize_p_decl;
pushdecl (ssdf_decl);
start_preparsed_function (ssdf_decl,
NULL_TREE,
SF_PRE_PARSED);
body = begin_compound_stmt (BCS_FN_BODY);
return body;
}
static void
finish_static_storage_duration_function (tree body)
{
finish_compound_stmt (body);
expand_or_defer_fn (finish_function (false));
}
static priority_info
get_priority_info (int priority)
{
priority_info pi;
splay_tree_node n;
n = splay_tree_lookup (priority_info_map,
(splay_tree_key) priority);
if (!n)
{
pi = XNEW (struct priority_info_s);
pi->initializations_p = 0;
pi->destructions_p = 0;
splay_tree_insert (priority_info_map,
(splay_tree_key) priority,
(splay_tree_value) pi);
}
else
pi = (priority_info) n->value;
return pi;
}
#define DECL_EFFECTIVE_INIT_PRIORITY(decl)				      \
((!DECL_HAS_INIT_PRIORITY_P (decl) || DECL_INIT_PRIORITY (decl) == 0) \
? DEFAULT_INIT_PRIORITY : DECL_INIT_PRIORITY (decl))
#define NEEDS_GUARD_P(decl) (TREE_PUBLIC (decl) && (DECL_COMMON (decl)      \
|| DECL_ONE_ONLY (decl) \
|| DECL_WEAK (decl)))
static tree 
fix_temporary_vars_context_r (tree *node,
int  * ,
void * )
{
gcc_assert (current_function_decl);
if (TREE_CODE (*node) == BIND_EXPR)
{
tree var;
for (var = BIND_EXPR_VARS (*node); var; var = DECL_CHAIN (var))
if (VAR_P (var)
&& !DECL_NAME (var)
&& DECL_ARTIFICIAL (var)
&& !DECL_CONTEXT (var))
DECL_CONTEXT (var) = current_function_decl;
}
return NULL_TREE;
}
static void
one_static_initialization_or_destruction (tree decl, tree init, bool initp)
{
tree guard_if_stmt = NULL_TREE;
tree guard;
if (!initp
&& TYPE_HAS_TRIVIAL_DESTRUCTOR (TREE_TYPE (decl)))
return;
input_location = DECL_SOURCE_LOCATION (decl);
cp_walk_tree_without_duplicates (&init,
fix_temporary_vars_context_r,
NULL);
if (member_p (decl))
{
DECL_CONTEXT (current_function_decl) = DECL_CONTEXT (decl);
DECL_STATIC_FUNCTION_P (current_function_decl) = 1;
}
guard = NULL_TREE;
if (NEEDS_GUARD_P (decl))
{
tree guard_cond;
guard = get_guard (decl);
if (flag_use_cxa_atexit)
{
gcc_assert (initp);
guard_cond = get_guard_cond (guard, false);
}
else if (initp)
guard_cond
= cp_build_binary_op (input_location,
EQ_EXPR,
cp_build_unary_op (PREINCREMENT_EXPR,
guard,
true,
tf_warning_or_error),
integer_one_node,
tf_warning_or_error);
else
guard_cond
= cp_build_binary_op (input_location,
EQ_EXPR,
cp_build_unary_op (PREDECREMENT_EXPR,
guard,
true,
tf_warning_or_error),
integer_zero_node,
tf_warning_or_error);
guard_if_stmt = begin_if_stmt ();
finish_if_stmt_cond (guard_cond, guard_if_stmt);
}
if (guard && initp && flag_use_cxa_atexit)
finish_expr_stmt (set_guard (guard));
if (initp)
{
if (init)
{
finish_expr_stmt (init);
if (sanitize_flags_p (SANITIZE_ADDRESS, decl))
{
varpool_node *vnode = varpool_node::get (decl);
if (vnode)
vnode->dynamically_initialized = 1;
}
}
if (flag_use_cxa_atexit)
finish_expr_stmt (register_dtor_fn (decl));
}
else
finish_expr_stmt (build_cleanup (decl));
if (guard)
{
finish_then_clause (guard_if_stmt);
finish_if_stmt (guard_if_stmt);
}
DECL_CONTEXT (current_function_decl) = NULL_TREE;
DECL_STATIC_FUNCTION_P (current_function_decl) = 0;
}
static void
do_static_initialization_or_destruction (tree vars, bool initp)
{
tree node, init_if_stmt, cond;
init_if_stmt = begin_if_stmt ();
cond = initp ? integer_one_node : integer_zero_node;
cond = cp_build_binary_op (input_location,
EQ_EXPR,
initialize_p_decl,
cond,
tf_warning_or_error);
finish_if_stmt_cond (cond, init_if_stmt);
if (initp && (flag_sanitize & SANITIZE_ADDRESS))
finish_expr_stmt (asan_dynamic_init_call (false));
node = vars;
do {
tree decl = TREE_VALUE (node);
tree priority_if_stmt;
int priority;
priority_info pi;
if (!initp && TYPE_HAS_TRIVIAL_DESTRUCTOR (TREE_TYPE (decl)))
{
node = TREE_CHAIN (node);
continue;
}
priority = DECL_EFFECTIVE_INIT_PRIORITY (decl);
pi = get_priority_info (priority);
if (initp)
pi->initializations_p = 1;
else
pi->destructions_p = 1;
priority_if_stmt = begin_if_stmt ();
cond = cp_build_binary_op (input_location,
EQ_EXPR,
priority_decl,
build_int_cst (NULL_TREE, priority),
tf_warning_or_error);
finish_if_stmt_cond (cond, priority_if_stmt);
for (; node
&& DECL_EFFECTIVE_INIT_PRIORITY (TREE_VALUE (node)) == priority;
node = TREE_CHAIN (node))
one_static_initialization_or_destruction (TREE_VALUE (node),
TREE_PURPOSE (node), initp);
finish_then_clause (priority_if_stmt);
finish_if_stmt (priority_if_stmt);
} while (node);
if (initp && (flag_sanitize & SANITIZE_ADDRESS))
finish_expr_stmt (asan_dynamic_init_call (true));
finish_then_clause (init_if_stmt);
finish_if_stmt (init_if_stmt);
}
static tree
prune_vars_needing_no_initialization (tree *vars)
{
tree *var = vars;
tree result = NULL_TREE;
while (*var)
{
tree t = *var;
tree decl = TREE_VALUE (t);
tree init = TREE_PURPOSE (t);
if (error_operand_p (decl))
{
var = &TREE_CHAIN (t);
continue;
}
gcc_assert (VAR_P (decl));
if (DECL_EXTERNAL (decl))
{
var = &TREE_CHAIN (t);
continue;
}
if (init && TREE_CODE (init) == TREE_LIST
&& value_member (error_mark_node, init))
{
var = &TREE_CHAIN (t);
continue;
}
*var = TREE_CHAIN (t);
TREE_CHAIN (t) = result;
result = t;
}
return result;
}
static void
write_out_vars (tree vars)
{
tree v;
for (v = vars; v; v = TREE_CHAIN (v))
{
tree var = TREE_VALUE (v);
if (!var_finalized_p (var))
{
import_export_decl (var);
rest_of_decl_compilation (var, 1, 1);
}
}
}
static void
generate_ctor_or_dtor_function (bool constructor_p, int priority,
location_t *locus)
{
char function_key;
tree fndecl;
tree body;
size_t i;
input_location = *locus;
function_key = constructor_p ? 'I' : 'D';
body = NULL_TREE;
if (c_dialect_objc () && (priority == DEFAULT_INIT_PRIORITY)
&& constructor_p && objc_static_init_needed_p ())
{
body = start_objects (function_key, priority);
objc_generate_static_init_call (NULL_TREE);
}
FOR_EACH_VEC_SAFE_ELT (ssdf_decls, i, fndecl)
{
if (! (flags_from_decl_or_type (fndecl) & (ECF_CONST | ECF_PURE)))
{
tree call;
if (! body)
body = start_objects (function_key, priority);
call = cp_build_function_call_nary (fndecl, tf_warning_or_error,
build_int_cst (NULL_TREE,
constructor_p),
build_int_cst (NULL_TREE,
priority),
NULL_TREE);
finish_expr_stmt (call);
}
}
if (body)
finish_objects (function_key, priority, body);
}
static int
generate_ctor_and_dtor_functions_for_priority (splay_tree_node n, void * data)
{
location_t *locus = (location_t *) data;
int priority = (int) n->key;
priority_info pi = (priority_info) n->value;
if (pi->initializations_p)
generate_ctor_or_dtor_function (true, priority, locus);
if (pi->destructions_p)
generate_ctor_or_dtor_function (false, priority, locus);
return 0;
}
static int
cpp_check (tree t, cpp_operation op)
{
switch (op)
{
case HAS_DEPENDENT_TEMPLATE_ARGS:
{
tree ti = CLASSTYPE_TEMPLATE_INFO (t);
if (!ti)
return 0;
++processing_template_decl;
const bool dep = any_dependent_template_arguments_p (TI_ARGS (ti));
--processing_template_decl;
return dep;
}
case IS_ABSTRACT:
return DECL_PURE_VIRTUAL_P (t);
case IS_CONSTRUCTOR:
return DECL_CONSTRUCTOR_P (t);
case IS_DESTRUCTOR:
return DECL_DESTRUCTOR_P (t);
case IS_COPY_CONSTRUCTOR:
return DECL_COPY_CONSTRUCTOR_P (t);
case IS_MOVE_CONSTRUCTOR:
return DECL_MOVE_CONSTRUCTOR_P (t);
case IS_TEMPLATE:
return TREE_CODE (t) == TEMPLATE_DECL;
case IS_TRIVIAL:
return trivial_type_p (t);
default:
return 0;
}
}
static void 
collect_source_refs (tree namespc) 
{
for (tree t = NAMESPACE_LEVEL (namespc)->names; t; t = TREE_CHAIN (t))
if (DECL_IS_BUILTIN (t))
;
else if (TREE_CODE (t) == NAMESPACE_DECL && !DECL_NAMESPACE_ALIAS (t))
collect_source_refs (t);
else
collect_source_ref (DECL_SOURCE_FILE (t));
}
static void
collect_ada_namespace (tree namespc, const char *source_file)
{
tree decl = NAMESPACE_LEVEL (namespc)->names;
collect_ada_nodes (decl, source_file);
for (; decl; decl = TREE_CHAIN (decl))
if (TREE_CODE (decl) == NAMESPACE_DECL && !DECL_NAMESPACE_ALIAS (decl))
collect_ada_namespace (decl, source_file);
}
bool
decl_defined_p (tree decl)
{
if (TREE_CODE (decl) == FUNCTION_DECL)
return (DECL_INITIAL (decl) != NULL_TREE
|| (DECL_FRIEND_PSEUDO_TEMPLATE_INSTANTIATION (decl)
&& DECL_INITIAL (DECL_TEMPLATE_RESULT
(DECL_TI_TEMPLATE (decl)))));
else
{
gcc_assert (VAR_P (decl));
return !DECL_EXTERNAL (decl);
}
}
bool
decl_constant_var_p (tree decl)
{
if (!decl_maybe_constant_var_p (decl))
return false;
maybe_instantiate_decl (decl);
return DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P (decl);
}
bool
decl_maybe_constant_var_p (tree decl)
{
tree type = TREE_TYPE (decl);
if (!VAR_P (decl))
return false;
if (DECL_DECLARED_CONSTEXPR_P (decl))
return true;
if (DECL_HAS_VALUE_EXPR_P (decl))
return false;
if (TREE_CODE (type) == REFERENCE_TYPE)
;
else if (CP_TYPE_CONST_NON_VOLATILE_P (type)
&& INTEGRAL_OR_ENUMERATION_TYPE_P (type))
;
else
return false;
if (DECL_INITIAL (decl)
&& !DECL_INITIALIZED_BY_CONSTANT_EXPRESSION_P (decl))
return false;
else
return true;
}
void
no_linkage_error (tree decl)
{
if (cxx_dialect >= cxx11 && decl_defined_p (decl))
return;
tree t = no_linkage_check (TREE_TYPE (decl), false);
if (t == NULL_TREE)
;
else if (CLASS_TYPE_P (t) && TYPE_BEING_DEFINED (t))
vec_safe_push (no_linkage_decls, decl);
else if (TYPE_UNNAMED_P (t))
{
bool d = false;
if (cxx_dialect >= cxx11)
d = permerror (DECL_SOURCE_LOCATION (decl), "%q#D, declared using "
"unnamed type, is used but never defined", decl);
else if (DECL_EXTERN_C_P (decl))
;
else if (VAR_P (decl))
d = warning_at (DECL_SOURCE_LOCATION (decl), 0, "unnamed type "
"with no linkage used to declare variable %q#D with "
"linkage", decl);
else
d = permerror (DECL_SOURCE_LOCATION (decl), "unnamed type with no "
"linkage used to declare function %q#D with linkage",
decl);
if (d && is_typedef_decl (TYPE_NAME (t)))
inform (DECL_SOURCE_LOCATION (TYPE_NAME (t)), "%q#D does not refer "
"to the unqualified type, so it is not used for linkage",
TYPE_NAME (t));
}
else if (cxx_dialect >= cxx11)
{
if (VAR_P (decl) || !DECL_PURE_VIRTUAL_P (decl))
permerror (DECL_SOURCE_LOCATION (decl),
"%q#D, declared using local type "
"%qT, is used but never defined", decl, t);
}
else if (VAR_P (decl))
warning_at (DECL_SOURCE_LOCATION (decl), 0, "type %qT with no linkage "
"used to declare variable %q#D with linkage", t, decl);
else
permerror (DECL_SOURCE_LOCATION (decl), "type %qT with no linkage used "
"to declare function %q#D with linkage", t, decl);
}
static void
collect_all_refs (const char *source_file)
{
collect_ada_namespace (global_namespace, source_file);
}
static bool
clear_decl_external (struct cgraph_node *node, void * )
{
DECL_EXTERNAL (node->decl) = 0;
return false;
}
static void
handle_tls_init (void)
{
tree vars = prune_vars_needing_no_initialization (&tls_aggregates);
if (vars == NULL_TREE)
return;
location_t loc = DECL_SOURCE_LOCATION (TREE_VALUE (vars));
write_out_vars (vars);
tree guard = build_decl (loc, VAR_DECL, get_identifier ("__tls_guard"),
boolean_type_node);
TREE_PUBLIC (guard) = false;
TREE_STATIC (guard) = true;
DECL_ARTIFICIAL (guard) = true;
DECL_IGNORED_P (guard) = true;
TREE_USED (guard) = true;
CP_DECL_THREAD_LOCAL_P (guard) = true;
set_decl_tls_model (guard, decl_default_tls_model (guard));
pushdecl_top_level_and_finish (guard, NULL_TREE);
tree fn = get_local_tls_init_fn ();
start_preparsed_function (fn, NULL_TREE, SF_PRE_PARSED);
tree body = begin_function_body ();
tree if_stmt = begin_if_stmt ();
tree cond = cp_build_unary_op (TRUTH_NOT_EXPR, guard, false,
tf_warning_or_error);
finish_if_stmt_cond (cond, if_stmt);
finish_expr_stmt (cp_build_modify_expr (loc, guard, NOP_EXPR,
boolean_true_node,
tf_warning_or_error));
for (; vars; vars = TREE_CHAIN (vars))
{
tree var = TREE_VALUE (vars);
tree init = TREE_PURPOSE (vars);
one_static_initialization_or_destruction (var, init, true);
if (TARGET_SUPPORTS_ALIASES && TREE_PUBLIC (var))
{
tree single_init_fn = get_tls_init_fn (var);
if (single_init_fn == NULL_TREE)
continue;
cgraph_node *alias
= cgraph_node::get_create (fn)->create_same_body_alias
(single_init_fn, fn);
gcc_assert (alias != NULL);
}
}
finish_then_clause (if_stmt);
finish_if_stmt (if_stmt);
finish_function_body (body);
expand_or_defer_fn (finish_function (false));
}
static void
generate_mangling_alias (tree decl, tree id2)
{
struct cgraph_node *n = NULL;
if (TREE_CODE (decl) == FUNCTION_DECL)
{
n = cgraph_node::get (decl);
if (!n)
return;
}
tree *slot
= mangled_decls->find_slot_with_hash (id2, IDENTIFIER_HASH_VALUE (id2),
INSERT);
if (*slot)
return;
tree alias = make_alias_for (decl, id2);
*slot = alias;
DECL_IGNORED_P (alias) = 1;
TREE_PUBLIC (alias) = TREE_PUBLIC (decl);
DECL_VISIBILITY (alias) = DECL_VISIBILITY (decl);
if (vague_linkage_p (decl))
DECL_WEAK (alias) = 1;
if (n)
n->create_same_body_alias (alias, decl);
else
varpool_node::create_extra_name_alias (alias, decl);
}
void
note_mangling_alias (tree decl, tree id2)
{
if (TARGET_SUPPORTS_ALIASES)
{
if (!defer_mangling_aliases)
generate_mangling_alias (decl, id2);
else
{
vec_safe_push (mangling_aliases, decl);
vec_safe_push (mangling_aliases, id2);
}
}
}
void
generate_mangling_aliases ()
{
while (!vec_safe_is_empty (mangling_aliases))
{
tree id2 = mangling_aliases->pop();
tree decl = mangling_aliases->pop();
generate_mangling_alias (decl, id2);
}
defer_mangling_aliases = false;
}
void
record_mangling (tree decl, bool need_warning)
{
if (!mangled_decls)
mangled_decls = hash_table<mangled_decl_hash>::create_ggc (499);
gcc_checking_assert (DECL_ASSEMBLER_NAME_SET_P (decl));
tree id = DECL_ASSEMBLER_NAME_RAW (decl);
tree *slot
= mangled_decls->find_slot_with_hash (id, IDENTIFIER_HASH_VALUE (id),
INSERT);
if (*slot && DECL_ARTIFICIAL (*slot) && DECL_IGNORED_P (*slot))
if (symtab_node *n = symtab_node::get (*slot))
if (n->cpp_implicit_alias)
{
n->remove ();
*slot = NULL_TREE;
}
if (!*slot)
*slot = decl;
else if (need_warning)
{
error_at (DECL_SOURCE_LOCATION (decl),
"mangling of %q#D as %qE conflicts with a previous mangle",
decl, id);
inform (DECL_SOURCE_LOCATION (*slot),
"previous mangling %q#D", *slot);
inform (DECL_SOURCE_LOCATION (decl),
"a later -fabi-version= (or =0)"
" avoids this error with a change in mangling");
*slot = decl;
}
}
void
overwrite_mangling (tree decl, tree name)
{
if (tree id = DECL_ASSEMBLER_NAME_RAW (decl))
if ((TREE_CODE (decl) == VAR_DECL
|| TREE_CODE (decl) == FUNCTION_DECL)
&& mangled_decls)
if (tree *slot
= mangled_decls->find_slot_with_hash (id, IDENTIFIER_HASH_VALUE (id),
NO_INSERT))
if (*slot == decl)
{
mangled_decls->clear_slot (slot);
if (DECL_ARTIFICIAL (decl) && DECL_IGNORED_P (decl))
if (symtab_node *n = symtab_node::get (decl))
if (n->cpp_implicit_alias)
n->remove ();
}
DECL_ASSEMBLER_NAME_RAW (decl) = name;
}
static void
dump_tu (void)
{
dump_flags_t flags;
if (FILE *stream = dump_begin (raw_dump_id, &flags))
{
dump_node (global_namespace, flags & ~TDF_SLIM, stream);
dump_end (raw_dump_id, stream);
}
}
static location_t locus_at_end_of_parsing;
static void
maybe_warn_sized_delete (enum tree_code code)
{
tree sized = NULL_TREE;
tree unsized = NULL_TREE;
for (ovl_iterator iter (get_global_binding (ovl_op_identifier (false, code)));
iter; ++iter)
{
tree fn = *iter;
if (!usual_deallocation_fn_p (fn))
continue;
if (FUNCTION_ARG_CHAIN (fn) == void_list_node)
unsized = fn;
else
sized = fn;
}
if (DECL_INITIAL (unsized) && !DECL_INITIAL (sized))
warning_at (DECL_SOURCE_LOCATION (unsized), OPT_Wsized_deallocation,
"the program should also define %qD", sized);
else if (!DECL_INITIAL (unsized) && DECL_INITIAL (sized))
warning_at (DECL_SOURCE_LOCATION (sized), OPT_Wsized_deallocation,
"the program should also define %qD", unsized);
}
static void
maybe_warn_sized_delete ()
{
if (!flag_sized_deallocation || !warn_sized_deallocation)
return;
maybe_warn_sized_delete (DELETE_EXPR);
maybe_warn_sized_delete (VEC_DELETE_EXPR);
}
static void
lower_var_init ()
{
varpool_node *node;
FOR_EACH_VARIABLE (node)
{
tree d = node->decl;
if (tree init = DECL_INITIAL (d))
DECL_INITIAL (d) = cplus_expand_constant (init);
}
}
void
c_parse_final_cleanups (void)
{
tree vars;
bool reconsider;
size_t i;
unsigned ssdf_count = 0;
int retries = 0;
tree decl;
locus_at_end_of_parsing = input_location;
at_eof = 1;
if (! global_bindings_p () || current_class_type
|| !vec_safe_is_empty (decl_namespace_list))
return;
if (pch_file)
{
symtab_node *node;
FOR_EACH_SYMBOL (node)
if (! is_a <varpool_node *> (node)
|| ! DECL_HARD_REGISTER (node->decl))
DECL_ASSEMBLER_NAME (node->decl);
c_common_write_pch ();
dump_tu ();
flag_syntax_only = 1;
return;
}
timevar_stop (TV_PHASE_PARSING);
timevar_start (TV_PHASE_DEFERRED);
symtab->process_same_body_aliases ();
if (flag_dump_ada_spec || flag_dump_ada_spec_slim)
{
if (flag_dump_ada_spec_slim)
collect_source_ref (main_input_filename);
else
collect_source_refs (global_namespace);
dump_ada_specs (collect_all_refs, cpp_check);
}
emit_support_tinfos ();
do
{
tree t;
tree decl;
reconsider = false;
instantiate_pending_templates (retries);
ggc_collect ();
for (i = keyed_classes->length ();
keyed_classes->iterate (--i, &t);)
if (maybe_emit_vtables (t))
{
reconsider = true;
keyed_classes->unordered_remove (i);
}
input_location = locus_at_end_of_parsing;
for (i = unemitted_tinfo_decls->length ();
unemitted_tinfo_decls->iterate (--i, &t);)
if (emit_tinfo_decl (t))
{
reconsider = true;
unemitted_tinfo_decls->unordered_remove (i);
}
vars = prune_vars_needing_no_initialization (&static_aggregates);
if (vars)
{
tree ssdf_body;
input_location = locus_at_end_of_parsing;
ssdf_body = start_static_storage_duration_function (ssdf_count);
write_out_vars (vars);
if (vars)
do_static_initialization_or_destruction (vars, true);
if (!flag_use_cxa_atexit && vars)
{
vars = nreverse (vars);
do_static_initialization_or_destruction (vars, false);
}
else
vars = NULL_TREE;
input_location = locus_at_end_of_parsing;
finish_static_storage_duration_function (ssdf_body);
reconsider = true;
ssdf_count++;
}
handle_tls_init ();
FOR_EACH_VEC_SAFE_ELT (deferred_fns, i, decl)
{
if (DECL_DEFAULTED_FN (decl) && ! DECL_INITIAL (decl)
&& (! DECL_REALLY_EXTERN (decl) || possibly_inlined_p (decl)))
{
push_to_top_level ();
input_location = DECL_SOURCE_LOCATION (decl);
synthesize_method (decl);
pop_from_top_level ();
reconsider = true;
}
if (!DECL_INITIAL (decl) && decl_tls_wrapper_p (decl))
generate_tls_wrapper (decl);
if (!DECL_SAVED_TREE (decl))
continue;
cgraph_node *node = cgraph_node::get_create (decl);
import_export_decl (decl);
if (DECL_NOT_REALLY_EXTERN (decl)
&& DECL_INITIAL (decl)
&& decl_needed_p (decl))
{
if (node->cpp_implicit_alias)
node = node->get_alias_target ();
node->call_for_symbol_thunks_and_aliases (clear_decl_external,
NULL, true);
if (node->same_comdat_group)
for (cgraph_node *next
= dyn_cast<cgraph_node *> (node->same_comdat_group);
next != node;
next = dyn_cast<cgraph_node *> (next->same_comdat_group))
next->call_for_symbol_thunks_and_aliases (clear_decl_external,
NULL, true);
}
if (!DECL_EXTERNAL (decl)
&& decl_needed_p (decl)
&& !TREE_ASM_WRITTEN (decl)
&& !node->definition)
{
DECL_DEFER_OUTPUT (decl) = 0;
expand_or_defer_fn (decl);
if (flag_syntax_only)
TREE_ASM_WRITTEN (decl) = 1;
reconsider = true;
}
}
if (wrapup_namespace_globals ())
reconsider = true;
FOR_EACH_VEC_SAFE_ELT (pending_statics, i, decl)
{
if (var_finalized_p (decl) || DECL_REALLY_EXTERN (decl)
|| (DECL_IN_AGGR_P (decl) && !DECL_INLINE_VAR_P (decl)))
continue;
import_export_decl (decl);
if (DECL_NOT_REALLY_EXTERN (decl) && decl_needed_p (decl))
DECL_EXTERNAL (decl) = 0;
}
if (vec_safe_length (pending_statics) != 0
&& wrapup_global_declarations (pending_statics->address (),
pending_statics->length ()))
reconsider = true;
retries++;
}
while (reconsider);
lower_var_init ();
generate_mangling_aliases ();
FOR_EACH_VEC_SAFE_ELT (deferred_fns, i, decl)
{
if (
DECL_ODR_USED (decl) && DECL_DECLARED_INLINE_P (decl)
&& !DECL_INITIAL (decl)
&& !(DECL_TEMPLATE_INSTANTIATION (decl)
&& DECL_INITIAL (DECL_TEMPLATE_RESULT
(template_for_substitution (decl)))))
{
warning_at (DECL_SOURCE_LOCATION (decl), 0,
"inline function %qD used but never defined", decl);
TREE_NO_WARNING (decl) = 1;
}
}
FOR_EACH_VEC_SAFE_ELT (no_linkage_decls, i, decl)
no_linkage_error (decl);
maybe_warn_sized_delete ();
if (c_dialect_objc ())
objc_write_global_declarations ();
push_lang_context (lang_name_c);
if (priority_info_map)
splay_tree_foreach (priority_info_map,
generate_ctor_and_dtor_functions_for_priority,
&locus_at_end_of_parsing);
else if (c_dialect_objc () && objc_static_init_needed_p ())
generate_ctor_or_dtor_function (true,
DEFAULT_INIT_PRIORITY,
&locus_at_end_of_parsing);
if (priority_info_map)
splay_tree_delete (priority_info_map);
maybe_apply_pending_pragma_weaks ();
pop_lang_context ();
if (flag_vtable_verify)
{
vtv_recover_class_info ();
vtv_compute_class_hierarchy_transitive_closure ();
vtv_build_vtable_verify_fndecl ();
}
perform_deferred_noexcept_checks ();
finish_repo ();
fini_constexpr ();
dump_tu ();
if (flag_detailed_statistics)
{
dump_tree_statistics ();
dump_time_statistics ();
}
timevar_stop (TV_PHASE_DEFERRED);
timevar_start (TV_PHASE_PARSING);
at_eof = 2;
}
void
cxx_post_compilation_parsing_cleanups (void)
{
timevar_start (TV_PHASE_LATE_PARSING_CLEANUPS);
if (flag_vtable_verify)
{
vtv_generate_init_routine ();
}
input_location = locus_at_end_of_parsing;
if (flag_checking)
validate_conversion_obstack ();
timevar_stop (TV_PHASE_LATE_PARSING_CLEANUPS);
}
tree
build_offset_ref_call_from_tree (tree fn, vec<tree, va_gc> **args,
tsubst_flags_t complain)
{
tree orig_fn;
vec<tree, va_gc> *orig_args = NULL;
tree expr;
tree object;
orig_fn = fn;
object = TREE_OPERAND (fn, 0);
if (processing_template_decl)
{
gcc_assert (TREE_CODE (fn) == DOTSTAR_EXPR
|| TREE_CODE (fn) == MEMBER_REF);
if (type_dependent_expression_p (fn)
|| any_type_dependent_arguments_p (*args))
return build_min_nt_call_vec (fn, *args);
orig_args = make_tree_vector_copy (*args);
make_args_non_dependent (*args);
object = build_non_dependent_expr (object);
if (TREE_CODE (TREE_TYPE (fn)) == METHOD_TYPE)
{
if (TREE_CODE (fn) == DOTSTAR_EXPR)
object = cp_build_addr_expr (object, complain);
vec_safe_insert (*args, 0, object);
}
fn = build_non_dependent_expr (fn);
}
if (TREE_CODE (fn) == OFFSET_REF)
{
tree object_addr = cp_build_addr_expr (object, complain);
fn = TREE_OPERAND (fn, 1);
fn = get_member_function_from_ptrfunc (&object_addr, fn,
complain);
vec_safe_insert (*args, 0, object_addr);
}
if (CLASS_TYPE_P (TREE_TYPE (fn)))
expr = build_op_call (fn, args, complain);
else
expr = cp_build_function_call_vec (fn, args, complain);
if (processing_template_decl && expr != error_mark_node)
expr = build_min_non_dep_call_vec (expr, orig_fn, orig_args);
if (orig_args != NULL)
release_tree_vector (orig_args);
return expr;
}
void
check_default_args (tree x)
{
tree arg = TYPE_ARG_TYPES (TREE_TYPE (x));
bool saw_def = false;
int i = 0 - (TREE_CODE (TREE_TYPE (x)) == METHOD_TYPE);
for (; arg && arg != void_list_node; arg = TREE_CHAIN (arg), ++i)
{
if (TREE_PURPOSE (arg))
saw_def = true;
else if (saw_def && !PACK_EXPANSION_P (TREE_VALUE (arg)))
{
error ("default argument missing for parameter %P of %q+#D", i, x);
TREE_PURPOSE (arg) = error_mark_node;
}
}
}
bool
possibly_inlined_p (tree decl)
{
gcc_assert (TREE_CODE (decl) == FUNCTION_DECL);
if (DECL_UNINLINABLE (decl))
return false;
if (!optimize)
return DECL_DECLARED_INLINE_P (decl);
return true;
}
static void
maybe_instantiate_decl (tree decl)
{
if (DECL_LANG_SPECIFIC (decl)
&& DECL_TEMPLATE_INFO (decl)
&& (decl_maybe_constant_var_p (decl)
|| (TREE_CODE (decl) == FUNCTION_DECL
&& DECL_OMP_DECLARE_REDUCTION_P (decl))
|| undeduced_auto_decl (decl))
&& !DECL_DECLARED_CONCEPT_P (decl)
&& !uses_template_parms (DECL_TI_ARGS (decl)))
{
++function_depth;
instantiate_decl (decl, false,
false);
--function_depth;
}
}
bool
mark_used (tree decl, tsubst_flags_t complain)
{
if ((complain & tf_conv))
return true;
if (BASELINK_P (decl))
{
decl = BASELINK_FUNCTIONS (decl);
if (really_overloaded_fn (decl))
return true;
decl = OVL_FIRST (decl);
}
TREE_USED (decl) = 1;
if (DECL_DECOMPOSITION_P (decl) && DECL_DECOMP_BASE (decl))
TREE_USED (DECL_DECOMP_BASE (decl)) = 1;
if (TREE_CODE (decl) == TEMPLATE_DECL)
return true;
if (DECL_CLONED_FUNCTION_P (decl))
TREE_USED (DECL_CLONED_FUNCTION (decl)) = 1;
if (TREE_CODE (decl) == CONST_DECL)
used_types_insert (DECL_CONTEXT (decl));
if (TREE_CODE (decl) == FUNCTION_DECL
&& !maybe_instantiate_noexcept (decl, complain))
return false;
if (TREE_CODE (decl) == FUNCTION_DECL
&& DECL_DELETED_FN (decl))
{
if (DECL_ARTIFICIAL (decl)
&& DECL_CONV_FN_P (decl)
&& LAMBDA_TYPE_P (DECL_CONTEXT (decl)))
sorry ("converting lambda that uses %<...%> to function pointer");
else if (complain & tf_error)
{
error ("use of deleted function %qD", decl);
if (!maybe_explain_implicit_delete (decl))
inform (DECL_SOURCE_LOCATION (decl), "declared here");
}
return false;
}
if (TREE_DEPRECATED (decl) && (complain & tf_warning)
&& deprecated_state != DEPRECATED_SUPPRESS)
warn_deprecated_use (decl, NULL_TREE);
if (!VAR_OR_FUNCTION_DECL_P (decl)
|| DECL_LANG_SPECIFIC (decl) == NULL
|| DECL_THUNK_P (decl))
{
if (!processing_template_decl
&& !require_deduced_type (decl, complain))
return false;
return true;
}
if (DECL_ODR_USED (decl))
return true;
maybe_instantiate_decl (decl);
if (processing_template_decl || in_template_function ())
return true;
if (DECL_TEMPLATE_INFO (decl)
&& uses_template_parms (DECL_TI_ARGS (decl)))
return true;
if (!require_deduced_type (decl, complain))
return false;
if (builtin_pack_fn_p (decl))
{
error ("use of built-in parameter pack %qD outside of a template",
DECL_NAME (decl));
return false;
}
if (cp_unevaluated_operand || in_discarded_stmt)
return true;
DECL_ODR_USED (decl) = 1;
if (DECL_CLONED_FUNCTION_P (decl))
DECL_ODR_USED (DECL_CLONED_FUNCTION (decl)) = 1;
if (cxx_dialect > cxx98
&& decl_linkage (decl) != lk_none
&& !DECL_EXTERN_C_P (decl)
&& !DECL_ARTIFICIAL (decl)
&& !decl_defined_p (decl)
&& no_linkage_check (TREE_TYPE (decl), false))
{
if (is_local_extern (decl))
no_linkage_error (decl);
else
vec_safe_push (no_linkage_decls, decl);
}
if (TREE_CODE (decl) == FUNCTION_DECL && DECL_DECLARED_INLINE_P (decl)
&& !DECL_INITIAL (decl) && !DECL_ARTIFICIAL (decl))
note_vague_linkage_fn (decl);
if (TREE_CODE (decl) == FUNCTION_DECL
&& DECL_NONSTATIC_MEMBER_FUNCTION_P (decl)
&& DECL_DEFAULTED_FN (decl)
&& !DECL_DEFAULTED_OUTSIDE_CLASS_P (decl)
&& ! DECL_INITIAL (decl))
{
if (DECL_VIRTUAL_P (decl) && !at_eof)
{
note_vague_linkage_fn (decl);
return true;
}
DECL_SOURCE_LOCATION (decl) = input_location;
++function_depth;
synthesize_method (decl);
--function_depth;
}
else if (VAR_OR_FUNCTION_DECL_P (decl)
&& DECL_TEMPLATE_INFO (decl)
&& !DECL_DECLARED_CONCEPT_P (decl)
&& (!DECL_EXPLICIT_INSTANTIATION (decl)
|| always_instantiate_p (decl)))
{
++function_depth;
instantiate_decl (decl, true,
false);
--function_depth;
}
return true;
}
bool
mark_used (tree decl)
{
return mark_used (decl, tf_warning_or_error);
}
tree
vtv_start_verification_constructor_init_function (void)
{
return start_objects ('I', MAX_RESERVED_INIT_PRIORITY - 1);
}
tree
vtv_finish_verification_constructor_init_function (tree function_body)
{
tree fn;
finish_compound_stmt (function_body);
fn = finish_function (false);
DECL_STATIC_CONSTRUCTOR (fn) = 1;
decl_init_priority_insert (fn, MAX_RESERVED_INIT_PRIORITY - 1);
return fn;
}
#include "gt-cp-decl2.h"
