#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "cp-tree.h"
#include "stringpool.h"
#include "cgraph.h"
#include "stor-layout.h"
#include "attribs.h"
#include "flags.h"
#include "toplev.h"
#include "convert.h"
#include "dumpfile.h"
#include "gimplify.h"
#include "intl.h"
#include "asan.h"
int class_dump_id;
int current_class_depth;
typedef struct class_stack_node {
tree name;
tree type;
tree access;
splay_tree names_used;
size_t hidden;
}* class_stack_node_t;
struct vtbl_init_data
{
tree binfo;
tree derived;
tree rtti_binfo;
vec<constructor_elt, va_gc> *inits;
tree vbase;
vec<tree, va_gc> *fns;
tree index;
int primary_vtbl_p;
int ctor_vtbl_p;
bool generate_vcall_entries;
};
typedef int (*subobject_offset_fn) (tree, tree, splay_tree);
static int current_class_stack_size;
static class_stack_node_t current_class_stack;
static GTY (()) tree sizeof_biggest_empty_class;
vec<tree, va_gc> *local_classes;
static tree get_vfield_name (tree);
static void finish_struct_anon (tree);
static tree get_vtable_name (tree);
static void get_basefndecls (tree, tree, vec<tree> *);
static int build_primary_vtable (tree, tree);
static int build_secondary_vtable (tree);
static void finish_vtbls (tree);
static void modify_vtable_entry (tree, tree, tree, tree, tree *);
static void finish_struct_bits (tree);
static int alter_access (tree, tree, tree);
static void handle_using_decl (tree, tree);
static tree dfs_modify_vtables (tree, void *);
static tree modify_all_vtables (tree, tree);
static void determine_primary_bases (tree);
static void maybe_warn_about_overly_private_class (tree);
static void add_implicitly_declared_members (tree, tree*, int, int);
static tree fixed_type_or_null (tree, int *, int *);
static tree build_simple_base_path (tree expr, tree binfo);
static tree build_vtbl_ref_1 (tree, tree);
static void build_vtbl_initializer (tree, tree, tree, tree, int *,
vec<constructor_elt, va_gc> **);
static bool check_bitfield_decl (tree);
static bool check_field_decl (tree, tree, int *, int *);
static void check_field_decls (tree, tree *, int *, int *);
static tree *build_base_field (record_layout_info, tree, splay_tree, tree *);
static void build_base_fields (record_layout_info, splay_tree, tree *);
static void check_methods (tree);
static void remove_zero_width_bit_fields (tree);
static bool accessible_nvdtor_p (tree);
struct flexmems_t;
static void diagnose_flexarrays (tree, const flexmems_t *);
static void find_flexarrays (tree, flexmems_t *, bool = false,
tree = NULL_TREE, tree = NULL_TREE);
static void check_flexarrays (tree, flexmems_t * = NULL, bool = false);
static void check_bases (tree, int *, int *);
static void check_bases_and_members (tree);
static tree create_vtable_ptr (tree, tree *);
static void include_empty_classes (record_layout_info);
static void layout_class_type (tree, tree *);
static void propagate_binfo_offsets (tree, tree);
static void layout_virtual_bases (record_layout_info, splay_tree);
static void build_vbase_offset_vtbl_entries (tree, vtbl_init_data *);
static void add_vcall_offset_vtbl_entries_r (tree, vtbl_init_data *);
static void add_vcall_offset_vtbl_entries_1 (tree, vtbl_init_data *);
static void build_vcall_offset_vtbl_entries (tree, vtbl_init_data *);
static void add_vcall_offset (tree, tree, vtbl_init_data *);
static void layout_vtable_decl (tree, int);
static tree dfs_find_final_overrider_pre (tree, void *);
static tree dfs_find_final_overrider_post (tree, void *);
static tree find_final_overrider (tree, tree, tree);
static int make_new_vtable (tree, tree);
static tree get_primary_binfo (tree);
static int maybe_indent_hierarchy (FILE *, int, int);
static tree dump_class_hierarchy_r (FILE *, dump_flags_t, tree, tree, int);
static void dump_class_hierarchy (tree);
static void dump_class_hierarchy_1 (FILE *, dump_flags_t, tree);
static void dump_array (FILE *, tree);
static void dump_vtable (tree, tree, tree);
static void dump_vtt (tree, tree);
static void dump_thunk (FILE *, int, tree);
static tree build_vtable (tree, tree, tree);
static void initialize_vtable (tree, vec<constructor_elt, va_gc> *);
static void layout_nonempty_base_or_field (record_layout_info,
tree, tree, splay_tree);
static tree end_of_class (tree, int);
static bool layout_empty_base (record_layout_info, tree, tree, splay_tree);
static void accumulate_vtbl_inits (tree, tree, tree, tree, tree,
vec<constructor_elt, va_gc> **);
static void dfs_accumulate_vtbl_inits (tree, tree, tree, tree, tree,
vec<constructor_elt, va_gc> **);
static void build_rtti_vtbl_entries (tree, vtbl_init_data *);
static void build_vcall_and_vbase_vtbl_entries (tree, vtbl_init_data *);
static void clone_constructors_and_destructors (tree);
static tree build_clone (tree, tree);
static void update_vtable_entry_for_fn (tree, tree, tree, tree *, unsigned);
static void build_ctor_vtbl_group (tree, tree);
static void build_vtt (tree);
static tree binfo_ctor_vtable (tree);
static void build_vtt_inits (tree, tree, vec<constructor_elt, va_gc> **,
tree *);
static tree dfs_build_secondary_vptr_vtt_inits (tree, void *);
static tree dfs_fixup_binfo_vtbls (tree, void *);
static int record_subobject_offset (tree, tree, splay_tree);
static int check_subobject_offset (tree, tree, splay_tree);
static int walk_subobject_offsets (tree, subobject_offset_fn,
tree, splay_tree, tree, int);
static void record_subobject_offsets (tree, tree, splay_tree, bool);
static int layout_conflict_p (tree, tree, splay_tree, int);
static int splay_tree_compare_integer_csts (splay_tree_key k1,
splay_tree_key k2);
static void warn_about_ambiguous_bases (tree);
static bool type_requires_array_cookie (tree);
static bool base_derived_from (tree, tree);
static int empty_base_at_nonzero_offset_p (tree, tree, splay_tree);
static tree end_of_base (tree);
static tree get_vcall_index (tree, tree);
static bool type_maybe_constexpr_default_constructor (tree);
tree
build_if_in_charge (tree true_stmt, tree false_stmt)
{
gcc_assert (DECL_HAS_IN_CHARGE_PARM_P (current_function_decl));
tree cmp = build2 (NE_EXPR, boolean_type_node,
current_in_charge_parm, integer_zero_node);
tree type = unlowered_expr_type (true_stmt);
if (VOID_TYPE_P (type))
type = unlowered_expr_type (false_stmt);
tree cond = build3 (COND_EXPR, type,
cmp, true_stmt, false_stmt);
return cond;
}
tree
build_base_path (enum tree_code code,
tree expr,
tree binfo,
int nonnull,
tsubst_flags_t complain)
{
tree v_binfo = NULL_TREE;
tree d_binfo = NULL_TREE;
tree probe;
tree offset;
tree target_type;
tree null_test = NULL;
tree ptr_target_type;
int fixed_type_p;
int want_pointer = TYPE_PTR_P (TREE_TYPE (expr));
bool has_empty = false;
bool virtual_access;
bool rvalue = false;
if (expr == error_mark_node || binfo == error_mark_node || !binfo)
return error_mark_node;
for (probe = binfo; probe; probe = BINFO_INHERITANCE_CHAIN (probe))
{
d_binfo = probe;
if (is_empty_class (BINFO_TYPE (probe)))
has_empty = true;
if (!v_binfo && BINFO_VIRTUAL_P (probe))
v_binfo = probe;
}
probe = TYPE_MAIN_VARIANT (TREE_TYPE (expr));
if (want_pointer)
probe = TYPE_MAIN_VARIANT (TREE_TYPE (probe));
if (dependent_type_p (probe))
if (tree open = currently_open_class (probe))
probe = open;
if (code == PLUS_EXPR
&& !SAME_BINFO_TYPE_P (BINFO_TYPE (d_binfo), probe))
{
if (complain & tf_error)
{
tree base = lookup_base (probe, BINFO_TYPE (d_binfo),
ba_unique, NULL, complain);
gcc_assert (base == error_mark_node || !base);
}
return error_mark_node;
}
gcc_assert ((code == MINUS_EXPR
&& SAME_BINFO_TYPE_P (BINFO_TYPE (binfo), probe))
|| code == PLUS_EXPR);
if (binfo == d_binfo)
return expr;
if (code == MINUS_EXPR && v_binfo)
{
if (complain & tf_error)
{
if (SAME_BINFO_TYPE_P (BINFO_TYPE (binfo), BINFO_TYPE (v_binfo)))
{
if (want_pointer)
error ("cannot convert from pointer to base class %qT to "
"pointer to derived class %qT because the base is "
"virtual", BINFO_TYPE (binfo), BINFO_TYPE (d_binfo));
else
error ("cannot convert from base class %qT to derived "
"class %qT because the base is virtual",
BINFO_TYPE (binfo), BINFO_TYPE (d_binfo));
}	      
else
{
if (want_pointer)
error ("cannot convert from pointer to base class %qT to "
"pointer to derived class %qT via virtual base %qT",
BINFO_TYPE (binfo), BINFO_TYPE (d_binfo),
BINFO_TYPE (v_binfo));
else
error ("cannot convert from base class %qT to derived "
"class %qT via virtual base %qT", BINFO_TYPE (binfo),
BINFO_TYPE (d_binfo), BINFO_TYPE (v_binfo));
}
}
return error_mark_node;
}
if (!want_pointer)
{
rvalue = !lvalue_p (expr);
expr = cp_build_addr_expr (expr, complain);
}
else
expr = mark_rvalue_use (expr);
offset = BINFO_OFFSET (binfo);
fixed_type_p = resolves_to_fixed_type_p (expr, &nonnull);
target_type = code == PLUS_EXPR ? BINFO_TYPE (binfo) : BINFO_TYPE (d_binfo);
target_type = cp_build_qualified_type
(target_type, cp_type_quals (TREE_TYPE (TREE_TYPE (expr))));
ptr_target_type = build_pointer_type (target_type);
virtual_access = (v_binfo && fixed_type_p <= 0);
if (cp_unevaluated_operand != 0
|| processing_template_decl
|| in_template_function ())
{
expr = build_nop (ptr_target_type, expr);
goto indout;
}
if (virtual_access && fixed_type_p < 0
&& current_scope () != current_function_decl)
{
expr = build1 (CONVERT_EXPR, ptr_target_type, expr);
CONVERT_EXPR_VBASE_PATH (expr) = true;
goto indout;
}
if (want_pointer && !nonnull)
{
if (!virtual_access && integer_zerop (offset))
return build_nop (ptr_target_type, expr);
null_test = error_mark_node;
}
if (TREE_SIDE_EFFECTS (expr) && (null_test || virtual_access))
expr = save_expr (expr);
if (null_test)
{
tree zero = cp_convert (TREE_TYPE (expr), nullptr_node, complain);
null_test = build2_loc (input_location, NE_EXPR, boolean_type_node,
expr, zero);
TREE_NO_WARNING (null_test) = 1;
}
if (code == PLUS_EXPR && !virtual_access
&& !has_empty)
{
expr = cp_build_fold_indirect_ref (expr);
expr = build_simple_base_path (expr, binfo);
if (rvalue && lvalue_p (expr))
expr = move (expr);
if (want_pointer)
expr = build_address (expr);
target_type = TREE_TYPE (expr);
goto out;
}
if (virtual_access)
{
tree v_offset;
if (fixed_type_p < 0 && in_base_initializer)
{
tree t;
t = TREE_TYPE (TYPE_VFIELD (current_class_type));
t = build_pointer_type (t);
v_offset = fold_convert (t, current_vtt_parm);
v_offset = cp_build_fold_indirect_ref (v_offset);
}
else
{
tree t = expr;
if (sanitize_flags_p (SANITIZE_VPTR)
&& fixed_type_p == 0)
{
t = cp_ubsan_maybe_instrument_cast_to_vbase (input_location,
probe, expr);
if (t == NULL_TREE)
t = expr;
}
v_offset = build_vfield_ref (cp_build_fold_indirect_ref (t),
TREE_TYPE (TREE_TYPE (expr)));
}
if (v_offset == error_mark_node)
return error_mark_node;
v_offset = fold_build_pointer_plus (v_offset, BINFO_VPTR_FIELD (v_binfo));
v_offset = build1 (NOP_EXPR,
build_pointer_type (ptrdiff_type_node),
v_offset);
v_offset = cp_build_fold_indirect_ref (v_offset);
TREE_CONSTANT (v_offset) = 1;
offset = convert_to_integer (ptrdiff_type_node,
size_diffop_loc (input_location, offset,
BINFO_OFFSET (v_binfo)));
if (!integer_zerop (offset))
v_offset = build2 (code, ptrdiff_type_node, v_offset, offset);
if (fixed_type_p < 0)
offset = build_if_in_charge
(convert_to_integer (ptrdiff_type_node, BINFO_OFFSET (binfo)),
v_offset);
else
offset = v_offset;
}
if (want_pointer)
target_type = ptr_target_type;
expr = build1 (NOP_EXPR, ptr_target_type, expr);
if (!integer_zerop (offset))
{
offset = fold_convert (sizetype, offset);
if (code == MINUS_EXPR)
offset = fold_build1_loc (input_location, NEGATE_EXPR, sizetype, offset);
expr = fold_build_pointer_plus (expr, offset);
}
else
null_test = NULL;
indout:
if (!want_pointer)
{
expr = cp_build_fold_indirect_ref (expr);
if (rvalue)
expr = move (expr);
}
out:
if (null_test)
expr = fold_build3_loc (input_location, COND_EXPR, target_type, null_test, expr,
build_zero_cst (target_type));
return expr;
}
static tree
build_simple_base_path (tree expr, tree binfo)
{
tree type = BINFO_TYPE (binfo);
tree d_binfo = BINFO_INHERITANCE_CHAIN (binfo);
tree field;
if (d_binfo == NULL_TREE)
{
tree temp;
gcc_assert (TYPE_MAIN_VARIANT (TREE_TYPE (expr)) == type);
temp = unary_complex_lvalue (ADDR_EXPR, expr);
if (temp)
expr = cp_build_fold_indirect_ref (temp);
return expr;
}
expr = build_simple_base_path (expr, d_binfo);
for (field = TYPE_FIELDS (BINFO_TYPE (d_binfo));
field; field = DECL_CHAIN (field))
if (TREE_CODE (field) == FIELD_DECL
&& DECL_FIELD_IS_BASE (field)
&& TREE_TYPE (field) == type
&& (BINFO_INHERITANCE_CHAIN (d_binfo)
|| tree_int_cst_equal (byte_position (field),
BINFO_OFFSET (binfo))))
{
int type_quals = cp_type_quals (TREE_TYPE (expr));
expr = build3 (COMPONENT_REF,
cp_build_qualified_type (type, type_quals),
expr, field, NULL_TREE);
if (type_quals & TYPE_QUAL_CONST)
TREE_READONLY (expr) = 1;
if (type_quals & TYPE_QUAL_VOLATILE)
TREE_THIS_VOLATILE (expr) = 1;
return expr;
}
gcc_unreachable ();
}
tree
convert_to_base (tree object, tree type, bool check_access, bool nonnull,
tsubst_flags_t complain)
{
tree binfo;
tree object_type;
if (TYPE_PTR_P (TREE_TYPE (object)))
{
object_type = TREE_TYPE (TREE_TYPE (object));
type = TREE_TYPE (type);
}
else
object_type = TREE_TYPE (object);
binfo = lookup_base (object_type, type, check_access ? ba_check : ba_unique,
NULL, complain);
if (!binfo || binfo == error_mark_node)
return error_mark_node;
return build_base_path (PLUS_EXPR, object, binfo, nonnull, complain);
}
tree
convert_to_base_statically (tree expr, tree base)
{
tree expr_type;
expr_type = TREE_TYPE (expr);
if (!SAME_BINFO_TYPE_P (BINFO_TYPE (base), expr_type))
{
if (!is_empty_class (BINFO_TYPE (base)))
return build_simple_base_path (expr, base);
gcc_assert (!processing_template_decl);
expr = cp_build_addr_expr (expr, tf_warning_or_error);
if (!integer_zerop (BINFO_OFFSET (base)))
expr = fold_build_pointer_plus_loc (input_location,
expr, BINFO_OFFSET (base));
expr = fold_convert (build_pointer_type (BINFO_TYPE (base)), expr);
expr = build_fold_indirect_ref_loc (input_location, expr);
}
return expr;
}

tree
build_vfield_ref (tree datum, tree type)
{
tree vfield, vcontext;
if (datum == error_mark_node
|| !TYPE_VFIELD (type))
return error_mark_node;
if (!same_type_ignoring_top_level_qualifiers_p (TREE_TYPE (datum), type))
datum = convert_to_base (datum, type, false,
true, tf_warning_or_error);
vfield = TYPE_VFIELD (type);
vcontext = DECL_CONTEXT (vfield);
while (!same_type_ignoring_top_level_qualifiers_p (vcontext, type))
{
datum = build_simple_base_path (datum, CLASSTYPE_PRIMARY_BINFO (type));
type = TREE_TYPE (datum);
}
return build3 (COMPONENT_REF, TREE_TYPE (vfield), datum, vfield, NULL_TREE);
}
static tree
build_vtbl_ref_1 (tree instance, tree idx)
{
tree aref;
tree vtbl = NULL_TREE;
int cdtorp = 0;
tree fixed_type = fixed_type_or_null (instance, NULL, &cdtorp);
tree basetype = non_reference (TREE_TYPE (instance));
if (fixed_type && !cdtorp)
{
tree binfo = lookup_base (fixed_type, basetype,
ba_unique, NULL, tf_none);
if (binfo && binfo != error_mark_node)
vtbl = unshare_expr (BINFO_VTABLE (binfo));
}
if (!vtbl)
vtbl = build_vfield_ref (instance, basetype);
aref = build_array_ref (input_location, vtbl, idx);
TREE_CONSTANT (aref) |= TREE_CONSTANT (vtbl) && TREE_CONSTANT (idx);
return aref;
}
tree
build_vtbl_ref (tree instance, tree idx)
{
tree aref = build_vtbl_ref_1 (instance, idx);
return aref;
}
tree
build_vfn_ref (tree instance_ptr, tree idx)
{
tree aref;
aref = build_vtbl_ref_1 (cp_build_fold_indirect_ref (instance_ptr),
idx);
if (TARGET_VTABLE_USES_DESCRIPTORS)
aref = build1 (NOP_EXPR, TREE_TYPE (aref),
cp_build_addr_expr (aref, tf_warning_or_error));
aref = build3 (OBJ_TYPE_REF, TREE_TYPE (aref), aref, instance_ptr, idx);
return aref;
}
static tree
get_vtable_name (tree type)
{
return mangle_vtbl_for_type (type);
}
void
set_linkage_according_to_type (tree , tree decl)
{
TREE_PUBLIC (decl) = 1;
determine_visibility (decl);
}
static tree
build_vtable (tree class_type, tree name, tree vtable_type)
{
tree decl;
decl = build_lang_decl (VAR_DECL, name, vtable_type);
SET_DECL_ASSEMBLER_NAME (decl, name);
DECL_CONTEXT (decl) = class_type;
DECL_ARTIFICIAL (decl) = 1;
TREE_STATIC (decl) = 1;
TREE_READONLY (decl) = 1;
DECL_VIRTUAL_P (decl) = 1;
SET_DECL_ALIGN (decl, TARGET_VTABLE_ENTRY_ALIGN);
DECL_USER_ALIGN (decl) = true;
DECL_VTABLE_OR_VTT_P (decl) = 1;
set_linkage_according_to_type (class_type, decl);
DECL_EXTERNAL (decl) = 1;
DECL_NOT_REALLY_EXTERN (decl) = 1;
DECL_IGNORED_P (decl) = 1;
return decl;
}
tree
get_vtable_decl (tree type, int complete)
{
tree decl;
if (CLASSTYPE_VTABLES (type))
return CLASSTYPE_VTABLES (type);
decl = build_vtable (type, get_vtable_name (type), vtbl_type_node);
CLASSTYPE_VTABLES (type) = decl;
if (complete)
{
DECL_EXTERNAL (decl) = 1;
cp_finish_decl (decl, NULL_TREE, false, NULL_TREE, 0);
}
return decl;
}
static int
build_primary_vtable (tree binfo, tree type)
{
tree decl;
tree virtuals;
decl = get_vtable_decl (type, 0);
if (binfo)
{
if (BINFO_NEW_VTABLE_MARKED (binfo))
return 0;
virtuals = copy_list (BINFO_VIRTUALS (binfo));
TREE_TYPE (decl) = TREE_TYPE (get_vtbl_decl_for_binfo (binfo));
DECL_SIZE (decl) = TYPE_SIZE (TREE_TYPE (decl));
DECL_SIZE_UNIT (decl) = TYPE_SIZE_UNIT (TREE_TYPE (decl));
}
else
{
gcc_assert (TREE_TYPE (decl) == vtbl_type_node);
virtuals = NULL_TREE;
}
BINFO_VTABLE (TYPE_BINFO (type)) = decl;
BINFO_VIRTUALS (TYPE_BINFO (type)) = virtuals;
SET_BINFO_NEW_VTABLE_MARKED (TYPE_BINFO (type));
return 1;
}
static int
build_secondary_vtable (tree binfo)
{
if (BINFO_NEW_VTABLE_MARKED (binfo))
return 0;
SET_BINFO_NEW_VTABLE_MARKED (binfo);
BINFO_VIRTUALS (binfo) = copy_list (BINFO_VIRTUALS (binfo));
BINFO_VTABLE (binfo) = NULL_TREE;
return 1;
}
static int
make_new_vtable (tree t, tree binfo)
{
if (binfo == TYPE_BINFO (t))
return build_primary_vtable (binfo, t);
else
return build_secondary_vtable (binfo);
}
static void
modify_vtable_entry (tree t,
tree binfo,
tree fndecl,
tree delta,
tree *virtuals)
{
tree v;
v = *virtuals;
if (fndecl != BV_FN (v)
|| !tree_int_cst_equal (delta, BV_DELTA (v)))
{
if (make_new_vtable (t, binfo))
{
*virtuals = BINFO_VIRTUALS (binfo);
while (BV_FN (*virtuals) != BV_FN (v))
*virtuals = TREE_CHAIN (*virtuals);
v = *virtuals;
}
BV_DELTA (v) = delta;
BV_VCALL_INDEX (v) = NULL_TREE;
BV_FN (v) = fndecl;
}
}

bool
add_method (tree type, tree method, bool via_using)
{
if (method == error_mark_node)
return false;
gcc_assert (!DECL_EXTERN_C_P (method));
tree *slot = find_member_slot (type, DECL_NAME (method));
tree current_fns = slot ? *slot : NULL_TREE;
for (ovl_iterator iter (current_fns); iter; ++iter)
{
tree fn = *iter;
tree fn_type;
tree method_type;
tree parms1;
tree parms2;
if (TREE_CODE (fn) != TREE_CODE (method))
continue;
if (via_using && iter.using_p ()
&& ! DECL_CONSTRUCTOR_P (fn))
continue;
fn_type = TREE_TYPE (fn);
method_type = TREE_TYPE (method);
parms1 = TYPE_ARG_TYPES (fn_type);
parms2 = TYPE_ARG_TYPES (method_type);
if (! DECL_STATIC_FUNCTION_P (fn)
&& ! DECL_STATIC_FUNCTION_P (method)
&& (FUNCTION_REF_QUALIFIED (fn_type)
== FUNCTION_REF_QUALIFIED (method_type))
&& (type_memfn_quals (fn_type) != type_memfn_quals (method_type)
|| type_memfn_rqual (fn_type) != type_memfn_rqual (method_type)))
continue;
if (TREE_CODE (fn) == TEMPLATE_DECL
&& (!same_type_p (TREE_TYPE (fn_type),
TREE_TYPE (method_type))
|| !comp_template_parms (DECL_TEMPLATE_PARMS (fn),
DECL_TEMPLATE_PARMS (method))))
continue;
if (! DECL_STATIC_FUNCTION_P (fn))
parms1 = TREE_CHAIN (parms1);
if (! DECL_STATIC_FUNCTION_P (method))
parms2 = TREE_CHAIN (parms2);
if (ctor_omit_inherited_parms (fn))
parms1 = FUNCTION_FIRST_USER_PARMTYPE (DECL_ORIGIN (fn));
if (ctor_omit_inherited_parms (method))
parms2 = FUNCTION_FIRST_USER_PARMTYPE (DECL_ORIGIN (method));
if (compparms (parms1, parms2)
&& (!DECL_CONV_FN_P (fn)
|| same_type_p (TREE_TYPE (fn_type),
TREE_TYPE (method_type)))
&& equivalently_constrained (fn, method))
{
if (TREE_CODE (fn) == FUNCTION_DECL
&& maybe_version_functions (method, fn, true))
continue;
if (DECL_INHERITED_CTOR (method))
{
if (DECL_INHERITED_CTOR (fn))
{
tree basem = DECL_INHERITED_CTOR_BASE (method);
tree basef = DECL_INHERITED_CTOR_BASE (fn);
if (flag_new_inheriting_ctors)
{
if (basem == basef)
{
SET_DECL_INHERITED_CTOR
(fn, ovl_make (DECL_INHERITED_CTOR (method),
DECL_INHERITED_CTOR (fn)));
return false;
}
else
continue;
}
error_at (DECL_SOURCE_LOCATION (method),
"%q#D conflicts with version inherited from %qT",
method, basef);
inform (DECL_SOURCE_LOCATION (fn),
"version inherited from %qT declared here",
basef);
}
return false;
}
if (via_using)
return false;
else if (flag_new_inheriting_ctors
&& DECL_INHERITED_CTOR (fn))
{
current_fns = iter.remove_node (current_fns);
continue;
}
else
{
error_at (DECL_SOURCE_LOCATION (method),
"%q#D cannot be overloaded with %q#D", method, fn);
inform (DECL_SOURCE_LOCATION (fn),
"previous declaration %q#D", fn);
return false;
}
}
}
current_fns = ovl_insert (method, current_fns, via_using);
if (!COMPLETE_TYPE_P (type) && !DECL_CONV_FN_P (method)
&& !push_class_level_binding (DECL_NAME (method), current_fns))
return false;
if (!slot)
slot = add_member_slot (type, DECL_NAME (method));
grok_special_member_properties (method);
*slot = current_fns;
return true;
}
static int
alter_access (tree t, tree fdecl, tree access)
{
tree elem;
retrofit_lang_decl (fdecl);
gcc_assert (!DECL_DISCRIMINATOR_P (fdecl));
elem = purpose_member (t, DECL_ACCESS (fdecl));
if (elem)
{
if (TREE_VALUE (elem) != access)
{
if (TREE_CODE (TREE_TYPE (fdecl)) == FUNCTION_DECL)
error ("conflicting access specifications for method"
" %q+D, ignored", TREE_TYPE (fdecl));
else
error ("conflicting access specifications for field %qE, ignored",
DECL_NAME (fdecl));
}
else
{
;
}
}
else
{
perform_or_defer_access_check (TYPE_BINFO (t), fdecl, fdecl,
tf_warning_or_error);
DECL_ACCESS (fdecl) = tree_cons (t, access, DECL_ACCESS (fdecl));
return 1;
}
return 0;
}
tree
declared_access (tree decl)
{
return (TREE_PRIVATE (decl) ? access_private_node
: TREE_PROTECTED (decl) ? access_protected_node
: access_public_node);
}
static void
handle_using_decl (tree using_decl, tree t)
{
tree decl = USING_DECL_DECLS (using_decl);
tree name = DECL_NAME (using_decl);
tree access = declared_access (using_decl);
tree flist = NULL_TREE;
tree old_value;
gcc_assert (!processing_template_decl && decl);
old_value = lookup_member (t, name, 0, false,
tf_warning_or_error);
if (old_value)
{
old_value = OVL_FIRST (old_value);
if (DECL_P (old_value) && DECL_CONTEXT (old_value) == t)
;
else
old_value = NULL_TREE;
}
cp_emit_debug_info_for_using (decl, t);
if (is_overloaded_fn (decl))
flist = decl;
if (! old_value)
;
else if (is_overloaded_fn (old_value))
{
if (flist)
;
else
{
error_at (DECL_SOURCE_LOCATION (using_decl), "%qD invalid in %q#T "
"because of local method %q#D with same name",
using_decl, t, old_value);
inform (DECL_SOURCE_LOCATION (old_value),
"local method %q#D declared here", old_value);
return;
}
}
else if (!DECL_ARTIFICIAL (old_value))
{
error_at (DECL_SOURCE_LOCATION (using_decl), "%qD invalid in %q#T "
"because of local member %q#D with same name",
using_decl, t, old_value);
inform (DECL_SOURCE_LOCATION (old_value),
"local member %q#D declared here", old_value);
return;
}
if (flist)
for (ovl_iterator iter (flist); iter; ++iter)
{
add_method (t, *iter, true);
alter_access (t, *iter, access);
}
else
alter_access (t, decl, access);
}

struct abi_tag_data
{
tree t;		
tree subob;		
tree tags; 
};
static void
check_tag (tree tag, tree id, tree *tp, abi_tag_data *p)
{
if (!IDENTIFIER_MARKED (id))
{
if (p->tags != error_mark_node)
{
p->tags = tree_cons (NULL_TREE, tag, p->tags);
IDENTIFIER_MARKED (id) = true;
if (TYPE_P (p->t))
{
ABI_TAG_IMPLICIT (p->tags) = true;
return;
}
}
if (TREE_CODE (p->t) == FUNCTION_DECL)
{
if (warning (OPT_Wabi_tag, "%qD inherits the %E ABI tag "
"that %qT (used in its return type) has",
p->t, tag, *tp))
inform (location_of (*tp), "%qT declared here", *tp);
}
else if (VAR_P (p->t))
{
if (warning (OPT_Wabi_tag, "%qD inherits the %E ABI tag "
"that %qT (used in its type) has", p->t, tag, *tp))
inform (location_of (*tp), "%qT declared here", *tp);
}
else if (TYPE_P (p->subob))
{
if (warning (OPT_Wabi_tag, "%qT does not have the %E ABI tag "
"that base %qT has", p->t, tag, p->subob))
inform (location_of (p->subob), "%qT declared here",
p->subob);
}
else
{
if (warning (OPT_Wabi_tag, "%qT does not have the %E ABI tag "
"that %qT (used in the type of %qD) has",
p->t, tag, *tp, p->subob))
{
inform (location_of (p->subob), "%qD declared here",
p->subob);
inform (location_of (*tp), "%qT declared here", *tp);
}
}
}
}
static void
mark_or_check_attr_tags (tree attr, tree *tp, abi_tag_data *p, bool val)
{
if (!attr)
return;
for (; (attr = lookup_attribute ("abi_tag", attr));
attr = TREE_CHAIN (attr))
for (tree list = TREE_VALUE (attr); list;
list = TREE_CHAIN (list))
{
tree tag = TREE_VALUE (list);
tree id = get_identifier (TREE_STRING_POINTER (tag));
if (tp)
check_tag (tag, id, tp, p);
else
IDENTIFIER_MARKED (id) = val;
}
}
static void
mark_or_check_tags (tree t, tree *tp, abi_tag_data *p, bool val)
{
while (t != global_namespace)
{
tree attr;
if (TYPE_P (t))
{
attr = TYPE_ATTRIBUTES (t);
t = CP_TYPE_CONTEXT (t);
}
else
{
attr = DECL_ATTRIBUTES (t);
t = CP_DECL_CONTEXT (t);
}
mark_or_check_attr_tags (attr, tp, p, val);
}
}
static tree
find_abi_tags_r (tree *tp, int *walk_subtrees, void *data)
{
if (!OVERLOAD_TYPE_P (*tp))
return NULL_TREE;
*walk_subtrees = false;
abi_tag_data *p = static_cast<struct abi_tag_data*>(data);
mark_or_check_tags (*tp, tp, p, false);
return NULL_TREE;
}
static tree
mark_abi_tags_r (tree *tp, int *walk_subtrees, void *data)
{
if (!OVERLOAD_TYPE_P (*tp))
return NULL_TREE;
*walk_subtrees = false;
bool *valp = static_cast<bool*>(data);
mark_or_check_tags (*tp, NULL, NULL, *valp);
return NULL_TREE;
}
static void
mark_abi_tags (tree t, bool val)
{
mark_or_check_tags (t, NULL, NULL, val);
if (DECL_P (t))
{
if (DECL_LANG_SPECIFIC (t) && DECL_USE_TEMPLATE (t)
&& PRIMARY_TEMPLATE_P (DECL_TI_TEMPLATE (t)))
{
tree level = INNERMOST_TEMPLATE_ARGS (DECL_TI_ARGS (t));
for (int j = 0; j < TREE_VEC_LENGTH (level); ++j)
{
tree arg = TREE_VEC_ELT (level, j);
cp_walk_tree_without_duplicates (&arg, mark_abi_tags_r, &val);
}
}
if (TREE_CODE (t) == FUNCTION_DECL)
for (tree arg = FUNCTION_FIRST_USER_PARMTYPE (t); arg;
arg = TREE_CHAIN (arg))
cp_walk_tree_without_duplicates (&TREE_VALUE (arg),
mark_abi_tags_r, &val);
}
}
static tree
check_abi_tags (tree t, tree subob, bool just_checking = false)
{
bool inherit = DECL_P (t);
if (!inherit && !warn_abi_tag)
return NULL_TREE;
tree decl = TYPE_P (t) ? TYPE_NAME (t) : t;
if (!TREE_PUBLIC (decl))
return NULL_TREE;
mark_abi_tags (t, true);
tree subtype = TYPE_P (subob) ? subob : TREE_TYPE (subob);
struct abi_tag_data data = { t, subob, error_mark_node };
if (inherit)
data.tags = NULL_TREE;
cp_walk_tree_without_duplicates (&subtype, find_abi_tags_r, &data);
if (!(inherit && data.tags))
;
else if (just_checking)
for (tree t = data.tags; t; t = TREE_CHAIN (t))
{
tree id = get_identifier (TREE_STRING_POINTER (TREE_VALUE (t)));
IDENTIFIER_MARKED (id) = false;
}
else
{
tree attr = lookup_attribute ("abi_tag", DECL_ATTRIBUTES (t));
if (attr)
TREE_VALUE (attr) = chainon (data.tags, TREE_VALUE (attr));
else
DECL_ATTRIBUTES (t)
= tree_cons (get_identifier ("abi_tag"), data.tags,
DECL_ATTRIBUTES (t));
}
mark_abi_tags (t, false);
return data.tags;
}
void
check_abi_tags (tree decl)
{
if (VAR_P (decl))
check_abi_tags (decl, TREE_TYPE (decl));
else if (TREE_CODE (decl) == FUNCTION_DECL
&& !DECL_CONV_FN_P (decl)
&& !mangle_return_type_p (decl))
check_abi_tags (decl, TREE_TYPE (TREE_TYPE (decl)));
}
tree
missing_abi_tags (tree decl)
{
if (VAR_P (decl))
return check_abi_tags (decl, TREE_TYPE (decl), true);
else if (TREE_CODE (decl) == FUNCTION_DECL
&& !mangle_return_type_p (decl))
return check_abi_tags (decl, TREE_TYPE (TREE_TYPE (decl)), true);
else
return NULL_TREE;
}
void
inherit_targ_abi_tags (tree t)
{
if (!CLASS_TYPE_P (t)
|| CLASSTYPE_TEMPLATE_INFO (t) == NULL_TREE)
return;
mark_abi_tags (t, true);
tree args = CLASSTYPE_TI_ARGS (t);
struct abi_tag_data data = { t, NULL_TREE, NULL_TREE };
for (int i = 0; i < TMPL_ARGS_DEPTH (args); ++i)
{
tree level = TMPL_ARGS_LEVEL (args, i+1);
for (int j = 0; j < TREE_VEC_LENGTH (level); ++j)
{
tree arg = TREE_VEC_ELT (level, j);
data.subob = arg;
cp_walk_tree_without_duplicates (&arg, find_abi_tags_r, &data);
}
}
if (data.tags)
{
tree attr = lookup_attribute ("abi_tag", TYPE_ATTRIBUTES (t));
if (attr)
TREE_VALUE (attr) = chainon (data.tags, TREE_VALUE (attr));
else
TYPE_ATTRIBUTES (t)
= tree_cons (get_identifier ("abi_tag"), data.tags,
TYPE_ATTRIBUTES (t));
}
mark_abi_tags (t, false);
}
static bool
accessible_nvdtor_p (tree t)
{
tree dtor = CLASSTYPE_DESTRUCTOR (t);
if (!dtor)
return true;
if (DECL_VINDEX (dtor))
return false; 
if (!TREE_PRIVATE (dtor) && !TREE_PROTECTED (dtor))
return true;  
if (CLASSTYPE_FRIEND_CLASSES (t)
|| DECL_FRIENDLIST (TYPE_MAIN_DECL (t)))
return true;   
return false;
}
static void
check_bases (tree t,
int* cant_have_const_ctor_p,
int* no_const_asn_ref_p)
{
int i;
bool seen_non_virtual_nearly_empty_base_p = 0;
int seen_tm_mask = 0;
tree base_binfo;
tree binfo;
tree field = NULL_TREE;
if (!CLASSTYPE_NON_STD_LAYOUT (t))
for (field = TYPE_FIELDS (t); field; field = DECL_CHAIN (field))
if (TREE_CODE (field) == FIELD_DECL)
break;
for (binfo = TYPE_BINFO (t), i = 0;
BINFO_BASE_ITERATE (binfo, i, base_binfo); i++)
{
tree basetype = TREE_TYPE (base_binfo);
gcc_assert (COMPLETE_TYPE_P (basetype));
if (CLASSTYPE_FINAL (basetype))
error ("cannot derive from %<final%> base %qT in derived type %qT",
basetype, t);
if (!CLASSTYPE_LITERAL_P (basetype))
CLASSTYPE_LITERAL_P (t) = false;
if (TYPE_HAS_COPY_CTOR (basetype)
&& ! TYPE_HAS_CONST_COPY_CTOR (basetype))
*cant_have_const_ctor_p = 1;
if (TYPE_HAS_COPY_ASSIGN (basetype)
&& !TYPE_HAS_CONST_COPY_ASSIGN (basetype))
*no_const_asn_ref_p = 1;
if (BINFO_VIRTUAL_P (base_binfo))
;
else if (CLASSTYPE_NEARLY_EMPTY_P (basetype))
{
if (seen_non_virtual_nearly_empty_base_p)
CLASSTYPE_NEARLY_EMPTY_P (t) = 0;
else
seen_non_virtual_nearly_empty_base_p = 1;
}
else if (!is_empty_class (basetype))
CLASSTYPE_NEARLY_EMPTY_P (t) = 0;
TYPE_NEEDS_CONSTRUCTING (t) |= TYPE_NEEDS_CONSTRUCTING (basetype);
TYPE_HAS_NONTRIVIAL_DESTRUCTOR (t)
|= TYPE_HAS_NONTRIVIAL_DESTRUCTOR (basetype);
TYPE_HAS_COMPLEX_COPY_ASSIGN (t)
|= (TYPE_HAS_COMPLEX_COPY_ASSIGN (basetype)
|| !TYPE_HAS_COPY_ASSIGN (basetype));
TYPE_HAS_COMPLEX_COPY_CTOR (t) |= (TYPE_HAS_COMPLEX_COPY_CTOR (basetype)
|| !TYPE_HAS_COPY_CTOR (basetype));
TYPE_HAS_COMPLEX_MOVE_ASSIGN (t)
|= TYPE_HAS_COMPLEX_MOVE_ASSIGN (basetype);
TYPE_HAS_COMPLEX_MOVE_CTOR (t) |= TYPE_HAS_COMPLEX_MOVE_CTOR (basetype);
TYPE_POLYMORPHIC_P (t) |= TYPE_POLYMORPHIC_P (basetype);
CLASSTYPE_CONTAINS_EMPTY_CLASS_P (t)
|= CLASSTYPE_CONTAINS_EMPTY_CLASS_P (basetype);
TYPE_HAS_COMPLEX_DFLT (t) |= (!TYPE_HAS_DEFAULT_CONSTRUCTOR (basetype)
|| TYPE_HAS_COMPLEX_DFLT (basetype));
SET_CLASSTYPE_READONLY_FIELDS_NEED_INIT
(t, CLASSTYPE_READONLY_FIELDS_NEED_INIT (t)
| CLASSTYPE_READONLY_FIELDS_NEED_INIT (basetype));
SET_CLASSTYPE_REF_FIELDS_NEED_INIT
(t, CLASSTYPE_REF_FIELDS_NEED_INIT (t)
| CLASSTYPE_REF_FIELDS_NEED_INIT (basetype));
if (TYPE_HAS_MUTABLE_P (basetype))
CLASSTYPE_HAS_MUTABLE (t) = 1;
CLASSTYPE_NON_STD_LAYOUT (t) |= CLASSTYPE_NON_STD_LAYOUT (basetype);
if (!CLASSTYPE_NON_STD_LAYOUT (t))
{
tree basefield;
if (field && DECL_CONTEXT (field) == t
&& (same_type_ignoring_top_level_qualifiers_p
(TREE_TYPE (field), basetype)))
CLASSTYPE_NON_STD_LAYOUT (t) = 1;
else
for (basefield = TYPE_FIELDS (basetype); basefield;
basefield = DECL_CHAIN (basefield))
if (TREE_CODE (basefield) == FIELD_DECL
&& !(DECL_FIELD_IS_BASE (basefield)
&& integer_zerop (DECL_SIZE (basefield))))
{
if (field)
CLASSTYPE_NON_STD_LAYOUT (t) = 1;
else
field = basefield;
break;
}
}
if (flag_tm)
{
tree tm_attr = find_tm_attribute (TYPE_ATTRIBUTES (basetype));
if (tm_attr)
seen_tm_mask |= tm_attr_to_mask (tm_attr);
}
check_abi_tags (t, basetype);
}
if (seen_tm_mask && !find_tm_attribute (TYPE_ATTRIBUTES (t)))
{
tree tm_attr = tm_mask_to_attr (least_bit_hwi (seen_tm_mask));
TYPE_ATTRIBUTES (t) = tree_cons (tm_attr, NULL, TYPE_ATTRIBUTES (t));
}
}
static void
determine_primary_bases (tree t)
{
unsigned i;
tree primary = NULL_TREE;
tree type_binfo = TYPE_BINFO (t);
tree base_binfo;
for (base_binfo = TREE_CHAIN (type_binfo); base_binfo;
base_binfo = TREE_CHAIN (base_binfo))
{
tree primary = CLASSTYPE_PRIMARY_BINFO (BINFO_TYPE (base_binfo));
if (!BINFO_VIRTUAL_P (base_binfo))
{
tree parent = BINFO_INHERITANCE_CHAIN (base_binfo);
tree parent_primary = CLASSTYPE_PRIMARY_BINFO (BINFO_TYPE (parent));
if (parent_primary
&& SAME_BINFO_TYPE_P (BINFO_TYPE (base_binfo),
BINFO_TYPE (parent_primary)))
BINFO_PRIMARY_P (base_binfo) = 1;
}
if (primary && BINFO_VIRTUAL_P (primary))
{
tree this_primary = copied_binfo (primary, base_binfo);
if (BINFO_PRIMARY_P (this_primary))
BINFO_LOST_PRIMARY_P (base_binfo) = 1;
else
{
tree delta;
BINFO_PRIMARY_P (this_primary) = 1;
BINFO_INHERITANCE_CHAIN (this_primary) = base_binfo;
delta = size_diffop_loc (input_location,
fold_convert (ssizetype,
BINFO_OFFSET (base_binfo)),
fold_convert (ssizetype,
BINFO_OFFSET (this_primary)));
propagate_binfo_offsets (this_primary, delta);
}
}
}
for (i = 0; BINFO_BASE_ITERATE (type_binfo, i, base_binfo); i++)
{
tree basetype = BINFO_TYPE (base_binfo);
if (TYPE_CONTAINS_VPTR_P (basetype) && !BINFO_VIRTUAL_P (base_binfo))
{
primary = base_binfo;
goto found;
}
}
for (base_binfo = TREE_CHAIN (type_binfo); base_binfo;
base_binfo = TREE_CHAIN (base_binfo))
if (BINFO_VIRTUAL_P (base_binfo)
&& CLASSTYPE_NEARLY_EMPTY_P (BINFO_TYPE (base_binfo)))
{
if (!BINFO_PRIMARY_P (base_binfo))
{
primary = base_binfo;
goto found;
}
else if (!primary)
primary = base_binfo;
}
found:
if (primary)
{
tree basetype = BINFO_TYPE (primary);
CLASSTYPE_PRIMARY_BINFO (t) = primary;
if (BINFO_PRIMARY_P (primary))
BINFO_LOST_PRIMARY_P (BINFO_INHERITANCE_CHAIN (primary)) = 1;
BINFO_PRIMARY_P (primary) = 1;
if (BINFO_VIRTUAL_P (primary))
{
tree delta;
BINFO_INHERITANCE_CHAIN (primary) = type_binfo;
delta = size_diffop_loc (input_location, ssize_int (0),
fold_convert (ssizetype, BINFO_OFFSET (primary)));
propagate_binfo_offsets (primary, delta);
}
primary = TYPE_BINFO (basetype);
TYPE_VFIELD (t) = TYPE_VFIELD (basetype);
BINFO_VTABLE (type_binfo) = BINFO_VTABLE (primary);
BINFO_VIRTUALS (type_binfo) = BINFO_VIRTUALS (primary);
}
}
void
fixup_type_variants (tree t)
{
tree variants;
if (!t)
return;
for (variants = TYPE_NEXT_VARIANT (t);
variants;
variants = TYPE_NEXT_VARIANT (variants))
{
TYPE_HAS_USER_CONSTRUCTOR (variants) = TYPE_HAS_USER_CONSTRUCTOR (t);
TYPE_NEEDS_CONSTRUCTING (variants) = TYPE_NEEDS_CONSTRUCTING (t);
TYPE_HAS_NONTRIVIAL_DESTRUCTOR (variants)
= TYPE_HAS_NONTRIVIAL_DESTRUCTOR (t);
TYPE_POLYMORPHIC_P (variants) = TYPE_POLYMORPHIC_P (t);
TYPE_BINFO (variants) = TYPE_BINFO (t);
TYPE_VFIELD (variants) = TYPE_VFIELD (t);
TYPE_FIELDS (variants) = TYPE_FIELDS (t);
}
}
static void
fixup_may_alias (tree klass)
{
tree t, v;
for (t = TYPE_POINTER_TO (klass); t; t = TYPE_NEXT_PTR_TO (t))
for (v = TYPE_MAIN_VARIANT (t); v; v = TYPE_NEXT_VARIANT (v))
TYPE_REF_CAN_ALIAS_ALL (v) = true;
for (t = TYPE_REFERENCE_TO (klass); t; t = TYPE_NEXT_REF_TO (t))
for (v = TYPE_MAIN_VARIANT (t); v; v = TYPE_NEXT_VARIANT (v))
TYPE_REF_CAN_ALIAS_ALL (v) = true;
}
void
fixup_attribute_variants (tree t)
{
tree variants;
if (!t)
return;
tree attrs = TYPE_ATTRIBUTES (t);
unsigned align = TYPE_ALIGN (t);
bool user_align = TYPE_USER_ALIGN (t);
bool may_alias = lookup_attribute ("may_alias", attrs);
bool packed = TYPE_PACKED (t);
if (may_alias)
fixup_may_alias (t);
for (variants = TYPE_NEXT_VARIANT (t);
variants;
variants = TYPE_NEXT_VARIANT (variants))
{
TYPE_ATTRIBUTES (variants) = attrs;
unsigned valign = align;
if (TYPE_USER_ALIGN (variants))
valign = MAX (valign, TYPE_ALIGN (variants));
else
TYPE_USER_ALIGN (variants) = user_align;
SET_TYPE_ALIGN (variants, valign);
TYPE_PACKED (variants) = packed;
if (may_alias)
fixup_may_alias (variants);
}
}

static void
finish_struct_bits (tree t)
{
fixup_type_variants (t);
if (BINFO_N_BASE_BINFOS (TYPE_BINFO (t)) && TYPE_POLYMORPHIC_P (t))
get_pure_virtuals (t);
if (type_has_nontrivial_copy_init (t)
|| TYPE_HAS_NONTRIVIAL_DESTRUCTOR (t))
{
tree variants;
SET_DECL_MODE (TYPE_MAIN_DECL (t), BLKmode);
for (variants = t; variants; variants = TYPE_NEXT_VARIANT (variants))
{
SET_TYPE_MODE (variants, BLKmode);
TREE_ADDRESSABLE (variants) = 1;
}
}
}
static void
maybe_warn_about_overly_private_class (tree t)
{
int has_member_fn = 0;
int has_nonprivate_method = 0;
bool nonprivate_ctor = false;
if (!warn_ctor_dtor_privacy
|| (CLASSTYPE_FRIEND_CLASSES (t)
|| DECL_FRIENDLIST (TYPE_MAIN_DECL (t)))
|| CLASSTYPE_TEMPLATE_INSTANTIATION (t))
return;
for (tree fn = TYPE_FIELDS (t); fn; fn = DECL_CHAIN (fn))
if (TREE_CODE (fn) == USING_DECL
&& DECL_NAME (fn) == ctor_identifier
&& !TREE_PRIVATE (fn))
nonprivate_ctor = true;
else if (!DECL_DECLARES_FUNCTION_P (fn))
;
else if (DECL_ARTIFICIAL (fn))
;
else if (!TREE_PRIVATE (fn))
{
if (DECL_STATIC_FUNCTION_P (fn))
return;
has_nonprivate_method = 1;
}
else if (!DECL_CONSTRUCTOR_P (fn) && !DECL_DESTRUCTOR_P (fn))
has_member_fn = 1;
if (!has_nonprivate_method && has_member_fn)
{
unsigned i;
tree binfo = TYPE_BINFO (t);
for (i = 0; i != BINFO_N_BASE_BINFOS (binfo); i++)
if (BINFO_BASE_ACCESS (binfo, i) != access_private_node)
{
has_nonprivate_method = 1;
break;
}
if (!has_nonprivate_method)
{
warning (OPT_Wctor_dtor_privacy,
"all member functions in class %qT are private", t);
return;
}
}
if (tree dtor = CLASSTYPE_DESTRUCTOR (t))
if (TREE_PRIVATE (dtor))
{
warning (OPT_Wctor_dtor_privacy,
"%q#T only defines a private destructor and has no friends",
t);
return;
}
if (TYPE_HAS_USER_CONSTRUCTOR (t)
&& !CLASSTYPE_LAZY_DEFAULT_CTOR (t))
{
tree copy_or_move = NULL_TREE;
if (!TYPE_HAS_COPY_CTOR (t))
nonprivate_ctor = true;
else
for (ovl_iterator iter (CLASSTYPE_CONSTRUCTORS (t));
!nonprivate_ctor && iter; ++iter)
if (TREE_PRIVATE (*iter))
continue;
else if (copy_fn_p (*iter) || move_fn_p (*iter))
copy_or_move = *iter;
else
nonprivate_ctor = true;
if (!nonprivate_ctor)
{
warning (OPT_Wctor_dtor_privacy,
"%q#T only defines private constructors and has no friends",
t);
if (copy_or_move)
inform (DECL_SOURCE_LOCATION (copy_or_move),
"%q#D is public, but requires an existing %q#T object",
copy_or_move, t);
return;
}
}
}
static void
layout_vtable_decl (tree binfo, int n)
{
tree atype;
tree vtable;
atype = build_array_of_n_type (vtable_entry_type, n);
layout_type (atype);
vtable = get_vtbl_decl_for_binfo (binfo);
if (!same_type_p (TREE_TYPE (vtable), atype))
{
TREE_TYPE (vtable) = atype;
DECL_SIZE (vtable) = DECL_SIZE_UNIT (vtable) = NULL_TREE;
layout_decl (vtable, 0);
}
}
int
same_signature_p (const_tree fndecl, const_tree base_fndecl)
{
if (DECL_DESTRUCTOR_P (base_fndecl) && DECL_DESTRUCTOR_P (fndecl)
&& special_function_p (base_fndecl) == special_function_p (fndecl))
return 1;
if (DECL_DESTRUCTOR_P (base_fndecl) || DECL_DESTRUCTOR_P (fndecl))
return 0;
if (DECL_NAME (fndecl) == DECL_NAME (base_fndecl)
|| (DECL_CONV_FN_P (fndecl)
&& DECL_CONV_FN_P (base_fndecl)
&& same_type_p (DECL_CONV_FN_TYPE (fndecl),
DECL_CONV_FN_TYPE (base_fndecl))))
{
tree fntype = TREE_TYPE (fndecl);
tree base_fntype = TREE_TYPE (base_fndecl);
if (type_memfn_quals (fntype) == type_memfn_quals (base_fntype)
&& type_memfn_rqual (fntype) == type_memfn_rqual (base_fntype)
&& compparms (FUNCTION_FIRST_USER_PARMTYPE (fndecl),
FUNCTION_FIRST_USER_PARMTYPE (base_fndecl)))
return 1;
}
return 0;
}
static bool
base_derived_from (tree derived, tree base)
{
tree probe;
for (probe = base; probe; probe = BINFO_INHERITANCE_CHAIN (probe))
{
if (probe == derived)
return true;
else if (BINFO_VIRTUAL_P (probe))
return (binfo_for_vbase (BINFO_TYPE (probe), BINFO_TYPE (derived))
!= NULL_TREE);
}
return false;
}
struct find_final_overrider_data {
tree fn;
tree declaring_base;
tree candidates;
vec<tree> path;
};
static bool
dfs_find_final_overrider_1 (tree binfo,
find_final_overrider_data *ffod,
unsigned depth)
{
tree method;
if (depth)
{
depth--;
if (dfs_find_final_overrider_1
(ffod->path[depth], ffod, depth))
return true;
}
method = look_for_overrides_here (BINFO_TYPE (binfo), ffod->fn);
if (method)
{
tree *candidate = &ffod->candidates;
while (*candidate)
{
if (base_derived_from (TREE_VALUE (*candidate), binfo))
return true;
if (base_derived_from (binfo, TREE_VALUE (*candidate)))
*candidate = TREE_CHAIN (*candidate);
else
candidate = &TREE_CHAIN (*candidate);
}
ffod->candidates = tree_cons (method, binfo, ffod->candidates);
return true;
}
return false;
}
static tree
dfs_find_final_overrider_pre (tree binfo, void *data)
{
find_final_overrider_data *ffod = (find_final_overrider_data *) data;
if (binfo == ffod->declaring_base)
dfs_find_final_overrider_1 (binfo, ffod, ffod->path.length ());
ffod->path.safe_push (binfo);
return NULL_TREE;
}
static tree
dfs_find_final_overrider_post (tree , void *data)
{
find_final_overrider_data *ffod = (find_final_overrider_data *) data;
ffod->path.pop ();
return NULL_TREE;
}
static tree
find_final_overrider (tree derived, tree binfo, tree fn)
{
find_final_overrider_data ffod;
if (DECL_THUNK_P (fn))
fn = THUNK_TARGET (fn);
ffod.fn = fn;
ffod.declaring_base = binfo;
ffod.candidates = NULL_TREE;
ffod.path.create (30);
dfs_walk_all (derived, dfs_find_final_overrider_pre,
dfs_find_final_overrider_post, &ffod);
ffod.path.release ();
if (!ffod.candidates || TREE_CHAIN (ffod.candidates))
return error_mark_node;
return ffod.candidates;
}
static tree
get_vcall_index (tree fn, tree type)
{
vec<tree_pair_s, va_gc> *indices = CLASSTYPE_VCALL_INDICES (type);
tree_pair_p p;
unsigned ix;
FOR_EACH_VEC_SAFE_ELT (indices, ix, p)
if ((DECL_DESTRUCTOR_P (fn) && DECL_DESTRUCTOR_P (p->purpose))
|| same_signature_p (fn, p->purpose))
return p->value;
gcc_unreachable ();
}
static void
update_vtable_entry_for_fn (tree t, tree binfo, tree fn, tree* virtuals,
unsigned ix)
{
tree b;
tree overrider;
tree delta;
tree virtual_base;
tree first_defn;
tree overrider_fn, overrider_target;
tree target_fn = DECL_THUNK_P (fn) ? THUNK_TARGET (fn) : fn;
tree over_return, base_return;
bool lost = false;
for (b = binfo; ; b = get_primary_binfo (b))
{
gcc_assert (b);
if (look_for_overrides_here (BINFO_TYPE (b), target_fn))
break;
if (BINFO_LOST_PRIMARY_P (b))
lost = true;
}
first_defn = b;
overrider = find_final_overrider (TYPE_BINFO (t), b, target_fn);
if (overrider == error_mark_node)
{
error ("no unique final overrider for %qD in %qT", target_fn, t);
return;
}
overrider_target = overrider_fn = TREE_PURPOSE (overrider);
over_return = TREE_TYPE (TREE_TYPE (overrider_target));
base_return = TREE_TYPE (TREE_TYPE (target_fn));
if (POINTER_TYPE_P (over_return)
&& TREE_CODE (over_return) == TREE_CODE (base_return)
&& CLASS_TYPE_P (TREE_TYPE (over_return))
&& CLASS_TYPE_P (TREE_TYPE (base_return))
&& !DECL_INVALID_OVERRIDER_P (overrider_target))
{
tree fixed_offset, virtual_offset;
over_return = TREE_TYPE (over_return);
base_return = TREE_TYPE (base_return);
if (DECL_THUNK_P (fn))
{
gcc_assert (DECL_RESULT_THUNK_P (fn));
fixed_offset = ssize_int (THUNK_FIXED_OFFSET (fn));
virtual_offset = THUNK_VIRTUAL_OFFSET (fn);
}
else
fixed_offset = virtual_offset = NULL_TREE;
if (virtual_offset)
virtual_offset = binfo_for_vbase (BINFO_TYPE (virtual_offset),
over_return);
else if (!same_type_ignoring_top_level_qualifiers_p
(over_return, base_return))
{
tree thunk_binfo = NULL_TREE;
tree base_binfo = TYPE_BINFO (base_return);
if (base_binfo)
for (thunk_binfo = TYPE_BINFO (over_return); thunk_binfo;
thunk_binfo = TREE_CHAIN (thunk_binfo))
if (SAME_BINFO_TYPE_P (BINFO_TYPE (thunk_binfo),
BINFO_TYPE (base_binfo)))
break;
gcc_assert (thunk_binfo || errorcount);
for (virtual_offset = thunk_binfo;
virtual_offset;
virtual_offset = BINFO_INHERITANCE_CHAIN (virtual_offset))
if (BINFO_VIRTUAL_P (virtual_offset))
break;
if (virtual_offset
|| (thunk_binfo && !BINFO_OFFSET_ZEROP (thunk_binfo)))
{
tree offset = fold_convert (ssizetype, BINFO_OFFSET (thunk_binfo));
if (virtual_offset)
{
offset = 
size_diffop (offset,
fold_convert (ssizetype,
BINFO_OFFSET (virtual_offset)));
}
if (fixed_offset)
fixed_offset = size_binop (PLUS_EXPR, fixed_offset, offset);
else
fixed_offset = offset;
}
}
if (fixed_offset || virtual_offset)
overrider_fn = make_thunk (overrider_target, 0,
fixed_offset, virtual_offset);
}
else
gcc_assert (DECL_INVALID_OVERRIDER_P (overrider_target) ||
!DECL_THUNK_P (fn));
if (overrider_target != overrider_fn)
{
if (BINFO_TYPE (b) == DECL_CONTEXT (overrider_target))
b = get_primary_binfo (b);
for (; ; b = get_primary_binfo (b))
{
tree main_binfo = TYPE_BINFO (BINFO_TYPE (b));
tree bv = chain_index (ix, BINFO_VIRTUALS (main_binfo));
if (!DECL_THUNK_P (TREE_VALUE (bv)))
break;
if (BINFO_LOST_PRIMARY_P (b))
lost = true;
}
first_defn = b;
}
virtual_base = NULL_TREE;
for (; b; b = BINFO_INHERITANCE_CHAIN (b))
{
if (SAME_BINFO_TYPE_P (BINFO_TYPE (b),
BINFO_TYPE (TREE_VALUE (overrider))))
break;
if (BINFO_VIRTUAL_P (b))
{
virtual_base = b;
break;
}
}
if (virtual_base)
delta = size_diffop_loc (input_location,
fold_convert (ssizetype, BINFO_OFFSET (virtual_base)),
fold_convert (ssizetype, BINFO_OFFSET (first_defn)));
else if (lost)
delta = size_zero_node;
else
delta = size_diffop_loc (input_location,
fold_convert (ssizetype,
BINFO_OFFSET (TREE_VALUE (overrider))),
fold_convert (ssizetype, BINFO_OFFSET (binfo)));
modify_vtable_entry (t, binfo, overrider_fn, delta, virtuals);
if (virtual_base)
BV_VCALL_INDEX (*virtuals)
= get_vcall_index (overrider_target, BINFO_TYPE (virtual_base));
else
BV_VCALL_INDEX (*virtuals) = NULL_TREE;
BV_LOST_PRIMARY (*virtuals) = lost;
}
static tree
dfs_modify_vtables (tree binfo, void* data)
{
tree t = (tree) data;
tree virtuals;
tree old_virtuals;
unsigned ix;
if (!TYPE_CONTAINS_VPTR_P (BINFO_TYPE (binfo)))
return dfs_skip_bases;
if (SAME_BINFO_TYPE_P (BINFO_TYPE (binfo), t)
&& !CLASSTYPE_HAS_PRIMARY_BASE_P (t))
return NULL_TREE;
if (BINFO_PRIMARY_P (binfo) && !BINFO_VIRTUAL_P (binfo))
return NULL_TREE;
make_new_vtable (t, binfo);
for (ix = 0, virtuals = BINFO_VIRTUALS (binfo),
old_virtuals = BINFO_VIRTUALS (TYPE_BINFO (BINFO_TYPE (binfo)));
virtuals;
ix++, virtuals = TREE_CHAIN (virtuals),
old_virtuals = TREE_CHAIN (old_virtuals))
update_vtable_entry_for_fn (t,
binfo,
BV_FN (old_virtuals),
&virtuals, ix);
return NULL_TREE;
}
static tree
modify_all_vtables (tree t, tree virtuals)
{
tree binfo = TYPE_BINFO (t);
tree *fnsp;
if (TYPE_CONTAINS_VPTR_P (t))
get_vtable_decl (t, false);
dfs_walk_once (binfo, dfs_modify_vtables, NULL, t);
for (fnsp = &virtuals; *fnsp; )
{
tree fn = TREE_VALUE (*fnsp);
if (!value_member (fn, BINFO_VIRTUALS (binfo))
|| DECL_VINDEX (fn) == error_mark_node)
{
BV_DELTA (*fnsp) = integer_zero_node;
BV_VCALL_INDEX (*fnsp) = NULL_TREE;
fnsp = &TREE_CHAIN (*fnsp);
}
else
*fnsp = TREE_CHAIN (*fnsp);
}
return virtuals;
}
static void
get_basefndecls (tree name, tree t, vec<tree> *base_fndecls)
{
bool found_decls = false;
for (ovl_iterator iter (get_class_binding (t, name)); iter; ++iter)
{
tree method = *iter;
if (TREE_CODE (method) == FUNCTION_DECL && DECL_VINDEX (method))
{
base_fndecls->safe_push (method);
found_decls = true;
}
}
if (found_decls)
return;
int n_baseclasses = BINFO_N_BASE_BINFOS (TYPE_BINFO (t));
for (int i = 0; i < n_baseclasses; i++)
{
tree basetype = BINFO_TYPE (BINFO_BASE_BINFO (TYPE_BINFO (t), i));
get_basefndecls (name, basetype, base_fndecls);
}
}
void
check_for_override (tree decl, tree ctype)
{
bool overrides_found = false;
if (TREE_CODE (decl) == TEMPLATE_DECL)
return;
if ((DECL_DESTRUCTOR_P (decl)
|| IDENTIFIER_VIRTUAL_P (DECL_NAME (decl))
|| DECL_CONV_FN_P (decl))
&& look_for_overrides (ctype, decl)
&& !DECL_STATIC_FUNCTION_P (decl))
{
DECL_VINDEX (decl) = decl;
overrides_found = true;
if (warn_override && !DECL_OVERRIDE_P (decl)
&& !DECL_DESTRUCTOR_P (decl))
warning_at (DECL_SOURCE_LOCATION (decl), OPT_Wsuggest_override,
"%qD can be marked override", decl);
}
if (DECL_VIRTUAL_P (decl))
{
if (!DECL_VINDEX (decl))
DECL_VINDEX (decl) = error_mark_node;
IDENTIFIER_VIRTUAL_P (DECL_NAME (decl)) = 1;
if (DECL_DESTRUCTOR_P (decl))
TYPE_HAS_NONTRIVIAL_DESTRUCTOR (ctype) = true;
}
else if (DECL_FINAL_P (decl))
error ("%q+#D marked %<final%>, but is not virtual", decl);
if (DECL_OVERRIDE_P (decl) && !overrides_found)
error ("%q+#D marked %<override%>, but does not override", decl);
}
static void
warn_hidden (tree t)
{
if (vec<tree, va_gc> *member_vec = CLASSTYPE_MEMBER_VEC (t))
for (unsigned ix = member_vec->length (); ix--;)
{
tree fns = (*member_vec)[ix];
if (!OVL_P (fns))
continue;
tree name = OVL_NAME (fns);
auto_vec<tree, 20> base_fndecls;
tree base_binfo;
tree binfo;
unsigned j;
for (binfo = TYPE_BINFO (t), j = 0;
BINFO_BASE_ITERATE (binfo, j, base_binfo); j++)
{
tree basetype = BINFO_TYPE (base_binfo);
get_basefndecls (name, basetype, &base_fndecls);
}
if (base_fndecls.is_empty ())
continue;
for (ovl_iterator iter (fns); iter; ++iter)
{
tree fndecl = *iter;
if (TREE_CODE (fndecl) == FUNCTION_DECL
&& DECL_VINDEX (fndecl))
{
for (size_t k = 0; k < base_fndecls.length (); k++)
if (base_fndecls[k]
&& same_signature_p (fndecl, base_fndecls[k]))
base_fndecls[k] = NULL_TREE;
}
}
tree base_fndecl;
FOR_EACH_VEC_ELT (base_fndecls, j, base_fndecl)
if (base_fndecl)
{
warning_at (location_of (base_fndecl),
OPT_Woverloaded_virtual,
"%qD was hidden", base_fndecl);
warning_at (location_of (fns),
OPT_Woverloaded_virtual, "  by %qD", fns);
}
}
}
static void
finish_struct_anon_r (tree field, bool complain)
{
for (tree elt = TYPE_FIELDS (TREE_TYPE (field)); elt; elt = DECL_CHAIN (elt))
{
if (DECL_ARTIFICIAL (elt)
&& (!DECL_IMPLICIT_TYPEDEF_P (elt)
|| TYPE_UNNAMED_P (TREE_TYPE (elt))))
continue;
if (complain
&& (TREE_CODE (elt) != FIELD_DECL
|| (TREE_PRIVATE (elt) || TREE_PROTECTED (elt))))
{
if (!VAR_P (elt)
&& permerror (DECL_SOURCE_LOCATION (elt),
TREE_CODE (TREE_TYPE (field)) == UNION_TYPE
? "%q#D invalid; an anonymous union may "
"only have public non-static data members"
: "%q#D invalid; an anonymous struct may "
"only have public non-static data members", elt))
{
static bool hint;
if (flag_permissive && !hint)
{
hint = true;
inform (DECL_SOURCE_LOCATION (elt),
"this flexibility is deprecated and will be removed");
}
}
}
TREE_PRIVATE (elt) = TREE_PRIVATE (field);
TREE_PROTECTED (elt) = TREE_PROTECTED (field);
if (DECL_NAME (elt) == NULL_TREE
&& ANON_AGGR_TYPE_P (TREE_TYPE (elt)))
finish_struct_anon_r (elt, false);
}
}
static void
finish_struct_anon (tree t)
{
for (tree field = TYPE_FIELDS (t); field; field = DECL_CHAIN (field))
{
if (TREE_STATIC (field))
continue;
if (TREE_CODE (field) != FIELD_DECL)
continue;
if (DECL_NAME (field) == NULL_TREE
&& ANON_AGGR_TYPE_P (TREE_TYPE (field)))
finish_struct_anon_r (field, true);
}
}
void
maybe_add_class_template_decl_list (tree type, tree t, int friend_p)
{
if (CLASSTYPE_TEMPLATE_INFO (type))
CLASSTYPE_DECL_LIST (type)
= tree_cons (friend_p ? NULL_TREE : type,
t, CLASSTYPE_DECL_LIST (type));
}
static tree
dfs_declare_virt_assop_and_dtor (tree binfo, void *data)
{
tree bv, fn, t = (tree)data;
tree opname = assign_op_identifier;
gcc_assert (t && CLASS_TYPE_P (t));
gcc_assert (binfo && TREE_CODE (binfo) == TREE_BINFO);
if (!TYPE_CONTAINS_VPTR_P (BINFO_TYPE (binfo)))
return dfs_skip_bases;
if (BINFO_PRIMARY_P (binfo))
return NULL_TREE;
for (bv = BINFO_VIRTUALS (binfo); bv; bv = TREE_CHAIN (bv))
{
fn = BV_FN (bv);
if (DECL_NAME (fn) == opname)
{
if (CLASSTYPE_LAZY_COPY_ASSIGN (t))
lazily_declare_fn (sfk_copy_assignment, t);
if (CLASSTYPE_LAZY_MOVE_ASSIGN (t))
lazily_declare_fn (sfk_move_assignment, t);
}
else if (DECL_DESTRUCTOR_P (fn)
&& CLASSTYPE_LAZY_DESTRUCTOR (t))
lazily_declare_fn (sfk_destructor, t);
}
return NULL_TREE;
}
static void
declare_virt_assop_and_dtor (tree t)
{
if (!(TYPE_POLYMORPHIC_P (t)
&& (CLASSTYPE_LAZY_COPY_ASSIGN (t)
|| CLASSTYPE_LAZY_MOVE_ASSIGN (t)
|| CLASSTYPE_LAZY_DESTRUCTOR (t))))
return;
dfs_walk_all (TYPE_BINFO (t),
dfs_declare_virt_assop_and_dtor,
NULL, t);
}
static void
one_inheriting_sig (tree t, tree ctor, tree *parms, int nparms)
{
gcc_assert (TYPE_MAIN_VARIANT (t) == t);
if (nparms == 0)
return;
if (nparms == 1
&& TREE_CODE (parms[0]) == REFERENCE_TYPE)
{
tree parm = TYPE_MAIN_VARIANT (TREE_TYPE (parms[0]));
if (parm == t || parm == DECL_CONTEXT (ctor))
return;
}
tree parmlist = void_list_node;
for (int i = nparms - 1; i >= 0; i--)
parmlist = tree_cons (NULL_TREE, parms[i], parmlist);
tree fn = implicitly_declare_fn (sfk_inheriting_constructor,
t, false, ctor, parmlist);
if (add_method (t, fn, false))
{
DECL_CHAIN (fn) = TYPE_FIELDS (t);
TYPE_FIELDS (t) = fn;
}
}
static void
one_inherited_ctor (tree ctor, tree t, tree using_decl)
{
tree parms = FUNCTION_FIRST_USER_PARMTYPE (ctor);
if (flag_new_inheriting_ctors)
{
ctor = implicitly_declare_fn (sfk_inheriting_constructor,
t, false, ctor, parms);
add_method (t, ctor, using_decl != NULL_TREE);
TYPE_HAS_USER_CONSTRUCTOR (t) = true;
return;
}
tree *new_parms = XALLOCAVEC (tree, list_length (parms));
int i = 0;
for (; parms && parms != void_list_node; parms = TREE_CHAIN (parms))
{
if (TREE_PURPOSE (parms))
one_inheriting_sig (t, ctor, new_parms, i);
new_parms[i++] = TREE_VALUE (parms);
}
one_inheriting_sig (t, ctor, new_parms, i);
if (parms == NULL_TREE)
{
if (warning (OPT_Winherited_variadic_ctor,
"the ellipsis in %qD is not inherited", ctor))
inform (DECL_SOURCE_LOCATION (ctor), "%qD declared here", ctor);
}
}
static void
add_implicitly_declared_members (tree t, tree* access_decls,
int cant_have_const_cctor,
int cant_have_const_assignment)
{
if (!CLASSTYPE_DESTRUCTOR (t))
CLASSTYPE_LAZY_DESTRUCTOR (t) = 1;
bool move_ok = false;
if (cxx_dialect >= cxx11 && CLASSTYPE_LAZY_DESTRUCTOR (t)
&& !TYPE_HAS_COPY_CTOR (t) && !TYPE_HAS_COPY_ASSIGN (t)
&& !classtype_has_move_assign_or_move_ctor_p (t, false))
move_ok = true;
if (! TYPE_HAS_USER_CONSTRUCTOR (t))
{
TYPE_HAS_DEFAULT_CONSTRUCTOR (t) = 1;
CLASSTYPE_LAZY_DEFAULT_CTOR (t) = 1;
if (cxx_dialect >= cxx11)
TYPE_HAS_CONSTEXPR_CTOR (t)
= type_maybe_constexpr_default_constructor (t);
}
if (! TYPE_HAS_COPY_CTOR (t))
{
TYPE_HAS_COPY_CTOR (t) = 1;
TYPE_HAS_CONST_COPY_CTOR (t) = !cant_have_const_cctor;
CLASSTYPE_LAZY_COPY_CTOR (t) = 1;
if (move_ok)
CLASSTYPE_LAZY_MOVE_CTOR (t) = 1;
}
if (!TYPE_HAS_COPY_ASSIGN (t))
{
TYPE_HAS_COPY_ASSIGN (t) = 1;
TYPE_HAS_CONST_COPY_ASSIGN (t) = !cant_have_const_assignment;
CLASSTYPE_LAZY_COPY_ASSIGN (t) = 1;
if (move_ok && !LAMBDA_TYPE_P (t))
CLASSTYPE_LAZY_MOVE_ASSIGN (t) = 1;
}
declare_virt_assop_and_dtor (t);
while (*access_decls)
{
tree using_decl = TREE_VALUE (*access_decls);
tree decl = USING_DECL_DECLS (using_decl);
if (DECL_NAME (using_decl) == ctor_identifier)
{
tree ctor_list = decl;
location_t loc = input_location;
input_location = DECL_SOURCE_LOCATION (using_decl);
for (ovl_iterator iter (ctor_list); iter; ++iter)
one_inherited_ctor (*iter, t, using_decl);
*access_decls = TREE_CHAIN (*access_decls);
input_location = loc;
}
else
access_decls = &TREE_CHAIN (*access_decls);
}
}
static bool
check_bitfield_decl (tree field)
{
tree type = TREE_TYPE (field);
tree w;
w = DECL_BIT_FIELD_REPRESENTATIVE (field);
gcc_assert (w != NULL_TREE);
DECL_BIT_FIELD_REPRESENTATIVE (field) = NULL_TREE;
if (!INTEGRAL_OR_ENUMERATION_TYPE_P (type))
{
error ("bit-field %q+#D with non-integral type", field);
w = error_mark_node;
}
else
{
location_t loc = input_location;
STRIP_NOPS (w);
input_location = DECL_SOURCE_LOCATION (field);
w = cxx_constant_value (w);
input_location = loc;
if (TREE_CODE (w) != INTEGER_CST)
{
error ("bit-field %q+D width not an integer constant", field);
w = error_mark_node;
}
else if (tree_int_cst_sgn (w) < 0)
{
error ("negative width in bit-field %q+D", field);
w = error_mark_node;
}
else if (integer_zerop (w) && DECL_NAME (field) != 0)
{
error ("zero width for bit-field %q+D", field);
w = error_mark_node;
}
else if ((TREE_CODE (type) != ENUMERAL_TYPE
&& TREE_CODE (type) != BOOLEAN_TYPE
&& compare_tree_int (w, TYPE_PRECISION (type)) > 0)
|| ((TREE_CODE (type) == ENUMERAL_TYPE
|| TREE_CODE (type) == BOOLEAN_TYPE)
&& tree_int_cst_lt (TYPE_SIZE (type), w)))
warning_at (DECL_SOURCE_LOCATION (field), 0,
"width of %qD exceeds its type", field);
else if (TREE_CODE (type) == ENUMERAL_TYPE)
{
int prec = TYPE_PRECISION (ENUM_UNDERLYING_TYPE (type));
if (compare_tree_int (w, prec) < 0)
warning_at (DECL_SOURCE_LOCATION (field), 0,
"%qD is too small to hold all values of %q#T",
field, type);
}
}
if (w != error_mark_node)
{
DECL_SIZE (field) = fold_convert (bitsizetype, w);
DECL_BIT_FIELD (field) = 1;
return true;
}
else
{
DECL_BIT_FIELD (field) = 0;
CLEAR_DECL_C_BIT_FIELD (field);
return false;
}
}
static bool
check_field_decl (tree field,
tree t,
int* cant_have_const_ctor,
int* no_const_asn_ref)
{
tree type = strip_array_types (TREE_TYPE (field));
bool any_default_members = false;
if (ANON_UNION_TYPE_P (type) && cxx_dialect < cxx11)
;
else if (ANON_AGGR_TYPE_P (type))
{
for (tree fields = TYPE_FIELDS (type); fields;
fields = DECL_CHAIN (fields))
if (TREE_CODE (fields) == FIELD_DECL)
any_default_members |= check_field_decl (fields, t,
cant_have_const_ctor,
no_const_asn_ref);
}
else if (CLASS_TYPE_P (type))
{
abstract_virtuals_error (field, type);
if (TREE_CODE (t) == UNION_TYPE && cxx_dialect < cxx11)
{
static bool warned;
int oldcount = errorcount;
if (TYPE_NEEDS_CONSTRUCTING (type))
error ("member %q+#D with constructor not allowed in union",
field);
if (TYPE_HAS_NONTRIVIAL_DESTRUCTOR (type))
error ("member %q+#D with destructor not allowed in union", field);
if (TYPE_HAS_COMPLEX_COPY_ASSIGN (type))
error ("member %q+#D with copy assignment operator not allowed in union",
field);
if (!warned && errorcount > oldcount)
{
inform (DECL_SOURCE_LOCATION (field), "unrestricted unions "
"only available with -std=c++11 or -std=gnu++11");
warned = true;
}
}
else
{
TYPE_NEEDS_CONSTRUCTING (t) |= TYPE_NEEDS_CONSTRUCTING (type);
TYPE_HAS_NONTRIVIAL_DESTRUCTOR (t)
|= TYPE_HAS_NONTRIVIAL_DESTRUCTOR (type);
TYPE_HAS_COMPLEX_COPY_ASSIGN (t)
|= (TYPE_HAS_COMPLEX_COPY_ASSIGN (type)
|| !TYPE_HAS_COPY_ASSIGN (type));
TYPE_HAS_COMPLEX_COPY_CTOR (t) |= (TYPE_HAS_COMPLEX_COPY_CTOR (type)
|| !TYPE_HAS_COPY_CTOR (type));
TYPE_HAS_COMPLEX_MOVE_ASSIGN (t) |= TYPE_HAS_COMPLEX_MOVE_ASSIGN (type);
TYPE_HAS_COMPLEX_MOVE_CTOR (t) |= TYPE_HAS_COMPLEX_MOVE_CTOR (type);
TYPE_HAS_COMPLEX_DFLT (t) |= (!TYPE_HAS_DEFAULT_CONSTRUCTOR (type)
|| TYPE_HAS_COMPLEX_DFLT (type));
}
if (TYPE_HAS_COPY_CTOR (type)
&& !TYPE_HAS_CONST_COPY_CTOR (type))
*cant_have_const_ctor = 1;
if (TYPE_HAS_COPY_ASSIGN (type)
&& !TYPE_HAS_CONST_COPY_ASSIGN (type))
*no_const_asn_ref = 1;
}
check_abi_tags (t, field);
if (DECL_INITIAL (field) != NULL_TREE)
any_default_members = true;
return any_default_members;
}
static void
check_field_decls (tree t, tree *access_decls,
int *cant_have_const_ctor_p,
int *no_const_asn_ref_p)
{
tree *field;
tree *next;
bool has_pointers;
bool any_default_members;
int cant_pack = 0;
int field_access = -1;
*access_decls = NULL_TREE;
has_pointers = false;
any_default_members = false;
for (field = &TYPE_FIELDS (t); *field; field = next)
{
tree x = *field;
tree type = TREE_TYPE (x);
int this_field_access;
next = &DECL_CHAIN (x);
if (TREE_CODE (x) == USING_DECL)
{
*access_decls = tree_cons (NULL_TREE, x, *access_decls);
continue;
}
if (TREE_CODE (x) == TYPE_DECL
|| TREE_CODE (x) == TEMPLATE_DECL)
continue;
if (TREE_CODE (x) == FUNCTION_DECL)
continue;
if (TREE_CODE (x) != CONST_DECL)
DECL_CONTEXT (x) = t;
DECL_NONLOCAL (x) = 1;
if (TREE_CODE (t) == UNION_TYPE)
{
if (VAR_P (x) && cxx_dialect < cxx11)
{
error ("in C++98 %q+D may not be static because it is "
"a member of a union", x);
continue;
}
if (TREE_CODE (type) == REFERENCE_TYPE
&& TREE_CODE (x) == FIELD_DECL)
{
error ("non-static data member %q+D in a union may not "
"have reference type %qT", x, type);
continue;
}
}
if (TREE_CODE (type) == FUNCTION_TYPE)
{
error ("field %q+D invalidly declared function type", x);
type = build_pointer_type (type);
TREE_TYPE (x) = type;
}
else if (TREE_CODE (type) == METHOD_TYPE)
{
error ("field %q+D invalidly declared method type", x);
type = build_pointer_type (type);
TREE_TYPE (x) = type;
}
if (type == error_mark_node)
continue;
if (TREE_CODE (x) == CONST_DECL || VAR_P (x))
continue;
if (TREE_PRIVATE (x) || TREE_PROTECTED (x))
CLASSTYPE_NON_AGGREGATE (t) = 1;
if (COMPLETE_TYPE_P (type)
&& (!literal_type_p (type) || CP_TYPE_VOLATILE_P (type))) 
CLASSTYPE_LITERAL_P (t) = false;
this_field_access = TREE_PROTECTED (x) ? 1 : TREE_PRIVATE (x) ? 2 : 0;
if (field_access == -1)
field_access = this_field_access;
else if (this_field_access != field_access)
CLASSTYPE_NON_STD_LAYOUT (t) = 1;
if (TREE_CODE (type) == REFERENCE_TYPE)
{
CLASSTYPE_NON_LAYOUT_POD_P (t) = 1;
CLASSTYPE_NON_STD_LAYOUT (t) = 1;
if (DECL_INITIAL (x) == NULL_TREE)
SET_CLASSTYPE_REF_FIELDS_NEED_INIT (t, 1);
if (cxx_dialect < cxx11)
{
TYPE_HAS_COMPLEX_COPY_ASSIGN (t) = 1;
TYPE_HAS_COMPLEX_MOVE_ASSIGN (t) = 1;
}
}
type = strip_array_types (type);
if (TYPE_PACKED (t))
{
if (!layout_pod_type_p (type) && !TYPE_PACKED (type))
{
warning_at
(DECL_SOURCE_LOCATION (x), 0,
"ignoring packed attribute because of unpacked non-POD field %q#D",
x);
cant_pack = 1;
}
else if (DECL_C_BIT_FIELD (x)
|| TYPE_ALIGN (TREE_TYPE (x)) > BITS_PER_UNIT)
DECL_PACKED (x) = 1;
}
if (DECL_C_BIT_FIELD (x)
&& integer_zerop (DECL_BIT_FIELD_REPRESENTATIVE (x)))
;
else
{
CLASSTYPE_EMPTY_P (t) = 0;
CLASSTYPE_NEARLY_EMPTY_P (t) = 0;
if (CLASS_TYPE_P (type)
&& CLASSTYPE_CONTAINS_EMPTY_CLASS_P (type))
CLASSTYPE_CONTAINS_EMPTY_CLASS_P (t) = 1;
}
if (TYPE_PTR_P (type)
&& !TYPE_PTRFN_P (type))
has_pointers = true;
if (CLASS_TYPE_P (type))
{
if (CLASSTYPE_REF_FIELDS_NEED_INIT (type))
SET_CLASSTYPE_REF_FIELDS_NEED_INIT (t, 1);
if (CLASSTYPE_READONLY_FIELDS_NEED_INIT (type))
SET_CLASSTYPE_READONLY_FIELDS_NEED_INIT (t, 1);
}
if (DECL_MUTABLE_P (x) || TYPE_HAS_MUTABLE_P (type))
CLASSTYPE_HAS_MUTABLE (t) = 1;
if (DECL_MUTABLE_P (x))
{
if (CP_TYPE_CONST_P (type))
{
error ("member %q+D cannot be declared both %<const%> "
"and %<mutable%>", x);
continue;
}
if (TREE_CODE (type) == REFERENCE_TYPE)
{
error ("member %q+D cannot be declared as a %<mutable%> "
"reference", x);
continue;
}
}
if (! layout_pod_type_p (type))
CLASSTYPE_NON_LAYOUT_POD_P (t) = 1;
if (!std_layout_type_p (type))
CLASSTYPE_NON_STD_LAYOUT (t) = 1;
if (! zero_init_p (type))
CLASSTYPE_NON_ZERO_INIT_P (t) = 1;
if (DECL_C_BIT_FIELD (x))
check_bitfield_decl (x);
if (check_field_decl (x, t, cant_have_const_ctor_p, no_const_asn_ref_p))
{
if (any_default_members
&& TREE_CODE (t) == UNION_TYPE)
error ("multiple fields in union %qT initialized", t);
any_default_members = true;
}
if (DECL_INITIAL (x) && cxx_dialect < cxx14)
CLASSTYPE_NON_AGGREGATE (t) = true;
if (CP_TYPE_CONST_P (type))
{
C_TYPE_FIELDS_READONLY (t) = 1;
if (DECL_INITIAL (x) == NULL_TREE)
SET_CLASSTYPE_READONLY_FIELDS_NEED_INIT (t, 1);
if (cxx_dialect < cxx11)
{
TYPE_HAS_COMPLEX_COPY_ASSIGN (t) = 1;
TYPE_HAS_COMPLEX_MOVE_ASSIGN (t) = 1;
}
}
else if (CLASS_TYPE_P (type))
{
C_TYPE_FIELDS_READONLY (t) |= C_TYPE_FIELDS_READONLY (type);
SET_CLASSTYPE_READONLY_FIELDS_NEED_INIT (t,
CLASSTYPE_READONLY_FIELDS_NEED_INIT (t)
| CLASSTYPE_READONLY_FIELDS_NEED_INIT (type));
}
if (constructor_name_p (DECL_NAME (x), t)
&& TYPE_HAS_USER_CONSTRUCTOR (t))
permerror (DECL_SOURCE_LOCATION (x),
"field %q#D with same name as class", x);
}
if (warn_ecpp
&& has_pointers
&& TYPE_HAS_USER_CONSTRUCTOR (t)
&& TYPE_HAS_NONTRIVIAL_DESTRUCTOR (t)
&& !(TYPE_HAS_COPY_CTOR (t) && TYPE_HAS_COPY_ASSIGN (t)))
{
warning (OPT_Weffc__, "%q#T has pointer data members", t);
if (! TYPE_HAS_COPY_CTOR (t))
{
warning (OPT_Weffc__,
"  but does not override %<%T(const %T&)%>", t, t);
if (!TYPE_HAS_COPY_ASSIGN (t))
warning (OPT_Weffc__, "  or %<operator=(const %T&)%>", t);
}
else if (! TYPE_HAS_COPY_ASSIGN (t))
warning (OPT_Weffc__,
"  but does not override %<operator=(const %T&)%>", t);
}
if (any_default_members)
{
TYPE_NEEDS_CONSTRUCTING (t) = true;
TYPE_HAS_COMPLEX_DFLT (t) = true;
}
if (cant_pack)
TYPE_PACKED (t) = 0;
finish_struct_anon (t);
*access_decls = nreverse (*access_decls);
}
static int
record_subobject_offset (tree type, tree offset, splay_tree offsets)
{
splay_tree_node n;
if (!is_empty_class (type))
return 0;
n = splay_tree_lookup (offsets, (splay_tree_key) offset);
if (!n)
n = splay_tree_insert (offsets,
(splay_tree_key) offset,
(splay_tree_value) NULL_TREE);
n->value = ((splay_tree_value)
tree_cons (NULL_TREE,
type,
(tree) n->value));
return 0;
}
static int
check_subobject_offset (tree type, tree offset, splay_tree offsets)
{
splay_tree_node n;
tree t;
if (!is_empty_class (type))
return 0;
n = splay_tree_lookup (offsets, (splay_tree_key) offset);
if (!n)
return 0;
for (t = (tree) n->value; t; t = TREE_CHAIN (t))
if (same_type_p (TREE_VALUE (t), type))
return 1;
return 0;
}
static int
walk_subobject_offsets (tree type,
subobject_offset_fn f,
tree offset,
splay_tree offsets,
tree max_offset,
int vbases_p)
{
int r = 0;
tree type_binfo = NULL_TREE;
if (max_offset && tree_int_cst_lt (max_offset, offset))
return 0;
if (type == error_mark_node)
return 0;
if (!TYPE_P (type))
{
type_binfo = type;
type = BINFO_TYPE (type);
}
if (CLASS_TYPE_P (type))
{
tree field;
tree binfo;
int i;
if (!CLASSTYPE_CONTAINS_EMPTY_CLASS_P (type))
return 0;
r = (*f) (type, offset, offsets);
if (r)
return r;
if (!type_binfo)
type_binfo = TYPE_BINFO (type);
for (i = 0; BINFO_BASE_ITERATE (type_binfo, i, binfo); i++)
{
tree binfo_offset;
if (BINFO_VIRTUAL_P (binfo))
continue;
tree orig_binfo;
orig_binfo = BINFO_BASE_BINFO (TYPE_BINFO (type), i);
binfo_offset = size_binop (PLUS_EXPR,
offset,
BINFO_OFFSET (orig_binfo));
r = walk_subobject_offsets (binfo,
f,
binfo_offset,
offsets,
max_offset,
0);
if (r)
return r;
}
if (CLASSTYPE_VBASECLASSES (type))
{
unsigned ix;
vec<tree, va_gc> *vbases;
if (vbases_p)
for (vbases = CLASSTYPE_VBASECLASSES (type), ix = 0;
vec_safe_iterate (vbases, ix, &binfo); ix++)
{
r = walk_subobject_offsets (binfo,
f,
size_binop (PLUS_EXPR,
offset,
BINFO_OFFSET (binfo)),
offsets,
max_offset,
0);
if (r)
return r;
}
else
{
tree vbase = get_primary_binfo (type_binfo);
if (vbase && BINFO_VIRTUAL_P (vbase)
&& BINFO_PRIMARY_P (vbase)
&& BINFO_INHERITANCE_CHAIN (vbase) == type_binfo)
{
r = (walk_subobject_offsets
(vbase, f, offset,
offsets, max_offset, 0));
if (r)
return r;
}
}
}
for (field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
if (TREE_CODE (field) == FIELD_DECL
&& TREE_TYPE (field) != error_mark_node
&& !DECL_ARTIFICIAL (field))
{
tree field_offset;
field_offset = byte_position (field);
r = walk_subobject_offsets (TREE_TYPE (field),
f,
size_binop (PLUS_EXPR,
offset,
field_offset),
offsets,
max_offset,
1);
if (r)
return r;
}
}
else if (TREE_CODE (type) == ARRAY_TYPE)
{
tree element_type = strip_array_types (type);
tree domain = TYPE_DOMAIN (type);
tree index;
if (!CLASS_TYPE_P (element_type)
|| !CLASSTYPE_CONTAINS_EMPTY_CLASS_P (element_type)
|| !domain
|| integer_minus_onep (TYPE_MAX_VALUE (domain)))
return 0;
for (index = size_zero_node;
!tree_int_cst_lt (TYPE_MAX_VALUE (domain), index);
index = size_binop (PLUS_EXPR, index, size_one_node))
{
r = walk_subobject_offsets (TREE_TYPE (type),
f,
offset,
offsets,
max_offset,
1);
if (r)
return r;
offset = size_binop (PLUS_EXPR, offset,
TYPE_SIZE_UNIT (TREE_TYPE (type)));
if (max_offset && tree_int_cst_lt (max_offset, offset))
break;
}
}
return 0;
}
static void
record_subobject_offsets (tree type,
tree offset,
splay_tree offsets,
bool is_data_member)
{
tree max_offset;
if (is_data_member
|| !is_empty_class (BINFO_TYPE (type)))
max_offset = sizeof_biggest_empty_class;
else
max_offset = NULL_TREE;
walk_subobject_offsets (type, record_subobject_offset, offset,
offsets, max_offset, is_data_member);
}
static int
layout_conflict_p (tree type,
tree offset,
splay_tree offsets,
int vbases_p)
{
splay_tree_node max_node;
max_node = splay_tree_max (offsets);
if (!max_node)
return 0;
return walk_subobject_offsets (type, check_subobject_offset, offset,
offsets, (tree) (max_node->key),
vbases_p);
}
static void
layout_nonempty_base_or_field (record_layout_info rli,
tree decl,
tree binfo,
splay_tree offsets)
{
tree offset = NULL_TREE;
bool field_p;
tree type;
if (binfo)
{
type = TREE_TYPE (binfo);
field_p = false;
}
else
{
type = TREE_TYPE (decl);
field_p = true;
}
while (1)
{
struct record_layout_info_s old_rli = *rli;
place_field (rli, decl);
offset = byte_position (decl);
if (TREE_CODE (rli->t) == UNION_TYPE)
break;
if (layout_conflict_p (field_p ? type : binfo, offset,
offsets, field_p))
{
*rli = old_rli;
rli->bitpos
= size_binop (PLUS_EXPR, rli->bitpos,
bitsize_int (binfo
? CLASSTYPE_ALIGN (type)
: TYPE_ALIGN (type)));
normalize_rli (rli);
}
else if (TREE_CODE (type) == NULLPTR_TYPE
&& warn_abi && abi_version_crosses (9))
{
tree pos = rli_size_unit_so_far (&old_rli);
if (int_cst_value (pos) % TYPE_ALIGN_UNIT (ptr_type_node) != 0)
{
if (abi_version_at_least (9))
warning_at (DECL_SOURCE_LOCATION (decl), OPT_Wabi,
"alignment of %qD increased in -fabi-version=9 "
"(GCC 5.2)", decl);
else
warning_at (DECL_SOURCE_LOCATION (decl), OPT_Wabi, "alignment "
"of %qD will increase in -fabi-version=9", decl);
}
break;
}
else
break;
}
if (binfo && CLASS_TYPE_P (BINFO_TYPE (binfo)))
propagate_binfo_offsets (binfo,
size_diffop_loc (input_location,
fold_convert (ssizetype, offset),
fold_convert (ssizetype,
BINFO_OFFSET (binfo))));
}
static int
empty_base_at_nonzero_offset_p (tree type,
tree offset,
splay_tree )
{
return is_empty_class (type) && !integer_zerop (offset);
}
static bool
layout_empty_base (record_layout_info rli, tree binfo,
tree eoc, splay_tree offsets)
{
tree alignment;
tree basetype = BINFO_TYPE (binfo);
bool atend = false;
gcc_assert (is_empty_class (basetype));
alignment = ssize_int (CLASSTYPE_ALIGN_UNIT (basetype));
if (!integer_zerop (BINFO_OFFSET (binfo)))
propagate_binfo_offsets
(binfo, size_diffop_loc (input_location,
size_zero_node, BINFO_OFFSET (binfo)));
if (layout_conflict_p (binfo,
BINFO_OFFSET (binfo),
offsets,
0))
{
atend = true;
propagate_binfo_offsets (binfo, fold_convert (ssizetype, eoc));
while (1)
{
if (!layout_conflict_p (binfo,
BINFO_OFFSET (binfo),
offsets,
0))
break;
propagate_binfo_offsets (binfo, alignment);
}
}
if (CLASSTYPE_USER_ALIGN (basetype))
{
rli->record_align = MAX (rli->record_align, CLASSTYPE_ALIGN (basetype));
if (warn_packed)
rli->unpacked_align = MAX (rli->unpacked_align, CLASSTYPE_ALIGN (basetype));
TYPE_USER_ALIGN (rli->t) = 1;
}
return atend;
}
static tree
build_base_field_1 (tree t, tree basetype, tree *&next_field)
{
gcc_assert (CLASSTYPE_AS_BASE (basetype));
tree decl = build_decl (input_location,
FIELD_DECL, NULL_TREE, CLASSTYPE_AS_BASE (basetype));
DECL_ARTIFICIAL (decl) = 1;
DECL_IGNORED_P (decl) = 1;
DECL_FIELD_CONTEXT (decl) = t;
if (is_empty_class (basetype))
DECL_SIZE (decl) = DECL_SIZE_UNIT (decl) = size_zero_node;
else
{
DECL_SIZE (decl) = CLASSTYPE_SIZE (basetype);
DECL_SIZE_UNIT (decl) = CLASSTYPE_SIZE_UNIT (basetype);
}
SET_DECL_ALIGN (decl, CLASSTYPE_ALIGN (basetype));
DECL_USER_ALIGN (decl) = CLASSTYPE_USER_ALIGN (basetype);
SET_DECL_MODE (decl, TYPE_MODE (basetype));
DECL_FIELD_IS_BASE (decl) = 1;
DECL_CHAIN (decl) = *next_field;
*next_field = decl;
next_field = &DECL_CHAIN (decl);
return decl;
}
static tree *
build_base_field (record_layout_info rli, tree binfo,
splay_tree offsets, tree *next_field)
{
tree t = rli->t;
tree basetype = BINFO_TYPE (binfo);
if (!COMPLETE_TYPE_P (basetype))
return next_field;
if (!is_empty_class (basetype))
{
tree decl;
CLASSTYPE_EMPTY_P (t) = 0;
decl = build_base_field_1 (t, basetype, next_field);
layout_nonempty_base_or_field (rli, decl, binfo, offsets);
}
else
{
tree eoc;
bool atend;
eoc = round_up_loc (input_location,
rli_size_unit_so_far (rli),
CLASSTYPE_ALIGN_UNIT (basetype));
atend = layout_empty_base (rli, binfo, eoc, offsets);
if (!BINFO_VIRTUAL_P (binfo) && CLASSTYPE_NEARLY_EMPTY_P (t))
{
if (atend)
CLASSTYPE_NEARLY_EMPTY_P (t) = 0;
else if (walk_subobject_offsets (basetype,
empty_base_at_nonzero_offset_p,
size_zero_node,
NULL,
NULL_TREE,
true))
CLASSTYPE_NEARLY_EMPTY_P (t) = 0;
}
if (cxx_dialect >= cxx17 && !BINFO_VIRTUAL_P (binfo))
{
tree decl = build_base_field_1 (t, basetype, next_field);
DECL_FIELD_OFFSET (decl) = BINFO_OFFSET (binfo);
DECL_FIELD_BIT_OFFSET (decl) = bitsize_zero_node;
SET_DECL_OFFSET_ALIGN (decl, BITS_PER_UNIT);
}
}
record_subobject_offsets (binfo,
BINFO_OFFSET (binfo),
offsets,
false);
return next_field;
}
static void
build_base_fields (record_layout_info rli,
splay_tree offsets, tree *next_field)
{
tree t = rli->t;
int n_baseclasses = BINFO_N_BASE_BINFOS (TYPE_BINFO (t));
int i;
if (CLASSTYPE_HAS_PRIMARY_BASE_P (t))
next_field = build_base_field (rli, CLASSTYPE_PRIMARY_BINFO (t),
offsets, next_field);
for (i = 0; i < n_baseclasses; ++i)
{
tree base_binfo;
base_binfo = BINFO_BASE_BINFO (TYPE_BINFO (t), i);
if (base_binfo == CLASSTYPE_PRIMARY_BINFO (t))
continue;
if (BINFO_VIRTUAL_P (base_binfo))
continue;
next_field = build_base_field (rli, base_binfo,
offsets, next_field);
}
}
static void
check_methods (tree t)
{
for (tree x = TYPE_FIELDS (t); x; x = DECL_CHAIN (x))
if (DECL_DECLARES_FUNCTION_P (x))
{
check_for_override (x, t);
if (DECL_PURE_VIRTUAL_P (x)
&& (TREE_CODE (x) != FUNCTION_DECL || ! DECL_VINDEX (x)))
error ("initializer specified for non-virtual method %q+D", x);
if (TREE_CODE (x) == FUNCTION_DECL && DECL_VINDEX (x))
{
TYPE_POLYMORPHIC_P (t) = 1;
if (DECL_PURE_VIRTUAL_P (x))
vec_safe_push (CLASSTYPE_PURE_VIRTUALS (t), x);
}
if (DECL_DESTRUCTOR_P (x) && user_provided_p (x))
TYPE_HAS_NONTRIVIAL_DESTRUCTOR (t) = 1;
if (!DECL_VIRTUAL_P (x)
&& lookup_attribute ("transaction_safe_dynamic",
DECL_ATTRIBUTES (x)))
error_at (DECL_SOURCE_LOCATION (x),
"%<transaction_safe_dynamic%> may only be specified for "
"a virtual function");
}
}
static tree
build_clone (tree fn, tree name)
{
tree parms;
tree clone;
clone = copy_decl (fn);
DECL_NAME (clone) = name;
DECL_ABSTRACT_ORIGIN (clone) = fn;
DECL_CHAIN (clone) = DECL_CHAIN (fn);
DECL_CHAIN (fn) = clone;
if (TREE_CODE (clone) == TEMPLATE_DECL)
{
tree result = build_clone (DECL_TEMPLATE_RESULT (clone), name);
DECL_TEMPLATE_RESULT (clone) = result;
DECL_TEMPLATE_INFO (result) = copy_node (DECL_TEMPLATE_INFO (result));
DECL_TI_TEMPLATE (result) = clone;
TREE_TYPE (clone) = TREE_TYPE (result);
return clone;
}
else
{
if (flag_concepts)
if (tree ci = get_constraints (fn))
set_constraints (clone, copy_node (ci));
}
SET_DECL_ASSEMBLER_NAME (clone, NULL_TREE);
DECL_CLONED_FUNCTION (clone) = fn;
DECL_PENDING_INLINE_INFO (clone) = NULL;
DECL_PENDING_INLINE_P (clone) = 0;
if (name == base_dtor_identifier)
{
DECL_VIRTUAL_P (clone) = 0;
if (TREE_CODE (clone) != TEMPLATE_DECL)
DECL_VINDEX (clone) = NULL_TREE;
}
bool ctor_omit_inherited_parms_p = ctor_omit_inherited_parms (clone);
if (ctor_omit_inherited_parms_p)
gcc_assert (DECL_HAS_IN_CHARGE_PARM_P (clone));
if (DECL_HAS_IN_CHARGE_PARM_P (clone))
{
tree basetype;
tree parmtypes;
tree exceptions;
exceptions = TYPE_RAISES_EXCEPTIONS (TREE_TYPE (clone));
basetype = TYPE_METHOD_BASETYPE (TREE_TYPE (clone));
parmtypes = TYPE_ARG_TYPES (TREE_TYPE (clone));
parmtypes = TREE_CHAIN (parmtypes);
parmtypes = TREE_CHAIN (parmtypes);
if (DECL_HAS_VTT_PARM_P (fn)
&& ! DECL_NEEDS_VTT_PARM_P (clone))
parmtypes = TREE_CHAIN (parmtypes);
if (ctor_omit_inherited_parms_p)
{
gcc_assert (DECL_NEEDS_VTT_PARM_P (clone));
parmtypes = tree_cons (NULL_TREE, vtt_parm_type, void_list_node);
}
TREE_TYPE (clone)
= build_method_type_directly (basetype,
TREE_TYPE (TREE_TYPE (clone)),
parmtypes);
if (exceptions)
TREE_TYPE (clone) = build_exception_variant (TREE_TYPE (clone),
exceptions);
TREE_TYPE (clone)
= cp_build_type_attribute_variant (TREE_TYPE (clone),
TYPE_ATTRIBUTES (TREE_TYPE (fn)));
}
DECL_ARGUMENTS (clone) = copy_list (DECL_ARGUMENTS (clone));
if (DECL_HAS_IN_CHARGE_PARM_P (clone))
{
DECL_CHAIN (DECL_ARGUMENTS (clone))
= DECL_CHAIN (DECL_CHAIN (DECL_ARGUMENTS (clone)));
DECL_HAS_IN_CHARGE_PARM_P (clone) = 0;
}
if (DECL_HAS_VTT_PARM_P (fn))
{
if (DECL_NEEDS_VTT_PARM_P (clone))
DECL_HAS_VTT_PARM_P (clone) = 1;
else
{
DECL_CHAIN (DECL_ARGUMENTS (clone))
= DECL_CHAIN (DECL_CHAIN (DECL_ARGUMENTS (clone)));
DECL_HAS_VTT_PARM_P (clone) = 0;
}
}
if (ctor_omit_inherited_parms_p)
DECL_CHAIN (DECL_CHAIN (DECL_ARGUMENTS (clone))) = NULL_TREE;
for (parms = DECL_ARGUMENTS (clone); parms; parms = DECL_CHAIN (parms))
{
DECL_CONTEXT (parms) = clone;
cxx_dup_lang_specific_decl (parms);
}
SET_DECL_RTL (clone, NULL);
rest_of_decl_compilation (clone, 1, at_eof);
return clone;
}
tree *
decl_cloned_function_p (const_tree decl, bool just_testing)
{
tree *ptr;
if (just_testing)
decl = STRIP_TEMPLATE (decl);
if (TREE_CODE (decl) != FUNCTION_DECL
|| !DECL_LANG_SPECIFIC (decl)
|| DECL_LANG_SPECIFIC (decl)->u.fn.thunk_p)
{
#if defined ENABLE_TREE_CHECKING && (GCC_VERSION >= 2007)
if (!just_testing)
lang_check_failed (__FILE__, __LINE__, __FUNCTION__);
else
#endif
return NULL;
}
ptr = &DECL_LANG_SPECIFIC (decl)->u.fn.u5.cloned_function;
if (just_testing && *ptr == NULL_TREE)
return NULL;
else
return ptr;
}
void
clone_function_decl (tree fn, bool update_methods)
{
tree clone;
if (DECL_CHAIN (fn)
&& DECL_CLONED_FUNCTION_P (DECL_CHAIN (fn)))
return;
if (DECL_MAYBE_IN_CHARGE_CONSTRUCTOR_P (fn))
{
clone = build_clone (fn, complete_ctor_identifier);
if (update_methods)
add_method (DECL_CONTEXT (clone), clone, false);
clone = build_clone (fn, base_ctor_identifier);
if (update_methods)
add_method (DECL_CONTEXT (clone), clone, false);
}
else
{
gcc_assert (DECL_MAYBE_IN_CHARGE_DESTRUCTOR_P (fn));
if (DECL_VIRTUAL_P (fn))
{
clone = build_clone (fn, deleting_dtor_identifier);
if (update_methods)
add_method (DECL_CONTEXT (clone), clone, false);
}
clone = build_clone (fn, complete_dtor_identifier);
if (update_methods)
add_method (DECL_CONTEXT (clone), clone, false);
clone = build_clone (fn, base_dtor_identifier);
if (update_methods)
add_method (DECL_CONTEXT (clone), clone, false);
}
DECL_ABSTRACT_P (fn) = true;
}
void
adjust_clone_args (tree decl)
{
tree clone;
for (clone = DECL_CHAIN (decl); clone && DECL_CLONED_FUNCTION_P (clone);
clone = DECL_CHAIN (clone))
{
tree orig_clone_parms = TYPE_ARG_TYPES (TREE_TYPE (clone));
tree orig_decl_parms = TYPE_ARG_TYPES (TREE_TYPE (decl));
tree decl_parms, clone_parms;
clone_parms = orig_clone_parms;
orig_clone_parms = TREE_CHAIN (orig_clone_parms);
orig_decl_parms = TREE_CHAIN (orig_decl_parms);
if (DECL_HAS_IN_CHARGE_PARM_P (decl))
orig_decl_parms = TREE_CHAIN (orig_decl_parms);
if (DECL_HAS_VTT_PARM_P (decl))
orig_decl_parms = TREE_CHAIN (orig_decl_parms);
clone_parms = orig_clone_parms;
if (DECL_HAS_VTT_PARM_P (clone))
clone_parms = TREE_CHAIN (clone_parms);
for (decl_parms = orig_decl_parms; decl_parms;
decl_parms = TREE_CHAIN (decl_parms),
clone_parms = TREE_CHAIN (clone_parms))
{
if (clone_parms == void_list_node)
{
gcc_assert (decl_parms == clone_parms
|| ctor_omit_inherited_parms (clone));
break;
}
gcc_assert (same_type_p (TREE_TYPE (decl_parms),
TREE_TYPE (clone_parms)));
if (TREE_PURPOSE (decl_parms) && !TREE_PURPOSE (clone_parms))
{
tree exceptions = TYPE_RAISES_EXCEPTIONS (TREE_TYPE (clone));
tree attrs = TYPE_ATTRIBUTES (TREE_TYPE (clone));
tree basetype = TYPE_METHOD_BASETYPE (TREE_TYPE (clone));
tree type;
clone_parms = orig_decl_parms;
if (DECL_HAS_VTT_PARM_P (clone))
{
clone_parms = tree_cons (TREE_PURPOSE (orig_clone_parms),
TREE_VALUE (orig_clone_parms),
clone_parms);
TREE_TYPE (clone_parms) = TREE_TYPE (orig_clone_parms);
}
type = build_method_type_directly (basetype,
TREE_TYPE (TREE_TYPE (clone)),
clone_parms);
if (exceptions)
type = build_exception_variant (type, exceptions);
if (attrs)
type = cp_build_type_attribute_variant (type, attrs);
TREE_TYPE (clone) = type;
clone_parms = NULL_TREE;
break;
}
}
gcc_assert (!clone_parms || clone_parms == void_list_node);
}
}
static void
clone_constructors_and_destructors (tree t)
{
for (ovl_iterator iter (CLASSTYPE_CONSTRUCTORS (t)); iter; ++iter)
clone_function_decl (*iter, true);
if (tree dtor = CLASSTYPE_DESTRUCTOR (t))
clone_function_decl (dtor, true);
}
void
deduce_noexcept_on_destructor (tree dtor)
{
if (!TYPE_RAISES_EXCEPTIONS (TREE_TYPE (dtor)))
TREE_TYPE (dtor) = build_exception_variant (TREE_TYPE (dtor),
noexcept_deferred_spec);
}
static int
look_for_tm_attr_overrides (tree type, tree fndecl)
{
tree binfo = TYPE_BINFO (type);
tree base_binfo;
int ix, found = 0;
for (ix = 0; BINFO_BASE_ITERATE (binfo, ix, base_binfo); ++ix)
{
tree o, basetype = BINFO_TYPE (base_binfo);
if (!TYPE_POLYMORPHIC_P (basetype))
continue;
o = look_for_overrides_here (basetype, fndecl);
if (o)
{
if (lookup_attribute ("transaction_safe_dynamic",
DECL_ATTRIBUTES (o)))
;
else
found |= tm_attr_to_mask (find_tm_attribute
(TYPE_ATTRIBUTES (TREE_TYPE (o))));
}
else
found |= look_for_tm_attr_overrides (basetype, fndecl);
}
return found;
}
static void
set_one_vmethod_tm_attributes (tree type, tree fndecl)
{
tree tm_attr;
int found, have;
found = look_for_tm_attr_overrides (type, fndecl);
if (found == 0)
return;
tm_attr = find_tm_attribute (TYPE_ATTRIBUTES (TREE_TYPE (fndecl)));
have = tm_attr_to_mask (tm_attr);
if (have == TM_ATTR_PURE)
{
if (found != TM_ATTR_PURE)
{
found &= -found;
goto err_override;
}
}
else if (found == TM_ATTR_PURE && tm_attr)
goto err_override;
else if (found != TM_ATTR_PURE && (found & TM_ATTR_PURE))
{
found &= ~TM_ATTR_PURE;
found &= -found;
error_at (DECL_SOURCE_LOCATION (fndecl),
"method overrides both %<transaction_pure%> and %qE methods",
tm_mask_to_attr (found));
}
else if (tm_attr == NULL)
{
apply_tm_attr (fndecl, tm_mask_to_attr (least_bit_hwi (found)));
}
else
{
found &= -found;
if (found <= TM_ATTR_CALLABLE && have > found)
goto err_override;
}
return;
err_override:
error_at (DECL_SOURCE_LOCATION (fndecl),
"method declared %qE overriding %qE method",
tm_attr, tm_mask_to_attr (found));
}
static void
set_method_tm_attributes (tree t)
{
tree class_tm_attr, fndecl;
if (!flag_tm)
return;
if (TYPE_CONTAINS_VPTR_P (t))
{
tree vchain;
for (vchain = BINFO_VIRTUALS (TYPE_BINFO (t)); vchain;
vchain = TREE_CHAIN (vchain))
{
fndecl = BV_FN (vchain);
if (DECL_THUNK_P (fndecl))
fndecl = THUNK_TARGET (fndecl);
set_one_vmethod_tm_attributes (t, fndecl);
}
}
class_tm_attr = find_tm_attribute (TYPE_ATTRIBUTES (t));
if (class_tm_attr == NULL)
return;
for (fndecl = TYPE_FIELDS (t); fndecl; fndecl = DECL_CHAIN (fndecl))
if (DECL_DECLARES_FUNCTION_P (fndecl)
&& !find_tm_attribute (TYPE_ATTRIBUTES (TREE_TYPE (fndecl))))
apply_tm_attr (fndecl, class_tm_attr);
}
bool
default_ctor_p (tree fn)
{
return (DECL_CONSTRUCTOR_P (fn)
&& sufficient_parms_p (FUNCTION_FIRST_USER_PARMTYPE (fn)));
}
bool
type_has_user_nondefault_constructor (tree t)
{
if (!TYPE_HAS_USER_CONSTRUCTOR (t))
return false;
for (ovl_iterator iter (CLASSTYPE_CONSTRUCTORS (t)); iter; ++iter)
{
tree fn = *iter;
if (!DECL_ARTIFICIAL (fn)
&& (TREE_CODE (fn) == TEMPLATE_DECL
|| (skip_artificial_parms_for (fn, DECL_ARGUMENTS (fn))
!= NULL_TREE)))
return true;
}
return false;
}
tree
in_class_defaulted_default_constructor (tree t)
{
if (!TYPE_HAS_USER_CONSTRUCTOR (t))
return NULL_TREE;
for (ovl_iterator iter (CLASSTYPE_CONSTRUCTORS (t)); iter; ++iter)
{
tree fn = *iter;
if (DECL_DEFAULTED_IN_CLASS_P (fn)
&& default_ctor_p (fn))
return fn;
}
return NULL_TREE;
}
bool
user_provided_p (tree fn)
{
if (TREE_CODE (fn) == TEMPLATE_DECL)
return true;
else
return (!DECL_ARTIFICIAL (fn)
&& !(DECL_INITIALIZED_IN_CLASS_P (fn)
&& (DECL_DEFAULTED_FN (fn) || DECL_DELETED_FN (fn))));
}
bool
type_has_user_provided_constructor (tree t)
{
if (!CLASS_TYPE_P (t))
return false;
if (!TYPE_HAS_USER_CONSTRUCTOR (t))
return false;
for (ovl_iterator iter (CLASSTYPE_CONSTRUCTORS (t)); iter; ++iter)
if (user_provided_p (*iter))
return true;
return false;
}
bool
type_has_user_provided_or_explicit_constructor (tree t)
{
if (!CLASS_TYPE_P (t))
return false;
if (!TYPE_HAS_USER_CONSTRUCTOR (t))
return false;
for (ovl_iterator iter (CLASSTYPE_CONSTRUCTORS (t)); iter; ++iter)
{
tree fn = *iter;
if (user_provided_p (fn) || DECL_NONCONVERTING_P (fn))
return true;
}
return false;
}
bool
type_has_non_user_provided_default_constructor (tree t)
{
if (!TYPE_HAS_DEFAULT_CONSTRUCTOR (t))
return false;
if (CLASSTYPE_LAZY_DEFAULT_CTOR (t))
return true;
for (ovl_iterator iter (CLASSTYPE_CONSTRUCTORS (t)); iter; ++iter)
{
tree fn = *iter;
if (TREE_CODE (fn) == FUNCTION_DECL
&& default_ctor_p (fn)
&& !user_provided_p (fn))
return true;
}
return false;
}
bool
vbase_has_user_provided_move_assign (tree type)
{
if (!CLASSTYPE_LAZY_MOVE_ASSIGN (type))
for (ovl_iterator iter (get_class_binding_direct
(type, assign_op_identifier));
iter; ++iter)
if (!DECL_ARTIFICIAL (*iter) && move_fn_p (*iter))
return true;
tree binfo = TYPE_BINFO (type);
tree base_binfo;
for (int i = 0; BINFO_BASE_ITERATE (binfo, i, base_binfo); ++i)
if (vbase_has_user_provided_move_assign (BINFO_TYPE (base_binfo)))
return true;
for (tree field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
{
if (TREE_CODE (field) == FIELD_DECL
&& CLASS_TYPE_P (TREE_TYPE (field))
&& vbase_has_user_provided_move_assign (TREE_TYPE (field)))
return true;
}
return false;
}
tree
default_init_uninitialized_part (tree type)
{
tree t, r, binfo;
int i;
type = strip_array_types (type);
if (!CLASS_TYPE_P (type))
return type;
if (!type_has_non_user_provided_default_constructor (type))
return NULL_TREE;
for (binfo = TYPE_BINFO (type), i = 0;
BINFO_BASE_ITERATE (binfo, i, t); ++i)
{
r = default_init_uninitialized_part (BINFO_TYPE (t));
if (r)
return r;
}
for (t = TYPE_FIELDS (type); t; t = DECL_CHAIN (t))
if (TREE_CODE (t) == FIELD_DECL
&& !DECL_ARTIFICIAL (t)
&& !DECL_INITIAL (t))
{
r = default_init_uninitialized_part (TREE_TYPE (t));
if (r)
return DECL_P (r) ? r : t;
}
return NULL_TREE;
}
bool
trivial_default_constructor_is_constexpr (tree t)
{
gcc_assert (!TYPE_HAS_COMPLEX_DFLT (t));
return is_really_empty_class (t);
}
bool
type_has_constexpr_default_constructor (tree t)
{
tree fns;
if (!CLASS_TYPE_P (t))
{
gcc_assert (TREE_CODE (t) != ARRAY_TYPE);
return false;
}
if (CLASSTYPE_LAZY_DEFAULT_CTOR (t))
{
if (!TYPE_HAS_COMPLEX_DFLT (t))
return trivial_default_constructor_is_constexpr (t);
lazily_declare_fn (sfk_constructor, t);
}
fns = locate_ctor (t);
return (fns && DECL_DECLARED_CONSTEXPR_P (fns));
}
bool
type_maybe_constexpr_default_constructor (tree t)
{
if (CLASS_TYPE_P (t) && CLASSTYPE_LAZY_DEFAULT_CTOR (t)
&& TYPE_HAS_COMPLEX_DFLT (t))
return true;
return type_has_constexpr_default_constructor (t);
}
bool
type_has_virtual_destructor (tree type)
{
tree dtor;
if (!CLASS_TYPE_P (type))
return false;
gcc_assert (COMPLETE_TYPE_P (type));
dtor = CLASSTYPE_DESTRUCTOR (type);
return (dtor && DECL_VIRTUAL_P (dtor));
}
bool
classtype_has_move_assign_or_move_ctor_p (tree t, bool user_p)
{
gcc_assert (user_p
|| (!CLASSTYPE_LAZY_MOVE_CTOR (t)
&& !CLASSTYPE_LAZY_MOVE_ASSIGN (t)));
if (!CLASSTYPE_LAZY_MOVE_CTOR (t))
for (ovl_iterator iter (CLASSTYPE_CONSTRUCTORS (t)); iter; ++iter)
if ((!user_p || !DECL_ARTIFICIAL (*iter)) && move_fn_p (*iter))
return true;
if (!CLASSTYPE_LAZY_MOVE_ASSIGN (t))
for (ovl_iterator iter (get_class_binding_direct
(t, assign_op_identifier));
iter; ++iter)
if ((!user_p || !DECL_ARTIFICIAL (*iter)) && move_fn_p (*iter))
return true;
return false;
}
bool
classtype_has_non_deleted_move_ctor (tree t)
{
if (CLASSTYPE_LAZY_MOVE_CTOR (t))
lazily_declare_fn (sfk_move_constructor, t);
for (ovl_iterator iter (CLASSTYPE_CONSTRUCTORS (t)); iter; ++iter)
if (move_fn_p (*iter) && !DECL_DELETED_FN (*iter))
return true;
return false;
}
bool
type_build_ctor_call (tree t)
{
tree inner;
if (TYPE_NEEDS_CONSTRUCTING (t))
return true;
inner = strip_array_types (t);
if (!CLASS_TYPE_P (inner) || ANON_AGGR_TYPE_P (inner))
return false;
if (!TYPE_HAS_DEFAULT_CONSTRUCTOR (inner))
return true;
if (cxx_dialect < cxx11)
return false;
for (ovl_iterator iter (get_class_binding (inner, complete_ctor_identifier));
iter; ++iter)
{
tree fn = *iter;
if (!DECL_ARTIFICIAL (fn)
|| DECL_DELETED_FN (fn))
return true;
}
return false;
}
bool
type_build_dtor_call (tree t)
{
tree inner;
if (TYPE_HAS_NONTRIVIAL_DESTRUCTOR (t))
return true;
inner = strip_array_types (t);
if (!CLASS_TYPE_P (inner) || ANON_AGGR_TYPE_P (inner)
|| !COMPLETE_TYPE_P (inner))
return false;
if (cxx_dialect < cxx11)
return false;
for (ovl_iterator iter (get_class_binding (inner, complete_dtor_identifier));
iter; ++iter)
{
tree fn = *iter;
if (!DECL_ARTIFICIAL (fn)
|| DECL_DELETED_FN (fn))
return true;
}
return false;
}
static void
remove_zero_width_bit_fields (tree t)
{
tree *fieldsp;
fieldsp = &TYPE_FIELDS (t);
while (*fieldsp)
{
if (TREE_CODE (*fieldsp) == FIELD_DECL
&& DECL_C_BIT_FIELD (*fieldsp)
&& (DECL_SIZE (*fieldsp) == NULL_TREE
|| integer_zerop (DECL_SIZE (*fieldsp))))
*fieldsp = DECL_CHAIN (*fieldsp);
else
fieldsp = &DECL_CHAIN (*fieldsp);
}
}
static bool
type_requires_array_cookie (tree type)
{
tree fns;
bool has_two_argument_delete_p = false;
gcc_assert (CLASS_TYPE_P (type));
if (TYPE_HAS_NONTRIVIAL_DESTRUCTOR (type))
return true;
fns = lookup_fnfields (TYPE_BINFO (type),
ovl_op_identifier (false, VEC_DELETE_EXPR),
0);
if (!fns || fns == error_mark_node)
return false;
for (lkp_iterator iter (BASELINK_FUNCTIONS (fns)); iter; ++iter)
{
tree fn = *iter;
tree second_parm = TREE_CHAIN (TYPE_ARG_TYPES (TREE_TYPE (fn)));
if (second_parm == void_list_node)
return false;
if (!second_parm)
continue;
if (TREE_CHAIN (second_parm) == void_list_node
&& same_type_p (TREE_VALUE (second_parm), size_type_node))
has_two_argument_delete_p = true;
}
return has_two_argument_delete_p;
}
static void
finalize_literal_type_property (tree t)
{
tree fn;
if (cxx_dialect < cxx11
|| TYPE_HAS_NONTRIVIAL_DESTRUCTOR (t))
CLASSTYPE_LITERAL_P (t) = false;
else if (CLASSTYPE_LITERAL_P (t) && LAMBDA_TYPE_P (t))
CLASSTYPE_LITERAL_P (t) = (cxx_dialect >= cxx17);
else if (CLASSTYPE_LITERAL_P (t) && !TYPE_HAS_TRIVIAL_DFLT (t)
&& CLASSTYPE_NON_AGGREGATE (t)
&& !TYPE_HAS_CONSTEXPR_CTOR (t))
CLASSTYPE_LITERAL_P (t) = false;
if (cxx_dialect < cxx14
&& !CLASSTYPE_LITERAL_P (t) && !LAMBDA_TYPE_P (t))
for (fn = TYPE_FIELDS (t); fn; fn = DECL_CHAIN (fn))
if (TREE_CODE (fn) == FUNCTION_DECL
&& DECL_DECLARED_CONSTEXPR_P (fn)
&& DECL_NONSTATIC_MEMBER_FUNCTION_P (fn)
&& !DECL_CONSTRUCTOR_P (fn))
{
DECL_DECLARED_CONSTEXPR_P (fn) = false;
if (!DECL_GENERATED_P (fn)
&& pedwarn (DECL_SOURCE_LOCATION (fn), OPT_Wpedantic,
"enclosing class of %<constexpr%> non-static member "
"function %q+#D is not a literal type", fn))
explain_non_literal_class (t);
}
}
void
explain_non_literal_class (tree t)
{
static hash_set<tree> *diagnosed;
if (!CLASS_TYPE_P (t))
return;
t = TYPE_MAIN_VARIANT (t);
if (diagnosed == NULL)
diagnosed = new hash_set<tree>;
if (diagnosed->add (t))
return;
inform (UNKNOWN_LOCATION, "%q+T is not literal because:", t);
if (cxx_dialect < cxx17 && LAMBDA_TYPE_P (t))
inform (UNKNOWN_LOCATION,
"  %qT is a closure type, which is only literal in "
"C++17 and later", t);
else if (TYPE_HAS_NONTRIVIAL_DESTRUCTOR (t))
inform (UNKNOWN_LOCATION, "  %q+T has a non-trivial destructor", t);
else if (CLASSTYPE_NON_AGGREGATE (t)
&& !TYPE_HAS_TRIVIAL_DFLT (t)
&& !LAMBDA_TYPE_P (t)
&& !TYPE_HAS_CONSTEXPR_CTOR (t))
{
inform (UNKNOWN_LOCATION,
"  %q+T is not an aggregate, does not have a trivial "
"default constructor, and has no %<constexpr%> constructor that "
"is not a copy or move constructor", t);
if (type_has_non_user_provided_default_constructor (t))
for (ovl_iterator iter (CLASSTYPE_CONSTRUCTORS (t)); iter; ++iter)
{
tree fn = *iter;
tree parms = TYPE_ARG_TYPES (TREE_TYPE (fn));
parms = skip_artificial_parms_for (fn, parms);
if (sufficient_parms_p (parms))
{
if (DECL_DELETED_FN (fn))
maybe_explain_implicit_delete (fn);
else
explain_invalid_constexpr_fn (fn);
break;
}
}
}
else
{
tree binfo, base_binfo, field; int i;
for (binfo = TYPE_BINFO (t), i = 0;
BINFO_BASE_ITERATE (binfo, i, base_binfo); i++)
{
tree basetype = TREE_TYPE (base_binfo);
if (!CLASSTYPE_LITERAL_P (basetype))
{
inform (UNKNOWN_LOCATION,
"  base class %qT of %q+T is non-literal",
basetype, t);
explain_non_literal_class (basetype);
return;
}
}
for (field = TYPE_FIELDS (t); field; field = TREE_CHAIN (field))
{
tree ftype;
if (TREE_CODE (field) != FIELD_DECL)
continue;
ftype = TREE_TYPE (field);
if (!literal_type_p (ftype))
{
inform (DECL_SOURCE_LOCATION (field),
"  non-static data member %qD has non-literal type",
field);
if (CLASS_TYPE_P (ftype))
explain_non_literal_class (ftype);
}
if (CP_TYPE_VOLATILE_P (ftype))
inform (DECL_SOURCE_LOCATION (field),
"  non-static data member %qD has volatile type", field);
}
}
}
static void
check_bases_and_members (tree t)
{
int cant_have_const_ctor;
int no_const_asn_ref;
tree access_decls;
bool saved_complex_asn_ref;
bool saved_nontrivial_dtor;
tree fn;
cant_have_const_ctor = 0;
no_const_asn_ref = 0;
check_bases (t, &cant_have_const_ctor, &no_const_asn_ref);
if (cxx_dialect >= cxx11)
if (tree dtor = CLASSTYPE_DESTRUCTOR (t))
deduce_noexcept_on_destructor (dtor);
check_methods (t);
saved_complex_asn_ref = TYPE_HAS_COMPLEX_COPY_ASSIGN (t);
saved_nontrivial_dtor = TYPE_HAS_NONTRIVIAL_DESTRUCTOR (t);
check_field_decls (t, &access_decls,
&cant_have_const_ctor,
&no_const_asn_ref);
if (!TYPE_CONTAINS_VPTR_P (t))
CLASSTYPE_NEARLY_EMPTY_P (t) = 0;
TYPE_HAS_COMPLEX_COPY_CTOR (t) |= TYPE_CONTAINS_VPTR_P (t);
TYPE_HAS_COMPLEX_MOVE_CTOR (t) |= TYPE_CONTAINS_VPTR_P (t);
TYPE_NEEDS_CONSTRUCTING (t)
|= (type_has_user_provided_constructor (t) || TYPE_CONTAINS_VPTR_P (t));
CLASSTYPE_NON_AGGREGATE (t)
|= (type_has_user_provided_or_explicit_constructor (t)
|| TYPE_POLYMORPHIC_P (t));
CLASSTYPE_NON_LAYOUT_POD_P (t)
|= (CLASSTYPE_NON_AGGREGATE (t)
|| saved_nontrivial_dtor || saved_complex_asn_ref);
CLASSTYPE_NON_STD_LAYOUT (t) |= TYPE_CONTAINS_VPTR_P (t);
TYPE_HAS_COMPLEX_COPY_ASSIGN (t) |= TYPE_CONTAINS_VPTR_P (t);
TYPE_HAS_COMPLEX_MOVE_ASSIGN (t) |= TYPE_CONTAINS_VPTR_P (t);
TYPE_HAS_COMPLEX_DFLT (t) |= TYPE_CONTAINS_VPTR_P (t);
if (!TYPE_HAS_COMPLEX_DFLT (t)
&& TYPE_HAS_DEFAULT_CONSTRUCTOR (t)
&& !type_has_non_user_provided_default_constructor (t))
TYPE_HAS_COMPLEX_DFLT (t) = true;
if (TYPE_POLYMORPHIC_P (t) && warn_nonvdtor)
{
tree binfo = TYPE_BINFO (t);
vec<tree, va_gc> *accesses = BINFO_BASE_ACCESSES (binfo);
tree base_binfo;
unsigned i;
for (i = 0; BINFO_BASE_ITERATE (binfo, i, base_binfo); i++)
{
tree basetype = TREE_TYPE (base_binfo);
if ((*accesses)[i] == access_public_node
&& (TYPE_POLYMORPHIC_P (basetype) || warn_ecpp)
&& accessible_nvdtor_p (basetype))
warning (OPT_Wnon_virtual_dtor,
"base class %q#T has accessible non-virtual destructor",
basetype);
}
}
if (warn_uninitialized
&& !TYPE_HAS_USER_CONSTRUCTOR (t)
&& CLASSTYPE_NON_AGGREGATE (t))
{
tree field;
for (field = TYPE_FIELDS (t); field; field = DECL_CHAIN (field))
{
tree type;
if (TREE_CODE (field) != FIELD_DECL
|| DECL_INITIAL (field) != NULL_TREE)
continue;
type = TREE_TYPE (field);
if (TREE_CODE (type) == REFERENCE_TYPE)
warning_at (DECL_SOURCE_LOCATION (field),
OPT_Wuninitialized, "non-static reference %q#D "
"in class without a constructor", field);
else if (CP_TYPE_CONST_P (type)
&& (!CLASS_TYPE_P (type)
|| !TYPE_HAS_DEFAULT_CONSTRUCTOR (type)))
warning_at (DECL_SOURCE_LOCATION (field),
OPT_Wuninitialized, "non-static const member %q#D "
"in class without a constructor", field);
}
}
add_implicitly_declared_members (t, &access_decls,
cant_have_const_ctor,
no_const_asn_ref);
for (fn = TYPE_FIELDS (t); fn; fn = DECL_CHAIN (fn))
if (DECL_DECLARES_FUNCTION_P (fn)
&& !DECL_ARTIFICIAL (fn)
&& DECL_DEFAULTED_IN_CLASS_P (fn))
{
int copy = copy_fn_p (fn);
if (copy > 0)
{
bool imp_const_p
= (DECL_CONSTRUCTOR_P (fn) ? !cant_have_const_ctor
: !no_const_asn_ref);
bool fn_const_p = (copy == 2);
if (fn_const_p && !imp_const_p)
error ("%q+D declared to take const reference, but implicit "
"declaration would take non-const", fn);
}
defaulted_late_check (fn);
}
if (LAMBDA_TYPE_P (t))
{
CLASSTYPE_NON_AGGREGATE (t) = 1;
}
finalize_literal_type_property (t);
clone_constructors_and_destructors (t);
for (; access_decls; access_decls = TREE_CHAIN (access_decls))
handle_using_decl (TREE_VALUE (access_decls), t);
LANG_TYPE_CLASS_CHECK (t)->vec_new_uses_cookie
= type_requires_array_cookie (t);
}
static tree
create_vtable_ptr (tree t, tree* virtuals_p)
{
tree fn;
for (fn = TYPE_FIELDS (t); fn; fn = DECL_CHAIN (fn))
if (TREE_CODE (fn) == FUNCTION_DECL
&& DECL_VINDEX (fn) && !DECL_MAYBE_IN_CHARGE_DESTRUCTOR_P (fn)
&& TREE_CODE (DECL_VINDEX (fn)) != INTEGER_CST)
{
tree new_virtual = make_node (TREE_LIST);
BV_FN (new_virtual) = fn;
BV_DELTA (new_virtual) = integer_zero_node;
BV_VCALL_INDEX (new_virtual) = NULL_TREE;
TREE_CHAIN (new_virtual) = *virtuals_p;
*virtuals_p = new_virtual;
}
if (!TYPE_VFIELD (t) && (*virtuals_p || TYPE_CONTAINS_VPTR_P (t)))
{
tree field;
field = build_decl (input_location, 
FIELD_DECL, get_vfield_name (t), vtbl_ptr_type_node);
DECL_VIRTUAL_P (field) = 1;
DECL_ARTIFICIAL (field) = 1;
DECL_FIELD_CONTEXT (field) = t;
DECL_FCONTEXT (field) = t;
if (TYPE_PACKED (t))
DECL_PACKED (field) = 1;
TYPE_VFIELD (t) = field;
CLASSTYPE_EMPTY_P (t) = 0;
return field;
}
return NULL_TREE;
}
static void
propagate_binfo_offsets (tree binfo, tree offset)
{
int i;
tree primary_binfo;
tree base_binfo;
BINFO_OFFSET (binfo)
= fold_convert (sizetype,
size_binop (PLUS_EXPR,
fold_convert (ssizetype, BINFO_OFFSET (binfo)),
offset));
primary_binfo = get_primary_binfo (binfo);
if (primary_binfo && BINFO_INHERITANCE_CHAIN (primary_binfo) == binfo)
propagate_binfo_offsets (primary_binfo, offset);
for (i = 0; BINFO_BASE_ITERATE (binfo, i, base_binfo); ++i)
{
if (base_binfo == primary_binfo)
continue;
if (BINFO_VIRTUAL_P (base_binfo))
continue;
propagate_binfo_offsets (base_binfo, offset);
}
}
static void
layout_virtual_bases (record_layout_info rli, splay_tree offsets)
{
tree vbase;
tree t = rli->t;
tree *next_field;
if (BINFO_N_BASE_BINFOS (TYPE_BINFO (t)) == 0)
return;
next_field = &TYPE_FIELDS (t);
while (*next_field)
next_field = &DECL_CHAIN (*next_field);
for (vbase = TYPE_BINFO (t); vbase; vbase = TREE_CHAIN (vbase))
{
if (!BINFO_VIRTUAL_P (vbase))
continue;
if (!BINFO_PRIMARY_P (vbase))
{
next_field = build_base_field (rli, vbase,
offsets, next_field);
}
}
}
static tree
end_of_base (tree binfo)
{
tree size;
if (!CLASSTYPE_AS_BASE (BINFO_TYPE (binfo)))
size = TYPE_SIZE_UNIT (char_type_node);
else if (is_empty_class (BINFO_TYPE (binfo)))
size = TYPE_SIZE_UNIT (BINFO_TYPE (binfo));
else
size = CLASSTYPE_SIZE_UNIT (BINFO_TYPE (binfo));
return size_binop (PLUS_EXPR, BINFO_OFFSET (binfo), size);
}
static tree
end_of_class (tree t, int include_virtuals_p)
{
tree result = size_zero_node;
vec<tree, va_gc> *vbases;
tree binfo;
tree base_binfo;
tree offset;
int i;
for (binfo = TYPE_BINFO (t), i = 0;
BINFO_BASE_ITERATE (binfo, i, base_binfo); ++i)
{
if (!include_virtuals_p
&& BINFO_VIRTUAL_P (base_binfo)
&& (!BINFO_PRIMARY_P (base_binfo)
|| BINFO_INHERITANCE_CHAIN (base_binfo) != TYPE_BINFO (t)))
continue;
offset = end_of_base (base_binfo);
if (tree_int_cst_lt (result, offset))
result = offset;
}
if (include_virtuals_p)
for (vbases = CLASSTYPE_VBASECLASSES (t), i = 0;
vec_safe_iterate (vbases, i, &base_binfo); i++)
{
offset = end_of_base (base_binfo);
if (tree_int_cst_lt (result, offset))
result = offset;
}
return result;
}
static void
warn_about_ambiguous_bases (tree t)
{
int i;
vec<tree, va_gc> *vbases;
tree basetype;
tree binfo;
tree base_binfo;
if (!CLASSTYPE_REPEATED_BASE_P (t))
return;
for (binfo = TYPE_BINFO (t), i = 0;
BINFO_BASE_ITERATE (binfo, i, base_binfo); ++i)
{
basetype = BINFO_TYPE (base_binfo);
if (!uniquely_derived_from_p (basetype, t))
warning (0, "direct base %qT inaccessible in %qT due to ambiguity",
basetype, t);
}
if (extra_warnings)
for (vbases = CLASSTYPE_VBASECLASSES (t), i = 0;
vec_safe_iterate (vbases, i, &binfo); i++)
{
basetype = BINFO_TYPE (binfo);
if (!uniquely_derived_from_p (basetype, t))
warning (OPT_Wextra, "virtual base %qT inaccessible in %qT due "
"to ambiguity", basetype, t);
}
}
static int
splay_tree_compare_integer_csts (splay_tree_key k1, splay_tree_key k2)
{
return tree_int_cst_compare ((tree) k1, (tree) k2);
}
static void
include_empty_classes (record_layout_info rli)
{
tree eoc;
tree rli_size;
eoc = end_of_class (rli->t, CLASSTYPE_AS_BASE (rli->t) != NULL_TREE);
rli_size = rli_size_unit_so_far (rli);
if (TREE_CODE (rli_size) == INTEGER_CST
&& tree_int_cst_lt (rli_size, eoc))
{
gcc_assert (tree_int_cst_equal
(rli->bitpos, round_down (rli->bitpos, BITS_PER_UNIT)));
rli->bitpos
= size_binop (PLUS_EXPR,
rli->bitpos,
size_binop (MULT_EXPR,
fold_convert (bitsizetype,
size_binop (MINUS_EXPR,
eoc, rli_size)),
bitsize_int (BITS_PER_UNIT)));
normalize_rli (rli);
}
}
static void
layout_class_type (tree t, tree *virtuals_p)
{
tree non_static_data_members;
tree field;
tree vptr;
record_layout_info rli;
splay_tree empty_base_offsets;
bool last_field_was_bitfield = false;
tree *next_field;
non_static_data_members = TYPE_FIELDS (t);
rli = start_record_layout (t);
determine_primary_bases (t);
vptr = create_vtable_ptr (t, virtuals_p);
if (vptr)
{
DECL_CHAIN (vptr) = TYPE_FIELDS (t);
TYPE_FIELDS (t) = vptr;
next_field = &DECL_CHAIN (vptr);
place_field (rli, vptr);
}
else
next_field = &TYPE_FIELDS (t);
empty_base_offsets = splay_tree_new (splay_tree_compare_integer_csts,
NULL, NULL);
build_base_fields (rli, empty_base_offsets, next_field);
for (field = non_static_data_members; field; field = DECL_CHAIN (field))
{
tree type;
tree padding;
if (TREE_CODE (field) != FIELD_DECL)
{
place_field (rli, field);
if (VAR_P (field))
{
maybe_register_incomplete_var (field);
determine_visibility (field);
}
continue;
}
type = TREE_TYPE (field);
if (type == error_mark_node)
continue;
padding = NULL_TREE;
if (DECL_C_BIT_FIELD (field)
&& tree_int_cst_lt (TYPE_SIZE (type), DECL_SIZE (field)))
{
bool was_unnamed_p = false;
tree limit = size_int (MAX_FIXED_MODE_SIZE);
if (tree_int_cst_lt (DECL_SIZE (field), limit))
limit = DECL_SIZE (field);
tree integer_type = integer_types[itk_char];
for (unsigned itk = itk_char; itk != itk_none; itk++)
if (tree next = integer_types[itk])
{
if (tree_int_cst_lt (limit, TYPE_SIZE (next)))
break;
integer_type = next;
}
if (TREE_CODE (t) == UNION_TYPE)
padding = DECL_SIZE (field);
else
padding = size_binop (MINUS_EXPR, DECL_SIZE (field),
TYPE_SIZE (integer_type));
if (integer_zerop (padding))
padding = NULL_TREE;
if (PCC_BITFIELD_TYPE_MATTERS && !DECL_NAME (field))
{
was_unnamed_p = true;
DECL_NAME (field) = make_anon_name ();
}
DECL_SIZE (field) = TYPE_SIZE (integer_type);
SET_DECL_ALIGN (field, TYPE_ALIGN (integer_type));
DECL_USER_ALIGN (field) = TYPE_USER_ALIGN (integer_type);
layout_nonempty_base_or_field (rli, field, NULL_TREE,
empty_base_offsets);
if (was_unnamed_p)
DECL_NAME (field) = NULL_TREE;
DECL_SIZE (field) = TYPE_SIZE (type);
SET_DECL_MODE (field, TYPE_MODE (type));
}
else
layout_nonempty_base_or_field (rli, field, NULL_TREE,
empty_base_offsets);
record_subobject_offsets (TREE_TYPE (field),
byte_position(field),
empty_base_offsets,
true);
if (warn_abi
&& DECL_C_BIT_FIELD (field)
&& !TREE_NO_WARNING (field)
&& !last_field_was_bitfield
&& !integer_zerop (size_binop (TRUNC_MOD_EXPR,
DECL_FIELD_BIT_OFFSET (field),
bitsize_unit_node)))
warning_at (DECL_SOURCE_LOCATION (field), OPT_Wabi,
"offset of %qD is not ABI-compliant and may "
"change in a future version of GCC", field);
if (DECL_C_BIT_FIELD (field))
{
unsigned HOST_WIDE_INT width;
tree ftype = TREE_TYPE (field);
width = tree_to_uhwi (DECL_SIZE (field));
if (width != TYPE_PRECISION (ftype))
{
TREE_TYPE (field)
= c_build_bitfield_integer_type (width,
TYPE_UNSIGNED (ftype));
TREE_TYPE (field)
= cp_build_qualified_type (TREE_TYPE (field),
cp_type_quals (ftype));
}
}
if (padding)
{
tree padding_field;
padding_field = build_decl (input_location,
FIELD_DECL,
NULL_TREE,
char_type_node);
DECL_BIT_FIELD (padding_field) = 1;
DECL_SIZE (padding_field) = padding;
DECL_CONTEXT (padding_field) = t;
DECL_ARTIFICIAL (padding_field) = 1;
DECL_IGNORED_P (padding_field) = 1;
DECL_PADDING_P (padding_field) = 1;
layout_nonempty_base_or_field (rli, padding_field,
NULL_TREE,
empty_base_offsets);
}
last_field_was_bitfield = DECL_C_BIT_FIELD (field);
}
if (!integer_zerop (rli->bitpos))
{
rli->bitpos = round_up_loc (input_location, rli->bitpos, BITS_PER_UNIT);
normalize_rli (rli);
}
remove_zero_width_bit_fields (t);
if (CLASSTYPE_NON_LAYOUT_POD_P (t) || CLASSTYPE_EMPTY_P (t))
{
tree base_t = make_node (TREE_CODE (t));
tree eoc = end_of_class (t, 0);
TYPE_SIZE_UNIT (base_t)
= size_binop (MAX_EXPR,
fold_convert (sizetype,
size_binop (CEIL_DIV_EXPR,
rli_size_so_far (rli),
bitsize_int (BITS_PER_UNIT))),
eoc);
TYPE_SIZE (base_t)
= size_binop (MAX_EXPR,
rli_size_so_far (rli),
size_binop (MULT_EXPR,
fold_convert (bitsizetype, eoc),
bitsize_int (BITS_PER_UNIT)));
SET_TYPE_ALIGN (base_t, rli->record_align);
TYPE_USER_ALIGN (base_t) = TYPE_USER_ALIGN (t);
TYPE_TYPELESS_STORAGE (base_t) = TYPE_TYPELESS_STORAGE (t);
next_field = &TYPE_FIELDS (base_t);
for (field = TYPE_FIELDS (t); field; field = DECL_CHAIN (field))
if (TREE_CODE (field) == FIELD_DECL)
{
*next_field = copy_node (field);
DECL_CONTEXT (*next_field) = base_t;
next_field = &DECL_CHAIN (*next_field);
}
*next_field = NULL_TREE;
compute_record_mode (base_t);
TYPE_CONTEXT (base_t) = t;
CLASSTYPE_AS_BASE (t) = base_t;
}
else
CLASSTYPE_AS_BASE (t) = t;
if (CLASSTYPE_EMPTY_P (t))
CLASSTYPE_CONTAINS_EMPTY_CLASS_P (t) = 1;
layout_decl (TYPE_MAIN_DECL (t), 0);
layout_virtual_bases (rli, empty_base_offsets);
include_empty_classes (rli);
if (integer_zerop (rli_size_unit_so_far (rli)) && CLASSTYPE_EMPTY_P (t))
place_field (rli,
build_decl (input_location,
FIELD_DECL, NULL_TREE, char_type_node));
if (TYPE_PACKED (t) && !layout_pod_type_p (t))
rli->packed_maybe_necessary = true;
finish_record_layout (rli, true);
if (TYPE_SIZE_UNIT (t)
&& TREE_CODE (TYPE_SIZE_UNIT (t)) == INTEGER_CST
&& !TREE_OVERFLOW (TYPE_SIZE_UNIT (t))
&& !valid_constant_size_p (TYPE_SIZE_UNIT (t)))
error ("size of type %qT is too large (%qE bytes)", t, TYPE_SIZE_UNIT (t));
warn_about_ambiguous_bases (t);
for (field = TYPE_FIELDS (t); field; field = DECL_CHAIN (field))
if (DECL_ARTIFICIAL (field) && IS_FAKE_BASE_TYPE (TREE_TYPE (field)))
TREE_TYPE (field) = TYPE_CONTEXT (TREE_TYPE (field));
splay_tree_delete (empty_base_offsets);
if (CLASSTYPE_EMPTY_P (t)
&& tree_int_cst_lt (sizeof_biggest_empty_class,
TYPE_SIZE_UNIT (t)))
sizeof_biggest_empty_class = TYPE_SIZE_UNIT (t);
}
void
determine_key_method (tree type)
{
tree method;
if (processing_template_decl
|| CLASSTYPE_TEMPLATE_INSTANTIATION (type)
|| CLASSTYPE_INTERFACE_KNOWN (type))
return;
for (method = TYPE_FIELDS (type); method; method = DECL_CHAIN (method))
if (TREE_CODE (method) == FUNCTION_DECL
&& DECL_VINDEX (method) != NULL_TREE
&& ! DECL_DECLARED_INLINE_P (method)
&& ! DECL_PURE_VIRTUAL_P (method))
{
CLASSTYPE_KEY_METHOD (type) = method;
break;
}
return;
}
static inline bool
field_nonempty_p (const_tree fld)
{
if (TREE_CODE (fld) == ERROR_MARK)
return false;
tree type = TREE_TYPE (fld);
if (TREE_CODE (fld) == FIELD_DECL
&& TREE_CODE (type) != ERROR_MARK
&& (DECL_NAME (fld) || RECORD_OR_UNION_TYPE_P (type)))
{
return TYPE_SIZE (type)
&& (TREE_CODE (TYPE_SIZE (type)) != INTEGER_CST
|| !tree_int_cst_equal (size_zero_node, TYPE_SIZE (type)));
}
return false;
}
struct flexmems_t
{
tree array;
tree first;
tree after[2];
tree enclosing;
};
static void
find_flexarrays (tree t, flexmems_t *fmem, bool base_p,
tree pun ,
tree pstr )
{
if (!pun && TREE_CODE (t) == UNION_TYPE)
pun = t;
for (tree fld = TYPE_FIELDS (t); fld; fld = DECL_CHAIN (fld))
{
if (fld == error_mark_node)
return;
if (TREE_CODE (fld) == TYPE_DECL
&& DECL_IMPLICIT_TYPEDEF_P (fld)
&& CLASS_TYPE_P (TREE_TYPE (fld))
&& anon_aggrname_p (DECL_NAME (fld)))
{
check_flexarrays (TREE_TYPE (fld));
continue;
}
if (DECL_ARTIFICIAL (fld) || TREE_CODE (fld) != FIELD_DECL)
continue;
tree fldtype = TREE_TYPE (fld);
if (fldtype == error_mark_node)
return;
tree eltype = fldtype;
while (TREE_CODE (eltype) == ARRAY_TYPE
|| TREE_CODE (eltype) == POINTER_TYPE
|| TREE_CODE (eltype) == REFERENCE_TYPE)
eltype = TREE_TYPE (eltype);
if (RECORD_OR_UNION_TYPE_P (eltype))
{
if (fmem->array && !fmem->after[bool (pun)])
{
fmem->after[bool (pun)] = fld;
break;
}
if (eltype == fldtype || TYPE_UNNAMED_P (eltype))
{
tree first = fmem->first;
tree array = fmem->array;
if (first && !array && !ANON_AGGR_TYPE_P (eltype))
fmem->first = NULL_TREE;
find_flexarrays (eltype, fmem, false, pun,
!pstr && TREE_CODE (t) == RECORD_TYPE ? fld : pstr);
if (fmem->array != array)
continue;
if (first && !array && !ANON_AGGR_TYPE_P (eltype))
{
fmem->first = first;
}
if (base_p)
continue;
}
}
if (field_nonempty_p (fld))
{
if (!fmem->first)
fmem->first = fld;
if (fmem->array && !fmem->after[bool (pun)])
fmem->after[bool (pun)] = fld;
}
if (TREE_CODE (fldtype) != ARRAY_TYPE)
continue;
if (TYPE_DOMAIN (fldtype))
{
if (fmem->array)
{
if (!fmem->after[bool (pun)])
fmem->after[bool (pun)] = fld;
}
else if (integer_all_onesp (TYPE_MAX_VALUE (TYPE_DOMAIN (fldtype))))
{
fmem->array = fld;
fmem->enclosing = pstr;
}
}
else
{
if (fmem->array)
{
if (TYPE_DOMAIN (TREE_TYPE (fmem->array)))
{
fmem->after[bool (pun)] = NULL_TREE;
fmem->array = fld;
fmem->enclosing = pstr;
}
else if (!fmem->after[bool (pun)])
fmem->after[bool (pun)] = fld;
}
else
{
fmem->array = fld;
fmem->enclosing = pstr;
}
}
}
}
static void
diagnose_invalid_flexarray (const flexmems_t *fmem)
{
if (fmem->array && fmem->enclosing
&& pedwarn (location_of (fmem->enclosing), OPT_Wpedantic,
TYPE_DOMAIN (TREE_TYPE (fmem->array))
? G_("invalid use of %q#T with a zero-size array "
"in %q#D")
: G_("invalid use of %q#T with a flexible array member "
"in %q#T"),
DECL_CONTEXT (fmem->array),
DECL_CONTEXT (fmem->enclosing)))
inform (DECL_SOURCE_LOCATION (fmem->array),
"array member %q#D declared here", fmem->array);
}
static void
diagnose_flexarrays (tree t, const flexmems_t *fmem)
{
if (!fmem->array)
return;
if (fmem->first && !fmem->after[0])
{
diagnose_invalid_flexarray (fmem);
return;
}
bool diagd = false;
const char *msg = 0;
if (TYPE_DOMAIN (TREE_TYPE (fmem->array)))
{
if (fmem->after[0])
msg = G_("zero-size array member %qD not at end of %q#T");
else if (!fmem->first)
msg = G_("zero-size array member %qD in an otherwise empty %q#T");
if (msg)
{
location_t loc = DECL_SOURCE_LOCATION (fmem->array);
if (pedwarn (loc, OPT_Wpedantic, msg, fmem->array, t))
{
inform (location_of (t), "in the definition of %q#T", t);
diagd = true;
}
}
}
else
{
if (fmem->after[0])
msg = G_("flexible array member %qD not at end of %q#T");
else if (!fmem->first)
msg = G_("flexible array member %qD in an otherwise empty %q#T");
if (msg)
{
location_t loc = DECL_SOURCE_LOCATION (fmem->array);
diagd = true;
error_at (loc, msg, fmem->array, t);
if (fmem->after[0]
&& ((DECL_CONTEXT (fmem->after[0])
!= DECL_CONTEXT (fmem->array))))
{
inform (DECL_SOURCE_LOCATION (fmem->after[0]),
"next member %q#D declared here",
fmem->after[0]);
inform (location_of (t), "in the definition of %q#T", t);
}
}
}
if (!diagd && fmem->array && fmem->enclosing)
diagnose_invalid_flexarray (fmem);
}
static void
check_flexarrays (tree t, flexmems_t *fmem ,
bool base_p )
{
flexmems_t flexmems = flexmems_t ();
if (!fmem)
fmem = &flexmems;
else if (fmem->array && fmem->first && fmem->after[0])
return;
tree fam = fmem->array;
if (CLASSTYPE_HAS_PRIMARY_BASE_P (t))
{
tree basetype = BINFO_TYPE (CLASSTYPE_PRIMARY_BINFO (t));
check_flexarrays (basetype, fmem, true);
}
int nbases = TYPE_BINFO (t) ? BINFO_N_BASE_BINFOS (TYPE_BINFO (t)) : 0;
for (int i = 0; i < nbases; ++i)
{
tree base_binfo = BINFO_BASE_BINFO (TYPE_BINFO (t), i);
if (base_binfo == CLASSTYPE_PRIMARY_BINFO (t))
continue;
if (BINFO_VIRTUAL_P (base_binfo))
continue;
check_flexarrays (BINFO_TYPE (base_binfo), fmem, true);
}
if (fmem == &flexmems)
{
int i;
tree base_binfo;
vec<tree, va_gc> *vbases;
for (vbases = CLASSTYPE_VBASECLASSES (t), i = 0;
vec_safe_iterate (vbases, i, &base_binfo); i++)
{
tree basetype = TREE_TYPE (base_binfo);
check_flexarrays (basetype, fmem, true);
}
}
bool maybe_anon_p = TYPE_UNNAMED_P (t);
if (fmem != &flexmems || !maybe_anon_p)
find_flexarrays (t, fmem, base_p || fam != fmem->array);
if (fmem == &flexmems && !maybe_anon_p)
{
diagnose_flexarrays (t, fmem);
}
}
void
finish_struct_1 (tree t)
{
tree x;
tree virtuals = NULL_TREE;
if (COMPLETE_TYPE_P (t))
{
gcc_assert (MAYBE_CLASS_TYPE_P (t));
error ("redefinition of %q#T", t);
popclass ();
return;
}
TYPE_SIZE (t) = NULL_TREE;
CLASSTYPE_PRIMARY_BINFO (t) = NULL_TREE;
CLASSTYPE_EMPTY_P (t) = 1;
CLASSTYPE_NEARLY_EMPTY_P (t) = 1;
CLASSTYPE_CONTAINS_EMPTY_CLASS_P (t) = 0;
CLASSTYPE_LITERAL_P (t) = true;
check_bases_and_members (t);
if (TYPE_CONTAINS_VPTR_P (t))
{
if (targetm.cxx.key_method_may_be_inline ())
determine_key_method (t);
if (!CLASSTYPE_KEY_METHOD (t))
vec_safe_push (keyed_classes, t);
}
layout_class_type (t, &virtuals);
set_class_bindings (t);
check_flexarrays (t);
virtuals = modify_all_vtables (t, nreverse (virtuals));
if (virtuals || TYPE_CONTAINS_VPTR_P (t))
{
if (!CLASSTYPE_HAS_PRIMARY_BASE_P (t))
build_primary_vtable (NULL_TREE, t);
else if (! BINFO_NEW_VTABLE_MARKED (TYPE_BINFO (t)))
build_primary_vtable (CLASSTYPE_PRIMARY_BINFO (t), t);
if (warn_abi_tag)
for (tree v = virtuals; v; v = TREE_CHAIN (v))
check_abi_tags (t, TREE_VALUE (v));
}
if (TYPE_CONTAINS_VPTR_P (t))
{
int vindex;
tree fn;
if (BINFO_VTABLE (TYPE_BINFO (t)))
gcc_assert (DECL_VIRTUAL_P (BINFO_VTABLE (TYPE_BINFO (t))));
if (!CLASSTYPE_HAS_PRIMARY_BASE_P (t))
gcc_assert (BINFO_VIRTUALS (TYPE_BINFO (t)) == NULL_TREE);
BINFO_VIRTUALS (TYPE_BINFO (t))
= chainon (BINFO_VIRTUALS (TYPE_BINFO (t)), virtuals);
for (vindex = 0, fn = BINFO_VIRTUALS (TYPE_BINFO (t));
fn;
fn = TREE_CHAIN (fn),
vindex += (TARGET_VTABLE_USES_DESCRIPTORS
? TARGET_VTABLE_USES_DESCRIPTORS : 1))
{
tree fndecl = BV_FN (fn);
if (DECL_THUNK_P (fndecl))
DECL_VINDEX (fndecl) = NULL_TREE;
else if (TREE_CODE (DECL_VINDEX (fndecl)) != INTEGER_CST)
DECL_VINDEX (fndecl) = build_int_cst (NULL_TREE, vindex);
}
}
finish_struct_bits (t);
set_method_tm_attributes (t);
if (flag_openmp || flag_openmp_simd)
finish_omp_declare_simd_methods (t);
for (x = TYPE_FIELDS (t); x; x = DECL_CHAIN (x))
if (DECL_DECLARES_FUNCTION_P (x))
DECL_IN_AGGR_P (x) = false;
else if (VAR_P (x) && TREE_STATIC (x)
&& TREE_TYPE (x) != error_mark_node
&& same_type_p (TYPE_MAIN_VARIANT (TREE_TYPE (x)), t))
SET_DECL_MODE (x, TYPE_MODE (t));
constrain_class_visibility (t);
finish_vtbls (t);
build_vtt (t);
if (warn_nonvdtor
&& TYPE_POLYMORPHIC_P (t) && accessible_nvdtor_p (t)
&& !CLASSTYPE_FINAL (t))
warning (OPT_Wnon_virtual_dtor,
"%q#T has virtual functions and accessible"
" non-virtual destructor", t);
complete_vars (t);
if (warn_overloaded_virtual)
warn_hidden (t);
targetm.cxx.adjust_class_at_definition (t);
maybe_suppress_debug_info (t);
if (flag_vtable_verify)
vtv_save_class_info (t);
dump_class_hierarchy (t);
rest_of_type_compilation (t, ! LOCAL_CLASS_P (t));
if (TYPE_TRANSPARENT_AGGR (t))
{
tree field = first_field (t);
if (field == NULL_TREE || error_operand_p (field))
{
error ("type transparent %q#T does not have any fields", t);
TYPE_TRANSPARENT_AGGR (t) = 0;
}
else if (DECL_ARTIFICIAL (field))
{
if (DECL_FIELD_IS_BASE (field))
error ("type transparent class %qT has base classes", t);
else
{
gcc_checking_assert (DECL_VIRTUAL_P (field));
error ("type transparent class %qT has virtual functions", t);
}
TYPE_TRANSPARENT_AGGR (t) = 0;
}
else if (TYPE_MODE (t) != DECL_MODE (field))
{
error ("type transparent %q#T cannot be made transparent because "
"the type of the first field has a different ABI from the "
"class overall", t);
TYPE_TRANSPARENT_AGGR (t) = 0;
}
}
}
void
unreverse_member_declarations (tree t)
{
tree next;
tree prev;
tree x;
CLASSTYPE_DECL_LIST (t) = nreverse (CLASSTYPE_DECL_LIST (t));
prev = NULL_TREE;
for (x = TYPE_FIELDS (t);
x && TREE_CODE (x) != TYPE_DECL;
x = next)
{
next = DECL_CHAIN (x);
DECL_CHAIN (x) = prev;
prev = x;
}
if (prev)
{
DECL_CHAIN (TYPE_FIELDS (t)) = x;
TYPE_FIELDS (t) = prev;
}
}
tree
finish_struct (tree t, tree attributes)
{
location_t saved_loc = input_location;
unreverse_member_declarations (t);
cplus_decl_attributes (&t, attributes, (int) ATTR_FLAG_TYPE_IN_PLACE);
fixup_attribute_variants (t);
input_location = DECL_SOURCE_LOCATION (TYPE_NAME (t));
if (processing_template_decl)
{
tree x;
for (x = TYPE_FIELDS (t); x; x = DECL_CHAIN (x))
if (TREE_CODE (x) == USING_DECL)
{
tree fn = strip_using_decl (x);
if (OVL_P (fn))
for (lkp_iterator iter (fn); iter; ++iter)
add_method (t, *iter, true);
}
else if (DECL_DECLARES_FUNCTION_P (x))
DECL_IN_AGGR_P (x) = false;
tree ass_op = build_lang_decl (USING_DECL, assign_op_identifier,
NULL_TREE);
DECL_CONTEXT (ass_op) = t;
USING_DECL_SCOPE (ass_op) = t;
DECL_DEPENDENT_P (ass_op) = true;
DECL_ARTIFICIAL (ass_op) = true;
DECL_CHAIN (ass_op) = TYPE_FIELDS (t);
TYPE_FIELDS (t) = ass_op;
TYPE_SIZE (t) = bitsize_zero_node;
TYPE_SIZE_UNIT (t) = size_zero_node;
set_class_bindings (t);
CLASSTYPE_PURE_VIRTUALS (t) = NULL;
for (x = TYPE_FIELDS (t); x; x = DECL_CHAIN (x))
if (TREE_CODE (x) == FUNCTION_DECL && DECL_PURE_VIRTUAL_P (x))
vec_safe_push (CLASSTYPE_PURE_VIRTUALS (t), x);
complete_vars (t);
TYPE_PRECISION (t) = maximum_field_alignment;
for (x = TYPE_NEXT_VARIANT (t); x; x = TYPE_NEXT_VARIANT (x))
{
TYPE_SIZE (x) = TYPE_SIZE (t);
TYPE_SIZE_UNIT (x) = TYPE_SIZE_UNIT (t);
TYPE_FIELDS (x) = TYPE_FIELDS (t);
}
}
else
finish_struct_1 (t);
maybe_warn_about_overly_private_class (t);
if (is_std_init_list (t))
{
bool ok = false;
if (processing_template_decl)
{
tree f = next_initializable_field (TYPE_FIELDS (t));
if (f && TREE_CODE (TREE_TYPE (f)) == POINTER_TYPE)
{
f = next_initializable_field (DECL_CHAIN (f));
if (f && same_type_p (TREE_TYPE (f), size_type_node))
ok = true;
}
}
if (!ok)
fatal_error (input_location, "definition of %qD does not match "
"%<#include <initializer_list>%>", TYPE_NAME (t));
}
input_location = saved_loc;
TYPE_BEING_DEFINED (t) = 0;
if (current_class_type)
popclass ();
else
error ("trying to finish struct, but kicked out due to previous parse errors");
if (processing_template_decl && at_function_scope_p ()
&& !LAMBDA_TYPE_P (t))
add_stmt (build_min (TAG_DEFN, t));
return t;
}

static hash_table<nofree_ptr_hash<tree_node> > *fixed_type_or_null_ref_ht;
static tree
fixed_type_or_null (tree instance, int *nonnull, int *cdtorp)
{
#define RECUR(T) fixed_type_or_null((T), nonnull, cdtorp)
switch (TREE_CODE (instance))
{
case INDIRECT_REF:
if (POINTER_TYPE_P (TREE_TYPE (instance)))
return NULL_TREE;
else
return RECUR (TREE_OPERAND (instance, 0));
case CALL_EXPR:
if (CALL_EXPR_FN (instance)
&& TREE_HAS_CONSTRUCTOR (instance))
{
if (nonnull)
*nonnull = 1;
return TREE_TYPE (instance);
}
return NULL_TREE;
case SAVE_EXPR:
if (TREE_HAS_CONSTRUCTOR (instance))
{
if (nonnull)
*nonnull = 1;
return TREE_TYPE (instance);
}
return RECUR (TREE_OPERAND (instance, 0));
case POINTER_PLUS_EXPR:
case PLUS_EXPR:
case MINUS_EXPR:
if (TREE_CODE (TREE_OPERAND (instance, 0)) == ADDR_EXPR)
return RECUR (TREE_OPERAND (instance, 0));
if (TREE_CODE (TREE_OPERAND (instance, 1)) == INTEGER_CST)
return RECUR (TREE_OPERAND (instance, 0));
return NULL_TREE;
CASE_CONVERT:
return RECUR (TREE_OPERAND (instance, 0));
case ADDR_EXPR:
instance = TREE_OPERAND (instance, 0);
if (nonnull)
{
tree t = get_base_address (instance);
if (t && DECL_P (t))
*nonnull = 1;
}
return RECUR (instance);
case COMPONENT_REF:
if (DECL_FIELD_IS_BASE (TREE_OPERAND (instance, 1)))
return RECUR (TREE_OPERAND (instance, 0));
return RECUR (TREE_OPERAND (instance, 1));
case VAR_DECL:
case FIELD_DECL:
if (TREE_CODE (TREE_TYPE (instance)) == ARRAY_TYPE
&& MAYBE_CLASS_TYPE_P (TREE_TYPE (TREE_TYPE (instance))))
{
if (nonnull)
*nonnull = 1;
return TREE_TYPE (TREE_TYPE (instance));
}
case TARGET_EXPR:
case PARM_DECL:
case RESULT_DECL:
if (MAYBE_CLASS_TYPE_P (TREE_TYPE (instance)))
{
if (nonnull)
*nonnull = 1;
return TREE_TYPE (instance);
}
else if (instance == current_class_ptr)
{
if (nonnull)
*nonnull = 1;
if (current_scope () != current_function_decl
|| (DECL_LANG_SPECIFIC (current_function_decl)
&& (DECL_CONSTRUCTOR_P (current_function_decl)
|| DECL_DESTRUCTOR_P (current_function_decl))))
{
if (cdtorp)
*cdtorp = 1;
return TREE_TYPE (TREE_TYPE (instance));
}
}
else if (TREE_CODE (TREE_TYPE (instance)) == REFERENCE_TYPE)
{
if (!fixed_type_or_null_ref_ht)
fixed_type_or_null_ref_ht
= new hash_table<nofree_ptr_hash<tree_node> > (37);
if (nonnull)
*nonnull = 1;
if (VAR_P (instance)
&& DECL_INITIAL (instance)
&& !type_dependent_expression_p_push (DECL_INITIAL (instance))
&& !fixed_type_or_null_ref_ht->find (instance))
{
tree type;
tree_node **slot;
slot = fixed_type_or_null_ref_ht->find_slot (instance, INSERT);
*slot = instance;
type = RECUR (DECL_INITIAL (instance));
fixed_type_or_null_ref_ht->remove_elt (instance);
return type;
}
}
return NULL_TREE;
default:
return NULL_TREE;
}
#undef RECUR
}
int
resolves_to_fixed_type_p (tree instance, int* nonnull)
{
tree t = TREE_TYPE (instance);
int cdtorp = 0;
tree fixed;
if (in_template_function ())
{
if (nonnull)
*nonnull = true;
return true;
}
fixed = fixed_type_or_null (instance, nonnull, &cdtorp);
if (fixed == NULL_TREE)
return 0;
if (POINTER_TYPE_P (t))
t = TREE_TYPE (t);
if (!same_type_ignoring_top_level_qualifiers_p (t, fixed))
return 0;
return cdtorp ? -1 : 1;
}

void
init_class_processing (void)
{
current_class_depth = 0;
current_class_stack_size = 10;
current_class_stack
= XNEWVEC (struct class_stack_node, current_class_stack_size);
vec_alloc (local_classes, 8);
sizeof_biggest_empty_class = size_zero_node;
ridpointers[(int) RID_PUBLIC] = access_public_node;
ridpointers[(int) RID_PRIVATE] = access_private_node;
ridpointers[(int) RID_PROTECTED] = access_protected_node;
}
static void
restore_class_cache (void)
{
tree type;
push_binding_level (previous_class_level);
class_binding_level = previous_class_level;
for (type = class_binding_level->type_shadowed;
type;
type = TREE_CHAIN (type))
SET_IDENTIFIER_TYPE_VALUE (TREE_PURPOSE (type), TREE_TYPE (type));
}
void
pushclass (tree type)
{
class_stack_node_t csn;
type = TYPE_MAIN_VARIANT (type);
if (current_class_depth + 1 >= current_class_stack_size)
{
current_class_stack_size *= 2;
current_class_stack
= XRESIZEVEC (struct class_stack_node, current_class_stack,
current_class_stack_size);
}
csn = current_class_stack + current_class_depth;
csn->name = current_class_name;
csn->type = current_class_type;
csn->access = current_access_specifier;
csn->names_used = 0;
csn->hidden = 0;
current_class_depth++;
current_class_name = TYPE_NAME (type);
if (TREE_CODE (current_class_name) == TYPE_DECL)
current_class_name = DECL_NAME (current_class_name);
current_class_type = type;
current_access_specifier = (CLASSTYPE_DECLARED_CLASS (type)
? access_private_node
: access_public_node);
if (previous_class_level
&& type != previous_class_level->this_entity
&& current_class_depth == 1)
{
invalidate_class_lookup_cache ();
}
if (!previous_class_level
|| type != previous_class_level->this_entity
|| current_class_depth > 1)
pushlevel_class ();
else
restore_class_cache ();
}
void
invalidate_class_lookup_cache (void)
{
previous_class_level = NULL;
}
void
popclass (void)
{
poplevel_class ();
current_class_depth--;
current_class_name = current_class_stack[current_class_depth].name;
current_class_type = current_class_stack[current_class_depth].type;
current_access_specifier = current_class_stack[current_class_depth].access;
if (current_class_stack[current_class_depth].names_used)
splay_tree_delete (current_class_stack[current_class_depth].names_used);
}
void
push_class_stack (void)
{
if (current_class_depth)
++current_class_stack[current_class_depth - 1].hidden;
}
void
pop_class_stack (void)
{
if (current_class_depth)
--current_class_stack[current_class_depth - 1].hidden;
}
tree
currently_open_class (tree t)
{
int i;
if (!CLASS_TYPE_P (t))
return NULL_TREE;
t = TYPE_MAIN_VARIANT (t);
for (i = current_class_depth; i > 0; --i)
{
tree c;
if (i == current_class_depth)
c = current_class_type;
else
{
if (current_class_stack[i].hidden)
break;
c = current_class_stack[i].type;
}
if (!c)
continue;
if (same_type_p (c, t))
return c;
}
return NULL_TREE;
}
tree
currently_open_derived_class (tree t)
{
int i;
if (dependent_type_p (t))
return NULL_TREE;
if (!current_class_type)
return NULL_TREE;
if (DERIVED_FROM_P (t, current_class_type))
return current_class_type;
for (i = current_class_depth - 1; i > 0; --i)
{
if (current_class_stack[i].hidden)
break;
if (DERIVED_FROM_P (t, current_class_stack[i].type))
return current_class_stack[i].type;
}
return NULL_TREE;
}
tree
outermost_open_class (void)
{
if (!current_class_type)
return NULL_TREE;
tree r = NULL_TREE;
if (TYPE_BEING_DEFINED (current_class_type))
r = current_class_type;
for (int i = current_class_depth - 1; i > 0; --i)
{
if (current_class_stack[i].hidden)
break;
tree t = current_class_stack[i].type;
if (!TYPE_BEING_DEFINED (t))
break;
r = t;
}
return r;
}
tree
current_nonlambda_class_type (void)
{
tree type = current_class_type;
while (type && LAMBDA_TYPE_P (type))
type = decl_type_context (TYPE_NAME (type));
return type;
}
void
push_nested_class (tree type)
{
if (type == NULL_TREE
|| !CLASS_TYPE_P (type))
return;
push_nested_class (DECL_CONTEXT (TYPE_MAIN_DECL (type)));
pushclass (type);
}
void
pop_nested_class (void)
{
tree context = DECL_CONTEXT (TYPE_MAIN_DECL (current_class_type));
popclass ();
if (context && CLASS_TYPE_P (context))
pop_nested_class ();
}
int
current_lang_depth (void)
{
return vec_safe_length (current_lang_base);
}
void
push_lang_context (tree name)
{
vec_safe_push (current_lang_base, current_lang_name);
if (name == lang_name_cplusplus)
current_lang_name = name;
else if (name == lang_name_c)
current_lang_name = name;
else
error ("language string %<\"%E\"%> not recognized", name);
}
void
pop_lang_context (void)
{
current_lang_name = current_lang_base->pop ();
}

static tree
resolve_address_of_overloaded_function (tree target_type,
tree overload,
tsubst_flags_t complain,
bool template_only,
tree explicit_targs,
tree access_path)
{
int is_ptrmem = 0;
tree matches = NULL_TREE;
tree fn;
tree target_fn_type;
gcc_assert (!TYPE_PTR_P (target_type)
|| TREE_CODE (TREE_TYPE (target_type)) != METHOD_TYPE);
gcc_assert (is_overloaded_fn (overload));
if (TYPE_PTRFN_P (target_type)
|| TYPE_REFFN_P (target_type))
;
else if (TYPE_PTRMEMFUNC_P (target_type))
is_ptrmem = 1;
else if (TREE_CODE (target_type) == FUNCTION_TYPE)
target_type = build_reference_type (target_type);
else
{
if (complain & tf_error)
error ("cannot resolve overloaded function %qD based on"
" conversion to type %qT",
OVL_NAME (overload), target_type);
return error_mark_node;
}
target_fn_type = static_fn_type (target_type);
if (!template_only)
for (lkp_iterator iter (overload); iter; ++iter)
{
tree fn = *iter;
if (TREE_CODE (fn) == TEMPLATE_DECL)
continue;
if ((TREE_CODE (TREE_TYPE (fn)) == METHOD_TYPE) != is_ptrmem)
continue;
if (flag_noexcept_type
&& !maybe_instantiate_noexcept (fn, complain))
continue;
tree fntype = static_fn_type (fn);
if (same_type_p (target_fn_type, fntype)
|| fnptr_conv_p (target_fn_type, fntype))
matches = tree_cons (fn, NULL_TREE, matches);
}
if (!matches)
{
tree target_arg_types;
tree target_ret_type;
tree *args;
unsigned int nargs, ia;
tree arg;
target_arg_types = TYPE_ARG_TYPES (target_fn_type);
target_ret_type = TREE_TYPE (target_fn_type);
nargs = list_length (target_arg_types);
args = XALLOCAVEC (tree, nargs);
for (arg = target_arg_types, ia = 0;
arg != NULL_TREE && arg != void_list_node;
arg = TREE_CHAIN (arg), ++ia)
args[ia] = TREE_VALUE (arg);
nargs = ia;
for (lkp_iterator iter (overload); iter; ++iter)
{
tree fn = *iter;
tree instantiation;
tree targs;
if (TREE_CODE (fn) != TEMPLATE_DECL)
continue;
if ((TREE_CODE (TREE_TYPE (fn)) == METHOD_TYPE)
!= is_ptrmem)
continue;
tree ret = target_ret_type;
if (undeduced_auto_decl (fn))
ret = NULL_TREE;
targs = make_tree_vec (DECL_NTPARMS (fn));
instantiation = fn_type_unification (fn, explicit_targs, targs, args,
nargs, ret,
DEDUCE_EXACT, LOOKUP_NORMAL,
false, false);
if (instantiation == error_mark_node)
continue;
if (flag_concepts && !constraints_satisfied_p (instantiation))
continue;
if (undeduced_auto_decl (instantiation))
{
++function_depth;
instantiate_decl (instantiation, false, false);
--function_depth;
require_deduced_type (instantiation);
}
if (flag_noexcept_type)
maybe_instantiate_noexcept (instantiation, complain);
tree fntype = static_fn_type (instantiation);
if (same_type_p (target_fn_type, fntype)
|| fnptr_conv_p (target_fn_type, fntype))
matches = tree_cons (instantiation, fn, matches);
}
if (matches)
{
tree match = most_specialized_instantiation (matches);
if (match != error_mark_node)
matches = tree_cons (TREE_PURPOSE (match),
NULL_TREE,
NULL_TREE);
}
}
if (matches == NULL_TREE)
{
if (complain & tf_error)
{
error ("no matches converting function %qD to type %q#T",
OVL_NAME (overload), target_type);
print_candidates (overload);
}
return error_mark_node;
}
else if (TREE_CHAIN (matches))
{
tree match = NULL_TREE;
fn = TREE_PURPOSE (matches);
for (match = TREE_CHAIN (matches); match; match = TREE_CHAIN (match))
if (!decls_match (fn, TREE_PURPOSE (match))
&& !targetm.target_option.function_versions
(fn, TREE_PURPOSE (match)))
break;
if (match)
{
if (complain & tf_error)
{
error ("converting overloaded function %qD to type %q#T is ambiguous",
OVL_NAME (overload), target_type);
for (match = matches; match; match = TREE_CHAIN (match))
TREE_VALUE (match) = TREE_PURPOSE (match);
print_candidates (matches);
}
return error_mark_node;
}
}
fn = TREE_PURPOSE (matches);
if (DECL_NONSTATIC_MEMBER_FUNCTION_P (fn)
&& !(complain & tf_ptrmem_ok) && !flag_ms_extensions)
{
static int explained;
if (!(complain & tf_error))
return error_mark_node;
permerror (input_location, "assuming pointer to member %qD", fn);
if (!explained)
{
inform (input_location, "(a pointer to member can only be formed with %<&%E%>)", fn);
explained = 1;
}
}
if (DECL_FUNCTION_VERSIONED (fn))
{
fn = get_function_version_dispatcher (fn);
if (fn == NULL)
return error_mark_node;
if (!(complain & tf_conv))
mark_versions_used (fn);
}
if (!(complain & tf_conv))
{
if (DECL_DELETED_FN (fn) && !(complain & tf_error))
return error_mark_node;
if (!mark_used (fn, complain) && !(complain & tf_error))
return error_mark_node;
}
if (DECL_FUNCTION_MEMBER_P (fn))
{
gcc_assert (access_path);
perform_or_defer_access_check (access_path, fn, fn, complain);
}
if (TYPE_PTRFN_P (target_type) || TYPE_PTRMEMFUNC_P (target_type))
return cp_build_addr_expr (fn, complain);
else
{
cxx_mark_addressable (fn);
return fn;
}
}
tree
instantiate_type (tree lhstype, tree rhs, tsubst_flags_t complain)
{
tsubst_flags_t complain_in = complain;
tree access_path = NULL_TREE;
complain &= ~tf_ptrmem_ok;
if (lhstype == unknown_type_node)
{
if (complain & tf_error)
error ("not enough type information");
return error_mark_node;
}
if (TREE_TYPE (rhs) != NULL_TREE && ! (type_unknown_p (rhs)))
{
tree fntype = non_reference (lhstype);
if (same_type_p (fntype, TREE_TYPE (rhs)))
return rhs;
if (fnptr_conv_p (fntype, TREE_TYPE (rhs)))
return rhs;
if (flag_ms_extensions
&& TYPE_PTRMEMFUNC_P (fntype)
&& !TYPE_PTRMEMFUNC_P (TREE_TYPE (rhs)))
;
else
{
if (complain & tf_error)
error ("cannot convert %qE from type %qT to type %qT",
rhs, TREE_TYPE (rhs), fntype);
return error_mark_node;
}
}
if (TREE_CODE (rhs) == SAVE_EXPR)
rhs = TREE_OPERAND (rhs, 0);
if (BASELINK_P (rhs))
{
access_path = BASELINK_ACCESS_BINFO (rhs);
rhs = BASELINK_FUNCTIONS (rhs);
}
if (TREE_CODE (rhs) == NON_DEPENDENT_EXPR)
{
if (complain & tf_error)
error ("not enough type information");
return error_mark_node;
}
gcc_assert (TREE_CODE (rhs) == ADDR_EXPR
|| TREE_CODE (rhs) == COMPONENT_REF
|| is_overloaded_fn (rhs)
|| (flag_ms_extensions && TREE_CODE (rhs) == FUNCTION_DECL));
switch (TREE_CODE (rhs))
{
case COMPONENT_REF:
{
tree member = TREE_OPERAND (rhs, 1);
member = instantiate_type (lhstype, member, complain);
if (member != error_mark_node
&& TREE_SIDE_EFFECTS (TREE_OPERAND (rhs, 0)))
return build2 (COMPOUND_EXPR, TREE_TYPE (member),
TREE_OPERAND (rhs, 0), member);
return member;
}
case OFFSET_REF:
rhs = TREE_OPERAND (rhs, 1);
if (BASELINK_P (rhs))
return instantiate_type (lhstype, rhs, complain_in);
gcc_assert (TREE_CODE (rhs) == TEMPLATE_ID_EXPR);
case TEMPLATE_ID_EXPR:
{
tree fns = TREE_OPERAND (rhs, 0);
tree args = TREE_OPERAND (rhs, 1);
return
resolve_address_of_overloaded_function (lhstype, fns, complain_in,
true,
args, access_path);
}
case OVERLOAD:
case FUNCTION_DECL:
return
resolve_address_of_overloaded_function (lhstype, rhs, complain_in,
false,
NULL_TREE,
access_path);
case ADDR_EXPR:
{
if (PTRMEM_OK_P (rhs))
complain |= tf_ptrmem_ok;
return instantiate_type (lhstype, TREE_OPERAND (rhs, 0), complain);
}
case ERROR_MARK:
return error_mark_node;
default:
gcc_unreachable ();
}
return error_mark_node;
}

static tree
get_vfield_name (tree type)
{
tree binfo, base_binfo;
for (binfo = TYPE_BINFO (type);
BINFO_N_BASE_BINFOS (binfo);
binfo = base_binfo)
{
base_binfo = BINFO_BASE_BINFO (binfo, 0);
if (BINFO_VIRTUAL_P (base_binfo)
|| !TYPE_CONTAINS_VPTR_P (BINFO_TYPE (base_binfo)))
break;
}
type = BINFO_TYPE (binfo);
tree ctor_name = constructor_name (type);
char *buf = (char *) alloca (sizeof (VFIELD_NAME_FORMAT)
+ IDENTIFIER_LENGTH (ctor_name) + 2);
sprintf (buf, VFIELD_NAME_FORMAT, IDENTIFIER_POINTER (ctor_name));
return get_identifier (buf);
}
void
build_self_reference (void)
{
tree name = DECL_NAME (TYPE_NAME (current_class_type));
tree value = build_lang_decl (TYPE_DECL, name, current_class_type);
DECL_NONLOCAL (value) = 1;
DECL_CONTEXT (value) = current_class_type;
DECL_ARTIFICIAL (value) = 1;
SET_DECL_SELF_REFERENCE_P (value);
set_underlying_type (value);
if (processing_template_decl)
value = push_template_decl (value);
tree saved_cas = current_access_specifier;
current_access_specifier = access_public_node;
finish_member_declaration (value);
current_access_specifier = saved_cas;
}
int
is_empty_class (tree type)
{
if (type == error_mark_node)
return 0;
if (! CLASS_TYPE_P (type))
return 0;
return CLASSTYPE_EMPTY_P (type);
}
bool
is_really_empty_class (tree type)
{
if (CLASS_TYPE_P (type))
{
tree field;
tree binfo;
tree base_binfo;
int i;
if (COMPLETE_TYPE_P (type) && is_empty_class (type))
return true;
for (binfo = TYPE_BINFO (type), i = 0;
BINFO_BASE_ITERATE (binfo, i, base_binfo); ++i)
if (!is_really_empty_class (BINFO_TYPE (base_binfo)))
return false;
for (field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
if (TREE_CODE (field) == FIELD_DECL
&& !DECL_ARTIFICIAL (field)
&& !DECL_UNNAMED_BIT_FIELD (field)
&& !is_really_empty_class (TREE_TYPE (field)))
return false;
return true;
}
else if (TREE_CODE (type) == ARRAY_TYPE)
return (integer_zerop (array_type_nelts_top (type))
|| is_really_empty_class (TREE_TYPE (type)));
return false;
}
void
maybe_note_name_used_in_class (tree name, tree decl)
{
splay_tree names_used;
if (!(innermost_scope_kind() == sk_class
&& TYPE_BEING_DEFINED (current_class_type)
&& !LAMBDA_TYPE_P (current_class_type)))
return;
if (lookup_member (current_class_type, name,
0, false, tf_warning_or_error))
return;
if (!current_class_stack[current_class_depth - 1].names_used)
current_class_stack[current_class_depth - 1].names_used
= splay_tree_new (splay_tree_compare_pointers, 0, 0);
names_used = current_class_stack[current_class_depth - 1].names_used;
splay_tree_insert (names_used,
(splay_tree_key) name,
(splay_tree_value) decl);
}
void
note_name_declared_in_class (tree name, tree decl)
{
splay_tree names_used;
splay_tree_node n;
names_used
= current_class_stack[current_class_depth - 1].names_used;
if (!names_used)
return;
if ((!pedantic && current_lang_name == lang_name_c)
|| flag_ms_extensions)
return;
n = splay_tree_lookup (names_used, (splay_tree_key) name);
if (n)
{
permerror (input_location, "declaration of %q#D", decl);
permerror (location_of ((tree) n->value),
"changes meaning of %qD from %q#D",
OVL_NAME (decl), (tree) n->value);
}
}
tree
get_vtbl_decl_for_binfo (tree binfo)
{
tree decl;
decl = BINFO_VTABLE (binfo);
if (decl && TREE_CODE (decl) == POINTER_PLUS_EXPR)
{
gcc_assert (TREE_CODE (TREE_OPERAND (decl, 0)) == ADDR_EXPR);
decl = TREE_OPERAND (TREE_OPERAND (decl, 0), 0);
}
if (decl)
gcc_assert (VAR_P (decl));
return decl;
}
static tree
get_primary_binfo (tree binfo)
{
tree primary_base;
primary_base = CLASSTYPE_PRIMARY_BINFO (BINFO_TYPE (binfo));
if (!primary_base)
return NULL_TREE;
return copied_binfo (primary_base, binfo);
}
static tree
most_primary_binfo (tree binfo)
{
tree b = binfo;
while (CLASSTYPE_HAS_PRIMARY_BASE_P (BINFO_TYPE (b))
&& !BINFO_LOST_PRIMARY_P (b))
{
tree primary_base = get_primary_binfo (b);
gcc_assert (BINFO_PRIMARY_P (primary_base)
&& BINFO_INHERITANCE_CHAIN (primary_base) == b);
b = primary_base;
}
return b;
}
bool
vptr_via_virtual_p (tree binfo)
{
if (TYPE_P (binfo))
binfo = TYPE_BINFO (binfo);
tree primary = most_primary_binfo (binfo);
tree virt = binfo_via_virtual (primary, NULL_TREE);
return virt != NULL_TREE;
}
static int
maybe_indent_hierarchy (FILE * stream, int indent, int indented_p)
{
if (!indented_p)
fprintf (stream, "%*s", indent, "");
return 1;
}
static tree
dump_class_hierarchy_r (FILE *stream,
dump_flags_t flags,
tree binfo,
tree igo,
int indent)
{
int indented = 0;
tree base_binfo;
int i;
indented = maybe_indent_hierarchy (stream, indent, 0);
fprintf (stream, "%s (0x" HOST_WIDE_INT_PRINT_HEX ") ",
type_as_string (BINFO_TYPE (binfo), TFF_PLAIN_IDENTIFIER),
(HOST_WIDE_INT) (uintptr_t) binfo);
if (binfo != igo)
{
fprintf (stream, "alternative-path\n");
return igo;
}
igo = TREE_CHAIN (binfo);
fprintf (stream, HOST_WIDE_INT_PRINT_DEC,
tree_to_shwi (BINFO_OFFSET (binfo)));
if (is_empty_class (BINFO_TYPE (binfo)))
fprintf (stream, " empty");
else if (CLASSTYPE_NEARLY_EMPTY_P (BINFO_TYPE (binfo)))
fprintf (stream, " nearly-empty");
if (BINFO_VIRTUAL_P (binfo))
fprintf (stream, " virtual");
fprintf (stream, "\n");
indented = 0;
if (BINFO_PRIMARY_P (binfo))
{
indented = maybe_indent_hierarchy (stream, indent + 3, indented);
fprintf (stream, " primary-for %s (0x" HOST_WIDE_INT_PRINT_HEX ")",
type_as_string (BINFO_TYPE (BINFO_INHERITANCE_CHAIN (binfo)),
TFF_PLAIN_IDENTIFIER),
(HOST_WIDE_INT) (uintptr_t) BINFO_INHERITANCE_CHAIN (binfo));
}
if (BINFO_LOST_PRIMARY_P (binfo))
{
indented = maybe_indent_hierarchy (stream, indent + 3, indented);
fprintf (stream, " lost-primary");
}
if (indented)
fprintf (stream, "\n");
if (!(flags & TDF_SLIM))
{
int indented = 0;
if (BINFO_SUBVTT_INDEX (binfo))
{
indented = maybe_indent_hierarchy (stream, indent + 3, indented);
fprintf (stream, " subvttidx=%s",
expr_as_string (BINFO_SUBVTT_INDEX (binfo),
TFF_PLAIN_IDENTIFIER));
}
if (BINFO_VPTR_INDEX (binfo))
{
indented = maybe_indent_hierarchy (stream, indent + 3, indented);
fprintf (stream, " vptridx=%s",
expr_as_string (BINFO_VPTR_INDEX (binfo),
TFF_PLAIN_IDENTIFIER));
}
if (BINFO_VPTR_FIELD (binfo))
{
indented = maybe_indent_hierarchy (stream, indent + 3, indented);
fprintf (stream, " vbaseoffset=%s",
expr_as_string (BINFO_VPTR_FIELD (binfo),
TFF_PLAIN_IDENTIFIER));
}
if (BINFO_VTABLE (binfo))
{
indented = maybe_indent_hierarchy (stream, indent + 3, indented);
fprintf (stream, " vptr=%s",
expr_as_string (BINFO_VTABLE (binfo),
TFF_PLAIN_IDENTIFIER));
}
if (indented)
fprintf (stream, "\n");
}
for (i = 0; BINFO_BASE_ITERATE (binfo, i, base_binfo); i++)
igo = dump_class_hierarchy_r (stream, flags, base_binfo, igo, indent + 2);
return igo;
}
static void
dump_class_hierarchy_1 (FILE *stream, dump_flags_t flags, tree t)
{
fprintf (stream, "Class %s\n", type_as_string (t, TFF_PLAIN_IDENTIFIER));
fprintf (stream, "   size=%lu align=%lu\n",
(unsigned long)(tree_to_shwi (TYPE_SIZE (t)) / BITS_PER_UNIT),
(unsigned long)(TYPE_ALIGN (t) / BITS_PER_UNIT));
fprintf (stream, "   base size=%lu base align=%lu\n",
(unsigned long)(tree_to_shwi (TYPE_SIZE (CLASSTYPE_AS_BASE (t)))
/ BITS_PER_UNIT),
(unsigned long)(TYPE_ALIGN (CLASSTYPE_AS_BASE (t))
/ BITS_PER_UNIT));
dump_class_hierarchy_r (stream, flags, TYPE_BINFO (t), TYPE_BINFO (t), 0);
fprintf (stream, "\n");
}
void
debug_class (tree t)
{
dump_class_hierarchy_1 (stderr, TDF_SLIM, t);
}
static void
dump_class_hierarchy (tree t)
{
dump_flags_t flags;
if (FILE *stream = dump_begin (class_dump_id, &flags))
{
dump_class_hierarchy_1 (stream, flags, t);
dump_end (class_dump_id, stream);
}
}
static void
dump_array (FILE * stream, tree decl)
{
tree value;
unsigned HOST_WIDE_INT ix;
HOST_WIDE_INT elt;
tree size = TYPE_MAX_VALUE (TYPE_DOMAIN (TREE_TYPE (decl)));
elt = (tree_to_shwi (TYPE_SIZE (TREE_TYPE (TREE_TYPE (decl))))
/ BITS_PER_UNIT);
fprintf (stream, "%s:", decl_as_string (decl, TFF_PLAIN_IDENTIFIER));
fprintf (stream, " %s entries",
expr_as_string (size_binop (PLUS_EXPR, size, size_one_node),
TFF_PLAIN_IDENTIFIER));
fprintf (stream, "\n");
FOR_EACH_CONSTRUCTOR_VALUE (CONSTRUCTOR_ELTS (DECL_INITIAL (decl)),
ix, value)
fprintf (stream, "%-4ld  %s\n", (long)(ix * elt),
expr_as_string (value, TFF_PLAIN_IDENTIFIER));
}
static void
dump_vtable (tree t, tree binfo, tree vtable)
{
dump_flags_t flags;
FILE *stream = dump_begin (class_dump_id, &flags);
if (!stream)
return;
if (!(flags & TDF_SLIM))
{
int ctor_vtbl_p = TYPE_BINFO (t) != binfo;
fprintf (stream, "%s for %s",
ctor_vtbl_p ? "Construction vtable" : "Vtable",
type_as_string (BINFO_TYPE (binfo), TFF_PLAIN_IDENTIFIER));
if (ctor_vtbl_p)
{
if (!BINFO_VIRTUAL_P (binfo))
fprintf (stream, " (0x" HOST_WIDE_INT_PRINT_HEX " instance)",
(HOST_WIDE_INT) (uintptr_t) binfo);
fprintf (stream, " in %s", type_as_string (t, TFF_PLAIN_IDENTIFIER));
}
fprintf (stream, "\n");
dump_array (stream, vtable);
fprintf (stream, "\n");
}
dump_end (class_dump_id, stream);
}
static void
dump_vtt (tree t, tree vtt)
{
dump_flags_t flags;
FILE *stream = dump_begin (class_dump_id, &flags);
if (!stream)
return;
if (!(flags & TDF_SLIM))
{
fprintf (stream, "VTT for %s\n",
type_as_string (t, TFF_PLAIN_IDENTIFIER));
dump_array (stream, vtt);
fprintf (stream, "\n");
}
dump_end (class_dump_id, stream);
}
static void
dump_thunk (FILE *stream, int indent, tree thunk)
{
static const char spaces[] = "        ";
tree name = DECL_NAME (thunk);
tree thunks;
fprintf (stream, "%.*s%p %s %s", indent, spaces,
(void *)thunk,
!DECL_THUNK_P (thunk) ? "function"
: DECL_THIS_THUNK_P (thunk) ? "this-thunk" : "covariant-thunk",
name ? IDENTIFIER_POINTER (name) : "<unset>");
if (DECL_THUNK_P (thunk))
{
HOST_WIDE_INT fixed_adjust = THUNK_FIXED_OFFSET (thunk);
tree virtual_adjust = THUNK_VIRTUAL_OFFSET (thunk);
fprintf (stream, " fixed=" HOST_WIDE_INT_PRINT_DEC, fixed_adjust);
if (!virtual_adjust)
;
else if (DECL_THIS_THUNK_P (thunk))
fprintf (stream, " vcall="  HOST_WIDE_INT_PRINT_DEC,
tree_to_shwi (virtual_adjust));
else
fprintf (stream, " vbase=" HOST_WIDE_INT_PRINT_DEC "(%s)",
tree_to_shwi (BINFO_VPTR_FIELD (virtual_adjust)),
type_as_string (BINFO_TYPE (virtual_adjust), TFF_SCOPE));
if (THUNK_ALIAS (thunk))
fprintf (stream, " alias to %p", (void *)THUNK_ALIAS (thunk));
}
fprintf (stream, "\n");
for (thunks = DECL_THUNKS (thunk); thunks; thunks = TREE_CHAIN (thunks))
dump_thunk (stream, indent + 2, thunks);
}
void
debug_thunks (tree fn)
{
dump_thunk (stderr, 0, fn);
}
static void
finish_vtbls (tree t)
{
tree vbase;
vec<constructor_elt, va_gc> *v = NULL;
tree vtable = BINFO_VTABLE (TYPE_BINFO (t));
accumulate_vtbl_inits (TYPE_BINFO (t), TYPE_BINFO (t), TYPE_BINFO (t),
vtable, t, &v);
for (vbase = TYPE_BINFO (t); vbase; vbase = TREE_CHAIN (vbase))
{
if (!BINFO_VIRTUAL_P (vbase))
continue;
accumulate_vtbl_inits (vbase, vbase, TYPE_BINFO (t), vtable, t, &v);
}
if (BINFO_VTABLE (TYPE_BINFO (t)))
initialize_vtable (TYPE_BINFO (t), v);
}
static void
initialize_vtable (tree binfo, vec<constructor_elt, va_gc> *inits)
{
tree decl;
layout_vtable_decl (binfo, vec_safe_length (inits));
decl = get_vtbl_decl_for_binfo (binfo);
initialize_artificial_var (decl, inits);
dump_vtable (BINFO_TYPE (binfo), binfo, decl);
}
static void
build_vtt (tree t)
{
tree type;
tree vtt;
tree index;
vec<constructor_elt, va_gc> *inits;
inits = NULL;
index = size_zero_node;
build_vtt_inits (TYPE_BINFO (t), t, &inits, &index);
if (!inits)
return;
type = build_array_of_n_type (const_ptr_type_node,
inits->length ());
vtt = build_vtable (t, mangle_vtt_for_type (t), type);
initialize_artificial_var (vtt, inits);
DECL_CHAIN (vtt) = DECL_CHAIN (CLASSTYPE_VTABLES (t));
DECL_CHAIN (CLASSTYPE_VTABLES (t)) = vtt;
dump_vtt (t, vtt);
}
static tree
binfo_ctor_vtable (tree binfo)
{
tree vt;
while (1)
{
vt = BINFO_VTABLE (binfo);
if (TREE_CODE (vt) == TREE_LIST)
vt = TREE_VALUE (vt);
if (TREE_CODE (vt) == TREE_BINFO)
binfo = vt;
else
break;
}
return vt;
}
struct secondary_vptr_vtt_init_data
{
bool top_level_p;
tree index;
vec<constructor_elt, va_gc> *inits;
tree type_being_constructed;
};
static void
build_vtt_inits (tree binfo, tree t, vec<constructor_elt, va_gc> **inits,
tree *index)
{
int i;
tree b;
tree init;
secondary_vptr_vtt_init_data data;
int top_level_p = SAME_BINFO_TYPE_P (BINFO_TYPE (binfo), t);
if (!CLASSTYPE_VBASECLASSES (BINFO_TYPE (binfo)))
return;
if (!top_level_p)
{
build_ctor_vtbl_group (binfo, t);
BINFO_SUBVTT_INDEX (binfo) = *index;
}
init = binfo_ctor_vtable (binfo);
CONSTRUCTOR_APPEND_ELT (*inits, NULL_TREE, init);
if (top_level_p)
{
gcc_assert (!BINFO_VPTR_INDEX (binfo));
BINFO_VPTR_INDEX (binfo) = *index;
}
*index = size_binop (PLUS_EXPR, *index, TYPE_SIZE_UNIT (ptr_type_node));
for (i = 0; BINFO_BASE_ITERATE (binfo, i, b); ++i)
if (!BINFO_VIRTUAL_P (b))
build_vtt_inits (b, t, inits, index);
data.top_level_p = top_level_p;
data.index = *index;
data.inits = *inits;
data.type_being_constructed = BINFO_TYPE (binfo);
dfs_walk_once (binfo, dfs_build_secondary_vptr_vtt_inits, NULL, &data);
*index = data.index;
*inits = data.inits;
if (top_level_p)
for (b = TYPE_BINFO (BINFO_TYPE (binfo)); b; b = TREE_CHAIN (b))
{
if (!BINFO_VIRTUAL_P (b))
continue;
build_vtt_inits (b, t, inits, index);
}
else
dfs_walk_all (binfo, dfs_fixup_binfo_vtbls, NULL, binfo);
}
static tree
dfs_build_secondary_vptr_vtt_inits (tree binfo, void *data_)
{
secondary_vptr_vtt_init_data *data = (secondary_vptr_vtt_init_data *)data_;
if (!TYPE_VFIELD (BINFO_TYPE (binfo)))
return dfs_skip_bases;
if (SAME_BINFO_TYPE_P (BINFO_TYPE (binfo), data->type_being_constructed))
return NULL_TREE;
if (!(CLASSTYPE_VBASECLASSES (BINFO_TYPE (binfo))
|| binfo_via_virtual (binfo, data->type_being_constructed)))
return dfs_skip_bases;
if (!BINFO_VIRTUAL_P (binfo) && BINFO_PRIMARY_P (binfo))
return NULL_TREE;
if (data->top_level_p)
{
gcc_assert (!BINFO_VPTR_INDEX (binfo));
BINFO_VPTR_INDEX (binfo) = data->index;
if (BINFO_VIRTUAL_P (binfo))
{
while (BINFO_PRIMARY_P (binfo))
binfo = BINFO_INHERITANCE_CHAIN (binfo);
}
}
CONSTRUCTOR_APPEND_ELT (data->inits, NULL_TREE, binfo_ctor_vtable (binfo));
data->index = size_binop (PLUS_EXPR, data->index,
TYPE_SIZE_UNIT (ptr_type_node));
return NULL_TREE;
}
static tree
dfs_fixup_binfo_vtbls (tree binfo, void* data)
{
tree vtable = BINFO_VTABLE (binfo);
if (!TYPE_CONTAINS_VPTR_P (BINFO_TYPE (binfo)))
return dfs_skip_bases;
if (!vtable)
return NULL_TREE;
if (TREE_CODE (vtable) == TREE_LIST
&& (TREE_PURPOSE (vtable) == (tree) data))
BINFO_VTABLE (binfo) = TREE_CHAIN (vtable);
return NULL_TREE;
}
static void
build_ctor_vtbl_group (tree binfo, tree t)
{
tree type;
tree vtbl;
tree id;
tree vbase;
vec<constructor_elt, va_gc> *v;
id = mangle_ctor_vtbl_for_type (t, binfo);
if (get_global_binding (id))
return;
gcc_assert (!SAME_BINFO_TYPE_P (BINFO_TYPE (binfo), t));
vtbl = build_vtable (t, id, ptr_type_node);
DECL_CONSTRUCTION_VTABLE_P (vtbl) = 1;
DECL_VISIBILITY (vtbl) = VISIBILITY_HIDDEN;
DECL_VISIBILITY_SPECIFIED (vtbl) = true;
v = NULL;
accumulate_vtbl_inits (binfo, TYPE_BINFO (TREE_TYPE (binfo)),
binfo, vtbl, t, &v);
for (vbase = TYPE_BINFO (BINFO_TYPE (binfo));
vbase;
vbase = TREE_CHAIN (vbase))
{
tree b;
if (!BINFO_VIRTUAL_P (vbase))
continue;
b = copied_binfo (vbase, binfo);
accumulate_vtbl_inits (b, vbase, binfo, vtbl, t, &v);
}
type = build_array_of_n_type (vtable_entry_type, v->length ());
layout_type (type);
TREE_TYPE (vtbl) = type;
DECL_SIZE (vtbl) = DECL_SIZE_UNIT (vtbl) = NULL_TREE;
layout_decl (vtbl, 0);
CLASSTYPE_VTABLES (t) = chainon (CLASSTYPE_VTABLES (t), vtbl);
initialize_artificial_var (vtbl, v);
dump_vtable (t, binfo, vtbl);
}
static void
accumulate_vtbl_inits (tree binfo,
tree orig_binfo,
tree rtti_binfo,
tree vtbl,
tree t,
vec<constructor_elt, va_gc> **inits)
{
int i;
tree base_binfo;
int ctor_vtbl_p = !SAME_BINFO_TYPE_P (BINFO_TYPE (rtti_binfo), t);
gcc_assert (SAME_BINFO_TYPE_P (BINFO_TYPE (binfo), BINFO_TYPE (orig_binfo)));
if (!TYPE_CONTAINS_VPTR_P (BINFO_TYPE (binfo)))
return;
if (ctor_vtbl_p
&& !CLASSTYPE_VBASECLASSES (BINFO_TYPE (binfo))
&& !binfo_via_virtual (orig_binfo, BINFO_TYPE (rtti_binfo)))
return;
dfs_accumulate_vtbl_inits (binfo, orig_binfo, rtti_binfo, vtbl, t, inits);
for (i = 0; BINFO_BASE_ITERATE (binfo, i, base_binfo); ++i)
{
if (BINFO_VIRTUAL_P (base_binfo))
continue;
accumulate_vtbl_inits (base_binfo,
BINFO_BASE_BINFO (orig_binfo, i),
rtti_binfo, vtbl, t,
inits);
}
}
static void
dfs_accumulate_vtbl_inits (tree binfo,
tree orig_binfo,
tree rtti_binfo,
tree orig_vtbl,
tree t,
vec<constructor_elt, va_gc> **l)
{
tree vtbl = NULL_TREE;
int ctor_vtbl_p = !SAME_BINFO_TYPE_P (BINFO_TYPE (rtti_binfo), t);
int n_inits;
if (ctor_vtbl_p
&& BINFO_VIRTUAL_P (orig_binfo) && BINFO_PRIMARY_P (orig_binfo))
{
tree b;
tree last = NULL_TREE;
b = binfo;
while (BINFO_PRIMARY_P (b))
{
b = BINFO_INHERITANCE_CHAIN (b);
last = b;
if (BINFO_VIRTUAL_P (b) || b == rtti_binfo)
goto found;
}
for (b = last; b; b = BINFO_INHERITANCE_CHAIN (b))
if (BINFO_VIRTUAL_P (b) || b == rtti_binfo)
break;
found:
if (b == rtti_binfo
|| (b && binfo_for_vbase (BINFO_TYPE (b), BINFO_TYPE (rtti_binfo))))
vtbl = last;
}
else if (!BINFO_NEW_VTABLE_MARKED (orig_binfo))
return;
n_inits = vec_safe_length (*l);
if (!vtbl)
{
tree index;
int non_fn_entries;
build_vtbl_initializer (binfo, orig_binfo, t, rtti_binfo,
&non_fn_entries, l);
vtbl = build1 (ADDR_EXPR, vtbl_ptr_type_node, orig_vtbl);
index = size_binop (MULT_EXPR,
TYPE_SIZE_UNIT (vtable_entry_type),
size_int (non_fn_entries + n_inits));
vtbl = fold_build_pointer_plus (vtbl, index);
}
if (ctor_vtbl_p)
BINFO_VTABLE (binfo) = tree_cons (rtti_binfo, vtbl, BINFO_VTABLE (binfo));
else if (BINFO_PRIMARY_P (binfo) && BINFO_VIRTUAL_P (binfo))
(*l)->truncate (n_inits);
else
BINFO_VTABLE (binfo) = vtbl;
}
static GTY(()) tree abort_fndecl_addr;
static GTY(()) tree dvirt_fn;
static void
build_vtbl_initializer (tree binfo,
tree orig_binfo,
tree t,
tree rtti_binfo,
int* non_fn_entries_p,
vec<constructor_elt, va_gc> **inits)
{
tree v;
vtbl_init_data vid;
unsigned ix, jx;
tree vbinfo;
vec<tree, va_gc> *vbases;
constructor_elt *e;
memset (&vid, 0, sizeof (vid));
vid.binfo = binfo;
vid.derived = t;
vid.rtti_binfo = rtti_binfo;
vid.primary_vtbl_p = SAME_BINFO_TYPE_P (BINFO_TYPE (binfo), t);
vid.ctor_vtbl_p = !SAME_BINFO_TYPE_P (BINFO_TYPE (rtti_binfo), t);
vid.generate_vcall_entries = true;
vid.index = ssize_int(-3 * TARGET_VTABLE_DATA_ENTRY_DISTANCE);
build_rtti_vtbl_entries (binfo, &vid);
vec_alloc (vid.fns, 32);
build_vcall_and_vbase_vtbl_entries (binfo, &vid);
for (vbases = CLASSTYPE_VBASECLASSES (t), ix = 0;
vec_safe_iterate (vbases, ix, &vbinfo); ix++)
BINFO_VTABLE_PATH_MARKED (vbinfo) = 0;
if (TARGET_VTABLE_DATA_ENTRY_DISTANCE > 1)
{
int n_entries = vec_safe_length (vid.inits);
vec_safe_grow (vid.inits, TARGET_VTABLE_DATA_ENTRY_DISTANCE * n_entries);
for (ix = n_entries - 1;
vid.inits->iterate (ix, &e);
ix--)
{
int j;
int new_position = (TARGET_VTABLE_DATA_ENTRY_DISTANCE * ix
+ (TARGET_VTABLE_DATA_ENTRY_DISTANCE - 1));
(*vid.inits)[new_position] = *e;
for (j = 1; j < TARGET_VTABLE_DATA_ENTRY_DISTANCE; ++j)
{
constructor_elt *f = &(*vid.inits)[new_position - j];
f->index = NULL_TREE;
f->value = build1 (NOP_EXPR, vtable_entry_type,
null_pointer_node);
}
}
}
if (non_fn_entries_p)
*non_fn_entries_p = vec_safe_length (vid.inits);
jx = vec_safe_length (*inits);
vec_safe_grow (*inits, jx + vid.inits->length ());
for (ix = vid.inits->length () - 1;
vid.inits->iterate (ix, &e);
ix--, jx++)
(**inits)[jx] = *e;
for (v = BINFO_VIRTUALS (orig_binfo); v; v = TREE_CHAIN (v))
{
tree delta;
tree vcall_index;
tree fn, fn_original;
tree init = NULL_TREE;
fn = BV_FN (v);
fn_original = fn;
if (DECL_THUNK_P (fn))
{
if (!DECL_NAME (fn))
finish_thunk (fn);
if (THUNK_ALIAS (fn))
{
fn = THUNK_ALIAS (fn);
BV_FN (v) = fn;
}
fn_original = THUNK_TARGET (fn);
}
if (BV_LOST_PRIMARY (v))
init = size_zero_node;
if (! init)
{
delta = BV_DELTA (v);
vcall_index = BV_VCALL_INDEX (v);
gcc_assert (TREE_CODE (delta) == INTEGER_CST);
gcc_assert (TREE_CODE (fn) == FUNCTION_DECL);
if (DECL_PURE_VIRTUAL_P (fn_original))
{
fn = abort_fndecl;
if (!TARGET_VTABLE_USES_DESCRIPTORS)
{
if (abort_fndecl_addr == NULL)
abort_fndecl_addr
= fold_convert (vfunc_ptr_type_node,
build_fold_addr_expr (fn));
init = abort_fndecl_addr;
}
}
else if (DECL_DELETED_FN (fn_original))
{
if (!dvirt_fn)
{
tree name = get_identifier ("__cxa_deleted_virtual");
dvirt_fn = get_global_binding (name);
if (!dvirt_fn)
dvirt_fn = push_library_fn
(name,
build_function_type_list (void_type_node, NULL_TREE),
NULL_TREE, ECF_NORETURN | ECF_COLD);
}
fn = dvirt_fn;
if (!TARGET_VTABLE_USES_DESCRIPTORS)
init = fold_convert (vfunc_ptr_type_node,
build_fold_addr_expr (fn));
}
else
{
if (!integer_zerop (delta) || vcall_index)
{
fn = make_thunk (fn, 1,
delta, vcall_index);
if (!DECL_NAME (fn))
finish_thunk (fn);
}
if (!TARGET_VTABLE_USES_DESCRIPTORS)
init = fold_convert (vfunc_ptr_type_node,
build_fold_addr_expr (fn));
if (DECL_DESTRUCTOR_P (fn_original)
&& (CLASSTYPE_PURE_VIRTUALS (DECL_CONTEXT (fn_original))
|| orig_binfo != binfo))
init = size_zero_node;
}
}
if (TARGET_VTABLE_USES_DESCRIPTORS)
{
int i;
if (init == size_zero_node)
for (i = 0; i < TARGET_VTABLE_USES_DESCRIPTORS; ++i)
CONSTRUCTOR_APPEND_ELT (*inits, NULL_TREE, init);
else
for (i = 0; i < TARGET_VTABLE_USES_DESCRIPTORS; ++i)
{
tree fdesc = build2 (FDESC_EXPR, vfunc_ptr_type_node,
fn, build_int_cst (NULL_TREE, i));
TREE_CONSTANT (fdesc) = 1;
CONSTRUCTOR_APPEND_ELT (*inits, NULL_TREE, fdesc);
}
}
else
CONSTRUCTOR_APPEND_ELT (*inits, NULL_TREE, init);
}
}
static void
build_vcall_and_vbase_vtbl_entries (tree binfo, vtbl_init_data* vid)
{
tree b;
b = get_primary_binfo (binfo);
if (b)
build_vcall_and_vbase_vtbl_entries (b, vid);
build_vbase_offset_vtbl_entries (binfo, vid);
build_vcall_offset_vtbl_entries (binfo, vid);
}
static void
build_vbase_offset_vtbl_entries (tree binfo, vtbl_init_data* vid)
{
tree vbase;
tree t;
tree non_primary_binfo;
if (!CLASSTYPE_VBASECLASSES (BINFO_TYPE (binfo)))
return;
t = vid->derived;
non_primary_binfo = binfo;
while (BINFO_INHERITANCE_CHAIN (non_primary_binfo))
{
tree b;
if (BINFO_VIRTUAL_P (non_primary_binfo))
{
non_primary_binfo = vid->binfo;
break;
}
b = BINFO_INHERITANCE_CHAIN (non_primary_binfo);
if (get_primary_binfo (b) != non_primary_binfo)
break;
non_primary_binfo = b;
}
for (vbase = TYPE_BINFO (BINFO_TYPE (binfo));
vbase;
vbase = TREE_CHAIN (vbase))
{
tree b;
tree delta;
if (!BINFO_VIRTUAL_P (vbase))
continue;
b = copied_binfo (vbase, binfo);
if (BINFO_VTABLE_PATH_MARKED (b))
continue;
BINFO_VTABLE_PATH_MARKED (b) = 1;
delta = size_binop (MULT_EXPR,
vid->index,
fold_convert (ssizetype,
TYPE_SIZE_UNIT (vtable_entry_type)));
if (vid->primary_vtbl_p)
BINFO_VPTR_FIELD (b) = delta;
if (binfo != TYPE_BINFO (t))
gcc_assert (tree_int_cst_equal (delta, BINFO_VPTR_FIELD (vbase)));
vid->index = size_binop (MINUS_EXPR, vid->index,
ssize_int (TARGET_VTABLE_DATA_ENTRY_DISTANCE));
delta = size_diffop_loc (input_location,
BINFO_OFFSET (b), BINFO_OFFSET (non_primary_binfo));
CONSTRUCTOR_APPEND_ELT (vid->inits, NULL_TREE,
fold_build1_loc (input_location, NOP_EXPR,
vtable_entry_type, delta));
}
}
static void
build_vcall_offset_vtbl_entries (tree binfo, vtbl_init_data* vid)
{
if (binfo == TYPE_BINFO (vid->derived)
|| (BINFO_VIRTUAL_P (binfo) 
&& binfo != vid->rtti_binfo))
{
vid->vbase = binfo;
if (!BINFO_VIRTUAL_P (binfo))
vid->generate_vcall_entries = false;
add_vcall_offset_vtbl_entries_r (binfo, vid);
}
}
static void
add_vcall_offset_vtbl_entries_r (tree binfo, vtbl_init_data* vid)
{
int i;
tree primary_binfo;
tree base_binfo;
if (BINFO_VIRTUAL_P (binfo) && vid->vbase != binfo)
return;
primary_binfo = get_primary_binfo (binfo);
if (primary_binfo)
add_vcall_offset_vtbl_entries_r (primary_binfo, vid);
add_vcall_offset_vtbl_entries_1 (binfo, vid);
for (i = 0; BINFO_BASE_ITERATE (binfo, i, base_binfo); ++i)
if (base_binfo != primary_binfo)
add_vcall_offset_vtbl_entries_r (base_binfo, vid);
}
static void
add_vcall_offset_vtbl_entries_1 (tree binfo, vtbl_init_data* vid)
{
tree orig_fn;
for (orig_fn = TYPE_FIELDS (BINFO_TYPE (binfo));
orig_fn;
orig_fn = DECL_CHAIN (orig_fn))
if (TREE_CODE (orig_fn) == FUNCTION_DECL && DECL_VINDEX (orig_fn))
add_vcall_offset (orig_fn, binfo, vid);
}
static void
add_vcall_offset (tree orig_fn, tree binfo, vtbl_init_data *vid)
{
size_t i;
tree vcall_offset;
tree derived_entry;
FOR_EACH_VEC_SAFE_ELT (vid->fns, i, derived_entry)
{
if (same_signature_p (derived_entry, orig_fn)
|| (DECL_DESTRUCTOR_P (derived_entry)
&& DECL_DESTRUCTOR_P (orig_fn)))
return;
}
if (vid->binfo == TYPE_BINFO (vid->derived))
{
tree_pair_s elt = {orig_fn, vid->index};
vec_safe_push (CLASSTYPE_VCALL_INDICES (vid->derived), elt);
}
vid->index = size_binop (MINUS_EXPR, vid->index,
ssize_int (TARGET_VTABLE_DATA_ENTRY_DISTANCE));
vec_safe_push (vid->fns, orig_fn);
if (vid->generate_vcall_entries)
{
tree base;
tree fn;
fn = find_final_overrider (vid->rtti_binfo, binfo, orig_fn);
if (fn == error_mark_node)
vcall_offset = build_zero_cst (vtable_entry_type);
else
{
base = TREE_VALUE (fn);
vcall_offset = size_diffop_loc (input_location,
BINFO_OFFSET (base),
BINFO_OFFSET (vid->binfo));
vcall_offset = fold_build1_loc (input_location,
NOP_EXPR, vtable_entry_type,
vcall_offset);
}
CONSTRUCTOR_APPEND_ELT (vid->inits, NULL_TREE, vcall_offset);
}
}
static void
build_rtti_vtbl_entries (tree binfo, vtbl_init_data* vid)
{
tree b;
tree t;
tree offset;
tree decl;
tree init;
t = BINFO_TYPE (vid->rtti_binfo);
b = most_primary_binfo (binfo);
offset = size_diffop_loc (input_location,
BINFO_OFFSET (vid->rtti_binfo), BINFO_OFFSET (b));
if (flag_rtti)
decl = build_address (get_tinfo_decl (t));
else
decl = integer_zero_node;
init = build_nop (vfunc_ptr_type_node, decl);
CONSTRUCTOR_APPEND_ELT (vid->inits, NULL_TREE, init);
init = build_nop (vfunc_ptr_type_node, offset);
CONSTRUCTOR_APPEND_ELT (vid->inits, NULL_TREE, init);
}
bool
uniquely_derived_from_p (tree parent, tree type)
{
tree base = lookup_base (type, parent, ba_unique, NULL, tf_none);
return base && base != error_mark_node;
}
bool
publicly_uniquely_derived_p (tree parent, tree type)
{
tree base = lookup_base (type, parent, ba_ignore_scope | ba_check,
NULL, tf_none);
return base && base != error_mark_node;
}
tree
common_enclosing_class (tree ctx1, tree ctx2)
{
if (!TYPE_P (ctx1) || !TYPE_P (ctx2))
return NULL_TREE;
gcc_assert (ctx1 == TYPE_MAIN_VARIANT (ctx1)
&& ctx2 == TYPE_MAIN_VARIANT (ctx2));
if (ctx1 == ctx2)
return ctx1;
for (tree t = ctx1; TYPE_P (t); t = TYPE_CONTEXT (t))
TYPE_MARKED_P (t) = true;
tree found = NULL_TREE;
for (tree t = ctx2; TYPE_P (t); t = TYPE_CONTEXT (t))
if (TYPE_MARKED_P (t))
{
found = t;
break;
}
for (tree t = ctx1; TYPE_P (t); t = TYPE_CONTEXT (t))
TYPE_MARKED_P (t) = false;
return found;
}
#include "gt-cp-class.h"
