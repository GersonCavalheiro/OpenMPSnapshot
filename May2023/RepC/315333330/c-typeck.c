#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "memmodel.h"
#include "target.h"
#include "function.h"
#include "bitmap.h"
#include "c-tree.h"
#include "gimple-expr.h"
#include "predict.h"
#include "stor-layout.h"
#include "trans-mem.h"
#include "varasm.h"
#include "stmt.h"
#include "langhooks.h"
#include "c-lang.h"
#include "intl.h"
#include "tree-iterator.h"
#include "gimplify.h"
#include "tree-inline.h"
#include "omp-general.h"
#include "c-family/c-objc.h"
#include "c-family/c-ubsan.h"
#include "gomp-constants.h"
#include "spellcheck-tree.h"
#include "gcc-rich-location.h"
#include "stringpool.h"
#include "attribs.h"
#include "asan.h"
enum impl_conv {
ic_argpass,
ic_assign,
ic_init,
ic_return
};
int in_alignof;
int in_sizeof;
int in_typeof;
tree c_last_sizeof_arg;
location_t c_last_sizeof_loc;
static int found_missing_braces;
static int require_constant_value;
static int require_constant_elements;
static bool null_pointer_constant_p (const_tree);
static tree qualify_type (tree, tree);
static int tagged_types_tu_compatible_p (const_tree, const_tree, bool *,
bool *);
static int comp_target_types (location_t, tree, tree);
static int function_types_compatible_p (const_tree, const_tree, bool *,
bool *);
static int type_lists_compatible_p (const_tree, const_tree, bool *, bool *);
static tree lookup_field (tree, tree);
static int convert_arguments (location_t, vec<location_t>, tree,
vec<tree, va_gc> *, vec<tree, va_gc> *, tree,
tree);
static tree pointer_diff (location_t, tree, tree, tree *);
static tree convert_for_assignment (location_t, location_t, tree, tree, tree,
enum impl_conv, bool, tree, tree, int);
static tree valid_compound_expr_initializer (tree, tree);
static void push_string (const char *);
static void push_member_name (tree);
static int spelling_length (void);
static char *print_spelling (char *);
static void warning_init (location_t, int, const char *);
static tree digest_init (location_t, tree, tree, tree, bool, bool, int);
static void output_init_element (location_t, tree, tree, bool, tree, tree, bool,
bool, struct obstack *);
static void output_pending_init_elements (int, struct obstack *);
static bool set_designator (location_t, bool, struct obstack *);
static void push_range_stack (tree, struct obstack *);
static void add_pending_init (location_t, tree, tree, tree, bool,
struct obstack *);
static void set_nonincremental_init (struct obstack *);
static void set_nonincremental_init_from_string (tree, struct obstack *);
static tree find_init_member (tree, struct obstack *);
static void readonly_warning (tree, enum lvalue_use);
static int lvalue_or_else (location_t, const_tree, enum lvalue_use);
static void record_maybe_used_decl (tree);
static int comptypes_internal (const_tree, const_tree, bool *, bool *);

static bool
null_pointer_constant_p (const_tree expr)
{
tree type = TREE_TYPE (expr);
return (TREE_CODE (expr) == INTEGER_CST
&& !TREE_OVERFLOW (expr)
&& integer_zerop (expr)
&& (INTEGRAL_TYPE_P (type)
|| (TREE_CODE (type) == POINTER_TYPE
&& VOID_TYPE_P (TREE_TYPE (type))
&& TYPE_QUALS (TREE_TYPE (type)) == TYPE_UNQUALIFIED)));
}
static tree
note_integer_operands (tree expr)
{
tree ret;
if (TREE_CODE (expr) == INTEGER_CST && in_late_binary_op)
{
ret = copy_node (expr);
TREE_OVERFLOW (ret) = 1;
}
else
{
ret = build2 (C_MAYBE_CONST_EXPR, TREE_TYPE (expr), NULL_TREE, expr);
C_MAYBE_CONST_EXPR_INT_OPERANDS (ret) = 1;
}
return ret;
}
static inline tree
remove_c_maybe_const_expr (tree expr)
{
if (TREE_CODE (expr) == C_MAYBE_CONST_EXPR)
return C_MAYBE_CONST_EXPR_EXPR (expr);
else
return expr;
}

struct tagged_tu_seen_cache {
const struct tagged_tu_seen_cache * next;
const_tree t1;
const_tree t2;
int val;
};
static const struct tagged_tu_seen_cache * tagged_tu_seen_base;
static void free_all_tagged_tu_seen_up_to (const struct tagged_tu_seen_cache *);
tree
require_complete_type (location_t loc, tree value)
{
tree type = TREE_TYPE (value);
if (error_operand_p (value))
return error_mark_node;
if (COMPLETE_TYPE_P (type))
return value;
c_incomplete_type_error (loc, value, type);
return error_mark_node;
}
void
c_incomplete_type_error (location_t loc, const_tree value, const_tree type)
{
if (TREE_CODE (type) == ERROR_MARK)
return;
if (value != NULL_TREE && (VAR_P (value) || TREE_CODE (value) == PARM_DECL))
error_at (loc, "%qD has an incomplete type %qT", value, type);
else
{
retry:
switch (TREE_CODE (type))
{
case RECORD_TYPE:
case UNION_TYPE:
case ENUMERAL_TYPE:
break;
case VOID_TYPE:
error_at (loc, "invalid use of void expression");
return;
case ARRAY_TYPE:
if (TYPE_DOMAIN (type))
{
if (TYPE_MAX_VALUE (TYPE_DOMAIN (type)) == NULL)
{
error_at (loc, "invalid use of flexible array member");
return;
}
type = TREE_TYPE (type);
goto retry;
}
error_at (loc, "invalid use of array with unspecified bounds");
return;
default:
gcc_unreachable ();
}
if (TREE_CODE (TYPE_NAME (type)) == IDENTIFIER_NODE)
error_at (loc, "invalid use of undefined type %qT", type);
else
error_at (loc, "invalid use of incomplete typedef %qT", type);
}
}
tree
c_type_promotes_to (tree type)
{
tree ret = NULL_TREE;
if (TYPE_MAIN_VARIANT (type) == float_type_node)
ret = double_type_node;
else if (c_promoting_integer_type_p (type))
{
if (TYPE_UNSIGNED (type)
&& (TYPE_PRECISION (type) == TYPE_PRECISION (integer_type_node)))
ret = unsigned_type_node;
else
ret = integer_type_node;
}
if (ret != NULL_TREE)
return (TYPE_ATOMIC (type)
? c_build_qualified_type (ret, TYPE_QUAL_ATOMIC)
: ret);
return type;
}
static bool
addr_space_superset (addr_space_t as1, addr_space_t as2, addr_space_t *common)
{
if (as1 == as2)
{
*common = as1;
return true;
}
else if (targetm.addr_space.subset_p (as1, as2))
{
*common = as2;
return true;
}
else if (targetm.addr_space.subset_p (as2, as1))
{
*common = as1;
return true;
}
else
return false;
}
static tree
qualify_type (tree type, tree like)
{
addr_space_t as_type = TYPE_ADDR_SPACE (type);
addr_space_t as_like = TYPE_ADDR_SPACE (like);
addr_space_t as_common;
if (!addr_space_superset (as_type, as_like, &as_common))
{
as_common = as_type;
error ("%qT and %qT are in disjoint named address spaces",
type, like);
}
return c_build_qualified_type (type,
TYPE_QUALS_NO_ADDR_SPACE (type)
| TYPE_QUALS_NO_ADDR_SPACE_NO_ATOMIC (like)
| ENCODE_QUAL_ADDR_SPACE (as_common));
}
bool
c_vla_type_p (const_tree t)
{
if (TREE_CODE (t) == ARRAY_TYPE
&& C_TYPE_VARIABLE_SIZE (t))
return true;
return false;
}

tree
composite_type (tree t1, tree t2)
{
enum tree_code code1;
enum tree_code code2;
tree attributes;
if (t1 == t2) return t1;
if (t1 == error_mark_node)
return t2;
if (t2 == error_mark_node)
return t1;
code1 = TREE_CODE (t1);
code2 = TREE_CODE (t2);
attributes = targetm.merge_type_attributes (t1, t2);
if (code1 == ENUMERAL_TYPE && code2 == INTEGER_TYPE)
return t1;
if (code2 == ENUMERAL_TYPE && code1 == INTEGER_TYPE)
return t2;
gcc_assert (code1 == code2);
switch (code1)
{
case POINTER_TYPE:
{
tree pointed_to_1 = TREE_TYPE (t1);
tree pointed_to_2 = TREE_TYPE (t2);
tree target = composite_type (pointed_to_1, pointed_to_2);
t1 = build_pointer_type_for_mode (target, TYPE_MODE (t1), false);
t1 = build_type_attribute_variant (t1, attributes);
return qualify_type (t1, t2);
}
case ARRAY_TYPE:
{
tree elt = composite_type (TREE_TYPE (t1), TREE_TYPE (t2));
int quals;
tree unqual_elt;
tree d1 = TYPE_DOMAIN (t1);
tree d2 = TYPE_DOMAIN (t2);
bool d1_variable, d2_variable;
bool d1_zero, d2_zero;
bool t1_complete, t2_complete;
gcc_assert (!TYPE_QUALS_NO_ADDR_SPACE (t1)
&& !TYPE_QUALS_NO_ADDR_SPACE (t2));
t1_complete = COMPLETE_TYPE_P (t1);
t2_complete = COMPLETE_TYPE_P (t2);
d1_zero = d1 == NULL_TREE || !TYPE_MAX_VALUE (d1);
d2_zero = d2 == NULL_TREE || !TYPE_MAX_VALUE (d2);
d1_variable = (!d1_zero
&& (TREE_CODE (TYPE_MIN_VALUE (d1)) != INTEGER_CST
|| TREE_CODE (TYPE_MAX_VALUE (d1)) != INTEGER_CST));
d2_variable = (!d2_zero
&& (TREE_CODE (TYPE_MIN_VALUE (d2)) != INTEGER_CST
|| TREE_CODE (TYPE_MAX_VALUE (d2)) != INTEGER_CST));
d1_variable = d1_variable || (d1_zero && c_vla_type_p (t1));
d2_variable = d2_variable || (d2_zero && c_vla_type_p (t2));
if (elt == TREE_TYPE (t1) && TYPE_DOMAIN (t1)
&& (d2_variable || d2_zero || !d1_variable))
return build_type_attribute_variant (t1, attributes);
if (elt == TREE_TYPE (t2) && TYPE_DOMAIN (t2)
&& (d1_variable || d1_zero || !d2_variable))
return build_type_attribute_variant (t2, attributes);
if (elt == TREE_TYPE (t1) && !TYPE_DOMAIN (t2) && !TYPE_DOMAIN (t1))
return build_type_attribute_variant (t1, attributes);
if (elt == TREE_TYPE (t2) && !TYPE_DOMAIN (t2) && !TYPE_DOMAIN (t1))
return build_type_attribute_variant (t2, attributes);
quals = TYPE_QUALS (strip_array_types (elt));
unqual_elt = c_build_qualified_type (elt, TYPE_UNQUALIFIED);
t1 = build_array_type (unqual_elt,
TYPE_DOMAIN ((TYPE_DOMAIN (t1)
&& (d2_variable
|| d2_zero
|| !d1_variable))
? t1
: t2));
if (d1_zero && d2_zero
&& (t1_complete || t2_complete)
&& !COMPLETE_TYPE_P (t1))
{
TYPE_SIZE (t1) = bitsize_zero_node;
TYPE_SIZE_UNIT (t1) = size_zero_node;
}
t1 = c_build_qualified_type (t1, quals);
return build_type_attribute_variant (t1, attributes);
}
case ENUMERAL_TYPE:
case RECORD_TYPE:
case UNION_TYPE:
if (attributes != NULL)
{
if (attribute_list_equal (TYPE_ATTRIBUTES (t1), attributes))
return t1;
if (attribute_list_equal (TYPE_ATTRIBUTES (t2), attributes))
return t2;
}
return build_type_attribute_variant (t1, attributes);
case FUNCTION_TYPE:
{
tree valtype = composite_type (TREE_TYPE (t1), TREE_TYPE (t2));
tree p1 = TYPE_ARG_TYPES (t1);
tree p2 = TYPE_ARG_TYPES (t2);
int len;
tree newargs, n;
int i;
if (valtype == TREE_TYPE (t1) && !TYPE_ARG_TYPES (t2))
return build_type_attribute_variant (t1, attributes);
if (valtype == TREE_TYPE (t2) && !TYPE_ARG_TYPES (t1))
return build_type_attribute_variant (t2, attributes);
if (TYPE_ARG_TYPES (t1) == NULL_TREE)
{
t1 = build_function_type (valtype, TYPE_ARG_TYPES (t2));
t1 = build_type_attribute_variant (t1, attributes);
return qualify_type (t1, t2);
}
if (TYPE_ARG_TYPES (t2) == NULL_TREE)
{
t1 = build_function_type (valtype, TYPE_ARG_TYPES (t1));
t1 = build_type_attribute_variant (t1, attributes);
return qualify_type (t1, t2);
}
for (len = 0, newargs = p1;
newargs && newargs != void_list_node;
len++, newargs = TREE_CHAIN (newargs))
;
for (i = 0; i < len; i++)
newargs = tree_cons (NULL_TREE, NULL_TREE, newargs);
n = newargs;
for (; p1 && p1 != void_list_node;
p1 = TREE_CHAIN (p1), p2 = TREE_CHAIN (p2), n = TREE_CHAIN (n))
{
if (TREE_VALUE (p1) == NULL_TREE)
{
TREE_VALUE (n) = TREE_VALUE (p2);
goto parm_done;
}
if (TREE_VALUE (p2) == NULL_TREE)
{
TREE_VALUE (n) = TREE_VALUE (p1);
goto parm_done;
}
if (TREE_CODE (TREE_VALUE (p1)) == UNION_TYPE
&& TREE_VALUE (p1) != TREE_VALUE (p2))
{
tree memb;
tree mv2 = TREE_VALUE (p2);
if (mv2 && mv2 != error_mark_node
&& TREE_CODE (mv2) != ARRAY_TYPE)
mv2 = TYPE_MAIN_VARIANT (mv2);
for (memb = TYPE_FIELDS (TREE_VALUE (p1));
memb; memb = DECL_CHAIN (memb))
{
tree mv3 = TREE_TYPE (memb);
if (mv3 && mv3 != error_mark_node
&& TREE_CODE (mv3) != ARRAY_TYPE)
mv3 = TYPE_MAIN_VARIANT (mv3);
if (comptypes (mv3, mv2))
{
TREE_VALUE (n) = composite_type (TREE_TYPE (memb),
TREE_VALUE (p2));
pedwarn (input_location, OPT_Wpedantic,
"function types not truly compatible in ISO C");
goto parm_done;
}
}
}
if (TREE_CODE (TREE_VALUE (p2)) == UNION_TYPE
&& TREE_VALUE (p2) != TREE_VALUE (p1))
{
tree memb;
tree mv1 = TREE_VALUE (p1);
if (mv1 && mv1 != error_mark_node
&& TREE_CODE (mv1) != ARRAY_TYPE)
mv1 = TYPE_MAIN_VARIANT (mv1);
for (memb = TYPE_FIELDS (TREE_VALUE (p2));
memb; memb = DECL_CHAIN (memb))
{
tree mv3 = TREE_TYPE (memb);
if (mv3 && mv3 != error_mark_node
&& TREE_CODE (mv3) != ARRAY_TYPE)
mv3 = TYPE_MAIN_VARIANT (mv3);
if (comptypes (mv3, mv1))
{
TREE_VALUE (n) = composite_type (TREE_TYPE (memb),
TREE_VALUE (p1));
pedwarn (input_location, OPT_Wpedantic,
"function types not truly compatible in ISO C");
goto parm_done;
}
}
}
TREE_VALUE (n) = composite_type (TREE_VALUE (p1), TREE_VALUE (p2));
parm_done: ;
}
t1 = build_function_type (valtype, newargs);
t1 = qualify_type (t1, t2);
}
default:
return build_type_attribute_variant (t1, attributes);
}
}
static tree
common_pointer_type (tree t1, tree t2)
{
tree attributes;
tree pointed_to_1, mv1;
tree pointed_to_2, mv2;
tree target;
unsigned target_quals;
addr_space_t as1, as2, as_common;
int quals1, quals2;
if (t1 == t2) return t1;
if (t1 == error_mark_node)
return t2;
if (t2 == error_mark_node)
return t1;
gcc_assert (TREE_CODE (t1) == POINTER_TYPE
&& TREE_CODE (t2) == POINTER_TYPE);
attributes = targetm.merge_type_attributes (t1, t2);
mv1 = pointed_to_1 = TREE_TYPE (t1);
mv2 = pointed_to_2 = TREE_TYPE (t2);
if (TREE_CODE (mv1) != ARRAY_TYPE)
mv1 = TYPE_MAIN_VARIANT (pointed_to_1);
if (TREE_CODE (mv2) != ARRAY_TYPE)
mv2 = TYPE_MAIN_VARIANT (pointed_to_2);
target = composite_type (mv1, mv2);
quals1 = TYPE_QUALS_NO_ADDR_SPACE (strip_array_types (pointed_to_1));
quals2 = TYPE_QUALS_NO_ADDR_SPACE (strip_array_types (pointed_to_2));
if (TREE_CODE (pointed_to_1) == FUNCTION_TYPE)
target_quals = (quals1 & quals2);
else
target_quals = (quals1 | quals2);
as1 = TYPE_ADDR_SPACE (pointed_to_1);
as2 = TYPE_ADDR_SPACE (pointed_to_2);
if (!addr_space_superset (as1, as2, &as_common))
gcc_unreachable ();
target_quals |= ENCODE_QUAL_ADDR_SPACE (as_common);
t1 = build_pointer_type (c_build_qualified_type (target, target_quals));
return build_type_attribute_variant (t1, attributes);
}
static tree
c_common_type (tree t1, tree t2)
{
enum tree_code code1;
enum tree_code code2;
if (t1 == error_mark_node)
return t2;
if (t2 == error_mark_node)
return t1;
if (TYPE_QUALS (t1) != TYPE_UNQUALIFIED)
t1 = TYPE_MAIN_VARIANT (t1);
if (TYPE_QUALS (t2) != TYPE_UNQUALIFIED)
t2 = TYPE_MAIN_VARIANT (t2);
if (TYPE_ATTRIBUTES (t1) != NULL_TREE)
t1 = build_type_attribute_variant (t1, NULL_TREE);
if (TYPE_ATTRIBUTES (t2) != NULL_TREE)
t2 = build_type_attribute_variant (t2, NULL_TREE);
if (t1 == t2) return t1;
code1 = TREE_CODE (t1);
code2 = TREE_CODE (t2);
gcc_assert (code1 == VECTOR_TYPE || code1 == COMPLEX_TYPE
|| code1 == FIXED_POINT_TYPE || code1 == REAL_TYPE
|| code1 == INTEGER_TYPE);
gcc_assert (code2 == VECTOR_TYPE || code2 == COMPLEX_TYPE
|| code2 == FIXED_POINT_TYPE || code2 == REAL_TYPE
|| code2 == INTEGER_TYPE);
if ((DECIMAL_FLOAT_TYPE_P (t1) || DECIMAL_FLOAT_TYPE_P (t2))
&& !(DECIMAL_FLOAT_TYPE_P (t1) && DECIMAL_FLOAT_TYPE_P (t2)))
{
if (code1 == VECTOR_TYPE || code2 == VECTOR_TYPE)
{
error ("can%'t mix operands of decimal float and vector types");
return error_mark_node;
}
if (code1 == COMPLEX_TYPE || code2 == COMPLEX_TYPE)
{
error ("can%'t mix operands of decimal float and complex types");
return error_mark_node;
}
if (code1 == REAL_TYPE && code2 == REAL_TYPE)
{
error ("can%'t mix operands of decimal float and other float types");
return error_mark_node;
}
}
if (code1 == VECTOR_TYPE)
return t1;
if (code2 == VECTOR_TYPE)
return t2;
if (code1 == COMPLEX_TYPE || code2 == COMPLEX_TYPE)
{
tree subtype1 = code1 == COMPLEX_TYPE ? TREE_TYPE (t1) : t1;
tree subtype2 = code2 == COMPLEX_TYPE ? TREE_TYPE (t2) : t2;
tree subtype = c_common_type (subtype1, subtype2);
if (code1 == COMPLEX_TYPE && TREE_TYPE (t1) == subtype)
return t1;
else if (code2 == COMPLEX_TYPE && TREE_TYPE (t2) == subtype)
return t2;
else
return build_complex_type (subtype);
}
if (code1 == REAL_TYPE && code2 != REAL_TYPE)
return t1;
if (code2 == REAL_TYPE && code1 != REAL_TYPE)
return t2;
if (code1 == REAL_TYPE && code2 == REAL_TYPE)
{
if (TYPE_MAIN_VARIANT (t1) == dfloat128_type_node
|| TYPE_MAIN_VARIANT (t2) == dfloat128_type_node)
return dfloat128_type_node;
else if (TYPE_MAIN_VARIANT (t1) == dfloat64_type_node
|| TYPE_MAIN_VARIANT (t2) == dfloat64_type_node)
return dfloat64_type_node;
else if (TYPE_MAIN_VARIANT (t1) == dfloat32_type_node
|| TYPE_MAIN_VARIANT (t2) == dfloat32_type_node)
return dfloat32_type_node;
}
if (code1 == FIXED_POINT_TYPE || code2 == FIXED_POINT_TYPE)
{
unsigned int unsignedp = 0, satp = 0;
scalar_mode m1, m2;
unsigned int fbit1, ibit1, fbit2, ibit2, max_fbit, max_ibit;
m1 = SCALAR_TYPE_MODE (t1);
m2 = SCALAR_TYPE_MODE (t2);
if (TYPE_SATURATING (t1) || TYPE_SATURATING (t2))
satp = 1;
if ((TYPE_UNSIGNED (t1) && TYPE_UNSIGNED (t2)
&& code1 == FIXED_POINT_TYPE && code2 == FIXED_POINT_TYPE)
|| (code1 == FIXED_POINT_TYPE && code2 != FIXED_POINT_TYPE
&& TYPE_UNSIGNED (t1))
|| (code1 != FIXED_POINT_TYPE && code2 == FIXED_POINT_TYPE
&& TYPE_UNSIGNED (t2)))
unsignedp = 1;
if (unsignedp == 0)
{
if (code1 == FIXED_POINT_TYPE && TYPE_UNSIGNED (t1))
{
enum mode_class mclass = (enum mode_class) 0;
if (GET_MODE_CLASS (m1) == MODE_UFRACT)
mclass = MODE_FRACT;
else if (GET_MODE_CLASS (m1) == MODE_UACCUM)
mclass = MODE_ACCUM;
else
gcc_unreachable ();
m1 = as_a <scalar_mode>
(mode_for_size (GET_MODE_PRECISION (m1), mclass, 0));
}
if (code2 == FIXED_POINT_TYPE && TYPE_UNSIGNED (t2))
{
enum mode_class mclass = (enum mode_class) 0;
if (GET_MODE_CLASS (m2) == MODE_UFRACT)
mclass = MODE_FRACT;
else if (GET_MODE_CLASS (m2) == MODE_UACCUM)
mclass = MODE_ACCUM;
else
gcc_unreachable ();
m2 = as_a <scalar_mode>
(mode_for_size (GET_MODE_PRECISION (m2), mclass, 0));
}
}
if (code1 == FIXED_POINT_TYPE)
{
fbit1 = GET_MODE_FBIT (m1);
ibit1 = GET_MODE_IBIT (m1);
}
else
{
fbit1 = 0;
ibit1 = TYPE_PRECISION (t1) - (!TYPE_UNSIGNED (t1));
}
if (code2 == FIXED_POINT_TYPE)
{
fbit2 = GET_MODE_FBIT (m2);
ibit2 = GET_MODE_IBIT (m2);
}
else
{
fbit2 = 0;
ibit2 = TYPE_PRECISION (t2) - (!TYPE_UNSIGNED (t2));
}
max_ibit = ibit1 >= ibit2 ?  ibit1 : ibit2;
max_fbit = fbit1 >= fbit2 ?  fbit1 : fbit2;
return c_common_fixed_point_type_for_size (max_ibit, max_fbit, unsignedp,
satp);
}
if (TYPE_PRECISION (t1) > TYPE_PRECISION (t2))
return t1;
else if (TYPE_PRECISION (t2) > TYPE_PRECISION (t1))
return t2;
if (TYPE_MAIN_VARIANT (t1) == long_long_unsigned_type_node
|| TYPE_MAIN_VARIANT (t2) == long_long_unsigned_type_node)
return long_long_unsigned_type_node;
if (TYPE_MAIN_VARIANT (t1) == long_long_integer_type_node
|| TYPE_MAIN_VARIANT (t2) == long_long_integer_type_node)
{
if (TYPE_UNSIGNED (t1) || TYPE_UNSIGNED (t2))
return long_long_unsigned_type_node;
else
return long_long_integer_type_node;
}
if (TYPE_MAIN_VARIANT (t1) == long_unsigned_type_node
|| TYPE_MAIN_VARIANT (t2) == long_unsigned_type_node)
return long_unsigned_type_node;
if (TYPE_MAIN_VARIANT (t1) == long_integer_type_node
|| TYPE_MAIN_VARIANT (t2) == long_integer_type_node)
{
if (TYPE_UNSIGNED (t1) || TYPE_UNSIGNED (t2))
return long_unsigned_type_node;
else
return long_integer_type_node;
}
tree mv1 = TYPE_MAIN_VARIANT (t1);
tree mv2 = TYPE_MAIN_VARIANT (t2);
for (int i = NUM_FLOATN_TYPES - 1; i >= 0; i--)
if (mv1 == FLOATN_TYPE_NODE (i) || mv2 == FLOATN_TYPE_NODE (i))
return FLOATN_TYPE_NODE (i);
if (mv1 == long_double_type_node || mv2 == long_double_type_node)
return long_double_type_node;
if (mv1 == double_type_node || mv2 == double_type_node)
return double_type_node;
if (mv1 == float_type_node || mv2 == float_type_node)
return float_type_node;
for (int i = NUM_FLOATNX_TYPES - 1; i >= 0; i--)
if (mv1 == FLOATNX_TYPE_NODE (i) || mv2 == FLOATNX_TYPE_NODE (i))
return FLOATNX_TYPE_NODE (i);
if (TYPE_UNSIGNED (t1))
return t1;
else
return t2;
}

tree
common_type (tree t1, tree t2)
{
if (TREE_CODE (t1) == ENUMERAL_TYPE)
t1 = c_common_type_for_size (TYPE_PRECISION (t1), 1);
if (TREE_CODE (t2) == ENUMERAL_TYPE)
t2 = c_common_type_for_size (TYPE_PRECISION (t2), 1);
if (TREE_CODE (t1) == BOOLEAN_TYPE
&& TREE_CODE (t2) == BOOLEAN_TYPE)
return boolean_type_node;
if (TREE_CODE (t1) == BOOLEAN_TYPE)
return t2;
if (TREE_CODE (t2) == BOOLEAN_TYPE)
return t1;
return c_common_type (t1, t2);
}
int
comptypes (tree type1, tree type2)
{
const struct tagged_tu_seen_cache * tagged_tu_seen_base1 = tagged_tu_seen_base;
int val;
val = comptypes_internal (type1, type2, NULL, NULL);
free_all_tagged_tu_seen_up_to (tagged_tu_seen_base1);
return val;
}
static int
comptypes_check_enum_int (tree type1, tree type2, bool *enum_and_int_p)
{
const struct tagged_tu_seen_cache * tagged_tu_seen_base1 = tagged_tu_seen_base;
int val;
val = comptypes_internal (type1, type2, enum_and_int_p, NULL);
free_all_tagged_tu_seen_up_to (tagged_tu_seen_base1);
return val;
}
int
comptypes_check_different_types (tree type1, tree type2,
bool *different_types_p)
{
const struct tagged_tu_seen_cache * tagged_tu_seen_base1 = tagged_tu_seen_base;
int val;
val = comptypes_internal (type1, type2, NULL, different_types_p);
free_all_tagged_tu_seen_up_to (tagged_tu_seen_base1);
return val;
}

static int
comptypes_internal (const_tree type1, const_tree type2, bool *enum_and_int_p,
bool *different_types_p)
{
const_tree t1 = type1;
const_tree t2 = type2;
int attrval, val;
if (t1 == t2 || !t1 || !t2
|| TREE_CODE (t1) == ERROR_MARK || TREE_CODE (t2) == ERROR_MARK)
return 1;
if (TREE_CODE (t1) == ENUMERAL_TYPE && TREE_CODE (t2) != ENUMERAL_TYPE)
{
t1 = c_common_type_for_size (TYPE_PRECISION (t1), TYPE_UNSIGNED (t1));
if (TREE_CODE (t2) != VOID_TYPE)
{
if (enum_and_int_p != NULL)
*enum_and_int_p = true;
if (different_types_p != NULL)
*different_types_p = true;
}
}
else if (TREE_CODE (t2) == ENUMERAL_TYPE && TREE_CODE (t1) != ENUMERAL_TYPE)
{
t2 = c_common_type_for_size (TYPE_PRECISION (t2), TYPE_UNSIGNED (t2));
if (TREE_CODE (t1) != VOID_TYPE)
{
if (enum_and_int_p != NULL)
*enum_and_int_p = true;
if (different_types_p != NULL)
*different_types_p = true;
}
}
if (t1 == t2)
return 1;
if (TREE_CODE (t1) != TREE_CODE (t2))
return 0;
if (TYPE_QUALS (t1) != TYPE_QUALS (t2))
return 0;
if (TREE_CODE (t1) != ARRAY_TYPE
&& TYPE_MAIN_VARIANT (t1) == TYPE_MAIN_VARIANT (t2))
return 1;
if (!(attrval = comp_type_attributes (t1, t2)))
return 0;
val = 0;
switch (TREE_CODE (t1))
{
case INTEGER_TYPE:
case FIXED_POINT_TYPE:
case REAL_TYPE:
return TYPE_CANONICAL (t1) == TYPE_CANONICAL (t2);
case POINTER_TYPE:
if (TYPE_MODE (t1) != TYPE_MODE (t2))
break;
val = (TREE_TYPE (t1) == TREE_TYPE (t2)
? 1 : comptypes_internal (TREE_TYPE (t1), TREE_TYPE (t2),
enum_and_int_p, different_types_p));
break;
case FUNCTION_TYPE:
val = function_types_compatible_p (t1, t2, enum_and_int_p,
different_types_p);
break;
case ARRAY_TYPE:
{
tree d1 = TYPE_DOMAIN (t1);
tree d2 = TYPE_DOMAIN (t2);
bool d1_variable, d2_variable;
bool d1_zero, d2_zero;
val = 1;
if (TREE_TYPE (t1) != TREE_TYPE (t2)
&& (val = comptypes_internal (TREE_TYPE (t1), TREE_TYPE (t2),
enum_and_int_p,
different_types_p)) == 0)
return 0;
if (different_types_p != NULL
&& (d1 == NULL_TREE) != (d2 == NULL_TREE))
*different_types_p = true;
if (d1 == NULL_TREE || d2 == NULL_TREE || d1 == d2)
break;
d1_zero = !TYPE_MAX_VALUE (d1);
d2_zero = !TYPE_MAX_VALUE (d2);
d1_variable = (!d1_zero
&& (TREE_CODE (TYPE_MIN_VALUE (d1)) != INTEGER_CST
|| TREE_CODE (TYPE_MAX_VALUE (d1)) != INTEGER_CST));
d2_variable = (!d2_zero
&& (TREE_CODE (TYPE_MIN_VALUE (d2)) != INTEGER_CST
|| TREE_CODE (TYPE_MAX_VALUE (d2)) != INTEGER_CST));
d1_variable = d1_variable || (d1_zero && c_vla_type_p (t1));
d2_variable = d2_variable || (d2_zero && c_vla_type_p (t2));
if (different_types_p != NULL
&& d1_variable != d2_variable)
*different_types_p = true;
if (d1_variable || d2_variable)
break;
if (d1_zero && d2_zero)
break;
if (d1_zero || d2_zero
|| !tree_int_cst_equal (TYPE_MIN_VALUE (d1), TYPE_MIN_VALUE (d2))
|| !tree_int_cst_equal (TYPE_MAX_VALUE (d1), TYPE_MAX_VALUE (d2)))
val = 0;
break;
}
case ENUMERAL_TYPE:
case RECORD_TYPE:
case UNION_TYPE:
if (val != 1 && !same_translation_unit_p (t1, t2))
{
tree a1 = TYPE_ATTRIBUTES (t1);
tree a2 = TYPE_ATTRIBUTES (t2);
if (! attribute_list_contained (a1, a2)
&& ! attribute_list_contained (a2, a1))
break;
if (attrval != 2)
return tagged_types_tu_compatible_p (t1, t2, enum_and_int_p,
different_types_p);
val = tagged_types_tu_compatible_p (t1, t2, enum_and_int_p,
different_types_p);
}
break;
case VECTOR_TYPE:
val = (known_eq (TYPE_VECTOR_SUBPARTS (t1), TYPE_VECTOR_SUBPARTS (t2))
&& comptypes_internal (TREE_TYPE (t1), TREE_TYPE (t2),
enum_and_int_p, different_types_p));
break;
default:
break;
}
return attrval == 2 && val == 1 ? 2 : val;
}
static int
comp_target_types (location_t location, tree ttl, tree ttr)
{
int val;
int val_ped;
tree mvl = TREE_TYPE (ttl);
tree mvr = TREE_TYPE (ttr);
addr_space_t asl = TYPE_ADDR_SPACE (mvl);
addr_space_t asr = TYPE_ADDR_SPACE (mvr);
addr_space_t as_common;
bool enum_and_int_p;
if (!addr_space_superset (asl, asr, &as_common))
return 0;
val_ped = 1;
if (TREE_CODE (mvl) == ARRAY_TYPE
&& TREE_CODE (mvr) == ARRAY_TYPE)
val_ped = comptypes (mvl, mvr);
mvl = (TYPE_ATOMIC (strip_array_types (mvl))
? c_build_qualified_type (TYPE_MAIN_VARIANT (mvl), TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (mvl));
mvr = (TYPE_ATOMIC (strip_array_types (mvr))
? c_build_qualified_type (TYPE_MAIN_VARIANT (mvr), TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (mvr));
enum_and_int_p = false;
val = comptypes_check_enum_int (mvl, mvr, &enum_and_int_p);
if (val == 1 && val_ped != 1)
pedwarn (location, OPT_Wpedantic, "pointers to arrays with different qualifiers "
"are incompatible in ISO C");
if (val == 2)
pedwarn (location, OPT_Wpedantic, "types are not quite compatible");
if (val == 1 && enum_and_int_p && warn_cxx_compat)
warning_at (location, OPT_Wc___compat,
"pointer target types incompatible in C++");
return val;
}

bool
same_translation_unit_p (const_tree t1, const_tree t2)
{
while (t1 && TREE_CODE (t1) != TRANSLATION_UNIT_DECL)
switch (TREE_CODE_CLASS (TREE_CODE (t1)))
{
case tcc_declaration:
t1 = DECL_CONTEXT (t1); break;
case tcc_type:
t1 = TYPE_CONTEXT (t1); break;
case tcc_exceptional:
t1 = BLOCK_SUPERCONTEXT (t1); break;  
default: gcc_unreachable ();
}
while (t2 && TREE_CODE (t2) != TRANSLATION_UNIT_DECL)
switch (TREE_CODE_CLASS (TREE_CODE (t2)))
{
case tcc_declaration:
t2 = DECL_CONTEXT (t2); break;
case tcc_type:
t2 = TYPE_CONTEXT (t2); break;
case tcc_exceptional:
t2 = BLOCK_SUPERCONTEXT (t2); break;  
default: gcc_unreachable ();
}
return t1 == t2;
}
static struct tagged_tu_seen_cache *
alloc_tagged_tu_seen_cache (const_tree t1, const_tree t2)
{
struct tagged_tu_seen_cache *tu = XNEW (struct tagged_tu_seen_cache);
tu->next = tagged_tu_seen_base;
tu->t1 = t1;
tu->t2 = t2;
tagged_tu_seen_base = tu;
tu->val = 1;
return tu;
}
static void
free_all_tagged_tu_seen_up_to (const struct tagged_tu_seen_cache *tu_til)
{
const struct tagged_tu_seen_cache *tu = tagged_tu_seen_base;
while (tu != tu_til)
{
const struct tagged_tu_seen_cache *const tu1
= (const struct tagged_tu_seen_cache *) tu;
tu = tu1->next;
free (CONST_CAST (struct tagged_tu_seen_cache *, tu1));
}
tagged_tu_seen_base = tu_til;
}
static int
tagged_types_tu_compatible_p (const_tree t1, const_tree t2,
bool *enum_and_int_p, bool *different_types_p)
{
tree s1, s2;
bool needs_warning = false;
while (TYPE_NAME (t1)
&& TREE_CODE (TYPE_NAME (t1)) == TYPE_DECL
&& DECL_ORIGINAL_TYPE (TYPE_NAME (t1)))
t1 = DECL_ORIGINAL_TYPE (TYPE_NAME (t1));
while (TYPE_NAME (t2)
&& TREE_CODE (TYPE_NAME (t2)) == TYPE_DECL
&& DECL_ORIGINAL_TYPE (TYPE_NAME (t2)))
t2 = DECL_ORIGINAL_TYPE (TYPE_NAME (t2));
if (flag_isoc99 && TYPE_NAME (t1) != TYPE_NAME (t2))
return 0;
if (TYPE_SIZE (t1) == NULL
|| TYPE_SIZE (t2) == NULL)
return 1;
{
const struct tagged_tu_seen_cache * tts_i;
for (tts_i = tagged_tu_seen_base; tts_i != NULL; tts_i = tts_i->next)
if (tts_i->t1 == t1 && tts_i->t2 == t2)
return tts_i->val;
}
switch (TREE_CODE (t1))
{
case ENUMERAL_TYPE:
{
struct tagged_tu_seen_cache *tu = alloc_tagged_tu_seen_cache (t1, t2);
tree tv1 = TYPE_VALUES (t1);
tree tv2 = TYPE_VALUES (t2);
if (tv1 == tv2)
{
return 1;
}
for (;tv1 && tv2; tv1 = TREE_CHAIN (tv1), tv2 = TREE_CHAIN (tv2))
{
if (TREE_PURPOSE (tv1) != TREE_PURPOSE (tv2))
break;
if (simple_cst_equal (TREE_VALUE (tv1), TREE_VALUE (tv2)) != 1)
{
tu->val = 0;
return 0;
}
}
if (tv1 == NULL_TREE && tv2 == NULL_TREE)
{
return 1;
}
if (tv1 == NULL_TREE || tv2 == NULL_TREE)
{
tu->val = 0;
return 0;
}
if (list_length (TYPE_VALUES (t1)) != list_length (TYPE_VALUES (t2)))
{
tu->val = 0;
return 0;
}
for (s1 = TYPE_VALUES (t1); s1; s1 = TREE_CHAIN (s1))
{
s2 = purpose_member (TREE_PURPOSE (s1), TYPE_VALUES (t2));
if (s2 == NULL
|| simple_cst_equal (TREE_VALUE (s1), TREE_VALUE (s2)) != 1)
{
tu->val = 0;
return 0;
}
}
return 1;
}
case UNION_TYPE:
{
struct tagged_tu_seen_cache *tu = alloc_tagged_tu_seen_cache (t1, t2);
if (list_length (TYPE_FIELDS (t1)) != list_length (TYPE_FIELDS (t2)))
{
tu->val = 0;
return 0;
}
for (s1 = TYPE_FIELDS (t1), s2 = TYPE_FIELDS (t2); s1 && s2;
s1 = DECL_CHAIN (s1), s2 = DECL_CHAIN (s2))
{
int result;
if (DECL_NAME (s1) != DECL_NAME (s2))
break;
result = comptypes_internal (TREE_TYPE (s1), TREE_TYPE (s2),
enum_and_int_p, different_types_p);
if (result != 1 && !DECL_NAME (s1))
break;
if (result == 0)
{
tu->val = 0;
return 0;
}
if (result == 2)
needs_warning = true;
if (TREE_CODE (s1) == FIELD_DECL
&& simple_cst_equal (DECL_FIELD_BIT_OFFSET (s1),
DECL_FIELD_BIT_OFFSET (s2)) != 1)
{
tu->val = 0;
return 0;
}
}
if (!s1 && !s2)
{
tu->val = needs_warning ? 2 : 1;
return tu->val;
}
for (s1 = TYPE_FIELDS (t1); s1; s1 = DECL_CHAIN (s1))
{
bool ok = false;
for (s2 = TYPE_FIELDS (t2); s2; s2 = DECL_CHAIN (s2))
if (DECL_NAME (s1) == DECL_NAME (s2))
{
int result;
result = comptypes_internal (TREE_TYPE (s1), TREE_TYPE (s2),
enum_and_int_p,
different_types_p);
if (result != 1 && !DECL_NAME (s1))
continue;
if (result == 0)
{
tu->val = 0;
return 0;
}
if (result == 2)
needs_warning = true;
if (TREE_CODE (s1) == FIELD_DECL
&& simple_cst_equal (DECL_FIELD_BIT_OFFSET (s1),
DECL_FIELD_BIT_OFFSET (s2)) != 1)
break;
ok = true;
break;
}
if (!ok)
{
tu->val = 0;
return 0;
}
}
tu->val = needs_warning ? 2 : 10;
return tu->val;
}
case RECORD_TYPE:
{
struct tagged_tu_seen_cache *tu = alloc_tagged_tu_seen_cache (t1, t2);
for (s1 = TYPE_FIELDS (t1), s2 = TYPE_FIELDS (t2);
s1 && s2;
s1 = DECL_CHAIN (s1), s2 = DECL_CHAIN (s2))
{
int result;
if (TREE_CODE (s1) != TREE_CODE (s2)
|| DECL_NAME (s1) != DECL_NAME (s2))
break;
result = comptypes_internal (TREE_TYPE (s1), TREE_TYPE (s2),
enum_and_int_p, different_types_p);
if (result == 0)
break;
if (result == 2)
needs_warning = true;
if (TREE_CODE (s1) == FIELD_DECL
&& simple_cst_equal (DECL_FIELD_BIT_OFFSET (s1),
DECL_FIELD_BIT_OFFSET (s2)) != 1)
break;
}
if (s1 && s2)
tu->val = 0;
else
tu->val = needs_warning ? 2 : 1;
return tu->val;
}
default:
gcc_unreachable ();
}
}
static int
function_types_compatible_p (const_tree f1, const_tree f2,
bool *enum_and_int_p, bool *different_types_p)
{
tree args1, args2;
int val = 1;
int val1;
tree ret1, ret2;
ret1 = TREE_TYPE (f1);
ret2 = TREE_TYPE (f2);
if (TYPE_VOLATILE (ret1) != TYPE_VOLATILE (ret2))
pedwarn (input_location, 0, "function return types not compatible due to %<volatile%>");
if (TYPE_VOLATILE (ret1))
ret1 = build_qualified_type (TYPE_MAIN_VARIANT (ret1),
TYPE_QUALS (ret1) & ~TYPE_QUAL_VOLATILE);
if (TYPE_VOLATILE (ret2))
ret2 = build_qualified_type (TYPE_MAIN_VARIANT (ret2),
TYPE_QUALS (ret2) & ~TYPE_QUAL_VOLATILE);
val = comptypes_internal (ret1, ret2, enum_and_int_p, different_types_p);
if (val == 0)
return 0;
args1 = TYPE_ARG_TYPES (f1);
args2 = TYPE_ARG_TYPES (f2);
if (different_types_p != NULL
&& (args1 == NULL_TREE) != (args2 == NULL_TREE))
*different_types_p = true;
if (args1 == NULL_TREE)
{
if (!self_promoting_args_p (args2))
return 0;
if (TYPE_ACTUAL_ARG_TYPES (f1)
&& type_lists_compatible_p (args2, TYPE_ACTUAL_ARG_TYPES (f1),
enum_and_int_p, different_types_p) != 1)
val = 2;
return val;
}
if (args2 == NULL_TREE)
{
if (!self_promoting_args_p (args1))
return 0;
if (TYPE_ACTUAL_ARG_TYPES (f2)
&& type_lists_compatible_p (args1, TYPE_ACTUAL_ARG_TYPES (f2),
enum_and_int_p, different_types_p) != 1)
val = 2;
return val;
}
val1 = type_lists_compatible_p (args1, args2, enum_and_int_p,
different_types_p);
return val1 != 1 ? val1 : val;
}
static int
type_lists_compatible_p (const_tree args1, const_tree args2,
bool *enum_and_int_p, bool *different_types_p)
{
int val = 1;
int newval = 0;
while (1)
{
tree a1, mv1, a2, mv2;
if (args1 == NULL_TREE && args2 == NULL_TREE)
return val;
if (args1 == NULL_TREE || args2 == NULL_TREE)
return 0;
mv1 = a1 = TREE_VALUE (args1);
mv2 = a2 = TREE_VALUE (args2);
if (mv1 && mv1 != error_mark_node && TREE_CODE (mv1) != ARRAY_TYPE)
mv1 = (TYPE_ATOMIC (mv1)
? c_build_qualified_type (TYPE_MAIN_VARIANT (mv1),
TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (mv1));
if (mv2 && mv2 != error_mark_node && TREE_CODE (mv2) != ARRAY_TYPE)
mv2 = (TYPE_ATOMIC (mv2)
? c_build_qualified_type (TYPE_MAIN_VARIANT (mv2),
TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (mv2));
if (different_types_p != NULL
&& (a1 == NULL_TREE) != (a2 == NULL_TREE))
*different_types_p = true;
if (a1 == NULL_TREE)
{
if (c_type_promotes_to (a2) != a2)
return 0;
}
else if (a2 == NULL_TREE)
{
if (c_type_promotes_to (a1) != a1)
return 0;
}
else if (TREE_CODE (a1) == ERROR_MARK
|| TREE_CODE (a2) == ERROR_MARK)
;
else if (!(newval = comptypes_internal (mv1, mv2, enum_and_int_p,
different_types_p)))
{
if (different_types_p != NULL)
*different_types_p = true;
if (TREE_CODE (a1) == UNION_TYPE
&& (TYPE_NAME (a1) == NULL_TREE
|| TYPE_TRANSPARENT_AGGR (a1))
&& TREE_CODE (TYPE_SIZE (a1)) == INTEGER_CST
&& tree_int_cst_equal (TYPE_SIZE (a1),
TYPE_SIZE (a2)))
{
tree memb;
for (memb = TYPE_FIELDS (a1);
memb; memb = DECL_CHAIN (memb))
{
tree mv3 = TREE_TYPE (memb);
if (mv3 && mv3 != error_mark_node
&& TREE_CODE (mv3) != ARRAY_TYPE)
mv3 = (TYPE_ATOMIC (mv3)
? c_build_qualified_type (TYPE_MAIN_VARIANT (mv3),
TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (mv3));
if (comptypes_internal (mv3, mv2, enum_and_int_p,
different_types_p))
break;
}
if (memb == NULL_TREE)
return 0;
}
else if (TREE_CODE (a2) == UNION_TYPE
&& (TYPE_NAME (a2) == NULL_TREE
|| TYPE_TRANSPARENT_AGGR (a2))
&& TREE_CODE (TYPE_SIZE (a2)) == INTEGER_CST
&& tree_int_cst_equal (TYPE_SIZE (a2),
TYPE_SIZE (a1)))
{
tree memb;
for (memb = TYPE_FIELDS (a2);
memb; memb = DECL_CHAIN (memb))
{
tree mv3 = TREE_TYPE (memb);
if (mv3 && mv3 != error_mark_node
&& TREE_CODE (mv3) != ARRAY_TYPE)
mv3 = (TYPE_ATOMIC (mv3)
? c_build_qualified_type (TYPE_MAIN_VARIANT (mv3),
TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (mv3));
if (comptypes_internal (mv3, mv1, enum_and_int_p,
different_types_p))
break;
}
if (memb == NULL_TREE)
return 0;
}
else
return 0;
}
if (newval > val)
val = newval;
args1 = TREE_CHAIN (args1);
args2 = TREE_CHAIN (args2);
}
}

static tree
c_size_in_bytes (const_tree type)
{
enum tree_code code = TREE_CODE (type);
if (code == FUNCTION_TYPE || code == VOID_TYPE || code == ERROR_MARK
|| !COMPLETE_TYPE_P (type))
return size_one_node;
return size_binop_loc (input_location, CEIL_DIV_EXPR, TYPE_SIZE_UNIT (type),
size_int (TYPE_PRECISION (char_type_node)
/ BITS_PER_UNIT));
}

tree
decl_constant_value_1 (tree decl, bool in_init)
{
if (
TREE_CODE (decl) != PARM_DECL
&& !TREE_THIS_VOLATILE (decl)
&& TREE_READONLY (decl)
&& DECL_INITIAL (decl) != NULL_TREE
&& !error_operand_p (DECL_INITIAL (decl))
&& TREE_CONSTANT (DECL_INITIAL (decl))
&& (in_init || TREE_CODE (DECL_INITIAL (decl)) != CONSTRUCTOR))
return DECL_INITIAL (decl);
return decl;
}
tree
decl_constant_value (tree decl)
{
return current_function_decl ? decl_constant_value_1 (decl, false) : decl;
}
static tree
array_to_pointer_conversion (location_t loc, tree exp)
{
tree orig_exp = exp;
tree type = TREE_TYPE (exp);
tree adr;
tree restype = TREE_TYPE (type);
tree ptrtype;
gcc_assert (TREE_CODE (type) == ARRAY_TYPE);
STRIP_TYPE_NOPS (exp);
if (TREE_NO_WARNING (orig_exp))
TREE_NO_WARNING (exp) = 1;
ptrtype = build_pointer_type (restype);
if (INDIRECT_REF_P (exp))
return convert (ptrtype, TREE_OPERAND (exp, 0));
if (warn_cxx_compat && TREE_CODE (exp) == COMPOUND_LITERAL_EXPR)
{
tree decl = TREE_OPERAND (TREE_OPERAND (exp, 0), 0);
if (!TREE_READONLY (decl) && !TREE_STATIC (decl))
warning_at (DECL_SOURCE_LOCATION (decl), OPT_Wc___compat,
"converting an array compound literal to a pointer "
"is ill-formed in C++");
}
adr = build_unary_op (loc, ADDR_EXPR, exp, true);
return convert (ptrtype, adr);
}
static tree
function_to_pointer_conversion (location_t loc, tree exp)
{
tree orig_exp = exp;
gcc_assert (TREE_CODE (TREE_TYPE (exp)) == FUNCTION_TYPE);
STRIP_TYPE_NOPS (exp);
if (TREE_NO_WARNING (orig_exp))
TREE_NO_WARNING (exp) = 1;
return build_unary_op (loc, ADDR_EXPR, exp, false);
}
void
mark_exp_read (tree exp)
{
switch (TREE_CODE (exp))
{
case VAR_DECL:
case PARM_DECL:
DECL_READ_P (exp) = 1;
break;
case ARRAY_REF:
case COMPONENT_REF:
case MODIFY_EXPR:
case REALPART_EXPR:
case IMAGPART_EXPR:
CASE_CONVERT:
case ADDR_EXPR:
case VIEW_CONVERT_EXPR:
mark_exp_read (TREE_OPERAND (exp, 0));
break;
case COMPOUND_EXPR:
case C_MAYBE_CONST_EXPR:
mark_exp_read (TREE_OPERAND (exp, 1));
break;
default:
break;
}
}
struct c_expr
default_function_array_conversion (location_t loc, struct c_expr exp)
{
tree orig_exp = exp.value;
tree type = TREE_TYPE (exp.value);
enum tree_code code = TREE_CODE (type);
switch (code)
{
case ARRAY_TYPE:
{
bool not_lvalue = false;
bool lvalue_array_p;
while ((TREE_CODE (exp.value) == NON_LVALUE_EXPR
|| CONVERT_EXPR_P (exp.value))
&& TREE_TYPE (TREE_OPERAND (exp.value, 0)) == type)
{
if (TREE_CODE (exp.value) == NON_LVALUE_EXPR)
not_lvalue = true;
exp.value = TREE_OPERAND (exp.value, 0);
}
if (TREE_NO_WARNING (orig_exp))
TREE_NO_WARNING (exp.value) = 1;
lvalue_array_p = !not_lvalue && lvalue_p (exp.value);
if (!flag_isoc99 && !lvalue_array_p)
{
return exp;
}
exp.value = array_to_pointer_conversion (loc, exp.value);
}
break;
case FUNCTION_TYPE:
exp.value = function_to_pointer_conversion (loc, exp.value);
break;
default:
break;
}
return exp;
}
struct c_expr
default_function_array_read_conversion (location_t loc, struct c_expr exp)
{
mark_exp_read (exp.value);
return default_function_array_conversion (loc, exp);
}
static bool
really_atomic_lvalue (tree expr)
{
if (error_operand_p (expr))
return false;
if (!TYPE_ATOMIC (TREE_TYPE (expr)))
return false;
if (!lvalue_p (expr))
return false;
while (handled_component_p (expr))
{
if (TREE_CODE (expr) == COMPONENT_REF
&& DECL_C_BIT_FIELD (TREE_OPERAND (expr, 1)))
return false;
expr = TREE_OPERAND (expr, 0);
}
if (DECL_P (expr) && C_DECL_REGISTER (expr))
return false;
return true;
}
struct c_expr
convert_lvalue_to_rvalue (location_t loc, struct c_expr exp,
bool convert_p, bool read_p)
{
if (read_p)
mark_exp_read (exp.value);
if (convert_p)
exp = default_function_array_conversion (loc, exp);
if (really_atomic_lvalue (exp.value))
{
vec<tree, va_gc> *params;
tree nonatomic_type, tmp, tmp_addr, fndecl, func_call;
tree expr_type = TREE_TYPE (exp.value);
tree expr_addr = build_unary_op (loc, ADDR_EXPR, exp.value, false);
tree seq_cst = build_int_cst (integer_type_node, MEMMODEL_SEQ_CST);
gcc_assert (TYPE_ATOMIC (expr_type));
vec_alloc (params, 4);
nonatomic_type = build_qualified_type (expr_type, TYPE_UNQUALIFIED);
tmp = create_tmp_var_raw (nonatomic_type);
tmp_addr = build_unary_op (loc, ADDR_EXPR, tmp, false);
TREE_ADDRESSABLE (tmp) = 1;
TREE_NO_WARNING (tmp) = 1;
fndecl = builtin_decl_explicit (BUILT_IN_ATOMIC_LOAD);
params->quick_push (expr_addr);
params->quick_push (tmp_addr);
params->quick_push (seq_cst);
func_call = c_build_function_call_vec (loc, vNULL, fndecl, params, NULL);
mark_exp_read (exp.value);
exp.value = build4 (TARGET_EXPR, nonatomic_type, tmp, func_call,
NULL_TREE, NULL_TREE);
}
return exp;
}
tree
perform_integral_promotions (tree exp)
{
tree type = TREE_TYPE (exp);
enum tree_code code = TREE_CODE (type);
gcc_assert (INTEGRAL_TYPE_P (type));
if (code == ENUMERAL_TYPE)
{
type = c_common_type_for_size (MAX (TYPE_PRECISION (type),
TYPE_PRECISION (integer_type_node)),
((TYPE_PRECISION (type)
>= TYPE_PRECISION (integer_type_node))
&& TYPE_UNSIGNED (type)));
return convert (type, exp);
}
if (TREE_CODE (exp) == COMPONENT_REF
&& DECL_C_BIT_FIELD (TREE_OPERAND (exp, 1))
&& compare_tree_int (DECL_SIZE (TREE_OPERAND (exp, 1)),
TYPE_PRECISION (integer_type_node)) < 0)
return convert (integer_type_node, exp);
if (c_promoting_integer_type_p (type))
{
if (TYPE_UNSIGNED (type)
&& TYPE_PRECISION (type) == TYPE_PRECISION (integer_type_node))
return convert (unsigned_type_node, exp);
return convert (integer_type_node, exp);
}
return exp;
}
tree
default_conversion (tree exp)
{
tree orig_exp;
tree type = TREE_TYPE (exp);
enum tree_code code = TREE_CODE (type);
tree promoted_type;
mark_exp_read (exp);
gcc_assert (code != FUNCTION_TYPE);
if (code == ARRAY_TYPE)
return exp;
if (TREE_CODE (exp) == CONST_DECL)
exp = DECL_INITIAL (exp);
orig_exp = exp;
STRIP_TYPE_NOPS (exp);
if (TREE_NO_WARNING (orig_exp))
TREE_NO_WARNING (exp) = 1;
if (code == VOID_TYPE)
{
error_at (EXPR_LOC_OR_LOC (exp, input_location),
"void value not ignored as it ought to be");
return error_mark_node;
}
exp = require_complete_type (EXPR_LOC_OR_LOC (exp, input_location), exp);
if (exp == error_mark_node)
return error_mark_node;
promoted_type = targetm.promoted_type (type);
if (promoted_type)
return convert (promoted_type, exp);
if (INTEGRAL_TYPE_P (type))
return perform_integral_promotions (exp);
return exp;
}

static tree
lookup_field (tree type, tree component)
{
tree field;
if (TYPE_LANG_SPECIFIC (type) && TYPE_LANG_SPECIFIC (type)->s)
{
int bot, top, half;
tree *field_array = &TYPE_LANG_SPECIFIC (type)->s->elts[0];
field = TYPE_FIELDS (type);
bot = 0;
top = TYPE_LANG_SPECIFIC (type)->s->len;
while (top - bot > 1)
{
half = (top - bot + 1) >> 1;
field = field_array[bot+half];
if (DECL_NAME (field) == NULL_TREE)
{
while (DECL_NAME (field_array[bot]) == NULL_TREE)
{
field = field_array[bot++];
if (RECORD_OR_UNION_TYPE_P (TREE_TYPE (field)))
{
tree anon = lookup_field (TREE_TYPE (field), component);
if (anon)
return tree_cons (NULL_TREE, field, anon);
if (flag_plan9_extensions
&& TYPE_NAME (TREE_TYPE (field)) != NULL_TREE
&& (TREE_CODE (TYPE_NAME (TREE_TYPE (field)))
== TYPE_DECL)
&& (DECL_NAME (TYPE_NAME (TREE_TYPE (field)))
== component))
break;
}
}
if (bot > top)
return NULL_TREE;
continue;
}
if (DECL_NAME (field) == component)
break;
if (DECL_NAME (field) < component)
bot += half;
else
top = bot + half;
}
if (DECL_NAME (field_array[bot]) == component)
field = field_array[bot];
else if (DECL_NAME (field) != component)
return NULL_TREE;
}
else
{
for (field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
{
if (DECL_NAME (field) == NULL_TREE
&& RECORD_OR_UNION_TYPE_P (TREE_TYPE (field)))
{
tree anon = lookup_field (TREE_TYPE (field), component);
if (anon)
return tree_cons (NULL_TREE, field, anon);
if (flag_plan9_extensions
&& TYPE_NAME (TREE_TYPE (field)) != NULL_TREE
&& TREE_CODE (TYPE_NAME (TREE_TYPE (field))) == TYPE_DECL
&& (DECL_NAME (TYPE_NAME (TREE_TYPE (field)))
== component))
break;
}
if (DECL_NAME (field) == component)
break;
}
if (field == NULL_TREE)
return NULL_TREE;
}
return tree_cons (NULL_TREE, field, NULL_TREE);
}
static void
lookup_field_fuzzy_find_candidates (tree type, tree component,
vec<tree> *candidates)
{
tree field;
for (field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
{
if (DECL_NAME (field) == NULL_TREE
&& RECORD_OR_UNION_TYPE_P (TREE_TYPE (field)))
lookup_field_fuzzy_find_candidates (TREE_TYPE (field), component,
candidates);
if (DECL_NAME (field))
candidates->safe_push (DECL_NAME (field));
}
}
static tree
lookup_field_fuzzy (tree type, tree component)
{
gcc_assert (TREE_CODE (component) == IDENTIFIER_NODE);
auto_vec <tree> candidates;
lookup_field_fuzzy_find_candidates (type, component,
&candidates);
return find_closest_identifier (component, &candidates);
}
static bool
should_suggest_deref_p (tree datum_type)
{
if (c_dialect_objc ())
return false;
if (TREE_CODE (datum_type) != POINTER_TYPE)
return false;
tree underlying_type = TREE_TYPE (datum_type);
enum tree_code code = TREE_CODE (underlying_type);
if (code == RECORD_TYPE || code == UNION_TYPE)
return true;
else
return false;
}
tree
build_component_ref (location_t loc, tree datum, tree component,
location_t component_loc)
{
tree type = TREE_TYPE (datum);
enum tree_code code = TREE_CODE (type);
tree field = NULL;
tree ref;
bool datum_lvalue = lvalue_p (datum);
if (!objc_is_public (datum, component))
return error_mark_node;
if (c_dialect_objc ()
&& (ref = objc_maybe_build_component_ref (datum, component)))
return ref;
if (code == RECORD_TYPE || code == UNION_TYPE)
{
if (!COMPLETE_TYPE_P (type))
{
c_incomplete_type_error (loc, NULL_TREE, type);
return error_mark_node;
}
field = lookup_field (type, component);
if (!field)
{
tree guessed_id = lookup_field_fuzzy (type, component);
if (guessed_id)
{
location_t reported_loc
= (component_loc != UNKNOWN_LOCATION) ? component_loc : loc;
gcc_rich_location rich_loc (reported_loc);
if (component_loc != UNKNOWN_LOCATION)
rich_loc.add_fixit_misspelled_id (component_loc, guessed_id);
error_at (&rich_loc,
"%qT has no member named %qE; did you mean %qE?",
type, component, guessed_id);
}
else
error_at (loc, "%qT has no member named %qE", type, component);
return error_mark_node;
}
if (TYPE_ATOMIC (type) && c_inhibit_evaluation_warnings == 0)
{
if (code == RECORD_TYPE)
warning_at (loc, 0, "accessing a member %qE of an atomic "
"structure %qE", component, datum);
else
warning_at (loc, 0, "accessing a member %qE of an atomic "
"union %qE", component, datum);
}
do
{
tree subdatum = TREE_VALUE (field);
int quals;
tree subtype;
bool use_datum_quals;
if (TREE_TYPE (subdatum) == error_mark_node)
return error_mark_node;
use_datum_quals = (datum_lvalue
|| TREE_CODE (TREE_TYPE (subdatum)) != ARRAY_TYPE);
quals = TYPE_QUALS (strip_array_types (TREE_TYPE (subdatum)));
if (use_datum_quals)
quals |= TYPE_QUALS (TREE_TYPE (datum));
subtype = c_build_qualified_type (TREE_TYPE (subdatum), quals);
ref = build3 (COMPONENT_REF, subtype, datum, subdatum,
NULL_TREE);
SET_EXPR_LOCATION (ref, loc);
if (TREE_READONLY (subdatum)
|| (use_datum_quals && TREE_READONLY (datum)))
TREE_READONLY (ref) = 1;
if (TREE_THIS_VOLATILE (subdatum)
|| (use_datum_quals && TREE_THIS_VOLATILE (datum)))
TREE_THIS_VOLATILE (ref) = 1;
if (TREE_DEPRECATED (subdatum))
warn_deprecated_use (subdatum, NULL_TREE);
datum = ref;
field = TREE_CHAIN (field);
}
while (field);
return ref;
}
else if (should_suggest_deref_p (type))
{
rich_location richloc (line_table, loc);
richloc.add_fixit_replace ("->");
error_at (&richloc,
"%qE is a pointer; did you mean to use %<->%>?",
datum);
return error_mark_node;
}
else if (code != ERROR_MARK)
error_at (loc,
"request for member %qE in something not a structure or union",
component);
return error_mark_node;
}

tree
build_indirect_ref (location_t loc, tree ptr, ref_operator errstring)
{
tree pointer = default_conversion (ptr);
tree type = TREE_TYPE (pointer);
tree ref;
if (TREE_CODE (type) == POINTER_TYPE)
{
if (CONVERT_EXPR_P (pointer)
|| TREE_CODE (pointer) == VIEW_CONVERT_EXPR)
{
if (warn_strict_aliasing > 2)
if (strict_aliasing_warning (EXPR_LOCATION (pointer),
type, TREE_OPERAND (pointer, 0)))
TREE_NO_WARNING (pointer) = 1;
}
if (TREE_CODE (pointer) == ADDR_EXPR
&& (TREE_TYPE (TREE_OPERAND (pointer, 0))
== TREE_TYPE (type)))
{
ref = TREE_OPERAND (pointer, 0);
protected_set_expr_location (ref, loc);
return ref;
}
else
{
tree t = TREE_TYPE (type);
ref = build1 (INDIRECT_REF, t, pointer);
if (!COMPLETE_OR_VOID_TYPE_P (t) && TREE_CODE (t) != ARRAY_TYPE)
{
if (!C_TYPE_ERROR_REPORTED (TREE_TYPE (ptr)))
{
error_at (loc, "dereferencing pointer to incomplete type "
"%qT", t);
C_TYPE_ERROR_REPORTED (TREE_TYPE (ptr)) = 1;
}
return error_mark_node;
}
if (VOID_TYPE_P (t) && c_inhibit_evaluation_warnings == 0)
warning_at (loc, 0, "dereferencing %<void *%> pointer");
TREE_READONLY (ref) = TYPE_READONLY (t);
TREE_SIDE_EFFECTS (ref)
= TYPE_VOLATILE (t) || TREE_SIDE_EFFECTS (pointer);
TREE_THIS_VOLATILE (ref) = TYPE_VOLATILE (t);
protected_set_expr_location (ref, loc);
return ref;
}
}
else if (TREE_CODE (pointer) != ERROR_MARK)
invalid_indirection_error (loc, type, errstring);
return error_mark_node;
}
tree
build_array_ref (location_t loc, tree array, tree index)
{
tree ret;
bool swapped = false;
if (TREE_TYPE (array) == error_mark_node
|| TREE_TYPE (index) == error_mark_node)
return error_mark_node;
if (TREE_CODE (TREE_TYPE (array)) != ARRAY_TYPE
&& TREE_CODE (TREE_TYPE (array)) != POINTER_TYPE
&& !VECTOR_TYPE_P (TREE_TYPE (array)))
{
if (TREE_CODE (TREE_TYPE (index)) != ARRAY_TYPE
&& TREE_CODE (TREE_TYPE (index)) != POINTER_TYPE)
{
error_at (loc,
"subscripted value is neither array nor pointer nor vector");
return error_mark_node;
}
std::swap (array, index);
swapped = true;
}
if (!INTEGRAL_TYPE_P (TREE_TYPE (index)))
{
error_at (loc, "array subscript is not an integer");
return error_mark_node;
}
if (TREE_CODE (TREE_TYPE (TREE_TYPE (array))) == FUNCTION_TYPE)
{
error_at (loc, "subscripted value is pointer to function");
return error_mark_node;
}
if (!swapped)
warn_array_subscript_with_type_char (loc, index);
index = default_conversion (index);
if (index == error_mark_node)
return error_mark_node;
gcc_assert (TREE_CODE (TREE_TYPE (index)) == INTEGER_TYPE);
bool was_vector = VECTOR_TYPE_P (TREE_TYPE (array));
bool non_lvalue = convert_vector_to_array_for_subscript (loc, &array, index);
if (TREE_CODE (TREE_TYPE (array)) == ARRAY_TYPE)
{
tree rval, type;
if (TREE_CODE (index) != INTEGER_CST
|| (COMPLETE_TYPE_P (TREE_TYPE (TREE_TYPE (array)))
&& TREE_CODE (TYPE_SIZE (TREE_TYPE (TREE_TYPE (array)))) != INTEGER_CST))
{
if (!c_mark_addressable (array, true))
return error_mark_node;
}
if (TREE_CODE (index) == INTEGER_CST
&& TYPE_DOMAIN (TREE_TYPE (array))
&& !int_fits_type_p (index, TYPE_DOMAIN (TREE_TYPE (array))))
{
if (!c_mark_addressable (array))
return error_mark_node;
}
if ((pedantic || warn_c90_c99_compat)
&& ! was_vector)
{
tree foo = array;
while (TREE_CODE (foo) == COMPONENT_REF)
foo = TREE_OPERAND (foo, 0);
if (VAR_P (foo) && C_DECL_REGISTER (foo))
pedwarn (loc, OPT_Wpedantic,
"ISO C forbids subscripting %<register%> array");
else if (!lvalue_p (foo))
pedwarn_c90 (loc, OPT_Wpedantic,
"ISO C90 forbids subscripting non-lvalue "
"array");
}
type = TREE_TYPE (TREE_TYPE (array));
rval = build4 (ARRAY_REF, type, array, index, NULL_TREE, NULL_TREE);
TREE_READONLY (rval)
|= (TYPE_READONLY (TREE_TYPE (TREE_TYPE (array)))
| TREE_READONLY (array));
TREE_SIDE_EFFECTS (rval)
|= (TYPE_VOLATILE (TREE_TYPE (TREE_TYPE (array)))
| TREE_SIDE_EFFECTS (array));
TREE_THIS_VOLATILE (rval)
|= (TYPE_VOLATILE (TREE_TYPE (TREE_TYPE (array)))
| TREE_THIS_VOLATILE (array));
ret = require_complete_type (loc, rval);
protected_set_expr_location (ret, loc);
if (non_lvalue)
ret = non_lvalue_loc (loc, ret);
return ret;
}
else
{
tree ar = default_conversion (array);
if (ar == error_mark_node)
return ar;
gcc_assert (TREE_CODE (TREE_TYPE (ar)) == POINTER_TYPE);
gcc_assert (TREE_CODE (TREE_TYPE (TREE_TYPE (ar))) != FUNCTION_TYPE);
ret = build_indirect_ref (loc, build_binary_op (loc, PLUS_EXPR, ar,
index, false),
RO_ARRAY_INDEXING);
if (non_lvalue)
ret = non_lvalue_loc (loc, ret);
return ret;
}
}

tree
build_external_ref (location_t loc, tree id, bool fun, tree *type)
{
tree ref;
tree decl = lookup_name (id);
decl = objc_lookup_ivar (decl, id);
*type = NULL;
if (decl && decl != error_mark_node)
{
ref = decl;
*type = TREE_TYPE (ref);
}
else if (fun)
ref = implicitly_declare (loc, id);
else if (decl == error_mark_node)
return error_mark_node;
else
{
undeclared_variable (loc, id);
return error_mark_node;
}
if (TREE_TYPE (ref) == error_mark_node)
return error_mark_node;
if (TREE_DEPRECATED (ref))
warn_deprecated_use (ref, NULL_TREE);
if (ref != current_function_decl)
{
TREE_USED (ref) = 1;
}
if (TREE_CODE (ref) == FUNCTION_DECL && !in_alignof)
{
if (!in_sizeof && !in_typeof)
C_DECL_USED (ref) = 1;
else if (DECL_INITIAL (ref) == NULL_TREE
&& DECL_EXTERNAL (ref)
&& !TREE_PUBLIC (ref))
record_maybe_used_decl (ref);
}
if (TREE_CODE (ref) == CONST_DECL)
{
used_types_insert (TREE_TYPE (ref));
if (warn_cxx_compat
&& TREE_CODE (TREE_TYPE (ref)) == ENUMERAL_TYPE
&& C_TYPE_DEFINED_IN_STRUCT (TREE_TYPE (ref)))
{
warning_at (loc, OPT_Wc___compat,
("enum constant defined in struct or union "
"is not visible in C++"));
inform (DECL_SOURCE_LOCATION (ref), "enum constant defined here");
}
ref = DECL_INITIAL (ref);
TREE_CONSTANT (ref) = 1;
}
else if (current_function_decl != NULL_TREE
&& !DECL_FILE_SCOPE_P (current_function_decl)
&& (VAR_OR_FUNCTION_DECL_P (ref)
|| TREE_CODE (ref) == PARM_DECL))
{
tree context = decl_function_context (ref);
if (context != NULL_TREE && context != current_function_decl)
DECL_NONLOCAL (ref) = 1;
}
else if (current_function_decl != NULL_TREE
&& DECL_DECLARED_INLINE_P (current_function_decl)
&& DECL_EXTERNAL (current_function_decl)
&& VAR_OR_FUNCTION_DECL_P (ref)
&& (!VAR_P (ref) || TREE_STATIC (ref))
&& ! TREE_PUBLIC (ref)
&& DECL_CONTEXT (ref) != current_function_decl)
record_inline_static (loc, current_function_decl, ref,
csi_internal);
return ref;
}
struct maybe_used_decl
{
tree decl;
int level;
struct maybe_used_decl *next;
};
static struct maybe_used_decl *maybe_used_decls;
static void
record_maybe_used_decl (tree decl)
{
struct maybe_used_decl *t = XOBNEW (&parser_obstack, struct maybe_used_decl);
t->decl = decl;
t->level = in_sizeof + in_typeof;
t->next = maybe_used_decls;
maybe_used_decls = t;
}
void
pop_maybe_used (bool used)
{
struct maybe_used_decl *p = maybe_used_decls;
int cur_level = in_sizeof + in_typeof;
while (p && p->level > cur_level)
{
if (used)
{
if (cur_level == 0)
C_DECL_USED (p->decl) = 1;
else
p->level = cur_level;
}
p = p->next;
}
if (!used || cur_level == 0)
maybe_used_decls = p;
}
struct c_expr
c_expr_sizeof_expr (location_t loc, struct c_expr expr)
{
struct c_expr ret;
if (expr.value == error_mark_node)
{
ret.value = error_mark_node;
ret.original_code = ERROR_MARK;
ret.original_type = NULL;
pop_maybe_used (false);
}
else
{
bool expr_const_operands = true;
if (TREE_CODE (expr.value) == PARM_DECL
&& C_ARRAY_PARAMETER (expr.value))
{
if (warning_at (loc, OPT_Wsizeof_array_argument,
"%<sizeof%> on array function parameter %qE will "
"return size of %qT", expr.value,
TREE_TYPE (expr.value)))
inform (DECL_SOURCE_LOCATION (expr.value), "declared here");
}
tree folded_expr = c_fully_fold (expr.value, require_constant_value,
&expr_const_operands);
ret.value = c_sizeof (loc, TREE_TYPE (folded_expr));
c_last_sizeof_arg = expr.value;
c_last_sizeof_loc = loc;
ret.original_code = SIZEOF_EXPR;
ret.original_type = NULL;
if (c_vla_type_p (TREE_TYPE (folded_expr)))
{
ret.value = build2 (C_MAYBE_CONST_EXPR, TREE_TYPE (ret.value),
folded_expr, ret.value);
C_MAYBE_CONST_EXPR_NON_CONST (ret.value) = !expr_const_operands;
SET_EXPR_LOCATION (ret.value, loc);
}
pop_maybe_used (C_TYPE_VARIABLE_SIZE (TREE_TYPE (folded_expr)));
}
return ret;
}
struct c_expr
c_expr_sizeof_type (location_t loc, struct c_type_name *t)
{
tree type;
struct c_expr ret;
tree type_expr = NULL_TREE;
bool type_expr_const = true;
type = groktypename (t, &type_expr, &type_expr_const);
ret.value = c_sizeof (loc, type);
c_last_sizeof_arg = type;
c_last_sizeof_loc = loc;
ret.original_code = SIZEOF_EXPR;
ret.original_type = NULL;
if ((type_expr || TREE_CODE (ret.value) == INTEGER_CST)
&& c_vla_type_p (type))
{
if (!type_expr)
type_expr = integer_zero_node;
ret.value = build2 (C_MAYBE_CONST_EXPR, TREE_TYPE (ret.value),
type_expr, ret.value);
C_MAYBE_CONST_EXPR_NON_CONST (ret.value) = !type_expr_const;
}
pop_maybe_used (type != error_mark_node
? C_TYPE_VARIABLE_SIZE (type) : false);
return ret;
}
tree
build_function_call (location_t loc, tree function, tree params)
{
vec<tree, va_gc> *v;
tree ret;
vec_alloc (v, list_length (params));
for (; params; params = TREE_CHAIN (params))
v->quick_push (TREE_VALUE (params));
ret = c_build_function_call_vec (loc, vNULL, function, v, NULL);
vec_free (v);
return ret;
}
static void
inform_declaration (tree decl)
{
if (decl && (TREE_CODE (decl) != FUNCTION_DECL || !DECL_IS_BUILTIN (decl)))
inform (DECL_SOURCE_LOCATION (decl), "declared here");
}
tree
build_function_call_vec (location_t loc, vec<location_t> arg_loc,
tree function, vec<tree, va_gc> *params,
vec<tree, va_gc> *origtypes)
{
tree fntype, fundecl = NULL_TREE;
tree name = NULL_TREE, result;
tree tem;
int nargs;
tree *argarray;
STRIP_TYPE_NOPS (function);
if (TREE_CODE (function) == FUNCTION_DECL)
{
name = DECL_NAME (function);
if (flag_tm)
tm_malloc_replacement (function);
fundecl = function;
if (name && !strncmp (IDENTIFIER_POINTER (name), "__atomic_", 9))
origtypes = NULL;
}
if (TREE_CODE (TREE_TYPE (function)) == FUNCTION_TYPE)
function = function_to_pointer_conversion (loc, function);
if (params && !params->is_empty ())
function = objc_rewrite_function_call (function, (*params)[0]);
function = c_fully_fold (function, false, NULL);
fntype = TREE_TYPE (function);
if (TREE_CODE (fntype) == ERROR_MARK)
return error_mark_node;
if (!(TREE_CODE (fntype) == POINTER_TYPE
&& TREE_CODE (TREE_TYPE (fntype)) == FUNCTION_TYPE))
{
if (!flag_diagnostics_show_caret)
error_at (loc,
"called object %qE is not a function or function pointer",
function);
else if (DECL_P (function))
{
error_at (loc,
"called object %qD is not a function or function pointer",
function);
inform_declaration (function);
}
else
error_at (loc,
"called object is not a function or function pointer");
return error_mark_node;
}
if (fundecl && TREE_THIS_VOLATILE (fundecl))
current_function_returns_abnormally = 1;
fntype = TREE_TYPE (fntype);
nargs = convert_arguments (loc, arg_loc, TYPE_ARG_TYPES (fntype), params,
origtypes, function, fundecl);
if (nargs < 0)
return error_mark_node;
if (CONVERT_EXPR_P (function)
&& TREE_CODE (tem = TREE_OPERAND (function, 0)) == ADDR_EXPR
&& TREE_CODE (tem = TREE_OPERAND (tem, 0)) == FUNCTION_DECL
&& !comptypes (fntype, TREE_TYPE (tem)))
{
tree return_type = TREE_TYPE (fntype);
warning_at (loc, 0, "function called through a non-compatible type");
if (VOID_TYPE_P (return_type)
&& TYPE_QUALS (return_type) != TYPE_UNQUALIFIED)
pedwarn (loc, 0,
"function with qualified void return type called");
}
argarray = vec_safe_address (params);
if (fundecl
&& DECL_BUILT_IN (fundecl)
&& DECL_BUILT_IN_CLASS (fundecl) == BUILT_IN_NORMAL
&& !check_builtin_function_arguments (loc, arg_loc, fundecl, nargs,
argarray))
return error_mark_node;
bool warned_p = check_function_arguments (loc, fundecl, fntype,
nargs, argarray, &arg_loc);
if (name != NULL_TREE
&& !strncmp (IDENTIFIER_POINTER (name), "__builtin_", 10))
{
if (require_constant_value)
result
= fold_build_call_array_initializer_loc (loc, TREE_TYPE (fntype),
function, nargs, argarray);
else
result = fold_build_call_array_loc (loc, TREE_TYPE (fntype),
function, nargs, argarray);
if (TREE_CODE (result) == NOP_EXPR
&& TREE_CODE (TREE_OPERAND (result, 0)) == INTEGER_CST)
STRIP_TYPE_NOPS (result);
}
else
result = build_call_array_loc (loc, TREE_TYPE (fntype),
function, nargs, argarray);
if (warned_p && TREE_CODE (result) == CALL_EXPR)
TREE_NO_WARNING (result) = 1;
if (variably_modified_type_p (TREE_TYPE (fntype), NULL_TREE))
{
tree tmp = create_tmp_var_raw (TREE_TYPE (fntype));
result = build4 (TARGET_EXPR, TREE_TYPE (fntype), tmp, result,
NULL_TREE, NULL_TREE);
}
if (VOID_TYPE_P (TREE_TYPE (result)))
{
if (TYPE_QUALS (TREE_TYPE (result)) != TYPE_UNQUALIFIED)
pedwarn (loc, 0,
"function with qualified void return type called");
return result;
}
return require_complete_type (loc, result);
}
tree
c_build_function_call_vec (location_t loc, vec<location_t> arg_loc,
tree function, vec<tree, va_gc> *params,
vec<tree, va_gc> *origtypes)
{
STRIP_TYPE_NOPS (function);
if (TREE_CODE (function) == FUNCTION_DECL)
{
tree tem = resolve_overloaded_builtin (loc, function, params);
if (tem)
return tem;
}
return build_function_call_vec (loc, arg_loc, function, params, origtypes);
}

static int
convert_arguments (location_t loc, vec<location_t> arg_loc, tree typelist,
vec<tree, va_gc> *values, vec<tree, va_gc> *origtypes,
tree function, tree fundecl)
{
tree typetail, val;
unsigned int parmnum;
bool error_args = false;
const bool type_generic = fundecl
&& lookup_attribute ("type generic", TYPE_ATTRIBUTES (TREE_TYPE (fundecl)));
bool type_generic_remove_excess_precision = false;
bool type_generic_overflow_p = false;
tree selector;
if (TREE_CODE (function) == ADDR_EXPR
&& TREE_CODE (TREE_OPERAND (function, 0)) == FUNCTION_DECL)
function = TREE_OPERAND (function, 0);
selector = objc_message_selector ();
if (type_generic
&& DECL_BUILT_IN (fundecl)
&& DECL_BUILT_IN_CLASS (fundecl) == BUILT_IN_NORMAL)
{
switch (DECL_FUNCTION_CODE (fundecl))
{
case BUILT_IN_ISFINITE:
case BUILT_IN_ISINF:
case BUILT_IN_ISINF_SIGN:
case BUILT_IN_ISNAN:
case BUILT_IN_ISNORMAL:
case BUILT_IN_FPCLASSIFY:
type_generic_remove_excess_precision = true;
break;
case BUILT_IN_ADD_OVERFLOW_P:
case BUILT_IN_SUB_OVERFLOW_P:
case BUILT_IN_MUL_OVERFLOW_P:
type_generic_overflow_p = true;
break;
default:
break;
}
}
for (typetail = typelist, parmnum = 0;
values && values->iterate (parmnum, &val);
++parmnum)
{
tree type = typetail ? TREE_VALUE (typetail) : 0;
tree valtype = TREE_TYPE (val);
tree rname = function;
int argnum = parmnum + 1;
const char *invalid_func_diag;
bool excess_precision = false;
bool npc;
tree parmval;
location_t ploc
= !arg_loc.is_empty () && values->length () == arg_loc.length ()
? expansion_point_location_if_in_system_header (arg_loc[parmnum])
: input_location;
if (type == void_type_node)
{
if (selector)
error_at (loc, "too many arguments to method %qE", selector);
else
error_at (loc, "too many arguments to function %qE", function);
inform_declaration (fundecl);
return error_args ? -1 : (int) parmnum;
}
if (selector && argnum > 2)
{
rname = selector;
argnum -= 2;
}
npc = null_pointer_constant_p (val);
if (TREE_CODE (val) == EXCESS_PRECISION_EXPR
&& (type || !type_generic || !type_generic_remove_excess_precision))
{
val = TREE_OPERAND (val, 0);
excess_precision = true;
}
val = c_fully_fold (val, false, NULL);
STRIP_TYPE_NOPS (val);
val = require_complete_type (ploc, val);
bool promote_float_arg = false;
if (type == NULL_TREE
&& TREE_CODE (valtype) == REAL_TYPE
&& (TYPE_PRECISION (valtype)
<= TYPE_PRECISION (double_type_node))
&& TYPE_MAIN_VARIANT (valtype) != double_type_node
&& TYPE_MAIN_VARIANT (valtype) != long_double_type_node
&& !DECIMAL_FLOAT_MODE_P (TYPE_MODE (valtype)))
{
promote_float_arg = true;
for (int i = 0; i < NUM_FLOATN_NX_TYPES; i++)
if (TYPE_MAIN_VARIANT (valtype) == FLOATN_NX_TYPE_NODE (i))
{
promote_float_arg = false;
break;
}
}
if (type != NULL_TREE)
{
if (type == error_mark_node || !COMPLETE_TYPE_P (type))
{
error_at (ploc, "type of formal parameter %d is incomplete",
parmnum + 1);
parmval = val;
}
else
{
tree origtype;
if (warn_traditional_conversion || warn_traditional)
{
unsigned int formal_prec = TYPE_PRECISION (type);
if (INTEGRAL_TYPE_P (type)
&& TREE_CODE (valtype) == REAL_TYPE)
warning_at (ploc, OPT_Wtraditional_conversion,
"passing argument %d of %qE as integer rather "
"than floating due to prototype",
argnum, rname);
if (INTEGRAL_TYPE_P (type)
&& TREE_CODE (valtype) == COMPLEX_TYPE)
warning_at (ploc, OPT_Wtraditional_conversion,
"passing argument %d of %qE as integer rather "
"than complex due to prototype",
argnum, rname);
else if (TREE_CODE (type) == COMPLEX_TYPE
&& TREE_CODE (valtype) == REAL_TYPE)
warning_at (ploc, OPT_Wtraditional_conversion,
"passing argument %d of %qE as complex rather "
"than floating due to prototype",
argnum, rname);
else if (TREE_CODE (type) == REAL_TYPE
&& INTEGRAL_TYPE_P (valtype))
warning_at (ploc, OPT_Wtraditional_conversion,
"passing argument %d of %qE as floating rather "
"than integer due to prototype",
argnum, rname);
else if (TREE_CODE (type) == COMPLEX_TYPE
&& INTEGRAL_TYPE_P (valtype))
warning_at (ploc, OPT_Wtraditional_conversion,
"passing argument %d of %qE as complex rather "
"than integer due to prototype",
argnum, rname);
else if (TREE_CODE (type) == REAL_TYPE
&& TREE_CODE (valtype) == COMPLEX_TYPE)
warning_at (ploc, OPT_Wtraditional_conversion,
"passing argument %d of %qE as floating rather "
"than complex due to prototype",
argnum, rname);
else if (TREE_CODE (type) == REAL_TYPE
&& TREE_CODE (valtype) == REAL_TYPE)
{
if (formal_prec == TYPE_PRECISION (float_type_node)
&& type != dfloat32_type_node)
warning_at (ploc, 0,
"passing argument %d of %qE as %<float%> "
"rather than %<double%> due to prototype",
argnum, rname);
else if (type != valtype
&& (type == dfloat32_type_node
|| type == dfloat64_type_node
|| type == dfloat128_type_node
|| valtype == dfloat32_type_node
|| valtype == dfloat64_type_node
|| valtype == dfloat128_type_node)
&& (formal_prec
<= TYPE_PRECISION (valtype)
|| (type == dfloat128_type_node
&& (valtype
!= dfloat64_type_node
&& (valtype
!= dfloat32_type_node)))
|| (type == dfloat64_type_node
&& (valtype
!= dfloat32_type_node))))
warning_at (ploc, 0,
"passing argument %d of %qE as %qT "
"rather than %qT due to prototype",
argnum, rname, type, valtype);
}
else if (warn_traditional_conversion
&& INTEGRAL_TYPE_P (type)
&& INTEGRAL_TYPE_P (valtype))
{
tree would_have_been = default_conversion (val);
tree type1 = TREE_TYPE (would_have_been);
if (val == error_mark_node)
;
else if (TREE_CODE (type) == ENUMERAL_TYPE
&& (TYPE_MAIN_VARIANT (type)
== TYPE_MAIN_VARIANT (valtype)))
;
else if (formal_prec != TYPE_PRECISION (type1))
warning_at (ploc, OPT_Wtraditional_conversion,
"passing argument %d of %qE "
"with different width due to prototype",
argnum, rname);
else if (TYPE_UNSIGNED (type) == TYPE_UNSIGNED (type1))
;
else if (TREE_CODE (type) == ENUMERAL_TYPE)
;
else if (TREE_CODE (val) == INTEGER_CST
&& int_fits_type_p (val, type))
;
else if (TYPE_PRECISION (valtype) < TYPE_PRECISION (type)
&& TYPE_UNSIGNED (valtype))
;
else if (TYPE_UNSIGNED (type))
warning_at (ploc, OPT_Wtraditional_conversion,
"passing argument %d of %qE "
"as unsigned due to prototype",
argnum, rname);
else
warning_at (ploc, OPT_Wtraditional_conversion,
"passing argument %d of %qE "
"as signed due to prototype",
argnum, rname);
}
}
if (excess_precision)
val = build1 (EXCESS_PRECISION_EXPR, valtype, val);
origtype = (!origtypes) ? NULL_TREE : (*origtypes)[parmnum];
parmval = convert_for_assignment (loc, ploc, type,
val, origtype, ic_argpass,
npc, fundecl, function,
parmnum + 1);
if (targetm.calls.promote_prototypes (fundecl ? TREE_TYPE (fundecl) : 0)
&& INTEGRAL_TYPE_P (type)
&& (TYPE_PRECISION (type) < TYPE_PRECISION (integer_type_node)))
parmval = default_conversion (parmval);
}
}
else if (promote_float_arg)
{
if (type_generic)
parmval = val;
else
{
if (warn_double_promotion && !c_inhibit_evaluation_warnings)
warning_at (ploc, OPT_Wdouble_promotion,
"implicit conversion from %qT to %qT when passing "
"argument to function",
valtype, double_type_node);
parmval = convert (double_type_node, val);
}
}
else if ((excess_precision && !type_generic)
|| (type_generic_overflow_p && parmnum == 2))
parmval = convert (valtype, val);
else if ((invalid_func_diag =
targetm.calls.invalid_arg_for_unprototyped_fn (typelist, fundecl, val)))
{
error (invalid_func_diag);
return -1;
}
else if (TREE_CODE (val) == ADDR_EXPR && reject_gcc_builtin (val))
{
return -1;
}
else
parmval = default_conversion (val);
(*values)[parmnum] = parmval;
if (parmval == error_mark_node)
error_args = true;
if (typetail)
typetail = TREE_CHAIN (typetail);
}
gcc_assert (parmnum == vec_safe_length (values));
if (typetail != NULL_TREE && TREE_VALUE (typetail) != void_type_node)
{
error_at (loc, "too few arguments to function %qE", function);
inform_declaration (fundecl);
return -1;
}
return error_args ? -1 : (int) parmnum;
}

struct c_expr
parser_build_unary_op (location_t loc, enum tree_code code, struct c_expr arg)
{
struct c_expr result;
result.original_code = code;
result.original_type = NULL;
if (reject_gcc_builtin (arg.value))
{
result.value = error_mark_node;
}
else
{
result.value = build_unary_op (loc, code, arg.value, false);
if (TREE_OVERFLOW_P (result.value) && !TREE_OVERFLOW_P (arg.value))
overflow_warning (loc, result.value, arg.value);
}
set_c_expr_source_range (&result,
loc, arg.get_finish ());
return result;
}
static bool
char_type_p (tree type)
{
return (type == char_type_node
|| type == unsigned_char_type_node
|| type == signed_char_type_node
|| type == char16_type_node
|| type == char32_type_node);
}
struct c_expr
parser_build_binary_op (location_t location, enum tree_code code,
struct c_expr arg1, struct c_expr arg2)
{
struct c_expr result;
enum tree_code code1 = arg1.original_code;
enum tree_code code2 = arg2.original_code;
tree type1 = (arg1.original_type
? arg1.original_type
: TREE_TYPE (arg1.value));
tree type2 = (arg2.original_type
? arg2.original_type
: TREE_TYPE (arg2.value));
result.value = build_binary_op (location, code,
arg1.value, arg2.value, true);
result.original_code = code;
result.original_type = NULL;
if (TREE_CODE (result.value) == ERROR_MARK)
{
set_c_expr_source_range (&result,
arg1.get_start (),
arg2.get_finish ());
return result;
}
if (location != UNKNOWN_LOCATION)
protected_set_expr_location (result.value, location);
set_c_expr_source_range (&result,
arg1.get_start (),
arg2.get_finish ());
if (warn_parentheses)
warn_about_parentheses (location, code, code1, arg1.value, code2,
arg2.value);
if (warn_logical_op)
warn_logical_operator (location, code, TREE_TYPE (result.value),
code1, arg1.value, code2, arg2.value);
if (warn_tautological_compare)
{
tree lhs = arg1.value;
tree rhs = arg2.value;
if (TREE_CODE (lhs) == C_MAYBE_CONST_EXPR)
{
if (C_MAYBE_CONST_EXPR_PRE (lhs) != NULL_TREE
&& TREE_SIDE_EFFECTS (C_MAYBE_CONST_EXPR_PRE (lhs)))
lhs = NULL_TREE;
else
lhs = C_MAYBE_CONST_EXPR_EXPR (lhs);
}
if (TREE_CODE (rhs) == C_MAYBE_CONST_EXPR)
{
if (C_MAYBE_CONST_EXPR_PRE (rhs) != NULL_TREE
&& TREE_SIDE_EFFECTS (C_MAYBE_CONST_EXPR_PRE (rhs)))
rhs = NULL_TREE;
else
rhs = C_MAYBE_CONST_EXPR_EXPR (rhs);
}
if (lhs != NULL_TREE && rhs != NULL_TREE)
warn_tautological_cmp (location, code, lhs, rhs);
}
if (warn_logical_not_paren
&& TREE_CODE_CLASS (code) == tcc_comparison
&& code1 == TRUTH_NOT_EXPR
&& code2 != TRUTH_NOT_EXPR
&& (TREE_CODE (arg1.value) != NE_EXPR
|| !integer_zerop (TREE_OPERAND (arg1.value, 1))))
{
tree t = integer_zero_node;
if (TREE_CODE (arg1.value) == EQ_EXPR
&& integer_zerop (TREE_OPERAND (arg1.value, 1))
&& TREE_TYPE (TREE_OPERAND (arg1.value, 0)) == integer_type_node)
{
t = TREE_OPERAND (arg1.value, 0);
do
{
if (TREE_TYPE (t) != integer_type_node)
break;
if (TREE_CODE (t) == C_MAYBE_CONST_EXPR)
t = C_MAYBE_CONST_EXPR_EXPR (t);
else if (CONVERT_EXPR_P (t))
t = TREE_OPERAND (t, 0);
else
break;
}
while (1);
}
if (TREE_CODE (TREE_TYPE (t)) != BOOLEAN_TYPE)
warn_logical_not_parentheses (location, code, arg1.value, arg2.value);
}
if (code == EQ_EXPR || code == NE_EXPR)
{
if ((code1 == STRING_CST
&& !integer_zerop (tree_strip_nop_conversions (arg2.value)))
|| (code2 == STRING_CST
&& !integer_zerop (tree_strip_nop_conversions (arg1.value))))
warning_at (location, OPT_Waddress,
"comparison with string literal results in unspecified behavior");
if (POINTER_TYPE_P (type1)
&& null_pointer_constant_p (arg2.value)
&& char_type_p (type2)
&& warning_at (location, OPT_Wpointer_compare,
"comparison between pointer and zero character "
"constant"))
inform (arg1.get_start (), "did you mean to dereference the pointer?");
else if (POINTER_TYPE_P (type2)
&& null_pointer_constant_p (arg1.value)
&& char_type_p (type1)
&& warning_at (location, OPT_Wpointer_compare,
"comparison between pointer and zero character "
"constant"))
inform (arg2.get_start (), "did you mean to dereference the pointer?");
}
else if (TREE_CODE_CLASS (code) == tcc_comparison
&& (code1 == STRING_CST || code2 == STRING_CST))
warning_at (location, OPT_Waddress,
"comparison with string literal results in unspecified behavior");
if (TREE_OVERFLOW_P (result.value)
&& !TREE_OVERFLOW_P (arg1.value)
&& !TREE_OVERFLOW_P (arg2.value))
overflow_warning (location, result.value);
if (warn_enum_compare
&& TREE_CODE_CLASS (code) == tcc_comparison
&& TREE_CODE (type1) == ENUMERAL_TYPE
&& TREE_CODE (type2) == ENUMERAL_TYPE
&& TYPE_MAIN_VARIANT (type1) != TYPE_MAIN_VARIANT (type2))
warning_at (location, OPT_Wenum_compare,
"comparison between %qT and %qT",
type1, type2);
return result;
}

static tree
pointer_diff (location_t loc, tree op0, tree op1, tree *instrument_expr)
{
tree restype = ptrdiff_type_node;
tree result, inttype;
addr_space_t as0 = TYPE_ADDR_SPACE (TREE_TYPE (TREE_TYPE (op0)));
addr_space_t as1 = TYPE_ADDR_SPACE (TREE_TYPE (TREE_TYPE (op1)));
tree target_type = TREE_TYPE (TREE_TYPE (op0));
tree orig_op1 = op1;
if (as0 != as1)
{
addr_space_t as_common;
tree common_type;
if (!addr_space_superset (as0, as1, &as_common))
gcc_unreachable ();
common_type = common_pointer_type (TREE_TYPE (op0), TREE_TYPE (op1));
op0 = convert (common_type, op0);
op1 = convert (common_type, op1);
}
if (TYPE_PRECISION (restype) < TYPE_PRECISION (TREE_TYPE (op0)))
inttype = c_common_type_for_size (TYPE_PRECISION (TREE_TYPE (op0)), 0);
else
inttype = restype;
if (TREE_CODE (target_type) == VOID_TYPE)
pedwarn (loc, OPT_Wpointer_arith,
"pointer of type %<void *%> used in subtraction");
if (TREE_CODE (target_type) == FUNCTION_TYPE)
pedwarn (loc, OPT_Wpointer_arith,
"pointer to a function used in subtraction");
if (sanitize_flags_p (SANITIZE_POINTER_SUBTRACT))
{
gcc_assert (current_function_decl != NULL_TREE);
op0 = save_expr (op0);
op1 = save_expr (op1);
tree tt = builtin_decl_explicit (BUILT_IN_ASAN_POINTER_SUBTRACT);
*instrument_expr = build_call_expr_loc (loc, tt, 2, op0, op1);
}
if (TYPE_PRECISION (inttype) > TYPE_PRECISION (TREE_TYPE (op0)))
op0 = build_binary_op (loc, MINUS_EXPR, convert (inttype, op0),
convert (inttype, op1), false);
else
{
op0 = convert (c_common_type (TREE_TYPE (op0), TREE_TYPE (op0)), op0);
op1 = convert (c_common_type (TREE_TYPE (op1), TREE_TYPE (op1)), op1);
op0 = build2_loc (loc, POINTER_DIFF_EXPR, inttype, op0, op1);
}
if (!COMPLETE_OR_VOID_TYPE_P (TREE_TYPE (TREE_TYPE (orig_op1))))
error_at (loc, "arithmetic on pointer to an incomplete type");
op1 = c_size_in_bytes (target_type);
if (pointer_to_zero_sized_aggr_p (TREE_TYPE (orig_op1)))
error_at (loc, "arithmetic on pointer to an empty aggregate");
result = fold_build2_loc (loc, EXACT_DIV_EXPR, inttype,
op0, convert (inttype, op1));
return convert (restype, result);
}

static tree
build_atomic_assign (location_t loc, tree lhs, enum tree_code modifycode,
tree rhs, bool return_old_p)
{
tree fndecl, func_call;
vec<tree, va_gc> *params;
tree val, nonatomic_lhs_type, nonatomic_rhs_type, newval, newval_addr;
tree old, old_addr;
tree compound_stmt;
tree stmt, goto_stmt;
tree loop_label, loop_decl, done_label, done_decl;
tree lhs_type = TREE_TYPE (lhs);
tree lhs_addr = build_unary_op (loc, ADDR_EXPR, lhs, false);
tree seq_cst = build_int_cst (integer_type_node, MEMMODEL_SEQ_CST);
tree rhs_semantic_type = TREE_TYPE (rhs);
tree nonatomic_rhs_semantic_type;
tree rhs_type;
gcc_assert (TYPE_ATOMIC (lhs_type));
if (return_old_p)
gcc_assert (modifycode == PLUS_EXPR || modifycode == MINUS_EXPR);
vec_alloc (params, 6);
compound_stmt = c_begin_compound_stmt (false);
if (TREE_CODE (rhs) == EXCESS_PRECISION_EXPR)
{
gcc_assert (modifycode != NOP_EXPR);
rhs = TREE_OPERAND (rhs, 0);
}
rhs_type = TREE_TYPE (rhs);
if (modifycode != NOP_EXPR)
rhs = c_fully_fold (rhs, false, NULL);
nonatomic_lhs_type = build_qualified_type (lhs_type, TYPE_UNQUALIFIED);
nonatomic_rhs_type = build_qualified_type (rhs_type, TYPE_UNQUALIFIED);
nonatomic_rhs_semantic_type = build_qualified_type (rhs_semantic_type,
TYPE_UNQUALIFIED);
val = create_tmp_var_raw (nonatomic_rhs_type);
TREE_ADDRESSABLE (val) = 1;
TREE_NO_WARNING (val) = 1;
rhs = build4 (TARGET_EXPR, nonatomic_rhs_type, val, rhs, NULL_TREE,
NULL_TREE);
SET_EXPR_LOCATION (rhs, loc);
add_stmt (rhs);
if (modifycode == NOP_EXPR)
{
rhs = build_unary_op (loc, ADDR_EXPR, val, false);
fndecl = builtin_decl_explicit (BUILT_IN_ATOMIC_STORE);
params->quick_push (lhs_addr);
params->quick_push (rhs);
params->quick_push (seq_cst);
func_call = c_build_function_call_vec (loc, vNULL, fndecl, params, NULL);
add_stmt (func_call);
compound_stmt = c_end_compound_stmt (loc, compound_stmt, false);
return build2 (COMPOUND_EXPR, nonatomic_lhs_type, compound_stmt, val);
}
if ((TREE_CODE (lhs_type) == INTEGER_TYPE || POINTER_TYPE_P (lhs_type))
&& TREE_CODE (rhs_type) == INTEGER_TYPE)
{
built_in_function fncode;
switch (modifycode)
{
case PLUS_EXPR:
case POINTER_PLUS_EXPR:
fncode = (return_old_p
? BUILT_IN_ATOMIC_FETCH_ADD_N
: BUILT_IN_ATOMIC_ADD_FETCH_N);
break;
case MINUS_EXPR:
fncode = (return_old_p
? BUILT_IN_ATOMIC_FETCH_SUB_N
: BUILT_IN_ATOMIC_SUB_FETCH_N);
break;
case BIT_AND_EXPR:
fncode = (return_old_p
? BUILT_IN_ATOMIC_FETCH_AND_N
: BUILT_IN_ATOMIC_AND_FETCH_N);
break;
case BIT_IOR_EXPR:
fncode = (return_old_p
? BUILT_IN_ATOMIC_FETCH_OR_N
: BUILT_IN_ATOMIC_OR_FETCH_N);
break;
case BIT_XOR_EXPR:
fncode = (return_old_p
? BUILT_IN_ATOMIC_FETCH_XOR_N
: BUILT_IN_ATOMIC_XOR_FETCH_N);
break;
default:
goto cas_loop;
}
unsigned HOST_WIDE_INT size = tree_to_uhwi (TYPE_SIZE_UNIT (lhs_type));
if (size != 1 && size != 2 && size != 4 && size != 8 && size != 16)
goto cas_loop;
if (POINTER_TYPE_P (lhs_type))
{
if (!COMPLETE_TYPE_P (TREE_TYPE (lhs_type))
|| TYPE_RESTRICT (lhs_type))
goto cas_loop;
tree sz = TYPE_SIZE_UNIT (TREE_TYPE (lhs_type));
rhs = fold_build2_loc (loc, MULT_EXPR, ptrdiff_type_node,
convert (ptrdiff_type_node, rhs),
convert (ptrdiff_type_node, sz));
}
fndecl = builtin_decl_explicit (fncode);
params->quick_push (lhs_addr);
params->quick_push (rhs);
params->quick_push (seq_cst);
func_call = c_build_function_call_vec (loc, vNULL, fndecl, params, NULL);
newval = create_tmp_var_raw (nonatomic_lhs_type);
TREE_ADDRESSABLE (newval) = 1;
TREE_NO_WARNING (newval) = 1;
rhs = build4 (TARGET_EXPR, nonatomic_lhs_type, newval, func_call,
NULL_TREE, NULL_TREE);
SET_EXPR_LOCATION (rhs, loc);
add_stmt (rhs);
compound_stmt = c_end_compound_stmt (loc, compound_stmt, false);
return build2 (COMPOUND_EXPR, nonatomic_lhs_type, compound_stmt, newval);
}
cas_loop:
old = create_tmp_var_raw (nonatomic_lhs_type);
old_addr = build_unary_op (loc, ADDR_EXPR, old, false);
TREE_ADDRESSABLE (old) = 1;
TREE_NO_WARNING (old) = 1;
newval = create_tmp_var_raw (nonatomic_lhs_type);
newval_addr = build_unary_op (loc, ADDR_EXPR, newval, false);
TREE_ADDRESSABLE (newval) = 1;
TREE_NO_WARNING (newval) = 1;
loop_decl = create_artificial_label (loc);
loop_label = build1 (LABEL_EXPR, void_type_node, loop_decl);
done_decl = create_artificial_label (loc);
done_label = build1 (LABEL_EXPR, void_type_node, done_decl);
fndecl = builtin_decl_explicit (BUILT_IN_ATOMIC_LOAD);
params->quick_push (lhs_addr);
params->quick_push (old_addr);
params->quick_push (seq_cst);
func_call = c_build_function_call_vec (loc, vNULL, fndecl, params, NULL);
old = build4 (TARGET_EXPR, nonatomic_lhs_type, old, func_call, NULL_TREE,
NULL_TREE);
add_stmt (old);
params->truncate (0);
bool need_fenv = (flag_trapping_math
&& (FLOAT_TYPE_P (lhs_type) || FLOAT_TYPE_P (rhs_type)));
tree hold_call = NULL_TREE, clear_call = NULL_TREE, update_call = NULL_TREE;
if (need_fenv)
targetm.atomic_assign_expand_fenv (&hold_call, &clear_call, &update_call);
if (hold_call)
add_stmt (hold_call);
add_stmt (loop_label);
if (rhs_type != rhs_semantic_type)
val = build1 (EXCESS_PRECISION_EXPR, nonatomic_rhs_semantic_type, val);
rhs = build_binary_op (loc, modifycode, old, val, true);
if (TREE_CODE (rhs) == EXCESS_PRECISION_EXPR)
{
tree eptype = TREE_TYPE (rhs);
rhs = c_fully_fold (TREE_OPERAND (rhs, 0), false, NULL);
rhs = build1 (EXCESS_PRECISION_EXPR, eptype, rhs);
}
else
rhs = c_fully_fold (rhs, false, NULL);
rhs = convert_for_assignment (loc, UNKNOWN_LOCATION, nonatomic_lhs_type,
rhs, NULL_TREE, ic_assign, false, NULL_TREE,
NULL_TREE, 0);
if (rhs != error_mark_node)
{
rhs = build4 (TARGET_EXPR, nonatomic_lhs_type, newval, rhs, NULL_TREE,
NULL_TREE);
SET_EXPR_LOCATION (rhs, loc);
add_stmt (rhs);
}
fndecl = builtin_decl_explicit (BUILT_IN_ATOMIC_COMPARE_EXCHANGE);
params->quick_push (lhs_addr);
params->quick_push (old_addr);
params->quick_push (newval_addr);
params->quick_push (integer_zero_node);
params->quick_push (seq_cst);
params->quick_push (seq_cst);
func_call = c_build_function_call_vec (loc, vNULL, fndecl, params, NULL);
goto_stmt = build1 (GOTO_EXPR, void_type_node, done_decl);
SET_EXPR_LOCATION (goto_stmt, loc);
stmt = build3 (COND_EXPR, void_type_node, func_call, goto_stmt, NULL_TREE);
SET_EXPR_LOCATION (stmt, loc);
add_stmt (stmt);
if (clear_call)
add_stmt (clear_call);
goto_stmt  = build1 (GOTO_EXPR, void_type_node, loop_decl);
SET_EXPR_LOCATION (goto_stmt, loc);
add_stmt (goto_stmt);
add_stmt (done_label);
if (update_call)
add_stmt (update_call);
compound_stmt = c_end_compound_stmt (loc, compound_stmt, false);
return build2 (COMPOUND_EXPR, nonatomic_lhs_type, compound_stmt,
return_old_p ? old : newval);
}
tree
build_unary_op (location_t location, enum tree_code code, tree xarg,
bool noconvert)
{
tree arg = xarg;
tree argtype = NULL_TREE;
enum tree_code typecode;
tree val;
tree ret = error_mark_node;
tree eptype = NULL_TREE;
const char *invalid_op_diag;
bool int_operands;
int_operands = EXPR_INT_CONST_OPERANDS (xarg);
if (int_operands)
arg = remove_c_maybe_const_expr (arg);
if (code != ADDR_EXPR)
arg = require_complete_type (location, arg);
typecode = TREE_CODE (TREE_TYPE (arg));
if (typecode == ERROR_MARK)
return error_mark_node;
if (typecode == ENUMERAL_TYPE || typecode == BOOLEAN_TYPE)
typecode = INTEGER_TYPE;
if ((invalid_op_diag
= targetm.invalid_unary_op (code, TREE_TYPE (xarg))))
{
error_at (location, invalid_op_diag);
return error_mark_node;
}
if (TREE_CODE (arg) == EXCESS_PRECISION_EXPR)
{
eptype = TREE_TYPE (arg);
arg = TREE_OPERAND (arg, 0);
}
switch (code)
{
case CONVERT_EXPR:
if (!(typecode == INTEGER_TYPE || typecode == REAL_TYPE
|| typecode == FIXED_POINT_TYPE || typecode == COMPLEX_TYPE
|| typecode == VECTOR_TYPE))
{
error_at (location, "wrong type argument to unary plus");
return error_mark_node;
}
else if (!noconvert)
arg = default_conversion (arg);
arg = non_lvalue_loc (location, arg);
break;
case NEGATE_EXPR:
if (!(typecode == INTEGER_TYPE || typecode == REAL_TYPE
|| typecode == FIXED_POINT_TYPE || typecode == COMPLEX_TYPE
|| typecode == VECTOR_TYPE))
{
error_at (location, "wrong type argument to unary minus");
return error_mark_node;
}
else if (!noconvert)
arg = default_conversion (arg);
break;
case BIT_NOT_EXPR:
if (typecode == INTEGER_TYPE
|| (typecode == VECTOR_TYPE
&& !VECTOR_FLOAT_TYPE_P (TREE_TYPE (arg))))
{
tree e = arg;
while (TREE_CODE (e) == COMPOUND_EXPR)
e = TREE_OPERAND (e, 1);
if ((TREE_CODE (TREE_TYPE (arg)) == BOOLEAN_TYPE
|| truth_value_p (TREE_CODE (e)))
&& warning_at (location, OPT_Wbool_operation,
"%<~%> on a boolean expression"))
{
gcc_rich_location richloc (location);
richloc.add_fixit_insert_before (location, "!");
inform (&richloc, "did you mean to use logical not?");
}
if (!noconvert)
arg = default_conversion (arg);
}
else if (typecode == COMPLEX_TYPE)
{
code = CONJ_EXPR;
pedwarn (location, OPT_Wpedantic,
"ISO C does not support %<~%> for complex conjugation");
if (!noconvert)
arg = default_conversion (arg);
}
else
{
error_at (location, "wrong type argument to bit-complement");
return error_mark_node;
}
break;
case ABS_EXPR:
if (!(typecode == INTEGER_TYPE || typecode == REAL_TYPE))
{
error_at (location, "wrong type argument to abs");
return error_mark_node;
}
else if (!noconvert)
arg = default_conversion (arg);
break;
case CONJ_EXPR:
if (!(typecode == INTEGER_TYPE || typecode == REAL_TYPE
|| typecode == COMPLEX_TYPE))
{
error_at (location, "wrong type argument to conjugation");
return error_mark_node;
}
else if (!noconvert)
arg = default_conversion (arg);
break;
case TRUTH_NOT_EXPR:
if (typecode != INTEGER_TYPE && typecode != FIXED_POINT_TYPE
&& typecode != REAL_TYPE && typecode != POINTER_TYPE
&& typecode != COMPLEX_TYPE)
{
error_at (location,
"wrong type argument to unary exclamation mark");
return error_mark_node;
}
if (int_operands)
{
arg = c_objc_common_truthvalue_conversion (location, xarg);
arg = remove_c_maybe_const_expr (arg);
}
else
arg = c_objc_common_truthvalue_conversion (location, arg);
ret = invert_truthvalue_loc (location, arg);
if (EXPR_P (ret) && EXPR_HAS_LOCATION (ret))
location = EXPR_LOCATION (ret);
goto return_build_unary_op;
case REALPART_EXPR:
case IMAGPART_EXPR:
ret = build_real_imag_expr (location, code, arg);
if (ret == error_mark_node)
return error_mark_node;
if (eptype && TREE_CODE (eptype) == COMPLEX_TYPE)
eptype = TREE_TYPE (eptype);
goto return_build_unary_op;
case PREINCREMENT_EXPR:
case POSTINCREMENT_EXPR:
case PREDECREMENT_EXPR:
case POSTDECREMENT_EXPR:
if (TREE_CODE (arg) == C_MAYBE_CONST_EXPR)
{
tree inner = build_unary_op (location, code,
C_MAYBE_CONST_EXPR_EXPR (arg),
noconvert);
if (inner == error_mark_node)
return error_mark_node;
ret = build2 (C_MAYBE_CONST_EXPR, TREE_TYPE (inner),
C_MAYBE_CONST_EXPR_PRE (arg), inner);
gcc_assert (!C_MAYBE_CONST_EXPR_INT_OPERANDS (arg));
C_MAYBE_CONST_EXPR_NON_CONST (ret) = 1;
goto return_build_unary_op;
}
if (!objc_is_property_ref (arg)
&& !lvalue_or_else (location,
arg, ((code == PREINCREMENT_EXPR
|| code == POSTINCREMENT_EXPR)
? lv_increment
: lv_decrement)))
return error_mark_node;
if (warn_cxx_compat && TREE_CODE (TREE_TYPE (arg)) == ENUMERAL_TYPE)
{
if (code == PREINCREMENT_EXPR || code == POSTINCREMENT_EXPR)
warning_at (location, OPT_Wc___compat,
"increment of enumeration value is invalid in C++");
else
warning_at (location, OPT_Wc___compat,
"decrement of enumeration value is invalid in C++");
}
if (TREE_CODE (TREE_TYPE (arg)) == BOOLEAN_TYPE)
{
if (code == PREINCREMENT_EXPR || code == POSTINCREMENT_EXPR)
warning_at (location, OPT_Wbool_operation,
"increment of a boolean expression");
else
warning_at (location, OPT_Wbool_operation,
"decrement of a boolean expression");
}
arg = c_fully_fold (arg, false, NULL, true);
bool atomic_op;
atomic_op = really_atomic_lvalue (arg);
if (typecode == COMPLEX_TYPE)
{
tree real, imag;
pedwarn (location, OPT_Wpedantic,
"ISO C does not support %<++%> and %<--%> on complex types");
if (!atomic_op)
{
arg = stabilize_reference (arg);
real = build_unary_op (EXPR_LOCATION (arg), REALPART_EXPR, arg,
true);
imag = build_unary_op (EXPR_LOCATION (arg), IMAGPART_EXPR, arg,
true);
real = build_unary_op (EXPR_LOCATION (arg), code, real, true);
if (real == error_mark_node || imag == error_mark_node)
return error_mark_node;
ret = build2 (COMPLEX_EXPR, TREE_TYPE (arg),
real, imag);
goto return_build_unary_op;
}
}
if (typecode != POINTER_TYPE && typecode != FIXED_POINT_TYPE
&& typecode != INTEGER_TYPE && typecode != REAL_TYPE
&& typecode != COMPLEX_TYPE && typecode != VECTOR_TYPE)
{
if (code == PREINCREMENT_EXPR || code == POSTINCREMENT_EXPR)
error_at (location, "wrong type argument to increment");
else
error_at (location, "wrong type argument to decrement");
return error_mark_node;
}
{
tree inc;
argtype = TREE_TYPE (arg);
if (typecode == POINTER_TYPE)
{
if (!COMPLETE_OR_VOID_TYPE_P (TREE_TYPE (argtype)))
{
if (code == PREINCREMENT_EXPR || code == POSTINCREMENT_EXPR)
error_at (location,
"increment of pointer to an incomplete type %qT",
TREE_TYPE (argtype));
else
error_at (location,
"decrement of pointer to an incomplete type %qT",
TREE_TYPE (argtype));
}
else if (TREE_CODE (TREE_TYPE (argtype)) == FUNCTION_TYPE
|| TREE_CODE (TREE_TYPE (argtype)) == VOID_TYPE)
{
if (code == PREINCREMENT_EXPR || code == POSTINCREMENT_EXPR)
pedwarn (location, OPT_Wpointer_arith,
"wrong type argument to increment");
else
pedwarn (location, OPT_Wpointer_arith,
"wrong type argument to decrement");
}
inc = c_size_in_bytes (TREE_TYPE (argtype));
inc = convert_to_ptrofftype_loc (location, inc);
}
else if (FRACT_MODE_P (TYPE_MODE (argtype)))
{
if (code == PREINCREMENT_EXPR)
code = PREDECREMENT_EXPR;
else if (code == PREDECREMENT_EXPR)
code = PREINCREMENT_EXPR;
else if (code == POSTINCREMENT_EXPR)
code = POSTDECREMENT_EXPR;
else 
code = POSTINCREMENT_EXPR;
inc = integer_minus_one_node;
inc = convert (argtype, inc);
}
else
{
inc = VECTOR_TYPE_P (argtype)
? build_one_cst (argtype)
: integer_one_node;
inc = convert (argtype, inc);
}
if (objc_is_property_ref (arg))
return objc_build_incr_expr_for_property_ref (location, code,
arg, inc);
if (TYPE_READONLY (argtype))
{
readonly_error (location, arg,
((code == PREINCREMENT_EXPR
|| code == POSTINCREMENT_EXPR)
? lv_increment : lv_decrement));
return error_mark_node;
}
else if (TREE_READONLY (arg))
readonly_warning (arg,
((code == PREINCREMENT_EXPR
|| code == POSTINCREMENT_EXPR)
? lv_increment : lv_decrement));
if (atomic_op)
{
arg = stabilize_reference (arg);
ret = build_atomic_assign (location, arg,
((code == PREINCREMENT_EXPR
|| code == POSTINCREMENT_EXPR)
? PLUS_EXPR
: MINUS_EXPR),
(FRACT_MODE_P (TYPE_MODE (argtype))
? inc
: integer_one_node),
(code == POSTINCREMENT_EXPR
|| code == POSTDECREMENT_EXPR));
goto return_build_unary_op;
}
if (TREE_CODE (TREE_TYPE (arg)) == BOOLEAN_TYPE)
val = boolean_increment (code, arg);
else
val = build2 (code, TREE_TYPE (arg), arg, inc);
TREE_SIDE_EFFECTS (val) = 1;
if (TREE_CODE (val) != code)
TREE_NO_WARNING (val) = 1;
ret = val;
goto return_build_unary_op;
}
case ADDR_EXPR:
if (VOID_TYPE_P (TREE_TYPE (arg))
&& TYPE_QUALS (TREE_TYPE (arg)) == TYPE_UNQUALIFIED
&& (!INDIRECT_REF_P (arg) || !flag_isoc99))
pedwarn (location, 0, "taking address of expression of type %<void%>");
if (INDIRECT_REF_P (arg))
{
if (lvalue_p (TREE_OPERAND (arg, 0)))
return non_lvalue_loc (location, TREE_OPERAND (arg, 0));
ret = TREE_OPERAND (arg, 0);
goto return_build_unary_op;
}
if (typecode != FUNCTION_TYPE && !noconvert
&& !lvalue_or_else (location, arg, lv_addressof))
return error_mark_node;
if (TREE_CODE (arg) == C_MAYBE_CONST_EXPR)
{
tree inner = build_unary_op (location, code,
C_MAYBE_CONST_EXPR_EXPR (arg),
noconvert);
ret = build2 (C_MAYBE_CONST_EXPR, TREE_TYPE (inner),
C_MAYBE_CONST_EXPR_PRE (arg), inner);
gcc_assert (!C_MAYBE_CONST_EXPR_INT_OPERANDS (arg));
C_MAYBE_CONST_EXPR_NON_CONST (ret)
= C_MAYBE_CONST_EXPR_NON_CONST (arg);
goto return_build_unary_op;
}
argtype = TREE_TYPE (arg);
if ((DECL_P (arg) || REFERENCE_CLASS_P (arg))
&& (TREE_READONLY (arg) || TREE_THIS_VOLATILE (arg))
&& TREE_CODE (argtype) == FUNCTION_TYPE)
{
int orig_quals = TYPE_QUALS (strip_array_types (argtype));
int quals = orig_quals;
if (TREE_READONLY (arg))
quals |= TYPE_QUAL_CONST;
if (TREE_THIS_VOLATILE (arg))
quals |= TYPE_QUAL_VOLATILE;
argtype = c_build_qualified_type (argtype, quals);
}
switch (TREE_CODE (arg))
{
case COMPONENT_REF:
if (DECL_C_BIT_FIELD (TREE_OPERAND (arg, 1)))
{
error_at (location, "cannot take address of bit-field %qD",
TREE_OPERAND (arg, 1));
return error_mark_node;
}
case ARRAY_REF:
if (TYPE_REVERSE_STORAGE_ORDER (TREE_TYPE (TREE_OPERAND (arg, 0))))
{
if (!AGGREGATE_TYPE_P (TREE_TYPE (arg))
&& !VECTOR_TYPE_P (TREE_TYPE (arg)))
{
error_at (location, "cannot take address of scalar with "
"reverse storage order");
return error_mark_node;
}
if (TREE_CODE (TREE_TYPE (arg)) == ARRAY_TYPE
&& TYPE_REVERSE_STORAGE_ORDER (TREE_TYPE (arg)))
warning_at (location, OPT_Wscalar_storage_order,
"address of array with reverse scalar storage "
"order requested");
}
default:
break;
}
if (!c_mark_addressable (arg))
return error_mark_node;
gcc_assert (TREE_CODE (arg) != COMPONENT_REF
|| !DECL_C_BIT_FIELD (TREE_OPERAND (arg, 1)));
argtype = build_pointer_type (argtype);
val = get_base_address (arg);
if (val && INDIRECT_REF_P (val)
&& TREE_CONSTANT (TREE_OPERAND (val, 0)))
{
ret = fold_offsetof (arg, argtype);
goto return_build_unary_op;
}
val = build1 (ADDR_EXPR, argtype, arg);
ret = val;
goto return_build_unary_op;
default:
gcc_unreachable ();
}
if (argtype == NULL_TREE)
argtype = TREE_TYPE (arg);
if (TREE_CODE (arg) == INTEGER_CST)
ret = (require_constant_value
? fold_build1_initializer_loc (location, code, argtype, arg)
: fold_build1_loc (location, code, argtype, arg));
else
ret = build1 (code, argtype, arg);
return_build_unary_op:
gcc_assert (ret != error_mark_node);
if (TREE_CODE (ret) == INTEGER_CST && !TREE_OVERFLOW (ret)
&& !(TREE_CODE (xarg) == INTEGER_CST && !TREE_OVERFLOW (xarg)))
ret = build1 (NOP_EXPR, TREE_TYPE (ret), ret);
else if (TREE_CODE (ret) != INTEGER_CST && int_operands)
ret = note_integer_operands (ret);
if (eptype)
ret = build1 (EXCESS_PRECISION_EXPR, eptype, ret);
protected_set_expr_location (ret, location);
return ret;
}
bool
lvalue_p (const_tree ref)
{
const enum tree_code code = TREE_CODE (ref);
switch (code)
{
case REALPART_EXPR:
case IMAGPART_EXPR:
case COMPONENT_REF:
return lvalue_p (TREE_OPERAND (ref, 0));
case C_MAYBE_CONST_EXPR:
return lvalue_p (TREE_OPERAND (ref, 1));
case COMPOUND_LITERAL_EXPR:
case STRING_CST:
return true;
case INDIRECT_REF:
case ARRAY_REF:
case VAR_DECL:
case PARM_DECL:
case RESULT_DECL:
case ERROR_MARK:
return (TREE_CODE (TREE_TYPE (ref)) != FUNCTION_TYPE
&& TREE_CODE (TREE_TYPE (ref)) != METHOD_TYPE);
case BIND_EXPR:
return TREE_CODE (TREE_TYPE (ref)) == ARRAY_TYPE;
default:
return false;
}
}

static void
readonly_warning (tree arg, enum lvalue_use use)
{
switch (use)
{
case lv_assign:
warning (0, "assignment of read-only location %qE", arg);
break;
case lv_increment:
warning (0, "increment of read-only location %qE", arg);
break;
case lv_decrement:
warning (0, "decrement of read-only location %qE", arg);
break;
default:
gcc_unreachable ();
}
return;
}
static int
lvalue_or_else (location_t loc, const_tree ref, enum lvalue_use use)
{
int win = lvalue_p (ref);
if (!win)
lvalue_error (loc, use);
return win;
}

bool
c_mark_addressable (tree exp, bool array_ref_p)
{
tree x = exp;
while (1)
switch (TREE_CODE (x))
{
case VIEW_CONVERT_EXPR:
if (array_ref_p
&& TREE_CODE (TREE_TYPE (x)) == ARRAY_TYPE
&& VECTOR_TYPE_P (TREE_TYPE (TREE_OPERAND (x, 0))))
return true;
case COMPONENT_REF:
case ADDR_EXPR:
case ARRAY_REF:
case REALPART_EXPR:
case IMAGPART_EXPR:
x = TREE_OPERAND (x, 0);
break;
case COMPOUND_LITERAL_EXPR:
case CONSTRUCTOR:
TREE_ADDRESSABLE (x) = 1;
return true;
case VAR_DECL:
case CONST_DECL:
case PARM_DECL:
case RESULT_DECL:
if (C_DECL_REGISTER (x)
&& DECL_NONLOCAL (x))
{
if (TREE_PUBLIC (x) || is_global_var (x))
{
error
("global register variable %qD used in nested function", x);
return false;
}
pedwarn (input_location, 0, "register variable %qD used in nested function", x);
}
else if (C_DECL_REGISTER (x))
{
if (TREE_PUBLIC (x) || is_global_var (x))
error ("address of global register variable %qD requested", x);
else
error ("address of register variable %qD requested", x);
return false;
}
case FUNCTION_DECL:
TREE_ADDRESSABLE (x) = 1;
default:
return true;
}
}

static tree
ep_convert_and_check (location_t loc, tree type, tree expr,
tree semantic_type)
{
if (TREE_TYPE (expr) == type)
return expr;
if (flag_isoc11 || !semantic_type)
return convert_and_check (loc, type, expr);
if (TREE_CODE (TREE_TYPE (expr)) == INTEGER_TYPE
&& TREE_TYPE (expr) != semantic_type)
{
expr = convert_and_check (loc, semantic_type, expr);
}
return convert (type, expr);
}
tree
build_conditional_expr (location_t colon_loc, tree ifexp, bool ifexp_bcp,
tree op1, tree op1_original_type, location_t op1_loc,
tree op2, tree op2_original_type, location_t op2_loc)
{
tree type1;
tree type2;
enum tree_code code1;
enum tree_code code2;
tree result_type = NULL;
tree semantic_result_type = NULL;
tree orig_op1 = op1, orig_op2 = op2;
bool int_const, op1_int_operands, op2_int_operands, int_operands;
bool ifexp_int_operands;
tree ret;
op1_int_operands = EXPR_INT_CONST_OPERANDS (orig_op1);
if (op1_int_operands)
op1 = remove_c_maybe_const_expr (op1);
op2_int_operands = EXPR_INT_CONST_OPERANDS (orig_op2);
if (op2_int_operands)
op2 = remove_c_maybe_const_expr (op2);
ifexp_int_operands = EXPR_INT_CONST_OPERANDS (ifexp);
if (ifexp_int_operands)
ifexp = remove_c_maybe_const_expr (ifexp);
if (TREE_CODE (TREE_TYPE (op1)) != VOID_TYPE)
op1 = default_conversion (op1);
if (TREE_CODE (TREE_TYPE (op2)) != VOID_TYPE)
op2 = default_conversion (op2);
if (TREE_CODE (ifexp) == ERROR_MARK
|| TREE_CODE (TREE_TYPE (op1)) == ERROR_MARK
|| TREE_CODE (TREE_TYPE (op2)) == ERROR_MARK)
return error_mark_node;
type1 = TREE_TYPE (op1);
code1 = TREE_CODE (type1);
type2 = TREE_TYPE (op2);
code2 = TREE_CODE (type2);
if (code1 == POINTER_TYPE && reject_gcc_builtin (op1))
return error_mark_node;
if (code2 == POINTER_TYPE && reject_gcc_builtin (op2))
return error_mark_node;
if (code1 == ARRAY_TYPE || code2 == ARRAY_TYPE)
{
error_at (colon_loc, "non-lvalue array in conditional expression");
return error_mark_node;
}
if ((TREE_CODE (op1) == EXCESS_PRECISION_EXPR
|| TREE_CODE (op2) == EXCESS_PRECISION_EXPR)
&& (code1 == INTEGER_TYPE || code1 == REAL_TYPE
|| code1 == COMPLEX_TYPE)
&& (code2 == INTEGER_TYPE || code2 == REAL_TYPE
|| code2 == COMPLEX_TYPE))
{
semantic_result_type = c_common_type (type1, type2);
if (TREE_CODE (op1) == EXCESS_PRECISION_EXPR)
{
op1 = TREE_OPERAND (op1, 0);
type1 = TREE_TYPE (op1);
gcc_assert (TREE_CODE (type1) == code1);
}
if (TREE_CODE (op2) == EXCESS_PRECISION_EXPR)
{
op2 = TREE_OPERAND (op2, 0);
type2 = TREE_TYPE (op2);
gcc_assert (TREE_CODE (type2) == code2);
}
}
if (warn_cxx_compat)
{
tree t1 = op1_original_type ? op1_original_type : TREE_TYPE (orig_op1);
tree t2 = op2_original_type ? op2_original_type : TREE_TYPE (orig_op2);
if (TREE_CODE (t1) == ENUMERAL_TYPE
&& TREE_CODE (t2) == ENUMERAL_TYPE
&& TYPE_MAIN_VARIANT (t1) != TYPE_MAIN_VARIANT (t2))
warning_at (colon_loc, OPT_Wc___compat,
("different enum types in conditional is "
"invalid in C++: %qT vs %qT"),
t1, t2);
}
if (TYPE_MAIN_VARIANT (type1) == TYPE_MAIN_VARIANT (type2))
{
if (type1 == type2)
result_type = type1;
else
result_type = TYPE_MAIN_VARIANT (type1);
}
else if ((code1 == INTEGER_TYPE || code1 == REAL_TYPE
|| code1 == COMPLEX_TYPE)
&& (code2 == INTEGER_TYPE || code2 == REAL_TYPE
|| code2 == COMPLEX_TYPE))
{
tree eptype1 = type1;
tree eptype2 = type2;
if (flag_isoc11)
{
tree eptype;
if (ANY_INTEGRAL_TYPE_P (type1)
&& (eptype = excess_precision_type (type2)) != NULL_TREE)
{
eptype2 = eptype;
if (!semantic_result_type)
semantic_result_type = c_common_type (type1, type2);
}
else if (ANY_INTEGRAL_TYPE_P (type2)
&& (eptype = excess_precision_type (type1)) != NULL_TREE)
{
eptype1 = eptype;
if (!semantic_result_type)
semantic_result_type = c_common_type (type1, type2);
}
}
result_type = c_common_type (eptype1, eptype2);
if (result_type == error_mark_node)
return error_mark_node;
do_warn_double_promotion (result_type, type1, type2,
"implicit conversion from %qT to %qT to "
"match other result of conditional",
colon_loc);
if (c_inhibit_evaluation_warnings == 0)
{
int unsigned_op1 = TYPE_UNSIGNED (TREE_TYPE (orig_op1));
int unsigned_op2 = TYPE_UNSIGNED (TREE_TYPE (orig_op2));
if (unsigned_op1 ^ unsigned_op2)
{
bool ovf;
if (!TYPE_UNSIGNED (result_type))
;
else
{
bool op1_maybe_const = true;
bool op2_maybe_const = true;
c_inhibit_evaluation_warnings
+= (ifexp == truthvalue_false_node);
op1 = c_fully_fold (op1, require_constant_value,
&op1_maybe_const);
c_inhibit_evaluation_warnings
-= (ifexp == truthvalue_false_node);
c_inhibit_evaluation_warnings
+= (ifexp == truthvalue_true_node);
op2 = c_fully_fold (op2, require_constant_value,
&op2_maybe_const);
c_inhibit_evaluation_warnings
-= (ifexp == truthvalue_true_node);
if (warn_sign_compare)
{
if ((unsigned_op2
&& tree_expr_nonnegative_warnv_p (op1, &ovf))
|| (unsigned_op1
&& tree_expr_nonnegative_warnv_p (op2, &ovf)))
;
else if (unsigned_op2)
warning_at (op1_loc, OPT_Wsign_compare,
"operand of ?: changes signedness from "
"%qT to %qT due to unsignedness of other "
"operand", TREE_TYPE (orig_op1),
TREE_TYPE (orig_op2));
else
warning_at (op2_loc, OPT_Wsign_compare,
"operand of ?: changes signedness from "
"%qT to %qT due to unsignedness of other "
"operand", TREE_TYPE (orig_op2),
TREE_TYPE (orig_op1));
}
if (!op1_maybe_const || TREE_CODE (op1) != INTEGER_CST)
op1 = c_wrap_maybe_const (op1, !op1_maybe_const);
if (!op2_maybe_const || TREE_CODE (op2) != INTEGER_CST)
op2 = c_wrap_maybe_const (op2, !op2_maybe_const);
}
}
}
}
else if (code1 == VOID_TYPE || code2 == VOID_TYPE)
{
if (code1 != VOID_TYPE || code2 != VOID_TYPE)
pedwarn (colon_loc, OPT_Wpedantic,
"ISO C forbids conditional expr with only one void side");
result_type = void_type_node;
}
else if (code1 == POINTER_TYPE && code2 == POINTER_TYPE)
{
addr_space_t as1 = TYPE_ADDR_SPACE (TREE_TYPE (type1));
addr_space_t as2 = TYPE_ADDR_SPACE (TREE_TYPE (type2));
addr_space_t as_common;
if (comp_target_types (colon_loc, type1, type2))
result_type = common_pointer_type (type1, type2);
else if (null_pointer_constant_p (orig_op1))
result_type = type2;
else if (null_pointer_constant_p (orig_op2))
result_type = type1;
else if (!addr_space_superset (as1, as2, &as_common))
{
error_at (colon_loc, "pointers to disjoint address spaces "
"used in conditional expression");
return error_mark_node;
}
else if (VOID_TYPE_P (TREE_TYPE (type1))
&& !TYPE_ATOMIC (TREE_TYPE (type1)))
{
if ((TREE_CODE (TREE_TYPE (type2)) == ARRAY_TYPE)
&& (TYPE_QUALS (strip_array_types (TREE_TYPE (type2)))
& ~TYPE_QUALS (TREE_TYPE (type1))))
warning_at (colon_loc, OPT_Wdiscarded_array_qualifiers,
"pointer to array loses qualifier "
"in conditional expression");
if (TREE_CODE (TREE_TYPE (type2)) == FUNCTION_TYPE)
pedwarn (colon_loc, OPT_Wpedantic,
"ISO C forbids conditional expr between "
"%<void *%> and function pointer");
result_type = build_pointer_type (qualify_type (TREE_TYPE (type1),
TREE_TYPE (type2)));
}
else if (VOID_TYPE_P (TREE_TYPE (type2))
&& !TYPE_ATOMIC (TREE_TYPE (type2)))
{
if ((TREE_CODE (TREE_TYPE (type1)) == ARRAY_TYPE)
&& (TYPE_QUALS (strip_array_types (TREE_TYPE (type1)))
& ~TYPE_QUALS (TREE_TYPE (type2))))
warning_at (colon_loc, OPT_Wdiscarded_array_qualifiers,
"pointer to array loses qualifier "
"in conditional expression");
if (TREE_CODE (TREE_TYPE (type1)) == FUNCTION_TYPE)
pedwarn (colon_loc, OPT_Wpedantic,
"ISO C forbids conditional expr between "
"%<void *%> and function pointer");
result_type = build_pointer_type (qualify_type (TREE_TYPE (type2),
TREE_TYPE (type1)));
}
else if (objc_have_common_type (type1, type2, -3, NULL_TREE))
result_type = objc_common_type (type1, type2);
else
{
int qual = ENCODE_QUAL_ADDR_SPACE (as_common);
pedwarn (colon_loc, 0,
"pointer type mismatch in conditional expression");
result_type = build_pointer_type
(build_qualified_type (void_type_node, qual));
}
}
else if (code1 == POINTER_TYPE && code2 == INTEGER_TYPE)
{
if (!null_pointer_constant_p (orig_op2))
pedwarn (colon_loc, 0,
"pointer/integer type mismatch in conditional expression");
else
{
op2 = null_pointer_node;
}
result_type = type1;
}
else if (code2 == POINTER_TYPE && code1 == INTEGER_TYPE)
{
if (!null_pointer_constant_p (orig_op1))
pedwarn (colon_loc, 0,
"pointer/integer type mismatch in conditional expression");
else
{
op1 = null_pointer_node;
}
result_type = type2;
}
if (!result_type)
{
if (flag_cond_mismatch)
result_type = void_type_node;
else
{
error_at (colon_loc, "type mismatch in conditional expression");
return error_mark_node;
}
}
result_type
= build_type_variant (result_type,
TYPE_READONLY (type1) || TYPE_READONLY (type2),
TYPE_VOLATILE (type1) || TYPE_VOLATILE (type2));
op1 = ep_convert_and_check (colon_loc, result_type, op1,
semantic_result_type);
op2 = ep_convert_and_check (colon_loc, result_type, op2,
semantic_result_type);
if (ifexp_bcp && ifexp == truthvalue_true_node)
{
op2_int_operands = true;
op1 = c_fully_fold (op1, require_constant_value, NULL);
}
if (ifexp_bcp && ifexp == truthvalue_false_node)
{
op1_int_operands = true;
op2 = c_fully_fold (op2, require_constant_value, NULL);
}
int_const = int_operands = (ifexp_int_operands
&& op1_int_operands
&& op2_int_operands);
if (int_operands)
{
int_const = ((ifexp == truthvalue_true_node
&& TREE_CODE (orig_op1) == INTEGER_CST
&& !TREE_OVERFLOW (orig_op1))
|| (ifexp == truthvalue_false_node
&& TREE_CODE (orig_op2) == INTEGER_CST
&& !TREE_OVERFLOW (orig_op2)));
}
if (VECTOR_TYPE_P (TREE_TYPE (ifexp)))
{
tree vectype = TREE_TYPE (ifexp);
tree elem_type = TREE_TYPE (vectype);
tree zero = build_int_cst (elem_type, 0);
tree zero_vec = build_vector_from_val (vectype, zero);
tree cmp_type = build_same_sized_truth_vector_type (vectype);
ifexp = build2 (NE_EXPR, cmp_type, ifexp, zero_vec);
}
if (int_const || (ifexp_bcp && TREE_CODE (ifexp) == INTEGER_CST))
ret = fold_build3_loc (colon_loc, COND_EXPR, result_type, ifexp, op1, op2);
else
{
if (int_operands)
{
op1 = c_fully_fold (op1, false, NULL);
op2 = c_fully_fold (op2, false, NULL);
}
ret = build3 (COND_EXPR, result_type, ifexp, op1, op2);
if (int_operands)
ret = note_integer_operands (ret);
}
if (semantic_result_type)
ret = build1 (EXCESS_PRECISION_EXPR, semantic_result_type, ret);
protected_set_expr_location (ret, colon_loc);
if (warn_duplicated_branches
&& TREE_CODE (ret) == COND_EXPR
&& (op1 == op2 || operand_equal_p (op1, op2, 0)))
warning_at (EXPR_LOCATION (ret), OPT_Wduplicated_branches,
"this condition has identical branches");
return ret;
}

tree
build_compound_expr (location_t loc, tree expr1, tree expr2)
{
bool expr1_int_operands, expr2_int_operands;
tree eptype = NULL_TREE;
tree ret;
expr1_int_operands = EXPR_INT_CONST_OPERANDS (expr1);
if (expr1_int_operands)
expr1 = remove_c_maybe_const_expr (expr1);
expr2_int_operands = EXPR_INT_CONST_OPERANDS (expr2);
if (expr2_int_operands)
expr2 = remove_c_maybe_const_expr (expr2);
if (TREE_CODE (expr1) == EXCESS_PRECISION_EXPR)
expr1 = TREE_OPERAND (expr1, 0);
if (TREE_CODE (expr2) == EXCESS_PRECISION_EXPR)
{
eptype = TREE_TYPE (expr2);
expr2 = TREE_OPERAND (expr2, 0);
}
if (!TREE_SIDE_EFFECTS (expr1))
{
if (warn_unused_value)
{
if (VOID_TYPE_P (TREE_TYPE (expr1))
&& CONVERT_EXPR_P (expr1))
; 
else if (VOID_TYPE_P (TREE_TYPE (expr1))
&& TREE_CODE (expr1) == COMPOUND_EXPR
&& CONVERT_EXPR_P (TREE_OPERAND (expr1, 1)))
; 
else
warning_at (loc, OPT_Wunused_value,
"left-hand operand of comma expression has no effect");
}
}
else if (TREE_CODE (expr1) == COMPOUND_EXPR
&& warn_unused_value)
{
tree r = expr1;
location_t cloc = loc;
while (TREE_CODE (r) == COMPOUND_EXPR)
{
if (EXPR_HAS_LOCATION (r))
cloc = EXPR_LOCATION (r);
r = TREE_OPERAND (r, 1);
}
if (!TREE_SIDE_EFFECTS (r)
&& !VOID_TYPE_P (TREE_TYPE (r))
&& !CONVERT_EXPR_P (r))
warning_at (cloc, OPT_Wunused_value,
"right-hand operand of comma expression has no effect");
}
else if (warn_unused_value)
warn_if_unused_value (expr1, loc);
if (expr2 == error_mark_node)
return error_mark_node;
ret = build2 (COMPOUND_EXPR, TREE_TYPE (expr2), expr1, expr2);
if (flag_isoc99
&& expr1_int_operands
&& expr2_int_operands)
ret = note_integer_operands (ret);
if (eptype)
ret = build1 (EXCESS_PRECISION_EXPR, eptype, ret);
protected_set_expr_location (ret, loc);
return ret;
}
static void
handle_warn_cast_qual (location_t loc, tree type, tree otype)
{
tree in_type = type;
tree in_otype = otype;
int added = 0;
int discarded = 0;
bool is_const;
do
{
in_otype = TREE_TYPE (in_otype);
in_type = TREE_TYPE (in_type);
if (TREE_CODE (in_otype) == FUNCTION_TYPE
&& TREE_CODE (in_type) == FUNCTION_TYPE)
added |= (TYPE_QUALS_NO_ADDR_SPACE (in_type)
& ~TYPE_QUALS_NO_ADDR_SPACE (in_otype));
else
discarded |= (TYPE_QUALS_NO_ADDR_SPACE (in_otype)
& ~TYPE_QUALS_NO_ADDR_SPACE (in_type));
}
while (TREE_CODE (in_type) == POINTER_TYPE
&& TREE_CODE (in_otype) == POINTER_TYPE);
if (added)
warning_at (loc, OPT_Wcast_qual,
"cast adds %q#v qualifier to function type", added);
if (discarded)
warning_at (loc, OPT_Wcast_qual,
"cast discards %qv qualifier from pointer target type",
discarded);
if (added || discarded)
return;
if (TYPE_MAIN_VARIANT (in_type) != TYPE_MAIN_VARIANT (in_otype))
return;
if (TREE_CODE (TREE_TYPE (type)) != POINTER_TYPE)
return;
in_type = type;
in_otype = otype;
is_const = TYPE_READONLY (TREE_TYPE (in_type));
do
{
in_type = TREE_TYPE (in_type);
in_otype = TREE_TYPE (in_otype);
if ((TYPE_QUALS (in_type) &~ TYPE_QUALS (in_otype)) != 0
&& !is_const)
{
warning_at (loc, OPT_Wcast_qual,
"to be safe all intermediate pointers in cast from "
"%qT to %qT must be %<const%> qualified",
otype, type);
break;
}
if (is_const)
is_const = TYPE_READONLY (in_type);
}
while (TREE_CODE (in_type) == POINTER_TYPE);
}
static bool
c_safe_arg_type_equiv_p (tree t1, tree t2)
{
t1 = TYPE_MAIN_VARIANT (t1);
t2 = TYPE_MAIN_VARIANT (t2);
if (TREE_CODE (t1) == POINTER_TYPE
&& TREE_CODE (t2) == POINTER_TYPE)
return true;
if (INTEGRAL_TYPE_P (t1)
&& INTEGRAL_TYPE_P (t2)
&& TYPE_PRECISION (t1) == TYPE_PRECISION (t2)
&& (TYPE_UNSIGNED (t1) == TYPE_UNSIGNED (t2)
|| !targetm.calls.promote_prototypes (NULL_TREE)
|| TYPE_PRECISION (t1) >= TYPE_PRECISION (integer_type_node)))
return true;
return comptypes (t1, t2);
}
static bool
c_safe_function_type_cast_p (tree t1, tree t2)
{
if (TREE_TYPE (t1) == void_type_node &&
TYPE_ARG_TYPES (t1) == void_list_node)
return true;
if (TREE_TYPE (t2) == void_type_node &&
TYPE_ARG_TYPES (t2) == void_list_node)
return true;
if (!c_safe_arg_type_equiv_p (TREE_TYPE (t1), TREE_TYPE (t2)))
return false;
for (t1 = TYPE_ARG_TYPES (t1), t2 = TYPE_ARG_TYPES (t2);
t1 && t2;
t1 = TREE_CHAIN (t1), t2 = TREE_CHAIN (t2))
if (!c_safe_arg_type_equiv_p (TREE_VALUE (t1), TREE_VALUE (t2)))
return false;
return true;
}
tree
build_c_cast (location_t loc, tree type, tree expr)
{
tree value;
if (TREE_CODE (expr) == EXCESS_PRECISION_EXPR)
expr = TREE_OPERAND (expr, 0);
value = expr;
if (type == error_mark_node || expr == error_mark_node)
return error_mark_node;
if (objc_is_object_ptr (type) && objc_is_object_ptr (TREE_TYPE (expr)))
return build1 (NOP_EXPR, type, expr);
type = TYPE_MAIN_VARIANT (type);
if (TREE_CODE (type) == ARRAY_TYPE)
{
error_at (loc, "cast specifies array type");
return error_mark_node;
}
if (TREE_CODE (type) == FUNCTION_TYPE)
{
error_at (loc, "cast specifies function type");
return error_mark_node;
}
if (!VOID_TYPE_P (type))
{
value = require_complete_type (loc, value);
if (value == error_mark_node)
return error_mark_node;
}
if (type == TYPE_MAIN_VARIANT (TREE_TYPE (value)))
{
if (RECORD_OR_UNION_TYPE_P (type))
pedwarn (loc, OPT_Wpedantic,
"ISO C forbids casting nonscalar to the same type");
value = convert (type, value);
}
else if (TREE_CODE (type) == UNION_TYPE)
{
tree field;
for (field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
if (TREE_TYPE (field) != error_mark_node
&& comptypes (TYPE_MAIN_VARIANT (TREE_TYPE (field)),
TYPE_MAIN_VARIANT (TREE_TYPE (value))))
break;
if (field)
{
tree t;
bool maybe_const = true;
pedwarn (loc, OPT_Wpedantic, "ISO C forbids casts to union type");
t = c_fully_fold (value, false, &maybe_const);
t = build_constructor_single (type, field, t);
if (!maybe_const)
t = c_wrap_maybe_const (t, true);
t = digest_init (loc, type, t,
NULL_TREE, false, true, 0);
TREE_CONSTANT (t) = TREE_CONSTANT (value);
return t;
}
error_at (loc, "cast to union type from type not present in union");
return error_mark_node;
}
else
{
tree otype, ovalue;
if (type == void_type_node)
{
tree t = build1 (CONVERT_EXPR, type, value);
SET_EXPR_LOCATION (t, loc);
return t;
}
otype = TREE_TYPE (value);
if (warn_cast_qual
&& TREE_CODE (type) == POINTER_TYPE
&& TREE_CODE (otype) == POINTER_TYPE)
handle_warn_cast_qual (loc, type, otype);
if (TREE_CODE (type) == POINTER_TYPE
&& TREE_CODE (otype) == POINTER_TYPE
&& !null_pointer_constant_p (value))
{
addr_space_t as_to = TYPE_ADDR_SPACE (TREE_TYPE (type));
addr_space_t as_from = TYPE_ADDR_SPACE (TREE_TYPE (otype));
addr_space_t as_common;
if (!addr_space_superset (as_to, as_from, &as_common))
{
if (ADDR_SPACE_GENERIC_P (as_from))
warning_at (loc, 0, "cast to %s address space pointer "
"from disjoint generic address space pointer",
c_addr_space_name (as_to));
else if (ADDR_SPACE_GENERIC_P (as_to))
warning_at (loc, 0, "cast to generic address space pointer "
"from disjoint %s address space pointer",
c_addr_space_name (as_from));
else
warning_at (loc, 0, "cast to %s address space pointer "
"from disjoint %s address space pointer",
c_addr_space_name (as_to),
c_addr_space_name (as_from));
}
}
if ((STRICT_ALIGNMENT || warn_cast_align == 2)
&& TREE_CODE (type) == POINTER_TYPE
&& TREE_CODE (otype) == POINTER_TYPE
&& TREE_CODE (TREE_TYPE (otype)) != VOID_TYPE
&& TREE_CODE (TREE_TYPE (otype)) != FUNCTION_TYPE
&& !(RECORD_OR_UNION_TYPE_P (TREE_TYPE (otype))
&& TYPE_MODE (TREE_TYPE (otype)) == VOIDmode)
&& min_align_of_type (TREE_TYPE (type))
> min_align_of_type (TREE_TYPE (otype)))
warning_at (loc, OPT_Wcast_align,
"cast increases required alignment of target type");
if (TREE_CODE (type) == INTEGER_TYPE
&& TREE_CODE (otype) == POINTER_TYPE
&& TYPE_PRECISION (type) != TYPE_PRECISION (otype))
warning_at (loc, OPT_Wpointer_to_int_cast,
"cast from pointer to integer of different size");
if (TREE_CODE (value) == CALL_EXPR
&& TREE_CODE (type) != TREE_CODE (otype))
warning_at (loc, OPT_Wbad_function_cast,
"cast from function call of type %qT "
"to non-matching type %qT", otype, type);
if (TREE_CODE (type) == POINTER_TYPE
&& TREE_CODE (otype) == INTEGER_TYPE
&& TYPE_PRECISION (type) != TYPE_PRECISION (otype)
&& !TREE_CONSTANT (value))
warning_at (loc,
OPT_Wint_to_pointer_cast, "cast to pointer from integer "
"of different size");
if (warn_strict_aliasing <= 2)
strict_aliasing_warning (EXPR_LOCATION (value), type, expr);
if (pedantic
&& TREE_CODE (type) == POINTER_TYPE
&& TREE_CODE (otype) == POINTER_TYPE
&& TREE_CODE (TREE_TYPE (otype)) == FUNCTION_TYPE
&& TREE_CODE (TREE_TYPE (type)) != FUNCTION_TYPE)
pedwarn (loc, OPT_Wpedantic, "ISO C forbids "
"conversion of function pointer to object pointer type");
if (pedantic
&& TREE_CODE (type) == POINTER_TYPE
&& TREE_CODE (otype) == POINTER_TYPE
&& TREE_CODE (TREE_TYPE (type)) == FUNCTION_TYPE
&& TREE_CODE (TREE_TYPE (otype)) != FUNCTION_TYPE
&& !null_pointer_constant_p (value))
pedwarn (loc, OPT_Wpedantic, "ISO C forbids "
"conversion of object pointer to function pointer type");
if (TREE_CODE (type) == POINTER_TYPE
&& TREE_CODE (otype) == POINTER_TYPE
&& TREE_CODE (TREE_TYPE (type)) == FUNCTION_TYPE
&& TREE_CODE (TREE_TYPE (otype)) == FUNCTION_TYPE
&& !c_safe_function_type_cast_p (TREE_TYPE (type),
TREE_TYPE (otype)))
warning_at (loc, OPT_Wcast_function_type,
"cast between incompatible function types"
" from %qT to %qT", otype, type);
ovalue = value;
value = convert (type, value);
if (TREE_CODE (value) == INTEGER_CST && !FLOAT_TYPE_P (otype))
{
if (CONSTANT_CLASS_P (ovalue) && TREE_OVERFLOW (ovalue))
{
if (!TREE_OVERFLOW (value))
{
value = copy_node (value);
TREE_OVERFLOW (value) = TREE_OVERFLOW (ovalue);
}
}
else if (TREE_OVERFLOW (value))
value = wide_int_to_tree (TREE_TYPE (value), wi::to_wide (value));
}
}
if (lvalue_p (value))
value = non_lvalue_loc (loc, value);
if (TREE_CODE (value) == REAL_CST
|| TREE_CODE (value) == COMPLEX_CST
|| (TREE_CODE (value) == INTEGER_CST
&& !((TREE_CODE (expr) == INTEGER_CST
&& INTEGRAL_TYPE_P (TREE_TYPE (expr)))
|| TREE_CODE (expr) == REAL_CST
|| TREE_CODE (expr) == COMPLEX_CST)))
value = build1 (NOP_EXPR, type, value);
protected_set_expr_location (value, loc);
return value;
}
tree
c_cast_expr (location_t loc, struct c_type_name *type_name, tree expr)
{
tree type;
tree type_expr = NULL_TREE;
bool type_expr_const = true;
tree ret;
int saved_wsp = warn_strict_prototypes;
if (TREE_CODE (expr) == INTEGER_CST)
warn_strict_prototypes = 0;
type = groktypename (type_name, &type_expr, &type_expr_const);
warn_strict_prototypes = saved_wsp;
if (TREE_CODE (expr) == ADDR_EXPR && !VOID_TYPE_P (type)
&& reject_gcc_builtin (expr))
return error_mark_node;
ret = build_c_cast (loc, type, expr);
if (type_expr)
{
bool inner_expr_const = true;
ret = c_fully_fold (ret, require_constant_value, &inner_expr_const);
ret = build2 (C_MAYBE_CONST_EXPR, TREE_TYPE (ret), type_expr, ret);
C_MAYBE_CONST_EXPR_NON_CONST (ret) = !(type_expr_const
&& inner_expr_const);
SET_EXPR_LOCATION (ret, loc);
}
if (!EXPR_HAS_LOCATION (ret))
protected_set_expr_location (ret, loc);
if (warn_cxx_compat && type_name->specs->typespec_kind == ctsk_tagdef)
warning_at (loc, OPT_Wc___compat,
"defining a type in a cast is invalid in C++");
return ret;
}

tree
build_modify_expr (location_t location, tree lhs, tree lhs_origtype,
enum tree_code modifycode,
location_t rhs_loc, tree rhs, tree rhs_origtype)
{
tree result;
tree newrhs;
tree rhseval = NULL_TREE;
tree lhstype = TREE_TYPE (lhs);
tree olhstype = lhstype;
bool npc;
bool is_atomic_op;
lhs = require_complete_type (location, lhs);
if (TREE_CODE (lhs) == ERROR_MARK || TREE_CODE (rhs) == ERROR_MARK)
return error_mark_node;
if (TREE_CODE (lhstype) == ARRAY_TYPE)
{
error_at (location, "assignment to expression with array type");
return error_mark_node;
}
if (!objc_is_property_ref (lhs) && !lvalue_or_else (location, lhs, lv_assign))
return error_mark_node;
is_atomic_op = really_atomic_lvalue (lhs);
newrhs = rhs;
if (TREE_CODE (lhs) == C_MAYBE_CONST_EXPR)
{
tree inner = build_modify_expr (location, C_MAYBE_CONST_EXPR_EXPR (lhs),
lhs_origtype, modifycode, rhs_loc, rhs,
rhs_origtype);
if (inner == error_mark_node)
return error_mark_node;
result = build2 (C_MAYBE_CONST_EXPR, TREE_TYPE (inner),
C_MAYBE_CONST_EXPR_PRE (lhs), inner);
gcc_assert (!C_MAYBE_CONST_EXPR_INT_OPERANDS (lhs));
C_MAYBE_CONST_EXPR_NON_CONST (result) = 1;
protected_set_expr_location (result, location);
return result;
}
if (modifycode != NOP_EXPR)
{
lhs = c_fully_fold (lhs, false, NULL, true);
lhs = stabilize_reference (lhs);
if (!is_atomic_op)
{
if (TREE_SIDE_EFFECTS (rhs))
{
if (TREE_CODE (rhs) == EXCESS_PRECISION_EXPR)
newrhs = save_expr (TREE_OPERAND (rhs, 0));
else
newrhs = save_expr (rhs);
rhseval = newrhs;
if (TREE_CODE (rhs) == EXCESS_PRECISION_EXPR)
newrhs = build1 (EXCESS_PRECISION_EXPR, TREE_TYPE (rhs),
newrhs);
}
newrhs = build_binary_op (location,
modifycode, lhs, newrhs, true);
rhs_origtype = NULL_TREE;
}
}
if (c_dialect_objc ())
{
if (TREE_CODE (newrhs) == EXCESS_PRECISION_EXPR)
result = objc_maybe_build_modify_expr (lhs, TREE_OPERAND (newrhs, 0));
else
result = objc_maybe_build_modify_expr (lhs, newrhs);
if (result)
goto return_result;
if (!lvalue_or_else (location, lhs, lv_assign))
return error_mark_node;
}
if (TYPE_READONLY (lhstype)
|| (RECORD_OR_UNION_TYPE_P (lhstype)
&& C_TYPE_FIELDS_READONLY (lhstype)))
{
readonly_error (location, lhs, lv_assign);
return error_mark_node;
}
else if (TREE_READONLY (lhs))
readonly_warning (lhs, lv_assign);
if (TREE_CODE (lhs) == COMPONENT_REF
&& (TREE_CODE (lhstype) == INTEGER_TYPE
|| TREE_CODE (lhstype) == BOOLEAN_TYPE
|| TREE_CODE (lhstype) == REAL_TYPE
|| TREE_CODE (lhstype) == ENUMERAL_TYPE))
lhstype = TREE_TYPE (get_unwidened (lhs, 0));
if (lhstype != TREE_TYPE (lhs))
{
lhs = copy_node (lhs);
TREE_TYPE (lhs) = lhstype;
}
if (warn_cxx_compat
&& lhs_origtype != NULL_TREE
&& lhs_origtype != lhstype
&& TREE_CODE (lhs_origtype) == ENUMERAL_TYPE)
{
tree checktype = (rhs_origtype != NULL_TREE
? rhs_origtype
: TREE_TYPE (rhs));
if (checktype != error_mark_node
&& (TYPE_MAIN_VARIANT (checktype) != TYPE_MAIN_VARIANT (lhs_origtype)
|| (is_atomic_op && modifycode != NOP_EXPR)))
warning_at (location, OPT_Wc___compat,
"enum conversion in assignment is invalid in C++");
}
if (is_atomic_op)
{
lhstype = build_qualified_type (lhstype, 
(TYPE_QUALS (lhstype)
& ~TYPE_QUAL_ATOMIC));
olhstype = build_qualified_type (olhstype, 
(TYPE_QUALS (lhstype)
& ~TYPE_QUAL_ATOMIC));
}
if (!(is_atomic_op && modifycode != NOP_EXPR))
{
tree rhs_semantic_type = NULL_TREE;
if (TREE_CODE (newrhs) == EXCESS_PRECISION_EXPR)
{
rhs_semantic_type = TREE_TYPE (newrhs);
newrhs = TREE_OPERAND (newrhs, 0);
}
npc = null_pointer_constant_p (newrhs);
newrhs = c_fully_fold (newrhs, false, NULL);
if (rhs_semantic_type)
newrhs = build1 (EXCESS_PRECISION_EXPR, rhs_semantic_type, newrhs);
newrhs = convert_for_assignment (location, rhs_loc, lhstype, newrhs,
rhs_origtype, ic_assign, npc,
NULL_TREE, NULL_TREE, 0);
if (TREE_CODE (newrhs) == ERROR_MARK)
return error_mark_node;
}
if (c_dialect_objc () && flag_objc_gc)
{
result = objc_generate_write_barrier (lhs, modifycode, newrhs);
if (result)
{
protected_set_expr_location (result, location);
goto return_result;
}
}
if (is_atomic_op)
result = build_atomic_assign (location, lhs, modifycode, newrhs, false);
else
{
result = build2 (MODIFY_EXPR, lhstype, lhs, newrhs);
TREE_SIDE_EFFECTS (result) = 1;
protected_set_expr_location (result, location);
}
if (olhstype == TREE_TYPE (result))
goto return_result;
result = convert_for_assignment (location, rhs_loc, olhstype, result,
rhs_origtype, ic_assign, false, NULL_TREE,
NULL_TREE, 0);
protected_set_expr_location (result, location);
return_result:
if (rhseval)
result = build2 (COMPOUND_EXPR, TREE_TYPE (result), rhseval, result);
return result;
}

static bool
find_anonymous_field_with_type (tree struct_type, tree type)
{
tree field;
bool found;
gcc_assert (RECORD_OR_UNION_TYPE_P (struct_type));
found = false;
for (field = TYPE_FIELDS (struct_type);
field != NULL_TREE;
field = TREE_CHAIN (field))
{
tree fieldtype = (TYPE_ATOMIC (TREE_TYPE (field))
? c_build_qualified_type (TREE_TYPE (field),
TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (TREE_TYPE (field)));
if (DECL_NAME (field) == NULL
&& comptypes (type, fieldtype))
{
if (found)
return false;
found = true;
}
else if (DECL_NAME (field) == NULL
&& RECORD_OR_UNION_TYPE_P (TREE_TYPE (field))
&& find_anonymous_field_with_type (TREE_TYPE (field), type))
{
if (found)
return false;
found = true;
}
}
return found;
}
static tree
convert_to_anonymous_field (location_t location, tree type, tree rhs)
{
tree rhs_struct_type, lhs_main_type;
tree field, found_field;
bool found_sub_field;
tree ret;
gcc_assert (POINTER_TYPE_P (TREE_TYPE (rhs)));
rhs_struct_type = TREE_TYPE (TREE_TYPE (rhs));
gcc_assert (RECORD_OR_UNION_TYPE_P (rhs_struct_type));
gcc_assert (POINTER_TYPE_P (type));
lhs_main_type = (TYPE_ATOMIC (TREE_TYPE (type))
? c_build_qualified_type (TREE_TYPE (type),
TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (TREE_TYPE (type)));
found_field = NULL_TREE;
found_sub_field = false;
for (field = TYPE_FIELDS (rhs_struct_type);
field != NULL_TREE;
field = TREE_CHAIN (field))
{
if (DECL_NAME (field) != NULL_TREE
|| !RECORD_OR_UNION_TYPE_P (TREE_TYPE (field)))
continue;
tree fieldtype = (TYPE_ATOMIC (TREE_TYPE (field))
? c_build_qualified_type (TREE_TYPE (field),
TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (TREE_TYPE (field)));
if (comptypes (lhs_main_type, fieldtype))
{
if (found_field != NULL_TREE)
return NULL_TREE;
found_field = field;
}
else if (find_anonymous_field_with_type (TREE_TYPE (field),
lhs_main_type))
{
if (found_field != NULL_TREE)
return NULL_TREE;
found_field = field;
found_sub_field = true;
}
}
if (found_field == NULL_TREE)
return NULL_TREE;
ret = fold_build3_loc (location, COMPONENT_REF, TREE_TYPE (found_field),
build_fold_indirect_ref (rhs), found_field,
NULL_TREE);
ret = build_fold_addr_expr_loc (location, ret);
if (found_sub_field)
{
ret = convert_to_anonymous_field (location, type, ret);
gcc_assert (ret != NULL_TREE);
}
return ret;
}
static void
error_init (location_t loc, const char *gmsgid)
{
char *ofwhat;
error_at (loc, gmsgid);
ofwhat = print_spelling ((char *) alloca (spelling_length () + 1));
if (*ofwhat)
inform (loc, "(near initialization for %qs)", ofwhat);
}
static void ATTRIBUTE_GCC_DIAG (3,0)
pedwarn_init (location_t loc, int opt, const char *gmsgid, ...)
{
source_location exploc = expansion_point_location_if_in_system_header (loc);
va_list ap;
va_start (ap, gmsgid);
bool warned = emit_diagnostic_valist (DK_PEDWARN, exploc, opt, gmsgid, &ap);
va_end (ap);
char *ofwhat = print_spelling ((char *) alloca (spelling_length () + 1));
if (*ofwhat && warned)
inform (exploc, "(near initialization for %qs)", ofwhat);
}
static void
warning_init (location_t loc, int opt, const char *gmsgid)
{
char *ofwhat;
bool warned;
source_location exploc = expansion_point_location_if_in_system_header (loc);
warned = warning_at (exploc, opt, gmsgid);
ofwhat = print_spelling ((char *) alloca (spelling_length () + 1));
if (*ofwhat && warned)
inform (exploc, "(near initialization for %qs)", ofwhat);
}

void
maybe_warn_string_init (location_t loc, tree type, struct c_expr expr)
{
if (pedantic
&& TREE_CODE (type) == ARRAY_TYPE
&& TREE_CODE (expr.value) == STRING_CST
&& expr.original_code != STRING_CST)
pedwarn_init (loc, OPT_Wpedantic,
"array initialized from parenthesized string constant");
}
static location_t
get_fndecl_argument_location (tree fndecl, int argnum)
{
int i;
tree param;
for (i = 0, param = DECL_ARGUMENTS (fndecl);
i < argnum && param;
i++, param = TREE_CHAIN (param))
;
if (param == NULL)
return DECL_SOURCE_LOCATION (fndecl);
return DECL_SOURCE_LOCATION (param);
}
static void
inform_for_arg (tree fundecl, location_t ploc, int parmnum,
tree expected_type, tree actual_type)
{
location_t loc;
if (fundecl && !DECL_IS_BUILTIN (fundecl))
loc = get_fndecl_argument_location (fundecl, parmnum - 1);
else
loc = ploc;
inform (loc,
"expected %qT but argument is of type %qT",
expected_type, actual_type);
}
static tree
convert_for_assignment (location_t location, location_t expr_loc, tree type,
tree rhs, tree origtype, enum impl_conv errtype,
bool null_pointer_constant, tree fundecl,
tree function, int parmnum)
{
enum tree_code codel = TREE_CODE (type);
tree orig_rhs = rhs;
tree rhstype;
enum tree_code coder;
tree rname = NULL_TREE;
bool objc_ok = false;
location = expansion_point_location_if_in_system_header (location);
if (errtype == ic_argpass)
{
tree selector;
if (TREE_CODE (function) == ADDR_EXPR
&& TREE_CODE (TREE_OPERAND (function, 0)) == FUNCTION_DECL)
function = TREE_OPERAND (function, 0);
selector = objc_message_selector ();
rname = function;
if (selector && parmnum > 2)
{
rname = selector;
parmnum -= 2;
}
}
#define PEDWARN_FOR_ASSIGNMENT(LOCATION, PLOC, OPT, AR, AS, IN, RE)	 \
do {                                                                   \
switch (errtype)                                                     \
{                                                                  \
case ic_argpass:                                                   \
if (pedwarn (PLOC, OPT, AR, parmnum, rname))			 \
inform_for_arg (fundecl, (PLOC), parmnum, type, rhstype);	\
break;                                                           \
case ic_assign:                                                    \
pedwarn (LOCATION, OPT, AS);                                     \
break;                                                           \
case ic_init:                                                      \
pedwarn_init (LOCATION, OPT, IN);                                \
break;                                                           \
case ic_return:                                                    \
pedwarn (LOCATION, OPT, RE);					 \
break;                                                           \
default:                                                           \
gcc_unreachable ();                                              \
}                                                                  \
} while (0)
#define PEDWARN_FOR_QUALIFIERS(LOCATION, PLOC, OPT, AR, AS, IN, RE, QUALS) \
do {                                                                   \
switch (errtype)                                                     \
{                                                                  \
case ic_argpass:                                                   \
if (pedwarn (PLOC, OPT, AR, parmnum, rname, QUALS))		 \
inform_for_arg (fundecl, (PLOC), parmnum, type, rhstype);	\
break;                                                           \
case ic_assign:                                                    \
pedwarn (LOCATION, OPT, AS, QUALS);				 \
break;                                                           \
case ic_init:                                                      \
pedwarn (LOCATION, OPT, IN, QUALS);				 \
break;                                                           \
case ic_return:                                                    \
pedwarn (LOCATION, OPT, RE, QUALS);				 \
break;                                                           \
default:                                                           \
gcc_unreachable ();                                              \
}                                                                  \
} while (0)
#define WARNING_FOR_QUALIFIERS(LOCATION, PLOC, OPT, AR, AS, IN, RE, QUALS) \
do {                                                                   \
switch (errtype)                                                     \
{                                                                  \
case ic_argpass:                                                   \
if (warning_at (PLOC, OPT, AR, parmnum, rname, QUALS))           \
inform_for_arg (fundecl, (PLOC), parmnum, type, rhstype);      \
break;                                                           \
case ic_assign:                                                    \
warning_at (LOCATION, OPT, AS, QUALS);                           \
break;                                                           \
case ic_init:                                                      \
warning_at (LOCATION, OPT, IN, QUALS);                           \
break;                                                           \
case ic_return:                                                    \
warning_at (LOCATION, OPT, RE, QUALS);                           \
break;                                                           \
default:                                                           \
gcc_unreachable ();                                              \
}                                                                  \
} while (0)
if (TREE_CODE (rhs) == EXCESS_PRECISION_EXPR)
rhs = TREE_OPERAND (rhs, 0);
rhstype = TREE_TYPE (rhs);
coder = TREE_CODE (rhstype);
if (coder == ERROR_MARK)
return error_mark_node;
if (c_dialect_objc ())
{
int parmno;
switch (errtype)
{
case ic_return:
parmno = 0;
break;
case ic_assign:
parmno = -1;
break;
case ic_init:
parmno = -2;
break;
default:
parmno = parmnum;
break;
}
objc_ok = objc_compare_types (type, rhstype, parmno, rname);
}
if (warn_cxx_compat)
{
tree checktype = origtype != NULL_TREE ? origtype : rhstype;
if (checktype != error_mark_node
&& TREE_CODE (type) == ENUMERAL_TYPE
&& TYPE_MAIN_VARIANT (checktype) != TYPE_MAIN_VARIANT (type))
switch (errtype)
{
case ic_argpass:
if (pedwarn (expr_loc, OPT_Wc___compat, "enum conversion when "
"passing argument %d of %qE is invalid in C++",
parmnum, rname))
inform ((fundecl && !DECL_IS_BUILTIN (fundecl))
? DECL_SOURCE_LOCATION (fundecl) : expr_loc,
"expected %qT but argument is of type %qT",
type, rhstype);
break;
case ic_assign:
pedwarn (location, OPT_Wc___compat, "enum conversion from %qT to "
"%qT in assignment is invalid in C++", rhstype, type);
break;
case ic_init:
pedwarn_init (location, OPT_Wc___compat, "enum conversion from "
"%qT to %qT in initialization is invalid in C++",
rhstype, type);
break;
case ic_return:
pedwarn (location, OPT_Wc___compat, "enum conversion from %qT to "
"%qT in return is invalid in C++", rhstype, type);
break;
default:
gcc_unreachable ();
}
}
if (TYPE_MAIN_VARIANT (type) == TYPE_MAIN_VARIANT (rhstype))
return rhs;
if (coder == VOID_TYPE)
{
error_at (location, "void value not ignored as it ought to be");
return error_mark_node;
}
rhs = require_complete_type (location, rhs);
if (rhs == error_mark_node)
return error_mark_node;
if (coder == POINTER_TYPE && reject_gcc_builtin (rhs))
return error_mark_node;
if (codel == REFERENCE_TYPE && coder != REFERENCE_TYPE)
{
if (!lvalue_p (rhs))
{
error_at (location, "cannot pass rvalue to reference parameter");
return error_mark_node;
}
if (!c_mark_addressable (rhs))
return error_mark_node;
rhs = build1 (ADDR_EXPR, build_pointer_type (TREE_TYPE (rhs)), rhs);
SET_EXPR_LOCATION (rhs, location);
rhs = convert_for_assignment (location, expr_loc,
build_pointer_type (TREE_TYPE (type)),
rhs, origtype, errtype,
null_pointer_constant, fundecl, function,
parmnum);
if (rhs == error_mark_node)
return error_mark_node;
rhs = build1 (NOP_EXPR, type, rhs);
SET_EXPR_LOCATION (rhs, location);
return rhs;
}
else if (codel == VECTOR_TYPE && coder == VECTOR_TYPE
&& vector_types_convertible_p (type, TREE_TYPE (rhs), true))
return convert (type, rhs);
else if ((codel == INTEGER_TYPE || codel == REAL_TYPE
|| codel == FIXED_POINT_TYPE
|| codel == ENUMERAL_TYPE || codel == COMPLEX_TYPE
|| codel == BOOLEAN_TYPE)
&& (coder == INTEGER_TYPE || coder == REAL_TYPE
|| coder == FIXED_POINT_TYPE
|| coder == ENUMERAL_TYPE || coder == COMPLEX_TYPE
|| coder == BOOLEAN_TYPE))
{
tree ret;
bool save = in_late_binary_op;
if (codel == BOOLEAN_TYPE || codel == COMPLEX_TYPE
|| (coder == REAL_TYPE
&& (codel == INTEGER_TYPE || codel == ENUMERAL_TYPE)
&& sanitize_flags_p (SANITIZE_FLOAT_CAST)))
in_late_binary_op = true;
ret = convert_and_check (expr_loc != UNKNOWN_LOCATION
? expr_loc : location, type, orig_rhs);
in_late_binary_op = save;
return ret;
}
if ((codel == RECORD_TYPE || codel == UNION_TYPE)
&& codel == coder
&& comptypes (type, rhstype))
return convert_and_check (expr_loc != UNKNOWN_LOCATION
? expr_loc : location, type, rhs);
if (((codel == UNION_TYPE || codel == RECORD_TYPE)
&& TYPE_TRANSPARENT_AGGR (type))
&& errtype == ic_argpass)
{
tree memb, marginal_memb = NULL_TREE;
for (memb = TYPE_FIELDS (type); memb ; memb = DECL_CHAIN (memb))
{
tree memb_type = TREE_TYPE (memb);
if (comptypes (TYPE_MAIN_VARIANT (memb_type),
TYPE_MAIN_VARIANT (rhstype)))
break;
if (TREE_CODE (memb_type) != POINTER_TYPE)
continue;
if (coder == POINTER_TYPE)
{
tree ttl = TREE_TYPE (memb_type);
tree ttr = TREE_TYPE (rhstype);
if ((VOID_TYPE_P (ttl) && !TYPE_ATOMIC (ttl))
|| (VOID_TYPE_P (ttr) && !TYPE_ATOMIC (ttr))
|| comp_target_types (location, memb_type, rhstype))
{
int lquals = TYPE_QUALS (ttl) & ~TYPE_QUAL_ATOMIC;
int rquals = TYPE_QUALS (ttr) & ~TYPE_QUAL_ATOMIC;
if (lquals == rquals
|| ((TREE_CODE (ttr) == FUNCTION_TYPE
&& TREE_CODE (ttl) == FUNCTION_TYPE)
? ((lquals | rquals) == rquals)
: ((lquals | rquals) == lquals)))
break;
if (!marginal_memb)
marginal_memb = memb;
}
}
if (null_pointer_constant)
{
rhs = null_pointer_node;
break;
}
}
if (memb || marginal_memb)
{
if (!memb)
{
tree ttl = TREE_TYPE (TREE_TYPE (marginal_memb));
tree ttr = TREE_TYPE (rhstype);
if (TREE_CODE (ttr) == FUNCTION_TYPE
&& TREE_CODE (ttl) == FUNCTION_TYPE)
{
if (TYPE_QUALS_NO_ADDR_SPACE (ttl)
& ~TYPE_QUALS_NO_ADDR_SPACE (ttr))
PEDWARN_FOR_QUALIFIERS (location, expr_loc,
OPT_Wdiscarded_qualifiers,
G_("passing argument %d of %qE "
"makes %q#v qualified function "
"pointer from unqualified"),
G_("assignment makes %q#v qualified "
"function pointer from "
"unqualified"),
G_("initialization makes %q#v qualified "
"function pointer from "
"unqualified"),
G_("return makes %q#v qualified function "
"pointer from unqualified"),
TYPE_QUALS (ttl) & ~TYPE_QUALS (ttr));
}
else if (TYPE_QUALS_NO_ADDR_SPACE (ttr)
& ~TYPE_QUALS_NO_ADDR_SPACE (ttl))
PEDWARN_FOR_QUALIFIERS (location, expr_loc,
OPT_Wdiscarded_qualifiers,
G_("passing argument %d of %qE discards "
"%qv qualifier from pointer target type"),
G_("assignment discards %qv qualifier "
"from pointer target type"),
G_("initialization discards %qv qualifier "
"from pointer target type"),
G_("return discards %qv qualifier from "
"pointer target type"),
TYPE_QUALS (ttr) & ~TYPE_QUALS (ttl));
memb = marginal_memb;
}
if (!fundecl || !DECL_IN_SYSTEM_HEADER (fundecl))
pedwarn (location, OPT_Wpedantic,
"ISO C prohibits argument conversion to union type");
rhs = fold_convert_loc (location, TREE_TYPE (memb), rhs);
return build_constructor_single (type, memb, rhs);
}
}
else if ((codel == POINTER_TYPE || codel == REFERENCE_TYPE)
&& (coder == codel))
{
tree ttl = TREE_TYPE (type);
tree ttr = TREE_TYPE (rhstype);
tree mvl = ttl;
tree mvr = ttr;
bool is_opaque_pointer;
int target_cmp = 0;   
addr_space_t asl;
addr_space_t asr;
if (TREE_CODE (mvl) != ARRAY_TYPE)
mvl = (TYPE_ATOMIC (mvl)
? c_build_qualified_type (TYPE_MAIN_VARIANT (mvl),
TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (mvl));
if (TREE_CODE (mvr) != ARRAY_TYPE)
mvr = (TYPE_ATOMIC (mvr)
? c_build_qualified_type (TYPE_MAIN_VARIANT (mvr),
TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (mvr));
is_opaque_pointer = vector_targets_convertible_p (ttl, ttr);
if (flag_plan9_extensions
&& RECORD_OR_UNION_TYPE_P (mvl)
&& RECORD_OR_UNION_TYPE_P (mvr)
&& mvl != mvr)
{
tree new_rhs = convert_to_anonymous_field (location, type, rhs);
if (new_rhs != NULL_TREE)
{
rhs = new_rhs;
rhstype = TREE_TYPE (rhs);
coder = TREE_CODE (rhstype);
ttr = TREE_TYPE (rhstype);
mvr = TYPE_MAIN_VARIANT (ttr);
}
}
if (VOID_TYPE_P (ttr) && rhs != null_pointer_node && !VOID_TYPE_P (ttl))
warning_at (errtype == ic_argpass ? expr_loc : location,
OPT_Wc___compat,
"request for implicit conversion "
"from %qT to %qT not permitted in C++", rhstype, type);
asl = TYPE_ADDR_SPACE (ttl);
asr = TYPE_ADDR_SPACE (ttr);
if (!null_pointer_constant_p (rhs)
&& asr != asl && !targetm.addr_space.subset_p (asr, asl))
{
switch (errtype)
{
case ic_argpass:
error_at (expr_loc, "passing argument %d of %qE from pointer to "
"non-enclosed address space", parmnum, rname);
break;
case ic_assign:
error_at (location, "assignment from pointer to "
"non-enclosed address space");
break;
case ic_init:
error_at (location, "initialization from pointer to "
"non-enclosed address space");
break;
case ic_return:
error_at (location, "return from pointer to "
"non-enclosed address space");
break;
default:
gcc_unreachable ();
}
return error_mark_node;
}
if (warn_suggest_attribute_format
&& check_missing_format_attribute (type, rhstype))
{
switch (errtype)
{
case ic_argpass:
warning_at (expr_loc, OPT_Wsuggest_attribute_format,
"argument %d of %qE might be "
"a candidate for a format attribute",
parmnum, rname);
break;
case ic_assign:
warning_at (location, OPT_Wsuggest_attribute_format,
"assignment left-hand side might be "
"a candidate for a format attribute");
break;
case ic_init:
warning_at (location, OPT_Wsuggest_attribute_format,
"initialization left-hand side might be "
"a candidate for a format attribute");
break;
case ic_return:
warning_at (location, OPT_Wsuggest_attribute_format,
"return type might be "
"a candidate for a format attribute");
break;
default:
gcc_unreachable ();
}
}
if ((VOID_TYPE_P (ttl) && !TYPE_ATOMIC (ttl))
|| (VOID_TYPE_P (ttr) && !TYPE_ATOMIC (ttr))
|| (target_cmp = comp_target_types (location, type, rhstype))
|| is_opaque_pointer
|| ((c_common_unsigned_type (mvl)
== c_common_unsigned_type (mvr))
&& (c_common_signed_type (mvl)
== c_common_signed_type (mvr))
&& TYPE_ATOMIC (mvl) == TYPE_ATOMIC (mvr)))
{
if (TREE_CODE (ttr) == ARRAY_TYPE)
{
ttr = strip_array_types (ttr);
ttl = strip_array_types (ttl);
if (TYPE_QUALS_NO_ADDR_SPACE_NO_ATOMIC (ttr)
& ~TYPE_QUALS_NO_ADDR_SPACE_NO_ATOMIC (ttl))
WARNING_FOR_QUALIFIERS (location, expr_loc,
OPT_Wdiscarded_array_qualifiers,
G_("passing argument %d of %qE discards "
"%qv qualifier from pointer target type"),
G_("assignment discards %qv qualifier "
"from pointer target type"),
G_("initialization discards %qv qualifier "
"from pointer target type"),
G_("return discards %qv qualifier from "
"pointer target type"),
TYPE_QUALS (ttr) & ~TYPE_QUALS (ttl));
}
else if (pedantic
&& ((VOID_TYPE_P (ttl) && TREE_CODE (ttr) == FUNCTION_TYPE)
||
(VOID_TYPE_P (ttr)
&& !null_pointer_constant
&& TREE_CODE (ttl) == FUNCTION_TYPE)))
PEDWARN_FOR_ASSIGNMENT (location, expr_loc, OPT_Wpedantic,
G_("ISO C forbids passing argument %d of "
"%qE between function pointer "
"and %<void *%>"),
G_("ISO C forbids assignment between "
"function pointer and %<void *%>"),
G_("ISO C forbids initialization between "
"function pointer and %<void *%>"),
G_("ISO C forbids return between function "
"pointer and %<void *%>"));
else if (TREE_CODE (ttr) != FUNCTION_TYPE
&& TREE_CODE (ttl) != FUNCTION_TYPE)
{
if (!pedantic)
ttl = strip_array_types (ttl);
if (TYPE_QUALS_NO_ADDR_SPACE_NO_ATOMIC (ttr)
& ~TYPE_QUALS_NO_ADDR_SPACE_NO_ATOMIC (ttl))
{
PEDWARN_FOR_QUALIFIERS (location, expr_loc,
OPT_Wdiscarded_qualifiers,
G_("passing argument %d of %qE discards "
"%qv qualifier from pointer target type"),
G_("assignment discards %qv qualifier "
"from pointer target type"),
G_("initialization discards %qv qualifier "
"from pointer target type"),
G_("return discards %qv qualifier from "
"pointer target type"),
TYPE_QUALS (ttr) & ~TYPE_QUALS (ttl));
}
else if (VOID_TYPE_P (ttl) || VOID_TYPE_P (ttr)
|| target_cmp)
;
else if (warn_pointer_sign)
switch (errtype)
{
case ic_argpass:
if (pedwarn (expr_loc, OPT_Wpointer_sign,
"pointer targets in passing argument %d of "
"%qE differ in signedness", parmnum, rname))
inform ((fundecl && !DECL_IS_BUILTIN (fundecl))
? DECL_SOURCE_LOCATION (fundecl) : expr_loc,
"expected %qT but argument is of type %qT",
type, rhstype);
break;
case ic_assign:
pedwarn (location, OPT_Wpointer_sign,
"pointer targets in assignment from %qT to %qT "
"differ in signedness", rhstype, type);
break;
case ic_init:
pedwarn_init (location, OPT_Wpointer_sign,
"pointer targets in initialization of %qT "
"from %qT differ in signedness", type,
rhstype);
break;
case ic_return:
pedwarn (location, OPT_Wpointer_sign, "pointer targets in "
"returning %qT from a function with return type "
"%qT differ in signedness", rhstype, type);
break;
default:
gcc_unreachable ();
}
}
else if (TREE_CODE (ttl) == FUNCTION_TYPE
&& TREE_CODE (ttr) == FUNCTION_TYPE)
{
if (TYPE_QUALS_NO_ADDR_SPACE (ttl)
& ~TYPE_QUALS_NO_ADDR_SPACE (ttr))
PEDWARN_FOR_QUALIFIERS (location, expr_loc,
OPT_Wdiscarded_qualifiers,
G_("passing argument %d of %qE makes "
"%q#v qualified function pointer "
"from unqualified"),
G_("assignment makes %q#v qualified function "
"pointer from unqualified"),
G_("initialization makes %q#v qualified "
"function pointer from unqualified"),
G_("return makes %q#v qualified function "
"pointer from unqualified"),
TYPE_QUALS (ttl) & ~TYPE_QUALS (ttr));
}
}
else if (!objc_ok)
{
switch (errtype)
{
case ic_argpass:
if (pedwarn (expr_loc, OPT_Wincompatible_pointer_types,
"passing argument %d of %qE from incompatible "
"pointer type", parmnum, rname))
inform_for_arg (fundecl, expr_loc, parmnum, type, rhstype);
break;
case ic_assign:
pedwarn (location, OPT_Wincompatible_pointer_types,
"assignment to %qT from incompatible pointer type %qT",
type, rhstype);
break;
case ic_init:
pedwarn_init (location, OPT_Wincompatible_pointer_types,
"initialization of %qT from incompatible pointer "
"type %qT", type, rhstype);
break;
case ic_return:
pedwarn (location, OPT_Wincompatible_pointer_types,
"returning %qT from a function with incompatible "
"return type %qT", rhstype, type);
break;
default:
gcc_unreachable ();
}
}
return convert (type, rhs);
}
else if (codel == POINTER_TYPE && coder == ARRAY_TYPE)
{
error_at (location, "invalid use of non-lvalue array");
return error_mark_node;
}
else if (codel == POINTER_TYPE && coder == INTEGER_TYPE)
{
if (!null_pointer_constant)
switch (errtype)
{
case ic_argpass:
if (pedwarn (expr_loc, OPT_Wint_conversion,
"passing argument %d of %qE makes pointer from "
"integer without a cast", parmnum, rname))
inform_for_arg (fundecl, expr_loc, parmnum, type, rhstype);
break;
case ic_assign:
pedwarn (location, OPT_Wint_conversion,
"assignment to %qT from %qT makes pointer from integer "
"without a cast", type, rhstype);
break;
case ic_init:
pedwarn_init (location, OPT_Wint_conversion,
"initialization of %qT from %qT makes pointer from "
"integer without a cast", type, rhstype);
break;
case ic_return:
pedwarn (location, OPT_Wint_conversion, "returning %qT from a "
"function with return type %qT makes pointer from "
"integer without a cast", rhstype, type);
break;
default:
gcc_unreachable ();
}
return convert (type, rhs);
}
else if (codel == INTEGER_TYPE && coder == POINTER_TYPE)
{
switch (errtype)
{
case ic_argpass:
if (pedwarn (expr_loc, OPT_Wint_conversion,
"passing argument %d of %qE makes integer from "
"pointer without a cast", parmnum, rname))
inform_for_arg (fundecl, expr_loc, parmnum, type, rhstype);
break;
case ic_assign:
pedwarn (location, OPT_Wint_conversion,
"assignment to %qT from %qT makes integer from pointer "
"without a cast", type, rhstype);
break;
case ic_init:
pedwarn_init (location, OPT_Wint_conversion,
"initialization of %qT from %qT makes integer from "
"pointer without a cast", type, rhstype);
break;
case ic_return:
pedwarn (location, OPT_Wint_conversion, "returning %qT from a "
"function with return type %qT makes integer from "
"pointer without a cast", rhstype, type);
break;
default:
gcc_unreachable ();
}
return convert (type, rhs);
}
else if (codel == BOOLEAN_TYPE && coder == POINTER_TYPE)
{
tree ret;
bool save = in_late_binary_op;
in_late_binary_op = true;
ret = convert (type, rhs);
in_late_binary_op = save;
return ret;
}
switch (errtype)
{
case ic_argpass:
error_at (expr_loc, "incompatible type for argument %d of %qE", parmnum,
rname);
inform_for_arg (fundecl, expr_loc, parmnum, type, rhstype);
break;
case ic_assign:
error_at (location, "incompatible types when assigning to type %qT from "
"type %qT", type, rhstype);
break;
case ic_init:
error_at (location,
"incompatible types when initializing type %qT using type %qT",
type, rhstype);
break;
case ic_return:
error_at (location,
"incompatible types when returning type %qT but %qT was "
"expected", rhstype, type);
break;
default:
gcc_unreachable ();
}
return error_mark_node;
}

static tree
valid_compound_expr_initializer (tree value, tree endtype)
{
if (TREE_CODE (value) == COMPOUND_EXPR)
{
if (valid_compound_expr_initializer (TREE_OPERAND (value, 0), endtype)
== error_mark_node)
return error_mark_node;
return valid_compound_expr_initializer (TREE_OPERAND (value, 1),
endtype);
}
else if (!initializer_constant_valid_p (value, endtype))
return error_mark_node;
else
return value;
}

void
store_init_value (location_t init_loc, tree decl, tree init, tree origtype)
{
tree value, type;
bool npc = false;
type = TREE_TYPE (decl);
if (TREE_CODE (type) == ERROR_MARK)
return;
if (init)
npc = null_pointer_constant_p (init);
value = digest_init (init_loc, type, init, origtype, npc,
true, TREE_STATIC (decl));
if (!in_system_header_at (input_location)
&& AGGREGATE_TYPE_P (TREE_TYPE (decl)) && !TREE_STATIC (decl))
warning (OPT_Wtraditional, "traditional C rejects automatic "
"aggregate initialization");
if (value != error_mark_node || TREE_CODE (decl) != FUNCTION_DECL)
DECL_INITIAL (decl) = value;
STRIP_TYPE_NOPS (value);
if (TREE_STATIC (decl))
constant_expression_warning (value);
if (TREE_CODE (type) == ARRAY_TYPE
&& TYPE_DOMAIN (type) == NULL_TREE
&& value != error_mark_node)
{
tree inside_init = init;
STRIP_TYPE_NOPS (inside_init);
inside_init = fold (inside_init);
if (TREE_CODE (inside_init) == COMPOUND_LITERAL_EXPR)
{
tree cldecl = COMPOUND_LITERAL_EXPR_DECL (inside_init);
if (TYPE_DOMAIN (TREE_TYPE (cldecl)))
{
tree etype = strip_array_types (TREE_TYPE (decl));
type = build_distinct_type_copy (TYPE_MAIN_VARIANT (type));
TYPE_DOMAIN (type) = TYPE_DOMAIN (TREE_TYPE (cldecl));
layout_type (type);
layout_decl (cldecl, 0);
TREE_TYPE (decl)
= c_build_qualified_type (type, TYPE_QUALS (etype));
}
}
}
}

struct spelling
{
int kind;
union
{
unsigned HOST_WIDE_INT i;
const char *s;
} u;
};
#define SPELLING_STRING 1
#define SPELLING_MEMBER 2
#define SPELLING_BOUNDS 3
static struct spelling *spelling;	
static struct spelling *spelling_base;	
static int spelling_size;		
#define SPELLING_DEPTH() (spelling - spelling_base)
#define RESTORE_SPELLING_DEPTH(DEPTH) (spelling = spelling_base + (DEPTH))
#define PUSH_SPELLING(KIND, VALUE, MEMBER)				\
{									\
int depth = SPELLING_DEPTH ();					\
\
if (depth >= spelling_size)						\
{									\
spelling_size += 10;						\
spelling_base = XRESIZEVEC (struct spelling, spelling_base,	\
spelling_size);			\
RESTORE_SPELLING_DEPTH (depth);					\
}									\
\
spelling->kind = (KIND);						\
spelling->MEMBER = (VALUE);						\
spelling++;								\
}
static void
push_string (const char *string)
{
PUSH_SPELLING (SPELLING_STRING, string, u.s);
}
static void
push_member_name (tree decl)
{
const char *const string
= (DECL_NAME (decl)
? identifier_to_locale (IDENTIFIER_POINTER (DECL_NAME (decl)))
: _("<anonymous>"));
PUSH_SPELLING (SPELLING_MEMBER, string, u.s);
}
static void
push_array_bounds (unsigned HOST_WIDE_INT bounds)
{
PUSH_SPELLING (SPELLING_BOUNDS, bounds, u.i);
}
static int
spelling_length (void)
{
int size = 0;
struct spelling *p;
for (p = spelling_base; p < spelling; p++)
{
if (p->kind == SPELLING_BOUNDS)
size += 25;
else
size += strlen (p->u.s) + 1;
}
return size;
}
static char *
print_spelling (char *buffer)
{
char *d = buffer;
struct spelling *p;
for (p = spelling_base; p < spelling; p++)
if (p->kind == SPELLING_BOUNDS)
{
sprintf (d, "[" HOST_WIDE_INT_PRINT_UNSIGNED "]", p->u.i);
d += strlen (d);
}
else
{
const char *s;
if (p->kind == SPELLING_MEMBER)
*d++ = '.';
for (s = p->u.s; (*d = *s++); d++)
;
}
*d++ = '\0';
return buffer;
}
static tree
digest_init (location_t init_loc, tree type, tree init, tree origtype,
bool null_pointer_constant, bool strict_string,
int require_constant)
{
enum tree_code code = TREE_CODE (type);
tree inside_init = init;
tree semantic_type = NULL_TREE;
bool maybe_const = true;
if (type == error_mark_node
|| !init
|| error_operand_p (init))
return error_mark_node;
STRIP_TYPE_NOPS (inside_init);
if (TREE_CODE (inside_init) == EXCESS_PRECISION_EXPR)
{
semantic_type = TREE_TYPE (inside_init);
inside_init = TREE_OPERAND (inside_init, 0);
}
inside_init = c_fully_fold (inside_init, require_constant, &maybe_const);
if (code == ARRAY_TYPE && inside_init
&& TREE_CODE (inside_init) == STRING_CST)
{
tree typ1
= (TYPE_ATOMIC (TREE_TYPE (type))
? c_build_qualified_type (TYPE_MAIN_VARIANT (TREE_TYPE (type)),
TYPE_QUAL_ATOMIC)
: TYPE_MAIN_VARIANT (TREE_TYPE (type)));
bool char_array = (typ1 == char_type_node
|| typ1 == signed_char_type_node
|| typ1 == unsigned_char_type_node);
bool wchar_array = !!comptypes (typ1, wchar_type_node);
bool char16_array = !!comptypes (typ1, char16_type_node);
bool char32_array = !!comptypes (typ1, char32_type_node);
if (char_array || wchar_array || char16_array || char32_array)
{
struct c_expr expr;
tree typ2 = TYPE_MAIN_VARIANT (TREE_TYPE (TREE_TYPE (inside_init)));
expr.value = inside_init;
expr.original_code = (strict_string ? STRING_CST : ERROR_MARK);
expr.original_type = NULL;
maybe_warn_string_init (init_loc, type, expr);
if (TYPE_DOMAIN (type) && !TYPE_MAX_VALUE (TYPE_DOMAIN (type)))
pedwarn_init (init_loc, OPT_Wpedantic,
"initialization of a flexible array member");
if (comptypes (TYPE_MAIN_VARIANT (TREE_TYPE (inside_init)),
TYPE_MAIN_VARIANT (type)))
return inside_init;
if (char_array)
{
if (typ2 != char_type_node)
{
error_init (init_loc, "char-array initialized from wide "
"string");
return error_mark_node;
}
}
else
{
if (typ2 == char_type_node)
{
error_init (init_loc, "wide character array initialized "
"from non-wide string");
return error_mark_node;
}
else if (!comptypes(typ1, typ2))
{
error_init (init_loc, "wide character array initialized "
"from incompatible wide string");
return error_mark_node;
}
}
TREE_TYPE (inside_init) = type;
if (TYPE_DOMAIN (type) != NULL_TREE
&& TYPE_SIZE (type) != NULL_TREE
&& TREE_CODE (TYPE_SIZE (type)) == INTEGER_CST)
{
unsigned HOST_WIDE_INT len = TREE_STRING_LENGTH (inside_init);
if (compare_tree_int (TYPE_SIZE_UNIT (type),
(len - (TYPE_PRECISION (typ1)
/ BITS_PER_UNIT))) < 0)
pedwarn_init (init_loc, 0,
("initializer-string for array of chars "
"is too long"));
else if (warn_cxx_compat
&& compare_tree_int (TYPE_SIZE_UNIT (type), len) < 0)
warning_at (init_loc, OPT_Wc___compat,
("initializer-string for array chars "
"is too long for C++"));
}
return inside_init;
}
else if (INTEGRAL_TYPE_P (typ1))
{
error_init (init_loc, "array of inappropriate type initialized "
"from string constant");
return error_mark_node;
}
}
if (code == VECTOR_TYPE
&& VECTOR_TYPE_P (TREE_TYPE (inside_init))
&& vector_types_convertible_p (TREE_TYPE (inside_init), type, true)
&& TREE_CONSTANT (inside_init))
{
if (TREE_CODE (inside_init) == VECTOR_CST
&& comptypes (TYPE_MAIN_VARIANT (TREE_TYPE (inside_init)),
TYPE_MAIN_VARIANT (type)))
return inside_init;
if (TREE_CODE (inside_init) == CONSTRUCTOR)
{
unsigned HOST_WIDE_INT ix;
tree value;
bool constant_p = true;
FOR_EACH_CONSTRUCTOR_VALUE (CONSTRUCTOR_ELTS (inside_init), ix, value)
if (!CONSTANT_CLASS_P (value))
{
constant_p = false;
break;
}
if (constant_p)
return build_vector_from_ctor (type,
CONSTRUCTOR_ELTS (inside_init));
}
}
if (warn_sequence_point)
verify_sequence_points (inside_init);
if (inside_init && TREE_TYPE (inside_init) != NULL_TREE
&& (comptypes (TYPE_MAIN_VARIANT (TREE_TYPE (inside_init)),
TYPE_MAIN_VARIANT (type))
|| (code == ARRAY_TYPE
&& comptypes (TREE_TYPE (inside_init), type))
|| (code == VECTOR_TYPE
&& comptypes (TREE_TYPE (inside_init), type))
|| (code == POINTER_TYPE
&& TREE_CODE (TREE_TYPE (inside_init)) == ARRAY_TYPE
&& comptypes (TREE_TYPE (TREE_TYPE (inside_init)),
TREE_TYPE (type)))))
{
if (code == POINTER_TYPE)
{
if (TREE_CODE (TREE_TYPE (inside_init)) == ARRAY_TYPE)
{
if (TREE_CODE (inside_init) == STRING_CST
|| TREE_CODE (inside_init) == COMPOUND_LITERAL_EXPR)
inside_init = array_to_pointer_conversion
(init_loc, inside_init);
else
{
error_init (init_loc, "invalid use of non-lvalue array");
return error_mark_node;
}
}
}
if (code == VECTOR_TYPE)
inside_init = convert (type, inside_init);
if (require_constant
&& TREE_CODE (inside_init) == COMPOUND_LITERAL_EXPR)
{
if (flag_isoc99 && code != VECTOR_TYPE)
pedwarn_init (init_loc, OPT_Wpedantic, "initializer element "
"is not constant");
tree decl = COMPOUND_LITERAL_EXPR_DECL (inside_init);
inside_init = DECL_INITIAL (decl);
}
if (code == ARRAY_TYPE && TREE_CODE (inside_init) != STRING_CST
&& TREE_CODE (inside_init) != CONSTRUCTOR)
{
error_init (init_loc, "array initialized from non-constant array "
"expression");
return error_mark_node;
}
if (require_constant && pedantic
&& TREE_CODE (inside_init) == COMPOUND_EXPR)
{
inside_init
= valid_compound_expr_initializer (inside_init,
TREE_TYPE (inside_init));
if (inside_init == error_mark_node)
error_init (init_loc, "initializer element is not constant");
else
pedwarn_init (init_loc, OPT_Wpedantic,
"initializer element is not constant");
if (flag_pedantic_errors)
inside_init = error_mark_node;
}
else if (require_constant
&& !initializer_constant_valid_p (inside_init,
TREE_TYPE (inside_init)))
{
error_init (init_loc, "initializer element is not constant");
inside_init = error_mark_node;
}
else if (require_constant && !maybe_const)
pedwarn_init (init_loc, OPT_Wpedantic,
"initializer element is not a constant expression");
if (TREE_CODE (TREE_TYPE (inside_init)) == POINTER_TYPE)
inside_init = convert_for_assignment (init_loc, UNKNOWN_LOCATION,
type, inside_init, origtype,
ic_init, null_pointer_constant,
NULL_TREE, NULL_TREE, 0);
return inside_init;
}
if (code == INTEGER_TYPE || code == REAL_TYPE || code == FIXED_POINT_TYPE
|| code == POINTER_TYPE || code == ENUMERAL_TYPE || code == BOOLEAN_TYPE
|| code == COMPLEX_TYPE || code == VECTOR_TYPE)
{
if (TREE_CODE (TREE_TYPE (init)) == ARRAY_TYPE
&& (TREE_CODE (init) == STRING_CST
|| TREE_CODE (init) == COMPOUND_LITERAL_EXPR))
inside_init = init = array_to_pointer_conversion (init_loc, init);
if (semantic_type)
inside_init = build1 (EXCESS_PRECISION_EXPR, semantic_type,
inside_init);
inside_init
= convert_for_assignment (init_loc, UNKNOWN_LOCATION, type,
inside_init, origtype, ic_init,
null_pointer_constant, NULL_TREE, NULL_TREE,
0);
if (inside_init == error_mark_node)
;
else if (require_constant && !TREE_CONSTANT (inside_init))
{
error_init (init_loc, "initializer element is not constant");
inside_init = error_mark_node;
}
else if (require_constant
&& !initializer_constant_valid_p (inside_init,
TREE_TYPE (inside_init)))
{
error_init (init_loc, "initializer element is not computable at "
"load time");
inside_init = error_mark_node;
}
else if (require_constant && !maybe_const)
pedwarn_init (init_loc, OPT_Wpedantic,
"initializer element is not a constant expression");
return inside_init;
}
if (COMPLETE_TYPE_P (type) && TREE_CODE (TYPE_SIZE (type)) != INTEGER_CST)
{
error_init (init_loc, "variable-sized object may not be initialized");
return error_mark_node;
}
error_init (init_loc, "invalid initializer");
return error_mark_node;
}

static tree constructor_type;
static tree constructor_fields;
static tree constructor_index;
static tree constructor_max_index;
static tree constructor_unfilled_fields;
static tree constructor_unfilled_index;
static tree constructor_bit_index;
static vec<constructor_elt, va_gc> *constructor_elements;
static int constructor_incremental;
static int constructor_constant;
static int constructor_simple;
static int constructor_nonconst;
static int constructor_erroneous;
static int constructor_zeroinit;
struct init_node
{
struct init_node *left, *right;
struct init_node *parent;
int balance;
tree purpose;
tree value;
tree origtype;
};
static struct init_node *constructor_pending_elts;
static int constructor_depth;
static tree constructor_decl;
static int constructor_top_level;
static int constructor_designated;
static int designator_depth;
static int designator_erroneous;

struct constructor_range_stack;
struct constructor_stack
{
struct constructor_stack *next;
tree type;
tree fields;
tree index;
tree max_index;
tree unfilled_index;
tree unfilled_fields;
tree bit_index;
vec<constructor_elt, va_gc> *elements;
struct init_node *pending_elts;
int offset;
int depth;
struct c_expr replacement_value;
struct constructor_range_stack *range_stack;
char constant;
char simple;
char nonconst;
char implicit;
char erroneous;
char outer;
char incremental;
char designated;
int designator_depth;
};
static struct constructor_stack *constructor_stack;
struct constructor_range_stack
{
struct constructor_range_stack *next, *prev;
struct constructor_stack *stack;
tree range_start;
tree index;
tree range_end;
tree fields;
};
static struct constructor_range_stack *constructor_range_stack;
struct initializer_stack
{
struct initializer_stack *next;
tree decl;
struct constructor_stack *constructor_stack;
struct constructor_range_stack *constructor_range_stack;
vec<constructor_elt, va_gc> *elements;
struct spelling *spelling;
struct spelling *spelling_base;
int spelling_size;
char top_level;
char require_constant_value;
char require_constant_elements;
rich_location *missing_brace_richloc;
};
static struct initializer_stack *initializer_stack;

void
start_init (tree decl, tree asmspec_tree ATTRIBUTE_UNUSED, int top_level,
rich_location *richloc)
{
const char *locus;
struct initializer_stack *p = XNEW (struct initializer_stack);
p->decl = constructor_decl;
p->require_constant_value = require_constant_value;
p->require_constant_elements = require_constant_elements;
p->constructor_stack = constructor_stack;
p->constructor_range_stack = constructor_range_stack;
p->elements = constructor_elements;
p->spelling = spelling;
p->spelling_base = spelling_base;
p->spelling_size = spelling_size;
p->top_level = constructor_top_level;
p->next = initializer_stack;
p->missing_brace_richloc = richloc;
initializer_stack = p;
constructor_decl = decl;
constructor_designated = 0;
constructor_top_level = top_level;
if (decl != NULL_TREE && decl != error_mark_node)
{
require_constant_value = TREE_STATIC (decl);
require_constant_elements
= ((TREE_STATIC (decl) || (pedantic && !flag_isoc99))
&& AGGREGATE_TYPE_P (TREE_TYPE (decl)));
locus = identifier_to_locale (IDENTIFIER_POINTER (DECL_NAME (decl)));
}
else
{
require_constant_value = 0;
require_constant_elements = 0;
locus = _("(anonymous)");
}
constructor_stack = 0;
constructor_range_stack = 0;
found_missing_braces = 0;
spelling_base = 0;
spelling_size = 0;
RESTORE_SPELLING_DEPTH (0);
if (locus)
push_string (locus);
}
void
finish_init (void)
{
struct initializer_stack *p = initializer_stack;
while (constructor_stack)
{
struct constructor_stack *q = constructor_stack;
constructor_stack = q->next;
free (q);
}
gcc_assert (!constructor_range_stack);
free (spelling_base);
constructor_decl = p->decl;
require_constant_value = p->require_constant_value;
require_constant_elements = p->require_constant_elements;
constructor_stack = p->constructor_stack;
constructor_range_stack = p->constructor_range_stack;
constructor_elements = p->elements;
spelling = p->spelling;
spelling_base = p->spelling_base;
spelling_size = p->spelling_size;
constructor_top_level = p->top_level;
initializer_stack = p->next;
free (p);
}

void
really_start_incremental_init (tree type)
{
struct constructor_stack *p = XNEW (struct constructor_stack);
if (type == NULL_TREE)
type = TREE_TYPE (constructor_decl);
if (VECTOR_TYPE_P (type)
&& TYPE_VECTOR_OPAQUE (type))
error ("opaque vector types cannot be initialized");
p->type = constructor_type;
p->fields = constructor_fields;
p->index = constructor_index;
p->max_index = constructor_max_index;
p->unfilled_index = constructor_unfilled_index;
p->unfilled_fields = constructor_unfilled_fields;
p->bit_index = constructor_bit_index;
p->elements = constructor_elements;
p->constant = constructor_constant;
p->simple = constructor_simple;
p->nonconst = constructor_nonconst;
p->erroneous = constructor_erroneous;
p->pending_elts = constructor_pending_elts;
p->depth = constructor_depth;
p->replacement_value.value = 0;
p->replacement_value.original_code = ERROR_MARK;
p->replacement_value.original_type = NULL;
p->implicit = 0;
p->range_stack = 0;
p->outer = 0;
p->incremental = constructor_incremental;
p->designated = constructor_designated;
p->designator_depth = designator_depth;
p->next = 0;
constructor_stack = p;
constructor_constant = 1;
constructor_simple = 1;
constructor_nonconst = 0;
constructor_depth = SPELLING_DEPTH ();
constructor_elements = NULL;
constructor_pending_elts = 0;
constructor_type = type;
constructor_incremental = 1;
constructor_designated = 0;
constructor_zeroinit = 1;
designator_depth = 0;
designator_erroneous = 0;
if (RECORD_OR_UNION_TYPE_P (constructor_type))
{
constructor_fields = TYPE_FIELDS (constructor_type);
while (constructor_fields != NULL_TREE
&& DECL_UNNAMED_BIT_FIELD (constructor_fields))
constructor_fields = DECL_CHAIN (constructor_fields);
constructor_unfilled_fields = constructor_fields;
constructor_bit_index = bitsize_zero_node;
}
else if (TREE_CODE (constructor_type) == ARRAY_TYPE)
{
if (TYPE_DOMAIN (constructor_type))
{
constructor_max_index
= TYPE_MAX_VALUE (TYPE_DOMAIN (constructor_type));
if (constructor_max_index == NULL_TREE
&& TYPE_SIZE (constructor_type))
constructor_max_index = integer_minus_one_node;
if (constructor_max_index
&& TREE_CODE (constructor_max_index) != INTEGER_CST)
constructor_max_index = integer_minus_one_node;
constructor_index
= convert (bitsizetype,
TYPE_MIN_VALUE (TYPE_DOMAIN (constructor_type)));
}
else
{
constructor_index = bitsize_zero_node;
constructor_max_index = NULL_TREE;
}
constructor_unfilled_index = constructor_index;
}
else if (VECTOR_TYPE_P (constructor_type))
{
constructor_max_index =
bitsize_int (TYPE_VECTOR_SUBPARTS (constructor_type) - 1);
constructor_index = bitsize_zero_node;
constructor_unfilled_index = constructor_index;
}
else
{
constructor_fields = constructor_type;
constructor_unfilled_fields = constructor_type;
}
}

extern location_t last_init_list_comma;
void
finish_implicit_inits (location_t loc, struct obstack *braced_init_obstack)
{
while (constructor_stack->implicit)
{
if (RECORD_OR_UNION_TYPE_P (constructor_type)
&& constructor_fields == NULL_TREE)
process_init_element (input_location,
pop_init_level (loc, 1, braced_init_obstack,
last_init_list_comma),
true, braced_init_obstack);
else if (TREE_CODE (constructor_type) == ARRAY_TYPE
&& constructor_max_index
&& tree_int_cst_lt (constructor_max_index,
constructor_index))
process_init_element (input_location,
pop_init_level (loc, 1, braced_init_obstack,
last_init_list_comma),
true, braced_init_obstack);
else
break;
}
}
void
push_init_level (location_t loc, int implicit,
struct obstack *braced_init_obstack)
{
struct constructor_stack *p;
tree value = NULL_TREE;
if (implicit)
{
if (RECORD_OR_UNION_TYPE_P (constructor_type) && constructor_fields)
value = find_init_member (constructor_fields, braced_init_obstack);
else if (TREE_CODE (constructor_type) == ARRAY_TYPE)
value = find_init_member (constructor_index, braced_init_obstack);
}
p = XNEW (struct constructor_stack);
p->type = constructor_type;
p->fields = constructor_fields;
p->index = constructor_index;
p->max_index = constructor_max_index;
p->unfilled_index = constructor_unfilled_index;
p->unfilled_fields = constructor_unfilled_fields;
p->bit_index = constructor_bit_index;
p->elements = constructor_elements;
p->constant = constructor_constant;
p->simple = constructor_simple;
p->nonconst = constructor_nonconst;
p->erroneous = constructor_erroneous;
p->pending_elts = constructor_pending_elts;
p->depth = constructor_depth;
p->replacement_value.value = NULL_TREE;
p->replacement_value.original_code = ERROR_MARK;
p->replacement_value.original_type = NULL;
p->implicit = implicit;
p->outer = 0;
p->incremental = constructor_incremental;
p->designated = constructor_designated;
p->designator_depth = designator_depth;
p->next = constructor_stack;
p->range_stack = 0;
constructor_stack = p;
constructor_constant = 1;
constructor_simple = 1;
constructor_nonconst = 0;
constructor_depth = SPELLING_DEPTH ();
constructor_elements = NULL;
constructor_incremental = 1;
constructor_designated = 0;
constructor_pending_elts = 0;
if (!implicit)
{
p->range_stack = constructor_range_stack;
constructor_range_stack = 0;
designator_depth = 0;
designator_erroneous = 0;
}
if (constructor_type == NULL_TREE)
;
else if (RECORD_OR_UNION_TYPE_P (constructor_type))
{
if (constructor_fields == NULL_TREE)
constructor_type = NULL_TREE;
else
{
constructor_type = TREE_TYPE (constructor_fields);
push_member_name (constructor_fields);
constructor_depth++;
}
constructor_designated = p->designated;
}
else if (TREE_CODE (constructor_type) == ARRAY_TYPE)
{
constructor_type = TREE_TYPE (constructor_type);
push_array_bounds (tree_to_uhwi (constructor_index));
constructor_depth++;
}
if (constructor_type == NULL_TREE)
{
error_init (loc, "extra brace group at end of initializer");
constructor_fields = NULL_TREE;
constructor_unfilled_fields = NULL_TREE;
return;
}
if (value && TREE_CODE (value) == CONSTRUCTOR)
{
constructor_constant = TREE_CONSTANT (value);
constructor_simple = TREE_STATIC (value);
constructor_nonconst = CONSTRUCTOR_NON_CONST (value);
constructor_elements = CONSTRUCTOR_ELTS (value);
if (!vec_safe_is_empty (constructor_elements)
&& (TREE_CODE (constructor_type) == RECORD_TYPE
|| TREE_CODE (constructor_type) == ARRAY_TYPE))
set_nonincremental_init (braced_init_obstack);
}
if (implicit == 1)
{
found_missing_braces = 1;
if (initializer_stack->missing_brace_richloc)
initializer_stack->missing_brace_richloc->add_fixit_insert_before
(loc, "{");
}
if (RECORD_OR_UNION_TYPE_P (constructor_type))
{
constructor_fields = TYPE_FIELDS (constructor_type);
while (constructor_fields != NULL_TREE
&& DECL_UNNAMED_BIT_FIELD (constructor_fields))
constructor_fields = DECL_CHAIN (constructor_fields);
constructor_unfilled_fields = constructor_fields;
constructor_bit_index = bitsize_zero_node;
}
else if (VECTOR_TYPE_P (constructor_type))
{
constructor_max_index =
bitsize_int (TYPE_VECTOR_SUBPARTS (constructor_type) - 1);
constructor_index = bitsize_int (0);
constructor_unfilled_index = constructor_index;
}
else if (TREE_CODE (constructor_type) == ARRAY_TYPE)
{
if (TYPE_DOMAIN (constructor_type))
{
constructor_max_index
= TYPE_MAX_VALUE (TYPE_DOMAIN (constructor_type));
if (constructor_max_index == NULL_TREE
&& TYPE_SIZE (constructor_type))
constructor_max_index = integer_minus_one_node;
if (constructor_max_index
&& TREE_CODE (constructor_max_index) != INTEGER_CST)
constructor_max_index = integer_minus_one_node;
constructor_index
= convert (bitsizetype,
TYPE_MIN_VALUE (TYPE_DOMAIN (constructor_type)));
}
else
constructor_index = bitsize_zero_node;
constructor_unfilled_index = constructor_index;
if (value && TREE_CODE (value) == STRING_CST)
{
set_nonincremental_init_from_string (value, braced_init_obstack);
}
}
else
{
if (constructor_type != error_mark_node)
warning_init (input_location, 0, "braces around scalar initializer");
constructor_fields = constructor_type;
constructor_unfilled_fields = constructor_type;
}
}
struct c_expr
pop_init_level (location_t loc, int implicit,
struct obstack *braced_init_obstack,
location_t insert_before)
{
struct constructor_stack *p;
struct c_expr ret;
ret.value = NULL_TREE;
ret.original_code = ERROR_MARK;
ret.original_type = NULL;
if (implicit == 0)
{
while (constructor_stack->implicit)
process_init_element (input_location,
pop_init_level (loc, 1, braced_init_obstack,
insert_before),
true, braced_init_obstack);
gcc_assert (!constructor_range_stack);
}
else
if (initializer_stack->missing_brace_richloc)
initializer_stack->missing_brace_richloc->add_fixit_insert_before
(insert_before, "}");
constructor_incremental = 1;
output_pending_init_elements (1, braced_init_obstack);
p = constructor_stack;
if (constructor_type && constructor_fields
&& TREE_CODE (constructor_type) == ARRAY_TYPE
&& TYPE_DOMAIN (constructor_type)
&& !TYPE_MAX_VALUE (TYPE_DOMAIN (constructor_type)))
{
if (integer_zerop (constructor_unfilled_index))
constructor_type = NULL_TREE;
else
{
gcc_assert (!TYPE_SIZE (constructor_type));
if (constructor_depth > 2)
error_init (loc, "initialization of flexible array member in a nested context");
else
pedwarn_init (loc, OPT_Wpedantic,
"initialization of a flexible array member");
if (DECL_CHAIN (constructor_fields) != NULL_TREE)
constructor_type = NULL_TREE;
}
}
switch (vec_safe_length (constructor_elements))
{
case 0:
constructor_zeroinit = 1;
break;
case 1:
if (integer_zerop ((*constructor_elements)[0].value))
constructor_zeroinit = 1;
break;
default:
constructor_zeroinit = 0;
break;
}
if (!implicit && found_missing_braces && warn_missing_braces
&& !constructor_zeroinit)
{
gcc_assert (initializer_stack->missing_brace_richloc);
warning_at (initializer_stack->missing_brace_richloc,
OPT_Wmissing_braces,
"missing braces around initializer");
}
if (warn_missing_field_initializers
&& constructor_type
&& TREE_CODE (constructor_type) == RECORD_TYPE
&& constructor_unfilled_fields)
{
while (constructor_unfilled_fields
&& (!DECL_SIZE (constructor_unfilled_fields)
|| integer_zerop (DECL_SIZE (constructor_unfilled_fields))))
constructor_unfilled_fields = DECL_CHAIN (constructor_unfilled_fields);
if (constructor_unfilled_fields
&& !constructor_designated
&& !constructor_zeroinit)
{
if (warning_at (input_location, OPT_Wmissing_field_initializers,
"missing initializer for field %qD of %qT",
constructor_unfilled_fields,
constructor_type))
inform (DECL_SOURCE_LOCATION (constructor_unfilled_fields),
"%qD declared here", constructor_unfilled_fields);
}
}
if (p->replacement_value.value)
ret = p->replacement_value;
else if (constructor_type == NULL_TREE)
;
else if (!RECORD_OR_UNION_TYPE_P (constructor_type)
&& TREE_CODE (constructor_type) != ARRAY_TYPE
&& !VECTOR_TYPE_P (constructor_type))
{
if (vec_safe_is_empty (constructor_elements))
{
if (!constructor_erroneous)
error_init (loc, "empty scalar initializer");
ret.value = error_mark_node;
}
else if (vec_safe_length (constructor_elements) != 1)
{
error_init (loc, "extra elements in scalar initializer");
ret.value = (*constructor_elements)[0].value;
}
else
ret.value = (*constructor_elements)[0].value;
}
else
{
if (constructor_erroneous)
ret.value = error_mark_node;
else
{
ret.value = build_constructor (constructor_type,
constructor_elements);
if (constructor_constant)
TREE_CONSTANT (ret.value) = 1;
if (constructor_constant && constructor_simple)
TREE_STATIC (ret.value) = 1;
if (constructor_nonconst)
CONSTRUCTOR_NON_CONST (ret.value) = 1;
}
}
if (ret.value && TREE_CODE (ret.value) != CONSTRUCTOR)
{
if (constructor_nonconst)
ret.original_code = C_MAYBE_CONST_EXPR;
else if (ret.original_code == C_MAYBE_CONST_EXPR)
ret.original_code = ERROR_MARK;
}
constructor_type = p->type;
constructor_fields = p->fields;
constructor_index = p->index;
constructor_max_index = p->max_index;
constructor_unfilled_index = p->unfilled_index;
constructor_unfilled_fields = p->unfilled_fields;
constructor_bit_index = p->bit_index;
constructor_elements = p->elements;
constructor_constant = p->constant;
constructor_simple = p->simple;
constructor_nonconst = p->nonconst;
constructor_erroneous = p->erroneous;
constructor_incremental = p->incremental;
constructor_designated = p->designated;
designator_depth = p->designator_depth;
constructor_pending_elts = p->pending_elts;
constructor_depth = p->depth;
if (!p->implicit)
constructor_range_stack = p->range_stack;
RESTORE_SPELLING_DEPTH (constructor_depth);
constructor_stack = p->next;
free (p);
if (ret.value == NULL_TREE && constructor_stack == 0)
ret.value = error_mark_node;
return ret;
}
static bool
set_designator (location_t loc, bool array,
struct obstack *braced_init_obstack)
{
tree subtype;
enum tree_code subcode;
if (constructor_type == NULL_TREE)
return true;
if (designator_erroneous)
return true;
if (!designator_depth)
{
gcc_assert (!constructor_range_stack);
while (constructor_stack->implicit)
process_init_element (input_location,
pop_init_level (loc, 1, braced_init_obstack,
last_init_list_comma),
true, braced_init_obstack);
constructor_designated = 1;
return false;
}
switch (TREE_CODE (constructor_type))
{
case  RECORD_TYPE:
case  UNION_TYPE:
subtype = TREE_TYPE (constructor_fields);
if (subtype != error_mark_node)
subtype = TYPE_MAIN_VARIANT (subtype);
break;
case ARRAY_TYPE:
subtype = TYPE_MAIN_VARIANT (TREE_TYPE (constructor_type));
break;
default:
gcc_unreachable ();
}
subcode = TREE_CODE (subtype);
if (array && subcode != ARRAY_TYPE)
{
error_init (loc, "array index in non-array initializer");
return true;
}
else if (!array && subcode != RECORD_TYPE && subcode != UNION_TYPE)
{
error_init (loc, "field name not in record or union initializer");
return true;
}
constructor_designated = 1;
finish_implicit_inits (loc, braced_init_obstack);
push_init_level (loc, 2, braced_init_obstack);
return false;
}
static void
push_range_stack (tree range_end, struct obstack * braced_init_obstack)
{
struct constructor_range_stack *p;
p = (struct constructor_range_stack *)
obstack_alloc (braced_init_obstack,
sizeof (struct constructor_range_stack));
p->prev = constructor_range_stack;
p->next = 0;
p->fields = constructor_fields;
p->range_start = constructor_index;
p->index = constructor_index;
p->stack = constructor_stack;
p->range_end = range_end;
if (constructor_range_stack)
constructor_range_stack->next = p;
constructor_range_stack = p;
}
void
set_init_index (location_t loc, tree first, tree last,
struct obstack *braced_init_obstack)
{
if (set_designator (loc, true, braced_init_obstack))
return;
designator_erroneous = 1;
if (!INTEGRAL_TYPE_P (TREE_TYPE (first))
|| (last && !INTEGRAL_TYPE_P (TREE_TYPE (last))))
{
error_init (loc, "array index in initializer not of integer type");
return;
}
if (TREE_CODE (first) != INTEGER_CST)
{
first = c_fully_fold (first, false, NULL);
if (TREE_CODE (first) == INTEGER_CST)
pedwarn_init (loc, OPT_Wpedantic,
"array index in initializer is not "
"an integer constant expression");
}
if (last && TREE_CODE (last) != INTEGER_CST)
{
last = c_fully_fold (last, false, NULL);
if (TREE_CODE (last) == INTEGER_CST)
pedwarn_init (loc, OPT_Wpedantic,
"array index in initializer is not "
"an integer constant expression");
}
if (TREE_CODE (first) != INTEGER_CST)
error_init (loc, "nonconstant array index in initializer");
else if (last != NULL_TREE && TREE_CODE (last) != INTEGER_CST)
error_init (loc, "nonconstant array index in initializer");
else if (TREE_CODE (constructor_type) != ARRAY_TYPE)
error_init (loc, "array index in non-array initializer");
else if (tree_int_cst_sgn (first) == -1)
error_init (loc, "array index in initializer exceeds array bounds");
else if (constructor_max_index
&& tree_int_cst_lt (constructor_max_index, first))
error_init (loc, "array index in initializer exceeds array bounds");
else
{
constant_expression_warning (first);
if (last)
constant_expression_warning (last);
constructor_index = convert (bitsizetype, first);
if (tree_int_cst_lt (constructor_index, first))
{
constructor_index = copy_node (constructor_index);
TREE_OVERFLOW (constructor_index) = 1;
}
if (last)
{
if (tree_int_cst_equal (first, last))
last = NULL_TREE;
else if (tree_int_cst_lt (last, first))
{
error_init (loc, "empty index range in initializer");
last = NULL_TREE;
}
else
{
last = convert (bitsizetype, last);
if (constructor_max_index != NULL_TREE
&& tree_int_cst_lt (constructor_max_index, last))
{
error_init (loc, "array index range in initializer exceeds "
"array bounds");
last = NULL_TREE;
}
}
}
designator_depth++;
designator_erroneous = 0;
if (constructor_range_stack || last)
push_range_stack (last, braced_init_obstack);
}
}
void
set_init_label (location_t loc, tree fieldname, location_t fieldname_loc,
struct obstack *braced_init_obstack)
{
tree field;
if (set_designator (loc, false, braced_init_obstack))
return;
designator_erroneous = 1;
if (!RECORD_OR_UNION_TYPE_P (constructor_type))
{
error_init (loc, "field name not in record or union initializer");
return;
}
field = lookup_field (constructor_type, fieldname);
if (field == NULL_TREE)
{
tree guessed_id = lookup_field_fuzzy (constructor_type, fieldname);
if (guessed_id)
{
gcc_rich_location rich_loc (fieldname_loc);
rich_loc.add_fixit_misspelled_id (fieldname_loc, guessed_id);
error_at (&rich_loc,
"%qT has no member named %qE; did you mean %qE?",
constructor_type, fieldname, guessed_id);
}
else
error_at (fieldname_loc, "%qT has no member named %qE",
constructor_type, fieldname);
}
else
do
{
constructor_fields = TREE_VALUE (field);
designator_depth++;
designator_erroneous = 0;
if (constructor_range_stack)
push_range_stack (NULL_TREE, braced_init_obstack);
field = TREE_CHAIN (field);
if (field)
{
if (set_designator (loc, false, braced_init_obstack))
return;
}
}
while (field != NULL_TREE);
}

static void
add_pending_init (location_t loc, tree purpose, tree value, tree origtype,
bool implicit, struct obstack *braced_init_obstack)
{
struct init_node *p, **q, *r;
q = &constructor_pending_elts;
p = 0;
if (TREE_CODE (constructor_type) == ARRAY_TYPE)
{
while (*q != 0)
{
p = *q;
if (tree_int_cst_lt (purpose, p->purpose))
q = &p->left;
else if (tree_int_cst_lt (p->purpose, purpose))
q = &p->right;
else
{
if (!implicit)
{
if (TREE_SIDE_EFFECTS (p->value))
warning_init (loc, OPT_Woverride_init_side_effects,
"initialized field with side-effects "
"overwritten");
else if (warn_override_init)
warning_init (loc, OPT_Woverride_init,
"initialized field overwritten");
}
p->value = value;
p->origtype = origtype;
return;
}
}
}
else
{
tree bitpos;
bitpos = bit_position (purpose);
while (*q != NULL)
{
p = *q;
if (tree_int_cst_lt (bitpos, bit_position (p->purpose)))
q = &p->left;
else if (p->purpose != purpose)
q = &p->right;
else
{
if (!implicit)
{
if (TREE_SIDE_EFFECTS (p->value))
warning_init (loc, OPT_Woverride_init_side_effects,
"initialized field with side-effects "
"overwritten");
else if (warn_override_init)
warning_init (loc, OPT_Woverride_init,
"initialized field overwritten");
}
p->value = value;
p->origtype = origtype;
return;
}
}
}
r = (struct init_node *) obstack_alloc (braced_init_obstack,
sizeof (struct init_node));
r->purpose = purpose;
r->value = value;
r->origtype = origtype;
*q = r;
r->parent = p;
r->left = 0;
r->right = 0;
r->balance = 0;
while (p)
{
struct init_node *s;
if (r == p->left)
{
if (p->balance == 0)
p->balance = -1;
else if (p->balance < 0)
{
if (r->balance < 0)
{
p->left = r->right;
if (p->left)
p->left->parent = p;
r->right = p;
p->balance = 0;
r->balance = 0;
s = p->parent;
p->parent = r;
r->parent = s;
if (s)
{
if (s->left == p)
s->left = r;
else
s->right = r;
}
else
constructor_pending_elts = r;
}
else
{
struct init_node *t = r->right;
r->right = t->left;
if (r->right)
r->right->parent = r;
t->left = r;
p->left = t->right;
if (p->left)
p->left->parent = p;
t->right = p;
p->balance = t->balance < 0;
r->balance = -(t->balance > 0);
t->balance = 0;
s = p->parent;
p->parent = t;
r->parent = t;
t->parent = s;
if (s)
{
if (s->left == p)
s->left = t;
else
s->right = t;
}
else
constructor_pending_elts = t;
}
break;
}
else
{
p->balance = 0;
break;
}
}
else 
{
if (p->balance == 0)
p->balance++;
else if (p->balance > 0)
{
if (r->balance > 0)
{
p->right = r->left;
if (p->right)
p->right->parent = p;
r->left = p;
p->balance = 0;
r->balance = 0;
s = p->parent;
p->parent = r;
r->parent = s;
if (s)
{
if (s->left == p)
s->left = r;
else
s->right = r;
}
else
constructor_pending_elts = r;
}
else 
{
struct init_node *t = r->left;
r->left = t->right;
if (r->left)
r->left->parent = r;
t->right = r;
p->right = t->left;
if (p->right)
p->right->parent = p;
t->left = p;
r->balance = (t->balance < 0);
p->balance = -(t->balance > 0);
t->balance = 0;
s = p->parent;
p->parent = t;
r->parent = t;
t->parent = s;
if (s)
{
if (s->left == p)
s->left = t;
else
s->right = t;
}
else
constructor_pending_elts = t;
}
break;
}
else
{
p->balance = 0;
break;
}
}
r = p;
p = p->parent;
}
}
static void
set_nonincremental_init (struct obstack * braced_init_obstack)
{
unsigned HOST_WIDE_INT ix;
tree index, value;
if (TREE_CODE (constructor_type) != RECORD_TYPE
&& TREE_CODE (constructor_type) != ARRAY_TYPE)
return;
FOR_EACH_CONSTRUCTOR_ELT (constructor_elements, ix, index, value)
add_pending_init (input_location, index, value, NULL_TREE, true,
braced_init_obstack);
constructor_elements = NULL;
if (TREE_CODE (constructor_type) == RECORD_TYPE)
{
constructor_unfilled_fields = TYPE_FIELDS (constructor_type);
while (constructor_unfilled_fields != NULL_TREE
&& DECL_UNNAMED_BIT_FIELD (constructor_unfilled_fields))
constructor_unfilled_fields = TREE_CHAIN (constructor_unfilled_fields);
}
else if (TREE_CODE (constructor_type) == ARRAY_TYPE)
{
if (TYPE_DOMAIN (constructor_type))
constructor_unfilled_index
= convert (bitsizetype,
TYPE_MIN_VALUE (TYPE_DOMAIN (constructor_type)));
else
constructor_unfilled_index = bitsize_zero_node;
}
constructor_incremental = 0;
}
static void
set_nonincremental_init_from_string (tree str,
struct obstack * braced_init_obstack)
{
tree value, purpose, type;
HOST_WIDE_INT val[2];
const char *p, *end;
int byte, wchar_bytes, charwidth, bitpos;
gcc_assert (TREE_CODE (constructor_type) == ARRAY_TYPE);
wchar_bytes = TYPE_PRECISION (TREE_TYPE (TREE_TYPE (str))) / BITS_PER_UNIT;
charwidth = TYPE_PRECISION (char_type_node);
gcc_assert ((size_t) wchar_bytes * charwidth
<= ARRAY_SIZE (val) * HOST_BITS_PER_WIDE_INT);
type = TREE_TYPE (constructor_type);
p = TREE_STRING_POINTER (str);
end = p + TREE_STRING_LENGTH (str);
for (purpose = bitsize_zero_node;
p < end
&& !(constructor_max_index
&& tree_int_cst_lt (constructor_max_index, purpose));
purpose = size_binop (PLUS_EXPR, purpose, bitsize_one_node))
{
if (wchar_bytes == 1)
{
val[0] = (unsigned char) *p++;
val[1] = 0;
}
else
{
val[1] = 0;
val[0] = 0;
for (byte = 0; byte < wchar_bytes; byte++)
{
if (BYTES_BIG_ENDIAN)
bitpos = (wchar_bytes - byte - 1) * charwidth;
else
bitpos = byte * charwidth;
val[bitpos / HOST_BITS_PER_WIDE_INT]
|= ((unsigned HOST_WIDE_INT) ((unsigned char) *p++))
<< (bitpos % HOST_BITS_PER_WIDE_INT);
}
}
if (!TYPE_UNSIGNED (type))
{
bitpos = ((wchar_bytes - 1) * charwidth) + HOST_BITS_PER_CHAR;
if (bitpos < HOST_BITS_PER_WIDE_INT)
{
if (val[0] & (HOST_WIDE_INT_1 << (bitpos - 1)))
{
val[0] |= HOST_WIDE_INT_M1U << bitpos;
val[1] = -1;
}
}
else if (bitpos == HOST_BITS_PER_WIDE_INT)
{
if (val[0] < 0)
val[1] = -1;
}
else if (val[1] & (HOST_WIDE_INT_1
<< (bitpos - 1 - HOST_BITS_PER_WIDE_INT)))
val[1] |= HOST_WIDE_INT_M1U << (bitpos - HOST_BITS_PER_WIDE_INT);
}
value = wide_int_to_tree (type,
wide_int::from_array (val, 2,
HOST_BITS_PER_WIDE_INT * 2));
add_pending_init (input_location, purpose, value, NULL_TREE, true,
braced_init_obstack);
}
constructor_incremental = 0;
}
static tree
find_init_member (tree field, struct obstack * braced_init_obstack)
{
struct init_node *p;
if (TREE_CODE (constructor_type) == ARRAY_TYPE)
{
if (constructor_incremental
&& tree_int_cst_lt (field, constructor_unfilled_index))
set_nonincremental_init (braced_init_obstack);
p = constructor_pending_elts;
while (p)
{
if (tree_int_cst_lt (field, p->purpose))
p = p->left;
else if (tree_int_cst_lt (p->purpose, field))
p = p->right;
else
return p->value;
}
}
else if (TREE_CODE (constructor_type) == RECORD_TYPE)
{
tree bitpos = bit_position (field);
if (constructor_incremental
&& (!constructor_unfilled_fields
|| tree_int_cst_lt (bitpos,
bit_position (constructor_unfilled_fields))))
set_nonincremental_init (braced_init_obstack);
p = constructor_pending_elts;
while (p)
{
if (field == p->purpose)
return p->value;
else if (tree_int_cst_lt (bitpos, bit_position (p->purpose)))
p = p->left;
else
p = p->right;
}
}
else if (TREE_CODE (constructor_type) == UNION_TYPE)
{
if (!vec_safe_is_empty (constructor_elements)
&& (constructor_elements->last ().index == field))
return constructor_elements->last ().value;
}
return NULL_TREE;
}
static void
output_init_element (location_t loc, tree value, tree origtype,
bool strict_string, tree type, tree field, bool pending,
bool implicit, struct obstack * braced_init_obstack)
{
tree semantic_type = NULL_TREE;
bool maybe_const = true;
bool npc;
if (type == error_mark_node || value == error_mark_node)
{
constructor_erroneous = 1;
return;
}
if (TREE_CODE (TREE_TYPE (value)) == ARRAY_TYPE
&& (TREE_CODE (value) == STRING_CST
|| TREE_CODE (value) == COMPOUND_LITERAL_EXPR)
&& !(TREE_CODE (value) == STRING_CST
&& TREE_CODE (type) == ARRAY_TYPE
&& INTEGRAL_TYPE_P (TREE_TYPE (type)))
&& !comptypes (TYPE_MAIN_VARIANT (TREE_TYPE (value)),
TYPE_MAIN_VARIANT (type)))
value = array_to_pointer_conversion (input_location, value);
if (TREE_CODE (value) == COMPOUND_LITERAL_EXPR
&& require_constant_value && pending)
{
if (flag_isoc99)
pedwarn_init (loc, OPT_Wpedantic, "initializer element is not "
"constant");
tree decl = COMPOUND_LITERAL_EXPR_DECL (value);
value = DECL_INITIAL (decl);
}
npc = null_pointer_constant_p (value);
if (TREE_CODE (value) == EXCESS_PRECISION_EXPR)
{
semantic_type = TREE_TYPE (value);
value = TREE_OPERAND (value, 0);
}
value = c_fully_fold (value, require_constant_value, &maybe_const);
if (value == error_mark_node)
constructor_erroneous = 1;
else if (!TREE_CONSTANT (value))
constructor_constant = 0;
else if (!initializer_constant_valid_p (value,
TREE_TYPE (value),
AGGREGATE_TYPE_P (constructor_type)
&& TYPE_REVERSE_STORAGE_ORDER
(constructor_type))
|| (RECORD_OR_UNION_TYPE_P (constructor_type)
&& DECL_C_BIT_FIELD (field)
&& TREE_CODE (value) != INTEGER_CST))
constructor_simple = 0;
if (!maybe_const)
constructor_nonconst = 1;
tree new_value = value;
if (semantic_type)
new_value = build1 (EXCESS_PRECISION_EXPR, semantic_type, value);
new_value = digest_init (loc, type, new_value, origtype, npc, strict_string,
require_constant_value);
if (new_value == error_mark_node)
{
constructor_erroneous = 1;
return;
}
if (require_constant_value || require_constant_elements)
constant_expression_warning (new_value);
if (!initializer_constant_valid_p (value, TREE_TYPE (value)))
{
if (require_constant_value)
{
error_init (loc, "initializer element is not constant");
value = error_mark_node;
}
else if (require_constant_elements)
pedwarn (loc, OPT_Wpedantic,
"initializer element is not computable at load time");
}
else if (!maybe_const
&& (require_constant_value || require_constant_elements))
pedwarn_init (loc, OPT_Wpedantic,
"initializer element is not a constant expression");
if (warn_cxx_compat
&& field != NULL_TREE
&& TREE_CODE (field) == FIELD_DECL
&& DECL_BIT_FIELD_TYPE (field) != NULL_TREE
&& (TYPE_MAIN_VARIANT (DECL_BIT_FIELD_TYPE (field))
!= TYPE_MAIN_VARIANT (type))
&& TREE_CODE (DECL_BIT_FIELD_TYPE (field)) == ENUMERAL_TYPE)
{
tree checktype = origtype != NULL_TREE ? origtype : TREE_TYPE (value);
if (checktype != error_mark_node
&& (TYPE_MAIN_VARIANT (checktype)
!= TYPE_MAIN_VARIANT (DECL_BIT_FIELD_TYPE (field))))
warning_init (loc, OPT_Wc___compat,
"enum conversion in initialization is invalid in C++");
}
if (field
&& (TREE_TYPE (field) == error_mark_node
|| (COMPLETE_TYPE_P (TREE_TYPE (field))
&& integer_zerop (TYPE_SIZE (TREE_TYPE (field)))
&& !TREE_SIDE_EFFECTS (new_value)
&& (TREE_CODE (constructor_type) == ARRAY_TYPE
|| DECL_CHAIN (field)))))
return;
value = new_value;
if (TREE_CODE (constructor_type) == ARRAY_TYPE
&& (!constructor_incremental
|| !tree_int_cst_equal (field, constructor_unfilled_index)))
{
if (constructor_incremental
&& tree_int_cst_lt (field, constructor_unfilled_index))
set_nonincremental_init (braced_init_obstack);
add_pending_init (loc, field, value, origtype, implicit,
braced_init_obstack);
return;
}
else if (TREE_CODE (constructor_type) == RECORD_TYPE
&& (!constructor_incremental
|| field != constructor_unfilled_fields))
{
if (constructor_incremental)
{
if (!constructor_unfilled_fields)
set_nonincremental_init (braced_init_obstack);
else
{
tree bitpos, unfillpos;
bitpos = bit_position (field);
unfillpos = bit_position (constructor_unfilled_fields);
if (tree_int_cst_lt (bitpos, unfillpos))
set_nonincremental_init (braced_init_obstack);
}
}
add_pending_init (loc, field, value, origtype, implicit,
braced_init_obstack);
return;
}
else if (TREE_CODE (constructor_type) == UNION_TYPE
&& !vec_safe_is_empty (constructor_elements))
{
if (!implicit)
{
if (TREE_SIDE_EFFECTS (constructor_elements->last ().value))
warning_init (loc, OPT_Woverride_init_side_effects,
"initialized field with side-effects overwritten");
else if (warn_override_init)
warning_init (loc, OPT_Woverride_init,
"initialized field overwritten");
}
constructor_elements = NULL;
}
constructor_elt celt = {field, value};
vec_safe_push (constructor_elements, celt);
if (TREE_CODE (constructor_type) == ARRAY_TYPE)
constructor_unfilled_index
= size_binop_loc (input_location, PLUS_EXPR, constructor_unfilled_index,
bitsize_one_node);
else if (TREE_CODE (constructor_type) == RECORD_TYPE)
{
constructor_unfilled_fields
= DECL_CHAIN (constructor_unfilled_fields);
while (constructor_unfilled_fields != NULL_TREE
&& DECL_UNNAMED_BIT_FIELD (constructor_unfilled_fields))
constructor_unfilled_fields =
DECL_CHAIN (constructor_unfilled_fields);
}
else if (TREE_CODE (constructor_type) == UNION_TYPE)
constructor_unfilled_fields = NULL_TREE;
if (pending)
output_pending_init_elements (0, braced_init_obstack);
}
static int
init_field_decl_cmp (tree field1, tree field2)
{
if (field1 == field2)
return 0;
tree bitpos1 = bit_position (field1);
tree bitpos2 = bit_position (field2);
if (tree_int_cst_equal (bitpos1, bitpos2))
{
if (TREE_TYPE (field1) != error_mark_node
&& COMPLETE_TYPE_P (TREE_TYPE (field1))
&& integer_nonzerop (TREE_TYPE (field1)))
return 1;
if (TREE_TYPE (field2) != error_mark_node
&& COMPLETE_TYPE_P (TREE_TYPE (field2))
&& integer_nonzerop (TREE_TYPE (field2)))
return -1;
tree f1 = field1, f2 = field2;
while (1)
{
f1 = DECL_CHAIN (f1);
f2 = DECL_CHAIN (f2);
if (f1 == NULL_TREE)
{
gcc_assert (f2);
return 1;
}
if (f2 == NULL_TREE)
return -1;
if (f1 == field2)
return -1;
if (f2 == field1)
return 1;
if (!tree_int_cst_equal (bit_position (f1), bitpos1))
return 1;
if (!tree_int_cst_equal (bit_position (f2), bitpos1))
return -1;
}
}
else if (tree_int_cst_lt (bitpos1, bitpos2))
return -1;
else
return 1;
}
static void
output_pending_init_elements (int all, struct obstack * braced_init_obstack)
{
struct init_node *elt = constructor_pending_elts;
tree next;
retry:
next = NULL_TREE;
while (elt)
{
if (TREE_CODE (constructor_type) == ARRAY_TYPE)
{
if (tree_int_cst_equal (elt->purpose,
constructor_unfilled_index))
output_init_element (input_location, elt->value, elt->origtype,
true, TREE_TYPE (constructor_type),
constructor_unfilled_index, false, false,
braced_init_obstack);
else if (tree_int_cst_lt (constructor_unfilled_index,
elt->purpose))
{
if (elt->left)
elt = elt->left;
else
{
next = elt->purpose;
break;
}
}
else
{
if (elt->right)
elt = elt->right;
else
{
while (elt->parent && elt->parent->right == elt)
elt = elt->parent;
elt = elt->parent;
if (elt && tree_int_cst_lt (constructor_unfilled_index,
elt->purpose))
{
next = elt->purpose;
break;
}
}
}
}
else if (RECORD_OR_UNION_TYPE_P (constructor_type))
{
if (constructor_unfilled_fields == NULL_TREE)
break;
int cmp = init_field_decl_cmp (constructor_unfilled_fields,
elt->purpose);
if (cmp == 0)
output_init_element (input_location, elt->value, elt->origtype,
true, TREE_TYPE (elt->purpose),
elt->purpose, false, false,
braced_init_obstack);
else if (cmp < 0)
{
if (elt->left)
elt = elt->left;
else
{
next = elt->purpose;
break;
}
}
else
{
if (elt->right)
elt = elt->right;
else
{
while (elt->parent && elt->parent->right == elt)
elt = elt->parent;
elt = elt->parent;
if (elt
&& init_field_decl_cmp (constructor_unfilled_fields,
elt->purpose) < 0)
{
next = elt->purpose;
break;
}
}
}
}
}
if (!(all && next != NULL_TREE))
return;
if (RECORD_OR_UNION_TYPE_P (constructor_type))
constructor_unfilled_fields = next;
else if (TREE_CODE (constructor_type) == ARRAY_TYPE)
constructor_unfilled_index = next;
goto retry;
}

void
process_init_element (location_t loc, struct c_expr value, bool implicit,
struct obstack * braced_init_obstack)
{
tree orig_value = value.value;
int string_flag
= (orig_value != NULL_TREE && TREE_CODE (orig_value) == STRING_CST);
bool strict_string = value.original_code == STRING_CST;
bool was_designated = designator_depth != 0;
designator_depth = 0;
designator_erroneous = 0;
if (!implicit && value.value && !integer_zerop (value.value))
constructor_zeroinit = 0;
if (string_flag
&& constructor_type
&& !was_designated
&& TREE_CODE (constructor_type) == ARRAY_TYPE
&& INTEGRAL_TYPE_P (TREE_TYPE (constructor_type))
&& integer_zerop (constructor_unfilled_index))
{
if (constructor_stack->replacement_value.value)
error_init (loc, "excess elements in char array initializer");
constructor_stack->replacement_value = value;
return;
}
if (constructor_stack->replacement_value.value != NULL_TREE)
{
error_init (loc, "excess elements in struct initializer");
return;
}
if (constructor_type == NULL_TREE)
return;
if (!implicit && warn_designated_init && !was_designated
&& TREE_CODE (constructor_type) == RECORD_TYPE
&& lookup_attribute ("designated_init",
TYPE_ATTRIBUTES (constructor_type)))
warning_init (loc,
OPT_Wdesignated_init,
"positional initialization of field "
"in %<struct%> declared with %<designated_init%> attribute");
while (constructor_stack->implicit)
{
if (RECORD_OR_UNION_TYPE_P (constructor_type)
&& constructor_fields == NULL_TREE)
process_init_element (loc,
pop_init_level (loc, 1, braced_init_obstack,
last_init_list_comma),
true, braced_init_obstack);
else if ((TREE_CODE (constructor_type) == ARRAY_TYPE
|| VECTOR_TYPE_P (constructor_type))
&& constructor_max_index
&& tree_int_cst_lt (constructor_max_index,
constructor_index))
process_init_element (loc,
pop_init_level (loc, 1, braced_init_obstack,
last_init_list_comma),
true, braced_init_obstack);
else
break;
}
if (constructor_range_stack)
{
if (TREE_CODE (value.value) != COMPOUND_LITERAL_EXPR
|| !require_constant_value)
{
tree semantic_type = NULL_TREE;
if (TREE_CODE (value.value) == EXCESS_PRECISION_EXPR)
{
semantic_type = TREE_TYPE (value.value);
value.value = TREE_OPERAND (value.value, 0);
}
value.value = save_expr (value.value);
if (semantic_type)
value.value = build1 (EXCESS_PRECISION_EXPR, semantic_type,
value.value);
}
}
while (1)
{
if (TREE_CODE (constructor_type) == RECORD_TYPE)
{
tree fieldtype;
enum tree_code fieldcode;
if (constructor_fields == NULL_TREE)
{
pedwarn_init (loc, 0, "excess elements in struct initializer");
break;
}
fieldtype = TREE_TYPE (constructor_fields);
if (fieldtype != error_mark_node)
fieldtype = TYPE_MAIN_VARIANT (fieldtype);
fieldcode = TREE_CODE (fieldtype);
if (fieldcode == ARRAY_TYPE
&& !require_constant_value
&& TYPE_SIZE (fieldtype) == NULL_TREE
&& DECL_CHAIN (constructor_fields) == NULL_TREE)
{
error_init (loc, "non-static initialization of a flexible "
"array member");
break;
}
if (string_flag
&& fieldcode == ARRAY_TYPE
&& constructor_depth > 1
&& TYPE_SIZE (fieldtype) == NULL_TREE
&& DECL_CHAIN (constructor_fields) == NULL_TREE)
{
bool in_array_p = false;
for (struct constructor_stack *p = constructor_stack;
p && p->type; p = p->next)
if (TREE_CODE (p->type) == ARRAY_TYPE)
{
in_array_p = true;
break;
}
if (in_array_p)
{
error_init (loc, "initialization of flexible array "
"member in a nested context");
break;
}
}
if (value.value != NULL_TREE
&& fieldcode == ARRAY_TYPE
&& INTEGRAL_TYPE_P (TREE_TYPE (fieldtype))
&& string_flag)
value.value = orig_value;
else if (value.value != NULL_TREE
&& value.value != error_mark_node
&& TYPE_MAIN_VARIANT (TREE_TYPE (value.value)) != fieldtype
&& (fieldcode == RECORD_TYPE || fieldcode == ARRAY_TYPE
|| fieldcode == UNION_TYPE || fieldcode == VECTOR_TYPE))
{
push_init_level (loc, 1, braced_init_obstack);
continue;
}
if (value.value)
{
push_member_name (constructor_fields);
output_init_element (loc, value.value, value.original_type,
strict_string, fieldtype,
constructor_fields, true, implicit,
braced_init_obstack);
RESTORE_SPELLING_DEPTH (constructor_depth);
}
else
{
if (DECL_SIZE (constructor_fields))
constructor_bit_index
= size_binop_loc (input_location, PLUS_EXPR,
bit_position (constructor_fields),
DECL_SIZE (constructor_fields));
if (constructor_unfilled_fields == constructor_fields)
{
constructor_unfilled_fields = DECL_CHAIN (constructor_fields);
while (constructor_unfilled_fields != 0
&& (DECL_UNNAMED_BIT_FIELD
(constructor_unfilled_fields)))
constructor_unfilled_fields =
DECL_CHAIN (constructor_unfilled_fields);
}
}
constructor_fields = DECL_CHAIN (constructor_fields);
while (constructor_fields != NULL_TREE
&& DECL_UNNAMED_BIT_FIELD (constructor_fields))
constructor_fields = DECL_CHAIN (constructor_fields);
}
else if (TREE_CODE (constructor_type) == UNION_TYPE)
{
tree fieldtype;
enum tree_code fieldcode;
if (constructor_fields == NULL_TREE)
{
pedwarn_init (loc, 0,
"excess elements in union initializer");
break;
}
fieldtype = TREE_TYPE (constructor_fields);
if (fieldtype != error_mark_node)
fieldtype = TYPE_MAIN_VARIANT (fieldtype);
fieldcode = TREE_CODE (fieldtype);
if (!in_system_header_at (input_location) && !constructor_designated
&& !(value.value && (integer_zerop (value.value)
|| real_zerop (value.value))))
warning (OPT_Wtraditional, "traditional C rejects initialization "
"of unions");
if (value.value != NULL_TREE
&& fieldcode == ARRAY_TYPE
&& INTEGRAL_TYPE_P (TREE_TYPE (fieldtype))
&& string_flag)
value.value = orig_value;
else if (value.value != NULL_TREE
&& value.value != error_mark_node
&& TYPE_MAIN_VARIANT (TREE_TYPE (value.value)) != fieldtype
&& (fieldcode == RECORD_TYPE || fieldcode == ARRAY_TYPE
|| fieldcode == UNION_TYPE || fieldcode == VECTOR_TYPE))
{
push_init_level (loc, 1, braced_init_obstack);
continue;
}
if (value.value)
{
push_member_name (constructor_fields);
output_init_element (loc, value.value, value.original_type,
strict_string, fieldtype,
constructor_fields, true, implicit,
braced_init_obstack);
RESTORE_SPELLING_DEPTH (constructor_depth);
}
else
{
constructor_bit_index = DECL_SIZE (constructor_fields);
constructor_unfilled_fields = DECL_CHAIN (constructor_fields);
}
constructor_fields = NULL_TREE;
}
else if (TREE_CODE (constructor_type) == ARRAY_TYPE)
{
tree elttype = TYPE_MAIN_VARIANT (TREE_TYPE (constructor_type));
enum tree_code eltcode = TREE_CODE (elttype);
if (value.value != NULL_TREE
&& eltcode == ARRAY_TYPE
&& INTEGRAL_TYPE_P (TREE_TYPE (elttype))
&& string_flag)
value.value = orig_value;
else if (value.value != NULL_TREE
&& value.value != error_mark_node
&& TYPE_MAIN_VARIANT (TREE_TYPE (value.value)) != elttype
&& (eltcode == RECORD_TYPE || eltcode == ARRAY_TYPE
|| eltcode == UNION_TYPE || eltcode == VECTOR_TYPE))
{
push_init_level (loc, 1, braced_init_obstack);
continue;
}
if (constructor_max_index != NULL_TREE
&& (tree_int_cst_lt (constructor_max_index, constructor_index)
|| integer_all_onesp (constructor_max_index)))
{
pedwarn_init (loc, 0,
"excess elements in array initializer");
break;
}
if (value.value)
{
push_array_bounds (tree_to_uhwi (constructor_index));
output_init_element (loc, value.value, value.original_type,
strict_string, elttype,
constructor_index, true, implicit,
braced_init_obstack);
RESTORE_SPELLING_DEPTH (constructor_depth);
}
constructor_index
= size_binop_loc (input_location, PLUS_EXPR,
constructor_index, bitsize_one_node);
if (!value.value)
constructor_unfilled_index = constructor_index;
}
else if (VECTOR_TYPE_P (constructor_type))
{
tree elttype = TYPE_MAIN_VARIANT (TREE_TYPE (constructor_type));
if (tree_int_cst_lt (constructor_max_index, constructor_index))
{
pedwarn_init (loc, 0,
"excess elements in vector initializer");
break;
}
if (value.value)
{
if (TREE_CODE (value.value) == VECTOR_CST)
elttype = TYPE_MAIN_VARIANT (constructor_type);
output_init_element (loc, value.value, value.original_type,
strict_string, elttype,
constructor_index, true, implicit,
braced_init_obstack);
}
constructor_index
= size_binop_loc (input_location,
PLUS_EXPR, constructor_index, bitsize_one_node);
if (!value.value)
constructor_unfilled_index = constructor_index;
}
else if (constructor_type != error_mark_node
&& constructor_fields == NULL_TREE)
{
pedwarn_init (loc, 0,
"excess elements in scalar initializer");
break;
}
else
{
if (value.value)
output_init_element (loc, value.value, value.original_type,
strict_string, constructor_type,
NULL_TREE, true, implicit,
braced_init_obstack);
constructor_fields = NULL_TREE;
}
if (constructor_range_stack)
{
struct constructor_range_stack *p, *range_stack;
int finish = 0;
range_stack = constructor_range_stack;
constructor_range_stack = 0;
while (constructor_stack != range_stack->stack)
{
gcc_assert (constructor_stack->implicit);
process_init_element (loc,
pop_init_level (loc, 1,
braced_init_obstack,
last_init_list_comma),
true, braced_init_obstack);
}
for (p = range_stack;
!p->range_end || tree_int_cst_equal (p->index, p->range_end);
p = p->prev)
{
gcc_assert (constructor_stack->implicit);
process_init_element (loc,
pop_init_level (loc, 1,
braced_init_obstack,
last_init_list_comma),
true, braced_init_obstack);
}
p->index = size_binop_loc (input_location,
PLUS_EXPR, p->index, bitsize_one_node);
if (tree_int_cst_equal (p->index, p->range_end) && !p->prev)
finish = 1;
while (1)
{
constructor_index = p->index;
constructor_fields = p->fields;
if (finish && p->range_end && p->index == p->range_start)
{
finish = 0;
p->prev = 0;
}
p = p->next;
if (!p)
break;
finish_implicit_inits (loc, braced_init_obstack);
push_init_level (loc, 2, braced_init_obstack);
p->stack = constructor_stack;
if (p->range_end && tree_int_cst_equal (p->index, p->range_end))
p->index = p->range_start;
}
if (!finish)
constructor_range_stack = range_stack;
continue;
}
break;
}
constructor_range_stack = 0;
}

tree
build_asm_stmt (bool is_volatile, tree args)
{
if (is_volatile)
ASM_VOLATILE_P (args) = 1;
return add_stmt (args);
}
tree
build_asm_expr (location_t loc, tree string, tree outputs, tree inputs,
tree clobbers, tree labels, bool simple, bool is_inline)
{
tree tail;
tree args;
int i;
const char *constraint;
const char **oconstraints;
bool allows_mem, allows_reg, is_inout;
int ninputs, noutputs;
ninputs = list_length (inputs);
noutputs = list_length (outputs);
oconstraints = (const char **) alloca (noutputs * sizeof (const char *));
string = resolve_asm_operand_names (string, outputs, inputs, labels);
for (i = 0, tail = outputs; tail; ++i, tail = TREE_CHAIN (tail))
{
tree output = TREE_VALUE (tail);
output = c_fully_fold (output, false, NULL, true);
STRIP_NOPS (output);
if (!lvalue_or_else (loc, output, lv_asm))
output = error_mark_node;
if (output != error_mark_node
&& (TREE_READONLY (output)
|| TYPE_READONLY (TREE_TYPE (output))
|| (RECORD_OR_UNION_TYPE_P (TREE_TYPE (output))
&& C_TYPE_FIELDS_READONLY (TREE_TYPE (output)))))
readonly_error (loc, output, lv_asm);
constraint = TREE_STRING_POINTER (TREE_VALUE (TREE_PURPOSE (tail)));
oconstraints[i] = constraint;
if (parse_output_constraint (&constraint, i, ninputs, noutputs,
&allows_mem, &allows_reg, &is_inout))
{
if (!allows_reg && !c_mark_addressable (output))
output = error_mark_node;
if (!(!allows_reg && allows_mem)
&& output != error_mark_node
&& VOID_TYPE_P (TREE_TYPE (output)))
{
error_at (loc, "invalid use of void expression");
output = error_mark_node;
}
}
else
output = error_mark_node;
TREE_VALUE (tail) = output;
}
for (i = 0, tail = inputs; tail; ++i, tail = TREE_CHAIN (tail))
{
tree input;
constraint = TREE_STRING_POINTER (TREE_VALUE (TREE_PURPOSE (tail)));
input = TREE_VALUE (tail);
if (parse_input_constraint (&constraint, i, ninputs, noutputs, 0,
oconstraints, &allows_mem, &allows_reg))
{
if (!allows_reg && allows_mem)
{
input = c_fully_fold (input, false, NULL, true);
STRIP_NOPS (input);
if (!c_mark_addressable (input))
input = error_mark_node;
}
else
{
struct c_expr expr;
memset (&expr, 0, sizeof (expr));
expr.value = input;
expr = convert_lvalue_to_rvalue (loc, expr, true, false);
input = c_fully_fold (expr.value, false, NULL);
if (input != error_mark_node && VOID_TYPE_P (TREE_TYPE (input)))
{
error_at (loc, "invalid use of void expression");
input = error_mark_node;
}
}
}
else
input = error_mark_node;
TREE_VALUE (tail) = input;
}
gcc_assert (outputs == NULL || labels == NULL);
args = build_stmt (loc, ASM_EXPR, string, outputs, inputs, clobbers, labels);
ASM_INPUT_P (args) = simple;
ASM_VOLATILE_P (args) = (noutputs == 0);
ASM_INLINE_P (args) = is_inline;
return args;
}

tree
c_finish_goto_label (location_t loc, tree label)
{
tree decl = lookup_label_for_goto (loc, label);
if (!decl)
return NULL_TREE;
TREE_USED (decl) = 1;
{
add_stmt (build_predict_expr (PRED_GOTO, NOT_TAKEN));
tree t = build1 (GOTO_EXPR, void_type_node, decl);
SET_EXPR_LOCATION (t, loc);
return add_stmt (t);
}
}
tree
c_finish_goto_ptr (location_t loc, tree expr)
{
tree t;
pedwarn (loc, OPT_Wpedantic, "ISO C forbids %<goto *expr;%>");
expr = c_fully_fold (expr, false, NULL);
expr = convert (ptr_type_node, expr);
t = build1 (GOTO_EXPR, void_type_node, expr);
SET_EXPR_LOCATION (t, loc);
return add_stmt (t);
}
tree
c_finish_return (location_t loc, tree retval, tree origtype)
{
tree valtype = TREE_TYPE (TREE_TYPE (current_function_decl)), ret_stmt;
bool no_warning = false;
bool npc = false;
source_location xloc = expansion_point_location_if_in_system_header (loc);
if (TREE_THIS_VOLATILE (current_function_decl))
warning_at (xloc, 0,
"function declared %<noreturn%> has a %<return%> statement");
if (retval)
{
tree semantic_type = NULL_TREE;
npc = null_pointer_constant_p (retval);
if (TREE_CODE (retval) == EXCESS_PRECISION_EXPR)
{
semantic_type = TREE_TYPE (retval);
retval = TREE_OPERAND (retval, 0);
}
retval = c_fully_fold (retval, false, NULL);
if (semantic_type)
retval = build1 (EXCESS_PRECISION_EXPR, semantic_type, retval);
}
if (!retval)
{
current_function_returns_null = 1;
if ((warn_return_type || flag_isoc99)
&& valtype != NULL_TREE && TREE_CODE (valtype) != VOID_TYPE)
{
bool warned_here;
if (flag_isoc99)
warned_here = pedwarn
(loc, 0,
"%<return%> with no value, in function returning non-void");
else
warned_here = warning_at
(loc, OPT_Wreturn_type,
"%<return%> with no value, in function returning non-void");
no_warning = true;
if (warned_here)
inform (DECL_SOURCE_LOCATION (current_function_decl),
"declared here");
}
}
else if (valtype == NULL_TREE || TREE_CODE (valtype) == VOID_TYPE)
{
current_function_returns_null = 1;
bool warned_here;
if (TREE_CODE (TREE_TYPE (retval)) != VOID_TYPE)
warned_here = pedwarn
(xloc, 0,
"%<return%> with a value, in function returning void");
else
warned_here = pedwarn
(xloc, OPT_Wpedantic, "ISO C forbids "
"%<return%> with expression, in function returning void");
if (warned_here)
inform (DECL_SOURCE_LOCATION (current_function_decl),
"declared here");
}
else
{
tree t = convert_for_assignment (loc, UNKNOWN_LOCATION, valtype,
retval, origtype, ic_return,
npc, NULL_TREE, NULL_TREE, 0);
tree res = DECL_RESULT (current_function_decl);
tree inner;
bool save;
current_function_returns_value = 1;
if (t == error_mark_node)
return NULL_TREE;
save = in_late_binary_op;
if (TREE_CODE (TREE_TYPE (res)) == BOOLEAN_TYPE
|| TREE_CODE (TREE_TYPE (res)) == COMPLEX_TYPE
|| (TREE_CODE (TREE_TYPE (t)) == REAL_TYPE
&& (TREE_CODE (TREE_TYPE (res)) == INTEGER_TYPE
|| TREE_CODE (TREE_TYPE (res)) == ENUMERAL_TYPE)
&& sanitize_flags_p (SANITIZE_FLOAT_CAST)))
in_late_binary_op = true;
inner = t = convert (TREE_TYPE (res), t);
in_late_binary_op = save;
while (1)
{
switch (TREE_CODE (inner))
{
CASE_CONVERT:
case NON_LVALUE_EXPR:
case PLUS_EXPR:
case POINTER_PLUS_EXPR:
inner = TREE_OPERAND (inner, 0);
continue;
case MINUS_EXPR:
{
tree op1 = TREE_OPERAND (inner, 1);
while (!POINTER_TYPE_P (TREE_TYPE (op1))
&& (CONVERT_EXPR_P (op1)
|| TREE_CODE (op1) == NON_LVALUE_EXPR))
op1 = TREE_OPERAND (op1, 0);
if (POINTER_TYPE_P (TREE_TYPE (op1)))
break;
inner = TREE_OPERAND (inner, 0);
continue;
}
case ADDR_EXPR:
inner = TREE_OPERAND (inner, 0);
while (REFERENCE_CLASS_P (inner)
&& !INDIRECT_REF_P (inner))
inner = TREE_OPERAND (inner, 0);
if (DECL_P (inner)
&& !DECL_EXTERNAL (inner)
&& !TREE_STATIC (inner)
&& DECL_CONTEXT (inner) == current_function_decl)
{
if (TREE_CODE (inner) == LABEL_DECL)
warning_at (loc, OPT_Wreturn_local_addr,
"function returns address of label");
else
{
warning_at (loc, OPT_Wreturn_local_addr,
"function returns address of local variable");
tree zero = build_zero_cst (TREE_TYPE (res));
t = build2 (COMPOUND_EXPR, TREE_TYPE (res), t, zero);
}
}
break;
default:
break;
}
break;
}
retval = build2 (MODIFY_EXPR, TREE_TYPE (res), res, t);
SET_EXPR_LOCATION (retval, loc);
if (warn_sequence_point)
verify_sequence_points (retval);
}
ret_stmt = build_stmt (loc, RETURN_EXPR, retval);
TREE_NO_WARNING (ret_stmt) |= no_warning;
return add_stmt (ret_stmt);
}

struct c_switch {
tree switch_expr;
tree orig_type;
splay_tree cases;
struct c_spot_bindings *bindings;
struct c_switch *next;
bool bool_cond_p;
bool outside_range_p;
};
struct c_switch *c_switch_stack;
tree
c_start_case (location_t switch_loc,
location_t switch_cond_loc,
tree exp, bool explicit_cast_p)
{
tree orig_type = error_mark_node;
bool bool_cond_p = false;
struct c_switch *cs;
if (exp != error_mark_node)
{
orig_type = TREE_TYPE (exp);
if (!INTEGRAL_TYPE_P (orig_type))
{
if (orig_type != error_mark_node)
{
error_at (switch_cond_loc, "switch quantity not an integer");
orig_type = error_mark_node;
}
exp = integer_zero_node;
}
else
{
tree type = TYPE_MAIN_VARIANT (orig_type);
tree e = exp;
while (TREE_CODE (e) == COMPOUND_EXPR)
e = TREE_OPERAND (e, 1);
if ((TREE_CODE (type) == BOOLEAN_TYPE
|| truth_value_p (TREE_CODE (e)))
&& !(TREE_CODE (type) == INTEGER_TYPE
&& explicit_cast_p))
bool_cond_p = true;
if (!in_system_header_at (input_location)
&& (type == long_integer_type_node
|| type == long_unsigned_type_node))
warning_at (switch_cond_loc,
OPT_Wtraditional, "%<long%> switch expression not "
"converted to %<int%> in ISO C");
exp = c_fully_fold (exp, false, NULL);
exp = default_conversion (exp);
if (warn_sequence_point)
verify_sequence_points (exp);
}
}
cs = XNEW (struct c_switch);
cs->switch_expr = build2 (SWITCH_EXPR, orig_type, exp, NULL_TREE);
SET_EXPR_LOCATION (cs->switch_expr, switch_loc);
cs->orig_type = orig_type;
cs->cases = splay_tree_new (case_compare, NULL, NULL);
cs->bindings = c_get_switch_bindings ();
cs->bool_cond_p = bool_cond_p;
cs->outside_range_p = false;
cs->next = c_switch_stack;
c_switch_stack = cs;
return add_stmt (cs->switch_expr);
}
tree
do_case (location_t loc, tree low_value, tree high_value)
{
tree label = NULL_TREE;
if (low_value && TREE_CODE (low_value) != INTEGER_CST)
{
low_value = c_fully_fold (low_value, false, NULL);
if (TREE_CODE (low_value) == INTEGER_CST)
pedwarn (loc, OPT_Wpedantic,
"case label is not an integer constant expression");
}
if (high_value && TREE_CODE (high_value) != INTEGER_CST)
{
high_value = c_fully_fold (high_value, false, NULL);
if (TREE_CODE (high_value) == INTEGER_CST)
pedwarn (input_location, OPT_Wpedantic,
"case label is not an integer constant expression");
}
if (c_switch_stack == NULL)
{
if (low_value)
error_at (loc, "case label not within a switch statement");
else
error_at (loc, "%<default%> label not within a switch statement");
return NULL_TREE;
}
if (c_check_switch_jump_warnings (c_switch_stack->bindings,
EXPR_LOCATION (c_switch_stack->switch_expr),
loc))
return NULL_TREE;
label = c_add_case_label (loc, c_switch_stack->cases,
SWITCH_COND (c_switch_stack->switch_expr),
c_switch_stack->orig_type,
low_value, high_value,
&c_switch_stack->outside_range_p);
if (label == error_mark_node)
label = NULL_TREE;
return label;
}
void
c_finish_case (tree body, tree type)
{
struct c_switch *cs = c_switch_stack;
location_t switch_location;
SWITCH_BODY (cs->switch_expr) = body;
switch_location = EXPR_LOCATION (cs->switch_expr);
c_do_switch_warnings (cs->cases, switch_location,
type ? type : TREE_TYPE (cs->switch_expr),
SWITCH_COND (cs->switch_expr),
cs->bool_cond_p, cs->outside_range_p);
if (c_switch_covers_all_cases_p (cs->cases, TREE_TYPE (cs->switch_expr)))
SWITCH_ALL_CASES_P (cs->switch_expr) = 1;
c_switch_stack = cs->next;
splay_tree_delete (cs->cases);
c_release_switch_bindings (cs->bindings);
XDELETE (cs);
}

void
c_finish_if_stmt (location_t if_locus, tree cond, tree then_block,
tree else_block)
{
tree stmt;
stmt = build3 (COND_EXPR, void_type_node, cond, then_block, else_block);
SET_EXPR_LOCATION (stmt, if_locus);
add_stmt (stmt);
}
void
c_finish_loop (location_t start_locus, tree cond, tree incr, tree body,
tree blab, tree clab, bool cond_is_first)
{
tree entry = NULL, exit = NULL, t;
if (cond && integer_zerop (cond))
{
if (cond_is_first)
{
t = build_and_jump (&blab);
SET_EXPR_LOCATION (t, start_locus);
add_stmt (t);
}
}
else
{
tree top = build1 (LABEL_EXPR, void_type_node, NULL_TREE);
exit = build_and_jump (&LABEL_EXPR_LABEL (top));
if (cond && !integer_nonzerop (cond))
{
if (cond_is_first)
{
if (incr || !clab)
{
entry = build1 (LABEL_EXPR, void_type_node, NULL_TREE);
t = build_and_jump (&LABEL_EXPR_LABEL (entry));
}
else
t = build1 (GOTO_EXPR, void_type_node, clab);
SET_EXPR_LOCATION (t, start_locus);
add_stmt (t);
}
t = build_and_jump (&blab);
if (cond_is_first)
exit = fold_build3_loc (start_locus,
COND_EXPR, void_type_node, cond, exit, t);
else
exit = fold_build3_loc (input_location,
COND_EXPR, void_type_node, cond, exit, t);
}
else
{
location_t loc = EXPR_LOCATION (expr_first (body));
if (loc == UNKNOWN_LOCATION)
loc = start_locus;
SET_EXPR_LOCATION (exit, loc);
}
add_stmt (top);
}
if (body)
add_stmt (body);
if (clab)
add_stmt (build1 (LABEL_EXPR, void_type_node, clab));
if (incr)
add_stmt (incr);
if (entry)
add_stmt (entry);
if (exit)
add_stmt (exit);
if (blab)
add_stmt (build1 (LABEL_EXPR, void_type_node, blab));
}
tree
c_finish_bc_stmt (location_t loc, tree *label_p, bool is_break)
{
bool skip;
tree label = *label_p;
skip = !block_may_fallthru (cur_stmt_list);
if (!label)
{
if (!skip)
*label_p = label = create_artificial_label (loc);
}
else if (TREE_CODE (label) == LABEL_DECL)
;
else switch (TREE_INT_CST_LOW (label))
{
case 0:
if (is_break)
error_at (loc, "break statement not within loop or switch");
else
error_at (loc, "continue statement not within a loop");
return NULL_TREE;
case 1:
gcc_assert (is_break);
error_at (loc, "break statement used with OpenMP for loop");
return NULL_TREE;
case 2:
if (is_break) 
error ("break statement within %<#pragma simd%> loop body");
else 
error ("continue statement within %<#pragma simd%> loop body");
return NULL_TREE;
default:
gcc_unreachable ();
}
if (skip)
return NULL_TREE;
if (!is_break)
add_stmt (build_predict_expr (PRED_CONTINUE, NOT_TAKEN));
return add_stmt (build1 (GOTO_EXPR, void_type_node, label));
}
static void
emit_side_effect_warnings (location_t loc, tree expr)
{
if (expr == error_mark_node)
;
else if (!TREE_SIDE_EFFECTS (expr))
{
if (!VOID_TYPE_P (TREE_TYPE (expr)) && !TREE_NO_WARNING (expr))
warning_at (loc, OPT_Wunused_value, "statement with no effect");
}
else if (TREE_CODE (expr) == COMPOUND_EXPR)
{
tree r = expr;
location_t cloc = loc;
while (TREE_CODE (r) == COMPOUND_EXPR)
{
if (EXPR_HAS_LOCATION (r))
cloc = EXPR_LOCATION (r);
r = TREE_OPERAND (r, 1);
}
if (!TREE_SIDE_EFFECTS (r)
&& !VOID_TYPE_P (TREE_TYPE (r))
&& !CONVERT_EXPR_P (r)
&& !TREE_NO_WARNING (r)
&& !TREE_NO_WARNING (expr))
warning_at (cloc, OPT_Wunused_value,
"right-hand operand of comma expression has no effect");
}
else
warn_if_unused_value (expr, loc);
}
tree
c_process_expr_stmt (location_t loc, tree expr)
{
tree exprv;
if (!expr)
return NULL_TREE;
expr = c_fully_fold (expr, false, NULL);
if (warn_sequence_point)
verify_sequence_points (expr);
if (TREE_TYPE (expr) != error_mark_node
&& !COMPLETE_OR_VOID_TYPE_P (TREE_TYPE (expr))
&& TREE_CODE (TREE_TYPE (expr)) != ARRAY_TYPE)
error_at (loc, "expression statement has incomplete type");
if (!STATEMENT_LIST_STMT_EXPR (cur_stmt_list)
&& warn_unused_value)
emit_side_effect_warnings (EXPR_LOC_OR_LOC (expr, loc), expr);
exprv = expr;
while (TREE_CODE (exprv) == COMPOUND_EXPR)
exprv = TREE_OPERAND (exprv, 1);
while (CONVERT_EXPR_P (exprv))
exprv = TREE_OPERAND (exprv, 0);
if (DECL_P (exprv)
|| handled_component_p (exprv)
|| TREE_CODE (exprv) == ADDR_EXPR)
mark_exp_read (exprv);
if (DECL_P (expr) || CONSTANT_CLASS_P (expr))
{
expr = build1 (NOP_EXPR, TREE_TYPE (expr), expr);
SET_EXPR_LOCATION (expr, loc);
}
return expr;
}
tree
c_finish_expr_stmt (location_t loc, tree expr)
{
if (expr)
return add_stmt (c_process_expr_stmt (loc, expr));
else
return NULL;
}
tree
c_begin_stmt_expr (void)
{
tree ret;
keep_next_level ();
ret = c_begin_compound_stmt (true);
c_bindings_start_stmt_expr (c_switch_stack == NULL
? NULL
: c_switch_stack->bindings);
STATEMENT_LIST_STMT_EXPR (ret) = 1;
return ret;
}
tree
c_finish_stmt_expr (location_t loc, tree body)
{
tree last, type, tmp, val;
tree *last_p;
body = c_end_compound_stmt (loc, body, true);
c_bindings_end_stmt_expr (c_switch_stack == NULL
? NULL
: c_switch_stack->bindings);
last_p = &BIND_EXPR_BODY (body);
last = BIND_EXPR_BODY (body);
continue_searching:
if (TREE_CODE (last) == STATEMENT_LIST)
{
tree_stmt_iterator l = tsi_last (last);
while (!tsi_end_p (l) && TREE_CODE (tsi_stmt (l)) == DEBUG_BEGIN_STMT)
tsi_prev (&l);
if (tsi_end_p (l))
return body;
if (warn_unused_value)
{
for (tree_stmt_iterator i = tsi_start (last);
tsi_stmt (i) != tsi_stmt (l); tsi_next (&i))
{
location_t tloc;
tree t = tsi_stmt (i);
tloc = EXPR_HAS_LOCATION (t) ? EXPR_LOCATION (t) : loc;
emit_side_effect_warnings (tloc, t);
}
}
last_p = tsi_stmt_ptr (l);
last = *last_p;
}
if (TREE_CODE (last) == TRY_FINALLY_EXPR
|| TREE_CODE (last) == TRY_CATCH_EXPR)
{
last_p = &TREE_OPERAND (last, 0);
last = *last_p;
goto continue_searching;
}
if (last == error_mark_node)
return last;
if ((last == BIND_EXPR_BODY (body)
|| last == expr_first (BIND_EXPR_BODY (body)))
&& BIND_EXPR_VARS (body) == NULL)
{
last = c_wrap_maybe_const (last, true);
TREE_NO_WARNING (last) = 1;
return last;
}
type = TREE_TYPE (last);
if (!type || VOID_TYPE_P (type))
return body;
tmp = create_tmp_var_raw (type);
val = last;
if (TREE_CODE (val) == NOP_EXPR
&& TREE_TYPE (val) == TREE_TYPE (TREE_OPERAND (val, 0)))
val = TREE_OPERAND (val, 0);
*last_p = build2 (MODIFY_EXPR, void_type_node, tmp, val);
SET_EXPR_LOCATION (*last_p, EXPR_LOCATION (last));
{
tree t = build4 (TARGET_EXPR, type, tmp, body, NULL_TREE, NULL_TREE);
SET_EXPR_LOCATION (t, loc);
return t;
}
}

tree
c_begin_compound_stmt (bool do_scope)
{
tree stmt = push_stmt_list ();
if (do_scope)
push_scope ();
return stmt;
}
tree
c_end_compound_stmt (location_t loc, tree stmt, bool do_scope)
{
tree block = NULL;
if (do_scope)
{
if (c_dialect_objc ())
objc_clear_super_receiver ();
block = pop_scope ();
}
stmt = pop_stmt_list (stmt);
stmt = c_build_bind_expr (loc, block, stmt);
if (building_stmt_list_p ()
&& STATEMENT_LIST_STMT_EXPR (cur_stmt_list)
&& TREE_CODE (stmt) != BIND_EXPR)
{
stmt = build3 (BIND_EXPR, void_type_node, NULL, stmt, NULL);
TREE_SIDE_EFFECTS (stmt) = 1;
SET_EXPR_LOCATION (stmt, loc);
}
return stmt;
}
void
push_cleanup (tree decl, tree cleanup, bool eh_only)
{
enum tree_code code;
tree stmt, list;
bool stmt_expr;
code = eh_only ? TRY_CATCH_EXPR : TRY_FINALLY_EXPR;
stmt = build_stmt (DECL_SOURCE_LOCATION (decl), code, NULL, cleanup);
add_stmt (stmt);
stmt_expr = STATEMENT_LIST_STMT_EXPR (cur_stmt_list);
list = push_stmt_list ();
TREE_OPERAND (stmt, 0) = list;
STATEMENT_LIST_STMT_EXPR (list) = stmt_expr;
}

static tree
build_vec_cmp (tree_code code, tree type,
tree arg0, tree arg1)
{
tree zero_vec = build_zero_cst (type);
tree minus_one_vec = build_minus_one_cst (type);
tree cmp_type = build_same_sized_truth_vector_type (type);
tree cmp = build2 (code, cmp_type, arg0, arg1);
return build3 (VEC_COND_EXPR, type, cmp, minus_one_vec, zero_vec);
}
tree
build_binary_op (location_t location, enum tree_code code,
tree orig_op0, tree orig_op1, bool convert_p)
{
tree type0, type1, orig_type0, orig_type1;
tree eptype;
enum tree_code code0, code1;
tree op0, op1;
tree ret = error_mark_node;
const char *invalid_op_diag;
bool op0_int_operands, op1_int_operands;
bool int_const, int_const_or_overflow, int_operands;
enum tree_code resultcode = code;
tree result_type = NULL;
tree semantic_result_type = NULL;
int converted = 0;
tree build_type = NULL_TREE;
tree final_type = NULL_TREE;
int shorten = 0;
int short_compare = 0;
int short_shift = 0;
int common = 0;
bool objc_ok;
bool may_need_excess_precision;
bool boolean_op = false;
bool doing_div_or_mod = false;
bool doing_shift = false;
tree instrument_expr = NULL;
if (location == UNKNOWN_LOCATION)
location = input_location;
op0 = orig_op0;
op1 = orig_op1;
op0_int_operands = EXPR_INT_CONST_OPERANDS (orig_op0);
if (op0_int_operands)
op0 = remove_c_maybe_const_expr (op0);
op1_int_operands = EXPR_INT_CONST_OPERANDS (orig_op1);
if (op1_int_operands)
op1 = remove_c_maybe_const_expr (op1);
int_operands = (op0_int_operands && op1_int_operands);
if (int_operands)
{
int_const_or_overflow = (TREE_CODE (orig_op0) == INTEGER_CST
&& TREE_CODE (orig_op1) == INTEGER_CST);
int_const = (int_const_or_overflow
&& !TREE_OVERFLOW (orig_op0)
&& !TREE_OVERFLOW (orig_op1));
}
else
int_const = int_const_or_overflow = false;
if (convert_p
&& VECTOR_TYPE_P (TREE_TYPE (op0)) == VECTOR_TYPE_P (TREE_TYPE (op1)))
{
op0 = default_conversion (op0);
op1 = default_conversion (op1);
}
orig_type0 = type0 = TREE_TYPE (op0);
orig_type1 = type1 = TREE_TYPE (op1);
code0 = TREE_CODE (type0);
code1 = TREE_CODE (type1);
STRIP_TYPE_NOPS (op0);
STRIP_TYPE_NOPS (op1);
if (code0 == ERROR_MARK || code1 == ERROR_MARK)
return error_mark_node;
if (code0 == POINTER_TYPE
&& reject_gcc_builtin (op0, EXPR_LOCATION (orig_op0)))
return error_mark_node;
if (code1 == POINTER_TYPE
&& reject_gcc_builtin (op1, EXPR_LOCATION (orig_op1)))
return error_mark_node;
if ((invalid_op_diag
= targetm.invalid_binary_op (code, type0, type1)))
{
error_at (location, invalid_op_diag);
return error_mark_node;
}
switch (code)
{
case PLUS_EXPR:
case MINUS_EXPR:
case MULT_EXPR:
case TRUNC_DIV_EXPR:
case CEIL_DIV_EXPR:
case FLOOR_DIV_EXPR:
case ROUND_DIV_EXPR:
case EXACT_DIV_EXPR:
may_need_excess_precision = true;
break;
default:
may_need_excess_precision = false;
break;
}
if (TREE_CODE (op0) == EXCESS_PRECISION_EXPR)
{
op0 = TREE_OPERAND (op0, 0);
type0 = TREE_TYPE (op0);
}
else if (may_need_excess_precision
&& (eptype = excess_precision_type (type0)) != NULL_TREE)
{
type0 = eptype;
op0 = convert (eptype, op0);
}
if (TREE_CODE (op1) == EXCESS_PRECISION_EXPR)
{
op1 = TREE_OPERAND (op1, 0);
type1 = TREE_TYPE (op1);
}
else if (may_need_excess_precision
&& (eptype = excess_precision_type (type1)) != NULL_TREE)
{
type1 = eptype;
op1 = convert (eptype, op1);
}
objc_ok = objc_compare_types (type0, type1, -3, NULL_TREE);
if ((code0 == VECTOR_TYPE) != (code1 == VECTOR_TYPE))
{
enum stv_conv convert_flag = scalar_to_vector (location, code, op0, op1,
true);
switch (convert_flag)
{
case stv_error:
return error_mark_node;
case stv_firstarg:
{
bool maybe_const = true;
tree sc;
sc = c_fully_fold (op0, false, &maybe_const);
sc = save_expr (sc);
sc = convert (TREE_TYPE (type1), sc);
op0 = build_vector_from_val (type1, sc);
if (!maybe_const)
op0 = c_wrap_maybe_const (op0, true);
orig_type0 = type0 = TREE_TYPE (op0);
code0 = TREE_CODE (type0);
converted = 1;
break;
}
case stv_secondarg:
{
bool maybe_const = true;
tree sc;
sc = c_fully_fold (op1, false, &maybe_const);
sc = save_expr (sc);
sc = convert (TREE_TYPE (type0), sc);
op1 = build_vector_from_val (type0, sc);
if (!maybe_const)
op1 = c_wrap_maybe_const (op1, true);
orig_type1 = type1 = TREE_TYPE (op1);
code1 = TREE_CODE (type1);
converted = 1;
break;
}
default:
break;
}
}
switch (code)
{
case PLUS_EXPR:
if (code0 == POINTER_TYPE && code1 == INTEGER_TYPE)
{
ret = pointer_int_sum (location, PLUS_EXPR, op0, op1);
goto return_build_binary_op;
}
else if (code1 == POINTER_TYPE && code0 == INTEGER_TYPE)
{
ret = pointer_int_sum (location, PLUS_EXPR, op1, op0);
goto return_build_binary_op;
}
else
common = 1;
break;
case MINUS_EXPR:
if (code0 == POINTER_TYPE && code1 == POINTER_TYPE
&& comp_target_types (location, type0, type1))
{
ret = pointer_diff (location, op0, op1, &instrument_expr);
goto return_build_binary_op;
}
else if (code0 == POINTER_TYPE && code1 == INTEGER_TYPE)
{
ret = pointer_int_sum (location, MINUS_EXPR, op0, op1);
goto return_build_binary_op;
}
else
common = 1;
break;
case MULT_EXPR:
common = 1;
break;
case TRUNC_DIV_EXPR:
case CEIL_DIV_EXPR:
case FLOOR_DIV_EXPR:
case ROUND_DIV_EXPR:
case EXACT_DIV_EXPR:
doing_div_or_mod = true;
warn_for_div_by_zero (location, op1);
if ((code0 == INTEGER_TYPE || code0 == REAL_TYPE
|| code0 == FIXED_POINT_TYPE
|| code0 == COMPLEX_TYPE || code0 == VECTOR_TYPE)
&& (code1 == INTEGER_TYPE || code1 == REAL_TYPE
|| code1 == FIXED_POINT_TYPE
|| code1 == COMPLEX_TYPE || code1 == VECTOR_TYPE))
{
enum tree_code tcode0 = code0, tcode1 = code1;
if (code0 == COMPLEX_TYPE || code0 == VECTOR_TYPE)
tcode0 = TREE_CODE (TREE_TYPE (TREE_TYPE (op0)));
if (code1 == COMPLEX_TYPE || code1 == VECTOR_TYPE)
tcode1 = TREE_CODE (TREE_TYPE (TREE_TYPE (op1)));
if (!((tcode0 == INTEGER_TYPE && tcode1 == INTEGER_TYPE)
|| (tcode0 == FIXED_POINT_TYPE && tcode1 == FIXED_POINT_TYPE)))
resultcode = RDIV_EXPR;
else
shorten = (TYPE_UNSIGNED (TREE_TYPE (orig_op0))
|| (TREE_CODE (op1) == INTEGER_CST
&& !integer_all_onesp (op1)));
common = 1;
}
break;
case BIT_AND_EXPR:
case BIT_IOR_EXPR:
case BIT_XOR_EXPR:
if (code0 == INTEGER_TYPE && code1 == INTEGER_TYPE)
shorten = -1;
else if (code0 == VECTOR_TYPE
&& code1 == VECTOR_TYPE
&& !VECTOR_FLOAT_TYPE_P (type0)
&& !VECTOR_FLOAT_TYPE_P (type1))
common = 1;
break;
case TRUNC_MOD_EXPR:
case FLOOR_MOD_EXPR:
doing_div_or_mod = true;
warn_for_div_by_zero (location, op1);
if (code0 == VECTOR_TYPE && code1 == VECTOR_TYPE
&& TREE_CODE (TREE_TYPE (type0)) == INTEGER_TYPE
&& TREE_CODE (TREE_TYPE (type1)) == INTEGER_TYPE)
common = 1;
else if (code0 == INTEGER_TYPE && code1 == INTEGER_TYPE)
{
shorten = (TYPE_UNSIGNED (TREE_TYPE (orig_op0))
|| (TREE_CODE (op1) == INTEGER_CST
&& !integer_all_onesp (op1)));
common = 1;
}
break;
case TRUTH_ANDIF_EXPR:
case TRUTH_ORIF_EXPR:
case TRUTH_AND_EXPR:
case TRUTH_OR_EXPR:
case TRUTH_XOR_EXPR:
if ((code0 == INTEGER_TYPE || code0 == POINTER_TYPE
|| code0 == REAL_TYPE || code0 == COMPLEX_TYPE
|| code0 == FIXED_POINT_TYPE)
&& (code1 == INTEGER_TYPE || code1 == POINTER_TYPE
|| code1 == REAL_TYPE || code1 == COMPLEX_TYPE
|| code1 == FIXED_POINT_TYPE))
{
result_type = integer_type_node;
if (op0_int_operands)
{
op0 = c_objc_common_truthvalue_conversion (location, orig_op0);
op0 = remove_c_maybe_const_expr (op0);
}
else
op0 = c_objc_common_truthvalue_conversion (location, op0);
if (op1_int_operands)
{
op1 = c_objc_common_truthvalue_conversion (location, orig_op1);
op1 = remove_c_maybe_const_expr (op1);
}
else
op1 = c_objc_common_truthvalue_conversion (location, op1);
converted = 1;
boolean_op = true;
}
if (code == TRUTH_ANDIF_EXPR)
{
int_const_or_overflow = (int_operands
&& TREE_CODE (orig_op0) == INTEGER_CST
&& (op0 == truthvalue_false_node
|| TREE_CODE (orig_op1) == INTEGER_CST));
int_const = (int_const_or_overflow
&& !TREE_OVERFLOW (orig_op0)
&& (op0 == truthvalue_false_node
|| !TREE_OVERFLOW (orig_op1)));
}
else if (code == TRUTH_ORIF_EXPR)
{
int_const_or_overflow = (int_operands
&& TREE_CODE (orig_op0) == INTEGER_CST
&& (op0 == truthvalue_true_node
|| TREE_CODE (orig_op1) == INTEGER_CST));
int_const = (int_const_or_overflow
&& !TREE_OVERFLOW (orig_op0)
&& (op0 == truthvalue_true_node
|| !TREE_OVERFLOW (orig_op1)));
}
break;
case RSHIFT_EXPR:
if (code0 == VECTOR_TYPE && code1 == VECTOR_TYPE
&& TREE_CODE (TREE_TYPE (type0)) == INTEGER_TYPE
&& TREE_CODE (TREE_TYPE (type1)) == INTEGER_TYPE
&& known_eq (TYPE_VECTOR_SUBPARTS (type0),
TYPE_VECTOR_SUBPARTS (type1)))
{
result_type = type0;
converted = 1;
}
else if ((code0 == INTEGER_TYPE || code0 == FIXED_POINT_TYPE
|| (code0 == VECTOR_TYPE
&& TREE_CODE (TREE_TYPE (type0)) == INTEGER_TYPE))
&& code1 == INTEGER_TYPE)
{
doing_shift = true;
if (TREE_CODE (op1) == INTEGER_CST)
{
if (tree_int_cst_sgn (op1) < 0)
{
int_const = false;
if (c_inhibit_evaluation_warnings == 0)
warning_at (location, OPT_Wshift_count_negative,
"right shift count is negative");
}
else if (code0 == VECTOR_TYPE)
{
if (compare_tree_int (op1,
TYPE_PRECISION (TREE_TYPE (type0)))
>= 0)
{
int_const = false;
if (c_inhibit_evaluation_warnings == 0)
warning_at (location, OPT_Wshift_count_overflow,
"right shift count >= width of vector element");
}
}
else
{
if (!integer_zerop (op1))
short_shift = 1;
if (compare_tree_int (op1, TYPE_PRECISION (type0)) >= 0)
{
int_const = false;
if (c_inhibit_evaluation_warnings == 0)
warning_at (location, OPT_Wshift_count_overflow,
"right shift count >= width of type");
}
}
}
result_type = type0;
converted = 1;
}
break;
case LSHIFT_EXPR:
if (code0 == VECTOR_TYPE && code1 == VECTOR_TYPE
&& TREE_CODE (TREE_TYPE (type0)) == INTEGER_TYPE
&& TREE_CODE (TREE_TYPE (type1)) == INTEGER_TYPE
&& known_eq (TYPE_VECTOR_SUBPARTS (type0),
TYPE_VECTOR_SUBPARTS (type1)))
{
result_type = type0;
converted = 1;
}
else if ((code0 == INTEGER_TYPE || code0 == FIXED_POINT_TYPE
|| (code0 == VECTOR_TYPE
&& TREE_CODE (TREE_TYPE (type0)) == INTEGER_TYPE))
&& code1 == INTEGER_TYPE)
{
doing_shift = true;
if (TREE_CODE (op0) == INTEGER_CST
&& tree_int_cst_sgn (op0) < 0)
{
if (flag_isoc99)
int_const = false;
if (c_inhibit_evaluation_warnings == 0)
warning_at (location, OPT_Wshift_negative_value,
"left shift of negative value");
}
if (TREE_CODE (op1) == INTEGER_CST)
{
if (tree_int_cst_sgn (op1) < 0)
{
int_const = false;
if (c_inhibit_evaluation_warnings == 0)
warning_at (location, OPT_Wshift_count_negative,
"left shift count is negative");
}
else if (code0 == VECTOR_TYPE)
{
if (compare_tree_int (op1,
TYPE_PRECISION (TREE_TYPE (type0)))
>= 0)
{
int_const = false;
if (c_inhibit_evaluation_warnings == 0)
warning_at (location, OPT_Wshift_count_overflow,
"left shift count >= width of vector element");
}
}
else if (compare_tree_int (op1, TYPE_PRECISION (type0)) >= 0)
{
int_const = false;
if (c_inhibit_evaluation_warnings == 0)
warning_at (location, OPT_Wshift_count_overflow,
"left shift count >= width of type");
}
else if (TREE_CODE (op0) == INTEGER_CST
&& maybe_warn_shift_overflow (location, op0, op1)
&& flag_isoc99)
int_const = false;
}
result_type = type0;
converted = 1;
}
break;
case EQ_EXPR:
case NE_EXPR:
if (code0 == VECTOR_TYPE && code1 == VECTOR_TYPE)
{
tree intt;
if (!vector_types_compatible_elements_p (type0, type1))
{
error_at (location, "comparing vectors with different "
"element types");
return error_mark_node;
}
if (maybe_ne (TYPE_VECTOR_SUBPARTS (type0),
TYPE_VECTOR_SUBPARTS (type1)))
{
error_at (location, "comparing vectors with different "
"number of elements");
return error_mark_node;
}
if (TYPE_UNSIGNED (type0) != TYPE_UNSIGNED (type1))
{
if (!TYPE_UNSIGNED (type0))
op0 = build1 (VIEW_CONVERT_EXPR, type1, op0);
else
op1 = build1 (VIEW_CONVERT_EXPR, type0, op1);
warning_at (location, OPT_Wsign_compare, "comparison between "
"types %qT and %qT", type0, type1);
}
intt = c_common_type_for_size (GET_MODE_BITSIZE
(SCALAR_TYPE_MODE
(TREE_TYPE (type0))), 0);
if (!intt)
{
error_at (location, "could not find an integer type "
"of the same size as %qT",
TREE_TYPE (type0));
return error_mark_node;
}
result_type = build_opaque_vector_type (intt,
TYPE_VECTOR_SUBPARTS (type0));
converted = 1;
ret = build_vec_cmp (resultcode, result_type, op0, op1);
goto return_build_binary_op;
}
if (FLOAT_TYPE_P (type0) || FLOAT_TYPE_P (type1))
warning_at (location,
OPT_Wfloat_equal,
"comparing floating point with == or != is unsafe");
build_type = integer_type_node;
if ((code0 == INTEGER_TYPE || code0 == REAL_TYPE
|| code0 == FIXED_POINT_TYPE || code0 == COMPLEX_TYPE)
&& (code1 == INTEGER_TYPE || code1 == REAL_TYPE
|| code1 == FIXED_POINT_TYPE || code1 == COMPLEX_TYPE))
short_compare = 1;
else if (code0 == POINTER_TYPE && null_pointer_constant_p (orig_op1))
{
if (TREE_CODE (op0) == ADDR_EXPR
&& decl_with_nonnull_addr_p (TREE_OPERAND (op0, 0))
&& !from_macro_expansion_at (location))
{
if (code == EQ_EXPR)
warning_at (location,
OPT_Waddress,
"the comparison will always evaluate as %<false%> "
"for the address of %qD will never be NULL",
TREE_OPERAND (op0, 0));
else
warning_at (location,
OPT_Waddress,
"the comparison will always evaluate as %<true%> "
"for the address of %qD will never be NULL",
TREE_OPERAND (op0, 0));
}
result_type = type0;
}
else if (code1 == POINTER_TYPE && null_pointer_constant_p (orig_op0))
{
if (TREE_CODE (op1) == ADDR_EXPR
&& decl_with_nonnull_addr_p (TREE_OPERAND (op1, 0))
&& !from_macro_expansion_at (location))
{
if (code == EQ_EXPR)
warning_at (location,
OPT_Waddress,
"the comparison will always evaluate as %<false%> "
"for the address of %qD will never be NULL",
TREE_OPERAND (op1, 0));
else
warning_at (location,
OPT_Waddress,
"the comparison will always evaluate as %<true%> "
"for the address of %qD will never be NULL",
TREE_OPERAND (op1, 0));
}
result_type = type1;
}
else if (code0 == POINTER_TYPE && code1 == POINTER_TYPE)
{
tree tt0 = TREE_TYPE (type0);
tree tt1 = TREE_TYPE (type1);
addr_space_t as0 = TYPE_ADDR_SPACE (tt0);
addr_space_t as1 = TYPE_ADDR_SPACE (tt1);
addr_space_t as_common = ADDR_SPACE_GENERIC;
if (comp_target_types (location, type0, type1))
result_type = common_pointer_type (type0, type1);
else if (!addr_space_superset (as0, as1, &as_common))
{
error_at (location, "comparison of pointers to "
"disjoint address spaces");
return error_mark_node;
}
else if (VOID_TYPE_P (tt0) && !TYPE_ATOMIC (tt0))
{
if (pedantic && TREE_CODE (tt1) == FUNCTION_TYPE)
pedwarn (location, OPT_Wpedantic, "ISO C forbids "
"comparison of %<void *%> with function pointer");
}
else if (VOID_TYPE_P (tt1) && !TYPE_ATOMIC (tt1))
{
if (pedantic && TREE_CODE (tt0) == FUNCTION_TYPE)
pedwarn (location, OPT_Wpedantic, "ISO C forbids "
"comparison of %<void *%> with function pointer");
}
else
if (!objc_ok)
pedwarn (location, 0,
"comparison of distinct pointer types lacks a cast");
if (result_type == NULL_TREE)
{
int qual = ENCODE_QUAL_ADDR_SPACE (as_common);
result_type = build_pointer_type
(build_qualified_type (void_type_node, qual));
}
}
else if (code0 == POINTER_TYPE && code1 == INTEGER_TYPE)
{
result_type = type0;
pedwarn (location, 0, "comparison between pointer and integer");
}
else if (code0 == INTEGER_TYPE && code1 == POINTER_TYPE)
{
result_type = type1;
pedwarn (location, 0, "comparison between pointer and integer");
}
if ((TREE_CODE (TREE_TYPE (orig_op0)) == BOOLEAN_TYPE
|| truth_value_p (TREE_CODE (orig_op0)))
^ (TREE_CODE (TREE_TYPE (orig_op1)) == BOOLEAN_TYPE
|| truth_value_p (TREE_CODE (orig_op1))))
maybe_warn_bool_compare (location, code, orig_op0, orig_op1);
break;
case LE_EXPR:
case GE_EXPR:
case LT_EXPR:
case GT_EXPR:
if (code0 == VECTOR_TYPE && code1 == VECTOR_TYPE)
{
tree intt;
if (!vector_types_compatible_elements_p (type0, type1))
{
error_at (location, "comparing vectors with different "
"element types");
return error_mark_node;
}
if (maybe_ne (TYPE_VECTOR_SUBPARTS (type0),
TYPE_VECTOR_SUBPARTS (type1)))
{
error_at (location, "comparing vectors with different "
"number of elements");
return error_mark_node;
}
if (TYPE_UNSIGNED (type0) != TYPE_UNSIGNED (type1))
{
if (!TYPE_UNSIGNED (type0))
op0 = build1 (VIEW_CONVERT_EXPR, type1, op0);
else
op1 = build1 (VIEW_CONVERT_EXPR, type0, op1);
warning_at (location, OPT_Wsign_compare, "comparison between "
"types %qT and %qT", type0, type1);
}
intt = c_common_type_for_size (GET_MODE_BITSIZE
(SCALAR_TYPE_MODE
(TREE_TYPE (type0))), 0);
if (!intt)
{
error_at (location, "could not find an integer type "
"of the same size as %qT",
TREE_TYPE (type0));
return error_mark_node;
}
result_type = build_opaque_vector_type (intt,
TYPE_VECTOR_SUBPARTS (type0));
converted = 1;
ret = build_vec_cmp (resultcode, result_type, op0, op1);
goto return_build_binary_op;
}
build_type = integer_type_node;
if ((code0 == INTEGER_TYPE || code0 == REAL_TYPE
|| code0 == FIXED_POINT_TYPE)
&& (code1 == INTEGER_TYPE || code1 == REAL_TYPE
|| code1 == FIXED_POINT_TYPE))
short_compare = 1;
else if (code0 == POINTER_TYPE && code1 == POINTER_TYPE)
{
addr_space_t as0 = TYPE_ADDR_SPACE (TREE_TYPE (type0));
addr_space_t as1 = TYPE_ADDR_SPACE (TREE_TYPE (type1));
addr_space_t as_common;
if (comp_target_types (location, type0, type1))
{
result_type = common_pointer_type (type0, type1);
if (!COMPLETE_TYPE_P (TREE_TYPE (type0))
!= !COMPLETE_TYPE_P (TREE_TYPE (type1)))
pedwarn (location, 0,
"comparison of complete and incomplete pointers");
else if (TREE_CODE (TREE_TYPE (type0)) == FUNCTION_TYPE)
pedwarn (location, OPT_Wpedantic, "ISO C forbids "
"ordered comparisons of pointers to functions");
else if (null_pointer_constant_p (orig_op0)
|| null_pointer_constant_p (orig_op1))
warning_at (location, OPT_Wextra,
"ordered comparison of pointer with null pointer");
}
else if (!addr_space_superset (as0, as1, &as_common))
{
error_at (location, "comparison of pointers to "
"disjoint address spaces");
return error_mark_node;
}
else
{
int qual = ENCODE_QUAL_ADDR_SPACE (as_common);
result_type = build_pointer_type
(build_qualified_type (void_type_node, qual));
pedwarn (location, 0,
"comparison of distinct pointer types lacks a cast");
}
}
else if (code0 == POINTER_TYPE && null_pointer_constant_p (orig_op1))
{
result_type = type0;
if (pedantic)
pedwarn (location, OPT_Wpedantic,
"ordered comparison of pointer with integer zero");
else if (extra_warnings)
warning_at (location, OPT_Wextra,
"ordered comparison of pointer with integer zero");
}
else if (code1 == POINTER_TYPE && null_pointer_constant_p (orig_op0))
{
result_type = type1;
if (pedantic)
pedwarn (location, OPT_Wpedantic,
"ordered comparison of pointer with integer zero");
else if (extra_warnings)
warning_at (location, OPT_Wextra,
"ordered comparison of pointer with integer zero");
}
else if (code0 == POINTER_TYPE && code1 == INTEGER_TYPE)
{
result_type = type0;
pedwarn (location, 0, "comparison between pointer and integer");
}
else if (code0 == INTEGER_TYPE && code1 == POINTER_TYPE)
{
result_type = type1;
pedwarn (location, 0, "comparison between pointer and integer");
}
if ((code0 == POINTER_TYPE || code1 == POINTER_TYPE)
&& sanitize_flags_p (SANITIZE_POINTER_COMPARE))
{
op0 = save_expr (op0);
op1 = save_expr (op1);
tree tt = builtin_decl_explicit (BUILT_IN_ASAN_POINTER_COMPARE);
instrument_expr = build_call_expr_loc (location, tt, 2, op0, op1);
}
if ((TREE_CODE (TREE_TYPE (orig_op0)) == BOOLEAN_TYPE
|| truth_value_p (TREE_CODE (orig_op0)))
^ (TREE_CODE (TREE_TYPE (orig_op1)) == BOOLEAN_TYPE
|| truth_value_p (TREE_CODE (orig_op1))))
maybe_warn_bool_compare (location, code, orig_op0, orig_op1);
break;
default:
gcc_unreachable ();
}
if (code0 == ERROR_MARK || code1 == ERROR_MARK)
return error_mark_node;
if (code0 == VECTOR_TYPE && code1 == VECTOR_TYPE
&& (!tree_int_cst_equal (TYPE_SIZE (type0), TYPE_SIZE (type1))
|| !vector_types_compatible_elements_p (type0, type1)))
{
gcc_rich_location richloc (location);
richloc.maybe_add_expr (orig_op0);
richloc.maybe_add_expr (orig_op1);
binary_op_error (&richloc, code, type0, type1);
return error_mark_node;
}
if ((code0 == INTEGER_TYPE || code0 == REAL_TYPE || code0 == COMPLEX_TYPE
|| code0 == FIXED_POINT_TYPE || code0 == VECTOR_TYPE)
&&
(code1 == INTEGER_TYPE || code1 == REAL_TYPE || code1 == COMPLEX_TYPE
|| code1 == FIXED_POINT_TYPE || code1 == VECTOR_TYPE))
{
bool first_complex = (code0 == COMPLEX_TYPE);
bool second_complex = (code1 == COMPLEX_TYPE);
int none_complex = (!first_complex && !second_complex);
if (shorten || common || short_compare)
{
result_type = c_common_type (type0, type1);
do_warn_double_promotion (result_type, type0, type1,
"implicit conversion from %qT to %qT "
"to match other operand of binary "
"expression",
location);
if (result_type == error_mark_node)
return error_mark_node;
}
if (first_complex != second_complex
&& (code == PLUS_EXPR
|| code == MINUS_EXPR
|| code == MULT_EXPR
|| (code == TRUNC_DIV_EXPR && first_complex))
&& TREE_CODE (TREE_TYPE (result_type)) == REAL_TYPE
&& flag_signed_zeros)
{
tree real_type = TREE_TYPE (result_type);
tree real, imag;
if (type0 != orig_type0 || type1 != orig_type1)
{
gcc_assert (may_need_excess_precision && common);
semantic_result_type = c_common_type (orig_type0, orig_type1);
}
if (first_complex)
{
if (TREE_TYPE (op0) != result_type)
op0 = convert_and_check (location, result_type, op0);
if (TREE_TYPE (op1) != real_type)
op1 = convert_and_check (location, real_type, op1);
}
else
{
if (TREE_TYPE (op0) != real_type)
op0 = convert_and_check (location, real_type, op0);
if (TREE_TYPE (op1) != result_type)
op1 = convert_and_check (location, result_type, op1);
}
if (TREE_CODE (op0) == ERROR_MARK || TREE_CODE (op1) == ERROR_MARK)
return error_mark_node;
if (first_complex)
{
op0 = save_expr (op0);
real = build_unary_op (EXPR_LOCATION (orig_op0), REALPART_EXPR,
op0, true);
imag = build_unary_op (EXPR_LOCATION (orig_op0), IMAGPART_EXPR,
op0, true);
switch (code)
{
case MULT_EXPR:
case TRUNC_DIV_EXPR:
op1 = save_expr (op1);
imag = build2 (resultcode, real_type, imag, op1);
case PLUS_EXPR:
case MINUS_EXPR:
real = build2 (resultcode, real_type, real, op1);
break;
default:
gcc_unreachable();
}
}
else
{
op1 = save_expr (op1);
real = build_unary_op (EXPR_LOCATION (orig_op1), REALPART_EXPR,
op1, true);
imag = build_unary_op (EXPR_LOCATION (orig_op1), IMAGPART_EXPR,
op1, true);
switch (code)
{
case MULT_EXPR:
op0 = save_expr (op0);
imag = build2 (resultcode, real_type, op0, imag);
case PLUS_EXPR:
real = build2 (resultcode, real_type, op0, real);
break;
case MINUS_EXPR:
real = build2 (resultcode, real_type, op0, real);
imag = build1 (NEGATE_EXPR, real_type, imag);
break;
default:
gcc_unreachable();
}
}
ret = build2 (COMPLEX_EXPR, result_type, real, imag);
goto return_build_binary_op;
}
if (shorten && none_complex)
{
final_type = result_type;
result_type = shorten_binary_op (result_type, op0, op1,
shorten == -1);
}
if (short_shift)
{
int unsigned_arg;
tree arg0 = get_narrower (op0, &unsigned_arg);
final_type = result_type;
if (arg0 == op0 && final_type == TREE_TYPE (op0))
unsigned_arg = TYPE_UNSIGNED (TREE_TYPE (op0));
if (TYPE_PRECISION (TREE_TYPE (arg0)) < TYPE_PRECISION (result_type)
&& tree_int_cst_sgn (op1) > 0
&& compare_tree_int (op1, TYPE_PRECISION (TREE_TYPE (arg0))) < 0
&& (!TYPE_UNSIGNED (final_type) || unsigned_arg))
{
result_type
= c_common_signed_or_unsigned_type (unsigned_arg,
TREE_TYPE (arg0));
if (TREE_TYPE (op0) != result_type)
op0 = convert (result_type, op0);
converted = 1;
}
}
if (short_compare)
{
tree xop0 = op0, xop1 = op1, xresult_type = result_type;
enum tree_code xresultcode = resultcode;
tree val
= shorten_compare (location, &xop0, &xop1, &xresult_type,
&xresultcode);
if (val != NULL_TREE)
{
ret = val;
goto return_build_binary_op;
}
op0 = xop0, op1 = xop1;
converted = 1;
resultcode = xresultcode;
if (c_inhibit_evaluation_warnings == 0)
{
bool op0_maybe_const = true;
bool op1_maybe_const = true;
tree orig_op0_folded, orig_op1_folded;
if (in_late_binary_op)
{
orig_op0_folded = orig_op0;
orig_op1_folded = orig_op1;
}
else
{
c_inhibit_evaluation_warnings++;
op0 = c_fully_fold (op0, require_constant_value,
&op0_maybe_const);
op1 = c_fully_fold (op1, require_constant_value,
&op1_maybe_const);
c_inhibit_evaluation_warnings--;
orig_op0_folded = c_fully_fold (orig_op0,
require_constant_value,
NULL);
orig_op1_folded = c_fully_fold (orig_op1,
require_constant_value,
NULL);
}
if (warn_sign_compare)
warn_for_sign_compare (location, orig_op0_folded,
orig_op1_folded, op0, op1,
result_type, resultcode);
if (!in_late_binary_op && !int_operands)
{
if (!op0_maybe_const || TREE_CODE (op0) != INTEGER_CST)
op0 = c_wrap_maybe_const (op0, !op0_maybe_const);
if (!op1_maybe_const || TREE_CODE (op1) != INTEGER_CST)
op1 = c_wrap_maybe_const (op1, !op1_maybe_const);
}
}
}
}
if (!result_type)
{
gcc_rich_location richloc (location);
richloc.maybe_add_expr (orig_op0);
richloc.maybe_add_expr (orig_op1);
binary_op_error (&richloc, code, TREE_TYPE (op0), TREE_TYPE (op1));
return error_mark_node;
}
if (build_type == NULL_TREE)
{
build_type = result_type;
if ((type0 != orig_type0 || type1 != orig_type1)
&& !boolean_op)
{
gcc_assert (may_need_excess_precision && common);
semantic_result_type = c_common_type (orig_type0, orig_type1);
}
}
if (!converted)
{
op0 = ep_convert_and_check (location, result_type, op0,
semantic_result_type);
op1 = ep_convert_and_check (location, result_type, op1,
semantic_result_type);
if (TREE_CODE (op0) == ERROR_MARK || TREE_CODE (op1) == ERROR_MARK)
return error_mark_node;
}
if (sanitize_flags_p ((SANITIZE_SHIFT
| SANITIZE_DIVIDE | SANITIZE_FLOAT_DIVIDE))
&& current_function_decl != NULL_TREE
&& (doing_div_or_mod || doing_shift)
&& !require_constant_value)
{
op0 = save_expr (op0);
op1 = save_expr (op1);
op0 = c_fully_fold (op0, false, NULL);
op1 = c_fully_fold (op1, false, NULL);
if (doing_div_or_mod && (sanitize_flags_p ((SANITIZE_DIVIDE
| SANITIZE_FLOAT_DIVIDE))))
instrument_expr = ubsan_instrument_division (location, op0, op1);
else if (doing_shift && sanitize_flags_p (SANITIZE_SHIFT))
instrument_expr = ubsan_instrument_shift (location, code, op0, op1);
}
if (int_const_or_overflow)
ret = (require_constant_value
? fold_build2_initializer_loc (location, resultcode, build_type,
op0, op1)
: fold_build2_loc (location, resultcode, build_type, op0, op1));
else
ret = build2 (resultcode, build_type, op0, op1);
if (final_type != NULL_TREE)
ret = convert (final_type, ret);
return_build_binary_op:
gcc_assert (ret != error_mark_node);
if (TREE_CODE (ret) == INTEGER_CST && !TREE_OVERFLOW (ret) && !int_const)
ret = (int_operands
? note_integer_operands (ret)
: build1 (NOP_EXPR, TREE_TYPE (ret), ret));
else if (TREE_CODE (ret) != INTEGER_CST && int_operands
&& !in_late_binary_op)
ret = note_integer_operands (ret);
protected_set_expr_location (ret, location);
if (instrument_expr != NULL)
ret = fold_build2 (COMPOUND_EXPR, TREE_TYPE (ret),
instrument_expr, ret);
if (semantic_result_type)
ret = build1_loc (location, EXCESS_PRECISION_EXPR,
semantic_result_type, ret);
return ret;
}
tree
c_objc_common_truthvalue_conversion (location_t location, tree expr)
{
bool int_const, int_operands;
switch (TREE_CODE (TREE_TYPE (expr)))
{
case ARRAY_TYPE:
error_at (location, "used array that cannot be converted to pointer where scalar is required");
return error_mark_node;
case RECORD_TYPE:
error_at (location, "used struct type value where scalar is required");
return error_mark_node;
case UNION_TYPE:
error_at (location, "used union type value where scalar is required");
return error_mark_node;
case VOID_TYPE:
error_at (location, "void value not ignored as it ought to be");
return error_mark_node;
case POINTER_TYPE:
if (reject_gcc_builtin (expr))
return error_mark_node;
break;
case FUNCTION_TYPE:
gcc_unreachable ();
case VECTOR_TYPE:
error_at (location, "used vector type where scalar is required");
return error_mark_node;
default:
break;
}
int_const = (TREE_CODE (expr) == INTEGER_CST && !TREE_OVERFLOW (expr));
int_operands = EXPR_INT_CONST_OPERANDS (expr);
if (int_operands && TREE_CODE (expr) != INTEGER_CST)
{
expr = remove_c_maybe_const_expr (expr);
expr = build2 (NE_EXPR, integer_type_node, expr,
convert (TREE_TYPE (expr), integer_zero_node));
expr = note_integer_operands (expr);
}
else
expr = c_common_truthvalue_conversion (location, expr);
if (TREE_CODE (expr) == INTEGER_CST && int_operands && !int_const)
{
if (TREE_OVERFLOW (expr))
return expr;
else
return note_integer_operands (expr);
}
if (TREE_CODE (expr) == INTEGER_CST && !int_const)
return build1 (NOP_EXPR, TREE_TYPE (expr), expr);
return expr;
}

tree
c_expr_to_decl (tree expr, bool *tc ATTRIBUTE_UNUSED, bool *se)
{
if (TREE_CODE (expr) == COMPOUND_LITERAL_EXPR)
{
tree decl = COMPOUND_LITERAL_EXPR_DECL (expr);
if (!TREE_STATIC (decl))
*se = true;
return decl;
}
else
return expr;
}

tree
c_finish_omp_construct (location_t loc, enum tree_code code, tree body,
tree clauses)
{
body = c_end_compound_stmt (loc, body, true);
tree stmt = make_node (code);
TREE_TYPE (stmt) = void_type_node;
OMP_BODY (stmt) = body;
OMP_CLAUSES (stmt) = clauses;
SET_EXPR_LOCATION (stmt, loc);
return add_stmt (stmt);
}
tree
c_finish_oacc_data (location_t loc, tree clauses, tree block)
{
tree stmt;
block = c_end_compound_stmt (loc, block, true);
stmt = make_node (OACC_DATA);
TREE_TYPE (stmt) = void_type_node;
OACC_DATA_CLAUSES (stmt) = clauses;
OACC_DATA_BODY (stmt) = block;
SET_EXPR_LOCATION (stmt, loc);
return add_stmt (stmt);
}
tree
c_finish_oacc_host_data (location_t loc, tree clauses, tree block)
{
tree stmt;
block = c_end_compound_stmt (loc, block, true);
stmt = make_node (OACC_HOST_DATA);
TREE_TYPE (stmt) = void_type_node;
OACC_HOST_DATA_CLAUSES (stmt) = clauses;
OACC_HOST_DATA_BODY (stmt) = block;
SET_EXPR_LOCATION (stmt, loc);
return add_stmt (stmt);
}
tree
c_begin_omp_parallel (void)
{
tree block;
keep_next_level ();
block = c_begin_compound_stmt (true);
return block;
}
tree
c_finish_omp_parallel (location_t loc, tree clauses, tree block)
{
tree stmt;
block = c_end_compound_stmt (loc, block, true);
stmt = make_node (OMP_PARALLEL);
TREE_TYPE (stmt) = void_type_node;
OMP_PARALLEL_CLAUSES (stmt) = clauses;
OMP_PARALLEL_BODY (stmt) = block;
SET_EXPR_LOCATION (stmt, loc);
return add_stmt (stmt);
}
tree
c_begin_omp_task (void)
{
tree block;
keep_next_level ();
block = c_begin_compound_stmt (true);
return block;
}
tree
c_finish_omp_task (location_t loc, tree clauses, tree block)
{
tree stmt;
block = c_end_compound_stmt (loc, block, true);
stmt = make_node (OMP_TASK);
TREE_TYPE (stmt) = void_type_node;
OMP_TASK_CLAUSES (stmt) = clauses;
OMP_TASK_BODY (stmt) = block;
SET_EXPR_LOCATION (stmt, loc);
return add_stmt (stmt);
}
void
c_finish_omp_cancel (location_t loc, tree clauses)
{
tree fn = builtin_decl_explicit (BUILT_IN_GOMP_CANCEL);
int mask = 0;
if (omp_find_clause (clauses, OMP_CLAUSE_PARALLEL))
mask = 1;
else if (omp_find_clause (clauses, OMP_CLAUSE_FOR))
mask = 2;
else if (omp_find_clause (clauses, OMP_CLAUSE_SECTIONS))
mask = 4;
else if (omp_find_clause (clauses, OMP_CLAUSE_TASKGROUP))
mask = 8;
else
{
error_at (loc, "%<#pragma omp cancel%> must specify one of "
"%<parallel%>, %<for%>, %<sections%> or %<taskgroup%> "
"clauses");
return;
}
tree ifc = omp_find_clause (clauses, OMP_CLAUSE_IF);
if (ifc != NULL_TREE)
{
tree type = TREE_TYPE (OMP_CLAUSE_IF_EXPR (ifc));
ifc = fold_build2_loc (OMP_CLAUSE_LOCATION (ifc), NE_EXPR,
boolean_type_node, OMP_CLAUSE_IF_EXPR (ifc),
build_zero_cst (type));
}
else
ifc = boolean_true_node;
tree stmt = build_call_expr_loc (loc, fn, 2,
build_int_cst (integer_type_node, mask),
ifc);
add_stmt (stmt);
}
void
c_finish_omp_cancellation_point (location_t loc, tree clauses)
{
tree fn = builtin_decl_explicit (BUILT_IN_GOMP_CANCELLATION_POINT);
int mask = 0;
if (omp_find_clause (clauses, OMP_CLAUSE_PARALLEL))
mask = 1;
else if (omp_find_clause (clauses, OMP_CLAUSE_FOR))
mask = 2;
else if (omp_find_clause (clauses, OMP_CLAUSE_SECTIONS))
mask = 4;
else if (omp_find_clause (clauses, OMP_CLAUSE_TASKGROUP))
mask = 8;
else
{
error_at (loc, "%<#pragma omp cancellation point%> must specify one of "
"%<parallel%>, %<for%>, %<sections%> or %<taskgroup%> "
"clauses");
return;
}
tree stmt = build_call_expr_loc (loc, fn, 1,
build_int_cst (integer_type_node, mask));
add_stmt (stmt);
}
static tree
handle_omp_array_sections_1 (tree c, tree t, vec<tree> &types,
bool &maybe_zero_len, unsigned int &first_non_one,
enum c_omp_region_type ort)
{
tree ret, low_bound, length, type;
if (TREE_CODE (t) != TREE_LIST)
{
if (error_operand_p (t))
return error_mark_node;
ret = t;
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND
&& TYPE_ATOMIC (strip_array_types (TREE_TYPE (t))))
{
error_at (OMP_CLAUSE_LOCATION (c), "%<_Atomic%> %qE in %qs clause",
t, omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
if (TREE_CODE (t) == COMPONENT_REF
&& ort == C_ORT_OMP
&& (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_MAP
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_TO
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_FROM))
{
if (DECL_BIT_FIELD (TREE_OPERAND (t, 1)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"bit-field %qE in %qs clause",
t, omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
while (TREE_CODE (t) == COMPONENT_REF)
{
if (TREE_CODE (TREE_TYPE (TREE_OPERAND (t, 0))) == UNION_TYPE)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is a member of a union", t);
return error_mark_node;
}
t = TREE_OPERAND (t, 0);
}
}
if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL)
{
if (DECL_P (t))
error_at (OMP_CLAUSE_LOCATION (c),
"%qD is not a variable in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
else
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is not a variable in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
else if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND
&& TYPE_ATOMIC (TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c), "%<_Atomic%> %qD in %qs clause",
t, omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
else if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND
&& VAR_P (t)
&& DECL_THREAD_LOCAL_P (t))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qD is threadprivate variable in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_DEPEND
&& TYPE_ATOMIC (TREE_TYPE (t))
&& POINTER_TYPE_P (TREE_TYPE (t)))
{
c_expr expr;
memset (&expr, 0, sizeof (expr));
expr.value = ret;
expr = convert_lvalue_to_rvalue (OMP_CLAUSE_LOCATION (c),
expr, false, false);
ret = expr.value;
}
return ret;
}
ret = handle_omp_array_sections_1 (c, TREE_CHAIN (t), types,
maybe_zero_len, first_non_one, ort);
if (ret == error_mark_node || ret == NULL_TREE)
return ret;
type = TREE_TYPE (ret);
low_bound = TREE_PURPOSE (t);
length = TREE_VALUE (t);
if (low_bound == error_mark_node || length == error_mark_node)
return error_mark_node;
if (low_bound && !INTEGRAL_TYPE_P (TREE_TYPE (low_bound)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"low bound %qE of array section does not have integral type",
low_bound);
return error_mark_node;
}
if (length && !INTEGRAL_TYPE_P (TREE_TYPE (length)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"length %qE of array section does not have integral type",
length);
return error_mark_node;
}
if (low_bound
&& TREE_CODE (low_bound) == INTEGER_CST
&& TYPE_PRECISION (TREE_TYPE (low_bound))
> TYPE_PRECISION (sizetype))
low_bound = fold_convert (sizetype, low_bound);
if (length
&& TREE_CODE (length) == INTEGER_CST
&& TYPE_PRECISION (TREE_TYPE (length))
> TYPE_PRECISION (sizetype))
length = fold_convert (sizetype, length);
if (low_bound == NULL_TREE)
low_bound = integer_zero_node;
if (length != NULL_TREE)
{
if (!integer_nonzerop (length))
{
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_DEPEND
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION)
{
if (integer_zerop (length))
{
error_at (OMP_CLAUSE_LOCATION (c),
"zero length array section in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
}
else
maybe_zero_len = true;
}
if (first_non_one == types.length ()
&& (TREE_CODE (length) != INTEGER_CST || integer_onep (length)))
first_non_one++;
}
if (TREE_CODE (type) == ARRAY_TYPE)
{
if (length == NULL_TREE
&& (TYPE_DOMAIN (type) == NULL_TREE
|| TYPE_MAX_VALUE (TYPE_DOMAIN (type)) == NULL_TREE))
{
error_at (OMP_CLAUSE_LOCATION (c),
"for unknown bound array type length expression must "
"be specified");
return error_mark_node;
}
if (TREE_CODE (low_bound) == INTEGER_CST
&& tree_int_cst_sgn (low_bound) == -1)
{
error_at (OMP_CLAUSE_LOCATION (c),
"negative low bound in array section in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
if (length != NULL_TREE
&& TREE_CODE (length) == INTEGER_CST
&& tree_int_cst_sgn (length) == -1)
{
error_at (OMP_CLAUSE_LOCATION (c),
"negative length in array section in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
if (TYPE_DOMAIN (type)
&& TYPE_MAX_VALUE (TYPE_DOMAIN (type))
&& TREE_CODE (TYPE_MAX_VALUE (TYPE_DOMAIN (type)))
== INTEGER_CST)
{
tree size
= fold_convert (sizetype, TYPE_MAX_VALUE (TYPE_DOMAIN (type)));
size = size_binop (PLUS_EXPR, size, size_one_node);
if (TREE_CODE (low_bound) == INTEGER_CST)
{
if (tree_int_cst_lt (size, low_bound))
{
error_at (OMP_CLAUSE_LOCATION (c),
"low bound %qE above array section size "
"in %qs clause", low_bound,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
if (tree_int_cst_equal (size, low_bound))
{
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_DEPEND
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION)
{
error_at (OMP_CLAUSE_LOCATION (c),
"zero length array section in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
maybe_zero_len = true;
}
else if (length == NULL_TREE
&& first_non_one == types.length ()
&& tree_int_cst_equal
(TYPE_MAX_VALUE (TYPE_DOMAIN (type)),
low_bound))
first_non_one++;
}
else if (length == NULL_TREE)
{
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND
&& OMP_CLAUSE_CODE (c) != OMP_CLAUSE_REDUCTION)
maybe_zero_len = true;
if (first_non_one == types.length ())
first_non_one++;
}
if (length && TREE_CODE (length) == INTEGER_CST)
{
if (tree_int_cst_lt (size, length))
{
error_at (OMP_CLAUSE_LOCATION (c),
"length %qE above array section size "
"in %qs clause", length,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
if (TREE_CODE (low_bound) == INTEGER_CST)
{
tree lbpluslen
= size_binop (PLUS_EXPR,
fold_convert (sizetype, low_bound),
fold_convert (sizetype, length));
if (TREE_CODE (lbpluslen) == INTEGER_CST
&& tree_int_cst_lt (size, lbpluslen))
{
error_at (OMP_CLAUSE_LOCATION (c),
"high bound %qE above array section size "
"in %qs clause", lbpluslen,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
}
}
}
else if (length == NULL_TREE)
{
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND
&& OMP_CLAUSE_CODE (c) != OMP_CLAUSE_REDUCTION)
maybe_zero_len = true;
if (first_non_one == types.length ())
first_non_one++;
}
if (length == NULL_TREE && OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND)
{
tree lb = save_expr (low_bound);
if (lb != low_bound)
{
TREE_PURPOSE (t) = lb;
low_bound = lb;
}
}
}
else if (TREE_CODE (type) == POINTER_TYPE)
{
if (length == NULL_TREE)
{
error_at (OMP_CLAUSE_LOCATION (c),
"for pointer type length expression must be specified");
return error_mark_node;
}
if (length != NULL_TREE
&& TREE_CODE (length) == INTEGER_CST
&& tree_int_cst_sgn (length) == -1)
{
error_at (OMP_CLAUSE_LOCATION (c),
"negative length in array section in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND
&& TREE_CODE (TREE_CHAIN (t)) == TREE_LIST)
{
error_at (OMP_CLAUSE_LOCATION (c),
"array section is not contiguous in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
}
else
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE does not have pointer or array type", ret);
return error_mark_node;
}
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_DEPEND)
types.safe_push (TREE_TYPE (ret));
tree lb = save_expr (low_bound);
if (lb != low_bound)
{
TREE_PURPOSE (t) = lb;
low_bound = lb;
}
ret = build_array_ref (OMP_CLAUSE_LOCATION (c), ret, low_bound);
return ret;
}
static bool
handle_omp_array_sections (tree c, enum c_omp_region_type ort)
{
bool maybe_zero_len = false;
unsigned int first_non_one = 0;
auto_vec<tree, 10> types;
tree first = handle_omp_array_sections_1 (c, OMP_CLAUSE_DECL (c), types,
maybe_zero_len, first_non_one,
ort);
if (first == error_mark_node)
return true;
if (first == NULL_TREE)
return false;
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_DEPEND)
{
tree t = OMP_CLAUSE_DECL (c);
tree tem = NULL_TREE;
while (TREE_CODE (t) == TREE_LIST)
{
if (TREE_VALUE (t) && TREE_SIDE_EFFECTS (TREE_VALUE (t)))
{
if (tem == NULL_TREE)
tem = TREE_VALUE (t);
else
tem = build2 (COMPOUND_EXPR, TREE_TYPE (tem),
TREE_VALUE (t), tem);
}
t = TREE_CHAIN (t);
}
if (tem)
first = build2 (COMPOUND_EXPR, TREE_TYPE (first), tem, first);
first = c_fully_fold (first, false, NULL, true);
OMP_CLAUSE_DECL (c) = first;
}
else
{
unsigned int num = types.length (), i;
tree t, side_effects = NULL_TREE, size = NULL_TREE;
tree condition = NULL_TREE;
if (int_size_in_bytes (TREE_TYPE (first)) <= 0)
maybe_zero_len = true;
for (i = num, t = OMP_CLAUSE_DECL (c); i > 0;
t = TREE_CHAIN (t))
{
tree low_bound = TREE_PURPOSE (t);
tree length = TREE_VALUE (t);
i--;
if (low_bound
&& TREE_CODE (low_bound) == INTEGER_CST
&& TYPE_PRECISION (TREE_TYPE (low_bound))
> TYPE_PRECISION (sizetype))
low_bound = fold_convert (sizetype, low_bound);
if (length
&& TREE_CODE (length) == INTEGER_CST
&& TYPE_PRECISION (TREE_TYPE (length))
> TYPE_PRECISION (sizetype))
length = fold_convert (sizetype, length);
if (low_bound == NULL_TREE)
low_bound = integer_zero_node;
if (!maybe_zero_len && i > first_non_one)
{
if (integer_nonzerop (low_bound))
goto do_warn_noncontiguous;
if (length != NULL_TREE
&& TREE_CODE (length) == INTEGER_CST
&& TYPE_DOMAIN (types[i])
&& TYPE_MAX_VALUE (TYPE_DOMAIN (types[i]))
&& TREE_CODE (TYPE_MAX_VALUE (TYPE_DOMAIN (types[i])))
== INTEGER_CST)
{
tree size;
size = size_binop (PLUS_EXPR,
TYPE_MAX_VALUE (TYPE_DOMAIN (types[i])),
size_one_node);
if (!tree_int_cst_equal (length, size))
{
do_warn_noncontiguous:
error_at (OMP_CLAUSE_LOCATION (c),
"array section is not contiguous in %qs "
"clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return true;
}
}
if (length != NULL_TREE
&& TREE_SIDE_EFFECTS (length))
{
if (side_effects == NULL_TREE)
side_effects = length;
else
side_effects = build2 (COMPOUND_EXPR,
TREE_TYPE (side_effects),
length, side_effects);
}
}
else
{
tree l;
if (i > first_non_one
&& ((length && integer_nonzerop (length))
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION))
continue;
if (length)
l = fold_convert (sizetype, length);
else
{
l = size_binop (PLUS_EXPR,
TYPE_MAX_VALUE (TYPE_DOMAIN (types[i])),
size_one_node);
l = size_binop (MINUS_EXPR, l,
fold_convert (sizetype, low_bound));
}
if (i > first_non_one)
{
l = fold_build2 (NE_EXPR, boolean_type_node, l,
size_zero_node);
if (condition == NULL_TREE)
condition = l;
else
condition = fold_build2 (BIT_AND_EXPR, boolean_type_node,
l, condition);
}
else if (size == NULL_TREE)
{
size = size_in_bytes (TREE_TYPE (types[i]));
tree eltype = TREE_TYPE (types[num - 1]);
while (TREE_CODE (eltype) == ARRAY_TYPE)
eltype = TREE_TYPE (eltype);
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION)
{
if (integer_zerop (size)
|| integer_zerop (size_in_bytes (eltype)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"zero length array section in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
return error_mark_node;
}
size = size_binop (EXACT_DIV_EXPR, size,
size_in_bytes (eltype));
}
size = size_binop (MULT_EXPR, size, l);
if (condition)
size = fold_build3 (COND_EXPR, sizetype, condition,
size, size_zero_node);
}
else
size = size_binop (MULT_EXPR, size, l);
}
}
if (side_effects)
size = build2 (COMPOUND_EXPR, sizetype, side_effects, size);
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION)
{
size = size_binop (MINUS_EXPR, size, size_one_node);
size = c_fully_fold (size, false, NULL);
tree index_type = build_index_type (size);
tree eltype = TREE_TYPE (first);
while (TREE_CODE (eltype) == ARRAY_TYPE)
eltype = TREE_TYPE (eltype);
tree type = build_array_type (eltype, index_type);
tree ptype = build_pointer_type (eltype);
if (TREE_CODE (TREE_TYPE (t)) == ARRAY_TYPE)
t = build_fold_addr_expr (t);
tree t2 = build_fold_addr_expr (first);
t2 = fold_convert_loc (OMP_CLAUSE_LOCATION (c),
ptrdiff_type_node, t2);
t2 = fold_build2_loc (OMP_CLAUSE_LOCATION (c), MINUS_EXPR,
ptrdiff_type_node, t2,
fold_convert_loc (OMP_CLAUSE_LOCATION (c),
ptrdiff_type_node, t));
t2 = c_fully_fold (t2, false, NULL);
if (tree_fits_shwi_p (t2))
t = build2 (MEM_REF, type, t,
build_int_cst (ptype, tree_to_shwi (t2)));
else
{
t2 = fold_convert_loc (OMP_CLAUSE_LOCATION (c), sizetype, t2);
t = build2_loc (OMP_CLAUSE_LOCATION (c), POINTER_PLUS_EXPR,
TREE_TYPE (t), t, t2);
t = build2 (MEM_REF, type, t, build_int_cst (ptype, 0));
}
OMP_CLAUSE_DECL (c) = t;
return false;
}
first = c_fully_fold (first, false, NULL);
OMP_CLAUSE_DECL (c) = first;
if (size)
size = c_fully_fold (size, false, NULL);
OMP_CLAUSE_SIZE (c) = size;
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_MAP
|| (TREE_CODE (t) == COMPONENT_REF
&& TREE_CODE (TREE_TYPE (t)) == ARRAY_TYPE))
return false;
gcc_assert (OMP_CLAUSE_MAP_KIND (c) != GOMP_MAP_FORCE_DEVICEPTR);
if (ort == C_ORT_OMP || ort == C_ORT_ACC)
switch (OMP_CLAUSE_MAP_KIND (c))
{
case GOMP_MAP_ALLOC:
case GOMP_MAP_TO:
case GOMP_MAP_FROM:
case GOMP_MAP_TOFROM:
case GOMP_MAP_ALWAYS_TO:
case GOMP_MAP_ALWAYS_FROM:
case GOMP_MAP_ALWAYS_TOFROM:
case GOMP_MAP_RELEASE:
case GOMP_MAP_DELETE:
case GOMP_MAP_FORCE_TO:
case GOMP_MAP_FORCE_FROM:
case GOMP_MAP_FORCE_TOFROM:
case GOMP_MAP_FORCE_PRESENT:
OMP_CLAUSE_MAP_MAYBE_ZERO_LENGTH_ARRAY_SECTION (c) = 1;
break;
default:
break;
}
tree c2 = build_omp_clause (OMP_CLAUSE_LOCATION (c), OMP_CLAUSE_MAP);
if (ort != C_ORT_OMP && ort != C_ORT_ACC)
OMP_CLAUSE_SET_MAP_KIND (c2, GOMP_MAP_POINTER);
else if (TREE_CODE (t) == COMPONENT_REF)
OMP_CLAUSE_SET_MAP_KIND (c2, GOMP_MAP_ALWAYS_POINTER);
else
OMP_CLAUSE_SET_MAP_KIND (c2, GOMP_MAP_FIRSTPRIVATE_POINTER);
if (OMP_CLAUSE_MAP_KIND (c2) != GOMP_MAP_FIRSTPRIVATE_POINTER
&& !c_mark_addressable (t))
return false;
OMP_CLAUSE_DECL (c2) = t;
t = build_fold_addr_expr (first);
t = fold_convert_loc (OMP_CLAUSE_LOCATION (c), ptrdiff_type_node, t);
tree ptr = OMP_CLAUSE_DECL (c2);
if (!POINTER_TYPE_P (TREE_TYPE (ptr)))
ptr = build_fold_addr_expr (ptr);
t = fold_build2_loc (OMP_CLAUSE_LOCATION (c), MINUS_EXPR,
ptrdiff_type_node, t,
fold_convert_loc (OMP_CLAUSE_LOCATION (c),
ptrdiff_type_node, ptr));
t = c_fully_fold (t, false, NULL);
OMP_CLAUSE_SIZE (c2) = t;
OMP_CLAUSE_CHAIN (c2) = OMP_CLAUSE_CHAIN (c);
OMP_CLAUSE_CHAIN (c) = c2;
}
return false;
}
static tree
c_clone_omp_udr (tree stmt, tree omp_decl1, tree omp_decl2,
tree decl, tree placeholder)
{
copy_body_data id;
hash_map<tree, tree> decl_map;
decl_map.put (omp_decl1, placeholder);
decl_map.put (omp_decl2, decl);
memset (&id, 0, sizeof (id));
id.src_fn = DECL_CONTEXT (omp_decl1);
id.dst_fn = current_function_decl;
id.src_cfun = DECL_STRUCT_FUNCTION (id.src_fn);
id.decl_map = &decl_map;
id.copy_decl = copy_decl_no_change;
id.transform_call_graph_edges = CB_CGE_DUPLICATE;
id.transform_new_cfg = true;
id.transform_return_to_modify = false;
id.transform_lang_insert_block = NULL;
id.eh_lp_nr = 0;
walk_tree (&stmt, copy_tree_body_r, &id, NULL);
return stmt;
}
static tree
c_find_omp_placeholder_r (tree *tp, int *, void *data)
{
if (*tp == (tree) data)
return *tp;
return NULL_TREE;
}
tree
c_finish_omp_clauses (tree clauses, enum c_omp_region_type ort)
{
bitmap_head generic_head, firstprivate_head, lastprivate_head;
bitmap_head aligned_head, map_head, map_field_head, oacc_reduction_head;
tree c, t, type, *pc;
tree simdlen = NULL_TREE, safelen = NULL_TREE;
bool branch_seen = false;
bool copyprivate_seen = false;
bool linear_variable_step_check = false;
tree *nowait_clause = NULL;
bool ordered_seen = false;
tree schedule_clause = NULL_TREE;
bool oacc_async = false;
bitmap_obstack_initialize (NULL);
bitmap_initialize (&generic_head, &bitmap_default_obstack);
bitmap_initialize (&firstprivate_head, &bitmap_default_obstack);
bitmap_initialize (&lastprivate_head, &bitmap_default_obstack);
bitmap_initialize (&aligned_head, &bitmap_default_obstack);
bitmap_initialize (&map_head, &bitmap_default_obstack);
bitmap_initialize (&map_field_head, &bitmap_default_obstack);
bitmap_initialize (&oacc_reduction_head, &bitmap_default_obstack);
if (ort & C_ORT_ACC)
for (c = clauses; c; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_ASYNC)
{
oacc_async = true;
break;
}
for (pc = &clauses, c = clauses; c ; c = *pc)
{
bool remove = false;
bool need_complete = false;
bool need_implicitly_determined = false;
switch (OMP_CLAUSE_CODE (c))
{
case OMP_CLAUSE_SHARED:
need_implicitly_determined = true;
goto check_dup_generic;
case OMP_CLAUSE_PRIVATE:
need_complete = true;
need_implicitly_determined = true;
goto check_dup_generic;
case OMP_CLAUSE_REDUCTION:
need_implicitly_determined = true;
t = OMP_CLAUSE_DECL (c);
if (TREE_CODE (t) == TREE_LIST)
{
if (handle_omp_array_sections (c, ort))
{
remove = true;
break;
}
t = OMP_CLAUSE_DECL (c);
}
t = require_complete_type (OMP_CLAUSE_LOCATION (c), t);
if (t == error_mark_node)
{
remove = true;
break;
}
if (oacc_async)
c_mark_addressable (t);
type = TREE_TYPE (t);
if (TREE_CODE (t) == MEM_REF)
type = TREE_TYPE (type);
if (TREE_CODE (type) == ARRAY_TYPE)
{
tree oatype = type;
gcc_assert (TREE_CODE (t) != MEM_REF);
while (TREE_CODE (type) == ARRAY_TYPE)
type = TREE_TYPE (type);
if (integer_zerop (TYPE_SIZE_UNIT (type)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qD in %<reduction%> clause is a zero size array",
t);
remove = true;
break;
}
tree size = size_binop (EXACT_DIV_EXPR, TYPE_SIZE_UNIT (oatype),
TYPE_SIZE_UNIT (type));
if (integer_zerop (size))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qD in %<reduction%> clause is a zero size array",
t);
remove = true;
break;
}
size = size_binop (MINUS_EXPR, size, size_one_node);
tree index_type = build_index_type (size);
tree atype = build_array_type (type, index_type);
tree ptype = build_pointer_type (type);
if (TREE_CODE (TREE_TYPE (t)) == ARRAY_TYPE)
t = build_fold_addr_expr (t);
t = build2 (MEM_REF, atype, t, build_int_cst (ptype, 0));
OMP_CLAUSE_DECL (c) = t;
}
if (TYPE_ATOMIC (type))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<_Atomic%> %qE in %<reduction%> clause", t);
remove = true;
break;
}
if (OMP_CLAUSE_REDUCTION_PLACEHOLDER (c) == NULL_TREE
&& (FLOAT_TYPE_P (type)
|| TREE_CODE (type) == COMPLEX_TYPE))
{
enum tree_code r_code = OMP_CLAUSE_REDUCTION_CODE (c);
const char *r_name = NULL;
switch (r_code)
{
case PLUS_EXPR:
case MULT_EXPR:
case MINUS_EXPR:
break;
case MIN_EXPR:
if (TREE_CODE (type) == COMPLEX_TYPE)
r_name = "min";
break;
case MAX_EXPR:
if (TREE_CODE (type) == COMPLEX_TYPE)
r_name = "max";
break;
case BIT_AND_EXPR:
r_name = "&";
break;
case BIT_XOR_EXPR:
r_name = "^";
break;
case BIT_IOR_EXPR:
r_name = "|";
break;
case TRUTH_ANDIF_EXPR:
if (FLOAT_TYPE_P (type))
r_name = "&&";
break;
case TRUTH_ORIF_EXPR:
if (FLOAT_TYPE_P (type))
r_name = "||";
break;
default:
gcc_unreachable ();
}
if (r_name)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE has invalid type for %<reduction(%s)%>",
t, r_name);
remove = true;
break;
}
}
else if (OMP_CLAUSE_REDUCTION_PLACEHOLDER (c) == error_mark_node)
{
error_at (OMP_CLAUSE_LOCATION (c),
"user defined reduction not found for %qE", t);
remove = true;
break;
}
else if (OMP_CLAUSE_REDUCTION_PLACEHOLDER (c))
{
tree list = OMP_CLAUSE_REDUCTION_PLACEHOLDER (c);
type = TYPE_MAIN_VARIANT (type);
tree placeholder = build_decl (OMP_CLAUSE_LOCATION (c),
VAR_DECL, NULL_TREE, type);
tree decl_placeholder = NULL_TREE;
OMP_CLAUSE_REDUCTION_PLACEHOLDER (c) = placeholder;
DECL_ARTIFICIAL (placeholder) = 1;
DECL_IGNORED_P (placeholder) = 1;
if (TREE_CODE (t) == MEM_REF)
{
decl_placeholder = build_decl (OMP_CLAUSE_LOCATION (c),
VAR_DECL, NULL_TREE, type);
OMP_CLAUSE_REDUCTION_DECL_PLACEHOLDER (c) = decl_placeholder;
DECL_ARTIFICIAL (decl_placeholder) = 1;
DECL_IGNORED_P (decl_placeholder) = 1;
}
if (TREE_ADDRESSABLE (TREE_VEC_ELT (list, 0)))
c_mark_addressable (placeholder);
if (TREE_ADDRESSABLE (TREE_VEC_ELT (list, 1)))
c_mark_addressable (decl_placeholder ? decl_placeholder
: OMP_CLAUSE_DECL (c));
OMP_CLAUSE_REDUCTION_MERGE (c)
= c_clone_omp_udr (TREE_VEC_ELT (list, 2),
TREE_VEC_ELT (list, 0),
TREE_VEC_ELT (list, 1),
decl_placeholder ? decl_placeholder
: OMP_CLAUSE_DECL (c), placeholder);
OMP_CLAUSE_REDUCTION_MERGE (c)
= build3_loc (OMP_CLAUSE_LOCATION (c), BIND_EXPR,
void_type_node, NULL_TREE,
OMP_CLAUSE_REDUCTION_MERGE (c), NULL_TREE);
TREE_SIDE_EFFECTS (OMP_CLAUSE_REDUCTION_MERGE (c)) = 1;
if (TREE_VEC_LENGTH (list) == 6)
{
if (TREE_ADDRESSABLE (TREE_VEC_ELT (list, 3)))
c_mark_addressable (decl_placeholder ? decl_placeholder
: OMP_CLAUSE_DECL (c));
if (TREE_ADDRESSABLE (TREE_VEC_ELT (list, 4)))
c_mark_addressable (placeholder);
tree init = TREE_VEC_ELT (list, 5);
if (init == error_mark_node)
init = DECL_INITIAL (TREE_VEC_ELT (list, 3));
OMP_CLAUSE_REDUCTION_INIT (c)
= c_clone_omp_udr (init, TREE_VEC_ELT (list, 4),
TREE_VEC_ELT (list, 3),
decl_placeholder ? decl_placeholder
: OMP_CLAUSE_DECL (c), placeholder);
if (TREE_VEC_ELT (list, 5) == error_mark_node)
{
tree v = decl_placeholder ? decl_placeholder : t;
OMP_CLAUSE_REDUCTION_INIT (c)
= build2 (INIT_EXPR, TREE_TYPE (v), v,
OMP_CLAUSE_REDUCTION_INIT (c));
}
if (walk_tree (&OMP_CLAUSE_REDUCTION_INIT (c),
c_find_omp_placeholder_r,
placeholder, NULL))
OMP_CLAUSE_REDUCTION_OMP_ORIG_REF (c) = 1;
}
else
{
tree init;
tree v = decl_placeholder ? decl_placeholder : t;
if (AGGREGATE_TYPE_P (TREE_TYPE (v)))
init = build_constructor (TREE_TYPE (v), NULL);
else
init = fold_convert (TREE_TYPE (v), integer_zero_node);
OMP_CLAUSE_REDUCTION_INIT (c)
= build2 (INIT_EXPR, TREE_TYPE (v), v, init);
}
OMP_CLAUSE_REDUCTION_INIT (c)
= build3_loc (OMP_CLAUSE_LOCATION (c), BIND_EXPR,
void_type_node, NULL_TREE,
OMP_CLAUSE_REDUCTION_INIT (c), NULL_TREE);
TREE_SIDE_EFFECTS (OMP_CLAUSE_REDUCTION_INIT (c)) = 1;
}
if (TREE_CODE (t) == MEM_REF)
{
if (TYPE_SIZE_UNIT (TREE_TYPE (TREE_TYPE (t))) == NULL_TREE
|| TREE_CODE (TYPE_SIZE_UNIT (TREE_TYPE (TREE_TYPE (t))))
!= INTEGER_CST)
{
sorry ("variable length element type in array "
"%<reduction%> clause");
remove = true;
break;
}
t = TREE_OPERAND (t, 0);
if (TREE_CODE (t) == POINTER_PLUS_EXPR)
t = TREE_OPERAND (t, 0);
if (TREE_CODE (t) == ADDR_EXPR)
t = TREE_OPERAND (t, 0);
}
goto check_dup_generic_t;
case OMP_CLAUSE_COPYPRIVATE:
copyprivate_seen = true;
if (nowait_clause)
{
error_at (OMP_CLAUSE_LOCATION (*nowait_clause),
"%<nowait%> clause must not be used together "
"with %<copyprivate%>");
*nowait_clause = OMP_CLAUSE_CHAIN (*nowait_clause);
nowait_clause = NULL;
}
goto check_dup_generic;
case OMP_CLAUSE_COPYIN:
t = OMP_CLAUSE_DECL (c);
if (!VAR_P (t) || !DECL_THREAD_LOCAL_P (t))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE must be %<threadprivate%> for %<copyin%>", t);
remove = true;
break;
}
goto check_dup_generic;
case OMP_CLAUSE_LINEAR:
if (ort != C_ORT_OMP_DECLARE_SIMD)
need_implicitly_determined = true;
t = OMP_CLAUSE_DECL (c);
if (ort != C_ORT_OMP_DECLARE_SIMD
&& OMP_CLAUSE_LINEAR_KIND (c) != OMP_CLAUSE_LINEAR_DEFAULT)
{
error_at (OMP_CLAUSE_LOCATION (c),
"modifier should not be specified in %<linear%> "
"clause on %<simd%> or %<for%> constructs");
OMP_CLAUSE_LINEAR_KIND (c) = OMP_CLAUSE_LINEAR_DEFAULT;
}
if (!INTEGRAL_TYPE_P (TREE_TYPE (t))
&& TREE_CODE (TREE_TYPE (t)) != POINTER_TYPE)
{
error_at (OMP_CLAUSE_LOCATION (c),
"linear clause applied to non-integral non-pointer "
"variable with type %qT", TREE_TYPE (t));
remove = true;
break;
}
if (TYPE_ATOMIC (TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<_Atomic%> %qD in %<linear%> clause", t);
remove = true;
break;
}
if (ort == C_ORT_OMP_DECLARE_SIMD)
{
tree s = OMP_CLAUSE_LINEAR_STEP (c);
if (TREE_CODE (s) == PARM_DECL)
{
OMP_CLAUSE_LINEAR_VARIABLE_STRIDE (c) = 1;
if (!bitmap_bit_p (&map_head, DECL_UID (s)))
linear_variable_step_check = true;
goto check_dup_generic;
}
if (TREE_CODE (s) != INTEGER_CST)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<linear%> clause step %qE is neither constant "
"nor a parameter", s);
remove = true;
break;
}
}
if (TREE_CODE (TREE_TYPE (OMP_CLAUSE_DECL (c))) == POINTER_TYPE)
{
tree s = OMP_CLAUSE_LINEAR_STEP (c);
s = pointer_int_sum (OMP_CLAUSE_LOCATION (c), PLUS_EXPR,
OMP_CLAUSE_DECL (c), s);
s = fold_build2_loc (OMP_CLAUSE_LOCATION (c), MINUS_EXPR,
sizetype, fold_convert (sizetype, s),
fold_convert
(sizetype, OMP_CLAUSE_DECL (c)));
if (s == error_mark_node)
s = size_one_node;
OMP_CLAUSE_LINEAR_STEP (c) = s;
}
else
OMP_CLAUSE_LINEAR_STEP (c)
= fold_convert (TREE_TYPE (t), OMP_CLAUSE_LINEAR_STEP (c));
goto check_dup_generic;
check_dup_generic:
t = OMP_CLAUSE_DECL (c);
check_dup_generic_t:
if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is not a variable in clause %qs", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (ort == C_ORT_ACC
&& OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION)
{
if (bitmap_bit_p (&oacc_reduction_head, DECL_UID (t)))
{
error ("%qD appears more than once in reduction clauses", t);
remove = true;
}
else
bitmap_set_bit (&oacc_reduction_head, DECL_UID (t));
}
else if (bitmap_bit_p (&generic_head, DECL_UID (t))
|| bitmap_bit_p (&firstprivate_head, DECL_UID (t))
|| bitmap_bit_p (&lastprivate_head, DECL_UID (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE appears more than once in data clauses", t);
remove = true;
}
else if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_PRIVATE
&& bitmap_bit_p (&map_head, DECL_UID (t)))
{
if (ort == C_ORT_ACC)
error ("%qD appears more than once in data clauses", t);
else
error ("%qD appears both in data and map clauses", t);
remove = true;
}
else
bitmap_set_bit (&generic_head, DECL_UID (t));
break;
case OMP_CLAUSE_FIRSTPRIVATE:
t = OMP_CLAUSE_DECL (c);
need_complete = true;
need_implicitly_determined = true;
if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is not a variable in clause %<firstprivate%>", t);
remove = true;
}
else if (bitmap_bit_p (&generic_head, DECL_UID (t))
|| bitmap_bit_p (&firstprivate_head, DECL_UID (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE appears more than once in data clauses", t);
remove = true;
}
else if (bitmap_bit_p (&map_head, DECL_UID (t)))
{
if (ort == C_ORT_ACC)
error ("%qD appears more than once in data clauses", t);
else
error ("%qD appears both in data and map clauses", t);
remove = true;
}
else
bitmap_set_bit (&firstprivate_head, DECL_UID (t));
break;
case OMP_CLAUSE_LASTPRIVATE:
t = OMP_CLAUSE_DECL (c);
need_complete = true;
need_implicitly_determined = true;
if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is not a variable in clause %<lastprivate%>", t);
remove = true;
}
else if (bitmap_bit_p (&generic_head, DECL_UID (t))
|| bitmap_bit_p (&lastprivate_head, DECL_UID (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE appears more than once in data clauses", t);
remove = true;
}
else
bitmap_set_bit (&lastprivate_head, DECL_UID (t));
break;
case OMP_CLAUSE_ALIGNED:
t = OMP_CLAUSE_DECL (c);
if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is not a variable in %<aligned%> clause", t);
remove = true;
}
else if (!POINTER_TYPE_P (TREE_TYPE (t))
&& TREE_CODE (TREE_TYPE (t)) != ARRAY_TYPE)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE in %<aligned%> clause is neither a pointer nor "
"an array", t);
remove = true;
}
else if (TYPE_ATOMIC (TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<_Atomic%> %qD in %<aligned%> clause", t);
remove = true;
break;
}
else if (bitmap_bit_p (&aligned_head, DECL_UID (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE appears more than once in %<aligned%> clauses",
t);
remove = true;
}
else
bitmap_set_bit (&aligned_head, DECL_UID (t));
break;
case OMP_CLAUSE_DEPEND:
t = OMP_CLAUSE_DECL (c);
if (t == NULL_TREE)
{
gcc_assert (OMP_CLAUSE_DEPEND_KIND (c)
== OMP_CLAUSE_DEPEND_SOURCE);
break;
}
if (OMP_CLAUSE_DEPEND_KIND (c) == OMP_CLAUSE_DEPEND_SINK)
{
gcc_assert (TREE_CODE (t) == TREE_LIST);
for (; t; t = TREE_CHAIN (t))
{
tree decl = TREE_VALUE (t);
if (TREE_CODE (TREE_TYPE (decl)) == POINTER_TYPE)
{
tree offset = TREE_PURPOSE (t);
bool neg = wi::neg_p (wi::to_wide (offset));
offset = fold_unary (ABS_EXPR, TREE_TYPE (offset), offset);
tree t2 = pointer_int_sum (OMP_CLAUSE_LOCATION (c),
neg ? MINUS_EXPR : PLUS_EXPR,
decl, offset);
t2 = fold_build2_loc (OMP_CLAUSE_LOCATION (c), MINUS_EXPR,
sizetype,
fold_convert (sizetype, t2),
fold_convert (sizetype, decl));
if (t2 == error_mark_node)
{
remove = true;
break;
}
TREE_PURPOSE (t) = t2;
}
}
break;
}
if (TREE_CODE (t) == TREE_LIST)
{
if (handle_omp_array_sections (c, ort))
remove = true;
break;
}
if (t == error_mark_node)
remove = true;
else if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is not a variable in %<depend%> clause", t);
remove = true;
}
else if (!c_mark_addressable (t))
remove = true;
break;
case OMP_CLAUSE_MAP:
case OMP_CLAUSE_TO:
case OMP_CLAUSE_FROM:
case OMP_CLAUSE__CACHE_:
t = OMP_CLAUSE_DECL (c);
if (TREE_CODE (t) == TREE_LIST)
{
if (handle_omp_array_sections (c, ort))
remove = true;
else
{
t = OMP_CLAUSE_DECL (c);
if (!lang_hooks.types.omp_mappable_type (TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"array section does not have mappable type "
"in %qs clause",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (TYPE_ATOMIC (TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<_Atomic%> %qE in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
while (TREE_CODE (t) == ARRAY_REF)
t = TREE_OPERAND (t, 0);
if (TREE_CODE (t) == COMPONENT_REF
&& TREE_CODE (TREE_TYPE (t)) == ARRAY_TYPE)
{
while (TREE_CODE (t) == COMPONENT_REF)
t = TREE_OPERAND (t, 0);
if (bitmap_bit_p (&map_field_head, DECL_UID (t)))
break;
if (bitmap_bit_p (&map_head, DECL_UID (t)))
{
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_MAP)
error ("%qD appears more than once in motion"
" clauses", t);
else if (ort == C_ORT_ACC)
error ("%qD appears more than once in data"
" clauses", t);
else
error ("%qD appears more than once in map"
" clauses", t);
remove = true;
}
else
{
bitmap_set_bit (&map_head, DECL_UID (t));
bitmap_set_bit (&map_field_head, DECL_UID (t));
}
}
}
break;
}
if (t == error_mark_node)
{
remove = true;
break;
}
if (TREE_CODE (t) == COMPONENT_REF
&& (ort & C_ORT_OMP)
&& OMP_CLAUSE_CODE (c) != OMP_CLAUSE__CACHE_)
{
if (DECL_BIT_FIELD (TREE_OPERAND (t, 1)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"bit-field %qE in %qs clause",
t, omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (!lang_hooks.types.omp_mappable_type (TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE does not have a mappable type in %qs clause",
t, omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (TYPE_ATOMIC (TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<_Atomic%> %qE in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
while (TREE_CODE (t) == COMPONENT_REF)
{
if (TREE_CODE (TREE_TYPE (TREE_OPERAND (t, 0)))
== UNION_TYPE)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is a member of a union", t);
remove = true;
break;
}
t = TREE_OPERAND (t, 0);
}
if (remove)
break;
if (VAR_P (t) || TREE_CODE (t) == PARM_DECL)
{
if (bitmap_bit_p (&map_field_head, DECL_UID (t)))
break;
}
}
if (!VAR_P (t) && TREE_CODE (t) != PARM_DECL)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is not a variable in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (VAR_P (t) && DECL_THREAD_LOCAL_P (t))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qD is threadprivate variable in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if ((OMP_CLAUSE_CODE (c) != OMP_CLAUSE_MAP
|| (OMP_CLAUSE_MAP_KIND (c)
!= GOMP_MAP_FIRSTPRIVATE_POINTER))
&& !c_mark_addressable (t))
remove = true;
else if (!(OMP_CLAUSE_CODE (c) == OMP_CLAUSE_MAP
&& (OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_POINTER
|| (OMP_CLAUSE_MAP_KIND (c)
== GOMP_MAP_FIRSTPRIVATE_POINTER)
|| (OMP_CLAUSE_MAP_KIND (c)
== GOMP_MAP_FORCE_DEVICEPTR)))
&& t == OMP_CLAUSE_DECL (c)
&& !lang_hooks.types.omp_mappable_type (TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qD does not have a mappable type in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (TREE_TYPE (t) == error_mark_node)
remove = true;
else if (TYPE_ATOMIC (strip_array_types (TREE_TYPE (t))))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<_Atomic%> %qE in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_MAP
&& OMP_CLAUSE_MAP_KIND (c) == GOMP_MAP_FIRSTPRIVATE_POINTER)
{
if (bitmap_bit_p (&generic_head, DECL_UID (t))
|| bitmap_bit_p (&firstprivate_head, DECL_UID (t)))
{
error ("%qD appears more than once in data clauses", t);
remove = true;
}
else if (bitmap_bit_p (&map_head, DECL_UID (t)))
{
if (ort == C_ORT_ACC)
error ("%qD appears more than once in data clauses", t);
else
error ("%qD appears both in data and map clauses", t);
remove = true;
}
else
bitmap_set_bit (&generic_head, DECL_UID (t));
}
else if (bitmap_bit_p (&map_head, DECL_UID (t)))
{
if (OMP_CLAUSE_CODE (c) != OMP_CLAUSE_MAP)
error ("%qD appears more than once in motion clauses", t);
else if (ort == C_ORT_ACC)
error ("%qD appears more than once in data clauses", t);
else
error ("%qD appears more than once in map clauses", t);
remove = true;
}
else if (bitmap_bit_p (&generic_head, DECL_UID (t))
|| bitmap_bit_p (&firstprivate_head, DECL_UID (t)))
{
if (ort == C_ORT_ACC)
error ("%qD appears more than once in data clauses", t);
else
error ("%qD appears both in data and map clauses", t);
remove = true;
}
else
{
bitmap_set_bit (&map_head, DECL_UID (t));
if (t != OMP_CLAUSE_DECL (c)
&& TREE_CODE (OMP_CLAUSE_DECL (c)) == COMPONENT_REF)
bitmap_set_bit (&map_field_head, DECL_UID (t));
}
break;
case OMP_CLAUSE_TO_DECLARE:
case OMP_CLAUSE_LINK:
t = OMP_CLAUSE_DECL (c);
if (TREE_CODE (t) == FUNCTION_DECL
&& OMP_CLAUSE_CODE (c) == OMP_CLAUSE_TO_DECLARE)
;
else if (!VAR_P (t))
{
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_TO_DECLARE)
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is neither a variable nor a function name in "
"clause %qs", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
else
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is not a variable in clause %qs", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (DECL_THREAD_LOCAL_P (t))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qD is threadprivate variable in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
else if (!lang_hooks.types.omp_mappable_type (TREE_TYPE (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qD does not have a mappable type in %qs clause", t,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
if (remove)
break;
if (bitmap_bit_p (&generic_head, DECL_UID (t)))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE appears more than once on the same "
"%<declare target%> directive", t);
remove = true;
}
else
bitmap_set_bit (&generic_head, DECL_UID (t));
break;
case OMP_CLAUSE_UNIFORM:
t = OMP_CLAUSE_DECL (c);
if (TREE_CODE (t) != PARM_DECL)
{
if (DECL_P (t))
error_at (OMP_CLAUSE_LOCATION (c),
"%qD is not an argument in %<uniform%> clause", t);
else
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is not an argument in %<uniform%> clause", t);
remove = true;
break;
}
bitmap_set_bit (&map_head, DECL_UID (t));
goto check_dup_generic;
case OMP_CLAUSE_IS_DEVICE_PTR:
case OMP_CLAUSE_USE_DEVICE_PTR:
t = OMP_CLAUSE_DECL (c);
if (TREE_CODE (TREE_TYPE (t)) != POINTER_TYPE
&& TREE_CODE (TREE_TYPE (t)) != ARRAY_TYPE)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qs variable is neither a pointer nor an array",
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
goto check_dup_generic;
case OMP_CLAUSE_NOWAIT:
if (copyprivate_seen)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<nowait%> clause must not be used together "
"with %<copyprivate%>");
remove = true;
break;
}
nowait_clause = pc;
pc = &OMP_CLAUSE_CHAIN (c);
continue;
case OMP_CLAUSE_IF:
case OMP_CLAUSE_NUM_THREADS:
case OMP_CLAUSE_NUM_TEAMS:
case OMP_CLAUSE_THREAD_LIMIT:
case OMP_CLAUSE_DEFAULT:
case OMP_CLAUSE_UNTIED:
case OMP_CLAUSE_COLLAPSE:
case OMP_CLAUSE_FINAL:
case OMP_CLAUSE_MERGEABLE:
case OMP_CLAUSE_DEVICE:
case OMP_CLAUSE_DIST_SCHEDULE:
case OMP_CLAUSE_PARALLEL:
case OMP_CLAUSE_FOR:
case OMP_CLAUSE_SECTIONS:
case OMP_CLAUSE_TASKGROUP:
case OMP_CLAUSE_PROC_BIND:
case OMP_CLAUSE_PRIORITY:
case OMP_CLAUSE_GRAINSIZE:
case OMP_CLAUSE_NUM_TASKS:
case OMP_CLAUSE_NOGROUP:
case OMP_CLAUSE_THREADS:
case OMP_CLAUSE_SIMD:
case OMP_CLAUSE_HINT:
case OMP_CLAUSE_DEFAULTMAP:
case OMP_CLAUSE_NUM_GANGS:
case OMP_CLAUSE_NUM_WORKERS:
case OMP_CLAUSE_VECTOR_LENGTH:
case OMP_CLAUSE_ASYNC:
case OMP_CLAUSE_WAIT:
case OMP_CLAUSE_AUTO:
case OMP_CLAUSE_INDEPENDENT:
case OMP_CLAUSE_SEQ:
case OMP_CLAUSE_GANG:
case OMP_CLAUSE_WORKER:
case OMP_CLAUSE_VECTOR:
case OMP_CLAUSE_TILE:
pc = &OMP_CLAUSE_CHAIN (c);
continue;
case OMP_CLAUSE_SCHEDULE:
if (OMP_CLAUSE_SCHEDULE_KIND (c) & OMP_CLAUSE_SCHEDULE_NONMONOTONIC)
{
const char *p = NULL;
switch (OMP_CLAUSE_SCHEDULE_KIND (c) & OMP_CLAUSE_SCHEDULE_MASK)
{
case OMP_CLAUSE_SCHEDULE_STATIC: p = "static"; break;
case OMP_CLAUSE_SCHEDULE_DYNAMIC: break;
case OMP_CLAUSE_SCHEDULE_GUIDED: break;
case OMP_CLAUSE_SCHEDULE_AUTO: p = "auto"; break;
case OMP_CLAUSE_SCHEDULE_RUNTIME: p = "runtime"; break;
default: gcc_unreachable ();
}
if (p)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<nonmonotonic%> modifier specified for %qs "
"schedule kind", p);
OMP_CLAUSE_SCHEDULE_KIND (c)
= (enum omp_clause_schedule_kind)
(OMP_CLAUSE_SCHEDULE_KIND (c)
& ~OMP_CLAUSE_SCHEDULE_NONMONOTONIC);
}
}
schedule_clause = c;
pc = &OMP_CLAUSE_CHAIN (c);
continue;
case OMP_CLAUSE_ORDERED:
ordered_seen = true;
pc = &OMP_CLAUSE_CHAIN (c);
continue;
case OMP_CLAUSE_SAFELEN:
safelen = c;
pc = &OMP_CLAUSE_CHAIN (c);
continue;
case OMP_CLAUSE_SIMDLEN:
simdlen = c;
pc = &OMP_CLAUSE_CHAIN (c);
continue;
case OMP_CLAUSE_INBRANCH:
case OMP_CLAUSE_NOTINBRANCH:
if (branch_seen)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<inbranch%> clause is incompatible with "
"%<notinbranch%>");
remove = true;
break;
}
branch_seen = true;
pc = &OMP_CLAUSE_CHAIN (c);
continue;
default:
gcc_unreachable ();
}
if (!remove)
{
t = OMP_CLAUSE_DECL (c);
if (need_complete)
{
t = require_complete_type (OMP_CLAUSE_LOCATION (c), t);
if (t == error_mark_node)
remove = true;
}
if (need_implicitly_determined)
{
const char *share_name = NULL;
if (VAR_P (t) && DECL_THREAD_LOCAL_P (t))
share_name = "threadprivate";
else switch (c_omp_predetermined_sharing (t))
{
case OMP_CLAUSE_DEFAULT_UNSPECIFIED:
break;
case OMP_CLAUSE_DEFAULT_SHARED:
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_FIRSTPRIVATE
&& TREE_READONLY (t))
break;
share_name = "shared";
break;
case OMP_CLAUSE_DEFAULT_PRIVATE:
share_name = "private";
break;
default:
gcc_unreachable ();
}
if (share_name)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qE is predetermined %qs for %qs",
t, share_name,
omp_clause_code_name[OMP_CLAUSE_CODE (c)]);
remove = true;
}
}
}
if (remove)
*pc = OMP_CLAUSE_CHAIN (c);
else
pc = &OMP_CLAUSE_CHAIN (c);
}
if (simdlen
&& safelen
&& tree_int_cst_lt (OMP_CLAUSE_SAFELEN_EXPR (safelen),
OMP_CLAUSE_SIMDLEN_EXPR (simdlen)))
{
error_at (OMP_CLAUSE_LOCATION (simdlen),
"%<simdlen%> clause value is bigger than "
"%<safelen%> clause value");
OMP_CLAUSE_SIMDLEN_EXPR (simdlen)
= OMP_CLAUSE_SAFELEN_EXPR (safelen);
}
if (ordered_seen
&& schedule_clause
&& (OMP_CLAUSE_SCHEDULE_KIND (schedule_clause)
& OMP_CLAUSE_SCHEDULE_NONMONOTONIC))
{
error_at (OMP_CLAUSE_LOCATION (schedule_clause),
"%<nonmonotonic%> schedule modifier specified together "
"with %<ordered%> clause");
OMP_CLAUSE_SCHEDULE_KIND (schedule_clause)
= (enum omp_clause_schedule_kind)
(OMP_CLAUSE_SCHEDULE_KIND (schedule_clause)
& ~OMP_CLAUSE_SCHEDULE_NONMONOTONIC);
}
if (linear_variable_step_check)
for (pc = &clauses, c = clauses; c ; c = *pc)
{
bool remove = false;
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LINEAR
&& OMP_CLAUSE_LINEAR_VARIABLE_STRIDE (c)
&& !bitmap_bit_p (&map_head,
DECL_UID (OMP_CLAUSE_LINEAR_STEP (c))))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<linear%> clause step is a parameter %qD not "
"specified in %<uniform%> clause",
OMP_CLAUSE_LINEAR_STEP (c));
remove = true;
}
if (remove)
*pc = OMP_CLAUSE_CHAIN (c);
else
pc = &OMP_CLAUSE_CHAIN (c);
}
bitmap_obstack_release (NULL);
return clauses;
}
tree
c_omp_clause_copy_ctor (tree clause, tree dst, tree src)
{
if (!really_atomic_lvalue (dst) && !really_atomic_lvalue (src))
return build2 (MODIFY_EXPR, TREE_TYPE (dst), dst, src);
location_t loc = OMP_CLAUSE_LOCATION (clause);
tree type = TREE_TYPE (dst);
tree nonatomic_type = build_qualified_type (type, TYPE_UNQUALIFIED);
tree tmp = create_tmp_var (nonatomic_type);
tree tmp_addr = build_fold_addr_expr (tmp);
TREE_ADDRESSABLE (tmp) = 1;
TREE_NO_WARNING (tmp) = 1;
tree src_addr = build_fold_addr_expr (src);
tree dst_addr = build_fold_addr_expr (dst);
tree seq_cst = build_int_cst (integer_type_node, MEMMODEL_SEQ_CST);
vec<tree, va_gc> *params;
vec_alloc (params, 4);
tree fndecl = builtin_decl_explicit (BUILT_IN_ATOMIC_LOAD);
params->quick_push (src_addr);
params->quick_push (tmp_addr);
params->quick_push (seq_cst);
tree load = c_build_function_call_vec (loc, vNULL, fndecl, params, NULL);
vec_alloc (params, 4);
fndecl = builtin_decl_explicit (BUILT_IN_ATOMIC_STORE);
params->quick_push (dst_addr);
params->quick_push (tmp_addr);
params->quick_push (seq_cst);
tree store = c_build_function_call_vec (loc, vNULL, fndecl, params, NULL);
return build2 (COMPOUND_EXPR, void_type_node, load, store);
}
tree
c_finish_transaction (location_t loc, tree block, int flags)
{
tree stmt = build_stmt (loc, TRANSACTION_EXPR, block);
if (flags & TM_STMT_ATTR_OUTER)
TRANSACTION_EXPR_OUTER (stmt) = 1;
if (flags & TM_STMT_ATTR_RELAXED)
TRANSACTION_EXPR_RELAXED (stmt) = 1;
return add_stmt (stmt);
}
tree
c_build_qualified_type (tree type, int type_quals, tree orig_qual_type,
size_t orig_qual_indirect)
{
if (type == error_mark_node)
return type;
if (TREE_CODE (type) == ARRAY_TYPE)
{
tree t;
tree element_type = c_build_qualified_type (TREE_TYPE (type),
type_quals, orig_qual_type,
orig_qual_indirect - 1);
if (orig_qual_type && orig_qual_indirect == 0)
t = orig_qual_type;
else
for (t = TYPE_MAIN_VARIANT (type); t; t = TYPE_NEXT_VARIANT (t))
{
if (TYPE_QUALS (strip_array_types (t)) == type_quals
&& TYPE_NAME (t) == TYPE_NAME (type)
&& TYPE_CONTEXT (t) == TYPE_CONTEXT (type)
&& attribute_list_equal (TYPE_ATTRIBUTES (t),
TYPE_ATTRIBUTES (type)))
break;
}
if (!t)
{
tree domain = TYPE_DOMAIN (type);
t = build_variant_type_copy (type);
TREE_TYPE (t) = element_type;
if (TYPE_STRUCTURAL_EQUALITY_P (element_type)
|| (domain && TYPE_STRUCTURAL_EQUALITY_P (domain)))
SET_TYPE_STRUCTURAL_EQUALITY (t);
else if (TYPE_CANONICAL (element_type) != element_type
|| (domain && TYPE_CANONICAL (domain) != domain))
{
tree unqualified_canon
= build_array_type (TYPE_CANONICAL (element_type),
domain? TYPE_CANONICAL (domain)
: NULL_TREE);
if (TYPE_REVERSE_STORAGE_ORDER (type))
{
unqualified_canon
= build_distinct_type_copy (unqualified_canon);
TYPE_REVERSE_STORAGE_ORDER (unqualified_canon) = 1;
}
TYPE_CANONICAL (t)
= c_build_qualified_type (unqualified_canon, type_quals);
}
else
TYPE_CANONICAL (t) = t;
}
return t;
}
if ((type_quals & TYPE_QUAL_RESTRICT)
&& (!POINTER_TYPE_P (type)
|| !C_TYPE_OBJECT_OR_INCOMPLETE_P (TREE_TYPE (type))))
{
error ("invalid use of %<restrict%>");
type_quals &= ~TYPE_QUAL_RESTRICT;
}
tree var_type = (orig_qual_type && orig_qual_indirect == 0
? orig_qual_type
: build_qualified_type (type, type_quals));
if (RECORD_OR_UNION_TYPE_P (var_type)
&& TYPE_MAIN_VARIANT (var_type) != var_type)
C_TYPE_INCOMPLETE_VARS (var_type) = 0;
return var_type;
}
tree
c_build_va_arg (location_t loc1, tree expr, location_t loc2, tree type)
{
if (error_operand_p (type))
return error_mark_node;
else if (handled_component_p (expr)
&& reverse_storage_order_for_component_p (expr))
{
error_at (loc1, "cannot use %<va_arg%> with reverse storage order");
return error_mark_node;
}
else if (!COMPLETE_TYPE_P (type))
{
error_at (loc2, "second argument to %<va_arg%> is of incomplete "
"type %qT", type);
return error_mark_node;
}
else if (warn_cxx_compat && TREE_CODE (type) == ENUMERAL_TYPE)
warning_at (loc2, OPT_Wc___compat,
"C++ requires promoted type, not enum type, in %<va_arg%>");
return build_va_arg (loc2, expr, type);
}
bool
c_tree_equal (tree t1, tree t2)
{
enum tree_code code1, code2;
if (t1 == t2)
return true;
if (!t1 || !t2)
return false;
for (code1 = TREE_CODE (t1);
CONVERT_EXPR_CODE_P (code1)
|| code1 == NON_LVALUE_EXPR;
code1 = TREE_CODE (t1))
t1 = TREE_OPERAND (t1, 0);
for (code2 = TREE_CODE (t2);
CONVERT_EXPR_CODE_P (code2)
|| code2 == NON_LVALUE_EXPR;
code2 = TREE_CODE (t2))
t2 = TREE_OPERAND (t2, 0);
if (t1 == t2)
return true;
if (code1 != code2)
return false;
switch (code1)
{
case INTEGER_CST:
return wi::to_wide (t1) == wi::to_wide (t2);
case REAL_CST:
return real_equal (&TREE_REAL_CST (t1), &TREE_REAL_CST (t2));
case STRING_CST:
return TREE_STRING_LENGTH (t1) == TREE_STRING_LENGTH (t2)
&& !memcmp (TREE_STRING_POINTER (t1), TREE_STRING_POINTER (t2),
TREE_STRING_LENGTH (t1));
case FIXED_CST:
return FIXED_VALUES_IDENTICAL (TREE_FIXED_CST (t1),
TREE_FIXED_CST (t2));
case COMPLEX_CST:
return c_tree_equal (TREE_REALPART (t1), TREE_REALPART (t2))
&& c_tree_equal (TREE_IMAGPART (t1), TREE_IMAGPART (t2));
case VECTOR_CST:
return operand_equal_p (t1, t2, OEP_ONLY_CONST);
case CONSTRUCTOR:
if (!comptypes (TREE_TYPE (t1), TREE_TYPE (t2))
|| CONSTRUCTOR_NELTS (t1) != CONSTRUCTOR_NELTS (t2))
return false;
{
tree field, value;
unsigned int i;
FOR_EACH_CONSTRUCTOR_ELT (CONSTRUCTOR_ELTS (t1), i, field, value)
{
constructor_elt *elt2 = CONSTRUCTOR_ELT (t2, i);
if (!c_tree_equal (field, elt2->index)
|| !c_tree_equal (value, elt2->value))
return false;
}
}
return true;
case TREE_LIST:
if (!c_tree_equal (TREE_PURPOSE (t1), TREE_PURPOSE (t2)))
return false;
if (!c_tree_equal (TREE_VALUE (t1), TREE_VALUE (t2)))
return false;
return c_tree_equal (TREE_CHAIN (t1), TREE_CHAIN (t2));
case SAVE_EXPR:
return c_tree_equal (TREE_OPERAND (t1, 0), TREE_OPERAND (t2, 0));
case CALL_EXPR:
{
tree arg1, arg2;
call_expr_arg_iterator iter1, iter2;
if (!c_tree_equal (CALL_EXPR_FN (t1), CALL_EXPR_FN (t2)))
return false;
for (arg1 = first_call_expr_arg (t1, &iter1),
arg2 = first_call_expr_arg (t2, &iter2);
arg1 && arg2;
arg1 = next_call_expr_arg (&iter1),
arg2 = next_call_expr_arg (&iter2))
if (!c_tree_equal (arg1, arg2))
return false;
if (arg1 || arg2)
return false;
return true;
}
case TARGET_EXPR:
{
tree o1 = TREE_OPERAND (t1, 0);
tree o2 = TREE_OPERAND (t2, 0);
if (VAR_P (o1) && DECL_NAME (o1) == NULL_TREE
&& !DECL_RTL_SET_P (o1))
;
else if (VAR_P (o2) && DECL_NAME (o2) == NULL_TREE
&& !DECL_RTL_SET_P (o2))
;
else if (!c_tree_equal (o1, o2))
return false;
return c_tree_equal (TREE_OPERAND (t1, 1), TREE_OPERAND (t2, 1));
}
case COMPONENT_REF:
if (TREE_OPERAND (t1, 1) != TREE_OPERAND (t2, 1))
return false;
return c_tree_equal (TREE_OPERAND (t1, 0), TREE_OPERAND (t2, 0));
case PARM_DECL:
case VAR_DECL:
case CONST_DECL:
case FIELD_DECL:
case FUNCTION_DECL:
case IDENTIFIER_NODE:
case SSA_NAME:
return false;
case TREE_VEC:
{
unsigned ix;
if (TREE_VEC_LENGTH (t1) != TREE_VEC_LENGTH (t2))
return false;
for (ix = TREE_VEC_LENGTH (t1); ix--;)
if (!c_tree_equal (TREE_VEC_ELT (t1, ix),
TREE_VEC_ELT (t2, ix)))
return false;
return true;
}
default:
break;
}
switch (TREE_CODE_CLASS (code1))
{
case tcc_unary:
case tcc_binary:
case tcc_comparison:
case tcc_expression:
case tcc_vl_exp:
case tcc_reference:
case tcc_statement:
{
int i, n = TREE_OPERAND_LENGTH (t1);
switch (code1)
{
case PREINCREMENT_EXPR:
case PREDECREMENT_EXPR:
case POSTINCREMENT_EXPR:
case POSTDECREMENT_EXPR:
n = 1;
break;
case ARRAY_REF:
n = 2;
break;
default:
break;
}
if (TREE_CODE_CLASS (code1) == tcc_vl_exp
&& n != TREE_OPERAND_LENGTH (t2))
return false;
for (i = 0; i < n; ++i)
if (!c_tree_equal (TREE_OPERAND (t1, i), TREE_OPERAND (t2, i)))
return false;
return true;
}
case tcc_type:
return comptypes (t1, t2);
default:
gcc_unreachable ();
}
return false;
}
bool
c_decl_implicit (const_tree fndecl)
{
return C_DECL_IMPLICIT (fndecl);
}
