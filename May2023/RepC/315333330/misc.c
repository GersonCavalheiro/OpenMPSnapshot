#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "tree.h"
#include "diagnostic.h"
#include "opts.h"
#include "alias.h"
#include "fold-const.h"
#include "stor-layout.h"
#include "print-tree.h"
#include "toplev.h"
#include "langhooks.h"
#include "langhooks-def.h"
#include "plugin.h"
#include "calls.h"	
#include "dwarf2out.h"
#include "ada.h"
#include "adadecode.h"
#include "types.h"
#include "atree.h"
#include "namet.h"
#include "nlists.h"
#include "uintp.h"
#include "fe.h"
#include "sinfo.h"
#include "einfo.h"
#include "ada-tree.h"
#include "gigi.h"
void *callgraph_info_file = NULL;
unsigned int save_argc;
const char **save_argv;
extern int gnat_argc;
extern const char **gnat_argv;
#undef gnat_encodings
enum dwarf_gnat_encodings gnat_encodings = DWARF_GNAT_ENCODINGS_DEFAULT;
#undef optimize
int optimize;
#undef optimize_size
int optimize_size;
#undef flag_compare_debug
int flag_compare_debug;
#undef flag_short_enums
int flag_short_enums;
#undef flag_stack_check
enum stack_check_type flag_stack_check = NO_STACK_CHECK;
#ifdef __cplusplus
extern "C" {
#endif
extern void __gnat_initialize (void *);
extern void __gnat_install_SEH_handler (void *);
extern void adainit (void);
extern void _ada_gnat1drv (void);
#ifdef __cplusplus
}
#endif
static void
gnat_parse_file (void)
{
int seh[2];
__gnat_initialize (NULL);
__gnat_install_SEH_handler ((void *)seh);
adainit ();
_ada_gnat1drv ();
gnat_write_global_declarations ();
}
static unsigned int
gnat_option_lang_mask (void)
{
return CL_Ada;
}
static bool
gnat_handle_option (size_t scode, const char *arg, int value, int kind,
location_t loc, const struct cl_option_handlers *handlers)
{
enum opt_code code = (enum opt_code) scode;
switch (code)
{
case OPT_Wall:
handle_generated_option (&global_options, &global_options_set,
OPT_Wunused, NULL, value,
gnat_option_lang_mask (), kind, loc,
handlers, true, global_dc);
warn_uninitialized = value;
warn_maybe_uninitialized = value;
break;
case OPT_gant:
warning (0, "%<-gnat%> misspelled as %<-gant%>");
case OPT_gnat:
case OPT_gnatO:
case OPT_fRTS_:
case OPT_I:
case OPT_nostdinc:
case OPT_nostdlib:
break;
case OPT_fshort_enums:
case OPT_fsigned_char:
break;
case OPT_fbuiltin_printf:
break;
default:
gcc_unreachable ();
}
Ada_handle_option_auto (&global_options, &global_options_set,
scode, arg, value,
gnat_option_lang_mask (), kind, loc,
handlers, global_dc);
return true;
}
static void
gnat_init_options_struct (struct gcc_options *opts)
{
opts->x_flag_zero_initialized_in_bss = 0;
opts->x_flag_errno_math = 0;
opts->frontend_set_flag_errno_math = true;
}
static void
gnat_init_options (unsigned int decoded_options_count,
struct cl_decoded_option *decoded_options)
{
save_argv = XNEWVEC (const char *, 2 * decoded_options_count + 1);
save_argc = 0;
for (unsigned int i = 0; i < decoded_options_count; i++)
{
size_t num_elements = decoded_options[i].canonical_option_num_elements;
if (decoded_options[i].errors
|| decoded_options[i].opt_index == OPT_SPECIAL_unknown
|| num_elements == 0)
continue;
if (decoded_options[i].opt_index == OPT_I
&& num_elements == 2
&& decoded_options[i].canonical_option[1][0] == '-'
&& decoded_options[i].canonical_option[1][1] == '\0')
save_argv[save_argc++] = "-I-";
else
{
gcc_assert (num_elements >= 1 && num_elements <= 2);
save_argv[save_argc++] = decoded_options[i].canonical_option[0];
if (num_elements >= 2)
save_argv[save_argc++] = decoded_options[i].canonical_option[1];
}
}
save_argv[save_argc] = NULL;
gnat_argv = (const char **) xmalloc (sizeof (char *));
gnat_argv[0] = xstrdup (save_argv[0]);
gnat_argc = 1;
}
static bool
gnat_post_options (const char **pfilename ATTRIBUTE_UNUSED)
{
if (flag_excess_precision_cmdline == EXCESS_PRECISION_STANDARD)
sorry ("-fexcess-precision=standard for Ada");
flag_excess_precision_cmdline = EXCESS_PRECISION_FAST;
warn_psabi = 0;
warn_return_type = 0;
warn_stringop_overflow = 0;
if (!global_options_set.x_flag_diagnostics_show_caret)
global_dc->show_caret = false;
if (PREFERRED_DEBUGGING_TYPE != DBX_DEBUG && write_symbols == DBX_DEBUG)
warning (0, "STABS debugging information for Ada is obsolete and not "
"supported anymore");
gnat_encodings = global_options.x_gnat_encodings;
optimize = global_options.x_optimize;
optimize_size = global_options.x_optimize_size;
flag_compare_debug = global_options.x_flag_compare_debug;
flag_stack_check = global_options.x_flag_stack_check;
flag_short_enums = global_options.x_flag_short_enums;
if (flag_short_enums == 2)
flag_short_enums = targetm.default_short_enums ();
return false;
}
static void
internal_error_function (diagnostic_context *context, const char *msgid,
va_list *ap)
{
text_info tinfo;
char *buffer, *p, *loc;
String_Template temp, temp_loc;
String_Pointer sp, sp_loc;
expanded_location xloc;
warn_if_plugins ();
pp_clear_output_area (context->printer);
tinfo.format_spec = msgid;
tinfo.args_ptr = ap;
tinfo.err_no = errno;
pp_format_verbatim (context->printer, &tinfo);
buffer = xstrdup (pp_formatted_text (context->printer));
for (p = buffer; *p; p++)
if (*p == '\n')
{
*p = '\0';
break;
}
temp.Low_Bound = 1;
temp.High_Bound = p - buffer;
sp.Bounds = &temp;
sp.Array = buffer;
xloc = expand_location (input_location);
if (context->show_column && xloc.column != 0)
loc = xasprintf ("%s:%d:%d", xloc.file, xloc.line, xloc.column);
else
loc = xasprintf ("%s:%d", xloc.file, xloc.line);
temp_loc.Low_Bound = 1;
temp_loc.High_Bound = strlen (loc);
sp_loc.Bounds = &temp_loc;
sp_loc.Array = loc;
Current_Error_Node = error_gnat_node;
Compiler_Abort (sp, sp_loc, true);
}
static bool
gnat_init (void)
{
build_common_tree_nodes (flag_signed_char);
boolean_type_node = make_unsigned_type (8);
TREE_SET_CODE (boolean_type_node, BOOLEAN_TYPE);
SET_TYPE_RM_MAX_VALUE (boolean_type_node,
build_int_cst (boolean_type_node, 1));
SET_TYPE_RM_SIZE (boolean_type_node, bitsize_int (1));
boolean_true_node = TYPE_MAX_VALUE (boolean_type_node);
boolean_false_node = TYPE_MIN_VALUE (boolean_type_node);
sbitsize_one_node = sbitsize_int (1);
sbitsize_unit_node = sbitsize_int (BITS_PER_UNIT);
global_dc->internal_error = &internal_error_function;
return true;
}
void
gnat_init_gcc_eh (void)
{
if (No_Exception_Handlers_Set ())
return;
using_eh_for_cleanups ();
flag_exceptions = 1;
flag_delete_dead_exceptions = 1;
if (Suppress_Checks)
{
if (!global_options_set.x_flag_non_call_exceptions)
flag_non_call_exceptions = Machine_Overflows_On_Target && GNAT_Mode;
}
else
{
flag_non_call_exceptions = 1;
flag_aggressive_loop_optimizations = 0;
warn_aggressive_loop_optimizations = 0;
}
init_eh ();
}
void
gnat_init_gcc_fp (void)
{
if (Signed_Zeros_On_Target)
flag_signed_zeros = 1;
else if (!global_options_set.x_flag_signed_zeros)
flag_signed_zeros = 0;
if (Machine_Overflows_On_Target)
flag_trapping_math = 1;
else if (!global_options_set.x_flag_trapping_math)
flag_trapping_math = 0;
}
static void
gnat_print_decl (FILE *file, tree node, int indent)
{
switch (TREE_CODE (node))
{
case CONST_DECL:
print_node (file, "corresponding var",
DECL_CONST_CORRESPONDING_VAR (node), indent + 4);
break;
case FIELD_DECL:
print_node (file, "original field", DECL_ORIGINAL_FIELD (node),
indent + 4);
break;
case VAR_DECL:
if (DECL_LOOP_PARM_P (node))
print_node (file, "induction var", DECL_INDUCTION_VAR (node),
indent + 4);
else
print_node (file, "renamed object", DECL_RENAMED_OBJECT (node),
indent + 4);
break;
default:
break;
}
}
static void
gnat_print_type (FILE *file, tree node, int indent)
{
switch (TREE_CODE (node))
{
case FUNCTION_TYPE:
print_node (file, "ci/co list", TYPE_CI_CO_LIST (node), indent + 4);
break;
case INTEGER_TYPE:
if (TYPE_MODULAR_P (node))
print_node_brief (file, "modulus", TYPE_MODULUS (node), indent + 4);
else if (TYPE_FIXED_POINT_P (node))
print_node (file, "scale factor", TYPE_SCALE_FACTOR (node),
indent + 4);
else if (TYPE_HAS_ACTUAL_BOUNDS_P (node))
print_node (file, "actual bounds", TYPE_ACTUAL_BOUNDS (node),
indent + 4);
else
print_node (file, "index type", TYPE_INDEX_TYPE (node), indent + 4);
case ENUMERAL_TYPE:
case BOOLEAN_TYPE:
print_node_brief (file, "RM size", TYPE_RM_SIZE (node), indent + 4);
case REAL_TYPE:
print_node_brief (file, "RM min", TYPE_RM_MIN_VALUE (node), indent + 4);
print_node_brief (file, "RM max", TYPE_RM_MAX_VALUE (node), indent + 4);
break;
case ARRAY_TYPE:
print_node (file,"actual bounds", TYPE_ACTUAL_BOUNDS (node), indent + 4);
break;
case VECTOR_TYPE:
print_node (file,"representative array",
TYPE_REPRESENTATIVE_ARRAY (node), indent + 4);
break;
case RECORD_TYPE:
if (TYPE_FAT_POINTER_P (node) || TYPE_CONTAINS_TEMPLATE_P (node))
print_node (file, "unconstrained array",
TYPE_UNCONSTRAINED_ARRAY (node), indent + 4);
else
print_node (file, "Ada size", TYPE_ADA_SIZE (node), indent + 4);
break;
case UNION_TYPE:
case QUAL_UNION_TYPE:
print_node (file, "Ada size", TYPE_ADA_SIZE (node), indent + 4);
break;
default:
break;
}
if (TYPE_CAN_HAVE_DEBUG_TYPE_P (node) && TYPE_DEBUG_TYPE (node))
print_node_brief (file, "debug type", TYPE_DEBUG_TYPE (node), indent + 4);
if (TYPE_IMPL_PACKED_ARRAY_P (node) && TYPE_ORIGINAL_PACKED_ARRAY (node))
print_node_brief (file, "original packed array",
TYPE_ORIGINAL_PACKED_ARRAY (node), indent + 4);
}
static const char *
gnat_printable_name (tree decl, int verbosity)
{
const char *coded_name = IDENTIFIER_POINTER (DECL_NAME (decl));
char *ada_name = (char *) ggc_alloc_atomic (strlen (coded_name) * 2 + 60);
__gnat_decode (coded_name, ada_name, 0);
if (verbosity == 2 && !DECL_IS_BUILTIN (decl))
{
Set_Identifier_Casing (ada_name, DECL_SOURCE_FILE (decl));
return ggc_strdup (Name_Buffer);
}
return ada_name;
}
static const char *
gnat_dwarf_name (tree decl, int verbosity ATTRIBUTE_UNUSED)
{
gcc_assert (DECL_P (decl));
return (const char *) IDENTIFIER_POINTER (DECL_NAME (decl));
}
static tree
gnat_descriptive_type (const_tree type)
{
if (TYPE_STUB_DECL (type))
return DECL_PARALLEL_TYPE (TYPE_STUB_DECL (type));
else
return NULL_TREE;
}
static tree
gnat_enum_underlying_base_type (const_tree)
{
return void_type_node;
}
static tree
gnat_get_debug_type (const_tree type)
{
if (TYPE_CAN_HAVE_DEBUG_TYPE_P (type) && TYPE_DEBUG_TYPE (type))
{
type = TYPE_DEBUG_TYPE (type);
if (type && TYPE_CAN_HAVE_DEBUG_TYPE_P (type))
return const_cast<tree> (type);
}
return NULL_TREE;
}
static bool
gnat_get_fixed_point_type_info (const_tree type,
struct fixed_point_type_info *info)
{
tree scale_factor;
if (!TYPE_IS_FIXED_POINT_P (type)
|| gnat_encodings != DWARF_GNAT_ENCODINGS_MINIMAL)
return false;
scale_factor = TYPE_SCALE_FACTOR (type);
if (scale_factor == integer_zero_node)
{
info->scale_factor_kind = fixed_point_scale_factor_arbitrary;
info->scale_factor.arbitrary.numerator = 0;
info->scale_factor.arbitrary.denominator = 0;
return true;
}
if (TREE_CODE (scale_factor) == RDIV_EXPR)
{
const tree num = TREE_OPERAND (scale_factor, 0);
const tree den = TREE_OPERAND (scale_factor, 1);
if (TREE_CODE (den) == POWER_EXPR)
{
const tree base = TREE_OPERAND (den, 0);
const tree exponent = TREE_OPERAND (den, 1);
gcc_assert (num == integer_one_node
&& TREE_CODE (base) == INTEGER_CST
&& TREE_CODE (exponent) == INTEGER_CST);
switch (tree_to_shwi (base))
{
case 2:
info->scale_factor_kind = fixed_point_scale_factor_binary;
info->scale_factor.binary = -tree_to_shwi (exponent);
return true;
case 10:
info->scale_factor_kind = fixed_point_scale_factor_decimal;
info->scale_factor.decimal = -tree_to_shwi (exponent);
return true;
default:
gcc_unreachable ();
}
}
gcc_assert (TREE_CODE (num) == INTEGER_CST
&& TREE_CODE (den) == INTEGER_CST);
info->scale_factor_kind = fixed_point_scale_factor_arbitrary;
info->scale_factor.arbitrary.numerator = tree_to_uhwi (num);
info->scale_factor.arbitrary.denominator = tree_to_shwi (den);
return true;
}
gcc_unreachable ();
}
static bool
gnat_type_hash_eq (const_tree t1, const_tree t2)
{
gcc_assert (TREE_CODE (t1) == FUNCTION_TYPE);
return fntype_same_flags_p (t1, TYPE_CI_CO_LIST (t2),
TYPE_RETURN_UNCONSTRAINED_P (t2),
TYPE_RETURN_BY_DIRECT_REF_P (t2),
TREE_ADDRESSABLE (t2));
}
static tree
gnat_return_tree (tree t)
{
return t;
}
static alias_set_type
gnat_get_alias_set (tree type)
{
if (TYPE_IS_PADDING_P (type))
return get_alias_set (TREE_TYPE (TYPE_FIELDS (type)));
else if (TREE_CODE (type) == UNCONSTRAINED_ARRAY_TYPE)
return
get_alias_set (TREE_TYPE (TREE_TYPE (TYPE_FIELDS (TREE_TYPE (type)))));
else if (TYPE_P (type)
&& !TYPE_IS_DUMMY_P (type)
&& TYPE_UNIVERSAL_ALIASING_P (type))
return 0;
return -1;
}
static tree
gnat_type_max_size (const_tree gnu_type)
{
tree max_unitsize = max_size (TYPE_SIZE_UNIT (gnu_type), true);
if (!tree_fits_uhwi_p (max_unitsize))
{
if (RECORD_OR_UNION_TYPE_P (gnu_type)
&& !TYPE_FAT_POINTER_P (gnu_type)
&& TYPE_ADA_SIZE (gnu_type))
{
tree max_adasize = max_size (TYPE_ADA_SIZE (gnu_type), true);
if (tree_fits_uhwi_p (max_adasize))
max_unitsize
= size_binop (CEIL_DIV_EXPR,
round_up (max_adasize, TYPE_ALIGN (gnu_type)),
bitsize_unit_node);
}
else if (TREE_CODE (gnu_type) == ARRAY_TYPE
&& TYPE_INDEX_TYPE (TYPE_DOMAIN (gnu_type))
&& tree_fits_uhwi_p (TYPE_SIZE_UNIT (TREE_TYPE (gnu_type))))
{
tree lb = TYPE_MIN_VALUE (TYPE_INDEX_TYPE (TYPE_DOMAIN (gnu_type)));
tree hb = TYPE_MAX_VALUE (TYPE_INDEX_TYPE (TYPE_DOMAIN (gnu_type)));
if (TREE_CODE (lb) != INTEGER_CST
&& TYPE_RM_SIZE (TREE_TYPE (lb))
&& compare_tree_int (TYPE_RM_SIZE (TREE_TYPE (lb)), 16) <= 0)
lb = TYPE_MIN_VALUE (TREE_TYPE (lb));
if (TREE_CODE (hb) != INTEGER_CST
&& TYPE_RM_SIZE (TREE_TYPE (hb))
&& compare_tree_int (TYPE_RM_SIZE (TREE_TYPE (hb)), 16) <= 0)
hb = TYPE_MAX_VALUE (TREE_TYPE (hb));
if (TREE_CODE (lb) == INTEGER_CST && TREE_CODE (hb) == INTEGER_CST)
{
tree ctype = get_base_type (TREE_TYPE (lb));
lb = fold_convert (ctype, lb);
hb = fold_convert (ctype, hb);
if (tree_int_cst_le (lb, hb))
{
tree length
= fold_build2 (PLUS_EXPR, ctype,
fold_build2 (MINUS_EXPR, ctype, hb, lb),
build_int_cst (ctype, 1));
max_unitsize
= fold_build2 (MULT_EXPR, sizetype,
fold_convert (sizetype, length),
TYPE_SIZE_UNIT (TREE_TYPE (gnu_type)));
}
}
}
}
return max_unitsize;
}
static tree get_array_bit_stride (tree);
static bool
gnat_get_array_descr_info (const_tree const_type,
struct array_descr_info *info)
{
bool convention_fortran_p;
bool is_array = false;
bool is_fat_ptr = false;
bool is_packed_array = false;
tree type = const_cast<tree> (const_type);
const_tree first_dimen = NULL_TREE;
const_tree last_dimen = NULL_TREE;
const_tree dimen;
int i;
tree thinptr_template_expr = NULL_TREE;
tree thinptr_bound_field = NULL_TREE;
type = maybe_debug_type (type);
if (TYPE_IMPL_PACKED_ARRAY_P (type) && TYPE_ORIGINAL_PACKED_ARRAY (type))
{
type = TYPE_ORIGINAL_PACKED_ARRAY (type);
is_packed_array = true;
}
if (TREE_CODE (type) == ARRAY_TYPE
&& TYPE_DOMAIN (type)
&& TYPE_INDEX_TYPE (TYPE_DOMAIN (type)))
{
is_array = true;
first_dimen = type;
info->data_location = NULL_TREE;
}
else if (TYPE_IS_FAT_POINTER_P (type)
&& gnat_encodings == DWARF_GNAT_ENCODINGS_MINIMAL)
{
const tree ua_type = TYPE_UNCONSTRAINED_ARRAY (type);
const tree placeholder_expr = build0 (PLACEHOLDER_EXPR, type);
const tree ua_val
= maybe_unconstrained_array (build_unary_op (INDIRECT_REF,
ua_type,
placeholder_expr));
is_fat_ptr = true;
first_dimen = TREE_TYPE (ua_val);
info->data_location = TREE_OPERAND (ua_val, 0);
}
else if (TREE_CODE (type) == RECORD_TYPE
&& TYPE_CONTAINS_TEMPLATE_P (type)
&& gnat_encodings == DWARF_GNAT_ENCODINGS_MINIMAL)
{
const tree placeholder_expr = build0 (PLACEHOLDER_EXPR, type);
const tree placeholder_addr
= build_unary_op (ADDR_EXPR, NULL_TREE, placeholder_expr);
const tree bounds_field = TYPE_FIELDS (type);
const tree bounds_type = TREE_TYPE (bounds_field);
const tree array_field = DECL_CHAIN (bounds_field);
const tree array_type = TREE_TYPE (array_field);
const tree shift_amount
= fold_build1 (NEGATE_EXPR, sizetype, byte_position (array_field));
tree template_addr
= build_binary_op (POINTER_PLUS_EXPR, TREE_TYPE (placeholder_addr),
placeholder_addr, shift_amount);
template_addr
= fold_convert (TYPE_POINTER_TO (bounds_type), template_addr);
first_dimen = array_type;
info->data_location = NULL_TREE;
thinptr_template_expr = build_unary_op (INDIRECT_REF,
bounds_type,
template_addr);
thinptr_bound_field = TYPE_FIELDS (bounds_type);
}
else
return false;
if (TYPE_PACKED (first_dimen))
is_packed_array = true;
convention_fortran_p = TYPE_CONVENTION_FORTRAN_P (first_dimen);
info->ordering = (convention_fortran_p
? array_descr_ordering_column_major
: array_descr_ordering_row_major);
for (i = 0, dimen = first_dimen; ; ++i, dimen = TREE_TYPE (dimen))
{
if (i > 0
&& (TREE_CODE (dimen) != ARRAY_TYPE
|| !TYPE_MULTI_ARRAY_P (dimen)))
break;
last_dimen = dimen;
}
info->ndimensions = i;
info->rank = NULL_TREE;
if (info->ndimensions > DWARF2OUT_ARRAY_DESCR_INFO_MAX_DIMEN
|| TYPE_MULTI_ARRAY_P (first_dimen))
{
info->ndimensions = 1;
last_dimen = first_dimen;
}
info->element_type = TREE_TYPE (last_dimen);
for (i = (convention_fortran_p ? info->ndimensions - 1 : 0),
dimen = first_dimen;
IN_RANGE (i, 0, info->ndimensions - 1);
i += (convention_fortran_p ? -1 : 1),
dimen = TREE_TYPE (dimen))
{
tree index_type = TYPE_INDEX_TYPE (TYPE_DOMAIN (dimen));
if (is_array || is_fat_ptr)
{
if (TYPE_CONTEXT (first_dimen)
&& TREE_CODE (TYPE_CONTEXT (first_dimen)) != RECORD_TYPE
&& CONTAINS_PLACEHOLDER_P (TYPE_MIN_VALUE (index_type))
&& gnat_encodings != DWARF_GNAT_ENCODINGS_MINIMAL)
{
info->dimen[i].lower_bound = NULL_TREE;
info->dimen[i].upper_bound = NULL_TREE;
}
else
{
info->dimen[i].lower_bound
= maybe_character_value (TYPE_MIN_VALUE (index_type));
info->dimen[i].upper_bound
= maybe_character_value (TYPE_MAX_VALUE (index_type));
}
}
else
{
info->dimen[i].lower_bound
= build_component_ref (thinptr_template_expr, thinptr_bound_field,
false);
thinptr_bound_field = DECL_CHAIN (thinptr_bound_field);
info->dimen[i].upper_bound
= build_component_ref (thinptr_template_expr, thinptr_bound_field,
false);
thinptr_bound_field = DECL_CHAIN (thinptr_bound_field);
}
while (TREE_TYPE (index_type))
index_type = TREE_TYPE (index_type);
info->dimen[i].bounds_type = maybe_debug_type (index_type);
info->dimen[i].stride = NULL_TREE;
}
info->allocated = NULL_TREE;
info->associated = NULL_TREE;
if (gnat_encodings == DWARF_GNAT_ENCODINGS_MINIMAL)
{
tree source_element_type = info->element_type;
while (true)
{
if (TYPE_DEBUG_TYPE (source_element_type))
source_element_type = TYPE_DEBUG_TYPE (source_element_type);
else if (TYPE_IS_PADDING_P (source_element_type))
source_element_type
= TREE_TYPE (TYPE_FIELDS (source_element_type));
else
break;
}
if (TREE_CODE (TYPE_SIZE_UNIT (source_element_type)) != INTEGER_CST)
{
info->stride = TYPE_SIZE_UNIT (info->element_type);
info->stride_in_bits = false;
}
else if (is_packed_array)
{
info->stride = get_array_bit_stride (info->element_type);
info->stride_in_bits = true;
}
}
return true;
}
static tree
get_array_bit_stride (tree comp_type)
{
struct array_descr_info info;
tree stride;
if (INTEGRAL_TYPE_P (comp_type))
return TYPE_RM_SIZE (comp_type);
memset (&info, 0, sizeof (info));
if (!gnat_get_array_descr_info (comp_type, &info) || !info.stride)
return NULL_TREE;
stride = info.stride;
if (!info.stride_in_bits)
{
stride = fold_convert (bitsizetype, stride);
stride = build_binary_op (MULT_EXPR, bitsizetype,
stride, build_int_cst (bitsizetype, 8));
}
for (int i = 0; i < info.ndimensions; ++i)
{
tree count;
if (!info.dimen[i].lower_bound || !info.dimen[i].upper_bound)
return NULL_TREE;
count = build_binary_op (MINUS_EXPR, sbitsizetype,
fold_convert (sbitsizetype,
info.dimen[i].upper_bound),
fold_convert (sbitsizetype,
info.dimen[i].lower_bound)),
count = build_binary_op (PLUS_EXPR, sbitsizetype,
count, build_int_cst (sbitsizetype, 1));
count = build_binary_op (MAX_EXPR, sbitsizetype,
count,
build_int_cst (sbitsizetype, 0));
count = fold_convert (bitsizetype, count);
stride = build_binary_op (MULT_EXPR, bitsizetype, stride, count);
}
return stride;
}
static void
gnat_get_subrange_bounds (const_tree gnu_type, tree *lowval, tree *highval)
{
*lowval = TYPE_MIN_VALUE (gnu_type);
*highval = TYPE_MAX_VALUE (gnu_type);
}
static tree
gnat_get_type_bias (const_tree gnu_type)
{
if (TREE_CODE (gnu_type) == INTEGER_TYPE
&& TYPE_BIASED_REPRESENTATION_P (gnu_type)
&& gnat_encodings == DWARF_GNAT_ENCODINGS_MINIMAL)
return TYPE_RM_MIN_VALUE (gnu_type);
return NULL_TREE;
}
bool
default_pass_by_ref (tree gnu_type)
{
if (AGGREGATE_TYPE_P (gnu_type)
&& (!valid_constant_size_p (TYPE_SIZE_UNIT (gnu_type))
|| compare_tree_int (TYPE_SIZE_UNIT (gnu_type),
TYPE_ALIGN (gnu_type)) > 0))
return true;
if (pass_by_reference (NULL, TYPE_MODE (gnu_type), gnu_type, true))
return true;
if (targetm.calls.return_in_memory (gnu_type, NULL_TREE))
return true;
return false;
}
bool
must_pass_by_ref (tree gnu_type)
{
return (TREE_CODE (gnu_type) == UNCONSTRAINED_ARRAY_TYPE
|| TYPE_IS_BY_REFERENCE_P (gnu_type)
|| (TYPE_SIZE_UNIT (gnu_type)
&& TREE_CODE (TYPE_SIZE_UNIT (gnu_type)) != INTEGER_CST));
}
void
enumerate_modes (void (*f) (const char *, int, int, int, int, int, int, int))
{
const tree c_types[]
= { float_type_node, double_type_node, long_double_type_node };
const char *const c_names[]
= { "float", "double", "long double" };
int iloop;
fp_arith_may_widen = false;
for (iloop = 0; iloop < NUM_MACHINE_MODES; iloop++)
{
machine_mode i = (machine_mode) iloop;
machine_mode inner_mode = i;
bool float_p = false;
bool complex_p = false;
bool vector_p = false;
bool skip_p = false;
int digs = 0;
unsigned int nameloop;
Float_Rep_Kind float_rep = IEEE_Binary; 
switch (GET_MODE_CLASS (i))
{
case MODE_INT:
break;
case MODE_FLOAT:
float_p = true;
break;
case MODE_COMPLEX_INT:
complex_p = true;
inner_mode = GET_MODE_INNER (i);
break;
case MODE_COMPLEX_FLOAT:
float_p = true;
complex_p = true;
inner_mode = GET_MODE_INNER (i);
break;
case MODE_VECTOR_INT:
vector_p = true;
inner_mode = GET_MODE_INNER (i);
break;
case MODE_VECTOR_FLOAT:
float_p = true;
vector_p = true;
inner_mode = GET_MODE_INNER (i);
break;
default:
skip_p = true;
}
if (float_p)
{
const struct real_format *fmt = REAL_MODE_FORMAT (inner_mode);
if (!fmt)
continue;
if (fmt == &ieee_extended_motorola_format
|| fmt == &ieee_extended_intel_96_format
|| fmt == &ieee_extended_intel_96_round_53_format
|| fmt == &ieee_extended_intel_128_format)
{
#ifdef TARGET_FPMATH_DEFAULT
if (TARGET_FPMATH_DEFAULT == FPMATH_387)
#endif
fp_arith_may_widen = true;
}
if (fmt->b == 2)
digs = (fmt->p - 1) * 1233 / 4096; 
else if (fmt->b == 10)
digs = fmt->p;
else
gcc_unreachable ();
}
if (!skip_p && !vector_p)
for (nameloop = 0; nameloop < ARRAY_SIZE (c_types); nameloop++)
{
tree type = c_types[nameloop];
const char *name = c_names[nameloop];
if (TYPE_MODE (type) == i)
{
f (name, digs, complex_p, 0, float_rep, TYPE_PRECISION (type),
TREE_INT_CST_LOW (TYPE_SIZE (type)), TYPE_ALIGN (type));
skip_p = true;
}
}
int nunits, precision, bitsize;
if (!skip_p
&& GET_MODE_NUNITS (i).is_constant (&nunits)
&& GET_MODE_PRECISION (i).is_constant (&precision)
&& GET_MODE_BITSIZE (i).is_constant (&bitsize))
f (GET_MODE_NAME (i), digs, complex_p,
vector_p ? nunits : 0, float_rep,
precision, bitsize, GET_MODE_ALIGNMENT (i));
}
}
int
fp_prec_to_size (int prec)
{
opt_scalar_float_mode opt_mode;
FOR_EACH_MODE_IN_CLASS (opt_mode, MODE_FLOAT)
{
scalar_float_mode mode = opt_mode.require ();
if (GET_MODE_PRECISION (mode) == prec)
return GET_MODE_BITSIZE (mode);
}
gcc_unreachable ();
}
int
fp_size_to_prec (int size)
{
opt_scalar_float_mode opt_mode;
FOR_EACH_MODE_IN_CLASS (opt_mode, MODE_FLOAT)
{
scalar_mode mode = opt_mode.require ();
if (GET_MODE_BITSIZE (mode) == size)
return GET_MODE_PRECISION (mode);
}
gcc_unreachable ();
}
static GTY(()) tree gnat_eh_personality_decl;
static tree
gnat_eh_personality (void)
{
if (!gnat_eh_personality_decl)
gnat_eh_personality_decl = build_personality_function ("gnat");
return gnat_eh_personality_decl;
}
static void
gnat_init_ts (void)
{
MARK_TS_COMMON (UNCONSTRAINED_ARRAY_TYPE);
MARK_TS_TYPED (UNCONSTRAINED_ARRAY_REF);
MARK_TS_TYPED (NULL_EXPR);
MARK_TS_TYPED (PLUS_NOMOD_EXPR);
MARK_TS_TYPED (MINUS_NOMOD_EXPR);
MARK_TS_TYPED (POWER_EXPR);
MARK_TS_TYPED (ATTR_ADDR_EXPR);
MARK_TS_TYPED (STMT_STMT);
MARK_TS_TYPED (LOOP_STMT);
MARK_TS_TYPED (EXIT_STMT);
}
static size_t
gnat_tree_size (enum tree_code code)
{
gcc_checking_assert (code >= NUM_TREE_CODES);
switch (code)
{
case UNCONSTRAINED_ARRAY_TYPE:
return sizeof (tree_type_non_common);
default:
gcc_unreachable ();
}
}
struct lang_type *
get_lang_specific (tree node)
{
if (!TYPE_LANG_SPECIFIC (node))
TYPE_LANG_SPECIFIC (node) = ggc_cleared_alloc<struct lang_type> ();
return TYPE_LANG_SPECIFIC (node);
}
#undef  LANG_HOOKS_NAME
#define LANG_HOOKS_NAME			"GNU Ada"
#undef  LANG_HOOKS_IDENTIFIER_SIZE
#define LANG_HOOKS_IDENTIFIER_SIZE	sizeof (struct tree_identifier)
#undef  LANG_HOOKS_TREE_SIZE
#define LANG_HOOKS_TREE_SIZE		gnat_tree_size
#undef  LANG_HOOKS_INIT
#define LANG_HOOKS_INIT			gnat_init
#undef  LANG_HOOKS_OPTION_LANG_MASK
#define LANG_HOOKS_OPTION_LANG_MASK	gnat_option_lang_mask
#undef  LANG_HOOKS_INIT_OPTIONS_STRUCT
#define LANG_HOOKS_INIT_OPTIONS_STRUCT	gnat_init_options_struct
#undef  LANG_HOOKS_INIT_OPTIONS
#define LANG_HOOKS_INIT_OPTIONS		gnat_init_options
#undef  LANG_HOOKS_HANDLE_OPTION
#define LANG_HOOKS_HANDLE_OPTION	gnat_handle_option
#undef  LANG_HOOKS_POST_OPTIONS
#define LANG_HOOKS_POST_OPTIONS		gnat_post_options
#undef  LANG_HOOKS_PARSE_FILE
#define LANG_HOOKS_PARSE_FILE		gnat_parse_file
#undef  LANG_HOOKS_TYPE_HASH_EQ
#define LANG_HOOKS_TYPE_HASH_EQ		gnat_type_hash_eq
#undef  LANG_HOOKS_GETDECLS
#define LANG_HOOKS_GETDECLS		hook_tree_void_null
#undef  LANG_HOOKS_PUSHDECL
#define LANG_HOOKS_PUSHDECL		gnat_return_tree
#undef  LANG_HOOKS_WARN_UNUSED_GLOBAL_DECL
#define LANG_HOOKS_WARN_UNUSED_GLOBAL_DECL hook_bool_const_tree_false
#undef  LANG_HOOKS_GET_ALIAS_SET
#define LANG_HOOKS_GET_ALIAS_SET	gnat_get_alias_set
#undef  LANG_HOOKS_PRINT_DECL
#define LANG_HOOKS_PRINT_DECL		gnat_print_decl
#undef  LANG_HOOKS_PRINT_TYPE
#define LANG_HOOKS_PRINT_TYPE		gnat_print_type
#undef  LANG_HOOKS_TYPE_MAX_SIZE
#define LANG_HOOKS_TYPE_MAX_SIZE	gnat_type_max_size
#undef  LANG_HOOKS_DECL_PRINTABLE_NAME
#define LANG_HOOKS_DECL_PRINTABLE_NAME	gnat_printable_name
#undef  LANG_HOOKS_DWARF_NAME
#define LANG_HOOKS_DWARF_NAME		gnat_dwarf_name
#undef  LANG_HOOKS_GIMPLIFY_EXPR
#define LANG_HOOKS_GIMPLIFY_EXPR	gnat_gimplify_expr
#undef  LANG_HOOKS_TYPE_FOR_MODE
#define LANG_HOOKS_TYPE_FOR_MODE	gnat_type_for_mode
#undef  LANG_HOOKS_TYPE_FOR_SIZE
#define LANG_HOOKS_TYPE_FOR_SIZE	gnat_type_for_size
#undef  LANG_HOOKS_TYPES_COMPATIBLE_P
#define LANG_HOOKS_TYPES_COMPATIBLE_P	gnat_types_compatible_p
#undef  LANG_HOOKS_GET_ARRAY_DESCR_INFO
#define LANG_HOOKS_GET_ARRAY_DESCR_INFO	gnat_get_array_descr_info
#undef  LANG_HOOKS_GET_SUBRANGE_BOUNDS
#define LANG_HOOKS_GET_SUBRANGE_BOUNDS  gnat_get_subrange_bounds
#undef  LANG_HOOKS_GET_TYPE_BIAS
#define LANG_HOOKS_GET_TYPE_BIAS	gnat_get_type_bias
#undef  LANG_HOOKS_DESCRIPTIVE_TYPE
#define LANG_HOOKS_DESCRIPTIVE_TYPE	gnat_descriptive_type
#undef  LANG_HOOKS_ENUM_UNDERLYING_BASE_TYPE
#define LANG_HOOKS_ENUM_UNDERLYING_BASE_TYPE gnat_enum_underlying_base_type
#undef  LANG_HOOKS_GET_DEBUG_TYPE
#define LANG_HOOKS_GET_DEBUG_TYPE	gnat_get_debug_type
#undef  LANG_HOOKS_GET_FIXED_POINT_TYPE_INFO
#define LANG_HOOKS_GET_FIXED_POINT_TYPE_INFO gnat_get_fixed_point_type_info
#undef  LANG_HOOKS_ATTRIBUTE_TABLE
#define LANG_HOOKS_ATTRIBUTE_TABLE	gnat_internal_attribute_table
#undef  LANG_HOOKS_BUILTIN_FUNCTION
#define LANG_HOOKS_BUILTIN_FUNCTION	gnat_builtin_function
#undef  LANG_HOOKS_INIT_TS
#define LANG_HOOKS_INIT_TS		gnat_init_ts
#undef  LANG_HOOKS_EH_PERSONALITY
#define LANG_HOOKS_EH_PERSONALITY	gnat_eh_personality
#undef  LANG_HOOKS_DEEP_UNSHARING
#define LANG_HOOKS_DEEP_UNSHARING	true
#undef  LANG_HOOKS_CUSTOM_FUNCTION_DESCRIPTORS
#define LANG_HOOKS_CUSTOM_FUNCTION_DESCRIPTORS true
struct lang_hooks lang_hooks = LANG_HOOKS_INITIALIZER;
#include "gt-ada-misc.h"
