#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "function.h"
#include "rtl.h"
#include "tree.h"
#include "memmodel.h"
#include "tm_p.h"
#include "stringpool.h"
#include "regs.h"
#include "emit-rtl.h"
#include "cgraph.h"
#include "diagnostic-core.h"
#include "fold-const.h"
#include "stor-layout.h"
#include "varasm.h"
#include "print-tree.h"
#include "langhooks.h"
#include "tree-inline.h"
#include "dumpfile.h"
#include "gimplify.h"
#include "attribs.h"
#include "debug.h"
tree sizetype_tab[(int) stk_type_kind_last];
unsigned int maximum_field_alignment = TARGET_DEFAULT_PACK_STRUCT * BITS_PER_UNIT;
static tree self_referential_size (tree);
static void finalize_record_size (record_layout_info);
static void finalize_type_size (tree);
static void place_union_field (record_layout_info, tree);
static int excess_unit_span (HOST_WIDE_INT, HOST_WIDE_INT, HOST_WIDE_INT,
HOST_WIDE_INT, tree);
extern void debug_rli (record_layout_info);

tree
variable_size (tree size)
{
if (TREE_CONSTANT (size))
return size;
if (CONTAINS_PLACEHOLDER_P (size))
return self_referential_size (size);
if (lang_hooks.decls.global_bindings_p ())
return size;
return save_expr (size);
}
static GTY(()) vec<tree, va_gc> *size_functions;
static bool
self_referential_component_ref_p (tree t)
{
if (TREE_CODE (t) != COMPONENT_REF)
return false;
while (REFERENCE_CLASS_P (t))
t = TREE_OPERAND (t, 0);
return (TREE_CODE (t) == PLACEHOLDER_EXPR);
}
static tree
copy_self_referential_tree_r (tree *tp, int *walk_subtrees, void *data)
{
enum tree_code code = TREE_CODE (*tp);
if (TREE_CODE_CLASS (code) == tcc_type
|| TREE_CODE_CLASS (code) == tcc_declaration
|| TREE_CODE_CLASS (code) == tcc_constant)
{
*walk_subtrees = 0;
return NULL_TREE;
}
else if (code == ADDR_EXPR
&& TREE_CODE (TREE_OPERAND (*tp, 0)) == PLACEHOLDER_EXPR)
{
*walk_subtrees = 0;
return NULL_TREE;
}
else if (self_referential_component_ref_p (*tp))
{
*walk_subtrees = 0;
return NULL_TREE;
}
else if (code == SAVE_EXPR)
return error_mark_node;
else if (code == STATEMENT_LIST)
gcc_unreachable ();
return copy_tree_r (tp, walk_subtrees, data);
}
static tree
self_referential_size (tree size)
{
static unsigned HOST_WIDE_INT fnno = 0;
vec<tree> self_refs = vNULL;
tree param_type_list = NULL, param_decl_list = NULL;
tree t, ref, return_type, fntype, fnname, fndecl;
unsigned int i;
char buf[128];
vec<tree, va_gc> *args = NULL;
t = skip_simple_constant_arithmetic (size);
if (TREE_CODE (t) == CALL_EXPR || self_referential_component_ref_p (t))
return size;
find_placeholder_in_expr (size, &self_refs);
gcc_assert (self_refs.length () > 0);
t = size;
if (walk_tree (&t, copy_self_referential_tree_r, NULL, NULL) != NULL_TREE)
return size;
size = t;
vec_alloc (args, self_refs.length ());
FOR_EACH_VEC_ELT (self_refs, i, ref)
{
tree subst, param_name, param_type, param_decl;
if (DECL_P (ref))
{
gcc_assert (TREE_READONLY (ref));
subst = ref;
}
else if (TREE_CODE (ref) == ADDR_EXPR)
subst = ref;
else
subst = TREE_OPERAND (ref, 1);
sprintf (buf, "p%d", i);
param_name = get_identifier (buf);
param_type = TREE_TYPE (ref);
param_decl
= build_decl (input_location, PARM_DECL, param_name, param_type);
DECL_ARG_TYPE (param_decl) = param_type;
DECL_ARTIFICIAL (param_decl) = 1;
TREE_READONLY (param_decl) = 1;
size = substitute_in_expr (size, subst, param_decl);
param_type_list = tree_cons (NULL_TREE, param_type, param_type_list);
param_decl_list = chainon (param_decl, param_decl_list);
args->quick_push (ref);
}
self_refs.release ();
param_type_list = tree_cons (NULL_TREE, void_type_node, param_type_list);
param_type_list = nreverse (param_type_list);
param_decl_list = nreverse (param_decl_list);
return_type = TREE_TYPE (size);
fntype = build_function_type (return_type, param_type_list);
sprintf (buf, "SZ" HOST_WIDE_INT_PRINT_UNSIGNED, fnno++);
fnname = get_file_function_name (buf);
fndecl = build_decl (input_location, FUNCTION_DECL, fnname, fntype);
for (t = param_decl_list; t; t = DECL_CHAIN (t))
DECL_CONTEXT (t) = fndecl;
DECL_ARGUMENTS (fndecl) = param_decl_list;
DECL_RESULT (fndecl)
= build_decl (input_location, RESULT_DECL, 0, return_type);
DECL_CONTEXT (DECL_RESULT (fndecl)) = fndecl;
DECL_ARTIFICIAL (fndecl) = 1;
DECL_IGNORED_P (fndecl) = 1;
TREE_READONLY (fndecl) = 1;
TREE_NOTHROW (fndecl) = 1;
DECL_DECLARED_INLINE_P (fndecl) = 1;
DECL_INITIAL (fndecl) = make_node (BLOCK);
BLOCK_SUPERCONTEXT (DECL_INITIAL (fndecl)) = fndecl;
t = build2 (MODIFY_EXPR, return_type, DECL_RESULT (fndecl), size);
DECL_SAVED_TREE (fndecl) = build1 (RETURN_EXPR, void_type_node, t);
TREE_STATIC (fndecl) = 1;
vec_safe_push (size_functions, fndecl);
return build_call_expr_loc_vec (UNKNOWN_LOCATION, fndecl, args);
}
void
finalize_size_functions (void)
{
unsigned int i;
tree fndecl;
for (i = 0; size_functions && size_functions->iterate (i, &fndecl); i++)
{
allocate_struct_function (fndecl, false);
set_cfun (NULL);
dump_function (TDI_original, fndecl);
debug_hooks->size_function (fndecl);
gimplify_function_tree (fndecl);
cgraph_node::finalize_function (fndecl, false);
}
vec_free (size_functions);
}

opt_machine_mode
mode_for_size (poly_uint64 size, enum mode_class mclass, int limit)
{
machine_mode mode;
int i;
if (limit && maybe_gt (size, (unsigned int) MAX_FIXED_MODE_SIZE))
return opt_machine_mode ();
FOR_EACH_MODE_IN_CLASS (mode, mclass)
if (known_eq (GET_MODE_PRECISION (mode), size))
return mode;
if (mclass == MODE_INT || mclass == MODE_PARTIAL_INT)
for (i = 0; i < NUM_INT_N_ENTS; i ++)
if (known_eq (int_n_data[i].bitsize, size)
&& int_n_enabled_p[i])
return int_n_data[i].m;
return opt_machine_mode ();
}
opt_machine_mode
mode_for_size_tree (const_tree size, enum mode_class mclass, int limit)
{
unsigned HOST_WIDE_INT uhwi;
unsigned int ui;
if (!tree_fits_uhwi_p (size))
return opt_machine_mode ();
uhwi = tree_to_uhwi (size);
ui = uhwi;
if (uhwi != ui)
return opt_machine_mode ();
return mode_for_size (ui, mclass, limit);
}
machine_mode
smallest_mode_for_size (poly_uint64 size, enum mode_class mclass)
{
machine_mode mode = VOIDmode;
int i;
FOR_EACH_MODE_IN_CLASS (mode, mclass)
if (known_ge (GET_MODE_PRECISION (mode), size))
break;
gcc_assert (mode != VOIDmode);
if (mclass == MODE_INT || mclass == MODE_PARTIAL_INT)
for (i = 0; i < NUM_INT_N_ENTS; i ++)
if (known_ge (int_n_data[i].bitsize, size)
&& known_lt (int_n_data[i].bitsize, GET_MODE_PRECISION (mode))
&& int_n_enabled_p[i])
mode = int_n_data[i].m;
return mode;
}
opt_scalar_int_mode
int_mode_for_mode (machine_mode mode)
{
switch (GET_MODE_CLASS (mode))
{
case MODE_INT:
case MODE_PARTIAL_INT:
return as_a <scalar_int_mode> (mode);
case MODE_COMPLEX_INT:
case MODE_COMPLEX_FLOAT:
case MODE_FLOAT:
case MODE_DECIMAL_FLOAT:
case MODE_FRACT:
case MODE_ACCUM:
case MODE_UFRACT:
case MODE_UACCUM:
case MODE_VECTOR_BOOL:
case MODE_VECTOR_INT:
case MODE_VECTOR_FLOAT:
case MODE_VECTOR_FRACT:
case MODE_VECTOR_ACCUM:
case MODE_VECTOR_UFRACT:
case MODE_VECTOR_UACCUM:
case MODE_POINTER_BOUNDS:
return int_mode_for_size (GET_MODE_BITSIZE (mode), 0);
case MODE_RANDOM:
if (mode == BLKmode)
return opt_scalar_int_mode ();
case MODE_CC:
default:
gcc_unreachable ();
}
}
opt_machine_mode
bitwise_mode_for_mode (machine_mode mode)
{
scalar_int_mode int_mode;
if (is_a <scalar_int_mode> (mode, &int_mode)
&& GET_MODE_BITSIZE (int_mode) <= MAX_FIXED_MODE_SIZE)
return int_mode;
gcc_checking_assert ((int_mode_for_mode (mode), true));
poly_int64 bitsize = GET_MODE_BITSIZE (mode);
if (COMPLEX_MODE_P (mode))
{
machine_mode trial = mode;
if ((GET_MODE_CLASS (trial) == MODE_COMPLEX_INT
|| mode_for_size (bitsize, MODE_COMPLEX_INT, false).exists (&trial))
&& have_regs_of_mode[GET_MODE_INNER (trial)])
return trial;
}
if (VECTOR_MODE_P (mode)
|| maybe_gt (bitsize, MAX_FIXED_MODE_SIZE))
{
machine_mode trial = mode;
if ((GET_MODE_CLASS (trial) == MODE_VECTOR_INT
|| mode_for_size (bitsize, MODE_VECTOR_INT, 0).exists (&trial))
&& have_regs_of_mode[trial]
&& targetm.vector_mode_supported_p (trial))
return trial;
}
return mode_for_size (bitsize, MODE_INT, true);
}
tree
bitwise_type_for_mode (machine_mode mode)
{
if (!bitwise_mode_for_mode (mode).exists (&mode))
return NULL_TREE;
unsigned int inner_size = GET_MODE_UNIT_BITSIZE (mode);
tree inner_type = build_nonstandard_integer_type (inner_size, true);
if (VECTOR_MODE_P (mode))
return build_vector_type_for_mode (inner_type, mode);
if (COMPLEX_MODE_P (mode))
return build_complex_type (inner_type);
gcc_checking_assert (GET_MODE_INNER (mode) == mode);
return inner_type;
}
opt_machine_mode
mode_for_vector (scalar_mode innermode, poly_uint64 nunits)
{
machine_mode mode;
if (SCALAR_FLOAT_MODE_P (innermode))
mode = MIN_MODE_VECTOR_FLOAT;
else if (SCALAR_FRACT_MODE_P (innermode))
mode = MIN_MODE_VECTOR_FRACT;
else if (SCALAR_UFRACT_MODE_P (innermode))
mode = MIN_MODE_VECTOR_UFRACT;
else if (SCALAR_ACCUM_MODE_P (innermode))
mode = MIN_MODE_VECTOR_ACCUM;
else if (SCALAR_UACCUM_MODE_P (innermode))
mode = MIN_MODE_VECTOR_UACCUM;
else
mode = MIN_MODE_VECTOR_INT;
FOR_EACH_MODE_FROM (mode, mode)
if (known_eq (GET_MODE_NUNITS (mode), nunits)
&& GET_MODE_INNER (mode) == innermode)
return mode;
if (GET_MODE_CLASS (innermode) == MODE_INT)
{
poly_uint64 nbits = nunits * GET_MODE_BITSIZE (innermode);
if (int_mode_for_size (nbits, 0).exists (&mode)
&& have_regs_of_mode[mode])
return mode;
}
return opt_machine_mode ();
}
opt_machine_mode
mode_for_int_vector (unsigned int int_bits, poly_uint64 nunits)
{
scalar_int_mode int_mode;
machine_mode vec_mode;
if (int_mode_for_size (int_bits, 0).exists (&int_mode)
&& mode_for_vector (int_mode, nunits).exists (&vec_mode))
return vec_mode;
return opt_machine_mode ();
}
unsigned int
get_mode_alignment (machine_mode mode)
{
return MIN (BIGGEST_ALIGNMENT, MAX (1, mode_base_align[mode]*BITS_PER_UNIT));
}
static machine_mode
mode_for_array (tree elem_type, tree size)
{
tree elem_size;
poly_uint64 int_size, int_elem_size;
unsigned HOST_WIDE_INT num_elems;
bool limit_p;
elem_size = TYPE_SIZE (elem_type);
if (simple_cst_equal (size, elem_size))
return TYPE_MODE (elem_type);
limit_p = true;
if (poly_int_tree_p (size, &int_size)
&& poly_int_tree_p (elem_size, &int_elem_size)
&& maybe_ne (int_elem_size, 0U)
&& constant_multiple_p (int_size, int_elem_size, &num_elems))
{
machine_mode elem_mode = TYPE_MODE (elem_type);
machine_mode mode;
if (targetm.array_mode (elem_mode, num_elems).exists (&mode))
return mode;
if (targetm.array_mode_supported_p (elem_mode, num_elems))
limit_p = false;
}
return mode_for_size_tree (size, MODE_INT, limit_p).else_blk ();
}

static inline void
do_type_align (tree type, tree decl)
{
if (TYPE_ALIGN (type) > DECL_ALIGN (decl))
{
SET_DECL_ALIGN (decl, TYPE_ALIGN (type));
if (TREE_CODE (decl) == FIELD_DECL)
DECL_USER_ALIGN (decl) = TYPE_USER_ALIGN (type);
}
if (TYPE_WARN_IF_NOT_ALIGN (type) > DECL_WARN_IF_NOT_ALIGN (decl))
SET_DECL_WARN_IF_NOT_ALIGN (decl, TYPE_WARN_IF_NOT_ALIGN (type));
}
void
layout_decl (tree decl, unsigned int known_align)
{
tree type = TREE_TYPE (decl);
enum tree_code code = TREE_CODE (decl);
rtx rtl = NULL_RTX;
location_t loc = DECL_SOURCE_LOCATION (decl);
if (code == CONST_DECL)
return;
gcc_assert (code == VAR_DECL || code == PARM_DECL || code == RESULT_DECL
|| code == TYPE_DECL || code == FIELD_DECL);
rtl = DECL_RTL_IF_SET (decl);
if (type == error_mark_node)
type = void_type_node;
DECL_UNSIGNED (decl) = TYPE_UNSIGNED (type);
if (DECL_MODE (decl) == VOIDmode)
SET_DECL_MODE (decl, TYPE_MODE (type));
if (DECL_SIZE (decl) == 0)
{
DECL_SIZE (decl) = TYPE_SIZE (type);
DECL_SIZE_UNIT (decl) = TYPE_SIZE_UNIT (type);
}
else if (DECL_SIZE_UNIT (decl) == 0)
DECL_SIZE_UNIT (decl)
= fold_convert_loc (loc, sizetype,
size_binop_loc (loc, CEIL_DIV_EXPR, DECL_SIZE (decl),
bitsize_unit_node));
if (code != FIELD_DECL)
do_type_align (type, decl);
else
{
bool old_user_align = DECL_USER_ALIGN (decl);
bool zero_bitfield = false;
bool packed_p = DECL_PACKED (decl);
unsigned int mfa;
if (DECL_BIT_FIELD (decl))
{
DECL_BIT_FIELD_TYPE (decl) = type;
if (integer_zerop (DECL_SIZE (decl))
&& ! targetm.ms_bitfield_layout_p (DECL_FIELD_CONTEXT (decl)))
{
zero_bitfield = true;
packed_p = false;
if (PCC_BITFIELD_TYPE_MATTERS)
do_type_align (type, decl);
else
{
#ifdef EMPTY_FIELD_BOUNDARY
if (EMPTY_FIELD_BOUNDARY > DECL_ALIGN (decl))
{
SET_DECL_ALIGN (decl, EMPTY_FIELD_BOUNDARY);
DECL_USER_ALIGN (decl) = 0;
}
#endif
}
}
if (TYPE_SIZE (type) != 0
&& TREE_CODE (TYPE_SIZE (type)) == INTEGER_CST
&& GET_MODE_CLASS (TYPE_MODE (type)) == MODE_INT)
{
machine_mode xmode;
if (mode_for_size_tree (DECL_SIZE (decl),
MODE_INT, 1).exists (&xmode))
{
unsigned int xalign = GET_MODE_ALIGNMENT (xmode);
if (!(xalign > BITS_PER_UNIT && DECL_PACKED (decl))
&& (known_align == 0 || known_align >= xalign))
{
SET_DECL_ALIGN (decl, MAX (xalign, DECL_ALIGN (decl)));
SET_DECL_MODE (decl, xmode);
DECL_BIT_FIELD (decl) = 0;
}
}
}
if (TYPE_MODE (type) == BLKmode && DECL_MODE (decl) == BLKmode
&& known_align >= TYPE_ALIGN (type)
&& DECL_ALIGN (decl) >= TYPE_ALIGN (type))
DECL_BIT_FIELD (decl) = 0;
}
else if (packed_p && DECL_USER_ALIGN (decl))
;
else
do_type_align (type, decl);
if (packed_p
&& !old_user_align)
SET_DECL_ALIGN (decl, MIN (DECL_ALIGN (decl), BITS_PER_UNIT));
if (! packed_p && ! DECL_USER_ALIGN (decl))
{
#ifdef BIGGEST_FIELD_ALIGNMENT
SET_DECL_ALIGN (decl, MIN (DECL_ALIGN (decl),
(unsigned) BIGGEST_FIELD_ALIGNMENT));
#endif
#ifdef ADJUST_FIELD_ALIGN
SET_DECL_ALIGN (decl, ADJUST_FIELD_ALIGN (decl, TREE_TYPE (decl),
DECL_ALIGN (decl)));
#endif
}
if (zero_bitfield)
mfa = initial_max_fld_align * BITS_PER_UNIT;
else
mfa = maximum_field_alignment;
if (mfa != 0)
SET_DECL_ALIGN (decl, MIN (DECL_ALIGN (decl), mfa));
}
if (DECL_SIZE (decl) != 0 && TREE_CODE (DECL_SIZE (decl)) != INTEGER_CST)
DECL_SIZE (decl) = variable_size (DECL_SIZE (decl));
if (DECL_SIZE_UNIT (decl) != 0
&& TREE_CODE (DECL_SIZE_UNIT (decl)) != INTEGER_CST)
DECL_SIZE_UNIT (decl) = variable_size (DECL_SIZE_UNIT (decl));
if (warn_larger_than
&& (code == VAR_DECL || code == PARM_DECL)
&& ! DECL_EXTERNAL (decl))
{
tree size = DECL_SIZE_UNIT (decl);
if (size != 0 && TREE_CODE (size) == INTEGER_CST
&& compare_tree_int (size, larger_than_size) > 0)
{
int size_as_int = TREE_INT_CST_LOW (size);
if (compare_tree_int (size, size_as_int) == 0)
warning (OPT_Wlarger_than_, "size of %q+D is %d bytes", decl, size_as_int);
else
warning (OPT_Wlarger_than_, "size of %q+D is larger than %wd bytes",
decl, larger_than_size);
}
}
if (rtl)
{
PUT_MODE (rtl, DECL_MODE (decl));
SET_DECL_RTL (decl, 0);
if (MEM_P (rtl))
set_mem_attributes (rtl, decl, 1);
SET_DECL_RTL (decl, rtl);
}
}
void
relayout_decl (tree decl)
{
DECL_SIZE (decl) = DECL_SIZE_UNIT (decl) = 0;
SET_DECL_MODE (decl, VOIDmode);
if (!DECL_USER_ALIGN (decl))
SET_DECL_ALIGN (decl, 0);
if (DECL_RTL_SET_P (decl))
SET_DECL_RTL (decl, 0);
layout_decl (decl, 0);
}

record_layout_info
start_record_layout (tree t)
{
record_layout_info rli = XNEW (struct record_layout_info_s);
rli->t = t;
rli->record_align = MAX (BITS_PER_UNIT, TYPE_ALIGN (t));
rli->unpacked_align = rli->record_align;
rli->offset_align = MAX (rli->record_align, BIGGEST_ALIGNMENT);
#ifdef STRUCTURE_SIZE_BOUNDARY
if (! TYPE_PACKED (t))
{
unsigned tmp;
tmp = (unsigned) STRUCTURE_SIZE_BOUNDARY;
if (maximum_field_alignment != 0)
tmp = MIN (tmp, maximum_field_alignment);
rli->record_align = MAX (rli->record_align, tmp);
}
#endif
rli->offset = size_zero_node;
rli->bitpos = bitsize_zero_node;
rli->prev_field = 0;
rli->pending_statics = 0;
rli->packed_maybe_necessary = 0;
rli->remaining_in_alignment = 0;
return rli;
}
static tree
bits_from_bytes (tree x)
{
if (POLY_INT_CST_P (x))
return build_poly_int_cst
(bitsizetype,
poly_wide_int::from (poly_int_cst_value (x),
TYPE_PRECISION (bitsizetype),
TYPE_SIGN (TREE_TYPE (x))));
x = fold_convert (bitsizetype, x);
gcc_checking_assert (x);
return x;
}
tree
bit_from_pos (tree offset, tree bitpos)
{
return size_binop (PLUS_EXPR, bitpos,
size_binop (MULT_EXPR, bits_from_bytes (offset),
bitsize_unit_node));
}
tree
byte_from_pos (tree offset, tree bitpos)
{
tree bytepos;
if (TREE_CODE (bitpos) == MULT_EXPR
&& tree_int_cst_equal (TREE_OPERAND (bitpos, 1), bitsize_unit_node))
bytepos = TREE_OPERAND (bitpos, 0);
else
bytepos = size_binop (TRUNC_DIV_EXPR, bitpos, bitsize_unit_node);
return size_binop (PLUS_EXPR, offset, fold_convert (sizetype, bytepos));
}
void
pos_from_bit (tree *poffset, tree *pbitpos, unsigned int off_align,
tree pos)
{
tree toff_align = bitsize_int (off_align);
if (TREE_CODE (pos) == MULT_EXPR
&& tree_int_cst_equal (TREE_OPERAND (pos, 1), toff_align))
{
*poffset = size_binop (MULT_EXPR,
fold_convert (sizetype, TREE_OPERAND (pos, 0)),
size_int (off_align / BITS_PER_UNIT));
*pbitpos = bitsize_zero_node;
}
else
{
*poffset = size_binop (MULT_EXPR,
fold_convert (sizetype,
size_binop (FLOOR_DIV_EXPR, pos,
toff_align)),
size_int (off_align / BITS_PER_UNIT));
*pbitpos = size_binop (FLOOR_MOD_EXPR, pos, toff_align);
}
}
void
normalize_offset (tree *poffset, tree *pbitpos, unsigned int off_align)
{
if (compare_tree_int (*pbitpos, off_align) >= 0)
{
tree offset, bitpos;
pos_from_bit (&offset, &bitpos, off_align, *pbitpos);
*poffset = size_binop (PLUS_EXPR, *poffset, offset);
*pbitpos = bitpos;
}
}
DEBUG_FUNCTION void
debug_rli (record_layout_info rli)
{
print_node_brief (stderr, "type", rli->t, 0);
print_node_brief (stderr, "\noffset", rli->offset, 0);
print_node_brief (stderr, " bitpos", rli->bitpos, 0);
fprintf (stderr, "\naligns: rec = %u, unpack = %u, off = %u\n",
rli->record_align, rli->unpacked_align,
rli->offset_align);
if (targetm.ms_bitfield_layout_p (rli->t))
fprintf (stderr, "remaining in alignment = %u\n", rli->remaining_in_alignment);
if (rli->packed_maybe_necessary)
fprintf (stderr, "packed may be necessary\n");
if (!vec_safe_is_empty (rli->pending_statics))
{
fprintf (stderr, "pending statics:\n");
debug (rli->pending_statics);
}
}
void
normalize_rli (record_layout_info rli)
{
normalize_offset (&rli->offset, &rli->bitpos, rli->offset_align);
}
tree
rli_size_unit_so_far (record_layout_info rli)
{
return byte_from_pos (rli->offset, rli->bitpos);
}
tree
rli_size_so_far (record_layout_info rli)
{
return bit_from_pos (rli->offset, rli->bitpos);
}
unsigned int
update_alignment_for_field (record_layout_info rli, tree field,
unsigned int known_align)
{
unsigned int desired_align;
tree type = TREE_TYPE (field);
bool user_align;
bool is_bitfield;
if (TREE_CODE (type) == ERROR_MARK)
return 0;
layout_decl (field, known_align);
desired_align = DECL_ALIGN (field);
user_align = DECL_USER_ALIGN (field);
is_bitfield = (type != error_mark_node
&& DECL_BIT_FIELD_TYPE (field)
&& ! integer_zerop (TYPE_SIZE (type)));
if (targetm.ms_bitfield_layout_p (rli->t))
{
if (!is_bitfield
|| ((DECL_SIZE (field) == NULL_TREE
|| !integer_zerop (DECL_SIZE (field)))
? !DECL_PACKED (field)
: (rli->prev_field
&& DECL_BIT_FIELD_TYPE (rli->prev_field)
&& ! integer_zerop (DECL_SIZE (rli->prev_field)))))
{
unsigned int type_align = TYPE_ALIGN (type);
if (!is_bitfield && DECL_PACKED (field))
type_align = desired_align;
else
type_align = MAX (type_align, desired_align);
if (maximum_field_alignment != 0)
type_align = MIN (type_align, maximum_field_alignment);
rli->record_align = MAX (rli->record_align, type_align);
rli->unpacked_align = MAX (rli->unpacked_align, TYPE_ALIGN (type));
}
}
else if (is_bitfield && PCC_BITFIELD_TYPE_MATTERS)
{
if (DECL_NAME (field) != 0
|| targetm.align_anon_bitfield ())
{
unsigned int type_align = TYPE_ALIGN (type);
#ifdef ADJUST_FIELD_ALIGN
if (! TYPE_USER_ALIGN (type))
type_align = ADJUST_FIELD_ALIGN (field, type, type_align);
#endif
if (integer_zerop (DECL_SIZE (field)))
{
if (initial_max_fld_align)
type_align = MIN (type_align,
initial_max_fld_align * BITS_PER_UNIT);
}
else if (maximum_field_alignment != 0)
type_align = MIN (type_align, maximum_field_alignment);
else if (DECL_PACKED (field))
type_align = MIN (type_align, BITS_PER_UNIT);
rli->record_align = MAX (rli->record_align, desired_align);
rli->record_align = MAX (rli->record_align, type_align);
if (warn_packed)
rli->unpacked_align = MAX (rli->unpacked_align, TYPE_ALIGN (type));
user_align |= TYPE_USER_ALIGN (type);
}
}
else
{
rli->record_align = MAX (rli->record_align, desired_align);
rli->unpacked_align = MAX (rli->unpacked_align, TYPE_ALIGN (type));
}
TYPE_USER_ALIGN (rli->t) |= user_align;
return desired_align;
}
static void
handle_warn_if_not_align (tree field, unsigned int record_align)
{
tree type = TREE_TYPE (field);
if (type == error_mark_node)
return;
unsigned int warn_if_not_align = 0;
int opt_w = 0;
if (warn_if_not_aligned)
{
warn_if_not_align = DECL_WARN_IF_NOT_ALIGN (field);
if (!warn_if_not_align)
warn_if_not_align = TYPE_WARN_IF_NOT_ALIGN (type);
if (warn_if_not_align)
opt_w = OPT_Wif_not_aligned;
}
if (!warn_if_not_align
&& warn_packed_not_aligned
&& lookup_attribute ("aligned", TYPE_ATTRIBUTES (type)))
{
warn_if_not_align = TYPE_ALIGN (type);
opt_w = OPT_Wpacked_not_aligned;
}
if (!warn_if_not_align)
return;
tree context = DECL_CONTEXT (field);
warn_if_not_align /= BITS_PER_UNIT;
record_align /= BITS_PER_UNIT;
if ((record_align % warn_if_not_align) != 0)
warning (opt_w, "alignment %u of %qT is less than %u",
record_align, context, warn_if_not_align);
tree off = byte_position (field);
if (!multiple_of_p (TREE_TYPE (off), off, size_int (warn_if_not_align)))
{
if (TREE_CODE (off) == INTEGER_CST)
warning (opt_w, "%q+D offset %E in %qT isn%'t aligned to %u",
field, off, context, warn_if_not_align);
else
warning (opt_w, "%q+D offset %E in %qT may not be aligned to %u",
field, off, context, warn_if_not_align);
}
}
static void
place_union_field (record_layout_info rli, tree field)
{
update_alignment_for_field (rli, field, 0);
DECL_FIELD_OFFSET (field) = size_zero_node;
DECL_FIELD_BIT_OFFSET (field) = bitsize_zero_node;
SET_DECL_OFFSET_ALIGN (field, BIGGEST_ALIGNMENT);
handle_warn_if_not_align (field, rli->record_align);
if (TREE_CODE (TREE_TYPE (field)) == ERROR_MARK)
return;
if (AGGREGATE_TYPE_P (TREE_TYPE (field))
&& TYPE_TYPELESS_STORAGE (TREE_TYPE (field)))
TYPE_TYPELESS_STORAGE (rli->t) = 1;
if (TREE_CODE (rli->t) == UNION_TYPE)
rli->offset = size_binop (MAX_EXPR, rli->offset, DECL_SIZE_UNIT (field));
else if (TREE_CODE (rli->t) == QUAL_UNION_TYPE)
rli->offset = fold_build3 (COND_EXPR, sizetype, DECL_QUALIFIER (field),
DECL_SIZE_UNIT (field), rli->offset);
}
static int
excess_unit_span (HOST_WIDE_INT byte_offset, HOST_WIDE_INT bit_offset,
HOST_WIDE_INT size, HOST_WIDE_INT align, tree type)
{
unsigned HOST_WIDE_INT offset = byte_offset * BITS_PER_UNIT + bit_offset;
offset = offset % align;
return ((offset + size + align - 1) / align
> tree_to_uhwi (TYPE_SIZE (type)) / align);
}
void
place_field (record_layout_info rli, tree field)
{
unsigned int desired_align;
unsigned int known_align;
unsigned int actual_align;
tree type = TREE_TYPE (field);
gcc_assert (TREE_CODE (field) != ERROR_MARK);
if (VAR_P (field))
{
vec_safe_push (rli->pending_statics, field);
return;
}
else if (TREE_CODE (field) != FIELD_DECL)
return;
else if (TREE_CODE (rli->t) != RECORD_TYPE)
{
place_union_field (rli, field);
return;
}
else if (TREE_CODE (type) == ERROR_MARK)
{
DECL_FIELD_OFFSET (field) = rli->offset;
DECL_FIELD_BIT_OFFSET (field) = rli->bitpos;
SET_DECL_OFFSET_ALIGN (field, rli->offset_align);
handle_warn_if_not_align (field, rli->record_align);
return;
}
if (AGGREGATE_TYPE_P (type)
&& TYPE_TYPELESS_STORAGE (type))
TYPE_TYPELESS_STORAGE (rli->t) = 1;
if (! integer_zerop (rli->bitpos))
known_align = least_bit_hwi (tree_to_uhwi (rli->bitpos));
else if (integer_zerop (rli->offset))
known_align = 0;
else if (tree_fits_uhwi_p (rli->offset))
known_align = (BITS_PER_UNIT
* least_bit_hwi (tree_to_uhwi (rli->offset)));
else
known_align = rli->offset_align;
desired_align = update_alignment_for_field (rli, field, known_align);
if (known_align == 0)
known_align = MAX (BIGGEST_ALIGNMENT, rli->record_align);
if (warn_packed && DECL_PACKED (field))
{
if (known_align >= TYPE_ALIGN (type))
{
if (TYPE_ALIGN (type) > desired_align)
{
if (STRICT_ALIGNMENT)
warning (OPT_Wattributes, "packed attribute causes "
"inefficient alignment for %q+D", field);
else if (!TYPE_PACKED (rli->t))
warning (OPT_Wattributes, "packed attribute is "
"unnecessary for %q+D", field);
}
}
else
rli->packed_maybe_necessary = 1;
}
if (known_align < desired_align
&& (! targetm.ms_bitfield_layout_p (rli->t)
|| rli->prev_field == NULL))
{
if (!targetm.ms_bitfield_layout_p (rli->t)
&& DECL_SOURCE_LOCATION (field) != BUILTINS_LOCATION)
warning (OPT_Wpadded, "padding struct to align %q+D", field);
if (desired_align < rli->offset_align)
rli->bitpos = round_up (rli->bitpos, desired_align);
else
{
rli->offset
= size_binop (PLUS_EXPR, rli->offset,
fold_convert (sizetype,
size_binop (CEIL_DIV_EXPR, rli->bitpos,
bitsize_unit_node)));
rli->bitpos = bitsize_zero_node;
rli->offset = round_up (rli->offset, desired_align / BITS_PER_UNIT);
}
if (! TREE_CONSTANT (rli->offset))
rli->offset_align = desired_align;
}
if (PCC_BITFIELD_TYPE_MATTERS
&& ! targetm.ms_bitfield_layout_p (rli->t)
&& TREE_CODE (field) == FIELD_DECL
&& type != error_mark_node
&& DECL_BIT_FIELD (field)
&& (! DECL_PACKED (field)
|| TYPE_ALIGN (type) <= BITS_PER_UNIT)
&& maximum_field_alignment == 0
&& ! integer_zerop (DECL_SIZE (field))
&& tree_fits_uhwi_p (DECL_SIZE (field))
&& tree_fits_uhwi_p (rli->offset)
&& tree_fits_uhwi_p (TYPE_SIZE (type)))
{
unsigned int type_align = TYPE_ALIGN (type);
tree dsize = DECL_SIZE (field);
HOST_WIDE_INT field_size = tree_to_uhwi (dsize);
HOST_WIDE_INT offset = tree_to_uhwi (rli->offset);
HOST_WIDE_INT bit_offset = tree_to_shwi (rli->bitpos);
#ifdef ADJUST_FIELD_ALIGN
if (! TYPE_USER_ALIGN (type))
type_align = ADJUST_FIELD_ALIGN (field, type, type_align);
#endif
if (excess_unit_span (offset, bit_offset, field_size, type_align, type))
{
if (DECL_PACKED (field))
{
if (warn_packed_bitfield_compat == 1)
inform
(input_location,
"offset of packed bit-field %qD has changed in GCC 4.4",
field);
}
else
rli->bitpos = round_up (rli->bitpos, type_align);
}
if (! DECL_PACKED (field))
TYPE_USER_ALIGN (rli->t) |= TYPE_USER_ALIGN (type);
SET_TYPE_WARN_IF_NOT_ALIGN (rli->t,
TYPE_WARN_IF_NOT_ALIGN (type));
}
#ifdef BITFIELD_NBYTES_LIMITED
if (BITFIELD_NBYTES_LIMITED
&& ! targetm.ms_bitfield_layout_p (rli->t)
&& TREE_CODE (field) == FIELD_DECL
&& type != error_mark_node
&& DECL_BIT_FIELD_TYPE (field)
&& ! DECL_PACKED (field)
&& ! integer_zerop (DECL_SIZE (field))
&& tree_fits_uhwi_p (DECL_SIZE (field))
&& tree_fits_uhwi_p (rli->offset)
&& tree_fits_uhwi_p (TYPE_SIZE (type)))
{
unsigned int type_align = TYPE_ALIGN (type);
tree dsize = DECL_SIZE (field);
HOST_WIDE_INT field_size = tree_to_uhwi (dsize);
HOST_WIDE_INT offset = tree_to_uhwi (rli->offset);
HOST_WIDE_INT bit_offset = tree_to_shwi (rli->bitpos);
#ifdef ADJUST_FIELD_ALIGN
if (! TYPE_USER_ALIGN (type))
type_align = ADJUST_FIELD_ALIGN (field, type, type_align);
#endif
if (maximum_field_alignment != 0)
type_align = MIN (type_align, maximum_field_alignment);
else if (DECL_PACKED (field))
type_align = MIN (type_align, BITS_PER_UNIT);
if (excess_unit_span (offset, bit_offset, field_size, type_align, type))
rli->bitpos = round_up (rli->bitpos, type_align);
TYPE_USER_ALIGN (rli->t) |= TYPE_USER_ALIGN (type);
SET_TYPE_WARN_IF_NOT_ALIGN (rli->t,
TYPE_WARN_IF_NOT_ALIGN (type));
}
#endif
if (targetm.ms_bitfield_layout_p (rli->t))
{
tree prev_saved = rli->prev_field;
tree prev_type = prev_saved ? DECL_BIT_FIELD_TYPE (prev_saved) : NULL;
if (rli->prev_field)
{
bool realign_p = known_align < desired_align;
if (DECL_BIT_FIELD_TYPE (field)
&& !integer_zerop (DECL_SIZE (field))
&& !integer_zerop (DECL_SIZE (rli->prev_field))
&& tree_fits_shwi_p (DECL_SIZE (rli->prev_field))
&& tree_fits_uhwi_p (TYPE_SIZE (type))
&& simple_cst_equal (TYPE_SIZE (type), TYPE_SIZE (prev_type)))
{
HOST_WIDE_INT bitsize = tree_to_uhwi (DECL_SIZE (field));
if (rli->remaining_in_alignment < bitsize)
{
HOST_WIDE_INT typesize = tree_to_uhwi (TYPE_SIZE (type));
rli->bitpos
= size_binop (PLUS_EXPR, rli->bitpos,
bitsize_int (rli->remaining_in_alignment));
rli->prev_field = field;
if (typesize < bitsize)
rli->remaining_in_alignment = 0;
else
rli->remaining_in_alignment = typesize - bitsize;
}
else
{
rli->remaining_in_alignment -= bitsize;
realign_p = false;
}
}
else
{
if (!integer_zerop (DECL_SIZE (rli->prev_field)))
{
rli->bitpos
= size_binop (PLUS_EXPR, rli->bitpos,
bitsize_int (rli->remaining_in_alignment));
}
else
prev_saved = NULL;
if (!DECL_BIT_FIELD_TYPE (field)
|| integer_zerop (DECL_SIZE (field)))
rli->prev_field = NULL;
}
if (realign_p)
{
if (desired_align < rli->offset_align)
rli->bitpos = round_up (rli->bitpos, desired_align);
else
{
tree d = size_binop (CEIL_DIV_EXPR, rli->bitpos,
bitsize_unit_node);
rli->offset = size_binop (PLUS_EXPR, rli->offset,
fold_convert (sizetype, d));
rli->bitpos = bitsize_zero_node;
rli->offset = round_up (rli->offset,
desired_align / BITS_PER_UNIT);
}
if (! TREE_CONSTANT (rli->offset))
rli->offset_align = desired_align;
}
normalize_rli (rli);
}
if (!DECL_BIT_FIELD_TYPE (field)
|| (prev_saved != NULL
? !simple_cst_equal (TYPE_SIZE (type), TYPE_SIZE (prev_type))
: !integer_zerop (DECL_SIZE (field))))
{
unsigned int type_align = BITS_PER_UNIT;
if (DECL_SIZE (field) != NULL
&& tree_fits_uhwi_p (TYPE_SIZE (TREE_TYPE (field)))
&& tree_fits_uhwi_p (DECL_SIZE (field)))
{
unsigned HOST_WIDE_INT bitsize
= tree_to_uhwi (DECL_SIZE (field));
unsigned HOST_WIDE_INT typesize
= tree_to_uhwi (TYPE_SIZE (TREE_TYPE (field)));
if (typesize < bitsize)
rli->remaining_in_alignment = 0;
else
rli->remaining_in_alignment = typesize - bitsize;
}
if (! DECL_PACKED (field))
type_align = TYPE_ALIGN (TREE_TYPE (field));
if (maximum_field_alignment != 0)
type_align = MIN (type_align, maximum_field_alignment);
rli->bitpos = round_up (rli->bitpos, type_align);
rli->prev_field = NULL;
}
}
normalize_rli (rli);
DECL_FIELD_OFFSET (field) = rli->offset;
DECL_FIELD_BIT_OFFSET (field) = rli->bitpos;
SET_DECL_OFFSET_ALIGN (field, rli->offset_align);
handle_warn_if_not_align (field, rli->record_align);
if (TREE_CODE (DECL_FIELD_OFFSET (field)) != INTEGER_CST)
DECL_FIELD_OFFSET (field) = variable_size (DECL_FIELD_OFFSET (field));
if (! integer_zerop (DECL_FIELD_BIT_OFFSET (field)))
actual_align = least_bit_hwi (tree_to_uhwi (DECL_FIELD_BIT_OFFSET (field)));
else if (integer_zerop (DECL_FIELD_OFFSET (field)))
actual_align = MAX (BIGGEST_ALIGNMENT, rli->record_align);
else if (tree_fits_uhwi_p (DECL_FIELD_OFFSET (field)))
actual_align = (BITS_PER_UNIT
* least_bit_hwi (tree_to_uhwi (DECL_FIELD_OFFSET (field))));
else
actual_align = DECL_OFFSET_ALIGN (field);
if (known_align != actual_align)
layout_decl (field, actual_align);
if (rli->prev_field == NULL && DECL_BIT_FIELD_TYPE (field))
rli->prev_field = field;
if (DECL_SIZE (field) == 0)
;
else if (TREE_CODE (DECL_SIZE (field)) != INTEGER_CST
|| TREE_OVERFLOW (DECL_SIZE (field)))
{
rli->offset
= size_binop (PLUS_EXPR, rli->offset,
fold_convert (sizetype,
size_binop (CEIL_DIV_EXPR, rli->bitpos,
bitsize_unit_node)));
rli->offset
= size_binop (PLUS_EXPR, rli->offset, DECL_SIZE_UNIT (field));
rli->bitpos = bitsize_zero_node;
rli->offset_align = MIN (rli->offset_align, desired_align);
if (!multiple_of_p (bitsizetype, DECL_SIZE (field),
bitsize_int (rli->offset_align)))
{
tree type = strip_array_types (TREE_TYPE (field));
if (TREE_CODE (TYPE_SIZE (type)) == INTEGER_CST)
{
if (TREE_INT_CST_LOW (TYPE_SIZE (type)))
{
unsigned HOST_WIDE_INT sz
= least_bit_hwi (TREE_INT_CST_LOW (TYPE_SIZE (type)));
rli->offset_align = MIN (rli->offset_align, sz);
}
}
else
rli->offset_align = MIN (rli->offset_align, BITS_PER_UNIT);
}
}
else if (targetm.ms_bitfield_layout_p (rli->t))
{
rli->bitpos = size_binop (PLUS_EXPR, rli->bitpos, DECL_SIZE (field));
if (DECL_BIT_FIELD_TYPE (field)
&& !integer_zerop (DECL_SIZE (field)))
{
tree probe = field;
while ((probe = DECL_CHAIN (probe)))
if (TREE_CODE (probe) == FIELD_DECL)
break;
if (!probe)
rli->bitpos = size_binop (PLUS_EXPR, rli->bitpos,
bitsize_int (rli->remaining_in_alignment));
}
normalize_rli (rli);
}
else
{
rli->bitpos = size_binop (PLUS_EXPR, rli->bitpos, DECL_SIZE (field));
normalize_rli (rli);
}
}
static void
finalize_record_size (record_layout_info rli)
{
tree unpadded_size, unpadded_size_unit;
rli->offset_align = BITS_PER_UNIT;
normalize_rli (rli);
#ifdef ROUND_TYPE_ALIGN
SET_TYPE_ALIGN (rli->t, ROUND_TYPE_ALIGN (rli->t, TYPE_ALIGN (rli->t),
rli->record_align));
#else
SET_TYPE_ALIGN (rli->t, MAX (TYPE_ALIGN (rli->t), rli->record_align));
#endif
unpadded_size = rli_size_so_far (rli);
unpadded_size_unit = rli_size_unit_so_far (rli);
if (! integer_zerop (rli->bitpos))
unpadded_size_unit
= size_binop (PLUS_EXPR, unpadded_size_unit, size_one_node);
TYPE_SIZE (rli->t) = round_up (unpadded_size, TYPE_ALIGN (rli->t));
TYPE_SIZE_UNIT (rli->t)
= round_up (unpadded_size_unit, TYPE_ALIGN_UNIT (rli->t));
if (TREE_CONSTANT (unpadded_size)
&& simple_cst_equal (unpadded_size, TYPE_SIZE (rli->t)) == 0
&& input_location != BUILTINS_LOCATION)
warning (OPT_Wpadded, "padding struct size to alignment boundary");
if (warn_packed && TREE_CODE (rli->t) == RECORD_TYPE
&& TYPE_PACKED (rli->t) && ! rli->packed_maybe_necessary
&& TREE_CONSTANT (unpadded_size))
{
tree unpacked_size;
#ifdef ROUND_TYPE_ALIGN
rli->unpacked_align
= ROUND_TYPE_ALIGN (rli->t, TYPE_ALIGN (rli->t), rli->unpacked_align);
#else
rli->unpacked_align = MAX (TYPE_ALIGN (rli->t), rli->unpacked_align);
#endif
unpacked_size = round_up (TYPE_SIZE (rli->t), rli->unpacked_align);
if (simple_cst_equal (unpacked_size, TYPE_SIZE (rli->t)))
{
if (TYPE_NAME (rli->t))
{
tree name;
if (TREE_CODE (TYPE_NAME (rli->t)) == IDENTIFIER_NODE)
name = TYPE_NAME (rli->t);
else
name = DECL_NAME (TYPE_NAME (rli->t));
if (STRICT_ALIGNMENT)
warning (OPT_Wpacked, "packed attribute causes inefficient "
"alignment for %qE", name);
else
warning (OPT_Wpacked,
"packed attribute is unnecessary for %qE", name);
}
else
{
if (STRICT_ALIGNMENT)
warning (OPT_Wpacked,
"packed attribute causes inefficient alignment");
else
warning (OPT_Wpacked, "packed attribute is unnecessary");
}
}
}
}
void
compute_record_mode (tree type)
{
tree field;
machine_mode mode = VOIDmode;
SET_TYPE_MODE (type, BLKmode);
if (! tree_fits_uhwi_p (TYPE_SIZE (type)))
return;
for (field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
{
if (TREE_CODE (field) != FIELD_DECL)
continue;
if (TREE_CODE (TREE_TYPE (field)) == ERROR_MARK
|| (TYPE_MODE (TREE_TYPE (field)) == BLKmode
&& ! TYPE_NO_FORCE_BLK (TREE_TYPE (field))
&& !(TYPE_SIZE (TREE_TYPE (field)) != 0
&& integer_zerop (TYPE_SIZE (TREE_TYPE (field)))))
|| ! tree_fits_uhwi_p (bit_position (field))
|| DECL_SIZE (field) == 0
|| ! tree_fits_uhwi_p (DECL_SIZE (field)))
return;
if (simple_cst_equal (TYPE_SIZE (type), DECL_SIZE (field)))
mode = DECL_MODE (field);
if (targetm.member_type_forces_blk (field, mode))
return;
}
if (TREE_CODE (type) == RECORD_TYPE && mode != VOIDmode
&& tree_fits_uhwi_p (TYPE_SIZE (type))
&& known_eq (GET_MODE_BITSIZE (mode), tree_to_uhwi (TYPE_SIZE (type))))
;
else
mode = mode_for_size_tree (TYPE_SIZE (type), MODE_INT, 1).else_blk ();
if (mode != BLKmode
&& STRICT_ALIGNMENT
&& ! (TYPE_ALIGN (type) >= BIGGEST_ALIGNMENT
|| TYPE_ALIGN (type) >= GET_MODE_ALIGNMENT (mode)))
{
TYPE_NO_FORCE_BLK (type) = 1;
mode = BLKmode;
}
SET_TYPE_MODE (type, mode);
}
static void
finalize_type_size (tree type)
{
if (TYPE_MODE (type) != BLKmode
&& TYPE_MODE (type) != VOIDmode
&& (STRICT_ALIGNMENT || !AGGREGATE_TYPE_P (type)))
{
unsigned mode_align = GET_MODE_ALIGNMENT (TYPE_MODE (type));
if (mode_align >= TYPE_ALIGN (type))
{
SET_TYPE_ALIGN (type, mode_align);
TYPE_USER_ALIGN (type) = 0;
}
}
#ifdef ROUND_TYPE_ALIGN
SET_TYPE_ALIGN (type,
ROUND_TYPE_ALIGN (type, TYPE_ALIGN (type), BITS_PER_UNIT));
#endif
if (TYPE_SIZE_UNIT (type) == 0 && TYPE_SIZE (type) != 0)
TYPE_SIZE_UNIT (type)
= fold_convert (sizetype,
size_binop (FLOOR_DIV_EXPR, TYPE_SIZE (type),
bitsize_unit_node));
if (TYPE_SIZE (type) != 0)
{
TYPE_SIZE (type) = round_up (TYPE_SIZE (type), TYPE_ALIGN (type));
TYPE_SIZE_UNIT (type)
= round_up (TYPE_SIZE_UNIT (type), TYPE_ALIGN_UNIT (type));
}
if (TYPE_SIZE (type) != 0 && TREE_CODE (TYPE_SIZE (type)) != INTEGER_CST)
TYPE_SIZE (type) = variable_size (TYPE_SIZE (type));
if (TYPE_SIZE_UNIT (type) != 0
&& TREE_CODE (TYPE_SIZE_UNIT (type)) != INTEGER_CST)
TYPE_SIZE_UNIT (type) = variable_size (TYPE_SIZE_UNIT (type));
TYPE_EMPTY_P (type) = targetm.calls.empty_record_p (type);
if (TYPE_NEXT_VARIANT (type)
|| type != TYPE_MAIN_VARIANT (type))
{
tree variant;
tree size = TYPE_SIZE (type);
tree size_unit = TYPE_SIZE_UNIT (type);
unsigned int align = TYPE_ALIGN (type);
unsigned int precision = TYPE_PRECISION (type);
unsigned int user_align = TYPE_USER_ALIGN (type);
machine_mode mode = TYPE_MODE (type);
bool empty_p = TYPE_EMPTY_P (type);
for (variant = TYPE_MAIN_VARIANT (type);
variant != 0;
variant = TYPE_NEXT_VARIANT (variant))
{
TYPE_SIZE (variant) = size;
TYPE_SIZE_UNIT (variant) = size_unit;
unsigned valign = align;
if (TYPE_USER_ALIGN (variant))
valign = MAX (valign, TYPE_ALIGN (variant));
else
TYPE_USER_ALIGN (variant) = user_align;
SET_TYPE_ALIGN (variant, valign);
TYPE_PRECISION (variant) = precision;
SET_TYPE_MODE (variant, mode);
TYPE_EMPTY_P (variant) = empty_p;
}
}
}
static tree
start_bitfield_representative (tree field)
{
tree repr = make_node (FIELD_DECL);
DECL_FIELD_OFFSET (repr) = DECL_FIELD_OFFSET (field);
DECL_FIELD_BIT_OFFSET (repr)
= size_binop (BIT_AND_EXPR,
DECL_FIELD_BIT_OFFSET (field),
bitsize_int (~(BITS_PER_UNIT - 1)));
SET_DECL_OFFSET_ALIGN (repr, DECL_OFFSET_ALIGN (field));
DECL_SIZE (repr) = DECL_SIZE (field);
DECL_SIZE_UNIT (repr) = DECL_SIZE_UNIT (field);
DECL_PACKED (repr) = DECL_PACKED (field);
DECL_CONTEXT (repr) = DECL_CONTEXT (field);
DECL_NONADDRESSABLE_P (repr) = 1;
return repr;
}
static void
finish_bitfield_representative (tree repr, tree field)
{
unsigned HOST_WIDE_INT bitsize, maxbitsize;
tree nextf, size;
size = size_diffop (DECL_FIELD_OFFSET (field),
DECL_FIELD_OFFSET (repr));
while (TREE_CODE (size) == COMPOUND_EXPR)
size = TREE_OPERAND (size, 1);
gcc_assert (tree_fits_uhwi_p (size));
bitsize = (tree_to_uhwi (size) * BITS_PER_UNIT
+ tree_to_uhwi (DECL_FIELD_BIT_OFFSET (field))
- tree_to_uhwi (DECL_FIELD_BIT_OFFSET (repr))
+ tree_to_uhwi (DECL_SIZE (field)));
bitsize = (bitsize + BITS_PER_UNIT - 1) & ~(BITS_PER_UNIT - 1);
nextf = DECL_CHAIN (field);
while (nextf && TREE_CODE (nextf) != FIELD_DECL)
nextf = DECL_CHAIN (nextf);
if (nextf)
{
tree maxsize;
if (TREE_TYPE (nextf) == error_mark_node)
return;
maxsize = size_diffop (DECL_FIELD_OFFSET (nextf),
DECL_FIELD_OFFSET (repr));
if (tree_fits_uhwi_p (maxsize))
{
maxbitsize = (tree_to_uhwi (maxsize) * BITS_PER_UNIT
+ tree_to_uhwi (DECL_FIELD_BIT_OFFSET (nextf))
- tree_to_uhwi (DECL_FIELD_BIT_OFFSET (repr)));
maxbitsize = (maxbitsize + BITS_PER_UNIT - 1) & ~(BITS_PER_UNIT - 1);
}
else
maxbitsize = bitsize;
}
else
{
tree aggsize = lang_hooks.types.unit_size_without_reusable_padding
(DECL_CONTEXT (field));
tree maxsize = size_diffop (aggsize, DECL_FIELD_OFFSET (repr));
if (tree_fits_uhwi_p (maxsize))
maxbitsize = (tree_to_uhwi (maxsize) * BITS_PER_UNIT
- tree_to_uhwi (DECL_FIELD_BIT_OFFSET (repr)));
else
maxbitsize = bitsize;
}
gcc_assert (maxbitsize % BITS_PER_UNIT == 0);
opt_scalar_int_mode mode_iter;
FOR_EACH_MODE_IN_CLASS (mode_iter, MODE_INT)
if (GET_MODE_BITSIZE (mode_iter.require ()) >= bitsize)
break;
scalar_int_mode mode;
if (!mode_iter.exists (&mode)
|| GET_MODE_BITSIZE (mode) > maxbitsize
|| GET_MODE_BITSIZE (mode) > MAX_FIXED_MODE_SIZE)
{
DECL_SIZE (repr) = bitsize_int (bitsize);
DECL_SIZE_UNIT (repr) = size_int (bitsize / BITS_PER_UNIT);
SET_DECL_MODE (repr, BLKmode);
TREE_TYPE (repr) = build_array_type_nelts (unsigned_char_type_node,
bitsize / BITS_PER_UNIT);
}
else
{
unsigned HOST_WIDE_INT modesize = GET_MODE_BITSIZE (mode);
DECL_SIZE (repr) = bitsize_int (modesize);
DECL_SIZE_UNIT (repr) = size_int (modesize / BITS_PER_UNIT);
SET_DECL_MODE (repr, mode);
TREE_TYPE (repr) = lang_hooks.types.type_for_mode (mode, 1);
}
DECL_CHAIN (repr) = nextf;
}
void
finish_bitfield_layout (tree t)
{
tree field, prev;
tree repr = NULL_TREE;
if (TREE_CODE (t) != RECORD_TYPE)
return;
for (prev = NULL_TREE, field = TYPE_FIELDS (t);
field; field = DECL_CHAIN (field))
{
if (TREE_CODE (field) != FIELD_DECL)
continue;
if (!repr
&& DECL_BIT_FIELD_TYPE (field))
{
repr = start_bitfield_representative (field);
}
else if (repr
&& ! DECL_BIT_FIELD_TYPE (field))
{
finish_bitfield_representative (repr, prev);
repr = NULL_TREE;
}
else if (DECL_BIT_FIELD_TYPE (field))
{
gcc_assert (repr != NULL_TREE);
if (integer_zerop (DECL_SIZE (field)))
{
finish_bitfield_representative (repr, prev);
repr = NULL_TREE;
}
else if (!((tree_fits_uhwi_p (DECL_FIELD_OFFSET (repr))
&& tree_fits_uhwi_p (DECL_FIELD_OFFSET (field)))
|| operand_equal_p (DECL_FIELD_OFFSET (repr),
DECL_FIELD_OFFSET (field), 0)))
{
finish_bitfield_representative (repr, prev);
repr = start_bitfield_representative (field);
}
}
else
continue;
if (repr)
DECL_BIT_FIELD_REPRESENTATIVE (field) = repr;
prev = field;
}
if (repr)
finish_bitfield_representative (repr, prev);
}
void
finish_record_layout (record_layout_info rli, int free_p)
{
tree variant;
finalize_record_size (rli);
compute_record_mode (rli->t);
finalize_type_size (rli->t);
finish_bitfield_layout (rli->t);
for (variant = TYPE_NEXT_VARIANT (rli->t); variant;
variant = TYPE_NEXT_VARIANT (variant))
{
TYPE_PACKED (variant) = TYPE_PACKED (rli->t);
TYPE_REVERSE_STORAGE_ORDER (variant)
= TYPE_REVERSE_STORAGE_ORDER (rli->t);
}
while (!vec_safe_is_empty (rli->pending_statics))
layout_decl (rli->pending_statics->pop (), 0);
if (free_p)
{
vec_free (rli->pending_statics);
free (rli);
}
}

void
finish_builtin_struct (tree type, const char *name, tree fields,
tree align_type)
{
tree tail, next;
for (tail = NULL_TREE; fields; tail = fields, fields = next)
{
DECL_FIELD_CONTEXT (fields) = type;
next = DECL_CHAIN (fields);
DECL_CHAIN (fields) = tail;
}
TYPE_FIELDS (type) = tail;
if (align_type)
{
SET_TYPE_ALIGN (type, TYPE_ALIGN (align_type));
TYPE_USER_ALIGN (type) = TYPE_USER_ALIGN (align_type);
SET_TYPE_WARN_IF_NOT_ALIGN (type,
TYPE_WARN_IF_NOT_ALIGN (align_type));
}
layout_type (type);
#if 0 
TYPE_NAME (type) = make_type_decl (get_identifier (name), type);
#else
TYPE_NAME (type) = build_decl (BUILTINS_LOCATION,
TYPE_DECL, get_identifier (name), type);
#endif
TYPE_STUB_DECL (type) = TYPE_NAME (type);
layout_decl (TYPE_NAME (type), 0);
}
void
layout_type (tree type)
{
gcc_assert (type);
if (type == error_mark_node)
return;
type = TYPE_MAIN_VARIANT (type);
if (TYPE_SIZE (type))
return;
switch (TREE_CODE (type))
{
case LANG_TYPE:
gcc_unreachable ();
case BOOLEAN_TYPE:
case INTEGER_TYPE:
case ENUMERAL_TYPE:
{
scalar_int_mode mode
= smallest_int_mode_for_size (TYPE_PRECISION (type));
SET_TYPE_MODE (type, mode);
TYPE_SIZE (type) = bitsize_int (GET_MODE_BITSIZE (mode));
TYPE_SIZE_UNIT (type) = size_int (GET_MODE_SIZE (mode));
break;
}
case REAL_TYPE:
{
if (TYPE_MODE (type) == VOIDmode)
SET_TYPE_MODE
(type, float_mode_for_size (TYPE_PRECISION (type)).require ());
scalar_float_mode mode = as_a <scalar_float_mode> (TYPE_MODE (type));
TYPE_SIZE (type) = bitsize_int (GET_MODE_BITSIZE (mode));
TYPE_SIZE_UNIT (type) = size_int (GET_MODE_SIZE (mode));
break;
}
case FIXED_POINT_TYPE:
{
scalar_mode mode = SCALAR_TYPE_MODE (type);
TYPE_SIZE (type) = bitsize_int (GET_MODE_BITSIZE (mode));
TYPE_SIZE_UNIT (type) = size_int (GET_MODE_SIZE (mode));
break;
}
case COMPLEX_TYPE:
TYPE_UNSIGNED (type) = TYPE_UNSIGNED (TREE_TYPE (type));
SET_TYPE_MODE (type,
GET_MODE_COMPLEX_MODE (TYPE_MODE (TREE_TYPE (type))));
TYPE_SIZE (type) = bitsize_int (GET_MODE_BITSIZE (TYPE_MODE (type)));
TYPE_SIZE_UNIT (type) = size_int (GET_MODE_SIZE (TYPE_MODE (type)));
break;
case VECTOR_TYPE:
{
poly_uint64 nunits = TYPE_VECTOR_SUBPARTS (type);
tree innertype = TREE_TYPE (type);
if (TYPE_MODE (type) == VOIDmode)
SET_TYPE_MODE (type,
mode_for_vector (SCALAR_TYPE_MODE (innertype),
nunits).else_blk ());
TYPE_SATURATING (type) = TYPE_SATURATING (TREE_TYPE (type));
TYPE_UNSIGNED (type) = TYPE_UNSIGNED (TREE_TYPE (type));
if (VECTOR_BOOLEAN_TYPE_P (type)
&& type->type_common.mode != BLKmode)
TYPE_SIZE_UNIT (type)
= size_int (GET_MODE_SIZE (type->type_common.mode));
else
TYPE_SIZE_UNIT (type) = int_const_binop (MULT_EXPR,
TYPE_SIZE_UNIT (innertype),
size_int (nunits));
TYPE_SIZE (type) = int_const_binop
(MULT_EXPR,
bits_from_bytes (TYPE_SIZE_UNIT (type)),
bitsize_int (BITS_PER_UNIT));
SET_TYPE_ALIGN (type, targetm.vector_alignment (type));
gcc_assert (TYPE_ALIGN (type)
>= GET_MODE_ALIGNMENT (TYPE_MODE (type)));
break;
}
case VOID_TYPE:
SET_TYPE_ALIGN (type, 1);
TYPE_USER_ALIGN (type) = 0;
SET_TYPE_MODE (type, VOIDmode);
break;
case POINTER_BOUNDS_TYPE:
TYPE_SIZE (type) = bitsize_int (GET_MODE_BITSIZE (TYPE_MODE (type)));
TYPE_SIZE_UNIT (type) = size_int (GET_MODE_SIZE (TYPE_MODE (type)));
break;
case OFFSET_TYPE:
TYPE_SIZE (type) = bitsize_int (POINTER_SIZE);
TYPE_SIZE_UNIT (type) = size_int (POINTER_SIZE_UNITS);
SET_TYPE_MODE (type, int_mode_for_size (POINTER_SIZE, 0).require ());
TYPE_PRECISION (type) = POINTER_SIZE;
break;
case FUNCTION_TYPE:
case METHOD_TYPE:
SET_TYPE_MODE (type,
int_mode_for_size (FUNCTION_BOUNDARY, 0).else_blk ());
TYPE_SIZE (type) = bitsize_int (FUNCTION_BOUNDARY);
TYPE_SIZE_UNIT (type) = size_int (FUNCTION_BOUNDARY / BITS_PER_UNIT);
break;
case POINTER_TYPE:
case REFERENCE_TYPE:
{
scalar_int_mode mode = SCALAR_INT_TYPE_MODE (type);
TYPE_SIZE (type) = bitsize_int (GET_MODE_BITSIZE (mode));
TYPE_SIZE_UNIT (type) = size_int (GET_MODE_SIZE (mode));
TYPE_UNSIGNED (type) = 1;
TYPE_PRECISION (type) = GET_MODE_PRECISION (mode);
}
break;
case ARRAY_TYPE:
{
tree index = TYPE_DOMAIN (type);
tree element = TREE_TYPE (type);
if (index && TYPE_MAX_VALUE (index) && TYPE_MIN_VALUE (index)
&& TYPE_SIZE (element))
{
tree ub = TYPE_MAX_VALUE (index);
tree lb = TYPE_MIN_VALUE (index);
tree element_size = TYPE_SIZE (element);
tree length;
if (integer_zerop (element_size))
length = size_zero_node;
else
{
if (TREE_CODE (lb) == INTEGER_CST
&& TREE_CODE (ub) == INTEGER_CST
&& TYPE_UNSIGNED (TREE_TYPE (lb))
&& tree_int_cst_lt (ub, lb))
{
lb = wide_int_to_tree (ssizetype,
offset_int::from (wi::to_wide (lb),
SIGNED));
ub = wide_int_to_tree (ssizetype,
offset_int::from (wi::to_wide (ub),
SIGNED));
}
length
= fold_convert (sizetype,
size_binop (PLUS_EXPR,
build_int_cst (TREE_TYPE (lb), 1),
size_binop (MINUS_EXPR, ub, lb)));
}
if (integer_zerop (length)
&& TREE_OVERFLOW (length)
&& integer_zerop (lb))
length = size_zero_node;
TYPE_SIZE (type) = size_binop (MULT_EXPR, element_size,
bits_from_bytes (length));
if (TYPE_SIZE_UNIT (element))
TYPE_SIZE_UNIT (type)
= size_binop (MULT_EXPR, TYPE_SIZE_UNIT (element), length);
}
unsigned align = TYPE_ALIGN (element);
if (TYPE_USER_ALIGN (type))
align = MAX (align, TYPE_ALIGN (type));
else
TYPE_USER_ALIGN (type) = TYPE_USER_ALIGN (element);
if (!TYPE_WARN_IF_NOT_ALIGN (type))
SET_TYPE_WARN_IF_NOT_ALIGN (type,
TYPE_WARN_IF_NOT_ALIGN (element));
#ifdef ROUND_TYPE_ALIGN
align = ROUND_TYPE_ALIGN (type, align, BITS_PER_UNIT);
#else
align = MAX (align, BITS_PER_UNIT);
#endif
SET_TYPE_ALIGN (type, align);
SET_TYPE_MODE (type, BLKmode);
if (TYPE_SIZE (type) != 0
&& ! targetm.member_type_forces_blk (type, VOIDmode)
&& (TYPE_MODE (TREE_TYPE (type)) != BLKmode
|| TYPE_NO_FORCE_BLK (TREE_TYPE (type))))
{
SET_TYPE_MODE (type, mode_for_array (TREE_TYPE (type),
TYPE_SIZE (type)));
if (TYPE_MODE (type) != BLKmode
&& STRICT_ALIGNMENT && TYPE_ALIGN (type) < BIGGEST_ALIGNMENT
&& TYPE_ALIGN (type) < GET_MODE_ALIGNMENT (TYPE_MODE (type)))
{
TYPE_NO_FORCE_BLK (type) = 1;
SET_TYPE_MODE (type, BLKmode);
}
}
if (AGGREGATE_TYPE_P (element))
TYPE_TYPELESS_STORAGE (type) = TYPE_TYPELESS_STORAGE (element);
if (TYPE_SIZE_UNIT (element)
&& TREE_CODE (TYPE_SIZE_UNIT (element)) == INTEGER_CST
&& !TREE_OVERFLOW (TYPE_SIZE_UNIT (element))
&& !integer_zerop (TYPE_SIZE_UNIT (element))
&& compare_tree_int (TYPE_SIZE_UNIT (element),
TYPE_ALIGN_UNIT (element)) < 0)
error ("alignment of array elements is greater than element size");
break;
}
case RECORD_TYPE:
case UNION_TYPE:
case QUAL_UNION_TYPE:
{
tree field;
record_layout_info rli;
rli = start_record_layout (type);
if (TREE_CODE (type) == QUAL_UNION_TYPE)
TYPE_FIELDS (type) = nreverse (TYPE_FIELDS (type));
for (field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
place_field (rli, field);
if (TREE_CODE (type) == QUAL_UNION_TYPE)
TYPE_FIELDS (type) = nreverse (TYPE_FIELDS (type));
finish_record_layout (rli, true);
}
break;
default:
gcc_unreachable ();
}
if (!RECORD_OR_UNION_TYPE_P (type))
finalize_type_size (type);
if (AGGREGATE_TYPE_P (type))
gcc_assert (!TYPE_ALIAS_SET_KNOWN_P (type));
}
unsigned int
min_align_of_type (tree type)
{
unsigned int align = TYPE_ALIGN (type);
if (!TYPE_USER_ALIGN (type))
{
align = MIN (align, BIGGEST_ALIGNMENT);
#ifdef BIGGEST_FIELD_ALIGNMENT
align = MIN (align, BIGGEST_FIELD_ALIGNMENT);
#endif
unsigned int field_align = align;
#ifdef ADJUST_FIELD_ALIGN
field_align = ADJUST_FIELD_ALIGN (NULL_TREE, type, field_align);
#endif
align = MIN (align, field_align);
}
return align / BITS_PER_UNIT;
}

tree
make_signed_type (int precision)
{
tree type = make_node (INTEGER_TYPE);
TYPE_PRECISION (type) = precision;
fixup_signed_type (type);
return type;
}
tree
make_unsigned_type (int precision)
{
tree type = make_node (INTEGER_TYPE);
TYPE_PRECISION (type) = precision;
fixup_unsigned_type (type);
return type;
}

tree
make_fract_type (int precision, int unsignedp, int satp)
{
tree type = make_node (FIXED_POINT_TYPE);
TYPE_PRECISION (type) = precision;
if (satp)
TYPE_SATURATING (type) = 1;
TYPE_UNSIGNED (type) = unsignedp;
enum mode_class mclass = unsignedp ? MODE_UFRACT : MODE_FRACT;
SET_TYPE_MODE (type, mode_for_size (precision, mclass, 0).require ());
layout_type (type);
return type;
}
tree
make_accum_type (int precision, int unsignedp, int satp)
{
tree type = make_node (FIXED_POINT_TYPE);
TYPE_PRECISION (type) = precision;
if (satp)
TYPE_SATURATING (type) = 1;
TYPE_UNSIGNED (type) = unsignedp;
enum mode_class mclass = unsignedp ? MODE_UACCUM : MODE_ACCUM;
SET_TYPE_MODE (type, mode_for_size (precision, mclass, 0).require ());
layout_type (type);
return type;
}
void
initialize_sizetypes (void)
{
int precision, bprecision;
if (strcmp (SIZETYPE, "unsigned int") == 0)
precision = INT_TYPE_SIZE;
else if (strcmp (SIZETYPE, "long unsigned int") == 0)
precision = LONG_TYPE_SIZE;
else if (strcmp (SIZETYPE, "long long unsigned int") == 0)
precision = LONG_LONG_TYPE_SIZE;
else if (strcmp (SIZETYPE, "short unsigned int") == 0)
precision = SHORT_TYPE_SIZE;
else
{
int i;
precision = -1;
for (i = 0; i < NUM_INT_N_ENTS; i++)
if (int_n_enabled_p[i])
{
char name[50];
sprintf (name, "__int%d unsigned", int_n_data[i].bitsize);
if (strcmp (name, SIZETYPE) == 0)
{
precision = int_n_data[i].bitsize;
}
}
if (precision == -1)
gcc_unreachable ();
}
bprecision
= MIN (precision + LOG2_BITS_PER_UNIT + 1, MAX_FIXED_MODE_SIZE);
bprecision = GET_MODE_PRECISION (smallest_int_mode_for_size (bprecision));
if (bprecision > HOST_BITS_PER_DOUBLE_INT)
bprecision = HOST_BITS_PER_DOUBLE_INT;
sizetype = make_node (INTEGER_TYPE);
TYPE_NAME (sizetype) = get_identifier ("sizetype");
TYPE_PRECISION (sizetype) = precision;
TYPE_UNSIGNED (sizetype) = 1;
bitsizetype = make_node (INTEGER_TYPE);
TYPE_NAME (bitsizetype) = get_identifier ("bitsizetype");
TYPE_PRECISION (bitsizetype) = bprecision;
TYPE_UNSIGNED (bitsizetype) = 1;
scalar_int_mode mode = smallest_int_mode_for_size (precision);
SET_TYPE_MODE (sizetype, mode);
SET_TYPE_ALIGN (sizetype, GET_MODE_ALIGNMENT (TYPE_MODE (sizetype)));
TYPE_SIZE (sizetype) = bitsize_int (precision);
TYPE_SIZE_UNIT (sizetype) = size_int (GET_MODE_SIZE (mode));
set_min_and_max_values_for_integral_type (sizetype, precision, UNSIGNED);
mode = smallest_int_mode_for_size (bprecision);
SET_TYPE_MODE (bitsizetype, mode);
SET_TYPE_ALIGN (bitsizetype, GET_MODE_ALIGNMENT (TYPE_MODE (bitsizetype)));
TYPE_SIZE (bitsizetype) = bitsize_int (bprecision);
TYPE_SIZE_UNIT (bitsizetype) = size_int (GET_MODE_SIZE (mode));
set_min_and_max_values_for_integral_type (bitsizetype, bprecision, UNSIGNED);
ssizetype = make_signed_type (TYPE_PRECISION (sizetype));
TYPE_NAME (ssizetype) = get_identifier ("ssizetype");
sbitsizetype = make_signed_type (TYPE_PRECISION (bitsizetype));
TYPE_NAME (sbitsizetype) = get_identifier ("sbitsizetype");
}

void
set_min_and_max_values_for_integral_type (tree type,
int precision,
signop sgn)
{
if (precision < 1)
return;
TYPE_MIN_VALUE (type)
= wide_int_to_tree (type, wi::min_value (precision, sgn));
TYPE_MAX_VALUE (type)
= wide_int_to_tree (type, wi::max_value (precision, sgn));
}
void
fixup_signed_type (tree type)
{
int precision = TYPE_PRECISION (type);
set_min_and_max_values_for_integral_type (type, precision, SIGNED);
layout_type (type);
}
void
fixup_unsigned_type (tree type)
{
int precision = TYPE_PRECISION (type);
TYPE_UNSIGNED (type) = 1;
set_min_and_max_values_for_integral_type (type, precision, UNSIGNED);
layout_type (type);
}

bit_field_mode_iterator
::bit_field_mode_iterator (HOST_WIDE_INT bitsize, HOST_WIDE_INT bitpos,
poly_int64 bitregion_start,
poly_int64 bitregion_end,
unsigned int align, bool volatilep)
: m_mode (NARROWEST_INT_MODE), m_bitsize (bitsize),
m_bitpos (bitpos), m_bitregion_start (bitregion_start),
m_bitregion_end (bitregion_end), m_align (align),
m_volatilep (volatilep), m_count (0)
{
if (known_eq (m_bitregion_end, 0))
{
unsigned HOST_WIDE_INT units
= MIN (align, MAX (BIGGEST_ALIGNMENT, BITS_PER_WORD));
if (bitsize <= 0)
bitsize = 1;
HOST_WIDE_INT end = bitpos + bitsize + units - 1;
m_bitregion_end = end - end % units - 1;
}
}
bool
bit_field_mode_iterator::next_mode (scalar_int_mode *out_mode)
{
scalar_int_mode mode;
for (; m_mode.exists (&mode); m_mode = GET_MODE_WIDER_MODE (mode))
{
unsigned int unit = GET_MODE_BITSIZE (mode);
if (unit != GET_MODE_PRECISION (mode))
continue;
if (unit > MAX_FIXED_MODE_SIZE)
break;
if (m_count > 0 && unit > BITS_PER_WORD)
break;
unsigned HOST_WIDE_INT substart = (unsigned HOST_WIDE_INT) m_bitpos % unit;
unsigned HOST_WIDE_INT subend = substart + m_bitsize;
if (subend > unit)
continue;
HOST_WIDE_INT start = m_bitpos - substart;
if (maybe_ne (m_bitregion_start, 0)
&& maybe_lt (start, m_bitregion_start))
break;
HOST_WIDE_INT end = start + unit;
if (maybe_gt (end, m_bitregion_end + 1))
break;
if (GET_MODE_ALIGNMENT (mode) > m_align
&& targetm.slow_unaligned_access (mode, m_align))
break;
*out_mode = mode;
m_mode = GET_MODE_WIDER_MODE (mode);
m_count++;
return true;
}
return false;
}
bool
bit_field_mode_iterator::prefer_smaller_modes ()
{
return (m_volatilep
? targetm.narrow_volatile_bitfield ()
: !SLOW_BYTE_ACCESS);
}
bool
get_best_mode (int bitsize, int bitpos,
poly_uint64 bitregion_start, poly_uint64 bitregion_end,
unsigned int align,
unsigned HOST_WIDE_INT largest_mode_bitsize, bool volatilep,
scalar_int_mode *best_mode)
{
bit_field_mode_iterator iter (bitsize, bitpos, bitregion_start,
bitregion_end, align, volatilep);
scalar_int_mode mode;
bool found = false;
while (iter.next_mode (&mode)
&& GET_MODE_ALIGNMENT (mode) <= align
&& GET_MODE_BITSIZE (mode) <= largest_mode_bitsize)
{
*best_mode = mode;
found = true;
if (iter.prefer_smaller_modes ())
break;
}
return found;
}
void
get_mode_bounds (scalar_int_mode mode, int sign,
scalar_int_mode target_mode,
rtx *mmin, rtx *mmax)
{
unsigned size = GET_MODE_PRECISION (mode);
unsigned HOST_WIDE_INT min_val, max_val;
gcc_assert (size <= HOST_BITS_PER_WIDE_INT);
if (mode == BImode)
{
if (STORE_FLAG_VALUE < 0)
{
min_val = STORE_FLAG_VALUE;
max_val = 0;
}
else
{
min_val = 0;
max_val = STORE_FLAG_VALUE;
}
}
else if (sign)
{
min_val = -(HOST_WIDE_INT_1U << (size - 1));
max_val = (HOST_WIDE_INT_1U << (size - 1)) - 1;
}
else
{
min_val = 0;
max_val = (HOST_WIDE_INT_1U << (size - 1) << 1) - 1;
}
*mmin = gen_int_mode (min_val, target_mode);
*mmax = gen_int_mode (max_val, target_mode);
}
#include "gt-stor-layout.h"
