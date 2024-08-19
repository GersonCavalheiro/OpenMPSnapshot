#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "function.h"
#include "tree.h"
#include "stringpool.h"
#include "cgraph.h"
#include "diagnostic.h"
#include "alias.h"
#include "fold-const.h"
#include "stor-layout.h"
#include "attribs.h"
#include "varasm.h"
#include "toplev.h"
#include "output.h"
#include "debug.h"
#include "convert.h"
#include "common/common-target.h"
#include "langhooks.h"
#include "tree-dump.h"
#include "tree-inline.h"
#include "ada.h"
#include "types.h"
#include "atree.h"
#include "nlists.h"
#include "uintp.h"
#include "fe.h"
#include "sinfo.h"
#include "einfo.h"
#include "ada-tree.h"
#include "gigi.h"
int force_global;
int double_float_alignment;
int double_scalar_alignment;
bool fp_arith_may_widen = true;
tree gnat_std_decls[(int) ADT_LAST];
tree gnat_raise_decls[(int) LAST_REASON_CODE + 1];
tree gnat_raise_decls_ext[(int) LAST_REASON_CODE + 1];
static tree handle_const_attribute (tree *, tree, tree, int, bool *);
static tree handle_nothrow_attribute (tree *, tree, tree, int, bool *);
static tree handle_pure_attribute (tree *, tree, tree, int, bool *);
static tree handle_novops_attribute (tree *, tree, tree, int, bool *);
static tree handle_nonnull_attribute (tree *, tree, tree, int, bool *);
static tree handle_sentinel_attribute (tree *, tree, tree, int, bool *);
static tree handle_noreturn_attribute (tree *, tree, tree, int, bool *);
static tree handle_noinline_attribute (tree *, tree, tree, int, bool *);
static tree handle_noclone_attribute (tree *, tree, tree, int, bool *);
static tree handle_leaf_attribute (tree *, tree, tree, int, bool *);
static tree handle_always_inline_attribute (tree *, tree, tree, int, bool *);
static tree handle_malloc_attribute (tree *, tree, tree, int, bool *);
static tree handle_type_generic_attribute (tree *, tree, tree, int, bool *);
static tree handle_vector_size_attribute (tree *, tree, tree, int, bool *);
static tree handle_vector_type_attribute (tree *, tree, tree, int, bool *);
static tree fake_attribute_handler (tree *, tree, tree, int, bool *);
const struct attribute_spec gnat_internal_attribute_table[] =
{
{ "const",        0, 0,  true,  false, false, false,
handle_const_attribute, NULL },
{ "nothrow",      0, 0,  true,  false, false, false,
handle_nothrow_attribute, NULL },
{ "pure",         0, 0,  true,  false, false, false,
handle_pure_attribute, NULL },
{ "no vops",      0, 0,  true,  false, false, false,
handle_novops_attribute, NULL },
{ "nonnull",      0, -1, false, true,  true,  false,
handle_nonnull_attribute, NULL },
{ "sentinel",     0, 1,  false, true,  true,  false,
handle_sentinel_attribute, NULL },
{ "noreturn",     0, 0,  true,  false, false, false,
handle_noreturn_attribute, NULL },
{ "noinline",     0, 0,  true,  false, false, false,
handle_noinline_attribute, NULL },
{ "noclone",      0, 0,  true,  false, false, false,
handle_noclone_attribute, NULL },
{ "leaf",         0, 0,  true,  false, false, false,
handle_leaf_attribute, NULL },
{ "always_inline",0, 0,  true,  false, false, false,
handle_always_inline_attribute, NULL },
{ "malloc",       0, 0,  true,  false, false, false,
handle_malloc_attribute, NULL },
{ "type generic", 0, 0,  false, true, true, false,
handle_type_generic_attribute, NULL },
{ "vector_size",  1, 1,  false, true, false,  false,
handle_vector_size_attribute, NULL },
{ "vector_type",  0, 0,  false, true, false,  false,
handle_vector_type_attribute, NULL },
{ "may_alias",    0, 0, false, true, false, false, NULL, NULL },
{ "format",     3, 3,  false, true,  true,  false, fake_attribute_handler,
NULL },
{ "format_arg", 1, 1,  false, true,  true,  false, fake_attribute_handler,
NULL },
{ NULL,         0, 0, false, false, false, false, NULL, NULL }
};
static GTY((length ("max_gnat_nodes"))) tree *associate_gnat_to_gnu;
#define GET_GNU_TREE(GNAT_ENTITY)	\
associate_gnat_to_gnu[(GNAT_ENTITY) - First_Node_Id]
#define SET_GNU_TREE(GNAT_ENTITY,VAL)	\
associate_gnat_to_gnu[(GNAT_ENTITY) - First_Node_Id] = (VAL)
#define PRESENT_GNU_TREE(GNAT_ENTITY)	\
(associate_gnat_to_gnu[(GNAT_ENTITY) - First_Node_Id] != NULL_TREE)
static GTY((length ("max_gnat_nodes"))) tree *dummy_node_table;
#define GET_DUMMY_NODE(GNAT_ENTITY)	\
dummy_node_table[(GNAT_ENTITY) - First_Node_Id]
#define SET_DUMMY_NODE(GNAT_ENTITY,VAL)	\
dummy_node_table[(GNAT_ENTITY) - First_Node_Id] = (VAL)
#define PRESENT_DUMMY_NODE(GNAT_ENTITY)	\
(dummy_node_table[(GNAT_ENTITY) - First_Node_Id] != NULL_TREE)
static GTY(()) tree signed_and_unsigned_types[2 * MAX_BITS_PER_WORD + 1][2];
static GTY(()) tree float_types[NUM_MACHINE_MODES];
struct GTY((chain_next ("%h.chain"))) gnat_binding_level {
struct gnat_binding_level *chain;
tree block;
tree jmpbuf_decl;
};
static GTY(()) struct gnat_binding_level *current_binding_level;
static GTY((deletable)) struct gnat_binding_level *free_binding_level;
static GTY(()) tree global_context;
static GTY(()) vec<tree, va_gc> *global_decls;
static GTY(()) vec<tree, va_gc> *builtin_decls;
static GTY((deletable)) tree free_block_chain;
struct GTY((for_user)) pad_type_hash
{
hashval_t hash;
tree type;
};
struct pad_type_hasher : ggc_cache_ptr_hash<pad_type_hash>
{
static inline hashval_t hash (pad_type_hash *t) { return t->hash; }
static bool equal (pad_type_hash *a, pad_type_hash *b);
static int
keep_cache_entry (pad_type_hash *&t)
{
return ggc_marked_p (t->type);
}
};
static GTY ((cache)) hash_table<pad_type_hasher> *pad_type_hash_table;
static tree merge_sizes (tree, tree, tree, bool, bool);
static tree fold_bit_position (const_tree);
static tree compute_related_constant (tree, tree);
static tree split_plus (tree, tree *);
static tree float_type_for_precision (int, machine_mode);
static tree convert_to_fat_pointer (tree, tree);
static unsigned int scale_by_factor_of (tree, unsigned int);
static bool potential_alignment_gap (tree, tree, tree);
struct deferred_decl_context_node
{
tree decl;
Entity_Id gnat_scope;
int force_global;
vec<tree> types;
struct deferred_decl_context_node *next;
};
static struct deferred_decl_context_node *deferred_decl_context_queue = NULL;
static struct deferred_decl_context_node *
add_deferred_decl_context (tree decl, Entity_Id gnat_scope, int force_global);
static void add_deferred_type_context (struct deferred_decl_context_node *n,
tree type);

void
init_gnat_utils (void)
{
associate_gnat_to_gnu = ggc_cleared_vec_alloc<tree> (max_gnat_nodes);
dummy_node_table = ggc_cleared_vec_alloc<tree> (max_gnat_nodes);
pad_type_hash_table = hash_table<pad_type_hasher>::create_ggc (512);
}
void
destroy_gnat_utils (void)
{
ggc_free (associate_gnat_to_gnu);
associate_gnat_to_gnu = NULL;
ggc_free (dummy_node_table);
dummy_node_table = NULL;
pad_type_hash_table->empty ();
pad_type_hash_table = NULL;
}

void
save_gnu_tree (Entity_Id gnat_entity, tree gnu_decl, bool no_check)
{
gcc_assert (!(gnu_decl
&& (PRESENT_GNU_TREE (gnat_entity)
|| (!no_check && !DECL_P (gnu_decl)))));
SET_GNU_TREE (gnat_entity, gnu_decl);
}
tree
get_gnu_tree (Entity_Id gnat_entity)
{
gcc_assert (PRESENT_GNU_TREE (gnat_entity));
return GET_GNU_TREE (gnat_entity);
}
bool
present_gnu_tree (Entity_Id gnat_entity)
{
return PRESENT_GNU_TREE (gnat_entity);
}

tree
make_dummy_type (Entity_Id gnat_type)
{
Entity_Id gnat_equiv = Gigi_Equivalent_Type (Underlying_Type (gnat_type));
tree gnu_type, debug_type;
if (No (gnat_equiv))
gnat_equiv = gnat_type;
if (PRESENT_DUMMY_NODE (gnat_equiv))
return GET_DUMMY_NODE (gnat_equiv);
gnu_type = make_node (Is_Record_Type (gnat_equiv)
? tree_code_for_record_type (gnat_equiv)
: ENUMERAL_TYPE);
TYPE_NAME (gnu_type) = get_entity_name (gnat_type);
TYPE_DUMMY_P (gnu_type) = 1;
TYPE_STUB_DECL (gnu_type)
= create_type_stub_decl (TYPE_NAME (gnu_type), gnu_type);
if (Is_By_Reference_Type (gnat_equiv))
TYPE_BY_REFERENCE_P (gnu_type) = 1;
SET_DUMMY_NODE (gnat_equiv, gnu_type);
if (Needs_Debug_Info (gnat_type))
{
debug_type = make_node (LANG_TYPE);
TYPE_NAME (debug_type) = TYPE_NAME (gnu_type);
TYPE_ARTIFICIAL (debug_type) = TYPE_ARTIFICIAL (gnu_type);
SET_TYPE_DEBUG_TYPE (gnu_type, debug_type);
}
return gnu_type;
}
tree
get_dummy_type (Entity_Id gnat_type)
{
return GET_DUMMY_NODE (gnat_type);
}
void
build_dummy_unc_pointer_types (Entity_Id gnat_desig_type, tree gnu_desig_type)
{
tree gnu_template_type, gnu_ptr_template, gnu_array_type, gnu_ptr_array;
tree gnu_fat_type, fields, gnu_object_type;
gnu_template_type = make_node (RECORD_TYPE);
TYPE_NAME (gnu_template_type) = create_concat_name (gnat_desig_type, "XUB");
TYPE_DUMMY_P (gnu_template_type) = 1;
gnu_ptr_template = build_pointer_type (gnu_template_type);
gnu_array_type = make_node (ENUMERAL_TYPE);
TYPE_NAME (gnu_array_type) = create_concat_name (gnat_desig_type, "XUA");
TYPE_DUMMY_P (gnu_array_type) = 1;
gnu_ptr_array = build_pointer_type (gnu_array_type);
gnu_fat_type = make_node (RECORD_TYPE);
TYPE_NAME (gnu_fat_type)
= create_type_stub_decl (create_concat_name (gnat_desig_type, "XUP"),
gnu_fat_type);
fields = create_field_decl (get_identifier ("P_ARRAY"), gnu_ptr_array,
gnu_fat_type, NULL_TREE, NULL_TREE, 0, 0);
DECL_CHAIN (fields)
= create_field_decl (get_identifier ("P_BOUNDS"), gnu_ptr_template,
gnu_fat_type, NULL_TREE, NULL_TREE, 0, 0);
finish_fat_pointer_type (gnu_fat_type, fields);
SET_TYPE_UNCONSTRAINED_ARRAY (gnu_fat_type, gnu_desig_type);
TYPE_DECL_SUPPRESS_DEBUG (TYPE_STUB_DECL (gnu_fat_type)) = 1;
gnu_object_type = make_node (RECORD_TYPE);
TYPE_NAME (gnu_object_type) = create_concat_name (gnat_desig_type, "XUT");
TYPE_DUMMY_P (gnu_object_type) = 1;
TYPE_POINTER_TO (gnu_desig_type) = gnu_fat_type;
TYPE_REFERENCE_TO (gnu_desig_type) = gnu_fat_type;
TYPE_OBJECT_RECORD_TYPE (gnu_desig_type) = gnu_object_type;
}

bool
global_bindings_p (void)
{
return force_global || !current_function_decl;
}
void
gnat_pushlevel (void)
{
struct gnat_binding_level *newlevel = NULL;
if (free_binding_level)
{
newlevel = free_binding_level;
free_binding_level = free_binding_level->chain;
}
else
newlevel = ggc_alloc<gnat_binding_level> ();
if (free_block_chain)
{
newlevel->block = free_block_chain;
free_block_chain = BLOCK_CHAIN (free_block_chain);
BLOCK_CHAIN (newlevel->block) = NULL_TREE;
}
else
newlevel->block = make_node (BLOCK);
if (current_binding_level)
BLOCK_SUPERCONTEXT (newlevel->block) = current_binding_level->block;
BLOCK_VARS (newlevel->block) = NULL_TREE;
BLOCK_SUBBLOCKS (newlevel->block) = NULL_TREE;
TREE_USED (newlevel->block) = 1;
newlevel->chain = current_binding_level;
newlevel->jmpbuf_decl = NULL_TREE;
current_binding_level = newlevel;
}
void
set_current_block_context (tree fndecl)
{
BLOCK_SUPERCONTEXT (current_binding_level->block) = fndecl;
DECL_INITIAL (fndecl) = current_binding_level->block;
set_block_for_group (current_binding_level->block);
}
void
set_block_jmpbuf_decl (tree decl)
{
current_binding_level->jmpbuf_decl = decl;
}
tree
get_block_jmpbuf_decl (void)
{
return current_binding_level->jmpbuf_decl;
}
void
gnat_poplevel (void)
{
struct gnat_binding_level *level = current_binding_level;
tree block = level->block;
BLOCK_VARS (block) = nreverse (BLOCK_VARS (block));
BLOCK_SUBBLOCKS (block) = blocks_nreverse (BLOCK_SUBBLOCKS (block));
if (TREE_CODE (BLOCK_SUPERCONTEXT (block)) == FUNCTION_DECL)
;
else if (!BLOCK_VARS (block))
{
BLOCK_SUBBLOCKS (level->chain->block)
= block_chainon (BLOCK_SUBBLOCKS (block),
BLOCK_SUBBLOCKS (level->chain->block));
BLOCK_CHAIN (block) = free_block_chain;
free_block_chain = block;
}
else
{
BLOCK_CHAIN (block) = BLOCK_SUBBLOCKS (level->chain->block);
BLOCK_SUBBLOCKS (level->chain->block) = block;
TREE_USED (block) = 1;
set_block_for_group (block);
}
current_binding_level = level->chain;
level->chain = free_binding_level;
free_binding_level = level;
}
void
gnat_zaplevel (void)
{
struct gnat_binding_level *level = current_binding_level;
tree block = level->block;
BLOCK_CHAIN (block) = free_block_chain;
free_block_chain = block;
current_binding_level = level->chain;
level->chain = free_binding_level;
free_binding_level = level;
}

static void
gnat_set_type_context (tree type, tree context)
{
tree decl = TYPE_STUB_DECL (type);
TYPE_CONTEXT (type) = context;
while (decl && DECL_PARALLEL_TYPE (decl))
{
tree parallel_type = DECL_PARALLEL_TYPE (decl);
if (!TYPE_CONTEXT (parallel_type))
{
if (TYPE_STUB_DECL (parallel_type))
DECL_CONTEXT (TYPE_STUB_DECL (parallel_type)) = context;
TYPE_CONTEXT (parallel_type) = context;
}
decl = TYPE_STUB_DECL (DECL_PARALLEL_TYPE (decl));
}
}
Entity_Id
get_debug_scope (Node_Id gnat_node, bool *is_subprogram)
{
Entity_Id gnat_entity;
if (is_subprogram)
*is_subprogram = false;
if (Nkind (gnat_node) == N_Defining_Identifier
|| Nkind (gnat_node) == N_Defining_Operator_Symbol)
gnat_entity = Scope (gnat_node);
else
return Empty;
while (Present (gnat_entity))
{
switch (Ekind (gnat_entity))
{
case E_Function:
case E_Procedure:
if (Present (Protected_Body_Subprogram (gnat_entity)))
gnat_entity = Protected_Body_Subprogram (gnat_entity);
if (is_subprogram)
*is_subprogram = true;
return gnat_entity;
case E_Record_Type:
case E_Record_Subtype:
return gnat_entity;
default:
break;
}
gnat_entity = Scope (gnat_entity);
}
return Empty;
}
static void
defer_or_set_type_context (tree type, tree context,
struct deferred_decl_context_node *n)
{
if (n)
add_deferred_type_context (n, type);
else
gnat_set_type_context (type, context);
}
static tree
get_global_context (void)
{
if (!global_context)
{
global_context
= build_translation_unit_decl (get_identifier (main_input_filename));
debug_hooks->register_main_translation_unit (global_context);
}
return global_context;
}
void
gnat_pushdecl (tree decl, Node_Id gnat_node)
{
tree context = NULL_TREE;
struct deferred_decl_context_node *deferred_decl_context = NULL;
if (!((TREE_PUBLIC (decl) && DECL_EXTERNAL (decl)) || force_global == 1))
{
bool context_is_subprogram = false;
const Entity_Id gnat_scope
= get_debug_scope (gnat_node, &context_is_subprogram);
if (Present (gnat_scope)
&& !context_is_subprogram
&& TREE_CODE (decl) != FUNCTION_DECL
&& TREE_CODE (decl) != VAR_DECL)
deferred_decl_context
= add_deferred_decl_context (decl, gnat_scope, force_global);
else if (current_function_decl && force_global == 0)
context = current_function_decl;
}
if (!deferred_decl_context && !context)
context = get_global_context ();
if (TREE_CODE (decl) == FUNCTION_DECL
&& !TREE_PUBLIC (decl)
&& context
&& (TREE_CODE (context) == FUNCTION_DECL
|| decl_function_context (context)))
DECL_STATIC_CHAIN (decl) = 1;
if (!deferred_decl_context)
DECL_CONTEXT (decl) = context;
TREE_NO_WARNING (decl) = (No (gnat_node) || Warnings_Off (gnat_node));
if (Present (gnat_node) && !renaming_from_instantiation_p (gnat_node))
Sloc_to_locus (Sloc (gnat_node), &DECL_SOURCE_LOCATION (decl));
add_decl_expr (decl, gnat_node);
if (!(TREE_CODE (decl) == TYPE_DECL
&& TREE_CODE (TREE_TYPE (decl)) == UNCONSTRAINED_ARRAY_TYPE))
{
if (DECL_EXTERNAL (decl)
&& TREE_CODE (decl) == FUNCTION_DECL
&& DECL_BUILT_IN (decl))
vec_safe_push (builtin_decls, decl);
else if (global_bindings_p ())
vec_safe_push (global_decls, decl);
else
{
DECL_CHAIN (decl) = BLOCK_VARS (current_binding_level->block);
BLOCK_VARS (current_binding_level->block) = decl;
}
}
if (TREE_CODE (decl) == TYPE_DECL && DECL_NAME (decl))
{
tree t = TREE_TYPE (decl);
if (!(TYPE_NAME (t) && TREE_CODE (TYPE_NAME (t)) == TYPE_DECL)
&& ((TREE_CODE (t) != ARRAY_TYPE && TREE_CODE (t) != POINTER_TYPE)
|| DECL_ARTIFICIAL (decl)))
;
else if (!DECL_ARTIFICIAL (decl)
&& (TREE_CODE (t) == ARRAY_TYPE
|| TREE_CODE (t) == POINTER_TYPE
|| TYPE_IS_FAT_POINTER_P (t)))
{
tree tt = build_variant_type_copy (t);
TYPE_NAME (tt) = decl;
defer_or_set_type_context (tt,
DECL_CONTEXT (decl),
deferred_decl_context);
TREE_TYPE (decl) = tt;
if (TYPE_NAME (t)
&& TREE_CODE (TYPE_NAME (t)) == TYPE_DECL
&& DECL_ORIGINAL_TYPE (TYPE_NAME (t)))
DECL_ORIGINAL_TYPE (decl) = DECL_ORIGINAL_TYPE (TYPE_NAME (t));
else
DECL_ORIGINAL_TYPE (decl) = t;
if (TREE_CODE (t) == ARRAY_TYPE && !TYPE_NAME (t))
TYPE_NAME (t) = DECL_NAME (decl);
t = NULL_TREE;
}
else if (TYPE_NAME (t)
&& TREE_CODE (TYPE_NAME (t)) == TYPE_DECL
&& DECL_ARTIFICIAL (TYPE_NAME (t)) && !DECL_ARTIFICIAL (decl))
;
else
t = NULL_TREE;
if (t)
for (t = TYPE_MAIN_VARIANT (t); t; t = TYPE_NEXT_VARIANT (t))
if (!(TYPE_IS_FAT_POINTER_P (t)
&& TYPE_NAME (t)
&& TREE_CODE (TYPE_NAME (t)) == TYPE_DECL))
{
TYPE_NAME (t) = decl;
defer_or_set_type_context (t,
DECL_CONTEXT (decl),
deferred_decl_context);
}
}
}

tree
make_aligning_type (tree type, unsigned int align, tree size,
unsigned int base_align, int room, Node_Id gnat_node)
{
tree record_type = make_node (RECORD_TYPE);
tree record = build0 (PLACEHOLDER_EXPR, record_type);
tree record_addr_st
= convert (sizetype, build_unary_op (ADDR_EXPR, NULL_TREE, record));
tree room_st = size_int (room);
tree vblock_addr_st = size_binop (PLUS_EXPR, record_addr_st, room_st);
tree voffset_st, pos, field;
tree name = TYPE_IDENTIFIER (type);
name = concat_name (name, "ALIGN");
TYPE_NAME (record_type) = name;
voffset_st = size_binop (BIT_AND_EXPR,
fold_build1 (NEGATE_EXPR, sizetype, vblock_addr_st),
size_int ((align / BITS_PER_UNIT) - 1));
pos = size_binop (MULT_EXPR,
convert (bitsizetype,
size_binop (PLUS_EXPR, room_st, voffset_st)),
bitsize_unit_node);
field = create_field_decl (get_identifier ("F"), type, record_type, size,
pos, 1, -1);
TYPE_FIELDS (record_type) = field;
SET_TYPE_ALIGN (record_type, base_align);
TYPE_USER_ALIGN (record_type) = 1;
TYPE_SIZE (record_type)
= size_binop (PLUS_EXPR,
size_binop (MULT_EXPR, convert (bitsizetype, size),
bitsize_unit_node),
bitsize_int (align + room * BITS_PER_UNIT));
TYPE_SIZE_UNIT (record_type)
= size_binop (PLUS_EXPR, size,
size_int (room + align / BITS_PER_UNIT));
SET_TYPE_MODE (record_type, BLKmode);
relate_alias_sets (record_type, type, ALIAS_SET_COPY);
create_type_decl (name, record_type, true, false, gnat_node);
return record_type;
}
tree
make_packable_type (tree type, bool in_record, unsigned int max_align)
{
unsigned HOST_WIDE_INT size = tree_to_uhwi (TYPE_SIZE (type));
unsigned HOST_WIDE_INT new_size;
unsigned int align = TYPE_ALIGN (type);
unsigned int new_align;
if (size == 0)
return type;
tree new_type = make_node (TREE_CODE (type));
TYPE_NAME (new_type) = TYPE_NAME (type);
TYPE_JUSTIFIED_MODULAR_P (new_type) = TYPE_JUSTIFIED_MODULAR_P (type);
TYPE_CONTAINS_TEMPLATE_P (new_type) = TYPE_CONTAINS_TEMPLATE_P (type);
TYPE_REVERSE_STORAGE_ORDER (new_type) = TYPE_REVERSE_STORAGE_ORDER (type);
if (TREE_CODE (type) == RECORD_TYPE)
TYPE_PADDING_P (new_type) = TYPE_PADDING_P (type);
if (in_record && size <= MAX_FIXED_MODE_SIZE)
{
new_size = ceil_pow2 (size);
new_align = MIN (new_size, BIGGEST_ALIGNMENT);
SET_TYPE_ALIGN (new_type, new_align);
}
else
{
if (TYPE_CONTAINS_TEMPLATE_P (type)
|| !tree_fits_uhwi_p (TYPE_ADA_SIZE (type)))
return type;
new_size = tree_to_uhwi (TYPE_ADA_SIZE (type));
new_size = (new_size + BITS_PER_UNIT - 1) & -BITS_PER_UNIT;
if (new_size == size && (max_align == 0 || align <= max_align))
return type;
new_align = MIN (new_size & -new_size, BIGGEST_ALIGNMENT);
if (max_align > 0 && new_align > max_align)
new_align = max_align;
SET_TYPE_ALIGN (new_type, MIN (align, new_align));
}
TYPE_USER_ALIGN (new_type) = 1;
tree new_field_list = NULL_TREE;
for (tree field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
{
tree new_field_type = TREE_TYPE (field);
tree new_field, new_size;
if (RECORD_OR_UNION_TYPE_P (new_field_type)
&& !TYPE_FAT_POINTER_P (new_field_type)
&& tree_fits_uhwi_p (TYPE_SIZE (new_field_type)))
new_field_type = make_packable_type (new_field_type, true, max_align);
if (!DECL_CHAIN (field)
&& !TYPE_PACKED (type)
&& RECORD_OR_UNION_TYPE_P (new_field_type)
&& !TYPE_FAT_POINTER_P (new_field_type)
&& !TYPE_CONTAINS_TEMPLATE_P (new_field_type)
&& TYPE_ADA_SIZE (new_field_type))
new_size = TYPE_ADA_SIZE (new_field_type);
else
new_size = DECL_SIZE (field);
new_field
= create_field_decl (DECL_NAME (field), new_field_type, new_type,
new_size, bit_position (field),
TYPE_PACKED (type),
!DECL_NONADDRESSABLE_P (field));
DECL_INTERNAL_P (new_field) = DECL_INTERNAL_P (field);
SET_DECL_ORIGINAL_FIELD_TO_FIELD (new_field, field);
if (TREE_CODE (new_type) == QUAL_UNION_TYPE)
DECL_QUALIFIER (new_field) = DECL_QUALIFIER (field);
DECL_CHAIN (new_field) = new_field_list;
new_field_list = new_field;
}
finish_record_type (new_type, nreverse (new_field_list), 2, false);
relate_alias_sets (new_type, type, ALIAS_SET_COPY);
if (gnat_encodings == DWARF_GNAT_ENCODINGS_MINIMAL)
SET_TYPE_DEBUG_TYPE (new_type, TYPE_DEBUG_TYPE (type));
else if (TYPE_STUB_DECL (type))
SET_DECL_PARALLEL_TYPE (TYPE_STUB_DECL (new_type),
DECL_PARALLEL_TYPE (TYPE_STUB_DECL (type)));
if (TYPE_IS_PADDING_P (type) || TREE_CODE (type) == QUAL_UNION_TYPE)
{
TYPE_SIZE (new_type) = TYPE_SIZE (type);
TYPE_SIZE_UNIT (new_type) = TYPE_SIZE_UNIT (type);
new_size = size;
}
else
{
TYPE_SIZE (new_type) = bitsize_int (new_size);
TYPE_SIZE_UNIT (new_type) = size_int (new_size / BITS_PER_UNIT);
}
if (!TYPE_CONTAINS_TEMPLATE_P (type))
SET_TYPE_ADA_SIZE (new_type, TYPE_ADA_SIZE (type));
compute_record_mode (new_type);
if (in_record && TYPE_MODE (new_type) == BLKmode)
SET_TYPE_MODE (new_type,
mode_for_size_tree (TYPE_SIZE (new_type),
MODE_INT, 1).else_blk ());
if (TYPE_MODE (new_type) == BLKmode && new_size >= size && max_align == 0)
return type;
return new_type;
}
static inline bool
type_unsigned_for_rm (tree type)
{
if (TYPE_UNSIGNED (type))
return true;
if (TREE_CODE (TYPE_MIN_VALUE (type)) == INTEGER_CST
&& tree_int_cst_sgn (TYPE_MIN_VALUE (type)) >= 0)
return true;
return false;
}
tree
make_type_from_size (tree type, tree size_tree, bool for_biased)
{
unsigned HOST_WIDE_INT size;
bool biased_p;
tree new_type;
if (!size_tree || !tree_fits_uhwi_p (size_tree))
return type;
size = tree_to_uhwi (size_tree);
switch (TREE_CODE (type))
{
case INTEGER_TYPE:
case ENUMERAL_TYPE:
case BOOLEAN_TYPE:
biased_p = (TREE_CODE (type) == INTEGER_TYPE
&& TYPE_BIASED_REPRESENTATION_P (type));
if (size == 0)
size = 1;
if (TYPE_IS_PACKED_ARRAY_TYPE_P (type)
|| (TYPE_PRECISION (type) == size && biased_p == for_biased)
|| size > LONG_LONG_TYPE_SIZE)
break;
biased_p |= for_biased;
if (type_unsigned_for_rm (type) || biased_p)
new_type = make_unsigned_type (size);
else
new_type = make_signed_type (size);
TREE_TYPE (new_type) = TREE_TYPE (type) ? TREE_TYPE (type) : type;
SET_TYPE_RM_MIN_VALUE (new_type, TYPE_MIN_VALUE (type));
SET_TYPE_RM_MAX_VALUE (new_type, TYPE_MAX_VALUE (type));
TYPE_NAME (new_type) = TYPE_NAME (type);
TYPE_BIASED_REPRESENTATION_P (new_type) = biased_p;
SET_TYPE_RM_SIZE (new_type, bitsize_int (size));
return new_type;
case RECORD_TYPE:
if (TYPE_FAT_POINTER_P (type) && size < POINTER_SIZE * 2)
{
scalar_int_mode p_mode;
if (!int_mode_for_size (size, 0).exists (&p_mode)
|| !targetm.valid_pointer_mode (p_mode))
p_mode = ptr_mode;
return
build_pointer_type_for_mode
(TYPE_OBJECT_RECORD_TYPE (TYPE_UNCONSTRAINED_ARRAY (type)),
p_mode, 0);
}
break;
case POINTER_TYPE:
if (TYPE_IS_THIN_POINTER_P (type) && size >= POINTER_SIZE * 2)
return
build_pointer_type (TYPE_UNCONSTRAINED_ARRAY (TREE_TYPE (type)));
break;
default:
break;
}
return type;
}
bool
pad_type_hasher::equal (pad_type_hash *t1, pad_type_hash *t2)
{
tree type1, type2;
if (t1->hash != t2->hash)
return 0;
type1 = t1->type;
type2 = t2->type;
return
TREE_TYPE (TYPE_FIELDS (type1)) == TREE_TYPE (TYPE_FIELDS (type2))
&& TYPE_SIZE (type1) == TYPE_SIZE (type2)
&& TYPE_ALIGN (type1) == TYPE_ALIGN (type2)
&& TYPE_ADA_SIZE (type1) == TYPE_ADA_SIZE (type2)
&& TYPE_REVERSE_STORAGE_ORDER (type1) == TYPE_REVERSE_STORAGE_ORDER (type2);
}
static hashval_t
hash_pad_type (tree type)
{
hashval_t hashcode;
hashcode
= iterative_hash_object (TYPE_HASH (TREE_TYPE (TYPE_FIELDS (type))), 0);
hashcode = iterative_hash_expr (TYPE_SIZE (type), hashcode);
hashcode = iterative_hash_hashval_t (TYPE_ALIGN (type), hashcode);
hashcode = iterative_hash_expr (TYPE_ADA_SIZE (type), hashcode);
return hashcode;
}
static tree
canonicalize_pad_type (tree type)
{
const hashval_t hashcode = hash_pad_type (type);
struct pad_type_hash in, *h, **slot;
in.hash = hashcode;
in.type = type;
slot = pad_type_hash_table->find_slot_with_hash (&in, hashcode, INSERT);
h = *slot;
if (!h)
{
h = ggc_alloc<pad_type_hash> ();
h->hash = hashcode;
h->type = type;
*slot = h;
}
return h->type;
}
tree
maybe_pad_type (tree type, tree size, unsigned int align,
Entity_Id gnat_entity, bool is_component_type,
bool is_user_type, bool definition, bool set_rm_size)
{
tree orig_size = TYPE_SIZE (type);
unsigned int orig_align = TYPE_ALIGN (type);
tree record, field;
if (TYPE_IS_PADDING_P (type))
{
if ((!size
|| operand_equal_p (round_up (size, orig_align), orig_size, 0))
&& (align == 0 || align == orig_align))
return type;
if (!size)
size = orig_size;
if (align == 0)
align = orig_align;
type = TREE_TYPE (TYPE_FIELDS (type));
orig_size = TYPE_SIZE (type);
orig_align = TYPE_ALIGN (type);
}
if (size
&& (operand_equal_p (size, orig_size, 0)
|| (TREE_CODE (orig_size) == INTEGER_CST
&& tree_int_cst_lt (size, orig_size))))
size = NULL_TREE;
if (align == orig_align)
align = 0;
if (align == 0 && !size)
return type;
if (is_user_type)
create_type_decl (get_entity_name (gnat_entity), type,
!Comes_From_Source (gnat_entity),
!(TYPE_NAME (type)
&& TREE_CODE (TYPE_NAME (type)) == TYPE_DECL
&& DECL_IGNORED_P (TYPE_NAME (type))),
gnat_entity);
record = make_node (RECORD_TYPE);
TYPE_PADDING_P (record) = 1;
if (TYPE_IMPL_PACKED_ARRAY_P (type)
&& TYPE_ORIGINAL_PACKED_ARRAY (type)
&& gnat_encodings == DWARF_GNAT_ENCODINGS_MINIMAL)
TYPE_NAME (record) = TYPE_NAME (TYPE_ORIGINAL_PACKED_ARRAY (type));
else if (Present (gnat_entity))
TYPE_NAME (record) = create_concat_name (gnat_entity, "PAD");
SET_TYPE_ALIGN (record, align ? align : orig_align);
TYPE_SIZE (record) = size ? size : orig_size;
TYPE_SIZE_UNIT (record)
= convert (sizetype,
size_binop (CEIL_DIV_EXPR, TYPE_SIZE (record),
bitsize_unit_node));
if (align != 0
&& RECORD_OR_UNION_TYPE_P (type)
&& TYPE_MODE (type) == BLKmode
&& !TYPE_BY_REFERENCE_P (type)
&& TREE_CODE (orig_size) == INTEGER_CST
&& !TREE_OVERFLOW (orig_size)
&& compare_tree_int (orig_size, MAX_FIXED_MODE_SIZE) <= 0
&& (!size
|| (TREE_CODE (size) == INTEGER_CST
&& compare_tree_int (size, MAX_FIXED_MODE_SIZE) <= 0)))
{
tree packable_type = make_packable_type (type, true);
if (TYPE_MODE (packable_type) != BLKmode
&& align >= TYPE_ALIGN (packable_type))
type = packable_type;
}
field = create_field_decl (get_identifier ("F"), type, record, orig_size,
bitsize_zero_node, 0, 1);
DECL_INTERNAL_P (field) = 1;
finish_record_type (record, field, 1, false);
if (set_rm_size)
{
SET_TYPE_ADA_SIZE (record, size ? size : orig_size);
if (TREE_CONSTANT (TYPE_SIZE (record)))
{
tree canonical = canonicalize_pad_type (record);
if (canonical != record)
{
record = canonical;
goto built;
}
}
}
if (gnat_encodings == DWARF_GNAT_ENCODINGS_MINIMAL)
SET_TYPE_DEBUG_TYPE (record, maybe_debug_type (type));
if (TREE_CODE (orig_size) != INTEGER_CST
&& TYPE_NAME (record)
&& TYPE_NAME (type)
&& !(TREE_CODE (TYPE_NAME (type)) == TYPE_DECL
&& DECL_IGNORED_P (TYPE_NAME (type))))
{
tree name = TYPE_IDENTIFIER (record);
tree size_unit = TYPE_SIZE_UNIT (record);
if (size
&& TREE_CODE (size) != INTEGER_CST
&& (definition || global_bindings_p ()))
{
size_unit
= create_var_decl (concat_name (name, "XVZ"), NULL_TREE, sizetype,
size_unit, true, global_bindings_p (),
!definition && global_bindings_p (), false,
false, true, true, NULL, gnat_entity);
TYPE_SIZE_UNIT (record) = size_unit;
}
if (gnat_encodings != DWARF_GNAT_ENCODINGS_MINIMAL)
{
tree marker = make_node (RECORD_TYPE);
tree orig_name = TYPE_IDENTIFIER (type);
TYPE_NAME (marker) = concat_name (name, "XVS");
finish_record_type (marker,
create_field_decl (orig_name,
build_reference_type (type),
marker, NULL_TREE, NULL_TREE,
0, 0),
0, true);
TYPE_SIZE_UNIT (marker) = size_unit;
add_parallel_type (record, marker);
}
}
built:
if (!size
|| TREE_CODE (size) == COND_EXPR
|| TREE_CODE (size) == MAX_EXPR
|| No (gnat_entity))
return record;
if (type_annotate_only)
{
Entity_Id gnat_type
= is_component_type
? Component_Type (gnat_entity) : Etype (gnat_entity);
if (Is_Tagged_Type (gnat_type) || Is_Concurrent_Type (gnat_type))
return record;
}
if (CONTAINS_PLACEHOLDER_P (orig_size))
orig_size = max_size (orig_size, true);
if (align && AGGREGATE_TYPE_P (type))
orig_size = round_up (orig_size, align);
if (!operand_equal_p (size, orig_size, 0)
&& !(TREE_CODE (size) == INTEGER_CST
&& TREE_CODE (orig_size) == INTEGER_CST
&& (TREE_OVERFLOW (size)
|| TREE_OVERFLOW (orig_size)
|| tree_int_cst_lt (size, orig_size))))
{
Node_Id gnat_error_node = Empty;
if (Is_Packed_Array_Impl_Type (gnat_entity))
gnat_entity = Original_Array_Type (gnat_entity);
if ((Ekind (gnat_entity) == E_Component
|| Ekind (gnat_entity) == E_Discriminant)
&& Present (Component_Clause (gnat_entity)))
gnat_error_node = Last_Bit (Component_Clause (gnat_entity));
else if (Present (Size_Clause (gnat_entity)))
gnat_error_node = Expression (Size_Clause (gnat_entity));
if (Comes_From_Source (gnat_entity))
{
if (Present (gnat_error_node))
post_error_ne_tree ("{^ }bits of & unused?",
gnat_error_node, gnat_entity,
size_diffop (size, orig_size));
else if (is_component_type)
post_error_ne_tree ("component of& padded{ by ^ bits}?",
gnat_entity, gnat_entity,
size_diffop (size, orig_size));
}
}
return record;
}
bool
pad_type_has_rm_size (tree type)
{
if (!TREE_CONSTANT (TYPE_SIZE (type)))
return false;
const hashval_t hashcode = hash_pad_type (type);
struct pad_type_hash in, *h;
in.hash = hashcode;
in.type = type;
h = pad_type_hash_table->find_with_hash (&in, hashcode);
return h && h->type == type;
}
tree
set_reverse_storage_order_on_pad_type (tree type)
{
if (flag_checking)
{
tree inner_type = TREE_TYPE (TYPE_FIELDS (type));
gcc_assert (!AGGREGATE_TYPE_P (inner_type)
&& !VECTOR_TYPE_P (inner_type));
}
gcc_assert (TREE_CONSTANT (TYPE_SIZE (type)));
tree field = copy_node (TYPE_FIELDS (type));
type = copy_type (type);
DECL_CONTEXT (field) = type;
TYPE_FIELDS (type) = field;
TYPE_REVERSE_STORAGE_ORDER (type) = 1;
return canonicalize_pad_type (type);
}

void
relate_alias_sets (tree gnu_new_type, tree gnu_old_type, enum alias_set_op op)
{
while (TREE_CODE (gnu_old_type) == RECORD_TYPE
&& (TYPE_JUSTIFIED_MODULAR_P (gnu_old_type)
|| TYPE_PADDING_P (gnu_old_type)))
gnu_old_type = TREE_TYPE (TYPE_FIELDS (gnu_old_type));
if (TREE_CODE (gnu_old_type) == UNCONSTRAINED_ARRAY_TYPE)
gnu_old_type
= TREE_TYPE (TREE_TYPE (TYPE_FIELDS (TREE_TYPE (gnu_old_type))));
if (TREE_CODE (gnu_new_type) == UNCONSTRAINED_ARRAY_TYPE)
gnu_new_type
= TREE_TYPE (TREE_TYPE (TYPE_FIELDS (TREE_TYPE (gnu_new_type))));
if (TREE_CODE (gnu_new_type) == ARRAY_TYPE
&& TREE_CODE (TREE_TYPE (gnu_new_type)) == ARRAY_TYPE
&& TYPE_MULTI_ARRAY_P (TREE_TYPE (gnu_new_type)))
relate_alias_sets (TREE_TYPE (gnu_new_type), TREE_TYPE (gnu_old_type), op);
switch (op)
{
case ALIAS_SET_COPY:
if (flag_checking || flag_strict_aliasing)
gcc_assert (!(TREE_CODE (gnu_new_type) == ARRAY_TYPE
&& TREE_CODE (gnu_old_type) == ARRAY_TYPE
&& TYPE_NONALIASED_COMPONENT (gnu_new_type)
!= TYPE_NONALIASED_COMPONENT (gnu_old_type)));
TYPE_ALIAS_SET (gnu_new_type) = get_alias_set (gnu_old_type);
break;
case ALIAS_SET_SUBSET:
case ALIAS_SET_SUPERSET:
{
alias_set_type old_set = get_alias_set (gnu_old_type);
alias_set_type new_set = get_alias_set (gnu_new_type);
if (!alias_sets_conflict_p (old_set, new_set))
{
if (op == ALIAS_SET_SUBSET)
record_alias_subset (old_set, new_set);
else
record_alias_subset (new_set, old_set);
}
}
break;
default:
gcc_unreachable ();
}
record_component_aliases (gnu_new_type);
}

void
record_builtin_type (const char *name, tree type, bool artificial_p)
{
tree type_decl = build_decl (input_location,
TYPE_DECL, get_identifier (name), type);
DECL_ARTIFICIAL (type_decl) = artificial_p;
TYPE_ARTIFICIAL (type) = artificial_p;
gnat_pushdecl (type_decl, Empty);
if (debug_hooks->type_decl)
debug_hooks->type_decl (type_decl, false);
}

void
finish_character_type (tree char_type)
{
if (TYPE_UNSIGNED (char_type))
return;
tree unsigned_char_type
= (char_type == char_type_node
? unsigned_char_type_node
: copy_type (gnat_unsigned_type_for (char_type)));
TYPE_NAME (unsigned_char_type) = TYPE_NAME (char_type);
TYPE_STRING_FLAG (unsigned_char_type) = TYPE_STRING_FLAG (char_type);
TYPE_ARTIFICIAL (unsigned_char_type) = TYPE_ARTIFICIAL (char_type);
SET_TYPE_DEBUG_TYPE (char_type, unsigned_char_type);
if (TREE_TYPE (char_type))
{
tree base_unsigned_char_type = TYPE_DEBUG_TYPE (TREE_TYPE (char_type));
tree min_value = TYPE_RM_MIN_VALUE (char_type);
tree max_value = TYPE_RM_MAX_VALUE (char_type);
if (TREE_CODE (min_value) == INTEGER_CST)
min_value = fold_convert (base_unsigned_char_type, min_value);
if (TREE_CODE (max_value) == INTEGER_CST)
max_value = fold_convert (base_unsigned_char_type, max_value);
TREE_TYPE (unsigned_char_type) = base_unsigned_char_type;
SET_TYPE_RM_MIN_VALUE (unsigned_char_type, min_value);
SET_TYPE_RM_MAX_VALUE (unsigned_char_type, max_value);
}
SET_TYPE_RM_MIN_VALUE (char_type, TYPE_MIN_VALUE (unsigned_char_type));
SET_TYPE_RM_MAX_VALUE (char_type, TYPE_MAX_VALUE (unsigned_char_type));
}
void
finish_fat_pointer_type (tree record_type, tree field_list)
{
if (STRICT_ALIGNMENT)
SET_TYPE_ALIGN (record_type, MIN (BIGGEST_ALIGNMENT, 2 * POINTER_SIZE));
TYPE_FAT_POINTER_P (record_type) = 1;
finish_record_type (record_type, field_list, 0, false);
TYPE_CONTAINS_PLACEHOLDER_INTERNAL (record_type) = 2;
}
void
finish_record_type (tree record_type, tree field_list, int rep_level,
bool debug_info_p)
{
enum tree_code code = TREE_CODE (record_type);
tree name = TYPE_IDENTIFIER (record_type);
tree ada_size = bitsize_zero_node;
tree size = bitsize_zero_node;
bool had_size = TYPE_SIZE (record_type) != 0;
bool had_size_unit = TYPE_SIZE_UNIT (record_type) != 0;
bool had_align = TYPE_ALIGN (record_type) != 0;
tree field;
TYPE_FIELDS (record_type) = field_list;
TYPE_STUB_DECL (record_type) = create_type_stub_decl (name, record_type);
if (rep_level > 0)
{
SET_TYPE_ALIGN (record_type, MAX (BITS_PER_UNIT,
TYPE_ALIGN (record_type)));
if (!had_size_unit)
TYPE_SIZE_UNIT (record_type) = size_zero_node;
if (!had_size)
TYPE_SIZE (record_type) = bitsize_zero_node;
else if (code == QUAL_UNION_TYPE)
code = UNION_TYPE;
}
else
{
TYPE_SIZE (record_type) = 0;
#ifdef TARGET_MS_BITFIELD_LAYOUT
if (TARGET_MS_BITFIELD_LAYOUT && TYPE_PACKED (record_type))
decl_attributes (&record_type,
tree_cons (get_identifier ("gcc_struct"),
NULL_TREE, NULL_TREE),
ATTR_FLAG_TYPE_IN_PLACE);
#endif
layout_type (record_type);
}
if (code == QUAL_UNION_TYPE)
field_list = nreverse (field_list);
for (field = field_list; field; field = DECL_CHAIN (field))
{
tree type = TREE_TYPE (field);
tree pos = bit_position (field);
tree this_size = DECL_SIZE (field);
tree this_ada_size;
if (RECORD_OR_UNION_TYPE_P (type)
&& !TYPE_FAT_POINTER_P (type)
&& !TYPE_CONTAINS_TEMPLATE_P (type)
&& TYPE_ADA_SIZE (type))
this_ada_size = TYPE_ADA_SIZE (type);
else
this_ada_size = this_size;
if (DECL_BIT_FIELD (field)
&& operand_equal_p (this_size, TYPE_SIZE (type), 0))
{
unsigned int align = TYPE_ALIGN (type);
if (value_factor_p (pos, align))
{
if (TYPE_ALIGN (record_type) >= align)
{
SET_DECL_ALIGN (field, MAX (DECL_ALIGN (field), align));
DECL_BIT_FIELD (field) = 0;
}
else if (!had_align
&& rep_level == 0
&& value_factor_p (TYPE_SIZE (record_type), align)
&& (!TYPE_MAX_ALIGN (record_type)
|| TYPE_MAX_ALIGN (record_type) >= align))
{
SET_TYPE_ALIGN (record_type, align);
SET_DECL_ALIGN (field, MAX (DECL_ALIGN (field), align));
DECL_BIT_FIELD (field) = 0;
}
}
if (!STRICT_ALIGNMENT
&& DECL_BIT_FIELD (field)
&& value_factor_p (pos, BITS_PER_UNIT))
DECL_BIT_FIELD (field) = 0;
}
if (DECL_BIT_FIELD (field)
&& !(DECL_MODE (field) == BLKmode
&& value_factor_p (pos, BITS_PER_UNIT)))
DECL_NONADDRESSABLE_P (field) = 1;
if (rep_level > 0 && !DECL_BIT_FIELD (field))
SET_TYPE_ALIGN (record_type,
MAX (TYPE_ALIGN (record_type), DECL_ALIGN (field)));
switch (code)
{
case UNION_TYPE:
ada_size = size_binop (MAX_EXPR, ada_size, this_ada_size);
size = size_binop (MAX_EXPR, size, this_size);
break;
case QUAL_UNION_TYPE:
ada_size
= fold_build3 (COND_EXPR, bitsizetype, DECL_QUALIFIER (field),
this_ada_size, ada_size);
size = fold_build3 (COND_EXPR, bitsizetype, DECL_QUALIFIER (field),
this_size, size);
break;
case RECORD_TYPE:
ada_size
= merge_sizes (ada_size, pos, this_ada_size,
TREE_CODE (type) == QUAL_UNION_TYPE, rep_level > 0);
size
= merge_sizes (size, pos, this_size,
TREE_CODE (type) == QUAL_UNION_TYPE, rep_level > 0);
break;
default:
gcc_unreachable ();
}
}
if (code == QUAL_UNION_TYPE)
nreverse (field_list);
if (rep_level < 2)
{
if (TYPE_IS_PADDING_P (record_type) && TYPE_SIZE (record_type))
size = TYPE_SIZE (record_type);
if (!TYPE_FAT_POINTER_P (record_type)
&& !TYPE_CONTAINS_TEMPLATE_P (record_type))
SET_TYPE_ADA_SIZE (record_type, ada_size);
if (rep_level > 0)
{
tree size_unit = had_size_unit
? TYPE_SIZE_UNIT (record_type)
: convert (sizetype,
size_binop (CEIL_DIV_EXPR, size,
bitsize_unit_node));
unsigned int align = TYPE_ALIGN (record_type);
TYPE_SIZE (record_type) = variable_size (round_up (size, align));
TYPE_SIZE_UNIT (record_type)
= variable_size (round_up (size_unit, align / BITS_PER_UNIT));
compute_record_mode (record_type);
}
}
TYPE_MAX_ALIGN (record_type) = 0;
if (debug_info_p)
rest_of_record_type_compilation (record_type);
}
void
add_parallel_type (tree type, tree parallel_type)
{
tree decl = TYPE_STUB_DECL (type);
while (DECL_PARALLEL_TYPE (decl))
decl = TYPE_STUB_DECL (DECL_PARALLEL_TYPE (decl));
SET_DECL_PARALLEL_TYPE (decl, parallel_type);
if (TYPE_CONTEXT (parallel_type))
return;
if (TYPE_CONTEXT (type))
gnat_set_type_context (parallel_type, TYPE_CONTEXT (type));
}
static bool
has_parallel_type (tree type)
{
tree decl = TYPE_STUB_DECL (type);
return DECL_PARALLEL_TYPE (decl) != NULL_TREE;
}
void
rest_of_record_type_compilation (tree record_type)
{
bool var_size = false;
tree field;
if (TYPE_IS_PADDING_P (record_type))
return;
if (has_parallel_type (record_type))
return;
for (field = TYPE_FIELDS (record_type); field; field = DECL_CHAIN (field))
{
if (TREE_CODE (DECL_SIZE (field)) != INTEGER_CST
|| (TREE_CODE (record_type) == QUAL_UNION_TYPE
&& TREE_CODE (DECL_QUALIFIER (field)) != INTEGER_CST))
{
var_size = true;
break;
}
}
if (var_size && gnat_encodings != DWARF_GNAT_ENCODINGS_MINIMAL)
{
tree new_record_type
= make_node (TREE_CODE (record_type) == QUAL_UNION_TYPE
? UNION_TYPE : TREE_CODE (record_type));
tree orig_name = TYPE_IDENTIFIER (record_type), new_name;
tree last_pos = bitsize_zero_node;
tree old_field, prev_old_field = NULL_TREE;
new_name
= concat_name (orig_name, TREE_CODE (record_type) == QUAL_UNION_TYPE
? "XVU" : "XVE");
TYPE_NAME (new_record_type) = new_name;
SET_TYPE_ALIGN (new_record_type, BIGGEST_ALIGNMENT);
TYPE_STUB_DECL (new_record_type)
= create_type_stub_decl (new_name, new_record_type);
DECL_IGNORED_P (TYPE_STUB_DECL (new_record_type))
= DECL_IGNORED_P (TYPE_STUB_DECL (record_type));
gnat_pushdecl (TYPE_STUB_DECL (new_record_type), Empty);
TYPE_SIZE (new_record_type) = size_int (TYPE_ALIGN (record_type));
TYPE_SIZE_UNIT (new_record_type)
= size_int (TYPE_ALIGN (record_type) / BITS_PER_UNIT);
for (old_field = TYPE_FIELDS (record_type); old_field;
old_field = DECL_CHAIN (old_field))
{
tree field_type = TREE_TYPE (old_field);
tree field_name = DECL_NAME (old_field);
tree curpos = fold_bit_position (old_field);
tree pos, new_field;
bool var = false;
unsigned int align = 0;
if (TREE_CODE (new_record_type) == UNION_TYPE)
pos = bitsize_zero_node;
else
pos = compute_related_constant (curpos, last_pos);
if (!pos
&& TREE_CODE (curpos) == MULT_EXPR
&& tree_fits_uhwi_p (TREE_OPERAND (curpos, 1)))
{
tree offset = TREE_OPERAND (curpos, 0);
align = tree_to_uhwi (TREE_OPERAND (curpos, 1));
align = scale_by_factor_of (offset, align);
last_pos = round_up (last_pos, align);
pos = compute_related_constant (curpos, last_pos);
}
else if (!pos
&& TREE_CODE (curpos) == PLUS_EXPR
&& tree_fits_uhwi_p (TREE_OPERAND (curpos, 1))
&& TREE_CODE (TREE_OPERAND (curpos, 0)) == MULT_EXPR
&& tree_fits_uhwi_p
(TREE_OPERAND (TREE_OPERAND (curpos, 0), 1)))
{
tree offset = TREE_OPERAND (TREE_OPERAND (curpos, 0), 0);
unsigned HOST_WIDE_INT addend
= tree_to_uhwi (TREE_OPERAND (curpos, 1));
align
= tree_to_uhwi (TREE_OPERAND (TREE_OPERAND (curpos, 0), 1));
align = scale_by_factor_of (offset, align);
align = MIN (align, addend & -addend);
last_pos = round_up (last_pos, align);
pos = compute_related_constant (curpos, last_pos);
}
else if (potential_alignment_gap (prev_old_field, old_field, pos))
{
align = TYPE_ALIGN (field_type);
last_pos = round_up (last_pos, align);
pos = compute_related_constant (curpos, last_pos);
}
if (!pos)
pos = bitsize_zero_node;
if (TREE_CODE (DECL_SIZE (old_field)) != INTEGER_CST)
{
field_type = build_pointer_type (field_type);
if (align != 0 && TYPE_ALIGN (field_type) > align)
{
field_type = copy_type (field_type);
SET_TYPE_ALIGN (field_type, align);
}
var = true;
}
if (var || align != 0)
{
char suffix[16];
if (align != 0)
sprintf (suffix, "XV%c%u", var ? 'L' : 'A',
align / BITS_PER_UNIT);
else
strcpy (suffix, "XVL");
field_name = concat_name (field_name, suffix);
}
new_field
= create_field_decl (field_name, field_type, new_record_type,
DECL_SIZE (old_field), pos, 0, 0);
DECL_CHAIN (new_field) = TYPE_FIELDS (new_record_type);
TYPE_FIELDS (new_record_type) = new_field;
last_pos = size_binop (PLUS_EXPR, curpos,
(TREE_CODE (TREE_TYPE (old_field))
== QUAL_UNION_TYPE)
? bitsize_zero_node
: DECL_SIZE (old_field));
prev_old_field = old_field;
}
TYPE_FIELDS (new_record_type) = nreverse (TYPE_FIELDS (new_record_type));
add_parallel_type (record_type, new_record_type);
}
}
static tree
merge_sizes (tree last_size, tree first_bit, tree size, bool special,
bool has_rep)
{
tree type = TREE_TYPE (last_size);
tree new_size;
if (!special || TREE_CODE (size) != COND_EXPR)
{
new_size = size_binop (PLUS_EXPR, first_bit, size);
if (has_rep)
new_size = size_binop (MAX_EXPR, last_size, new_size);
}
else
new_size = fold_build3 (COND_EXPR, type, TREE_OPERAND (size, 0),
integer_zerop (TREE_OPERAND (size, 1))
? last_size : merge_sizes (last_size, first_bit,
TREE_OPERAND (size, 1),
1, has_rep),
integer_zerop (TREE_OPERAND (size, 2))
? last_size : merge_sizes (last_size, first_bit,
TREE_OPERAND (size, 2),
1, has_rep));
while (TREE_CODE (new_size) == NON_LVALUE_EXPR)
new_size = TREE_OPERAND (new_size, 0);
return new_size;
}
static tree
fold_bit_position (const_tree field)
{
tree offset = DECL_FIELD_OFFSET (field);
if (TREE_CODE (offset) == MULT_EXPR || TREE_CODE (offset) == PLUS_EXPR)
offset = size_binop (TREE_CODE (offset),
fold_convert (bitsizetype, TREE_OPERAND (offset, 0)),
fold_convert (bitsizetype, TREE_OPERAND (offset, 1)));
else
offset = fold_convert (bitsizetype, offset);
return size_binop (PLUS_EXPR, DECL_FIELD_BIT_OFFSET (field),
size_binop (MULT_EXPR, offset, bitsize_unit_node));
}
static tree
compute_related_constant (tree op0, tree op1)
{
tree factor, op0_var, op1_var, op0_cst, op1_cst, result;
if (TREE_CODE (op0) == MULT_EXPR
&& TREE_CODE (op1) == MULT_EXPR
&& TREE_CODE (TREE_OPERAND (op0, 1)) == INTEGER_CST
&& TREE_OPERAND (op1, 1) == TREE_OPERAND (op0, 1))
{
factor = TREE_OPERAND (op0, 1);
op0 = TREE_OPERAND (op0, 0);
op1 = TREE_OPERAND (op1, 0);
}
else
factor = NULL_TREE;
op0_cst = split_plus (op0, &op0_var);
op1_cst = split_plus (op1, &op1_var);
result = size_binop (MINUS_EXPR, op0_cst, op1_cst);
if (operand_equal_p (op0_var, op1_var, 0))
return factor ? size_binop (MULT_EXPR, factor, result) : result;
return NULL_TREE;
}
static tree
split_plus (tree in, tree *pvar)
{
in = remove_conversions (in, false);
*pvar = convert (bitsizetype, in);
if (TREE_CODE (in) == INTEGER_CST)
{
*pvar = bitsize_zero_node;
return convert (bitsizetype, in);
}
else if (TREE_CODE (in) == PLUS_EXPR || TREE_CODE (in) == MINUS_EXPR)
{
tree lhs_var, rhs_var;
tree lhs_con = split_plus (TREE_OPERAND (in, 0), &lhs_var);
tree rhs_con = split_plus (TREE_OPERAND (in, 1), &rhs_var);
if (lhs_var == TREE_OPERAND (in, 0)
&& rhs_var == TREE_OPERAND (in, 1))
return bitsize_zero_node;
*pvar = size_binop (TREE_CODE (in), lhs_var, rhs_var);
return size_binop (TREE_CODE (in), lhs_con, rhs_con);
}
else
return bitsize_zero_node;
}

tree
copy_type (tree type)
{
tree new_type = copy_node (type);
if (TYPE_LANG_SPECIFIC (type))
{
TYPE_LANG_SPECIFIC (new_type) = NULL;
SET_TYPE_LANG_SPECIFIC (new_type, GET_TYPE_LANG_SPECIFIC (type));
}
if ((INTEGRAL_TYPE_P (type) || TREE_CODE (type) == REAL_TYPE)
&& TYPE_RM_VALUES (type))
{
TYPE_RM_VALUES (new_type) = NULL_TREE;
SET_TYPE_RM_SIZE (new_type, TYPE_RM_SIZE (type));
SET_TYPE_RM_MIN_VALUE (new_type, TYPE_RM_MIN_VALUE (type));
SET_TYPE_RM_MAX_VALUE (new_type, TYPE_RM_MAX_VALUE (type));
}
TYPE_STUB_DECL (new_type) = TYPE_STUB_DECL (type);
TYPE_POINTER_TO (new_type) = NULL_TREE;
TYPE_REFERENCE_TO (new_type) = NULL_TREE;
TYPE_MAIN_VARIANT (new_type) = new_type;
TYPE_NEXT_VARIANT (new_type) = NULL_TREE;
TYPE_CANONICAL (new_type) = new_type;
return new_type;
}

tree
create_index_type (tree min, tree max, tree index, Node_Id gnat_node)
{
tree type = build_nonshared_range_type (sizetype, min, max);
SET_TYPE_INDEX_TYPE (type, index);
create_type_decl (NULL_TREE, type, true, false, gnat_node);
return type;
}
tree
create_range_type (tree type, tree min, tree max)
{
tree range_type;
if (!type)
type = sizetype;
range_type = build_nonshared_range_type (type, TYPE_MIN_VALUE (type),
TYPE_MAX_VALUE (type));
SET_TYPE_RM_MIN_VALUE (range_type, min);
SET_TYPE_RM_MAX_VALUE (range_type, max);
return range_type;
}

tree
create_type_stub_decl (tree name, tree type)
{
tree type_decl = build_decl (input_location, TYPE_DECL, name, type);
DECL_ARTIFICIAL (type_decl) = 1;
TYPE_ARTIFICIAL (type) = 1;
return type_decl;
}
tree
create_type_decl (tree name, tree type, bool artificial_p, bool debug_info_p,
Node_Id gnat_node)
{
enum tree_code code = TREE_CODE (type);
bool is_named
= TYPE_NAME (type) && TREE_CODE (TYPE_NAME (type)) == TYPE_DECL;
tree type_decl;
gcc_assert (!TYPE_IS_DUMMY_P (type));
if (!is_named && TYPE_STUB_DECL (type))
{
type_decl = TYPE_STUB_DECL (type);
DECL_NAME (type_decl) = name;
}
else
type_decl = build_decl (input_location, TYPE_DECL, name, type);
DECL_ARTIFICIAL (type_decl) = artificial_p;
TYPE_ARTIFICIAL (type) = artificial_p;
gnat_pushdecl (type_decl, gnat_node);
if (!is_named && type != DECL_ORIGINAL_TYPE (type_decl))
TYPE_STUB_DECL (type) = type_decl;
if (code == UNCONSTRAINED_ARRAY_TYPE || !debug_info_p)
DECL_IGNORED_P (type_decl) = 1;
return type_decl;
}

tree
create_var_decl (tree name, tree asm_name, tree type, tree init,
bool const_flag, bool public_flag, bool extern_flag,
bool static_flag, bool volatile_flag, bool artificial_p,
bool debug_info_p, struct attrib *attr_list,
Node_Id gnat_node, bool const_decl_allowed_p)
{
const bool static_storage = static_flag || global_bindings_p ();
const bool init_const
= (init
&& gnat_types_compatible_p (type, TREE_TYPE (init))
&& (extern_flag || static_storage
? initializer_constant_valid_p (init, TREE_TYPE (init))
!= NULL_TREE
: TREE_CONSTANT (init)));
const bool constant_p = const_flag && init_const;
tree var_decl
= build_decl (input_location,
(constant_p
&& const_decl_allowed_p
&& !AGGREGATE_TYPE_P (type) ? CONST_DECL : VAR_DECL),
name, type);
if (const_flag && init && POINTER_TYPE_P (type))
{
tree inner = init;
if (TREE_CODE (inner) == COMPOUND_EXPR)
inner = TREE_OPERAND (inner, 1);
inner = remove_conversions (inner, true);
if (TREE_CODE (inner) == ADDR_EXPR
&& ((TREE_CODE (TREE_OPERAND (inner, 0)) == CALL_EXPR
&& !call_is_atomic_load (TREE_OPERAND (inner, 0)))
|| (TREE_CODE (TREE_OPERAND (inner, 0)) == VAR_DECL
&& DECL_RETURN_VALUE_P (TREE_OPERAND (inner, 0)))))
DECL_RETURN_VALUE_P (var_decl) = 1;
}
if ((extern_flag && !constant_p)
|| (type_annotate_only && init && !TREE_CONSTANT (init)))
init = NULL_TREE;
if (init && !init_const && global_bindings_p ())
Check_Elaboration_Code_Allowed (gnat_node);
DECL_INITIAL (var_decl) = init;
DECL_ARTIFICIAL (var_decl) = artificial_p;
DECL_EXTERNAL (var_decl) = extern_flag;
TREE_CONSTANT (var_decl) = constant_p;
TREE_READONLY (var_decl) = const_flag;
TREE_PUBLIC (var_decl) = extern_flag || (public_flag && static_storage);
TREE_STATIC (var_decl) = !extern_flag && static_storage;
TREE_SIDE_EFFECTS (var_decl)
= TREE_THIS_VOLATILE (var_decl)
= TYPE_VOLATILE (type) | volatile_flag;
if (TREE_SIDE_EFFECTS (var_decl))
TREE_ADDRESSABLE (var_decl) = 1;
if (!flag_no_common
&& TREE_CODE (var_decl) == VAR_DECL
&& TREE_PUBLIC (var_decl)
&& !have_global_bss_p ())
DECL_COMMON (var_decl) = 1;
if (!debug_info_p
|| (TREE_CODE (var_decl) == CONST_DECL && !optimize)
|| (extern_flag
&& constant_p
&& init
&& initializer_constant_valid_p (init, TREE_TYPE (init))
!= null_pointer_node))
DECL_IGNORED_P (var_decl) = 1;
if (TREE_CODE (var_decl) == VAR_DECL)
process_attributes (&var_decl, &attr_list, true, gnat_node);
gnat_pushdecl (var_decl, gnat_node);
if (TREE_CODE (var_decl) == VAR_DECL && asm_name)
{
if (*IDENTIFIER_POINTER (asm_name) != '*')
asm_name = targetm.mangle_decl_assembler_name (var_decl, asm_name);
SET_DECL_ASSEMBLER_NAME (var_decl, asm_name);
}
return var_decl;
}

static bool
aggregate_type_contains_array_p (tree type)
{
switch (TREE_CODE (type))
{
case RECORD_TYPE:
case UNION_TYPE:
case QUAL_UNION_TYPE:
{
tree field;
for (field = TYPE_FIELDS (type); field; field = DECL_CHAIN (field))
if (AGGREGATE_TYPE_P (TREE_TYPE (field))
&& aggregate_type_contains_array_p (TREE_TYPE (field)))
return true;
return false;
}
case ARRAY_TYPE:
return true;
default:
gcc_unreachable ();
}
}
tree
create_field_decl (tree name, tree type, tree record_type, tree size, tree pos,
int packed, int addressable)
{
tree field_decl = build_decl (input_location, FIELD_DECL, name, type);
DECL_CONTEXT (field_decl) = record_type;
TREE_READONLY (field_decl) = TYPE_READONLY (type);
if (packed && (TYPE_MODE (type) == BLKmode
|| (!pos
&& AGGREGATE_TYPE_P (type)
&& aggregate_type_contains_array_p (type))))
SET_DECL_ALIGN (field_decl, BITS_PER_UNIT);
if (size)
size = convert (bitsizetype, size);
else if (packed == 1)
{
size = rm_size (type);
if (TYPE_MODE (type) == BLKmode)
size = round_up (size, BITS_PER_UNIT);
}
if (addressable >= 0
&& size
&& TREE_CODE (size) == INTEGER_CST
&& TREE_CODE (TYPE_SIZE (type)) == INTEGER_CST
&& (!tree_int_cst_equal (size, TYPE_SIZE (type))
|| (pos && !value_factor_p (pos, TYPE_ALIGN (type)))
|| packed
|| (TYPE_ALIGN (record_type) != 0
&& TYPE_ALIGN (record_type) < TYPE_ALIGN (type))))
{
DECL_BIT_FIELD (field_decl) = 1;
DECL_SIZE (field_decl) = size;
if (!packed && !pos)
{
if (TYPE_ALIGN (record_type) != 0
&& TYPE_ALIGN (record_type) < TYPE_ALIGN (type))
SET_DECL_ALIGN (field_decl, TYPE_ALIGN (record_type));
else
SET_DECL_ALIGN (field_decl, TYPE_ALIGN (type));
}
}
DECL_PACKED (field_decl) = pos ? DECL_BIT_FIELD (field_decl) : packed;
{
unsigned int bit_align
= (DECL_BIT_FIELD (field_decl) ? 1
: packed && TYPE_MODE (type) != BLKmode ? BITS_PER_UNIT : 0);
if (bit_align > DECL_ALIGN (field_decl))
SET_DECL_ALIGN (field_decl, bit_align);
else if (!bit_align && TYPE_ALIGN (type) > DECL_ALIGN (field_decl))
{
SET_DECL_ALIGN (field_decl, TYPE_ALIGN (type));
DECL_USER_ALIGN (field_decl) = TYPE_USER_ALIGN (type);
}
}
if (pos)
{
unsigned int known_align;
if (tree_fits_uhwi_p (pos))
known_align = tree_to_uhwi (pos) & - tree_to_uhwi (pos);
else
known_align = BITS_PER_UNIT;
if (TYPE_ALIGN (record_type)
&& (known_align == 0 || known_align > TYPE_ALIGN (record_type)))
known_align = TYPE_ALIGN (record_type);
layout_decl (field_decl, known_align);
SET_DECL_OFFSET_ALIGN (field_decl,
tree_fits_uhwi_p (pos) ? BIGGEST_ALIGNMENT
: BITS_PER_UNIT);
pos_from_bit (&DECL_FIELD_OFFSET (field_decl),
&DECL_FIELD_BIT_OFFSET (field_decl),
DECL_OFFSET_ALIGN (field_decl), pos);
}
if (!addressable && !type_for_nonaliased_component_p (type))
addressable = 1;
DECL_NONADDRESSABLE_P (field_decl) = !addressable;
return field_decl;
}

tree
create_param_decl (tree name, tree type)
{
tree param_decl = build_decl (input_location, PARM_DECL, name, type);
if (targetm.calls.promote_prototypes (NULL_TREE)
&& INTEGRAL_TYPE_P (type)
&& TYPE_PRECISION (type) < TYPE_PRECISION (integer_type_node))
{
if (TREE_CODE (type) == INTEGER_TYPE
&& TYPE_BIASED_REPRESENTATION_P (type))
{
tree subtype
= make_unsigned_type (TYPE_PRECISION (integer_type_node));
TREE_TYPE (subtype) = integer_type_node;
TYPE_BIASED_REPRESENTATION_P (subtype) = 1;
SET_TYPE_RM_MIN_VALUE (subtype, TYPE_MIN_VALUE (type));
SET_TYPE_RM_MAX_VALUE (subtype, TYPE_MAX_VALUE (type));
type = subtype;
}
else
type = integer_type_node;
}
DECL_ARG_TYPE (param_decl) = type;
return param_decl;
}

void
process_attributes (tree *node, struct attrib **attr_list, bool in_place,
Node_Id gnat_node)
{
struct attrib *attr;
for (attr = *attr_list; attr; attr = attr->next)
switch (attr->type)
{
case ATTR_MACHINE_ATTRIBUTE:
Sloc_to_locus (Sloc (gnat_node), &input_location);
decl_attributes (node, tree_cons (attr->name, attr->args, NULL_TREE),
in_place ? ATTR_FLAG_TYPE_IN_PLACE : 0);
break;
case ATTR_LINK_ALIAS:
if (!DECL_EXTERNAL (*node))
{
TREE_STATIC (*node) = 1;
assemble_alias (*node, attr->name);
}
break;
case ATTR_WEAK_EXTERNAL:
if (SUPPORTS_WEAK)
declare_weak (*node);
else
post_error ("?weak declarations not supported on this target",
attr->error_point);
break;
case ATTR_LINK_SECTION:
if (targetm_common.have_named_sections)
{
set_decl_section_name (*node, IDENTIFIER_POINTER (attr->name));
DECL_COMMON (*node) = 0;
}
else
post_error ("?section attributes are not supported for this target",
attr->error_point);
break;
case ATTR_LINK_CONSTRUCTOR:
DECL_STATIC_CONSTRUCTOR (*node) = 1;
TREE_USED (*node) = 1;
break;
case ATTR_LINK_DESTRUCTOR:
DECL_STATIC_DESTRUCTOR (*node) = 1;
TREE_USED (*node) = 1;
break;
case ATTR_THREAD_LOCAL_STORAGE:
set_decl_tls_model (*node, decl_default_tls_model (*node));
DECL_COMMON (*node) = 0;
break;
}
*attr_list = NULL;
}
bool
value_factor_p (tree value, HOST_WIDE_INT factor)
{
if (tree_fits_uhwi_p (value))
return tree_to_uhwi (value) % factor == 0;
if (TREE_CODE (value) == MULT_EXPR)
return (value_factor_p (TREE_OPERAND (value, 0), factor)
|| value_factor_p (TREE_OPERAND (value, 1), factor));
return false;
}
bool
renaming_from_instantiation_p (Node_Id gnat_node)
{
if (Nkind (gnat_node) != N_Defining_Identifier
|| !Is_Object (gnat_node)
|| Comes_From_Source (gnat_node)
|| !Present (Renamed_Object (gnat_node)))
return false;
gnat_node = Renamed_Object (gnat_node);
if (Nkind (gnat_node) != N_Identifier)
return false;
gnat_node = Entity (gnat_node);
if (!Present (Parent (gnat_node)))
return false;
gnat_node = Parent (gnat_node);
return
(Present (gnat_node)
&& Nkind (gnat_node) == N_Object_Declaration
&& Present (Corresponding_Generic_Association (gnat_node)));
}
static struct deferred_decl_context_node *
add_deferred_decl_context (tree decl, Entity_Id gnat_scope, int force_global)
{
struct deferred_decl_context_node *new_node;
new_node
= (struct deferred_decl_context_node * ) xmalloc (sizeof (*new_node));
new_node->decl = decl;
new_node->gnat_scope = gnat_scope;
new_node->force_global = force_global;
new_node->types.create (1);
new_node->next = deferred_decl_context_queue;
deferred_decl_context_queue = new_node;
return new_node;
}
static void
add_deferred_type_context (struct deferred_decl_context_node *n, tree type)
{
n->types.safe_push (type);
}
static tree
compute_deferred_decl_context (Entity_Id gnat_scope)
{
tree context;
if (present_gnu_tree (gnat_scope))
context = get_gnu_tree (gnat_scope);
else
return NULL_TREE;
if (TREE_CODE (context) == TYPE_DECL)
{
const tree context_type = TREE_TYPE (context);
if (TYPE_DUMMY_P (context_type))
return NULL_TREE;
else
context = context_type;
}
return context;
}
void
process_deferred_decl_context (bool force)
{
struct deferred_decl_context_node **it = &deferred_decl_context_queue;
struct deferred_decl_context_node *node;
while (*it)
{
bool processed = false;
tree context = NULL_TREE;
Entity_Id gnat_scope;
node = *it;
gnat_scope = node->gnat_scope;
while (Present (gnat_scope))
{
context = compute_deferred_decl_context (gnat_scope);
if (!force || context)
break;
gnat_scope = get_debug_scope (gnat_scope, NULL);
}
if (context && node->force_global > 0)
{
tree ctx = context;
while (ctx)
{
gcc_assert (TREE_CODE (ctx) != FUNCTION_DECL);
ctx = DECL_P (ctx) ? DECL_CONTEXT (ctx) : TYPE_CONTEXT (ctx);
}
}
if (force && !context)
context = get_global_context ();
if (context)
{
tree t;
int i;
DECL_CONTEXT (node->decl) = context;
FOR_EACH_VEC_ELT (node->types, i, t)
{
gnat_set_type_context (t, context);
}
processed = true;
}
if (processed)
{
*it = node->next;
node->types.release ();
free (node);
}
else
it = &node->next;
}
}
static unsigned int
scale_by_factor_of (tree expr, unsigned int value)
{
unsigned HOST_WIDE_INT addend = 0;
unsigned HOST_WIDE_INT factor = 1;
expr = remove_conversions (expr, true);
if (TREE_CODE (expr) == CALL_EXPR)
expr = maybe_inline_call_in_expr (expr);
if (TREE_CODE (expr) == PLUS_EXPR
&& TREE_CODE (TREE_OPERAND (expr, 1)) == INTEGER_CST
&& tree_fits_uhwi_p (TREE_OPERAND (expr, 1)))
{
addend = TREE_INT_CST_LOW (TREE_OPERAND (expr, 1));
expr = TREE_OPERAND (expr, 0);
}
if (TREE_CODE (expr) == BIT_AND_EXPR
&& TREE_CODE (TREE_OPERAND (expr, 1)) == INTEGER_CST)
{
unsigned HOST_WIDE_INT mask = TREE_INT_CST_LOW (TREE_OPERAND (expr, 1));
unsigned int i = 0;
while ((mask & 1) == 0 && i < HOST_BITS_PER_WIDE_INT)
{
mask >>= 1;
factor *= 2;
i++;
}
}
if (addend % factor != 0)
factor = 1;
return factor * value;
}
static bool
potential_alignment_gap (tree prev_field, tree curr_field, tree offset)
{
if (!prev_field)
return false;
if (TREE_CODE (TREE_TYPE (prev_field)) == QUAL_UNION_TYPE)
return false;
if (offset && tree_fits_uhwi_p (offset))
return !integer_zerop (offset);
if (tree_fits_uhwi_p (DECL_SIZE (prev_field))
&& tree_fits_uhwi_p (bit_position (prev_field)))
return ((tree_to_uhwi (bit_position (prev_field))
+ tree_to_uhwi (DECL_SIZE (prev_field)))
% DECL_ALIGN (curr_field) != 0);
if (value_factor_p (bit_position (prev_field), DECL_ALIGN (curr_field))
&& value_factor_p (DECL_SIZE (prev_field), DECL_ALIGN (curr_field)))
return false;
return true;
}
tree
create_label_decl (tree name, Node_Id gnat_node)
{
tree label_decl
= build_decl (input_location, LABEL_DECL, name, void_type_node);
SET_DECL_MODE (label_decl, VOIDmode);
gnat_pushdecl (label_decl, gnat_node);
return label_decl;
}

tree
create_subprog_decl (tree name, tree asm_name, tree type, tree param_decl_list,
enum inline_status_t inline_status, bool public_flag,
bool extern_flag, bool artificial_p, bool debug_info_p,
bool definition, struct attrib *attr_list,
Node_Id gnat_node)
{
tree subprog_decl = build_decl (input_location, FUNCTION_DECL, name, type);
DECL_ARGUMENTS (subprog_decl) = param_decl_list;
DECL_ARTIFICIAL (subprog_decl) = artificial_p;
DECL_EXTERNAL (subprog_decl) = extern_flag;
TREE_PUBLIC (subprog_decl) = public_flag;
if (!debug_info_p)
DECL_IGNORED_P (subprog_decl) = 1;
if (definition)
DECL_FUNCTION_IS_DEF (subprog_decl) = 1;
switch (inline_status)
{
case is_suppressed:
DECL_UNINLINABLE (subprog_decl) = 1;
break;
case is_disabled:
break;
case is_required:
if (Back_End_Inlining)
{
decl_attributes (&subprog_decl,
tree_cons (get_identifier ("always_inline"),
NULL_TREE, NULL_TREE),
ATTR_FLAG_TYPE_IN_PLACE);
TREE_PUBLIC (subprog_decl) = 0;
}
case is_enabled:
DECL_DECLARED_INLINE_P (subprog_decl) = 1;
DECL_NO_INLINE_WARNING_P (subprog_decl) = artificial_p;
break;
default:
gcc_unreachable ();
}
process_attributes (&subprog_decl, &attr_list, true, gnat_node);
finish_subprog_decl (subprog_decl, asm_name, type);
gnat_pushdecl (subprog_decl, gnat_node);
rest_of_decl_compilation (subprog_decl, global_bindings_p (), 0);
return subprog_decl;
}
void
finish_subprog_decl (tree decl, tree asm_name, tree type)
{
tree result_decl
= build_decl (DECL_SOURCE_LOCATION (decl), RESULT_DECL, NULL_TREE,
TREE_TYPE (type));
DECL_ARTIFICIAL (result_decl) = 1;
DECL_IGNORED_P (result_decl) = 1;
DECL_BY_REFERENCE (result_decl) = TREE_ADDRESSABLE (type);
DECL_RESULT (decl) = result_decl;
TREE_READONLY (decl) = TYPE_READONLY (type);
TREE_THIS_VOLATILE (decl) = TYPE_VOLATILE (type);
if (asm_name)
{
if (*IDENTIFIER_POINTER (asm_name) != '*')
asm_name = targetm.mangle_decl_assembler_name (decl, asm_name);
SET_DECL_ASSEMBLER_NAME (decl, asm_name);
if (asm_name == main_identifier_node)
DECL_NAME (decl) = main_identifier_node;
}
}

void
begin_subprog_body (tree subprog_decl)
{
tree param_decl;
announce_function (subprog_decl);
TREE_STATIC (subprog_decl) = 1;
gcc_assert (current_function_decl == decl_function_context (subprog_decl));
current_function_decl = subprog_decl;
gnat_pushlevel ();
for (param_decl = DECL_ARGUMENTS (subprog_decl); param_decl;
param_decl = DECL_CHAIN (param_decl))
DECL_CONTEXT (param_decl) = subprog_decl;
make_decl_rtl (subprog_decl);
}
void
end_subprog_body (tree body)
{
tree fndecl = current_function_decl;
BLOCK_SUPERCONTEXT (current_binding_level->block) = fndecl;
DECL_INITIAL (fndecl) = current_binding_level->block;
gnat_poplevel ();
DECL_CONTEXT (DECL_RESULT (fndecl)) = fndecl;
if (TREE_CODE (body) == BIND_EXPR)
{
BLOCK_SUPERCONTEXT (BIND_EXPR_BLOCK (body)) = fndecl;
DECL_INITIAL (fndecl) = BIND_EXPR_BLOCK (body);
}
DECL_SAVED_TREE (fndecl) = body;
current_function_decl = decl_function_context (fndecl);
}
void
rest_of_subprog_body_compilation (tree subprog_decl)
{
error_gnat_node = Empty;
if (type_annotate_only)
return;
dump_function (TDI_original, subprog_decl);
if (!decl_function_context (subprog_decl))
cgraph_node::finalize_function (subprog_decl, false);
else
(void) cgraph_node::get_create (subprog_decl);
}
tree
gnat_builtin_function (tree decl)
{
gnat_pushdecl (decl, Empty);
return decl;
}
tree
gnat_type_for_size (unsigned precision, int unsignedp)
{
tree t;
char type_name[20];
if (precision <= 2 * MAX_BITS_PER_WORD
&& signed_and_unsigned_types[precision][unsignedp])
return signed_and_unsigned_types[precision][unsignedp];
if (unsignedp)
t = make_unsigned_type (precision);
else
t = make_signed_type (precision);
TYPE_ARTIFICIAL (t) = 1;
if (precision <= 2 * MAX_BITS_PER_WORD)
signed_and_unsigned_types[precision][unsignedp] = t;
if (!TYPE_NAME (t))
{
sprintf (type_name, "%sSIGNED_%u", unsignedp ? "UN" : "", precision);
TYPE_NAME (t) = get_identifier (type_name);
}
return t;
}
static tree
float_type_for_precision (int precision, machine_mode mode)
{
tree t;
char type_name[20];
if (float_types[(int) mode])
return float_types[(int) mode];
float_types[(int) mode] = t = make_node (REAL_TYPE);
TYPE_PRECISION (t) = precision;
layout_type (t);
gcc_assert (TYPE_MODE (t) == mode);
if (!TYPE_NAME (t))
{
sprintf (type_name, "FLOAT_%d", precision);
TYPE_NAME (t) = get_identifier (type_name);
}
return t;
}
tree
gnat_type_for_mode (machine_mode mode, int unsignedp)
{
if (mode == BLKmode)
return NULL_TREE;
if (mode == VOIDmode)
return void_type_node;
if (COMPLEX_MODE_P (mode))
return NULL_TREE;
scalar_float_mode float_mode;
if (is_a <scalar_float_mode> (mode, &float_mode))
return float_type_for_precision (GET_MODE_PRECISION (float_mode),
float_mode);
scalar_int_mode int_mode;
if (is_a <scalar_int_mode> (mode, &int_mode))
return gnat_type_for_size (GET_MODE_BITSIZE (int_mode), unsignedp);
if (VECTOR_MODE_P (mode))
{
machine_mode inner_mode = GET_MODE_INNER (mode);
tree inner_type = gnat_type_for_mode (inner_mode, unsignedp);
if (inner_type)
return build_vector_type_for_mode (inner_type, mode);
}
return NULL_TREE;
}
tree
gnat_signed_or_unsigned_type_for (int unsignedp, tree type_node)
{
if (type_node == char_type_node)
return unsignedp ? unsigned_char_type_node : signed_char_type_node;
tree type = gnat_type_for_size (TYPE_PRECISION (type_node), unsignedp);
if (TREE_CODE (type_node) == INTEGER_TYPE && TYPE_MODULAR_P (type_node))
{
type = copy_type (type);
TREE_TYPE (type) = type_node;
}
else if (TREE_TYPE (type_node)
&& TREE_CODE (TREE_TYPE (type_node)) == INTEGER_TYPE
&& TYPE_MODULAR_P (TREE_TYPE (type_node)))
{
type = copy_type (type);
TREE_TYPE (type) = TREE_TYPE (type_node);
}
return type;
}
int
gnat_types_compatible_p (tree t1, tree t2)
{
enum tree_code code;
if (TYPE_MAIN_VARIANT (t1) == TYPE_MAIN_VARIANT (t2))
return 1;
if ((code = TREE_CODE (t1)) != TREE_CODE (t2))
return 0;
if (code == VECTOR_TYPE
&& known_eq (TYPE_VECTOR_SUBPARTS (t1), TYPE_VECTOR_SUBPARTS (t2))
&& TREE_CODE (TREE_TYPE (t1)) == TREE_CODE (TREE_TYPE (t2))
&& TYPE_PRECISION (TREE_TYPE (t1)) == TYPE_PRECISION (TREE_TYPE (t2)))
return 1;
if (code == ARRAY_TYPE
&& (TYPE_DOMAIN (t1) == TYPE_DOMAIN (t2)
|| (TYPE_DOMAIN (t1)
&& TYPE_DOMAIN (t2)
&& tree_int_cst_equal (TYPE_MIN_VALUE (TYPE_DOMAIN (t1)),
TYPE_MIN_VALUE (TYPE_DOMAIN (t2)))
&& tree_int_cst_equal (TYPE_MAX_VALUE (TYPE_DOMAIN (t1)),
TYPE_MAX_VALUE (TYPE_DOMAIN (t2)))))
&& (TREE_TYPE (t1) == TREE_TYPE (t2)
|| (TREE_CODE (TREE_TYPE (t1)) == ARRAY_TYPE
&& gnat_types_compatible_p (TREE_TYPE (t1), TREE_TYPE (t2))))
&& TYPE_REVERSE_STORAGE_ORDER (t1) == TYPE_REVERSE_STORAGE_ORDER (t2))
return 1;
return 0;
}
bool
gnat_useless_type_conversion (tree expr)
{
if (CONVERT_EXPR_P (expr)
|| TREE_CODE (expr) == VIEW_CONVERT_EXPR
|| TREE_CODE (expr) == NON_LVALUE_EXPR)
return gnat_types_compatible_p (TREE_TYPE (expr),
TREE_TYPE (TREE_OPERAND (expr, 0)));
return false;
}
bool
fntype_same_flags_p (const_tree t, tree cico_list, bool return_unconstrained_p,
bool return_by_direct_ref_p, bool return_by_invisi_ref_p)
{
return TYPE_CI_CO_LIST (t) == cico_list
&& TYPE_RETURN_UNCONSTRAINED_P (t) == return_unconstrained_p
&& TYPE_RETURN_BY_DIRECT_REF_P (t) == return_by_direct_ref_p
&& TREE_ADDRESSABLE (t) == return_by_invisi_ref_p;
}

tree
max_size (tree exp, bool max_p)
{
enum tree_code code = TREE_CODE (exp);
tree type = TREE_TYPE (exp);
tree op0, op1, op2;
switch (TREE_CODE_CLASS (code))
{
case tcc_declaration:
case tcc_constant:
return exp;
case tcc_exceptional:
gcc_assert (code == SSA_NAME);
return exp;
case tcc_vl_exp:
if (code == CALL_EXPR)
{
tree t, *argarray;
int n, i;
t = maybe_inline_call_in_expr (exp);
if (t)
return max_size (t, max_p);
n = call_expr_nargs (exp);
gcc_assert (n > 0);
argarray = XALLOCAVEC (tree, n);
for (i = 0; i < n; i++)
argarray[i] = max_size (CALL_EXPR_ARG (exp, i), max_p);
return build_call_array (type, CALL_EXPR_FN (exp), n, argarray);
}
break;
case tcc_reference:
if (CONTAINS_PLACEHOLDER_P (exp))
{
tree val_type = TREE_TYPE (TREE_OPERAND (exp, 1));
tree val = (max_p ? TYPE_MAX_VALUE (type) : TYPE_MIN_VALUE (type));
return
convert (type,
max_size (convert (get_base_type (val_type), val), true));
}
return exp;
case tcc_comparison:
return build_int_cst (type, max_p ? 1 : 0);
case tcc_unary:
op0 = TREE_OPERAND (exp, 0);
if (code == NON_LVALUE_EXPR)
return max_size (op0, max_p);
if (VOID_TYPE_P (TREE_TYPE (op0)))
return max_p ? TYPE_MAX_VALUE (type) : TYPE_MIN_VALUE (type);
op0 = max_size (op0, code == NEGATE_EXPR ? !max_p : max_p);
if (op0 == TREE_OPERAND (exp, 0))
return exp;
return fold_build1 (code, type, op0);
case tcc_binary:
{
tree lhs = max_size (TREE_OPERAND (exp, 0), max_p);
tree rhs = max_size (TREE_OPERAND (exp, 1),
code == MINUS_EXPR ? !max_p : max_p);
if (max_p && code == MIN_EXPR)
{
if (TREE_CODE (rhs) == INTEGER_CST && TREE_OVERFLOW (rhs))
return lhs;
if (TREE_CODE (lhs) == INTEGER_CST && TREE_OVERFLOW (lhs))
return rhs;
}
if ((code == MINUS_EXPR || code == PLUS_EXPR)
&& TREE_CODE (lhs) == INTEGER_CST
&& TREE_OVERFLOW (lhs)
&& TREE_CODE (rhs) != INTEGER_CST)
return lhs;
if (code == MINUS_EXPR
&& TYPE_UNSIGNED (type)
&& TREE_CODE (rhs) == INTEGER_CST
&& !TREE_OVERFLOW (rhs)
&& tree_int_cst_sign_bit (rhs) != 0)
{
rhs = fold_build1 (NEGATE_EXPR, type, rhs);
code = PLUS_EXPR;
}
if (lhs == TREE_OPERAND (exp, 0) && rhs == TREE_OPERAND (exp, 1))
return exp;
return size_binop (code, lhs, rhs);
}
case tcc_expression:
switch (TREE_CODE_LENGTH (code))
{
case 1:
if (code == SAVE_EXPR)
return exp;
op0 = max_size (TREE_OPERAND (exp, 0),
code == TRUTH_NOT_EXPR ? !max_p : max_p);
if (op0 == TREE_OPERAND (exp, 0))
return exp;
return fold_build1 (code, type, op0);
case 2:
if (code == COMPOUND_EXPR)
return max_size (TREE_OPERAND (exp, 1), max_p);
op0 = max_size (TREE_OPERAND (exp, 0), max_p);
op1 = max_size (TREE_OPERAND (exp, 1), max_p);
if (op0 == TREE_OPERAND (exp, 0) && op1 == TREE_OPERAND (exp, 1))
return exp;
return fold_build2 (code, type, op0, op1);
case 3:
if (code == COND_EXPR)
{
op1 = TREE_OPERAND (exp, 1);
op2 = TREE_OPERAND (exp, 2);
if (!op1 || !op2)
return exp;
return
fold_build2 (max_p ? MAX_EXPR : MIN_EXPR, type,
max_size (op1, max_p), max_size (op2, max_p));
}
break;
default:
break;
}
default:
break;
}
gcc_unreachable ();
}

tree
build_template (tree template_type, tree array_type, tree expr)
{
vec<constructor_elt, va_gc> *template_elts = NULL;
tree bound_list = NULL_TREE;
tree field;
while (TREE_CODE (array_type) == RECORD_TYPE
&& (TYPE_PADDING_P (array_type)
|| TYPE_JUSTIFIED_MODULAR_P (array_type)))
array_type = TREE_TYPE (TYPE_FIELDS (array_type));
if (TREE_CODE (array_type) == ARRAY_TYPE
|| (TREE_CODE (array_type) == INTEGER_TYPE
&& TYPE_HAS_ACTUAL_BOUNDS_P (array_type)))
bound_list = TYPE_ACTUAL_BOUNDS (array_type);
for (field = TYPE_FIELDS (template_type); field;
(bound_list
? (bound_list = TREE_CHAIN (bound_list))
: (array_type = TREE_TYPE (array_type))),
field = DECL_CHAIN (DECL_CHAIN (field)))
{
tree bounds, min, max;
if (bound_list)
bounds = TREE_VALUE (bound_list);
else if (TREE_CODE (array_type) == ARRAY_TYPE)
bounds = TYPE_INDEX_TYPE (TYPE_DOMAIN (array_type));
else if (expr && TREE_CODE (expr) == PARM_DECL
&& DECL_BY_COMPONENT_PTR_P (expr))
bounds = TREE_TYPE (field);
else
gcc_unreachable ();
min = convert (TREE_TYPE (field), TYPE_MIN_VALUE (bounds));
max = convert (TREE_TYPE (DECL_CHAIN (field)), TYPE_MAX_VALUE (bounds));
min = SUBSTITUTE_PLACEHOLDER_IN_EXPR (min, expr);
max = SUBSTITUTE_PLACEHOLDER_IN_EXPR (max, expr);
CONSTRUCTOR_APPEND_ELT (template_elts, field, min);
CONSTRUCTOR_APPEND_ELT (template_elts, DECL_CHAIN (field), max);
}
return gnat_build_constructor (template_type, template_elts);
}

static bool
type_for_vector_element_p (tree type)
{
machine_mode mode;
if (!INTEGRAL_TYPE_P (type)
&& !SCALAR_FLOAT_TYPE_P (type)
&& !FIXED_POINT_TYPE_P (type))
return false;
mode = TYPE_MODE (type);
if (GET_MODE_CLASS (mode) != MODE_INT
&& !SCALAR_FLOAT_MODE_P (mode)
&& !ALL_SCALAR_FIXED_POINT_MODE_P (mode))
return false;
return true;
}
static tree
build_vector_type_for_size (tree inner_type, tree size, tree attribute)
{
unsigned HOST_WIDE_INT size_int, inner_size_int;
int nunits;
if (!tree_fits_uhwi_p (size))
return NULL_TREE;
size_int = tree_to_uhwi (size);
if (!type_for_vector_element_p (inner_type))
{
if (attribute)
error ("invalid element type for attribute %qs",
IDENTIFIER_POINTER (attribute));
return NULL_TREE;
}
inner_size_int = tree_to_uhwi (TYPE_SIZE_UNIT (inner_type));
if (size_int % inner_size_int)
{
if (attribute)
error ("vector size not an integral multiple of component size");
return NULL_TREE;
}
if (size_int == 0)
{
if (attribute)
error ("zero vector size");
return NULL_TREE;
}
nunits = size_int / inner_size_int;
if (nunits & (nunits - 1))
{
if (attribute)
error ("number of components of vector not a power of two");
return NULL_TREE;
}
return build_vector_type (inner_type, nunits);
}
static tree
build_vector_type_for_array (tree array_type, tree attribute)
{
tree vector_type = build_vector_type_for_size (TREE_TYPE (array_type),
TYPE_SIZE_UNIT (array_type),
attribute);
if (!vector_type)
return NULL_TREE;
TYPE_REPRESENTATIVE_ARRAY (vector_type) = array_type;
return vector_type;
}

tree
build_unc_object_type (tree template_type, tree object_type, tree name,
bool debug_info_p)
{
tree decl;
tree type = make_node (RECORD_TYPE);
tree template_field
= create_field_decl (get_identifier ("BOUNDS"), template_type, type,
NULL_TREE, NULL_TREE, 0, 1);
tree array_field
= create_field_decl (get_identifier ("ARRAY"), object_type, type,
NULL_TREE, NULL_TREE, 0, 1);
TYPE_NAME (type) = name;
TYPE_CONTAINS_TEMPLATE_P (type) = 1;
DECL_CHAIN (template_field) = array_field;
finish_record_type (type, template_field, 0, true);
decl = create_type_decl (name, type, true, debug_info_p, Empty);
gnat_set_type_context (template_type, decl);
return type;
}
tree
build_unc_object_type_from_ptr (tree thin_fat_ptr_type, tree object_type,
tree name, bool debug_info_p)
{
tree template_type;
gcc_assert (TYPE_IS_FAT_OR_THIN_POINTER_P (thin_fat_ptr_type));
template_type
= (TYPE_IS_FAT_POINTER_P (thin_fat_ptr_type)
? TREE_TYPE (TREE_TYPE (DECL_CHAIN (TYPE_FIELDS (thin_fat_ptr_type))))
: TREE_TYPE (TYPE_FIELDS (TREE_TYPE (thin_fat_ptr_type))));
return
build_unc_object_type (template_type, object_type, name, debug_info_p);
}

void
update_pointer_to (tree old_type, tree new_type)
{
tree ptr = TYPE_POINTER_TO (old_type);
tree ref = TYPE_REFERENCE_TO (old_type);
tree t;
if (TYPE_MAIN_VARIANT (old_type) == old_type)
for (t = TYPE_NEXT_VARIANT (old_type); t; t = TYPE_NEXT_VARIANT (t))
update_pointer_to (t, new_type);
if (!ptr && !ref)
return;
new_type
= build_qualified_type (new_type,
TYPE_QUALS (old_type) | TYPE_QUALS (new_type));
if (old_type == new_type)
return;
if (TREE_CODE (new_type) != UNCONSTRAINED_ARRAY_TYPE)
{
tree new_ptr, new_ref;
if ((ptr && TREE_TYPE (ptr) == new_type)
|| (ref && TREE_TYPE (ref) == new_type))
return;
new_ptr = TYPE_POINTER_TO (new_type);
if (new_ptr)
{
while (TYPE_NEXT_PTR_TO (new_ptr))
new_ptr = TYPE_NEXT_PTR_TO (new_ptr);
TYPE_NEXT_PTR_TO (new_ptr) = ptr;
}
else
TYPE_POINTER_TO (new_type) = ptr;
for (; ptr; ptr = TYPE_NEXT_PTR_TO (ptr))
for (t = TYPE_MAIN_VARIANT (ptr); t; t = TYPE_NEXT_VARIANT (t))
{
TREE_TYPE (t) = new_type;
if (TYPE_NULL_BOUNDS (t))
TREE_TYPE (TREE_OPERAND (TYPE_NULL_BOUNDS (t), 0)) = new_type;
}
new_ref = TYPE_REFERENCE_TO (new_type);
if (new_ref)
{
while (TYPE_NEXT_REF_TO (new_ref))
new_ref = TYPE_NEXT_REF_TO (new_ref);
TYPE_NEXT_REF_TO (new_ref) = ref;
}
else
TYPE_REFERENCE_TO (new_type) = ref;
for (; ref; ref = TYPE_NEXT_REF_TO (ref))
for (t = TYPE_MAIN_VARIANT (ref); t; t = TYPE_NEXT_VARIANT (t))
TREE_TYPE (t) = new_type;
TYPE_POINTER_TO (old_type) = NULL_TREE;
TYPE_REFERENCE_TO (old_type) = NULL_TREE;
}
else
{
tree new_ptr = TYPE_POINTER_TO (new_type);
gcc_assert (TYPE_IS_FAT_POINTER_P (ptr));
if (TYPE_UNCONSTRAINED_ARRAY (ptr) == new_type)
return;
update_pointer_to
(TREE_TYPE (TREE_TYPE (TYPE_FIELDS (ptr))),
TREE_TYPE (TREE_TYPE (TYPE_FIELDS (new_ptr))));
update_pointer_to
(TREE_TYPE (TREE_TYPE (DECL_CHAIN (TYPE_FIELDS (ptr)))),
TREE_TYPE (TREE_TYPE (DECL_CHAIN (TYPE_FIELDS (new_ptr)))));
update_pointer_to (TYPE_OBJECT_RECORD_TYPE (old_type),
TYPE_OBJECT_RECORD_TYPE (new_type));
TYPE_POINTER_TO (old_type) = NULL_TREE;
TYPE_REFERENCE_TO (old_type) = NULL_TREE;
}
}

static tree
convert_to_fat_pointer (tree type, tree expr)
{
tree template_type = TREE_TYPE (TREE_TYPE (DECL_CHAIN (TYPE_FIELDS (type))));
tree p_array_type = TREE_TYPE (TYPE_FIELDS (type));
tree etype = TREE_TYPE (expr);
tree template_addr;
vec<constructor_elt, va_gc> *v;
vec_alloc (v, 2);
if (integer_zerop (expr))
{
tree ptr_template_type = TREE_TYPE (DECL_CHAIN (TYPE_FIELDS (type)));
tree null_bounds, t;
if (TYPE_NULL_BOUNDS (ptr_template_type))
null_bounds = TYPE_NULL_BOUNDS (ptr_template_type);
else
{
t = build_constructor (template_type, NULL);
TREE_CONSTANT (t) = TREE_STATIC (t) = 1;
null_bounds = build_unary_op (ADDR_EXPR, NULL_TREE, t);
SET_TYPE_NULL_BOUNDS (ptr_template_type, null_bounds);
}
CONSTRUCTOR_APPEND_ELT (v, TYPE_FIELDS (type),
fold_convert (p_array_type, null_pointer_node));
CONSTRUCTOR_APPEND_ELT (v, DECL_CHAIN (TYPE_FIELDS (type)), null_bounds);
t = build_constructor (type, v);
TREE_CONSTANT (t) = 0;
TREE_STATIC (t) = 1;
return t;
}
if (TYPE_IS_THIN_POINTER_P (etype))
{
tree field = TYPE_FIELDS (TREE_TYPE (etype));
expr = gnat_protect_expr (expr);
if (TYPE_UNCONSTRAINED_ARRAY (TREE_TYPE (etype)))
{
template_addr
= build_binary_op (POINTER_PLUS_EXPR, etype, expr,
fold_build1 (NEGATE_EXPR, sizetype,
byte_position
(DECL_CHAIN (field))));
template_addr
= fold_convert (TREE_TYPE (DECL_CHAIN (TYPE_FIELDS (type))),
template_addr);
}
else
{
expr = build_unary_op (INDIRECT_REF, NULL_TREE, expr);
template_addr
= build_unary_op (ADDR_EXPR, NULL_TREE,
build_component_ref (expr, field, false));
expr = build_unary_op (ADDR_EXPR, NULL_TREE,
build_component_ref (expr, DECL_CHAIN (field),
false));
}
}
else
template_addr
= build_unary_op (ADDR_EXPR, NULL_TREE,
build_template (template_type, TREE_TYPE (etype),
expr));
CONSTRUCTOR_APPEND_ELT (v, TYPE_FIELDS (type), convert (p_array_type, expr));
CONSTRUCTOR_APPEND_ELT (v, DECL_CHAIN (TYPE_FIELDS (type)), template_addr);
return gnat_build_constructor (type, v);
}

tree
convert (tree type, tree expr)
{
tree etype = TREE_TYPE (expr);
enum tree_code ecode = TREE_CODE (etype);
enum tree_code code = TREE_CODE (type);
if (etype == type)
return expr;
else if (code == RECORD_TYPE && ecode == RECORD_TYPE
&& TYPE_PADDING_P (type) && TYPE_PADDING_P (etype)
&& (!TREE_CONSTANT (TYPE_SIZE (type))
|| !TREE_CONSTANT (TYPE_SIZE (etype))
|| TYPE_MAIN_VARIANT (type) == TYPE_MAIN_VARIANT (etype)
|| TYPE_NAME (TREE_TYPE (TYPE_FIELDS (type)))
== TYPE_NAME (TREE_TYPE (TYPE_FIELDS (etype)))))
;
else if (code == RECORD_TYPE && TYPE_PADDING_P (type))
{
if (TREE_CODE (expr) == VIEW_CONVERT_EXPR
&& (!TREE_CONSTANT (TYPE_SIZE (type))
|| (ecode == RECORD_TYPE
&& TYPE_NAME (etype)
== TYPE_NAME (TREE_TYPE (TREE_OPERAND (expr, 0))))))
expr = TREE_OPERAND (expr, 0);
if (TREE_CODE (expr) == COMPONENT_REF
&& TYPE_IS_PADDING_P (TREE_TYPE (TREE_OPERAND (expr, 0)))
&& (!TREE_CONSTANT (TYPE_SIZE (type))
|| TYPE_MAIN_VARIANT (type)
== TYPE_MAIN_VARIANT (TREE_TYPE (TREE_OPERAND (expr, 0)))
|| (ecode == RECORD_TYPE
&& TYPE_NAME (etype)
== TYPE_NAME (TREE_TYPE (TYPE_FIELDS (type))))))
return convert (type, TREE_OPERAND (expr, 0));
if (ecode == RECORD_TYPE
&& CONTAINS_PLACEHOLDER_P (DECL_SIZE (TYPE_FIELDS (type)))
&& TYPE_MAIN_VARIANT (etype)
!= TYPE_MAIN_VARIANT (TREE_TYPE (TYPE_FIELDS (type))))
{
if (TREE_CODE (TYPE_SIZE (etype)) == INTEGER_CST)
expr = convert (maybe_pad_type (etype, TYPE_SIZE (type), 0, Empty,
false, false, false, true),
expr);
return unchecked_convert (type, expr, false);
}
if (ecode == ARRAY_TYPE
&& TREE_CODE (TREE_TYPE (TYPE_FIELDS (type))) == ARRAY_TYPE
&& !TREE_CONSTANT (TYPE_SIZE (etype))
&& !TREE_CONSTANT (TYPE_SIZE (type)))
return unchecked_convert (type,
convert (TREE_TYPE (TYPE_FIELDS (type)),
expr),
false);
tree t = convert (TREE_TYPE (TYPE_FIELDS (type)), expr);
if (TREE_CODE (t) == VIEW_CONVERT_EXPR
&& TREE_CODE (TREE_OPERAND (t, 0)) == CONSTRUCTOR
&& TREE_CONSTANT (TYPE_SIZE (TREE_TYPE (TREE_OPERAND (t, 0))))
&& tree_int_cst_equal (TYPE_SIZE (type),
TYPE_SIZE (TREE_TYPE (TREE_OPERAND (t, 0)))))
return build1 (VIEW_CONVERT_EXPR, type, TREE_OPERAND (t, 0));
vec<constructor_elt, va_gc> *v;
vec_alloc (v, 1);
CONSTRUCTOR_APPEND_ELT (v, TYPE_FIELDS (type), t);
return gnat_build_constructor (type, v);
}
else if (ecode == RECORD_TYPE && TYPE_PADDING_P (etype))
{
tree unpadded;
if (TREE_CODE (expr) == CONSTRUCTOR)
unpadded = CONSTRUCTOR_ELT (expr, 0)->value;
else
unpadded = build_component_ref (expr, TYPE_FIELDS (etype), false);
return convert (type, unpadded);
}
if (ecode == INTEGER_TYPE && TYPE_BIASED_REPRESENTATION_P (etype))
return convert (type, fold_build2 (PLUS_EXPR, TREE_TYPE (etype),
fold_convert (TREE_TYPE (etype), expr),
convert (TREE_TYPE (etype),
TYPE_MIN_VALUE (etype))));
if (ecode == RECORD_TYPE && TYPE_JUSTIFIED_MODULAR_P (etype)
&& code != UNCONSTRAINED_ARRAY_TYPE
&& TYPE_MAIN_VARIANT (type) != TYPE_MAIN_VARIANT (etype))
return
convert (type, build_component_ref (expr, TYPE_FIELDS (etype), false));
if (code == RECORD_TYPE && TYPE_CONTAINS_TEMPLATE_P (type))
{
tree obj_type = TREE_TYPE (DECL_CHAIN (TYPE_FIELDS (type)));
vec<constructor_elt, va_gc> *v;
vec_alloc (v, 2);
expr = maybe_unconstrained_array (expr);
CONSTRUCTOR_APPEND_ELT (v, TYPE_FIELDS (type),
build_template (TREE_TYPE (TYPE_FIELDS (type)),
obj_type, NULL_TREE));
if (expr)
CONSTRUCTOR_APPEND_ELT (v, DECL_CHAIN (TYPE_FIELDS (type)),
convert (obj_type, expr));
return gnat_build_constructor (type, v);
}
switch (TREE_CODE (expr))
{
case ERROR_MARK:
return expr;
case NULL_EXPR:
expr = copy_node (expr);
TREE_TYPE (expr) = type;
return expr;
case STRING_CST:
if (code == ecode && AGGREGATE_TYPE_P (etype)
&& !(TREE_CODE (TYPE_SIZE (etype)) == INTEGER_CST
&& TREE_CODE (TYPE_SIZE (type)) != INTEGER_CST))
{
expr = copy_node (expr);
TREE_TYPE (expr) = type;
return expr;
}
break;
case VECTOR_CST:
if (code == ecode && gnat_types_compatible_p (type, etype))
{
expr = copy_node (expr);
TREE_TYPE (expr) = type;
return expr;
}
break;
case CONSTRUCTOR:
if (code == ecode
&& (gnat_types_compatible_p (type, etype)
|| (code == RECORD_TYPE
&& TYPE_PADDING_P (type) && TYPE_PADDING_P (etype)
&& TREE_TYPE (TYPE_FIELDS (type))
== TREE_TYPE (TYPE_FIELDS (etype)))))
{
expr = copy_node (expr);
TREE_TYPE (expr) = type;
CONSTRUCTOR_ELTS (expr) = vec_safe_copy (CONSTRUCTOR_ELTS (expr));
return expr;
}
if (code == ecode
&& code == RECORD_TYPE
&& (TYPE_NAME (type) == TYPE_NAME (etype)
|| tree_int_cst_equal (TYPE_SIZE (type), TYPE_SIZE (etype))))
{
vec<constructor_elt, va_gc> *e = CONSTRUCTOR_ELTS (expr);
unsigned HOST_WIDE_INT len = vec_safe_length (e);
vec<constructor_elt, va_gc> *v;
vec_alloc (v, len);
tree efield = TYPE_FIELDS (etype), field = TYPE_FIELDS (type);
unsigned HOST_WIDE_INT idx;
tree index, value;
bool clear_constant = false;
FOR_EACH_CONSTRUCTOR_ELT(e, idx, index, value)
{
while (efield && field && !SAME_FIELD_P (efield, index))
{
efield = DECL_CHAIN (efield);
field = DECL_CHAIN (field);
}
if (!(efield && field && SAME_FIELD_P (efield, field)))
break;
constructor_elt elt
= {field, convert (TREE_TYPE (field), value)};
v->quick_push (elt);
if (!clear_constant
&& TREE_CONSTANT (expr)
&& !CONSTRUCTOR_BITFIELD_P (efield)
&& CONSTRUCTOR_BITFIELD_P (field)
&& !initializer_constant_valid_for_bitfield_p (value))
clear_constant = true;
efield = DECL_CHAIN (efield);
field = DECL_CHAIN (field);
}
if (idx == len)
{
expr = copy_node (expr);
TREE_TYPE (expr) = type;
CONSTRUCTOR_ELTS (expr) = v;
if (clear_constant)
TREE_CONSTANT (expr) = TREE_STATIC (expr) = 0;
return expr;
}
}
else if (code == VECTOR_TYPE
&& ecode == ARRAY_TYPE
&& gnat_types_compatible_p (TYPE_REPRESENTATIVE_ARRAY (type),
etype))
{
vec<constructor_elt, va_gc> *e = CONSTRUCTOR_ELTS (expr);
unsigned HOST_WIDE_INT len = vec_safe_length (e);
vec<constructor_elt, va_gc> *v;
unsigned HOST_WIDE_INT ix;
tree value;
if (TREE_CONSTANT (expr))
{
bool constant_p = true;
FOR_EACH_CONSTRUCTOR_VALUE (e, ix, value)
if (!CONSTANT_CLASS_P (value))
{
constant_p = false;
break;
}
if (constant_p)
return build_vector_from_ctor (type,
CONSTRUCTOR_ELTS (expr));
}
vec_alloc (v, len);
FOR_EACH_CONSTRUCTOR_VALUE (e, ix, value)
{
constructor_elt elt = {NULL_TREE, value};
v->quick_push (elt);
}
expr = copy_node (expr);
TREE_TYPE (expr) = type;
CONSTRUCTOR_ELTS (expr) = v;
return expr;
}
break;
case UNCONSTRAINED_ARRAY_REF:
expr = maybe_unconstrained_array (expr);
etype = TREE_TYPE (expr);
ecode = TREE_CODE (etype);
break;
case VIEW_CONVERT_EXPR:
{
tree op0 = TREE_OPERAND (expr, 0);
if (type == TREE_TYPE (op0))
return op0;
if ((AGGREGATE_TYPE_P (type) && AGGREGATE_TYPE_P (etype))
|| (VECTOR_TYPE_P (type) && VECTOR_TYPE_P (etype)))
{
if (gnat_types_compatible_p (type, etype))
return build1 (VIEW_CONVERT_EXPR, type, op0);
else if (!TYPE_IS_FAT_POINTER_P (type)
&& !TYPE_IS_FAT_POINTER_P (etype))
return convert (type, op0);
}
break;
}
default:
break;
}
if (TYPE_IS_FAT_POINTER_P (type) && !TYPE_IS_FAT_POINTER_P (etype))
return convert_to_fat_pointer (type, expr);
else if ((code == ecode
&& (AGGREGATE_TYPE_P (type) || VECTOR_TYPE_P (type))
&& gnat_types_compatible_p (type, etype))
|| (code == VECTOR_TYPE
&& ecode == ARRAY_TYPE
&& gnat_types_compatible_p (TYPE_REPRESENTATIVE_ARRAY (type),
etype)))
return build1 (VIEW_CONVERT_EXPR, type, expr);
else if (ecode == RECORD_TYPE && code == RECORD_TYPE
&& TYPE_ALIGN_OK (etype) && TYPE_ALIGN_OK (type)
&& !type_annotate_only)
{
tree child_etype = etype;
do {
tree field = TYPE_FIELDS (child_etype);
if (DECL_NAME (field) == parent_name_id && TREE_TYPE (field) == type)
return build_component_ref (expr, field, false);
child_etype = TREE_TYPE (field);
} while (TREE_CODE (child_etype) == RECORD_TYPE);
}
else if (ecode == RECORD_TYPE && code == RECORD_TYPE
&& smaller_form_type_p (etype, type))
{
expr = convert (maybe_pad_type (etype, TYPE_SIZE (type), 0, Empty,
false, false, false, true),
expr);
return build1 (VIEW_CONVERT_EXPR, type, expr);
}
else if (TYPE_MAIN_VARIANT (type) == TYPE_MAIN_VARIANT (etype))
return fold_convert (type, expr);
switch (code)
{
case VOID_TYPE:
return fold_build1 (CONVERT_EXPR, type, expr);
case INTEGER_TYPE:
if (TYPE_HAS_ACTUAL_BOUNDS_P (type)
&& (ecode == ARRAY_TYPE || ecode == UNCONSTRAINED_ARRAY_TYPE
|| (ecode == RECORD_TYPE && TYPE_CONTAINS_TEMPLATE_P (etype))))
return unchecked_convert (type, expr, false);
if (TYPE_BIASED_REPRESENTATION_P (type))
return fold_convert (type,
fold_build2 (MINUS_EXPR, TREE_TYPE (type),
convert (TREE_TYPE (type), expr),
convert (TREE_TYPE (type),
TYPE_MIN_VALUE (type))));
case ENUMERAL_TYPE:
case BOOLEAN_TYPE:
if (code == INTEGER_TYPE
&& ecode == INTEGER_TYPE
&& TYPE_PRECISION (type) < TYPE_PRECISION (etype)
&& (TREE_CODE (expr) == PLUS_EXPR || TREE_CODE (expr) == MINUS_EXPR))
{
tree op0 = get_unwidened (TREE_OPERAND (expr, 0), type);
if ((TREE_CODE (TREE_TYPE (op0)) == INTEGER_TYPE
&& TYPE_BIASED_REPRESENTATION_P (TREE_TYPE (op0)))
|| CONTAINS_PLACEHOLDER_P (expr))
return build1 (NOP_EXPR, type, expr);
}
return fold (convert_to_integer (type, expr));
case POINTER_TYPE:
case REFERENCE_TYPE:
if (TYPE_IS_THIN_POINTER_P (etype) && TYPE_IS_THIN_POINTER_P (type))
{
tree etype_pos
= TYPE_UNCONSTRAINED_ARRAY (TREE_TYPE (etype))
? byte_position (DECL_CHAIN (TYPE_FIELDS (TREE_TYPE (etype))))
: size_zero_node;
tree type_pos
= TYPE_UNCONSTRAINED_ARRAY (TREE_TYPE (type))
? byte_position (DECL_CHAIN (TYPE_FIELDS (TREE_TYPE (type))))
: size_zero_node;
tree byte_diff = size_diffop (type_pos, etype_pos);
expr = build1 (NOP_EXPR, type, expr);
if (integer_zerop (byte_diff))
return expr;
return build_binary_op (POINTER_PLUS_EXPR, type, expr,
fold_convert (sizetype, byte_diff));
}
if (TYPE_IS_FAT_POINTER_P (etype))
expr = build_component_ref (expr, TYPE_FIELDS (etype), false);
return fold (convert_to_pointer (type, expr));
case REAL_TYPE:
return fold (convert_to_real (type, expr));
case RECORD_TYPE:
if (TYPE_JUSTIFIED_MODULAR_P (type) && !AGGREGATE_TYPE_P (etype))
{
vec<constructor_elt, va_gc> *v;
vec_alloc (v, 1);
CONSTRUCTOR_APPEND_ELT (v, TYPE_FIELDS (type),
convert (TREE_TYPE (TYPE_FIELDS (type)),
expr));
return gnat_build_constructor (type, v);
}
return unchecked_convert (type, expr, false);
case ARRAY_TYPE:
if (TREE_CODE (expr) == INDIRECT_REF
&& ecode == ARRAY_TYPE
&& TREE_TYPE (etype) == TREE_TYPE (type))
{
tree ptr_type = build_pointer_type (type);
tree t = build_unary_op (INDIRECT_REF, NULL_TREE,
fold_convert (ptr_type,
TREE_OPERAND (expr, 0)));
TREE_READONLY (t) = TREE_READONLY (expr);
TREE_THIS_NOTRAP (t) = TREE_THIS_NOTRAP (expr);
return t;
}
return unchecked_convert (type, expr, false);
case UNION_TYPE:
return unchecked_convert (type, expr, false);
case UNCONSTRAINED_ARRAY_TYPE:
if (ecode == VECTOR_TYPE)
{
expr = convert (TYPE_REPRESENTATIVE_ARRAY (etype), expr);
etype = TREE_TYPE (expr);
ecode = TREE_CODE (etype);
}
if (ecode == ARRAY_TYPE
|| (ecode == INTEGER_TYPE && TYPE_HAS_ACTUAL_BOUNDS_P (etype))
|| (ecode == RECORD_TYPE && TYPE_CONTAINS_TEMPLATE_P (etype))
|| (ecode == RECORD_TYPE && TYPE_JUSTIFIED_MODULAR_P (etype)))
return
build_unary_op
(INDIRECT_REF, NULL_TREE,
convert_to_fat_pointer (TREE_TYPE (type),
build_unary_op (ADDR_EXPR,
NULL_TREE, expr)));
else if (ecode == UNCONSTRAINED_ARRAY_TYPE)
return
build_unary_op (INDIRECT_REF, NULL_TREE,
convert (TREE_TYPE (type),
build_unary_op (ADDR_EXPR,
NULL_TREE, expr)));
else
gcc_unreachable ();
case COMPLEX_TYPE:
return fold (convert_to_complex (type, expr));
default:
gcc_unreachable ();
}
}
tree
convert_to_index_type (tree expr)
{
enum tree_code code = TREE_CODE (expr);
tree type = TREE_TYPE (expr);
if (TYPE_UNSIGNED (type) || !optimize)
return convert (sizetype, expr);
switch (code)
{
case VAR_DECL:
if (DECL_LOOP_PARM_P (expr) && DECL_INDUCTION_VAR (expr))
expr = DECL_INDUCTION_VAR (expr);
break;
CASE_CONVERT:
{
tree otype = TREE_TYPE (TREE_OPERAND (expr, 0));
if (TYPE_PRECISION (type) != TYPE_PRECISION (otype)
|| TYPE_UNSIGNED (type) != TYPE_UNSIGNED (otype))
break;
}
case NON_LVALUE_EXPR:
return fold_build1 (code, sizetype,
convert_to_index_type (TREE_OPERAND (expr, 0)));
case PLUS_EXPR:
case MINUS_EXPR:
case MULT_EXPR:
return fold_build2 (code, sizetype,
convert_to_index_type (TREE_OPERAND (expr, 0)),
convert_to_index_type (TREE_OPERAND (expr, 1)));
case COMPOUND_EXPR:
return fold_build2 (code, sizetype, TREE_OPERAND (expr, 0),
convert_to_index_type (TREE_OPERAND (expr, 1)));
case COND_EXPR:
return fold_build3 (code, sizetype, TREE_OPERAND (expr, 0),
convert_to_index_type (TREE_OPERAND (expr, 1)),
convert_to_index_type (TREE_OPERAND (expr, 2)));
default:
break;
}
return convert (sizetype, expr);
}

tree
remove_conversions (tree exp, bool true_address)
{
switch (TREE_CODE (exp))
{
case CONSTRUCTOR:
if (true_address
&& TREE_CODE (TREE_TYPE (exp)) == RECORD_TYPE
&& TYPE_JUSTIFIED_MODULAR_P (TREE_TYPE (exp)))
return
remove_conversions (CONSTRUCTOR_ELT (exp, 0)->value, true);
break;
case COMPONENT_REF:
if (TYPE_IS_PADDING_P (TREE_TYPE (TREE_OPERAND (exp, 0))))
return remove_conversions (TREE_OPERAND (exp, 0), true_address);
break;
CASE_CONVERT:
case VIEW_CONVERT_EXPR:
case NON_LVALUE_EXPR:
return remove_conversions (TREE_OPERAND (exp, 0), true_address);
default:
break;
}
return exp;
}

tree
maybe_unconstrained_array (tree exp)
{
enum tree_code code = TREE_CODE (exp);
tree type = TREE_TYPE (exp);
switch (TREE_CODE (type))
{
case UNCONSTRAINED_ARRAY_TYPE:
if (code == UNCONSTRAINED_ARRAY_REF)
{
const bool read_only = TREE_READONLY (exp);
const bool no_trap = TREE_THIS_NOTRAP (exp);
exp = TREE_OPERAND (exp, 0);
type = TREE_TYPE (exp);
if (TREE_CODE (exp) == COND_EXPR)
{
tree op1
= build_unary_op (INDIRECT_REF, NULL_TREE,
build_component_ref (TREE_OPERAND (exp, 1),
TYPE_FIELDS (type),
false));
tree op2
= build_unary_op (INDIRECT_REF, NULL_TREE,
build_component_ref (TREE_OPERAND (exp, 2),
TYPE_FIELDS (type),
false));
exp = build3 (COND_EXPR,
TREE_TYPE (TREE_TYPE (TYPE_FIELDS (type))),
TREE_OPERAND (exp, 0), op1, op2);
}
else
{
exp = build_unary_op (INDIRECT_REF, NULL_TREE,
build_component_ref (exp,
TYPE_FIELDS (type),
false));
TREE_READONLY (exp) = read_only;
TREE_THIS_NOTRAP (exp) = no_trap;
}
}
else if (code == NULL_EXPR)
exp = build1 (NULL_EXPR,
TREE_TYPE (TREE_TYPE (TYPE_FIELDS (TREE_TYPE (type)))),
TREE_OPERAND (exp, 0));
break;
case RECORD_TYPE:
if (TYPE_PADDING_P (type)
&& TREE_CODE (TREE_TYPE (TYPE_FIELDS (type))) == RECORD_TYPE
&& TYPE_CONTAINS_TEMPLATE_P (TREE_TYPE (TYPE_FIELDS (type))))
{
exp = convert (TREE_TYPE (TYPE_FIELDS (type)), exp);
code = TREE_CODE (exp);
type = TREE_TYPE (exp);
}
if (TYPE_CONTAINS_TEMPLATE_P (type))
{
if (code == CONSTRUCTOR && CONSTRUCTOR_NELTS (exp) < 2)
return NULL_TREE;
exp = build_component_ref (exp, DECL_CHAIN (TYPE_FIELDS (type)),
false);
type = TREE_TYPE (exp);
if (TYPE_IS_PADDING_P (type))
exp = convert (TREE_TYPE (TYPE_FIELDS (type)), exp);
}
break;
default:
break;
}
return exp;
}

static bool
can_fold_for_view_convert_p (tree expr)
{
tree t1, t2;
if (TREE_CODE (expr) != NOP_EXPR)
return true;
t1 = TREE_TYPE (expr);
t2 = TREE_TYPE (TREE_OPERAND (expr, 0));
if (!(INTEGRAL_TYPE_P (t1) && INTEGRAL_TYPE_P (t2)))
return true;
if (TYPE_PRECISION (t1) == TYPE_PRECISION (t2)
&& operand_equal_p (rm_size (t1), rm_size (t2), 0))
return true;
return false;
}
tree
unchecked_convert (tree type, tree expr, bool notrunc_p)
{
tree etype = TREE_TYPE (expr);
enum tree_code ecode = TREE_CODE (etype);
enum tree_code code = TREE_CODE (type);
const bool ebiased
= (ecode == INTEGER_TYPE && TYPE_BIASED_REPRESENTATION_P (etype));
const bool biased
= (code == INTEGER_TYPE && TYPE_BIASED_REPRESENTATION_P (type));
const bool ereverse
= (AGGREGATE_TYPE_P (etype) && TYPE_REVERSE_STORAGE_ORDER (etype));
const bool reverse
= (AGGREGATE_TYPE_P (type) && TYPE_REVERSE_STORAGE_ORDER (type));
tree tem;
int c = 0;
if (etype == type)
return expr;
if (((INTEGRAL_TYPE_P (type)
|| (POINTER_TYPE_P (type) && !TYPE_IS_THIN_POINTER_P (type))
|| (code == RECORD_TYPE && TYPE_JUSTIFIED_MODULAR_P (type)))
&& (INTEGRAL_TYPE_P (etype)
|| (POINTER_TYPE_P (etype) && !TYPE_IS_THIN_POINTER_P (etype))
|| (ecode == RECORD_TYPE && TYPE_JUSTIFIED_MODULAR_P (etype))))
|| code == UNCONSTRAINED_ARRAY_TYPE)
{
if (ebiased)
{
tree ntype = copy_type (etype);
TYPE_BIASED_REPRESENTATION_P (ntype) = 0;
TYPE_MAIN_VARIANT (ntype) = ntype;
expr = build1 (NOP_EXPR, ntype, expr);
}
if (biased)
{
tree rtype = copy_type (type);
TYPE_BIASED_REPRESENTATION_P (rtype) = 0;
TYPE_MAIN_VARIANT (rtype) = rtype;
expr = convert (rtype, expr);
expr = build1 (NOP_EXPR, type, expr);
}
else
expr = convert (type, expr);
}
else if ((INTEGRAL_TYPE_P (type)
&& TYPE_RM_SIZE (type)
&& ((c = tree_int_cst_compare (TYPE_RM_SIZE (type),
TYPE_SIZE (type))) < 0
|| ereverse))
|| (SCALAR_FLOAT_TYPE_P (type) && ereverse))
{
tree rec_type = make_node (RECORD_TYPE);
tree field_type, field;
TYPE_REVERSE_STORAGE_ORDER (rec_type) = ereverse;
if (c < 0)
{
const unsigned HOST_WIDE_INT prec
= TREE_INT_CST_LOW (TYPE_RM_SIZE (type));
if (type_unsigned_for_rm (type))
field_type = make_unsigned_type (prec);
else
field_type = make_signed_type (prec);
SET_TYPE_RM_SIZE (field_type, TYPE_RM_SIZE (type));
}
else
field_type = type;
field = create_field_decl (get_identifier ("OBJ"), field_type, rec_type,
NULL_TREE, bitsize_zero_node, c < 0, 0);
finish_record_type (rec_type, field, 1, false);
expr = unchecked_convert (rec_type, expr, notrunc_p);
expr = build_component_ref (expr, field, false);
expr = fold_build1 (NOP_EXPR, type, expr);
}
else if ((INTEGRAL_TYPE_P (etype)
&& TYPE_RM_SIZE (etype)
&& ((c = tree_int_cst_compare (TYPE_RM_SIZE (etype),
TYPE_SIZE (etype))) < 0
|| reverse))
|| (SCALAR_FLOAT_TYPE_P (etype) && reverse))
{
tree rec_type = make_node (RECORD_TYPE);
vec<constructor_elt, va_gc> *v;
vec_alloc (v, 1);
tree field_type, field;
TYPE_REVERSE_STORAGE_ORDER (rec_type) = reverse;
if (c < 0)
{
const unsigned HOST_WIDE_INT prec
= TREE_INT_CST_LOW (TYPE_RM_SIZE (etype));
if (type_unsigned_for_rm (etype))
field_type = make_unsigned_type (prec);
else
field_type = make_signed_type (prec);
SET_TYPE_RM_SIZE (field_type, TYPE_RM_SIZE (etype));
}
else
field_type = etype;
field = create_field_decl (get_identifier ("OBJ"), field_type, rec_type,
NULL_TREE, bitsize_zero_node, c < 0, 0);
finish_record_type (rec_type, field, 1, false);
expr = fold_build1 (NOP_EXPR, field_type, expr);
CONSTRUCTOR_APPEND_ELT (v, field, expr);
expr = gnat_build_constructor (rec_type, v);
expr = unchecked_convert (type, expr, notrunc_p);
}
else if (!REFERENCE_CLASS_P (expr)
&& !AGGREGATE_TYPE_P (etype)
&& TREE_CODE (TYPE_SIZE (type)) == INTEGER_CST
&& (c = tree_int_cst_compare (TYPE_SIZE (etype), TYPE_SIZE (type))))
{
if (c < 0)
{
expr = convert (maybe_pad_type (etype, TYPE_SIZE (type), 0, Empty,
false, false, false, true),
expr);
expr = unchecked_convert (type, expr, notrunc_p);
}
else
{
tree rec_type = maybe_pad_type (type, TYPE_SIZE (etype), 0, Empty,
false, false, false, true);
expr = unchecked_convert (rec_type, expr, notrunc_p);
expr = build_component_ref (expr, TYPE_FIELDS (rec_type), false);
}
}
else if (ecode == code && code == UNCONSTRAINED_ARRAY_TYPE)
expr = build_unary_op (INDIRECT_REF, NULL_TREE,
build1 (VIEW_CONVERT_EXPR, TREE_TYPE (type),
build_unary_op (ADDR_EXPR, NULL_TREE,
expr)));
else if (code == VECTOR_TYPE
&& ecode == ARRAY_TYPE
&& gnat_types_compatible_p (TYPE_REPRESENTATIVE_ARRAY (type),
etype))
expr = convert (type, expr);
else if (code == VECTOR_TYPE
&& ecode == ARRAY_TYPE
&& (tem = build_vector_type_for_array (etype, NULL_TREE)))
{
expr = convert (tem, expr);
return unchecked_convert (type, expr, notrunc_p);
}
else if (TREE_CODE (expr) == CONSTRUCTOR
&& code == RECORD_TYPE
&& TYPE_ALIGN (etype) < TYPE_ALIGN (type))
{
expr = convert (maybe_pad_type (etype, NULL_TREE, TYPE_ALIGN (type),
Empty, false, false, false, true),
expr);
return unchecked_convert (type, expr, notrunc_p);
}
else
{
expr = maybe_unconstrained_array (expr);
etype = TREE_TYPE (expr);
ecode = TREE_CODE (etype);
if (can_fold_for_view_convert_p (expr))
expr = fold_build1 (VIEW_CONVERT_EXPR, type, expr);
else
expr = build1 (VIEW_CONVERT_EXPR, type, expr);
}
if (!notrunc_p
&& !biased
&& INTEGRAL_TYPE_P (type)
&& TYPE_RM_SIZE (type)
&& tree_int_cst_compare (TYPE_RM_SIZE (type), TYPE_SIZE (type)) < 0
&& !(INTEGRAL_TYPE_P (etype)
&& type_unsigned_for_rm (type) == type_unsigned_for_rm (etype)
&& (type_unsigned_for_rm (type)
|| tree_int_cst_compare (TYPE_RM_SIZE (type),
TYPE_RM_SIZE (etype)
? TYPE_RM_SIZE (etype)
: TYPE_SIZE (etype)) == 0)))
{
if (integer_zerop (TYPE_RM_SIZE (type)))
expr = build_int_cst (type, 0);
else
{
tree base_type
= gnat_type_for_size (TREE_INT_CST_LOW (TYPE_SIZE (type)),
type_unsigned_for_rm (type));
tree shift_expr
= convert (base_type,
size_binop (MINUS_EXPR,
TYPE_SIZE (type), TYPE_RM_SIZE (type)));
expr
= convert (type,
build_binary_op (RSHIFT_EXPR, base_type,
build_binary_op (LSHIFT_EXPR, base_type,
convert (base_type,
expr),
shift_expr),
shift_expr));
}
}
if (TREE_CODE (expr) == INTEGER_CST)
TREE_OVERFLOW (expr) = 0;
if (TREE_CODE (expr) == VIEW_CONVERT_EXPR
&& !operand_equal_p (TYPE_SIZE_UNIT (type), TYPE_SIZE_UNIT (etype),
OEP_ONLY_CONST))
TREE_CONSTANT (expr) = 0;
return expr;
}

enum tree_code
tree_code_for_record_type (Entity_Id gnat_type)
{
Node_Id component_list, component;
if (!Is_Unchecked_Union (gnat_type))
return RECORD_TYPE;
gnat_type = Implementation_Base_Type (gnat_type);
component_list
= Component_List (Type_Definition (Declaration_Node (gnat_type)));
for (component = First_Non_Pragma (Component_Items (component_list));
Present (component);
component = Next_Non_Pragma (component))
if (Ekind (Defining_Entity (component)) == E_Component)
return RECORD_TYPE;
return UNION_TYPE;
}
bool
is_double_float_or_array (Entity_Id gnat_type, bool *align_clause)
{
gnat_type = Underlying_Type (gnat_type);
*align_clause = Present (Alignment_Clause (gnat_type));
if (Is_Array_Type (gnat_type))
{
gnat_type = Underlying_Type (Component_Type (gnat_type));
if (Present (Alignment_Clause (gnat_type)))
*align_clause = true;
}
if (!Is_Floating_Point_Type (gnat_type))
return false;
if (UI_To_Int (Esize (gnat_type)) != 64)
return false;
return true;
}
bool
is_double_scalar_or_array (Entity_Id gnat_type, bool *align_clause)
{
gnat_type = Underlying_Type (gnat_type);
*align_clause = Present (Alignment_Clause (gnat_type));
if (Is_Array_Type (gnat_type))
{
gnat_type = Underlying_Type (Component_Type (gnat_type));
if (Present (Alignment_Clause (gnat_type)))
*align_clause = true;
}
if (!Is_Scalar_Type (gnat_type))
return false;
if (UI_To_Int (Esize (gnat_type)) < 64)
return false;
return true;
}
bool
type_for_nonaliased_component_p (tree gnu_type)
{
if (must_pass_by_ref (gnu_type) || default_pass_by_ref (gnu_type))
return false;
if (AGGREGATE_TYPE_P (gnu_type))
return false;
return true;
}
bool
smaller_form_type_p (tree type, tree orig_type)
{
tree size, osize;
if (TYPE_MAIN_VARIANT (type) == TYPE_MAIN_VARIANT (orig_type))
return false;
if (TYPE_NAME (type) != TYPE_NAME (orig_type))
return false;
size = TYPE_SIZE (type);
osize = TYPE_SIZE (orig_type);
if (!(TREE_CODE (size) == INTEGER_CST && TREE_CODE (osize) == INTEGER_CST))
return false;
return tree_int_cst_lt (size, osize) != 0;
}
bool
can_materialize_object_renaming_p (Node_Id expr)
{
while (true)
{
expr = Original_Node (expr);
switch Nkind (expr)
{
case N_Identifier:
case N_Expanded_Name:
if (!Present (Renamed_Object (Entity (expr))))
return true;
expr = Renamed_Object (Entity (expr));
break;
case N_Selected_Component:
{
if (Is_Packed (Underlying_Type (Etype (Prefix (expr)))))
return false;
const Uint bitpos
= Normalized_First_Bit (Entity (Selector_Name (expr)));
if (!UI_Is_In_Int_Range (bitpos)
|| (bitpos != UI_No_Uint && bitpos != UI_From_Int (0)))
return false;
expr = Prefix (expr);
break;
}
case N_Indexed_Component:
case N_Slice:
{
const Entity_Id t = Underlying_Type (Etype (Prefix (expr)));
if (Is_Array_Type (t) && Present (Packed_Array_Impl_Type (t)))
return false;
expr = Prefix (expr);
break;
}
case N_Explicit_Dereference:
expr = Prefix (expr);
break;
default:
return true;
};
}
}
static GTY (()) tree dummy_global;
void
gnat_write_global_declarations (void)
{
unsigned int i;
tree iter;
if (first_global_object_name)
{
struct varpool_node *node;
char *label;
ASM_FORMAT_PRIVATE_NAME (label, first_global_object_name, 0);
dummy_global
= build_decl (BUILTINS_LOCATION, VAR_DECL, get_identifier (label),
void_type_node);
DECL_HARD_REGISTER (dummy_global) = 1;
TREE_STATIC (dummy_global) = 1;
node = varpool_node::get_create (dummy_global);
node->definition = 1;
node->force_output = 1;
if (types_used_by_cur_var_decl)
while (!types_used_by_cur_var_decl->is_empty ())
{
tree t = types_used_by_cur_var_decl->pop ();
types_used_by_var_decl_insert (t, dummy_global);
}
}
FOR_EACH_VEC_SAFE_ELT (global_decls, i, iter)
if (TREE_CODE (iter) == TYPE_DECL && !DECL_IGNORED_P (iter))
debug_hooks->type_decl (iter, false);
FOR_EACH_VEC_SAFE_ELT (global_decls, i, iter)
if (TREE_CODE (iter) == FUNCTION_DECL
&& DECL_EXTERNAL (iter)
&& DECL_INITIAL (iter) == NULL
&& !DECL_IGNORED_P (iter)
&& DECL_FUNCTION_IS_DEF (iter))
debug_hooks->early_global_decl (iter);
FOR_EACH_VEC_SAFE_ELT (global_decls, i, iter)
if (TREE_CODE (iter) == VAR_DECL
&& (!DECL_EXTERNAL (iter) || !DECL_IGNORED_P (iter)))
rest_of_decl_compilation (iter, true, 0);
FOR_EACH_VEC_SAFE_ELT (global_decls, i, iter)
if (TREE_CODE (iter) == IMPORTED_DECL && !DECL_IGNORED_P (iter))
debug_hooks->imported_module_or_decl (iter, DECL_NAME (iter),
DECL_CONTEXT (iter), false, false);
}
tree
builtin_decl_for (tree name)
{
unsigned i;
tree decl;
FOR_EACH_VEC_SAFE_ELT (builtin_decls, i, decl)
if (DECL_NAME (decl) == name)
return decl;
return NULL_TREE;
}
enum c_tree_index
{
CTI_SIGNED_SIZE_TYPE, 
CTI_STRING_TYPE,
CTI_CONST_STRING_TYPE,
CTI_MAX
};
static tree c_global_trees[CTI_MAX];
#define signed_size_type_node	c_global_trees[CTI_SIGNED_SIZE_TYPE]
#define string_type_node	c_global_trees[CTI_STRING_TYPE]
#define const_string_type_node	c_global_trees[CTI_CONST_STRING_TYPE]
#define wint_type_node    void_type_node
#define intmax_type_node  void_type_node
#define uintmax_type_node void_type_node
static tree
builtin_type_for_size (int size, bool unsignedp)
{
tree type = gnat_type_for_size (size, unsignedp);
return type ? type : error_mark_node;
}
static void
install_builtin_elementary_types (void)
{
signed_size_type_node = gnat_signed_type_for (size_type_node);
pid_type_node = integer_type_node;
string_type_node = build_pointer_type (char_type_node);
const_string_type_node
= build_pointer_type (build_qualified_type
(char_type_node, TYPE_QUAL_CONST));
}
enum c_builtin_type
{
#define DEF_PRIMITIVE_TYPE(NAME, VALUE) NAME,
#define DEF_FUNCTION_TYPE_0(NAME, RETURN) NAME,
#define DEF_FUNCTION_TYPE_1(NAME, RETURN, ARG1) NAME,
#define DEF_FUNCTION_TYPE_2(NAME, RETURN, ARG1, ARG2) NAME,
#define DEF_FUNCTION_TYPE_3(NAME, RETURN, ARG1, ARG2, ARG3) NAME,
#define DEF_FUNCTION_TYPE_4(NAME, RETURN, ARG1, ARG2, ARG3, ARG4) NAME,
#define DEF_FUNCTION_TYPE_5(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5) NAME,
#define DEF_FUNCTION_TYPE_6(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6) NAME,
#define DEF_FUNCTION_TYPE_7(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7) NAME,
#define DEF_FUNCTION_TYPE_8(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8) NAME,
#define DEF_FUNCTION_TYPE_9(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8, ARG9) NAME,
#define DEF_FUNCTION_TYPE_10(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8, ARG9, ARG10) NAME,
#define DEF_FUNCTION_TYPE_11(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8, ARG9, ARG10, ARG11) NAME,
#define DEF_FUNCTION_TYPE_VAR_0(NAME, RETURN) NAME,
#define DEF_FUNCTION_TYPE_VAR_1(NAME, RETURN, ARG1) NAME,
#define DEF_FUNCTION_TYPE_VAR_2(NAME, RETURN, ARG1, ARG2) NAME,
#define DEF_FUNCTION_TYPE_VAR_3(NAME, RETURN, ARG1, ARG2, ARG3) NAME,
#define DEF_FUNCTION_TYPE_VAR_4(NAME, RETURN, ARG1, ARG2, ARG3, ARG4) NAME,
#define DEF_FUNCTION_TYPE_VAR_5(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5) \
NAME,
#define DEF_FUNCTION_TYPE_VAR_6(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6) NAME,
#define DEF_FUNCTION_TYPE_VAR_7(NAME, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7) NAME,
#define DEF_POINTER_TYPE(NAME, TYPE) NAME,
#include "builtin-types.def"
#undef DEF_PRIMITIVE_TYPE
#undef DEF_FUNCTION_TYPE_0
#undef DEF_FUNCTION_TYPE_1
#undef DEF_FUNCTION_TYPE_2
#undef DEF_FUNCTION_TYPE_3
#undef DEF_FUNCTION_TYPE_4
#undef DEF_FUNCTION_TYPE_5
#undef DEF_FUNCTION_TYPE_6
#undef DEF_FUNCTION_TYPE_7
#undef DEF_FUNCTION_TYPE_8
#undef DEF_FUNCTION_TYPE_9
#undef DEF_FUNCTION_TYPE_10
#undef DEF_FUNCTION_TYPE_11
#undef DEF_FUNCTION_TYPE_VAR_0
#undef DEF_FUNCTION_TYPE_VAR_1
#undef DEF_FUNCTION_TYPE_VAR_2
#undef DEF_FUNCTION_TYPE_VAR_3
#undef DEF_FUNCTION_TYPE_VAR_4
#undef DEF_FUNCTION_TYPE_VAR_5
#undef DEF_FUNCTION_TYPE_VAR_6
#undef DEF_FUNCTION_TYPE_VAR_7
#undef DEF_POINTER_TYPE
BT_LAST
};
typedef enum c_builtin_type builtin_type;
static GTY(()) tree builtin_types[(int) BT_LAST + 1];
static void
def_fn_type (builtin_type def, builtin_type ret, bool var, int n, ...)
{
tree t;
tree *args = XALLOCAVEC (tree, n);
va_list list;
int i;
va_start (list, n);
for (i = 0; i < n; ++i)
{
builtin_type a = (builtin_type) va_arg (list, int);
t = builtin_types[a];
if (t == error_mark_node)
goto egress;
args[i] = t;
}
t = builtin_types[ret];
if (t == error_mark_node)
goto egress;
if (var)
t = build_varargs_function_type_array (t, n, args);
else
t = build_function_type_array (t, n, args);
egress:
builtin_types[def] = t;
va_end (list);
}
static void
install_builtin_function_types (void)
{
tree va_list_ref_type_node;
tree va_list_arg_type_node;
if (TREE_CODE (va_list_type_node) == ARRAY_TYPE)
{
va_list_arg_type_node = va_list_ref_type_node =
build_pointer_type (TREE_TYPE (va_list_type_node));
}
else
{
va_list_arg_type_node = va_list_type_node;
va_list_ref_type_node = build_reference_type (va_list_type_node);
}
#define DEF_PRIMITIVE_TYPE(ENUM, VALUE) \
builtin_types[ENUM] = VALUE;
#define DEF_FUNCTION_TYPE_0(ENUM, RETURN) \
def_fn_type (ENUM, RETURN, 0, 0);
#define DEF_FUNCTION_TYPE_1(ENUM, RETURN, ARG1) \
def_fn_type (ENUM, RETURN, 0, 1, ARG1);
#define DEF_FUNCTION_TYPE_2(ENUM, RETURN, ARG1, ARG2) \
def_fn_type (ENUM, RETURN, 0, 2, ARG1, ARG2);
#define DEF_FUNCTION_TYPE_3(ENUM, RETURN, ARG1, ARG2, ARG3) \
def_fn_type (ENUM, RETURN, 0, 3, ARG1, ARG2, ARG3);
#define DEF_FUNCTION_TYPE_4(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4) \
def_fn_type (ENUM, RETURN, 0, 4, ARG1, ARG2, ARG3, ARG4);
#define DEF_FUNCTION_TYPE_5(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5)	\
def_fn_type (ENUM, RETURN, 0, 5, ARG1, ARG2, ARG3, ARG4, ARG5);
#define DEF_FUNCTION_TYPE_6(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6)					\
def_fn_type (ENUM, RETURN, 0, 6, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6);
#define DEF_FUNCTION_TYPE_7(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7)					\
def_fn_type (ENUM, RETURN, 0, 7, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7);
#define DEF_FUNCTION_TYPE_8(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8)				\
def_fn_type (ENUM, RETURN, 0, 8, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6,	\
ARG7, ARG8);
#define DEF_FUNCTION_TYPE_9(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7, ARG8, ARG9)			\
def_fn_type (ENUM, RETURN, 0, 9, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6,	\
ARG7, ARG8, ARG9);
#define DEF_FUNCTION_TYPE_10(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5,\
ARG6, ARG7, ARG8, ARG9, ARG10)		\
def_fn_type (ENUM, RETURN, 0, 10, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6,	\
ARG7, ARG8, ARG9, ARG10);
#define DEF_FUNCTION_TYPE_11(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5,\
ARG6, ARG7, ARG8, ARG9, ARG10, ARG11)	\
def_fn_type (ENUM, RETURN, 0, 11, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6,	\
ARG7, ARG8, ARG9, ARG10, ARG11);
#define DEF_FUNCTION_TYPE_VAR_0(ENUM, RETURN) \
def_fn_type (ENUM, RETURN, 1, 0);
#define DEF_FUNCTION_TYPE_VAR_1(ENUM, RETURN, ARG1) \
def_fn_type (ENUM, RETURN, 1, 1, ARG1);
#define DEF_FUNCTION_TYPE_VAR_2(ENUM, RETURN, ARG1, ARG2) \
def_fn_type (ENUM, RETURN, 1, 2, ARG1, ARG2);
#define DEF_FUNCTION_TYPE_VAR_3(ENUM, RETURN, ARG1, ARG2, ARG3) \
def_fn_type (ENUM, RETURN, 1, 3, ARG1, ARG2, ARG3);
#define DEF_FUNCTION_TYPE_VAR_4(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4) \
def_fn_type (ENUM, RETURN, 1, 4, ARG1, ARG2, ARG3, ARG4);
#define DEF_FUNCTION_TYPE_VAR_5(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5) \
def_fn_type (ENUM, RETURN, 1, 5, ARG1, ARG2, ARG3, ARG4, ARG5);
#define DEF_FUNCTION_TYPE_VAR_6(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6)				\
def_fn_type (ENUM, RETURN, 1, 6, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6);
#define DEF_FUNCTION_TYPE_VAR_7(ENUM, RETURN, ARG1, ARG2, ARG3, ARG4, ARG5, \
ARG6, ARG7)				\
def_fn_type (ENUM, RETURN, 1, 7, ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7);
#define DEF_POINTER_TYPE(ENUM, TYPE) \
builtin_types[(int) ENUM] = build_pointer_type (builtin_types[(int) TYPE]);
#include "builtin-types.def"
#undef DEF_PRIMITIVE_TYPE
#undef DEF_FUNCTION_TYPE_0
#undef DEF_FUNCTION_TYPE_1
#undef DEF_FUNCTION_TYPE_2
#undef DEF_FUNCTION_TYPE_3
#undef DEF_FUNCTION_TYPE_4
#undef DEF_FUNCTION_TYPE_5
#undef DEF_FUNCTION_TYPE_6
#undef DEF_FUNCTION_TYPE_7
#undef DEF_FUNCTION_TYPE_8
#undef DEF_FUNCTION_TYPE_9
#undef DEF_FUNCTION_TYPE_10
#undef DEF_FUNCTION_TYPE_11
#undef DEF_FUNCTION_TYPE_VAR_0
#undef DEF_FUNCTION_TYPE_VAR_1
#undef DEF_FUNCTION_TYPE_VAR_2
#undef DEF_FUNCTION_TYPE_VAR_3
#undef DEF_FUNCTION_TYPE_VAR_4
#undef DEF_FUNCTION_TYPE_VAR_5
#undef DEF_FUNCTION_TYPE_VAR_6
#undef DEF_FUNCTION_TYPE_VAR_7
#undef DEF_POINTER_TYPE
builtin_types[(int) BT_LAST] = NULL_TREE;
}
enum built_in_attribute
{
#define DEF_ATTR_NULL_TREE(ENUM) ENUM,
#define DEF_ATTR_INT(ENUM, VALUE) ENUM,
#define DEF_ATTR_STRING(ENUM, VALUE) ENUM,
#define DEF_ATTR_IDENT(ENUM, STRING) ENUM,
#define DEF_ATTR_TREE_LIST(ENUM, PURPOSE, VALUE, CHAIN) ENUM,
#include "builtin-attrs.def"
#undef DEF_ATTR_NULL_TREE
#undef DEF_ATTR_INT
#undef DEF_ATTR_STRING
#undef DEF_ATTR_IDENT
#undef DEF_ATTR_TREE_LIST
ATTR_LAST
};
static GTY(()) tree built_in_attributes[(int) ATTR_LAST];
static void
install_builtin_attributes (void)
{
#define DEF_ATTR_NULL_TREE(ENUM)				\
built_in_attributes[(int) ENUM] = NULL_TREE;
#define DEF_ATTR_INT(ENUM, VALUE)				\
built_in_attributes[(int) ENUM] = build_int_cst (NULL_TREE, VALUE);
#define DEF_ATTR_STRING(ENUM, VALUE)				\
built_in_attributes[(int) ENUM] = build_string (strlen (VALUE), VALUE);
#define DEF_ATTR_IDENT(ENUM, STRING)				\
built_in_attributes[(int) ENUM] = get_identifier (STRING);
#define DEF_ATTR_TREE_LIST(ENUM, PURPOSE, VALUE, CHAIN)	\
built_in_attributes[(int) ENUM]			\
= tree_cons (built_in_attributes[(int) PURPOSE],	\
built_in_attributes[(int) VALUE],	\
built_in_attributes[(int) CHAIN]);
#include "builtin-attrs.def"
#undef DEF_ATTR_NULL_TREE
#undef DEF_ATTR_INT
#undef DEF_ATTR_STRING
#undef DEF_ATTR_IDENT
#undef DEF_ATTR_TREE_LIST
}
static tree
handle_const_attribute (tree *node, tree ARG_UNUSED (name),
tree ARG_UNUSED (args), int ARG_UNUSED (flags),
bool *no_add_attrs)
{
if (TREE_CODE (*node) == FUNCTION_DECL)
TREE_READONLY (*node) = 1;
else
*no_add_attrs = true;
return NULL_TREE;
}
static tree
handle_nothrow_attribute (tree *node, tree ARG_UNUSED (name),
tree ARG_UNUSED (args), int ARG_UNUSED (flags),
bool *no_add_attrs)
{
if (TREE_CODE (*node) == FUNCTION_DECL)
TREE_NOTHROW (*node) = 1;
else
*no_add_attrs = true;
return NULL_TREE;
}
static tree
handle_pure_attribute (tree *node, tree name, tree ARG_UNUSED (args),
int ARG_UNUSED (flags), bool *no_add_attrs)
{
if (TREE_CODE (*node) == FUNCTION_DECL)
DECL_PURE_P (*node) = 1;
else
{
warning (OPT_Wattributes, "%qs attribute ignored",
IDENTIFIER_POINTER (name));
*no_add_attrs = true;
}
return NULL_TREE;
}
static tree
handle_novops_attribute (tree *node, tree ARG_UNUSED (name),
tree ARG_UNUSED (args), int ARG_UNUSED (flags),
bool *ARG_UNUSED (no_add_attrs))
{
gcc_assert (TREE_CODE (*node) == FUNCTION_DECL);
DECL_IS_NOVOPS (*node) = 1;
return NULL_TREE;
}
static bool
get_nonnull_operand (tree arg_num_expr, unsigned HOST_WIDE_INT *valp)
{
if (!tree_fits_uhwi_p (arg_num_expr))
return false;
*valp = TREE_INT_CST_LOW (arg_num_expr);
return true;
}
static tree
handle_nonnull_attribute (tree *node, tree ARG_UNUSED (name),
tree args, int ARG_UNUSED (flags),
bool *no_add_attrs)
{
tree type = *node;
unsigned HOST_WIDE_INT attr_arg_num;
if (!args)
{
if (!prototype_p (type)
&& (!TYPE_ATTRIBUTES (type)
|| !lookup_attribute ("type generic", TYPE_ATTRIBUTES (type))))
{
error ("nonnull attribute without arguments on a non-prototype");
*no_add_attrs = true;
}
return NULL_TREE;
}
for (attr_arg_num = 1; args; args = TREE_CHAIN (args))
{
unsigned HOST_WIDE_INT arg_num = 0, ck_num;
if (!get_nonnull_operand (TREE_VALUE (args), &arg_num))
{
error ("nonnull argument has invalid operand number (argument %lu)",
(unsigned long) attr_arg_num);
*no_add_attrs = true;
return NULL_TREE;
}
if (prototype_p (type))
{
function_args_iterator iter;
tree argument;
function_args_iter_init (&iter, type);
for (ck_num = 1; ; ck_num++, function_args_iter_next (&iter))
{
argument = function_args_iter_cond (&iter);
if (!argument || ck_num == arg_num)
break;
}
if (!argument
|| TREE_CODE (argument) == VOID_TYPE)
{
error ("nonnull argument with out-of-range operand number "
"(argument %lu, operand %lu)",
(unsigned long) attr_arg_num, (unsigned long) arg_num);
*no_add_attrs = true;
return NULL_TREE;
}
if (TREE_CODE (argument) != POINTER_TYPE)
{
error ("nonnull argument references non-pointer operand "
"(argument %lu, operand %lu)",
(unsigned long) attr_arg_num, (unsigned long) arg_num);
*no_add_attrs = true;
return NULL_TREE;
}
}
}
return NULL_TREE;
}
static tree
handle_sentinel_attribute (tree *node, tree name, tree args,
int ARG_UNUSED (flags), bool *no_add_attrs)
{
if (!prototype_p (*node))
{
warning (OPT_Wattributes,
"%qs attribute requires prototypes with named arguments",
IDENTIFIER_POINTER (name));
*no_add_attrs = true;
}
else
{
if (!stdarg_p (*node))
{
warning (OPT_Wattributes,
"%qs attribute only applies to variadic functions",
IDENTIFIER_POINTER (name));
*no_add_attrs = true;
}
}
if (args)
{
tree position = TREE_VALUE (args);
if (TREE_CODE (position) != INTEGER_CST)
{
warning (0, "requested position is not an integer constant");
*no_add_attrs = true;
}
else
{
if (tree_int_cst_lt (position, integer_zero_node))
{
warning (0, "requested position is less than zero");
*no_add_attrs = true;
}
}
}
return NULL_TREE;
}
static tree
handle_noreturn_attribute (tree *node, tree name, tree ARG_UNUSED (args),
int ARG_UNUSED (flags), bool *no_add_attrs)
{
tree type = TREE_TYPE (*node);
if (TREE_CODE (*node) == FUNCTION_DECL)
TREE_THIS_VOLATILE (*node) = 1;
else if (TREE_CODE (type) == POINTER_TYPE
&& TREE_CODE (TREE_TYPE (type)) == FUNCTION_TYPE)
TREE_TYPE (*node)
= build_pointer_type
(change_qualified_type (TREE_TYPE (type), TYPE_QUAL_VOLATILE));
else
{
warning (OPT_Wattributes, "%qs attribute ignored",
IDENTIFIER_POINTER (name));
*no_add_attrs = true;
}
return NULL_TREE;
}
static tree
handle_noinline_attribute (tree *node, tree name,
tree ARG_UNUSED (args),
int ARG_UNUSED (flags), bool *no_add_attrs)
{
if (TREE_CODE (*node) == FUNCTION_DECL)
{
if (lookup_attribute ("always_inline", DECL_ATTRIBUTES (*node)))
{
warning (OPT_Wattributes, "%qE attribute ignored due to conflict "
"with attribute %qs", name, "always_inline");
*no_add_attrs = true;
}
else
DECL_UNINLINABLE (*node) = 1;
}
else
{
warning (OPT_Wattributes, "%qE attribute ignored", name);
*no_add_attrs = true;
}
return NULL_TREE;
}
static tree
handle_noclone_attribute (tree *node, tree name,
tree ARG_UNUSED (args),
int ARG_UNUSED (flags), bool *no_add_attrs)
{
if (TREE_CODE (*node) != FUNCTION_DECL)
{
warning (OPT_Wattributes, "%qE attribute ignored", name);
*no_add_attrs = true;
}
return NULL_TREE;
}
static tree
handle_leaf_attribute (tree *node, tree name, tree ARG_UNUSED (args),
int ARG_UNUSED (flags), bool *no_add_attrs)
{
if (TREE_CODE (*node) != FUNCTION_DECL)
{
warning (OPT_Wattributes, "%qE attribute ignored", name);
*no_add_attrs = true;
}
if (!TREE_PUBLIC (*node))
{
warning (OPT_Wattributes, "%qE attribute has no effect", name);
*no_add_attrs = true;
}
return NULL_TREE;
}
static tree
handle_always_inline_attribute (tree *node, tree name, tree ARG_UNUSED (args),
int ARG_UNUSED (flags), bool *no_add_attrs)
{
if (TREE_CODE (*node) == FUNCTION_DECL)
{
DECL_DISREGARD_INLINE_LIMITS (*node) = 1;
}
else
{
warning (OPT_Wattributes, "%qE attribute ignored", name);
*no_add_attrs = true;
}
return NULL_TREE;
}
static tree
handle_malloc_attribute (tree *node, tree name, tree ARG_UNUSED (args),
int ARG_UNUSED (flags), bool *no_add_attrs)
{
if (TREE_CODE (*node) == FUNCTION_DECL
&& POINTER_TYPE_P (TREE_TYPE (TREE_TYPE (*node))))
DECL_IS_MALLOC (*node) = 1;
else
{
warning (OPT_Wattributes, "%qs attribute ignored",
IDENTIFIER_POINTER (name));
*no_add_attrs = true;
}
return NULL_TREE;
}
tree
fake_attribute_handler (tree * ARG_UNUSED (node),
tree ARG_UNUSED (name),
tree ARG_UNUSED (args),
int  ARG_UNUSED (flags),
bool * ARG_UNUSED (no_add_attrs))
{
return NULL_TREE;
}
static tree
handle_type_generic_attribute (tree *node, tree ARG_UNUSED (name),
tree ARG_UNUSED (args), int ARG_UNUSED (flags),
bool * ARG_UNUSED (no_add_attrs))
{
gcc_assert (TREE_CODE (*node) == FUNCTION_TYPE);
gcc_assert (!prototype_p (*node) || stdarg_p (*node));
return NULL_TREE;
}
static tree
handle_vector_size_attribute (tree *node, tree name, tree args,
int ARG_UNUSED (flags), bool *no_add_attrs)
{
tree type = *node;
tree vector_type;
*no_add_attrs = true;
while (POINTER_TYPE_P (type)
|| TREE_CODE (type) == FUNCTION_TYPE
|| TREE_CODE (type) == ARRAY_TYPE)
type = TREE_TYPE (type);
vector_type = build_vector_type_for_size (type, TREE_VALUE (args), name);
if (!vector_type)
return NULL_TREE;
*node = reconstruct_complex_type (*node, vector_type);
return NULL_TREE;
}
static tree
handle_vector_type_attribute (tree *node, tree name, tree ARG_UNUSED (args),
int ARG_UNUSED (flags), bool *no_add_attrs)
{
tree type = *node;
tree vector_type;
*no_add_attrs = true;
if (TREE_CODE (type) != ARRAY_TYPE)
{
error ("attribute %qs applies to array types only",
IDENTIFIER_POINTER (name));
return NULL_TREE;
}
vector_type = build_vector_type_for_array (type, name);
if (!vector_type)
return NULL_TREE;
TYPE_REPRESENTATIVE_ARRAY (vector_type) = type;
*node = vector_type;
return NULL_TREE;
}
static void
def_builtin_1 (enum built_in_function fncode,
const char *name,
enum built_in_class fnclass,
tree fntype, tree libtype,
bool both_p, bool fallback_p,
bool nonansi_p ATTRIBUTE_UNUSED,
tree fnattrs, bool implicit_p)
{
tree decl;
const char *libname;
if (builtin_decl_explicit (fncode))
return;
if (fntype == error_mark_node)
return;
gcc_assert ((!both_p && !fallback_p)
|| !strncmp (name, "__builtin_",
strlen ("__builtin_")));
libname = name + strlen ("__builtin_");
decl = add_builtin_function (name, fntype, fncode, fnclass,
(fallback_p ? libname : NULL),
fnattrs);
if (both_p)
add_builtin_function (libname, libtype, fncode, fnclass,
NULL, fnattrs);
set_builtin_decl (fncode, decl, implicit_p);
}
static int flag_isoc94 = 0;
static int flag_isoc99 = 0;
static int flag_isoc11 = 0;
static void
install_builtin_functions (void)
{
#define DEF_BUILTIN(ENUM, NAME, CLASS, TYPE, LIBTYPE, BOTH_P, FALLBACK_P, \
NONANSI_P, ATTRS, IMPLICIT, COND)			\
if (NAME && COND)							\
def_builtin_1 (ENUM, NAME, CLASS,                                   \
builtin_types[(int) TYPE],                           \
builtin_types[(int) LIBTYPE],                        \
BOTH_P, FALLBACK_P, NONANSI_P,                       \
built_in_attributes[(int) ATTRS], IMPLICIT);
#include "builtins.def"
}
void
gnat_install_builtins (void)
{
install_builtin_elementary_types ();
install_builtin_function_types ();
install_builtin_attributes ();
build_common_builtin_nodes ();
targetm.init_builtins ();
install_builtin_functions ();
}
#include "gt-ada-utils.h"
#include "gtype-ada.h"
