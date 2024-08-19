#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "function.h"
#include "bitmap.h"
#include "tree.h"
#include "gimple-expr.h"
#include "stringpool.h"
#include "cgraph.h"
#include "predict.h"
#include "diagnostic.h"
#include "alias.h"
#include "fold-const.h"
#include "stor-layout.h"
#include "stmt.h"
#include "varasm.h"
#include "output.h"
#include "libfuncs.h"	
#include "tree-iterator.h"
#include "gimplify.h"
#include "opts.h"
#include "common/common-target.h"
#include "stringpool.h"
#include "attribs.h"
#include "ada.h"
#include "adadecode.h"
#include "types.h"
#include "atree.h"
#include "namet.h"
#include "nlists.h"
#include "snames.h"
#include "stringt.h"
#include "uintp.h"
#include "urealp.h"
#include "fe.h"
#include "sinfo.h"
#include "einfo.h"
#include "gadaint.h"
#include "ada-tree.h"
#include "gigi.h"
#define ALLOCA_THRESHOLD 1000
struct Node *Nodes_Ptr;
struct Flags *Flags_Ptr;
Node_Id *Next_Node_Ptr;
Node_Id *Prev_Node_Ptr;
struct Elist_Header *Elists_Ptr;
struct Elmt_Item *Elmts_Ptr;
struct String_Entry *Strings_Ptr;
Char_Code *String_Chars_Ptr;
struct List_Header *List_Headers_Ptr;
int max_gnat_nodes;
Node_Id error_gnat_node;
bool type_annotate_only;
static vec<Node_Id> gnat_validate_uc_list;
struct GTY (()) parm_attr_d {
int id; 
int dim;
tree first;
tree last;
tree length;
};
typedef struct parm_attr_d *parm_attr;
struct GTY(()) language_function {
vec<parm_attr, va_gc> *parm_attr_cache;
bitmap named_ret_val;
vec<tree, va_gc> *other_ret_val;
int gnat_ret;
};
#define f_parm_attr_cache \
DECL_STRUCT_FUNCTION (current_function_decl)->language->parm_attr_cache
#define f_named_ret_val \
DECL_STRUCT_FUNCTION (current_function_decl)->language->named_ret_val
#define f_other_ret_val \
DECL_STRUCT_FUNCTION (current_function_decl)->language->other_ret_val
#define f_gnat_ret \
DECL_STRUCT_FUNCTION (current_function_decl)->language->gnat_ret
struct GTY((chain_next ("%h.previous"))) stmt_group {
struct stmt_group *previous;	
tree stmt_list;		
tree block;			
tree cleanups;		
};
static GTY(()) struct stmt_group *current_stmt_group;
static GTY((deletable)) struct stmt_group *stmt_group_free_list;
struct GTY((chain_next ("%h.next"))) elab_info {
struct elab_info *next;	
tree elab_proc;		
int gnat_node;		
};
static GTY(()) struct elab_info *elab_info_list;
static GTY(()) vec<tree, va_gc> *gnu_except_ptr_stack;
static GTY(()) tree gnu_incoming_exc_ptr;
static GTY(()) vec<tree, va_gc> *gnu_elab_proc_stack;
static GTY(()) vec<tree, va_gc> *gnu_return_label_stack;
static GTY(()) vec<tree, va_gc> *gnu_return_var_stack;
struct GTY(()) range_check_info_d {
tree low_bound;
tree high_bound;
tree disp;
bool neg_p;
tree type;
tree invariant_cond;
tree inserted_cond;
};
typedef struct range_check_info_d *range_check_info;
struct GTY(()) loop_info_d {
tree stmt;
tree loop_var;
tree low_bound;
tree high_bound;
vec<range_check_info, va_gc> *checks;
};
typedef struct loop_info_d *loop_info;
static GTY(()) vec<loop_info, va_gc> *gnu_loop_stack;
static vec<Entity_Id> gnu_constraint_error_label_stack;
static vec<Entity_Id> gnu_storage_error_label_stack;
static vec<Entity_Id> gnu_program_error_label_stack;
static enum tree_code gnu_codes[Number_Node_Kinds];
static void init_code_table (void);
static tree get_elaboration_procedure (void);
static void Compilation_Unit_to_gnu (Node_Id);
static bool empty_stmt_list_p (tree);
static void record_code_position (Node_Id);
static void insert_code_for (Node_Id);
static void add_cleanup (tree, Node_Id);
static void add_stmt_list (List_Id);
static tree build_stmt_group (List_Id, bool);
static inline bool stmt_group_may_fallthru (void);
static enum gimplify_status gnat_gimplify_stmt (tree *);
static void elaborate_all_entities (Node_Id);
static void process_freeze_entity (Node_Id);
static void process_decls (List_Id, List_Id, Node_Id, bool, bool);
static tree emit_range_check (tree, Node_Id, Node_Id);
static tree emit_check (tree, tree, int, Node_Id);
static tree build_unary_op_trapv (enum tree_code, tree, tree, Node_Id);
static tree build_binary_op_trapv (enum tree_code, tree, tree, tree, Node_Id);
static tree convert_with_check (Entity_Id, tree, bool, bool, bool, Node_Id);
static bool addressable_p (tree, tree);
static tree assoc_to_constructor (Entity_Id, Node_Id, tree);
static tree pos_to_constructor (Node_Id, tree, Entity_Id);
static void validate_unchecked_conversion (Node_Id);
static Node_Id adjust_for_implicit_deref (Node_Id);
static tree maybe_implicit_deref (tree);
static void set_expr_location_from_node (tree, Node_Id, bool = false);
static void set_gnu_expr_location_from_node (tree, Node_Id);
static bool set_end_locus_from_node (tree, Node_Id);
static int lvalue_required_p (Node_Id, tree, bool, bool, bool);
static tree build_raise_check (int, enum exception_info_kind);
static tree create_init_temporary (const char *, tree, tree *, Node_Id);
static const char *extract_encoding (const char *) ATTRIBUTE_UNUSED;
static const char *decode_name (const char *) ATTRIBUTE_UNUSED;

void
gigi (Node_Id gnat_root,
int max_gnat_node,
int number_name ATTRIBUTE_UNUSED,
struct Node *nodes_ptr,
struct Flags *flags_ptr,
Node_Id *next_node_ptr,
Node_Id *prev_node_ptr,
struct Elist_Header *elists_ptr,
struct Elmt_Item *elmts_ptr,
struct String_Entry *strings_ptr,
Char_Code *string_chars_ptr,
struct List_Header *list_headers_ptr,
Nat number_file,
struct File_Info_Type *file_info_ptr,
Entity_Id standard_boolean,
Entity_Id standard_integer,
Entity_Id standard_character,
Entity_Id standard_long_long_float,
Entity_Id standard_exception_type,
Int gigi_operating_mode)
{
Node_Id gnat_iter;
Entity_Id gnat_literal;
tree t, ftype, int64_type;
struct elab_info *info;
int i;
max_gnat_nodes = max_gnat_node;
Nodes_Ptr = nodes_ptr;
Flags_Ptr = flags_ptr;
Next_Node_Ptr = next_node_ptr;
Prev_Node_Ptr = prev_node_ptr;
Elists_Ptr = elists_ptr;
Elmts_Ptr = elmts_ptr;
Strings_Ptr = strings_ptr;
String_Chars_Ptr = string_chars_ptr;
List_Headers_Ptr = list_headers_ptr;
type_annotate_only = (gigi_operating_mode == 1);
for (i = 0; i < number_file; i++)
{
const char *filename
= IDENTIFIER_POINTER
(get_identifier
(__gnat_to_canonical_file_spec
(Get_Name_String (file_info_ptr[i].File_Name))));
gcc_assert ((int) LINEMAPS_ORDINARY_USED (line_table) == i);
linemap_add (line_table, LC_ENTER, 0, filename, 1);
linemap_line_start (line_table, file_info_ptr[i].Num_Source_Lines, 252);
linemap_position_for_column (line_table, 252 - 1);
linemap_add (line_table, LC_LEAVE, 0, NULL, 0);
}
gcc_assert (Nkind (gnat_root) == N_Compilation_Unit);
t = create_concat_name (Defining_Entity (Unit (gnat_root)), NULL);
first_global_object_name = ggc_strdup (IDENTIFIER_POINTER (t));
init_code_table ();
init_gnat_decl ();
init_gnat_utils ();
if (type_annotate_only)
{
TYPE_SIZE (void_type_node) = bitsize_zero_node;
TYPE_SIZE_UNIT (void_type_node) = size_zero_node;
}
if (!Stack_Check_Probes_On_Target)
set_stack_check_libfunc ("_gnat_stack_check");
double_float_alignment = get_target_double_float_alignment ();
double_scalar_alignment = get_target_double_scalar_alignment ();
record_builtin_type ("integer", integer_type_node, false);
record_builtin_type ("character", char_type_node, false);
record_builtin_type ("boolean", boolean_type_node, false);
record_builtin_type ("void", void_type_node, false);
save_gnu_tree (Base_Type (standard_integer),
TYPE_NAME (integer_type_node),
false);
finish_character_type (char_type_node);
save_gnu_tree (Base_Type (standard_character),
TYPE_NAME (char_type_node),
false);
save_gnu_tree (Base_Type (standard_boolean),
TYPE_NAME (boolean_type_node),
false);
gnat_literal = First_Literal (Base_Type (standard_boolean));
t = UI_To_gnu (Enumeration_Rep (gnat_literal), boolean_type_node);
gcc_assert (t == boolean_false_node);
t = create_var_decl (get_entity_name (gnat_literal), NULL_TREE,
boolean_type_node, t, true, false, false, false, false,
true, false, NULL, gnat_literal);
save_gnu_tree (gnat_literal, t, false);
gnat_literal = Next_Literal (gnat_literal);
t = UI_To_gnu (Enumeration_Rep (gnat_literal), boolean_type_node);
gcc_assert (t == boolean_true_node);
t = create_var_decl (get_entity_name (gnat_literal), NULL_TREE,
boolean_type_node, t, true, false, false, false, false,
true, false, NULL, gnat_literal);
save_gnu_tree (gnat_literal, t, false);
void_list_node = build_tree_list (NULL_TREE, void_type_node);
void_ftype = build_function_type_list (void_type_node, NULL_TREE);
ptr_void_ftype = build_pointer_type (void_ftype);
malloc_decl
= create_subprog_decl (get_identifier ("__gnat_malloc"), NULL_TREE,
build_function_type_list (ptr_type_node, sizetype,
NULL_TREE),
NULL_TREE, is_disabled, true, true, true, false,
false, NULL, Empty);
DECL_IS_MALLOC (malloc_decl) = 1;
free_decl
= create_subprog_decl (get_identifier ("__gnat_free"), NULL_TREE,
build_function_type_list (void_type_node,
ptr_type_node, NULL_TREE),
NULL_TREE, is_disabled, true, true, true, false,
false, NULL, Empty);
realloc_decl
= create_subprog_decl (get_identifier ("__gnat_realloc"), NULL_TREE,
build_function_type_list (ptr_type_node,
ptr_type_node, sizetype,
NULL_TREE),
NULL_TREE, is_disabled, true, true, true, false,
false, NULL, Empty);
int64_type = gnat_type_for_size (64, 0);
mulv64_decl
= create_subprog_decl (get_identifier ("__gnat_mulv64"), NULL_TREE,
build_function_type_list (int64_type, int64_type,
int64_type, NULL_TREE),
NULL_TREE, is_disabled, true, true, true, false,
false, NULL, Empty);
parent_name_id = get_identifier (Get_Name_String (Name_uParent));
exception_data_name_id
= get_identifier ("system__standard_library__exception_data");
except_type_node = gnat_to_gnu_type (Base_Type (standard_exception_type));
jmpbuf_type
= build_array_type (gnat_type_for_mode (Pmode, 0),
build_index_type (size_int (5)));
record_builtin_type ("JMPBUF_T", jmpbuf_type, true);
jmpbuf_ptr_type = build_pointer_type (jmpbuf_type);
get_jmpbuf_decl
= create_subprog_decl
(get_identifier ("system__soft_links__get_jmpbuf_address_soft"),
NULL_TREE, build_function_type_list (jmpbuf_ptr_type, NULL_TREE),
NULL_TREE, is_disabled, true, true, true, false, false, NULL, Empty);
set_jmpbuf_decl
= create_subprog_decl
(get_identifier ("system__soft_links__set_jmpbuf_address_soft"),
NULL_TREE, build_function_type_list (void_type_node, jmpbuf_ptr_type,
NULL_TREE),
NULL_TREE, is_disabled, true, true, true, false, false, NULL, Empty);
get_excptr_decl
= create_subprog_decl
(get_identifier ("system__soft_links__get_gnat_exception"), NULL_TREE,
build_function_type_list (build_pointer_type (except_type_node),
NULL_TREE),
NULL_TREE, is_disabled, true, true, true, false, false, NULL, Empty);
not_handled_by_others_decl = get_identifier ("not_handled_by_others");
for (t = TYPE_FIELDS (except_type_node); t; t = DECL_CHAIN (t))
if (DECL_NAME (t) == not_handled_by_others_decl)
{
not_handled_by_others_decl = t;
break;
}
gcc_assert (DECL_P (not_handled_by_others_decl));
setjmp_decl
= create_subprog_decl
(get_identifier ("__builtin_setjmp"), NULL_TREE,
build_function_type_list (integer_type_node, jmpbuf_ptr_type,
NULL_TREE),
NULL_TREE, is_disabled, true, true, true, false, false, NULL, Empty);
DECL_BUILT_IN_CLASS (setjmp_decl) = BUILT_IN_NORMAL;
DECL_FUNCTION_CODE (setjmp_decl) = BUILT_IN_SETJMP;
update_setjmp_buf_decl
= create_subprog_decl
(get_identifier ("__builtin_update_setjmp_buf"), NULL_TREE,
build_function_type_list (void_type_node, jmpbuf_ptr_type, NULL_TREE),
NULL_TREE, is_disabled, true, true, true, false, false, NULL, Empty);
DECL_BUILT_IN_CLASS (update_setjmp_buf_decl) = BUILT_IN_NORMAL;
DECL_FUNCTION_CODE (update_setjmp_buf_decl) = BUILT_IN_UPDATE_SETJMP_BUF;
ftype = build_function_type_list (void_type_node,
build_pointer_type (except_type_node),
NULL_TREE);
ftype = build_qualified_type (ftype, TYPE_QUAL_VOLATILE);
raise_nodefer_decl
= create_subprog_decl
(get_identifier ("__gnat_raise_nodefer_with_msg"), NULL_TREE, ftype,
NULL_TREE, is_disabled, true, true, true, false, false, NULL, Empty);
set_exception_parameter_decl
= create_subprog_decl
(get_identifier ("__gnat_set_exception_parameter"), NULL_TREE,
build_function_type_list (void_type_node, ptr_type_node, ptr_type_node,
NULL_TREE),
NULL_TREE, is_disabled, true, true, true, false, false, NULL, Empty);
ftype = build_function_type_list (void_type_node, ptr_type_node, NULL_TREE);
begin_handler_decl
= create_subprog_decl (get_identifier ("__gnat_begin_handler"), NULL_TREE,
ftype, NULL_TREE,
is_disabled, true, true, true, false, false, NULL,
Empty);
TREE_NOTHROW (begin_handler_decl) = 1;
end_handler_decl
= create_subprog_decl (get_identifier ("__gnat_end_handler"), NULL_TREE,
ftype, NULL_TREE,
is_disabled, true, true, true, false, false, NULL,
Empty);
unhandled_except_decl
= create_subprog_decl (get_identifier ("__gnat_unhandled_except_handler"),
NULL_TREE, ftype, NULL_TREE,
is_disabled, true, true, true, false, false, NULL,
Empty);
ftype = build_qualified_type (ftype, TYPE_QUAL_VOLATILE);
reraise_zcx_decl
= create_subprog_decl (get_identifier ("__gnat_reraise_zcx"), NULL_TREE,
ftype, NULL_TREE,
is_disabled, true, true, true, false, false, NULL,
Empty);
others_decl
= create_var_decl (get_identifier ("OTHERS"),
get_identifier ("__gnat_others_value"),
char_type_node, NULL_TREE,
true, false, true, false, false, true, false,
NULL, Empty);
all_others_decl
= create_var_decl (get_identifier ("ALL_OTHERS"),
get_identifier ("__gnat_all_others_value"),
char_type_node, NULL_TREE,
true, false, true, false, false, true, false,
NULL, Empty);
unhandled_others_decl
= create_var_decl (get_identifier ("UNHANDLED_OTHERS"),
get_identifier ("__gnat_unhandled_others_value"),
char_type_node, NULL_TREE,
true, false, true, false, false, true, false,
NULL, Empty);
if (No_Exception_Handlers_Set ())
{
ftype = build_function_type_list (void_type_node,
build_pointer_type (char_type_node),
integer_type_node, NULL_TREE);
ftype = build_qualified_type (ftype, TYPE_QUAL_VOLATILE);
tree decl
= create_subprog_decl
(get_identifier ("__gnat_last_chance_handler"), NULL_TREE, ftype,
NULL_TREE, is_disabled, true, true, true, false, false, NULL,
Empty);
for (i = 0; i < (int) ARRAY_SIZE (gnat_raise_decls); i++)
gnat_raise_decls[i] = decl;
}
else
{
for (i = 0; i < (int) ARRAY_SIZE (gnat_raise_decls); i++)
gnat_raise_decls[i] = build_raise_check (i, exception_simple);
for (i = 0; i < (int) ARRAY_SIZE (gnat_raise_decls_ext); i++)
gnat_raise_decls_ext[i]
= build_raise_check (i,
i == CE_Index_Check_Failed
|| i == CE_Range_Check_Failed
|| i == CE_Invalid_Data
? exception_range : exception_column);
}
if (TARGET_VTABLE_USES_DESCRIPTORS)
{
tree null_node = fold_convert (ptr_void_ftype, null_pointer_node);
tree field_list = NULL_TREE;
int j;
vec<constructor_elt, va_gc> *null_vec = NULL;
constructor_elt *elt;
fdesc_type_node = make_node (RECORD_TYPE);
vec_safe_grow (null_vec, TARGET_VTABLE_USES_DESCRIPTORS);
elt = (null_vec->address () + TARGET_VTABLE_USES_DESCRIPTORS - 1);
for (j = 0; j < TARGET_VTABLE_USES_DESCRIPTORS; j++)
{
tree field
= create_field_decl (NULL_TREE, ptr_void_ftype, fdesc_type_node,
NULL_TREE, NULL_TREE, 0, 1);
DECL_CHAIN (field) = field_list;
field_list = field;
elt->index = field;
elt->value = null_node;
elt--;
}
finish_record_type (fdesc_type_node, nreverse (field_list), 0, false);
record_builtin_type ("descriptor", fdesc_type_node, true);
null_fdesc_node = gnat_build_constructor (fdesc_type_node, null_vec);
}
longest_float_type_node
= get_unpadded_type (Base_Type (standard_long_long_float));
main_identifier_node = get_identifier ("main");
if (Back_End_Exceptions ())
gnat_init_gcc_eh ();
gnat_init_gcc_fp ();
gnat_install_builtins ();
vec_safe_push (gnu_except_ptr_stack, NULL_TREE);
gnu_constraint_error_label_stack.safe_push (Empty);
gnu_storage_error_label_stack.safe_push (Empty);
gnu_program_error_label_stack.safe_push (Empty);
if (Present (Ident_String (Main_Unit)))
targetm.asm_out.output_ident
(TREE_STRING_POINTER (gnat_to_gnu (Ident_String (Main_Unit))));
if (No_Strict_Aliasing_CP)
flag_strict_aliasing = 0;
optimization_default_node = build_optimization_node (&global_options);
optimization_current_node = optimization_default_node;
Compilation_Unit_to_gnu (gnat_root);
for (i = 0; gnat_validate_uc_list.iterate (i, &gnat_iter); i++)
validate_unchecked_conversion (gnat_iter);
gnat_validate_uc_list.release ();
for (info = elab_info_list; info; info = info->next)
{
tree gnu_body = DECL_SAVED_TREE (info->elab_proc);
tree gnu_stmts = gnu_body;
if (TREE_CODE (gnu_stmts) == BIND_EXPR)
gnu_stmts = BIND_EXPR_BODY (gnu_stmts);
if (!gnu_stmts || empty_stmt_list_p (gnu_stmts))
Set_Has_No_Elaboration_Code (info->gnat_node, 1);
else
{
begin_subprog_body (info->elab_proc);
end_subprog_body (gnu_body);
rest_of_subprog_body_compilation (info->elab_proc);
}
}
destroy_gnat_decl ();
destroy_gnat_utils ();
error_gnat_node = Empty;
}

static tree
build_raise_check (int check, enum exception_info_kind kind)
{
tree result, ftype;
const char pfx[] = "__gnat_rcheck_";
strcpy (Name_Buffer, pfx);
Name_Len = sizeof (pfx) - 1;
Get_RT_Exception_Name (check);
if (kind == exception_simple)
{
Name_Buffer[Name_Len] = 0;
ftype
= build_function_type_list (void_type_node,
build_pointer_type (char_type_node),
integer_type_node, NULL_TREE);
}
else
{
tree t = (kind == exception_column ? NULL_TREE : integer_type_node);
strcpy (Name_Buffer + Name_Len, "_ext");
Name_Buffer[Name_Len + 4] = 0;
ftype
= build_function_type_list (void_type_node,
build_pointer_type (char_type_node),
integer_type_node, integer_type_node,
t, t, NULL_TREE);
}
ftype = build_qualified_type (ftype, TYPE_QUAL_VOLATILE);
result
= create_subprog_decl (get_identifier (Name_Buffer), NULL_TREE, ftype,
NULL_TREE, is_disabled, true, true, true, false,
false, NULL, Empty);
return result;
}

static int
lvalue_required_for_attribute_p (Node_Id gnat_node)
{
switch (Get_Attribute_Id (Attribute_Name (gnat_node)))
{
case Attr_Pos:
case Attr_Val:
case Attr_Pred:
case Attr_Succ:
case Attr_First:
case Attr_Last:
case Attr_Range_Length:
case Attr_Length:
case Attr_Object_Size:
case Attr_Value_Size:
case Attr_Component_Size:
case Attr_Descriptor_Size:
case Attr_Max_Size_In_Storage_Elements:
case Attr_Min:
case Attr_Max:
case Attr_Null_Parameter:
case Attr_Passed_By_Reference:
case Attr_Mechanism_Code:
case Attr_Machine:
case Attr_Model:
return 0;
case Attr_Address:
case Attr_Access:
case Attr_Unchecked_Access:
case Attr_Unrestricted_Access:
case Attr_Code_Address:
case Attr_Pool_Address:
case Attr_Size:
case Attr_Alignment:
case Attr_Bit_Position:
case Attr_Position:
case Attr_First_Bit:
case Attr_Last_Bit:
case Attr_Bit:
case Attr_Asm_Input:
case Attr_Asm_Output:
default:
return 1;
}
}
static int
lvalue_required_p (Node_Id gnat_node, tree gnu_type, bool constant,
bool address_of_constant, bool aliased)
{
Node_Id gnat_parent = Parent (gnat_node), gnat_temp;
switch (Nkind (gnat_parent))
{
case N_Reference:
return 1;
case N_Attribute_Reference:
return lvalue_required_for_attribute_p (gnat_parent);
case N_Parameter_Association:
case N_Function_Call:
case N_Procedure_Call_Statement:
return (!constant
|| must_pass_by_ref (gnu_type)
|| default_pass_by_ref (gnu_type));
case N_Indexed_Component:
if (Prefix (gnat_parent) != gnat_node)
return 0;
for (gnat_temp = First (Expressions (gnat_parent));
Present (gnat_temp);
gnat_temp = Next (gnat_temp))
if (Nkind (gnat_temp) != N_Character_Literal
&& Nkind (gnat_temp) != N_Integer_Literal
&& !(Is_Entity_Name (gnat_temp)
&& Ekind (Entity (gnat_temp)) == E_Enumeration_Literal))
return 1;
case N_Slice:
if (Prefix (gnat_parent) != gnat_node)
return 0;
aliased |= Has_Aliased_Components (Etype (gnat_node));
return lvalue_required_p (gnat_parent, gnu_type, constant,
address_of_constant, aliased);
case N_Selected_Component:
aliased |= Is_Aliased (Entity (Selector_Name (gnat_parent)));
return lvalue_required_p (gnat_parent, gnu_type, constant,
address_of_constant, aliased);
case N_Object_Renaming_Declaration:
return 1;
case N_Object_Declaration:
return (!constant
||(Is_Composite_Type (Underlying_Type (Etype (gnat_node)))
&& Is_Atomic_Or_VFA (Defining_Entity (gnat_parent)))
|| Ekind ((Etype (Defining_Entity (gnat_parent))))
== E_Class_Wide_Subtype);
case N_Assignment_Statement:
return (!constant
|| Name (gnat_parent) == gnat_node
|| (Is_Composite_Type (Underlying_Type (Etype (gnat_node)))
&& Is_Entity_Name (Name (gnat_parent))
&& Is_Atomic_Or_VFA (Entity (Name (gnat_parent)))));
case N_Unchecked_Type_Conversion:
if (!constant)
return 1;
case N_Type_Conversion:
case N_Qualified_Expression:
return lvalue_required_p (gnat_parent,
get_unpadded_type (Etype (gnat_parent)),
constant, address_of_constant, aliased);
case N_Allocator:
return Is_Composite_Type (Underlying_Type (Etype (gnat_node)));
case N_Explicit_Dereference:
if (constant && address_of_constant)
return lvalue_required_p (gnat_parent,
get_unpadded_type (Etype (gnat_parent)),
true, false, true);
default:
return 0;
}
gcc_unreachable ();
}
static bool
constant_decl_with_initializer_p (tree t)
{
if (!TREE_CONSTANT (t) || !DECL_P (t) || !DECL_INITIAL (t))
return false;
if (AGGREGATE_TYPE_P (TREE_TYPE (t))
&& !TYPE_IS_FAT_POINTER_P (TREE_TYPE (t))
&& type_contains_placeholder_p (TREE_TYPE (t)))
return false;
return true;
}
static tree
fold_constant_decl_in_expr (tree exp)
{
enum tree_code code = TREE_CODE (exp);
tree op0;
switch (code)
{
case CONST_DECL:
case VAR_DECL:
if (!constant_decl_with_initializer_p (exp))
return exp;
return DECL_INITIAL (exp);
case COMPONENT_REF:
op0 = fold_constant_decl_in_expr (TREE_OPERAND (exp, 0));
if (op0 == TREE_OPERAND (exp, 0))
return exp;
return fold_build3 (COMPONENT_REF, TREE_TYPE (exp), op0,
TREE_OPERAND (exp, 1), NULL_TREE);
case BIT_FIELD_REF:
op0 = fold_constant_decl_in_expr (TREE_OPERAND (exp, 0));
if (op0 == TREE_OPERAND (exp, 0))
return exp;
return fold_build3 (BIT_FIELD_REF, TREE_TYPE (exp), op0,
TREE_OPERAND (exp, 1), TREE_OPERAND (exp, 2));
case ARRAY_REF:
case ARRAY_RANGE_REF:
if (!TREE_CONSTANT (TREE_OPERAND (exp, 1)))
return exp;
op0 = fold_constant_decl_in_expr (TREE_OPERAND (exp, 0));
if (op0 == TREE_OPERAND (exp, 0))
return exp;
return fold (build4 (code, TREE_TYPE (exp), op0, TREE_OPERAND (exp, 1),
TREE_OPERAND (exp, 2), NULL_TREE));
case REALPART_EXPR:
case IMAGPART_EXPR:
case VIEW_CONVERT_EXPR:
op0 = fold_constant_decl_in_expr (TREE_OPERAND (exp, 0));
if (op0 == TREE_OPERAND (exp, 0))
return exp;
return fold_build1 (code, TREE_TYPE (exp), op0);
default:
return exp;
}
gcc_unreachable ();
}
static tree
Identifier_to_gnu (Node_Id gnat_node, tree *gnu_result_type_p)
{
Node_Id gnat_temp, gnat_temp_type;
tree gnu_result, gnu_result_type;
int require_lvalue = -1;
bool use_constant_initializer = false;
gnat_temp = ((Nkind (gnat_node) == N_Defining_Identifier
|| Nkind (gnat_node) == N_Defining_Operator_Symbol)
? gnat_node : Entity (gnat_node));
gnat_temp_type = Etype (gnat_temp);
gcc_assert (Etype (gnat_node) == gnat_temp_type
|| (Is_Packed (gnat_temp_type)
&& (Etype (gnat_node)
== Packed_Array_Impl_Type (gnat_temp_type)))
|| (Is_Class_Wide_Type (Etype (gnat_node)))
|| (Is_Incomplete_Or_Private_Type (gnat_temp_type)
&& Present (Full_View (gnat_temp_type))
&& ((Etype (gnat_node) == Full_View (gnat_temp_type))
|| (Is_Packed (Full_View (gnat_temp_type))
&& (Etype (gnat_node)
== Packed_Array_Impl_Type
(Full_View (gnat_temp_type))))))
|| (Is_Incomplete_Type (gnat_temp_type)
&& From_Limited_With (gnat_temp_type)
&& Present (Non_Limited_View (gnat_temp_type))
&& Etype (gnat_node) == Non_Limited_View (gnat_temp_type))
|| (Is_Itype (Etype (gnat_node)) && Is_Itype (gnat_temp_type))
|| !(Ekind (gnat_temp) == E_Variable
|| Ekind (gnat_temp) == E_Component
|| Ekind (gnat_temp) == E_Constant
|| Ekind (gnat_temp) == E_Loop_Parameter
|| Is_Formal (gnat_temp)));
if (Ekind (gnat_temp) == E_Constant
&& Is_Private_Type (gnat_temp_type)
&& (Has_Unknown_Discriminants (gnat_temp_type)
|| (Present (Full_View (gnat_temp_type))
&& Has_Discriminants (Full_View (gnat_temp_type))))
&& Present (Full_View (gnat_temp)))
{
gnat_temp = Full_View (gnat_temp);
gnat_temp_type = Etype (gnat_temp);
}
else
{
if ((Ekind (gnat_temp) == E_Constant
|| Ekind (gnat_temp) == E_Variable || Is_Formal (gnat_temp))
&& !(Is_Array_Type (Etype (gnat_temp))
&& Present (Packed_Array_Impl_Type (Etype (gnat_temp))))
&& Present (Actual_Subtype (gnat_temp))
&& present_gnu_tree (Actual_Subtype (gnat_temp)))
gnat_temp_type = Actual_Subtype (gnat_temp);
else
gnat_temp_type = Etype (gnat_node);
}
gnu_result_type = get_unpadded_type (gnat_temp_type);
if (Ekind (gnat_temp) == E_Constant
&& Is_Elementary_Type (gnat_temp_type)
&& !Is_Imported (gnat_temp)
&& Present (Address_Clause (gnat_temp)))
{
require_lvalue = lvalue_required_p (gnat_node, gnu_result_type, true,
false, Is_Aliased (gnat_temp));
use_constant_initializer = !require_lvalue;
}
if (use_constant_initializer)
{
if (Present (Full_View (gnat_temp)))
gnat_temp = Full_View (gnat_temp);
gnu_result = gnat_to_gnu (Expression (Declaration_Node (gnat_temp)));
}
else
gnu_result = gnat_to_gnu_entity (gnat_temp, NULL_TREE, false);
if (DECL_P (gnu_result)
&& (DECL_BY_REF_P (gnu_result)
|| (TREE_CODE (gnu_result) == PARM_DECL
&& DECL_BY_COMPONENT_PTR_P (gnu_result))))
{
const bool read_only = DECL_POINTS_TO_READONLY_P (gnu_result);
if (TREE_CODE (gnu_result) == PARM_DECL
&& DECL_BY_COMPONENT_PTR_P (gnu_result))
gnu_result
= convert (build_pointer_type (gnu_result_type), gnu_result);
else if (TREE_CODE (gnu_result) == CONST_DECL
&& !(DECL_CONST_ADDRESS_P (gnu_result)
&& lvalue_required_p (gnat_node, gnu_result_type, true,
true, false)))
gnu_result = DECL_INITIAL (gnu_result);
if (TREE_CODE (gnu_result) == VAR_DECL
&& !DECL_LOOP_PARM_P (gnu_result)
&& DECL_RENAMED_OBJECT (gnu_result))
gnu_result = DECL_RENAMED_OBJECT (gnu_result);
else
{
gnu_result = build_unary_op (INDIRECT_REF, NULL_TREE, gnu_result);
if ((TREE_CODE (gnu_result) == INDIRECT_REF
|| TREE_CODE (gnu_result) == UNCONSTRAINED_ARRAY_REF)
&& No (Address_Clause (gnat_temp)))
TREE_THIS_NOTRAP (gnu_result) = 1;
if (read_only)
TREE_READONLY (gnu_result) = 1;
}
}
if (constant_decl_with_initializer_p (gnu_result))
{
bool constant_only = (TREE_CODE (gnu_result) == CONST_DECL
&& !DECL_CONST_CORRESPONDING_VAR (gnu_result));
bool address_of_constant = (TREE_CODE (gnu_result) == CONST_DECL
&& DECL_CONST_ADDRESS_P (gnu_result));
if ((!constant_only || address_of_constant) && require_lvalue < 0)
require_lvalue
= lvalue_required_p (gnat_node, gnu_result_type, true,
address_of_constant, Is_Aliased (gnat_temp));
if ((constant_only && !address_of_constant) || !require_lvalue)
gnu_result = DECL_INITIAL (gnu_result);
}
else if (Ekind (gnat_temp) == E_Constant
&& Is_Elementary_Type (gnat_temp_type)
&& Present (Renamed_Object (gnat_temp)))
{
if (require_lvalue < 0)
require_lvalue
= lvalue_required_p (gnat_node, gnu_result_type, true, false,
Is_Aliased (gnat_temp));
if (!require_lvalue)
gnu_result = fold_constant_decl_in_expr (gnu_result);
}
if (TREE_CODE (TREE_TYPE (gnu_result)) == FUNCTION_TYPE
|| Is_Constr_Subt_For_UN_Aliased (gnat_temp_type)
|| (Ekind (gnat_temp) == E_Constant
&& Present (Full_View (gnat_temp))
&& Has_Discriminants (gnat_temp_type)
&& TREE_CODE (gnu_result) == CONSTRUCTOR))
{
gnu_result_type = TREE_TYPE (gnu_result);
if (TYPE_IS_PADDING_P (gnu_result_type))
gnu_result_type = TREE_TYPE (TYPE_FIELDS (gnu_result_type));
}
*gnu_result_type_p = gnu_result_type;
return gnu_result;
}

static tree
Pragma_to_gnu (Node_Id gnat_node)
{
tree gnu_result = alloc_stmt_list ();
unsigned char pragma_id;
Node_Id gnat_temp;
if (type_annotate_only
|| !Is_Pragma_Name (Chars (Pragma_Identifier (gnat_node))))
return gnu_result;
pragma_id = Get_Pragma_Id (Chars (Pragma_Identifier (gnat_node)));
switch (pragma_id)
{
case Pragma_Inspection_Point:
if (global_bindings_p ())
break;
for (gnat_temp = First (Pragma_Argument_Associations (gnat_node));
Present (gnat_temp);
gnat_temp = Next (gnat_temp))
{
Node_Id gnat_expr = Expression (gnat_temp);
tree gnu_expr = gnat_to_gnu (gnat_expr);
tree asm_constraint = NULL_TREE;
#ifdef ASM_COMMENT_START
char *comment;
#endif
gnu_expr = maybe_unconstrained_array (gnu_expr);
gnat_mark_addressable (gnu_expr);
#ifdef ASM_COMMENT_START
comment = concat (ASM_COMMENT_START,
" inspection point: ",
Get_Name_String (Chars (gnat_expr)),
" is at %0",
NULL);
asm_constraint = build_string (strlen (comment), comment);
free (comment);
#endif
gnu_expr = build5 (ASM_EXPR, void_type_node,
asm_constraint,
NULL_TREE,
tree_cons
(build_tree_list (NULL_TREE,
build_string (1, "m")),
gnu_expr, NULL_TREE),
NULL_TREE, NULL_TREE);
ASM_VOLATILE_P (gnu_expr) = 1;
set_expr_location_from_node (gnu_expr, gnat_node);
append_to_statement_list (gnu_expr, &gnu_result);
}
break;
case Pragma_Loop_Optimize:
for (gnat_temp = First (Pragma_Argument_Associations (gnat_node));
Present (gnat_temp);
gnat_temp = Next (gnat_temp))
{
tree gnu_loop_stmt = gnu_loop_stack->last ()->stmt;
switch (Chars (Expression (gnat_temp)))
{
case Name_Ivdep:
LOOP_STMT_IVDEP (gnu_loop_stmt) = 1;
break;
case Name_No_Unroll:
LOOP_STMT_NO_UNROLL (gnu_loop_stmt) = 1;
break;
case Name_Unroll:
LOOP_STMT_UNROLL (gnu_loop_stmt) = 1;
break;
case Name_No_Vector:
LOOP_STMT_NO_VECTOR (gnu_loop_stmt) = 1;
break;
case Name_Vector:
LOOP_STMT_VECTOR (gnu_loop_stmt) = 1;
break;
default:
gcc_unreachable ();
}
}
break;
case Pragma_Optimize:
switch (Chars (Expression
(First (Pragma_Argument_Associations (gnat_node)))))
{
case Name_Off:
if (optimize)
post_error ("must specify -O0?", gnat_node);
break;
case Name_Space:
if (!optimize_size)
post_error ("must specify -Os?", gnat_node);
break;
case Name_Time:
if (!optimize)
post_error ("insufficient -O value?", gnat_node);
break;
default:
gcc_unreachable ();
}
break;
case Pragma_Reviewable:
if (write_symbols == NO_DEBUG)
post_error ("must specify -g?", gnat_node);
break;
case Pragma_Warning_As_Error:
case Pragma_Warnings:
{
Node_Id gnat_expr;
const location_t location = input_location;
struct cl_option_handlers handlers;
unsigned int option_index;
diagnostic_t kind;
bool imply;
gnat_temp = First (Pragma_Argument_Associations (gnat_node));
if (Nkind (Expression (gnat_temp)) == N_String_Literal)
{
switch (pragma_id)
{
case Pragma_Warning_As_Error:
kind = DK_ERROR;
imply = false;
break;
case Pragma_Warnings:
kind = DK_WARNING;
imply = true;
break;
default:
gcc_unreachable ();
}
gnat_expr = Expression (gnat_temp);
}
else if (Nkind (Expression (gnat_temp)) == N_Identifier)
{
switch (Chars (Expression (gnat_temp)))
{
case Name_Off:
kind = DK_IGNORED;
break;
case Name_On:
kind = DK_WARNING;
break;
default:
gcc_unreachable ();
}
if (Present (Next (gnat_temp))
&& Chars (Next (gnat_temp)) != Name_Reason)
{
if (Nkind (Expression (Next (gnat_temp))) != N_String_Literal)
break;
gnat_expr = Expression (Next (gnat_temp));
}
else
gnat_expr = Empty;
imply = false;
}
else
gcc_unreachable ();
const unsigned int lang_mask = CL_Ada | CL_COMMON;
const char *arg = NULL;
if (Present (gnat_expr))
{
tree gnu_expr = gnat_to_gnu (gnat_expr);
const char *option_string = TREE_STRING_POINTER (gnu_expr);
const int len = TREE_STRING_LENGTH (gnu_expr);
if (len < 3 || option_string[0] != '-' || option_string[1] != 'W')
break;
option_index = find_opt (option_string + 1, lang_mask);
if (option_index == OPT_SPECIAL_unknown)
{
post_error ("?unknown -W switch", gnat_node);
break;
}
else if (!(cl_options[option_index].flags & CL_WARNING))
{
post_error ("?-W switch does not control warning", gnat_node);
break;
}
else if (!(cl_options[option_index].flags & lang_mask))
{
post_error ("?-W switch not valid for Ada", gnat_node);
break;
}
if (cl_options[option_index].flags & CL_JOINED)
arg = option_string + 1 + cl_options[option_index].opt_len;
}
else
option_index = 0;
set_default_handlers (&handlers, NULL);
control_warning_option (option_index, (int) kind, arg, imply, location,
lang_mask, &handlers, &global_options,
&global_options_set, global_dc);
}
break;
default:
break;
}
return gnu_result;
}

static void
check_inlining_for_nested_subprog (tree fndecl)
{
if (DECL_IGNORED_P (current_function_decl) || DECL_IGNORED_P (fndecl))
return;
if (DECL_DECLARED_INLINE_P (fndecl))
return;
tree parent_decl = decl_function_context (fndecl);
if (DECL_EXTERNAL (parent_decl) && DECL_DECLARED_INLINE_P (parent_decl))
{
const location_t loc1 = DECL_SOURCE_LOCATION (fndecl);
const location_t loc2 = DECL_SOURCE_LOCATION (parent_decl);
if (lookup_attribute ("always_inline", DECL_ATTRIBUTES (parent_decl)))
{
error_at (loc1, "subprogram %q+F not marked Inline_Always", fndecl);
error_at (loc2, "parent subprogram cannot be inlined");
}
else
{
warning_at (loc1, OPT_Winline, "subprogram %q+F not marked Inline",
fndecl);
warning_at (loc2, OPT_Winline, "parent subprogram cannot be inlined");
}
DECL_DECLARED_INLINE_P (parent_decl) = 0;
DECL_UNINLINABLE (parent_decl) = 1;
}
}

static tree
get_type_length (tree type, tree result_type)
{
tree comp_type = get_base_type (result_type);
tree base_type = maybe_character_type (get_base_type (type));
tree lb = convert (base_type, TYPE_MIN_VALUE (type));
tree hb = convert (base_type, TYPE_MAX_VALUE (type));
tree length
= build_binary_op (PLUS_EXPR, comp_type,
build_binary_op (MINUS_EXPR, comp_type,
convert (comp_type, hb),
convert (comp_type, lb)),
build_int_cst (comp_type, 1));
length
= build_cond_expr (result_type,
build_binary_op (GE_EXPR, boolean_type_node, hb, lb),
convert (result_type, length),
build_int_cst (result_type, 0));
return length;
}
static tree
Attribute_to_gnu (Node_Id gnat_node, tree *gnu_result_type_p, int attribute)
{
const Node_Id gnat_prefix = Prefix (gnat_node);
tree gnu_prefix = gnat_to_gnu (gnat_prefix);
tree gnu_type = TREE_TYPE (gnu_prefix);
tree gnu_expr, gnu_result_type, gnu_result = error_mark_node;
bool prefix_unused = false;
if (TREE_CODE (gnu_prefix) == NULL_EXPR)
{
gnu_result_type = get_unpadded_type (Etype (gnat_node));
*gnu_result_type_p = gnu_result_type;
return build1 (NULL_EXPR, gnu_result_type, TREE_OPERAND (gnu_prefix, 0));
}
switch (attribute)
{
case Attr_Pos:
case Attr_Val:
gnu_expr = gnat_to_gnu (First (Expressions (gnat_node)));
if (attribute == Attr_Pos)
gnu_expr = maybe_character_value (gnu_expr);
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_result = convert (gnu_result_type, gnu_expr);
break;
case Attr_Pred:
case Attr_Succ:
gnu_expr = gnat_to_gnu (First (Expressions (gnat_node)));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_type = maybe_character_type (gnu_result_type);
if (TREE_TYPE (gnu_expr) != gnu_type)
gnu_expr = convert (gnu_type, gnu_expr);
gnu_result
= build_binary_op (attribute == Attr_Pred ? MINUS_EXPR : PLUS_EXPR,
gnu_type, gnu_expr, build_int_cst (gnu_type, 1));
break;
case Attr_Address:
case Attr_Unrestricted_Access:
gnu_expr = remove_conversions (gnu_prefix,
!Must_Be_Byte_Aligned (gnat_node));
if (REFERENCE_CLASS_P (gnu_expr))
gnu_prefix = gnu_expr;
if (attribute == Attr_Address)
gnu_prefix = maybe_unconstrained_array (gnu_prefix);
else if (TARGET_VTABLE_USES_DESCRIPTORS
&& Is_Dispatch_Table_Entity (Etype (gnat_node)))
{
tree gnu_field, t;
bool build_descriptor = (global_bindings_p () != 0);
int i;
vec<constructor_elt, va_gc> *gnu_vec = NULL;
constructor_elt *elt;
gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (!build_descriptor)
{
gnu_result = build_unary_op (ADDR_EXPR, NULL_TREE, gnu_prefix);
gnu_result = fold_convert (build_pointer_type (gnu_result_type),
gnu_result);
gnu_result = build1 (INDIRECT_REF, gnu_result_type, gnu_result);
}
vec_safe_grow (gnu_vec, TARGET_VTABLE_USES_DESCRIPTORS);
elt = (gnu_vec->address () + TARGET_VTABLE_USES_DESCRIPTORS - 1);
for (gnu_field = TYPE_FIELDS (gnu_result_type), i = 0;
i < TARGET_VTABLE_USES_DESCRIPTORS;
gnu_field = DECL_CHAIN (gnu_field), i++)
{
if (build_descriptor)
{
t = build2 (FDESC_EXPR, TREE_TYPE (gnu_field), gnu_prefix,
build_int_cst (NULL_TREE, i));
TREE_CONSTANT (t) = 1;
}
else
t = build3 (COMPONENT_REF, ptr_void_ftype, gnu_result,
gnu_field, NULL_TREE);
elt->index = gnu_field;
elt->value = t;
elt--;
}
gnu_result = gnat_build_constructor (gnu_result_type, gnu_vec);
break;
}
case Attr_Access:
case Attr_Unchecked_Access:
case Attr_Code_Address:
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_result
= build_unary_op (((attribute == Attr_Address
|| attribute == Attr_Unrestricted_Access)
&& !Must_Be_Byte_Aligned (gnat_node))
? ATTR_ADDR_EXPR : ADDR_EXPR,
gnu_result_type, gnu_prefix);
if (attribute == Attr_Code_Address)
{
gnu_expr = remove_conversions (gnu_result, false);
if (TREE_CODE (gnu_expr) == ADDR_EXPR)
TREE_NO_TRAMPOLINE (gnu_expr) = TREE_CONSTANT (gnu_expr) = 1;
if (targetm.calls.custom_function_descriptors == 0)
gnu_result
= build_unary_op (INDIRECT_REF, NULL_TREE,
convert (build_pointer_type (gnu_result_type),
gnu_result));
}
else if (attribute == Attr_Access
&& Nkind (gnat_prefix) == N_Identifier
&& is_cplusplus_method (Entity (gnat_prefix)))
post_error ("access to C++ constructor or member function not allowed",
gnat_node);
else if (TREE_CODE (TREE_TYPE (gnu_prefix)) == FUNCTION_TYPE)
{
gnu_expr = remove_conversions (gnu_result, false);
if (TREE_CODE (gnu_expr) == ADDR_EXPR
&& decl_function_context (TREE_OPERAND (gnu_expr, 0)))
{
set_expr_location_from_node (gnu_expr, gnat_node);
check_inlining_for_nested_subprog (TREE_OPERAND (gnu_expr, 0));
if ((attribute == Attr_Access
|| attribute == Attr_Unrestricted_Access)
&& targetm.calls.custom_function_descriptors > 0
&& Can_Use_Internal_Rep (Etype (gnat_node)))
FUNC_ADDR_BY_DESCRIPTOR (gnu_expr) = 1;
else if (targetm.calls.custom_function_descriptors != 0)
Check_Implicit_Dynamic_Code_Allowed (gnat_node);
}
}
break;
case Attr_Pool_Address:
{
tree gnu_ptr = gnu_prefix;
tree gnu_obj_type;
gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (TYPE_IS_FAT_POINTER_P (TREE_TYPE (gnu_ptr)))
gnu_ptr
= convert (build_pointer_type
(TYPE_OBJECT_RECORD_TYPE
(TYPE_UNCONSTRAINED_ARRAY (TREE_TYPE (gnu_ptr)))),
gnu_ptr);
gnu_obj_type = TREE_TYPE (TREE_TYPE (gnu_ptr));
if (TYPE_IS_THIN_POINTER_P (TREE_TYPE (gnu_ptr)))
gnu_ptr
= build_binary_op (POINTER_PLUS_EXPR, TREE_TYPE (gnu_ptr),
gnu_ptr,
fold_build1 (NEGATE_EXPR, sizetype,
byte_position
(DECL_CHAIN
TYPE_FIELDS ((gnu_obj_type)))));
gnu_result = convert (gnu_result_type, gnu_ptr);
}
break;
case Attr_Size:
case Attr_Object_Size:
case Attr_Value_Size:
case Attr_Max_Size_In_Storage_Elements:
gnu_expr = gnu_prefix;
while (TREE_CODE (gnu_expr) == NOP_EXPR
|| (TREE_CODE (gnu_expr) == VIEW_CONVERT_EXPR
&& TREE_CODE (TREE_TYPE (gnu_expr)) == RECORD_TYPE
&& TREE_CODE (TREE_TYPE (TREE_OPERAND (gnu_expr, 0)))
== RECORD_TYPE
&& TYPE_NAME (TREE_TYPE (gnu_expr))
== TYPE_NAME (TREE_TYPE (TREE_OPERAND (gnu_expr, 0)))))
gnu_expr = TREE_OPERAND (gnu_expr, 0);
gnu_prefix = remove_conversions (gnu_prefix, true);
prefix_unused = true;
gnu_type = TREE_TYPE (gnu_prefix);
if (TREE_CODE (gnu_type) == UNCONSTRAINED_ARRAY_TYPE)
{
gnu_type = TYPE_OBJECT_RECORD_TYPE (gnu_type);
if (attribute != Attr_Max_Size_In_Storage_Elements)
gnu_type = TREE_TYPE (DECL_CHAIN (TYPE_FIELDS (gnu_type)));
}
if (TREE_CODE (gnu_prefix) == COMPONENT_REF)
gnu_result = DECL_SIZE (TREE_OPERAND (gnu_prefix, 1));
else if ((TREE_CODE (gnu_prefix) != TYPE_DECL
&& !(TYPE_IS_PADDING_P (gnu_type)
&& TREE_CODE (gnu_expr) == COMPONENT_REF
&& pad_type_has_rm_size (gnu_type)))
|| attribute == Attr_Object_Size
|| attribute == Attr_Max_Size_In_Storage_Elements)
{
if (Nkind (gnat_prefix) == N_Explicit_Dereference)
{
Node_Id gnat_actual_subtype
= Actual_Designated_Subtype (gnat_prefix);
tree gnu_ptr_type
= TREE_TYPE (gnat_to_gnu (Prefix (gnat_prefix)));
if (TYPE_IS_FAT_OR_THIN_POINTER_P (gnu_ptr_type)
&& Present (gnat_actual_subtype))
{
tree gnu_actual_obj_type
= gnat_to_gnu_type (gnat_actual_subtype);
gnu_type
= build_unc_object_type_from_ptr (gnu_ptr_type,
gnu_actual_obj_type,
get_identifier ("SIZE"),
false);
}
}
gnu_result = TYPE_SIZE (gnu_type);
}
else
gnu_result = rm_size (gnu_type);
if (CONTAINS_PLACEHOLDER_P (gnu_result))
{
if (TREE_CODE (gnu_prefix) == TYPE_DECL)
gnu_result = max_size (gnu_result, true);
else
gnu_result = substitute_placeholder_in_expr (gnu_result, gnu_expr);
}
if (TREE_CODE (gnu_type) == RECORD_TYPE
&& TYPE_CONTAINS_TEMPLATE_P (gnu_type))
gnu_result = size_binop (MINUS_EXPR, gnu_result,
DECL_SIZE (TYPE_FIELDS (gnu_type)));
if (attribute == Attr_Max_Size_In_Storage_Elements)
gnu_result = size_binop (CEIL_DIV_EXPR, gnu_result, bitsize_unit_node);
gnu_result_type = get_unpadded_type (Etype (gnat_node));
break;
case Attr_Alignment:
{
unsigned int align;
if (TREE_CODE (gnu_prefix) == COMPONENT_REF
&& TYPE_IS_PADDING_P (TREE_TYPE (TREE_OPERAND (gnu_prefix, 0))))
gnu_prefix = TREE_OPERAND (gnu_prefix, 0);
gnu_type = TREE_TYPE (gnu_prefix);
gnu_result_type = get_unpadded_type (Etype (gnat_node));
prefix_unused = true;
if (TREE_CODE (gnu_prefix) == COMPONENT_REF)
align = DECL_ALIGN (TREE_OPERAND (gnu_prefix, 1)) / BITS_PER_UNIT;
else
{
Entity_Id gnat_type = Etype (gnat_prefix);
unsigned int double_align;
bool is_capped_double, align_clause;
if ((double_align = double_float_alignment) > 0)
is_capped_double
= is_double_float_or_array (gnat_type, &align_clause);
else if ((double_align = double_scalar_alignment) > 0)
is_capped_double
= is_double_scalar_or_array (gnat_type, &align_clause);
else
is_capped_double = align_clause = false;
if (is_capped_double
&& Nkind (gnat_prefix) == N_Identifier
&& Present (Alignment_Clause (Entity (gnat_prefix))))
align_clause = true;
if (is_capped_double && !align_clause)
align = double_align;
else
align = TYPE_ALIGN (gnu_type) / BITS_PER_UNIT;
}
gnu_result = size_int (align);
}
break;
case Attr_First:
case Attr_Last:
case Attr_Range_Length:
prefix_unused = true;
if (INTEGRAL_TYPE_P (gnu_type) || TREE_CODE (gnu_type) == REAL_TYPE)
{
gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (attribute == Attr_First)
gnu_result = TYPE_MIN_VALUE (gnu_type);
else if (attribute == Attr_Last)
gnu_result = TYPE_MAX_VALUE (gnu_type);
else
gnu_result = get_type_length (gnu_type, gnu_result_type);
break;
}
case Attr_Length:
{
int Dimension = (Present (Expressions (gnat_node))
? UI_To_Int (Intval (First (Expressions (gnat_node))))
: 1), i;
struct parm_attr_d *pa = NULL;
Entity_Id gnat_param = Empty;
bool unconstrained_ptr_deref = false;
gnu_prefix = maybe_implicit_deref (gnu_prefix);
gnu_prefix = maybe_unconstrained_array (gnu_prefix);
if (!Is_Constrained (Etype (gnat_prefix)))
switch (Nkind (gnat_prefix))
{
case N_Identifier:
if (Ekind (Entity (gnat_prefix)) == E_In_Parameter)
gnat_param = Entity (gnat_prefix);
break;
case N_Explicit_Dereference:
if (Nkind (Prefix (gnat_prefix)) == N_Identifier
&& Ekind (Entity (Prefix (gnat_prefix))) == E_In_Parameter)
{
if (Can_Never_Be_Null (Entity (Prefix (gnat_prefix))))
gnat_param = Entity (Prefix (gnat_prefix));
}
else
unconstrained_ptr_deref = true;
break;
default:
break;
}
if (TREE_CODE (gnu_prefix) == VIEW_CONVERT_EXPR
&& CONTAINS_PLACEHOLDER_P (TYPE_SIZE (TREE_TYPE (gnu_prefix)))
&& !CONTAINS_PLACEHOLDER_P
(TYPE_SIZE (TREE_TYPE (TREE_OPERAND (gnu_prefix, 0)))))
gnu_type = TREE_TYPE (TREE_OPERAND (gnu_prefix, 0));
else
gnu_type = TREE_TYPE (gnu_prefix);
prefix_unused = true;
gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (TYPE_CONVENTION_FORTRAN_P (gnu_type))
{
int ndim;
tree gnu_type_temp;
for (ndim = 1, gnu_type_temp = gnu_type;
TREE_CODE (TREE_TYPE (gnu_type_temp)) == ARRAY_TYPE
&& TYPE_MULTI_ARRAY_P (TREE_TYPE (gnu_type_temp));
ndim++, gnu_type_temp = TREE_TYPE (gnu_type_temp))
;
Dimension = ndim + 1 - Dimension;
}
for (i = 1; i < Dimension; i++)
gnu_type = TREE_TYPE (gnu_type);
gcc_assert (TREE_CODE (gnu_type) == ARRAY_TYPE);
if (!optimize
&& Present (gnat_param)
&& !(Present (Actual_Subtype (gnat_param))
&& Needs_Debug_Info (Actual_Subtype (gnat_param))))
{
FOR_EACH_VEC_SAFE_ELT (f_parm_attr_cache, i, pa)
if (pa->id == gnat_param && pa->dim == Dimension)
break;
if (!pa)
{
pa = ggc_cleared_alloc<parm_attr_d> ();
pa->id = gnat_param;
pa->dim = Dimension;
vec_safe_push (f_parm_attr_cache, pa);
}
}
if (attribute == Attr_First)
{
if (pa && pa->first)
{
gnu_result = pa->first;
break;
}
gnu_result
= TYPE_MIN_VALUE (TYPE_INDEX_TYPE (TYPE_DOMAIN (gnu_type)));
}
else if (attribute == Attr_Last)
{
if (pa && pa->last)
{
gnu_result = pa->last;
break;
}
gnu_result
= TYPE_MAX_VALUE (TYPE_INDEX_TYPE (TYPE_DOMAIN (gnu_type)));
}
else 
{
if (pa && pa->length)
{
gnu_result = pa->length;
break;
}
gnu_result
= get_type_length (TYPE_INDEX_TYPE (TYPE_DOMAIN (gnu_type)),
gnu_result_type);
}
gnu_result = SUBSTITUTE_PLACEHOLDER_IN_EXPR (gnu_result, gnu_prefix);
if (pa)
{
gnu_result
= build1 (SAVE_EXPR, TREE_TYPE (gnu_result), gnu_result);
switch (attribute)
{
case Attr_First:
pa->first = gnu_result;
break;
case Attr_Last:
pa->last = gnu_result;
break;
case Attr_Length:
case Attr_Range_Length:
pa->length = gnu_result;
break;
default:
gcc_unreachable ();
}
}
else
switch (attribute)
{
case Attr_First:
case Attr_Last:
if (unconstrained_ptr_deref)
gnu_result
= build1 (SAVE_EXPR, TREE_TYPE (gnu_result), gnu_result);
break;
case Attr_Length:
case Attr_Range_Length:
if (TREE_CODE (gnu_result) == COND_EXPR
&& EXPR_P (TREE_OPERAND (gnu_result, 0)))
set_expr_location_from_node (TREE_OPERAND (gnu_result, 0),
gnat_node);
break;
default:
gcc_unreachable ();
}
break;
}
case Attr_Bit_Position:
case Attr_Position:
case Attr_First_Bit:
case Attr_Last_Bit:
case Attr_Bit:
{
poly_int64 bitsize;
poly_int64 bitpos;
tree gnu_offset;
tree gnu_field_bitpos;
tree gnu_field_offset;
tree gnu_inner;
machine_mode mode;
int unsignedp, reversep, volatilep;
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_prefix = remove_conversions (gnu_prefix, true);
prefix_unused = true;
if (attribute == Attr_Bit
&& TREE_CODE (gnu_prefix) != COMPONENT_REF
&& TREE_CODE (gnu_prefix) != FIELD_DECL)
{
gnu_result = integer_zero_node;
break;
}
else
gcc_assert (TREE_CODE (gnu_prefix) == COMPONENT_REF
|| (attribute == Attr_Bit_Position
&& TREE_CODE (gnu_prefix) == FIELD_DECL));
get_inner_reference (gnu_prefix, &bitsize, &bitpos, &gnu_offset,
&mode, &unsignedp, &reversep, &volatilep);
if (TREE_CODE (gnu_prefix) == COMPONENT_REF)
{
gnu_field_bitpos = bit_position (TREE_OPERAND (gnu_prefix, 1));
gnu_field_offset = byte_position (TREE_OPERAND (gnu_prefix, 1));
for (gnu_inner = TREE_OPERAND (gnu_prefix, 0);
TREE_CODE (gnu_inner) == COMPONENT_REF
&& DECL_INTERNAL_P (TREE_OPERAND (gnu_inner, 1));
gnu_inner = TREE_OPERAND (gnu_inner, 0))
{
gnu_field_bitpos
= size_binop (PLUS_EXPR, gnu_field_bitpos,
bit_position (TREE_OPERAND (gnu_inner, 1)));
gnu_field_offset
= size_binop (PLUS_EXPR, gnu_field_offset,
byte_position (TREE_OPERAND (gnu_inner, 1)));
}
}
else if (TREE_CODE (gnu_prefix) == FIELD_DECL)
{
gnu_field_bitpos = bit_position (gnu_prefix);
gnu_field_offset = byte_position (gnu_prefix);
}
else
{
gnu_field_bitpos = bitsize_zero_node;
gnu_field_offset = size_zero_node;
}
switch (attribute)
{
case Attr_Position:
gnu_result = gnu_field_offset;
break;
case Attr_First_Bit:
case Attr_Bit:
gnu_result = size_int (num_trailing_bits (bitpos));
break;
case Attr_Last_Bit:
gnu_result = bitsize_int (num_trailing_bits (bitpos));
gnu_result = size_binop (PLUS_EXPR, gnu_result,
TYPE_SIZE (TREE_TYPE (gnu_prefix)));
if (integer_zerop (gnu_result))
gnu_result = integer_minus_one_node;
else
gnu_result
= size_binop (MINUS_EXPR, gnu_result, bitsize_one_node);
break;
case Attr_Bit_Position:
gnu_result = gnu_field_bitpos;
break;
}
gnu_result = SUBSTITUTE_PLACEHOLDER_IN_EXPR (gnu_result, gnu_prefix);
break;
}
case Attr_Min:
case Attr_Max:
{
tree gnu_lhs = gnat_to_gnu (First (Expressions (gnat_node)));
tree gnu_rhs = gnat_to_gnu (Next (First (Expressions (gnat_node))));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (SCALAR_FLOAT_TYPE_P (gnu_result_type)
&& !Machine_Overflows_On_Target)
{
const bool lhs_side_effects_p = TREE_SIDE_EFFECTS (gnu_lhs);
const bool rhs_side_effects_p = TREE_SIDE_EFFECTS (gnu_rhs);
tree t = builtin_decl_explicit (BUILT_IN_ISNAN);
tree lhs_is_nan, rhs_is_nan;
if (lhs_side_effects_p)
gnu_lhs = gnat_protect_expr (gnu_lhs);
if (rhs_side_effects_p)
gnu_rhs = gnat_protect_expr (gnu_rhs);
lhs_is_nan = fold_build2 (NE_EXPR, boolean_type_node,
build_call_expr (t, 1, gnu_lhs),
integer_zero_node);
rhs_is_nan = fold_build2 (NE_EXPR, boolean_type_node,
build_call_expr (t, 1, gnu_rhs),
integer_zero_node);
gnu_result = build_binary_op (attribute == Attr_Min
? MIN_EXPR : MAX_EXPR,
gnu_result_type, gnu_lhs, gnu_rhs);
gnu_result = fold_build3 (COND_EXPR, gnu_result_type,
rhs_is_nan, gnu_lhs, gnu_result);
gnu_result = fold_build3 (COND_EXPR, gnu_result_type,
lhs_is_nan, gnu_rhs, gnu_result);
if (lhs_side_effects_p)
gnu_result
= build2 (COMPOUND_EXPR, gnu_result_type, gnu_lhs, gnu_result);
if (rhs_side_effects_p)
gnu_result
= build2 (COMPOUND_EXPR, gnu_result_type, gnu_rhs, gnu_result);
}
else
gnu_result = build_binary_op (attribute == Attr_Min
? MIN_EXPR : MAX_EXPR,
gnu_result_type, gnu_lhs, gnu_rhs);
}
break;
case Attr_Passed_By_Reference:
gnu_result = size_int (default_pass_by_ref (gnu_type)
|| must_pass_by_ref (gnu_type));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
break;
case Attr_Component_Size:
if (TREE_CODE (gnu_prefix) == COMPONENT_REF
&& TYPE_IS_PADDING_P (TREE_TYPE (TREE_OPERAND (gnu_prefix, 0))))
gnu_prefix = TREE_OPERAND (gnu_prefix, 0);
gnu_prefix = maybe_implicit_deref (gnu_prefix);
gnu_type = TREE_TYPE (gnu_prefix);
if (TREE_CODE (gnu_type) == UNCONSTRAINED_ARRAY_TYPE)
gnu_type = TREE_TYPE (TREE_TYPE (TYPE_FIELDS (TREE_TYPE (gnu_type))));
while (TREE_CODE (TREE_TYPE (gnu_type)) == ARRAY_TYPE
&& TYPE_MULTI_ARRAY_P (TREE_TYPE (gnu_type)))
gnu_type = TREE_TYPE (gnu_type);
gcc_assert (TREE_CODE (gnu_type) == ARRAY_TYPE);
gnu_result = TYPE_SIZE (TREE_TYPE (gnu_type));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
prefix_unused = true;
break;
case Attr_Descriptor_Size:
gnu_type = TREE_TYPE (gnu_prefix);
gcc_assert (TREE_CODE (gnu_type) == UNCONSTRAINED_ARRAY_TYPE);
gnu_type = TYPE_OBJECT_RECORD_TYPE (gnu_type);
gnu_result = bit_position (DECL_CHAIN (TYPE_FIELDS (gnu_type)));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
prefix_unused = true;
break;
case Attr_Null_Parameter:
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_result
= build_unary_op (INDIRECT_REF, NULL_TREE,
convert (build_pointer_type (gnu_result_type),
integer_zero_node));
TREE_PRIVATE (gnu_result) = 1;
break;
case Attr_Mechanism_Code:
{
Entity_Id gnat_obj = Entity (gnat_prefix);
int code;
prefix_unused = true;
gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (Present (Expressions (gnat_node)))
{
int i = UI_To_Int (Intval (First (Expressions (gnat_node))));
for (gnat_obj = First_Formal (gnat_obj); i > 1;
i--, gnat_obj = Next_Formal (gnat_obj))
;
}
code = Mechanism (gnat_obj);
if (code == Default)
code = ((present_gnu_tree (gnat_obj)
&& (DECL_BY_REF_P (get_gnu_tree (gnat_obj))
|| ((TREE_CODE (get_gnu_tree (gnat_obj))
== PARM_DECL)
&& (DECL_BY_COMPONENT_PTR_P
(get_gnu_tree (gnat_obj))))))
? By_Reference : By_Copy);
gnu_result = convert (gnu_result_type, size_int (- code));
}
break;
case Attr_Model:
case Attr_Machine:
prefix_unused = true;
gnu_expr = gnat_to_gnu (First (Expressions (gnat_node)));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_result = convert (gnu_result_type, gnu_expr);
if (TREE_CODE (gnu_result) != REAL_CST
&& fp_arith_may_widen
&& TYPE_PRECISION (gnu_result_type)
< TYPE_PRECISION (longest_float_type_node))
{
tree rec_type = make_node (RECORD_TYPE);
tree field
= create_field_decl (get_identifier ("OBJ"), gnu_result_type,
rec_type, NULL_TREE, NULL_TREE, 0, 0);
tree rec_val, asm_expr;
finish_record_type (rec_type, field, 0, false);
rec_val = build_constructor_single (rec_type, field, gnu_result);
rec_val = build1 (SAVE_EXPR, rec_type, rec_val);
asm_expr
= build5 (ASM_EXPR, void_type_node,
build_string (0, ""),
tree_cons (build_tree_list (NULL_TREE,
build_string (2, "=m")),
rec_val, NULL_TREE),
tree_cons (build_tree_list (NULL_TREE,
build_string (1, "m")),
rec_val, NULL_TREE),
NULL_TREE, NULL_TREE);
ASM_VOLATILE_P (asm_expr) = 1;
gnu_result
= build_compound_expr (gnu_result_type, asm_expr,
build_component_ref (rec_val, field,
false));
}
break;
case Attr_Deref:
prefix_unused = true;
gnu_expr = gnat_to_gnu (First (Expressions (gnat_node)));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_expr
= convert (build_pointer_type_for_mode (gnu_result_type, ptr_mode,
true),
gnu_expr);
gnu_result = build_unary_op (INDIRECT_REF, NULL_TREE, gnu_expr);
break;
default:
gcc_unreachable ();
}
if (prefix_unused
&& TREE_SIDE_EFFECTS (gnu_prefix)
&& !Is_Entity_Name (gnat_prefix))
gnu_result
= build_compound_expr (TREE_TYPE (gnu_result), gnu_prefix, gnu_result);
*gnu_result_type_p = gnu_result_type;
return gnu_result;
}

static tree
Case_Statement_to_gnu (Node_Id gnat_node)
{
tree gnu_result, gnu_expr, gnu_type, gnu_label;
Node_Id gnat_when;
location_t end_locus;
bool may_fallthru = false;
gnu_expr = gnat_to_gnu (Expression (gnat_node));
gnu_expr = convert (get_base_type (TREE_TYPE (gnu_expr)), gnu_expr);
gnu_expr = maybe_character_value (gnu_expr);
gnu_type = TREE_TYPE (gnu_expr);
if (!Sloc_to_locus (End_Location (gnat_node), &end_locus))
end_locus = input_location;
gnu_label = create_artificial_label (end_locus);
start_stmt_group ();
for (gnat_when = First_Non_Pragma (Alternatives (gnat_node));
Present (gnat_when);
gnat_when = Next_Non_Pragma (gnat_when))
{
bool choices_added_p = false;
Node_Id gnat_choice;
for (gnat_choice = First (Discrete_Choices (gnat_when));
Present (gnat_choice);
gnat_choice = Next (gnat_choice))
{
tree gnu_low = NULL_TREE, gnu_high = NULL_TREE;
tree label = create_artificial_label (input_location);
switch (Nkind (gnat_choice))
{
case N_Range:
gnu_low = gnat_to_gnu (Low_Bound (gnat_choice));
gnu_high = gnat_to_gnu (High_Bound (gnat_choice));
break;
case N_Subtype_Indication:
gnu_low = gnat_to_gnu (Low_Bound (Range_Expression
(Constraint (gnat_choice))));
gnu_high = gnat_to_gnu (High_Bound (Range_Expression
(Constraint (gnat_choice))));
break;
case N_Identifier:
case N_Expanded_Name:
if (Is_Type (Entity (gnat_choice)))
{
tree gnu_type = get_unpadded_type (Entity (gnat_choice));
gnu_low = TYPE_MIN_VALUE (gnu_type);
gnu_high = TYPE_MAX_VALUE (gnu_type);
break;
}
case N_Character_Literal:
case N_Integer_Literal:
gnu_low = gnat_to_gnu (gnat_choice);
break;
case N_Others_Choice:
break;
default:
gcc_unreachable ();
}
gcc_assert (!gnu_low  || TREE_CODE (gnu_low)  == INTEGER_CST);
gcc_assert (!gnu_high || TREE_CODE (gnu_high) == INTEGER_CST);
if (gnu_low && TREE_TYPE (gnu_low) != gnu_type)
gnu_low = convert (gnu_type, gnu_low);
if (gnu_high && TREE_TYPE (gnu_high) != gnu_type)
gnu_high = convert (gnu_type, gnu_high);
add_stmt_with_node (build_case_label (gnu_low, gnu_high, label),
gnat_choice);
choices_added_p = true;
}
if (choices_added_p)
{
const bool is_case_expression
= (Nkind (Parent (gnat_node)) == N_Expression_With_Actions);
tree group
= build_stmt_group (Statements (gnat_when), !is_case_expression);
bool group_may_fallthru = block_may_fallthru (group);
add_stmt (group);
if (group_may_fallthru)
{
tree stmt = build1 (GOTO_EXPR, void_type_node, gnu_label);
SET_EXPR_LOCATION (stmt, end_locus);
add_stmt (stmt);
may_fallthru = true;
}
}
}
if (may_fallthru)
add_stmt (build1 (LABEL_EXPR, void_type_node, gnu_label));
gnu_result = build2 (SWITCH_EXPR, gnu_type, gnu_expr, end_stmt_group ());
return gnu_result;
}

static inline bool
inside_loop_p (void)
{
return !vec_safe_is_empty (gnu_loop_stack);
}
static struct loop_info_d *
find_loop_for (tree expr, tree *disp = NULL, bool *neg_p = NULL)
{
tree var, add, cst;
bool minus_p;
struct loop_info_d *iter = NULL;
unsigned int i;
if (is_simple_additive_expression (expr, &add, &cst, &minus_p))
{
var = add;
if (disp)
*disp = cst;
if (neg_p)
*neg_p = minus_p;
}
else
{
var = expr;
if (disp)
*disp =  NULL_TREE;
if (neg_p)
*neg_p = false;
}
var = remove_conversions (var, false);
if (TREE_CODE (var) != VAR_DECL)
return NULL;
if (decl_function_context (var) != current_function_decl)
return NULL;
gcc_assert (vec_safe_length (gnu_loop_stack) > 0);
FOR_EACH_VEC_ELT_REVERSE (*gnu_loop_stack, i, iter)
if (var == iter->loop_var)
break;
return iter;
}
static bool
can_equal_min_or_max_val_p (tree val, tree type, bool max)
{
tree min_or_max_val = (max ? TYPE_MAX_VALUE (type) : TYPE_MIN_VALUE (type));
if (TREE_CODE (min_or_max_val) != INTEGER_CST)
return true;
if (TREE_CODE (val) == NOP_EXPR)
val = (max
? TYPE_MAX_VALUE (TREE_TYPE (TREE_OPERAND (val, 0)))
: TYPE_MIN_VALUE (TREE_TYPE (TREE_OPERAND (val, 0))));
if (TREE_CODE (val) != INTEGER_CST)
return true;
if (max)
return tree_int_cst_lt (val, min_or_max_val) == 0;
else
return tree_int_cst_lt (min_or_max_val, val) == 0;
}
static inline bool
can_equal_min_val_p (tree val, tree type, bool reverse)
{
return can_equal_min_or_max_val_p (val, type, reverse);
}
static inline bool
can_equal_max_val_p (tree val, tree type, bool reverse)
{
return can_equal_min_or_max_val_p (val, type, !reverse);
}
static bool
can_be_lower_p (tree val1, tree val2)
{
if (TREE_CODE (val1) == NOP_EXPR)
val1 = TYPE_MIN_VALUE (TREE_TYPE (TREE_OPERAND (val1, 0)));
if (TREE_CODE (val1) != INTEGER_CST)
return true;
if (TREE_CODE (val2) == NOP_EXPR)
val2 = TYPE_MAX_VALUE (TREE_TYPE (TREE_OPERAND (val2, 0)));
if (TREE_CODE (val2) != INTEGER_CST)
return true;
return tree_int_cst_lt (val1, val2);
}
static bool
make_invariant (tree *expr1, tree *expr2)
{
tree inv_expr1 = gnat_invariant_expr (*expr1);
tree inv_expr2 = gnat_invariant_expr (*expr2);
if (inv_expr1)
*expr1 = inv_expr1;
if (inv_expr2)
*expr2 = inv_expr2;
return inv_expr1 && inv_expr2;
}
static tree
scan_rhs_r (tree *tp, int *walk_subtrees, void *data)
{
bitmap *params = (bitmap *)data;
tree t = *tp;
if (IS_TYPE_OR_DECL_P (t))
*walk_subtrees = 0;
if (TREE_CODE (t) == PARM_DECL && bitmap_bit_p (*params, DECL_UID (t)))
return t;
return NULL_TREE;
}
static bool
independent_iterations_p (tree stmt_list)
{
tree_stmt_iterator tsi;
bitmap params = BITMAP_GGC_ALLOC();
auto_vec<tree> rhs;
tree iter;
int i;
if (TREE_CODE (stmt_list) == BIND_EXPR)
stmt_list = BIND_EXPR_BODY (stmt_list);
for (tsi = tsi_start (stmt_list); !tsi_end_p (tsi); tsi_next (&tsi))
{
tree stmt = tsi_stmt (tsi);
switch (TREE_CODE (stmt))
{
case COND_EXPR:
{
if (COND_EXPR_ELSE (stmt))
return false;
if (TREE_CODE (COND_EXPR_THEN (stmt)) != CALL_EXPR)
return false;
tree func = get_callee_fndecl (COND_EXPR_THEN (stmt));
if (!(func && TREE_THIS_VOLATILE (func)))
return false;
break;
}
case MODIFY_EXPR:
{
tree lhs = TREE_OPERAND (stmt, 0);
while (handled_component_p (lhs))
lhs = TREE_OPERAND (lhs, 0);
if (TREE_CODE (lhs) != INDIRECT_REF)
return false;
lhs = TREE_OPERAND (lhs, 0);
if (!(TREE_CODE (lhs) == PARM_DECL
&& DECL_RESTRICTED_ALIASING_P (lhs)))
return false;
bitmap_set_bit (params, DECL_UID (lhs));
rhs.safe_push (TREE_OPERAND (stmt, 1));
break;
}
default:
return false;
}
}
FOR_EACH_VEC_ELT (rhs, i, iter)
if (walk_tree_without_duplicates (&iter, scan_rhs_r, &params))
return false;
return true;
}
static tree
Loop_Statement_to_gnu (Node_Id gnat_node)
{
const Node_Id gnat_iter_scheme = Iteration_Scheme (gnat_node);
struct loop_info_d *gnu_loop_info = ggc_cleared_alloc<loop_info_d> ();
tree gnu_loop_stmt = build4 (LOOP_STMT, void_type_node, NULL_TREE,
NULL_TREE, NULL_TREE, NULL_TREE);
tree gnu_loop_label = create_artificial_label (input_location);
tree gnu_cond_expr = NULL_TREE, gnu_low = NULL_TREE, gnu_high = NULL_TREE;
tree gnu_result;
vec_safe_push (gnu_loop_stack, gnu_loop_info);
set_expr_location_from_node (gnu_loop_stmt, gnat_node);
Sloc_to_locus (Sloc (End_Label (gnat_node)),
&DECL_SOURCE_LOCATION (gnu_loop_label));
LOOP_STMT_LABEL (gnu_loop_stmt) = gnu_loop_label;
gnu_loop_info->stmt = gnu_loop_stmt;
if (No (gnat_iter_scheme))
;
else if (Present (Condition (gnat_iter_scheme)))
LOOP_STMT_COND (gnu_loop_stmt)
= gnat_to_gnu (Condition (gnat_iter_scheme));
else
{
Node_Id gnat_loop_spec = Loop_Parameter_Specification (gnat_iter_scheme);
Entity_Id gnat_loop_var = Defining_Entity (gnat_loop_spec);
Entity_Id gnat_type = Etype (gnat_loop_var);
tree gnu_type = get_unpadded_type (gnat_type);
tree gnu_base_type = maybe_character_type (get_base_type (gnu_type));
tree gnu_one_node = build_int_cst (gnu_base_type, 1);
tree gnu_loop_var, gnu_loop_iv, gnu_first, gnu_last, gnu_stmt;
enum tree_code update_code, test_code, shift_code;
bool reverse = Reverse_Present (gnat_loop_spec), use_iv = false;
gnu_low = convert (gnu_base_type, TYPE_MIN_VALUE (gnu_type));
gnu_high = convert (gnu_base_type, TYPE_MAX_VALUE (gnu_type));
if (reverse)
{
gnu_first = gnu_high;
gnu_last = gnu_low;
update_code = MINUS_NOMOD_EXPR;
test_code = GE_EXPR;
shift_code = PLUS_NOMOD_EXPR;
}
else
{
gnu_first = gnu_low;
gnu_last = gnu_high;
update_code = PLUS_NOMOD_EXPR;
test_code = LE_EXPR;
shift_code = MINUS_NOMOD_EXPR;
}
if (optimize)
{
if (!can_equal_min_val_p (gnu_first, gnu_base_type, reverse))
;
else
{
if (TYPE_PRECISION (gnu_base_type)
> TYPE_PRECISION (size_type_node))
gnu_base_type
= gnat_type_for_size (TYPE_PRECISION (gnu_base_type), 1);
else
gnu_base_type = size_type_node;
gnu_first = convert (gnu_base_type, gnu_first);
gnu_last = convert (gnu_base_type, gnu_last);
gnu_one_node = build_int_cst (gnu_base_type, 1);
use_iv = true;
}
gnu_first
= build_binary_op (shift_code, gnu_base_type, gnu_first,
gnu_one_node);
LOOP_STMT_TOP_UPDATE_P (gnu_loop_stmt) = 1;
LOOP_STMT_BOTTOM_COND_P (gnu_loop_stmt) = 1;
}
else
{
if (!can_equal_max_val_p (gnu_last, gnu_base_type, reverse))
;
else if (!can_equal_min_val_p (gnu_first, gnu_base_type, reverse)
&& !can_equal_min_val_p (gnu_last, gnu_base_type, reverse))
{
gnu_first
= build_binary_op (shift_code, gnu_base_type, gnu_first,
gnu_one_node);
gnu_last
= build_binary_op (shift_code, gnu_base_type, gnu_last,
gnu_one_node);
LOOP_STMT_TOP_UPDATE_P (gnu_loop_stmt) = 1;
}
else
LOOP_STMT_BOTTOM_COND_P (gnu_loop_stmt) = 1;
}
if (LOOP_STMT_BOTTOM_COND_P (gnu_loop_stmt))
{
test_code = NE_EXPR;
if (can_be_lower_p (gnu_high, gnu_low))
{
gnu_cond_expr
= build3 (COND_EXPR, void_type_node,
build_binary_op (LE_EXPR, boolean_type_node,
gnu_low, gnu_high),
NULL_TREE, alloc_stmt_list ());
set_expr_location_from_node (gnu_cond_expr, gnat_loop_spec);
}
}
start_stmt_group ();
gnat_pushlevel ();
if (use_iv)
{
gnu_loop_iv
= create_init_temporary ("I", gnu_first, &gnu_stmt, gnat_loop_var);
add_stmt (gnu_stmt);
gnu_first = NULL_TREE;
}
else
gnu_loop_iv = NULL_TREE;
gnu_loop_var = gnat_to_gnu_entity (gnat_loop_var, gnu_first, true);
if (DECL_BY_REF_P (gnu_loop_var))
gnu_loop_var = build_unary_op (INDIRECT_REF, NULL_TREE, gnu_loop_var);
else if (use_iv)
{
gcc_assert (DECL_LOOP_PARM_P (gnu_loop_var));
SET_DECL_INDUCTION_VAR (gnu_loop_var, gnu_loop_iv);
}
gnu_loop_info->loop_var = gnu_loop_var;
gnu_loop_info->low_bound = gnu_low;
gnu_loop_info->high_bound = gnu_high;
gnu_loop_var = convert (gnu_base_type, gnu_loop_var);
if (use_iv)
LOOP_STMT_COND (gnu_loop_stmt)
= build_binary_op (test_code, boolean_type_node, gnu_loop_iv,
gnu_last);
else
LOOP_STMT_COND (gnu_loop_stmt)
= build_binary_op (test_code, boolean_type_node, gnu_loop_var,
gnu_last);
if (use_iv)
{
gnu_stmt
= build_binary_op (MODIFY_EXPR, NULL_TREE, gnu_loop_iv,
build_binary_op (update_code, gnu_base_type,
gnu_loop_iv, gnu_one_node));
set_expr_location_from_node (gnu_stmt, gnat_iter_scheme);
append_to_statement_list (gnu_stmt,
&LOOP_STMT_UPDATE (gnu_loop_stmt));
gnu_stmt
= build_binary_op (MODIFY_EXPR, NULL_TREE, gnu_loop_var,
gnu_loop_iv);
set_expr_location_from_node (gnu_stmt, gnat_iter_scheme);
append_to_statement_list (gnu_stmt,
&LOOP_STMT_UPDATE (gnu_loop_stmt));
}
else
{
gnu_stmt
= build_binary_op (MODIFY_EXPR, NULL_TREE, gnu_loop_var,
build_binary_op (update_code, gnu_base_type,
gnu_loop_var, gnu_one_node));
set_expr_location_from_node (gnu_stmt, gnat_iter_scheme);
LOOP_STMT_UPDATE (gnu_loop_stmt) = gnu_stmt;
}
}
if (Present (Identifier (gnat_node)))
save_gnu_tree (Entity (Identifier (gnat_node)), gnu_loop_label, true);
LOOP_STMT_BODY (gnu_loop_stmt)
= build_stmt_group (Statements (gnat_node), true);
TREE_SIDE_EFFECTS (gnu_loop_stmt) = 1;
if (Present (gnat_iter_scheme) && No (Condition (gnat_iter_scheme)))
{
if (vec_safe_length (gnu_loop_info->checks) > 0
&& (make_invariant (&gnu_low, &gnu_high) || optimize >= 3))
{
struct range_check_info_d *rci;
unsigned int i, n_remaining_checks = 0;
FOR_EACH_VEC_ELT (*gnu_loop_info->checks, i, rci)
{
tree low_ok, high_ok;
if (rci->low_bound)
{
tree gnu_adjusted_low = convert (rci->type, gnu_low);
if (rci->disp)
gnu_adjusted_low
= fold_build2 (rci->neg_p ? MINUS_EXPR : PLUS_EXPR,
rci->type, gnu_adjusted_low, rci->disp);
low_ok
= build_binary_op (GE_EXPR, boolean_type_node,
gnu_adjusted_low, rci->low_bound);
}
else
low_ok = boolean_true_node;
if (rci->high_bound)
{
tree gnu_adjusted_high = convert (rci->type, gnu_high);
if (rci->disp)
gnu_adjusted_high
= fold_build2 (rci->neg_p ? MINUS_EXPR : PLUS_EXPR,
rci->type, gnu_adjusted_high, rci->disp);
high_ok
= build_binary_op (LE_EXPR, boolean_type_node,
gnu_adjusted_high, rci->high_bound);
}
else
high_ok = boolean_true_node;
tree range_ok
= build_binary_op (TRUTH_ANDIF_EXPR, boolean_type_node,
low_ok, high_ok);
rci->invariant_cond
= build_unary_op (TRUTH_NOT_EXPR, boolean_type_node, range_ok);
if (rci->invariant_cond == boolean_false_node)
TREE_OPERAND (rci->inserted_cond, 0) = rci->invariant_cond;
else
n_remaining_checks++;
}
if (IN_RANGE (n_remaining_checks, 1, 3)
&& optimize >= 2
&& !optimize_size)
FOR_EACH_VEC_ELT (*gnu_loop_info->checks, i, rci)
if (rci->invariant_cond != boolean_false_node)
{
TREE_OPERAND (rci->inserted_cond, 0) = rci->invariant_cond;
if (optimize >= 3)
add_stmt_with_node_force (rci->inserted_cond, gnat_node);
}
}
if (optimize >= 3
&& independent_iterations_p (LOOP_STMT_BODY (gnu_loop_stmt)))
LOOP_STMT_IVDEP (gnu_loop_stmt) = 1;
add_stmt (gnu_loop_stmt);
gnat_poplevel ();
gnu_loop_stmt = end_stmt_group ();
}
if (gnu_cond_expr)
{
COND_EXPR_THEN (gnu_cond_expr) = gnu_loop_stmt;
TREE_SIDE_EFFECTS (gnu_cond_expr) = 1;
gnu_result = gnu_cond_expr;
}
else
gnu_result = gnu_loop_stmt;
gnu_loop_stack->pop ();
return gnu_result;
}

struct nrv_data
{
bitmap nrv;
tree result;
Node_Id gnat_ret;
hash_set<tree> *visited;
};
static inline bool
is_nrv_p (bitmap nrv, tree t)
{
return TREE_CODE (t) == VAR_DECL && bitmap_bit_p (nrv, DECL_UID (t));
}
static tree
prune_nrv_r (tree *tp, int *walk_subtrees, void *data)
{
struct nrv_data *dp = (struct nrv_data *)data;
tree t = *tp;
if (IS_TYPE_OR_DECL_P (t))
*walk_subtrees = 0;
if (is_nrv_p (dp->nrv, t))
bitmap_clear_bit (dp->nrv, DECL_UID (t));
return NULL_TREE;
}
static bool
prune_nrv_in_block (bitmap nrv, tree block)
{
bool has_nrv = false;
tree t;
for (t = BLOCK_SUBBLOCKS (block); t; t = BLOCK_CHAIN (t))
has_nrv |= prune_nrv_in_block (nrv, t);
for (t = BLOCK_VARS (block); t; t = DECL_CHAIN (t))
if (is_nrv_p (nrv, t))
{
if (has_nrv)
bitmap_clear_bit (nrv, DECL_UID (t));
else
has_nrv = true;
}
return has_nrv;
}
static tree
finalize_nrv_r (tree *tp, int *walk_subtrees, void *data)
{
struct nrv_data *dp = (struct nrv_data *)data;
tree t = *tp;
if (TYPE_P (t))
*walk_subtrees = 0;
else if (TREE_CODE (t) == RETURN_EXPR
&& TREE_CODE (TREE_OPERAND (t, 0)) == INIT_EXPR)
{
tree ret_val = TREE_OPERAND (TREE_OPERAND (t, 0), 1);
if (gnat_useless_type_conversion (ret_val))
ret_val = TREE_OPERAND (ret_val, 0);
if (is_nrv_p (dp->nrv, ret_val))
TREE_OPERAND (t, 0) = dp->result;
}
else if (TREE_CODE (t) == DECL_EXPR
&& is_nrv_p (dp->nrv, DECL_EXPR_DECL (t)))
{
tree var = DECL_EXPR_DECL (t), init;
if (DECL_INITIAL (var))
{
init = build_binary_op (INIT_EXPR, NULL_TREE, dp->result,
DECL_INITIAL (var));
SET_EXPR_LOCATION (init, EXPR_LOCATION (t));
DECL_INITIAL (var) = NULL_TREE;
}
else
init = build_empty_stmt (EXPR_LOCATION (t));
*tp = init;
SET_DECL_VALUE_EXPR (var, dp->result);
DECL_HAS_VALUE_EXPR_P (var) = 1;
DECL_SIZE (var) = bitsize_unit_node;
DECL_SIZE_UNIT (var) = size_one_node;
}
else if (is_nrv_p (dp->nrv, t))
*tp = convert (TREE_TYPE (t), dp->result);
if (dp->visited->add (*tp))
*walk_subtrees = 0;
return NULL_TREE;
}
static tree
finalize_nrv_unc_r (tree *tp, int *walk_subtrees, void *data)
{
struct nrv_data *dp = (struct nrv_data *)data;
tree t = *tp;
if (TYPE_P (t))
*walk_subtrees = 0;
else if (TREE_CODE (t) == BIND_EXPR)
walk_tree (&BIND_EXPR_BODY (t), finalize_nrv_unc_r, data, NULL);
else if (TREE_CODE (t) == RETURN_EXPR
&& TREE_CODE (TREE_OPERAND (t, 0)) == INIT_EXPR)
{
tree ret_val = TREE_OPERAND (TREE_OPERAND (t, 0), 1);
if (TREE_CODE (ret_val) == COMPOUND_EXPR
&& TREE_CODE (TREE_OPERAND (ret_val, 0)) == INIT_EXPR)
{
tree rhs = TREE_OPERAND (TREE_OPERAND (ret_val, 0), 1);
if (TYPE_IS_FAT_POINTER_P (TREE_TYPE (ret_val)))
ret_val = CONSTRUCTOR_ELT (rhs, 1)->value;
else
ret_val = rhs;
}
if (gnat_useless_type_conversion (ret_val)
|| TREE_CODE (ret_val) == VIEW_CONVERT_EXPR)
ret_val = TREE_OPERAND (ret_val, 0);
if (TREE_CODE (ret_val) == COMPONENT_REF
&& TYPE_IS_PADDING_P (TREE_TYPE (TREE_OPERAND (ret_val, 0))))
ret_val = TREE_OPERAND (ret_val, 0);
if (is_nrv_p (dp->nrv, ret_val))
TREE_OPERAND (TREE_OPERAND (t, 0), 1)
= TREE_OPERAND (DECL_INITIAL (ret_val), 0);
}
else if (TREE_CODE (t) == DECL_EXPR
&& is_nrv_p (dp->nrv, DECL_EXPR_DECL (t)))
{
tree saved_current_function_decl = current_function_decl;
tree var = DECL_EXPR_DECL (t);
tree alloc, p_array, new_var, new_ret;
vec<constructor_elt, va_gc> *v;
vec_alloc (v, 2);
current_function_decl = decl_function_context (var);
start_stmt_group ();
gnat_pushlevel ();
alloc = build_allocator (TREE_TYPE (var), DECL_INITIAL (var),
TREE_TYPE (dp->result),
Procedure_To_Call (dp->gnat_ret),
Storage_Pool (dp->gnat_ret),
Empty, false);
new_var
= build_decl (DECL_SOURCE_LOCATION (var), VAR_DECL, DECL_NAME (var),
build_reference_type (TREE_TYPE (var)));
DECL_BY_REFERENCE (new_var) = 1;
if (TYPE_IS_FAT_POINTER_P (TREE_TYPE (alloc)))
{
tree cst = TREE_OPERAND (alloc, 1);
DECL_INITIAL (new_var)
= build2 (COMPOUND_EXPR, TREE_TYPE (new_var),
TREE_OPERAND (alloc, 0),
CONSTRUCTOR_ELT (cst, 0)->value);
p_array = TYPE_FIELDS (TREE_TYPE (alloc));
CONSTRUCTOR_APPEND_ELT (v, p_array,
fold_convert (TREE_TYPE (p_array), new_var));
CONSTRUCTOR_APPEND_ELT (v, DECL_CHAIN (p_array),
CONSTRUCTOR_ELT (cst, 1)->value);
new_ret = build_constructor (TREE_TYPE (alloc), v);
}
else
{
DECL_INITIAL (new_var) = alloc;
new_ret = fold_convert (TREE_TYPE (alloc), new_var);
}
gnat_pushdecl (new_var, Empty);
gnat_zaplevel ();
*tp = end_stmt_group ();
current_function_decl = saved_current_function_decl;
DECL_CHAIN (new_var) = DECL_CHAIN (var);
DECL_CHAIN (var) = new_var;
DECL_IGNORED_P (var) = 1;
DECL_INITIAL (var)
= build2 (COMPOUND_EXPR, TREE_TYPE (var), new_ret,
build1 (INDIRECT_REF, TREE_TYPE (var), new_var));
DECL_CONTEXT (var) = NULL_TREE;
}
else if (is_nrv_p (dp->nrv, t))
*tp = TREE_OPERAND (DECL_INITIAL (t), 1);
if (dp->visited->add (*tp))
*walk_subtrees = 0;
return NULL_TREE;
}
static void
finalize_nrv (tree fndecl, bitmap nrv, vec<tree, va_gc> *other, Node_Id gnat_ret)
{
struct cgraph_node *node;
struct nrv_data data;
walk_tree_fn func;
unsigned int i;
tree iter;
gcc_assert (!TYPE_IS_BY_REFERENCE_P (TREE_TYPE (TREE_TYPE (fndecl))));
data.nrv = nrv;
data.result = NULL_TREE;
data.gnat_ret = Empty;
data.visited = NULL;
FOR_EACH_VEC_SAFE_ELT (other, i, iter)
walk_tree_without_duplicates (&iter, prune_nrv_r, &data);
if (bitmap_empty_p (nrv))
return;
node = cgraph_node::get_create (fndecl);
for (node = node->nested; node; node = node->next_nested)
walk_tree_without_duplicates (&DECL_SAVED_TREE (node->decl), prune_nrv_r,
&data);
if (bitmap_empty_p (nrv))
return;
if (!prune_nrv_in_block (nrv, DECL_INITIAL (fndecl)))
return;
data.nrv = nrv;
data.result = DECL_RESULT (fndecl);
data.gnat_ret = gnat_ret;
data.visited = new hash_set<tree>;
if (TYPE_RETURN_UNCONSTRAINED_P (TREE_TYPE (fndecl)))
func = finalize_nrv_unc_r;
else
func = finalize_nrv_r;
walk_tree (&DECL_SAVED_TREE (fndecl), func, &data, NULL);
delete data.visited;
}
static bool
return_value_ok_for_nrv_p (tree ret_obj, tree ret_val)
{
if (TREE_CODE (ret_val) != VAR_DECL)
return false;
if (TREE_THIS_VOLATILE (ret_val))
return false;
if (DECL_CONTEXT (ret_val) != current_function_decl)
return false;
if (TREE_STATIC (ret_val))
return false;
if (ret_obj && TREE_ADDRESSABLE (ret_val))
return false;
if (ret_obj && DECL_ALIGN (ret_val) > DECL_ALIGN (ret_obj))
return false;
if (!ret_obj
&& DECL_INITIAL (ret_val)
&& TREE_CODE (DECL_INITIAL (ret_val)) == NULL_EXPR)
return false;
return true;
}
static tree
build_return_expr (tree ret_obj, tree ret_val)
{
tree result_expr;
if (ret_val)
{
tree operation_type = TREE_TYPE (ret_obj);
if (operation_type != TREE_TYPE (ret_val))
ret_val = convert (operation_type, ret_val);
result_expr = build2 (INIT_EXPR, void_type_node, ret_obj, ret_val);
if (optimize
&& AGGREGATE_TYPE_P (operation_type)
&& !TYPE_IS_FAT_POINTER_P (operation_type)
&& TYPE_MODE (operation_type) == BLKmode
&& aggregate_value_p (operation_type, current_function_decl))
{
if (gnat_useless_type_conversion (ret_val))
ret_val = TREE_OPERAND (ret_val, 0);
if (return_value_ok_for_nrv_p (ret_obj, ret_val))
{
if (!f_named_ret_val)
f_named_ret_val = BITMAP_GGC_ALLOC ();
bitmap_set_bit (f_named_ret_val, DECL_UID (ret_val));
}
else if (EXPR_P (ret_val))
vec_safe_push (f_other_ret_val, ret_val);
}
}
else
result_expr = ret_obj;
return build1 (RETURN_EXPR, void_type_node, result_expr);
}

static void
Subprogram_Body_to_gnu (Node_Id gnat_node)
{
Entity_Id gnat_param;
Entity_Id gnat_subprog_id
= (Present (Corresponding_Spec (gnat_node))
? Corresponding_Spec (gnat_node) : Defining_Entity (gnat_node));
tree gnu_subprog_decl;
tree gnu_result_decl;
tree gnu_subprog_type;
tree gnu_cico_list;
tree gnu_return_var_elmt = NULL_TREE;
tree gnu_result;
location_t locus;
struct language_function *gnu_subprog_language;
vec<parm_attr, va_gc> *cache;
if (Ekind (gnat_subprog_id) == E_Generic_Procedure
|| Ekind (gnat_subprog_id) == E_Generic_Function
|| Is_Eliminated (gnat_subprog_id))
return;
gnu_subprog_decl
= gnat_to_gnu_entity (gnat_subprog_id, NULL_TREE,
Acts_As_Spec (gnat_node)
&& !present_gnu_tree (gnat_subprog_id));
DECL_FUNCTION_IS_DEF (gnu_subprog_decl) = true;
gnu_result_decl = DECL_RESULT (gnu_subprog_decl);
gnu_subprog_type = TREE_TYPE (gnu_subprog_decl);
gnu_cico_list = TYPE_CI_CO_LIST (gnu_subprog_type);
if (gnu_cico_list && TREE_VALUE (gnu_cico_list) == void_type_node)
gnu_return_var_elmt = gnu_cico_list;
if (TREE_ADDRESSABLE (gnu_subprog_type))
{
TREE_TYPE (gnu_result_decl)
= build_reference_type (TREE_TYPE (gnu_result_decl));
relayout_decl (gnu_result_decl);
}
if (!Sloc_to_locus (Sloc (gnat_node), &locus))
locus = input_location;
DECL_SOURCE_LOCATION (gnu_subprog_decl) = locus;
if (Was_Expression_Function (gnat_node))
DECL_DISREGARD_INLINE_LIMITS (gnu_subprog_decl) = 1;
allocate_struct_function (gnu_subprog_decl, false);
gnu_subprog_language = ggc_cleared_alloc<language_function> ();
DECL_STRUCT_FUNCTION (gnu_subprog_decl)->language = gnu_subprog_language;
DECL_STRUCT_FUNCTION (gnu_subprog_decl)->function_start_locus = locus;
set_cfun (NULL);
begin_subprog_body (gnu_subprog_decl);
if (gnu_cico_list)
{
tree gnu_return_var = NULL_TREE;
vec_safe_push (gnu_return_label_stack,
create_artificial_label (input_location));
start_stmt_group ();
gnat_pushlevel ();
if (gnu_return_var_elmt && !TREE_ADDRESSABLE (gnu_subprog_type))
{
tree gnu_return_type
= TREE_TYPE (TREE_PURPOSE (gnu_return_var_elmt));
gnu_return_var
= create_var_decl (get_identifier ("RETVAL"), NULL_TREE,
gnu_return_type, NULL_TREE,
false, false, false, false, false,
true, false, NULL, gnat_subprog_id);
TREE_VALUE (gnu_return_var_elmt) = gnu_return_var;
}
vec_safe_push (gnu_return_var_stack, gnu_return_var);
for (gnat_param = First_Formal_With_Extras (gnat_subprog_id);
Present (gnat_param);
gnat_param = Next_Formal_With_Extras (gnat_param))
if (!present_gnu_tree (gnat_param))
{
tree gnu_cico_entry = gnu_cico_list;
tree gnu_decl;
while (gnu_cico_entry && TREE_VALUE (gnu_cico_entry))
gnu_cico_entry = TREE_CHAIN (gnu_cico_entry);
gnu_decl = gnat_to_gnu_entity (gnat_param, NULL_TREE, true);
gcc_assert (DECL_P (gnu_decl));
if (DECL_BY_REF_P (gnu_decl))
gnu_decl = build_unary_op (INDIRECT_REF, NULL_TREE, gnu_decl);
TREE_VALUE (gnu_cico_entry)
= convert (TREE_TYPE (TREE_PURPOSE (gnu_cico_entry)), gnu_decl);
}
}
else
vec_safe_push (gnu_return_label_stack, NULL_TREE);
start_stmt_group ();
gnat_pushlevel ();
process_decls (Declarations (gnat_node), Empty, Empty, true, true);
add_stmt (gnat_to_gnu (Handled_Statement_Sequence (gnat_node)));
gnat_poplevel ();
gnu_result = end_stmt_group ();
cache = gnu_subprog_language->parm_attr_cache;
if (cache)
{
struct parm_attr_d *pa;
int i;
start_stmt_group ();
FOR_EACH_VEC_ELT (*cache, i, pa)
{
if (pa->first)
add_stmt_with_node_force (pa->first, gnat_node);
if (pa->last)
add_stmt_with_node_force (pa->last, gnat_node);
if (pa->length)
add_stmt_with_node_force (pa->length, gnat_node);
}
add_stmt (gnu_result);
gnu_result = end_stmt_group ();
gnu_subprog_language->parm_attr_cache = NULL;
}
if (gnu_cico_list)
{
const Node_Id gnat_end_label
= End_Label (Handled_Statement_Sequence (gnat_node));
gnu_return_var_stack->pop ();
add_stmt (gnu_result);
add_stmt (build1 (LABEL_EXPR, void_type_node,
gnu_return_label_stack->last ()));
if (TREE_ADDRESSABLE (gnu_subprog_type))
{
tree gnu_ret_deref
= build_unary_op (INDIRECT_REF, NULL_TREE, gnu_result_decl);
tree t;
gcc_assert (TREE_VALUE (gnu_cico_list) == void_type_node);
for (t = TREE_CHAIN (gnu_cico_list); t; t = TREE_CHAIN (t))
{
tree gnu_field_deref
= build_component_ref (gnu_ret_deref, TREE_PURPOSE (t), true);
gnu_result = build2 (MODIFY_EXPR, void_type_node,
gnu_field_deref, TREE_VALUE (t));
add_stmt_with_node (gnu_result, gnat_end_label);
}
}
else
{
tree gnu_retval;
if (list_length (gnu_cico_list) == 1)
gnu_retval = TREE_VALUE (gnu_cico_list);
else
gnu_retval
= build_constructor_from_list (TREE_TYPE (gnu_subprog_type),
gnu_cico_list);
gnu_result = build_return_expr (gnu_result_decl, gnu_retval);
add_stmt_with_node (gnu_result, gnat_end_label);
}
gnat_poplevel ();
gnu_result = end_stmt_group ();
}
gnu_return_label_stack->pop ();
set_end_locus_from_node (gnu_result, gnat_node);
set_end_locus_from_node (gnu_subprog_decl, gnat_node);
if (DECL_NAME (gnu_subprog_decl) == main_identifier_node
&& targetm_common.except_unwind_info (&global_options) == UI_SEH)
{
tree t;
tree etype;
t = build_call_expr (builtin_decl_explicit (BUILT_IN_EH_POINTER),
1, integer_zero_node);
t = build_call_n_expr (unhandled_except_decl, 1, t);
etype = build_unary_op (ADDR_EXPR, NULL_TREE, unhandled_others_decl);
etype = tree_cons (NULL_TREE, etype, NULL_TREE);
t = build2 (CATCH_EXPR, void_type_node, etype, t);
gnu_result = build2 (TRY_CATCH_EXPR, TREE_TYPE (gnu_result),
gnu_result, t);
}
end_subprog_body (gnu_result);
for (gnat_param = First_Formal_With_Extras (gnat_subprog_id);
Present (gnat_param);
gnat_param = Next_Formal_With_Extras (gnat_param))
{
tree gnu_param = get_gnu_tree (gnat_param);
bool is_var_decl = (TREE_CODE (gnu_param) == VAR_DECL);
annotate_object (gnat_param, TREE_TYPE (gnu_param), NULL_TREE,
DECL_BY_REF_P (gnu_param));
if (is_var_decl)
save_gnu_tree (gnat_param, NULL_TREE, false);
}
if (gnu_return_var_elmt)
TREE_VALUE (gnu_return_var_elmt) = void_type_node;
if (optimize && gnu_subprog_language->named_ret_val)
{
finalize_nrv (gnu_subprog_decl,
gnu_subprog_language->named_ret_val,
gnu_subprog_language->other_ret_val,
gnu_subprog_language->gnat_ret);
gnu_subprog_language->named_ret_val = NULL;
gnu_subprog_language->other_ret_val = NULL;
}
if (DECL_EXTERNAL (gnu_subprog_decl) && DECL_UNINLINABLE (gnu_subprog_decl))
DECL_SAVED_TREE (gnu_subprog_decl) = NULL_TREE;
else
rest_of_subprog_body_compilation (gnu_subprog_decl);
}

static bool
node_is_atomic (Node_Id gnat_node)
{
Entity_Id gnat_entity;
switch (Nkind (gnat_node))
{
case N_Identifier:
case N_Expanded_Name:
gnat_entity = Entity (gnat_node);
if (Ekind (gnat_entity) != E_Variable)
break;
return Is_Atomic (gnat_entity) || Is_Atomic (Etype (gnat_entity));
case N_Selected_Component:
gnat_entity = Entity (Selector_Name (gnat_node));
return Is_Atomic (gnat_entity) || Is_Atomic (Etype (gnat_entity));
case N_Indexed_Component:
if (Has_Atomic_Components (Etype (Prefix (gnat_node))))
return true;
if (Is_Entity_Name (Prefix (gnat_node))
&& Has_Atomic_Components (Entity (Prefix (gnat_node))))
return true;
case N_Explicit_Dereference:
return Is_Atomic (Etype (gnat_node));
default:
break;
}
return false;
}
static bool
node_has_volatile_full_access (Node_Id gnat_node)
{
Entity_Id gnat_entity;
switch (Nkind (gnat_node))
{
case N_Identifier:
case N_Expanded_Name:
gnat_entity = Entity (gnat_node);
if (!Is_Object (gnat_entity))
break;
return Is_Volatile_Full_Access (gnat_entity)
|| Is_Volatile_Full_Access (Etype (gnat_entity));
case N_Selected_Component:
gnat_entity = Entity (Selector_Name (gnat_node));
return Is_Volatile_Full_Access (gnat_entity)
|| Is_Volatile_Full_Access (Etype (gnat_entity));
case N_Indexed_Component:
case N_Explicit_Dereference:
return Is_Volatile_Full_Access (Etype (gnat_node));
default:
break;
}
return false;
}
static Node_Id
gnat_strip_type_conversion (Node_Id gnat_node)
{
Node_Kind kind = Nkind (gnat_node);
if (kind == N_Type_Conversion || kind == N_Unchecked_Type_Conversion)
gnat_node = Expression (gnat_node);
return gnat_node;
}
static bool
outer_atomic_access_required_p (Node_Id gnat_node)
{
gnat_node = gnat_strip_type_conversion (gnat_node);
while (true)
{
switch (Nkind (gnat_node))
{
case N_Identifier:
case N_Expanded_Name:
if (No (Renamed_Object (Entity (gnat_node))))
return false;
gnat_node
= gnat_strip_type_conversion (Renamed_Object (Entity (gnat_node)));
break;
case N_Indexed_Component:
case N_Selected_Component:
case N_Slice:
gnat_node = gnat_strip_type_conversion (Prefix (gnat_node));
if (node_has_volatile_full_access (gnat_node))
return true;
break;
default:
return false;
}
}
gcc_unreachable ();
}
static bool
atomic_access_required_p (Node_Id gnat_node, bool *sync)
{
const Node_Id gnat_parent = Parent (gnat_node);
unsigned char attr_id;
bool as_a_whole = true;
switch (Nkind (gnat_parent))
{
case N_Attribute_Reference:
attr_id = Get_Attribute_Id (Attribute_Name (gnat_parent));
if (attr_id == Attr_Asm_Input || attr_id == Attr_Asm_Output)
return false;
case N_Reference:
if (Prefix (gnat_parent) == gnat_node)
return false;
break;
case N_Indexed_Component:
case N_Selected_Component:
case N_Slice:
if (Prefix (gnat_parent) == gnat_node)
as_a_whole = false;
break;
case N_Object_Renaming_Declaration:
return false;
default:
break;
}
gnat_node = gnat_strip_type_conversion (gnat_node);
if (!(as_a_whole && node_is_atomic (gnat_node))
&& !node_has_volatile_full_access (gnat_node))
return false;
if (outer_atomic_access_required_p (gnat_node))
return false;
*sync = Atomic_Sync_Required (gnat_node);
return true;
}

static tree
create_temporary (const char *prefix, tree type)
{
tree gnu_temp
= create_var_decl (create_tmp_var_name (prefix), NULL_TREE,
type, NULL_TREE,
false, false, false, false, false,
true, false, NULL, Empty);
return gnu_temp;
}
static tree
create_init_temporary (const char *prefix, tree gnu_init, tree *gnu_init_stmt,
Node_Id gnat_node)
{
tree gnu_temp = create_temporary (prefix, TREE_TYPE (gnu_init));
*gnu_init_stmt = build_binary_op (INIT_EXPR, NULL_TREE, gnu_temp, gnu_init);
set_expr_location_from_node (*gnu_init_stmt, gnat_node);
return gnu_temp;
}
static tree
Call_to_gnu (Node_Id gnat_node, tree *gnu_result_type_p, tree gnu_target,
bool outer_atomic_access, bool atomic_access, bool atomic_sync)
{
const bool function_call = (Nkind (gnat_node) == N_Function_Call);
const bool returning_value = (function_call && !gnu_target);
tree gnu_subprog = gnat_to_gnu (Name (gnat_node));
tree gnu_subprog_type = TREE_TYPE (gnu_subprog);
tree gnu_result_type = TREE_TYPE (gnu_subprog_type);
tree gnu_subprog_addr = build_unary_op (ADDR_EXPR, NULL_TREE, gnu_subprog);
vec<tree, va_gc> *gnu_actual_vec = NULL;
tree gnu_name_list = NULL_TREE;
tree gnu_stmt_list = NULL_TREE;
tree gnu_after_list = NULL_TREE;
tree gnu_retval = NULL_TREE;
tree gnu_call, gnu_result;
bool by_descriptor = false;
bool went_into_elab_proc = false;
bool pushed_binding_level = false;
Entity_Id gnat_formal;
Node_Id gnat_actual;
bool sync;
gcc_assert (TREE_CODE (gnu_subprog_type) == FUNCTION_TYPE);
if (TREE_CODE (gnu_subprog) == FUNCTION_DECL && DECL_STUBBED_P (gnu_subprog))
{
tree call_expr = build_call_raise (PE_Stubbed_Subprogram_Called,
gnat_node, N_Raise_Program_Error);
for (gnat_actual = First_Actual (gnat_node);
Present (gnat_actual);
gnat_actual = Next_Actual (gnat_actual))
add_stmt (gnat_to_gnu (gnat_actual));
if (returning_value)
{
*gnu_result_type_p = gnu_result_type;
return build1 (NULL_EXPR, gnu_result_type, call_expr);
}
return call_expr;
}
if (TREE_CODE (gnu_subprog) == FUNCTION_DECL)
{
if (decl_function_context (gnu_subprog))
check_inlining_for_nested_subprog (gnu_subprog);
if (gnu_subprog == current_function_decl)
DECL_DISREGARD_INLINE_LIMITS (gnu_subprog) = 0;
}
if (Nkind (Name (gnat_node)) == N_Explicit_Dereference)
{
gnat_formal = First_Formal_With_Extras (Etype (Name (gnat_node)));
if (targetm.calls.custom_function_descriptors > 0
&& Can_Use_Internal_Rep (Etype (Prefix (Name (gnat_node)))))
by_descriptor = true;
}
else if (Nkind (Name (gnat_node)) == N_Attribute_Reference)
gnat_formal = Empty;
else
gnat_formal = First_Formal_With_Extras (Entity (Name (gnat_node)));
if (!current_function_decl)
{
current_function_decl = get_elaboration_procedure ();
went_into_elab_proc = true;
}
if (function_call
&& ((!gnu_target && TYPE_CI_CO_LIST (gnu_subprog_type))
|| (!gnu_target
&& Nkind (Parent (gnat_node)) != N_Object_Declaration
&& Nkind (Parent (gnat_node)) != N_Object_Renaming_Declaration
&& Nkind (Parent (gnat_node)) != N_Simple_Return_Statement
&& (!(Nkind (Parent (gnat_node)) == N_Qualified_Expression
&& Nkind (Parent (Parent (gnat_node))) == N_Allocator)
|| type_is_padding_self_referential (gnu_result_type))
&& AGGREGATE_TYPE_P (gnu_result_type)
&& !TYPE_IS_FAT_POINTER_P (gnu_result_type))
|| (gnu_target
&& (TREE_CODE (gnu_target) == ARRAY_RANGE_REF
|| (TREE_CODE (TREE_TYPE (gnu_target)) == ARRAY_TYPE
&& TREE_CODE (TYPE_SIZE (TREE_TYPE (gnu_target)))
== INTEGER_CST))
&& TREE_CODE (TYPE_SIZE (gnu_result_type)) != INTEGER_CST)))
{
gnu_retval = create_temporary ("R", gnu_result_type);
DECL_RETURN_VALUE_P (gnu_retval) = 1;
}
if (!returning_value || gnu_retval)
{
start_stmt_group ();
gnat_pushlevel ();
pushed_binding_level = true;
}
for (gnat_actual = First_Actual (gnat_node);
Present (gnat_actual);
gnat_formal = Next_Formal_With_Extras (gnat_formal),
gnat_actual = Next_Actual (gnat_actual))
{
Entity_Id gnat_formal_type = Etype (gnat_formal);
tree gnu_formal_type = gnat_to_gnu_type (gnat_formal_type);
tree gnu_formal = present_gnu_tree (gnat_formal)
? get_gnu_tree (gnat_formal) : NULL_TREE;
const bool in_param = (Ekind (gnat_formal) == E_In_Parameter);
const bool is_true_formal_parm
= gnu_formal && TREE_CODE (gnu_formal) == PARM_DECL;
const bool is_by_ref_formal_parm
= is_true_formal_parm
&& (DECL_BY_REF_P (gnu_formal)
|| DECL_BY_COMPONENT_PTR_P (gnu_formal));
const bool suppress_type_conversion
= ((Nkind (gnat_actual) == N_Unchecked_Type_Conversion
&& (!in_param
|| (Is_Composite_Type (Underlying_Type (gnat_formal_type))
&& !Is_Constrained (Underlying_Type (gnat_formal_type)))))
|| (Nkind (gnat_actual) == N_Type_Conversion
&& Is_Composite_Type (Underlying_Type (gnat_formal_type))));
Node_Id gnat_name = suppress_type_conversion
? Expression (gnat_actual) : gnat_actual;
tree gnu_name = gnat_to_gnu (gnat_name), gnu_name_type;
if (!in_param && !is_by_ref_formal_parm)
{
tree init = NULL_TREE;
gnu_name = gnat_stabilize_reference (gnu_name, true, &init);
if (init)
gnu_name
= build_compound_expr (TREE_TYPE (gnu_name), init, gnu_name);
}
if (is_by_ref_formal_parm
&& (gnu_name_type = gnat_to_gnu_type (Etype (gnat_name)))
&& !addressable_p (gnu_name, gnu_name_type))
{
tree gnu_orig = gnu_name, gnu_temp, gnu_stmt;
if (TREE_CODE (gnu_name) == CONSTRUCTOR)
;
else if (TYPE_IS_BY_REFERENCE_P (gnu_formal_type))
post_error ("misaligned actual cannot be passed by reference",
gnat_actual);
else if (Is_Valued_Procedure (Entity (Name (gnat_node))))
{
post_error
("?possible violation of implicit assumption", gnat_actual);
post_error_ne
("?made by pragma Import_Valued_Procedure on &", gnat_actual,
Entity (Name (gnat_node)));
post_error_ne ("?because of misalignment of &", gnat_actual,
gnat_formal);
}
if (TREE_TYPE (gnu_name) == gnu_name_type
&& !CONTAINS_PLACEHOLDER_P (TYPE_SIZE (gnu_name_type)))
;
else if (TREE_CODE (gnu_name) == COMPONENT_REF
&& TYPE_IS_PADDING_P
(TREE_TYPE (TREE_OPERAND (gnu_name, 0))))
gnu_orig = gnu_name = TREE_OPERAND (gnu_name, 0);
else if ((TREE_CODE (gnu_name_type) == RECORD_TYPE
&& (TYPE_JUSTIFIED_MODULAR_P (gnu_name_type)
|| smaller_form_type_p (TREE_TYPE (gnu_name),
gnu_name_type)))
|| (INTEGRAL_TYPE_P (gnu_name_type)
&& smaller_form_type_p (gnu_name_type,
TREE_TYPE (gnu_name))))
gnu_name = convert (gnu_name_type, gnu_name);
if (!in_param && returning_value && !gnu_retval)
{
gnu_retval = create_temporary ("R", gnu_result_type);
DECL_RETURN_VALUE_P (gnu_retval) = 1;
}
if (!pushed_binding_level && (!returning_value || gnu_retval))
{
start_stmt_group ();
gnat_pushlevel ();
pushed_binding_level = true;
}
gnu_temp
= create_init_temporary ("A", gnu_name, &gnu_stmt, gnat_actual);
gnu_name = build_compound_expr (TREE_TYPE (gnu_name), gnu_stmt,
gnu_temp);
if (!in_param)
{
if (TREE_CODE (gnu_orig) == COND_EXPR
&& TREE_CODE (TREE_OPERAND (gnu_orig, 1)) == COMPOUND_EXPR
&& integer_zerop
(TREE_OPERAND (TREE_OPERAND (gnu_orig, 1), 1)))
gnu_orig = TREE_OPERAND (gnu_orig, 2);
gnu_stmt
= build_binary_op (MODIFY_EXPR, NULL_TREE, gnu_orig, gnu_temp);
set_expr_location_from_node (gnu_stmt, gnat_node);
append_to_statement_list (gnu_stmt, &gnu_after_list);
}
}
tree gnu_actual = gnu_name;
if (is_true_formal_parm
&& !is_by_ref_formal_parm
&& Ekind (gnat_formal) != E_Out_Parameter
&& atomic_access_required_p (gnat_actual, &sync))
gnu_actual = build_atomic_load (gnu_actual, sync);
if (Ekind (gnat_formal) != E_Out_Parameter
&& TYPE_IS_PADDING_P (TREE_TYPE (gnu_actual)))
gnu_actual
= convert (get_unpadded_type (Etype (gnat_actual)), gnu_actual);
tree gnu_actual_type = gnat_to_gnu_type (Etype (gnat_actual));
if (TYPE_IS_DUMMY_P (gnu_actual_type))
gcc_assert (is_true_formal_parm && DECL_BY_REF_P (gnu_formal));
else if (suppress_type_conversion
&& Nkind (gnat_actual) == N_Unchecked_Type_Conversion)
gnu_actual = unchecked_convert (gnu_actual_type, gnu_actual,
No_Truncation (gnat_actual));
else
gnu_actual = convert (gnu_actual_type, gnu_actual);
if (Ekind (gnat_formal) != E_Out_Parameter
&& Do_Range_Check (gnat_actual))
gnu_actual
= emit_range_check (gnu_actual, gnat_formal_type, gnat_actual);
if (!in_param
&& TREE_CODE (gnu_name) == CONSTRUCTOR
&& TREE_CODE (TREE_TYPE (gnu_name)) == RECORD_TYPE
&& TYPE_JUSTIFIED_MODULAR_P (TREE_TYPE (gnu_name)))
gnu_name
= convert (TREE_TYPE (TYPE_FIELDS (TREE_TYPE (gnu_name))), gnu_name);
if (is_true_formal_parm && DECL_BY_REF_P (gnu_formal))
{
if (!in_param)
{
gnu_actual = gnu_name;
if (TYPE_IS_PADDING_P (TREE_TYPE (gnu_actual)))
gnu_actual = convert (get_unpadded_type (Etype (gnat_actual)),
gnu_actual);
if (TREE_CODE (TREE_TYPE (gnu_actual)) == RECORD_TYPE
&& TYPE_CONTAINS_TEMPLATE_P (TREE_TYPE (gnu_actual))
&& Is_Constr_Subt_For_UN_Aliased (Etype (gnat_actual))
&& Is_Array_Type (Underlying_Type (Etype (gnat_actual))))
gnu_actual = convert (gnu_actual_type, gnu_actual);
}
if (TREE_CODE (gnu_formal_type) == UNCONSTRAINED_ARRAY_TYPE)
{
if (!in_param && suppress_type_conversion)
gnu_actual = convert (gnu_actual_type, gnu_actual);
gnu_actual = convert (gnu_formal_type, gnu_actual);
}
gnu_formal_type = TREE_TYPE (gnu_formal);
gnu_actual = build_unary_op (ADDR_EXPR, gnu_formal_type, gnu_actual);
}
else if (is_true_formal_parm && DECL_BY_COMPONENT_PTR_P (gnu_formal))
{
gnu_actual = maybe_implicit_deref (gnu_actual);
gnu_actual = maybe_unconstrained_array (gnu_actual);
gnu_formal_type = TREE_TYPE (gnu_formal);
gnu_actual = build_unary_op (ADDR_EXPR, gnu_formal_type, gnu_actual);
}
else
{
tree gnu_size;
if (!in_param)
gnu_name_list = tree_cons (NULL_TREE, gnu_name, gnu_name_list);
if (!is_true_formal_parm)
{
if (TREE_SIDE_EFFECTS (gnu_name))
{
tree addr = build_unary_op (ADDR_EXPR, NULL_TREE, gnu_name);
append_to_statement_list (addr, &gnu_stmt_list);
}
continue;
}
gnu_actual = convert (gnu_formal_type, gnu_actual);
if (TREE_CODE (gnu_actual) == INDIRECT_REF
&& TREE_PRIVATE (gnu_actual)
&& (gnu_size = TYPE_SIZE (TREE_TYPE (gnu_actual)))
&& TREE_CODE (gnu_size) == INTEGER_CST
&& compare_tree_int (gnu_size, BITS_PER_WORD) <= 0)
{
tree type_for_size
= gnat_type_for_size (TREE_INT_CST_LOW (gnu_size), 1);
gnu_actual
= unchecked_convert (DECL_ARG_TYPE (gnu_formal),
build_int_cst (type_for_size, 0),
false);
}
else
gnu_actual = convert (DECL_ARG_TYPE (gnu_formal), gnu_actual);
}
vec_safe_push (gnu_actual_vec, gnu_actual);
}
gnu_call
= build_call_vec (gnu_result_type, gnu_subprog_addr, gnu_actual_vec);
CALL_EXPR_BY_DESCRIPTOR (gnu_call) = by_descriptor;
set_expr_location_from_node (gnu_call, gnat_node);
if (gnu_retval)
{
tree gnu_stmt
= build_binary_op (INIT_EXPR, NULL_TREE, gnu_retval, gnu_call);
set_expr_location_from_node (gnu_stmt, gnat_node);
append_to_statement_list (gnu_stmt, &gnu_stmt_list);
gnu_call = gnu_retval;
}
if (TYPE_CI_CO_LIST (gnu_subprog_type))
{
tree gnu_cico_list = TYPE_CI_CO_LIST (gnu_subprog_type);
const int length = list_length (gnu_cico_list);
if (length > 1)
{
if (!gnu_retval)
{
tree gnu_stmt;
gnu_call
= create_init_temporary ("P", gnu_call, &gnu_stmt, gnat_node);
append_to_statement_list (gnu_stmt, &gnu_stmt_list);
}
gnu_name_list = nreverse (gnu_name_list);
}
if (function_call)
gnu_cico_list = TREE_CHAIN (gnu_cico_list);
if (Nkind (Name (gnat_node)) == N_Explicit_Dereference)
gnat_formal = First_Formal_With_Extras (Etype (Name (gnat_node)));
else
gnat_formal = First_Formal_With_Extras (Entity (Name (gnat_node)));
for (gnat_actual = First_Actual (gnat_node);
Present (gnat_actual);
gnat_formal = Next_Formal_With_Extras (gnat_formal),
gnat_actual = Next_Actual (gnat_actual))
if (!(present_gnu_tree (gnat_formal)
&& TREE_CODE (get_gnu_tree (gnat_formal)) == PARM_DECL
&& (DECL_BY_REF_P (get_gnu_tree (gnat_formal))
|| DECL_BY_COMPONENT_PTR_P (get_gnu_tree (gnat_formal))))
&& Ekind (gnat_formal) != E_In_Parameter)
{
tree gnu_result
= length == 1
? gnu_call
: build_component_ref (gnu_call, TREE_PURPOSE (gnu_cico_list),
false);
tree gnu_actual
= maybe_unconstrained_array (TREE_VALUE (gnu_name_list));
if (TYPE_IS_PADDING_P (TREE_TYPE (gnu_result)))
gnu_result
= convert (TREE_TYPE (TYPE_FIELDS (TREE_TYPE (gnu_result))),
gnu_result);
if (Nkind (gnat_actual) == N_Type_Conversion)
{
gnu_result
= convert_with_check
(Etype (Expression (gnat_actual)), gnu_result,
Do_Overflow_Check (gnat_actual),
Do_Range_Check (Expression (gnat_actual)),
Float_Truncate (gnat_actual), gnat_actual);
if (!Is_Composite_Type (Underlying_Type (Etype (gnat_formal))))
gnu_actual = convert (TREE_TYPE (gnu_result), gnu_actual);
}
else if (Nkind (gnat_actual) == N_Unchecked_Type_Conversion)
gnu_result = unchecked_convert (TREE_TYPE (gnu_actual),
gnu_result,
No_Truncation (gnat_actual));
else
{
if (Do_Range_Check (gnat_actual))
gnu_result
= emit_range_check (gnu_result, Etype (gnat_actual),
gnat_actual);
if (!(!TREE_CONSTANT (TYPE_SIZE (TREE_TYPE (gnu_actual)))
&& TREE_CONSTANT (TYPE_SIZE (TREE_TYPE (gnu_result)))))
gnu_result = convert (TREE_TYPE (gnu_actual), gnu_result);
}
if (outer_atomic_access_required_p (gnat_actual))
gnu_result
= build_load_modify_store (gnu_actual, gnu_result, gnat_node);
else if (atomic_access_required_p (gnat_actual, &sync))
gnu_result = build_atomic_store (gnu_actual, gnu_result, sync);
else
gnu_result = build_binary_op (MODIFY_EXPR, NULL_TREE,
gnu_actual, gnu_result);
if (EXPR_P (gnu_result))
set_expr_location_from_node (gnu_result, gnat_node);
append_to_statement_list (gnu_result, &gnu_stmt_list);
gnu_cico_list = TREE_CHAIN (gnu_cico_list);
gnu_name_list = TREE_CHAIN (gnu_name_list);
}
}
if (function_call)
{
if (TYPE_CI_CO_LIST (gnu_subprog_type))
{
tree gnu_elmt = TYPE_CI_CO_LIST (gnu_subprog_type);
gnu_call
= build_component_ref (gnu_call, TREE_PURPOSE (gnu_elmt), false);
gnu_result_type = TREE_TYPE (gnu_call);
}
if (TYPE_RETURN_UNCONSTRAINED_P (gnu_subprog_type)
|| TYPE_RETURN_BY_DIRECT_REF_P (gnu_subprog_type))
gnu_call = build_unary_op (INDIRECT_REF, NULL_TREE, gnu_call);
if (gnu_target)
{
Node_Id gnat_parent = Parent (gnat_node);
enum tree_code op_code;
if (Do_Range_Check (gnat_node))
gnu_call
= emit_range_check (gnu_call, Etype (Name (gnat_parent)),
gnat_parent);
if (return_type_with_variable_size_p (gnu_result_type))
op_code = INIT_EXPR;
else
op_code = MODIFY_EXPR;
if (outer_atomic_access)
gnu_call
= build_load_modify_store (gnu_target, gnu_call, gnat_node);
else if (atomic_access)
gnu_call = build_atomic_store (gnu_target, gnu_call, atomic_sync);
else
gnu_call
= build_binary_op (op_code, NULL_TREE, gnu_target, gnu_call);
if (EXPR_P (gnu_call))
set_expr_location_from_node (gnu_call, gnat_parent);
append_to_statement_list (gnu_call, &gnu_stmt_list);
}
else
*gnu_result_type_p = get_unpadded_type (Etype (gnat_node));
}
else if (!TYPE_CI_CO_LIST (gnu_subprog_type))
append_to_statement_list (gnu_call, &gnu_stmt_list);
append_to_statement_list (gnu_after_list, &gnu_stmt_list);
if (went_into_elab_proc)
current_function_decl = NULL_TREE;
if (pushed_binding_level)
{
add_stmt (gnu_stmt_list);
gnat_poplevel ();
gnu_result = end_stmt_group ();
}
else if (gnu_stmt_list)
gnu_result = gnu_stmt_list;
else
return gnu_call;
if (returning_value)
{
tree first = expr_first (gnu_result), last = expr_last (gnu_result);
if (first == last)
gnu_result = first;
gnu_result
= build_compound_expr (TREE_TYPE (gnu_call), gnu_result, gnu_call);
}
return gnu_result;
}

static tree
Handled_Sequence_Of_Statements_to_gnu (Node_Id gnat_node)
{
tree gnu_jmpsave_decl = NULL_TREE;
tree gnu_jmpbuf_decl = NULL_TREE;
bool gcc_eh = (!type_annotate_only
&& Present (Exception_Handlers (gnat_node))
&& Back_End_Exceptions ());
bool fe_sjlj
= (!type_annotate_only && Present (Exception_Handlers (gnat_node))
&& Exception_Mechanism == Front_End_SJLJ);
bool at_end = !type_annotate_only && Present (At_End_Proc (gnat_node));
bool binding_for_block = (at_end || gcc_eh || fe_sjlj);
tree gnu_inner_block; 
tree gnu_result;
tree gnu_expr;
Node_Id gnat_temp;
if (binding_for_block)
{
start_stmt_group ();
gnat_pushlevel ();
}
if (fe_sjlj)
{
gnu_jmpsave_decl
= create_var_decl (get_identifier ("JMPBUF_SAVE"), NULL_TREE,
jmpbuf_ptr_type,
build_call_n_expr (get_jmpbuf_decl, 0),
false, false, false, false, false, true, false,
NULL, gnat_node);
TREE_NO_WARNING (gnu_jmpsave_decl) = 1;
gnu_jmpbuf_decl
= create_var_decl (get_identifier ("JMP_BUF"), NULL_TREE,
jmpbuf_type,
NULL_TREE,
false, false, false, false, false, true, false,
NULL, gnat_node);
set_block_jmpbuf_decl (gnu_jmpbuf_decl);
add_cleanup (build_call_n_expr (set_jmpbuf_decl, 1, gnu_jmpsave_decl),
Present (End_Label (gnat_node))
? End_Label (gnat_node) : gnat_node);
}
if (at_end)
{
tree proc_decl = gnat_to_gnu (At_End_Proc (gnat_node));
if (!optimize)
DECL_DECLARED_INLINE_P (proc_decl) = 0;
add_cleanup (build_call_n_expr (proc_decl, 0),
Present (End_Label (gnat_node))
? End_Label (gnat_node) : At_End_Proc (gnat_node));
}
start_stmt_group ();
if (fe_sjlj)
{
gnu_expr = build_call_n_expr (set_jmpbuf_decl, 1,
build_unary_op (ADDR_EXPR, NULL_TREE,
gnu_jmpbuf_decl));
set_expr_location_from_node (gnu_expr, gnat_node);
add_stmt (gnu_expr);
}
if (Present (First_Real_Statement (gnat_node)))
process_decls (Statements (gnat_node), Empty,
First_Real_Statement (gnat_node), true, true);
for (gnat_temp = (Present (First_Real_Statement (gnat_node))
? First_Real_Statement (gnat_node)
: First (Statements (gnat_node)));
Present (gnat_temp); gnat_temp = Next (gnat_temp))
add_stmt (gnat_to_gnu (gnat_temp));
gnu_inner_block = end_stmt_group ();
if (fe_sjlj)
{
tree *gnu_else_ptr = 0;
tree gnu_handler;
start_stmt_group ();
gnat_pushlevel ();
vec_safe_push (gnu_except_ptr_stack,
create_var_decl (get_identifier ("EXCEPT_PTR"), NULL_TREE,
build_pointer_type (except_type_node),
build_call_n_expr (get_excptr_decl, 0),
false, false, false, false, false,
true, false, NULL, gnat_node));
for (gnat_temp = First_Non_Pragma (Exception_Handlers (gnat_node));
Present (gnat_temp); gnat_temp = Next_Non_Pragma (gnat_temp))
{
gnu_expr = gnat_to_gnu (gnat_temp);
if (!gnu_else_ptr)
add_stmt (gnu_expr);
else
*gnu_else_ptr = gnu_expr;
gnu_else_ptr = &COND_EXPR_ELSE (gnu_expr);
}
gnu_expr = build_call_n_expr (raise_nodefer_decl, 1,
gnu_except_ptr_stack->last ());
set_expr_location_from_node
(gnu_expr,
Present (End_Label (gnat_node)) ? End_Label (gnat_node) : gnat_node);
if (gnu_else_ptr)
*gnu_else_ptr = gnu_expr;
else
add_stmt (gnu_expr);
gnu_except_ptr_stack->pop ();
gnat_poplevel ();
gnu_handler = end_stmt_group ();
start_stmt_group ();
add_stmt_with_node (build_call_n_expr (set_jmpbuf_decl, 1,
gnu_jmpsave_decl),
gnat_node);
add_stmt (gnu_handler);
gnu_handler = end_stmt_group ();
gnu_result = build3 (COND_EXPR, void_type_node,
(build_call_n_expr
(setjmp_decl, 1,
build_unary_op (ADDR_EXPR, NULL_TREE,
gnu_jmpbuf_decl))),
gnu_handler, gnu_inner_block);
}
else if (gcc_eh)
{
tree gnu_handlers;
location_t locus;
start_stmt_group ();
for (gnat_temp = First_Non_Pragma (Exception_Handlers (gnat_node));
Present (gnat_temp);
gnat_temp = Next_Non_Pragma (gnat_temp))
add_stmt (gnat_to_gnu (gnat_temp));
gnu_handlers = end_stmt_group ();
gnu_result = build2 (TRY_CATCH_EXPR, void_type_node,
gnu_inner_block, gnu_handlers);
if (Present (End_Label (gnat_node))
&& Sloc_to_locus (Sloc (End_Label (gnat_node)), &locus))
SET_EXPR_LOCATION (gnu_result, locus);
else
set_expr_location_from_node (gnu_result, gnat_node, true);
}
else
gnu_result = gnu_inner_block;
if (binding_for_block)
{
add_stmt (gnu_result);
gnat_poplevel ();
gnu_result = end_stmt_group ();
}
return gnu_result;
}

static tree
Exception_Handler_to_gnu_fe_sjlj (Node_Id gnat_node)
{
tree gnu_choice = boolean_false_node;
tree gnu_body = build_stmt_group (Statements (gnat_node), false);
Node_Id gnat_temp;
for (gnat_temp = First (Exception_Choices (gnat_node));
gnat_temp; gnat_temp = Next (gnat_temp))
{
tree this_choice;
if (Nkind (gnat_temp) == N_Others_Choice)
{
if (All_Others (gnat_temp))
this_choice = boolean_true_node;
else
this_choice
= build_binary_op
(EQ_EXPR, boolean_type_node,
convert
(integer_type_node,
build_component_ref
(build_unary_op
(INDIRECT_REF, NULL_TREE,
gnu_except_ptr_stack->last ()),
not_handled_by_others_decl,
false)),
integer_zero_node);
}
else if (Nkind (gnat_temp) == N_Identifier
|| Nkind (gnat_temp) == N_Expanded_Name)
{
Entity_Id gnat_ex_id = Entity (gnat_temp);
tree gnu_expr;
if (Present (Renamed_Object (gnat_ex_id)))
gnat_ex_id = Renamed_Object (gnat_ex_id);
gnu_expr = gnat_to_gnu_entity (gnat_ex_id, NULL_TREE, false);
this_choice
= build_binary_op
(EQ_EXPR, boolean_type_node,
gnu_except_ptr_stack->last (),
convert (TREE_TYPE (gnu_except_ptr_stack->last ()),
build_unary_op (ADDR_EXPR, NULL_TREE, gnu_expr)));
}
else
gcc_unreachable ();
gnu_choice = build_binary_op (TRUTH_ORIF_EXPR, boolean_type_node,
gnu_choice, this_choice);
}
return build3 (COND_EXPR, void_type_node, gnu_choice, gnu_body, NULL_TREE);
}

static bool
stmt_list_cannot_alter_control_flow_p (List_Id gnat_list)
{
if (No (gnat_list))
return true;
for (Node_Id gnat_node = First (gnat_list);
Present (gnat_node);
gnat_node = Next (gnat_node))
{
if (Nkind (gnat_node) != N_Assignment_Statement)
return false;
if (Nkind (Name (gnat_node)) != N_Identifier)
return false;
Node_Kind nkind = Nkind (Expression (gnat_node));
if (nkind != N_Identifier
&& nkind != N_Integer_Literal
&& nkind != N_Real_Literal)
return false;
}
return true;
}
static tree
Exception_Handler_to_gnu_gcc (Node_Id gnat_node)
{
tree gnu_etypes_list = NULL_TREE;
for (Node_Id gnat_temp = First (Exception_Choices (gnat_node));
gnat_temp;
gnat_temp = Next (gnat_temp))
{
tree gnu_expr, gnu_etype;
if (Nkind (gnat_temp) == N_Others_Choice)
{
gnu_expr = All_Others (gnat_temp) ? all_others_decl : others_decl;
gnu_etype = build_unary_op (ADDR_EXPR, NULL_TREE, gnu_expr);
}
else if (Nkind (gnat_temp) == N_Identifier
|| Nkind (gnat_temp) == N_Expanded_Name)
{
Entity_Id gnat_ex_id = Entity (gnat_temp);
if (Present (Renamed_Object (gnat_ex_id)))
gnat_ex_id = Renamed_Object (gnat_ex_id);
gnu_expr = gnat_to_gnu_entity (gnat_ex_id, NULL_TREE, false);
gnu_etype = build_unary_op (ADDR_EXPR, NULL_TREE, gnu_expr);
}
else
gcc_unreachable ();
gnu_etypes_list = tree_cons (NULL_TREE, gnu_etype, gnu_etypes_list);
}
start_stmt_group ();
gnat_pushlevel ();
tree gnu_current_exc_ptr
= build_call_expr (builtin_decl_explicit (BUILT_IN_EH_POINTER),
1, integer_zero_node);
tree prev_gnu_incoming_exc_ptr = gnu_incoming_exc_ptr;
gnu_incoming_exc_ptr
= create_var_decl (get_identifier ("EXPTR"), NULL_TREE,
ptr_type_node, gnu_current_exc_ptr,
false, false, false, false, false, true, true,
NULL, gnat_node);
add_stmt_with_node (build_call_n_expr (begin_handler_decl, 1,
gnu_incoming_exc_ptr),
gnat_node);
if (Present (Choice_Parameter (gnat_node)))
{
tree gnu_param
= gnat_to_gnu_entity (Choice_Parameter (gnat_node), NULL_TREE, true);
add_stmt (build_call_n_expr
(set_exception_parameter_decl, 2,
build_unary_op (ADDR_EXPR, NULL_TREE, gnu_param),
gnu_incoming_exc_ptr));
}
add_stmt_list (Statements (gnat_node));
tree stmt = build_call_n_expr (end_handler_decl, 1, gnu_incoming_exc_ptr);
if (stmt_list_cannot_alter_control_flow_p (Statements (gnat_node)))
add_stmt_with_node (stmt, gnat_node);
else
add_cleanup (stmt, gnat_node);
gnat_poplevel ();
gnu_incoming_exc_ptr = prev_gnu_incoming_exc_ptr;
return
build2 (CATCH_EXPR, void_type_node, gnu_etypes_list, end_stmt_group ());
}

static void
Compilation_Unit_to_gnu (Node_Id gnat_node)
{
const Node_Id gnat_unit = Unit (gnat_node);
const bool body_p = (Nkind (gnat_unit) == N_Package_Body
|| Nkind (gnat_unit) == N_Subprogram_Body);
const Entity_Id gnat_unit_entity = Defining_Entity (gnat_unit);
Entity_Id gnat_entity;
Node_Id gnat_pragma;
tree gnu_elab_proc_decl
= create_subprog_decl
(create_concat_name (gnat_unit_entity, body_p ? "elabb" : "elabs"),
NULL_TREE, void_ftype, NULL_TREE,
is_disabled, true, false, true, true, false, NULL, gnat_unit);
struct elab_info *info;
vec_safe_push (gnu_elab_proc_stack, gnu_elab_proc_decl);
DECL_ELABORATION_PROC_P (gnu_elab_proc_decl) = 1;
allocate_struct_function (gnu_elab_proc_decl, false);
set_cfun (NULL);
current_function_decl = NULL_TREE;
start_stmt_group ();
gnat_pushlevel ();
if (Nkind (gnat_unit) == N_Package_Body
|| (Nkind (gnat_unit) == N_Subprogram_Body && !Acts_As_Spec (gnat_node)))
add_stmt (gnat_to_gnu (Library_Unit (gnat_node)));
if (type_annotate_only && gnat_node == Cunit (Main_Unit))
{
elaborate_all_entities (gnat_node);
if (Nkind (gnat_unit) == N_Subprogram_Declaration
|| Nkind (gnat_unit) == N_Generic_Package_Declaration
|| Nkind (gnat_unit) == N_Generic_Subprogram_Declaration)
return;
}
for (gnat_pragma = First (Context_Items (gnat_node));
Present (gnat_pragma);
gnat_pragma = Next (gnat_pragma))
if (Nkind (gnat_pragma) == N_Pragma)
add_stmt (gnat_to_gnu (gnat_pragma));
process_decls (Declarations (Aux_Decls_Node (gnat_node)), Empty, Empty,
true, true);
add_stmt (gnat_to_gnu (gnat_unit));
for (gnat_entity = First_Inlined_Subprogram (gnat_node);
Present (gnat_entity);
gnat_entity = Next_Inlined_Subprogram (gnat_entity))
{
Node_Id gnat_body;
if (!optimize && !Has_Pragma_Inline_Always (gnat_entity))
continue;
if (!Is_Public (gnat_entity))
continue;
gnat_body = Parent (Declaration_Node (gnat_entity));
if (Nkind (gnat_body) != N_Subprogram_Body)
{
if (No (Corresponding_Body (gnat_body)))
continue;
gnat_body
= Parent (Declaration_Node (Corresponding_Body (gnat_body)));
}
gnat_to_gnu_entity (gnat_entity, NULL_TREE, false);
add_stmt (gnat_to_gnu (gnat_body));
}
add_stmt_list (Pragmas_After (Aux_Decls_Node (gnat_node)));
add_stmt_list (Actions (Aux_Decls_Node (gnat_node)));
finalize_from_limited_with ();
set_current_block_context (gnu_elab_proc_decl);
gnat_poplevel ();
DECL_SAVED_TREE (gnu_elab_proc_decl) = end_stmt_group ();
set_end_locus_from_node (gnu_elab_proc_decl, gnat_unit);
gnu_elab_proc_stack->pop ();
info = ggc_alloc<elab_info> ();
info->next = elab_info_list;
info->elab_proc = gnu_elab_proc_decl;
info->gnat_node = gnat_node;
elab_info_list = info;
process_deferred_decl_context (true);
}

static tree
build_noreturn_cond (tree cond)
{
tree fn = builtin_decl_explicit (BUILT_IN_EXPECT);
tree arg_types = TYPE_ARG_TYPES (TREE_TYPE (fn));
tree pred_type = TREE_VALUE (arg_types);
tree expected_type = TREE_VALUE (TREE_CHAIN (arg_types));
tree t = build_call_expr (fn, 3,
fold_convert (pred_type, cond),
build_int_cst (expected_type, 0),
build_int_cst (integer_type_node,
PRED_NORETURN));
return build1 (NOP_EXPR, boolean_type_node, t);
}
static void
Range_to_gnu (Node_Id gnat_range, tree *gnu_low, tree *gnu_high)
{
switch (Nkind (gnat_range))
{
case N_Range:
*gnu_low = gnat_to_gnu (Low_Bound (gnat_range));
*gnu_high = gnat_to_gnu (High_Bound (gnat_range));
break;
case N_Expanded_Name:
case N_Identifier:
{
tree gnu_range_type = get_unpadded_type (Entity (gnat_range));
tree gnu_range_base_type = get_base_type (gnu_range_type);
*gnu_low
= convert (gnu_range_base_type, TYPE_MIN_VALUE (gnu_range_type));
*gnu_high
= convert (gnu_range_base_type, TYPE_MAX_VALUE (gnu_range_type));
}
break;
default:
gcc_unreachable ();
}
}
static tree
Raise_Error_to_gnu (Node_Id gnat_node, tree *gnu_result_type_p)
{
const Node_Kind kind = Nkind (gnat_node);
const int reason = UI_To_Int (Reason (gnat_node));
const Node_Id gnat_cond = Condition (gnat_node);
const bool with_extra_info
= Exception_Extra_Info
&& !No_Exception_Handlers_Set ()
&& No (get_exception_label (kind));
tree gnu_result = NULL_TREE, gnu_cond = NULL_TREE;
switch (reason)
{
case CE_Access_Check_Failed:
if (with_extra_info)
gnu_result = build_call_raise_column (reason, gnat_node, kind);
break;
case CE_Index_Check_Failed:
case CE_Range_Check_Failed:
case CE_Invalid_Data:
if (Present (gnat_cond) && Nkind (gnat_cond) == N_Op_Not)
{
Node_Id gnat_index, gnat_type;
tree gnu_type, gnu_index, gnu_low_bound, gnu_high_bound, disp;
bool neg_p;
struct loop_info_d *loop;
switch (Nkind (Right_Opnd (gnat_cond)))
{
case N_In:
Range_to_gnu (Right_Opnd (Right_Opnd (gnat_cond)),
&gnu_low_bound, &gnu_high_bound);
break;
case N_Op_Ge:
gnu_low_bound = gnat_to_gnu (Right_Opnd (Right_Opnd (gnat_cond)));
gnu_high_bound = NULL_TREE;
break;
case N_Op_Le:
gnu_low_bound = NULL_TREE;
gnu_high_bound = gnat_to_gnu (Right_Opnd (Right_Opnd (gnat_cond)));
break;
default:
goto common;
}
gnat_index = Left_Opnd (Right_Opnd (gnat_cond));
gnat_type = Etype (gnat_index);
gnu_type = maybe_character_type (get_unpadded_type (gnat_type));
gnu_index = gnat_to_gnu (gnat_index);
if (TREE_TYPE (gnu_index) != gnu_type)
{
if (gnu_low_bound)
gnu_low_bound = convert (gnu_type, gnu_low_bound);
if (gnu_high_bound)
gnu_high_bound = convert (gnu_type, gnu_high_bound);
gnu_index = convert (gnu_type, gnu_index);
}
if (with_extra_info
&& gnu_low_bound
&& gnu_high_bound
&& Known_Esize (gnat_type)
&& UI_To_Int (Esize (gnat_type)) <= 32)
gnu_result
= build_call_raise_range (reason, gnat_node, kind, gnu_index,
gnu_low_bound, gnu_high_bound);
if (optimize
&& inside_loop_p ()
&& (!gnu_low_bound
|| (gnu_low_bound = gnat_invariant_expr (gnu_low_bound)))
&& (!gnu_high_bound
|| (gnu_high_bound = gnat_invariant_expr (gnu_high_bound)))
&& (loop = find_loop_for (gnu_index, &disp, &neg_p)))
{
struct range_check_info_d *rci = ggc_alloc<range_check_info_d> ();
rci->low_bound = gnu_low_bound;
rci->high_bound = gnu_high_bound;
rci->disp = disp;
rci->neg_p = neg_p;
rci->type = gnu_type;
rci->inserted_cond
= build1 (SAVE_EXPR, boolean_type_node, boolean_true_node);
vec_safe_push (loop->checks, rci);
gnu_cond = build_noreturn_cond (gnat_to_gnu (gnat_cond));
if (optimize >= 3)
gnu_cond = build_binary_op (TRUTH_ANDIF_EXPR,
boolean_type_node,
rci->inserted_cond,
gnu_cond);
else
gnu_cond = build_binary_op (TRUTH_ANDIF_EXPR,
boolean_type_node,
gnu_cond,
rci->inserted_cond);
}
}
break;
default:
break;
}
common:
if (!gnu_result)
gnu_result = build_call_raise (reason, gnat_node, kind);
set_expr_location_from_node (gnu_result, gnat_node);
*gnu_result_type_p = get_unpadded_type (Etype (gnat_node));
if (VOID_TYPE_P (*gnu_result_type_p))
{
if (Present (gnat_cond))
{
if (!gnu_cond)
gnu_cond = gnat_to_gnu (gnat_cond);
gnu_result = build3 (COND_EXPR, void_type_node, gnu_cond, gnu_result,
alloc_stmt_list ());
}
}
else
gnu_result = build1 (NULL_EXPR, *gnu_result_type_p, gnu_result);
return gnu_result;
}

static bool
lhs_or_actual_p (Node_Id gnat_node)
{
Node_Id gnat_parent = Parent (gnat_node);
Node_Kind kind = Nkind (gnat_parent);
if (kind == N_Assignment_Statement && Name (gnat_parent) == gnat_node)
return true;
if ((kind == N_Procedure_Call_Statement || kind == N_Function_Call)
&& Name (gnat_parent) != gnat_node)
return true;
if (kind == N_Parameter_Association)
return true;
return false;
}
static bool
present_in_lhs_or_actual_p (Node_Id gnat_node)
{
Node_Kind kind;
if (lhs_or_actual_p (gnat_node))
return true;
kind = Nkind (Parent (gnat_node));
if ((kind == N_Type_Conversion || kind == N_Unchecked_Type_Conversion)
&& lhs_or_actual_p (Parent (gnat_node)))
return true;
return false;
}
static bool
unchecked_conversion_nop (Node_Id gnat_node)
{
Entity_Id from_type, to_type;
if (!lhs_or_actual_p (gnat_node))
return false;
from_type = Etype (Expression (gnat_node));
if (!Is_Private_Type (from_type))
return false;
from_type = Underlying_Type (from_type);
to_type = Etype (gnat_node);
if (to_type == from_type)
return true;
if (Ekind (from_type) == E_Array_Subtype
&& to_type == Packed_Array_Impl_Type (from_type))
return true;
if (Ekind (from_type) == E_Record_Subtype
&& to_type == Etype (from_type))
return true;
return false;
}
static bool
statement_node_p (Node_Id gnat_node)
{
const Node_Kind kind = Nkind (gnat_node);
if (kind == N_Label)
return true;
if (IN (kind, N_Statement_Other_Than_Procedure_Call))
return true;
if (kind == N_Procedure_Call_Statement)
return true;
if (IN (kind, N_Raise_xxx_Error) && Ekind (Etype (gnat_node)) == E_Void)
return true;
return false;
}
tree
gnat_to_gnu (Node_Id gnat_node)
{
const Node_Kind kind = Nkind (gnat_node);
bool went_into_elab_proc = false;
tree gnu_result = error_mark_node; 
tree gnu_result_type = void_type_node;
tree gnu_expr, gnu_lhs, gnu_rhs;
Node_Id gnat_temp;
bool sync = false;
error_gnat_node = gnat_node;
Sloc_to_locus (Sloc (gnat_node), &input_location);
if (type_annotate_only && statement_node_p (gnat_node))
return alloc_stmt_list ();
if (type_annotate_only
&& IN (kind, N_Subexpr)
&& !(((IN (kind, N_Op) && kind != N_Op_Expon)
|| kind == N_Type_Conversion)
&& Is_Integer_Type (Etype (gnat_node)))
&& !(kind == N_Attribute_Reference
&& Get_Attribute_Id (Attribute_Name (gnat_node)) == Attr_Length
&& Ekind (Etype (Prefix (gnat_node))) == E_Array_Subtype
&& !Is_Constr_Subt_For_U_Nominal (Etype (Prefix (gnat_node))))
&& kind != N_Expanded_Name
&& kind != N_Identifier
&& !Compile_Time_Known_Value (gnat_node))
return build1 (NULL_EXPR, get_unpadded_type (Etype (gnat_node)),
build_call_raise (CE_Range_Check_Failed, gnat_node,
N_Raise_Constraint_Error));
if ((statement_node_p (gnat_node) && kind != N_Null_Statement)
|| kind == N_Handled_Sequence_Of_Statements
|| kind == N_Implicit_Label_Declaration)
{
tree current_elab_proc = get_elaboration_procedure ();
if (!current_function_decl)
{
current_function_decl = current_elab_proc;
went_into_elab_proc = true;
}
if (current_function_decl == current_elab_proc
&& kind != N_Handled_Sequence_Of_Statements
&& kind != N_Implicit_Label_Declaration)
Check_Elaboration_Code_Allowed (gnat_node);
}
switch (kind)
{
case N_Identifier:
case N_Expanded_Name:
case N_Operator_Symbol:
case N_Defining_Identifier:
case N_Defining_Operator_Symbol:
gnu_result = Identifier_to_gnu (gnat_node, &gnu_result_type);
if (atomic_access_required_p (gnat_node, &sync)
&& !present_in_lhs_or_actual_p (gnat_node))
gnu_result = build_atomic_load (gnu_result, sync);
break;
case N_Integer_Literal:
{
tree gnu_type;
gnu_type = gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (TREE_CODE (gnu_type) == RECORD_TYPE
&& TYPE_JUSTIFIED_MODULAR_P (gnu_type))
gnu_type = TREE_TYPE (TYPE_FIELDS (gnu_type));
gnu_result = UI_To_gnu (Intval (gnat_node), gnu_type);
gcc_assert (!TREE_OVERFLOW (gnu_result));
}
break;
case N_Character_Literal:
gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (Present (Entity (gnat_node)))
gnu_result = DECL_INITIAL (get_gnu_tree (Entity (gnat_node)));
else
gnu_result
= build_int_cst (gnu_result_type,
UI_To_CC (Char_Literal_Value (gnat_node)));
break;
case N_Real_Literal:
gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (Is_Fixed_Point_Type (Underlying_Type (Etype (gnat_node))))
{
gnu_result = UI_To_gnu (Corresponding_Integer_Value (gnat_node),
gnu_result_type);
gcc_assert (!TREE_OVERFLOW (gnu_result));
}
else
{
Ureal ur_realval = Realval (gnat_node);
if (!Is_Machine_Number (gnat_node))
ur_realval
= Machine (Base_Type (Underlying_Type (Etype (gnat_node))),
ur_realval, Round_Even, gnat_node);
if (UR_Is_Zero (ur_realval))
gnu_result = build_real (gnu_result_type, dconst0);
else
{
REAL_VALUE_TYPE tmp;
gnu_result = UI_To_gnu (Numerator (ur_realval), gnu_result_type);
gcc_assert (Rbase (ur_realval) == 2);
real_ldexp (&tmp, &TREE_REAL_CST (gnu_result),
- UI_To_Int (Denominator (ur_realval)));
gnu_result = build_real (gnu_result_type, tmp);
}
if (UR_Is_Negative (Realval (gnat_node)))
gnu_result
= build_unary_op (NEGATE_EXPR, get_base_type (gnu_result_type),
gnu_result);
}
break;
case N_String_Literal:
gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (TYPE_PRECISION (TREE_TYPE (gnu_result_type)) == HOST_BITS_PER_CHAR)
{
String_Id gnat_string = Strval (gnat_node);
int length = String_Length (gnat_string);
int i;
char *string;
if (length >= ALLOCA_THRESHOLD)
string = XNEWVEC (char, length + 1);
else
string = (char *) alloca (length + 1);
for (i = 0; i < length; i++)
string[i] = Get_String_Char (gnat_string, i + 1);
string[i] = 0;
gnu_result = build_string (length, string);
TREE_TYPE (gnu_result) = gnu_result_type;
if (length >= ALLOCA_THRESHOLD)
free (string);
}
else
{
String_Id gnat_string = Strval (gnat_node);
int length = String_Length (gnat_string);
int i;
tree gnu_idx = TYPE_MIN_VALUE (TYPE_DOMAIN (gnu_result_type));
tree gnu_one_node = convert (TREE_TYPE (gnu_idx), integer_one_node);
vec<constructor_elt, va_gc> *gnu_vec;
vec_alloc (gnu_vec, length);
for (i = 0; i < length; i++)
{
tree t = build_int_cst (TREE_TYPE (gnu_result_type),
Get_String_Char (gnat_string, i + 1));
CONSTRUCTOR_APPEND_ELT (gnu_vec, gnu_idx, t);
gnu_idx = int_const_binop (PLUS_EXPR, gnu_idx, gnu_one_node);
}
gnu_result = gnat_build_constructor (gnu_result_type, gnu_vec);
}
break;
case N_Pragma:
gnu_result = Pragma_to_gnu (gnat_node);
break;
case N_Subtype_Declaration:
case N_Full_Type_Declaration:
case N_Incomplete_Type_Declaration:
case N_Private_Type_Declaration:
case N_Private_Extension_Declaration:
case N_Task_Type_Declaration:
process_type (Defining_Entity (gnat_node));
gnu_result = alloc_stmt_list ();
break;
case N_Object_Declaration:
case N_Exception_Declaration:
gnat_temp = Defining_Entity (gnat_node);
gnu_result = alloc_stmt_list ();
if (type_annotate_only
&& (((Is_Array_Type (Etype (gnat_temp))
|| Is_Record_Type (Etype (gnat_temp)))
&& !Is_Constrained (Etype (gnat_temp)))
|| Is_Concurrent_Type (Etype (gnat_temp))))
break;
if (Present (Expression (gnat_node))
&& !(kind == N_Object_Declaration && No_Initialization (gnat_node))
&& (!type_annotate_only
|| Compile_Time_Known_Value (Expression (gnat_node))))
{
gnu_expr = gnat_to_gnu (Expression (gnat_node));
if (Do_Range_Check (Expression (gnat_node)))
gnu_expr
= emit_range_check (gnu_expr, Etype (gnat_temp), gnat_node);
if (type_annotate_only && TREE_CODE (gnu_expr) == ERROR_MARK)
gnu_expr = NULL_TREE;
}
else
gnu_expr = NULL_TREE;
if (Ekind (gnat_temp) == E_Constant
&& Present (Address_Clause (gnat_temp))
&& Present (Full_View (gnat_temp)))
save_gnu_tree (Full_View (gnat_temp), error_mark_node, true);
if (Present (Freeze_Node (gnat_temp)))
{
if (gnu_expr)
{
gnu_result = gnat_save_expr (gnu_expr);
save_gnu_tree (gnat_node, gnu_result, true);
}
}
else
gnat_to_gnu_entity (gnat_temp, gnu_expr, true);
break;
case N_Object_Renaming_Declaration:
gnat_temp = Defining_Entity (gnat_node);
gnu_result = alloc_stmt_list ();
if ((!Is_Renaming_Of_Object (gnat_temp)
|| (Needs_Debug_Info (gnat_temp)
&& !optimize
&& can_materialize_object_renaming_p
(Renamed_Object (gnat_temp))))
&& ! (type_annotate_only
&& (Is_Array_Type (Etype (gnat_temp))
|| Is_Record_Type (Etype (gnat_temp))
|| Is_Concurrent_Type (Etype (gnat_temp)))))
{
tree gnu_temp
= gnat_to_gnu_entity (gnat_temp,
gnat_to_gnu (Renamed_Object (gnat_temp)),
true);
if (TREE_SIDE_EFFECTS (gnu_temp))
gnu_result = build_unary_op (ADDR_EXPR, NULL_TREE, gnu_temp);
}
break;
case N_Exception_Renaming_Declaration:
gnat_temp = Defining_Entity (gnat_node);
gnu_result = alloc_stmt_list ();
if (Present (Renamed_Entity (gnat_temp)))
{
tree gnu_temp
= gnat_to_gnu_entity (gnat_temp,
gnat_to_gnu (Renamed_Entity (gnat_temp)),
true);
if (TREE_SIDE_EFFECTS (gnu_temp))
gnu_result = build_unary_op (ADDR_EXPR, NULL_TREE, gnu_temp);
}
break;
case N_Subprogram_Renaming_Declaration:
{
const Node_Id gnat_renaming = Defining_Entity (gnat_node);
const Node_Id gnat_renamed = Renamed_Entity (gnat_renaming);
gnu_result = alloc_stmt_list ();
if (!type_annotate_only
&& Needs_Debug_Info (gnat_renaming)
&& No (Freeze_Node (gnat_renaming))
&& Present (gnat_renamed)
&& (Ekind (gnat_renamed) == E_Function
|| Ekind (gnat_renamed) == E_Procedure)
&& !Is_Intrinsic_Subprogram (gnat_renaming)
&& !Is_Intrinsic_Subprogram (gnat_renamed))
gnat_to_gnu_entity (gnat_renaming, gnat_to_gnu (gnat_renamed), true);
break;
}
case N_Implicit_Label_Declaration:
gnat_to_gnu_entity (Defining_Entity (gnat_node), NULL_TREE, true);
gnu_result = alloc_stmt_list ();
break;
case N_Number_Declaration:
case N_Package_Renaming_Declaration:
gnu_result = alloc_stmt_list ();
break;
case N_Explicit_Dereference:
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_result = gnat_to_gnu (Prefix (gnat_node));
gnu_result = build_unary_op (INDIRECT_REF, NULL_TREE, gnu_result);
if (atomic_access_required_p (gnat_node, &sync)
&& !present_in_lhs_or_actual_p (gnat_node))
gnu_result = build_atomic_load (gnu_result, sync);
break;
case N_Indexed_Component:
{
tree gnu_array_object
= gnat_to_gnu (adjust_for_implicit_deref (Prefix (gnat_node)));
tree gnu_type;
int ndim;
int i;
Node_Id *gnat_expr_array;
gnu_array_object = maybe_implicit_deref (gnu_array_object);
if (VECTOR_TYPE_P (TREE_TYPE (gnu_array_object)))
{
if (present_in_lhs_or_actual_p (gnat_node))
gnat_mark_addressable (gnu_array_object);
gnu_array_object = maybe_vector_array (gnu_array_object);
}
gnu_array_object = maybe_unconstrained_array (gnu_array_object);
if (TYPE_IS_PADDING_P (TREE_TYPE (gnu_array_object)))
gnu_array_object
= convert (TREE_TYPE (TYPE_FIELDS (TREE_TYPE (gnu_array_object))),
gnu_array_object);
gcc_assert (TREE_CODE (TREE_TYPE (gnu_array_object)) == ARRAY_TYPE);
for (ndim = 1, gnu_type = TREE_TYPE (gnu_array_object);
TREE_CODE (TREE_TYPE (gnu_type)) == ARRAY_TYPE
&& TYPE_MULTI_ARRAY_P (TREE_TYPE (gnu_type));
ndim++, gnu_type = TREE_TYPE (gnu_type))
;
gnat_expr_array = XALLOCAVEC (Node_Id, ndim);
if (TYPE_CONVENTION_FORTRAN_P (TREE_TYPE (gnu_array_object)))
for (i = ndim - 1, gnat_temp = First (Expressions (gnat_node));
i >= 0;
i--, gnat_temp = Next (gnat_temp))
gnat_expr_array[i] = gnat_temp;
else
for (i = 0, gnat_temp = First (Expressions (gnat_node));
i < ndim;
i++, gnat_temp = Next (gnat_temp))
gnat_expr_array[i] = gnat_temp;
gnu_result = gnu_array_object;
for (i = 0, gnu_type = TREE_TYPE (gnu_array_object);
i < ndim;
i++, gnu_type = TREE_TYPE (gnu_type))
{
gcc_assert (TREE_CODE (gnu_type) == ARRAY_TYPE);
gnat_temp = gnat_expr_array[i];
gnu_expr = maybe_character_value (gnat_to_gnu (gnat_temp));
gnu_result
= build_binary_op (ARRAY_REF, NULL_TREE, gnu_result, gnu_expr);
}
gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (atomic_access_required_p (gnat_node, &sync)
&& !present_in_lhs_or_actual_p (gnat_node))
gnu_result = build_atomic_load (gnu_result, sync);
}
break;
case N_Slice:
{
tree gnu_array_object
= gnat_to_gnu (adjust_for_implicit_deref (Prefix (gnat_node)));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_array_object = maybe_implicit_deref (gnu_array_object);
gnu_array_object = maybe_unconstrained_array (gnu_array_object);
gnu_expr = TYPE_MIN_VALUE (TYPE_DOMAIN (gnu_result_type));
gnu_expr = maybe_character_value (gnu_expr);
if (!TREE_CONSTANT (TYPE_SIZE_UNIT (gnu_result_type))
&& TREE_CONSTANT (TYPE_SIZE_UNIT (TREE_TYPE (gnu_array_object))))
TYPE_ARRAY_MAX_SIZE (gnu_result_type)
= TYPE_SIZE_UNIT (TREE_TYPE (gnu_array_object));
gnu_result = build_binary_op (ARRAY_RANGE_REF, gnu_result_type,
gnu_array_object, gnu_expr);
}
break;
case N_Selected_Component:
{
Entity_Id gnat_prefix
= adjust_for_implicit_deref (Prefix (gnat_node));
Entity_Id gnat_field = Entity (Selector_Name (gnat_node));
tree gnu_prefix = gnat_to_gnu (gnat_prefix);
gnu_prefix = maybe_implicit_deref (gnu_prefix);
if (Ekind (gnat_field) == E_Discriminant)
{
if (Is_Tagged_Type (Underlying_Type (Etype (gnat_prefix))))
while (Present (Corresponding_Discriminant (gnat_field)))
gnat_field = Corresponding_Discriminant (gnat_field);
else if (Present (Corresponding_Discriminant (gnat_field)))
gnat_field = Original_Record_Component (gnat_field);
}
if (TREE_CODE (TREE_TYPE (gnu_prefix)) == COMPLEX_TYPE)
gnu_result = build_unary_op (Present (Next_Entity (gnat_field))
? REALPART_EXPR : IMAGPART_EXPR,
NULL_TREE, gnu_prefix);
else
{
tree gnu_field = gnat_to_gnu_field_decl (gnat_field);
gnu_result
= build_component_ref (gnu_prefix, gnu_field,
(Nkind (Parent (gnat_node))
== N_Attribute_Reference)
&& lvalue_required_for_attribute_p
(Parent (gnat_node)));
}
gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (atomic_access_required_p (gnat_node, &sync)
&& !present_in_lhs_or_actual_p (gnat_node))
gnu_result = build_atomic_load (gnu_result, sync);
}
break;
case N_Attribute_Reference:
{
const int attr = Get_Attribute_Id (Attribute_Name (gnat_node));
if (attr == Attr_Elab_Spec || attr == Attr_Elab_Body)
return
create_subprog_decl (create_concat_name
(Entity (Prefix (gnat_node)),
attr == Attr_Elab_Body ? "elabb" : "elabs"),
NULL_TREE, void_ftype, NULL_TREE, is_disabled,
true, true, true, true, false, NULL,
gnat_node);
gnu_result = Attribute_to_gnu (gnat_node, &gnu_result_type, attr);
}
break;
case N_Reference:
gnu_result = gnat_to_gnu (Prefix (gnat_node));
gnu_result = build_unary_op (ADDR_EXPR, NULL_TREE, gnu_result);
gnu_result_type = get_unpadded_type (Etype (gnat_node));
break;
case N_Aggregate:
case N_Extension_Aggregate:
{
tree gnu_aggr_type;
gcc_assert (!Expansion_Delayed (gnat_node));
gnu_aggr_type = gnu_result_type
= get_unpadded_type (Etype (gnat_node));
if (TREE_CODE (gnu_result_type) == RECORD_TYPE
&& TYPE_CONTAINS_TEMPLATE_P (gnu_result_type))
gnu_aggr_type
= TREE_TYPE (DECL_CHAIN (TYPE_FIELDS (gnu_result_type)));
else if (TREE_CODE (gnu_result_type) == VECTOR_TYPE)
gnu_aggr_type = TYPE_REPRESENTATIVE_ARRAY (gnu_result_type);
if (Null_Record_Present (gnat_node))
gnu_result = gnat_build_constructor (gnu_aggr_type, NULL);
else if (TREE_CODE (gnu_aggr_type) == RECORD_TYPE
|| TREE_CODE (gnu_aggr_type) == UNION_TYPE)
gnu_result
= assoc_to_constructor (Etype (gnat_node),
First (Component_Associations (gnat_node)),
gnu_aggr_type);
else if (TREE_CODE (gnu_aggr_type) == ARRAY_TYPE)
gnu_result = pos_to_constructor (First (Expressions (gnat_node)),
gnu_aggr_type,
Component_Type (Etype (gnat_node)));
else if (TREE_CODE (gnu_aggr_type) == COMPLEX_TYPE)
gnu_result
= build_binary_op
(COMPLEX_EXPR, gnu_aggr_type,
gnat_to_gnu (Expression (First
(Component_Associations (gnat_node)))),
gnat_to_gnu (Expression
(Next
(First (Component_Associations (gnat_node))))));
else
gcc_unreachable ();
gnu_result = convert (gnu_result_type, gnu_result);
}
break;
case N_Null:
if (TARGET_VTABLE_USES_DESCRIPTORS
&& Ekind (Etype (gnat_node)) == E_Access_Subprogram_Type
&& Is_Dispatch_Table_Entity (Etype (gnat_node)))
gnu_result = null_fdesc_node;
else
gnu_result = null_pointer_node;
gnu_result_type = get_unpadded_type (Etype (gnat_node));
break;
case N_Type_Conversion:
case N_Qualified_Expression:
gnu_expr = maybe_character_value (gnat_to_gnu (Expression (gnat_node)));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (kind == N_Qualified_Expression && Is_Tagged_Type (Etype (gnat_node)))
used_types_insert (gnu_result_type);
gnu_result
= convert_with_check (Etype (gnat_node), gnu_expr,
Do_Overflow_Check (gnat_node),
Do_Range_Check (Expression (gnat_node)),
kind == N_Type_Conversion
&& Float_Truncate (gnat_node), gnat_node);
break;
case N_Unchecked_Type_Conversion:
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_expr = maybe_character_value (gnat_to_gnu (Expression (gnat_node)));
if (unchecked_conversion_nop (gnat_node))
{
gnu_result = gnu_expr;
gnu_result_type = TREE_TYPE (gnu_result);
break;
}
if (STRICT_ALIGNMENT && POINTER_TYPE_P (gnu_result_type)
&& Is_Access_Type (Etype (gnat_node)))
{
unsigned int align = known_alignment (gnu_expr);
tree gnu_obj_type = TREE_TYPE (gnu_result_type);
unsigned int oalign = TYPE_ALIGN (gnu_obj_type);
if (align != 0 && align < oalign && !TYPE_ALIGN_OK (gnu_obj_type))
post_error_ne_tree_2
("?source alignment (^) '< alignment of & (^)",
gnat_node, Designated_Type (Etype (gnat_node)),
size_int (align / BITS_PER_UNIT), oalign / BITS_PER_UNIT);
}
if (TARGET_VTABLE_USES_DESCRIPTORS
&& TREE_TYPE (gnu_expr) == fdesc_type_node
&& POINTER_TYPE_P (gnu_result_type))
gnu_expr = build_unary_op (ADDR_EXPR, NULL_TREE, gnu_expr);
gnu_result = unchecked_convert (gnu_result_type, gnu_expr,
No_Truncation (gnat_node));
break;
case N_In:
case N_Not_In:
{
tree gnu_obj = gnat_to_gnu (Left_Opnd (gnat_node));
tree gnu_low, gnu_high;
Range_to_gnu (Right_Opnd (gnat_node), &gnu_low, &gnu_high);
gnu_result_type = get_unpadded_type (Etype (gnat_node));
tree gnu_op_type = maybe_character_type (TREE_TYPE (gnu_obj));
if (TREE_TYPE (gnu_obj) != gnu_op_type)
{
gnu_obj = convert (gnu_op_type, gnu_obj);
gnu_low = convert (gnu_op_type, gnu_low);
gnu_high = convert (gnu_op_type, gnu_high);
}
if (operand_equal_p (gnu_low, gnu_high, 0))
gnu_result
= build_binary_op (EQ_EXPR, gnu_result_type, gnu_obj, gnu_low);
else
{
tree t1, t2;
gnu_obj = gnat_protect_expr (gnu_obj);
t1 = build_binary_op (GE_EXPR, gnu_result_type, gnu_obj, gnu_low);
if (EXPR_P (t1))
set_expr_location_from_node (t1, gnat_node);
t2 = build_binary_op (LE_EXPR, gnu_result_type, gnu_obj, gnu_high);
if (EXPR_P (t2))
set_expr_location_from_node (t2, gnat_node);
gnu_result
= build_binary_op (TRUTH_ANDIF_EXPR, gnu_result_type, t1, t2);
}
if (kind == N_Not_In)
gnu_result
= invert_truthvalue_loc (EXPR_LOCATION (gnu_result), gnu_result);
}
break;
case N_Op_Divide:
gnu_lhs = gnat_to_gnu (Left_Opnd (gnat_node));
gnu_rhs = gnat_to_gnu (Right_Opnd (gnat_node));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_result = build_binary_op (FLOAT_TYPE_P (gnu_result_type)
? RDIV_EXPR
: (Rounded_Result (gnat_node)
? ROUND_DIV_EXPR : TRUNC_DIV_EXPR),
gnu_result_type, gnu_lhs, gnu_rhs);
break;
case N_Op_And:
case N_Op_Or:
case N_Op_Xor:
if (Is_Modular_Integer_Type (Underlying_Type (Etype (gnat_node))))
{
enum tree_code code
= (kind == N_Op_Or ? BIT_IOR_EXPR
: kind == N_Op_And ? BIT_AND_EXPR
: BIT_XOR_EXPR);
gnu_lhs = gnat_to_gnu (Left_Opnd (gnat_node));
gnu_rhs = gnat_to_gnu (Right_Opnd (gnat_node));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_result = build_binary_op (code, gnu_result_type,
gnu_lhs, gnu_rhs);
break;
}
case N_Op_Eq:
case N_Op_Ne:
case N_Op_Lt:
case N_Op_Le:
case N_Op_Gt:
case N_Op_Ge:
case N_Op_Add:
case N_Op_Subtract:
case N_Op_Multiply:
case N_Op_Mod:
case N_Op_Rem:
case N_Op_Rotate_Left:
case N_Op_Rotate_Right:
case N_Op_Shift_Left:
case N_Op_Shift_Right:
case N_Op_Shift_Right_Arithmetic:
case N_And_Then:
case N_Or_Else:
{
enum tree_code code = gnu_codes[kind];
bool ignore_lhs_overflow = false;
location_t saved_location = input_location;
tree gnu_type;
gnu_lhs = gnat_to_gnu (Left_Opnd (gnat_node));
gnu_rhs = gnat_to_gnu (Right_Opnd (gnat_node));
gnu_type = gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_lhs = maybe_vector_array (gnu_lhs);
gnu_rhs = maybe_vector_array (gnu_rhs);
if (TREE_CODE_CLASS (code) == tcc_comparison)
{
gnu_lhs = maybe_unconstrained_array (gnu_lhs);
gnu_rhs = maybe_unconstrained_array (gnu_rhs);
tree gnu_op_type = maybe_character_type (TREE_TYPE (gnu_lhs));
if (TREE_TYPE (gnu_lhs) != gnu_op_type)
{
gnu_lhs = convert (gnu_op_type, gnu_lhs);
gnu_rhs = convert (gnu_op_type, gnu_rhs);
}
}
if (IN (kind, N_Op_Shift) && !Shift_Count_OK (gnat_node))
{
tree gnu_count_type = get_base_type (TREE_TYPE (gnu_rhs));
tree gnu_max_shift
= convert (gnu_count_type, TYPE_SIZE (gnu_type));
if (kind == N_Op_Rotate_Left || kind == N_Op_Rotate_Right)
gnu_rhs = build_binary_op (TRUNC_MOD_EXPR, gnu_count_type,
gnu_rhs, gnu_max_shift);
else if (kind == N_Op_Shift_Right_Arithmetic)
gnu_rhs
= build_binary_op
(MIN_EXPR, gnu_count_type,
build_binary_op (MINUS_EXPR,
gnu_count_type,
gnu_max_shift,
build_int_cst (gnu_count_type, 1)),
gnu_rhs);
}
if (kind == N_Op_Shift_Right && !TYPE_UNSIGNED (gnu_type))
{
gnu_type = gnat_unsigned_type_for (gnu_type);
ignore_lhs_overflow = true;
}
else if (kind == N_Op_Shift_Right_Arithmetic
&& TYPE_UNSIGNED (gnu_type))
{
gnu_type = gnat_signed_type_for (gnu_type);
ignore_lhs_overflow = true;
}
if (gnu_type != gnu_result_type)
{
tree gnu_old_lhs = gnu_lhs;
gnu_lhs = convert (gnu_type, gnu_lhs);
if (TREE_CODE (gnu_lhs) == INTEGER_CST && ignore_lhs_overflow)
TREE_OVERFLOW (gnu_lhs) = TREE_OVERFLOW (gnu_old_lhs);
gnu_rhs = convert (gnu_type, gnu_rhs);
}
if (Do_Overflow_Check (gnat_node)
&& Backend_Overflow_Checks_On_Target
&& (code == PLUS_EXPR || code == MINUS_EXPR || code == MULT_EXPR)
&& !TYPE_UNSIGNED (gnu_type)
&& !FLOAT_TYPE_P (gnu_type))
gnu_result = build_binary_op_trapv (code, gnu_type,
gnu_lhs, gnu_rhs, gnat_node);
else
{
input_location = saved_location;
gnu_result = build_binary_op (code, gnu_type, gnu_lhs, gnu_rhs);
}
if ((kind == N_Op_Shift_Left || kind == N_Op_Shift_Right)
&& !Shift_Count_OK (gnat_node))
gnu_result
= build_cond_expr
(gnu_type,
build_binary_op (GE_EXPR, boolean_type_node,
gnu_rhs,
convert (TREE_TYPE (gnu_rhs),
TYPE_SIZE (gnu_type))),
build_int_cst (gnu_type, 0),
gnu_result);
}
break;
case N_If_Expression:
{
tree gnu_cond = gnat_to_gnu (First (Expressions (gnat_node)));
tree gnu_true = gnat_to_gnu (Next (First (Expressions (gnat_node))));
tree gnu_false
= gnat_to_gnu (Next (Next (First (Expressions (gnat_node)))));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_result
= build_cond_expr (gnu_result_type, gnu_cond, gnu_true, gnu_false);
}
break;
case N_Op_Plus:
gnu_result = gnat_to_gnu (Right_Opnd (gnat_node));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
break;
case N_Op_Not:
if (Is_Modular_Integer_Type (Underlying_Type (Etype (gnat_node))))
{
gnu_expr = gnat_to_gnu (Right_Opnd (gnat_node));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
gnu_result = build_unary_op (BIT_NOT_EXPR, gnu_result_type,
gnu_expr);
break;
}
case N_Op_Minus:
case N_Op_Abs:
gnu_expr = gnat_to_gnu (Right_Opnd (gnat_node));
gnu_result_type = get_unpadded_type (Etype (gnat_node));
if (Do_Overflow_Check (gnat_node)
&& Backend_Overflow_Checks_On_Target
&& !TYPE_UNSIGNED (gnu_result_type)
&& !FLOAT_TYPE_P (gnu_result_type))
gnu_result
= build_unary_op_trapv (gnu_codes[kind],
gnu_result_type, gnu_expr, gnat_node);
else
gnu_result = build_unary_op (gnu_codes[kind],
gnu_result_type, gnu_expr);
break;
case N_Allocator:
{
tree gnu_init = NULL_TREE;
tree gnu_type;
bool ignore_init_type = false;
gnat_temp = Expression (gnat_node);
if (Nkind (gnat_temp) == N_Identifier
|| Nkind (gnat_temp) == N_Expanded_Name)
gnu_type = gnat_to_gnu_type (Entity (gnat_temp));
else if (Nkind (gnat_temp) == N_Qualified_Expression)
{
Entity_Id gnat_desig_type
= Designated_Type (Underlying_Type (Etype (gnat_node)));
ignore_init_type = Has_Constrained_Partial_View (gnat_desig_type);
gnu_init = gnat_to_gnu (Expression (gnat_temp));
gnu_init = maybe_unconstrained_array (gnu_init);
if (Do_Range_Check (Expression (gnat_temp)))
gnu_init
= emit_range_check (gnu_init, gnat_desig_type, gnat_temp);
if (Is_Elementary_Type (gnat_desig_type)
|| Is_Constrained (gnat_desig_type))
gnu_type = gnat_to_gnu_type (gnat_desig_type);
else
{
gnu_type = gnat_to_gnu_type (Etype (Expression (gnat_temp)));
if (TREE_CODE (gnu_type) == UNCONSTRAINED_ARRAY_TYPE)
gnu_type = TREE_TYPE (gnu_init);
}
if (Is_Tagged_Type (gnat_desig_type))
used_types_insert (gnu_type);
gnu_init = convert (gnu_type, gnu_init);
}
else
gcc_unreachable ();
gnu_result_type = get_unpadded_type (Etype (gnat_node));
return build_allocator (gnu_type, gnu_init, gnu_result_type,
Procedure_To_Call (gnat_node),
Storage_Pool (gnat_node), gnat_node,
ignore_init_type);
}
break;
case N_Label:
gnu_result = build1 (LABEL_EXPR, void_type_node,
gnat_to_gnu (Identifier (gnat_node)));
break;
case N_Null_Statement:
if (!optimize && Comes_From_Source (gnat_node))
{
tree stmt, label = create_label_decl (NULL_TREE, gnat_node);
DECL_IGNORED_P (label) = 1;
start_stmt_group ();
stmt = build1 (GOTO_EXPR, void_type_node, label);
set_expr_location_from_node (stmt, gnat_node);
add_stmt (stmt);
stmt = build1 (LABEL_EXPR, void_type_node, label);
set_expr_location_from_node (stmt, gnat_node);
add_stmt (stmt);
gnu_result = end_stmt_group ();
}
else
gnu_result = alloc_stmt_list ();
break;
case N_Assignment_Statement:
gnu_lhs = maybe_unconstrained_array (gnat_to_gnu (Name (gnat_node)));
if (TREE_CODE (TYPE_SIZE_UNIT (TREE_TYPE (gnu_lhs))) == INTEGER_CST
&& !valid_constant_size_p (TYPE_SIZE_UNIT (TREE_TYPE (gnu_lhs))))
gnu_result = build_call_raise (SE_Object_Too_Large, gnat_node,
N_Raise_Storage_Error);
else if (Nkind (Expression (gnat_node)) == N_Function_Call)
{
bool outer_atomic_access
= outer_atomic_access_required_p (Name (gnat_node));
bool atomic_access
= !outer_atomic_access
&& atomic_access_required_p (Name (gnat_node), &sync);
gnu_result
= Call_to_gnu (Expression (gnat_node), &gnu_result_type, gnu_lhs,
outer_atomic_access, atomic_access, sync);
}
else
{
const Node_Id gnat_expr = Expression (gnat_node);
const Entity_Id gnat_type
= Underlying_Type (Etype (Name (gnat_node)));
const bool regular_array_type_p
= (Is_Array_Type (gnat_type) && !Is_Bit_Packed_Array (gnat_type));
const bool use_memset_p
= (regular_array_type_p
&& Nkind (gnat_expr) == N_Aggregate
&& Is_Others_Aggregate (gnat_expr));
if (use_memset_p)
{
Node_Id gnat_inner
= Expression (First (Component_Associations (gnat_expr)));
while (Nkind (gnat_inner) == N_Aggregate
&& Is_Others_Aggregate (gnat_inner))
gnat_inner
= Expression (First (Component_Associations (gnat_inner)));
gnu_rhs = gnat_to_gnu (gnat_inner);
}
else
gnu_rhs = maybe_unconstrained_array (gnat_to_gnu (gnat_expr));
if (Do_Range_Check (gnat_expr))
gnu_rhs = emit_range_check (gnu_rhs, Etype (Name (gnat_node)),
gnat_node);
if (outer_atomic_access_required_p (Name (gnat_node)))
gnu_result = build_load_modify_store (gnu_lhs, gnu_rhs, gnat_node);
else if (atomic_access_required_p (Name (gnat_node), &sync))
gnu_result = build_atomic_store (gnu_lhs, gnu_rhs, sync);
else if (use_memset_p)
{
tree value
= real_zerop (gnu_rhs)
? integer_zero_node
: fold_convert (integer_type_node, gnu_rhs);
tree dest = build_fold_addr_expr (gnu_lhs);
tree t = builtin_decl_explicit (BUILT_IN_MEMSET);
tree size;
if (TREE_CODE (gnu_lhs) == COMPONENT_REF)
size = DECL_SIZE_UNIT (TREE_OPERAND (gnu_lhs, 1));
else if (DECL_P (gnu_lhs))
size = DECL_SIZE_UNIT (gnu_lhs);
else
size = TYPE_SIZE_UNIT (TREE_TYPE (gnu_lhs));
size = SUBSTITUTE_PLACEHOLDER_IN_EXPR (size, gnu_lhs);
if (TREE_CODE (value) == INTEGER_CST && !integer_zerop (value))
{
tree mask
= build_int_cst (integer_type_node,
((HOST_WIDE_INT) 1 << BITS_PER_UNIT) - 1);
value = int_const_binop (BIT_AND_EXPR, value, mask);
}
gnu_result = build_call_expr (t, 3, dest, value, size);
}
else
gnu_result
= build_binary_op (MODIFY_EXPR, NULL_TREE, gnu_lhs, gnu_rhs);
if (TREE_CODE (gnu_result) == MODIFY_EXPR
&& regular_array_type_p
&& !(Forwards_OK (gnat_node) && Backwards_OK (gnat_node)))
{
tree to = TREE_OPERAND (gnu_result, 0);
tree from = TREE_OPERAND (gnu_result, 1);
tree type = TREE_TYPE (from);
tree size
= SUBSTITUTE_PLACEHOLDER_IN_EXPR (TYPE_SIZE_UNIT (type), from);
tree to_ptr = build_fold_addr_expr (to);
tree from_ptr = build_fold_addr_expr (from);
tree t = builtin_decl_explicit (BUILT_IN_MEMMOVE);
gnu_result = build_call_expr (t, 3, to_ptr, from_ptr, size);
}
}
break;
case N_If_Statement:
{
tree *gnu_else_ptr; 
gnu_result = build3 (COND_EXPR, void_type_node,
gnat_to_gnu (Condition (gnat_node)),
NULL_TREE, NULL_TREE);
COND_EXPR_THEN (gnu_result)
= build_stmt_group (Then_Statements (gnat_node), false);
TREE_SIDE_EFFECTS (gnu_result) = 1;
gnu_else_ptr = &COND_EXPR_ELSE (gnu_result);
if (Present (Elsif_Parts (gnat_node)))
for (gnat_temp = First (Elsif_Parts (gnat_node));
Present (gnat_temp); gnat_temp = Next (gnat_temp))
{
gnu_expr = build3 (COND_EXPR, void_type_node,
gnat_to_gnu (Condition (gnat_temp)),
NULL_TREE, NULL_TREE);
COND_EXPR_THEN (gnu_expr)
= build_stmt_group (Then_Statements (gnat_temp), false);
TREE_SIDE_EFFECTS (gnu_expr) = 1;
set_expr_location_from_node (gnu_expr, gnat_temp);
*gnu_else_ptr = gnu_expr;
gnu_else_ptr = &COND_EXPR_ELSE (gnu_expr);
}
*gnu_else_ptr = build_stmt_group (Else_Statements (gnat_node), false);
}
break;
case N_Case_Statement:
gnu_result = Case_Statement_to_gnu (gnat_node);
break;
case N_Loop_Statement:
gnu_result = Loop_Statement_to_gnu (gnat_node);
break;
case N_Block_Statement:
if (stmt_group_may_fallthru ())
{
start_stmt_group ();
gnat_pushlevel ();
process_decls (Declarations (gnat_node), Empty, Empty, true, true);
add_stmt (gnat_to_gnu (Handled_Statement_Sequence (gnat_node)));
gnat_poplevel ();
gnu_result = end_stmt_group ();
}
else
gnu_result = alloc_stmt_list ();
break;
case N_Exit_Statement:
gnu_result
= build2 (EXIT_STMT, void_type_node,
(Present (Condition (gnat_node))
? gnat_to_gnu (Condition (gnat_node)) : NULL_TREE),
(Present (Name (gnat_node))
? get_gnu_tree (Entity (Name (gnat_node)))
: LOOP_STMT_LABEL (gnu_loop_stack->last ()->stmt)));
break;
case N_Simple_Return_Statement:
{
tree gnu_ret_obj, gnu_ret_val;
if (Present (Expression (gnat_node)))
{
tree gnu_subprog_type = TREE_TYPE (current_function_decl);
if (TYPE_CI_CO_LIST (gnu_subprog_type)
&& !TREE_ADDRESSABLE (gnu_subprog_type))
gnu_ret_obj = gnu_return_var_stack->last ();
else
gnu_ret_obj = DECL_RESULT (current_function_decl);
gnu_ret_val = gnat_to_gnu (Expression (gnat_node));
if (TREE_CODE (gnu_ret_val) == COMPONENT_REF
&& type_is_padding_self_referential
(TREE_TYPE (TREE_OPERAND (gnu_ret_val, 0))))
gnu_ret_val = TREE_OPERAND (gnu_ret_val, 0);
if (TYPE_RETURN_BY_DIRECT_REF_P (gnu_subprog_type)
|| By_Ref (gnat_node))
gnu_ret_val = build_unary_op (ADDR_EXPR, NULL_TREE, gnu_ret_val);
else if (TYPE_RETURN_UNCONSTRAINED_P (gnu_subprog_type))
{
gnu_ret_val = maybe_unconstrained_array (gnu_ret_val);
if (!TYPE_CI_CO_LIST (gnu_subprog_type) && optimize)
{
tree ret_val = gnu_ret_val;
if (gnat_useless_type_conversion (ret_val))
ret_val = TREE_OPERAND (ret_val, 0);
if (TREE_CODE (ret_val) == COMPONENT_REF
&& TYPE_IS_PADDING_P
(TREE_TYPE (TREE_OPERAND (ret_val, 0))))
ret_val = TREE_OPERAND (ret_val, 0);
if (return_value_ok_for_nrv_p (NULL_TREE, ret_val))
{
if (!f_named_ret_val)
f_named_ret_val = BITMAP_GGC_ALLOC ();
bitmap_set_bit (f_named_ret_val, DECL_UID (ret_val));
if (!f_gnat_ret)
f_gnat_ret = gnat_node;
}
}
gnu_ret_val = build_allocator (TREE_TYPE (gnu_ret_val),
gnu_ret_val,
TREE_TYPE (gnu_ret_obj),
Procedure_To_Call (gnat_node),
Storage_Pool (gnat_node),
gnat_node, false);
}
else if (TREE_ADDRESSABLE (gnu_subprog_type))
{
tree gnu_ret_deref
= build_unary_op (INDIRECT_REF, TREE_TYPE (gnu_ret_val),
gnu_ret_obj);
gnu_result = build2 (INIT_EXPR, void_type_node,
gnu_ret_deref, gnu_ret_val);
add_stmt_with_node (gnu_result, gnat_node);
gnu_ret_val = NULL_TREE;
}
}
else
gnu_ret_obj = gnu_ret_val = NULL_TREE;
if (gnu_return_label_stack->last ())
{
if (gnu_ret_val)
add_stmt (build_binary_op (MODIFY_EXPR, NULL_TREE, gnu_ret_obj,
gnu_ret_val));
gnu_result = build1 (GOTO_EXPR, void_type_node,
gnu_return_label_stack->last ());
if (!optimize && Comes_From_Source (gnat_node))
DECL_ARTIFICIAL (gnu_return_label_stack->last ()) = 0;
}
else
gnu_result = build_return_expr (gnu_ret_obj, gnu_ret_val);
}
break;
case N_Goto_Statement:
gnu_expr = gnat_to_gnu (Name (gnat_node));
gnu_result = build1 (GOTO_EXPR, void_type_node, gnu_expr);
TREE_USED (gnu_expr) = 1;
break;
case N_Subprogram_Declaration:
if (No (Freeze_Node (Defining_Entity (Specification (gnat_node)))))
gnat_to_gnu_entity (Defining_Entity (Specification (gnat_node)),
NULL_TREE, true);
gnu_result = alloc_stmt_list ();
break;
case N_Abstract_Subprogram_Declaration:
if (No (Freeze_Node (Defining_Entity (Specification (gnat_node)))))
for (gnat_temp
= First_Formal_With_Extras
(Defining_Entity (Specification (gnat_node)));
Present (gnat_temp);
gnat_temp = Next_Formal_With_Extras (gnat_temp))
if (Is_Itype (Etype (gnat_temp))
&& !From_Limited_With (Etype (gnat_temp)))
gnat_to_gnu_entity (Etype (gnat_temp), NULL_TREE, false);
{
Entity_Id gnat_temp_type
= Etype (Defining_Entity (Specification (gnat_node)));
if (Is_Itype (gnat_temp_type) && !From_Limited_With (gnat_temp_type))
gnat_to_gnu_entity (Etype (gnat_temp_type), NULL_TREE, false);
}
gnu_result = alloc_stmt_list ();
break;
case N_Defining_Program_Unit_Name:
gnu_result = gnat_to_gnu (Parent (gnat_node));
break;
case N_Subprogram_Body:
Subprogram_Body_to_gnu (gnat_node);
gnu_result = alloc_stmt_list ();
break;
case N_Function_Call:
case N_Procedure_Call_Statement:
gnu_result = Call_to_gnu (gnat_node, &gnu_result_type, NULL_TREE,
false, false, false);
break;
case N_Package_Declaration:
gnu_result = gnat_to_gnu (Specification (gnat_node));
break;
case N_Package_Specification:
start_stmt_group ();
process_decls (Visible_Declarations (gnat_node),
Private_Declarations (gnat_node), Empty, true, true);
gnu_result = end_stmt_group ();
break;
case N_Package_Body:
if (Ekind (Corresponding_Spec (gnat_node)) == E_Generic_Package)
{
gnu_result = alloc_stmt_list ();
break;
}
start_stmt_group ();
process_decls (Declarations (gnat_node), Empty, Empty, true, true);
if (Present (Handled_Statement_Sequence (gnat_node)))
add_stmt (gnat_to_gnu (Handled_Statement_Sequence (gnat_node)));
gnu_result = end_stmt_group ();
break;
case N_Use_Package_Clause:
case N_Use_Type_Clause:
gnu_result = alloc_stmt_list ();
break;
case N_Protected_Type_Declaration:
gnu_result = alloc_stmt_list ();
break;
case N_Single_Task_Declaration:
gnat_to_gnu_entity (Defining_Entity (gnat_node), NULL_TREE, true);
gnu_result = alloc_stmt_list ();
break;
case N_Compilation_Unit:
Compilation_Unit_to_gnu (gnat_node);
gnu_result = alloc_stmt_list ();
break;
case N_Subunit:
gnu_result = gnat_to_gnu (Proper_Body (gnat_node));
break;
case N_Entry_Body:
case N_Protected_Body:
case N_Task_Body:
gcc_assert (type_annotate_only);
process_decls (Declarations (gnat_node), Empty, Empty, true, true);
gnu_result = alloc_stmt_list ();
break;
case N_Subprogram_Body_Stub:
case N_Package_Body_Stub:
case N_Protected_Body_Stub:
case N_Task_Body_Stub:
if (Present (Library_Unit (gnat_node)))
gnu_result = gnat_to_gnu (Unit (Library_Unit (gnat_node)));
else
{
gcc_assert (type_annotate_only);
gnu_result = alloc_stmt_list ();
}
break;
case N_Handled_Sequence_Of_Statements:
gcc_assert (type_annotate_only
|| !Front_End_Exceptions ()
|| No (At_End_Proc (gnat_node))
|| Present (Exception_Handlers (gnat_node))
|| No_Exception_Handlers_Set ());
gnu_result = Handled_Sequence_Of_Statements_to_gnu (gnat_node);
break;
case N_Exception_Handler:
if (Exception_Mechanism == Front_End_SJLJ)
gnu_result = Exception_Handler_to_gnu_fe_sjlj (gnat_node);
else if (Back_End_Exceptions ())
gnu_result = Exception_Handler_to_gnu_gcc (gnat_node);
else
gcc_unreachable ();
break;
case N_Raise_Statement:
gcc_assert (No (Name (gnat_node))
&& Back_End_Exceptions ());
start_stmt_group ();
gnat_pushlevel ();
gnu_expr = create_var_decl (get_identifier ("SAVED_EXPTR"), NULL_TREE,
ptr_type_node, gnu_incoming_exc_ptr,
false, false, false, false, false,
true, true, NULL, gnat_node);
add_stmt (build_binary_op (MODIFY_EXPR, NULL_TREE, gnu_incoming_exc_ptr,
build_int_cst (ptr_type_node, 0)));
add_stmt (build_call_n_expr (reraise_zcx_decl, 1, gnu_expr));
gnat_poplevel ();
gnu_result = end_stmt_group ();
break;
case N_Push_Constraint_Error_Label:
gnu_constraint_error_label_stack.safe_push (Exception_Label (gnat_node));
break;
case N_Push_Storage_Error_Label:
gnu_storage_error_label_stack.safe_push (Exception_Label (gnat_node));
break;
case N_Push_Program_Error_Label:
gnu_program_error_label_stack.safe_push (Exception_Label (gnat_node));
break;
case N_Pop_Constraint_Error_Label:
gnat_temp = gnu_constraint_error_label_stack.pop ();
if (Present (gnat_temp)
&& !TREE_USED (gnat_to_gnu_entity (gnat_temp, NULL_TREE, false)))
Warn_If_No_Local_Raise (gnat_temp);
break;
case N_Pop_Storage_Error_Label:
gnat_temp = gnu_storage_error_label_stack.pop ();
if (Present (gnat_temp)
&& !TREE_USED (gnat_to_gnu_entity (gnat_temp, NULL_TREE, false)))
Warn_If_No_Local_Raise (gnat_temp);
break;
case N_Pop_Program_Error_Label:
gnat_temp = gnu_program_error_label_stack.pop ();
if (Present (gnat_temp)
&& !TREE_USED (gnat_to_gnu_entity (gnat_temp, NULL_TREE, false)))
Warn_If_No_Local_Raise (gnat_temp);
break;
case N_Generic_Function_Renaming_Declaration:
case N_Generic_Package_Renaming_Declaration:
case N_Generic_Procedure_Renaming_Declaration:
case N_Generic_Package_Declaration:
case N_Generic_Subprogram_Declaration:
case N_Package_Instantiation:
case N_Procedure_Instantiation:
case N_Function_Instantiation:
gnu_result = alloc_stmt_list ();
break;
case N_Attribute_Definition_Clause:
gnu_result = alloc_stmt_list ();
if (Get_Attribute_Id (Chars (gnat_node)) != Attr_Address)
break;
gnat_temp = Entity (Name (gnat_node));
if (No (Freeze_Node (gnat_temp)))
break;
save_gnu_tree (gnat_temp, gnat_to_gnu (Expression (gnat_node)), true);
break;
case N_Enumeration_Representation_Clause:
case N_Record_Representation_Clause:
case N_At_Clause:
gnu_result = alloc_stmt_list ();
break;
case N_Code_Statement:
if (!type_annotate_only)
{
tree gnu_template = gnat_to_gnu (Asm_Template (gnat_node));
tree gnu_inputs = NULL_TREE, gnu_outputs = NULL_TREE;
tree gnu_clobbers = NULL_TREE, tail;
bool allows_mem, allows_reg, fake;
int ninputs, noutputs, i;
const char **oconstraints;
const char *constraint;
char *clobber;
Setup_Asm_Outputs (gnat_node);
while (Present (gnat_temp = Asm_Output_Variable ()))
{
tree gnu_value = gnat_to_gnu (gnat_temp);
tree gnu_constr = build_tree_list (NULL_TREE, gnat_to_gnu
(Asm_Output_Constraint ()));
gnu_outputs = tree_cons (gnu_constr, gnu_value, gnu_outputs);
Next_Asm_Output ();
}
Setup_Asm_Inputs (gnat_node);
while (Present (gnat_temp = Asm_Input_Value ()))
{
tree gnu_value = gnat_to_gnu (gnat_temp);
tree gnu_constr = build_tree_list (NULL_TREE, gnat_to_gnu
(Asm_Input_Constraint ()));
gnu_inputs = tree_cons (gnu_constr, gnu_value, gnu_inputs);
Next_Asm_Input ();
}
Clobber_Setup (gnat_node);
while ((clobber = Clobber_Get_Next ()))
gnu_clobbers
= tree_cons (NULL_TREE,
build_string (strlen (clobber) + 1, clobber),
gnu_clobbers);
gnu_outputs = nreverse (gnu_outputs);
noutputs = list_length (gnu_outputs);
gnu_inputs = nreverse (gnu_inputs);
ninputs = list_length (gnu_inputs);
oconstraints = XALLOCAVEC (const char *, noutputs);
for (i = 0, tail = gnu_outputs; tail; ++i, tail = TREE_CHAIN (tail))
{
tree output = TREE_VALUE (tail);
constraint
= TREE_STRING_POINTER (TREE_VALUE (TREE_PURPOSE (tail)));
oconstraints[i] = constraint;
if (parse_output_constraint (&constraint, i, ninputs, noutputs,
&allows_mem, &allows_reg, &fake))
{
if (!allows_reg)
{
output = remove_conversions (output, false);
if (TREE_CODE (output) == CONST_DECL
&& DECL_CONST_CORRESPONDING_VAR (output))
output = DECL_CONST_CORRESPONDING_VAR (output);
if (!gnat_mark_addressable (output))
output = error_mark_node;
}
}
else
output = error_mark_node;
TREE_VALUE (tail) = output;
}
for (i = 0, tail = gnu_inputs; tail; ++i, tail = TREE_CHAIN (tail))
{
tree input = TREE_VALUE (tail);
constraint
= TREE_STRING_POINTER (TREE_VALUE (TREE_PURPOSE (tail)));
if (parse_input_constraint (&constraint, i, ninputs, noutputs,
0, oconstraints,
&allows_mem, &allows_reg))
{
if (!allows_reg && allows_mem)
{
input = remove_conversions (input, false);
if (TREE_CODE (input) == CONST_DECL
&& DECL_CONST_CORRESPONDING_VAR (input))
input = DECL_CONST_CORRESPONDING_VAR (input);
if (!gnat_mark_addressable (input))
input = error_mark_node;
}
}
else
input = error_mark_node;
TREE_VALUE (tail) = input;
}
gnu_result = build5 (ASM_EXPR,  void_type_node,
gnu_template, gnu_outputs,
gnu_inputs, gnu_clobbers, NULL_TREE);
ASM_VOLATILE_P (gnu_result) = Is_Asm_Volatile (gnat_node);
}
else
gnu_result = alloc_stmt_list ();
break;
case N_Call_Marker:
case N_Variable_Reference_Marker:
gnu_result = alloc_stmt_list ();
break;
case N_Expression_With_Actions:
start_stmt_group ();
add_stmt_list (Actions (gnat_node));
gnu_expr = gnat_to_gnu (Expression (gnat_node));
gnu_result = end_stmt_group ();
gnu_result = build1 (SAVE_EXPR, void_type_node, gnu_result);
TREE_SIDE_EFFECTS (gnu_result) = 1;
gnu_result
= build_compound_expr (TREE_TYPE (gnu_expr), gnu_result, gnu_expr);
gnu_result_type = get_unpadded_type (Etype (gnat_node));
break;
case N_Freeze_Entity:
start_stmt_group ();
process_freeze_entity (gnat_node);
process_decls (Actions (gnat_node), Empty, Empty, true, true);
gnu_result = end_stmt_group ();
break;
case N_Freeze_Generic_Entity:
gnu_result = alloc_stmt_list ();
break;
case N_Itype_Reference:
if (!present_gnu_tree (Itype (gnat_node)))
process_type (Itype (gnat_node));
gnu_result = alloc_stmt_list ();
break;
case N_Free_Statement:
if (!type_annotate_only)
{
tree gnu_ptr
= gnat_to_gnu (adjust_for_implicit_deref (Expression (gnat_node)));
tree gnu_ptr_type = TREE_TYPE (gnu_ptr);
tree gnu_obj_type, gnu_actual_obj_type;
if (TYPE_IS_THIN_POINTER_P (TREE_TYPE (gnu_ptr)))
gnu_ptr = build_unary_op (ADDR_EXPR, NULL_TREE,
build_unary_op (INDIRECT_REF, NULL_TREE,
gnu_ptr));
if (TYPE_IS_FAT_POINTER_P (TREE_TYPE (gnu_ptr)))
gnu_ptr
= convert (build_pointer_type
(TYPE_OBJECT_RECORD_TYPE
(TYPE_UNCONSTRAINED_ARRAY (TREE_TYPE (gnu_ptr)))),
gnu_ptr);
gnu_obj_type = TREE_TYPE (TREE_TYPE (gnu_ptr));
if (TYPE_IS_THIN_POINTER_P (TREE_TYPE (gnu_ptr)))
gnu_ptr
= build_binary_op (POINTER_PLUS_EXPR, TREE_TYPE (gnu_ptr),
gnu_ptr,
fold_build1 (NEGATE_EXPR, sizetype,
byte_position
(DECL_CHAIN
TYPE_FIELDS ((gnu_obj_type)))));
if (Present (Actual_Designated_Subtype (gnat_node)))
{
gnu_actual_obj_type
= gnat_to_gnu_type (Actual_Designated_Subtype (gnat_node));
if (TYPE_IS_FAT_OR_THIN_POINTER_P (gnu_ptr_type))
gnu_actual_obj_type
= build_unc_object_type_from_ptr (gnu_ptr_type,
gnu_actual_obj_type,
get_identifier ("DEALLOC"),
false);
}
else
gnu_actual_obj_type = gnu_obj_type;
tree gnu_size = TYPE_SIZE_UNIT (gnu_actual_obj_type);
gnu_size = SUBSTITUTE_PLACEHOLDER_IN_EXPR (gnu_size, gnu_ptr);
gnu_result
= build_call_alloc_dealloc (gnu_ptr, gnu_size, gnu_obj_type,
Procedure_To_Call (gnat_node),
Storage_Pool (gnat_node),
gnat_node);
}
break;
case N_Raise_Constraint_Error:
case N_Raise_Program_Error:
case N_Raise_Storage_Error:
if (type_annotate_only)
gnu_result = alloc_stmt_list ();
else
gnu_result = Raise_Error_to_gnu (gnat_node, &gnu_result_type);
break;
case N_Validate_Unchecked_Conversion:
if (flag_strict_aliasing)
gnat_validate_uc_list.safe_push (gnat_node);
gnu_result = alloc_stmt_list ();
break;
case N_Function_Specification:
case N_Procedure_Specification:
case N_Op_Concat:
case N_Component_Association:
gcc_assert (type_annotate_only);
gnu_result = alloc_stmt_list ();
break;
default:
gcc_unreachable ();
}
if (went_into_elab_proc)
current_function_decl = NULL_TREE;
if (!optimize
&& TREE_CODE (gnu_result) != INTEGER_CST
&& TREE_CODE (gnu_result) != TYPE_DECL
&& (kind == N_Identifier
|| kind == N_Expanded_Name
|| kind == N_Explicit_Dereference
|| kind == N_Indexed_Component
|| kind == N_Selected_Component)
&& TREE_CODE (get_base_type (gnu_result_type)) == BOOLEAN_TYPE
&& !lvalue_required_p (gnat_node, gnu_result_type, false, false, false))
{
gnu_result
= build_binary_op (NE_EXPR, gnu_result_type,
convert (gnu_result_type, gnu_result),
convert (gnu_result_type, boolean_false_node));
if (TREE_CODE (gnu_result) != INTEGER_CST)
set_gnu_expr_location_from_node (gnu_result, gnat_node);
}
else if (kind != N_Identifier && gnu_result && EXPR_P (gnu_result))
set_gnu_expr_location_from_node (gnu_result, gnat_node);
if (TREE_CODE (gnu_result_type) == VOID_TYPE)
return gnu_result;
if (TREE_CODE (gnu_result) == INTEGER_CST && TREE_OVERFLOW (gnu_result))
{
post_error ("?`Constraint_Error` will be raised at run time", gnat_node);
gnu_result
= build1 (NULL_EXPR, gnu_result_type,
build_call_raise (CE_Overflow_Check_Failed, gnat_node,
N_Raise_Constraint_Error));
}
if (TREE_SIDE_EFFECTS (gnu_result)
&& (TREE_CODE (gnu_result_type) == UNCONSTRAINED_ARRAY_TYPE
|| CONTAINS_PLACEHOLDER_P (TYPE_SIZE (gnu_result_type)))
&& !(TREE_CODE (gnu_result) == CALL_EXPR
&& type_is_padding_self_referential (TREE_TYPE (gnu_result))
&& (Nkind (Parent (gnat_node)) == N_Object_Declaration
|| Nkind (Parent (gnat_node)) == N_Object_Renaming_Declaration
|| Nkind (Parent (gnat_node)) == N_Simple_Return_Statement)))
gnu_result = gnat_protect_expr (gnu_result);
if (Present (Parent (gnat_node))
&& (lhs_or_actual_p (gnat_node)
|| (Nkind (Parent (gnat_node)) == N_Unchecked_Type_Conversion
&& unchecked_conversion_nop (Parent (gnat_node)))
|| (Nkind (Parent (gnat_node)) == N_Unchecked_Type_Conversion
&& !AGGREGATE_TYPE_P (gnu_result_type)
&& !AGGREGATE_TYPE_P (TREE_TYPE (gnu_result))))
&& !(TYPE_SIZE (gnu_result_type)
&& TYPE_SIZE (TREE_TYPE (gnu_result))
&& (AGGREGATE_TYPE_P (gnu_result_type)
== AGGREGATE_TYPE_P (TREE_TYPE (gnu_result)))
&& ((TREE_CODE (TYPE_SIZE (gnu_result_type)) == INTEGER_CST
&& (TREE_CODE (TYPE_SIZE (TREE_TYPE (gnu_result)))
!= INTEGER_CST))
|| (TREE_CODE (TYPE_SIZE (gnu_result_type)) != INTEGER_CST
&& !CONTAINS_PLACEHOLDER_P (TYPE_SIZE (gnu_result_type))
&& (CONTAINS_PLACEHOLDER_P
(TYPE_SIZE (TREE_TYPE (gnu_result))))))
&& !(TREE_CODE (gnu_result_type) == RECORD_TYPE
&& TYPE_JUSTIFIED_MODULAR_P (gnu_result_type))))
{
if (type_is_padding_self_referential (TREE_TYPE (gnu_result)))
gnu_result = convert (TREE_TYPE (TYPE_FIELDS (TREE_TYPE (gnu_result))),
gnu_result);
}
else if (TREE_CODE (gnu_result) == LABEL_DECL
|| TREE_CODE (gnu_result) == FIELD_DECL
|| TREE_CODE (gnu_result) == ERROR_MARK
|| (TYPE_NAME (gnu_result_type)
== TYPE_NAME (TREE_TYPE (gnu_result))
&& TREE_CODE (gnu_result_type) == RECORD_TYPE
&& TREE_CODE (TREE_TYPE (gnu_result)) == RECORD_TYPE))
{
if (TYPE_IS_PADDING_P (TREE_TYPE (gnu_result)))
gnu_result = convert (TREE_TYPE (TYPE_FIELDS (TREE_TYPE (gnu_result))),
gnu_result);
}
else if (gnu_result == error_mark_node || gnu_result_type == void_type_node)
gnu_result = error_mark_node;
else if (Present (Parent (gnat_node))
&& (Nkind (Parent (gnat_node)) == N_Object_Declaration
|| Nkind (Parent (gnat_node)) == N_Object_Renaming_Declaration)
&& TREE_CODE (gnu_result) == CALL_EXPR
&& return_type_with_variable_size_p (TREE_TYPE (gnu_result)))
;
else if (TREE_TYPE (gnu_result) != gnu_result_type)
gnu_result = convert (gnu_result_type, gnu_result);
while ((TREE_CODE (gnu_result) == NOP_EXPR
|| TREE_CODE (gnu_result) == NON_LVALUE_EXPR)
&& TREE_TYPE (TREE_OPERAND (gnu_result, 0)) == TREE_TYPE (gnu_result))
gnu_result = TREE_OPERAND (gnu_result, 0);
return gnu_result;
}
tree
gnat_to_gnu_external (Node_Id gnat_node)
{
const int save_force_global = force_global;
bool went_into_elab_proc = false;
if (!current_function_decl)
{
current_function_decl = get_elaboration_procedure ();
went_into_elab_proc = true;
}
force_global = 0;
gnat_pushlevel ();
tree gnu_result = gnat_to_gnu (gnat_node);
gnat_zaplevel ();
force_global = save_force_global;
if (went_into_elab_proc)
current_function_decl = NULL_TREE;
if (gnu_result && EXPR_P (gnu_result))
SET_EXPR_LOCATION (gnu_result, UNKNOWN_LOCATION);
return gnu_result;
}

static bool
empty_stmt_list_p (tree stmt_list)
{
tree_stmt_iterator tsi;
for (tsi = tsi_start (stmt_list); !tsi_end_p (tsi); tsi_next (&tsi))
{
tree stmt = tsi_stmt (tsi);
if (TREE_CODE (stmt) != STMT_STMT || STMT_STMT_STMT (stmt))
return false;
}
return true;
}
static void
record_code_position (Node_Id gnat_node)
{
tree stmt_stmt = build1 (STMT_STMT, void_type_node, NULL_TREE);
add_stmt_with_node (stmt_stmt, gnat_node);
save_gnu_tree (gnat_node, stmt_stmt, true);
}
static void
insert_code_for (Node_Id gnat_node)
{
tree code = gnat_to_gnu (gnat_node);
if (!empty_stmt_list_p (code))
STMT_STMT_STMT (get_gnu_tree (gnat_node)) = code;
save_gnu_tree (gnat_node, NULL_TREE, true);
}

void
start_stmt_group (void)
{
struct stmt_group *group = stmt_group_free_list;
if (group)
stmt_group_free_list = group->previous;
else
group = ggc_alloc<stmt_group> ();
group->previous = current_stmt_group;
group->stmt_list = group->block = group->cleanups = NULL_TREE;
current_stmt_group = group;
}
void
add_stmt (tree gnu_stmt)
{
append_to_statement_list (gnu_stmt, &current_stmt_group->stmt_list);
}
void
add_stmt_force (tree gnu_stmt)
{
append_to_statement_list_force (gnu_stmt, &current_stmt_group->stmt_list);
}
void
add_stmt_with_node (tree gnu_stmt, Node_Id gnat_node)
{
if (Present (gnat_node) && !renaming_from_instantiation_p (gnat_node))
set_expr_location_from_node (gnu_stmt, gnat_node);
add_stmt (gnu_stmt);
}
void
add_stmt_with_node_force (tree gnu_stmt, Node_Id gnat_node)
{
if (Present (gnat_node))
set_expr_location_from_node (gnu_stmt, gnat_node);
add_stmt_force (gnu_stmt);
}
void
add_decl_expr (tree gnu_decl, Entity_Id gnat_entity)
{
tree type = TREE_TYPE (gnu_decl);
tree gnu_stmt, gnu_init;
if (!DECL_P (gnu_decl)
|| (TREE_CODE (gnu_decl) == TYPE_DECL
&& TREE_CODE (type) == UNCONSTRAINED_ARRAY_TYPE))
return;
gnu_stmt = build1 (DECL_EXPR, void_type_node, gnu_decl);
if (DECL_EXTERNAL (gnu_decl) || global_bindings_p ())
{
MARK_VISITED (gnu_stmt);
if (TREE_CODE (gnu_decl) == VAR_DECL
|| TREE_CODE (gnu_decl) == CONST_DECL)
{
MARK_VISITED (DECL_SIZE (gnu_decl));
MARK_VISITED (DECL_SIZE_UNIT (gnu_decl));
MARK_VISITED (DECL_INITIAL (gnu_decl));
}
else if (TREE_CODE (gnu_decl) == TYPE_DECL
&& RECORD_OR_UNION_TYPE_P (type)
&& !TYPE_FAT_POINTER_P (type))
MARK_VISITED (TYPE_ADA_SIZE (type));
}
else
add_stmt_with_node (gnu_stmt, gnat_entity);
if (TREE_CODE (gnu_decl) == VAR_DECL
&& (gnu_init = DECL_INITIAL (gnu_decl))
&& (!gnat_types_compatible_p (type, TREE_TYPE (gnu_init))
|| (TREE_STATIC (gnu_decl)
&& !initializer_constant_valid_p (gnu_init,
TREE_TYPE (gnu_init)))))
{
DECL_INITIAL (gnu_decl) = NULL_TREE;
if (TREE_READONLY (gnu_decl))
{
TREE_READONLY (gnu_decl) = 0;
DECL_READONLY_ONCE_ELAB (gnu_decl) = 1;
}
if (TYPE_IS_PADDING_P (type))
gnu_decl = convert (TREE_TYPE (TYPE_FIELDS (type)), gnu_decl);
gnu_stmt = build_binary_op (INIT_EXPR, NULL_TREE, gnu_decl, gnu_init);
add_stmt_with_node (gnu_stmt, gnat_entity);
}
}
static tree
mark_visited_r (tree *tp, int *walk_subtrees, void *data ATTRIBUTE_UNUSED)
{
tree t = *tp;
if (TREE_VISITED (t))
*walk_subtrees = 0;
else if (!TYPE_IS_DUMMY_P (t))
TREE_VISITED (t) = 1;
if (TYPE_P (t))
TYPE_SIZES_GIMPLIFIED (t) = 1;
return NULL_TREE;
}
void
mark_visited (tree t)
{
walk_tree (&t, mark_visited_r, NULL, NULL);
}
static void
add_cleanup (tree gnu_cleanup, Node_Id gnat_node)
{
if (Present (gnat_node))
set_expr_location_from_node (gnu_cleanup, gnat_node, true);
append_to_statement_list (gnu_cleanup, &current_stmt_group->cleanups);
}
void
set_block_for_group (tree gnu_block)
{
gcc_assert (!current_stmt_group->block);
current_stmt_group->block = gnu_block;
}
tree
end_stmt_group (void)
{
struct stmt_group *group = current_stmt_group;
tree gnu_retval = group->stmt_list;
if (!gnu_retval)
gnu_retval = alloc_stmt_list ();
if (group->cleanups)
gnu_retval = build2 (TRY_FINALLY_EXPR, void_type_node, gnu_retval,
group->cleanups);
if (current_stmt_group->block)
gnu_retval = build3 (BIND_EXPR, void_type_node, BLOCK_VARS (group->block),
gnu_retval, group->block);
current_stmt_group = group->previous;
group->previous = stmt_group_free_list;
stmt_group_free_list = group;
return gnu_retval;
}
static inline bool
stmt_group_may_fallthru (void)
{
if (current_stmt_group->stmt_list)
return block_may_fallthru (current_stmt_group->stmt_list);
else
return true;
}
static void
add_stmt_list (List_Id gnat_list)
{
Node_Id gnat_node;
if (Present (gnat_list))
for (gnat_node = First (gnat_list); Present (gnat_node);
gnat_node = Next (gnat_node))
add_stmt (gnat_to_gnu (gnat_node));
}
static tree
build_stmt_group (List_Id gnat_list, bool binding_p)
{
start_stmt_group ();
if (binding_p)
gnat_pushlevel ();
add_stmt_list (gnat_list);
if (binding_p)
gnat_poplevel ();
return end_stmt_group ();
}

int
gnat_gimplify_expr (tree *expr_p, gimple_seq *pre_p,
gimple_seq *post_p ATTRIBUTE_UNUSED)
{
tree expr = *expr_p;
tree type = TREE_TYPE (expr);
tree op;
if (IS_ADA_STMT (expr))
return gnat_gimplify_stmt (expr_p);
switch (TREE_CODE (expr))
{
case NULL_EXPR:
if (AGGREGATE_TYPE_P (type)
|| TREE_CODE (type) == UNCONSTRAINED_ARRAY_TYPE)
*expr_p = build_unary_op (INDIRECT_REF, NULL_TREE,
convert (build_pointer_type (type),
integer_zero_node));
else
{
*expr_p = create_tmp_var (type, NULL);
TREE_NO_WARNING (*expr_p) = 1;
}
gimplify_and_add (TREE_OPERAND (expr, 0), pre_p);
return GS_OK;
case UNCONSTRAINED_ARRAY_REF:
*expr_p = TREE_OPERAND (*expr_p, 0);
return GS_OK;
case ADDR_EXPR:
op = TREE_OPERAND (expr, 0);
if (TREE_CODE (op) == CONSTRUCTOR && TREE_CONSTANT (op))
{
tree addr = build_fold_addr_expr (tree_output_constant_def (op));
*expr_p = fold_convert (type, addr);
return GS_ALL_DONE;
}
if (TREE_SIDE_EFFECTS (op))
while (handled_component_p (op) || CONVERT_EXPR_P (op))
{
tree inner = TREE_OPERAND (op, 0);
if (TREE_CODE (inner) == CALL_EXPR && call_is_atomic_load (inner))
{
tree t = CALL_EXPR_ARG (inner, 0);
if (TREE_CODE (t) == NOP_EXPR)
t = TREE_OPERAND (t, 0);
if (TREE_CODE (t) == ADDR_EXPR)
TREE_OPERAND (op, 0) = TREE_OPERAND (t, 0);
else
TREE_OPERAND (op, 0) = build_fold_indirect_ref (t);
}
else
op = inner;
}
return GS_UNHANDLED;
case VIEW_CONVERT_EXPR:
op = TREE_OPERAND (expr, 0);
if ((TREE_CODE (op) == CONSTRUCTOR || TREE_CODE (op) == CALL_EXPR)
&& AGGREGATE_TYPE_P (TREE_TYPE (op))
&& !AGGREGATE_TYPE_P (type))
{
tree mod, new_var = create_tmp_var_raw (TREE_TYPE (op), "C");
gimple_add_tmp_var (new_var);
mod = build2 (INIT_EXPR, TREE_TYPE (new_var), new_var, op);
gimplify_and_add (mod, pre_p);
TREE_OPERAND (expr, 0) = new_var;
return GS_OK;
}
return GS_UNHANDLED;
case DECL_EXPR:
op = DECL_EXPR_DECL (expr);
if ((TREE_CODE (op) == TYPE_DECL || TREE_CODE (op) == VAR_DECL)
&& !TYPE_SIZES_GIMPLIFIED (TREE_TYPE (op)))
switch (TREE_CODE (TREE_TYPE (op)))
{
case INTEGER_TYPE:
case ENUMERAL_TYPE:
case BOOLEAN_TYPE:
case REAL_TYPE:
{
tree type = TYPE_MAIN_VARIANT (TREE_TYPE (op)), t, val;
val = TYPE_RM_MIN_VALUE (type);
if (val)
{
gimplify_one_sizepos (&val, pre_p);
for (t = type; t; t = TYPE_NEXT_VARIANT (t))
SET_TYPE_RM_MIN_VALUE (t, val);
}
val = TYPE_RM_MAX_VALUE (type);
if (val)
{
gimplify_one_sizepos (&val, pre_p);
for (t = type; t; t = TYPE_NEXT_VARIANT (t))
SET_TYPE_RM_MAX_VALUE (t, val);
}
}
break;
default:
break;
}
default:
return GS_UNHANDLED;
}
}
static enum gimplify_status
gnat_gimplify_stmt (tree *stmt_p)
{
tree stmt = *stmt_p;
switch (TREE_CODE (stmt))
{
case STMT_STMT:
*stmt_p = STMT_STMT_STMT (stmt);
return GS_OK;
case LOOP_STMT:
{
tree gnu_start_label = create_artificial_label (input_location);
tree gnu_cond = LOOP_STMT_COND (stmt);
tree gnu_update = LOOP_STMT_UPDATE (stmt);
tree gnu_end_label = LOOP_STMT_LABEL (stmt);
if (gnu_cond)
{
if (LOOP_STMT_IVDEP (stmt))
gnu_cond = build3 (ANNOTATE_EXPR, TREE_TYPE (gnu_cond), gnu_cond,
build_int_cst (integer_type_node,
annot_expr_ivdep_kind),
integer_zero_node);
if (LOOP_STMT_NO_UNROLL (stmt))
gnu_cond = build3 (ANNOTATE_EXPR, TREE_TYPE (gnu_cond), gnu_cond,
build_int_cst (integer_type_node,
annot_expr_unroll_kind),
integer_one_node);
if (LOOP_STMT_UNROLL (stmt))
gnu_cond = build3 (ANNOTATE_EXPR, TREE_TYPE (gnu_cond), gnu_cond,
build_int_cst (integer_type_node,
annot_expr_unroll_kind),
build_int_cst (NULL_TREE, USHRT_MAX));
if (LOOP_STMT_NO_VECTOR (stmt))
gnu_cond = build3 (ANNOTATE_EXPR, TREE_TYPE (gnu_cond), gnu_cond,
build_int_cst (integer_type_node,
annot_expr_no_vector_kind),
integer_zero_node);
if (LOOP_STMT_VECTOR (stmt))
gnu_cond = build3 (ANNOTATE_EXPR, TREE_TYPE (gnu_cond), gnu_cond,
build_int_cst (integer_type_node,
annot_expr_vector_kind),
integer_zero_node);
gnu_cond
= build3 (COND_EXPR, void_type_node, gnu_cond, NULL_TREE,
build1 (GOTO_EXPR, void_type_node, gnu_end_label));
}
*stmt_p = NULL_TREE;
append_to_statement_list (build1 (LABEL_EXPR, void_type_node,
gnu_start_label),
stmt_p);
if (gnu_cond && !LOOP_STMT_BOTTOM_COND_P (stmt))
append_to_statement_list (gnu_cond, stmt_p);
if (gnu_update && LOOP_STMT_TOP_UPDATE_P (stmt))
append_to_statement_list (gnu_update, stmt_p);
append_to_statement_list (LOOP_STMT_BODY (stmt), stmt_p);
if (gnu_cond && LOOP_STMT_BOTTOM_COND_P (stmt))
append_to_statement_list (gnu_cond, stmt_p);
if (gnu_update && !LOOP_STMT_TOP_UPDATE_P (stmt))
append_to_statement_list (gnu_update, stmt_p);
tree t = build1 (GOTO_EXPR, void_type_node, gnu_start_label);
SET_EXPR_LOCATION (t, DECL_SOURCE_LOCATION (gnu_end_label));
append_to_statement_list (t, stmt_p);
append_to_statement_list (build1 (LABEL_EXPR, void_type_node,
gnu_end_label),
stmt_p);
return GS_OK;
}
case EXIT_STMT:
*stmt_p = build1 (GOTO_EXPR, void_type_node, EXIT_STMT_LABEL (stmt));
if (EXIT_STMT_COND (stmt))
*stmt_p = build3 (COND_EXPR, void_type_node,
EXIT_STMT_COND (stmt), *stmt_p, alloc_stmt_list ());
return GS_OK;
default:
gcc_unreachable ();
}
}

static void
elaborate_all_entities_for_package (Entity_Id gnat_package)
{
Entity_Id gnat_entity;
for (gnat_entity = First_Entity (gnat_package);
Present (gnat_entity);
gnat_entity = Next_Entity (gnat_entity))
{
const Entity_Kind kind = Ekind (gnat_entity);
if (!Is_Public (gnat_entity))
continue;
if (Convention (gnat_entity) == Convention_Intrinsic)
continue;
if (kind == E_Operator)
continue;
if (IN (kind, Subprogram_Kind) && Is_Intrinsic_Subprogram (gnat_entity))
continue;
if (Is_Itype (gnat_entity))
continue;
if (IN (kind, Named_Kind))
continue;
if (IN (kind, Generic_Unit_Kind))
continue;
if (IN (kind, Formal_Object_Kind))
continue;
if (kind == E_Package_Body)
continue;
if (IN (kind, Incomplete_Kind)
&& From_Limited_With (gnat_entity)
&& In_Extended_Main_Code_Unit (Non_Limited_View (gnat_entity)))
continue;
if (IN (kind, Type_Kind) && !Is_Frozen (gnat_entity))
continue;
if (kind == E_Package)
{
if (No (Renamed_Entity (gnat_entity))
&& !In_Extended_Main_Code_Unit (gnat_entity))
elaborate_all_entities_for_package (gnat_entity);
}
else
gnat_to_gnu_entity (gnat_entity, NULL_TREE, false);
}
}
static void
elaborate_all_entities (Node_Id gnat_node)
{
Entity_Id gnat_with_clause;
if (!present_gnu_tree (gnat_node))
save_gnu_tree (gnat_node, integer_zero_node, true);
for (gnat_with_clause = First (Context_Items (gnat_node));
Present (gnat_with_clause);
gnat_with_clause = Next (gnat_with_clause))
if (Nkind (gnat_with_clause) == N_With_Clause
&& !present_gnu_tree (Library_Unit (gnat_with_clause))
&& Library_Unit (gnat_with_clause) != Library_Unit (Cunit (Main_Unit)))
{
Node_Id gnat_unit = Library_Unit (gnat_with_clause);
Entity_Id gnat_entity = Entity (Name (gnat_with_clause));
elaborate_all_entities (gnat_unit);
if (Ekind (gnat_entity) == E_Package)
elaborate_all_entities_for_package (gnat_entity);
else if (Ekind (gnat_entity) == E_Generic_Package)
{
Node_Id gnat_body = Corresponding_Body (Unit (gnat_unit));
while (Present (gnat_body)
&& Nkind (gnat_body) != N_Compilation_Unit)
gnat_body = Parent (gnat_body);
if (Present (gnat_body))
elaborate_all_entities (gnat_body);
}
}
if (Nkind (Unit (gnat_node)) == N_Package_Body)
elaborate_all_entities (Library_Unit (gnat_node));
}

static void
process_freeze_entity (Node_Id gnat_node)
{
const Entity_Id gnat_entity = Entity (gnat_node);
const Entity_Kind kind = Ekind (gnat_entity);
tree gnu_old, gnu_new;
if (kind == E_Package)
{
const Node_Id gnat_decl = Parent (Declaration_Node (gnat_entity));
if (Present (Corresponding_Body (gnat_decl)))
insert_code_for (Parent (Corresponding_Body (gnat_decl)));
return;
}
if (kind == E_Class_Wide_Type)
return;
gnu_old
= present_gnu_tree (gnat_entity) && No (Address_Clause (gnat_entity))
? get_gnu_tree (gnat_entity) : NULL_TREE;
if (gnu_old
&& ((TREE_CODE (gnu_old) == FUNCTION_DECL
&& (kind == E_Function || kind == E_Procedure))
|| (TREE_CODE (TREE_TYPE (gnu_old)) == FUNCTION_TYPE
&& kind == E_Subprogram_Type)))
return;
if (gnu_old
&& !(TREE_CODE (gnu_old) == TYPE_DECL
&& TYPE_IS_DUMMY_P (TREE_TYPE (gnu_old))))
{
gcc_assert (Is_Concurrent_Type (gnat_entity)
|| (Is_Record_Type (gnat_entity)
&& Is_Concurrent_Record_Type (gnat_entity)));
return;
}
if (gnu_old)
{
save_gnu_tree (gnat_entity, NULL_TREE, false);
if (Is_Incomplete_Or_Private_Type (gnat_entity)
&& Present (Full_View (gnat_entity)))
{
Entity_Id full_view = Full_View (gnat_entity);
save_gnu_tree (full_view, NULL_TREE, false);
if (Is_Private_Type (full_view)
&& Present (Underlying_Full_View (full_view)))
{
full_view = Underlying_Full_View (full_view);
save_gnu_tree (full_view, NULL_TREE, false);
}
}
if (Is_Type (gnat_entity)
&& Present (Class_Wide_Type (gnat_entity))
&& Root_Type (Class_Wide_Type (gnat_entity)) == gnat_entity)
save_gnu_tree (Class_Wide_Type (gnat_entity), NULL_TREE, false);
}
if (Is_Incomplete_Or_Private_Type (gnat_entity)
&& Present (Full_View (gnat_entity)))
{
Entity_Id full_view = Full_View (gnat_entity);
if (Is_Private_Type (full_view)
&& Present (Underlying_Full_View (full_view)))
full_view = Underlying_Full_View (full_view);
gnu_new = gnat_to_gnu_entity (full_view, NULL_TREE, true);
if (Unknown_Alignment (gnat_entity))
Set_Alignment (gnat_entity, Alignment (full_view));
if (Unknown_Esize (gnat_entity))
Set_Esize (gnat_entity, Esize (full_view));
if (Unknown_RM_Size (gnat_entity))
Set_RM_Size (gnat_entity, RM_Size (full_view));
if (!present_gnu_tree (gnat_entity))
save_gnu_tree (gnat_entity, gnu_new, false);
}
else
{
tree gnu_init
= (Nkind (Declaration_Node (gnat_entity)) == N_Object_Declaration
&& present_gnu_tree (Declaration_Node (gnat_entity)))
? get_gnu_tree (Declaration_Node (gnat_entity)) : NULL_TREE;
gnu_new = gnat_to_gnu_entity (gnat_entity, gnu_init, true);
}
if (Is_Type (gnat_entity)
&& Present (Class_Wide_Type (gnat_entity))
&& Root_Type (Class_Wide_Type (gnat_entity)) == gnat_entity)
save_gnu_tree (Class_Wide_Type (gnat_entity), gnu_new, false);
if (gnu_old)
{
update_pointer_to (TYPE_MAIN_VARIANT (TREE_TYPE (gnu_old)),
TREE_TYPE (gnu_new));
if (TYPE_DUMMY_IN_PROFILE_P (TREE_TYPE (gnu_old)))
update_profiles_with (TREE_TYPE (gnu_old));
if (DECL_TAFT_TYPE_P (gnu_old))
used_types_insert (TREE_TYPE (gnu_new));
}
}

static void
process_decls (List_Id gnat_decls, List_Id gnat_decls2,
Node_Id gnat_end_list, bool pass1p, bool pass2p)
{
List_Id gnat_decl_array[2];
Node_Id gnat_decl;
int i;
gnat_decl_array[0] = gnat_decls, gnat_decl_array[1] = gnat_decls2;
if (pass1p)
for (i = 0; i <= 1; i++)
if (Present (gnat_decl_array[i]))
for (gnat_decl = First (gnat_decl_array[i]);
gnat_decl != gnat_end_list; gnat_decl = Next (gnat_decl))
{
if (Nkind (gnat_decl) == N_Package_Declaration
&& (Nkind (Specification (gnat_decl)
== N_Package_Specification)))
process_decls (Visible_Declarations (Specification (gnat_decl)),
Private_Declarations (Specification (gnat_decl)),
Empty, true, false);
else if (Nkind (gnat_decl) == N_Freeze_Entity)
{
process_freeze_entity (gnat_decl);
process_decls (Actions (gnat_decl), Empty, Empty, true, false);
}
else if (Nkind (gnat_decl) == N_Package_Body
&& Present (Freeze_Node (Corresponding_Spec (gnat_decl))))
record_code_position (gnat_decl);
else if (Nkind (gnat_decl) == N_Package_Body_Stub
&& Present (Library_Unit (gnat_decl))
&& Present (Freeze_Node
(Corresponding_Spec
(Proper_Body (Unit
(Library_Unit (gnat_decl)))))))
record_code_position
(Proper_Body (Unit (Library_Unit (gnat_decl))));
else if (Nkind (gnat_decl) == N_Subprogram_Body)
{
if (Acts_As_Spec (gnat_decl))
{
Node_Id gnat_subprog_id = Defining_Entity (gnat_decl);
if (Ekind (gnat_subprog_id) != E_Generic_Procedure
&& Ekind (gnat_subprog_id) != E_Generic_Function)
gnat_to_gnu_entity (gnat_subprog_id, NULL_TREE, true);
}
}
else if (Nkind (gnat_decl) == N_Subprogram_Body_Stub)
{
Node_Id gnat_subprog_id
= Defining_Entity (Specification (gnat_decl));
if (Ekind (gnat_subprog_id) != E_Subprogram_Body
&& Ekind (gnat_subprog_id) != E_Generic_Procedure
&& Ekind (gnat_subprog_id) != E_Generic_Function)
gnat_to_gnu_entity (gnat_subprog_id, NULL_TREE, true);
}
else if (Nkind (gnat_decl) == N_Task_Body_Stub
|| Nkind (gnat_decl) == N_Protected_Body_Stub)
;
else if (Nkind (gnat_decl) == N_Subprogram_Renaming_Declaration)
;
else
add_stmt (gnat_to_gnu (gnat_decl));
}
if (pass2p)
for (i = 0; i <= 1; i++)
if (Present (gnat_decl_array[i]))
for (gnat_decl = First (gnat_decl_array[i]);
gnat_decl != gnat_end_list; gnat_decl = Next (gnat_decl))
{
if (Nkind (gnat_decl) == N_Subprogram_Body
|| Nkind (gnat_decl) == N_Subprogram_Body_Stub
|| Nkind (gnat_decl) == N_Task_Body_Stub
|| Nkind (gnat_decl) == N_Protected_Body_Stub)
add_stmt (gnat_to_gnu (gnat_decl));
else if (Nkind (gnat_decl) == N_Package_Declaration
&& (Nkind (Specification (gnat_decl)
== N_Package_Specification)))
process_decls (Visible_Declarations (Specification (gnat_decl)),
Private_Declarations (Specification (gnat_decl)),
Empty, false, true);
else if (Nkind (gnat_decl) == N_Freeze_Entity)
process_decls (Actions (gnat_decl), Empty, Empty, false, true);
else if (Nkind (gnat_decl) == N_Subprogram_Renaming_Declaration)
add_stmt (gnat_to_gnu (gnat_decl));
}
}

static tree
build_unary_op_trapv (enum tree_code code, tree gnu_type, tree operand,
Node_Id gnat_node)
{
gcc_assert (code == NEGATE_EXPR || code == ABS_EXPR);
operand = gnat_protect_expr (operand);
return emit_check (build_binary_op (EQ_EXPR, boolean_type_node,
operand, TYPE_MIN_VALUE (gnu_type)),
build_unary_op (code, gnu_type, operand),
CE_Overflow_Check_Failed, gnat_node);
}
static tree
build_binary_op_trapv (enum tree_code code, tree gnu_type, tree left,
tree right, Node_Id gnat_node)
{
const unsigned int precision = TYPE_PRECISION (gnu_type);
tree lhs = gnat_protect_expr (left);
tree rhs = gnat_protect_expr (right);
tree type_max = TYPE_MAX_VALUE (gnu_type);
tree type_min = TYPE_MIN_VALUE (gnu_type);
tree gnu_expr, check;
int sgn;
gcc_assert ((precision & (precision - 1)) == 0);
if (TREE_CODE (rhs) != INTEGER_CST
&& TREE_CODE (lhs) == INTEGER_CST
&& (code == PLUS_EXPR || code == MULT_EXPR))
{
tree tmp = lhs;
lhs = rhs;
rhs = tmp;
}
gnu_expr = build_binary_op (code, gnu_type, lhs, rhs);
if (TREE_CODE (gnu_expr) == INTEGER_CST)
return gnu_expr;
if (TREE_CODE (lhs) != INTEGER_CST && TREE_CODE (rhs) != INTEGER_CST)
{
if (code == MULT_EXPR && precision == 64 && BITS_PER_WORD < 64)
{
tree int64 = gnat_type_for_size (64, 0);
return convert (gnu_type, build_call_n_expr (mulv64_decl, 2,
convert (int64, lhs),
convert (int64, rhs)));
}
enum internal_fn icode;
switch (code)
{
case PLUS_EXPR:
icode = IFN_ADD_OVERFLOW;
break;
case MINUS_EXPR:
icode = IFN_SUB_OVERFLOW;
break;
case MULT_EXPR:
icode = IFN_MUL_OVERFLOW;
break;
default:
gcc_unreachable ();
}
tree gnu_ctype = build_complex_type (gnu_type);
tree call
= build_call_expr_internal_loc (UNKNOWN_LOCATION, icode, gnu_ctype, 2,
lhs, rhs);
tree tgt = save_expr (call);
gnu_expr = build1 (REALPART_EXPR, gnu_type, tgt);
check = fold_build2 (NE_EXPR, boolean_type_node,
build1 (IMAGPART_EXPR, gnu_type, tgt),
build_int_cst (gnu_type, 0));
return
emit_check (check, gnu_expr, CE_Overflow_Check_Failed, gnat_node);
}
switch (code)
{
case PLUS_EXPR:
sgn = tree_int_cst_sgn (rhs);
if (sgn > 0)
check = build_binary_op (GT_EXPR, boolean_type_node, lhs,
build_binary_op (MINUS_EXPR, gnu_type,
type_max, rhs));
else if (sgn < 0)
check = build_binary_op (LT_EXPR, boolean_type_node, lhs,
build_binary_op (MINUS_EXPR, gnu_type,
type_min, rhs));
else
return gnu_expr;
break;
case MINUS_EXPR:
if (TREE_CODE (lhs) == INTEGER_CST)
{
sgn = tree_int_cst_sgn (lhs);
if (sgn > 0)
check = build_binary_op (LT_EXPR, boolean_type_node, rhs,
build_binary_op (MINUS_EXPR, gnu_type,
lhs, type_max));
else if (sgn < 0)
check = build_binary_op (GT_EXPR, boolean_type_node, rhs,
build_binary_op (MINUS_EXPR, gnu_type,
lhs, type_min));
else
return gnu_expr;
}
else
{
sgn = tree_int_cst_sgn (rhs);
if (sgn > 0)
check = build_binary_op (LT_EXPR, boolean_type_node, lhs,
build_binary_op (PLUS_EXPR, gnu_type,
type_min, rhs));
else if (sgn < 0)
check = build_binary_op (GT_EXPR, boolean_type_node, lhs,
build_binary_op (PLUS_EXPR, gnu_type,
type_max, rhs));
else
return gnu_expr;
}
break;
case MULT_EXPR:
sgn = tree_int_cst_sgn (rhs);
if (sgn > 0)
{
if (integer_onep (rhs))
return gnu_expr;
tree lb = build_binary_op (TRUNC_DIV_EXPR, gnu_type, type_min, rhs);
tree ub = build_binary_op (TRUNC_DIV_EXPR, gnu_type, type_max, rhs);
check
= build_binary_op (TRUTH_ORIF_EXPR, boolean_type_node,
build_binary_op (LT_EXPR, boolean_type_node,
lhs, lb),
build_binary_op (GT_EXPR, boolean_type_node,
lhs, ub));
}
else if (sgn < 0)
{
tree lb = build_binary_op (TRUNC_DIV_EXPR, gnu_type, type_max, rhs);
tree ub = build_binary_op (TRUNC_DIV_EXPR, gnu_type, type_min, rhs);
if (integer_minus_onep (rhs))
check
= build_binary_op (EQ_EXPR, boolean_type_node, lhs, type_min);
else
check
= build_binary_op (TRUTH_ORIF_EXPR, boolean_type_node,
build_binary_op (LT_EXPR, boolean_type_node,
lhs, lb),
build_binary_op (GT_EXPR, boolean_type_node,
lhs, ub));
}
else
return gnu_expr;
break;
default:
gcc_unreachable ();
}
return emit_check (check, gnu_expr, CE_Overflow_Check_Failed, gnat_node);
}
static tree
emit_range_check (tree gnu_expr, Entity_Id gnat_range_type, Node_Id gnat_node)
{
tree gnu_range_type = get_unpadded_type (gnat_range_type);
tree gnu_compare_type = get_base_type (TREE_TYPE (gnu_expr));
if (gnu_compare_type == gnu_range_type)
return gnu_expr;
gcc_assert (INTEGRAL_TYPE_P (gnu_range_type)
|| SCALAR_FLOAT_TYPE_P (gnu_range_type));
if (INTEGRAL_TYPE_P (TREE_TYPE (gnu_expr))
&& (TYPE_PRECISION (gnu_compare_type)
< TYPE_PRECISION (get_base_type (gnu_range_type))))
return gnu_expr;
gnu_expr = gnat_protect_expr (gnu_expr);
return emit_check
(build_binary_op (TRUTH_ORIF_EXPR, boolean_type_node,
invert_truthvalue
(build_binary_op (GE_EXPR, boolean_type_node,
convert (gnu_compare_type, gnu_expr),
convert (gnu_compare_type,
TYPE_MIN_VALUE
(gnu_range_type)))),
invert_truthvalue
(build_binary_op (LE_EXPR, boolean_type_node,
convert (gnu_compare_type, gnu_expr),
convert (gnu_compare_type,
TYPE_MAX_VALUE
(gnu_range_type))))),
gnu_expr, CE_Range_Check_Failed, gnat_node);
}

static tree
emit_check (tree gnu_cond, tree gnu_expr, int reason, Node_Id gnat_node)
{
tree gnu_call
= build_call_raise (reason, gnat_node, N_Raise_Constraint_Error);
return
fold_build3 (COND_EXPR, TREE_TYPE (gnu_expr), gnu_cond,
build2 (COMPOUND_EXPR, TREE_TYPE (gnu_expr), gnu_call,
SCALAR_FLOAT_TYPE_P (TREE_TYPE (gnu_expr))
? build_real (TREE_TYPE (gnu_expr), dconst0)
: build_int_cst (TREE_TYPE (gnu_expr), 0)),
gnu_expr);
}

static tree
convert_with_check (Entity_Id gnat_type, tree gnu_expr, bool overflow_p,
bool range_p, bool truncate_p, Node_Id gnat_node)
{
tree gnu_type = get_unpadded_type (gnat_type);
tree gnu_base_type = get_base_type (gnu_type);
tree gnu_in_type = TREE_TYPE (gnu_expr);
tree gnu_in_base_type = get_base_type (gnu_in_type);
tree gnu_result = gnu_expr;
if (!range_p
&& !overflow_p
&& INTEGRAL_TYPE_P (gnu_base_type)
&& !FLOAT_TYPE_P (gnu_in_base_type))
return convert (gnu_type, gnu_expr);
if (TYPE_MODE (gnu_in_base_type) != TYPE_MODE (gnu_in_type)
&& !(TREE_CODE (gnu_in_type) == INTEGER_TYPE
&& TYPE_BIASED_REPRESENTATION_P (gnu_in_type)))
gnu_in_base_type = gnat_type_for_mode (TYPE_MODE (gnu_in_type),
TYPE_UNSIGNED (gnu_in_type));
if (TREE_CODE (gnu_type) != UNCONSTRAINED_ARRAY_TYPE)
gnu_result = convert (gnu_in_base_type, gnu_result);
if (overflow_p
&& !(FLOAT_TYPE_P (gnu_base_type) && INTEGRAL_TYPE_P (gnu_in_base_type)))
{
tree gnu_input = gnat_protect_expr (gnu_result);
tree gnu_cond = boolean_false_node;
tree gnu_in_lb = TYPE_MIN_VALUE (gnu_in_base_type);
tree gnu_in_ub = TYPE_MAX_VALUE (gnu_in_base_type);
tree gnu_out_lb = TYPE_MIN_VALUE (gnu_base_type);
tree gnu_out_ub = TYPE_MAX_VALUE (gnu_base_type);
if (INTEGRAL_TYPE_P (gnu_in_base_type)
&& TYPE_UNSIGNED (gnu_in_base_type))
gnu_in_lb
= convert (gnat_signed_type_for (gnu_in_base_type), gnu_in_lb);
if (INTEGRAL_TYPE_P (gnu_in_base_type)
&& !TYPE_UNSIGNED (gnu_in_base_type))
gnu_in_ub
= convert (gnat_unsigned_type_for (gnu_in_base_type), gnu_in_ub);
if (INTEGRAL_TYPE_P (gnu_base_type) && TYPE_UNSIGNED (gnu_base_type))
gnu_out_lb
= convert (gnat_signed_type_for (gnu_base_type), gnu_out_lb);
if (INTEGRAL_TYPE_P (gnu_base_type) && !TYPE_UNSIGNED (gnu_base_type))
gnu_out_ub
= convert (gnat_unsigned_type_for (gnu_base_type), gnu_out_ub);
if (INTEGRAL_TYPE_P (gnu_in_base_type)
? tree_int_cst_lt (gnu_in_lb, gnu_out_lb)
: (FLOAT_TYPE_P (gnu_base_type)
? real_less (&TREE_REAL_CST (gnu_in_lb),
&TREE_REAL_CST (gnu_out_lb))
: 1))
gnu_cond
= invert_truthvalue
(build_binary_op (GE_EXPR, boolean_type_node,
gnu_input, convert (gnu_in_base_type,
gnu_out_lb)));
if (INTEGRAL_TYPE_P (gnu_in_base_type)
? tree_int_cst_lt (gnu_out_ub, gnu_in_ub)
: (FLOAT_TYPE_P (gnu_base_type)
? real_less (&TREE_REAL_CST (gnu_out_ub),
&TREE_REAL_CST (gnu_in_ub))
: 1))
gnu_cond
= build_binary_op (TRUTH_ORIF_EXPR, boolean_type_node, gnu_cond,
invert_truthvalue
(build_binary_op (LE_EXPR, boolean_type_node,
gnu_input,
convert (gnu_in_base_type,
gnu_out_ub))));
if (!integer_zerop (gnu_cond))
gnu_result = emit_check (gnu_cond, gnu_input,
CE_Overflow_Check_Failed, gnat_node);
}
if (INTEGRAL_TYPE_P (gnu_base_type)
&& FLOAT_TYPE_P (gnu_in_base_type)
&& !truncate_p)
{
REAL_VALUE_TYPE half_minus_pred_half, pred_half;
tree gnu_conv, gnu_zero, gnu_comp, calc_type;
tree gnu_pred_half, gnu_add_pred_half, gnu_subtract_pred_half;
const struct real_format *fmt;
calc_type
= fp_arith_may_widen ? longest_float_type_node : gnu_in_base_type;
fmt = REAL_MODE_FORMAT (TYPE_MODE (calc_type));
real_2expN (&half_minus_pred_half, -(fmt->p) - 1, TYPE_MODE (calc_type));
real_arithmetic (&pred_half, MINUS_EXPR, &dconsthalf,
&half_minus_pred_half);
gnu_pred_half = build_real (calc_type, pred_half);
gnu_zero = build_real (gnu_in_base_type, dconst0);
gnu_result = gnat_protect_expr (gnu_result);
gnu_conv = convert (calc_type, gnu_result);
gnu_comp
= fold_build2 (GE_EXPR, boolean_type_node, gnu_result, gnu_zero);
gnu_add_pred_half
= fold_build2 (PLUS_EXPR, calc_type, gnu_conv, gnu_pred_half);
gnu_subtract_pred_half
= fold_build2 (MINUS_EXPR, calc_type, gnu_conv, gnu_pred_half);
gnu_result = fold_build3 (COND_EXPR, calc_type, gnu_comp,
gnu_add_pred_half, gnu_subtract_pred_half);
}
if (TREE_CODE (gnu_base_type) == INTEGER_TYPE
&& TYPE_HAS_ACTUAL_BOUNDS_P (gnu_base_type)
&& TREE_CODE (gnu_result) == UNCONSTRAINED_ARRAY_REF)
gnu_result = unchecked_convert (gnu_base_type, gnu_result, false);
else
gnu_result = convert (gnu_base_type, gnu_result);
if (range_p
|| (overflow_p
&& TREE_CODE (gnu_base_type) == INTEGER_TYPE
&& TYPE_MODULAR_P (gnu_base_type)))
gnu_result = emit_range_check (gnu_result, gnat_type, gnat_node);
return convert (gnu_type, gnu_result);
}

static bool
addressable_p (tree gnu_expr, tree gnu_type)
{
if (gnu_type
&& INTEGRAL_TYPE_P (gnu_type)
&& smaller_form_type_p (gnu_type, TREE_TYPE (gnu_expr)))
return false;
if (gnu_type
&& TREE_CODE (gnu_type) == RECORD_TYPE
&& smaller_form_type_p (TREE_TYPE (gnu_expr), gnu_type))
return false;
switch (TREE_CODE (gnu_expr))
{
case VAR_DECL:
case PARM_DECL:
case FUNCTION_DECL:
case RESULT_DECL:
return true;
case UNCONSTRAINED_ARRAY_REF:
case INDIRECT_REF:
return true;
case STRING_CST:
case INTEGER_CST:
return true;
case CONSTRUCTOR:
return TREE_STATIC (gnu_expr) ? true : false;
case NULL_EXPR:
case SAVE_EXPR:
case CALL_EXPR:
case PLUS_EXPR:
case MINUS_EXPR:
case BIT_IOR_EXPR:
case BIT_XOR_EXPR:
case BIT_AND_EXPR:
case BIT_NOT_EXPR:
return true;
case COMPOUND_EXPR:
return addressable_p (TREE_OPERAND (gnu_expr, 1), gnu_type);
case COND_EXPR:
return (addressable_p (TREE_OPERAND (gnu_expr, 1), NULL_TREE)
&& addressable_p (TREE_OPERAND (gnu_expr, 2), NULL_TREE));
case COMPONENT_REF:
return (((!DECL_BIT_FIELD (TREE_OPERAND (gnu_expr, 1))
&& (!STRICT_ALIGNMENT
|| DECL_ALIGN (TREE_OPERAND (gnu_expr, 1))
>= TYPE_ALIGN (TREE_TYPE (gnu_expr))))
|| TYPE_IS_PADDING_P (TREE_TYPE (TREE_OPERAND (gnu_expr, 0))))
&& addressable_p (TREE_OPERAND (gnu_expr, 0), NULL_TREE));
case ARRAY_REF:  case ARRAY_RANGE_REF:
case REALPART_EXPR:  case IMAGPART_EXPR:
case NOP_EXPR:
return addressable_p (TREE_OPERAND (gnu_expr, 0), NULL_TREE);
case CONVERT_EXPR:
return (AGGREGATE_TYPE_P (TREE_TYPE (gnu_expr))
&& addressable_p (TREE_OPERAND (gnu_expr, 0), NULL_TREE));
case VIEW_CONVERT_EXPR:
{
tree type = TREE_TYPE (gnu_expr);
tree inner_type = TREE_TYPE (TREE_OPERAND (gnu_expr, 0));
return (((TYPE_MODE (type) == TYPE_MODE (inner_type)
&& (!STRICT_ALIGNMENT
|| TYPE_ALIGN (type) <= TYPE_ALIGN (inner_type)
|| TYPE_ALIGN (inner_type) >= BIGGEST_ALIGNMENT))
|| ((TYPE_MODE (type) == BLKmode
|| TYPE_MODE (inner_type) == BLKmode)
&& (!STRICT_ALIGNMENT
|| TYPE_ALIGN (type) <= TYPE_ALIGN (inner_type)
|| TYPE_ALIGN (inner_type) >= BIGGEST_ALIGNMENT
|| TYPE_ALIGN_OK (type)
|| TYPE_ALIGN_OK (inner_type))))
&& addressable_p (TREE_OPERAND (gnu_expr, 0), NULL_TREE));
}
default:
return false;
}
}

void
process_type (Entity_Id gnat_entity)
{
tree gnu_old
= present_gnu_tree (gnat_entity) ? get_gnu_tree (gnat_entity) : NULL_TREE;
if (Present (Freeze_Node (gnat_entity)))
{
elaborate_entity (gnat_entity);
if (!gnu_old)
{
tree gnu_decl = TYPE_STUB_DECL (make_dummy_type (gnat_entity));
save_gnu_tree (gnat_entity, gnu_decl, false);
if (Is_Incomplete_Or_Private_Type (gnat_entity)
&& Present (Full_View (gnat_entity)))
{
if (Has_Completion_In_Body (gnat_entity))
DECL_TAFT_TYPE_P (gnu_decl) = 1;
save_gnu_tree (Full_View (gnat_entity), gnu_decl, false);
}
}
return;
}
if (gnu_old)
{
gcc_assert (TREE_CODE (gnu_old) == TYPE_DECL
&& TYPE_IS_DUMMY_P (TREE_TYPE (gnu_old)));
save_gnu_tree (gnat_entity, NULL_TREE, false);
}
tree gnu_new = gnat_to_gnu_entity (gnat_entity, NULL_TREE, true);
gcc_assert (TREE_CODE (gnu_new) == TYPE_DECL);
if (gnu_old)
{
update_pointer_to (TYPE_MAIN_VARIANT (TREE_TYPE (gnu_old)),
TREE_TYPE (gnu_new));
if (DECL_TAFT_TYPE_P (gnu_old))
used_types_insert (TREE_TYPE (gnu_new));
}
if (Is_Record_Type (gnat_entity)
&& Is_Concurrent_Record_Type (gnat_entity)
&& present_gnu_tree (Corresponding_Concurrent_Type (gnat_entity)))
{
tree gnu_task_old
= get_gnu_tree (Corresponding_Concurrent_Type (gnat_entity));
save_gnu_tree (Corresponding_Concurrent_Type (gnat_entity),
NULL_TREE, false);
save_gnu_tree (Corresponding_Concurrent_Type (gnat_entity),
gnu_new, false);
update_pointer_to (TYPE_MAIN_VARIANT (TREE_TYPE (gnu_task_old)),
TREE_TYPE (gnu_new));
}
}

static tree
extract_values (tree values, tree record_type)
{
vec<constructor_elt, va_gc> *v = NULL;
tree field;
for (field = TYPE_FIELDS (record_type); field; field = DECL_CHAIN (field))
{
tree tem, value = NULL_TREE;
if ((tem = purpose_member (field, values)))
{
value = TREE_VALUE (tem);
TREE_ADDRESSABLE (tem) = 1;
}
else if (DECL_INTERNAL_P (field))
{
value = extract_values (values, TREE_TYPE (field));
if (TREE_CODE (value) == CONSTRUCTOR
&& vec_safe_is_empty (CONSTRUCTOR_ELTS (value)))
value = NULL_TREE;
}
else
for (tem = values; tem; tem = TREE_CHAIN (tem))
if (DECL_NAME (TREE_PURPOSE (tem)) == DECL_NAME (field))
{
value = convert (TREE_TYPE (field), TREE_VALUE (tem));
TREE_ADDRESSABLE (tem) = 1;
}
if (!value)
continue;
CONSTRUCTOR_APPEND_ELT (v, field, value);
}
return gnat_build_constructor (record_type, v);
}
static tree
assoc_to_constructor (Entity_Id gnat_entity, Node_Id gnat_assoc, tree gnu_type)
{
tree gnu_list = NULL_TREE, gnu_result;
for (; Present (gnat_assoc); gnat_assoc = Next (gnat_assoc))
{
Node_Id gnat_field = First (Choices (gnat_assoc));
tree gnu_field = gnat_to_gnu_field_decl (Entity (gnat_field));
tree gnu_expr = gnat_to_gnu (Expression (gnat_assoc));
gcc_assert (No (Next (gnat_field)));
if (Ekind (Entity (gnat_field)) == E_Discriminant
&& Present (Corresponding_Discriminant (Entity (gnat_field)))
&& Is_Tagged_Type (Scope (Entity (gnat_field))))
continue;
if (Ekind (Entity (gnat_field)) == E_Discriminant
&& Is_Unchecked_Union (gnat_entity))
continue;
if (Do_Range_Check (Expression (gnat_assoc)))
gnu_expr = emit_range_check (gnu_expr, Etype (gnat_field), Empty);
gnu_expr = convert (TREE_TYPE (gnu_field), gnu_expr);
gnu_list = tree_cons (gnu_field, gnu_expr, gnu_list);
}
gnu_result = extract_values (gnu_list, gnu_type);
if (flag_checking)
{
for (; gnu_list; gnu_list = TREE_CHAIN (gnu_list))
gcc_assert (TREE_ADDRESSABLE (gnu_list));
}
return gnu_result;
}
static tree
pos_to_constructor (Node_Id gnat_expr, tree gnu_array_type,
Entity_Id gnat_component_type)
{
tree gnu_index = TYPE_MIN_VALUE (TYPE_DOMAIN (gnu_array_type));
vec<constructor_elt, va_gc> *gnu_expr_vec = NULL;
for (; Present (gnat_expr); gnat_expr = Next (gnat_expr))
{
tree gnu_expr;
if (Nkind (gnat_expr) == N_Aggregate
&& TREE_CODE (TREE_TYPE (gnu_array_type)) == ARRAY_TYPE
&& TYPE_MULTI_ARRAY_P (TREE_TYPE (gnu_array_type)))
gnu_expr = pos_to_constructor (First (Expressions (gnat_expr)),
TREE_TYPE (gnu_array_type),
gnat_component_type);
else
{
if (Nkind (gnat_expr) == N_Type_Conversion
&& Is_Array_Type (Etype (gnat_expr))
&& !Is_Constrained (Etype (gnat_expr)))
gnu_expr = gnat_to_gnu (Expression (gnat_expr));
else
gnu_expr = gnat_to_gnu (gnat_expr);
if (Do_Range_Check (gnat_expr))
gnu_expr = emit_range_check (gnu_expr, gnat_component_type, Empty);
}
CONSTRUCTOR_APPEND_ELT (gnu_expr_vec, gnu_index,
convert (TREE_TYPE (gnu_array_type), gnu_expr));
gnu_index = int_const_binop (PLUS_EXPR, gnu_index,
convert (TREE_TYPE (gnu_index),
integer_one_node));
}
return gnat_build_constructor (gnu_array_type, gnu_expr_vec);
}

static void
validate_unchecked_conversion (Node_Id gnat_node)
{
tree gnu_source_type = gnat_to_gnu_type (Source_Type (gnat_node));
tree gnu_target_type = gnat_to_gnu_type (Target_Type (gnat_node));
if (POINTER_TYPE_P (gnu_target_type)
&& !TYPE_REF_CAN_ALIAS_ALL (gnu_target_type))
{
tree gnu_source_desig_type = POINTER_TYPE_P (gnu_source_type)
? TREE_TYPE (gnu_source_type)
: NULL_TREE;
tree gnu_target_desig_type = TREE_TYPE (gnu_target_type);
alias_set_type target_alias_set = get_alias_set (gnu_target_desig_type);
if (target_alias_set != 0
&& (!POINTER_TYPE_P (gnu_source_type)
|| !alias_sets_conflict_p (get_alias_set (gnu_source_desig_type),
target_alias_set)))
{
post_error_ne ("?possible aliasing problem for type&",
gnat_node, Target_Type (gnat_node));
post_error ("\\?use -fno-strict-aliasing switch for references",
gnat_node);
post_error_ne ("\\?or use `pragma No_Strict_Aliasing (&);`",
gnat_node, Target_Type (gnat_node));
}
}
else if (TYPE_IS_FAT_POINTER_P (gnu_target_type))
{
tree gnu_source_desig_type
= TYPE_IS_FAT_POINTER_P (gnu_source_type)
? TREE_TYPE (TREE_TYPE (TYPE_FIELDS (gnu_source_type)))
: NULL_TREE;
tree gnu_target_desig_type
= TREE_TYPE (TREE_TYPE (TYPE_FIELDS (gnu_target_type)));
alias_set_type target_alias_set = get_alias_set (gnu_target_desig_type);
if (target_alias_set != 0
&& (!TYPE_IS_FAT_POINTER_P (gnu_source_type)
|| !alias_sets_conflict_p (get_alias_set (gnu_source_desig_type),
target_alias_set)))
{
post_error_ne ("?possible aliasing problem for type&",
gnat_node, Target_Type (gnat_node));
post_error ("\\?use -fno-strict-aliasing switch for references",
gnat_node);
}
}
}

static Node_Id
adjust_for_implicit_deref (Node_Id exp)
{
Entity_Id type = Underlying_Type (Etype (exp));
if (Is_Access_Type (type))
gnat_to_gnu_entity (Designated_Type (type), NULL_TREE, false);
return exp;
}
static tree
maybe_implicit_deref (tree exp)
{
if (POINTER_TYPE_P (TREE_TYPE (exp))
|| TYPE_IS_FAT_POINTER_P (TREE_TYPE (exp)))
exp = build_unary_op (INDIRECT_REF, NULL_TREE, exp);
if (TYPE_IS_PADDING_P (TREE_TYPE (exp)))
exp = convert (TREE_TYPE (TYPE_FIELDS (TREE_TYPE (exp))), exp);
return exp;
}

bool
Sloc_to_locus (Source_Ptr Sloc, location_t *locus, bool clear_column)
{
if (Sloc == No_Location)
return false;
if (Sloc <= Standard_Location)
{
*locus = BUILTINS_LOCATION;
return false;
}
Source_File_Index file = Get_Source_File_Index (Sloc);
Logical_Line_Number line = Get_Logical_Line_Number (Sloc);
Column_Number column = (clear_column ? 0 : Get_Column_Number (Sloc));
line_map_ordinary *map = LINEMAPS_ORDINARY_MAP_AT (line_table, file - 1);
if (line < 1)
line = 1;
*locus
= linemap_position_for_line_and_column (line_table, map, line, column);
return true;
}
static void
set_expr_location_from_node (tree node, Node_Id gnat_node, bool clear_column)
{
location_t locus;
if (!Sloc_to_locus (Sloc (gnat_node), &locus, clear_column))
return;
SET_EXPR_LOCATION (node, locus);
}
static void
set_gnu_expr_location_from_node (tree node, Node_Id gnat_node)
{
switch (TREE_CODE (node))
{
CASE_CONVERT:
case NON_LVALUE_EXPR:
case SAVE_EXPR:
break;
case COMPOUND_EXPR:
if (EXPR_P (TREE_OPERAND (node, 1)))
set_gnu_expr_location_from_node (TREE_OPERAND (node, 1), gnat_node);
default:
if (!REFERENCE_CLASS_P (node) && !EXPR_HAS_LOCATION (node))
{
set_expr_location_from_node (node, gnat_node);
set_end_locus_from_node (node, gnat_node);
}
break;
}
}
static bool
set_end_locus_from_node (tree gnu_node, Node_Id gnat_node)
{
Node_Id gnat_end_label;
location_t end_locus;
switch (Nkind (gnat_node))
{
case N_Package_Body:
case N_Subprogram_Body:
case N_Block_Statement:
gnat_end_label = End_Label (Handled_Statement_Sequence (gnat_node));
break;
case N_Package_Declaration:
gnat_end_label = End_Label (Specification (gnat_node));
break;
default:
return false;
}
if (Present (gnat_end_label))
gnat_node = gnat_end_label;
if (!Sloc_to_locus (Sloc (gnat_node), &end_locus,
No (gnat_end_label)
&& (Nkind (gnat_node) == N_Block_Statement)))
return false;
switch (TREE_CODE (gnu_node))
{
case BIND_EXPR:
BLOCK_SOURCE_END_LOCATION (BIND_EXPR_BLOCK (gnu_node)) = end_locus;
return true;
case FUNCTION_DECL:
DECL_STRUCT_FUNCTION (gnu_node)->function_end_locus = end_locus;
return true;
default:
return false;
}
}

static const char *
extract_encoding (const char *name)
{
char *encoding = (char *) ggc_alloc_atomic (strlen (name));
get_encoding (name, encoding);
return encoding;
}
static const char *
decode_name (const char *name)
{
char *decoded = (char *) ggc_alloc_atomic (strlen (name) * 2 + 60);
__gnat_decode (name, decoded, 0);
return decoded;
}

void
post_error (const char *msg, Node_Id node)
{
String_Template temp;
String_Pointer sp;
if (No (node))
return;
temp.Low_Bound = 1;
temp.High_Bound = strlen (msg);
sp.Bounds = &temp;
sp.Array = msg;
Error_Msg_N (sp, node);
}
void
post_error_ne (const char *msg, Node_Id node, Entity_Id ent)
{
String_Template temp;
String_Pointer sp;
if (No (node))
return;
temp.Low_Bound = 1;
temp.High_Bound = strlen (msg);
sp.Bounds = &temp;
sp.Array = msg;
Error_Msg_NE (sp, node, ent);
}
void
post_error_ne_num (const char *msg, Node_Id node, Entity_Id ent, int num)
{
Error_Msg_Uint_1 = UI_From_Int (num);
post_error_ne (msg, node, ent);
}
void
post_error_ne_tree (const char *msg, Node_Id node, Entity_Id ent, tree t)
{
char *new_msg = XALLOCAVEC (char, strlen (msg) + 1);
char start_yes, end_yes, start_no, end_no;
const char *p;
char *q;
if (TREE_CODE (t) == INTEGER_CST)
{
Error_Msg_Uint_1 = UI_From_gnu (t);
start_yes = '{', end_yes = '}', start_no = '[', end_no = ']';
}
else
start_yes = '[', end_yes = ']', start_no = '{', end_no = '}';
for (p = msg, q = new_msg; *p; p++)
{
if (*p == start_yes)
for (p++; *p != end_yes; p++)
*q++ = *p;
else if (*p == start_no)
for (p++; *p != end_no; p++)
;
else
*q++ = *p;
}
*q = 0;
post_error_ne (new_msg, node, ent);
}
void
post_error_ne_tree_2 (const char *msg, Node_Id node, Entity_Id ent, tree t,
int num)
{
Error_Msg_Uint_2 = UI_From_Int (num);
post_error_ne_tree (msg, node, ent, t);
}
Entity_Id
get_exception_label (char kind)
{
switch (kind)
{
case N_Raise_Constraint_Error:
return gnu_constraint_error_label_stack.last ();
case N_Raise_Storage_Error:
return gnu_storage_error_label_stack.last ();
case N_Raise_Program_Error:
return gnu_program_error_label_stack.last ();
default:
return Empty;
}
gcc_unreachable ();
}
static tree
get_elaboration_procedure (void)
{
return gnu_elab_proc_stack->last ();
}
static void
init_code_table (void)
{
gnu_codes[N_Op_And] = TRUTH_AND_EXPR;
gnu_codes[N_Op_Or] = TRUTH_OR_EXPR;
gnu_codes[N_Op_Xor] = TRUTH_XOR_EXPR;
gnu_codes[N_Op_Eq] = EQ_EXPR;
gnu_codes[N_Op_Ne] = NE_EXPR;
gnu_codes[N_Op_Lt] = LT_EXPR;
gnu_codes[N_Op_Le] = LE_EXPR;
gnu_codes[N_Op_Gt] = GT_EXPR;
gnu_codes[N_Op_Ge] = GE_EXPR;
gnu_codes[N_Op_Add] = PLUS_EXPR;
gnu_codes[N_Op_Subtract] = MINUS_EXPR;
gnu_codes[N_Op_Multiply] = MULT_EXPR;
gnu_codes[N_Op_Mod] = FLOOR_MOD_EXPR;
gnu_codes[N_Op_Rem] = TRUNC_MOD_EXPR;
gnu_codes[N_Op_Minus] = NEGATE_EXPR;
gnu_codes[N_Op_Abs] = ABS_EXPR;
gnu_codes[N_Op_Not] = TRUTH_NOT_EXPR;
gnu_codes[N_Op_Rotate_Left] = LROTATE_EXPR;
gnu_codes[N_Op_Rotate_Right] = RROTATE_EXPR;
gnu_codes[N_Op_Shift_Left] = LSHIFT_EXPR;
gnu_codes[N_Op_Shift_Right] = RSHIFT_EXPR;
gnu_codes[N_Op_Shift_Right_Arithmetic] = RSHIFT_EXPR;
gnu_codes[N_And_Then] = TRUTH_ANDIF_EXPR;
gnu_codes[N_Or_Else] = TRUTH_ORIF_EXPR;
}
#include "gt-ada-trans.h"
