#include "fortran03-buildscope.h"
#include "fortran03-scope.h"
#include "fortran03-exprtype.h"
#include "fortran03-prettyprint.h"
#include "fortran03-typeutils.h"
#include "fortran03-intrinsics.h"
#include "fortran03-modules.h"
#include "fortran03-codegen.h"
#include "cxx-ast.h"
#include "cxx-scope.h"
#include "cxx-buildscope.h"
#include "cxx-utils.h"
#include "cxx-entrylist.h"
#include "cxx-typeutils.h"
#include "cxx-tltype.h"
#include "cxx-exprtype.h"
#include "cxx-ambiguity.h"
#include "cxx-limits.h"
#include "cxx-nodecl.h"
#include "cxx-nodecl-output.h"
#include "cxx-pragma.h"
#include "cxx-diagnostic.h"
#include "cxx-placeholders.h"
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "red_black_tree.h"
typedef
enum build_scope_delay_category_tag
{
DELAY_AFTER_USE_STATEMENT,
DELAY_AFTER_IMPLICIT_STATEMENT,
DELAY_AFTER_DECLARATIONS,
DELAY_AFTER_EXECUTABLE_STATEMENTS,
DELAY_AFTER_PROGRAM_UNIT,
DELAY_NUM_CATEGORIES,
} build_scope_delay_category_t;
typedef void build_scope_delay_fun_t(void*, nodecl_t*);
typedef
struct build_scope_delay_info_tag
{
build_scope_delay_fun_t* fun;
void *data;
} build_scope_delay_info_t;
typedef
struct build_scope_single_category_list_tag
{
int num_delayed;
build_scope_delay_info_t* list;
} build_scope_single_category_list_t;
typedef
struct build_scope_delay_tag
{
build_scope_single_category_list_t categories[DELAY_NUM_CATEGORIES];
} build_scope_delay_list_t;
enum { BUILD_SCOPE_DELAY_STACK_MAX = 16 };
int _current_delay_stack_idx = 0;
static build_scope_delay_list_t* _current_delay_stack[BUILD_SCOPE_DELAY_STACK_MAX];
static void build_scope_delay_list_push(build_scope_delay_list_t* delay_list)
{
ERROR_CONDITION(_current_delay_stack_idx == BUILD_SCOPE_DELAY_STACK_MAX, "Too many delayed scopes", 0);
_current_delay_stack[_current_delay_stack_idx] = delay_list;
_current_delay_stack_idx++;
}
static void build_scope_delay_list_pop(void)
{
ERROR_CONDITION(_current_delay_stack_idx == 0, "Empty stack", 0);
_current_delay_stack[_current_delay_stack_idx - 1] = NULL;
_current_delay_stack_idx--;
}
static build_scope_delay_list_t* build_scope_delay_list_current(void)
{
ERROR_CONDITION(_current_delay_stack_idx == 0, "Empty stack", 0);
return _current_delay_stack[_current_delay_stack_idx - 1];
}
static void build_scope_delay_list_run(
build_scope_delay_category_t delay_category,
nodecl_t *nodecl_output)
{
build_scope_delay_list_t *delay_list = build_scope_delay_list_current();
while (delay_list->categories[delay_category].num_delayed != 0)
{
int i;
int num_delayed = delay_list->categories[delay_category].num_delayed;
build_scope_delay_info_t *copy_delayed
= NEW_VEC0(build_scope_delay_info_t, num_delayed);
memcpy(copy_delayed,
delay_list->categories[delay_category].list,
sizeof(*copy_delayed) * num_delayed);
DELETE(delay_list->categories[delay_category].list);
delay_list->categories[delay_category].num_delayed = 0;
delay_list->categories[delay_category].list = NULL;
for (i = 0; i < num_delayed; i++)
{
nodecl_t nodecl_current = nodecl_null();
(copy_delayed[i].fun)(copy_delayed[i].data, &nodecl_current);
if (nodecl_output != NULL)
{
*nodecl_output
= nodecl_concat_lists(*nodecl_output, nodecl_current);
}
else if (!nodecl_is_null(nodecl_current))
{
internal_error(
"Delayed action generates nodecl but there is no output "
"nodecl\n",
0);
}
}
DELETE(copy_delayed);
}
}
static void build_scope_delay_list_add(
build_scope_delay_category_t delay_category,
build_scope_delay_fun_t *fun,
void *data)
{
if (_current_delay_stack_idx == 0)
return;
build_scope_delay_info_t new_delayed = { fun, data };
build_scope_delay_list_t *current_delay_list
= _current_delay_stack[_current_delay_stack_idx - 1];
P_LIST_ADD(current_delay_list->categories[delay_category].list,
current_delay_list->categories[delay_category].num_delayed,
new_delayed);
}
typedef char build_scope_delay_list_cmp_fun_t(void *key, void *data);
static void build_scope_delay_list_run_now(
build_scope_delay_category_t delay_category,
void *key,
build_scope_delay_fun_t *fun,
build_scope_delay_list_cmp_fun_t *cmp_fun,
nodecl_t *nodecl_output)
{
ERROR_CONDITION(_current_delay_stack_idx == 0, "No active delay list", 0);
build_scope_delay_list_t *current_delay_list
= _current_delay_stack[_current_delay_stack_idx - 1];
char found = 0;
int i;
for (i = 0; i < current_delay_list->categories[delay_category].num_delayed
&& !found;
i++)
{
if (fun == current_delay_list->categories[delay_category].list[i].fun
&& cmp_fun(
key,
current_delay_list->categories[delay_category].list[i].data))
{
(current_delay_list->categories[delay_category].list[i].fun)(
current_delay_list->categories[delay_category].list[i].data,
nodecl_output);
current_delay_list->categories[delay_category].num_delayed--;
for (;
i < current_delay_list->categories[delay_category].num_delayed;
i++)
{
current_delay_list->categories[delay_category].list[i]
= current_delay_list->categories[delay_category]
.list[i + 1];
}
found = 1;
}
}
ERROR_CONDITION(!found, "Delayed element not found", 0);
}
static char build_scope_delay_list_remove(
build_scope_delay_category_t delay_category,
void *key,
build_scope_delay_fun_t *fun,
build_scope_delay_list_cmp_fun_t *cmp_fun)
{
ERROR_CONDITION(_current_delay_stack_idx == 0, "No active delay list", 0);
build_scope_delay_list_t *current_delay_list
= _current_delay_stack[_current_delay_stack_idx - 1];
char found = 0;
int i;
for (i = 0; i < current_delay_list->categories[delay_category].num_delayed
&& !found;
i++)
{
if (fun == current_delay_list->categories[delay_category].list[i].fun
&& cmp_fun(
key,
current_delay_list->categories[delay_category].list[i].data))
{
current_delay_list->categories[delay_category].num_delayed--;
for (;
i < current_delay_list->categories[delay_category].num_delayed;
i++)
{
current_delay_list->categories[delay_category].list[i]
= current_delay_list->categories[delay_category]
.list[i + 1];
}
found = 1;
}
}
return found;
}
static void unsupported_statement(AST a, const char* name);
static void null_dtor(const void* p UNUSED_PARAMETER) { }
static void fortran_init_globals(const decl_context_t* decl_context);
void fortran_initialize_translation_unit_scope(translation_unit_t* translation_unit)
{
const decl_context_t* decl_context;
CURRENT_CONFIGURATION->source_language = SOURCE_LANGUAGE_C;
{
initialize_translation_unit_scope(translation_unit, &decl_context);
c_initialize_builtin_symbols(decl_context);
}
CURRENT_CONFIGURATION->source_language = SOURCE_LANGUAGE_FORTRAN;
translation_unit->module_file_cache = rb_tree_create((int (*)(const void*, const void*))strcasecmp, null_dtor, null_dtor);
fortran_init_kinds();
fortran_init_globals(decl_context);
fortran_init_intrinsics(decl_context);
}
static void fortran_init_globals(const decl_context_t* decl_context)
{
type_t* int_8 = fortran_choose_int_type_from_kind(1);
type_t* int_16 = fortran_choose_int_type_from_kind(2);
type_t* int_32 = fortran_choose_int_type_from_kind(4);
type_t* int_64 = fortran_choose_int_type_from_kind(8);
struct {
const char* symbol_name;
type_t* backing_type;
int explicit_size;
} intrinsic_globals[] =
{
{ "mercurium_c_int", get_signed_int_type(), 0},
{ "mercurium_c_short", get_signed_short_int_type(), 0},
{ "mercurium_c_long", get_signed_long_int_type(), 0},
{ "mercurium_c_long_long", get_signed_long_long_int_type(), 0},
{ "mercurium_c_signed_char", get_signed_byte_type(), 0},
{ "mercurium_c_size_t", get_size_t_type(), 0},
{ "mercurium_c_int8_t", int_8, 0},
{ "mercurium_c_int16_t", int_16, 0},
{ "mercurium_c_int32_t", int_32, 0},
{ "mercurium_c_int64_t", int_64, 0},
{ "mercurium_c_int_least8_t", int_8, 0},
{ "mercurium_c_int_least16_t", int_16, 0},
{ "mercurium_c_int_least32_t", int_32, 0},
{ "mercurium_c_int_least64_t", int_64, 0},
{ "mercurium_c_int_fast8_t", int_8, 0},
{ "mercurium_c_int_fast16_t", int_16, 0},
{ "mercurium_c_int_fast32_t", int_32, 0},
{ "mercurium_c_int_fast64_t", int_64, 0},
{ "mercurium_c_intmax_t", int_64, 0},
{ "mercurium_c_intptr_t", get_pointer_type(get_void_type()), 0},
{ "mercurium_c_float", get_float_type(), 0},
{ "mercurium_c_double", get_double_type(), 0},
{ "mercurium_c_long_double", get_long_double_type(), 0},
{ "mercurium_c_float_complex", get_float_type(), 0},
{ "mercurium_c_double_complex", get_double_type(), 0},
{ "mercurium_c_long_double_complex", get_long_double_type(), 0},
{ "mercurium_c_bool", get_bool_type(), 0},
{ "mercurium_c_char", get_char_type(), 0},
{ "mercurium_c_ptr", NULL, CURRENT_CONFIGURATION->type_environment->sizeof_pointer },
{ "mercurium_c_funptr", NULL, CURRENT_CONFIGURATION->type_environment->sizeof_function_pointer },
{ NULL, NULL, 0}
};
int i;
for (i = 0; intrinsic_globals[i].symbol_name != NULL; i++)
{
scope_entry_t* mercurium_intptr = new_symbol(decl_context, decl_context->global_scope,
uniquestr(intrinsic_globals[i].symbol_name));
mercurium_intptr->kind = SK_VARIABLE;
mercurium_intptr->type_information = get_const_qualified_type(fortran_get_default_integer_type());
_size_t size = 0;
if (intrinsic_globals[i].backing_type != NULL)
{
size = type_get_size(intrinsic_globals[i].backing_type);
}
else
{
size = intrinsic_globals[i].explicit_size;
}
mercurium_intptr->value = const_value_to_nodecl(const_value_get_signed_int(size));
mercurium_intptr->do_not_print = 1;
}
}
static void resolve_external_calls_inside_file(nodecl_t nodecl_program_units);
nodecl_t build_scope_fortran_translation_unit(translation_unit_t* translation_unit)
{
AST a = translation_unit->parsed_tree;
const decl_context_t* decl_context = translation_unit->global_decl_context;
nodecl_t nodecl_program_units = nodecl_null();
AST list = ASTSon0(a);
if (list != NULL)
{
build_scope_program_unit_seq(list, decl_context, &nodecl_program_units);
}
if (!CURRENT_CONFIGURATION->fortran_no_whole_file)
{
resolve_external_calls_inside_file(nodecl_program_units);
}
return nodecl_program_units;
}
static void build_scope_program_unit_internal(AST program_unit,
const decl_context_t* decl_context,
scope_entry_t** program_unit_symbol,
nodecl_t* nodecl_output);
void build_scope_program_unit_seq(AST program_unit_seq,
const decl_context_t* decl_context,
nodecl_t* nodecl_output)
{
AST it;
for_each_element(program_unit_seq, it)
{
nodecl_t nodecl_top_level_items = nodecl_null();
build_scope_program_unit_internal(ASTSon1(it), 
decl_context, 
NULL, &nodecl_top_level_items);
*nodecl_output = nodecl_concat_lists(*nodecl_output,
nodecl_top_level_items);
}
}
static scope_entry_t* get_special_symbol(const decl_context_t* decl_context, const char *name)
{
ERROR_CONDITION(name == NULL || name[0] != '.', "Name '%s' is not special enough\n", name);
decl_context_t* global_context = decl_context_clone(decl_context);
global_context->current_scope = global_context->function_scope;
scope_entry_list_t* entry_list = query_in_scope_str_flags(global_context, name, NULL, DF_ONLY_CURRENT_SCOPE);
if (entry_list == NULL)
{
return NULL;
}
scope_entry_t* unknown_info = entry_list_head(entry_list);
entry_list_free(entry_list);
return unknown_info;
}
static scope_entry_t* get_or_create_special_symbol(const decl_context_t* decl_context, const char* name)
{
scope_entry_t* unknown_info = get_special_symbol(decl_context, name);
if (unknown_info == NULL)
{
decl_context_t* global_context = decl_context_clone(decl_context);
global_context->current_scope = global_context->function_scope;
unknown_info = new_symbol(global_context, global_context->current_scope, name);
unknown_info->kind = SK_OTHER;
}
return unknown_info;
}
scope_entry_t* fortran_get_data_symbol_info(const decl_context_t* decl_context)
{
return get_special_symbol(decl_context, UNIQUESTR_LITERAL(".data"));
}
static scope_entry_t* get_or_create_data_symbol_info(const decl_context_t* decl_context)
{
return get_or_create_special_symbol(decl_context, UNIQUESTR_LITERAL(".data"));
}
scope_entry_t* fortran_get_equivalence_symbol_info(const decl_context_t* decl_context)
{
return get_special_symbol(decl_context, UNIQUESTR_LITERAL(".equivalence"));
}
scope_entry_t* get_or_create_used_modules_symbol_info(const decl_context_t* decl_context)
{
ERROR_CONDITION(decl_context->current_scope->related_entry == NULL, "No related symbol in the current scope!", 0);
if (symbol_entity_specs_get_used_modules(decl_context->current_scope->related_entry) == NULL)
{
decl_context_t* function_context = decl_context_clone(decl_context);
function_context->current_scope = function_context->function_scope;
scope_entry_t* new_sym = new_symbol(
function_context,
function_context->current_scope,
UNIQUESTR_LITERAL(".used_modules"));
new_sym->kind = SK_OTHER;
symbol_entity_specs_set_used_modules(decl_context->current_scope->related_entry, new_sym);
}
return symbol_entity_specs_get_used_modules(decl_context->current_scope->related_entry);
}
static scope_entry_t* get_or_create_equivalence_symbol_info(const decl_context_t* decl_context)
{
return get_or_create_special_symbol(decl_context, UNIQUESTR_LITERAL(".equivalence"));
}
static void add_delay_check_symbol_needs_type_specifier(
const decl_context_t *decl_context, scope_entry_t *entry);
static scope_entry_t* create_fortran_symbol_for_name_(const decl_context_t* decl_context, 
AST location, const char* name,
char no_implicit)
{ 
scope_entry_t* result = new_fortran_symbol(decl_context, name);
if (!no_implicit)
{
result->type_information = get_implicit_type_for_symbol(decl_context, result->symbol_name);
}
else
{
result->type_information = get_void_type();
}
add_delay_check_symbol_needs_type_specifier(decl_context, result);
symbol_entity_specs_set_is_implicit_basic_type(result, 1);
result->locus = ast_get_locus(location);
if (decl_context->current_scope->related_entry != NULL
&& (decl_context->current_scope->related_entry->kind == SK_MODULE
|| decl_context->current_scope->related_entry->kind == SK_BLOCKDATA))
{
scope_entry_t* module = decl_context->current_scope->related_entry;
symbol_entity_specs_insert_related_symbols(module, result);
symbol_entity_specs_set_in_module(result, module);
}
return result;
}
static scope_entry_list_t* get_symbols_for_name(const decl_context_t* decl_context, 
AST location, const char* name)
{
scope_entry_list_t* result = query_in_scope_str_flags(decl_context, strtolower(name), NULL, DF_ONLY_CURRENT_SCOPE);
if (result == NULL)
{
result = entry_list_new(create_fortran_symbol_for_name_(decl_context, location, name,  0));
}
return result;
}
static scope_entry_t* get_symbol_for_name(const decl_context_t* decl_context, 
AST location, const char* name)
{
scope_entry_list_t* entry_list = get_symbols_for_name(decl_context, location, name);
scope_entry_t* result = entry_list_head(entry_list);
entry_list_free(entry_list);
return result;
}
static void build_scope_main_program_unit(
AST program_unit,
const decl_context_t *decl_context,
scope_entry_t **program_unit_symbol,
nodecl_t *nodecl_output);
static void build_scope_subroutine_program_unit(
AST program_unit,
const decl_context_t *program_unit_context,
scope_entry_t **program_unit_symbol,
nodecl_t *nodecl_output);
static void build_scope_function_program_unit(
AST program_unit,
const decl_context_t *program_unit_context,
scope_entry_t **program_unit_symbol,
nodecl_t *nodecl_output);
static void build_scope_module_program_unit(
AST program_unit,
const decl_context_t *program_unit_context,
scope_entry_t **program_unit_symbol,
nodecl_t *nodecl_output);
static void build_scope_block_data_program_unit(
AST program_unit,
const decl_context_t *program_unit_context,
scope_entry_t **program_unit_symbol,
nodecl_t *nodecl_output);
static void build_global_program_unit(AST program_unit);
static void handle_opt_value_list(AST io_stmt, AST opt_value_list,
const decl_context_t* decl_context,
nodecl_t* nodecl_output);
static void build_scope_program_unit_internal(
AST program_unit,
const decl_context_t *decl_context,
scope_entry_t **program_unit_symbol,
nodecl_t *nodecl_output)
{
scope_entry_t *_program_unit_symbol = NULL;
build_scope_delay_list_t program_unit_delayed = {};
build_scope_delay_list_push(&program_unit_delayed);
switch (ASTKind(program_unit))
{
case AST_MAIN_PROGRAM_UNIT:
{
build_scope_main_program_unit(program_unit,
decl_context,
&_program_unit_symbol,
nodecl_output);
break;
}
case AST_SUBROUTINE_PROGRAM_UNIT:
{
build_scope_subroutine_program_unit(program_unit,
decl_context,
&_program_unit_symbol,
nodecl_output);
break;
}
case AST_FUNCTION_PROGRAM_UNIT:
{
build_scope_function_program_unit(program_unit,
decl_context,
&_program_unit_symbol,
nodecl_output);
break;
}
case AST_MODULE_PROGRAM_UNIT:
{
build_scope_module_program_unit(program_unit,
decl_context,
&_program_unit_symbol,
nodecl_output);
break;
}
case AST_BLOCK_DATA_PROGRAM_UNIT:
{
build_scope_block_data_program_unit(program_unit,
decl_context,
&_program_unit_symbol,
nodecl_output);
break;
}
case AST_GLOBAL_PROGRAM_UNIT:
{
build_global_program_unit(program_unit);
break;
}
case AST_PRAGMA_CUSTOM_CONSTRUCT:
{
AST pragma_line = ASTSon0(program_unit);
AST nested_program_unit = ASTSon1(program_unit);
build_scope_program_unit_internal(nested_program_unit,
decl_context,
&_program_unit_symbol,
nodecl_output);
if (_program_unit_symbol != NULL && !nodecl_is_null(*nodecl_output))
{
const decl_context_t *context_in_scope
= _program_unit_symbol->related_decl_context;
nodecl_t nodecl_pragma_line = nodecl_null();
common_build_scope_pragma_custom_line(pragma_line,
NULL,
context_in_scope,
&nodecl_pragma_line);
nodecl_t nodecl_nested_pragma = nodecl_null();
if (ASTKind(nested_program_unit) == AST_PRAGMA_CUSTOM_CONSTRUCT)
{
int num_items = 0;
nodecl_t *list
= nodecl_unpack_list(*nodecl_output, &num_items);
ERROR_CONDITION(
(num_items != 2),
"This list does not have the expected shape",
0);
ERROR_CONDITION(nodecl_get_kind(list[1])
!= NODECL_PRAGMA_CUSTOM_DECLARATION,
"Invalid kind for second item of the list",
0);
nodecl_nested_pragma = list[1];
DELETE(list);
}
nodecl_t nodecl_pragma_declaration
= nodecl_make_pragma_custom_declaration(
nodecl_pragma_line,
nodecl_nested_pragma,
nodecl_make_pragma_context(context_in_scope,
ast_get_locus(program_unit)),
nodecl_make_pragma_context(context_in_scope,
ast_get_locus(program_unit)),
_program_unit_symbol,
strtolower(ASTText(program_unit)),
ast_get_locus(program_unit));
*nodecl_output
= nodecl_make_list_2(nodecl_list_head(*nodecl_output),
nodecl_pragma_declaration);
}
break;
}
case AST_UNKNOWN_PRAGMA:
{
break;
}
default:
{
internal_error("Unhandled node type '%s'\n",
ast_print_node_type(ASTKind(program_unit)));
}
}
build_scope_delay_list_pop();
if (program_unit_symbol != NULL)
{
*program_unit_symbol = _program_unit_symbol;
}
}
typedef
struct delayed_initialization_tag
{
scope_entry_t* entry;
AST initialization;
const decl_context_t *decl_context;
char is_pointer_init;
char adjust_assumed_character_length;
char is_parameter;
char is_variable;
} delayed_initialization_t;
static void delay_check_initialization(void *info,
UNUSED_PARAMETER nodecl_t *nodecl_output)
{
delayed_initialization_t *delayed_info = (delayed_initialization_t *)info;
nodecl_t nodecl_init = nodecl_null();
fortran_check_initialization(delayed_info->entry,
delayed_info->initialization,
delayed_info->decl_context,
delayed_info->is_pointer_init,
&nodecl_init);
if (!nodecl_is_err_expr(nodecl_init))
{
delayed_info->entry->value = nodecl_init;
if (delayed_info->adjust_assumed_character_length
&& fortran_is_character_type(delayed_info->entry->type_information)
&& array_type_is_unknown_size(delayed_info->entry->type_information)
&& !nodecl_is_null(nodecl_init)
&& !nodecl_is_err_expr(nodecl_init)
&& nodecl_is_constant(nodecl_init)
&& const_value_is_string(nodecl_get_constant(nodecl_init)))
{
int num_elements = const_value_get_num_elements(
nodecl_get_constant(nodecl_init));
delayed_info->entry->type_information = get_array_type_bounds(
array_type_get_element_type(
delayed_info->entry->type_information),
nodecl_make_integer_literal(
get_signed_int_type(),
const_value_get_one(fortran_get_default_integer_type_kind(),
1),
ast_get_locus(delayed_info->initialization)),
nodecl_make_integer_literal(
get_signed_int_type(),
const_value_get_integer(
num_elements,
fortran_get_default_integer_type_kind(),
1),
ast_get_locus(delayed_info->initialization)),
delayed_info->decl_context);
}
if (delayed_info->is_variable)
{
symbol_entity_specs_set_is_static(delayed_info->entry, 1);
}
else if (delayed_info->is_parameter)
{
delayed_info->entry->type_information = get_const_qualified_type(
delayed_info->entry->type_information);
}
}
DELETE(delayed_info);
}
static void fortran_delay_check_initialization(scope_entry_t* entry,
AST initialization,
const decl_context_t* decl_context,
char is_pointer_init,
char adjust_assumed_character_length,
char is_parameter,
char is_variable)
{
delayed_initialization_t *data = NEW(delayed_initialization_t);
data->entry = entry;
data->initialization = initialization;
data->decl_context = decl_context;
data->is_pointer_init = is_pointer_init;
data->adjust_assumed_character_length = adjust_assumed_character_length;
data->is_parameter = is_parameter;
data->is_variable = is_variable;
build_scope_delay_list_add(
DELAY_AFTER_DECLARATIONS,
delay_check_initialization,
data);
}
static void fortran_immediate_check_initialization(scope_entry_t* entry,
AST initialization,
const decl_context_t* decl_context,
char is_pointer_init,
char adjust_assumed_character_length,
char is_parameter,
char is_variable)
{
delayed_initialization_t *data = NEW(delayed_initialization_t);
data->entry = entry;
data->initialization = initialization;
data->decl_context = decl_context;
data->is_pointer_init = is_pointer_init;
data->adjust_assumed_character_length = adjust_assumed_character_length;
data->is_parameter = is_parameter;
data->is_variable = is_variable;
delay_check_initialization(data, NULL);
}
typedef
struct delayed_character_length_tag
{
type_t* character_type;
AST length;
const decl_context_t* decl_context;
int num_symbols;
scope_entry_t** symbols;
} delayed_character_length_t;
static type_t* delayed_character_length_update_type(type_t* original_type, type_t* new_type)
{
if (is_lvalue_reference_type(original_type))
{
return get_lvalue_reference_type(
delayed_character_length_update_type(
reference_type_get_referenced_type(original_type),
new_type));
}
else if (fortran_is_character_type(original_type))
{
cv_qualifier_t cv_qualif = get_cv_qualifier(original_type);
return get_cv_qualified_type(new_type, cv_qualif);
}
else if (is_pointer_type(original_type))
{
cv_qualifier_t cv_qualif = get_cv_qualifier(original_type);
return get_cv_qualified_type(
get_pointer_type(
delayed_character_length_update_type(
pointer_type_get_pointee_type(original_type),
new_type)),
cv_qualif);
}
else if (fortran_is_array_type(original_type))
{
cv_qualifier_t cv_qualif = get_cv_qualifier(original_type);
return get_cv_qualified_type(
array_type_rebase(
original_type,
delayed_character_length_update_type(
array_type_get_element_type(original_type),
new_type)),
cv_qualif);
}
else if (is_function_type(original_type))
{
return function_type_replace_return_type(original_type,
delayed_character_length_update_type(
function_type_get_return_type(original_type),
new_type));
}
else
{
internal_error("Unexpected type '%s'\n", print_declarator(original_type));
}
}
static delayed_character_length_t* delayed_character_length_new(
type_t* character_type,
AST character_length,
const decl_context_t* decl_context,
int num_symbols,
scope_entry_t* symbols[])
{
delayed_character_length_t* result = NEW0(delayed_character_length_t);
ERROR_CONDITION(!fortran_is_character_type(character_type), "Invalid type", 0);
result->character_type = character_type;
result->length = character_length;
result->decl_context = decl_context;
result->num_symbols = num_symbols;
result->symbols = NEW_VEC0(scope_entry_t*, num_symbols);
memcpy(result->symbols, symbols, sizeof(*result->symbols)*num_symbols);
return result;
}
static type_t *compute_character_length_type(AST character_length,
type_t *character_type,
const decl_context_t *decl_context,
char *is_star,
char *is_colon)
{
ERROR_CONDITION(!fortran_is_character_type(character_type),
"This must be a CHARACTER type",
0);
nodecl_t nodecl_len = nodecl_null();
*is_star = 0;
*is_colon = 0;
if (ASTKind(character_length) == AST_SYMBOL
&& strcmp(ASTText(character_length), "*") == 0)
{
*is_star = 1;
}
else if (ASTKind(character_length) == AST_SYMBOL
&& strcmp(ASTText(character_length), ":") == 0)
{
*is_colon = 1;
}
else
{
fortran_check_expression(character_length, decl_context, &nodecl_len);
}
if (*is_star)
{
return character_type;
}
else if (*is_colon)
{
type_t *updated_char_type = get_array_type_bounds_with_descriptor(
array_type_get_element_type(character_type),
nodecl_null(),
nodecl_null(),
decl_context);
return updated_char_type;
}
else if (nodecl_is_err_expr(nodecl_len))
{
return get_error_type();
}
else
{
nodecl_len = fortran_expression_as_value(nodecl_len);
nodecl_t lower_bound = nodecl_make_integer_literal(
get_signed_int_type(),
const_value_get_one(type_get_size(get_signed_int_type()), 1),
nodecl_get_locus(nodecl_len));
return get_array_type_bounds(
array_type_get_element_type(character_type),
lower_bound,
nodecl_len,
decl_context);
}
}
static void delay_character_check_is_dummy_or_parameter(void *info,
UNUSED_PARAMETER nodecl_t *nodecl_output)
{
scope_entry_t *entry = (scope_entry_t*)info;
if (!is_const_qualified_type(entry->type_information)
&& !symbol_is_parameter_of_function(entry, entry->decl_context->current_scope->related_entry))
{
error_printf_at(entry->locus,
"'%s' has assumed CHARACTER length but is not a dummy argument or a parameter\n",
entry->symbol_name);
}
}
static void delay_character_check_is_allocatable_or_pointer(void *info,
UNUSED_PARAMETER nodecl_t *nodecl_output)
{
scope_entry_t *entry = (scope_entry_t*)info;
if (!symbol_entity_specs_get_is_allocatable(entry)
&& !is_pointer_type(no_ref(entry->type_information)))
{
error_printf_at(entry->locus,
"'%s' has deferred CHARACTER length but is not ALLOCATABLE or POINTER\n",
entry->symbol_name);
}
}
static void delayed_compute_character_length(
void *info, nodecl_t *nodecl_output UNUSED_PARAMETER)
{
delayed_character_length_t *data = (delayed_character_length_t *)info;
char is_star = 0;
char is_colon = 0;
DEBUG_CODE()
{
fprintf(stderr,
"BUILDSCOPE: Computing delayed character length of '%s'\n",
prettyprint_in_buffer(data->length));
}
type_t *updated_char_type = compute_character_length_type(
data->length, data->character_type, data->decl_context,
&is_star,
&is_colon);
if (is_error_type(updated_char_type))
{
int i;
for (i = 0; i < data->num_symbols; i++)
{
scope_entry_t *current_symbol = data->symbols[i];
current_symbol->type_information = get_error_type();
}
}
else
{
int i;
for (i = 0; i < data->num_symbols; i++)
{
scope_entry_t *current_symbol = data->symbols[i];
if (is_star)
{
build_scope_delay_list_add(DELAY_AFTER_DECLARATIONS,
delay_character_check_is_dummy_or_parameter,
current_symbol);
}
else
{
if (is_colon)
{
build_scope_delay_list_add(
DELAY_AFTER_DECLARATIONS,
delay_character_check_is_allocatable_or_pointer,
current_symbol);
}
current_symbol->type_information
= delayed_character_length_update_type(
current_symbol->type_information, updated_char_type);
}
}
}
DELETE(data->symbols);
DELETE(data);
}
typedef
struct postponed_function_type_spec_t
{
const decl_context_t* decl_context;
AST postponed_function_type_spec;
} postponed_function_type_spec_t;
static void delayed_solve_postponed_function_type_spec(void *info, UNUSED_PARAMETER nodecl_t* nodecl_out)
{
postponed_function_type_spec_t* postponed_info = (postponed_function_type_spec_t*)info;
AST postponed_function_type_spec = postponed_info->postponed_function_type_spec;
const decl_context_t* decl_context = postponed_info->decl_context;
AST length = NULL;
type_t* function_type_spec =
fortran_gather_type_from_declaration_type_spec(postponed_function_type_spec, decl_context, &length);
if (is_error_type(function_type_spec))
{
DELETE(info);
return;
}
scope_entry_t* current_function = decl_context->current_scope->related_entry;
current_function->type_information = fortran_update_basic_type_with_type(current_function->type_information,
function_type_spec);
symbol_entity_specs_set_is_implicit_basic_type(current_function, 0);
scope_entry_t* result_name = symbol_entity_specs_get_result_var(current_function);
if (result_name != NULL)
{
result_name->type_information = fortran_update_basic_type_with_type(result_name->type_information,
function_type_spec);
symbol_entity_specs_set_is_implicit_basic_type(result_name, 0);
}
if (fortran_is_character_type(function_type_spec)
&& length != NULL)
{
delayed_character_length_t *data = 
delayed_character_length_new(
function_type_spec,
length,
decl_context,
2,
(scope_entry_t*[2]){current_function, result_name});
build_scope_delay_list_add(
DELAY_AFTER_DECLARATIONS, delayed_compute_character_length, data);
}
DELETE(info);
}
static scope_entry_t* new_procedure_symbol(
const decl_context_t* decl_context,
const decl_context_t* program_unit_context,
AST name, AST prefix, AST suffix, AST dummy_arg_name_list,
char is_function);
#define ORDER_CLASS_TABLE \
ORDER_CLASS(SOC_USE, 0) \
ORDER_CLASS(SOC_IMPORT, 1) \
ORDER_CLASS(SOC_FORMAT, 2) \
ORDER_CLASS(SOC_ENTRY, 3) \
ORDER_CLASS(SOC_PARAMETER, 4) \
ORDER_CLASS(SOC_IMPLICIT, 5) \
ORDER_CLASS(SOC_IMPLICIT_NONE, 6) \
ORDER_CLASS(SOC_DATA, 7) \
ORDER_CLASS(SOC_DECLARATION, 8) \
ORDER_CLASS(SOC_EXECUTABLE, 9) \
ORDER_CLASS(SOC_EVERYWHERE, 10)
typedef enum statement_order_class_t {
SOC_INVALID = 0,
#define ORDER_CLASS(n, v) \
n = 1 << v,
ORDER_CLASS_TABLE
#undef ORDER_CLASS
} statement_order_class_t;
static const char *statement_order_class_print(
statement_order_class_t order_class)
{
if (order_class == 0)
return "SOC_INVALID";
const char *ret = NULL;
#define ORDER_CLASS(n, _) \
if (order_class & n)  { \
if (ret) \
uniquestr_sprintf(&ret, "%s | %s", ret, #n); \
else \
ret = #n; \
}
ORDER_CLASS_TABLE
#undef ORDER_CLASS
return ret;
}
#define ORDER_CLASS(n, _) \
static statement_order_class_t GET_##n(AST a UNUSED_PARAMETER) { return n; }
ORDER_CLASS_TABLE
#undef ORDER_CLASS
static const statement_order_class_t statement_order[] = {
SOC_EVERYWHERE | SOC_USE,
SOC_EVERYWHERE | SOC_IMPORT,
SOC_EVERYWHERE | SOC_FORMAT | SOC_ENTRY | SOC_IMPLICIT_NONE,
SOC_EVERYWHERE | SOC_FORMAT | SOC_ENTRY | SOC_PARAMETER | SOC_IMPLICIT,
SOC_EVERYWHERE | SOC_FORMAT | SOC_ENTRY | SOC_PARAMETER | SOC_DATA | SOC_DECLARATION,
SOC_EVERYWHERE | SOC_FORMAT | SOC_ENTRY | SOC_DATA      | SOC_EXECUTABLE,
};
enum { STATEMENT_ORDER_NUM = sizeof(statement_order) / sizeof(*statement_order) };
static const statement_order_class_t *statement_order_end = &statement_order[STATEMENT_ORDER_NUM];
typedef struct statement_constraint_checker_t statement_constraint_checker_t;
typedef char (*allowed_statement_t)(statement_constraint_checker_t *,
AST,
const decl_context_t *);
struct statement_constraint_checker_t
{
const statement_order_class_t *current_order_class;
allowed_statement_t allowed_statement;
};
static statement_constraint_checker_t statement_constraint_checker_init(
allowed_statement_t allowed_statement)
{
statement_constraint_checker_t ret
= { &statement_order[0], allowed_statement };
return ret;
}
static char statement_constraint_checker_update_order(
statement_constraint_checker_t *order_tracker UNUSED_PARAMETER, AST stmt);
static char statement_constraint_checker_check_statement(
statement_constraint_checker_t *order_tracker,
AST stmt,
const decl_context_t *decl_context)
{
ERROR_CONDITION(ASTKind(stmt) == AST_AMBIGUITY, "Ambiguity nodes are not allowed here", 0);
if (!(order_tracker->allowed_statement)(order_tracker, stmt, decl_context))
return 0;
if (!statement_constraint_checker_update_order(order_tracker, stmt))
{
error_printf_at(
ast_get_locus(stmt),
"statement is incorrectly located within the program unit\n");
return 0;
}
return 1;
}
static char allow_all_statements(
statement_constraint_checker_t *checker UNUSED_PARAMETER,
AST a UNUSED_PARAMETER,
const decl_context_t *decl_context UNUSED_PARAMETER)
{
return 1;
}
static statement_order_class_t statement_get_order_class(AST statement);
static char module_specification_part_allowed_statements(
statement_constraint_checker_t *checker UNUSED_PARAMETER,
AST a,
const decl_context_t *decl_context UNUSED_PARAMETER)
{
statement_order_class_t order_class = statement_get_order_class(a);
if (order_class == SOC_EXECUTABLE || order_class == SOC_ENTRY
|| order_class == SOC_FORMAT
|| ASTKind(a) == AST_STATEMENT_FUNCTION_STATEMENT
|| (ASTKind(a) == AST_LABELED_STATEMENT
&& ASTKind(ASTSon1(a)) == AST_STATEMENT_FUNCTION_STATEMENT))
{
error_printf_at(ast_get_locus(a),
"this statement is not allowed in the specification "
"part of a MODULE\n");
return 0;
}
return 1;
}
static char internal_subprogram_allowed_statements(
statement_constraint_checker_t *checker UNUSED_PARAMETER,
AST a,
const decl_context_t *decl_context UNUSED_PARAMETER)
{
statement_order_class_t order_class = statement_get_order_class(a);
if (order_class == SOC_ENTRY)
{
error_printf_at(
ast_get_locus(a),
"an ENTRY statement is not allowed in an internal subprogram\n");
return 0;
}
return 1;
}
static char executable_construct_allowed_statements(
statement_constraint_checker_t *checker UNUSED_PARAMETER,
AST a,
const decl_context_t *decl_context UNUSED_PARAMETER)
{
statement_order_class_t order_class = statement_get_order_class(a);
if (order_class == SOC_ENTRY)
{
error_printf_at(
ast_get_locus(a),
"an ENTRY statement is not allowed within an executable construct\n");
return 0;
}
else if (!(order_class
& (SOC_EVERYWHERE | SOC_FORMAT | SOC_DATA | SOC_EXECUTABLE)))
{
error_printf_at(
ast_get_locus(a),
"this statement is not allowed within an executable construct\n");
return 0;
}
return 1;
}
static char block_construct_allowed_statements(
statement_constraint_checker_t *checker UNUSED_PARAMETER,
AST a,
const decl_context_t *decl_context UNUSED_PARAMETER)
{
if (ASTKind(a) == AST_LABELED_STATEMENT)
a = ASTSon1(a);
switch (ASTKind(a))
{
case AST_COMMON_STATEMENT:
case AST_EQUIVALENCE_STATEMENT:
case AST_IMPLICIT_STATEMENT:
case AST_INTENT_STATEMENT:
case AST_NAMELIST_STATEMENT:
case AST_OPTIONAL_STATEMENT:
case AST_STATEMENT_FUNCTION_STATEMENT:
case AST_VALUE_STATEMENT:
error_printf_at(ast_get_locus(a), "invalid statement within BLOCK construct");
return 0;
default:
return 1;
}
return 1;
}
static char block_data_allowed_statements(
statement_constraint_checker_t *checker UNUSED_PARAMETER,
AST a,
const decl_context_t *decl_context UNUSED_PARAMETER)
{
if (ASTKind(a) == AST_LABELED_STATEMENT)
a = ASTSon1(a);
switch (ASTKind(a))
{
default:
error_printf_at(
ast_get_locus(a),
"this statement is not allowed in a BLOCK DATA program unit\n");
return 0;
case AST_DERIVED_TYPE_DEF:
case AST_ASYNCHRONOUS_STATEMENT:
case AST_BIND_STATEMENT:
case AST_COMMON_STATEMENT:
case AST_DATA_STATEMENT:
case AST_DIMENSION_STATEMENT:
case AST_EQUIVALENCE_STATEMENT:
case AST_IMPLICIT_NONE_STATEMENT:
case AST_IMPLICIT_STATEMENT:
case AST_INTRINSIC_STATEMENT:
case AST_PARAMETER_STATEMENT:
case AST_POINTER_STATEMENT:
case AST_SAVE_STATEMENT:
case AST_TARGET_STATEMENT:
case AST_USE_STATEMENT:
case AST_USE_ONLY_STATEMENT:
case AST_VOLATILE_STATEMENT:
case AST_DECLARATION_STATEMENT:
return 1;
}
}
static void delay_review_symbol_has_known_kind(
void *data, UNUSED_PARAMETER nodecl_t *nodecl_out)
{
scope_entry_t *sym = (scope_entry_t *)data;
if (sym->kind == SK_UNDEFINED)
{
DEBUG_CODE()
{
fprintf(stderr,
"BUILDSCOPE: Making symbol '%s' which is still "
"SK_UNDEFINED a SK_VARIABLE\n",
sym->symbol_name);
}
sym->kind = SK_VARIABLE;
}
}
void add_delay_check_kind_of_symbol(
UNUSED_PARAMETER const decl_context_t *decl_context, scope_entry_t *entry)
{
build_scope_delay_list_add(
DELAY_AFTER_PROGRAM_UNIT, delay_review_symbol_has_known_kind, entry);
}
typedef
struct delay_check_is_parameter_tag
{
const decl_context_t* decl_context;
scope_entry_t* entry;
} delay_check_is_parameter_t;
static void delay_check_symbol_is_parameter(
void *data, UNUSED_PARAMETER nodecl_t *nodecl_out)
{
delay_check_is_parameter_t* check_param_info = (delay_check_is_parameter_t*)data;
const decl_context_t* decl_context = check_param_info->decl_context;
scope_entry_t *entry = check_param_info->entry;
DEBUG_CODE()
{
fprintf(stderr,
"BUILDSCOPE: Delayed check if '%s' is a parameter\n",
entry->symbol_name);
}
if (!symbol_is_parameter_of_function(
entry, decl_context->current_scope->related_entry))
{
error_printf_at(entry->locus,
"entity '%s' is not a dummy argument\n",
entry->symbol_name);
}
DELETE(data);
}
static void add_delay_check_symbol_is_dummy(const decl_context_t *decl_context,
scope_entry_t *entry)
{
delay_check_is_parameter_t *param_info = NEW0(delay_check_is_parameter_t);
param_info->decl_context = decl_context;
param_info->entry = entry;
build_scope_delay_list_add(
DELAY_AFTER_PROGRAM_UNIT, delay_check_symbol_is_parameter, param_info);
}
static char cmp_intent_declared_symbol(void *key, void *data)
{
scope_entry_t* entry = (scope_entry_t*)key;
delay_check_is_parameter_t* check_param_info = (delay_check_is_parameter_t*)data;
return entry == check_param_info->entry;
}
static void remove_intent_declared_symbol(scope_entry_t *entry)
{
build_scope_delay_list_remove(
DELAY_AFTER_PROGRAM_UNIT,
entry,
delay_check_symbol_is_parameter,
cmp_intent_declared_symbol);
}
static void delay_check_fully_defined_symbol(
void *data, UNUSED_PARAMETER nodecl_t *nodecl_out)
{
scope_entry_t *entry = (scope_entry_t *)data;
DEBUG_CODE()
{
fprintf(stderr,
"BUILDSCOPE: Delayed check if '%s' has been fully defined\n",
entry->symbol_name);
}
if (!entry->defined && entry->kind == SK_COMMON)
{
error_printf_at(entry->locus,
"COMMON '%s' does not exist\n",
entry->symbol_name + strlen(".common."));
}
else if (!entry->defined && entry->kind == SK_FUNCTION)
{
error_printf_at(entry->locus,
"%sPROCEDURE '%s' does not exist\n",
symbol_entity_specs_get_is_module_procedure(entry) ?
"MODULE " : "",
entry->symbol_name);
}
else if (!entry->defined && entry->kind == SK_CLASS)
{
error_printf_at(entry->locus,
"derived type name 'TYPE(%s)' has not been defined\n",
entry->symbol_name);
}
}
static void add_delay_check_fully_defined_symbol(
UNUSED_PARAMETER const decl_context_t *decl_context,
scope_entry_t* entry)
{
build_scope_delay_list_add(
DELAY_AFTER_PROGRAM_UNIT, delay_check_fully_defined_symbol, entry);
}
static void build_scope_program_unit_body(
AST program_unit_stmts,
AST internal_subprograms,
AST end_statement,
const decl_context_t* decl_context,
statement_constraint_checker_t* constraint_checker,
nodecl_t* nodecl_output,
nodecl_t* nodecl_internal_subprograms);
static void build_scope_main_program_unit(
AST program_unit,
const decl_context_t *decl_context,
scope_entry_t **program_unit_symbol,
nodecl_t *nodecl_output)
{
const decl_context_t* program_unit_context = new_program_unit_context(decl_context);
ERROR_CONDITION(program_unit_symbol == NULL, "Invalid parameter", 0)
DEBUG_CODE()
{
fprintf(stderr, "==== [%s] Program unit: PROGRAM ===\n", ast_location(program_unit));
}
AST program_stmt = ASTSon0(program_unit);
const char * program_name = "__MAIN__";
if (program_stmt != NULL)
{
AST name = ASTSon0(program_stmt);
program_name = ASTText(name);
}
scope_entry_t* program_sym = new_fortran_symbol_not_unknown(decl_context, program_name);
program_sym->kind = SK_PROGRAM;
program_sym->locus = ast_get_locus(program_unit);
symbol_entity_specs_set_is_global_hidden(program_sym, 1);
program_sym->related_decl_context = program_unit_context;
program_unit_context->current_scope->related_entry = program_sym;
*program_unit_symbol = program_sym;
AST program_body = ASTSon1(program_unit);
AST end_statement = ASTSon2(program_unit);
nodecl_t nodecl_body = nodecl_null();
nodecl_t nodecl_internal_subprograms = nodecl_null();
if (program_body != NULL)
{
AST top_level = ASTSon0(program_body);
AST statement_seq = ASTSon0(top_level);
AST internal_subprograms = ASTSon1(program_body);
statement_constraint_checker_t constraint_checker
= statement_constraint_checker_init(allow_all_statements);
build_scope_program_unit_body(statement_seq,
internal_subprograms,
end_statement,
program_unit_context,
&constraint_checker,
&nodecl_body,
&nodecl_internal_subprograms);
}
if (nodecl_is_null(nodecl_body) && nodecl_is_null(nodecl_internal_subprograms))
{
nodecl_body = nodecl_make_list_1(nodecl_make_empty_statement(ast_get_locus(program_unit)));
}
else
{
nodecl_body = nodecl_concat_lists(nodecl_body, nodecl_internal_subprograms);
}
nodecl_t function_code = nodecl_make_function_code(
nodecl_make_context(
nodecl_body,
program_unit_context,
ast_get_locus(program_unit)),
nodecl_null(),
program_sym,
ast_get_locus(program_unit));
symbol_entity_specs_set_function_code(program_sym, function_code);
*nodecl_output = nodecl_make_list_1(function_code);
}
static scope_entry_t* register_function(AST program_unit,
const decl_context_t* decl_context,
const decl_context_t* program_unit_context)
{
AST function_stmt = ASTSon0(program_unit);
AST prefix = ASTSon0(function_stmt);
AST name = ASTSon1(function_stmt);
AST function_prototype = ASTSon2(function_stmt);
AST dummy_arg_name_list = ASTSon0(function_prototype);
AST suffix = ASTSon1(function_prototype);
scope_entry_t *new_entry = new_procedure_symbol(
decl_context,
program_unit_context,
name, prefix, suffix,
dummy_arg_name_list,  1);
return new_entry;
}
static char inside_interface(AST a)
{
if (a == NULL)
return 0;
if (ASTKind(a) == AST_INTERFACE_BLOCK)
return 1;
else
return inside_interface(ASTParent(a));
}
static void build_scope_function_program_unit(
AST program_unit,
const decl_context_t *decl_context,
scope_entry_t **program_unit_symbol,
nodecl_t *nodecl_output)
{
const decl_context_t* program_unit_context = new_program_unit_context(decl_context);
ERROR_CONDITION(program_unit_symbol == NULL, "Invalid parameter", 0)
DEBUG_CODE()
{
fprintf(stderr, "==== [%s] Program unit: FUNCTION ===\n", ast_location(program_unit));
}
scope_entry_t* new_entry = register_function(program_unit, decl_context, program_unit_context);
if (new_entry == NULL)
return;
if (inside_interface(program_unit))
{
program_unit_context->current_scope->contained_in = program_unit_context->global_scope;
}
*program_unit_symbol = new_entry;
nodecl_t nodecl_body = nodecl_null();
nodecl_t nodecl_internal_subprograms = nodecl_null();
AST program_body = ASTSon1(program_unit);
AST end_statement = ASTSon2(program_unit);
if (program_body != NULL)
{
AST top_level = ASTSon0(program_body);
AST statement_seq = ASTSon0(top_level);
AST internal_subprograms = ASTSon1(program_body);
statement_constraint_checker_t constraint_checker
= statement_constraint_checker_init(allow_all_statements);
build_scope_program_unit_body(statement_seq,
internal_subprograms,
end_statement,
program_unit_context,
&constraint_checker,
&nodecl_body,
&nodecl_internal_subprograms);
}
if (nodecl_is_null(nodecl_body) && nodecl_is_null(nodecl_internal_subprograms))
{
nodecl_body = nodecl_make_list_1(nodecl_make_empty_statement(ast_get_locus(program_unit)));
}
else
{
nodecl_body = nodecl_concat_lists(nodecl_body, nodecl_internal_subprograms);
}
int i, num_params = symbol_entity_specs_get_num_related_symbols(new_entry);
for (i = 0; i < num_params; i++)
{
if (symbol_entity_specs_get_related_symbols_num(new_entry, i)->kind == SK_UNDEFINED)
{
symbol_entity_specs_get_related_symbols_num(new_entry, i)->kind = SK_VARIABLE;
}
}
nodecl_t function_code = nodecl_make_function_code(
nodecl_make_context(
nodecl_body,
program_unit_context,
ast_get_locus(program_unit)),
nodecl_null(),
new_entry,
ast_get_locus(program_unit));
symbol_entity_specs_set_function_code(new_entry, function_code);
*nodecl_output = nodecl_make_list_1(function_code);
}
static scope_entry_t* register_subroutine(AST program_unit,
const decl_context_t* decl_context,
const decl_context_t* program_unit_context)
{
AST subroutine_stmt = ASTSon0(program_unit);
AST prefix = ASTSon0(subroutine_stmt);
AST name = ASTSon1(subroutine_stmt);
AST function_prototype = ASTSon2(subroutine_stmt);
AST dummy_arg_name_list = NULL;
AST suffix = NULL;
if (function_prototype != NULL)
{
dummy_arg_name_list = ASTSon0(function_prototype);
suffix = ASTSon1(function_prototype);
}
scope_entry_t *new_entry = new_procedure_symbol(
decl_context,
program_unit_context,
name, prefix, suffix, 
dummy_arg_name_list,  0);
return new_entry;
}
static void build_scope_subroutine_program_unit(
AST program_unit,
const decl_context_t *decl_context,
scope_entry_t **program_unit_symbol,
nodecl_t *nodecl_output)
{
const decl_context_t* program_unit_context = new_program_unit_context(decl_context);
ERROR_CONDITION(program_unit_symbol == NULL, "Invalid parameter", 0)
DEBUG_CODE()
{
fprintf(stderr, "==== [%s] Program unit: SUBROUTINE ===\n", ast_location(program_unit));
}
scope_entry_t *new_entry = register_subroutine(program_unit, decl_context, program_unit_context);
if (new_entry == NULL)
return;
if (inside_interface(program_unit))
{
program_unit_context->current_scope->contained_in = program_unit_context->global_scope;
}
*program_unit_symbol = new_entry;
symbol_entity_specs_set_is_implicit_basic_type(new_entry, 0);
*program_unit_symbol = new_entry;
nodecl_t nodecl_body = nodecl_null();
nodecl_t nodecl_internal_subprograms = nodecl_null();
AST program_body = ASTSon1(program_unit);
AST end_statement = ASTSon2(program_unit);
if (program_body != NULL)
{
AST top_level = ASTSon0(program_body);
AST statement_seq = ASTSon0(top_level);
AST internal_subprograms = ASTSon1(program_body);
statement_constraint_checker_t constraint_checker
= statement_constraint_checker_init(allow_all_statements);
build_scope_program_unit_body(statement_seq,
internal_subprograms,
end_statement,
program_unit_context,
&constraint_checker,
&nodecl_body,
&nodecl_internal_subprograms);
}
if (nodecl_is_null(nodecl_body) && nodecl_is_null(nodecl_internal_subprograms))
{
nodecl_body = nodecl_make_list_1(nodecl_make_empty_statement(ast_get_locus(program_unit)));
}
else
{
nodecl_body = nodecl_concat_lists(nodecl_body, nodecl_internal_subprograms);
}
int i, num_params = symbol_entity_specs_get_num_related_symbols(new_entry);
for (i = 0; i < num_params; i++)
{
if (symbol_entity_specs_get_related_symbols_num(new_entry, i)->kind == SK_UNDEFINED)
{
symbol_entity_specs_get_related_symbols_num(new_entry, i)->kind = SK_VARIABLE;
}
}
nodecl_t function_code = nodecl_make_function_code(
nodecl_make_context(
nodecl_body,
program_unit_context,
ast_get_locus(program_unit)),
nodecl_null(),
new_entry,
ast_get_locus(program_unit));
if (!inside_interface(program_unit))
{
symbol_entity_specs_set_function_code(new_entry, function_code);
}
*nodecl_output = nodecl_make_list_1(function_code);
}
static void build_scope_module_program_unit(
AST program_unit,
const decl_context_t *decl_context,
scope_entry_t **program_unit_symbol,
nodecl_t *nodecl_output)
{
const decl_context_t* program_unit_context = new_program_unit_context(decl_context);
ERROR_CONDITION(program_unit_symbol == NULL, "Invalid parameter", 0)
DEBUG_CODE()
{
fprintf(stderr, "==== [%s] Program unit: MODULE ===\n", ast_location(program_unit));
}
AST module_stmt = ASTSon0(program_unit);
AST module_name = ASTSon0(module_stmt);
AST module_nature = ASTSon1(module_stmt);
scope_entry_t* new_entry = new_fortran_symbol_not_unknown(decl_context, ASTText(module_name));
new_entry->kind = SK_MODULE;
if (new_entry->decl_context->current_scope == decl_context->global_scope)
symbol_entity_specs_set_is_global_hidden(new_entry, 1);
if (module_nature != NULL)
{
if (strcasecmp(ASTText(module_nature), "intrinsic") == 0)
{
symbol_entity_specs_set_is_builtin(new_entry, 1);
}
else
{
error_printf_at(ast_get_locus(module_nature),
"invalid module nature. Only INTRINSIC is allowed\n");
}
}
new_entry->related_decl_context = program_unit_context;
new_entry->locus = ast_get_locus(module_stmt);
new_entry->defined = 1;
program_unit_context->current_scope->related_entry = new_entry;
AST module_body = ASTSon1(program_unit);
*program_unit_symbol = new_entry;
nodecl_t nodecl_body = nodecl_null();
nodecl_t nodecl_internal_subprograms = nodecl_null();
if (module_body != NULL)
{
AST statement_seq = ASTSon0(module_body);
AST internal_subprograms = ASTSon1(module_body);
statement_constraint_checker_t constraint_checker
= statement_constraint_checker_init(module_specification_part_allowed_statements);
build_scope_program_unit_body(statement_seq,
internal_subprograms,
NULL,
program_unit_context,
&constraint_checker,
&nodecl_body,
&nodecl_internal_subprograms);
}
if (nodecl_is_null(nodecl_internal_subprograms))
{
*nodecl_output = nodecl_make_list_1(
nodecl_make_object_init(new_entry, ast_get_locus(program_unit)));
}
else
{
*nodecl_output = nodecl_internal_subprograms;
}
*nodecl_output = nodecl_concat_lists(nodecl_body, *nodecl_output);
int i, num_symbols = symbol_entity_specs_get_num_related_symbols(new_entry);
for (i = 0; i < num_symbols; i++)
{
if (symbol_entity_specs_get_related_symbols_num(new_entry, i)->kind == SK_UNDEFINED)
{
symbol_entity_specs_get_related_symbols_num(new_entry, i)->kind = SK_VARIABLE;
}
if (symbol_entity_specs_get_access(
symbol_entity_specs_get_related_symbols_num(new_entry, i)) == AS_UNKNOWN)
{
if (symbol_entity_specs_get_access(new_entry) == AS_PRIVATE)
symbol_entity_specs_set_access(
symbol_entity_specs_get_related_symbols_num(new_entry, i), AS_PRIVATE);
else
symbol_entity_specs_set_access(
symbol_entity_specs_get_related_symbols_num(new_entry, i), AS_PUBLIC);
}
}
rb_tree_insert(CURRENT_COMPILED_FILE->module_file_cache, strtolower(new_entry->symbol_name), new_entry);
if (diagnostics_get_error_count() == 0)
{
dump_module_info(new_entry);
}
}
static void build_scope_block_data_program_unit(
AST program_unit,
const decl_context_t *decl_context,
scope_entry_t **program_unit_symbol,
nodecl_t *nodecl_output)
{
const decl_context_t* program_unit_context = new_program_unit_context(decl_context);
DEBUG_CODE()
{
fprintf(stderr, "=== [%s] Program unit: BLOCK DATA ===\n", ast_location(program_unit));
}
ERROR_CONDITION(program_unit_symbol == NULL, "Invalid parameter", 0)
DEBUG_CODE()
{
fprintf(stderr, "==== [%s] Program unit: PROGRAM ===\n", ast_location(program_unit));
}
AST program_stmt = ASTSon0(program_unit);
const char * program_name = "__BLOCK_DATA_UNNAMED__";
AST name = ASTSon0(program_stmt);
if (name != NULL)
program_name = ASTText(name);
scope_entry_t* program_sym = new_fortran_symbol_not_unknown(decl_context, program_name);
program_sym->kind = SK_BLOCKDATA;
program_sym->locus = ast_get_locus(program_unit);
if (program_sym->decl_context->current_scope == decl_context->global_scope)
symbol_entity_specs_set_is_global_hidden(program_sym, 1);
program_sym->related_decl_context = program_unit_context;
program_unit_context->current_scope->related_entry = program_sym;
*program_unit_symbol = program_sym;
AST program_body = ASTSon1(program_unit);
nodecl_t nodecl_body = nodecl_null();
nodecl_t nodecl_internal_subprograms = nodecl_null();
if (program_body != NULL)
{
AST statement_seq = program_body;
AST internal_subprograms = NULL;
statement_constraint_checker_t constraint_checker
= statement_constraint_checker_init(block_data_allowed_statements);
build_scope_program_unit_body(statement_seq,
internal_subprograms,
NULL,
program_unit_context,
&constraint_checker,
&nodecl_body,
&nodecl_internal_subprograms);
}
*nodecl_output = nodecl_make_list_1(
nodecl_make_object_init(program_sym, ast_get_locus(program_unit)));
}
static void build_global_program_unit(
AST program_unit)
{
decl_context_t* program_unit_context = decl_context_clone(CURRENT_COMPILED_FILE->global_decl_context);
program_unit_context->function_scope = program_unit_context->current_scope;
AST program_body = ASTSon0(program_unit);
AST statement_seq = ASTSon0(program_body);
nodecl_t nodecl_body = nodecl_null();
nodecl_t nodecl_internal_subprograms = nodecl_null();
statement_constraint_checker_t constraint_checker
= statement_constraint_checker_init(allow_all_statements);
build_scope_program_unit_body(
statement_seq,
NULL,
NULL,
program_unit_context, 
&constraint_checker, 
&nodecl_body, 
&nodecl_internal_subprograms);
}
static type_t* fortran_gather_type_from_declaration_type_spec_(AST a, 
const decl_context_t* decl_context, AST *character_length_out);
type_t* fortran_gather_type_from_declaration_type_spec(AST a, const decl_context_t* decl_context, AST *character_length_out)
{
return fortran_gather_type_from_declaration_type_spec_(a, decl_context, character_length_out);
}
static type_t *fortran_gather_type_from_type_spec(
AST a, const decl_context_t *decl_context)
{
AST character_length_out = NULL;
type_t *basic_type = fortran_gather_type_from_declaration_type_spec_(
a, decl_context, &character_length_out);
if (fortran_is_character_type(basic_type) && character_length_out != NULL)
{
char is_star = 0, is_colon = 0;
basic_type = compute_character_length_type(character_length_out,
basic_type,
decl_context,
&is_star,
&is_colon);
}
return basic_type;
}
static type_t* get_derived_type_name(AST a, const decl_context_t* decl_context);
static type_t* fortran_gather_type_from_declaration_type_spec_of_component(AST a, const decl_context_t* decl_context,
char is_pointer_component)
{
type_t* result = NULL;
if (is_pointer_component
&& ASTKind(a) == AST_TYPE_NAME)
{
result = get_derived_type_name(ASTSon0(a), decl_context);
if (result == NULL)
{
AST derived_type_name = ASTSon0(a);
AST name = ASTSon0(derived_type_name);
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
entry->kind = SK_CLASS;
add_delay_check_fully_defined_symbol(decl_context, entry);
result = get_user_defined_type(entry);
}
}
else
{
result = fortran_gather_type_from_declaration_type_spec_(a, decl_context,
NULL);
}
return result;
}
static void check_bind_spec(scope_entry_t *entry,
AST bind_spec,
const decl_context_t *decl_context);
typedef
struct implicit_update_info_tag
{
scope_entry_t* entry;
const decl_context_t* decl_context;
} implicit_update_info_t;
static void delay_update_implicit_type(void *data,
UNUSED_PARAMETER nodecl_t *nodecl_out);
static void add_delay_update_type_on_implicit_stmt(const decl_context_t *decl_context,
scope_entry_t *entry)
{
implicit_update_info_t *implicit_info = NEW0(implicit_update_info_t);
implicit_info->entry = entry;
implicit_info->decl_context = decl_context;
build_scope_delay_list_add(DELAY_AFTER_IMPLICIT_STATEMENT,
delay_update_implicit_type,
implicit_info);
}
typedef
struct check_has_type_spec_info_tag
{
scope_entry_t* entry;
} check_has_type_spec_info_t;
static void delay_check_has_type_spec(void *data,
UNUSED_PARAMETER nodecl_t *nodecl_out)
{
check_has_type_spec_info_t *info = (check_has_type_spec_info_t *)data;
scope_entry_t *entry = info->entry;
if (symbol_entity_specs_get_is_implicit_basic_type(entry)
&& fortran_basic_type_is_implicit_none(entry->type_information))
{
error_printf_at(entry->locus, "symbol '%s' has no IMPLICIT type\n",
entry->symbol_name);
}
DELETE(info);
}
static void add_delay_check_symbol_needs_type_specifier_(
build_scope_delay_category_t delay_category,
UNUSED_PARAMETER const decl_context_t *decl_context,
scope_entry_t *entry)
{
check_has_type_spec_info_t *check_has_type_spec_info
= NEW0(check_has_type_spec_info_t);
check_has_type_spec_info->entry = entry;
build_scope_delay_list_add(
delay_category, delay_check_has_type_spec, check_has_type_spec_info);
}
static void add_delay_check_symbol_needs_type_specifier(
UNUSED_PARAMETER const decl_context_t *decl_context, scope_entry_t *entry)
{
add_delay_check_symbol_needs_type_specifier_(DELAY_AFTER_DECLARATIONS,
decl_context, entry);
}
void add_delay_check_symbol_needs_type_specifier_at_end(
UNUSED_PARAMETER const decl_context_t *decl_context, scope_entry_t *entry)
{
add_delay_check_symbol_needs_type_specifier_(DELAY_AFTER_PROGRAM_UNIT,
decl_context, entry);
}
static scope_entry_t* new_procedure_symbol(
const decl_context_t* decl_context,
const decl_context_t* program_unit_context,
AST name, AST prefix, AST suffix, AST dummy_arg_name_list,
char is_function)
{
scope_entry_t* entry = NULL;
if (decl_context->current_scope != decl_context->global_scope)
{
scope_entry_list_t* entry_list = query_in_scope_str_flags(decl_context, strtolower(ASTText(name)), NULL, DF_ONLY_CURRENT_SCOPE);
if (entry_list != NULL)
{
entry = entry_list_head(entry_list);
entry_list_free(entry_list);
}
}
if (entry != NULL)
{
if (entry->decl_context->current_scope != decl_context->current_scope)
{
entry = NULL;
}
else if (entry->kind == SK_GENERIC_NAME)
{
entry = NULL;
}
else
{
if (entry->defined
|| (!symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry)
&& !symbol_entity_specs_get_is_module_procedure(entry)
&& !(entry->kind == SK_UNDEFINED
&& symbol_entity_specs_get_in_module(entry) != NULL)))
{
error_printf_at(ast_get_locus(name), "redeclaration of entity '%s'\n",
ASTText(name));
return NULL;
}
if (symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
entry->kind = SK_VARIABLE;
}
}
}
if (entry == NULL)
{
entry = new_fortran_symbol_not_unknown(decl_context, ASTText(name));
}
program_unit_context->current_scope->related_entry = entry;
if (entry->kind == SK_UNDEFINED)
entry->kind = SK_FUNCTION;
entry->locus = ast_get_locus(name);
symbol_entity_specs_set_is_implicit_basic_type(entry, 1);
add_delay_update_type_on_implicit_stmt(decl_context, entry);
entry->defined = 1;
if (entry->decl_context->current_scope == decl_context->global_scope)
symbol_entity_specs_set_is_global_hidden(entry, 1);
type_t* return_type = get_void_type();
if (is_function)
{
return_type = get_implicit_type_for_symbol(program_unit_context, entry->symbol_name);
}
else
{
symbol_entity_specs_set_is_implicit_basic_type(entry, 0);
}
if (prefix != NULL)
{
AST it;
for_each_element(prefix, it)
{
AST prefix_spec = ASTSon1(it);
ERROR_CONDITION(ASTKind(prefix_spec) != AST_ATTR_SPEC, "Invalid tree", 0);
const char* prefix_spec_str = ASTText(prefix_spec);
if (strcasecmp(prefix_spec_str, "__declaration__") == 0)
{
if (!is_function)
{
error_printf_at(ast_get_locus(prefix_spec), "declaration type-specifier is only valid for FUNCTION statement\n");
}
else
{
AST declaration_type_spec = ASTSon0(prefix_spec);
postponed_function_type_spec_t *postponed_info = NEW0(postponed_function_type_spec_t);
postponed_info->decl_context = program_unit_context;
postponed_info->postponed_function_type_spec = declaration_type_spec;
build_scope_delay_list_add(
DELAY_AFTER_USE_STATEMENT,
delayed_solve_postponed_function_type_spec,
postponed_info);
}
}
else if (strcasecmp(prefix_spec_str, "elemental") == 0)
{
symbol_entity_specs_set_is_elemental(entry, 1);
}
else if (strcasecmp(prefix_spec_str, "pure") == 0)
{
symbol_entity_specs_set_is_pure(entry, 1);
}
else if (strcasecmp(prefix_spec_str, "recursive") == 0)
{
symbol_entity_specs_set_is_recursive(entry, 1);
}
else if ((strcasecmp(prefix_spec_str, "impure") == 0)
|| (strcasecmp(prefix_spec_str, "module") == 0))
{
error_printf_at(ast_get_locus(prefix_spec),
"unsupported specifier for procedures '%s'\n",
fortran_prettyprint_in_buffer(prefix_spec));
}
else
{
internal_error("Invalid tree kind '%s' with spec '%s'\n", 
ast_print_node_type(ASTKind(prefix_spec)),
ASTText(prefix_spec));
}
}
if (symbol_entity_specs_get_is_elemental(entry)
&& symbol_entity_specs_get_is_recursive(entry))
{
error_printf_at(ast_get_locus(prefix), "RECURSIVE and ELEMENTAL cannot be specified at the same time\n");
}
}
int num_dummy_arguments = 0;
if (dummy_arg_name_list != NULL)
{
int num_alternate_returns = 0;
AST it;
for_each_element(dummy_arg_name_list, it)
{
AST dummy_arg_name = ASTSon1(it);
scope_entry_t* dummy_arg = NULL;
if (strcmp(ASTText(dummy_arg_name), "*") == 0)
{
if (is_function)
{
error_printf_at(ast_get_locus(dummy_arg_name), "alternate return is not allowed in a FUNCTION specification\n");
continue;
}
char alternate_return_name[64];
snprintf(alternate_return_name, 64, ".alternate-return-%d", num_alternate_returns);
alternate_return_name[63] = '\0';
dummy_arg = NEW0(scope_entry_t);
dummy_arg->symbol_name = uniquestr(alternate_return_name);
dummy_arg->kind = SK_LABEL;
dummy_arg->type_information = get_void_type();
dummy_arg->decl_context = program_unit_context;
num_alternate_returns++;
}
else
{
dummy_arg = get_symbol_for_name(program_unit_context, dummy_arg_name, ASTText(dummy_arg_name));
if (dummy_arg->type_information != NULL
&& is_lvalue_reference_type(dummy_arg->type_information))
{
error_printf_at(ast_get_locus(dummy_arg_name),
"duplicated '%s' dummy argument\n", fortran_prettyprint_in_buffer(dummy_arg_name));
continue;
}
dummy_arg->type_information = get_lvalue_reference_type(dummy_arg->type_information);
add_delay_update_type_on_implicit_stmt(program_unit_context, dummy_arg);
}
dummy_arg->locus = ast_get_locus(dummy_arg_name);
symbol_set_as_parameter_of_function(dummy_arg, entry,
0,
symbol_entity_specs_get_num_related_symbols(entry));
symbol_entity_specs_add_related_symbols(entry, dummy_arg);
num_dummy_arguments++;
}
}
AST result = NULL;
AST bind_spec = NULL;
if (suffix != NULL)
{
bind_spec = ASTSon0(suffix);
result = ASTSon1(suffix);
}
scope_entry_t* result_sym = NULL;
if (result != NULL)
{
if (!is_function)
{
error_printf_at(ast_get_locus(result), "RESULT is only valid for FUNCTION statement\n");
}
else
{
result_sym = get_symbol_for_name(program_unit_context, result, ASTText(result));
result_sym->kind = SK_VARIABLE;
result_sym->locus = ast_get_locus(result);
symbol_entity_specs_set_is_result_var(result_sym, 1);
symbol_entity_specs_set_is_implicit_basic_type(result_sym, 1);
add_delay_update_type_on_implicit_stmt(program_unit_context, result_sym);
result_sym->type_information = return_type;
return_type = get_mutable_indirect_type(result_sym);
symbol_entity_specs_set_result_var(entry, result_sym);
if (strcasecmp(ASTText(result), entry->symbol_name) == 0)
{
error_printf_at(ast_get_locus(result), "RESULT name is the same as the FUNCTION name\n");
}
else
{
insert_entry(program_unit_context->current_scope, entry);
}
}
}
else if (is_function)
{
result_sym = new_symbol(program_unit_context, program_unit_context->current_scope, entry->symbol_name);
result_sym->kind = SK_VARIABLE;
result_sym->locus = entry->locus;
symbol_entity_specs_set_is_result_var(result_sym, 1);
result_sym->type_information = return_type;
symbol_entity_specs_set_is_implicit_basic_type(result_sym, 1);
add_delay_update_type_on_implicit_stmt(program_unit_context, result_sym);
return_type = get_mutable_indirect_type(result_sym);
symbol_entity_specs_set_result_var(entry, result_sym);
}
else if (!is_function)
{
insert_entry(program_unit_context->current_scope, entry);
}
else
{
internal_error("Code unreachable", 0);
}
if (bind_spec != NULL)
{
check_bind_spec(entry, bind_spec, decl_context);
}
parameter_info_t parameter_info[num_dummy_arguments + 1];
memset(parameter_info, 0, sizeof(parameter_info));
int i;
for (i = 0; i < num_dummy_arguments; i++)
{
parameter_info[i].type_info = get_mutable_indirect_type(symbol_entity_specs_get_related_symbols_num(entry, i));
}
type_t* function_type = get_new_function_type(return_type, parameter_info, num_dummy_arguments,
REF_QUALIFIER_NONE);
entry->type_information = function_type;
if (symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
entry->type_information = get_lvalue_reference_type(entry->type_information);
}
symbol_entity_specs_set_is_implicit_basic_type(entry, 0);
entry->related_decl_context = program_unit_context;
if (program_unit_context->current_scope->contained_in != NULL)
{
scope_entry_t* enclosing_symbol = program_unit_context->current_scope->contained_in->related_entry;
if (enclosing_symbol != NULL)
{
if (enclosing_symbol->kind == SK_MODULE)
{
symbol_entity_specs_set_in_module(entry, enclosing_symbol);
symbol_entity_specs_add_related_symbols(enclosing_symbol, entry);
}
}
}
return entry;
}
static scope_entry_t* new_entry_symbol(const decl_context_t* decl_context, 
AST name, AST suffix, AST dummy_arg_name_list,
scope_entry_t* principal_procedure)
{
char is_function = !is_void_type(function_type_get_return_type(principal_procedure->type_information));
if (symbol_entity_specs_get_is_nested_function(principal_procedure))
{
error_printf_at(ast_get_locus(name), "internal subprograms cannot have an alternate ENTRY\n");
return NULL;
}
scope_entry_t* existing_name = NULL;
existing_name = fortran_query_name_str(principal_procedure->decl_context, ASTText(name), ast_get_locus(name));
if (existing_name != NULL)
{
if (existing_name->defined
|| ( 
!symbol_entity_specs_get_is_module_procedure(existing_name)
&& !(existing_name->kind == SK_UNDEFINED
&& symbol_entity_specs_get_in_module(existing_name) != NULL)))
{
error_printf_at(ast_get_locus(name), "redeclaration of entity '%s'\n", 
ASTText(name));
return NULL;
}
}
scope_entry_t* entry = existing_name;
if (entry == NULL)
{
entry = new_symbol(principal_procedure->decl_context, 
principal_procedure->decl_context->current_scope,
strtolower(ASTText(name)));
}
entry->decl_context = decl_context;
entry->kind = SK_FUNCTION;
entry->locus = ast_get_locus(name);
symbol_entity_specs_set_is_entry(entry, 1);
symbol_entity_specs_set_is_implicit_basic_type(entry, 1);
add_delay_update_type_on_implicit_stmt(principal_procedure->decl_context, entry);
entry->defined = 1;
if (principal_procedure->decl_context->current_scope == principal_procedure->decl_context->global_scope)
{
symbol_entity_specs_set_is_global_hidden(entry, 1);
}
symbol_entity_specs_set_is_recursive(entry, symbol_entity_specs_get_is_recursive(principal_procedure));
symbol_entity_specs_set_is_pure(entry, symbol_entity_specs_get_is_pure(principal_procedure));
symbol_entity_specs_set_is_elemental(entry, symbol_entity_specs_get_is_elemental(principal_procedure));
type_t* return_type = get_void_type();
if (is_function)
{
if (existing_name != NULL)
{
return_type = existing_name->type_information;
symbol_entity_specs_set_is_implicit_basic_type(entry, 0);
}
else
{
return_type = get_implicit_type_for_symbol(decl_context, entry->symbol_name);
}
}
else
{
symbol_entity_specs_set_is_implicit_basic_type(entry, 0);
}
int num_dummy_arguments = 0;
if (dummy_arg_name_list != NULL)
{
int num_alternate_returns = 0;
AST it;
for_each_element(dummy_arg_name_list, it)
{
AST dummy_arg_name = ASTSon1(it);
scope_entry_t* dummy_arg = NULL;
if (strcmp(ASTText(dummy_arg_name), "*") == 0)
{
if (is_function)
{
error_printf_at(ast_get_locus(dummy_arg_name), "alternate return is not allowed in a FUNCTION specification\n");
continue;
}
char alternate_return_name[64];
snprintf(alternate_return_name, 64, ".alternate-return-%d", num_alternate_returns);
alternate_return_name[63] = '\0';
dummy_arg = NEW0(scope_entry_t);
dummy_arg->symbol_name = uniquestr(alternate_return_name);
dummy_arg->kind = SK_LABEL;
dummy_arg->type_information = get_void_type();
dummy_arg->decl_context = decl_context;
num_alternate_returns++;
}
else
{
dummy_arg = get_symbol_for_name(decl_context, dummy_arg_name, ASTText(dummy_arg_name));
if (!is_lvalue_reference_type(dummy_arg->type_information)) 
{
dummy_arg->type_information = get_lvalue_reference_type(dummy_arg->type_information);
}
remove_intent_declared_symbol(dummy_arg);
}
dummy_arg->locus = ast_get_locus(dummy_arg_name);
symbol_set_as_parameter_of_function(dummy_arg, entry,
0,
symbol_entity_specs_get_num_related_symbols(entry));
symbol_entity_specs_add_related_symbols(entry, dummy_arg);
num_dummy_arguments++;
}
}
AST result = NULL;
AST bind_spec = NULL;
if (suffix != NULL)
{
bind_spec = ASTSon0(suffix);
result = ASTSon1(suffix);
}
scope_entry_t* result_sym = NULL;
if (result != NULL)
{
if (!is_function)
{
error_printf_at(ast_get_locus(result), "RESULT is not valid in an ENTRY of a SUBROUTINE\n");
}
else
{
result_sym = get_symbol_for_name(decl_context, result, ASTText(result));
ERROR_CONDITION(result_sym == existing_name, "Wrong symbol found", 0);
result_sym->kind = SK_VARIABLE;
result_sym->locus = ast_get_locus(result);
symbol_entity_specs_set_is_result_var(result_sym, 1);
if (symbol_entity_specs_get_is_implicit_basic_type(result_sym))
{
result_sym->type_information = get_lvalue_reference_type(return_type);
}
return_type = get_mutable_indirect_type(result_sym);
symbol_entity_specs_set_result_var(entry, result_sym);
if (strcasecmp(entry->symbol_name, result_sym->symbol_name) == 0)
{
error_printf_at(ast_get_locus(result), "RESULT name is the same as ENTRY name\n");
}
else
{
insert_entry(decl_context->current_scope, entry);
}
}
}
else if (is_function)
{
result_sym = get_symbol_for_name(decl_context, name, entry->symbol_name);
result_sym->kind = SK_VARIABLE;
result_sym->locus = entry->locus;
symbol_entity_specs_set_is_result_var(result_sym, 1);
if (symbol_entity_specs_get_is_implicit_basic_type(result_sym))
{
result_sym->type_information = get_lvalue_reference_type(return_type);
}
return_type = get_mutable_indirect_type(result_sym);
symbol_entity_specs_set_result_var(entry, result_sym);
}
else if (!is_function)
{
insert_entry(decl_context->current_scope, entry);
}
else
{
internal_error("Code unreachable", 0);
}
if (bind_spec != NULL)
{
check_bind_spec(entry, bind_spec, decl_context);
}
parameter_info_t parameter_info[num_dummy_arguments + 1];
memset(parameter_info, 0, sizeof(parameter_info));
int i;
for (i = 0; i < num_dummy_arguments; i++)
{
parameter_info[i].type_info = get_mutable_indirect_type(symbol_entity_specs_get_related_symbols_num(entry, i));
}
type_t* function_type = get_new_function_type(return_type, parameter_info, num_dummy_arguments,
REF_QUALIFIER_NONE);
entry->type_information = function_type;
symbol_entity_specs_set_is_implicit_basic_type(entry, 0);
entry->related_decl_context = decl_context;
if (symbol_entity_specs_get_is_module_procedure(principal_procedure))
{
scope_entry_t * sym_module = symbol_entity_specs_get_in_module(principal_procedure);
symbol_entity_specs_set_is_module_procedure(entry, 1);
symbol_entity_specs_set_in_module(entry, sym_module);
symbol_entity_specs_add_related_symbols(sym_module, entry);
}
return entry;
}
static void build_scope_ambiguity_statement(AST ambig_stmt, const decl_context_t* decl_context, char is_declaration);
static void fortran_build_scope_statement(AST statement, const decl_context_t* decl_context, nodecl_t* nodecl_output);
static void build_scope_program_unit_body_declarations(
statement_constraint_checker_t *constraint_checker,
AST program_unit_stmts,
const decl_context_t* decl_context,
AST *first_executable_statement,
nodecl_t* nodecl_output)
{
char still_possible_use_stmt = 1;
char seen_implicit = 0;
char still_possible_implicit_stmt = 1;
if (program_unit_stmts != NULL)
{
AST it;
for_each_element(program_unit_stmts, it)
{
AST stmt = ASTSon1(it);
if (ASTKind(stmt) == AST_AMBIGUITY)
{
build_scope_ambiguity_statement(stmt, decl_context,  1);
}
if (statement_get_order_class(stmt) == SOC_EXECUTABLE)
{
*first_executable_statement = it;
break;
}
if (!statement_constraint_checker_check_statement(
constraint_checker, stmt, decl_context))
continue;
if (still_possible_use_stmt
&& !(*constraint_checker->current_order_class & SOC_USE))
{
still_possible_use_stmt = 0;
build_scope_delay_list_run(DELAY_AFTER_USE_STATEMENT,
NULL);
}
if (statement_get_order_class(stmt) == SOC_IMPLICIT
|| statement_get_order_class(stmt) == SOC_IMPLICIT_NONE)
{
seen_implicit = 1;
}
if (seen_implicit && still_possible_implicit_stmt
&& !(*constraint_checker->current_order_class & SOC_IMPLICIT)
&& !(*constraint_checker->current_order_class
& SOC_IMPLICIT_NONE))
{
still_possible_implicit_stmt = 0;
build_scope_delay_list_run(DELAY_AFTER_IMPLICIT_STATEMENT,
NULL);
}
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement(
stmt, decl_context, &nodecl_statement);
*nodecl_output
= nodecl_concat_lists(*nodecl_output, nodecl_statement);
}
}
if (still_possible_use_stmt)
{
still_possible_use_stmt = 0;
build_scope_delay_list_run(DELAY_AFTER_USE_STATEMENT,
NULL);
}
if (seen_implicit && still_possible_implicit_stmt)
{
still_possible_implicit_stmt = 0;
build_scope_delay_list_run(DELAY_AFTER_IMPLICIT_STATEMENT,
NULL);
}
}
static void build_scope_program_unit_body_executable(
statement_constraint_checker_t *constraint_checker,
AST program_unit_stmts,
AST first_executable_statement,
AST end_statement,
const decl_context_t* decl_context,
nodecl_t* nodecl_output)
{
AST it;
if (program_unit_stmts != NULL)
{
for_each_element(program_unit_stmts, it)
{
if (it == first_executable_statement)
break;
AST stmt = ASTSon1(it);
if (statement_get_order_class(stmt) == SOC_ENTRY)
{
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement(
stmt, decl_context, &nodecl_statement);
*nodecl_output
= nodecl_concat_lists(*nodecl_output, nodecl_statement);
}
}
}
if (first_executable_statement != NULL)
{
for_each_element_in_range(first_executable_statement, program_unit_stmts, it)
{
AST stmt = ASTSon1(it);
if (ASTKind(stmt) == AST_AMBIGUITY)
{
build_scope_ambiguity_statement(stmt, decl_context,  0);
}
if (!statement_constraint_checker_check_statement(
constraint_checker, stmt, decl_context))
continue;
if (statement_get_order_class(stmt) == SOC_ENTRY)
{
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement(
stmt, decl_context, &nodecl_statement);
ERROR_CONDITION(
!nodecl_is_null(nodecl_statement),
"ENTRY statement generated a nodecl when it didn't have to",
0);
}
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement(
stmt, decl_context, &nodecl_statement);
*nodecl_output
= nodecl_concat_lists(*nodecl_output, nodecl_statement);
}
}
if (end_statement != NULL
&& (ASTKind(end_statement) == AST_LABELED_STATEMENT))
{
AST label = ASTSon0(end_statement);
scope_entry_t* label_sym = fortran_query_label(label, decl_context,  1);
*nodecl_output = nodecl_append_to_list(
*nodecl_output,
nodecl_make_labeled_statement(
nodecl_make_list_1(nodecl_make_empty_statement(ast_get_locus(end_statement))),
label_sym,
ast_get_locus(end_statement)));
}
}
typedef
struct internal_subprograms_info_tag
{
scope_entry_t* symbol;
const decl_context_t* decl_context;
nodecl_t nodecl_output;
nodecl_t nodecl_pragma;
AST program_unit_stmts;
AST end_statement;
AST internal_subprograms;
AST first_executable_statement;
const locus_t* locus;
statement_constraint_checker_t constraint_checker;
build_scope_delay_list_t delayed_list;
} internal_subprograms_info_t;
static int count_internal_subprograms(AST internal_subprograms)
{
int num_internal_program_units = 0;
if (internal_subprograms != NULL)
{
AST it;
for_each_element(internal_subprograms, it)
{
num_internal_program_units++;
}
}
return num_internal_program_units;
}
static scope_entry_t* build_scope_internal_subprogram(
AST subprogram,
const decl_context_t* decl_context,
internal_subprograms_info_t* internal_subprograms_info)
{
build_scope_delay_list_push(&internal_subprograms_info->delayed_list);
const decl_context_t* subprogram_unit_context = new_internal_program_unit_context(decl_context);
scope_entry_t* new_entry = NULL;
switch (ASTKind(subprogram))
{
case AST_SUBROUTINE_PROGRAM_UNIT:
{
new_entry = register_subroutine(subprogram, decl_context, subprogram_unit_context);
break;
}
case AST_FUNCTION_PROGRAM_UNIT:
{
new_entry = register_function(subprogram, decl_context, subprogram_unit_context);
break;
}
case AST_PRAGMA_CUSTOM_CONSTRUCT:
{
AST pragma_line = ASTSon0(subprogram);
AST internal_subprogram = ASTSon1(subprogram);
new_entry = build_scope_internal_subprogram(
internal_subprogram, 
decl_context,
internal_subprograms_info);
if (new_entry != NULL)
{
const decl_context_t* context_in_scope = new_entry->related_decl_context;
nodecl_t nodecl_pragma_line = nodecl_null();
common_build_scope_pragma_custom_line(pragma_line,  NULL, context_in_scope, &nodecl_pragma_line);
nodecl_t nodecl_pragma_declaration = 
nodecl_make_pragma_custom_declaration(nodecl_pragma_line, 
internal_subprograms_info->nodecl_pragma,
nodecl_make_pragma_context(context_in_scope, ast_get_locus(subprogram)),
nodecl_make_pragma_context(context_in_scope, ast_get_locus(subprogram)),
new_entry,
strtolower(ASTText(subprogram)),
ast_get_locus(subprogram));
internal_subprograms_info->nodecl_pragma = nodecl_pragma_declaration;
}
break;
}
case AST_UNKNOWN_PRAGMA:
{
break;
}
default:
{
internal_error("Unexpected node of kind %s\n", ast_print_node_type(ASTKind(subprogram)));
}
}
if ((ASTKind(subprogram) == AST_SUBROUTINE_PROGRAM_UNIT
|| ASTKind(subprogram) == AST_FUNCTION_PROGRAM_UNIT)
&& (new_entry != NULL))
{
AST program_body = ASTSon1(subprogram);
AST end_statement = ASTSon2(subprogram);
AST program_part = ASTSon0(program_body);
AST n_internal_subprograms = ASTSon1(program_body);
AST program_unit_stmts = ASTSon0(program_part);
internal_subprograms_info->symbol = new_entry;
internal_subprograms_info->decl_context = subprogram_unit_context;
internal_subprograms_info->program_unit_stmts = program_unit_stmts;
internal_subprograms_info->end_statement = end_statement;
internal_subprograms_info->internal_subprograms
= n_internal_subprograms;
internal_subprograms_info->locus = ast_get_locus(program_body);
scope_entry_t* enclosing_sym = decl_context->current_scope->related_entry;
if (enclosing_sym != NULL
&& enclosing_sym->kind == SK_MODULE)
{
internal_subprograms_info->constraint_checker
= statement_constraint_checker_init(allow_all_statements);
symbol_entity_specs_set_is_module_procedure(new_entry, 1);
}
else
{
internal_subprograms_info->constraint_checker
= statement_constraint_checker_init(internal_subprogram_allowed_statements);
symbol_entity_specs_set_is_nested_function(new_entry, 1);
}
build_scope_program_unit_body_declarations(
&internal_subprograms_info->constraint_checker,
internal_subprograms_info->program_unit_stmts, 
internal_subprograms_info->decl_context,
&internal_subprograms_info->first_executable_statement,
&(internal_subprograms_info->nodecl_output));
}
build_scope_delay_list_pop();
return new_entry;
}
static void build_scope_program_unit_body_internal_subprograms_declarations(
AST internal_subprograms, 
int num_internal_program_units,
internal_subprograms_info_t *internal_subprograms_info,
const decl_context_t* decl_context)
{
if (internal_subprograms == NULL)
return;
int i = 0;
AST it;
for_each_element(internal_subprograms, it)
{
ERROR_CONDITION(i >= num_internal_program_units, "Too many internal subprograms", 0);
AST subprogram = ASTSon1(it);
build_scope_internal_subprogram(subprogram,
decl_context, 
&internal_subprograms_info[i]);
i++;
}
}
static void build_scope_program_unit_body_internal_subprograms_executable(
AST internal_subprograms,
int num_internal_program_units,
internal_subprograms_info_t *internal_subprograms_info,
const decl_context_t* decl_context UNUSED_PARAMETER)
{
if (internal_subprograms == NULL)
return;
ERROR_CONDITION(num_internal_program_units < 0,
"Invalid number of internal program units",
0);
internal_subprograms_info_t
*n_internal_subprograms_info[num_internal_program_units + 1];
int n_num_internal_program_units[num_internal_program_units + 1];
int i = 0;
AST it;
for_each_element(internal_subprograms, it)
{
ERROR_CONDITION(i >= num_internal_program_units, "Too many internal program units", 0);
if (internal_subprograms_info[i].symbol != NULL)
{
AST n_internal_subprograms
= internal_subprograms_info[i].internal_subprograms;
n_num_internal_program_units[i]
= count_internal_subprograms(n_internal_subprograms);
n_internal_subprograms_info[i] = NEW_VEC0(
internal_subprograms_info_t, n_num_internal_program_units[i]);
}
i++;
}
i = 0;
for_each_element(internal_subprograms, it)
{
ERROR_CONDITION(i >= num_internal_program_units, "Too many internal program units", 0);
build_scope_delay_list_push(&internal_subprograms_info[i].delayed_list);
if (internal_subprograms_info[i].symbol != NULL)
{
AST n_internal_subprograms = internal_subprograms_info[i].internal_subprograms;
build_scope_program_unit_body_internal_subprograms_declarations(
n_internal_subprograms, 
n_num_internal_program_units[i],
n_internal_subprograms_info[i],
internal_subprograms_info[i].decl_context);
}
build_scope_delay_list_pop();
i++;
}
i = 0;
for_each_element(internal_subprograms, it)
{
ERROR_CONDITION(i >= num_internal_program_units, "Too many internal program units", 0);
build_scope_delay_list_push(&internal_subprograms_info[i].delayed_list);
if (internal_subprograms_info[i].symbol != NULL)
{
build_scope_delay_list_run(
DELAY_AFTER_DECLARATIONS,
&(internal_subprograms_info[i].nodecl_output));
}
build_scope_delay_list_pop();
i++;
}
i = 0;
for_each_element(internal_subprograms, it)
{
ERROR_CONDITION(i >= num_internal_program_units, "Too many internal program units", 0);
build_scope_delay_list_push(&internal_subprograms_info[i].delayed_list);
if (internal_subprograms_info[i].symbol != NULL)
{
AST n_internal_subprograms
= internal_subprograms_info[i].internal_subprograms;
build_scope_program_unit_body_executable(
&internal_subprograms_info[i].constraint_checker,
internal_subprograms_info[i].program_unit_stmts,
internal_subprograms_info[i].first_executable_statement,
internal_subprograms_info[i].end_statement,
internal_subprograms_info[i].decl_context,
&(internal_subprograms_info[i].nodecl_output));
build_scope_delay_list_run(
DELAY_AFTER_EXECUTABLE_STATEMENTS,
&(internal_subprograms_info[i].nodecl_output));
build_scope_program_unit_body_internal_subprograms_executable(
n_internal_subprograms, 
n_num_internal_program_units[i],
n_internal_subprograms_info[i],
internal_subprograms_info[i].decl_context);
build_scope_delay_list_run(
DELAY_AFTER_PROGRAM_UNIT,
&(internal_subprograms_info[i].nodecl_output));
nodecl_t nodecl_internal_subprograms = nodecl_null();
int j;
for (j = 0; j < n_num_internal_program_units[i]; j++)
{
nodecl_internal_subprograms =
nodecl_append_to_list(nodecl_internal_subprograms, 
n_internal_subprograms_info[i][j].nodecl_output);
if (!nodecl_is_null(n_internal_subprograms_info[i][j].nodecl_pragma))
{
nodecl_internal_subprograms =
nodecl_append_to_list(nodecl_internal_subprograms, 
n_internal_subprograms_info[i][j].nodecl_pragma);
}
}
scope_entry_t* function_symbol = internal_subprograms_info[i].symbol;
int num_params = symbol_entity_specs_get_num_related_symbols(function_symbol);
for (j = 0; j < num_params; j++)
{
if (symbol_entity_specs_get_related_symbols_num(function_symbol, j)->kind == SK_UNDEFINED)
{
symbol_entity_specs_get_related_symbols_num(function_symbol, j)->kind = SK_VARIABLE;
}
}
nodecl_t nodecl_statements = internal_subprograms_info[i].nodecl_output;
if (nodecl_is_null(nodecl_statements) && nodecl_is_null(nodecl_internal_subprograms))
{
nodecl_statements = nodecl_make_list_1(
nodecl_make_empty_statement(
internal_subprograms_info[i].locus));
}
else
{
nodecl_statements = nodecl_concat_lists(nodecl_statements, nodecl_internal_subprograms);
}
nodecl_t function_code = 
nodecl_make_function_code(
nodecl_make_context(
nodecl_statements,
internal_subprograms_info[i].decl_context,
internal_subprograms_info[i].locus),
nodecl_null(),
internal_subprograms_info[i].symbol,
internal_subprograms_info[i].locus);
symbol_entity_specs_set_function_code(internal_subprograms_info[i].symbol, function_code);
internal_subprograms_info[i].nodecl_output = function_code;
}
build_scope_delay_list_pop();
i++;
}
i = 0;
for_each_element(internal_subprograms, it)
{
ERROR_CONDITION(i >= num_internal_program_units,
"Too many internal program units",
0);
if (internal_subprograms_info[i].symbol != NULL)
{
DELETE(n_internal_subprograms_info[i]);
}
i++;
}
}
static void build_scope_program_unit_body(
AST program_unit_stmts,
AST internal_subprograms,
AST end_statement,
const decl_context_t* decl_context,
statement_constraint_checker_t* constraint_checker,
nodecl_t* nodecl_output,
nodecl_t* nodecl_internal_subprograms)
{
AST first_executable_statement = NULL;
build_scope_program_unit_body_declarations(
constraint_checker,
program_unit_stmts, 
decl_context, 
&first_executable_statement,
nodecl_output);
int num_internal_program_units = count_internal_subprograms(internal_subprograms);
internal_subprograms_info_t internal_program_units_info[num_internal_program_units + 1];
memset(internal_program_units_info, 0, sizeof(internal_program_units_info));
build_scope_program_unit_body_internal_subprograms_declarations(
internal_subprograms, 
num_internal_program_units,
internal_program_units_info,
decl_context);
build_scope_delay_list_run(DELAY_AFTER_DECLARATIONS, nodecl_output);
build_scope_program_unit_body_executable(
constraint_checker,
program_unit_stmts,
first_executable_statement, 
end_statement,
decl_context,
nodecl_output);
build_scope_delay_list_run(DELAY_AFTER_EXECUTABLE_STATEMENTS, nodecl_output);
build_scope_program_unit_body_internal_subprograms_executable(
internal_subprograms, 
num_internal_program_units,
internal_program_units_info,
decl_context);
build_scope_delay_list_run(DELAY_AFTER_PROGRAM_UNIT, nodecl_output);
int i;
for (i = 0; i < num_internal_program_units; i++)
{
*nodecl_internal_subprograms =
nodecl_append_to_list(*nodecl_internal_subprograms, 
internal_program_units_info[i].nodecl_output);
if (!nodecl_is_null(internal_program_units_info[i].nodecl_pragma))
{
*nodecl_internal_subprograms =
nodecl_append_to_list(*nodecl_internal_subprograms, 
internal_program_units_info[i].nodecl_pragma);
}
}
}
typedef void (*build_scope_statement_function_t)(AST statement, const decl_context_t*, nodecl_t* nodecl_output);
static statement_order_class_t get_soc_from_son_1(AST a);
typedef struct build_scope_statement_handler_tag
{
node_t ast_kind;
build_scope_statement_function_t handler;
statement_order_class_t (*get_order_class)(AST);
} build_scope_statement_handler_t;
#define STATEMENT_HANDLER_TABLE \
STATEMENT_HANDLER(AST_ACCESS_STATEMENT,              build_scope_access_stmt,           GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_ALLOCATABLE_STATEMENT,         build_scope_allocatable_stmt,      GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_ALLOCATE_STATEMENT,            build_scope_allocate_stmt,         GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_ALL_STOP_STATEMENT,            build_scope_allstop_stmt,          GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_ARITHMETIC_IF_STATEMENT,       build_scope_arithmetic_if_stmt,    GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_EXPRESSION_STATEMENT,          build_scope_expression_stmt,       GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_ASSOCIATE_CONSTRUCT,           build_scope_associate_construct,   GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_ASYNCHRONOUS_STATEMENT,        build_scope_asynchronous_stmt,     GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_IO_STATEMENT,                  build_io_stmt,                     GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_BIND_STATEMENT,                build_scope_bind_stmt,             GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_BLOCK_CONSTRUCT,               build_scope_block_construct,       GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_SWITCH_STATEMENT,              build_scope_case_construct,        GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_CASE_STATEMENT,                build_scope_case_statement,        GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_DEFAULT_STATEMENT,             build_scope_default_statement,     GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_CLOSE_STATEMENT,               build_scope_close_stmt,            GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_CODIMENSION_STATEMENT,         build_scope_codimension_stmt,      GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_COMMON_STATEMENT,              build_scope_common_stmt,           GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_COMPOUND_STATEMENT,            build_scope_compound_statement,    GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_COMPUTED_GOTO_STATEMENT,       build_scope_computed_goto_stmt,    GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_ASSIGNED_GOTO_STATEMENT,       build_scope_assigned_goto_stmt,    GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_LABEL_ASSIGN_STATEMENT,        build_scope_label_assign_stmt,     GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_LABELED_STATEMENT,             build_scope_labeled_stmt,          get_soc_from_son_1) \
STATEMENT_HANDLER(AST_EMPTY_STATEMENT,               build_scope_continue_stmt,         GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_CRITICAL_CONSTRUCT,            build_scope_critical_construct,    GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_CONTINUE_STATEMENT,            build_scope_cycle_stmt,            GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_DATA_STATEMENT,                build_scope_data_stmt,             GET_SOC_DATA) \
STATEMENT_HANDLER(AST_DEALLOCATE_STATEMENT,          build_scope_deallocate_stmt,       GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_DERIVED_TYPE_DEF,              build_scope_derived_type_def,      GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_DIMENSION_STATEMENT,           build_scope_dimension_stmt,        GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_FOR_STATEMENT,                 build_scope_do_construct,          GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_ENUM_DEF,                      build_scope_enum_def,              GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_EQUIVALENCE_STATEMENT,         build_scope_equivalence_stmt,      GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_BREAK_STATEMENT,               build_scope_exit_stmt,             GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_EXTERNAL_STATEMENT,            build_scope_external_stmt,         GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_FORALL_CONSTRUCT,              build_scope_forall_construct,      GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_FORALL_STATEMENT,              build_scope_forall_stmt,           GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_FORMAT_STATEMENT,              build_scope_format_stmt,           GET_SOC_FORMAT) \
STATEMENT_HANDLER(AST_GOTO_STATEMENT,                build_scope_goto_stmt,             GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_IF_ELSE_STATEMENT,             build_scope_if_construct,          GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_IMPLICIT_STATEMENT,            build_scope_implicit_stmt,         GET_SOC_IMPLICIT) \
STATEMENT_HANDLER(AST_IMPLICIT_NONE_STATEMENT,       build_scope_implicit_stmt,         GET_SOC_IMPLICIT_NONE) \
STATEMENT_HANDLER(AST_IMPORT_STATEMENT,              build_scope_import_stmt,           GET_SOC_IMPORT) \
STATEMENT_HANDLER(AST_INTENT_STATEMENT,              build_scope_intent_stmt,           GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_INTERFACE_BLOCK,               build_scope_interface_block,       GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_INTRINSIC_STATEMENT,           build_scope_intrinsic_stmt,        GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_LOCK_STATEMENT,                build_scope_lock_stmt,             GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_NAMELIST_STATEMENT,            build_scope_namelist_stmt,         GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_NULLIFY_STATEMENT,             build_scope_nullify_stmt,          GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_OPEN_STATEMENT,                build_scope_open_stmt,             GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_OPTIONAL_STATEMENT,            build_scope_optional_stmt,         GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_PARAMETER_STATEMENT,           build_scope_parameter_stmt,        GET_SOC_PARAMETER) \
STATEMENT_HANDLER(AST_CRAY_POINTER_STATEMENT,        build_scope_cray_pointer_stmt,     GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_POINTER_STATEMENT,             build_scope_pointer_stmt,          GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_PRINT_STATEMENT,               build_scope_print_stmt,            GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_PROCEDURE_DECL_STATEMENT,      build_scope_procedure_decl_stmt,   GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_PROTECTED_STATEMENT,           build_scope_protected_stmt,        GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_READ_STATEMENT,                build_scope_read_stmt,             GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_RETURN_STATEMENT,              build_scope_return_stmt,           GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_SAVE_STATEMENT,                build_scope_save_stmt,             GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_SELECT_TYPE_CONSTRUCT,         build_scope_select_type_construct, GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_STATEMENT_FUNCTION_STATEMENT,  build_scope_stmt_function_stmt,    GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_STOP_STATEMENT,                build_scope_stop_stmt,             GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_PAUSE_STATEMENT,               build_scope_pause_stmt,            GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_SYNC_ALL_STATEMENT,            build_scope_sync_all_stmt,         GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_SYNC_IMAGES_STATEMENT,         build_scope_sync_images_stmt,      GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_SYNC_MEMORY_STATEMENT,         build_scope_sync_memory_stmt,      GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_TARGET_STATEMENT,              build_scope_target_stmt,           GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_DECLARATION_STATEMENT,         build_scope_declaration_stmt,      GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_UNLOCK_STATEMENT,              build_scope_unlock_stmt,           GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_USE_STATEMENT,                 build_scope_use_stmt,              GET_SOC_USE) \
STATEMENT_HANDLER(AST_USE_ONLY_STATEMENT,            build_scope_use_stmt,              GET_SOC_USE) \
STATEMENT_HANDLER(AST_VALUE_STATEMENT,               build_scope_value_stmt,            GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_VOLATILE_STATEMENT,            build_scope_volatile_stmt,         GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_WAIT_STATEMENT,                build_scope_wait_stmt,             GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_WHERE_CONSTRUCT,               build_scope_where_construct,       GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_WHERE_STATEMENT,               build_scope_where_stmt,            GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_WHILE_STATEMENT,               build_scope_while_stmt,            GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_WRITE_STATEMENT,               build_scope_write_stmt,            GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_PRAGMA_CUSTOM_CONSTRUCT,       build_scope_pragma_custom_ctr,     GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_PRAGMA_CUSTOM_DIRECTIVE,       build_scope_pragma_custom_dir,     GET_SOC_EVERYWHERE) \
STATEMENT_HANDLER(AST_UNKNOWN_PRAGMA,                build_scope_unknown_pragma,        GET_SOC_EVERYWHERE) \
STATEMENT_HANDLER(AST_STATEMENT_PLACEHOLDER,         build_scope_statement_placeholder, GET_SOC_EXECUTABLE) \
STATEMENT_HANDLER(AST_ENTRY_STATEMENT,               build_scope_entry_stmt,            GET_SOC_ENTRY) \
STATEMENT_HANDLER(AST_TYPEDEF_DECLARATION_STATEMENT, build_scope_typedef_stmt,          GET_SOC_DECLARATION) \
STATEMENT_HANDLER(AST_NODECL_LITERAL,                build_scope_nodecl_literal,        GET_SOC_EXECUTABLE) \
#define STATEMENT_HANDLER(_kind, _handler, _) \
static void _handler(AST, const decl_context_t*, nodecl_t*);
STATEMENT_HANDLER_TABLE
#undef STATEMENT_HANDLER
#define STATEMENT_HANDLER(_kind, _handler, _order_class) \
{ .ast_kind = _kind, .handler = _handler, .get_order_class = _order_class},
static build_scope_statement_handler_t build_scope_statement_function[] = 
{
STATEMENT_HANDLER_TABLE
};
#undef STATEMENT_HANDLER
static int build_scope_statement_function_init = 0;
static int build_scope_statement_function_compare(const void *a, const void *b)
{
build_scope_statement_handler_t *pa = (build_scope_statement_handler_t*)a;
build_scope_statement_handler_t *pb = (build_scope_statement_handler_t*)b;
if (pa->ast_kind < pb->ast_kind)
return -1;
else if (pa->ast_kind > pb->ast_kind)
return 1;
else
return 0;
}
static void init_statement_array(void)
{
if (!build_scope_statement_function_init)
{
qsort(build_scope_statement_function, 
sizeof(build_scope_statement_function) / sizeof(build_scope_statement_function[0]),
sizeof(build_scope_statement_function[0]),
build_scope_statement_function_compare);
build_scope_statement_function_init = 1;
}
}
static build_scope_statement_handler_t *statement_get_statement_handler(node_t ast_kind)
{
init_statement_array();
build_scope_statement_handler_t key = {.ast_kind = ast_kind };
build_scope_statement_handler_t *handler = NULL;
handler = (build_scope_statement_handler_t *)bsearch(
&key,
build_scope_statement_function,
sizeof(build_scope_statement_function)
/ sizeof(build_scope_statement_function[0]),
sizeof(build_scope_statement_function[0]),
build_scope_statement_function_compare);
return handler;
}
static statement_order_class_t statement_get_order_class(AST statement)
{
build_scope_statement_handler_t *handler
= statement_get_statement_handler(ASTKind(statement));
ERROR_CONDITION(handler == NULL || handler->get_order_class == NULL,
"Invalid statement order class %s",
ast_print_node_type(ASTKind(statement)));
return (handler->get_order_class)(statement);
}
static statement_order_class_t get_soc_from_son_1(AST statement)
{
return statement_get_order_class(ASTSon1(statement));
}
static char statement_constraint_checker_update_order(
statement_constraint_checker_t *order_tracker, AST stmt)
{
statement_order_class_t stmt_order_class = statement_get_order_class(stmt);
const statement_order_class_t *current_order_class
= order_tracker->current_order_class;
while (current_order_class < statement_order_end)
{
DEBUG_CODE()
{
fprintf(stderr,
"BUILDSCOPE: Checking order of statement %s (%s) against order %zd (%s)\n",
ast_print_node_type(ASTKind(stmt)),
statement_order_class_print(stmt_order_class),
current_order_class - statement_order,
statement_order_class_print(*current_order_class));
}
if (*current_order_class & stmt_order_class)
{
if (order_tracker->current_order_class != current_order_class)
{
DEBUG_CODE()
{
fprintf(stderr,
"BUILDSCOPE: Order updated to %zd (%s)\n",
current_order_class - statement_order,
statement_order_class_print(*current_order_class));
}
order_tracker->current_order_class = current_order_class;
}
else
{
DEBUG_CODE()
{
fprintf(stderr, "BUILDSCOPE: Order unchanged\n");
}
}
return 1;
}
current_order_class++;
}
return 0;
}
static void fortran_build_scope_statement(AST statement, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
DEBUG_CODE()
{
fprintf(stderr, "=== [%s] Statement ===\n", ast_location(statement));
}
build_scope_statement_handler_t *handler = statement_get_statement_handler(ASTKind(statement));
if (handler == NULL
|| handler->handler == NULL)
{
fatal_printf_at(ast_get_locus(statement),
"unhandled statement %s\n", ast_print_node_type(ASTKind(statement)));
}
else
{
(handler->handler)(statement, decl_context, nodecl_output);
}
}
static void build_scope_compound_statement_relaxed(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output);
void fortran_build_scope_statement_from_source(AST statement, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
build_scope_statement_handler_t* compound_stmt_handler = statement_get_statement_handler(AST_COMPOUND_STATEMENT);
compound_stmt_handler->handler = build_scope_compound_statement_relaxed;
fortran_build_scope_statement(statement, decl_context, nodecl_output);
compound_stmt_handler->handler = build_scope_compound_statement;
}
static void fortran_build_scope_statement_inside_block_context(
AST statement,
const decl_context_t* decl_context,
nodecl_t* nodecl_output)
{
const decl_context_t* new_context = fortran_new_block_context(decl_context);
fortran_build_scope_statement(statement, new_context, nodecl_output);
if (nodecl_is_null(*nodecl_output))
{
*nodecl_output = nodecl_make_list_1(
nodecl_make_empty_statement(ast_get_locus(statement))
);
}
*nodecl_output =
nodecl_make_list_1(
nodecl_make_context(
*nodecl_output,
new_context,
nodecl_get_locus(*nodecl_output)));
}
const char* get_name_of_generic_spec(AST generic_spec)
{
switch (ASTKind(generic_spec))
{
case AST_SYMBOL:
{
return strtolower(ASTText(generic_spec));
}
case AST_OPERATOR_NAME:
{
return strtolower(strappend(".operator.", ASTText(generic_spec)));
}
case AST_IO_SPEC:
{
sorry_printf_at(ast_get_locus(generic_spec),
"io-specifiers for generic-specifiers not supported\n");
}
default:
{
internal_error("%s: Invalid generic spec '%s'", 
ast_location(generic_spec), ast_print_node_type(ASTKind(generic_spec)));
}
}
return NULL;
}
static int compute_kind_specifier(AST kind_expr, const decl_context_t* decl_context,
int (*default_kind)(void),
nodecl_t* nodecl_output,
char *interoperable)
{
*interoperable = 0;
fortran_check_expression(kind_expr, decl_context, nodecl_output);
if (!nodecl_is_err_expr(*nodecl_output)
&& nodecl_is_constant(*nodecl_output))
{
scope_entry_t* symbol = fortran_data_ref_get_symbol(*nodecl_output);
if (symbol != NULL)
{
if (symbol_entity_specs_get_from_module(symbol) != NULL
&& symbol_entity_specs_get_is_builtin(symbol_entity_specs_get_from_module(symbol))
&& strcasecmp(symbol_entity_specs_get_from_module(symbol)->symbol_name, "iso_c_binding") == 0)
{
*interoperable = 1;
}
}
return const_value_cast_to_4(nodecl_get_constant(*nodecl_output));
}
else
{
int result = default_kind();
warn_printf_at(ast_get_locus(kind_expr), "could not compute KIND specifier, assuming %d\n", result);
return result;
}
}
static type_t* choose_type_from_kind_function(nodecl_t expr, 
type_t* (*kind_function)(int kind),
int kind_size,
int default_kind_size,
const char* type_name)
{
type_t* result = kind_function(kind_size);
if (result == NULL)
{
error_printf_at(nodecl_get_locus(expr), "%s(KIND=%d) not supported\n", type_name, kind_size);
result = kind_function(default_kind_size);
if (result == NULL)
{
result = kind_function(1);
}
ERROR_CONDITION(result == NULL, "Fallback kind should not be NULL", 0);
}
return result;
}
type_t* choose_int_type_from_kind(nodecl_t expr, int kind_size)
{
return choose_type_from_kind_function(expr,
fortran_choose_int_type_from_kind,
kind_size,
fortran_get_default_integer_type_kind(),
"INTEGER");
}
type_t* choose_float_type_from_kind(nodecl_t expr, int kind_size)
{
return choose_type_from_kind_function(expr,
fortran_choose_float_type_from_kind,
kind_size,
fortran_get_default_real_type_kind(),
"REAL");
}
type_t* choose_logical_type_from_kind(nodecl_t expr, int kind_size)
{
return choose_type_from_kind_function(expr,
fortran_choose_logical_type_from_kind,
kind_size,
fortran_get_default_logical_type_kind(),
"LOGICAL");
}
type_t* choose_character_type_from_kind(nodecl_t expr, int kind_size)
{
return choose_type_from_kind_function(expr,
fortran_choose_character_type_from_kind,
kind_size,
fortran_get_default_character_type_kind(),
"CHARACTER");
}
static type_t* choose_type_from_kind(AST expr, const decl_context_t* decl_context, type_t* (*fun)(nodecl_t expr, int kind_size),
int (*default_kind)(void))
{
nodecl_t nodecl_output = nodecl_null();
char is_interoperable = 0;
int kind_size = compute_kind_specifier(expr, decl_context, default_kind, &nodecl_output, &is_interoperable);
type_t* result = fun(nodecl_output, kind_size);
if (is_interoperable)
{
result = get_variant_type_interoperable(result);
}
return result;
}
static type_t* get_derived_type_name(AST a, const decl_context_t* decl_context)
{
ERROR_CONDITION(ASTKind(a) != AST_DERIVED_TYPE_NAME, "Invalid tree '%s'\n", ast_print_node_type(ASTKind(a)));
AST name = ASTSon0(a);
if (ASTSon1(a) != NULL)
{
sorry_printf_at(ast_get_locus(ASTSon1(a)), "unsupported generic type-names");
}
type_t* result = NULL;
scope_entry_t* entry = fortran_query_name_str(decl_context, strtolower(ASTText(name)), 
ast_get_locus(name));
if (entry != NULL)
{
if (entry->kind == SK_TYPEDEF
&& is_named_type(advance_over_typedefs(entry->type_information)))
{
entry = named_type_get_symbol(advance_over_typedefs(entry->type_information));
}
if (entry->kind == SK_CLASS)
{
result = get_user_defined_type(entry);
}
else if (entry->kind == SK_ENUM)
{
result = enum_type_get_underlying_type(entry->type_information);
}
}
return result;
}
static type_t* fortran_gather_type_from_declaration_type_spec_(AST a, 
const decl_context_t* decl_context,
AST *character_length_out)
{
if (character_length_out != NULL)
*character_length_out = NULL;
type_t* result = NULL;
switch (ASTKind(a))
{
case AST_INT_TYPE:
{
result = fortran_get_default_integer_type();
if (ASTSon0(a) != NULL)
{
result = choose_type_from_kind(ASTSon0(a), decl_context, 
choose_int_type_from_kind, fortran_get_default_integer_type_kind);
}
break;
}
case AST_FLOAT_TYPE:
{
result = fortran_get_default_real_type();
if (ASTSon0(a) != NULL)
{
result = choose_type_from_kind(ASTSon0(a), decl_context, 
choose_float_type_from_kind, fortran_get_default_real_type_kind);
}
break;
}
case AST_DOUBLE_TYPE:
{
result = fortran_get_doubleprecision_type();
break;
}
case AST_COMPLEX_TYPE:
{
type_t* element_type = NULL; 
if (ASTKind(ASTSon0(a)) == AST_DECIMAL_LITERAL)
{
element_type = choose_type_from_kind(ASTSon0(a), decl_context, 
choose_float_type_from_kind, fortran_get_default_real_type_kind);
}
else
{
element_type = fortran_gather_type_from_declaration_type_spec_(ASTSon0(a), decl_context,
NULL);
}
result = get_complex_type(element_type);
break;
}
case AST_CHARACTER_TYPE:
{
result = fortran_get_default_character_type();
AST char_selector = ASTSon0(a);
AST len = NULL;
AST kind = NULL;
if (char_selector != NULL)
{
len = ASTSon0(char_selector);
kind = ASTSon1(char_selector);
}
if (kind != NULL)
{
result = choose_type_from_kind(kind, decl_context, 
choose_character_type_from_kind, 
fortran_get_default_character_type_kind);
}
if (len == NULL
|| character_length_out == NULL)
{
nodecl_t nodecl_len = nodecl_null();
if (len == NULL)
{
nodecl_len = const_value_to_nodecl(const_value_get_one(fortran_get_default_integer_type_kind(), 1));
}
else
{
fortran_check_expression(len, decl_context, &nodecl_len);
nodecl_len = fortran_expression_as_value(nodecl_len);
}
nodecl_t lower_bound = nodecl_make_integer_literal(
get_signed_int_type(),
const_value_get_one(type_get_size(get_signed_int_type()), 1),
nodecl_get_locus(nodecl_len));
result = get_array_type_bounds(result, lower_bound, nodecl_len, decl_context);
}
else
{
*character_length_out = len;
result = get_array_type(result, nodecl_null(), decl_context);
}
break;
}
case AST_BOOL_TYPE:
{
result = get_bool_of_integer_type(fortran_get_default_logical_type());
if (ASTSon0(a) != NULL)
{
result = choose_type_from_kind(ASTSon0(a), decl_context, 
choose_logical_type_from_kind, fortran_get_default_logical_type_kind);
}
break;
}
case AST_TYPE_NAME:
{
result = get_derived_type_name(ASTSon0(a), decl_context);
if (result == NULL)
{
error_printf_at(ast_get_locus(a), "invalid type-specifier '%s'\n",
fortran_prettyprint_in_buffer(a));
result = get_error_type();
}
break;
}
case AST_VECTOR_TYPE:
{
type_t* element_type = fortran_gather_type_from_declaration_type_spec_(ASTSon0(a), decl_context,
NULL);
result = get_vector_type_by_bytes(element_type, 0);
break;
}
case AST_PIXEL_TYPE:
{
error_printf_at(ast_get_locus(a), "sorry: PIXEL type-specifier not implemented\n");
result = get_error_type();
break;
}
case AST_CLASS_NAME:
{
result = get_derived_type_name(ASTSon0(a), decl_context);
if (result == NULL || !is_class_type(result))
{
error_printf_at(ast_get_locus(a), "invalid type-specifier '%s'\n",
fortran_prettyprint_in_buffer(a));
result = get_error_type();
}
result = get_variant_type_fortran_polymorphic(result);
break;
}
case AST_TYPE_LITERAL_REF:
{
const char *prefix = NULL;
void *p = NULL;
const char *tmp = ASTText(ASTSon0(a));
unpack_pointer(tmp, &prefix, &p);
ERROR_CONDITION(prefix == NULL || p == NULL || strcmp(prefix, "type") != 0,
"Failure during unpack of type", 0);
result = (type_t*)p;
break;
}
default:
{
internal_error("Unexpected node '%s'\n", ast_print_node_type(ASTKind(a)));
}
}
return result;
}
typedef
struct attr_spec_tag
{
char is_allocatable;
char is_asynchronous;
char is_codimension;
AST coarray_spec;
char is_contiguous;
char is_dimension;
AST array_spec;
char is_external;
char is_intent;
intent_kind_t intent_kind;
char is_intrinsic;
char is_optional;
char is_constant;
char is_pointer;
char is_protected;
char is_save;
char is_target;
char is_value;
char is_volatile;
char is_public;
char is_private;
nodecl_t bind_info;
char is_nopass;
const char* pass_name;
char is_non_overridable;
char is_deferred;
char is_variable;
} attr_spec_t;
#define ATTR_SPEC_HANDLER_LIST \
ATTR_SPEC_HANDLER(allocatable) \
ATTR_SPEC_HANDLER(asynchronous) \
ATTR_SPEC_HANDLER(codimension) \
ATTR_SPEC_HANDLER(contiguous) \
ATTR_SPEC_HANDLER(dimension) \
ATTR_SPEC_HANDLER(external) \
ATTR_SPEC_HANDLER(intent) \
ATTR_SPEC_HANDLER(intrinsic) \
ATTR_SPEC_HANDLER(optional) \
ATTR_SPEC_HANDLER(parameter) \
ATTR_SPEC_HANDLER(pointer) \
ATTR_SPEC_HANDLER(protected) \
ATTR_SPEC_HANDLER(save) \
ATTR_SPEC_HANDLER(target) \
ATTR_SPEC_HANDLER(value) \
ATTR_SPEC_HANDLER(public) \
ATTR_SPEC_HANDLER(private) \
ATTR_SPEC_HANDLER(volatile) \
ATTR_SPEC_HANDLER(bind) \
ATTR_SPEC_HANDLER(pass) \
ATTR_SPEC_HANDLER(nopass) \
ATTR_SPEC_HANDLER(non_overridable) \
ATTR_SPEC_HANDLER(deferred) \
ATTR_SPEC_HANDLER_STR(is_variable, "@IS_VARIABLE@")
#define ATTR_SPEC_HANDLER(_name) \
static void attr_spec_##_name##_handler(AST attr_spec_item, const decl_context_t* decl_context, attr_spec_t* attr_spec);
#define ATTR_SPEC_HANDLER_STR(_name, _) ATTR_SPEC_HANDLER(_name)
ATTR_SPEC_HANDLER_LIST
#undef ATTR_SPEC_HANDLER
#undef ATTR_SPEC_HANDLER_STR
typedef struct attr_spec_handler_tag {
const char* attr_name;
void (*handler)(AST attr_spec_item, const decl_context_t* decl_context, attr_spec_t* attr_spec);
} attr_spec_handler_t;
attr_spec_handler_t attr_spec_handler_table[] = {
#define ATTR_SPEC_HANDLER(_name) \
{ #_name , attr_spec_##_name##_handler },
#define ATTR_SPEC_HANDLER_STR(_name, _str) \
{ _str, attr_spec_##_name##_handler },
ATTR_SPEC_HANDLER_LIST
#undef ATTR_SPEC_HANDLER
#undef ATTR_SPEC_HANDLER_STR
};
static int attr_handler_cmp(const void *a, const void *b)
{
return strcasecmp(((attr_spec_handler_t*)a)->attr_name,
((attr_spec_handler_t*)b)->attr_name);
}
static char attr_spec_handler_table_init = 0;
static void gather_attr_spec_item(AST attr_spec_item, const decl_context_t* decl_context, attr_spec_t *attr_spec)
{
if (!attr_spec_handler_table_init)
{
qsort(attr_spec_handler_table, 
sizeof(attr_spec_handler_table) / sizeof(attr_spec_handler_table[0]),
sizeof(attr_spec_handler_table[0]),
attr_handler_cmp);
attr_spec_handler_table_init = 1;
}
switch (ASTKind(attr_spec_item))
{
case AST_ATTR_SPEC:
{
attr_spec_handler_t key = { .attr_name = ASTText(attr_spec_item) };
attr_spec_handler_t* handler = (attr_spec_handler_t*)bsearch(
&key,
attr_spec_handler_table, 
sizeof(attr_spec_handler_table) / sizeof(attr_spec_handler_table[0]),
sizeof(attr_spec_handler_table[0]),
attr_handler_cmp);
if (handler == NULL 
|| handler->handler == NULL)
{
internal_error("Unhandled handler of '%s' (%s)\n", ASTText(attr_spec_item), ast_print_node_type(ASTKind(attr_spec_item)));
}
(handler->handler)(attr_spec_item, decl_context, attr_spec);
break;
}
default:
{
internal_error("Unhandled tree '%s'\n", ast_print_node_type(ASTKind(attr_spec_item)));
}
}
}
static void attr_spec_allocatable_handler(AST a UNUSED_PARAMETER, 
const decl_context_t* decl_context UNUSED_PARAMETER, attr_spec_t* attr_spec)
{
attr_spec->is_allocatable = 1;
}
static void attr_spec_asynchronous_handler(AST a UNUSED_PARAMETER, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
attr_spec_t* attr_spec)
{
attr_spec->is_asynchronous = 1;
}
static void attr_spec_codimension_handler(AST a, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
attr_spec_t* attr_spec)
{
attr_spec->is_codimension = 1;
attr_spec->coarray_spec = ASTSon0(a);
}
static void attr_spec_contiguous_handler(AST a UNUSED_PARAMETER, 
const decl_context_t* decl_context UNUSED_PARAMETER,  
attr_spec_t* attr_spec)
{
attr_spec->is_contiguous = 1;
}
static void attr_spec_dimension_handler(AST a, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
attr_spec_t* attr_spec)
{
attr_spec->is_dimension = 1;
attr_spec->array_spec = ASTSon0(a);
}
static void attr_spec_external_handler(AST a UNUSED_PARAMETER, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
attr_spec_t* attr_spec)
{
attr_spec->is_external = 1;
}
static void attr_spec_intent_handler(AST a, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
attr_spec_t* attr_spec)
{
attr_spec->is_intent = 1;
const char* intent_kind_str = ASTText(ASTSon0(a));
if (strcasecmp(intent_kind_str, "in") == 0)
{
attr_spec->intent_kind = INTENT_IN;
}
else if (strcasecmp(intent_kind_str, "out") == 0)
{
attr_spec->intent_kind = INTENT_OUT;
}
else if (strcasecmp(intent_kind_str, "inout") == 0)
{
attr_spec->intent_kind = INTENT_INOUT;
}
else
{
internal_error("Invalid intent kind '%s'\n", intent_kind_str);
}
}
static void attr_spec_intrinsic_handler(AST a UNUSED_PARAMETER, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
attr_spec_t* attr_spec)
{
attr_spec->is_intrinsic = 1;
}
static void attr_spec_optional_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_optional = 1;
}
static void attr_spec_parameter_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_constant = 1;
}
static void attr_spec_pointer_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_pointer = 1;
}
static void attr_spec_protected_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_protected = 1;
}
static void attr_spec_save_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_save = 1;
}
static void attr_spec_target_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_target = 1;
}
static void attr_spec_value_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_value = 1;
}
static void attr_spec_volatile_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_volatile = 1;
}
static void attr_spec_public_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_public = 1;
}
static void attr_spec_private_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_private = 1;
}
static void attr_spec_bind_handler(
AST a,
const decl_context_t *decl_context
UNUSED_PARAMETER,
attr_spec_t *attr_spec)
{
AST bind_kind = ASTSon0(a);
if (strcmp(ASTText(bind_kind), "c") != 0)
{
error_printf_at(
ast_get_locus(bind_kind),
"BIND specifier other than BIND(C, ...) is not supported");
attr_spec->bind_info = nodecl_make_err_expr(ast_get_locus(a));
return;
}
nodecl_t nodecl_bind_name = nodecl_null();
AST bind_name_expr = ASTSon1(a);
if (bind_name_expr != NULL)
{
fortran_check_expression(bind_name_expr, decl_context, &nodecl_bind_name);
if (nodecl_is_err_expr(nodecl_bind_name))
{
attr_spec->bind_info = nodecl_bind_name;
return;
}
else if (!nodecl_is_constant(nodecl_bind_name)
|| !fortran_is_character_type(no_ref(nodecl_get_type(nodecl_bind_name))))
{
error_printf_at(ast_get_locus(bind_name_expr),
"NAME of BIND(C) must be a constant character expression\n");
attr_spec->bind_info = nodecl_make_err_expr(ast_get_locus(a));
return;
}
}
attr_spec->bind_info = nodecl_make_fortran_bind_c(nodecl_bind_name, ast_get_locus(a));
}
static nodecl_t check_bind(AST bind_spec, const decl_context_t *decl_context)
{
attr_spec_t attr_spec;
attr_spec.bind_info = nodecl_null();
ERROR_CONDITION(
strcmp(ast_get_text(bind_spec), "bind") != 0, "Invalid tree", 0);
attr_spec_bind_handler(bind_spec, decl_context, &attr_spec);
return attr_spec.bind_info;
}
static void check_bind_spec(scope_entry_t *entry,
AST bind_spec,
const decl_context_t *decl_context)
{
nodecl_t n = check_bind(bind_spec, decl_context);
if (!nodecl_is_err_expr(n))
{
symbol_entity_specs_set_bind_info(entry, n);
}
}
static void attr_spec_is_variable_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_variable = 1;
}
static void attr_spec_pass_handler(AST a,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_nopass = 0;
AST pass_name = ASTSon0(a);
if (pass_name != NULL)
{
attr_spec->pass_name = ASTText(pass_name);
}
}
static void attr_spec_nopass_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_nopass = 1;
}
static void attr_spec_non_overridable_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_non_overridable = 1;
}
static void attr_spec_deferred_handler(AST a UNUSED_PARAMETER,
const decl_context_t* decl_context UNUSED_PARAMETER,
attr_spec_t* attr_spec)
{
attr_spec->is_deferred = 1;
}
static void gather_attr_spec_list(AST attr_spec_list, const decl_context_t* decl_context, attr_spec_t *attr_spec)
{
AST it;
for_each_element(attr_spec_list, it)
{
AST attr_spec_item = ASTSon1(it);
gather_attr_spec_item(attr_spec_item, decl_context, attr_spec);
}
}
typedef
enum array_spec_kind_tag
{
ARRAY_SPEC_KIND_NONE = 0,
ARRAY_SPEC_KIND_EXPLICIT_SHAPE,
ARRAY_SPEC_KIND_ASSUMED_SHAPE,
ARRAY_SPEC_KIND_DEFERRED_SHAPE,
ARRAY_SPEC_KIND_ASSUMED_SIZE,
ARRAY_SPEC_KIND_IMPLIED_SHAPE,
ARRAY_SPEC_KIND_ERROR,
} array_spec_kind_t;
static type_t* eval_array_spec(type_t* basic_type, 
AST array_spec_list, 
const decl_context_t* decl_context,
char check_expressions,
nodecl_t* nodecl_output)
{
char was_ref = is_lvalue_reference_type(basic_type);
if (decl_context->current_scope->related_entry->kind != SK_FUNCTION
&& !(decl_context->current_scope->related_entry->kind == SK_VARIABLE
&& symbol_is_parameter_of_function(
decl_context->current_scope->related_entry,
decl_context->current_scope->related_entry->decl_context->current_scope->related_entry)))
{
nodecl_output = NULL;
}
array_spec_kind_t kind = ARRAY_SPEC_KIND_NONE;
nodecl_t lower_bound_seq[MCXX_MAX_ARRAY_SPECIFIER];
memset(lower_bound_seq, 0, sizeof(lower_bound_seq));
nodecl_t upper_bound_seq[MCXX_MAX_ARRAY_SPECIFIER];
memset(upper_bound_seq, 0, sizeof(upper_bound_seq));
int i = 0;
AST it = NULL;
for_each_element(array_spec_list, it)
{
ERROR_CONDITION(i == MCXX_MAX_ARRAY_SPECIFIER, "Too many array specifiers", 0);
AST array_spec_item = ASTSon1(it);
AST lower_bound_tree = ASTSon0(array_spec_item);
AST upper_bound_tree = ASTSon1(array_spec_item);
nodecl_t lower_bound = nodecl_null();
nodecl_t upper_bound = nodecl_null();
if (check_expressions
&& lower_bound_tree != NULL
&& (ASTKind(lower_bound_tree) != AST_SYMBOL
|| (strcmp(ASTText(lower_bound_tree), "*") != 0) ))
{
fortran_check_array_bounds_expression(lower_bound_tree, decl_context, &lower_bound);
if (!nodecl_is_err_expr(lower_bound)
&& !is_integer_type(no_ref(nodecl_get_type(lower_bound))))
{
error_printf_at(nodecl_get_locus(lower_bound), "expression '%s' must be of integer type\n",
codegen_to_str(lower_bound, nodecl_retrieve_context(lower_bound)));
}
else if (nodecl_is_err_expr(lower_bound))
{
kind = ARRAY_SPEC_KIND_ERROR;
}
}
if (check_expressions
&& upper_bound_tree != NULL
&& (ASTKind(upper_bound_tree) != AST_SYMBOL
|| (strcmp(ASTText(upper_bound_tree), "*") != 0) ))
{
fortran_check_array_bounds_expression(upper_bound_tree, decl_context, &upper_bound);
if (!nodecl_is_err_expr(upper_bound)
&& !is_integer_type(no_ref(nodecl_get_type(upper_bound))))
{
error_printf_at(nodecl_get_locus(upper_bound), "expression '%s' must be of integer type\n",
codegen_to_str(upper_bound, nodecl_retrieve_context(upper_bound)));
}
else if (nodecl_is_err_expr(upper_bound))
{
kind = ARRAY_SPEC_KIND_ERROR;
}
}
if (lower_bound_tree == NULL
&& upper_bound_tree == NULL)
{
if (kind == ARRAY_SPEC_KIND_NONE)
{
kind = ARRAY_SPEC_KIND_DEFERRED_SHAPE;
}
else if (kind != ARRAY_SPEC_KIND_DEFERRED_SHAPE
&& kind != ARRAY_SPEC_KIND_ASSUMED_SHAPE
&& kind != ARRAY_SPEC_KIND_ERROR)
{
kind = ARRAY_SPEC_KIND_ERROR;
}
}
else if (upper_bound_tree != NULL
&& ASTKind(upper_bound_tree) == AST_SYMBOL
&& strcmp(ASTText(upper_bound_tree), "*") == 0)
{
if (kind == ARRAY_SPEC_KIND_NONE)
{
kind = ARRAY_SPEC_KIND_IMPLIED_SHAPE;
}
else if (kind == ARRAY_SPEC_KIND_EXPLICIT_SHAPE)
{
kind = ARRAY_SPEC_KIND_ASSUMED_SIZE;
}
else if (kind != ARRAY_SPEC_KIND_ASSUMED_SIZE
&& kind != ARRAY_SPEC_KIND_IMPLIED_SHAPE
&& kind != ARRAY_SPEC_KIND_ERROR)
{
kind = ARRAY_SPEC_KIND_ERROR;
}
}
else if (lower_bound_tree != NULL
&& upper_bound_tree == NULL)
{
if (kind == ARRAY_SPEC_KIND_NONE
|| kind == ARRAY_SPEC_KIND_DEFERRED_SHAPE)
{
kind = ARRAY_SPEC_KIND_ASSUMED_SHAPE;
}
else if (kind != ARRAY_SPEC_KIND_ASSUMED_SHAPE
&& kind != ARRAY_SPEC_KIND_ERROR)
{
kind = ARRAY_SPEC_KIND_ERROR;
}
}
else if (upper_bound_tree != NULL)
{
if (kind == ARRAY_SPEC_KIND_NONE)
{
kind = ARRAY_SPEC_KIND_EXPLICIT_SHAPE;
}
else if (kind != ARRAY_SPEC_KIND_EXPLICIT_SHAPE
&& kind != ARRAY_SPEC_KIND_ERROR)
{
kind = ARRAY_SPEC_KIND_ERROR;
}
if (lower_bound_tree == NULL)
{
lower_bound = nodecl_make_integer_literal(
get_signed_int_type(),
const_value_get_one(type_get_size(get_signed_int_type()), 1),
ast_get_locus(upper_bound_tree));
}
}
if (kind == ARRAY_SPEC_KIND_ERROR)
break;
static int vla_counter = 0;
if (!nodecl_is_null(lower_bound)
&& !nodecl_is_constant(lower_bound))
{
if (nodecl_output == NULL)
{
error_printf_at(nodecl_get_locus(lower_bound), "dimension specifier '%s' must be constant in this context\n",
codegen_to_str(lower_bound, nodecl_retrieve_context(lower_bound)));
}
else
{
const char* vla_name = NULL;
uniquestr_sprintf(&vla_name, "mfc_vla_l_%d", vla_counter);
vla_counter++;
scope_entry_t* new_vla_dim = new_symbol(decl_context, decl_context->current_scope, vla_name);
if (!equivalent_types(
get_unqualified_type(no_ref(nodecl_get_type(lower_bound))),
get_ptrdiff_t_type()))
{
lower_bound = nodecl_make_conversion(lower_bound,
get_ptrdiff_t_type(),
nodecl_get_locus(lower_bound));
}
new_vla_dim->kind = SK_VARIABLE;
new_vla_dim->locus = nodecl_get_locus(lower_bound);
new_vla_dim->value = lower_bound;
new_vla_dim->type_information = get_ptrdiff_t_type();
symbol_entity_specs_set_is_saved_expression(new_vla_dim, 1);
lower_bound = nodecl_make_symbol(new_vla_dim,
new_vla_dim->locus);
nodecl_set_type(lower_bound, new_vla_dim->type_information);
*nodecl_output = nodecl_append_to_list(*nodecl_output,
nodecl_make_object_init(new_vla_dim, 
nodecl_get_locus(lower_bound)));
}
}
if (!nodecl_is_null(upper_bound)
&& !nodecl_is_constant(upper_bound))
{
if (nodecl_output == NULL)
{
error_printf_at(nodecl_get_locus(upper_bound), "dimension specifier '%s' must be constant in this context\n",
codegen_to_str(upper_bound, nodecl_retrieve_context(upper_bound)));
}
else
{
const char* vla_name = NULL;
uniquestr_sprintf(&vla_name, "mfc_vla_u_%d", vla_counter);
vla_counter++;
scope_entry_t* new_vla_dim = new_symbol(decl_context, decl_context->current_scope, vla_name);
if (!equivalent_types(
get_unqualified_type(no_ref(nodecl_get_type(upper_bound))),
get_ptrdiff_t_type()))
{
upper_bound = nodecl_make_conversion(upper_bound,
get_ptrdiff_t_type(),
nodecl_get_locus(upper_bound));
}
new_vla_dim->kind = SK_VARIABLE;
new_vla_dim->locus = nodecl_get_locus(upper_bound);
new_vla_dim->value = upper_bound;
new_vla_dim->type_information = get_ptrdiff_t_type();
symbol_entity_specs_set_is_saved_expression(new_vla_dim, 1);
upper_bound = nodecl_make_symbol(new_vla_dim,
new_vla_dim->locus);
nodecl_set_type(upper_bound, new_vla_dim->type_information);
*nodecl_output = nodecl_append_to_list(*nodecl_output,
nodecl_make_object_init(new_vla_dim, 
nodecl_get_locus(upper_bound)));
}
}
lower_bound_seq[i] = lower_bound;
upper_bound_seq[i] = upper_bound;
i++;
}
type_t* array_type = no_ref(basic_type);
if (kind != ARRAY_SPEC_KIND_ERROR)
{
char needs_descriptor = ((kind == ARRAY_SPEC_KIND_ASSUMED_SHAPE)
|| (kind == ARRAY_SPEC_KIND_DEFERRED_SHAPE));
int j;
for (j = 0; j < i; j++)
{
if (needs_descriptor)
{
array_type = get_array_type_bounds_with_descriptor(array_type, lower_bound_seq[j], upper_bound_seq[j], decl_context);
}
else
{
array_type = get_array_type_bounds(array_type, lower_bound_seq[j], upper_bound_seq[j], decl_context);
}
}
}
else
{
array_type = get_error_type();
}
if (was_ref
&& !is_error_type(array_type))
{
array_type = get_lvalue_reference_type(array_type);
}
return array_type;
}
typedef
struct delayed_array_spec_tag
{
scope_entry_t* entry;
type_t* basic_type;
AST array_spec_list;
const decl_context_t* decl_context;
} delayed_array_spec_t;
static type_t* delayed_array_spec_update_type(type_t* original_type, type_t* new_array)
{
if (is_lvalue_reference_type(original_type))
{
return get_lvalue_reference_type(
delayed_array_spec_update_type(
reference_type_get_referenced_type(original_type), new_array));
}
else if (is_pointer_type(original_type))
{
cv_qualifier_t cv_qualif = get_cv_qualifier(original_type);
return get_cv_qualified_type(
get_pointer_type(
delayed_array_spec_update_type(
pointer_type_get_pointee_type(original_type), new_array)),
cv_qualif);
}
else if (fortran_is_array_type(original_type))
{
cv_qualifier_t cv_qualif = get_cv_qualifier(original_type);
type_t* updated_element_type = NULL;
if (fortran_is_array_type(array_type_get_element_type(original_type)))
{
ERROR_CONDITION(!fortran_is_array_type(new_array),
"This should be an array too", 0);
updated_element_type = delayed_array_spec_update_type(
array_type_get_element_type(original_type),
array_type_get_element_type(new_array));
}
else
{
updated_element_type =
array_type_get_element_type(original_type);
}
type_t* result_type = array_type_rebase(new_array,
updated_element_type);
return get_cv_qualified_type(
result_type,
cv_qualif);
}
else
{
internal_error("Unexpected type '%s' here\n", print_declarator(original_type));
}
}
static char delayed_array_specifier_cmp(void *key, void *info)
{
scope_entry_t* entry = (scope_entry_t*)key;
delayed_array_spec_t* data = (delayed_array_spec_t*)info;
return (data->entry == entry);
}
static void delayed_compute_type_from_array_spec(void *info, nodecl_t* nodecl_output)
{
delayed_array_spec_t* data = (delayed_array_spec_t*)info;
type_t* array_type = eval_array_spec(data->basic_type,
data->array_spec_list,
data->decl_context,
1,
nodecl_output);
if (is_error_type(array_type))
{
data->entry->type_information = array_type;
}
else
{
data->entry->type_information =
delayed_array_spec_update_type(data->entry->type_information, array_type);
}
fortran_cast_initialization(data->entry, &data->entry->value);
DELETE(data);
}
static void compute_type_from_array_spec(
scope_entry_t* entry,
type_t* basic_type,
AST array_spec_list,
const decl_context_t* decl_context,
char allow_nonconstant)
{
if (!allow_nonconstant)
{
type_t *array_type = eval_array_spec(basic_type,
array_spec_list,
decl_context,
1,
NULL);
entry->type_information = array_type;
}
else
{
type_t *array_type = eval_array_spec(basic_type,
array_spec_list,
decl_context,
0,
NULL);
entry->type_information = array_type;
delayed_array_spec_t *data = NEW(delayed_array_spec_t);
data->entry = entry;
data->basic_type = basic_type;
data->array_spec_list = array_spec_list;
data->decl_context = decl_context;
build_scope_delay_list_add(DELAY_AFTER_DECLARATIONS,
delayed_compute_type_from_array_spec,
data);
}
}
static char array_type_is_deferred_shape(type_t* t)
{
ERROR_CONDITION(!fortran_is_array_type(t), "Invalid type", 0);
if (!array_type_with_descriptor(t))
return 0;
while (fortran_is_array_type(t))
{
if (!nodecl_is_null(array_type_get_array_lower_bound(t))
|| !nodecl_is_null(array_type_get_array_upper_bound(t)))
return 0;
t = array_type_get_element_type(t);
}
return 1;
}
static void check_array_type_is_valid_for_allocatable(type_t* t,
scope_entry_t* entry,
const locus_t* locus)
{
if (!array_type_is_deferred_shape(t))
{
error_printf_at(locus, "ALLOCATABLE entity '%s' does not have a deferred shape DIMENSION attribute\n",
entry->symbol_name);
}
}
static void check_array_type_is_valid_for_pointer(type_t* t,
scope_entry_t* entry,
const locus_t* locus)
{
if (!array_type_is_deferred_shape(
pointer_type_get_pointee_type(t)))
{
error_printf_at(locus, "POINTER entity '%s' does not have a deferred shape DIMENSION attribute\n",
entry->symbol_name);
}
}
static void build_scope_access_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
attr_spec_t attr_spec;
memset(&attr_spec, 0, sizeof(attr_spec));
AST access_spec = ASTSon0(a);
gather_attr_spec_item(access_spec, decl_context, &attr_spec);
AST access_id_list = ASTSon1(a);
if (access_id_list != NULL)
{
AST it;
for_each_element(access_id_list, it)
{
AST access_id = ASTSon1(it);
const char* name = get_name_of_generic_spec(access_id);
scope_entry_list_t* entry_list = get_symbols_for_name(decl_context, access_id, name);
scope_entry_list_iterator_t *entry_it = NULL;
for (entry_it = entry_list_iterator_begin(entry_list);
!entry_list_iterator_end(entry_it);
entry_list_iterator_next(entry_it))
{
scope_entry_t* sym = entry_list_iterator_current(entry_it);
if (symbol_entity_specs_get_access(sym) != AS_UNKNOWN)
{
error_printf_at(ast_get_locus(access_id), "access specifier already given for entity '%s'\n",
sym->symbol_name);
}
else
{
if (attr_spec.is_public)
{
symbol_entity_specs_set_access(sym, AS_PUBLIC);
}
else if (attr_spec.is_private)
{
symbol_entity_specs_set_access(sym, AS_PRIVATE);
}
else
{
internal_error("Code unreachable", 0);
}
}
}
entry_list_iterator_free(entry_it);
}
}
else
{
scope_entry_t* current_sym = decl_context->current_scope->related_entry;
if (current_sym == NULL
|| current_sym->kind != SK_MODULE)
{
error_printf_at(ast_get_locus(a), "wrong usage of access-statement\n");
}
else
{
if (symbol_entity_specs_get_access(current_sym) != AS_UNKNOWN)
{
error_printf_at(ast_get_locus(a), "module '%s' already given a default access\n",
current_sym->symbol_name);
}
if (attr_spec.is_public)
{
symbol_entity_specs_set_access(current_sym, AS_PUBLIC);
}
else if (attr_spec.is_private)
{
symbol_entity_specs_set_access(current_sym, AS_PRIVATE);
}
else
{
internal_error("Code unreachable", 0);
}
}
}
}
static void build_scope_allocatable_stmt(AST a, const decl_context_t* decl_context,
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST allocatable_decl_list = ASTSon0(a);
AST it;
for_each_element(allocatable_decl_list, it)
{
AST allocatable_decl = ASTSon1(it);
AST name = NULL;
AST array_spec = NULL;
if (ASTKind(allocatable_decl) == AST_SYMBOL)
{
name = allocatable_decl;
}
else if (ASTKind(allocatable_decl) == AST_DIMENSION_DECL)
{
name = ASTSon0(allocatable_decl);
array_spec = ASTSon1(allocatable_decl);
}
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
if (entry->kind == SK_UNDEFINED)
{
entry->kind = SK_VARIABLE;
}
if (entry->kind != SK_VARIABLE)
{
error_printf_at(ast_get_locus(name), "invalid entity '%s' in ALLOCATABLE clause\n", 
ASTText(name));
continue;
}
if (is_pointer_type(entry->type_information))
{
error_printf_at(ast_get_locus(name), "attribute POINTER conflicts with ALLOCATABLE\n");
continue;
}
if (symbol_entity_specs_get_is_allocatable(entry))
{
error_printf_at(ast_get_locus(name), "attribute ALLOCATABLE was already set for entity '%s'\n",
ASTText(name));
continue;
}
symbol_entity_specs_set_is_allocatable(entry, 1);
if (array_spec != NULL)
{
if (fortran_is_array_type(no_ref(entry->type_information))
|| fortran_is_pointer_to_array_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(a), "entity '%s' has already a DIMENSION attribute\n",
entry->symbol_name);
continue;
}
char was_ref = is_lvalue_reference_type(entry->type_information);
if (!is_error_type(entry->type_information))
{
compute_type_from_array_spec(
entry,
no_ref(entry->type_information),
array_spec,
decl_context,
0);
if (was_ref)
{
entry->type_information = get_lvalue_reference_type(entry->type_information);
}
}
}
if (fortran_is_array_type(no_ref(entry->type_information)))
{
check_array_type_is_valid_for_allocatable(no_ref(entry->type_information),
entry,
ast_get_locus(allocatable_decl_list));
}
}
}
static void build_scope_allocate_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST type_spec = ASTSon0(a);
AST allocation_list = ASTSon1(a);
AST alloc_opt_list = ASTSon2(a);
nodecl_t nodecl_type = nodecl_null();
if (type_spec != NULL)
{
type_t* allocate_type = fortran_gather_type_from_type_spec(type_spec, decl_context);
nodecl_type = nodecl_make_type(allocate_type, ast_get_locus(type_spec));
}
nodecl_t nodecl_allocate_list = nodecl_null();
char error = 0;
AST it;
for_each_element(allocation_list, it)
{
AST allocate_object = ASTSon1(it);
if (ASTKind(allocate_object) == AST_DIMENSION_DECL)
{
sorry_printf_at(ast_get_locus(allocate_object),
"coarrays not supported\n");
}
AST data_ref = allocate_object;
nodecl_t nodecl_data_ref;
fortran_check_expression(data_ref, decl_context, &nodecl_data_ref);
if (!nodecl_is_err_expr(nodecl_data_ref))
{
scope_entry_t* entry = fortran_data_ref_get_symbol(nodecl_data_ref);
if (entry == NULL
|| (!symbol_entity_specs_get_is_allocatable(entry)
&& !is_pointer_type(no_ref(entry->type_information))))
{
if (entry != NULL)
{
error_printf_at(ast_get_locus(a), "entity '%s' does not have ALLOCATABLE or POINTER attribute\n",
entry->symbol_name);
}
else
{
error_printf_at(ast_get_locus(a), "entity '%s' does not have ALLOCATABLE or POINTER attribute\n",
codegen_to_str(nodecl_data_ref, decl_context));
}
error = 1;
continue;
}
}
nodecl_allocate_list = nodecl_append_to_list(nodecl_allocate_list, nodecl_data_ref);
}
if (error)
{
*nodecl_output = nodecl_make_list_1(
nodecl_make_err_statement(ast_get_locus(a))
);
return;
}
nodecl_t nodecl_opt_value = nodecl_null();
handle_opt_value_list(a, alloc_opt_list, decl_context, &nodecl_opt_value);
*nodecl_output = nodecl_make_list_1(
nodecl_make_fortran_allocate_statement(nodecl_allocate_list, 
nodecl_opt_value,
nodecl_type,
ast_get_locus(a)));
}
static void unsupported_statement(AST a, const char* name)
{
sorry_printf_at(ast_get_locus(a), "%s statement not supported\n", name);
}
static void build_scope_allstop_stmt(AST a, const decl_context_t* decl_context UNUSED_PARAMETER, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
unsupported_statement(a, "ALLSTOP");
}
static void build_scope_arithmetic_if_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST numeric_expr = ASTSon0(a);
AST label_set = ASTSon1(a);
AST lower = ASTSon0(label_set);
AST equal = ASTSon1(label_set);
AST upper = ASTSon2(label_set);
nodecl_t nodecl_numeric_expr = nodecl_null();
fortran_check_expression(numeric_expr, decl_context, &nodecl_numeric_expr);
scope_entry_t* lower_label = fortran_query_label(lower, decl_context,  0);
scope_entry_t* equal_label = fortran_query_label(equal, decl_context,  0);
scope_entry_t* upper_label = fortran_query_label(upper, decl_context,  0);
if (nodecl_is_err_expr(nodecl_numeric_expr)
|| lower_label == NULL
|| equal_label == NULL
|| upper_label == NULL)
{
*nodecl_output = nodecl_make_list_1(
nodecl_make_err_statement(ast_get_locus(a))
);
return;
}
nodecl_numeric_expr = fortran_expression_as_value(nodecl_numeric_expr);
*nodecl_output = nodecl_make_list_1(
nodecl_make_fortran_arithmetic_if_statement(
nodecl_numeric_expr,
nodecl_make_symbol(lower_label, ast_get_locus(lower)),
nodecl_make_symbol(equal_label, ast_get_locus(equal)),
nodecl_make_symbol(upper_label, ast_get_locus(upper)),
ast_get_locus(a)));
}
static void build_scope_expression_stmt(AST a,
const decl_context_t* decl_context UNUSED_PARAMETER,
nodecl_t* nodecl_output)
{
DEBUG_CODE()
{
fprintf(stderr, "== [%s] Expression statement ==\n",
ast_location(a));
}
AST expr = ASTSon0(a);
nodecl_t nodecl_expr = nodecl_null();
fortran_check_expression(expr, decl_context, &nodecl_expr);
if (!nodecl_is_err_expr(nodecl_expr))
{
*nodecl_output = nodecl_make_expression_statement(nodecl_expr,
ast_get_locus(expr));
}
else
{
*nodecl_output = nodecl_make_err_statement(ast_get_locus(a));
}
*nodecl_output = nodecl_make_list_1(*nodecl_output);
}
static void build_scope_associate_construct(AST a, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
unsupported_statement(a, "ASSOCIATE");
}
static void build_scope_asynchronous_stmt(AST a, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
unsupported_statement(a, "ASYNCHRONOUS");
}
static void build_scope_input_output_item_list(AST input_output_item_list, const decl_context_t* decl_context, nodecl_t* nodecl_output);
static void build_io_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST io_spec_list = ASTSon0(a);
nodecl_t nodecl_io_spec_list = nodecl_null();
handle_opt_value_list(a, io_spec_list, decl_context, &nodecl_io_spec_list);
AST input_output_item_list = ASTSon1(a);
nodecl_t nodecl_io_items = nodecl_null();
if (input_output_item_list != NULL)
{
build_scope_input_output_item_list(input_output_item_list, decl_context, &nodecl_io_items);
}
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_io_statement(
nodecl_io_spec_list, nodecl_io_items, ASTText(a), ast_get_locus(a)));
}
static const char* get_common_name_str(const char* common_name)
{
const char *common_name_str = ".common._unnamed";
if (common_name != NULL)
{
common_name_str = strappend(".common.", strtolower(common_name));
}
return common_name_str;
}
scope_entry_t* query_common_name(const decl_context_t* decl_context, 
const char* common_name,
const locus_t* locus)
{
const decl_context_t* program_unit_context = decl_context->current_scope->related_entry->related_decl_context;
scope_entry_t* result = fortran_query_name_str(decl_context, 
get_common_name_str(common_name), locus);
if (result != NULL
&& result->decl_context->current_scope != program_unit_context->current_scope)
result = NULL;
return result;
}
static void build_scope_bind_stmt(AST a,
const decl_context_t* decl_context,
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST language_binding_spec = ASTSon0(a);
AST bind_entity_list = ASTSon1(a);
nodecl_t bind_c_name = check_bind(language_binding_spec, decl_context);
if (nodecl_is_err_expr(bind_c_name))
return;
AST it;
for_each_element(bind_entity_list, it)
{
AST bind_entity = ASTSon1(it);
scope_entry_t* entry = NULL;
if (ASTKind(bind_entity) == AST_COMMON_NAME)
{
entry = query_common_name(decl_context, 
ASTText(ASTSon0(bind_entity)),
ast_get_locus(ASTSon0(bind_entity)));
}
else
{
entry = get_symbol_for_name(decl_context, bind_entity, ASTText(bind_entity));
}
if (entry == NULL)
{
error_printf_at(ast_get_locus(bind_entity), "unknown entity '%s' in BIND statement\n",
fortran_prettyprint_in_buffer(bind_entity));
continue;
}
symbol_entity_specs_set_bind_info(entry, bind_c_name);
}
}
static void build_scope_construct_statements(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output,
statement_constraint_checker_t *constraint_checker)
{
AST it;
AST list = ASTSon0(a);
nodecl_t nodecl_list = nodecl_null();
for_each_element(list, it)
{
AST statement = ASTSon1(it);
if (!statement_constraint_checker_check_statement(
constraint_checker, statement, decl_context))
continue;
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement(statement, decl_context, &nodecl_statement);
nodecl_list = nodecl_concat_lists(nodecl_list, nodecl_statement);
}
*nodecl_output = nodecl_list;
}
static void build_scope_block_construct(AST a, 
const decl_context_t* decl_context, 
nodecl_t* nodecl_output)
{
const decl_context_t *new_context = fortran_new_block_context(decl_context);
AST block = ASTSon1(a);
nodecl_t nodecl_body = nodecl_null();
statement_constraint_checker_t constraint_checker
= statement_constraint_checker_init(block_construct_allowed_statements);
build_scope_construct_statements(
block, new_context, &nodecl_body, &constraint_checker);
*nodecl_output = nodecl_make_list_1(
nodecl_make_context(nodecl_make_list_1(nodecl_make_compound_statement(
nodecl_body, nodecl_null(), ast_get_locus(a))),
new_context,
ast_get_locus(a)));
}
static void build_scope_case_construct(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST expr = ASTSon0(a);
AST statement = ASTSon1(a);
nodecl_t nodecl_expr = nodecl_null();
fortran_check_expression(expr, decl_context, &nodecl_expr);
nodecl_expr = fortran_expression_as_value(nodecl_expr);
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement_inside_block_context(statement, decl_context, &nodecl_statement);
*nodecl_output =
nodecl_make_list_1(
nodecl_make_switch_statement(
nodecl_expr,
nodecl_statement,
ast_get_locus(a)));
}
static void build_scope_case_statement(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST case_selector = ASTSon0(a);
AST statement = ASTSon1(a);
nodecl_t nodecl_expr_list = nodecl_null();
AST case_value_range_list = ASTSon0(case_selector);
AST it;
for_each_element(case_value_range_list, it)
{
AST case_value_range = ASTSon1(it);
if (ASTKind(case_value_range) == AST_CASE_VALUE_RANGE)
{
AST lower_bound = ASTSon0(case_value_range);
AST upper_bound = ASTSon1(case_value_range);
nodecl_t nodecl_lower_bound = nodecl_null();
nodecl_t nodecl_upper_bound = nodecl_null();
if (lower_bound != NULL)
{
fortran_check_expression(lower_bound, decl_context, &nodecl_lower_bound);
nodecl_lower_bound = fortran_expression_as_value(nodecl_lower_bound);
}
if (upper_bound != NULL)
{
fortran_check_expression(upper_bound, decl_context, &nodecl_upper_bound);
nodecl_upper_bound = fortran_expression_as_value(nodecl_upper_bound);
}
nodecl_t nodecl_stride = const_value_to_nodecl(const_value_get_one( fortran_get_default_integer_type_kind(),  1));
nodecl_t nodecl_triplet = nodecl_make_range(
nodecl_lower_bound,
nodecl_upper_bound,
nodecl_stride,
fortran_get_default_integer_type(),
ast_get_locus(case_value_range));
nodecl_expr_list = nodecl_append_to_list(nodecl_expr_list, nodecl_triplet);
}
else
{
nodecl_t nodecl_case_value_range = nodecl_null();
fortran_check_expression(case_value_range, decl_context, &nodecl_case_value_range);
nodecl_case_value_range = fortran_expression_as_value(nodecl_case_value_range);
nodecl_expr_list = nodecl_append_to_list(nodecl_expr_list, 
nodecl_case_value_range);
}
}
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement_inside_block_context(statement, decl_context, &nodecl_statement);
if (!nodecl_is_list(nodecl_statement))
{
nodecl_statement = nodecl_make_list_1(nodecl_statement);
}
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_case_statement(nodecl_expr_list, nodecl_statement, ast_get_locus(a)));
}
static void build_scope_default_statement(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST statement = ASTSon0(a);
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement_inside_block_context(statement, decl_context, &nodecl_statement);
if (!nodecl_is_list((nodecl_statement)))
{
nodecl_statement = nodecl_make_list_1(nodecl_statement);
}
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_default_statement(nodecl_statement, ast_get_locus(a)));
}
static void build_scope_compound_statement_relaxed(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
statement_constraint_checker_t constraint_checker
= statement_constraint_checker_init(allow_all_statements);
build_scope_construct_statements(
a, decl_context, nodecl_output, &constraint_checker);
}
static void build_scope_compound_statement(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
statement_constraint_checker_t constraint_checker
= statement_constraint_checker_init(
executable_construct_allowed_statements);
build_scope_construct_statements(
a, decl_context, nodecl_output, &constraint_checker);
}
static void build_scope_close_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST close_spec_list = ASTSon0(a);
nodecl_t nodecl_opt_value = nodecl_null();
handle_opt_value_list(a, close_spec_list, decl_context, &nodecl_opt_value);
*nodecl_output = nodecl_make_list_1(nodecl_make_fortran_close_statement(nodecl_opt_value, ast_get_locus(a)));
}
static void build_scope_codimension_stmt(AST a UNUSED_PARAMETER, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
unsupported_statement(a, "CODIMENSION");
}
static scope_entry_t* new_common(const decl_context_t* decl_context, const char* common_name)
{
const decl_context_t* program_unit_context = decl_context->current_scope->related_entry->related_decl_context;
scope_entry_t* common_sym = new_fortran_symbol(program_unit_context, get_common_name_str(common_name));
common_sym->kind = SK_COMMON;
return common_sym;
}
static void build_scope_common_stmt(AST a, 
const decl_context_t* decl_context, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST common_block_item_list = ASTSon0(a);
AST it;
for_each_element(common_block_item_list, it)
{
AST common_block_item = ASTSon1(it);
AST common_block_object_list = ASTSon1(common_block_item);
const char* common_name_str = NULL;
AST common_name = ASTSon0(common_block_item);
if (common_name != NULL)
{
common_name_str = ASTText(common_name);
}
scope_entry_t* common_sym = query_common_name(decl_context, 
common_name_str,
ast_get_locus(common_block_item));
if (common_sym == NULL)
{
common_sym = new_common(decl_context, common_name_str);
common_sym->locus = ast_get_locus(a);
}
else
{
common_sym->defined = 1;
}
AST it2;
for_each_element(common_block_object_list, it2)
{
AST common_block_object = ASTSon1(it2);
AST name = NULL;
AST array_spec = NULL;
if (ASTKind(common_block_object) == AST_SYMBOL)
{
name = common_block_object;
}
else if (ASTKind(common_block_object) == AST_DIMENSION_DECL)
{
name = ASTSon0(common_block_object);
array_spec = ASTSon1(common_block_object);
}
else
{
internal_error("Unexpected node '%s'\n", ast_print_node_type(ASTKind(common_block_object)));
}
scope_entry_t* sym = get_symbol_for_name(decl_context, name, ASTText(name));
if (symbol_entity_specs_get_is_in_common(sym))
{
error_printf_at(ast_get_locus(name), "entity '%s' is already in a COMMON\n",
sym->symbol_name);
continue;
}
if (sym->kind == SK_UNDEFINED)
{
sym->kind = SK_VARIABLE;
}
symbol_entity_specs_set_is_static(sym, 0);
symbol_entity_specs_set_is_in_common(sym, 1);
symbol_entity_specs_set_in_common(sym, common_sym);
if (array_spec != NULL)
{
if (fortran_is_array_type(no_ref(sym->type_information))
|| fortran_is_pointer_to_array_type(no_ref(sym->type_information)))
{
error_printf_at(ast_get_locus(a), "entity '%s' has already a DIMENSION attribute\n",
sym->symbol_name);
continue;
}
char was_ref = is_lvalue_reference_type(sym->type_information);
if (!is_error_type(sym->type_information))
{
compute_type_from_array_spec(
sym,
no_ref(sym->type_information),
array_spec,
decl_context,
0);
if (was_ref)
{
sym->type_information = get_lvalue_reference_type(sym->type_information);
}
}
}
symbol_entity_specs_add_related_symbols(common_sym, sym);
}
}
}
static void build_scope_computed_goto_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST label_list = ASTSon0(a);
nodecl_t nodecl_label_list = nodecl_null();
AST it;
for_each_element(label_list, it)
{
AST label = ASTSon1(it);
scope_entry_t* label_sym = fortran_query_label(label, decl_context,  0);
nodecl_label_list = nodecl_append_to_list(nodecl_label_list, 
nodecl_make_symbol(label_sym, ast_get_locus(label)));
}
nodecl_t nodecl_expr = nodecl_null();
fortran_check_expression(ASTSon1(a), decl_context, &nodecl_expr);
nodecl_expr = fortran_expression_as_value(nodecl_expr);
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_computed_goto_statement(
nodecl_label_list,
nodecl_expr,
ast_get_locus(a)));
}
static void build_scope_assigned_goto_stmt(AST a UNUSED_PARAMETER, const decl_context_t* decl_context UNUSED_PARAMETER, nodecl_t* nodecl_output)
{
AST label_name = ASTSon0(a);
scope_entry_t* label_var = fortran_get_variable_with_locus(decl_context, label_name, ASTText(label_name));
if (label_var == NULL)
{
error_printf_at(ast_get_locus(label_name), "symbol '%s' is unknown\n", ASTText(label_name));
*nodecl_output = nodecl_make_list_1(
nodecl_make_err_statement(ast_get_locus(a))
);
return;
}
AST label_list = ASTSon1(a);
nodecl_t nodecl_label_list = nodecl_null();
if (label_list != NULL)
{
AST it;
for_each_element(label_list, it)
{
AST label = ASTSon1(it);
scope_entry_t* label_sym = fortran_query_label(label, decl_context,  0);
nodecl_label_list = nodecl_append_to_list(nodecl_label_list,
nodecl_make_symbol(label_sym, ast_get_locus(label)));
}
}
*nodecl_output =
nodecl_make_list_1(
nodecl_make_fortran_assigned_goto_statement(
nodecl_make_symbol(label_var, ast_get_locus(a)),
nodecl_label_list,
ast_get_locus(a)));
}
static void build_scope_label_assign_stmt(AST a UNUSED_PARAMETER, const decl_context_t* decl_context UNUSED_PARAMETER, nodecl_t* nodecl_output)
{
AST literal_const = ASTSon0(a);
scope_entry_t* entry = fortran_query_label(literal_const,
decl_context,
0);
nodecl_t nodecl_label = nodecl_make_symbol(entry, ast_get_locus(literal_const));
AST label_name = ASTSon1(a);
scope_entry_t* label_var = fortran_get_variable_with_locus(decl_context, label_name, ASTText(label_name));
if (label_var == NULL)
{
error_printf_at(ast_get_locus(label_name), "symbol '%s' is unknown\n", ASTText(label_name));
*nodecl_output = nodecl_make_list_1(
nodecl_make_err_statement(ast_get_locus(a))
);
return;
}
ERROR_CONDITION(label_var == NULL, "Invalid symbol", 0);
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_label_assign_statement(
nodecl_label,
nodecl_make_symbol(label_var, ast_get_locus(label_name)),
ast_get_locus(a)));
}
scope_entry_t* fortran_query_label_str_(const char* label, 
const decl_context_t* decl_context, 
const locus_t* locus,
char is_definition)
{
const char* label_text = strappend(".label_", label);
const decl_context_t* program_unit_context = decl_context->current_scope->related_entry->related_decl_context;
scope_entry_list_t* entry_list = query_name_str_flags(program_unit_context, label_text, NULL, DF_ONLY_CURRENT_SCOPE);
scope_entry_t* new_label = NULL;
if (entry_list == NULL)
{
new_label = new_symbol(program_unit_context, program_unit_context->current_scope, label_text);
new_label->symbol_name = label;
new_label->kind = SK_LABEL;
new_label->locus = locus;
new_label->do_not_print = 1;
new_label->defined = is_definition;
}
else
{
new_label = entry_list_head(entry_list);
if (is_definition)
{
if (new_label->defined)
{
error_printf_at(locus, "label %s has already been defined in %s\n",
new_label->symbol_name,
locus_to_str(new_label->locus));
}
else
{
new_label->defined = 1;
}
}
}
entry_list_free(entry_list);
return new_label;
}
scope_entry_t* fortran_query_label(AST label, 
const decl_context_t* decl_context, 
char is_definition)
{
return fortran_query_label_str_(ASTText(label),
decl_context,
ast_get_locus(label),
is_definition);
}
scope_entry_t* fortran_query_construct_name_str(
const char* construct_name,
const decl_context_t* decl_context, char is_definition,
const locus_t* locus
)
{
construct_name = strtolower(construct_name);
const decl_context_t* program_unit_context = decl_context->current_scope->related_entry->related_decl_context;
scope_entry_list_t* entry_list = query_name_str_flags(program_unit_context, construct_name, NULL, DF_ONLY_CURRENT_SCOPE);
scope_entry_t* new_label = NULL;
if (entry_list == NULL)
{
if (is_definition)
{
new_label = new_symbol(program_unit_context, program_unit_context->current_scope, construct_name);
new_label->kind = SK_LABEL;
new_label->locus = locus;
new_label->do_not_print = 1;
new_label->defined = 1;
}
}
else
{
new_label = entry_list_head(entry_list);
entry_list_free(entry_list);
if (new_label->kind != SK_LABEL
&& new_label->kind != SK_UNDEFINED)
{
error_printf_at(locus, "name '%s' cannot be used as a construct name\n",
new_label->symbol_name);
return NULL;
}
if (is_definition)
{
if (new_label->defined)
{
error_printf_at(locus, "construct name %s has already been defined in %s\n",
new_label->symbol_name,
locus_to_str(new_label->locus));
}
else
{
if (new_label == SK_UNDEFINED)
{
new_label->kind = SK_LABEL;
}
new_label->defined = 1;
}
}
}
return new_label;
}
static void build_scope_labeled_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST label = ASTSon0(a);
AST statement = ASTSon1(a);
scope_entry_t* label_sym = fortran_query_label(label, decl_context,  1);
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement(statement, decl_context, &nodecl_statement);
if (!nodecl_is_null(nodecl_statement))
{
if (!nodecl_is_list(nodecl_statement))
{
nodecl_statement = nodecl_make_list_1(nodecl_statement);
}
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_labeled_statement(nodecl_statement, label_sym, ast_get_locus(a))
);
}
}
static void build_scope_continue_stmt(AST a, const decl_context_t* decl_context UNUSED_PARAMETER, nodecl_t* nodecl_output)
{
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_empty_statement(ast_get_locus(a))
);
}
static void build_scope_critical_construct(AST a, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
unsupported_statement(a, "CRITICAL");
}
static nodecl_t get_construct_name(AST construct_name, const decl_context_t* decl_context)
{
if (construct_name == NULL)
return nodecl_null();
else 
{
scope_entry_t* construct_name_sym = fortran_query_construct_name_str(
ASTText(construct_name), decl_context,  0,
ast_get_locus(construct_name));
if (construct_name_sym == NULL)
{
error_printf_at(ast_get_locus(construct_name), "construct name '%s' not defined\n", ASTText(construct_name));
return nodecl_null();
}
else
{
return nodecl_make_symbol(construct_name_sym, ast_get_locus(construct_name));
}
}
}
static void build_scope_cycle_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST loop_name = ASTSon0(a);
nodecl_t nodecl_construct_name = get_construct_name(loop_name, decl_context);
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_continue_statement(
nodecl_construct_name,
ast_get_locus(a))
);
}
static void generic_implied_do_handler(AST a, const decl_context_t* decl_context,
void (*rec_handler)(AST, const decl_context_t*, nodecl_t* nodecl_output),
nodecl_t* nodecl_output)
{
AST implied_do_object_list = ASTSon0(a);
AST implied_do_control = ASTSon1(a);
AST io_do_variable = ASTSon0(implied_do_control);
AST lower_bound = ASTSon1(implied_do_control);
AST upper_bound = ASTSon2(implied_do_control);
AST stride = ASTSon3(implied_do_control);
nodecl_t nodecl_lower = nodecl_null();
fortran_check_expression(lower_bound, decl_context, &nodecl_lower);
nodecl_lower = fortran_expression_as_value(nodecl_lower);
nodecl_t nodecl_upper = nodecl_null();
fortran_check_expression(upper_bound, decl_context, &nodecl_upper);
nodecl_upper = fortran_expression_as_value(nodecl_upper);
nodecl_t nodecl_stride = nodecl_null();
if (stride != NULL)
{
fortran_check_expression(stride, decl_context, &nodecl_stride);
nodecl_stride = fortran_expression_as_value(nodecl_stride);
}
else
{
nodecl_stride = const_value_to_nodecl(const_value_get_one( fortran_get_default_integer_type_kind(),  1));
}
scope_entry_t* do_variable = fortran_get_variable_with_locus(decl_context, io_do_variable, ASTText(io_do_variable));
if (do_variable == NULL)
{
error_printf_at(ast_get_locus(io_do_variable), "unknown symbol '%s' in io-implied-do\n", ASTText(io_do_variable));
*nodecl_output = nodecl_make_err_expr(ast_get_locus(io_do_variable));
return;
}
if (do_variable->kind == SK_UNDEFINED)
{
do_variable->kind = SK_VARIABLE;
}
else if (do_variable->kind != SK_VARIABLE)
{
error_printf_at(ast_get_locus(io_do_variable), "invalid name '%s' for io-implied-do\n", ASTText(io_do_variable));
*nodecl_output = nodecl_make_err_expr(ast_get_locus(io_do_variable));
return;
}
nodecl_t nodecl_rec = nodecl_null();
rec_handler(implied_do_object_list, decl_context, &nodecl_rec);
*nodecl_output = nodecl_make_fortran_implied_do(
nodecl_make_symbol(do_variable, ast_get_locus(io_do_variable)),
nodecl_make_range(nodecl_lower, nodecl_upper, nodecl_stride, 
fortran_get_default_integer_type(),
ast_get_locus(implied_do_control)),
nodecl_rec,
ast_get_locus(a));
}
static void build_scope_data_stmt_object_list(AST data_stmt_object_list, const decl_context_t* decl_context, 
nodecl_t* nodecl_output)
{
AST it2;
for_each_element(data_stmt_object_list, it2)
{
AST data_stmt_object = ASTSon1(it2);
if (ASTKind(data_stmt_object) == AST_IMPLIED_DO)
{
nodecl_t nodecl_implied_do = nodecl_null();
generic_implied_do_handler(data_stmt_object, decl_context,
build_scope_data_stmt_object_list, &nodecl_implied_do);
*nodecl_output = nodecl_append_to_list(*nodecl_output, nodecl_implied_do);
}
else
{
nodecl_t nodecl_data_stmt_object = nodecl_null();
fortran_check_expression(data_stmt_object, decl_context, &nodecl_data_stmt_object);
*nodecl_output = nodecl_append_to_list(*nodecl_output, 
nodecl_data_stmt_object);
scope_entry_t* entry = nodecl_get_symbol(nodecl_data_stmt_object);
if (entry != NULL && !symbol_entity_specs_get_is_in_common(entry))
{
symbol_entity_specs_set_is_static(entry, 1);
}
}
}
}
static void build_scope_data_stmt_do(AST a, const decl_context_t* decl_context,
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST data_stmt_set_list = ASTSon0(a);
scope_entry_t* entry = get_or_create_data_symbol_info(decl_context);
AST it;
for_each_element(data_stmt_set_list, it)
{
AST data_stmt_set = ASTSon1(it);
AST data_stmt_object_list = ASTSon0(data_stmt_set);
nodecl_t nodecl_item_set = nodecl_null();
build_scope_data_stmt_object_list(data_stmt_object_list, decl_context, &nodecl_item_set);
nodecl_t nodecl_data_set = nodecl_null();
AST data_stmt_value_list = ASTSon1(data_stmt_set);
AST it2;
for_each_element(data_stmt_value_list, it2)
{
AST data_stmt_value = ASTSon1(it2);
if (ASTKind(data_stmt_value) == AST_MUL)
{
nodecl_t nodecl_repeat;
fortran_check_expression(ASTSon0(data_stmt_value), decl_context, &nodecl_repeat);
if (!nodecl_is_constant(nodecl_repeat))
{
error_printf_at(nodecl_get_locus(nodecl_repeat), "data-stmt-repeat '%s' is not a constant expression\n",
codegen_to_str(nodecl_repeat, nodecl_retrieve_context(nodecl_repeat)));
}
nodecl_t nodecl_value;
fortran_check_expression(ASTSon1(data_stmt_value), decl_context, &nodecl_value);
if (!nodecl_is_constant(nodecl_value))
{
error_printf_at(nodecl_get_locus(nodecl_value), "data-stmt-value '%s' is not a constant expression\n",
codegen_to_str(nodecl_value, nodecl_retrieve_context(nodecl_value)));
}
if (!nodecl_is_constant(nodecl_repeat)
|| !nodecl_is_constant(nodecl_value))
continue;
if (const_value_is_nonzero
(const_value_lt(nodecl_get_constant(nodecl_repeat), 
const_value_get_zero(fortran_get_default_integer_type_kind(), 1))))
{
error_printf_at(nodecl_get_locus(nodecl_repeat), "data-stmt-repeat is negative\n");
continue;
}
uint64_t repeat = const_value_cast_to_8(nodecl_get_constant(nodecl_repeat));
uint64_t i;
for (i = 0; i < repeat; i++)
{
nodecl_data_set = nodecl_append_to_list(nodecl_data_set, nodecl_shallow_copy(nodecl_value));
}
}
else
{
nodecl_t nodecl_value = nodecl_null();
fortran_check_expression(data_stmt_value, decl_context, &nodecl_value);
if (!nodecl_is_constant(nodecl_value))
{
error_printf_at(nodecl_get_locus(nodecl_value), "data-stmt-value '%s' is not a constant expression\n",
codegen_to_str(nodecl_value, nodecl_retrieve_context(nodecl_value)));
continue;
}
nodecl_data_set = nodecl_append_to_list(nodecl_data_set, nodecl_value);
}
}
entry->value = nodecl_append_to_list(entry->value,
nodecl_make_fortran_data(nodecl_item_set,
nodecl_data_set, ast_get_locus(data_stmt_set)));
}
}
typedef
struct delayed_data_statement_tag
{
AST a;
const decl_context_t* decl_context;
} delayed_data_statement_t;
static void delayed_compute_data_stmt(void * info, nodecl_t* nodecl_output)
{
delayed_data_statement_t *data = (delayed_data_statement_t*)info;
build_scope_data_stmt_do(data->a, data->decl_context, nodecl_output);
DELETE(data);
}
static void build_scope_data_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
delayed_data_statement_t *data = NEW(delayed_data_statement_t);
data->a = a;
data->decl_context = decl_context;
build_scope_delay_list_add(
DELAY_AFTER_EXECUTABLE_STATEMENTS, delayed_compute_data_stmt, data);
}
static void build_scope_deallocate_stmt(AST a,
const decl_context_t* decl_context,
nodecl_t* nodecl_output)
{
AST allocate_object_list = ASTSon0(a);
AST dealloc_opt_list = ASTSon1(a);
char error = 0;
nodecl_t nodecl_expr_list = nodecl_null();
AST it;
for_each_element(allocate_object_list, it)
{
AST allocate_object = ASTSon1(it);
if (ASTKind(allocate_object) == AST_DIMENSION_DECL)
{
sorry_printf_at(ast_get_locus(allocate_object),
"coarrays not supported\n");
}
AST data_ref = allocate_object;
nodecl_t nodecl_data_ref = nodecl_null();
fortran_check_expression(data_ref, decl_context, &nodecl_data_ref);
if (!nodecl_is_err_expr(nodecl_data_ref))
{
scope_entry_t* entry = fortran_data_ref_get_symbol(nodecl_data_ref);
if (entry == NULL
|| (!symbol_entity_specs_get_is_allocatable(entry)
&& !is_pointer_type(no_ref(entry->type_information))))
{
error_printf_at(ast_get_locus(a), "only ALLOCATABLE or POINTER can be used in a DEALLOCATE statement\n");
error = 1;
continue;
}
}
nodecl_expr_list = nodecl_append_to_list(nodecl_expr_list, 
nodecl_data_ref);
}
if (error)
{
*nodecl_output = nodecl_make_list_1(
nodecl_make_err_statement(ast_get_locus(a))
);
return;
}
nodecl_t nodecl_opt_value = nodecl_null();
handle_opt_value_list(a, dealloc_opt_list, decl_context, &nodecl_opt_value);
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_deallocate_statement(nodecl_expr_list, 
nodecl_opt_value, 
ast_get_locus(a)));
}
static char array_is_assumed_shape(scope_entry_t* entry, const decl_context_t* decl_context)
{
if (!fortran_is_array_type(no_ref(entry->type_information)))
return 0;
if (symbol_entity_specs_get_is_allocatable(entry))
return 1;
if (!symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
return 0;
type_t* t = no_ref(entry->type_information);
while (fortran_is_array_type(t))
{
if (!nodecl_is_null(array_type_get_array_lower_bound(t))
|| !nodecl_is_null(array_type_get_array_upper_bound(t)))
return 0;
t = array_type_get_element_type(t);
}
return 1;
}
static void copy_interface(scope_entry_t* orig, scope_entry_t* dest)
{
type_t* function_type = no_ref(orig->type_information);
if (is_pointer_type(function_type))
function_type = pointer_type_get_pointee_type(function_type);
ERROR_CONDITION(!is_function_type(function_type), "Function type is not", 0);
dest->type_information = function_type;
symbol_entity_specs_set_is_elemental(dest, symbol_entity_specs_get_is_elemental(orig));
symbol_entity_specs_set_is_pure(dest, symbol_entity_specs_get_is_pure(orig));
dest->related_decl_context = orig->related_decl_context;
symbol_entity_specs_copy_related_symbols_from(dest, orig);
int i, N = symbol_entity_specs_get_num_related_symbols(dest);
for (i = 0; i < N; i++)
{
scope_entry_t* param = symbol_entity_specs_get_related_symbols_num(dest, i);
symbol_set_as_parameter_of_function(param, dest,
0,
symbol_get_parameter_position_in_function(param, orig));
}
symbol_entity_specs_set_is_implicit_basic_type(dest, 0);
symbol_entity_specs_set_procedure_decl_stmt_proc_interface(dest, orig);
}
static void synthesize_procedure_type(
scope_entry_t* entry,
scope_entry_t* interface,
type_t* return_type,
UNUSED_PARAMETER const decl_context_t* decl_context,
char do_pointer,
char is_pass_proc_component)
{
char was_ref = is_lvalue_reference_type(entry->type_information);
if (interface == NULL)
{
type_t* new_type;
if (return_type == NULL)
{
new_type = get_nonproto_function_type(get_void_type(), 0);
symbol_entity_specs_set_is_implicit_basic_type(entry, 0);
}
else
{
new_type = get_nonproto_function_type(return_type, 0);
symbol_entity_specs_set_is_implicit_basic_type(entry, 0);
}
entry->type_information = new_type;
}
else
{
copy_interface(interface, entry);
}
if (do_pointer)
{
if (!is_pass_proc_component)
{
entry->type_information = get_pointer_type(entry->type_information);
}
else
{
entry->type_information = get_pointer_to_member_type(
entry->type_information,
symbol_entity_specs_get_class_type(entry));
}
}
if (was_ref)
{
entry->type_information = get_lvalue_reference_type(entry->type_information);
}
symbol_entity_specs_set_is_implicit_basic_type(entry, 0);
}
static void delayed_build_scope_derived_type_proc_component_def_(
AST component_def_stmt,
scope_entry_t *class_name,
char fields_are_private UNUSED_PARAMETER,
const decl_context_t *decl_context,
const decl_context_t *inner_decl_context)
{
ERROR_CONDITION(ASTKind(component_def_stmt) != AST_PROC_COMPONENT_DEF_STATEMENT, "Invalid tree", 0);
AST proc_interface = ASTSon0(component_def_stmt);
AST component_attr_spec_list = ASTSon1(component_def_stmt);
AST component_decl_list = ASTSon2(component_def_stmt);
type_t* return_type = NULL;
scope_entry_t* interface = NULL;
if (proc_interface != NULL)
{
if (ASTKind(proc_interface) == AST_SYMBOL)
{
interface = fortran_query_name_str(decl_context,
strtolower(ASTText(proc_interface)),
ast_get_locus(proc_interface));
if (interface != NULL
&& (interface->kind == SK_FUNCTION
|| (interface->kind == SK_VARIABLE
&& (is_function_type(no_ref(interface->type_information))
|| is_pointer_to_function_type(no_ref(interface->type_information))))))
{
type_t* function_type = no_ref(interface->type_information);
if (is_pointer_type(function_type))
{
function_type = pointer_type_get_pointee_type(function_type);
}
if (function_type_get_lacking_prototype(function_type))
{
error_printf_at(ast_get_locus(proc_interface), "'%s' does not have an explicit interface\n",
interface->symbol_name);
interface = NULL;
}
}
else
{
error_printf_at(ast_get_locus(proc_interface), "'%s' is not a valid procedure interface\n",
interface->symbol_name);
interface = NULL;
}
}
else
{
return_type = fortran_gather_type_from_declaration_type_spec(proc_interface,
decl_context,
NULL);
}
}
attr_spec_t attr_spec;
memset(&attr_spec, 0, sizeof(attr_spec));
if (component_attr_spec_list != NULL)
gather_attr_spec_list(component_attr_spec_list, decl_context, &attr_spec);
if (!attr_spec.is_pointer)
{
error_printf_at(ast_get_locus(component_attr_spec_list),
"a procedure component declaration must have the POINTER attribute\n");
return;
}
if (!attr_spec.is_nopass && interface == NULL)
{
error_printf_at(ast_get_locus(component_attr_spec_list),
"a procedure component with the PASS atribute must have a procedure interface\n");
return;
}
AST it2;
for_each_element(component_decl_list, it2)
{
AST name = ASTSon1(it2);
AST init = NULL;
if (ASTKind(name) == AST_PROCEDURE_DECL)
{
init = ASTSon1(name);
name = ASTSon0(name);
}
scope_entry_t* entry = get_symbol_for_name(inner_decl_context, name, ASTText(name));
entry->kind = SK_VARIABLE;
entry->defined = 1;
symbol_entity_specs_set_is_member(entry, 1);
symbol_entity_specs_set_class_type(entry, get_user_defined_type(class_name));
symbol_entity_specs_set_is_implicit_basic_type(entry, 0);
symbol_entity_specs_set_is_procedure_decl_stmt(entry, 1);
synthesize_procedure_type(entry, interface, return_type, decl_context,  1, !attr_spec.is_nopass);
if (init != NULL)
{
fortran_immediate_check_initialization(
entry,
init,
decl_context,
1,
0,
0,
0);
}
class_type_add_member(class_name->type_information,
entry,
entry->decl_context,
1);
}
}
typedef struct delayed_proc_component_data_tag
{
AST component_def_stmt;
scope_entry_t *class_name;
char fields_are_private;
const decl_context_t *decl_context;
const decl_context_t *inner_decl_context;
} delayed_proc_component_data_t;
static void delayed_build_scope_derived_type_proc_component_def(void *info, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST component_def_stmt = ((delayed_proc_component_data_t*)info)->component_def_stmt;
scope_entry_t *class_name =((delayed_proc_component_data_t*)info)->class_name;
char fields_are_private =((delayed_proc_component_data_t*)info)->fields_are_private;
const decl_context_t *decl_context =((delayed_proc_component_data_t*)info)->decl_context;
const decl_context_t *inner_decl_context = ((delayed_proc_component_data_t*)info)->inner_decl_context;
delayed_build_scope_derived_type_proc_component_def_(component_def_stmt, class_name,
fields_are_private, decl_context, inner_decl_context);
}
static void build_scope_derived_type_proc_component_def(
AST component_def_stmt,
scope_entry_t *class_name,
char fields_are_private UNUSED_PARAMETER,
const decl_context_t *decl_context,
const decl_context_t *inner_decl_context)
{
ERROR_CONDITION(ASTKind(component_def_stmt) != AST_PROC_COMPONENT_DEF_STATEMENT, "Invalid tree", 0);
delayed_proc_component_data_t *data = NEW0(delayed_proc_component_data_t);
data->component_def_stmt = component_def_stmt;
data->class_name = class_name;
data->fields_are_private = fields_are_private;
data->decl_context = decl_context;
data->inner_decl_context = inner_decl_context;
build_scope_delay_list_add(
DELAY_AFTER_DECLARATIONS, delayed_build_scope_derived_type_proc_component_def, data);
}
static void build_scope_derived_type_data_component_def(
AST component_def_stmt,
scope_entry_t *class_name,
char fields_are_private,
const decl_context_t *decl_context,
const decl_context_t *inner_decl_context)
{
ERROR_CONDITION(ASTKind(component_def_stmt)
!= AST_DATA_COMPONENT_DEF_STATEMENT,
"Invalid tree",
0);
AST declaration_type_spec = ASTSon0(component_def_stmt);
AST component_attr_spec_list = ASTSon1(component_def_stmt);
AST component_decl_list = ASTSon2(component_def_stmt);
attr_spec_t attr_spec;
memset(&attr_spec, 0, sizeof(attr_spec));
if (component_attr_spec_list != NULL)
{
gather_attr_spec_list(
component_attr_spec_list, decl_context, &attr_spec);
}
type_t *basic_type
= fortran_gather_type_from_declaration_type_spec_of_component(
declaration_type_spec, decl_context, attr_spec.is_pointer);
AST it2;
for_each_element(component_decl_list, it2)
{
attr_spec_t current_attr_spec = attr_spec;
AST declaration = ASTSon1(it2);
AST component_name = ASTSon0(declaration);
AST entity_decl_specs = ASTSon1(declaration);
scope_entry_t *entry
= new_fortran_symbol(inner_decl_context, ASTText(component_name));
entry->kind = SK_VARIABLE;
entry->locus = ast_get_locus(declaration);
entry->type_information = basic_type;
symbol_entity_specs_set_is_implicit_basic_type(entry, 0);
entry->defined = 1;
AST initialization = NULL;
AST array_spec = NULL;
AST coarray_spec = NULL;
AST char_length = NULL;
if (entity_decl_specs != NULL)
{
array_spec = ASTSon0(entity_decl_specs);
coarray_spec = ASTSon1(entity_decl_specs);
char_length = ASTSon2(entity_decl_specs);
initialization = ASTSon3(entity_decl_specs);
}
if (array_spec != NULL)
{
current_attr_spec.is_dimension = 1;
current_attr_spec.array_spec = array_spec;
}
if (coarray_spec != NULL)
{
if (current_attr_spec.is_codimension)
{
error_printf_at(ast_get_locus(declaration),
"CODIMENSION attribute specified twice\n");
}
else
{
current_attr_spec.is_codimension = 1;
current_attr_spec.coarray_spec = coarray_spec;
}
}
if (char_length != NULL)
{
if (!fortran_is_character_type(no_ref(entry->type_information)))
{
error_printf_at(
ast_get_locus(declaration),
"char-length specified but type is not CHARACTER\n");
}
if (ASTKind(char_length) != AST_SYMBOL
|| strcmp(ASTText(char_length), "*") != 0)
{
nodecl_t nodecl_char_length = nodecl_null();
fortran_check_expression(
char_length, decl_context, &nodecl_char_length);
nodecl_char_length
= fortran_expression_as_value(nodecl_char_length);
nodecl_t lower_bound = nodecl_make_integer_literal(
get_signed_int_type(),
const_value_get_one(type_get_size(get_signed_int_type()),
1),
ast_get_locus(char_length));
entry->type_information = get_array_type_bounds(
array_type_get_element_type(entry->type_information),
lower_bound,
nodecl_char_length,
decl_context);
}
else
{
entry->type_information = get_array_type(
array_type_get_element_type(entry->type_information),
nodecl_null(),
decl_context);
}
}
if (current_attr_spec.is_codimension)
{
error_printf_at(ast_get_locus(declaration),
"sorry: coarrays are not supported\n");
}
if (current_attr_spec.is_asynchronous)
{
error_printf_at(ast_get_locus(declaration),
"sorry: ASYNCHRONOUS attribute not supported\n");
}
if (current_attr_spec.is_dimension
&& !is_error_type(entry->type_information))
{
compute_type_from_array_spec(entry,
entry->type_information,
current_attr_spec.array_spec,
decl_context,
0);
}
if (current_attr_spec.is_allocatable)
{
if (is_pointer_type(entry->type_information))
{
error_printf_at(
ast_get_locus(declaration),
"attribute POINTER conflicts with ALLOCATABLE\n");
}
else
{
symbol_entity_specs_set_is_allocatable(entry, 1);
entry->kind = SK_VARIABLE;
}
}
if (symbol_entity_specs_get_is_allocatable(entry)
&& fortran_is_array_type(entry->type_information))
{
check_array_type_is_valid_for_allocatable(
entry->type_information, entry, ast_get_locus(declaration));
}
symbol_entity_specs_set_is_target(entry, current_attr_spec.is_target);
if (fields_are_private
&& symbol_entity_specs_get_access(entry) == AS_UNKNOWN)
{
symbol_entity_specs_set_access(entry, AS_PRIVATE);
}
if (current_attr_spec.is_pointer
&& !is_error_type(entry->type_information))
{
if (symbol_entity_specs_get_is_allocatable(entry))
{
error_printf_at(
ast_get_locus(declaration),
"attribute ALLOCATABLE conflicts with POINTER\n");
}
else
{
entry->type_information
= get_pointer_type(entry->type_information);
}
}
if (fortran_is_pointer_to_array_type(entry->type_information))
{
check_array_type_is_valid_for_pointer(
entry->type_information, entry, ast_get_locus(declaration));
}
symbol_entity_specs_set_is_member(entry, 1);
symbol_entity_specs_set_class_type(entry,
get_user_defined_type(class_name));
if (current_attr_spec.is_contiguous)
{
if (!array_is_assumed_shape(entry, decl_context)
&& !fortran_is_pointer_to_array_type(entry->type_information))
{
error_printf_at(
ast_get_locus(component_name),
"CONTIGUOUS attribute is only valid for pointers to arrays "
"or assumed-shape arrays\n");
}
symbol_entity_specs_set_is_contiguous(entry, 1);
}
if (initialization != NULL)
{
entry->kind = SK_VARIABLE;
if (ASTKind(initialization) == AST_POINTER_INITIALIZATION
&& current_attr_spec.is_pointer)
{
initialization = ASTSon0(initialization);
fortran_delay_check_initialization(
entry,
initialization,
decl_context,
1,
0,
0,
0);
}
else if (current_attr_spec.is_pointer)
{
error_printf_at(ast_get_locus(initialization),
"a POINTER must be initialized using pointer "
"initialization\n");
}
else if (ASTKind(initialization) == AST_POINTER_INITIALIZATION)
{
error_printf_at(ast_get_locus(initialization),
"no POINTER attribute, required for pointer "
"initialization\n");
}
else
{
fortran_delay_check_initialization(
entry,
initialization,
decl_context,
0,
0,
0,
0);
}
}
class_type_add_member(class_name->type_information,
entry,
entry->decl_context,
1);
}
}
static void build_scope_derived_type_procedure_binding_def_specific(
AST type_bound_proc_binding_def,
scope_entry_t *class_name,
char proc_bindings_are_private,
const decl_context_t *decl_context,
const decl_context_t *inner_decl_context)
{
ERROR_CONDITION(ASTKind(type_bound_proc_binding_def)
!= AST_TYPE_BOUND_PROCEDURE_STATEMENT,
"Invalid node",
0);
AST interface_name = ASTSon0(type_bound_proc_binding_def);
AST binding_attr_list = ASTSon1(type_bound_proc_binding_def);
AST type_bound_proc_decl_list = ASTSon2(type_bound_proc_binding_def);
if (interface_name != NULL)
{
sorry_printf_at(ast_get_locus(interface_name),
"DEFERRED type-bound procedures not implemented yet\n");
}
attr_spec_t attr_spec;
memset(&attr_spec, 0, sizeof(attr_spec));
if (binding_attr_list != NULL)
{
gather_attr_spec_list(binding_attr_list, decl_context, &attr_spec);
}
if (!attr_spec.is_nopass && attr_spec.pass_name != NULL)
{
sorry_printf_at(ast_get_locus(binding_attr_list),
"PASS with a pass-name is not implemented yet\n");
}
AST it;
for_each_element(type_bound_proc_decl_list, it)
{
AST type_bound_proc_decl = ASTSon1(it);
ERROR_CONDITION(ASTKind(type_bound_proc_decl) != AST_SYMBOL
&& ASTKind(type_bound_proc_decl) != AST_RENAME,
"Invalid node", 0);
AST binding_name = type_bound_proc_decl;
AST procedure_name = type_bound_proc_decl;
if (ASTKind(type_bound_proc_decl) == AST_RENAME)
{
binding_name = ASTSon0(type_bound_proc_decl);
procedure_name = ASTSon1(type_bound_proc_decl);
}
scope_entry_t *binding_entry
= new_fortran_symbol(inner_decl_context, ASTText(binding_name));
binding_entry->kind = SK_FUNCTION;
binding_entry->locus = ast_get_locus(binding_name);
binding_entry->defined = 1;
symbol_entity_specs_set_is_member(binding_entry, 1);
if (attr_spec.is_nopass)
symbol_entity_specs_set_is_static(binding_entry, 1);
if (attr_spec.is_deferred)
symbol_entity_specs_set_is_virtual(binding_entry, 1);
if (attr_spec.is_non_overridable)
symbol_entity_specs_set_is_final(binding_entry, 1);
if (proc_bindings_are_private)
symbol_entity_specs_set_access(binding_entry, AS_PRIVATE);
class_type_add_member(class_name->type_information,
binding_entry,
binding_entry->decl_context,
1);
scope_entry_t *procedure_entry = NULL;
scope_entry_list_t *entry_list
= query_in_scope_str_flags(decl_context,
strtolower(ASTText(procedure_name)),
NULL,
DF_ONLY_CURRENT_SCOPE);
if (entry_list != NULL)
{
ERROR_CONDITION(
entry_list_size(entry_list) != 1, "Unhandled size list", 0);
procedure_entry = entry_list_head(entry_list);
entry_list_free(entry_list);
}
else
{
procedure_entry = create_fortran_symbol_for_name_(
decl_context,
procedure_name,
strtolower(ASTText(procedure_name)),
1);
procedure_entry->kind = SK_UNDEFINED;
add_delay_check_fully_defined_symbol(decl_context, procedure_entry);
}
symbol_entity_specs_set_alias_to(binding_entry, procedure_entry);
}
}
static void build_scope_derived_type_procedure_binding_def(
AST type_bound_proc_binding_def,
scope_entry_t *class_name,
char proc_bindings_are_private,
const decl_context_t *decl_context,
const decl_context_t *inner_decl_context)
{
switch (ASTKind(type_bound_proc_binding_def))
{
case AST_TYPE_BOUND_PROCEDURE_STATEMENT:
build_scope_derived_type_procedure_binding_def_specific(
type_bound_proc_binding_def,
class_name,
proc_bindings_are_private,
decl_context,
inner_decl_context);
break;
case AST_TYPE_BOUND_GENERIC_PROCEDURE:
sorry_printf_at(ast_get_locus(type_bound_proc_binding_def),
"generic type-bound procedure bindings are not "
"implemented yet\n");
break;
case AST_FINAL_STATEMENT:
sorry_printf_at(
ast_get_locus(type_bound_proc_binding_def),
"FINAL type-bound procedures not implemented yet\n");
break;
default:
internal_error("Code unreachable", 0);
}
}
static void build_scope_derived_type_def(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST derived_type_stmt = ASTSon0(a);
AST type_attr_spec_list = ASTSon0(derived_type_stmt);
AST name = ASTSon1(derived_type_stmt);
AST type_param_name_list = ASTSon2(derived_type_stmt);
if (type_param_name_list != NULL)
{
sorry_printf_at(ast_get_locus(type_param_name_list),
"derived types with type-parameters are not supported\n");
}
attr_spec_t attr_spec;
memset(&attr_spec, 0, sizeof(attr_spec));
AST it;
if (type_attr_spec_list != NULL)
{
for_each_element(type_attr_spec_list, it)
{
AST type_attr_spec = ASTSon1(it);
switch (ASTKind(type_attr_spec))
{
case AST_ABSTRACT:
{
sorry_printf_at(
ast_get_locus(type_attr_spec),
"ABSTRACT derived types are not supported\n");
break;
}
case AST_ATTR_SPEC:
{
gather_attr_spec_item(type_attr_spec, decl_context, &attr_spec);
break;
}
default:
{
internal_error("%s: unexpected tree\n",
ast_location(type_attr_spec));
}
}
}
}
scope_entry_t* class_name = fortran_query_name_str(decl_context, ASTText(name),
ast_get_locus(name));
if (class_name != NULL)
{
if (class_name->kind == SK_UNDEFINED
&& symbol_entity_specs_get_is_implicit_basic_type(class_name))
{
}
else if (class_name->kind != SK_CLASS)
{
if (class_name->decl_context->current_scope != class_name->decl_context->global_scope)
{
error_printf_at(ast_get_locus(name), "name '%s' is not a type name\n",
ASTText(name));
return;
}
else
{
class_name = NULL;
}
}
else if (class_name->defined)
{
if (decl_context->current_scope == class_name->decl_context->current_scope
|| symbol_entity_specs_get_from_module(class_name) != NULL)
{
error_printf_at(ast_get_locus(name), "derived type 'TYPE(%s)' already defined\n",
ASTText(name));
return;
}
else
{
class_name = NULL;
}
}
}
if (class_name == NULL)
{
class_name = new_fortran_symbol(decl_context, ASTText(name));
}
class_name->kind = SK_CLASS;
class_name->locus = ast_get_locus(name);
class_name->type_information = get_new_class_type(decl_context, TT_STRUCT);
if (!nodecl_is_null(attr_spec.bind_info)
&& !nodecl_is_err_expr(attr_spec.bind_info))
symbol_entity_specs_set_bind_info(class_name, attr_spec.bind_info);
class_name->defined = 1;
if (attr_spec.is_public)
{
symbol_entity_specs_set_access(class_name, AS_PUBLIC);
}
else if (attr_spec.is_private)
{
symbol_entity_specs_set_access(class_name, AS_PRIVATE);
}
AST type_param_def_stmt_seq = NULL,
private_or_sequence_seq = NULL,
component_part = NULL,
type_bound_procedure_part = NULL;
AST derived_type_body = ASTSon1(a);
if (derived_type_body != NULL)
{
type_param_def_stmt_seq = ASTSon0(derived_type_body);
private_or_sequence_seq = ASTSon1(derived_type_body);
component_part = ASTSon2(derived_type_body);
type_bound_procedure_part = ASTSon3(derived_type_body);
}
if (type_param_def_stmt_seq != NULL)
{
sorry_printf_at(ast_get_locus(type_param_def_stmt_seq),
"type-parameter definitions are not supported\n");
}
char is_sequence = 0;
char fields_are_private = 0;
if (private_or_sequence_seq != NULL)
{
for_each_element(private_or_sequence_seq, it)
{
AST private_or_sequence = ASTSon1(it);
if (ASTKind(private_or_sequence) == AST_SEQUENCE_STATEMENT)
{
if (is_sequence)
{
error_printf_at(ast_get_locus(private_or_sequence), "SEQUENCE statement specified twice\n");
}
is_sequence = 1;
}
else if (ASTKind(private_or_sequence) == AST_ACCESS_STATEMENT)
{
if (fields_are_private)
{
error_printf_at(ast_get_locus(private_or_sequence), "PRIVATE statement specified twice\n");
}
fields_are_private = 1;
}
else
{
internal_error("%s: Unexpected statement '%s'\n", 
ast_location(private_or_sequence),
ast_print_node_type(ASTKind(private_or_sequence)));
}
}
}
const decl_context_t* inner_decl_context = new_class_context(class_name->decl_context, class_name);
class_type_set_inner_context(class_name->type_information, inner_decl_context);
if (component_part != NULL)
{
for_each_element(component_part, it)
{
AST component_def_stmt = ASTSon1(it);
if (ASTKind(component_def_stmt) == AST_PROC_COMPONENT_DEF_STATEMENT)
{
build_scope_derived_type_proc_component_def(
component_def_stmt, class_name, fields_are_private, decl_context, inner_decl_context);
}
else if (ASTKind(component_def_stmt) == AST_DATA_COMPONENT_DEF_STATEMENT)
{
build_scope_derived_type_data_component_def(
component_def_stmt, class_name, fields_are_private, decl_context, inner_decl_context);
}
else
{
internal_error("Unexpected '%s' node\n", ast_print_node_type(ASTKind(component_def_stmt)));
}
}
}
if (type_bound_procedure_part != NULL)
{
AST private_stmt = ASTSon0(type_bound_procedure_part);
AST type_bound_proc_binding_seq = ASTSon1(type_bound_procedure_part);
char proc_bindings_are_private = private_stmt != NULL;
for_each_element(type_bound_proc_binding_seq, it)
{
AST type_bound_proc_binding_def = ASTSon1(it);
build_scope_derived_type_procedure_binding_def(
type_bound_proc_binding_def,
class_name,
proc_bindings_are_private,
decl_context,
inner_decl_context);
}
}
set_is_complete_type(class_name->type_information, 1);
class_type_set_is_packed(class_name->type_information, is_sequence);
if (decl_context->current_scope->related_entry != NULL
&& decl_context->current_scope->related_entry->kind == SK_MODULE)
{
scope_entry_t* module = decl_context->current_scope->related_entry;
symbol_entity_specs_add_related_symbols(module, class_name);
symbol_entity_specs_set_in_module(class_name, module);
}
}
static void build_scope_dimension_stmt(AST a, const decl_context_t* decl_context,
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST array_name_dim_spec_list = ASTSon0(a);
AST it;
for_each_element(array_name_dim_spec_list, it)
{
AST dimension_decl = ASTSon1(it);
AST name = ASTSon0(dimension_decl);
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
if (fortran_is_array_type(no_ref(entry->type_information))
|| fortran_is_pointer_to_array_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(name), "entity '%s' already has a DIMENSION attribute\n",
ASTText(name));
continue;
}
char was_ref = is_lvalue_reference_type(entry->type_information);
char is_pointer = is_pointer_type(no_ref(entry->type_information));
if (is_pointer_type(no_ref(entry->type_information)))
{
entry->type_information = pointer_type_get_pointee_type(no_ref(entry->type_information));
}
char is_parameter = is_const_qualified_type(no_ref(entry->type_information));
AST array_spec = ASTSon1(dimension_decl);
compute_type_from_array_spec(
entry,
no_ref(entry->type_information), 
array_spec,
decl_context,
!is_parameter);
if (entry->kind == SK_UNDEFINED)
{
entry->kind = SK_VARIABLE;
}
if (!is_error_type(entry->type_information))
{
if (is_pointer)
{
entry->type_information = get_pointer_type(no_ref(entry->type_information));
}
if (fortran_is_pointer_to_array_type(entry->type_information))
{
check_array_type_is_valid_for_pointer(
entry->type_information,
entry,
ast_get_locus(dimension_decl));
}
if (symbol_entity_specs_get_is_allocatable(entry)
&& fortran_is_array_type(entry->type_information))
{
check_array_type_is_valid_for_allocatable(entry->type_information,
entry,
ast_get_locus(dimension_decl));
}
if (was_ref)
{
entry->type_information = get_lvalue_reference_type(entry->type_information);
}
}
}
}
static void build_scope_do_construct(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST loop_control = ASTSon0(a);
AST block = ASTSon1(a);
AST end_do_statement = ASTSon2(a);
AST do_variable = ASTSon0(loop_control);
AST lower = ASTSon1(loop_control);
AST upper = ASTSon2(loop_control);
AST stride = ASTSon3(loop_control);
const char* construct_name = ASTText(a);
char error_signaled = 0;
nodecl_t nodecl_named_label = nodecl_null();
if (construct_name != NULL)
{
scope_entry_t* named_label = fortran_query_construct_name_str(
construct_name, decl_context,  1,
ast_get_locus(a));
nodecl_named_label = nodecl_make_symbol(named_label, ast_get_locus(a));
}
scope_entry_t* ind_var = NULL;
if (do_variable != NULL)
{
nodecl_t nodecl_var = nodecl_null();
fortran_check_expression(do_variable, decl_context, &nodecl_var);
if (!nodecl_is_err_expr(nodecl_var))
{
ind_var = fortran_data_ref_get_symbol(nodecl_var);
if (ind_var != NULL
&& !is_integer_type(no_ref(ind_var->type_information)))
{
warn_printf_at(ast_get_locus(a), "loop variable '%s' should be of integer type\n",
codegen_to_str(nodecl_var, nodecl_retrieve_context(nodecl_var)));
}
}
else
{
error_signaled = 1;
}
}
char unbounded_loop = lower == NULL
&& upper == NULL
&& stride == NULL;
nodecl_t nodecl_lower = nodecl_null();
if (lower != NULL)
{
fortran_check_expression(lower, decl_context, &nodecl_lower);
if (nodecl_is_err_expr(nodecl_lower))
{
error_signaled = 1;
}
else
{
nodecl_lower = fortran_expression_as_value(nodecl_lower);
}
}
nodecl_t nodecl_upper = nodecl_null();
if (upper != NULL)
{
fortran_check_expression(upper, decl_context, &nodecl_upper);
if (nodecl_is_err_expr(nodecl_upper))
{
error_signaled = 1;
}
else
{
nodecl_upper = fortran_expression_as_value(nodecl_upper);
}
}
nodecl_t nodecl_stride = nodecl_null();
if (stride != NULL)
{
fortran_check_expression(stride, decl_context, &nodecl_stride);
if (nodecl_is_err_expr(nodecl_stride))
{
error_signaled = 1;
}
else
{
nodecl_stride = fortran_expression_as_value(nodecl_stride);
}
}
else
{
nodecl_stride = const_value_to_nodecl(const_value_get_one( fortran_get_default_integer_type_kind(),  1));
}
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement_inside_block_context(block, decl_context, &nodecl_statement);
if (error_signaled)
{
*nodecl_output
= nodecl_make_list_1(
nodecl_make_err_statement(ast_get_locus(a)));
return;
}
if (end_do_statement != NULL
&& ASTKind(end_do_statement) == AST_LABELED_STATEMENT)
{
AST label = ASTSon0(end_do_statement);
scope_entry_t* label_sym = fortran_query_label(label, decl_context,  1);
nodecl_t nodecl_labeled_empty_statement = 
nodecl_make_labeled_statement(
nodecl_make_list_1(
nodecl_make_empty_statement(ast_get_locus(end_do_statement))
),
label_sym,
ast_get_locus(end_do_statement));
nodecl_statement = nodecl_append_to_list(nodecl_statement, nodecl_labeled_empty_statement);
}
if (!unbounded_loop)
{
nodecl_t nodecl_ind_var = nodecl_make_symbol(ind_var, ast_get_locus(loop_control));
nodecl_set_type(nodecl_ind_var, lvalue_ref(ind_var->type_information));
*nodecl_output =
nodecl_make_list_1(
nodecl_make_for_statement(
nodecl_make_range_loop_control(
nodecl_ind_var,
nodecl_lower,
nodecl_upper,
nodecl_stride,
ast_get_locus(loop_control)),
nodecl_statement,
nodecl_named_label,
ast_get_locus(a)));
}
else 
{
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_for_statement(
nodecl_make_unbounded_loop_control(
ast_get_locus(loop_control)),
nodecl_statement,
nodecl_named_label,
ast_get_locus(a)));
}
}
static void build_scope_entry_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
static scope_entry_t error_entry_;
static scope_entry_t *error_entry = &error_entry_;
if (nodecl_get_symbol(_nodecl_wrap(a)) == NULL)
{
AST name = ASTSon0(a);
AST dummy_arg_list = ASTSon1(a);
AST suffix = ASTSon2(a);
scope_entry_t* related_sym = decl_context->current_scope->related_entry; 
if (related_sym == NULL)
{
internal_error("%s: error: code unreachable\n", 
ast_location(a));
}
else if (related_sym->kind == SK_PROGRAM) 
{
error_printf_at(ast_get_locus(a), "entry statement '%s' cannot appear within a program\n",
ASTText(name));
nodecl_set_symbol(_nodecl_wrap(a), error_entry);
return;
}
scope_entry_t* entry = new_entry_symbol(decl_context, name, suffix, dummy_arg_list, related_sym);
if (entry != NULL)
{
if (symbol_entity_specs_get_is_module_procedure(related_sym))
{
insert_entry(symbol_entity_specs_get_in_module(related_sym)->related_decl_context->current_scope, entry);
}
nodecl_set_symbol(_nodecl_wrap(a), entry);
}
else
{
nodecl_set_symbol(_nodecl_wrap(a), error_entry);
}
}
else
{
scope_entry_t* entry = nodecl_get_symbol(_nodecl_wrap(a));
nodecl_set_symbol(_nodecl_wrap(a), NULL);
if (entry == error_entry)
return;
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_entry_statement(entry, ast_get_locus(a))
);
}
}
static void build_scope_enum_def(AST a,
const decl_context_t *decl_context
UNUSED_PARAMETER,
nodecl_t *nodecl_output UNUSED_PARAMETER)
{
AST enum_def_stmt = ASTSon0(a);
AST enumerator_def_stmt_seq = ASTSon1(a);
static int num_enums = 0;
const char *enum_name = NULL;
uniquestr_sprintf(&enum_name, ".fortran_enum_%d", num_enums);
num_enums++;
scope_entry_t *new_enum
= new_symbol(decl_context, decl_context->current_scope, enum_name);
new_enum->locus = ast_get_locus(enum_def_stmt);
new_enum->kind = SK_ENUM;
new_enum->type_information
= get_new_enum_type(decl_context,  0);
symbol_entity_specs_set_is_user_declared(new_enum, 1);
enum_type_set_underlying_type(new_enum->type_information,
get_signed_int_type());
type_t *enum_type = get_user_defined_type(new_enum);
const_value_t *current_enum_value = const_value_get_signed_int(0);
if (decl_context->current_scope->related_entry != NULL
&& decl_context->current_scope->related_entry->kind == SK_MODULE)
{
scope_entry_t *module = decl_context->current_scope->related_entry;
symbol_entity_specs_add_related_symbols(module, new_enum);
symbol_entity_specs_set_in_module(new_enum, module);
}
AST it;
for_each_element(enumerator_def_stmt_seq, it)
{
AST enumerator_def_stmt = ASTSon1(it);
AST enumerator_list = ASTSon0(enumerator_def_stmt);
AST enumerator_it;
for_each_element(enumerator_list, enumerator_it)
{
AST enumerator = ASTSon1(enumerator_it);
AST name = ASTSon0(enumerator);
const char *enumerator_name = strtolower(ASTText(name));
AST expr = ASTSon1(enumerator);
scope_entry_t *existing_name = fortran_query_name_str(decl_context, enumerator_name, ast_get_locus(name));
if (existing_name != NULL
&& existing_name->kind != SK_UNDEFINED)
{
error_printf_at(ast_get_locus(name), "name '%s' has already been defined\n", ASTText(name));
continue;
}
scope_entry_t* new_enumerator = existing_name;
if (existing_name == NULL)
{
new_enumerator = new_symbol(decl_context, decl_context->current_scope, enumerator_name);
}
new_enumerator->kind = SK_ENUMERATOR;
new_enumerator->locus = ast_get_locus(name);
new_enumerator->type_information = enum_type;
nodecl_t nodecl_expr = nodecl_null();
if (expr != NULL)
{
fortran_check_expression(expr, decl_context, &nodecl_expr);
if (nodecl_is_err_expr(nodecl_expr))
{
nodecl_expr = nodecl_null();
}
else if (!nodecl_is_constant(nodecl_expr)
|| !is_integer_type(nodecl_get_type(nodecl_expr)))
{
error_printf_at(
nodecl_get_locus(nodecl_expr),
"value of enumerator must be an integer constant");
nodecl_expr = nodecl_null();
}
}
if (!nodecl_is_null(nodecl_expr))
{
const_value_t *val = const_value_cast_to_signed_int_value(
nodecl_get_constant(nodecl_expr));
new_enumerator->value = const_value_to_nodecl(val);
current_enum_value = val;
}
else
{
new_enumerator->value = const_value_to_nodecl(current_enum_value);
}
current_enum_value = const_value_add(current_enum_value,
const_value_get_signed_int(1));
enum_type_add_enumerator(enum_type, new_enumerator);
if (decl_context->current_scope->related_entry != NULL
&& decl_context->current_scope->related_entry->kind == SK_MODULE)
{
scope_entry_t* module = decl_context->current_scope->related_entry;
symbol_entity_specs_add_related_symbols(module, new_enumerator);
symbol_entity_specs_set_in_module(new_enumerator, module);
}
}
}
}
static void do_build_scope_equivalence_stmt(AST a, 
const decl_context_t* decl_context, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST equivalence_set_list = ASTSon0(a);
scope_entry_t* equivalence_info = get_or_create_equivalence_symbol_info(decl_context);
AST it;
for_each_element(equivalence_set_list, it)
{
AST equivalence_set = ASTSon1(it);
AST equivalence_object = ASTSon0(equivalence_set);
AST equivalence_object_list = ASTSon1(equivalence_set);
nodecl_t nodecl_equivalence_object = nodecl_null();
fortran_check_expression(equivalence_object, decl_context, &nodecl_equivalence_object);
nodecl_t nodecl_equivalence_set = nodecl_null();
AST it2;
for_each_element(equivalence_object_list, it2)
{
AST equiv_obj = ASTSon1(it2);
nodecl_t nodecl_current_equiv_obj = nodecl_null();
fortran_check_expression(equiv_obj, decl_context, &nodecl_current_equiv_obj);
nodecl_equivalence_set = nodecl_append_to_list(nodecl_equivalence_set, 
nodecl_current_equiv_obj);
}
nodecl_t nodecl_equivalence = nodecl_make_fortran_equivalence(
nodecl_equivalence_object,
nodecl_equivalence_set,
ast_get_locus(equivalence_set));
equivalence_info->value = nodecl_append_to_list(equivalence_info->value, 
nodecl_equivalence);
}
}
typedef
struct delayed_equivalence_statement_tag
{
AST a;
const decl_context_t* decl_context;
} delayed_equivalence_statement_t;
static void delayed_equivalence_statement(void *info, nodecl_t* nodecl_output)
{
delayed_equivalence_statement_t* data = (delayed_equivalence_statement_t*)info;
do_build_scope_equivalence_stmt(data->a, data->decl_context, nodecl_output);
DELETE(data);
}
static void build_scope_equivalence_stmt(AST a,
const decl_context_t* decl_context,
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
delayed_equivalence_statement_t *data
= NEW(delayed_equivalence_statement_t);
data->a = a;
data->decl_context = decl_context;
build_scope_delay_list_add(
DELAY_AFTER_DECLARATIONS, delayed_equivalence_statement, data);
}
static void build_scope_exit_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST loop_name = ASTSon0(a);
nodecl_t nodecl_construct_name = get_construct_name(loop_name, decl_context);
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_break_statement(
nodecl_construct_name,
ast_get_locus(a)));
}
static void build_scope_external_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST name_list = ASTSon0(a);
AST it;
for_each_element(name_list, it)
{
AST name = ASTSon1(it);
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
if (entry->kind == SK_FUNCTION)
{
if (symbol_entity_specs_get_is_builtin(entry))
{
error_printf_at(ast_get_locus(name), "entity '%s' already has INTRINSIC attribute and INTRINSIC attribute conflicts with EXTERNAL attribute\n",
entry->symbol_name);
continue;
}
else if (symbol_entity_specs_get_is_extern(entry))
{
error_printf_at(ast_get_locus(name), "entity '%s' already has EXTERNAL attribute\n",
entry->symbol_name);
continue;
}
}
else if (entry->kind == SK_VARIABLE)
{
if (symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
if (is_function_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(name), "entity '%s' already has EXTERNAL attribute\n",
entry->symbol_name);
continue;
}
}
else
{
error_printf_at(ast_get_locus(name), "entity '%s' cannot have EXTERNAL attribute\n",
entry->symbol_name);
continue;
}
}
if (entry->kind == SK_UNDEFINED)
{
if (!symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
entry->kind = SK_FUNCTION;
symbol_entity_specs_set_is_extern(entry, 1);
}
else
{
entry->kind = SK_VARIABLE;
}
}
type_t* type = entry->type_information;
char was_ref = 0;
if (is_lvalue_reference_type(type))
{
was_ref = 1;
type = no_ref(type);
}
char was_pointer = 0;
if (is_pointer_type(type))
{
was_pointer = 1;
type = pointer_type_get_pointee_type(type);
}
type_t* new_type = NULL;
if (is_void_type(type))
{
new_type = get_nonproto_function_type(get_void_type(), 0);
}
else
{
new_type = get_nonproto_function_type(type, 0);
}
if (was_pointer)
{
new_type = get_pointer_type(new_type);
}
if (was_ref)
{
new_type = get_lvalue_reference_type(new_type);
}
entry->type_information = new_type;
}
}
static void build_scope_forall_header(AST a, const decl_context_t* decl_context, 
nodecl_t* loop_control_list, nodecl_t* nodecl_mask_expr)
{
AST type_spec = ASTSon0(a);
if (type_spec != NULL)
{
sorry_printf_at(ast_get_locus(a),
"type-specifier not supported in FORALL header\n");
}
AST forall_triplet_list = ASTSon1(a);
AST mask_expr = ASTSon2(a);
AST it;
for_each_element(forall_triplet_list, it)
{
AST forall_triplet_spec = ASTSon1(it);
AST name = ASTSon0(forall_triplet_spec);
AST forall_lower = ASTSon1(forall_triplet_spec);
AST forall_upper = ASTSon2(forall_triplet_spec);
AST forall_step  = ASTSon3(forall_triplet_spec);
nodecl_t nodecl_name = nodecl_null();
fortran_check_expression(name, decl_context, &nodecl_name);
nodecl_t nodecl_lower = nodecl_null();
fortran_check_expression(forall_lower, decl_context, &nodecl_lower);
nodecl_lower = fortran_expression_as_value(nodecl_lower);
nodecl_t nodecl_upper = nodecl_null();
fortran_check_expression(forall_upper, decl_context, &nodecl_upper);
nodecl_upper = fortran_expression_as_value(nodecl_upper);
nodecl_t nodecl_step = nodecl_null();
if (forall_step != NULL)
{
fortran_check_expression(forall_step, decl_context, &nodecl_step);
nodecl_step = fortran_expression_as_value(nodecl_step);
}
nodecl_t nodecl_triplet = nodecl_make_range_loop_control(
nodecl_name,
nodecl_lower,
nodecl_upper,
nodecl_step,
ast_get_locus(a));
*loop_control_list = nodecl_append_to_list(*loop_control_list,
nodecl_triplet);
}
if (mask_expr != NULL)
{
fortran_check_expression(mask_expr, decl_context, nodecl_mask_expr);
*nodecl_mask_expr = fortran_expression_as_value(*nodecl_mask_expr);
}
}
static void build_scope_forall_construct(AST a, 
const decl_context_t* decl_context, 
nodecl_t* nodecl_output)
{
AST forall_construct_stmt = ASTSon0(a);
AST forall_body_construct_seq = ASTSon1(a);
AST forall_header = ASTSon1(forall_construct_stmt);
nodecl_t nodecl_mask = nodecl_null();
nodecl_t nodecl_loop_control_list = nodecl_null();
build_scope_forall_header(forall_header, decl_context, 
&nodecl_loop_control_list, &nodecl_mask);
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement_inside_block_context(forall_body_construct_seq, decl_context, &nodecl_statement);
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_forall(nodecl_loop_control_list, 
nodecl_mask, 
nodecl_statement,
ast_get_locus(a)));
}
static void build_scope_forall_stmt(AST a, 
const decl_context_t* decl_context, 
nodecl_t* nodecl_output)
{
AST forall_header = ASTSon0(a);
AST forall_assignment_stmts = ASTSon1(a);
nodecl_t nodecl_mask = nodecl_null();
nodecl_t nodecl_loop_control_list = nodecl_null();
build_scope_forall_header(forall_header, decl_context, 
&nodecl_loop_control_list, &nodecl_mask);
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement_inside_block_context(forall_assignment_stmts, decl_context, &nodecl_statement);
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_forall(nodecl_loop_control_list, 
nodecl_mask, 
nodecl_statement,
ast_get_locus(a)));
}
static void build_scope_format_stmt(AST a,
const decl_context_t* decl_context, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST label = ASTSon0(a);
AST format = ASTSon1(a);
scope_entry_t* label_sym = fortran_query_label(label, decl_context,  0);
label_sym->value = nodecl_make_text(ASTText(format), ast_get_locus(format));
}
static void build_scope_goto_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
scope_entry_t* label_symbol = fortran_query_label(ASTSon0(a), decl_context,  0);
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_goto_statement(label_symbol, ast_get_locus(a)));
}
static void build_scope_if_construct(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST logical_expr = ASTSon0(a);
AST then_statement = ASTSon1(a);
AST else_statement = ASTSon2(a);
AST endif_statement = ASTSon3(a);
const char* construct_name = ASTText(a);
if (construct_name != NULL)
{
fortran_query_construct_name_str(
construct_name, decl_context,  1,
ast_get_locus(a));
}
nodecl_t nodecl_logical_expr = nodecl_null();
fortran_check_expression(logical_expr, decl_context, &nodecl_logical_expr);
nodecl_logical_expr = fortran_expression_as_value(nodecl_logical_expr);
nodecl_t nodecl_then = nodecl_null();
fortran_build_scope_statement_inside_block_context(then_statement, decl_context, &nodecl_then);
nodecl_t nodecl_else = nodecl_null();
if (else_statement != NULL)
{
fortran_build_scope_statement_inside_block_context(else_statement, decl_context, &nodecl_else);
}
if (!nodecl_is_list(nodecl_then))
{
nodecl_then = nodecl_make_list_1(nodecl_then);
}
if (!nodecl_is_list(nodecl_else))
{
nodecl_else = nodecl_make_list_1(nodecl_else);
}
if (endif_statement != NULL
&& ASTKind(endif_statement) == AST_LABELED_STATEMENT)
{
AST label = ASTSon0(endif_statement);
scope_entry_t* label_sym = fortran_query_label(label, decl_context,  1);
nodecl_t nodecl_labeled_empty_statement = 
nodecl_make_labeled_statement(
nodecl_make_list_1(
nodecl_make_empty_statement(ast_get_locus(endif_statement))
),
label_sym,
ast_get_locus(endif_statement));
if (else_statement == NULL)
{
nodecl_then = nodecl_append_to_list(nodecl_then, nodecl_labeled_empty_statement);
}
else
{
nodecl_else = nodecl_append_to_list(nodecl_else, nodecl_labeled_empty_statement);
}
}
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_if_else_statement(
nodecl_logical_expr,
nodecl_then,
nodecl_else,
ast_get_locus(a)));
}
static void delay_update_implicit_type(void *data,
UNUSED_PARAMETER nodecl_t *nodecl_out)
{
implicit_update_info_t *info = (implicit_update_info_t *)data;
scope_entry_t *entry = info->entry;
const decl_context_t *decl_context = info->decl_context;
DEBUG_CODE()
{
fprintf(stderr,
"BUILDSCOPE: Need to update existing implicit type of symbol "
"'%s' at '%s'\n",
entry->symbol_name,
locus_to_str(entry->locus));
}
ERROR_CONDITION(entry->type_information == NULL, "Invalid type for unknown entity '%s'\n", entry->symbol_name);
if (!symbol_entity_specs_get_is_implicit_basic_type(entry))
{
DEBUG_CODE()
{
fprintf(stderr,
"BUILDSCOPE: Type of '%s' does not need updating\n",
entry->symbol_name);
}
DELETE(data);
return;
}
type_t *implicit_type
= get_implicit_type_for_symbol(decl_context, entry->symbol_name);
entry->type_information = fortran_update_basic_type_with_type(
entry->type_information, implicit_type);
if (!is_implicit_none_type(implicit_type))
{
DEBUG_CODE()
{
fprintf(stderr,
"BUILDSCOPE: Implicit type of symbol '%s' at '%s' updated to %s\n",
entry->symbol_name,
locus_to_str(entry->locus),
entry->type_information == NULL ?
"<<NULL>>" :
print_declarator(entry->type_information));
}
}
else
{
DEBUG_CODE()
{
fprintf(stderr,
"BUILDSCOPE: Type of '%s' could not be updated (possibly "
"due to implicit none)\n",
entry->symbol_name);
}
}
DELETE(data);
}
static void build_scope_implicit_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
ERROR_CONDITION(ASTKind(a) != AST_IMPLICIT_STATEMENT
&& ASTKind(a) != AST_IMPLICIT_NONE_STATEMENT,
"Invalid node",
0);
AST implicit_spec_list = ASTSon0(a);
if (implicit_spec_list == NULL)
{
if (implicit_has_been_set(decl_context))
{
if (is_implicit_none(decl_context))
{
error_printf_at(ast_get_locus(a), "IMPLICIT NONE specified twice\n");
}
else 
{
error_printf_at(ast_get_locus(a), "IMPLICIT NONE after IMPLICIT\n");
}
}
set_implicit_none(decl_context);
}
else
{
if (implicit_has_been_set(decl_context)
&& is_implicit_none(decl_context))
{
error_printf_at(ast_get_locus(a), "IMPLICIT after IMPLICIT NONE\n");
}
AST it;
for_each_element(implicit_spec_list, it)
{
AST implicit_spec = ASTSon1(it);
AST declaration_type_spec = ASTSon0(implicit_spec);
AST letter_spec_list = ASTSon1(implicit_spec);
type_t* basic_type = fortran_gather_type_from_declaration_type_spec(
declaration_type_spec,
decl_context,
NULL);
if (basic_type == NULL)
{
error_printf_at(ast_get_locus(declaration_type_spec), "invalid type specifier '%s' in IMPLICIT statement\n",
fortran_prettyprint_in_buffer(declaration_type_spec));
continue;
}
AST it2;
for_each_element(letter_spec_list, it2)
{
AST letter_spec = ASTSon1(it2);
AST letter0 = ASTSon0(letter_spec);
AST letter1 = ASTSon1(letter_spec);
const char* letter0_str = ASTText(letter0);
const char* letter1_str = NULL;
if (letter1 != NULL)
{
letter1_str = ASTText(letter1);
}
char valid = 1;
if (strlen(letter0_str) != 1
|| !(('a' <= tolower(letter0_str[0]))
&& (tolower(letter0_str[0]) <= 'z'))
|| (letter1_str != NULL 
&& (strlen(letter1_str) != 1
|| !(('a' <= tolower(letter1_str[0]))
&& (tolower(letter1_str[0]) <= 'z')))))
{
error_printf_at(ast_get_locus(letter_spec), "invalid IMPLICIT letter specifier '%s'\n",
fortran_prettyprint_in_buffer(letter_spec));
valid = 0;
}
if (valid)
{
if (letter1_str == NULL)
letter1_str = letter0_str;
set_implicit_info(decl_context, letter0_str[0], letter1_str[0], basic_type);
}
}
}
}
}
static void build_scope_import_stmt(AST a, 
const decl_context_t* decl_context, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
if (!inside_interface(a))
{
error_printf_at(ast_get_locus(a), "IMPORT statement is only valid inside an INTERFACE block\n");
return;
}
AST import_name_list = ASTSon0(a);
if (import_name_list == NULL)
{
scope_entry_t* current_procedure = decl_context->current_scope->related_entry;
decl_context->current_scope->contained_in = current_procedure->decl_context->current_scope;
}
else
{
const decl_context_t* enclosing_context = decl_context->current_scope->related_entry->decl_context;
AST it;
for_each_element(import_name_list, it)
{
AST name = ASTSon1(it);
scope_entry_t* entry = fortran_query_name_str(enclosing_context, ASTText(name),
ast_get_locus(name));
if (entry == NULL)
{
error_printf_at(ast_get_locus(name), "name '%s' in IMPORT statement not found in host associated scope\n",
ASTText(name));
continue;
}
insert_entry(decl_context->current_scope, entry);
}
}
}
static void build_scope_intent_stmt(AST a, const decl_context_t* decl_context, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST intent_spec = ASTSon0(a);
AST dummy_arg_name_list = ASTSon1(a);
AST it;
for_each_element(dummy_arg_name_list, it)
{
AST dummy_arg = ASTSon1(it);
scope_entry_t* entry = get_symbol_for_name(decl_context, dummy_arg, ASTText(dummy_arg));
if (!symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
add_delay_check_symbol_is_dummy(decl_context, entry);
}
if (symbol_entity_specs_get_intent_kind(entry) != INTENT_INVALID)
{
error_printf_at(ast_get_locus(dummy_arg), "entity '%s' already has an INTENT attribute\n",
fortran_prettyprint_in_buffer(dummy_arg));
continue;
}
attr_spec_t attr_spec;
memset(&attr_spec, 0, sizeof(attr_spec));
gather_attr_spec_item(intent_spec, decl_context, &attr_spec);
symbol_entity_specs_set_intent_kind(entry, attr_spec.intent_kind);
}
}
static scope_entry_list_t* build_scope_single_interface_specification(
AST interface_specification,
AST generic_spec,
const decl_context_t* decl_context,
int *num_related_symbols,
scope_entry_t*** related_symbols,
nodecl_t* nodecl_pragma)
{
scope_entry_list_t* result_entry_list = NULL;
if (ASTKind(interface_specification) == AST_UNKNOWN_PRAGMA)
{
}
else if (ASTKind(interface_specification) == AST_PROCEDURE)
{
unsupported_statement(interface_specification, "PROCEDURE");
}
else if (ASTKind(interface_specification) == AST_MODULE_PROCEDURE)
{
AST procedure_name_list = ASTSon0(interface_specification);
AST it2;
if (decl_context->current_scope->related_entry->kind == SK_MODULE)
{
for_each_element(procedure_name_list, it2)
{
AST procedure_name = ASTSon1(it2);
scope_entry_t* entry = NULL;
scope_entry_list_t* entry_list = query_in_scope_str_flags(
decl_context, strtolower(ASTText(procedure_name)), NULL, DF_ONLY_CURRENT_SCOPE);
char is_generic_name_of_this_module = 0;
scope_entry_list_iterator_t* it = NULL;
for (it = entry_list_iterator_begin(entry_list);
!entry_list_iterator_end(it);
entry_list_iterator_next(it))
{
scope_entry_t* current = entry_list_iterator_current(it);
if (symbol_entity_specs_get_in_module(current) == decl_context->current_scope->related_entry
&& symbol_entity_specs_get_from_module(current) == NULL)
{
if (current->kind == SK_GENERIC_NAME)
{
is_generic_name_of_this_module = 1;
}
else if (current->kind == SK_FUNCTION
|| current->kind == SK_UNDEFINED)
{
entry = current;
break;
}
}
else if (symbol_entity_specs_get_from_module(current) != NULL
&& current->kind == SK_FUNCTION
&& symbol_entity_specs_get_is_module_procedure(current))
{
entry = current;
break;
}
}
if (entry_list != NULL
&& entry == NULL)
{
if (!is_generic_name_of_this_module)
{
error_printf_at(ast_get_locus(procedure_name), "name '%s' is not a MODULE PROCEDURE\n",
prettyprint_in_buffer(procedure_name));
break;
}
else
{
entry = NULL;
}
}
entry_list_free(entry_list);
if (entry == NULL)
{
entry = create_fortran_symbol_for_name_(decl_context,
procedure_name, strtolower(ASTText(procedure_name)),  1);
add_delay_check_fully_defined_symbol(decl_context, entry);
}
entry->kind = SK_FUNCTION;
symbol_entity_specs_set_is_module_procedure(entry, 1);
result_entry_list = entry_list_add(result_entry_list, entry);
if (generic_spec != NULL)
{
P_LIST_ADD((*related_symbols),
(*num_related_symbols),
entry);
}
}
}
else
{
for_each_element(procedure_name_list, it2)
{
AST procedure_name = ASTSon1(it2);
scope_entry_list_t* entry_list = get_symbols_for_name(decl_context, procedure_name,
ASTText(procedure_name));
scope_entry_t* entry = NULL;
scope_entry_list_iterator_t* it = NULL;
for (it = entry_list_iterator_begin(entry_list);
!entry_list_iterator_end(it);
entry_list_iterator_next(it))
{
scope_entry_t* current = entry_list_iterator_current(it);
if (current->kind == SK_FUNCTION
&& symbol_entity_specs_get_is_module_procedure(current))
{
entry = current;
break;
}
}
entry_list_free(entry_list);
if (entry == NULL)
{
error_printf_at(ast_get_locus(procedure_name), "name '%s' is not a MODULE PROCEDURE\n",
prettyprint_in_buffer(procedure_name));
}
else
{
result_entry_list = entry_list_add(result_entry_list, entry);
if (generic_spec != NULL)
{
P_LIST_ADD((*related_symbols),
(*num_related_symbols),
entry);
}
}
}
}
}
else if (ASTKind(interface_specification) == AST_SUBROUTINE_PROGRAM_UNIT
|| ASTKind(interface_specification) == AST_FUNCTION_PROGRAM_UNIT)
{
scope_entry_t* entry = NULL;
nodecl_t nodecl_program_unit = nodecl_null();
build_scope_program_unit_internal(interface_specification,
decl_context,
&entry,
&nodecl_program_unit);
if (entry == NULL)
return NULL;
result_entry_list = entry_list_add(result_entry_list, entry);
if (generic_spec != NULL)
{
P_LIST_ADD((*related_symbols),
(*num_related_symbols),
entry);
}
}
else if (ASTKind(interface_specification) == AST_PRAGMA_CUSTOM_CONSTRUCT)
{
AST pragma_line = ASTSon0(interface_specification);
AST declaration = ASTSon1(interface_specification);
nodecl_t nodecl_inner_pragma = nodecl_null();
scope_entry_list_t* entry_list =
build_scope_single_interface_specification(declaration,
generic_spec,
decl_context,
num_related_symbols,
related_symbols,
&nodecl_inner_pragma);
if (entry_list != NULL)
{
if (entry_list_size(entry_list) > 1)
{
entry_list_free(entry_list);
error_printf_at(ast_get_locus(interface_specification), "a directive cannot appear before a MODULE PROCEDURE with more than one declaration\n");
return NULL;
}
scope_entry_t* entry = entry_list_head(entry_list);
nodecl_t nodecl_pragma_line = nodecl_null();
common_build_scope_pragma_custom_line(pragma_line,  NULL, decl_context, &nodecl_pragma_line);
*nodecl_pragma =
nodecl_make_pragma_custom_declaration(
nodecl_pragma_line,
nodecl_inner_pragma,
nodecl_make_pragma_context(entry->related_decl_context, ast_get_locus(interface_specification)),
nodecl_make_pragma_context(entry->related_decl_context, ast_get_locus(interface_specification)),
entry,
strtolower(ASTText(interface_specification)),
ast_get_locus(interface_specification));
}
result_entry_list = entry_list;
}
else
{
internal_error("Invalid tree '%s'\n", ast_print_node_type(ASTKind(interface_specification)));
}
return result_entry_list;
}
static void build_scope_interface_block(AST a,
const decl_context_t* decl_context,
nodecl_t* nodecl_output)
{
AST interface_stmt = ASTSon0(a);
AST interface_specification_seq = ASTSon1(a);
AST abstract = ASTSon0(interface_stmt);
char is_abstract_interface = (abstract != NULL);
AST generic_spec = ASTSon1(interface_stmt);
scope_entry_t** related_symbols = NULL;
int num_related_symbols = 0;
scope_entry_t* generic_spec_sym = NULL;
if (generic_spec != NULL)
{
const char* name = get_name_of_generic_spec(generic_spec);
scope_entry_list_t* entry_list = query_in_scope_str_flags(decl_context, name, NULL, DF_ONLY_CURRENT_SCOPE);
scope_entry_t* previous_generic_spec_sym = NULL;
if (entry_list != NULL)
{
scope_entry_list_iterator_t* it = NULL;
for (it = entry_list_iterator_begin(entry_list);
!entry_list_iterator_end(it);
entry_list_iterator_next(it))
{
scope_entry_t* current_sym = entry_list_iterator_current(it);
if (current_sym->kind == SK_UNDEFINED)
{
ERROR_CONDITION(generic_spec_sym != NULL, "Repeated undefined name '%s'\n", generic_spec_sym->symbol_name);
previous_generic_spec_sym = NULL;
generic_spec_sym = current_sym;
}
else if (current_sym->kind == SK_FUNCTION
|| current_sym->kind == SK_GENERIC_NAME)
{
if (symbol_entity_specs_get_from_module(current_sym) != NULL)
{
previous_generic_spec_sym = current_sym;
}
else if (current_sym->kind == SK_GENERIC_NAME)
{
previous_generic_spec_sym = NULL;
generic_spec_sym = current_sym;
}
}
else 
{
error_printf_at(ast_get_locus(generic_spec), "redefining symbol '%s'\n",
name);
entry_list_iterator_free(it);
return;
}
}
entry_list_iterator_free(it);
}
if (generic_spec_sym == NULL)
{
generic_spec_sym = create_fortran_symbol_for_name_(decl_context, generic_spec, name,  1);
generic_spec_sym->type_information = get_void_type();
generic_spec_sym->locus = ast_get_locus(generic_spec);
}
generic_spec_sym->kind = SK_GENERIC_NAME;
symbol_entity_specs_set_is_implicit_basic_type(generic_spec_sym, 0);
if (previous_generic_spec_sym != NULL
&& symbol_entity_specs_get_access(previous_generic_spec_sym) != AS_UNKNOWN)
{
symbol_entity_specs_set_access(generic_spec_sym, symbol_entity_specs_get_access(previous_generic_spec_sym));
}
}
if (interface_specification_seq != NULL)
{
AST it;
for_each_element(interface_specification_seq, it)
{
AST interface_specification = ASTSon1(it);
nodecl_t nodecl_pragma = nodecl_null();
scope_entry_list_t* entry_list =
build_scope_single_interface_specification(
interface_specification,
generic_spec,
decl_context,
&num_related_symbols,
&related_symbols,
&nodecl_pragma);
if (is_abstract_interface)
{
scope_entry_list_iterator_t *entry_it = NULL;
for (entry_it = entry_list_iterator_begin(entry_list);
!entry_list_iterator_end(entry_it);
entry_list_iterator_next(entry_it))
{
scope_entry_t* sym = entry_list_iterator_current(entry_it);
symbol_entity_specs_set_is_abstract(sym, 1);
}
entry_list_iterator_free(entry_it);
}
entry_list_free(entry_list);
if (!nodecl_is_null(nodecl_pragma))
{
*nodecl_output = nodecl_append_to_list(*nodecl_output,
nodecl_pragma);
}
}
}
if (generic_spec_sym != NULL)
{
int i;
for (i = 0; i < num_related_symbols; i++)
{
symbol_entity_specs_add_related_symbols(generic_spec_sym, related_symbols[i]);
}
}
}
static void build_scope_intrinsic_stmt(AST a, 
const decl_context_t* decl_context, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST intrinsic_list = ASTSon0(a);
scope_entry_t* current_program_unit = decl_context->current_scope->related_entry;
AST it;
for_each_element(intrinsic_list, it)
{
AST name = ASTSon1(it);
scope_entry_t* entry = NULL;
scope_entry_list_t* entry_list = 
query_in_scope_str_flags(current_program_unit->related_decl_context, strtolower(ASTText(name)), NULL, DF_ONLY_CURRENT_SCOPE);
if (entry_list != NULL)
{
entry = entry_list_head(entry_list);
entry_list_free(entry_list);
}
scope_entry_t* entry_intrinsic = fortran_query_intrinsic_name_str(decl_context, ASTText(name));
if (entry != NULL)
{
if (entry->kind == SK_FUNCTION) 
{
if (symbol_entity_specs_get_is_builtin(entry))
{
error_printf_at(ast_get_locus(name), "entity '%s' already has INTRINSIC attribute\n",
entry->symbol_name);
continue;
}
else
{
error_printf_at(ast_get_locus(name), "entity '%s' already has EXTERNAL attribute and EXTERNAL attribute conflicts with INTRINSIC attribute\n",
entry->symbol_name);
continue;
}
}
else
{
if (entry_intrinsic == NULL || !symbol_entity_specs_get_is_builtin(entry_intrinsic))
{
error_printf_at(ast_get_locus(name), "name '%s' is not known as an intrinsic\n",
ASTText(name));
continue;
}
entry->kind = SK_FUNCTION;
}
}
else
{
if (entry_intrinsic == NULL 
|| !symbol_entity_specs_get_is_builtin(entry_intrinsic))
{
error_printf_at(ast_get_locus(name), "name '%s' is not known as an intrinsic\n",
ASTText(name));
continue;
}
entry = get_symbol_for_name(decl_context, name, ASTText(name));
}
copy_intrinsic_function_info(entry, entry_intrinsic);
}
}
static void build_scope_lock_stmt(AST a UNUSED_PARAMETER, const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
unsupported_statement(a, "LOCK");
}
static void build_scope_namelist_stmt(AST a, const decl_context_t* decl_context, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST namelist_item_list = ASTSon0(a);
AST it;
for_each_element(namelist_item_list, it)
{
AST namelist_item = ASTSon1(it);
AST common_name = ASTSon0(namelist_item);
AST namelist_group_object_list = ASTSon1(namelist_item);
AST name = ASTSon0(common_name);
scope_entry_t* new_namelist
= fortran_query_name_str(decl_context, ASTText(name), ast_get_locus(name));
if (new_namelist != NULL
&& new_namelist->kind != SK_UNDEFINED
&& new_namelist->kind != SK_NAMELIST)
{
error_printf_at(ast_get_locus(name), "name '%s' cannot be used as a namelist\n",
ASTText(name));
new_namelist = NULL;
}
if (new_namelist == NULL)
{
new_namelist = new_fortran_symbol(decl_context, ASTText(name));
if (decl_context->current_scope->related_entry != NULL
&& decl_context->current_scope->related_entry->kind == SK_MODULE)
{
scope_entry_t* module = decl_context->current_scope->related_entry;
symbol_entity_specs_add_related_symbols(module, new_namelist);
symbol_entity_specs_set_in_module(new_namelist, module);
}
}
new_namelist->kind = SK_NAMELIST;
new_namelist->locus = ast_get_locus(a);
AST it2;
for_each_element(namelist_group_object_list, it2)
{
AST namelist_item_name = ASTSon1(it2);
scope_entry_t* namelist_element =
fortran_get_variable_with_locus(decl_context, namelist_item_name, ASTText(namelist_item_name));
if (namelist_element == NULL)
{
namelist_element = get_symbol_for_name(decl_context, namelist_item_name, ASTText(namelist_item_name));
}
symbol_entity_specs_set_is_in_namelist(namelist_element, 1);
symbol_entity_specs_set_namelist(namelist_element, new_namelist);
symbol_entity_specs_add_related_symbols(new_namelist,
namelist_element);
}
}
}
static void build_scope_nullify_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST pointer_object_list = ASTSon0(a);
char error = 0;
nodecl_t nodecl_expr_list = nodecl_null();
AST it;
for_each_element(pointer_object_list, it)
{
AST pointer_object = ASTSon1(it);
nodecl_t nodecl_pointer_obj = nodecl_null();
fortran_check_expression(pointer_object, decl_context, &nodecl_pointer_obj);
scope_entry_t* sym = fortran_data_ref_get_symbol(nodecl_pointer_obj);
if (sym == NULL ||
!is_pointer_type(no_ref(sym->type_information)))
{
error_printf_at(ast_get_locus(a), "'%s' does not designate a POINTER\n",
fortran_prettyprint_in_buffer(pointer_object));
error = 1;
continue;
}
nodecl_expr_list = nodecl_append_to_list(nodecl_expr_list, 
nodecl_pointer_obj);
}
if (error)
{
*nodecl_output = nodecl_make_list_1(
nodecl_make_err_statement(ast_get_locus(a))
);
return;
}
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_nullify_statement(nodecl_expr_list, ast_get_locus(a)));
}
static void build_scope_open_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST connect_spec_list = ASTSon0(a);
nodecl_t nodecl_opt_value = nodecl_null();
handle_opt_value_list(a, connect_spec_list, decl_context, &nodecl_opt_value);
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_open_statement(nodecl_opt_value, ast_get_locus(a)));
}
static void build_scope_optional_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST name_list = ASTSon0(a);
AST it;
for_each_element(name_list, it)
{
AST name = ASTSon1(it);
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
if (!symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
error_printf_at(ast_get_locus(name), "entity '%s' is not a dummy argument\n",
ASTText(name));
continue;
}
symbol_entity_specs_set_is_optional(entry, 1);
}
}
static void build_scope_parameter_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST named_constant_def_list = ASTSon0(a);
AST it;
for_each_element(named_constant_def_list, it)
{
AST named_constant_def = ASTSon1(it);
AST name = ASTSon0(named_constant_def);
AST constant_expr = ASTSon1(named_constant_def);
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
if (is_void_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(name), "unknown entity '%s' in PARAMETER statement\n",
ASTText(name));
continue;
}
if (symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
error_printf_at(ast_get_locus(a), "PARAMETER attribute is not valid for dummy arguments\n");
continue;
}
if (is_const_qualified_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(a), "PARAMETER attribute already specified\n");
continue;
}
if (is_pointer_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(a), "PARAMETER attribute is not compatible with POINTER attribute\n");
continue;
}
if (entry->kind == SK_UNDEFINED)
{
entry->kind = SK_VARIABLE;
}
if (fortran_is_array_type(entry->type_information))
{
build_scope_delay_list_run_now(
DELAY_AFTER_DECLARATIONS,
entry,
delayed_compute_type_from_array_spec,
delayed_array_specifier_cmp,
NULL);
}
fortran_immediate_check_initialization(
entry,
constant_expr,
decl_context,
0,
1,
1,
0);
}
}
static void build_scope_cray_pointer_stmt(AST a, const decl_context_t* decl_context,
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST cray_pointer_spec_list = ASTSon0(a);
AST it;
for_each_element(cray_pointer_spec_list, it)
{
AST cray_pointer_spec = ASTSon1(it);
AST pointer_name = ASTSon0(cray_pointer_spec);
AST pointee_decl = ASTSon1(cray_pointer_spec);
scope_entry_t* pointer_entry = get_symbol_for_name(decl_context, pointer_name, ASTText(pointer_name));
if (pointer_entry->kind == SK_UNDEFINED)
{
pointer_entry->kind = SK_VARIABLE;
nodecl_t nodecl_sym = nodecl_make_symbol(pointer_entry, ast_get_locus(a));
char needs_reference = symbol_is_parameter_of_function(pointer_entry, decl_context->current_scope->related_entry);
pointer_entry->type_information = choose_int_type_from_kind(nodecl_sym, 
CURRENT_CONFIGURATION->type_environment->sizeof_pointer);
if (needs_reference)
{
pointer_entry->type_information = get_lvalue_reference_type(pointer_entry->type_information);
}
}
else if (pointer_entry->kind == SK_VARIABLE)
{
if (!is_integer_type(pointer_entry->type_information))
{
error_printf_at(ast_get_locus(pointer_name), "a Cray pointer must have integer type\n");
continue;
}
}
else
{
error_printf_at(ast_get_locus(pointer_name), "invalid entity '%s' for Cray pointer\n",
ASTText(pointer_name));
continue;
}
symbol_entity_specs_set_is_cray_pointer(pointer_entry, 1);
AST pointee_name = pointee_decl;
AST array_spec = NULL;
if (ASTKind(pointee_decl) == AST_DIMENSION_DECL)
{
pointee_name = ASTSon0(pointee_decl);
array_spec = ASTSon1(pointee_decl);
}
scope_entry_t* pointee_entry = get_symbol_for_name(decl_context, pointer_name, ASTText(pointee_name));
if (symbol_entity_specs_get_is_cray_pointee(pointee_entry))
{
error_printf_at(ast_get_locus(pointee_name), "entity '%s' is already a pointee of Cray pointer '%s'\n",
pointee_entry->symbol_name,
symbol_entity_specs_get_cray_pointer(pointee_entry)->symbol_name);
continue;
}
if (array_spec != NULL)
{
if (fortran_is_array_type(no_ref(pointee_entry->type_information))
|| fortran_is_pointer_to_array_type(no_ref(pointee_entry->type_information)))
{
error_printf_at(ast_get_locus(pointee_name), "entity '%s' has already a DIMENSION attribute\n",
pointee_entry->symbol_name);
continue;
}
compute_type_from_array_spec(
pointee_entry,
no_ref(pointee_entry->type_information), 
array_spec,
decl_context,
1);
pointee_entry->kind = SK_VARIABLE;
}
symbol_entity_specs_set_is_cray_pointee(pointee_entry, 1);
symbol_entity_specs_set_cray_pointer(pointee_entry, pointer_entry);
}
}
static void build_scope_pointer_stmt(AST a, const decl_context_t* decl_context,
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST pointer_decl_list = ASTSon0(a);
AST it;
for_each_element(pointer_decl_list, it)
{
AST pointer_decl = ASTSon1(it);
AST name = pointer_decl;
AST array_spec = NULL;
if (ASTKind(pointer_decl) == AST_DIMENSION_DECL)
{
name = ASTSon0(pointer_decl);
array_spec = ASTSon1(pointer_decl);
}
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
char was_ref = is_lvalue_reference_type(entry->type_information);
if (is_pointer_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(pointer_decl), "entity '%s' has already the POINTER attribute\n",
entry->symbol_name);
continue;
}
if (symbol_entity_specs_get_is_allocatable(entry))
{
error_printf_at(ast_get_locus(name), "attribute ALLOCATABLE conflicts with POINTER\n");
continue;
}
if (is_const_qualified_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(pointer_decl), "POINTER attribute is not compatible with PARAMETER attribute\n");
continue;
}
if (array_spec != NULL)
{
if (fortran_is_array_type(no_ref(entry->type_information))
|| fortran_is_pointer_to_array_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(pointer_decl), "entity '%s' has already a DIMENSION attribute\n",
entry->symbol_name);
continue;
}
compute_type_from_array_spec(
entry,
no_ref(entry->type_information), 
array_spec,
decl_context,
1);
}
if (entry->kind == SK_UNDEFINED)
{
entry->kind = SK_VARIABLE;
}
if (entry->kind == SK_FUNCTION)
{
symbol_entity_specs_set_is_extern(entry, 0);
entry->kind = SK_VARIABLE;
}
if (!is_error_type(entry->type_information))
{
entry->type_information = get_pointer_type(no_ref(entry->type_information));
if (fortran_is_pointer_to_array_type(entry->type_information))
{
check_array_type_is_valid_for_pointer(
entry->type_information,
entry,
ast_get_locus(pointer_decl));
}
if (was_ref)
{
entry->type_information = get_lvalue_reference_type(entry->type_information);
}
}
}
}
static void build_scope_input_output_item(AST input_output_item, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
if (ASTKind(input_output_item) == AST_IMPLIED_DO)
{
generic_implied_do_handler(input_output_item, decl_context,
build_scope_input_output_item_list, nodecl_output);
}
else 
{
fortran_check_expression(input_output_item, decl_context, nodecl_output);
}
}
static void build_scope_input_output_item_list(AST input_output_item_list, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST it;
for_each_element(input_output_item_list, it)
{
nodecl_t nodecl_item = nodecl_null();
build_scope_input_output_item(ASTSon1(it), decl_context, &nodecl_item);
*nodecl_output = nodecl_append_to_list(*nodecl_output, nodecl_item);
}
}
static void opt_fmt_value(AST value, const decl_context_t* decl_context, nodecl_t* nodecl_output);
static void build_scope_print_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST format = ASTSon0(a);
AST input_output_item_list = ASTSon1(a);
nodecl_t nodecl_io_items = nodecl_null();
if (input_output_item_list != NULL)
{
build_scope_input_output_item_list(input_output_item_list, decl_context, &nodecl_io_items);
}
nodecl_t nodecl_format = nodecl_null();
opt_fmt_value(format, decl_context, &nodecl_format);
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_print_statement(nodecl_get_child(nodecl_format, 0), nodecl_io_items, ast_get_locus(a)));
}
typedef struct delayed_procedure_decl_stmt_data_tag
{
AST a;
const decl_context_t* decl_context;
} delayed_procedure_decl_stmt_data_t;
static void delayed_build_scope_procedure_decl_stmt_(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST proc_interface = ASTSon0(a);
AST proc_attr_spec_list = ASTSon1(a);
AST proc_decl_list = ASTSon2(a);
type_t* return_type = NULL;
scope_entry_t* interface = NULL;
if (proc_interface != NULL)
{
if (ASTKind(proc_interface) == AST_SYMBOL)
{
interface = fortran_query_name_str(decl_context,
strtolower(ASTText(proc_interface)),
ast_get_locus(proc_interface));
if (interface != NULL
&& (interface->kind == SK_FUNCTION
|| (interface->kind == SK_VARIABLE
&& (is_function_type(no_ref(interface->type_information))
|| is_pointer_to_function_type(no_ref(interface->type_information))))))
{
type_t* function_type = no_ref(interface->type_information);
if (is_pointer_type(function_type))
{
function_type = pointer_type_get_pointee_type(function_type);
}
if (function_type_get_lacking_prototype(function_type))
{
error_printf_at(ast_get_locus(proc_interface), "'%s' does not have an explicit interface\n",
interface->symbol_name);
interface = NULL;
}
}
else
{
error_printf_at(ast_get_locus(proc_interface), "'%s' is not a valid procedure interface\n",
interface->symbol_name);
interface = NULL;
}
}
else
{
return_type = fortran_gather_type_from_declaration_type_spec(proc_interface,
decl_context,
NULL);
}
}
attr_spec_t attr_spec;
memset(&attr_spec, 0, sizeof(attr_spec));
if (proc_attr_spec_list != NULL)
gather_attr_spec_list(proc_attr_spec_list, decl_context, &attr_spec);
AST it;
for_each_element(proc_decl_list, it)
{
AST name = ASTSon1(it);
AST init = NULL;
if (ASTKind(name) == AST_PROCEDURE_DECL)
{
init = ASTSon1(name);
name = ASTSon0(name);
}
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
symbol_entity_specs_set_is_procedure_decl_stmt(entry, 1);
if (symbol_entity_specs_get_is_builtin(entry))
{
error_printf_at(ast_get_locus(name), "entity '%s' already has INTRINSIC attribute and INTRINSIC attribute conflicts with EXTERNAL attribute\n",
entry->symbol_name);
continue;
}
if (attr_spec.is_save)
{
if (symbol_entity_specs_get_is_static(entry))
{
error_printf_at(ast_get_locus(name),
"SAVE attribute already specified for symbol '%s'\n",
entry->symbol_name);
}
symbol_entity_specs_set_is_static(entry, 1);
}
if (attr_spec.is_optional)
{
if (!symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
error_printf_at(ast_get_locus(name), "OPTIONAL attribute is only for dummy arguments\n");
}
if (symbol_entity_specs_get_is_optional(entry))
{
error_printf_at(ast_get_locus(name), "OPTIONAL attribute already specified for symbol '%s'\n",
entry->symbol_name);
}
symbol_entity_specs_set_is_optional(entry, 1);
}
if (!attr_spec.is_pointer
&& !is_pointer_type(no_ref(entry->type_information)))
{
if (entry->kind == SK_UNDEFINED)
{
if (!symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
entry->kind = SK_FUNCTION;
}
else
{
entry->kind = SK_VARIABLE;
}
synthesize_procedure_type(entry, interface, return_type,
decl_context,  0,  0);
}
else if (entry->kind == SK_FUNCTION)
{
error_printf_at(ast_get_locus(name), "entity '%s' already has EXTERNAL attribute\n",
entry->symbol_name);
}
else if (entry->kind == SK_VARIABLE
&& symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry)
&& is_function_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(name), "entity '%s' already has EXTERNAL attribute\n",
entry->symbol_name);
}
else
{
error_printf_at(ast_get_locus(name), "entity '%s' cannot appear in a PROCEDURE statement\n",
entry->symbol_name);
}
}
else
{
if (attr_spec.is_pointer
&& is_pointer_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(name),
"POINTER attribute already specified for symbol '%s'\n",
entry->symbol_name);
}
else
{
entry->kind = SK_VARIABLE;
synthesize_procedure_type(entry, interface, return_type,
decl_context,  1,  0);
}
}
if (init != NULL)
{
if (!is_pointer_type(entry->type_information))
{
error_printf_at(ast_get_locus(name),
"only procedure pointers can be initialized in a procedure declaration statement\n");
}
fortran_immediate_check_initialization(
entry,
init,
decl_context,
1,
0,
0,
1);
}
}
}
static void delayed_build_scope_procedure_decl_stmt(void* info, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
delayed_procedure_decl_stmt_data_t* data = (delayed_procedure_decl_stmt_data_t*) info;
delayed_build_scope_procedure_decl_stmt_(data->a, data->decl_context, nodecl_output);
}
static void build_scope_procedure_decl_stmt(AST a, const decl_context_t* decl_context,
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST proc_decl_list = ASTSon2(a);
AST it;
for_each_element(proc_decl_list, it)
{
AST name = ASTSon1(it);
if (ASTKind(name) == AST_PROCEDURE_DECL)
{
name = ASTSon0(name);
}
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
symbol_entity_specs_set_is_implicit_basic_type(entry, 0);
}
delayed_procedure_decl_stmt_data_t *data = NEW0(delayed_procedure_decl_stmt_data_t);
data->a = a;
data->decl_context = decl_context;
build_scope_delay_list_add(
DELAY_AFTER_DECLARATIONS, delayed_build_scope_procedure_decl_stmt, data);
}
static void build_scope_protected_stmt(AST a, const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
unsupported_statement(a, "PROTECTED");
}
static void build_scope_read_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST io_control_spec_list = ASTSon0(a);
nodecl_t nodecl_opt_value = nodecl_null();
handle_opt_value_list(a, io_control_spec_list, decl_context, &nodecl_opt_value);
nodecl_t nodecl_io_items = nodecl_null();
if (ASTSon1(a) != NULL)
{
build_scope_input_output_item_list(ASTSon1(a), decl_context, &nodecl_io_items);
}
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_read_statement(nodecl_opt_value, nodecl_io_items, ast_get_locus(a)));
}
static void build_scope_return_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
if (decl_context->current_scope->related_entry == NULL
|| decl_context->current_scope->related_entry->kind != SK_FUNCTION)
{
error_printf_at(ast_get_locus(a), "RETURN statement not valid in this context\n");
*nodecl_output = nodecl_make_list_1(
nodecl_make_err_statement(ast_get_locus(a))
);
return;
}
scope_entry_t* current_function = decl_context->current_scope->related_entry;
AST int_expr = ASTSon1(a);
if (int_expr != NULL)
{
nodecl_t nodecl_return = nodecl_null();
fortran_check_expression(ASTSon1(a), decl_context, &nodecl_return);
nodecl_return = fortran_expression_as_value(nodecl_return);
if (nodecl_is_err_expr(nodecl_return))
{
*nodecl_output = nodecl_return;
}
if (!is_void_type(function_type_get_return_type(current_function->type_information)))
{
error_printf_at(ast_get_locus(a), "RETURN with alternate return is only valid in a SUBROUTINE program unit\n");
*nodecl_output = nodecl_make_list_1(
nodecl_make_err_statement(ast_get_locus(a))
);
return;
}
*nodecl_output = nodecl_make_fortran_alternate_return_statement(nodecl_return, ast_get_locus(a));
}
else
{
if (is_void_type(function_type_get_return_type(current_function->type_information)))
{
*nodecl_output = nodecl_make_return_statement(nodecl_null(), ast_get_locus(a));
}
else
{
*nodecl_output = nodecl_make_return_statement(
nodecl_make_symbol(function_get_result_symbol(current_function), ast_get_locus(a)), 
ast_get_locus(a));
}
}
*nodecl_output = nodecl_make_list_1(*nodecl_output);
}
static void build_scope_save_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST saved_entity_list = ASTSon0(a);
AST it;
scope_entry_t* program_unit = decl_context->current_scope->related_entry;
if (symbol_entity_specs_get_is_saved_program_unit(program_unit))
{
warn_printf_at(ast_get_locus(a),
"SAVE statement '%s' specified after blanket SAVE statement\n",
fortran_prettyprint_in_buffer(a));
}
if (saved_entity_list == NULL)
{
symbol_entity_specs_set_is_saved_program_unit(program_unit, 1);
return;
}
for_each_element(saved_entity_list, it)
{
AST saved_entity = ASTSon1(it);
scope_entry_t* entry = NULL;
if (ASTKind(saved_entity) == AST_COMMON_NAME)
{
entry = query_common_name(decl_context, ASTText(ASTSon0(saved_entity)),
ast_get_locus(ASTSon0(saved_entity)));
if (entry == NULL)
{
entry = new_common(decl_context,ASTText(ASTSon0(saved_entity)));
entry->locus = ast_get_locus(a);
add_delay_check_fully_defined_symbol(decl_context, entry);
}
}
else
{
entry = get_symbol_for_name(decl_context, saved_entity, ASTText(saved_entity));
}
if (entry->kind == SK_UNDEFINED)
{
entry->kind = SK_VARIABLE;
}
symbol_entity_specs_set_is_static(entry, 1);
}
}
static void build_scope_select_type_construct(AST a, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
unsupported_statement(a, "SELECT TYPE");
}
static void build_scope_stmt_function_stmt(AST a, const decl_context_t* decl_context, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST name = ASTSon0(a);
AST dummy_arg_name_list = ASTSon1(a);
AST expr = ASTSon2(a);
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
entry->kind = SK_FUNCTION;
symbol_entity_specs_set_is_stmt_function(entry, 1);
int num_dummy_arguments = 0;
if (dummy_arg_name_list != NULL)
{
AST it;
for_each_element(dummy_arg_name_list, it)
{
AST dummy_arg_item = ASTSon1(it);
scope_entry_t* dummy_arg = get_symbol_for_name(decl_context, dummy_arg_item, ASTText(dummy_arg_item));
if (!fortran_is_scalar_type(no_ref(dummy_arg->type_information)))
{
error_printf_at(ast_get_locus(dummy_arg_item), "dummy argument '%s' of statement function statement is not a scalar\n",
fortran_prettyprint_in_buffer(dummy_arg_item));
return;
}
if (dummy_arg->kind == SK_UNDEFINED)
{
dummy_arg->kind = SK_VARIABLE;
}
symbol_set_as_parameter_of_function(dummy_arg, entry,
0,
symbol_entity_specs_get_num_related_symbols(entry));
symbol_entity_specs_add_related_symbols(entry, dummy_arg);
num_dummy_arguments++;
}
}
scope_entry_t* result_sym = NEW0(scope_entry_t);
result_sym->symbol_name = entry->symbol_name;
result_sym->kind = SK_VARIABLE;
result_sym->decl_context = decl_context;
result_sym->type_information = entry->type_information;
symbol_entity_specs_set_is_result_var(result_sym, 1);
symbol_entity_specs_set_result_var(entry, result_sym);
parameter_info_t parameter_info[1 + num_dummy_arguments];
memset(parameter_info, 0, sizeof(parameter_info));
int i;
for (i = 0; i < num_dummy_arguments; i++)
{
parameter_info[i].type_info = get_mutable_indirect_type(symbol_entity_specs_get_related_symbols_num(entry, i));
}
type_t* new_type = get_new_function_type(entry->type_information, 
parameter_info, num_dummy_arguments, REF_QUALIFIER_NONE);
entry->type_information = new_type;
fortran_check_expression(expr, decl_context, &entry->value);
}
static void build_scope_stop_stmt(AST a, const decl_context_t* decl_context, 
nodecl_t* nodecl_output)
{
nodecl_t nodecl_stop_code = nodecl_null();
AST stop_code = ASTSon0(a);
if (stop_code != NULL)
{
fortran_check_expression(stop_code, decl_context, &nodecl_stop_code);
nodecl_stop_code = fortran_expression_as_value(nodecl_stop_code);
}
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_stop_statement(nodecl_stop_code, ast_get_locus(a))
);
}
static void build_scope_pause_stmt(AST a UNUSED_PARAMETER, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
nodecl_t nodecl_pause_code = nodecl_null();
AST pause_code = ASTSon0(a);
if (pause_code != NULL)
{
fortran_check_expression(pause_code, decl_context, &nodecl_pause_code);
nodecl_pause_code = fortran_expression_as_value(nodecl_pause_code);
}
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_pause_statement(nodecl_pause_code, ast_get_locus(a)));
}
static void build_scope_sync_all_stmt(AST a UNUSED_PARAMETER, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
unsupported_statement(a, "SYNC ALL");
}
static void build_scope_sync_images_stmt(AST a UNUSED_PARAMETER, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
unsupported_statement(a, "SYNC IMAGES");
}
static void build_scope_sync_memory_stmt(AST a UNUSED_PARAMETER, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
unsupported_statement(a, "SYNC MEMORY");
}
static void build_scope_target_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST target_decl_list = ASTSon0(a);
AST it;
for_each_element(target_decl_list, it)
{
AST target_decl = ASTSon1(it);
AST name = NULL;
AST array_spec = NULL;
AST coarray_spec = NULL;
if (ASTKind(target_decl) == AST_SYMBOL)
{
name = target_decl;
}
else if (ASTKind(target_decl) == AST_DIMENSION_DECL)
{
name = ASTSon0(target_decl);
array_spec = ASTSon1(target_decl);
coarray_spec = ASTSon2(target_decl);
}
else
{
internal_error("Unexpected node '%s'\n", ast_print_node_type(ASTKind(a)));
}
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
if (coarray_spec != NULL)
{
sorry_printf_at(ast_get_locus(name),
"coarrays are not supported\n");
}
if (array_spec != NULL)
{
if (fortran_is_array_type(no_ref(entry->type_information))
|| fortran_is_pointer_to_array_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(a), "DIMENSION attribute specified twice for entity '%s'\n",
entry->symbol_name);
continue;
}
char was_ref = is_lvalue_reference_type(entry->type_information);
compute_type_from_array_spec(
entry,
no_ref(entry->type_information),
array_spec,
decl_context,
1);
if (!is_error_type(entry->type_information))
{
if (was_ref)
{
entry->type_information = get_lvalue_reference_type(entry->type_information);
}
}
}
if (symbol_entity_specs_get_is_target(entry))
{
error_printf_at(ast_get_locus(target_decl), "entity '%s' already has TARGET attribute\n",
entry->symbol_name);
continue;
}
if (entry->kind == SK_UNDEFINED)
{
entry->kind = SK_VARIABLE;
}
symbol_entity_specs_set_is_target(entry, 1);
}
}
static void build_scope_declaration_common_stmt(AST a, const decl_context_t* decl_context, 
char is_typedef,
nodecl_t* nodecl_output UNUSED_PARAMETER)
{
DEBUG_CODE()
{
fprintf(stderr, "== [%s] Declaration statement ==\n", ast_location(a));
}
AST declaration_type_spec = ASTSon0(a);
AST attr_spec_list = ASTSon1(a);
AST entity_decl_list = ASTSon2(a);
AST character_length_out = NULL;
type_t* basic_type =
fortran_gather_type_from_declaration_type_spec(declaration_type_spec,
decl_context,
&character_length_out);
attr_spec_t attr_spec;
memset(&attr_spec, 0, sizeof(attr_spec));
if (attr_spec_list != NULL)
{
gather_attr_spec_list(attr_spec_list, decl_context, &attr_spec);
}
scope_entry_t** delayed_character_symbols = NULL;
int num_delayed_character_symbols = 0;
AST it;
for_each_element(entity_decl_list, it)
{
attr_spec_t current_attr_spec = attr_spec;
AST declaration = ASTSon1(it);
AST name = ASTSon0(declaration);
AST entity_decl_specs = ASTSon1(declaration);
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
scope_entry_t* entry_intrinsic = fortran_query_intrinsic_name_str(decl_context, ASTText(name));
char could_be_an_intrinsic =
(entry_intrinsic != NULL
&& !symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry)
&& entry != decl_context->current_scope->related_entry
&& entry->kind == SK_UNDEFINED);
if (could_be_an_intrinsic)
{
entry->type_information = basic_type;
}
if (!symbol_entity_specs_get_is_implicit_basic_type(entry))
{
error_printf_at(ast_get_locus(name), "entity '%s' already has a basic type\n",
entry->symbol_name);
continue;
}
if (entry->defined)
{
error_printf_at(ast_get_locus(declaration), "redeclaration of entity '%s', first declared at '%s'\n",
entry->symbol_name,
locus_to_str(entry->locus));
continue;
}
if (is_typedef)
{
if (entry->kind != SK_UNDEFINED)
{
fatal_printf_at(ast_get_locus(declaration), "TYPEDEF would overwrite a non undefined entity\n");
}
entry->kind = SK_TYPEDEF;
}
entry->type_information = fortran_update_basic_type_with_type(entry->type_information, basic_type);
symbol_entity_specs_set_is_implicit_basic_type(entry, 0);
if (fortran_is_character_type(basic_type)
&& character_length_out != NULL)
{
P_LIST_ADD(delayed_character_symbols,
num_delayed_character_symbols,
entry);
}
entry->locus = ast_get_locus(declaration);
if (entry->kind == SK_FUNCTION)
{
scope_entry_t* sym = symbol_entity_specs_get_result_var(entry);
if (sym != NULL)
{
sym->type_information = fortran_update_basic_type_with_type(sym->type_information, basic_type);
symbol_entity_specs_set_is_implicit_basic_type(sym, 0);
}
}
AST array_spec = NULL;
AST coarray_spec = NULL;
AST char_length = NULL;
AST initialization = NULL;
if (entity_decl_specs != NULL)
{
array_spec = ASTSon0(entity_decl_specs);
coarray_spec = ASTSon1(entity_decl_specs);
char_length = ASTSon2(entity_decl_specs);
initialization = ASTSon3(entity_decl_specs);
}
if (array_spec != NULL)
{
current_attr_spec.is_dimension = 1;
current_attr_spec.array_spec = array_spec;
}
if (coarray_spec != NULL)
{
if (current_attr_spec.is_codimension)
{
error_printf_at(ast_get_locus(declaration), "CODIMENSION attribute specified twice\n");
continue;
}
current_attr_spec.is_codimension = 1;
current_attr_spec.coarray_spec = coarray_spec;
}
if (char_length != NULL)
{
type_t* new_basic_type = NULL;
type_t* rank0 = fortran_get_rank0_type(basic_type);
if (!fortran_is_character_type(rank0))
{
error_printf_at(ast_get_locus(declaration), "char-length specified but type is not CHARACTER\n");
continue;
}
rank0 = array_type_get_element_type(rank0);
if (ASTKind(char_length) != AST_SYMBOL
|| strcmp(ASTText(char_length), "*") != 0)
{
nodecl_t nodecl_char_length = nodecl_null();
fortran_check_expression(char_length, decl_context, &nodecl_char_length);
if (nodecl_is_err_expr(nodecl_char_length))
continue;
nodecl_char_length = fortran_expression_as_value(nodecl_char_length);
nodecl_t lower_bound = nodecl_make_integer_literal(
get_signed_int_type(),
const_value_get_one(type_get_size(get_signed_int_type()), 1),
ast_get_locus(char_length));
new_basic_type = get_array_type_bounds(
rank0,
lower_bound, nodecl_char_length, decl_context);
}
else
{
new_basic_type = get_array_type(
rank0,
nodecl_null(), decl_context);
}
entry->type_information = fortran_update_basic_type_with_type(entry->type_information, new_basic_type);
}
if (current_attr_spec.is_codimension)
{
error_printf_at(ast_get_locus(declaration), "sorry: coarrays are not supported\n");
}
if (current_attr_spec.is_asynchronous)
{
error_printf_at(ast_get_locus(declaration), "sorry: ASYNCHRONOUS attribute not supported\n");
}
if (current_attr_spec.is_dimension
&& !is_error_type(no_ref(entry->type_information)))
{
char was_ref = is_lvalue_reference_type(entry->type_information);
cv_qualifier_t cv_qualif = get_cv_qualifier(entry->type_information);
char is_parameter = is_const_qualified_type(no_ref(entry->type_information))
|| current_attr_spec.is_constant;
compute_type_from_array_spec(
entry,
get_unqualified_type(no_ref(entry->type_information)),
current_attr_spec.array_spec,
decl_context,
!is_parameter);
if (!is_typedef)
{
entry->kind = SK_VARIABLE;
}
if (!is_error_type(entry->type_information))
{
entry->type_information = get_cv_qualified_type(entry->type_information, cv_qualif);
if (was_ref)
{
entry->type_information = get_lvalue_reference_type(entry->type_information);
}
}
}
if (current_attr_spec.is_value)
{
if (!symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
error_printf_at(ast_get_locus(declaration), "VALUE attribute is only for dummy arguments\n");
continue;
}
else
{
char was_ref = is_lvalue_reference_type(entry->type_information);
if (!was_ref)
{
error_printf_at(ast_get_locus(declaration), "VALUE attribute already set\n");
}
else
{
entry->type_information = reference_type_get_referenced_type(entry->type_information);
}
}
}
if (current_attr_spec.is_intent)
{
if (!symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
add_delay_check_symbol_is_dummy(decl_context, entry);
}
symbol_entity_specs_set_intent_kind(entry, current_attr_spec.intent_kind);
}
if (current_attr_spec.is_optional)
{
if (!symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
error_printf_at(ast_get_locus(declaration), "OPTIONAL attribute is only for dummy arguments\n");
continue;
}
symbol_entity_specs_set_is_optional(entry, 1);
}
if (current_attr_spec.is_allocatable)
{
if (is_pointer_type(entry->type_information))
{
error_printf_at(ast_get_locus(name), "attribute POINTER conflicts with ALLOCATABLE\n");
continue;
}
symbol_entity_specs_set_is_allocatable(entry, 1);
entry->kind = SK_VARIABLE;
}
if (symbol_entity_specs_get_is_allocatable(entry)
&& fortran_is_array_type(no_ref(entry->type_information)))
{
check_array_type_is_valid_for_allocatable(
no_ref(entry->type_information),
entry,
ast_get_locus(declaration));
}
if (current_attr_spec.is_intrinsic)
{
scope_entry_t* intrinsic_name = fortran_query_intrinsic_name_str(decl_context, entry->symbol_name);
if (intrinsic_name == NULL
|| !symbol_entity_specs_get_is_builtin(intrinsic_name))
{
error_printf_at(ast_get_locus(name), "name '%s' is not known as an intrinsic\n",
ASTText(name));
}
else
{
remove_entry(entry->decl_context->current_scope, entry);
insert_alias(entry->decl_context->current_scope, intrinsic_name, intrinsic_name->symbol_name);
continue;
}
}
if (current_attr_spec.is_external
&& !is_error_type(entry->type_information))
{
if (is_function_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(name), "entity '%s' already has the EXTERNAL attribute\n",
entry->symbol_name);
}
char was_ref = is_lvalue_reference_type(entry->type_information);
entry->type_information = get_nonproto_function_type(entry->type_information, 0);
if (was_ref)
{
entry->type_information = get_lvalue_reference_type(entry->type_information);
}
if (!current_attr_spec.is_pointer
&& !is_pointer_type(no_ref(entry->type_information)))
{
if (!symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
entry->kind = SK_FUNCTION;
symbol_entity_specs_set_is_extern(entry, 1);
}
else
{
entry->kind = SK_VARIABLE;
}
}
}
if (current_attr_spec.is_pointer
&& !is_error_type(entry->type_information))
{
if (current_attr_spec.is_pointer
&& is_pointer_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(name), "entity '%s' already has the POINTER attribute\n",
entry->symbol_name);
}
else if (symbol_entity_specs_get_is_allocatable(entry))
{
error_printf_at(ast_get_locus(name), "attribute ALLOCATABLE conflicts with POINTER\n");
}
else
{
entry->kind = SK_VARIABLE;
symbol_entity_specs_set_is_extern(entry, 0);
char was_ref = is_lvalue_reference_type(entry->type_information);
entry->type_information = get_pointer_type(no_ref(entry->type_information));
if (was_ref)
{
entry->type_information = get_lvalue_reference_type(entry->type_information);
}
}
}
if (fortran_is_pointer_to_array_type(no_ref(entry->type_information)))
{
check_array_type_is_valid_for_pointer(
no_ref(entry->type_information),
entry,
ast_get_locus(declaration));
}
if (current_attr_spec.is_save)
{
symbol_entity_specs_set_is_static(entry, 1);
}
if (current_attr_spec.is_contiguous)
{
if (!array_is_assumed_shape(entry, decl_context)
&& !fortran_is_pointer_to_array_type(entry->type_information))
{
error_printf_at(ast_get_locus(name),
"CONTIGUOUS attribute is only valid for pointers to arrays "
"or assumed-shape arrays\n");
}
symbol_entity_specs_set_is_contiguous(entry, 1);
}
symbol_entity_specs_set_is_target(entry, current_attr_spec.is_target);
if (initialization != NULL)
{
entry->kind = SK_VARIABLE;
if (ASTKind(initialization) == AST_POINTER_INITIALIZATION
&& is_pointer_type(no_ref(entry->type_information)))
{
initialization = ASTSon0(initialization);
fortran_delay_check_initialization(
entry,
initialization,
decl_context,
1,
0,
0,
1);
}
else if (is_pointer_type(no_ref(entry->type_information)))
{
error_printf_at(ast_get_locus(initialization), "a POINTER must be initialized using pointer initialization\n");
}
else if (ASTKind(initialization) == AST_POINTER_INITIALIZATION)
{
error_printf_at(ast_get_locus(initialization), "no POINTER attribute, required for pointer initialization\n");
}
else
{
if (current_attr_spec.is_constant)
{
fortran_immediate_check_initialization(
entry,
initialization,
decl_context,
0,
1,
current_attr_spec.is_constant,
!current_attr_spec.is_constant);
}
else
{
fortran_delay_check_initialization(
entry,
initialization,
decl_context,
0,
1,
current_attr_spec.is_constant,
!current_attr_spec.is_constant);
}
}
}
if (is_pointer_type(no_ref(entry->type_information))
&& current_attr_spec.is_constant)
{
error_printf_at(ast_get_locus(declaration), "PARAMETER attribute is not compatible with POINTER attribute\n");
}
if (current_attr_spec.is_constant
&& initialization == NULL)
{
error_printf_at(ast_get_locus(declaration), "PARAMETER is missing an initializer\n");
}
if (current_attr_spec.is_public
|| current_attr_spec.is_private)
{
if (symbol_entity_specs_get_access(entry) != AS_UNKNOWN)
{
error_printf_at(ast_get_locus(declaration), "access specifier already given for entity '%s'\n",
entry->symbol_name);
}
if (current_attr_spec.is_public)
{
symbol_entity_specs_set_access(entry, AS_PUBLIC);
}
else if (current_attr_spec.is_private)
{
symbol_entity_specs_set_access(entry, AS_PRIVATE);
}
}
DEBUG_CODE()
{
fprintf(stderr, "BUILDSCOPE: Type of symbol '%s' is '%s'\n", entry->symbol_name, print_declarator(entry->type_information));
}
if (current_attr_spec.is_variable)
{
if (entry->kind == SK_VARIABLE)
{
}
else if (entry->kind == SK_UNDEFINED)
{
entry->kind = SK_VARIABLE;
}
else
{
internal_error("%s: internal access specifier <is-variable> was passed but the name "
"'%s' was not an undefined nor a variable name\n",
ast_location(declaration),
entry->symbol_name);
}
}
if (!nodecl_is_null(current_attr_spec.bind_info)
&& !nodecl_is_err_expr(current_attr_spec.bind_info))
{
symbol_entity_specs_set_bind_info(entry,
current_attr_spec.bind_info);
}
}
if (num_delayed_character_symbols > 0)
{
delayed_character_length_t *data = 
delayed_character_length_new(
basic_type,
character_length_out,
decl_context,
num_delayed_character_symbols,
delayed_character_symbols);
build_scope_delay_list_add(
DELAY_AFTER_DECLARATIONS, delayed_compute_character_length, data);
DELETE(delayed_character_symbols);
delayed_character_symbols = NULL;
}
}
static void build_scope_declaration_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
build_scope_declaration_common_stmt(a, decl_context, 
0,
nodecl_output);
}
static void build_scope_typedef_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
build_scope_declaration_common_stmt(a, 
decl_context, 
1,
nodecl_output);
}
static void build_scope_nodecl_literal(AST a, const decl_context_t* decl_context UNUSED_PARAMETER, nodecl_t* nodecl_output)
{
*nodecl_output = nodecl_make_from_ast_nodecl_literal(a);
if (!nodecl_is_list(*nodecl_output))
{
*nodecl_output = nodecl_make_list_1(*nodecl_output);
}
}
static void build_scope_unlock_stmt(AST a, const decl_context_t* decl_context UNUSED_PARAMETER, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
unsupported_statement(a, "UNLOCK");
}
static char come_from_the_same_module(scope_entry_t* new_symbol_used,
scope_entry_t* existing_symbol)
{
new_symbol_used = fortran_get_ultimate_symbol(new_symbol_used);
existing_symbol = fortran_get_ultimate_symbol(existing_symbol);
return new_symbol_used == existing_symbol;
}
scope_entry_t* insert_symbol_from_module(scope_entry_t* entry, 
const decl_context_t* decl_context, 
const char* local_name, 
scope_entry_t* module_symbol,
const locus_t* locus)
{
ERROR_CONDITION(local_name == NULL, "Invalid alias name", 0);
scope_entry_list_t* check_repeated_name = query_in_scope_str_flags(decl_context, local_name, NULL, DF_ONLY_CURRENT_SCOPE);
if (check_repeated_name != NULL
&& strcasecmp(local_name, entry->symbol_name) == 0)
{
scope_entry_list_iterator_t *it = NULL;
for (it = entry_list_iterator_begin(check_repeated_name);
!entry_list_iterator_end(it);
entry_list_iterator_next(it))
{
scope_entry_t* existing_name = entry_list_iterator_current(it);
if (come_from_the_same_module(entry, existing_name))
{
entry_list_iterator_free(it);
return existing_name;
}
}
entry_list_iterator_free(it);
}
entry_list_free(check_repeated_name);
scope_entry_t* named_entry_from_module = entry;
if (symbol_entity_specs_get_from_module(entry) != NULL)
{
ERROR_CONDITION(symbol_entity_specs_get_alias_to(entry) == NULL, 
"Bad symbol with from_module attribute but no alias set", 0);
entry = symbol_entity_specs_get_alias_to(entry);
}
scope_entry_t* current_symbol = NULL;
current_symbol = new_fortran_symbol_not_unknown(decl_context, local_name);
*current_symbol = *entry;
symbol_clear_indirect_types(current_symbol);
current_symbol->decl_context = decl_context;
current_symbol->symbol_name = local_name;
current_symbol->locus = locus;
symbol_entity_specs_set_from_module(current_symbol, module_symbol);
symbol_entity_specs_set_alias_to(current_symbol, entry);
symbol_entity_specs_set_is_renamed(current_symbol, 0);
if (symbol_entity_specs_get_from_module(entry) != NULL
&& symbol_entity_specs_get_alias_to(entry) != NULL)
{
symbol_entity_specs_set_alias_to(current_symbol, symbol_entity_specs_get_alias_to(entry));
}
symbol_entity_specs_set_access(current_symbol, AS_UNKNOWN);
if (strcmp(local_name, named_entry_from_module->symbol_name) != 0)
{
symbol_entity_specs_set_is_renamed(current_symbol, 1);
}
symbol_entity_specs_set_from_module_name(current_symbol, named_entry_from_module->symbol_name);
if (decl_context->current_scope->related_entry != NULL
&& decl_context->current_scope->related_entry->kind == SK_MODULE)
{
scope_entry_t* module = decl_context->current_scope->related_entry;
symbol_entity_specs_add_related_symbols(module, current_symbol);
symbol_entity_specs_set_in_module(current_symbol, module);
}
else
{
symbol_entity_specs_set_in_module(current_symbol, NULL);
}
return current_symbol;
}
scope_entry_t* fortran_load_module(const char* module_name_str, char must_be_intrinsic_module,
const locus_t* locus)
{
scope_entry_t* module_symbol = NULL;
DEBUG_CODE()
{
fprintf(stderr, "BUILDSCOPE: Loading module '%s'\n", module_name_str);
}
rb_red_blk_node* query = rb_tree_query(CURRENT_COMPILED_FILE->module_file_cache, module_name_str);
char must_load = 1;
if (query != NULL)
{
module_symbol = (scope_entry_t*)rb_node_get_info(query);
if (module_symbol->defined)
{
DEBUG_CODE()
{
fprintf(stderr, "BUILDSCOPE: Module '%s' was in the module cache of this file and already loaded\n", module_name_str);
}
must_load = 0;
}
else
{
DEBUG_CODE()
{
fprintf(stderr, "BUILDSCOPE: Module '%s' was in the module cache of this file but not already loaded\n", module_name_str);
}
}
}
else
{
DEBUG_CODE()
{
fprintf(stderr, "BUILDSCOPE: Module '%s' was not in the module cache of this file\n", module_name_str);
}
}
if (must_load)
{
DEBUG_CODE()
{
fprintf(stderr, "BUILDSCOPE: Loading module '%s' from the filesystem\n", module_name_str);
}
load_module_info(module_name_str, &module_symbol);
if (module_symbol == NULL)
{
if (must_be_intrinsic_module)
{
error_printf_at(locus, "module '%s' is not an INTRINSIC module\n", module_name_str);
}
else
{
fatal_printf_at(locus, "cannot load module '%s'\n", module_name_str);
}
}
ERROR_CONDITION(module_symbol == NULL, "Invalid symbol", 0);
rb_tree_insert(CURRENT_COMPILED_FILE->module_file_cache, module_name_str, module_symbol);
}
if (must_be_intrinsic_module
&& !symbol_entity_specs_get_is_builtin(module_symbol))
{
error_printf_at(locus, "loaded module '%s' is not an INTRINSIC module\n", module_name_str);
}
return module_symbol;
}
static void build_scope_use_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST module_nature = NULL;
AST module_name = NULL;
AST rename_list = NULL;
AST only_list = NULL;
char is_only = 0;
if (ASTKind(a) == AST_USE_STATEMENT)
{
module_nature = ASTSon0(a);
module_name = ASTSon1(a);
rename_list = ASTSon2(a);
}
else if (ASTKind(a) == AST_USE_ONLY_STATEMENT)
{
module_nature = ASTSon0(a);
module_name = ASTSon1(a);
only_list = ASTSon2(a);
is_only = 1;
}
else
{
internal_error("Unexpected node %s", ast_print_node_type(ASTKind(a)));
}
char must_be_intrinsic_module = 0;
if (module_nature != NULL)
{
must_be_intrinsic_module = (strcasecmp(ASTText(module_nature), "INTRINSIC") == 0);
}
const char* module_name_str = strtolower(ASTText(module_name));
scope_entry_t* module_symbol = fortran_load_module(module_name_str, must_be_intrinsic_module,
ast_get_locus(a));
scope_entry_t* used_modules = get_or_create_used_modules_symbol_info(decl_context);
symbol_entity_specs_add_related_symbols(used_modules, module_symbol);
nodecl_t nodecl_fortran_use;
nodecl_t nodecl_used_symbols = nodecl_null();
if (!is_only)
{
int num_renamed_symbols = 0;
scope_entry_t* renamed_symbols[MCXX_MAX_RENAMED_SYMBOLS];
memset(renamed_symbols, 0, sizeof(renamed_symbols));
if (rename_list != NULL)
{
AST it;
for_each_element(rename_list, it)
{
AST rename_tree = ASTSon1(it);
AST local_name = ASTSon0(rename_tree);
AST sym_in_module_name = ASTSon1(rename_tree);
const char* sym_in_module_name_str = get_name_of_generic_spec(sym_in_module_name);
scope_entry_list_t* syms_in_module = 
fortran_query_module_for_name(module_symbol, sym_in_module_name_str);
if (syms_in_module == NULL)
{
fatal_printf_at(ast_get_locus(sym_in_module_name),
"symbol '%s' not found in module '%s'\n",
prettyprint_in_buffer(sym_in_module_name),
module_symbol->symbol_name);
}
scope_entry_list_iterator_t* entry_list_it = NULL;
for (entry_list_it = entry_list_iterator_begin(syms_in_module);
!entry_list_iterator_end(entry_list_it);
entry_list_iterator_next(entry_list_it))
{
scope_entry_t* sym_in_module = entry_list_iterator_current(entry_list_it);
scope_entry_t* inserted_symbol = insert_symbol_from_module(sym_in_module, 
decl_context, 
get_name_of_generic_spec(local_name), 
module_symbol, 
ast_get_locus(local_name));
nodecl_used_symbols = nodecl_append_to_list(
nodecl_used_symbols,
nodecl_make_symbol(inserted_symbol, 
ast_get_locus(local_name)));
char found = 0;
int i;
for (i = 0; i < num_renamed_symbols && found; i++)
{
found = (renamed_symbols[i] == sym_in_module);
}
if (!found)
{
ERROR_CONDITION(num_renamed_symbols == MCXX_MAX_RENAMED_SYMBOLS, "Too many renames", 0);
renamed_symbols[num_renamed_symbols] = sym_in_module;
num_renamed_symbols++;
}
}
entry_list_iterator_free(entry_list_it);
}
}
int i;
for (i = 0; i < symbol_entity_specs_get_num_related_symbols(module_symbol); i++)
{
scope_entry_t* sym_in_module = symbol_entity_specs_get_related_symbols_num(module_symbol, i);
if (symbol_entity_specs_get_access(sym_in_module) == AS_PRIVATE)
continue;
char found = 0;
int j;
for (j = 0; j < num_renamed_symbols && !found; j++)
{
found = (renamed_symbols[j] == sym_in_module);
}
if (!found)
{
insert_symbol_from_module(sym_in_module, 
decl_context, 
sym_in_module->symbol_name, 
module_symbol,
ast_get_locus(a));
}
}
nodecl_fortran_use = nodecl_make_fortran_use(
nodecl_make_symbol(module_symbol, ast_get_locus(a)),
nodecl_used_symbols, ast_get_locus(a));
}
else  if (only_list != NULL) 
{
AST it;
for_each_element(only_list, it)
{
AST only = ASTSon1(it);
switch (ASTKind(only))
{
case AST_RENAME:
{
AST local_name = ASTSon0(only);
AST sym_in_module_name = ASTSon1(only);
const char * sym_in_module_name_str = get_name_of_generic_spec(sym_in_module_name);
scope_entry_list_t* syms_in_module = 
fortran_query_module_for_name(module_symbol, sym_in_module_name_str);
if (syms_in_module == NULL)
{
fatal_printf_at(ast_get_locus(sym_in_module_name),
"symbol '%s' not found in module '%s'\n",
prettyprint_in_buffer(sym_in_module_name),
module_symbol->symbol_name);
}
scope_entry_list_iterator_t* entry_list_it = NULL;
for (entry_list_it = entry_list_iterator_begin(syms_in_module);
!entry_list_iterator_end(entry_list_it);
entry_list_iterator_next(entry_list_it))
{
scope_entry_t* sym_in_module = entry_list_iterator_current(entry_list_it);
scope_entry_t* inserted_symbol = insert_symbol_from_module(sym_in_module, 
decl_context, 
get_name_of_generic_spec(local_name), 
module_symbol, 
ast_get_locus(local_name));
nodecl_used_symbols = nodecl_append_to_list(
nodecl_used_symbols,
nodecl_make_symbol(inserted_symbol, 
ast_get_locus(local_name)));
}
entry_list_iterator_free(entry_list_it);
break;
}
default:
{
AST sym_in_module_name = only;
const char * sym_in_module_name_str = get_name_of_generic_spec(sym_in_module_name);
scope_entry_list_t* syms_in_module = 
fortran_query_module_for_name(module_symbol, sym_in_module_name_str);
if (syms_in_module == NULL)
{
fatal_printf_at(
ast_get_locus(sym_in_module_name),
"symbol '%s' not found in module '%s'\n", 
prettyprint_in_buffer(sym_in_module_name),
module_symbol->symbol_name);
}
scope_entry_list_iterator_t* entry_list_it = NULL;
for (entry_list_it = entry_list_iterator_begin(syms_in_module);
!entry_list_iterator_end(entry_list_it);
entry_list_iterator_next(entry_list_it))
{
scope_entry_t* sym_in_module = entry_list_iterator_current(entry_list_it);
scope_entry_t* inserted_symbol = insert_symbol_from_module(sym_in_module, 
decl_context, 
sym_in_module->symbol_name, 
module_symbol,
ast_get_locus(sym_in_module_name));
nodecl_used_symbols = nodecl_append_to_list(
nodecl_used_symbols,
nodecl_make_symbol(inserted_symbol,
ast_get_locus(sym_in_module_name)));
}
entry_list_iterator_free(entry_list_it);
break;
}
}
}
nodecl_fortran_use = nodecl_make_fortran_use_only(
nodecl_make_symbol(module_symbol, ast_get_locus(a)),
nodecl_used_symbols, ast_get_locus(a));
}
else 
{
return;
}
used_modules->value = nodecl_append_to_list(used_modules->value, nodecl_fortran_use);
}
static void build_scope_value_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST name_list = ASTSon0(a);
AST it;
for_each_element(name_list, it)
{
AST name = ASTSon1(it);
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
if (!symbol_is_parameter_of_function(entry, decl_context->current_scope->related_entry))
{
error_printf_at(ast_get_locus(name), "entity '%s' is not a dummy argument\n",
entry->symbol_name);
continue;
}
if (entry->kind == SK_UNDEFINED)
{
entry->kind = SK_VARIABLE;
}
if (is_lvalue_reference_type(entry->type_information))
{
entry->type_information = reference_type_get_referenced_type(entry->type_information);
}
else
{
error_printf_at(ast_get_locus(name), "entity '%s' already had VALUE attribute\n",
entry->symbol_name);
}
}
}
static void build_scope_volatile_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
AST name_list = ASTSon0(a);
AST it;
for_each_element(name_list, it)
{
AST name = ASTSon1(it);
scope_entry_t* entry = get_symbol_for_name(decl_context, name, ASTText(name));
if (entry->kind == SK_UNDEFINED)
{
entry->kind = SK_VARIABLE;
}
char is_ref = is_lvalue_reference_type(entry->type_information);
if (!is_error_type(entry->type_information))
{
if (!is_volatile_qualified_type(no_ref(entry->type_information)))
{
if (!is_ref)
{
entry->type_information = get_volatile_qualified_type(entry->type_information);
}
else
{
entry->type_information = get_lvalue_reference_type(
get_volatile_qualified_type(no_ref(entry->type_information)));
}
}
else
{
error_printf_at(ast_get_locus(a), "entity '%s' already has VOLATILE attribute\n", entry->symbol_name);
continue;
}
}
}
}
static void build_scope_wait_stmt(AST a, const decl_context_t* decl_context UNUSED_PARAMETER, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
unsupported_statement(a, "WAIT");
}
static void build_scope_where_body_construct_seq(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output UNUSED_PARAMETER)
{
if (a == NULL)
return;
AST it;
for_each_element(a, it)
{
AST statement = ASTSon1(it);
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement_inside_block_context(statement, decl_context, &nodecl_statement);
*nodecl_output = nodecl_concat_lists(*nodecl_output, nodecl_statement);
}
}
static void build_scope_mask_elsewhere_part_seq(AST mask_elsewhere_part_seq, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
if (mask_elsewhere_part_seq == NULL)
return;
AST it;
for_each_element(mask_elsewhere_part_seq, it)
{
AST mask_elsewhere_part = ASTSon1(it);
AST masked_elsewhere_stmt = ASTSon0(mask_elsewhere_part);
AST where_body_construct_seq = ASTSon1(mask_elsewhere_part);
AST expr = ASTSon0(masked_elsewhere_stmt);
nodecl_t nodecl_expr = nodecl_null();
fortran_check_expression(expr, decl_context, &nodecl_expr);
nodecl_expr = fortran_expression_as_value(nodecl_expr);
nodecl_t nodecl_statement = nodecl_null();
build_scope_where_body_construct_seq(where_body_construct_seq, decl_context, &nodecl_statement);
*nodecl_output = nodecl_append_to_list(*nodecl_output,
nodecl_make_fortran_where_pair(
nodecl_expr,
nodecl_statement,
ast_get_locus(expr)));
}
}
static void build_scope_where_construct(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST where_construct_stmt = ASTSon0(a);
AST mask_expr = ASTSon1(where_construct_stmt);
nodecl_t nodecl_mask_expr = nodecl_null();
fortran_check_expression(mask_expr, decl_context, &nodecl_mask_expr);
nodecl_mask_expr = fortran_expression_as_value(nodecl_mask_expr);
AST where_construct_body = ASTSon1(a);
if (where_construct_body == NULL)
{
nodecl_t nodecl_where_parts = nodecl_make_list_1(
nodecl_make_fortran_where_pair(
nodecl_mask_expr, nodecl_null(),
ast_get_locus(a)));
*nodecl_output =
nodecl_make_list_1(
nodecl_make_fortran_where(nodecl_where_parts, ast_get_locus(a))
);
}
else
{
AST main_where_body = ASTSon0(where_construct_body);
nodecl_t nodecl_body = nodecl_null();
nodecl_t nodecl_where_parts;
if (main_where_body != NULL)
{
build_scope_where_body_construct_seq(main_where_body, decl_context, &nodecl_body);
nodecl_where_parts = nodecl_make_list_1(
nodecl_make_fortran_where_pair(
nodecl_mask_expr,
nodecl_body,
ast_get_locus(a)));
}
else
{
nodecl_where_parts = nodecl_make_list_1(
nodecl_make_fortran_where_pair(
nodecl_mask_expr, nodecl_null(),
ast_get_locus(a)));
}
AST mask_elsewhere_part_seq = ASTSon1(where_construct_body);
nodecl_t nodecl_elsewhere_parts = nodecl_null();
build_scope_mask_elsewhere_part_seq(mask_elsewhere_part_seq, decl_context, &nodecl_elsewhere_parts);
nodecl_where_parts = nodecl_concat_lists(nodecl_where_parts, nodecl_elsewhere_parts);
AST elsewhere_body = ASTSon3(where_construct_body);
if (elsewhere_body != NULL)
{
nodecl_t nodecl_elsewhere_body = nodecl_null();
build_scope_where_body_construct_seq(elsewhere_body, decl_context, &nodecl_elsewhere_body);
nodecl_where_parts = nodecl_concat_lists(nodecl_where_parts,
nodecl_make_list_1(nodecl_make_fortran_where_pair(
nodecl_null(),
nodecl_elsewhere_body,
ast_get_locus(a))));
}
*nodecl_output =
nodecl_make_list_1(
nodecl_make_fortran_where(nodecl_where_parts, ast_get_locus(a))
);
}
}
static void build_scope_where_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST mask_expr = ASTSon0(a);
nodecl_t nodecl_mask_expr = nodecl_null();
fortran_check_expression(mask_expr, decl_context, &nodecl_mask_expr);
nodecl_mask_expr = fortran_expression_as_value(nodecl_mask_expr);
AST where_assignment_stmt = ASTSon1(a);
nodecl_t nodecl_expression = nodecl_null();
build_scope_expression_stmt(where_assignment_stmt, decl_context, &nodecl_expression);
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_where(
nodecl_make_list_1(
nodecl_make_fortran_where_pair(
nodecl_mask_expr,
nodecl_expression,
ast_get_locus(a))),
ast_get_locus(a)));
}
static void build_scope_while_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST expr = ASTSon0(a);
AST block = ASTSon1(a);
AST end_do_statement = ASTSon2(a);
const char* construct_name = strtolower(ASTText(a));
nodecl_t nodecl_named_label = nodecl_null();
if (construct_name != NULL)
{
scope_entry_t* named_label = fortran_query_construct_name_str(
construct_name, decl_context,  1,
ast_get_locus(a));
nodecl_named_label = nodecl_make_symbol(named_label, ast_get_locus(a));
}
nodecl_t nodecl_expr = nodecl_null();
fortran_check_expression(expr, decl_context, &nodecl_expr);
nodecl_expr = fortran_expression_as_value(nodecl_expr);
if (!is_bool_type(nodecl_get_type(nodecl_expr)))
{
error_printf_at(ast_get_locus(expr), "condition of DO WHILE loop is not a logical expression\n");
}
nodecl_t nodecl_statement = nodecl_null();
fortran_build_scope_statement_inside_block_context(block, decl_context, &nodecl_statement);
if (end_do_statement != NULL
&& ASTKind(end_do_statement) == AST_LABELED_STATEMENT)
{
AST label = ASTSon0(end_do_statement);
scope_entry_t* label_sym = fortran_query_label(label, decl_context,  1);
nodecl_t nodecl_labeled_empty_statement = 
nodecl_make_labeled_statement(
nodecl_make_list_1(
nodecl_make_empty_statement(ast_get_locus(end_do_statement))
),
label_sym,
ast_get_locus(end_do_statement));
nodecl_statement = nodecl_append_to_list(nodecl_statement, nodecl_labeled_empty_statement);
}
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_while_statement(nodecl_expr,
nodecl_statement, 
nodecl_named_label,
ast_get_locus(a)));
}
static void build_scope_write_stmt(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
nodecl_t nodecl_opt_value = nodecl_null();
handle_opt_value_list(a, ASTSon0(a), decl_context, &nodecl_opt_value);
nodecl_t nodecl_io_items = nodecl_null();
AST input_output_item_list = ASTSon1(a);
if (input_output_item_list != NULL)
{
build_scope_input_output_item_list(input_output_item_list, decl_context, &nodecl_io_items);
}
*nodecl_output = 
nodecl_make_list_1(
nodecl_make_fortran_write_statement(nodecl_opt_value, nodecl_io_items, ast_get_locus(a))
);
}
void fortran_build_scope_statement_pragma(AST a, 
const decl_context_t* decl_context, 
nodecl_t* nodecl_output, 
void* info UNUSED_PARAMETER)
{
fortran_build_scope_statement_inside_block_context(a, decl_context, nodecl_output);
}
static void build_scope_pragma_custom_ctr(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
nodecl_t nodecl_pragma_line = nodecl_null();
common_build_scope_pragma_custom_statement(a, decl_context, nodecl_output, &nodecl_pragma_line,
fortran_build_scope_statement_pragma, NULL);
*nodecl_output = nodecl_make_list_1(*nodecl_output);
}
static void build_scope_pragma_custom_dir(AST a, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
common_build_scope_pragma_custom_directive(a, decl_context, nodecl_output);
*nodecl_output = nodecl_make_list_1(*nodecl_output);
}
static void build_scope_unknown_pragma(AST a, const decl_context_t* decl_context UNUSED_PARAMETER, nodecl_t* nodecl_output)
{
*nodecl_output =
nodecl_make_list_1(
nodecl_make_unknown_pragma(ASTText(a), ast_get_locus(a)));
}
static void build_scope_statement_placeholder(AST a, const decl_context_t* decl_context UNUSED_PARAMETER, nodecl_t* nodecl_output)
{
check_statement_placeholder(a, decl_context, nodecl_output);
}
typedef void opt_value_fun_handler_t(AST io_stmt, AST opt_value, const decl_context_t*, nodecl_t*);
typedef struct opt_value_map_tag
{
const char* name;
opt_value_fun_handler_t *handler;
} opt_value_map_t;
#define OPT_VALUE_LIST \
OPT_VALUE(access) \
OPT_VALUE(acquired) \
OPT_VALUE(action) \
OPT_VALUE(advance) \
OPT_VALUE(asynchronous) \
OPT_VALUE(blank) \
OPT_VALUE(buffered) \
OPT_VALUE(convert) \
OPT_VALUE(decimal) \
OPT_VALUE(delim) \
OPT_VALUE(direct) \
OPT_VALUE(encoding) \
OPT_VALUE(eor) \
OPT_VALUE(err) \
OPT_VALUE(end) \
OPT_VALUE(errmsg) \
OPT_VALUE(exist) \
OPT_VALUE(file) \
OPT_VALUE(fmt) \
OPT_VALUE(form) \
OPT_VALUE(formatted) \
OPT_VALUE(id) \
OPT_VALUE(iomsg) \
OPT_VALUE(iostat) \
OPT_VALUE(iolength) \
OPT_VALUE(mold) \
OPT_VALUE(name) \
OPT_VALUE(named) \
OPT_VALUE(newunit) \
OPT_VALUE(nextrec) \
OPT_VALUE(nml) \
OPT_VALUE(number) \
OPT_VALUE(opened) \
OPT_VALUE(pad) \
OPT_VALUE(pending) \
OPT_VALUE(pos) \
OPT_VALUE(position) \
OPT_VALUE(read) \
OPT_VALUE(readwrite) \
OPT_VALUE(rec) \
OPT_VALUE(recl) \
OPT_VALUE(round) \
OPT_VALUE(sequential) \
OPT_VALUE(sign) \
OPT_VALUE(size) \
OPT_VALUE(source) \
OPT_VALUE(stat) \
OPT_VALUE(status) \
OPT_VALUE(stream) \
OPT_VALUE(unformatted) \
OPT_VALUE(unit) \
OPT_VALUE(write) \
OPT_VALUE(ambiguous_io_spec) 
#define OPT_VALUE(_name) \
static opt_value_fun_handler_t opt_##_name##_handler;
OPT_VALUE_LIST
#undef OPT_VALUE
static opt_value_map_t opt_value_map[] =
{
#define OPT_VALUE(_name) \
{ #_name, opt_##_name##_handler },
OPT_VALUE_LIST
#undef OPT_VALUE
};
static char opt_value_list_init = 0;
static int opt_value_map_compare(const void* v1, const void* v2)
{
const opt_value_map_t* p1 = (const opt_value_map_t*) v1;
const opt_value_map_t* p2 = (const opt_value_map_t*) v2;
return strcasecmp(p1->name, p2->name);
}
static void handle_opt_value(AST io_stmt, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
opt_value_map_t key;
key.name = ASTText(opt_value);
ERROR_CONDITION(key.name == NULL, "Invalid opt_value without name of opt", 0);
opt_value_map_t *elem =
(opt_value_map_t*)bsearch(&key, opt_value_map, 
sizeof(opt_value_map) / sizeof(opt_value_map[1]),
sizeof(opt_value_map[0]),
opt_value_map_compare);
ERROR_CONDITION(elem == NULL, "Invalid opt-value '%s' at %s\n", key.name, ast_location(opt_value));
ERROR_CONDITION(elem->handler == NULL, "Invalid handler for opt-value '%s'\n", key.name);
(elem->handler)(io_stmt, opt_value, decl_context, nodecl_output);
}
static void handle_opt_value_list(AST io_stmt, AST opt_value_list, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
if (!opt_value_list_init)
{
qsort(opt_value_map, 
sizeof(opt_value_map) / sizeof(opt_value_map[1]),
sizeof(opt_value_map[0]),
opt_value_map_compare);
opt_value_list_init = 1;
}
if (opt_value_list == NULL)
return;
AST it;
for_each_element(opt_value_list, it)
{
AST opt_value = ASTSon1(it);
nodecl_t nodecl_opt_value = nodecl_null();
handle_opt_value(io_stmt, opt_value, decl_context, &nodecl_opt_value);
*nodecl_output = nodecl_append_to_list(*nodecl_output, nodecl_opt_value);
}
}
static char check_opt_common_int_expr(nodecl_t* nodecl_value)
{
type_t* t = no_ref(nodecl_get_type(*nodecl_value));
return is_integer_type(t);
}
static char opt_common_int_expr(AST value, const decl_context_t* decl_context, const char* opt_name, nodecl_t* nodecl_value)
{
fortran_check_expression(value, decl_context, nodecl_value);
*nodecl_value = fortran_expression_as_value(*nodecl_value);
char ok = check_opt_common_int_expr(nodecl_value);
if (!ok)
{
error_printf_at(ast_get_locus(value), "specifier %s requires a character expression\n",
opt_name);
return 0;
}
return 1;
}
static char opt_common_character_expr(AST value, const decl_context_t* decl_context, const char* opt_name, nodecl_t* nodecl_value)
{
fortran_check_expression(value, decl_context, nodecl_value);
*nodecl_value = fortran_expression_as_value(*nodecl_value);
if (!fortran_is_character_type(nodecl_get_type(*nodecl_value)))
{
error_printf_at(ast_get_locus(value), "specifier %s requires a character expression\n",
opt_name);
return 0;
}
return 1;
}
static char opt_common_const_character_expr(AST value, const decl_context_t* decl_context, const char* opt_name, nodecl_t* nodecl_value)
{
return opt_common_character_expr(value, decl_context, opt_name, nodecl_value);
}
static char opt_common_int_variable(AST value, const decl_context_t* decl_context, const char* opt_name, nodecl_t* nodecl_value)
{
fortran_check_expression(value, decl_context, nodecl_value);
*nodecl_value = fortran_expression_as_variable(*nodecl_value);
type_t* t = nodecl_get_type(*nodecl_value);
char ok = is_lvalue_reference_type(t) && check_opt_common_int_expr(nodecl_value);
if (!ok)
{
error_printf_at(ast_get_locus(value), "specifier %s requires an integer variable\n",
opt_name);
return 0;
}
return 1;
}
static char opt_common_logical_variable(AST value, const decl_context_t* decl_context, const char* opt_name, nodecl_t* nodecl_value)
{
fortran_check_expression(value, decl_context, nodecl_value);
*nodecl_value = fortran_expression_as_variable(*nodecl_value);
type_t* t = nodecl_get_type(*nodecl_value);
char ok = is_lvalue_reference_type(t) && is_bool_type(no_ref(t));
if (!ok)
{
error_printf_at(ast_get_locus(value), "specifier %s requires a logical variable\n",
opt_name);
return 0;
}
return 1;
}
static void opt_access_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "ACCESS", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "ACCESS", ast_get_locus(opt_value));
}
static void opt_convert_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "CONVERT", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "CONVERT", ast_get_locus(opt_value));
}
static void opt_acquired_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
fortran_check_expression(value, decl_context, &nodecl_value);
nodecl_value = fortran_expression_as_value(nodecl_value);
if (fortran_data_ref_get_symbol(nodecl_value) == NULL
|| !is_bool_type(no_ref(fortran_data_ref_get_symbol(nodecl_value)->type_information)))
{
error_printf_at(ast_get_locus(value), "specifier 'ACQUIRED LOCK' requires a logical variable\n");
}
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "ACQUIRED LOCK", ast_get_locus(opt_value));
}
static void opt_action_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "ACTION", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "ACTION", ast_get_locus(opt_value));
}
static void opt_advance_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
nodecl_t nodecl_value = nodecl_null();
AST value = ASTSon0(opt_value);
opt_common_const_character_expr(value, decl_context, "ADVANCE", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "ADVANCE", ast_get_locus(opt_value));
}
static void opt_asynchronous_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "ASYNCHRONOUS", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "ASYNCHRONOUS", ast_get_locus(opt_value));
}
static void opt_blank_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "BLANK", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "BLANK", ast_get_locus(opt_value));
}
static void opt_buffered_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "BUFFERED", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "BUFFERED", ast_get_locus(opt_value));
}
static void opt_decimal_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "DECIMAL", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "DECIMAL", ast_get_locus(opt_value));
}
static void opt_delim_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "DELIM", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "DELIM", ast_get_locus(opt_value));
}
static void opt_direct_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "DIRECT", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "DIRECT", ast_get_locus(opt_value));
}
static void opt_encoding_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "ENCODING", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "ENCODING", ast_get_locus(opt_value));
}
static void opt_eor_handler(AST io_stmt UNUSED_PARAMETER, 
AST opt_value UNUSED_PARAMETER, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output)
{
AST label = ASTSon0(opt_value);
scope_entry_t* entry = fortran_query_label(label, decl_context,  0);
*nodecl_output = nodecl_make_fortran_io_spec(
nodecl_make_symbol(entry, ast_get_locus(label)), 
"EOR", ast_get_locus(opt_value));
}
static void opt_err_handler(AST io_stmt UNUSED_PARAMETER, 
AST opt_value UNUSED_PARAMETER, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output)
{
AST label = ASTSon0(opt_value);
scope_entry_t* entry = fortran_query_label(label, decl_context,  0);
*nodecl_output = nodecl_make_fortran_io_spec(
nodecl_make_symbol(entry, ast_get_locus(label)), 
"ERR", ast_get_locus(opt_value));
}
static void opt_end_handler(AST io_stmt UNUSED_PARAMETER, 
AST opt_value UNUSED_PARAMETER, 
const decl_context_t* decl_context UNUSED_PARAMETER, 
nodecl_t* nodecl_output)
{
AST label = ASTSon0(opt_value);
scope_entry_t* entry = fortran_query_label(label, decl_context,  0);
*nodecl_output = nodecl_make_fortran_io_spec(
nodecl_make_symbol(entry, ast_get_locus(label)), 
"END", ast_get_locus(opt_value));
}
static void opt_errmsg_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, 
nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "ERRMSG", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "ERRMSG", ast_get_locus(opt_value));
}
static void opt_exist_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_logical_variable(value, decl_context, "EXIST", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "EXIST", ast_get_locus(opt_value));
}
static void opt_file_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "FILE", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "FILE", ast_get_locus(opt_value));
}
static void opt_fmt_value(AST value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
if (!(ASTKind(value) == AST_SYMBOL
&& strcmp(ASTText(value), "*") == 0))
{
char valid = 1;
nodecl_t nodecl_value = nodecl_null();
if (ASTKind(value) == AST_DECIMAL_LITERAL)
{
scope_entry_t* entry = fortran_query_label(value, decl_context,  0);
if (entry == NULL)
{
valid = 0;
}
else
{
nodecl_value = nodecl_make_symbol(entry, ast_get_locus(value));
}
}
else 
{
fortran_check_expression(value, decl_context, &nodecl_value);
type_t* t = nodecl_get_type(nodecl_value);
scope_entry_t* entry = fortran_data_ref_get_symbol(nodecl_value);
if (fortran_is_character_type(no_ref(t)) 
|| (fortran_is_array_type(no_ref(t)) && 
fortran_is_character_type(no_ref(get_unqualified_type(fortran_get_rank0_type(t))))))
{
nodecl_value = fortran_expression_as_value(nodecl_value);
}
else if (entry != NULL
&& entry->kind == SK_VARIABLE
&& equivalent_types(entry->type_information, fortran_get_default_integer_type()))
{
}
else
{
valid = 0;
}
}
if (!valid)
{
error_printf_at(ast_get_locus(value), "specifier FMT requires a character expression, a label of a FORMAT statement or an ASSIGNED variable\n");
}
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "FMT", ast_get_locus(value));
}
else
{
*nodecl_output = nodecl_make_fortran_io_spec(
nodecl_make_text("*", ast_get_locus(value)), 
"FMT", ast_get_locus(value));
}
}
static void opt_fmt_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
opt_fmt_value(value, decl_context, nodecl_output);
}
static void opt_form_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "FORM", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "FORM", ast_get_locus(opt_value));
}
static void opt_formatted_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "FORMATTED", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "FORMATTED", ast_get_locus(opt_value));
}
static void opt_id_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_int_expr(value, decl_context, "ID", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "ID", ast_get_locus(opt_value));
}
static void opt_iomsg_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "IOMSG", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "IOMSG", ast_get_locus(opt_value));
}
static void opt_iostat_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_int_expr(value, decl_context, "IOSTAT", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "IOSTAT", ast_get_locus(opt_value));
}
static void opt_iolength_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_int_variable(value, decl_context, "IOLENGTH", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "IOLENGTH", ast_get_locus(opt_value));
}
static void opt_mold_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
fortran_check_expression(value, decl_context, &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "MOLD", ast_get_locus(opt_value));
}
static void opt_name_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "NAME", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "NAME", ast_get_locus(opt_value));
}
static void opt_named_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_logical_variable(value, decl_context, "NAMED", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "NAMED", ast_get_locus(opt_value));
}
static void opt_newunit_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_int_variable(value, decl_context, "NEWUNIT", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "NEWUNIT", ast_get_locus(opt_value));
}
static void opt_nextrec_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_int_variable(value, decl_context, "NEXTREC", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "NEXTREC", ast_get_locus(opt_value));
}
static void opt_nml_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
scope_entry_t* entry = fortran_query_name_str(decl_context, ASTText(value),
ast_get_locus(value));
if (entry == NULL
|| entry->kind != SK_NAMELIST)
{
error_printf_at(ast_get_locus(value), "entity '%s' in NML specifier is not a namelist\n",
ASTText(value));
}
*nodecl_output = nodecl_make_fortran_io_spec(
nodecl_make_symbol(entry, ast_get_locus(value)), 
"NML", ast_get_locus(opt_value));
}
static void opt_number_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_int_variable(value, decl_context, "NUMBER", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "NUMBER", ast_get_locus(opt_value));
}
static void opt_opened_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_logical_variable(value, decl_context, "OPENED", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "OPENED", ast_get_locus(opt_value));
}
static void opt_pad_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "PAD", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "PAD", ast_get_locus(opt_value));
}
static void opt_pending_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_logical_variable(value, decl_context, "PENDING", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "PENDING", ast_get_locus(opt_value));
}
static void opt_pos_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_int_expr(value, decl_context, "POS", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "POS", ast_get_locus(opt_value));
}
static void opt_position_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "POSITION", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "POSITION", ast_get_locus(opt_value));
}
static void opt_read_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "READ", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "READ", ast_get_locus(opt_value));
}
static void opt_readwrite_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "READWRITE", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "READWRITE", ast_get_locus(opt_value));
}
static void opt_rec_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_int_expr(value, decl_context, "REC", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "REC", ast_get_locus(opt_value));
}
static void opt_recl_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_int_expr(value, decl_context, "RECL", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "RECL", ast_get_locus(opt_value));
}
static void opt_round_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "ROUND", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "ROUND", ast_get_locus(opt_value));
}
static void opt_sequential_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "SEQUENTIAL", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "SEQUENTIAL", ast_get_locus(opt_value));
}
static void opt_sign_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "SIGN", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "SIGN", ast_get_locus(opt_value));
}
static void opt_size_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_int_expr(value, decl_context, "SIZE", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "SIZE", ast_get_locus(opt_value));
}
static void opt_source_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
fortran_check_expression(value, decl_context, &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "SOURCE", ast_get_locus(opt_value));
}
static void opt_stat_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_int_variable(value, decl_context, "STAT", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "STAT", ast_get_locus(opt_value));
}
static void opt_status_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "STATUS", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "STATUS", ast_get_locus(opt_value));
}
static void opt_stream_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "STREAM", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "STREAM", ast_get_locus(opt_value));
}
static void opt_unformatted_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "UNFORMATTED", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "UNFORMATTED", ast_get_locus(opt_value));
}
static void opt_unit_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
if (!(ASTKind(value) == AST_SYMBOL
&& strcmp(ASTText(value), "*") == 0))
{
nodecl_t nodecl_value = nodecl_null();
fortran_check_expression(value, decl_context, &nodecl_value);
type_t* t = nodecl_get_type(nodecl_value);
if (!(is_integer_type(no_ref(t))
|| (fortran_data_ref_get_symbol(nodecl_value) != NULL
&& fortran_is_character_type_or_pointer_to(
fortran_get_rank0_type(no_ref(t))))))
{
error_printf_at(ast_get_locus(value), "specifier UNIT requires a character variable or a scalar integer expression\n");
}
nodecl_value = fortran_expression_as_value(nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "UNIT", ast_get_locus(opt_value));
}
else
{
*nodecl_output = nodecl_make_fortran_io_spec(
nodecl_make_text("*", ast_get_locus(value)),
"UNIT", ast_get_locus(opt_value));
}
}
static void opt_write_handler(AST io_stmt UNUSED_PARAMETER, AST opt_value, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
AST value = ASTSon0(opt_value);
nodecl_t nodecl_value = nodecl_null();
opt_common_character_expr(value, decl_context, "WRITE", &nodecl_value);
*nodecl_output = nodecl_make_fortran_io_spec(nodecl_value, "WRITE", ast_get_locus(opt_value));
}
static int get_position_in_io_spec_list(AST value)
{
AST list = ASTParent(value);
int n = 0;
AST it;
for_each_element(list, it)
{
n++;
}
return n;
}
static void opt_ambiguous_io_spec_handler(AST io_stmt, AST opt_value_ambig, const decl_context_t* decl_context, nodecl_t* nodecl_output)
{
int io_unit_option = -1;
int namelist_option = -1;
int format_option = -1;
int i;
for (i = 0; i < ast_get_num_ambiguities(opt_value_ambig); i++)
{
AST option = ast_get_ambiguity(opt_value_ambig, i);
const char* t = ASTText(option);
ERROR_CONDITION((t == NULL), "io-spec is missing text", 0);
int *p = NULL;
if (strcasecmp(t, "unit") == 0)
{
p = &io_unit_option;
}
else if (strcasecmp(t, "fmt") == 0)
{
p = &format_option;
}
else if (strcasecmp(t, "nml") == 0)
{
p = &namelist_option;
}
else
{
internal_error("%s: Unexpected opt_value_ambig io-spec '%s'\n", ast_location(option), t);
}
ERROR_CONDITION(*p >= 0, "%s Repeated ambiguity tree!", ast_location(option));
*p = i;
}
int position = get_position_in_io_spec_list(opt_value_ambig);
char bad = 0;
if (position == 1)
{
if (io_unit_option < 0)
{
bad = 1;
}
else
{
ast_replace_with_ambiguity(opt_value_ambig, io_unit_option);
}
}
else if (position == 2)
{
AST parent = ASTParent(opt_value_ambig);
AST previous = ASTSon0(parent);
AST io_spec = ASTSon1(previous);
if ((ASTText(io_spec) == NULL)
|| (strcasecmp(ASTText(io_spec), "unit") != 0))
{
bad = 1;
}
else
{
if (namelist_option < 0)
{
if (format_option < 0)
{
bad = 1;
}
else
{
ast_replace_with_ambiguity(opt_value_ambig, format_option);
}
}
else
{
AST nml_io_spec = ast_get_ambiguity(opt_value_ambig, namelist_option);
AST value = ASTSon0(nml_io_spec);
scope_entry_t* entry = fortran_query_name_str(decl_context, ASTText(value),
ast_get_locus(value));
if (entry == NULL
|| entry->kind != SK_NAMELIST)
{
if (format_option < 0)
{
bad = 1;
}
else
{
ast_replace_with_ambiguity(opt_value_ambig, format_option);
}
}
else
{
ast_replace_with_ambiguity(opt_value_ambig, namelist_option);
}
}
}
}
else
{
bad = 1;
}
if (!bad)
{
handle_opt_value(io_stmt, opt_value_ambig, decl_context, nodecl_output);
}
else
{
error_printf_at(ast_get_locus(opt_value_ambig), "invalid io-control-spec '%s'\n",
fortran_prettyprint_in_buffer(opt_value_ambig));
*nodecl_output = nodecl_make_err_expr(ast_get_locus(opt_value_ambig));
}
}
static char check_statement_function_statement(AST stmt, const decl_context_t* decl_context)
{
AST name = ASTSon0(stmt);
AST dummy_arg_name_list = ASTSon1(stmt);
AST expr = ASTSon2(stmt);
scope_entry_t* entry = fortran_query_name_str(decl_context, ASTText(name),
ast_get_locus(name));
if (entry == NULL)
{
entry = 
create_fortran_symbol_for_name_(decl_context, name, ASTText(name),  0);
}
if (symbol_entity_specs_get_from_module(entry) != NULL)
return 0;
if (!fortran_is_scalar_type(no_ref(entry->type_information)))
return 0;
if (dummy_arg_name_list != NULL)
{
AST it;
for_each_element(dummy_arg_name_list, it)
{
AST dummy_name = ASTSon1(it);
scope_entry_t* dummy_arg = get_symbol_for_name(decl_context, dummy_name, ASTText(dummy_name));
if (!fortran_is_scalar_type(no_ref(dummy_arg->type_information)))
return 0;
}
}
diagnostic_context_push_buffered();
nodecl_t nodecl_dummy = nodecl_null();
fortran_check_expression(expr, decl_context, &nodecl_dummy);
diagnostic_context_pop_and_discard();
if (nodecl_is_err_expr(nodecl_dummy))
return 0;
return 1;
}
static void build_scope_ambiguity_statement(AST ambig_stmt, const decl_context_t* decl_context, char is_declaration)
{
ERROR_CONDITION(ASTKind(ambig_stmt) != AST_AMBIGUITY, "Invalid tree %s\n", ast_print_node_type(ASTKind(ambig_stmt)));
ERROR_CONDITION(strcmp(ASTText(ambig_stmt), "ASSIGNMENT") != 0, "Invalid ambiguity", 0);
int num_ambig = ast_get_num_ambiguities(ambig_stmt);
int i;
int result = -1;
int index_expr = -1;
for (i = 0; i < num_ambig; i++)
{
AST stmt = ast_get_ambiguity(ambig_stmt, i);
if (ASTKind(stmt) == AST_LABELED_STATEMENT)
stmt = ASTSon1(stmt);
char ok = 0;
switch (ASTKind(stmt))
{
case AST_EXPRESSION_STATEMENT:
{
index_expr = i;
if (!is_declaration)
{
diagnostic_context_push_buffered();
nodecl_t nodecl_dummy = nodecl_null();
fortran_check_expression(ASTSon0(stmt), decl_context, &nodecl_dummy);
ok = !nodecl_is_err_expr(nodecl_dummy);
diagnostic_context_pop_and_discard();
}
break;
}
case AST_STATEMENT_FUNCTION_STATEMENT:
{
if (is_declaration)
{
ok = check_statement_function_statement(stmt, decl_context);
}
break;
}
default:
{
internal_error("Invalid node '%s' at %s\n", ast_print_node_type(ASTKind(ambig_stmt)), ast_location(ambig_stmt));
}
}
if (ok)
{
if (result == -1)
{
result = i;
}
else
{
result = -2;
}
}
}
ERROR_CONDITION(index_expr < 0, "Something is utterly broken in this ambiguous node\n", 0);
if (result < 0)
{
ast_replace_with_ambiguity(ambig_stmt, index_expr);
}
else
{
ast_replace_with_ambiguity(ambig_stmt, result);
}
}
scope_entry_t* function_get_result_symbol(scope_entry_t* entry)
{
return symbol_entity_specs_get_result_var(entry);
}
static scope_entry_t* symbol_name_is_in_external_list(const char *name,
scope_entry_list_t* external_function_list)
{
scope_entry_list_iterator_t* it;
for (it = entry_list_iterator_begin(external_function_list);
!entry_list_iterator_end(it);
entry_list_iterator_next(it))
{
scope_entry_t* current = entry_list_iterator_current(it);
if (strcmp(current->symbol_name, name) == 0)
{
entry_list_iterator_free(it);
return current;
}
}
entry_list_iterator_free(it);
return NULL;
}
static void resolve_external_calls_rec(nodecl_t node,
scope_entry_list_t* external_function_list)
{
if (nodecl_is_null(node))
return;
int i;
for (i = 0; i < MCXX_MAX_AST_CHILDREN; i++)
{
resolve_external_calls_rec(
nodecl_get_child(node, i),
external_function_list);
}
nodecl_t called;
if (nodecl_get_kind(node) == NODECL_FUNCTION_CALL
&& nodecl_get_kind((called = nodecl_get_child(node, 0))) == NODECL_SYMBOL)
{
scope_entry_t* entry = fortran_data_ref_get_symbol(called);
if (entry->kind == SK_FUNCTION
&& (entry->decl_context->current_scope->related_entry == NULL
|| !symbol_is_parameter_of_function(entry,
entry->decl_context->current_scope->related_entry))
&& !symbol_entity_specs_get_is_nested_function(entry)
&& !symbol_entity_specs_get_is_module_procedure(entry)
&& !symbol_entity_specs_get_is_builtin(entry)
&& !symbol_entity_specs_get_is_stmt_function(entry))
{
DEBUG_CODE()
{
fprintf(stderr, "%s: info: reference to external procedure '%s'\n",
nodecl_locus_to_str(node),
entry->symbol_name);
}
scope_entry_t* external_symbol = symbol_name_is_in_external_list(entry->symbol_name, external_function_list);
if (external_symbol != NULL)
{
type_t* current_type = entry->type_information;
ERROR_CONDITION(!is_function_type(current_type), "Something is amiss here", 0);
if (function_type_get_lacking_prototype(current_type))
{
DEBUG_CODE()
{
fprintf(stderr, "%s: info: fixing reference to unknown interface '%s'\n",
nodecl_locus_to_str(node),
entry->symbol_name);
fprintf(stderr, "%s: info: to this program unit definition\n",
locus_to_str(external_symbol->locus));
}
nodecl_set_symbol(called, external_symbol);
nodecl_t alternate_name = nodecl_get_child(node, 2);
if (nodecl_is_null(alternate_name))
{
alternate_name = nodecl_make_symbol(entry, 
nodecl_get_locus(node));
nodecl_set_child(node, 2, alternate_name);
}
}
}
}
}
}
static void resolve_external_calls_inside_a_function(nodecl_t function_code,
scope_entry_list_t* external_function_list)
{
nodecl_t body = nodecl_get_child(function_code, 0);
nodecl_t internals = nodecl_get_child(function_code, 1);
resolve_external_calls_rec(body, external_function_list);
int i, n = 0;
nodecl_t* list = nodecl_unpack_list(internals, &n);
for (i = 0; i < n; i++)
{
if (nodecl_get_kind(list[i]) == NODECL_FUNCTION_CODE)
{
resolve_external_calls_inside_a_function(list[i], external_function_list);
}
}
DELETE(list);
}
static void resolve_external_calls_inside_file(nodecl_t nodecl_program_units)
{
if (nodecl_is_null(nodecl_program_units))
return;
DEBUG_CODE()
{
fprintf(stderr, "Resolving calls with unknown interface\n");
}
int i, n = 0;
nodecl_t* list = nodecl_unpack_list(nodecl_program_units, &n);
scope_entry_list_t* external_function_list = NULL;
for (i = 0; i < n; i++)
{
if (nodecl_get_kind(list[i]) == NODECL_FUNCTION_CODE)
{
scope_entry_t* function = nodecl_get_symbol(list[i]);
if (function->kind == SK_FUNCTION
&& !symbol_entity_specs_get_is_nested_function(function)
&& !symbol_entity_specs_get_is_module_procedure(function)
&& (function->decl_context->current_scope == function->decl_context->global_scope))
{
DEBUG_CODE()
{
fprintf(stderr, "%s: info: found program unit of external procedure '%s'\n",
nodecl_locus_to_str(list[i]),
function->symbol_name);
}
external_function_list = entry_list_add(external_function_list, function);
}
}
}
for (i = 0; i < n; i++)
{
if (nodecl_get_kind(list[i]) == NODECL_FUNCTION_CODE)
{
resolve_external_calls_inside_a_function(list[i], external_function_list);
}
}
DELETE(list);
DEBUG_CODE()
{
fprintf(stderr, "End resolving calls with unknown interface\n");
}
}
