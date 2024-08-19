#ifndef GCC_CP_PARSER_H
#define GCC_CP_PARSER_H
#include "tree.h"
#include "cp/cp-tree.h"
#include "c-family/c-pragma.h"
struct GTY(()) tree_check {
tree value;
vec<deferred_access_check, va_gc> *checks;
tree qualifying_scope;
};
struct GTY (()) cp_token {
ENUM_BITFIELD (cpp_ttype) type : 8;
ENUM_BITFIELD (rid) keyword : 8;
unsigned char flags;
BOOL_BITFIELD implicit_extern_c : 1;
BOOL_BITFIELD error_reported : 1;
BOOL_BITFIELD purged_p : 1;
location_t location;
union cp_token_value {
struct tree_check* GTY((tag ("1"))) tree_check_value;
tree GTY((tag ("0"))) value;
} GTY((desc ("(%1.type == CPP_TEMPLATE_ID)"
"|| (%1.type == CPP_NESTED_NAME_SPECIFIER)"
"|| (%1.type == CPP_DECLTYPE)"))) u;
};
typedef struct cp_token *cp_token_position;
struct GTY (()) cp_lexer {
vec<cp_token, va_gc> *buffer;
cp_token_position GTY ((skip)) last_token;
cp_token_position GTY ((skip)) next_token;
vec<cp_token_position> GTY ((skip)) saved_tokens;
struct cp_lexer *next;
bool debugging_p;
bool in_pragma;
};
struct GTY(()) cp_token_cache {
cp_token * GTY((skip)) first;
cp_token * GTY ((skip)) last;
};
typedef cp_token_cache *cp_token_cache_ptr;
struct cp_token_ident
{
unsigned int ident_len;
const char *ident_str;
unsigned int before_len;
const char *before_str;
unsigned int after_len;
const char *after_str;
};
struct GTY(()) cp_default_arg_entry {
tree class_type;
tree decl;
};
struct GTY(()) cp_unparsed_functions_entry {
vec<cp_default_arg_entry, va_gc> *funs_with_default_args;
vec<tree, va_gc> *funs_with_definitions;
vec<tree, va_gc> *nsdmis;
vec<tree, va_gc> *classes;
};
enum cp_parser_status_kind
{
CP_PARSER_STATUS_KIND_NO_ERROR,
CP_PARSER_STATUS_KIND_ERROR,
CP_PARSER_STATUS_KIND_COMMITTED
};
struct GTY (()) cp_parser_context {
enum cp_parser_status_kind status;
tree object_type;
struct cp_parser_context *next;
};
struct cp_omp_declare_simd_data {
bool error_seen; 
bool fndecl_seen; 
vec<cp_token_cache_ptr> tokens;
tree clauses;
};
struct cp_oacc_routine_data : cp_omp_declare_simd_data {
location_t loc;
};
struct GTY(()) cp_parser {
cp_lexer *lexer;
tree scope;
tree object_scope;
tree qualifying_scope;
cp_parser_context *context;
bool allow_gnu_extensions_p;
bool greater_than_is_operator_p;
bool default_arg_ok_p;
bool integral_constant_expression_p;
bool allow_non_integral_constant_expression_p;
bool non_integral_constant_expression_p;
bool local_variables_forbidden_p;
bool in_unbraced_linkage_specification_p;
bool in_declarator_p;
bool in_template_argument_list_p;
#define IN_SWITCH_STMT		1
#define IN_ITERATION_STMT	2
#define IN_OMP_BLOCK		4
#define IN_OMP_FOR		8
#define IN_IF_STMT             16
unsigned char in_statement;
bool in_switch_statement_p;
bool in_type_id_in_expr_p;
bool implicit_extern_c;
bool translate_strings_p;
bool in_function_body;
unsigned char in_transaction;
bool colon_corrects_to_scope_p;
bool colon_doesnt_start_class_def_p;
const char *type_definition_forbidden_message;
vec<cp_unparsed_functions_entry, va_gc> *unparsed_queues;
unsigned num_classes_being_defined;
unsigned num_template_parameter_lists;
cp_omp_declare_simd_data * GTY((skip)) omp_declare_simd;
cp_oacc_routine_data * GTY((skip)) oacc_routine;
bool auto_is_implicit_function_template_parm_p;
bool fully_implicit_function_template_p;
tree implicit_template_parms;
cp_binding_level* implicit_template_scope;
bool in_result_type_constraint_p;
int prevent_constrained_type_specifiers;
location_t innermost_linkage_specification_location;
};
extern void debug (cp_token &ref);
extern void debug (cp_token *ptr);
extern void cp_lexer_debug_tokens (vec<cp_token, va_gc> *);
extern void debug (vec<cp_token, va_gc> &ref);
extern void debug (vec<cp_token, va_gc> *ptr);
extern void cp_debug_parser (FILE *, cp_parser *);
extern void debug (cp_parser &ref);
extern void debug (cp_parser *ptr);
extern bool cp_keyword_starts_decl_specifier_p (enum rid keyword);
#endif  
