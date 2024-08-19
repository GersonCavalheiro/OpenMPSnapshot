#include "config.h"
#define INCLUDE_UNIQUE_PTR
#include "system.h"
#include "coretypes.h"
#include "cp-tree.h"
#include "c-family/c-common.h"
#include "timevar.h"
#include "stringpool.h"
#include "cgraph.h"
#include "print-tree.h"
#include "attribs.h"
#include "trans-mem.h"
#include "intl.h"
#include "decl.h"
#include "c-family/c-objc.h"
#include "plugin.h"
#include "tree-pretty-print.h"
#include "parser.h"
#include "gomp-constants.h"
#include "omp-general.h"
#include "omp-offload.h"
#include "c-family/c-indentation.h"
#include "context.h"
#include "gcc-rich-location.h"
#include "tree-iterator.h"
#include "c-family/name-hint.h"

static cp_token eof_token =
{
CPP_EOF, RID_MAX, 0, false, false, false, 0, { NULL }
};
enum non_integral_constant {
NIC_NONE,
NIC_FLOAT,
NIC_THIS,
NIC_FUNC_NAME,
NIC_PRETTY_FUNC,
NIC_C99_FUNC,
NIC_VA_ARG,
NIC_CAST,
NIC_TYPEID,
NIC_NCC,
NIC_FUNC_CALL,
NIC_INC,
NIC_DEC,
NIC_ARRAY_REF,
NIC_ARROW,
NIC_POINT,
NIC_ADDR_LABEL,
NIC_STAR,
NIC_ADDR,
NIC_PREINCREMENT,
NIC_PREDECREMENT,
NIC_NEW,
NIC_DEL,
NIC_OVERLOADED,
NIC_ASSIGNMENT,
NIC_COMMA,
NIC_CONSTRUCTOR,
NIC_TRANSACTION
};
enum name_lookup_error {
NLE_NULL,
NLE_TYPE,
NLE_CXX98,
NLE_NOT_CXX98
};
enum required_token {
RT_NONE,
RT_SEMICOLON,  
RT_OPEN_PAREN, 
RT_CLOSE_BRACE, 
RT_OPEN_BRACE,  
RT_CLOSE_SQUARE, 
RT_OPEN_SQUARE,  
RT_COMMA, 
RT_SCOPE, 
RT_LESS, 
RT_GREATER, 
RT_EQ, 
RT_ELLIPSIS, 
RT_MULT, 
RT_COMPL, 
RT_COLON, 
RT_COLON_SCOPE, 
RT_CLOSE_PAREN, 
RT_COMMA_CLOSE_PAREN, 
RT_PRAGMA_EOL, 
RT_NAME, 
RT_NEW, 
RT_DELETE, 
RT_RETURN, 
RT_WHILE, 
RT_EXTERN, 
RT_STATIC_ASSERT, 
RT_DECLTYPE, 
RT_OPERATOR, 
RT_CLASS, 
RT_TEMPLATE, 
RT_NAMESPACE, 
RT_USING, 
RT_ASM, 
RT_TRY, 
RT_CATCH, 
RT_THROW, 
RT_LABEL, 
RT_AT_TRY, 
RT_AT_SYNCHRONIZED, 
RT_AT_THROW, 
RT_SELECT,  
RT_ITERATION, 
RT_JUMP, 
RT_CLASS_KEY, 
RT_CLASS_TYPENAME_TEMPLATE, 
RT_TRANSACTION_ATOMIC, 
RT_TRANSACTION_RELAXED, 
RT_TRANSACTION_CANCEL 
};
class type_id_in_expr_sentinel
{
cp_parser *parser;
bool saved;
public:
type_id_in_expr_sentinel (cp_parser *parser, bool set = true)
: parser (parser),
saved (parser->in_type_id_in_expr_p)
{ parser->in_type_id_in_expr_p = set; }
~type_id_in_expr_sentinel ()
{ parser->in_type_id_in_expr_p = saved; }
};
static cp_lexer *cp_lexer_new_main
(void);
static cp_lexer *cp_lexer_new_from_tokens
(cp_token_cache *tokens);
static void cp_lexer_destroy
(cp_lexer *);
static int cp_lexer_saving_tokens
(const cp_lexer *);
static cp_token *cp_lexer_token_at
(cp_lexer *, cp_token_position);
static void cp_lexer_get_preprocessor_token
(cp_lexer *, cp_token *);
static inline cp_token *cp_lexer_peek_token
(cp_lexer *);
static cp_token *cp_lexer_peek_nth_token
(cp_lexer *, size_t);
static inline bool cp_lexer_next_token_is
(cp_lexer *, enum cpp_ttype);
static bool cp_lexer_next_token_is_not
(cp_lexer *, enum cpp_ttype);
static bool cp_lexer_next_token_is_keyword
(cp_lexer *, enum rid);
static cp_token *cp_lexer_consume_token
(cp_lexer *);
static void cp_lexer_purge_token
(cp_lexer *);
static void cp_lexer_purge_tokens_after
(cp_lexer *, cp_token_position);
static void cp_lexer_save_tokens
(cp_lexer *);
static void cp_lexer_commit_tokens
(cp_lexer *);
static void cp_lexer_rollback_tokens
(cp_lexer *);
static void cp_lexer_print_token
(FILE *, cp_token *);
static inline bool cp_lexer_debugging_p
(cp_lexer *);
static void cp_lexer_start_debugging
(cp_lexer *) ATTRIBUTE_UNUSED;
static void cp_lexer_stop_debugging
(cp_lexer *) ATTRIBUTE_UNUSED;
static cp_token_cache *cp_token_cache_new
(cp_token *, cp_token *);
static void cp_parser_initial_pragma
(cp_token *);
static bool cp_parser_omp_declare_reduction_exprs
(tree, cp_parser *);
static void cp_finalize_oacc_routine
(cp_parser *, tree, bool);
#define CP_LEXER_BUFFER_SIZE ((256 * 1024) / sizeof (cp_token))
#define CP_SAVED_TOKEN_STACK 5
static FILE *cp_lexer_debug_stream;
int cp_unevaluated_operand;
static void
cp_lexer_dump_tokens (FILE *file, vec<cp_token, va_gc> *buffer,
cp_token *start_token, unsigned num,
cp_token *curr_token)
{
unsigned i, nprinted;
cp_token *token;
bool do_print;
fprintf (file, "%u tokens\n", vec_safe_length (buffer));
if (buffer == NULL)
return;
if (num == 0)
num = buffer->length ();
if (start_token == NULL)
start_token = buffer->address ();
if (start_token > buffer->address ())
{
cp_lexer_print_token (file, &(*buffer)[0]);
fprintf (file, " ... ");
}
do_print = false;
nprinted = 0;
for (i = 0; buffer->iterate (i, &token) && nprinted < num; i++)
{
if (token == start_token)
do_print = true;
if (!do_print)
continue;
nprinted++;
if (token == curr_token)
fprintf (file, "[[");
cp_lexer_print_token (file, token);
if (token == curr_token)
fprintf (file, "]]");
switch (token->type)
{
case CPP_SEMICOLON:
case CPP_OPEN_BRACE:
case CPP_CLOSE_BRACE:
case CPP_EOF:
fputc ('\n', file);
break;
default:
fputc (' ', file);
}
}
if (i == num && i < buffer->length ())
{
fprintf (file, " ... ");
cp_lexer_print_token (file, &buffer->last ());
}
fprintf (file, "\n");
}
void
cp_lexer_debug_tokens (vec<cp_token, va_gc> *buffer)
{
cp_lexer_dump_tokens (stderr, buffer, NULL, 0, NULL);
}
DEBUG_FUNCTION void
debug (vec<cp_token, va_gc> &ref)
{
cp_lexer_dump_tokens (stderr, &ref, NULL, 0, NULL);
}
DEBUG_FUNCTION void
debug (vec<cp_token, va_gc> *ptr)
{
if (ptr)
debug (*ptr);
else
fprintf (stderr, "<nil>\n");
}
static void
cp_debug_print_tree_if_set (FILE *file, const char *desc, tree t)
{
if (t)
{
fprintf (file, "%s: ", desc);
print_node_brief (file, "", t, 0);
}
}
static void
cp_debug_print_context (FILE *file, cp_parser_context *c)
{
const char *status_s[] = { "OK", "ERROR", "COMMITTED" };
fprintf (file, "{ status = %s, scope = ", status_s[c->status]);
print_node_brief (file, "", c->object_type, 0);
fprintf (file, "}\n");
}
static void
cp_debug_print_context_stack (FILE *file, cp_parser_context *first)
{
unsigned i;
cp_parser_context *c;
fprintf (file, "Parsing context stack:\n");
for (i = 0, c = first; c; c = c->next, i++)
{
fprintf (file, "\t#%u: ", i);
cp_debug_print_context (file, c);
}
}
static void
cp_debug_print_flag (FILE *file, const char *desc, bool flag)
{
if (flag)
fprintf (file, "%s: true\n", desc);
}
static void
cp_debug_print_unparsed_function (FILE *file, cp_unparsed_functions_entry *uf)
{
unsigned i;
cp_default_arg_entry *default_arg_fn;
tree fn;
fprintf (file, "\tFunctions with default args:\n");
for (i = 0;
vec_safe_iterate (uf->funs_with_default_args, i, &default_arg_fn);
i++)
{
fprintf (file, "\t\tClass type: ");
print_node_brief (file, "", default_arg_fn->class_type, 0);
fprintf (file, "\t\tDeclaration: ");
print_node_brief (file, "", default_arg_fn->decl, 0);
fprintf (file, "\n");
}
fprintf (file, "\n\tFunctions with definitions that require "
"post-processing\n\t\t");
for (i = 0; vec_safe_iterate (uf->funs_with_definitions, i, &fn); i++)
{
print_node_brief (file, "", fn, 0);
fprintf (file, " ");
}
fprintf (file, "\n");
fprintf (file, "\n\tNon-static data members with initializers that require "
"post-processing\n\t\t");
for (i = 0; vec_safe_iterate (uf->nsdmis, i, &fn); i++)
{
print_node_brief (file, "", fn, 0);
fprintf (file, " ");
}
fprintf (file, "\n");
}
static void
cp_debug_print_unparsed_queues (FILE *file,
vec<cp_unparsed_functions_entry, va_gc> *s)
{
unsigned i;
cp_unparsed_functions_entry *uf;
fprintf (file, "Unparsed functions\n");
for (i = 0; vec_safe_iterate (s, i, &uf); i++)
{
fprintf (file, "#%u:\n", i);
cp_debug_print_unparsed_function (file, uf);
}
}
static void
cp_debug_parser_tokens (FILE *file, cp_parser *parser, int window_size)
{
cp_token *next_token, *first_token, *start_token;
if (file == NULL)
file = stderr;
next_token = parser->lexer->next_token;
first_token = parser->lexer->buffer->address ();
start_token = (next_token > first_token + window_size / 2)
? next_token - window_size / 2
: first_token;
cp_lexer_dump_tokens (file, parser->lexer->buffer, start_token, window_size,
next_token);
}
void
cp_debug_parser (FILE *file, cp_parser *parser)
{
const size_t window_size = 20;
cp_token *token;
expanded_location eloc;
if (file == NULL)
file = stderr;
fprintf (file, "Parser state\n\n");
fprintf (file, "Number of tokens: %u\n",
vec_safe_length (parser->lexer->buffer));
cp_debug_print_tree_if_set (file, "Lookup scope", parser->scope);
cp_debug_print_tree_if_set (file, "Object scope",
parser->object_scope);
cp_debug_print_tree_if_set (file, "Qualifying scope",
parser->qualifying_scope);
cp_debug_print_context_stack (file, parser->context);
cp_debug_print_flag (file, "Allow GNU extensions",
parser->allow_gnu_extensions_p);
cp_debug_print_flag (file, "'>' token is greater-than",
parser->greater_than_is_operator_p);
cp_debug_print_flag (file, "Default args allowed in current "
"parameter list", parser->default_arg_ok_p);
cp_debug_print_flag (file, "Parsing integral constant-expression",
parser->integral_constant_expression_p);
cp_debug_print_flag (file, "Allow non-constant expression in current "
"constant-expression",
parser->allow_non_integral_constant_expression_p);
cp_debug_print_flag (file, "Seen non-constant expression",
parser->non_integral_constant_expression_p);
cp_debug_print_flag (file, "Local names and 'this' forbidden in "
"current context",
parser->local_variables_forbidden_p);
cp_debug_print_flag (file, "In unbraced linkage specification",
parser->in_unbraced_linkage_specification_p);
cp_debug_print_flag (file, "Parsing a declarator",
parser->in_declarator_p);
cp_debug_print_flag (file, "In template argument list",
parser->in_template_argument_list_p);
cp_debug_print_flag (file, "Parsing an iteration statement",
parser->in_statement & IN_ITERATION_STMT);
cp_debug_print_flag (file, "Parsing a switch statement",
parser->in_statement & IN_SWITCH_STMT);
cp_debug_print_flag (file, "Parsing a structured OpenMP block",
parser->in_statement & IN_OMP_BLOCK);
cp_debug_print_flag (file, "Parsing a an OpenMP loop",
parser->in_statement & IN_OMP_FOR);
cp_debug_print_flag (file, "Parsing an if statement",
parser->in_statement & IN_IF_STMT);
cp_debug_print_flag (file, "Parsing a type-id in an expression "
"context", parser->in_type_id_in_expr_p);
cp_debug_print_flag (file, "Declarations are implicitly extern \"C\"",
parser->implicit_extern_c);
cp_debug_print_flag (file, "String expressions should be translated "
"to execution character set",
parser->translate_strings_p);
cp_debug_print_flag (file, "Parsing function body outside of a "
"local class", parser->in_function_body);
cp_debug_print_flag (file, "Auto correct a colon to a scope operator",
parser->colon_corrects_to_scope_p);
cp_debug_print_flag (file, "Colon doesn't start a class definition",
parser->colon_doesnt_start_class_def_p);
if (parser->type_definition_forbidden_message)
fprintf (file, "Error message for forbidden type definitions: %s\n",
parser->type_definition_forbidden_message);
cp_debug_print_unparsed_queues (file, parser->unparsed_queues);
fprintf (file, "Number of class definitions in progress: %u\n",
parser->num_classes_being_defined);
fprintf (file, "Number of template parameter lists for the current "
"declaration: %u\n", parser->num_template_parameter_lists);
cp_debug_parser_tokens (file, parser, window_size);
token = parser->lexer->next_token;
fprintf (file, "Next token to parse:\n");
fprintf (file, "\tToken:  ");
cp_lexer_print_token (file, token);
eloc = expand_location (token->location);
fprintf (file, "\n\tFile:   %s\n", eloc.file);
fprintf (file, "\tLine:   %d\n", eloc.line);
fprintf (file, "\tColumn: %d\n", eloc.column);
}
DEBUG_FUNCTION void
debug (cp_parser &ref)
{
cp_debug_parser (stderr, &ref);
}
DEBUG_FUNCTION void
debug (cp_parser *ptr)
{
if (ptr)
debug (*ptr);
else
fprintf (stderr, "<nil>\n");
}
static cp_lexer *
cp_lexer_alloc (void)
{
cp_lexer *lexer;
c_common_no_more_pch ();
lexer = ggc_cleared_alloc<cp_lexer> ();
lexer->debugging_p = false;
lexer->saved_tokens.create (CP_SAVED_TOKEN_STACK);
vec_alloc (lexer->buffer, CP_LEXER_BUFFER_SIZE);
return lexer;
}
static cp_lexer *
cp_lexer_new_main (void)
{
cp_lexer *lexer;
cp_token token;
cp_parser_initial_pragma (&token);
lexer = cp_lexer_alloc ();
lexer->buffer->quick_push (token);
while (token.type != CPP_EOF)
{
cp_lexer_get_preprocessor_token (lexer, &token);
vec_safe_push (lexer->buffer, token);
}
lexer->last_token = lexer->buffer->address ()
+ lexer->buffer->length ()
- 1;
lexer->next_token = lexer->buffer->length ()
? lexer->buffer->address ()
: &eof_token;
done_lexing = true;
gcc_assert (!lexer->next_token->purged_p);
return lexer;
}
static cp_lexer *
cp_lexer_new_from_tokens (cp_token_cache *cache)
{
cp_token *first = cache->first;
cp_token *last = cache->last;
cp_lexer *lexer = ggc_cleared_alloc<cp_lexer> ();
lexer->buffer = NULL;
lexer->next_token = first == last ? &eof_token : first;
lexer->last_token = last;
lexer->saved_tokens.create (CP_SAVED_TOKEN_STACK);
lexer->debugging_p = false;
gcc_assert (!lexer->next_token->purged_p);
return lexer;
}
static void
cp_lexer_destroy (cp_lexer *lexer)
{
vec_free (lexer->buffer);
lexer->saved_tokens.release ();
ggc_free (lexer);
}
#define LEXER_DEBUGGING_ENABLED_P false
static inline bool
cp_lexer_debugging_p (cp_lexer *lexer)
{
if (!LEXER_DEBUGGING_ENABLED_P)
return false;
return lexer->debugging_p;
}
static inline cp_token_position
cp_lexer_token_position (cp_lexer *lexer, bool previous_p)
{
gcc_assert (!previous_p || lexer->next_token != &eof_token);
return lexer->next_token - previous_p;
}
static inline cp_token *
cp_lexer_token_at (cp_lexer * , cp_token_position pos)
{
return pos;
}
static inline void
cp_lexer_set_token_position (cp_lexer *lexer, cp_token_position pos)
{
lexer->next_token = cp_lexer_token_at (lexer, pos);
}
static inline cp_token_position
cp_lexer_previous_token_position (cp_lexer *lexer)
{
if (lexer->next_token == &eof_token)
return lexer->last_token - 1;
else
return cp_lexer_token_position (lexer, true);
}
static inline cp_token *
cp_lexer_previous_token (cp_lexer *lexer)
{
cp_token_position tp = cp_lexer_previous_token_position (lexer);
while (tp->purged_p)
{
gcc_assert (tp != vec_safe_address (lexer->buffer));
tp--;
}
return cp_lexer_token_at (lexer, tp);
}
static inline int
cp_lexer_saving_tokens (const cp_lexer* lexer)
{
return lexer->saved_tokens.length () != 0;
}
static void
cp_lexer_get_preprocessor_token (cp_lexer *lexer, cp_token *token)
{
static int is_extern_c = 0;
token->type
= c_lex_with_flags (&token->u.value, &token->location, &token->flags,
lexer == NULL ? 0 : C_LEX_STRING_NO_JOIN);
token->keyword = RID_MAX;
token->purged_p = false;
token->error_reported = false;
is_extern_c += pending_lang_change;
pending_lang_change = 0;
token->implicit_extern_c = is_extern_c > 0;
if (token->type == CPP_NAME)
{
if (IDENTIFIER_KEYWORD_P (token->u.value))
{
token->type = CPP_KEYWORD;
token->keyword = C_RID_CODE (token->u.value);
}
else
{
if (warn_cxx11_compat
&& C_RID_CODE (token->u.value) >= RID_FIRST_CXX11
&& C_RID_CODE (token->u.value) <= RID_LAST_CXX11)
{
warning (OPT_Wc__11_compat, 
"identifier %qE is a keyword in C++11",
token->u.value);
C_SET_RID_CODE (token->u.value, RID_MAX);
}
token->keyword = RID_MAX;
}
}
else if (token->type == CPP_AT_NAME)
{
token->type = CPP_KEYWORD;
switch (C_RID_CODE (token->u.value))
{
case RID_CLASS:     token->keyword = RID_AT_CLASS; break;
case RID_PRIVATE:   token->keyword = RID_AT_PRIVATE; break;
case RID_PROTECTED: token->keyword = RID_AT_PROTECTED; break;
case RID_PUBLIC:    token->keyword = RID_AT_PUBLIC; break;
case RID_THROW:     token->keyword = RID_AT_THROW; break;
case RID_TRY:       token->keyword = RID_AT_TRY; break;
case RID_CATCH:     token->keyword = RID_AT_CATCH; break;
case RID_SYNCHRONIZED: token->keyword = RID_AT_SYNCHRONIZED; break;
default:            token->keyword = C_RID_CODE (token->u.value);
}
}
}
static inline void
cp_lexer_set_source_position_from_token (cp_token *token)
{
if (token->type != CPP_EOF)
{
input_location = token->location;
}
}
static inline void
cp_lexer_set_source_position (cp_lexer *lexer)
{
cp_token *token = cp_lexer_peek_token (lexer);
cp_lexer_set_source_position_from_token (token);
}
static inline cp_token *
cp_lexer_peek_token (cp_lexer *lexer)
{
if (cp_lexer_debugging_p (lexer))
{
fputs ("cp_lexer: peeking at token: ", cp_lexer_debug_stream);
cp_lexer_print_token (cp_lexer_debug_stream, lexer->next_token);
putc ('\n', cp_lexer_debug_stream);
}
return lexer->next_token;
}
static inline bool
cp_lexer_next_token_is (cp_lexer* lexer, enum cpp_ttype type)
{
return cp_lexer_peek_token (lexer)->type == type;
}
static inline bool
cp_lexer_next_token_is_not (cp_lexer* lexer, enum cpp_ttype type)
{
return !cp_lexer_next_token_is (lexer, type);
}
static inline bool
cp_lexer_next_token_is_keyword (cp_lexer* lexer, enum rid keyword)
{
return cp_lexer_peek_token (lexer)->keyword == keyword;
}
static inline bool
cp_lexer_nth_token_is (cp_lexer* lexer, size_t n, enum cpp_ttype type)
{
return cp_lexer_peek_nth_token (lexer, n)->type == type;
}
static inline bool
cp_lexer_nth_token_is_keyword (cp_lexer* lexer, size_t n, enum rid keyword)
{
return cp_lexer_peek_nth_token (lexer, n)->keyword == keyword;
}
static inline bool
cp_lexer_next_token_is_not_keyword (cp_lexer* lexer, enum rid keyword)
{
return cp_lexer_peek_token (lexer)->keyword != keyword;
}
bool
cp_keyword_starts_decl_specifier_p (enum rid keyword)
{
switch (keyword)
{
case RID_AUTO:
case RID_REGISTER:
case RID_STATIC:
case RID_EXTERN:
case RID_MUTABLE:
case RID_THREAD:
case RID_ENUM:
case RID_CLASS:
case RID_STRUCT:
case RID_UNION:
case RID_TYPENAME:
case RID_CHAR:
case RID_CHAR16:
case RID_CHAR32:
case RID_WCHAR:
case RID_BOOL:
case RID_SHORT:
case RID_INT:
case RID_LONG:
case RID_SIGNED:
case RID_UNSIGNED:
case RID_FLOAT:
case RID_DOUBLE:
case RID_VOID:
case RID_ATTRIBUTE:
case RID_TYPEOF:
case RID_DECLTYPE:
case RID_UNDERLYING_TYPE:
case RID_CONSTEXPR:
return true;
default:
if (keyword >= RID_FIRST_INT_N
&& keyword < RID_FIRST_INT_N + NUM_INT_N_ENTS
&& int_n_enabled_p[keyword - RID_FIRST_INT_N])
return true;
return false;
}
}
static bool
cp_lexer_next_token_is_decl_specifier_keyword (cp_lexer *lexer)
{
cp_token *token;
token = cp_lexer_peek_token (lexer);
return cp_keyword_starts_decl_specifier_p (token->keyword);
}
static bool
token_is_decltype (cp_token *t)
{
return (t->keyword == RID_DECLTYPE
|| t->type == CPP_DECLTYPE);
}
static bool
cp_lexer_next_token_is_decltype (cp_lexer *lexer)
{
cp_token *t = cp_lexer_peek_token (lexer);
return token_is_decltype (t);
}
static tree
saved_checks_value (struct tree_check *check_value)
{
vec<deferred_access_check, va_gc> *checks;
deferred_access_check *chk;
checks = check_value->checks;
if (checks)
{
int i;
FOR_EACH_VEC_SAFE_ELT (checks, i, chk)
perform_or_defer_access_check (chk->binfo,
chk->decl,
chk->diag_decl, tf_warning_or_error);
}
return check_value->value;
}
static cp_token *
cp_lexer_peek_nth_token (cp_lexer* lexer, size_t n)
{
cp_token *token;
gcc_assert (n > 0);
if (cp_lexer_debugging_p (lexer))
fprintf (cp_lexer_debug_stream,
"cp_lexer: peeking ahead %ld at token: ", (long)n);
--n;
token = lexer->next_token;
gcc_assert (!n || token != &eof_token);
while (n != 0)
{
++token;
if (token == lexer->last_token)
{
token = &eof_token;
break;
}
if (!token->purged_p)
--n;
}
if (cp_lexer_debugging_p (lexer))
{
cp_lexer_print_token (cp_lexer_debug_stream, token);
putc ('\n', cp_lexer_debug_stream);
}
return token;
}
static cp_token *
cp_lexer_consume_token (cp_lexer* lexer)
{
cp_token *token = lexer->next_token;
gcc_assert (token != &eof_token);
gcc_assert (!lexer->in_pragma || token->type != CPP_PRAGMA_EOL);
do
{
lexer->next_token++;
if (lexer->next_token == lexer->last_token)
{
lexer->next_token = &eof_token;
break;
}
}
while (lexer->next_token->purged_p);
cp_lexer_set_source_position_from_token (token);
if (cp_lexer_debugging_p (lexer))
{
fputs ("cp_lexer: consuming token: ", cp_lexer_debug_stream);
cp_lexer_print_token (cp_lexer_debug_stream, token);
putc ('\n', cp_lexer_debug_stream);
}
return token;
}
static void
cp_lexer_purge_token (cp_lexer *lexer)
{
cp_token *tok = lexer->next_token;
gcc_assert (tok != &eof_token);
tok->purged_p = true;
tok->location = UNKNOWN_LOCATION;
tok->u.value = NULL_TREE;
tok->keyword = RID_MAX;
do
{
tok++;
if (tok == lexer->last_token)
{
tok = &eof_token;
break;
}
}
while (tok->purged_p);
lexer->next_token = tok;
}
static void
cp_lexer_purge_tokens_after (cp_lexer *lexer, cp_token *tok)
{
cp_token *peek = lexer->next_token;
if (peek == &eof_token)
peek = lexer->last_token;
gcc_assert (tok < peek);
for ( tok += 1; tok != peek; tok += 1)
{
tok->purged_p = true;
tok->location = UNKNOWN_LOCATION;
tok->u.value = NULL_TREE;
tok->keyword = RID_MAX;
}
}
static void
cp_lexer_save_tokens (cp_lexer* lexer)
{
if (cp_lexer_debugging_p (lexer))
fprintf (cp_lexer_debug_stream, "cp_lexer: saving tokens\n");
lexer->saved_tokens.safe_push (lexer->next_token);
}
static void
cp_lexer_commit_tokens (cp_lexer* lexer)
{
if (cp_lexer_debugging_p (lexer))
fprintf (cp_lexer_debug_stream, "cp_lexer: committing tokens\n");
lexer->saved_tokens.pop ();
}
static void
cp_lexer_rollback_tokens (cp_lexer* lexer)
{
if (cp_lexer_debugging_p (lexer))
fprintf (cp_lexer_debug_stream, "cp_lexer: restoring tokens\n");
lexer->next_token = lexer->saved_tokens.pop ();
}
struct saved_token_sentinel
{
cp_lexer *lexer;
unsigned len;
bool commit;
saved_token_sentinel(cp_lexer *lexer): lexer(lexer), commit(true)
{
len = lexer->saved_tokens.length ();
cp_lexer_save_tokens (lexer);
}
void rollback ()
{
cp_lexer_rollback_tokens (lexer);
commit = false;
}
~saved_token_sentinel()
{
if (commit)
cp_lexer_commit_tokens (lexer);
gcc_assert (lexer->saved_tokens.length () == len);
}
};
static void
cp_lexer_print_token (FILE * stream, cp_token *token)
{
static const char *const token_names[] = {
#define OP(e, s) #e,
#define TK(e, s) #e,
TTYPE_TABLE
#undef OP
#undef TK
"KEYWORD",
"TEMPLATE_ID",
"NESTED_NAME_SPECIFIER",
};
switch (token->type)
{
case CPP_KEYWORD:
if (!identifier_p (token->u.value))
break;
case CPP_NAME:
fputs (IDENTIFIER_POINTER (token->u.value), stream);
break;
case CPP_STRING:
case CPP_STRING16:
case CPP_STRING32:
case CPP_WSTRING:
case CPP_UTF8STRING:
fprintf (stream, " \"%s\"", TREE_STRING_POINTER (token->u.value));
break;
case CPP_NUMBER:
print_generic_expr (stream, token->u.value);
break;
default:
if (token->type < ARRAY_SIZE(token_names))
fputs (token_names[token->type], stream);
else
fprintf (stream, "[%d]", token->type);
break;
}
}
DEBUG_FUNCTION void
debug (cp_token &ref)
{
cp_lexer_print_token (stderr, &ref);
fprintf (stderr, "\n");
}
DEBUG_FUNCTION void
debug (cp_token *ptr)
{
if (ptr)
debug (*ptr);
else
fprintf (stderr, "<nil>\n");
}
static void
cp_lexer_start_debugging (cp_lexer* lexer)
{
if (!LEXER_DEBUGGING_ENABLED_P)
fatal_error (input_location,
"LEXER_DEBUGGING_ENABLED_P is not set to true");
lexer->debugging_p = true;
cp_lexer_debug_stream = stderr;
}
static void
cp_lexer_stop_debugging (cp_lexer* lexer)
{
if (!LEXER_DEBUGGING_ENABLED_P)
fatal_error (input_location,
"LEXER_DEBUGGING_ENABLED_P is not set to true");
lexer->debugging_p = false;
cp_lexer_debug_stream = NULL;
}
static cp_token_cache *
cp_token_cache_new (cp_token *first, cp_token *last)
{
cp_token_cache *cache = ggc_alloc<cp_token_cache> ();
cache->first = first;
cache->last = last;
return cache;
}
static inline void
cp_ensure_no_omp_declare_simd (cp_parser *parser)
{
if (parser->omp_declare_simd && !parser->omp_declare_simd->error_seen)
{
error ("%<#pragma omp declare simd%> not immediately followed by "
"function declaration or definition");
parser->omp_declare_simd = NULL;
}
}
static inline void
cp_finalize_omp_declare_simd (cp_parser *parser, tree fndecl)
{
if (__builtin_expect (parser->omp_declare_simd != NULL, 0))
{
if (fndecl == error_mark_node)
{
parser->omp_declare_simd = NULL;
return;
}
if (TREE_CODE (fndecl) != FUNCTION_DECL)
{
cp_ensure_no_omp_declare_simd (parser);
return;
}
}
}
static inline void
cp_ensure_no_oacc_routine (cp_parser *parser)
{
if (parser->oacc_routine && !parser->oacc_routine->error_seen)
{
error_at (parser->oacc_routine->loc,
"%<#pragma acc routine%> not immediately followed by "
"function declaration or definition");
parser->oacc_routine = NULL;
}
}

static void
clear_decl_specs (cp_decl_specifier_seq *decl_specs)
{
memset (decl_specs, 0, sizeof (cp_decl_specifier_seq));
}
static cp_declarator *make_call_declarator
(cp_declarator *, tree, cp_cv_quals, cp_virt_specifiers, cp_ref_qualifier, tree, tree, tree, tree);
static cp_declarator *make_array_declarator
(cp_declarator *, tree);
static cp_declarator *make_pointer_declarator
(cp_cv_quals, cp_declarator *, tree);
static cp_declarator *make_reference_declarator
(cp_cv_quals, cp_declarator *, bool, tree);
static cp_declarator *make_ptrmem_declarator
(cp_cv_quals, tree, cp_declarator *, tree);
static cp_declarator *cp_error_declarator;
static struct obstack declarator_obstack;
static inline void *
alloc_declarator (size_t bytes)
{
return obstack_alloc (&declarator_obstack, bytes);
}
static cp_declarator *
make_declarator (cp_declarator_kind kind)
{
cp_declarator *declarator;
declarator = (cp_declarator *) alloc_declarator (sizeof (cp_declarator));
declarator->kind = kind;
declarator->parenthesized = UNKNOWN_LOCATION;
declarator->attributes = NULL_TREE;
declarator->std_attributes = NULL_TREE;
declarator->declarator = NULL;
declarator->parameter_pack_p = false;
declarator->id_loc = UNKNOWN_LOCATION;
return declarator;
}
static cp_declarator *
make_id_declarator (tree qualifying_scope, tree unqualified_name,
special_function_kind sfk)
{
cp_declarator *declarator;
if (qualifying_scope && TYPE_P (qualifying_scope))
qualifying_scope = TYPE_MAIN_VARIANT (qualifying_scope);
gcc_assert (identifier_p (unqualified_name)
|| TREE_CODE (unqualified_name) == BIT_NOT_EXPR
|| TREE_CODE (unqualified_name) == TEMPLATE_ID_EXPR);
declarator = make_declarator (cdk_id);
declarator->u.id.qualifying_scope = qualifying_scope;
declarator->u.id.unqualified_name = unqualified_name;
declarator->u.id.sfk = sfk;
return declarator;
}
cp_declarator *
make_pointer_declarator (cp_cv_quals cv_qualifiers, cp_declarator *target,
tree attributes)
{
cp_declarator *declarator;
declarator = make_declarator (cdk_pointer);
declarator->declarator = target;
declarator->u.pointer.qualifiers = cv_qualifiers;
declarator->u.pointer.class_type = NULL_TREE;
if (target)
{
declarator->id_loc = target->id_loc;
declarator->parameter_pack_p = target->parameter_pack_p;
target->parameter_pack_p = false;
}
else
declarator->parameter_pack_p = false;
declarator->std_attributes = attributes;
return declarator;
}
cp_declarator *
make_reference_declarator (cp_cv_quals cv_qualifiers, cp_declarator *target,
bool rvalue_ref, tree attributes)
{
cp_declarator *declarator;
declarator = make_declarator (cdk_reference);
declarator->declarator = target;
declarator->u.reference.qualifiers = cv_qualifiers;
declarator->u.reference.rvalue_ref = rvalue_ref;
if (target)
{
declarator->id_loc = target->id_loc;
declarator->parameter_pack_p = target->parameter_pack_p;
target->parameter_pack_p = false;
}
else
declarator->parameter_pack_p = false;
declarator->std_attributes = attributes;
return declarator;
}
cp_declarator *
make_ptrmem_declarator (cp_cv_quals cv_qualifiers, tree class_type,
cp_declarator *pointee,
tree attributes)
{
cp_declarator *declarator;
declarator = make_declarator (cdk_ptrmem);
declarator->declarator = pointee;
declarator->u.pointer.qualifiers = cv_qualifiers;
declarator->u.pointer.class_type = class_type;
if (pointee)
{
declarator->parameter_pack_p = pointee->parameter_pack_p;
pointee->parameter_pack_p = false;
}
else
declarator->parameter_pack_p = false;
declarator->std_attributes = attributes;
return declarator;
}
cp_declarator *
make_call_declarator (cp_declarator *target,
tree parms,
cp_cv_quals cv_qualifiers,
cp_virt_specifiers virt_specifiers,
cp_ref_qualifier ref_qualifier,
tree tx_qualifier,
tree exception_specification,
tree late_return_type,
tree requires_clause)
{
cp_declarator *declarator;
declarator = make_declarator (cdk_function);
declarator->declarator = target;
declarator->u.function.parameters = parms;
declarator->u.function.qualifiers = cv_qualifiers;
declarator->u.function.virt_specifiers = virt_specifiers;
declarator->u.function.ref_qualifier = ref_qualifier;
declarator->u.function.tx_qualifier = tx_qualifier;
declarator->u.function.exception_specification = exception_specification;
declarator->u.function.late_return_type = late_return_type;
declarator->u.function.requires_clause = requires_clause;
if (target)
{
declarator->id_loc = target->id_loc;
declarator->parameter_pack_p = target->parameter_pack_p;
target->parameter_pack_p = false;
}
else
declarator->parameter_pack_p = false;
return declarator;
}
cp_declarator *
make_array_declarator (cp_declarator *element, tree bounds)
{
cp_declarator *declarator;
declarator = make_declarator (cdk_array);
declarator->declarator = element;
declarator->u.array.bounds = bounds;
if (element)
{
declarator->id_loc = element->id_loc;
declarator->parameter_pack_p = element->parameter_pack_p;
element->parameter_pack_p = false;
}
else
declarator->parameter_pack_p = false;
return declarator;
}
static bool 
declarator_can_be_parameter_pack (cp_declarator *declarator)
{
if (declarator && declarator->parameter_pack_p)
return false;
bool found = false;
while (declarator && !found)
{
switch ((int)declarator->kind)
{
case cdk_id:
case cdk_array:
case cdk_decomp:
found = true;
break;
case cdk_error:
return true;
default:
declarator = declarator->declarator;
break;
}
}
return !found;
}
cp_parameter_declarator *no_parameters;
cp_parameter_declarator *
make_parameter_declarator (cp_decl_specifier_seq *decl_specifiers,
cp_declarator *declarator,
tree default_argument,
location_t loc,
bool template_parameter_pack_p = false)
{
cp_parameter_declarator *parameter;
parameter = ((cp_parameter_declarator *)
alloc_declarator (sizeof (cp_parameter_declarator)));
parameter->next = NULL;
if (decl_specifiers)
parameter->decl_specifiers = *decl_specifiers;
else
clear_decl_specs (&parameter->decl_specifiers);
parameter->declarator = declarator;
parameter->default_argument = default_argument;
parameter->template_parameter_pack_p = template_parameter_pack_p;
parameter->loc = loc;
return parameter;
}
static bool
function_declarator_p (const cp_declarator *declarator)
{
while (declarator)
{
if (declarator->kind == cdk_function
&& declarator->declarator->kind == cdk_id)
return true;
if (declarator->kind == cdk_id
|| declarator->kind == cdk_decomp
|| declarator->kind == cdk_error)
return false;
declarator = declarator->declarator;
}
return false;
}
enum
{
CP_PARSER_FLAGS_NONE = 0x0,
CP_PARSER_FLAGS_OPTIONAL = 0x1,
CP_PARSER_FLAGS_NO_USER_DEFINED_TYPES = 0x2,
CP_PARSER_FLAGS_NO_TYPE_DEFINITIONS = 0x4,
CP_PARSER_FLAGS_ONLY_TYPE_OR_CONSTEXPR = 0x8,
CP_PARSER_FLAGS_ONLY_MUTABLE_OR_CONSTEXPR = 0x10
};
typedef int cp_parser_flags;
enum cp_parser_declarator_kind
{
CP_PARSER_DECLARATOR_ABSTRACT,
CP_PARSER_DECLARATOR_NAMED,
CP_PARSER_DECLARATOR_EITHER
};
enum cp_parser_prec
{
PREC_NOT_OPERATOR,
PREC_LOGICAL_OR_EXPRESSION,
PREC_LOGICAL_AND_EXPRESSION,
PREC_INCLUSIVE_OR_EXPRESSION,
PREC_EXCLUSIVE_OR_EXPRESSION,
PREC_AND_EXPRESSION,
PREC_EQUALITY_EXPRESSION,
PREC_RELATIONAL_EXPRESSION,
PREC_SHIFT_EXPRESSION,
PREC_ADDITIVE_EXPRESSION,
PREC_MULTIPLICATIVE_EXPRESSION,
PREC_PM_EXPRESSION,
NUM_PREC_VALUES = PREC_PM_EXPRESSION
};
struct cp_parser_binary_operations_map_node
{
enum cpp_ttype token_type;
enum tree_code tree_type;
enum cp_parser_prec prec;
};
struct cp_parser_expression_stack_entry
{
cp_expr lhs;
enum tree_code lhs_type;
enum tree_code tree_type;
enum cp_parser_prec prec;
location_t loc;
};
typedef struct cp_parser_expression_stack_entry
cp_parser_expression_stack[NUM_PREC_VALUES];
static cp_parser_context *cp_parser_context_new
(cp_parser_context *);
static GTY((deletable)) cp_parser_context* cp_parser_context_free_list;
static const cp_parser_binary_operations_map_node binops[] = {
{ CPP_DEREF_STAR, MEMBER_REF, PREC_PM_EXPRESSION },
{ CPP_DOT_STAR, DOTSTAR_EXPR, PREC_PM_EXPRESSION },
{ CPP_MULT, MULT_EXPR, PREC_MULTIPLICATIVE_EXPRESSION },
{ CPP_DIV, TRUNC_DIV_EXPR, PREC_MULTIPLICATIVE_EXPRESSION },
{ CPP_MOD, TRUNC_MOD_EXPR, PREC_MULTIPLICATIVE_EXPRESSION },
{ CPP_PLUS, PLUS_EXPR, PREC_ADDITIVE_EXPRESSION },
{ CPP_MINUS, MINUS_EXPR, PREC_ADDITIVE_EXPRESSION },
{ CPP_LSHIFT, LSHIFT_EXPR, PREC_SHIFT_EXPRESSION },
{ CPP_RSHIFT, RSHIFT_EXPR, PREC_SHIFT_EXPRESSION },
{ CPP_LESS, LT_EXPR, PREC_RELATIONAL_EXPRESSION },
{ CPP_GREATER, GT_EXPR, PREC_RELATIONAL_EXPRESSION },
{ CPP_LESS_EQ, LE_EXPR, PREC_RELATIONAL_EXPRESSION },
{ CPP_GREATER_EQ, GE_EXPR, PREC_RELATIONAL_EXPRESSION },
{ CPP_EQ_EQ, EQ_EXPR, PREC_EQUALITY_EXPRESSION },
{ CPP_NOT_EQ, NE_EXPR, PREC_EQUALITY_EXPRESSION },
{ CPP_AND, BIT_AND_EXPR, PREC_AND_EXPRESSION },
{ CPP_XOR, BIT_XOR_EXPR, PREC_EXCLUSIVE_OR_EXPRESSION },
{ CPP_OR, BIT_IOR_EXPR, PREC_INCLUSIVE_OR_EXPRESSION },
{ CPP_AND_AND, TRUTH_ANDIF_EXPR, PREC_LOGICAL_AND_EXPRESSION },
{ CPP_OR_OR, TRUTH_ORIF_EXPR, PREC_LOGICAL_OR_EXPRESSION }
};
static cp_parser_binary_operations_map_node binops_by_token[N_CP_TTYPES];
static cp_parser_context *
cp_parser_context_new (cp_parser_context* next)
{
cp_parser_context *context;
if (cp_parser_context_free_list != NULL)
{
context = cp_parser_context_free_list;
cp_parser_context_free_list = context->next;
memset (context, 0, sizeof (*context));
}
else
context = ggc_cleared_alloc<cp_parser_context> ();
context->status = CP_PARSER_STATUS_KIND_NO_ERROR;
if (next)
{
context->object_type = next->object_type;
context->next = next;
}
return context;
}
#define unparsed_funs_with_default_args \
parser->unparsed_queues->last ().funs_with_default_args
#define unparsed_funs_with_definitions \
parser->unparsed_queues->last ().funs_with_definitions
#define unparsed_nsdmis \
parser->unparsed_queues->last ().nsdmis
#define unparsed_classes \
parser->unparsed_queues->last ().classes
static void
push_unparsed_function_queues (cp_parser *parser)
{
cp_unparsed_functions_entry e = {NULL, make_tree_vector (), NULL, NULL};
vec_safe_push (parser->unparsed_queues, e);
}
static void
pop_unparsed_function_queues (cp_parser *parser)
{
release_tree_vector (unparsed_funs_with_definitions);
parser->unparsed_queues->pop ();
}
static cp_parser *cp_parser_new
(void);
static cp_expr cp_parser_identifier
(cp_parser *);
static cp_expr cp_parser_string_literal
(cp_parser *, bool, bool, bool);
static cp_expr cp_parser_userdef_char_literal
(cp_parser *);
static tree cp_parser_userdef_string_literal
(tree);
static cp_expr cp_parser_userdef_numeric_literal
(cp_parser *);
static bool cp_parser_translation_unit
(cp_parser *);
static cp_expr cp_parser_primary_expression
(cp_parser *, bool, bool, bool, cp_id_kind *);
static cp_expr cp_parser_id_expression
(cp_parser *, bool, bool, bool *, bool, bool);
static cp_expr cp_parser_unqualified_id
(cp_parser *, bool, bool, bool, bool);
static tree cp_parser_nested_name_specifier_opt
(cp_parser *, bool, bool, bool, bool, bool = false);
static tree cp_parser_nested_name_specifier
(cp_parser *, bool, bool, bool, bool);
static tree cp_parser_qualifying_entity
(cp_parser *, bool, bool, bool, bool, bool);
static cp_expr cp_parser_postfix_expression
(cp_parser *, bool, bool, bool, bool, cp_id_kind *);
static tree cp_parser_postfix_open_square_expression
(cp_parser *, tree, bool, bool);
static tree cp_parser_postfix_dot_deref_expression
(cp_parser *, enum cpp_ttype, cp_expr, bool, cp_id_kind *, location_t);
static vec<tree, va_gc> *cp_parser_parenthesized_expression_list
(cp_parser *, int, bool, bool, bool *, location_t * = NULL,
bool = false);
enum { non_attr = 0, normal_attr = 1, id_attr = 2 };
static void cp_parser_pseudo_destructor_name
(cp_parser *, tree, tree *, tree *);
static cp_expr cp_parser_unary_expression
(cp_parser *, cp_id_kind * = NULL, bool = false, bool = false, bool = false);
static enum tree_code cp_parser_unary_operator
(cp_token *);
static tree cp_parser_new_expression
(cp_parser *);
static vec<tree, va_gc> *cp_parser_new_placement
(cp_parser *);
static tree cp_parser_new_type_id
(cp_parser *, tree *);
static cp_declarator *cp_parser_new_declarator_opt
(cp_parser *);
static cp_declarator *cp_parser_direct_new_declarator
(cp_parser *);
static vec<tree, va_gc> *cp_parser_new_initializer
(cp_parser *);
static tree cp_parser_delete_expression
(cp_parser *);
static cp_expr cp_parser_cast_expression
(cp_parser *, bool, bool, bool, cp_id_kind *);
static cp_expr cp_parser_binary_expression
(cp_parser *, bool, bool, enum cp_parser_prec, cp_id_kind *);
static tree cp_parser_question_colon_clause
(cp_parser *, cp_expr);
static cp_expr cp_parser_assignment_expression
(cp_parser *, cp_id_kind * = NULL, bool = false, bool = false);
static enum tree_code cp_parser_assignment_operator_opt
(cp_parser *);
static cp_expr cp_parser_expression
(cp_parser *, cp_id_kind * = NULL, bool = false, bool = false);
static cp_expr cp_parser_constant_expression
(cp_parser *, bool = false, bool * = NULL, bool = false);
static cp_expr cp_parser_builtin_offsetof
(cp_parser *);
static cp_expr cp_parser_lambda_expression
(cp_parser *);
static void cp_parser_lambda_introducer
(cp_parser *, tree);
static bool cp_parser_lambda_declarator_opt
(cp_parser *, tree);
static void cp_parser_lambda_body
(cp_parser *, tree);
static void cp_parser_statement
(cp_parser *, tree, bool, bool *, vec<tree> * = NULL, location_t * = NULL);
static void cp_parser_label_for_labeled_statement
(cp_parser *, tree);
static tree cp_parser_expression_statement
(cp_parser *, tree);
static tree cp_parser_compound_statement
(cp_parser *, tree, int, bool);
static void cp_parser_statement_seq_opt
(cp_parser *, tree);
static tree cp_parser_selection_statement
(cp_parser *, bool *, vec<tree> *);
static tree cp_parser_condition
(cp_parser *);
static tree cp_parser_iteration_statement
(cp_parser *, bool *, bool, unsigned short);
static bool cp_parser_init_statement
(cp_parser *, tree *decl);
static tree cp_parser_for
(cp_parser *, bool, unsigned short);
static tree cp_parser_c_for
(cp_parser *, tree, tree, bool, unsigned short);
static tree cp_parser_range_for
(cp_parser *, tree, tree, tree, bool, unsigned short);
static void do_range_for_auto_deduction
(tree, tree);
static tree cp_parser_perform_range_for_lookup
(tree, tree *, tree *);
static tree cp_parser_range_for_member_function
(tree, tree);
static tree cp_parser_jump_statement
(cp_parser *);
static void cp_parser_declaration_statement
(cp_parser *);
static tree cp_parser_implicitly_scoped_statement
(cp_parser *, bool *, const token_indent_info &, vec<tree> * = NULL);
static void cp_parser_already_scoped_statement
(cp_parser *, bool *, const token_indent_info &);
static void cp_parser_declaration_seq_opt
(cp_parser *);
static void cp_parser_declaration
(cp_parser *);
static void cp_parser_block_declaration
(cp_parser *, bool);
static void cp_parser_simple_declaration
(cp_parser *, bool, tree *);
static void cp_parser_decl_specifier_seq
(cp_parser *, cp_parser_flags, cp_decl_specifier_seq *, int *);
static tree cp_parser_storage_class_specifier_opt
(cp_parser *);
static tree cp_parser_function_specifier_opt
(cp_parser *, cp_decl_specifier_seq *);
static tree cp_parser_type_specifier
(cp_parser *, cp_parser_flags, cp_decl_specifier_seq *, bool,
int *, bool *);
static tree cp_parser_simple_type_specifier
(cp_parser *, cp_decl_specifier_seq *, cp_parser_flags);
static tree cp_parser_type_name
(cp_parser *, bool);
static tree cp_parser_type_name
(cp_parser *);
static tree cp_parser_nonclass_name 
(cp_parser* parser);
static tree cp_parser_elaborated_type_specifier
(cp_parser *, bool, bool);
static tree cp_parser_enum_specifier
(cp_parser *);
static void cp_parser_enumerator_list
(cp_parser *, tree);
static void cp_parser_enumerator_definition
(cp_parser *, tree);
static tree cp_parser_namespace_name
(cp_parser *);
static void cp_parser_namespace_definition
(cp_parser *);
static void cp_parser_namespace_body
(cp_parser *);
static tree cp_parser_qualified_namespace_specifier
(cp_parser *);
static void cp_parser_namespace_alias_definition
(cp_parser *);
static bool cp_parser_using_declaration
(cp_parser *, bool);
static void cp_parser_using_directive
(cp_parser *);
static tree cp_parser_alias_declaration
(cp_parser *);
static void cp_parser_asm_definition
(cp_parser *);
static void cp_parser_linkage_specification
(cp_parser *);
static void cp_parser_static_assert
(cp_parser *, bool);
static tree cp_parser_decltype
(cp_parser *);
static tree cp_parser_decomposition_declaration
(cp_parser *, cp_decl_specifier_seq *, tree *, location_t *);
static tree cp_parser_init_declarator
(cp_parser *, cp_decl_specifier_seq *, vec<deferred_access_check, va_gc> *,
bool, bool, int, bool *, tree *, location_t *, tree *);
static cp_declarator *cp_parser_declarator
(cp_parser *, cp_parser_declarator_kind, int *, bool *, bool, bool);
static cp_declarator *cp_parser_direct_declarator
(cp_parser *, cp_parser_declarator_kind, int *, bool, bool);
static enum tree_code cp_parser_ptr_operator
(cp_parser *, tree *, cp_cv_quals *, tree *);
static cp_cv_quals cp_parser_cv_qualifier_seq_opt
(cp_parser *);
static cp_virt_specifiers cp_parser_virt_specifier_seq_opt
(cp_parser *);
static cp_ref_qualifier cp_parser_ref_qualifier_opt
(cp_parser *);
static tree cp_parser_tx_qualifier_opt
(cp_parser *);
static tree cp_parser_late_return_type_opt
(cp_parser *, cp_declarator *, tree &, cp_cv_quals);
static tree cp_parser_declarator_id
(cp_parser *, bool);
static tree cp_parser_type_id
(cp_parser *);
static tree cp_parser_template_type_arg
(cp_parser *);
static tree cp_parser_trailing_type_id (cp_parser *);
static tree cp_parser_type_id_1
(cp_parser *, bool, bool);
static void cp_parser_type_specifier_seq
(cp_parser *, bool, bool, cp_decl_specifier_seq *);
static tree cp_parser_parameter_declaration_clause
(cp_parser *);
static tree cp_parser_parameter_declaration_list
(cp_parser *, bool *);
static cp_parameter_declarator *cp_parser_parameter_declaration
(cp_parser *, bool, bool *);
static tree cp_parser_default_argument 
(cp_parser *, bool);
static void cp_parser_function_body
(cp_parser *, bool);
static tree cp_parser_initializer
(cp_parser *, bool *, bool *, bool = false);
static cp_expr cp_parser_initializer_clause
(cp_parser *, bool *);
static cp_expr cp_parser_braced_list
(cp_parser*, bool*);
static vec<constructor_elt, va_gc> *cp_parser_initializer_list
(cp_parser *, bool *);
static void cp_parser_ctor_initializer_opt_and_function_body
(cp_parser *, bool);
static tree cp_parser_late_parsing_omp_declare_simd
(cp_parser *, tree);
static tree cp_parser_late_parsing_oacc_routine
(cp_parser *, tree);
static tree synthesize_implicit_template_parm
(cp_parser *, tree);
static tree finish_fully_implicit_template
(cp_parser *, tree);
static void abort_fully_implicit_template
(cp_parser *);
static tree cp_parser_class_name
(cp_parser *, bool, bool, enum tag_types, bool, bool, bool, bool = false);
static tree cp_parser_class_specifier
(cp_parser *);
static tree cp_parser_class_head
(cp_parser *, bool *);
static enum tag_types cp_parser_class_key
(cp_parser *);
static void cp_parser_type_parameter_key
(cp_parser* parser);
static void cp_parser_member_specification_opt
(cp_parser *);
static void cp_parser_member_declaration
(cp_parser *);
static tree cp_parser_pure_specifier
(cp_parser *);
static tree cp_parser_constant_initializer
(cp_parser *);
static tree cp_parser_base_clause
(cp_parser *);
static tree cp_parser_base_specifier
(cp_parser *);
static tree cp_parser_conversion_function_id
(cp_parser *);
static tree cp_parser_conversion_type_id
(cp_parser *);
static cp_declarator *cp_parser_conversion_declarator_opt
(cp_parser *);
static void cp_parser_ctor_initializer_opt
(cp_parser *);
static void cp_parser_mem_initializer_list
(cp_parser *);
static tree cp_parser_mem_initializer
(cp_parser *);
static tree cp_parser_mem_initializer_id
(cp_parser *);
static cp_expr cp_parser_operator_function_id
(cp_parser *);
static cp_expr cp_parser_operator
(cp_parser *);
static void cp_parser_template_declaration
(cp_parser *, bool);
static tree cp_parser_template_parameter_list
(cp_parser *);
static tree cp_parser_template_parameter
(cp_parser *, bool *, bool *);
static tree cp_parser_type_parameter
(cp_parser *, bool *);
static tree cp_parser_template_id
(cp_parser *, bool, bool, enum tag_types, bool);
static tree cp_parser_template_name
(cp_parser *, bool, bool, bool, enum tag_types, bool *);
static tree cp_parser_template_argument_list
(cp_parser *);
static tree cp_parser_template_argument
(cp_parser *);
static void cp_parser_explicit_instantiation
(cp_parser *);
static void cp_parser_explicit_specialization
(cp_parser *);
static tree cp_parser_try_block
(cp_parser *);
static void cp_parser_function_try_block
(cp_parser *);
static void cp_parser_handler_seq
(cp_parser *);
static void cp_parser_handler
(cp_parser *);
static tree cp_parser_exception_declaration
(cp_parser *);
static tree cp_parser_throw_expression
(cp_parser *);
static tree cp_parser_exception_specification_opt
(cp_parser *);
static tree cp_parser_type_id_list
(cp_parser *);
static tree cp_parser_asm_specification_opt
(cp_parser *);
static tree cp_parser_asm_operand_list
(cp_parser *);
static tree cp_parser_asm_clobber_list
(cp_parser *);
static tree cp_parser_asm_label_list
(cp_parser *);
static bool cp_next_tokens_can_be_attribute_p
(cp_parser *);
static bool cp_next_tokens_can_be_gnu_attribute_p
(cp_parser *);
static bool cp_next_tokens_can_be_std_attribute_p
(cp_parser *);
static bool cp_nth_tokens_can_be_std_attribute_p
(cp_parser *, size_t);
static bool cp_nth_tokens_can_be_gnu_attribute_p
(cp_parser *, size_t);
static bool cp_nth_tokens_can_be_attribute_p
(cp_parser *, size_t);
static tree cp_parser_attributes_opt
(cp_parser *);
static tree cp_parser_gnu_attributes_opt
(cp_parser *);
static tree cp_parser_gnu_attribute_list
(cp_parser *);
static tree cp_parser_std_attribute
(cp_parser *, tree);
static tree cp_parser_std_attribute_spec
(cp_parser *);
static tree cp_parser_std_attribute_spec_seq
(cp_parser *);
static size_t cp_parser_skip_attributes_opt
(cp_parser *, size_t);
static bool cp_parser_extension_opt
(cp_parser *, int *);
static void cp_parser_label_declaration
(cp_parser *);
static tree cp_parser_requires_clause
(cp_parser *);
static tree cp_parser_requires_clause_opt
(cp_parser *);
static tree cp_parser_requires_expression
(cp_parser *);
static tree cp_parser_requirement_parameter_list
(cp_parser *);
static tree cp_parser_requirement_body
(cp_parser *);
static tree cp_parser_requirement_list
(cp_parser *);
static tree cp_parser_requirement
(cp_parser *);
static tree cp_parser_simple_requirement
(cp_parser *);
static tree cp_parser_compound_requirement
(cp_parser *);
static tree cp_parser_type_requirement
(cp_parser *);
static tree cp_parser_nested_requirement
(cp_parser *);
static tree cp_parser_transaction
(cp_parser *, cp_token *);
static tree cp_parser_transaction_expression
(cp_parser *, enum rid);
static void cp_parser_function_transaction
(cp_parser *, enum rid);
static tree cp_parser_transaction_cancel
(cp_parser *);
enum pragma_context {
pragma_external,
pragma_member,
pragma_objc_icode,
pragma_stmt,
pragma_compound
};
static bool cp_parser_pragma
(cp_parser *, enum pragma_context, bool *);
static tree cp_parser_objc_message_receiver
(cp_parser *);
static tree cp_parser_objc_message_args
(cp_parser *);
static tree cp_parser_objc_message_expression
(cp_parser *);
static cp_expr cp_parser_objc_encode_expression
(cp_parser *);
static tree cp_parser_objc_defs_expression
(cp_parser *);
static tree cp_parser_objc_protocol_expression
(cp_parser *);
static tree cp_parser_objc_selector_expression
(cp_parser *);
static cp_expr cp_parser_objc_expression
(cp_parser *);
static bool cp_parser_objc_selector_p
(enum cpp_ttype);
static tree cp_parser_objc_selector
(cp_parser *);
static tree cp_parser_objc_protocol_refs_opt
(cp_parser *);
static void cp_parser_objc_declaration
(cp_parser *, tree);
static tree cp_parser_objc_statement
(cp_parser *);
static bool cp_parser_objc_valid_prefix_attributes
(cp_parser *, tree *);
static void cp_parser_objc_at_property_declaration 
(cp_parser *) ;
static void cp_parser_objc_at_synthesize_declaration 
(cp_parser *) ;
static void cp_parser_objc_at_dynamic_declaration
(cp_parser *) ;
static tree cp_parser_objc_struct_declaration
(cp_parser *) ;
static cp_expr cp_parser_lookup_name
(cp_parser *, tree, enum tag_types, bool, bool, bool, tree *, location_t);
static tree cp_parser_lookup_name_simple
(cp_parser *, tree, location_t);
static tree cp_parser_maybe_treat_template_as_class
(tree, bool);
static bool cp_parser_check_declarator_template_parameters
(cp_parser *, cp_declarator *, location_t);
static bool cp_parser_check_template_parameters
(cp_parser *, unsigned, bool, location_t, cp_declarator *);
static cp_expr cp_parser_simple_cast_expression
(cp_parser *);
static tree cp_parser_global_scope_opt
(cp_parser *, bool);
static bool cp_parser_constructor_declarator_p
(cp_parser *, bool);
static tree cp_parser_function_definition_from_specifiers_and_declarator
(cp_parser *, cp_decl_specifier_seq *, tree, const cp_declarator *);
static tree cp_parser_function_definition_after_declarator
(cp_parser *, bool);
static bool cp_parser_template_declaration_after_export
(cp_parser *, bool);
static void cp_parser_perform_template_parameter_access_checks
(vec<deferred_access_check, va_gc> *);
static tree cp_parser_single_declaration
(cp_parser *, vec<deferred_access_check, va_gc> *, bool, bool, bool *);
static cp_expr cp_parser_functional_cast
(cp_parser *, tree);
static tree cp_parser_save_member_function_body
(cp_parser *, cp_decl_specifier_seq *, cp_declarator *, tree);
static tree cp_parser_save_nsdmi
(cp_parser *);
static tree cp_parser_enclosed_template_argument_list
(cp_parser *);
static void cp_parser_save_default_args
(cp_parser *, tree);
static void cp_parser_late_parsing_for_member
(cp_parser *, tree);
static tree cp_parser_late_parse_one_default_arg
(cp_parser *, tree, tree, tree);
static void cp_parser_late_parsing_nsdmi
(cp_parser *, tree);
static void cp_parser_late_parsing_default_args
(cp_parser *, tree);
static tree cp_parser_sizeof_operand
(cp_parser *, enum rid);
static cp_expr cp_parser_trait_expr
(cp_parser *, enum rid);
static bool cp_parser_declares_only_class_p
(cp_parser *);
static void cp_parser_set_storage_class
(cp_parser *, cp_decl_specifier_seq *, enum rid, cp_token *);
static void cp_parser_set_decl_spec_type
(cp_decl_specifier_seq *, tree, cp_token *, bool);
static void set_and_check_decl_spec_loc
(cp_decl_specifier_seq *decl_specs,
cp_decl_spec ds, cp_token *);
static bool cp_parser_friend_p
(const cp_decl_specifier_seq *);
static void cp_parser_required_error
(cp_parser *, required_token, bool, location_t);
static cp_token *cp_parser_require
(cp_parser *, enum cpp_ttype, required_token, location_t = UNKNOWN_LOCATION);
static cp_token *cp_parser_require_keyword
(cp_parser *, enum rid, required_token);
static bool cp_parser_token_starts_function_definition_p
(cp_token *);
static bool cp_parser_next_token_starts_class_definition_p
(cp_parser *);
static bool cp_parser_next_token_ends_template_argument_p
(cp_parser *);
static bool cp_parser_nth_token_starts_template_argument_list_p
(cp_parser *, size_t);
static enum tag_types cp_parser_token_is_class_key
(cp_token *);
static enum tag_types cp_parser_token_is_type_parameter_key
(cp_token *);
static void cp_parser_check_class_key
(enum tag_types, tree type);
static void cp_parser_check_access_in_redeclaration
(tree type, location_t location);
static bool cp_parser_optional_template_keyword
(cp_parser *);
static void cp_parser_pre_parsed_nested_name_specifier
(cp_parser *);
static bool cp_parser_cache_group
(cp_parser *, enum cpp_ttype, unsigned);
static tree cp_parser_cache_defarg
(cp_parser *parser, bool nsdmi);
static void cp_parser_parse_tentatively
(cp_parser *);
static void cp_parser_commit_to_tentative_parse
(cp_parser *);
static void cp_parser_commit_to_topmost_tentative_parse
(cp_parser *);
static void cp_parser_abort_tentative_parse
(cp_parser *);
static bool cp_parser_parse_definitely
(cp_parser *);
static inline bool cp_parser_parsing_tentatively
(cp_parser *);
static bool cp_parser_uncommitted_to_tentative_parse_p
(cp_parser *);
static void cp_parser_error
(cp_parser *, const char *);
static void cp_parser_name_lookup_error
(cp_parser *, tree, tree, name_lookup_error, location_t);
static bool cp_parser_simulate_error
(cp_parser *);
static bool cp_parser_check_type_definition
(cp_parser *);
static void cp_parser_check_for_definition_in_return_type
(cp_declarator *, tree, location_t type_location);
static void cp_parser_check_for_invalid_template_id
(cp_parser *, tree, enum tag_types, location_t location);
static bool cp_parser_non_integral_constant_expression
(cp_parser *, non_integral_constant);
static void cp_parser_diagnose_invalid_type_name
(cp_parser *, tree, location_t);
static bool cp_parser_parse_and_diagnose_invalid_type_name
(cp_parser *);
static int cp_parser_skip_to_closing_parenthesis
(cp_parser *, bool, bool, bool);
static void cp_parser_skip_to_end_of_statement
(cp_parser *);
static void cp_parser_consume_semicolon_at_end_of_statement
(cp_parser *);
static void cp_parser_skip_to_end_of_block_or_statement
(cp_parser *);
static bool cp_parser_skip_to_closing_brace
(cp_parser *);
static void cp_parser_skip_to_end_of_template_parameter_list
(cp_parser *);
static void cp_parser_skip_to_pragma_eol
(cp_parser*, cp_token *);
static bool cp_parser_error_occurred
(cp_parser *);
static bool cp_parser_allow_gnu_extensions_p
(cp_parser *);
static bool cp_parser_is_pure_string_literal
(cp_token *);
static bool cp_parser_is_string_literal
(cp_token *);
static bool cp_parser_is_keyword
(cp_token *, enum rid);
static tree cp_parser_make_typename_type
(cp_parser *, tree, location_t location);
static cp_declarator * cp_parser_make_indirect_declarator
(enum tree_code, tree, cp_cv_quals, cp_declarator *, tree);
static bool cp_parser_compound_literal_p
(cp_parser *);
static bool cp_parser_array_designator_p
(cp_parser *);
static bool cp_parser_init_statement_p
(cp_parser *);
static bool cp_parser_skip_to_closing_square_bracket
(cp_parser *);
static tree cp_parser_maybe_concept_name       (cp_parser *, tree);
static tree cp_parser_maybe_partial_concept_id (cp_parser *, tree, tree);
cp_unevaluated::cp_unevaluated ()
{
++cp_unevaluated_operand;
++c_inhibit_evaluation_warnings;
}
cp_unevaluated::~cp_unevaluated ()
{
--c_inhibit_evaluation_warnings;
--cp_unevaluated_operand;
}
static inline bool
cp_parser_parsing_tentatively (cp_parser* parser)
{
return parser->context->next != NULL;
}
static bool
cp_parser_is_pure_string_literal (cp_token* token)
{
return (token->type == CPP_STRING ||
token->type == CPP_STRING16 ||
token->type == CPP_STRING32 ||
token->type == CPP_WSTRING ||
token->type == CPP_UTF8STRING);
}
static bool
cp_parser_is_string_literal (cp_token* token)
{
return (cp_parser_is_pure_string_literal (token) ||
token->type == CPP_STRING_USERDEF ||
token->type == CPP_STRING16_USERDEF ||
token->type == CPP_STRING32_USERDEF ||
token->type == CPP_WSTRING_USERDEF ||
token->type == CPP_UTF8STRING_USERDEF);
}
static bool
cp_parser_is_keyword (cp_token* token, enum rid keyword)
{
return token->keyword == keyword;
}
static enum pragma_kind
cp_parser_pragma_kind (cp_token *token)
{
if (token->type != CPP_PRAGMA)
return PRAGMA_NONE;
return (enum pragma_kind) TREE_INT_CST_LOW (token->u.value);
}
static bool
cp_lexer_peek_conflict_marker (cp_lexer *lexer, enum cpp_ttype tok1_kind,
location_t *out_loc)
{
cp_token *token2 = cp_lexer_peek_nth_token (lexer, 2);
if (token2->type != tok1_kind)
return false;
cp_token *token3 = cp_lexer_peek_nth_token (lexer, 3);
if (token3->type != tok1_kind)
return false;
cp_token *token4 = cp_lexer_peek_nth_token (lexer, 4);
if (token4->type != conflict_marker_get_final_tok_kind (tok1_kind))
return false;
location_t start_loc = cp_lexer_peek_token (lexer)->location;
if (LOCATION_COLUMN (start_loc) != 1)
return false;
location_t finish_loc = get_finish (token4->location);
*out_loc = make_location (start_loc, start_loc, finish_loc);
return true;
}
static const char *
get_matching_symbol (required_token token_desc)
{
switch (token_desc)
{
default:
gcc_unreachable ();
return "";
case RT_CLOSE_BRACE:
return "{";
case RT_CLOSE_PAREN:
return "(";
}
}
static enum cpp_ttype
get_required_cpp_ttype (required_token token_desc)
{
switch (token_desc)
{
case RT_SEMICOLON:
return CPP_SEMICOLON;
case RT_OPEN_PAREN:
return CPP_OPEN_PAREN;
case RT_CLOSE_BRACE:
return CPP_CLOSE_BRACE;
case RT_OPEN_BRACE:
return CPP_OPEN_BRACE;
case RT_CLOSE_SQUARE:
return CPP_CLOSE_SQUARE;
case RT_OPEN_SQUARE:
return CPP_OPEN_SQUARE;
case RT_COMMA:
return CPP_COMMA;
case RT_COLON:
return CPP_COLON;
case RT_CLOSE_PAREN:
return CPP_CLOSE_PAREN;
default:
return CPP_EOF;
}
}
static void
cp_parser_error_1 (cp_parser* parser, const char* gmsgid,
required_token missing_token_desc,
location_t matching_location)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
cp_lexer_set_source_position_from_token (token);
if (token->type == CPP_PRAGMA)
{
error_at (token->location,
"%<#pragma%> is not allowed here");
cp_parser_skip_to_pragma_eol (parser, token);
return;
}
if (token->type == CPP_LSHIFT
|| token->type == CPP_RSHIFT
|| token->type == CPP_EQ_EQ)
{
location_t loc;
if (cp_lexer_peek_conflict_marker (parser->lexer, token->type, &loc))
{
error_at (loc, "version control conflict marker in file");
return;
}
}
gcc_rich_location richloc (input_location);
bool added_matching_location = false;
if (missing_token_desc != RT_NONE)
{
enum cpp_ttype ttype = get_required_cpp_ttype (missing_token_desc);
location_t prev_token_loc
= cp_lexer_previous_token (parser->lexer)->location;
maybe_suggest_missing_token_insertion (&richloc, ttype, prev_token_loc);
if (matching_location != UNKNOWN_LOCATION)
added_matching_location
= richloc.add_location_if_nearby (matching_location);
}
c_parse_error (gmsgid,
(token->type == CPP_KEYWORD ? CPP_NAME : token->type),
token->u.value, token->flags, &richloc);
if (missing_token_desc != RT_NONE)
{
if (matching_location != UNKNOWN_LOCATION
&& !added_matching_location)
inform (matching_location, "to match this %qs",
get_matching_symbol (missing_token_desc));
}
}
static void
cp_parser_error (cp_parser* parser, const char* gmsgid)
{
if (!cp_parser_simulate_error (parser))
cp_parser_error_1 (parser, gmsgid, RT_NONE, UNKNOWN_LOCATION);
}
static void
cp_parser_name_lookup_error (cp_parser* parser,
tree name,
tree decl,
name_lookup_error desired,
location_t location)
{
if (decl == error_mark_node)
{
if (parser->scope && parser->scope != global_namespace)
error_at (location, "%<%E::%E%> has not been declared",
parser->scope, name);
else if (parser->scope == global_namespace)
error_at (location, "%<::%E%> has not been declared", name);
else if (parser->object_scope
&& !CLASS_TYPE_P (parser->object_scope))
error_at (location, "request for member %qE in non-class type %qT",
name, parser->object_scope);
else if (parser->object_scope)
error_at (location, "%<%T::%E%> has not been declared",
parser->object_scope, name);
else
error_at (location, "%qE has not been declared", name);
}
else if (parser->scope && parser->scope != global_namespace)
{
switch (desired)
{
case NLE_TYPE:
error_at (location, "%<%E::%E%> is not a type",
parser->scope, name);
break;
case NLE_CXX98:
error_at (location, "%<%E::%E%> is not a class or namespace",
parser->scope, name);
break;
case NLE_NOT_CXX98:
error_at (location,
"%<%E::%E%> is not a class, namespace, or enumeration",
parser->scope, name);
break;
default:
gcc_unreachable ();
}
}
else if (parser->scope == global_namespace)
{
switch (desired)
{
case NLE_TYPE:
error_at (location, "%<::%E%> is not a type", name);
break;
case NLE_CXX98:
error_at (location, "%<::%E%> is not a class or namespace", name);
break;
case NLE_NOT_CXX98:
error_at (location,
"%<::%E%> is not a class, namespace, or enumeration",
name);
break;
default:
gcc_unreachable ();
}
}
else
{
switch (desired)
{
case NLE_TYPE:
error_at (location, "%qE is not a type", name);
break;
case NLE_CXX98:
error_at (location, "%qE is not a class or namespace", name);
break;
case NLE_NOT_CXX98:
error_at (location,
"%qE is not a class, namespace, or enumeration", name);
break;
default:
gcc_unreachable ();
}
}
}
static bool
cp_parser_simulate_error (cp_parser* parser)
{
if (cp_parser_uncommitted_to_tentative_parse_p (parser))
{
parser->context->status = CP_PARSER_STATUS_KIND_ERROR;
return true;
}
return false;
}
static bool
cp_parser_check_type_definition (cp_parser* parser)
{
if (parser->type_definition_forbidden_message)
{
error (parser->type_definition_forbidden_message);
return false;
}
return true;
}
static void
cp_parser_check_for_definition_in_return_type (cp_declarator *declarator,
tree type, location_t type_location)
{
while (declarator
&& (declarator->kind == cdk_pointer
|| declarator->kind == cdk_reference
|| declarator->kind == cdk_ptrmem))
declarator = declarator->declarator;
if (declarator
&& declarator->kind == cdk_function)
{
error_at (type_location,
"new types may not be defined in a return type");
inform (type_location, 
"(perhaps a semicolon is missing after the definition of %qT)",
type);
}
}
static void
cp_parser_check_for_invalid_template_id (cp_parser* parser,
tree type,
enum tag_types tag_type,
location_t location)
{
cp_token_position start = 0;
if (cp_lexer_next_token_is (parser->lexer, CPP_LESS))
{
if (TREE_CODE (type) == TYPE_DECL)
type = TREE_TYPE (type);
if (TYPE_P (type) && !template_placeholder_p (type))
error_at (location, "%qT is not a template", type);
else if (identifier_p (type))
{
if (tag_type != none_type)
error_at (location, "%qE is not a class template", type);
else
error_at (location, "%qE is not a template", type);
}
else
error_at (location, "invalid template-id");
if (cp_parser_uncommitted_to_tentative_parse_p (parser))
start = cp_lexer_token_position (parser->lexer, true);
cp_lexer_consume_token (parser->lexer);
cp_parser_enclosed_template_argument_list (parser);
if (start)
cp_lexer_purge_tokens_after (parser->lexer, start);
}
}
static bool
cp_parser_non_integral_constant_expression (cp_parser  *parser,
non_integral_constant thing)
{
parser->non_integral_constant_expression_p = true;
if (parser->integral_constant_expression_p)
{
if (!parser->allow_non_integral_constant_expression_p)
{
const char *msg = NULL;
switch (thing)
{
case NIC_FLOAT:
pedwarn (input_location, OPT_Wpedantic,
"ISO C++ forbids using a floating-point literal "
"in a constant-expression");
return true;
case NIC_CAST:
error ("a cast to a type other than an integral or "
"enumeration type cannot appear in a "
"constant-expression");
return true;
case NIC_TYPEID:
error ("%<typeid%> operator "
"cannot appear in a constant-expression");
return true;
case NIC_NCC:
error ("non-constant compound literals "
"cannot appear in a constant-expression");
return true;
case NIC_FUNC_CALL:
error ("a function call "
"cannot appear in a constant-expression");
return true;
case NIC_INC:
error ("an increment "
"cannot appear in a constant-expression");
return true;
case NIC_DEC:
error ("an decrement "
"cannot appear in a constant-expression");
return true;
case NIC_ARRAY_REF:
error ("an array reference "
"cannot appear in a constant-expression");
return true;
case NIC_ADDR_LABEL:
error ("the address of a label "
"cannot appear in a constant-expression");
return true;
case NIC_OVERLOADED:
error ("calls to overloaded operators "
"cannot appear in a constant-expression");
return true;
case NIC_ASSIGNMENT:
error ("an assignment cannot appear in a constant-expression");
return true;
case NIC_COMMA:
error ("a comma operator "
"cannot appear in a constant-expression");
return true;
case NIC_CONSTRUCTOR:
error ("a call to a constructor "
"cannot appear in a constant-expression");
return true;
case NIC_TRANSACTION:
error ("a transaction expression "
"cannot appear in a constant-expression");
return true;
case NIC_THIS:
msg = "this";
break;
case NIC_FUNC_NAME:
msg = "__FUNCTION__";
break;
case NIC_PRETTY_FUNC:
msg = "__PRETTY_FUNCTION__";
break;
case NIC_C99_FUNC:
msg = "__func__";
break;
case NIC_VA_ARG:
msg = "va_arg";
break;
case NIC_ARROW:
msg = "->";
break;
case NIC_POINT:
msg = ".";
break;
case NIC_STAR:
msg = "*";
break;
case NIC_ADDR:
msg = "&";
break;
case NIC_PREINCREMENT:
msg = "++";
break;
case NIC_PREDECREMENT:
msg = "--";
break;
case NIC_NEW:
msg = "new";
break;
case NIC_DEL:
msg = "delete";
break;
default:
gcc_unreachable ();
}
if (msg)
error ("%qs cannot appear in a constant-expression", msg);
return true;
}
}
return false;
}
static void
cp_parser_diagnose_invalid_type_name (cp_parser *parser, tree id,
location_t location)
{
tree decl, ambiguous_decls;
cp_parser_commit_to_tentative_parse (parser);
decl = cp_parser_lookup_name (parser, id, none_type,
false,
false,
true,
&ambiguous_decls, location);
if (ambiguous_decls)
return;
if (DECL_TYPE_TEMPLATE_P (decl))
{
error_at (location,
"invalid use of template-name %qE without an argument list",
decl);
if (DECL_CLASS_TEMPLATE_P (decl) && cxx_dialect < cxx17)
inform (location, "class template argument deduction is only available "
"with -std=c++17 or -std=gnu++17");
inform (DECL_SOURCE_LOCATION (decl), "%qD declared here", decl);
}
else if (TREE_CODE (id) == BIT_NOT_EXPR)
error_at (location, "invalid use of destructor %qD as a type", id);
else if (TREE_CODE (decl) == TYPE_DECL)
error_at (location, "invalid combination of multiple type-specifiers");
else if (!parser->scope)
{
name_hint hint;
if (TREE_CODE (id) == IDENTIFIER_NODE)
hint = lookup_name_fuzzy (id, FUZZY_LOOKUP_TYPENAME, location);
if (hint)
{
gcc_rich_location richloc (location);
richloc.add_fixit_replace (hint.suggestion ());
error_at (&richloc,
"%qE does not name a type; did you mean %qs?",
id, hint.suggestion ());
}
else
error_at (location, "%qE does not name a type", id);
if (cxx_dialect < cxx11 && id == ridpointers[(int)RID_CONSTEXPR])
inform (location, "C++11 %<constexpr%> only available with "
"-std=c++11 or -std=gnu++11");
else if (cxx_dialect < cxx11 && id == ridpointers[(int)RID_NOEXCEPT])
inform (location, "C++11 %<noexcept%> only available with "
"-std=c++11 or -std=gnu++11");
else if (cxx_dialect < cxx11
&& TREE_CODE (id) == IDENTIFIER_NODE
&& id_equal (id, "thread_local"))
inform (location, "C++11 %<thread_local%> only available with "
"-std=c++11 or -std=gnu++11");
else if (!flag_concepts && id == ridpointers[(int)RID_CONCEPT])
inform (location, "%<concept%> only available with -fconcepts");
else if (processing_template_decl && current_class_type
&& TYPE_BINFO (current_class_type))
{
tree b;
for (b = TREE_CHAIN (TYPE_BINFO (current_class_type));
b;
b = TREE_CHAIN (b))
{
tree base_type = BINFO_TYPE (b);
if (CLASS_TYPE_P (base_type)
&& dependent_type_p (base_type))
{
tree field;
base_type = CLASSTYPE_PRIMARY_TEMPLATE_TYPE (base_type);
for (field = TYPE_FIELDS (base_type);
field;
field = DECL_CHAIN (field))
if (TREE_CODE (field) == TYPE_DECL
&& DECL_NAME (field) == id)
{
inform (location, 
"(perhaps %<typename %T::%E%> was intended)",
BINFO_TYPE (b), id);
break;
}
if (field)
break;
}
}
}
}
else if (parser->scope != error_mark_node)
{
if (TREE_CODE (parser->scope) == NAMESPACE_DECL)
{
if (cp_lexer_next_token_is (parser->lexer, CPP_LESS))
error_at (location_of (id),
"%qE in namespace %qE does not name a template type",
id, parser->scope);
else if (TREE_CODE (id) == TEMPLATE_ID_EXPR)
error_at (location_of (id),
"%qE in namespace %qE does not name a template type",
TREE_OPERAND (id, 0), parser->scope);
else
error_at (location_of (id),
"%qE in namespace %qE does not name a type",
id, parser->scope);
if (DECL_P (decl))
inform (DECL_SOURCE_LOCATION (decl), "%qD declared here", decl);
else if (decl == error_mark_node)
suggest_alternative_in_explicit_scope (location, id,
parser->scope);
}
else if (CLASS_TYPE_P (parser->scope)
&& constructor_name_p (id, parser->scope))
{
error_at (location, "%<%T::%E%> names the constructor, not"
" the type", parser->scope, id);
if (cp_lexer_next_token_is (parser->lexer, CPP_LESS))
error_at (location, "and %qT has no template constructors",
parser->scope);
}
else if (TYPE_P (parser->scope)
&& dependent_scope_p (parser->scope))
{
if (TREE_CODE (parser->scope) == TYPENAME_TYPE)
error_at (location,
"need %<typename%> before %<%T::%D::%E%> because "
"%<%T::%D%> is a dependent scope",
TYPE_CONTEXT (parser->scope),
TYPENAME_TYPE_FULLNAME (parser->scope),
id,
TYPE_CONTEXT (parser->scope),
TYPENAME_TYPE_FULLNAME (parser->scope));
else
error_at (location, "need %<typename%> before %<%T::%E%> because "
"%qT is a dependent scope",
parser->scope, id, parser->scope);
}
else if (TYPE_P (parser->scope))
{
if (!COMPLETE_TYPE_P (parser->scope))
cxx_incomplete_type_error (location_of (id), NULL_TREE,
parser->scope);
else if (cp_lexer_next_token_is (parser->lexer, CPP_LESS))
error_at (location_of (id),
"%qE in %q#T does not name a template type",
id, parser->scope);
else if (TREE_CODE (id) == TEMPLATE_ID_EXPR)
error_at (location_of (id),
"%qE in %q#T does not name a template type",
TREE_OPERAND (id, 0), parser->scope);
else
error_at (location_of (id),
"%qE in %q#T does not name a type",
id, parser->scope);
if (DECL_P (decl))
inform (DECL_SOURCE_LOCATION (decl), "%qD declared here", decl);
}
else
gcc_unreachable ();
}
}
static bool
cp_parser_parse_and_diagnose_invalid_type_name (cp_parser *parser)
{
tree id;
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_NESTED_NAME_SPECIFIER)
{
cp_token *next = cp_lexer_peek_nth_token (parser->lexer, 2);
if (next->type == CPP_NAME && next->error_reported)
goto out;
}
cp_parser_parse_tentatively (parser);
id = cp_parser_id_expression (parser,
false,
true,
NULL,
false,
false);
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN)
|| TREE_CODE (id) == TYPE_DECL)
{
cp_parser_abort_tentative_parse (parser);
return false;
}
if (!cp_parser_parse_definitely (parser))
return false;
cp_parser_diagnose_invalid_type_name (parser, id, token->location);
out:
if (!parser->in_declarator_p)
cp_parser_skip_to_end_of_block_or_statement (parser);
return true;
}
static int
cp_parser_skip_to_closing_parenthesis_1 (cp_parser *parser,
bool recovering,
cpp_ttype or_ttype,
bool consume_paren)
{
unsigned paren_depth = 0;
unsigned brace_depth = 0;
unsigned square_depth = 0;
if (recovering && or_ttype == CPP_EOF
&& cp_parser_uncommitted_to_tentative_parse_p (parser))
return 0;
while (true)
{
cp_token * token = cp_lexer_peek_token (parser->lexer);
if (token->type == or_ttype && or_ttype != CPP_EOF
&& !brace_depth && !paren_depth && !square_depth)
return -1;
switch (token->type)
{
case CPP_EOF:
case CPP_PRAGMA_EOL:
return 0;
case CPP_OPEN_SQUARE:
++square_depth;
break;
case CPP_CLOSE_SQUARE:
if (!square_depth--)
return 0;
break;
case CPP_SEMICOLON:
if (!brace_depth)
return 0;
break;
case CPP_OPEN_BRACE:
++brace_depth;
break;
case CPP_CLOSE_BRACE:
if (!brace_depth--)
return 0;
break;
case CPP_OPEN_PAREN:
if (!brace_depth)
++paren_depth;
break;
case CPP_CLOSE_PAREN:
if (!brace_depth && !paren_depth--)
{
if (consume_paren)
cp_lexer_consume_token (parser->lexer);
return 1;
}
break;
default:
break;
}
cp_lexer_consume_token (parser->lexer);
}
}
static int
cp_parser_skip_to_closing_parenthesis (cp_parser *parser,
bool recovering,
bool or_comma,
bool consume_paren)
{
cpp_ttype ttype = or_comma ? CPP_COMMA : CPP_EOF;
return cp_parser_skip_to_closing_parenthesis_1 (parser, recovering,
ttype, consume_paren);
}
static void
cp_parser_skip_to_end_of_statement (cp_parser* parser)
{
unsigned nesting_depth = 0;
if (parser->fully_implicit_function_template_p)
abort_fully_implicit_template (parser);
while (true)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
switch (token->type)
{
case CPP_EOF:
case CPP_PRAGMA_EOL:
return;
case CPP_SEMICOLON:
if (!nesting_depth)
return;
break;
case CPP_CLOSE_BRACE:
if (nesting_depth == 0)
return;
if (--nesting_depth == 0)
{
cp_lexer_consume_token (parser->lexer);
return;
}
break;
case CPP_OPEN_BRACE:
++nesting_depth;
break;
default:
break;
}
cp_lexer_consume_token (parser->lexer);
}
}
static void
cp_parser_consume_semicolon_at_end_of_statement (cp_parser *parser)
{
if (!cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON))
{
cp_parser_skip_to_end_of_statement (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
cp_lexer_consume_token (parser->lexer);
}
}
static void
cp_parser_skip_to_end_of_block_or_statement (cp_parser* parser)
{
int nesting_depth = 0;
if (parser->fully_implicit_function_template_p)
abort_fully_implicit_template (parser);
while (nesting_depth >= 0)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
switch (token->type)
{
case CPP_EOF:
case CPP_PRAGMA_EOL:
return;
case CPP_SEMICOLON:
if (!nesting_depth)
nesting_depth = -1;
break;
case CPP_CLOSE_BRACE:
nesting_depth--;
if (nesting_depth < 0)
return;
if (!nesting_depth)
nesting_depth = -1;
break;
case CPP_OPEN_BRACE:
nesting_depth++;
break;
default:
break;
}
cp_lexer_consume_token (parser->lexer);
}
}
static bool
cp_parser_skip_to_closing_brace (cp_parser *parser)
{
unsigned nesting_depth = 0;
while (true)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
switch (token->type)
{
case CPP_EOF:
case CPP_PRAGMA_EOL:
return false;
case CPP_CLOSE_BRACE:
if (nesting_depth-- == 0)
return true;
break;
case CPP_OPEN_BRACE:
++nesting_depth;
break;
default:
break;
}
cp_lexer_consume_token (parser->lexer);
}
}
static void
cp_parser_skip_to_pragma_eol (cp_parser* parser, cp_token *pragma_tok)
{
cp_token *token;
parser->lexer->in_pragma = false;
do
token = cp_lexer_consume_token (parser->lexer);
while (token->type != CPP_PRAGMA_EOL && token->type != CPP_EOF);
cp_lexer_purge_tokens_after (parser->lexer, pragma_tok);
}
static void
cp_parser_require_pragma_eol (cp_parser *parser, cp_token *pragma_tok)
{
parser->lexer->in_pragma = false;
if (!cp_parser_require (parser, CPP_PRAGMA_EOL, RT_PRAGMA_EOL))
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
}
static tree
cp_parser_make_typename_type (cp_parser *parser, tree id,
location_t id_location)
{
tree result;
if (identifier_p (id))
{
result = make_typename_type (parser->scope, id, typename_type,
tf_none);
if (result == error_mark_node)
cp_parser_diagnose_invalid_type_name (parser, id, id_location);
return result;
}
return make_typename_type (parser->scope, id, typename_type, tf_error);
}
static cp_declarator *
cp_parser_make_indirect_declarator (enum tree_code code, tree class_type,
cp_cv_quals cv_qualifiers,
cp_declarator *target,
tree attributes)
{
if (code == ERROR_MARK || target == cp_error_declarator)
return cp_error_declarator;
if (code == INDIRECT_REF)
if (class_type == NULL_TREE)
return make_pointer_declarator (cv_qualifiers, target, attributes);
else
return make_ptrmem_declarator (cv_qualifiers, class_type,
target, attributes);
else if (code == ADDR_EXPR && class_type == NULL_TREE)
return make_reference_declarator (cv_qualifiers, target,
false, attributes);
else if (code == NON_LVALUE_EXPR && class_type == NULL_TREE)
return make_reference_declarator (cv_qualifiers, target,
true, attributes);
gcc_unreachable ();
}
static cp_parser *
cp_parser_new (void)
{
cp_parser *parser;
cp_lexer *lexer;
unsigned i;
lexer = cp_lexer_new_main ();
for (i = 0; i < sizeof (binops) / sizeof (binops[0]); i++)
binops_by_token[binops[i].token_type] = binops[i];
parser = ggc_cleared_alloc<cp_parser> ();
parser->lexer = lexer;
parser->context = cp_parser_context_new (NULL);
parser->allow_gnu_extensions_p = 1;
parser->greater_than_is_operator_p = true;
parser->default_arg_ok_p = true;
parser->integral_constant_expression_p = false;
parser->allow_non_integral_constant_expression_p = false;
parser->non_integral_constant_expression_p = false;
parser->local_variables_forbidden_p = false;
parser->in_unbraced_linkage_specification_p = false;
parser->in_declarator_p = false;
parser->in_template_argument_list_p = false;
parser->in_statement = 0;
parser->in_switch_statement_p = false;
parser->in_type_id_in_expr_p = false;
parser->implicit_extern_c = false;
parser->translate_strings_p = true;
parser->in_function_body = false;
parser->colon_corrects_to_scope_p = true;
push_unparsed_function_queues (parser);
parser->num_classes_being_defined = 0;
parser->num_template_parameter_lists = 0;
parser->omp_declare_simd = NULL;
parser->oacc_routine = NULL;
parser->auto_is_implicit_function_template_parm_p = false;
parser->fully_implicit_function_template_p = false;
parser->implicit_template_parms = 0;
parser->implicit_template_scope = 0;
parser->prevent_constrained_type_specifiers = 0;
parser->innermost_linkage_specification_location = UNKNOWN_LOCATION;
return parser;
}
static void
cp_parser_push_lexer_for_tokens (cp_parser *parser, cp_token_cache *cache)
{
cp_lexer *lexer = cp_lexer_new_from_tokens (cache);
lexer->next = parser->lexer;
parser->lexer = lexer;
cp_lexer_set_source_position_from_token (lexer->next_token);
}
static void
cp_parser_pop_lexer (cp_parser *parser)
{
cp_lexer *lexer = parser->lexer;
parser->lexer = lexer->next;
cp_lexer_destroy (lexer);
cp_lexer_set_source_position_from_token (parser->lexer->next_token);
}
static cp_expr
cp_parser_identifier (cp_parser* parser)
{
cp_token *token;
token = cp_parser_require (parser, CPP_NAME, RT_NAME);
if (token)
return cp_expr (token->u.value, token->location);
else
return error_mark_node;
}
static cp_expr
cp_parser_string_literal (cp_parser *parser, bool translate, bool wide_ok,
bool lookup_udlit = true)
{
tree value;
size_t count;
struct obstack str_ob;
cpp_string str, istr, *strs;
cp_token *tok;
enum cpp_ttype type, curr_type;
int have_suffix_p = 0;
tree string_tree;
tree suffix_id = NULL_TREE;
bool curr_tok_is_userdef_p = false;
tok = cp_lexer_peek_token (parser->lexer);
if (!cp_parser_is_string_literal (tok))
{
cp_parser_error (parser, "expected string-literal");
return error_mark_node;
}
location_t loc = tok->location;
if (cpp_userdef_string_p (tok->type))
{
string_tree = USERDEF_LITERAL_VALUE (tok->u.value);
curr_type = cpp_userdef_string_remove_type (tok->type);
curr_tok_is_userdef_p = true;
}
else
{
string_tree = tok->u.value;
curr_type = tok->type;
}
type = curr_type;
if (!cp_parser_is_string_literal
(cp_lexer_peek_nth_token (parser->lexer, 2)))
{
cp_lexer_consume_token (parser->lexer);
str.text = (const unsigned char *)TREE_STRING_POINTER (string_tree);
str.len = TREE_STRING_LENGTH (string_tree);
count = 1;
if (curr_tok_is_userdef_p)
{
suffix_id = USERDEF_LITERAL_SUFFIX_ID (tok->u.value);
have_suffix_p = 1;
curr_type = cpp_userdef_string_remove_type (tok->type);
}
else
curr_type = tok->type;
strs = &str;
}
else
{
location_t last_tok_loc = tok->location;
gcc_obstack_init (&str_ob);
count = 0;
do
{
cp_lexer_consume_token (parser->lexer);
count++;
str.text = (const unsigned char *)TREE_STRING_POINTER (string_tree);
str.len = TREE_STRING_LENGTH (string_tree);
if (curr_tok_is_userdef_p)
{
tree curr_suffix_id = USERDEF_LITERAL_SUFFIX_ID (tok->u.value);
if (have_suffix_p == 0)
{
suffix_id = curr_suffix_id;
have_suffix_p = 1;
}
else if (have_suffix_p == 1
&& curr_suffix_id != suffix_id)
{
error ("inconsistent user-defined literal suffixes"
" %qD and %qD in string literal",
suffix_id, curr_suffix_id);
have_suffix_p = -1;
}
curr_type = cpp_userdef_string_remove_type (tok->type);
}
else
curr_type = tok->type;
if (type != curr_type)
{
if (type == CPP_STRING)
type = curr_type;
else if (curr_type != CPP_STRING)
{
rich_location rich_loc (line_table, tok->location);
rich_loc.add_range (last_tok_loc, false);
error_at (&rich_loc,
"unsupported non-standard concatenation "
"of string literals");
}
}
obstack_grow (&str_ob, &str, sizeof (cpp_string));
last_tok_loc = tok->location;
tok = cp_lexer_peek_token (parser->lexer);
if (cpp_userdef_string_p (tok->type))
{
string_tree = USERDEF_LITERAL_VALUE (tok->u.value);
curr_type = cpp_userdef_string_remove_type (tok->type);
curr_tok_is_userdef_p = true;
}
else
{
string_tree = tok->u.value;
curr_type = tok->type;
curr_tok_is_userdef_p = false;
}
}
while (cp_parser_is_string_literal (tok));
loc = make_location (loc, loc, get_finish (last_tok_loc));
strs = (cpp_string *) obstack_finish (&str_ob);
}
if (type != CPP_STRING && !wide_ok)
{
cp_parser_error (parser, "a wide string is invalid in this context");
type = CPP_STRING;
}
if ((translate ? cpp_interpret_string : cpp_interpret_string_notranslate)
(parse_in, strs, count, &istr, type))
{
value = build_string (istr.len, (const char *)istr.text);
free (CONST_CAST (unsigned char *, istr.text));
switch (type)
{
default:
case CPP_STRING:
case CPP_UTF8STRING:
TREE_TYPE (value) = char_array_type_node;
break;
case CPP_STRING16:
TREE_TYPE (value) = char16_array_type_node;
break;
case CPP_STRING32:
TREE_TYPE (value) = char32_array_type_node;
break;
case CPP_WSTRING:
TREE_TYPE (value) = wchar_array_type_node;
break;
}
value = fix_string_type (value);
if (have_suffix_p)
{
tree literal = build_userdef_literal (suffix_id, value,
OT_NONE, NULL_TREE);
if (lookup_udlit)
value = cp_parser_userdef_string_literal (literal);
else
value = literal;
}
}
else
value = error_mark_node;
if (count > 1)
obstack_free (&str_ob, 0);
return cp_expr (value, loc);
}
static tree
lookup_literal_operator (tree name, vec<tree, va_gc> *args)
{
tree decl;
decl = lookup_name (name);
if (!decl || !is_overloaded_fn (decl))
return error_mark_node;
for (lkp_iterator iter (decl); iter; ++iter)
{
unsigned int ix;
bool found = true;
tree fn = *iter;
tree parmtypes = TYPE_ARG_TYPES (TREE_TYPE (fn));
if (parmtypes != NULL_TREE)
{
for (ix = 0; ix < vec_safe_length (args) && parmtypes != NULL_TREE;
++ix, parmtypes = TREE_CHAIN (parmtypes))
{
tree tparm = TREE_VALUE (parmtypes);
tree targ = TREE_TYPE ((*args)[ix]);
bool ptr = TYPE_PTR_P (tparm);
bool arr = TREE_CODE (targ) == ARRAY_TYPE;
if ((ptr || arr || !same_type_p (tparm, targ))
&& (!ptr || !arr
|| !same_type_p (TREE_TYPE (tparm),
TREE_TYPE (targ))))
found = false;
}
if (found
&& ix == vec_safe_length (args)
&& parmtypes == void_list_node)
return decl;
}
}
return error_mark_node;
}
static cp_expr
cp_parser_userdef_char_literal (cp_parser *parser)
{
cp_token *token = cp_lexer_consume_token (parser->lexer);
tree literal = token->u.value;
tree suffix_id = USERDEF_LITERAL_SUFFIX_ID (literal);
tree value = USERDEF_LITERAL_VALUE (literal);
tree name = cp_literal_operator_id (IDENTIFIER_POINTER (suffix_id));
tree decl, result;
vec<tree, va_gc> *args = make_tree_vector ();
vec_safe_push (args, value);
decl = lookup_literal_operator (name, args);
if (!decl || decl == error_mark_node)
{
error ("unable to find character literal operator %qD with %qT argument",
name, TREE_TYPE (value));
release_tree_vector (args);
return error_mark_node;
}
result = finish_call_expr (decl, &args, false, true, tf_warning_or_error);
release_tree_vector (args);
return result;
}
static tree
make_char_string_pack (tree value)
{
tree charvec;
tree argpack = make_node (NONTYPE_ARGUMENT_PACK);
const char *str = TREE_STRING_POINTER (value);
int i, len = TREE_STRING_LENGTH (value) - 1;
tree argvec = make_tree_vec (1);
charvec = make_tree_vec (len);
for (i = 0; i < len; ++i)
TREE_VEC_ELT (charvec, i) = build_int_cst (char_type_node, str[i]);
SET_ARGUMENT_PACK_ARGS (argpack, charvec);
TREE_VEC_ELT (argvec, 0) = argpack;
return argvec;
}
static tree
make_string_pack (tree value)
{
tree charvec;
tree argpack = make_node (NONTYPE_ARGUMENT_PACK);
const unsigned char *str
= (const unsigned char *) TREE_STRING_POINTER (value);
int sz = TREE_INT_CST_LOW (TYPE_SIZE_UNIT (TREE_TYPE (TREE_TYPE (value))));
int len = TREE_STRING_LENGTH (value) / sz - 1;
tree argvec = make_tree_vec (2);
tree str_char_type_node = TREE_TYPE (TREE_TYPE (value));
str_char_type_node = TYPE_MAIN_VARIANT (str_char_type_node);
TREE_VEC_ELT (argvec, 0) = str_char_type_node;
charvec = make_tree_vec (len);
for (int i = 0; i < len; ++i)
TREE_VEC_ELT (charvec, i)
= double_int_to_tree (str_char_type_node,
double_int::from_buffer (str + i * sz, sz));
SET_ARGUMENT_PACK_ARGS (argpack, charvec);
TREE_VEC_ELT (argvec, 1) = argpack;
return argvec;
}
static cp_expr
cp_parser_userdef_numeric_literal (cp_parser *parser)
{
cp_token *token = cp_lexer_consume_token (parser->lexer);
tree literal = token->u.value;
tree suffix_id = USERDEF_LITERAL_SUFFIX_ID (literal);
tree value = USERDEF_LITERAL_VALUE (literal);
int overflow = USERDEF_LITERAL_OVERFLOW (literal);
tree num_string = USERDEF_LITERAL_NUM_STRING (literal);
tree name = cp_literal_operator_id (IDENTIFIER_POINTER (suffix_id));
tree decl, result;
vec<tree, va_gc> *args;
args = make_tree_vector ();
vec_safe_push (args, value);
decl = lookup_literal_operator (name, args);
if (decl && decl != error_mark_node)
{
result = finish_call_expr (decl, &args, false, true,
tf_warning_or_error);
if (TREE_CODE (TREE_TYPE (value)) == INTEGER_TYPE && overflow > 0)
{
warning_at (token->location, OPT_Woverflow,
"integer literal exceeds range of %qT type",
long_long_unsigned_type_node);
}
else
{
if (overflow > 0)
warning_at (token->location, OPT_Woverflow,
"floating literal exceeds range of %qT type",
long_double_type_node);
else if (overflow < 0)
warning_at (token->location, OPT_Woverflow,
"floating literal truncated to zero");
}
release_tree_vector (args);
return result;
}
release_tree_vector (args);
args = make_tree_vector ();
vec_safe_push (args, num_string);
decl = lookup_literal_operator (name, args);
if (decl && decl != error_mark_node)
{
result = finish_call_expr (decl, &args, false, true,
tf_warning_or_error);
release_tree_vector (args);
return result;
}
release_tree_vector (args);
args = make_tree_vector ();
decl = lookup_literal_operator (name, args);
if (decl && decl != error_mark_node)
{
tree tmpl_args = make_char_string_pack (num_string);
decl = lookup_template_function (decl, tmpl_args);
result = finish_call_expr (decl, &args, false, true,
tf_warning_or_error);
release_tree_vector (args);
return result;
}
release_tree_vector (args);
bool ext = cpp_get_options (parse_in)->ext_numeric_literals;
bool i14 = (cxx_dialect > cxx11
&& (id_equal (suffix_id, "i")
|| id_equal (suffix_id, "if")
|| id_equal (suffix_id, "il")));
diagnostic_t kind = DK_ERROR;
int opt = 0;
if (i14 && ext)
{
tree cxlit = lookup_qualified_name (std_node,
get_identifier ("complex_literals"),
0, false, false);
if (cxlit == error_mark_node)
{
kind = DK_PEDWARN;
opt = OPT_Wpedantic;
}
}
bool complained
= emit_diagnostic (kind, input_location, opt,
"unable to find numeric literal operator %qD", name);
if (!complained)
;
else if (i14)
{
inform (token->location, "add %<using namespace std::complex_literals%> "
"(from <complex>) to enable the C++14 user-defined literal "
"suffixes");
if (ext)
inform (token->location, "or use %<j%> instead of %<i%> for the "
"GNU built-in suffix");
}
else if (!ext)
inform (token->location, "use -fext-numeric-literals "
"to enable more built-in suffixes");
if (kind == DK_ERROR)
value = error_mark_node;
else
{
tree type;
if (id_equal (suffix_id, "i"))
{
if (TREE_CODE (value) == INTEGER_CST)
type = integer_type_node;
else
type = double_type_node;
}
else if (id_equal (suffix_id, "if"))
type = float_type_node;
else 
type = long_double_type_node;
value = build_complex (build_complex_type (type),
fold_convert (type, integer_zero_node),
fold_convert (type, value));
}
if (cp_parser_uncommitted_to_tentative_parse_p (parser))
token->u.value = value;
return value;
}
static tree
cp_parser_userdef_string_literal (tree literal)
{
tree suffix_id = USERDEF_LITERAL_SUFFIX_ID (literal);
tree name = cp_literal_operator_id (IDENTIFIER_POINTER (suffix_id));
tree value = USERDEF_LITERAL_VALUE (literal);
int len = TREE_STRING_LENGTH (value)
/ TREE_INT_CST_LOW (TYPE_SIZE_UNIT (TREE_TYPE (TREE_TYPE (value)))) - 1;
tree decl, result;
vec<tree, va_gc> *args;
args = make_tree_vector ();
vec_safe_push (args, value);
vec_safe_push (args, build_int_cst (size_type_node, len));
decl = lookup_literal_operator (name, args);
if (decl && decl != error_mark_node)
{
result = finish_call_expr (decl, &args, false, true,
tf_warning_or_error);
release_tree_vector (args);
return result;
}
release_tree_vector (args);
args = make_tree_vector ();
decl = lookup_literal_operator (name, args);
if (decl && decl != error_mark_node)
{
tree tmpl_args = make_string_pack (value);
decl = lookup_template_function (decl, tmpl_args);
result = finish_call_expr (decl, &args, false, true,
tf_warning_or_error);
release_tree_vector (args);
return result;
}
release_tree_vector (args);
error ("unable to find string literal operator %qD with %qT, %qT arguments",
name, TREE_TYPE (value), size_type_node);
return error_mark_node;
}
static bool
cp_parser_translation_unit (cp_parser* parser)
{
static void *declarator_obstack_base;
bool success;
if (!cp_error_declarator)
{
gcc_obstack_init (&declarator_obstack);
cp_error_declarator = make_declarator (cdk_error);
no_parameters = make_parameter_declarator (NULL, NULL, NULL_TREE,
UNKNOWN_LOCATION);
declarator_obstack_base = obstack_next_free (&declarator_obstack);
}
cp_parser_declaration_seq_opt (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_EOF))
{
cp_lexer_destroy (parser->lexer);
parser->lexer = NULL;
if (parser->implicit_extern_c)
{
pop_lang_context ();
parser->implicit_extern_c = false;
}
finish_translation_unit ();
success = true;
}
else
{
cp_parser_error (parser, "expected declaration");
success = false;
}
gcc_assert (obstack_next_free (&declarator_obstack)
== declarator_obstack_base);
return success;
}
static inline tsubst_flags_t
complain_flags (bool decltype_p)
{
tsubst_flags_t complain = tf_warning_or_error;
if (decltype_p)
complain |= tf_decltype;
return complain;
}
static cp_token_position
cp_parser_start_tentative_firewall (cp_parser *parser)
{
if (!cp_parser_uncommitted_to_tentative_parse_p (parser))
return 0;
cp_parser_parse_tentatively (parser);
cp_parser_commit_to_topmost_tentative_parse (parser);
return cp_lexer_token_position (parser->lexer, false);
}
static void
cp_parser_end_tentative_firewall (cp_parser *parser, cp_token_position start,
tree expr)
{
if (!start)
return;
cp_parser_parse_definitely (parser);
cp_token *token = cp_lexer_token_at (parser->lexer, start);
token->type = CPP_PREPARSED_EXPR;
token->u.value = expr;
token->keyword = RID_MAX;
cp_lexer_purge_tokens_after (parser->lexer, start);
}
struct tentative_firewall
{
cp_parser *parser;
bool set;
tentative_firewall (cp_parser *p): parser(p)
{
if ((set = cp_parser_uncommitted_to_tentative_parse_p (parser)))
{
cp_parser_parse_tentatively (parser);
cp_parser_commit_to_topmost_tentative_parse (parser);
cp_parser_parse_tentatively (parser);
}
}
~tentative_firewall()
{
if (set)
{
bool err = cp_parser_error_occurred (parser);
cp_parser_parse_definitely (parser);
cp_parser_parse_definitely (parser);
if (err)
cp_parser_simulate_error (parser);
}
}
};
template <typename traits_t>
class token_pair
{
public:
token_pair () : m_open_loc (UNKNOWN_LOCATION) {}
bool require_open (cp_parser *parser)
{
m_open_loc = cp_lexer_peek_token (parser->lexer)->location;
return cp_parser_require (parser, traits_t::open_token_type,
traits_t::required_token_open);
}
cp_token * consume_open (cp_parser *parser)
{
cp_token *tok = cp_lexer_consume_token (parser->lexer);
gcc_assert (tok->type == traits_t::open_token_type);
m_open_loc = tok->location;
return tok;
}
cp_token *require_close (cp_parser *parser) const
{
return cp_parser_require (parser, traits_t::close_token_type,
traits_t::required_token_close,
m_open_loc);
}
private:
location_t m_open_loc;
};
struct matching_paren_traits
{
static const enum cpp_ttype open_token_type = CPP_OPEN_PAREN;
static const enum required_token required_token_open  = RT_OPEN_PAREN;
static const enum cpp_ttype close_token_type = CPP_CLOSE_PAREN;
static const enum required_token required_token_close = RT_CLOSE_PAREN;
};
typedef token_pair<matching_paren_traits> matching_parens;
struct matching_brace_traits
{
static const enum cpp_ttype open_token_type = CPP_OPEN_BRACE;
static const enum required_token required_token_open = RT_OPEN_BRACE;
static const enum cpp_ttype close_token_type = CPP_CLOSE_BRACE;
static const enum required_token required_token_close = RT_CLOSE_BRACE;
};
typedef token_pair<matching_brace_traits> matching_braces;
static cp_expr
cp_parser_statement_expr (cp_parser *parser)
{
cp_token_position start = cp_parser_start_tentative_firewall (parser);
location_t start_loc = cp_lexer_peek_token (parser->lexer)->location;
matching_parens parens;
parens.consume_open (parser);
tree expr = begin_stmt_expr ();
cp_parser_compound_statement (parser, expr, BCS_NORMAL, false);
expr = finish_stmt_expr (expr, false);
location_t finish_loc = cp_lexer_peek_token (parser->lexer)->location;
if (!parens.require_close (parser))
cp_parser_skip_to_end_of_statement (parser);
cp_parser_end_tentative_firewall (parser, start, expr);
location_t combined_loc = make_location (start_loc, start_loc, finish_loc);
return cp_expr (expr, combined_loc);
}
static int
cp_parser_fold_operator (cp_token *token)
{
switch (token->type)
{
case CPP_PLUS: return PLUS_EXPR;
case CPP_MINUS: return MINUS_EXPR;
case CPP_MULT: return MULT_EXPR;
case CPP_DIV: return TRUNC_DIV_EXPR;
case CPP_MOD: return TRUNC_MOD_EXPR;
case CPP_XOR: return BIT_XOR_EXPR;
case CPP_AND: return BIT_AND_EXPR;
case CPP_OR: return BIT_IOR_EXPR;
case CPP_LSHIFT: return LSHIFT_EXPR;
case CPP_RSHIFT: return RSHIFT_EXPR;
case CPP_EQ: return -NOP_EXPR;
case CPP_PLUS_EQ: return -PLUS_EXPR;
case CPP_MINUS_EQ: return -MINUS_EXPR;
case CPP_MULT_EQ: return -MULT_EXPR;
case CPP_DIV_EQ: return -TRUNC_DIV_EXPR;
case CPP_MOD_EQ: return -TRUNC_MOD_EXPR;
case CPP_XOR_EQ: return -BIT_XOR_EXPR;
case CPP_AND_EQ: return -BIT_AND_EXPR;
case CPP_OR_EQ: return -BIT_IOR_EXPR;
case CPP_LSHIFT_EQ: return -LSHIFT_EXPR;
case CPP_RSHIFT_EQ: return -RSHIFT_EXPR;
case CPP_EQ_EQ: return EQ_EXPR;
case CPP_NOT_EQ: return NE_EXPR;
case CPP_LESS: return LT_EXPR;
case CPP_GREATER: return GT_EXPR;
case CPP_LESS_EQ: return LE_EXPR;
case CPP_GREATER_EQ: return GE_EXPR;
case CPP_AND_AND: return TRUTH_ANDIF_EXPR;
case CPP_OR_OR: return TRUTH_ORIF_EXPR;
case CPP_COMMA: return COMPOUND_EXPR;
case CPP_DOT_STAR: return DOTSTAR_EXPR;
case CPP_DEREF_STAR: return MEMBER_REF;
default: return ERROR_MARK;
}
}
static bool
is_binary_op (tree_code code)
{
switch (code)
{
case PLUS_EXPR:
case POINTER_PLUS_EXPR:
case MINUS_EXPR:
case MULT_EXPR:
case TRUNC_DIV_EXPR:
case TRUNC_MOD_EXPR:
case BIT_XOR_EXPR:
case BIT_AND_EXPR:
case BIT_IOR_EXPR:
case LSHIFT_EXPR:
case RSHIFT_EXPR:
case MODOP_EXPR:
case EQ_EXPR:
case NE_EXPR:
case LE_EXPR:
case GE_EXPR:
case LT_EXPR:
case GT_EXPR:
case TRUTH_ANDIF_EXPR:
case TRUTH_ORIF_EXPR:
case COMPOUND_EXPR:
case DOTSTAR_EXPR:
case MEMBER_REF:
return true;
default:
return false;
}
}
static int
cp_parser_fold_operator (cp_parser *parser)
{
cp_token* token = cp_lexer_peek_token (parser->lexer);
int code = cp_parser_fold_operator (token);
if (code != ERROR_MARK)
cp_lexer_consume_token (parser->lexer);
return code;
}
static cp_expr
cp_parser_fold_expression (cp_parser *parser, tree expr1)
{
cp_id_kind pidk;
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
cp_lexer_consume_token (parser->lexer);
int op = cp_parser_fold_operator (parser);
if (op == ERROR_MARK)
{
cp_parser_error (parser, "expected binary operator");
return error_mark_node;
}
tree expr = cp_parser_cast_expression (parser, false, false,
false, &pidk);
if (expr == error_mark_node)
return error_mark_node;
return finish_left_unary_fold_expr (expr, op);
}
const cp_token* token = cp_lexer_peek_token (parser->lexer);
int op = cp_parser_fold_operator (parser);
if (op == ERROR_MARK)
{
cp_parser_error (parser, "expected binary operator");
return error_mark_node;
}
if (cp_lexer_next_token_is_not (parser->lexer, CPP_ELLIPSIS))
{
cp_parser_error (parser, "expected ...");
return error_mark_node;
}
cp_lexer_consume_token (parser->lexer);
if (EXPR_P (expr1) && TREE_NO_WARNING (expr1))
;
else if (is_binary_op (TREE_CODE (expr1)))
error_at (location_of (expr1),
"binary expression in operand of fold-expression");
else if (TREE_CODE (expr1) == COND_EXPR
|| (REFERENCE_REF_P (expr1)
&& TREE_CODE (TREE_OPERAND (expr1, 0)) == COND_EXPR))
error_at (location_of (expr1),
"conditional expression in operand of fold-expression");
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_PAREN))
return finish_right_unary_fold_expr (expr1, op);
if (cp_lexer_next_token_is_not (parser->lexer, token->type))
{
cp_parser_error (parser, "mismatched operator in fold-expression");
return error_mark_node;
}
cp_lexer_consume_token (parser->lexer);
tree expr2 = cp_parser_cast_expression (parser, false, false, false, &pidk);
if (expr2 == error_mark_node)
return error_mark_node;
return finish_binary_fold_expr (expr1, expr2, op);
}
static cp_expr
cp_parser_primary_expression (cp_parser *parser,
bool address_p,
bool cast_p,
bool template_arg_p,
bool decltype_p,
cp_id_kind *idk)
{
cp_token *token = NULL;
*idk = CP_ID_KIND_NONE;
token = cp_lexer_peek_token (parser->lexer);
switch ((int) token->type)
{
case CPP_CHAR:
case CPP_CHAR16:
case CPP_CHAR32:
case CPP_WCHAR:
case CPP_UTF8CHAR:
case CPP_NUMBER:
case CPP_PREPARSED_EXPR:
if (TREE_CODE (token->u.value) == USERDEF_LITERAL)
return cp_parser_userdef_numeric_literal (parser);
token = cp_lexer_consume_token (parser->lexer);
if (TREE_CODE (token->u.value) == FIXED_CST)
{
error_at (token->location,
"fixed-point types not supported in C++");
return error_mark_node;
}
if (TREE_CODE (token->u.value) == REAL_CST
&& parser->integral_constant_expression_p
&& pedantic)
{
if (cast_p)
{
cp_token *next_token;
next_token = cp_lexer_peek_token (parser->lexer);
if (
next_token->type != CPP_COMMA
&& next_token->type != CPP_CLOSE_BRACE
&& next_token->type != CPP_SEMICOLON
&& next_token->type != CPP_CLOSE_PAREN
&& next_token->type != CPP_CLOSE_SQUARE
&& (next_token->type != CPP_GREATER
|| parser->greater_than_is_operator_p)
&& (next_token->type != CPP_RSHIFT
|| (cxx_dialect == cxx98)
|| parser->greater_than_is_operator_p))
cast_p = false;
}
if (!cast_p)
cp_parser_non_integral_constant_expression (parser, NIC_FLOAT);
}
return cp_expr (token->u.value, token->location);
case CPP_CHAR_USERDEF:
case CPP_CHAR16_USERDEF:
case CPP_CHAR32_USERDEF:
case CPP_WCHAR_USERDEF:
case CPP_UTF8CHAR_USERDEF:
return cp_parser_userdef_char_literal (parser);
case CPP_STRING:
case CPP_STRING16:
case CPP_STRING32:
case CPP_WSTRING:
case CPP_UTF8STRING:
case CPP_STRING_USERDEF:
case CPP_STRING16_USERDEF:
case CPP_STRING32_USERDEF:
case CPP_WSTRING_USERDEF:
case CPP_UTF8STRING_USERDEF:
return cp_parser_string_literal (parser,
parser->translate_strings_p,
true);
case CPP_OPEN_PAREN:
if (cp_parser_allow_gnu_extensions_p (parser)
&& cp_lexer_nth_token_is (parser->lexer, 2, CPP_OPEN_BRACE))
{
pedwarn (token->location, OPT_Wpedantic,
"ISO C++ forbids braced-groups within expressions");
if (!parser->in_function_body
|| parser->in_template_argument_list_p)
{
error_at (token->location,
"statement-expressions are not allowed outside "
"functions nor in template-argument lists");
cp_parser_skip_to_end_of_block_or_statement (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_PAREN))
cp_lexer_consume_token (parser->lexer);
return error_mark_node;
}
else
return cp_parser_statement_expr (parser);
}
{
cp_expr expr;
bool saved_greater_than_is_operator_p;
location_t open_paren_loc = token->location;
matching_parens parens;
parens.consume_open (parser);
saved_greater_than_is_operator_p
= parser->greater_than_is_operator_p;
parser->greater_than_is_operator_p = true;
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
expr = NULL_TREE;
else
expr = cp_parser_expression (parser, idk, cast_p, decltype_p);
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_ELLIPSIS || cp_parser_fold_operator (token))
{
expr = cp_parser_fold_expression (parser, expr);
if (expr != error_mark_node
&& cxx_dialect < cxx17
&& !in_system_header_at (input_location))
pedwarn (input_location, 0, "fold-expressions only available "
"with -std=c++17 or -std=gnu++17");
}
else
expr = finish_parenthesized_expr (expr);
if (*idk != CP_ID_KIND_QUALIFIED)
*idk = CP_ID_KIND_NONE;
parser->greater_than_is_operator_p
= saved_greater_than_is_operator_p;
token = cp_lexer_peek_token (parser->lexer);
location_t close_paren_loc = token->location;
expr.set_range (open_paren_loc, close_paren_loc);
if (!parens.require_close (parser)
&& !cp_parser_uncommitted_to_tentative_parse_p (parser))
cp_parser_skip_to_end_of_statement (parser);
return expr;
}
case CPP_OPEN_SQUARE:
{
if (c_dialect_objc ())
{
cp_parser_parse_tentatively (parser);
tree msg = cp_parser_objc_message_expression (parser);
if (cp_parser_parse_definitely (parser))
return msg;
}
cp_expr lam = cp_parser_lambda_expression (parser);
if (cp_parser_error_occurred (parser))
return error_mark_node;
maybe_warn_cpp0x (CPP0X_LAMBDA_EXPR);
return lam;
}
case CPP_OBJC_STRING:
if (c_dialect_objc ())
return cp_parser_objc_expression (parser);
cp_parser_error (parser, "expected primary-expression");
return error_mark_node;
case CPP_KEYWORD:
switch (token->keyword)
{
case RID_TRUE:
cp_lexer_consume_token (parser->lexer);
return cp_expr (boolean_true_node, token->location);
case RID_FALSE:
cp_lexer_consume_token (parser->lexer);
return cp_expr (boolean_false_node, token->location);
case RID_NULL:
cp_lexer_consume_token (parser->lexer);
return cp_expr (null_node, token->location);
case RID_NULLPTR:
cp_lexer_consume_token (parser->lexer);
return cp_expr (nullptr_node, token->location);
case RID_THIS:
cp_lexer_consume_token (parser->lexer);
if (parser->local_variables_forbidden_p)
{
error_at (token->location,
"%<this%> may not be used in this context");
return error_mark_node;
}
if (cp_parser_non_integral_constant_expression (parser, NIC_THIS))
return error_mark_node;
return cp_expr (finish_this_expr (), token->location);
case RID_OPERATOR:
goto id_expression;
case RID_FUNCTION_NAME:
case RID_PRETTY_FUNCTION_NAME:
case RID_C99_FUNCTION_NAME:
{
non_integral_constant name;
token = cp_lexer_consume_token (parser->lexer);
switch (token->keyword)
{
case RID_FUNCTION_NAME:
name = NIC_FUNC_NAME;
break;
case RID_PRETTY_FUNCTION_NAME:
name = NIC_PRETTY_FUNC;
break;
case RID_C99_FUNCTION_NAME:
name = NIC_C99_FUNC;
break;
default:
gcc_unreachable ();
}
if (cp_parser_non_integral_constant_expression (parser, name))
return error_mark_node;
return finish_fname (token->u.value);
}
case RID_VA_ARG:
{
tree expression;
tree type;
source_location type_location;
location_t start_loc
= cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);
matching_parens parens;
parens.require_open (parser);
expression = cp_parser_assignment_expression (parser);
cp_parser_require (parser, CPP_COMMA, RT_COMMA);
type_location = cp_lexer_peek_token (parser->lexer)->location;
{
type_id_in_expr_sentinel s (parser);
type = cp_parser_type_id (parser);
}
location_t finish_loc
= cp_lexer_peek_token (parser->lexer)->location;
parens.require_close (parser);
if (cp_parser_non_integral_constant_expression (parser,
NIC_VA_ARG))
return error_mark_node;
location_t combined_loc
= make_location (type_location, start_loc, finish_loc);
return build_x_va_arg (combined_loc, expression, type);
}
case RID_OFFSETOF:
return cp_parser_builtin_offsetof (parser);
case RID_HAS_NOTHROW_ASSIGN:
case RID_HAS_NOTHROW_CONSTRUCTOR:
case RID_HAS_NOTHROW_COPY:	  
case RID_HAS_TRIVIAL_ASSIGN:
case RID_HAS_TRIVIAL_CONSTRUCTOR:
case RID_HAS_TRIVIAL_COPY:	  
case RID_HAS_TRIVIAL_DESTRUCTOR:
case RID_HAS_UNIQUE_OBJ_REPRESENTATIONS:
case RID_HAS_VIRTUAL_DESTRUCTOR:
case RID_IS_ABSTRACT:
case RID_IS_AGGREGATE:
case RID_IS_BASE_OF:
case RID_IS_CLASS:
case RID_IS_EMPTY:
case RID_IS_ENUM:
case RID_IS_FINAL:
case RID_IS_LITERAL_TYPE:
case RID_IS_POD:
case RID_IS_POLYMORPHIC:
case RID_IS_SAME_AS:
case RID_IS_STD_LAYOUT:
case RID_IS_TRIVIAL:
case RID_IS_TRIVIALLY_ASSIGNABLE:
case RID_IS_TRIVIALLY_CONSTRUCTIBLE:
case RID_IS_TRIVIALLY_COPYABLE:
case RID_IS_UNION:
case RID_IS_ASSIGNABLE:
case RID_IS_CONSTRUCTIBLE:
return cp_parser_trait_expr (parser, token->keyword);
case RID_REQUIRES:
return cp_parser_requires_expression (parser);
case RID_AT_ENCODE:
case RID_AT_PROTOCOL:
case RID_AT_SELECTOR:
return cp_parser_objc_expression (parser);
case RID_TEMPLATE:
if (parser->in_function_body
&& (cp_lexer_peek_nth_token (parser->lexer, 2)->type
== CPP_LESS))
{
error_at (token->location,
"a template declaration cannot appear at block scope");
cp_parser_skip_to_end_of_block_or_statement (parser);
return error_mark_node;
}
default:
cp_parser_error (parser, "expected primary-expression");
return error_mark_node;
}
case CPP_NAME:
case CPP_SCOPE:
case CPP_TEMPLATE_ID:
case CPP_NESTED_NAME_SPECIFIER:
{
id_expression:
cp_expr id_expression;
cp_expr decl;
const char *error_msg;
bool template_p;
bool done;
cp_token *id_expr_token;
id_expression
= cp_parser_id_expression (parser,
false,
true,
&template_p,
false,
false);
if (id_expression == error_mark_node)
return error_mark_node;
id_expr_token = token;
token = cp_lexer_peek_token (parser->lexer);
done = (token->type != CPP_OPEN_SQUARE
&& token->type != CPP_OPEN_PAREN
&& token->type != CPP_DOT
&& token->type != CPP_DEREF
&& token->type != CPP_PLUS_PLUS
&& token->type != CPP_MINUS_MINUS);
if (TREE_CODE (id_expression) == TEMPLATE_ID_EXPR
|| TREE_CODE (id_expression) == TYPE_DECL)
decl = id_expression;
else
{
tree ambiguous_decls;
if (id_expr_token->type == CPP_NAME
&& id_expr_token->error_reported)
{
cp_parser_simulate_error (parser);
return error_mark_node;
}
decl = cp_parser_lookup_name (parser, id_expression,
none_type,
template_p,
false,
true,
&ambiguous_decls,
id_expr_token->location);
if (ambiguous_decls)
return error_mark_node;
if (c_dialect_objc ()
&& cp_lexer_peek_token (parser->lexer)->type == CPP_DOT
&& TREE_CODE (decl) == TYPE_DECL
&& objc_is_class_name (decl))
{
tree component;
cp_lexer_consume_token (parser->lexer);
component = cp_parser_identifier (parser);
if (component == error_mark_node)
return error_mark_node;
tree result = objc_build_class_component_ref (id_expression,
component);
location_t combined_loc
= make_location (input_location, id_expression.get_start (),
get_finish (input_location));
protected_set_expr_location (result, combined_loc);
return result;
}
tree decl_tree = objc_lookup_ivar (decl.get_value (),
id_expression);
if (decl_tree != decl.get_value ())
decl = cp_expr (decl_tree);
if (TREE_CODE (decl) == SCOPE_REF)
{
return decl;
}
if (parser->local_variables_forbidden_p
&& local_variable_p (decl))
{
decl = check_for_out_of_scope_variable (decl);
if (local_variable_p (decl))
{
error_at (id_expr_token->location,
"local variable %qD may not appear in this context",
decl.get_value ());
return error_mark_node;
}
}
}
if (processing_template_decl && is_overloaded_fn (decl))
lookup_keep (get_fns (decl), true);
decl = (finish_id_expression
(id_expression, decl, parser->scope,
idk,
parser->integral_constant_expression_p,
parser->allow_non_integral_constant_expression_p,
&parser->non_integral_constant_expression_p,
template_p, done, address_p,
template_arg_p,
&error_msg,
id_expression.get_location ()));
if (error_msg)
cp_parser_error (parser, error_msg);
decl.set_location (id_expr_token->location);
return decl;
}
default:
cp_parser_error (parser, "expected primary-expression");
return error_mark_node;
}
}
static inline cp_expr
cp_parser_primary_expression (cp_parser *parser,
bool address_p,
bool cast_p,
bool template_arg_p,
cp_id_kind *idk)
{
return cp_parser_primary_expression (parser, address_p, cast_p, template_arg_p,
false, idk);
}
static cp_expr
cp_parser_id_expression (cp_parser *parser,
bool template_keyword_p,
bool check_dependency_p,
bool *template_p,
bool declarator_p,
bool optional_p)
{
bool global_scope_p;
bool nested_name_specifier_p;
if (template_p)
*template_p = template_keyword_p;
global_scope_p
= (!template_keyword_p
&& (cp_parser_global_scope_opt (parser,
false)
!= NULL_TREE));
nested_name_specifier_p
= (cp_parser_nested_name_specifier_opt (parser,
false,
check_dependency_p,
false,
declarator_p,
template_keyword_p)
!= NULL_TREE);
if (nested_name_specifier_p)
{
tree saved_scope;
tree saved_object_scope;
tree saved_qualifying_scope;
cp_expr unqualified_id;
bool is_template;
if (!template_p)
template_p = &is_template;
*template_p = cp_parser_optional_template_keyword (parser);
saved_scope = parser->scope;
saved_object_scope = parser->object_scope;
saved_qualifying_scope = parser->qualifying_scope;
unqualified_id = cp_parser_unqualified_id (parser, *template_p,
check_dependency_p,
declarator_p,
false);
parser->scope = saved_scope;
parser->object_scope = saved_object_scope;
parser->qualifying_scope = saved_qualifying_scope;
return unqualified_id;
}
else if (global_scope_p)
{
cp_token *token;
tree id;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_NAME
&& !cp_parser_nth_token_starts_template_argument_list_p
(parser, 2))
return cp_parser_identifier (parser);
cp_parser_parse_tentatively (parser);
id = cp_parser_template_id (parser,
false,
true,
none_type,
declarator_p);
if (cp_parser_parse_definitely (parser))
return id;
token = cp_lexer_peek_token (parser->lexer);
switch (token->type)
{
case CPP_NAME:
return cp_parser_identifier (parser);
case CPP_KEYWORD:
if (token->keyword == RID_OPERATOR)
return cp_parser_operator_function_id (parser);
default:
cp_parser_error (parser, "expected id-expression");
return error_mark_node;
}
}
else
return cp_parser_unqualified_id (parser, template_keyword_p,
true,
declarator_p,
optional_p);
}
static cp_expr
cp_parser_unqualified_id (cp_parser* parser,
bool template_keyword_p,
bool check_dependency_p,
bool declarator_p,
bool optional_p)
{
cp_token *token;
token = cp_lexer_peek_token (parser->lexer);
switch ((int) token->type)
{
case CPP_NAME:
{
tree id;
cp_parser_parse_tentatively (parser);
id = cp_parser_template_id (parser, template_keyword_p,
check_dependency_p,
none_type,
declarator_p);
if (cp_parser_parse_definitely (parser))
return id;
return cp_parser_identifier (parser);
}
case CPP_TEMPLATE_ID:
return cp_parser_template_id (parser, template_keyword_p,
check_dependency_p,
none_type,
declarator_p);
case CPP_COMPL:
{
tree type_decl;
tree qualifying_scope;
tree object_scope;
tree scope;
bool done;
cp_lexer_consume_token (parser->lexer);
scope = parser->scope;
object_scope = parser->object_scope;
qualifying_scope = parser->qualifying_scope;
if (scope == error_mark_node)
{
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
cp_lexer_consume_token (parser->lexer);
return error_mark_node;
}
if (scope && TREE_CODE (scope) == NAMESPACE_DECL)
{
if (!cp_parser_uncommitted_to_tentative_parse_p (parser))
error_at (token->location,
"scope %qT before %<~%> is not a class-name",
scope);
cp_parser_simulate_error (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
cp_lexer_consume_token (parser->lexer);
return error_mark_node;
}
gcc_assert (!scope || TYPE_P (scope));
token = cp_lexer_peek_token (parser->lexer);
if (scope
&& token->type == CPP_NAME
&& (cp_lexer_peek_nth_token (parser->lexer, 2)->type
!= CPP_LESS)
&& (token->u.value == TYPE_IDENTIFIER (scope)
|| (CLASS_TYPE_P (scope)
&& constructor_name_p (token->u.value, scope))))
{
cp_lexer_consume_token (parser->lexer);
return build_nt (BIT_NOT_EXPR, scope);
}
if (cp_parser_is_keyword (token, RID_AUTO))
{
if (cxx_dialect < cxx14)
pedwarn (input_location, 0,
"%<~auto%> only available with "
"-std=c++14 or -std=gnu++14");
cp_lexer_consume_token (parser->lexer);
return build_nt (BIT_NOT_EXPR, make_auto ());
}
done = false;
type_decl = NULL_TREE;
if (scope)
{
cp_parser_parse_tentatively (parser);
type_decl = cp_parser_class_name (parser,
false,
false,
typename_type,
false,
false,
declarator_p);
if (cp_parser_parse_definitely (parser))
done = true;
}
if (!done && scope && qualifying_scope)
{
cp_parser_parse_tentatively (parser);
parser->scope = qualifying_scope;
parser->object_scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
type_decl
= cp_parser_class_name (parser,
false,
false,
typename_type,
false,
false,
declarator_p);
if (cp_parser_parse_definitely (parser))
done = true;
}
else if (!done && object_scope)
{
cp_parser_parse_tentatively (parser);
parser->scope = object_scope;
parser->object_scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
type_decl
= cp_parser_class_name (parser,
false,
false,
typename_type,
false,
false,
declarator_p);
if (cp_parser_parse_definitely (parser))
done = true;
}
if (!done)
{
parser->scope = NULL_TREE;
parser->object_scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
if (processing_template_decl)
cp_parser_parse_tentatively (parser);
type_decl
= cp_parser_class_name (parser,
false,
false,
typename_type,
false,
false,
declarator_p);
if (processing_template_decl
&& ! cp_parser_parse_definitely (parser))
{
if (cp_parser_uncommitted_to_tentative_parse_p (parser))
{
cp_parser_simulate_error (parser);
return error_mark_node;
}
type_decl = cp_parser_identifier (parser);
if (type_decl != error_mark_node)
type_decl = build_nt (BIT_NOT_EXPR, type_decl);
return type_decl;
}
}
if (type_decl == error_mark_node && scope)
return build_nt (BIT_NOT_EXPR, scope);
else if (type_decl == error_mark_node)
return error_mark_node;
if (declarator_p && scope && !check_dtor_name (scope, type_decl))
{
if (!cp_parser_uncommitted_to_tentative_parse_p (parser))
error_at (token->location,
"declaration of %<~%T%> as member of %qT",
type_decl, scope);
cp_parser_simulate_error (parser);
return error_mark_node;
}
if (declarator_p
&& !DECL_IMPLICIT_TYPEDEF_P (type_decl)
&& !DECL_SELF_REFERENCE_P (type_decl)
&& !cp_parser_uncommitted_to_tentative_parse_p (parser))
error_at (token->location,
"typedef-name %qD used as destructor declarator",
type_decl);
return build_nt (BIT_NOT_EXPR, TREE_TYPE (type_decl));
}
case CPP_KEYWORD:
if (token->keyword == RID_OPERATOR)
{
cp_expr id;
cp_parser_parse_tentatively (parser);
id = cp_parser_template_id (parser, template_keyword_p,
true,
none_type,
declarator_p);
if (cp_parser_parse_definitely (parser))
return id;
cp_parser_parse_tentatively (parser);
id = cp_parser_operator_function_id (parser);
if (!cp_parser_parse_definitely (parser))
id = cp_parser_conversion_function_id (parser);
return id;
}
default:
if (optional_p)
return NULL_TREE;
cp_parser_error (parser, "expected unqualified-id");
return error_mark_node;
}
}
static tree
cp_parser_nested_name_specifier_opt (cp_parser *parser,
bool typename_keyword_p,
bool check_dependency_p,
bool type_p,
bool is_declaration,
bool template_keyword_p )
{
bool success = false;
cp_token_position start = 0;
cp_token *token;
if (cp_parser_uncommitted_to_tentative_parse_p (parser))
{
start = cp_lexer_token_position (parser->lexer, false);
push_deferring_access_checks (dk_deferred);
}
while (true)
{
tree new_scope;
tree old_scope;
tree saved_qualifying_scope;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_NESTED_NAME_SPECIFIER)
{
cp_parser_pre_parsed_nested_name_specifier (parser);
if (is_declaration
&& TREE_CODE (parser->scope) == TYPENAME_TYPE)
{
new_scope = resolve_typename_type (parser->scope,
false);
if (TREE_CODE (new_scope) != TYPENAME_TYPE)
parser->scope = new_scope;
}
success = true;
continue;
}
if (success && token->keyword == RID_TEMPLATE)
;
else if (token->type == CPP_TEMPLATE_ID)
;
else if (token_is_decltype (token))
;
else
{
if (token->type != CPP_NAME)
break;
token = cp_lexer_peek_nth_token (parser->lexer, 2);
if (token->type == CPP_COLON
&& parser->colon_corrects_to_scope_p
&& cp_lexer_peek_nth_token (parser->lexer, 3)->type == CPP_NAME)
{
gcc_rich_location richloc (token->location);
richloc.add_fixit_replace ("::");
error_at (&richloc,
"found %<:%> in nested-name-specifier, "
"expected %<::%>");
token->type = CPP_SCOPE;
}
if (token->type != CPP_SCOPE
&& !cp_parser_nth_token_starts_template_argument_list_p
(parser, 2))
break;
}
cp_parser_parse_tentatively (parser);
if (success)
template_keyword_p = cp_parser_optional_template_keyword (parser);
old_scope = parser->scope;
saved_qualifying_scope = parser->qualifying_scope;
if (is_declaration
&& !typename_keyword_p
&& parser->scope
&& TREE_CODE (parser->scope) == TYPENAME_TYPE)
parser->scope = resolve_typename_type (parser->scope,
false);
new_scope
= cp_parser_qualifying_entity (parser,
typename_keyword_p,
template_keyword_p,
check_dependency_p,
type_p,
is_declaration);
cp_parser_require (parser, CPP_SCOPE, RT_SCOPE);
if (!cp_parser_parse_definitely (parser))
{
bool error_p = false;
parser->scope = old_scope;
parser->qualifying_scope = saved_qualifying_scope;
if (cp_lexer_next_token_is (parser->lexer, CPP_DECLTYPE)
&& (cp_lexer_peek_nth_token (parser->lexer, 2)->type
== CPP_SCOPE))
{
token = cp_lexer_consume_token (parser->lexer);
error_at (token->location, "decltype evaluates to %qT, "
"which is not a class or enumeration type",
token->u.tree_check_value->value);
parser->scope = error_mark_node;
error_p = true;
success = true;
cp_lexer_consume_token (parser->lexer);
}
if (cp_lexer_next_token_is (parser->lexer, CPP_TEMPLATE_ID)
&& cp_lexer_nth_token_is (parser->lexer, 2, CPP_SCOPE))
{
token = cp_lexer_peek_token (parser->lexer);
tree tid = token->u.tree_check_value->value;
if (TREE_CODE (tid) == TEMPLATE_ID_EXPR
&& TREE_CODE (TREE_OPERAND (tid, 0)) != IDENTIFIER_NODE)
{
tree tmpl = NULL_TREE;
if (is_overloaded_fn (tid))
{
tree fns = get_fns (tid);
if (OVL_SINGLE_P (fns))
tmpl = OVL_FIRST (fns);
error_at (token->location, "function template-id %qD "
"in nested-name-specifier", tid);
}
else
{
tmpl = TREE_OPERAND (tid, 0);
gcc_assert (variable_template_p (tmpl));
error_at (token->location, "variable template-id %qD "
"in nested-name-specifier", tid);
}
if (tmpl)
inform (DECL_SOURCE_LOCATION (tmpl),
"%qD declared here", tmpl);
parser->scope = error_mark_node;
error_p = true;
success = true;
cp_lexer_consume_token (parser->lexer);
cp_lexer_consume_token (parser->lexer);
}
}
if (cp_parser_uncommitted_to_tentative_parse_p (parser))
break;
while (cp_lexer_next_token_is (parser->lexer, CPP_NAME)
&& (cp_lexer_peek_nth_token (parser->lexer, 2)->type
== CPP_SCOPE)
&& (cp_lexer_peek_nth_token (parser->lexer, 3)->type
!= CPP_COMPL))
{
token = cp_lexer_consume_token (parser->lexer);
if (!error_p)
{
if (!token->error_reported)
{
tree decl;
tree ambiguous_decls;
decl = cp_parser_lookup_name (parser, token->u.value,
none_type,
false,
false,
true,
&ambiguous_decls,
token->location);
if (TREE_CODE (decl) == TEMPLATE_DECL)
error_at (token->location,
"%qD used without template parameters",
decl);
else if (ambiguous_decls)
{
if (cp_parser_uncommitted_to_tentative_parse_p
(parser))
{
error_at (token->location,
"reference to %qD is ambiguous",
token->u.value);
print_candidates (ambiguous_decls);
}
decl = error_mark_node;
}
else
{
if (cxx_dialect != cxx98)
cp_parser_name_lookup_error
(parser, token->u.value, decl, NLE_NOT_CXX98,
token->location);
else
cp_parser_name_lookup_error
(parser, token->u.value, decl, NLE_CXX98,
token->location);
}
}
parser->scope = error_mark_node;
error_p = true;
success = true;
}
cp_lexer_consume_token (parser->lexer);
}
break;
}
success = true;
if (TREE_CODE (new_scope) == TYPE_DECL)
new_scope = TREE_TYPE (new_scope);
if (template_keyword_p
&& !(CLASS_TYPE_P (new_scope)
&& ((CLASSTYPE_USE_TEMPLATE (new_scope)
&& PRIMARY_TEMPLATE_P (CLASSTYPE_TI_TEMPLATE (new_scope)))
|| CLASSTYPE_IS_TEMPLATE (new_scope)))
&& !(TREE_CODE (new_scope) == TYPENAME_TYPE
&& (TREE_CODE (TYPENAME_TYPE_FULLNAME (new_scope))
== TEMPLATE_ID_EXPR)))
permerror (input_location, TYPE_P (new_scope)
? G_("%qT is not a template")
: G_("%qD is not a template"),
new_scope);
if (TYPE_P (new_scope)
&& !COMPLETE_TYPE_P (new_scope)
&& !dependent_type_p (new_scope))
{
new_scope = complete_type (new_scope);
if (!COMPLETE_TYPE_P (new_scope)
&& currently_open_class (new_scope))
new_scope = TYPE_MAIN_VARIANT (new_scope);
}
parser->scope = new_scope;
}
if (success && start)
{
cp_token *token;
token = cp_lexer_token_at (parser->lexer, start);
token->type = CPP_NESTED_NAME_SPECIFIER;
token->u.tree_check_value = ggc_cleared_alloc<struct tree_check> ();
token->u.tree_check_value->value = parser->scope;
token->u.tree_check_value->checks = get_deferred_access_checks ();
token->u.tree_check_value->qualifying_scope =
parser->qualifying_scope;
token->keyword = RID_MAX;
cp_lexer_purge_tokens_after (parser->lexer, start);
}
if (start)
pop_to_parent_deferring_access_checks ();
return success ? parser->scope : NULL_TREE;
}
static tree
cp_parser_nested_name_specifier (cp_parser *parser,
bool typename_keyword_p,
bool check_dependency_p,
bool type_p,
bool is_declaration)
{
tree scope;
scope = cp_parser_nested_name_specifier_opt (parser,
typename_keyword_p,
check_dependency_p,
type_p,
is_declaration);
if (!scope)
{
cp_parser_error (parser, "expected nested-name-specifier");
parser->scope = NULL_TREE;
}
return scope;
}
static tree
cp_parser_qualifying_entity (cp_parser *parser,
bool typename_keyword_p,
bool template_keyword_p,
bool check_dependency_p,
bool type_p,
bool is_declaration)
{
tree saved_scope;
tree saved_qualifying_scope;
tree saved_object_scope;
tree scope;
bool only_class_p;
bool successful_parse_p;
if (cp_lexer_next_token_is_decltype (parser->lexer))
{
scope = cp_parser_decltype (parser);
if (TREE_CODE (scope) != ENUMERAL_TYPE
&& !MAYBE_CLASS_TYPE_P (scope))
{
cp_parser_simulate_error (parser);
return error_mark_node;
}
if (TYPE_NAME (scope))
scope = TYPE_NAME (scope);
return scope;
}
saved_scope = parser->scope;
saved_qualifying_scope = parser->qualifying_scope;
saved_object_scope = parser->object_scope;
only_class_p = template_keyword_p 
|| (saved_scope && TYPE_P (saved_scope) && cxx_dialect == cxx98);
if (!only_class_p)
cp_parser_parse_tentatively (parser);
scope = cp_parser_class_name (parser,
typename_keyword_p,
template_keyword_p,
type_p ? class_type : none_type,
check_dependency_p,
false,
is_declaration,
cxx_dialect > cxx98);
successful_parse_p = only_class_p || cp_parser_parse_definitely (parser);
if (!only_class_p && !successful_parse_p)
{
parser->scope = saved_scope;
parser->qualifying_scope = saved_qualifying_scope;
parser->object_scope = saved_object_scope;
if (cp_lexer_next_token_is_not (parser->lexer, CPP_NAME)
|| cp_lexer_peek_nth_token (parser->lexer, 2)->type != CPP_SCOPE)
return error_mark_node;
scope = cp_parser_namespace_name (parser);
}
return scope;
}
static bool
cp_parser_compound_literal_p (cp_parser *parser)
{
cp_lexer_save_tokens (parser->lexer);
bool compound_literal_p
= (cp_parser_skip_to_closing_parenthesis (parser, false, false,
true)
&& cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE));
cp_lexer_rollback_tokens (parser->lexer);
return compound_literal_p;
}
static bool
literal_integer_zerop (const_tree expr)
{
STRIP_ANY_LOCATION_WRAPPER (expr);
return integer_zerop (expr);
}
static cp_expr
cp_parser_postfix_expression (cp_parser *parser, bool address_p, bool cast_p,
bool member_access_only_p, bool decltype_p,
cp_id_kind * pidk_return)
{
cp_token *token;
location_t loc;
enum rid keyword;
cp_id_kind idk = CP_ID_KIND_NONE;
cp_expr postfix_expression = NULL_TREE;
bool is_member_access = false;
token = cp_lexer_peek_token (parser->lexer);
loc = token->location;
location_t start_loc = get_range_from_loc (line_table, loc).m_start;
keyword = token->keyword;
switch (keyword)
{
case RID_DYNCAST:
case RID_STATCAST:
case RID_REINTCAST:
case RID_CONSTCAST:
{
tree type;
cp_expr expression;
const char *saved_message;
bool saved_in_type_id_in_expr_p;
cp_lexer_consume_token (parser->lexer);
saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in casts");
cp_parser_require (parser, CPP_LESS, RT_LESS);
saved_in_type_id_in_expr_p = parser->in_type_id_in_expr_p;
parser->in_type_id_in_expr_p = true;
type = cp_parser_type_id (parser);
parser->in_type_id_in_expr_p = saved_in_type_id_in_expr_p;
cp_parser_require (parser, CPP_GREATER, RT_GREATER);
parser->type_definition_forbidden_message = saved_message;
bool saved_greater_than_is_operator_p
= parser->greater_than_is_operator_p;
parser->greater_than_is_operator_p = true;
matching_parens parens;
parens.require_open (parser);
expression = cp_parser_expression (parser, & idk, true);
cp_token *close_paren = cp_parser_require (parser, CPP_CLOSE_PAREN,
RT_CLOSE_PAREN);
location_t end_loc = close_paren ?
close_paren->location : UNKNOWN_LOCATION;
parser->greater_than_is_operator_p
= saved_greater_than_is_operator_p;
if (!cast_valid_in_integral_constant_expression_p (type)
&& cp_parser_non_integral_constant_expression (parser, NIC_CAST))
{
postfix_expression = error_mark_node;
break;
}
switch (keyword)
{
case RID_DYNCAST:
postfix_expression
= build_dynamic_cast (type, expression, tf_warning_or_error);
break;
case RID_STATCAST:
postfix_expression
= build_static_cast (type, expression, tf_warning_or_error);
break;
case RID_REINTCAST:
postfix_expression
= build_reinterpret_cast (type, expression, 
tf_warning_or_error);
break;
case RID_CONSTCAST:
postfix_expression
= build_const_cast (type, expression, tf_warning_or_error);
break;
default:
gcc_unreachable ();
}
location_t cp_cast_loc = make_location (start_loc, start_loc, end_loc);
postfix_expression.set_location (cp_cast_loc);
}
break;
case RID_TYPEID:
{
tree type;
const char *saved_message;
bool saved_in_type_id_in_expr_p;
cp_lexer_consume_token (parser->lexer);
matching_parens parens;
parens.require_open (parser);
saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in a %<typeid%> expression");
cp_parser_parse_tentatively (parser);
saved_in_type_id_in_expr_p = parser->in_type_id_in_expr_p;
parser->in_type_id_in_expr_p = true;
type = cp_parser_type_id (parser);
parser->in_type_id_in_expr_p = saved_in_type_id_in_expr_p;
cp_token *close_paren = parens.require_close (parser);
if (cp_parser_parse_definitely (parser))
postfix_expression = get_typeid (type, tf_warning_or_error);
else
{
tree expression;
expression = cp_parser_expression (parser, & idk);
postfix_expression = build_typeid (expression, tf_warning_or_error);
close_paren = parens.require_close (parser);
}
parser->type_definition_forbidden_message = saved_message;
if (cp_parser_non_integral_constant_expression (parser, NIC_TYPEID))
postfix_expression = error_mark_node;
if (close_paren)
{
location_t typeid_loc
= make_location (start_loc, start_loc, close_paren->location);
postfix_expression.set_location (typeid_loc);
postfix_expression.maybe_add_location_wrapper ();
}
}
break;
case RID_TYPENAME:
{
tree type;
++parser->prevent_constrained_type_specifiers;
type = cp_parser_elaborated_type_specifier (parser,
false,
false);
--parser->prevent_constrained_type_specifiers;
postfix_expression = cp_parser_functional_cast (parser, type);
}
break;
case RID_ADDRESSOF:
case RID_BUILTIN_SHUFFLE:
case RID_BUILTIN_LAUNDER:
{
vec<tree, va_gc> *vec;
unsigned int i;
tree p;
cp_lexer_consume_token (parser->lexer);
vec = cp_parser_parenthesized_expression_list (parser, non_attr,
false, true,
NULL);
if (vec == NULL)
{
postfix_expression = error_mark_node;
break;
}
FOR_EACH_VEC_ELT (*vec, i, p)
mark_exp_read (p);
switch (keyword)
{
case RID_ADDRESSOF:
if (vec->length () == 1)
postfix_expression
= cp_build_addressof (loc, (*vec)[0], tf_warning_or_error);
else
{
error_at (loc, "wrong number of arguments to "
"%<__builtin_addressof%>");
postfix_expression = error_mark_node;
}
break;
case RID_BUILTIN_LAUNDER:
if (vec->length () == 1)
postfix_expression = finish_builtin_launder (loc, (*vec)[0],
tf_warning_or_error);
else
{
error_at (loc, "wrong number of arguments to "
"%<__builtin_launder%>");
postfix_expression = error_mark_node;
}
break;
case RID_BUILTIN_SHUFFLE:
if (vec->length () == 2)
postfix_expression
= build_x_vec_perm_expr (loc, (*vec)[0], NULL_TREE,
(*vec)[1], tf_warning_or_error);
else if (vec->length () == 3)
postfix_expression
= build_x_vec_perm_expr (loc, (*vec)[0], (*vec)[1],
(*vec)[2], tf_warning_or_error);
else
{
error_at (loc, "wrong number of arguments to "
"%<__builtin_shuffle%>");
postfix_expression = error_mark_node;
}
break;
default:
gcc_unreachable ();
}
break;
}
default:
{
tree type;
cp_parser_parse_tentatively (parser);
++parser->prevent_constrained_type_specifiers;
type = cp_parser_simple_type_specifier (parser,
NULL,
CP_PARSER_FLAGS_NONE);
--parser->prevent_constrained_type_specifiers;
if (!cp_parser_error_occurred (parser))
postfix_expression
= cp_parser_functional_cast (parser, type);
if (cp_parser_parse_definitely (parser))
break;
if (cp_parser_allow_gnu_extensions_p (parser)
&& cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
cp_expr initializer = NULL_TREE;
cp_parser_parse_tentatively (parser);
matching_parens parens;
parens.consume_open (parser);
if (!cp_parser_compound_literal_p (parser))
cp_parser_simulate_error (parser);
else
{
bool saved_in_type_id_in_expr_p = parser->in_type_id_in_expr_p;
parser->in_type_id_in_expr_p = true;
type = cp_parser_type_id (parser);
parser->in_type_id_in_expr_p = saved_in_type_id_in_expr_p;
parens.require_close (parser);
}
if (!cp_parser_error_occurred (parser))
{
bool non_constant_p;
initializer = cp_parser_braced_list (parser,
&non_constant_p);
}
if (cp_parser_parse_definitely (parser))
{
pedwarn (input_location, OPT_Wpedantic,
"ISO C++ forbids compound-literals");
if (cp_parser_non_integral_constant_expression (parser,
NIC_NCC))
{
postfix_expression = error_mark_node;
break;
}
postfix_expression
= finish_compound_literal (type, initializer,
tf_warning_or_error, fcl_c99);
postfix_expression.set_location (initializer.get_location ());
break;
}
}
postfix_expression
= cp_parser_primary_expression (parser, address_p, cast_p,
false,
decltype_p,
&idk);
}
break;
}
while (true)
{
if (idk == CP_ID_KIND_UNQUALIFIED
&& identifier_p (postfix_expression)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_OPEN_PAREN))
postfix_expression
= unqualified_name_lookup_error (postfix_expression);
token = cp_lexer_peek_token (parser->lexer);
switch (token->type)
{
case CPP_OPEN_SQUARE:
if (cp_next_tokens_can_be_std_attribute_p (parser))
{
cp_parser_error (parser,
"two consecutive %<[%> shall "
"only introduce an attribute");
return error_mark_node;
}
postfix_expression
= cp_parser_postfix_open_square_expression (parser,
postfix_expression,
false,
decltype_p);
postfix_expression.set_range (start_loc,
postfix_expression.get_location ());
idk = CP_ID_KIND_NONE;
is_member_access = false;
break;
case CPP_OPEN_PAREN:
{
bool koenig_p;
bool is_builtin_constant_p;
bool saved_integral_constant_expression_p = false;
bool saved_non_integral_constant_expression_p = false;
tsubst_flags_t complain = complain_flags (decltype_p);
vec<tree, va_gc> *args;
location_t close_paren_loc = UNKNOWN_LOCATION;
is_member_access = false;
is_builtin_constant_p
= DECL_IS_BUILTIN_CONSTANT_P (postfix_expression);
if (is_builtin_constant_p)
{
saved_integral_constant_expression_p
= parser->integral_constant_expression_p;
saved_non_integral_constant_expression_p
= parser->non_integral_constant_expression_p;
parser->integral_constant_expression_p = false;
}
args = (cp_parser_parenthesized_expression_list
(parser, non_attr,
false, true,
NULL,
&close_paren_loc,
true));
if (is_builtin_constant_p)
{
parser->integral_constant_expression_p
= saved_integral_constant_expression_p;
parser->non_integral_constant_expression_p
= saved_non_integral_constant_expression_p;
}
if (args == NULL)
{
postfix_expression = error_mark_node;
break;
}
if (! builtin_valid_in_constant_expr_p (postfix_expression)
&& cp_parser_non_integral_constant_expression (parser,
NIC_FUNC_CALL))
{
postfix_expression = error_mark_node;
release_tree_vector (args);
break;
}
koenig_p = false;
if (idk == CP_ID_KIND_UNQUALIFIED
|| idk == CP_ID_KIND_TEMPLATE_ID)
{
if (identifier_p (postfix_expression))
{
if (!args->is_empty ())
{
koenig_p = true;
if (!any_type_dependent_arguments_p (args))
postfix_expression
= perform_koenig_lookup (postfix_expression, args,
complain);
}
else
postfix_expression
= unqualified_fn_lookup_error (postfix_expression);
}
else if (!args->is_empty ()
&& is_overloaded_fn (postfix_expression))
{
tree fn = get_first_fn (postfix_expression);
fn = STRIP_TEMPLATE (fn);
if (!((TREE_CODE (fn) == USING_DECL && DECL_DEPENDENT_P (fn))
|| DECL_FUNCTION_MEMBER_P (fn)
|| DECL_LOCAL_FUNCTION_P (fn)))
{
koenig_p = true;
if (!any_type_dependent_arguments_p (args))
postfix_expression
= perform_koenig_lookup (postfix_expression, args,
complain);
}
}
}
if (TREE_CODE (postfix_expression) == FUNCTION_DECL
&& DECL_BUILT_IN_CLASS (postfix_expression) == BUILT_IN_NORMAL
&& DECL_FUNCTION_CODE (postfix_expression) == BUILT_IN_MEMSET
&& vec_safe_length (args) == 3)
{
tree arg0 = (*args)[0];
tree arg1 = (*args)[1];
tree arg2 = (*args)[2];
int literal_mask = ((literal_integer_zerop (arg1) << 1)
| (literal_integer_zerop (arg2) << 2));
warn_for_memset (input_location, arg0, arg2, literal_mask);
}
if (TREE_CODE (postfix_expression) == COMPONENT_REF)
{
tree instance = TREE_OPERAND (postfix_expression, 0);
tree fn = TREE_OPERAND (postfix_expression, 1);
if (processing_template_decl
&& (type_dependent_object_expression_p (instance)
|| (!BASELINK_P (fn)
&& TREE_CODE (fn) != FIELD_DECL)
|| type_dependent_expression_p (fn)
|| any_type_dependent_arguments_p (args)))
{
maybe_generic_this_capture (instance, fn);
postfix_expression
= build_min_nt_call_vec (postfix_expression, args);
release_tree_vector (args);
break;
}
if (BASELINK_P (fn))
{
postfix_expression
= (build_new_method_call
(instance, fn, &args, NULL_TREE,
(idk == CP_ID_KIND_QUALIFIED
? LOOKUP_NORMAL|LOOKUP_NONVIRTUAL
: LOOKUP_NORMAL),
NULL,
complain));
}
else
postfix_expression
= finish_call_expr (postfix_expression, &args,
false,
false,
complain);
}
else if (TREE_CODE (postfix_expression) == OFFSET_REF
|| TREE_CODE (postfix_expression) == MEMBER_REF
|| TREE_CODE (postfix_expression) == DOTSTAR_EXPR)
postfix_expression = (build_offset_ref_call_from_tree
(postfix_expression, &args,
complain));
else if (idk == CP_ID_KIND_QUALIFIED)
postfix_expression
= finish_call_expr (postfix_expression, &args,
true,
koenig_p,
complain);
else
postfix_expression
= finish_call_expr (postfix_expression, &args,
false,
koenig_p,
complain);
if (close_paren_loc != UNKNOWN_LOCATION)
{
location_t combined_loc = make_location (token->location,
start_loc,
close_paren_loc);
postfix_expression.set_location (combined_loc);
}
idk = CP_ID_KIND_NONE;
release_tree_vector (args);
}
break;
case CPP_DOT:
case CPP_DEREF:
cp_lexer_consume_token (parser->lexer);
postfix_expression
= cp_parser_postfix_dot_deref_expression (parser, token->type,
postfix_expression,
false, &idk, loc);
is_member_access = true;
break;
case CPP_PLUS_PLUS:
cp_lexer_consume_token (parser->lexer);
postfix_expression
= finish_increment_expr (postfix_expression,
POSTINCREMENT_EXPR);
if (cp_parser_non_integral_constant_expression (parser, NIC_INC))
postfix_expression = error_mark_node;
idk = CP_ID_KIND_NONE;
is_member_access = false;
break;
case CPP_MINUS_MINUS:
cp_lexer_consume_token (parser->lexer);
postfix_expression
= finish_increment_expr (postfix_expression,
POSTDECREMENT_EXPR);
if (cp_parser_non_integral_constant_expression (parser, NIC_DEC))
postfix_expression = error_mark_node;
idk = CP_ID_KIND_NONE;
is_member_access = false;
break;
default:
if (pidk_return != NULL)
* pidk_return = idk;
if (member_access_only_p)
return is_member_access
? postfix_expression
: cp_expr (error_mark_node);
else
return postfix_expression;
}
}
gcc_unreachable ();
return error_mark_node;
}
static tree
cp_parser_postfix_open_square_expression (cp_parser *parser,
tree postfix_expression,
bool for_offsetof,
bool decltype_p)
{
tree index = NULL_TREE;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
bool saved_greater_than_is_operator_p;
cp_lexer_consume_token (parser->lexer);
saved_greater_than_is_operator_p = parser->greater_than_is_operator_p;
parser->greater_than_is_operator_p = true;
if (for_offsetof)
index = cp_parser_constant_expression (parser);
else
{
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
bool expr_nonconst_p;
cp_lexer_set_source_position (parser->lexer);
maybe_warn_cpp0x (CPP0X_INITIALIZER_LISTS);
index = cp_parser_braced_list (parser, &expr_nonconst_p);
}
else
index = cp_parser_expression (parser);
}
parser->greater_than_is_operator_p = saved_greater_than_is_operator_p;
cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE);
postfix_expression = grok_array_decl (loc, postfix_expression,
index, decltype_p);
if (!for_offsetof
&& (cp_parser_non_integral_constant_expression (parser, NIC_ARRAY_REF)))
postfix_expression = error_mark_node;
return postfix_expression;
}
bool
cp_parser_dot_deref_incomplete (tree *scope, cp_expr *postfix_expression,
bool *dependent_p)
{
diagnostic_t kind = (processing_template_decl
&& MAYBE_CLASS_TYPE_P (*scope) ? DK_PEDWARN : DK_ERROR);
switch (TREE_CODE (*postfix_expression))
{
case CAST_EXPR:
case REINTERPRET_CAST_EXPR:
case CONST_CAST_EXPR:
case STATIC_CAST_EXPR:
case DYNAMIC_CAST_EXPR:
case IMPLICIT_CONV_EXPR:
case VIEW_CONVERT_EXPR:
case NON_LVALUE_EXPR:
kind = DK_ERROR;
break;
case OVERLOAD:
kind = DK_IGNORED;
break;
default:
if (!EXPR_P (*postfix_expression))
kind = DK_ERROR;
break;
}
if (kind == DK_IGNORED)
return false;
location_t exploc = location_of (*postfix_expression);
cxx_incomplete_type_diagnostic (exploc, *postfix_expression, *scope, kind);
if (!MAYBE_CLASS_TYPE_P (*scope))
return true;
if (kind == DK_ERROR)
*scope = *postfix_expression = error_mark_node;
else if (processing_template_decl)
{
*dependent_p = true;
*scope = TREE_TYPE (*postfix_expression) = NULL_TREE;
}
return false;
}
static tree
cp_parser_postfix_dot_deref_expression (cp_parser *parser,
enum cpp_ttype token_type,
cp_expr postfix_expression,
bool for_offsetof, cp_id_kind *idk,
location_t location)
{
tree name;
bool dependent_p;
bool pseudo_destructor_p;
tree scope = NULL_TREE;
location_t start_loc = postfix_expression.get_start ();
if (token_type == CPP_DEREF)
postfix_expression = build_x_arrow (location, postfix_expression,
tf_warning_or_error);
dependent_p = type_dependent_object_expression_p (postfix_expression);
parser->scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
parser->object_scope = NULL_TREE;
*idk = CP_ID_KIND_NONE;
if (!dependent_p)
{
scope = TREE_TYPE (postfix_expression);
scope = non_reference (scope);
if (postfix_expression != current_class_ref
&& scope != error_mark_node
&& !currently_open_class (scope))
{
scope = complete_type (scope);
if (!COMPLETE_TYPE_P (scope)
&& cp_parser_dot_deref_incomplete (&scope, &postfix_expression,
&dependent_p))
return error_mark_node;
}
if (!dependent_p)
{
parser->context->object_type = scope;
if (!scope)
scope = error_mark_node;
if (scope == error_mark_node)
postfix_expression = error_mark_node;
}
}
if (dependent_p)
parser->context->object_type = unknown_type_node;
pseudo_destructor_p = false;
if ((scope && SCALAR_TYPE_P (scope)) || dependent_p)
{
tree s;
tree type;
cp_parser_parse_tentatively (parser);
s = NULL_TREE;
cp_parser_pseudo_destructor_name (parser, postfix_expression,
&s, &type);
if (dependent_p
&& (cp_parser_error_occurred (parser)
|| !SCALAR_TYPE_P (type)))
cp_parser_abort_tentative_parse (parser);
else if (cp_parser_parse_definitely (parser))
{
pseudo_destructor_p = true;
postfix_expression
= finish_pseudo_destructor_expr (postfix_expression,
s, type, location);
}
}
if (!pseudo_destructor_p)
{
bool template_p;
cp_token *token = cp_lexer_peek_token (parser->lexer);
name = (cp_parser_id_expression
(parser,
cp_parser_optional_template_keyword (parser),
true,
&template_p,
false,
false));
if (parser->scope)
*idk = CP_ID_KIND_QUALIFIED;
if (TREE_CODE (name) == TYPE_DECL)
{
error_at (token->location, "invalid use of %qD", name);
postfix_expression = error_mark_node;
}
else
{
if (name != error_mark_node && !BASELINK_P (name) && parser->scope)
{
if (TREE_CODE (parser->scope) == NAMESPACE_DECL)
{
error_at (token->location, "%<%D::%D%> is not a class member",
parser->scope, name);
postfix_expression = error_mark_node;
}
else
name = build_qualified_name (NULL_TREE,
parser->scope,
name,
template_p);
parser->scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
parser->object_scope = NULL_TREE;
}
if (parser->scope && name && BASELINK_P (name))
adjust_result_of_qualified_name_lookup
(name, parser->scope, scope);
postfix_expression
= finish_class_member_access_expr (postfix_expression, name,
template_p, 
tf_warning_or_error);
location_t end_loc
= get_finish (cp_lexer_previous_token (parser->lexer)->location);
location_t combined_loc
= make_location (input_location, start_loc, end_loc);
protected_set_expr_location (postfix_expression, combined_loc);
}
}
parser->context->object_type = NULL_TREE;
if (!for_offsetof
&& (cp_parser_non_integral_constant_expression
(parser, token_type == CPP_DEREF ? NIC_ARROW : NIC_POINT)))
postfix_expression = error_mark_node;
return postfix_expression;
}
static vec<tree, va_gc> *
cp_parser_parenthesized_expression_list (cp_parser* parser,
int is_attribute_list,
bool cast_p,
bool allow_expansion_p,
bool *non_constant_p,
location_t *close_paren_loc,
bool wrap_locations_p)
{
vec<tree, va_gc> *expression_list;
bool fold_expr_p = is_attribute_list != non_attr;
tree identifier = NULL_TREE;
bool saved_greater_than_is_operator_p;
if (non_constant_p)
*non_constant_p = false;
matching_parens parens;
if (!parens.require_open (parser))
return NULL;
expression_list = make_tree_vector ();
saved_greater_than_is_operator_p
= parser->greater_than_is_operator_p;
parser->greater_than_is_operator_p = true;
cp_expr expr (NULL_TREE);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_CLOSE_PAREN))
while (true)
{
if (is_attribute_list == id_attr
&& cp_lexer_peek_token (parser->lexer)->type == CPP_NAME)
{
cp_token *token;
token = cp_lexer_consume_token (parser->lexer);
identifier = token->u.value;
}
else
{
bool expr_non_constant_p;
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
cp_lexer_set_source_position (parser->lexer);
maybe_warn_cpp0x (CPP0X_INITIALIZER_LISTS);
expr = cp_parser_braced_list (parser, &expr_non_constant_p);
if (non_constant_p && expr_non_constant_p)
*non_constant_p = true;
}
else if (non_constant_p)
{
expr = (cp_parser_constant_expression
(parser, true,
&expr_non_constant_p));
if (expr_non_constant_p)
*non_constant_p = true;
}
else
expr = cp_parser_assignment_expression (parser, NULL,
cast_p);
if (fold_expr_p)
expr = instantiate_non_dependent_expr (expr);
if (allow_expansion_p
&& cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
cp_lexer_consume_token (parser->lexer);
expr = make_pack_expansion (expr);
}
if (wrap_locations_p)
expr.maybe_add_location_wrapper ();
vec_safe_push (expression_list, expr.get_value ());
if (expr == error_mark_node)
goto skip_comma;
}
is_attribute_list = non_attr;
get_comma:;
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA))
break;
cp_lexer_consume_token (parser->lexer);
}
if (close_paren_loc)
*close_paren_loc = cp_lexer_peek_token (parser->lexer)->location;
if (!parens.require_close (parser))
{
int ending;
skip_comma:;
ending = cp_parser_skip_to_closing_parenthesis (parser,
true,
true,
true);
if (ending < 0)
goto get_comma;
if (!ending)
{
parser->greater_than_is_operator_p
= saved_greater_than_is_operator_p;
return NULL;
}
}
parser->greater_than_is_operator_p
= saved_greater_than_is_operator_p;
if (identifier)
vec_safe_insert (expression_list, 0, identifier);
return expression_list;
}
static void
cp_parser_pseudo_destructor_name (cp_parser* parser,
tree object,
tree* scope,
tree* type)
{
bool nested_name_specifier_p;
if (cp_lexer_next_token_is (parser->lexer, CPP_COMPL)
&& cp_lexer_nth_token_is_keyword (parser->lexer, 2, RID_AUTO)
&& !type_dependent_expression_p (object))
{
if (cxx_dialect < cxx14)
pedwarn (input_location, 0,
"%<~auto%> only available with "
"-std=c++14 or -std=gnu++14");
cp_lexer_consume_token (parser->lexer);
cp_lexer_consume_token (parser->lexer);
*scope = NULL_TREE;
*type = TREE_TYPE (object);
return;
}
*type = error_mark_node;
cp_parser_global_scope_opt (parser, true);
nested_name_specifier_p
= (cp_parser_nested_name_specifier_opt (parser,
false,
true,
false,
false)
!= NULL_TREE);
if (nested_name_specifier_p
&& cp_lexer_next_token_is_keyword (parser->lexer, RID_TEMPLATE))
{
cp_lexer_consume_token (parser->lexer);
cp_parser_template_id (parser,
true,
false,
class_type,
true);
cp_parser_require (parser, CPP_SCOPE, RT_SCOPE);
}
else if (cp_lexer_next_token_is_not (parser->lexer, CPP_COMPL))
{
if (cp_lexer_peek_token (parser->lexer)->type != CPP_NAME
|| cp_lexer_peek_nth_token (parser->lexer, 2)->type != CPP_SCOPE
|| cp_lexer_peek_nth_token (parser->lexer, 3)->type != CPP_COMPL)
{
cp_parser_error (parser, "non-scalar type");
return;
}
*scope = TREE_TYPE (cp_parser_nonclass_name (parser));
if (*scope == error_mark_node)
return;
cp_parser_require (parser, CPP_SCOPE, RT_SCOPE);
}
else
*scope = NULL_TREE;
cp_parser_require (parser, CPP_COMPL, RT_COMPL);
if (!processing_template_decl && !cp_parser_error_occurred (parser))
cp_parser_commit_to_topmost_tentative_parse (parser);
*type = TREE_TYPE (cp_parser_nonclass_name (parser));
}
static cp_expr
cp_parser_unary_expression (cp_parser *parser, cp_id_kind * pidk,
bool address_p, bool cast_p, bool decltype_p)
{
cp_token *token;
enum tree_code unary_operator;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_KEYWORD)
{
enum rid keyword = token->keyword;
switch (keyword)
{
case RID_ALIGNOF:
case RID_SIZEOF:
{
tree operand, ret;
enum tree_code op;
location_t start_loc = token->location;
op = keyword == RID_ALIGNOF ? ALIGNOF_EXPR : SIZEOF_EXPR;
bool std_alignof = id_equal (token->u.value, "alignof");
cp_lexer_consume_token (parser->lexer);
operand = cp_parser_sizeof_operand (parser, keyword);
if (TYPE_P (operand))
ret = cxx_sizeof_or_alignof_type (operand, op, std_alignof,
true);
else
{
if (std_alignof)
pedwarn (token->location, OPT_Wpedantic,
"ISO C++ does not allow %<alignof%> "
"with a non-type");
ret = cxx_sizeof_or_alignof_expr (operand, op, true);
}
if (op == SIZEOF_EXPR && ret != error_mark_node)
{
if (TREE_CODE (ret) != SIZEOF_EXPR || TYPE_P (operand))
{
if (!processing_template_decl && TYPE_P (operand))
{
ret = build_min (SIZEOF_EXPR, size_type_node,
build1 (NOP_EXPR, operand,
error_mark_node));
SIZEOF_EXPR_TYPE_P (ret) = 1;
}
else
ret = build_min (SIZEOF_EXPR, size_type_node, operand);
TREE_SIDE_EFFECTS (ret) = 0;
TREE_READONLY (ret) = 1;
}
}
location_t finish_loc
= cp_lexer_previous_token (parser->lexer)->location;
location_t compound_loc
= make_location (start_loc, start_loc, finish_loc);
cp_expr ret_expr (ret);
ret_expr.set_location (compound_loc);
ret_expr = ret_expr.maybe_add_location_wrapper ();
return ret_expr;
}
case RID_NEW:
return cp_parser_new_expression (parser);
case RID_DELETE:
return cp_parser_delete_expression (parser);
case RID_EXTENSION:
{
int saved_pedantic;
tree expr;
cp_parser_extension_opt (parser, &saved_pedantic);
expr = cp_parser_simple_cast_expression (parser);
pedantic = saved_pedantic;
return expr;
}
case RID_REALPART:
case RID_IMAGPART:
{
tree expression;
cp_lexer_consume_token (parser->lexer);
expression = cp_parser_simple_cast_expression (parser);
return build_x_unary_op (token->location,
(keyword == RID_REALPART
? REALPART_EXPR : IMAGPART_EXPR),
expression,
tf_warning_or_error);
}
break;
case RID_TRANSACTION_ATOMIC:
case RID_TRANSACTION_RELAXED:
return cp_parser_transaction_expression (parser, keyword);
case RID_NOEXCEPT:
{
tree expr;
const char *saved_message;
bool saved_integral_constant_expression_p;
bool saved_non_integral_constant_expression_p;
bool saved_greater_than_is_operator_p;
location_t start_loc = token->location;
cp_lexer_consume_token (parser->lexer);
matching_parens parens;
parens.require_open (parser);
saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in %<noexcept%> expressions");
saved_integral_constant_expression_p
= parser->integral_constant_expression_p;
saved_non_integral_constant_expression_p
= parser->non_integral_constant_expression_p;
parser->integral_constant_expression_p = false;
saved_greater_than_is_operator_p
= parser->greater_than_is_operator_p;
parser->greater_than_is_operator_p = true;
++cp_unevaluated_operand;
++c_inhibit_evaluation_warnings;
++cp_noexcept_operand;
expr = cp_parser_expression (parser);
--cp_noexcept_operand;
--c_inhibit_evaluation_warnings;
--cp_unevaluated_operand;
parser->greater_than_is_operator_p
= saved_greater_than_is_operator_p;
parser->integral_constant_expression_p
= saved_integral_constant_expression_p;
parser->non_integral_constant_expression_p
= saved_non_integral_constant_expression_p;
parser->type_definition_forbidden_message = saved_message;
location_t finish_loc
= cp_lexer_peek_token (parser->lexer)->location;
parens.require_close (parser);
location_t noexcept_loc
= make_location (start_loc, start_loc, finish_loc);
return cp_expr (finish_noexcept_expr (expr, tf_warning_or_error),
noexcept_loc);
}
default:
break;
}
}
if (cp_lexer_next_token_is (parser->lexer, CPP_SCOPE))
{
enum rid keyword;
keyword = cp_lexer_peek_nth_token (parser->lexer, 2)->keyword;
if (keyword == RID_NEW)
return cp_parser_new_expression (parser);
else if (keyword == RID_DELETE)
return cp_parser_delete_expression (parser);
}
unary_operator = cp_parser_unary_operator (token);
if (unary_operator == ERROR_MARK)
{
if (token->type == CPP_PLUS_PLUS)
unary_operator = PREINCREMENT_EXPR;
else if (token->type == CPP_MINUS_MINUS)
unary_operator = PREDECREMENT_EXPR;
else if (cp_parser_allow_gnu_extensions_p (parser)
&& token->type == CPP_AND_AND)
{
tree identifier;
tree expression;
location_t start_loc = token->location;
cp_lexer_consume_token (parser->lexer);
location_t finish_loc
= get_finish (cp_lexer_peek_token (parser->lexer)->location);
identifier = cp_parser_identifier (parser);
location_t combined_loc
= make_location (start_loc, start_loc, finish_loc);
expression = finish_label_address_expr (identifier, combined_loc);
if (cp_parser_non_integral_constant_expression (parser,
NIC_ADDR_LABEL))
expression = error_mark_node;
return expression;
}
}
if (unary_operator != ERROR_MARK)
{
cp_expr cast_expression;
cp_expr expression = error_mark_node;
non_integral_constant non_constant_p = NIC_NONE;
location_t loc = token->location;
tsubst_flags_t complain = complain_flags (decltype_p);
token = cp_lexer_consume_token (parser->lexer);
enum cpp_ttype op_ttype = cp_lexer_peek_token (parser->lexer)->type;
cast_expression
= cp_parser_cast_expression (parser,
unary_operator == ADDR_EXPR,
false,
false,
pidk);
loc = make_location (loc, loc, cast_expression.get_finish ());
switch (unary_operator)
{
case INDIRECT_REF:
non_constant_p = NIC_STAR;
expression = build_x_indirect_ref (loc, cast_expression,
RO_UNARY_STAR,
complain);
expression.set_location (loc);
break;
case ADDR_EXPR:
non_constant_p = NIC_ADDR;
case BIT_NOT_EXPR:
expression = build_x_unary_op (loc, unary_operator,
cast_expression,
complain);
expression.set_location (loc);
break;
case PREINCREMENT_EXPR:
case PREDECREMENT_EXPR:
non_constant_p = unary_operator == PREINCREMENT_EXPR
? NIC_PREINCREMENT : NIC_PREDECREMENT;
case NEGATE_EXPR:
if (unary_operator == NEGATE_EXPR && op_ttype == CPP_NUMBER
&& CONSTANT_CLASS_P (cast_expression)
&& !integer_zerop (cast_expression)
&& !TREE_OVERFLOW (cast_expression))
{
tree folded = fold_build1 (unary_operator,
TREE_TYPE (cast_expression),
cast_expression);
if (CONSTANT_CLASS_P (folded) && !TREE_OVERFLOW (folded))
{
expression = cp_expr (folded, loc);
break;
}
}
case UNARY_PLUS_EXPR:
case TRUTH_NOT_EXPR:
expression = finish_unary_op_expr (loc, unary_operator,
cast_expression, complain);
break;
default:
gcc_unreachable ();
}
if (non_constant_p != NIC_NONE
&& cp_parser_non_integral_constant_expression (parser,
non_constant_p))
expression = error_mark_node;
return expression;
}
return cp_parser_postfix_expression (parser, address_p, cast_p,
false,
decltype_p,
pidk);
}
static enum tree_code
cp_parser_unary_operator (cp_token* token)
{
switch (token->type)
{
case CPP_MULT:
return INDIRECT_REF;
case CPP_AND:
return ADDR_EXPR;
case CPP_PLUS:
return UNARY_PLUS_EXPR;
case CPP_MINUS:
return NEGATE_EXPR;
case CPP_NOT:
return TRUTH_NOT_EXPR;
case CPP_COMPL:
return BIT_NOT_EXPR;
default:
return ERROR_MARK;
}
}
static tree
cp_parser_new_expression (cp_parser* parser)
{
bool global_scope_p;
vec<tree, va_gc> *placement;
tree type;
vec<tree, va_gc> *initializer;
tree nelts = NULL_TREE;
tree ret;
location_t start_loc = cp_lexer_peek_token (parser->lexer)->location;
global_scope_p
= (cp_parser_global_scope_opt (parser,
false)
!= NULL_TREE);
cp_parser_require_keyword (parser, RID_NEW, RT_NEW);
cp_parser_parse_tentatively (parser);
placement = cp_parser_new_placement (parser);
if (!cp_parser_parse_definitely (parser))
{
if (placement != NULL)
release_tree_vector (placement);
placement = NULL;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
cp_token *token;
const char *saved_message = parser->type_definition_forbidden_message;
matching_parens parens;
parens.consume_open (parser);
parser->type_definition_forbidden_message
= G_("types may not be defined in a new-expression");
{
type_id_in_expr_sentinel s (parser);
type = cp_parser_type_id (parser);
}
parser->type_definition_forbidden_message = saved_message;
parens.require_close (parser);
token = cp_lexer_peek_token (parser->lexer);
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_SQUARE))
{
error_at (token->location,
"array bound forbidden after parenthesized type-id");
inform (token->location, 
"try removing the parentheses around the type-id");
cp_parser_direct_new_declarator (parser);
}
}
else
type = cp_parser_new_type_id (parser, &nelts);
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_OPEN_PAREN
|| token->type == CPP_OPEN_BRACE)
initializer = cp_parser_new_initializer (parser);
else
initializer = NULL;
if (cp_parser_non_integral_constant_expression (parser, NIC_NEW))
ret = error_mark_node;
else if ((ret = type_uses_auto (type))
&& !CLASS_PLACEHOLDER_TEMPLATE (ret)
&& (vec_safe_length (initializer) != 1
|| (BRACE_ENCLOSED_INITIALIZER_P ((*initializer)[0])
&& CONSTRUCTOR_NELTS ((*initializer)[0]) != 1)))
{
error_at (token->location,
"initialization of new-expression for type %<auto%> "
"requires exactly one element");
ret = error_mark_node;
}
else
{
cp_token *end_tok = cp_lexer_previous_token (parser->lexer);
location_t end_loc = get_finish (end_tok->location);
location_t combined_loc = make_location (start_loc, start_loc, end_loc);
ret = build_new (&placement, type, nelts, &initializer, global_scope_p,
tf_warning_or_error);
protected_set_expr_location (ret, combined_loc);
}
if (placement != NULL)
release_tree_vector (placement);
if (initializer != NULL)
release_tree_vector (initializer);
return ret;
}
static vec<tree, va_gc> *
cp_parser_new_placement (cp_parser* parser)
{
vec<tree, va_gc> *expression_list;
expression_list = (cp_parser_parenthesized_expression_list
(parser, non_attr, false,
true,
NULL));
if (expression_list && expression_list->is_empty ())
error ("expected expression-list or type-id");
return expression_list;
}
static tree
cp_parser_new_type_id (cp_parser* parser, tree *nelts)
{
cp_decl_specifier_seq type_specifier_seq;
cp_declarator *new_declarator;
cp_declarator *declarator;
cp_declarator *outer_declarator;
const char *saved_message;
saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in a new-type-id");
cp_parser_type_specifier_seq (parser, false,
false,
&type_specifier_seq);
parser->type_definition_forbidden_message = saved_message;
if (type_specifier_seq.type == error_mark_node)
return error_mark_node;
new_declarator = cp_parser_new_declarator_opt (parser);
*nelts = NULL_TREE;
declarator = new_declarator;
outer_declarator = NULL;
while (declarator && (declarator->kind == cdk_pointer
|| declarator->kind == cdk_ptrmem))
{
outer_declarator = declarator;
declarator = declarator->declarator;
}
while (declarator
&& declarator->kind == cdk_array
&& declarator->declarator
&& declarator->declarator->kind == cdk_array)
{
outer_declarator = declarator;
declarator = declarator->declarator;
}
if (declarator && declarator->kind == cdk_array)
{
*nelts = declarator->u.array.bounds;
if (*nelts == error_mark_node)
*nelts = integer_one_node;
if (outer_declarator)
outer_declarator->declarator = declarator->declarator;
else
new_declarator = NULL;
}
return groktypename (&type_specifier_seq, new_declarator, false);
}
static cp_declarator *
cp_parser_new_declarator_opt (cp_parser* parser)
{
enum tree_code code;
tree type, std_attributes = NULL_TREE;
cp_cv_quals cv_quals;  
cp_parser_parse_tentatively (parser);
code = cp_parser_ptr_operator (parser, &type, &cv_quals, &std_attributes);
if (cp_parser_parse_definitely (parser))
{
cp_declarator *declarator;
declarator = cp_parser_new_declarator_opt (parser);
declarator = cp_parser_make_indirect_declarator
(code, type, cv_quals, declarator, std_attributes);
return declarator;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_SQUARE))
return cp_parser_direct_new_declarator (parser);
return NULL;
}
static cp_declarator *
cp_parser_direct_new_declarator (cp_parser* parser)
{
cp_declarator *declarator = NULL;
while (true)
{
tree expression;
cp_token *token;
cp_parser_require (parser, CPP_OPEN_SQUARE, RT_OPEN_SQUARE);
token = cp_lexer_peek_token (parser->lexer);
expression = cp_parser_expression (parser);
if (!processing_template_decl)
{
expression
= build_expr_type_conversion (WANT_INT | WANT_ENUM,
expression,
true);
if (!expression)
{
error_at (token->location,
"expression in new-declarator must have integral "
"or enumeration type");
expression = error_mark_node;
}
}
cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE);
declarator = make_array_declarator (declarator, expression);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_OPEN_SQUARE))
break;
}
return declarator;
}
static vec<tree, va_gc> *
cp_parser_new_initializer (cp_parser* parser)
{
vec<tree, va_gc> *expression_list;
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
tree t;
bool expr_non_constant_p;
cp_lexer_set_source_position (parser->lexer);
maybe_warn_cpp0x (CPP0X_INITIALIZER_LISTS);
t = cp_parser_braced_list (parser, &expr_non_constant_p);
CONSTRUCTOR_IS_DIRECT_INIT (t) = 1;
expression_list = make_tree_vector_single (t);
}
else
expression_list = (cp_parser_parenthesized_expression_list
(parser, non_attr, false,
true,
NULL));
return expression_list;
}
static tree
cp_parser_delete_expression (cp_parser* parser)
{
bool global_scope_p;
bool array_p;
tree expression;
global_scope_p
= (cp_parser_global_scope_opt (parser,
false)
!= NULL_TREE);
cp_parser_require_keyword (parser, RID_DELETE, RT_DELETE);
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_SQUARE))
{
cp_lexer_consume_token (parser->lexer);
cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE);
array_p = true;
}
else
array_p = false;
expression = cp_parser_simple_cast_expression (parser);
if (cp_parser_non_integral_constant_expression (parser, NIC_DEL))
return error_mark_node;
return delete_sanity (expression, NULL_TREE, array_p, global_scope_p,
tf_warning_or_error);
}
static int
cp_parser_tokens_start_cast_expression (cp_parser *parser)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
switch (token->type)
{
case CPP_COMMA:
case CPP_SEMICOLON:
case CPP_QUERY:
case CPP_COLON:
case CPP_CLOSE_SQUARE:
case CPP_CLOSE_PAREN:
case CPP_CLOSE_BRACE:
case CPP_OPEN_BRACE:
case CPP_DOT:
case CPP_DOT_STAR:
case CPP_DEREF:
case CPP_DEREF_STAR:
case CPP_DIV:
case CPP_MOD:
case CPP_LSHIFT:
case CPP_RSHIFT:
case CPP_LESS:
case CPP_GREATER:
case CPP_LESS_EQ:
case CPP_GREATER_EQ:
case CPP_EQ_EQ:
case CPP_NOT_EQ:
case CPP_EQ:
case CPP_MULT_EQ:
case CPP_DIV_EQ:
case CPP_MOD_EQ:
case CPP_PLUS_EQ:
case CPP_MINUS_EQ:
case CPP_RSHIFT_EQ:
case CPP_LSHIFT_EQ:
case CPP_AND_EQ:
case CPP_XOR_EQ:
case CPP_OR_EQ:
case CPP_XOR:
case CPP_OR:
case CPP_OR_OR:
case CPP_EOF:
case CPP_ELLIPSIS:
return 0;
case CPP_OPEN_PAREN:
return cp_lexer_peek_nth_token (parser->lexer, 2)->type
!= CPP_CLOSE_PAREN;
case CPP_OPEN_SQUARE:
if (cxx_dialect >= cxx11)
return -1;
return c_dialect_objc ();
case CPP_PLUS_PLUS:
case CPP_MINUS_MINUS:
return -1;
default:
return 1;
}
}
static const char *
get_cast_suggestion (tree dst_type, tree orig_expr)
{
tree trial;
if (processing_template_decl)
return NULL;
trial = build_const_cast (dst_type, orig_expr, tf_none);
if (trial != error_mark_node)
return "const_cast";
trial = build_static_cast (dst_type, orig_expr, tf_none);
if (trial != error_mark_node)
return "static_cast";
trial = build_reinterpret_cast (dst_type, orig_expr, tf_none);
if (trial != error_mark_node)
return "reinterpret_cast";
return NULL;
}
static void
maybe_add_cast_fixit (rich_location *rich_loc, location_t open_paren_loc,
location_t close_paren_loc, tree orig_expr,
tree dst_type)
{
if (!warn_old_style_cast)
return;
const char *cast_suggestion = get_cast_suggestion (dst_type, orig_expr);
if (!cast_suggestion)
return;
pretty_printer pp;
pp_printf (&pp, "%s<", cast_suggestion);
rich_loc->add_fixit_replace (open_paren_loc, pp_formatted_text (&pp));
rich_loc->add_fixit_replace (close_paren_loc, "> (");
rich_loc->add_fixit_insert_after (")");
}
static cp_expr
cp_parser_cast_expression (cp_parser *parser, bool address_p, bool cast_p,
bool decltype_p, cp_id_kind * pidk)
{
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
tree type = NULL_TREE;
cp_expr expr (NULL_TREE);
int cast_expression = 0;
const char *saved_message;
cp_parser_parse_tentatively (parser);
saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in casts");
matching_parens parens;
cp_token *open_paren = parens.consume_open (parser);
location_t open_paren_loc = open_paren->location;
location_t close_paren_loc = UNKNOWN_LOCATION;
cp_lexer_save_tokens (parser->lexer);
if (cp_parser_skip_to_closing_parenthesis (parser, false, false,
true))
cast_expression
= cp_parser_tokens_start_cast_expression (parser);
cp_lexer_rollback_tokens (parser->lexer);
if (!cast_expression)
cp_parser_simulate_error (parser);
else
{
bool saved_in_type_id_in_expr_p = parser->in_type_id_in_expr_p;
parser->in_type_id_in_expr_p = true;
type = cp_parser_type_id (parser);
cp_token *close_paren = parens.require_close (parser);
if (close_paren)
close_paren_loc = close_paren->location;
parser->in_type_id_in_expr_p = saved_in_type_id_in_expr_p;
}
parser->type_definition_forbidden_message = saved_message;
if (!cp_parser_error_occurred (parser))
{
if (cast_expression > 0)
cp_parser_commit_to_topmost_tentative_parse (parser);
expr = cp_parser_cast_expression (parser,
false,
true,
false,
pidk);
if (cp_parser_parse_definitely (parser))
{
if (warn_old_style_cast
&& !in_system_header_at (input_location)
&& !VOID_TYPE_P (type)
&& current_lang_name != lang_name_c)
{
gcc_rich_location rich_loc (input_location);
maybe_add_cast_fixit (&rich_loc, open_paren_loc, close_paren_loc,
expr, type);
warning_at (&rich_loc, OPT_Wold_style_cast,
"use of old-style cast to %q#T", type);
}
if (!cast_valid_in_integral_constant_expression_p (type)
&& cp_parser_non_integral_constant_expression (parser,
NIC_CAST))
return error_mark_node;
location_t cast_loc = make_location (open_paren_loc,
open_paren_loc,
expr.get_finish ());
expr = build_c_cast (cast_loc, type, expr);
return expr;
}
}
else 
cp_parser_abort_tentative_parse (parser);
}
return cp_parser_unary_expression (parser, pidk, address_p,
cast_p, decltype_p);
}
#define TOKEN_PRECEDENCE(token)				     \
(((token->type == CPP_GREATER				     \
|| ((cxx_dialect != cxx98) && token->type == CPP_RSHIFT)) \
&& !parser->greater_than_is_operator_p)		     \
? PREC_NOT_OPERATOR					     \
: binops_by_token[token->type].prec)
static cp_expr
cp_parser_binary_expression (cp_parser* parser, bool cast_p,
bool no_toplevel_fold_p,
bool decltype_p,
enum cp_parser_prec prec,
cp_id_kind * pidk)
{
cp_parser_expression_stack stack;
cp_parser_expression_stack_entry *sp = &stack[0];
cp_parser_expression_stack_entry current;
cp_expr rhs;
cp_token *token;
enum tree_code rhs_type;
enum cp_parser_prec new_prec, lookahead_prec;
tree overload;
current.lhs_type = (cp_lexer_next_token_is (parser->lexer, CPP_NOT)
? TRUTH_NOT_EXPR : ERROR_MARK);
current.lhs = cp_parser_cast_expression (parser, false,
cast_p, decltype_p, pidk);
current.prec = prec;
if (cp_parser_error_occurred (parser))
return error_mark_node;
for (;;)
{
token = cp_lexer_peek_token (parser->lexer);
if (warn_cxx11_compat
&& token->type == CPP_RSHIFT
&& !parser->greater_than_is_operator_p)
{
if (warning_at (token->location, OPT_Wc__11_compat,
"%<>>%> operator is treated"
" as two right angle brackets in C++11"))
inform (token->location,
"suggest parentheses around %<>>%> expression");
}
new_prec = TOKEN_PRECEDENCE (token);
if (new_prec != PREC_NOT_OPERATOR
&& cp_lexer_nth_token_is (parser->lexer, 2, CPP_ELLIPSIS))
new_prec = PREC_NOT_OPERATOR;
if (new_prec <= current.prec)
{
if (sp == stack)
break;
else
goto pop;
}
get_rhs:
current.tree_type = binops_by_token[token->type].tree_type;
current.loc = token->location;
cp_lexer_consume_token (parser->lexer);
if (current.tree_type == TRUTH_ANDIF_EXPR)
c_inhibit_evaluation_warnings +=
cp_fully_fold (current.lhs) == truthvalue_false_node;
else if (current.tree_type == TRUTH_ORIF_EXPR)
c_inhibit_evaluation_warnings +=
cp_fully_fold (current.lhs) == truthvalue_true_node;
rhs_type = (cp_lexer_next_token_is (parser->lexer, CPP_NOT)
? TRUTH_NOT_EXPR : ERROR_MARK);
rhs = cp_parser_simple_cast_expression (parser);
token = cp_lexer_peek_token (parser->lexer);
lookahead_prec = TOKEN_PRECEDENCE (token);
if (lookahead_prec != PREC_NOT_OPERATOR
&& cp_lexer_nth_token_is (parser->lexer, 2, CPP_ELLIPSIS))
lookahead_prec = PREC_NOT_OPERATOR;
if (lookahead_prec > new_prec)
{
*sp = current;
++sp;
current.lhs = rhs;
current.lhs_type = rhs_type;
current.prec = new_prec;
new_prec = lookahead_prec;
goto get_rhs;
pop:
lookahead_prec = new_prec;
rhs = current.lhs;
rhs_type = current.lhs_type;
--sp;
current = *sp;
}
if (current.tree_type == TRUTH_ANDIF_EXPR)
c_inhibit_evaluation_warnings -=
cp_fully_fold (current.lhs) == truthvalue_false_node;
else if (current.tree_type == TRUTH_ORIF_EXPR)
c_inhibit_evaluation_warnings -=
cp_fully_fold (current.lhs) == truthvalue_true_node;
if (warn_logical_not_paren
&& TREE_CODE_CLASS (current.tree_type) == tcc_comparison
&& current.lhs_type == TRUTH_NOT_EXPR
&& (TREE_CODE (current.lhs) != NE_EXPR
|| !integer_zerop (TREE_OPERAND (current.lhs, 1)))
&& (TREE_CODE (current.lhs) != TRUTH_NOT_EXPR
|| (TREE_CODE (TREE_OPERAND (current.lhs, 0)) != TRUTH_NOT_EXPR
&& (TREE_TYPE (TREE_OPERAND (current.lhs, 0)) == NULL_TREE
|| (TREE_CODE (TREE_TYPE (TREE_OPERAND (current.lhs, 0)))
!= BOOLEAN_TYPE))))
&& (!DECL_P (current.lhs)
|| TREE_TYPE (current.lhs) == NULL_TREE
|| TREE_CODE (TREE_TYPE (current.lhs)) != BOOLEAN_TYPE))
warn_logical_not_parentheses (current.loc, current.tree_type,
current.lhs, maybe_constant_value (rhs));
overload = NULL;
location_t combined_loc = make_location (current.loc,
current.lhs.get_start (),
rhs.get_finish ());
if (no_toplevel_fold_p
&& lookahead_prec <= current.prec
&& sp == stack)
{
if (current.lhs == error_mark_node || rhs == error_mark_node)
current.lhs = error_mark_node;
else
{
current.lhs
= build_min (current.tree_type,
TREE_CODE_CLASS (current.tree_type)
== tcc_comparison
? boolean_type_node : TREE_TYPE (current.lhs),
current.lhs.get_value (), rhs.get_value ());
SET_EXPR_LOCATION (current.lhs, combined_loc);
}
}
else
{
current.lhs = build_x_binary_op (combined_loc, current.tree_type,
current.lhs, current.lhs_type,
rhs, rhs_type, &overload,
complain_flags (decltype_p));
current.lhs.set_location (combined_loc);
}
current.lhs_type = current.tree_type;
if (overload
&& cp_parser_non_integral_constant_expression (parser,
NIC_OVERLOADED))
return error_mark_node;
}
return current.lhs;
}
static cp_expr
cp_parser_binary_expression (cp_parser* parser, bool cast_p,
bool no_toplevel_fold_p,
enum cp_parser_prec prec,
cp_id_kind * pidk)
{
return cp_parser_binary_expression (parser, cast_p, no_toplevel_fold_p,
false, prec, pidk);
}
static tree
cp_parser_question_colon_clause (cp_parser* parser, cp_expr logical_or_expr)
{
tree expr, folded_logical_or_expr = cp_fully_fold (logical_or_expr);
cp_expr assignment_expr;
struct cp_token *token;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);
token = cp_lexer_peek_token (parser->lexer);
if (cp_parser_allow_gnu_extensions_p (parser)
&& token->type == CPP_COLON)
{
pedwarn (token->location, OPT_Wpedantic, 
"ISO C++ does not allow ?: with omitted middle operand");
expr = NULL_TREE;
c_inhibit_evaluation_warnings +=
folded_logical_or_expr == truthvalue_true_node;
warn_for_omitted_condop (token->location, logical_or_expr);
}
else
{
bool saved_colon_corrects_to_scope_p = parser->colon_corrects_to_scope_p;
parser->colon_corrects_to_scope_p = false;
c_inhibit_evaluation_warnings +=
folded_logical_or_expr == truthvalue_false_node;
expr = cp_parser_expression (parser);
c_inhibit_evaluation_warnings +=
((folded_logical_or_expr == truthvalue_true_node)
- (folded_logical_or_expr == truthvalue_false_node));
parser->colon_corrects_to_scope_p = saved_colon_corrects_to_scope_p;
}
cp_parser_require (parser, CPP_COLON, RT_COLON);
assignment_expr = cp_parser_assignment_expression (parser);
c_inhibit_evaluation_warnings -=
folded_logical_or_expr == truthvalue_true_node;
loc = make_location (loc,
logical_or_expr.get_start (),
assignment_expr.get_finish ());
return build_x_conditional_expr (loc, logical_or_expr,
expr,
assignment_expr,
tf_warning_or_error);
}
static cp_expr
cp_parser_assignment_expression (cp_parser* parser, cp_id_kind * pidk,
bool cast_p, bool decltype_p)
{
cp_expr expr;
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_THROW))
expr = cp_parser_throw_expression (parser);
else
{
expr = cp_parser_binary_expression (parser, cast_p, false,
decltype_p,
PREC_NOT_OPERATOR, pidk);
if (cp_lexer_next_token_is (parser->lexer, CPP_QUERY))
return cp_parser_question_colon_clause (parser, expr);
else
{
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
enum tree_code assignment_operator
= cp_parser_assignment_operator_opt (parser);
if (assignment_operator != ERROR_MARK)
{
bool non_constant_p;
cp_expr rhs = cp_parser_initializer_clause (parser,
&non_constant_p);
if (BRACE_ENCLOSED_INITIALIZER_P (rhs))
maybe_warn_cpp0x (CPP0X_INITIALIZER_LISTS);
if (cp_parser_non_integral_constant_expression (parser,
NIC_ASSIGNMENT))
return error_mark_node;
loc = make_location (loc,
expr.get_start (),
rhs.get_finish ());
expr = build_x_modify_expr (loc, expr,
assignment_operator,
rhs,
complain_flags (decltype_p));
expr.set_location (loc);
}
}
}
return expr;
}
static enum tree_code
cp_parser_assignment_operator_opt (cp_parser* parser)
{
enum tree_code op;
cp_token *token;
token = cp_lexer_peek_token (parser->lexer);
switch (token->type)
{
case CPP_EQ:
op = NOP_EXPR;
break;
case CPP_MULT_EQ:
op = MULT_EXPR;
break;
case CPP_DIV_EQ:
op = TRUNC_DIV_EXPR;
break;
case CPP_MOD_EQ:
op = TRUNC_MOD_EXPR;
break;
case CPP_PLUS_EQ:
op = PLUS_EXPR;
break;
case CPP_MINUS_EQ:
op = MINUS_EXPR;
break;
case CPP_RSHIFT_EQ:
op = RSHIFT_EXPR;
break;
case CPP_LSHIFT_EQ:
op = LSHIFT_EXPR;
break;
case CPP_AND_EQ:
op = BIT_AND_EXPR;
break;
case CPP_XOR_EQ:
op = BIT_XOR_EXPR;
break;
case CPP_OR_EQ:
op = BIT_IOR_EXPR;
break;
default:
op = ERROR_MARK;
}
if (op != ERROR_MARK
&& cp_lexer_nth_token_is (parser->lexer, 2, CPP_ELLIPSIS))
op = ERROR_MARK;
if (op != ERROR_MARK)
cp_lexer_consume_token (parser->lexer);
return op;
}
static cp_expr
cp_parser_expression (cp_parser* parser, cp_id_kind * pidk,
bool cast_p, bool decltype_p)
{
cp_expr expression = NULL_TREE;
location_t loc = UNKNOWN_LOCATION;
while (true)
{
cp_expr assignment_expression;
assignment_expression
= cp_parser_assignment_expression (parser, pidk, cast_p, decltype_p);
if (decltype_p && !processing_template_decl
&& TREE_CODE (assignment_expression) == CALL_EXPR
&& CLASS_TYPE_P (TREE_TYPE (assignment_expression))
&& cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
assignment_expression
= build_cplus_new (TREE_TYPE (assignment_expression),
assignment_expression, tf_warning_or_error);
if (!expression)
expression = assignment_expression;
else
{
loc = make_location (loc,
expression.get_start (),
assignment_expression.get_finish ());
expression = build_x_compound_expr (loc, expression,
assignment_expression,
complain_flags (decltype_p));
expression.set_location (loc);
}
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA)
|| cp_lexer_nth_token_is (parser->lexer, 2, CPP_ELLIPSIS))
break;
loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);
if (cp_parser_non_integral_constant_expression (parser, NIC_COMMA))
expression = error_mark_node;
}
return expression;
}
static cp_expr
cp_parser_constant_expression (cp_parser* parser,
bool allow_non_constant_p,
bool *non_constant_p,
bool strict_p)
{
bool saved_integral_constant_expression_p;
bool saved_allow_non_integral_constant_expression_p;
bool saved_non_integral_constant_expression_p;
cp_expr expression;
saved_integral_constant_expression_p = parser->integral_constant_expression_p;
saved_allow_non_integral_constant_expression_p
= parser->allow_non_integral_constant_expression_p;
saved_non_integral_constant_expression_p = parser->non_integral_constant_expression_p;
parser->integral_constant_expression_p = true;
parser->allow_non_integral_constant_expression_p
= (allow_non_constant_p || cxx_dialect >= cxx11);
parser->non_integral_constant_expression_p = false;
if (strict_p)
{
expression = cp_parser_binary_expression (parser, false, false, false,
PREC_NOT_OPERATOR, NULL);
if (cp_lexer_next_token_is (parser->lexer, CPP_QUERY))
expression = cp_parser_question_colon_clause (parser, expression);
}
else
expression = cp_parser_assignment_expression (parser);
parser->integral_constant_expression_p
= saved_integral_constant_expression_p;
parser->allow_non_integral_constant_expression_p
= saved_allow_non_integral_constant_expression_p;
if (cxx_dialect >= cxx11)
{
tree decay = expression;
if (TREE_TYPE (expression)
&& TREE_CODE (TREE_TYPE (expression)) == ARRAY_TYPE)
decay = build_address (expression);
bool is_const = potential_rvalue_constant_expression (decay);
parser->non_integral_constant_expression_p = !is_const;
if (!is_const && !allow_non_constant_p)
require_potential_rvalue_constant_expression (decay);
}
if (allow_non_constant_p)
*non_constant_p = parser->non_integral_constant_expression_p;
parser->non_integral_constant_expression_p
= saved_non_integral_constant_expression_p;
return expression;
}
static cp_expr
cp_parser_builtin_offsetof (cp_parser *parser)
{
int save_ice_p, save_non_ice_p;
tree type;
cp_expr expr;
cp_id_kind dummy;
cp_token *token;
location_t finish_loc;
save_ice_p = parser->integral_constant_expression_p;
save_non_ice_p = parser->non_integral_constant_expression_p;
location_t start_loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);
matching_parens parens;
parens.require_open (parser);
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
{
const char *saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined within __builtin_offsetof");
type = cp_parser_type_id (parser);
parser->type_definition_forbidden_message = saved_message;
}
cp_parser_require (parser, CPP_COMMA, RT_COMMA);
token = cp_lexer_peek_token (parser->lexer);
tree object_ptr
= build_static_cast (build_pointer_type (type), null_pointer_node,
tf_warning_or_error);
expr = cp_parser_postfix_dot_deref_expression (parser, CPP_DEREF, object_ptr,
true, &dummy, token->location);
while (true)
{
token = cp_lexer_peek_token (parser->lexer);
switch (token->type)
{
case CPP_OPEN_SQUARE:
expr = cp_parser_postfix_open_square_expression (parser, expr,
true, false);
break;
case CPP_DEREF:
expr = grok_array_decl (token->location, expr,
integer_zero_node, false);
case CPP_DOT:
cp_lexer_consume_token (parser->lexer);
expr = cp_parser_postfix_dot_deref_expression (parser, CPP_DOT,
expr, true, &dummy,
token->location);
break;
case CPP_CLOSE_PAREN:
finish_loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);
goto success;
default:
parens.require_close (parser);
cp_parser_skip_to_closing_parenthesis (parser, true, false, true);
expr = error_mark_node;
goto failure;
}
}
success:
loc = make_location (loc, start_loc, finish_loc);
expr = cp_expr (finish_offsetof (object_ptr, expr, loc), loc);
failure:
parser->integral_constant_expression_p = save_ice_p;
parser->non_integral_constant_expression_p = save_non_ice_p;
expr = expr.maybe_add_location_wrapper ();
return expr;
}
static cp_expr
cp_parser_trait_expr (cp_parser* parser, enum rid keyword)
{
cp_trait_kind kind;
tree type1, type2 = NULL_TREE;
bool binary = false;
bool variadic = false;
switch (keyword)
{
case RID_HAS_NOTHROW_ASSIGN:
kind = CPTK_HAS_NOTHROW_ASSIGN;
break;
case RID_HAS_NOTHROW_CONSTRUCTOR:
kind = CPTK_HAS_NOTHROW_CONSTRUCTOR;
break;
case RID_HAS_NOTHROW_COPY:
kind = CPTK_HAS_NOTHROW_COPY;
break;
case RID_HAS_TRIVIAL_ASSIGN:
kind = CPTK_HAS_TRIVIAL_ASSIGN;
break;
case RID_HAS_TRIVIAL_CONSTRUCTOR:
kind = CPTK_HAS_TRIVIAL_CONSTRUCTOR;
break;
case RID_HAS_TRIVIAL_COPY:
kind = CPTK_HAS_TRIVIAL_COPY;
break;
case RID_HAS_TRIVIAL_DESTRUCTOR:
kind = CPTK_HAS_TRIVIAL_DESTRUCTOR;
break;
case RID_HAS_UNIQUE_OBJ_REPRESENTATIONS:
kind = CPTK_HAS_UNIQUE_OBJ_REPRESENTATIONS;
break;
case RID_HAS_VIRTUAL_DESTRUCTOR:
kind = CPTK_HAS_VIRTUAL_DESTRUCTOR;
break;
case RID_IS_ABSTRACT:
kind = CPTK_IS_ABSTRACT;
break;
case RID_IS_AGGREGATE:
kind = CPTK_IS_AGGREGATE;
break;
case RID_IS_BASE_OF:
kind = CPTK_IS_BASE_OF;
binary = true;
break;
case RID_IS_CLASS:
kind = CPTK_IS_CLASS;
break;
case RID_IS_EMPTY:
kind = CPTK_IS_EMPTY;
break;
case RID_IS_ENUM:
kind = CPTK_IS_ENUM;
break;
case RID_IS_FINAL:
kind = CPTK_IS_FINAL;
break;
case RID_IS_LITERAL_TYPE:
kind = CPTK_IS_LITERAL_TYPE;
break;
case RID_IS_POD:
kind = CPTK_IS_POD;
break;
case RID_IS_POLYMORPHIC:
kind = CPTK_IS_POLYMORPHIC;
break;
case RID_IS_SAME_AS:
kind = CPTK_IS_SAME_AS;
binary = true;
break;
case RID_IS_STD_LAYOUT:
kind = CPTK_IS_STD_LAYOUT;
break;
case RID_IS_TRIVIAL:
kind = CPTK_IS_TRIVIAL;
break;
case RID_IS_TRIVIALLY_ASSIGNABLE:
kind = CPTK_IS_TRIVIALLY_ASSIGNABLE;
binary = true;
break;
case RID_IS_TRIVIALLY_CONSTRUCTIBLE:
kind = CPTK_IS_TRIVIALLY_CONSTRUCTIBLE;
variadic = true;
break;
case RID_IS_TRIVIALLY_COPYABLE:
kind = CPTK_IS_TRIVIALLY_COPYABLE;
break;
case RID_IS_UNION:
kind = CPTK_IS_UNION;
break;
case RID_UNDERLYING_TYPE:
kind = CPTK_UNDERLYING_TYPE;
break;
case RID_BASES:
kind = CPTK_BASES;
break;
case RID_DIRECT_BASES:
kind = CPTK_DIRECT_BASES;
break;
case RID_IS_ASSIGNABLE:
kind = CPTK_IS_ASSIGNABLE;
binary = true;
break;
case RID_IS_CONSTRUCTIBLE:
kind = CPTK_IS_CONSTRUCTIBLE;
variadic = true;
break;
default:
gcc_unreachable ();
}
location_t start_loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);
matching_parens parens;
parens.require_open (parser);
{
type_id_in_expr_sentinel s (parser);
type1 = cp_parser_type_id (parser);
}
if (type1 == error_mark_node)
return error_mark_node;
if (binary)
{
cp_parser_require (parser, CPP_COMMA, RT_COMMA);
{
type_id_in_expr_sentinel s (parser);
type2 = cp_parser_type_id (parser);
}
if (type2 == error_mark_node)
return error_mark_node;
}
else if (variadic)
{
while (cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
{
cp_lexer_consume_token (parser->lexer);
tree elt = cp_parser_type_id (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
cp_lexer_consume_token (parser->lexer);
elt = make_pack_expansion (elt);
}
if (elt == error_mark_node)
return error_mark_node;
type2 = tree_cons (NULL_TREE, elt, type2);
}
}
location_t finish_loc = cp_lexer_peek_token (parser->lexer)->location;
parens.require_close (parser);
location_t trait_loc = make_location (start_loc, start_loc, finish_loc);
switch (kind)
{
case CPTK_UNDERLYING_TYPE:
return cp_expr (finish_underlying_type (type1), trait_loc);
case CPTK_BASES:
return cp_expr (finish_bases (type1, false), trait_loc);
case CPTK_DIRECT_BASES:
return cp_expr (finish_bases (type1, true), trait_loc);
default:
return cp_expr (finish_trait_expr (kind, type1, type2), trait_loc);
}
}
static cp_expr
cp_parser_lambda_expression (cp_parser* parser)
{
tree lambda_expr = build_lambda_expr ();
tree type;
bool ok = true;
cp_token *token = cp_lexer_peek_token (parser->lexer);
cp_token_position start = 0;
LAMBDA_EXPR_LOCATION (lambda_expr) = token->location;
if (cp_unevaluated_operand)
{
if (!token->error_reported)
{
error_at (LAMBDA_EXPR_LOCATION (lambda_expr),
"lambda-expression in unevaluated context");
token->error_reported = true;
}
ok = false;
}
else if (parser->in_template_argument_list_p)
{
if (!token->error_reported)
{
error_at (token->location, "lambda-expression in template-argument");
token->error_reported = true;
}
ok = false;
}
push_deferring_access_checks (dk_no_deferred);
cp_parser_lambda_introducer (parser, lambda_expr);
type = begin_lambda_type (lambda_expr);
if (type == error_mark_node)
return error_mark_node;
record_lambda_scope (lambda_expr);
determine_visibility (TYPE_NAME (type));
register_capture_members (LAMBDA_EXPR_CAPTURE_LIST (lambda_expr));
{
unsigned int saved_num_template_parameter_lists
= parser->num_template_parameter_lists;
unsigned char in_statement = parser->in_statement;
bool in_switch_statement_p = parser->in_switch_statement_p;
bool fully_implicit_function_template_p
= parser->fully_implicit_function_template_p;
tree implicit_template_parms = parser->implicit_template_parms;
cp_binding_level* implicit_template_scope = parser->implicit_template_scope;
bool auto_is_implicit_function_template_parm_p
= parser->auto_is_implicit_function_template_parm_p;
parser->num_template_parameter_lists = 0;
parser->in_statement = 0;
parser->in_switch_statement_p = false;
parser->fully_implicit_function_template_p = false;
parser->implicit_template_parms = 0;
parser->implicit_template_scope = 0;
parser->auto_is_implicit_function_template_parm_p = false;
ok &= cp_parser_lambda_declarator_opt (parser, lambda_expr);
if (ok && cp_parser_error_occurred (parser))
ok = false;
if (ok)
{
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE)
&& cp_parser_start_tentative_firewall (parser))
start = token;
cp_parser_lambda_body (parser, lambda_expr);
}
else if (cp_parser_require (parser, CPP_OPEN_BRACE, RT_OPEN_BRACE))
{
if (cp_parser_skip_to_closing_brace (parser))
cp_lexer_consume_token (parser->lexer);
}
LAMBDA_EXPR_CAPTURE_LIST (lambda_expr)
= nreverse (LAMBDA_EXPR_CAPTURE_LIST (lambda_expr));
if (ok)
maybe_add_lambda_conv_op (type);
type = finish_struct (type, NULL_TREE);
parser->num_template_parameter_lists = saved_num_template_parameter_lists;
parser->in_statement = in_statement;
parser->in_switch_statement_p = in_switch_statement_p;
parser->fully_implicit_function_template_p
= fully_implicit_function_template_p;
parser->implicit_template_parms = implicit_template_parms;
parser->implicit_template_scope = implicit_template_scope;
parser->auto_is_implicit_function_template_parm_p
= auto_is_implicit_function_template_parm_p;
}
LAMBDA_EXPR_THIS_CAPTURE (lambda_expr) = NULL_TREE;
gcc_assert (LAMBDA_EXPR_PENDING_PROXIES (lambda_expr) == NULL);
insert_pending_capture_proxies ();
if (ok)
lambda_expr = build_lambda_object (lambda_expr);
else
lambda_expr = error_mark_node;
cp_parser_end_tentative_firewall (parser, start, lambda_expr);
pop_deferring_access_checks ();
return lambda_expr;
}
static void
cp_parser_lambda_introducer (cp_parser* parser, tree lambda_expr)
{
bool first = true;
cp_parser_require (parser, CPP_OPEN_SQUARE, RT_OPEN_SQUARE);
if (cp_lexer_next_token_is (parser->lexer, CPP_AND)
&& cp_lexer_peek_nth_token (parser->lexer, 2)->type != CPP_NAME)
LAMBDA_EXPR_DEFAULT_CAPTURE_MODE (lambda_expr) = CPLD_REFERENCE;
else if (cp_lexer_next_token_is (parser->lexer, CPP_EQ))
LAMBDA_EXPR_DEFAULT_CAPTURE_MODE (lambda_expr) = CPLD_COPY;
if (LAMBDA_EXPR_DEFAULT_CAPTURE_MODE (lambda_expr) != CPLD_NONE)
{
cp_lexer_consume_token (parser->lexer);
first = false;
}
while (cp_lexer_next_token_is_not (parser->lexer, CPP_CLOSE_SQUARE))
{
cp_token* capture_token;
tree capture_id;
tree capture_init_expr;
cp_id_kind idk = CP_ID_KIND_NONE;
bool explicit_init_p = false;
enum capture_kind_type
{
BY_COPY,
BY_REFERENCE
};
enum capture_kind_type capture_kind = BY_COPY;
if (cp_lexer_next_token_is (parser->lexer, CPP_EOF))
{
error ("expected end of capture-list");
return;
}
if (first)
first = false;
else
cp_parser_require (parser, CPP_COMMA, RT_COMMA);
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_THIS))
{
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
if (cxx_dialect < cxx2a
&& LAMBDA_EXPR_DEFAULT_CAPTURE_MODE (lambda_expr) == CPLD_COPY)
pedwarn (loc, 0, "explicit by-copy capture of %<this%> redundant "
"with by-copy capture default");
cp_lexer_consume_token (parser->lexer);
add_capture (lambda_expr,
this_identifier,
finish_this_expr (),
true,
explicit_init_p);
continue;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_MULT)
&& cp_lexer_nth_token_is_keyword (parser->lexer, 2, RID_THIS))
{
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
if (cxx_dialect < cxx17)
pedwarn (loc, 0, "%<*this%> capture only available with "
"-std=c++17 or -std=gnu++17");
cp_lexer_consume_token (parser->lexer);
cp_lexer_consume_token (parser->lexer);
add_capture (lambda_expr,
this_identifier,
finish_this_expr (),
false,
explicit_init_p);
continue;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_AND))
{
capture_kind = BY_REFERENCE;
cp_lexer_consume_token (parser->lexer);
}
capture_token = cp_lexer_peek_token (parser->lexer);
capture_id = cp_parser_identifier (parser);
if (capture_id == error_mark_node)
{
cp_parser_skip_to_closing_parenthesis (parser,
true,
true,
true);
break;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_EQ)
|| cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN)
|| cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
bool direct, non_constant;
if (cxx_dialect < cxx14)
pedwarn (input_location, 0,
"lambda capture initializers "
"only available with -std=c++14 or -std=gnu++14");
capture_init_expr = cp_parser_initializer (parser, &direct,
&non_constant, true);
explicit_init_p = true;
if (capture_init_expr == NULL_TREE)
{
error ("empty initializer for lambda init-capture");
capture_init_expr = error_mark_node;
}
}
else
{
const char* error_msg;
capture_init_expr
= cp_parser_lookup_name_simple (parser, capture_id,
capture_token->location);
if (capture_init_expr == error_mark_node)
{
unqualified_name_lookup_error (capture_id);
continue;
}
else if (!VAR_P (capture_init_expr)
&& TREE_CODE (capture_init_expr) != PARM_DECL)
{
error_at (capture_token->location,
"capture of non-variable %qE",
capture_init_expr);
if (DECL_P (capture_init_expr))
inform (DECL_SOURCE_LOCATION (capture_init_expr),
"%q#D declared here", capture_init_expr);
continue;
}
if (VAR_P (capture_init_expr)
&& decl_storage_duration (capture_init_expr) != dk_auto)
{
if (pedwarn (capture_token->location, 0, "capture of variable "
"%qD with non-automatic storage duration",
capture_init_expr))
inform (DECL_SOURCE_LOCATION (capture_init_expr),
"%q#D declared here", capture_init_expr);
continue;
}
capture_init_expr
= finish_id_expression
(capture_id,
capture_init_expr,
parser->scope,
&idk,
false,
false,
NULL,
false,
true,
false,
false,
&error_msg,
capture_token->location);
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
cp_lexer_consume_token (parser->lexer);
capture_init_expr = make_pack_expansion (capture_init_expr);
}
}
if (LAMBDA_EXPR_DEFAULT_CAPTURE_MODE (lambda_expr) != CPLD_NONE
&& !explicit_init_p)
{
if (LAMBDA_EXPR_DEFAULT_CAPTURE_MODE (lambda_expr) == CPLD_COPY
&& capture_kind == BY_COPY)
pedwarn (capture_token->location, 0, "explicit by-copy capture "
"of %qD redundant with by-copy capture default",
capture_id);
if (LAMBDA_EXPR_DEFAULT_CAPTURE_MODE (lambda_expr) == CPLD_REFERENCE
&& capture_kind == BY_REFERENCE)
pedwarn (capture_token->location, 0, "explicit by-reference "
"capture of %qD redundant with by-reference capture "
"default", capture_id);
}
add_capture (lambda_expr,
capture_id,
capture_init_expr,
capture_kind == BY_REFERENCE,
explicit_init_p);
parser->scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
parser->object_scope = NULL_TREE;
}
cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE);
}
static bool
cp_parser_lambda_declarator_opt (cp_parser* parser, tree lambda_expr)
{
tree param_list = void_list_node;
tree attributes = NULL_TREE;
tree exception_spec = NULL_TREE;
tree template_param_list = NULL_TREE;
tree tx_qual = NULL_TREE;
tree return_type = NULL_TREE;
cp_decl_specifier_seq lambda_specs;
clear_decl_specs (&lambda_specs);
if (cp_lexer_next_token_is (parser->lexer, CPP_LESS))
{
if (cxx_dialect < cxx14)
pedwarn (parser->lexer->next_token->location, 0,
"lambda templates are only available with "
"-std=c++14 or -std=gnu++14");
else if (cxx_dialect < cxx2a)
pedwarn (parser->lexer->next_token->location, OPT_Wpedantic,
"lambda templates are only available with "
"-std=c++2a or -std=gnu++2a");
cp_lexer_consume_token (parser->lexer);
template_param_list = cp_parser_template_parameter_list (parser);
cp_parser_skip_to_end_of_template_parameter_list (parser);
++parser->num_template_parameter_lists;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
matching_parens parens;
parens.consume_open (parser);
begin_scope (sk_function_parms, NULL_TREE);
param_list = cp_parser_parameter_declaration_clause (parser);
if (cxx_dialect < cxx14)
for (tree t = param_list; t; t = TREE_CHAIN (t))
if (TREE_PURPOSE (t) && DECL_P (TREE_VALUE (t)))
pedwarn (DECL_SOURCE_LOCATION (TREE_VALUE (t)), OPT_Wpedantic,
"default argument specified for lambda parameter");
parens.require_close (parser);
attributes = cp_parser_attributes_opt (parser);
int declares_class_or_enum;
if (cp_lexer_next_token_is_decl_specifier_keyword (parser->lexer))
cp_parser_decl_specifier_seq (parser,
CP_PARSER_FLAGS_ONLY_MUTABLE_OR_CONSTEXPR,
&lambda_specs, &declares_class_or_enum);
if (lambda_specs.storage_class == sc_mutable)
{
LAMBDA_EXPR_MUTABLE_P (lambda_expr) = 1;
if (lambda_specs.conflicting_specifiers_p)
error_at (lambda_specs.locations[ds_storage_class],
"duplicate %<mutable%>");
}
tx_qual = cp_parser_tx_qualifier_opt (parser);
exception_spec = cp_parser_exception_specification_opt (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_DEREF))
{
cp_lexer_consume_token (parser->lexer);
return_type = cp_parser_trailing_type_id (parser);
}
pop_bindings_and_leave_scope ();
}
else if (template_param_list != NULL_TREE) 
cp_parser_require (parser, CPP_OPEN_PAREN, RT_OPEN_PAREN);
{
cp_decl_specifier_seq return_type_specs;
cp_declarator* declarator;
tree fco;
int quals;
void *p;
clear_decl_specs (&return_type_specs);
if (return_type)
return_type_specs.type = return_type;
else
return_type_specs.type = make_auto ();
if (lambda_specs.locations[ds_constexpr])
{
if (cxx_dialect >= cxx17)
return_type_specs.locations[ds_constexpr]
= lambda_specs.locations[ds_constexpr];
else
error_at (lambda_specs.locations[ds_constexpr], "%<constexpr%> "
"lambda only available with -std=c++17 or -std=gnu++17");
}
p = obstack_alloc (&declarator_obstack, 0);
declarator = make_id_declarator (NULL_TREE, call_op_identifier, sfk_none);
quals = (LAMBDA_EXPR_MUTABLE_P (lambda_expr)
? TYPE_UNQUALIFIED : TYPE_QUAL_CONST);
declarator = make_call_declarator (declarator, param_list, quals,
VIRT_SPEC_UNSPECIFIED,
REF_QUAL_NONE,
tx_qual,
exception_spec,
NULL_TREE,
NULL_TREE);
declarator->id_loc = LAMBDA_EXPR_LOCATION (lambda_expr);
fco = grokmethod (&return_type_specs,
declarator,
attributes);
if (fco != error_mark_node)
{
DECL_INITIALIZED_IN_CLASS_P (fco) = 1;
DECL_ARTIFICIAL (fco) = 1;
DECL_NAME (DECL_ARGUMENTS (fco)) = get_identifier ("__closure");
DECL_LAMBDA_FUNCTION (fco) = 1;
if (return_type)
TYPE_HAS_LATE_RETURN_TYPE (TREE_TYPE (fco)) = 1;
}
if (template_param_list)
{
fco = finish_member_template_decl (fco);
finish_template_decl (template_param_list);
--parser->num_template_parameter_lists;
}
else if (parser->fully_implicit_function_template_p)
fco = finish_fully_implicit_template (parser, fco);
finish_member_declaration (fco);
obstack_free (&declarator_obstack, p);
return (fco != error_mark_node);
}
}
static void
cp_parser_lambda_body (cp_parser* parser, tree lambda_expr)
{
bool nested = (current_function_decl != NULL_TREE);
bool local_variables_forbidden_p = parser->local_variables_forbidden_p;
bool in_function_body = parser->in_function_body;
if (nested)
push_function_context ();
else
++function_depth;
vec<tree> omp_privatization_save;
save_omp_privatization_clauses (omp_privatization_save);
parser->local_variables_forbidden_p = false;
parser->in_function_body = true;
{
local_specialization_stack s (lss_copy);
tree fco = lambda_function (lambda_expr);
tree body = start_lambda_function (fco, lambda_expr);
matching_braces braces;
if (braces.require_open (parser))
{
tree compound_stmt = begin_compound_stmt (0);
while (cp_lexer_next_token_is_keyword (parser->lexer, RID_LABEL))
cp_parser_label_declaration (parser);
cp_parser_statement_seq_opt (parser, NULL_TREE);
braces.require_close (parser);
finish_compound_stmt (compound_stmt);
}
finish_lambda_function (body);
}
restore_omp_privatization_clauses (omp_privatization_save);
parser->local_variables_forbidden_p = local_variables_forbidden_p;
parser->in_function_body = in_function_body;
if (nested)
pop_function_context();
else
--function_depth;
}
static void
add_debug_begin_stmt (location_t loc)
{
if (!MAY_HAVE_DEBUG_MARKER_STMTS)
return;
if (DECL_DECLARED_CONCEPT_P (current_function_decl))
return;
tree stmt = build0 (DEBUG_BEGIN_STMT, void_type_node);
SET_EXPR_LOCATION (stmt, loc);
add_stmt (stmt);
}
static void
cp_parser_statement (cp_parser* parser, tree in_statement_expr,
bool in_compound, bool *if_p, vec<tree> *chain,
location_t *loc_after_labels)
{
tree statement, std_attrs = NULL_TREE;
cp_token *token;
location_t statement_location, attrs_location;
restart:
if (if_p != NULL)
*if_p = false;
statement = NULL_TREE;
saved_token_sentinel saved_tokens (parser->lexer);
attrs_location = cp_lexer_peek_token (parser->lexer)->location;
if (c_dialect_objc ())
cp_parser_parse_tentatively (parser);
std_attrs = cp_parser_std_attribute_spec_seq (parser);
if (c_dialect_objc ())
{
if (!cp_parser_parse_definitely (parser))
std_attrs = NULL_TREE;
}
token = cp_lexer_peek_token (parser->lexer);
statement_location = token->location;
add_debug_begin_stmt (statement_location);
if (token->type == CPP_KEYWORD)
{
enum rid keyword = token->keyword;
switch (keyword)
{
case RID_CASE:
case RID_DEFAULT:
cp_parser_label_for_labeled_statement (parser, std_attrs);
in_compound = false;
goto restart;
case RID_IF:
case RID_SWITCH:
statement = cp_parser_selection_statement (parser, if_p, chain);
break;
case RID_WHILE:
case RID_DO:
case RID_FOR:
statement = cp_parser_iteration_statement (parser, if_p, false, 0);
break;
case RID_BREAK:
case RID_CONTINUE:
case RID_RETURN:
case RID_GOTO:
statement = cp_parser_jump_statement (parser);
break;
case RID_AT_TRY:
case RID_AT_CATCH:
case RID_AT_FINALLY:
case RID_AT_SYNCHRONIZED:
case RID_AT_THROW:
statement = cp_parser_objc_statement (parser);
break;
case RID_TRY:
statement = cp_parser_try_block (parser);
break;
case RID_NAMESPACE:
cp_parser_declaration_statement (parser);
return;
case RID_TRANSACTION_ATOMIC:
case RID_TRANSACTION_RELAXED:
case RID_SYNCHRONIZED:
case RID_ATOMIC_NOEXCEPT:
case RID_ATOMIC_CANCEL:
statement = cp_parser_transaction (parser, token);
break;
case RID_TRANSACTION_CANCEL:
statement = cp_parser_transaction_cancel (parser);
break;
default:
break;
}
}
else if (token->type == CPP_NAME)
{
token = cp_lexer_peek_nth_token (parser->lexer, 2);
if (token->type == CPP_COLON)
{
cp_parser_label_for_labeled_statement (parser, std_attrs);
in_compound = false;
goto restart;
}
}
else if (token->type == CPP_OPEN_BRACE)
statement = cp_parser_compound_statement (parser, NULL, BCS_NORMAL, false);
else if (token->type == CPP_PRAGMA)
{
if (in_compound)
cp_parser_pragma (parser, pragma_compound, if_p);
else if (!cp_parser_pragma (parser, pragma_stmt, if_p))
goto restart;
return;
}
else if (token->type == CPP_EOF)
{
cp_parser_error (parser, "expected statement");
return;
}
if (!statement)
{
if (cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
{
if (std_attrs != NULL_TREE)
{
saved_tokens.rollback();
std_attrs = NULL_TREE;
}
cp_parser_parse_tentatively (parser);
cp_parser_declaration_statement (parser);
if (cp_parser_parse_definitely (parser))
return;
}
if (loc_after_labels != NULL)
*loc_after_labels = statement_location;
statement = cp_parser_expression_statement (parser, in_statement_expr);
if (attribute_fallthrough_p (std_attrs))
{
if (statement == NULL_TREE)
{
statement = build_call_expr_internal_loc (statement_location,
IFN_FALLTHROUGH,
void_type_node, 0);
finish_expr_stmt (statement);
}
else
warning_at (statement_location, OPT_Wattributes,
"%<fallthrough%> attribute not followed by %<;%>");
std_attrs = NULL_TREE;
}
}
if (statement && STATEMENT_CODE_P (TREE_CODE (statement)))
SET_EXPR_LOCATION (statement, statement_location);
if (std_attrs != NULL_TREE)
warning_at (attrs_location,
OPT_Wattributes,
"attributes at the beginning of statement are ignored");
}
static tree
attr_chainon (tree attrs, tree attr)
{
if (attrs == error_mark_node)
return error_mark_node;
if (attr == error_mark_node)
return error_mark_node;
return chainon (attrs, attr);
}
static void
cp_parser_label_for_labeled_statement (cp_parser* parser, tree attributes)
{
cp_token *token;
tree label = NULL_TREE;
bool saved_colon_corrects_to_scope_p = parser->colon_corrects_to_scope_p;
token = cp_lexer_peek_token (parser->lexer);
if (token->type != CPP_NAME
&& token->type != CPP_KEYWORD)
{
cp_parser_error (parser, "expected labeled-statement");
return;
}
bool fallthrough_p = token->flags & PREV_FALLTHROUGH;
parser->colon_corrects_to_scope_p = false;
switch (token->keyword)
{
case RID_CASE:
{
tree expr, expr_hi;
cp_token *ellipsis;
cp_lexer_consume_token (parser->lexer);
expr = cp_parser_constant_expression (parser);
if (check_for_bare_parameter_packs (expr))
expr = error_mark_node;
ellipsis = cp_lexer_peek_token (parser->lexer);
if (ellipsis->type == CPP_ELLIPSIS)
{
cp_lexer_consume_token (parser->lexer);
expr_hi = cp_parser_constant_expression (parser);
if (check_for_bare_parameter_packs (expr_hi))
expr_hi = error_mark_node;
}
else
expr_hi = NULL_TREE;
if (parser->in_switch_statement_p)
{
tree l = finish_case_label (token->location, expr, expr_hi);
if (l && TREE_CODE (l) == CASE_LABEL_EXPR)
FALLTHROUGH_LABEL_P (CASE_LABEL (l)) = fallthrough_p;
}
else
error_at (token->location,
"case label %qE not within a switch statement",
expr);
}
break;
case RID_DEFAULT:
cp_lexer_consume_token (parser->lexer);
if (parser->in_switch_statement_p)
{
tree l = finish_case_label (token->location, NULL_TREE, NULL_TREE);
if (l && TREE_CODE (l) == CASE_LABEL_EXPR)
FALLTHROUGH_LABEL_P (CASE_LABEL (l)) = fallthrough_p;
}
else
error_at (token->location, "case label not within a switch statement");
break;
default:
label = finish_label_stmt (cp_parser_identifier (parser));
if (label && TREE_CODE (label) == LABEL_DECL)
FALLTHROUGH_LABEL_P (label) = fallthrough_p;
break;
}
cp_parser_require (parser, CPP_COLON, RT_COLON);
if (label != NULL_TREE
&& cp_next_tokens_can_be_gnu_attribute_p (parser))
{
tree attrs;
cp_parser_parse_tentatively (parser);
attrs = cp_parser_gnu_attributes_opt (parser);
if (attrs == NULL_TREE
|| cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
cp_parser_abort_tentative_parse (parser);
else if (!cp_parser_parse_definitely (parser))
;
else
attributes = attr_chainon (attributes, attrs);
}
if (attributes != NULL_TREE)
cplus_decl_attributes (&label, attributes, 0);
parser->colon_corrects_to_scope_p = saved_colon_corrects_to_scope_p;
}
static tree
cp_parser_expression_statement (cp_parser* parser, tree in_statement_expr)
{
tree statement = NULL_TREE;
cp_token *token = cp_lexer_peek_token (parser->lexer);
location_t loc = token->location;
tree attr = cp_parser_gnu_attributes_opt (parser);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
{
statement = cp_parser_expression (parser);
if (statement == error_mark_node
&& !cp_parser_uncommitted_to_tentative_parse_p (parser))
{
cp_parser_skip_to_end_of_block_or_statement (parser);
return error_mark_node;
}
}
if (attribute_fallthrough_p (attr))
{
if (statement == NULL_TREE)
statement = build_call_expr_internal_loc (loc, IFN_FALLTHROUGH,
void_type_node, 0);
else
warning_at (loc, OPT_Wattributes,
"%<fallthrough%> attribute not followed by %<;%>");
attr = NULL_TREE;
}
if (attr != NULL_TREE)
warning_at (loc, OPT_Wattributes,
"attributes at the beginning of statement are ignored");
if (cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON)
&& !cp_parser_uncommitted_to_tentative_parse_p (parser))
{
if (TREE_CODE (statement) == SCOPE_REF)
error_at (token->location, "need %<typename%> before %qE because "
"%qT is a dependent scope",
statement, TREE_OPERAND (statement, 0));
else if (is_overloaded_fn (statement)
&& DECL_CONSTRUCTOR_P (get_first_fn (statement)))
{
tree fn = get_first_fn (statement);
error_at (token->location,
"%<%T::%D%> names the constructor, not the type",
DECL_CONTEXT (fn), DECL_NAME (fn));
}
}
cp_parser_consume_semicolon_at_end_of_statement (parser);
if (in_statement_expr
&& cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_BRACE))
statement = finish_stmt_expr_expr (statement, in_statement_expr);
else if (statement)
statement = finish_expr_stmt (statement);
return statement;
}
static tree
cp_parser_compound_statement (cp_parser *parser, tree in_statement_expr,
int bcs_flags, bool function_body)
{
tree compound_stmt;
matching_braces braces;
if (!braces.require_open (parser))
return error_mark_node;
if (DECL_DECLARED_CONSTEXPR_P (current_function_decl)
&& !function_body && cxx_dialect < cxx14)
pedwarn (input_location, OPT_Wpedantic,
"compound-statement in %<constexpr%> function");
compound_stmt = begin_compound_stmt (bcs_flags);
while (cp_lexer_next_token_is_keyword (parser->lexer, RID_LABEL))
cp_parser_label_declaration (parser);
cp_parser_statement_seq_opt (parser, in_statement_expr);
finish_compound_stmt (compound_stmt);
braces.require_close (parser);
return compound_stmt;
}
static void
cp_parser_statement_seq_opt (cp_parser* parser, tree in_statement_expr)
{
while (true)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_CLOSE_BRACE
|| token->type == CPP_EOF
|| token->type == CPP_PRAGMA_EOL
|| (token->type == CPP_KEYWORD && token->keyword == RID_AT_END))
break;
else if (token->type == CPP_KEYWORD && token->keyword == RID_ELSE)
{
if (parser->in_statement & IN_IF_STMT) 
break;
else
{
token = cp_lexer_consume_token (parser->lexer);
error_at (token->location, "%<else%> without a previous %<if%>");
}
}
cp_parser_statement (parser, in_statement_expr, true, NULL);
}
}
static bool
cp_parser_init_statement_p (cp_parser *parser)
{
cp_lexer_save_tokens (parser->lexer);
int ret = cp_parser_skip_to_closing_parenthesis_1 (parser,
false,
CPP_SEMICOLON,
false);
cp_lexer_rollback_tokens (parser->lexer);
return ret == -1;
}
static tree
cp_parser_selection_statement (cp_parser* parser, bool *if_p,
vec<tree> *chain)
{
cp_token *token;
enum rid keyword;
token_indent_info guard_tinfo;
if (if_p != NULL)
*if_p = false;
token = cp_parser_require (parser, CPP_KEYWORD, RT_SELECT);
guard_tinfo = get_token_indent_info (token);
keyword = token->keyword;
switch (keyword)
{
case RID_IF:
case RID_SWITCH:
{
tree statement;
tree condition;
bool cx = false;
if (keyword == RID_IF
&& cp_lexer_next_token_is_keyword (parser->lexer,
RID_CONSTEXPR))
{
cx = true;
cp_token *tok = cp_lexer_consume_token (parser->lexer);
if (cxx_dialect < cxx17 && !in_system_header_at (tok->location))
pedwarn (tok->location, 0, "%<if constexpr%> only available "
"with -std=c++17 or -std=gnu++17");
}
matching_parens parens;
if (!parens.require_open (parser))
{
cp_parser_skip_to_end_of_statement (parser);
return error_mark_node;
}
if (keyword == RID_IF)
{
statement = begin_if_stmt ();
IF_STMT_CONSTEXPR_P (statement) = cx;
}
else
statement = begin_switch_stmt ();
if (cp_parser_init_statement_p (parser))
{
tree decl;
if (cxx_dialect < cxx17)
pedwarn (cp_lexer_peek_token (parser->lexer)->location, 0,
"init-statement in selection statements only available "
"with -std=c++17 or -std=gnu++17");
cp_parser_init_statement (parser, &decl);
}
condition = cp_parser_condition (parser);
if (!parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true, false,
true);
if (keyword == RID_IF)
{
bool nested_if;
unsigned char in_statement;
condition = finish_if_stmt_cond (condition, statement);
if (warn_duplicated_cond)
warn_duplicated_cond_add_or_warn (token->location, condition,
&chain);
in_statement = parser->in_statement;
parser->in_statement |= IN_IF_STMT;
bool was_discarded = in_discarded_stmt;
bool discard_then = (cx && !processing_template_decl
&& integer_zerop (condition));
if (discard_then)
{
in_discarded_stmt = true;
++c_inhibit_evaluation_warnings;
}
cp_parser_implicitly_scoped_statement (parser, &nested_if,
guard_tinfo);
parser->in_statement = in_statement;
finish_then_clause (statement);
if (discard_then)
{
THEN_CLAUSE (statement) = NULL_TREE;
in_discarded_stmt = was_discarded;
--c_inhibit_evaluation_warnings;
}
if (cp_lexer_next_token_is_keyword (parser->lexer,
RID_ELSE))
{
bool discard_else = (cx && !processing_template_decl
&& integer_nonzerop (condition));
if (discard_else)
{
in_discarded_stmt = true;
++c_inhibit_evaluation_warnings;
}
guard_tinfo
= get_token_indent_info (cp_lexer_peek_token (parser->lexer));
cp_lexer_consume_token (parser->lexer);
if (warn_duplicated_cond)
{
if (cp_lexer_next_token_is_keyword (parser->lexer,
RID_IF)
&& chain == NULL)
{
chain = new vec<tree> ();
if (!CONSTANT_CLASS_P (condition)
&& !TREE_SIDE_EFFECTS (condition))
{
tree e = build1 (NOP_EXPR, TREE_TYPE (condition),
condition);
SET_EXPR_LOCATION (e, token->location);
chain->safe_push (e);
}
}
else if (!cp_lexer_next_token_is_keyword (parser->lexer,
RID_IF))
{
delete chain;
chain = NULL;
}
}
begin_else_clause (statement);
cp_parser_implicitly_scoped_statement (parser, NULL,
guard_tinfo, chain);
finish_else_clause (statement);
if (if_p != NULL)
*if_p = true;
if (discard_else)
{
ELSE_CLAUSE (statement) = NULL_TREE;
in_discarded_stmt = was_discarded;
--c_inhibit_evaluation_warnings;
}
}
else
{
if (nested_if)
warning_at (EXPR_LOCATION (statement), OPT_Wdangling_else,
"suggest explicit braces to avoid ambiguous"
" %<else%>");
if (warn_duplicated_cond)
{
delete chain;
chain = NULL;
}
}
finish_if_stmt (statement);
}
else
{
bool in_switch_statement_p;
unsigned char in_statement;
finish_switch_cond (condition, statement);
in_switch_statement_p = parser->in_switch_statement_p;
in_statement = parser->in_statement;
parser->in_switch_statement_p = true;
parser->in_statement |= IN_SWITCH_STMT;
cp_parser_implicitly_scoped_statement (parser, if_p,
guard_tinfo);
parser->in_switch_statement_p = in_switch_statement_p;
parser->in_statement = in_statement;
finish_switch_stmt (statement);
}
return statement;
}
break;
default:
cp_parser_error (parser, "expected selection-statement");
return error_mark_node;
}
}
static tree
cp_parser_condition (cp_parser* parser)
{
cp_decl_specifier_seq type_specifiers;
const char *saved_message;
int declares_class_or_enum;
cp_parser_parse_tentatively (parser);
saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in conditions");
cp_parser_decl_specifier_seq (parser,
CP_PARSER_FLAGS_ONLY_TYPE_OR_CONSTEXPR,
&type_specifiers,
&declares_class_or_enum);
parser->type_definition_forbidden_message = saved_message;
if (!cp_parser_error_occurred (parser))
{
tree decl;
tree asm_specification;
tree attributes;
cp_declarator *declarator;
tree initializer = NULL_TREE;
declarator = cp_parser_declarator (parser, CP_PARSER_DECLARATOR_NAMED,
NULL,
NULL,
false,
false);
attributes = cp_parser_attributes_opt (parser);
asm_specification = cp_parser_asm_specification_opt (parser);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_EQ)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_OPEN_BRACE))
cp_parser_simulate_error (parser);
if (cp_parser_parse_definitely (parser))
{
tree pushed_scope;
bool non_constant_p;
int flags = LOOKUP_ONLYCONVERTING;
decl = start_decl (declarator, &type_specifiers,
true,
attributes, NULL_TREE,
&pushed_scope);
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
initializer = cp_parser_braced_list (parser, &non_constant_p);
CONSTRUCTOR_IS_DIRECT_INIT (initializer) = 1;
flags = 0;
}
else
{
cp_parser_require (parser, CPP_EQ, RT_EQ);
initializer = cp_parser_initializer_clause (parser, &non_constant_p);
}
if (BRACE_ENCLOSED_INITIALIZER_P (initializer))
maybe_warn_cpp0x (CPP0X_INITIALIZER_LISTS);
cp_finish_decl (decl,
initializer, !non_constant_p,
asm_specification,
flags);
if (pushed_scope)
pop_scope (pushed_scope);
return convert_from_reference (decl);
}
}
else
cp_parser_abort_tentative_parse (parser);
return cp_parser_expression (parser);
}
static tree
cp_parser_for (cp_parser *parser, bool ivdep, unsigned short unroll)
{
tree init, scope, decl;
bool is_range_for;
scope = begin_for_scope (&init);
is_range_for = cp_parser_init_statement (parser, &decl);
if (is_range_for)
return cp_parser_range_for (parser, scope, init, decl, ivdep, unroll);
else
return cp_parser_c_for (parser, scope, init, ivdep, unroll);
}
static tree
cp_parser_c_for (cp_parser *parser, tree scope, tree init, bool ivdep,
unsigned short unroll)
{
tree condition = NULL_TREE;
tree expression = NULL_TREE;
tree stmt;
stmt = begin_for_stmt (scope, init);
finish_init_stmt (stmt);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
condition = cp_parser_condition (parser);
else if (ivdep)
{
cp_parser_error (parser, "missing loop condition in loop with "
"%<GCC ivdep%> pragma");
condition = error_mark_node;
}
else if (unroll)
{
cp_parser_error (parser, "missing loop condition in loop with "
"%<GCC unroll%> pragma");
condition = error_mark_node;
}
finish_for_cond (condition, stmt, ivdep, unroll);
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_CLOSE_PAREN))
expression = cp_parser_expression (parser);
finish_for_expr (expression, stmt);
return stmt;
}
static tree
cp_parser_range_for (cp_parser *parser, tree scope, tree init, tree range_decl,
bool ivdep, unsigned short unroll)
{
tree stmt, range_expr;
auto_vec <cxx_binding *, 16> bindings;
auto_vec <tree, 16> names;
tree decomp_first_name = NULL_TREE;
unsigned int decomp_cnt = 0;
if (range_decl != error_mark_node)
{
if (DECL_HAS_VALUE_EXPR_P (range_decl))
{
tree v = DECL_VALUE_EXPR (range_decl);
if (TREE_CODE (v) == ARRAY_REF
&& VAR_P (TREE_OPERAND (v, 0))
&& DECL_DECOMPOSITION_P (TREE_OPERAND (v, 0)))
{
tree d = range_decl;
range_decl = TREE_OPERAND (v, 0);
decomp_cnt = tree_to_uhwi (TREE_OPERAND (v, 1)) + 1;
decomp_first_name = d;
for (unsigned int i = 0; i < decomp_cnt; i++, d = DECL_CHAIN (d))
{
tree name = DECL_NAME (d);
names.safe_push (name);
bindings.safe_push (IDENTIFIER_BINDING (name));
IDENTIFIER_BINDING (name)
= IDENTIFIER_BINDING (name)->previous;
}
}
}
if (names.is_empty ())
{
tree name = DECL_NAME (range_decl);
names.safe_push (name);
bindings.safe_push (IDENTIFIER_BINDING (name));
IDENTIFIER_BINDING (name) = IDENTIFIER_BINDING (name)->previous;
}
}
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
bool expr_non_constant_p;
range_expr = cp_parser_braced_list (parser, &expr_non_constant_p);
}
else
range_expr = cp_parser_expression (parser);
for (unsigned int i = 0; i < names.length (); i++)
{
cxx_binding *binding = bindings[i];
binding->previous = IDENTIFIER_BINDING (names[i]);
IDENTIFIER_BINDING (names[i]) = binding;
}
if (processing_template_decl)
{
if (check_for_bare_parameter_packs (range_expr))
range_expr = error_mark_node;
stmt = begin_range_for_stmt (scope, init);
if (ivdep)
RANGE_FOR_IVDEP (stmt) = 1;
if (unroll)
RANGE_FOR_UNROLL (stmt) = build_int_cst (integer_type_node, unroll);
finish_range_for_decl (stmt, range_decl, range_expr);
if (!type_dependent_expression_p (range_expr)
&& !BRACE_ENCLOSED_INITIALIZER_P (range_expr))
do_range_for_auto_deduction (range_decl, range_expr);
}
else
{
stmt = begin_for_stmt (scope, init);
stmt = cp_convert_range_for (stmt, range_decl, range_expr,
decomp_first_name, decomp_cnt, ivdep,
unroll);
}
return stmt;
}
static tree
build_range_temp (tree range_expr)
{
tree range_type, range_temp;
range_type = cp_build_reference_type (make_auto (), true);
range_type = do_auto_deduction (range_type, range_expr,
type_uses_auto (range_type));
range_temp = build_decl (input_location, VAR_DECL,
get_identifier ("__for_range"), range_type);
TREE_USED (range_temp) = 1;
DECL_ARTIFICIAL (range_temp) = 1;
return range_temp;
}
static void
do_range_for_auto_deduction (tree decl, tree range_expr)
{
tree auto_node = type_uses_auto (TREE_TYPE (decl));
if (auto_node)
{
tree begin_dummy, end_dummy, range_temp, iter_type, iter_decl;
range_temp = convert_from_reference (build_range_temp (range_expr));
iter_type = (cp_parser_perform_range_for_lookup
(range_temp, &begin_dummy, &end_dummy));
if (iter_type)
{
iter_decl = build_decl (input_location, VAR_DECL, NULL_TREE,
iter_type);
iter_decl = build_x_indirect_ref (input_location, iter_decl,
RO_UNARY_STAR,
tf_warning_or_error);
TREE_TYPE (decl) = do_auto_deduction (TREE_TYPE (decl),
iter_decl, auto_node);
}
}
}
tree
cp_convert_range_for (tree statement, tree range_decl, tree range_expr,
tree decomp_first_name, unsigned int decomp_cnt,
bool ivdep, unsigned short unroll)
{
tree begin, end;
tree iter_type, begin_expr, end_expr;
tree condition, expression;
range_expr = mark_lvalue_use (range_expr);
if (range_decl == error_mark_node || range_expr == error_mark_node)
begin_expr = end_expr = iter_type = error_mark_node;
else
{
tree range_temp;
if (VAR_P (range_expr)
&& array_of_runtime_bound_p (TREE_TYPE (range_expr)))
range_temp = range_expr;
else
{
range_temp = build_range_temp (range_expr);
pushdecl (range_temp);
cp_finish_decl (range_temp, range_expr,
false, NULL_TREE,
LOOKUP_ONLYCONVERTING);
range_temp = convert_from_reference (range_temp);
}
iter_type = cp_parser_perform_range_for_lookup (range_temp,
&begin_expr, &end_expr);
}
begin = build_decl (input_location, VAR_DECL,
get_identifier ("__for_begin"), iter_type);
TREE_USED (begin) = 1;
DECL_ARTIFICIAL (begin) = 1;
pushdecl (begin);
cp_finish_decl (begin, begin_expr,
false, NULL_TREE,
LOOKUP_ONLYCONVERTING);
if (cxx_dialect >= cxx17)
iter_type = cv_unqualified (TREE_TYPE (end_expr));
end = build_decl (input_location, VAR_DECL,
get_identifier ("__for_end"), iter_type);
TREE_USED (end) = 1;
DECL_ARTIFICIAL (end) = 1;
pushdecl (end);
cp_finish_decl (end, end_expr,
false, NULL_TREE,
LOOKUP_ONLYCONVERTING);
finish_init_stmt (statement);
condition = build_x_binary_op (input_location, NE_EXPR,
begin, ERROR_MARK,
end, ERROR_MARK,
NULL, tf_warning_or_error);
finish_for_cond (condition, statement, ivdep, unroll);
expression = finish_unary_op_expr (input_location,
PREINCREMENT_EXPR, begin,
tf_warning_or_error);
finish_for_expr (expression, statement);
if (VAR_P (range_decl) && DECL_DECOMPOSITION_P (range_decl))
cp_maybe_mangle_decomp (range_decl, decomp_first_name, decomp_cnt);
cp_finish_decl (range_decl,
build_x_indirect_ref (input_location, begin, RO_UNARY_STAR,
tf_warning_or_error),
false, NULL_TREE,
LOOKUP_ONLYCONVERTING);
if (VAR_P (range_decl) && DECL_DECOMPOSITION_P (range_decl))
cp_finish_decomp (range_decl, decomp_first_name, decomp_cnt);
return statement;
}
static tree
cp_parser_perform_range_for_lookup (tree range, tree *begin, tree *end)
{
if (error_operand_p (range))
{
*begin = *end = error_mark_node;
return error_mark_node;
}
if (!COMPLETE_TYPE_P (complete_type (TREE_TYPE (range))))
{
error ("range-based %<for%> expression of type %qT "
"has incomplete type", TREE_TYPE (range));
*begin = *end = error_mark_node;
return error_mark_node;
}
if (TREE_CODE (TREE_TYPE (range)) == ARRAY_TYPE)
{
*begin = decay_conversion (range, tf_warning_or_error);
*end = build_binary_op (input_location, PLUS_EXPR,
range,
array_type_nelts_top (TREE_TYPE (range)),
false);
return TREE_TYPE (*begin);
}
else
{
tree id_begin, id_end;
tree member_begin, member_end;
*begin = *end = error_mark_node;
id_begin = get_identifier ("begin");
id_end = get_identifier ("end");
member_begin = lookup_member (TREE_TYPE (range), id_begin,
2, false,
tf_warning_or_error);
member_end = lookup_member (TREE_TYPE (range), id_end,
2, false,
tf_warning_or_error);
if (member_begin != NULL_TREE && member_end != NULL_TREE)
{
*begin = cp_parser_range_for_member_function (range, id_begin);
*end = cp_parser_range_for_member_function (range, id_end);
}
else
{
vec<tree, va_gc> *vec;
vec = make_tree_vector ();
vec_safe_push (vec, range);
member_begin = perform_koenig_lookup (id_begin, vec,
tf_warning_or_error);
*begin = finish_call_expr (member_begin, &vec, false, true,
tf_warning_or_error);
member_end = perform_koenig_lookup (id_end, vec,
tf_warning_or_error);
*end = finish_call_expr (member_end, &vec, false, true,
tf_warning_or_error);
release_tree_vector (vec);
}
if (*begin == error_mark_node || *end == error_mark_node)
{
*begin = *end = error_mark_node;
return error_mark_node;
}
else if (type_dependent_expression_p (*begin)
|| type_dependent_expression_p (*end))
return NULL_TREE;
else
{
tree iter_type = cv_unqualified (TREE_TYPE (*begin));
if (!same_type_p (iter_type, cv_unqualified (TREE_TYPE (*end))))
{
if (cxx_dialect >= cxx17
&& (build_x_binary_op (input_location, NE_EXPR,
*begin, ERROR_MARK,
*end, ERROR_MARK,
NULL, tf_none)
!= error_mark_node))
;
else
error ("inconsistent begin/end types in range-based %<for%> "
"statement: %qT and %qT",
TREE_TYPE (*begin), TREE_TYPE (*end));
}
return iter_type;
}
}
}
static tree
cp_parser_range_for_member_function (tree range, tree identifier)
{
tree member, res;
vec<tree, va_gc> *vec;
member = finish_class_member_access_expr (range, identifier,
false, tf_warning_or_error);
if (member == error_mark_node)
return error_mark_node;
vec = make_tree_vector ();
res = finish_call_expr (member, &vec,
false,
false,
tf_warning_or_error);
release_tree_vector (vec);
return res;
}
static tree
cp_parser_iteration_statement (cp_parser* parser, bool *if_p, bool ivdep,
unsigned short unroll)
{
cp_token *token;
enum rid keyword;
tree statement;
unsigned char in_statement;
token_indent_info guard_tinfo;
token = cp_parser_require (parser, CPP_KEYWORD, RT_ITERATION);
if (!token)
return error_mark_node;
guard_tinfo = get_token_indent_info (token);
in_statement = parser->in_statement;
keyword = token->keyword;
switch (keyword)
{
case RID_WHILE:
{
tree condition;
statement = begin_while_stmt ();
matching_parens parens;
parens.require_open (parser);
condition = cp_parser_condition (parser);
finish_while_stmt_cond (condition, statement, ivdep, unroll);
parens.require_close (parser);
parser->in_statement = IN_ITERATION_STMT;
bool prev = note_iteration_stmt_body_start ();
cp_parser_already_scoped_statement (parser, if_p, guard_tinfo);
note_iteration_stmt_body_end (prev);
parser->in_statement = in_statement;
finish_while_stmt (statement);
}
break;
case RID_DO:
{
tree expression;
statement = begin_do_stmt ();
parser->in_statement = IN_ITERATION_STMT;
bool prev = note_iteration_stmt_body_start ();
cp_parser_implicitly_scoped_statement (parser, NULL, guard_tinfo);
note_iteration_stmt_body_end (prev);
parser->in_statement = in_statement;
finish_do_body (statement);
cp_parser_require_keyword (parser, RID_WHILE, RT_WHILE);
matching_parens parens;
parens.require_open (parser);
expression = cp_parser_expression (parser);
finish_do_stmt (expression, statement, ivdep, unroll);
parens.require_close (parser);
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
}
break;
case RID_FOR:
{
matching_parens parens;
parens.require_open (parser);
statement = cp_parser_for (parser, ivdep, unroll);
parens.require_close (parser);
parser->in_statement = IN_ITERATION_STMT;
bool prev = note_iteration_stmt_body_start ();
cp_parser_already_scoped_statement (parser, if_p, guard_tinfo);
note_iteration_stmt_body_end (prev);
parser->in_statement = in_statement;
finish_for_stmt (statement);
}
break;
default:
cp_parser_error (parser, "expected iteration-statement");
statement = error_mark_node;
break;
}
return statement;
}
static bool
cp_parser_init_statement (cp_parser* parser, tree *decl)
{
if (cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
{
bool is_range_for = false;
bool saved_colon_corrects_to_scope_p = parser->colon_corrects_to_scope_p;
parser->colon_corrects_to_scope_p = false;
cp_parser_parse_tentatively (parser);
cp_parser_simple_declaration (parser,
false,
decl);
parser->colon_corrects_to_scope_p = saved_colon_corrects_to_scope_p;
if (cp_lexer_next_token_is (parser->lexer, CPP_COLON))
{
cp_lexer_consume_token (parser->lexer);
is_range_for = true;
if (cxx_dialect < cxx11)
pedwarn (cp_lexer_peek_token (parser->lexer)->location, 0,
"range-based %<for%> loops only available with "
"-std=c++11 or -std=gnu++11");
}
else
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
if (cp_parser_parse_definitely (parser))
return is_range_for;
}
cp_parser_expression_statement (parser, NULL_TREE);
return false;
}
static tree
cp_parser_jump_statement (cp_parser* parser)
{
tree statement = error_mark_node;
cp_token *token;
enum rid keyword;
unsigned char in_statement;
token = cp_parser_require (parser, CPP_KEYWORD, RT_JUMP);
if (!token)
return error_mark_node;
keyword = token->keyword;
switch (keyword)
{
case RID_BREAK:
in_statement = parser->in_statement & ~IN_IF_STMT;      
switch (in_statement)
{
case 0:
error_at (token->location, "break statement not within loop or switch");
break;
default:
gcc_assert ((in_statement & IN_SWITCH_STMT)
|| in_statement == IN_ITERATION_STMT);
statement = finish_break_stmt ();
if (in_statement == IN_ITERATION_STMT)
break_maybe_infinite_loop ();
break;
case IN_OMP_BLOCK:
error_at (token->location, "invalid exit from OpenMP structured block");
break;
case IN_OMP_FOR:
error_at (token->location, "break statement used with OpenMP for loop");
break;
}
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
break;
case RID_CONTINUE:
switch (parser->in_statement & ~(IN_SWITCH_STMT | IN_IF_STMT))
{
case 0:
error_at (token->location, "continue statement not within a loop");
break;
case IN_ITERATION_STMT:
case IN_OMP_FOR:
statement = finish_continue_stmt ();
break;
case IN_OMP_BLOCK:
error_at (token->location, "invalid exit from OpenMP structured block");
break;
default:
gcc_unreachable ();
}
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
break;
case RID_RETURN:
{
tree expr;
bool expr_non_constant_p;
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
cp_lexer_set_source_position (parser->lexer);
maybe_warn_cpp0x (CPP0X_INITIALIZER_LISTS);
expr = cp_parser_braced_list (parser, &expr_non_constant_p);
}
else if (cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
expr = cp_parser_expression (parser);
else
expr = NULL_TREE;
if (current_function_auto_return_pattern && in_discarded_stmt)
;
else
statement = finish_return_stmt (expr);
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
}
break;
case RID_GOTO:
if (parser->in_function_body
&& DECL_DECLARED_CONSTEXPR_P (current_function_decl))
{
error ("%<goto%> in %<constexpr%> function");
cp_function_chain->invalid_constexpr = true;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_MULT))
{
pedwarn (token->location, OPT_Wpedantic, "ISO C++ forbids computed gotos");
cp_lexer_consume_token (parser->lexer);
finish_goto_stmt (cp_parser_expression (parser));
}
else
finish_goto_stmt (cp_parser_identifier (parser));
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
break;
default:
cp_parser_error (parser, "expected jump-statement");
break;
}
return statement;
}
static void
cp_parser_declaration_statement (cp_parser* parser)
{
void *p;
p = obstack_alloc (&declarator_obstack, 0);
cp_parser_block_declaration (parser, true);
obstack_free (&declarator_obstack, p);
}
static tree
cp_parser_implicitly_scoped_statement (cp_parser* parser, bool *if_p,
const token_indent_info &guard_tinfo,
vec<tree> *chain)
{
tree statement;
location_t body_loc = cp_lexer_peek_token (parser->lexer)->location;
location_t body_loc_after_labels = UNKNOWN_LOCATION;
token_indent_info body_tinfo
= get_token_indent_info (cp_lexer_peek_token (parser->lexer));
if (if_p != NULL)
*if_p = false;
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
{
cp_lexer_consume_token (parser->lexer);
statement = add_stmt (build_empty_stmt (body_loc));
if (guard_tinfo.keyword == RID_IF
&& !cp_lexer_next_token_is_keyword (parser->lexer, RID_ELSE))
warning_at (body_loc, OPT_Wempty_body,
"suggest braces around empty body in an %<if%> statement");
else if (guard_tinfo.keyword == RID_ELSE)
warning_at (body_loc, OPT_Wempty_body,
"suggest braces around empty body in an %<else%> statement");
}
else if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
statement = cp_parser_compound_statement (parser, NULL, BCS_NORMAL, false);
else
{
statement = begin_compound_stmt (0);
cp_parser_statement (parser, NULL_TREE, false, if_p, chain,
&body_loc_after_labels);
finish_compound_stmt (statement);
}
token_indent_info next_tinfo
= get_token_indent_info (cp_lexer_peek_token (parser->lexer));
warn_for_misleading_indentation (guard_tinfo, body_tinfo, next_tinfo);
if (body_loc_after_labels != UNKNOWN_LOCATION
&& next_tinfo.type != CPP_SEMICOLON)
warn_for_multistatement_macros (body_loc_after_labels, next_tinfo.location,
guard_tinfo.location, guard_tinfo.keyword);
return statement;
}
static void
cp_parser_already_scoped_statement (cp_parser* parser, bool *if_p,
const token_indent_info &guard_tinfo)
{
if (cp_lexer_next_token_is_not (parser->lexer, CPP_OPEN_BRACE))
{
token_indent_info body_tinfo
= get_token_indent_info (cp_lexer_peek_token (parser->lexer));
location_t loc_after_labels = UNKNOWN_LOCATION;
cp_parser_statement (parser, NULL_TREE, false, if_p, NULL,
&loc_after_labels);
token_indent_info next_tinfo
= get_token_indent_info (cp_lexer_peek_token (parser->lexer));
warn_for_misleading_indentation (guard_tinfo, body_tinfo, next_tinfo);
if (loc_after_labels != UNKNOWN_LOCATION
&& next_tinfo.type != CPP_SEMICOLON)
warn_for_multistatement_macros (loc_after_labels, next_tinfo.location,
guard_tinfo.location,
guard_tinfo.keyword);
}
else
{
matching_braces braces;
braces.require_open (parser);
while (cp_lexer_next_token_is_keyword (parser->lexer, RID_LABEL))
cp_parser_label_declaration (parser);
cp_parser_statement_seq_opt (parser, NULL_TREE);
braces.require_close (parser);
}
}
static void
cp_parser_declaration_seq_opt (cp_parser* parser)
{
while (true)
{
cp_token *token;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_CLOSE_BRACE
|| token->type == CPP_EOF
|| token->type == CPP_PRAGMA_EOL)
break;
if (token->type == CPP_SEMICOLON)
{
cp_lexer_consume_token (parser->lexer);
if (!in_system_header_at (input_location))
pedwarn (input_location, OPT_Wpedantic, "extra %<;%>");
continue;
}
if (!parser->implicit_extern_c && token->implicit_extern_c)
{
push_lang_context (lang_name_c);
parser->implicit_extern_c = true;
}
else if (parser->implicit_extern_c && !token->implicit_extern_c)
{
pop_lang_context ();
parser->implicit_extern_c = false;
}
if (token->type == CPP_PRAGMA)
{
cp_parser_pragma (parser, pragma_external, NULL);
continue;
}
cp_parser_declaration (parser);
}
}
static void
cp_parser_declaration (cp_parser* parser)
{
cp_token token1;
cp_token token2;
int saved_pedantic;
void *p;
tree attributes = NULL_TREE;
if (cp_parser_extension_opt (parser, &saved_pedantic))
{
cp_parser_declaration (parser);
pedantic = saved_pedantic;
return;
}
token1 = *cp_lexer_peek_token (parser->lexer);
if (token1.type != CPP_EOF)
token2 = *cp_lexer_peek_nth_token (parser->lexer, 2);
else
{
token2.type = CPP_EOF;
token2.keyword = RID_MAX;
}
p = obstack_alloc (&declarator_obstack, 0);
if (token1.keyword == RID_EXTERN
&& cp_parser_is_pure_string_literal (&token2))
cp_parser_linkage_specification (parser);
else if (token1.keyword == RID_TEMPLATE)
{
if (token2.type == CPP_LESS
&& cp_lexer_peek_nth_token (parser->lexer, 3)->type == CPP_GREATER)
cp_parser_explicit_specialization (parser);
else if (token2.type == CPP_LESS)
cp_parser_template_declaration (parser, false);
else
cp_parser_explicit_instantiation (parser);
}
else if (token1.keyword == RID_EXPORT)
cp_parser_template_declaration (parser, false);
else if (cp_parser_allow_gnu_extensions_p (parser)
&& (token1.keyword == RID_EXTERN
|| token1.keyword == RID_STATIC
|| token1.keyword == RID_INLINE)
&& token2.keyword == RID_TEMPLATE)
cp_parser_explicit_instantiation (parser);
else if (token1.keyword == RID_NAMESPACE
&& (
(token2.type == CPP_NAME
&& (cp_lexer_peek_nth_token (parser->lexer, 3)->type
!= CPP_EQ))
|| (token2.type == CPP_OPEN_SQUARE
&& cp_lexer_peek_nth_token (parser->lexer, 3)->type
== CPP_OPEN_SQUARE)
|| token2.type == CPP_OPEN_BRACE
|| token2.keyword == RID_ATTRIBUTE))
cp_parser_namespace_definition (parser);
else if (token1.keyword == RID_INLINE
&& token2.keyword == RID_NAMESPACE)
cp_parser_namespace_definition (parser);
else if (c_dialect_objc () && OBJC_IS_AT_KEYWORD (token1.keyword))
cp_parser_objc_declaration (parser, NULL_TREE);
else if (c_dialect_objc ()
&& token1.keyword == RID_ATTRIBUTE
&& cp_parser_objc_valid_prefix_attributes (parser, &attributes))
cp_parser_objc_declaration (parser, attributes);
else if (flag_concepts
&& cp_parser_template_declaration_after_export (parser,
false))
;
else
cp_parser_block_declaration (parser, false);
obstack_free (&declarator_obstack, p);
}
static void
cp_parser_block_declaration (cp_parser *parser,
bool      statement_p)
{
cp_token *token1;
int saved_pedantic;
if (cp_parser_extension_opt (parser, &saved_pedantic))
{
cp_parser_block_declaration (parser, statement_p);
pedantic = saved_pedantic;
return;
}
token1 = cp_lexer_peek_token (parser->lexer);
if (token1->keyword == RID_ASM)
{
if (statement_p)
cp_parser_commit_to_tentative_parse (parser);
cp_parser_asm_definition (parser);
}
else if (token1->keyword == RID_NAMESPACE)
cp_parser_namespace_alias_definition (parser);
else if (token1->keyword == RID_USING)
{
cp_token *token2;
if (statement_p)
cp_parser_commit_to_tentative_parse (parser);
token2 = cp_lexer_peek_nth_token (parser->lexer, 2);
if (token2->keyword == RID_NAMESPACE)
cp_parser_using_directive (parser);
else if (cxx_dialect >= cxx11
&& token2->type == CPP_NAME
&& ((cp_lexer_peek_nth_token (parser->lexer, 3)->type == CPP_EQ)
|| (cp_nth_tokens_can_be_attribute_p (parser, 3))))
cp_parser_alias_declaration (parser);
else
cp_parser_using_declaration (parser,
false);
}
else if (token1->keyword == RID_LABEL)
{
cp_lexer_consume_token (parser->lexer);
error_at (token1->location, "%<__label__%> not at the beginning of a block");
cp_parser_skip_to_end_of_statement (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
cp_lexer_consume_token (parser->lexer);
}
else if (token1->keyword == RID_STATIC_ASSERT)
cp_parser_static_assert (parser, false);
else
cp_parser_simple_declaration (parser, !statement_p,
NULL);
}
static void
cp_parser_simple_declaration (cp_parser* parser,
bool function_definition_allowed_p,
tree *maybe_range_for_decl)
{
cp_decl_specifier_seq decl_specifiers;
int declares_class_or_enum;
bool saw_declarator;
location_t comma_loc = UNKNOWN_LOCATION;
location_t init_loc = UNKNOWN_LOCATION;
if (maybe_range_for_decl)
*maybe_range_for_decl = NULL_TREE;
push_deferring_access_checks (dk_deferred);
cp_parser_decl_specifier_seq (parser,
CP_PARSER_FLAGS_OPTIONAL,
&decl_specifiers,
&declares_class_or_enum);
stop_deferring_access_checks ();
if (!function_definition_allowed_p
&& !decl_specifiers.any_specifiers_p)
{
cp_parser_error (parser, "expected declaration");
goto done;
}
if (!decl_specifiers.any_type_specifiers_p
&& cp_parser_parse_and_diagnose_invalid_type_name (parser))
{
cp_parser_commit_to_tentative_parse (parser);
goto done;
}
if (decl_specifiers.any_specifiers_p
&& cp_lexer_next_token_is_not (parser->lexer, CPP_OPEN_PAREN)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_OPEN_BRACE)
&& !cp_parser_error_occurred (parser))
cp_parser_commit_to_tentative_parse (parser);
for (size_t n = 1; ; n++)
if (cp_lexer_nth_token_is (parser->lexer, n, CPP_AND)
|| cp_lexer_nth_token_is (parser->lexer, n, CPP_AND_AND))
continue;
else if (cp_lexer_nth_token_is (parser->lexer, n, CPP_OPEN_SQUARE)
&& !cp_lexer_nth_token_is (parser->lexer, n + 1, CPP_OPEN_SQUARE)
&& decl_specifiers.any_specifiers_p)
{
tree decl
= cp_parser_decomposition_declaration (parser, &decl_specifiers,
maybe_range_for_decl,
&init_loc);
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_SEMICOLON)
goto finish;
else if (maybe_range_for_decl)
{
if (*maybe_range_for_decl == NULL_TREE)
*maybe_range_for_decl = error_mark_node;
goto finish;
}
else
{
if ((decl != error_mark_node
&& DECL_INITIAL (decl) != error_mark_node)
|| cp_parser_uncommitted_to_tentative_parse_p (parser))
cp_parser_error (parser, "expected %<,%> or %<;%>");
cp_parser_skip_to_end_of_statement (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
cp_lexer_consume_token (parser->lexer);
goto done;
}
}
else
break;
tree last_type;
bool auto_specifier_p;
tree auto_function_declaration;
last_type = NULL_TREE;
auto_specifier_p
= decl_specifiers.type && type_uses_auto (decl_specifiers.type);
auto_function_declaration = NULL_TREE;
saw_declarator = false;
while (cp_lexer_next_token_is_not (parser->lexer,
CPP_SEMICOLON))
{
cp_token *token;
bool function_definition_p;
tree decl;
tree auto_result = NULL_TREE;
if (saw_declarator)
{
token = cp_lexer_peek_token (parser->lexer);
gcc_assert (token->type == CPP_COMMA);
cp_lexer_consume_token (parser->lexer);
if (maybe_range_for_decl)
{
*maybe_range_for_decl = error_mark_node;
if (comma_loc == UNKNOWN_LOCATION)
comma_loc = token->location;
}
}
else
saw_declarator = true;
decl = cp_parser_init_declarator (parser, &decl_specifiers,
NULL,
function_definition_allowed_p,
false,
declares_class_or_enum,
&function_definition_p,
maybe_range_for_decl,
&init_loc,
&auto_result);
if (cp_parser_error_occurred (parser))
goto done;
if (auto_specifier_p && cxx_dialect >= cxx14)
{
if (auto_function_declaration == NULL_TREE)
auto_function_declaration
= TREE_CODE (decl) == FUNCTION_DECL ? decl : error_mark_node;
else if (TREE_CODE (decl) == FUNCTION_DECL
|| auto_function_declaration != error_mark_node)
{
error_at (decl_specifiers.locations[ds_type_spec],
"non-variable %qD in declaration with more than one "
"declarator with placeholder type",
TREE_CODE (decl) == FUNCTION_DECL
? decl : auto_function_declaration);
auto_function_declaration = error_mark_node;
}
}
if (auto_result
&& (!processing_template_decl || !type_uses_auto (auto_result)))
{
if (last_type
&& last_type != error_mark_node
&& !same_type_p (auto_result, last_type))
{
error_at (decl_specifiers.locations[ds_type_spec],
"inconsistent deduction for %qT: %qT and then %qT",
decl_specifiers.type, last_type, auto_result);
last_type = error_mark_node;
}
else
last_type = auto_result;
}
if (function_definition_p)
{
if (cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
error_at (token->location,
"mixing"
" declarations and function-definitions is forbidden");
}
else
{
pop_deferring_access_checks ();
return;
}
}
if (maybe_range_for_decl && *maybe_range_for_decl == NULL_TREE)
*maybe_range_for_decl = decl;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_COMMA)
;
else if (token->type == CPP_SEMICOLON)
break;
else if (maybe_range_for_decl)
{
if ((declares_class_or_enum & 2) && token->type == CPP_COLON)
permerror (decl_specifiers.locations[ds_type_spec],
"types may not be defined in a for-range-declaration");
break;
}
else
{
if ((decl != error_mark_node
&& DECL_INITIAL (decl) != error_mark_node)
|| cp_parser_uncommitted_to_tentative_parse_p (parser))
cp_parser_error (parser, "expected %<,%> or %<;%>");
cp_parser_skip_to_end_of_statement (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
cp_lexer_consume_token (parser->lexer);
goto done;
}
function_definition_allowed_p = false;
}
if (!saw_declarator)
{
if (cp_parser_declares_only_class_p (parser))
{
if (!declares_class_or_enum
&& decl_specifiers.type
&& OVERLOAD_TYPE_P (decl_specifiers.type))
decl_specifiers.type = NULL_TREE;
shadow_tag (&decl_specifiers);
}
perform_deferred_access_checks (tf_warning_or_error);
}
finish:
if (!maybe_range_for_decl)
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
else if (cp_lexer_next_token_is (parser->lexer, CPP_COLON))
{
if (init_loc != UNKNOWN_LOCATION)
error_at (init_loc, "initializer in range-based %<for%> loop");
if (comma_loc != UNKNOWN_LOCATION)
error_at (comma_loc,
"multiple declarations in range-based %<for%> loop");
}
done:
pop_deferring_access_checks ();
}
static tree
cp_parser_decomposition_declaration (cp_parser *parser,
cp_decl_specifier_seq *decl_specifiers,
tree *maybe_range_for_decl,
location_t *init_loc)
{
cp_ref_qualifier ref_qual = cp_parser_ref_qualifier_opt (parser);
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
cp_parser_require (parser, CPP_OPEN_SQUARE, RT_OPEN_SQUARE);
auto_vec<cp_expr, 10> v;
if (!cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_SQUARE))
while (true)
{
cp_expr e = cp_parser_identifier (parser);
if (e.get_value () == error_mark_node)
break;
v.safe_push (e);
if (!cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
break;
cp_lexer_consume_token (parser->lexer);
}
location_t end_loc = cp_lexer_peek_token (parser->lexer)->location;
if (!cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE))
{
end_loc = UNKNOWN_LOCATION;
cp_parser_skip_to_closing_parenthesis_1 (parser, true, CPP_CLOSE_SQUARE,
false);
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_SQUARE))
cp_lexer_consume_token (parser->lexer);
else
{
cp_parser_skip_to_end_of_statement (parser);
return error_mark_node;
}
}
if (cxx_dialect < cxx17)
pedwarn (loc, 0, "structured bindings only available with "
"-std=c++17 or -std=gnu++17");
tree pushed_scope;
cp_declarator *declarator = make_declarator (cdk_decomp);
loc = end_loc == UNKNOWN_LOCATION ? loc : make_location (loc, loc, end_loc);
declarator->id_loc = loc;
if (ref_qual != REF_QUAL_NONE)
declarator = make_reference_declarator (TYPE_UNQUALIFIED, declarator,
ref_qual == REF_QUAL_RVALUE,
NULL_TREE);
tree decl = start_decl (declarator, decl_specifiers, SD_INITIALIZED,
NULL_TREE, decl_specifiers->attributes,
&pushed_scope);
tree orig_decl = decl;
unsigned int i;
cp_expr e;
cp_decl_specifier_seq decl_specs;
clear_decl_specs (&decl_specs);
decl_specs.type = make_auto ();
tree prev = decl;
FOR_EACH_VEC_ELT (v, i, e)
{
if (i == 0)
declarator = make_id_declarator (NULL_TREE, e.get_value (), sfk_none);
else
declarator->u.id.unqualified_name = e.get_value ();
declarator->id_loc = e.get_location ();
tree elt_pushed_scope;
tree decl2 = start_decl (declarator, &decl_specs, SD_INITIALIZED,
NULL_TREE, NULL_TREE, &elt_pushed_scope);
if (decl2 == error_mark_node)
decl = error_mark_node;
else if (decl != error_mark_node && DECL_CHAIN (decl2) != prev)
{
gcc_assert (errorcount);
decl = error_mark_node;
}
else
prev = decl2;
if (elt_pushed_scope)
pop_scope (elt_pushed_scope);
}
if (v.is_empty ())
{
error_at (loc, "empty structured binding declaration");
decl = error_mark_node;
}
if (maybe_range_for_decl == NULL
|| cp_lexer_next_token_is_not (parser->lexer, CPP_COLON))
{
bool non_constant_p = false, is_direct_init = false;
*init_loc = cp_lexer_peek_token (parser->lexer)->location;
tree initializer = cp_parser_initializer (parser, &is_direct_init,
&non_constant_p);
if (initializer == NULL_TREE
|| (TREE_CODE (initializer) == TREE_LIST
&& TREE_CHAIN (initializer))
|| (is_direct_init
&& BRACE_ENCLOSED_INITIALIZER_P (initializer)
&& CONSTRUCTOR_NELTS (initializer) != 1))
{
error_at (loc, "invalid initializer for structured binding "
"declaration");
initializer = error_mark_node;
}
if (decl != error_mark_node)
{
cp_maybe_mangle_decomp (decl, prev, v.length ());
cp_finish_decl (decl, initializer, non_constant_p, NULL_TREE,
is_direct_init ? LOOKUP_NORMAL : LOOKUP_IMPLICIT);
cp_finish_decomp (decl, prev, v.length ());
}
}
else if (decl != error_mark_node)
{
*maybe_range_for_decl = prev;
cp_finish_decomp (decl, prev, v.length ());
}
if (pushed_scope)
pop_scope (pushed_scope);
if (decl == error_mark_node && DECL_P (orig_decl))
{
if (DECL_NAMESPACE_SCOPE_P (orig_decl))
SET_DECL_ASSEMBLER_NAME (orig_decl, get_identifier ("<decomp>"));
}
return decl;
}
static void
cp_parser_decl_specifier_seq (cp_parser* parser,
cp_parser_flags flags,
cp_decl_specifier_seq *decl_specs,
int* declares_class_or_enum)
{
bool constructor_possible_p = !parser->in_declarator_p;
bool found_decl_spec = false;
cp_token *start_token = NULL;
cp_decl_spec ds;
clear_decl_specs (decl_specs);
*declares_class_or_enum = 0;
while (true)
{
bool constructor_p;
cp_token *token;
ds = ds_last;
token = cp_lexer_peek_token (parser->lexer);
if (!start_token)
start_token = token;
if (cp_next_tokens_can_be_attribute_p (parser))
{
tree attrs = cp_parser_attributes_opt (parser);
if (cxx11_attribute_p (attrs))
{
if (!found_decl_spec)
;
else
{
if (decl_specs->type && CLASS_TYPE_P (decl_specs->type))
{
if (decl_specs->type_definition_p)
warn_misplaced_attr_for_class_type (token->location,
decl_specs->type);
attrs = NULL_TREE;
}
else
{
decl_specs->std_attributes
= attr_chainon (decl_specs->std_attributes, attrs);
if (decl_specs->locations[ds_std_attribute] == 0)
decl_specs->locations[ds_std_attribute] = token->location;
}
continue;
}
}
decl_specs->attributes
= attr_chainon (decl_specs->attributes, attrs);
if (decl_specs->locations[ds_attribute] == 0)
decl_specs->locations[ds_attribute] = token->location;
continue;
}
found_decl_spec = true;
switch (token->keyword)
{
case RID_FRIEND:
if (!at_class_scope_p ())
{
gcc_rich_location richloc (token->location);
richloc.add_fixit_remove ();
error_at (&richloc, "%<friend%> used outside of class");
cp_lexer_purge_token (parser->lexer);
}
else
{
ds = ds_friend;
cp_lexer_consume_token (parser->lexer);
}
break;
case RID_CONSTEXPR:
ds = ds_constexpr;
cp_lexer_consume_token (parser->lexer);
break;
case RID_CONCEPT:
ds = ds_concept;
cp_lexer_consume_token (parser->lexer);
break;
case RID_INLINE:
case RID_VIRTUAL:
case RID_EXPLICIT:
cp_parser_function_specifier_opt (parser, decl_specs);
break;
case RID_TYPEDEF:
ds = ds_typedef;
cp_lexer_consume_token (parser->lexer);
constructor_possible_p = false;
cp_parser_commit_to_tentative_parse (parser);
if (decl_specs->storage_class != sc_none)
decl_specs->conflicting_specifiers_p = true;
break;
case RID_AUTO:
if (cxx_dialect == cxx98) 
{
cp_lexer_consume_token (parser->lexer);
gcc_rich_location richloc (token->location);
richloc.add_fixit_remove ();
warning_at (&richloc, OPT_Wc__11_compat,
"%<auto%> changes meaning in C++11; "
"please remove it");
cp_parser_set_storage_class (parser, decl_specs, RID_AUTO,
token);
}
else
found_decl_spec = false;
break;
case RID_REGISTER:
case RID_STATIC:
case RID_EXTERN:
case RID_MUTABLE:
cp_lexer_consume_token (parser->lexer);
cp_parser_set_storage_class (parser, decl_specs, token->keyword,
token);
break;
case RID_THREAD:
ds = ds_thread;
cp_lexer_consume_token (parser->lexer);
break;
default:
found_decl_spec = false;
break;
}
if (found_decl_spec
&& (flags & CP_PARSER_FLAGS_ONLY_TYPE_OR_CONSTEXPR)
&& token->keyword != RID_CONSTEXPR)
error ("decl-specifier invalid in condition");
if (found_decl_spec
&& (flags & CP_PARSER_FLAGS_ONLY_MUTABLE_OR_CONSTEXPR)
&& token->keyword != RID_MUTABLE
&& token->keyword != RID_CONSTEXPR)
error_at (token->location, "%qD invalid in lambda",
ridpointers[token->keyword]);
if (ds != ds_last)
set_and_check_decl_spec_loc (decl_specs, ds, token);
constructor_p
= (!found_decl_spec
&& constructor_possible_p
&& (cp_parser_constructor_declarator_p
(parser, decl_spec_seq_has_spec_p (decl_specs, ds_friend))));
if (!found_decl_spec && !constructor_p)
{
int decl_spec_declares_class_or_enum;
bool is_cv_qualifier;
tree type_spec;
type_spec
= cp_parser_type_specifier (parser, flags,
decl_specs,
true,
&decl_spec_declares_class_or_enum,
&is_cv_qualifier);
*declares_class_or_enum |= decl_spec_declares_class_or_enum;
if (type_spec && !is_cv_qualifier)
flags |= CP_PARSER_FLAGS_NO_USER_DEFINED_TYPES;
if (type_spec)
{
constructor_possible_p = false;
found_decl_spec = true;
if (!is_cv_qualifier)
decl_specs->any_type_specifiers_p = true;
}
}
if (!found_decl_spec)
break;
decl_specs->any_specifiers_p = true;
flags |= CP_PARSER_FLAGS_OPTIONAL;
}
if (decl_spec_seq_has_spec_p (decl_specs, ds_friend)
&& (*declares_class_or_enum & 2))
error_at (decl_specs->locations[ds_friend],
"class definition may not be declared a friend");
}
static tree
cp_parser_storage_class_specifier_opt (cp_parser* parser)
{
switch (cp_lexer_peek_token (parser->lexer)->keyword)
{
case RID_AUTO:
if (cxx_dialect != cxx98)
return NULL_TREE;
gcc_fallthrough ();
case RID_REGISTER:
case RID_STATIC:
case RID_EXTERN:
case RID_MUTABLE:
case RID_THREAD:
return cp_lexer_consume_token (parser->lexer)->u.value;
default:
return NULL_TREE;
}
}
static tree
cp_parser_function_specifier_opt (cp_parser* parser,
cp_decl_specifier_seq *decl_specs)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
switch (token->keyword)
{
case RID_INLINE:
set_and_check_decl_spec_loc (decl_specs, ds_inline, token);
break;
case RID_VIRTUAL:
if (PROCESSING_REAL_TEMPLATE_DECL_P ()
&& current_class_type)
error_at (token->location, "templates may not be %<virtual%>");
else
set_and_check_decl_spec_loc (decl_specs, ds_virtual, token);
break;
case RID_EXPLICIT:
set_and_check_decl_spec_loc (decl_specs, ds_explicit, token);
break;
default:
return NULL_TREE;
}
return cp_lexer_consume_token (parser->lexer)->u.value;
}
static void
cp_parser_linkage_specification (cp_parser* parser)
{
tree linkage;
cp_token *extern_token
= cp_parser_require_keyword (parser, RID_EXTERN, RT_EXTERN);
cp_token *string_token = cp_lexer_peek_token (parser->lexer);
linkage = cp_parser_string_literal (parser, false, false);
if (strlen (TREE_STRING_POINTER (linkage))
!= (size_t) (TREE_STRING_LENGTH (linkage) - 1))
{
cp_parser_error (parser, "invalid linkage-specification");
linkage = lang_name_cplusplus;
}
else
linkage = get_identifier (TREE_STRING_POINTER (linkage));
push_lang_context (linkage);
location_t saved_location
= parser->innermost_linkage_specification_location;
parser->innermost_linkage_specification_location
= make_location (extern_token->location,
extern_token->location,
get_finish (string_token->location));
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
cp_ensure_no_omp_declare_simd (parser);
cp_ensure_no_oacc_routine (parser);
matching_braces braces;
braces.consume_open (parser)->location;
cp_parser_declaration_seq_opt (parser);
braces.require_close (parser);
}
else
{
bool saved_in_unbraced_linkage_specification_p;
saved_in_unbraced_linkage_specification_p
= parser->in_unbraced_linkage_specification_p;
parser->in_unbraced_linkage_specification_p = true;
cp_parser_declaration (parser);
parser->in_unbraced_linkage_specification_p
= saved_in_unbraced_linkage_specification_p;
}
pop_lang_context ();
parser->innermost_linkage_specification_location = saved_location;
}
static void 
cp_parser_static_assert(cp_parser *parser, bool member_p)
{
cp_expr condition;
location_t token_loc;
tree message;
bool dummy;
token_loc = cp_lexer_peek_token (parser->lexer)->location;
if (!cp_parser_require_keyword (parser, RID_STATIC_ASSERT, 
RT_STATIC_ASSERT))
return;
if (cp_parser_parsing_tentatively (parser))
cp_parser_commit_to_tentative_parse (parser);
matching_parens parens;
parens.require_open (parser);
condition = 
cp_parser_constant_expression (parser,
true,
&dummy);
if (cp_lexer_peek_token (parser->lexer)->type == CPP_CLOSE_PAREN)
{
if (cxx_dialect < cxx17)
pedwarn (input_location, OPT_Wpedantic,
"static_assert without a message "
"only available with -std=c++17 or -std=gnu++17");
cp_lexer_consume_token (parser->lexer);
message = build_string (1, "");
TREE_TYPE (message) = char_array_type_node;
fix_string_type (message);
}
else
{
cp_parser_require (parser, CPP_COMMA, RT_COMMA);
message = cp_parser_string_literal (parser, 
false,
true);
if (!parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, 
true, 
false,
true);
}
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
location_t assert_loc = condition.get_location ();
if (assert_loc == UNKNOWN_LOCATION)
assert_loc = token_loc;
finish_static_assert (condition, message, assert_loc, member_p);
}
static tree
cp_parser_decltype_expr (cp_parser *parser,
bool &id_expression_or_member_access_p)
{
cp_token *id_expr_start_token;
tree expr;
tentative_firewall firewall (parser);
id_expr_start_token = cp_lexer_peek_token (parser->lexer);
cp_parser_parse_tentatively (parser);
expr = cp_parser_id_expression (parser,
false,
true,
NULL,
false,
false);
if (!cp_parser_error_occurred (parser) && expr != error_mark_node)
{
bool non_integral_constant_expression_p = false;
tree id_expression = expr;
cp_id_kind idk;
const char *error_msg;
if (identifier_p (expr))
expr = cp_parser_lookup_name_simple (parser, expr,
id_expr_start_token->location);
if (expr && TREE_CODE (expr) == TEMPLATE_DECL)
expr = error_mark_node;
if (expr
&& expr != error_mark_node
&& TREE_CODE (expr) != TYPE_DECL
&& (TREE_CODE (expr) != BIT_NOT_EXPR
|| !TYPE_P (TREE_OPERAND (expr, 0)))
&& cp_lexer_peek_token (parser->lexer)->type == CPP_CLOSE_PAREN)
{
expr = (finish_id_expression
(id_expression, expr, parser->scope, &idk,
false,
true,
&non_integral_constant_expression_p,
false,
true,
false,
false,
&error_msg,
id_expr_start_token->location));
if (expr == error_mark_node)
id_expression_or_member_access_p = true;
}
if (expr 
&& expr != error_mark_node
&& cp_lexer_peek_token (parser->lexer)->type == CPP_CLOSE_PAREN)
id_expression_or_member_access_p = true;
}
if (!id_expression_or_member_access_p)
{
cp_parser_abort_tentative_parse (parser);
cp_parser_parse_tentatively (parser);
expr = cp_parser_postfix_expression (parser, false,
false, true,
true, NULL);
if (expr 
&& expr != error_mark_node
&& cp_lexer_peek_token (parser->lexer)->type == CPP_CLOSE_PAREN)
id_expression_or_member_access_p = true;
}
if (id_expression_or_member_access_p)
cp_parser_parse_definitely (parser);
else
{
cp_parser_abort_tentative_parse (parser);
cp_parser_commit_to_tentative_parse (parser);
expr = cp_parser_expression (parser, NULL, false,
true);
}
return expr;
}
static tree
cp_parser_decltype (cp_parser *parser)
{
bool id_expression_or_member_access_p = false;
cp_token *start_token = cp_lexer_peek_token (parser->lexer);
if (start_token->type == CPP_DECLTYPE)
{
cp_lexer_consume_token (parser->lexer);
return saved_checks_value (start_token->u.tree_check_value);
}
if (!cp_parser_require_keyword (parser, RID_DECLTYPE, RT_DECLTYPE))
return error_mark_node;
matching_parens parens;
if (!parens.require_open (parser))
return error_mark_node;
push_deferring_access_checks (dk_deferred);
tree expr = NULL_TREE;
if (cxx_dialect >= cxx14
&& cp_lexer_next_token_is_keyword (parser->lexer, RID_AUTO))
cp_lexer_consume_token (parser->lexer);
else
{
const char *saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in %<decltype%> expressions");
bool saved_integral_constant_expression_p
= parser->integral_constant_expression_p;
bool saved_non_integral_constant_expression_p
= parser->non_integral_constant_expression_p;
parser->integral_constant_expression_p = false;
bool saved_greater_than_is_operator_p
= parser->greater_than_is_operator_p;
parser->greater_than_is_operator_p = true;
++cp_unevaluated_operand;
++c_inhibit_evaluation_warnings;
expr = cp_parser_decltype_expr (parser, id_expression_or_member_access_p);
--cp_unevaluated_operand;
--c_inhibit_evaluation_warnings;
parser->greater_than_is_operator_p
= saved_greater_than_is_operator_p;
parser->type_definition_forbidden_message = saved_message;
parser->integral_constant_expression_p
= saved_integral_constant_expression_p;
parser->non_integral_constant_expression_p
= saved_non_integral_constant_expression_p;
}
if (!parens.require_close (parser))
{
cp_parser_skip_to_closing_parenthesis (parser, true, false,
true);
pop_deferring_access_checks ();
return error_mark_node;
}
if (!expr)
{
expr = make_decltype_auto ();
AUTO_IS_DECLTYPE (expr) = true;
}
else
expr = finish_decltype_type (expr, id_expression_or_member_access_p,
tf_warning_or_error);
start_token->type = CPP_DECLTYPE;
start_token->u.tree_check_value = ggc_cleared_alloc<struct tree_check> ();
start_token->u.tree_check_value->value = expr;
start_token->u.tree_check_value->checks = get_deferred_access_checks ();
start_token->keyword = RID_MAX;
cp_lexer_purge_tokens_after (parser->lexer, start_token);
pop_to_parent_deferring_access_checks ();
return expr;
}
static tree
cp_parser_conversion_function_id (cp_parser* parser)
{
tree type;
tree saved_scope;
tree saved_qualifying_scope;
tree saved_object_scope;
tree pushed_scope = NULL_TREE;
if (!cp_parser_require_keyword (parser, RID_OPERATOR, RT_OPERATOR))
return error_mark_node;
saved_scope = parser->scope;
saved_qualifying_scope = parser->qualifying_scope;
saved_object_scope = parser->object_scope;
if (saved_scope)
pushed_scope = push_scope (saved_scope);
type = cp_parser_conversion_type_id (parser);
if (pushed_scope)
pop_scope (pushed_scope);
parser->scope = saved_scope;
parser->qualifying_scope = saved_qualifying_scope;
parser->object_scope = saved_object_scope;
if (type == error_mark_node)
return error_mark_node;
return make_conv_op_name (type);
}
static tree
cp_parser_conversion_type_id (cp_parser* parser)
{
tree attributes;
cp_decl_specifier_seq type_specifiers;
cp_declarator *declarator;
tree type_specified;
const char *saved_message;
attributes = cp_parser_attributes_opt (parser);
saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in a conversion-type-id");
cp_parser_type_specifier_seq (parser, false,
false,
&type_specifiers);
parser->type_definition_forbidden_message = saved_message;
if (type_specifiers.type == error_mark_node)
return error_mark_node;
declarator = cp_parser_conversion_declarator_opt (parser);
type_specified =  grokdeclarator (declarator, &type_specifiers, TYPENAME,
0, &attributes);
if (attributes)
cplus_decl_attributes (&type_specified, attributes, 0);
if (! cp_parser_uncommitted_to_tentative_parse_p (parser)
&& type_uses_auto (type_specified))
{
if (cxx_dialect < cxx14)
{
error ("invalid use of %<auto%> in conversion operator");
return error_mark_node;
}
else if (template_parm_scope_p ())
warning (0, "use of %<auto%> in member template "
"conversion operator can never be deduced");
}
return type_specified;
}
static cp_declarator *
cp_parser_conversion_declarator_opt (cp_parser* parser)
{
enum tree_code code;
tree class_type, std_attributes = NULL_TREE;
cp_cv_quals cv_quals;
cp_parser_parse_tentatively (parser);
code = cp_parser_ptr_operator (parser, &class_type, &cv_quals,
&std_attributes);
if (cp_parser_parse_definitely (parser))
{
cp_declarator *declarator;
declarator = cp_parser_conversion_declarator_opt (parser);
declarator = cp_parser_make_indirect_declarator
(code, class_type, cv_quals, declarator, std_attributes);
return declarator;
}
return NULL;
}
static void
cp_parser_ctor_initializer_opt (cp_parser* parser)
{
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COLON))
{
if (DECL_CONSTRUCTOR_P (current_function_decl))
finish_mem_initializers (NULL_TREE);
return;
}
cp_lexer_consume_token (parser->lexer);
cp_parser_mem_initializer_list (parser);
}
static void
cp_parser_mem_initializer_list (cp_parser* parser)
{
tree mem_initializer_list = NULL_TREE;
tree target_ctor = error_mark_node;
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (!DECL_CONSTRUCTOR_P (current_function_decl))
error_at (token->location,
"only constructors take member initializers");
while (true)
{
tree mem_initializer;
token = cp_lexer_peek_token (parser->lexer);
mem_initializer = cp_parser_mem_initializer (parser);
bool ellipsis = cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS);
if (ellipsis
|| (mem_initializer != error_mark_node
&& check_for_bare_parameter_packs (TREE_PURPOSE
(mem_initializer))))
{
if (ellipsis)
cp_lexer_consume_token (parser->lexer);
if (mem_initializer != error_mark_node
&& !TYPE_P (TREE_PURPOSE (mem_initializer)))
{
error_at (token->location,
"cannot expand initializer for member %qD",
TREE_PURPOSE (mem_initializer));
mem_initializer = error_mark_node;
}
if (mem_initializer != error_mark_node)
mem_initializer = make_pack_expansion (mem_initializer);
}
if (target_ctor != error_mark_node
&& mem_initializer != error_mark_node)
{
error ("mem-initializer for %qD follows constructor delegation",
TREE_PURPOSE (mem_initializer));
mem_initializer = error_mark_node;
}
if (mem_initializer != error_mark_node
&& CLASS_TYPE_P (TREE_PURPOSE (mem_initializer))
&& same_type_p (TREE_PURPOSE (mem_initializer), current_class_type))
{
maybe_warn_cpp0x (CPP0X_DELEGATING_CTORS);
if (mem_initializer_list)
{
error ("constructor delegation follows mem-initializer for %qD",
TREE_PURPOSE (mem_initializer_list));
mem_initializer = error_mark_node;
}
target_ctor = mem_initializer;
}
if (mem_initializer != error_mark_node)
{
TREE_CHAIN (mem_initializer) = mem_initializer_list;
mem_initializer_list = mem_initializer;
}
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA))
break;
cp_lexer_consume_token (parser->lexer);
}
if (DECL_CONSTRUCTOR_P (current_function_decl))
finish_mem_initializers (mem_initializer_list);
}
static tree
cp_parser_mem_initializer (cp_parser* parser)
{
tree mem_initializer_id;
tree expression_list;
tree member;
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
permerror (token->location,
"anachronistic old-style base class initializer");
mem_initializer_id = NULL_TREE;
}
else
{
mem_initializer_id = cp_parser_mem_initializer_id (parser);
if (mem_initializer_id == error_mark_node)
return mem_initializer_id;
}
member = expand_member_init (mem_initializer_id);
if (member && !DECL_P (member))
in_base_initializer = 1;
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
bool expr_non_constant_p;
cp_lexer_set_source_position (parser->lexer);
maybe_warn_cpp0x (CPP0X_INITIALIZER_LISTS);
expression_list = cp_parser_braced_list (parser, &expr_non_constant_p);
CONSTRUCTOR_IS_DIRECT_INIT (expression_list) = 1;
expression_list = build_tree_list (NULL_TREE, expression_list);
}
else
{
vec<tree, va_gc> *vec;
vec = cp_parser_parenthesized_expression_list (parser, non_attr,
false,
true,
NULL);
if (vec == NULL)
return error_mark_node;
expression_list = build_tree_list_vec (vec);
release_tree_vector (vec);
}
if (expression_list == error_mark_node)
return error_mark_node;
if (!expression_list)
expression_list = void_type_node;
in_base_initializer = 0;
return member ? build_tree_list (member, expression_list) : error_mark_node;
}
static tree
cp_parser_mem_initializer_id (cp_parser* parser)
{
bool global_scope_p;
bool nested_name_specifier_p;
bool template_p = false;
tree id;
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TYPENAME))
{
error_at (token->location, 
"keyword %<typename%> not allowed in this context (a qualified "
"member initializer is implicitly a type)");
cp_lexer_consume_token (parser->lexer);
}
global_scope_p
= (cp_parser_global_scope_opt (parser,
false)
!= NULL_TREE);
nested_name_specifier_p
= (cp_parser_nested_name_specifier_opt (parser,
true,
true,
true,
true)
!= NULL_TREE);
if (nested_name_specifier_p)
template_p = cp_parser_optional_template_keyword (parser);
if (global_scope_p || nested_name_specifier_p)
return cp_parser_class_name (parser,
true,
template_p,
typename_type,
true,
false,
true);
cp_parser_parse_tentatively (parser);
if (cp_lexer_next_token_is_decltype (parser->lexer))
id = cp_parser_decltype (parser);
else
id = cp_parser_class_name (parser,
true,
false,
none_type,
true,
false,
true);
if (cp_parser_parse_definitely (parser))
return id;
return cp_parser_identifier (parser);
}
static cp_expr
cp_parser_operator_function_id (cp_parser* parser)
{
if (!cp_parser_require_keyword (parser, RID_OPERATOR, RT_OPERATOR))
return error_mark_node;
return cp_parser_operator (parser);
}
tree
cp_literal_operator_id (const char* name)
{
tree identifier;
char *buffer = XNEWVEC (char, strlen (UDLIT_OP_ANSI_PREFIX)
+ strlen (name) + 10);
sprintf (buffer, UDLIT_OP_ANSI_FORMAT, name);
identifier = get_identifier (buffer);
return identifier;
}
static cp_expr
cp_parser_operator (cp_parser* parser)
{
tree id = NULL_TREE;
cp_token *token;
bool utf8 = false;
token = cp_lexer_peek_token (parser->lexer);
location_t start_loc = token->location;
enum tree_code op = ERROR_MARK;
bool assop = false;
bool consumed = false;
switch (token->type)
{
case CPP_KEYWORD:
{
if (token->keyword == RID_NEW)
op = NEW_EXPR;
else if (token->keyword == RID_DELETE)
op = DELETE_EXPR;
else
break;
location_t end_loc = cp_lexer_consume_token (parser->lexer)->location;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_OPEN_SQUARE)
{
cp_lexer_consume_token (parser->lexer);
if (cp_token *close_token
= cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE))
end_loc = close_token->location;
op = op == NEW_EXPR ? VEC_NEW_EXPR : VEC_DELETE_EXPR;
}
start_loc = make_location (start_loc, start_loc, end_loc);
consumed = true;
break;
}
case CPP_PLUS:
op = PLUS_EXPR;
break;
case CPP_MINUS:
op = MINUS_EXPR;
break;
case CPP_MULT:
op = MULT_EXPR;
break;
case CPP_DIV:
op = TRUNC_DIV_EXPR;
break;
case CPP_MOD:
op = TRUNC_MOD_EXPR;
break;
case CPP_XOR:
op = BIT_XOR_EXPR;
break;
case CPP_AND:
op = BIT_AND_EXPR;
break;
case CPP_OR:
op = BIT_IOR_EXPR;
break;
case CPP_COMPL:
op = BIT_NOT_EXPR;
break;
case CPP_NOT:
op = TRUTH_NOT_EXPR;
break;
case CPP_EQ:
assop = true;
op = NOP_EXPR;
break;
case CPP_LESS:
op = LT_EXPR;
break;
case CPP_GREATER:
op = GT_EXPR;
break;
case CPP_PLUS_EQ:
assop = true;
op = PLUS_EXPR;
break;
case CPP_MINUS_EQ:
assop = true;
op = MINUS_EXPR;
break;
case CPP_MULT_EQ:
assop = true;
op = MULT_EXPR;
break;
case CPP_DIV_EQ:
assop = true;
op = TRUNC_DIV_EXPR;
break;
case CPP_MOD_EQ:
assop = true;
op = TRUNC_MOD_EXPR;
break;
case CPP_XOR_EQ:
assop = true;
op = BIT_XOR_EXPR;
break;
case CPP_AND_EQ:
assop = true;
op = BIT_AND_EXPR;
break;
case CPP_OR_EQ:
assop = true;
op = BIT_IOR_EXPR;
break;
case CPP_LSHIFT:
op = LSHIFT_EXPR;
break;
case CPP_RSHIFT:
op = RSHIFT_EXPR;
break;
case CPP_LSHIFT_EQ:
assop = true;
op = LSHIFT_EXPR;
break;
case CPP_RSHIFT_EQ:
assop = true;
op = RSHIFT_EXPR;
break;
case CPP_EQ_EQ:
op = EQ_EXPR;
break;
case CPP_NOT_EQ:
op = NE_EXPR;
break;
case CPP_LESS_EQ:
op = LE_EXPR;
break;
case CPP_GREATER_EQ:
op = GE_EXPR;
break;
case CPP_AND_AND:
op = TRUTH_ANDIF_EXPR;
break;
case CPP_OR_OR:
op = TRUTH_ORIF_EXPR;
break;
case CPP_PLUS_PLUS:
op = POSTINCREMENT_EXPR;
break;
case CPP_MINUS_MINUS:
op = PREDECREMENT_EXPR;
break;
case CPP_COMMA:
op = COMPOUND_EXPR;
break;
case CPP_DEREF_STAR:
op = MEMBER_REF;
break;
case CPP_DEREF:
op = COMPONENT_REF;
break;
case CPP_OPEN_PAREN:
{
matching_parens parens;
parens.consume_open (parser);
parens.require_close (parser);
op = CALL_EXPR;
consumed = true;
break;
}
case CPP_OPEN_SQUARE:
cp_lexer_consume_token (parser->lexer);
cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE);
op = ARRAY_REF;
consumed = true;
break;
case CPP_UTF8STRING:
case CPP_UTF8STRING_USERDEF:
utf8 = true;
case CPP_STRING:
case CPP_WSTRING:
case CPP_STRING16:
case CPP_STRING32:
case CPP_STRING_USERDEF:
case CPP_WSTRING_USERDEF:
case CPP_STRING16_USERDEF:
case CPP_STRING32_USERDEF:
{
tree str, string_tree;
int sz, len;
if (cxx_dialect == cxx98)
maybe_warn_cpp0x (CPP0X_USER_DEFINED_LITERALS);
str = cp_parser_string_literal (parser, true,
true, false);
if (str == error_mark_node)
return error_mark_node;
else if (TREE_CODE (str) == USERDEF_LITERAL)
{
string_tree = USERDEF_LITERAL_VALUE (str);
id = USERDEF_LITERAL_SUFFIX_ID (str);
}
else
{
string_tree = str;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_NAME)
id = cp_parser_identifier (parser);
else if (token->type == CPP_KEYWORD)
{
error ("unexpected keyword;"
" remove space between quotes and suffix identifier");
return error_mark_node;
}
else
{
error ("expected suffix identifier");
return error_mark_node;
}
}
sz = TREE_INT_CST_LOW (TYPE_SIZE_UNIT
(TREE_TYPE (TREE_TYPE (string_tree))));
len = TREE_STRING_LENGTH (string_tree) / sz - 1;
if (len != 0)
{
error ("expected empty string after %<operator%> keyword");
return error_mark_node;
}
if (utf8 || TYPE_MAIN_VARIANT (TREE_TYPE (TREE_TYPE (string_tree)))
!= char_type_node)
{
error ("invalid encoding prefix in literal operator");
return error_mark_node;
}
if (id != error_mark_node)
{
const char *name = IDENTIFIER_POINTER (id);
id = cp_literal_operator_id (name);
}
return id;
}
default:
break;
}
if (op != ERROR_MARK)
{
id = ovl_op_identifier (assop, op);
if (!consumed)
cp_lexer_consume_token (parser->lexer);
}
else
{
cp_parser_error (parser, "expected operator");
id = error_mark_node;
}
return cp_expr (id, start_loc);
}
static void
cp_parser_template_declaration (cp_parser* parser, bool member_p)
{
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_EXPORT))
{
cp_lexer_consume_token (parser->lexer);
warning (0, "keyword %<export%> not implemented, and will be ignored");
}
cp_parser_template_declaration_after_export (parser, member_p);
}
static tree
cp_parser_template_parameter_list (cp_parser* parser)
{
tree parameter_list = NULL_TREE;
begin_template_parm_list ();
while (true)
{
tree parameter;
bool is_non_type;
bool is_parameter_pack;
location_t parm_loc;
parm_loc = cp_lexer_peek_token (parser->lexer)->location;
parameter = cp_parser_template_parameter (parser, 
&is_non_type,
&is_parameter_pack);
if (parameter != error_mark_node)
parameter_list = process_template_parm (parameter_list,
parm_loc,
parameter,
is_non_type,
is_parameter_pack);
else
{
tree err_parm = build_tree_list (parameter, parameter);
parameter_list = chainon (parameter_list, err_parm);
}
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA))
break;
cp_lexer_consume_token (parser->lexer);
}
return end_template_parm_list (parameter_list);
}
static tree
cp_parser_introduction_list (cp_parser *parser)
{
vec<tree, va_gc> *introduction_vec = make_tree_vector ();
while (true)
{
bool is_pack = cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS);
if (is_pack)
cp_lexer_consume_token (parser->lexer);
tree parm = build_nt (WILDCARD_DECL);
DECL_SOURCE_LOCATION (parm)
= cp_lexer_peek_token (parser->lexer)->location;
DECL_NAME (parm) = cp_parser_identifier (parser);
WILDCARD_PACK_P (parm) = is_pack;
vec_safe_push (introduction_vec, parm);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA))
break;
cp_lexer_consume_token (parser->lexer);
}
tree introduction_list = make_tree_vec (introduction_vec->length ());
unsigned int n;
tree parm;
FOR_EACH_VEC_ELT (*introduction_vec, n, parm)
TREE_VEC_ELT (introduction_list, n) = parm;
release_tree_vector (introduction_vec);
return introduction_list;
}
static inline cp_declarator*
get_id_declarator (cp_declarator *declarator)
{
cp_declarator *d = declarator;
while (d && d->kind != cdk_id)
d = d->declarator;
return d;
}
static inline tree
get_unqualified_id (cp_declarator *declarator)
{
declarator = get_id_declarator (declarator);
if (declarator)
return declarator->u.id.unqualified_name;
else
return NULL_TREE;
}
static inline bool
is_constrained_parameter (tree decl)
{
return (decl
&& TREE_CODE (decl) == TYPE_DECL
&& CONSTRAINED_PARM_CONCEPT (decl)
&& DECL_P (CONSTRAINED_PARM_CONCEPT (decl)));
}
static inline bool
is_constrained_parameter (cp_parameter_declarator *parm)
{
return is_constrained_parameter (parm->decl_specifiers.type);
}
bool
cp_parser_check_constrained_type_parm (cp_parser *parser,
cp_parameter_declarator *parm)
{
if (!parm->declarator)
return true;
if (parm->declarator->kind != cdk_id)
{
cp_parser_error (parser, "invalid constrained type parameter");
return false;
}
if (decl_spec_seq_has_spec_p (&parm->decl_specifiers, ds_const)
|| decl_spec_seq_has_spec_p (&parm->decl_specifiers, ds_volatile))
{
cp_parser_error (parser, "cv-qualified type parameter");
return false;
}
return true;
}
static inline tree
cp_parser_constrained_type_template_parm (cp_parser *parser,
tree id,
cp_parameter_declarator* parmdecl)
{
if (cp_parser_check_constrained_type_parm (parser, parmdecl))
return finish_template_type_parm (class_type_node, id);
else
return error_mark_node;
}
static tree
finish_constrained_template_template_parm (tree proto, tree id)
{
tree saved_parms = current_template_parms;
begin_template_parm_list ();
current_template_parms = DECL_TEMPLATE_PARMS (proto);
end_template_parm_list ();
tree parm = finish_template_template_parm (class_type_node, id);
current_template_parms = saved_parms;
return parm;
}
static tree
cp_parser_constrained_template_template_parm (cp_parser *parser,
tree proto,
tree id,
cp_parameter_declarator *parmdecl)
{
if (!cp_parser_check_constrained_type_parm (parser, parmdecl))
return error_mark_node;
return finish_constrained_template_template_parm (proto, id);
}
static tree
constrained_non_type_template_parm (bool *is_non_type,
cp_parameter_declarator *parm)
{
*is_non_type = true;
cp_declarator *decl = parm->declarator;
cp_decl_specifier_seq *specs = &parm->decl_specifiers;
specs->type = TREE_TYPE (DECL_INITIAL (specs->type));
return grokdeclarator (decl, specs, TPARM, 0, NULL);
}
static tree
finish_constrained_parameter (cp_parser *parser,
cp_parameter_declarator *parmdecl,
bool *is_non_type,
bool *is_parameter_pack)
{
tree decl = parmdecl->decl_specifiers.type;
tree id = get_unqualified_id (parmdecl->declarator);
tree def = parmdecl->default_argument;
tree proto = DECL_INITIAL (decl);
bool is_variadic = template_parameter_pack_p (proto);
if (is_variadic && !*is_parameter_pack)
cp_parser_error (parser, "variadic constraint introduced without %<...%>");
tree parm;
if (TREE_CODE (proto) == TYPE_DECL)
parm = cp_parser_constrained_type_template_parm (parser, id, parmdecl);
else if (TREE_CODE (proto) == TEMPLATE_DECL)
parm = cp_parser_constrained_template_template_parm (parser, proto, id,
parmdecl);
else
parm = constrained_non_type_template_parm (is_non_type, parmdecl);
if (parm == error_mark_node)
return error_mark_node;
parm = build_tree_list (def, parm);
TEMPLATE_PARM_CONSTRAINTS (parm) = decl;
return parm;
}
static inline bool
declares_constrained_type_template_parameter (tree type)
{
return (is_constrained_parameter (type)
&& TREE_CODE (TREE_TYPE (type)) == TEMPLATE_TYPE_PARM);
}
static bool
declares_constrained_template_template_parameter (tree type)
{
return (is_constrained_parameter (type)
&& TREE_CODE (TREE_TYPE (type)) == TEMPLATE_TEMPLATE_PARM);
}
static tree
cp_parser_default_type_template_argument (cp_parser *parser)
{
gcc_assert (cp_lexer_next_token_is (parser->lexer, CPP_EQ));
cp_lexer_consume_token (parser->lexer);
cp_token *token = cp_lexer_peek_token (parser->lexer);
push_deferring_access_checks (dk_no_deferred);
tree default_argument = cp_parser_type_id (parser);
pop_deferring_access_checks ();
if (flag_concepts && type_uses_auto (default_argument))
{
error_at (token->location,
"invalid use of %<auto%> in default template argument");
return error_mark_node;
}
return default_argument;
}
static tree
cp_parser_default_template_template_argument (cp_parser *parser)
{
gcc_assert (cp_lexer_next_token_is (parser->lexer, CPP_EQ));
bool is_template;
cp_lexer_consume_token (parser->lexer);
push_deferring_access_checks (dk_no_deferred);
const cp_token* token = cp_lexer_peek_token (parser->lexer);
tree default_argument
= cp_parser_id_expression (parser,
false,
true,
&is_template,
false,
false);
if (TREE_CODE (default_argument) == TYPE_DECL)
;
else
default_argument
= cp_parser_lookup_name (parser, default_argument,
none_type,
is_template,
false,
true,
NULL,
token->location);
default_argument = check_template_template_default_arg (default_argument);
pop_deferring_access_checks ();
return default_argument;
}
static tree
cp_parser_template_parameter (cp_parser* parser, bool *is_non_type,
bool *is_parameter_pack)
{
cp_token *token;
cp_parameter_declarator *parameter_declarator;
tree parm;
*is_non_type = false;
*is_parameter_pack = false;
token = cp_lexer_peek_token (parser->lexer);
if (token->keyword == RID_TEMPLATE)
return cp_parser_type_parameter (parser, is_parameter_pack);
if (token->keyword == RID_TYPENAME || token->keyword == RID_CLASS)
{
token = cp_lexer_peek_nth_token (parser->lexer, 2);
if (token->type == CPP_ELLIPSIS)
return cp_parser_type_parameter (parser, is_parameter_pack);
if (token->type == CPP_NAME)
token = cp_lexer_peek_nth_token (parser->lexer, 3);
if (token->type == CPP_COMMA
|| token->type == CPP_EQ
|| token->type == CPP_GREATER)
return cp_parser_type_parameter (parser, is_parameter_pack);
}
parameter_declarator
= cp_parser_parameter_declaration (parser, true,
NULL);
if (!parameter_declarator)
return error_mark_node;
if (parameter_declarator->template_parameter_pack_p)
*is_parameter_pack = true;
if (parameter_declarator->default_argument)
{
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
cp_lexer_consume_token (parser->lexer);
}
if (is_constrained_parameter (parameter_declarator))
return finish_constrained_parameter (parser,
parameter_declarator,
is_non_type,
is_parameter_pack);
*is_non_type = true;
parm = grokdeclarator (parameter_declarator->declarator,
&parameter_declarator->decl_specifiers,
TPARM, 0,
NULL);
if (parm == error_mark_node)
return error_mark_node;
return build_tree_list (parameter_declarator->default_argument, parm);
}
static tree
cp_parser_type_parameter (cp_parser* parser, bool *is_parameter_pack)
{
cp_token *token;
tree parameter;
token = cp_parser_require (parser, CPP_KEYWORD, RT_CLASS_TYPENAME_TEMPLATE);
if (!token)
return error_mark_node;
switch (token->keyword)
{
case RID_CLASS:
case RID_TYPENAME:
{
tree identifier;
tree default_argument;
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
cp_lexer_consume_token (parser->lexer);
maybe_warn_variadic_templates ();
*is_parameter_pack = true;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
identifier = cp_parser_identifier (parser);
else
identifier = NULL_TREE;
parameter = finish_template_type_parm (class_type_node, identifier);
if (cp_lexer_next_token_is (parser->lexer, CPP_EQ))
{
default_argument
= cp_parser_default_type_template_argument (parser);
if (*is_parameter_pack)
{
if (identifier)
error_at (token->location,
"template parameter pack %qD cannot have a "
"default argument", identifier);
else
error_at (token->location,
"template parameter packs cannot have "
"default arguments");
default_argument = NULL_TREE;
}
else if (check_for_bare_parameter_packs (default_argument))
default_argument = error_mark_node;
}
else
default_argument = NULL_TREE;
parameter = build_tree_list (default_argument, parameter);
}
break;
case RID_TEMPLATE:
{
tree identifier;
tree default_argument;
cp_parser_require (parser, CPP_LESS, RT_LESS);
cp_parser_template_parameter_list (parser);
cp_parser_require (parser, CPP_GREATER, RT_GREATER);
if (flag_concepts)
{
tree reqs = get_shorthand_constraints (current_template_parms);
if (tree r = cp_parser_requires_clause_opt (parser))
reqs = conjoin_constraints (reqs, normalize_expression (r));
TEMPLATE_PARMS_CONSTRAINTS (current_template_parms) = reqs;
}
cp_parser_type_parameter_key (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
cp_lexer_consume_token (parser->lexer);
maybe_warn_variadic_templates ();
*is_parameter_pack = true;
}
if (cp_lexer_next_token_is_not (parser->lexer, CPP_EQ)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_GREATER)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA))
{
identifier = cp_parser_identifier (parser);
if (identifier == error_mark_node)
identifier = NULL_TREE;
}
else
identifier = NULL_TREE;
parameter = finish_template_template_parm (class_type_node,
identifier);
if (cp_lexer_next_token_is (parser->lexer, CPP_EQ))
{
default_argument
= cp_parser_default_template_template_argument (parser);
if (*is_parameter_pack)
{
if (identifier)
error_at (token->location,
"template parameter pack %qD cannot "
"have a default argument",
identifier);
else
error_at (token->location, "template parameter packs cannot "
"have default arguments");
default_argument = NULL_TREE;
}
}
else
default_argument = NULL_TREE;
parameter = build_tree_list (default_argument, parameter);
}
break;
default:
gcc_unreachable ();
break;
}
return parameter;
}
static tree
cp_parser_template_id (cp_parser *parser,
bool template_keyword_p,
bool check_dependency_p,
enum tag_types tag_type,
bool is_declaration)
{
tree templ;
tree arguments;
tree template_id;
cp_token_position start_of_id = 0;
cp_token *next_token = NULL, *next_token_2 = NULL;
bool is_identifier;
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_TEMPLATE_ID)
{
cp_lexer_consume_token (parser->lexer);
return saved_checks_value (token->u.tree_check_value);
}
if ((token->type != CPP_NAME && token->keyword != RID_OPERATOR)
|| (token->type == CPP_NAME
&& !cp_parser_nth_token_starts_template_argument_list_p
(parser, 2)))
{
cp_parser_error (parser, "expected template-id");
return error_mark_node;
}
if (cp_parser_uncommitted_to_tentative_parse_p (parser))
start_of_id = cp_lexer_token_position (parser->lexer, false);
push_deferring_access_checks (dk_deferred);
is_identifier = false;
templ = cp_parser_template_name (parser, template_keyword_p,
check_dependency_p,
is_declaration,
tag_type,
&is_identifier);
if (templ == error_mark_node || is_identifier)
{
pop_deferring_access_checks ();
return templ;
}
tentative_firewall firewall (parser);
if (((next_token = cp_lexer_peek_token (parser->lexer))->type
== CPP_OPEN_SQUARE)
&& next_token->flags & DIGRAPH
&& ((next_token_2 = cp_lexer_peek_nth_token (parser->lexer, 2))->type
== CPP_COLON)
&& !(next_token_2->flags & PREV_WHITE))
{
cp_parser_parse_tentatively (parser);
next_token_2->type = CPP_SCOPE;
cp_lexer_consume_token (parser->lexer);
arguments = cp_parser_enclosed_template_argument_list (parser);
if (!cp_parser_parse_definitely (parser))
{
next_token_2->type = CPP_COLON;
cp_parser_error (parser, "expected %<<%>");
pop_deferring_access_checks ();
return error_mark_node;
}
if (permerror (next_token->location,
"%<<::%> cannot begin a template-argument list"))
{
static bool hint = false;
inform (next_token->location,
"%<<:%> is an alternate spelling for %<[%>."
" Insert whitespace between %<<%> and %<::%>");
if (!hint && !flag_permissive)
{
inform (next_token->location, "(if you use %<-fpermissive%> "
"or %<-std=c++11%>, or %<-std=gnu++11%> G++ will "
"accept your code)");
hint = true;
}
}
}
else
{
if (!cp_parser_require (parser, CPP_LESS, RT_LESS))
{
pop_deferring_access_checks ();
return error_mark_node;
}
arguments = cp_parser_enclosed_template_argument_list (parser);
}
location_t finish_loc
= get_finish (cp_lexer_previous_token (parser->lexer)->location);
location_t combined_loc
= make_location (token->location, token->location, finish_loc);
if (flag_concepts && check_auto_in_tmpl_args (templ, arguments))
template_id = error_mark_node;
else if (identifier_p (templ))
template_id = build_min_nt_loc (combined_loc,
TEMPLATE_ID_EXPR,
templ, arguments);
else if (DECL_TYPE_TEMPLATE_P (templ)
|| DECL_TEMPLATE_TEMPLATE_PARM_P (templ))
{
bool entering_scope;
entering_scope = (template_parm_scope_p ()
&& cp_lexer_next_token_is (parser->lexer,
CPP_SCOPE));
template_id
= finish_template_type (templ, arguments, entering_scope);
}
else if (flag_concepts
&& (template_id = (cp_parser_maybe_partial_concept_id
(parser, templ, arguments))))
return template_id;
else if (variable_template_p (templ))
{
template_id = lookup_template_variable (templ, arguments);
if (TREE_CODE (template_id) == TEMPLATE_ID_EXPR)
SET_EXPR_LOCATION (template_id, combined_loc);
}
else
{
gcc_assert ((DECL_FUNCTION_TEMPLATE_P (templ)
|| TREE_CODE (templ) == OVERLOAD
|| BASELINK_P (templ)));
template_id = lookup_template_function (templ, arguments);
if (TREE_CODE (template_id) == TEMPLATE_ID_EXPR)
SET_EXPR_LOCATION (template_id, combined_loc);
}
if (start_of_id
&& !(cp_parser_error_occurred (parser)
&& cp_parser_parsing_tentatively (parser)
&& parser->in_declarator_p))
{
token->type = CPP_TEMPLATE_ID;
token->location = combined_loc;
if (is_overloaded_fn (template_id))
lookup_keep (get_fns (template_id), true);
token->u.tree_check_value = ggc_cleared_alloc<struct tree_check> ();
token->u.tree_check_value->value = template_id;
token->u.tree_check_value->checks = get_deferred_access_checks ();
token->keyword = RID_MAX;
cp_lexer_purge_tokens_after (parser->lexer, start_of_id);
if (cp_parser_error_occurred (parser) && template_id != error_mark_node)
error_at (token->location, "parse error in template argument list");
}
pop_to_parent_deferring_access_checks ();
return template_id;
}
static tree
cp_parser_template_name (cp_parser* parser,
bool template_keyword_p,
bool check_dependency_p,
bool is_declaration,
enum tag_types tag_type,
bool *is_identifier)
{
tree identifier;
tree decl;
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_OPERATOR))
{
cp_parser_parse_tentatively (parser);
identifier = cp_parser_operator_function_id (parser);
if (!cp_parser_parse_definitely (parser))
{
cp_parser_error (parser, "expected template-name");
return error_mark_node;
}
}
else
identifier = cp_parser_identifier (parser);
if (identifier == error_mark_node)
return error_mark_node;
if (processing_template_decl
&& cp_parser_nth_token_starts_template_argument_list_p (parser, 1))
{
if (is_declaration
&& !template_keyword_p
&& parser->scope && TYPE_P (parser->scope)
&& check_dependency_p
&& dependent_scope_p (parser->scope)
&& !constructor_name_p (identifier, parser->scope))
{
cp_token_position start = 0;
error_at (token->location, "non-template %qD used as template",
identifier);
inform (token->location, "use %<%T::template %D%> to indicate that it is a template",
parser->scope, identifier);
if (cp_parser_simulate_error (parser))
start = cp_lexer_token_position (parser->lexer, true);
cp_lexer_consume_token (parser->lexer);
cp_parser_enclosed_template_argument_list (parser);
cp_parser_skip_to_closing_parenthesis (parser,
true,
true,
false);
if (start)
cp_lexer_purge_tokens_after (parser->lexer, start);
if (is_identifier)
*is_identifier = true;
parser->context->object_type = NULL_TREE;
return identifier;
}
if (template_keyword_p)
{
tree scope = (parser->scope ? parser->scope
: parser->context->object_type);
if (scope && TYPE_P (scope)
&& (!CLASS_TYPE_P (scope)
|| (check_dependency_p && dependent_type_p (scope))))
{
parser->context->object_type = NULL_TREE;
return identifier;
}
}
}
decl = cp_parser_lookup_name (parser, identifier,
tag_type,
true,
false,
check_dependency_p,
NULL,
token->location);
decl = strip_using_decl (decl);
if (TREE_CODE (decl) == TEMPLATE_DECL)
{
if (TREE_DEPRECATED (decl)
&& deprecated_state != DEPRECATED_SUPPRESS)
warn_deprecated_use (decl, NULL_TREE);
}
else
{
bool found = false;
for (lkp_iterator iter (MAYBE_BASELINK_FUNCTIONS (decl));
!found && iter; ++iter)
if (TREE_CODE (*iter) == TEMPLATE_DECL)
found = true;
if (!found)
{
cp_parser_error (parser, "expected template-name");
return error_mark_node;
}
}
if (DECL_FUNCTION_TEMPLATE_P (decl) || !DECL_P (decl))
{
tree scope = ovl_scope (decl);
if (TYPE_P (scope) && dependent_type_p (scope))
return identifier;
}
return decl;
}
static tree
cp_parser_template_argument_list (cp_parser* parser)
{
tree fixed_args[10];
unsigned n_args = 0;
unsigned alloced = 10;
tree *arg_ary = fixed_args;
tree vec;
bool saved_in_template_argument_list_p;
bool saved_ice_p;
bool saved_non_ice_p;
saved_in_template_argument_list_p = parser->in_template_argument_list_p;
parser->in_template_argument_list_p = true;
saved_ice_p = parser->integral_constant_expression_p;
parser->integral_constant_expression_p = false;
saved_non_ice_p = parser->non_integral_constant_expression_p;
parser->non_integral_constant_expression_p = false;
do
{
tree argument;
if (n_args)
cp_lexer_consume_token (parser->lexer);
argument = cp_parser_template_argument (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
if (argument == error_mark_node)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
error_at (token->location,
"expected parameter pack before %<...%>");
}
cp_lexer_consume_token (parser->lexer);
argument = make_pack_expansion (argument);
}
if (n_args == alloced)
{
alloced *= 2;
if (arg_ary == fixed_args)
{
arg_ary = XNEWVEC (tree, alloced);
memcpy (arg_ary, fixed_args, sizeof (tree) * n_args);
}
else
arg_ary = XRESIZEVEC (tree, arg_ary, alloced);
}
arg_ary[n_args++] = argument;
}
while (cp_lexer_next_token_is (parser->lexer, CPP_COMMA));
vec = make_tree_vec (n_args);
while (n_args--)
TREE_VEC_ELT (vec, n_args) = arg_ary[n_args];
if (arg_ary != fixed_args)
free (arg_ary);
parser->non_integral_constant_expression_p = saved_non_ice_p;
parser->integral_constant_expression_p = saved_ice_p;
parser->in_template_argument_list_p = saved_in_template_argument_list_p;
if (CHECKING_P)
SET_NON_DEFAULT_TEMPLATE_ARGS_COUNT (vec, TREE_VEC_LENGTH (vec));
return vec;
}
static tree
cp_parser_template_argument (cp_parser* parser)
{
tree argument;
bool template_p;
bool address_p;
bool maybe_type_id = false;
cp_token *token = NULL, *argument_start_token = NULL;
location_t loc = 0;
cp_id_kind idk;
cp_parser_parse_tentatively (parser);
argument = cp_parser_template_type_arg (parser);
if (!cp_parser_error_occurred (parser)
&& cxx_dialect == cxx98
&& cp_lexer_next_token_is (parser->lexer, CPP_RSHIFT))
{
maybe_type_id = true;
cp_parser_abort_tentative_parse (parser);
}
else
{
if (!cp_parser_next_token_ends_template_argument_p (parser))
cp_parser_error (parser, "expected template-argument");
if (cp_parser_parse_definitely (parser))
return argument;
}
cp_parser_parse_tentatively (parser);
argument_start_token = cp_lexer_peek_token (parser->lexer);
argument = cp_parser_id_expression (parser,
false,
true,
&template_p,
false,
false);
if (!cp_parser_next_token_ends_template_argument_p (parser))
cp_parser_error (parser, "expected template-argument");
if (!cp_parser_error_occurred (parser))
{
if (TREE_CODE (argument) != TYPE_DECL)
argument = cp_parser_lookup_name (parser, argument,
none_type,
template_p,
false,
true,
NULL,
argument_start_token->location);
if (tree decl = cp_parser_maybe_concept_name (parser, argument))
argument = decl;
else if (TREE_CODE (argument) != TEMPLATE_DECL
&& TREE_CODE (argument) != UNBOUND_CLASS_TEMPLATE)
cp_parser_error (parser, "expected template-name");
}
if (cp_parser_parse_definitely (parser))
{
if (TREE_DEPRECATED (argument))
warn_deprecated_use (argument, NULL_TREE);
return argument;
}
if (cxx_dialect > cxx14)
goto general_expr;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
cp_parser_parse_tentatively (parser);
argument = cp_parser_primary_expression (parser,
false,
false,
true,
&idk);
if (TREE_CODE (argument) != TEMPLATE_PARM_INDEX
|| !cp_parser_next_token_ends_template_argument_p (parser))
cp_parser_simulate_error (parser);
if (cp_parser_parse_definitely (parser))
return argument;
}
address_p = cp_lexer_next_token_is (parser->lexer, CPP_AND);
if (address_p)
{
loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);
}
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_NAME
|| token->keyword == RID_OPERATOR
|| token->type == CPP_SCOPE
|| token->type == CPP_TEMPLATE_ID
|| token->type == CPP_NESTED_NAME_SPECIFIER)
{
cp_parser_parse_tentatively (parser);
argument = cp_parser_primary_expression (parser,
address_p,
false,
true,
&idk);
if (cp_parser_error_occurred (parser)
|| !cp_parser_next_token_ends_template_argument_p (parser))
cp_parser_abort_tentative_parse (parser);
else
{
tree probe;
if (INDIRECT_REF_P (argument))
{
gcc_assert (REFERENCE_REF_P (argument));
argument = TREE_OPERAND (argument, 0);
}
probe = argument;
if (TREE_CODE (probe) == SCOPE_REF)
probe = TREE_OPERAND (probe, 1);
if (VAR_P (probe))
{
if (!address_p && !DECL_EXTERNAL_LINKAGE_P (probe))
cp_parser_simulate_error (parser);
}
else if (is_overloaded_fn (argument))
;
else if (address_p
&& (TREE_CODE (argument) == OFFSET_REF
|| TREE_CODE (argument) == SCOPE_REF))
;
else if (TREE_CODE (argument) == TEMPLATE_PARM_INDEX)
;
else
cp_parser_simulate_error (parser);
if (cp_parser_parse_definitely (parser))
{
if (address_p)
argument = build_x_unary_op (loc, ADDR_EXPR, argument,
tf_warning_or_error);
else
argument = convert_from_reference (argument);
return argument;
}
}
}
if (address_p)
{
cp_parser_error (parser, "invalid non-type template argument");
return error_mark_node;
}
general_expr:
if (maybe_type_id)
cp_parser_parse_tentatively (parser);
if (cxx_dialect <= cxx14)
argument = cp_parser_constant_expression (parser);
else
{
argument = cp_parser_assignment_expression (parser);
require_potential_constant_expression (argument);
}
if (!maybe_type_id)
return argument;
if (!cp_parser_next_token_ends_template_argument_p (parser))
cp_parser_error (parser, "expected template-argument");
if (cp_parser_parse_definitely (parser))
return argument;
return cp_parser_template_type_arg (parser);
}
static void
cp_parser_explicit_instantiation (cp_parser* parser)
{
int declares_class_or_enum;
cp_decl_specifier_seq decl_specifiers;
tree extension_specifier = NULL_TREE;
timevar_push (TV_TEMPLATE_INST);
if (cp_parser_allow_gnu_extensions_p (parser))
{
extension_specifier
= cp_parser_storage_class_specifier_opt (parser);
if (!extension_specifier)
extension_specifier
= cp_parser_function_specifier_opt (parser,
NULL);
}
cp_parser_require_keyword (parser, RID_TEMPLATE, RT_TEMPLATE);
begin_explicit_instantiation ();
push_deferring_access_checks (dk_no_check);
cp_parser_decl_specifier_seq (parser,
CP_PARSER_FLAGS_OPTIONAL,
&decl_specifiers,
&declares_class_or_enum);
if (declares_class_or_enum && cp_parser_declares_only_class_p (parser))
{
tree type;
type = check_tag_decl (&decl_specifiers,
true);
pop_deferring_access_checks ();
if (type)
do_type_instantiation (type, extension_specifier,
tf_error);
}
else
{
cp_declarator *declarator;
tree decl;
declarator
= cp_parser_declarator (parser, CP_PARSER_DECLARATOR_NAMED,
NULL,
NULL,
false,
false);
if (declares_class_or_enum & 2)
cp_parser_check_for_definition_in_return_type (declarator,
decl_specifiers.type,
decl_specifiers.locations[ds_type_spec]);
if (declarator != cp_error_declarator)
{
if (decl_spec_seq_has_spec_p (&decl_specifiers, ds_inline))
permerror (decl_specifiers.locations[ds_inline],
"explicit instantiation shall not use"
" %<inline%> specifier");
if (decl_spec_seq_has_spec_p (&decl_specifiers, ds_constexpr))
permerror (decl_specifiers.locations[ds_constexpr],
"explicit instantiation shall not use"
" %<constexpr%> specifier");
decl = grokdeclarator (declarator, &decl_specifiers,
NORMAL, 0, &decl_specifiers.attributes);
pop_deferring_access_checks ();
do_decl_instantiation (decl, extension_specifier);
}
else
{
pop_deferring_access_checks ();
cp_parser_skip_to_end_of_statement (parser);
}
}
end_explicit_instantiation ();
cp_parser_consume_semicolon_at_end_of_statement (parser);
timevar_pop (TV_TEMPLATE_INST);
}
static void
cp_parser_explicit_specialization (cp_parser* parser)
{
bool need_lang_pop;
cp_token *token = cp_lexer_peek_token (parser->lexer);
cp_parser_require_keyword (parser, RID_TEMPLATE, RT_TEMPLATE);
cp_parser_require (parser, CPP_LESS, RT_LESS);
cp_parser_require (parser, CPP_GREATER, RT_GREATER);
++parser->num_template_parameter_lists;
if (current_lang_name == lang_name_c)
{
error_at (token->location, "template specialization with C linkage");
maybe_show_extern_c_location ();
push_lang_context (lang_name_cplusplus);
need_lang_pop = true;
}
else
need_lang_pop = false;
if (!begin_specialization ())
{
end_specialization ();
return;
}
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TEMPLATE))
{
if (cp_lexer_peek_nth_token (parser->lexer, 2)->type == CPP_LESS
&& cp_lexer_peek_nth_token (parser->lexer, 3)->type != CPP_GREATER)
cp_parser_template_declaration_after_export (parser,
false);
else
cp_parser_explicit_specialization (parser);
}
else
cp_parser_single_declaration (parser,
NULL,
false,
true,
NULL);
end_specialization ();
if (need_lang_pop)
pop_lang_context ();
--parser->num_template_parameter_lists;
}
static tree
cp_parser_type_specifier (cp_parser* parser,
cp_parser_flags flags,
cp_decl_specifier_seq *decl_specs,
bool is_declaration,
int* declares_class_or_enum,
bool* is_cv_qualifier)
{
tree type_spec = NULL_TREE;
cp_token *token;
enum rid keyword;
cp_decl_spec ds = ds_last;
if (declares_class_or_enum)
*declares_class_or_enum = 0;
if (is_cv_qualifier)
*is_cv_qualifier = false;
token = cp_lexer_peek_token (parser->lexer);
keyword = token->keyword;
switch (keyword)
{
case RID_ENUM:
if ((flags & CP_PARSER_FLAGS_NO_TYPE_DEFINITIONS))
goto elaborated_type_specifier;
type_spec = cp_parser_enum_specifier (parser);
if (type_spec)
{
if (declares_class_or_enum)
*declares_class_or_enum = 2;
if (decl_specs)
cp_parser_set_decl_spec_type (decl_specs,
type_spec,
token,
true);
return type_spec;
}
else
goto elaborated_type_specifier;
case RID_CLASS:
case RID_STRUCT:
case RID_UNION:
if ((flags & CP_PARSER_FLAGS_NO_TYPE_DEFINITIONS))
goto elaborated_type_specifier;
cp_parser_parse_tentatively (parser);
type_spec = cp_parser_class_specifier (parser);
invoke_plugin_callbacks (PLUGIN_FINISH_TYPE, type_spec);
if (cp_parser_parse_definitely (parser))
{
if (declares_class_or_enum)
*declares_class_or_enum = 2;
if (decl_specs)
cp_parser_set_decl_spec_type (decl_specs,
type_spec,
token,
true);
return type_spec;
}
elaborated_type_specifier:
if (declares_class_or_enum)
*declares_class_or_enum = 1;
case RID_TYPENAME:
type_spec
= (cp_parser_elaborated_type_specifier
(parser,
decl_spec_seq_has_spec_p (decl_specs, ds_friend),
is_declaration));
if (decl_specs)
cp_parser_set_decl_spec_type (decl_specs,
type_spec,
token,
false);
return type_spec;
case RID_CONST:
ds = ds_const;
if (is_cv_qualifier)
*is_cv_qualifier = true;
break;
case RID_VOLATILE:
ds = ds_volatile;
if (is_cv_qualifier)
*is_cv_qualifier = true;
break;
case RID_RESTRICT:
ds = ds_restrict;
if (is_cv_qualifier)
*is_cv_qualifier = true;
break;
case RID_COMPLEX:
ds = ds_complex;
break;
default:
break;
}
if (ds != ds_last)
{
if (decl_specs)
{
set_and_check_decl_spec_loc (decl_specs, ds, token);
decl_specs->any_specifiers_p = true;
}
return cp_lexer_consume_token (parser->lexer)->u.value;
}
type_spec = cp_parser_simple_type_specifier (parser,
decl_specs,
flags);
if (!type_spec && !(flags & CP_PARSER_FLAGS_OPTIONAL))
{
cp_parser_error (parser, "expected type specifier");
return error_mark_node;
}
return type_spec;
}
static tree
cp_parser_simple_type_specifier (cp_parser* parser,
cp_decl_specifier_seq *decl_specs,
cp_parser_flags flags)
{
tree type = NULL_TREE;
cp_token *token;
int idx;
token = cp_lexer_peek_token (parser->lexer);
switch (token->keyword)
{
case RID_CHAR:
if (decl_specs)
decl_specs->explicit_char_p = true;
type = char_type_node;
break;
case RID_CHAR16:
type = char16_type_node;
break;
case RID_CHAR32:
type = char32_type_node;
break;
case RID_WCHAR:
type = wchar_type_node;
break;
case RID_BOOL:
type = boolean_type_node;
break;
case RID_SHORT:
set_and_check_decl_spec_loc (decl_specs, ds_short, token);
type = short_integer_type_node;
break;
case RID_INT:
if (decl_specs)
decl_specs->explicit_int_p = true;
type = integer_type_node;
break;
case RID_INT_N_0:
case RID_INT_N_1:
case RID_INT_N_2:
case RID_INT_N_3:
idx = token->keyword - RID_INT_N_0;
if (! int_n_enabled_p [idx])
break;
if (decl_specs)
{
decl_specs->explicit_intN_p = true;
decl_specs->int_n_idx = idx;
}
type = int_n_trees [idx].signed_type;
break;
case RID_LONG:
if (decl_specs)
set_and_check_decl_spec_loc (decl_specs, ds_long, token);
type = long_integer_type_node;
break;
case RID_SIGNED:
set_and_check_decl_spec_loc (decl_specs, ds_signed, token);
type = integer_type_node;
break;
case RID_UNSIGNED:
set_and_check_decl_spec_loc (decl_specs, ds_unsigned, token);
type = unsigned_type_node;
break;
case RID_FLOAT:
type = float_type_node;
break;
case RID_DOUBLE:
type = double_type_node;
break;
case RID_VOID:
type = void_type_node;
break;
case RID_AUTO:
maybe_warn_cpp0x (CPP0X_AUTO);
if (parser->auto_is_implicit_function_template_parm_p)
{
bool have_trailing_return_fn_decl = false;
cp_parser_parse_tentatively (parser);
cp_lexer_consume_token (parser->lexer);
while (cp_lexer_next_token_is_not (parser->lexer, CPP_EQ)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_CLOSE_PAREN)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_EOF))
{
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
cp_lexer_consume_token (parser->lexer);
cp_parser_skip_to_closing_parenthesis (parser,
false,
false,
true);
continue;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_DEREF))
{
have_trailing_return_fn_decl = true;
break;
}
cp_lexer_consume_token (parser->lexer);
}
cp_parser_abort_tentative_parse (parser);
if (have_trailing_return_fn_decl)
{
type = make_auto ();
break;
}
if (cxx_dialect >= cxx14)
{
type = synthesize_implicit_template_parm (parser, NULL_TREE);
type = TREE_TYPE (type);
}
else
type = error_mark_node;
if (current_class_type && LAMBDA_TYPE_P (current_class_type))
{
if (cxx_dialect < cxx14)
error_at (token->location,
"use of %<auto%> in lambda parameter declaration "
"only available with "
"-std=c++14 or -std=gnu++14");
}
else if (cxx_dialect < cxx14)
error_at (token->location,
"use of %<auto%> in parameter declaration "
"only available with "
"-std=c++14 or -std=gnu++14");
else if (!flag_concepts)
pedwarn (token->location, 0,
"use of %<auto%> in parameter declaration "
"only available with -fconcepts");
}
else
type = make_auto ();
break;
case RID_DECLTYPE:
cp_parser_decltype (parser);
cp_lexer_set_token_position (parser->lexer, token);
break;
case RID_TYPEOF:
cp_lexer_consume_token (parser->lexer);
type = cp_parser_sizeof_operand (parser, RID_TYPEOF);
if (!TYPE_P (type))
type = finish_typeof (type);
if (decl_specs)
cp_parser_set_decl_spec_type (decl_specs, type,
token,
false);
return type;
case RID_UNDERLYING_TYPE:
type = cp_parser_trait_expr (parser, RID_UNDERLYING_TYPE);
if (decl_specs)
cp_parser_set_decl_spec_type (decl_specs, type,
token,
false);
return type;
case RID_BASES:
case RID_DIRECT_BASES:
type = cp_parser_trait_expr (parser, token->keyword);
if (decl_specs)
cp_parser_set_decl_spec_type (decl_specs, type,
token,
false);
return type;
default:
break;
}
if (token->type == CPP_DECLTYPE
&& cp_lexer_peek_nth_token (parser->lexer, 2)->type != CPP_SCOPE)
{
type = saved_checks_value (token->u.tree_check_value);
if (decl_specs)
{
cp_parser_set_decl_spec_type (decl_specs, type,
token,
false);
decl_specs->decltype_p = true;
}
cp_lexer_consume_token (parser->lexer);
return type;
}
if (type)
{
if (decl_specs
&& (token->keyword != RID_SIGNED
&& token->keyword != RID_UNSIGNED
&& token->keyword != RID_SHORT
&& token->keyword != RID_LONG))
cp_parser_set_decl_spec_type (decl_specs,
type,
token,
false);
if (decl_specs)
decl_specs->any_specifiers_p = true;
cp_lexer_consume_token (parser->lexer);
if (type == error_mark_node)
return error_mark_node;
cp_parser_check_for_invalid_template_id (parser, type, none_type,
token->location);
return TYPE_NAME (type);
}
if (!(flags & CP_PARSER_FLAGS_NO_USER_DEFINED_TYPES))
{
bool qualified_p;
bool global_p;
if ((flags & CP_PARSER_FLAGS_OPTIONAL) || cxx_dialect >= cxx17)
cp_parser_parse_tentatively (parser);
token = cp_lexer_peek_token (parser->lexer);
global_p
= (cp_parser_global_scope_opt (parser,
false)
!= NULL_TREE);
qualified_p
= (cp_parser_nested_name_specifier_opt (parser,
false,
true,
false,
false)
!= NULL_TREE);
if (parser->scope
&& cp_parser_optional_template_keyword (parser))
{
type = cp_parser_template_id (parser,
true,
true,
none_type,
false);
if (TREE_CODE (type) != TYPE_DECL)
{
cp_parser_error (parser, "expected template-id for type");
type = NULL_TREE;
}
}
else
type = cp_parser_type_name (parser);
if (type
&& !global_p
&& !qualified_p
&& TREE_CODE (type) == TYPE_DECL
&& identifier_p (DECL_NAME (type)))
maybe_note_name_used_in_class (DECL_NAME (type), type);
if (((flags & CP_PARSER_FLAGS_OPTIONAL) || cxx_dialect >= cxx17)
&& !cp_parser_parse_definitely (parser))
type = NULL_TREE;
if (!type && cxx_dialect >= cxx17)
{
if (flags & CP_PARSER_FLAGS_OPTIONAL)
cp_parser_parse_tentatively (parser);
cp_parser_global_scope_opt (parser,
false);
cp_parser_nested_name_specifier_opt (parser,
false,
true,
false,
false);
tree name = cp_parser_identifier (parser);
if (name && TREE_CODE (name) == IDENTIFIER_NODE
&& parser->scope != error_mark_node)
{
tree tmpl = cp_parser_lookup_name (parser, name,
none_type,
false,
false,
true,
NULL,
token->location);
if (tmpl && tmpl != error_mark_node
&& (DECL_CLASS_TEMPLATE_P (tmpl)
|| DECL_TEMPLATE_TEMPLATE_PARM_P (tmpl)))
type = make_template_placeholder (tmpl);
else
{
type = error_mark_node;
if (!cp_parser_simulate_error (parser))
cp_parser_name_lookup_error (parser, name, tmpl,
NLE_TYPE, token->location);
}
}
else
type = error_mark_node;
if ((flags & CP_PARSER_FLAGS_OPTIONAL)
&& !cp_parser_parse_definitely (parser))
type = NULL_TREE;
}
if (type && decl_specs)
cp_parser_set_decl_spec_type (decl_specs, type,
token,
false);
}
if (!type && !(flags & CP_PARSER_FLAGS_OPTIONAL))
{
cp_parser_error (parser, "expected type-name");
return error_mark_node;
}
if (type && type != error_mark_node)
{
if (c_dialect_objc () && !parser->scope
&& (objc_is_id (type) || objc_is_class_name (type)))
{
tree protos = cp_parser_objc_protocol_refs_opt (parser);
tree qual_type = objc_get_protocol_qualified_type (type, protos);
if (decl_specs)
decl_specs->type = qual_type;
return qual_type;
}
cp_parser_check_for_invalid_template_id (parser, type,
none_type,
token->location);
}
return type;
}
static tree
cp_parser_type_name (cp_parser* parser)
{
return cp_parser_type_name (parser, false);
}
static tree
cp_parser_type_name (cp_parser* parser, bool typename_keyword_p)
{
tree type_decl;
cp_parser_parse_tentatively (parser);
type_decl = cp_parser_class_name (parser,
typename_keyword_p,
false,
none_type,
true,
false,
false);
if (!cp_parser_parse_definitely (parser))
{
if (cxx_dialect < cxx11)
return cp_parser_nonclass_name (parser);
cp_parser_parse_tentatively (parser);
type_decl = cp_parser_template_id (parser,
false,
true,
none_type,
false);
if (type_decl != NULL_TREE
&& TREE_CODE (type_decl) == TYPE_DECL
&& TYPE_DECL_ALIAS_P (type_decl))
gcc_assert (DECL_TEMPLATE_INSTANTIATION (type_decl));
else if (is_constrained_parameter (type_decl))
;
else
cp_parser_simulate_error (parser);
if (!cp_parser_parse_definitely (parser))
return cp_parser_nonclass_name (parser);
}
return type_decl;
}
static tree
cp_parser_maybe_constrained_type_specifier (cp_parser *parser,
tree decl, tree args)
{
gcc_assert (args ? TREE_CODE (args) == TREE_VEC : true);
if (parser->prevent_constrained_type_specifiers)
return NULL_TREE;
if (TREE_CODE (decl) != OVERLOAD && TREE_CODE (decl) != TEMPLATE_DECL)
return NULL_TREE;
tree placeholder = build_nt (WILDCARD_DECL);
tree check = build_concept_check (decl, placeholder, args);
if (check == error_mark_node)
return NULL_TREE;
tree conc;
tree proto;
if (!deduce_constrained_parameter (check, conc, proto))
return NULL_TREE;
if (processing_template_parmlist)
return build_constrained_parameter (conc, proto, args);
if (parser->auto_is_implicit_function_template_parm_p)
{
tree x = build_constrained_parameter (conc, proto, args);
return synthesize_implicit_template_parm (parser, x);
}
else
{
return make_constrained_auto (conc, args);
}
return NULL_TREE;
}
static tree
cp_parser_maybe_concept_name (cp_parser* parser, tree decl)
{
if (flag_concepts
&& (TREE_CODE (decl) == OVERLOAD
|| BASELINK_P (decl)
|| variable_concept_p (decl)))
return cp_parser_maybe_constrained_type_specifier (parser, decl, NULL_TREE);
else
return NULL_TREE;
}
static tree
cp_parser_maybe_partial_concept_id (cp_parser *parser, tree decl, tree args)
{
return cp_parser_maybe_constrained_type_specifier (parser, decl, args);
}
static tree
cp_parser_nonclass_name (cp_parser* parser)
{
tree type_decl;
tree identifier;
cp_token *token = cp_lexer_peek_token (parser->lexer);
identifier = cp_parser_identifier (parser);
if (identifier == error_mark_node)
return error_mark_node;
type_decl = cp_parser_lookup_name_simple (parser, identifier, token->location);
type_decl = strip_using_decl (type_decl);
if (tree decl = cp_parser_maybe_concept_name (parser, type_decl))
type_decl = decl;
if (TREE_CODE (type_decl) != TYPE_DECL
&& (objc_is_id (identifier) || objc_is_class_name (identifier)))
{
tree protos = cp_parser_objc_protocol_refs_opt (parser);
tree type = objc_get_protocol_qualified_type (identifier, protos);
if (type)
type_decl = TYPE_NAME (type);
}
if (TREE_CODE (type_decl) != TYPE_DECL
|| (objc_is_class_name (TREE_TYPE (type_decl))
&& cp_lexer_peek_token (parser->lexer)->type == CPP_DOT))
{
if (!cp_parser_simulate_error (parser))
cp_parser_name_lookup_error (parser, identifier, type_decl,
NLE_TYPE, token->location);
return error_mark_node;
}
else if (type_decl != error_mark_node
&& !parser->scope)
maybe_note_name_used_in_class (identifier, type_decl);
return type_decl;
}
static tree
cp_parser_elaborated_type_specifier (cp_parser* parser,
bool is_friend,
bool is_declaration)
{
enum tag_types tag_type;
tree identifier;
tree type = NULL_TREE;
tree attributes = NULL_TREE;
tree globalscope;
cp_token *token = NULL;
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_ENUM))
{
cp_lexer_consume_token (parser->lexer);
tag_type = enum_type;
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (cp_parser_is_keyword (token, RID_CLASS)
|| cp_parser_is_keyword (token, RID_STRUCT))
{
gcc_rich_location richloc (token->location);
richloc.add_range (input_location, false);
richloc.add_fixit_remove ();
pedwarn (&richloc, 0, "elaborated-type-specifier for "
"a scoped enum must not use the %qD keyword",
token->u.value);
cp_lexer_consume_token (parser->lexer);
}
attributes = cp_parser_attributes_opt (parser);
}
else if (cp_lexer_next_token_is_keyword (parser->lexer,
RID_TYPENAME))
{
cp_lexer_consume_token (parser->lexer);
tag_type = typename_type;
}
else
{
tag_type = cp_parser_class_key (parser);
if (tag_type == none_type)
return error_mark_node;
attributes = cp_parser_attributes_opt (parser);
}
globalscope =  cp_parser_global_scope_opt (parser,
false);
tree nested_name_specifier;
if (tag_type == typename_type && !globalscope)
{
nested_name_specifier
= cp_parser_nested_name_specifier (parser,
true,
true,
true,
is_declaration);
if (!nested_name_specifier)
return error_mark_node;
}
else
nested_name_specifier
= cp_parser_nested_name_specifier_opt (parser,
true,
true,
true,
is_declaration);
if (tag_type != enum_type)
{
bool template_p = false;
tree decl;
template_p = cp_parser_optional_template_keyword (parser);
if (!template_p)
cp_parser_parse_tentatively (parser);
token = cp_lexer_peek_token (parser->lexer);
decl = cp_parser_template_id (parser, template_p,
true,
tag_type,
is_declaration);
if (!template_p && !cp_parser_parse_definitely (parser))
;
else if (tag_type == typename_type && BASELINK_P (decl))
{
cp_parser_diagnose_invalid_type_name (parser, decl, token->location);
type = error_mark_node;
}
else if (TREE_CODE (decl) == TEMPLATE_ID_EXPR
&& tag_type == typename_type)
type = make_typename_type (parser->scope, decl,
typename_type,
tf_error);
else if (tag_type == typename_type && TREE_CODE (decl) != TYPE_DECL)
; 
else if (TREE_CODE (decl) == TYPE_DECL)
{
type = check_elaborated_type_specifier (tag_type, decl,
true);
if (type != error_mark_node
&& !nested_name_specifier && !is_friend
&& cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
check_unqualified_spec_or_inst (type, token->location);
}
else if (decl == error_mark_node)
type = error_mark_node; 
}
if (!type)
{
token = cp_lexer_peek_token (parser->lexer);
identifier = cp_parser_identifier (parser);
if (identifier == error_mark_node)
{
parser->scope = NULL_TREE;
return error_mark_node;
}
if (tag_type == typename_type
&& TREE_CODE (parser->scope) != NAMESPACE_DECL)
return cp_parser_make_typename_type (parser, identifier,
token->location);
bool template_parm_lists_apply
= parser->num_template_parameter_lists;
if (template_parm_lists_apply)
for (cp_binding_level *s = current_binding_level;
s && s->kind != sk_template_parms;
s = s->level_chain)
if (s->kind == sk_function_parms)
template_parm_lists_apply = false;
if (parser->scope)
{
tree decl;
tree ambiguous_decls;
decl = cp_parser_lookup_name (parser, identifier,
tag_type,
false,
false,
true,
&ambiguous_decls,
token->location);
if (ambiguous_decls)
return error_mark_node;
decl = (cp_parser_maybe_treat_template_as_class
(decl, is_friend
&& template_parm_lists_apply));
if (TREE_CODE (decl) != TYPE_DECL)
{
cp_parser_diagnose_invalid_type_name (parser,
identifier,
token->location);
return error_mark_node;
}
if (TREE_CODE (TREE_TYPE (decl)) != TYPENAME_TYPE)
{
bool allow_template = (template_parm_lists_apply
|| DECL_SELF_REFERENCE_P (decl));
type = check_elaborated_type_specifier (tag_type, decl,
allow_template);
if (type == error_mark_node)
return error_mark_node;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON)
&& !is_friend && !processing_explicit_instantiation)
warning (0, "declaration %qD does not declare anything", decl);
type = TREE_TYPE (decl);
}
else
{
tag_scope ts;
bool template_p;
if (is_friend)
ts = ts_within_enclosing_non_class;
else if (is_declaration
&& cp_lexer_next_token_is (parser->lexer,
CPP_SEMICOLON))
ts = ts_current;
else
ts = ts_global;
template_p =
(template_parm_lists_apply
&& (cp_parser_next_token_starts_class_definition_p (parser)
|| cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON)));
if (template_parm_lists_apply
&& !cp_parser_check_template_parameters (parser,
0,
false,
token->location,
NULL))
return error_mark_node;
type = xref_tag (tag_type, identifier, ts, template_p);
}
}
if (type == error_mark_node)
return error_mark_node;
if (attributes)
{
if (TREE_CODE (type) == TYPENAME_TYPE)
warning (OPT_Wattributes,
"attributes ignored on uninstantiated type");
else if (tag_type != enum_type && CLASSTYPE_TEMPLATE_INSTANTIATION (type)
&& ! processing_explicit_instantiation)
warning (OPT_Wattributes,
"attributes ignored on template instantiation");
else if (is_declaration && cp_parser_declares_only_class_p (parser))
cplus_decl_attributes (&type, attributes, (int) ATTR_FLAG_TYPE_IN_PLACE);
else
warning (OPT_Wattributes,
"attributes ignored on elaborated-type-specifier that is not a forward declaration");
}
if (tag_type != enum_type)
{
if (CLASS_TYPE_P (type))
CLASSTYPE_DECLARED_CLASS (type) = (tag_type == class_type);
cp_parser_check_class_key (tag_type, type);
}
cp_parser_check_for_invalid_template_id (parser, type, tag_type,
token->location);
return type;
}
static tree
cp_parser_enum_specifier (cp_parser* parser)
{
tree identifier;
tree type = NULL_TREE;
tree prev_scope;
tree nested_name_specifier = NULL_TREE;
tree attributes;
bool scoped_enum_p = false;
bool has_underlying_type = false;
bool nested_being_defined = false;
bool new_value_list = false;
bool is_new_type = false;
bool is_unnamed = false;
tree underlying_type = NULL_TREE;
cp_token *type_start_token = NULL;
bool saved_colon_corrects_to_scope_p = parser->colon_corrects_to_scope_p;
parser->colon_corrects_to_scope_p = false;
cp_parser_parse_tentatively (parser);
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_CLASS)
|| cp_lexer_next_token_is_keyword (parser->lexer, RID_STRUCT))
{
if (cxx_dialect < cxx11)
maybe_warn_cpp0x (CPP0X_SCOPED_ENUMS);
cp_lexer_consume_token (parser->lexer);
scoped_enum_p = true;
}
attributes = cp_parser_attributes_opt (parser);
parser->scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
parser->object_scope = NULL_TREE;
prev_scope = current_scope ();
type_start_token = cp_lexer_peek_token (parser->lexer);
push_deferring_access_checks (dk_no_check);
nested_name_specifier
= cp_parser_nested_name_specifier_opt (parser,
true,
false,
false,
false);
if (nested_name_specifier)
{
tree name;
identifier = cp_parser_identifier (parser);
name =  cp_parser_lookup_name (parser, identifier,
enum_type,
false,
false,
true,
NULL,
input_location);
if (name && name != error_mark_node)
{
type = TREE_TYPE (name);
if (TREE_CODE (type) == TYPENAME_TYPE)
{
if (template_parm_scope_p ())
pedwarn (type_start_token->location, OPT_Wpedantic,
"%qD is an enumeration template", name);
type = NULL_TREE;
}
}
else if (nested_name_specifier == error_mark_node)
;
else
{
error_at (type_start_token->location,
"%qD does not name an enumeration in %qT",
identifier, nested_name_specifier);
nested_name_specifier = error_mark_node;
}
}
else
{
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
identifier = cp_parser_identifier (parser);
else
{
identifier = make_anon_name ();
is_unnamed = true;
if (scoped_enum_p)
error_at (type_start_token->location,
"unnamed scoped enum is not allowed");
}
}
pop_deferring_access_checks ();
if (cp_lexer_next_token_is (parser->lexer, CPP_COLON))
{
cp_decl_specifier_seq type_specifiers;
cp_lexer_consume_token (parser->lexer);
cp_parser_type_specifier_seq (parser, false,
false,
&type_specifiers);
if (!cp_parser_parse_definitely (parser))
return NULL_TREE;
if (cxx_dialect < cxx11)
maybe_warn_cpp0x (CPP0X_SCOPED_ENUMS);
has_underlying_type = true;
if (type_specifiers.type != error_mark_node)
{
underlying_type = grokdeclarator (NULL, &type_specifiers, TYPENAME,
0, NULL);
if (underlying_type == error_mark_node
|| check_for_bare_parameter_packs (underlying_type))
underlying_type = NULL_TREE;
}
}
if (!cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
if (cxx_dialect < cxx11 || (!scoped_enum_p && !underlying_type))
{
cp_parser_error (parser, "expected %<{%>");
if (has_underlying_type)
{
type = NULL_TREE;
goto out;
}
}
if ((scoped_enum_p || underlying_type)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
{
cp_parser_error (parser, "expected %<;%> or %<{%>");
if (has_underlying_type)
{
type = NULL_TREE;
goto out;
}
}
}
if (!has_underlying_type && !cp_parser_parse_definitely (parser))
return NULL_TREE;
if (nested_name_specifier)
{
if (CLASS_TYPE_P (nested_name_specifier))
{
nested_being_defined = TYPE_BEING_DEFINED (nested_name_specifier);
TYPE_BEING_DEFINED (nested_name_specifier) = 1;
push_scope (nested_name_specifier);
}
else if (TREE_CODE (nested_name_specifier) == NAMESPACE_DECL)
{
push_nested_namespace (nested_name_specifier);
}
}
if (!cp_parser_check_type_definition (parser))
type = error_mark_node;
else
type = start_enum (identifier, type, underlying_type,
attributes, scoped_enum_p, &is_new_type);
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
timevar_push (TV_PARSE_ENUM);
if (nested_name_specifier
&& nested_name_specifier != error_mark_node)
{
if (!processing_specialization
&& CLASS_TYPE_P (nested_name_specifier)
&& CLASSTYPE_USE_TEMPLATE (nested_name_specifier))
error_at (type_start_token->location, "cannot add an enumerator "
"list to a template instantiation");
if (TREE_CODE (nested_name_specifier) == TYPENAME_TYPE)
{
error_at (type_start_token->location,
"%<%T::%E%> has not been declared",
TYPE_CONTEXT (nested_name_specifier),
nested_name_specifier);
type = error_mark_node;
}
else if (TREE_CODE (nested_name_specifier) != NAMESPACE_DECL
&& !CLASS_TYPE_P (nested_name_specifier))
{
error_at (type_start_token->location, "nested name specifier "
"%qT for enum declaration does not name a class "
"or namespace", nested_name_specifier);
type = error_mark_node;
}
else if (prev_scope && !is_ancestor (prev_scope,
nested_name_specifier))
{
if (at_namespace_scope_p ())
error_at (type_start_token->location,
"declaration of %qD in namespace %qD which does not "
"enclose %qD",
type, prev_scope, nested_name_specifier);
else
error_at (type_start_token->location,
"declaration of %qD in %qD which does not "
"enclose %qD",
type, prev_scope, nested_name_specifier);
type = error_mark_node;
}
else if (CLASS_TYPE_P (nested_name_specifier)
&& CLASS_TYPE_P (prev_scope)
&& same_type_p (nested_name_specifier, prev_scope))
{
permerror (type_start_token->location,
"extra qualification not allowed");
nested_name_specifier = NULL_TREE;
}
}
if (scoped_enum_p)
begin_scope (sk_scoped_enum, type);
matching_braces braces;
braces.consume_open (parser);
if (type == error_mark_node)
; 
else if (OPAQUE_ENUM_P (type)
|| (cxx_dialect > cxx98 && processing_specialization))
{
new_value_list = true;
SET_OPAQUE_ENUM_P (type, false);
DECL_SOURCE_LOCATION (TYPE_NAME (type)) = type_start_token->location;
}
else
{
error_at (type_start_token->location,
"multiple definition of %q#T", type);
inform (DECL_SOURCE_LOCATION (TYPE_MAIN_DECL (type)),
"previous definition here");
type = error_mark_node;
}
if (type == error_mark_node)
cp_parser_skip_to_end_of_block_or_statement (parser);
else if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_BRACE))
{
if (is_unnamed && !scoped_enum_p)
pedwarn (type_start_token->location, OPT_Wpedantic,
"ISO C++ forbids empty unnamed enum");
}
else
cp_parser_enumerator_list (parser, type);
braces.require_close (parser);
if (scoped_enum_p)
finish_scope ();
timevar_pop (TV_PARSE_ENUM);
}
else
{
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
{
if (is_unnamed)
error_at (type_start_token->location,
"opaque-enum-specifier without name");
else if (nested_name_specifier)
error_at (type_start_token->location,
"opaque-enum-specifier must use a simple identifier");
}
}
if (cp_parser_allow_gnu_extensions_p (parser))
{
tree trailing_attr = cp_parser_gnu_attributes_opt (parser);
cplus_decl_attributes (&type,
trailing_attr,
(int) ATTR_FLAG_TYPE_IN_PLACE);
}
if (type != error_mark_node)
{
if (new_value_list)
finish_enum_value_list (type);
if (is_new_type)
finish_enum (type);
}
if (nested_name_specifier)
{
if (CLASS_TYPE_P (nested_name_specifier))
{
TYPE_BEING_DEFINED (nested_name_specifier) = nested_being_defined;
pop_scope (nested_name_specifier);
}
else if (TREE_CODE (nested_name_specifier) == NAMESPACE_DECL)
{
pop_nested_namespace (nested_name_specifier);
}
}
out:
parser->colon_corrects_to_scope_p = saved_colon_corrects_to_scope_p;
return type;
}
static void
cp_parser_enumerator_list (cp_parser* parser, tree type)
{
while (true)
{
cp_parser_enumerator_definition (parser, type);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA))
break;
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_BRACE))
{
if (cxx_dialect < cxx11 && !in_system_header_at (input_location))
pedwarn (input_location, OPT_Wpedantic,
"comma at end of enumerator list");
break;
}
}
}
static void
cp_parser_enumerator_definition (cp_parser* parser, tree type)
{
tree identifier;
tree value;
location_t loc;
loc = cp_lexer_peek_token (parser->lexer)->location;
identifier = cp_parser_identifier (parser);
if (identifier == error_mark_node)
return;
tree attrs = cp_parser_attributes_opt (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_EQ))
{
cp_lexer_consume_token (parser->lexer);
value = cp_parser_constant_expression (parser);
}
else
value = NULL_TREE;
if (check_for_bare_parameter_packs (value))
value = error_mark_node;
build_enumerator (identifier, value, type, attrs, loc);
}
static tree
cp_parser_namespace_name (cp_parser* parser)
{
tree identifier;
tree namespace_decl;
cp_token *token = cp_lexer_peek_token (parser->lexer);
identifier = cp_parser_identifier (parser);
if (identifier == error_mark_node)
return error_mark_node;
namespace_decl = cp_parser_lookup_name (parser, identifier,
none_type,
false,
true,
true,
NULL,
token->location);
if (namespace_decl == error_mark_node
|| TREE_CODE (namespace_decl) != NAMESPACE_DECL)
{
if (!cp_parser_uncommitted_to_tentative_parse_p (parser))
{
error_at (token->location, "%qD is not a namespace-name", identifier);
if (namespace_decl == error_mark_node
&& parser->scope && TREE_CODE (parser->scope) == NAMESPACE_DECL)
suggest_alternative_in_explicit_scope (token->location, identifier,
parser->scope);
}
cp_parser_error (parser, "expected namespace-name");
namespace_decl = error_mark_node;
}
return namespace_decl;
}
static void
cp_parser_namespace_definition (cp_parser* parser)
{
tree identifier;
int nested_definition_count = 0;
cp_ensure_no_omp_declare_simd (parser);
cp_ensure_no_oacc_routine (parser);
bool is_inline = cp_lexer_next_token_is_keyword (parser->lexer, RID_INLINE);
if (is_inline)
{
maybe_warn_cpp0x (CPP0X_INLINE_NAMESPACES);
cp_lexer_consume_token (parser->lexer);
}
cp_token* token
= cp_parser_require_keyword (parser, RID_NAMESPACE, RT_NAMESPACE);
tree attribs = cp_parser_attributes_opt (parser);
for (;;)
{
identifier = NULL_TREE;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
identifier = cp_parser_identifier (parser);
attribs = attr_chainon (attribs, cp_parser_attributes_opt (parser));
}
if (cp_lexer_next_token_is_not (parser->lexer, CPP_SCOPE))
break;
if (!nested_definition_count && cxx_dialect < cxx17)
pedwarn (input_location, OPT_Wpedantic,
"nested namespace definitions only available with "
"-std=c++17 or -std=gnu++17");
if (int count = identifier ? push_namespace (identifier) : 0)
nested_definition_count += count;
else
cp_parser_error (parser, "nested namespace name required");
cp_lexer_consume_token (parser->lexer);
}
if (nested_definition_count && !identifier)
cp_parser_error (parser, "namespace name required");
if (nested_definition_count && attribs)
error_at (token->location,
"a nested namespace definition cannot have attributes");
if (nested_definition_count && is_inline)
error_at (token->location,
"a nested namespace definition cannot be inline");
nested_definition_count += push_namespace (identifier, is_inline);
bool has_visibility = handle_namespace_attrs (current_namespace, attribs);
warning  (OPT_Wnamespaces, "namespace %qD entered", current_namespace);
matching_braces braces;
if (braces.require_open (parser))
{
cp_parser_namespace_body (parser);
braces.require_close (parser);
}
if (has_visibility)
pop_visibility (1);
while (nested_definition_count--)
pop_namespace ();
}
static void
cp_parser_namespace_body (cp_parser* parser)
{
cp_parser_declaration_seq_opt (parser);
}
static void
cp_parser_namespace_alias_definition (cp_parser* parser)
{
tree identifier;
tree namespace_specifier;
cp_token *token = cp_lexer_peek_token (parser->lexer);
cp_parser_require_keyword (parser, RID_NAMESPACE, RT_NAMESPACE);
identifier = cp_parser_identifier (parser);
if (identifier == error_mark_node)
return;
if (!cp_parser_uncommitted_to_tentative_parse_p (parser)
&& cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE)) 
{
error_at (token->location, "%<namespace%> definition is not allowed here");
cp_lexer_consume_token (parser->lexer);
if (cp_parser_skip_to_closing_brace (parser))
cp_lexer_consume_token (parser->lexer);
return;
}
cp_parser_require (parser, CPP_EQ, RT_EQ);
namespace_specifier
= cp_parser_qualified_namespace_specifier (parser);
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
do_namespace_alias (identifier, namespace_specifier);
}
static tree
cp_parser_qualified_namespace_specifier (cp_parser* parser)
{
cp_parser_global_scope_opt (parser,
false);
cp_parser_nested_name_specifier_opt (parser,
false,
true,
false,
true);
return cp_parser_namespace_name (parser);
}
static bool
cp_parser_using_declaration (cp_parser* parser, 
bool access_declaration_p)
{
cp_token *token;
bool typename_p = false;
bool global_scope_p;
tree decl;
tree identifier;
tree qscope;
int oldcount = errorcount;
cp_token *diag_token = NULL;
if (access_declaration_p)
{
diag_token = cp_lexer_peek_token (parser->lexer);
cp_parser_parse_tentatively (parser);
}
else
{
cp_parser_require_keyword (parser, RID_USING, RT_USING);
again:
token = cp_lexer_peek_token (parser->lexer);
if (token->keyword == RID_TYPENAME)
{
typename_p = true;
cp_lexer_consume_token (parser->lexer);
}
}
global_scope_p
= (cp_parser_global_scope_opt (parser,
false)
!= NULL_TREE);
if (typename_p || !global_scope_p)
{
qscope = cp_parser_nested_name_specifier (parser, typename_p,
true,
false,
true);
if (!qscope && !cp_parser_uncommitted_to_tentative_parse_p (parser))
{
cp_parser_skip_to_end_of_block_or_statement (parser);
return false;
}
}
else
qscope = cp_parser_nested_name_specifier_opt (parser,
false,
true,
false,
true);
if (!qscope)
qscope = global_namespace;
else if (UNSCOPED_ENUM_P (qscope))
qscope = CP_TYPE_CONTEXT (qscope);
if (access_declaration_p && cp_parser_error_occurred (parser))
return cp_parser_parse_definitely (parser);
token = cp_lexer_peek_token (parser->lexer);
identifier = cp_parser_unqualified_id (parser,
false,
true,
true,
false);
if (access_declaration_p)
{
if (cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
cp_parser_simulate_error (parser);
if (!cp_parser_parse_definitely (parser))
return false;
}
else if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
cp_token *ell = cp_lexer_consume_token (parser->lexer);
if (cxx_dialect < cxx17
&& !in_system_header_at (ell->location))
pedwarn (ell->location, 0,
"pack expansion in using-declaration only available "
"with -std=c++17 or -std=gnu++17");
qscope = make_pack_expansion (qscope);
}
if (qscope == error_mark_node || identifier == error_mark_node)
;
else if (!identifier_p (identifier)
&& TREE_CODE (identifier) != BIT_NOT_EXPR)
error_at (token->location,
"a template-id may not appear in a using-declaration");
else
{
if (at_class_scope_p ())
{
decl = do_class_using_decl (qscope, identifier);
if (decl && typename_p)
USING_DECL_TYPENAME_P (decl) = 1;
if (check_for_bare_parameter_packs (decl))
{
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
return false;
}
else
finish_member_declaration (decl);
}
else
{
decl = cp_parser_lookup_name_simple (parser,
identifier,
token->location);
if (decl == error_mark_node)
cp_parser_name_lookup_error (parser, identifier,
decl, NLE_NULL,
token->location);
else if (check_for_bare_parameter_packs (decl))
{
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
return false;
}
else if (!at_namespace_scope_p ())
finish_local_using_decl (decl, qscope, identifier);
else
finish_namespace_using_decl (decl, qscope, identifier);
}
}
if (!access_declaration_p
&& cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
{
cp_token *comma = cp_lexer_consume_token (parser->lexer);
if (cxx_dialect < cxx17)
pedwarn (comma->location, 0,
"comma-separated list in using-declaration only available "
"with -std=c++17 or -std=gnu++17");
goto again;
}
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
if (access_declaration_p && errorcount == oldcount)
warning_at (diag_token->location, OPT_Wdeprecated,
"access declarations are deprecated "
"in favour of using-declarations; "
"suggestion: add the %<using%> keyword");
return true;
}
static tree
cp_parser_alias_declaration (cp_parser* parser)
{
tree id, type, decl, pushed_scope = NULL_TREE, attributes;
location_t id_location;
cp_declarator *declarator;
cp_decl_specifier_seq decl_specs;
bool member_p;
const char *saved_message = NULL;
cp_token *using_token
= cp_parser_require_keyword (parser, RID_USING, RT_USING);
if (using_token == NULL)
return error_mark_node;
id_location = cp_lexer_peek_token (parser->lexer)->location;
id = cp_parser_identifier (parser);
if (id == error_mark_node)
return error_mark_node;
cp_token *attrs_token = cp_lexer_peek_token (parser->lexer);
attributes = cp_parser_attributes_opt (parser);
if (attributes == error_mark_node)
return error_mark_node;
cp_parser_require (parser, CPP_EQ, RT_EQ);
if (cp_parser_error_occurred (parser))
return error_mark_node;
cp_parser_commit_to_tentative_parse (parser);
if (parser->num_template_parameter_lists)
{
saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message =
G_("types may not be defined in alias template declarations");
}
type = cp_parser_type_id (parser);
if (parser->num_template_parameter_lists)
parser->type_definition_forbidden_message = saved_message;
if (type == error_mark_node
|| !cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON))
{
cp_parser_skip_to_end_of_block_or_statement (parser);
return error_mark_node;
}
clear_decl_specs (&decl_specs);
decl_specs.type = type;
if (attributes != NULL_TREE)
{
decl_specs.attributes = attributes;
set_and_check_decl_spec_loc (&decl_specs,
ds_attribute,
attrs_token);
}
set_and_check_decl_spec_loc (&decl_specs,
ds_typedef,
using_token);
set_and_check_decl_spec_loc (&decl_specs,
ds_alias,
using_token);
if (parser->num_template_parameter_lists
&& !cp_parser_check_template_parameters (parser,
0,
false,
id_location,
NULL))
return error_mark_node;
declarator = make_id_declarator (NULL_TREE, id, sfk_none);
declarator->id_loc = id_location;
member_p = at_class_scope_p ();
if (member_p)
decl = grokfield (declarator, &decl_specs, NULL_TREE, false,
NULL_TREE, attributes);
else
decl = start_decl (declarator, &decl_specs, 0,
attributes, NULL_TREE, &pushed_scope);
if (decl == error_mark_node)
return decl;
if (flag_concepts && current_template_parms)
{
tree reqs = TEMPLATE_PARMS_CONSTRAINTS (current_template_parms);
tree constr = build_constraints (reqs, NULL_TREE);
set_constraints (decl, constr);
}
cp_finish_decl (decl, NULL_TREE, 0, NULL_TREE, 0);
if (pushed_scope)
pop_scope (pushed_scope);
if (DECL_LANG_SPECIFIC (decl)
&& DECL_TEMPLATE_INFO (decl)
&& PRIMARY_TEMPLATE_P (DECL_TI_TEMPLATE (decl)))
{
decl = DECL_TI_TEMPLATE (decl);
if (member_p)
check_member_template (decl);
}
return decl;
}
static void
cp_parser_using_directive (cp_parser* parser)
{
tree namespace_decl;
tree attribs;
cp_parser_require_keyword (parser, RID_USING, RT_USING);
cp_parser_require_keyword (parser, RID_NAMESPACE, RT_NAMESPACE);
cp_parser_global_scope_opt (parser, false);
cp_parser_nested_name_specifier_opt (parser,
false,
true,
false,
true);
namespace_decl = cp_parser_namespace_name (parser);
attribs = cp_parser_attributes_opt (parser);
if (namespace_bindings_p ())
finish_namespace_using_directive (namespace_decl, attribs);
else
finish_local_using_directive (namespace_decl, attribs);
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
}
static void
cp_parser_asm_definition (cp_parser* parser)
{
tree string;
tree outputs = NULL_TREE;
tree inputs = NULL_TREE;
tree clobbers = NULL_TREE;
tree labels = NULL_TREE;
tree asm_stmt;
bool extended_p = false;
bool invalid_inputs_p = false;
bool invalid_outputs_p = false;
required_token missing = RT_NONE;
cp_parser_require_keyword (parser, RID_ASM, RT_ASM);
if (parser->in_function_body
&& DECL_DECLARED_CONSTEXPR_P (current_function_decl))
{
error ("%<asm%> in %<constexpr%> function");
cp_function_chain->invalid_constexpr = true;
}
location_t volatile_loc = UNKNOWN_LOCATION;
location_t inline_loc = UNKNOWN_LOCATION;
location_t goto_loc = UNKNOWN_LOCATION;
if (cp_parser_allow_gnu_extensions_p (parser) && parser->in_function_body)
for (;;)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
location_t loc = token->location;
switch (cp_lexer_peek_token (parser->lexer)->keyword)
{
case RID_VOLATILE:
if (volatile_loc)
{
error_at (loc, "duplicate asm qualifier %qT", token->u.value);
inform (volatile_loc, "first seen here");
}
else
volatile_loc = loc;
cp_lexer_consume_token (parser->lexer);
continue;
case RID_INLINE:
if (inline_loc)
{
error_at (loc, "duplicate asm qualifier %qT", token->u.value);
inform (inline_loc, "first seen here");
}
else
inline_loc = loc;
cp_lexer_consume_token (parser->lexer);
continue;
case RID_GOTO:
if (goto_loc)
{
error_at (loc, "duplicate asm qualifier %qT", token->u.value);
inform (goto_loc, "first seen here");
}
else
goto_loc = loc;
cp_lexer_consume_token (parser->lexer);
continue;
case RID_CONST:
case RID_RESTRICT:
error_at (loc, "%qT is not an asm qualifier", token->u.value);
cp_lexer_consume_token (parser->lexer);
continue;
default:
break;
}
break;
}
bool volatile_p = (volatile_loc != UNKNOWN_LOCATION);
bool inline_p = (inline_loc != UNKNOWN_LOCATION);
bool goto_p = (goto_loc != UNKNOWN_LOCATION);
if (!cp_parser_require (parser, CPP_OPEN_PAREN, RT_OPEN_PAREN))
return;
string = cp_parser_string_literal (parser, false, false);
if (string == error_mark_node)
{
cp_parser_skip_to_closing_parenthesis (parser, true, false,
true);
return;
}
if (cp_parser_allow_gnu_extensions_p (parser)
&& parser->in_function_body
&& (cp_lexer_next_token_is (parser->lexer, CPP_COLON)
|| cp_lexer_next_token_is (parser->lexer, CPP_SCOPE)))
{
bool inputs_p = false;
bool clobbers_p = false;
bool labels_p = false;
extended_p = true;
if (cp_lexer_next_token_is (parser->lexer, CPP_COLON))
{
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is_not (parser->lexer,
CPP_COLON)
&& cp_lexer_next_token_is_not (parser->lexer,
CPP_SCOPE)
&& cp_lexer_next_token_is_not (parser->lexer,
CPP_CLOSE_PAREN)
&& !goto_p)
{
outputs = cp_parser_asm_operand_list (parser);
if (outputs == error_mark_node)
invalid_outputs_p = true;
}
}
else if (cp_lexer_next_token_is (parser->lexer, CPP_SCOPE))
inputs_p = true;
if (inputs_p
|| cp_lexer_next_token_is (parser->lexer, CPP_COLON))
{
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is_not (parser->lexer,
CPP_COLON)
&& cp_lexer_next_token_is_not (parser->lexer,
CPP_SCOPE)
&& cp_lexer_next_token_is_not (parser->lexer,
CPP_CLOSE_PAREN))
{
inputs = cp_parser_asm_operand_list (parser);
if (inputs == error_mark_node)
invalid_inputs_p = true;
}
}
else if (cp_lexer_next_token_is (parser->lexer, CPP_SCOPE))
clobbers_p = true;
if (clobbers_p
|| cp_lexer_next_token_is (parser->lexer, CPP_COLON))
{
clobbers_p = true;
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is_not (parser->lexer,
CPP_COLON)
&& cp_lexer_next_token_is_not (parser->lexer,
CPP_CLOSE_PAREN))
clobbers = cp_parser_asm_clobber_list (parser);
}
else if (goto_p && cp_lexer_next_token_is (parser->lexer, CPP_SCOPE))
labels_p = true;
if (labels_p
|| (goto_p && cp_lexer_next_token_is (parser->lexer, CPP_COLON)))
{
labels_p = true;
cp_lexer_consume_token (parser->lexer);
labels = cp_parser_asm_label_list (parser);
}
if (goto_p && !labels_p)
missing = clobbers_p ? RT_COLON : RT_COLON_SCOPE;
}
else if (goto_p)
missing = RT_COLON_SCOPE;
if (!cp_parser_require (parser, missing ? CPP_COLON : CPP_CLOSE_PAREN,
missing ? missing : RT_CLOSE_PAREN))
cp_parser_skip_to_closing_parenthesis (parser, true, false,
true);
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
if (!invalid_inputs_p && !invalid_outputs_p)
{
if (parser->in_function_body)
{
asm_stmt = finish_asm_stmt (volatile_p, string, outputs,
inputs, clobbers, labels, inline_p);
if (!extended_p)
{
tree temp = asm_stmt;
if (TREE_CODE (temp) == CLEANUP_POINT_EXPR)
temp = TREE_OPERAND (temp, 0);
ASM_INPUT_P (temp) = 1;
}
}
else
symtab->finalize_toplevel_asm (string);
}
}
static tree
strip_declarator_types (tree type, cp_declarator *declarator)
{
for (cp_declarator *d = declarator; d;)
switch (d->kind)
{
case cdk_id:
case cdk_decomp:
case cdk_error:
d = NULL;
break;
default:
if (TYPE_PTRMEMFUNC_P (type))
type = TYPE_PTRMEMFUNC_FN_TYPE (type);
type = TREE_TYPE (type);
d = d->declarator;
break;
}
return type;
}
static tree
cp_parser_init_declarator (cp_parser* parser,
cp_decl_specifier_seq *decl_specifiers,
vec<deferred_access_check, va_gc> *checks,
bool function_definition_allowed_p,
bool member_p,
int declares_class_or_enum,
bool* function_definition_p,
tree* maybe_range_for_decl,
location_t* init_loc,
tree* auto_result)
{
cp_token *token = NULL, *asm_spec_start_token = NULL,
*attributes_start_token = NULL;
cp_declarator *declarator;
tree prefix_attributes;
tree attributes = NULL;
tree asm_specification;
tree initializer;
tree decl = NULL_TREE;
tree scope;
int is_initialized;
enum cpp_ttype initialization_kind;
bool is_direct_init = false;
bool is_non_constant_init;
int ctor_dtor_or_conv_p;
bool friend_p = cp_parser_friend_p (decl_specifiers);
tree pushed_scope = NULL_TREE;
bool range_for_decl_p = false;
bool saved_default_arg_ok_p = parser->default_arg_ok_p;
location_t tmp_init_loc = UNKNOWN_LOCATION;
prefix_attributes = decl_specifiers->attributes;
if (function_definition_p)
*function_definition_p = false;
if (decl_spec_seq_has_spec_p (decl_specifiers, ds_typedef))
parser->default_arg_ok_p = false;
resume_deferring_access_checks ();
token = cp_lexer_peek_token (parser->lexer);
declarator
= cp_parser_declarator (parser, CP_PARSER_DECLARATOR_NAMED,
&ctor_dtor_or_conv_p,
NULL,
member_p, friend_p);
stop_deferring_access_checks ();
parser->default_arg_ok_p = saved_default_arg_ok_p;
if (declarator == cp_error_declarator)
return error_mark_node;
if (!cp_parser_check_declarator_template_parameters (parser, declarator,
token->location))
return error_mark_node;
if (declares_class_or_enum & 2)
cp_parser_check_for_definition_in_return_type (declarator,
decl_specifiers->type,
decl_specifiers->locations[ds_type_spec]);
scope = get_scope_of_declarator (declarator);
decl_specifiers->type
= maybe_update_decl_type (decl_specifiers->type, scope);
if (cp_parser_allow_gnu_extensions_p (parser))
{
asm_spec_start_token = cp_lexer_peek_token (parser->lexer);
asm_specification = cp_parser_asm_specification_opt (parser);
}
else
asm_specification = NULL_TREE;
attributes_start_token = cp_lexer_peek_token (parser->lexer);
attributes = cp_parser_attributes_opt (parser);
token = cp_lexer_peek_token (parser->lexer);
bool bogus_implicit_tmpl = false;
if (function_declarator_p (declarator))
{
if (!decl_specifiers->type
&& ctor_dtor_or_conv_p <= 0
&& cxx_dialect >= cxx17)
{
cp_declarator *id = get_id_declarator (declarator);
tree name = id->u.id.unqualified_name;
parser->scope = id->u.id.qualifying_scope;
tree tmpl = cp_parser_lookup_name_simple (parser, name, id->id_loc);
if (tmpl
&& (DECL_CLASS_TEMPLATE_P (tmpl)
|| DECL_TEMPLATE_TEMPLATE_PARM_P (tmpl)))
{
id->u.id.unqualified_name = dguide_name (tmpl);
id->u.id.sfk = sfk_deduction_guide;
ctor_dtor_or_conv_p = 1;
}
}
if (cp_parser_token_starts_function_definition_p (token))
{
if (!function_definition_allowed_p)
{
cp_parser_error (parser,
"a function-definition is not allowed here");
return error_mark_node;
}
location_t func_brace_location
= cp_lexer_peek_token (parser->lexer)->location;
if (asm_specification)
error_at (asm_spec_start_token->location,
"an asm-specification is not allowed "
"on a function-definition");
if (attributes)
error_at (attributes_start_token->location,
"attributes are not allowed "
"on a function-definition");
*function_definition_p = true;
if (member_p)
decl = cp_parser_save_member_function_body (parser,
decl_specifiers,
declarator,
prefix_attributes);
else
decl =
(cp_parser_function_definition_from_specifiers_and_declarator
(parser, decl_specifiers, prefix_attributes, declarator));
if (decl != error_mark_node && DECL_STRUCT_FUNCTION (decl))
{
DECL_STRUCT_FUNCTION (decl)->function_start_locus
= func_brace_location;
}
return decl;
}
}
else if (parser->fully_implicit_function_template_p)
{
bogus_implicit_tmpl = true;
finish_fully_implicit_template (parser, NULL_TREE);
}
if (!decl_specifiers->any_specifiers_p && ctor_dtor_or_conv_p <= 0)
{
cp_parser_error (parser,
"expected constructor, destructor, or type conversion");
return error_mark_node;
}
if (token->type == CPP_EQ
|| token->type == CPP_OPEN_PAREN
|| token->type == CPP_OPEN_BRACE)
{
is_initialized = SD_INITIALIZED;
initialization_kind = token->type;
if (maybe_range_for_decl)
*maybe_range_for_decl = error_mark_node;
tmp_init_loc = token->location;
if (init_loc && *init_loc == UNKNOWN_LOCATION)
*init_loc = tmp_init_loc;
if (token->type == CPP_EQ
&& function_declarator_p (declarator))
{
cp_token *t2 = cp_lexer_peek_nth_token (parser->lexer, 2);
if (t2->keyword == RID_DEFAULT)
is_initialized = SD_DEFAULTED;
else if (t2->keyword == RID_DELETE)
is_initialized = SD_DELETED;
}
}
else
{
if (token->type != CPP_COMMA
&& token->type != CPP_SEMICOLON)
{
if (maybe_range_for_decl && *maybe_range_for_decl != error_mark_node)
range_for_decl_p = true;
else
{
if (!maybe_range_for_decl)
cp_parser_error (parser, "expected initializer");
return error_mark_node;
}
}
is_initialized = SD_UNINITIALIZED;
initialization_kind = CPP_EOF;
}
cp_parser_commit_to_tentative_parse (parser);
if (!member_p)
{
if (parser->in_unbraced_linkage_specification_p)
decl_specifiers->storage_class = sc_extern;
decl = start_decl (declarator, decl_specifiers,
range_for_decl_p? SD_INITIALIZED : is_initialized,
attributes, prefix_attributes, &pushed_scope);
cp_finalize_omp_declare_simd (parser, decl);
cp_finalize_oacc_routine (parser, decl, false);
if (DECL_P (decl)
&& declarator->id_loc != UNKNOWN_LOCATION
&& DECL_SOURCE_LOCATION (decl) == input_location)
DECL_SOURCE_LOCATION (decl) = declarator->id_loc;
}
else if (scope)
pushed_scope = push_scope (scope);
if (!member_p && decl)
{
tree saved_current_function_decl = NULL_TREE;
if (TREE_CODE (decl) == FUNCTION_DECL)
{
saved_current_function_decl = current_function_decl;
current_function_decl = decl;
}
cp_parser_perform_template_parameter_access_checks (checks);
perform_deferred_access_checks (tf_warning_or_error);
if (TREE_CODE (decl) == FUNCTION_DECL)
current_function_decl = saved_current_function_decl;
}
initializer = NULL_TREE;
is_direct_init = false;
is_non_constant_init = true;
if (is_initialized)
{
if (function_declarator_p (declarator))
{
if (initialization_kind == CPP_EQ)
initializer = cp_parser_pure_specifier (parser);
else
{
if (decl != error_mark_node)
error_at (tmp_init_loc, "initializer provided for function");
cp_parser_skip_to_closing_parenthesis (parser,
true,
false,
true);
}
}
else
{
if (!member_p && processing_template_decl && decl != error_mark_node)
start_lambda_scope (decl);
initializer = cp_parser_initializer (parser,
&is_direct_init,
&is_non_constant_init);
if (!member_p && processing_template_decl && decl != error_mark_node)
finish_lambda_scope ();
if (initializer == error_mark_node)
cp_parser_skip_to_end_of_statement (parser);
}
}
if (cp_parser_allow_gnu_extensions_p (parser)
&& initialization_kind == CPP_OPEN_PAREN
&& cp_parser_attributes_opt (parser)
&& permerror (input_location,
"attributes after parenthesized initializer ignored"))
{
static bool hint;
if (flag_permissive && !hint)
{
hint = true;
inform (input_location,
"this flexibility is deprecated and will be removed");
}
}
if (bogus_implicit_tmpl && decl != error_mark_node)
error_at (DECL_SOURCE_LOCATION (decl),
"non-function %qD declared as implicit template", decl);
if (member_p)
{
if (pushed_scope)
{
pop_scope (pushed_scope);
pushed_scope = NULL_TREE;
}
decl = grokfield (declarator, decl_specifiers,
initializer, !is_non_constant_init,
NULL_TREE,
attr_chainon (attributes, prefix_attributes));
if (decl && TREE_CODE (decl) == FUNCTION_DECL)
cp_parser_save_default_args (parser, decl);
cp_finalize_omp_declare_simd (parser, decl);
cp_finalize_oacc_routine (parser, decl, false);
}
if (!member_p && decl && decl != error_mark_node && !range_for_decl_p)
{
cp_finish_decl (decl,
initializer, !is_non_constant_init,
asm_specification,
((is_direct_init || !is_initialized)
? LOOKUP_NORMAL : LOOKUP_IMPLICIT));
}
else if ((cxx_dialect != cxx98) && friend_p
&& decl && TREE_CODE (decl) == FUNCTION_DECL)
check_default_tmpl_args (decl, current_template_parms, true, 
false, 1);
if (!friend_p && pushed_scope)
pop_scope (pushed_scope);
if (function_declarator_p (declarator)
&& parser->fully_implicit_function_template_p)
{
if (member_p)
decl = finish_fully_implicit_template (parser, decl);
else
finish_fully_implicit_template (parser, 0);
}
if (auto_result && is_initialized && decl_specifiers->type
&& type_uses_auto (decl_specifiers->type))
*auto_result = strip_declarator_types (TREE_TYPE (decl), declarator);
return decl;
}
static cp_declarator *
cp_parser_declarator (cp_parser* parser,
cp_parser_declarator_kind dcl_kind,
int* ctor_dtor_or_conv_p,
bool* parenthesized_p,
bool member_p, bool friend_p)
{
cp_declarator *declarator;
enum tree_code code;
cp_cv_quals cv_quals;
tree class_type;
tree gnu_attributes = NULL_TREE, std_attributes = NULL_TREE;
if (ctor_dtor_or_conv_p)
*ctor_dtor_or_conv_p = 0;
if (cp_parser_allow_gnu_extensions_p (parser))
gnu_attributes = cp_parser_gnu_attributes_opt (parser);
cp_parser_parse_tentatively (parser);
code = cp_parser_ptr_operator (parser,
&class_type,
&cv_quals,
&std_attributes);
if (cp_parser_parse_definitely (parser))
{
if (parenthesized_p)
*parenthesized_p = true;
if (dcl_kind != CP_PARSER_DECLARATOR_NAMED)
cp_parser_parse_tentatively (parser);
declarator = cp_parser_declarator (parser, dcl_kind,
NULL,
NULL,
false,
friend_p);
if (dcl_kind != CP_PARSER_DECLARATOR_NAMED
&& !cp_parser_parse_definitely (parser))
declarator = NULL;
declarator = cp_parser_make_indirect_declarator
(code, class_type, cv_quals, declarator, std_attributes);
}
else
{
if (parenthesized_p)
*parenthesized_p = cp_lexer_next_token_is (parser->lexer,
CPP_OPEN_PAREN);
declarator = cp_parser_direct_declarator (parser, dcl_kind,
ctor_dtor_or_conv_p,
member_p, friend_p);
}
if (gnu_attributes && declarator && declarator != cp_error_declarator)
declarator->attributes = gnu_attributes;
return declarator;
}
static cp_declarator *
cp_parser_direct_declarator (cp_parser* parser,
cp_parser_declarator_kind dcl_kind,
int* ctor_dtor_or_conv_p,
bool member_p, bool friend_p)
{
cp_token *token;
cp_declarator *declarator = NULL;
tree scope = NULL_TREE;
bool saved_default_arg_ok_p = parser->default_arg_ok_p;
bool saved_in_declarator_p = parser->in_declarator_p;
bool first = true;
tree pushed_scope = NULL_TREE;
cp_token *open_paren = NULL, *close_paren = NULL;
while (true)
{
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_OPEN_PAREN)
{
if (!first || dcl_kind != CP_PARSER_DECLARATOR_NAMED)
{
tree params;
bool is_declarator = false;
open_paren = NULL;
if (!member_p)
cp_parser_parse_tentatively (parser);
matching_parens parens;
parens.consume_open (parser);
if (first)
{
parser->default_arg_ok_p = false;
parser->in_declarator_p = true;
}
begin_scope (sk_function_parms, NULL_TREE);
params = cp_parser_parameter_declaration_clause (parser);
parens.require_close (parser);
if (member_p || cp_parser_parse_definitely (parser))
{
cp_cv_quals cv_quals;
cp_virt_specifiers virt_specifiers;
cp_ref_qualifier ref_qual;
tree exception_specification;
tree late_return;
tree attrs;
bool memfn = (member_p || (pushed_scope
&& CLASS_TYPE_P (pushed_scope)));
is_declarator = true;
if (ctor_dtor_or_conv_p)
*ctor_dtor_or_conv_p = *ctor_dtor_or_conv_p < 0;
first = false;
cv_quals = cp_parser_cv_qualifier_seq_opt (parser);
ref_qual = cp_parser_ref_qualifier_opt (parser);
tree tx_qual = cp_parser_tx_qualifier_opt (parser);
exception_specification
= cp_parser_exception_specification_opt (parser);
attrs = cp_parser_std_attribute_spec_seq (parser);
tree gnu_attrs = NULL_TREE;
tree requires_clause = NULL_TREE;
late_return = (cp_parser_late_return_type_opt
(parser, declarator, requires_clause,
memfn ? cv_quals : -1));
virt_specifiers = cp_parser_virt_specifier_seq_opt (parser);
declarator = make_call_declarator (declarator,
params,
cv_quals,
virt_specifiers,
ref_qual,
tx_qual,
exception_specification,
late_return,
requires_clause);
declarator->std_attributes = attrs;
declarator->attributes = gnu_attrs;
parser->default_arg_ok_p = false;
}
pop_bindings_and_leave_scope ();
if (is_declarator)
continue;
}
if (first)
{
bool saved_in_type_id_in_expr_p;
parser->default_arg_ok_p = saved_default_arg_ok_p;
parser->in_declarator_p = saved_in_declarator_p;
open_paren = token;
matching_parens parens;
parens.consume_open (parser);
saved_in_type_id_in_expr_p = parser->in_type_id_in_expr_p;
parser->in_type_id_in_expr_p = true;
declarator
= cp_parser_declarator (parser, dcl_kind, ctor_dtor_or_conv_p,
NULL,
member_p, friend_p);
parser->in_type_id_in_expr_p = saved_in_type_id_in_expr_p;
first = false;
close_paren = cp_lexer_peek_token (parser->lexer);
if (!parens.require_close (parser))
declarator = cp_error_declarator;
if (declarator == cp_error_declarator)
break;
goto handle_declarator;
}
else
break;
}
else if ((!first || dcl_kind != CP_PARSER_DECLARATOR_NAMED)
&& token->type == CPP_OPEN_SQUARE
&& !cp_next_tokens_can_be_attribute_p (parser))
{
tree bounds, attrs;
if (ctor_dtor_or_conv_p)
*ctor_dtor_or_conv_p = 0;
open_paren = NULL;
first = false;
parser->default_arg_ok_p = false;
parser->in_declarator_p = true;
cp_lexer_consume_token (parser->lexer);
token = cp_lexer_peek_token (parser->lexer);
if (token->type != CPP_CLOSE_SQUARE)
{
bool non_constant_p;
bounds
= cp_parser_constant_expression (parser,
true,
&non_constant_p);
if (!non_constant_p)
;
else if (error_operand_p (bounds))
;
else if (!parser->in_function_body
|| current_binding_level->kind == sk_function_parms)
{
cp_parser_error (parser,
"array bound is not an integer constant");
bounds = error_mark_node;
}
else if (processing_template_decl
&& !type_dependent_expression_p (bounds))
{
bounds = build_nop (TREE_TYPE (bounds), bounds);
TREE_SIDE_EFFECTS (bounds) = 1;
}
}
else
bounds = NULL_TREE;
if (!cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE))
{
declarator = cp_error_declarator;
break;
}
attrs = cp_parser_std_attribute_spec_seq (parser);
declarator = make_array_declarator (declarator, bounds);
declarator->std_attributes = attrs;
}
else if (first && dcl_kind != CP_PARSER_DECLARATOR_ABSTRACT)
{
{
tree qualifying_scope;
tree unqualified_name;
tree attrs;
special_function_kind sfk;
bool abstract_ok;
bool pack_expansion_p = false;
cp_token *declarator_id_start_token;
abstract_ok = (dcl_kind == CP_PARSER_DECLARATOR_EITHER);
if (abstract_ok)
{
cp_parser_parse_tentatively (parser);
if (token->type == CPP_ELLIPSIS)
{
cp_lexer_consume_token (parser->lexer);
pack_expansion_p = true;
}
}
declarator_id_start_token = cp_lexer_peek_token (parser->lexer);
unqualified_name
= cp_parser_declarator_id (parser, abstract_ok);
qualifying_scope = parser->scope;
if (abstract_ok)
{
bool okay = false;
if (!unqualified_name && pack_expansion_p)
{
okay = !cp_parser_error_occurred (parser);
cp_parser_abort_tentative_parse (parser);
}
else
okay = cp_parser_parse_definitely (parser);
if (!okay)
unqualified_name = error_mark_node;
else if (unqualified_name
&& (qualifying_scope
|| (!identifier_p (unqualified_name))))
{
cp_parser_error (parser, "expected unqualified-id");
unqualified_name = error_mark_node;
}
}
if (!unqualified_name)
return NULL;
if (unqualified_name == error_mark_node)
{
declarator = cp_error_declarator;
pack_expansion_p = false;
declarator->parameter_pack_p = false;
break;
}
attrs = cp_parser_std_attribute_spec_seq (parser);
if (qualifying_scope && at_namespace_scope_p ()
&& TREE_CODE (qualifying_scope) == TYPENAME_TYPE)
{
tree type;
type = resolve_typename_type (qualifying_scope,
false);
if (TREE_CODE (type) == TYPENAME_TYPE)
{
if (typedef_variant_p (type))
error_at (declarator_id_start_token->location,
"cannot define member of dependent typedef "
"%qT", type);
else
error_at (declarator_id_start_token->location,
"%<%T::%E%> is not a type",
TYPE_CONTEXT (qualifying_scope),
TYPE_IDENTIFIER (qualifying_scope));
}
qualifying_scope = type;
}
sfk = sfk_none;
if (unqualified_name)
{
tree class_type;
if (qualifying_scope
&& CLASS_TYPE_P (qualifying_scope))
class_type = qualifying_scope;
else
class_type = current_class_type;
if (TREE_CODE (unqualified_name) == TYPE_DECL)
{
tree name_type = TREE_TYPE (unqualified_name);
if (!class_type || !same_type_p (name_type, class_type))
{
cp_parser_error (parser, "invalid declarator");
declarator = cp_error_declarator;
break;
}
else if (qualifying_scope
&& CLASSTYPE_USE_TEMPLATE (name_type))
{
error_at (declarator_id_start_token->location,
"invalid use of constructor as a template");
inform (declarator_id_start_token->location,
"use %<%T::%D%> instead of %<%T::%D%> to "
"name the constructor in a qualified name",
class_type,
DECL_NAME (TYPE_TI_TEMPLATE (class_type)),
class_type, name_type);
declarator = cp_error_declarator;
break;
}
unqualified_name = constructor_name (class_type);
}
if (class_type)
{
if (TREE_CODE (unqualified_name) == BIT_NOT_EXPR)
sfk = sfk_destructor;
else if (identifier_p (unqualified_name)
&& IDENTIFIER_CONV_OP_P (unqualified_name))
sfk = sfk_conversion;
else if (
!TYPE_WAS_UNNAMED (class_type)
&& (!friend_p || class_type == qualifying_scope)
&& constructor_name_p (unqualified_name,
class_type))
sfk = sfk_constructor;
else if (is_overloaded_fn (unqualified_name)
&& DECL_CONSTRUCTOR_P (get_first_fn
(unqualified_name)))
sfk = sfk_constructor;
if (ctor_dtor_or_conv_p && sfk != sfk_none)
*ctor_dtor_or_conv_p = -1;
}
}
declarator = make_id_declarator (qualifying_scope,
unqualified_name,
sfk);
declarator->std_attributes = attrs;
declarator->id_loc = token->location;
declarator->parameter_pack_p = pack_expansion_p;
if (pack_expansion_p)
maybe_warn_variadic_templates ();
}
handle_declarator:;
scope = get_scope_of_declarator (declarator);
if (scope)
{
if (at_function_scope_p ())
{
cp_parser_error (parser, "qualified-id in declaration");
declarator = cp_error_declarator;
break;
}
pushed_scope = push_scope (scope);
}
parser->in_declarator_p = true;
if ((ctor_dtor_or_conv_p && *ctor_dtor_or_conv_p)
|| (declarator && declarator->kind == cdk_id))
parser->default_arg_ok_p = saved_default_arg_ok_p;
else
parser->default_arg_ok_p = false;
first = false;
}
else
break;
}
if (!declarator)
cp_parser_error (parser, "expected declarator");
else if (open_paren)
{
if (declarator->kind == cdk_array)
{
expanded_location open = expand_location (open_paren->location);
expanded_location close = expand_location (close_paren->location);
if (open.line != close.line || open.file != close.file)
open_paren = NULL;
}
if (open_paren)
declarator->parenthesized = open_paren->location;
}
if (pushed_scope)
pop_scope (pushed_scope);
parser->default_arg_ok_p = saved_default_arg_ok_p;
parser->in_declarator_p = saved_in_declarator_p;
return declarator;
}
static enum tree_code
cp_parser_ptr_operator (cp_parser* parser,
tree* type,
cp_cv_quals *cv_quals,
tree *attributes)
{
enum tree_code code = ERROR_MARK;
cp_token *token;
tree attrs = NULL_TREE;
*type = NULL_TREE;
*cv_quals = TYPE_UNQUALIFIED;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_MULT)
code = INDIRECT_REF;
else if (token->type == CPP_AND)
code = ADDR_EXPR;
else if ((cxx_dialect != cxx98) &&
token->type == CPP_AND_AND) 
code = NON_LVALUE_EXPR;
if (code != ERROR_MARK)
{
cp_lexer_consume_token (parser->lexer);
if (code == INDIRECT_REF
|| cp_parser_allow_gnu_extensions_p (parser))
*cv_quals = cp_parser_cv_qualifier_seq_opt (parser);
attrs = cp_parser_std_attribute_spec_seq (parser);
if (attributes != NULL)
*attributes = attrs;
}
else
{
cp_parser_parse_tentatively (parser);
cp_parser_global_scope_opt (parser,
false);
token = cp_lexer_peek_token (parser->lexer);
cp_parser_nested_name_specifier (parser,
false,
true,
false,
false);
if (!cp_parser_error_occurred (parser)
&& cp_parser_require (parser, CPP_MULT, RT_MULT))
{
code = INDIRECT_REF;
if (TREE_CODE (parser->scope) == NAMESPACE_DECL)
error_at (token->location, "%qD is a namespace", parser->scope);
else if (TREE_CODE (parser->scope) == ENUMERAL_TYPE)
error_at (token->location, "cannot form pointer to member of "
"non-class %q#T", parser->scope);
else
{
*type = parser->scope;
parser->scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
parser->object_scope = NULL_TREE;
attrs = cp_parser_std_attribute_spec_seq (parser);
if (attributes != NULL)
*attributes = attrs;
*cv_quals = cp_parser_cv_qualifier_seq_opt (parser);
}
}
if (!cp_parser_parse_definitely (parser))
cp_parser_error (parser, "expected ptr-operator");
}
return code;
}
static cp_cv_quals
cp_parser_cv_qualifier_seq_opt (cp_parser* parser)
{
cp_cv_quals cv_quals = TYPE_UNQUALIFIED;
while (true)
{
cp_token *token;
cp_cv_quals cv_qualifier;
token = cp_lexer_peek_token (parser->lexer);
switch (token->keyword)
{
case RID_CONST:
cv_qualifier = TYPE_QUAL_CONST;
break;
case RID_VOLATILE:
cv_qualifier = TYPE_QUAL_VOLATILE;
break;
case RID_RESTRICT:
cv_qualifier = TYPE_QUAL_RESTRICT;
break;
default:
cv_qualifier = TYPE_UNQUALIFIED;
break;
}
if (!cv_qualifier)
break;
if (cv_quals & cv_qualifier)
{
gcc_rich_location richloc (token->location);
richloc.add_fixit_remove ();
error_at (&richloc, "duplicate cv-qualifier");
cp_lexer_purge_token (parser->lexer);
}
else
{
cp_lexer_consume_token (parser->lexer);
cv_quals |= cv_qualifier;
}
}
return cv_quals;
}
static cp_ref_qualifier
cp_parser_ref_qualifier_opt (cp_parser* parser)
{
cp_ref_qualifier ref_qual = REF_QUAL_NONE;
if (cxx_dialect < cxx11 && cp_parser_parsing_tentatively (parser))
return ref_qual;
while (true)
{
cp_ref_qualifier curr_ref_qual = REF_QUAL_NONE;
cp_token *token = cp_lexer_peek_token (parser->lexer);
switch (token->type)
{
case CPP_AND:
curr_ref_qual = REF_QUAL_LVALUE;
break;
case CPP_AND_AND:
curr_ref_qual = REF_QUAL_RVALUE;
break;
default:
curr_ref_qual = REF_QUAL_NONE;
break;
}
if (!curr_ref_qual)
break;
else if (ref_qual)
{
error_at (token->location, "multiple ref-qualifiers");
cp_lexer_purge_token (parser->lexer);
}
else
{
ref_qual = curr_ref_qual;
cp_lexer_consume_token (parser->lexer);
}
}
return ref_qual;
}
static tree
cp_parser_tx_qualifier_opt (cp_parser *parser)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_NAME)
{
tree name = token->u.value;
const char *p = IDENTIFIER_POINTER (name);
const int len = strlen ("transaction_safe");
if (!strncmp (p, "transaction_safe", len))
{
p += len;
if (*p == '\0'
|| !strcmp (p, "_dynamic"))
{
cp_lexer_consume_token (parser->lexer);
if (!flag_tm)
{
error ("%qE requires %<-fgnu-tm%>", name);
return NULL_TREE;
}
else
return name;
}
}
}
return NULL_TREE;
}
static cp_virt_specifiers
cp_parser_virt_specifier_seq_opt (cp_parser* parser)
{
cp_virt_specifiers virt_specifiers = VIRT_SPEC_UNSPECIFIED;
while (true)
{
cp_token *token;
cp_virt_specifiers virt_specifier;
token = cp_lexer_peek_token (parser->lexer);
if (token->type != CPP_NAME)
break;
if (id_equal (token->u.value, "override"))
{
maybe_warn_cpp0x (CPP0X_OVERRIDE_CONTROLS);
virt_specifier = VIRT_SPEC_OVERRIDE;
}
else if (id_equal (token->u.value, "final"))
{
maybe_warn_cpp0x (CPP0X_OVERRIDE_CONTROLS);
virt_specifier = VIRT_SPEC_FINAL;
}
else if (id_equal (token->u.value, "__final"))
{
virt_specifier = VIRT_SPEC_FINAL;
}
else
break;
if (virt_specifiers & virt_specifier)
{
gcc_rich_location richloc (token->location);
richloc.add_fixit_remove ();
error_at (&richloc, "duplicate virt-specifier");
cp_lexer_purge_token (parser->lexer);
}
else
{
cp_lexer_consume_token (parser->lexer);
virt_specifiers |= virt_specifier;
}
}
return virt_specifiers;
}
void
inject_this_parameter (tree ctype, cp_cv_quals quals)
{
tree this_parm;
if (current_class_ptr)
{
tree type = TREE_TYPE (TREE_TYPE (current_class_ptr));
if (DECL_P (current_class_ptr)
&& DECL_CONTEXT (current_class_ptr) == NULL_TREE
&& same_type_ignoring_top_level_qualifiers_p (ctype, type)
&& cp_type_quals (type) == quals)
return;
}
this_parm = build_this_parm (NULL_TREE, ctype, quals);
current_class_ptr = NULL_TREE;
current_class_ref
= cp_build_fold_indirect_ref (this_parm);
current_class_ptr = this_parm;
}
bool
parsing_nsdmi (void)
{
if (current_class_ptr
&& TREE_CODE (current_class_ptr) == PARM_DECL
&& DECL_CONTEXT (current_class_ptr) == NULL_TREE)
return true;
return false;
}
static tree
cp_parser_late_return_type_opt (cp_parser* parser, cp_declarator *declarator,
tree& requires_clause, cp_cv_quals quals)
{
cp_token *token;
tree type = NULL_TREE;
bool declare_simd_p = (parser->omp_declare_simd
&& declarator
&& declarator->kind == cdk_id);
bool oacc_routine_p = (parser->oacc_routine
&& declarator
&& declarator->kind == cdk_id);
token = cp_lexer_peek_token (parser->lexer);
if (token->type != CPP_DEREF
&& token->keyword != RID_REQUIRES
&& !(token->type == CPP_NAME
&& token->u.value == ridpointers[RID_REQUIRES])
&& !(declare_simd_p || oacc_routine_p))
return NULL_TREE;
tree save_ccp = current_class_ptr;
tree save_ccr = current_class_ref;
if (quals >= 0)
{
inject_this_parameter (current_class_type, quals);
}
if (token->type == CPP_DEREF)
{
cp_lexer_consume_token (parser->lexer);
type = cp_parser_trailing_type_id (parser);
}
requires_clause = cp_parser_requires_clause_opt (parser);
if (declare_simd_p)
declarator->attributes
= cp_parser_late_parsing_omp_declare_simd (parser,
declarator->attributes);
if (oacc_routine_p)
declarator->attributes
= cp_parser_late_parsing_oacc_routine (parser,
declarator->attributes);
if (quals >= 0)
{
current_class_ptr = save_ccp;
current_class_ref = save_ccr;
}
return type;
}
static tree
cp_parser_declarator_id (cp_parser* parser, bool optional_p)
{
tree id;
id = cp_parser_id_expression (parser,
false,
false,
NULL,
true,
optional_p);
if (id && BASELINK_P (id))
id = BASELINK_FUNCTIONS (id);
return id;
}
static tree
cp_parser_type_id_1 (cp_parser* parser, bool is_template_arg,
bool is_trailing_return)
{
cp_decl_specifier_seq type_specifier_seq;
cp_declarator *abstract_declarator;
cp_parser_type_specifier_seq (parser, false,
is_trailing_return,
&type_specifier_seq);
if (is_template_arg && type_specifier_seq.type
&& TREE_CODE (type_specifier_seq.type) == TEMPLATE_TYPE_PARM
&& CLASS_PLACEHOLDER_TEMPLATE (type_specifier_seq.type))
{
gcc_assert (cp_parser_uncommitted_to_tentative_parse_p (parser));
cp_parser_simulate_error (parser);
return error_mark_node;
}
if (type_specifier_seq.type == error_mark_node)
return error_mark_node;
cp_parser_parse_tentatively (parser);
abstract_declarator
= cp_parser_declarator (parser, CP_PARSER_DECLARATOR_ABSTRACT, NULL,
NULL,
false,
false);
if (!cp_parser_parse_definitely (parser))
abstract_declarator = NULL;
if (type_specifier_seq.type
&& (!flag_concepts || parser->in_type_id_in_expr_p)
&& !(cxx_dialect >= cxx14 && is_trailing_return))
if (tree auto_node = type_uses_auto (type_specifier_seq.type))
{
if (abstract_declarator
&& abstract_declarator->kind == cdk_function
&& abstract_declarator->u.function.late_return_type)
;
else if (parser->in_result_type_constraint_p)
;
else
{
location_t loc = type_specifier_seq.locations[ds_type_spec];
if (tree tmpl = CLASS_PLACEHOLDER_TEMPLATE (auto_node))
{
error_at (loc, "missing template arguments after %qT",
auto_node);
inform (DECL_SOURCE_LOCATION (tmpl), "%qD declared here",
tmpl);
}
else
error_at (loc, "invalid use of %qT", auto_node);
return error_mark_node;
}
}
return groktypename (&type_specifier_seq, abstract_declarator,
is_template_arg);
}
static tree
cp_parser_type_id (cp_parser *parser)
{
return cp_parser_type_id_1 (parser, false, false);
}
static tree
cp_parser_template_type_arg (cp_parser *parser)
{
tree r;
const char *saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in template arguments");
r = cp_parser_type_id_1 (parser, true, false);
parser->type_definition_forbidden_message = saved_message;
if (cxx_dialect >= cxx14 && !flag_concepts && type_uses_auto (r))
{
error ("invalid use of %<auto%> in template argument");
r = error_mark_node;
}
return r;
}
static tree
cp_parser_trailing_type_id (cp_parser *parser)
{
return cp_parser_type_id_1 (parser, false, true);
}
static void
cp_parser_type_specifier_seq (cp_parser* parser,
bool is_declaration,
bool is_trailing_return,
cp_decl_specifier_seq *type_specifier_seq)
{
bool seen_type_specifier = false;
cp_parser_flags flags = CP_PARSER_FLAGS_OPTIONAL;
cp_token *start_token = NULL;
clear_decl_specs (type_specifier_seq);
if (is_trailing_return)
flags |= CP_PARSER_FLAGS_NO_TYPE_DEFINITIONS;
while (true)
{
tree type_specifier;
bool is_cv_qualifier;
if (cp_next_tokens_can_be_attribute_p (parser))
{
type_specifier_seq->attributes
= attr_chainon (type_specifier_seq->attributes,
cp_parser_attributes_opt (parser));
continue;
}
if (!start_token)
start_token = cp_lexer_peek_token (parser->lexer);
type_specifier = cp_parser_type_specifier (parser,
flags,
type_specifier_seq,
false,
NULL,
&is_cv_qualifier);
if (!type_specifier)
{
if (!seen_type_specifier)
{
int in_decl = parser->in_declarator_p;
parser->in_declarator_p = true;
if (cp_parser_uncommitted_to_tentative_parse_p (parser)
|| !cp_parser_parse_and_diagnose_invalid_type_name (parser))
cp_parser_error (parser, "expected type-specifier");
parser->in_declarator_p = in_decl;
type_specifier_seq->type = error_mark_node;
return;
}
break;
}
seen_type_specifier = true;
if (is_declaration && !is_cv_qualifier)
flags |= CP_PARSER_FLAGS_NO_USER_DEFINED_TYPES;
}
}
static bool
function_being_declared_is_template_p (cp_parser* parser)
{
if (!current_template_parms || processing_template_parmlist)
return false;
if (parser->implicit_template_scope)
return true;
if (at_class_scope_p ()
&& TYPE_BEING_DEFINED (current_class_type))
return parser->num_template_parameter_lists != 0;
return ((int) parser->num_template_parameter_lists > template_class_depth
(current_class_type));
}
static tree
cp_parser_parameter_declaration_clause (cp_parser* parser)
{
tree parameters;
cp_token *token;
bool ellipsis_p;
bool is_error;
temp_override<bool> cleanup
(parser->auto_is_implicit_function_template_parm_p);
if (!processing_specialization
&& !processing_template_parmlist
&& !processing_explicit_instantiation
&& parser->default_arg_ok_p)
if (!current_function_decl
|| (current_class_type && LAMBDA_TYPE_P (current_class_type)))
parser->auto_is_implicit_function_template_parm_p = true;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_ELLIPSIS)
{
cp_lexer_consume_token (parser->lexer);
return NULL_TREE;
}
else if (token->type == CPP_CLOSE_PAREN)
{
#ifndef NO_IMPLICIT_EXTERN_C
if (in_system_header_at (input_location)
&& current_class_type == NULL
&& current_lang_name == lang_name_c)
return NULL_TREE;
else
#endif
return void_list_node;
}
else if (token->keyword == RID_VOID
&& (cp_lexer_peek_nth_token (parser->lexer, 2)->type
== CPP_CLOSE_PAREN))
{
cp_lexer_consume_token (parser->lexer);
return void_list_node;
}
parameters = cp_parser_parameter_declaration_list (parser, &is_error);
if (is_error)
return NULL;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_COMMA)
{
cp_lexer_consume_token (parser->lexer);
ellipsis_p
= (cp_parser_require (parser, CPP_ELLIPSIS, RT_ELLIPSIS) != NULL);
}
else if (token->type == CPP_ELLIPSIS)
{
cp_lexer_consume_token (parser->lexer);
ellipsis_p = true;
}
else
ellipsis_p = false;
if (!ellipsis_p)
parameters = chainon (parameters, void_list_node);
return parameters;
}
static tree
cp_parser_parameter_declaration_list (cp_parser* parser, bool *is_error)
{
tree parameters = NULL_TREE;
tree *tail = &parameters;
bool saved_in_unbraced_linkage_specification_p;
int index = 0;
*is_error = false;
saved_in_unbraced_linkage_specification_p
= parser->in_unbraced_linkage_specification_p;
parser->in_unbraced_linkage_specification_p = false;
while (true)
{
cp_parameter_declarator *parameter;
tree decl = error_mark_node;
bool parenthesized_p = false;
parameter
= cp_parser_parameter_declaration (parser,
false,
&parenthesized_p);
deprecated_state = DEPRECATED_SUPPRESS;
if (parameter)
{
decl = grokdeclarator (parameter->declarator,
&parameter->decl_specifiers,
PARM,
parameter->default_argument != NULL_TREE,
&parameter->decl_specifiers.attributes);
if (decl != error_mark_node && parameter->loc != UNKNOWN_LOCATION)
DECL_SOURCE_LOCATION (decl) = parameter->loc;
}
deprecated_state = DEPRECATED_NORMAL;
if (decl == error_mark_node)
{
*is_error = true;
parameters = error_mark_node;
break;
}
if (parameter->decl_specifiers.attributes)
cplus_decl_attributes (&decl,
parameter->decl_specifiers.attributes,
0);
if (DECL_NAME (decl))
decl = pushdecl (decl);
if (decl != error_mark_node)
{
retrofit_lang_decl (decl);
DECL_PARM_INDEX (decl) = ++index;
DECL_PARM_LEVEL (decl) = function_parm_depth ();
}
*tail = build_tree_list (parameter->default_argument, decl);
tail = &TREE_CHAIN (*tail);
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_PAREN)
|| cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS)
|| cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON)
|| cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
break;
else if (cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
{
cp_token *token;
token = cp_lexer_peek_nth_token (parser->lexer, 2);
if (token->type == CPP_ELLIPSIS)
break;
cp_lexer_consume_token (parser->lexer);
if (!parser->in_template_argument_list_p
&& !parser->in_type_id_in_expr_p
&& cp_parser_uncommitted_to_tentative_parse_p (parser)
&& !parenthesized_p)
cp_parser_commit_to_tentative_parse (parser);
}
else
{
cp_parser_error (parser, "expected %<,%> or %<...%>");
if (!cp_parser_uncommitted_to_tentative_parse_p (parser))
cp_parser_skip_to_closing_parenthesis (parser,
true,
false,
false);
break;
}
}
parser->in_unbraced_linkage_specification_p
= saved_in_unbraced_linkage_specification_p;
if (cp_binding_level *its = parser->implicit_template_scope)
if (cp_binding_level *maybe_its = current_binding_level->level_chain)
{
while (maybe_its->kind == sk_class)
maybe_its = maybe_its->level_chain;
if (maybe_its == its)
{
parser->implicit_template_parms = 0;
parser->implicit_template_scope = 0;
}
}
return parameters;
}
static cp_parameter_declarator *
cp_parser_parameter_declaration (cp_parser *parser,
bool template_parm_p,
bool *parenthesized_p)
{
int declares_class_or_enum;
cp_decl_specifier_seq decl_specifiers;
cp_declarator *declarator;
tree default_argument;
cp_token *token = NULL, *declarator_token_start = NULL;
const char *saved_message;
bool template_parameter_pack_p = false;
saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in parameter types");
int template_parm_idx = (function_being_declared_is_template_p (parser) ?
TREE_VEC_LENGTH (INNERMOST_TEMPLATE_PARMS
(current_template_parms)) : 0);
cp_token *decl_spec_token_start = cp_lexer_peek_token (parser->lexer);
cp_parser_decl_specifier_seq (parser,
CP_PARSER_FLAGS_NONE,
&decl_specifiers,
&declares_class_or_enum);
if (!decl_specifiers.any_type_specifiers_p
&& cp_parser_parse_and_diagnose_invalid_type_name (parser))
decl_specifiers.type = error_mark_node;
if (cp_parser_error_occurred (parser))
{
parser->type_definition_forbidden_message = saved_message;
return NULL;
}
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_CLOSE_PAREN
|| token->type == CPP_COMMA
|| token->type == CPP_EQ
|| token->type == CPP_GREATER)
{
declarator = NULL;
if (parenthesized_p)
*parenthesized_p = false;
}
else
{
bool saved_default_arg_ok_p = parser->default_arg_ok_p;
parser->default_arg_ok_p = false;
if (!parser->in_template_argument_list_p
&& !parser->in_type_id_in_expr_p
&& cp_parser_uncommitted_to_tentative_parse_p (parser)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_OPEN_BRACE)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_OPEN_PAREN))
cp_parser_commit_to_tentative_parse (parser);
declarator_token_start = token;
declarator = cp_parser_declarator (parser,
CP_PARSER_DECLARATOR_EITHER,
NULL,
parenthesized_p,
false,
false);
parser->default_arg_ok_p = saved_default_arg_ok_p;
decl_specifiers.attributes
= attr_chainon (decl_specifiers.attributes,
cp_parser_attributes_opt (parser));
if (template_parm_p && declarator && declarator->parameter_pack_p)
{
declarator->parameter_pack_p = false;
template_parameter_pack_p = true;
}
}
token = cp_lexer_peek_token (parser->lexer);
if (parser->implicit_template_parms
&& ((token->type == CPP_ELLIPSIS
&& declarator_can_be_parameter_pack (declarator))
|| (declarator && declarator->parameter_pack_p)))
{
int latest_template_parm_idx = TREE_VEC_LENGTH
(INNERMOST_TEMPLATE_PARMS (current_template_parms));
if (latest_template_parm_idx != template_parm_idx)
decl_specifiers.type = convert_generic_types_to_packs
(decl_specifiers.type,
template_parm_idx, latest_template_parm_idx);
}
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
tree type = decl_specifiers.type;
if (type && DECL_P (type))
type = TREE_TYPE (type);
if (((type
&& TREE_CODE (type) != TYPE_PACK_EXPANSION
&& (template_parm_p || uses_parameter_packs (type)))
|| (!type && template_parm_p))
&& declarator_can_be_parameter_pack (declarator))
{
cp_lexer_consume_token (parser->lexer);
maybe_warn_variadic_templates ();
if (template_parm_p)
template_parameter_pack_p = true;
else if (declarator)
declarator->parameter_pack_p = true;
else
decl_specifiers.type = make_pack_expansion (type);
}
}
parser->type_definition_forbidden_message = saved_message;
if (cp_lexer_next_token_is (parser->lexer, CPP_EQ))
{
tree type = decl_specifiers.type;
token = cp_lexer_peek_token (parser->lexer);
if (!template_parm_p && at_class_scope_p ()
&& TYPE_BEING_DEFINED (current_class_type)
&& !LAMBDA_TYPE_P (current_class_type))
default_argument = cp_parser_cache_defarg (parser, false);
else if (declares_constrained_type_template_parameter (type))
default_argument
= cp_parser_default_type_template_argument (parser);
else if (declares_constrained_template_template_parameter (type))
default_argument
= cp_parser_default_template_template_argument (parser);
else
default_argument
= cp_parser_default_argument (parser, template_parm_p);
if (!parser->default_arg_ok_p)
{
permerror (token->location,
"default arguments are only "
"permitted for function parameters");
}
else if ((declarator && declarator->parameter_pack_p)
|| template_parameter_pack_p
|| (decl_specifiers.type
&& PACK_EXPANSION_P (decl_specifiers.type)))
{
cp_declarator *id_declarator = declarator;
while (id_declarator && id_declarator->kind != cdk_id)
id_declarator = id_declarator->declarator;
if (id_declarator && id_declarator->kind == cdk_id)
error_at (declarator_token_start->location,
template_parm_p
? G_("template parameter pack %qD "
"cannot have a default argument")
: G_("parameter pack %qD cannot have "
"a default argument"),
id_declarator->u.id.unqualified_name);
else
error_at (declarator_token_start->location,
template_parm_p
? G_("template parameter pack cannot have "
"a default argument")
: G_("parameter pack cannot have a "
"default argument"));
default_argument = NULL_TREE;
}
}
else
default_argument = NULL_TREE;
location_t caret_loc = (declarator && declarator->id_loc != UNKNOWN_LOCATION
? declarator->id_loc
: decl_spec_token_start->location);
location_t param_loc = make_location (caret_loc,
decl_spec_token_start->location,
input_location);
return make_parameter_declarator (&decl_specifiers,
declarator,
default_argument,
param_loc,
template_parameter_pack_p);
}
static tree
cp_parser_default_argument (cp_parser *parser, bool template_parm_p)
{
tree default_argument = NULL_TREE;
bool saved_greater_than_is_operator_p;
bool saved_local_variables_forbidden_p;
bool non_constant_p, is_direct_init;
saved_greater_than_is_operator_p = parser->greater_than_is_operator_p;
parser->greater_than_is_operator_p = !template_parm_p;
saved_local_variables_forbidden_p = parser->local_variables_forbidden_p;
parser->local_variables_forbidden_p = true;
if (template_parm_p)
push_deferring_access_checks (dk_no_deferred);
tree saved_class_ptr = NULL_TREE;
tree saved_class_ref = NULL_TREE;
if (cfun)
{
saved_class_ptr = current_class_ptr;
cp_function_chain->x_current_class_ptr = NULL_TREE;
saved_class_ref = current_class_ref;
cp_function_chain->x_current_class_ref = NULL_TREE;
}
default_argument
= cp_parser_initializer (parser, &is_direct_init, &non_constant_p);
if (cfun)
{
cp_function_chain->x_current_class_ptr = saved_class_ptr;
cp_function_chain->x_current_class_ref = saved_class_ref;
}
if (BRACE_ENCLOSED_INITIALIZER_P (default_argument))
maybe_warn_cpp0x (CPP0X_INITIALIZER_LISTS);
if (template_parm_p)
pop_deferring_access_checks ();
parser->greater_than_is_operator_p = saved_greater_than_is_operator_p;
parser->local_variables_forbidden_p = saved_local_variables_forbidden_p;
return default_argument;
}
static void
cp_parser_function_body (cp_parser *parser, bool in_function_try_block)
{
cp_parser_compound_statement (parser, NULL, (in_function_try_block
? BCS_TRY_BLOCK : BCS_NORMAL),
true);
}
static void
cp_parser_ctor_initializer_opt_and_function_body (cp_parser *parser,
bool in_function_try_block)
{
tree body, list;
const bool check_body_p =
DECL_CONSTRUCTOR_P (current_function_decl)
&& DECL_DECLARED_CONSTEXPR_P (current_function_decl);
tree last = NULL;
body = begin_function_body ();
cp_parser_ctor_initializer_opt (parser);
if (check_body_p)
{
list = cur_stmt_list;
if (STATEMENT_LIST_TAIL (list))
last = STATEMENT_LIST_TAIL (list)->stmt;
}
cp_parser_function_body (parser, in_function_try_block);
if (check_body_p)
check_constexpr_ctor_body (last, list, true);
finish_function_body (body);
}
static tree
cp_parser_initializer (cp_parser* parser, bool* is_direct_init,
bool* non_constant_p, bool subexpression_p)
{
cp_token *token;
tree init;
token = cp_lexer_peek_token (parser->lexer);
*is_direct_init = (token->type != CPP_EQ);
*non_constant_p = false;
if (token->type == CPP_EQ)
{
cp_lexer_consume_token (parser->lexer);
init = cp_parser_initializer_clause (parser, non_constant_p);
}
else if (token->type == CPP_OPEN_PAREN)
{
vec<tree, va_gc> *vec;
vec = cp_parser_parenthesized_expression_list (parser, non_attr,
false,
true,
non_constant_p);
if (vec == NULL)
return error_mark_node;
init = build_tree_list_vec (vec);
release_tree_vector (vec);
}
else if (token->type == CPP_OPEN_BRACE)
{
cp_lexer_set_source_position (parser->lexer);
maybe_warn_cpp0x (CPP0X_INITIALIZER_LISTS);
init = cp_parser_braced_list (parser, non_constant_p);
CONSTRUCTOR_IS_DIRECT_INIT (init) = 1;
}
else
{
cp_parser_error (parser, "expected initializer");
init = error_mark_node;
}
if (!subexpression_p && check_for_bare_parameter_packs (init))
init = error_mark_node;
return init;
}
static cp_expr
cp_parser_initializer_clause (cp_parser* parser, bool* non_constant_p)
{
cp_expr initializer;
*non_constant_p = false;
if (cp_lexer_next_token_is_not (parser->lexer, CPP_OPEN_BRACE))
{
initializer
= cp_parser_constant_expression (parser,
true,
non_constant_p);
}
else
initializer = cp_parser_braced_list (parser, non_constant_p);
return initializer;
}
static cp_expr
cp_parser_braced_list (cp_parser* parser, bool* non_constant_p)
{
tree initializer;
location_t start_loc = cp_lexer_peek_token (parser->lexer)->location;
matching_braces braces;
braces.require_open (parser);
initializer = make_node (CONSTRUCTOR);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_CLOSE_BRACE))
{
CONSTRUCTOR_ELTS (initializer)
= cp_parser_initializer_list (parser, non_constant_p);
if (cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
cp_lexer_consume_token (parser->lexer);
}
else
*non_constant_p = false;
location_t finish_loc = cp_lexer_peek_token (parser->lexer)->location;
braces.require_close (parser);
TREE_TYPE (initializer) = init_list_type_node;
cp_expr result (initializer);
location_t combined_loc = make_location (start_loc, start_loc, finish_loc);
result.set_location (combined_loc);
return result;
}
static bool
cp_parser_skip_to_closing_square_bracket (cp_parser *parser)
{
unsigned square_depth = 0;
while (true)
{
cp_token * token = cp_lexer_peek_token (parser->lexer);
switch (token->type)
{
case CPP_EOF:
case CPP_PRAGMA_EOL:
return false;
case CPP_OPEN_SQUARE:
++square_depth;
break;
case CPP_CLOSE_SQUARE:
if (!square_depth--)
{
cp_lexer_consume_token (parser->lexer);
return true;
}
break;
default:
break;
}
cp_lexer_consume_token (parser->lexer);
}
}
static bool
cp_parser_array_designator_p (cp_parser *parser)
{
cp_lexer_consume_token (parser->lexer);
cp_lexer_save_tokens (parser->lexer);
bool array_designator_p
= (cp_parser_skip_to_closing_square_bracket (parser)
&& cp_lexer_next_token_is (parser->lexer, CPP_EQ));
cp_lexer_rollback_tokens (parser->lexer);
return array_designator_p;
}
static vec<constructor_elt, va_gc> *
cp_parser_initializer_list (cp_parser* parser, bool* non_constant_p)
{
vec<constructor_elt, va_gc> *v = NULL;
bool first_p = true;
tree first_designator = NULL_TREE;
*non_constant_p = false;
while (true)
{
cp_token *token;
tree designator;
tree initializer;
bool clause_non_constant_p;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
if ((cxx_dialect >= cxx2a
|| cp_parser_allow_gnu_extensions_p (parser))
&& cp_lexer_next_token_is (parser->lexer, CPP_DOT)
&& cp_lexer_peek_nth_token (parser->lexer, 2)->type == CPP_NAME
&& (cp_lexer_peek_nth_token (parser->lexer, 3)->type == CPP_EQ
|| (cp_lexer_peek_nth_token (parser->lexer, 3)->type
== CPP_OPEN_BRACE)))
{
if (cxx_dialect < cxx2a)
pedwarn (loc, OPT_Wpedantic,
"C++ designated initializers only available with "
"-std=c++2a or -std=gnu++2a");
cp_lexer_consume_token (parser->lexer);
designator = cp_lexer_consume_token (parser->lexer)->u.value;
if (cp_lexer_next_token_is (parser->lexer, CPP_EQ))
cp_lexer_consume_token (parser->lexer);
}
else if (cp_parser_allow_gnu_extensions_p (parser)
&& cp_lexer_next_token_is (parser->lexer, CPP_NAME)
&& (cp_lexer_peek_nth_token (parser->lexer, 2)->type
== CPP_COLON))
{
pedwarn (loc, OPT_Wpedantic,
"ISO C++ does not allow GNU designated initializers");
designator = cp_lexer_consume_token (parser->lexer)->u.value;
cp_lexer_consume_token (parser->lexer);
}
else if (cp_parser_allow_gnu_extensions_p (parser)
&& !c_dialect_objc ()
&& cp_lexer_next_token_is (parser->lexer, CPP_OPEN_SQUARE))
{
bool non_const = false;
cp_parser_parse_tentatively (parser);
if (!cp_parser_array_designator_p (parser))
{
cp_parser_simulate_error (parser);
designator = NULL_TREE;
}
else
{
designator = cp_parser_constant_expression (parser, true,
&non_const);
cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE);
cp_parser_require (parser, CPP_EQ, RT_EQ);
}
if (!cp_parser_parse_definitely (parser))
designator = NULL_TREE;
else if (non_const
&& (!require_potential_rvalue_constant_expression
(designator)))
designator = NULL_TREE;
if (designator)
pedwarn (loc, OPT_Wpedantic,
"ISO C++ does not allow C99 designated initializers");
}
else
designator = NULL_TREE;
if (first_p)
{
first_designator = designator;
first_p = false;
}
else if (cxx_dialect >= cxx2a
&& first_designator != error_mark_node
&& (!first_designator != !designator))
{
error_at (loc, "either all initializer clauses should be designated "
"or none of them should be");
first_designator = error_mark_node;
}
else if (cxx_dialect < cxx2a && !first_designator)
first_designator = designator;
initializer = cp_parser_initializer_clause (parser,
&clause_non_constant_p);
if (clause_non_constant_p)
*non_constant_p = true;
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);
if (designator && cxx_dialect >= cxx2a)
error_at (loc,
"%<...%> not allowed in designated initializer list");
initializer = make_pack_expansion (initializer);
}
CONSTRUCTOR_APPEND_ELT (v, designator, initializer);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA))
break;
token = cp_lexer_peek_nth_token (parser->lexer, 2);
if (token->type == CPP_CLOSE_BRACE)
break;
cp_lexer_consume_token (parser->lexer);
}
if (first_designator)
{
unsigned int i;
tree designator, val;
FOR_EACH_CONSTRUCTOR_ELT (v, i, designator, val)
if (designator && TREE_CODE (designator) == IDENTIFIER_NODE)
{
if (IDENTIFIER_MARKED (designator))
{
error_at (EXPR_LOC_OR_LOC (val, input_location),
"%<.%s%> designator used multiple times in "
"the same initializer list",
IDENTIFIER_POINTER (designator));
(*v)[i].index = NULL_TREE;
}
else
IDENTIFIER_MARKED (designator) = 1;
}
FOR_EACH_CONSTRUCTOR_ELT (v, i, designator, val)
if (designator && TREE_CODE (designator) == IDENTIFIER_NODE)
IDENTIFIER_MARKED (designator) = 0;
}
return v;
}
static tree
cp_parser_class_name (cp_parser *parser,
bool typename_keyword_p,
bool template_keyword_p,
enum tag_types tag_type,
bool check_dependency_p,
bool class_head_p,
bool is_declaration,
bool enum_ok)
{
tree decl;
tree scope;
bool typename_p;
cp_token *token;
tree identifier = NULL_TREE;
token = cp_lexer_peek_token (parser->lexer);
if (token->type != CPP_NAME && token->type != CPP_TEMPLATE_ID)
{
cp_parser_error (parser, "expected class-name");
return error_mark_node;
}
scope = parser->scope;
if (scope == error_mark_node)
return error_mark_node;
typename_p = (typename_keyword_p && scope && TYPE_P (scope)
&& dependent_type_p (scope));
if (token->type == CPP_NAME
&& !cp_parser_nth_token_starts_template_argument_list_p (parser, 2))
{
cp_token *identifier_token;
bool ambiguous_p;
identifier_token = cp_lexer_peek_token (parser->lexer);
ambiguous_p = identifier_token->error_reported;
identifier = cp_parser_identifier (parser);
if (identifier == error_mark_node)
decl = error_mark_node;
else if (typename_p)
decl = identifier;
else
{
tree ambiguous_decls;
if (ambiguous_p)
{
cp_parser_simulate_error (parser);
return error_mark_node;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_SCOPE))
tag_type = scope_type;
decl = cp_parser_lookup_name (parser, identifier,
tag_type,
false,
false,
check_dependency_p,
&ambiguous_decls,
identifier_token->location);
if (ambiguous_decls)
{
if (cp_parser_parsing_tentatively (parser))
cp_parser_simulate_error (parser);
return error_mark_node;
}
}
}
else
{
decl = cp_parser_template_id (parser, template_keyword_p,
check_dependency_p,
tag_type,
is_declaration);
if (decl == error_mark_node)
return error_mark_node;
}
decl = cp_parser_maybe_treat_template_as_class (decl, class_head_p);
if (typename_p && decl != error_mark_node)
{
decl = make_typename_type (scope, decl, typename_type,
tf_error);
if (decl != error_mark_node)
decl = TYPE_NAME (decl);
}
decl = strip_using_decl (decl);
if (TREE_CODE (decl) == TEMPLATE_ID_EXPR
&& identifier_p (TREE_OPERAND (decl, 0))
&& cp_lexer_next_token_is (parser->lexer, CPP_SCOPE))
{
decl = make_typename_type (scope, decl, tag_type, tf_error);
if (decl != error_mark_node)
decl = TYPE_NAME (decl);
}
else if (TREE_CODE (decl) != TYPE_DECL
|| TREE_TYPE (decl) == error_mark_node
|| !(MAYBE_CLASS_TYPE_P (TREE_TYPE (decl))
|| (enum_ok && TREE_CODE (TREE_TYPE (decl)) == ENUMERAL_TYPE))
|| (c_dialect_objc ()
&& cp_lexer_peek_token (parser->lexer)->type == CPP_DOT 
&& objc_is_class_name (decl)))
decl = error_mark_node;
if (decl == error_mark_node)
cp_parser_error (parser, "expected class-name");
else if (identifier && !parser->scope)
maybe_note_name_used_in_class (identifier, decl);
return decl;
}
static tree
cp_parser_class_specifier_1 (cp_parser* parser)
{
tree type;
tree attributes = NULL_TREE;
bool nested_name_specifier_p;
unsigned saved_num_template_parameter_lists;
bool saved_in_function_body;
unsigned char in_statement;
bool in_switch_statement_p;
bool saved_in_unbraced_linkage_specification_p;
tree old_scope = NULL_TREE;
tree scope = NULL_TREE;
cp_token *closing_brace;
push_deferring_access_checks (dk_no_deferred);
type = cp_parser_class_head (parser,
&nested_name_specifier_p);
if (!type)
{
cp_parser_skip_to_end_of_block_or_statement (parser);
pop_deferring_access_checks ();
return error_mark_node;
}
matching_braces braces;
if (!braces.require_open (parser))
{
pop_deferring_access_checks ();
return error_mark_node;
}
cp_ensure_no_omp_declare_simd (parser);
cp_ensure_no_oacc_routine (parser);
bool type_definition_ok_p = cp_parser_check_type_definition (parser);
++parser->num_classes_being_defined;
saved_num_template_parameter_lists
= parser->num_template_parameter_lists;
parser->num_template_parameter_lists = 0;
saved_in_function_body = parser->in_function_body;
parser->in_function_body = false;
in_statement = parser->in_statement;
parser->in_statement = 0;
in_switch_statement_p = parser->in_switch_statement_p;
parser->in_switch_statement_p = false;
saved_in_unbraced_linkage_specification_p
= parser->in_unbraced_linkage_specification_p;
parser->in_unbraced_linkage_specification_p = false;
if (flag_concepts)
type = associate_classtype_constraints (type);
if (nested_name_specifier_p)
{
scope = CP_DECL_CONTEXT (TYPE_MAIN_DECL (type));
old_scope = push_inner_scope (scope);
}
type = begin_class_definition (type);
if (type == error_mark_node)
cp_parser_skip_to_closing_brace (parser);
else
cp_parser_member_specification_opt (parser);
closing_brace = braces.require_close (parser);
if (cp_parser_allow_gnu_extensions_p (parser))
attributes = cp_parser_gnu_attributes_opt (parser);
if (type != error_mark_node)
type = finish_struct (type, attributes);
if (nested_name_specifier_p)
pop_inner_scope (old_scope, scope);
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
bool want_semicolon = true;
if (cp_next_tokens_can_be_std_attribute_p (parser))
want_semicolon = false;
switch (token->type)
{
case CPP_NAME:
case CPP_SEMICOLON:
case CPP_MULT:
case CPP_AND:
case CPP_OPEN_PAREN:
case CPP_CLOSE_PAREN:
case CPP_COMMA:
want_semicolon = false;
break;
case CPP_KEYWORD:
if (keyword_is_decl_specifier (token->keyword))
{
cp_token *lookahead = cp_lexer_peek_nth_token (parser->lexer, 2);
want_semicolon
= (lookahead->type == CPP_KEYWORD
&& keyword_begins_type_specifier (lookahead->keyword));
}
break;
default:
break;
}
if (closing_brace && TYPE_P (type) && want_semicolon)
{
cp_token_position prev
= cp_lexer_previous_token_position (parser->lexer);
cp_token *prev_token = cp_lexer_token_at (parser->lexer, prev);
location_t loc = prev_token->location;
location_t next_loc = loc;
if (!linemap_location_from_macro_expansion_p (line_table, loc))
next_loc = linemap_position_for_loc_and_offset (line_table, loc, 1);
rich_location richloc (line_table, next_loc);
if (next_loc != loc)
richloc.add_fixit_insert_before (next_loc, ";");
if (CLASSTYPE_DECLARED_CLASS (type))
error_at (&richloc,
"expected %<;%> after class definition");
else if (TREE_CODE (type) == RECORD_TYPE)
error_at (&richloc,
"expected %<;%> after struct definition");
else if (TREE_CODE (type) == UNION_TYPE)
error_at (&richloc,
"expected %<;%> after union definition");
else
gcc_unreachable ();
cp_lexer_set_token_position (parser->lexer, prev);
token = cp_lexer_peek_token (parser->lexer);
token->type = CPP_SEMICOLON;
token->keyword = RID_MAX;
}
}
if (--parser->num_classes_being_defined == 0)
{
tree decl;
tree class_type = NULL_TREE;
tree pushed_scope = NULL_TREE;
unsigned ix;
cp_default_arg_entry *e;
tree save_ccp, save_ccr;
if (!type_definition_ok_p || any_erroneous_template_args_p (type))
{
vec_safe_truncate (unparsed_funs_with_default_args, 0);
vec_safe_truncate (unparsed_nsdmis, 0);
vec_safe_truncate (unparsed_classes, 0);
vec_safe_truncate (unparsed_funs_with_definitions, 0);
}
FOR_EACH_VEC_SAFE_ELT (unparsed_funs_with_default_args, ix, e)
{
decl = e->decl;
if (class_type != e->class_type)
{
if (pushed_scope)
pop_scope (pushed_scope);
class_type = e->class_type;
pushed_scope = push_scope (class_type);
}
maybe_begin_member_template_processing (decl);
cp_parser_late_parsing_default_args (parser, decl);
maybe_end_member_template_processing ();
}
vec_safe_truncate (unparsed_funs_with_default_args, 0);
save_ccp = current_class_ptr;
save_ccr = current_class_ref;
FOR_EACH_VEC_SAFE_ELT (unparsed_nsdmis, ix, decl)
{
if (class_type != DECL_CONTEXT (decl))
{
if (pushed_scope)
pop_scope (pushed_scope);
class_type = DECL_CONTEXT (decl);
pushed_scope = push_scope (class_type);
}
inject_this_parameter (class_type, TYPE_UNQUALIFIED);
cp_parser_late_parsing_nsdmi (parser, decl);
}
vec_safe_truncate (unparsed_nsdmis, 0);
current_class_ptr = save_ccp;
current_class_ref = save_ccr;
if (pushed_scope)
pop_scope (pushed_scope);
FOR_EACH_VEC_SAFE_ELT (unparsed_classes, ix, class_type)
after_nsdmi_defaulted_late_checks (class_type);
vec_safe_truncate (unparsed_classes, 0);
after_nsdmi_defaulted_late_checks (type);
if (flag_openmp)
{
FOR_EACH_VEC_SAFE_ELT (unparsed_funs_with_definitions, ix, decl)
if (DECL_OMP_DECLARE_REDUCTION_P (decl))
cp_parser_late_parsing_for_member (parser, decl);
FOR_EACH_VEC_SAFE_ELT (unparsed_funs_with_definitions, ix, decl)
if (!DECL_OMP_DECLARE_REDUCTION_P (decl))
cp_parser_late_parsing_for_member (parser, decl);
}
else
FOR_EACH_VEC_SAFE_ELT (unparsed_funs_with_definitions, ix, decl)
cp_parser_late_parsing_for_member (parser, decl);
vec_safe_truncate (unparsed_funs_with_definitions, 0);
}
else
vec_safe_push (unparsed_classes, type);
pop_deferring_access_checks ();
parser->in_switch_statement_p = in_switch_statement_p;
parser->in_statement = in_statement;
parser->in_function_body = saved_in_function_body;
parser->num_template_parameter_lists
= saved_num_template_parameter_lists;
parser->in_unbraced_linkage_specification_p
= saved_in_unbraced_linkage_specification_p;
return type;
}
static tree
cp_parser_class_specifier (cp_parser* parser)
{
tree ret;
timevar_push (TV_PARSE_STRUCT);
ret = cp_parser_class_specifier_1 (parser);
timevar_pop (TV_PARSE_STRUCT);
return ret;
}
static tree
cp_parser_class_head (cp_parser* parser,
bool* nested_name_specifier_p)
{
tree nested_name_specifier;
enum tag_types class_key;
tree id = NULL_TREE;
tree type = NULL_TREE;
tree attributes;
tree bases;
cp_virt_specifiers virt_specifiers = VIRT_SPEC_UNSPECIFIED;
bool template_id_p = false;
bool qualified_p = false;
bool invalid_nested_name_p = false;
bool invalid_explicit_specialization_p = false;
bool saved_colon_corrects_to_scope_p = parser->colon_corrects_to_scope_p;
tree pushed_scope = NULL_TREE;
unsigned num_templates;
cp_token *type_start_token = NULL, *nested_name_specifier_token_start = NULL;
*nested_name_specifier_p = false;
num_templates = 0;
parser->colon_corrects_to_scope_p = false;
class_key = cp_parser_class_key (parser);
if (class_key == none_type)
return error_mark_node;
location_t class_head_start_location = input_location;
attributes = cp_parser_attributes_opt (parser);
if (cp_parser_global_scope_opt (parser, false))
qualified_p = true;
push_deferring_access_checks (dk_no_check);
nested_name_specifier_token_start = cp_lexer_peek_token (parser->lexer);
nested_name_specifier
= cp_parser_nested_name_specifier_opt (parser,
false,
false,
true,
false);
cp_token *bad_template_keyword = NULL;
if (nested_name_specifier)
{
type_start_token = cp_lexer_peek_token (parser->lexer);
cp_parser_parse_tentatively (parser);
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TEMPLATE))
bad_template_keyword = cp_lexer_consume_token (parser->lexer);
type = cp_parser_class_name (parser,
false,
false,
class_type,
false,
true,
false);
if (!cp_parser_parse_definitely (parser))
{
invalid_nested_name_p = true;
type_start_token = cp_lexer_peek_token (parser->lexer);
id = cp_parser_identifier (parser);
if (id == error_mark_node)
id = NULL_TREE;
}
if (type == error_mark_node)
nested_name_specifier = NULL_TREE;
else
{
tree scope;
for (scope = TREE_TYPE (type);
scope && TREE_CODE (scope) != NAMESPACE_DECL;
scope = get_containing_scope (scope))
if (TYPE_P (scope)
&& CLASS_TYPE_P (scope)
&& CLASSTYPE_TEMPLATE_INFO (scope)
&& PRIMARY_TEMPLATE_P (CLASSTYPE_TI_TEMPLATE (scope))
&& (!CLASSTYPE_TEMPLATE_SPECIALIZATION (scope)
|| uses_template_parms (CLASSTYPE_TI_ARGS (scope))))
++num_templates;
}
}
else
{
cp_parser_parse_tentatively (parser);
type_start_token = cp_lexer_peek_token (parser->lexer);
id = cp_parser_template_id (parser,
false,
true,
class_key,
true);
if (!cp_parser_parse_definitely (parser))
{
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
type_start_token = cp_lexer_peek_token (parser->lexer);
id = cp_parser_identifier (parser);
}
else
id = NULL_TREE;
}
else
{
template_id_p = true;
++num_templates;
}
}
pop_deferring_access_checks ();
if (id)
{
cp_parser_check_for_invalid_template_id (parser, id,
class_key,
type_start_token->location);
}
virt_specifiers = cp_parser_virt_specifier_seq_opt (parser);
if (!cp_parser_next_token_starts_class_definition_p (parser))
{
cp_parser_error (parser, "expected %<{%> or %<:%>");
type = error_mark_node;
goto out;
}
cp_parser_commit_to_tentative_parse (parser);
if (virt_specifiers & VIRT_SPEC_OVERRIDE)
{
cp_parser_error (parser,
"cannot specify %<override%> for a class");
type = error_mark_node;
goto out;
}
if (qualified_p)
{
cp_parser_error (parser,
"global qualification of class name is invalid");
type = error_mark_node;
goto out;
}
else if (invalid_nested_name_p)
{
cp_parser_error (parser,
"qualified name does not name a class");
type = error_mark_node;
goto out;
}
else if (nested_name_specifier)
{
tree scope;
if (bad_template_keyword)
pedwarn (bad_template_keyword->location, OPT_Wpedantic,
"keyword %<template%> not allowed in class-head-name");
if (!DECL_IMPLICIT_TYPEDEF_P (type))
{
error_at (type_start_token->location,
"invalid class name in declaration of %qD",
type);
type = NULL_TREE;
goto done;
}
scope = current_scope ();
if (scope && !is_ancestor (scope, nested_name_specifier))
{
if (at_namespace_scope_p ())
error_at (type_start_token->location,
"declaration of %qD in namespace %qD which does not "
"enclose %qD",
type, scope, nested_name_specifier);
else
error_at (type_start_token->location,
"declaration of %qD in %qD which does not enclose %qD",
type, scope, nested_name_specifier);
type = NULL_TREE;
goto done;
}
if (scope == nested_name_specifier)
{
permerror (nested_name_specifier_token_start->location,
"extra qualification not allowed");
nested_name_specifier = NULL_TREE;
num_templates = 0;
}
}
if (at_namespace_scope_p ()
&& parser->num_template_parameter_lists == 0
&& !processing_template_parmlist
&& template_id_p)
{
location_t reported_loc
= make_location (class_head_start_location,
class_head_start_location,
get_finish (type_start_token->location));
rich_location richloc (line_table, reported_loc);
richloc.add_fixit_insert_before (class_head_start_location,
"template <> ");
error_at (&richloc,
"an explicit specialization must be preceded by"
" %<template <>%>");
invalid_explicit_specialization_p = true;
++parser->num_template_parameter_lists;
begin_specialization ();
}
if (!cp_parser_check_template_parameters (parser, num_templates,
template_id_p,
type_start_token->location,
NULL))
{
type = NULL_TREE;
goto done;
}
if (template_id_p)
{
if (TREE_CODE (id) == TEMPLATE_ID_EXPR
&& (DECL_FUNCTION_TEMPLATE_P (TREE_OPERAND (id, 0))
|| TREE_CODE (TREE_OPERAND (id, 0)) == OVERLOAD))
{
error_at (type_start_token->location,
"function template %qD redeclared as a class template", id);
type = error_mark_node;
}
else
{
type = TREE_TYPE (id);
type = maybe_process_partial_specialization (type);
if (type != error_mark_node)
check_unqualified_spec_or_inst (type, type_start_token->location);
}
if (nested_name_specifier)
pushed_scope = push_scope (nested_name_specifier);
}
else if (nested_name_specifier)
{
tree class_type;
if (TREE_CODE (TREE_TYPE (type)) == TYPENAME_TYPE)
{
class_type = resolve_typename_type (TREE_TYPE (type),
false);
if (TREE_CODE (class_type) != TYPENAME_TYPE)
type = TYPE_NAME (class_type);
else
{
cp_parser_error (parser, "could not resolve typename type");
type = error_mark_node;
}
}
if (maybe_process_partial_specialization (TREE_TYPE (type))
== error_mark_node)
{
type = NULL_TREE;
goto done;
}
class_type = current_class_type;
pushed_scope = push_scope (nested_name_specifier);
type = TYPE_MAIN_DECL (TREE_TYPE (type));
if ((PROCESSING_REAL_TEMPLATE_DECL_P ()
|| CLASSTYPE_TEMPLATE_INFO (TREE_TYPE (type)))
&& !CLASSTYPE_TEMPLATE_SPECIALIZATION (TREE_TYPE (type)))
{
type = push_template_decl (type);
if (type == error_mark_node)
{
type = NULL_TREE;
goto done;
}
}
type = TREE_TYPE (type);
*nested_name_specifier_p = true;
}
else      
{
if (!id)
id = make_anon_name ();
tag_scope tag_scope = (parser->in_type_id_in_expr_p
? ts_within_enclosing_non_class
: ts_current);
type = xref_tag (class_key, id, tag_scope,
parser->num_template_parameter_lists);
}
if (TREE_CODE (type) == RECORD_TYPE)
CLASSTYPE_DECLARED_CLASS (type) = (class_key == class_type);
cp_parser_check_class_key (class_key, type);
if (type != error_mark_node && COMPLETE_TYPE_P (type))
{
error_at (type_start_token->location, "redefinition of %q#T",
type);
inform (location_of (type), "previous definition of %q#T",
type);
type = NULL_TREE;
goto done;
}
else if (type == error_mark_node)
type = NULL_TREE;
if (type)
{
cplus_decl_attributes (&type, attributes, (int)ATTR_FLAG_TYPE_IN_PLACE);
fixup_attribute_variants (type);
}
if (cp_lexer_next_token_is (parser->lexer, CPP_COLON))
{
if (type)
pushclass (type);
bases = cp_parser_base_clause (parser);
if (type)
popclass ();
}
else
bases = NULL_TREE;
if (type && cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
xref_basetypes (type, bases);
done:
if (pushed_scope)
pop_scope (pushed_scope);
if (invalid_explicit_specialization_p)
{
end_specialization ();
--parser->num_template_parameter_lists;
}
if (type)
DECL_SOURCE_LOCATION (TYPE_NAME (type)) = type_start_token->location;
if (type && (virt_specifiers & VIRT_SPEC_FINAL))
CLASSTYPE_FINAL (type) = 1;
out:
parser->colon_corrects_to_scope_p = saved_colon_corrects_to_scope_p;
return type;
}
static enum tag_types
cp_parser_class_key (cp_parser* parser)
{
cp_token *token;
enum tag_types tag_type;
token = cp_parser_require (parser, CPP_KEYWORD, RT_CLASS_KEY);
if (!token)
return none_type;
tag_type = cp_parser_token_is_class_key (token);
if (!tag_type)
cp_parser_error (parser, "expected class-key");
return tag_type;
}
static void
cp_parser_type_parameter_key (cp_parser* parser)
{
enum tag_types tag_type = none_type;
cp_token *token = cp_lexer_peek_token (parser->lexer);
if ((tag_type = cp_parser_token_is_type_parameter_key (token)) != none_type)
{
cp_lexer_consume_token (parser->lexer);
if (pedantic && tag_type == typename_type && cxx_dialect < cxx17)
pedwarn (token->location, OPT_Wpedantic, 
"ISO C++ forbids typename key in template template parameter;"
" use -std=c++17 or -std=gnu++17");
}
else
cp_parser_error (parser, "expected %<class%> or %<typename%>");
return;
}
static void
cp_parser_member_specification_opt (cp_parser* parser)
{
while (true)
{
cp_token *token;
enum rid keyword;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_CLOSE_BRACE
|| token->type == CPP_EOF
|| token->type == CPP_PRAGMA_EOL)
break;
keyword = token->keyword;
switch (keyword)
{
case RID_PUBLIC:
case RID_PROTECTED:
case RID_PRIVATE:
cp_lexer_consume_token (parser->lexer);
current_access_specifier = token->u.value;
cp_parser_require (parser, CPP_COLON, RT_COLON);
break;
default:
if (token->type == CPP_PRAGMA)
{
cp_parser_pragma (parser, pragma_member, NULL);
break;
}
cp_parser_member_declaration (parser);
}
}
}
static void
cp_parser_member_declaration (cp_parser* parser)
{
cp_decl_specifier_seq decl_specifiers;
tree prefix_attributes;
tree decl;
int declares_class_or_enum;
bool friend_p;
cp_token *token = NULL;
cp_token *decl_spec_token_start = NULL;
cp_token *initializer_token_start = NULL;
int saved_pedantic;
bool saved_colon_corrects_to_scope_p = parser->colon_corrects_to_scope_p;
if (cp_parser_extension_opt (parser, &saved_pedantic))
{
cp_parser_member_declaration (parser);
pedantic = saved_pedantic;
return;
}
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TEMPLATE))
{
if (cp_lexer_peek_nth_token (parser->lexer, 2)->type == CPP_LESS
&& cp_lexer_peek_nth_token (parser->lexer, 3)->type == CPP_GREATER)
cp_parser_explicit_specialization (parser);
else
cp_parser_template_declaration (parser, true);
return;
}
else if (cp_parser_template_declaration_after_export (parser, true))
return;
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_USING))
{
if (cxx_dialect < cxx11)
{
cp_parser_using_declaration (parser,
false);
return;
}
else
{
tree decl;
bool alias_decl_expected;
cp_parser_parse_tentatively (parser);
decl = cp_parser_alias_declaration (parser);
alias_decl_expected =
!cp_parser_uncommitted_to_tentative_parse_p (parser);
cp_parser_parse_definitely (parser);
if (alias_decl_expected)
finish_member_declaration (decl);
else
cp_parser_using_declaration (parser,
false);
return;
}
}
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_AT_DEFS))
{
tree ivar, member;
tree ivar_chains = cp_parser_objc_defs_expression (parser);
ivar = ivar_chains;
while (ivar)
{
member = ivar;
ivar = TREE_CHAIN (member);
TREE_CHAIN (member) = NULL_TREE;
finish_member_declaration (member);
}
return;
}
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_STATIC_ASSERT))
{
cp_parser_static_assert (parser, true);
return;
}
parser->colon_corrects_to_scope_p = false;
if (cp_parser_using_declaration (parser, true))
goto out;
decl_spec_token_start = cp_lexer_peek_token (parser->lexer);
cp_parser_decl_specifier_seq (parser,
CP_PARSER_FLAGS_OPTIONAL,
&decl_specifiers,
&declares_class_or_enum);
if (!decl_specifiers.any_type_specifiers_p
&& cp_parser_parse_and_diagnose_invalid_type_name (parser))
goto out;
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
{
if (!decl_specifiers.any_specifiers_p)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (!in_system_header_at (token->location))
{
gcc_rich_location richloc (token->location);
richloc.add_fixit_remove ();
pedwarn (&richloc, OPT_Wpedantic, "extra %<;%>");
}
}
else
{
tree type;
friend_p = cp_parser_friend_p (&decl_specifiers);
type = check_tag_decl (&decl_specifiers,
false);
if (friend_p)
{
if (!declares_class_or_enum && cxx_dialect < cxx11)
pedwarn (decl_spec_token_start->location, OPT_Wpedantic,
"in C++03 a class-key must be used "
"when declaring a friend");
if (!type)
{
type = decl_specifiers.type;
if (type && TREE_CODE (type) == TYPE_DECL)
type = TREE_TYPE (type);
}
if (!type || !TYPE_P (type))
error_at (decl_spec_token_start->location,
"friend declaration does not name a class or "
"function");
else
make_friend_class (current_class_type, type,
true);
}
else if (!type || type == error_mark_node)
;
else if (ANON_AGGR_TYPE_P (type))
{
if (decl_specifiers.storage_class != sc_none)
error_at (decl_spec_token_start->location,
"a storage class on an anonymous aggregate "
"in class scope is not allowed");
fixup_anonymous_aggr (type);
decl = build_decl (decl_spec_token_start->location,
FIELD_DECL, NULL_TREE, type);
finish_member_declaration (decl);
}
else
cp_parser_check_access_in_redeclaration
(TYPE_NAME (type),
decl_spec_token_start->location);
}
}
else
{
bool assume_semicolon = false;
prefix_attributes = decl_specifiers.attributes;
decl_specifiers.attributes = NULL_TREE;
friend_p = cp_parser_friend_p (&decl_specifiers);
while (cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
{
tree attributes = NULL_TREE;
tree first_attribute;
tree initializer;
bool named_bitfld = false;
token = cp_lexer_peek_token (parser->lexer);
if (cp_next_tokens_can_be_attribute_p (parser)
|| (token->type == CPP_NAME
&& cp_nth_tokens_can_be_attribute_p (parser, 2)
&& (named_bitfld = true)))
{
size_t n
= cp_parser_skip_attributes_opt (parser, 1 + named_bitfld);
token = cp_lexer_peek_nth_token (parser->lexer, n);
}
if (token->type == CPP_COLON
|| (token->type == CPP_NAME
&& token == cp_lexer_peek_token (parser->lexer)
&& cp_lexer_nth_token_is (parser->lexer, 2, CPP_COLON)
&& (named_bitfld = true)))
{
tree identifier;
tree width;
tree late_attributes = NULL_TREE;
if (named_bitfld)
identifier = cp_parser_identifier (parser);
else
identifier = NULL_TREE;
attributes = cp_parser_attributes_opt (parser);
cp_lexer_consume_token (parser->lexer);
width = cp_parser_constant_expression (parser, false, NULL,
cxx_dialect >= cxx11);
initializer = NULL_TREE;
if (cxx_dialect >= cxx11
&& (cp_lexer_next_token_is (parser->lexer, CPP_EQ)
|| cp_lexer_next_token_is (parser->lexer,
CPP_OPEN_BRACE)))
{
location_t loc
= cp_lexer_peek_token (parser->lexer)->location;
if (cxx_dialect < cxx2a
&& !in_system_header_at (loc)
&& identifier != NULL_TREE)
pedwarn (loc, 0,
"default member initializers for bit-fields "
"only available with -std=c++2a or "
"-std=gnu++2a");
initializer = cp_parser_save_nsdmi (parser);
if (identifier == NULL_TREE)
{
error_at (loc, "default member initializer for "
"unnamed bit-field");
initializer = NULL_TREE;
}
}
else
{ 
if (cp_next_tokens_can_be_std_attribute_p (parser))
pedwarn (input_location, OPT_Wpedantic,
"ISO C++ allows bit-field attributes only "
"before the %<:%> token");
late_attributes = cp_parser_attributes_opt (parser);
}
attributes = attr_chainon (attributes, late_attributes);
first_attribute = attributes;
attributes = attr_chainon (prefix_attributes, attributes);
decl = grokbitfield (identifier
? make_id_declarator (NULL_TREE,
identifier,
sfk_none)
: NULL,
&decl_specifiers,
width, initializer,
attributes);
}
else
{
cp_declarator *declarator;
tree asm_specification;
int ctor_dtor_or_conv_p;
declarator
= cp_parser_declarator (parser, CP_PARSER_DECLARATOR_NAMED,
&ctor_dtor_or_conv_p,
NULL,
true,
friend_p);
if (declarator == cp_error_declarator)
{
cp_parser_skip_to_end_of_statement (parser);
if (cp_lexer_next_token_is (parser->lexer,
CPP_SEMICOLON))
cp_lexer_consume_token (parser->lexer);
goto out;
}
if (declares_class_or_enum & 2)
cp_parser_check_for_definition_in_return_type
(declarator, decl_specifiers.type,
decl_specifiers.locations[ds_type_spec]);
asm_specification = cp_parser_asm_specification_opt (parser);
attributes = cp_parser_attributes_opt (parser);
first_attribute = attributes;
attributes = attr_chainon (prefix_attributes, attributes);
if (cp_lexer_next_token_is (parser->lexer, CPP_EQ))
{
initializer_token_start = cp_lexer_peek_token (parser->lexer);
if (function_declarator_p (declarator)
|| (decl_specifiers.type
&& TREE_CODE (decl_specifiers.type) == TYPE_DECL
&& declarator->kind == cdk_id
&& (TREE_CODE (TREE_TYPE (decl_specifiers.type))
== FUNCTION_TYPE)))
initializer = cp_parser_pure_specifier (parser);
else if (decl_specifiers.storage_class != sc_static)
initializer = cp_parser_save_nsdmi (parser);
else if (cxx_dialect >= cxx11)
{
bool nonconst;
cp_lexer_consume_token (parser->lexer);
initializer = cp_parser_initializer_clause (parser,
&nonconst);
}
else
initializer = cp_parser_constant_initializer (parser);
}
else if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE)
&& !function_declarator_p (declarator))
{
bool x;
if (decl_specifiers.storage_class != sc_static)
initializer = cp_parser_save_nsdmi (parser);
else
initializer = cp_parser_initializer (parser, &x, &x);
}
else
initializer = NULL_TREE;
if (cp_parser_token_starts_function_definition_p
(cp_lexer_peek_token (parser->lexer)))
{
if (initializer && initializer_token_start)
error_at (initializer_token_start->location,
"pure-specifier on function-definition");
decl = cp_parser_save_member_function_body (parser,
&decl_specifiers,
declarator,
attributes);
if (parser->fully_implicit_function_template_p)
decl = finish_fully_implicit_template (parser, decl);
if (!friend_p)
finish_member_declaration (decl);
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_SEMICOLON)
{
location_t semicolon_loc
= cp_lexer_consume_token (parser->lexer)->location;
gcc_rich_location richloc (semicolon_loc);
richloc.add_fixit_remove ();
warning_at (&richloc, OPT_Wextra_semi,
"extra %<;%> after in-class "
"function definition");
}
goto out;
}
else
if (declarator->kind == cdk_function)
declarator->id_loc = token->location;
decl = grokfield (declarator, &decl_specifiers,
initializer, true,
asm_specification, attributes);
if (parser->fully_implicit_function_template_p)
{
if (friend_p)
finish_fully_implicit_template (parser, 0);
else
decl = finish_fully_implicit_template (parser, decl);
}
}
cp_finalize_omp_declare_simd (parser, decl);
cp_finalize_oacc_routine (parser, decl, false);
if (attributes != error_mark_node)
{
while (attributes && TREE_CHAIN (attributes) != first_attribute)
attributes = TREE_CHAIN (attributes);
if (attributes)
TREE_CHAIN (attributes) = NULL_TREE;
}
parser->scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
parser->object_scope = NULL_TREE;
if (cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
{
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
{
cp_token *token = cp_lexer_previous_token (parser->lexer);
gcc_rich_location richloc (token->location);
richloc.add_fixit_remove ();
error_at (&richloc, "stray %<,%> at end of "
"member declaration");
}
}
else if (cp_lexer_next_token_is_not (parser->lexer,
CPP_SEMICOLON))
{
cp_token *token = cp_lexer_previous_token (parser->lexer);
gcc_rich_location richloc (token->location);
richloc.add_fixit_insert_after (";");
error_at (&richloc, "expected %<;%> at end of "
"member declaration");
assume_semicolon = true;
}
if (decl)
{
if (!friend_p
|| !DECL_DECLARES_FUNCTION_P (decl))
finish_member_declaration (decl);
if (TREE_CODE (decl) == FUNCTION_DECL)
cp_parser_save_default_args (parser, decl);
else if (TREE_CODE (decl) == FIELD_DECL
&& DECL_INITIAL (decl))
vec_safe_push (unparsed_nsdmis, decl);
}
if (assume_semicolon)
goto out;
}
}
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
out:
parser->colon_corrects_to_scope_p = saved_colon_corrects_to_scope_p;
}
static tree
cp_parser_pure_specifier (cp_parser* parser)
{
cp_token *token;
if (!cp_parser_require (parser, CPP_EQ, RT_EQ))
return error_mark_node;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_EOF
|| token->type == CPP_PRAGMA_EOL)
return error_mark_node;
cp_lexer_consume_token (parser->lexer);
if (token->keyword == RID_DEFAULT
|| token->keyword == RID_DELETE)
{
maybe_warn_cpp0x (CPP0X_DEFAULTED_DELETED);
return token->u.value;
}
if (token->type != CPP_NUMBER || !(token->flags & PURE_ZERO))
{
cp_parser_error (parser,
"invalid pure specifier (only %<= 0%> is allowed)");
cp_parser_skip_to_end_of_statement (parser);
return error_mark_node;
}
if (PROCESSING_REAL_TEMPLATE_DECL_P ())
{
error_at (token->location, "templates may not be %<virtual%>");
return error_mark_node;
}
return integer_zero_node;
}
static tree
cp_parser_constant_initializer (cp_parser* parser)
{
if (!cp_parser_require (parser, CPP_EQ, RT_EQ))
return error_mark_node;
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
cp_parser_error (parser,
"a brace-enclosed initializer is not allowed here");
matching_braces braces;
braces.consume_open (parser);
cp_parser_skip_to_closing_brace (parser);
braces.require_close (parser);
return error_mark_node;
}
return cp_parser_constant_expression (parser);
}
static tree
cp_parser_base_clause (cp_parser* parser)
{
tree bases = NULL_TREE;
cp_parser_require (parser, CPP_COLON, RT_COLON);
while (true)
{
cp_token *token;
tree base;
bool pack_expansion_p = false;
base = cp_parser_base_specifier (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
cp_lexer_consume_token (parser->lexer);
pack_expansion_p = true;
}
if (base && base != error_mark_node)
{
if (pack_expansion_p)
TREE_VALUE (base) = make_pack_expansion (TREE_VALUE (base));
if (!check_for_bare_parameter_packs (TREE_VALUE (base)))
{
TREE_CHAIN (base) = bases;
bases = base;
}
}
token = cp_lexer_peek_token (parser->lexer);
if (token->type != CPP_COMMA)
break;
cp_lexer_consume_token (parser->lexer);
}
parser->scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
parser->object_scope = NULL_TREE;
return nreverse (bases);
}
static tree
cp_parser_base_specifier (cp_parser* parser)
{
cp_token *token;
bool done = false;
bool virtual_p = false;
bool duplicate_virtual_error_issued_p = false;
bool duplicate_access_error_issued_p = false;
bool class_scope_p, template_p;
tree access = access_default_node;
tree type;
while (!done)
{
token = cp_lexer_peek_token (parser->lexer);
switch (token->keyword)
{
case RID_VIRTUAL:
if (virtual_p && !duplicate_virtual_error_issued_p)
{
cp_parser_error (parser,
"%<virtual%> specified more than once in base-specifier");
duplicate_virtual_error_issued_p = true;
}
virtual_p = true;
cp_lexer_consume_token (parser->lexer);
break;
case RID_PUBLIC:
case RID_PROTECTED:
case RID_PRIVATE:
if (access != access_default_node
&& !duplicate_access_error_issued_p)
{
cp_parser_error (parser,
"more than one access specifier in base-specifier");
duplicate_access_error_issued_p = true;
}
access = ridpointers[(int) token->keyword];
cp_lexer_consume_token (parser->lexer);
break;
default:
done = true;
break;
}
}
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TYPENAME))
{
token = cp_lexer_peek_token (parser->lexer);
if (!processing_template_decl)
error_at (token->location,
"keyword %<typename%> not allowed outside of templates");
else
error_at (token->location,
"keyword %<typename%> not allowed in this context "
"(the base class is implicitly a type)");
cp_lexer_consume_token (parser->lexer);
}
cp_parser_global_scope_opt (parser, false);
cp_parser_nested_name_specifier_opt (parser,
true,
true,
true,
true);
class_scope_p = (parser->scope && TYPE_P (parser->scope));
template_p = class_scope_p && cp_parser_optional_template_keyword (parser);
if (!parser->scope
&& cp_lexer_next_token_is_decltype (parser->lexer))
type = cp_parser_decltype (parser);
else
{
type = cp_parser_class_name (parser,
class_scope_p,
template_p,
typename_type,
true,
false,
true);
type = TREE_TYPE (type);
}
if (type == error_mark_node)
return error_mark_node;
return finish_base_specifier (type, access, virtual_p);
}
static tree
cp_parser_noexcept_specification_opt (cp_parser* parser,
bool require_constexpr,
bool* consumed_expr,
bool return_cond)
{
cp_token *token;
const char *saved_message;
token = cp_lexer_peek_token (parser->lexer);
if (cp_parser_is_keyword (token, RID_NOEXCEPT))
{
tree expr;
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_peek_token (parser->lexer)->type == CPP_OPEN_PAREN)
{
matching_parens parens;
parens.consume_open (parser);
if (require_constexpr)
{
saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in an exception-specification");
expr = cp_parser_constant_expression (parser);
parser->type_definition_forbidden_message = saved_message;
}
else
{
expr = cp_parser_expression (parser);
*consumed_expr = true;
}
parens.require_close (parser);
}
else
{
expr = boolean_true_node;
if (!require_constexpr)
*consumed_expr = false;
}
if (!return_cond)
return build_noexcept_spec (expr, tf_warning_or_error);
else
return expr;
}
else
return NULL_TREE;
}
static tree
cp_parser_exception_specification_opt (cp_parser* parser)
{
cp_token *token;
tree type_id_list;
const char *saved_message;
token = cp_lexer_peek_token (parser->lexer);
type_id_list = cp_parser_noexcept_specification_opt (parser, true, NULL,
false);
if (type_id_list != NULL_TREE)
return type_id_list;
if (!cp_parser_is_keyword (token, RID_THROW))
return NULL_TREE;
location_t loc = token->location;
cp_lexer_consume_token (parser->lexer);
matching_parens parens;
parens.require_open (parser);
token = cp_lexer_peek_token (parser->lexer);
if (token->type != CPP_CLOSE_PAREN)
{
saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in an exception-specification");
type_id_list = cp_parser_type_id_list (parser);
parser->type_definition_forbidden_message = saved_message;
if (cxx_dialect >= cxx17)
{
error_at (loc, "ISO C++17 does not allow dynamic exception "
"specifications");
type_id_list = NULL_TREE;
}
else if (cxx_dialect >= cxx11 && !in_system_header_at (loc))
warning_at (loc, OPT_Wdeprecated,
"dynamic exception specifications are deprecated in "
"C++11");
}
else if (cxx_dialect >= cxx17)
type_id_list = noexcept_true_spec;
else
type_id_list = empty_except_spec;
parens.require_close (parser);
return type_id_list;
}
static tree
cp_parser_type_id_list (cp_parser* parser)
{
tree types = NULL_TREE;
while (true)
{
cp_token *token;
tree type;
token = cp_lexer_peek_token (parser->lexer);
type = cp_parser_type_id (parser);
if (flag_concepts && type_uses_auto (type))
{
error_at (token->location,
"invalid use of %<auto%> in exception-specification");
type = error_mark_node;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
cp_lexer_consume_token (parser->lexer);
type = make_pack_expansion (type);
}
types = add_exception_specifier (types, type, 1);
token = cp_lexer_peek_token (parser->lexer);
if (token->type != CPP_COMMA)
break;
cp_lexer_consume_token (parser->lexer);
}
return nreverse (types);
}
static tree
cp_parser_try_block (cp_parser* parser)
{
tree try_block;
cp_parser_require_keyword (parser, RID_TRY, RT_TRY);
if (parser->in_function_body
&& DECL_DECLARED_CONSTEXPR_P (current_function_decl))
error ("%<try%> in %<constexpr%> function");
try_block = begin_try_block ();
cp_parser_compound_statement (parser, NULL, BCS_TRY_BLOCK, false);
finish_try_block (try_block);
cp_parser_handler_seq (parser);
finish_handler_sequence (try_block);
return try_block;
}
static void
cp_parser_function_try_block (cp_parser* parser)
{
tree compound_stmt;
tree try_block;
if (!cp_parser_require_keyword (parser, RID_TRY, RT_TRY))
return;
try_block = begin_function_try_block (&compound_stmt);
cp_parser_ctor_initializer_opt_and_function_body
(parser, true);
finish_function_try_block (try_block);
cp_parser_handler_seq (parser);
finish_function_handler_sequence (try_block, compound_stmt);
}
static void
cp_parser_handler_seq (cp_parser* parser)
{
while (true)
{
cp_token *token;
cp_parser_handler (parser);
token = cp_lexer_peek_token (parser->lexer);
if (!cp_parser_is_keyword (token, RID_CATCH))
break;
}
}
static void
cp_parser_handler (cp_parser* parser)
{
tree handler;
tree declaration;
cp_parser_require_keyword (parser, RID_CATCH, RT_CATCH);
handler = begin_handler ();
matching_parens parens;
parens.require_open (parser);
declaration = cp_parser_exception_declaration (parser);
finish_handler_parms (declaration, handler);
parens.require_close (parser);
cp_parser_compound_statement (parser, NULL, BCS_NORMAL, false);
finish_handler (handler);
}
static tree
cp_parser_exception_declaration (cp_parser* parser)
{
cp_decl_specifier_seq type_specifiers;
cp_declarator *declarator;
const char *saved_message;
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
cp_lexer_consume_token (parser->lexer);
return NULL_TREE;
}
saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in exception-declarations");
cp_parser_type_specifier_seq (parser, true,
false,
&type_specifiers);
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_PAREN))
declarator = NULL;
else
declarator = cp_parser_declarator (parser, CP_PARSER_DECLARATOR_EITHER,
NULL,
NULL,
false,
false);
parser->type_definition_forbidden_message = saved_message;
if (!type_specifiers.any_specifiers_p)
return error_mark_node;
return grokdeclarator (declarator, &type_specifiers, CATCHPARM, 1, NULL);
}
static tree
cp_parser_throw_expression (cp_parser* parser)
{
tree expression;
cp_token* token;
cp_parser_require_keyword (parser, RID_THROW, RT_THROW);
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_COMMA
|| token->type == CPP_SEMICOLON
|| token->type == CPP_CLOSE_PAREN
|| token->type == CPP_CLOSE_SQUARE
|| token->type == CPP_CLOSE_BRACE
|| token->type == CPP_COLON)
expression = NULL_TREE;
else
expression = cp_parser_assignment_expression (parser);
return build_throw (expression);
}
static tree
cp_parser_asm_specification_opt (cp_parser* parser)
{
cp_token *token;
tree asm_specification;
token = cp_lexer_peek_token (parser->lexer);
if (!cp_parser_is_keyword (token, RID_ASM))
return NULL_TREE;
cp_lexer_consume_token (parser->lexer);
matching_parens parens;
parens.require_open (parser);
asm_specification = cp_parser_string_literal (parser, false, false);
parens.require_close (parser);
return asm_specification;
}
static tree
cp_parser_asm_operand_list (cp_parser* parser)
{
tree asm_operands = NULL_TREE;
bool invalid_operands = false;
while (true)
{
tree string_literal;
tree expression;
tree name;
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_SQUARE))
{
cp_lexer_consume_token (parser->lexer);
name = cp_parser_identifier (parser);
if (name != error_mark_node)
name = build_string (IDENTIFIER_LENGTH (name),
IDENTIFIER_POINTER (name));
cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE);
}
else
name = NULL_TREE;
string_literal = cp_parser_string_literal (parser, false, false);
matching_parens parens;
parens.require_open (parser);
expression = cp_parser_expression (parser);
parens.require_close (parser);
if (name == error_mark_node 
|| string_literal == error_mark_node 
|| expression == error_mark_node)
invalid_operands = true;
asm_operands = tree_cons (build_tree_list (name, string_literal),
expression,
asm_operands);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA))
break;
cp_lexer_consume_token (parser->lexer);
}
return invalid_operands ? error_mark_node : nreverse (asm_operands);
}
static tree
cp_parser_asm_clobber_list (cp_parser* parser)
{
tree clobbers = NULL_TREE;
while (true)
{
tree string_literal;
string_literal = cp_parser_string_literal (parser, false, false);
clobbers = tree_cons (NULL_TREE, string_literal, clobbers);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA))
break;
cp_lexer_consume_token (parser->lexer);
}
return clobbers;
}
static tree
cp_parser_asm_label_list (cp_parser* parser)
{
tree labels = NULL_TREE;
while (true)
{
tree identifier, label, name;
identifier = cp_parser_identifier (parser);
if (!error_operand_p (identifier))
{
label = lookup_label (identifier);
if (TREE_CODE (label) == LABEL_DECL)
{
TREE_USED (label) = 1;
check_goto (label);
name = build_string (IDENTIFIER_LENGTH (identifier),
IDENTIFIER_POINTER (identifier));
labels = tree_cons (name, label, labels);
}
}
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA))
break;
cp_lexer_consume_token (parser->lexer);
}
return nreverse (labels);
}
static bool
cp_next_tokens_can_be_gnu_attribute_p (cp_parser *parser)
{
return cp_nth_tokens_can_be_gnu_attribute_p (parser, 1);
}
static bool
cp_next_tokens_can_be_std_attribute_p (cp_parser *parser)
{
return cp_nth_tokens_can_be_std_attribute_p (parser, 1);
}
static bool
cp_nth_tokens_can_be_std_attribute_p (cp_parser *parser, size_t n)
{
cp_token *token = cp_lexer_peek_nth_token (parser->lexer, n);
return (cxx_dialect >= cxx11
&& ((token->type == CPP_KEYWORD && token->keyword == RID_ALIGNAS)
|| (token->type == CPP_OPEN_SQUARE
&& (token = cp_lexer_peek_nth_token (parser->lexer, n + 1))
&& token->type == CPP_OPEN_SQUARE)));
}
static bool
cp_nth_tokens_can_be_gnu_attribute_p (cp_parser *parser, size_t n)
{
cp_token *token = cp_lexer_peek_nth_token (parser->lexer, n);
return token->type == CPP_KEYWORD && token->keyword == RID_ATTRIBUTE;
}
static bool
cp_next_tokens_can_be_attribute_p (cp_parser *parser)
{
return (cp_next_tokens_can_be_gnu_attribute_p (parser)
|| cp_next_tokens_can_be_std_attribute_p (parser));
}
static bool
cp_nth_tokens_can_be_attribute_p (cp_parser *parser, size_t n)
{
return (cp_nth_tokens_can_be_gnu_attribute_p (parser, n)
|| cp_nth_tokens_can_be_std_attribute_p (parser, n));
}
static tree
cp_parser_attributes_opt (cp_parser *parser)
{
if (cp_next_tokens_can_be_gnu_attribute_p (parser))
return cp_parser_gnu_attributes_opt (parser);
return cp_parser_std_attribute_spec_seq (parser);
}
static tree
cp_parser_gnu_attributes_opt (cp_parser* parser)
{
tree attributes = NULL_TREE;
temp_override<bool> cleanup
(parser->auto_is_implicit_function_template_parm_p, false);
while (true)
{
cp_token *token;
tree attribute_list;
bool ok = true;
token = cp_lexer_peek_token (parser->lexer);
if (token->keyword != RID_ATTRIBUTE)
break;
cp_lexer_consume_token (parser->lexer);
matching_parens outer_parens;
outer_parens.require_open (parser);
matching_parens inner_parens;
inner_parens.require_open (parser);
token = cp_lexer_peek_token (parser->lexer);
if (token->type != CPP_CLOSE_PAREN)
attribute_list = cp_parser_gnu_attribute_list (parser);
else
attribute_list = NULL;
if (!inner_parens.require_close (parser))
ok = false;
if (!outer_parens.require_close (parser))
ok = false;
if (!ok)
cp_parser_skip_to_end_of_statement (parser);
attributes = attr_chainon (attributes, attribute_list);
}
return attributes;
}
static tree
cp_parser_gnu_attribute_list (cp_parser* parser)
{
tree attribute_list = NULL_TREE;
bool save_translate_strings_p = parser->translate_strings_p;
parser->translate_strings_p = false;
while (true)
{
cp_token *token;
tree identifier;
tree attribute;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_NAME
|| token->type == CPP_KEYWORD)
{
tree arguments = NULL_TREE;
cp_token *id_token = cp_lexer_consume_token (parser->lexer);
identifier = (token->type == CPP_KEYWORD) 
? ridpointers[(int) token->keyword]
: id_token->u.value;
identifier = canonicalize_attr_name (identifier);
attribute = build_tree_list (identifier, NULL_TREE);
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_OPEN_PAREN)
{
vec<tree, va_gc> *vec;
int attr_flag = (attribute_takes_identifier_p (identifier)
? id_attr : normal_attr);
vec = cp_parser_parenthesized_expression_list 
(parser, attr_flag, false, 
false, 
NULL);
if (vec == NULL)
arguments = error_mark_node;
else
{
arguments = build_tree_list_vec (vec);
release_tree_vector (vec);
}
TREE_VALUE (attribute) = arguments;
}
if (arguments != error_mark_node)
{
TREE_CHAIN (attribute) = attribute_list;
attribute_list = attribute;
}
token = cp_lexer_peek_token (parser->lexer);
}
if (token->type != CPP_COMMA)
break;
cp_lexer_consume_token (parser->lexer);
}
parser->translate_strings_p = save_translate_strings_p;
return nreverse (attribute_list);
}
static tree
cp_parser_std_attribute (cp_parser *parser, tree attr_ns)
{
tree attribute, attr_id = NULL_TREE, arguments;
cp_token *token;
temp_override<bool> cleanup
(parser->auto_is_implicit_function_template_parm_p, false);
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_NAME)
attr_id = token->u.value;
else if (token->type == CPP_KEYWORD)
attr_id = ridpointers[(int) token->keyword];
else if (token->flags & NAMED_OP)
attr_id = get_identifier (cpp_type2name (token->type, token->flags));
if (attr_id == NULL_TREE)
return NULL_TREE;
cp_lexer_consume_token (parser->lexer);
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_SCOPE)
{
cp_lexer_consume_token (parser->lexer);
if (attr_ns)
error_at (token->location, "attribute using prefix used together "
"with scoped attribute token");
attr_ns = attr_id;
token = cp_lexer_consume_token (parser->lexer);
if (token->type == CPP_NAME)
attr_id = token->u.value;
else if (token->type == CPP_KEYWORD)
attr_id = ridpointers[(int) token->keyword];
else if (token->flags & NAMED_OP)
attr_id = get_identifier (cpp_type2name (token->type, token->flags));
else
{
error_at (token->location,
"expected an identifier for the attribute name");
return error_mark_node;
}
attr_ns = canonicalize_attr_name (attr_ns);
attr_id = canonicalize_attr_name (attr_id);
attribute = build_tree_list (build_tree_list (attr_ns, attr_id),
NULL_TREE);
token = cp_lexer_peek_token (parser->lexer);
}
else if (attr_ns)
{
attr_ns = canonicalize_attr_name (attr_ns);
attr_id = canonicalize_attr_name (attr_id);
attribute = build_tree_list (build_tree_list (attr_ns, attr_id),
NULL_TREE);
}
else
{
attr_id = canonicalize_attr_name (attr_id);
attribute = build_tree_list (build_tree_list (NULL_TREE, attr_id),
NULL_TREE);
if (is_attribute_p ("noreturn", attr_id))
TREE_PURPOSE (TREE_PURPOSE (attribute)) = get_identifier ("gnu");
else if (is_attribute_p ("deprecated", attr_id))
TREE_PURPOSE (TREE_PURPOSE (attribute)) = get_identifier ("gnu");
else if (is_attribute_p ("fallthrough", attr_id))
TREE_PURPOSE (TREE_PURPOSE (attribute)) = get_identifier ("gnu");
else if (is_attribute_p ("optimize_for_synchronized", attr_id))
TREE_PURPOSE (attribute)
= get_identifier ("transaction_callable");
else if (tm_attr_to_mask (attr_id))
TREE_PURPOSE (attribute) = attr_id;
}
if (token->type != CPP_OPEN_PAREN)
return attribute;
{
vec<tree, va_gc> *vec;
int attr_flag = normal_attr;
if (attr_ns == get_identifier ("gnu")
&& attribute_takes_identifier_p (attr_id))
attr_flag = id_attr;
vec = cp_parser_parenthesized_expression_list
(parser, attr_flag, false,
true,
NULL);
if (vec == NULL)
arguments = error_mark_node;
else
{
arguments = build_tree_list_vec (vec);
release_tree_vector (vec);
}
if (arguments == error_mark_node)
attribute = error_mark_node;
else
TREE_VALUE (attribute) = arguments;
}
return attribute;
}
static void
cp_parser_check_std_attribute (tree attributes, tree attribute)
{
if (attributes)
{
tree name = get_attribute_name (attribute);
if (is_attribute_p ("noreturn", name)
&& lookup_attribute ("noreturn", attributes))
error ("attribute %<noreturn%> can appear at most once "
"in an attribute-list");
else if (is_attribute_p ("deprecated", name)
&& lookup_attribute ("deprecated", attributes))
error ("attribute %<deprecated%> can appear at most once "
"in an attribute-list");
}
}
static tree
cp_parser_std_attribute_list (cp_parser *parser, tree attr_ns)
{
tree attributes = NULL_TREE, attribute = NULL_TREE;
cp_token *token = NULL;
while (true)
{
attribute = cp_parser_std_attribute (parser, attr_ns);
if (attribute == error_mark_node)
break;
if (attribute != NULL_TREE)
{
cp_parser_check_std_attribute (attributes, attribute);
TREE_CHAIN (attribute) = attributes;
attributes = attribute;
}
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_ELLIPSIS)
{
cp_lexer_consume_token (parser->lexer);
if (attribute == NULL_TREE)
error_at (token->location,
"expected attribute before %<...%>");
else
{
tree pack = make_pack_expansion (TREE_VALUE (attribute));
if (pack == error_mark_node)
return error_mark_node;
TREE_VALUE (attribute) = pack;
}
token = cp_lexer_peek_token (parser->lexer);
}
if (token->type != CPP_COMMA)
break;
cp_lexer_consume_token (parser->lexer);
}
attributes = nreverse (attributes);
return attributes;
}
static tree
cp_parser_std_attribute_spec (cp_parser *parser)
{
tree attributes = NULL_TREE;
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_OPEN_SQUARE
&& cp_lexer_peek_nth_token (parser->lexer, 2)->type == CPP_OPEN_SQUARE)
{
tree attr_ns = NULL_TREE;
cp_lexer_consume_token (parser->lexer);
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_USING))
{
token = cp_lexer_peek_nth_token (parser->lexer, 2);
if (token->type == CPP_NAME)
attr_ns = token->u.value;
else if (token->type == CPP_KEYWORD)
attr_ns = ridpointers[(int) token->keyword];
else if (token->flags & NAMED_OP)
attr_ns = get_identifier (cpp_type2name (token->type,
token->flags));
if (attr_ns
&& cp_lexer_nth_token_is (parser->lexer, 3, CPP_COLON))
{
if (cxx_dialect < cxx17
&& !in_system_header_at (input_location))
pedwarn (input_location, 0,
"attribute using prefix only available "
"with -std=c++17 or -std=gnu++17");
cp_lexer_consume_token (parser->lexer);
cp_lexer_consume_token (parser->lexer);
cp_lexer_consume_token (parser->lexer);
}
else
attr_ns = NULL_TREE;
}
attributes = cp_parser_std_attribute_list (parser, attr_ns);
if (!cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE)
|| !cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE))
cp_parser_skip_to_end_of_statement (parser);
else
maybe_warn_cpp0x (CPP0X_ATTRIBUTES);
}
else
{
tree alignas_expr;
token = cp_lexer_peek_token (parser->lexer);
if (token->type != CPP_KEYWORD
|| token->keyword != RID_ALIGNAS)
return NULL_TREE;
cp_lexer_consume_token (parser->lexer);
maybe_warn_cpp0x (CPP0X_ATTRIBUTES);
matching_parens parens;
if (!parens.require_open (parser))
return error_mark_node;
cp_parser_parse_tentatively (parser);
alignas_expr = cp_parser_type_id (parser);
if (!cp_parser_parse_definitely (parser))
{
alignas_expr = cp_parser_assignment_expression (parser);
if (alignas_expr == error_mark_node)
cp_parser_skip_to_end_of_statement (parser);
if (alignas_expr == NULL_TREE
|| alignas_expr == error_mark_node)
return alignas_expr;
}
alignas_expr = cxx_alignas_expr (alignas_expr);
alignas_expr = build_tree_list (NULL_TREE, alignas_expr);
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
cp_lexer_consume_token (parser->lexer);
alignas_expr = make_pack_expansion (alignas_expr);
}
if (alignas_expr == error_mark_node)
return error_mark_node;
if (!parens.require_close (parser))
return error_mark_node;
attributes =
build_tree_list (build_tree_list (get_identifier ("gnu"),
get_identifier ("aligned")),
alignas_expr);
}
return attributes;
}
static tree
cp_parser_std_attribute_spec_seq (cp_parser *parser)
{
tree attr_specs = NULL_TREE;
tree attr_last = NULL_TREE;
while (true)
{
tree attr_spec = cp_parser_std_attribute_spec (parser);
if (attr_spec == NULL_TREE)
break;
if (attr_spec == error_mark_node)
return error_mark_node;
if (attr_last)
TREE_CHAIN (attr_last) = attr_spec;
else
attr_specs = attr_last = attr_spec;
attr_last = tree_last (attr_last);
}
return attr_specs;
}
static size_t
cp_parser_skip_balanced_tokens (cp_parser *parser, size_t n)
{
size_t orig_n = n;
int nparens = 0, nbraces = 0, nsquares = 0;
do
switch (cp_lexer_peek_nth_token (parser->lexer, n++)->type)
{
case CPP_EOF:
case CPP_PRAGMA_EOL:
return orig_n;
case CPP_OPEN_PAREN:
++nparens;
break;
case CPP_OPEN_BRACE:
++nbraces;
break;
case CPP_OPEN_SQUARE:
++nsquares;
break;
case CPP_CLOSE_PAREN:
--nparens;
break;
case CPP_CLOSE_BRACE:
--nbraces;
break;
case CPP_CLOSE_SQUARE:
--nsquares;
break;
default:
break;
}
while (nparens || nbraces || nsquares);
return n;
}
static size_t
cp_parser_skip_gnu_attributes_opt (cp_parser *parser, size_t n)
{
while (true)
{
if (!cp_lexer_nth_token_is_keyword (parser->lexer, n, RID_ATTRIBUTE)
|| !cp_lexer_nth_token_is (parser->lexer, n + 1, CPP_OPEN_PAREN)
|| !cp_lexer_nth_token_is (parser->lexer, n + 2, CPP_OPEN_PAREN))
break;
size_t n2 = cp_parser_skip_balanced_tokens (parser, n + 2);
if (n2 == n + 2)
break;
if (!cp_lexer_nth_token_is (parser->lexer, n2, CPP_CLOSE_PAREN))
break;
n = n2 + 1;
}
return n;
}
static size_t
cp_parser_skip_std_attribute_spec_seq (cp_parser *parser, size_t n)
{
while (true)
{
if (cp_lexer_nth_token_is (parser->lexer, n, CPP_OPEN_SQUARE)
&& cp_lexer_nth_token_is (parser->lexer, n + 1, CPP_OPEN_SQUARE))
{
size_t n2 = cp_parser_skip_balanced_tokens (parser, n + 1);
if (n2 == n + 1)
break;
if (!cp_lexer_nth_token_is (parser->lexer, n2, CPP_CLOSE_SQUARE))
break;
n = n2 + 1;
}
else if (cp_lexer_nth_token_is_keyword (parser->lexer, n, RID_ALIGNAS)
&& cp_lexer_nth_token_is (parser->lexer, n + 1, CPP_OPEN_PAREN))
{
size_t n2 = cp_parser_skip_balanced_tokens (parser, n + 1);
if (n2 == n + 1)
break;
n = n2;
}
else
break;
}
return n;
}
static size_t
cp_parser_skip_attributes_opt (cp_parser *parser, size_t n)
{
if (cp_nth_tokens_can_be_gnu_attribute_p (parser, n))
return cp_parser_skip_gnu_attributes_opt (parser, n);
return cp_parser_skip_std_attribute_spec_seq (parser, n);
}
static bool
cp_parser_extension_opt (cp_parser* parser, int* saved_pedantic)
{
*saved_pedantic = pedantic;
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_EXTENSION))
{
cp_lexer_consume_token (parser->lexer);
pedantic = 0;
return true;
}
return false;
}
static void
cp_parser_label_declaration (cp_parser* parser)
{
cp_parser_require_keyword (parser, RID_LABEL, RT_LABEL);
while (true)
{
tree identifier;
identifier = cp_parser_identifier (parser);
if (identifier == error_mark_node)
break;
finish_label_decl (identifier);
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
break;
cp_parser_require (parser, CPP_COMMA, RT_COMMA);
}
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
}
static tree
cp_parser_requires_clause (cp_parser *parser)
{
++processing_template_decl;
tree expr = cp_parser_binary_expression (parser, false, false,
PREC_NOT_OPERATOR, NULL);
if (check_for_bare_parameter_packs (expr))
expr = error_mark_node;
--processing_template_decl;
return expr;
}
static tree
cp_parser_requires_clause_opt (cp_parser *parser)
{
cp_token *tok = cp_lexer_peek_token (parser->lexer);
if (tok->keyword != RID_REQUIRES)
{
if (!flag_concepts && tok->type == CPP_NAME
&& tok->u.value == ridpointers[RID_REQUIRES])
{
error_at (cp_lexer_peek_token (parser->lexer)->location,
"%<requires%> only available with -fconcepts");
cp_lexer_consume_token (parser->lexer);
cp_parser_requires_clause (parser);
}
return NULL_TREE;
}
cp_lexer_consume_token (parser->lexer);
return cp_parser_requires_clause (parser);
}
static tree
cp_parser_requires_expression (cp_parser *parser)
{
gcc_assert (cp_lexer_next_token_is_keyword (parser->lexer, RID_REQUIRES));
location_t loc = cp_lexer_consume_token (parser->lexer)->location;
if (!processing_template_decl)
{
error_at (loc, "a requires expression cannot appear outside a template");
cp_parser_skip_to_end_of_statement (parser);
return error_mark_node;
}
tree parms, reqs;
{
struct scope_sentinel
{
scope_sentinel ()
{
++cp_unevaluated_operand;
begin_scope (sk_block, NULL_TREE);
}
~scope_sentinel ()
{
pop_bindings_and_leave_scope ();
--cp_unevaluated_operand;
}
} s;
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
parms = cp_parser_requirement_parameter_list (parser);
if (parms == error_mark_node)
return error_mark_node;
}
else
parms = NULL_TREE;
reqs = cp_parser_requirement_body (parser);
if (reqs == error_mark_node)
return error_mark_node;
}
grokparms (parms, &parms);
return finish_requires_expr (parms, reqs);
}
static tree
cp_parser_requirement_parameter_list (cp_parser *parser)
{
matching_parens parens;
if (!parens.require_open (parser))
return error_mark_node;
tree parms = cp_parser_parameter_declaration_clause (parser);
if (!parens.require_close (parser))
return error_mark_node;
return parms;
}
static tree
cp_parser_requirement_body (cp_parser *parser)
{
matching_braces braces;
if (!braces.require_open (parser))
return error_mark_node;
tree reqs = cp_parser_requirement_list (parser);
if (!braces.require_close (parser))
return error_mark_node;
return reqs;
}
static tree
cp_parser_requirement_list (cp_parser *parser)
{
tree result = NULL_TREE;
while (true)
{
tree req = cp_parser_requirement (parser);
if (req == error_mark_node)
return error_mark_node;
result = tree_cons (NULL_TREE, req, result);
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_BRACE))
break;
}
return nreverse (result);
}
static tree
cp_parser_requirement (cp_parser *parser)
{
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
return cp_parser_compound_requirement (parser);
else if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TYPENAME))
return cp_parser_type_requirement (parser);
else if (cp_lexer_next_token_is_keyword (parser->lexer, RID_REQUIRES))
return cp_parser_nested_requirement (parser);
else
return cp_parser_simple_requirement (parser);
}
static tree
cp_parser_simple_requirement (cp_parser *parser)
{
tree expr = cp_parser_expression (parser, NULL, false, false);
if (!expr || expr == error_mark_node)
return error_mark_node;
if (!cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON))
return error_mark_node;
return finish_simple_requirement (expr);
}
static tree
cp_parser_type_requirement (cp_parser *parser)
{
cp_lexer_consume_token (parser->lexer);
tree saved_scope = parser->scope;
tree saved_object_scope = parser->object_scope;
tree saved_qualifying_scope = parser->qualifying_scope;
cp_parser_global_scope_opt (parser, true);
cp_parser_nested_name_specifier_opt (parser,
true,
false,
true,
false);
tree type;
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TEMPLATE))
{
cp_lexer_consume_token (parser->lexer);
type = cp_parser_template_id (parser,
true,
false,
none_type,
false);
type = make_typename_type (parser->scope, type, typename_type,
tf_error);
}
else
type = cp_parser_type_name (parser, true);
if (TREE_CODE (type) == TYPE_DECL)
type = TREE_TYPE (type);
parser->scope = saved_scope;
parser->object_scope = saved_object_scope;
parser->qualifying_scope = saved_qualifying_scope;
if (type == error_mark_node)
cp_parser_skip_to_end_of_statement (parser);
if (!cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON))
return error_mark_node;
if (type == error_mark_node)
return error_mark_node;
return finish_type_requirement (type);
}
static tree
cp_parser_compound_requirement (cp_parser *parser)
{
matching_braces braces;
if (!braces.require_open (parser))
return error_mark_node;
tree expr = cp_parser_expression (parser, NULL, false, false);
if (!expr || expr == error_mark_node)
return error_mark_node;
if (!braces.require_close (parser))
return error_mark_node;
bool noexcept_p = false;
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_NOEXCEPT))
{
cp_lexer_consume_token (parser->lexer);
noexcept_p = true;
}
tree type = NULL_TREE;
if (cp_lexer_next_token_is (parser->lexer, CPP_DEREF))
{
cp_lexer_consume_token (parser->lexer);
bool saved_result_type_constraint_p = parser->in_result_type_constraint_p;
parser->in_result_type_constraint_p = true;
type = cp_parser_trailing_type_id (parser);
parser->in_result_type_constraint_p = saved_result_type_constraint_p;
if (type == error_mark_node)
return error_mark_node;
}
return finish_compound_requirement (expr, type, noexcept_p);
}
static tree
cp_parser_nested_requirement (cp_parser *parser)
{
cp_lexer_consume_token (parser->lexer);
tree req = cp_parser_requires_clause (parser);
if (req == error_mark_node)
return error_mark_node;
return finish_nested_requirement (req);
}
static inline int
prefer_type_arg (tag_types tag_type, bool template_mem_access = false)
{
if (template_mem_access)
return 2;
switch (tag_type)
{
case none_type:  return 0;	
case scope_type: return 1;	
default:         return 2;	
}
}
static cp_expr
cp_parser_lookup_name (cp_parser *parser, tree name,
enum tag_types tag_type,
bool is_template,
bool is_namespace,
bool check_dependency,
tree *ambiguous_decls,
location_t name_location)
{
tree decl;
tree object_type = parser->context->object_type;
if (ambiguous_decls)
*ambiguous_decls = NULL_TREE;
parser->context->object_type = NULL_TREE;
if (name == error_mark_node)
return error_mark_node;
if (TREE_CODE (name) == TEMPLATE_ID_EXPR)
return name;
if (BASELINK_P (name))
{
gcc_assert (TREE_CODE (BASELINK_FUNCTIONS (name))
== TEMPLATE_ID_EXPR);
return name;
}
if (TREE_CODE (name) == BIT_NOT_EXPR)
{
tree type;
if (parser->scope)
type = parser->scope;
else if (object_type)
type = object_type;
else
type = current_class_type;
if (!type || !CLASS_TYPE_P (type))
return error_mark_node;
if (CLASSTYPE_LAZY_DESTRUCTOR (type))
lazily_declare_fn (sfk_destructor, type);
if (tree dtor = CLASSTYPE_DESTRUCTOR (type))
return dtor;
return error_mark_node;
}
gcc_assert (identifier_p (name));
if (parser->scope)
{
bool dependent_p;
if (parser->scope == error_mark_node)
return error_mark_node;
dependent_p = (TYPE_P (parser->scope)
&& dependent_scope_p (parser->scope));
if ((check_dependency || !CLASS_TYPE_P (parser->scope))
&& dependent_p)
decl = error_mark_node;
else
{
tree pushed_scope = NULL_TREE;
if (dependent_p)
pushed_scope = push_scope (parser->scope);
decl = lookup_qualified_name (parser->scope, name,
prefer_type_arg (tag_type),
true);
if (tag_type == none_type
&& DECL_SELF_REFERENCE_P (decl)
&& same_type_p (DECL_CONTEXT (decl), parser->scope))
decl = lookup_qualified_name (parser->scope, ctor_identifier,
prefer_type_arg (tag_type),
true);
if (TREE_CODE (decl) == OVERLOAD
&& !really_overloaded_fn (decl))
decl = OVL_FUNCTION (decl);
if (pushed_scope)
pop_scope (pushed_scope);
}
if (decl == error_mark_node && TYPE_P (parser->scope)
&& dependent_type_p (parser->scope))
{
if (tag_type)
{
tree type;
type = make_typename_type (parser->scope, name, tag_type,
tf_error);
if (type != error_mark_node)
decl = TYPE_NAME (type);
}
else if (is_template
&& (cp_parser_next_token_ends_template_argument_p (parser)
|| cp_lexer_next_token_is (parser->lexer,
CPP_CLOSE_PAREN)))
decl = make_unbound_class_template (parser->scope,
name, NULL_TREE,
tf_error);
else
decl = build_qualified_name (NULL_TREE,
parser->scope, name,
is_template);
}
parser->qualifying_scope = parser->scope;
parser->object_scope = NULL_TREE;
}
else if (object_type)
{
if (CLASS_TYPE_P (object_type))
decl = lookup_member (object_type,
name,
0,
prefer_type_arg (tag_type),
tf_warning_or_error);
else
decl = NULL_TREE;
if (!decl)
decl = lookup_name_real (name, prefer_type_arg (tag_type, is_template),
0,
true, is_namespace, 0);
if (object_type == unknown_type_node)
object_type = NULL_TREE;
parser->object_scope = object_type;
parser->qualifying_scope = NULL_TREE;
}
else
{
decl = lookup_name_real (name, prefer_type_arg (tag_type),
0,
true, is_namespace, 0);
parser->qualifying_scope = NULL_TREE;
parser->object_scope = NULL_TREE;
}
if (!decl || decl == error_mark_node)
return error_mark_node;
if (is_template)
decl = maybe_get_template_decl_from_type_decl (decl);
if (TREE_CODE (decl) == TREE_LIST)
{
if (ambiguous_decls)
*ambiguous_decls = decl;
if (!cp_parser_simulate_error (parser))
{
error_at (name_location, "reference to %qD is ambiguous",
name);
print_candidates (decl);
}
return error_mark_node;
}
gcc_assert (DECL_P (decl)
|| TREE_CODE (decl) == OVERLOAD
|| TREE_CODE (decl) == SCOPE_REF
|| TREE_CODE (decl) == UNBOUND_CLASS_TEMPLATE
|| BASELINK_P (decl));
if (DECL_P (decl))
check_accessibility_of_qualified_id (decl, object_type, parser->scope);
maybe_record_typedef_use (decl);
return cp_expr (decl, name_location);
}
static tree
cp_parser_lookup_name_simple (cp_parser* parser, tree name, location_t location)
{
return cp_parser_lookup_name (parser, name,
none_type,
false,
false,
true,
NULL,
location);
}
static tree
cp_parser_maybe_treat_template_as_class (tree decl, bool tag_name_p)
{
if (DECL_CLASS_TEMPLATE_P (decl) && tag_name_p)
return DECL_TEMPLATE_RESULT (decl);
return decl;
}
static bool
cp_parser_check_declarator_template_parameters (cp_parser* parser,
cp_declarator *declarator,
location_t declarator_location)
{
switch (declarator->kind)
{
case cdk_id:
{
unsigned num_templates = 0;
tree scope = declarator->u.id.qualifying_scope;
bool template_id_p = false;
if (scope)
num_templates = num_template_headers_for_class (scope);
else if (TREE_CODE (declarator->u.id.unqualified_name)
== TEMPLATE_ID_EXPR)
{
++num_templates;
template_id_p = true;
}
return cp_parser_check_template_parameters 
(parser, num_templates, template_id_p, declarator_location,
declarator);
}
case cdk_function:
case cdk_array:
case cdk_pointer:
case cdk_reference:
case cdk_ptrmem:
return (cp_parser_check_declarator_template_parameters
(parser, declarator->declarator, declarator_location));
case cdk_decomp:
case cdk_error:
return true;
default:
gcc_unreachable ();
}
return false;
}
static bool
cp_parser_check_template_parameters (cp_parser* parser,
unsigned num_templates,
bool template_id_p,
location_t location,
cp_declarator *declarator)
{
if (parser->num_template_parameter_lists == num_templates)
return true;
if (!template_id_p
&& parser->num_template_parameter_lists == num_templates + 1)
return true;
if (parser->num_template_parameter_lists < num_templates)
{
if (declarator && !current_function_decl)
error_at (location, "specializing member %<%T::%E%> "
"requires %<template<>%> syntax", 
declarator->u.id.qualifying_scope,
declarator->u.id.unqualified_name);
else if (declarator)
error_at (location, "invalid declaration of %<%T::%E%>",
declarator->u.id.qualifying_scope,
declarator->u.id.unqualified_name);
else 
error_at (location, "too few template-parameter-lists");
return false;
}
error_at (location, "too many template-parameter-lists");
return false;
}
static tree
cp_parser_global_scope_opt (cp_parser* parser, bool current_scope_valid_p)
{
cp_token *token;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_SCOPE)
{
cp_lexer_consume_token (parser->lexer);
parser->scope = global_namespace;
parser->qualifying_scope = global_namespace;
parser->object_scope = NULL_TREE;
return parser->scope;
}
else if (!current_scope_valid_p)
{
parser->scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
parser->object_scope = NULL_TREE;
}
return NULL_TREE;
}
static bool
cp_parser_constructor_declarator_p (cp_parser *parser, bool friend_p)
{
bool constructor_p;
bool outside_class_specifier_p;
tree nested_name_specifier;
cp_token *next_token;
if (parser->in_function_body)
return false;
next_token = cp_lexer_peek_token (parser->lexer);
if (next_token->type != CPP_NAME
&& next_token->type != CPP_SCOPE
&& next_token->type != CPP_NESTED_NAME_SPECIFIER
&& next_token->type != CPP_TEMPLATE_ID)
return false;
cp_parser_parse_tentatively (parser);
constructor_p = true;
cp_parser_global_scope_opt (parser,
false);
nested_name_specifier
= (cp_parser_nested_name_specifier_opt (parser,
false,
false,
false,
false));
outside_class_specifier_p = (!at_class_scope_p ()
|| !TYPE_BEING_DEFINED (current_class_type)
|| friend_p);
if (!nested_name_specifier && outside_class_specifier_p
&& cxx_dialect < cxx17)
constructor_p = false;
else if (nested_name_specifier == error_mark_node)
constructor_p = false;
if (constructor_p && nested_name_specifier
&& CLASS_TYPE_P (nested_name_specifier))
{
tree id = cp_parser_unqualified_id (parser,
false,
false,
true,
false);
if (is_overloaded_fn (id))
id = DECL_NAME (get_first_fn (id));
if (!constructor_name_p (id, nested_name_specifier))
constructor_p = false;
}
else if (constructor_p)
{
if (cxx_dialect >= cxx17)
cp_parser_parse_tentatively (parser);
tree type_decl;
type_decl = cp_parser_class_name (parser,
false,
false,
none_type,
false,
false,
false);
if (cxx_dialect >= cxx17
&& !cp_parser_parse_definitely (parser))
{
type_decl = NULL_TREE;
tree tmpl = cp_parser_template_name (parser,
false,
false,
false,
none_type,
NULL);
if (DECL_CLASS_TEMPLATE_P (tmpl)
|| DECL_TEMPLATE_TEMPLATE_PARM_P (tmpl))
;
else
cp_parser_simulate_error (parser);
}
constructor_p = (!cp_parser_error_occurred (parser)
&& (outside_class_specifier_p
|| type_decl == NULL_TREE
|| type_decl == error_mark_node
|| same_type_p (current_class_type,
TREE_TYPE (type_decl))));
if (constructor_p
&& !cp_parser_require (parser, CPP_OPEN_PAREN, RT_OPEN_PAREN))
constructor_p = false;
if (constructor_p
&& cp_lexer_next_token_is_not (parser->lexer, CPP_CLOSE_PAREN)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_ELLIPSIS)
&& !cp_lexer_next_token_is_decl_specifier_keyword (parser->lexer))
{
tree type;
tree pushed_scope = NULL_TREE;
unsigned saved_num_template_parameter_lists;
if (current_class_type)
type = NULL_TREE;
else if (type_decl)
{
type = TREE_TYPE (type_decl);
if (TREE_CODE (type) == TYPENAME_TYPE)
{
type = resolve_typename_type (type,
false);
if (TREE_CODE (type) == TYPENAME_TYPE)
{
cp_parser_abort_tentative_parse (parser);
return false;
}
}
pushed_scope = push_scope (type);
}
saved_num_template_parameter_lists
= parser->num_template_parameter_lists;
parser->num_template_parameter_lists = 0;
cp_parser_type_specifier (parser,
CP_PARSER_FLAGS_NONE,
NULL,
true,
NULL,
NULL);
parser->num_template_parameter_lists
= saved_num_template_parameter_lists;
if (pushed_scope)
pop_scope (pushed_scope);
constructor_p = !cp_parser_error_occurred (parser);
}
}
cp_parser_abort_tentative_parse (parser);
return constructor_p;
}
static tree
cp_parser_function_definition_from_specifiers_and_declarator
(cp_parser* parser,
cp_decl_specifier_seq *decl_specifiers,
tree attributes,
const cp_declarator *declarator)
{
tree fn;
bool success_p;
success_p = start_function (decl_specifiers, declarator, attributes);
reset_specialization ();
perform_deferred_access_checks (tf_warning_or_error);
if (success_p)
{
cp_finalize_omp_declare_simd (parser, current_function_decl);
parser->omp_declare_simd = NULL;
cp_finalize_oacc_routine (parser, current_function_decl, true);
parser->oacc_routine = NULL;
}
if (!success_p)
{
cp_parser_skip_to_end_of_block_or_statement (parser);
fn = error_mark_node;
}
else if (DECL_INITIAL (current_function_decl) != error_mark_node)
{
cp_parser_skip_to_end_of_block_or_statement (parser);
fn = current_function_decl;
current_function_decl = NULL_TREE;
if (current_class_name)
pop_nested_class ();
}
else
{
timevar_id_t tv;
if (DECL_DECLARED_INLINE_P (current_function_decl))
tv = TV_PARSE_INLINE;
else
tv = TV_PARSE_FUNC;
timevar_push (tv);
fn = cp_parser_function_definition_after_declarator (parser,
false);
timevar_pop (tv);
}
return fn;
}
static tree
cp_parser_function_definition_after_declarator (cp_parser* parser,
bool inline_p)
{
tree fn;
bool saved_in_unbraced_linkage_specification_p;
bool saved_in_function_body;
unsigned saved_num_template_parameter_lists;
cp_token *token;
bool fully_implicit_function_template_p
= parser->fully_implicit_function_template_p;
parser->fully_implicit_function_template_p = false;
tree implicit_template_parms
= parser->implicit_template_parms;
parser->implicit_template_parms = 0;
cp_binding_level* implicit_template_scope
= parser->implicit_template_scope;
parser->implicit_template_scope = 0;
saved_in_function_body = parser->in_function_body;
parser->in_function_body = true;
token = cp_lexer_peek_token (parser->lexer);
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_RETURN))
{
cp_lexer_consume_token (parser->lexer);
cp_parser_identifier (parser);
error_at (token->location,
"named return values are no longer supported");
while (true)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_OPEN_BRACE
|| token->type == CPP_EOF
|| token->type == CPP_PRAGMA_EOL)
break;
cp_lexer_consume_token (parser->lexer);
}
}
saved_in_unbraced_linkage_specification_p
= parser->in_unbraced_linkage_specification_p;
parser->in_unbraced_linkage_specification_p = false;
saved_num_template_parameter_lists
= parser->num_template_parameter_lists;
parser->num_template_parameter_lists = 0;
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TRANSACTION_ATOMIC))
cp_parser_function_transaction (parser, RID_TRANSACTION_ATOMIC);
else if (cp_lexer_next_token_is_keyword (parser->lexer,
RID_TRANSACTION_RELAXED))
cp_parser_function_transaction (parser, RID_TRANSACTION_RELAXED);
else if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TRY))
cp_parser_function_try_block (parser);
else
cp_parser_ctor_initializer_opt_and_function_body
(parser, false);
fn = finish_function (inline_p);
expand_or_defer_fn (fn);
parser->in_unbraced_linkage_specification_p
= saved_in_unbraced_linkage_specification_p;
parser->num_template_parameter_lists
= saved_num_template_parameter_lists;
parser->in_function_body = saved_in_function_body;
parser->fully_implicit_function_template_p
= fully_implicit_function_template_p;
parser->implicit_template_parms
= implicit_template_parms;
parser->implicit_template_scope
= implicit_template_scope;
if (parser->fully_implicit_function_template_p)
finish_fully_implicit_template (parser, 0);
return fn;
}
static void
cp_parser_template_declaration_after_parameters (cp_parser* parser,
tree parameter_list,
bool member_p)
{
tree decl = NULL_TREE;
bool friend_p = false;
++parser->num_template_parameter_lists;
vec<deferred_access_check, va_gc> *checks = get_deferred_access_checks ();
if (cp_parser_template_declaration_after_export (parser, member_p))
;
else if (cxx_dialect >= cxx11
&& cp_lexer_next_token_is_keyword (parser->lexer, RID_USING))
decl = cp_parser_alias_declaration (parser);
else
{
push_deferring_access_checks (dk_no_check);
cp_token *token = cp_lexer_peek_token (parser->lexer);
decl = cp_parser_single_declaration (parser,
checks,
member_p,
false,
&friend_p);
pop_deferring_access_checks ();
if (member_p && !friend_p && decl)
{
if (TREE_CODE (decl) == TYPE_DECL)
cp_parser_check_access_in_redeclaration (decl, token->location);
decl = finish_member_template_decl (decl);
}
else if (friend_p && decl
&& DECL_DECLARES_TYPE_P (decl))
make_friend_class (current_class_type, TREE_TYPE (decl),
true);
}
--parser->num_template_parameter_lists;
pop_deferring_access_checks ();
finish_template_decl (parameter_list);
if (decl
&& DECL_DECLARES_FUNCTION_P (decl)
&& UDLIT_OPER_P (DECL_NAME (decl)))
{
bool ok = true;
if (parameter_list == NULL_TREE)
ok = false;
else
{
int num_parms = TREE_VEC_LENGTH (parameter_list);
if (num_parms == 1)
{
tree parm_list = TREE_VEC_ELT (parameter_list, 0);
tree parm = INNERMOST_TEMPLATE_PARMS (parm_list);
if (TREE_TYPE (parm) != char_type_node
|| !TEMPLATE_PARM_PARAMETER_PACK (DECL_INITIAL (parm)))
ok = false;
}
else if (num_parms == 2 && cxx_dialect >= cxx14)
{
tree parm_type = TREE_VEC_ELT (parameter_list, 0);
tree type = INNERMOST_TEMPLATE_PARMS (parm_type);
tree parm_list = TREE_VEC_ELT (parameter_list, 1);
tree parm = INNERMOST_TEMPLATE_PARMS (parm_list);
if (parm == error_mark_node
|| TREE_TYPE (parm) != TREE_TYPE (type)
|| !TEMPLATE_PARM_PARAMETER_PACK (DECL_INITIAL (parm)))
ok = false;
}
else
ok = false;
}
if (!ok)
{
if (cxx_dialect >= cxx14)
error ("literal operator template %qD has invalid parameter list."
"  Expected non-type template argument pack <char...>"
" or <typename CharT, CharT...>",
decl);
else
error ("literal operator template %qD has invalid parameter list."
"  Expected non-type template argument pack <char...>",
decl);
}
}
if (member_p && !friend_p && decl && !DECL_CLASS_TEMPLATE_P (decl))
finish_member_declaration (decl);
if (member_p && decl
&& DECL_DECLARES_FUNCTION_P (decl))
vec_safe_push (unparsed_funs_with_definitions, decl);
}
static bool
cp_parser_template_introduction (cp_parser* parser, bool member_p)
{
cp_parser_parse_tentatively (parser);
tree saved_scope = parser->scope;
tree saved_object_scope = parser->object_scope;
tree saved_qualifying_scope = parser->qualifying_scope;
cp_parser_global_scope_opt (parser,
false);
cp_parser_nested_name_specifier_opt (parser,
false,
true,
false,
false);
cp_token *token = cp_lexer_peek_token (parser->lexer);
tree concept_name = cp_parser_identifier (parser);
tree tmpl_decl = cp_parser_lookup_name_simple (parser, concept_name,
token->location);
parser->scope = saved_scope;
parser->object_scope = saved_object_scope;
parser->qualifying_scope = saved_qualifying_scope;
if (concept_name == error_mark_node)
cp_parser_simulate_error (parser);
matching_braces braces;
braces.require_open (parser);
if (!cp_parser_parse_definitely (parser))
return false;
push_deferring_access_checks (dk_deferred);
tree introduction_list = cp_parser_introduction_list (parser);
int nargs = TREE_VEC_LENGTH (introduction_list);
if (nargs == 0)
{
error ("empty introduction-list");
return true;
}
if (!braces.require_close (parser))
return true;
if (tmpl_decl == error_mark_node)
{
cp_parser_name_lookup_error (parser, concept_name, tmpl_decl, NLE_NULL,
token->location);
return true;
}
tree parms = finish_template_introduction (tmpl_decl, introduction_list);
if (parms && parms != error_mark_node)
{
cp_parser_template_declaration_after_parameters (parser, parms,
member_p);
return true;
}
error_at (token->location, "no matching concept for template-introduction");
return true;
}
static void
cp_parser_explicit_template_declaration (cp_parser* parser, bool member_p)
{
tree parameter_list;
bool need_lang_pop;
location_t location = input_location;
if (!cp_parser_require (parser, CPP_LESS, RT_LESS))
return;
if (at_class_scope_p () && current_function_decl)
{
error_at (location,
"invalid declaration of member template in local class");
cp_parser_skip_to_end_of_block_or_statement (parser);
return;
}
if (current_lang_name == lang_name_c)
{
error_at (location, "template with C linkage");
maybe_show_extern_c_location ();
push_lang_context (lang_name_cplusplus);
need_lang_pop = true;
}
else
need_lang_pop = false;
push_deferring_access_checks (dk_deferred);
if (cp_lexer_next_token_is (parser->lexer, CPP_GREATER))
{
cp_parser_error (parser, "invalid explicit specialization");
begin_specialization ();
parameter_list = NULL_TREE;
}
else
{
parameter_list = cp_parser_template_parameter_list (parser);
}
cp_parser_skip_to_end_of_template_parameter_list (parser);
if (flag_concepts)
{
tree reqs = get_shorthand_constraints (current_template_parms);
if (tree r = cp_parser_requires_clause_opt (parser))
reqs = conjoin_constraints (reqs, normalize_expression (r));
TEMPLATE_PARMS_CONSTRAINTS (current_template_parms) = reqs;
}
cp_parser_template_declaration_after_parameters (parser, parameter_list,
member_p);
if (need_lang_pop)
pop_lang_context ();
}
static bool
cp_parser_template_declaration_after_export (cp_parser* parser, bool member_p)
{
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TEMPLATE))
{
cp_lexer_consume_token (parser->lexer);
cp_parser_explicit_template_declaration (parser, member_p);
return true;
}
else if (flag_concepts)
return cp_parser_template_introduction (parser, member_p);
return false;
}
static void
cp_parser_perform_template_parameter_access_checks (vec<deferred_access_check, va_gc> *checks)
{
++processing_template_parmlist;
perform_access_checks (checks, tf_warning_or_error);
--processing_template_parmlist;
}
static tree
cp_parser_single_declaration (cp_parser* parser,
vec<deferred_access_check, va_gc> *checks,
bool member_p,
bool explicit_specialization_p,
bool* friend_p)
{
int declares_class_or_enum;
tree decl = NULL_TREE;
cp_decl_specifier_seq decl_specifiers;
bool function_definition_p = false;
cp_token *decl_spec_token_start;
gcc_assert (innermost_scope_kind () == sk_template_parms
|| innermost_scope_kind () == sk_template_spec);
push_deferring_access_checks (dk_deferred);
decl_spec_token_start = cp_lexer_peek_token (parser->lexer);
cp_parser_decl_specifier_seq (parser,
CP_PARSER_FLAGS_OPTIONAL,
&decl_specifiers,
&declares_class_or_enum);
if (friend_p)
*friend_p = cp_parser_friend_p (&decl_specifiers);
if (decl_spec_seq_has_spec_p (&decl_specifiers, ds_typedef))
{
error_at (decl_spec_token_start->location,
"template declaration of %<typedef%>");
decl = error_mark_node;
}
stop_deferring_access_checks ();
if (declares_class_or_enum)
{
if (cp_parser_declares_only_class_p (parser)
|| (declares_class_or_enum & 2))
{
if (declares_class_or_enum == 1)
associate_classtype_constraints (decl_specifiers.type);
decl = shadow_tag (&decl_specifiers);
if (friend_p && *friend_p
&& !decl
&& decl_specifiers.type
&& TYPE_P (decl_specifiers.type))
decl = decl_specifiers.type;
if (decl && decl != error_mark_node)
decl = TYPE_NAME (decl);
else
decl = error_mark_node;
cp_parser_perform_template_parameter_access_checks (checks);
if (!cp_parser_declares_only_class_p (parser)
&& !seen_error ())
{
error_at (cp_lexer_peek_token (parser->lexer)->location,
"a class template declaration must not declare "
"anything else");
cp_parser_skip_to_end_of_block_or_statement (parser);
goto out;
}
}
}
if (!decl_specifiers.any_type_specifiers_p
&& cp_parser_parse_and_diagnose_invalid_type_name (parser))
{
decl = error_mark_node;
goto out;
}
if (!decl
&& (cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON)
|| decl_specifiers.type != error_mark_node))
{
decl = cp_parser_init_declarator (parser,
&decl_specifiers,
checks,
true,
member_p,
declares_class_or_enum,
&function_definition_p,
NULL, NULL, NULL);
if (decl
&& explicit_specialization_p
&& decl_specifiers.storage_class != sc_none)
{
error_at (decl_spec_token_start->location,
"explicit template specialization cannot have a storage class");
decl = error_mark_node;
}
if (decl && VAR_P (decl))
check_template_variable (decl);
}
if (!function_definition_p
&& (decl == error_mark_node
|| !cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON)))
cp_parser_skip_to_end_of_block_or_statement (parser);
out:
pop_deferring_access_checks ();
parser->scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
parser->object_scope = NULL_TREE;
return decl;
}
static cp_expr
cp_parser_simple_cast_expression (cp_parser *parser)
{
return cp_parser_cast_expression (parser, false,
false, false, NULL);
}
static cp_expr
cp_parser_functional_cast (cp_parser* parser, tree type)
{
vec<tree, va_gc> *vec;
tree expression_list;
cp_expr cast;
bool nonconst_p;
location_t start_loc = input_location;
if (!type)
type = error_mark_node;
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
cp_lexer_set_source_position (parser->lexer);
maybe_warn_cpp0x (CPP0X_INITIALIZER_LISTS);
expression_list = cp_parser_braced_list (parser, &nonconst_p);
CONSTRUCTOR_IS_DIRECT_INIT (expression_list) = 1;
if (TREE_CODE (type) == TYPE_DECL)
type = TREE_TYPE (type);
cast = finish_compound_literal (type, expression_list,
tf_warning_or_error, fcl_functional);
location_t finish_loc
= get_finish (cp_lexer_previous_token (parser->lexer)->location);
location_t combined_loc = make_location (start_loc, start_loc,
finish_loc);
cast.set_location (combined_loc);
return cast;
}
vec = cp_parser_parenthesized_expression_list (parser, non_attr,
true,
true,
NULL);
if (vec == NULL)
expression_list = error_mark_node;
else
{
expression_list = build_tree_list_vec (vec);
release_tree_vector (vec);
}
cast = build_functional_cast (type, expression_list,
tf_warning_or_error);
if (TREE_CODE (type) == TYPE_DECL)
type = TREE_TYPE (type);
if (cast != error_mark_node
&& !cast_valid_in_integral_constant_expression_p (type)
&& cp_parser_non_integral_constant_expression (parser,
NIC_CONSTRUCTOR))
return error_mark_node;
location_t finish_loc
= get_finish (cp_lexer_previous_token (parser->lexer)->location);
location_t combined_loc = make_location (start_loc, start_loc, finish_loc);
cast.set_location (combined_loc);
return cast;
}
static tree
cp_parser_save_member_function_body (cp_parser* parser,
cp_decl_specifier_seq *decl_specifiers,
cp_declarator *declarator,
tree attributes)
{
cp_token *first;
cp_token *last;
tree fn;
bool function_try_block = false;
fn = grokmethod (decl_specifiers, declarator, attributes);
cp_finalize_omp_declare_simd (parser, fn);
cp_finalize_oacc_routine (parser, fn, true);
if (fn == error_mark_node)
{
if (cp_parser_token_starts_function_definition_p
(cp_lexer_peek_token (parser->lexer)))
cp_parser_skip_to_end_of_block_or_statement (parser);
return error_mark_node;
}
cp_parser_save_default_args (parser, fn);
first = parser->lexer->next_token;
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TRANSACTION_RELAXED))
cp_lexer_consume_token (parser->lexer);
else if (cp_lexer_next_token_is_keyword (parser->lexer,
RID_TRANSACTION_ATOMIC))
{
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_SQUARE)
&& cp_lexer_nth_token_is (parser->lexer, 2, CPP_OPEN_SQUARE)
&& (cp_lexer_nth_token_is (parser->lexer, 3, CPP_NAME)
|| cp_lexer_nth_token_is (parser->lexer, 3, CPP_KEYWORD))
&& cp_lexer_nth_token_is (parser->lexer, 4, CPP_CLOSE_SQUARE)
&& cp_lexer_nth_token_is (parser->lexer, 5, CPP_CLOSE_SQUARE))
{
cp_lexer_consume_token (parser->lexer);
cp_lexer_consume_token (parser->lexer);
cp_lexer_consume_token (parser->lexer);
cp_lexer_consume_token (parser->lexer);
cp_lexer_consume_token (parser->lexer);
}
else
while (cp_next_tokens_can_be_gnu_attribute_p (parser)
&& cp_lexer_nth_token_is (parser->lexer, 2, CPP_OPEN_PAREN))
{
cp_lexer_consume_token (parser->lexer);
if (cp_parser_cache_group (parser, CPP_CLOSE_PAREN, 0))
break;
}
}
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TRY))
{
cp_lexer_consume_token (parser->lexer);
function_try_block = true;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_COLON))
{
cp_lexer_consume_token (parser->lexer);
while (cp_lexer_next_token_is_not (parser->lexer, CPP_OPEN_BRACE))
{
if (cp_parser_cache_group (parser, CPP_CLOSE_PAREN, 0))
break;
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
cp_lexer_consume_token (parser->lexer);
}
}
cp_parser_cache_group (parser, CPP_CLOSE_BRACE, 0);
if (function_try_block)
while (cp_lexer_next_token_is_keyword (parser->lexer, RID_CATCH))
cp_parser_cache_group (parser, CPP_CLOSE_BRACE, 0);
last = parser->lexer->next_token;
DECL_PENDING_INLINE_INFO (fn) = cp_token_cache_new (first, last);
DECL_PENDING_INLINE_P (fn) = 1;
DECL_INITIALIZED_IN_CLASS_P (fn) = 1;
vec_safe_push (unparsed_funs_with_definitions, fn);
return fn;
}
static tree
cp_parser_save_nsdmi (cp_parser* parser)
{
return cp_parser_cache_defarg (parser, true);
}
static tree
cp_parser_enclosed_template_argument_list (cp_parser* parser)
{
tree arguments;
tree saved_scope;
tree saved_qualifying_scope;
tree saved_object_scope;
bool saved_greater_than_is_operator_p;
int saved_unevaluated_operand;
int saved_inhibit_evaluation_warnings;
saved_greater_than_is_operator_p
= parser->greater_than_is_operator_p;
parser->greater_than_is_operator_p = false;
saved_scope = parser->scope;
saved_qualifying_scope = parser->qualifying_scope;
saved_object_scope = parser->object_scope;
saved_unevaluated_operand = cp_unevaluated_operand;
cp_unevaluated_operand = 0;
saved_inhibit_evaluation_warnings = c_inhibit_evaluation_warnings;
c_inhibit_evaluation_warnings = 0;
if (cp_lexer_next_token_is (parser->lexer, CPP_GREATER)
|| cp_lexer_next_token_is (parser->lexer, CPP_RSHIFT))
arguments = NULL_TREE;
else
arguments = cp_parser_template_argument_list (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_RSHIFT))
{
if (cxx_dialect != cxx98)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
token->type = CPP_GREATER;
}
else if (!saved_greater_than_is_operator_p)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
gcc_rich_location richloc (token->location);
richloc.add_fixit_replace ("> >");
error_at (&richloc, "%<>>%> should be %<> >%> "
"within a nested template argument list");
token->type = CPP_GREATER;
}
else
{
cp_token *token = cp_lexer_consume_token (parser->lexer);
error_at (token->location,
"spurious %<>>%>, use %<>%> to terminate "
"a template argument list");
}
}
else
cp_parser_skip_to_end_of_template_parameter_list (parser);
parser->greater_than_is_operator_p
= saved_greater_than_is_operator_p;
parser->scope = saved_scope;
parser->qualifying_scope = saved_qualifying_scope;
parser->object_scope = saved_object_scope;
cp_unevaluated_operand = saved_unevaluated_operand;
c_inhibit_evaluation_warnings = saved_inhibit_evaluation_warnings;
return arguments;
}
static void
cp_parser_late_parsing_for_member (cp_parser* parser, tree member_function)
{
timevar_push (TV_PARSE_INMETH);
if (DECL_FUNCTION_TEMPLATE_P (member_function))
member_function = DECL_TEMPLATE_RESULT (member_function);
gcc_assert (parser->num_classes_being_defined == 0);
push_unparsed_function_queues (parser);
maybe_begin_member_template_processing (member_function);
if (DECL_PENDING_INLINE_P (member_function))
{
tree function_scope;
cp_token_cache *tokens;
tokens = DECL_PENDING_INLINE_INFO (member_function);
DECL_PENDING_INLINE_INFO (member_function) = NULL;
DECL_PENDING_INLINE_P (member_function) = 0;
function_scope = current_function_decl;
if (function_scope)
push_function_context ();
cp_parser_push_lexer_for_tokens (parser, tokens);
start_preparsed_function (member_function, NULL_TREE,
SF_PRE_PARSED | SF_INCLASS_INLINE);
if (processing_template_decl)
push_deferring_access_checks (dk_no_check);
if (DECL_OMP_DECLARE_REDUCTION_P (member_function))
{
parser->lexer->in_pragma = true;
cp_parser_omp_declare_reduction_exprs (member_function, parser);
finish_function (true);
cp_check_omp_declare_reduction (member_function);
}
else
cp_parser_function_definition_after_declarator (parser,
true);
if (processing_template_decl)
pop_deferring_access_checks ();
if (function_scope)
pop_function_context ();
cp_parser_pop_lexer (parser);
}
maybe_end_member_template_processing ();
pop_unparsed_function_queues (parser);
timevar_pop (TV_PARSE_INMETH);
}
static void
cp_parser_save_default_args (cp_parser* parser, tree decl)
{
tree probe;
for (probe = TYPE_ARG_TYPES (TREE_TYPE (decl));
probe;
probe = TREE_CHAIN (probe))
if (TREE_PURPOSE (probe))
{
cp_default_arg_entry entry = {current_class_type, decl};
vec_safe_push (unparsed_funs_with_default_args, entry);
break;
}
}
static tree
cp_parser_late_parse_one_default_arg (cp_parser *parser, tree decl,
tree default_arg, tree parmtype)
{
cp_token_cache *tokens;
tree parsed_arg;
bool dummy;
if (default_arg == error_mark_node)
return error_mark_node;
tokens = DEFARG_TOKENS (default_arg);
cp_parser_push_lexer_for_tokens (parser, tokens);
start_lambda_scope (decl);
parsed_arg = cp_parser_initializer (parser, &dummy, &dummy);
if (BRACE_ENCLOSED_INITIALIZER_P (parsed_arg))
maybe_warn_cpp0x (CPP0X_INITIALIZER_LISTS);
finish_lambda_scope ();
if (parsed_arg == error_mark_node)
cp_parser_skip_to_end_of_statement (parser);
if (!processing_template_decl)
{
if (TREE_CODE (decl) == PARM_DECL)
parsed_arg = check_default_argument (parmtype, parsed_arg,
tf_warning_or_error);
else if (maybe_reject_flexarray_init (decl, parsed_arg))
parsed_arg = error_mark_node;
else
parsed_arg = digest_nsdmi_init (decl, parsed_arg, tf_warning_or_error);
}
if (!cp_lexer_next_token_is (parser->lexer, CPP_EOF))
{
if (TREE_CODE (decl) == PARM_DECL)
cp_parser_error (parser, "expected %<,%>");
else
cp_parser_error (parser, "expected %<;%>");
}
cp_parser_pop_lexer (parser);
return parsed_arg;
}
static void
cp_parser_late_parsing_nsdmi (cp_parser *parser, tree field)
{
tree def;
maybe_begin_member_template_processing (field);
push_unparsed_function_queues (parser);
def = cp_parser_late_parse_one_default_arg (parser, field,
DECL_INITIAL (field),
NULL_TREE);
pop_unparsed_function_queues (parser);
maybe_end_member_template_processing ();
DECL_INITIAL (field) = def;
}
static void
cp_parser_late_parsing_default_args (cp_parser *parser, tree fn)
{
bool saved_local_variables_forbidden_p;
tree parm, parmdecl;
push_unparsed_function_queues (parser);
saved_local_variables_forbidden_p = parser->local_variables_forbidden_p;
parser->local_variables_forbidden_p = true;
push_defarg_context (fn);
for (parm = TYPE_ARG_TYPES (TREE_TYPE (fn)),
parmdecl = DECL_ARGUMENTS (fn);
parm && parm != void_list_node;
parm = TREE_CHAIN (parm),
parmdecl = DECL_CHAIN (parmdecl))
{
tree default_arg = TREE_PURPOSE (parm);
tree parsed_arg;
vec<tree, va_gc> *insts;
tree copy;
unsigned ix;
if (!default_arg)
continue;
if (TREE_CODE (default_arg) != DEFAULT_ARG)
continue;
parsed_arg
= cp_parser_late_parse_one_default_arg (parser, parmdecl,
default_arg,
TREE_VALUE (parm));
TREE_PURPOSE (parm) = parsed_arg;
for (insts = DEFARG_INSTANTIATIONS (default_arg), ix = 0;
vec_safe_iterate (insts, ix, &copy); ix++)
TREE_PURPOSE (copy) = parsed_arg;
}
pop_defarg_context ();
check_default_args (fn);
parser->local_variables_forbidden_p = saved_local_variables_forbidden_p;
pop_unparsed_function_queues (parser);
}
static tree
cp_parser_sizeof_pack (cp_parser *parser)
{
cp_lexer_consume_token (parser->lexer);
maybe_warn_variadic_templates ();
matching_parens parens;
bool paren = cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN);
if (paren)
parens.consume_open (parser);
else
permerror (cp_lexer_peek_token (parser->lexer)->location,
"%<sizeof...%> argument must be surrounded by parentheses");
cp_token *token = cp_lexer_peek_token (parser->lexer);
tree name = cp_parser_identifier (parser);
if (name == error_mark_node)
return error_mark_node;
parser->scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
parser->object_scope = NULL_TREE;
tree expr = cp_parser_lookup_name_simple (parser, name, token->location);
if (expr == error_mark_node)
cp_parser_name_lookup_error (parser, name, expr, NLE_NULL,
token->location);
if (TREE_CODE (expr) == TYPE_DECL || TREE_CODE (expr) == TEMPLATE_DECL)
expr = TREE_TYPE (expr);
else if (TREE_CODE (expr) == CONST_DECL)
expr = DECL_INITIAL (expr);
expr = make_pack_expansion (expr);
PACK_EXPANSION_SIZEOF_P (expr) = true;
if (paren)
parens.require_close (parser);
return expr;
}
static tree
cp_parser_sizeof_operand (cp_parser* parser, enum rid keyword)
{
tree expr = NULL_TREE;
const char *saved_message;
char *tmp;
bool saved_integral_constant_expression_p;
bool saved_non_integral_constant_expression_p;
if (keyword == RID_SIZEOF
&& cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
return cp_parser_sizeof_pack (parser);
saved_message = parser->type_definition_forbidden_message;
tmp = concat ("types may not be defined in %<",
IDENTIFIER_POINTER (ridpointers[keyword]),
"%> expressions", NULL);
parser->type_definition_forbidden_message = tmp;
saved_integral_constant_expression_p
= parser->integral_constant_expression_p;
saved_non_integral_constant_expression_p
= parser->non_integral_constant_expression_p;
parser->integral_constant_expression_p = false;
++cp_unevaluated_operand;
++c_inhibit_evaluation_warnings;
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
tree type = NULL_TREE;
cp_parser_parse_tentatively (parser);
matching_parens parens;
parens.consume_open (parser);
if (cp_parser_compound_literal_p (parser))
cp_parser_simulate_error (parser);
else
{
bool saved_in_type_id_in_expr_p = parser->in_type_id_in_expr_p;
parser->in_type_id_in_expr_p = true;
type = cp_parser_type_id (parser);
parens.require_close (parser);
parser->in_type_id_in_expr_p = saved_in_type_id_in_expr_p;
}
if (cp_parser_parse_definitely (parser))
{
cp_decl_specifier_seq decl_specs;
clear_decl_specs (&decl_specs);
decl_specs.type = type;
expr = grokdeclarator (NULL,
&decl_specs,
TYPENAME,
0,
NULL);
}
}
if (!expr)
expr = cp_parser_unary_expression (parser);
--cp_unevaluated_operand;
--c_inhibit_evaluation_warnings;
free (tmp);
parser->type_definition_forbidden_message = saved_message;
parser->integral_constant_expression_p
= saved_integral_constant_expression_p;
parser->non_integral_constant_expression_p
= saved_non_integral_constant_expression_p;
return expr;
}
static bool
cp_parser_declares_only_class_p (cp_parser *parser)
{
return (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON)
|| cp_lexer_next_token_is (parser->lexer, CPP_COMMA));
}
static void
cp_parser_set_storage_class (cp_parser *parser,
cp_decl_specifier_seq *decl_specs,
enum rid keyword,
cp_token *token)
{
cp_storage_class storage_class;
if (parser->in_unbraced_linkage_specification_p)
{
error_at (token->location, "invalid use of %qD in linkage specification",
ridpointers[keyword]);
return;
}
else if (decl_specs->storage_class != sc_none)
{
decl_specs->conflicting_specifiers_p = true;
return;
}
if ((keyword == RID_EXTERN || keyword == RID_STATIC)
&& decl_spec_seq_has_spec_p (decl_specs, ds_thread)
&& decl_specs->gnu_thread_keyword_p)
{
pedwarn (decl_specs->locations[ds_thread], 0,
"%<__thread%> before %qD", ridpointers[keyword]);
}
switch (keyword)
{
case RID_AUTO:
storage_class = sc_auto;
break;
case RID_REGISTER:
storage_class = sc_register;
break;
case RID_STATIC:
storage_class = sc_static;
break;
case RID_EXTERN:
storage_class = sc_extern;
break;
case RID_MUTABLE:
storage_class = sc_mutable;
break;
default:
gcc_unreachable ();
}
decl_specs->storage_class = storage_class;
set_and_check_decl_spec_loc (decl_specs, ds_storage_class, token);
if (decl_spec_seq_has_spec_p (decl_specs, ds_typedef))
decl_specs->conflicting_specifiers_p = true;
}
static void
cp_parser_set_decl_spec_type (cp_decl_specifier_seq *decl_specs,
tree type_spec,
cp_token *token,
bool type_definition_p)
{
decl_specs->any_specifiers_p = true;
if (decl_spec_seq_has_spec_p (decl_specs, ds_typedef)
&& !type_definition_p
&& (type_spec == boolean_type_node
|| type_spec == char16_type_node
|| type_spec == char32_type_node
|| type_spec == wchar_type_node)
&& (decl_specs->type
|| decl_spec_seq_has_spec_p (decl_specs, ds_long)
|| decl_spec_seq_has_spec_p (decl_specs, ds_short)
|| decl_spec_seq_has_spec_p (decl_specs, ds_unsigned)
|| decl_spec_seq_has_spec_p (decl_specs, ds_signed)))
{
decl_specs->redefined_builtin_type = type_spec;
set_and_check_decl_spec_loc (decl_specs,
ds_redefined_builtin_type_spec,
token);
if (!decl_specs->type)
{
decl_specs->type = type_spec;
decl_specs->type_definition_p = false;
set_and_check_decl_spec_loc (decl_specs,ds_type_spec, token);
}
}
else if (decl_specs->type)
decl_specs->multiple_types_p = true;
else
{
decl_specs->type = type_spec;
decl_specs->type_definition_p = type_definition_p;
decl_specs->redefined_builtin_type = NULL_TREE;
set_and_check_decl_spec_loc (decl_specs, ds_type_spec, token);
}
}
static bool
token_is__thread (cp_token *token)
{
gcc_assert (token->keyword == RID_THREAD);
return id_equal (token->u.value, "__thread");
}
static void
set_and_check_decl_spec_loc (cp_decl_specifier_seq *decl_specs,
cp_decl_spec ds, cp_token *token)
{
gcc_assert (ds < ds_last);
if (decl_specs == NULL)
return;
source_location location = token->location;
if (decl_specs->locations[ds] == 0)
{
decl_specs->locations[ds] = location;
if (ds == ds_thread)
decl_specs->gnu_thread_keyword_p = token_is__thread (token);
}
else
{
if (ds == ds_long)
{
if (decl_specs->locations[ds_long_long] != 0)
error_at (location,
"%<long long long%> is too long for GCC");
else
{
decl_specs->locations[ds_long_long] = location;
pedwarn_cxx98 (location,
OPT_Wlong_long, 
"ISO C++ 1998 does not support %<long long%>");
}
}
else if (ds == ds_thread)
{
bool gnu = token_is__thread (token);
if (gnu != decl_specs->gnu_thread_keyword_p)
error_at (location,
"both %<__thread%> and %<thread_local%> specified");
else
{
gcc_rich_location richloc (location);
richloc.add_fixit_remove ();
error_at (&richloc, "duplicate %qD", token->u.value);
}
}
else
{
static const char *const decl_spec_names[] = {
"signed",
"unsigned",
"short",
"long",
"const",
"volatile",
"restrict",
"inline",
"virtual",
"explicit",
"friend",
"typedef",
"using",
"constexpr",
"__complex"
};
gcc_rich_location richloc (location);
richloc.add_fixit_remove ();
error_at (&richloc, "duplicate %qs", decl_spec_names[ds]);
}
}
}
bool
decl_spec_seq_has_spec_p (const cp_decl_specifier_seq * decl_specs,
cp_decl_spec ds)
{
gcc_assert (ds < ds_last);
if (decl_specs == NULL)
return false;
return decl_specs->locations[ds] != 0;
}
static bool
cp_parser_friend_p (const cp_decl_specifier_seq *decl_specifiers)
{
return decl_spec_seq_has_spec_p (decl_specifiers, ds_friend);
}
static void
cp_parser_required_error (cp_parser *parser,
required_token token_desc,
bool keyword,
location_t matching_location)
{
if (cp_parser_simulate_error (parser))
return;
const char *gmsgid = NULL;
switch (token_desc)
{
case RT_NEW:
gmsgid = G_("expected %<new%>");
break;
case RT_DELETE:
gmsgid = G_("expected %<delete%>");
break;
case RT_RETURN:
gmsgid = G_("expected %<return%>");
break;
case RT_WHILE:
gmsgid = G_("expected %<while%>");
break;
case RT_EXTERN:
gmsgid = G_("expected %<extern%>");
break;
case RT_STATIC_ASSERT:
gmsgid = G_("expected %<static_assert%>");
break;
case RT_DECLTYPE:
gmsgid = G_("expected %<decltype%>");
break;
case RT_OPERATOR:
gmsgid = G_("expected %<operator%>");
break;
case RT_CLASS:
gmsgid = G_("expected %<class%>");
break;
case RT_TEMPLATE:
gmsgid = G_("expected %<template%>");
break;
case RT_NAMESPACE:
gmsgid = G_("expected %<namespace%>");
break;
case RT_USING:
gmsgid = G_("expected %<using%>");
break;
case RT_ASM:
gmsgid = G_("expected %<asm%>");
break;
case RT_TRY:
gmsgid = G_("expected %<try%>");
break;
case RT_CATCH:
gmsgid = G_("expected %<catch%>");
break;
case RT_THROW:
gmsgid = G_("expected %<throw%>");
break;
case RT_LABEL:
gmsgid = G_("expected %<__label__%>");
break;
case RT_AT_TRY:
gmsgid = G_("expected %<@try%>");
break;
case RT_AT_SYNCHRONIZED:
gmsgid = G_("expected %<@synchronized%>");
break;
case RT_AT_THROW:
gmsgid = G_("expected %<@throw%>");
break;
case RT_TRANSACTION_ATOMIC:
gmsgid = G_("expected %<__transaction_atomic%>");
break;
case RT_TRANSACTION_RELAXED:
gmsgid = G_("expected %<__transaction_relaxed%>");
break;
default:
break;
}
if (!gmsgid && !keyword)
{
switch (token_desc)
{
case RT_SEMICOLON:
gmsgid = G_("expected %<;%>");
break;
case RT_OPEN_PAREN:
gmsgid = G_("expected %<(%>");
break;
case RT_CLOSE_BRACE:
gmsgid = G_("expected %<}%>");
break;
case RT_OPEN_BRACE:
gmsgid = G_("expected %<{%>");
break;
case RT_CLOSE_SQUARE:
gmsgid = G_("expected %<]%>");
break;
case RT_OPEN_SQUARE:
gmsgid = G_("expected %<[%>");
break;
case RT_COMMA:
gmsgid = G_("expected %<,%>");
break;
case RT_SCOPE:
gmsgid = G_("expected %<::%>");
break;
case RT_LESS:
gmsgid = G_("expected %<<%>");
break;
case RT_GREATER:
gmsgid = G_("expected %<>%>");
break;
case RT_EQ:
gmsgid = G_("expected %<=%>");
break;
case RT_ELLIPSIS:
gmsgid = G_("expected %<...%>");
break;
case RT_MULT:
gmsgid = G_("expected %<*%>");
break;
case RT_COMPL:
gmsgid = G_("expected %<~%>");
break;
case RT_COLON:
gmsgid = G_("expected %<:%>");
break;
case RT_COLON_SCOPE:
gmsgid = G_("expected %<:%> or %<::%>");
break;
case RT_CLOSE_PAREN:
gmsgid = G_("expected %<)%>");
break;
case RT_COMMA_CLOSE_PAREN:
gmsgid = G_("expected %<,%> or %<)%>");
break;
case RT_PRAGMA_EOL:
gmsgid = G_("expected end of line");
break;
case RT_NAME:
gmsgid = G_("expected identifier");
break;
case RT_SELECT:
gmsgid = G_("expected selection-statement");
break;
case RT_ITERATION:
gmsgid = G_("expected iteration-statement");
break;
case RT_JUMP:
gmsgid = G_("expected jump-statement");
break;
case RT_CLASS_KEY:
gmsgid = G_("expected class-key");
break;
case RT_CLASS_TYPENAME_TEMPLATE:
gmsgid = G_("expected %<class%>, %<typename%>, or %<template%>");
break;
default:
gcc_unreachable ();
}
}
if (gmsgid)
cp_parser_error_1 (parser, gmsgid, token_desc, matching_location);
}
static cp_token *
cp_parser_require (cp_parser* parser,
enum cpp_ttype type,
required_token token_desc,
location_t matching_location)
{
if (cp_lexer_next_token_is (parser->lexer, type))
return cp_lexer_consume_token (parser->lexer);
else
{
if (!cp_parser_simulate_error (parser))
cp_parser_required_error (parser, token_desc, false,
matching_location);
return NULL;
}
}
static void
cp_parser_skip_to_end_of_template_parameter_list (cp_parser* parser)
{
unsigned level = 0;
unsigned nesting_depth = 0;
if (cp_parser_require (parser, CPP_GREATER, RT_GREATER))
return;
while (true)
{
switch (cp_lexer_peek_token (parser->lexer)->type)
{
case CPP_LESS:
if (!nesting_depth)
++level;
break;
case CPP_RSHIFT:
if (cxx_dialect == cxx98)
break;
else if (!nesting_depth && level-- == 0)
{
cp_lexer_consume_token (parser->lexer);
return;
}
gcc_fallthrough ();
case CPP_GREATER:
if (!nesting_depth && level-- == 0)
{
cp_lexer_consume_token (parser->lexer);
return;
}
break;
case CPP_OPEN_PAREN:
case CPP_OPEN_SQUARE:
++nesting_depth;
break;
case CPP_CLOSE_PAREN:
case CPP_CLOSE_SQUARE:
if (nesting_depth-- == 0)
return;
break;
case CPP_EOF:
case CPP_PRAGMA_EOL:
case CPP_SEMICOLON:
case CPP_OPEN_BRACE:
case CPP_CLOSE_BRACE:
return;
default:
break;
}
cp_lexer_consume_token (parser->lexer);
}
}
static cp_token *
cp_parser_require_keyword (cp_parser* parser,
enum rid keyword,
required_token token_desc)
{
cp_token *token = cp_parser_require (parser, CPP_KEYWORD, token_desc);
if (token && token->keyword != keyword)
{
cp_parser_required_error (parser, token_desc, true,
UNKNOWN_LOCATION);
return NULL;
}
return token;
}
static bool
cp_parser_token_starts_function_definition_p (cp_token* token)
{
return (
token->type == CPP_OPEN_BRACE
|| token->type == CPP_COLON
|| token->keyword == RID_TRY
|| token->keyword == RID_TRANSACTION_ATOMIC
|| token->keyword == RID_TRANSACTION_RELAXED
|| token->keyword == RID_RETURN);
}
static bool
cp_parser_next_token_starts_class_definition_p (cp_parser *parser)
{
cp_token *token;
token = cp_lexer_peek_token (parser->lexer);
return (token->type == CPP_OPEN_BRACE
|| (token->type == CPP_COLON
&& !parser->colon_doesnt_start_class_def_p));
}
static bool
cp_parser_next_token_ends_template_argument_p (cp_parser *parser)
{
cp_token *token;
token = cp_lexer_peek_token (parser->lexer);
return (token->type == CPP_COMMA 
|| token->type == CPP_GREATER
|| token->type == CPP_ELLIPSIS
|| ((cxx_dialect != cxx98) && token->type == CPP_RSHIFT));
}
static bool
cp_parser_nth_token_starts_template_argument_list_p (cp_parser * parser,
size_t n)
{
cp_token *token;
token = cp_lexer_peek_nth_token (parser->lexer, n);
if (token->type == CPP_LESS)
return true;
if (token->type == CPP_OPEN_SQUARE && token->flags & DIGRAPH)
{
cp_token *token2;
token2 = cp_lexer_peek_nth_token (parser->lexer, n+1);
if (token2->type == CPP_COLON && !(token2->flags & PREV_WHITE))
return true;
}
return false;
}
static enum tag_types
cp_parser_token_is_class_key (cp_token* token)
{
switch (token->keyword)
{
case RID_CLASS:
return class_type;
case RID_STRUCT:
return record_type;
case RID_UNION:
return union_type;
default:
return none_type;
}
}
static enum tag_types
cp_parser_token_is_type_parameter_key (cp_token* token)
{
if (!token)
return none_type;
switch (token->keyword)
{
case RID_CLASS:
return class_type;
case RID_TYPENAME:
return typename_type;
default:
return none_type;
}
}
static void
cp_parser_check_class_key (enum tag_types class_key, tree type)
{
if (type == error_mark_node)
return;
if ((TREE_CODE (type) == UNION_TYPE) != (class_key == union_type))
{
if (permerror (input_location, "%qs tag used in naming %q#T",
class_key == union_type ? "union"
: class_key == record_type ? "struct" : "class",
type))
inform (DECL_SOURCE_LOCATION (TYPE_NAME (type)),
"%q#T was previously declared here", type);
}
}
static void
cp_parser_check_access_in_redeclaration (tree decl, location_t location)
{
if (!decl
|| (!CLASS_TYPE_P (TREE_TYPE (decl))
&& TREE_CODE (TREE_TYPE (decl)) != ENUMERAL_TYPE))
return;
if ((TREE_PRIVATE (decl)
!= (current_access_specifier == access_private_node))
|| (TREE_PROTECTED (decl)
!= (current_access_specifier == access_protected_node)))
error_at (location, "%qD redeclared with different access", decl);
}
static bool
cp_parser_optional_template_keyword (cp_parser *parser)
{
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TEMPLATE))
{
if (!processing_template_decl
&& pedantic && cxx_dialect == cxx98)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
pedwarn (token->location, OPT_Wpedantic,
"in C++98 %<template%> (as a disambiguator) is only "
"allowed within templates");
cp_lexer_purge_token (parser->lexer);
return false;
}
else
{
cp_lexer_consume_token (parser->lexer);
return true;
}
}
return false;
}
static void
cp_parser_pre_parsed_nested_name_specifier (cp_parser *parser)
{
struct tree_check *check_value;
check_value = cp_lexer_consume_token (parser->lexer)->u.tree_check_value;
parser->scope = saved_checks_value (check_value);
parser->qualifying_scope = check_value->qualifying_scope;
parser->object_scope = NULL_TREE;
}
static bool
cp_parser_cache_group (cp_parser *parser,
enum cpp_ttype end,
unsigned depth)
{
while (true)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
if ((end == CPP_CLOSE_PAREN || depth == 0)
&& token->type == CPP_SEMICOLON)
return true;
if (token->type == CPP_EOF
|| (end != CPP_PRAGMA_EOL
&& token->type == CPP_PRAGMA_EOL))
return true;
if (token->type == CPP_CLOSE_BRACE && depth == 0)
return true;
cp_lexer_consume_token (parser->lexer);
if (token->type == CPP_OPEN_BRACE)
{
cp_parser_cache_group (parser, CPP_CLOSE_BRACE, depth + 1);
if (depth == 0)
return false;
}
else if (token->type == CPP_OPEN_PAREN)
{
cp_parser_cache_group (parser, CPP_CLOSE_PAREN, depth + 1);
if (depth == 0 && end == CPP_CLOSE_PAREN)
return false;
}
else if (token->type == CPP_PRAGMA)
cp_parser_cache_group (parser, CPP_PRAGMA_EOL, depth + 1);
else if (token->type == end)
return false;
}
}
static tree
cp_parser_cache_defarg (cp_parser *parser, bool nsdmi)
{
unsigned depth = 0;
int maybe_template_id = 0;
cp_token *first_token;
cp_token *token;
tree default_argument;
first_token = cp_lexer_peek_token (parser->lexer);
if (first_token->type == CPP_OPEN_BRACE)
{
cp_parser_cache_group (parser, CPP_CLOSE_BRACE, 0);
token = cp_lexer_peek_token (parser->lexer);
}
else while (true)
{
bool done = false;
token = cp_lexer_peek_token (parser->lexer);
switch (token->type)
{
case CPP_COMMA:
if (depth == 0 && maybe_template_id)
{
bool error = false;
cp_token *peek;
bool saved_italp = parser->in_template_argument_list_p;
parser->in_template_argument_list_p = true;
cp_parser_parse_tentatively (parser);
if (nsdmi)
{
do
{
int ctor_dtor_or_conv_p;
cp_lexer_consume_token (parser->lexer);
cp_parser_declarator (parser, CP_PARSER_DECLARATOR_NAMED,
&ctor_dtor_or_conv_p,
NULL,
true,
false);
peek = cp_lexer_peek_token (parser->lexer);
if (cp_parser_error_occurred (parser))
break;
}
while (peek->type == CPP_COMMA);
error = (peek->type != CPP_EQ
&& peek->type != CPP_SEMICOLON);
}
else
{
cp_lexer_consume_token (parser->lexer);
begin_scope (sk_function_parms, NULL_TREE);
cp_parser_parameter_declaration_list (parser, &error);
pop_bindings_and_leave_scope ();
}
if (!cp_parser_error_occurred (parser) && !error)
done = true;
cp_parser_abort_tentative_parse (parser);
parser->in_template_argument_list_p = saved_italp;
break;
}
case CPP_CLOSE_PAREN:
case CPP_ELLIPSIS:
case CPP_SEMICOLON:
case CPP_CLOSE_BRACE:
case CPP_CLOSE_SQUARE:
if (depth == 0
&& token->type != CPP_ELLIPSIS)
done = true;
else if (token->type == CPP_CLOSE_PAREN
|| token->type == CPP_CLOSE_BRACE
|| token->type == CPP_CLOSE_SQUARE)
--depth;
break;
case CPP_OPEN_PAREN:
case CPP_OPEN_SQUARE:
case CPP_OPEN_BRACE:
++depth;
break;
case CPP_LESS:
if (depth == 0)
++maybe_template_id;
break;
case CPP_RSHIFT:
if (cxx_dialect == cxx98)
break;
gcc_fallthrough ();
case CPP_GREATER:
if (depth == 0)
{
maybe_template_id -= 1 + (token->type == CPP_RSHIFT);
if (maybe_template_id < 0)
maybe_template_id = 0;
}
break;
case CPP_EOF:
case CPP_PRAGMA_EOL:
error_at (token->location, "file ends in default argument");
return error_mark_node;
case CPP_NAME:
case CPP_SCOPE:
break;
default:
break;
}
if (done)
break;
token = cp_lexer_consume_token (parser->lexer);
}
default_argument = make_node (DEFAULT_ARG);
DEFARG_TOKENS (default_argument)
= cp_token_cache_new (first_token, token);
DEFARG_INSTANTIATIONS (default_argument) = NULL;
return default_argument;
}
location_t
defarg_location (tree default_argument)
{
cp_token_cache *tokens = DEFARG_TOKENS (default_argument);
location_t start = tokens->first->location;
location_t end = tokens->last->location;
return make_location (start, start, end);
}
static void
cp_parser_parse_tentatively (cp_parser* parser)
{
parser->context = cp_parser_context_new (parser->context);
cp_lexer_save_tokens (parser->lexer);
push_deferring_access_checks (dk_deferred);
}
static void
cp_parser_commit_to_tentative_parse (cp_parser* parser)
{
cp_parser_context *context;
cp_lexer *lexer;
lexer = parser->lexer;
for (context = parser->context; context->next; context = context->next)
{
if (context->status == CP_PARSER_STATUS_KIND_COMMITTED)
break;
context->status = CP_PARSER_STATUS_KIND_COMMITTED;
while (!cp_lexer_saving_tokens (lexer))
lexer = lexer->next;
cp_lexer_commit_tokens (lexer);
}
}
static void
cp_parser_commit_to_topmost_tentative_parse (cp_parser* parser)
{
cp_parser_context *context = parser->context;
cp_lexer *lexer = parser->lexer;
if (context)
{
if (context->status == CP_PARSER_STATUS_KIND_COMMITTED)
return;
context->status = CP_PARSER_STATUS_KIND_COMMITTED;
while (!cp_lexer_saving_tokens (lexer))
lexer = lexer->next;
cp_lexer_commit_tokens (lexer);
}
}
static void
cp_parser_abort_tentative_parse (cp_parser* parser)
{
gcc_assert (parser->context->status != CP_PARSER_STATUS_KIND_COMMITTED
|| errorcount > 0);
cp_parser_simulate_error (parser);
cp_parser_parse_definitely (parser);
}
static bool
cp_parser_parse_definitely (cp_parser* parser)
{
bool error_occurred;
cp_parser_context *context;
error_occurred = cp_parser_error_occurred (parser);
context = parser->context;
parser->context = context->next;
if (!error_occurred)
{
if (context->status != CP_PARSER_STATUS_KIND_COMMITTED)
cp_lexer_commit_tokens (parser->lexer);
pop_to_parent_deferring_access_checks ();
}
else
{
cp_lexer_rollback_tokens (parser->lexer);
pop_deferring_access_checks ();
}
context->next = cp_parser_context_free_list;
cp_parser_context_free_list = context;
return !error_occurred;
}
static bool
cp_parser_uncommitted_to_tentative_parse_p (cp_parser* parser)
{
return (cp_parser_parsing_tentatively (parser)
&& parser->context->status != CP_PARSER_STATUS_KIND_COMMITTED);
}
static bool
cp_parser_error_occurred (cp_parser* parser)
{
return (cp_parser_parsing_tentatively (parser)
&& parser->context->status == CP_PARSER_STATUS_KIND_ERROR);
}
static bool
cp_parser_allow_gnu_extensions_p (cp_parser* parser)
{
return parser->allow_gnu_extensions_p;
}

static cp_expr
cp_parser_objc_expression (cp_parser* parser)
{
cp_token *kwd = cp_lexer_peek_token (parser->lexer);
switch (kwd->type)
{
case CPP_OPEN_SQUARE:
return cp_parser_objc_message_expression (parser);
case CPP_OBJC_STRING:
kwd = cp_lexer_consume_token (parser->lexer);
return objc_build_string_object (kwd->u.value);
case CPP_KEYWORD:
switch (kwd->keyword)
{
case RID_AT_ENCODE:
return cp_parser_objc_encode_expression (parser);
case RID_AT_PROTOCOL:
return cp_parser_objc_protocol_expression (parser);
case RID_AT_SELECTOR:
return cp_parser_objc_selector_expression (parser);
default:
break;
}
default:
error_at (kwd->location,
"misplaced %<@%D%> Objective-C++ construct",
kwd->u.value);
cp_parser_skip_to_end_of_block_or_statement (parser);
}
return error_mark_node;
}
static tree
cp_parser_objc_message_expression (cp_parser* parser)
{
tree receiver, messageargs;
location_t start_loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);  
receiver = cp_parser_objc_message_receiver (parser);
messageargs = cp_parser_objc_message_args (parser);
location_t end_loc = cp_lexer_peek_token (parser->lexer)->location;
cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE);
tree result = objc_build_message_expr (receiver, messageargs);
location_t combined_loc = make_location (start_loc, start_loc, end_loc);
protected_set_expr_location (result, combined_loc);
return result;
}
static tree
cp_parser_objc_message_receiver (cp_parser* parser)
{
tree rcv;
cp_parser_parse_tentatively (parser);
rcv = cp_parser_expression (parser);
if (cp_parser_parse_definitely (parser))
return rcv;
cp_parser_parse_tentatively (parser);
rcv = cp_parser_simple_type_specifier (parser,
NULL,
CP_PARSER_FLAGS_NONE);
if (cp_parser_parse_definitely (parser))
return objc_get_class_reference (rcv);
cp_parser_error (parser, "objective-c++ message receiver expected");
return error_mark_node;
}
static tree
cp_parser_objc_message_args (cp_parser* parser)
{
tree sel_args = NULL_TREE, addl_args = NULL_TREE;
bool maybe_unary_selector_p = true;
cp_token *token = cp_lexer_peek_token (parser->lexer);
while (cp_parser_objc_selector_p (token->type) || token->type == CPP_COLON)
{
tree selector = NULL_TREE, arg;
if (token->type != CPP_COLON)
selector = cp_parser_objc_selector (parser);
if (maybe_unary_selector_p
&& cp_lexer_next_token_is_not (parser->lexer, CPP_COLON))
return build_tree_list (selector, NULL_TREE);
maybe_unary_selector_p = false;
cp_parser_require (parser, CPP_COLON, RT_COLON);
arg = cp_parser_assignment_expression (parser);
sel_args
= chainon (sel_args,
build_tree_list (selector, arg));
token = cp_lexer_peek_token (parser->lexer);
}
while (token->type == CPP_COMMA)
{
tree arg;
cp_lexer_consume_token (parser->lexer);
arg = cp_parser_assignment_expression (parser);
addl_args
= chainon (addl_args,
build_tree_list (NULL_TREE, arg));
token = cp_lexer_peek_token (parser->lexer);
}
if (sel_args == NULL_TREE && addl_args == NULL_TREE)
{
cp_parser_error (parser, "objective-c++ message argument(s) are expected");
return build_tree_list (error_mark_node, error_mark_node);
}
return build_tree_list (sel_args, addl_args);
}
static cp_expr
cp_parser_objc_encode_expression (cp_parser* parser)
{
tree type;
cp_token *token;
location_t start_loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);  
matching_parens parens;
parens.require_open (parser);
token = cp_lexer_peek_token (parser->lexer);
type = complete_type (cp_parser_type_id (parser));
parens.require_close (parser);
if (!type)
{
error_at (token->location, 
"%<@encode%> must specify a type as an argument");
return error_mark_node;
}
if (dependent_type_p (type))
{
tree value = build_min (AT_ENCODE_EXPR, size_type_node, type);
TREE_READONLY (value) = 1;
return value;
}
location_t combined_loc
= make_location (start_loc, start_loc,
cp_lexer_previous_token (parser->lexer)->location);
return cp_expr (objc_build_encode_expr (type), combined_loc);
}
static tree
cp_parser_objc_defs_expression (cp_parser *parser)
{
tree name;
cp_lexer_consume_token (parser->lexer);  
matching_parens parens;
parens.require_open (parser);
name = cp_parser_identifier (parser);
parens.require_close (parser);
return objc_get_class_ivars (name);
}
static tree
cp_parser_objc_protocol_expression (cp_parser* parser)
{
tree proto;
location_t start_loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);  
matching_parens parens;
parens.require_open (parser);
proto = cp_parser_identifier (parser);
parens.require_close (parser);
location_t combined_loc
= make_location (start_loc, start_loc,
cp_lexer_previous_token (parser->lexer)->location);
tree result = objc_build_protocol_expr (proto);
protected_set_expr_location (result, combined_loc);
return result;
}
static tree
cp_parser_objc_selector_expression (cp_parser* parser)
{
tree sel_seq = NULL_TREE;
bool maybe_unary_selector_p = true;
cp_token *token;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);  
matching_parens parens;
parens.require_open (parser);
token = cp_lexer_peek_token (parser->lexer);
while (cp_parser_objc_selector_p (token->type) || token->type == CPP_COLON
|| token->type == CPP_SCOPE)
{
tree selector = NULL_TREE;
if (token->type != CPP_COLON
|| token->type == CPP_SCOPE)
selector = cp_parser_objc_selector (parser);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COLON)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_SCOPE))
{
if (maybe_unary_selector_p)
{
sel_seq = selector;
goto finish_selector;
}
else
{
cp_parser_error (parser, "expected %<:%>");
}
}
maybe_unary_selector_p = false;
token = cp_lexer_consume_token (parser->lexer);
if (token->type == CPP_SCOPE)
{
sel_seq
= chainon (sel_seq,
build_tree_list (selector, NULL_TREE));
sel_seq
= chainon (sel_seq,
build_tree_list (NULL_TREE, NULL_TREE));
}
else
sel_seq
= chainon (sel_seq,
build_tree_list (selector, NULL_TREE));
token = cp_lexer_peek_token (parser->lexer);
}
finish_selector:
parens.require_close (parser);
location_t combined_loc
= make_location (loc, loc,
cp_lexer_previous_token (parser->lexer)->location);
tree result = objc_build_selector_expr (combined_loc, sel_seq);
protected_set_expr_location (result, combined_loc);
return result;
}
static tree
cp_parser_objc_identifier_list (cp_parser* parser)
{
tree identifier;
tree list;
cp_token *sep;
identifier = cp_parser_identifier (parser);
if (identifier == error_mark_node)
return error_mark_node;      
list = build_tree_list (NULL_TREE, identifier);
sep = cp_lexer_peek_token (parser->lexer);
while (sep->type == CPP_COMMA)
{
cp_lexer_consume_token (parser->lexer);  
identifier = cp_parser_identifier (parser);
if (identifier == error_mark_node)
return list;
list = chainon (list, build_tree_list (NULL_TREE,
identifier));
sep = cp_lexer_peek_token (parser->lexer);
}
return list;
}
static void
cp_parser_objc_alias_declaration (cp_parser* parser)
{
tree alias, orig;
cp_lexer_consume_token (parser->lexer);  
alias = cp_parser_identifier (parser);
orig = cp_parser_identifier (parser);
objc_declare_alias (alias, orig);
cp_parser_consume_semicolon_at_end_of_statement (parser);
}
static void
cp_parser_objc_class_declaration (cp_parser* parser)
{
cp_lexer_consume_token (parser->lexer);  
while (true)
{
tree id;
id = cp_parser_identifier (parser);
if (id == error_mark_node)
break;
objc_declare_class (id);
if (cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
cp_lexer_consume_token (parser->lexer);
else
break;
}
cp_parser_consume_semicolon_at_end_of_statement (parser);
}
static tree
cp_parser_objc_protocol_refs_opt (cp_parser* parser)
{
tree protorefs = NULL_TREE;
if(cp_lexer_next_token_is (parser->lexer, CPP_LESS))
{
cp_lexer_consume_token (parser->lexer);  
protorefs = cp_parser_objc_identifier_list (parser);
cp_parser_require (parser, CPP_GREATER, RT_GREATER);
}
return protorefs;
}
static void
cp_parser_objc_visibility_spec (cp_parser* parser)
{
cp_token *vis = cp_lexer_peek_token (parser->lexer);
switch (vis->keyword)
{
case RID_AT_PRIVATE:
objc_set_visibility (OBJC_IVAR_VIS_PRIVATE);
break;
case RID_AT_PROTECTED:
objc_set_visibility (OBJC_IVAR_VIS_PROTECTED);
break;
case RID_AT_PUBLIC:
objc_set_visibility (OBJC_IVAR_VIS_PUBLIC);
break;
case RID_AT_PACKAGE:
objc_set_visibility (OBJC_IVAR_VIS_PACKAGE);
break;
default:
return;
}
cp_lexer_consume_token (parser->lexer);
}
static inline bool
cp_parser_objc_method_type (cp_parser* parser)
{
if (cp_lexer_consume_token (parser->lexer)->type == CPP_PLUS)
return true;
else
return false;
}
static tree
cp_parser_objc_protocol_qualifiers (cp_parser* parser)
{
tree quals = NULL_TREE, node;
cp_token *token = cp_lexer_peek_token (parser->lexer);
node = token->u.value;
while (node && identifier_p (node)
&& (node == ridpointers [(int) RID_IN]
|| node == ridpointers [(int) RID_OUT]
|| node == ridpointers [(int) RID_INOUT]
|| node == ridpointers [(int) RID_BYCOPY]
|| node == ridpointers [(int) RID_BYREF]
|| node == ridpointers [(int) RID_ONEWAY]))
{
quals = tree_cons (NULL_TREE, node, quals);
cp_lexer_consume_token (parser->lexer);
token = cp_lexer_peek_token (parser->lexer);
node = token->u.value;
}
return quals;
}
static tree
cp_parser_objc_typename (cp_parser* parser)
{
tree type_name = NULL_TREE;
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
tree proto_quals, cp_type = NULL_TREE;
matching_parens parens;
parens.consume_open (parser); 
proto_quals = cp_parser_objc_protocol_qualifiers (parser);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_CLOSE_PAREN))
{
cp_type = cp_parser_type_id (parser);
if (cp_type == error_mark_node)
{
cp_type = NULL_TREE;
cp_parser_skip_to_closing_parenthesis (parser,
true,
false,
false);
}
}
parens.require_close (parser);
type_name = build_tree_list (proto_quals, cp_type);
}
return type_name;
}
static bool
cp_parser_objc_selector_p (enum cpp_ttype type)
{
return (type == CPP_NAME || type == CPP_KEYWORD
|| type == CPP_AND_AND || type == CPP_AND_EQ || type == CPP_AND
|| type == CPP_OR || type == CPP_COMPL || type == CPP_NOT
|| type == CPP_NOT_EQ || type == CPP_OR_OR || type == CPP_OR_EQ
|| type == CPP_XOR || type == CPP_XOR_EQ);
}
static tree
cp_parser_objc_selector (cp_parser* parser)
{
cp_token *token = cp_lexer_consume_token (parser->lexer);
if (!cp_parser_objc_selector_p (token->type))
{
error_at (token->location, "invalid Objective-C++ selector name");
return error_mark_node;
}
switch (token->type)
{
case CPP_AND_AND: return get_identifier ("and");
case CPP_AND_EQ: return get_identifier ("and_eq");
case CPP_AND: return get_identifier ("bitand");
case CPP_OR: return get_identifier ("bitor");
case CPP_COMPL: return get_identifier ("compl");
case CPP_NOT: return get_identifier ("not");
case CPP_NOT_EQ: return get_identifier ("not_eq");
case CPP_OR_OR: return get_identifier ("or");
case CPP_OR_EQ: return get_identifier ("or_eq");
case CPP_XOR: return get_identifier ("xor");
case CPP_XOR_EQ: return get_identifier ("xor_eq");
default: return token->u.value;
}
}
static tree
cp_parser_objc_method_keyword_params (cp_parser* parser, tree* attributes)
{
tree params = NULL_TREE;
bool maybe_unary_selector_p = true;
cp_token *token = cp_lexer_peek_token (parser->lexer);
while (cp_parser_objc_selector_p (token->type) || token->type == CPP_COLON)
{
tree selector = NULL_TREE, type_name, identifier;
tree parm_attr = NULL_TREE;
if (token->keyword == RID_ATTRIBUTE)
break;
if (token->type != CPP_COLON)
selector = cp_parser_objc_selector (parser);
if (maybe_unary_selector_p
&& cp_lexer_next_token_is_not (parser->lexer, CPP_COLON))
{
params = selector; 
break;
}
maybe_unary_selector_p = false;
if (!cp_parser_require (parser, CPP_COLON, RT_COLON))
{
break;
}
type_name = cp_parser_objc_typename (parser);
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_ATTRIBUTE))
parm_attr = cp_parser_attributes_opt (parser);
identifier = cp_parser_identifier (parser);
params
= chainon (params,
objc_build_keyword_decl (selector,
type_name,
identifier,
parm_attr));
token = cp_lexer_peek_token (parser->lexer);
}
if (params == NULL_TREE)
{
cp_parser_error (parser, "objective-c++ method declaration is expected");
return error_mark_node;
}
if (token->keyword == RID_ATTRIBUTE)
{
*attributes = cp_parser_attributes_opt (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON)
|| cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
return params;
cp_parser_error (parser, 
"method attributes must be specified at the end");
return error_mark_node;
}
if (params == NULL_TREE)
{
cp_parser_error (parser, "objective-c++ method declaration is expected");
return error_mark_node;
}
return params;
}
static tree
cp_parser_objc_method_tail_params_opt (cp_parser* parser, bool *ellipsisp, 
tree* attributes)
{
tree params = make_node (TREE_LIST);
cp_token *token = cp_lexer_peek_token (parser->lexer);
*ellipsisp = false;  
while (token->type == CPP_COMMA)
{
cp_parameter_declarator *parmdecl;
tree parm;
cp_lexer_consume_token (parser->lexer);  
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_ELLIPSIS)
{
cp_lexer_consume_token (parser->lexer);  
*ellipsisp = true;
token = cp_lexer_peek_token (parser->lexer);
break;
}
parmdecl = cp_parser_parameter_declaration (parser, false, NULL);
parm = grokdeclarator (parmdecl->declarator,
&parmdecl->decl_specifiers,
PARM, 0,
NULL);
chainon (params, build_tree_list (NULL_TREE, parm));
token = cp_lexer_peek_token (parser->lexer);
}
if (token->keyword == RID_ATTRIBUTE)
{
if (*attributes == NULL_TREE)
{
*attributes = cp_parser_attributes_opt (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON)
|| cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
return params;
}
else        
*attributes = cp_parser_attributes_opt (parser);
cp_parser_error (parser, 
"method attributes must be specified at the end");
return error_mark_node;
}
return params;
}
static void
cp_parser_objc_interstitial_code (cp_parser* parser)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (token->keyword == RID_EXTERN
&& cp_parser_is_pure_string_literal
(cp_lexer_peek_nth_token (parser->lexer, 2)))
cp_parser_linkage_specification (parser);
else if (token->type == CPP_PRAGMA)
cp_parser_pragma (parser, pragma_objc_icode, NULL);
else if (token->type == CPP_SEMICOLON)
cp_lexer_consume_token (parser->lexer);
else if (token->keyword == RID_AT_OPTIONAL)
{
cp_lexer_consume_token (parser->lexer);
objc_set_method_opt (true);
}
else if (token->keyword == RID_AT_REQUIRED)
{
cp_lexer_consume_token (parser->lexer);
objc_set_method_opt (false);
}
else if (token->keyword == RID_NAMESPACE)
cp_parser_namespace_definition (parser);
else if (token->type == CPP_OPEN_BRACE || token->type == CPP_CLOSE_BRACE)
{
cp_lexer_consume_token (parser->lexer);
error ("stray %qs between Objective-C++ methods",
token->type == CPP_OPEN_BRACE ? "{" : "}");
}
else
cp_parser_block_declaration (parser, false);
}
static tree
cp_parser_objc_method_signature (cp_parser* parser, tree* attributes)
{
tree rettype, kwdparms, optparms;
bool ellipsis = false;
bool is_class_method;
is_class_method = cp_parser_objc_method_type (parser);
rettype = cp_parser_objc_typename (parser);
*attributes = NULL_TREE;
kwdparms = cp_parser_objc_method_keyword_params (parser, attributes);
if (kwdparms == error_mark_node)
return error_mark_node;
optparms = cp_parser_objc_method_tail_params_opt (parser, &ellipsis, attributes);
if (optparms == error_mark_node)
return error_mark_node;
return objc_build_method_signature (is_class_method, rettype, kwdparms, optparms, ellipsis);
}
static bool
cp_parser_objc_method_maybe_bad_prefix_attributes (cp_parser* parser)
{
tree tattr;  
cp_lexer_save_tokens (parser->lexer);
tattr = cp_parser_attributes_opt (parser);
gcc_assert (tattr) ;
if (cp_lexer_next_token_is (parser->lexer, CPP_PLUS)
|| cp_lexer_next_token_is (parser->lexer, CPP_MINUS))
return true;
cp_lexer_rollback_tokens (parser->lexer);
return false;  
}
static void
cp_parser_objc_method_prototype_list (cp_parser* parser)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
while (token->keyword != RID_AT_END && token->type != CPP_EOF)
{
if (token->type == CPP_PLUS || token->type == CPP_MINUS)
{
tree attributes, sig;
bool is_class_method;
if (token->type == CPP_PLUS)
is_class_method = true;
else
is_class_method = false;
sig = cp_parser_objc_method_signature (parser, &attributes);
if (sig == error_mark_node)
{
cp_parser_skip_to_end_of_block_or_statement (parser);
token = cp_lexer_peek_token (parser->lexer);
continue;
}
objc_add_method_declaration (is_class_method, sig, attributes);
cp_parser_consume_semicolon_at_end_of_statement (parser);
}
else if (token->keyword == RID_AT_PROPERTY)
cp_parser_objc_at_property_declaration (parser);
else if (token->keyword == RID_ATTRIBUTE 
&& cp_parser_objc_method_maybe_bad_prefix_attributes(parser))
warning_at (cp_lexer_peek_token (parser->lexer)->location, 
OPT_Wattributes, 
"prefix attributes are ignored for methods");
else
cp_parser_objc_interstitial_code (parser);
token = cp_lexer_peek_token (parser->lexer);
}
if (token->type != CPP_EOF)
cp_lexer_consume_token (parser->lexer);  
else
cp_parser_error (parser, "expected %<@end%>");
objc_finish_interface ();
}
static void
cp_parser_objc_method_definition_list (cp_parser* parser)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
while (token->keyword != RID_AT_END && token->type != CPP_EOF)
{
tree meth;
if (token->type == CPP_PLUS || token->type == CPP_MINUS)
{
cp_token *ptk;
tree sig, attribute;
bool is_class_method;
if (token->type == CPP_PLUS)
is_class_method = true;
else
is_class_method = false;
push_deferring_access_checks (dk_deferred);
sig = cp_parser_objc_method_signature (parser, &attribute);
if (sig == error_mark_node)
{
cp_parser_skip_to_end_of_block_or_statement (parser);
token = cp_lexer_peek_token (parser->lexer);
continue;
}
objc_start_method_definition (is_class_method, sig, attribute,
NULL_TREE);
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
cp_lexer_consume_token (parser->lexer);
ptk = cp_lexer_peek_token (parser->lexer);
if (!(ptk->type == CPP_PLUS || ptk->type == CPP_MINUS 
|| ptk->type == CPP_EOF || ptk->keyword == RID_AT_END))
{
perform_deferred_access_checks (tf_warning_or_error);
stop_deferring_access_checks ();
meth = cp_parser_function_definition_after_declarator (parser,
false);
pop_deferring_access_checks ();
objc_finish_method_definition (meth);
}
}
else if (token->keyword == RID_AT_PROPERTY)
cp_parser_objc_at_property_declaration (parser);
else if (token->keyword == RID_AT_SYNTHESIZE)
cp_parser_objc_at_synthesize_declaration (parser);
else if (token->keyword == RID_AT_DYNAMIC)
cp_parser_objc_at_dynamic_declaration (parser);
else if (token->keyword == RID_ATTRIBUTE 
&& cp_parser_objc_method_maybe_bad_prefix_attributes(parser))
warning_at (token->location, OPT_Wattributes,
"prefix attributes are ignored for methods");
else
cp_parser_objc_interstitial_code (parser);
token = cp_lexer_peek_token (parser->lexer);
}
if (token->type != CPP_EOF)
cp_lexer_consume_token (parser->lexer);  
else
cp_parser_error (parser, "expected %<@end%>");
objc_finish_implementation ();
}
static void
cp_parser_objc_class_ivars (cp_parser* parser)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
if (token->type != CPP_OPEN_BRACE)
return;	
cp_lexer_consume_token (parser->lexer);  
token = cp_lexer_peek_token (parser->lexer);
while (token->type != CPP_CLOSE_BRACE 
&& token->keyword != RID_AT_END && token->type != CPP_EOF)
{
cp_decl_specifier_seq declspecs;
int decl_class_or_enum_p;
tree prefix_attributes;
cp_parser_objc_visibility_spec (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_BRACE))
break;
cp_parser_decl_specifier_seq (parser,
CP_PARSER_FLAGS_OPTIONAL,
&declspecs,
&decl_class_or_enum_p);
if (declspecs.storage_class != sc_none)
{
cp_parser_error (parser, "invalid type for instance variable");	  
declspecs.storage_class = sc_none;
}
if (decl_spec_seq_has_spec_p (&declspecs, ds_thread))
{
cp_parser_error (parser, "invalid type for instance variable");
declspecs.locations[ds_thread] = 0;
}
if (decl_spec_seq_has_spec_p (&declspecs, ds_typedef))
{
cp_parser_error (parser, "invalid type for instance variable");
declspecs.locations[ds_typedef] = 0;
}
prefix_attributes = declspecs.attributes;
declspecs.attributes = NULL_TREE;
while (cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
{
tree width = NULL_TREE, attributes, first_attribute, decl;
cp_declarator *declarator = NULL;
int ctor_dtor_or_conv_p;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_COLON)
goto eat_colon;
if (token->type == CPP_NAME
&& (cp_lexer_peek_nth_token (parser->lexer, 2)->type
== CPP_COLON))
{
declarator = make_id_declarator (NULL_TREE,
cp_parser_identifier (parser),
sfk_none);
eat_colon:
cp_lexer_consume_token (parser->lexer);  
width
= cp_parser_constant_expression (parser);
}
else
{
declarator
= cp_parser_declarator (parser, CP_PARSER_DECLARATOR_NAMED,
&ctor_dtor_or_conv_p,
NULL,
false,
false);
}
attributes = cp_parser_attributes_opt (parser);
first_attribute = attributes;
attributes = attr_chainon (prefix_attributes, attributes);
if (width)
decl = grokbitfield (declarator, &declspecs,
width, NULL_TREE, attributes);
else
decl = grokfield (declarator, &declspecs,
NULL_TREE, false,
NULL_TREE, attributes);
if (decl != error_mark_node && decl != NULL_TREE)
objc_add_instance_variable (decl);
if (attributes != error_mark_node)
{
while (attributes && TREE_CHAIN (attributes) != first_attribute)
attributes = TREE_CHAIN (attributes);
if (attributes)
TREE_CHAIN (attributes) = NULL_TREE;
}
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_COMMA)
{
cp_lexer_consume_token (parser->lexer);  
continue;
}
break;
}
cp_parser_consume_semicolon_at_end_of_statement (parser);
token = cp_lexer_peek_token (parser->lexer);
}
if (token->keyword == RID_AT_END)
cp_parser_error (parser, "expected %<}%>");
if (token->keyword != RID_AT_END && token->type != CPP_EOF)
cp_lexer_consume_token (parser->lexer);  
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
cp_lexer_consume_token (parser->lexer);
}
static void
cp_parser_objc_protocol_declaration (cp_parser* parser, tree attributes)
{
tree proto, protorefs;
cp_token *tok;
cp_lexer_consume_token (parser->lexer);  
if (cp_lexer_next_token_is_not (parser->lexer, CPP_NAME))
{
tok = cp_lexer_peek_token (parser->lexer);
error_at (tok->location, "identifier expected after %<@protocol%>");
cp_parser_consume_semicolon_at_end_of_statement (parser);
return;
}
tok = cp_lexer_peek_nth_token (parser->lexer, 2);
if (tok->type == CPP_COMMA || tok->type == CPP_SEMICOLON)
{
while (true)
{
tree id;
id = cp_parser_identifier (parser);
if (id == error_mark_node)
break;
objc_declare_protocol (id, attributes);
if(cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
cp_lexer_consume_token (parser->lexer);
else
break;
}
cp_parser_consume_semicolon_at_end_of_statement (parser);
}
else
{
proto = cp_parser_identifier (parser);
protorefs = cp_parser_objc_protocol_refs_opt (parser);
objc_start_protocol (proto, protorefs, attributes);
cp_parser_objc_method_prototype_list (parser);
}
}
static void
cp_parser_objc_superclass_or_category (cp_parser *parser, 
bool iface_p,
tree *super,
tree *categ, bool *is_class_extension)
{
cp_token *next = cp_lexer_peek_token (parser->lexer);
*super = *categ = NULL_TREE;
*is_class_extension = false;
if (next->type == CPP_COLON)
{
cp_lexer_consume_token (parser->lexer);  
*super = cp_parser_identifier (parser);
}
else if (next->type == CPP_OPEN_PAREN)
{
matching_parens parens;
parens.consume_open (parser);  
if (iface_p && cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_PAREN))
{
*categ = NULL_TREE;
*is_class_extension = true;
}
else
*categ = cp_parser_identifier (parser);
parens.require_close (parser);
}
}
static void
cp_parser_objc_class_interface (cp_parser* parser, tree attributes)
{
tree name, super, categ, protos;
bool is_class_extension;
cp_lexer_consume_token (parser->lexer);  
name = cp_parser_identifier (parser);
if (name == error_mark_node)
{
return;
}
cp_parser_objc_superclass_or_category (parser, true, &super, &categ,
&is_class_extension);
protos = cp_parser_objc_protocol_refs_opt (parser);
if (categ || is_class_extension)
objc_start_category_interface (name, categ, protos, attributes);
else
{
objc_start_class_interface (name, super, protos, attributes);
cp_parser_objc_class_ivars (parser);
objc_continue_interface ();
}
cp_parser_objc_method_prototype_list (parser);
}
static void
cp_parser_objc_class_implementation (cp_parser* parser)
{
tree name, super, categ;
bool is_class_extension;
cp_lexer_consume_token (parser->lexer);  
name = cp_parser_identifier (parser);
if (name == error_mark_node)
{
return;
}
cp_parser_objc_superclass_or_category (parser, false, &super, &categ,
&is_class_extension);
if (categ)
objc_start_category_implementation (name, categ);
else
{
objc_start_class_implementation (name, super);
cp_parser_objc_class_ivars (parser);
objc_continue_implementation ();
}
cp_parser_objc_method_definition_list (parser);
}
static void
cp_parser_objc_end_implementation (cp_parser* parser)
{
cp_lexer_consume_token (parser->lexer);  
objc_finish_implementation ();
}
static void
cp_parser_objc_declaration (cp_parser* parser, tree attributes)
{
cp_token *kwd = cp_lexer_peek_token (parser->lexer);
if (attributes)
switch (kwd->keyword)
{
case RID_AT_ALIAS:
case RID_AT_CLASS:
case RID_AT_END:
error_at (kwd->location, "attributes may not be specified before"
" the %<@%D%> Objective-C++ keyword",
kwd->u.value);
attributes = NULL;
break;
case RID_AT_IMPLEMENTATION:
warning_at (kwd->location, OPT_Wattributes,
"prefix attributes are ignored before %<@%D%>",
kwd->u.value);
attributes = NULL;
default:
break;
}
switch (kwd->keyword)
{
case RID_AT_ALIAS:
cp_parser_objc_alias_declaration (parser);
break;
case RID_AT_CLASS:
cp_parser_objc_class_declaration (parser);
break;
case RID_AT_PROTOCOL:
cp_parser_objc_protocol_declaration (parser, attributes);
break;
case RID_AT_INTERFACE:
cp_parser_objc_class_interface (parser, attributes);
break;
case RID_AT_IMPLEMENTATION:
cp_parser_objc_class_implementation (parser);
break;
case RID_AT_END:
cp_parser_objc_end_implementation (parser);
break;
default:
error_at (kwd->location, "misplaced %<@%D%> Objective-C++ construct",
kwd->u.value);
cp_parser_skip_to_end_of_block_or_statement (parser);
}
}
static tree
cp_parser_objc_try_catch_finally_statement (cp_parser *parser)
{
location_t location;
tree stmt;
cp_parser_require_keyword (parser, RID_AT_TRY, RT_AT_TRY);
location = cp_lexer_peek_token (parser->lexer)->location;
objc_maybe_warn_exceptions (location);
stmt = push_stmt_list ();
cp_parser_compound_statement (parser, NULL, BCS_NORMAL, false);
objc_begin_try_stmt (location, pop_stmt_list (stmt));
while (cp_lexer_next_token_is_keyword (parser->lexer, RID_AT_CATCH))
{
cp_parameter_declarator *parm;
tree parameter_declaration = error_mark_node;
bool seen_open_paren = false;
matching_parens parens;
cp_lexer_consume_token (parser->lexer);
if (parens.require_open (parser))
seen_open_paren = true;
if (cp_lexer_next_token_is (parser->lexer, CPP_ELLIPSIS))
{
cp_lexer_consume_token (parser->lexer);
parameter_declaration = NULL_TREE;
}
else
{
parm = cp_parser_parameter_declaration (parser, false, NULL);
if (parm == NULL)
parameter_declaration = error_mark_node;
else
parameter_declaration = grokdeclarator (parm->declarator,
&parm->decl_specifiers,
PARM, 0,
NULL);
}
if (seen_open_paren)
parens.require_close (parser);
else
{
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_PAREN))
cp_lexer_consume_token (parser->lexer);
}
objc_begin_catch_clause (parameter_declaration);
cp_parser_compound_statement (parser, NULL, BCS_NORMAL, false);
objc_finish_catch_clause ();
}
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_AT_FINALLY))
{
cp_lexer_consume_token (parser->lexer);
location = cp_lexer_peek_token (parser->lexer)->location;
stmt = push_stmt_list ();
cp_parser_compound_statement (parser, NULL, BCS_NORMAL, false);
objc_build_finally_clause (location, pop_stmt_list (stmt));
}
return objc_finish_try_stmt ();
}
static tree
cp_parser_objc_synchronized_statement (cp_parser *parser)
{
location_t location;
tree lock, stmt;
cp_parser_require_keyword (parser, RID_AT_SYNCHRONIZED, RT_AT_SYNCHRONIZED);
location = cp_lexer_peek_token (parser->lexer)->location;
objc_maybe_warn_exceptions (location);
matching_parens parens;
parens.require_open (parser);
lock = cp_parser_expression (parser);
parens.require_close (parser);
stmt = push_stmt_list ();
cp_parser_compound_statement (parser, NULL, BCS_NORMAL, false);
return objc_build_synchronized (location, lock, pop_stmt_list (stmt));
}
static tree
cp_parser_objc_throw_statement (cp_parser *parser)
{
tree expr = NULL_TREE;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
cp_parser_require_keyword (parser, RID_AT_THROW, RT_AT_THROW);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
expr = cp_parser_expression (parser);
cp_parser_consume_semicolon_at_end_of_statement (parser);
return objc_build_throw_stmt (loc, expr);
}
static tree
cp_parser_objc_statement (cp_parser * parser)
{
cp_token *kwd = cp_lexer_peek_token (parser->lexer);
switch (kwd->keyword)
{
case RID_AT_TRY:
return cp_parser_objc_try_catch_finally_statement (parser);
case RID_AT_SYNCHRONIZED:
return cp_parser_objc_synchronized_statement (parser);
case RID_AT_THROW:
return cp_parser_objc_throw_statement (parser);
default:
error_at (kwd->location, "misplaced %<@%D%> Objective-C++ construct",
kwd->u.value);
cp_parser_skip_to_end_of_block_or_statement (parser);
}
return error_mark_node;
}
static bool
cp_parser_objc_valid_prefix_attributes (cp_parser* parser, tree *attrib)
{
cp_lexer_save_tokens (parser->lexer);
*attrib = cp_parser_attributes_opt (parser);
gcc_assert (*attrib);
if (OBJC_IS_AT_KEYWORD (cp_lexer_peek_token (parser->lexer)->keyword))
{
cp_lexer_commit_tokens (parser->lexer);
return true;
}
cp_lexer_rollback_tokens (parser->lexer);
return false;  
}
static tree
cp_parser_objc_struct_declaration (cp_parser *parser)
{
tree decls = NULL_TREE;
cp_decl_specifier_seq declspecs;
int decl_class_or_enum_p;
tree prefix_attributes;
cp_parser_decl_specifier_seq (parser,
CP_PARSER_FLAGS_NONE,
&declspecs,
&decl_class_or_enum_p);
if (declspecs.type == error_mark_node)
return error_mark_node;
if (declspecs.storage_class != sc_none)
{
cp_parser_error (parser, "invalid type for property");
declspecs.storage_class = sc_none;
}
if (decl_spec_seq_has_spec_p (&declspecs, ds_thread))
{
cp_parser_error (parser, "invalid type for property");
declspecs.locations[ds_thread] = 0;
}
if (decl_spec_seq_has_spec_p (&declspecs, ds_typedef))
{
cp_parser_error (parser, "invalid type for property");
declspecs.locations[ds_typedef] = 0;
}
prefix_attributes = declspecs.attributes;
declspecs.attributes = NULL_TREE;
while (cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
{
tree attributes, first_attribute, decl;
cp_declarator *declarator;
cp_token *token;
declarator = cp_parser_declarator (parser, CP_PARSER_DECLARATOR_NAMED,
NULL, NULL, false, false);
attributes = cp_parser_attributes_opt (parser);
first_attribute = attributes;
attributes = attr_chainon (prefix_attributes, attributes);
decl = grokfield (declarator, &declspecs,
NULL_TREE, false,
NULL_TREE, attributes);
if (decl == error_mark_node || decl == NULL_TREE)
return error_mark_node;
if (attributes != error_mark_node)
{
while (attributes && TREE_CHAIN (attributes) != first_attribute)
attributes = TREE_CHAIN (attributes);
if (attributes)
TREE_CHAIN (attributes) = NULL_TREE;
}
DECL_CHAIN (decl) = decls;
decls = decl;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_COMMA)
{
cp_lexer_consume_token (parser->lexer);  
continue;
}
else
break;
}
return decls;
}
static void 
cp_parser_objc_at_property_declaration (cp_parser *parser)
{
bool property_assign = false;
bool property_copy = false;
tree property_getter_ident = NULL_TREE;
bool property_nonatomic = false;
bool property_readonly = false;
bool property_readwrite = false;
bool property_retain = false;
tree property_setter_ident = NULL_TREE;
tree properties;
location_t loc;
loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);  
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
matching_parens parens;
parens.consume_open (parser);
while (true)
{
bool syntax_error = false;
cp_token *token = cp_lexer_peek_token (parser->lexer);
enum rid keyword;
if (token->type != CPP_NAME)
{
cp_parser_error (parser, "expected identifier");
break;
}
keyword = C_RID_CODE (token->u.value);
cp_lexer_consume_token (parser->lexer);
switch (keyword)
{
case RID_ASSIGN:    property_assign = true;    break;
case RID_COPY:      property_copy = true;      break;
case RID_NONATOMIC: property_nonatomic = true; break;
case RID_READONLY:  property_readonly = true;  break;
case RID_READWRITE: property_readwrite = true; break;
case RID_RETAIN:    property_retain = true;    break;
case RID_GETTER:
case RID_SETTER:
if (cp_lexer_next_token_is_not (parser->lexer, CPP_EQ))
{
if (keyword == RID_GETTER)
cp_parser_error (parser,
"missing %<=%> (after %<getter%> attribute)");
else
cp_parser_error (parser,
"missing %<=%> (after %<setter%> attribute)");
syntax_error = true;
break;
}
cp_lexer_consume_token (parser->lexer); 
if (!cp_parser_objc_selector_p (cp_lexer_peek_token (parser->lexer)->type))
{
cp_parser_error (parser, "expected identifier");
syntax_error = true;
break;
}
if (keyword == RID_SETTER)
{
if (property_setter_ident != NULL_TREE)
{
cp_parser_error (parser, "the %<setter%> attribute may only be specified once");
cp_lexer_consume_token (parser->lexer);
}
else
property_setter_ident = cp_parser_objc_selector (parser);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COLON))
cp_parser_error (parser, "setter name must terminate with %<:%>");
else
cp_lexer_consume_token (parser->lexer);
}
else
{
if (property_getter_ident != NULL_TREE)
{
cp_parser_error (parser, "the %<getter%> attribute may only be specified once");
cp_lexer_consume_token (parser->lexer);
}
else
property_getter_ident = cp_parser_objc_selector (parser);
}
break;
default:
cp_parser_error (parser, "unknown property attribute");
syntax_error = true;
break;
}
if (syntax_error)
break;
if (cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
cp_lexer_consume_token (parser->lexer);
else
break;
}
if (!parens.require_close (parser))
{
cp_parser_skip_to_closing_parenthesis (parser,
true,
false,
true);
}
}
properties = cp_parser_objc_struct_declaration (parser);
if (properties == error_mark_node)
{
cp_parser_skip_to_end_of_statement (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
cp_lexer_consume_token (parser->lexer);
return;
}
if (properties == NULL_TREE)
cp_parser_error (parser, "expected identifier");
else
{
properties = nreverse (properties);
for (; properties; properties = TREE_CHAIN (properties))
objc_add_property_declaration (loc, copy_node (properties),
property_readonly, property_readwrite,
property_assign, property_retain,
property_copy, property_nonatomic,
property_getter_ident, property_setter_ident);
}
cp_parser_consume_semicolon_at_end_of_statement (parser);
}
static void 
cp_parser_objc_at_synthesize_declaration (cp_parser *parser)
{
tree list = NULL_TREE;
location_t loc;
loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);  
while (true)
{
tree property, ivar;
property = cp_parser_identifier (parser);
if (property == error_mark_node)
{
cp_parser_consume_semicolon_at_end_of_statement (parser);
return;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_EQ))
{
cp_lexer_consume_token (parser->lexer);
ivar = cp_parser_identifier (parser);
if (ivar == error_mark_node)
{
cp_parser_consume_semicolon_at_end_of_statement (parser);
return;
}
}
else
ivar = NULL_TREE;
list = chainon (list, build_tree_list (ivar, property));
if (cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
cp_lexer_consume_token (parser->lexer);
else
break;
}
cp_parser_consume_semicolon_at_end_of_statement (parser);
objc_add_synthesize_declaration (loc, list);
}
static void 
cp_parser_objc_at_dynamic_declaration (cp_parser *parser)
{
tree list = NULL_TREE;
location_t loc;
loc = cp_lexer_peek_token (parser->lexer)->location;
cp_lexer_consume_token (parser->lexer);  
while (true)
{
tree property;
property = cp_parser_identifier (parser);
if (property == error_mark_node)
{
cp_parser_consume_semicolon_at_end_of_statement (parser);
return;
}
list = chainon (list, build_tree_list (NULL, property));
if (cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
cp_lexer_consume_token (parser->lexer);
else
break;
}
cp_parser_consume_semicolon_at_end_of_statement (parser);
objc_add_dynamic_declaration (loc, list);
}

static pragma_omp_clause
cp_parser_omp_clause_name (cp_parser *parser)
{
pragma_omp_clause result = PRAGMA_OMP_CLAUSE_NONE;
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_AUTO))
result = PRAGMA_OACC_CLAUSE_AUTO;
else if (cp_lexer_next_token_is_keyword (parser->lexer, RID_IF))
result = PRAGMA_OMP_CLAUSE_IF;
else if (cp_lexer_next_token_is_keyword (parser->lexer, RID_DEFAULT))
result = PRAGMA_OMP_CLAUSE_DEFAULT;
else if (cp_lexer_next_token_is_keyword (parser->lexer, RID_DELETE))
result = PRAGMA_OACC_CLAUSE_DELETE;
else if (cp_lexer_next_token_is_keyword (parser->lexer, RID_PRIVATE))
result = PRAGMA_OMP_CLAUSE_PRIVATE;
else if (cp_lexer_next_token_is_keyword (parser->lexer, RID_FOR))
result = PRAGMA_OMP_CLAUSE_FOR;
else if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
switch (p[0])
{
case 'a':
if (!strcmp ("aligned", p))
result = PRAGMA_OMP_CLAUSE_ALIGNED;
else if (!strcmp ("async", p))
result = PRAGMA_OACC_CLAUSE_ASYNC;
break;
case 'c':
if (!strcmp ("collapse", p))
result = PRAGMA_OMP_CLAUSE_COLLAPSE;
else if (!strcmp ("copy", p))
result = PRAGMA_OACC_CLAUSE_COPY;
else if (!strcmp ("copyin", p))
result = PRAGMA_OMP_CLAUSE_COPYIN;
else if (!strcmp ("copyout", p))
result = PRAGMA_OACC_CLAUSE_COPYOUT;
else if (!strcmp ("copyprivate", p))
result = PRAGMA_OMP_CLAUSE_COPYPRIVATE;
else if (!strcmp ("create", p))
result = PRAGMA_OACC_CLAUSE_CREATE;
break;
case 'd':
if (!strcmp ("defaultmap", p))
result = PRAGMA_OMP_CLAUSE_DEFAULTMAP;
else if (!strcmp ("depend", p))
result = PRAGMA_OMP_CLAUSE_DEPEND;
else if (!strcmp ("device", p))
result = PRAGMA_OMP_CLAUSE_DEVICE;
else if (!strcmp ("deviceptr", p))
result = PRAGMA_OACC_CLAUSE_DEVICEPTR;
else if (!strcmp ("device_resident", p))
result = PRAGMA_OACC_CLAUSE_DEVICE_RESIDENT;
else if (!strcmp ("dist_schedule", p))
result = PRAGMA_OMP_CLAUSE_DIST_SCHEDULE;
break;
case 'f':
if (!strcmp ("final", p))
result = PRAGMA_OMP_CLAUSE_FINAL;
else if (!strcmp ("firstprivate", p))
result = PRAGMA_OMP_CLAUSE_FIRSTPRIVATE;
else if (!strcmp ("from", p))
result = PRAGMA_OMP_CLAUSE_FROM;
break;
case 'g':
if (!strcmp ("gang", p))
result = PRAGMA_OACC_CLAUSE_GANG;
else if (!strcmp ("grainsize", p))
result = PRAGMA_OMP_CLAUSE_GRAINSIZE;
break;
case 'h':
if (!strcmp ("hint", p))
result = PRAGMA_OMP_CLAUSE_HINT;
else if (!strcmp ("host", p))
result = PRAGMA_OACC_CLAUSE_HOST;
break;
case 'i':
if (!strcmp ("inbranch", p))
result = PRAGMA_OMP_CLAUSE_INBRANCH;
else if (!strcmp ("independent", p))
result = PRAGMA_OACC_CLAUSE_INDEPENDENT;
else if (!strcmp ("is_device_ptr", p))
result = PRAGMA_OMP_CLAUSE_IS_DEVICE_PTR;
break;
case 'l':
if (!strcmp ("lastprivate", p))
result = PRAGMA_OMP_CLAUSE_LASTPRIVATE;
else if (!strcmp ("linear", p))
result = PRAGMA_OMP_CLAUSE_LINEAR;
else if (!strcmp ("link", p))
result = PRAGMA_OMP_CLAUSE_LINK;
break;
case 'm':
if (!strcmp ("map", p))
result = PRAGMA_OMP_CLAUSE_MAP;
else if (!strcmp ("mergeable", p))
result = PRAGMA_OMP_CLAUSE_MERGEABLE;
break;
case 'n':
if (!strcmp ("nogroup", p))
result = PRAGMA_OMP_CLAUSE_NOGROUP;
else if (!strcmp ("notinbranch", p))
result = PRAGMA_OMP_CLAUSE_NOTINBRANCH;
else if (!strcmp ("nowait", p))
result = PRAGMA_OMP_CLAUSE_NOWAIT;
else if (!strcmp ("num_gangs", p))
result = PRAGMA_OACC_CLAUSE_NUM_GANGS;
else if (!strcmp ("num_tasks", p))
result = PRAGMA_OMP_CLAUSE_NUM_TASKS;
else if (!strcmp ("num_teams", p))
result = PRAGMA_OMP_CLAUSE_NUM_TEAMS;
else if (!strcmp ("num_threads", p))
result = PRAGMA_OMP_CLAUSE_NUM_THREADS;
else if (!strcmp ("num_workers", p))
result = PRAGMA_OACC_CLAUSE_NUM_WORKERS;
break;
case 'o':
if (!strcmp ("ordered", p))
result = PRAGMA_OMP_CLAUSE_ORDERED;
break;
case 'p':
if (!strcmp ("parallel", p))
result = PRAGMA_OMP_CLAUSE_PARALLEL;
else if (!strcmp ("present", p))
result = PRAGMA_OACC_CLAUSE_PRESENT;
else if (!strcmp ("present_or_copy", p)
|| !strcmp ("pcopy", p))
result = PRAGMA_OACC_CLAUSE_PRESENT_OR_COPY;
else if (!strcmp ("present_or_copyin", p)
|| !strcmp ("pcopyin", p))
result = PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYIN;
else if (!strcmp ("present_or_copyout", p)
|| !strcmp ("pcopyout", p))
result = PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYOUT;
else if (!strcmp ("present_or_create", p)
|| !strcmp ("pcreate", p))
result = PRAGMA_OACC_CLAUSE_PRESENT_OR_CREATE;
else if (!strcmp ("priority", p))
result = PRAGMA_OMP_CLAUSE_PRIORITY;
else if (!strcmp ("proc_bind", p))
result = PRAGMA_OMP_CLAUSE_PROC_BIND;
break;
case 'r':
if (!strcmp ("reduction", p))
result = PRAGMA_OMP_CLAUSE_REDUCTION;
break;
case 's':
if (!strcmp ("safelen", p))
result = PRAGMA_OMP_CLAUSE_SAFELEN;
else if (!strcmp ("schedule", p))
result = PRAGMA_OMP_CLAUSE_SCHEDULE;
else if (!strcmp ("sections", p))
result = PRAGMA_OMP_CLAUSE_SECTIONS;
else if (!strcmp ("self", p))
result = PRAGMA_OACC_CLAUSE_SELF;
else if (!strcmp ("seq", p))
result = PRAGMA_OACC_CLAUSE_SEQ;
else if (!strcmp ("shared", p))
result = PRAGMA_OMP_CLAUSE_SHARED;
else if (!strcmp ("simd", p))
result = PRAGMA_OMP_CLAUSE_SIMD;
else if (!strcmp ("simdlen", p))
result = PRAGMA_OMP_CLAUSE_SIMDLEN;
break;
case 't':
if (!strcmp ("taskgroup", p))
result = PRAGMA_OMP_CLAUSE_TASKGROUP;
else if (!strcmp ("thread_limit", p))
result = PRAGMA_OMP_CLAUSE_THREAD_LIMIT;
else if (!strcmp ("threads", p))
result = PRAGMA_OMP_CLAUSE_THREADS;
else if (!strcmp ("tile", p))
result = PRAGMA_OACC_CLAUSE_TILE;
else if (!strcmp ("to", p))
result = PRAGMA_OMP_CLAUSE_TO;
break;
case 'u':
if (!strcmp ("uniform", p))
result = PRAGMA_OMP_CLAUSE_UNIFORM;
else if (!strcmp ("untied", p))
result = PRAGMA_OMP_CLAUSE_UNTIED;
else if (!strcmp ("use_device", p))
result = PRAGMA_OACC_CLAUSE_USE_DEVICE;
else if (!strcmp ("use_device_ptr", p))
result = PRAGMA_OMP_CLAUSE_USE_DEVICE_PTR;
break;
case 'v':
if (!strcmp ("vector", p))
result = PRAGMA_OACC_CLAUSE_VECTOR;
else if (!strcmp ("vector_length", p))
result = PRAGMA_OACC_CLAUSE_VECTOR_LENGTH;
break;
case 'w':
if (!strcmp ("wait", p))
result = PRAGMA_OACC_CLAUSE_WAIT;
else if (!strcmp ("worker", p))
result = PRAGMA_OACC_CLAUSE_WORKER;
break;
}
}
if (result != PRAGMA_OMP_CLAUSE_NONE)
cp_lexer_consume_token (parser->lexer);
return result;
}
static void
check_no_duplicate_clause (tree clauses, enum omp_clause_code code,
const char *name, location_t location)
{
tree c;
for (c = clauses; c ; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_CODE (c) == code)
{
error_at (location, "too many %qs clauses", name);
break;
}
}
static tree
cp_parser_omp_var_list_no_open (cp_parser *parser, enum omp_clause_code kind,
tree list, bool *colon)
{
cp_token *token;
bool saved_colon_corrects_to_scope_p = parser->colon_corrects_to_scope_p;
if (colon)
{
parser->colon_corrects_to_scope_p = false;
*colon = false;
}
while (1)
{
tree name, decl;
token = cp_lexer_peek_token (parser->lexer);
if (kind != 0
&& current_class_ptr
&& cp_parser_is_keyword (token, RID_THIS))
{
decl = finish_this_expr ();
if (TREE_CODE (decl) == NON_LVALUE_EXPR
|| CONVERT_EXPR_P (decl))
decl = TREE_OPERAND (decl, 0);
cp_lexer_consume_token (parser->lexer);
}
else
{
name = cp_parser_id_expression (parser, false,
true,
NULL,
false,
false);
if (name == error_mark_node)
goto skip_comma;
if (identifier_p (name))
decl = cp_parser_lookup_name_simple (parser, name, token->location);
else
decl = name;
if (decl == error_mark_node)
cp_parser_name_lookup_error (parser, name, decl, NLE_NULL,
token->location);
}
if (decl == error_mark_node)
;
else if (kind != 0)
{
switch (kind)
{
case OMP_CLAUSE__CACHE_:
if (cp_lexer_peek_token (parser->lexer)->type != CPP_OPEN_SQUARE)
{
error_at (token->location, "expected %<[%>");
decl = error_mark_node;
break;
}
case OMP_CLAUSE_MAP:
case OMP_CLAUSE_FROM:
case OMP_CLAUSE_TO:
while (cp_lexer_next_token_is (parser->lexer, CPP_DOT))
{
location_t loc
= cp_lexer_peek_token (parser->lexer)->location;
cp_id_kind idk = CP_ID_KIND_NONE;
cp_lexer_consume_token (parser->lexer);
decl = convert_from_reference (decl);
decl
= cp_parser_postfix_dot_deref_expression (parser, CPP_DOT,
decl, false,
&idk, loc);
}
case OMP_CLAUSE_DEPEND:
case OMP_CLAUSE_REDUCTION:
while (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_SQUARE))
{
tree low_bound = NULL_TREE, length = NULL_TREE;
parser->colon_corrects_to_scope_p = false;
cp_lexer_consume_token (parser->lexer);
if (!cp_lexer_next_token_is (parser->lexer, CPP_COLON))
low_bound = cp_parser_expression (parser);
if (!colon)
parser->colon_corrects_to_scope_p
= saved_colon_corrects_to_scope_p;
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_SQUARE))
length = integer_one_node;
else
{
if (!cp_parser_require (parser, CPP_COLON, RT_COLON))
goto skip_comma;
if (!cp_lexer_next_token_is (parser->lexer,
CPP_CLOSE_SQUARE))
length = cp_parser_expression (parser);
}
if (!cp_parser_require (parser, CPP_CLOSE_SQUARE,
RT_CLOSE_SQUARE))
goto skip_comma;
decl = tree_cons (low_bound, length, decl);
}
break;
default:
break;
}
tree u = build_omp_clause (token->location, kind);
OMP_CLAUSE_DECL (u) = decl;
OMP_CLAUSE_CHAIN (u) = list;
list = u;
}
else
list = tree_cons (decl, NULL_TREE, list);
get_comma:
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA))
break;
cp_lexer_consume_token (parser->lexer);
}
if (colon)
parser->colon_corrects_to_scope_p = saved_colon_corrects_to_scope_p;
if (colon != NULL && cp_lexer_next_token_is (parser->lexer, CPP_COLON))
{
*colon = true;
cp_parser_require (parser, CPP_COLON, RT_COLON);
return list;
}
if (!cp_parser_require (parser, CPP_CLOSE_PAREN, RT_CLOSE_PAREN))
{
int ending;
skip_comma:
if (colon)
parser->colon_corrects_to_scope_p = saved_colon_corrects_to_scope_p;
ending = cp_parser_skip_to_closing_parenthesis (parser,
true,
true,
true);
if (ending < 0)
goto get_comma;
}
return list;
}
static tree
cp_parser_omp_var_list (cp_parser *parser, enum omp_clause_code kind, tree list)
{
if (cp_parser_require (parser, CPP_OPEN_PAREN, RT_OPEN_PAREN))
return cp_parser_omp_var_list_no_open (parser, kind, list, NULL);
return list;
}
static tree
cp_parser_oacc_data_clause (cp_parser *parser, pragma_omp_clause c_kind,
tree list)
{
enum gomp_map_kind kind;
switch (c_kind)
{
case PRAGMA_OACC_CLAUSE_COPY:
kind = GOMP_MAP_FORCE_TOFROM;
break;
case PRAGMA_OACC_CLAUSE_COPYIN:
kind = GOMP_MAP_FORCE_TO;
break;
case PRAGMA_OACC_CLAUSE_COPYOUT:
kind = GOMP_MAP_FORCE_FROM;
break;
case PRAGMA_OACC_CLAUSE_CREATE:
kind = GOMP_MAP_FORCE_ALLOC;
break;
case PRAGMA_OACC_CLAUSE_DELETE:
kind = GOMP_MAP_DELETE;
break;
case PRAGMA_OACC_CLAUSE_DEVICE:
kind = GOMP_MAP_FORCE_TO;
break;
case PRAGMA_OACC_CLAUSE_DEVICE_RESIDENT:
kind = GOMP_MAP_DEVICE_RESIDENT;
break;
case PRAGMA_OACC_CLAUSE_HOST:
case PRAGMA_OACC_CLAUSE_SELF:
kind = GOMP_MAP_FORCE_FROM;
break;
case PRAGMA_OACC_CLAUSE_LINK:
kind = GOMP_MAP_LINK;
break;
case PRAGMA_OACC_CLAUSE_PRESENT:
kind = GOMP_MAP_FORCE_PRESENT;
break;
case PRAGMA_OACC_CLAUSE_PRESENT_OR_COPY:
kind = GOMP_MAP_TOFROM;
break;
case PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYIN:
kind = GOMP_MAP_TO;
break;
case PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYOUT:
kind = GOMP_MAP_FROM;
break;
case PRAGMA_OACC_CLAUSE_PRESENT_OR_CREATE:
kind = GOMP_MAP_ALLOC;
break;
default:
gcc_unreachable ();
}
tree nl, c;
nl = cp_parser_omp_var_list (parser, OMP_CLAUSE_MAP, list);
for (c = nl; c != list; c = OMP_CLAUSE_CHAIN (c))
OMP_CLAUSE_SET_MAP_KIND (c, kind);
return nl;
}
static tree
cp_parser_oacc_data_clause_deviceptr (cp_parser *parser, tree list)
{
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
tree vars, t;
vars = cp_parser_omp_var_list (parser, OMP_CLAUSE_ERROR, NULL);
for (t = vars; t; t = TREE_CHAIN (t))
{
tree v = TREE_PURPOSE (t);
tree u = build_omp_clause (loc, OMP_CLAUSE_MAP);
OMP_CLAUSE_SET_MAP_KIND (u, GOMP_MAP_FORCE_DEVICEPTR);
OMP_CLAUSE_DECL (u) = v;
OMP_CLAUSE_CHAIN (u) = list;
list = u;
}
return list;
}
static tree
cp_parser_oacc_simple_clause (cp_parser * ,
enum omp_clause_code code,
tree list, location_t location)
{
check_no_duplicate_clause (list, code, omp_clause_code_name[code], location);
tree c = build_omp_clause (location, code);
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_oacc_single_int_clause (cp_parser *parser, omp_clause_code code,
const char *str, tree list)
{
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
matching_parens parens;
if (!parens.require_open (parser))
return list;
tree t = cp_parser_assignment_expression (parser, NULL, false, false);
if (t == error_mark_node
|| !parens.require_close (parser))
{
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
return list;
}
check_no_duplicate_clause (list, code, str, loc);
tree c = build_omp_clause (loc, code);
OMP_CLAUSE_OPERAND (c, 0) = t;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_oacc_shape_clause (cp_parser *parser, omp_clause_code kind,
const char *str, tree list)
{
const char *id = "num";
cp_lexer *lexer = parser->lexer;
tree ops[2] = { NULL_TREE, NULL_TREE }, c;
location_t loc = cp_lexer_peek_token (lexer)->location;
if (kind == OMP_CLAUSE_VECTOR)
id = "length";
if (cp_lexer_next_token_is (lexer, CPP_OPEN_PAREN))
{
matching_parens parens;
parens.consume_open (parser);
do
{
cp_token *next = cp_lexer_peek_token (lexer);
int idx = 0;
if (kind == OMP_CLAUSE_GANG
&& cp_lexer_next_token_is_keyword (lexer, RID_STATIC))
{
cp_lexer_consume_token (lexer);
if (!cp_parser_require (parser, CPP_COLON, RT_COLON))
goto cleanup_error;
idx = 1;
if (ops[idx] != NULL)
{
cp_parser_error (parser, "too many %<static%> arguments");
goto cleanup_error;
}
if (cp_lexer_next_token_is (lexer, CPP_MULT)
&& (cp_lexer_nth_token_is (parser->lexer, 2, CPP_COMMA)
|| cp_lexer_nth_token_is (parser->lexer, 2,
CPP_CLOSE_PAREN)))
{
cp_lexer_consume_token (lexer);
ops[idx] = integer_minus_one_node;
if (cp_lexer_next_token_is (lexer, CPP_COMMA))
{
cp_lexer_consume_token (lexer);
continue;
}
else break;
}
}
else if (cp_lexer_next_token_is (lexer, CPP_NAME)
&& id_equal (next->u.value, id)
&& cp_lexer_nth_token_is (lexer, 2, CPP_COLON))
{
cp_lexer_consume_token (lexer);  
cp_lexer_consume_token (lexer);  
}
if (ops[idx] != NULL_TREE)
{
cp_parser_error (parser, "unexpected argument");
goto cleanup_error;
}
tree expr = cp_parser_assignment_expression (parser, NULL, false,
false);
if (expr == error_mark_node)
goto cleanup_error;
mark_exp_read (expr);
ops[idx] = expr;
if (kind == OMP_CLAUSE_GANG
&& cp_lexer_next_token_is (lexer, CPP_COMMA))
{
cp_lexer_consume_token (lexer);
continue;
}
break;
}
while (1);
if (!parens.require_close (parser))
goto cleanup_error;
}
check_no_duplicate_clause (list, kind, str, loc);
c = build_omp_clause (loc, kind);
if (ops[1])
OMP_CLAUSE_OPERAND (c, 1) = ops[1];
OMP_CLAUSE_OPERAND (c, 0) = ops[0];
OMP_CLAUSE_CHAIN (c) = list;
return c;
cleanup_error:
cp_parser_skip_to_closing_parenthesis (parser, false, false, true);
return list;
}
static tree
cp_parser_oacc_clause_tile (cp_parser *parser, location_t clause_loc, tree list)
{
tree c, expr = error_mark_node;
tree tile = NULL_TREE;
check_no_duplicate_clause (list, OMP_CLAUSE_TILE, "tile", clause_loc);
check_no_duplicate_clause (list, OMP_CLAUSE_COLLAPSE, "collapse",
clause_loc);
if (!cp_parser_require (parser, CPP_OPEN_PAREN, RT_OPEN_PAREN))
return list;
do
{
if (tile && !cp_parser_require (parser, CPP_COMMA, RT_COMMA))
return list;
if (cp_lexer_next_token_is (parser->lexer, CPP_MULT)
&& (cp_lexer_nth_token_is (parser->lexer, 2, CPP_COMMA)
|| cp_lexer_nth_token_is (parser->lexer, 2, CPP_CLOSE_PAREN)))
{
cp_lexer_consume_token (parser->lexer);
expr = integer_zero_node;
}
else
expr = cp_parser_constant_expression (parser);
tile = tree_cons (NULL_TREE, expr, tile);
}
while (cp_lexer_next_token_is_not (parser->lexer, CPP_CLOSE_PAREN));
cp_lexer_consume_token (parser->lexer);
c = build_omp_clause (clause_loc, OMP_CLAUSE_TILE);
tile = nreverse (tile);
OMP_CLAUSE_TILE_LIST (c) = tile;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_oacc_wait_list (cp_parser *parser, location_t clause_loc, tree list)
{
vec<tree, va_gc> *args;
tree t, args_tree;
args = cp_parser_parenthesized_expression_list (parser, non_attr,
false,
true,
NULL);
if (args == NULL || args->length () == 0)
{
cp_parser_error (parser, "expected integer expression before ')'");
if (args != NULL)
release_tree_vector (args);
return list;
}
args_tree = build_tree_list_vec (args);
release_tree_vector (args);
for (t = args_tree; t; t = TREE_CHAIN (t))
{
tree targ = TREE_VALUE (t);
if (targ != error_mark_node)
{
if (!INTEGRAL_TYPE_P (TREE_TYPE (targ)))
error ("%<wait%> expression must be integral");
else
{
tree c = build_omp_clause (clause_loc, OMP_CLAUSE_WAIT);
targ = mark_rvalue_use (targ);
OMP_CLAUSE_DECL (c) = targ;
OMP_CLAUSE_CHAIN (c) = list;
list = c;
}
}
}
return list;
}
static tree
cp_parser_oacc_clause_wait (cp_parser *parser, tree list)
{
location_t location = cp_lexer_peek_token (parser->lexer)->location;
if (cp_lexer_peek_token (parser->lexer)->type != CPP_OPEN_PAREN)
return list;
list = cp_parser_oacc_wait_list (parser, location, list);
return list;
}
static tree
cp_parser_omp_clause_collapse (cp_parser *parser, tree list, location_t location)
{
tree c, num;
location_t loc;
HOST_WIDE_INT n;
loc = cp_lexer_peek_token (parser->lexer)->location;
matching_parens parens;
if (!parens.require_open (parser))
return list;
num = cp_parser_constant_expression (parser);
if (!parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
if (num == error_mark_node)
return list;
num = fold_non_dependent_expr (num);
if (!tree_fits_shwi_p (num)
|| !INTEGRAL_TYPE_P (TREE_TYPE (num))
|| (n = tree_to_shwi (num)) <= 0
|| (int) n != n)
{
error_at (loc, "collapse argument needs positive constant integer expression");
return list;
}
check_no_duplicate_clause (list, OMP_CLAUSE_COLLAPSE, "collapse", location);
check_no_duplicate_clause (list, OMP_CLAUSE_TILE, "tile", location);
c = build_omp_clause (loc, OMP_CLAUSE_COLLAPSE);
OMP_CLAUSE_CHAIN (c) = list;
OMP_CLAUSE_COLLAPSE_EXPR (c) = num;
return c;
}
static tree
cp_parser_omp_clause_default (cp_parser *parser, tree list,
location_t location, bool is_oacc)
{
enum omp_clause_default_kind kind = OMP_CLAUSE_DEFAULT_UNSPECIFIED;
tree c;
matching_parens parens;
if (!parens.require_open (parser))
return list;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
switch (p[0])
{
case 'n':
if (strcmp ("none", p) != 0)
goto invalid_kind;
kind = OMP_CLAUSE_DEFAULT_NONE;
break;
case 'p':
if (strcmp ("present", p) != 0 || !is_oacc)
goto invalid_kind;
kind = OMP_CLAUSE_DEFAULT_PRESENT;
break;
case 's':
if (strcmp ("shared", p) != 0 || is_oacc)
goto invalid_kind;
kind = OMP_CLAUSE_DEFAULT_SHARED;
break;
default:
goto invalid_kind;
}
cp_lexer_consume_token (parser->lexer);
}
else
{
invalid_kind:
if (is_oacc)
cp_parser_error (parser, "expected %<none%> or %<present%>");
else
cp_parser_error (parser, "expected %<none%> or %<shared%>");
}
if (kind == OMP_CLAUSE_DEFAULT_UNSPECIFIED
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
if (kind == OMP_CLAUSE_DEFAULT_UNSPECIFIED)
return list;
check_no_duplicate_clause (list, OMP_CLAUSE_DEFAULT, "default", location);
c = build_omp_clause (location, OMP_CLAUSE_DEFAULT);
OMP_CLAUSE_CHAIN (c) = list;
OMP_CLAUSE_DEFAULT_KIND (c) = kind;
return c;
}
static tree
cp_parser_omp_clause_final (cp_parser *parser, tree list, location_t location)
{
tree t, c;
matching_parens parens;
if (!parens.require_open (parser))
return list;
t = cp_parser_condition (parser);
if (t == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
check_no_duplicate_clause (list, OMP_CLAUSE_FINAL, "final", location);
c = build_omp_clause (location, OMP_CLAUSE_FINAL);
OMP_CLAUSE_FINAL_EXPR (c) = t;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_if (cp_parser *parser, tree list, location_t location,
bool is_omp)
{
tree t, c;
enum tree_code if_modifier = ERROR_MARK;
matching_parens parens;
if (!parens.require_open (parser))
return list;
if (is_omp && cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
int n = 2;
if (strcmp ("parallel", p) == 0)
if_modifier = OMP_PARALLEL;
else if (strcmp ("task", p) == 0)
if_modifier = OMP_TASK;
else if (strcmp ("taskloop", p) == 0)
if_modifier = OMP_TASKLOOP;
else if (strcmp ("target", p) == 0)
{
if_modifier = OMP_TARGET;
if (cp_lexer_nth_token_is (parser->lexer, 2, CPP_NAME))
{
id = cp_lexer_peek_nth_token (parser->lexer, 2)->u.value;
p = IDENTIFIER_POINTER (id);
if (strcmp ("data", p) == 0)
if_modifier = OMP_TARGET_DATA;
else if (strcmp ("update", p) == 0)
if_modifier = OMP_TARGET_UPDATE;
else if (strcmp ("enter", p) == 0)
if_modifier = OMP_TARGET_ENTER_DATA;
else if (strcmp ("exit", p) == 0)
if_modifier = OMP_TARGET_EXIT_DATA;
if (if_modifier != OMP_TARGET)
n = 3;
else
{
location_t loc
= cp_lexer_peek_nth_token (parser->lexer, 2)->location;
error_at (loc, "expected %<data%>, %<update%>, %<enter%> "
"or %<exit%>");
if_modifier = ERROR_MARK;
}
if (if_modifier == OMP_TARGET_ENTER_DATA
|| if_modifier == OMP_TARGET_EXIT_DATA)
{
if (cp_lexer_nth_token_is (parser->lexer, 3, CPP_NAME))
{
id = cp_lexer_peek_nth_token (parser->lexer, 3)->u.value;
p = IDENTIFIER_POINTER (id);
if (strcmp ("data", p) == 0)
n = 4;
}
if (n != 4)
{
location_t loc
= cp_lexer_peek_nth_token (parser->lexer, 3)->location;
error_at (loc, "expected %<data%>");
if_modifier = ERROR_MARK;
}
}
}
}
if (if_modifier != ERROR_MARK)
{
if (cp_lexer_nth_token_is (parser->lexer, n, CPP_COLON))
{
while (n-- > 0)
cp_lexer_consume_token (parser->lexer);
}
else
{
if (n > 2)
{
location_t loc
= cp_lexer_peek_nth_token (parser->lexer, n)->location;
error_at (loc, "expected %<:%>");
}
if_modifier = ERROR_MARK;
}
}
}
t = cp_parser_condition (parser);
if (t == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
for (c = list; c ; c = OMP_CLAUSE_CHAIN (c))
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_IF)
{
if (if_modifier != ERROR_MARK
&& OMP_CLAUSE_IF_MODIFIER (c) == if_modifier)
{
const char *p = NULL;
switch (if_modifier)
{
case OMP_PARALLEL: p = "parallel"; break;
case OMP_TASK: p = "task"; break;
case OMP_TASKLOOP: p = "taskloop"; break;
case OMP_TARGET_DATA: p = "target data"; break;
case OMP_TARGET: p = "target"; break;
case OMP_TARGET_UPDATE: p = "target update"; break;
case OMP_TARGET_ENTER_DATA: p = "enter data"; break;
case OMP_TARGET_EXIT_DATA: p = "exit data"; break;
default: gcc_unreachable ();
}
error_at (location, "too many %<if%> clauses with %qs modifier",
p);
return list;
}
else if (OMP_CLAUSE_IF_MODIFIER (c) == if_modifier)
{
if (!is_omp)
error_at (location, "too many %<if%> clauses");
else
error_at (location, "too many %<if%> clauses without modifier");
return list;
}
else if (if_modifier == ERROR_MARK
|| OMP_CLAUSE_IF_MODIFIER (c) == ERROR_MARK)
{
error_at (location, "if any %<if%> clause has modifier, then all "
"%<if%> clauses have to use modifier");
return list;
}
}
c = build_omp_clause (location, OMP_CLAUSE_IF);
OMP_CLAUSE_IF_MODIFIER (c) = if_modifier;
OMP_CLAUSE_IF_EXPR (c) = t;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_mergeable (cp_parser * ,
tree list, location_t location)
{
tree c;
check_no_duplicate_clause (list, OMP_CLAUSE_MERGEABLE, "mergeable",
location);
c = build_omp_clause (location, OMP_CLAUSE_MERGEABLE);
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_nowait (cp_parser * ,
tree list, location_t location)
{
tree c;
check_no_duplicate_clause (list, OMP_CLAUSE_NOWAIT, "nowait", location);
c = build_omp_clause (location, OMP_CLAUSE_NOWAIT);
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_num_threads (cp_parser *parser, tree list,
location_t location)
{
tree t, c;
matching_parens parens;
if (!parens.require_open (parser))
return list;
t = cp_parser_expression (parser);
if (t == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
check_no_duplicate_clause (list, OMP_CLAUSE_NUM_THREADS,
"num_threads", location);
c = build_omp_clause (location, OMP_CLAUSE_NUM_THREADS);
OMP_CLAUSE_NUM_THREADS_EXPR (c) = t;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_num_tasks (cp_parser *parser, tree list,
location_t location)
{
tree t, c;
matching_parens parens;
if (!parens.require_open (parser))
return list;
t = cp_parser_expression (parser);
if (t == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
check_no_duplicate_clause (list, OMP_CLAUSE_NUM_TASKS,
"num_tasks", location);
c = build_omp_clause (location, OMP_CLAUSE_NUM_TASKS);
OMP_CLAUSE_NUM_TASKS_EXPR (c) = t;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_grainsize (cp_parser *parser, tree list,
location_t location)
{
tree t, c;
matching_parens parens;
if (!parens.require_open (parser))
return list;
t = cp_parser_expression (parser);
if (t == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
check_no_duplicate_clause (list, OMP_CLAUSE_GRAINSIZE,
"grainsize", location);
c = build_omp_clause (location, OMP_CLAUSE_GRAINSIZE);
OMP_CLAUSE_GRAINSIZE_EXPR (c) = t;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_priority (cp_parser *parser, tree list,
location_t location)
{
tree t, c;
matching_parens parens;
if (!parens.require_open (parser))
return list;
t = cp_parser_expression (parser);
if (t == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
check_no_duplicate_clause (list, OMP_CLAUSE_PRIORITY,
"priority", location);
c = build_omp_clause (location, OMP_CLAUSE_PRIORITY);
OMP_CLAUSE_PRIORITY_EXPR (c) = t;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_hint (cp_parser *parser, tree list,
location_t location)
{
tree t, c;
matching_parens parens;
if (!parens.require_open (parser))
return list;
t = cp_parser_expression (parser);
if (t == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
check_no_duplicate_clause (list, OMP_CLAUSE_HINT, "hint", location);
c = build_omp_clause (location, OMP_CLAUSE_HINT);
OMP_CLAUSE_HINT_EXPR (c) = t;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_defaultmap (cp_parser *parser, tree list,
location_t location)
{
tree c, id;
const char *p;
matching_parens parens;
if (!parens.require_open (parser))
return list;
if (!cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
cp_parser_error (parser, "expected %<tofrom%>");
goto out_err;
}
id = cp_lexer_peek_token (parser->lexer)->u.value;
p = IDENTIFIER_POINTER (id);
if (strcmp (p, "tofrom") != 0)
{
cp_parser_error (parser, "expected %<tofrom%>");
goto out_err;
}
cp_lexer_consume_token (parser->lexer);
if (!cp_parser_require (parser, CPP_COLON, RT_COLON))
goto out_err;
if (!cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
cp_parser_error (parser, "expected %<scalar%>");
goto out_err;
}
id = cp_lexer_peek_token (parser->lexer)->u.value;
p = IDENTIFIER_POINTER (id);
if (strcmp (p, "scalar") != 0)
{
cp_parser_error (parser, "expected %<scalar%>");
goto out_err;
}
cp_lexer_consume_token (parser->lexer);
if (!parens.require_close (parser))
goto out_err;
check_no_duplicate_clause (list, OMP_CLAUSE_DEFAULTMAP, "defaultmap",
location);
c = build_omp_clause (location, OMP_CLAUSE_DEFAULTMAP);
OMP_CLAUSE_CHAIN (c) = list;
return c;
out_err:
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
return list;
}
static tree
cp_parser_omp_clause_ordered (cp_parser *parser,
tree list, location_t location)
{
tree c, num = NULL_TREE;
HOST_WIDE_INT n;
check_no_duplicate_clause (list, OMP_CLAUSE_ORDERED,
"ordered", location);
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
matching_parens parens;
parens.consume_open (parser);
num = cp_parser_constant_expression (parser);
if (!parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
if (num == error_mark_node)
return list;
num = fold_non_dependent_expr (num);
if (!tree_fits_shwi_p (num)
|| !INTEGRAL_TYPE_P (TREE_TYPE (num))
|| (n = tree_to_shwi (num)) <= 0
|| (int) n != n)
{
error_at (location,
"ordered argument needs positive constant integer "
"expression");
return list;
}
}
c = build_omp_clause (location, OMP_CLAUSE_ORDERED);
OMP_CLAUSE_ORDERED_EXPR (c) = num;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_reduction (cp_parser *parser, tree list)
{
enum tree_code code = ERROR_MARK;
tree nlist, c, id = NULL_TREE;
if (!cp_parser_require (parser, CPP_OPEN_PAREN, RT_OPEN_PAREN))
return list;
switch (cp_lexer_peek_token (parser->lexer)->type)
{
case CPP_PLUS: code = PLUS_EXPR; break;
case CPP_MULT: code = MULT_EXPR; break;
case CPP_MINUS: code = MINUS_EXPR; break;
case CPP_AND: code = BIT_AND_EXPR; break;
case CPP_XOR: code = BIT_XOR_EXPR; break;
case CPP_OR: code = BIT_IOR_EXPR; break;
case CPP_AND_AND: code = TRUTH_ANDIF_EXPR; break;
case CPP_OR_OR: code = TRUTH_ORIF_EXPR; break;
default: break;
}
if (code != ERROR_MARK)
cp_lexer_consume_token (parser->lexer);
else
{
bool saved_colon_corrects_to_scope_p;
saved_colon_corrects_to_scope_p = parser->colon_corrects_to_scope_p;
parser->colon_corrects_to_scope_p = false;
id = cp_parser_id_expression (parser, false,
true,
NULL,
false,
false);
parser->colon_corrects_to_scope_p = saved_colon_corrects_to_scope_p;
if (identifier_p (id))
{
const char *p = IDENTIFIER_POINTER (id);
if (strcmp (p, "min") == 0)
code = MIN_EXPR;
else if (strcmp (p, "max") == 0)
code = MAX_EXPR;
else if (id == ovl_op_identifier (false, PLUS_EXPR))
code = PLUS_EXPR;
else if (id == ovl_op_identifier (false, MULT_EXPR))
code = MULT_EXPR;
else if (id == ovl_op_identifier (false, MINUS_EXPR))
code = MINUS_EXPR;
else if (id == ovl_op_identifier (false, BIT_AND_EXPR))
code = BIT_AND_EXPR;
else if (id == ovl_op_identifier (false, BIT_IOR_EXPR))
code = BIT_IOR_EXPR;
else if (id == ovl_op_identifier (false, BIT_XOR_EXPR))
code = BIT_XOR_EXPR;
else if (id == ovl_op_identifier (false, TRUTH_ANDIF_EXPR))
code = TRUTH_ANDIF_EXPR;
else if (id == ovl_op_identifier (false, TRUTH_ORIF_EXPR))
code = TRUTH_ORIF_EXPR;
id = omp_reduction_id (code, id, NULL_TREE);
tree scope = parser->scope;
if (scope)
id = build_qualified_name (NULL_TREE, scope, id, false);
parser->scope = NULL_TREE;
parser->qualifying_scope = NULL_TREE;
parser->object_scope = NULL_TREE;
}
else
{
error ("invalid reduction-identifier");
resync_fail:
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
return list;
}
}
if (!cp_parser_require (parser, CPP_COLON, RT_COLON))
goto resync_fail;
nlist = cp_parser_omp_var_list_no_open (parser, OMP_CLAUSE_REDUCTION, list,
NULL);
for (c = nlist; c != list; c = OMP_CLAUSE_CHAIN (c))
{
OMP_CLAUSE_REDUCTION_CODE (c) = code;
OMP_CLAUSE_REDUCTION_PLACEHOLDER (c) = id;
}
return nlist;
}
static tree
cp_parser_omp_clause_schedule (cp_parser *parser, tree list, location_t location)
{
tree c, t;
int modifiers = 0, nmodifiers = 0;
matching_parens parens;
if (!parens.require_open (parser))
return list;
c = build_omp_clause (location, OMP_CLAUSE_SCHEDULE);
while (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp ("simd", p) == 0)
OMP_CLAUSE_SCHEDULE_SIMD (c) = 1;
else if (strcmp ("monotonic", p) == 0)
modifiers |= OMP_CLAUSE_SCHEDULE_MONOTONIC;
else if (strcmp ("nonmonotonic", p) == 0)
modifiers |= OMP_CLAUSE_SCHEDULE_NONMONOTONIC;
else
break;
cp_lexer_consume_token (parser->lexer);
if (nmodifiers++ == 0
&& cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
cp_lexer_consume_token (parser->lexer);
else
{
cp_parser_require (parser, CPP_COLON, RT_COLON);
break;
}
}
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
switch (p[0])
{
case 'd':
if (strcmp ("dynamic", p) != 0)
goto invalid_kind;
OMP_CLAUSE_SCHEDULE_KIND (c) = OMP_CLAUSE_SCHEDULE_DYNAMIC;
break;
case 'g':
if (strcmp ("guided", p) != 0)
goto invalid_kind;
OMP_CLAUSE_SCHEDULE_KIND (c) = OMP_CLAUSE_SCHEDULE_GUIDED;
break;
case 'r':
if (strcmp ("runtime", p) != 0)
goto invalid_kind;
OMP_CLAUSE_SCHEDULE_KIND (c) = OMP_CLAUSE_SCHEDULE_RUNTIME;
break;
default:
goto invalid_kind;
}
}
else if (cp_lexer_next_token_is_keyword (parser->lexer, RID_STATIC))
OMP_CLAUSE_SCHEDULE_KIND (c) = OMP_CLAUSE_SCHEDULE_STATIC;
else if (cp_lexer_next_token_is_keyword (parser->lexer, RID_AUTO))
OMP_CLAUSE_SCHEDULE_KIND (c) = OMP_CLAUSE_SCHEDULE_AUTO;
else
goto invalid_kind;
cp_lexer_consume_token (parser->lexer);
if ((modifiers & (OMP_CLAUSE_SCHEDULE_MONOTONIC
| OMP_CLAUSE_SCHEDULE_NONMONOTONIC))
== (OMP_CLAUSE_SCHEDULE_MONOTONIC
| OMP_CLAUSE_SCHEDULE_NONMONOTONIC))
{
error_at (location, "both %<monotonic%> and %<nonmonotonic%> modifiers "
"specified");
modifiers = 0;
}
if (cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
{
cp_token *token;
cp_lexer_consume_token (parser->lexer);
token = cp_lexer_peek_token (parser->lexer);
t = cp_parser_assignment_expression (parser);
if (t == error_mark_node)
goto resync_fail;
else if (OMP_CLAUSE_SCHEDULE_KIND (c) == OMP_CLAUSE_SCHEDULE_RUNTIME)
error_at (token->location, "schedule %<runtime%> does not take "
"a %<chunk_size%> parameter");
else if (OMP_CLAUSE_SCHEDULE_KIND (c) == OMP_CLAUSE_SCHEDULE_AUTO)
error_at (token->location, "schedule %<auto%> does not take "
"a %<chunk_size%> parameter");
else
OMP_CLAUSE_SCHEDULE_CHUNK_EXPR (c) = t;
if (!parens.require_close (parser))
goto resync_fail;
}
else if (!cp_parser_require (parser, CPP_CLOSE_PAREN, RT_COMMA_CLOSE_PAREN))
goto resync_fail;
OMP_CLAUSE_SCHEDULE_KIND (c)
= (enum omp_clause_schedule_kind)
(OMP_CLAUSE_SCHEDULE_KIND (c) | modifiers);
check_no_duplicate_clause (list, OMP_CLAUSE_SCHEDULE, "schedule", location);
OMP_CLAUSE_CHAIN (c) = list;
return c;
invalid_kind:
cp_parser_error (parser, "invalid schedule kind");
resync_fail:
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
return list;
}
static tree
cp_parser_omp_clause_untied (cp_parser * ,
tree list, location_t location)
{
tree c;
check_no_duplicate_clause (list, OMP_CLAUSE_UNTIED, "untied", location);
c = build_omp_clause (location, OMP_CLAUSE_UNTIED);
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_branch (cp_parser * , enum omp_clause_code code,
tree list, location_t location)
{
check_no_duplicate_clause (list, code, omp_clause_code_name[code], location);
tree c = build_omp_clause (location, code);
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_cancelkind (cp_parser * ,
enum omp_clause_code code,
tree list, location_t location)
{
tree c = build_omp_clause (location, code);
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_nogroup (cp_parser * ,
tree list, location_t location)
{
check_no_duplicate_clause (list, OMP_CLAUSE_NOGROUP, "nogroup", location);
tree c = build_omp_clause (location, OMP_CLAUSE_NOGROUP);
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_orderedkind (cp_parser * ,
enum omp_clause_code code,
tree list, location_t location)
{
check_no_duplicate_clause (list, code, omp_clause_code_name[code], location);
tree c = build_omp_clause (location, code);
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_num_teams (cp_parser *parser, tree list,
location_t location)
{
tree t, c;
matching_parens parens;
if (!parens.require_open (parser))
return list;
t = cp_parser_expression (parser);
if (t == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
check_no_duplicate_clause (list, OMP_CLAUSE_NUM_TEAMS,
"num_teams", location);
c = build_omp_clause (location, OMP_CLAUSE_NUM_TEAMS);
OMP_CLAUSE_NUM_TEAMS_EXPR (c) = t;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_thread_limit (cp_parser *parser, tree list,
location_t location)
{
tree t, c;
matching_parens parens;
if (!parens.require_open (parser))
return list;
t = cp_parser_expression (parser);
if (t == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
check_no_duplicate_clause (list, OMP_CLAUSE_THREAD_LIMIT,
"thread_limit", location);
c = build_omp_clause (location, OMP_CLAUSE_THREAD_LIMIT);
OMP_CLAUSE_THREAD_LIMIT_EXPR (c) = t;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_aligned (cp_parser *parser, tree list)
{
tree nlist, c, alignment = NULL_TREE;
bool colon;
matching_parens parens;
if (!parens.require_open (parser))
return list;
nlist = cp_parser_omp_var_list_no_open (parser, OMP_CLAUSE_ALIGNED, list,
&colon);
if (colon)
{
alignment = cp_parser_constant_expression (parser);
if (!parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
if (alignment == error_mark_node)
alignment = NULL_TREE;
}
for (c = nlist; c != list; c = OMP_CLAUSE_CHAIN (c))
OMP_CLAUSE_ALIGNED_ALIGNMENT (c) = alignment;
return nlist;
}
static tree
cp_parser_omp_clause_linear (cp_parser *parser, tree list, 
bool declare_simd)
{
tree nlist, c, step = integer_one_node;
bool colon;
enum omp_clause_linear_kind kind = OMP_CLAUSE_LINEAR_DEFAULT;
matching_parens parens;
if (!parens.require_open (parser))
return list;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp ("ref", p) == 0)
kind = OMP_CLAUSE_LINEAR_REF;
else if (strcmp ("val", p) == 0)
kind = OMP_CLAUSE_LINEAR_VAL;
else if (strcmp ("uval", p) == 0)
kind = OMP_CLAUSE_LINEAR_UVAL;
if (cp_lexer_nth_token_is (parser->lexer, 2, CPP_OPEN_PAREN))
cp_lexer_consume_token (parser->lexer);
else
kind = OMP_CLAUSE_LINEAR_DEFAULT;
}
if (kind == OMP_CLAUSE_LINEAR_DEFAULT)
nlist = cp_parser_omp_var_list_no_open (parser, OMP_CLAUSE_LINEAR, list,
&colon);
else
{
nlist = cp_parser_omp_var_list (parser, OMP_CLAUSE_LINEAR, list);
colon = cp_lexer_next_token_is (parser->lexer, CPP_COLON);
if (colon)
cp_parser_require (parser, CPP_COLON, RT_COLON);
else if (!parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
}
if (colon)
{
step = NULL_TREE;
if (declare_simd
&& cp_lexer_next_token_is (parser->lexer, CPP_NAME)
&& cp_lexer_nth_token_is (parser->lexer, 2, CPP_CLOSE_PAREN))
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
cp_parser_parse_tentatively (parser);
step = cp_parser_id_expression (parser, false,
true,
NULL,
false,
false);
if (step != error_mark_node)
step = cp_parser_lookup_name_simple (parser, step, token->location);
if (step == error_mark_node)
{
step = NULL_TREE;
cp_parser_abort_tentative_parse (parser);
}
else if (!cp_parser_parse_definitely (parser))
step = NULL_TREE;
}
if (!step)
step = cp_parser_expression (parser);
if (!parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
if (step == error_mark_node)
return list;
}
for (c = nlist; c != list; c = OMP_CLAUSE_CHAIN (c))
{
OMP_CLAUSE_LINEAR_STEP (c) = step;
OMP_CLAUSE_LINEAR_KIND (c) = kind;
}
return nlist;
}
static tree
cp_parser_omp_clause_safelen (cp_parser *parser, tree list,
location_t location)
{
tree t, c;
matching_parens parens;
if (!parens.require_open (parser))
return list;
t = cp_parser_constant_expression (parser);
if (t == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
check_no_duplicate_clause (list, OMP_CLAUSE_SAFELEN, "safelen", location);
c = build_omp_clause (location, OMP_CLAUSE_SAFELEN);
OMP_CLAUSE_SAFELEN_EXPR (c) = t;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_simdlen (cp_parser *parser, tree list,
location_t location)
{
tree t, c;
matching_parens parens;
if (!parens.require_open (parser))
return list;
t = cp_parser_constant_expression (parser);
if (t == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
check_no_duplicate_clause (list, OMP_CLAUSE_SIMDLEN, "simdlen", location);
c = build_omp_clause (location, OMP_CLAUSE_SIMDLEN);
OMP_CLAUSE_SIMDLEN_EXPR (c) = t;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_depend_sink (cp_parser *parser, location_t clause_loc,
tree list)
{
tree vec = NULL;
if (cp_lexer_next_token_is_not (parser->lexer, CPP_NAME))
{
cp_parser_error (parser, "expected identifier");
return list;
}
while (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
location_t id_loc = cp_lexer_peek_token (parser->lexer)->location;
tree t, identifier = cp_parser_identifier (parser);
tree addend = NULL;
if (identifier == error_mark_node)
t = error_mark_node;
else
{
t = cp_parser_lookup_name_simple
(parser, identifier,
cp_lexer_peek_token (parser->lexer)->location);
if (t == error_mark_node)
cp_parser_name_lookup_error (parser, identifier, t, NLE_NULL,
id_loc);
}
bool neg = false;
if (cp_lexer_next_token_is (parser->lexer, CPP_MINUS))
neg = true;
else if (!cp_lexer_next_token_is (parser->lexer, CPP_PLUS))
{
addend = integer_zero_node;
goto add_to_vector;
}
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_NUMBER))
{
cp_parser_error (parser, "expected integer");
return list;
}
addend = cp_lexer_peek_token (parser->lexer)->u.value;
if (TREE_CODE (addend) != INTEGER_CST)
{
cp_parser_error (parser, "expected integer");
return list;
}
cp_lexer_consume_token (parser->lexer);
add_to_vector:
if (t != error_mark_node)
{
vec = tree_cons (addend, t, vec);
if (neg)
OMP_CLAUSE_DEPEND_SINK_NEGATIVE (vec) = 1;
}
if (cp_lexer_next_token_is_not (parser->lexer, CPP_COMMA))
break;
cp_lexer_consume_token (parser->lexer);
}
if (cp_parser_require (parser, CPP_CLOSE_PAREN, RT_CLOSE_PAREN) && vec)
{
tree u = build_omp_clause (clause_loc, OMP_CLAUSE_DEPEND);
OMP_CLAUSE_DEPEND_KIND (u) = OMP_CLAUSE_DEPEND_SINK;
OMP_CLAUSE_DECL (u) = nreverse (vec);
OMP_CLAUSE_CHAIN (u) = list;
return u;
}
return list;
}
static tree
cp_parser_omp_clause_depend (cp_parser *parser, tree list, location_t loc)
{
tree nlist, c;
enum omp_clause_depend_kind kind = OMP_CLAUSE_DEPEND_INOUT;
matching_parens parens;
if (!parens.require_open (parser))
return list;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp ("in", p) == 0)
kind = OMP_CLAUSE_DEPEND_IN;
else if (strcmp ("inout", p) == 0)
kind = OMP_CLAUSE_DEPEND_INOUT;
else if (strcmp ("out", p) == 0)
kind = OMP_CLAUSE_DEPEND_OUT;
else if (strcmp ("source", p) == 0)
kind = OMP_CLAUSE_DEPEND_SOURCE;
else if (strcmp ("sink", p) == 0)
kind = OMP_CLAUSE_DEPEND_SINK;
else
goto invalid_kind;
}
else
goto invalid_kind;
cp_lexer_consume_token (parser->lexer);
if (kind == OMP_CLAUSE_DEPEND_SOURCE)
{
c = build_omp_clause (loc, OMP_CLAUSE_DEPEND);
OMP_CLAUSE_DEPEND_KIND (c) = kind;
OMP_CLAUSE_DECL (c) = NULL_TREE;
OMP_CLAUSE_CHAIN (c) = list;
if (!parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
return c;
}
if (!cp_parser_require (parser, CPP_COLON, RT_COLON))
goto resync_fail;
if (kind == OMP_CLAUSE_DEPEND_SINK)
nlist = cp_parser_omp_clause_depend_sink (parser, loc, list);
else
{
nlist = cp_parser_omp_var_list_no_open (parser, OMP_CLAUSE_DEPEND,
list, NULL);
for (c = nlist; c != list; c = OMP_CLAUSE_CHAIN (c))
OMP_CLAUSE_DEPEND_KIND (c) = kind;
}
return nlist;
invalid_kind:
cp_parser_error (parser, "invalid depend kind");
resync_fail:
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
return list;
}
static tree
cp_parser_omp_clause_map (cp_parser *parser, tree list)
{
tree nlist, c;
enum gomp_map_kind kind = GOMP_MAP_TOFROM;
bool always = false;
if (!cp_parser_require (parser, CPP_OPEN_PAREN, RT_OPEN_PAREN))
return list;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp ("always", p) == 0)
{
int nth = 2;
if (cp_lexer_peek_nth_token (parser->lexer, 2)->type == CPP_COMMA)
nth++;
if ((cp_lexer_peek_nth_token (parser->lexer, nth)->type == CPP_NAME
|| (cp_lexer_peek_nth_token (parser->lexer, nth)->keyword
== RID_DELETE))
&& (cp_lexer_peek_nth_token (parser->lexer, nth + 1)->type
== CPP_COLON))
{
always = true;
cp_lexer_consume_token (parser->lexer);
if (nth == 3)
cp_lexer_consume_token (parser->lexer);
}
}
}
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME)
&& cp_lexer_peek_nth_token (parser->lexer, 2)->type == CPP_COLON)
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp ("alloc", p) == 0)
kind = GOMP_MAP_ALLOC;
else if (strcmp ("to", p) == 0)
kind = always ? GOMP_MAP_ALWAYS_TO : GOMP_MAP_TO;
else if (strcmp ("from", p) == 0)
kind = always ? GOMP_MAP_ALWAYS_FROM : GOMP_MAP_FROM;
else if (strcmp ("tofrom", p) == 0)
kind = always ? GOMP_MAP_ALWAYS_TOFROM : GOMP_MAP_TOFROM;
else if (strcmp ("release", p) == 0)
kind = GOMP_MAP_RELEASE;
else
{
cp_parser_error (parser, "invalid map kind");
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
return list;
}
cp_lexer_consume_token (parser->lexer);
cp_lexer_consume_token (parser->lexer);
}
else if (cp_lexer_next_token_is_keyword (parser->lexer, RID_DELETE)
&& cp_lexer_peek_nth_token (parser->lexer, 2)->type == CPP_COLON)
{
kind = GOMP_MAP_DELETE;
cp_lexer_consume_token (parser->lexer);
cp_lexer_consume_token (parser->lexer);
}
nlist = cp_parser_omp_var_list_no_open (parser, OMP_CLAUSE_MAP, list,
NULL);
for (c = nlist; c != list; c = OMP_CLAUSE_CHAIN (c))
OMP_CLAUSE_SET_MAP_KIND (c, kind);
return nlist;
}
static tree
cp_parser_omp_clause_device (cp_parser *parser, tree list,
location_t location)
{
tree t, c;
matching_parens parens;
if (!parens.require_open (parser))
return list;
t = cp_parser_expression (parser);
if (t == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
check_no_duplicate_clause (list, OMP_CLAUSE_DEVICE,
"device", location);
c = build_omp_clause (location, OMP_CLAUSE_DEVICE);
OMP_CLAUSE_DEVICE_ID (c) = t;
OMP_CLAUSE_CHAIN (c) = list;
return c;
}
static tree
cp_parser_omp_clause_dist_schedule (cp_parser *parser, tree list,
location_t location)
{
tree c, t;
matching_parens parens;
if (!parens.require_open (parser))
return list;
c = build_omp_clause (location, OMP_CLAUSE_DIST_SCHEDULE);
if (!cp_lexer_next_token_is_keyword (parser->lexer, RID_STATIC))
goto invalid_kind;
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
{
cp_lexer_consume_token (parser->lexer);
t = cp_parser_assignment_expression (parser);
if (t == error_mark_node)
goto resync_fail;
OMP_CLAUSE_DIST_SCHEDULE_CHUNK_EXPR (c) = t;
if (!parens.require_close (parser))
goto resync_fail;
}
else if (!cp_parser_require (parser, CPP_CLOSE_PAREN, RT_COMMA_CLOSE_PAREN))
goto resync_fail;
check_no_duplicate_clause (list, OMP_CLAUSE_DIST_SCHEDULE, "dist_schedule",
location);
OMP_CLAUSE_CHAIN (c) = list;
return c;
invalid_kind:
cp_parser_error (parser, "invalid dist_schedule kind");
resync_fail:
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
return list;
}
static tree
cp_parser_omp_clause_proc_bind (cp_parser *parser, tree list,
location_t location)
{
tree c;
enum omp_clause_proc_bind_kind kind;
if (!cp_parser_require (parser, CPP_OPEN_PAREN, RT_OPEN_PAREN))
return list;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp ("master", p) == 0)
kind = OMP_CLAUSE_PROC_BIND_MASTER;
else if (strcmp ("close", p) == 0)
kind = OMP_CLAUSE_PROC_BIND_CLOSE;
else if (strcmp ("spread", p) == 0)
kind = OMP_CLAUSE_PROC_BIND_SPREAD;
else
goto invalid_kind;
}
else
goto invalid_kind;
cp_lexer_consume_token (parser->lexer);
if (!cp_parser_require (parser, CPP_CLOSE_PAREN, RT_COMMA_CLOSE_PAREN))
goto resync_fail;
c = build_omp_clause (location, OMP_CLAUSE_PROC_BIND);
check_no_duplicate_clause (list, OMP_CLAUSE_PROC_BIND, "proc_bind",
location);
OMP_CLAUSE_PROC_BIND_KIND (c) = kind;
OMP_CLAUSE_CHAIN (c) = list;
return c;
invalid_kind:
cp_parser_error (parser, "invalid depend kind");
resync_fail:
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
return list;
}
static tree
cp_parser_oacc_clause_async (cp_parser *parser, tree list)
{
tree c, t;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
t = build_int_cst (integer_type_node, GOMP_ASYNC_NOVAL);
if (cp_lexer_peek_token (parser->lexer)->type == CPP_OPEN_PAREN)
{
matching_parens parens;
parens.consume_open (parser);
t = cp_parser_expression (parser);
if (t == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
}
check_no_duplicate_clause (list, OMP_CLAUSE_ASYNC, "async", loc);
c = build_omp_clause (loc, OMP_CLAUSE_ASYNC);
OMP_CLAUSE_ASYNC_EXPR (c) = t;
OMP_CLAUSE_CHAIN (c) = list;
list = c;
return list;
}
static tree
cp_parser_oacc_all_clauses (cp_parser *parser, omp_clause_mask mask,
const char *where, cp_token *pragma_tok,
bool finish_p = true)
{
tree clauses = NULL;
bool first = true;
while (cp_lexer_next_token_is_not (parser->lexer, CPP_PRAGMA_EOL))
{
location_t here;
pragma_omp_clause c_kind;
omp_clause_code code;
const char *c_name;
tree prev = clauses;
if (!first && cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
cp_lexer_consume_token (parser->lexer);
here = cp_lexer_peek_token (parser->lexer)->location;
c_kind = cp_parser_omp_clause_name (parser);
switch (c_kind)
{
case PRAGMA_OACC_CLAUSE_ASYNC:
clauses = cp_parser_oacc_clause_async (parser, clauses);
c_name = "async";
break;
case PRAGMA_OACC_CLAUSE_AUTO:
clauses = cp_parser_oacc_simple_clause (parser, OMP_CLAUSE_AUTO,
clauses, here);
c_name = "auto";
break;
case PRAGMA_OACC_CLAUSE_COLLAPSE:
clauses = cp_parser_omp_clause_collapse (parser, clauses, here);
c_name = "collapse";
break;
case PRAGMA_OACC_CLAUSE_COPY:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "copy";
break;
case PRAGMA_OACC_CLAUSE_COPYIN:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "copyin";
break;
case PRAGMA_OACC_CLAUSE_COPYOUT:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "copyout";
break;
case PRAGMA_OACC_CLAUSE_CREATE:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "create";
break;
case PRAGMA_OACC_CLAUSE_DELETE:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "delete";
break;
case PRAGMA_OMP_CLAUSE_DEFAULT:
clauses = cp_parser_omp_clause_default (parser, clauses, here, true);
c_name = "default";
break;
case PRAGMA_OACC_CLAUSE_DEVICE:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "device";
break;
case PRAGMA_OACC_CLAUSE_DEVICEPTR:
clauses = cp_parser_oacc_data_clause_deviceptr (parser, clauses);
c_name = "deviceptr";
break;
case PRAGMA_OACC_CLAUSE_DEVICE_RESIDENT:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "device_resident";
break;
case PRAGMA_OACC_CLAUSE_FIRSTPRIVATE:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_FIRSTPRIVATE,
clauses);
c_name = "firstprivate";
break;
case PRAGMA_OACC_CLAUSE_GANG:
c_name = "gang";
clauses = cp_parser_oacc_shape_clause (parser, OMP_CLAUSE_GANG,
c_name, clauses);
break;
case PRAGMA_OACC_CLAUSE_HOST:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "host";
break;
case PRAGMA_OACC_CLAUSE_IF:
clauses = cp_parser_omp_clause_if (parser, clauses, here, false);
c_name = "if";
break;
case PRAGMA_OACC_CLAUSE_INDEPENDENT:
clauses = cp_parser_oacc_simple_clause (parser,
OMP_CLAUSE_INDEPENDENT,
clauses, here);
c_name = "independent";
break;
case PRAGMA_OACC_CLAUSE_LINK:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "link";
break;
case PRAGMA_OACC_CLAUSE_NUM_GANGS:
code = OMP_CLAUSE_NUM_GANGS;
c_name = "num_gangs";
clauses = cp_parser_oacc_single_int_clause (parser, code, c_name,
clauses);
break;
case PRAGMA_OACC_CLAUSE_NUM_WORKERS:
c_name = "num_workers";
code = OMP_CLAUSE_NUM_WORKERS;
clauses = cp_parser_oacc_single_int_clause (parser, code, c_name,
clauses);
break;
case PRAGMA_OACC_CLAUSE_PRESENT:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "present";
break;
case PRAGMA_OACC_CLAUSE_PRESENT_OR_COPY:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "present_or_copy";
break;
case PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYIN:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "present_or_copyin";
break;
case PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYOUT:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "present_or_copyout";
break;
case PRAGMA_OACC_CLAUSE_PRESENT_OR_CREATE:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "present_or_create";
break;
case PRAGMA_OACC_CLAUSE_PRIVATE:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_PRIVATE,
clauses);
c_name = "private";
break;
case PRAGMA_OACC_CLAUSE_REDUCTION:
clauses = cp_parser_omp_clause_reduction (parser, clauses);
c_name = "reduction";
break;
case PRAGMA_OACC_CLAUSE_SELF:
clauses = cp_parser_oacc_data_clause (parser, c_kind, clauses);
c_name = "self";
break;
case PRAGMA_OACC_CLAUSE_SEQ:
clauses = cp_parser_oacc_simple_clause (parser, OMP_CLAUSE_SEQ,
clauses, here);
c_name = "seq";
break;
case PRAGMA_OACC_CLAUSE_TILE:
clauses = cp_parser_oacc_clause_tile (parser, here, clauses);
c_name = "tile";
break;
case PRAGMA_OACC_CLAUSE_USE_DEVICE:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_USE_DEVICE_PTR,
clauses);
c_name = "use_device";
break;
case PRAGMA_OACC_CLAUSE_VECTOR:
c_name = "vector";
clauses = cp_parser_oacc_shape_clause (parser, OMP_CLAUSE_VECTOR,
c_name, clauses);
break;
case PRAGMA_OACC_CLAUSE_VECTOR_LENGTH:
c_name = "vector_length";
code = OMP_CLAUSE_VECTOR_LENGTH;
clauses = cp_parser_oacc_single_int_clause (parser, code, c_name,
clauses);
break;
case PRAGMA_OACC_CLAUSE_WAIT:
clauses = cp_parser_oacc_clause_wait (parser, clauses);
c_name = "wait";
break;
case PRAGMA_OACC_CLAUSE_WORKER:
c_name = "worker";
clauses = cp_parser_oacc_shape_clause (parser, OMP_CLAUSE_WORKER,
c_name, clauses);
break;
default:
cp_parser_error (parser, "expected %<#pragma acc%> clause");
goto saw_error;
}
first = false;
if (((mask >> c_kind) & 1) == 0)
{
clauses = prev;
error_at (here, "%qs is not valid for %qs", c_name, where);
}
}
saw_error:
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
if (finish_p)
return finish_omp_clauses (clauses, C_ORT_ACC);
return clauses;
}
static tree
cp_parser_omp_all_clauses (cp_parser *parser, omp_clause_mask mask,
const char *where, cp_token *pragma_tok,
bool finish_p = true)
{
tree clauses = NULL;
bool first = true;
cp_token *token = NULL;
while (cp_lexer_next_token_is_not (parser->lexer, CPP_PRAGMA_EOL))
{
pragma_omp_clause c_kind;
const char *c_name;
tree prev = clauses;
if (!first && cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
cp_lexer_consume_token (parser->lexer);
token = cp_lexer_peek_token (parser->lexer);
c_kind = cp_parser_omp_clause_name (parser);
switch (c_kind)
{
case PRAGMA_OMP_CLAUSE_COLLAPSE:
clauses = cp_parser_omp_clause_collapse (parser, clauses,
token->location);
c_name = "collapse";
break;
case PRAGMA_OMP_CLAUSE_COPYIN:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_COPYIN, clauses);
c_name = "copyin";
break;
case PRAGMA_OMP_CLAUSE_COPYPRIVATE:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_COPYPRIVATE,
clauses);
c_name = "copyprivate";
break;
case PRAGMA_OMP_CLAUSE_DEFAULT:
clauses = cp_parser_omp_clause_default (parser, clauses,
token->location, false);
c_name = "default";
break;
case PRAGMA_OMP_CLAUSE_FINAL:
clauses = cp_parser_omp_clause_final (parser, clauses, token->location);
c_name = "final";
break;
case PRAGMA_OMP_CLAUSE_FIRSTPRIVATE:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_FIRSTPRIVATE,
clauses);
c_name = "firstprivate";
break;
case PRAGMA_OMP_CLAUSE_GRAINSIZE:
clauses = cp_parser_omp_clause_grainsize (parser, clauses,
token->location);
c_name = "grainsize";
break;
case PRAGMA_OMP_CLAUSE_HINT:
clauses = cp_parser_omp_clause_hint (parser, clauses,
token->location);
c_name = "hint";
break;
case PRAGMA_OMP_CLAUSE_DEFAULTMAP:
clauses = cp_parser_omp_clause_defaultmap (parser, clauses,
token->location);
c_name = "defaultmap";
break;
case PRAGMA_OMP_CLAUSE_USE_DEVICE_PTR:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_USE_DEVICE_PTR,
clauses);
c_name = "use_device_ptr";
break;
case PRAGMA_OMP_CLAUSE_IS_DEVICE_PTR:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_IS_DEVICE_PTR,
clauses);
c_name = "is_device_ptr";
break;
case PRAGMA_OMP_CLAUSE_IF:
clauses = cp_parser_omp_clause_if (parser, clauses, token->location,
true);
c_name = "if";
break;
case PRAGMA_OMP_CLAUSE_LASTPRIVATE:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_LASTPRIVATE,
clauses);
c_name = "lastprivate";
break;
case PRAGMA_OMP_CLAUSE_MERGEABLE:
clauses = cp_parser_omp_clause_mergeable (parser, clauses,
token->location);
c_name = "mergeable";
break;
case PRAGMA_OMP_CLAUSE_NOWAIT:
clauses = cp_parser_omp_clause_nowait (parser, clauses, token->location);
c_name = "nowait";
break;
case PRAGMA_OMP_CLAUSE_NUM_TASKS:
clauses = cp_parser_omp_clause_num_tasks (parser, clauses,
token->location);
c_name = "num_tasks";
break;
case PRAGMA_OMP_CLAUSE_NUM_THREADS:
clauses = cp_parser_omp_clause_num_threads (parser, clauses,
token->location);
c_name = "num_threads";
break;
case PRAGMA_OMP_CLAUSE_ORDERED:
clauses = cp_parser_omp_clause_ordered (parser, clauses,
token->location);
c_name = "ordered";
break;
case PRAGMA_OMP_CLAUSE_PRIORITY:
clauses = cp_parser_omp_clause_priority (parser, clauses,
token->location);
c_name = "priority";
break;
case PRAGMA_OMP_CLAUSE_PRIVATE:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_PRIVATE,
clauses);
c_name = "private";
break;
case PRAGMA_OMP_CLAUSE_REDUCTION:
clauses = cp_parser_omp_clause_reduction (parser, clauses);
c_name = "reduction";
break;
case PRAGMA_OMP_CLAUSE_SCHEDULE:
clauses = cp_parser_omp_clause_schedule (parser, clauses,
token->location);
c_name = "schedule";
break;
case PRAGMA_OMP_CLAUSE_SHARED:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_SHARED,
clauses);
c_name = "shared";
break;
case PRAGMA_OMP_CLAUSE_UNTIED:
clauses = cp_parser_omp_clause_untied (parser, clauses,
token->location);
c_name = "untied";
break;
case PRAGMA_OMP_CLAUSE_INBRANCH:
clauses = cp_parser_omp_clause_branch (parser, OMP_CLAUSE_INBRANCH,
clauses, token->location);
c_name = "inbranch";
break;
case PRAGMA_OMP_CLAUSE_NOTINBRANCH:
clauses = cp_parser_omp_clause_branch (parser,
OMP_CLAUSE_NOTINBRANCH,
clauses, token->location);
c_name = "notinbranch";
break;
case PRAGMA_OMP_CLAUSE_PARALLEL:
clauses = cp_parser_omp_clause_cancelkind (parser, OMP_CLAUSE_PARALLEL,
clauses, token->location);
c_name = "parallel";
if (!first)
{
clause_not_first:
error_at (token->location, "%qs must be the first clause of %qs",
c_name, where);
clauses = prev;
}
break;
case PRAGMA_OMP_CLAUSE_FOR:
clauses = cp_parser_omp_clause_cancelkind (parser, OMP_CLAUSE_FOR,
clauses, token->location);
c_name = "for";
if (!first)
goto clause_not_first;
break;
case PRAGMA_OMP_CLAUSE_SECTIONS:
clauses = cp_parser_omp_clause_cancelkind (parser, OMP_CLAUSE_SECTIONS,
clauses, token->location);
c_name = "sections";
if (!first)
goto clause_not_first;
break;
case PRAGMA_OMP_CLAUSE_TASKGROUP:
clauses = cp_parser_omp_clause_cancelkind (parser, OMP_CLAUSE_TASKGROUP,
clauses, token->location);
c_name = "taskgroup";
if (!first)
goto clause_not_first;
break;
case PRAGMA_OMP_CLAUSE_LINK:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_LINK, clauses);
c_name = "to";
break;
case PRAGMA_OMP_CLAUSE_TO:
if ((mask & (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_LINK)) != 0)
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_TO_DECLARE,
clauses);
else
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_TO, clauses);
c_name = "to";
break;
case PRAGMA_OMP_CLAUSE_FROM:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_FROM, clauses);
c_name = "from";
break;
case PRAGMA_OMP_CLAUSE_UNIFORM:
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_UNIFORM,
clauses);
c_name = "uniform";
break;
case PRAGMA_OMP_CLAUSE_NUM_TEAMS:
clauses = cp_parser_omp_clause_num_teams (parser, clauses,
token->location);
c_name = "num_teams";
break;
case PRAGMA_OMP_CLAUSE_THREAD_LIMIT:
clauses = cp_parser_omp_clause_thread_limit (parser, clauses,
token->location);
c_name = "thread_limit";
break;
case PRAGMA_OMP_CLAUSE_ALIGNED:
clauses = cp_parser_omp_clause_aligned (parser, clauses);
c_name = "aligned";
break;
case PRAGMA_OMP_CLAUSE_LINEAR:
{
bool declare_simd = false;
if (((mask >> PRAGMA_OMP_CLAUSE_UNIFORM) & 1) != 0)
declare_simd = true;
clauses = cp_parser_omp_clause_linear (parser, clauses, declare_simd);
}
c_name = "linear";
break;
case PRAGMA_OMP_CLAUSE_DEPEND:
clauses = cp_parser_omp_clause_depend (parser, clauses,
token->location);
c_name = "depend";
break;
case PRAGMA_OMP_CLAUSE_MAP:
clauses = cp_parser_omp_clause_map (parser, clauses);
c_name = "map";
break;
case PRAGMA_OMP_CLAUSE_DEVICE:
clauses = cp_parser_omp_clause_device (parser, clauses,
token->location);
c_name = "device";
break;
case PRAGMA_OMP_CLAUSE_DIST_SCHEDULE:
clauses = cp_parser_omp_clause_dist_schedule (parser, clauses,
token->location);
c_name = "dist_schedule";
break;
case PRAGMA_OMP_CLAUSE_PROC_BIND:
clauses = cp_parser_omp_clause_proc_bind (parser, clauses,
token->location);
c_name = "proc_bind";
break;
case PRAGMA_OMP_CLAUSE_SAFELEN:
clauses = cp_parser_omp_clause_safelen (parser, clauses,
token->location);
c_name = "safelen";
break;
case PRAGMA_OMP_CLAUSE_SIMDLEN:
clauses = cp_parser_omp_clause_simdlen (parser, clauses,
token->location);
c_name = "simdlen";
break;
case PRAGMA_OMP_CLAUSE_NOGROUP:
clauses = cp_parser_omp_clause_nogroup (parser, clauses,
token->location);
c_name = "nogroup";
break;
case PRAGMA_OMP_CLAUSE_THREADS:
clauses
= cp_parser_omp_clause_orderedkind (parser, OMP_CLAUSE_THREADS,
clauses, token->location);
c_name = "threads";
break;
case PRAGMA_OMP_CLAUSE_SIMD:
clauses
= cp_parser_omp_clause_orderedkind (parser, OMP_CLAUSE_SIMD,
clauses, token->location);
c_name = "simd";
break;
default:
cp_parser_error (parser, "expected %<#pragma omp%> clause");
goto saw_error;
}
first = false;
if (((mask >> c_kind) & 1) == 0)
{
clauses = prev;
error_at (token->location, "%qs is not valid for %qs", c_name, where);
}
}
saw_error:
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
if (finish_p)
{
if ((mask & (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_UNIFORM)) != 0)
return finish_omp_clauses (clauses, C_ORT_OMP_DECLARE_SIMD);
else
return finish_omp_clauses (clauses, C_ORT_OMP);
}
return clauses;
}
static unsigned
cp_parser_begin_omp_structured_block (cp_parser *parser)
{
unsigned save = parser->in_statement;
if (parser->in_statement)
parser->in_statement = IN_OMP_BLOCK;
return save;
}
static void
cp_parser_end_omp_structured_block (cp_parser *parser, unsigned save)
{
parser->in_statement = save;
}
static tree
cp_parser_omp_structured_block (cp_parser *parser, bool *if_p)
{
tree stmt = begin_omp_structured_block ();
unsigned int save = cp_parser_begin_omp_structured_block (parser);
cp_parser_statement (parser, NULL_TREE, false, if_p);
cp_parser_end_omp_structured_block (parser, save);
return finish_omp_structured_block (stmt);
}
static void
cp_parser_omp_atomic (cp_parser *parser, cp_token *pragma_tok)
{
tree lhs = NULL_TREE, rhs = NULL_TREE, v = NULL_TREE, lhs1 = NULL_TREE;
tree rhs1 = NULL_TREE, orig_lhs;
enum tree_code code = OMP_ATOMIC, opcode = NOP_EXPR;
bool structured_block = false;
bool seq_cst = false;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (!strcmp (p, "seq_cst"))
{
seq_cst = true;
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is (parser->lexer, CPP_COMMA)
&& cp_lexer_peek_nth_token (parser->lexer, 2)->type == CPP_NAME)
cp_lexer_consume_token (parser->lexer);
}
}
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (!strcmp (p, "read"))
code = OMP_ATOMIC_READ;
else if (!strcmp (p, "write"))
code = NOP_EXPR;
else if (!strcmp (p, "update"))
code = OMP_ATOMIC;
else if (!strcmp (p, "capture"))
code = OMP_ATOMIC_CAPTURE_NEW;
else
p = NULL;
if (p)
cp_lexer_consume_token (parser->lexer);
}
if (!seq_cst)
{
if (cp_lexer_next_token_is (parser->lexer, CPP_COMMA)
&& cp_lexer_peek_nth_token (parser->lexer, 2)->type == CPP_NAME)
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (!strcmp (p, "seq_cst"))
{
seq_cst = true;
cp_lexer_consume_token (parser->lexer);
}
}
}
cp_parser_require_pragma_eol (parser, pragma_tok);
switch (code)
{
case OMP_ATOMIC_READ:
case NOP_EXPR: 
v = cp_parser_unary_expression (parser);
if (v == error_mark_node)
goto saw_error;
if (!cp_parser_require (parser, CPP_EQ, RT_EQ))
goto saw_error;
if (code == NOP_EXPR)
lhs = cp_parser_expression (parser);
else
lhs = cp_parser_unary_expression (parser);
if (lhs == error_mark_node)
goto saw_error;
if (code == NOP_EXPR)
{
code = OMP_ATOMIC;
rhs = lhs;
lhs = v;
v = NULL_TREE;
}
goto done;
case OMP_ATOMIC_CAPTURE_NEW:
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
cp_lexer_consume_token (parser->lexer);
structured_block = true;
}
else
{
v = cp_parser_unary_expression (parser);
if (v == error_mark_node)
goto saw_error;
if (!cp_parser_require (parser, CPP_EQ, RT_EQ))
goto saw_error;
}
default:
break;
}
restart:
lhs = cp_parser_unary_expression (parser);
orig_lhs = lhs;
switch (TREE_CODE (lhs))
{
case ERROR_MARK:
goto saw_error;
case POSTINCREMENT_EXPR:
if (code == OMP_ATOMIC_CAPTURE_NEW && !structured_block)
code = OMP_ATOMIC_CAPTURE_OLD;
case PREINCREMENT_EXPR:
lhs = TREE_OPERAND (lhs, 0);
opcode = PLUS_EXPR;
rhs = integer_one_node;
break;
case POSTDECREMENT_EXPR:
if (code == OMP_ATOMIC_CAPTURE_NEW && !structured_block)
code = OMP_ATOMIC_CAPTURE_OLD;
case PREDECREMENT_EXPR:
lhs = TREE_OPERAND (lhs, 0);
opcode = MINUS_EXPR;
rhs = integer_one_node;
break;
case COMPOUND_EXPR:
if (TREE_CODE (TREE_OPERAND (lhs, 0)) == SAVE_EXPR
&& TREE_CODE (TREE_OPERAND (lhs, 1)) == COMPOUND_EXPR
&& TREE_CODE (TREE_OPERAND (TREE_OPERAND (lhs, 1), 0)) == MODIFY_EXPR
&& TREE_OPERAND (TREE_OPERAND (lhs, 1), 1) == TREE_OPERAND (lhs, 0)
&& TREE_CODE (TREE_TYPE (TREE_OPERAND (TREE_OPERAND
(TREE_OPERAND (lhs, 1), 0), 0)))
== BOOLEAN_TYPE)
lhs = TREE_OPERAND (TREE_OPERAND (lhs, 1), 0);
case MODIFY_EXPR:
if (TREE_CODE (lhs) == MODIFY_EXPR
&& TREE_CODE (TREE_TYPE (TREE_OPERAND (lhs, 0))) == BOOLEAN_TYPE)
{
if (integer_onep (TREE_OPERAND (lhs, 1)))
{
rhs = TREE_OPERAND (lhs, 1);
lhs = TREE_OPERAND (lhs, 0);
opcode = NOP_EXPR;
if (code == OMP_ATOMIC_CAPTURE_NEW
&& !structured_block
&& TREE_CODE (orig_lhs) == COMPOUND_EXPR)
code = OMP_ATOMIC_CAPTURE_OLD;
break;
}
}
default:
switch (cp_lexer_peek_token (parser->lexer)->type)
{
case CPP_MULT_EQ:
opcode = MULT_EXPR;
break;
case CPP_DIV_EQ:
opcode = TRUNC_DIV_EXPR;
break;
case CPP_PLUS_EQ:
opcode = PLUS_EXPR;
break;
case CPP_MINUS_EQ:
opcode = MINUS_EXPR;
break;
case CPP_LSHIFT_EQ:
opcode = LSHIFT_EXPR;
break;
case CPP_RSHIFT_EQ:
opcode = RSHIFT_EXPR;
break;
case CPP_AND_EQ:
opcode = BIT_AND_EXPR;
break;
case CPP_OR_EQ:
opcode = BIT_IOR_EXPR;
break;
case CPP_XOR_EQ:
opcode = BIT_XOR_EXPR;
break;
case CPP_EQ:
enum cp_parser_prec oprec;
cp_token *token;
cp_lexer_consume_token (parser->lexer);
cp_parser_parse_tentatively (parser);
rhs1 = cp_parser_simple_cast_expression (parser);
if (rhs1 == error_mark_node)
{
cp_parser_abort_tentative_parse (parser);
cp_parser_simple_cast_expression (parser);
goto saw_error;
}
token = cp_lexer_peek_token (parser->lexer);
if (token->type != CPP_SEMICOLON && !cp_tree_equal (lhs, rhs1))
{
cp_parser_abort_tentative_parse (parser);
cp_parser_parse_tentatively (parser);
rhs = cp_parser_binary_expression (parser, false, true,
PREC_NOT_OPERATOR, NULL);
if (rhs == error_mark_node)
{
cp_parser_abort_tentative_parse (parser);
cp_parser_binary_expression (parser, false, true,
PREC_NOT_OPERATOR, NULL);
goto saw_error;
}
switch (TREE_CODE (rhs))
{
case MULT_EXPR:
case TRUNC_DIV_EXPR:
case RDIV_EXPR:
case PLUS_EXPR:
case MINUS_EXPR:
case LSHIFT_EXPR:
case RSHIFT_EXPR:
case BIT_AND_EXPR:
case BIT_IOR_EXPR:
case BIT_XOR_EXPR:
if (cp_tree_equal (lhs, TREE_OPERAND (rhs, 1)))
{
if (cp_parser_parse_definitely (parser))
{
opcode = TREE_CODE (rhs);
rhs1 = TREE_OPERAND (rhs, 0);
rhs = TREE_OPERAND (rhs, 1);
goto stmt_done;
}
else
goto saw_error;
}
break;
default:
break;
}
cp_parser_abort_tentative_parse (parser);
if (structured_block && code == OMP_ATOMIC_CAPTURE_OLD)
{
rhs = cp_parser_expression (parser);
if (rhs == error_mark_node)
goto saw_error;
opcode = NOP_EXPR;
rhs1 = NULL_TREE;
goto stmt_done;
}
cp_parser_error (parser,
"invalid form of %<#pragma omp atomic%>");
goto saw_error;
}
if (!cp_parser_parse_definitely (parser))
goto saw_error;
switch (token->type)
{
case CPP_SEMICOLON:
if (structured_block && code == OMP_ATOMIC_CAPTURE_NEW)
{
code = OMP_ATOMIC_CAPTURE_OLD;
v = lhs;
lhs = NULL_TREE;
lhs1 = rhs1;
rhs1 = NULL_TREE;
cp_lexer_consume_token (parser->lexer);
goto restart;
}
else if (structured_block)
{
opcode = NOP_EXPR;
rhs = rhs1;
rhs1 = NULL_TREE;
goto stmt_done;
}
cp_parser_error (parser,
"invalid form of %<#pragma omp atomic%>");
goto saw_error;
case CPP_MULT:
opcode = MULT_EXPR;
break;
case CPP_DIV:
opcode = TRUNC_DIV_EXPR;
break;
case CPP_PLUS:
opcode = PLUS_EXPR;
break;
case CPP_MINUS:
opcode = MINUS_EXPR;
break;
case CPP_LSHIFT:
opcode = LSHIFT_EXPR;
break;
case CPP_RSHIFT:
opcode = RSHIFT_EXPR;
break;
case CPP_AND:
opcode = BIT_AND_EXPR;
break;
case CPP_OR:
opcode = BIT_IOR_EXPR;
break;
case CPP_XOR:
opcode = BIT_XOR_EXPR;
break;
default:
cp_parser_error (parser,
"invalid operator for %<#pragma omp atomic%>");
goto saw_error;
}
oprec = TOKEN_PRECEDENCE (token);
gcc_assert (oprec != PREC_NOT_OPERATOR);
if (commutative_tree_code (opcode))
oprec = (enum cp_parser_prec) (oprec - 1);
cp_lexer_consume_token (parser->lexer);
rhs = cp_parser_binary_expression (parser, false, false,
oprec, NULL);
if (rhs == error_mark_node)
goto saw_error;
goto stmt_done;
default:
cp_parser_error (parser,
"invalid operator for %<#pragma omp atomic%>");
goto saw_error;
}
cp_lexer_consume_token (parser->lexer);
rhs = cp_parser_expression (parser);
if (rhs == error_mark_node)
goto saw_error;
break;
}
stmt_done:
if (structured_block && code == OMP_ATOMIC_CAPTURE_NEW)
{
if (!cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON))
goto saw_error;
v = cp_parser_unary_expression (parser);
if (v == error_mark_node)
goto saw_error;
if (!cp_parser_require (parser, CPP_EQ, RT_EQ))
goto saw_error;
lhs1 = cp_parser_unary_expression (parser);
if (lhs1 == error_mark_node)
goto saw_error;
}
if (structured_block)
{
cp_parser_consume_semicolon_at_end_of_statement (parser);
cp_parser_require (parser, CPP_CLOSE_BRACE, RT_CLOSE_BRACE);
}
done:
finish_omp_atomic (code, opcode, lhs, rhs, v, lhs1, rhs1, seq_cst);
if (!structured_block)
cp_parser_consume_semicolon_at_end_of_statement (parser);
return;
saw_error:
cp_parser_skip_to_end_of_block_or_statement (parser);
if (structured_block)
{
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_BRACE))
cp_lexer_consume_token (parser->lexer);
else if (code == OMP_ATOMIC_CAPTURE_NEW)
{
cp_parser_skip_to_end_of_block_or_statement (parser);
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_BRACE))
cp_lexer_consume_token (parser->lexer);
}
}
}
static void
cp_parser_omp_barrier (cp_parser *parser, cp_token *pragma_tok)
{
cp_parser_require_pragma_eol (parser, pragma_tok);
finish_omp_barrier ();
}
#define OMP_CRITICAL_CLAUSE_MASK		\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_HINT) )
static tree
cp_parser_omp_critical (cp_parser *parser, cp_token *pragma_tok, bool *if_p)
{
tree stmt, name = NULL_TREE, clauses = NULL_TREE;
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
matching_parens parens;
parens.consume_open (parser);
name = cp_parser_identifier (parser);
if (name == error_mark_node
|| !parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
if (name == error_mark_node)
name = NULL;
clauses = cp_parser_omp_all_clauses (parser,
OMP_CRITICAL_CLAUSE_MASK,
"#pragma omp critical", pragma_tok);
}
else
cp_parser_require_pragma_eol (parser, pragma_tok);
stmt = cp_parser_omp_structured_block (parser, if_p);
return c_finish_omp_critical (input_location, stmt, name, clauses);
}
static void
cp_parser_omp_flush (cp_parser *parser, cp_token *pragma_tok)
{
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
(void) cp_parser_omp_var_list (parser, OMP_CLAUSE_ERROR, NULL);
cp_parser_require_pragma_eol (parser, pragma_tok);
finish_omp_flush ();
}
static tree
cp_parser_omp_for_cond (cp_parser *parser, tree decl)
{
tree cond = cp_parser_binary_expression (parser, false, true,
PREC_NOT_OPERATOR, NULL);
if (cond == error_mark_node
|| cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
{
cp_parser_skip_to_end_of_statement (parser);
return error_mark_node;
}
switch (TREE_CODE (cond))
{
case GT_EXPR:
case GE_EXPR:
case LT_EXPR:
case LE_EXPR:
break;
case NE_EXPR:
gcc_fallthrough ();
default:
return error_mark_node;
}
if (decl
&& (type_dependent_expression_p (decl)
|| CLASS_TYPE_P (TREE_TYPE (decl))))
return cond;
return build_x_binary_op (EXPR_LOC_OR_LOC (cond, input_location),
TREE_CODE (cond),
TREE_OPERAND (cond, 0), ERROR_MARK,
TREE_OPERAND (cond, 1), ERROR_MARK,
NULL, tf_warning_or_error);
}
static tree
cp_parser_omp_for_incr (cp_parser *parser, tree decl)
{
cp_token *token = cp_lexer_peek_token (parser->lexer);
enum tree_code op;
tree lhs, rhs;
cp_id_kind idk;
bool decl_first;
if (token->type == CPP_PLUS_PLUS || token->type == CPP_MINUS_MINUS)
{
op = (token->type == CPP_PLUS_PLUS
? PREINCREMENT_EXPR : PREDECREMENT_EXPR);
cp_lexer_consume_token (parser->lexer);
lhs = cp_parser_simple_cast_expression (parser);
if (lhs != decl
&& (!processing_template_decl || !cp_tree_equal (lhs, decl)))
return error_mark_node;
return build2 (op, TREE_TYPE (decl), decl, NULL_TREE);
}
lhs = cp_parser_primary_expression (parser, false, false, false, &idk);
if (lhs != decl
&& (!processing_template_decl || !cp_tree_equal (lhs, decl)))
return error_mark_node;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_PLUS_PLUS || token->type == CPP_MINUS_MINUS)
{
op = (token->type == CPP_PLUS_PLUS
? POSTINCREMENT_EXPR : POSTDECREMENT_EXPR);
cp_lexer_consume_token (parser->lexer);
return build2 (op, TREE_TYPE (decl), decl, NULL_TREE);
}
op = cp_parser_assignment_operator_opt (parser);
if (op == ERROR_MARK)
return error_mark_node;
if (op != NOP_EXPR)
{
rhs = cp_parser_assignment_expression (parser);
rhs = build2 (op, TREE_TYPE (decl), decl, rhs);
return build2 (MODIFY_EXPR, TREE_TYPE (decl), decl, rhs);
}
lhs = cp_parser_binary_expression (parser, false, false,
PREC_ADDITIVE_EXPRESSION, NULL);
token = cp_lexer_peek_token (parser->lexer);
decl_first = (lhs == decl
|| (processing_template_decl && cp_tree_equal (lhs, decl)));
if (decl_first)
lhs = NULL_TREE;
if (token->type != CPP_PLUS
&& token->type != CPP_MINUS)
return error_mark_node;
do
{
op = token->type == CPP_PLUS ? PLUS_EXPR : MINUS_EXPR;
cp_lexer_consume_token (parser->lexer);
rhs = cp_parser_binary_expression (parser, false, false,
PREC_ADDITIVE_EXPRESSION, NULL);
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_PLUS || token->type == CPP_MINUS || decl_first)
{
if (lhs == NULL_TREE)
{
if (op == PLUS_EXPR)
lhs = rhs;
else
lhs = build_x_unary_op (input_location, NEGATE_EXPR, rhs,
tf_warning_or_error);
}
else
lhs = build_x_binary_op (input_location, op, lhs, ERROR_MARK, rhs,
ERROR_MARK, NULL, tf_warning_or_error);
}
}
while (token->type == CPP_PLUS || token->type == CPP_MINUS);
if (!decl_first)
{
if ((rhs != decl
&& (!processing_template_decl || !cp_tree_equal (rhs, decl)))
|| op == MINUS_EXPR)
return error_mark_node;
rhs = build2 (op, TREE_TYPE (decl), lhs, decl);
}
else
rhs = build2 (PLUS_EXPR, TREE_TYPE (decl), decl, lhs);
return build2 (MODIFY_EXPR, TREE_TYPE (decl), decl, rhs);
}
static tree
cp_parser_omp_for_loop_init (cp_parser *parser,
tree &this_pre_body,
vec<tree, va_gc> *&for_block,
tree &init,
tree &orig_init,
tree &decl,
tree &real_decl)
{
if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
return NULL_TREE;
tree add_private_clause = NULL_TREE;
cp_decl_specifier_seq type_specifiers;
cp_parser_parse_tentatively (parser);
cp_parser_type_specifier_seq (parser, true,
false,
&type_specifiers);
if (cp_parser_parse_definitely (parser))
{
tree asm_specification, attributes;
cp_declarator *declarator;
declarator = cp_parser_declarator (parser,
CP_PARSER_DECLARATOR_NAMED,
NULL,
NULL,
false,
false);
attributes = cp_parser_attributes_opt (parser);
asm_specification = cp_parser_asm_specification_opt (parser);
if (declarator == cp_error_declarator) 
cp_parser_skip_to_end_of_statement (parser);
else 
{
tree pushed_scope, auto_node;
decl = start_decl (declarator, &type_specifiers,
SD_INITIALIZED, attributes,
NULL_TREE,
&pushed_scope);
auto_node = type_uses_auto (TREE_TYPE (decl));
if (cp_lexer_next_token_is_not (parser->lexer, CPP_EQ))
{
if (cp_lexer_next_token_is (parser->lexer, 
CPP_OPEN_PAREN))
error ("parenthesized initialization is not allowed in "
"OpenMP %<for%> loop");
else
cp_parser_require (parser, CPP_EQ, RT_EQ);
init = error_mark_node;
cp_parser_skip_to_end_of_statement (parser);
}
else if (CLASS_TYPE_P (TREE_TYPE (decl))
|| type_dependent_expression_p (decl)
|| auto_node)
{
bool is_direct_init, is_non_constant_init;
init = cp_parser_initializer (parser,
&is_direct_init,
&is_non_constant_init);
if (auto_node)
{
TREE_TYPE (decl)
= do_auto_deduction (TREE_TYPE (decl), init,
auto_node);
if (!CLASS_TYPE_P (TREE_TYPE (decl))
&& !type_dependent_expression_p (decl))
goto non_class;
}
cp_finish_decl (decl, init, !is_non_constant_init,
asm_specification,
LOOKUP_ONLYCONVERTING);
orig_init = init;
if (CLASS_TYPE_P (TREE_TYPE (decl)))
{
vec_safe_push (for_block, this_pre_body);
init = NULL_TREE;
}
else
{
init = pop_stmt_list (this_pre_body);
if (init && TREE_CODE (init) == STATEMENT_LIST)
{
tree_stmt_iterator i = tsi_start (init);
while (!tsi_end_p (i))
{
tree t = tsi_stmt (i);
if (TREE_CODE (t) == DECL_EXPR
&& TREE_CODE (DECL_EXPR_DECL (t)) == TYPE_DECL)
{
tsi_delink (&i);
vec_safe_push (for_block, t);
continue;
}
break;
}
if (tsi_one_before_end_p (i))
{
tree t = tsi_stmt (i);
tsi_delink (&i);
free_stmt_list (init);
init = t;
}
}
}
this_pre_body = NULL_TREE;
}
else
{
cp_lexer_consume_token (parser->lexer);
init = cp_parser_assignment_expression (parser);
non_class:
if (TREE_CODE (TREE_TYPE (decl)) == REFERENCE_TYPE)
init = error_mark_node;
else
cp_finish_decl (decl, NULL_TREE,
false,
asm_specification,
LOOKUP_ONLYCONVERTING);
}
if (pushed_scope)
pop_scope (pushed_scope);
}
}
else 
{
cp_id_kind idk;
cp_parser_parse_tentatively (parser);
decl = cp_parser_primary_expression (parser, false, false,
false, &idk);
cp_token *last_tok = cp_lexer_peek_token (parser->lexer);
if (!cp_parser_error_occurred (parser)
&& decl
&& (TREE_CODE (decl) == COMPONENT_REF
|| (TREE_CODE (decl) == SCOPE_REF && TREE_TYPE (decl))))
{
cp_parser_abort_tentative_parse (parser);
cp_parser_parse_tentatively (parser);
cp_token *token = cp_lexer_peek_token (parser->lexer);
tree name = cp_parser_id_expression (parser, false,
true,
NULL,
false,
false);
if (name != error_mark_node
&& last_tok == cp_lexer_peek_token (parser->lexer))
{
decl = cp_parser_lookup_name_simple (parser, name,
token->location);
if (TREE_CODE (decl) == FIELD_DECL)
add_private_clause = omp_privatize_field (decl, false);
}
cp_parser_abort_tentative_parse (parser);
cp_parser_parse_tentatively (parser);
decl = cp_parser_primary_expression (parser, false, false,
false, &idk);
}
if (!cp_parser_error_occurred (parser)
&& decl
&& DECL_P (decl)
&& CLASS_TYPE_P (TREE_TYPE (decl)))
{
tree rhs;
cp_parser_parse_definitely (parser);
cp_parser_require (parser, CPP_EQ, RT_EQ);
rhs = cp_parser_assignment_expression (parser);
orig_init = rhs;
finish_expr_stmt (build_x_modify_expr (EXPR_LOCATION (rhs),
decl, NOP_EXPR,
rhs,
tf_warning_or_error));
if (!add_private_clause)
add_private_clause = decl;
}
else
{
decl = NULL;
cp_parser_abort_tentative_parse (parser);
init = cp_parser_expression (parser);
if (init)
{
if (TREE_CODE (init) == MODIFY_EXPR
|| TREE_CODE (init) == MODOP_EXPR)
real_decl = TREE_OPERAND (init, 0);
}
}
}
return add_private_clause;
}
static tree
cp_parser_omp_for_loop (cp_parser *parser, enum tree_code code, tree clauses,
tree *cclauses, bool *if_p)
{
tree init, orig_init, cond, incr, body, decl, pre_body = NULL_TREE, ret;
tree real_decl, initv, condv, incrv, declv;
tree this_pre_body, cl, ordered_cl = NULL_TREE;
location_t loc_first;
bool collapse_err = false;
int i, collapse = 1, ordered = 0, count, nbraces = 0;
vec<tree, va_gc> *for_block = make_tree_vector ();
auto_vec<tree, 4> orig_inits;
bool tiling = false;
for (cl = clauses; cl; cl = OMP_CLAUSE_CHAIN (cl))
if (OMP_CLAUSE_CODE (cl) == OMP_CLAUSE_COLLAPSE)
collapse = tree_to_shwi (OMP_CLAUSE_COLLAPSE_EXPR (cl));
else if (OMP_CLAUSE_CODE (cl) == OMP_CLAUSE_TILE)
{
tiling = true;
collapse = list_length (OMP_CLAUSE_TILE_LIST (cl));
}
else if (OMP_CLAUSE_CODE (cl) == OMP_CLAUSE_ORDERED
&& OMP_CLAUSE_ORDERED_EXPR (cl))
{
ordered_cl = cl;
ordered = tree_to_shwi (OMP_CLAUSE_ORDERED_EXPR (cl));
}
if (ordered && ordered < collapse)
{
error_at (OMP_CLAUSE_LOCATION (ordered_cl),
"%<ordered%> clause parameter is less than %<collapse%>");
OMP_CLAUSE_ORDERED_EXPR (ordered_cl)
= build_int_cst (NULL_TREE, collapse);
ordered = collapse;
}
if (ordered)
{
for (tree *pc = &clauses; *pc; )
if (OMP_CLAUSE_CODE (*pc) == OMP_CLAUSE_LINEAR)
{
error_at (OMP_CLAUSE_LOCATION (*pc),
"%<linear%> clause may not be specified together "
"with %<ordered%> clause with a parameter");
*pc = OMP_CLAUSE_CHAIN (*pc);
}
else
pc = &OMP_CLAUSE_CHAIN (*pc);
}
gcc_assert (tiling || (collapse >= 1 && ordered >= 0));
count = ordered ? ordered : collapse;
declv = make_tree_vec (count);
initv = make_tree_vec (count);
condv = make_tree_vec (count);
incrv = make_tree_vec (count);
loc_first = cp_lexer_peek_token (parser->lexer)->location;
for (i = 0; i < count; i++)
{
int bracecount = 0;
tree add_private_clause = NULL_TREE;
location_t loc;
if (!cp_lexer_next_token_is_keyword (parser->lexer, RID_FOR))
{
if (!collapse_err)
cp_parser_error (parser, "for statement expected");
return NULL;
}
loc = cp_lexer_consume_token (parser->lexer)->location;
matching_parens parens;
if (!parens.require_open (parser))
return NULL;
init = orig_init = decl = real_decl = NULL;
this_pre_body = push_stmt_list ();
add_private_clause
= cp_parser_omp_for_loop_init (parser, this_pre_body, for_block,
init, orig_init, decl, real_decl);
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
if (this_pre_body)
{
this_pre_body = pop_stmt_list (this_pre_body);
if (pre_body)
{
tree t = pre_body;
pre_body = push_stmt_list ();
add_stmt (t);
add_stmt (this_pre_body);
pre_body = pop_stmt_list (pre_body);
}
else
pre_body = this_pre_body;
}
if (decl)
real_decl = decl;
if (cclauses != NULL
&& cclauses[C_OMP_CLAUSE_SPLIT_PARALLEL] != NULL
&& real_decl != NULL_TREE)
{
tree *c;
for (c = &cclauses[C_OMP_CLAUSE_SPLIT_PARALLEL]; *c ; )
if (OMP_CLAUSE_CODE (*c) == OMP_CLAUSE_FIRSTPRIVATE
&& OMP_CLAUSE_DECL (*c) == real_decl)
{
error_at (loc, "iteration variable %qD"
" should not be firstprivate", real_decl);
*c = OMP_CLAUSE_CHAIN (*c);
}
else if (OMP_CLAUSE_CODE (*c) == OMP_CLAUSE_LASTPRIVATE
&& OMP_CLAUSE_DECL (*c) == real_decl)
{
tree l = *c;
*c = OMP_CLAUSE_CHAIN (*c);
if (code == OMP_SIMD)
{
OMP_CLAUSE_CHAIN (l) = cclauses[C_OMP_CLAUSE_SPLIT_FOR];
cclauses[C_OMP_CLAUSE_SPLIT_FOR] = l;
}
else
{
OMP_CLAUSE_CHAIN (l) = clauses;
clauses = l;
}
add_private_clause = NULL_TREE;
}
else
{
if (OMP_CLAUSE_CODE (*c) == OMP_CLAUSE_PRIVATE
&& OMP_CLAUSE_DECL (*c) == real_decl)
add_private_clause = NULL_TREE;
c = &OMP_CLAUSE_CHAIN (*c);
}
}
if (add_private_clause)
{
tree c;
for (c = clauses; c ; c = OMP_CLAUSE_CHAIN (c))
{
if ((OMP_CLAUSE_CODE (c) == OMP_CLAUSE_PRIVATE
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LASTPRIVATE)
&& OMP_CLAUSE_DECL (c) == decl)
break;
else if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_FIRSTPRIVATE
&& OMP_CLAUSE_DECL (c) == decl)
error_at (loc, "iteration variable %qD "
"should not be firstprivate",
decl);
else if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_REDUCTION
&& OMP_CLAUSE_DECL (c) == decl)
error_at (loc, "iteration variable %qD should not be reduction",
decl);
}
if (c == NULL)
{
if (code != OMP_SIMD)
c = build_omp_clause (loc, OMP_CLAUSE_PRIVATE);
else if (collapse == 1)
c = build_omp_clause (loc, OMP_CLAUSE_LINEAR);
else
c = build_omp_clause (loc, OMP_CLAUSE_LASTPRIVATE);
OMP_CLAUSE_DECL (c) = add_private_clause;
c = finish_omp_clauses (c, C_ORT_OMP);
if (c)
{
OMP_CLAUSE_CHAIN (c) = clauses;
clauses = c;
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LINEAR)
OMP_CLAUSE_LINEAR_STEP (c) = NULL_TREE;
}
}
}
cond = NULL;
if (cp_lexer_next_token_is_not (parser->lexer, CPP_SEMICOLON))
cond = cp_parser_omp_for_cond (parser, decl);
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
incr = NULL;
if (cp_lexer_next_token_is_not (parser->lexer, CPP_CLOSE_PAREN))
{
if (real_decl
&& ((processing_template_decl
&& (TREE_TYPE (real_decl) == NULL_TREE
|| !POINTER_TYPE_P (TREE_TYPE (real_decl))))
|| CLASS_TYPE_P (TREE_TYPE (real_decl))))
incr = cp_parser_omp_for_incr (parser, real_decl);
else
incr = cp_parser_expression (parser);
if (!EXPR_HAS_LOCATION (incr))
protected_set_expr_location (incr, input_location);
}
if (!parens.require_close (parser))
cp_parser_skip_to_closing_parenthesis (parser, true,
false,
true);
TREE_VEC_ELT (declv, i) = decl;
TREE_VEC_ELT (initv, i) = init;
TREE_VEC_ELT (condv, i) = cond;
TREE_VEC_ELT (incrv, i) = incr;
if (orig_init)
{
orig_inits.safe_grow_cleared (i + 1);
orig_inits[i] = orig_init;
}
if (i == count - 1)
break;
cp_parser_parse_tentatively (parser);
for (;;)
{
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_FOR))
break;
else if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_BRACE))
{
cp_lexer_consume_token (parser->lexer);
bracecount++;
}
else if (bracecount
&& cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
cp_lexer_consume_token (parser->lexer);
else
{
loc = cp_lexer_peek_token (parser->lexer)->location;
error_at (loc, "not enough for loops to collapse");
collapse_err = true;
cp_parser_abort_tentative_parse (parser);
declv = NULL_TREE;
break;
}
}
if (declv)
{
cp_parser_parse_definitely (parser);
nbraces += bracecount;
}
}
if (nbraces)
if_p = NULL;
parser->in_statement = IN_OMP_FOR;
body = push_stmt_list ();
cp_parser_statement (parser, NULL_TREE, false, if_p);
body = pop_stmt_list (body);
if (declv == NULL_TREE)
ret = NULL_TREE;
else
ret = finish_omp_for (loc_first, code, declv, NULL, initv, condv, incrv,
body, pre_body, &orig_inits, clauses);
while (nbraces)
{
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_BRACE))
{
cp_lexer_consume_token (parser->lexer);
nbraces--;
}
else if (cp_lexer_next_token_is (parser->lexer, CPP_SEMICOLON))
cp_lexer_consume_token (parser->lexer);
else
{
if (!collapse_err)
{
error_at (cp_lexer_peek_token (parser->lexer)->location,
"collapsed loops not perfectly nested");
}
collapse_err = true;
cp_parser_statement_seq_opt (parser, NULL);
if (cp_lexer_next_token_is (parser->lexer, CPP_EOF))
break;
}
}
while (!for_block->is_empty ())
{
tree t = for_block->pop ();
if (TREE_CODE (t) == STATEMENT_LIST)
add_stmt (pop_stmt_list (t));
else
add_stmt (t);
}
release_tree_vector (for_block);
return ret;
}
static void
cp_omp_split_clauses (location_t loc, enum tree_code code,
omp_clause_mask mask, tree clauses, tree *cclauses)
{
int i;
c_omp_split_clauses (loc, code, mask, clauses, cclauses);
for (i = 0; i < C_OMP_CLAUSE_SPLIT_COUNT; i++)
if (cclauses[i])
cclauses[i] = finish_omp_clauses (cclauses[i], C_ORT_OMP);
}
#define OMP_SIMD_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_SAFELEN)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_SIMDLEN)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_LINEAR)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_ALIGNED)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_LASTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_REDUCTION)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_COLLAPSE))
static tree
cp_parser_omp_simd (cp_parser *parser, cp_token *pragma_tok,
char *p_name, omp_clause_mask mask, tree *cclauses,
bool *if_p)
{
tree clauses, sb, ret;
unsigned int save;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
strcat (p_name, " simd");
mask |= OMP_SIMD_CLAUSE_MASK;
clauses = cp_parser_omp_all_clauses (parser, mask, p_name, pragma_tok,
cclauses == NULL);
if (cclauses)
{
cp_omp_split_clauses (loc, OMP_SIMD, mask, clauses, cclauses);
clauses = cclauses[C_OMP_CLAUSE_SPLIT_SIMD];
tree c = omp_find_clause (cclauses[C_OMP_CLAUSE_SPLIT_FOR],
OMP_CLAUSE_ORDERED);
if (c && OMP_CLAUSE_ORDERED_EXPR (c))
{
error_at (OMP_CLAUSE_LOCATION (c),
"%<ordered%> clause with parameter may not be specified "
"on %qs construct", p_name);
OMP_CLAUSE_ORDERED_EXPR (c) = NULL_TREE;
}
}
sb = begin_omp_structured_block ();
save = cp_parser_begin_omp_structured_block (parser);
ret = cp_parser_omp_for_loop (parser, OMP_SIMD, clauses, cclauses, if_p);
cp_parser_end_omp_structured_block (parser, save);
add_stmt (finish_omp_structured_block (sb));
return ret;
}
#define OMP_FOR_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FIRSTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_LASTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_LINEAR)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_REDUCTION)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_ORDERED)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_SCHEDULE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NOWAIT)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_COLLAPSE))
static tree
cp_parser_omp_for (cp_parser *parser, cp_token *pragma_tok,
char *p_name, omp_clause_mask mask, tree *cclauses,
bool *if_p)
{
tree clauses, sb, ret;
unsigned int save;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
strcat (p_name, " for");
mask |= OMP_FOR_CLAUSE_MASK;
if (cclauses && (mask & (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_MAP)) == 0)
mask &= ~(OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NOWAIT);
if ((mask & (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DIST_SCHEDULE)) != 0)
mask &= ~(OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_ORDERED);
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp (p, "simd") == 0)
{
tree cclauses_buf[C_OMP_CLAUSE_SPLIT_COUNT];
if (cclauses == NULL)
cclauses = cclauses_buf;
cp_lexer_consume_token (parser->lexer);
if (!flag_openmp)  
return cp_parser_omp_simd (parser, pragma_tok, p_name, mask,
cclauses, if_p);
sb = begin_omp_structured_block ();
save = cp_parser_begin_omp_structured_block (parser);
ret = cp_parser_omp_simd (parser, pragma_tok, p_name, mask,
cclauses, if_p);
cp_parser_end_omp_structured_block (parser, save);
tree body = finish_omp_structured_block (sb);
if (ret == NULL)
return ret;
ret = make_node (OMP_FOR);
TREE_TYPE (ret) = void_type_node;
OMP_FOR_BODY (ret) = body;
OMP_FOR_CLAUSES (ret) = cclauses[C_OMP_CLAUSE_SPLIT_FOR];
SET_EXPR_LOCATION (ret, loc);
add_stmt (ret);
return ret;
}
}
if (!flag_openmp)  
{
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return NULL_TREE;
}
if ((mask & (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DIST_SCHEDULE)) != 0)
mask &= ~(OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_LINEAR);
clauses = cp_parser_omp_all_clauses (parser, mask, p_name, pragma_tok,
cclauses == NULL);
if (cclauses)
{
cp_omp_split_clauses (loc, OMP_FOR, mask, clauses, cclauses);
clauses = cclauses[C_OMP_CLAUSE_SPLIT_FOR];
}
sb = begin_omp_structured_block ();
save = cp_parser_begin_omp_structured_block (parser);
ret = cp_parser_omp_for_loop (parser, OMP_FOR, clauses, cclauses, if_p);
cp_parser_end_omp_structured_block (parser, save);
add_stmt (finish_omp_structured_block (sb));
return ret;
}
static tree
cp_parser_omp_master (cp_parser *parser, cp_token *pragma_tok, bool *if_p)
{
cp_parser_require_pragma_eol (parser, pragma_tok);
return c_finish_omp_master (input_location,
cp_parser_omp_structured_block (parser, if_p));
}
#define OMP_ORDERED_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_THREADS)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_SIMD))
#define OMP_ORDERED_DEPEND_CLAUSE_MASK				\
(OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEPEND)
static bool
cp_parser_omp_ordered (cp_parser *parser, cp_token *pragma_tok,
enum pragma_context context, bool *if_p)
{
location_t loc = pragma_tok->location;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp (p, "depend") == 0)
{
if (!flag_openmp)	
{
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return false;
}
if (context == pragma_stmt)
{
error_at (pragma_tok->location, "%<#pragma omp ordered%> with "
"%<depend%> clause may only be used in compound "
"statements");
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return false;
}
tree clauses
= cp_parser_omp_all_clauses (parser,
OMP_ORDERED_DEPEND_CLAUSE_MASK,
"#pragma omp ordered", pragma_tok);
c_finish_omp_ordered (loc, clauses, NULL_TREE);
return false;
}
}
tree clauses
= cp_parser_omp_all_clauses (parser, OMP_ORDERED_CLAUSE_MASK,
"#pragma omp ordered", pragma_tok);
if (!flag_openmp     
&& omp_find_clause (clauses, OMP_CLAUSE_SIMD) == NULL_TREE)
return false;
c_finish_omp_ordered (loc, clauses,
cp_parser_omp_structured_block (parser, if_p));
return true;
}
static tree
cp_parser_omp_sections_scope (cp_parser *parser)
{
tree stmt, substmt;
bool error_suppress = false;
cp_token *tok;
matching_braces braces;
if (!braces.require_open (parser))
return NULL_TREE;
stmt = push_stmt_list ();
if (cp_parser_pragma_kind (cp_lexer_peek_token (parser->lexer))
!= PRAGMA_OMP_SECTION)
{
substmt = cp_parser_omp_structured_block (parser, NULL);
substmt = build1 (OMP_SECTION, void_type_node, substmt);
add_stmt (substmt);
}
while (1)
{
tok = cp_lexer_peek_token (parser->lexer);
if (tok->type == CPP_CLOSE_BRACE)
break;
if (tok->type == CPP_EOF)
break;
if (cp_parser_pragma_kind (tok) == PRAGMA_OMP_SECTION)
{
cp_lexer_consume_token (parser->lexer);
cp_parser_require_pragma_eol (parser, tok);
error_suppress = false;
}
else if (!error_suppress)
{
cp_parser_error (parser, "expected %<#pragma omp section%> or %<}%>");
error_suppress = true;
}
substmt = cp_parser_omp_structured_block (parser, NULL);
substmt = build1 (OMP_SECTION, void_type_node, substmt);
add_stmt (substmt);
}
braces.require_close (parser);
substmt = pop_stmt_list (stmt);
stmt = make_node (OMP_SECTIONS);
TREE_TYPE (stmt) = void_type_node;
OMP_SECTIONS_BODY (stmt) = substmt;
add_stmt (stmt);
return stmt;
}
#define OMP_SECTIONS_CLAUSE_MASK				\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FIRSTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_LASTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_REDUCTION)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NOWAIT))
static tree
cp_parser_omp_sections (cp_parser *parser, cp_token *pragma_tok,
char *p_name, omp_clause_mask mask, tree *cclauses)
{
tree clauses, ret;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
strcat (p_name, " sections");
mask |= OMP_SECTIONS_CLAUSE_MASK;
if (cclauses)
mask &= ~(OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NOWAIT);
clauses = cp_parser_omp_all_clauses (parser, mask, p_name, pragma_tok,
cclauses == NULL);
if (cclauses)
{
cp_omp_split_clauses (loc, OMP_SECTIONS, mask, clauses, cclauses);
clauses = cclauses[C_OMP_CLAUSE_SPLIT_SECTIONS];
}
ret = cp_parser_omp_sections_scope (parser);
if (ret)
OMP_SECTIONS_CLAUSES (ret) = clauses;
return ret;
}
#define OMP_PARALLEL_CLAUSE_MASK				\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_IF)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FIRSTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEFAULT)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_SHARED)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_COPYIN)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_REDUCTION)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NUM_THREADS)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PROC_BIND))
static tree
cp_parser_omp_parallel (cp_parser *parser, cp_token *pragma_tok,
char *p_name, omp_clause_mask mask, tree *cclauses,
bool *if_p)
{
tree stmt, clauses, block;
unsigned int save;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
strcat (p_name, " parallel");
mask |= OMP_PARALLEL_CLAUSE_MASK;
if ((mask & (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_MAP)) != 0
&& (mask & (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DIST_SCHEDULE)) == 0)
mask &= ~(OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_COPYIN);
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_FOR))
{
tree cclauses_buf[C_OMP_CLAUSE_SPLIT_COUNT];
if (cclauses == NULL)
cclauses = cclauses_buf;
cp_lexer_consume_token (parser->lexer);
if (!flag_openmp)  
return cp_parser_omp_for (parser, pragma_tok, p_name, mask, cclauses,
if_p);
block = begin_omp_parallel ();
save = cp_parser_begin_omp_structured_block (parser);
tree ret = cp_parser_omp_for (parser, pragma_tok, p_name, mask, cclauses,
if_p);
cp_parser_end_omp_structured_block (parser, save);
stmt = finish_omp_parallel (cclauses[C_OMP_CLAUSE_SPLIT_PARALLEL],
block);
if (ret == NULL_TREE)
return ret;
OMP_PARALLEL_COMBINED (stmt) = 1;
return stmt;
}
else if (cclauses
&& (mask & (OMP_CLAUSE_MASK_1
<< PRAGMA_OMP_CLAUSE_DIST_SCHEDULE)) != 0)
{
error_at (loc, "expected %<for%> after %qs", p_name);
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return NULL_TREE;
}
else if (!flag_openmp)  
{
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return NULL_TREE;
}
else if (cclauses == NULL && cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp (p, "sections") == 0)
{
tree cclauses_buf[C_OMP_CLAUSE_SPLIT_COUNT];
cclauses = cclauses_buf;
cp_lexer_consume_token (parser->lexer);
block = begin_omp_parallel ();
save = cp_parser_begin_omp_structured_block (parser);
cp_parser_omp_sections (parser, pragma_tok, p_name, mask, cclauses);
cp_parser_end_omp_structured_block (parser, save);
stmt = finish_omp_parallel (cclauses[C_OMP_CLAUSE_SPLIT_PARALLEL],
block);
OMP_PARALLEL_COMBINED (stmt) = 1;
return stmt;
}
}
clauses = cp_parser_omp_all_clauses (parser, mask, p_name, pragma_tok,
cclauses == NULL);
if (cclauses)
{
cp_omp_split_clauses (loc, OMP_PARALLEL, mask, clauses, cclauses);
clauses = cclauses[C_OMP_CLAUSE_SPLIT_PARALLEL];
}
block = begin_omp_parallel ();
save = cp_parser_begin_omp_structured_block (parser);
cp_parser_statement (parser, NULL_TREE, false, if_p);
cp_parser_end_omp_structured_block (parser, save);
stmt = finish_omp_parallel (clauses, block);
return stmt;
}
#define OMP_SINGLE_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FIRSTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_COPYPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NOWAIT))
static tree
cp_parser_omp_single (cp_parser *parser, cp_token *pragma_tok, bool *if_p)
{
tree stmt = make_node (OMP_SINGLE);
TREE_TYPE (stmt) = void_type_node;
OMP_SINGLE_CLAUSES (stmt)
= cp_parser_omp_all_clauses (parser, OMP_SINGLE_CLAUSE_MASK,
"#pragma omp single", pragma_tok);
OMP_SINGLE_BODY (stmt) = cp_parser_omp_structured_block (parser, if_p);
return add_stmt (stmt);
}
#define OMP_TASK_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_IF)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_UNTIED)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEFAULT)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FIRSTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_SHARED)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FINAL)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_MERGEABLE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEPEND)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PRIORITY))
static tree
cp_parser_omp_task (cp_parser *parser, cp_token *pragma_tok, bool *if_p)
{
tree clauses, block;
unsigned int save;
clauses = cp_parser_omp_all_clauses (parser, OMP_TASK_CLAUSE_MASK,
"#pragma omp task", pragma_tok);
block = begin_omp_task ();
save = cp_parser_begin_omp_structured_block (parser);
cp_parser_statement (parser, NULL_TREE, false, if_p);
cp_parser_end_omp_structured_block (parser, save);
return finish_omp_task (clauses, block);
}
static void
cp_parser_omp_taskwait (cp_parser *parser, cp_token *pragma_tok)
{
cp_parser_require_pragma_eol (parser, pragma_tok);
finish_omp_taskwait ();
}
static void
cp_parser_omp_taskyield (cp_parser *parser, cp_token *pragma_tok)
{
cp_parser_require_pragma_eol (parser, pragma_tok);
finish_omp_taskyield ();
}
static tree
cp_parser_omp_taskgroup (cp_parser *parser, cp_token *pragma_tok, bool *if_p)
{
cp_parser_require_pragma_eol (parser, pragma_tok);
return c_finish_omp_taskgroup (input_location,
cp_parser_omp_structured_block (parser,
if_p));
}
static void
cp_parser_omp_threadprivate (cp_parser *parser, cp_token *pragma_tok)
{
tree vars;
vars = cp_parser_omp_var_list (parser, OMP_CLAUSE_ERROR, NULL);
cp_parser_require_pragma_eol (parser, pragma_tok);
finish_omp_threadprivate (vars);
}
#define OMP_CANCEL_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PARALLEL)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FOR)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_SECTIONS)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_TASKGROUP)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_IF))
static void
cp_parser_omp_cancel (cp_parser *parser, cp_token *pragma_tok)
{
tree clauses = cp_parser_omp_all_clauses (parser, OMP_CANCEL_CLAUSE_MASK,
"#pragma omp cancel", pragma_tok);
finish_omp_cancel (clauses);
}
#define OMP_CANCELLATION_POINT_CLAUSE_MASK			\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PARALLEL)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FOR)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_SECTIONS)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_TASKGROUP))
static void
cp_parser_omp_cancellation_point (cp_parser *parser, cp_token *pragma_tok,
enum pragma_context context)
{
tree clauses;
bool point_seen = false;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp (p, "point") == 0)
{
cp_lexer_consume_token (parser->lexer);
point_seen = true;
}
}
if (!point_seen)
{
cp_parser_error (parser, "expected %<point%>");
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return;
}
if (context != pragma_compound)
{
if (context == pragma_stmt)
error_at (pragma_tok->location,
"%<#pragma %s%> may only be used in compound statements",
"omp cancellation point");
else
cp_parser_error (parser, "expected declaration specifiers");
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return;
}
clauses = cp_parser_omp_all_clauses (parser,
OMP_CANCELLATION_POINT_CLAUSE_MASK,
"#pragma omp cancellation point",
pragma_tok);
finish_omp_cancellation_point (clauses);
}
#define OMP_DISTRIBUTE_CLAUSE_MASK				\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FIRSTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_LASTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DIST_SCHEDULE)\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_COLLAPSE))
static tree
cp_parser_omp_distribute (cp_parser *parser, cp_token *pragma_tok,
char *p_name, omp_clause_mask mask, tree *cclauses,
bool *if_p)
{
tree clauses, sb, ret;
unsigned int save;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
strcat (p_name, " distribute");
mask |= OMP_DISTRIBUTE_CLAUSE_MASK;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
bool simd = false;
bool parallel = false;
if (strcmp (p, "simd") == 0)
simd = true;
else
parallel = strcmp (p, "parallel") == 0;
if (parallel || simd)
{
tree cclauses_buf[C_OMP_CLAUSE_SPLIT_COUNT];
if (cclauses == NULL)
cclauses = cclauses_buf;
cp_lexer_consume_token (parser->lexer);
if (!flag_openmp)  
{
if (simd)
return cp_parser_omp_simd (parser, pragma_tok, p_name, mask,
cclauses, if_p);
else
return cp_parser_omp_parallel (parser, pragma_tok, p_name, mask,
cclauses, if_p);
}
sb = begin_omp_structured_block ();
save = cp_parser_begin_omp_structured_block (parser);
if (simd)
ret = cp_parser_omp_simd (parser, pragma_tok, p_name, mask,
cclauses, if_p);
else
ret = cp_parser_omp_parallel (parser, pragma_tok, p_name, mask,
cclauses, if_p);
cp_parser_end_omp_structured_block (parser, save);
tree body = finish_omp_structured_block (sb);
if (ret == NULL)
return ret;
ret = make_node (OMP_DISTRIBUTE);
TREE_TYPE (ret) = void_type_node;
OMP_FOR_BODY (ret) = body;
OMP_FOR_CLAUSES (ret) = cclauses[C_OMP_CLAUSE_SPLIT_DISTRIBUTE];
SET_EXPR_LOCATION (ret, loc);
add_stmt (ret);
return ret;
}
}
if (!flag_openmp)  
{
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return NULL_TREE;
}
clauses = cp_parser_omp_all_clauses (parser, mask, p_name, pragma_tok,
cclauses == NULL);
if (cclauses)
{
cp_omp_split_clauses (loc, OMP_DISTRIBUTE, mask, clauses, cclauses);
clauses = cclauses[C_OMP_CLAUSE_SPLIT_DISTRIBUTE];
}
sb = begin_omp_structured_block ();
save = cp_parser_begin_omp_structured_block (parser);
ret = cp_parser_omp_for_loop (parser, OMP_DISTRIBUTE, clauses, NULL, if_p);
cp_parser_end_omp_structured_block (parser, save);
add_stmt (finish_omp_structured_block (sb));
return ret;
}
#define OMP_TEAMS_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FIRSTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_SHARED)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_REDUCTION)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NUM_TEAMS)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_THREAD_LIMIT)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEFAULT))
static tree
cp_parser_omp_teams (cp_parser *parser, cp_token *pragma_tok,
char *p_name, omp_clause_mask mask, tree *cclauses,
bool *if_p)
{
tree clauses, sb, ret;
unsigned int save;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
strcat (p_name, " teams");
mask |= OMP_TEAMS_CLAUSE_MASK;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp (p, "distribute") == 0)
{
tree cclauses_buf[C_OMP_CLAUSE_SPLIT_COUNT];
if (cclauses == NULL)
cclauses = cclauses_buf;
cp_lexer_consume_token (parser->lexer);
if (!flag_openmp)  
return cp_parser_omp_distribute (parser, pragma_tok, p_name, mask,
cclauses, if_p);
sb = begin_omp_structured_block ();
save = cp_parser_begin_omp_structured_block (parser);
ret = cp_parser_omp_distribute (parser, pragma_tok, p_name, mask,
cclauses, if_p);
cp_parser_end_omp_structured_block (parser, save);
tree body = finish_omp_structured_block (sb);
if (ret == NULL)
return ret;
clauses = cclauses[C_OMP_CLAUSE_SPLIT_TEAMS];
ret = make_node (OMP_TEAMS);
TREE_TYPE (ret) = void_type_node;
OMP_TEAMS_CLAUSES (ret) = clauses;
OMP_TEAMS_BODY (ret) = body;
OMP_TEAMS_COMBINED (ret) = 1;
SET_EXPR_LOCATION (ret, loc);
return add_stmt (ret);
}
}
if (!flag_openmp)  
{
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return NULL_TREE;
}
clauses = cp_parser_omp_all_clauses (parser, mask, p_name, pragma_tok,
cclauses == NULL);
if (cclauses)
{
cp_omp_split_clauses (loc, OMP_TEAMS, mask, clauses, cclauses);
clauses = cclauses[C_OMP_CLAUSE_SPLIT_TEAMS];
}
tree stmt = make_node (OMP_TEAMS);
TREE_TYPE (stmt) = void_type_node;
OMP_TEAMS_CLAUSES (stmt) = clauses;
OMP_TEAMS_BODY (stmt) = cp_parser_omp_structured_block (parser, if_p);
SET_EXPR_LOCATION (stmt, loc);
return add_stmt (stmt);
}
#define OMP_TARGET_DATA_CLAUSE_MASK				\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEVICE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_MAP)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_IF)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_USE_DEVICE_PTR))
static tree
cp_parser_omp_target_data (cp_parser *parser, cp_token *pragma_tok, bool *if_p)
{
tree clauses
= cp_parser_omp_all_clauses (parser, OMP_TARGET_DATA_CLAUSE_MASK,
"#pragma omp target data", pragma_tok);
int map_seen = 0;
for (tree *pc = &clauses; *pc;)
{
if (OMP_CLAUSE_CODE (*pc) == OMP_CLAUSE_MAP)
switch (OMP_CLAUSE_MAP_KIND (*pc))
{
case GOMP_MAP_TO:
case GOMP_MAP_ALWAYS_TO:
case GOMP_MAP_FROM:
case GOMP_MAP_ALWAYS_FROM:
case GOMP_MAP_TOFROM:
case GOMP_MAP_ALWAYS_TOFROM:
case GOMP_MAP_ALLOC:
map_seen = 3;
break;
case GOMP_MAP_FIRSTPRIVATE_POINTER:
case GOMP_MAP_FIRSTPRIVATE_REFERENCE:
case GOMP_MAP_ALWAYS_POINTER:
break;
default:
map_seen |= 1;
error_at (OMP_CLAUSE_LOCATION (*pc),
"%<#pragma omp target data%> with map-type other "
"than %<to%>, %<from%>, %<tofrom%> or %<alloc%> "
"on %<map%> clause");
*pc = OMP_CLAUSE_CHAIN (*pc);
continue;
}
pc = &OMP_CLAUSE_CHAIN (*pc);
}
if (map_seen != 3)
{
if (map_seen == 0)
error_at (pragma_tok->location,
"%<#pragma omp target data%> must contain at least "
"one %<map%> clause");
return NULL_TREE;
}
tree stmt = make_node (OMP_TARGET_DATA);
TREE_TYPE (stmt) = void_type_node;
OMP_TARGET_DATA_CLAUSES (stmt) = clauses;
keep_next_level (true);
OMP_TARGET_DATA_BODY (stmt) = cp_parser_omp_structured_block (parser, if_p);
SET_EXPR_LOCATION (stmt, pragma_tok->location);
return add_stmt (stmt);
}
#define OMP_TARGET_ENTER_DATA_CLAUSE_MASK			\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEVICE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_MAP)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_IF)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEPEND)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NOWAIT))
static tree
cp_parser_omp_target_enter_data (cp_parser *parser, cp_token *pragma_tok,
enum pragma_context context)
{
bool data_seen = false;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp (p, "data") == 0)
{
cp_lexer_consume_token (parser->lexer);
data_seen = true;
}
}
if (!data_seen)
{
cp_parser_error (parser, "expected %<data%>");
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return NULL_TREE;
}
if (context == pragma_stmt)
{
error_at (pragma_tok->location,
"%<#pragma %s%> may only be used in compound statements",
"omp target enter data");
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return NULL_TREE;
}
tree clauses
= cp_parser_omp_all_clauses (parser, OMP_TARGET_ENTER_DATA_CLAUSE_MASK,
"#pragma omp target enter data", pragma_tok);
int map_seen = 0;
for (tree *pc = &clauses; *pc;)
{
if (OMP_CLAUSE_CODE (*pc) == OMP_CLAUSE_MAP)
switch (OMP_CLAUSE_MAP_KIND (*pc))
{
case GOMP_MAP_TO:
case GOMP_MAP_ALWAYS_TO:
case GOMP_MAP_ALLOC:
map_seen = 3;
break;
case GOMP_MAP_FIRSTPRIVATE_POINTER:
case GOMP_MAP_FIRSTPRIVATE_REFERENCE:
case GOMP_MAP_ALWAYS_POINTER:
break;
default:
map_seen |= 1;
error_at (OMP_CLAUSE_LOCATION (*pc),
"%<#pragma omp target enter data%> with map-type other "
"than %<to%> or %<alloc%> on %<map%> clause");
*pc = OMP_CLAUSE_CHAIN (*pc);
continue;
}
pc = &OMP_CLAUSE_CHAIN (*pc);
}
if (map_seen != 3)
{
if (map_seen == 0)
error_at (pragma_tok->location,
"%<#pragma omp target enter data%> must contain at least "
"one %<map%> clause");
return NULL_TREE;
}
tree stmt = make_node (OMP_TARGET_ENTER_DATA);
TREE_TYPE (stmt) = void_type_node;
OMP_TARGET_ENTER_DATA_CLAUSES (stmt) = clauses;
SET_EXPR_LOCATION (stmt, pragma_tok->location);
return add_stmt (stmt);
}
#define OMP_TARGET_EXIT_DATA_CLAUSE_MASK			\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEVICE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_MAP)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_IF)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEPEND)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NOWAIT))
static tree
cp_parser_omp_target_exit_data (cp_parser *parser, cp_token *pragma_tok,
enum pragma_context context)
{
bool data_seen = false;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp (p, "data") == 0)
{
cp_lexer_consume_token (parser->lexer);
data_seen = true;
}
}
if (!data_seen)
{
cp_parser_error (parser, "expected %<data%>");
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return NULL_TREE;
}
if (context == pragma_stmt)
{
error_at (pragma_tok->location,
"%<#pragma %s%> may only be used in compound statements",
"omp target exit data");
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return NULL_TREE;
}
tree clauses
= cp_parser_omp_all_clauses (parser, OMP_TARGET_EXIT_DATA_CLAUSE_MASK,
"#pragma omp target exit data", pragma_tok);
int map_seen = 0;
for (tree *pc = &clauses; *pc;)
{
if (OMP_CLAUSE_CODE (*pc) == OMP_CLAUSE_MAP)
switch (OMP_CLAUSE_MAP_KIND (*pc))
{
case GOMP_MAP_FROM:
case GOMP_MAP_ALWAYS_FROM:
case GOMP_MAP_RELEASE:
case GOMP_MAP_DELETE:
map_seen = 3;
break;
case GOMP_MAP_FIRSTPRIVATE_POINTER:
case GOMP_MAP_FIRSTPRIVATE_REFERENCE:
case GOMP_MAP_ALWAYS_POINTER:
break;
default:
map_seen |= 1;
error_at (OMP_CLAUSE_LOCATION (*pc),
"%<#pragma omp target exit data%> with map-type other "
"than %<from%>, %<release%> or %<delete%> on %<map%>"
" clause");
*pc = OMP_CLAUSE_CHAIN (*pc);
continue;
}
pc = &OMP_CLAUSE_CHAIN (*pc);
}
if (map_seen != 3)
{
if (map_seen == 0)
error_at (pragma_tok->location,
"%<#pragma omp target exit data%> must contain at least "
"one %<map%> clause");
return NULL_TREE;
}
tree stmt = make_node (OMP_TARGET_EXIT_DATA);
TREE_TYPE (stmt) = void_type_node;
OMP_TARGET_EXIT_DATA_CLAUSES (stmt) = clauses;
SET_EXPR_LOCATION (stmt, pragma_tok->location);
return add_stmt (stmt);
}
#define OMP_TARGET_UPDATE_CLAUSE_MASK				\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FROM)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_TO)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEVICE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_IF)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEPEND)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NOWAIT))
static bool
cp_parser_omp_target_update (cp_parser *parser, cp_token *pragma_tok,
enum pragma_context context)
{
if (context == pragma_stmt)
{
error_at (pragma_tok->location,
"%<#pragma %s%> may only be used in compound statements",
"omp target update");
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return false;
}
tree clauses
= cp_parser_omp_all_clauses (parser, OMP_TARGET_UPDATE_CLAUSE_MASK,
"#pragma omp target update", pragma_tok);
if (omp_find_clause (clauses, OMP_CLAUSE_TO) == NULL_TREE
&& omp_find_clause (clauses, OMP_CLAUSE_FROM) == NULL_TREE)
{
error_at (pragma_tok->location,
"%<#pragma omp target update%> must contain at least one "
"%<from%> or %<to%> clauses");
return false;
}
tree stmt = make_node (OMP_TARGET_UPDATE);
TREE_TYPE (stmt) = void_type_node;
OMP_TARGET_UPDATE_CLAUSES (stmt) = clauses;
SET_EXPR_LOCATION (stmt, pragma_tok->location);
add_stmt (stmt);
return false;
}
#define OMP_TARGET_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEVICE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_MAP)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_IF)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEPEND)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NOWAIT)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FIRSTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEFAULTMAP)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_IS_DEVICE_PTR))
static bool
cp_parser_omp_target (cp_parser *parser, cp_token *pragma_tok,
enum pragma_context context, bool *if_p)
{
tree *pc = NULL, stmt;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
enum tree_code ccode = ERROR_MARK;
if (strcmp (p, "teams") == 0)
ccode = OMP_TEAMS;
else if (strcmp (p, "parallel") == 0)
ccode = OMP_PARALLEL;
else if (strcmp (p, "simd") == 0)
ccode = OMP_SIMD;
if (ccode != ERROR_MARK)
{
tree cclauses[C_OMP_CLAUSE_SPLIT_COUNT];
char p_name[sizeof ("#pragma omp target teams distribute "
"parallel for simd")];
cp_lexer_consume_token (parser->lexer);
strcpy (p_name, "#pragma omp target");
if (!flag_openmp)  
{
tree stmt;
switch (ccode)
{
case OMP_TEAMS:
stmt = cp_parser_omp_teams (parser, pragma_tok, p_name,
OMP_TARGET_CLAUSE_MASK,
cclauses, if_p);
break;
case OMP_PARALLEL:
stmt = cp_parser_omp_parallel (parser, pragma_tok, p_name,
OMP_TARGET_CLAUSE_MASK,
cclauses, if_p);
break;
case OMP_SIMD:
stmt = cp_parser_omp_simd (parser, pragma_tok, p_name,
OMP_TARGET_CLAUSE_MASK,
cclauses, if_p);
break;
default:
gcc_unreachable ();
}
return stmt != NULL_TREE;
}
keep_next_level (true);
tree sb = begin_omp_structured_block (), ret;
unsigned save = cp_parser_begin_omp_structured_block (parser);
switch (ccode)
{
case OMP_TEAMS:
ret = cp_parser_omp_teams (parser, pragma_tok, p_name,
OMP_TARGET_CLAUSE_MASK, cclauses,
if_p);
break;
case OMP_PARALLEL:
ret = cp_parser_omp_parallel (parser, pragma_tok, p_name,
OMP_TARGET_CLAUSE_MASK, cclauses,
if_p);
break;
case OMP_SIMD:
ret = cp_parser_omp_simd (parser, pragma_tok, p_name,
OMP_TARGET_CLAUSE_MASK, cclauses,
if_p);
break;
default:
gcc_unreachable ();
}
cp_parser_end_omp_structured_block (parser, save);
tree body = finish_omp_structured_block (sb);
if (ret == NULL_TREE)
return false;
if (ccode == OMP_TEAMS && !processing_template_decl)
{
tree c;
for (c = cclauses[C_OMP_CLAUSE_SPLIT_TEAMS];
c; c = OMP_CLAUSE_CHAIN (c))
if ((OMP_CLAUSE_CODE (c) == OMP_CLAUSE_NUM_TEAMS
|| OMP_CLAUSE_CODE (c) == OMP_CLAUSE_THREAD_LIMIT)
&& TREE_CODE (OMP_CLAUSE_OPERAND (c, 0)) != INTEGER_CST)
{
tree expr = OMP_CLAUSE_OPERAND (c, 0);
expr = force_target_expr (TREE_TYPE (expr), expr, tf_none);
if (expr == error_mark_node)
continue;
tree tmp = TARGET_EXPR_SLOT (expr);
add_stmt (expr);
OMP_CLAUSE_OPERAND (c, 0) = expr;
tree tc = build_omp_clause (OMP_CLAUSE_LOCATION (c),
OMP_CLAUSE_FIRSTPRIVATE);
OMP_CLAUSE_DECL (tc) = tmp;
OMP_CLAUSE_CHAIN (tc)
= cclauses[C_OMP_CLAUSE_SPLIT_TARGET];
cclauses[C_OMP_CLAUSE_SPLIT_TARGET] = tc;
}
}
tree stmt = make_node (OMP_TARGET);
TREE_TYPE (stmt) = void_type_node;
OMP_TARGET_CLAUSES (stmt) = cclauses[C_OMP_CLAUSE_SPLIT_TARGET];
OMP_TARGET_BODY (stmt) = body;
OMP_TARGET_COMBINED (stmt) = 1;
SET_EXPR_LOCATION (stmt, pragma_tok->location);
add_stmt (stmt);
pc = &OMP_TARGET_CLAUSES (stmt);
goto check_clauses;
}
else if (!flag_openmp)  
{
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return false;
}
else if (strcmp (p, "data") == 0)
{
cp_lexer_consume_token (parser->lexer);
cp_parser_omp_target_data (parser, pragma_tok, if_p);
return true;
}
else if (strcmp (p, "enter") == 0)
{
cp_lexer_consume_token (parser->lexer);
cp_parser_omp_target_enter_data (parser, pragma_tok, context);
return false;
}
else if (strcmp (p, "exit") == 0)
{
cp_lexer_consume_token (parser->lexer);
cp_parser_omp_target_exit_data (parser, pragma_tok, context);
return false;
}
else if (strcmp (p, "update") == 0)
{
cp_lexer_consume_token (parser->lexer);
return cp_parser_omp_target_update (parser, pragma_tok, context);
}
}
if (!flag_openmp)  
{
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return false;
}
stmt = make_node (OMP_TARGET);
TREE_TYPE (stmt) = void_type_node;
OMP_TARGET_CLAUSES (stmt)
= cp_parser_omp_all_clauses (parser, OMP_TARGET_CLAUSE_MASK,
"#pragma omp target", pragma_tok);
pc = &OMP_TARGET_CLAUSES (stmt);
keep_next_level (true);
OMP_TARGET_BODY (stmt) = cp_parser_omp_structured_block (parser, if_p);
SET_EXPR_LOCATION (stmt, pragma_tok->location);
add_stmt (stmt);
check_clauses:
while (*pc)
{
if (OMP_CLAUSE_CODE (*pc) == OMP_CLAUSE_MAP)
switch (OMP_CLAUSE_MAP_KIND (*pc))
{
case GOMP_MAP_TO:
case GOMP_MAP_ALWAYS_TO:
case GOMP_MAP_FROM:
case GOMP_MAP_ALWAYS_FROM:
case GOMP_MAP_TOFROM:
case GOMP_MAP_ALWAYS_TOFROM:
case GOMP_MAP_ALLOC:
case GOMP_MAP_FIRSTPRIVATE_POINTER:
case GOMP_MAP_FIRSTPRIVATE_REFERENCE:
case GOMP_MAP_ALWAYS_POINTER:
break;
default:
error_at (OMP_CLAUSE_LOCATION (*pc),
"%<#pragma omp target%> with map-type other "
"than %<to%>, %<from%>, %<tofrom%> or %<alloc%> "
"on %<map%> clause");
*pc = OMP_CLAUSE_CHAIN (*pc);
continue;
}
pc = &OMP_CLAUSE_CHAIN (*pc);
}
return true;
}
static tree
cp_parser_oacc_cache (cp_parser *parser, cp_token *pragma_tok)
{
tree stmt, clauses;
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE__CACHE_, NULL_TREE);
clauses = finish_omp_clauses (clauses, C_ORT_ACC);
cp_parser_require_pragma_eol (parser, cp_lexer_peek_token (parser->lexer));
stmt = make_node (OACC_CACHE);
TREE_TYPE (stmt) = void_type_node;
OACC_CACHE_CLAUSES (stmt) = clauses;
SET_EXPR_LOCATION (stmt, pragma_tok->location);
add_stmt (stmt);
return stmt;
}
#define OACC_DATA_CLAUSE_MASK						\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPY)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPYIN)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPYOUT)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_CREATE)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_DEVICEPTR)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_IF)			\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_COPY)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYIN)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYOUT)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_CREATE))
static tree
cp_parser_oacc_data (cp_parser *parser, cp_token *pragma_tok, bool *if_p)
{
tree stmt, clauses, block;
unsigned int save;
clauses = cp_parser_oacc_all_clauses (parser, OACC_DATA_CLAUSE_MASK,
"#pragma acc data", pragma_tok);
block = begin_omp_parallel ();
save = cp_parser_begin_omp_structured_block (parser);
cp_parser_statement (parser, NULL_TREE, false, if_p);
cp_parser_end_omp_structured_block (parser, save);
stmt = finish_oacc_data (clauses, block);
return stmt;
}
#define OACC_HOST_DATA_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_USE_DEVICE) )
static tree
cp_parser_oacc_host_data (cp_parser *parser, cp_token *pragma_tok, bool *if_p)
{
tree stmt, clauses, block;
unsigned int save;
clauses = cp_parser_oacc_all_clauses (parser, OACC_HOST_DATA_CLAUSE_MASK,
"#pragma acc host_data", pragma_tok);
block = begin_omp_parallel ();
save = cp_parser_begin_omp_structured_block (parser);
cp_parser_statement (parser, NULL_TREE, false, if_p);
cp_parser_end_omp_structured_block (parser, save);
stmt = finish_oacc_host_data (clauses, block);
return stmt;
}
#define OACC_DECLARE_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPY)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPYIN)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPYOUT)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_CREATE)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_DEVICEPTR)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_DEVICE_RESIDENT)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_LINK)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_COPY)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYIN)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYOUT)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_CREATE))
static tree
cp_parser_oacc_declare (cp_parser *parser, cp_token *pragma_tok)
{
tree clauses, stmt;
bool error = false;
clauses = cp_parser_oacc_all_clauses (parser, OACC_DECLARE_CLAUSE_MASK,
"#pragma acc declare", pragma_tok, true);
if (omp_find_clause (clauses, OMP_CLAUSE_MAP) == NULL_TREE)
{
error_at (pragma_tok->location,
"no valid clauses specified in %<#pragma acc declare%>");
return NULL_TREE;
}
for (tree t = clauses; t; t = OMP_CLAUSE_CHAIN (t))
{
location_t loc = OMP_CLAUSE_LOCATION (t);
tree decl = OMP_CLAUSE_DECL (t);
if (!DECL_P (decl))
{
error_at (loc, "array section in %<#pragma acc declare%>");
error = true;
continue;
}
gcc_assert (OMP_CLAUSE_CODE (t) == OMP_CLAUSE_MAP);
switch (OMP_CLAUSE_MAP_KIND (t))
{
case GOMP_MAP_FIRSTPRIVATE_POINTER:
case GOMP_MAP_FORCE_ALLOC:
case GOMP_MAP_FORCE_TO:
case GOMP_MAP_FORCE_DEVICEPTR:
case GOMP_MAP_DEVICE_RESIDENT:
break;
case GOMP_MAP_LINK:
if (!global_bindings_p ()
&& (TREE_STATIC (decl)
|| !DECL_EXTERNAL (decl)))
{
error_at (loc,
"%qD must be a global variable in "
"%<#pragma acc declare link%>",
decl);
error = true;
continue;
}
break;
default:
if (global_bindings_p ())
{
error_at (loc, "invalid OpenACC clause at file scope");
error = true;
continue;
}
if (DECL_EXTERNAL (decl))
{
error_at (loc,
"invalid use of %<extern%> variable %qD "
"in %<#pragma acc declare%>", decl);
error = true;
continue;
}
else if (TREE_PUBLIC (decl))
{
error_at (loc,
"invalid use of %<global%> variable %qD "
"in %<#pragma acc declare%>", decl);
error = true;
continue;
}
break;
}
if (lookup_attribute ("omp declare target", DECL_ATTRIBUTES (decl))
|| lookup_attribute ("omp declare target link",
DECL_ATTRIBUTES (decl)))
{
error_at (loc, "variable %qD used more than once with "
"%<#pragma acc declare%>", decl);
error = true;
continue;
}
if (!error)
{
tree id;
if (OMP_CLAUSE_MAP_KIND (t) == GOMP_MAP_LINK)
id = get_identifier ("omp declare target link");
else
id = get_identifier ("omp declare target");
DECL_ATTRIBUTES (decl)
= tree_cons (id, NULL_TREE, DECL_ATTRIBUTES (decl));
if (global_bindings_p ())
{
symtab_node *node = symtab_node::get (decl);
if (node != NULL)
{
node->offloadable = 1;
if (ENABLE_OFFLOADING)
{
g->have_offload = true;
if (is_a <varpool_node *> (node))
vec_safe_push (offload_vars, decl);
}
}
}
}
}
if (error || global_bindings_p ())
return NULL_TREE;
stmt = make_node (OACC_DECLARE);
TREE_TYPE (stmt) = void_type_node;
OACC_DECLARE_CLAUSES (stmt) = clauses;
SET_EXPR_LOCATION (stmt, pragma_tok->location);
add_stmt (stmt);
return NULL_TREE;
}
#define OACC_ENTER_DATA_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_IF)			\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_ASYNC)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPYIN)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_CREATE)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYIN)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_CREATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_WAIT) )
#define OACC_EXIT_DATA_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_IF)			\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_ASYNC)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPYOUT)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_DELETE) 		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_WAIT) )
static tree
cp_parser_oacc_enter_exit_data (cp_parser *parser, cp_token *pragma_tok,
bool enter)
{
location_t loc = pragma_tok->location;
tree stmt, clauses;
const char *p = "";
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
p = IDENTIFIER_POINTER (cp_lexer_peek_token (parser->lexer)->u.value);
if (strcmp (p, "data") != 0)
{
error_at (loc, "expected %<data%> after %<#pragma acc %s%>",
enter ? "enter" : "exit");
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return NULL_TREE;
}
cp_lexer_consume_token (parser->lexer);
if (enter)
clauses = cp_parser_oacc_all_clauses (parser, OACC_ENTER_DATA_CLAUSE_MASK,
"#pragma acc enter data", pragma_tok);
else
clauses = cp_parser_oacc_all_clauses (parser, OACC_EXIT_DATA_CLAUSE_MASK,
"#pragma acc exit data", pragma_tok);
if (omp_find_clause (clauses, OMP_CLAUSE_MAP) == NULL_TREE)
{
error_at (loc, "%<#pragma acc %s data%> has no data movement clause",
enter ? "enter" : "exit");
return NULL_TREE;
}
stmt = enter ? make_node (OACC_ENTER_DATA) : make_node (OACC_EXIT_DATA);
TREE_TYPE (stmt) = void_type_node;
OMP_STANDALONE_CLAUSES (stmt) = clauses;
SET_EXPR_LOCATION (stmt, pragma_tok->location);
add_stmt (stmt);
return stmt;
}
#define OACC_LOOP_CLAUSE_MASK						\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COLLAPSE)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRIVATE)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_REDUCTION)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_GANG)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_VECTOR)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_WORKER)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_AUTO)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_INDEPENDENT)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_SEQ)			\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_TILE))
static tree
cp_parser_oacc_loop (cp_parser *parser, cp_token *pragma_tok, char *p_name,
omp_clause_mask mask, tree *cclauses, bool *if_p)
{
bool is_parallel = ((mask >> PRAGMA_OACC_CLAUSE_REDUCTION) & 1) == 1;
strcat (p_name, " loop");
mask |= OACC_LOOP_CLAUSE_MASK;
tree clauses = cp_parser_oacc_all_clauses (parser, mask, p_name, pragma_tok,
cclauses == NULL);
if (cclauses)
{
clauses = c_oacc_split_loop_clauses (clauses, cclauses, is_parallel);
if (*cclauses)
*cclauses = finish_omp_clauses (*cclauses, C_ORT_ACC);
if (clauses)
clauses = finish_omp_clauses (clauses, C_ORT_ACC);
}
tree block = begin_omp_structured_block ();
int save = cp_parser_begin_omp_structured_block (parser);
tree stmt = cp_parser_omp_for_loop (parser, OACC_LOOP, clauses, NULL, if_p);
cp_parser_end_omp_structured_block (parser, save);
add_stmt (finish_omp_structured_block (block));
return stmt;
}
#define OACC_KERNELS_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_ASYNC)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPY)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPYIN)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPYOUT)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_CREATE)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_DEFAULT)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_DEVICEPTR)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_IF)			\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_NUM_GANGS)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_NUM_WORKERS)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_COPY)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYIN)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYOUT)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_CREATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_VECTOR_LENGTH)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_WAIT) )
#define OACC_PARALLEL_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_ASYNC)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPY)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPYIN)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_COPYOUT)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_CREATE)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_DEFAULT)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_DEVICEPTR)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_FIRSTPRIVATE)       	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_IF)			\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_NUM_GANGS)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_NUM_WORKERS)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_COPY)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYIN)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_COPYOUT)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRESENT_OR_CREATE)   \
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_PRIVATE)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_REDUCTION)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_VECTOR_LENGTH)       \
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_WAIT) )
static tree
cp_parser_oacc_kernels_parallel (cp_parser *parser, cp_token *pragma_tok,
char *p_name, bool *if_p)
{
omp_clause_mask mask;
enum tree_code code;
switch (cp_parser_pragma_kind (pragma_tok))
{
case PRAGMA_OACC_KERNELS:
strcat (p_name, " kernels");
mask = OACC_KERNELS_CLAUSE_MASK;
code = OACC_KERNELS;
break;
case PRAGMA_OACC_PARALLEL:
strcat (p_name, " parallel");
mask = OACC_PARALLEL_CLAUSE_MASK;
code = OACC_PARALLEL;
break;
default:
gcc_unreachable ();
}
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
const char *p
= IDENTIFIER_POINTER (cp_lexer_peek_token (parser->lexer)->u.value);
if (strcmp (p, "loop") == 0)
{
cp_lexer_consume_token (parser->lexer);
tree block = begin_omp_parallel ();
tree clauses;
cp_parser_oacc_loop (parser, pragma_tok, p_name, mask, &clauses,
if_p);
return finish_omp_construct (code, block, clauses);
}
}
tree clauses = cp_parser_oacc_all_clauses (parser, mask, p_name, pragma_tok);
tree block = begin_omp_parallel ();
unsigned int save = cp_parser_begin_omp_structured_block (parser);
cp_parser_statement (parser, NULL_TREE, false, if_p);
cp_parser_end_omp_structured_block (parser, save);
return finish_omp_construct (code, block, clauses);
}
#define OACC_UPDATE_CLAUSE_MASK						\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_ASYNC)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_DEVICE)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_HOST)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_IF)			\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_SELF)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_WAIT))
static tree
cp_parser_oacc_update (cp_parser *parser, cp_token *pragma_tok)
{
tree stmt, clauses;
clauses = cp_parser_oacc_all_clauses (parser, OACC_UPDATE_CLAUSE_MASK,
"#pragma acc update", pragma_tok);
if (omp_find_clause (clauses, OMP_CLAUSE_MAP) == NULL_TREE)
{
error_at (pragma_tok->location,
"%<#pragma acc update%> must contain at least one "
"%<device%> or %<host%> or %<self%> clause");
return NULL_TREE;
}
stmt = make_node (OACC_UPDATE);
TREE_TYPE (stmt) = void_type_node;
OACC_UPDATE_CLAUSES (stmt) = clauses;
SET_EXPR_LOCATION (stmt, pragma_tok->location);
add_stmt (stmt);
return stmt;
}
#define OACC_WAIT_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_ASYNC))
static tree
cp_parser_oacc_wait (cp_parser *parser, cp_token *pragma_tok)
{
tree clauses, list = NULL_TREE, stmt = NULL_TREE;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
if (cp_lexer_peek_token (parser->lexer)->type == CPP_OPEN_PAREN)
list = cp_parser_oacc_wait_list (parser, loc, list);
clauses = cp_parser_oacc_all_clauses (parser, OACC_WAIT_CLAUSE_MASK,
"#pragma acc wait", pragma_tok);
stmt = c_finish_oacc_wait (loc, list, clauses);
stmt = finish_expr_stmt (stmt);
return stmt;
}
#define OMP_DECLARE_SIMD_CLAUSE_MASK				\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_SIMDLEN)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_LINEAR)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_ALIGNED)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_UNIFORM)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_INBRANCH)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NOTINBRANCH))
static void
cp_parser_omp_declare_simd (cp_parser *parser, cp_token *pragma_tok,
enum pragma_context context)
{
bool first_p = parser->omp_declare_simd == NULL;
cp_omp_declare_simd_data data;
if (first_p)
{
data.error_seen = false;
data.fndecl_seen = false;
data.tokens = vNULL;
data.clauses = NULL_TREE;
parser->omp_declare_simd = &data;
}
while (cp_lexer_next_token_is_not (parser->lexer, CPP_PRAGMA_EOL)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_EOF))
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_PRAGMA_EOL))
parser->omp_declare_simd->error_seen = true;
cp_parser_require_pragma_eol (parser, pragma_tok);
struct cp_token_cache *cp
= cp_token_cache_new (pragma_tok, cp_lexer_peek_token (parser->lexer));
parser->omp_declare_simd->tokens.safe_push (cp);
if (first_p)
{
while (cp_lexer_next_token_is (parser->lexer, CPP_PRAGMA))
cp_parser_pragma (parser, context, NULL);
switch (context)
{
case pragma_external:
cp_parser_declaration (parser);
break;
case pragma_member:
cp_parser_member_declaration (parser);
break;
case pragma_objc_icode:
cp_parser_block_declaration (parser, false);
break;
default:
cp_parser_declaration_statement (parser);
break;
}
if (parser->omp_declare_simd
&& !parser->omp_declare_simd->error_seen
&& !parser->omp_declare_simd->fndecl_seen)
error_at (pragma_tok->location,
"%<#pragma omp declare simd%> not immediately followed by "
"function declaration or definition");
data.tokens.release ();
parser->omp_declare_simd = NULL;
}
}
static tree
cp_parser_late_parsing_omp_declare_simd (cp_parser *parser, tree attrs)
{
struct cp_token_cache *ce;
cp_omp_declare_simd_data *data = parser->omp_declare_simd;
int i;
if (!data->error_seen && data->fndecl_seen)
{
error ("%<#pragma omp declare simd%> not immediately followed by "
"a single function declaration or definition");
data->error_seen = true;
}
if (data->error_seen)
return attrs;
FOR_EACH_VEC_ELT (data->tokens, i, ce)
{
tree c, cl;
cp_parser_push_lexer_for_tokens (parser, ce);
parser->lexer->in_pragma = true;
gcc_assert (cp_lexer_peek_token (parser->lexer)->type == CPP_PRAGMA);
cp_token *pragma_tok = cp_lexer_consume_token (parser->lexer);
cp_lexer_consume_token (parser->lexer);
cl = cp_parser_omp_all_clauses (parser, OMP_DECLARE_SIMD_CLAUSE_MASK,
"#pragma omp declare simd", pragma_tok);
cp_parser_pop_lexer (parser);
if (cl)
cl = tree_cons (NULL_TREE, cl, NULL_TREE);
c = build_tree_list (get_identifier ("omp declare simd"), cl);
TREE_CHAIN (c) = attrs;
if (processing_template_decl)
ATTR_IS_DEPENDENT (c) = 1;
attrs = c;
}
data->fndecl_seen = true;
return attrs;
}
#define OMP_DECLARE_TARGET_CLAUSE_MASK				\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_TO)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_LINK))
static void
cp_parser_omp_declare_target (cp_parser *parser, cp_token *pragma_tok)
{
tree clauses = NULL_TREE;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
clauses
= cp_parser_omp_all_clauses (parser, OMP_DECLARE_TARGET_CLAUSE_MASK,
"#pragma omp declare target", pragma_tok);
else if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
clauses = cp_parser_omp_var_list (parser, OMP_CLAUSE_TO_DECLARE,
clauses);
clauses = finish_omp_clauses (clauses, C_ORT_OMP);
cp_parser_require_pragma_eol (parser, pragma_tok);
}
else
{
cp_parser_require_pragma_eol (parser, pragma_tok);
scope_chain->omp_declare_target_attribute++;
return;
}
if (scope_chain->omp_declare_target_attribute)
error_at (pragma_tok->location,
"%<#pragma omp declare target%> with clauses in between "
"%<#pragma omp declare target%> without clauses and "
"%<#pragma omp end declare target%>");
for (tree c = clauses; c; c = OMP_CLAUSE_CHAIN (c))
{
tree t = OMP_CLAUSE_DECL (c), id;
tree at1 = lookup_attribute ("omp declare target", DECL_ATTRIBUTES (t));
tree at2 = lookup_attribute ("omp declare target link",
DECL_ATTRIBUTES (t));
if (OMP_CLAUSE_CODE (c) == OMP_CLAUSE_LINK)
{
id = get_identifier ("omp declare target link");
std::swap (at1, at2);
}
else
id = get_identifier ("omp declare target");
if (at2)
{
error_at (OMP_CLAUSE_LOCATION (c),
"%qD specified both in declare target %<link%> and %<to%>"
" clauses", t);
continue;
}
if (!at1)
{
DECL_ATTRIBUTES (t) = tree_cons (id, NULL_TREE, DECL_ATTRIBUTES (t));
if (TREE_CODE (t) != FUNCTION_DECL && !is_global_var (t))
continue;
symtab_node *node = symtab_node::get (t);
if (node != NULL)
{
node->offloadable = 1;
if (ENABLE_OFFLOADING)
{
g->have_offload = true;
if (is_a <varpool_node *> (node))
vec_safe_push (offload_vars, t);
}
}
}
}
}
static void
cp_parser_omp_end_declare_target (cp_parser *parser, cp_token *pragma_tok)
{
const char *p = "";
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
p = IDENTIFIER_POINTER (id);
}
if (strcmp (p, "declare") == 0)
{
cp_lexer_consume_token (parser->lexer);
p = "";
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
p = IDENTIFIER_POINTER (id);
}
if (strcmp (p, "target") == 0)
cp_lexer_consume_token (parser->lexer);
else
{
cp_parser_error (parser, "expected %<target%>");
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return;
}
}
else
{
cp_parser_error (parser, "expected %<declare%>");
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return;
}
cp_parser_require_pragma_eol (parser, pragma_tok);
if (!scope_chain->omp_declare_target_attribute)
error_at (pragma_tok->location,
"%<#pragma omp end declare target%> without corresponding "
"%<#pragma omp declare target%>");
else
scope_chain->omp_declare_target_attribute--;
}
static bool
cp_parser_omp_declare_reduction_exprs (tree fndecl, cp_parser *parser)
{
tree type = TREE_VALUE (TYPE_ARG_TYPES (TREE_TYPE (fndecl)));
gcc_assert (TREE_CODE (type) == REFERENCE_TYPE);
type = TREE_TYPE (type);
tree omp_out = build_lang_decl (VAR_DECL, get_identifier ("omp_out"), type);
DECL_ARTIFICIAL (omp_out) = 1;
pushdecl (omp_out);
add_decl_expr (omp_out);
tree omp_in = build_lang_decl (VAR_DECL, get_identifier ("omp_in"), type);
DECL_ARTIFICIAL (omp_in) = 1;
pushdecl (omp_in);
add_decl_expr (omp_in);
tree combiner;
tree omp_priv = NULL_TREE, omp_orig = NULL_TREE, initializer = NULL_TREE;
keep_next_level (true);
tree block = begin_omp_structured_block ();
combiner = cp_parser_expression (parser);
finish_expr_stmt (combiner);
block = finish_omp_structured_block (block);
add_stmt (block);
if (!cp_parser_require (parser, CPP_CLOSE_PAREN, RT_CLOSE_PAREN))
return false;
const char *p = "";
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
p = IDENTIFIER_POINTER (id);
}
if (strcmp (p, "initializer") == 0)
{
cp_lexer_consume_token (parser->lexer);
matching_parens parens;
if (!parens.require_open (parser))
return false;
p = "";
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
p = IDENTIFIER_POINTER (id);
}
omp_priv = build_lang_decl (VAR_DECL, get_identifier ("omp_priv"), type);
DECL_ARTIFICIAL (omp_priv) = 1;
pushdecl (omp_priv);
add_decl_expr (omp_priv);
omp_orig = build_lang_decl (VAR_DECL, get_identifier ("omp_orig"), type);
DECL_ARTIFICIAL (omp_orig) = 1;
pushdecl (omp_orig);
add_decl_expr (omp_orig);
keep_next_level (true);
block = begin_omp_structured_block ();
bool ctor = false;
if (strcmp (p, "omp_priv") == 0)
{
bool is_direct_init, is_non_constant_init;
ctor = true;
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is (parser->lexer, CPP_CLOSE_PAREN)
|| (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN)
&& cp_lexer_peek_nth_token (parser->lexer, 2)->type
== CPP_CLOSE_PAREN
&& cp_lexer_peek_nth_token (parser->lexer, 3)->type
== CPP_CLOSE_PAREN))
{
finish_omp_structured_block (block);
error ("invalid initializer clause");
return false;
}
initializer = cp_parser_initializer (parser, &is_direct_init,
&is_non_constant_init);
cp_finish_decl (omp_priv, initializer, !is_non_constant_init,
NULL_TREE, LOOKUP_ONLYCONVERTING);
}
else
{
cp_parser_parse_tentatively (parser);
tree fn_name = cp_parser_id_expression (parser, false,
true,
NULL,
false,
false);
vec<tree, va_gc> *args;
if (fn_name == error_mark_node
|| cp_parser_error_occurred (parser)
|| !cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN)
|| ((args = cp_parser_parenthesized_expression_list
(parser, non_attr, false,
true,
NULL)),
cp_parser_error_occurred (parser)))
{
finish_omp_structured_block (block);
cp_parser_abort_tentative_parse (parser);
cp_parser_error (parser, "expected id-expression (arguments)");
return false;
}
unsigned int i;
tree arg;
FOR_EACH_VEC_SAFE_ELT (args, i, arg)
if (arg == omp_priv
|| (TREE_CODE (arg) == ADDR_EXPR
&& TREE_OPERAND (arg, 0) == omp_priv))
break;
cp_parser_abort_tentative_parse (parser);
if (arg == NULL_TREE)
error ("one of the initializer call arguments should be %<omp_priv%>"
" or %<&omp_priv%>");
initializer = cp_parser_postfix_expression (parser, false, false, false,
false, NULL);
finish_expr_stmt (initializer);
}
block = finish_omp_structured_block (block);
cp_walk_tree (&block, cp_remove_omp_priv_cleanup_stmt, omp_priv, NULL);
add_stmt (block);
if (ctor)
add_decl_expr (omp_orig);
if (!parens.require_close (parser))
return false;
}
if (!cp_lexer_next_token_is (parser->lexer, CPP_PRAGMA_EOL))
cp_parser_required_error (parser, RT_PRAGMA_EOL, false,
UNKNOWN_LOCATION);
return true;
}
static void
cp_parser_omp_declare_reduction (cp_parser *parser, cp_token *pragma_tok,
enum pragma_context)
{
auto_vec<tree> types;
enum tree_code reduc_code = ERROR_MARK;
tree reduc_id = NULL_TREE, orig_reduc_id = NULL_TREE, type;
unsigned int i;
cp_token *first_token;
cp_token_cache *cp;
int errs;
void *p;
p = obstack_alloc (&declarator_obstack, 0);
if (!cp_parser_require (parser, CPP_OPEN_PAREN, RT_OPEN_PAREN))
goto fail;
switch (cp_lexer_peek_token (parser->lexer)->type)
{
case CPP_PLUS:
reduc_code = PLUS_EXPR;
break;
case CPP_MULT:
reduc_code = MULT_EXPR;
break;
case CPP_MINUS:
reduc_code = MINUS_EXPR;
break;
case CPP_AND:
reduc_code = BIT_AND_EXPR;
break;
case CPP_XOR:
reduc_code = BIT_XOR_EXPR;
break;
case CPP_OR:
reduc_code = BIT_IOR_EXPR;
break;
case CPP_AND_AND:
reduc_code = TRUTH_ANDIF_EXPR;
break;
case CPP_OR_OR:
reduc_code = TRUTH_ORIF_EXPR;
break;
case CPP_NAME:
reduc_id = orig_reduc_id = cp_parser_identifier (parser);
break;
default:
cp_parser_error (parser, "expected %<+%>, %<*%>, %<-%>, %<&%>, %<^%>, "
"%<|%>, %<&&%>, %<||%> or identifier");
goto fail;
}
if (reduc_code != ERROR_MARK)
cp_lexer_consume_token (parser->lexer);
reduc_id = omp_reduction_id (reduc_code, reduc_id, NULL_TREE);
if (reduc_id == error_mark_node)
goto fail;
if (!cp_parser_require (parser, CPP_COLON, RT_COLON))
goto fail;
const char *saved_message;
saved_message = parser->type_definition_forbidden_message;
parser->type_definition_forbidden_message
= G_("types may not be defined in declare reduction type list");
bool saved_colon_corrects_to_scope_p;
saved_colon_corrects_to_scope_p = parser->colon_corrects_to_scope_p;
parser->colon_corrects_to_scope_p = false;
bool saved_colon_doesnt_start_class_def_p;
saved_colon_doesnt_start_class_def_p
= parser->colon_doesnt_start_class_def_p;
parser->colon_doesnt_start_class_def_p = true;
while (true)
{
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
type = cp_parser_type_id (parser);
if (type == error_mark_node)
;
else if (ARITHMETIC_TYPE_P (type)
&& (orig_reduc_id == NULL_TREE
|| (TREE_CODE (type) != COMPLEX_TYPE
&& (id_equal (orig_reduc_id, "min")
|| id_equal (orig_reduc_id, "max")))))
error_at (loc, "predeclared arithmetic type %qT in "
"%<#pragma omp declare reduction%>", type);
else if (TREE_CODE (type) == FUNCTION_TYPE
|| TREE_CODE (type) == METHOD_TYPE
|| TREE_CODE (type) == ARRAY_TYPE)
error_at (loc, "function or array type %qT in "
"%<#pragma omp declare reduction%>", type);
else if (TREE_CODE (type) == REFERENCE_TYPE)
error_at (loc, "reference type %qT in "
"%<#pragma omp declare reduction%>", type);
else if (TYPE_QUALS_NO_ADDR_SPACE (type))
error_at (loc, "const, volatile or __restrict qualified type %qT in "
"%<#pragma omp declare reduction%>", type);
else
types.safe_push (type);
if (cp_lexer_next_token_is (parser->lexer, CPP_COMMA))
cp_lexer_consume_token (parser->lexer);
else
break;
}
parser->type_definition_forbidden_message = saved_message;
parser->colon_corrects_to_scope_p = saved_colon_corrects_to_scope_p;
parser->colon_doesnt_start_class_def_p
= saved_colon_doesnt_start_class_def_p;
if (!cp_parser_require (parser, CPP_COLON, RT_COLON)
|| types.is_empty ())
{
fail:
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
goto done;
}
first_token = cp_lexer_peek_token (parser->lexer);
cp = NULL;
errs = errorcount;
FOR_EACH_VEC_ELT (types, i, type)
{
tree fntype
= build_function_type_list (void_type_node,
cp_build_reference_type (type, false),
NULL_TREE);
tree this_reduc_id = reduc_id;
if (!dependent_type_p (type))
this_reduc_id = omp_reduction_id (ERROR_MARK, reduc_id, type);
tree fndecl = build_lang_decl (FUNCTION_DECL, this_reduc_id, fntype);
DECL_SOURCE_LOCATION (fndecl) = pragma_tok->location;
DECL_ARTIFICIAL (fndecl) = 1;
DECL_EXTERNAL (fndecl) = 1;
DECL_DECLARED_INLINE_P (fndecl) = 1;
DECL_IGNORED_P (fndecl) = 1;
DECL_OMP_DECLARE_REDUCTION_P (fndecl) = 1;
SET_DECL_ASSEMBLER_NAME (fndecl, get_identifier ("<udr>"));
DECL_ATTRIBUTES (fndecl)
= tree_cons (get_identifier ("gnu_inline"), NULL_TREE,
DECL_ATTRIBUTES (fndecl));
if (processing_template_decl)
fndecl = push_template_decl (fndecl);
bool block_scope = false;
tree block = NULL_TREE;
if (current_function_decl)
{
block_scope = true;
DECL_CONTEXT (fndecl) = global_namespace;
if (!processing_template_decl)
pushdecl (fndecl);
}
else if (current_class_type)
{
if (cp == NULL)
{
while (cp_lexer_next_token_is_not (parser->lexer, CPP_PRAGMA_EOL)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_EOF))
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_PRAGMA_EOL))
goto fail;
cp = cp_token_cache_new (first_token,
cp_lexer_peek_nth_token (parser->lexer,
2));
}
DECL_STATIC_FUNCTION_P (fndecl) = 1;
finish_member_declaration (fndecl);
DECL_PENDING_INLINE_INFO (fndecl) = cp;
DECL_PENDING_INLINE_P (fndecl) = 1;
vec_safe_push (unparsed_funs_with_definitions, fndecl);
continue;
}
else
{
DECL_CONTEXT (fndecl) = current_namespace;
pushdecl (fndecl);
}
if (!block_scope)
start_preparsed_function (fndecl, NULL_TREE, SF_PRE_PARSED);
else
block = begin_omp_structured_block ();
if (cp)
{
cp_parser_push_lexer_for_tokens (parser, cp);
parser->lexer->in_pragma = true;
}
if (!cp_parser_omp_declare_reduction_exprs (fndecl, parser))
{
if (!block_scope)
finish_function (false);
else
DECL_CONTEXT (fndecl) = current_function_decl;
if (cp)
cp_parser_pop_lexer (parser);
goto fail;
}
if (cp)
cp_parser_pop_lexer (parser);
if (!block_scope)
finish_function (false);
else
{
DECL_CONTEXT (fndecl) = current_function_decl;
block = finish_omp_structured_block (block);
if (TREE_CODE (block) == BIND_EXPR)
DECL_SAVED_TREE (fndecl) = BIND_EXPR_BODY (block);
else if (TREE_CODE (block) == STATEMENT_LIST)
DECL_SAVED_TREE (fndecl) = block;
if (processing_template_decl)
add_decl_expr (fndecl);
}
cp_check_omp_declare_reduction (fndecl);
if (cp == NULL && types.length () > 1)
cp = cp_token_cache_new (first_token,
cp_lexer_peek_nth_token (parser->lexer, 2));
if (errs != errorcount)
break;
}
cp_parser_require_pragma_eol (parser, pragma_tok);
done:
obstack_free (&declarator_obstack, p);
}
static bool
cp_parser_omp_declare (cp_parser *parser, cp_token *pragma_tok,
enum pragma_context context)
{
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp (p, "simd") == 0)
{
cp_lexer_consume_token (parser->lexer);
cp_parser_omp_declare_simd (parser, pragma_tok,
context);
return true;
}
cp_ensure_no_omp_declare_simd (parser);
if (strcmp (p, "reduction") == 0)
{
cp_lexer_consume_token (parser->lexer);
cp_parser_omp_declare_reduction (parser, pragma_tok,
context);
return false;
}
if (!flag_openmp)  
{
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return false;
}
if (strcmp (p, "target") == 0)
{
cp_lexer_consume_token (parser->lexer);
cp_parser_omp_declare_target (parser, pragma_tok);
return false;
}
}
cp_parser_error (parser, "expected %<simd%> or %<reduction%> "
"or %<target%>");
cp_parser_require_pragma_eol (parser, pragma_tok);
return false;
}
#define OMP_TASKLOOP_CLAUSE_MASK				\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_SHARED)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FIRSTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_LASTPRIVATE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_DEFAULT)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_GRAINSIZE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NUM_TASKS)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_COLLAPSE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_UNTIED)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_IF)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_FINAL)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_MERGEABLE)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_NOGROUP)	\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OMP_CLAUSE_PRIORITY))
static tree
cp_parser_omp_taskloop (cp_parser *parser, cp_token *pragma_tok,
char *p_name, omp_clause_mask mask, tree *cclauses,
bool *if_p)
{
tree clauses, sb, ret;
unsigned int save;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
strcat (p_name, " taskloop");
mask |= OMP_TASKLOOP_CLAUSE_MASK;
if (cp_lexer_next_token_is (parser->lexer, CPP_NAME))
{
tree id = cp_lexer_peek_token (parser->lexer)->u.value;
const char *p = IDENTIFIER_POINTER (id);
if (strcmp (p, "simd") == 0)
{
tree cclauses_buf[C_OMP_CLAUSE_SPLIT_COUNT];
if (cclauses == NULL)
cclauses = cclauses_buf;
cp_lexer_consume_token (parser->lexer);
if (!flag_openmp)  
return cp_parser_omp_simd (parser, pragma_tok, p_name, mask,
cclauses, if_p);
sb = begin_omp_structured_block ();
save = cp_parser_begin_omp_structured_block (parser);
ret = cp_parser_omp_simd (parser, pragma_tok, p_name, mask,
cclauses, if_p);
cp_parser_end_omp_structured_block (parser, save);
tree body = finish_omp_structured_block (sb);
if (ret == NULL)
return ret;
ret = make_node (OMP_TASKLOOP);
TREE_TYPE (ret) = void_type_node;
OMP_FOR_BODY (ret) = body;
OMP_FOR_CLAUSES (ret) = cclauses[C_OMP_CLAUSE_SPLIT_TASKLOOP];
SET_EXPR_LOCATION (ret, loc);
add_stmt (ret);
return ret;
}
}
if (!flag_openmp)  
{
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return NULL_TREE;
}
clauses = cp_parser_omp_all_clauses (parser, mask, p_name, pragma_tok,
cclauses == NULL);
if (cclauses)
{
cp_omp_split_clauses (loc, OMP_TASKLOOP, mask, clauses, cclauses);
clauses = cclauses[C_OMP_CLAUSE_SPLIT_TASKLOOP];
}
sb = begin_omp_structured_block ();
save = cp_parser_begin_omp_structured_block (parser);
ret = cp_parser_omp_for_loop (parser, OMP_TASKLOOP, clauses, cclauses,
if_p);
cp_parser_end_omp_structured_block (parser, save);
add_stmt (finish_omp_structured_block (sb));
return ret;
}
#define OACC_ROUTINE_CLAUSE_MASK					\
( (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_GANG)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_WORKER)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_VECTOR)		\
| (OMP_CLAUSE_MASK_1 << PRAGMA_OACC_CLAUSE_SEQ))
static void
cp_parser_oacc_routine (cp_parser *parser, cp_token *pragma_tok,
enum pragma_context context)
{
gcc_checking_assert (context == pragma_external);
gcc_checking_assert (parser->oacc_routine == NULL);
cp_oacc_routine_data data;
data.error_seen = false;
data.fndecl_seen = false;
data.tokens = vNULL;
data.clauses = NULL_TREE;
data.loc = pragma_tok->location;
parser->oacc_routine = &data;
if (cp_lexer_next_token_is (parser->lexer, CPP_OPEN_PAREN))
{
matching_parens parens;
parens.consume_open (parser); 
location_t name_loc = cp_lexer_peek_token (parser->lexer)->location;
tree name = cp_parser_id_expression (parser,
false,
false,
NULL,
false,
false);
tree decl = (identifier_p (name)
? cp_parser_lookup_name_simple (parser, name, name_loc)
: name);
if (name != error_mark_node && decl == error_mark_node)
cp_parser_name_lookup_error (parser, name, decl, NLE_NULL, name_loc);
if (decl == error_mark_node
|| !parens.require_close (parser))
{
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
parser->oacc_routine = NULL;
return;
}
data.clauses
= cp_parser_oacc_all_clauses (parser, OACC_ROUTINE_CLAUSE_MASK,
"#pragma acc routine",
cp_lexer_peek_token (parser->lexer));
if (decl && is_overloaded_fn (decl)
&& (TREE_CODE (decl) != FUNCTION_DECL
|| DECL_FUNCTION_TEMPLATE_P  (decl)))
{
error_at (name_loc,
"%<#pragma acc routine%> names a set of overloads");
parser->oacc_routine = NULL;
return;
}
if (!DECL_NAMESPACE_SCOPE_P (decl))
{
error_at (name_loc,
"%qD does not refer to a namespace scope function", decl);
parser->oacc_routine = NULL;
return;
}
if (TREE_CODE (decl) != FUNCTION_DECL)
{
error_at (name_loc, "%qD does not refer to a function", decl);
parser->oacc_routine = NULL;
return;
}
cp_finalize_oacc_routine (parser, decl, false);
parser->oacc_routine = NULL;
}
else 
{
while (cp_lexer_next_token_is_not (parser->lexer, CPP_PRAGMA_EOL)
&& cp_lexer_next_token_is_not (parser->lexer, CPP_EOF))
cp_lexer_consume_token (parser->lexer);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_PRAGMA_EOL))
parser->oacc_routine->error_seen = true;
cp_parser_require_pragma_eol (parser, pragma_tok);
struct cp_token_cache *cp
= cp_token_cache_new (pragma_tok, cp_lexer_peek_token (parser->lexer));
parser->oacc_routine->tokens.safe_push (cp);
if (cp_lexer_next_token_is (parser->lexer, CPP_PRAGMA))
{
cp_ensure_no_oacc_routine (parser);
data.tokens.release ();
return;
}
cp_parser_declaration (parser);
if (parser->oacc_routine
&& !parser->oacc_routine->fndecl_seen)
cp_ensure_no_oacc_routine (parser);
else
parser->oacc_routine = NULL;
data.tokens.release ();
}
}
static tree
cp_parser_late_parsing_oacc_routine (cp_parser *parser, tree attrs)
{
struct cp_token_cache *ce;
cp_oacc_routine_data *data = parser->oacc_routine;
if (!data->error_seen && data->fndecl_seen)
{
error_at (data->loc,
"%<#pragma acc routine%> not immediately followed by "
"a single function declaration or definition");
data->error_seen = true;
}
if (data->error_seen)
return attrs;
gcc_checking_assert (data->tokens.length () == 1);
ce = data->tokens[0];
cp_parser_push_lexer_for_tokens (parser, ce);
parser->lexer->in_pragma = true;
gcc_assert (cp_lexer_peek_token (parser->lexer)->type == CPP_PRAGMA);
cp_token *pragma_tok = cp_lexer_consume_token (parser->lexer);
gcc_checking_assert (parser->oacc_routine->clauses == NULL_TREE);
parser->oacc_routine->clauses
= cp_parser_oacc_all_clauses (parser, OACC_ROUTINE_CLAUSE_MASK,
"#pragma acc routine", pragma_tok);
cp_parser_pop_lexer (parser);
return attrs;
}
static void
cp_finalize_oacc_routine (cp_parser *parser, tree fndecl, bool is_defn)
{
if (__builtin_expect (parser->oacc_routine != NULL, 0))
{
if (parser->oacc_routine->error_seen
|| fndecl == error_mark_node)
return;
if (parser->oacc_routine->fndecl_seen)
{
error_at (parser->oacc_routine->loc,
"%<#pragma acc routine%> not immediately followed by"
" a single function declaration or definition");
parser->oacc_routine = NULL;
return;
}
if (TREE_CODE (fndecl) != FUNCTION_DECL)
{
cp_ensure_no_oacc_routine (parser);
return;
}
if (oacc_get_fn_attrib (fndecl))
{
error_at (parser->oacc_routine->loc,
"%<#pragma acc routine%> already applied to %qD", fndecl);
parser->oacc_routine = NULL;
return;
}
if (TREE_USED (fndecl) || (!is_defn && DECL_SAVED_TREE (fndecl)))
{
error_at (parser->oacc_routine->loc,
TREE_USED (fndecl)
? G_("%<#pragma acc routine%> must be applied before use")
: G_("%<#pragma acc routine%> must be applied before "
"definition"));
parser->oacc_routine = NULL;
return;
}
tree dims = oacc_build_routine_dims (parser->oacc_routine->clauses);
oacc_replace_fn_attrib (fndecl, dims);
DECL_ATTRIBUTES (fndecl)
= tree_cons (get_identifier ("omp declare target"),
NULL_TREE, DECL_ATTRIBUTES (fndecl));
parser->oacc_routine->fndecl_seen = true;
}
}
static void
cp_parser_omp_construct (cp_parser *parser, cp_token *pragma_tok, bool *if_p)
{
tree stmt;
char p_name[sizeof "#pragma omp teams distribute parallel for simd"];
omp_clause_mask mask (0);
switch (cp_parser_pragma_kind (pragma_tok))
{
case PRAGMA_OACC_ATOMIC:
cp_parser_omp_atomic (parser, pragma_tok);
return;
case PRAGMA_OACC_CACHE:
stmt = cp_parser_oacc_cache (parser, pragma_tok);
break;
case PRAGMA_OACC_DATA:
stmt = cp_parser_oacc_data (parser, pragma_tok, if_p);
break;
case PRAGMA_OACC_ENTER_DATA:
stmt = cp_parser_oacc_enter_exit_data (parser, pragma_tok, true);
break;
case PRAGMA_OACC_EXIT_DATA:
stmt = cp_parser_oacc_enter_exit_data (parser, pragma_tok, false);
break;
case PRAGMA_OACC_HOST_DATA:
stmt = cp_parser_oacc_host_data (parser, pragma_tok, if_p);
break;
case PRAGMA_OACC_KERNELS:
case PRAGMA_OACC_PARALLEL:
strcpy (p_name, "#pragma acc");
stmt = cp_parser_oacc_kernels_parallel (parser, pragma_tok, p_name,
if_p);
break;
case PRAGMA_OACC_LOOP:
strcpy (p_name, "#pragma acc");
stmt = cp_parser_oacc_loop (parser, pragma_tok, p_name, mask, NULL,
if_p);
break;
case PRAGMA_OACC_UPDATE:
stmt = cp_parser_oacc_update (parser, pragma_tok);
break;
case PRAGMA_OACC_WAIT:
stmt = cp_parser_oacc_wait (parser, pragma_tok);
break;
case PRAGMA_OMP_ATOMIC:
cp_parser_omp_atomic (parser, pragma_tok);
return;
case PRAGMA_OMP_CRITICAL:
stmt = cp_parser_omp_critical (parser, pragma_tok, if_p);
break;
case PRAGMA_OMP_DISTRIBUTE:
strcpy (p_name, "#pragma omp");
stmt = cp_parser_omp_distribute (parser, pragma_tok, p_name, mask, NULL,
if_p);
break;
case PRAGMA_OMP_FOR:
strcpy (p_name, "#pragma omp");
stmt = cp_parser_omp_for (parser, pragma_tok, p_name, mask, NULL,
if_p);
break;
case PRAGMA_OMP_MASTER:
stmt = cp_parser_omp_master (parser, pragma_tok, if_p);
break;
case PRAGMA_OMP_PARALLEL:
strcpy (p_name, "#pragma omp");
stmt = cp_parser_omp_parallel (parser, pragma_tok, p_name, mask, NULL,
if_p);
break;
case PRAGMA_OMP_SECTIONS:
strcpy (p_name, "#pragma omp");
stmt = cp_parser_omp_sections (parser, pragma_tok, p_name, mask, NULL);
break;
case PRAGMA_OMP_SIMD:
strcpy (p_name, "#pragma omp");
stmt = cp_parser_omp_simd (parser, pragma_tok, p_name, mask, NULL,
if_p);
break;
case PRAGMA_OMP_SINGLE:
stmt = cp_parser_omp_single (parser, pragma_tok, if_p);
break;
case PRAGMA_OMP_TASK:
stmt = cp_parser_omp_task (parser, pragma_tok, if_p);
break;
case PRAGMA_OMP_TASKGROUP:
stmt = cp_parser_omp_taskgroup (parser, pragma_tok, if_p);
break;
case PRAGMA_OMP_TASKLOOP:
strcpy (p_name, "#pragma omp");
stmt = cp_parser_omp_taskloop (parser, pragma_tok, p_name, mask, NULL,
if_p);
break;
case PRAGMA_OMP_TEAMS:
strcpy (p_name, "#pragma omp");
stmt = cp_parser_omp_teams (parser, pragma_tok, p_name, mask, NULL,
if_p);
break;
default:
gcc_unreachable ();
}
protected_set_expr_location (stmt, pragma_tok->location);
}

static tree
cp_parser_txn_attribute_opt (cp_parser *parser)
{
cp_token *token;
tree attr_name, attr = NULL;
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_ATTRIBUTE))
return cp_parser_attributes_opt (parser);
if (cp_lexer_next_token_is_not (parser->lexer, CPP_OPEN_SQUARE))
return NULL_TREE;
cp_lexer_consume_token (parser->lexer);
if (!cp_parser_require (parser, CPP_OPEN_SQUARE, RT_OPEN_SQUARE))
goto error1;
token = cp_lexer_peek_token (parser->lexer);
if (token->type == CPP_NAME || token->type == CPP_KEYWORD)
{
token = cp_lexer_consume_token (parser->lexer);
attr_name = (token->type == CPP_KEYWORD
? ridpointers[(int) token->keyword]
: token->u.value);
attr = build_tree_list (attr_name, NULL_TREE);
}
else
cp_parser_error (parser, "expected identifier");
cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE);
error1:
cp_parser_require (parser, CPP_CLOSE_SQUARE, RT_CLOSE_SQUARE);
return attr;
}
static tree
cp_parser_transaction (cp_parser *parser, cp_token *token)
{
unsigned char old_in = parser->in_transaction;
unsigned char this_in = 1, new_in;
enum rid keyword = token->keyword;
tree stmt, attrs, noex;
cp_lexer_consume_token (parser->lexer);
if (keyword == RID_TRANSACTION_RELAXED
|| keyword == RID_SYNCHRONIZED)
this_in |= TM_STMT_ATTR_RELAXED;
else
{
attrs = cp_parser_txn_attribute_opt (parser);
if (attrs)
this_in |= parse_tm_stmt_attr (attrs, TM_STMT_ATTR_OUTER);
}
if (keyword == RID_ATOMIC_NOEXCEPT)
noex = boolean_true_node;
else if (keyword == RID_ATOMIC_CANCEL)
{
sorry ("atomic_cancel");
noex = NULL_TREE;
}
else
noex = cp_parser_noexcept_specification_opt (parser, true, NULL, true);
new_in = this_in | (old_in & TM_STMT_ATTR_OUTER);
stmt = begin_transaction_stmt (token->location, NULL, this_in);
parser->in_transaction = new_in;
cp_parser_compound_statement (parser, NULL, BCS_TRANSACTION, false);
parser->in_transaction = old_in;
finish_transaction_stmt (stmt, NULL, this_in, noex);
return stmt;
}
static tree
cp_parser_transaction_expression (cp_parser *parser, enum rid keyword)
{
unsigned char old_in = parser->in_transaction;
unsigned char this_in = 1;
cp_token *token;
tree expr, noex;
bool noex_expr;
location_t loc = cp_lexer_peek_token (parser->lexer)->location;
gcc_assert (keyword == RID_TRANSACTION_ATOMIC
|| keyword == RID_TRANSACTION_RELAXED);
if (!flag_tm)
error_at (loc,
keyword == RID_TRANSACTION_RELAXED
? G_("%<__transaction_relaxed%> without transactional memory "
"support enabled")
: G_("%<__transaction_atomic%> without transactional memory "
"support enabled"));
token = cp_parser_require_keyword (parser, keyword,
(keyword == RID_TRANSACTION_ATOMIC ? RT_TRANSACTION_ATOMIC
: RT_TRANSACTION_RELAXED));
gcc_assert (token != NULL);
if (keyword == RID_TRANSACTION_RELAXED)
this_in |= TM_STMT_ATTR_RELAXED;
parser->in_transaction = this_in;
noex = cp_parser_noexcept_specification_opt (parser, false, &noex_expr,
true);
if (!noex || !noex_expr
|| cp_lexer_peek_token (parser->lexer)->type == CPP_OPEN_PAREN)
{
matching_parens parens;
parens.require_open (parser);
expr = cp_parser_expression (parser);
expr = finish_parenthesized_expr (expr);
parens.require_close (parser);
}
else
{
expr = noex;
noex = boolean_true_node;
}
expr = build_transaction_expr (token->location, expr, this_in, noex);
parser->in_transaction = old_in;
if (cp_parser_non_integral_constant_expression (parser, NIC_TRANSACTION))
return error_mark_node;
return (flag_tm ? expr : error_mark_node);
}
static void
cp_parser_function_transaction (cp_parser *parser, enum rid keyword)
{
unsigned char old_in = parser->in_transaction;
unsigned char new_in = 1;
tree compound_stmt, stmt, attrs;
cp_token *token;
gcc_assert (keyword == RID_TRANSACTION_ATOMIC
|| keyword == RID_TRANSACTION_RELAXED);
token = cp_parser_require_keyword (parser, keyword,
(keyword == RID_TRANSACTION_ATOMIC ? RT_TRANSACTION_ATOMIC
: RT_TRANSACTION_RELAXED));
gcc_assert (token != NULL);
if (keyword == RID_TRANSACTION_RELAXED)
new_in |= TM_STMT_ATTR_RELAXED;
else
{
attrs = cp_parser_txn_attribute_opt (parser);
if (attrs)
new_in |= parse_tm_stmt_attr (attrs, TM_STMT_ATTR_OUTER);
}
stmt = begin_transaction_stmt (token->location, &compound_stmt, new_in);
parser->in_transaction = new_in;
if (cp_lexer_next_token_is_keyword (parser->lexer, RID_TRY))
cp_parser_function_try_block (parser);
else
cp_parser_ctor_initializer_opt_and_function_body
(parser, false);
parser->in_transaction = old_in;
finish_transaction_stmt (stmt, compound_stmt, new_in, NULL_TREE);
}
static tree
cp_parser_transaction_cancel (cp_parser *parser)
{
cp_token *token;
bool is_outer = false;
tree stmt, attrs;
token = cp_parser_require_keyword (parser, RID_TRANSACTION_CANCEL,
RT_TRANSACTION_CANCEL);
gcc_assert (token != NULL);
attrs = cp_parser_txn_attribute_opt (parser);
if (attrs)
is_outer = (parse_tm_stmt_attr (attrs, TM_STMT_ATTR_OUTER) != 0);
cp_parser_require (parser, CPP_SEMICOLON, RT_SEMICOLON);
if (!flag_tm)
{
error_at (token->location, "%<__transaction_cancel%> without "
"transactional memory support enabled");
return error_mark_node;
}
else if (parser->in_transaction & TM_STMT_ATTR_RELAXED)
{
error_at (token->location, "%<__transaction_cancel%> within a "
"%<__transaction_relaxed%>");
return error_mark_node;
}
else if (is_outer)
{
if ((parser->in_transaction & TM_STMT_ATTR_OUTER) == 0
&& !is_tm_may_cancel_outer (current_function_decl))
{
error_at (token->location, "outer %<__transaction_cancel%> not "
"within outer %<__transaction_atomic%>");
error_at (token->location,
"  or a %<transaction_may_cancel_outer%> function");
return error_mark_node;
}
}
else if (parser->in_transaction == 0)
{
error_at (token->location, "%<__transaction_cancel%> not within "
"%<__transaction_atomic%>");
return error_mark_node;
}
stmt = build_tm_abort_call (token->location, is_outer);
add_stmt (stmt);
return stmt;
}

static GTY (()) cp_parser *the_parser;

static void
cp_parser_initial_pragma (cp_token *first_token)
{
tree name = NULL;
cp_lexer_get_preprocessor_token (NULL, first_token);
if (cp_parser_pragma_kind (first_token) != PRAGMA_GCC_PCH_PREPROCESS)
return;
cp_lexer_get_preprocessor_token (NULL, first_token);
if (first_token->type == CPP_STRING)
{
name = first_token->u.value;
cp_lexer_get_preprocessor_token (NULL, first_token);
if (first_token->type != CPP_PRAGMA_EOL)
error_at (first_token->location,
"junk at end of %<#pragma GCC pch_preprocess%>");
}
else
error_at (first_token->location, "expected string literal");
while (first_token->type != CPP_PRAGMA_EOL && first_token->type != CPP_EOF)
cp_lexer_get_preprocessor_token (NULL, first_token);
if (name)
c_common_pch_pragma (parse_in, TREE_STRING_POINTER (name));
cp_lexer_get_preprocessor_token (NULL, first_token);
}
static bool
cp_parser_pragma_ivdep (cp_parser *parser, cp_token *pragma_tok)
{
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return true;
}
static unsigned short
cp_parser_pragma_unroll (cp_parser *parser, cp_token *pragma_tok)
{
location_t location = cp_lexer_peek_token (parser->lexer)->location;
tree expr = cp_parser_constant_expression (parser);
unsigned short unroll;
expr = maybe_constant_value (expr);
HOST_WIDE_INT lunroll = 0;
if (!INTEGRAL_TYPE_P (TREE_TYPE (expr))
|| TREE_CODE (expr) != INTEGER_CST
|| (lunroll = tree_to_shwi (expr)) < 0
|| lunroll >= USHRT_MAX)
{
error_at (location, "%<#pragma GCC unroll%> requires an"
" assignment-expression that evaluates to a non-negative"
" integral constant less than %u", USHRT_MAX);
unroll = 0;
}
else
{
unroll = (unsigned short)lunroll;
if (unroll == 0)
unroll = 1;
}
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return unroll;
}
static bool
cp_parser_pragma (cp_parser *parser, enum pragma_context context, bool *if_p)
{
cp_token *pragma_tok;
unsigned int id;
tree stmt;
bool ret;
pragma_tok = cp_lexer_consume_token (parser->lexer);
gcc_assert (pragma_tok->type == CPP_PRAGMA);
parser->lexer->in_pragma = true;
id = cp_parser_pragma_kind (pragma_tok);
if (id != PRAGMA_OMP_DECLARE && id != PRAGMA_OACC_ROUTINE)
cp_ensure_no_omp_declare_simd (parser);
switch (id)
{
case PRAGMA_GCC_PCH_PREPROCESS:
error_at (pragma_tok->location,
"%<#pragma GCC pch_preprocess%> must be first");
break;
case PRAGMA_OMP_BARRIER:
switch (context)
{
case pragma_compound:
cp_parser_omp_barrier (parser, pragma_tok);
return false;
case pragma_stmt:
error_at (pragma_tok->location, "%<#pragma %s%> may only be "
"used in compound statements", "omp barrier");
break;
default:
goto bad_stmt;
}
break;
case PRAGMA_OMP_FLUSH:
switch (context)
{
case pragma_compound:
cp_parser_omp_flush (parser, pragma_tok);
return false;
case pragma_stmt:
error_at (pragma_tok->location, "%<#pragma %s%> may only be "
"used in compound statements", "omp flush");
break;
default:
goto bad_stmt;
}
break;
case PRAGMA_OMP_TASKWAIT:
switch (context)
{
case pragma_compound:
cp_parser_omp_taskwait (parser, pragma_tok);
return false;
case pragma_stmt:
error_at (pragma_tok->location,
"%<#pragma %s%> may only be used in compound statements",
"omp taskwait");
break;
default:
goto bad_stmt;
}
break;
case PRAGMA_OMP_TASKYIELD:
switch (context)
{
case pragma_compound:
cp_parser_omp_taskyield (parser, pragma_tok);
return false;
case pragma_stmt:
error_at (pragma_tok->location,
"%<#pragma %s%> may only be used in compound statements",
"omp taskyield");
break;
default:
goto bad_stmt;
}
break;
case PRAGMA_OMP_CANCEL:
switch (context)
{
case pragma_compound:
cp_parser_omp_cancel (parser, pragma_tok);
return false;
case pragma_stmt:
error_at (pragma_tok->location,
"%<#pragma %s%> may only be used in compound statements",
"omp cancel");
break;
default:
goto bad_stmt;
}
break;
case PRAGMA_OMP_CANCELLATION_POINT:
cp_parser_omp_cancellation_point (parser, pragma_tok, context);
return false;
case PRAGMA_OMP_THREADPRIVATE:
cp_parser_omp_threadprivate (parser, pragma_tok);
return false;
case PRAGMA_OMP_DECLARE:
return cp_parser_omp_declare (parser, pragma_tok, context);
case PRAGMA_OACC_DECLARE:
cp_parser_oacc_declare (parser, pragma_tok);
return false;
case PRAGMA_OACC_ENTER_DATA:
if (context == pragma_stmt)
{
error_at (pragma_tok->location,
"%<#pragma %s%> may only be used in compound statements",
"acc enter data");
break;
}
else if (context != pragma_compound)
goto bad_stmt;
cp_parser_omp_construct (parser, pragma_tok, if_p);
return true;
case PRAGMA_OACC_EXIT_DATA:
if (context == pragma_stmt)
{
error_at (pragma_tok->location,
"%<#pragma %s%> may only be used in compound statements",
"acc exit data");
break;
}
else if (context != pragma_compound)
goto bad_stmt;
cp_parser_omp_construct (parser, pragma_tok, if_p);
return true;
case PRAGMA_OACC_ROUTINE:
if (context != pragma_external)
{
error_at (pragma_tok->location,
"%<#pragma acc routine%> must be at file scope");
break;
}
cp_parser_oacc_routine (parser, pragma_tok, context);
return false;
case PRAGMA_OACC_UPDATE:
if (context == pragma_stmt)
{
error_at (pragma_tok->location,
"%<#pragma %s%> may only be used in compound statements",
"acc update");
break;
}
else if (context != pragma_compound)
goto bad_stmt;
cp_parser_omp_construct (parser, pragma_tok, if_p);
return true;
case PRAGMA_OACC_WAIT:
if (context == pragma_stmt)
{
error_at (pragma_tok->location,
"%<#pragma %s%> may only be used in compound statements",
"acc wait");
break;
}
else if (context != pragma_compound)
goto bad_stmt;
cp_parser_omp_construct (parser, pragma_tok, if_p);
return true;
case PRAGMA_OACC_ATOMIC:
case PRAGMA_OACC_CACHE:
case PRAGMA_OACC_DATA:
case PRAGMA_OACC_HOST_DATA:
case PRAGMA_OACC_KERNELS:
case PRAGMA_OACC_PARALLEL:
case PRAGMA_OACC_LOOP:
case PRAGMA_OMP_ATOMIC:
case PRAGMA_OMP_CRITICAL:
case PRAGMA_OMP_DISTRIBUTE:
case PRAGMA_OMP_FOR:
case PRAGMA_OMP_MASTER:
case PRAGMA_OMP_PARALLEL:
case PRAGMA_OMP_SECTIONS:
case PRAGMA_OMP_SIMD:
case PRAGMA_OMP_SINGLE:
case PRAGMA_OMP_TASK:
case PRAGMA_OMP_TASKGROUP:
case PRAGMA_OMP_TASKLOOP:
case PRAGMA_OMP_TEAMS:
if (context != pragma_stmt && context != pragma_compound)
goto bad_stmt;
stmt = push_omp_privatization_clauses (false);
cp_parser_omp_construct (parser, pragma_tok, if_p);
pop_omp_privatization_clauses (stmt);
return true;
case PRAGMA_OMP_ORDERED:
if (context != pragma_stmt && context != pragma_compound)
goto bad_stmt;
stmt = push_omp_privatization_clauses (false);
ret = cp_parser_omp_ordered (parser, pragma_tok, context, if_p);
pop_omp_privatization_clauses (stmt);
return ret;
case PRAGMA_OMP_TARGET:
if (context != pragma_stmt && context != pragma_compound)
goto bad_stmt;
stmt = push_omp_privatization_clauses (false);
ret = cp_parser_omp_target (parser, pragma_tok, context, if_p);
pop_omp_privatization_clauses (stmt);
return ret;
case PRAGMA_OMP_END_DECLARE_TARGET:
cp_parser_omp_end_declare_target (parser, pragma_tok);
return false;
case PRAGMA_OMP_SECTION:
error_at (pragma_tok->location, 
"%<#pragma omp section%> may only be used in "
"%<#pragma omp sections%> construct");
break;
case PRAGMA_IVDEP:
{
if (context == pragma_external)
{
error_at (pragma_tok->location,
"%<#pragma GCC ivdep%> must be inside a function");
break;
}
const bool ivdep = cp_parser_pragma_ivdep (parser, pragma_tok);
unsigned short unroll;
cp_token *tok = cp_lexer_peek_token (the_parser->lexer);
if (tok->type == CPP_PRAGMA
&& cp_parser_pragma_kind (tok) == PRAGMA_UNROLL)
{
tok = cp_lexer_consume_token (parser->lexer);
unroll = cp_parser_pragma_unroll (parser, tok);
tok = cp_lexer_peek_token (the_parser->lexer);
}
else
unroll = 0;
if (tok->type != CPP_KEYWORD
|| (tok->keyword != RID_FOR
&& tok->keyword != RID_WHILE
&& tok->keyword != RID_DO))
{
cp_parser_error (parser, "for, while or do statement expected");
return false;
}
cp_parser_iteration_statement (parser, if_p, ivdep, unroll);
return true;
}
case PRAGMA_UNROLL:
{
if (context == pragma_external)
{
error_at (pragma_tok->location,
"%<#pragma GCC unroll%> must be inside a function");
break;
}
const unsigned short unroll
= cp_parser_pragma_unroll (parser, pragma_tok);
bool ivdep;
cp_token *tok = cp_lexer_peek_token (the_parser->lexer);
if (tok->type == CPP_PRAGMA
&& cp_parser_pragma_kind (tok) == PRAGMA_IVDEP)
{
tok = cp_lexer_consume_token (parser->lexer);
ivdep = cp_parser_pragma_ivdep (parser, tok);
tok = cp_lexer_peek_token (the_parser->lexer);
}
else
ivdep = false;
if (tok->type != CPP_KEYWORD
|| (tok->keyword != RID_FOR
&& tok->keyword != RID_WHILE
&& tok->keyword != RID_DO))
{
cp_parser_error (parser, "for, while or do statement expected");
return false;
}
cp_parser_iteration_statement (parser, if_p, ivdep, unroll);
return true;
}
default:
gcc_assert (id >= PRAGMA_FIRST_EXTERNAL);
c_invoke_pragma_handler (id);
break;
bad_stmt:
cp_parser_error (parser, "expected declaration specifiers");
break;
}
cp_parser_skip_to_pragma_eol (parser, pragma_tok);
return false;
}
enum cpp_ttype
pragma_lex (tree *value, location_t *loc)
{
cp_token *tok = cp_lexer_peek_token (the_parser->lexer);
enum cpp_ttype ret = tok->type;
*value = tok->u.value;
if (loc)
*loc = tok->location;
if (ret == CPP_PRAGMA_EOL || ret == CPP_EOF)
ret = CPP_EOF;
else if (ret == CPP_STRING)
*value = cp_parser_string_literal (the_parser, false, false);
else
{
if (ret == CPP_KEYWORD)
ret = CPP_NAME;
cp_lexer_consume_token (the_parser->lexer);
}
return ret;
}

void
c_parse_file (void)
{
static bool already_called = false;
if (already_called)
fatal_error (input_location,
"inter-module optimizations not implemented for C++");
already_called = true;
the_parser = cp_parser_new ();
push_deferring_access_checks (flag_access_control
? dk_no_deferred : dk_no_check);
cp_parser_translation_unit (the_parser);
the_parser = NULL;
}
static GTY(()) int generic_parm_count;
static tree
make_generic_type_name ()
{
char buf[32];
sprintf (buf, "auto:%d", ++generic_parm_count);
return get_identifier (buf);
}
static tree
synthesize_implicit_template_parm  (cp_parser *parser, tree constr)
{
gcc_assert (current_binding_level->kind == sk_function_parms);
if (parser->implicit_template_scope && constr)
{
tree t = parser->implicit_template_parms;
while (t)
{
if (equivalent_placeholder_constraints (TREE_TYPE (t), constr))
{
tree d = TREE_VALUE (t);
if (TREE_CODE (d) == PARM_DECL)
d = DECL_INITIAL (d);
return d;
}
t = TREE_CHAIN (t);
}
}
cp_binding_level *const entry_scope = current_binding_level;
bool become_template = false;
cp_binding_level *parent_scope = 0;
if (parser->implicit_template_scope)
{
gcc_assert (parser->implicit_template_parms);
current_binding_level = parser->implicit_template_scope;
}
else
{
cp_binding_level *scope = entry_scope;
while (scope->kind == sk_function_parms)
{
parent_scope = scope;
scope = scope->level_chain;
}
if (current_class_type && !LAMBDA_TYPE_P (current_class_type))
{
while (scope->kind == sk_class && !scope->defining_class_p)
{
parent_scope = scope;
scope = scope->level_chain;
}
}
current_binding_level = scope;
if (scope->kind != sk_template_parms
|| !function_being_declared_is_template_p (parser))
{
become_template = true;
parser->implicit_template_scope
= begin_scope (sk_template_parms, NULL);
++processing_template_decl;
parser->fully_implicit_function_template_p = true;
++parser->num_template_parameter_lists;
}
else
{
gcc_assert (current_template_parms);
parser->implicit_template_scope = scope;
tree v = INNERMOST_TEMPLATE_PARMS (current_template_parms);
parser->implicit_template_parms
= TREE_VEC_ELT (v, TREE_VEC_LENGTH (v) - 1);
}
}
tree proto = constr ? DECL_INITIAL (constr) : NULL_TREE;
tree synth_id = make_generic_type_name ();
tree synth_tmpl_parm;
bool non_type = false;
if (proto == NULL_TREE || TREE_CODE (proto) == TYPE_DECL)
synth_tmpl_parm
= finish_template_type_parm (class_type_node, synth_id);
else if (TREE_CODE (proto) == TEMPLATE_DECL)
synth_tmpl_parm
= finish_constrained_template_template_parm (proto, synth_id);
else
{
synth_tmpl_parm = copy_decl (proto);
DECL_NAME (synth_tmpl_parm) = synth_id;
non_type = true;
}
tree node = build_tree_list (NULL_TREE, synth_tmpl_parm);
TREE_TYPE (node) = constr;
tree new_parm
= process_template_parm (parser->implicit_template_parms,
input_location,
node,
non_type,
false);
if (parser->implicit_template_parms)
parser->implicit_template_parms
= TREE_CHAIN (parser->implicit_template_parms);
else
parser->implicit_template_parms = new_parm;
tree new_decl = get_local_decls ();
if (non_type)
new_decl = DECL_INITIAL (new_decl);
if (become_template)
{
parent_scope->level_chain = current_binding_level;
tree new_parms = make_tree_vec (1);
TREE_VEC_ELT (new_parms, 0) = parser->implicit_template_parms;
current_template_parms = tree_cons (size_int (processing_template_decl),
new_parms, current_template_parms);
}
else
{
tree& new_parms = INNERMOST_TEMPLATE_PARMS (current_template_parms);
int new_parm_idx = TREE_VEC_LENGTH (new_parms);
new_parms = grow_tree_vec (new_parms, new_parm_idx + 1);
TREE_VEC_ELT (new_parms, new_parm_idx) = parser->implicit_template_parms;
}
if (tree req = TEMPLATE_PARM_CONSTRAINTS (tree_last (new_parm)))
{
tree reqs = TEMPLATE_PARMS_CONSTRAINTS (current_template_parms);
reqs = conjoin_constraints (reqs, req);
TEMPLATE_PARMS_CONSTRAINTS (current_template_parms) = reqs;
}
current_binding_level = entry_scope;
return new_decl;
}
static tree
finish_fully_implicit_template (cp_parser *parser, tree member_decl_opt)
{
gcc_assert (parser->fully_implicit_function_template_p);
if (member_decl_opt && member_decl_opt != error_mark_node
&& DECL_VIRTUAL_P (member_decl_opt))
{
error_at (DECL_SOURCE_LOCATION (member_decl_opt),
"implicit templates may not be %<virtual%>");
DECL_VIRTUAL_P (member_decl_opt) = false;
}
if (member_decl_opt)
member_decl_opt = finish_member_template_decl (member_decl_opt);
end_template_decl ();
parser->fully_implicit_function_template_p = false;
parser->implicit_template_parms = 0;
parser->implicit_template_scope = 0;
--parser->num_template_parameter_lists;
return member_decl_opt;
}
static void
abort_fully_implicit_template (cp_parser *parser)
{
cp_binding_level *return_to_scope = current_binding_level;
if (parser->implicit_template_scope
&& return_to_scope != parser->implicit_template_scope)
{
cp_binding_level *child = return_to_scope;
for (cp_binding_level *scope = child->level_chain;
scope != parser->implicit_template_scope;
scope = child->level_chain)
child = scope;
child->level_chain = parser->implicit_template_scope->level_chain;
parser->implicit_template_scope->level_chain = return_to_scope;
current_binding_level = parser->implicit_template_scope;
}
else
return_to_scope = return_to_scope->level_chain;
finish_fully_implicit_template (parser, NULL);
gcc_assert (current_binding_level == return_to_scope);
}
void
maybe_show_extern_c_location (void)
{
if (the_parser->innermost_linkage_specification_location != UNKNOWN_LOCATION)
inform (the_parser->innermost_linkage_specification_location,
"%<extern \"C\"%> linkage started here");
}
#include "gt-cp-parser.h"
