#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#ifdef USE_BUNDLED_REGEX
#include "mf95-regex.h"
#else
#include <regex.h>
#endif
#include "prescanner-identifier.h"
#include <signal.h>
#include "cxx-utils.h"
enum prescanner_lex_tokens
{
KW_REAL = 1,
KW_INTEGER,
KW_LOGICAL,
KW_COMPLEX,
KW_DOUBLEPRECISION,
KW_CHARACTER,
KW_PREFIX,
KW_FUNCTION,
KW_SUBROUTINE,
KW_END_APPEND,
KW_END_COMMENT,
KW_END,
KW_MODULE_PROC,
KW_OMP_DIRECTIVE,
KW_OMP_CLAUSE,
KW_OMP_CLAUSE_EXPR,
KW_OMP_EXPR_TOK,
KW_OMP_UNKNOWN_DIRECTIVE,
KW_OTHER
};
#define ECHO fprintf(stderr, "Error unknown token: -%s-\n", prescannertext);
#include <prescanner-scanner.h>
typedef
struct sentence_information_tag
{
char has_free_assign;
char has_free_comma;
char has_free_doublecolon;
char* statement;
char* original_text;
} sentence_information_t;
typedef
struct line_information_tag
{
int num_line;
char has_label;
char is_comment;
char is_prepro_line;
char label[6];
int num_of_statements;
int room_for_statements;
struct sentence_information_tag* statement_list;
char* comment_text;
} line_information_t;
struct statements_information_tag
{
char is_declaration;
char needs_space;
};
typedef struct statements_information_tag statements_information_t;
#define STATEMENT_SET \
STATEMENT_INFO(ST_ASSIGNMENT, 0, 0, NULL) \
STATEMENT_INFO(ST_TYPESPEC, 1, 1, NULL) \
STATEMENT_INFO(ST_INITIALIZATION, 1, 0, NULL) \
STATEMENT_INFO(ST_TYPEDECL, 1, 1, NULL) \
STATEMENT_INFO(ST_PRIVATE, 1, 1, "private") \
STATEMENT_INFO(ST_PUBLIC, 1, 1, "public") \
STATEMENT_INFO(ST_PROGRAM, 1, 1, "program") \
STATEMENT_INFO(ST_SUBROUTINE, 1, 1, "subroutine") \
STATEMENT_INFO(ST_FUNCTION, 1, 1, "function") \
STATEMENT_INFO(ST_MODULE, 1, 1, "module") \
STATEMENT_INFO(ST_MODULE_PROCEDURE, 1, 1, "module procedure") \
STATEMENT_INFO(ST_BLOCKDATA, 1, 1, "blockdata") \
STATEMENT_INFO(ST_USE, 1, 1, "use") \
STATEMENT_INFO(ST_IMPLICIT, 1, 1, "implicit") \
STATEMENT_INFO(ST_PARAMETER, 1, 0, "parameter") \
STATEMENT_INFO(ST_FORMAT, 1, 1, "format") \
STATEMENT_INFO(ST_ENTRY, 1, 1, "entry") \
STATEMENT_INFO(ST_ACCESS, 1, 1, "access") \
STATEMENT_INFO(ST_ALLOCATABLE, 1, 1, "allocatable") \
STATEMENT_INFO(ST_COMMON, 1, 1, "common") \
STATEMENT_INFO(ST_CONTAINS, 0, 0, "contains") \
STATEMENT_INFO(ST_DATA, 1, 1, "data") \
STATEMENT_INFO(ST_DIMENSION, 1, 1, "dimension") \
STATEMENT_INFO(ST_EQUIVALENCE, 1, 0, "equivalence") \
STATEMENT_INFO(ST_EXTERNAL, 1, 1, "external") \
STATEMENT_INFO(ST_INTENT, 1, 0, "intent") \
STATEMENT_INFO(ST_INTRINSIC, 1, 1, "intrinsic") \
STATEMENT_INFO(ST_NAMELIST, 1, 0, "namelist") \
STATEMENT_INFO(ST_OPTIONAL, 1, 1, "optional") \
STATEMENT_INFO(ST_POINTER, 1, 1, "pointer") \
STATEMENT_INFO(ST_SAVE, 1, 1, "save") \
STATEMENT_INFO(ST_VALUE, 1, 1, "value") \
STATEMENT_INFO(ST_VOLATILE, 1, 1, "volatile") \
STATEMENT_INFO(ST_TARGET, 1, 1, "target") \
STATEMENT_INFO(ST_DO, 0, 1, "do") \
STATEMENT_INFO(ST_LABELED_DO, 0, 1, "do") \
STATEMENT_INFO(ST_FORALL, 0, 0, "forall") \
STATEMENT_INFO(ST_ARITHMETIC_IF, 0, 0, "if") \
STATEMENT_INFO(ST_IF, 0, 0, "if") \
STATEMENT_INFO(ST_WHERE, 0, 0, "where") \
STATEMENT_INFO(ST_ALLOCATE, 0, 0, "allocate") \
STATEMENT_INFO(ST_BLOCK, 0, 1, "block") \
STATEMENT_INFO(ST_BACKSPACE, 0, 1, "backspace") \
STATEMENT_INFO(ST_CALL, 0, 1, "call") \
STATEMENT_INFO(ST_CLOSE, 0, 0, "close") \
STATEMENT_INFO(ST_CONTINUE, 0, 0, "continue") \
STATEMENT_INFO(ST_CYCLE, 0, 1, "cycle") \
STATEMENT_INFO(ST_DEALLOCATE, 0, 0, "deallocate") \
STATEMENT_INFO(ST_ENDFILE, 0, 1, "endfile") \
STATEMENT_INFO(ST_EXIT, 0, 1, "exit") \
STATEMENT_INFO(ST_GOTO, 0, 1, "goto") \
STATEMENT_INFO(ST_LABEL_ASSIGN, 0, 1, "assign") \
STATEMENT_INFO(ST_INQUIRE, 0, 0, "inquire") \
STATEMENT_INFO(ST_NULLIFY, 0, 0, "nullify") \
STATEMENT_INFO(ST_OPEN, 0, 0, "open") \
STATEMENT_INFO(ST_PRINT, 0, 1, "print") \
STATEMENT_INFO(ST_READ, 0, 0, "read") \
STATEMENT_INFO(ST_RETURN, 0, 1, "return") \
STATEMENT_INFO(ST_REWIND, 0, 1, "rewind") \
STATEMENT_INFO(ST_PAUSE, 0, 1, "pause") \
STATEMENT_INFO(ST_STOP, 0, 1, "stop") \
STATEMENT_INFO(ST_WRITE, 0, 0, "write") \
STATEMENT_INFO(ST_INTERFACE, 0, 1, "interface") \
STATEMENT_INFO(ST_ELSE, 0, 1, "else") \
STATEMENT_INFO(ST_ELSEIF, 0, 1, "elseif") \
STATEMENT_INFO(ST_SELECTCASE, 0, 0, "selectcase") \
STATEMENT_INFO(ST_CASE, 0, 1, "case") \
STATEMENT_INFO(ST_END, 0, 1, "end") \
STATEMENT_INFO(ST_IF_STMT, 0, 1, "if") \
STATEMENT_INFO(ST_BIND, 1, 0, "bind") \
STATEMENT_INFO(DC_INCLUDE, 0, 1, "include") \
statements_information_t statements_info[] =
{
#define STATEMENT_INFO(statement, is_decl, needs_space, keyword) \
[statement] = {is_decl, needs_space},
STATEMENT_SET
#undef STATEMENT_INFO
};
const char * statement_names[] =
{
#define STATEMENT_INFO(statement, _, __, ___) \
[statement] = #statement ,
STATEMENT_SET
#undef STATEMENT_INFO
};
static int i_isblank(int c);
static line_information_t* get_information_from_line(prescanner_t* prescanner, char* c);
static language_level identify_and_convert_line(prescanner_t* prescanner, language_level previous, line_information_t* li, int sentence_index);
static void identify_and_convert_omp_directive(line_information_t* li);
static void remove_all_spaces(char** line);
static void add_blank(char** line, char* keyword);
static void add_blank_function(char** line);
static void add_blank_subroutine(char** line);
static void add_blank_end(char** line);
static void add_blank_if_statement(prescanner_t* prescanner, char** line, int num_line);
static void add_blank_labeled_do(char** line);
static void add_blank_module_procedure(char** line);
static void add_blank_label_assign_statement(char** line);
static void add_blank_entry_statement(char** line, char* keyword);
static void add_blank_elseif_statement(char** line, char* keyword);
language_level convert_line(prescanner_t* prescanner, language_level previous, char** line, int num_line)
{
int i;
int original_size;
line_information_t* li;
language_level next;
original_size = strlen(*line);
li = get_information_from_line(prescanner, *line);
li->num_line = num_line;
next = previous;
if (!li->is_comment)
{
for (i = 0;  i < li -> num_of_statements; i++)
{
next = identify_and_convert_line(prescanner, next, li, i);
}
}
else if (prescanner->openmp_processing)
{
identify_and_convert_omp_directive(li);
}
DELETE(*line);
*line = NEW_VEC0(char, strlen(li->label) + 1 + original_size*2);
if (li->is_comment
|| li->is_prepro_line)
{
strcat(*line, li->comment_text);
DELETE(li->comment_text);
}
else
{
if (li->has_label)
{
strcat(*line, li->label);
strcat(*line, " ");
}
for (i = 0; i < li->num_of_statements; i++)
{
if (i > 0) strcat(*line, "; ");
strcat(*line, li->statement_list[i].statement);
DELETE(li->statement_list[i].statement);
}
DELETE(li->statement_list);
}
return next;
}
static line_information_t* get_information_from_line(prescanner_t* prescanner, char* c)
{
line_information_t* li;
li = NEW0(line_information_t);
char* t = c;
while (i_isblank(*t)) t++;
if (*t == '!')
{
if (prescanner->openmp_processing && (strncmp(c, "!$ ", 3) == 0))
{
c[0] = ' ';
c[1] = ' ';
}
else
{
li->is_comment = 1;
li->comment_text = xstrdup(c);
return li;
}
}
if (*t == '#')
{
li->is_prepro_line = 1;
li->comment_text = xstrdup(c);
return li;
}
char* p = c;
int i, j = 0;
for(i = 0; i < 5; i++)
{
if (!i_isblank(*p)) 
{
li->has_label = 1;
li->label[j] = *p;
j++;
}
p++;
}
li->label[j] = '\0';
li->room_for_statements = 5;
li->statement_list = NEW_VEC0(sentence_information_t, li->room_for_statements);
li->num_of_statements = 0;
p = &c[6];
int parenthesis_level = 0;
char in_string = 0, delim = 0;
int current_sentence = 0;
char* start_current_sentence = p;
while (*p != '\0')
{
if (!in_string)
{
if (*p == '!')
{
break;
}
else if ((*p == '\"') || (*p == '\''))
{
delim = *p;
in_string = 1;
}
else if (*p == '(')
{
parenthesis_level++;
}
else if (*p == ')')
{
parenthesis_level--;
}
else if ((*p == ';') && (parenthesis_level == 0))
{
*p = '\0';
li->statement_list[current_sentence].statement = xstrdup(start_current_sentence);
*p = ';';
li->num_of_statements++;
current_sentence++;
if (current_sentence >= li->room_for_statements)
{
li->statement_list = NEW_REALLOC(sentence_information_t, li->statement_list, 2 * li->room_for_statements);
memset(&li->statement_list[li->room_for_statements], 0, sizeof(*li->statement_list)*li->room_for_statements);
li->room_for_statements *= 2;
}
start_current_sentence = p + 1;
}
else if ((*p == ',') && (parenthesis_level == 0))
{
li->statement_list[current_sentence].has_free_comma = 1;
}
else if ((*p == '=') && (parenthesis_level == 0))
{
char* q = p + 1;
while ((*q != '\0') && i_isblank(*q)) q++;
if ((*q != '\0') && (*q != '=') ) 
{
li->statement_list[current_sentence].has_free_assign = 1;
}
else if ((*q != '\0') && (*q == '='))
{
p = q;
}
}
else if ((*p == ':') && (parenthesis_level == 0))
{
char* q = p + 1;
while ((*q != '\0') && i_isblank(*q)) q++;
if ((*q != '\0') && (*q == ':'))
{
li->statement_list[current_sentence].has_free_doublecolon = 1;
p = q;
}
}
}
else 
{
if (*p == delim)
{
if (*(p+1) != delim)
{
in_string = 0;
}
else 
{
p++;
}
}
}
p++;
}
if (start_current_sentence < p)
{
li->statement_list[current_sentence].statement = xstrdup(start_current_sentence);
li->num_of_statements++;
}
for (i = 0; i < li->num_of_statements; i++)
{
li->statement_list[i].original_text = xstrdup(li->statement_list[i].statement);
remove_all_spaces(&li->statement_list[i].statement);
}
return li;
}
static void remove_all_spaces(char** line)
{
int allocated_size = strlen(*line) + 5;
char* newline = NEW_VEC0(char, allocated_size);
char in_string = 0, inlined_comment = 0, delim = 0;
char *p, *q;
q = newline;
p = *line;
while (*p != '\0')
{
ERROR_CONDITION((q - newline) >= allocated_size, "Buffer overflow detected\n", 0);
if (!in_string)
{
if (!inlined_comment) 
{
if (*p != ' ' && *p != '\t' && *p != '!')
{
*q = *p;
q++;
}
else if (*p == '!')
{
*q = ' '; q++;
*q = *p; q++;
inlined_comment = 1;
}
if(*p == '\'' || *p == '"')
{
delim = *p;
in_string = 1;
}
}
else 
{
*q = *p;
q++;
}
}
else
{
*q = *p;
q++;
if ((*p == delim) && (*(p+1) != delim))
{
in_string = 0;
}
}
p++;
}
*q = '\0';
DELETE(*line);
*line = newline;
}
static const char *remove_blanks_keeping_holleriths(const char* c)
{
const char *p = c;
char* result = NEW_VEC0(char, strlen(c) + 1);
char *q = result;
int length = -1;
while (*p != '\0')
{
if ('0' <= *p && *p <= '9')
{
if (length < 0) length = 0;
length = length * 10 + (*p - '0');
*q = *p; p++; q++;
}
else if (*p == 'H' || *p == 'h')
{
if (length < 0)
{
*q = *p; p++; q++;
}
else
{
*q = *p; p++; q++;
int i;
for (i = length; i > 0; i--)
{
*q = *p; p++; q++;
}
length = -1;
}
}
else if (*p == '\'' || *p == '"')
{
char delim = *p;
*q = *p; p++; q++;
while ((*p != delim
|| *(p+1) == delim)
&& *p == '\0')
{
if (*p == delim
&& *(p+1) == delim)
{
*q = *p; p++; q++;
*q = *p; p++; q++;
}
else
{
*q = *p; p++; q++;
}
}
*q = *p; p++; q++;
length = -1;
}
else if (*p == ' ')
{
p++;
}
else
{
*q = *p; p++; q++;
length = -1;
}
}
*q = '\0';
return result;
}
static language_level identify_and_convert_line(prescanner_t* prescanner, 
language_level previous, line_information_t* li, int statement_index)
{
language_level next = previous;
int statement;
if (strlen(li->statement_list[statement_index].statement) == 0)
{
return next;
}
if (li->statement_list[statement_index].has_free_assign &&
!li->statement_list[statement_index].has_free_comma)
{
if (li->statement_list[statement_index].has_free_doublecolon)
{
statement = ST_INITIALIZATION;
}
else
{
statement = ST_ASSIGNMENT;
}
}
else
{
prescanner_flush_buffer(YY_CURRENT_BUFFER);
prescanner_delete_buffer(YY_CURRENT_BUFFER);
YY_BUFFER_STATE yybuf = prescanner_scan_string(li->statement_list[statement_index].statement);
prescanner_flex_debug = 0;
prescanner_switch_to_buffer(yybuf);
statement = prescannerlex();
if (!statement)
{
fprintf(stderr, "%s:%d: warning: could not classify statement\n",
prescanner->input_filename, li->num_line);
fprintf(stderr, "%s:%d: info: unrecognized statement follows\n\n",
prescanner->input_filename, li->num_line);
fprintf(stderr, "%s\n\n", li->statement_list[statement_index].statement);
}
}
if (statement == DC_INCLUDE)
{
add_blank(&li->statement_list[statement_index].statement, prescannertext);
return next;
}
if (statement != ST_DUBIOUS_FUNCTION)
{
if (statements_info[statement].is_declaration)
{
DEBUG_CODE()
{
fprintf(stderr, "Switching to LANG_DECLARATION_PART (%s) (statement = %s)\n",
li->statement_list[statement_index].statement,
statement_names[statement]);
}
next = LANG_DECLARATION_PART;
}
else
{
DEBUG_CODE()
{
fprintf(stderr, "Switching to LANG_INSTRUCTION_PART (%s) (statement = %s)\n",
li->statement_list[statement_index].statement,
statement_names[statement]);
}
next = LANG_INSTRUCTION_PART;
}
if (statements_info[statement].needs_space)
{
if (statement == ST_FUNCTION)
{
add_blank_function(&li->statement_list[statement_index].statement);
}
else if (statement == ST_SUBROUTINE)
{
add_blank_subroutine(&li->statement_list[statement_index].statement);
}
else if (statement == ST_TYPESPEC)
{
DEBUG_CODE()
{
fprintf(stderr, "'%s' We have to add spaces to this type declaration (1)\n", li->statement_list[statement_index].statement);
}
add_blank(&li->statement_list[statement_index].statement, prescannertext);
}
else if (statement == ST_IF_STMT)
{
DEBUG_CODE()
{
fprintf(stderr, "'%s' is an IF statement\n", li->statement_list[statement_index].statement);
}
add_blank_if_statement(prescanner, &li->statement_list[statement_index].statement, li->num_line);
}
else if (statement == ST_END)
{
add_blank_end(&li->statement_list[statement_index].statement);
}
else if (statement == ST_LABELED_DO)
{
DEBUG_CODE()
{
fprintf(stderr, "'%s' is a labeled do\n", li->statement_list[statement_index].statement);
}
add_blank_labeled_do(&li->statement_list[statement_index].statement);
}
else if (statement == ST_MODULE_PROCEDURE)
{
add_blank_module_procedure(&li->statement_list[statement_index].statement);
}
else if (statement == ST_LABEL_ASSIGN)
{
add_blank_label_assign_statement(&li->statement_list[statement_index].statement);
}
else if (statement == ST_ENTRY)
{
add_blank_entry_statement(&li->statement_list[statement_index].statement, prescannertext);
}
else if (statement == ST_FORMAT
|| statement == ST_DATA)
{
const char* keyword = NULL;
switch (statement)
{
case ST_FORMAT:
keyword = "FORMAT";
break;
case ST_DATA:
keyword = "DATA";
break;
default:
{
internal_error("Code unreachable", 0);
}
}
int n = strlen(keyword) + strlen(li->statement_list[statement_index].original_text) + 2;
char *c = NEW_VEC(char, n);
c[0] = '\0';
const char *p = NULL;
p = li->statement_list[statement_index].original_text;
const char *current_letter = keyword;
while (*current_letter != '\0')
{
while (*p == ' ' || *p == '\t')
p++;
if (toupper(*p) == *current_letter)
{
p++;
current_letter++;
}
else
{
internal_error("Invalid text '%s'\n", p);
}
}
while (*p == ' ' || *p == '\t')
p++;
if (statement == ST_DATA)
{
p = remove_blanks_keeping_holleriths(p);
}
DELETE(li->statement_list[statement_index].statement);
snprintf(c, n, "%s %s", keyword, p);
c[n-1] = '\0';
li->statement_list[statement_index].statement = c;
}
else if (statement == ST_ELSEIF)
{
add_blank_elseif_statement(&li->statement_list[statement_index].statement, prescannertext);
}
else
{
add_blank(&li->statement_list[statement_index].statement, prescannertext);
}
}
}
else
{
if (next == LANG_TOP_LEVEL || next == LANG_INSTRUCTION_PART)
{
DEBUG_CODE()
{
fprintf(stderr, "'%s' is a function\n", li->statement_list[statement_index].statement);
}
add_blank_function(&li->statement_list[statement_index].statement);
next = LANG_DECLARATION_PART;
}
else
{
DEBUG_CODE()
{
fprintf(stderr, "'%s' We have to add spaces to this type declaration (2)\n", li->statement_list[statement_index].statement);
}
prescanner_flush_buffer(YY_CURRENT_BUFFER);
prescanner_delete_buffer(YY_CURRENT_BUFFER);
YY_BUFFER_STATE yybuf = prescanner_scan_string(li->statement_list[statement_index].statement);
prescanner_flex_debug = 0;
prescanner_switch_to_buffer(yybuf);
BEGIN(PREFIX_SPEC);
prescannerlex();
BEGIN(0);
add_blank(&li->statement_list[statement_index].statement, prescannertext);
next = LANG_DECLARATION_PART;
}
}
return next;
}
static void add_blank(char** line, char* keyword)
{
char* new = NULL;
new = NEW_VEC0(char, 1 + strlen(*line) + 1);
strncat(new, *line, strlen(keyword));
strcat(new, " ");
strcat(new, ((*line) + strlen(keyword)));
DELETE(*line);
*line = new;
}
static void add_blank_function(char** line)
{
char* temp;
int keyword, scanned_length = 0;
int in_parenthesis = 0;
temp = NEW_VEC0(char, strlen(*line)*2);
prescanner_flush_buffer(YY_CURRENT_BUFFER);
prescanner_delete_buffer(YY_CURRENT_BUFFER);
YY_BUFFER_STATE yybuf = prescanner_scan_string(*line);
prescanner_flex_debug = 0;
prescanner_switch_to_buffer(yybuf);
BEGIN(PREFIX_SPEC);
keyword = prescannerlex();
scanned_length += strlen(prescannertext);
while (keyword != KW_FUNCTION)
{
switch (keyword)
{
case KW_PREFIX:
{
strcat(temp, prescannertext);
if (!in_parenthesis)
{
strcat(temp, " ");
}
break;
}
case KW_OTHER :
{
strcat(temp, prescannertext);
if (*prescannertext == '(')
{   
in_parenthesis++;
}
else if (in_parenthesis && *prescannertext == ')')
{   
in_parenthesis--;
if (!in_parenthesis) strcat(temp, " ");
}
break;
}
}
keyword = prescannerlex();
scanned_length += strlen(prescannertext);
}
strcat(temp, prescannertext);
strcat(temp, " ");
strcat(temp, &(*line)[scanned_length]);
DELETE(*line);
*line = temp;
BEGIN(0);
}
static void add_blank_subroutine(char** line)
{
char* temp;
int  keyword, scanned_length = 0;
temp = NEW_VEC0(char, strlen(*line)*2);
prescanner_flush_buffer(YY_CURRENT_BUFFER);
prescanner_delete_buffer(YY_CURRENT_BUFFER);
YY_BUFFER_STATE yybuf = prescanner_scan_string(*line);
prescanner_flex_debug = 0;
prescanner_switch_to_buffer(yybuf);
BEGIN(PREFIX_SPEC);
keyword = prescannerlex();
scanned_length += strlen(prescannertext);
while (keyword != KW_SUBROUTINE)
{
switch (keyword)
{
case KW_PREFIX : 
{
strcat(temp, prescannertext);
strcat(temp, " ");
break;
}
case KW_OTHER :
{
strcat(temp, prescannertext);
break;
}
}
keyword = prescannerlex();
scanned_length += strlen(prescannertext);
}
strcat(temp, prescannertext);
strcat(temp, " ");
strcat(temp, &(*line)[scanned_length]);
DELETE(*line);
*line = temp;
BEGIN(0);
}
static void add_blank_end(char** line)
{
char* temp;
char keyword_after_end = 0;
int  keyword, scanned_length = 0;
temp = NEW_VEC0(char, strlen(*line)*2);
prescanner_flush_buffer(YY_CURRENT_BUFFER);
prescanner_delete_buffer(YY_CURRENT_BUFFER);
YY_BUFFER_STATE yybuf = prescanner_scan_string(*line);
prescanner_flex_debug = 0;
prescanner_switch_to_buffer(yybuf);
BEGIN(END_APPENDINGS);
keyword = prescannerlex();
while (keyword != 0 && !keyword_after_end)
{
switch (keyword)
{
case KW_END_APPEND :
{
keyword_after_end = 1;
strcat(temp, prescannertext);
strcat(temp, " ");
break;
}
case KW_END :
{
strcat(temp, prescannertext);
strcat(temp, " ");
break;
}
case KW_END_COMMENT :
{
keyword_after_end = 1;
strcat(temp, prescannertext);
break;
}
}
scanned_length += strlen(prescannertext);
if (!keyword_after_end)
keyword = prescannerlex();
}
strcat(temp, &(*line)[scanned_length]);
DELETE(*line);
*line = temp;
BEGIN(0);
}
static void add_blank_module_procedure(char** line)
{
char* temp;
int scanned_length = 0;
temp = NEW_VEC0(char, strlen(*line)*2);
prescanner_flush_buffer(YY_CURRENT_BUFFER);
prescanner_delete_buffer(YY_CURRENT_BUFFER);
YY_BUFFER_STATE yybuf = prescanner_scan_string(*line);
prescanner_flex_debug = 0;
prescanner_switch_to_buffer(yybuf);
BEGIN(MODULE_PROC);
prescannerlex();
scanned_length += strlen(prescannertext);
strcat(temp, prescannertext);
strcat(temp, " ");
prescannerlex();
scanned_length += strlen(prescannertext);
strcat(temp, prescannertext);
strcat(temp, " ");
strcat(temp, &(*line)[scanned_length]);
DELETE(*line);
*line = temp;
BEGIN(0);
}
static void add_blank_if_statement(prescanner_t* prescanner, char** line, int num_line)
{
char level = 1, delim = 0;
char in_string = 0;
char* p = &(*line)[3]; 
while ((*p != '\0') && (level > 0))
{
if (!in_string && (*p == ')'))
{
level--;
}
else if (!in_string && (*p == '('))
{
level++;
}
else if ((*p == '\'') || (*p == '"'))
{
if (!in_string)
{
in_string = 1;
delim = *p;
}
else if (*(p+1) != delim)
{
in_string = 0;
}
else if (*(p+1) == delim)
{
p++;
}
}
p++;
}
if (level > 0)
{
fprintf(stderr, "Malformed IF statement. Rejecting to modify this line.\n'%s'\n", *line);
}
else
{
char* c = NEW_VEC0(char, 6 + strlen(p) + 1);
strcat(c, "      ");
strcat(c, p);
DEBUG_CODE()
{
fprintf(stderr, "Before converting '%s'\n", c);
}
convert_line(prescanner, LANG_INSTRUCTION_PART, &c, num_line);
DEBUG_CODE()
{
fprintf(stderr, "After converting '%s'\n", c);
}
char* temp = NEW_VEC0(char, strlen(*line)*2);
*p = '\0';
strcat(temp, *line);
strcat(temp, c);
DELETE(c);
DELETE(*line);
*line = temp;
}
}
static void add_blank_labeled_do(char** line)
{
int i;
char ending;
char* temp = NEW_VEC0(char, strlen(*line)*2);
ending = (*line)[2];
(*line)[2] = '\0';
strcat(temp, (*line));
strcat(temp, " ");
(*line)[2] = ending;
i = 2;
while (isdigit((*line)[i])) i++;
ending = (*line)[i];
(*line)[i] = '\0';
strcat(temp, &(*line)[2]);
strcat(temp, " ");
(*line)[i] = ending;    
strcat(temp, &(*line)[i]);
DELETE(*line);
*line = temp;
}
static void add_blank_label_assign_statement(char** line)
{
int i;
char ending;
char* temp = NEW_VEC0(char, strlen(*line)*2);
ending = (*line)[6];
(*line)[6] = '\0';
strcat(temp, (*line));
strcat(temp, " ");
(*line)[6] = ending;
i = 6;
while (isdigit((*line)[i])) i++;
ending = (*line)[i];
(*line)[i] = '\0';
strcat(temp, &(*line)[6]);
strcat(temp, " ");
(*line)[i] = ending;
ending = (*line)[i+2];
(*line)[i+2] = '\0';
strcat(temp, &(*line)[i]);
strcat(temp, " ");
(*line)[i+2] = ending;
strcat(temp, &(*line)[i+2]);
DELETE(*line);
*line = temp;
}
static void add_blank_entry_statement(char** line, char* keyword)
{
regex_t match_problematic;
regmatch_t sub_matching[2];
int code;
char* temp;
temp = NEW_VEC0(char, strlen(*line)*2);
if ((code = regcomp(&match_problematic, "^ENTRY([A-Z][0-9A-Z]*)RESULT[(][A-Z][0-9A-Z]*[)]$", REG_EXTENDED | REG_ICASE)) != 0)
{
char error_message[120];
regerror(code, &match_problematic, error_message, 120);
internal_error("Error when compiling regular expression (%s)\n", error_message);
}
if (regexec(&match_problematic, *line, 2, sub_matching, 0) == 0)
{
strcat(temp, prescannertext);
strcat(temp, " ");
char mark = (*line)[sub_matching[1].rm_eo];
(*line)[sub_matching[1].rm_eo] = '\0';
strcat(temp,&((*line)[sub_matching[1].rm_so]));
(*line)[sub_matching[1].rm_eo] = mark;
strcat(temp, " ");
strcat(temp, &((*line)[sub_matching[1].rm_eo]));
DELETE(*line);
*line = temp;
}
else 
{
add_blank(line, keyword);
}
regfree(&match_problematic);
}
static void add_blank_elseif_statement(char** line, char* keyword)
{
const char* p = *line;
p += strlen(keyword);
if (*p != '(')
{
fprintf(stderr, "Malformed ELSE IF statement, expecting '('. Rejecting to modify this line.\n'%s'\n", *line);
return;
}
int parent_nest = 1;
char in_string = 0;
char str_delim = '\0';
while (*p != '\0')
{
if (!in_string
&& *p == ')'
&& --parent_nest == 1)
{
break;
}
else if (!in_string
&& *p == '(')
{
parent_nest++;
}
else if (!in_string && (*p == '\'' || *p == '"'))
{
in_string = 1;
str_delim = *p;
}
else if (in_string
&& *p == str_delim
&& *(p+1) == str_delim)
{
p++;
}
else if (in_string
&& *p == str_delim
&& *(p+1) != str_delim)
{
in_string = 0;
str_delim = '\0';
}
p++;
}
if (*p != ')')
{
fprintf(stderr, "Malformed ELSE IF statement, expecting ')'. Rejecting to modify this line.\n'%s'\n", *line);
return;
}
p++;
if (strncasecmp(p, "THEN", strlen("THEN")) != 0)
{
fprintf(stderr, "Malformed ELSE IF statement, expecting 'THEN' Rejecting to modify this line.\n'%s'\n", *line);
return;
}
p += strlen("THEN");
if (*p != '\0')
{
const char* q = *line;
int length = (p - q) + 1 + strlen(p) + 1;
char *new_str = NEW_VEC0(char, length);
strncat(new_str, q, p - q);
strcat(new_str, " ");
strcat(new_str, p);
DELETE(*line);
*line = new_str;
}
}
static void identify_and_convert_omp_directive(line_information_t* li)
{
if ((strncasecmp(li->comment_text, "!$omp ", 6) != 0))
{
return;
}
char* c = xstrdup(&li->comment_text[6]);
char* result = NEW_VEC0(char, strlen(c)*2);
strncat(result, li->comment_text, 6);
remove_all_spaces(&c);
prescanner_flush_buffer(YY_CURRENT_BUFFER);
prescanner_delete_buffer(YY_CURRENT_BUFFER);
YY_BUFFER_STATE yybuf = prescanner_scan_string(c);
prescanner_flex_debug = 0;
prescanner_switch_to_buffer(yybuf);
BEGIN(OMP_DIRECTIVE);
int lex = prescannerlex();
if (lex == KW_OMP_UNKNOWN_DIRECTIVE)
{
BEGIN(0);
DELETE(c);
return;
}
int parenthesis_level = 0;
strcat(result, prescannertext);
while (YYSTATE == OMP_DIRECTIVE)
{
prescannerlex();
strcat(result, " ");
strcat(result, prescannertext);
}
lex = prescannerlex();
while (lex != 0)
{
switch (lex)
{
case KW_OMP_CLAUSE :
{
strcat(result, " ");
strcat(result, prescannertext);
break;
}
case KW_OMP_CLAUSE_EXPR : 
{
strcat(result, " ");
strcat(result, prescannertext);
parenthesis_level = 0;
break;
}
case KW_OMP_EXPR_TOK :
{
if (strlen(prescannertext) == 1)
{
switch (prescannertext[0])
{
case '(' :
{
parenthesis_level++;
break;
}
case ')' :
{
parenthesis_level--;
if (parenthesis_level == 0)
{
BEGIN(OMP_CLAUSE);
}
break;
}
default:
break;
}
}
strcat(result, prescannertext);
break;
}
default: 
break;
}
lex = prescannerlex();
}
BEGIN(0);
DELETE(li->comment_text);
li->comment_text = result;
DELETE(c);
}
static int i_isblank(int c)
{
return (c == ' ' || c == '\t');
}
