#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include <string.h>
#include <errno.h>
#include <signal.h>
#include "cxx-ast.h"
#include "fortran03-utils.h"
#include "fortran03-split.h"
#include "fortran03-lexer.h"
#include "fortran03-parser-internal.h"
#include "cxx-utils.h"
#include "cxx-driver-utils.h"
static char check_for_comment(char* c);
static char check_for_construct(char *c, char *prefix, int max_length);
static void double_continuate(FILE* output, const char* c, int width, int* column);
static void double_continuate_construct(FILE* output, 
const char* prefix, 
const char* c, int width, int* column);
static char* read_whole_line(FILE* input);
static void trim_right_line(char* c);
typedef void* YY_BUFFER_STATE;
extern int mf03_prepare_string_for_scanning(const char* str);
extern int mf03lex(void);
extern YYSTYPE mf03lval;
void fortran_split_lines(FILE* input, FILE* output, int width)
{
ERROR_CONDITION(width <= 0, "Invalid width = %d\n", width);
int length;
char* line;
while ((line = read_whole_line(input)) != NULL)
{
trim_right_line(line);
char prefix[33] = { 0 };
char is_construct = check_for_construct(line, prefix, 32);
char is_comment = check_for_comment(line);
length = strlen(line);
if ((length <= width))
{
fputs(line, output);
}
else if (is_construct)
{
int column = 1;
double_continuate_construct(output, prefix, line, width, &column);
fprintf(output, "\n");
}
else if (is_comment)
{
fputs(line, output);
}
else
{
int column, next_column;
char* position;
char* next_position;
mf03_prepare_string_for_scanning(line);
column = 1;
position = line;
int token = mf03lex();
while (token != EOS)
{
next_position = strstr(position, mf03lval.token_atrib.token_text);
if (next_position == NULL)
{
fatal_error("Serious problem when splitting line. '%s' not found:\n\n %s", mf03lval.token_atrib.token_text, position);
}
next_column = column + (next_position - position);
if (column == width
|| (next_column + (int)strlen(mf03lval.token_atrib.token_text) >= width))
{
DEBUG_CODE() DEBUG_MESSAGE("Cutting at '%s'", mf03lval.token_atrib.token_text);
fprintf(output, "&\n");
column = 1;
}
char* c;
for (c = position; c < next_position; c++)
{
DEBUG_CODE() DEBUG_MESSAGE("%d - Blank - '%c'", column, *c);
fprintf(output, "%c", *c);
column++;
}
if ((column + (int)strlen(mf03lval.token_atrib.token_text)) >= width)
{
double_continuate(output, mf03lval.token_atrib.token_text, width, &column);
}
else
{
DEBUG_CODE() DEBUG_MESSAGE("%d - Token '%s'", column, mf03lval.token_atrib.token_text);
fprintf(output, "%s", mf03lval.token_atrib.token_text);
column += strlen(mf03lval.token_atrib.token_text);
}
position = next_position + strlen(mf03lval.token_atrib.token_text);
token = mf03lex();
}
fprintf(output, "\n");
}
DELETE(line);
}
}
static void double_continuate(FILE* output, const char* c, int width, int* column)
{
for (; *c != '\0'; c++)
{
if ((*column == width) && (*c != '\n'))
{
fprintf(output, "&\n&");
*column = 2;
DEBUG_CODE() DEBUG_MESSAGE("Cutting at '%c'", *c);
}
DEBUG_CODE() DEBUG_MESSAGE("%d - Letter - '%c'", *column, *c);
fprintf(output, "%c", *c);
(*column)++;
}
}
static void double_continuate_construct(FILE* output, 
const char* prefix, 
const char* c, int width, int* column)
{
char prefix_start[64];
snprintf(prefix_start, 63, "!$%s&", prefix);
for (; *c != '\0'; c++)
{
if ((*column == width) && (*c != '\n'))
{
fprintf(output, "&\n%s", prefix_start);
*column = 1 + strlen(prefix_start);
DEBUG_CODE() DEBUG_MESSAGE("Cutting at '%c'", *c);
}
DEBUG_CODE() DEBUG_MESSAGE("%d - Letter - '%c'", *column, *c);
fprintf(output, "%c", *c);
(*column)++;
}
}
static char check_for_comment(char* c)
{
char* iter = c;
while (*iter == ' ' || *iter == '\t') iter++;
return (*iter == '!');
}
static char check_for_construct(char *c, char *prefix, int max_length)
{
char* iter = c;
while (*iter == ' ' || *iter == '\t') iter++;
if (*iter != '!')
return 0;
iter++;
if (*iter != '$')
return 0;
iter++;
char *q = prefix;
*q = '\0';
int length = 0;
while (*iter != ' ' 
&& *iter != '\t' 
&& *iter != '\0')
{
if (length >= max_length)
{
return 0;
}
*q = *iter;
q++;
iter++;
length++;
}
int i;
char found = 0;
for (i = 0; i < CURRENT_CONFIGURATION->num_pragma_custom_prefix; i++)
{
if (strcasecmp(prefix, CURRENT_CONFIGURATION->pragma_custom_prefix[i]) == 0)
{
found = 1;
break;
}
}
return found;
}
static char* read_whole_line(FILE* input)
{
int buffer_size = 1024;
int was_eof;
int length_read;
char* temporal_buffer = NEW_VEC0(char, buffer_size);
if (fgets(temporal_buffer, buffer_size, input) == NULL)
{
if (ferror(input))
{
fatal_error("error: while starting to split file\n");
}
}
if (temporal_buffer[0] == '\0')
{
DELETE(temporal_buffer);
return NULL;
}
length_read = strlen(temporal_buffer);
was_eof = feof(input);
while ((temporal_buffer[length_read - 1] != '\n') && !was_eof)
{
temporal_buffer = NEW_REALLOC(char, temporal_buffer, 2*buffer_size);
if (fgets(&temporal_buffer[length_read], buffer_size, input) == NULL)
{
if (ferror(input))
{
fatal_error("error: while splitting file\n");
}
}
length_read = strlen(temporal_buffer);
buffer_size = buffer_size * 2;
was_eof = feof(input);
}
return temporal_buffer;
}
static void trim_right_line(char* c)
{
int save_newline = 0;
int length = strlen(c);
if (c[length-1] == '\n')
{
length--;
save_newline = 1;
}
if (length > 0)
{
length--;
while ((length >= 0) && (c[length] == ' ')) length--;
if (!save_newline)
{
c[length + 1] = '\0';
}
else
{
c[length + 1] = '\n';
c[length + 2] = '\0';
}
}
}
