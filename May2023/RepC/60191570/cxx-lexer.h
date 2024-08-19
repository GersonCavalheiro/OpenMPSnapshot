#ifndef CXX_LEXER_H
#define CXX_LEXER_H
#include <stdio.h>
#include "libmcxx-common.h"
#include "cxx-driver-decls.h"
#include "cxx-macros.h"
MCXX_BEGIN_DECLS
typedef 
struct token_atrib_tag 
{
const char* token_text;
} token_atrib_t;
typedef struct parser_location_tag
{
const char* first_filename;
int first_line;
int first_column;
} parser_location_t;
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
do                                                                  \
if (N)                                                            \
{                                                               \
(Current).first_filename = YYRHSLOC (Rhs, 1).first_filename;  \
(Current).first_line     = YYRHSLOC (Rhs, 1).first_line;      \
(Current).first_column   = YYRHSLOC (Rhs, 1).first_column;    \
}                                                               \
else                                                              \
{                                                               \
(Current).first_filename   =                                  \
YYRHSLOC (Rhs, 0).first_filename;                           \
(Current).first_line =                                        \
YYRHSLOC (Rhs, 0).first_line;                               \
(Current).first_column =                                      \
YYRHSLOC (Rhs, 0).first_column;                             \
}                                                               \
while (0)
LIBMCXX_EXTERN void update_parser_location(const char* current_text, parser_location_t* loc);
#define YYLTYPE parser_location_t
LIBMCXX_EXTERN int mcxx_open_file_for_scanning(const char* scanned_filename, const char* input_filename);
LIBMCXX_EXTERN int mc99_open_file_for_scanning(const char* scanned_filename, const char* input_filename);
LIBMCXX_EXTERN int mcxx_prepare_string_for_scanning(const char* str);
LIBMCXX_EXTERN int mc99_prepare_string_for_scanning(const char* str);
LIBMCXX_EXTERN void register_new_directive(
compilation_configuration_t* configuration,
const char* prefix, const char* directive, char is_construct, 
char bound_to_single_stmt);
LIBMCXX_EXTERN pragma_directive_kind_t lookup_pragma_directive(const char* prefix, const char* directive);
LIBMCXX_EXTERN int mc99_flex_debug;
LIBMCXX_EXTERN int mcxx_flex_debug;
LIBMCXX_EXTERN int mcxxdebug;
LIBMCXX_EXTERN int mc99debug;
MCXX_END_DECLS
#endif 
