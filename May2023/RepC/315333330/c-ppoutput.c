#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "c-common.h"		
#include "../libcpp/internal.h"
#include "c-pragma.h"		
#include "file-prefix-map.h"    
static struct
{
FILE *outf;			
const cpp_token *prev;	
const cpp_token *source;	
int src_line;			
bool printed;			
bool first_time;		
bool prev_was_system_token;	
const char *src_file;		
} print;
struct macro_queue
{
struct macro_queue *next;	
char *macro;			
};
static macro_queue *define_queue, *undef_queue;
static void scan_translation_unit (cpp_reader *);
static void print_lines_directives_only (int, const void *, size_t);
static void scan_translation_unit_directives_only (cpp_reader *);
static void scan_translation_unit_trad (cpp_reader *);
static void account_for_newlines (const unsigned char *, size_t);
static int dump_macro (cpp_reader *, cpp_hashnode *, void *);
static void dump_queued_macros (cpp_reader *);
static bool print_line_1 (source_location, const char*, FILE *);
static bool print_line (source_location, const char *);
static bool maybe_print_line_1 (source_location, FILE *);
static bool maybe_print_line (source_location);
static bool do_line_change (cpp_reader *, const cpp_token *,
source_location, int);
static void cb_line_change (cpp_reader *, const cpp_token *, int);
static void cb_define (cpp_reader *, source_location, cpp_hashnode *);
static void cb_undef (cpp_reader *, source_location, cpp_hashnode *);
static void cb_used_define (cpp_reader *, source_location, cpp_hashnode *);
static void cb_used_undef (cpp_reader *, source_location, cpp_hashnode *);
static void cb_include (cpp_reader *, source_location, const unsigned char *,
const char *, int, const cpp_token **);
static void cb_ident (cpp_reader *, source_location, const cpp_string *);
static void cb_def_pragma (cpp_reader *, source_location);
static void cb_read_pch (cpp_reader *pfile, const char *name,
int fd, const char *orig_name);
void
preprocess_file (cpp_reader *pfile)
{
if (flag_no_output && pfile->buffer)
{
while (pfile->buffer->prev)
cpp_scan_nooutput (pfile);
cpp_scan_nooutput (pfile);
}
else if (cpp_get_options (pfile)->traditional)
scan_translation_unit_trad (pfile);
else if (cpp_get_options (pfile)->directives_only
&& !cpp_get_options (pfile)->preprocessed)
scan_translation_unit_directives_only (pfile);
else
scan_translation_unit (pfile);
if (flag_dump_macros == 'M')
cpp_forall_identifiers (pfile, dump_macro, NULL);
if (print.printed)
putc ('\n', print.outf);
}
void
init_pp_output (FILE *out_stream)
{
cpp_callbacks *cb = cpp_get_callbacks (parse_in);
if (!flag_no_output)
{
cb->line_change = cb_line_change;
if (cpp_get_options (parse_in)->lang != CLK_ASM)
{
cb->ident      = cb_ident;
cb->def_pragma = cb_def_pragma;
}
}
if (flag_dump_includes)
cb->include  = cb_include;
if (flag_pch_preprocess)
{
cb->valid_pch = c_common_valid_pch;
cb->read_pch = cb_read_pch;
}
if (flag_dump_macros == 'N' || flag_dump_macros == 'D')
{
cb->define = cb_define;
cb->undef  = cb_undef;
}
if (flag_dump_macros == 'U')
{
cb->before_define = dump_queued_macros;
cb->used_define = cb_used_define;
cb->used_undef = cb_used_undef;
}
cb->has_attribute = c_common_has_attribute;
cb->get_source_date_epoch = cb_get_source_date_epoch;
cb->remap_filename = remap_macro_filename;
print.src_line = 1;
print.printed = false;
print.prev = 0;
print.outf = out_stream;
print.first_time = 1;
print.src_file = "";
print.prev_was_system_token = false;
}
static void
scan_translation_unit (cpp_reader *pfile)
{
bool avoid_paste = false;
bool do_line_adjustments
= cpp_get_options (parse_in)->lang != CLK_ASM
&& !flag_no_line_commands;
bool in_pragma = false;
bool line_marker_emitted = false;
print.source = NULL;
for (;;)
{
source_location loc;
const cpp_token *token = cpp_get_token_with_location (pfile, &loc);
if (token->type == CPP_PADDING)
{
avoid_paste = true;
if (print.source == NULL
|| (!(print.source->flags & PREV_WHITE)
&& token->val.source == NULL))
print.source = token->val.source;
continue;
}
if (token->type == CPP_EOF)
break;
if (avoid_paste)
{
int src_line = LOCATION_LINE (loc);
if (print.source == NULL)
print.source = token;
if (src_line != print.src_line
&& do_line_adjustments
&& !in_pragma)
{
line_marker_emitted = do_line_change (pfile, token, loc, false);
putc (' ', print.outf);
print.printed = true;
}
else if (print.source->flags & PREV_WHITE
|| (print.prev
&& cpp_avoid_paste (pfile, print.prev, token))
|| (print.prev == NULL && token->type == CPP_HASH))
{
putc (' ', print.outf);
print.printed = true;
}
}
else if (token->flags & PREV_WHITE)
{
int src_line = LOCATION_LINE (loc);
if (src_line != print.src_line
&& do_line_adjustments
&& !in_pragma)
line_marker_emitted = do_line_change (pfile, token, loc, false);
putc (' ', print.outf);
print.printed = true;
}
avoid_paste = false;
print.source = NULL;
print.prev = token;
if (token->type == CPP_PRAGMA)
{
const char *space;
const char *name;
line_marker_emitted = maybe_print_line (token->src_loc);
fputs ("#pragma ", print.outf);
c_pp_lookup_pragma (token->val.pragma, &space, &name);
if (space)
fprintf (print.outf, "%s %s", space, name);
else
fprintf (print.outf, "%s", name);
print.printed = true;
in_pragma = true;
}
else if (token->type == CPP_PRAGMA_EOL)
{
maybe_print_line (token->src_loc);
in_pragma = false;
}
else
{
if (cpp_get_options (parse_in)->debug)
linemap_dump_location (line_table, token->src_loc, print.outf);
if (do_line_adjustments
&& !in_pragma
&& !line_marker_emitted
&& print.prev_was_system_token != !!in_system_header_at (loc)
&& !is_location_from_builtin_token (loc))
{
do_line_change (pfile, token, loc, false);
print.prev_was_system_token = !!in_system_header_at (loc);
}
cpp_output_token (token, print.outf);
line_marker_emitted = false;
print.printed = true;
}
if (cpp_token_val_index (token) == CPP_TOKEN_FLD_STR)
account_for_newlines (token->val.str.text, token->val.str.len);
}
}
static void
print_lines_directives_only (int lines, const void *buf, size_t size)
{
print.src_line += lines;
fwrite (buf, 1, size, print.outf);
}
static void
scan_translation_unit_directives_only (cpp_reader *pfile)
{
struct _cpp_dir_only_callbacks cb;
cb.print_lines = print_lines_directives_only;
cb.maybe_print_line = maybe_print_line;
_cpp_preprocess_dir_only (pfile, &cb);
}
static void
account_for_newlines (const unsigned char *str, size_t len)
{
while (len--)
if (*str++ == '\n')
print.src_line++;
}
static void
scan_translation_unit_trad (cpp_reader *pfile)
{
while (_cpp_read_logical_line_trad (pfile))
{
size_t len = pfile->out.cur - pfile->out.base;
maybe_print_line (pfile->out.first_line);
fwrite (pfile->out.base, 1, len, print.outf);
print.printed = true;
if (!CPP_OPTION (pfile, discard_comments))
account_for_newlines (pfile->out.base, len);
}
}
static bool
maybe_print_line_1 (source_location src_loc, FILE *stream)
{
bool emitted_line_marker = false;
int src_line = LOCATION_LINE (src_loc);
const char *src_file = LOCATION_FILE (src_loc);
if (print.printed)
{
putc ('\n', stream);
print.src_line++;
print.printed = false;
}
if (!flag_no_line_commands
&& src_line >= print.src_line
&& src_line < print.src_line + 8
&& strcmp (src_file, print.src_file) == 0)
{
while (src_line > print.src_line)
{
putc ('\n', stream);
print.src_line++;
}
}
else
emitted_line_marker = print_line_1 (src_loc, "", stream);
return emitted_line_marker;
}
static bool
maybe_print_line (source_location src_loc)
{
if (cpp_get_options (parse_in)->debug)
linemap_dump_location (line_table, src_loc,
print.outf);
return maybe_print_line_1 (src_loc, print.outf);
}
static bool
print_line_1 (source_location src_loc, const char *special_flags, FILE *stream)
{
bool emitted_line_marker = false;
if (print.printed)
putc ('\n', stream);
print.printed = false;
if (!flag_no_line_commands)
{
const char *file_path = LOCATION_FILE (src_loc);
int sysp;
size_t to_file_len = strlen (file_path);
unsigned char *to_file_quoted =
(unsigned char *) alloca (to_file_len * 4 + 1);
unsigned char *p;
print.src_line = LOCATION_LINE (src_loc);
print.src_file = file_path;
p = cpp_quote_string (to_file_quoted,
(const unsigned char *) file_path,
to_file_len);
*p = '\0';
fprintf (stream, "# %u \"%s\"%s",
print.src_line == 0 ? 1 : print.src_line,
to_file_quoted, special_flags);
sysp = in_system_header_at (src_loc);
if (sysp == 2)
fputs (" 3 4", stream);
else if (sysp == 1)
fputs (" 3", stream);
putc ('\n', stream);
emitted_line_marker = true;
}
return emitted_line_marker;
}
static bool
print_line (source_location src_loc, const char *special_flags)
{
if (cpp_get_options (parse_in)->debug)
linemap_dump_location (line_table, src_loc,
print.outf);
return print_line_1 (src_loc, special_flags, print.outf);
}
static bool
do_line_change (cpp_reader *pfile, const cpp_token *token,
source_location src_loc, int parsing_args)
{
bool emitted_line_marker = false;
if (define_queue || undef_queue)
dump_queued_macros (pfile);
if (token->type == CPP_EOF || parsing_args)
return false;
emitted_line_marker = maybe_print_line (src_loc);
print.prev = 0;
print.source = 0;
if (!CPP_OPTION (pfile, traditional))
{
int spaces = LOCATION_COLUMN (src_loc) - 2;
print.printed = true;
while (-- spaces >= 0)
putc (' ', print.outf);
}
return emitted_line_marker;
}
static void
cb_line_change (cpp_reader *pfile, const cpp_token *token,
int parsing_args)
{
do_line_change (pfile, token, token->src_loc, parsing_args);
}
static void
cb_ident (cpp_reader *pfile ATTRIBUTE_UNUSED, source_location line,
const cpp_string *str)
{
maybe_print_line (line);
fprintf (print.outf, "#ident %s\n", str->text);
print.src_line++;
}
static void
cb_define (cpp_reader *pfile, source_location line, cpp_hashnode *node)
{
const line_map_ordinary *map;
maybe_print_line (line);
fputs ("#define ", print.outf);
if (flag_dump_macros == 'D')
fputs ((const char *) cpp_macro_definition (pfile, node),
print.outf);
else
fputs ((const char *) NODE_NAME (node), print.outf);
putc ('\n', print.outf);
print.printed = false;
linemap_resolve_location (line_table, line,
LRK_MACRO_DEFINITION_LOCATION,
&map);
if (LINEMAP_LINE (map) != 0)
print.src_line++;
}
static void
cb_undef (cpp_reader *pfile ATTRIBUTE_UNUSED, source_location line,
cpp_hashnode *node)
{
maybe_print_line (line);
fprintf (print.outf, "#undef %s\n", NODE_NAME (node));
print.src_line++;
}
static void
cb_used_define (cpp_reader *pfile, source_location line ATTRIBUTE_UNUSED,
cpp_hashnode *node)
{
macro_queue *q;
if (node->flags & NODE_BUILTIN)
return;
q = XNEW (macro_queue);
q->macro = xstrdup ((const char *) cpp_macro_definition (pfile, node));
q->next = define_queue;
define_queue = q;
}
static void
cb_used_undef (cpp_reader *pfile ATTRIBUTE_UNUSED,
source_location line ATTRIBUTE_UNUSED,
cpp_hashnode *node)
{
macro_queue *q;
q = XNEW (macro_queue);
q->macro = xstrdup ((const char *) NODE_NAME (node));
q->next = undef_queue;
undef_queue = q;
}
static void
dump_queued_macros (cpp_reader *pfile ATTRIBUTE_UNUSED)
{
macro_queue *q;
if (print.printed)
{
putc ('\n', print.outf);
print.src_line++;
print.printed = false;
}
for (q = define_queue; q;)
{
macro_queue *oq;
fputs ("#define ", print.outf);
fputs (q->macro, print.outf);
putc ('\n', print.outf);
print.printed = false;
print.src_line++;
oq = q;
q = q->next;
free (oq->macro);
free (oq);
}
define_queue = NULL;
for (q = undef_queue; q;)
{
macro_queue *oq;
fprintf (print.outf, "#undef %s\n", q->macro);
print.src_line++;
oq = q;
q = q->next;
free (oq->macro);
free (oq);
}
undef_queue = NULL;
}
static void
cb_include (cpp_reader *pfile ATTRIBUTE_UNUSED, source_location line,
const unsigned char *dir, const char *header, int angle_brackets,
const cpp_token **comments)
{
maybe_print_line (line);
if (angle_brackets)
fprintf (print.outf, "#%s <%s>", dir, header);
else
fprintf (print.outf, "#%s \"%s\"", dir, header);
if (comments != NULL)
{
while (*comments != NULL)
{
if ((*comments)->flags & PREV_WHITE)
putc (' ', print.outf);
cpp_output_token (*comments, print.outf);
++comments;
}
}
putc ('\n', print.outf);
print.printed = false;
print.src_line++;
}
void
pp_dir_change (cpp_reader *pfile ATTRIBUTE_UNUSED, const char *dir)
{
size_t to_file_len = strlen (dir);
unsigned char *to_file_quoted =
(unsigned char *) alloca (to_file_len * 4 + 1);
unsigned char *p;
p = cpp_quote_string (to_file_quoted, (const unsigned char *) dir, to_file_len);
*p = '\0';
fprintf (print.outf, "# 1 \"%s
}
void
pp_file_change (const line_map_ordinary *map)
{
const char *flags = "";
if (flag_no_line_commands)
return;
if (map != NULL)
{
input_location = map->start_location;
if (print.first_time)
{
if (!cpp_get_options (parse_in)->preprocessed)
print_line (map->start_location, flags);
print.first_time = 0;
}
else
{
if (map->reason == LC_ENTER)
{
const line_map_ordinary *from = INCLUDED_FROM (line_table, map);
maybe_print_line (LAST_SOURCE_LINE_LOCATION (from));
}
if (map->reason == LC_ENTER)
flags = " 1";
else if (map->reason == LC_LEAVE)
flags = " 2";
print_line (map->start_location, flags);
}
}
}
static void
cb_def_pragma (cpp_reader *pfile, source_location line)
{
maybe_print_line (line);
fputs ("#pragma ", print.outf);
cpp_output_line (pfile, print.outf);
print.printed = false;
print.src_line++;
}
static int
dump_macro (cpp_reader *pfile, cpp_hashnode *node, void *v ATTRIBUTE_UNUSED)
{
if (node->type == NT_MACRO && !(node->flags & NODE_BUILTIN))
{
fputs ("#define ", print.outf);
fputs ((const char *) cpp_macro_definition (pfile, node),
print.outf);
putc ('\n', print.outf);
print.printed = false;
print.src_line++;
}
return 1;
}
static void
cb_read_pch (cpp_reader *pfile, const char *name,
int fd, const char *orig_name ATTRIBUTE_UNUSED)
{
c_common_read_pch (pfile, name, fd, orig_name);
fprintf (print.outf, "#pragma GCC pch_preprocess \"%s\"\n", name);
print.src_line++;
}
