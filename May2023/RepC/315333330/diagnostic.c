#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "version.h"
#include "demangle.h"
#include "intl.h"
#include "backtrace.h"
#include "diagnostic.h"
#include "diagnostic-color.h"
#include "edit-context.h"
#include "selftest.h"
#include "selftest-diagnostic.h"
#ifdef HAVE_TERMIOS_H
# include <termios.h>
#endif
#ifdef GWINSZ_IN_SYS_IOCTL
# include <sys/ioctl.h>
#endif
#define pedantic_warning_kind(DC)			\
((DC)->pedantic_errors ? DK_ERROR : DK_WARNING)
#define permissive_error_kind(DC) ((DC)->permissive ? DK_WARNING : DK_ERROR)
#define permissive_error_option(DC) ((DC)->opt_permissive)
static bool diagnostic_impl (rich_location *, int, const char *,
va_list *, diagnostic_t) ATTRIBUTE_GCC_DIAG(3,0);
static bool diagnostic_n_impl (rich_location *, int, unsigned HOST_WIDE_INT,
const char *, const char *, va_list *,
diagnostic_t) ATTRIBUTE_GCC_DIAG(5,0);
static void error_recursion (diagnostic_context *) ATTRIBUTE_NORETURN;
static void real_abort (void) ATTRIBUTE_NORETURN;
const char *progname;
static diagnostic_context global_diagnostic_context;
diagnostic_context *global_dc = &global_diagnostic_context;

char *
build_message_string (const char *msg, ...)
{
char *str;
va_list ap;
va_start (ap, msg);
str = xvasprintf (msg, ap);
va_end (ap);
return str;
}
char *
file_name_as_prefix (diagnostic_context *context, const char *f)
{
const char *locus_cs
= colorize_start (pp_show_color (context->printer), "locus");
const char *locus_ce = colorize_stop (pp_show_color (context->printer));
return build_message_string ("%s%s:%s ", locus_cs, f, locus_ce);
}

int
get_terminal_width (void)
{
const char * s = getenv ("COLUMNS");
if (s != NULL) {
int n = atoi (s);
if (n > 0)
return n;
}
#ifdef TIOCGWINSZ
struct winsize w;
w.ws_col = 0;
if (ioctl (0, TIOCGWINSZ, &w) == 0 && w.ws_col > 0)
return w.ws_col;
#endif
return INT_MAX;
}
void
diagnostic_set_caret_max_width (diagnostic_context *context, int value)
{
value = value ? value - 1 
: (isatty (fileno (pp_buffer (context->printer)->stream))
? get_terminal_width () - 1: INT_MAX);
if (value <= 0) 
value = INT_MAX;
context->caret_max_width = value;
}
void
diagnostic_initialize (diagnostic_context *context, int n_opts)
{
int i;
context->printer = XNEW (pretty_printer);
new (context->printer) pretty_printer ();
memset (context->diagnostic_count, 0, sizeof context->diagnostic_count);
context->warning_as_error_requested = false;
context->n_opts = n_opts;
context->classify_diagnostic = XNEWVEC (diagnostic_t, n_opts);
for (i = 0; i < n_opts; i++)
context->classify_diagnostic[i] = DK_UNSPECIFIED;
context->show_caret = false;
diagnostic_set_caret_max_width (context, pp_line_cutoff (context->printer));
for (i = 0; i < rich_location::STATICALLY_ALLOCATED_RANGES; i++)
context->caret_chars[i] = '^';
context->show_option_requested = false;
context->abort_on_error = false;
context->show_column = false;
context->pedantic_errors = false;
context->permissive = false;
context->opt_permissive = 0;
context->fatal_errors = false;
context->dc_inhibit_warnings = false;
context->dc_warn_system_headers = false;
context->max_errors = 0;
context->internal_error = NULL;
diagnostic_starter (context) = default_diagnostic_starter;
context->start_span = default_diagnostic_start_span_fn;
diagnostic_finalizer (context) = default_diagnostic_finalizer;
context->option_enabled = NULL;
context->option_state = NULL;
context->option_name = NULL;
context->last_location = UNKNOWN_LOCATION;
context->last_module = 0;
context->x_data = NULL;
context->lock = 0;
context->inhibit_notes_p = false;
context->colorize_source_p = false;
context->show_ruler_p = false;
context->parseable_fixits_p = false;
context->edit_context_ptr = NULL;
}
void
diagnostic_color_init (diagnostic_context *context, int value )
{
if (value < 0)
{
if (DIAGNOSTICS_COLOR_DEFAULT == -1)
{
if (!getenv ("GCC_COLORS"))
return;
value = DIAGNOSTICS_COLOR_AUTO;
}
else
value = DIAGNOSTICS_COLOR_DEFAULT;
}
pp_show_color (context->printer)
= colorize_init ((diagnostic_color_rule_t) value);
}
void
diagnostic_finish (diagnostic_context *context)
{
if (diagnostic_kind_count (context, DK_WERROR))
{
if (context->warning_as_error_requested)
pp_verbatim (context->printer,
_("%s: all warnings being treated as errors"),
progname);
else
pp_verbatim (context->printer,
_("%s: some warnings being treated as errors"),
progname);
pp_newline_and_flush (context->printer);
}
diagnostic_file_cache_fini ();
XDELETEVEC (context->classify_diagnostic);
context->classify_diagnostic = NULL;
context->printer->~pretty_printer ();
XDELETE (context->printer);
context->printer = NULL;
if (context->edit_context_ptr)
{
delete context->edit_context_ptr;
context->edit_context_ptr = NULL;
}
}
void
diagnostic_set_info_translated (diagnostic_info *diagnostic, const char *msg,
va_list *args, rich_location *richloc,
diagnostic_t kind)
{
gcc_assert (richloc);
diagnostic->message.err_no = errno;
diagnostic->message.args_ptr = args;
diagnostic->message.format_spec = msg;
diagnostic->message.m_richloc = richloc;
diagnostic->richloc = richloc;
diagnostic->kind = kind;
diagnostic->option_index = 0;
}
void
diagnostic_set_info (diagnostic_info *diagnostic, const char *gmsgid,
va_list *args, rich_location *richloc,
diagnostic_t kind)
{
gcc_assert (richloc);
diagnostic_set_info_translated (diagnostic, _(gmsgid), args, richloc, kind);
}
static const char *const diagnostic_kind_color[] = {
#define DEFINE_DIAGNOSTIC_KIND(K, T, C) (C),
#include "diagnostic.def"
#undef DEFINE_DIAGNOSTIC_KIND
NULL
};
const char *
diagnostic_get_color_for_kind (diagnostic_t kind)
{
return diagnostic_kind_color[kind];
}
static const char *
maybe_line_and_column (int line, int col)
{
static char result[32];
if (line)
{
size_t l = snprintf (result, sizeof (result),
col ? ":%d:%d" : ":%d", line, col);
gcc_checking_assert (l < sizeof (result));
}
else
result[0] = 0;
return result;
}
static char *
diagnostic_get_location_text (diagnostic_context *context,
expanded_location s)
{
pretty_printer *pp = context->printer;
const char *locus_cs = colorize_start (pp_show_color (pp), "locus");
const char *locus_ce = colorize_stop (pp_show_color (pp));
const char *file = s.file ? s.file : progname;
int line = strcmp (file, N_("<built-in>")) ? s.line : 0;
int col = context->show_column ? s.column : 0;
const char *line_col = maybe_line_and_column (line, col);
return build_message_string ("%s%s%s:%s", locus_cs, file,
line_col, locus_ce);
}
char *
diagnostic_build_prefix (diagnostic_context *context,
const diagnostic_info *diagnostic)
{
static const char *const diagnostic_kind_text[] = {
#define DEFINE_DIAGNOSTIC_KIND(K, T, C) (T),
#include "diagnostic.def"
#undef DEFINE_DIAGNOSTIC_KIND
"must-not-happen"
};
gcc_assert (diagnostic->kind < DK_LAST_DIAGNOSTIC_KIND);
const char *text = _(diagnostic_kind_text[diagnostic->kind]);
const char *text_cs = "", *text_ce = "";
pretty_printer *pp = context->printer;
if (diagnostic_kind_color[diagnostic->kind])
{
text_cs = colorize_start (pp_show_color (pp),
diagnostic_kind_color[diagnostic->kind]);
text_ce = colorize_stop (pp_show_color (pp));
}
expanded_location s = diagnostic_expand_location (diagnostic);
char *location_text = diagnostic_get_location_text (context, s);
char *result = build_message_string ("%s %s%s%s", location_text,
text_cs, text, text_ce);
free (location_text);
return result;
}
static const char * const bt_stop[] =
{
"main",
"toplev::main",
"execute_one_pass",
"compile_file",
};
static int
bt_callback (void *data, uintptr_t pc, const char *filename, int lineno,
const char *function)
{
int *pcount = (int *) data;
if (filename == NULL && function == NULL)
return 0;
if (*pcount == 0
&& filename != NULL
&& strcmp (lbasename (filename), "diagnostic.c") == 0)
return 0;
if (*pcount >= 20)
{
return 1;
}
++*pcount;
char *alc = NULL;
if (function != NULL)
{
char *str = cplus_demangle_v3 (function,
(DMGL_VERBOSE | DMGL_ANSI
| DMGL_GNU_V3 | DMGL_PARAMS));
if (str != NULL)
{
alc = str;
function = str;
}
for (size_t i = 0; i < ARRAY_SIZE (bt_stop); ++i)
{
size_t len = strlen (bt_stop[i]);
if (strncmp (function, bt_stop[i], len) == 0
&& (function[len] == '\0' || function[len] == '('))
{
if (alc != NULL)
free (alc);
return 1;
}
}
}
fprintf (stderr, "0x%lx %s\n\t%s:%d\n",
(unsigned long) pc,
function == NULL ? "???" : function,
filename == NULL ? "???" : filename,
lineno);
if (alc != NULL)
free (alc);
return 0;
}
static void
bt_err_callback (void *data ATTRIBUTE_UNUSED, const char *msg, int errnum)
{
if (errnum < 0)
{
return;
}
fprintf (stderr, "%s%s%s\n", msg, errnum == 0 ? "" : ": ",
errnum == 0 ? "" : xstrerror (errnum));
}
void
diagnostic_check_max_errors (diagnostic_context *context, bool flush)
{
if (!context->max_errors)
return;
int count = (diagnostic_kind_count (context, DK_ERROR)
+ diagnostic_kind_count (context, DK_SORRY)
+ diagnostic_kind_count (context, DK_WERROR));
if (count >= context->max_errors)
{
fnotice (stderr,
"compilation terminated due to -fmax-errors=%u.\n",
context->max_errors);
if (flush)
diagnostic_finish (context);
exit (FATAL_EXIT_CODE);
}
}
void
diagnostic_action_after_output (diagnostic_context *context,
diagnostic_t diag_kind)
{
switch (diag_kind)
{
case DK_DEBUG:
case DK_NOTE:
case DK_ANACHRONISM:
case DK_WARNING:
break;
case DK_ERROR:
case DK_SORRY:
if (context->abort_on_error)
real_abort ();
if (context->fatal_errors)
{
fnotice (stderr, "compilation terminated due to -Wfatal-errors.\n");
diagnostic_finish (context);
exit (FATAL_EXIT_CODE);
}
break;
case DK_ICE:
case DK_ICE_NOBT:
{
struct backtrace_state *state = NULL;
if (diag_kind == DK_ICE)
state = backtrace_create_state (NULL, 0, bt_err_callback, NULL);
int count = 0;
if (state != NULL)
backtrace_full (state, 2, bt_callback, bt_err_callback,
(void *) &count);
if (context->abort_on_error)
real_abort ();
fnotice (stderr, "Please submit a full bug report,\n"
"with preprocessed source if appropriate.\n");
if (count > 0)
fnotice (stderr,
("Please include the complete backtrace "
"with any bug report.\n"));
fnotice (stderr, "See %s for instructions.\n", bug_report_url);
exit (ICE_EXIT_CODE);
}
case DK_FATAL:
if (context->abort_on_error)
real_abort ();
diagnostic_finish (context);
fnotice (stderr, "compilation terminated.\n");
exit (FATAL_EXIT_CODE);
default:
gcc_unreachable ();
}
}
static bool
last_module_changed_p (diagnostic_context *context,
const line_map_ordinary *map)
{
return context->last_module != map;
}
static void
set_last_module (diagnostic_context *context, const line_map_ordinary *map)
{
context->last_module = map;
}
void
diagnostic_report_current_module (diagnostic_context *context, location_t where)
{
const line_map_ordinary *map = NULL;
if (pp_needs_newline (context->printer))
{
pp_newline (context->printer);
pp_needs_newline (context->printer) = false;
}
if (where <= BUILTINS_LOCATION)
return;
linemap_resolve_location (line_table, where,
LRK_MACRO_DEFINITION_LOCATION,
&map);
if (map && last_module_changed_p (context, map))
{
set_last_module (context, map);
if (! MAIN_FILE_P (map))
{
map = INCLUDED_FROM (line_table, map);
const char *line_col
= maybe_line_and_column (LAST_SOURCE_LINE (map),
context->show_column
? LAST_SOURCE_COLUMN (map) : 0);
pp_verbatim (context->printer,
"In file included from %r%s%s%R", "locus",
LINEMAP_FILE (map), line_col);
while (! MAIN_FILE_P (map))
{
map = INCLUDED_FROM (line_table, map);
line_col = maybe_line_and_column (LAST_SOURCE_LINE (map), 0);
pp_verbatim (context->printer,
",\n                 from %r%s%s%R", "locus",
LINEMAP_FILE (map), line_col);
}
pp_verbatim (context->printer, ":");
pp_newline (context->printer);
}
}
}
void
default_diagnostic_starter (diagnostic_context *context,
diagnostic_info *diagnostic)
{
diagnostic_report_current_module (context, diagnostic_location (diagnostic));
pp_set_prefix (context->printer, diagnostic_build_prefix (context,
diagnostic));
}
void
default_diagnostic_start_span_fn (diagnostic_context *context,
expanded_location exploc)
{
pp_set_prefix (context->printer,
diagnostic_get_location_text (context, exploc));
pp_string (context->printer, "");
pp_newline (context->printer);
}
void
default_diagnostic_finalizer (diagnostic_context *context,
diagnostic_info *diagnostic)
{
diagnostic_show_locus (context, diagnostic->richloc, diagnostic->kind);
pp_destroy_prefix (context->printer);
pp_flush (context->printer);
}
diagnostic_t
diagnostic_classify_diagnostic (diagnostic_context *context,
int option_index,
diagnostic_t new_kind,
location_t where)
{
diagnostic_t old_kind;
if (option_index < 0
|| option_index >= context->n_opts
|| new_kind >= DK_LAST_DIAGNOSTIC_KIND)
return DK_UNSPECIFIED;
old_kind = context->classify_diagnostic[option_index];
if (where != UNKNOWN_LOCATION)
{
int i;
if (old_kind == DK_UNSPECIFIED)
{
old_kind = !context->option_enabled (option_index,
context->option_state)
? DK_IGNORED : (context->warning_as_error_requested
? DK_ERROR : DK_WARNING);
context->classify_diagnostic[option_index] = old_kind;
}
for (i = context->n_classification_history - 1; i >= 0; i --)
if (context->classification_history[i].option == option_index)
{
old_kind = context->classification_history[i].kind;
break;
}
i = context->n_classification_history;
context->classification_history =
(diagnostic_classification_change_t *) xrealloc (context->classification_history, (i + 1)
* sizeof (diagnostic_classification_change_t));
context->classification_history[i].location = where;
context->classification_history[i].option = option_index;
context->classification_history[i].kind = new_kind;
context->n_classification_history ++;
}
else
context->classify_diagnostic[option_index] = new_kind;
return old_kind;
}
void
diagnostic_push_diagnostics (diagnostic_context *context, location_t where ATTRIBUTE_UNUSED)
{
context->push_list = (int *) xrealloc (context->push_list, (context->n_push + 1) * sizeof (int));
context->push_list[context->n_push ++] = context->n_classification_history;
}
void
diagnostic_pop_diagnostics (diagnostic_context *context, location_t where)
{
int jump_to;
int i;
if (context->n_push)
jump_to = context->push_list [-- context->n_push];
else
jump_to = 0;
i = context->n_classification_history;
context->classification_history =
(diagnostic_classification_change_t *) xrealloc (context->classification_history, (i + 1)
* sizeof (diagnostic_classification_change_t));
context->classification_history[i].location = where;
context->classification_history[i].option = jump_to;
context->classification_history[i].kind = DK_POP;
context->n_classification_history ++;
}
static void
print_escaped_string (pretty_printer *pp, const char *text)
{
gcc_assert (pp);
gcc_assert (text);
pp_character (pp, '"');
for (const char *ch = text; *ch; ch++)
{
switch (*ch)
{
case '\\':
pp_string (pp, "\\\\");
break;
case '\t':
pp_string (pp, "\\t");
break;
case '\n':
pp_string (pp, "\\n");
break;
case '"':
pp_string (pp, "\\\"");
break;
default:
if (ISPRINT (*ch))
pp_character (pp, *ch);
else
{
unsigned char c = (*ch & 0xff);
pp_printf (pp, "\\%o%o%o", (c / 64), (c / 8) & 007, c & 007);
}
break;
}
}
pp_character (pp, '"');
}
static void
print_parseable_fixits (pretty_printer *pp, rich_location *richloc)
{
gcc_assert (pp);
gcc_assert (richloc);
for (unsigned i = 0; i < richloc->get_num_fixit_hints (); i++)
{
const fixit_hint *hint = richloc->get_fixit_hint (i);
source_location start_loc = hint->get_start_loc ();
expanded_location start_exploc = expand_location (start_loc);
pp_string (pp, "fix-it:");
print_escaped_string (pp, start_exploc.file);
source_location next_loc = hint->get_next_loc ();
expanded_location next_exploc = expand_location (next_loc);
pp_printf (pp, ":{%i:%i-%i:%i}:",
start_exploc.line, start_exploc.column,
next_exploc.line, next_exploc.column);
print_escaped_string (pp, hint->get_string ());
pp_newline (pp);
}
}
static diagnostic_t
update_effective_level_from_pragmas (diagnostic_context *context,
diagnostic_info *diagnostic)
{
diagnostic_t diag_class = DK_UNSPECIFIED;
if (context->n_classification_history > 0)
{
location_t location = diagnostic_location (diagnostic);
for (int i = context->n_classification_history - 1; i >= 0; i --)
{
if (linemap_location_before_p
(line_table,
context->classification_history[i].location,
location))
{
if (context->classification_history[i].kind == (int) DK_POP)
{
i = context->classification_history[i].option;
continue;
}
int option = context->classification_history[i].option;
if (option == 0 || option == diagnostic->option_index)
{
diag_class = context->classification_history[i].kind;
if (diag_class != DK_UNSPECIFIED)
diagnostic->kind = diag_class;
break;
}
}
}
}
return diag_class;
}
static void
print_option_information (diagnostic_context *context,
const diagnostic_info *diagnostic,
diagnostic_t orig_diag_kind)
{
char *option_text;
option_text = context->option_name (context, diagnostic->option_index,
orig_diag_kind, diagnostic->kind);
if (option_text)
{
pretty_printer *pp = context->printer;
pp_string (pp, " [");
pp_string (pp, colorize_start (pp_show_color (pp),
diagnostic_kind_color[diagnostic->kind]));
pp_string (pp, option_text);
pp_string (pp, colorize_stop (pp_show_color (pp)));
pp_character (pp, ']');
free (option_text);
}
}
bool
diagnostic_report_diagnostic (diagnostic_context *context,
diagnostic_info *diagnostic)
{
location_t location = diagnostic_location (diagnostic);
diagnostic_t orig_diag_kind = diagnostic->kind;
if ((diagnostic->kind == DK_WARNING || diagnostic->kind == DK_PEDWARN)
&& !diagnostic_report_warnings_p (context, location))
return false;
if (diagnostic->kind == DK_PEDWARN)
{
diagnostic->kind = pedantic_warning_kind (context);
orig_diag_kind = diagnostic->kind;
}
if (diagnostic->kind == DK_NOTE && context->inhibit_notes_p)
return false;
if (context->lock > 0)
{
if ((diagnostic->kind == DK_ICE || diagnostic->kind == DK_ICE_NOBT)
&& context->lock == 1)
pp_newline_and_flush (context->printer);
else
error_recursion (context);
}
if (context->warning_as_error_requested
&& diagnostic->kind == DK_WARNING)
diagnostic->kind = DK_ERROR;
if (diagnostic->option_index
&& diagnostic->option_index != permissive_error_option (context))
{
if (! context->option_enabled (diagnostic->option_index,
context->option_state))
return false;
diagnostic_t diag_class
= update_effective_level_from_pragmas (context, diagnostic);
if (diag_class == DK_UNSPECIFIED
&& (context->classify_diagnostic[diagnostic->option_index]
!= DK_UNSPECIFIED))
diagnostic->kind
= context->classify_diagnostic[diagnostic->option_index];
if (diagnostic->kind == DK_IGNORED)
return false;
}
if (diagnostic->kind != DK_NOTE)
diagnostic_check_max_errors (context);
context->lock++;
if (diagnostic->kind == DK_ICE || diagnostic->kind == DK_ICE_NOBT)
{
if (!CHECKING_P
&& (diagnostic_kind_count (context, DK_ERROR) > 0
|| diagnostic_kind_count (context, DK_SORRY) > 0)
&& !context->abort_on_error)
{
expanded_location s 
= expand_location (diagnostic_location (diagnostic));
fnotice (stderr, "%s:%d: confused by earlier errors, bailing out\n",
s.file, s.line);
exit (ICE_EXIT_CODE);
}
if (context->internal_error)
(*context->internal_error) (context,
diagnostic->message.format_spec,
diagnostic->message.args_ptr);
}
if (diagnostic->kind == DK_ERROR && orig_diag_kind == DK_WARNING)
++diagnostic_kind_count (context, DK_WERROR);
else
++diagnostic_kind_count (context, diagnostic->kind);
diagnostic->message.x_data = &diagnostic->x_data;
diagnostic->x_data = NULL;
pp_format (context->printer, &diagnostic->message);
(*diagnostic_starter (context)) (context, diagnostic);
pp_output_formatted_text (context->printer);
if (context->show_option_requested)
print_option_information (context, diagnostic, orig_diag_kind);
(*diagnostic_finalizer (context)) (context, diagnostic);
if (context->parseable_fixits_p)
{
print_parseable_fixits (context->printer, diagnostic->richloc);
pp_flush (context->printer);
}
diagnostic_action_after_output (context, diagnostic->kind);
diagnostic->x_data = NULL;
if (context->edit_context_ptr)
if (diagnostic->richloc->fixits_can_be_auto_applied_p ())
context->edit_context_ptr->add_fixits (diagnostic->richloc);
context->lock--;
return true;
}
const char *
trim_filename (const char *name)
{
static const char this_file[] = __FILE__;
const char *p = name, *q = this_file;
while (p[0] == '.' && p[1] == '.' && IS_DIR_SEPARATOR (p[2]))
p += 3;
while (q[0] == '.' && q[1] == '.' && IS_DIR_SEPARATOR (q[2]))
q += 3;
while (*p == *q && *p != 0 && *q != 0)
p++, q++;
while (p > name && !IS_DIR_SEPARATOR (p[-1]))
p--;
return p;
}

void
verbatim (const char *gmsgid, ...)
{
text_info text;
va_list ap;
va_start (ap, gmsgid);
text.err_no = errno;
text.args_ptr = &ap;
text.format_spec = _(gmsgid);
text.x_data = NULL;
pp_format_verbatim (global_dc->printer, &text);
pp_newline_and_flush (global_dc->printer);
va_end (ap);
}
void
diagnostic_append_note (diagnostic_context *context,
location_t location,
const char * gmsgid, ...)
{
diagnostic_info diagnostic;
va_list ap;
rich_location richloc (line_table, location);
va_start (ap, gmsgid);
diagnostic_set_info (&diagnostic, gmsgid, &ap, &richloc, DK_NOTE);
if (context->inhibit_notes_p)
{
va_end (ap);
return;
}
char *saved_prefix = pp_take_prefix (context->printer);
pp_set_prefix (context->printer,
diagnostic_build_prefix (context, &diagnostic));
pp_format (context->printer, &diagnostic.message);
pp_output_formatted_text (context->printer);
pp_destroy_prefix (context->printer);
pp_set_prefix (context->printer, saved_prefix);
diagnostic_show_locus (context, &richloc, DK_NOTE);
va_end (ap);
}
static bool
diagnostic_impl (rich_location *richloc, int opt,
const char *gmsgid,
va_list *ap, diagnostic_t kind)
{
diagnostic_info diagnostic;
if (kind == DK_PERMERROR)
{
diagnostic_set_info (&diagnostic, gmsgid, ap, richloc,
permissive_error_kind (global_dc));
diagnostic.option_index = permissive_error_option (global_dc);
}
else
{
diagnostic_set_info (&diagnostic, gmsgid, ap, richloc, kind);
if (kind == DK_WARNING || kind == DK_PEDWARN)
diagnostic.option_index = opt;
}
return diagnostic_report_diagnostic (global_dc, &diagnostic);
}
static bool
diagnostic_n_impl (rich_location *richloc, int opt, unsigned HOST_WIDE_INT n,
const char *singular_gmsgid,
const char *plural_gmsgid,
va_list *ap, diagnostic_t kind)
{
diagnostic_info diagnostic;
unsigned long gtn;
if (sizeof n <= sizeof gtn)
gtn = n;
else
gtn = n <= ULONG_MAX ? n : n % 1000000LU + 1000000LU;
const char *text = ngettext (singular_gmsgid, plural_gmsgid, gtn);
diagnostic_set_info_translated (&diagnostic, text, ap, richloc, kind);
if (kind == DK_WARNING)
diagnostic.option_index = opt;
return diagnostic_report_diagnostic (global_dc, &diagnostic);
}
bool
emit_diagnostic (diagnostic_t kind, location_t location, int opt,
const char *gmsgid, ...)
{
va_list ap;
va_start (ap, gmsgid);
rich_location richloc (line_table, location);
bool ret = diagnostic_impl (&richloc, opt, gmsgid, &ap, kind);
va_end (ap);
return ret;
}
bool
emit_diagnostic_valist (diagnostic_t kind, location_t location, int opt,
const char *gmsgid, va_list *ap)
{
rich_location richloc (line_table, location);
return diagnostic_impl (&richloc, opt, gmsgid, ap, kind);
}
void
inform (location_t location, const char *gmsgid, ...)
{
va_list ap;
va_start (ap, gmsgid);
rich_location richloc (line_table, location);
diagnostic_impl (&richloc, -1, gmsgid, &ap, DK_NOTE);
va_end (ap);
}
void
inform (rich_location *richloc, const char *gmsgid, ...)
{
gcc_assert (richloc);
va_list ap;
va_start (ap, gmsgid);
diagnostic_impl (richloc, -1, gmsgid, &ap, DK_NOTE);
va_end (ap);
}
void
inform_n (location_t location, unsigned HOST_WIDE_INT n,
const char *singular_gmsgid, const char *plural_gmsgid, ...)
{
va_list ap;
va_start (ap, plural_gmsgid);
rich_location richloc (line_table, location);
diagnostic_n_impl (&richloc, -1, n, singular_gmsgid, plural_gmsgid,
&ap, DK_NOTE);
va_end (ap);
}
bool
warning (int opt, const char *gmsgid, ...)
{
va_list ap;
va_start (ap, gmsgid);
rich_location richloc (line_table, input_location);
bool ret = diagnostic_impl (&richloc, opt, gmsgid, &ap, DK_WARNING);
va_end (ap);
return ret;
}
bool
warning_at (location_t location, int opt, const char *gmsgid, ...)
{
va_list ap;
va_start (ap, gmsgid);
rich_location richloc (line_table, location);
bool ret = diagnostic_impl (&richloc, opt, gmsgid, &ap, DK_WARNING);
va_end (ap);
return ret;
}
bool
warning_at (rich_location *richloc, int opt, const char *gmsgid, ...)
{
gcc_assert (richloc);
va_list ap;
va_start (ap, gmsgid);
bool ret = diagnostic_impl (richloc, opt, gmsgid, &ap, DK_WARNING);
va_end (ap);
return ret;
}
bool
warning_n (rich_location *richloc, int opt, unsigned HOST_WIDE_INT n,
const char *singular_gmsgid, const char *plural_gmsgid, ...)
{
gcc_assert (richloc);
va_list ap;
va_start (ap, plural_gmsgid);
bool ret = diagnostic_n_impl (richloc, opt, n,
singular_gmsgid, plural_gmsgid,
&ap, DK_WARNING);
va_end (ap);
return ret;
}
bool
warning_n (location_t location, int opt, unsigned HOST_WIDE_INT n,
const char *singular_gmsgid, const char *plural_gmsgid, ...)
{
va_list ap;
va_start (ap, plural_gmsgid);
rich_location richloc (line_table, location);
bool ret = diagnostic_n_impl (&richloc, opt, n,
singular_gmsgid, plural_gmsgid,
&ap, DK_WARNING);
va_end (ap);
return ret;
}
bool
pedwarn (location_t location, int opt, const char *gmsgid, ...)
{
va_list ap;
va_start (ap, gmsgid);
rich_location richloc (line_table, location);
bool ret = diagnostic_impl (&richloc, opt, gmsgid, &ap, DK_PEDWARN);
va_end (ap);
return ret;
}
bool
pedwarn (rich_location *richloc, int opt, const char *gmsgid, ...)
{
gcc_assert (richloc);
va_list ap;
va_start (ap, gmsgid);
bool ret = diagnostic_impl (richloc, opt, gmsgid, &ap, DK_PEDWARN);
va_end (ap);
return ret;
}
bool
permerror (location_t location, const char *gmsgid, ...)
{
va_list ap;
va_start (ap, gmsgid);
rich_location richloc (line_table, location);
bool ret = diagnostic_impl (&richloc, -1, gmsgid, &ap, DK_PERMERROR);
va_end (ap);
return ret;
}
bool
permerror (rich_location *richloc, const char *gmsgid, ...)
{
gcc_assert (richloc);
va_list ap;
va_start (ap, gmsgid);
bool ret = diagnostic_impl (richloc, -1, gmsgid, &ap, DK_PERMERROR);
va_end (ap);
return ret;
}
void
error (const char *gmsgid, ...)
{
va_list ap;
va_start (ap, gmsgid);
rich_location richloc (line_table, input_location);
diagnostic_impl (&richloc, -1, gmsgid, &ap, DK_ERROR);
va_end (ap);
}
void
error_n (location_t location, unsigned HOST_WIDE_INT n,
const char *singular_gmsgid, const char *plural_gmsgid, ...)
{
va_list ap;
va_start (ap, plural_gmsgid);
rich_location richloc (line_table, location);
diagnostic_n_impl (&richloc, -1, n, singular_gmsgid, plural_gmsgid,
&ap, DK_ERROR);
va_end (ap);
}
void
error_at (location_t loc, const char *gmsgid, ...)
{
va_list ap;
va_start (ap, gmsgid);
rich_location richloc (line_table, loc);
diagnostic_impl (&richloc, -1, gmsgid, &ap, DK_ERROR);
va_end (ap);
}
void
error_at (rich_location *richloc, const char *gmsgid, ...)
{
gcc_assert (richloc);
va_list ap;
va_start (ap, gmsgid);
diagnostic_impl (richloc, -1, gmsgid, &ap, DK_ERROR);
va_end (ap);
}
void
sorry (const char *gmsgid, ...)
{
va_list ap;
va_start (ap, gmsgid);
rich_location richloc (line_table, input_location);
diagnostic_impl (&richloc, -1, gmsgid, &ap, DK_SORRY);
va_end (ap);
}
bool
seen_error (void)
{
return errorcount || sorrycount;
}
void
fatal_error (location_t loc, const char *gmsgid, ...)
{
va_list ap;
va_start (ap, gmsgid);
rich_location richloc (line_table, loc);
diagnostic_impl (&richloc, -1, gmsgid, &ap, DK_FATAL);
va_end (ap);
gcc_unreachable ();
}
void
internal_error (const char *gmsgid, ...)
{
va_list ap;
va_start (ap, gmsgid);
rich_location richloc (line_table, input_location);
diagnostic_impl (&richloc, -1, gmsgid, &ap, DK_ICE);
va_end (ap);
gcc_unreachable ();
}
void
internal_error_no_backtrace (const char *gmsgid, ...)
{
va_list ap;
va_start (ap, gmsgid);
rich_location richloc (line_table, input_location);
diagnostic_impl (&richloc, -1, gmsgid, &ap, DK_ICE_NOBT);
va_end (ap);
gcc_unreachable ();
}

void
fnotice (FILE *file, const char *cmsgid, ...)
{
va_list ap;
va_start (ap, cmsgid);
vfprintf (file, _(cmsgid), ap);
va_end (ap);
}
static void
error_recursion (diagnostic_context *context)
{
if (context->lock < 3)
pp_newline_and_flush (context->printer);
fnotice (stderr,
"Internal compiler error: Error reporting routines re-entered.\n");
diagnostic_action_after_output (context, DK_ICE);
real_abort ();
}
void
fancy_abort (const char *file, int line, const char *function)
{
internal_error ("in %s, at %s:%d", function, trim_filename (file), line);
}
#undef abort
static void
real_abort (void)
{
abort ();
}
#if CHECKING_P
namespace selftest {
static void
assert_print_escaped_string (const location &loc, const char *expected_output,
const char *input)
{
pretty_printer pp;
print_escaped_string (&pp, input);
ASSERT_STREQ_AT (loc, expected_output, pp_formatted_text (&pp));
}
#define ASSERT_PRINT_ESCAPED_STRING_STREQ(EXPECTED_OUTPUT, INPUT) \
assert_print_escaped_string (SELFTEST_LOCATION, EXPECTED_OUTPUT, INPUT)
static void
test_print_escaped_string ()
{
ASSERT_PRINT_ESCAPED_STRING_STREQ ("\"\"", "");
ASSERT_PRINT_ESCAPED_STRING_STREQ ("\"hello world\"", "hello world");
ASSERT_PRINT_ESCAPED_STRING_STREQ ("\"before\\\\after\"",
"before\\after");
ASSERT_PRINT_ESCAPED_STRING_STREQ ("\"before\\tafter\"",
"before\tafter");
ASSERT_PRINT_ESCAPED_STRING_STREQ ("\"before\\nafter\"",
"before\nafter");
ASSERT_PRINT_ESCAPED_STRING_STREQ ("\"before\\\"after\"",
"before\"after");
ASSERT_PRINT_ESCAPED_STRING_STREQ ("\"before\\007after\"",
"before\aafter");
ASSERT_PRINT_ESCAPED_STRING_STREQ ("\"before\\013after\"",
"before\vafter");
}
static void
test_print_parseable_fixits_none ()
{
pretty_printer pp;
rich_location richloc (line_table, UNKNOWN_LOCATION);
print_parseable_fixits (&pp, &richloc);
ASSERT_STREQ ("", pp_formatted_text (&pp));
}
static void
test_print_parseable_fixits_insert ()
{
pretty_printer pp;
rich_location richloc (line_table, UNKNOWN_LOCATION);
linemap_add (line_table, LC_ENTER, false, "test.c", 0);
linemap_line_start (line_table, 5, 100);
linemap_add (line_table, LC_LEAVE, false, NULL, 0);
location_t where = linemap_position_for_column (line_table, 10);
richloc.add_fixit_insert_before (where, "added content");
print_parseable_fixits (&pp, &richloc);
ASSERT_STREQ ("fix-it:\"test.c\":{5:10-5:10}:\"added content\"\n",
pp_formatted_text (&pp));
}
static void
test_print_parseable_fixits_remove ()
{
pretty_printer pp;
rich_location richloc (line_table, UNKNOWN_LOCATION);
linemap_add (line_table, LC_ENTER, false, "test.c", 0);
linemap_line_start (line_table, 5, 100);
linemap_add (line_table, LC_LEAVE, false, NULL, 0);
source_range where;
where.m_start = linemap_position_for_column (line_table, 10);
where.m_finish = linemap_position_for_column (line_table, 20);
richloc.add_fixit_remove (where);
print_parseable_fixits (&pp, &richloc);
ASSERT_STREQ ("fix-it:\"test.c\":{5:10-5:21}:\"\"\n",
pp_formatted_text (&pp));
}
static void
test_print_parseable_fixits_replace ()
{
pretty_printer pp;
rich_location richloc (line_table, UNKNOWN_LOCATION);
linemap_add (line_table, LC_ENTER, false, "test.c", 0);
linemap_line_start (line_table, 5, 100);
linemap_add (line_table, LC_LEAVE, false, NULL, 0);
source_range where;
where.m_start = linemap_position_for_column (line_table, 10);
where.m_finish = linemap_position_for_column (line_table, 20);
richloc.add_fixit_replace (where, "replacement");
print_parseable_fixits (&pp, &richloc);
ASSERT_STREQ ("fix-it:\"test.c\":{5:10-5:21}:\"replacement\"\n",
pp_formatted_text (&pp));
}
static void
assert_location_text (const char *expected_loc_text,
const char *filename, int line, int column,
bool show_column)
{
test_diagnostic_context dc;
dc.show_column = show_column;
expanded_location xloc;
xloc.file = filename;
xloc.line = line;
xloc.column = column;
xloc.data = NULL;
xloc.sysp = false;
char *actual_loc_text = diagnostic_get_location_text (&dc, xloc);
ASSERT_STREQ (expected_loc_text, actual_loc_text);
free (actual_loc_text);
}
static void
test_diagnostic_get_location_text ()
{
const char *old_progname = progname;
progname = "PROGNAME";
assert_location_text ("PROGNAME:", NULL, 0, 0, true);
assert_location_text ("<built-in>:", "<built-in>", 42, 10, true);
assert_location_text ("foo.c:42:10:", "foo.c", 42, 10, true);
assert_location_text ("foo.c:42:", "foo.c", 42, 0, true);
assert_location_text ("foo.c:", "foo.c", 0, 10, true);
assert_location_text ("foo.c:42:", "foo.c", 42, 10, false);
assert_location_text ("foo.c:", "foo.c", 0, 10, false);
maybe_line_and_column (INT_MAX, INT_MAX);
maybe_line_and_column (INT_MIN, INT_MIN);
progname = old_progname;
}
void
diagnostic_c_tests ()
{
test_print_escaped_string ();
test_print_parseable_fixits_none ();
test_print_parseable_fixits_insert ();
test_print_parseable_fixits_remove ();
test_print_parseable_fixits_replace ();
test_diagnostic_get_location_text ();
}
} 
#endif 
