#ifndef GCC_DIAGNOSTIC_H
#define GCC_DIAGNOSTIC_H
#include "pretty-print.h"
#include "diagnostic-core.h"
struct diagnostic_info
{
text_info message;
rich_location *richloc;
void *x_data;
diagnostic_t kind;
int option_index;
};
struct diagnostic_classification_change_t
{
location_t location;
int option;
diagnostic_t kind;
};
typedef void (*diagnostic_starter_fn) (diagnostic_context *,
diagnostic_info *);
typedef void (*diagnostic_start_span_fn) (diagnostic_context *,
expanded_location);
typedef diagnostic_starter_fn diagnostic_finalizer_fn;
class edit_context;
struct diagnostic_context
{
pretty_printer *printer;
int diagnostic_count[DK_LAST_DIAGNOSTIC_KIND];
bool warning_as_error_requested;
int n_opts;
diagnostic_t *classify_diagnostic;
diagnostic_classification_change_t *classification_history;
int n_classification_history;
int *push_list;
int n_push;
bool show_caret;
int caret_max_width;
char caret_chars[rich_location::STATICALLY_ALLOCATED_RANGES];
bool show_option_requested;
bool abort_on_error;
bool show_column;
bool pedantic_errors;
bool permissive;
int opt_permissive;
bool fatal_errors;
bool dc_inhibit_warnings;
bool dc_warn_system_headers;
int max_errors;
diagnostic_starter_fn begin_diagnostic;
diagnostic_start_span_fn start_span;
diagnostic_finalizer_fn end_diagnostic;
void (*internal_error) (diagnostic_context *, const char *, va_list *);
int (*option_enabled) (int, void *);
void *option_state;
char *(*option_name) (diagnostic_context *, int, diagnostic_t, diagnostic_t);
void *x_data;
location_t last_location;
const line_map_ordinary *last_module;
int lock;
bool inhibit_notes_p;
bool colorize_source_p;
bool show_ruler_p;
bool parseable_fixits_p;
edit_context *edit_context_ptr;
};
static inline void
diagnostic_inhibit_notes (diagnostic_context * context)
{
context->inhibit_notes_p = true;
}
#define diagnostic_starter(DC) (DC)->begin_diagnostic
#define diagnostic_finalizer(DC) (DC)->end_diagnostic
#define diagnostic_context_auxiliary_data(DC) (DC)->x_data
#define diagnostic_info_auxiliary_data(DI) (DI)->x_data
#define diagnostic_format_decoder(DC) ((DC)->printer->format_decoder)
#define diagnostic_prefixing_rule(DC) ((DC)->printer->wrapping.rule)
#define diagnostic_abort_on_error(DC) \
(DC)->abort_on_error = true
extern diagnostic_context *global_dc;
#define diagnostic_kind_count(DC, DK) (DC)->diagnostic_count[(int) (DK)]
#define errorcount diagnostic_kind_count (global_dc, DK_ERROR)
#define warningcount diagnostic_kind_count (global_dc, DK_WARNING)
#define werrorcount diagnostic_kind_count (global_dc, DK_WERROR)
#define sorrycount diagnostic_kind_count (global_dc, DK_SORRY)
#define diagnostic_report_warnings_p(DC, LOC)				\
(!(DC)->dc_inhibit_warnings						\
&& !(in_system_header_at (LOC) && !(DC)->dc_warn_system_headers))
static inline void
diagnostic_override_option_index (diagnostic_info *info, int optidx)
{
info->option_index = optidx;
}
extern void diagnostic_initialize (diagnostic_context *, int);
extern void diagnostic_color_init (diagnostic_context *, int value = -1);
extern void diagnostic_finish (diagnostic_context *);
extern void diagnostic_report_current_module (diagnostic_context *, location_t);
extern void diagnostic_show_locus (diagnostic_context *,
rich_location *richloc,
diagnostic_t diagnostic_kind);
extern diagnostic_t diagnostic_classify_diagnostic (diagnostic_context *,
int ,
diagnostic_t ,
location_t);
extern void diagnostic_push_diagnostics (diagnostic_context *, location_t);
extern void diagnostic_pop_diagnostics (diagnostic_context *, location_t);
extern bool diagnostic_report_diagnostic (diagnostic_context *,
diagnostic_info *);
#ifdef ATTRIBUTE_GCC_DIAG
extern void diagnostic_set_info (diagnostic_info *, const char *, va_list *,
rich_location *, diagnostic_t) ATTRIBUTE_GCC_DIAG(2,0);
extern void diagnostic_set_info_translated (diagnostic_info *, const char *,
va_list *, rich_location *,
diagnostic_t)
ATTRIBUTE_GCC_DIAG(2,0);
extern void diagnostic_append_note (diagnostic_context *, location_t,
const char *, ...) ATTRIBUTE_GCC_DIAG(3,4);
#endif
extern char *diagnostic_build_prefix (diagnostic_context *, const diagnostic_info *);
void default_diagnostic_starter (diagnostic_context *, diagnostic_info *);
void default_diagnostic_start_span_fn (diagnostic_context *,
expanded_location);
void default_diagnostic_finalizer (diagnostic_context *, diagnostic_info *);
void diagnostic_set_caret_max_width (diagnostic_context *context, int value);
void diagnostic_action_after_output (diagnostic_context *, diagnostic_t);
void diagnostic_check_max_errors (diagnostic_context *, bool flush = false);
void diagnostic_file_cache_fini (void);
int get_terminal_width (void);
static inline location_t
diagnostic_location (const diagnostic_info * diagnostic, int which = 0)
{
return diagnostic->message.get_location (which);
}
static inline unsigned int
diagnostic_num_locations (const diagnostic_info * diagnostic)
{
return diagnostic->message.m_richloc->get_num_locations ();
}
static inline expanded_location
diagnostic_expand_location (const diagnostic_info * diagnostic, int which = 0)
{
return diagnostic->richloc->get_expanded_location (which);
}
const int CARET_LINE_MARGIN = 10;
static inline bool
diagnostic_same_line (const diagnostic_context *context,
expanded_location s1, expanded_location s2)
{
return s2.column && s1.line == s2.line 
&& context->caret_max_width - CARET_LINE_MARGIN > abs (s1.column - s2.column);
}
extern const char *diagnostic_get_color_for_kind (diagnostic_t kind);
extern char *file_name_as_prefix (diagnostic_context *, const char *);
extern char *build_message_string (const char *, ...) ATTRIBUTE_PRINTF_1;
#endif 
