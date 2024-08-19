#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "target.h"
#include "rtl.h"
#include "tree.h"
#include "gimple.h"
#include "alloc-pool.h"
#include "timevar.h"
#include "memmodel.h"
#include "tm_p.h"
#include "optabs-libfuncs.h"
#include "insn-config.h"
#include "ira.h"
#include "recog.h"
#include "cgraph.h"
#include "coverage.h"
#include "diagnostic.h"
#include "varasm.h"
#include "tree-inline.h"
#include "realmpfr.h"	
#include "version.h"
#include "flags.h"
#include "insn-attr.h"
#include "output.h"
#include "toplev.h"
#include "expr.h"
#include "intl.h"
#include "tree-diagnostic.h"
#include "params.h"
#include "reload.h"
#include "lra.h"
#include "dwarf2asm.h"
#include "debug.h"
#include "common/common-target.h"
#include "langhooks.h"
#include "cfgloop.h" 
#include "hosthooks.h"
#include "opts.h"
#include "opts-diagnostic.h"
#include "stringpool.h"
#include "attribs.h"
#include "asan.h"
#include "tsan.h"
#include "plugin.h"
#include "context.h"
#include "pass_manager.h"
#include "auto-profile.h"
#include "dwarf2out.h"
#include "ipa-reference.h"
#include "symbol-summary.h"
#include "tree-vrp.h"
#include "ipa-prop.h"
#include "gcse.h"
#include "tree-chkp.h"
#include "omp-offload.h"
#include "hsa-common.h"
#include "edit-context.h"
#include "tree-pass.h"
#include "dumpfile.h"
#include "ipa-fnsummary.h"
#if defined(DBX_DEBUGGING_INFO) || defined(XCOFF_DEBUGGING_INFO)
#include "dbxout.h"
#endif
#ifdef XCOFF_DEBUGGING_INFO
#include "xcoffout.h"		
#endif
#include "selftest.h"
#ifdef HAVE_isl
#include <isl/version.h>
#endif
static void general_init (const char *, bool);
static void do_compile ();
static void process_options (void);
static void backend_init (void);
static int lang_dependent_init (const char *);
static void init_asm_output (const char *);
static void finalize (bool);
static void crash_signal (int) ATTRIBUTE_NORETURN;
static void compile_file (void);
static bool no_backend;
#define MAX_LINE 75
struct cl_decoded_option *save_decoded_options;
unsigned int save_decoded_options_count;
#define AUTODETECT_VALUE 2
const struct gcc_debug_hooks *debug_hooks;
tree current_function_decl;
const char * current_function_func_begin_label;
static const char *flag_random_seed;
unsigned local_tick;
HOST_WIDE_INT random_seed;
rtx stack_limit_rtx;
struct target_flag_state default_target_flag_state;
#if SWITCHABLE_TARGET
struct target_flag_state *this_target_flag_state = &default_target_flag_state;
#else
#define this_target_flag_state (&default_target_flag_state)
#endif
const char *user_label_prefix;
FILE *asm_out_file;
FILE *aux_info_file;
FILE *stack_usage_file = NULL;
static const char *src_pwd;
bool
set_src_pwd (const char *pwd)
{
if (src_pwd)
{
if (strcmp (src_pwd, pwd) == 0)
return true;
else
return false;
}
src_pwd = xstrdup (pwd);
return true;
}
const char *
get_src_pwd (void)
{
if (! src_pwd)
{
src_pwd = getpwd ();
if (!src_pwd)
src_pwd = ".";
}
return src_pwd;
}
void
announce_function (tree decl)
{
if (!quiet_flag)
{
if (rtl_dump_and_exit)
fprintf (stderr, "%s ",
identifier_to_locale (IDENTIFIER_POINTER (DECL_NAME (decl))));
else
fprintf (stderr, " %s",
identifier_to_locale (lang_hooks.decl_printable_name (decl, 2)));
fflush (stderr);
pp_needs_newline (global_dc->printer) = true;
diagnostic_set_last_function (global_dc, (diagnostic_info *) NULL);
}
}
static void
init_local_tick (void)
{
if (!flag_random_seed)
{
#ifdef HAVE_GETTIMEOFDAY
{
struct timeval tv;
gettimeofday (&tv, NULL);
local_tick = (unsigned) tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
#else
{
time_t now = time (NULL);
if (now != (time_t)-1)
local_tick = (unsigned) now;
}
#endif
}
else
local_tick = -1;
}
HOST_WIDE_INT
get_random_seed (bool noinit)
{
if (!random_seed && !noinit)
{
int fd = open ("/dev/urandom", O_RDONLY);
if (fd >= 0)
{
if (read (fd, &random_seed, sizeof (random_seed))
!= sizeof (random_seed))
random_seed = 0;
close (fd);
}
if (!random_seed)
random_seed = local_tick ^ getpid ();
}
return random_seed;
}
void
set_random_seed (const char *val)
{
flag_random_seed = val;
if (flag_random_seed)
{
char *endp;
random_seed = strtoul (flag_random_seed, &endp, 0);
if (!(endp > flag_random_seed && *endp == 0))
random_seed = crc32_string (0, flag_random_seed);
}
}
static void
crash_signal (int signo)
{
signal (signo, SIG_DFL);
if (this_is_asm_operands)
{
output_operand_lossage ("unrecoverable error");
exit (FATAL_EXIT_CODE);
}
internal_error ("%s", strsignal (signo));
}
void
wrapup_global_declaration_1 (tree decl)
{
if (CODE_CONTAINS_STRUCT (TREE_CODE (decl), TS_DECL_WITH_VIS)
&& DECL_DEFER_OUTPUT (decl) != 0)
DECL_DEFER_OUTPUT (decl) = 0;
if (VAR_P (decl) && DECL_SIZE (decl) == 0)
lang_hooks.finish_incomplete_decl (decl);
}
bool
wrapup_global_declaration_2 (tree decl)
{
if (TREE_ASM_WRITTEN (decl) || DECL_EXTERNAL (decl)
|| (VAR_P (decl) && DECL_HAS_VALUE_EXPR_P (decl)))
return false;
if (VAR_P (decl) && TREE_STATIC (decl))
{
varpool_node *node;
bool needed = true;
node = varpool_node::get (decl);
if (!node && flag_ltrans)
needed = false;
else if (node && node->definition)
needed = false;
else if (node && node->alias)
needed = false;
else if (!symtab->global_info_ready
&& (TREE_USED (decl)
|| TREE_USED (DECL_ASSEMBLER_NAME (decl))))
;
else if (node && node->analyzed)
;
else if (DECL_COMDAT (decl))
needed = false;
else if (TREE_READONLY (decl) && !TREE_PUBLIC (decl)
&& (optimize || !flag_keep_static_consts
|| DECL_ARTIFICIAL (decl)))
needed = false;
if (needed)
{
rest_of_decl_compilation (decl, 1, 1);
return true;
}
}
return false;
}
bool
wrapup_global_declarations (tree *vec, int len)
{
bool reconsider, output_something = false;
int i;
for (i = 0; i < len; i++)
wrapup_global_declaration_1 (vec[i]);
do
{
reconsider = false;
for (i = 0; i < len; i++)
reconsider |= wrapup_global_declaration_2 (vec[i]);
if (reconsider)
output_something = true;
}
while (reconsider);
return output_something;
}
static void
compile_file (void)
{
timevar_start (TV_PHASE_PARSING);
timevar_push (TV_PARSE_GLOBAL);
lang_hooks.parse_file ();
timevar_pop (TV_PARSE_GLOBAL);
timevar_stop (TV_PHASE_PARSING);
if (flag_dump_locations)
dump_location_info (stderr);
if (flag_syntax_only || flag_wpa)
return;
maximum_field_alignment = initial_max_fld_align * BITS_PER_UNIT;
ggc_protect_identifiers = false;
if (!in_lto_p)
{
timevar_start (TV_PHASE_OPT_GEN);
symtab->finalize_compilation_unit ();
timevar_stop (TV_PHASE_OPT_GEN);
}
if (lang_hooks.decls.post_compilation_parsing_cleanups)
lang_hooks.decls.post_compilation_parsing_cleanups ();
if (seen_error ())
return;
timevar_start (TV_PHASE_LATE_ASM);
if (in_lto_p || !flag_lto || flag_fat_lto_objects)
{
if (flag_sanitize & SANITIZE_ADDRESS)
asan_finish_file ();
if (flag_sanitize & SANITIZE_THREAD)
tsan_finish_file ();
if (flag_check_pointer_bounds)
chkp_finish_file ();
omp_finish_file ();
hsa_output_brig ();
output_shared_constant_pool ();
output_object_blocks ();
finish_tm_clone_pairs ();
weak_finish ();
targetm.asm_out.code_end ();
timevar_push (TV_SYMOUT);
#if defined DWARF2_DEBUGGING_INFO || defined DWARF2_UNWIND_INFO
dwarf2out_frame_finish ();
#endif
(*debug_hooks->finish) (main_input_filename);
timevar_pop (TV_SYMOUT);
dw2_output_indirect_constants ();
process_pending_assemble_externals ();
}
if (flag_generate_lto || flag_generate_offload)
{
#if defined ASM_OUTPUT_ALIGNED_DECL_COMMON
ASM_OUTPUT_ALIGNED_DECL_COMMON (asm_out_file, NULL_TREE,
"__gnu_lto_v1",
HOST_WIDE_INT_1U, 8);
#elif defined ASM_OUTPUT_ALIGNED_COMMON
ASM_OUTPUT_ALIGNED_COMMON (asm_out_file, "__gnu_lto_v1",
HOST_WIDE_INT_1U, 8);
#else
ASM_OUTPUT_COMMON (asm_out_file, "__gnu_lto_v1",
HOST_WIDE_INT_1U,
HOST_WIDE_INT_1U);
#endif
}
if (flag_generate_lto && !flag_fat_lto_objects)
{
#if defined ASM_OUTPUT_ALIGNED_DECL_COMMON
ASM_OUTPUT_ALIGNED_DECL_COMMON (asm_out_file, NULL_TREE, "__gnu_lto_slim",
HOST_WIDE_INT_1U, 8);
#elif defined ASM_OUTPUT_ALIGNED_COMMON
ASM_OUTPUT_ALIGNED_COMMON (asm_out_file, "__gnu_lto_slim",
HOST_WIDE_INT_1U, 8);
#else
ASM_OUTPUT_COMMON (asm_out_file, "__gnu_lto_slim",
HOST_WIDE_INT_1U,
HOST_WIDE_INT_1U);
#endif
}
if (!flag_no_ident)
{
const char *pkg_version = "(GNU) ";
char *ident_str;
if (strcmp ("(GCC) ", pkgversion_string))
pkg_version = pkgversion_string;
ident_str = ACONCAT (("GCC: ", pkg_version, version_string, NULL));
targetm.asm_out.output_ident (ident_str);
}
if (flag_auto_profile)
end_auto_profile ();
invoke_plugin_callbacks (PLUGIN_FINISH_UNIT, NULL);
targetm.asm_out.file_end ();
timevar_stop (TV_PHASE_LATE_ASM);
}
void
print_version (FILE *file, const char *indent, bool show_global_state)
{
static const char fmt1[] =
#ifdef __GNUC__
N_("%s%s%s %sversion %s (%s)\n%s\tcompiled by GNU C version %s, ")
#else
N_("%s%s%s %sversion %s (%s) compiled by CC, ")
#endif
;
static const char fmt2[] =
N_("GMP version %s, MPFR version %s, MPC version %s, isl version %s\n");
static const char fmt3[] =
N_("%s%swarning: %s header version %s differs from library version %s.\n");
static const char fmt4[] =
N_("%s%sGGC heuristics: --param ggc-min-expand=%d --param ggc-min-heapsize=%d\n");
#ifndef __VERSION__
#define __VERSION__ "[?]"
#endif
fprintf (file,
file == stderr ? _(fmt1) : fmt1,
indent, *indent != 0 ? " " : "",
lang_hooks.name, pkgversion_string, version_string, TARGET_NAME,
indent, __VERSION__);
#define GCC_GMP_STRINGIFY_VERSION3(X) #X
#define GCC_GMP_STRINGIFY_VERSION2(X) GCC_GMP_STRINGIFY_VERSION3 (X)
#define GCC_GMP_VERSION_NUM(X,Y,Z) (((X) << 16L) | ((Y) << 8) | (Z))
#define GCC_GMP_VERSION \
GCC_GMP_VERSION_NUM(__GNU_MP_VERSION, __GNU_MP_VERSION_MINOR, __GNU_MP_VERSION_PATCHLEVEL)
#if GCC_GMP_VERSION < GCC_GMP_VERSION_NUM(4,3,0) && __GNU_MP_VERSION_PATCHLEVEL == 0
#define GCC_GMP_STRINGIFY_VERSION \
GCC_GMP_STRINGIFY_VERSION2 (__GNU_MP_VERSION) "." \
GCC_GMP_STRINGIFY_VERSION2 (__GNU_MP_VERSION_MINOR)
#else
#define GCC_GMP_STRINGIFY_VERSION \
GCC_GMP_STRINGIFY_VERSION2 (__GNU_MP_VERSION) "." \
GCC_GMP_STRINGIFY_VERSION2 (__GNU_MP_VERSION_MINOR) "." \
GCC_GMP_STRINGIFY_VERSION2 (__GNU_MP_VERSION_PATCHLEVEL)
#endif
fprintf (file,
file == stderr ? _(fmt2) : fmt2,
GCC_GMP_STRINGIFY_VERSION, MPFR_VERSION_STRING, MPC_VERSION_STRING,
#ifndef HAVE_isl
"none"
#else
isl_version ()
#endif
);
if (strcmp (GCC_GMP_STRINGIFY_VERSION, gmp_version))
fprintf (file,
file == stderr ? _(fmt3) : fmt3,
indent, *indent != 0 ? " " : "",
"GMP", GCC_GMP_STRINGIFY_VERSION, gmp_version);
if (strcmp (MPFR_VERSION_STRING, mpfr_get_version ()))
fprintf (file,
file == stderr ? _(fmt3) : fmt3,
indent, *indent != 0 ? " " : "",
"MPFR", MPFR_VERSION_STRING, mpfr_get_version ());
if (strcmp (MPC_VERSION_STRING, mpc_get_version ()))
fprintf (file,
file == stderr ? _(fmt3) : fmt3,
indent, *indent != 0 ? " " : "",
"MPC", MPC_VERSION_STRING, mpc_get_version ());
if (show_global_state)
{
fprintf (file,
file == stderr ? _(fmt4) : fmt4,
indent, *indent != 0 ? " " : "",
PARAM_VALUE (GGC_MIN_EXPAND), PARAM_VALUE (GGC_MIN_HEAPSIZE));
print_plugins_versions (file, indent);
}
}
static int
print_to_asm_out_file (print_switch_type type, const char * text)
{
bool prepend_sep = true;
switch (type)
{
case SWITCH_TYPE_LINE_END:
putc ('\n', asm_out_file);
return 1;
case SWITCH_TYPE_LINE_START:
fputs (ASM_COMMENT_START, asm_out_file);
return strlen (ASM_COMMENT_START);
case SWITCH_TYPE_DESCRIPTIVE:
if (ASM_COMMENT_START[0] == 0)
prepend_sep = false;
case SWITCH_TYPE_PASSED:
case SWITCH_TYPE_ENABLED:
if (prepend_sep)
fputc (' ', asm_out_file);
fputs (text, asm_out_file);
return 0;
default:
return -1;
}
}
static int
print_to_stderr (print_switch_type type, const char * text)
{
switch (type)
{
case SWITCH_TYPE_LINE_END:
putc ('\n', stderr);
return 1;
case SWITCH_TYPE_LINE_START:
return 0;
case SWITCH_TYPE_PASSED:
case SWITCH_TYPE_ENABLED:
fputc (' ', stderr);
case SWITCH_TYPE_DESCRIPTIVE:
fputs (text, stderr);
return 0;
default:
return -1;
}
}
static int
print_single_switch (print_switch_fn_type print_fn,
int pos,
print_switch_type type,
const char * text)
{
int len = strlen (text) + 1;
if (pos != 0
&& pos + len > MAX_LINE)
{
print_fn (SWITCH_TYPE_LINE_END, NULL);
pos = 0;
}
if (pos == 0)
pos += print_fn (SWITCH_TYPE_LINE_START, NULL);
print_fn (type, text);
return pos + len;
}
static void
print_switch_values (print_switch_fn_type print_fn)
{
int pos = 0;
size_t j;
pos = print_single_switch (print_fn, pos,
SWITCH_TYPE_DESCRIPTIVE, _("options passed: "));
for (j = 1; j < save_decoded_options_count; j++)
{
switch (save_decoded_options[j].opt_index)
{
case OPT_o:
case OPT_d:
case OPT_dumpbase:
case OPT_dumpdir:
case OPT_auxbase:
case OPT_quiet:
case OPT_version:
continue;
}
pos = print_single_switch (print_fn, pos, SWITCH_TYPE_PASSED,
save_decoded_options[j].orig_option_with_args_text);
}
if (pos > 0)
print_fn (SWITCH_TYPE_LINE_END, NULL);
pos = print_single_switch (print_fn, 0,
SWITCH_TYPE_DESCRIPTIVE, _("options enabled: "));
for (j = 0; j < cl_options_count; j++)
if (cl_options[j].cl_report
&& option_enabled (j, &global_options) > 0)
pos = print_single_switch (print_fn, pos,
SWITCH_TYPE_ENABLED, cl_options[j].opt_text);
print_fn (SWITCH_TYPE_LINE_END, NULL);
}
static void
init_asm_output (const char *name)
{
if (name == NULL && asm_file_name == 0)
asm_out_file = stdout;
else
{
if (asm_file_name == 0)
{
int len = strlen (dump_base_name);
char *dumpname = XNEWVEC (char, len + 6);
memcpy (dumpname, dump_base_name, len + 1);
strip_off_ending (dumpname, len);
strcat (dumpname, ".s");
asm_file_name = dumpname;
}
if (!strcmp (asm_file_name, "-"))
asm_out_file = stdout;
else if (!canonical_filename_eq (asm_file_name, name)
|| !strcmp (asm_file_name, HOST_BIT_BUCKET))
asm_out_file = fopen (asm_file_name, "w");
else
fatal_error (UNKNOWN_LOCATION,
"input file %qs is the same as output file",
asm_file_name);
if (asm_out_file == 0)
fatal_error (UNKNOWN_LOCATION,
"can%'t open %qs for writing: %m", asm_file_name);
}
if (!flag_syntax_only)
{
targetm.asm_out.file_start ();
if (flag_record_gcc_switches)
{
if (targetm.asm_out.record_gcc_switches)
{
targetm.asm_out.record_gcc_switches (SWITCH_TYPE_DESCRIPTIVE,
NULL);
print_switch_values (targetm.asm_out.record_gcc_switches);
targetm.asm_out.record_gcc_switches (SWITCH_TYPE_DESCRIPTIVE,
NULL);
}
else
inform (UNKNOWN_LOCATION,
"-frecord-gcc-switches is not supported by "
"the current target");
}
if (flag_verbose_asm)
{
print_version (asm_out_file, ASM_COMMENT_START, true);
print_switch_values (print_to_asm_out_file);
putc ('\n', asm_out_file);
}
}
}
static void *
realloc_for_line_map (void *ptr, size_t len)
{
return ggc_realloc (ptr, len);
}
static void *
alloc_for_identifier_to_locale (size_t len)
{
return ggc_alloc_atomic (len);
}
void
output_stack_usage (void)
{
static bool warning_issued = false;
enum stack_usage_kind_type { STATIC = 0, DYNAMIC, DYNAMIC_BOUNDED };
const char *stack_usage_kind_str[] = {
"static",
"dynamic",
"dynamic,bounded"
};
HOST_WIDE_INT stack_usage = current_function_static_stack_size;
enum stack_usage_kind_type stack_usage_kind;
if (stack_usage < 0)
{
if (!warning_issued)
{
warning (0, "stack usage computation not supported for this target");
warning_issued = true;
}
return;
}
stack_usage_kind = STATIC;
if (maybe_ne (current_function_pushed_stack_size, 0))
{
HOST_WIDE_INT extra;
if (current_function_pushed_stack_size.is_constant (&extra))
{
stack_usage += extra;
stack_usage_kind = DYNAMIC_BOUNDED;
}
else
{
extra = constant_lower_bound (current_function_pushed_stack_size);
stack_usage += extra;
stack_usage_kind = DYNAMIC;
}
}
if (current_function_allocates_dynamic_stack_space)
{
if (stack_usage_kind != DYNAMIC)
{
if (current_function_has_unbounded_dynamic_stack_size)
stack_usage_kind = DYNAMIC;
else
stack_usage_kind = DYNAMIC_BOUNDED;
}
stack_usage += current_function_dynamic_stack_size;
}
if (flag_stack_usage)
{
expanded_location loc
= expand_location (DECL_SOURCE_LOCATION (current_function_decl));
const char *suffix
= strchr (IDENTIFIER_POINTER (DECL_NAME (current_function_decl)), '.');
const char *name
= lang_hooks.decl_printable_name (current_function_decl, 2);
if (suffix)
{
const char *dot = strchr (name, '.');
while (dot && strcasecmp (dot, suffix) != 0)
{
name = dot + 1;
dot = strchr (name, '.');
}
}
else
{
const char *dot = strrchr (name, '.');
if (dot)
name = dot + 1;
}
fprintf (stack_usage_file,
"%s:%d:%d:%s\t" HOST_WIDE_INT_PRINT_DEC"\t%s\n",
lbasename (loc.file),
loc.line,
loc.column,
name,
stack_usage,
stack_usage_kind_str[stack_usage_kind]);
}
if (warn_stack_usage >= 0)
{
const location_t loc = DECL_SOURCE_LOCATION (current_function_decl);
if (stack_usage_kind == DYNAMIC)
warning_at (loc, OPT_Wstack_usage_, "stack usage might be unbounded");
else if (stack_usage > warn_stack_usage)
{
if (stack_usage_kind == DYNAMIC_BOUNDED)
warning_at (loc,
OPT_Wstack_usage_, "stack usage might be %wd bytes",
stack_usage);
else
warning_at (loc, OPT_Wstack_usage_, "stack usage is %wd bytes",
stack_usage);
}
}
}
static FILE *
open_auxiliary_file (const char *ext)
{
char *filename;
FILE *file;
filename = concat (aux_base_name, ".", ext, NULL);
file = fopen (filename, "w");
if (!file)
fatal_error (input_location, "can%'t open %s for writing: %m", filename);
free (filename);
return file;
}
static void
internal_error_reentered (diagnostic_context *, const char *, va_list *)
{
if (dump_file)
fflush (dump_file);
}
static void
internal_error_function (diagnostic_context *, const char *, va_list *)
{
global_dc->internal_error = internal_error_reentered;
warn_if_plugins ();
emergency_dump_function ();
}
static void
general_init (const char *argv0, bool init_signals)
{
const char *p;
p = argv0 + strlen (argv0);
while (p != argv0 && !IS_DIR_SEPARATOR (p[-1]))
--p;
progname = p;
xmalloc_set_program_name (progname);
hex_init ();
unlock_std_streams ();
gcc_init_libintl ();
identifier_to_locale_alloc = alloc_for_identifier_to_locale;
identifier_to_locale_free = ggc_free;
diagnostic_initialize (global_dc, N_OPTS);
tree_diagnostics_defaults (global_dc);
global_dc->show_caret
= global_options_init.x_flag_diagnostics_show_caret;
global_dc->show_option_requested
= global_options_init.x_flag_diagnostics_show_option;
global_dc->show_column
= global_options_init.x_flag_show_column;
global_dc->internal_error = internal_error_function;
global_dc->option_enabled = option_enabled;
global_dc->option_state = &global_options;
global_dc->option_name = option_name;
if (init_signals)
{
#ifdef SIGSEGV
signal (SIGSEGV, crash_signal);
#endif
#ifdef SIGILL
signal (SIGILL, crash_signal);
#endif
#ifdef SIGBUS
signal (SIGBUS, crash_signal);
#endif
#ifdef SIGABRT
signal (SIGABRT, crash_signal);
#endif
#if defined SIGIOT && (!defined SIGABRT || SIGABRT != SIGIOT)
signal (SIGIOT, crash_signal);
#endif
#ifdef SIGFPE
signal (SIGFPE, crash_signal);
#endif
(*host_hooks.extra_signals)();
}
init_ggc ();
init_stringpool ();
input_location = UNKNOWN_LOCATION;
line_table = ggc_alloc<line_maps> ();
linemap_init (line_table, BUILTINS_LOCATION);
line_table->reallocator = realloc_for_line_map;
line_table->round_alloc_size = ggc_round_alloc_size;
line_table->default_range_bits = 5;
init_ttree ();
init_reg_sets ();
global_init_params ();
init_ggc_heuristics ();
g = new gcc::context ();
g->get_dumps ()->register_dumps ();
g->set_passes (new gcc::pass_manager (g));
symtab = new (ggc_cleared_alloc <symbol_table> ()) symbol_table ();
statistics_early_init ();
finish_params ();
}
static bool
target_supports_section_anchors_p (void)
{
if (targetm.min_anchor_offset == 0 && targetm.max_anchor_offset == 0)
return false;
if (targetm.asm_out.output_anchor == NULL)
return false;
return true;
}
static void
init_alignments (void)
{
if (align_loops <= 0)
align_loops = 1;
if (align_loops_max_skip > align_loops)
align_loops_max_skip = align_loops - 1;
align_loops_log = floor_log2 (align_loops * 2 - 1);
if (align_jumps <= 0)
align_jumps = 1;
if (align_jumps_max_skip > align_jumps)
align_jumps_max_skip = align_jumps - 1;
align_jumps_log = floor_log2 (align_jumps * 2 - 1);
if (align_labels <= 0)
align_labels = 1;
align_labels_log = floor_log2 (align_labels * 2 - 1);
if (align_labels_max_skip > align_labels)
align_labels_max_skip = align_labels - 1;
if (align_functions <= 0)
align_functions = 1;
align_functions_log = floor_log2 (align_functions * 2 - 1);
}
static void
process_options (void)
{
debug_hooks = &do_nothing_debug_hooks;
maximum_field_alignment = initial_max_fld_align * BITS_PER_UNIT;
no_backend = lang_hooks.post_options (&main_input_filename);
location_t saved_location = input_location;
input_location = UNKNOWN_LOCATION;
targetm.target_option.override ();
input_location = saved_location;
if (flag_diagnostics_generate_patch)
global_dc->edit_context_ptr = new edit_context ();
if (flag_compare_debug) 
diagnostic_inhibit_notes (global_dc);
if (flag_section_anchors && !target_supports_section_anchors_p ())
{
warning_at (UNKNOWN_LOCATION, OPT_fsection_anchors,
"this target does not support %qs",
"-fsection-anchors");
flag_section_anchors = 0;
}
if (flag_short_enums == 2)
flag_short_enums = targetm.default_short_enums ();
if (aux_base_name)
;
else if (main_input_filename)
{
char *name = xstrdup (lbasename (main_input_filename));
strip_off_ending (name, strlen (name));
aux_base_name = name;
}
else
aux_base_name = "gccaux";
#ifndef HAVE_isl
if (flag_graphite
|| flag_loop_nest_optimize
|| flag_graphite_identity
|| flag_loop_parallelize_all)
sorry ("Graphite loop optimizations cannot be used (isl is not available) "
"(-fgraphite, -fgraphite-identity, -floop-nest-optimize, "
"-floop-parallelize-all)");
#endif
if (flag_cf_protection != CF_NONE
&& !(flag_cf_protection & CF_SET))
{
if (flag_cf_protection == CF_FULL)
{
error_at (UNKNOWN_LOCATION,
"%<-fcf-protection=full%> is not supported for this "
"target");
flag_cf_protection = CF_NONE;
}
if (flag_cf_protection == CF_BRANCH)
{
error_at (UNKNOWN_LOCATION,
"%<-fcf-protection=branch%> is not supported for this "
"target");
flag_cf_protection = CF_NONE;
}
if (flag_cf_protection == CF_RETURN)
{
error_at (UNKNOWN_LOCATION,
"%<-fcf-protection=return%> is not supported for this "
"target");
flag_cf_protection = CF_NONE;
}
}
if (flag_check_pointer_bounds)
{
if (targetm.chkp_bound_mode () == VOIDmode)
{
error_at (UNKNOWN_LOCATION,
"%<-fcheck-pointer-bounds%> is not supported for this "
"target");
flag_check_pointer_bounds = 0;
}
if (flag_sanitize & SANITIZE_BOUNDS_STRICT)
{
error_at (UNKNOWN_LOCATION,
"%<-fcheck-pointer-bounds%> is not supported with "
"%<-fsanitize=bounds-strict%>");
flag_check_pointer_bounds = 0;
}
else if (flag_sanitize & SANITIZE_BOUNDS)
{
error_at (UNKNOWN_LOCATION,
"%<-fcheck-pointer-bounds%> is not supported with "
"%<-fsanitize=bounds%>");
flag_check_pointer_bounds = 0;
}
if (flag_sanitize & SANITIZE_ADDRESS)
{
error_at (UNKNOWN_LOCATION,
"%<-fcheck-pointer-bounds%> is not supported with "
"Address Sanitizer");
flag_check_pointer_bounds = 0;
}
if (flag_sanitize & SANITIZE_THREAD)
{
error_at (UNKNOWN_LOCATION,
"%<-fcheck-pointer-bounds%> is not supported with "
"Thread Sanitizer");
flag_check_pointer_bounds = 0;
}
}
if (flag_ira_region == IRA_REGION_AUTODETECT)
flag_ira_region
= optimize_size || !optimize ? IRA_REGION_ONE : IRA_REGION_MIXED;
if (!abi_version_at_least (2))
{
error_at (UNKNOWN_LOCATION,
"%<-fabi-version=1%> is no longer supported");
flag_abi_version = 2;
}
if (flag_unroll_all_loops)
flag_unroll_loops = 1;
if (flag_web == AUTODETECT_VALUE)
flag_web = flag_unroll_loops;
if (flag_rename_registers == AUTODETECT_VALUE)
flag_rename_registers = flag_unroll_loops;
if (flag_non_call_exceptions)
flag_asynchronous_unwind_tables = 1;
if (flag_asynchronous_unwind_tables)
flag_unwind_tables = 1;
if (flag_value_profile_transformations)
flag_profile_values = 1;
#ifndef INSN_SCHEDULING
if (flag_schedule_insns || flag_schedule_insns_after_reload)
warning_at (UNKNOWN_LOCATION, 0,
"instruction scheduling not supported on this target machine");
#endif
if (!DELAY_SLOTS && flag_delayed_branch)
warning_at (UNKNOWN_LOCATION, 0,
"this target machine does not have delayed branches");
user_label_prefix = USER_LABEL_PREFIX;
if (flag_leading_underscore != -1)
{
if (user_label_prefix[0] == 0 ||
(user_label_prefix[0] == '_' && user_label_prefix[1] == 0))
{
user_label_prefix = flag_leading_underscore ? "_" : "";
}
else
warning_at (UNKNOWN_LOCATION, 0,
"-f%sleading-underscore not supported on this "
"target machine", flag_leading_underscore ? "" : "no-");
}
if (version_flag)
{
print_version (stderr, "", true);
if (! quiet_flag)
print_switch_values (print_to_stderr);
}
if (flag_syntax_only)
{
write_symbols = NO_DEBUG;
profile_flag = 0;
}
if (flag_gtoggle)
{
if (debug_info_level == DINFO_LEVEL_NONE)
{
debug_info_level = DINFO_LEVEL_NORMAL;
if (write_symbols == NO_DEBUG)
write_symbols = PREFERRED_DEBUGGING_TYPE;
}
else
debug_info_level = DINFO_LEVEL_NONE;
}
if (flag_dump_final_insns && !flag_syntax_only && !no_backend)
{
FILE *final_output = fopen (flag_dump_final_insns, "w");
if (!final_output)
{
error_at (UNKNOWN_LOCATION,
"could not open final insn dump file %qs: %m",
flag_dump_final_insns);
flag_dump_final_insns = NULL;
}
else if (fclose (final_output))
{
error_at (UNKNOWN_LOCATION,
"could not close zeroed insn dump file %qs: %m",
flag_dump_final_insns);
flag_dump_final_insns = NULL;
}
}
if (debug_info_level == DINFO_LEVEL_NONE)
write_symbols = NO_DEBUG;
if (write_symbols == NO_DEBUG)
;
#if defined(DBX_DEBUGGING_INFO)
else if (write_symbols == DBX_DEBUG)
debug_hooks = &dbx_debug_hooks;
#endif
#if defined(XCOFF_DEBUGGING_INFO)
else if (write_symbols == XCOFF_DEBUG)
debug_hooks = &xcoff_debug_hooks;
#endif
#ifdef DWARF2_DEBUGGING_INFO
else if (write_symbols == DWARF2_DEBUG)
debug_hooks = &dwarf2_debug_hooks;
#endif
#ifdef VMS_DEBUGGING_INFO
else if (write_symbols == VMS_DEBUG || write_symbols == VMS_AND_DWARF2_DEBUG)
debug_hooks = &vmsdbg_debug_hooks;
#endif
#ifdef DWARF2_LINENO_DEBUGGING_INFO
else if (write_symbols == DWARF2_DEBUG)
debug_hooks = &dwarf2_lineno_debug_hooks;
#endif
else
error_at (UNKNOWN_LOCATION,
"target system does not support the %qs debug format",
debug_type_names[write_symbols]);
if (debug_info_level < DINFO_LEVEL_NORMAL
|| debug_hooks->var_location == do_nothing_debug_hooks.var_location)
{
if (flag_var_tracking == 1
|| flag_var_tracking_uninit == 1)
{
if (debug_info_level < DINFO_LEVEL_NORMAL)
warning_at (UNKNOWN_LOCATION, 0,
"variable tracking requested, but useless unless "
"producing debug info");
else
warning_at (UNKNOWN_LOCATION, 0,
"variable tracking requested, but not supported "
"by this debug format");
}
flag_var_tracking = 0;
flag_var_tracking_uninit = 0;
}
if (flag_dump_go_spec != NULL)
debug_hooks = dump_go_spec_init (flag_dump_go_spec, debug_hooks);
if (flag_var_tracking_uninit == 1)
flag_var_tracking = 1;
if (flag_var_tracking == AUTODETECT_VALUE)
flag_var_tracking = optimize >= 1;
if (flag_var_tracking_uninit == AUTODETECT_VALUE)
flag_var_tracking_uninit = flag_var_tracking;
if (flag_var_tracking_assignments == AUTODETECT_VALUE)
flag_var_tracking_assignments
= (flag_var_tracking
&& !(flag_selective_scheduling || flag_selective_scheduling2));
if (flag_var_tracking_assignments_toggle)
flag_var_tracking_assignments = !flag_var_tracking_assignments;
if (flag_var_tracking_assignments && !flag_var_tracking)
flag_var_tracking = flag_var_tracking_assignments = -1;
if (flag_var_tracking_assignments
&& (flag_selective_scheduling || flag_selective_scheduling2))
warning_at (UNKNOWN_LOCATION, 0,
"var-tracking-assignments changes selective scheduling");
if (debug_nonbind_markers_p == AUTODETECT_VALUE)
debug_nonbind_markers_p
= (optimize
&& debug_info_level >= DINFO_LEVEL_NORMAL
&& (write_symbols == DWARF2_DEBUG
|| write_symbols == VMS_AND_DWARF2_DEBUG)
&& !(flag_selective_scheduling || flag_selective_scheduling2));
if (dwarf2out_as_loc_support == AUTODETECT_VALUE)
dwarf2out_as_loc_support
= dwarf2out_default_as_loc_support ();
if (dwarf2out_as_locview_support == AUTODETECT_VALUE)
dwarf2out_as_locview_support
= dwarf2out_default_as_locview_support ();
if (debug_variable_location_views == AUTODETECT_VALUE)
{
debug_variable_location_views
= (flag_var_tracking
&& debug_info_level >= DINFO_LEVEL_NORMAL
&& (write_symbols == DWARF2_DEBUG
|| write_symbols == VMS_AND_DWARF2_DEBUG)
&& !dwarf_strict
&& dwarf2out_as_loc_support
&& dwarf2out_as_locview_support);
}
else if (debug_variable_location_views == -1 && dwarf_version != 5)
{
warning_at (UNKNOWN_LOCATION, 0,
"without -gdwarf-5, -gvariable-location-views=incompat5 "
"is equivalent to -gvariable-location-views");
debug_variable_location_views = 1;
}
if (debug_internal_reset_location_views == 2)
{
debug_internal_reset_location_views
= (debug_variable_location_views
&& targetm.reset_location_view);
}
else if (debug_internal_reset_location_views
&& !debug_variable_location_views)
{
warning_at (UNKNOWN_LOCATION, 0,
"-ginternal-reset-location-views is forced disabled "
"without -gvariable-location-views");
debug_internal_reset_location_views = 0;
}
if (debug_inline_points == AUTODETECT_VALUE)
debug_inline_points = debug_variable_location_views;
else if (debug_inline_points && !debug_nonbind_markers_p)
{
warning_at (UNKNOWN_LOCATION, 0,
"-ginline-points is forced disabled without "
"-gstatement-frontiers");
debug_inline_points = 0;
}
if (flag_tree_cselim == AUTODETECT_VALUE)
{
if (HAVE_conditional_move)
flag_tree_cselim = 1;
else
flag_tree_cselim = 0;
}
if (flag_gen_aux_info)
{
aux_info_file = fopen (aux_info_file_name, "w");
if (aux_info_file == 0)
fatal_error (UNKNOWN_LOCATION,
"can%'t open %s: %m", aux_info_file_name);
}
if (!targetm_common.have_named_sections)
{
if (flag_function_sections)
{
warning_at (UNKNOWN_LOCATION, 0,
"-ffunction-sections not supported for this target");
flag_function_sections = 0;
}
if (flag_data_sections)
{
warning_at (UNKNOWN_LOCATION, 0,
"-fdata-sections not supported for this target");
flag_data_sections = 0;
}
}
if (flag_prefetch_loop_arrays > 0 && !targetm.code_for_prefetch)
{
warning_at (UNKNOWN_LOCATION, 0,
"-fprefetch-loop-arrays not supported for this target");
flag_prefetch_loop_arrays = 0;
}
else if (flag_prefetch_loop_arrays > 0 && !targetm.have_prefetch ())
{
warning_at (UNKNOWN_LOCATION, 0,
"-fprefetch-loop-arrays not supported for this target "
"(try -march switches)");
flag_prefetch_loop_arrays = 0;
}
if (flag_prefetch_loop_arrays > 0 && optimize_size)
{
warning_at (UNKNOWN_LOCATION, 0,
"-fprefetch-loop-arrays is not supported with -Os");
flag_prefetch_loop_arrays = 0;
}
if (flag_signaling_nans)
flag_trapping_math = 1;
if (flag_associative_math && (flag_trapping_math || flag_signed_zeros))
{
warning_at (UNKNOWN_LOCATION, 0,
"-fassociative-math disabled; other options take "
"precedence");
flag_associative_math = 0;
}
if (flag_stack_clash_protection && !STACK_GROWS_DOWNWARD)
{
warning_at (UNKNOWN_LOCATION, 0,
"%<-fstack-clash-protection%> is not supported on targets "
"where the stack grows from lower to higher addresses");
flag_stack_clash_protection = 0;
}
if (flag_stack_check != NO_STACK_CHECK && flag_stack_clash_protection)
{
warning_at (UNKNOWN_LOCATION, 0,
"%<-fstack-check=%> and %<-fstack-clash_protection%> are "
"mutually exclusive.  Disabling %<-fstack-check=%>");
flag_stack_check = NO_STACK_CHECK;
}
if (flag_cx_limited_range)
flag_complex_method = 0;
if (flag_cx_fortran_rules)
flag_complex_method = 1;
if (!FRAME_GROWS_DOWNWARD && flag_stack_protect)
{
warning_at (UNKNOWN_LOCATION, 0,
"-fstack-protector not supported for this target");
flag_stack_protect = 0;
}
if (!flag_stack_protect)
warn_stack_protect = 0;
if ((flag_sanitize & SANITIZE_ADDRESS)
&& !FRAME_GROWS_DOWNWARD)
{
warning_at (UNKNOWN_LOCATION, 0,
"-fsanitize=address and -fsanitize=kernel-address "
"are not supported for this target");
flag_sanitize &= ~SANITIZE_ADDRESS;
}
if ((flag_sanitize & SANITIZE_USER_ADDRESS)
&& targetm.asan_shadow_offset == NULL)
{
warning_at (UNKNOWN_LOCATION, 0,
"-fsanitize=address not supported for this target");
flag_sanitize &= ~SANITIZE_ADDRESS;
}
if (profile_flag || function_entry_patch_area_size
|| !targetm.have_prologue () || !targetm.have_epilogue ())
flag_ipa_ra = 0;
if (!global_options_set.x_warnings_are_errors
&& warn_coverage_mismatch
&& (global_dc->classify_diagnostic[OPT_Wcoverage_mismatch] ==
DK_UNSPECIFIED))
diagnostic_classify_diagnostic (global_dc, OPT_Wcoverage_mismatch,
DK_ERROR, UNKNOWN_LOCATION);
optimization_default_node = build_optimization_node (&global_options);
optimization_current_node = optimization_default_node;
}
static void
backend_init_target (void)
{
init_alignments ();
init_fake_stack_mems ();
init_alias_target ();
if (!ira_use_lra_p)
init_reload ();
recog_init ();
init_dummy_function_start ();
init_expmed ();
init_lower_subreg ();
init_set_costs ();
init_expr_target ();
ira_init ();
caller_save_initialized_p = false;
expand_dummy_function_end ();
}
static void
backend_init (void)
{
init_emit_once ();
init_rtlanal ();
init_inline_once ();
init_varasm_once ();
save_register_info ();
init_emit_regs ();
init_regs ();
}
static void
init_excess_precision (void)
{
gcc_assert (flag_excess_precision_cmdline != EXCESS_PRECISION_DEFAULT);
flag_excess_precision = flag_excess_precision_cmdline;
}
static void
lang_dependent_init_target (void)
{
init_excess_precision ();
init_optabs ();
gcc_assert (!this_target_rtl->target_specific_initialized);
}
static int rtl_initialized;
void
initialize_rtl (void)
{
auto_timevar tv (g_timer, TV_INITIALIZE_RTL);
if (!rtl_initialized)
ira_init_once ();
rtl_initialized = true;
if (!this_target_rtl->target_specific_initialized)
{
backend_init_target ();
this_target_rtl->target_specific_initialized = true;
}
}
static int
lang_dependent_init (const char *name)
{
location_t save_loc = input_location;
if (dump_base_name == 0)
dump_base_name = name && name[0] ? name : "gccdump";
input_location = BUILTINS_LOCATION;
if (lang_hooks.init () == 0)
return 0;
input_location = save_loc;
if (!flag_wpa)
{
init_asm_output (name);
if (flag_stack_usage)
stack_usage_file = open_auxiliary_file ("su");
}
init_eh ();
lang_dependent_init_target ();
if (!flag_wpa)
{
timevar_push (TV_SYMOUT);
(*debug_hooks->init) (name);
timevar_pop (TV_SYMOUT);
}
return 1;
}
void
target_reinit (void)
{
struct rtl_data saved_x_rtl;
rtx *saved_regno_reg_rtx;
tree saved_optimization_current_node;
struct target_optabs *saved_this_fn_optabs;
saved_optimization_current_node = optimization_current_node;
saved_this_fn_optabs = this_fn_optabs;
if (saved_optimization_current_node != optimization_default_node)
{
optimization_current_node = optimization_default_node;
cl_optimization_restore
(&global_options,
TREE_OPTIMIZATION (optimization_default_node));
}
this_fn_optabs = this_target_optabs;
saved_regno_reg_rtx = regno_reg_rtx;
if (saved_regno_reg_rtx)
{  
saved_x_rtl = *crtl;
memset (crtl, '\0', sizeof (*crtl));
regno_reg_rtx = NULL;
}
this_target_rtl->target_specific_initialized = false;
init_emit_regs ();
init_regs ();
lang_dependent_init_target ();
if (saved_optimization_current_node != optimization_default_node)
{
optimization_current_node = saved_optimization_current_node;
cl_optimization_restore (&global_options,
TREE_OPTIMIZATION (optimization_current_node));
}
this_fn_optabs = saved_this_fn_optabs;
if (saved_regno_reg_rtx)
{
*crtl = saved_x_rtl;
regno_reg_rtx = saved_regno_reg_rtx;
saved_regno_reg_rtx = NULL;
}
}
void
dump_memory_report (bool final)
{
dump_line_table_statistics ();
ggc_print_statistics ();
stringpool_statistics ();
dump_tree_statistics ();
dump_gimple_statistics ();
dump_rtx_statistics ();
dump_alloc_pool_statistics ();
dump_bitmap_statistics ();
dump_hash_table_loc_statistics ();
dump_vec_loc_statistics ();
dump_ggc_loc_statistics (final);
dump_alias_stats (stderr);
dump_pta_stats (stderr);
}
static void
finalize (bool no_backend)
{
if (flag_gen_aux_info)
{
fclose (aux_info_file);
aux_info_file = NULL;
if (seen_error ())
unlink (aux_info_file_name);
}
if (asm_out_file)
{
if (ferror (asm_out_file) != 0)
fatal_error (input_location, "error writing to %s: %m", asm_file_name);
if (fclose (asm_out_file) != 0)
fatal_error (input_location, "error closing %s: %m", asm_file_name);
asm_out_file = NULL;
}
if (stack_usage_file)
{
fclose (stack_usage_file);
stack_usage_file = NULL;
}
if (seen_error ())
coverage_remove_note_file ();
if (!no_backend)
{
statistics_fini ();
g->get_passes ()->finish_optimization_passes ();
lra_finish_once ();
}
if (mem_report)
dump_memory_report (true);
if (profile_report)
dump_profile_report ();
lang_hooks.finish ();
}
static bool
standard_type_bitsize (int bitsize)
{
if (bitsize == 128)
return false;
if (bitsize == CHAR_TYPE_SIZE
|| bitsize == SHORT_TYPE_SIZE
|| bitsize == INT_TYPE_SIZE
|| bitsize == LONG_TYPE_SIZE
|| bitsize == LONG_LONG_TYPE_SIZE)
return true;
return false;
}
static void
do_compile ()
{
process_options ();
if (!seen_error ())
{
int i;
timevar_start (TV_PHASE_SETUP);
init_adjust_machine_modes ();
init_derived_machine_modes ();
for (i = 0; i < NUM_INT_N_ENTS; i ++)
if (targetm.scalar_mode_supported_p (int_n_data[i].m)
&& ! standard_type_bitsize (int_n_data[i].bitsize))
int_n_enabled_p[i] = true;
else
int_n_enabled_p[i] = false;
if (!no_backend)
backend_init ();
if (lang_dependent_init (main_input_filename))
{
ggc_protect_identifiers = true;
symtab->initialize ();
init_final (main_input_filename);
coverage_init (aux_base_name);
statistics_init ();
invoke_plugin_callbacks (PLUGIN_START_UNIT, NULL);
timevar_stop (TV_PHASE_SETUP);
compile_file ();
}
else
{
timevar_stop (TV_PHASE_SETUP);
}
timevar_start (TV_PHASE_FINALIZE);
finalize (no_backend);
timevar_stop (TV_PHASE_FINALIZE);
}
}
toplev::toplev (timer *external_timer,
bool init_signals)
: m_use_TV_TOTAL (external_timer == NULL),
m_init_signals (init_signals)
{
if (external_timer)
g_timer = external_timer;
}
toplev::~toplev ()
{
if (g_timer && m_use_TV_TOTAL)
{
g_timer->stop (TV_TOTAL);
g_timer->print (stderr);
delete g_timer;
g_timer = NULL;
}
}
void
toplev::start_timevars ()
{
if (time_report || !quiet_flag  || flag_detailed_statistics)
timevar_init ();
timevar_start (TV_TOTAL);
}
void
toplev::run_self_tests ()
{
if (no_backend)
{
error_at (UNKNOWN_LOCATION, "self-tests incompatible with -E");
return;
}
#if CHECKING_P
input_location = UNKNOWN_LOCATION;
bitmap_obstack_initialize (NULL);
::selftest::run_tests ();
bitmap_obstack_release (NULL);
#else
inform (UNKNOWN_LOCATION, "self-tests are not enabled in this build");
#endif 
}
int
toplev::main (int argc, char **argv)
{
stack_limit_increase (64 * 1024 * 1024);
expandargv (&argc, &argv);
general_init (argv[0], m_init_signals);
init_options_once ();
init_opts_obstack ();
init_options_struct (&global_options, &global_options_set);
lang_hooks.init_options_struct (&global_options);
decode_cmdline_options_to_array_default_mask (argc,
CONST_CAST2 (const char **,
char **, argv),
&save_decoded_options,
&save_decoded_options_count);
lang_hooks.init_options (save_decoded_options_count, save_decoded_options);
decode_options (&global_options, &global_options_set,
save_decoded_options, save_decoded_options_count,
UNKNOWN_LOCATION, global_dc,
targetm.target_option.override);
handle_common_deferred_options ();
init_local_tick ();
initialize_plugins ();
if (version_flag)
print_version (stderr, "", true);
if (help_flag)
print_plugins_help (stderr, "");
if (!exit_after_options)
{
if (m_use_TV_TOTAL)
start_timevars ();
do_compile ();
}
if (warningcount || errorcount || werrorcount)
print_ignored_options ();
if (flag_self_test)
run_self_tests ();
invoke_plugin_callbacks (PLUGIN_FINISH, NULL);
if (flag_diagnostics_generate_patch)
{
gcc_assert (global_dc->edit_context_ptr);
pretty_printer pp;
pp_show_color (&pp) = pp_show_color (global_dc->printer);
global_dc->edit_context_ptr->print_diff (&pp, true);
pp_flush (&pp);
}
diagnostic_finish (global_dc);
finalize_plugins ();
after_memory_report = true;
if (seen_error () || werrorcount)
return (FATAL_EXIT_CODE);
return (SUCCESS_EXIT_CODE);
}
void
toplev::finalize (void)
{
rtl_initialized = false;
this_target_rtl->target_specific_initialized = false;
ipa_reference_c_finalize ();
ipa_fnsummary_c_finalize ();
cgraph_c_finalize ();
cgraphunit_c_finalize ();
dwarf2out_c_finalize ();
gcse_c_finalize ();
ipa_cp_c_finalize ();
ira_costs_c_finalize ();
params_c_finalize ();
finalize_options_struct (&global_options);
finalize_options_struct (&global_options_set);
obstack_free (&opts_obstack, NULL);
XDELETEVEC (save_decoded_options);
save_decoded_options = NULL;
save_decoded_options_count = 0;
delete g;
g = NULL;
}
