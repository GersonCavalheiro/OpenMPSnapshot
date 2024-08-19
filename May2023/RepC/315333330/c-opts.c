#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "tm.h"
#include "c-target.h"
#include "c-common.h"
#include "memmodel.h"
#include "tm_p.h"		
#include "diagnostic.h"
#include "c-pragma.h"
#include "flags.h"
#include "toplev.h"
#include "langhooks.h"
#include "tree-diagnostic.h" 
#include "intl.h"
#include "cppdefault.h"
#include "incpath.h"
#include "debug.h"		
#include "opts.h"
#include "plugin.h"		
#include "mkdeps.h"
#include "dumpfile.h"
#include "file-prefix-map.h"    
#ifndef DOLLARS_IN_IDENTIFIERS
# define DOLLARS_IN_IDENTIFIERS true
#endif
#ifndef TARGET_SYSTEM_ROOT
# define TARGET_SYSTEM_ROOT NULL
#endif
#ifndef TARGET_OPTF
#define TARGET_OPTF(ARG)
#endif
cpp_options *cpp_opts;
static const char *this_input_filename;
static const char *out_fname;
static FILE *out_stream;
static bool deps_append;
static bool deps_seen;
static bool verbose;
static const char *deps_file;
static const char *iprefix;
static const char *imultilib;
static const char *sysroot = TARGET_SYSTEM_ROOT;
static bool std_inc = true;
static bool std_cxx_inc = true;
static bool quote_chain_split;
static size_t deferred_count;
static size_t include_cursor;
static FILE *original_dump_file = NULL;
static dump_flags_t original_dump_flags;
static bool done_preinclude;
static void handle_OPT_d (const char *);
static void set_std_cxx98 (int);
static void set_std_cxx11 (int);
static void set_std_cxx14 (int);
static void set_std_cxx17 (int);
static void set_std_cxx2a (int);
static void set_std_c89 (int, int);
static void set_std_c99 (int);
static void set_std_c11 (int);
static void set_std_c17 (int);
static void check_deps_environment_vars (void);
static void handle_deferred_opts (void);
static void sanitize_cpp_opts (void);
static void add_prefixed_path (const char *, incpath_kind);
static void push_command_line_include (void);
static void cb_file_change (cpp_reader *, const line_map_ordinary *);
static void cb_dir_change (cpp_reader *, const char *);
static void c_finish_options (void);
#ifndef STDC_0_IN_SYSTEM_HEADERS
#define STDC_0_IN_SYSTEM_HEADERS 0
#endif
static void defer_opt (enum opt_code, const char *);
static struct deferred_opt
{
enum opt_code code;
const char *arg;
} *deferred_opts;
extern const unsigned int 
c_family_lang_mask = (CL_C | CL_CXX | CL_ObjC | CL_ObjCXX);
static void
defer_opt (enum opt_code code, const char *arg)
{
deferred_opts[deferred_count].code = code;
deferred_opts[deferred_count].arg = arg;
deferred_count++;
}
unsigned int
c_common_option_lang_mask (void)
{
static const unsigned int lang_flags[] = {CL_C, CL_ObjC, CL_CXX, CL_ObjCXX};
return lang_flags[c_language];
}
static void
c_diagnostic_finalizer (diagnostic_context *context,
diagnostic_info *diagnostic)
{
diagnostic_show_locus (context, diagnostic->richloc, diagnostic->kind);
virt_loc_aware_diagnostic_finalizer (context, diagnostic);
pp_destroy_prefix (context->printer);
pp_flush (context->printer);
}
void
c_common_diagnostics_set_defaults (diagnostic_context *context)
{
diagnostic_finalizer (context) = c_diagnostic_finalizer;
context->opt_permissive = OPT_fpermissive;
}
static bool accept_all_c_family_options = false;
bool
c_common_complain_wrong_lang_p (const struct cl_option *option)
{
if (accept_all_c_family_options
&& (option->flags & c_family_lang_mask))
return false;
return true;
}
void
c_common_init_options_struct (struct gcc_options *opts)
{
opts->x_flag_exceptions = c_dialect_cxx ();
opts->x_warn_pointer_arith = c_dialect_cxx ();
opts->x_warn_write_strings = c_dialect_cxx ();
opts->x_flag_warn_unused_result = true;
opts->x_flag_complex_method = 2;
}
void
c_common_init_options (unsigned int decoded_options_count,
struct cl_decoded_option *decoded_options)
{
unsigned int i;
struct cpp_callbacks *cb;
g_string_concat_db
= new (ggc_alloc <string_concat_db> ()) string_concat_db ();
parse_in = cpp_create_reader (c_dialect_cxx () ? CLK_GNUCXX: CLK_GNUC89,
ident_hash, line_table);
cb = cpp_get_callbacks (parse_in);
cb->error = c_cpp_error;
cpp_opts = cpp_get_options (parse_in);
cpp_opts->dollars_in_ident = DOLLARS_IN_IDENTIFIERS;
cpp_opts->objc = c_dialect_objc ();
cpp_opts->warn_dollars = 0;
deferred_opts = XNEWVEC (struct deferred_opt, decoded_options_count);
if (c_language == clk_c)
{
set_std_c17 (false );
for (i = 1; i < decoded_options_count; i++)
if (decoded_options[i].opt_index == OPT_lang_asm)
{
accept_all_c_family_options = true;
break;
}
}
if (c_dialect_cxx ())
set_std_cxx14 (false);
global_dc->colorize_source_p = true;
}
bool
c_common_handle_option (size_t scode, const char *arg, int value,
int kind, location_t loc,
const struct cl_option_handlers *handlers)
{
const struct cl_option *option = &cl_options[scode];
enum opt_code code = (enum opt_code) scode;
bool result = true;
bool preprocessing_asm_p = (cpp_get_options (parse_in)->lang == CLK_ASM);
switch (code)
{
default:
if (cl_options[code].flags & c_family_lang_mask)
{
if ((option->flags & CL_TARGET)
&& ! targetcm.handle_c_option (scode, arg, value))
result = false;
break;
}
result = false;
break;
case OPT__output_pch_:
pch_file = arg;
break;
case OPT_A:
defer_opt (code, arg);
break;
case OPT_C:
cpp_opts->discard_comments = 0;
break;
case OPT_CC:
cpp_opts->discard_comments = 0;
cpp_opts->discard_comments_in_macro_exp = 0;
break;
case OPT_D:
defer_opt (code, arg);
break;
case OPT_H:
cpp_opts->print_include_names = 1;
break;
case OPT_F:
TARGET_OPTF (xstrdup (arg));
break;
case OPT_I:
if (strcmp (arg, "-"))
add_path (xstrdup (arg), INC_BRACKET, 0, true);
else
{
if (quote_chain_split)
error ("-I- specified twice");
quote_chain_split = true;
split_quote_chain ();
inform (input_location, "obsolete option -I- used, please use -iquote instead");
}
break;
case OPT_M:
case OPT_MM:
cpp_opts->deps.style = (code == OPT_M ? DEPS_SYSTEM: DEPS_USER);
flag_no_output = 1;
break;
case OPT_MD:
case OPT_MMD:
cpp_opts->deps.style = (code == OPT_MD ? DEPS_SYSTEM: DEPS_USER);
cpp_opts->deps.need_preprocessor_output = true;
deps_file = arg;
break;
case OPT_MF:
deps_seen = true;
deps_file = arg;
break;
case OPT_MG:
deps_seen = true;
cpp_opts->deps.missing_files = true;
break;
case OPT_MP:
deps_seen = true;
cpp_opts->deps.phony_targets = true;
break;
case OPT_MQ:
case OPT_MT:
deps_seen = true;
defer_opt (code, arg);
break;
case OPT_P:
flag_no_line_commands = 1;
break;
case OPT_U:
defer_opt (code, arg);
break;
case OPT_Wall:
cpp_opts->warn_num_sign_change = value;
break;
case OPT_Walloca_larger_than_:
if (!value)
inform (loc, "-Walloca-larger-than=0 is meaningless");
break;
case OPT_Wvla_larger_than_:
if (!value)
inform (loc, "-Wvla-larger-than=0 is meaningless");
break;
case OPT_Wunknown_pragmas:
warn_unknown_pragmas = value * 2;
break;
case OPT_ansi:
if (!c_dialect_cxx ())
set_std_c89 (false, true);
else
set_std_cxx98 (true);
break;
case OPT_d:
handle_OPT_d (arg);
break;
case OPT_Wabi_:
warn_abi = true;
if (value == 1)
{
warning (0, "%<-Wabi=1%> is not supported, using =2");
value = 2;
}
warn_abi_version = value;
break;
case OPT_fcanonical_system_headers:
cpp_opts->canonical_system_headers = value;
break;
case OPT_fcond_mismatch:
if (!c_dialect_cxx ())
{
flag_cond_mismatch = value;
break;
}
warning (0, "switch %qs is no longer supported", option->opt_text);
break;
case OPT_fbuiltin_:
if (value)
result = false;
else
disable_builtin_function (arg);
break;
case OPT_fdirectives_only:
cpp_opts->directives_only = value;
break;
case OPT_fdollars_in_identifiers:
cpp_opts->dollars_in_ident = value;
break;
case OPT_fmacro_prefix_map_:
add_macro_prefix_map (arg);
break;
case OPT_ffreestanding:
value = !value;
case OPT_fhosted:
flag_hosted = value;
flag_no_builtin = !value;
break;
case OPT_fconstant_string_class_:
constant_string_class_name = arg;
break;
case OPT_fextended_identifiers:
cpp_opts->extended_identifiers = value;
break;
case OPT_foperator_names:
cpp_opts->operator_names = value;
break;
case OPT_fpch_deps:
cpp_opts->restore_pch_deps = value;
break;
case OPT_fpch_preprocess:
flag_pch_preprocess = value;
break;
case OPT_fpermissive:
flag_permissive = value;
global_dc->permissive = value;
break;
case OPT_fpreprocessed:
cpp_opts->preprocessed = value;
break;
case OPT_fdebug_cpp:
cpp_opts->debug = 1;
break;
case OPT_ftrack_macro_expansion:
if (value)
value = 2;
case OPT_ftrack_macro_expansion_:
if (arg && *arg != '\0')
cpp_opts->track_macro_expansion = value;
else
cpp_opts->track_macro_expansion = 2;
break;
case OPT_frepo:
flag_use_repository = value;
if (value)
flag_implicit_templates = 0;
break;
case OPT_ftabstop_:
if (value >= 1 && value <= 100)
cpp_opts->tabstop = value;
break;
case OPT_fexec_charset_:
cpp_opts->narrow_charset = arg;
break;
case OPT_fwide_exec_charset_:
cpp_opts->wide_charset = arg;
break;
case OPT_finput_charset_:
cpp_opts->input_charset = arg;
break;
case OPT_ftemplate_depth_:
max_tinst_depth = value;
break;
case OPT_fvisibility_inlines_hidden:
visibility_options.inlines_hidden = value;
break;
case OPT_femit_struct_debug_baseonly:
set_struct_debug_option (&global_options, loc, "base");
break;
case OPT_femit_struct_debug_reduced:
set_struct_debug_option (&global_options, loc,
"dir:ord:sys,dir:gen:any,ind:base");
break;
case OPT_femit_struct_debug_detailed_:
set_struct_debug_option (&global_options, loc, arg);
break;
case OPT_fext_numeric_literals:
cpp_opts->ext_numeric_literals = value;
break;
case OPT_idirafter:
add_path (xstrdup (arg), INC_AFTER, 0, true);
break;
case OPT_imacros:
case OPT_include:
defer_opt (code, arg);
break;
case OPT_imultilib:
imultilib = arg;
break;
case OPT_iprefix:
iprefix = arg;
break;
case OPT_iquote:
add_path (xstrdup (arg), INC_QUOTE, 0, true);
break;
case OPT_isysroot:
sysroot = arg;
break;
case OPT_isystem:
add_path (xstrdup (arg), INC_SYSTEM, 0, true);
break;
case OPT_iwithprefix:
add_prefixed_path (arg, INC_SYSTEM);
break;
case OPT_iwithprefixbefore:
add_prefixed_path (arg, INC_BRACKET);
break;
case OPT_lang_asm:
cpp_set_lang (parse_in, CLK_ASM);
cpp_opts->dollars_in_ident = false;
break;
case OPT_nostdinc:
std_inc = false;
break;
case OPT_nostdinc__:
std_cxx_inc = false;
break;
case OPT_o:
if (!out_fname)
out_fname = arg;
else
error ("output filename specified twice");
break;
case OPT_print_objc_runtime_info:
print_struct_values = 1;
break;
case OPT_remap:
cpp_opts->remap = 1;
break;
case OPT_std_c__98:
case OPT_std_gnu__98:
if (!preprocessing_asm_p)
set_std_cxx98 (code == OPT_std_c__98 );
break;
case OPT_std_c__11:
case OPT_std_gnu__11:
if (!preprocessing_asm_p)
set_std_cxx11 (code == OPT_std_c__11 );
break;
case OPT_std_c__14:
case OPT_std_gnu__14:
if (!preprocessing_asm_p)
set_std_cxx14 (code == OPT_std_c__14 );
break;
case OPT_std_c__17:
case OPT_std_gnu__17:
if (!preprocessing_asm_p)
set_std_cxx17 (code == OPT_std_c__17 );
break;
case OPT_std_c__2a:
case OPT_std_gnu__2a:
if (!preprocessing_asm_p)
set_std_cxx2a (code == OPT_std_c__2a );
break;
case OPT_std_c90:
case OPT_std_iso9899_199409:
if (!preprocessing_asm_p)
set_std_c89 (code == OPT_std_iso9899_199409 , true );
break;
case OPT_std_gnu90:
if (!preprocessing_asm_p)
set_std_c89 (false , false );
break;
case OPT_std_c99:
if (!preprocessing_asm_p)
set_std_c99 (true );
break;
case OPT_std_gnu99:
if (!preprocessing_asm_p)
set_std_c99 (false );
break;
case OPT_std_c11:
if (!preprocessing_asm_p)
set_std_c11 (true );
break;
case OPT_std_gnu11:
if (!preprocessing_asm_p)
set_std_c11 (false );
break;
case OPT_std_c17:
if (!preprocessing_asm_p)
set_std_c17 (true );
break;
case OPT_std_gnu17:
if (!preprocessing_asm_p)
set_std_c17 (false );
break;
case OPT_trigraphs:
cpp_opts->trigraphs = 1;
break;
case OPT_traditional_cpp:
cpp_opts->traditional = 1;
break;
case OPT_v:
verbose = true;
break;
}
switch (c_language)
{
case clk_c:
C_handle_option_auto (&global_options, &global_options_set, 
scode, arg, value, 
c_family_lang_mask, kind,
loc, handlers, global_dc);
break;
case clk_objc:
ObjC_handle_option_auto (&global_options, &global_options_set,
scode, arg, value, 
c_family_lang_mask, kind,
loc, handlers, global_dc);
break;
case clk_cxx:
CXX_handle_option_auto (&global_options, &global_options_set,
scode, arg, value,
c_family_lang_mask, kind,
loc, handlers, global_dc);
break;
case clk_objcxx:
ObjCXX_handle_option_auto (&global_options, &global_options_set,
scode, arg, value,
c_family_lang_mask, kind,
loc, handlers, global_dc);
break;
default:
gcc_unreachable ();
}
cpp_handle_option_auto (&global_options, scode, cpp_opts);
return result;
}
bool
default_handle_c_option (size_t code ATTRIBUTE_UNUSED,
const char *arg ATTRIBUTE_UNUSED,
int value ATTRIBUTE_UNUSED)
{
return false;
}
bool
c_common_post_options (const char **pfilename)
{
struct cpp_callbacks *cb;
if (in_fnames == NULL)
{
in_fnames = XNEWVEC (const char *, 1);
in_fnames[0] = "";
}
else if (strcmp (in_fnames[0], "-") == 0)
{
if (pch_file)
error ("cannot use %<-%> as input filename for a precompiled header");
in_fnames[0] = "";
}
if (out_fname == NULL || !strcmp (out_fname, "-"))
out_fname = "";
if (cpp_opts->deps.style == DEPS_NONE)
check_deps_environment_vars ();
handle_deferred_opts ();
sanitize_cpp_opts ();
register_include_chains (parse_in, sysroot, iprefix, imultilib,
std_inc, std_cxx_inc && c_dialect_cxx (), verbose);
#ifdef C_COMMON_OVERRIDE_OPTIONS
C_COMMON_OVERRIDE_OPTIONS;
#endif
if (c_dialect_cxx ())
{
if (flag_excess_precision_cmdline == EXCESS_PRECISION_STANDARD)
sorry ("-fexcess-precision=standard for C++");
flag_excess_precision_cmdline = EXCESS_PRECISION_FAST;
}
else if (flag_excess_precision_cmdline == EXCESS_PRECISION_DEFAULT)
flag_excess_precision_cmdline = (flag_iso
? EXCESS_PRECISION_STANDARD
: EXCESS_PRECISION_FAST);
if (flag_iso
&& !c_dialect_cxx ()
&& (global_options_set.x_flag_fp_contract_mode
== (enum fp_contract_mode) 0)
&& flag_unsafe_math_optimizations == 0)
flag_fp_contract_mode = FP_CONTRACT_OFF;
if (!flag_iso
&& !c_dialect_cxx ()
&& (global_options_set.x_flag_permitted_flt_eval_methods
== PERMITTED_FLT_EVAL_METHODS_DEFAULT))
flag_permitted_flt_eval_methods = PERMITTED_FLT_EVAL_METHODS_TS_18661;
else
flag_permitted_flt_eval_methods = PERMITTED_FLT_EVAL_METHODS_C11;
if (flag_gnu89_inline == -1)
flag_gnu89_inline = !flag_isoc99;
else if (!flag_gnu89_inline && !flag_isoc99)
error ("-fno-gnu89-inline is only supported in GNU99 or C99 mode");
if (flag_objc_sjlj_exceptions < 0)
flag_objc_sjlj_exceptions = flag_next_runtime;
if (flag_objc_exceptions && !flag_objc_sjlj_exceptions)
flag_exceptions = 1;
if (!global_options_set.x_flag_tree_loop_distribute_patterns
&& flag_no_builtin)
flag_tree_loop_distribute_patterns = 0;
if (c_dialect_cxx ())
warn_overlength_strings = 0;
if (warn_main == -1)
warn_main = (c_dialect_cxx () && flag_hosted) ? 1 : 0;
else if (warn_main == 2)
warn_main = flag_hosted ? 1 : 0;
if (warn_enum_compare == -1)
warn_enum_compare = c_dialect_cxx () ? 1 : 0;
if (warn_packed_bitfield_compat == -1)
warn_packed_bitfield_compat = 1;
if (!warn_format)
{
warning (OPT_Wformat_y2k,
"-Wformat-y2k ignored without -Wformat");
warning (OPT_Wformat_extra_args,
"-Wformat-extra-args ignored without -Wformat");
warning (OPT_Wformat_zero_length,
"-Wformat-zero-length ignored without -Wformat");
warning (OPT_Wformat_nonliteral,
"-Wformat-nonliteral ignored without -Wformat");
warning (OPT_Wformat_contains_nul,
"-Wformat-contains-nul ignored without -Wformat");
warning (OPT_Wformat_security,
"-Wformat-security ignored without -Wformat");
}
if (warn_implicit_function_declaration == -1)
warn_implicit_function_declaration = flag_isoc99;
if (warn_implicit_int == -1)
warn_implicit_int = flag_isoc99;
if (warn_shift_overflow == -1)
warn_shift_overflow = cxx_dialect >= cxx11 || flag_isoc99;
if (warn_shift_negative_value == -1)
warn_shift_negative_value = (extra_warnings
&& (cxx_dialect >= cxx11 || flag_isoc99));
if (!global_options_set.x_warn_register)
warn_register = cxx_dialect >= cxx17;
if (flag_declone_ctor_dtor == -1)
flag_declone_ctor_dtor = optimize_size;
if (flag_abi_compat_version == 1)
{
warning (0, "%<-fabi-compat-version=1%> is not supported, using =2");
flag_abi_compat_version = 2;
}
const int latest_abi_version = 13;
const int abi_compat_default = 11;
#define clamp(X) if (X == 0 || X > latest_abi_version) X = latest_abi_version
clamp (flag_abi_version);
clamp (warn_abi_version);
clamp (flag_abi_compat_version);
#undef clamp
if (warn_abi_version == -1 && flag_abi_compat_version != -1)
warn_abi_version = flag_abi_compat_version;
else if (flag_abi_compat_version == -1 && warn_abi_version != -1)
flag_abi_compat_version = warn_abi_version;
else if (warn_abi_version == -1 && flag_abi_compat_version == -1)
{
warn_abi_version = latest_abi_version;
if (flag_abi_version == latest_abi_version)
{
if (warning (OPT_Wabi, "-Wabi won't warn about anything"))
{
inform (input_location, "-Wabi warns about differences "
"from the most up-to-date ABI, which is also used "
"by default");
inform (input_location, "use e.g. -Wabi=11 to warn about "
"changes from GCC 7");
}
flag_abi_compat_version = abi_compat_default;
}
else
flag_abi_compat_version = latest_abi_version;
}
if (!global_options_set.x_flag_new_inheriting_ctors)
flag_new_inheriting_ctors = abi_version_at_least (11);
if (!global_options_set.x_flag_new_ttp)
flag_new_ttp = (cxx_dialect >= cxx17);
if (cxx_dialect >= cxx11)
{
warn_cxx11_compat = 0;
cpp_opts->cpp_warn_cxx11_compat = 0;
if (warn_narrowing == -1)
warn_narrowing = 1;
if (flag_iso && !global_options_set.x_flag_ext_numeric_literals)
cpp_opts->ext_numeric_literals = 0;
}
else if (warn_narrowing == -1)
warn_narrowing = 0;
if (c_dialect_cxx ()
&& flag_strong_eval_order == -1)
flag_strong_eval_order = (cxx_dialect >= cxx17 ? 2 : 1);
if (flag_sized_deallocation == -1)
flag_sized_deallocation = (cxx_dialect >= cxx14);
if (flag_extern_tls_init)
{
if (!TARGET_SUPPORTS_ALIASES || !SUPPORTS_WEAK)
{
if (flag_extern_tls_init > 0)
sorry ("external TLS initialization functions not supported "
"on this target");
flag_extern_tls_init = 0;
}
else
flag_extern_tls_init = 1;
}
if (warn_return_type == -1)
warn_return_type = c_dialect_cxx ();
if (num_in_fnames > 1)
error ("too many filenames given.  Type %s --help for usage",
progname);
if (flag_preprocess_only)
{
if (out_fname[0] == '\0')
out_stream = stdout;
else
out_stream = fopen (out_fname, "w");
if (out_stream == NULL)
{
fatal_error (input_location, "opening output file %s: %m", out_fname);
return false;
}
init_pp_output (out_stream);
}
else
{
init_c_lex ();
if (pch_file)
{
c_common_no_more_pch ();
if (write_symbols != NO_DEBUG && write_symbols != DWARF2_DEBUG)
warning (OPT_Wdeprecated,
"the \"%s\" debug format cannot be used with "
"pre-compiled headers", debug_type_names[write_symbols]);
}
else if (write_symbols != NO_DEBUG && write_symbols != DWARF2_DEBUG)
c_common_no_more_pch ();
input_location = UNKNOWN_LOCATION;
}
cb = cpp_get_callbacks (parse_in);
cb->file_change = cb_file_change;
cb->dir_change = cb_dir_change;
cpp_post_options (parse_in);
init_global_opts_from_cpp (&global_options, cpp_get_options (parse_in));
input_location = UNKNOWN_LOCATION;
*pfilename = this_input_filename
= cpp_read_main_file (parse_in, in_fnames[0]);
if (this_input_filename == NULL)
{
errorcount++;
return false;
}
if (flag_working_directory
&& flag_preprocess_only && !flag_no_line_commands)
pp_dir_change (parse_in, get_src_pwd ());
if (pch_file && flag_lto)
{
flag_lto = 0;
flag_generate_lto = 0;
}
return flag_preprocess_only;
}
bool
c_common_init (void)
{
cpp_opts->precision = TYPE_PRECISION (intmax_type_node);
cpp_opts->char_precision = TYPE_PRECISION (char_type_node);
cpp_opts->int_precision = TYPE_PRECISION (integer_type_node);
cpp_opts->wchar_precision = TYPE_PRECISION (wchar_type_node);
cpp_opts->unsigned_wchar = TYPE_UNSIGNED (wchar_type_node);
cpp_opts->bytes_big_endian = BYTES_BIG_ENDIAN;
cpp_init_iconv (parse_in);
if (version_flag)
{
int i;
fputs ("Compiler executable checksum: ", stderr);
for (i = 0; i < 16; i++)
fprintf (stderr, "%02x", executable_checksum[i]);
putc ('\n', stderr);
}
init_pragma ();
if (flag_preprocess_only)
{
c_finish_options ();
preprocess_file (parse_in);
return false;
}
return true;
}
void
c_common_parse_file (void)
{
unsigned int i;
i = 0;
for (;;)
{
c_finish_options ();
original_dump_file = dump_begin (TDI_original, &original_dump_flags);
pch_init ();
push_file_scope ();
c_parse_file ();
pop_file_scope ();
if (debug_hooks->start_end_main_source_file)
(*debug_hooks->end_source_file) (0);
if (++i >= num_in_fnames)
break;
cpp_undef_all (parse_in);
cpp_clear_file_cache (parse_in);
this_input_filename
= cpp_read_main_file (parse_in, in_fnames[i]);
if (original_dump_file)
{
dump_end (TDI_original, original_dump_file);
original_dump_file = NULL;
}
if (!this_input_filename)
break;
}
c_parse_final_cleanups ();
}
FILE *
get_dump_info (int phase, dump_flags_t *flags)
{
gcc_assert (phase == TDI_original);
*flags = original_dump_flags;
return original_dump_file;
}
void
c_common_finish (void)
{
FILE *deps_stream = NULL;
if (cpp_opts->deps.style != DEPS_NONE)
{
if (!deps_file)
deps_stream = out_stream;
else if (deps_file[0] == '-' && deps_file[1] == '\0')
deps_stream = stdout;
else
{
deps_stream = fopen (deps_file, deps_append ? "a": "w");
if (!deps_stream)
fatal_error (input_location, "opening dependency file %s: %m",
deps_file);
}
}
cpp_finish (parse_in, deps_stream);
if (deps_stream && deps_stream != out_stream && deps_stream != stdout
&& (ferror (deps_stream) || fclose (deps_stream)))
fatal_error (input_location, "closing dependency file %s: %m", deps_file);
if (out_stream && (ferror (out_stream) || fclose (out_stream)))
fatal_error (input_location, "when writing output to %s: %m", out_fname);
}
static void
check_deps_environment_vars (void)
{
char *spec;
spec = getenv ("DEPENDENCIES_OUTPUT");
if (spec)
cpp_opts->deps.style = DEPS_USER;
else
{
spec = getenv ("SUNPRO_DEPENDENCIES");
if (spec)
{
cpp_opts->deps.style = DEPS_SYSTEM;
cpp_opts->deps.ignore_main_file = true;
}
}
if (spec)
{
char *s = strchr (spec, ' ');
if (s)
{
defer_opt (OPT_MT, s + 1);
*s = '\0';
}
if (!deps_file)
deps_file = spec;
deps_append = 1;
deps_seen = true;
}
}
static void
handle_deferred_opts (void)
{
size_t i;
struct deps *deps;
if (!deps_seen)
return;
deps = cpp_get_deps (parse_in);
for (i = 0; i < deferred_count; i++)
{
struct deferred_opt *opt = &deferred_opts[i];
if (opt->code == OPT_MT || opt->code == OPT_MQ)
deps_add_target (deps, opt->arg, opt->code == OPT_MQ);
}
}
static void
sanitize_cpp_opts (void)
{
if (deps_seen && cpp_opts->deps.style == DEPS_NONE)
error ("to generate dependencies you must specify either -M or -MM");
if (flag_dump_macros == 'M')
flag_no_output = 1;
if (cpp_opts->directives_only && !cpp_opts->preprocessed && !flag_dump_macros)
flag_dump_macros = 'D';
if (flag_no_output)
{
if (flag_dump_macros != 'M')
flag_dump_macros = 0;
flag_dump_includes = 0;
flag_no_line_commands = 1;
}
else if (cpp_opts->deps.missing_files)
error ("-MG may only be used with -M or -MM");
cpp_opts->unsigned_char = !flag_signed_char;
cpp_opts->stdc_0_in_system_headers = STDC_0_IN_SYSTEM_HEADERS;
if (warn_long_long == -1)
{
warn_long_long = ((pedantic || warn_traditional)
&& (c_dialect_cxx () ? cxx_dialect == cxx98 : !flag_isoc99));
cpp_opts->cpp_warn_long_long = warn_long_long;
}
if (flag_working_directory == -1)
flag_working_directory = (debug_info_level != DINFO_LEVEL_NONE);
if (warn_implicit_fallthrough < 5)
cpp_opts->cpp_warn_implicit_fallthrough = warn_implicit_fallthrough;
else
cpp_opts->cpp_warn_implicit_fallthrough = 0;
if (cpp_opts->directives_only)
{
if (cpp_warn_unused_macros)
error ("-fdirectives-only is incompatible with -Wunused_macros");
if (cpp_opts->traditional)
error ("-fdirectives-only is incompatible with -traditional");
}
}
static void
add_prefixed_path (const char *suffix, incpath_kind chain)
{
char *path;
const char *prefix;
size_t prefix_len, suffix_len;
suffix_len = strlen (suffix);
prefix     = iprefix ? iprefix : cpp_GCC_INCLUDE_DIR;
prefix_len = iprefix ? strlen (iprefix) : cpp_GCC_INCLUDE_DIR_len;
path = (char *) xmalloc (prefix_len + suffix_len + 1);
memcpy (path, prefix, prefix_len);
memcpy (path + prefix_len, suffix, suffix_len);
path[prefix_len + suffix_len] = '\0';
add_path (path, chain, 0, false);
}
static void
c_finish_options (void)
{
if (!cpp_opts->preprocessed)
{
size_t i;
cb_file_change (parse_in,
linemap_check_ordinary (linemap_add (line_table,
LC_RENAME, 0,
_("<built-in>"),
0)));
source_location builtins_loc = BUILTINS_LOCATION;
cpp_force_token_locations (parse_in, &builtins_loc);
cpp_init_builtins (parse_in, flag_hosted);
c_cpp_builtins (parse_in);
cpp_stop_forcing_token_locations (parse_in);
cpp_opts->warn_dollars = (cpp_opts->cpp_pedantic && !cpp_opts->c99);
cb_file_change (parse_in,
linemap_check_ordinary (linemap_add (line_table, LC_RENAME, 0,
_("<command-line>"), 0)));
for (i = 0; i < deferred_count; i++)
{
struct deferred_opt *opt = &deferred_opts[i];
if (opt->code == OPT_D)
cpp_define (parse_in, opt->arg);
else if (opt->code == OPT_U)
cpp_undef (parse_in, opt->arg);
else if (opt->code == OPT_A)
{
if (opt->arg[0] == '-')
cpp_unassert (parse_in, opt->arg + 1);
else
cpp_assert (parse_in, opt->arg);
}
}
if (debug_hooks->start_end_main_source_file
&& !flag_preprocess_only)
(*debug_hooks->start_source_file) (0, this_input_filename);
for (i = 0; i < deferred_count; i++)
{
struct deferred_opt *opt = &deferred_opts[i];
if (opt->code == OPT_imacros
&& cpp_push_include (parse_in, opt->arg))
{
include_cursor = deferred_count + 1;
cpp_scan_nooutput (parse_in);
}
}
}
else
{
if (cpp_opts->directives_only)
cpp_init_special_builtins (parse_in);
if (debug_hooks->start_end_main_source_file
&& !flag_preprocess_only)
(*debug_hooks->start_source_file) (0, this_input_filename);
}
include_cursor = 0;
push_command_line_include ();
}
static void
push_command_line_include (void)
{
if (include_cursor > deferred_count)
return;
if (!done_preinclude)
{
done_preinclude = true;
if (flag_hosted && std_inc && !cpp_opts->preprocessed)
{
const char *preinc = targetcm.c_preinclude ();
if (preinc && cpp_push_default_include (parse_in, preinc))
return;
}
}
pch_cpp_save_state ();
while (include_cursor < deferred_count)
{
struct deferred_opt *opt = &deferred_opts[include_cursor++];
if (!cpp_opts->preprocessed && opt->code == OPT_include
&& cpp_push_include (parse_in, opt->arg))
return;
}
if (include_cursor == deferred_count)
{
include_cursor++;
cpp_opts->warn_unused_macros = cpp_warn_unused_macros;
if (!cpp_opts->preprocessed)
cpp_change_file (parse_in, LC_RENAME, this_input_filename);
line_table->trace_includes = cpp_opts->print_include_names;
}
}
static void
cb_file_change (cpp_reader * ARG_UNUSED (pfile),
const line_map_ordinary *new_map)
{
if (flag_preprocess_only)
pp_file_change (new_map);
else
fe_file_change (new_map);
if (new_map 
&& (new_map->reason == LC_ENTER || new_map->reason == LC_RENAME))
{
invoke_plugin_callbacks 
(PLUGIN_INCLUDE_FILE,
const_cast<char*> (ORDINARY_MAP_FILE_NAME (new_map)));
}
if (new_map == 0 || (new_map->reason == LC_LEAVE && MAIN_FILE_P (new_map)))
{
pch_cpp_save_state ();
push_command_line_include ();
}
}
void
cb_dir_change (cpp_reader * ARG_UNUSED (pfile), const char *dir)
{
if (!set_src_pwd (dir))
warning (0, "too late for # directive to set debug directory");
}
static void
set_std_c89 (int c94, int iso)
{
cpp_set_lang (parse_in, c94 ? CLK_STDC94: iso ? CLK_STDC89: CLK_GNUC89);
flag_iso = iso;
flag_no_asm = iso;
flag_no_gnu_keywords = iso;
flag_no_nonansi_builtin = iso;
flag_isoc94 = c94;
flag_isoc99 = 0;
flag_isoc11 = 0;
lang_hooks.name = "GNU C89";
}
static void
set_std_c99 (int iso)
{
cpp_set_lang (parse_in, iso ? CLK_STDC99: CLK_GNUC99);
flag_no_asm = iso;
flag_no_nonansi_builtin = iso;
flag_iso = iso;
flag_isoc11 = 0;
flag_isoc99 = 1;
flag_isoc94 = 1;
lang_hooks.name = "GNU C99";
}
static void
set_std_c11 (int iso)
{
cpp_set_lang (parse_in, iso ? CLK_STDC11: CLK_GNUC11);
flag_no_asm = iso;
flag_no_nonansi_builtin = iso;
flag_iso = iso;
flag_isoc11 = 1;
flag_isoc99 = 1;
flag_isoc94 = 1;
lang_hooks.name = "GNU C11";
}
static void
set_std_c17 (int iso)
{
cpp_set_lang (parse_in, iso ? CLK_STDC17: CLK_GNUC17);
flag_no_asm = iso;
flag_no_nonansi_builtin = iso;
flag_iso = iso;
flag_isoc11 = 1;
flag_isoc99 = 1;
flag_isoc94 = 1;
lang_hooks.name = "GNU C17";
}
static void
set_std_cxx98 (int iso)
{
cpp_set_lang (parse_in, iso ? CLK_CXX98: CLK_GNUCXX);
flag_no_gnu_keywords = iso;
flag_no_nonansi_builtin = iso;
flag_iso = iso;
flag_isoc94 = 0;
flag_isoc99 = 0;
cxx_dialect = cxx98;
lang_hooks.name = "GNU C++98";
}
static void
set_std_cxx11 (int iso)
{
cpp_set_lang (parse_in, iso ? CLK_CXX11: CLK_GNUCXX11);
flag_no_gnu_keywords = iso;
flag_no_nonansi_builtin = iso;
flag_iso = iso;
flag_isoc94 = 1;
flag_isoc99 = 1;
cxx_dialect = cxx11;
lang_hooks.name = "GNU C++11";
}
static void
set_std_cxx14 (int iso)
{
cpp_set_lang (parse_in, iso ? CLK_CXX14: CLK_GNUCXX14);
flag_no_gnu_keywords = iso;
flag_no_nonansi_builtin = iso;
flag_iso = iso;
flag_isoc94 = 1;
flag_isoc99 = 1;
cxx_dialect = cxx14;
lang_hooks.name = "GNU C++14";
}
static void
set_std_cxx17 (int iso)
{
cpp_set_lang (parse_in, iso ? CLK_CXX17: CLK_GNUCXX17);
flag_no_gnu_keywords = iso;
flag_no_nonansi_builtin = iso;
flag_iso = iso;
flag_isoc94 = 1;
flag_isoc99 = 1;
flag_isoc11 = 1;
cxx_dialect = cxx17;
lang_hooks.name = "GNU C++17";
}
static void
set_std_cxx2a (int iso)
{
cpp_set_lang (parse_in, iso ? CLK_CXX2A: CLK_GNUCXX2A);
flag_no_gnu_keywords = iso;
flag_no_nonansi_builtin = iso;
flag_iso = iso;
flag_isoc94 = 1;
flag_isoc99 = 1;
flag_isoc11 = 1;
cxx_dialect = cxx2a;
lang_hooks.name = "GNU C++17"; 
}
static void
handle_OPT_d (const char *arg)
{
char c;
while ((c = *arg++) != '\0')
switch (c)
{
case 'M':			
case 'N':			
case 'D':			
case 'U':			
flag_dump_macros = c;
break;
case 'I':
flag_dump_includes = 1;
break;
}
}
