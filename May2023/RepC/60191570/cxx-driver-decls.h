#ifndef CXX_DRIVER_DECLS_H
#define CXX_DRIVER_DECLS_H
#include "cxx-macros.h"
#include "cxx-scope-decls.h"
#include "cxx-buildscope-decls.h"
#include "cxx-nodecl-decls.h"
#include "fortran03-typeenviron-decls.h"
#include <stddef.h>
MCXX_BEGIN_DECLS
typedef enum codegen_parameter_tag
{
CODEGEN_PARAM_NONTYPE_TEMPLATE_ARGUMENT = 0,
}
codegen_parameter_t;
#define BITMAP(X) (1<<X)
typedef enum source_kind_tag
{
SOURCE_KIND_UNKNOWN = 0,
SOURCE_KIND_NOT_PREPROCESSED = BITMAP(0),
SOURCE_KIND_PREPROCESSED = BITMAP(1),
SOURCE_KIND_FIXED_FORM = BITMAP(2),
SOURCE_KIND_FREE_FORM = BITMAP(3),
SOURCE_KIND_DO_NOT_PROCESS = BITMAP(4),
SOURCE_KIND_DO_NOT_COMPILE = BITMAP(5),
SOURCE_KIND_DO_NOT_LINK = BITMAP(6),
SOURCE_KIND_DO_NOT_EMBED = BITMAP(7),
} source_kind_t;
#undef BITMAP
typedef enum source_language_tag
{
SOURCE_LANGUAGE_UNKNOWN     = 0,
SOURCE_LANGUAGE_C           = 1,
SOURCE_LANGUAGE_CXX         = 2,
SOURCE_LANGUAGE_FORTRAN     = 3,
SOURCE_LANGUAGE_ASSEMBLER   = 4,
SOURCE_LANGUAGE_LINKER_DATA = 5,
SOURCE_IS_SUBLANGUAGE       = 16,
SOURCE_SUBLANGUAGE_CUDA     = (SOURCE_IS_SUBLANGUAGE | 1),
SOURCE_SUBLANGUAGE_OPENCL   = (SOURCE_IS_SUBLANGUAGE | 2),
} source_language_t;
#define NATIVE_VENDORS_LIST \
NATIVE_VENDOR(GNU, gnu) \
NATIVE_VENDOR(INTEL, intel) \
NATIVE_VENDOR(IBM, ibm) \
NATIVE_VENDOR(NVIDIA, nvidia) \
NATIVE_VENDOR(CRAY, cray) \
NATIVE_VENDOR(PGI, pgi) \
typedef enum native_vendor_tag
{
NATIVE_VENDOR_UNKNOWN = 0,
#define NATIVE_VENDOR(NAME, STR) NATIVE_VENDOR_##NAME,
NATIVE_VENDORS_LIST
#undef NATIVE_VENDOR
} native_vendor_t;
typedef struct sublanguage_profile_tag
{
source_language_t sublanguage;
const char* profile;
} sublanguage_profile_t;
extern sublanguage_profile_t sublanguage_profiles[];
extern const char* source_language_names[];
struct extensions_table_t
{
const char* name;
source_language_t source_language;
source_kind_t source_kind;
};
typedef struct include_tag
{
const char *included_file;
char system_include;
} include_t;
typedef struct module_to_wrap_info_tag
{
const char* module_name;
const char* native_file;
const char* mercurium_file;
} module_to_wrap_info_t;
typedef struct translation_unit_tag
{
const char* input_filename;
const char* output_filename;
struct AST_tag* parsed_tree;
nodecl_t nodecl;
const decl_context_t* global_decl_context;
int num_includes;
include_t **include_list;
rb_red_blk_tree *module_file_cache;
int num_modules_to_wrap;
module_to_wrap_info_t** modules_to_wrap;
int num_module_files_to_hide;
const char** module_files_to_hide;
void *dto;
} translation_unit_t;
struct compilation_configuration_tag;
struct configuration_directive_t
{
const char* name;
int (*funct)(struct compilation_configuration_tag*, const char* index, const char* value);
};
struct debug_flags_list_t
{
const char* name;
size_t flag_offset;
const char* description;
};
typedef struct debug_options_tag
{
char abort_on_ice;
char abort_on_error_message;
char backtrace_on_ice;
char print_scope;
char enable_debug_code;
char debug_lexer;
char debug_parser;
char print_nodecl_graphviz;
char print_nodecl_html;
char print_ast_graphviz;
char print_memory_report;
char print_memory_report_in_bytes;
char debug_sizeof;
char do_not_run_gdb;
char binary_check;
char analysis_verbose;
char ranges_verbose;
char tdg_verbose;
char analysis_perf;
char analysis_info;
char print_pcfg;
char print_pcfg_w_context;
char print_pcfg_w_analysis;
char print_pcfg_full;
char print_tdg;
char tdg_to_json;
char print_dt;
char do_not_codegen;
char show_template_packs;
char vectorization_verbose;
char stats_string_table;
} debug_options_t;
extern debug_options_t debug_options;
typedef struct external_var_tag {
const char* name;
const char* value;
} external_var_t;
typedef enum pragma_directive_kind_tag
{
PDK_NONE = 0,
PDK_DIRECTIVE,
PDK_CONSTRUCT,
PDK_CONSTRUCT_NOEND
} pragma_directive_kind_t;
typedef struct pragma_directive_set_tag
{
int num_directives;
const char **directive_names;
pragma_directive_kind_t *directive_kinds;
} pragma_directive_set_t;
struct compilation_file_process_tag;
typedef
enum parameter_flag_value_tag
{
PFV_UNDEFINED = 0,
PFV_FALSE = 1,
PFV_TRUE = 2,
} parameter_flag_value_t;
typedef struct parameter_flags_tag
{
const char *name;
parameter_flag_value_t value;
} parameter_flags_t;
typedef struct subgoal_tag
{
const char* linked_subgoal_filename;
struct compilation_configuration_tag* configuration;
} subgoal_t;
typedef struct compilation_process_tag
{
int execution_result;
const char *config_dir;
struct compilation_file_process_tag** translation_units;
int num_translation_units;
const char *linked_output_filename;
int num_subgoals;
subgoal_t *subgoals;
int argc;
const char** argv;
int original_argc;
const char** original_argv;
const char* exec_basename;
const char* home_directory;
int num_configurations;
struct compilation_configuration_tag** configuration_set;
int num_parameter_flags;
parameter_flags_t **parameter_flags;
struct compilation_configuration_tag *command_line_configuration;
struct compilation_file_process_tag* current_file_process;
struct compilation_configuration_tag *current_compilation_configuration;
char parallel_process; 
} compilation_process_t;
typedef struct compilation_configuration_conditional_flags
{
const char *flag;
char value;
} compilation_configuration_conditional_flags_t;
typedef struct flag_expr_tag flag_expr_t;
typedef struct compilation_configuration_line
{
const char *name;
const char *index;
const char *value;
flag_expr_t* flag_expr;
const char *filename;
int line;
} compilation_configuration_line_t;
#if 0
typedef struct embed_map_tag
{
const char* profile;
const char* command;
} embed_map_t;
typedef struct identifier_map_tag
{
const char* profile;
const char* action;
} identifier_map_t;
#endif
typedef struct target_options_map_tag
{
const char* profile;
char do_sublink;
char do_embedding;
enum
{
EMBEDDING_MODE_INVALID = 0,
EMBEDDING_MODE_BFD = 1,
EMBEDDING_MODE_PARTIAL_LINKING = 2,
} embedding_mode;
char do_combining;
enum
{
COMBINING_MODE_INVALID = 0,
COMBINING_MODE_SPU_ELF,
COMBINING_MODE_INCBIN,
} combining_mode;
} target_options_map_t;
typedef struct code_shape_tag
{
char short_enums;
} code_shape_t;
typedef struct compiler_phase_loader_tag compiler_phase_loader_t;
typedef struct parameter_linker_command_tag
{
const char *argument;
translation_unit_t *translation_unit;
} parameter_linker_command_t;
typedef const char* (*print_vector_type_fun)(const decl_context_t*, type_t*, print_symbol_callback_t, void*);
typedef struct compilation_configuration_tag
{
const char *configuration_name;
struct compilation_configuration_tag* base_configuration;
int num_configuration_lines;
struct compilation_configuration_line ** configuration_lines;
char verbose;
char keep_files;
char keep_temporaries;
char do_not_process_files;
char do_not_parse;
char do_not_prettyprint;
char do_not_compile;
char do_not_link;
char generate_assembler;
char enable_openmp;
char force_language;
char warnings_as_errors;
source_language_t source_language;
const char* preprocessor_name;
const char** preprocessor_options;
char preprocessor_uses_stdout;
const char* fortran_preprocessor_name;
const char** fortran_preprocessor_options;
const char* prescanner_name;
const char** prescanner_options;
int input_column_width;
int output_column_width;
int num_disabled_intrinsics;
const char ** disabled_intrinsics_list;
char do_not_wrap_fortran_modules;
int num_module_dirs;
const char** module_dirs;
const char* module_out_dir;
const char* module_out_pattern;
const char* module_native_dir;
const char* lock_dir;
char disable_locking;
const char* multifile_dir;
int default_integer_kind;
int default_real_kind;
int default_logical_kind;
int default_character_kind;
int doubleprecision_kind;
source_kind_t force_source_kind;
const char* native_compiler_name;
const char** native_compiler_options;
const char* linker_name;
int num_args_linker_command;
parameter_linker_command_t** linker_command;
const char** linker_options;
const char** linker_options_pre;
const char** linker_options_post;
const char* target_objcopy;
const char* target_objdump;
const char* target_ld;
const char* target_ar;
const char* output_directory;
int num_include_dirs;
const char** include_dirs;
int num_compiler_phases;
compiler_phase_loader_t** phase_loader;
void* codegen_phase;
char phases_loaded;
int num_external_vars;
external_var_t** external_vars;
int num_pragma_custom_prefix;
const char** pragma_custom_prefix;
pragma_directive_set_t **pragma_custom_prefix_info;
char disable_gxx_type_traits;
char enable_ms_builtin_types;
char enable_intel_builtins_syntax;
char enable_intel_intrinsics;
char enable_intel_vector_types;
char explicit_instantiation;
char disable_sizeof;
char disable_float128_token;
char pass_through;
struct type_environment_tag* type_environment;
struct fortran_array_descriptor_t* fortran_array_descriptor;
struct fortran_name_mangling_t* fortran_name_mangling;
code_shape_t code_shape;
char enable_upc;
const char *upc_threads;
char explicit_std_version;
char enable_c11;
char enable_cxx11;
char enable_cxx14;
char enable_f03;
char enable_f08;
char enable_cuda;
char enable_opencl;
const char* opencl_build_options;
int num_target_option_maps;
target_options_map_t** target_options_maps;
char disable_empty_sentinels;
print_vector_type_fun print_vector_type;
print_vector_type_fun print_mask_type;
char preserve_parentheses;
char fortran_no_whole_file;
native_vendor_t native_vendor;
char line_markers;
int num_errors;
const char** error_messages;
} compilation_configuration_t;
struct compiler_phase_loader_tag
{
void (*func)(compilation_configuration_t* compilation_configuration, const char* data);
const char* data;
};
typedef struct compilation_file_process_tag
{
translation_unit_t *translation_unit;
compilation_configuration_t *compilation_configuration;
int tag; 
char already_compiled;
int num_secondary_translation_units;
struct compilation_file_process_tag **secondary_translation_units;
} compilation_file_process_t;
#define CURRENT_CONFIGURATION ((compilation_configuration_t*)compilation_process.current_compilation_configuration)
#define CURRENT_COMPILED_FILE ((translation_unit_t*)compilation_process.current_file_process->translation_unit)
#define CURRENT_FILE_PROCESS ((compilation_file_process_t*)compilation_process.current_file_process)
#define SET_CURRENT_FILE_PROCESS(_x) (compilation_process.current_file_process = _x)
#define SET_CURRENT_CONFIGURATION(_x) (compilation_process.current_compilation_configuration = _x)
MCXX_END_DECLS
#endif 
