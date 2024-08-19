#include "cxx-driver-utils.h"
#include "cxx-process.h"
#include "cxx-profile.h"
#include "cxx-target-tools.h"
#include <string.h>
#include <stdlib.h>
#include "mem.h"
#include "uniquestr.h"
#include <stdio.h>
compilation_configuration_t* new_compilation_configuration(
const char* name,
compilation_configuration_t* base)
{
compilation_configuration_t* result = NEW0(compilation_configuration_t);
result->configuration_name = uniquestr(name);
result->base_configuration = base;
result->target_objcopy = TARGET_OBJCOPY;
result->target_objdump = TARGET_OBJDUMP;
result->target_ld = TARGET_LD;
result->target_ar = TARGET_AR;
result->input_column_width = 72;
result->output_column_width = 132;
return result;
}
static void initialize_with_base_config(compilation_configuration_t* dst, 
compilation_configuration_t const* base);
void initialize_with_base(compilation_configuration_t* config)
{
if (config->base_configuration != NULL)
{
initialize_with_base_config(config, config->base_configuration);
}
}
static const char** copy_null_ended_const_char_array(const char** orig)
{
const char** result = NULL;
if (orig == NULL)
{
return result;
}
int num = count_null_ended_array((void**)orig);
result = NEW_VEC(const char*, num + 1);
int i;
for (i = 0; i < num; i++)
{
result[i] = orig[i];
}
result[i] = NULL;
return result;
}
#define DEF_COPY_ARRAY(_fun_name, element_type) \
static element_type** _fun_name(element_type** orig, int num_elems) \
{ \
element_type** result = NEW_VEC0(element_type*, num_elems); \
int i; \
for (i = 0; i < num_elems; i++) \
{ \
result[i] = orig[i]; \
} \
return result; \
}
DEF_COPY_ARRAY(copy_external_vars, external_var_t)
DEF_COPY_ARRAY(copy_pragma_directive_set, pragma_directive_set_t)
DEF_COPY_ARRAY(copy_compiler_phase_loader, compiler_phase_loader_t)
DEF_COPY_ARRAY(copy_pragma_custom_prefix, const char)
DEF_COPY_ARRAY(copy_target_options_maps, target_options_map_t)
DEF_COPY_ARRAY(copy_error_messages, const char)
static void initialize_with_base_config(compilation_configuration_t* dst, 
compilation_configuration_t const * base)
{
const char* configuration_name = dst->configuration_name;
int num_configuration_lines = dst->num_configuration_lines;
struct compilation_configuration_line ** configuration_lines = dst->configuration_lines;
*dst = *base;
dst->configuration_name = configuration_name;
dst->num_configuration_lines = num_configuration_lines;
dst->configuration_lines = configuration_lines;
dst->prescanner_options = copy_null_ended_const_char_array(base->prescanner_options);
dst->preprocessor_options = copy_null_ended_const_char_array(base->preprocessor_options);
dst->fortran_preprocessor_options = copy_null_ended_const_char_array(base->fortran_preprocessor_options);
dst->native_compiler_options = copy_null_ended_const_char_array(base->native_compiler_options);
dst->linker_options = copy_null_ended_const_char_array(base->linker_options);
dst->phase_loader = copy_compiler_phase_loader(base->phase_loader, 
base->num_compiler_phases);
dst->external_vars = copy_external_vars(base->external_vars, 
base->num_external_vars);
dst->pragma_custom_prefix = copy_pragma_custom_prefix(base->pragma_custom_prefix, 
base->num_pragma_custom_prefix);
dst->pragma_custom_prefix_info = copy_pragma_directive_set(base->pragma_custom_prefix_info, 
base->num_pragma_custom_prefix);
dst->target_options_maps = copy_target_options_maps(base->target_options_maps,
base->num_target_option_maps);
dst->error_messages = copy_error_messages(base->error_messages, base->num_errors);
}
compilation_configuration_t* get_compilation_configuration(const char* config_name)
{
int i;
for (i = 0; i < compilation_process.num_configurations; i++)
{
if (strcmp(compilation_process.configuration_set[i]->configuration_name, config_name) == 0)
{
return compilation_process.configuration_set[i];
}
}
return NULL;
}
