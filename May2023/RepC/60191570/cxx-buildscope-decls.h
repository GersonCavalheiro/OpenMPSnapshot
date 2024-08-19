#ifndef CXX_BUILDSCOPE_DECLS_H
#define CXX_BUILDSCOPE_DECLS_H
#include <stdbool.h>
#include "cxx-macros.h"
#include "cxx-scope-decls.h"
#include "cxx-gccsupport-decls.h"
#include "cxx-limits.h"
MCXX_BEGIN_DECLS
typedef
struct arguments_info_tag
{
scope_entry_t* entry;
nodecl_t argument;
const decl_context_t* context;
} arguments_info_t;
typedef 
struct gather_decl_spec_tag {
bool no_declarators:1;
bool parameter_declaration:1;
bool is_template:1;
bool is_explicit_specialization:1; 
bool is_explicit_instantiation:1;  
bool inside_class_specifier:1;
bool is_auto_storage:1;
bool is_auto_type:1;
bool is_decltype_auto:1;
bool is_register:1;
bool is_static:1;
bool is_extern:1;
bool is_mutable:1;
bool is_thread:1;
bool is_thread_local:1;
bool is_friend:1;
bool is_typedef:1;
bool is_signed:1;
bool is_unsigned:1;
bool is_short:1;
bool is_const:1;
bool is_volatile:1;
bool is_restrict:1;
bool is_inline:1;
bool is_virtual:1;
bool is_explicit:1;
bool is_complex:1;
bool is_overriden_type:1;
bool emit_always:1;
bool any_exception:1; 
bool is_vector:1;
bool is_final:1;
bool is_hides_member:1;
bool is_override:1;
bool is_constexpr:1;
bool is_atomic:1;
bool is_noreturn:1;
bool is_transparent_union:1;
bool is_boolean_integer;
bool is_mask_integer;
bool is_mcc_hidden;
bool is_cxx_new_declarator;
bool allow_class_template_names;
unsigned int is_long:2;
scope_entry_t* defined_type;
struct type_tag* mode_type;
int num_exceptions;
struct type_tag** exceptions;
nodecl_t noexception;
unsigned int vector_size;
int num_arguments_info;
arguments_info_t* arguments_info;
int num_vla_dimension_symbols;
scope_entry_t** vla_dimension_symbols;
int num_gcc_attributes;
gcc_attribute_t* gcc_attributes;
int num_ms_attributes;
gcc_attribute_t* ms_attributes;
struct
{
bool is_shared:1;
bool is_relaxed:1;
bool is_strict:1;
AST shared_layout;
} upc;
struct
{
bool is_global:1;
bool is_device:1;
bool is_host:1;
bool is_shared:1;
bool is_constant:1;
} cuda;
struct
{
bool is_kernel:1;
bool is_constant:1;
bool is_global:1;
bool is_local:1;
} opencl;
access_specifier_t current_access;
AST gcc_asm_spec;
int num_xl_pragmas;
const char** xl_pragmas;
nodecl_t alignas_list;
} gather_decl_spec_t;
typedef
struct gather_decl_spec_list_tag
{
int num_items;
gather_decl_spec_t* items;
} gather_decl_spec_list_t;
MCXX_END_DECLS
#endif 
