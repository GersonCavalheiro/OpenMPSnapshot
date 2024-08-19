#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "target.h"
#include "function.h"
#include "brig-to-generic.h"
#include "stringpool.h"
#include "tree-iterator.h"
#include "toplev.h"
#include "gimplify.h"
#include "gimple-expr.h"
#include "print-tree.h"
#include "hsa-brig-format.h"
#include "stor-layout.h"
#include "diagnostic-core.h"
#include "brig-code-entry-handler.h"
#include "brig-machine.h"
#include "brig-util.h"
#include "phsa.h"
#include "tree-pretty-print.h"
#include "dumpfile.h"
#include "profile-count.h"
#include "tree-cfg.h"
#include "errors.h"
#include "fold-const.h"
#include "cgraph.h"
#include "dumpfile.h"
#include "tree-pretty-print.h"
extern int gccbrig_verbose;
tree brig_to_generic::s_fp16_type;
tree brig_to_generic::s_fp32_type;
tree brig_to_generic::s_fp64_type;
brig_to_generic::brig_to_generic ()
: m_cf (NULL), m_analyzing (true), m_total_group_segment_usage (0),
m_brig (NULL), m_next_private_offset (0)
{
m_globals = NULL_TREE;
s_fp16_type = make_node (REAL_TYPE);
TYPE_PRECISION (s_fp16_type) = 16;
TYPE_SIZE (s_fp16_type) = bitsize_int (16);
TYPE_SIZE_UNIT (s_fp16_type) = size_int (2);
SET_TYPE_ALIGN (s_fp16_type, 16);
layout_type (s_fp16_type);
s_fp32_type = gccbrig_tree_type_for_hsa_type (BRIG_TYPE_F32);
s_fp64_type = gccbrig_tree_type_for_hsa_type (BRIG_TYPE_F64);
m_default_float_rounding_mode = BRIG_ROUND_FLOAT_ZERO;
m_dump_file = dump_begin (TDI_original, &m_dump_flags);
}
class unimplemented_entry_handler : public brig_code_entry_handler
{
public:
unimplemented_entry_handler (brig_to_generic &parent)
: brig_code_entry_handler (parent)
{
}
size_t
operator () (const BrigBase *base)
{
gcc_unreachable ();
return base->byteCount;
}
};
class skipped_entry_handler : public brig_code_entry_handler
{
public:
skipped_entry_handler (brig_to_generic &parent)
: brig_code_entry_handler (parent)
{
}
size_t
operator () (const BrigBase *base)
{
return base->byteCount;
}
};
class brig_reg_use_analyzer : public brig_code_entry_handler
{
public:
brig_reg_use_analyzer (brig_to_generic &parent)
: brig_code_entry_handler (parent)
{
}
size_t
operator () (const BrigBase *base)
{
const BrigInstBase *brig_inst = (const BrigInstBase *) base;
analyze_operands (*brig_inst);
return base->byteCount;
}
};
struct code_entry_handler_info
{
BrigKind kind;
brig_code_entry_handler *handler;
};
void
brig_to_generic::find_brig_sections ()
{
m_data = m_code = m_operand = NULL;
const BrigModuleHeader *mheader = (const BrigModuleHeader *) m_brig;
for (uint32_t sec = 0; sec < mheader->sectionCount; ++sec)
{
uint64_t offset
= ((const uint64_t *) (m_brig + mheader->sectionIndex))[sec];
const BrigSectionHeader *section_header
= (const BrigSectionHeader *) (m_brig + offset);
std::string name ((const char *) (&section_header->name),
section_header->nameLength);
if (sec == BRIG_SECTION_INDEX_DATA && name == "hsa_data")
{
m_data = (const char *) section_header;
m_data_size = section_header->byteCount;
}
else if (sec == BRIG_SECTION_INDEX_CODE && name == "hsa_code")
{
m_code = (const char *) section_header;
m_code_size = section_header->byteCount;
}
else if (sec == BRIG_SECTION_INDEX_OPERAND && name == "hsa_operand")
{
m_operand = (const char *) section_header;
m_operand_size = section_header->byteCount;
}
else
{
gcc_unreachable ();
}
}
if (m_code == NULL)
gcc_unreachable ();
if (m_data == NULL)
gcc_unreachable ();
if (m_operand == NULL)
gcc_unreachable ();
}
void
brig_to_generic::analyze (const char *brig_blob)
{
const BrigModuleHeader *mheader = (const BrigModuleHeader *) brig_blob;
if (strncmp (mheader->identification, "HSA BRIG", 8) != 0)
fatal_error (UNKNOWN_LOCATION, PHSA_ERROR_PREFIX_INCOMPATIBLE_MODULE
"Unrecognized file format.");
if (mheader->brigMajor != 1 || mheader->brigMinor != 0)
fatal_error (UNKNOWN_LOCATION, PHSA_ERROR_PREFIX_INCOMPATIBLE_MODULE
"BRIG version not supported. BRIG 1.0 required.");
m_brig = brig_blob;
find_brig_sections ();
brig_directive_variable_handler var_handler (*this);
brig_directive_fbarrier_handler fbar_handler (*this);
brig_directive_function_handler func_handler (*this);
brig_reg_use_analyzer reg_use_analyzer (*this);
brig_directive_module_handler module_handler (*this);
skipped_entry_handler skipped_handler (*this);
const BrigSectionHeader *csection_header = (const BrigSectionHeader *) m_code;
code_entry_handler_info handlers[]
= {{BRIG_KIND_INST_BASIC, &reg_use_analyzer},
{BRIG_KIND_INST_MOD, &reg_use_analyzer},
{BRIG_KIND_INST_CMP, &reg_use_analyzer},
{BRIG_KIND_INST_MEM, &reg_use_analyzer},
{BRIG_KIND_INST_CVT, &reg_use_analyzer},
{BRIG_KIND_INST_SEG_CVT, &reg_use_analyzer},
{BRIG_KIND_INST_SEG, &reg_use_analyzer},
{BRIG_KIND_INST_ADDR, &reg_use_analyzer},
{BRIG_KIND_INST_SOURCE_TYPE, &reg_use_analyzer},
{BRIG_KIND_INST_ATOMIC, &reg_use_analyzer},
{BRIG_KIND_INST_SIGNAL, &reg_use_analyzer},
{BRIG_KIND_INST_BR, &reg_use_analyzer},
{BRIG_KIND_INST_LANE, &reg_use_analyzer},
{BRIG_KIND_INST_QUEUE, &reg_use_analyzer},
{BRIG_KIND_DIRECTIVE_VARIABLE, &var_handler},
{BRIG_KIND_DIRECTIVE_FBARRIER, &fbar_handler},
{BRIG_KIND_DIRECTIVE_KERNEL, &func_handler},
{BRIG_KIND_DIRECTIVE_MODULE, &module_handler},
{BRIG_KIND_DIRECTIVE_FUNCTION, &func_handler}};
m_analyzing = true;
for (size_t b = csection_header->headerByteCount; b < m_code_size;)
{
const BrigBase *entry = (const BrigBase *) (m_code + b);
brig_code_entry_handler *handler = &skipped_handler;
if (m_cf != NULL && b >= m_cf->m_brig_def->nextModuleEntry)
{
m_total_group_segment_usage += m_cf->m_local_group_variables.size ();
delete m_cf;
m_cf = NULL;
}
for (size_t i = 0;
i < sizeof (handlers) / sizeof (code_entry_handler_info); ++i)
{
if (handlers[i].kind == entry->kind)
handler = handlers[i].handler;
}
int bytes_processed = (*handler) (entry);
if (bytes_processed == 0)
fatal_error (UNKNOWN_LOCATION, PHSA_ERROR_PREFIX_CORRUPTED_MODULE
"Element with 0 bytes.");
b += bytes_processed;
}
if (m_cf != NULL)
{
m_total_group_segment_usage += m_cf->m_local_group_variables.size ();
delete m_cf;
m_cf = NULL;
}
m_total_group_segment_usage += m_module_group_variables.size ();
m_analyzing = false;
}
void
brig_to_generic::parse (const char *brig_blob)
{
m_brig = brig_blob;
find_brig_sections ();
brig_basic_inst_handler inst_handler (*this);
brig_branch_inst_handler branch_inst_handler (*this);
brig_cvt_inst_handler cvt_inst_handler (*this);
brig_seg_inst_handler seg_inst_handler (*this);
brig_copy_move_inst_handler copy_move_inst_handler (*this);
brig_signal_inst_handler signal_inst_handler (*this);
brig_atomic_inst_handler atomic_inst_handler (*this);
brig_cmp_inst_handler cmp_inst_handler (*this);
brig_mem_inst_handler mem_inst_handler (*this);
brig_inst_mod_handler inst_mod_handler (*this);
brig_directive_label_handler label_handler (*this);
brig_directive_variable_handler var_handler (*this);
brig_directive_fbarrier_handler fbar_handler (*this);
brig_directive_comment_handler comment_handler (*this);
brig_directive_function_handler func_handler (*this);
brig_directive_control_handler control_handler (*this);
brig_directive_arg_block_handler arg_block_handler (*this);
brig_directive_module_handler module_handler (*this);
brig_lane_inst_handler lane_inst_handler (*this);
brig_queue_inst_handler queue_inst_handler (*this);
skipped_entry_handler skipped_handler (*this);
unimplemented_entry_handler unimplemented_handler (*this);
struct code_entry_handler_info
{
BrigKind kind;
brig_code_entry_handler *handler;
};
code_entry_handler_info handlers[]
= {{BRIG_KIND_INST_BASIC, &inst_handler},
{BRIG_KIND_INST_CMP, &cmp_inst_handler},
{BRIG_KIND_INST_MEM, &mem_inst_handler},
{BRIG_KIND_INST_MOD, &inst_mod_handler},
{BRIG_KIND_INST_CVT, &cvt_inst_handler},
{BRIG_KIND_INST_SEG_CVT, &seg_inst_handler},
{BRIG_KIND_INST_SEG, &seg_inst_handler},
{BRIG_KIND_INST_ADDR, &copy_move_inst_handler},
{BRIG_KIND_INST_SOURCE_TYPE, &copy_move_inst_handler},
{BRIG_KIND_INST_ATOMIC, &atomic_inst_handler},
{BRIG_KIND_INST_SIGNAL, &signal_inst_handler},
{BRIG_KIND_INST_BR, &branch_inst_handler},
{BRIG_KIND_INST_LANE, &lane_inst_handler},
{BRIG_KIND_INST_QUEUE, &queue_inst_handler},
{BRIG_KIND_INST_MEM_FENCE, &skipped_handler},
{BRIG_KIND_DIRECTIVE_LABEL, &label_handler},
{BRIG_KIND_DIRECTIVE_VARIABLE, &var_handler},
{BRIG_KIND_DIRECTIVE_ARG_BLOCK_START, &arg_block_handler},
{BRIG_KIND_DIRECTIVE_ARG_BLOCK_END, &arg_block_handler},
{BRIG_KIND_DIRECTIVE_FBARRIER, &fbar_handler},
{BRIG_KIND_DIRECTIVE_COMMENT, &comment_handler},
{BRIG_KIND_DIRECTIVE_KERNEL, &func_handler},
{BRIG_KIND_DIRECTIVE_SIGNATURE, &func_handler},
{BRIG_KIND_DIRECTIVE_FUNCTION, &func_handler},
{BRIG_KIND_DIRECTIVE_INDIRECT_FUNCTION, &func_handler},
{BRIG_KIND_DIRECTIVE_MODULE, &module_handler},
{BRIG_KIND_DIRECTIVE_LOC, &skipped_handler},
{BRIG_KIND_DIRECTIVE_PRAGMA, &skipped_handler},
{BRIG_KIND_DIRECTIVE_CONTROL, &control_handler},
{BRIG_KIND_DIRECTIVE_EXTENSION, &skipped_handler},
{BRIG_KIND_NONE, &skipped_handler}};
const BrigSectionHeader *csection_header = (const BrigSectionHeader *) m_code;
for (size_t b = csection_header->headerByteCount; b < m_code_size;)
{
const BrigBase *entry = (const BrigBase *) (m_code + b);
brig_code_entry_handler *handler = &unimplemented_handler;
if (m_cf != NULL && b >= m_cf->m_brig_def->nextModuleEntry)
finish_function (); 
for (size_t i = 0;
i < sizeof (handlers) / sizeof (code_entry_handler_info); ++i)
{
if (handlers[i].kind == entry->kind)
handler = handlers[i].handler;
}
b += (*handler) (entry);
}
finish_function ();
}
const BrigData *
brig_to_generic::get_brig_data_entry (size_t entry_offset) const
{
return (const BrigData *) (m_data + entry_offset);
}
const BrigBase *
brig_to_generic::get_brig_operand_entry (size_t entry_offset) const
{
return (const BrigBase *) (m_operand + entry_offset);
}
const BrigBase *
brig_to_generic::get_brig_code_entry (size_t entry_offset) const
{
return (const BrigBase *) (m_code + entry_offset);
}
void
brig_to_generic::append_global (tree g)
{
if (m_globals == NULL_TREE)
{
m_globals = g;
return;
}
else
{
tree last = tree_last (m_globals);
TREE_CHAIN (last) = g;
}
}
tree
brig_to_generic::global_variable (const std::string &name) const
{
label_index::const_iterator i = m_global_variables.find (name);
if (i == m_global_variables.end ())
return NULL_TREE;
else
return (*i).second;
}
tree
brig_to_generic::function_decl (const std::string &name)
{
label_index::const_iterator i = m_function_index.find (name);
if (i == m_function_index.end ())
return NULL_TREE;
return (*i).second;
}
void
brig_to_generic::add_function_decl (const std::string &name, tree func_decl)
{
m_function_index[name] = func_decl;
}
void
brig_to_generic::add_global_variable (const std::string &name, tree var_decl)
{
append_global (var_decl);
m_global_variables[name] = var_decl;
std::string host_def_var_name
= std::string (PHSA_HOST_DEF_PTR_PREFIX) + name;
tree host_def_var = global_variable (host_def_var_name);
if (host_def_var == NULL_TREE)
return;
tree ptype = build_pointer_type (TREE_TYPE (var_decl));
tree var_addr = build1 (ADDR_EXPR, ptype, var_decl);
DECL_INITIAL (host_def_var) = var_addr;
TREE_PUBLIC (host_def_var) = 0;
}
void
brig_to_generic::add_host_def_var_ptr (const std::string &name, tree var_decl)
{
std::string var_name = std::string (PHSA_HOST_DEF_PTR_PREFIX) + name;
tree name_identifier = get_identifier (var_name.c_str ());
tree ptr_var = build_decl (UNKNOWN_LOCATION, VAR_DECL, name_identifier,
build_pointer_type (TREE_TYPE (var_decl)));
DECL_EXTERNAL (ptr_var) = 0;
DECL_ARTIFICIAL (ptr_var) = 0;
TREE_PUBLIC (ptr_var) = 1;
TREE_USED (ptr_var) = 1;
TREE_ADDRESSABLE (ptr_var) = 1;
TREE_STATIC (ptr_var) = 1;
append_global (ptr_var);
m_global_variables[var_name] = ptr_var;
}
std::string
brig_to_generic::get_mangled_name
(const BrigDirectiveExecutable *func) const
{
std::string func_name = get_string (func->name).substr (1);
if (func->linkage == BRIG_LINKAGE_MODULE)
{
func_name = "gccbrig." + m_module_name + "." + func_name;
}
return func_name;
}
std::string
brig_to_generic::get_string (size_t entry_offset) const
{
const BrigData *data_item = get_brig_data_entry (entry_offset);
return std::string ((const char *) &data_item->bytes, data_item->byteCount);
}
tree
build_stmt (enum tree_code code, ...)
{
tree ret;
int length, i;
va_list p;
bool side_effects;
gcc_assert (TREE_CODE_CLASS (code) != tcc_vl_exp);
va_start (p, code);
ret = make_node (code);
TREE_TYPE (ret) = void_type_node;
length = TREE_CODE_LENGTH (code);
side_effects = false;
for (i = 0; i < length; i++)
{
tree t = va_arg (p, tree);
if (t && !TYPE_P (t))
side_effects |= TREE_SIDE_EFFECTS (t);
TREE_OPERAND (ret, i) = t;
}
TREE_SIDE_EFFECTS (ret) |= side_effects;
va_end (p);
return ret;
}
tree
build_resize_convert_view (tree destination_type, tree source)
{
gcc_assert (source && destination_type && TREE_TYPE (source) != NULL_TREE
&& destination_type != NULL_TREE);
tree source_type = TREE_TYPE (source);
if (TREE_CODE (source) == CALL_EXPR)
{
tree func_decl = TREE_OPERAND (TREE_OPERAND (source, 1), 0);
source_type = TREE_TYPE (TREE_TYPE (func_decl));
}
if (destination_type == source_type)
return source;
size_t src_size = int_size_in_bytes (source_type);
size_t dst_size = int_size_in_bytes (destination_type);
if (src_size == dst_size)
return build1 (VIEW_CONVERT_EXPR, destination_type, source);
else 
{
tree resized = convert (get_scalar_unsigned_int_type (destination_type),
build_reinterpret_to_uint (source));
gcc_assert ((size_t)int_size_in_bytes (TREE_TYPE (resized)) == dst_size);
return build_resize_convert_view (destination_type, resized);
}
}
tree build_reinterpret_to_uint (tree source)
{
tree src_type = TREE_TYPE (source);
if (INTEGRAL_TYPE_P (src_type) && TYPE_UNSIGNED (src_type))
return source;
tree dest_type = get_scalar_unsigned_int_type (src_type);
return build1 (VIEW_CONVERT_EXPR, dest_type, source);
}
brig_function *
brig_to_generic::get_finished_function (tree func_decl)
{
std::string func_name
= identifier_to_locale (IDENTIFIER_POINTER (DECL_NAME (func_decl)));
std::map<std::string, brig_function *>::iterator i
= m_finished_functions.find (func_name);
if (i != m_finished_functions.end ())
return (*i).second;
else
return NULL;
}
void
brig_to_generic::add_group_variable (const std::string &name, size_t size,
size_t alignment, bool function_scope)
{
if (!function_scope)
{
if (!m_module_group_variables.has_variable (name))
m_module_group_variables.add (name, size, alignment);
return;
}
if (!m_cf->m_local_group_variables.has_variable (name))
m_cf->m_local_group_variables.add (name, size, alignment);
}
void
brig_to_generic::finish_function ()
{
if (m_cf == NULL || m_cf->m_func_decl == NULL_TREE)
{
m_cf = NULL;
return;
}
if (!m_cf->m_is_kernel)
{
tree bind_expr = m_cf->m_current_bind_expr;
tree stmts = BIND_EXPR_BODY (bind_expr);
m_cf->finish ();
m_cf->emit_metadata (stmts);
dump_function (m_dump_file, m_cf);
gimplify_function_tree (m_cf->m_func_decl);
cgraph_node::finalize_function (m_cf->m_func_decl, true);
}
else
m_kernels.push_back (m_cf);
pop_cfun ();
m_finished_functions[m_cf->m_name] = m_cf;
m_cf = NULL;
}
void
brig_to_generic::start_function (tree f)
{
if (DECL_STRUCT_FUNCTION (f) == NULL)
push_struct_function (f);
else
push_cfun (DECL_STRUCT_FUNCTION (f));
m_cf->m_func_decl = f;
}
void
brig_to_generic::append_private_variable (const std::string &name,
size_t size, size_t alignment)
{
size_t align_padding = m_next_private_offset % alignment == 0 ?
0 : (alignment - m_next_private_offset % alignment);
size_t per_var_padding = size % alignment == 0 ?
0 : (alignment - size % alignment);
m_private_data_sizes[name] = size + per_var_padding;
m_next_private_offset += align_padding;
m_private_offsets[name] = m_next_private_offset;
m_next_private_offset += size + per_var_padding;
}
size_t
brig_to_generic::private_variable_segment_offset
(const std::string &name) const
{
var_offset_table::const_iterator i = m_private_offsets.find (name);
gcc_assert (i != m_private_offsets.end ());
return (*i).second;
}
bool
brig_to_generic::has_private_variable (const std::string &name) const
{
std::map<std::string, size_t>::const_iterator i
= m_private_data_sizes.find (name);
return i != m_private_data_sizes.end ();
}
size_t
brig_to_generic::private_variable_size (const std::string &name) const
{
std::map<std::string, size_t>::const_iterator i
= m_private_data_sizes.find (name);
gcc_assert (i != m_private_data_sizes.end ());
return (*i).second;
}
size_t
brig_to_generic::private_segment_size () const
{
return m_next_private_offset;
}
typedef std::map<std::string, tree> builtin_index;
builtin_index builtin_cache_;
tree
call_builtin (tree pdecl, int nargs, tree rettype, ...)
{
if (rettype == error_mark_node)
return error_mark_node;
tree *types = new tree[nargs];
tree *args = new tree[nargs];
va_list ap;
va_start (ap, rettype);
for (int i = 0; i < nargs; ++i)
{
types[i] = va_arg (ap, tree);
tree arg = va_arg (ap, tree);
args[i] = build_resize_convert_view (types[i], arg);
if (types[i] == error_mark_node || args[i] == error_mark_node)
{
delete[] types;
delete[] args;
va_end (ap);
return error_mark_node;
}
}
va_end (ap);
tree fnptr = build_fold_addr_expr (pdecl);
tree ret = build_call_array (rettype, fnptr, nargs, args);
delete[] types;
delete[] args;
return ret;
}
void
brig_to_generic::write_globals ()
{
for (size_t i = 0; i < m_kernels.size (); ++i)
{
brig_function *f = m_kernels[i];
f->finish_kernel ();
dump_function (m_dump_file, f);
gimplify_function_tree (f->m_func_decl);
cgraph_node::finalize_function (f->m_func_decl, true);
f->m_descriptor.is_kernel = 1;
f->m_descriptor.group_segment_size = m_total_group_segment_usage;
f->m_descriptor.private_segment_size = private_segment_size ();
f->m_descriptor.kernarg_segment_size = f->m_next_kernarg_offset;
if (f->m_descriptor.kernarg_segment_size % 16 > 0)
f->m_descriptor.kernarg_segment_size
+= 16 - f->m_next_kernarg_offset % 16;
f->m_descriptor.kernarg_max_align = f->m_kernarg_max_align;
tree launcher = f->emit_launcher_and_metadata ();
append_global (launcher);
gimplify_function_tree (launcher);
cgraph_node::finalize_function (launcher, true);
pop_cfun ();
}
int no_globals = list_length (m_globals);
tree *vec = new tree[no_globals];
int i = 0;
tree global = m_globals;
while (global)
{
vec[i] = global;
++i;
global = TREE_CHAIN (global);
}
wrapup_global_declarations (vec, no_globals);
delete[] vec;
}
tree
get_unsigned_int_type (tree original_type)
{
if (VECTOR_TYPE_P (original_type))
{
size_t esize
= int_size_in_bytes (TREE_TYPE (original_type)) * BITS_PER_UNIT;
poly_uint64 ecount = TYPE_VECTOR_SUBPARTS (original_type);
return build_vector_type (build_nonstandard_integer_type (esize, true),
ecount);
}
else
return build_nonstandard_integer_type (int_size_in_bytes (original_type)
* BITS_PER_UNIT,
true);
}
tree
get_scalar_unsigned_int_type (tree original_type)
{
return build_nonstandard_integer_type (int_size_in_bytes (original_type)
* BITS_PER_UNIT, true);
}
void
dump_function (FILE *dump_file, brig_function *f)
{
if (dump_file)
{
fprintf (dump_file, "\n;; Function %s", f->m_name.c_str ());
fprintf (dump_file, "\n;; enabled by -%s\n\n",
dump_flag_name (TDI_original));
print_generic_decl (dump_file, f->m_func_decl, 0);
print_generic_expr (dump_file, f->m_current_bind_expr, 0);
fprintf (dump_file, "\n");
}
}
void
brig_to_generic::add_reg_used_as_type (const BrigOperandRegister &brig_reg,
tree type)
{
gcc_assert (m_cf);
reg_use_info &info
= m_fn_regs_use_index[m_cf->m_name][gccbrig_hsa_reg_id (brig_reg)];
if (info.m_type_refs_lookup.count (type))
info.m_type_refs[info.m_type_refs_lookup[type]].second++;
else
{
info.m_type_refs.push_back (std::make_pair (type, 1));
info.m_type_refs_lookup[type] = info.m_type_refs.size () - 1;
}
}