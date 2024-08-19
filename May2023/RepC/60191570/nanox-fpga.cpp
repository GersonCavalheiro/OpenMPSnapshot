#include <errno.h>
#include <fstream>
#include "cxx-diagnostic.h"
#include "tl-devices.hpp"
#include "tl-compilerpipeline.hpp"
#include "tl-multifile.hpp"
#include "tl-source.hpp"
#include "codegen-phase.hpp"
#include "codegen-cxx.hpp"
#include "cxx-cexpr.h"
#include "cxx-driver-utils.h"
#include "cxx-process.h"
#include "cxx-cexpr.h"
#include "nanox-fpga.hpp"
#include "cxx-nodecl.h"
#include "cxx-graphviz.h"
#include "tl-nanos.hpp"
#include "tl-symbol-utils.hpp"
using namespace TL;
using namespace TL::Nanox;
const std::string DeviceFPGA::HLS_VPREF = "_hls_var_";
const std::string DeviceFPGA::HLS_I = HLS_VPREF + "i";
const std::string DeviceFPGA::hls_in = HLS_VPREF + "in";
const std::string DeviceFPGA::hls_out = HLS_VPREF + "out";
static std::string fpga_outline_name(const std::string &name)
{
return "fpga_" + name;
}
UNUSED_PARAMETER static void print_ast_dot(const Nodecl::NodeclBase &node)
{
std::cerr << std::endl << std::endl;
ast_dump_graphviz(nodecl_get_ast(node.get_internal_nodecl()), stderr);
std::cerr << std::endl << std::endl;
}
void DeviceFPGA::create_outline(CreateOutlineInfo &info,
Nodecl::NodeclBase &outline_placeholder,
Nodecl::NodeclBase &output_statements,
Nodecl::Utils::SimpleSymbolMap* &symbol_map)
{
if (IS_FORTRAN_LANGUAGE)
fatal_error("Fortran for FPGA devices is not supported yet\n");
Lowering* lowering = info._lowering;
const std::string& device_outline_name = fpga_outline_name(info._outline_name);
const Nodecl::NodeclBase& original_statements = info._original_statements;
const TL::Symbol& arguments_struct = info._arguments_struct;
const TL::Symbol& called_task = info._called_task;
lowering->seen_fpga_task = true;
symbol_map = new Nodecl::Utils::SimpleSymbolMap(&_copied_fpga_functions);
TL::Symbol current_function = original_statements.retrieve_context().get_related_symbol();
if (current_function.is_nested_function())
{
if (IS_C_LANGUAGE || IS_CXX_LANGUAGE)
fatal_printf_at(original_statements.get_locus(), "nested functions are not supported\n");
}
if (called_task.is_valid())
{
if ( (IS_C_LANGUAGE || IS_CXX_LANGUAGE) && !called_task.get_function_code().is_null())
{
if (_copied_fpga_functions.map(called_task) == called_task)
{
TL::Symbol new_function = SymbolUtils::new_function_symbol_for_deep_copy(
called_task,
called_task.get_name() + "_hls");
_copied_fpga_functions.add_map(called_task, new_function);
_fpga_file_code.append (Nodecl::Utils::deep_copy(
called_task.get_function_code(),
called_task.get_scope(),
*symbol_map)
);
}
}
else if (IS_FORTRAN_LANGUAGE)
{
fatal_error("There is no fortran support for FPGA devices\n");
}
else
{
fatal_error("Inline tasks not supported yet\n");
}
}
Source unpacked_arguments, private_entities;
TL::ObjectList<OutlineDataItem*> data_items = info._data_items;
for (TL::ObjectList<OutlineDataItem*>::iterator it = data_items.begin();
it != data_items.end();
it++)
{
switch ((*it)->get_sharing())
{
case OutlineDataItem::SHARING_PRIVATE:
{
break;
}
case OutlineDataItem::SHARING_SHARED:
case OutlineDataItem::SHARING_CAPTURE:
case OutlineDataItem::SHARING_CAPTURE_ADDRESS:
{
TL::Type param_type = (*it)->get_in_outline_type();
Source argument;
if (IS_C_LANGUAGE || IS_CXX_LANGUAGE)
{
if ((*it)->get_sharing() == OutlineDataItem::SHARING_SHARED
&& !(IS_CXX_LANGUAGE && (*it)->get_symbol().get_name() == "this"))
{
argument << "*(args." << (*it)->get_field_name() << ")";
}
else
{
argument << "args." << (*it)->get_field_name();
}
if (IS_CXX_LANGUAGE
&& (*it)->get_allocation_policy() == OutlineDataItem::ALLOCATION_POLICY_TASK_MUST_DESTROY)
{
internal_error("Not yet implemented: call the destructor", 0);
}
}
else
{
internal_error("running error", 0);
}
unpacked_arguments.append_with_separator(argument, ", ");
break;
}
case OutlineDataItem::SHARING_REDUCTION:
{
WARNING_MESSAGE("Reductions are not tested for FPGA", "");
Source argument;
if (IS_C_LANGUAGE || IS_CXX_LANGUAGE)
{
argument << "*(args." << (*it)->get_field_name() << ")";
}
else
{
internal_error("running error", 0);
}
unpacked_arguments.append_with_separator(argument, ", ");
break;
}
default:
{
std::cerr << "Warning: Cannot copy function code to the device file" << std::endl;
}
}
}
TL::Source dummy_init_statements, dummy_final_statements;
TL::Symbol unpacked_function = new_function_symbol_unpacked(
current_function,
device_outline_name + "_unpacked",
info,
symbol_map,
dummy_init_statements,
dummy_final_statements);
symbol_entity_specs_set_is_static(unpacked_function.get_internal_symbol(), 0);
if (IS_C_LANGUAGE)
{
symbol_entity_specs_set_linkage_spec(unpacked_function.get_internal_symbol(), "\"C\"");
}
Nodecl::NodeclBase unpacked_function_code, unpacked_function_body;
SymbolUtils::build_empty_body_for_function(unpacked_function,
unpacked_function_code,
unpacked_function_body);
Source fpga_params;
Source unpacked_source;
unpacked_source
<< dummy_init_statements
<< private_entities
<< fpga_params
<< statement_placeholder(outline_placeholder)
<< dummy_final_statements
;
Nodecl::NodeclBase new_unpacked_body =
unpacked_source.parse_statement(unpacked_function_body);
unpacked_function_body.replace(new_unpacked_body);
Nodecl::Utils::prepend_to_enclosing_top_level_location(original_statements, unpacked_function_code);
if (IS_CXX_LANGUAGE)
{
Nodecl::NodeclBase nodecl_decl = Nodecl::CxxDecl::make(
nodecl_null(),
unpacked_function,
original_statements.get_locus());
Nodecl::Utils::prepend_to_enclosing_top_level_location(original_statements, nodecl_decl);
}
ObjectList<std::string> structure_name;
structure_name.append("args");
ObjectList<TL::Type> structure_type;
structure_type.append(TL::Type(
get_user_defined_type(
arguments_struct.get_internal_symbol())).get_lvalue_reference_to());
TL::Symbol outline_function = SymbolUtils::new_function_symbol(
current_function,
device_outline_name,
TL::Type::get_void_type(),
structure_name,
structure_type);
Nodecl::NodeclBase outline_function_code, outline_function_body;
SymbolUtils::build_empty_body_for_function(outline_function,
outline_function_code,
outline_function_body);
Source outline_src;
Source instrument_before,
instrument_after;
if (instrumentation_enabled())
{
get_instrumentation_code(
info._called_task,
outline_function,
outline_function_body,
info._task_label,
info._instr_locus,
instrument_before,
instrument_after);
}
outline_src
<< "{"
<<      instrument_before
<<      unpacked_function.get_qualified_name_for_expression(
(current_function.get_type().is_template_specialized_type()
&& current_function.get_type().is_dependent())
) << "(" << unpacked_arguments << ");"
<<      instrument_after
<< "}"
;
Nodecl::NodeclBase new_outline_body = outline_src.parse_statement(outline_function_body);
outline_function_body.replace(new_outline_body);
Nodecl::Utils::prepend_to_enclosing_top_level_location(original_statements, outline_function_code);
output_statements = Nodecl::EmptyStatement::make(
original_statements.get_locus());
}
DeviceFPGA::DeviceFPGA()
: DeviceProvider(std::string("fpga"))
{
set_phase_name("Nanox FPGA support");
set_phase_description("This phase is used by Nanox phases to implement FPGA device support");
}
void DeviceFPGA::pre_run(DTO& dto)
{
}
void DeviceFPGA::run(DTO& dto)
{
DeviceProvider::run(dto);
}
bool DeviceFPGA::task_has_scalars(TL::ObjectList<OutlineDataItem*> & dataitems)
{
for (ObjectList<OutlineDataItem*>::iterator it = dataitems.begin();
it != dataitems.end();
it++)
{
if((*it)->get_copies().empty())
{
return true;
}
}
return false;
}
void DeviceFPGA::get_device_descriptor(DeviceDescriptorInfo& info,
Source &ancillary_device_description,
Source &device_descriptor,
Source &fortran_dynamic_init)
{
const std::string& outline_name = fpga_outline_name(info._outline_name);
const std::string& arguments_struct = info._arguments_struct;
TL::Symbol current_function = info._current_function;
std::string original_name = current_function.get_name();
current_function.set_name(outline_name);
Nodecl::NodeclBase code = current_function.get_function_code();
Nodecl::Context context = (code.is<Nodecl::TemplateFunctionCode>())
? code.as<Nodecl::TemplateFunctionCode>().get_statements().as<Nodecl::Context>()
: code.as<Nodecl::FunctionCode>().get_statements().as<Nodecl::Context>();
bool without_template_args =
!current_function.get_type().is_template_specialized_type()
|| current_function.get_scope().get_template_parameters()->is_explicit_specialization;
TL::Scope function_scope = context.retrieve_context();
std::string qualified_name = current_function.get_qualified_name(function_scope, without_template_args);
current_function.set_name(original_name);
ObjectList<Nodecl::NodeclBase> onto_clause = info._target_info.get_onto();
Nodecl::Utils::SimpleSymbolMap param_to_args_map = info._target_info.get_param_arg_map();
std::string acc_num = "-1";
if (onto_clause.size() >= 1)
{
Nodecl::NodeclBase onto_acc = onto_clause[0];
if (onto_clause.size() > 1)
{
warn_printf_at(onto_acc.get_locus(), "More than one argument in onto clause. Using only first one\n");
}
if (onto_clause[0].is_constant())
{
const_value_t *ct_val = onto_acc.get_constant();
if (!const_value_is_integer(ct_val))
{
error_printf_at(onto_acc.get_locus(), "Constant is not integer type in onto clause\n");
}
else
{
int acc = const_value_cast_to_signed_int(ct_val);
std::stringstream tmp_str;
tmp_str << acc;
acc_num = tmp_str.str();
}
}
else
{
if (onto_acc.get_symbol().is_valid() ) {
acc_num = as_symbol(onto_acc.get_symbol());
}
}
}
else
{
}
if (!IS_FORTRAN_LANGUAGE)
{
std::string ref = IS_CXX_LANGUAGE ? "&" : "*";
std::string extra_cast = "(void(*)(" + arguments_struct + ref + "))";
Source args_name;
args_name << outline_name << "_args";
ancillary_device_description
<< comment("device argument type")
<< "static nanos_fpga_args_t " << args_name << ";"
<< args_name << ".outline = (void(*)(void*)) " << extra_cast << " &" << qualified_name << ";"
<< args_name << ".acc_num = " << acc_num << ";"
;
device_descriptor
<< "{"
<<  "&nanos_fpga_factory, &" << outline_name << "_args"
<< "}"
;
}
else
{
internal_error("Fortran is not supperted in fpga devices", 0);
}
}
bool DeviceFPGA::remove_function_task_from_original_source() const
{
return true;
}
void DeviceFPGA::phase_cleanup(DTO& data_flow)
{
if (!_fpga_file_code.is_null())
{
std::string original_filename = TL::CompilationProcess::get_current_file().get_filename();
std::string new_filename = "hls_" + original_filename;
std::ofstream hls_file;
hls_file.open(new_filename.c_str()); 
if (! hls_file.is_open())
{
fatal_error("%s: error: cannot open file '%s'. %s\n",
original_filename.c_str(),
new_filename.c_str(),
strerror(errno));
}
ObjectList<IncludeLine> includes = CurrentFile::get_included_files();
for (ObjectList<IncludeLine>::iterator it = includes.begin(); it != includes.end(); it++)
{
hls_file << it->get_preprocessor_line() << std::endl;
}
hls_file << _fpga_file_code.prettyprint();
hls_file.close();
#if 0
FILE* ancillary_file = fopen(new_filename.c_str(), "w");
if (ancillary_file == NULL)
{
fatal_error("%s: error: cannot open file '%s'. %s\n",
original_filename.c_str(),
new_filename.c_str(),
strerror(errno));
}
compilation_configuration_t* configuration = CURRENT_CONFIGURATION;
ERROR_CONDITION (configuration == NULL, "The compilation configuration cannot be NULL", 0);
load_compiler_phases(configuration);
TL::CompilationProcess::add_file(new_filename, "fpga");
::mark_file_for_cleanup(new_filename.c_str());
Codegen::CxxBase* phase = reinterpret_cast<Codegen::CxxBase*>(configuration->codegen_phase);
phase->codegen_top_level(_fpga_file_code, ancillary_file);
fclose(ancillary_file);
#endif
_fpga_file_code = Nodecl::List();
}
}
Source DeviceFPGA::fpga_param_code(
TL::ObjectList<OutlineDataItem*> &data_items,
Nodecl::Utils::SymbolMap *symbol_map,
Scope sc
)
{
Source args_src;
args_src
<< "int fd = open(\"/dev/mem\", NANOS_O_RDWR);"    
<< "unsigned int *acc_handle = "
<< "    (unsigned int *) mmap(0, NANOS_MMAP_SIZE,"     
<< "    NANOS_PROT_READ|NANOS_PROT_WRITE, NANOS_MAP_SHARED,"           
<< "    fd, NANOS_AXI_BASE_ADDRESS);"
;
int argIndex = 0x14/4;  
for (TL::ObjectList<OutlineDataItem*>::iterator it = data_items.begin();
it != data_items.end();
it++)
{
Symbol outline_symbol = symbol_map->map((*it)->get_symbol());
const TL::ObjectList<OutlineDataItem::CopyItem> &copies = (*it)->get_copies();
if (copies.empty())
{
const Type & type = (*it)->get_field_type();
args_src
<< "acc_handle[" << argIndex << "] = "
<< outline_symbol.get_name() << ";"
;
argIndex+=2;
if (type.get_size() >= 8)
{
argIndex ++;
}
}
}
args_src << "acc_handle[0] = 1;"
<< "munmap(acc_handle, NANOS_MMAP_SIZE);"
<< "close(fd);"
;
return args_src;
}
void DeviceFPGA::add_hls_pragmas(
Nodecl::NodeclBase &task,
TL::ObjectList<OutlineDataItem*> &data_items
)
{
std::cerr << ast_node_type_name(task.get_kind()) 
<< " in_list: " << task.is_in_list()
<< " locus: " << task.get_locus()
<< std::endl;
Nodecl::NodeclBase::Children tchildren = task.children();
Nodecl::NodeclBase& context = tchildren.front();
Nodecl::NodeclBase::Children cchildren = context.children();
Nodecl::List list(cchildren.front().get_internal_nodecl());
Nodecl::List stlist(list.begin()->children().front().get_internal_nodecl());
Nodecl::UnknownPragma ctrl_bus = Nodecl::UnknownPragma::make(
"AP resource core=AXI_SLAVE variable=return metadata=\"-bus_bundle AXIlite\" port_map={{ap_start START} {ap_done DONE} {ap_idle IDLE} {ap_return RETURN}}");
stlist.prepend(ctrl_bus);
for (TL::ObjectList<OutlineDataItem*>::iterator it = data_items.begin();
it != data_items.end();
it++)
{
std::string field_name = (*it)->get_field_name();
Nodecl::UnknownPragma pragma_node;
if ((*it)->get_copies().empty())
{
pragma_node = Nodecl::UnknownPragma::make("HLS INTERFACE ap_none port=" + field_name);
stlist.prepend(pragma_node);
pragma_node = Nodecl::UnknownPragma::make("AP resource core=AXI_SLAVE variable="
+ field_name
+ " metadata=\"-bus_bundle AXIlite\"");
stlist.prepend(pragma_node);
}
else
{
pragma_node = Nodecl::UnknownPragma::make(
"HLS resource core=AXI4Stream variable=" + field_name);
stlist.prepend(pragma_node);
pragma_node = Nodecl::UnknownPragma::make(
"HLS interface ap_fifo port=" + field_name);
stlist.prepend(pragma_node);
}
}
}
static void get_inout_decl(ObjectList<OutlineDataItem*>& data_items, std::string &in_type, std::string &out_type)
{
in_type = "";
out_type = "";
for (ObjectList<OutlineDataItem*>::iterator it = data_items.begin();
it != data_items.end();
it++)
{
const ObjectList<OutlineDataItem::CopyItem> &copies = (*it)->get_copies();
if (!copies.empty())
{
Scope scope = (*it)->get_symbol().get_scope();
if (copies.front().directionality == OutlineDataItem::COPY_IN
&& in_type == "")
{
in_type = (*it)->get_field_type().get_simple_declaration(scope, "");
}
else if  (copies.front().directionality == OutlineDataItem::COPY_OUT
&& out_type == "")
{
out_type = (*it)->get_field_type().get_simple_declaration(scope, "");
} else if (copies.front().directionality == OutlineDataItem::COPY_INOUT)
{
out_type = (*it)->get_field_type().get_simple_declaration(scope, "");
in_type = out_type;
return;
}
}
}
}
static int get_copy_elements(Nodecl::NodeclBase expr)
{
int elems;
DataReference datareference(expr);
if (!datareference.is_valid())
{
internal_error("invalid data reference (%s)", datareference.get_locus_str().c_str());
}
Type type = datareference.get_data_type();
if (type.array_is_region()) 
{
Nodecl::NodeclBase cp_size = type.array_get_region_size();
elems = const_value_cast_to_4(cp_size.get_constant());
}
else if (type.is_array()) 
{
Nodecl::NodeclBase lower, upper;
type.array_get_bounds(lower, upper);
elems = const_value_cast_to_4(upper.get_constant()) - const_value_cast_to_4(lower.get_constant()) + 1;
}
else 
{
internal_error("Data copies must be an array region expression (%d)", datareference.get_locus_str().c_str());
}
return elems;
}
Nodecl::NodeclBase DeviceFPGA::gen_hls_wrapper(const Symbol &func_symbol, ObjectList<OutlineDataItem*>& data_items)
{
if (!func_symbol.is_function())
{
fatal_error("Only function-tasks are supperted at this moment");
}
Scope fun_scope = func_symbol.get_scope();
std::string in_dec, out_dec;
get_inout_decl(data_items, in_dec, out_dec);
Source pragmas_src;
if (task_has_scalars(data_items))
{
pragmas_src
<< "#pragma HLS resource core=AXI_SLAVE variable=return metadata=\"-bus_bundle AXIlite\" "
<< "port_map={{ap_start START} {ap_done DONE} {ap_idle IDLE} {ap_return RETURN}}\n";
;
}
Source args;
if (in_dec != "")
{
args << in_dec << hls_in;
pragmas_src
<< "#pragma HLS resource core=AXI4Stream variable=" << hls_in << "\n"
<< "#pragma HLS interface ap_fifo port=" << hls_in << "\n"
;
}
if (out_dec != "")
{
args.append_with_separator(out_dec + hls_out, ",");
pragmas_src
<< "#pragma HLS resource core=AXI4Stream variable=" << hls_out << "\n"
<< "#pragma HLS interface ap_fifo port=" << hls_out << "\n"
;
}
Source copies_src;
Source in_copies, out_copies;
Source fun_params;
Source local_decls;
int in_offset  = 0;
int out_offset = 0;
for (ObjectList<OutlineDataItem*>::iterator it = data_items.begin();
it != data_items.end();
it++)
{
fun_params.append_with_separator((*it)->get_field_name(), ",");
const std::string &field_name = (*it)->get_field_name();
const Scope &scope = (*it)->get_symbol().get_scope();
const ObjectList<OutlineDataItem::CopyItem> &copies = (*it)->get_copies();
if (!copies.empty())
{
Nodecl::NodeclBase expr = copies.front().expression;
if (copies.size() > 1)
{
internal_error("Only one copy per object (in/out/inout) is allowed (%s)",
expr.get_locus_str().c_str());
}
int n_elements = get_copy_elements(expr);
const Type &field_type = (*it)->get_field_type();
Type elem_type;
if (field_type.is_pointer())
{
elem_type = field_type.points_to();
}
else if (field_type.is_array())
{
elem_type = field_type.array_element();
}
else
{
internal_error("invalid type for input/output, only pointer and array is allowed (%d)",
expr.get_locus_str().c_str());
}
std::string par_simple_decl = elem_type.get_simple_declaration(scope, field_name);
local_decls
<< par_simple_decl << "[" << n_elements << "];\n";
if (copies.front().directionality == OutlineDataItem::COPY_IN
or copies.front().directionality == OutlineDataItem::COPY_INOUT)
{
in_copies
<< "for (" << HLS_I << "=0;" << HLS_I << "<" << n_elements << "; " << HLS_I << "++)"
<< "{"
<< "  " << field_name << "[" << HLS_I << "] = " << hls_in << "[" << HLS_I << "+" << in_offset << "];"
<< "}"
;
in_offset += n_elements;
}
if (copies.front().directionality == OutlineDataItem::COPY_OUT
or copies.front().directionality == OutlineDataItem::COPY_INOUT)
{
out_copies
<< "for (" << HLS_I << "=0;" << HLS_I << "<" << n_elements << "; " << HLS_I << "++)"
<< "{"
<< "  "  << hls_out << "[" << HLS_I << "+" << out_offset << "] = " << field_name << "[" << HLS_I << "];"
<< "}"
;
out_offset += n_elements;
}
}
else
{
Source par_src;
par_src
<< (*it)->get_field_type().get_simple_declaration(scope, field_name)
;
args.append_with_separator(par_src, ",");
pragmas_src
<< "#pragma HLS INTERFACE ap_none port=" <<  field_name << "\n"
<< "#pragma AP resource core=AXI_SLAVE variable=" << field_name << " metadata=\"-bus_bundle AXIlite\"\n"
;
}
}
Nodecl::NodeclBase fun_code =  func_symbol.get_function_code();
Source wrapper_src;
wrapper_src
<< "void core_hw_accelerator(" << args<< "){"
;
local_decls << "unsigned int " << HLS_I << ";";
wrapper_src
<< pragmas_src
<< local_decls
<< in_copies
<< func_symbol.get_name() << "(" << fun_params << ");"
<< out_copies
<< "}"
;
ReferenceScope refscope(func_symbol.get_scope());
Nodecl::NodeclBase wrapper_node = wrapper_src.parse_global(refscope);
return wrapper_node;
}
void DeviceFPGA::copy_stuff_to_device_file(
const TL::ObjectList<Nodecl::NodeclBase>& stuff_to_be_copied)
{
for (TL::ObjectList<Nodecl::NodeclBase>::const_iterator it = stuff_to_be_copied.begin();
it != stuff_to_be_copied.end();
++it)
{
if (it->is<Nodecl::FunctionCode>()
|| it->is<Nodecl::TemplateFunctionCode>())
{
TL::Symbol function = it->get_symbol();
TL::Symbol new_function = SymbolUtils::new_function_symbol(function, function.get_name() + "_hls");
_copied_fpga_functions.add_map(function, new_function);
_fpga_file_code.append(Nodecl::Utils::deep_copy(*it, *it, _copied_fpga_functions));
}
else
{
_fpga_file_code.append(Nodecl::Utils::deep_copy(*it, *it));
}
}
}
EXPORT_PHASE(TL::Nanox::DeviceFPGA);
