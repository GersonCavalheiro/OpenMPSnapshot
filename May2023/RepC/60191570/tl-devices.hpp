#ifndef NANOX_DEVICES_HPP
#define NANOX_DEVICES_HPP
#include "tl-compilerphase.hpp"
#include "tl-objectlist.hpp"
#include "tl-source.hpp"
#include "tl-outline-info.hpp"
#include "tl-nanox-nodecl.hpp"
#include "tl-target-information.hpp"
namespace TL { namespace Nanox {
struct DeviceDescriptorInfo
{
const std::string& _outline_name;
const std::string& _arguments_struct;
const TL::Symbol& _current_function;
const TargetInformation& _target_info;
int _fortran_device_index;
ObjectList<OutlineDataItem*> _data_items;
TL::Symbol _called_task;
DeviceDescriptorInfo(
const std::string& outline_name,
const std::string& arguments_struct,
const TL::Symbol& current_function,
const TargetInformation& target_info,
int fortran_device_index,
ObjectList<OutlineDataItem*> data_items,
const TL::Symbol& called_task)
:
_outline_name(outline_name),
_arguments_struct(arguments_struct),
_current_function(current_function),
_target_info(target_info),
_fortran_device_index(fortran_device_index),
_data_items(data_items),
_called_task(called_task)
{
}
};
struct CreateOutlineInfo
{
Lowering* _lowering;
const std::string& _outline_name;
ObjectList<OutlineDataItem*> _data_items;
const TargetInformation& _target_info;
Nodecl::NodeclBase _original_statements;
Nodecl::NodeclBase _task_statements;
Nodecl::NodeclBase _task_label;
const TL::Symbol& _arguments_struct;
TL::Symbol _called_task;
const locus_t* _instr_locus;
CreateOutlineInfo(
Lowering* lowering,
std::string& outline_name,
ObjectList<OutlineDataItem*> data_items,
const TargetInformation& target_info,
Nodecl::NodeclBase original_statements,
Nodecl::NodeclBase task_statements,
Nodecl::NodeclBase task_label,
TL::Symbol& args_struct,
const TL::Symbol& called_task,
const locus_t* instr_locus)
:
_lowering(lowering),
_outline_name(outline_name),
_data_items(data_items),
_target_info(target_info),
_original_statements(original_statements),
_task_statements(task_statements),
_task_label(task_label),
_arguments_struct(args_struct),
_called_task(called_task),
_instr_locus(instr_locus)
{
}
};
class DeviceProvider : public TL::CompilerPhase
{
protected:
bool instrumentation_enabled();
const std::string _device_name;
Nodecl::List _extra_c_code;
private:
bool _enable_instrumentation;
std::string _enable_instrumentation_str;
void set_instrumentation(const std::string& str);
void common_constructor_code();
public:
DeviceProvider(const std::string& device_name);
virtual ~DeviceProvider() { }
std::string get_name() const;
virtual void pre_run(DTO& dto) { }
virtual void run(DTO& dto) { }
virtual void create_outline(CreateOutlineInfo &info,
Nodecl::NodeclBase &outline_placeholder,
Nodecl::NodeclBase &output_statements,
Nodecl::Utils::SimpleSymbolMap* &symbol_map) = 0;
virtual void get_device_descriptor(DeviceDescriptorInfo& info,
Source &ancillary_device_description,
Source &device_descriptor,
Source &fortran_dynamic_init) = 0;
void get_instrumentation_code(
const TL::Symbol& called_task,
const TL::Symbol& outline_function,
Nodecl::NodeclBase outline_function_body,
Nodecl::NodeclBase task_label,
const locus_t* locus,
Source& instrumentation_before,
Source& instrumentation_after);
virtual void generate_outline_events_before(
Source& function_name_instr,
Source& extra_cast,
Source& instrumentation_before);
virtual void generate_outline_events_after(
Source& function_name_instr,
Source& extra_cast,
Source& instrumentation_after);
virtual bool remove_function_task_from_original_source() const = 0;
virtual void copy_stuff_to_device_file(
const TL::ObjectList<Nodecl::NodeclBase>& stuff_to_be_copied) = 0;
virtual bool allow_mandatory_creation()
{
return false;
}
TL::Symbol new_function_symbol_forward(
TL::Symbol current_function,
const std::string& function_name,
const CreateOutlineInfo& info);
TL::Symbol new_function_symbol_unpacked(
TL::Symbol current_function,
const std::string& function_name,
const CreateOutlineInfo& info,
Nodecl::Utils::SimpleSymbolMap* out_symbol_map,
Source &initial_statements,
Source &final_statements);
TL::Symbol new_function_symbol_unpacked(
TL::Symbol current_function,
const std::string& function_name,
const CreateOutlineInfo& info,
bool make_it_global,
Nodecl::Utils::SimpleSymbolMap* out_symbol_map,
Source &initial_statements,
Source &final_statements);
void add_forward_function_code_to_extra_c_code(
const TL::Symbol&  fortran_forward_symbol,
Nodecl::NodeclBase parse_context);
TL::Type rewrite_type_of_vla_in_outline(
TL::Type t,
const TL::ObjectList<OutlineDataItem*> &data_items,
TL::Symbol &arguments_symbol);
void update_ndrange_and_shmem_expressions(
const TL::Scope& related_scope,
const TargetInformation& target_info,
Nodecl::Utils::SimpleSymbolMap* symbol_map,
TL::ObjectList<Nodecl::NodeclBase>& new_ndrange_exprs,
TL::ObjectList<Nodecl::NodeclBase>& new_shmem_exprs);
};
class DeviceHandler
{
public:
static DeviceHandler& get_device_handler();
void register_device(DeviceProvider* nanox_device_provider);
void register_device(const std::string& str,
DeviceProvider* nanox_device_provider);
DeviceProvider* get_device(const std::string& str);
private:
typedef std::map<std::string, DeviceProvider*> nanox_devices_map_t;
nanox_devices_map_t _nanox_devices;
};
void add_used_types(const TL::ObjectList<OutlineDataItem*> &data_items, TL::Scope sc);
void duplicate_internal_subprograms(
TL::ObjectList<Nodecl::NodeclBase>& internal_function_codes,
TL::Scope scope_of_unpacked,
Nodecl::Utils::SimpleSymbolMap* &symbol_map,
Nodecl::NodeclBase& output_statements
);
void duplicate_nested_functions(
TL::ObjectList<Nodecl::NodeclBase>& internal_function_codes,
TL::Scope scope_of_unpacked,
Nodecl::Utils::SimpleSymbolMap* &symbol_map,
Nodecl::NodeclBase& output_statements
);
} }
#endif 
