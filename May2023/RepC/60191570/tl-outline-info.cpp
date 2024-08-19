#include "tl-outline-info.hpp"
#include "tl-nodecl-visitor.hpp"
#include "tl-datareference.hpp"
#include "tl-counters.hpp"
#include "tl-predicateutils.hpp"
#include "codegen-phase.hpp"
#include "cxx-diagnostic.h"
#include "cxx-exprtype.h"
#include "fortran03-typeutils.h"
#include "tl-target-information.hpp"
#include "tl-counters.hpp"
#include "fortran03-typeutils.h"
#include "fortran03-typeenviron.h"
#include "tl-nanox-ptr.hpp"
namespace TL { namespace Nanox {
std::string OutlineInfo::get_field_name(std::string name)
{
if (IS_CXX_LANGUAGE
&& name == "this")
name = "this_";
if (IS_C_LANGUAGE || IS_CXX_LANGUAGE)
{
if (name == "__PRETTY_FUNCTION__" 
|| name == "__FUNCTION__" 
|| name == "__MERCURIUM_PRETTY_FUNCTION__") 
name = strtolower(name.c_str());
else if (name == "__func__") 
name = "__function__";
}
int times_name_appears = 0;
for (ObjectList<OutlineDataItem*>::iterator it = _data_env_items.begin();
it != _data_env_items.end();
it++)
{
if ((*it)->get_symbol().is_valid()
&& (*it)->get_symbol().get_name() == name)
{
times_name_appears++;
}
}
std::stringstream ss;
ss << name;
if (times_name_appears > 0)
ss << "_" << times_name_appears;
return ss.str();
}
OutlineDataItem& OutlineInfo::get_entity_for_symbol(TL::Symbol sym)
{
for (ObjectList<OutlineDataItem*>::iterator it = _data_env_items.begin();
it != _data_env_items.end();
it++)
{
if ((*it)->get_symbol() == sym)
{
return *(*it);
}
}
std::string field_name = get_field_name(sym.get_name());
OutlineDataItem* env_item = new OutlineDataItem(sym, field_name);
_data_env_items.append(env_item);
return (*_data_env_items.back());
}
struct MatchingSymbol
{
TL::Symbol _sym;
MatchingSymbol(OutlineDataItem& item)
: _sym(item.get_symbol())
{
}
bool operator()(OutlineDataItem* item)
{
return (item != NULL
&& item->get_symbol() == _sym);
}
};
void OutlineInfo::remove_entity(OutlineDataItem& item)
{
_data_env_items.erase(
std::remove_if(
_data_env_items.begin(),
_data_env_items.end(),
MatchingSymbol(item)));
}
void OutlineInfoRegisterEntities::add_shared(Symbol sym)
{
add_shared_common(sym, sym.get_type().no_ref().get_pointer_to());
}
void OutlineInfoRegisterEntities::add_shared_opaque(Symbol sym)
{
add_shared_common(sym, TL::Type::get_void_type().get_pointer_to());
}
void OutlineInfoRegisterEntities::add_shared_common(Symbol sym, TL::Type field_type)
{
OutlineDataItem &outline_info = _outline_info.get_entity_for_symbol(sym);
ERROR_CONDITION(
(outline_info.get_sharing() != OutlineDataItem::SHARING_UNDEFINED
&& outline_info.get_sharing() != OutlineDataItem::SHARING_SHARED),
"Overwriting data-sharing for symbol '%s'", sym.get_name().c_str());
if (outline_info.get_sharing() == OutlineDataItem::SHARING_UNDEFINED)
outline_info.set_sharing(OutlineDataItem::SHARING_SHARED);
Type t = sym.get_type();
if (t.is_any_reference())
{
t = t.references_to();
}
if (IS_CXX_LANGUAGE
&& sym.get_name() == "this")
{
outline_info.set_field_type(t.get_unqualified_type_but_keep_restrict());
outline_info.set_in_outline_type(t.get_unqualified_type_but_keep_restrict());
}
else
{
outline_info.set_field_type(field_type);
TL::Type in_outline_type = t.get_lvalue_reference_to();
in_outline_type = add_extra_dimensions(sym, in_outline_type, &outline_info);
outline_info.set_in_outline_type(in_outline_type);
}
}
OutlineDataItem* OutlineInfoRegisterEntities::capture_descriptor(
OutlineDataItem &outline_info,
Symbol sym)
{
TL::Type array_type = sym.get_type().no_ref();
if (array_type.is_pointer())
{
array_type = array_type.points_to();
}
ERROR_CONDITION(!array_type.is_fortran_array(), "Invalid type", 0);
TL::Type void_pointer = TL::Type::get_void_type().get_pointer_to();
TL::Type suitable_integer = fortran_choose_int_type_from_kind(void_pointer.get_size());
size_t size_of_array_descriptor = fortran_size_of_array_descriptor(
fortran_get_rank0_type(array_type.get_internal_type()),
fortran_get_rank_of_type(array_type.get_internal_type()));
ERROR_CONDITION((size_of_array_descriptor % suitable_integer.get_size()) != 0,
"The size of the descriptor is not a multiple of the integer type", 0);
int num_items = size_of_array_descriptor / suitable_integer.get_size();
Counter& counter = CounterManager::get_counter("array-descriptor-copies");
std::stringstream ss;
ss << outline_info.get_field_name() << "_descriptor_" << (int)counter;
counter++;
TL::Symbol captured_array_descriptor = _sc.new_symbol(ss.str());
captured_array_descriptor.get_internal_symbol()->kind = SK_VARIABLE;
captured_array_descriptor.get_internal_symbol()->type_information =
get_array_type_bounds(suitable_integer.get_internal_type(),
const_value_to_nodecl(const_value_get_signed_int(1)),
const_value_to_nodecl(const_value_get_signed_int(num_items)),
_sc.get_decl_context());
TL::Symbol copy_descriptor_function = get_copy_descriptor_function(
captured_array_descriptor,
sym,
_sc);
Nodecl::NodeclBase prepare_capture;
{
Source src;
src
<< "{"
<< as_expression(copy_descriptor_function.make_nodecl( true))
<< "(" << as_expression(captured_array_descriptor.make_nodecl( true)) << ","
<<        as_expression(sym.make_nodecl( true)) << ");"
<< "}"
;
Source::source_language = SourceLanguage::C;
prepare_capture = src.parse_statement(_sc);
Source::source_language = SourceLanguage::Fortran;
}
if (sym.is_optional())
{
Source guard_src;
guard_src
<< "IF (PRESENT(" << sym.get_name() << ")) THEN\n"
<<    as_statement(prepare_capture)
<< "ELSE\n"
<<    as_symbol(captured_array_descriptor) << " = 0\n"
<< "END IF\n"
;
prepare_capture = guard_src.parse_statement(_sc);
}
this->add_capture(captured_array_descriptor);
OutlineDataItem &captured_array_descriptor_info = _outline_info.get_entity_for_symbol(captured_array_descriptor);
captured_array_descriptor_info.set_prepare_capture_code(prepare_capture);
captured_array_descriptor_info.set_in_outline_type(outline_info.get_in_outline_type());
captured_array_descriptor_info.set_is_copy_of_array_descriptor_allocatable(sym.is_allocatable());
return &captured_array_descriptor_info;
}
void OutlineInfoRegisterEntities::add_shared_opaque_and_captured_array_descriptor(Symbol sym)
{
ERROR_CONDITION(!IS_FORTRAN_LANGUAGE, "This function is only for Fortran", 0);
OutlineDataItem &outline_info = _outline_info.get_entity_for_symbol(sym);
outline_info.set_sharing(OutlineDataItem::SHARING_SHARED);
Type t = sym.get_type();
if (t.is_any_reference())
{
t = t.references_to();
}
TL::Type void_pointer = TL::Type::get_void_type().get_pointer_to();
outline_info.set_field_type(void_pointer);
TL::Type in_outline_type = t.get_lvalue_reference_to();
in_outline_type = add_extra_dimensions(sym, in_outline_type, &outline_info);
outline_info.set_in_outline_type(in_outline_type);
OutlineDataItem* captured_array_descriptor_info = this->capture_descriptor(outline_info, sym);
outline_info.set_copy_of_array_descriptor(captured_array_descriptor_info);
}
void OutlineInfoRegisterEntities::add_shared_alloca(Symbol sym)
{
ERROR_CONDITION(!IS_C_LANGUAGE && !IS_CXX_LANGUAGE, "This function is only for C/C++", 0);
OutlineDataItem &outline_info = _outline_info.get_entity_for_symbol(sym);
outline_info.set_sharing(OutlineDataItem::SHARING_SHARED_ALLOCA);
Type t = sym.get_type();
if (t.is_any_reference())
{
t = t.references_to();
}
if (t.is_array())
{
t = t.array_element().get_pointer_to();
}
outline_info.set_field_type(t.get_unqualified_type_but_keep_restrict());
TL::Type in_outline_type = t.get_unqualified_type_but_keep_restrict();
in_outline_type = add_extra_dimensions(sym, in_outline_type, &outline_info);
outline_info.set_in_outline_type(in_outline_type);
}
void OutlineInfoRegisterEntities::add_alloca(Symbol sym, TL::DataReference& data_ref)
{
ERROR_CONDITION(!IS_C_LANGUAGE && !IS_CXX_LANGUAGE, "This function is only for C/C++", 0);
OutlineDataItem &outline_info = _outline_info.get_entity_for_symbol(sym);
outline_info.set_sharing(OutlineDataItem::SHARING_ALLOCA);
Type t = data_ref.get_type();
if (t.is_any_reference())
{
t = t.references_to();
}
if (t.is_array())
{
t = t.array_element().get_pointer_to();
}
outline_info.set_field_type(t.get_unqualified_type_but_keep_restrict());
TL::Type in_outline_type = t.get_unqualified_type_but_keep_restrict();
in_outline_type = add_extra_dimensions(sym, in_outline_type, &outline_info);
outline_info.set_in_outline_type(in_outline_type);
}
void OutlineInfoRegisterEntities::add_capture_address(Symbol sym, TL::DataReference& data_ref)
{
internal_error("Not yet implemented", 0);
#if 0
ERROR_CONDITION(!IS_C_LANGUAGE && !IS_CXX_LANGUAGE, "This function is only for C/C++", 0);
OutlineDataItem &outline_info = _outline_info.get_entity_for_symbol(sym);
outline_info.set_sharing(OutlineDataItem::SHARING_CAPTURE_ADDRESS);
outline_info.set_base_address_expression(data_ref.get_address_of_symbol());
Type t = sym.get_type();
if (t.is_any_reference())
{
t = t.references_to();
}
if (t.is_array())
{
t = t.array_element().get_pointer_to();
}
outline_info.set_field_type(t.get_unqualified_type_but_keep_restrict());
TL::Type in_outline_type = t.get_unqualified_type_but_keep_restrict();
in_outline_type = add_extra_dimensions(sym, in_outline_type, &outline_info);
outline_info.set_in_outline_type(in_outline_type);
#endif
}
void OutlineInfoRegisterEntities::add_private_with_init(Symbol sym,
Nodecl::NodeclBase init)
{
OutlineDataItem &outline_info = _outline_info.get_entity_for_symbol(sym);
ERROR_CONDITION(
(outline_info.get_sharing() != OutlineDataItem::SHARING_UNDEFINED
&& outline_info.get_sharing() != OutlineDataItem::SHARING_PRIVATE),
"Overwriting data-sharing", 0);
if (outline_info.get_sharing() == OutlineDataItem::SHARING_UNDEFINED)
outline_info.set_sharing(OutlineDataItem::SHARING_PRIVATE);
TL::Type t = sym.get_type();
if (t.is_any_reference())
t = t.references_to();
TL::Type in_outline_type = add_extra_dimensions(sym, t, &outline_info);
outline_info.set_in_outline_type(in_outline_type);
outline_info.set_private_type(t);
outline_info.set_captured_value(init);
}
void OutlineInfoRegisterEntities::add_private(Symbol sym)
{
add_private_with_init(sym, Nodecl::NodeclBase::null());
}
void OutlineInfoRegisterEntities::add_shared_with_private_storage(Symbol sym, bool captured)
{
if (captured)
{
this->add_capture(sym);
}
else
{
this->add_private(sym);
}
OutlineDataItem &outline_info = _outline_info.get_entity_for_symbol(sym);
outline_info.set_is_lastprivate(true);
TL::Type t = sym.get_type();
if (t.is_any_reference())
t = t.references_to();
outline_info.set_private_type(t);
if (IS_C_LANGUAGE
|| IS_CXX_LANGUAGE)
{
TL::Symbol new_addr_symbol = _sc.new_symbol(sym.get_name() + "_addr");
new_addr_symbol.get_internal_symbol()->kind = SK_VARIABLE;
new_addr_symbol.get_internal_symbol()->type_information = t.get_pointer_to().get_internal_type();
Nodecl::Symbol sym_ref = Nodecl::Symbol::make(sym);
sym_ref.set_type(t.get_lvalue_reference_to());
Nodecl::NodeclBase value =
Nodecl::Reference::make(sym_ref, t.get_pointer_to());
this->add_capture_with_value(
new_addr_symbol,
value);
OutlineDataItem* lastprivate_shared = &_outline_info.get_entity_for_symbol(new_addr_symbol);
ERROR_CONDITION(lastprivate_shared == NULL, "This should not be NULL here", 0);
outline_info.set_lastprivate_shared(lastprivate_shared);
}
else if (IS_FORTRAN_LANGUAGE)
{
if ((sym.get_type().no_ref().is_fortran_array()
&& sym.get_type().no_ref().array_requires_descriptor())
|| (sym.get_type().no_ref().is_pointer()
&& sym.get_type().no_ref().points_to().is_fortran_array()))
{
OutlineDataItem* captured_array_descriptor_info = this->capture_descriptor(outline_info, sym);
TL::Type shared_type = sym.get_type().no_ref().get_lvalue_reference_to();
captured_array_descriptor_info->set_in_outline_type(shared_type);
outline_info.set_lastprivate_shared(captured_array_descriptor_info);
}
else
{
TL::Symbol new_addr_symbol = _sc.new_symbol(sym.get_name() + "_addr");
new_addr_symbol.get_internal_symbol()->kind = SK_VARIABLE;
new_addr_symbol.get_internal_symbol()->type_information =
TL::Type::get_void_type().get_pointer_to().get_internal_type();
Nodecl::Symbol sym_ref = Nodecl::Symbol::make(sym);
sym_ref.set_type(t.get_lvalue_reference_to());
Nodecl::NodeclBase value =
Nodecl::Reference::make(sym_ref, t.get_pointer_to());
this->add_capture_with_value(
new_addr_symbol,
value);
OutlineDataItem &new_outline_info = _outline_info.get_entity_for_symbol(new_addr_symbol);
TL::Type new_type = add_extra_dimensions(sym, sym.get_type());
new_outline_info.set_in_outline_type(new_type.no_ref().get_lvalue_reference_to());
OutlineDataItem* lastprivate_shared = &_outline_info.get_entity_for_symbol(new_addr_symbol);
outline_info.set_lastprivate_shared(lastprivate_shared);
}
}
}
TL::Type OutlineInfoRegisterEntities::add_extra_dimensions(TL::Symbol sym, TL::Type t)
{
return add_extra_dimensions(sym, t, NULL);
}
TL::Type OutlineInfoRegisterEntities::add_extra_dimensions(TL::Symbol sym, TL::Type t,
OutlineDataItem* outline_data_item)
{
Nodecl::NodeclBase conditional_bound;
if (sym.is_allocatable()
&& outline_data_item != NULL
&& ((outline_data_item->get_sharing() == OutlineDataItem::SHARING_PRIVATE)
|| (outline_data_item->get_sharing() == OutlineDataItem::SHARING_REDUCTION))
&& conditional_bound.is_null())
{
Counter& counter = CounterManager::get_counter("array-allocatable");
std::stringstream ss;
ss << "mcc_is_allocated_" << (int)counter++;
TL::Symbol is_allocated_sym = _sc.new_symbol(ss.str());
is_allocated_sym.get_internal_symbol()->kind = SK_VARIABLE;
is_allocated_sym.get_internal_symbol()->type_information = fortran_get_default_logical_type();
Nodecl::NodeclBase symbol_ref = Nodecl::Symbol::make(sym, make_locus("", 0, 0));
TL::Type sym_type = sym.get_type();
if (!sym_type.is_any_reference())
sym_type = sym_type.get_lvalue_reference_to();
symbol_ref.set_type(sym_type);
Source allocated_src;
allocated_src << "ALLOCATED(" << as_expression(symbol_ref) << ")";
Nodecl::NodeclBase allocated_tree = allocated_src.parse_expression(_sc);
this->add_capture_with_value(is_allocated_sym, allocated_tree);
conditional_bound = allocated_tree;
}
bool make_allocatable = sym.is_allocatable();
TL::Type res = add_extra_dimensions_rec(sym, t, outline_data_item, make_allocatable, conditional_bound);
if (make_allocatable
&& outline_data_item != NULL
&& outline_data_item->get_sharing() == OutlineDataItem::SHARING_CAPTURE)
{
outline_data_item->set_allocation_policy(
outline_data_item->get_allocation_policy() |
OutlineDataItem::ALLOCATION_POLICY_TASK_MUST_DEALLOCATE_ALLOCATABLE);
}
if (t.is_any_reference())
t = t.references_to();
if (t.is_array()
&& outline_data_item != NULL
&& outline_data_item->get_sharing() == OutlineDataItem::SHARING_CAPTURE)
{
if (!IS_FORTRAN_LANGUAGE 
&& t.array_is_vla())
{
outline_data_item->set_allocation_policy(
outline_data_item->get_allocation_policy() |
OutlineDataItem::ALLOCATION_POLICY_OVERALLOCATED);
outline_data_item->set_field_type(
TL::Type::get_void_type().get_pointer_to());
CXX_LANGUAGE()
{
res = TL::Type::get_void_type().get_pointer_to();
}
}
}
else
{
if (t.depends_on_nonconstant_values())
{
outline_data_item->set_field_type(
TL::Type::get_void_type().get_pointer_to());
CXX_LANGUAGE()
{
res = TL::Type::get_void_type().get_pointer_to();
}
}
}
return res;
}
TL::Type OutlineInfoRegisterEntities::add_extra_dimensions_rec(TL::Symbol sym, TL::Type t,
OutlineDataItem* outline_data_item,
bool &make_allocatable,
Nodecl::NodeclBase &conditional_bound)
{
if (t.is_lvalue_reference())
{
TL::Type res = add_extra_dimensions_rec(sym, t.references_to(), outline_data_item, make_allocatable, conditional_bound);
return res.get_lvalue_reference_to();
}
else if (t.is_pointer())
{
TL::Type res = add_extra_dimensions_rec(sym, t.points_to(), outline_data_item, make_allocatable, conditional_bound);
return res.get_pointer_to().get_as_qualified_as(t);
}
else if (t.is_function())
{
return t;
}
else if (t.is_class())
{
return t;
}
else if ((IS_C_LANGUAGE || IS_CXX_LANGUAGE)
&& t.is_array())
{
Nodecl::NodeclBase array_size = t.array_get_size();
if (array_size.is<Nodecl::Symbol>()
&& array_size.get_symbol().is_saved_expression())
{
OutlineDataItem& item = _outline_info.get_entity_for_symbol(array_size.get_symbol());
_outline_info.move_at_beginning(item);
}
t = add_extra_dimensions_rec(sym, t.array_element(), outline_data_item, make_allocatable, conditional_bound);
return t.get_array_to(array_size, _sc);
}
else if (IS_FORTRAN_LANGUAGE
&& t.is_fortran_array())
{
Nodecl::NodeclBase lower, upper;
t.array_get_bounds(lower, upper);
Nodecl::NodeclBase result_lower, result_upper;
if (t.array_requires_descriptor()
&& outline_data_item != NULL
&& (outline_data_item->get_sharing() == OutlineDataItem::SHARING_SHARED
|| outline_data_item->get_sharing() == OutlineDataItem::SHARING_REDUCTION))
return t;
if (sym.is_optional()
&& conditional_bound.is_null())
{
Source present_src;
present_src << "PRESENT(" << sym.get_name() << ")";
Nodecl::NodeclBase allocated_tree = present_src.parse_expression(_sc);
conditional_bound = allocated_tree;
}
if (lower.is_null())
{
if (t.array_requires_descriptor())
{
make_allocatable = !sym.get_type().no_ref().is_pointer();
if (!sym.get_type().no_ref().is_pointer() &&
outline_data_item->get_sharing() == OutlineDataItem::SHARING_PRIVATE)
{
Counter& counter = CounterManager::get_counter("array-lower-boundaries");
std::stringstream ss;
ss << "mcc_lower_bound_" << (int)counter++;
TL::Symbol bound_sym = _sc.new_symbol(ss.str());
bound_sym.get_internal_symbol()->kind = SK_VARIABLE;
bound_sym.get_internal_symbol()->type_information = get_signed_int_type();
int dim = fortran_get_rank_of_type(t.get_internal_type());
Nodecl::NodeclBase symbol_ref = Nodecl::Symbol::make(sym, make_locus("", 0, 0));
TL::Type sym_type = sym.get_type();
if (!sym_type.no_ref().is_pointer())
{
if (!sym_type.is_any_reference())
sym_type = sym_type.get_lvalue_reference_to();
symbol_ref.set_type(sym_type);
}
else
{
symbol_ref.set_type(sym_type);
symbol_ref = Nodecl::Dereference::make(
symbol_ref,
sym_type.no_ref().points_to().get_lvalue_reference_to());
}
Source lbound_src;
lbound_src << "LBOUND(" << as_expression(symbol_ref) << ", DIM=" << dim << ")";
Nodecl::NodeclBase lbound_tree = lbound_src.parse_expression(_sc);
this->add_capture_with_value(bound_sym, lbound_tree, conditional_bound);
result_lower = bound_sym.make_nodecl( true);
}
}
else
{
}
}
else if (lower.is<Nodecl::Symbol>()
&& lower.get_symbol().is_saved_expression())
{
result_lower = lower;
make_allocatable = true;
}
else
{
ERROR_CONDITION(!lower.is_constant(), "This should be constant", 0);
result_lower = lower;
}
if (upper.is_null())
{
if (t.array_requires_descriptor())
{
make_allocatable = !sym.get_type().no_ref().is_pointer();
if (!sym.get_type().no_ref().is_pointer() &&
outline_data_item->get_sharing() == OutlineDataItem::SHARING_PRIVATE)
{
Counter& counter = CounterManager::get_counter("array-upper-boundaries");
std::stringstream ss;
ss << "mcc_upper_bound_" << (int)counter++;
TL::Symbol bound_sym = _sc.new_symbol(ss.str());
bound_sym.get_internal_symbol()->kind = SK_VARIABLE;
bound_sym.get_internal_symbol()->type_information = get_signed_int_type();
int dim = fortran_get_rank_of_type(t.get_internal_type());
Nodecl::NodeclBase symbol_ref = Nodecl::Symbol::make(sym, make_locus("", 0, 0));
TL::Type sym_type = sym.get_type();
if (!sym_type.no_ref().is_pointer())
{
if (!sym_type.is_any_reference())
sym_type = sym_type.get_lvalue_reference_to();
symbol_ref.set_type(sym_type);
}
else
{
symbol_ref.set_type(sym_type);
symbol_ref = Nodecl::Dereference::make(
symbol_ref,
sym_type.no_ref().points_to().get_lvalue_reference_to());
}
Source ubound_src;
ubound_src << "UBOUND(" << as_expression(symbol_ref) << ", DIM=" << dim << ")";
Nodecl::NodeclBase ubound_tree = ubound_src.parse_expression(_sc);
this->add_capture_with_value(bound_sym, ubound_tree, conditional_bound);
result_upper = bound_sym.make_nodecl( true);
}
}
else
{
}
}
else if (upper.is<Nodecl::Symbol>()
&& upper.get_symbol().is_saved_expression())
{
result_upper = upper;
make_allocatable = true;
}
else
{
ERROR_CONDITION(!upper.is_constant(), "This should be constant", 0);
result_upper = upper;
}
TL::Type res = add_extra_dimensions_rec(sym, t.array_element(), outline_data_item, make_allocatable, conditional_bound);
if (upper.is_null()
&& !t.array_requires_descriptor())
{
make_allocatable = false;
}
if (make_allocatable)
{
res = res.get_array_to_with_descriptor(result_lower, result_upper, _sc);
}
else
{
res = res.get_array_to(result_lower, result_upper, _sc);
}
if (make_allocatable
&& outline_data_item != NULL)
{
if (outline_data_item->get_sharing() == OutlineDataItem::SHARING_CAPTURE)
{
outline_data_item->set_field_type(
::fortran_get_n_ranked_type_with_descriptor(
::fortran_get_rank0_type(t.get_internal_type()),
::fortran_get_rank_of_type(t.get_internal_type()), _sc.get_decl_context()));
}
}
return res;
}
return t;
}
void OutlineInfoRegisterEntities::add_dependence(
Nodecl::NodeclBase node, OutlineDataItem::DependencyDirectionality directionality)
{
TL::DataReference data_ref(node);
if (data_ref.is_valid())
{
TL::Symbol sym = data_ref.get_base_symbol();
OutlineDataItem &dep_item = _outline_info.get_entity_for_symbol(sym);
dep_item.get_dependences().append(OutlineDataItem::DependencyItem(data_ref, directionality));
}
else
{
internal_error("%s: data reference '%s' must be valid at this point!\n",
node.get_locus_str().c_str(),
Codegen::get_current().codegen_to_str(node, node.retrieve_context()).c_str()
);
}
}
void OutlineInfoRegisterEntities::add_dependences(Nodecl::List list, OutlineDataItem::DependencyDirectionality directionality)
{
for (Nodecl::List::iterator it = list.begin();
it != list.end();
it++)
{
add_dependence(*it, directionality);
}
}
void OutlineInfoRegisterEntities::add_copies(Nodecl::List list, OutlineDataItem::CopyDirectionality copy_directionality)
{
for (Nodecl::List::iterator it = list.begin();
it != list.end();
it++)
{
TL::DataReference data_ref(*it);
if (data_ref.is_valid())
{
TL::Symbol sym = data_ref.get_base_symbol();
OutlineDataItem &outline_info = _outline_info.get_entity_for_symbol(sym);
outline_info.get_copies().append(OutlineDataItem::CopyItem(data_ref, copy_directionality));
}
else
{
internal_error("%s: data reference '%s' must be valid at this point!\n",
it->get_locus_str().c_str(),
Codegen::get_current().codegen_to_str(*it, it->retrieve_context()).c_str()
);
}
}
}
void OutlineInfoRegisterEntities::add_capture(Symbol sym)
{
add_capture_with_value(sym, Nodecl::NodeclBase::null(), Nodecl::NodeclBase::null());
}
void OutlineInfoRegisterEntities::add_capture_with_value(Symbol sym, Nodecl::NodeclBase expr)
{
add_capture_with_value(sym, expr, Nodecl::NodeclBase::null());
}
void OutlineInfoRegisterEntities::add_capture_with_value(Symbol sym, Nodecl::NodeclBase value, Nodecl::NodeclBase condition)
{
OutlineDataItem &outline_info = _outline_info.get_entity_for_symbol(sym);
ERROR_CONDITION(
(outline_info.get_sharing() != OutlineDataItem::SHARING_UNDEFINED
&& outline_info.get_sharing() != OutlineDataItem::SHARING_CAPTURE),
"Overwriting data-sharing", 0);
if (outline_info.get_sharing() == OutlineDataItem::SHARING_UNDEFINED)
outline_info.set_sharing(OutlineDataItem::SHARING_CAPTURE);
Type t = sym.get_type();
if (t.is_any_reference())
{
t = t.references_to();
}
outline_info.set_field_type(t);
if (t.is_function())
{
TL::Type pointer_type;
if (IS_FORTRAN_LANGUAGE)
{
pointer_type = TL::Type::get_void_type().get_pointer_to();
if (value.is_null())
{
value = Nodecl::Reference::make(
sym.make_nodecl( true),
pointer_type);
}
}
else
{
pointer_type = t.get_pointer_to();
}
outline_info.set_field_type(pointer_type);
}
TL::Type in_outline_type;
if (IS_FORTRAN_LANGUAGE
|| _outline_info.firstprivates_always_by_reference())
{
in_outline_type = t.get_lvalue_reference_to();
}
else
{
if (t.is_integral_type()
|| t.is_floating_type()
|| t.is_pointer())
{
in_outline_type = t;
}
else if (t.is_array()
&& !t.depends_on_nonconstant_values())
{
in_outline_type = t.array_element().get_pointer_to().get_restrict_type();
}
else
{
in_outline_type = t.get_lvalue_reference_to();
}
}
in_outline_type = add_extra_dimensions(sym, in_outline_type, &outline_info);
if ((outline_info.get_allocation_policy() & OutlineDataItem::ALLOCATION_POLICY_TASK_MUST_DEALLOCATE_ALLOCATABLE)
== OutlineDataItem::ALLOCATION_POLICY_TASK_MUST_DEALLOCATE_ALLOCATABLE)
{
outline_info.set_in_outline_type(outline_info.get_field_type().get_lvalue_reference_to());
}
else
{
outline_info.set_in_outline_type(in_outline_type);
}
if (IS_CXX_LANGUAGE
&& (t.is_dependent()
|| !t.is_pod()))
{
outline_info.set_allocation_policy(
outline_info.get_allocation_policy() |
OutlineDataItem::ALLOCATION_POLICY_TASK_MUST_DESTROY);
}
ERROR_CONDITION(!outline_info.get_captured_value().is_null()
&& !value.is_null(), "Overwriting captured value", 0);
if (!value.is_null())
outline_info.set_captured_value(value);
ERROR_CONDITION(!outline_info.get_conditional_capture_value().is_null()
&& !condition.is_null(), "Overwriting conditional captured value", 0);
if (!condition.is_null())
outline_info.set_conditional_capture_value(condition);
}
void OutlineInfoRegisterEntities::add_reduction(TL::Symbol symbol,
TL::Type reduction_type,
OpenMP::Reduction* reduction,
OutlineDataItem::Sharing kind)
{
OutlineDataItem &outline_info = _outline_info.get_entity_for_symbol(symbol);
outline_info.set_sharing(kind);
outline_info.set_reduction_info(reduction, reduction_type);
TL::Type t = symbol.get_type();
if (t.is_any_reference())
{
t = t.references_to();
}
if (IS_FORTRAN_LANGUAGE)
{
outline_info.set_field_type(TL::Type::get_void_type().get_pointer_to());
}
else
{
outline_info.set_field_type(t.get_pointer_to());
}
TL::Type in_outline_type = t.get_lvalue_reference_to();
in_outline_type = add_extra_dimensions(symbol, in_outline_type, &outline_info);
outline_info.set_in_outline_type(in_outline_type);
TL::Type private_type;
if (IS_FORTRAN_LANGUAGE)
{
if ((symbol.get_type().no_ref().is_fortran_array()
&& symbol.get_type().no_ref().array_requires_descriptor())
|| (symbol.get_type().no_ref().is_pointer()
&& symbol.get_type().no_ref().points_to().is_fortran_array()))
{
OutlineDataItem* captured_array_descriptor_info =
this->capture_descriptor(outline_info, symbol);
outline_info.set_copy_of_array_descriptor(captured_array_descriptor_info);
}
private_type = in_outline_type.no_ref();
}
else
{
private_type = reduction_type;
}
outline_info.set_private_type(private_type);
}
void OutlineInfoRegisterEntities::add_copy_of_outline_data_item(const OutlineDataItem& data_item)
{
_outline_info.add_copy_of_outline_data_item(data_item);
}
class OutlineInfoSetupVisitor : public Nodecl::ExhaustiveVisitor<void>, public OutlineInfoRegisterEntities
{
private:
OutlineInfo& _outline_info;
bool _is_task_construct;
public:
OutlineInfoSetupVisitor(OutlineInfo& outline_info, TL::Scope sc, bool is_task_construct)
: OutlineInfoRegisterEntities(outline_info, sc),
_outline_info(outline_info), _is_task_construct(is_task_construct)
{
}
void visit(const Nodecl::OpenMP::Auto& shared)
{
Nodecl::List l = shared.get_symbols().as<Nodecl::List>();
for (Nodecl::List::iterator it = l.begin();
it != l.end();
it++)
{
TL::Symbol sym = it->as<Nodecl::Symbol>().get_symbol();
error_printf_at(it->get_locus(),
"entity '%s' with unresolved 'auto' data sharing\n",
sym.get_name().c_str());
}
if (!l.empty())
{
fatal_printf_at(shared.get_locus(), "unresolved auto data sharings\n");
}
}
void visit(const Nodecl::OpenMP::Shared& shared)
{
Nodecl::List l = shared.get_symbols().as<Nodecl::List>();
for (Nodecl::List::iterator it = l.begin();
it != l.end();
it++)
{
TL::Symbol sym = it->as<Nodecl::Symbol>().get_symbol();
if (IS_FORTRAN_LANGUAGE)
{
if ((sym.get_type().no_ref().is_fortran_array()
&& sym.get_type().no_ref().array_requires_descriptor())
|| (sym.get_type().no_ref().is_pointer()
&& sym.get_type().no_ref().points_to().is_fortran_array()))
{
add_shared_opaque_and_captured_array_descriptor(sym);
}
else
{
add_shared_opaque(sym);
}
}
else
{
add_shared(sym);
}
}
}
void visit(const Nodecl::OmpSs::Alloca& alloca)
{
Nodecl::List l = alloca.get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = l.begin();
it != l.end();
it++)
{
TL::DataReference data_ref(*it);
ERROR_CONDITION(!data_ref.is_valid(), "%s: data reference '%s' must be valid at this point!\n",
it->get_locus_str().c_str(),
Codegen::get_current().codegen_to_str(*it, it->retrieve_context()).c_str());
TL::Symbol sym = data_ref.get_base_symbol();
add_alloca(sym, data_ref);
}
}
void visit(const Nodecl::OmpSs::SharedAndAlloca& alloca)
{
Nodecl::List l = alloca.get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = l.begin();
it != l.end();
it++)
{
TL::DataReference data_ref(*it);
ERROR_CONDITION(!data_ref.is_valid(), "%s: data reference '%s' must be valid at this point!\n",
it->get_locus_str().c_str(),
Codegen::get_current().codegen_to_str(*it, it->retrieve_context()).c_str());
TL::Symbol sym = data_ref.get_base_symbol();
add_shared_alloca(sym);
}
}
void visit(const Nodecl::OmpSs::DepWeakIn& dep_weak)
{
error_printf_at(dep_weak.get_locus(),
"weak dependences are not supported in Nanos++\n");
}
void visit(const Nodecl::OmpSs::DepWeakOut& dep_weak)
{
error_printf_at(dep_weak.get_locus(),
"weak dependences are not supported in Nanos++\n");
}
void visit(const Nodecl::OmpSs::DepWeakInout& dep_weak)
{
error_printf_at(dep_weak.get_locus(),
"weak dependences are not supported in Nanos++\n");
}
void visit(const Nodecl::OmpSs::DepWeakCommutative& dep_weak)
{
error_printf_at(dep_weak.get_locus(),
"weak dependences are not supported in Nanos++\n");
}
void visit(const Nodecl::OpenMP::DepIn& dep_in)
{
add_dependences(dep_in.get_exprs().as<Nodecl::List>(), OutlineDataItem::DEP_IN);
}
void visit(const Nodecl::OmpSs::DepInPrivate& dep_in_private)
{
add_dependences(dep_in_private.get_exprs().as<Nodecl::List>(), OutlineDataItem::DEP_IN_PRIVATE);
}
void visit(const Nodecl::OpenMP::DepOut& dep_out)
{
add_dependences(dep_out.get_exprs().as<Nodecl::List>(), OutlineDataItem::DEP_OUT);
}
void visit(const Nodecl::OpenMP::DepInout& dep_inout)
{
add_dependences(dep_inout.get_exprs().as<Nodecl::List>(), OutlineDataItem::DEP_INOUT);
}
void visit(const Nodecl::OmpSs::DepConcurrent& concurrent)
{
add_dependences(concurrent.get_exprs().as<Nodecl::List>(), OutlineDataItem::DEP_CONCURRENT);
}
void visit(const Nodecl::OmpSs::DepCommutative& commutative)
{
add_dependences(commutative.get_exprs().as<Nodecl::List>(), OutlineDataItem::DEP_COMMUTATIVE);
}
void visit(const Nodecl::OmpSs::DepReduction& dep_reduction)
{
add_dependences(dep_reduction.get_exprs().as<Nodecl::List>(), OutlineDataItem::DEP_CONCURRENT);
}
void visit(const Nodecl::OmpSs::CopyIn& copy_in)
{
add_copies(copy_in.get_input_copies().as<Nodecl::List>(), OutlineDataItem::COPY_IN);
}
void visit(const Nodecl::OmpSs::CopyOut& copy_out)
{
add_copies(copy_out.get_output_copies().as<Nodecl::List>(), OutlineDataItem::COPY_OUT);
}
void visit(const Nodecl::OmpSs::CopyInout& copy_inout)
{
add_copies(copy_inout.get_inout_copies().as<Nodecl::List>(), OutlineDataItem::COPY_INOUT);
}
void visit(const Nodecl::OmpSs::Implements& implements)
{
_outline_info.handle_implements_clause(
implements.get_function_name().as<Nodecl::Symbol>().get_symbol(),
implements.get_device().as<Nodecl::Text>().get_text());
}
void visit(const Nodecl::OmpSs::NDRange& ndrange)
{
_outline_info.set_ndrange(_outline_info.get_funct_symbol(),
ndrange.get_ndrange_expressions().as<Nodecl::List>().to_object_list());
}
void visit(const Nodecl::OmpSs::ShMem& shmem)
{
_outline_info.set_shmem(_outline_info.get_funct_symbol(),
shmem.get_shmem_expressions().as<Nodecl::List>().to_object_list());
}
void visit(const Nodecl::OmpSs::Onto& onto)
{
_outline_info.set_onto(_outline_info.get_funct_symbol(),
onto.get_onto_expressions().as<Nodecl::List>().to_object_list());
}
void visit(const Nodecl::OmpSs::File& file)
{
_outline_info.set_file(_outline_info.get_funct_symbol(),
file.get_filename().get_text());
}
void visit(const Nodecl::OmpSs::Name& name)
{
_outline_info.set_name(_outline_info.get_funct_symbol(),
name.get_name().get_text());
}
void visit(const Nodecl::OpenMP::Firstprivate& firstprivate)
{
Nodecl::List l = firstprivate.get_symbols().as<Nodecl::List>();
for (Nodecl::List::iterator it = l.begin();
it != l.end();
it++)
{
TL::Symbol sym = it->as<Nodecl::Symbol>().get_symbol();
add_capture(sym);
}
}
void visit(const Nodecl::OpenMP::Lastprivate& lastprivate)
{
Nodecl::List l = lastprivate.get_symbols().as<Nodecl::List>();
for (Nodecl::List::iterator it = l.begin();
it != l.end();
it++)
{
TL::Symbol sym = it->as<Nodecl::Symbol>().get_symbol();
add_shared_with_private_storage(sym,  false);
}
}
void visit(const Nodecl::OpenMP::FirstLastprivate& firstlastprivate)
{
Nodecl::List l = firstlastprivate.get_symbols().as<Nodecl::List>();
for (Nodecl::List::iterator it = l.begin();
it != l.end();
it++)
{
TL::Symbol sym = it->as<Nodecl::Symbol>().get_symbol();
add_shared_with_private_storage(sym,  true);
}
}
void visit(const Nodecl::OpenMP::Private& private_)
{
Nodecl::List l = private_.get_symbols().as<Nodecl::List>();
for (Nodecl::List::iterator it = l.begin();
it != l.end();
it++)
{
TL::Symbol sym = it->as<Nodecl::Symbol>().get_symbol();
add_private(sym);
}
}
void visit(const Nodecl::OpenMP::PrivateInit& private_init)
{
TL::Symbol sym = private_init.get_symbol();
if (private_init.get_value().is_null())
{
add_private_with_init(sym, sym.get_value());
}
else
{
add_private_with_init(sym, private_init.get_value());
}
}
void visit(const Nodecl::OpenMP::Threadprivate& threadprivate)
{
}
void visit(const Nodecl::OpenMP::Reduction& reduction)
{
Nodecl::List reductions = reduction.get_reductions().as<Nodecl::List>();
for (Nodecl::List::iterator it = reductions.begin();
it != reductions.end();
it++)
{
Nodecl::OpenMP::ReductionItem red_item = it->as<Nodecl::OpenMP::ReductionItem>();
TL::Symbol reductor_sym = red_item.get_reductor().get_symbol();
TL::Symbol reduction_symbol = red_item.get_reduced_symbol().get_symbol();
TL::Type reduction_type = red_item.get_reduction_type().get_type();
OpenMP::Reduction* red = OpenMP::Reduction::get_reduction_info_from_symbol(reductor_sym);
ERROR_CONDITION(red == NULL, "Invalid value for red_item", 0);
if (!_is_task_construct)
{
add_reduction(reduction_symbol, reduction_type, red, OutlineDataItem::SHARING_REDUCTION);
}
else
{
OutlineDataItem &outline_info = _outline_info.get_entity_for_symbol(reduction_symbol);
outline_info.set_reduction_info(red, reduction_type);
}
}
}
void visit(const Nodecl::OmpSs::Target& target)
{
Nodecl::List devices = target.get_devices().as<Nodecl::List>();
for (Nodecl::List::iterator it = devices.begin();
it != devices.end();
it++)
{
std::string current_device = it->as<Nodecl::Text>().get_text();
_outline_info.add_device_name(_outline_info.get_funct_symbol(), current_device);
}
walk(target.get_items());
}
void purge_saved_expressions()
{
this->OutlineInfoRegisterEntities::purge_saved_expressions();
}
};
OutlineInfo::OutlineInfo(Nanox::Lowering& lowering)
: _lowering(lowering)
{}
OutlineInfo::OutlineInfo(
Nanox::Lowering& lowering,
Nodecl::NodeclBase environment,
TL::Symbol funct_symbol,
bool is_task_construct,
std::shared_ptr<TL::OmpSs::FunctionTaskSet> function_task_set)
: _lowering(lowering), _funct_symbol(funct_symbol), _function_task_set(function_task_set)
{
TL::Scope sc(CURRENT_COMPILED_FILE->global_decl_context);
if (!environment.is_null())
{
sc = environment.retrieve_context();
}
if (_funct_symbol.is_valid())
{
_implementation_table[_funct_symbol]._outline_name = get_outline_name(_funct_symbol);
}
OutlineInfoSetupVisitor setup_visitor(*this, sc, is_task_construct);
setup_visitor.walk(environment);
setup_visitor.purge_saved_expressions();
OutlineInfoRegisterEntities &ol_register = setup_visitor;
finish_multicopies(ol_register, sc);
reset_array_counters();
}
OutlineInfo::~OutlineInfo()
{
for (ObjectList<OutlineDataItem*>::iterator it = _data_env_items.begin();
it != _data_env_items.end();
it++)
{
delete *it;
}
}
void OutlineInfo::finish_multicopies(OutlineInfoRegisterEntities& ol_register, TL::Scope sc)
{
int num_dynamic_copies_without_static_copies = 0;
TL::ObjectList<OutlineDataItem*> data_items = this->get_data_items();
for (TL::ObjectList<OutlineDataItem*>::iterator it = data_items.begin();
it != data_items.end();
it++)
{
bool has_static = false;
bool has_dynamic = false;
const TL::ObjectList<OutlineDataItem::CopyItem> &items = (*it)->get_copies();
for (TL::ObjectList<OutlineDataItem::CopyItem>::const_iterator it_items = items.begin();
it_items != items.end() && !has_static;
it_items++)
{
DataReference data_ref(it_items->expression);
if (data_ref.is_multireference())
{
has_dynamic = true;
}
else
{
has_static = true;
}
}
if (has_dynamic && !has_static)
{
num_dynamic_copies_without_static_copies++;
}
}
if (num_dynamic_copies_without_static_copies <= 1)
return;
Counter& counter_multicopies_array = CounterManager::get_counter("multicopies-array-size");
std::stringstream ss;
ss << "multicopies_index_" << (int)counter_multicopies_array << "__";
counter_multicopies_array++;
TL::Symbol multicopies_index = sc.new_symbol(ss.str());
multicopies_index.get_internal_symbol()->kind = SK_VARIABLE;
multicopies_index.get_internal_symbol()->type_information =
TL::Type::get_int_type().get_array_to(
const_value_to_nodecl(
const_value_get_signed_int(num_dynamic_copies_without_static_copies - 1)
),
TL::Scope::get_global_scope()).get_internal_type();
symbol_entity_specs_set_is_user_declared(multicopies_index.get_internal_symbol(), 1);
ol_register.add_capture(multicopies_index);
set_multicopies_index_symbol(multicopies_index);
}
TL::Symbol OutlineInfo::get_multicopies_index_symbol() const
{
return _multicopies_index_symbol;
}
void OutlineInfo::set_multicopies_index_symbol(TL::Symbol sym)
{
_multicopies_index_symbol = sym;
}
void OutlineInfoRegisterEntities::purge_saved_expressions()
{
ObjectList<OutlineDataItem*> fields = _outline_info.get_data_items();
for (ObjectList<OutlineDataItem*>::iterator it = fields.begin();
it != fields.end();
it++)
{
OutlineDataItem &data_item = *(*it);
if (data_item.get_sharing() == OutlineDataItem::SHARING_UNDEFINED)
{
TL::Symbol sym = data_item.get_symbol();
ERROR_CONDITION(!sym.is_saved_expression(), "Symbol '%s' is missing a data sharing", sym.get_name().c_str());
_outline_info.remove_entity(data_item);
}
}
}
void OutlineInfo::reset_array_counters()
{
Counter& counter_allocatable = CounterManager::get_counter("array-allocatable");
counter_allocatable = 0;
Counter& counter_upper_boundaries = CounterManager::get_counter("array-upper-boundaries");
counter_upper_boundaries = 0;
Counter& counter_lower_boundaries = CounterManager::get_counter("array-lower-boundaries");
counter_lower_boundaries = 0;
}
ObjectList<OutlineDataItem*> OutlineInfo::get_data_items()
{
return _data_env_items;
}
TL::Symbol OutlineInfo::get_funct_symbol() const
{
return _funct_symbol;
}
OutlineDataItem& OutlineInfo::prepend_field(TL::Symbol sym)
{
std::string field_name = get_field_name(sym.get_name());
OutlineDataItem* env_item = new OutlineDataItem(sym, field_name);
_data_env_items.std::vector<OutlineDataItem*>::insert(_data_env_items.begin(), env_item);
return *(_data_env_items.front());
}
OutlineDataItem& OutlineInfo::append_field(TL::Symbol sym)
{
std::string field_name = get_field_name(sym.get_name());
OutlineDataItem* env_item = new OutlineDataItem(sym, field_name);
_data_env_items.append(env_item);
return *(_data_env_items.back());
}
void OutlineInfo::move_at_beginning(OutlineDataItem& item)
{
TL::ObjectList<OutlineDataItem*> new_list;
new_list.append(&item);
for (TL::ObjectList<OutlineDataItem*>::iterator it = _data_env_items.begin();
it != _data_env_items.end();
it++)
{
if (*it != &item)
{
new_list.append(*it);
}
}
std::swap(_data_env_items, new_list);
}
void OutlineInfo::add_device_name(TL::Symbol function_symbol, const std::string& device_name)
{
implementation_must_be_present(function_symbol);
_implementation_table[function_symbol]._device_names.append(device_name);
}
void OutlineInfo::set_file(TL::Symbol function_symbol, const std::string& file)
{
implementation_must_be_present(function_symbol);
_implementation_table[function_symbol]._file = file;
}
void OutlineInfo::set_name(TL::Symbol function_symbol,const std::string& name)
{
implementation_must_be_present(function_symbol);
_implementation_table[function_symbol]._name = name;
}
void OutlineInfo::set_ndrange(TL::Symbol function_symbol, const ObjectList<Nodecl::NodeclBase>& ndrange_exprs)
{
implementation_must_be_present(function_symbol);
_implementation_table[function_symbol]._ndrange_exprs = ndrange_exprs;
}
void OutlineInfo::set_shmem(TL::Symbol function_symbol, const ObjectList<Nodecl::NodeclBase>& shmem_exprs)
{
implementation_must_be_present(function_symbol);
_implementation_table[function_symbol]._shmem_exprs = shmem_exprs;
}
void OutlineInfo::set_onto(TL::Symbol function_symbol, const ObjectList<Nodecl::NodeclBase>& onto_exprs)
{
implementation_must_be_present(function_symbol);
_implementation_table[function_symbol]._onto_exprs = onto_exprs;
}
void OutlineInfo::set_param_arg_map(TL::Symbol function_symbol, Nodecl::Utils::SimpleSymbolMap param_arg_map)
{
implementation_must_be_present(function_symbol);
_implementation_table[function_symbol]._param_to_args = param_arg_map;
}
void OutlineInfo::add_new_implementation(
TL::Symbol function_symbol,
const std::string& device_name,
const std::string& file_args,
const std::string& name_args,
const TL::ObjectList<Nodecl::NodeclBase>& ndrange_args,
const TL::ObjectList<Nodecl::NodeclBase>& shmem_args,
const TL::ObjectList<Nodecl::NodeclBase>& onto_args)
{
if(_implementation_table.count(function_symbol) == 0)
{
_implementation_table[function_symbol]._outline_name = get_outline_name(function_symbol);
set_file(function_symbol, file_args);
set_name(function_symbol, name_args);
set_ndrange(function_symbol, ndrange_args);
set_shmem(function_symbol, shmem_args);
set_onto(function_symbol, onto_args);
}
_implementation_table[function_symbol]._device_names.append(device_name);
}
const TargetInformation& OutlineInfo::get_target_information(TL::Symbol function_symbol) const
{
implementation_must_be_present(function_symbol);
return _implementation_table.find(function_symbol)->second;
}
void OutlineInfo::handle_implements_clause(TL::Symbol implementor_symbol, std::string device_name)
{
ERROR_CONDITION(_function_task_set == NULL, "Unreachable code", 0);
TL::OmpSs::TargetInfo& target_info =
_function_task_set->get_function_task(implementor_symbol).get_target_info();
add_new_implementation(
implementor_symbol,
device_name,
target_info.get_file(),
target_info.get_name(),
target_info.get_ndrange(),
target_info.get_shmem(),
target_info.get_onto());
}
std::string OutlineInfo::get_outline_name(TL::Symbol function_symbol)
{
std::string outline_name;
std::string orig_function_name = function_symbol.get_name();
std::string prefix = "", suffix = "";
Counter& task_counter = CounterManager::get_counter("nanos++-outline");
if (IS_FORTRAN_LANGUAGE)
{
std::stringstream ss;
ss << "ol_";
if (function_symbol.is_nested_function())
{
TL::Symbol enclosing_function_symbol = function_symbol.get_scope().get_related_symbol();
ss << enclosing_function_symbol.get_name() << "_";
if (enclosing_function_symbol.is_in_module())
{
ss << enclosing_function_symbol.in_module().get_name() << "_";
}
}
else if (function_symbol.is_in_module())
{
ss << function_symbol.in_module().get_name() << "_";
}
ss << function_symbol.get_filename() << "_" << (int)task_counter;
unsigned int hash_int = simple_hash_str(ss.str().c_str());
static char base_syms[] = "0123456789abcdefghijklmnopqrstuvwxyz_";
const unsigned int nbase =
sizeof(base_syms) / sizeof(base_syms[0]) - 1; 
if (hash_int == 0)
{
prefix = "0";
}
else
{
while (hash_int > 0)
{
prefix += base_syms[(hash_int % nbase)];
hash_int /= nbase;
}
}
outline_name = prefix + function_symbol.get_name();
}
else if (IS_C_LANGUAGE || IS_CXX_LANGUAGE)
{
prefix = "ol_";
std::stringstream ss;
ss << "_" << (int)task_counter;
suffix = ss.str();
if (IS_C_LANGUAGE)
{
outline_name = prefix + function_symbol.get_name() + suffix;
}
else
{
std::string unmangle_name = unmangle_symbol_name(function_symbol.get_internal_symbol());
if (unmangle_name.find("operator ") != std::string::npos)
unmangle_name = "operator";
outline_name = prefix + unmangle_name + suffix;
}
}
task_counter++;
return outline_name;
}
const OutlineInfo::implementation_table_t& OutlineInfo::get_implementation_table() const
{
return _implementation_table;
}
void OutlineInfo::add_copy_of_outline_data_item(const OutlineDataItem& ol)
{
_data_env_items.append(new OutlineDataItem(ol));
}
namespace
{
bool is_not_private(OutlineDataItem* it)
{
return it->get_sharing() != OutlineDataItem::SHARING_PRIVATE;
}
}
ObjectList<OutlineDataItem*> OutlineInfo::get_fields() const
{
return _data_env_items.filter(is_not_private);
}
bool OutlineInfo::only_has_smp_or_mpi_implementations() const
{
for (implementation_table_t::const_iterator it = _implementation_table.begin();
it != _implementation_table.end();
++it)
{
const TargetInformation& target_info = it->second;
const ObjectList<std::string>& devices = target_info.get_device_names();
for (ObjectList<std::string>::const_iterator it2 = devices.begin();
it2 != devices.end();
++it2)
{
if (*it2 != "smp" && *it2 != "mpi")
return false;
}
}
return true;
}
bool OutlineInfo::firstprivates_always_by_reference() const
{
return _lowering.firstprivates_always_by_reference();
}
void OutlineInfo::implementation_must_be_present(TL::Symbol function_symbol) const
{
ERROR_CONDITION(_implementation_table.count(function_symbol) == 0,
"Function symbol '%s' not found in outline info implementation table",
function_symbol.get_name().c_str());
}
} }
